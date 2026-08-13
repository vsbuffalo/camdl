# Synthetic observations were quantized to the trajectory recording grid for four months

- Date: 2026-08-12
- Issue: gh#589
- Classification: **code-vs-code** — the implementation contradicts a separation
  the codebase already encodes in its own types (gh#233's `OutputTimes` /
  `EffectTimes` / `ObsTimes`). The fix is code plus a test pinning the
  agreement.
- Introduced: `b7fe919e` (2026-04-15),
  `feat(cli): synthetic observations for camdl simulate`
- Status: guard shipped; correct fix is an arc (see §6)

## Summary in one paragraph

`camdl simulate --obs` derives observation values from the **recorded
trajectory** rather than from integrator state at observation times. When an
observation time is absent from the recorded snapshots — a coarser or merely
misaligned recording cadence, **or the default schedule against a sub-unit emit
schedule, or `fit predict` projecting at irregular observed-data times** — that
observation reads the preceding snapshot instead. For a flow (`incidence`), the
accumulated interval collapses onto the snapshot boundary: six zeros and a lump.
For a stock (`prevalence`), the series becomes a step function. Nothing warns.
The emitted file still carries daily timestamps and a daily header, so it is
labelled as daily surveillance data and is weekly lumps. Because `--obs` exists
to generate synthetic data that is then **fitted**, the corrupted series becomes
the input to inference.

## 1. Reproduction

Model with a daily emit schedule (`q.camdl`, abridged):

```camdl
transitions {
  infection : S --> I @ beta * S * (I / (S + I + R))
  recovery  : I --> R @ gamma * I
}
init { S = 990  I = 10  R = 0 }

observations {
  daily_cases {
    columns       { time : time, daily_cases : count }
    projected     = incidence(infection)
    emit_schedule = every 1 'days
    daily_cases   ~ poisson(rate = projected)
  }
}
simulate { from = 0 'days  to = 21 'days }
```

Same model, same seed, same `emit_schedule`. The only difference is the
recording cadence:

```
$ camdl simulate q.camdl --params p.toml --seed 3 --obs obsA.tsv --output-dir rA
$ camdl simulate q.camdl --params p.toml --seed 3 --output-every 7 --obs obsB.tsv --output-dir rB
```

```
time    default   --output-every 7
0       0         0
1       4         0
2       3         0
3       15        0
4       7         0
5       8         0
6       6         0
7       8         49
8       13        0
```

No warning, no note, on either run. `obsB.tsv`'s header is `time daily_cases`
and it has a row per day.

## 2. Mechanism

`project_all_obs_times` (`rust/crates/cli/src/main.rs:2310`) contains the
`incidence_over` closure (`:2320`), which:

1. walks `traj.snapshots` accumulating a running cumulative flow, producing
   `cum_at_snap: Vec<(t, cumulative)>` — **one entry per recorded snapshot**;
2. for each observation time, advances to the last snapshot at or before it
   (`:2334-2345`) and takes that cumulative value;
3. differences consecutive observation times.

With snapshots at t ∈ {0, 7, 14, …} and observations at t ∈ {1, 2, …}:

```
obs t = 1..6  → last snapshot ≤ t is t = 0  → cum = C(0)
obs t = 7     → last snapshot ≤ 7 is t = 7  → cum = C(7)

differenced   → 0, 0, 0, 0, 0, 0, C(7) − C(0)
```

Stocks take the same path through `snap_at(traj, obs_t)` (`main.rs:2150`), which
returns the last snapshot at or before the observation time — hence the step
function.

The simulation _computed_ the daily flows. They were discarded by a recording
flag and the remainder reported as daily data.

## 3. How it was detected

Not by a test. It surfaced in the run-spec audit
(`docs/dev/reviews/2026-08-11-run-spec-audit.md`, finding 22) as a single-agent,
uncorroborated observation while rebuilding the spec against the implementation.
It was ranked from its section heading, challenged ("are we sure this is
wrong?"), then traced in code and reproduced.

Worth recording that the challenge was the load-bearing step. The finding was
initially accepted on a heading; the mechanism, the blast radius, and the
severity framing all came out of being asked to substantiate it.

## 4. Root cause

**The emission path is seated at the wrong seam.** `--obs` was added to the sink
(`b7fe919e`), and a sink receives a `CellResult` whose `traj` is already
complete. Sampling from integrator state at observation times would have meant
hooking into the run itself; reading the finished trajectory was what was
available where the code was written. So observation emission was implemented
**downstream of recording**, and silently inherited recording's cadence.

`incidence_over` did not introduce this. It arrived later in `298494bf`
(`fix(dsl): incidence() over a stratified transition family sums strata`), which
refactored the existing snapshot-walking logic to handle strata families. It
inherited the defect.

The uncomfortable part is that the codebase already knows better. gh#233
introduced `OutputTimes`, `EffectTimes` and `ObsTimes` as **distinct types**
precisely because they mean _record_, _fire_, and _score-and-reset_ — three axes
that must not be conflated, with the newtypes justified on the grounds that a
swap between them type-checks and is silently wrong. This path collapses
_record_ into _score_, in a module that never touches those types.

## 5. Impact

**Affected — every synthetic and predictive emission path**, not only
`simulate --obs`. All five callers of `project_all_obs_times` share the defect.
Note `fit predict` is not driven by `emit_schedule` at all: its projection times
are the **loaded observed-data times** (`predict.rs:1142-1151`), which for real
surveillance data are irregular and need not land on any regular grid. That
makes it the highest-exposure path, and it is reachable with no `output` block
and no unusual flag:

| caller                 | path                               |
| ---------------------- | ---------------------------------- |
| `main.rs:2143`         | `simulate --obs` / `--obs-dir`     |
| `main.rs:1613`         | `observations.<stream>` quantities |
| `batch.rs:1698`        | `batch run` obs emission           |
| `fit/synthetic.rs:146` | synthetic data generation          |
| `fit/predict.rs:754`   | **posterior predictive `y_rep`**   |

The posterior-predictive case deserves separate mention: stair-stepped `y_rep`
compared against real daily data reads as _model misfit_, so the failure mode is
tuning a model to match a recording artifact.

**Not affected — the likelihood.** Fitting real data scores against the live
filter, not a recorded trajectory. `multi_stream_obs.rs:303` mentions
`project_all_obs_times` only in a doc comment. So no published fit to real data
is wrong because of this.

**The sharpest consequence is methodological**, and this framing is owed to the
run-spec agent: for simulation-based calibration and parameter-recovery studies,
the data-generating process stops matching the likelihood. A coverage result
then measures _that mismatch_ rather than the sampler. The study reports
miscalibration and the sampler is not at fault.

**The recovery suite is not affected, accidentally.** `tests/recovery/cases/*`
declare `emit_schedule = every 7 'days` and no `output` block, so recording
stays at the default fine cadence and the weekly observation times land exactly
on snapshots. The synth recipe (`tests/recovery/Makefile:62-65`) passes no
`--output-every`. Observations _coarser_ than recording is the safe direction.
Nothing enforces this — adding `output { trajectories { every = 7 } }` to a
recovery case for file size would silently convert it into the trap above.

## 6. Remediation

**Now — make it loud.** Reject when observation times are not a subset of output
times. The condition is **misalignment, not coarseness**: an `emit_schedule` at
t = 3.5 against output every 1 snaps exactly as badly as daily-against-weekly.
Shipped in #597, inside `project_all_obs_times` so all five callers are covered.

**Then — re-seat the emission**, and this is harder than an earlier draft of
this report claimed. That draft said the run "already stops at observation
times, we just don't keep what it saw", reading `next_stop`'s formula
(`min(t_end, next_output, next_effect, next_obs)`) without checking whether the
axis it consults is populated. It is not: `Schedule::new` sets
`obs_times: Vec::new()` (`schedule.rs:214`) and the comment says so — the
observation axis is filled only for the inference drivers. **The forward path
does not visit observation times at all.**

Adding those stops is not free either. For the exact backends a new boundary
changes reaction-versus-boundary competition, hence RNG consumption order, hence
the trajectories themselves — moving every stored artifact digest. The
constraint is not "don't change `traj.tsv`", it is "don't change what was
simulated", which likely pushes the design toward a separate capture or
interpolation rather than new stops. `chain_binomial` cannot represent arbitrary
off-`dt` observation times by construction (Snap policy), so it needs a
gh#125-style observation-to-`dt` check regardless, while still letting the
_output_ cadence differ.

**And consolidate, because it is the same work.** The five callers above each
re-implement _compile sampler → project → snap → sample_ slightly differently.
The defect is therefore wrong in five places at once, and a correct fix applied
once requires the emission to live once. Consolidation is not a follow-up to the
fix; it is the shape of the fix.

## 7. Testing: what would have caught this

Nothing in the suite varies the recording cadence and then inspects emitted
observations, which is why four months passed. The pins to add:

1. **Cadence invariance.** Same model, same seed, `emit_schedule` fixed; run at
   two different output cadences; assert the emitted observation series are
   identical. This is the property that actually matters and it is one
   assertion. It fails today and passes after the re-seating.
2. **The guard fires.** A misaligned `emit_schedule` is rejected (or warns),
   including the non-integer-offset case, not just the coarser-output case.
3. **A recovery-suite tripwire.** Assert the recovery cases' observation times
   are a subset of their output times, so the accidental safety in §5 becomes
   enforced.

Test 1 is the one to write first: it is independent of which remediation is
chosen, and it converts "the emission must not depend on recording" from a
comment into a check.

## 8. Process changes this suggests

- **A single-agent audit finding is a lead, not a fact — and challenging it is
  cheap.** This one was ranked from a heading, and the ranking was right by
  luck; the mechanism, the five-caller blast radius, and the SBC framing only
  appeared under "are we sure this is wrong?". Findings marked `[agent]` without
  corroboration should carry that status into any issue filed from them.
- **A flag that reads as presentational deserves a check that it is.**
  `--output-every` is documented as an output view. Any flag in that category
  that reaches a data path is a candidate for the same defect; the audit's
  finding 17 (`simulate --parallel` accepted and discarded) is the same family
  of "accepted, then silently does something other than advertised".
- **Type-level separations only help where they are used.** gh#233 built
  `OutputTimes`/`ObsTimes` to make this class unrepresentable, and the emission
  path simply does not use them. A newtype defends the code that adopts it and
  nothing else — worth remembering when a future consolidation claims a class of
  bug is closed.

## References

- gh#589 — the issue
- `docs/dev/reviews/2026-08-11-run-spec-audit.md` finding 22 — origin
- gh#233 — `OutputTimes` / `EffectTimes` / `ObsTimes`, and the reasoning for
  keeping the three axes distinct
- `b7fe919e` — introduced the emission path
- `298494bf` — refactored `incidence_over` for strata; inherited the defect
