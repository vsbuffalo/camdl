# What the model managed at an observation

Status: proposed

A PGAS chain refused at initialisation says which observation killed it by its
position in a queue:

```
every particle scored -inf at observation 16 (substep 96)
```

With twelve streams bound and interleaved in time, that number names nothing a
modeller can act on. Worse, it is not a stable identifier: it is a position on
the union axis, whose composition changes whenever the bound stream set changes.
Unbinding two streams renumbers everything, so the index cannot even be used to
compare one ablation against another — which is the only method available
without it. A downstream team ran four ablations, thirteen refusals each, and
could not name a single responsible measurement.

The index is about to get worse rather than better. Under
`2026-09-05-observation-time-as-a-sum-type.md` an interval stream contributes
two boundaries per row, which severs `obs_idx`'s one-to-one correspondence with
observations outright. Any diagnostic keyed on that number needs rewriting then.
One keyed on `(stream, time)` does not.

## What this is not

Not a new subsystem. Every quantity below is already computed at the moment of
refusal and discarded one line later. The change is a reduction over live data
on a path that is about to abandon the chain anyway.

## The object

One type, three consumers. `StreamScore` (`sim/src/inference/prequential.rs`)
already carries `stream`, `y_obs`, `y_pred_samples`, `log_score` for the running
fit; `ObsFilterEss` (`cli/src/fit/filter_ess.rs`) already carries `obs`, `time`,
`mean`, `min`. The refusal path should emit the same shape rather than a
parallel one.

```rust
/// What the ensemble managed at one stream of one observation.
pub struct StreamAttempt {
    /// `ObservationModel.name` — the declared stream, not its queue position.
    pub stream: String,
    /// The observation's time on the model axis, and its calendar date when
    /// the run is anchored.
    pub t: f64,
    pub date: Option<String>,
    /// The observed value. `None` is a hole: scheduled, no value, no term.
    pub observed: Option<f64>,
    /// The stream's projected quantity across the ensemble — the value the
    /// likelihood family is built on (`rate` for poisson, the mean's argument
    /// for neg_binomial). Not the family's mean in general, and named so.
    pub projected_max: f64,
    pub projected_median: f64,
    /// Particles whose projection was exactly zero.
    pub n_projected_zero: usize,
    /// Particles whose per-stream log-density was -inf. Distinguishes a
    /// support violation from a small mean; see below.
    pub n_neg_inf: usize,
    pub n_particles: usize,
}
```

`projected_max` is deliberately not called a predicted mean. For
`poisson(rate = projected)` the two coincide; for
`neg_binomial(mean = p_report * projected, r = k)` they do not, and reporting
the family's mean would require evaluating the family per particle rather than
reading a value already in hand.

## The three readings it separates

These have different fixes and are currently indistinguishable.

| what the numbers say                                   | what it means                                                  |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| `projected_max` exactly 0, `observed` positive         | the model cannot produce this observation here — structural    |
| `projected_max` 0.3, `observed` 18                     | the model can; this prior draw is far out — the prior is wrong |
| `projected_max` plausible, `n_neg_inf` = `n_particles` | a support violation in the family, not a location problem      |

The third row is the one a max-and-mean report alone would miss, and it is the
mechanism a bound denominator makes reachable: a `beta_binomial(n = tests, …)`
whose modelled flow exceeds the observed specimen count is _impossible_ rather
than unlikely, at any location. `n_neg_inf` against a finite `projected_max` is
that signature.

## Where it is produced

`pgas_init.rs:358` already detects the collapse:

```rust
if !log_weights.iter().any(|w| w.is_finite()) { … }
```

Everything the object needs is in scope there. Two pieces already exist and are
reused rather than rewritten:

- `log_likelihood_per_stream_from_flows_and_counts` (gh#269) gives the
  per-stream split whose sum is the joint, so identifying _which_ streams
  returned `-inf` needs no new scoring code.
- `score_streams` already resolves `t = self.obs_times[obs_idx]`, reads
  `observed` from the stream's own local cell, and skips streams not scheduled
  at this union index. `ObservationModel.name` is on `StreamSpec.ir_model`.

## The one change to the filter loop

The reset runs inside the closure that scores:

```rust
*lw = … log_likelihood_from_flows_and_counts(a, cnt, obs_idx, params);
for f in cflows.iter_mut() { *f = 0; }
obs_model.reset_due_acc(obs_idx, a);
});
if !log_weights.iter().any(|w| w.is_finite()) { … }
```

so the accumulator is zeroed before the collapse is detected and the evidence is
gone. Snapshotting `acc` every observation to preserve it would be a real
per-observation cost paid to diagnose a rare event.

The loop splits into **score → check → reset**. The reset moves to its own pass
over cache-warm data; nothing is allocated and no arithmetic changes.

This is the shape the accumulator is moving to regardless: the observation-time
proposal splits `reset_due_acc` into reset-at-a-start and score-at-a-stop. Doing
it here means doing it once, in a small change with an isolated test, instead of
as collateral inside the harder one.

It lands as its own commit, gated on a byte-identical A/B of trajectory and
log-likelihood, with the cost measured rather than asserted. The observation
path is not the inner loop — the substep loop is — so the expectation is well
under 1%, but that is a prediction to check, not a claim.

## What is deliberately not done

**No instrumentation in the likelihood's return type.** A
`Likelihood::Valid |
Likelihood::NegInf(payload)` enum would construct a
diagnostic payload on the common path: `-inf` is not an error in a particle
filter, it is the mechanism by which particles die, and individual particles
score it routinely and correctly. The rare event is _unanimous_ `-inf`, which
the existing `any(is_finite)` predicate already detects from plain `f64`s.
Paying continuously to diagnose that would be work spent on the happy path for a
failure path's benefit.

**No change to what `-inf` means.** It currently overloads "this particle is
impossible here", "arithmetic produced `-inf`", and "structurally infeasible".
That is a real question, but `error.rs:323` already draws the line deliberately
— no `InitFallback` variant asserts `p(y | θ) = 0`, which is reserved for
gh#784's support logic — and reopening it while the observation-time work is in
flight would tangle two hard changes.

**No dry-run pre-flight subcommand.** A short probe config at the real particle
count answers the same question.

**No environment flag to toggle the instrumentation.** There are two costs here
and neither wants a runtime switch.

The reduction at the refusal site runs only when a chain is being abandoned, so
it costs nothing when nothing fails — there is no happy-path cost to gate.
Adding a flag would be a knob whose only setting worth using is "on".

The score/check/reset split _is_ unconditional, so its cost is real, but a flag
is the wrong instrument for it. Gating the hot loop on an environment variable
means carrying two versions of the scoring loop indefinitely, which is the "v1
alongside v2" shape `.claude/rules/rust-conventions.md` calls delete-on-sight;
and two paths through a loop that must produce identical arithmetic is a
silent-wrong risk that outlives the measurement it was added for. Worse, a knob
that changes what the filter does would have to enter run identity or else two
runs differing only in it would collide.

The measurement is a one-time question about two commits, not a permanent
capability. `rust/crates/sim/benches/` already has the pattern —
`binomial_ab.rs`, `eval_ab.rs`, `flat_eval.rs`: both arms in one process,
interleaved per rep with the order alternating by rep parity so thermal drift
and background load hit them equally, median-of-9. `binomial_ab.rs` states the
principle for this exact case: both arms run "in one process override rather
than an env var."

So: an `obs_reset_ab` bench in that shape, run once, its number recorded in the
commit message and in a dev note. If the split ever needs revisiting, the bench
is there to re-run against a new pair of commits.

Step 3 is where this reasoning could change. Instrumenting the _running_ filter
does cost something on every observation of every sweep, and if that turns out
not to be free it needs a gate. If so, the gate reuses the diagnostics surface
camdl already has — `RUST_LOG=camdl_sim=debug`, alongside the -inf and skipped
observation logging in `pgas.rs` and `particle_filter.rs`, or
`CAMDL_TRACE_STEPS` — rather than adding a parallel mechanism. That decision
belongs to step 3, on the strength of a measurement, not now.

## Staging

1. The score/check/reset split, alone. Gated on a byte-identical A/B of
   trajectory and log-likelihood at a fixed seed, with the cost measured by an
   `obs_reset_ab` bench in the established shape. If the split turns out to cost
   more than a percent, that is a result worth having before step 2 is written,
   not after.
2. `StreamAttempt` and the reduction at the refusal site; the prose `reason`
   names the stream and date, and `bad_init` in `diagnostics.json` carries the
   structured fields. The prose stays — it is what gets read first.
3. Only then, the same reduction inside the running filter, so a mid-fit
   collapse at observation 40 reads the same way as a refusal at observation 3.
   This touches `particle_filter.rs`, which `CLAUDE.md` names high-risk, and is
   worth doing only once the object has proven right on the cheaper path.

## Tests

1. Red first: a model whose stream cannot produce a positive observation refuses
   with that stream's name and date in the message, not an index.
2. A support violation (`beta_binomial` with a bound denominator below the
   modelled flow) reports `n_neg_inf == n_particles` against a finite
   `projected_max` — the row a mean-only report would misread.
3. A hole at the failing index contributes no term and is reported as
   `observed: None` rather than as a zero.
4. Renumbering: unbinding a stream changes no `StreamAttempt` field for the
   streams that remain. This is the property the index does not have and the
   reason the identifier changed.
5. The score/check/reset split leaves trajectory and log-likelihood
   byte-identical on a fixed seed.
