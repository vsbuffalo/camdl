# The conditioned scenario predictive: give the forked arms an observation draw

Date: 2026-08-20 Status: SUPERSEDED — do not implement Related: gh#322
(contrasts), gh#326 (observation namespace in contrasts, deferred), gh#561 (a
scenario horizon `fit predict` cannot honour)

> **Superseded before implementation.** Adversarial review returned four KILLS.
> The gap this names is real; the design is not, and the question it reopens was
> decided six weeks ago in a direction this document never cites.
>
> **The scientific objection is the deepest one.** The contrast fork is derived
> as the last saved snapshot _strictly before_ the toggled intervention fires —
> an **interior** time inside the observed window. Verified: `fork at t=29` on
> data running to `t=60`. So a conditioned arm starts from a draw of the
> **smoothing** distribution `p(x_29 | y_1:60)` — conditioned on 31 days of data
> generated _after_ the fork — and then generates `y_rep` freely forward across
> that same region. For a **difference**, the shared `X_i(fork)` largely cancels
> this, which is why the existing `contrasts/` design is sound. For an
> **absolute calibrated band laid over the observed counts** — this document's
> stated purpose — it does not cancel: the band would be anchored on a state
> informed by the very observations it is plotted against.
>
> gh#641 settled the taxonomy with a citation this document should have found:
> Särkkä (2013), _Bayesian Filtering and Smoothing_, §1.3 Eqs (1.4)–(1.6) gives
> filtering `p(x_k | y_1:k)`, prediction `p(x_{k+n} | y_1:k)` and smoothing
> `p(x_k | y_1:T)`. A forecast is the _prediction_ distribution, which by
> definition originates at the **filtering** distribution — and the taxonomy has
> **no cell** for iterating forward from a smoothing distribution. gh#641 also
> deferred interior-time origins explicitly, on the ground that interior
> particle draws are the numerically weak ones.
>
> gh#642 asked the exact question this document reopens — _"whether it is a
> `simulate` flag or a `fit predict --horizon forecast`"_ — and closed as the
> flag, shipped at `242891ab` (`simulate --init-state`, keyed by the state
> file), documented at `docs/camdl-run-spec.md:5373` and tested in
> `rust/crates/cli/tests/gh641_init_state_forecast.rs`.
>
> ### The three mechanical kills, for the record
>
> - **The gh#561 carve-out is a loophole, not a distinction.** Nothing in the
>   conditioned path can honour a declared horizon either — `contrasts.rs`
>   hardcodes `run_end = model.simulation.t_end`, and `check_arm_horizons`
>   _forbids_ an arm from declaring otherwise. Today a declared-horizon scenario
>   hard-errors from two independent guards. Skipping the refusal would convert
>   that into a silent drop on both row families.
> - **The `scenario` column would fuse two independently-determined sets.**
>   `emit_contrasts` takes no scenario refs; arms come from the contrast _body_,
>   while `--scenario` names a different set from the _command line_. Verified
>   with no `--scenario` at all: `predictive/` carries `{fitted}` while
>   `contrasts/` was built from `{fitted, ctrl}`.
> - **The gh#326 dismissal is false in three ways.** `emit_schedule` is optional
>   and absent for fit-only models (the target class), and its absence is a hard
>   error; a data-supplied aux denominator has no value past `last_obs`, so a
>   binomial stream's band would be an identically-zero ribbon; and the arm
>   spans `[fork, run_end]`, so **most of it sits inside the observed window** —
>   which is exactly where gh#326's axis question lives. The first bin is also
>   mis-windowed: `incidence_over` starts accumulating at the fork, so a fork at
>   29 with a 5-day cadence emits a 1-day count wearing a 5-day label.
>
> **The recommended path instead**, which `main.rs:1116` already names in a live
> error message: lift the `--init-state` × `--draws` conflict by feeding
> `simulate` the paired `(θ_i, X_i)` join that `joint.rs` already computes, at
> the **terminal** origin. That path already honours a scenario's own horizon
> (gh#561's own resolution), supports anchored `--to` (gh#626), and is
> identity-keyed. Its blocker was gh#607, which has since landed.
>
> Kept for the measurement in §1, which stands, and as the record of a proposal
> that reopened a decided question because its author did not read the issues.

## The problem, in one measurement

An 8-chain × 2,000-sweep × 1,200-particle fit of a live outbreak model, then
`camdl fit predict` naming five scenario presets. Every arm lands in
`predictive/cases_national.tsv`:

```
82 baseline        free_forward   max time 88
82 control_10      free_forward   max time 88
82 control_25      free_forward   max time 88
82 control_50      free_forward   max time 88
82 control_25_ramp free_forward   max time 88
82 fitted          free_forward   max time 88
```

Six arms, all stopping at `last_obs`, all byte-identical there — because the
control forcings are identically zero before the fork. **The scenario axis is
populated and says nothing.**

Meanwhile the same run's `quantities/` sidecar does separate the arms, over the
full model horizon — but free-forward from `t = 0`. Cumulative infections at
`last_obs + 8 weeks`:

|                                               | q05    | q50    | q95     |
| --------------------------------------------- | ------ | ------ | ------- |
| free-forward from t = 0 (`quantities/`)       | 371    | 14,007 | 360,088 |
| forked from the smoothed state (`contrasts/`) | 27,229 | 54,410 | 131,259 |

(That posterior is not converged; read the shape, not the values.) These are not
the same object and only one of them is a forecast. A free-forward replay
re-simulates the epidemic from its initial condition under a posterior draw; it
answers "what does this parameter vector imply from scratch". A forked replay
starts from the smoothed state the filter actually inferred at the fork time; it
answers "given where we are, what happens next". The second is the question a
situation report asks.

**Today the conditioned answer exists on disk in a file nothing displays, and
the file that is displayed carries the unconditioned one.** `camdl 'scope` reads
`predictive/<stream>.tsv` and is already built for a forecast extent —
`PredictiveTab.tsx` carries a `windowMode: 'data' | 'full'` toggle whose comment
anticipates "a forecast [that] extends past [data end]". It reads `contrasts/`
not at all.

## Why this is one composition, not a new surface

Both halves exist, are correct, and are **already in the same command**.

`fit predict` calls `contrasts::emit_contrasts` directly (`predict.rs:1636`).
That reducer already does, per forkable posterior draw and per arm
(`contrasts.rs` module doc):

1. resolve the arm's θ through the 5-tier resolver;
2. fork from the smoothed `X_i(fork)` — `io::trajectories::read_state_at` plus
   `sim::chain_binomial::Resume { start: Some(..) }`, where `fork` is the last
   saved snapshot strictly before the toggled intervention's fire time;
3. **run `[fork, run_end]`**, `run_end` being the model's simulation horizon;
4. evaluate operand quantities, difference, band.

`fit predict` owns the other half: a `PredictiveSection` per
`(scenario,
horizon)`, `scenario` as the leading column, and the
`predictive/<stream>.tsv` writer.

So the forked trajectories already exist inside `fit predict`, already run to
the model horizon, and are already CRN-coupled across arms. **What is missing is
that they never reach the observation draw.** Step 4 differences quantities and
stops.

## What is actually in the way

`predict.rs:1008` refuses a scenario horizon, deliberately and correctly:

> gh#561: `fit predict` emits at the OBSERVED times (`leaf_times`, from the
> bound data), so a scenario's own `simulate { to }` cannot move this command's
> output window — honouring it here would be a no-op, and silently ignoring a
> declared horizon is the exact bug gh#561 is about.

That guard is right **for a free-forward arm**, which has no horizon of its own
that `fit predict` can honour. A **forked** arm is different: it already carries
one, `run_end`, derived by the contrast machinery rather than declared by the
user. The measured symptom — six arms stopping at `t = 88` — is the gh#561 guard
correctly refusing a horizon, applied to an arm whose horizon was never in
question.

## Design

Add a third horizon kind alongside `free_forward` and `one_step`:
**`conditioned`** — the arm forked from the smoothed state at the contrast fork
and run to `run_end`, with `y_rep` drawn through the observation model exactly
as the existing horizons do.

- **Reuses the existing section model.** `PredictiveSection` is already keyed on
  `(scenario, horizon)`. A conditioned arm is a new value of the second
  coordinate, not a new axis, so the writer, the header and the `scenario`
  column are untouched.
- **Reuses the existing forked trajectories.** `emit_contrasts` already produced
  them. The change is to return them (or to expose the per-arm replay it already
  performs) rather than to re-derive.
- **The gh#561 guard stays for `free_forward` and is not consulted for
  `conditioned`,** because a conditioned arm's horizon is derived, not declared.
  That distinction should be stated in the code where the guard lives, or the
  next reader will reasonably delete one of the two branches.
- **Emission times past `last_obs` come from the stream's own `emit_schedule`.**
  There are no observation times to reconcile with out there — see the gh#326
  note below.

### CLI

No new subcommand and no new flag is strictly required: a model that declares
`contrasts {}` and is asked for scenario predictives already names the arms.
Whether `conditioned` should be opt-in (`--horizon conditioned`) or automatic
when a contrast exists is the one genuine UX question, and it turns on whether a
conditioned arm should ever _replace_ the free-forward rows or only sit beside
them. Beside them is the safer default — the `horizon` column already
distinguishes them, and a reader comparing the two learns something.

### DSL

**None.** No new syntax, no IR change, no `ir/VERSION` bump. The `contrasts {}`
block, the scenario presets and the intervention toggles are all as they are.

### Workflow

The artifact lands in the run directory the viewer already reads, so no pipeline
change. `camdl 'scope`'s existing `windowMode: 'full'` toggle renders it without
modification.

## How this sits with gh#326

gh#326 defers the observation namespace inside contrasts because the obs-time
axis over a counterfactual window is unspecified. **The forecast case is
narrower**: past `last_obs` there are no observation times to reconcile with,
only the stream's own `emit_schedule`. A conditioned predictive therefore does
not need the general answer, and can land ahead of it.

The pre-fork region is where the general problem bites, and a conditioned arm
does not span it — it starts at the fork.

## The improvement over what exists

|                       | today                                     | with this                                         |
| --------------------- | ----------------------------------------- | ------------------------------------------------- |
| conditioned forecast  | in `contrasts/`, as a _difference_ only   | in `predictive/`, per arm, with a calibrated band |
| what the viewer shows | free-forward arms, byte-identical         | the arms as they diverge                          |
| observation scatter   | absent — contrasts band a latent quantity | drawn through the observation model               |
| horizon               | `last_obs`                                | the model's `run_end`                             |

The workaround available today, which the downstream team is using and which is
worth documenting either way: a quantity that _computes_ an observation mean
from latent state is not a reduction of the stream, so it is a legal contrast
operand —

```camdl
expected_daily_cases = rho * (tau * I + gamma * q_comm * I)
```

— and gives the per-arm policy curve in reportable units. But it is a **mean**,
so it cannot be laid over the observed counts as a calibrated band, and a smooth
line against scattered points reads as more confidence than the model has. That
gap is the reason for this proposal rather than a documentation note.

## Verification

- A model with `contrasts {}` and scenario arms produces `conditioned` rows
  whose times extend past `last_obs` to `run_end`, and whose values **differ
  between arms** — the current failure is that they do not.
- The conditioned rows at the fork time are identical across arms (CRN holds at
  the fork by construction) and diverge after it. That is the sharpest available
  oracle and it is cheap.
- A model with no `contrasts {}` is **byte-identical** — this must be a pure
  addition.
- The gh#561 refusal still fires for a declared scenario horizon on a
  free-forward arm. A test that only checks conditioned rows appear would let a
  regression through here.
- The forkable subset is honoured: conditioned rows exist only for draws that
  have a saved trajectory, and the count is reported rather than silently
  reduced (see the adjacent `thin` × stride issue).

## Out of scope

The general gh#326 observation namespace inside contrasts; `by <instant>`
decoupling; and the parameter-only-scenario fork (gh#327), which the downstream
now reports as a convenience rather than a blocker since a zero-effect
`transfer(fraction = 0.0, …)` marker drives the fork at no cost to inference —
verified byte-identical posterior at the same seed.
