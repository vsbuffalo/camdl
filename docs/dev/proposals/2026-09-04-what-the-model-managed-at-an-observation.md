# What the model managed at an observation

Status: proposed

A PGAS chain refused at initialisation names the observation that killed it by
its position in a queue:

```
every particle scored -inf at observation 16 (substep 96)
```

With twelve streams bound and interleaved in time, that number names nothing a
modeller can act on. Worse, it is not a stable identifier: it is a position on
the union axis, whose composition changes whenever the bound stream set changes.
Unbinding two streams renumbers everything, so the index cannot be used to
compare one ablation against another — the only method available without it.

The index is about to get worse. Under
`2026-09-05-observation-time-as-a-sum-type.md` an interval stream contributes
two boundaries per row, severing `obs_idx`'s one-to-one correspondence with
observations. A diagnostic keyed on `(stream, time)` survives that; one keyed on
the index does not.

Reported by the ebola-bdbv-camdl modelling team: four ablations of a
three-province model, sixteen chains each, thirteen refusals per run, no
responsible measurement identifiable. Those figures are their report, not a repo
artifact.

## The object

`StreamScore` (`sim/src/inference/prequential.rs:77`) already carries `stream`,
`y_obs`, `y_pred_samples`, `log_score` for the running fit. The refusal path
should read in the same vocabulary rather than inventing a third.

```rust
/// What the ensemble managed at one stream of one observation.
pub struct StreamAttempt {
    /// The declared stream name, not its queue position.
    pub stream: String,
    /// The observation's time on the model axis, and its calendar date when
    /// the run is anchored. `time` matches `ObsFilterEss.time`.
    pub time: f64,
    pub date: Option<String>,
    /// Three states the code already distinguishes and a bare `Option` would
    /// collapse.
    pub cell: ObsCellState,
    /// The stream's projected quantity across the LIVE particles — the value
    /// the likelihood family is built on. Not the family's mean in general.
    /// `None` when no particle was live, or when every projection was NaN.
    pub projected_max: Option<f64>,
    pub projected_median: Option<f64>,
    pub n_projected_zero: usize,
    pub n_projected_nan: usize,
    /// Live particles whose per-stream log-density was -inf, with the cause
    /// resolved. See "Why, not just which".
    pub n_neg_inf: usize,
    pub neg_inf_causes: Vec<(NegInfCause, usize)>,
    /// Denominators. `n_live + n_dead == n_particles`.
    pub n_live: usize,
    pub n_dead: usize,
    pub n_particles: usize,
}

pub enum ObsCellState {
    /// Scheduled here and observed.
    Scored { y_obs: f64 },
    /// Scheduled here, value missing (`NA`). No likelihood term; the
    /// accumulator reset still fires.
    Hole,
    /// Not scheduled at this union index — a sibling stream's cadence.
    NotScheduled,
}
```

`Hole` and `NotScheduled` are separated because the code separates them
deliberately (`multi_stream_obs.rs:541`, `:887`) and both return `0.0` from
`score_streams` (`:1315`, `:1324`), as does a genuine zero-density row —
`binom_logpmf` and `beta_binomial_logpmf` return exactly `0.0` for
`n == 0, k == 0` (`obs_loglik.rs:492`, `:534`), which `obs_loglik.rs:523`
describes as routine surveillance. Three states, one number, if we let it
collapse.

## The dead-particle mask is not optional

`pgas_init.rs:348` scores as:

```rust
obs_model.fold_into_acc(cflows, a);        // runs for dead particles too
*lw = if dead {
    f64::NEG_INFINITY                       // never touches the obs model
} else {
    obs_model.log_likelihood_from_flows_and_counts(a, cnt, obs_idx, params)
};
```

A particle killed earlier by a chain-binomial overshoot carries `-inf` without
the observation model having been consulted. `deaths.all_dead()`
(`pgas_init.rs:321`) catches only the all-dead case, so a mixture — most
particles dead from the process model, the rest scoring `-inf` on the
observation — reaches the collapse check and would be reported as a unanimous
observation refusal. That is the exact confusion the instrument exists to
remove, so **every field is reduced over live particles only**, and `n_dead` is
reported beside them. A refusal with `n_dead` near `n_particles` is a
process-model finding, not an observation-model one.

`fold_into_acc` runs before the branch and unconditionally, so dead particles'
flows are in `acc`. Projections must therefore be taken per live particle, not
from a pooled accumulator.

## Why, not just which

`log_likelihood_per_stream_from_flows_and_counts` (gh#269,
`multi_stream_obs.rs:1366`) gives the per-stream split whose sum is the joint,
which says _which_ stream returned `-inf`. It cannot say why, and the causes
have different fixes. For `beta_binomial_logpmf` (`obs_loglik.rs:534-543`) they
are:

| guard                               | depends on         |
| ----------------------------------- | ------------------ |
| `n == 0 && k != 0`                  | data only          |
| `alpha.is_nan() \|\| beta.is_nan()` | model — parameters |
| `k > n`                             | data only          |
| `alpha <= 0.0 \|\| beta <= 0.0`     | model — parameters |

This corrects a mechanism the modelling team and gh#752 both assumed. Their
hypothesis was that a modelled confirmation flow exceeding the observed specimen
count is impossible rather than unlikely. With `n = tests` bound from the aux
column and `k` the observed value, `k > n` involves **only data** — it is
constant across particles and across θ, so the modelled flow cannot cause it.
The model-dependent routes are NaN or non-positive shapes. The comment at
`obs_loglik.rs:530` names the live case directly: a mean written
`k * projected / denom` has NaN shapes exactly there. Tell them.

So the refusal path re-scores the failing observation once, per live particle,
through a diagnostic scorer that returns the guard that fired rather than a bare
`f64`. This is a failure-path-only call: the hot loop's `f64` return is
untouched.

## The three readings

Valid when `projected` and `y_obs` share units — a `FlowSum` stream under
`poisson(rate = projected)` or `neg_binomial(mean = f(projected))`. For a family
where `projected` feeds a probability (`Binomial`, `BetaBinomial`, `Bernoulli`,
`Beta`) or a proportion via `StreamProjection::Expr`
(`multi_stream_obs.rs:190`), rows 1 and 2 do not read directly and only the
cause table above applies.

| the numbers                                   | the reading                                |
| --------------------------------------------- | ------------------------------------------ |
| `projected_max` exactly 0, `y_obs` positive   | the model cannot produce this — structural |
| `projected_max` 0.3, `y_obs` 18               | it can; this draw is far out — prior       |
| `projected_max` finite, `n_neg_inf == n_live` | read `neg_inf_causes`, not the location    |
| `n_dead` near `n_particles`                   | a process-model failure, not observation   |

## What this costs to build

More API surface than "surface what is already computed". Only the joint scalar
is computed on this path; the per-stream split is computable but not computed,
and `score_streams` (`multi_stream_obs.rs:1300`), `project_stream_from_acc`
(`:1191`), `MultiStreamObsModel.streams` (`:924`) and `Stream` (`:887`) are all
private. New public accessors are needed for per-union scheduled/hole state,
per-stream projection, and the diagnostic scorer. Cheap at runtime — it fires
only on a chain being abandoned — but it is new code, not a re-exposure.

`projected` is a single scalar per stream per particle and is defined for
prevalence streams too, which project from `counts` rather than the accumulator
(`multi_stream_obs.rs:1201`). It can be NaN — an `Expr` projection such as
`I/(S+I+R)` at zero population — so `projected_max` and `projected_median` are
computed over the non-NaN live subset, `n_projected_nan` is reported, and both
are `None` when that subset is empty. `f64::max` silently absorbs NaN and
`partial_cmp().unwrap()` panics; neither is acceptable on a path already
handling a failure.

## The one change to the filter loop

`reset_due_acc` (`pgas_init.rs:356`) runs inside the closure that scores, so the
accumulator is zeroed before the collapse check at `:358` and the evidence is
gone. The loop splits into **score → check → reset**.

`fold_into_acc` (`:347`) stays in the score pass. It accumulates
(`acc[k] += bin`, `multi_stream_obs.rs:1124`); moving it to the reset pass
produces a silently wrong likelihood.

Every slot in the closure is particle-local — `log_weights`, `cum_flows`, `acc`,
`counts`, `deaths` are zipped index-wise (`pgas_init.rs:337`) — nothing between
the passes mutates `acc` or `cum_flows`, and the check is a read-only reduction.
No parallelism hazard.

Scope: `unconditional_smc_pass` runs **once per chain at initialisation**
(`pgas_init.rs:120`), not per sweep. So this is not a hot path and a dedicated
A/B bench could not resolve its cost above noise in an end-to-end fit. The gate
is a byte-identical A/B of trajectory and log-likelihood at a fixed seed —
correctness, not speed. The other ten `reset_due_acc*` call sites are left
alone; they belong to the observation-time work.

## Relationship to the observation-time proposal

Narrower than it may look, and the earlier framing of this overstated it. That
proposal splits reset and score across _different_ union boundaries — reset at a
period's `start`, score at its `stop`, two distinct indices driven by
`Period`/`StreamTimes` (`2026-09-05-...md`, "What moves in the runtime"). This
change keeps both at the same `obs_idx` and only reorders them into separate
passes. It does none of that work: `obs_idx` keeps its meaning, `Schedule` is
untouched, no start boundary exists, and ten call sites remain. The honest claim
is that the two are compatible, not that this does the other's job.

## What is deliberately not done

**No payload in the hot-path return type.** A
`Likelihood::Valid |
Likelihood::NegInf(payload)` enum constructs a payload only
when the value is `-inf`, so the earlier claim that it taxes the common path was
wrong. What the common path would actually pay is a wider return value and a
match at the sum site. The argument that survives is narrower: per-particle
`-inf` is routine — it is the mechanism by which particles die — so payload
construction scales with exactly the degenerate regimes where the filter is
already working hardest. The discrimination the payload would give is real and
is delivered instead by re-scoring once on the failure path, which costs the hot
loop nothing.

**No change to what `-inf` means.** It overloads "this particle is impossible
here", "arithmetic produced `-inf`", and "structurally infeasible".
`error.rs:321` draws that line deliberately — no `InitFallback` variant asserts
`p(y | θ) = 0`, reserved for gh#784's support logic — and reopening it while the
observation-time work is in flight would tangle two hard changes.

**No environment flag.** The reduction fires only when a chain is abandoned, so
there is no happy-path cost to gate. The score/check/reset split is
unconditional but runs once per chain, and gating it would mean two versions of
a loop that must produce identical arithmetic — the "v1 alongside v2" shape
`.claude/rules/rust-conventions.md` calls delete-on-sight, and a silent-wrong
risk outliving the measurement. If step 3 turns out not to be free, that gate
reuses `RUST_LOG=camdl_sim=debug`, which already carries the `-inf` and
skipped-observation logging in `pgas.rs` and `particle_filter.rs`, rather than
adding a parallel mechanism.

**No dry-run pre-flight subcommand.** A short probe config at the real particle
count answers the same question.

## Where the evidence lives, and how it reaches JSON

Today structure is destroyed in the middle of the path:

```
pgas_init.rs:359   InitFallback::SwarmCollapsed { obs_index, substep }
                       │   a value returned in Ok(...), not an error
                       │   carried on SimError::NonFiniteChainStart
                       ▼
error.rs:346       Display → "every particle scored -inf at observation 16 (substep 96)"
                       │   ← all structure is lost here
                       ▼
pgas.rs:1033       DiagnosticKind::BadInit { chain_id, params, reason: String }
                       ▼
                   diagnostics.json   (serde, tag = "type", snake_case)
```

`InitFallback` derives `Debug, Clone, PartialEq` only (`error.rs:326`) and never
reaches JSON; `DiagnosticKind` is the serialization boundary
(`diagnostic.rs:94`). So the observation index leaves the crate inside prose,
and a regex over `reason` is not a consumer's preference but the only available
interface.

The evidence must ride on `InitFallback`: the collapse is detected in `sim`, the
diagnostic is built in `cli` (`pgas.rs:1033`, `pmmh.rs:640`, `runner.rs:2326`),
and the error is the only carrier between them.

**Both ends change, in one step.**

- `InitFallback::SwarmCollapsed` gains `attempts: Vec<StreamAttempt>`. Not a
  sibling variant — two variants for one condition would force every match site
  to handle both, which is the fork `.claude/rules/rust-conventions.md` warns
  against.
- `DiagnosticKind::BadInit` gains `attempts: Vec<StreamAttempt>`.
- `reason` stays, and is **rendered from that same data** so the prose and the
  fields cannot drift. Shipping the prose first and the fields later would mean
  writing the rendering twice.

Consequences, stated so the implementer does not rediscover them:

- `Serialize` is needed on `InitFallback` and `StreamAttempt`. `SimError` is
  `Debug + thiserror::Error` today (`error.rs:1`), so this adds a serde surface
  to the error module that does not exist yet.
- `InitFallback`'s derived `PartialEq` is not reflexive over the NaN a
  projection can produce. Every external use is in
  `sim/tests/gh784_unconditional_init.rs:456,469,501` and is a `matches!` or a
  destructure on `obs_index`; none compares a whole `InitFallback`. Keep the
  derive with the caveat documented, or drop it — nothing depends on it.
- Adding a field to `BadInit` is additive. `cli/tests/pgas_bad_init_skip.rs:270`
  matches on `kind.type` and clones the whole object, so it is undisturbed. It
  is still a schema change to a consumed artifact.

## Staging

1. The score/check/reset split, alone, gated on a byte-identical A/B at a fixed
   seed.
2. `StreamAttempt`, `ObsCellState`, the diagnostic scorer and the public
   accessors. The prose `reason` names the stream and date; `bad_init` in
   `diagnostics.json` carries the structured fields, which requires `Serialize`
   on `InitFallback` (today `Debug, Clone, PartialEq` only, `error.rs:326`) and
   a `PartialEq` that is honest about NaN. The prose stays — it is read first,
   and `cli/tests/pgas_bad_init_skip.rs:268` already consumes the record.
3. Extend `WeightCollapse` (`pgas.rs:402`, recorded at `:2683`), which already
   tracks `n_windows`, `first_substep` and `min_alive` per sweep of the running
   filter. This is deliberately not a new mechanism: a third parallel collapse
   diagnostic would contradict the reuse argument this proposal opens with. The
   near-miss field `min_alive` exists because a binary flag discards the sweep
   that came within one particle of collapse; `StreamAttempt` should be
   attachable to that, not beside it.

## Tests

1. Red first: a model whose stream cannot produce a positive observation refuses
   naming that stream and date, not an index.
2. A `beta_binomial` whose shape parameters go NaN — the `k * projected / denom`
   form of `obs_loglik.rs:530` — reports `n_neg_inf == n_live` with the
   NaN-shape cause, against a finite `projected_max`.
3. A mixture of dead and `-inf`-scoring particles reports `n_dead` and
   `n_neg_inf` separately, and does not describe a process-model failure as an
   observation refusal.
4. The three cell states are distinguishable: a stream on another cadence
   reports `NotScheduled`, an `NA` reports `Hole`, and a `n == 0, k == 0` row
   reports `Scored { y_obs: 0.0 }` with a finite score.
5. Renumbering: unbinding a stream changes no `StreamAttempt` field for the
   streams that remain. This is the property the index lacks and the reason the
   identifier changed.
6. An `Expr` projection that goes NaN yields `projected_max: None` with
   `n_projected_nan` set, and neither panics nor reports a spurious maximum.
7. The score/check/reset split leaves trajectory and log-likelihood
   byte-identical at a fixed seed.
