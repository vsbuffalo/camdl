# Reactive interventions through the effect agenda

- Date: 2026-06-18
- Status: Draft implementation RFC
- Amendment (2026-06-18): the `scope = exogenous | particle` key shown in the
  examples below was **removed** from the shipped surface (IR 0.17 → 0.18).
  Only the exogenous (reported-surveillance) behavior is implemented, so a
  single-value key + an unwired `particle` arm were dead surface; triggers are
  exogenous implicitly. The `scope` key and `AgendaScope` return when
  latent-scope triggers are actually wired (PR4/PR5), at which point the key
  carries the inference-safety seam that justifies it. The `scope = …` lines in
  the examples are retained as the original design record.
- Supersedes: `2026-05-14-reactive-interventions-and-evsi.md` for the
  reactive-intervention design. The older document remains useful as motivation
  for EVSI and public-health use cases.
- Depends on: gh#233 boundary-authority consolidation (`Schedule::next_stop`,
  `Schedule::arrive`, `Schedule::substeps`, `ExactInferenceTimeline`)
- Scope: reactive policy DSL, IR shape, runtime agenda, forward simulation, PF /
  IF2-safe exogenous reactions
- Non-goals in phase 1: PGAS complete-data density for particle-local reactive
  policies, ODE root-finding triggers, EVSI orchestration

## Summary

Reactive interventions should not patch each backend's inner loop. The current
runtime already has the right substrate:

```rust
// schedule.rs
pub struct TimelineStop {
    pub t: f64,
    reasons: SmallVec<[StopReason; 4]>,
}

pub enum StopReason {
    Output,
    ScheduledEffect,
    Observation,
    End,
}

impl Schedule {
    pub fn next_stop(&self, cursor: &Cursor, t: f64) -> Option<TimelineStop>;
    pub fn arrive<S>(..., apply_effects: impl FnMut(&mut S, f64), record: impl FnMut(&mut S, f64));
    pub fn substeps(&self, cursor: Cursor, t_start: f64) -> Substeps<'_>;
}
```

Reactive interventions are an agenda problem:

```text
static schedule/event definitions
    + observation or state-triggered policy
    -> dynamic effect agenda
    -> ordinary effect batches at ordinary schedule stops
```

The runtime should add future effect times to an agenda, then feed those times
through the existing cursor-keyed effect path. That keeps one answer to "where
does time stop?" and one answer to "what fires there?"

## Lifecycle Rule

The ordering at a shared timestamp is part of the scientific semantics and must
be declared once:

```text
1. advance process to t
2. apply always-active events due at t
3. apply scheduled interventions due at t
4. balance / validate
5. output snapshot
6. score or ingest observation at t
7. reset observation accumulators
8. run reactive policies
9. enqueue future effects strictly after the observation boundary
```

Default rule: an observation at `t` may trigger a campaign at `t + lag`, where
`lag >= 0`, but the campaign is enqueued after scoring/inserting the observation.
It does not change the likelihood or output snapshot at the same `t` unless the
DSL later adds an explicit `stage = "post_observation_immediate"` feature.

## DSL

### First-class reactive block

Use a new block instead of overloading `interventions {}`. Scheduled policy and
reactive policy are different fire sources.

```camdl
reactive_interventions {
  cVDPV2_sia : when reported_afp_28d >= afp_trigger_threshold {
    after    = 21 'days
    action   = transfer(fraction = sia_coverage, from = S, to = V)
    once     = true
    scope    = exogenous
  }
}
```

The block reads like a policy:

- `when`: boolean expression over allowed trigger inputs.
- `after`: non-negative lag before the effect fires. Default `0`.
- `action`: same action grammar as interventions/events: `transfer`, `add`,
  block-form `set`.
- `once`: fire-and-disable. Default `true`.
- `cooldown`: optional minimum time between firings when `once = false`.
- `scope`: explicit trigger scope. Phase 1 supports `exogenous`; later phases
  can add `particle`.

### Trigger variables

Phase 1 should support observation-derived and deterministic accumulator
triggers, not arbitrary latent-state triggers.

```camdl
observations {
  weekly_afp : incidence(paralysis) {
    emit_schedule = every 7 'days
    likelihood = neg_binomial(mean = rho * expected, r = k)
  }
}

reactive_interventions {
  sia_after_detection : when sum_observed(weekly_afp, window = 28 'days) >= 2 {
    after  = 21 'days
    action = transfer(fraction = 0.70, from = S, to = V)
    once   = false
    cooldown = 180 'days
    scope  = exogenous
  }
}
```

Important UX rule: `observed(...)` means the data stream visible to policy, not
the latent model truth. If a user wants latent-state triggers, that should be
spelled explicitly and gated:

```camdl
reactive_interventions {
  # Later phase, not phase 1.
  sia_if_true_prevalence_high : when latent((I1 + I2) / N) > 0.01 {
    after  = 14 'days
    action = transfer(fraction = 0.7, from = S, to = V)
    scope  = particle
  }
}
```

### Reusable policy examples

Polio response after AFP detection:

```camdl
parameters {
  afp_trigger_threshold : count = 2
  sia_coverage          : probability = 0.7
}

reactive_interventions {
  mop_up_sia[p in province] : when sum_observed(weekly_afp[p], window = 28 'days)
                                   >= afp_trigger_threshold {
    after    = 21 'days
    action   = transfer(fraction = sia_coverage, from = S[p], to = V[p])
    cooldown = 180 'days
    once     = false
    scope    = exogenous
  }
}
```

Measles outbreak response:

```camdl
reactive_interventions {
  school_campaign[dist in district] : when reported_cases[dist].rolling(14 'days) >= 10 {
    after  = 7 'days
    action = transfer(fraction = min(max_campaign_doses / S_school[dist], 0.95),
                      from = S_school[dist], to = V_school[dist])
    once   = true
    scope  = exogenous
  }
}
```

Cholera ORV with repeated cooldown:

```camdl
reactive_interventions {
  orv_zone[z in zone] : when sum_observed(cholera_cases[z], window = 7 'days) > 100 {
    after    = 3 'days
    action   = transfer(count = min(S[z], daily_dose_capacity * 7),
                        from = S[z], to = V[z])
    cooldown = 30 'days
    once     = false
    scope    = exogenous
  }
}
```

Intervention-state readback for decaying effect:

```camdl
let days_since_sia[p in province] = t - mop_up_sia[p].t_last_fired
let recent_sia[p in province] =
  if mop_up_sia[p].times_fired > 0
  then exp(-log(2) * days_since_sia[p] / 90 'days)
  else 0

transitions {
  infection[p in province] : S[p] --> I[p]
    @ beta * (1 - vacc_eff * recent_sia[p]) * S[p] * I[p] / N[p]
}
```

## Capability Story

Reactive policies are not one capability. They split by agenda scope:

```rust
pub enum AgendaScope {
    /// All particles share the same future agenda. Trigger inputs are external
    /// observations or deterministic data available equally to all particles.
    SharedExogenous,
    /// Schedule resolves from the current parameter vector, but is still shared
    /// for a single fit-stage likelihood evaluation.
    ParameterDependent,
    /// Each particle has its own agenda because triggers depend on latent state.
    ParticleLocal,
}
```

Phase 1 supports:

```text
forward chain-binomial: SharedExogenous yes
forward gillespie:      SharedExogenous yes, if agenda produces ordinary effect times
forward ODE:            only precomputed exogenous agendas; no continuous root triggers
bootstrap PF / IF2:     SharedExogenous yes
correlated PF:          SharedExogenous yes, but agenda updates must be deterministic
PGAS density:           no reactive policies in phase 1
PMMH:                  same as underlying PF; SharedExogenous only
```

Phase 1 rejects:

```text
scope = particle
observed(...) triggers inside PGAS complete-data density
latent(...) triggers in inference
ODE root-finding triggers
parametric reactive fire times under Exact inference unless the schedule is
resolved once per likelihood evaluation and shared
```

This is not a weakness. It prevents the silent-wrong case where particles have
different future boundary sequences but the filter still assumes one shared
`Schedule`.

## IR Shape

Prefer extending `InterventionKind` and introducing a fire-source layer rather
than adding a second action system.

Current IR:

```rust
// ir/src/intervention.rs
pub enum InterventionKind {
    Scenario,
    Event,
}

pub struct Intervention {
    pub name: String,
    pub base_name: Option<String>,
    pub schedule: InterventionSchedule,
    pub actions: Vec<Action>,
    pub kind: InterventionKind,
}
```

Target IR:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionKind {
    Scenario,
    Event,
    Reactive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FireSource {
    Scheduled(InterventionSchedule),
    Reactive(ReactiveTrigger),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactiveTrigger {
    pub when: Expr,
    pub after: f64,
    pub once: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown: Option<f64>,
    pub scope: AgendaScope,
}

pub struct Intervention {
    pub name: String,
    pub base_name: Option<String>,
    pub fire: FireSource,
    pub actions: Vec<Action>,
    pub kind: InterventionKind,
}
```

Migration note: if this is too large for the first PR, keep existing
`schedule: InterventionSchedule` for `Scenario`/`Event` and add
`reactive_interventions: Vec<ReactiveIntervention>` to the model. The final
shape should still converge on "one action type, one effect resolver, different
fire sources."

## Runtime Shape

### Dynamic effect agenda

Do not mutate `Schedule` directly. Add a dynamic agenda that can answer "what is
the next reactive effect time?" and "what reactive effects fire here?"

```rust
pub enum EffectSource {
    Static(usize),   // index into model.model.interventions
    Reactive(usize), // index into model.model.reactive_interventions or unified Intervention
}

pub struct DueEffects {
    pub static_effects: Vec<usize>,
    pub reactive_effects: Vec<usize>,
}

pub trait EffectAgenda {
    fn next_time(&self) -> Option<f64>;
    fn due_at(&mut self, t: f64) -> DueEffects;
    fn observe_and_update(
        &mut self,
        t: f64,
        obs: &ObservationFrame,
        state: effects::StateRef<'_>,
        params: &[f64],
    ) -> Result<(), SimError>;
}
```

For static interventions, `TimelineEffects` remains the compact indexed table:

```rust
pub struct TimelineEffects {
    pub times: Vec<f64>,
    pub batches: Vec<Vec<usize>>,
}
```

For reactive policies, phase 1 can use an agenda with a min-heap:

```rust
#[derive(Clone)]
pub struct ReactiveAgenda {
    pending: BinaryHeap<Reverse<PendingEffect>>,
    last_fired: Vec<Option<f64>>,
    times_fired: Vec<u32>,
}

pub struct PendingEffect {
    pub t: f64,
    pub reactive_idx: usize,
    pub sequence: u64, // stable tie-breaker: declaration order / enqueue order
}
```

### Schedule integration

The current `Schedule` owns static effect times. Reactive agendas add new
runtime effect times, so the boundary authority must take both into account.
Do this by adding a merged query instead of cloning/rebuilding `Schedule` on
every enqueue:

```rust
pub struct AgendaView<'a> {
    pub schedule: &'a Schedule,
    pub agenda: &'a dyn EffectAgenda,
}

impl AgendaView<'_> {
    pub fn next_stop(&self, cursor: &Cursor, t: f64) -> Option<TimelineStop> {
        let static_stop = self.schedule.next_stop(cursor, t)?;
        let reactive_t = self.agenda.next_time().unwrap_or(f64::INFINITY);
        if reactive_t < static_stop.t - EFFECT_EPS {
            return Some(TimelineStop::reactive(reactive_t));
        }
        if (reactive_t - static_stop.t).abs() <= EFFECT_EPS {
            return Some(static_stop.with_reason(StopReason::ScheduledEffect));
        }
        Some(static_stop)
    }
}
```

That sketch implies either:

1. Add `StopReason::ReactiveEffect`, or
2. Treat reactive effects as `ScheduledEffect` once they are enqueued.

Prefer option 2 for phase 1. Once an effect has a concrete fire time, the
application lifecycle is identical to a scheduled intervention. The source
distinction lives in the agenda batch, not in the lifecycle order.

### Applying reactive effects

Reuse the existing pure effect resolver. A reactive effect's actions should be
resolved exactly like scheduled intervention actions:

```rust
fn apply_due_reactive_effects(
    model: &CompiledModel,
    agenda: &mut ReactiveAgenda,
    due: DueEffects,
    t: f64,
    int_s: &mut IntState,
    real_s: &mut RealState,
    params: &[f64],
) -> Result<(), SimError> {
    for idx in due.reactive_effects {
        let iv = &model.model.reactive_interventions[idx];
        crate::effects::apply_intervention_effects(
            &iv.as_intervention_view(),
            model,
            effects::StateMut { int: int_s, real: real_s },
            t,
            params,
        )?;
        agenda.mark_fired(idx, t);
    }
    Ok(())
}
```

No new arithmetic. No duplicate transfer rounding. No backend-specific action
semantics.

## Inference

### Shared-exogenous policies

A shared-exogenous policy is one whose trigger inputs are identical for every
particle at a given observation boundary. Examples:

```camdl
when observed(weekly_cases) >= 10
when sum_observed(weekly_afp, window = 28 'days) >= 2
```

In PF / IF2, the agenda update happens after scoring the observation at `t` and
before the next observation window begins. Because the agenda is shared, every
particle sees the same added future effect times. This preserves the shared
`Schedule::substeps` CRN shape.

Code shape:

```rust
for obs_idx in 0..obs.len() {
    let cursor = cursor_for_obs(obs_idx);
    particles.par_iter_mut().for_each(|p| {
        for (t0, step_dt, fired_effect) in schedule.substeps(cursor, t) {
            p.advance(t0, step_dt, fired_effect)?;
        }
        p.score_observation(obs_idx)?;
    });

    // single-threaded, shared agenda update after scoring
    agenda.observe_and_update(t_obs, &obs_frame, population_summary, params)?;
}
```

### Particle-local policies

Particle-local policies are later-phase only:

```camdl
when latent(I / N) > 0.01
```

They require agenda state to be part of particle state:

```rust
pub struct Particle {
    state: IntState,
    real: RealState,
    agenda: ReactiveAgenda,
}
```

Resampling must clone agenda state. PGAS ancestor sampling must account for
agenda history. Correlated PF loses the simple shared-boundary coupling. This
is supportable, but not phase 1.

### PGAS

PGAS should reject reactive policies in phase 1. The old proposal was right that
complete-data density is the hard surface; the new scheduling spine only fixes
where events land. It does not automatically give the density term for a
state-dependent policy.

The phase-2 density requirement is:

```text
log p(path | theta, policy)
  = sum substep transition log probabilities
    + deterministic policy consistency checks
```

For latent deterministic triggers, the policy term is either `0` if the recorded
agenda/firing history matches the path and `-Inf` otherwise. For observed
triggers, the observed-data process that triggered the policy must be included
explicitly. Do not infer this from trajectory side effects.

## CAS / RunInput

Reactive policy is semantic input. It must enter model identity through the IR
and therefore through the existing `runid` model digest. Agenda runtime state is
not an input; it is a deterministic consequence of model + data + seed.

Add differential tests:

```text
change threshold        -> run_id changes
change after lag        -> run_id changes
change cooldown         -> run_id changes
change once             -> run_id changes
change label/comment    -> run_id stable
change observed data    -> fit/pfilter run_id changes through data digest
```

## Implementation Plan

### PR 1 — DSL/IR skeleton with hard runtime rejection

- Add parser/AST for `reactive_interventions`.
- Add IR type and serde.
- Dim-check `when` as boolean and `after/cooldown` as time.
- Validate action targets using the same intervention validation.
- Runtime rejects any model with reactive interventions with a clear capability
  error.
- IR golden bump.

Acceptance:

```text
camdlc accepts the examples above and emits reactive IR
bad target compartment is rejected
non-boolean when is rejected
negative after/cooldown is rejected
runtime error says reactive interventions are parsed but unsupported
```

### PR 2 — Agenda runtime for forward chain-binomial

- Add `ReactiveAgenda`.
- Add phase-1 trigger primitives: `observed(stream)`,
  `sum_observed(stream, window = ...)`, `rolling(...)`.
- Evaluate policies after observation/output boundary, enqueue effects.
- Apply due reactive effects through existing `effects` resolver.
- Record fired reactive effects in event log or a small `reactive_log.tsv`.

Acceptance:

```text
threshold crossed at t=28, after=21 -> action fires at t=49
cooldown suppresses repeated threshold crossings
once=true fires exactly once
reactive transfer uses the same rounding/clamp as scheduled transfer
same model with equivalent precomputed scheduled SIA matches reactive agenda output
```

### PR 3 — Forward gillespie / ODE capability

Gillespie can consume enqueued concrete effect times through `Schedule::clip` /
`next_stop` once the agenda view exists. ODE can consume enqueued concrete times
too, but continuous root triggers remain out of scope. Both should support only
observation-triggered/exogenous agendas, not latent root-finding.

Acceptance:

```text
gillespie and chain agree on a zero-rate model with a reactive transfer
ODE accepts observed exogenous agenda but rejects latent/root trigger syntax
```

### PR 4 — PF / IF2 shared-exogenous support

- Agenda updates happen after scoring each observation.
- The schedule for the next window includes newly enqueued concrete effect times.
- Reject particle-local triggers.

Acceptance:

```text
PF loglik is deterministic across thread counts
reactive policy changes pfilter run_id
same shared agenda applied to all particles
particle-local syntax gives a capability error
```

### PR 5 — PGAS design and density

Separate PR and review. Do not combine with syntax/runtime support.

Acceptance before enabling:

```text
complete-data density matches forward generated path with reactive firing
inconsistent recorded firing history has -Inf density
gradient finite-difference check passes on a reactive model
PGAS baseline without reactive policies remains byte-identical
```

## Validation Fixtures

The validation is phased to match the implementation: PR1 pins the **IR shape and
the rejection**; PR2 pins **behavior** with `reactive_log.tsv` and deterministic
trajectory expectations on tiny eyeball-able models. The single most important
fixture is the **equivalence oracle** (PR2 #4): a reactive-derived fire time must
produce the same trajectory as an ordinary scheduled intervention placed at that
time.

### PR1 — compiler / IR goldens (runtime still rejects)

Compile-only goldens (the runtime cannot simulate an active reactive policy yet),
asserted by compiling the `.camdl` and comparing the emitted IR.

1. **`reactive_sir_observed_threshold.camdl`** — minimal SIR with an observation
   stream and a single reactive policy:

   ```camdl
   reactive_interventions {
     sia : when sum_observed(weekly_cases, window = 28 'days) >= trigger_threshold {
       after  = 21 'days
       action = transfer(fraction = sia_cov, from = S, to = V)
       once   = true
       scope  = exogenous
     }
   }
   ```

   Pins `FireSource::Reactive`, `TriggerExpr`, `after`, `once`, `scope`.

2. **`reactive_indexed_patch_sia.camdl`** — an indexed policy:

   ```camdl
   reactive_interventions {
     sia[p in patch] : when sum_observed(weekly_cases[p], window = 28 'days) >= 2 {
       after    = 14 'days
       action   = transfer(fraction = cov[p], from = S[p], to = V[p])
       cooldown = 180 'days
       once     = false
       scope    = exogenous
     }
   }
   ```

   Pins stratification expansion, `base_name`, indexed stream references, and
   indexed action targets.

3. **Negative compiler fixtures** — each rejected with a located diagnostic:
   - `observed()` in a transition rate (**E278**).
   - `.rolling(...)` method syntax (unsupported — bare syntax error **E001**).
   - non-boolean / non-comparison `when` (**E273**).
   - unknown observation stream (**E279**).
   - negative or non-finite `window` / `after` / `cooldown` (**E274**).
   - `once = true` together with `cooldown` (**E276**).
   - unknown action target — rejected through the shared intervention action
     validation (E265 / `resolve_comp_name`), not a reactive-specific path.

### PR1 — runtime (capability) tests

- A reactive model with the policy **inactive/default** runs the baseline (the
  dormant policy is inert).
- The same model with `--enable sia` rejects with the `REACTIVE_INTERVENTIONS`
  capability message.
- A scenario that `enable`s the reactive policy also rejects.
- The fit / pfilter path rejects an active reactive policy through the inference
  method capability validation.

### PR2 — runtime behavior goldens

Tiny deterministic models whose expected values can be eyeballed. Commit the
source TSV and the expected `reactive_log.tsv`.

1. **Lag.** Observation crosses at `t = 28`, `after = 21` → the action fires at
   `t = 49`. Expected `reactive_log.tsv`:

   ```text
   trigger_time  policy  trigger_value  threshold  fire_time  action
   28            sia     2              2          49         transfer
   ```

   Expected trajectory: `S` drops and `V` rises at exactly `49`, not `28`.

2. **Cooldown.** Observations exceed the threshold at `t = 14, 21, 28, 60`,
   `cooldown = 30`: fires at the first eligible crossing, suppresses the middle
   crossings, fires again after the cooldown elapses.

3. **Once.** Multiple threshold crossings, `once = true` → exactly one firing.

4. **Equivalence (the semantic oracle).** A reactive policy driven by a known
   observed sequence produces the **same trajectory** as a precomputed scheduled
   intervention placed at the resulting fire time.

5. **Default-off.** The same model with and without `--enable sia`: the baseline
   has no firing; the enabled run fires.

### Figures (docs only — not the primary gate)

Generated from committed TSVs; commit the script and the source/expected TSVs,
the PNG optional unless docs need it. One figure suffices: observed weekly cases
(bars), the threshold (horizontal line), the trigger time and the campaign fire
time (two vertical lines), and the `S` / `V` trajectories showing the effect at
the fire time, not the trigger time.

## Phasing / what's next

- **PR1 (this PR, gh#204):** IR `FireSource` ADT, `TriggerExpr`, the DSL surface,
  compiler validation, and the capability rejection — plus the PR1 fixtures
  above. Lands the schema/golden break (IR 0.16 → 0.17) in isolation.
- **PR2 — forward chain-binomial agenda runtime:** `ReactiveAgenda`, the trigger
  primitives (`observed` / `sum_observed`), evaluate-after-observation +
  enqueue-after-the-boundary, apply through the existing `effects` resolver,
  `reactive_log.tsv`, and the PR2 behavior goldens. Scope-limited to
  `SharedExogenous`.
- **PR3 — gillespie / ODE:** consume enqueued concrete effect times via the
  schedule view; observed/exogenous agendas only, no continuous root triggers.
- **PR4 — PF / IF2 shared-exogenous:** agenda update after scoring each
  observation; the shared agenda preserves the CRN coupling; reject
  particle-local triggers.
- **PR5 — PGAS density:** a separate design + review surface (the complete-data
  density term for a state-dependent policy); not bundled with runtime support.

## Sharp UX Points

1. **Observed vs latent must be explicit.** Public-health users often mean
   reported cases, not true infections. We should warn a user if latent is used.
2. **Lag defaults to zero but same-time semantics are post-observation.** An
   observation-triggered campaign at `t` does not affect scoring at `t`.
   Zero lag settings also should get a user warning.
3. **Cooldown is not once.** `once=true` disables forever; `cooldown` suppresses
   repeated firings temporarily.
4. **Scope is not optional internally.** The compiler may default to
   `scope=exogenous` when only `observed(...)` appears, but the IR must carry
   the scope so inference can reject unsafe combinations.
5. **Reactive action names need stable identity.** Indexed policies expand with
   `base_name` like interventions so scenarios/logs can address `sia_borno`.
6. **Every firing should be auditable.** Users need `reactive_log.tsv` with
   `time`, `policy`, `trigger_value`, `threshold`, `action`, and `count_moved`.

## What Not To Do

- Do not put `when` inside ordinary `interventions {}` as another schedule form.
  That hides the agenda-scope distinction.
- Do not evaluate triggers inside every backend ad hoc.
- Do not support latent-state particle-local triggers in PF/PGAS until agenda
  state is part of particle state and resampling/ancestor tracing clone it.
- Do not let reactive policies affect the observation that triggered them by
  accident.
- Do not add new transfer/add/set arithmetic. Reuse `effects.rs`.

## Agent Instructions

If implementing this proposal:

1. Start with PR 1 only. It is okay for the runtime to reject the new IR.
2. Preserve the current `Schedule` invariants. Reactive agenda should feed the
   boundary authority, not replace it.
3. Keep the first supported policy class `SharedExogenous`.
4. Route all actions through `effects.rs`.
5. Add capability errors before broadening support.
6. Treat PGAS as a separate design surface.

