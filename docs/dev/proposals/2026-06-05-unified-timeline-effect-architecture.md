---
date: 2026-06-05
status: proposal
related:
  - 2026-06-05-observation-data-binding.md
  - 2026-05-14-reactive-interventions-and-evsi.md
  - archive/pre-alpha/2026-05-04-ode-inference-three-phase.md
area: simulation engine / inference / DSL
issue: TBD
---

# Unified timeline-effect architecture

## Problem

A simulation in camdl is one thing: a latent state advancing through time, with
**events on a timeline** — observations read off it, interventions and cohort
entries written onto it, conservation constraints imposed on it, and (later)
reactive policies fired when it crosses a threshold. Today that one notion is
expressed through several parallel surfaces, each with its own loading path, its
own application logic, and — most consequentially — its own way of mapping
continuous time onto integrator steps. Every such surface is a place the forward
simulator and the inference engine can quietly disagree, and a place a bug can
enter without a test noticing.

This proposal consolidates the surfaces. The thesis: exactly one component is
legitimately special — the **process**, which advances the true latent state
with randomness — and everything else (observation, intervention, event,
balance, reactive intervention) is a **triggered effect on a shared timeline**,
applied in one **canonical substep lifecycle**, expressible through one set of
types. The forward simulator and the inference engine become thin _drivers_ over
that shared substrate, diverging only where they genuinely must.

It also makes one long-standing implicit inconsistency explicit and
user-controllable: today the bootstrap particle filter steps _exactly_ to
observation times while PGAS _snaps_ them to the integrator grid. We expose that
as a single `snap | exact` knob, default `snap`, with a staged path to flipping
it once it is profiled and validated.

## Terminology

The vocabulary this proposal uses throughout. Defined once here so the design
sections can stay dense.

- **Process / kernel** — the one legitimately special component: the
  state-advance that moves the true latent state forward with randomness (the
  propensity draws). Each backend (Gillespie, tau-leap, ODE, chain-binomial) has
  its own kernel; this is the part the design does _not_ unify.
- **Substep** — one integrator `dt` step. The process advances substep by
  substep; many substeps occur between two boundaries.
- **Boundary** — a point in time where the integrator _must_ stop because
  something is due there: the run end (`t_end`), an output-snapshot time, or a
  scheduled effect time. Between boundaries the process only advances. The
  `Boundary` enum names the kinds, and several can coincide at one time (an
  output and a cull both at t=10).
- **Effect** — a scheduled modification of, or read from, the state on the
  timeline: an **observation** (read), an **event** (cohort entry / importation,
  fired every substep), an **intervention** (cull / vaccinate at scheduled
  times), a **balance** constraint (population conservation), or a future
  **reactive** intervention (fired on a state threshold). The typed taxonomy is
  Observe / Event / Intervene / Constrain / Reset.
- **Spine / schedule** — the merged, sorted timeline of all boundaries,
  immutable and shared. Every driver consumes the same spine. Implemented as the
  `Schedule` type (`sim/src/schedule.rs`).
- **Cursor** — a small, `Copy`, per-particle _position_ in the spine: the
  indices marking the next un-emitted output (`output_idx`, the **output
  cursor**) and the next un-applied effect (`effect_idx`). Named for the moving
  pointer into the sorted times, like a text cursor. Each particle in a swarm
  carries its own; the spine is never mutated (the CRN invariant below).
- **Driver** — a top-level loop that turns parameters into a result by walking
  the spine: `run_forward` (generate a trajectory), `run_filter` (score a
  likelihood), `run_trajmatch` (deterministic, reserved). Drivers are thin; the
  kernel and the effects do the work.
- **snap vs exact** — the two boundary policies. **snap** rounds output/effect
  times onto the `dt` grid (chain-binomial today; PGAS); **exact** lands the
  integrator precisely on each boundary (tau-leap / ODE; the bootstrap PF). For
  dt-independent backends (Gillespie, ODE) they coincide.
- **fire_steps** — the per-intervention step _indices_ (not times) at which an
  intervention fires, resolved from the run `dt` by `resolve_fire_steps`. This
  is how chain-binomial snaps an off-grid intervention time onto a step, inside
  `step_one` — the mechanism the Stage-1 extraction deliberately leaves in
  place.
- **CRN** — common random numbers / paired-seed coupling: the same seed must
  produce the same RNG draw order so paired scenarios stay coupled. Any reorder
  of draws breaks it. The spine guarantees its half: N particles walk an
  identically-ordered boundary sequence because the cursor is the only
  per-particle state and `next_boundary` is pure.

## The existing infrastructure, and how it is spread

The inventory the design must honour. Citations are to the current tree.

### Process: two trait hierarchies, one kernel

- `Simulate` (`sim/src/lib.rs`) — the forward path. `OdeSim`, `TauLeapSim`,
  `GillespieSim`, `ChainBinomialSim`, each owning its own stepping loop and
  recording a full `Trajectory`.
- `ProcessModel` / `DensityProcess` (`sim/src/inference/traits.rs:40,141`) — the
  inference path, implemented by exactly **one** type, `ChainBinomialProcess`
  (`chain_binomial_process.rs:52`). Inference is chain-binomial-only;
  `DensityProcess` (the transition density PGAS needs) is chain-binomial by
  design.

The transition math is **not** duplicated: both paths call the same kernel
`step_one` (`chain_binomial.rs:269`); `ChainBinomialProcess::step`
(`chain_binomial_process.rs:91-98`) resolves `fire_steps` and delegates to it.
What differs is the _loop around the kernel_ and an allocation contract:
`ProcessModel::step` is the hot inner loop (`n_particles × n_substeps × n_obs`
calls, parallel across particles) and **must not allocate** — it threads a
reusable `Scratch` (`traits.rs:62`). The forward driver allocates freely.

### Stepping: three idioms, two boundary policies

1. **Forward — merged boundary.** `tau_leap.rs:111-116`, `ode.rs:210-215`:
   `next_boundary = min(t_end, next_output, next_intervention)`, step
   `dt.min(next_boundary − t)`. Output and intervention times hit **exactly**.
2. **Inference filters — step-to-obs.** `particle_filter.rs:231-244`, `if2.rs`
   (two sites), `correlated_pf.rs` —
   `while t_local < obs_time { step_dt =
   dt.min(obs_time − t_local) }`. Four
   copies. Observations hit **exactly**; interventions applied inside `step_one`
   via rounded `fire_steps`.
3. **PGAS — uniform grid.** `pgas.rs` stores
   `Trajectory { substeps:
   Vec<SubstepRecord> }`, one record per uniform `dt`
   step, mapping obs to substeps by rounding (`build_obs_at_substep:261` →
   `interval_steps`). The only path that **snaps** observations, and the only
   one whose density paths reconstruct time as `t = t_start + s*dt`
   (`pgas.rs:568,606`, `pgas_grad.rs:397`).

So two boundary policies already coexist _implicitly_: the PF lands exactly,
PGAS snaps. This proposal makes that a deliberate, uniform, named choice.

### Observation: scoring unified, loading scattered

Scoring is one seam: `ObservationModel<S>` (`traits.rs:89`), its required method
`log_likelihood(&self, state: &S, obs_idx, params) -> f64` (`:94`) called by all
four algorithms. Implementor: `MultiStreamObsModel` (`multi_stream_obs.rs:246`);
projection ADT `StreamProjection = FlowSum | IntCompSum | Expr` (`:72`), with
`resets_after_observation()` true only for `FlowSum` (`:87`). Loading is
duplicated across `pfilter.rs`, `profile.rs`, `fit/runner.rs`, `survey.rs`.
Observation scoring reads state (`&S`), but an `Interval` observation _drives_ a
state mutation — `state.reset_flows()` (`particle_filter.rs:401`) — through the
algorithm loop. That reset is a real write, and the design represents it.

### Interventions, events, balance: shared schedule, separate constraint, fused ordering

- `Intervention { schedule, actions, always_active }`
  (`ir/src/intervention.rs:70`); `always_active = true` is an **event** (every
  substep), `false` a scheduled **intervention**. Shared
  `InterventionSchedule = AtTimes | AtTimesExpr |
  Recurring` (`:17-29`,
  `AtTimesExpr` = gh#69 parametric fire times) and
  `Action
  = FractionTransfer | AbsoluteTransfer | Set | Add` (`:59-66`).
- **Balance** is a `ResolvedBalance` on `CompiledModel`
  (`compiled_model.rs:406`), a structural constraint overwriting one target
  compartment.
- Within a substep the order is fixed and semantic
  (`chain_binomial.rs::step_one`): transition deltas and event deltas are
  **computed from the start-of-step snapshot and applied atomically together**
  (`:424-433`), _then_ interventions on the post-transition state (`:489`),
  _then_ balance last (`:503`, target exempt from the negative-count check).
  None consumes RNG.
- Gillespie has a special obligation (spec §2.3.1, `gillespie.rs:174`): after
  any state mutation, recompute all propensities and draw a fresh exponential —
  it cannot carry remaining exponential time across a mutation.

### Reactive: proposal only

`docs/dev/proposals/2026-05-14-reactive-interventions-and-evsi.md` specifies a
`reactive_interventions {}` block (state-condition trigger). Nothing is
implemented; `InterventionSchedule` has no state-conditioned variant.

## dt-dependence vs dt-independence

The organizing axis for everything below: whether a backend's result depends on
the _step size_.

- **dt-dependent (stochastic, fixed-step):** chain-binomial, tau-leap. A step
  over interval `h` draws `Binom(N, 1−e^{−λh})` / a Poisson, **freezing the rate
  `λ` at the start-of-step value**. For a state-dependent rate (`λ_SI = βI/N`),
  two steps of `h/2` re-evaluate `λ` at the midpoint — a _finer_, more accurate
  approximation. So the realized trajectory distribution is a function of where
  the step boundaries fall. This is not error vs. correctness; both converge to
  the exact process as `dt → 0` (O(dt) rate-freezing difference).
- **dt-independent:** Gillespie (exact SSA, event-driven, no discretization
  error — indifferent to where you stop) and ODE (deterministic — integrates to
  whatever time you ask, no noise).

The consequence runs through the whole design: **landing exactly on an off-grid
boundary changes the result only for dt-dependent backends.** Gillespie and ODE
land exactly for free. This is precisely why the `snap | exact` choice matters
for chain-binomial/tau-leap and is a no-op for Gillespie/ODE, and why the PGAS
exactness migration (which would introduce non-uniform substeps under a
dt-dependent kernel) is the one genuinely delicate piece.

## The unified model

### Process stays special

The process alone advances the true latent state, and alone consumes randomness
to do so. That specialness is real: the alloc-free kernel, the RNG draw
_ordering_ (paired-seed common-random-number coupling and the `gamma_used` /
`binomial_z` hooks PGAS and the correlated PF depend on it), and — for gradient
inference — the transition density and its derivative. What is _not_ special is
the loop around the kernel; that consolidates. The kernel `step_one` is already
singular.

A single fixed-step `step(dt)` contract subsumes chain-binomial, tau-leap, and
ODE (ODE steps in `dt` with RK4 sub-stepping; it ignores `rng` because it is
deterministic). It does **not** subsume Gillespie (event-driven). All four share
the `Schedule`; Gillespie's _kernel_ differs (it proposes the next time; the
schedule can only clip it — see below).

### Everything else is a typed timeline of triggered effects

Observation, intervention, event, balance, reactive intervention are effects at
points on the timeline, differing along three axes that the design makes
**types, not conventions**, because each is where a generic "effect" would leak:

1. **Trigger** — when it fires, and what inference contracts that firing
   satisfies.
2. **Relation to state** — read (observation), mutate (intervention/event/
   reactive), or constrain (balance). The read/write split is type-enforced.
3. **Lifecycle stage** — where in the substep it applies, and what state it
   reads.

### The canonical substep lifecycle

The within-substep order is a first-class, documented object — the analogue of
SLiM's published tick/generation cycle, whose defining virtue is that a modeller
can reason precisely about _when_ their script runs relative to reproduction and
selection (Haller & Messer 2019, _MBE_ 36:632; the SLiM manual's lifecycle
diagrams). camdl's substep lifecycle, matching `chain_binomial`'s `step_one`
(forward `tau_leap`/`ode` currently _invert_ event vs intervention —
canonicalizing them is a Stage-2 behaviour change, see the cross-backend leak
note):

```
┌─ start of substep: snapshot x_t ───────────────────────────────┐
│  1. PROPOSE    transition draws (rates frozen at x_t)           │
│                event deltas (computed from the x_t snapshot)    │
│  2. ADVANCE    apply transition + event deltas ATOMICALLY → x'  │  fused — one stage
│  3. INTERVENE  apply scheduled interventions on x' (current)    │
│  4. BALANCE    enforce conservation (last; target exempt)       │
│  5. OBSERVE    read projection of post-effect state; score/emit │  read-only
│  6. RESET      if an Interval obs fired here: zero THAT          │  represented
│                stream's flow accumulators                       │  write
└─ end of substep: x_{t+dt} ─────────────────────────────────────┘
```

Two corrections this canonization bakes in. "Events read the snapshot" is a
property of **stage 1** (the delta is _computed_ from `x_t`), not a separate
later phase — transitions and events apply _together_ in stage 2 (a single,
fused stage, not `Transition < Event`). And the accumulator reset is **stage
6**, a represented per-stream write, not a hidden side effect of an observation.
This lifecycle belongs in user-facing docs (the language spec / user-features)
with a polished figure; it is how a modeller reasons about "does my intervention
see my event." It ships as its own small documentation PR alongside Stage 0 — it
canonizes `step_one` as it already is (zero unification risk) and fixes the
contract everything else refactors against.

## The concrete types

A design sketch (proposed). Names indicative.

### The timeline spine

```rust
/// A point on the timeline and what is due there. Several kinds can coincide.
pub enum Boundary {
    Substep,                 // an internal dt step: process advances only
    Output(usize),           // forward only: record a snapshot
    Observation(usize),      // inference scores here; forward emits here
    Effect(EffectId),        // a scheduled Mutate/Constrain fires here
}

/// Merged sorted boundary timeline. INVARIANT (the actual contract the spine
/// sells, and the one place a regression breaks CRN *silently*): `Schedule: Sync`
/// immutable; the per-particle `Cursor: Copy`; `next_boundary` is a PURE function
/// of (Schedule, cursor, t) with no interior mutability — so N particles in the
/// parallel swarm hit IDENTICALLY ordered boundaries (paired-seed/CRN coupling
/// depends on this; a shared-mutable cursor would corrupt it without failing any
/// on-grid golden). Pinned by a proptest: N independent cursors over one Schedule
/// yield byte-identical boundary sequences. The snap-vs-exact grid is an explicit
/// FIELD here, the single source of truth for every time→step mapping — NOT a
/// per-call-site convention (see the snap-grid leak note below).
pub struct Schedule { /* immutable dt-grid + sorted obs/effect cursors + snap policy */ }

impl Schedule {
    /// Fixed-step drivers: the next boundary at or after t, and what is due.
    /// Coincident kinds at one time are returned together; the driver applies
    /// them in lifecycle order (Effect/Constrain before Observation).
    pub fn next_boundary(&self, cursor: &mut Cursor, t: f64) -> (f64, SmallVec<[Boundary; 4]>);

    /// Event-driven (Gillespie): the process PROPOSES t_proposed; the schedule
    /// can only clip it to the nearest earlier boundary (or pass it through).
    pub fn clip(&self, cursor: &Cursor, t_proposed: f64) -> ClipResult;
}
```

Two distinct entry points because the spine genuinely forks: fixed-step drivers
_ask_ the schedule for the next time; Gillespie _proposes_ a time and the
schedule clips it. Both share the boundary set and cursor; only the query
differs. `t_end` is the schedule's terminal boundary; effect/obs times beyond it
are a load error; an empty schedule yields `(t_end, [])` once and terminates.

### Triggers and capabilities

```rust
pub enum Trigger {
    AtTimes(Vec<f64>),              // intervention; AtTimesExpr resolved once → AtTimes
    Recurring(RecurringSchedule),
    EverySubstep,                   // event, balance — fires every Substep boundary
    StateCondition(ResolvedExpr),   // reactive — evaluated every substep, forward only
    ObservationTime,                // observation
}

/// The inference contracts an effect satisfies. Computed from the effect's
/// ACTUAL expressions, not the trigger variant.
pub struct EffectCaps { pub differentiable: bool, pub markov: bool }

/// Runs Rust-side at driver construction (after `estimated` is resolved — it is
/// a fit-time set, not an IR field). A dedicated SMOOTHNESS predicate, NOT a
/// `differentiate` call: `differentiate` silently returns `Const 0.0` for
/// Floor/Ceil/comparison ops and leaves Cond predicates undifferentiated
/// (autodiff.ml:117,161-162,186) — so a param entering only through a Cond
/// predicate or Floor would pass as differentiable with a wrong ZERO gradient.
/// Sound rule: differentiable = false iff collect_param_refs(expr) ∩ estimated
/// ≠ ∅ (the reachability half — it DOES see Cond.pred / TableLookup,
/// pgas.rs:33-64) AND the expr contains any non-smooth node
/// (Mod | Floor | Ceil | comparison | Cond-pred). StateCondition forces {false,false}.
fn effect_caps(effect: &Effect, estimated: &ParamSet) -> EffectCaps;
```

### The lifecycle stage and the read/write split — two orthogonal axes

**Stage** (when in the substep) is the full 6-valued lifecycle order carried by
_every_ effect — it is the sort key the driver uses. **Relation to state** (read
/ mutate / constrain) is the method signature — it is the read/write the type
system enforces. Keeping them separate is what the round-1 review's "lifecycle
collapse" leak required: a 3-valued `Stage` that omitted `Observe` and `Reset`
could not order all six steps.

```rust
/// The full substep lifecycle as a total order. EVERY effect carries one.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage { Advance, Intervene, Balance, Observe, Reset }
//   txn+event FUSED = Advance  <  Intervene  <  Balance  <  Observe  <  Reset

/// READ — observation (Stage::Observe). Gets &State, never &mut.
pub struct Observe { pub trigger: Trigger, pub projection: StreamProjection,
    pub kind: TemporalKind, pub likelihood: ResolvedLikelihood,
    fn project(&self, state: &State, t: f64) -> f64; }

/// EVENT — Stage::Advance. A fused delta CONTRIBUTOR, not an independent &mut:
/// its delta is computed from the start-of-step snapshot and applied atomically
/// with the transition draws (chain_binomial.rs:424-433). Hence propose_delta,
/// not apply — an apply(&mut State) could not be atomic with transitions.
pub struct Event { pub trigger: Trigger /*EverySubstep*/, pub actions: Vec<Action>,
    fn propose_delta(&self, snapshot: &State, t: f64) -> Deltas; }

/// INTERVENTION — Stage::Intervene. Applied to the post-transition current
/// state, after the fused Advance. A genuine &mut. No RNG.
pub struct Intervene { pub trigger: Trigger, pub actions: Vec<Action>,
    fn apply(&self, state: &mut State, t: f64); }

/// CONSTRAIN — balance (Stage::Balance), last among writes; target exempt from
/// the negative-count check.
pub struct Constrain { pub target: usize, pub expr: ResolvedExpr,
    fn enforce(&self, state: &mut State, t: f64); }

/// RESET — Stage::Reset. The Interval flow-accumulator window close, keyed to the
/// firing stream's flow indices (NOT global) — the §5.2.1 per-stream fix.
pub struct ResetWindow { pub flow_indices: Vec<usize>,
    fn reset(&self, state: &mut State); }
```

The event/intervention split is _forced_ by the fusion: an event contributes a
delta applied atomically with transitions, so its read-source (the snapshot) is
in the type via `propose_delta`, while an intervention's is the current state
via `apply` — not a convention keyed on a `stage` field. `Action::Set/Add`
carrying a `StreamProjection` must preserve the documented hash-position
contract (`observation.rs:12-16` — variant index is permanent; reshaping churns
every stored `run_id`).

### The drivers — generate, filter, trajectory-match

All consume the same compiled model (kernel + effects + schedule), diverging
only per-boundary. Two ship here — `run_forward` and `run_filter`;
`run_trajmatch` is a _reserved seam_ sketched to show the substrate generalizes
to deterministic inference (its full design is the deferred ODE-inference
proposal), not shipped architecture.

```rust
pub fn run_forward(model: &Compiled, params: &[f64], seed: u64, cfg: &SimConfig)
    -> Trajectory;          // GENERATE: emit (RNG) + record; all triggers incl. reactive

pub fn run_filter<P: ProcessModel>(            // EVALUATE (stochastic): score, no RNG
    process: &P, obs: &dyn ObservationModel<P::State>,
    effects: &CompiledEffects, schedule: &Schedule, /* particles … */) -> Loglik;
    // PF/IF2/PMMH need ProcessModel; PGAS additionally requires P: DensityProcess
    // and records SubstepRecord (carrying the realized (t0, dt_substep), below).

pub fn run_trajmatch(                           // EVALUATE (deterministic): integrate once
    ode: &OdeProcess, obs: &dyn ObservationModel<OdeState>,
    effects: &CompiledEffects, schedule: &Schedule, params: &[f64]) -> (Loglik, Grad);
    // [future] grad via forward sensitivity ODE; FlowSum-over-flow_accumulators
    // does NOT exist for OdeState — the trait is shared, the projection impl is not.
```

**Why they diverge** (forced, not arbitrary): allocation (forward records,
filter is alloc-free); generate-vs-evaluate (forward samples `y ~ p(y|x)` with
RNG, inference scores `log p(y|x)` without); density+gradient (PGAS needs
`DensityProcess`); RNG ordering / CRN (filter pins draw order); capability
gating (reactive rejected from gradient/PGAS); reference trajectory (PGAS
records and conditions on one). The trajectory-match driver adds two more from
determinism: the filter degenerates (integrate once, no resampling of identical
particles) and the gradient comes from the forward sensitivity equations, not
`rate_grad`. Same schedule, same effects, same scoring seam — different ways to
turn `θ` into a likelihood. The full ODE-inference design is its own proposal;
here we only reserve the driver seam (a deterministic `ProcessModel` + a
non-filter driver), and note one normative constraint: a `θ`-dependent
`Set`/`Constrain` at a boundary discontinuously reseeds `∂x/∂θ`, so
trajectory-match must capability-gate those until a sensitivity-jump rule is
implemented.

## The `snap | exact` boundary policy

The implicit per-algorithm inconsistency (PF exact, PGAS snap) becomes one
explicit, uniform, user-controllable option: `--obs-alignment snap | exact`
(bundled in `fit.toml`).

- **`snap`** — observation/effect times are rounded to the integrator grid
  (today's PGAS behaviour; for the PF, round the obs time before stepping).
  Keeps a single uniform `dt`, fully reproducible, no per-substep bookkeeping.
- **`exact`** — the integrator lands exactly on each boundary (today's bootstrap
  PF behaviour, generalized: tile each inter-boundary window with its own
  uniform sub-grid of `⌈W/dt⌉` equal steps, so boundaries are hit with no tiny
  remainder step). Lossless timing.

For **dt-independent** backends (Gillespie, ODE) the two coincide — there is no
noise to perturb. (Caveat: Gillespie is dt-independent only for
time-_homogeneous_ rates; gh#95 is exactly the current implementation's
inhomogeneous-rate bias, so "Gillespie lands exactly for free" is an
idealization the code does not yet meet — do not lean on it as a clean invariant
until gh#95 is fixed.) For **dt-dependent** backends (chain-binomial, tau-leap)
they differ at finite `dt` (the rate-freezing granularity changes) and converge
as `dt → 0`; `exact` is the more accurate, `snap` the more reproducible.

**The gate, and the consolidation it forces.** `exact` is **not** a
`Capabilities` bitflag. The existing `Capabilities` (`sim/src/lib.rs`) is a
_model × backend_ axis (`required_capabilities()` scans the IR; each forward
backend's `capabilities()` declares support), but `obs-alignment` is a
_run-option × algorithm_ axis — PGAS is an algorithm, has no `capabilities()`,
and never passes through `Simulate`. And the alignment-relevant gating today is
_two_ separate call sites — `util.rs:1699` (forward) and
`check_model_capabilities` (`fit/methods.rs`, a hard-coded
`match backend { … }`) — neither seeing algorithm identity. So the gate is a new
`(algorithm, obs_alignment)` support check at the fit-dispatch seam, and
**consolidating those two existing gates into it is part of this work** —
otherwise "one clean error / one place" is false today. With that done,
`exact` + PGAS hits one clean error ("not implemented: PGAS supports `snap`
only; use `--obs-alignment snap`, or `algorithm = if2|pfilter` for `exact`"),
and the test asserting it is a positive test that routing is consolidated to a
single seam. (Verify PGAS genuinely _lacks_ the `exact` capability — absent, not
defaulted-true.)

## Staging and default policy

Sequencing is **oracle-first, then gh#175, then extract** — you cannot prove a
refactor byte-identical against a baseline that never ran the hard case, and you
cannot trust a PGAS-touching change while the PGAS gradient is broken. The
default is conservative and only flips after evidence.

**Stage 0 — build the comparison oracle (FIRST, before any extraction).** The
existing forward ratchet (`gate_trajectory_baseline.rs`: per model × backend ×
`SEED=42`, FNV-hash the trajectory vs a committed table) is the right _shape_
but covers only forward simulation on an all-on-grid corpus. Stage 0 closes the
gap:

- **Corner-case fixtures** (`tests/fixtures/` → `ocaml/golden/`): an off-grid
  observation (e.g. `t = 7.3`), a coincident observation+intervention, a
  `θ`-dependent `set()` at a fractional time, an irregular multi-cadence stream,
  and a fractional `output.end` (the `seir_vaccine_seasonal` 1095.7275 case).
- **Forward baselines** — extend `gate_trajectory_baseline.rs`'s `BASELINES`
  (captured from _current_ code on this machine, per its platform caveat).
- **Inference baselines (the missing piece)** — a new
  `gate_inference_baseline.rs`: per model × algorithm (PF/IF2/PMMH/PGAS) ×
  `SEED=42`, pin the marginal loglik _and_ the per-observation scored
  contributions, from current code. This is the actual oracle, since the
  refactor rewrites the inference loops; the existing ratchet only covers
  forward `Simulate`.
- **RNG draw-sequence baseline** — a harness logging (kind, count, order) of
  draws per run, so an inserted/reordered draw fails loudly.
- **Runtime collision-guard test** — feed two distinct sub-`dt` obs times and
  assert the _runtime_ hard error (not just a proptest generator constraint).

None of this touches production code; it is the ratchet everything else
refactors against, and the gating deliverable — not a testing footnote. The
canonical substep-lifecycle doc/figure ships here too (zero-risk; it fixes the
contract).

**Stage 1 — spine, byte-identical.** Extract the `Schedule`, route the forward
backends and the four filter loops through it, install the canonical substep
lifecycle and the corrected effect types. Each path keeps its _current_ boundary
behaviour (PF exact, PGAS snap, interventions as today). Strictly byte-identical
against the Stage-0 oracle (forward, inference, and RNG-sequence baselines),
including the off-grid and coincident-boundary fixtures. This is loop
consolidation and the bug-surface win — it _names_ the PF/PGAS divergence and
deletes the duplicated loops, but does not yet _close_ the divergence (that is
Stage 2).

**Stage 2 — expose the knob; default to the most-accurate alignment each
algorithm supports.** Add `--obs-alignment exact | snap` (bundled in
`fit.toml`). The default is **`exact` where the algorithm supports it**. The
support matrix is NOT uniform — three classes, not "PF-family vs PGAS":

- **bootstrap PF / IF2**: exact on any obs (on- or off-grid) — they hold no
  per-window substep-count assumption. Default exact, byte-identical today.
- **PMMH**: rho-dependent. Plain PMMH (`rho` unset) is the bootstrap PF → exact
  on any obs. **Correlated** PMMH (`rho` set → CPM) is exact only on **uniform
  ON-GRID** obs: its pre-drawn-noise array is sized by a _scalar_
  `steps_per_obs = interval_steps(0, obs_dt, dt)` and indexed
  `i*steps_per_obs + substep` (`correlated_pf.rs:185,332`); off-grid obs make
  the realized substep count exceed `steps_per_obs`, so the index overruns into
  the next particle's block or trips the `< len` guard and **silently falls back
  to fresh RNG** — decorrelating the estimator (the leak below). So the gate
  must classify PMMH as exact-on-grid-or-snap, _not_ lump it with the bootstrap
  PF; `--obs-alignment
  exact` + PMMH + off-grid obs is a clean capability
  error, not a silent wrong answer. (A real off-grid PMMH needs the
  variable-substep-count noise layout — Stage 3.)
- **PGAS**: `snap` only, falls back to `snap`; `--obs-alignment exact` + PGAS is
  the clean "not implemented" error.

Nothing regresses (every algorithm's _current, validated_ regime keeps its
behaviour byte-for-byte). The `(algorithm, obs_alignment)` capability gate lands
here — **consolidating today's two scattered checks** (the forward gate in
`util.rs` and `check_model_capabilities` in `fit/methods.rs`, a hard-coded
`match backend`) into one fit-dispatch seam; otherwise "one clean error / one
place" is false. (Audit the corpus for hidden fractional times so the
distinction is exercised, e.g. `seir_vaccine_seasonal`'s
`output.end = 1095.7275`.) This default rule is chosen so that when exact-PGAS
lands (Stage 3) the default becomes **uniform exact** with no policy change —
the fallback simply disappears.

**Stage 3 — exact everywhere (the committed end-state), evidence-gated.**
Implement **exact-PGAS**: move the uniform-grid assumption out of PGAS so each
`SubstepRecord` carries its realized `(t0, dt_substep)` and **no path recomputes
`s*dt`** — the eight reconstruction sites converted together
(`pgas.rs:568,605,716,869,1079`, `pgas_grad.rs:397`, and the `interval_steps`
obs mapping at `pgas.rs:268,704`); a single missed site silently reconstructs
the wrong time → wrong rate freeze → wrong density. With exact-PGAS landed _and
validated by the evidence below_, PGAS's `snap` fallback is removed and the
"exact where supported" default becomes **uniform exact** — the clean end-state.
The validation uses **non-hierarchical** PGAS (which mixes today); gh#175 blocks
only _hierarchical_-model exact-PGAS, a smaller subset deferred with it, so it
does not gate this.

**The evidence that earns the default (Stage 3 gate).** A study, not a vibe,
exact vs snap across the model-feature matrix:

- **Recovery** — plant θ → simulate → fit under both alignments; both must
  recover θ within the MC bracket, on **SIR, an off-grid-obs model, an
  intervention model, an event model, a seasonal (time-inhomogeneous) model, and
  an OVERDISPERSED off-grid model**. The overdispersed case is non-negotiable:
  the gamma multiplier (`shape = dt/σ²`, `scale = σ²/dt`) is the single most
  `dt_substep`-sensitive density term, and it only exists for
  `overdispersed(...)` transitions — without it the density/gradient checks
  below test the fragile path _vacuously_. Exact should match or beat snap on
  the off-grid/seasonal cases (where snap's rounding bites) and tie elsewhere.
- **exact-PGAS density correctness** — on a fixture whose terminal substep is
  genuinely shortened (`rec.dt_substep ≈ 0.9125 ≠ 1.0`, NOT an on-grid window
  where it is vacuous), the per-substep transition density equals a from-scratch
  recompute using `rec.dt_substep` (never `s*dt`), comparing density _values_
  (not the `(counts, flows, gammas)` tuples). Pin
  `gate_pgas_density_baseline`-style; the overdispersed off-grid model exercises
  the gamma term here.
- **exact-PGAS gradient correctness** — a **finite-difference check** on the
  NUTS gradient under the shortened substep
  (`|∂L/∂θ_analytic − ∂L/∂θ_FD| < tol`): recovery cannot catch a wrong gradient
  because a wrong gradient mixes and lies.
- **On-grid parity** — exact-PGAS on an on-grid model reproduces the snap result
  byte-for-byte (where exact == snap), proving the migration left the common
  case untouched.

This is the "exact earned, not asserted" gate: only after all four does `exact`
become the default.

## Consolidation: substrate, not algorithms

The consolidation reduces cross-backend bug surface at the **substrate** layer —
the schedule, the substep lifecycle, effect application, the kernel — which is
shared by _all_ drivers, including the particle filter and PGAS. It does **not**
merge the algorithms: bootstrap filtering and conditional-SMC-with-ancestor-
sampling (the reference trajectory, the density, the gradient) are genuinely
different and stay distinct above the substrate. So PF and PGAS share when/how
the timeline advances and effects apply (the bug-prone part), and keep their own
inference logic (the part that should stay separate). In this push PGAS keeps
its uniform grid (it honours `snap` only); its exactness migration is the
deferred increment, so the substrate consolidation lands without touching the
delicate reference-trajectory path.

Honest accounting of "surface": deleted are the four step-to-obs loops and the
three hand-rolled forward boundary cursors. Added are the `Schedule`/`Boundary`/
`Trigger`/`Effect` types and the driver trio; the existing IR types are kept and
_mapped_, not removed. The win is consolidating control flow into one typed
spine, not reducing the type count — and that is where the cross-backend bugs
live.

### Filter architecture: existing vs proposed

The trait spine is already well-factored and is **kept**: `ProcessModel` (the
kernel `step`), `ObservationModel` (`log_likelihood`/`obs_time`),
`DensityProcess:
ProcessModel` (`log_transition_density`, PGAS only),
`Resettable`; plus the shared types `ParticleState` / `ParticleSwarm` /
`systematic_resample` / `log_sum_exp`. What is _not_ consolidated today is the
**filter loop**: four functions each hand-roll the same per-observation
structure.

Existing (each a separate `propagate → weight → resample` loop;
`Schedule.substep` just routed under each this session, but the loops are still
four):

```
bootstrap_filter             SIS; systematic resample; per-particle death-on-recoverable-error
bootstrap_filter_correlated  = bootstrap, but PROPAGATE injects pre-drawn correlated noise (CPM, for PMMH)
run_if2                      outer iteration loop: perturb θ per-obs + cool, running a full filter each iteration
run_pgas  (the sweep)        CONDITIONAL PF: particle 0 = reference trajectory; ANCESTOR resample (not systematic);
                             needs the transition DENSITY (ancestor weights + NUTS gradient); RECORDS the path;
                             uniform s*dt grid (snap)
```

Proposed — consolidate the genuinely-shared **spine**, keep the divergent
**bodies** per-driver. The shared unit is a substep **iterator**, not a
parameterised filter function:

```
schedule.substeps(cursor, t_start) -> Iterator<Item = (t_local, step_dt)>   ← the ONE shared primitive (all four)
        yields the inner substep walk: termination at the next obs boundary
        + Schedule.substep + the t_local advance. Alignment (exact|snap) is the
        Schedule's, so it threads to every caller at once.

each driver keeps its OWN loop body over the iterator, because the bodies differ for real:
  bootstrap PF    for (t,dt) in substeps { step; recoverable err -> mark DEAD }        + systematic resample
  correlated PF   for (t,dt) in substeps { inject PreDrawn noise; step; propagate err } + systematic resample
  if2             the bootstrap body, WRAPPED in  for iter { …; perturb θ; cool }       // optimiser over the filter
  run_pgas_sweep  for (t,dt) in substeps { conditional step }  + ANCESTOR resample + density + record
```

Where consolidation **stops on purpose** (further merging would leak): the
iterator shares the spine that is genuinely identical; the bodies are **not**
folded into a `propagate_window(policy, noise, conditional)` or a
`run_filter(strategy)`. The death-policy (mark-dead vs propagate), the
pre-drawn-noise injection, and PGAS's conditioning/ancestor/density/record are
_real_ differences — absorbing them into toggle params is exactly the leaky
god-function. So the consolidation is: the shared **iterator** (the spine) + the
already-shared trait spine
(`ProcessModel`/`ObservationModel`/`DensityProcess`) + the shared helpers
(`systematic_resample`, `log_sum_exp`); the four bodies stay distinct and
honest. (A `run_filter` that wraps the _outer_ obs-loop for the three
non-conditional filters is a candidate further step, but only if it composes
_without_ death/noise toggles — decide at implementation; the iterator is the
part we're committing to now.) That is "a family of reused functions that can't
merge further without leaking is the right answer", applied — the natural seam
is the iterator.

## Leaky abstractions the types must honour

- **Read/write erasure** — prevented by `Observe` (`&State`) vs `Mutate`
  (`&mut`); the accumulator reset is a represented `ResetWindow`, not hidden.
- **Lifecycle collapse** — prevented by the canonical lifecycle with the
  transition+event _fusion_ modelled honestly and the snapshot read at stage 1.
- **RNG reordering / insertion** — `sample` (forward, RNG) stays out of scoring;
  and a variable last step _inserts_ a draw, so the RNG invariant is on draw
  _count and order_, exercised by an off-grid corpus (below). This is why `snap`
  (no inserted draws) is the reproducible default.
- **Interval/Instant + reset** — `TemporalKind` first-class; reset per-stream by
  flow index (stage 6), not global.
- **Gillespie propensity invalidation** — a per-backend hook the driver calls
  after applying `Mutate`s; expressed, not assumed away. And the schedule `clip`
  query (not `next_boundary`) for the event-driven kernel.
- **Off-grid under a dt-dependent kernel** — handled by the `snap | exact`
  policy and the staging, never silently: `snap` is byte-identical on-grid;
  `exact` is a validated behaviour change, not a refactor.
- **The snap grid must be ONE field, not three disagreeing call-site conventions
  (the leak the spine exists to seal, and the one this list previously
  missed).** Today interventions snap via `fire_steps` at `cfg.dt` (forward
  chain/tau/ode), via `iv_resolution_dt = model.simulation.dt.unwrap_or(1.0)`
  (Gillespie — a _phantom_ grid the integrator never walks, so Gillespie's
  interventions and observations snap to _different_ grids on one run), and obs
  via `interval_steps` (PGAS, `pgas.rs:268`). The merged `Schedule` owns the
  snap grid explicitly: the realized `(t0, dt_substep)` record is the single
  source `time_to_step` / `interval_steps` / `resolve_fire_steps` all read, and
  a dt-independent backend snaps interventions to the boundary set it clips obs
  to (kill the `unwrap_or(1.0)`). Oracle fixture: off-grid intervention +
  on-grid obs on Gillespie, asserting the exact fire time — exposes the
  disagreement today.
- **The realized `dt_substep` threads into the density _magnitude_, not only the
  time.** `p = 1 − exp(−rate·dt)`, the gamma/overdispersion density
  (`shape = dt/σ²`, `scale = σ²/dt`, so `Var = σ²/dt` is dt-dependent,
  `pgas.rs:629`), and the gradient (`∂/∂θ` through `rate·dt`) all consume `dt`
  as a magnitude. exact-PGAS must use `rec.dt_substep` in _every_
  density/gradient call, or a 0.9125-vs-1.0 substep gives a finite-but-wrong
  density → a silently shifted posterior. The byte-identical record check must
  compare density _values_, not just the `(counts, flows, gammas)` tuples, or it
  passes vacuously.
- **Coincident-boundary order is non-canonical across backends today** —
  `chain_binomial` fuses events with transitions then applies interventions, but
  forward `tau_leap`/`ode` apply interventions _then_ events
  (`apply_interventions_at` then `apply_events_at`). The `Stage` order
  canonicalizes this, but rewriting tau/ode to the canonical order is a
  _behaviour change_, so it moves to **Stage 2** (with a re-baseline), NOT
  Stage-1 byte-identical. "Matches `step_one` exactly" holds only for
  chain-binomial. **The re-baseline is not self-validating**: it pins the new
  numbers as truth but does not show the canonical (chain-binomial) order is the
  _correct_ one for tau/ode. Stage 2 needs a hand-computed fixture asserting the
  _intended_ order (a `θ`-dependent intervention coincident with an event,
  checked against the spec's substep lifecycle), not just a fresh hash.
- **The correlated-PF (CPM) pre-drawn-noise layout assumes a fixed substep count
  per window.** The noise arrays are sized by a _scalar_
  `steps_per_obs =
  interval_steps(0, obs_dt, dt)` and indexed
  `i*steps_per_obs + substep` (`correlated_pf.rs:185,332`); the
  `if noise_idx < len` guard (`:333,345`) _silently_ falls back to fresh RNG
  when the realized substep count exceeds `steps_per_obs` — which `exact` +
  off-grid obs causes (8 substeps to land on 7.3 at `dt=1` vs
  `steps_per_obs = 7`). Silent decorrelation of the CPM estimator is the
  camdl-bar leak. Honor it two ways: (1) the capability gate (above) keeps PMMH
  out of the off-grid-`exact` class; (2) the guard becomes a hard error, not a
  silent fallback. A true off-grid PMMH needs the noise array sized from the
  _realized_ per-window substep count (Stage 3), not a precomputed scalar.

## How the pieces relate and flow

Existing types kept and mapped into new effects; one compiled model feeds three
drivers:

```
 EXISTING (kept)                     NEW (this proposal)
 ───────────────                     ───────────────────────────────
 step_one ............ kernel        Schedule ......... merged timeline
 ParticleState (i64 counts)          Boundary = Substep|Output|Obs|Effect
 OdeState      (f64)                 Trigger  = AtTimes|Recurring|EverySubstep
 DensityProcess (PGAS only)                     |StateCondition|ObservationTime
 ObservationModel / log_likelihood   EffectCaps{ differentiable, markov }
                                     Stage = Advance(txn+event) < Intervene < Balance
                                     snap | exact  (obs-alignment policy)

   existing IR types        ── map ──►   new effect types
   StreamProjection, Likelihood ──────►  Observe   (Read,  &state)
   Intervention/Action/Schedule ──────►  Mutate    (&mut state, Stage)
   ResolvedBalance ───────────────────►  Constrain (structural, last)
   (FlowSum reset, today global) ─────►  ResetWindow (per-stream, stage 6)

                    ┌──────────────────────────────────┐
                    │          COMPILED MODEL           │
                    │  kernel + [effects] + Schedule    │
                    └──────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                         ▼
     run_forward              run_filter               run_trajmatch [future]
     GENERATE                 FILTER (stochastic)      INTEGRATE (determ.)
     emit + record            score + resample         integrate + score
     all triggers (reactive)  PF/IF2/PMMH/PGAS         NLopt / MH / NUTS
     chain/tau/ode/gillespie  chain-binomial           ODE
                              ParticleState            OdeState + ∂x/∂θ
                              grad: rate_grad          grad: sensitivity ODE
```

Trait/type view — only new structs, no new traits; drivers are free functions
bounded by the existing traits:

```
 TRAITS (existing)                       implemented by
 ProcessModel : Send+Sync                ChainBinomialProcess (State=ParticleState)
   type State : Clone+Send+Resettable    OdeProcess           (State=OdeState) [future]
   fn step(&mut State, θ, t, dt, rng, scratch)   ← shared kernel (step_one|integrate)
 DensityProcess : ProcessModel           ChainBinomialProcess only  (PGAS / gradient)
 ObservationModel<S> : Send+Sync         MultiStreamObsModel
   fn log_likelihood(&self,&S,i,θ)->f64  ← scoring seam, READ-ONLY &S

 NEW (structs; signatures enforce read vs write)
 Observe{trigger,projection,kind,likelihood}   project(&self,&State,t)->f64
 Mutate{trigger,stage,actions}                 apply(&self,&mut State,&State,t)
 Constrain{target,expr}                        enforce(&self,&mut State,t)
 ResetWindow{flow_indices}                     reset(&self,&mut State)
 Schedule  next_boundary(&self,&mut Cursor,t)->(f64,[Boundary]) ; clip(&self,&Cursor,t)->…
 Trigger / EffectCaps / Stage / Boundary
```

## Testing

Correctness-critical, refactor-heavy code. The spine is a refactor (parity is
the spine); the `exact` policy and the deferred PGAS migration are behaviour
changes (external oracles). The discipline is asymmetric to match.

### Parity is the spine — and the corpus must be off-grid

- **Old path live behind a flag** (the `CAMDL_EVAL_UNRESOLVED`
  differential-oracle pattern); the new path matches byte-for-byte on a corpus
  before the old path is deleted.
- **The corpus MUST include an off-grid observation and a `θ`-dependent
  effect.** On-grid goldens cannot exercise the short-substep / inserted-draw
  hazards, so an on-grid-only corpus passes _vacuously_. This is the single most
  important testing requirement.
- **RNG draw order _and count_ invariant.** A harness logging the draw sequence
  asserts it is identical old-vs-new; a variable step that _inserts_ a draw
  fails here. Under `snap` (Stage 1/2) the count is invariant by construction.

### Per-stage gates

| stage                          | what changes              | gate                                                                                               |
| ------------------------------ | ------------------------- | -------------------------------------------------------------------------------------------------- |
| 1. spine                       | refactor                  | full golden corpus (incl. off-grid) **byte-identical**; RNG count+order invariant; CRN preserved   |
| 2. `snap` knob + default       | none on-grid              | on-grid identical; off-grid PF pinned to `exact` reproduces its prior result; capability-gate test |
| `exact` for PF/forward         | behaviour (off-grid only) | off-grid: validated against the Richardson `dt → 0` ladder converging to the same limit            |
| exact-PGAS [deferred, Stage 3] | behaviour (silent path)   | external-oracle battery below; gated on gh#175 mixing                                              |

### Cross-cutting invariants

- **Schedule (proptest):** every obs time is a boundary hit exactly; two
  _distinct_ times within `dt` collide → **runtime hard error** (not a generator
  invariant — feed two sub-`dt`-separated obs and assert the error); the merged
  sequence is sorted/monotone; on-grid substep counts match `interval_steps`.
- **Substep lifecycle:** a model exercising transitions + events + interventions
  - balance + a coincident obs+intervention, hand-computed — events read the
    stage-1 snapshot, interventions the post-transition state, balance last, and
    the obs scores the post-effect state. Asserted on actual counts.
- **Read/write at the type level:** a `trybuild` test that a mutating `Observe`
  _fails to compile_.
- **Capability gate (consolidation test):** `exact` + PGAS → the clean
  not-implemented error; the same model runs under `simulate`/`pfilter`.
- **Cross-backend fire-time:** same model, off-grid intervention, scored across
  chain-binomial and tau-leap, consistent and → 0 as `dt → 0` (red-first).
- **Gillespie propensity invalidation:** a model where skipping the
  post-mutation recompute is detectably wrong; assert it fires; assert `clip`
  (not `next_boundary`) is the query.

### External oracles for the deferred exact-PGAS (Stage 3)

The only step that can silently shift a posterior. It does not ship on parity:
the He et al. (2010) measles **pomp cross-check** (already caught gh#52/gh#53),
the **Richardson `dt`-ladder** convergence _rate_, the **FD gradient battery**
re-validated after the reference moves, and **posterior non-drift** (means
within prior credible bands + a KS check on marginals). Plus: byte-identical
`(counts_before, flows, gammas)` records between pre- and post-migration
`simulate_reference` _and_ between `step_one` and the Schedule-driven
free-particle propagation on the off-grid corpus.

### What "done" means

No stage merges without a clean full `make test`; parity stages additionally
require the differential-oracle corpus (off-grid) green and the RNG count+order
invariant intact; the deferred PGAS step additionally requires the
external-oracle battery. No `--no-verify`, no widened tolerance, no skipped
gate.

## Relationship to the observation-data proposal

Complementary, kept as separate documents (the obs-data proposal is re-edited
after this lands). Its **data layer** — `LongRow` parse, `bind`, `BoundObs`,
cardinality, `Counted`, the NaN guard — constructs the `Observe` effects'
observed series and is independent of the timeline; it proceeds in parallel. Its
**temporal layer** — off-grid policy, `--snap-observations`, dt-collisions —
**re-homes here** as the `snap | exact` knob plus the schedule's collision
guard; that machinery is not built in the obs-data proposal. The per-stream
`Interval` reset it deferred is the `ResetWindow` stage here.

## Future entry points (deferred seams)

Three extension axes the architecture leaves open, none built in this push:
**Trigger** as a first-class enum (reactive `StateCondition`, windowed
`set(param)` gh#50, activation dates gh#171); **Projection** composable
(stratum-subset sums, effort weighting, gh#171); a separate **Reduction** axis
(trajectory-functionals — `peak`, `n_episodes`, gh#172 — the substrate for
summary-statistic / synthetic-likelihood scoring). Out of the unification
entirely: vital dynamics and spatial coupling (transition-graph changes, not
timeline effects); reporting-delay convolutions (scoring-with-memory).

## Out of scope

- Gillespie's internal advance under a fixed `step(dt)` contract — event-driven;
  it uses the schedule via `clip`, not `next_boundary`.
- The full ODE-inference design (its own proposal); only the driver seam is
  reserved here.
- exact-PGAS — deferred to Stage 3, gated on gh#175.
- reactive triggers, the reduction axis — their own proposals when consumers
  exist.
