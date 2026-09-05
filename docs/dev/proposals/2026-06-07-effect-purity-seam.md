# Effect resolution as a pure seam

Date: 2026-06-07 Status: Implemented (compartment scope) Implemented: 8b9111d
(types + pure resolver), d8a9259 (interventions), 973cc2e (events +
events-on-real), 01f92f1 (ODE de-quantization). Discrete backends
byte-identical; 3 ODE corner-case baselines re-derived (exact
fraction-transfer), verified by the continuous unit oracle. Reactive parameter
effects remain out of scope (see below). Supersedes: the `CountStoreMut`
state-view from the lifecycle design review; reframes Tier 2 item "3a" of the
lifecycle consolidation. Follow-on: reactive parameter effects (param-target
actions) are out of scope here — see "Out of scope" — because they violate an
inference invariant that needs its own design.

## Summary

The within-substep effect system has two orthogonal axes. One is
**representation** (integer `i64` compartments vs continuous `f64`
compartments). The other is **purity**: _resolving_ an effect (reading a state
snapshot and computing a value or delta — no side effect) versus _applying_ it
(writing state). The current code conflates both axes at every effect site, and
that conflation is where the bugs live: events silently dropped when they target
a real compartment, and the rounding/clamp logic duplicated between the
intervention and event interpreters.

This proposal makes purity a first-class type boundary: a single **pure
resolver** that reads an immutable `StateRef` and emits **typed deltas**
(`Int | Real`), and a **trivial applier** that writes them. All rounding,
clamping, snapshot arithmetic, and arena dispatch — the entire bug-prone surface
— lives in the pure resolver, where it is testable as plain data. Application
becomes a `+=` that cannot carry a bug.

It supersedes the `CountStoreMut` representation-only seam and fixes the
events-on-real silent-drop bug outright. Compartment effects only (int/real);
parameter-target effects are deferred (see "Out of scope").

**Correctness over preservation, stated precisely.** This is pre-release
software on a feature branch; the goal is _right_, not byte-identical to the
current `i64` effect outputs. But "byte-identity is off" is not "anything goes":
this proposal has two kinds of change and they must not be conflated.

- **The typing change** (route events + interventions through one pure resolver,
  keeping the _current_ rounding rules) is provably neutral — the resolver is
  RNG-free and event deltas stay out of the density's flow terms — and is gated
  to stay byte-identical on the discrete backends.
- **The behaviour changes** (ODE stops quantizing at effect boundaries;
  events-on-real now apply; any rounding rule we deliberately rationalize) _move
  numbers_, and each is verified as a **change** against an independent oracle,
  never re-blessed.

The invariants that remain are RNG draw order (paired-seed CRN) and the PGAS
complete-data density + gradient; the proposal preserves them structurally for
the typing change and verifies them as unmoved across the behaviour changes.

## The two axes

```
              Discrete (i64)                 Continuous (f64)
resolve   round / floor / clamp → IntDelta   exact → RealDelta     ← all bug-prone arithmetic
apply     int[i] += delta                    real[i] += delta      ← trivial, branch-free
```

`CountStoreMut` placed both axes inside one enum whose methods (`add_raw`,
`fraction_transfer`, …) branched on representation _and_ rounded _and_ mutated,
per call. Separating the axes makes each cell trivial: representation is carried
by the delta's _type_ (so the apply has no runtime `match`), and purity is
carried by the `StateRef`/`StateMut` _types_ (so a resolver cannot mutate).

## Current state (what exists today)

| Layer            | Today                                                                                                                                                                                                                                                                                                         | On the purity axis                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Schedule         | `Schedule::substep(&self, cursor, t) -> Option<f64>` — documented PURE (`schedule.rs:152`); `Cursor` is `Copy`; only mutation is `cursor.pass_*()`.                                                                                                                                                           | already split                       |
| Observation      | `log_likelihood(state, i, params) -> f64`, projections `state -> value`; the `FlowSum` projection reads `flows: &[u64]` (`multi_stream_obs.rs:192`). Pure reads.                                                                                                                                              | already split                       |
| Events           | `inject_event_deltas(.., snapshot: &IntState, real_snapshot: &RealState, .., pending: &mut Vec<(usize,i64)>)` (`lifecycle.rs:42`, `intervention.rs:154`) — pure producer into a sink, but **i64-only**: every arm guards on `global_to_int` (`:184,:195,:212,:226`) and silently drops a real-targeted event. | pure but half-typed                 |
| Interventions    | `apply_intervention(.., int_s: &mut IntState, real_s: &mut RealState, ..)` (`intervention.rs:296`) — evaluates, branches on arena, and writes inline; rounding duplicated from the event path (`floor(src*frac)` at `:329` and `:196`; round for add/set/absolute at `:357/:379/:387` and `:178/:207/:224`).  | mixes resolve + apply               |
| Balance          | `let v = eval_resolved(&bal.expr,&ctx); current.counts[idx] = v.round() as i64` (`lifecycle.rs:95-101`); warns (not errors) on negative; a second copy exists at PGAS init (`pgas.rs:997-1002`).                                                                                                              | pure eval, ad-hoc apply, duplicated |
| Kernel (ADVANCE) | multinomial / Poisson / RK4 / SSA draws — RNG + state.                                                                                                                                                                                                                                                        | irreducibly impure                  |

The engine already wants to be a **pure compute core** (schedule queries, effect
resolution, obs projection) with exactly **two loci of mutation**: the cursor
advance (`++`) and the kernel draw (the one RNG-consuming, genuinely stateful
step). Interventions are the only layer still straddling both, and the straddle
is where the leaks are. (The same-class stale-read bug — propensities reading a
disconnected `scratch.real_s` — was fixed in 5c7585c; it is what motivated
naming this seam, and `StateRef`/`StateMut` make that class a compile error
rather than a runtime guard.)

## The seam

```rust
// Representation collapses into the delta TYPE — no runtime branch at apply time.
struct IntDelta  { idx: usize, delta: i64 }
struct RealDelta { idx: usize, delta: f64 }
struct EffectDeltas { int: Vec<IntDelta>, real: Vec<RealDelta> }

// Purity becomes two TYPES, so the compiler enforces the split.
struct StateRef<'a> { int: &'a [i64],     real: &'a [f64]     }  // resolvers CANNOT mutate
struct StateMut<'a> { int: &'a mut [i64], real: &'a mut [f64] }

// THE pure resolver: every round / floor / clamp / snapshot-subtract / arena-dispatch,
// once. No RNG, no mutation. Returns data — golden-testable directly.
fn resolve_effects(
    actions: &[Action], snap: StateRef<'_>,
    model: &CompiledModel, params: &[f64], t: f64, dt: f64,
) -> Result<EffectDeltas, SimError>;

// THE trivial applier — no arithmetic, no branch on representation.
fn apply_effects(d: &EffectDeltas, s: StateMut<'_>);   // int[i] += …; real[i] += …
```

The effect kinds become _the same pure producer at different snapshots_:

```
Events:        resolve_effects(events, PRE-advance  snap) → int/real deltas fused into the kernel draw
Interventions: resolve_effects(ivs,    POST-advance snap) → apply after the draw
Balance:       a derived Set effect — eval(bal.expr) → IntDelta — resolved by the same path
```

### Why deltas

For **events** the delta is mandatory, not stylistic: an event must apply
_atomically with_ the multinomial draw, from the start-of-step snapshot
(`chain_binomial.rs:461-469`, `tau_leap.rs:312-320`). An in-place mutator cannot
express that — it would read and write the same buffer the draw is about to
touch. So the snapshot-relative delta is the right shape there.

For **interventions** there is no draw to fuse with; they could mutate
post-advance state directly. They adopt the delta representation anyway so they
share the one resolver (the bug-prone `round/floor/clamp/arena-dispatch`
arithmetic exists once instead of in two interpreters), accepting a redundant
`round(v) − snap[i]` then `+=` round-trip — `O(actions)`, negligible. That is
the trade: a redundant subtract-then-add buys a single `Set` arm instead of two
that must stay in sync. The win is the shared _resolver_, not the delta being
intrinsically more correct for interventions.

## What changes versus today

- **Events gain a real branch.** `inject_event_deltas`' int-only arms
  (`intervention.rs:184/195/212/226`) become a `match` over the arena the
  compiler forces you to complete. An event targeting a real compartment now
  applies instead of vanishing. This is a confirmed silent-drop bug with _no
  current golden coverage_ — it needs a purpose-built red→green fixture with a
  non-integer amount (to pin that the real path does not round).
- **Interventions become resolve-then-apply.** `apply_intervention`'s inline
  arena branch + write (`intervention.rs:296-418`) is replaced by
  `resolve_effects` (post-advance snapshot)
  - `apply_effects`. The round/floor duplication against the event path
    collapses into the single resolver. The existing-but-untested real branches
    (`:332,:360,:380,:411`) are preserved — and need a guard fixture, since they
    have no coverage today.
- **ODE stops quantizing its integrator state at effect boundaries.** Today ODE
  rounds f64→i64 at every effect boundary (`ode.rs:150 to_states`, back-cast
  `:242/:290`). Applying ODE's integer compartments through the exact/`Real`
  path lets the fractional state the RK4 integrator accumulated survive the
  boundary. **Scope of the claim:** this de-quantizes the _integrator state_;
  the _recorded output_ still rounds int compartments at every snapshot
  (`ode.rs:203/248/297/310`), and the boundary flow accumulation (`ode.rs:264`)
  still reads a rounded snapshot, both by the output contract. So the visible
  compartment trajectory moves (fractional carry across the boundary) but is
  still reported as integers.
- **Balance is a derived `Set` effect**, resolved by the same path; the
  PGAS-init second copy (`pgas.rs:997-1002`) folds in. (Caveat below — it keeps
  its own _policy_.)
- **`CountStoreMut` is not introduced.** Its representation-only job is done by
  typed deltas; its rounding methods move up into the pure resolver.
- **The Tier-1 guards re-home cleanly.** `finite_action_value`
  (`intervention.rs:30`) becomes a check on each resolved amount inside the
  resolver (pure); the post-INTERVENE/BALANCE negative scan
  (`lifecycle.rs:104-120`) stays a check on post-apply state. Both currently
  cover only the integer path — the relocation must add a real-path analogue, or
  the guards silently miss the new `RealDelta` path.
- **The typing change keeps today's rounding; rounding rationalization is
  separate.** Routing the two interpreters through one resolver is done with the
  _current_ per-action rules preserved (so it is byte-identical on the discrete
  backends and density-neutral). Any deliberate change to a rounding rule (e.g.
  making fraction-floor vs absolute-round consistent) is a _separate,
  explicitly-blessed_ commit with a per-fixture justification — not folded into
  the typing change.

## Consolidations this opens (compartment scope)

1. **One read view for the effect/rate path.** A day-1 `StateRef{int, real}`
   behind `resolve_effects` and the rate evaluator's read of compartments. This
   does **not** extend to a single read type "everywhere" — see the non-goal
   below; the obs and inference read contexts are genuinely different and stay
   separate.
2. **Effects become a third pure primitive the filter bodies call.** Alongside
   the already- shared `Schedule::substeps` iterator (`particle_filter.rs:250`,
   `if2.rs:423`, `correlated_pf.rs:367`) and obs scoring,
   `resolve_effects`/`apply_effects` is a third pure piece every backend and
   filter reuses. The filter **bodies stay distinct** — resampling, death
   policy, IF2 perturbation, CPM noise injection, watchdogs, ancestor sampling
   are not shared and must not be folded into a god-driver (this is the Layer-5
   conclusion of the topology map; the purity seam strengthens the shared
   primitives, it does not merge the bodies).
3. **Gillespie de-specializes.** Its `clip` query (`schedule.rs:189`, already
   pure) is just a different pure time-advance; effects resolve/apply
   identically.

## Non-goals / where the seam stops

- **No shared filter driver.** Forward, PF, IF2, CPM, and PGAS share the _pure
  primitives_ (schedule, effects, obs) but keep four distinct bodies.
  "Everything but the kernel is shared" is false: resampling, log-weight
  accumulation, ESS/wall-clock/iteration watchdogs, IF2 perturbation + joint
  resample, CPM pre-drawn-noise + sorted resample, ancestry / prequential
  recording, and PGAS's per-substep records + ancestor sampling are forward-
  absent and per-algorithm. The seam ends at the shared primitive.
- **No "one read type everywhere."** `StateRef{int, real}` serves the
  effect/rate path. `EvalCtx` additionally carries `int_float_override` (the ODE
  f64-read-through-int trick, `ode.rs:60-66`, `propensity.rs:56-59`) and
  `projected` (obs likelihood); `ParticleState` has _no_ real compartments and
  `u64` flows and is memcpy'd into a scratch `IntState` for rate eval
  (`multi_stream_obs.rs:50-60`). Unifying all of these behind one view is the
  leaky god-view; keep the effect-read view distinct from the obs-read and
  inference-state contexts.
- **Do not merge balance's apply policy.** Balance shares the pure eval and the
  `Set` shape, but reads post-intervention state and _warns_ on negative rather
  than erroring (`lifecycle.rs:97-100`), and its target is exempt from the
  negative-count scan (`:110-112`). Share the resolver's eval; keep the timing +
  policy distinct.
- **Do not fold the kernel into the pure layer.** The draw is RNG + state by
  nature; it stays the impure closure.

## Invariants (and exactly how each is held)

- **RNG draw order / paired-seed CRN.** The resolver consumes zero RNG
  (`intervention.rs` and `lifecycle.rs` contain no draw calls), and event deltas
  are still resolved from the pre-advance snapshot and fused after all draws
  (`chain_binomial.rs:304-315` snapshot, `:392-455` draws, `:461` propose,
  `:467-469` apply). Re-typing does not move any of this. **But** once a
  previously-dropped real-event applies, subsequent substeps read a changed
  state and therefore draw differently — a real-event forward trajectory _will_
  diverge from today's bytes by construction. That is a correct change, verified
  as such, not a CRN violation.
- **PGAS complete-data density + gradient.** The density scores transition
  `flows` given `counts_before` (`pgas.rs:556`), and the gradient does the same
  (`pgas_grad.rs:57`); events land in `counts_after`, never in `flows` — pinned
  by `pgas_event_density.rs`. The resolver _returning data_ rather than mutating
  `counts_before` is what keeps this true; it is a typed guarantee, not a
  convention. The _formula and its conditioning_ are preserved by the typing
  change. The _numbers_ move wherever a previously-dropped effect now applies (a
  real-event changes `counts_after` → next `counts_before`), and those moves are
  verified, not asserted away. Both `pgas.rs` (density) and `pgas_grad.rs`
  (gradient, used by NUTS within PGAS) are on the verification surface, as is
  the CPM pre-drawn-noise mapping (`correlated_pf.rs:202-233`).

## Sequencing

1. **Types, behaviour-neutral.** Add `StateRef`/`StateMut`,
   `IntDelta`/`RealDelta`, `EffectDeltas`; wrap existing state. Touch no
   `EvalCtx`, no rate eval, no kernel. The read-interface unification (the
   `EvalCtx` reshape) is **not** in steps 1–4. Gate: the full byte-identity
   suite stays green; if any baseline moves, step 1 was not neutral — stop.
2. **Interventions + ODE de-quantization — correctness-bearing (not low-risk).**
   Move `apply_intervention` and balance to `resolve_effects` (compartment
   targets), keeping current rounding. Apply ODE integer compartments through
   the exact path so fractional integrator state survives the boundary. This
   _moves the ODE trajectory_ and has no current oracle; it lands red-first
   against the analytic fixture below, in its own commit, separate from the
   type-only step 1.
3. **Re-type the event path — the risky one.** `inject_event_deltas` emits
   `EffectDeltas` with the real branch added (red→green events-on-real fixture).
   Keep the pre-advance- snapshot, fused-with-draw timing. Verify (must stay
   green) the density, gradient, and CPM invariants; verify (moves, blessed) any
   rounding rationalization with a hand/pomp anchor. Re-run the off-grid
   event-misfire guard (`inference_event_misfire_guard.rs`, all three cases) — a
   re-typed event on a shortened final substep must still be rejected by
   `Schedule::reject_event_misfire`, not silently misfire.

Step 1 is low-risk (additive types). Steps 2 and 3 are **co-equal
correctness-bearing** changes, each isolated in its own commit with its own
red-first oracle.

## Verification (byte gates are gone — this is the replacement)

Gate inventory:

| Gate                                | Pins                                                   | Under this proposal                                                                                                         | Independent oracle?                                                                               |
| ----------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `gate_trajectory_baseline`          | full-traj hash, 4 backends, `ocaml/golden/*`           | **does not move** — it `model.interventions.clear()` (`:266`), so baselines run zero effects                                | n/a (doesn't exercise the path)                                                                   |
| `gate_corner_case_baseline`         | full-traj hash, 4 backends, `corner_cases/*`           | **moves** — ODE rows of `off_grid_intervention`/`coincident_obs_intervention`/`all_lifecycle` (step 2); event rows (step 3) | weak — only `event_intervention_agree` is hand-anchored (rate≡0, A=75/B=75), and only for _order_ |
| `gate_inference_baseline`           | PF marginal loglik, SIR on/off-grid                    | should **not** move (no effects in fixture) — negative control for step 3                                                   | invariance check only                                                                             |
| `gate_pgas_density_baseline`        | transition log-density, `seasonal_drift`               | should **not** move (events not in flow terms) — negative control                                                           | invariance check only                                                                             |
| `cross_backend_lifecycle_agreement` | all-backend agreement on coincident event+intervention | **rate ≡ 0 / integer-exact by construction** (`:18-21`) — cannot witness either move                                        | structurally blind here                                                                           |

The cross-backend agreement oracle is rate≡0 and integer-only, so it cannot see
the fractional-ODE move (no fractional state exists at k=0) or events-on-real
(every compartment is integer; no fixture anywhere targets a real compartment).
Required new oracles, before the step they gate:

1. **Analytic fractional-ODE-across-intervention fixture (step 2).** Pure decay
   `dN/dt = −μN` with a `transfer(fraction=f)` at an off-grid `t`: post-cull
   `N(t⁺) = N₀ e^{−μt} (1−f)` is closed-form. Assert the ODE trajectory matches
   the analytic value to RK4 tolerance (green) and _differs_ from the pre-fix
   rounded value (red). This checks the moved number rather than bracketing it —
   the discrete backends do **not** bracket it, because the move comes precisely
   from fractional state the integer backends never carry.
2. **events-on-real red→green fixture (step 3).** Real-targeted
   `add`/`transfer`: real compartment unchanged on current code (the drop),
   changed by the resolved amount after; use a **non-integer** amount so it also
   pins that the real path does not round.
3. **interventions-on-real guard (step 2).** `apply_intervention` already has
   real branches (`:332-411`) with no coverage; step 2 dedups that rounding, so
   a regression would pass silently today. A fractional transfer/set/add on a
   real compartment, asserted exact.
4. **Real-path D-finite / D-negative analogue (step 2/3).** When
   `finite_action_value` and the negative scan move into the resolver, add a
   non-finite/negative resolved amount on a real compartment, or the relocated
   guards drop the `RealDelta` path silently.
5. **Hand/pomp anchor for ≥1 discrete golden per rounding-policy change
   (separate commit).** If a rounding rule is rationalized, at least one moved
   discrete baseline's new value is checked against a hand computation or
   pomp/R, not re-blessed.

Red-first set per step: step 1 — no behaviour test possible (that _is_ the
assertion; byte suite stays green). Step 2 — oracle (1) red-then-green, oracle
(3), the D-guards stay green + gain a real case. Step 3 — oracle (2)
red-then-green, the density/gradient/CPM invariance controls stay green, the
off-grid guard re-runs, oracle (5) where rounding changed.

## Risks

- **Step 3 touches the density/gradient-adjacent fusion.** Mitigation: keep the
  draw sequence and the pre-advance/fused timing fixed; the density and gradient
  are _invariance controls_ (must stay green), and the chain==tau byte-identical
  fused-read equality is the CRN witness.
- **Re-blessing risk.** The moved ODE and discrete baselines have weak existing
  anchors; mitigation is oracles (1)–(5) above, which validate the new values
  independently.
- **`StateRef` flow membership.** Deferred: introduce `StateRef{int, real}`
  first; flows enter only when (and if) the obs read path is unified, which is a
  separate proposal.

## Out of scope: reactive parameter effects

A parameter-target effect (an NPI that sets or scales `beta` mid-run) is the
same resolve→apply _shape_, but it is deferred to its own proposal because it
violates an inference invariant the whole gradient stack rests on:

- `autodiff.ml:34` treats every `Param` as a **time-invariant constant**
  (`∂rate/∂θ = 1` uniform; `autodiff.ml:7-10`), and both the complete-data
  density (`pgas.rs:717`) and its gradient (`pgas_grad.rs:401`) evaluate at a
  _single trajectory-wide_ `params` slice. A mid-run parameter mutation makes
  the realized trajectory use `θ` then `θ′`, while the density and gradient
  assume constant `θ` — so the density is inconsistent with the trajectory and
  the gradient is _actively wrong_ (it biases NUTS/PMMH proposals, not merely
  stalls them as gh#186 does). There is no per-substep parameter mutation
  anywhere today; reactive params introduce a time-varying parameter the
  inference stack assumes does not exist.

The interim, when reactive params land, is forward-simulation-only with a hard
inference reject at the fit preflight (the pattern already used for real-coupled
inference, gh#191). That follow-on must also scope: the multi-crate IR change a
param-target `Action` requires (OCaml `ast.ml`/`parser.mly`/`ir.ml`/`serde.ml`,
`expander.ml` parameter-table resolution + dimcheck against the parameter's
dimension, the `runid` `ContentAddressed for Action` hasher arm, golden regen);
that it is unrelated to the existing pre-sim `set`/`scale` scenario overrides
(`params_resolver.rs:389-426`); and whether a `ParamScale` peer is needed (two
simultaneous multiplicative NPIs compose as a product, which absolute-set cannot
express).
