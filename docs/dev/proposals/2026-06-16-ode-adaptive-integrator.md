# Adaptive (RK45) ODE integrator + unified flow accounting

- **Status:** Draft (Phase 0 RFC)
- **Issue:** gh#166
- **Relates:** gh#52 / gh#227 (Richardson dt-check), gh#54 (`Expr::Dt`),
  [`2026-06-15-ode-gradient-inference.md`](2026-06-15-ode-gradient-inference.md)

## Problem

The ODE backend integrates with a **fixed-step RK4** at the user's `dt`, clipped
only to land on output and intervention boundaries. For a long-horizon,
mostly-quiescent epidemic the trajectory is near-flat for long inter-outbreak
stretches, yet fixed-step RK4 pays the full per-step cost everywhere — and a
deterministic MLE/profile fit pays that cost across hundreds of objective
evaluations. The reported case: a 23-patch × 2-age cVDPV2 model over 4382 days,
AFP monthly, 4 estimated params, `nl-sbplx` on `backend = "ode"`, dt=1 — **~1107
s for one converged MLE** (single chain). Coarsening `dt` is not a clean lever:
it trades trajectory accuracy and risks misplacing SIA-pulse intervention times,
which fall on arbitrary integer dates.

An adaptive, error-controlled integrator takes large steps where the trajectory
is smooth and small steps only where it is rapidly changing — the right shape
for this cost profile, without the dt-coarsening accuracy tax.

This RFC also corrects a latent accuracy asymmetry that the adaptive work
exposes: the ODE backend integrates _state_ to high order but accumulates
_incidence flows_ with first-order Euler. We unify both onto the same integrator
(decision: §"Flow accounting" Q1B).

## Current state (verified against code)

- `rust/crates/sim/src/ode.rs:99` `rk4_step` — classic fixed-step 4-stage RK4,
  four RHS evaluations per step, state clamped `≥ 0` after each step.
- Step driver `run_ode` (`ode.rs:166`): builds
  `Schedule::new(cfg.dt, …, StepPolicy::Exact, output_times, intervention_times)`
  and advances by `schedule.substep = dt.min(next_boundary − t)` (`ode.rs:232`).
  The step is the nominal `cfg.dt` **clipped to land exactly on the next output
  / intervention / event boundary**. Snapshots are emitted on the fixed
  `model.output.times` grid; effects (events then interventions) are applied at
  boundaries through `EffectBatch`. The comment at `ode.rs:183` already records
  "ODE is dt-independent, so EXACT and snap coincide."
- Flow (incidence) accumulation is **explicit Euler** (`ode.rs:277`):
  `flow_acc[i] += rate(t_start)·dt`, a left-rectangle rule via a _separate_
  `eval_propensities` call distinct from the RK stages — global order O(dt). The
  integrated _state_ is O(dt⁴) (RK4); the _flow_ is O(dt).
- Per ODE step there are therefore **5 propensity evaluations**: 4 inside the RK
  stages (for `dX/dt`) + 1 standalone for the flow. The stage propensities are
  computed and discarded.
- After each output the snapshot's flows are emitted and `flow_acc` is zeroed
  (`ode.rs:310`, `:323`) — so `snapshot.flows` carries _per-interval_ incidence,
  reset at each observation. `compute_ode_loglik`
  (`rust/crates/cli/src/fit/runner.rs:776`) depends on this: it walks snapshots,
  sums `flows` into a running cumulative, and scores at each obs time.
- No integrator abstraction: `rk4_step` is a free function, `run_ode`
  monolithic.
- ODE capabilities: `REAL_COMPARTMENTS | RUNTIME_DT`. `Expr::Dt` rates (gh#54)
  evaluate at the realized substep length `dt_actual`.
- IR surface (`rust/crates/ir/src/model.rs:68` `SimulationConfig`;
  `ir/schema.json` `simulation_config`):
  `{ t_start, t_end, time_semantics, dt?,
  rng_seed? }`. DSL `simulate {}`
  accepts only `from`, `to`, `dt` (`ocaml/lib/compiler/parser.mly:829` rejects
  any other key by name).

## We keep fixed-step RK4 — both integrators coexist behind one trait

Fixed RK4 stays, and stays the **default**:

1. **Goldens / byte-identical reproducibility** — it is the deterministic
   reference the golden surface is pinned to.
2. **`Expr::Dt` models require a fixed `dt`** (gh#54). They are capability-gated
   out of rk45, so fixed RK4 is their only integrator.
3. **Auditability** — a fixed, inspectable step, not an adaptive controller's
   step sequence.
4. **No Jacobian, well-tested.**

rk45 is opt-in (`integrator = "rk45"`). The two live behind a small seam so
"support both" is a shared substrate, not a fork:

```rust
/// One integrated ODE state: compartments plus the cumulative per-transition
/// flow integrals. `flow` is `∫ rate dt` since the last output reset (monotone
/// non-decreasing within an interval; reset to 0 at each output boundary so
/// `snapshot.flows` stays per-interval incidence — pomp's accumulator-variable
/// semantics, King et al. 2016 JSS).
struct OdeState {
    int:  Vec<f64>,   // integer-compartment values, clamped ≥0
    real: Vec<f64>,   // real-compartment values, clamped ≥0
    flow: Vec<f64>,   // cumulative ∫rate_i dt per transition (NOT clamped)
}

/// Advance state across ONE [boundary, next_boundary) interval. `h_max` is the
/// distance to the next boundary; the stepper must not cross it. Returns the
/// step actually taken (`Rk4Fixed` == h_max; `Dopri5` ≤ h_max, re-entered until
/// the boundary is reached).
trait OdeStepper {
    fn advance(
        &mut self,
        model: &CompiledModel, params: &[f64],
        t: f64, h_max: f64,
        state: &mut OdeState,
    ) -> Result<f64 /* h_taken */, SimError>;
}

struct Rk4Fixed;                         // wraps today's rk4_step; h_taken == h_max
struct Dopri5 { atol: f64, rtol: f64, h: f64 /* carried step guess */ }
```

`run_ode` becomes a thin driver: the `Schedule`/`Cursor`/`EffectBatch` boundary
loop is unchanged; only the inner "advance to next boundary" call routes through
`OdeStepper`. The boundary-landing trick reduces to `h = self.h.min(h_max)` —
the controller's natural step is clipped to the next output/intervention time,
so SIA pulses land exactly (the issue's concern). `Dopri5` is the standard
machinery:

```rust
impl OdeStepper for Dopri5 {
    fn advance(&mut self, model, params, t, h_max, state) -> Result<f64, SimError> {
        let mut h = self.h.min(h_max);
        loop {
            // 7-stage DOPRI5: y5 (5th order) and the embedded y4 from the SAME
            // stage rate evals k1..k7. Flows ride along as augmented state.
            let (y5, y4) = dopri5_stages(model, params, t, h, state)?;
            let err = scaled_error(&y5, &y4, state, self.atol, self.rtol); // ‖·‖/tol
            if err <= 1.0 {
                *state = y5;                        // accept
                self.h = pi_controller(h, err);     // grow on easy regions; carried
                return Ok(h);
            }
            h *= shrink_factor(err);                // reject, retry smaller
            if h < H_MIN { return Err(SimError::OdeStepUnderflow { t }); }
        }
    }
}
```

## Flow accounting — the design decision (Q1B, chosen)

Frame it as **two integrators × two flow schemes**; only one cell is contested:

|               | **Euler flow** `c += rate(t)·dt`, O(dt) | **Augmented flow** `dc/dt = rate`, integrator-order |
| ------------- | --------------------------------------- | --------------------------------------------------- |
| **Fixed RK4** | today                                   | **chosen (Q1B)**                                    |
| **RK45**      | pointless (incidence stuck at O(dt))    | **required**                                        |

For rk45 the right column is mandatory: a 5th-order _state_ integrator with
O(dt) _incidence_ is self-defeating, and silently so (prevalence looks converged
while incidence lags). So the only question is the fixed-RK4 cell.

**What augmented flow is.** Treat each cumulative flow as an integrated variable
`c_i` with `dc_i/dt = rate_i(X(t), t)`, carried through the same RK stages as
the state and reset to 0 at each output boundary (preserving the per-interval
`snapshot.flows` semantics `compute_ode_loglik` relies on). The per-transition
propensities needed for `dc_i/dt` are the **same ones the stages already compute
for `dX/dt`** and currently discard.

Two verified facts drive the decision:

1. **The state path is independent of flows.** Nothing in `dX/dt` reads `c_i` —
   flows are write-only accumulators. So adding `c_i` to the integrated vector
   does **not** perturb the compartment integration: **prevalence stays
   byte-identical** under either choice. Only the _flow_ numbers change.
2. **The trajectory baseline hashes flows.**
   `rust/crates/sim/tests/gate_trajectory_baseline.rs:67` `trajectory_hash`
   mixes `int_state.counts`, `real_state.values`, **and** `Flows::Real` bits
   (lines 76–96). So even though prevalence is unchanged, **every ODE entry's
   hash moves** under Q1B (the raw flow vector changes for any model with a
   nonzero transition) — the ~41 ODE baselines, plus any expected-output TSV
   carrying `flow_<name>` columns, plus possibly the cross-language
   `test_ocaml_to_rust.sh` drift check.

Bonus: augmenting reuses the stage propensities, so the fixed path drops from
**5 → 4** propensity evals per step — augmented flow is simultaneously _more
accurate, one mechanism instead of two, and slightly faster_. The cost is
entirely the golden movement.

### Options considered

**Q1A — leave fixed RK4 on Euler; only rk45 gets augmented flow.**

- _Pro:_ zero golden movement; purely additive Phase 1; the dt-check's
  documented incidence-O(dt) caveat stays true for the default path.
- _Con:_ two flow code paths coexist; the latent O(dt) incidence bias persists
  on the _default_ integrator — a user can only escape it by switching to rk45.

**Q1B — unify: both integrators use augmented flow. (CHOSEN)**

- _Pro:_ one flow mechanism; incidence is high-order on the default path too,
  fixing the latent wart; the dt-check incidence caveat _dissolves_; removes the
  redundant 5th eval. States and flows always integrated by the same method.
- _Con:_ moves every ODE trajectory-baseline hash (flows differ) even for
  prevalence-only models; a deliberate human-loop golden review; changes
  existing users' incidence numbers; requires an _independent oracle_
  (scipy/deSolve incidence) to prove the new values are _more correct_, not
  merely different.

**Rationale for Q1B.** camdl is approaching stability; carrying a known
first-order-biased default flow path is tech debt that compounds (every new
ODE-incidence consumer inherits the bias, and the dt-check would forever
footnote it). Doing it once, now, with an external oracle, removes a class of
downstream surprises. Backwards compatibility is a non-goal at alpha; the golden
movement is acceptable as an _explicit, reviewed, oracle-validated correctness
commit_ (separate from the rk45 feature so each commit is one thing).

## Method provenance & references

The numerics here are standard; the camdl-specific work is the type seam, the
boundary-clipped integration, and the flow unification — not the algorithms.

- **DOPRI5 (embedded RK4(5)) + PI step-size control** — Dormand & Prince (1980),
  _J. Comp. Appl. Math._ 6(1):19–26; Hairer, Nørsett & Wanner (1993), _Solving
  Ordinary Differential Equations I: Nonstiff Problems_, 2nd ed., Springer,
  §II.4–II.5 (the tableau and the standard PI controller); Press et al.,
  _Numerical Recipes_ 3rd ed., ch. 17.2 (adaptive-stepsize RK). There is one
  canonical tableau; it is transcribed, not designed.
- **Flow as augmented ODE state** — the textbook quadrature-by-augmentation
  trick used by SUNDIALS/CVODES (Hindmarsh et al. 2005, _ACM TOMS_
  31(3):363–396) and R `deSolve`. In epidemiology specifically it is the
  cumulative-incidence compartment idiom; pomp's **accumulator variables**,
  zeroed at each observation time, are the same construct and the same reset
  semantics we use (King, Nguyen & Ionides 2016, _J. Stat. Soft._ 69(12), §2.1
  accumulator variables).
- **Stiff solvers** (Phase 2, not now) — Rosenbrock / BDF need the state
  Jacobian; see §"Algorithm choice".

## Surface (DSL + IR + CLI)

rk45 is opt-in. ODE has no RNG, so the adaptive trajectory is deterministic
given `(model, θ, atol, rtol)`, but it is **not byte-identical** to fixed-step
RK4 — making it the default would move every ODE golden. Default stays `rk4`.

DSL (`simulate {}` gains keys; `dt` becomes the fixed-RK4-only knob):

```camdl
simulate {
  from = 0 'years
  to   = 40 'years
  integrator = "rk45"      # "rk4" (default) | "rk45"
  atol = 1e-8              # absolute tolerance (rk45 only)
  rtol = 1e-6              # relative tolerance (rk45 only)
}
```

CLI: `camdl simulate … --integrator rk45 --atol 1e-8 --rtol 1e-6`, with the same
override on `fit run` so deterministic stages (`nl-sbplx`, `nl-bobyqa`, `mh`)
can opt in.

IR: `simulation_config` gains `integrator: "rk4" | "rk45"`, `atol?`, `rtol?`.
This is an **IR schema change** — the atomic update (CLAUDE.md "Changing the IR
schema"): `ir/schema.json` + bump `ir/VERSION` (0.14 → 0.15), OCaml `ir/`
types + (de)serialize, Rust `ir/` types, then
`make update-golden && make
update-expected`, in one commit. The DSL change (new
`simulate` keys, dimensionless `atol`/`rtol`) touches `lexer.mll` / `parser.mly`
/ `dimcheck.ml` and must give the spec's named-key error for any unknown key,
plus a migration line in `docs/language-changes.md`.

## Capability interaction: `Expr::Dt`

Under adaptive stepping there is no single nominal `dt`, so a rate referencing
`Expr::Dt` (gh#54, `RUNTIME_DT`) has no well-defined value. Phase 1:
**capability-gate `RUNTIME_DT` models out of rk45** with an honest hard error
("model uses `dt` in a rate; the rk45 integrator has no fixed step — use
`integrator = \"rk4\"`"), rather than silently redefining `dt`.

There is a subtler interaction _on the fixed path under Q1B_: today the Euler
flow evaluates `rate(…, dt_actual)·dt`, so a `RUNTIME_DT` rate's `dt` is the
realized substep. Augmented flow integrates `dc/dt = rate(…, ?)` through stages
that have no single `dt`. This must be resolved (task B2): the leading candidate
is to **keep the Euler flow path for `RUNTIME_DT` models on fixed RK4** (they
are already a special, fixed-step-only class) and use augmented flow for all
others — documented and tested, not silent.

## Algorithm choice

- **Phase 1: DOPRI5 (Dormand–Prince RK45)** — explicit, embedded 4th/5th-order
  error estimate, standard PI controller. Directly addresses the reported pain
  (near-quiescent stretches between outbreaks); explicit, allocation-light, no
  Jacobian.
- **Stiffness:** TB latency (per-decade reactivation vs fast progression) and
  SIA pulses create real timescale separation. Explicit DOPRI5 copes by
  shrinking steps in stiff regions (partial loss of speedup, never wrong). A
  genuine stiff/implicit method (Rosenbrock / BDF) needs the **state Jacobian**
  ∂(dX/dt)/∂X. Reality check: `ocaml/lib/ir/autodiff.ml` currently
  differentiates rates **wrt parameters only** — compartment counts are
  constants (`autodiff.ml:26`, `:57`: `Pop _ -> false`) — so the Jacobian is
  _not_ available today. The symbolic-diff engine could gain a
  differentiate-wrt-`Pop` target (a contained extension, not new autodiff), but
  that is **Phase 2**, gated on a model explicit DOPRI5 actually chokes on.

## Validation & external oracles

External validation against independent implementations is a first-class
deliverable, following the `tests/external/cases/` pattern (reference generated
once, **cached fixtures committed so CI needs neither Python nor R** — as in the
He-2010 pfilter loglik gates).

1. **Primary oracle — scipy `solve_ivp`.** Encode each canonical model's ODE RHS
   in Python; integrate with `method="RK45"` (DOPRI5 — algorithm-level agreement
   with our rk45) and `method="LSODA"` (an independent adaptive reference).
   Compute incidence with an explicit cumulative-incidence variable added to the
   system. Assert camdl agrees at the observation grid to a stated tolerance,
   for **both prevalence and incidence**.
2. **Secondary oracle — R `deSolve::lsoda`** (Hindmarsh/Petzold LSODA) on the
   same models, as a second independent implementation in a second language.
3. **Canonical models:** SIR, SEIR, the 2-stage-latency TB model, and one model
   with a mid-horizon intervention pulse (exercises exact-boundary landing under
   adaptive stepping).
4. **Internal agreement gate** (no external dep): fixed-RK4 at fine `dt` and
   rk45 at tight `(atol, rtol)` must agree on **both prevalence and incidence**
   to a stated tolerance — pins the augmented-flow design and catches an
   incidence regression a prevalence-only check would miss.
5. **State-isolation gate:** a _state-only_ trajectory hash (int+real, excluding
   flows) must be **byte-identical across the Euler→augmented flow change** —
   proves Q1B did not perturb the compartment integration (only flows move).
6. **Analytic gate:** with a time-varying propensity where Euler is provably
   O(dt)-wrong, augmented flow must match the analytic ∫rate dt to high order.
7. **Determinism gate:** same `(model, θ, atol, rtol)` → byte-identical
   trajectory across runs.

The cVDPV2 ~1107 s → target wall-clock is a **benchmark reported in the PR**,
not a CI gate (hardware-dependent).

## Phasing

Three commits, each one thing:

- **Phase A — seam, byte-identical.** Introduce `OdeState` + `OdeStepper`;
  refactor `run_ode` to drive via the trait; `Rk4Fixed` reproduces today exactly
  _including_ the Euler flow. No golden moves. Add the state-only hash gate
  (capture baselines now, while Euler) so Phase B can prove state is untouched.
- **Phase B — flow unification (Q1B), goldens move, oracle-validated.** Switch
  flow accounting to augmented state (`dc/dt = rate` through the RK stages;
  reset at outputs; remove the redundant 5th eval); resolve `Expr::Dt` ×
  augmented flow (task B2). Regenerate ODE goldens as an explicit reviewed
  commit whose _subject is the golden movement_. Prove state-only hashes
  unchanged; flow values validated against scipy/deSolve incidence.
- **Phase C — adaptive rk45 (opt-in) + IR schema.** `Dopri5` behind the trait
  (reusing augmented flow); IR schema + DSL + CLI surface; `RUNTIME_DT`
  capability gate; internal fixed-vs-rk45 agreement gate; external rk45
  validation; determinism gate; cVDPV2 benchmark in the PR.

## Task list (correctness-first)

### Phase A — extraction (must stay byte-identical)

- **A1.** Define `OdeState { int, real, flow }` and the `OdeStepper` trait;
  refactor `run_ode` to drive boundaries via the trait. `Rk4Fixed` wraps the
  existing `rk4_step` and the existing Euler `flow_acc` exactly.
- **A2.** Gate: the existing `gate_trajectory_baseline` ODE entries stay
  byte-identical (proves A1 changed nothing).
- **A3.** Add a **state-only** trajectory hash gate for ODE models (hash
  int+real only, exclude flows); capture baselines now (Euler era). This is the
  instrument Phase B uses to prove prevalence is untouched.

### Phase B — flow unification (Q1B)

- **B1.** Implement augmented flow: extend the integrated vector with `c_i`,
  `dc_i/dt = propensity_i`, advanced by the same RK4 stages; reset `c_i` at each
  output boundary so `snapshot.flows` remains per-interval incidence; delete the
  standalone `eval_propensities` flow eval (5→4 evals/step).
- **B2.** Resolve `Expr::Dt` × augmented flow on fixed RK4 (keep Euler flow for
  `RUNTIME_DT` models, or define `dt` in the augmented derivative); implement
  the chosen semantics with a test pinning it. **Decision needed before coding
  B1 for `RUNTIME_DT` models.**
- **B3.** Analytic correctness test (gate #6): time-varying-rate model where the
  augmented incidence matches ∫rate dt to high order and Euler would be O(dt).
- **B4.** State-isolation: the Phase-A state-only hashes are **unchanged** after
  B1 (gate #5) — prevalence byte-identical.
- **B5.** External incidence validation vs scipy `solve_ivp` (RK45 + LSODA) and
  deSolve on SIR/SEIR/TB/pulse, **for both prevalence and incidence** (gate #1,
  #2); cache fixtures.
- **B6.** Regenerate ODE goldens (`update-golden`/`update-expected` + recapture
  `gate_trajectory_baseline`); **explicit reviewed commit, subject = the golden
  movement**, citing the oracle that confirms the new flows are the accurate
  ones.
- **B7.** Confirm `compute_ode_loglik` incidence scoring is unchanged in shape
  (still sums `snapshot.flows`) and now consumes high-order flows; add/extend an
  ODE-inference incidence test.

### Phase C — adaptive rk45 + schema

- **C1.** IR schema change (atomic): `simulation_config` += `integrator`,
  `atol`, `rtol`; bump `VERSION` 0.14→0.15; OCaml + Rust ir types/serde; DSL
  keys (`lexer.mll`/`parser.mly`/`dimcheck.ml`, dimensionless tolerances);
  named-key error for unknowns; `docs/language-changes.md` entry; golden regen
  in the same commit.
- **C2.** `Dopri5`: 7-stage DOPRI5 tableau, embedded 4(5) error, PI controller,
  `h` carried across boundaries and clipped to `h_max`, `H_MIN` underflow error,
  max-rejections guard. Flows via the Phase-B augmented mechanism.
- **C3.** Capability gate: `RUNTIME_DT` models rejected on rk45 with the honest
  hard error; test it.
- **C4.** CLI `simulate --integrator rk4|rk45` override only (method-only): it
  mutates the model in `resolve_run_model` _before_ the run-id is computed, so
  the choice is captured in the content hash. Tolerances and the **inference
  path** stay model-declared (`simulate { integrator = rk45 { atol, rtol } }`,
  which `run_ode` honors on the fit path). No `fit run --integrator` / `--atol`
  / `--rtol`: the integrator changes the numerics, so it belongs to the model's
  content identity — a CLI override on the inference path is a second, un-hashed
  way to change numerics that must be kept consistent with the recorded run-id
  by hand. The reproducible mechanism is the model-declared block.
- **C5.** Internal agreement gate (#4): fixed-RK4(fine dt) vs rk45(tight tol)
  agree on **prevalence and incidence** to tol.
- **C6.** External rk45 validation vs scipy + deSolve on the canonical models
  (#1, #2); cached fixtures.
- **C7.** Determinism gate (#7): same inputs → byte-identical trajectory.
- **C8.** Default `(atol, rtol)` calibration: pick values where rk45 agrees with
  fine-`dt` RK4 to ~sub-nat loglik on the validation models; record the
  calibration.
- **C9.** cVDPV2 speedup benchmark in the PR description (not a CI gate).
- **C10.** (Optional, follow-up) extend the dt-check `run_ladder` to a
  _tolerance_ ladder for rk45 (halve `atol`/`rtol`, check loglik stable) — same
  driver, different per-rung knob.

## Open questions

1. **B2:** `Expr::Dt` × augmented flow on fixed RK4 — keep Euler for
   `RUNTIME_DT` models, or define the augmented semantics? (Audit `RUNTIME_DT`
   usage in goldens/tests to size it.)
2. **C8:** default `atol`/`rtol` values (calibration-pinned).
3. **C2:** PI-controller gains, min/max step, max rejections before surfacing a
   `SimError` — standard defaults, but record them for reproducibility.
4. Should `Rk4Fixed` flows also reset-at-output via the same code path as
   `Dopri5`, so the reset logic is shared rather than duplicated? (Lean yes —
   single mechanism.)
