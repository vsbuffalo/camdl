---
status: proposal
date: 2026-06-26
tracking: gh#275
supersedes: docs/dev/proposals/2026-06-15-ode-gradient-inference.md (Phases 0–1 shipped; this is the gradient half)
prerequisites: gh#342 (3b — derive the differentiation traversal and fold the rate path onto Diffable/DerivEntry); gh#374 (transform Jacobian/derivative at the [lo,hi] bound); docs/dev/proposals/2026-07-03-unified-obs-gradient-autodiff.md (the one differentiation authority — §4.3 shared obs-gradient seam, §10 coordination, §11 = 3b)
related: gh#180 (obs ∂projected/∂θ term — LANDED), gh#78 (runtime --check-grads)
ir_version: 0.24 → 0.25 (state_grad, added as a Diffable position — see §1b)
---

# ODE + NUTS: the gradient spine and a gradient-based Bayesian sampler

## TL;DR

The ODE backend can fit by maximum likelihood (`nl-sbplx`, `nl-bobyqa`) and by
random-walk Bayesian inference (`mh` on `ode`). Both are gradient-free. This
proposal adds **`nuts` on `ode`** — gradient-guided Bayesian sampling of the
deterministic ODE likelihood — for faster, better-mixing fits of correlated and
ridge-shaped posteriors, the regime malaria models live in as they grow.

The entire cost is **producing one gradient**: `∇_θ log p(y | θ)` for the
deterministic ODE likelihood. NUTS itself is nearly free — the sampler
(`nuts::nuts_step`) is already model-agnostic. So the work is one hard phase and
one thin phase:

| Phase | Deliverable                                                       | User-facing method |
| ----- | ----------------------------------------------------------------- | ------------------ |
| 1     | The gradient spine: `state_grad` + symbolic forward sensitivities | none (builds it)   |
| 2     | `nuts` on `ode`                                                   | `nuts`             |

Two principles carried from the shipped half:

1. **Symbolic gradients only.** Derivatives come from source-to-source
   differentiation in the OCaml compiler, evaluated by the existing Rust
   `eval_resolved`. Finite differences appear **only** as the gradient-check
   oracle (Phase 1), never as a production path.
2. **A genuine capability gap is expressed in code, not omitted.** A model whose
   gradient depends on something we cannot differentiate (a state-dependent
   binomial denominator, a parameter-dependent event time) is **refused at
   dispatch** by the `DerivEntry::Unsupported` preflight — the single refusal
   path 3b made by subsuming the old `coeff_guard` (gh#119) — carrying the
   compiler's own reason string, never silently mishandled.

## What is already in place (verified against main)

The gradient-free ODE stack shipped and this proposal builds on it unchanged:

- **Real-valued ODE flow.** `Flows::Real(Vec<f64>)`
  (`rust/crates/sim/src/state.rs:110`) and the real-`acc` scoring path
  (`MultiStreamObsModel::fold_into_acc_real`,
  `log_likelihood_from_flows_and_counts_real`,
  `rust/crates/sim/src/inference/multi_stream_obs.rs:930,1155`) — incidence flow
  is continuous, no rounding to `u64`.
- **`mh` on `ode`.** `Stage::Mh` (`rust/crates/cli/src/fit/config_v2.rs:1189`),
  the `("mh","ode")` Beta entry in `methods.rs::METHODS`, and the `is_ode_mh`
  seam through `run_pmmh` (`rust/crates/cli/src/fit/pmmh.rs:153`).
- **`compute_ode_loglik`** (`rust/crates/cli/src/fit/runner.rs:775`) — solves
  the ODE, walks snapshots, accumulates real flows, scores via the shared seam.
- **`TemporalKind::{Interval, Instant}`**
  (`rust/crates/ir/src/observation.rs:36`) — the typed incidence/prevalence
  classifier the chain rule below pivots on.
- **`coeff_guard`** — already rejects NUTS fits that depend on a parameter
  buried inside an undifferentiated coefficient.
- **The NUTS sampler** (`rust/crates/sim/src/inference/nuts.rs`) — `nuts_step`
  takes `&dyn Fn(&[f64]) -> (f64, Vec<f64>)` and a `MassMatrix`; it contains no
  PGAS-specific code.
- **`rate_grad`** — `∂rate/∂θ` per transition, emitted by `autodiff.ml`, keyed
  by parameter name (`HashMap<String, Expr>`,
  `rust/crates/ir/src/transition.rs:53`). This is one of the two Jacobians the
  sensitivity equations need; the other (`∂rate/∂x`) does not exist and is the
  core of Phase 1.

## The gradient object

ODE-NUTS samples `p(θ | y) ∝ p(y | θ, ODE skeleton) · π(θ)` directly — no CSMC,
no discrete-event approximation, no Gibbs coupling. NUTS needs the gradient in
the unconstrained parameter space `z` (where `θ = θ(z)` via the per-parameter
`Log`/`Logit`/`None` transform):

```
∇_z [ log p(y | θ(z)) + log π(θ(z)) + log|dθ/dz| ]
```

The prior term `∇_z log π` and the change-of-variables term `∇_z log|dθ/dz|`
already have analytic gradients and are reused verbatim from PGAS
(`prior_log_density_and_grad_z`, `Transform::transform_deriv`/`log_jacobian`/
`jacobian_grad`; `rust/crates/sim/src/inference/pgas.rs:1752`, `.../types.rs`).
**The only new object is `∇_θ log p(y | θ)` for the ODE likelihood.**

### The chain rule — split by temporal kind

The likelihood is a sum over observations,
`log p(y|θ) = Σ_t log p(y_t | g_t(θ))`, where `g_t` is the projected observation
— a function of the trajectory. By the chain rule,

```
∇_θ log p = Σ_t (∂ log p(y_t | ·) / ∂g_t) · (∂g_t / ∂θ)
```

The first factor — the score with respect to the distribution mean — **exists
today** (`negbin_logpmf_grad`, `discretized_normal_logpmf_grad`,
`poisson_logpmf_grad`, `beta_binomial_logpmf_grad`,
`rust/crates/sim/src/inference/obs_loglik.rs`), composed with `∂mean/∂projected`
(the mean expression is a function of `projected`). The second factor,
`∂g_t/∂θ`, is **new**, and it is a _different object per temporal kind_:

- **Prevalence (`Instant`).** `g_t` reads compartment state `x_t`. Then
  `∂g_t/∂θ = (∂g_t/∂x) · S(t)`, where `S(t) = ∂x(t)/∂θ` is the forward
  sensitivity matrix.
- **Incidence (`Interval`).** `g_t` reads an accumulated-and-reset flow over the
  observation interval, `acc_k = ∫ stochₖ · rate(x(s), θ) ds`. Its sensitivity
  is a _separate accumulator_: `∂acc_k/∂θ = ∫ (J_θ,k + J_x,k · S(s)) ds`,
  integrated over the interval and **reset on the same per-stream schedule** as
  `reset_due_acc`. Chaining the obs score against `S(t)` for an incidence stream
  is silently wrong.

The gradient assembly therefore carries **two** sensitivity accumulators: `S(t)`
for `Instant` streams, and a per-`Interval`-slot `∂acc/∂θ` that integrates
`J_θ + J_x S` and resets per stream. `TemporalKind` already tells them apart at
bind time. Make that distinction a **type**, not a dispatch discipline:

```rust
enum Sensitivity {
    Prevalence(StateSens),    // S(t) = ∂x/∂θ; chains against ∂g/∂x
    Incidence(IntervalSens),  // per-Interval ∂acc/∂θ; chains against ∂g/∂acc, resets per stream
}
```

where the obs-score chain accepts only the matching variant. Then "chain the
score against `S(t)` for an incidence stream" is a **compile error**, not a
runtime-correctness obligation the assembly has to get right on every path. This
is the illegal-states-unrepresentable hardening of the gh#187 silent-gap class:
the type closes the hole the Phase-1 gate below can otherwise only catch after
the fact.

> **Gate (the gh#187 silent-gap class).** The gradient-check oracle (Phase 1)
> **must** include an incidence stream with ≥2 reset intervals. A
> prevalence-only check passes with the wrong incidence formula, because the
> reset term never fires.

Today the observation path treats `projected` as a constant input
(`obs_model.rs` evaluates the likelihood at a fixed projected value), so the
`∂g_t/∂(state or acc)` factor does not exist and is built here. For the common
projections — `FlowSum` (incidence) and `IntCompSum` (prevalence) — `g_t` is a
clean linear function of the accumulator / state, so `∂g_t/∂(state or acc)` is
immediate. A **parametric `DerivedExpr` projection** additionally needs the
`∂projected/∂θ` term. That term is supplied by the unified obs-gradient
authority (`docs/dev/proposals/2026-07-03-unified-obs-gradient-autodiff.md`,
gh#180): every differentiable expression is sealed behind a
`DerivEntry = Grad | Unsupported{reason}` ADT, and a fit-time preflight admits
the now-differentiable parametric projection (`qgam`) and refuses only the
genuinely-uncovered cases with the compiler's own reason string. **`det_grad`
consumes that shared obs-gradient seam (2026-07-03 §4.3), it does not
re-derive** — re-deriving the obs gradient inside ODE-NUTS would replace the
`eval_resolved_deriv` fork the authority just removed with a fresh PGAS/ODE
fork. The state-derivative chain built here (`∂g/∂(state or acc)`) and the
parameter-derivative chain from the authority (`∂projected/∂θ`) are orthogonal
factors of the same projection, along different variables — one differentiation
engine, evaluated twice.

## Phase 1 — The gradient spine

No user-facing method ships in Phase 1; it builds `det_grad`, the function that
returns `(log p(y|θ), ∇_θ log p(y|θ))`, once, for Phases 2 (and any future
gradient-MLE).

**Land first (sequenced prerequisites).** Two pieces land _before_ this phase,
on their own branches — do not develop them concurrently with `state_grad`, they
share `autodiff.ml` and would collide:

1. **3b (gh#342)** — derive the differentiation traversal
   (`#[derive(Differentiate)]`) and fold the rate path onto
   `Diffable`/`DerivEntry`, retiring `rate_grad: HashMap<_, Expr>` and subsuming
   `coeff_guard`. `state_grad` is then a `Diffable` position on the sealed
   traversal (§1a/§1b), not a parallel emitter — the whole point of doing 3b
   first is that `state_grad` cannot fork.
2. **gh#374** — the `Log`/`Logit` transform Jacobian/derivative at the
   `[lo, hi]` bound. NUTS explores to parameter bounds; a wrong Jacobian there
   produces boundary-clustered divergences, which is exactly the failure the
   Phase-2 gate below treats as a canary. Landing gh#374 first removes a known
   cause so the canary stays meaningful.

### 1a. `state_grad` from the compiler — generalize the differentiation target

`autodiff.ml` differentiates a rate expression with respect to a named parameter
(`differentiate`, `ocaml/lib/ir/autodiff.ml:171`), and the emitted `rate_grad`
gives `J_θ`'s ingredient `∂rates/∂θ`. The sensitivity equations also need
`J_x`'s ingredient `∂rates/∂x` — `∂rate/∂Pop(Cₖ)`. This is a new **target** on
the one differentiation engine, not a second emitter:

```ocaml
type diff_target =
  | WrtParam of string   (* existing rate_grad: ∂rate/∂param *)
  | WrtPop   of string   (* new state_grad:  ∂rate/∂compartment *)
```

Because 3b (gh#342, a prerequisite) has already derived the differentiation
traversal and folded the rate path onto `Diffable`/`DerivEntry`, `state_grad`
enters as another `Diffable` position the derived traversal covers — coverage by
type, not a hand-maintained parallel pass. **Consolidation to bank while here:**
`ocaml/lib/ir/lineage.ml`'s `deriv_num_wrt_pop` (`:182`) is a hand-rolled
`WrtPop` differentiator — it reimplements the product/quotient/sum rules to read
a parent-pool weight. It is a _specialization_ (numerator only, treating the
normalizer as frozen at the event instant, and assuming degree-1 linearity), so
it is not `state_grad`; but once `WrtPop` exists it should be re-expressed as
`differentiate(numerator, WrtPop c)` over its existing `split_frac` numerator,
deleting the duplicate recursion. One engine, two callers with different inputs.

This is **not** a one-line toggle. Two parts of the current code bake in the
"state is constant" premise — correct for PGAS (where the trajectory is fixed in
the θ|X step), wrong for an ODE (where the trajectory is a function of θ):

- **The fused base case** (`autodiff.ml:146`):
  `Const | Pop _ | PopSum _ | Time | Dt | Projected | ObsColumnRef _ → Const 0.0`
  must be un-fused. For `WrtPop name`: `Pop n → [n = name]`,
  `PopSum members → [name ∈ members]` (the force-of-infection / coupling terms —
  the source of the off-diagonal `J_x`), everything else → 0. Mirror in
  `mentions` (`autodiff.ml:57`).
- **`BindingRef`** is the subtle part. Hoisted bindings are today zeroed and
  asserted state-only/param-free (`autodiff.ml:66,325`; the param-free invariant
  is enforced at dimcheck/expr-analysis as E512). For `WrtPop` the premise
  _inverts_ — bindings are functions of state, so `∂binding/∂x` is generally
  nonzero (a hoisted FOI binding is exactly where the coupling lives). `WrtPop`
  must thread the model binding table into `differentiate`, resolve-and-recurse
  through binding bodies **with cycle protection** (none exists today), and
  reconcile the invariant so it holds for `WrtParam` but not `WrtPop`. Forcings
  and tables remain state-free (`∂/∂x = 0`), so `WrtPop` is simpler there — and
  this is why seasonal forcing does not complicate `J_x` (see Malaria fitness).

### 1b. IR schema change (atomic)

`state_grad` is a new per-transition differentiable **position** — a `Diffable`
field carrying its `DerivEntry` per compartment name, the same literal type 3b
folded the rate path onto (do **not** reintroduce a parallel
`HashMap<String, Expr>` — that is the representation 3b just retired). The
version guard hard-rejects any mismatch (`rust/crates/ir/src/envelope.rs`), so
there is no backward-compat path: bumping `ir/VERSION` `0.24 → 0.25` regenerates
every golden atomically and old IR no longer loads. The "model lacks a coverable
`state_grad` → no gradient method" case is therefore the
**`DerivEntry::Unsupported` fit-time preflight** (which 3b already made the
single refusal path, subsuming `coeff_guard`) on the current-version model, not
deserialization tolerance.

Procedure (per CLAUDE.md "Changing the IR schema"): `ir/schema.json` +
`ir/VERSION` → OCaml `ocaml/lib/ir/{ir,serialize,deserialize}.ml` → Rust
`rust/crates/ir/src/transition.rs` → `make test-fast` →
`make update-golden &&
make update-expected` → one atomic, human-reviewed commit
with the golden diff called out in the subject.

### 1c. Forward sensitivities — one augmented, shared-stage RK4

`S` solves `Ṡ = J_x S + J_θ` alongside `ẋ = f(x, θ)`, with `f = stoich · rates`,
`J_x = stoich · (∂rates/∂x) ∈ ℝⁿˣⁿ`, `J_θ = stoich · (∂rates/∂θ) ∈ ℝⁿˣᵈ` (`n`
compartments, `d` estimated params). Multiplying the symbolic `state_grad` /
`rate_grad` by stoichiometry is a trivial Rust matmul.

**Decision: a single augmented `(x, S)` system integrated by one RK4 whose four
stages share intermediate states** (the answers.md "Option B", stacked into one
`Vec`). This is required, not a preference: NUTS simulates a frictionless
Hamiltonian trajectory and relies on energy conservation; if `S` is advanced by
a decoupled endpoint step it is only first-order accurate while the state is
`O(dt⁴)`, the gradient is `O(dt)`-inconsistent with the value, and NUTS produces
spurious divergences. Sharing the four stage evaluations makes `J_x`, `J_θ`, and
the `S`-stages advance in lockstep with the state stages. `OdeSim` is already
generic over a `Vec<f64>` state (`rust/crates/sim/src/ode.rs`; dimension comes
from `model.compartments.len()` at runtime), so the `n → n + n·d` augmentation
is a larger allocation, not type surgery. Forward mode is `O(d)`; adjoint
(`O(1)` in `d`) is out of scope — see below.

### 1d. Continuous prevalence evaluation

Phase 0 made incidence flow real-valued, but prevalence still scores through
rounded `i64` counts (`ode.rs` rounds state at snapshot time via `to_states`).
For a smooth gradient — and for low-count prevalence-observed models — the
prevalence projection must read the `f64` `int_vals` via the existing
`EvalCtx.int_float_override` pattern (`ode.rs`), through a
`log_likelihood_continuous(real_counts: &[f64], …)` entry. Two distribution
notes, carried forward because they are load-bearing: camdl's `Normal` is the
**discretized-count** likelihood (a Φ-difference), smooth in its mean `μ` (which
is all the gradient needs) but not a continuous PDF — the gradient is correct,
the framing must be accurate; and `Binomial`/`BetaBinomial` treat the
denominator `n` as a rounded constant, so a **state-dependent** denominator
(e.g. `n = S+I+R`) cannot yield a smooth gradient — `coeff_guard` rejects it
under gradient methods rather than silently dropping the term.

### 1e. Crate relocation

`compute_ode_loglik` lives in `cli` (`rust/crates/cli/src/fit/runner.rs:775`)
but references only `sim` types and the `sim` scoring seam. The gradient
assembler (`det_grad`) must live in `sim`, so Phase 1 begins by moving
`compute_ode_loglik` down into `sim` and repointing its `cli` callers
(`profile.rs`, `survey.rs`, `nlopt_stage.rs`). (Aside: CLAUDE.md's "cli → io →
observe → sim → ir" is stale — there is no `observe` crate; the real layering is
`cli → io → sim → ir`.)

### 1f. Gradient-check oracle (the only finite difference in the system)

Extend `gradient_check.rs`: assert `‖∇_symbolic − ∇_FD‖_∞ < 10⁻⁴` across all
estimated params, on a model that exercises every new path at once —

- an **incidence stream over ≥2 reset intervals** (the per-`Interval`
  sensitivity and its reset),
- a **hoisted binding used in a rate** (the `BindingRef` state-diff),
- a **parameter-dependent event** (the sensitivity jump at an event).

Red-then-green per CLAUDE.md TDD: write the check against a deliberately wrong
`J_x` first, confirm it fails, then land the correct emission and confirm it
passes. Finite differences live here and nowhere else. The same FD machinery is
what gh#78's user-facing `--check-grads` would expose at the model's use site;
this oracle is its dev-test ancestor.

### Phase 1 gates

`det_grad` agrees with the FD oracle to `10⁻⁴` on the combined model; the PGAS
gradient is **byte-identical** before and after (the state-derivative arm must
not perturb the PGAS path, which relies on `∂rate/∂x` being zeroed — distinct
entry points, gated by a PGAS byte-identity test); `make test` green with the
golden diff from the schema bump reviewed.

## Phase 2 — `nuts` on `ode`

With `det_grad` in hand, NUTS is a thin wiring layer plus two extractions. The
sampler core does not change. The two pieces PGAS currently inlines are lifted
out **here** (ODE-NUTS is their only second consumer), each behind a hard
PGAS-byte-identity gate:

- **`nuts_warmup.rs`** — the dual-averaging step-size + Welford mass-matrix
  adaptation block (`pgas.rs:~2453–2514`). In PGAS it is per-rung state
  entangled with parallel tempering (`rungs[rung].*`, the
  `mass_adapt_end = 0.7·adapt_end` schedule, cold-rung prints). The extracted
  `NutsWarmup` is instantiated per-rung and parameterized on cold-rung identity;
  PGAS instantiates one per rung, ODE-NUTS instantiates exactly one.
- **`posterior_target.rs`** — the prior + transform + Jacobian wrapper
  (`pgas.rs:1752` + the closure at `~2362–2400`). The β-tempering and the
  data-term are lifted to a pluggable θ-space data-term: PGAS plugs in
  `complete_data_loglik_grad`, ODE-NUTS plugs in `det_grad`. Same byte-identity
  gate.

Then `nuts_stage.rs` is a thin driver:
`build_nuts_target(det_grad, priors,
transforms)` → a single `NutsWarmup` over a
contiguous warmup window → sample with `nuts_step`. Registry: a `("nuts","ode")`
entry in `methods.rs::METHODS` and a `Stage::Nuts` dispatch arm, removing the
generic rejection for that cell.

### Phase 2 gates

The ODE-NUTS posterior agrees with an **independent** high-ESS reference (not
merely with the Phase-1 MH chain, which on a curved ridge may not have
converged), and with PGAS in the low-noise / large-population limit where the
two likelihoods coincide. Divergence rate is low and **not** clustered at
integer or parameter-boundary values (a cluster there signals the
continuous-eval or transform-Jacobian arm is wrong). PGAS posteriors remain
byte-identical after the two extractions.

## Malaria fitness

This stack is well-matched to the malaria models on the roadmap:

- **Dimension.** Forward sensitivities are `O(d)` in the parameter count, which
  is fine for the `d ≲ 30` typical malaria fit. NUTS's advantage over
  random-walk MH _grows_ with dimension and with posterior correlation — exactly
  the regime a multi-compartment malaria model (humans
  `S`/`I`/asymptomatic/treated, mosquito stages, seasonal drivers) enters.
- **Seasonal forcing is free.** Rainfall- and temperature-driven forcing terms
  are **state-free**, so they contribute nothing to the new `J_x`
  (`∂forcing/∂x =
  0`); only their parameter derivatives matter, and those
  already exist as `rate_grad`. Seasonality does not complicate the hard part of
  Phase 1.
- **Stiffness.** Malaria is far less stiff than the TB-latency case (no
  `10⁻⁶/day` reactivation timescale); explicit RK4 — or the adaptive Dopri5
  integrator already implemented — is appropriate, and the deterministic
  `dt`-convergence check guards against silent under-resolution.

The ceiling to know about: a **hierarchical** malaria fit sharing
hyperparameters across many sites pushes `d` into the hundreds, where forward
mode becomes prohibitive and adjoint sensitivities (`O(1)` in `d`) become
mandatory. That is the named follow-up below, not this proposal.

## Out of scope (named follow-ups)

- **Gradient-based MLE (`nl-lbfgs`, `nl-slsqp`).** A free payoff of the same
  `det_grad` — fill NLopt's gradient slice instead of discarding it — but **not
  on the NUTS critical path**. File as a separate issue once `det_grad` lands;
  the only change is widening the `optimize_det` seam to
  `FnMut(&[f64], Option<&mut [f64]>) -> f64`.
- **Adjoint-mode sensitivities.** Required for hierarchical malaria (`d` in the
  hundreds); a backward solve + checkpointing. Forward mode is the simpler
  correct choice for `d ≲ 30`. → gh issue, blocked on a concrete hierarchical
  model.
- **Stiff (implicit) ODE solvers.** The right long-term answer for TB latency,
  not needed for malaria.
- **Model comparison for ODE fits** (gh#312). Independent of gradients: emit an
  ODE posterior-predictive `PrequentialTrace` so the existing `camdl compare`
  consumes ODE fits (builds on the gh#277 emitter, tagged per gh#295). Works on
  today's `mh`-on-`ode` posteriors — no gradient needed.
- **Reactive interventions / parameter-dependent event times under gradient
  methods.** Gated out via `coeff_guard`, not mishandled.
- **`--method auto`.** Choosing between the deterministic and stochastic
  likelihoods is too high-stakes for silent selection.

## Risks

- **`BindingRef` state-differentiation** is the subtlest part of the autodiff
  change (inverted invariant + cycle protection); the gradient-check on a model
  using a binding in a rate is its gate.
- **Incidence-sensitivity reset** is the subtlest part of the gradient assembly;
  the gradient-check on an incidence stream with ≥2 reset intervals is its gate.
- **Continuous-eval regression into PGAS.** Enabling the state-derivative arm of
  the obs gradient must not perturb the existing PGAS gradient (which relies on
  it being zeroed); distinct entry points, gated by the PGAS byte-identity test.
- **Order-mismatched sensitivities.** A decoupled `S` step silently degrades
  NUTS Hamiltonian conservation; the shared-stage RK4 (1c) is the mitigation,
  and the "divergences not clustered at boundaries" Phase-2 gate is the canary.
