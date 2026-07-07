---
status: implemented
date: 2026-06-26
tracking: gh#275
supersedes: docs/dev/proposals/2026-06-15-ode-gradient-inference.md (Phases 0–1 shipped; this is the gradient half)
prerequisites: gh#342 (3b — derive the obs-gradient traversal and fold the rate path onto the classified GradMap/DerivEntry — LANDED); gh#374 (the Log{lo,hi} transform derivative/Jacobian at the bound; the Logit{lo,hi} case is already consistent); docs/dev/proposals/2026-07-03-unified-obs-gradient-autodiff.md (the one differentiation authority — §4.3 shared obs-gradient seam, §10 coordination, §11 = 3b)
related: gh#180 (obs ∂projected/∂θ term — LANDED), gh#78 (runtime --check-grads)
ir_version: 0.26 → 0.27 (rate_state_grad, a per-transition CompGradMap keyed by compartment, + ic_grad, ∂init/∂θ per parameterized IC compartment — the rate_grad siblings; see §1b)
code-baseline: this proposal targets the post-gh#342 (3b) tree — PR #385 / branch worktree-sealed-grads (ir/VERSION 0.26, rate_grad a classified GradMap). It does NOT describe pre-3b main. Re-pin to the 3b merge commit once #385 lands.
---

# ODE + NUTS: the gradient spine and a gradient-based Bayesian sampler

**Status: implemented** on branch `gh275-ode-nuts-gradient-spine` (not yet
merged). The Phase 1 gradient spine, Phase 2 `nuts`-on-`ode`, and the CLI method
are shipped; all four Risks are closed — the `ic_grad` seed (estimating initial
conditions) was the last, in `8620f5ac` (Rust consumer) and `d113a764` (OCaml
emission). A well-specified ODE model now fits by NUTS with pgas-parity on
trajectory, observation, and initial-condition parameters. Two follow-ups
remain, both tracked: gh#390 (§1g event-jump sensitivity — models with scheduled
interventions still refuse `nuts`) and gh#374 (the `Log{lo,hi}`
bounded-parameter Jacobian at the clamp, a pre-existing cross-method transform
issue).

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

| Phase | Deliverable                                                                                                          | User-facing method |
| ----- | -------------------------------------------------------------------------------------------------------------------- | ------------------ |
| 1     | The gradient spine: `rate_state_grad` + `ic_grad` seed + forward sensitivities + the gradient-method capability gate | none (builds it)   |
| 2     | `nuts` on `ode` (deterministic-likelihood only)                                                                      | `nuts`             |

Two principles carried from the shipped half:

1. **Symbolic gradients only.** Derivatives come from source-to-source
   differentiation in the OCaml compiler, evaluated by the existing Rust
   `eval_resolved`. Finite differences appear **only** as the gradient-check
   oracle (Phase 1), never as a production path.
2. **A genuine capability gap is expressed in code, not omitted.** A model whose
   gradient depends on something we cannot differentiate is **refused before
   sampling**, carrying the compiler's own reason string, never silently
   mishandled. Post-3b this refusal is **two** mechanisms, not one: the
   `DerivEntry::Unsupported` fit-time preflight (rate / obs / σ² coefficients —
   a Periodic step value, a `lag`, a non-constant table index) and the narrowed
   `coeff_guard` (a coefficient reached _only_ through an initial condition,
   gh#342 P4b). **Neither currently covers the gaps this proposal introduces** —
   a state-dependent binomial denominator (`n` is θ-independent by contract and
   carries no `DerivEntry`, so no scan sees it), a parameter-dependent event
   time, or a model lacking a coverable `rate_state_grad`. Those refusals are
   **built here** (§1b, §1d) and must run on the ODE-NUTS dispatch path, which —
   by design (no CSMC, no Gibbs coupling) — does **not** route through
   `run_pgas` where the existing preflight lives (§2). Wiring the refusal to the
   new cell is part of the work, not inherited.

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
- **`coeff_guard`** (`rust/crates/cli/src/fit/coeff_guard.rs`) — post-3b P4b,
  narrowed to the **initial-condition domain**: it refuses a NUTS fit whose
  parameter reaches a forcing/table coefficient _only_ through an `init`
  expression. It does **not** cover rate/obs coefficients (those moved to the
  `DerivEntry::Unsupported` preflight) and does **not** cover a state-dependent
  binomial `n`. Its own header notes that IC / state sensitivity is deferred to
  "the separate gh#275 surface" — i.e. this proposal owns the `init`-parameter
  gradient (see §1c, C-seed).
- **The NUTS sampler** (`rust/crates/sim/src/inference/nuts.rs`) — `nuts_step`
  takes `&dyn Fn(&[f64]) -> (f64, Vec<f64>)` and a `MassMatrix`; it contains no
  PGAS-specific code.
- **`rate_grad`** — `∂rate/∂θ` per transition, emitted by `autodiff.ml`, now a
  **classified `GradMap`** keyed by parameter name
  (`GradMap = HashMap<String,
  DerivEntry>` where
  `DerivEntry = Grad(Expr) | Unsupported{node,code}`;
  `rust/crates/ir/src/transition.rs:135`, `deriv.rs`). This is one of the two
  Jacobian ingredients the sensitivity equations need; the other (`∂rate/∂x`)
  does not exist and is the core of Phase 1. **`rate_state_grad` is its
  sibling** — the same `GradMap` shape, keyed by _compartment_ instead of
  parameter (§1b).

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
`jacobian_grad`; `rust/crates/sim/src/inference/pgas.rs:1859`, `.../types.rs`).
**The only new object is `∇_θ log p(y | θ)` for the ODE likelihood.**

### The chain rule — split by temporal kind

The likelihood is a sum over observations,
`log p(y|θ) = Σ_t log p(y_t | g_t(θ))`, where `g_t` is the projected observation
— a function of the trajectory. By the chain rule,

```
∇_θ log p = Σ_t (∂ log p(y_t | ·) / ∂g_t) · (∂g_t / ∂θ)
```

The first factor — the score with respect to **each** distribution argument —
**exists today** (`negbin_logpmf_grad`, `discretized_normal_logpmf_grad`,
`poisson_logpmf_grad`, `beta_binomial_logpmf_grad`,
`rust/crates/sim/src/inference/obs_loglik.rs`, each returning the partials
w.r.t. every argument, not just the mean). It must be summed over every argument
that depends on `projected`, not composed with `∂mean/∂projected` alone:

```
∂ log p(y_t | ·) / ∂g_t = Σ_arg (∂ log p / ∂arg) · (∂arg / ∂projected)
```

This is load-bearing: the flagship discretized-Normal uses the He et al.
mean-linked variance `ρ·C·(1 − ρ + ψ²·ρ·C)` with `C = projected`, so its `sd` is
a strong function of `projected` — reducing the factor to `∂mean/∂projected`
drops the entire variance-through-`C` term and is **silently wrong for the
default obs model**. (The existing PGAS obs-grad seam already accumulates over
all arguments — `d_mu·dm + d_sd·ds` in `obs_model.rs`; ODE-NUTS builds the same
reduction, but against `∂arg/∂projected`.) The second factor, `∂g_t/∂θ`, is
**new**, and it is a _different object per temporal kind_:

- **Prevalence (`Instant`).** `g_t` reads compartment state `x_t`. Then
  `∂g_t/∂θ = (∂g_t/∂x) · S(t)`, where `S(t) = ∂x(t)/∂θ` is the forward
  sensitivity matrix, **seeded at `t_start` from the initial condition**:
  `S(t_start) = ∂(initial_state)/∂θ`, non-zero for any `init`/`ivp` parameter
  (§1c — omitting this seed is silent-zero for `i0`/`e0`/`s0`). For a linear
  projection `∂g_t/∂x` is a 0/1 selection; for a `DerivedExpr` prevalence it is
  a general expression derivative (see below), not a selection.
- **Incidence (`Interval`).** `g_t` reads an accumulated-and-reset flow over the
  observation interval, `acc_k = Σ_{i∈slot_k} ∫ rate_i(x(s), θ) ds`, where the
  flow selection `slot_k` is `interval_slots.flow_indices`. Its sensitivity is a
  _separate accumulator_ over the **flow-selected raw-rate derivatives** — not
  the compartment-stoich Jacobian of §1c:
  `∂acc_k/∂θ = Σ_{i∈slot_k} ∫ (∂rate_i/∂θ + Σ_j ∂rate_i/∂x_j · S_j(s)) ds`,
  integrated over the interval and **reset on the same per-stream schedule** as
  `reset_due_acc`. (Reusing §1c's `J_θ = stoich·∂rate/∂θ` for the accumulator
  computes a net-compartment-change sensitivity, not an incidence-flow one — a
  wrong gradient; keep the two `J` uses distinct.) Chaining the obs score
  against `S(t)` for an incidence stream is silently wrong.

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
`∂g_t/∂(state or acc)` factor does not exist and is built here.

For a **linear projection** — `FlowSum` (incidence) and `IntCompSum` /
`CurrentPop` (prevalence) — `g_t` is a 0/1-weighted function of the accumulator
/ state, so `∂g_t/∂(state or acc)` is a selection, immediate. A **`DerivedExpr`
prevalence projection** (`Instant` — e.g. a proportion `I/(S+I+R)`) is _not_
linear: `∂g_t/∂x = Σ_k (∂h/∂x_k)·S_k` needs `∂h/∂x_k`, a general **expression**
derivative of the _projection_ expression. That is a new `WrtPop`-differentiated
object (§1a), distinct from the per-transition `rate_state_grad`, which
differentiates _rates_, not projection expressions. So a `DerivedExpr`
prevalence stream must either emit `∂DerivedExpr/∂x` **or be refused** — it must
not silently fall through the linear-selection arm. The `Sensitivity` type above
splits `Instant` from `Interval` but not linear-from-general _within_ `Instant`;
closing that is a Phase-1 obligation, gated by an oracle model with a
`DerivedExpr` prevalence stream.

A **parametric `DerivedExpr` projection** additionally needs the `∂projected/∂θ`
term — the orthogonal factor along θ (not x). That term is the unified
obs-gradient authority's
(`docs/dev/proposals/2026-07-03-unified-obs-gradient-autodiff.md`, gh#180 / 3b):
every differentiable expression is sealed behind a
`DerivEntry = Grad | Unsupported{reason}` ADT, and the fit-time preflight admits
the now-differentiable parametric projection (`qgam`) and refuses the
genuinely-uncovered cases with the compiler's own reason string. **`det_grad`
consumes that shared seam — `resolve_grad_map` + `eval_emitted_grad`
(`resolved_expr.rs`), the same path PGAS's gradient uses — it does not
re-derive.** Re-deriving the obs gradient inside ODE-NUTS would re-fork the
evaluator 3a/3b just unified. The state-derivative chain built here
(`∂g/∂(state or acc)`) and the parameter-derivative chain from the authority
(`∂projected/∂θ`) are orthogonal factors of the same projection, along different
variables — one differentiation engine, evaluated twice.

## Phase 1 — The gradient spine

No user-facing method ships in Phase 1; it builds `det_grad`, the function that
returns `(log p(y|θ), ∇_θ log p(y|θ))`, once, for Phases 2 (and any future
gradient-MLE).

**Land first (sequenced prerequisites).** Two pieces land _before_ this phase,
on their own branches — do not develop them concurrently with `rate_state_grad`,
they share `autodiff.ml` and would collide:

1. **3b (gh#342) — LANDED.** It folded the rate path onto the classified
   `GradMap` (`= HashMap<String, DerivEntry>`, not `Diffable`; `Diffable` is the
   obs-argument wrapper only) and made the `DerivEntry::Unsupported` preflight
   the rate/obs refusal path, narrowing `coeff_guard` to the initial-condition
   domain. `rate_state_grad` is therefore the **`GradMap` sibling of
   `rate_grad`** (§1b), and it is emitted by a **new hand-written producer
   pass** — 3b's Rust-side `#[derive(Differentiate)]` covers the obs `Diffable`
   fields for the preflight scan/hash, but the OCaml `∂rate/∂θ` producer is a
   plain functional update (`{ t with rate_grad }`), not a sealing
   reconstruction. So a `rate_state_grad` pass can silently forget a transition;
   its producer and a coverage check must be designed explicitly (§1a), not
   assumed free by type.
2. **gh#374** — the `Log{lo,hi}` transform derivative/Jacobian at the bound (the
   `Logit{lo,hi}` case is already consistent, so only `Log{lo,hi}` is the gap).
   NUTS explores to parameter bounds; a wrong Jacobian there produces
   boundary-clustered divergences — exactly the failure the Phase-2 gate treats
   as a canary. Landing gh#374 first removes a known cause so the canary stays
   meaningful.

### 1a. `rate_state_grad` from the compiler — generalize the differentiation target

`autodiff.ml` differentiates a rate expression with respect to a named parameter
(`differentiate`, `ocaml/lib/ir/autodiff.ml:171`), and the emitted `rate_grad`
gives `J_θ`'s ingredient `∂rates/∂θ`. The sensitivity equations also need
`J_x`'s ingredient `∂rates/∂x` — `∂rate/∂Pop(Cₖ)`. This is a new **target** on
the one differentiation engine, not a second emitter:

```ocaml
type diff_target =
  | WrtParam of string   (* existing rate_grad: ∂rate/∂param *)
  | WrtPop   of string   (* new rate_state_grad:  ∂rate/∂compartment *)
```

The single recursive `differentiate` engine is shared, but coverage is **not**
free "by type": the OCaml differentiation surface is four hand-written driver
passes (`differentiate_rate`, `differentiate_likelihood`,
`differentiate_observations`, `differentiate_overdispersion`, `autodiff.mli`),
and `rate_state_grad` is a **fifth** — a new `WrtPop` producer over the
transitions, plus its own resolve/preflight wiring. The win from doing 3b first
is that the recursive kernel and the `DerivEntry` classification already exist
to reuse; it is not that forgetting `rate_state_grad` becomes a compile error
(it does not — see "Land first" above).

**Do _not_ fold `lineage.ml`'s `deriv_num_wrt_pop` into this.** It looks like a
hand-rolled `WrtPop` differentiator, but it is **not** behavior-equivalent to
`differentiate(_, WrtPop c)` and re-expressing it would silently change a
shipped individual-sampling weight. `deriv_num_wrt_pop` (`lineage.ml`) _freezes
the denominator_ — its `Div` arm is `∂(f/g)/∂count = f'/g`
(`lineage.ml:201-206`), the parent-pool weight lineage intends — whereas the
general `differentiate` applies the full quotient rule `(f'g − fg')/g²`
(`autodiff.ml` Div arm). They disagree the moment the normalizer depends on the
compartment: for `β·S·(a + I/N)` with `N = PopSum[S;I;R]`, `deriv_num_wrt_pop I`
gives `β·S/N` (intended) while `differentiate(_, WrtPop I)` gives `β·S·(S+R)/N²`
(`∂N/∂I = 1`). `split_frac` does not guarantee the numerator is free of a
compartment-dependent `Div` (one nested under `Add`/`Cond`/`Pow` survives), so
the consolidation is a latent correctness regression on the lineage path, not a
cleanup. Keep them distinct.

This is **not** a one-line toggle. Several parts of the current code bake in the
"state is constant" premise — correct for PGAS (where the trajectory is fixed in
the θ|X step), wrong for an ODE (where the trajectory is a function of θ):

- **The fused base case** (`autodiff.ml:192`, post-3b it returns the classified
  `Known (Const 0.0)`, not a bare `Const 0.0`):
  `Const | Pop _ | PopSum _ | Time | Dt | Projected | ObsColumnRef _ → Known (Const 0.0)`
  must be un-fused. For `WrtPop name`: `Pop n → Known (Const 1.0)` iff
  `n = name`; `PopSum members → Known (Const 1.0)` iff `name ∈ members`
  (`PopSum of string
  list` is the fully-expanded member list — gh#185 `where`
  is expanded at compile time — so `∂PopSum/∂Pop(name) = [name ∈ members]` is
  well-defined; this is the force-of-infection / coupling term, the source of
  the off-diagonal `J_x`); everything else → `Known (Const 0.0)`. `mentions`
  (`autodiff.ml:78`) returns `false` for `Pop`/`PopSum`/`BindingRef`, and it
  gates the `Mod`/non-const-index guards at three sites — a `mentions_pop` is
  not a mechanical mirror; it must additionally recurse into referenced binding
  bodies (below).
- **`WrtPop` needs its own classification policy**, not the `WrtParam` one. The
  `Known | Omitted | Unsupported` arms and the E600 driver are all worded and
  scoped for _parameters_ ("parameter '%s' drives …"); for `WrtPop` they are
  dead. And the "forcings and tables are state-free" blanket has a broader hole
  than tables: **any nonsmooth function of state** is undifferentiable, not
  zero, and must be a `WrtPop` refusal, not a silent `Const 0.0` — a
  state-indexed `TableLookup(t, [Pop c])` (discrete cell-selection),
  `Mod`/`Floor`/`Ceil` of a `Pop`, a `Cond` predicate on a `Pop`, and
  `Min`/`Max`/`Abs` of state unless explicitly accepted as piecewise-smooth.
  These route through the §1h gate. Forcings _are_ state-free (`∂/∂x = 0`), so
  seasonal forcing genuinely does not complicate `J_x` (see Malaria fitness) —
  the nonsmooth-of-state constructs are the exceptions the driver must catch.
- **`BindingRef`** is the subtle part. Hoisted bindings are today zeroed
  (`autodiff.ml:90` in `mentions`, `:422` in `differentiate`) and asserted
  param-free (enforced as **E512**, `compiler.ml`). For `WrtPop` the premise
  _inverts_ — bindings are functions of state (a hoisted FOI binding is exactly
  where the coupling lives), so `∂binding/∂x` is generally nonzero. `WrtPop`
  must thread the model binding table into `differentiate` and
  resolve-and-recurse through binding bodies. No cycle guard is needed: model
  bindings are **acyclic by construction** — topologically ordered and enforced
  (`ir.ml`, `validate.ml`, `time_typing.ml`), so following that order
  terminates. E512 needs no reconciliation either: it stays a hard error for all
  models, so bindings remain param-free (correct `Const 0.0` for `WrtParam`)
  _and_ state-bearing (the `WrtPop` recursion target) simultaneously — no fork
  of the invariant. **Memoize** the binding derivatives —
  `(binding_id, WrtPop c) → DerivEntry` and
  `(binding_id, WrtParam p) →
  DerivEntry` — because a shared FOI binding
  differentiated afresh for every transition × compartment explodes expression
  size; a shared FOI is exactly the common case. Keep a debug-only visited-set /
  fuel guard as cheap insurance against hand-written IR or future validator
  drift, even though the invariant says it can never trip.

### 1b. IR schema change (atomic)

Two new compiler-emitted objects, both the classified `DerivEntry`-valued map
shape (`HashMap<String, DerivEntry>`) 3b established for `rate_grad` — **not**
`Diffable`, which bundles an `expr` a gradient map does not carry, and would
duplicate the rate expression already on `Transition::rate`:

- **`rate_state_grad`** — a new per-transition field, `∂rate/∂compartment`,
  keyed by **compartment** name (the `rate_grad` sibling, which is keyed by
  parameter). It is `J_x`'s ingredient. Name it `rate_state_grad`, **not**
  `state_grad`: it is `∂rate/∂state`, and a bare `state_grad` reads as
  `∂state/∂θ` — the runtime forward sensitivity `S(t)` (`StateSens`), a
  different object. The collision is a future-bug magnet; keep the two names
  apart.
- **`ic_grad`** — `∂(initial_state)/∂θ`, keyed by parameter, per parameterized
  initial-condition compartment. It is the forward-sensitivity **seed**
  `S(t_start)` (§1c, C-seed); without it every `init`/`ivp` parameter gets an
  identically-zero gradient — the exact IC/state surface `coeff_guard` defers to
  this proposal. It is a `WrtParam` differentiation of the
  `InitialConditions::Parameterized` expressions: the same engine, a new
  producer over the IC map.

**Newtype the two map keyings, do not share one `GradMap`.** `rate_grad` and
`ic_grad` are parameter-keyed; `rate_state_grad` is compartment-keyed. Give them
distinct types — `ParamGradMap` and `CompGradMap` over the shared `DerivEntry`
value — so that resolving a compartment-keyed map with the parameter resolver
(or vice-versa) is a **compile error**, not a silent index-by-the-wrong-name.
This is the illegal-states-unrepresentable form of the `∂rate/∂θ` vs `∂rate/∂x`
split the sensitivity assembly depends on.

The version guard hard-rejects any mismatch (`rust/crates/ir/src/envelope.rs`),
so there is no backward-compat path: bumping `ir/VERSION` **`0.26 → 0.27`**
regenerates every golden atomically and old IR no longer loads. "The model needs
a gradient this build cannot emit" is then a **fit-time** refusal on the
current-version model, not deserialization tolerance — but that refusal must be
_wired to the ODE-NUTS dispatch_, which does not route through `run_pgas` where
the existing preflight lives (§2, principle 2); building it is part of Phase 1.

Procedure (per CLAUDE.md "Changing the IR schema"): `ir/schema.json` +
`ir/VERSION` → OCaml `ocaml/lib/ir/{ir,serialize,deserialize}.ml` → Rust
`rust/crates/ir/src/transition.rs` (+ the `ic_grad` carrier) → `make test-fast`
→ `make update-golden && make update-expected` → one atomic, human-reviewed
commit with the golden diff called out in the subject.

### 1c. Forward sensitivities — one augmented, shared-stage fixed-step RK4

`S(t) = ∂x(t)/∂θ` solves `Ṡ = J_x S + J_θ` alongside `ẋ = f(x, θ)`, with
`f = stoich · rates`, `J_x = stoich · (∂rates/∂x)`, `J_θ = stoich · (∂rates/∂θ)`
(`n` compartments, `d` estimated params). **Assemble `Ṡ` transitionwise and
sparse — do _not_ materialize a dense `J_x ∈ ℝⁿˣⁿ` and matmul.** A dense `n×n`
Jacobian is the wrong abstraction for a stratified model: at national scale `n`
is in the hundreds–thousands and `n²` is exactly the bottleneck the
sparse-coupling work exists to avoid. Instead, for each transition `r` with
stoichiometry `stoich_r`, its `rate_grad_r` (`∂rate_r/∂θ`, `ParamGradMap`) and
its `rate_state_grad_r` (`∂rate_r/∂x`, `CompGradMap`):

```
total_dr_dθ[p] = rate_grad_r[p] + Σ_j rate_state_grad_r[j] · S[j, p]
ẋ          += stoich_r · rate_r
Ṡ[:, p]    += stoich_r · total_dr_dθ[p]           # state Jacobian THROUGH stoich
if r feeds incidence slot k:
    acċ[k]         += rate_r
    acc_senṡ[k, p] += total_dr_dθ[p]              # raw flow derivative — NO stoich
```

This touches only the compartments each transition moves (fast), and it makes
the "state Jacobian through stoichiometry" (`Ṡ`) structurally distinct from the
"raw incidence-flow derivative" (`acc_sens`) — the two `J` uses the chain-rule
section flags as a silent-wrong-gradient trap. The per-transition `DerivEntry`
values are evaluated by the shared `eval_emitted_grad` seam.

**C-seed — the initial condition of `S` is not zero.** The ODE initial state is
a function of θ (`model.initial_state(params)` over
`InitialConditions::Parameterized`), so
`S(t_start) = ∂(initial_state)/∂θ =
ic_grad` (§1b), nonzero for any
seeding/`ivp` parameter (`i0`/`e0`/`s0`). Seeding `S(t_start) = 0` — the natural
default — makes `∂loglik/∂i0 ≡ 0`: NUTS never moves the epidemic's size/timing
parameters and their marginals collapse to the prior, a silent-wrong posterior
for exactly what incidence data informs. No refusal path catches it (it is the
IC surface `coeff_guard` defers here), so the seed is a Phase-1 deliverable and
the §1f oracle **must** estimate an IC parameter.

**Decision: a single augmented `(x, acc, S, acc_sens)` system integrated by one
fixed-step RK4 whose four stages share intermediate states**, stacked into one
contiguous block. Required, not a preference: NUTS simulates a frictionless
Hamiltonian and relies on energy conservation; if any sensitivity block is
advanced by a decoupled endpoint step while its value counterpart is `O(dt⁴)`,
the gradient is `O(dt)`-inconsistent with the value and NUTS produces spurious
divergences. **Both** the incidence accumulator `acc` and its sensitivity
`acc_sens` must advance at the same stage accuracy as `x`/`S` — `acc` is already
the `OdeState.flow` field, so `acc_sens` (`∂flow/∂θ`) joins the sensitivity
block; Euler-updating `acc_sens` while `acc` is RK4-accurate recreates the very
mismatch this decision exists to prevent. The ODE state is a concrete
`OdeState { int, real, flow }` (`ode.rs`), **not** a bare `Vec<f64>`, and
derivatives are assembled field-by-field in `ode_derivs` — so the augmentation
is **new derivative-assembly code**, not merely a larger allocation. Forward
mode is `O(d)`; adjoint (`O(1)` in `d`) is out of scope — see below.

**Decision (rk45/Dopri5 under `nuts`): refuse it — `nuts` on `ode` requires the
fixed-step RK4.** An adaptive integrator is already selectable
(`integrator = "rk45"`, `Dopri5`, honored on the ODE fit path — pinned by
`mh_ode_recovery.rs`), and it is wrong for a gradient sampler two ways: (i) the
PI controller sizes steps from the state-error norm only, so the augmented `S`
is absent from error control and under-resolved relative to `x`; (ii) the step
sequence `h(θ)` is discontinuous in θ, so the adaptively-solved loglik is only
piecewise-smooth — the FD-vs-symbolic oracle (§1f) can disagree even when both
arms are "correct," and NUTS energy conservation degrades. A silent fallback to
RK4 under `nuts` would be worse (value on rk45, gradient on RK4 — a
value/gradient integrator mismatch), so this is a hard, reasoned refusal at
dispatch, not a fallback. Adaptive-integrator sensitivities (`S` in the
controller, or a fixed step grid) are a named follow-up.

**Clamp caveat — refuse an active clamp under `nuts` (v1).** RK4 clamps each
compartment `int_vals[i].max(0.0)` per accepted step (and snapshots round;
`ode.rs`), a non-smooth operation the augmented `S` does not model: when a
compartment is pinned at 0 the true `∂x/∂θ` in the clamped direction is 0 but
`S` keeps propagating a nonzero value. "Zeroing the corresponding `S` rows at
the kink" is technically the derivative of the clamped map away from the kink,
but it yields a posterior over a **nonphysical, nonsmooth solver artifact** —
not what we want, and firing exactly in the low-count prevalence regime §1d
markets. So for v1 an **active clamp under a gradient method is a hard
dispatch-time refusal** ("reduce `dt`, change the model, or use gradient-free
`mh` on `ode`"); an explicit opt-in `differentiate_through_clamp` experimental
mode is a later follow-up, never the default. The §1f oracle includes a clamping
trajectory as an **expected-refusal** test.

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
(e.g. `n = S+I+R`) cannot yield a smooth gradient. **No existing path refuses
this** — `n` is `#[differentiate(skip)]`, so it carries no `DerivEntry` for the
preflight to scan; the preflight's `ParametricN` case is about _θ_-dependence
(not state-dependence); and post-P4b `coeff_guard` scans only forcing/table
coefficients. So a state-dependent-`n` refusal under gradient methods is a
**new** dispatch-time check this phase adds (§1h): scan the `n` expression for a
`Pop`/`PopSum` **or `Projected`** reference — `n = projected` (e.g.
`binomial(n = projected, p = rho)`) is state/flow-dependent through the
projection even with no bare `Pop`, so `Projected` in `n` is gradient-dependent
too. Not an inherited check — else the gradient silently treats `n` as constant.

### 1e. Crate relocation

`compute_ode_loglik` lives in `cli` (`rust/crates/cli/src/fit/runner.rs`) but
references only `sim` types and the `sim` scoring seam (`CompiledModel`, `ode`,
`MultiStreamObsModel`, `Simulate`), so the move down into `sim` is type-clean.
The gradient assembler (`det_grad`) must live in `sim`, so Phase 1 begins by
moving it and repointing **all** its callers: `pmmh.rs` (×3 — the ODE-`mh`
production path), `dt_check.rs` (the dt-convergence re-eval gate), `survey.rs`,
and `nlopt_stage.rs`. Enumerate by grep, not from the function's own docstring —
that docstring is itself stale (it lists only `nlopt_stage`/`survey`, missing
`pmmh` and `dt_check`, which is exactly how the caller list drifted);
`profile.rs` only mentions it in comments and is **not** a caller. Fix the
docstring as part of the move. (Aside: CLAUDE.md's "cli → io → observe → sim →
ir" is stale — there is no `observe` crate; the real layering is
`cli → io → sim → ir`.)

### 1f. Gradient-check oracle (the only finite difference in the system)

Extend `gradient_check.rs`: assert `‖∇_symbolic − ∇_FD‖_∞ < 10⁻⁴` across all
estimated params, on a model that exercises every new path at once. Each bullet
below is a path some silent-wrong finding rides; omitting it lets the
FD-vs-symbolic check pass green with that arm bugged —

- an **incidence stream over ≥2 reset intervals** (the per-`Interval`
  sensitivity and its reset),
- a **hoisted binding used in a rate** (the `BindingRef` state-diff),
- a **fixed-time, θ-dependent-magnitude event jump** (the supportable event
  case, §1g — `S⁺ = ∂Φ/∂x·S⁻ + ∂Φ/∂θ`),
- an **estimated initial-condition parameter** (`i0`) — exercises the `ic_grad`
  seed `S(t_start)`; without it the oracle passes with the seed bugged (§1c
  C-seed),
- a **`projected`-dependent-variance likelihood** (the He et al.
  discretized-Normal) — exercises the obs-score sum over non-mean arguments; a
  Poisson-only oracle passes with the variance-through-`projected` term dropped
  (see "The chain rule"),
- a **`DerivedExpr` prevalence stream** (a nonlinear projection, e.g.
  `I/(S+I+R)`) — exercises `∂DerivedExpr/∂x`, not just the linear-selection arm.

**But not only the combined model.** A single large model can hide a bug through
cancellation, so pair it with **isolated** per-path checks, each small enough to
localise a failure: a closed-form state sensitivity (exponential decay /
SIR-lite), the incidence-integral-with-two-resets alone, a `DerivedExpr`
prevalence alone, a hoisted `let N = S+I+R` inside an FOI alone, and a
fixed-time θ-magnitude jump alone. Use a **mixed tolerance**, not a bare
`‖·‖_∞ < 10⁻⁴`: `abs_err < atol + rtol · max(1, |fd|, |symbolic|)`. And check
the gradient in **both** spaces — the θ-space data gradient _and_ the final
z-space posterior gradient — because a wrong transform/Jacobian (gh#374) cancels
in θ-space and only shows up in z.

**Expected-refusal tests** are part of the oracle, not an afterthought — one per
§1h bullet, and crucially a test that the two **lifted** refusals (a rate/obs
`DerivEntry::Unsupported`, and an IC-only `coeff_guard` parameter) actually fire
on the **ODE-NUTS dispatch path** — the whole point of moving the gate out of
`run_pgas`, so a test that only exercised the old PGAS path would prove nothing
here. Plus: an adaptive integrator, an active clamp, a nonsmooth function of
state (each of a state-indexed table, `Mod`/`Floor`/`Ceil` of a `Pop`, a `Cond`
predicate on a `Pop`), a state-dependent / `Projected` binomial `n`, a
θ-dependent (or reactive) event time, and a `DerivedExpr` prevalence with no
`∂DerivedExpr/∂x` — each asserts the right refusal fires with the right message.

Red-then-green per CLAUDE.md TDD: write each check against a deliberately wrong
`J_x` (or a missing refusal) first, confirm it fails, then land the correct
emission and confirm it passes. Finite differences live here and nowhere else.
The same FD machinery is what gh#78's user-facing `--check-grads` would expose
at the model's use site; this oracle is its dev-test ancestor.

### 1g. Event / intervention sensitivity

Interventions and events modify state at a time `t_e` via a jump map
`x⁺ = Φ(x⁻, θ)` (`intervention.rs`). "Parameter-dependent event" is two
different cases, and they are gated differently:

- **Fixed time `t_e`, θ-dependent _magnitude_ — supportable.** Propagate the
  sensitivity through the jump: `S⁺ = (∂Φ/∂x)·S⁻ + ∂Φ/∂θ` (and `acc_sens` is
  unchanged unless Φ moves a flow). This is the event case the §1f oracle
  exercises.
- **θ-dependent event _time_ `t_e(θ)` — refuse (v1).** The sensitivity picks up
  a boundary term `−(ẋ⁺ − ẋ⁻)·(∂t_e/∂θ)` the fixed-grid RK4 does not carry;
  emitting the jump term alone is a silent-wrong gradient.
- **State/reactive event time — refuse (v1).** As above, plus an implicit
  `t_e(x)` dependence (gh#204 reactive interventions).

And the integrator must **step exactly to** every event, observation, and reset
time: an event or reset that lands _inside_ an RK4 step splits the value and
sensitivity paths unless both use the identical substep boundary — the
discrete-time analogue of §1c's shared-stage requirement.

### 1h. The gradient-method capability gate

The refusals scattered through §1a–§1g are **one gate**, not a per-site
discipline. Introduce a single `preflight_gradient_method(model, estimated)` (a
`GradientCapability` check) that ODE-NUTS **and** any future gradient-MLE
(`nl-lbfgs`, out of scope) both route through, so the gradient column of the
backend×method matrix is defined **once** and cannot drift between methods — the
CLAUDE.md dense-matrix rule applied to the gradient axis. It refuses, carrying
the compiler's own reason string where one exists:

- a rate/obs/σ² coefficient the compiler could not differentiate — the existing
  `DerivEntry::Unsupported` scan, **lifted out of `run_pgas`** to here so every
  gradient method sees it (§1b, §2);
- a parameter reaching a coefficient only through an initial condition — the
  existing IC `coeff_guard`, called from here;
- a **state-dependent** binomial/beta-binomial `n` (`Pop`/`PopSum`/`Projected`
  in `n`; §1d);
- a **nonsmooth function of state** in a rate or projection:
  `Mod`/`Floor`/`Ceil` of `Pop`, a `Cond` predicate on `Pop`, a state-indexed
  `TableLookup(…, Pop…)`, and `Min`/`Max`/`Abs` of state unless explicitly
  accepted as piecewise-smooth (§1a);
- a **`DerivedExpr` prevalence** stream with no emitted `∂DerivedExpr/∂x` (see
  "The chain rule");
- an **active nonnegativity clamp** on the trajectory (§1c);
- an **adaptive integrator** (`rk45`/`Dopri5`) under `nuts` (§1c);
- a **θ-dependent or reactive event time** (§1g).

Every bullet is an expected-refusal test in §1f. This is the "capability gap
expressed in code" of principle 2, made concrete: one function, one list, one
place the answer to "can this model be fit by a gradient method?" lives.

### Phase 1 gates

`det_grad` agrees with the FD oracle to `10⁻⁴` on the combined model; the PGAS
gradient is **byte-identical** before and after (the state-derivative arm must
not perturb the PGAS path, which relies on `∂rate/∂x` being zeroed — distinct
entry points). **This PGAS gradient-byte-identity gate does not exist yet** and
is itself a Phase-1 deliverable, built before the state-derivative arm lands:
`gate_pgas_thread_invariance.rs` exists precisely because that invariant is
fragile, and `gate_pgas_density_baseline.rs` pins the transition-_density_
scalar, not the gradient. `make test` green with the golden diff from the schema
bump reviewed.

## Phase 2 — `nuts` on `ode`

With `det_grad` in hand, the NUTS _sampler core_ is unchanged and nearly free —
but "thin wiring" understates the production surface: non-finite /
invalid-gradient handling, chain storage, warmup and mass-matrix configuration
and persistence, divergence diagnostics, and failure messages all still need
building. The two pieces PGAS currently inlines are lifted out **here**
(ODE-NUTS is their only second consumer), each gated on PGAS byte-identity (the
gate this work builds — Phase-1 gates):

- **`nuts_warmup.rs`** — the dual-averaging step-size + Welford mass-matrix
  adaptation. In PGAS this state lives on the sweep struct (`pgas.rs:~1948`), is
  initialized per-rung (`DualAveraging::new` + Welford fields, `~2288–2311`),
  and updated in the warmup loop (`~2560–2800`), entangled with parallel
  tempering (`rungs[rung].*`, the `mass_adapt_end = 0.7·adapt_end` schedule at
  `~2615`, cold-rung prints). (The `~2453–2514` region is CSMC trajectory-warmup
  sweeps + rate-grad pre-resolution, not the adaptation — do not extract from
  there.) The extracted `NutsWarmup` is parameterized on cold-rung identity;
  PGAS instantiates one per rung, ODE-NUTS exactly one.
- **`posterior_target.rs`** — the prior + transform + Jacobian wrapper
  (`prior_log_density_and_grad_z`, `pgas.rs:1859`) plus the θ→params gradient
  closure (`~2524`; note `~2362–2400` is the `DerivEntry::Unsupported` preflight
  closure, a different block). The β-tempering and the data-term are lifted to a
  pluggable θ-space data-term: PGAS plugs in `complete_data_loglik_grad`,
  ODE-NUTS plugs in `det_grad`.

Then `nuts_stage.rs` is a thin driver:
`build_nuts_target(det_grad, priors,
transforms)` → a single `NutsWarmup` over a
contiguous warmup window → sample with `nuts_step`.

**Decision (`nuts` as a method): a deterministic-likelihood method, not a
backend-agnostic one.** NUTS-direct needs `∇_θ log p(y | θ)` in closed form,
which only a **directly-differentiable (deterministic) likelihood** provides —
`ode` today. On a stochastic backend the marginal likelihood is an intractable
integral over latent trajectories and cannot be differentiated directly; that is
exactly why gradient-NUTS there lives _inside_ PGAS (the `use_nuts` θ|X step,
gradient via `complete_data_loglik_grad`). So `nuts × chain_binomial` is **not**
an "unsupported cell" to hard-error — gradient sampling _is_ available there, as
PGAS — and the honest routing is: `nuts` parses to a `FitAlgorithm` whose
capability requirement is a directly-differentiable likelihood (satisfied by
`ode`, and any future deterministic backend); on a stochastic backend it keeps
today's **steer-to-`pgas`** hint ("use `pgas`, which runs NUTS on the
conditioned trajectory"), not a bare capability error. Concretely: (i) `nuts`
parses to a `FitAlgorithm`; (ii) a `Stage::Nuts` + dispatch arm; (iii) a
`("nuts","ode")` supported entry in `METHODS`; (iv) stochastic cells route to
the pgas hint. The token parses uniformly, but `nuts` is scoped to what it
actually is — the sampler for a likelihood you can differentiate — rather than a
method that "works everywhere except where we reject it."

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
  `10⁻⁶/day` reactivation timescale); explicit RK4 is appropriate, and the
  deterministic `dt`-convergence check guards against silent under-resolution.
  (The adaptive Dopri5 integrator stays available for the gradient-free ODE
  fits, but is **refused under `nuts`** — §1c; a fixed step grid is what keeps
  the Hamiltonian and the FD oracle well-defined.)

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
  methods.** Refused at dispatch by the new gradient-refusal check (§1b/§2) —
  not `coeff_guard`, which is now IC-only and does not see event times.
- **Adaptive-integrator (rk45/Dopri5) sensitivities under `nuts`.** Requires `S`
  in the step-error controller (or a fixed step grid the adaptive path honors);
  §1c refuses rk45 under `nuts` for now. → gh issue.
- **`--method auto`.** Choosing between the deterministic and stochastic
  likelihoods is too high-stakes for silent selection.

## Risks

Ordered by how silently a mistake bites. The first three are
silent-wrong-gradient holes whose only guard is the §1f oracle covering the
corresponding path — so the oracle model, not the code, is the real gate.

- **Missing IC-sensitivity seed (§1c C-seed).** The highest risk: without
  `S(t_start) = ic_grad`, every `init`/`ivp` parameter gets an identically-zero
  gradient, no refusal path catches it, and the epidemic's size/timing marginals
  silently collapse to the prior. Gate: an estimated IC parameter in the oracle.
- **Dropped non-mean obs-score terms ("The chain rule").** Reducing the obs
  score to `∂mean/∂projected` drops the variance-through-`projected` term of the
  default He et al. model — a wrong gradient on the flagship obs model. Gate: a
  `projected`-dependent-variance likelihood in the oracle.
- **Unrefused state-dependent `n` / `DerivedExpr` prevalence.** Neither is
  caught by an existing path; each must be built (a refusal for state-dependent
  `n`, and either `∂DerivedExpr/∂x` or a refusal for nonlinear prevalence).
  Gate: both in the oracle.
- **`BindingRef` state-differentiation** is the subtlest part of the autodiff
  change (the invariant _inverts_ for `WrtPop` — bindings become state-bearing —
  and the binding table must be threaded through `differentiate`; no cycle guard
  is needed, bindings are acyclic by construction). Gate: a binding used in a
  rate.
- **Lineage consolidation is a trap, not a cleanup** (§1a): `deriv_num_wrt_pop`
  freezes the normalizer where the general `differentiate` does not — do not
  merge them.
- **Incidence-sensitivity reset** is the subtlest part of the gradient assembly;
  gate: an incidence stream with ≥2 reset intervals.
- **Continuous-eval / clamp regression into PGAS.** The state-derivative arm
  must not perturb the existing PGAS gradient (which relies on `∂rate/∂x` being
  zeroed); gated by the PGAS byte-identity test — which itself does not exist
  yet and is a Phase-1 deliverable.
- **Order-mismatched sensitivities.** A decoupled `S` step (or an
  adaptive-`h(θ)` integrator, or the `.max(0.0)` clamp) silently degrades NUTS
  Hamiltonian conservation; the shared-stage fixed-step RK4 + rk45 refusal +
  clamp handling (§1c) are the mitigations, and "divergences not clustered at
  boundaries" is the Phase-2 canary.
