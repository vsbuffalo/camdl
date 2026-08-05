---
paths:
  - "rust/crates/sim/src/inference/**"
description: Inference stack — required reading, the dense backend × method matrix, capability dispatch, RNG coupling
---

# Inference

## Required reading

The proposal that introduced the feature (under `docs/dev/proposals/`), the
relevant module in `rust/crates/sim/src/inference/`, and any related incident
reports in `docs/dev/incidents/`.

## The inference stack

- `if2.rs` — Iterated filtering for maximum likelihood estimation
- `pgas.rs` — Particle Gibbs with Ancestor Sampling (default Bayesian method)
- `pgas_grad.rs` — Gradient evaluation for PGAS (uses compiler-emitted
  `rate_grad`)
- `nuts.rs` — No-U-Turn Sampler for gradient-based parameter proposals within
  PGAS
- `pmmh.rs` — Particle Marginal Metropolis-Hastings (production; prefer PGAS for
  long observation series)
- `particle_filter.rs` — Bootstrap particle filter
- `dmeasure.rs` — Observation likelihood compilation
- `obs_loglik.rs` — Distribution log-PMFs + analytical gradients (incl. digamma)

The OCaml compiler (`ocaml/lib/ir/autodiff.ml`) performs source-to-source
symbolic differentiation of rate expressions, emitting `rate_grad` fields in the
IR. The Rust backend evaluates these derivative expressions via `eval_expr` — no
runtime autodiff, no finite differences.

## Every backend × inference method is a supported cell — no silent gaps

The product of forward backends (chain_binomial, gillespie, ode) and inference
methods (particle filter, IF2, PGAS, PMMH) is a dense matrix. Every cell must
either work and be tested, or fail loudly through the capability system — there
is no third option. A combination that is silently untested, silently skipped,
or excluded from a cross-cutting test behind a "covered elsewhere" hand-wave is
a latent silent-wrong-answer bug.

This is how gh#187 hid: the PGAS path silently dropped scheduled interventions,
behind a cross-backend lifecycle test that excluded PGAS and a comment claiming
another test covered it — it did not.

- **Consolidate to the shared substrate before the matrix can drift.** Push the
  bug-prone, genuinely-shared mechanism into one path every cell routes through
  (e.g. every backend and the PGAS producer step with
  `chain_binomial::step_one`, which owns intervention/event/balance application
  via the `effects` seam) so a feature cannot be live in one cell and silently
  absent in another. Unify the shared substrate, keep the distinct algorithms
  distinct. Reimplementing shared behaviour per-cell is how cells diverge.
- **A genuine capability gap is expressed in code, not omitted.** If a
  combination truly cannot be supported, route it through the `Capabilities`
  bitflags (`rust/crates/sim/src/lib.rs`:
  `CompiledModel::required_capabilities()` vs each backend's
  `Simulate::capabilities()`), which hard-errors at dispatch with a message
  naming the limitation — and the error tells the user. Never drop the
  combination from a test or skip it silently.
- **Tests follow the matrix.** A property that must hold across cells is tested
  for each cell it applies to. A "covered by test X" claim must name X, and X
  must actually exercise that property for that cell — verify it, don't assert
  it.

## Backend capabilities

The `Capabilities` bitflags (`rust/crates/sim/src/lib.rs`) are **one of three**
compatibility axes — model-feature × backend.
`CompiledModel::required_capabilities()` derives a model's needs from the IR (a
DSL primitive: `overdispersed(...)`, `balance {}`, a real compartment, `dt` in a
rate); each backend declares what it provides; mismatch → hard error at dispatch.

- `OVERDISPERSION`: `overdispersed(rate, σ²)` transitions require chain-binomial
  (NegBinomial draws). Gillespie and ODE reject these models with a hard error.
- `REAL_COMPARTMENTS`: real-valued compartments with ODE equations.

Subtlety: the "what a backend provides" side **forks by execution mode** —
`Simulate::capabilities()` (simulate path) vs a separate hardcoded table in
`fit/methods.rs::check_model_capabilities` (inference path), which deliberately
withholds `REAL_COMPARTMENTS` from chain-binomial inference (gh#191). The other
two axes — algorithm × backend (the `METHODS` registry) and model-feature ×
algorithm (scattered ad-hoc checks) — plus the known gaps are mapped in
[`docs/dev/capabilities-system.md`](../../docs/dev/capabilities-system.md); read
it before touching any backend/algorithm/capability check.

## RNG and paired-seed coupling

The runtime uses a plain ChaCha8 `StatefulRng`. Paired scenarios with the same
seed produce identical trajectories only while the RNG is consumed in the same
order on both sides: pre-intervention trajectories are byte-identical for
`enable`/`disable` scenarios, and correlated-but-not-identical for `set`/`scale`
scenarios that modify propensities from t=0. Any structural change that reorders
draws also breaks the coupling — this is paired-seed CRN, NOT event-keyed RNG.

## Scheduled interventions and simulation backends

Interventions are deterministic state modifications (not stochastic events).
Each backend handles them differently and the interaction is non-trivial — see
§2.3.1 of [`docs/compartmental-ir-spec.md`](../../docs/compartmental-ir-spec.md)
for the Gillespie/ODE/discrete-time specifics. The key constraint: after a
Gillespie intervention, propensities must be fully recomputed from the modified
state; do not resume with remaining exponential time.

`events {}` is the sister construct to `interventions {}` but fires every
substep (`add()`, `transfer()`, `set()` actions); `balance {}` is the
population-conservation constraint, applied last in each substep after
transitions and events.
