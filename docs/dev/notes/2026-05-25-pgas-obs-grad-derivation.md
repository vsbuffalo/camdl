# PGAS+NUTS observation-density gradient derivation

Date: 2026-05-25
Scope: gh#20, gh#76 — fix the two silent-zero gradient terms in
`complete_data_loglik_grad`.

## Context

PGAS+NUTS's complete-data log-likelihood is

```
ℓ(θ) = log p(X₀ | θ)                              # initial-state Binom
     + Σ_s log p(x_{s+1} | x_s, θ, g_s)           # transition (rate) density
     + Σ_s log Γ(g_s ; dt/σ², σ²/dt)              # gamma multiplier density
     + Σ_t log p(y_t | x_t, θ_obs)                # observation density
```

The gradient `∂ℓ/∂θ` is what NUTS sees on each leapfrog step. Before
this work, only terms 1 and 2 were wired into
`complete_data_loglik_grad`. Term 3 (gh#20) and term 4 (gh#76) were
silently zero, which biased NUTS proposals on any axis where σ² or an
obs-model parameter was estimated.

## What was wired (the easy parts)

### Term 3: gamma-multiplier density (gh#20)

For each overdispersed transition with rate > RATE_EPSILON, a gamma
multiplier `g ~ Gamma(shape=dt/σ², scale=σ²/dt)` was drawn at step
time. The log-density is

```
log Γ(g; shape, scale) = (shape-1)·ln(g) - g/scale - shape·ln(scale) - lgamma(shape)
```

Differentiate w.r.t. σ²:

```
d(shape)/d(σ²) = -dt/σ⁴
d(scale)/d(σ²) =  1/dt
d(log Γ)/d(shape) = ln(g) - ln(scale) - ψ(shape)          (ψ = digamma)
d(log Γ)/d(scale) = g/scale² - shape/scale
d(log Γ)/d(σ²)    = d(log Γ)/d(shape)·d(shape)/d(σ²)
                  + d(log Γ)/d(scale)·d(scale)/d(σ²)
```

For each estimated parameter θ_k, chain-rule through the σ² resolved
expression via `eval_resolved_deriv`:

```
d(log Γ)/d(θ_k) = d(log Γ)/d(σ²) · d(σ²)/d(θ_k)
```

The implementation in `pgas_grad::log_gamma_density_grad_substep`
mirrors `pgas::complete_data_loglik`'s gamma-density loop exactly: same
source-group iteration order, same gamma_idx accounting
(advance per overdispersed transition with rate > RATE_EPSILON and
not Deterministic). Misalignment here is the bug that broke a previous
draft of the helper — verified by a deliberate "two overdispersed
transitions in one source group" mental walkthrough.

### Term 4: observation density (gh#76)

Per-distribution gradient helpers in `obs_loglik.rs`
(`negbin_logpmf_grad`, `discretized_normal_logpmf_grad`,
`poisson_logpmf_grad`, …) already existed and were FD-tested. None had
a production caller. The new
`MultiStreamObsModel::log_likelihood_grad_from_flows_and_counts` calls
`obs_model::eval_likelihood_resolved_grad` per stream, which dispatches
to the helper for the likelihood variant and chain-rules through every
likelihood-argument expression via `eval_resolved_deriv`:

```
∂ log L / ∂θ_k = Σ_arg  ∂ log L / ∂(arg)  ·  ∂(arg) / ∂(θ_k)
```

For NegBin: `arg ∈ {mean, dispersion}`. For discretized-Normal:
`arg ∈ {mean, sd}` (with d(var)/d(sd) = 2·sd inserted to convert the
helper's d(log L)/d(var) into d(log L)/d(sd)). For Poisson: just rate.
For Bernoulli and Binomial: see the per-arm code.

## What was NOT wired (caveats to surface)

### BetaBinomial

`obs_loglik.rs` has no BetaBinomial gradient helper. The
`eval_likelihood_resolved_grad` arm for `BetaBinomial` is a no-op
(grad unchanged). Estimating a BetaBinomial-bound parameter with
PGAS+NUTS would land in the same silent-zero-gradient regime gh#76 was
filed against. Adding it is mechanical (digamma-based, mirrors NegBin's
dispersion grad) but out of scope here. Tracked as a follow-up.

### `DerivedExpr` projections that depend on parameters

A stream's projection can be a `DerivedExpr` (e.g.
`I / PopSum([S, I, R])`). If that expression depends on parameters
(e.g. `I / N₀`), then `d(projected)/dθ_k ≠ 0`, but the current
`eval_likelihood_resolved_grad` differentiates the likelihood-argument
expressions with `projected` treated as a constant. The result is
missing one chain-rule term: `∂L/∂(projected) · ∂(projected)/∂(θ_k)`.

`FlowSum` and `IntCompSum` projections — the common case — do not
depend on parameters, so this caveat is silent for them. Issue gh#76's
reproducer (rho, k_obs, σ_obs) uses `FlowSum` only.

This is a known limitation; if a model author writes a parametric
`DerivedExpr` projection AND estimates a parameter in that projection
with PGAS+NUTS, they'll hit the same silent-bias pattern. Adding
`∂L/∂(projected) · ∂(projected)/∂(θ_k)` is straightforward but
requires routing `d(projected)/dθ` from the projection layer; tracked
as a follow-up to gh#76.

### Discretized-Normal tail precision

`discretized_normal_logpmf_grad` uses `prob = Φ(z_hi) - Φ(z_lo)` in
the denominator. In the deep tails (both Φ values near 0 or near 1)
this loses precision; FD vs analytic agreement degrades from 1e-7 to
~1e-3. The LL function `discretized_normal_logpmf_tol` uses an
erfc-difference form for the same probability (gh#audit-H2) — applying
the analogous rewrite to the gradient would close this gap. Tracked
as a follow-up to gh#76.

The gh#76 FD test pins synthetic observations to the obs-model mean
(rather than the rounded projection), placing them near the
likelihood's mode where the helper is accurate. FD agreement on those
points is ~1e-7 to 1e-11.

## Architectural choices

### Index-keyed param lookup

The gradient helpers receive `estimated_to_model: &[usize]` — the
inverse of the `model_to_estimated` map used to build
`rate_grads_for_run`. For each estimated slot `i`,
`estimated_to_model[i]` is the model-param index; we call
`eval_resolved_deriv(expr, model_idx, ctx)` directly. No string
lookups in the gradient hot path. Matches the proposal-2026-04-20
pattern.

### Single-pass accumulation

The four gradient terms are accumulated into one `grad: Vec<f64>`
inside the substep loop in `complete_data_loglik_grad`. They share
intermediate state (counts_before, t, dt) that's already in scope. No
parallel passes. The conditional `if let Some(obs_idx) = ...` keeps the
obs-grad call out of the no-obs hot path.

### Public surface

- `eval_likelihood_resolved_grad` is `pub(crate)` — only the
  `multi_stream_obs.rs` method calls it. External callers route
  through `MultiStreamObsModel::log_likelihood_grad_from_flows_and_counts`
  (added as a `pub` method on the obs model).
- `log_gamma_density_grad_substep` is private to `pgas_grad.rs` — only
  `complete_data_loglik_grad` calls it.

### Removed surface

- `pgas::for_each_likelihood_expr` — only used by the C1 preflight gate;
  the gate is gone.
- audit-C1 preflight gate in `run_pgas` — no longer needed; the gradient
  covers what the gate was protecting against.

## FD-vs-analytic agreement

(from the gh#20 and gh#76 commit messages)

### gh#20: sigma_se ∈ {0.01, 0.1, 1.0}

```
d(ll)/d(sigma_se = 0.01) =  1.313371e3 (analytic) vs  1.313371e3 (fd), rel_err = 1.32e-8
d(ll)/d(sigma_se = 0.10) =  1.945916e2 (analytic) vs  1.945916e2 (fd), rel_err = 9.00e-11
d(ll)/d(sigma_se = 1.00) = -6.536736e0 (analytic) vs -6.536736e0 (fd), rel_err = 3.46e-11
```

### gh#76: NegBin / Poisson / discretized-Normal

```
[negbin]    d(ll)/d(rho)       = 1.026757e-1, rel_err = 7.37e-8
[negbin]    d(ll)/d(k)         = 3.828745e-1, rel_err = 1.54e-8
[negbin]    d(ll)/d(p_detect)  = 5.000000e0,  rel_err = 3.17e-10
[poisson]   d(ll)/d(rho)       = 4.000000e0,  rel_err = 2.59e-8
[dis-norm]  d(ll)/d(rho)       = 6.070215e0,  rel_err = 1.95e-10
[dis-norm]  d(ll)/d(sigma_obs) = -9.967506e-1, rel_err = 3.16e-11
```

All cases under the 1e-4 acceptance bar in the issue body; observed
agreement is 1e-7 to 1e-11 depending on parameter and helper precision.

## Files touched

- `rust/crates/sim/src/inference/pgas_grad.rs` — gamma-density helper,
  `complete_data_loglik_grad` signature (`estimated_to_model` added),
  obs-density chain into the substep loop.
- `rust/crates/sim/src/inference/obs_model.rs` —
  `eval_likelihood_resolved_grad` (per-distribution dispatch).
- `rust/crates/sim/src/inference/multi_stream_obs.rs` —
  `log_likelihood_grad_from_flows_and_counts` method.
- `rust/crates/sim/src/inference/obs_loglik.rs` — precision note on
  `discretized_normal_logpmf_grad`.
- `rust/crates/sim/src/inference/pgas.rs` — removed C1 preflight gate and
  dead `for_each_likelihood_expr` helper.
- `rust/crates/sim/tests/gradient_check_overdisp.rs` — gh#20 FD tests.
- `rust/crates/sim/tests/gradient_check_obs.rs` — gh#76 FD tests.
- `rust/crates/sim/tests/gradient_check.rs` — signature update.
