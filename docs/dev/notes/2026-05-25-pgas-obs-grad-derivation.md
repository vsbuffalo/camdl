# PGAS+NUTS observation-density gradient derivation

Date: 2026-05-25 (updated 2026-05-25, post-cleanup)
Scope: gh#20, gh#76 — fix the two silent-zero gradient terms in
`complete_data_loglik_grad`, then the cleanup pass that addressed
five concerns from upstream code review (the gate regression, the
value/grad asymmetry, multi-overdispersed lockstep, Binomial obs
coverage, and tail FD points).

## Post-cleanup status (top-of-page)

After commits `8fe3543` → `d464322` on
`worktree-agent-aa7f73880ff417dea`:

  * **C1 gate is back**, narrowed. NUTS still refuses estimated
    parameters routed through (a) a `BetaBinomial` obs likelihood or
    (b) a parametric `DerivedExpr` projection. Both arms remain
    documented no-ops in the gradient; the gate keeps NUTS from
    silently producing biased posteriors on them.
  * **Value and gradient are now numerically symmetric on
    `discretized_normal`.** Both use the same erfc-stable `prob`
    expression. The prior asymmetry (value erfc-difference, gradient
    Φ-difference) was a classical recipe for energy non-conservation
    and spurious divergences. Side finding: the *audit-H2* branches
    in the value were swapped (both branches used erfc of negative
    args → cancellation against 2.0); both are corrected.
  * **Multi-overdispersed-transition lockstep is now tested.**
    `gradient_check_overdisp::gh76_cleanup_two_overdisp_grad_matches_fd_*`
    runs a fixture with two overdispersed transitions out of the same
    source compartment (`sir_two_overdispersed.{camdl,ir.json}`) and
    pins gamma-density gradients at multiple σ values, including an
    asymmetric case where σ_inf ≠ σ_loss. Iterator alignment
    verified at rel_err ≤ 4e-9.
  * **Structural fix for the lockstep was deferred** as gh#79. Two
    independent `gamma_idx` counters still walk `model.source_groups`
    in parallel; the test above pins the alignment but doesn't make
    drift structurally impossible. gh#79 also notes a latent
    asymmetry in the σ² evaluation `EvalCtx` (value uses zero-filled
    IntState, gradient uses `counts_before` — silent because σ² is a
    parameter constant in practice).
  * **Binomial obs FD test landed** —
    `gradient_check_obs::gh76_binomial_obs_grad_matches_fd`. The
    Binomial dispatch arm at `obs_model.rs:204-218` is now end-to-end
    verified at 1e-4. Rel_err ≤ 4e-9.
  * **Tail FD points landed** — both unit-level (in
    `obs_loglik::tests`) at 3σ/5σ/8σ above μ and integration-level
    (in `gradient_check_obs.rs`) using `project_trajectory_to_obs_shifted`.
    All pass at 1e-4 after the erfc port.

The remainder of this note preserves the original derivation (terms
3 and 4) and is left intact below as the reference for those terms.

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

### BetaBinomial — gated, not silently zero

`obs_loglik.rs` has no BetaBinomial gradient helper. The
`eval_likelihood_resolved_grad` arm for `BetaBinomial` is a no-op
(grad unchanged). Estimating a BetaBinomial-bound parameter with
PGAS+NUTS would land in the same silent-zero-gradient regime gh#76
was filed against.

Post-cleanup (commit `8fe3543`), this is now caught by the narrowed
C1 preflight gate: a fit configured to estimate a parameter that
appears in a BetaBinomial likelihood argument returns

  PGAS+NUTS gradient does not cover BetaBinomial obs likelihoods
  or parametric DerivedExpr projections (gh#76 follow-up). …
  Blocked parameters: '<name>' (in a BetaBinomial likelihood arg).
  Either fix these parameters … switch to a non-gradient method
  (IF2, PMMH), or wait for the missing gradient arm to land.

Adding the BetaBinomial helper itself is still mechanical
(digamma-based, mirrors NegBin's dispersion grad) but out of scope
for the cleanup. Tracked as a follow-up; the gate is the safety net
until then.

### `DerivedExpr` projections that depend on parameters — gated

A stream's projection can be a `DerivedExpr` (e.g.
`I / PopSum([S, I, R])`). If that expression depends on parameters
(e.g. `I / N₀`), then `d(projected)/dθ_k ≠ 0`, but the current
`eval_likelihood_resolved_grad` differentiates the likelihood-argument
expressions with `projected` treated as a constant. The result is
missing one chain-rule term: `∂L/∂(projected) · ∂(projected)/∂(θ_k)`.

`FlowSum` and `IntCompSum` projections — the common case — do not
depend on parameters, so this caveat is silent for them. Issue gh#76's
reproducer (rho, k_obs, σ_obs) uses `FlowSum` only.

Post-cleanup, the narrowed C1 gate also covers this case: any
parameter referenced inside a `DerivedExpr` obs projection is
refused. Adding the chain-rule term is straightforward but requires
routing `d(projected)/dθ` from the projection layer; tracked as a
follow-up.

### Discretized-Normal tail precision — fixed in cleanup

**Updated 2026-05-25 cleanup.** The gradient now uses the same
erfc-stable `prob` expression as the value (commit `1e41cbe`). FD-vs-
analytic agreement at 3σ/5σ/8σ tail points is now rel_err 7e-12 to
9e-11, matching the near-mode regime.

While porting the value-side form into the gradient, an unrelated
bug was found in the *audit-H2* branches of
`discretized_normal_logpmf_tol`: both "upper" and "lower" branches
called erfc with **negative** arguments (`erfc(-z_hi/√2)` etc), which
produces values near 2.0 — and subtracting two near-2 values is
exactly the cancellation audit-H2 was meant to fix. The corrected
forms use erfc with positive arguments (small erfc values, no
cancellation against 2.0):

  * Upper tail: `0.5·(erfc(z_lo/√2) − erfc(z_hi/√2))`
  * Lower tail: `0.5·(erfc(−z_hi/√2) − erfc(−z_lo/√2))`

Verified against an external mpmath-precision reference:

  z_lo=9.5, z_hi=10:   pre = 0.0 (cancellation collapsed)
                       post = 1.04e-21 (correct)
  z_lo=11.5, z_hi=12:  pre = 0.0
                       post = 6.58e-31

The downstream effect (silent until this cleanup): deep-tail
observations in polio AFP-surveillance and similar rare-event
inference were scored against a `prob` floored to 0 (then to the
tol-floor → constant `-log(tol)`) rather than the model's actual
predicted probability. Restoring this turns those observations from
constant-floor (no information content) into informative — both for
the LL (value side) and for NUTS proposals (gradient side).

The gradient also now respects the tol-floor symmetrically: when
`prob_raw <= tol`, the gradient returns `(0, 0)` rather than dividing
the (un-floored) `dp_dmu` by the floor. This matches what a clean FD
against the value computes when both functions agree the value is at
the floor.

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

### Removed-then-restored surface (cleanup, commit `8fe3543`)

- `pgas::for_each_likelihood_expr` — restored from b981d60^. Walks
  every `Expr` inside an obs `Likelihood` and applies a callback;
  used by the narrowed C1 gate to detect parameters routed through
  uncovered arms.
- `pgas::collect_param_refs` — restored, same source. Helper that
  walks an `Expr` tree and accumulates `Param` names.
- audit-C1 preflight gate in `run_pgas` — restored, narrowed. The
  original removal in `b981d60` claimed "the gradient covers what
  the gate was protecting against," but the gradient still has two
  documented no-op arms (BetaBinomial, parametric `DerivedExpr`
  projections). Without the gate those arms silently re-open the
  gh#76 vulnerability. Narrowed predicate: refuse only parameters
  whose reachability traverses one of these two arms — leaves
  NegBin/Poisson/Normal/Binomial/Bernoulli + σ² estimation
  unaffected.

### Why approach (i) over (ii) on the gate

Per the cleanup brief, two acceptable shapes:
(i) narrowed reachability gate at preflight; (ii) hard-error the
gradient arms at evaluation time. Approach (ii) would require
propagating a `Result<()>` through three layers
(`eval_likelihood_resolved_grad` → `log_likelihood_grad_from_flows_and_counts`
→ `complete_data_loglik_grad`) plus the NUTS hot path, where every
leapfrog evaluation now becomes fallible. The preflight gate
provides the same protection at a single once-per-fit cost and the
recoverable history gave us the walker as a starting point. (i) was
chosen.

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

Original gh#20 + gh#76 commits:

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

Cleanup follow-up commits (5):

- `rust/crates/sim/src/inference/pgas.rs` — restored
  `for_each_likelihood_expr` + `collect_param_refs`; narrowed C1
  preflight gate refusing BetaBinomial-routed and parametric-
  DerivedExpr-projection-routed estimated params.
- `rust/crates/sim/src/inference/obs_loglik.rs` —
  `discretized_normal_logpmf_grad` ported to the erfc-stable `prob`
  form; corrected the (swapped) audit-H2 branches in
  `discretized_normal_logpmf_tol`; tol-floor symmetry added on the
  gradient.
- `rust/crates/sim/tests/pgas_gate_betabinomial.rs` — new regression
  test for the gate (`run_pgas` with a BetaBinomial-routed estimated
  parameter must return `Err(SimError::Validation(...))` naming the
  blocked parameter and the uncovered arm).
- `rust/crates/sim/tests/gradient_check_obs.rs` — tail FD points
  (3σ, 5σ, 8σ) via `project_trajectory_to_obs_shifted`; new Binomial
  obs FD test.
- `rust/crates/sim/tests/gradient_check_overdisp.rs` —
  multi-overdispersed-transition lockstep FD tests at four σ regimes.
- `ocaml/golden/sir_two_overdispersed.{camdl,ir.json}` — new fixture
  with two overdispersed transitions sharing source compartment S.

Follow-ups filed:
- gh#79: restructure shared gamma-density iterator + fix latent
  IntState asymmetry in σ² evaluation context.
- (existing) BetaBinomial gradient helper and parametric
  DerivedExpr-projection chain-rule term — the two arms the new
  C1 gate refuses.
