//! Gradient evaluation for the PGAS complete-data log-likelihood.
//!
//! Uses compiler-emitted derivative expressions (`rate_grad` on each transition)
//! to compute ∂log p(y,X|θ)/∂θ analytically. No runtime autodiff or finite
//! differences — just evaluating pre-differentiated expression trees.
//!
//! The chain rule through p_total and binom_logpmf is hardcoded here:
//!   ∂/∂θ log Binom(k; n, p(θ)) = [k/p - (n-k)/(1-p)] × dp/dθ
//!   dp/dθ = dt × exp(-total_rate × dt) × d(total_rate)/dθ
//!
//! The d(rate)/dθ terms come from the OCaml compiler's symbolic differentiation.

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::propensity::{eval_propensities, EvalCtx};
use crate::resolved_expr::{eval_resolved, eval_emitted_grad, eval_deriv_entry, ResolvedGradMap};
use crate::state::{IntState, RealState};
use crate::inference::obs_loglik::{binom_logpmf, digamma, gamma_multiplier_log_density};
use crate::inference::numerics::BINOM_PROB_EPS;
use crate::inference::pgas::{PGASTrajectory, OVERDISP_SIGMA_SQ_FLOOR};
use crate::inference::particle_filter::Observation;

/// Build a run-specific rate-gradient table re-keyed to estimated-param indices.
///
/// `rate_grads_indexed[tr_idx]` is a [`ResolvedGradMap`] of `(model_param_idx,
/// entry)`, each entry a real `Grad` or a carried `Unsupported`.
/// `model_to_estimated[model_idx]` maps a model param index to its position in
/// the estimated-param vector, or `None` if that param is not estimated this run.
///
/// The returned table is re-keyed to estimated-param indices: `(est_idx, entry)`.
/// Only entries whose param is estimated are kept — fixed parameters are dropped.
pub fn resolve_rate_grad_for_run(
    rate_grads_indexed: &[ResolvedGradMap],
    model_to_estimated: &[Option<usize>],
) -> Vec<ResolvedGradMap> {
    rate_grads_indexed.iter()
        .map(|tr_grads| {
            tr_grads.iter()
                .filter_map(|(model_idx, entry)| {
                    model_to_estimated.get(*model_idx)
                        .and_then(|opt| *opt)
                        .map(|est_idx| (est_idx, entry.clone()))
                })
                .collect()
        })
        .collect()
}

/// Evaluate log transition density AND its gradient w.r.t. estimated parameters
/// for a single substep.
///
/// Returns (log_p, grad) where grad[i] = ∂log_p/∂θ_i for i in 0..d.
///
/// `rate_grads_for_run` is pre-resolved via `resolve_rate_grad_for_run`: each
/// entry is `(estimated_param_idx, ResolvedDerivEntry)`. No string lookup happens
/// here — the hot path is index-only.
pub fn log_transition_density_grad(
    model: &CompiledModel,
    counts_before: &[i64],
    flows: &[u64],
    gammas: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    // gh#272 LICM: per-eval prologue staged at the PGAS-grad θ-stable boundary
    // (`complete_data_loglik_grad`) and threaded in. `None` ⇒ on-demand.
    per_eval: Option<&[f64]>,
    d: usize,
    rate_grads_for_run: &[ResolvedGradMap],
) -> Result<(f64, Vec<f64>), SimError> {
    let n_int = model.int_local_to_global.len();
    let n_tr = model.model.transitions.len();

    let mut int_s = IntState::new(n_int);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = RealState::new(model.real_local_to_global.len());

    let mut propensities = vec![0.0; n_tr];
    eval_propensities(model, &int_s, &real_s, params, t, dt, per_eval, &mut propensities)?;

    // Evaluates the resolved `rate_grad` expressions below (which carry
    // `PerEvalRef` after LICM), so it threads the staged scratch — the
    // gradient-path half of the gh#272 hoist.
    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt, projected: None, aux: None, int_float_override: None, per_eval,
    };

    let mut log_p = 0.0;
    let mut grad = vec![0.0; d];
    let mut handled = vec![false; n_tr];
    let mut gamma_idx = 0;

    // Source-grouped transitions (mirrors step_one + log_transition_density_substep)
    for &(src_local, ref group) in &model.source_groups {
        let n_src = counts_before[src_local].max(0);
        if n_src == 0 {
            for &tr_idx in group {
                if flows[tr_idx] > 0 { return Ok((f64::NEG_INFINITY, vec![0.0; d])); }
                handled[tr_idx] = true;
            }
            continue;
        }

        // Compute effective per-capita rates AND their gradients.
        //
        // IM7+IM9 (2026-04-19 inference review): previously this loop
        // (a) skipped transitions with `rate <= 0.0` rather than
        // `rate <= RATE_EPSILON` — diverging from chain_binomial's
        // step_one and pgas.rs's density, and (b) advanced
        // `gamma_idx` once per source group rather than once per
        // overdispersed transition with rate above the threshold.
        // Either drift made the gradient disagree with the density
        // for any model with multiple overdispersed transitions in
        // one source group (spatial polio, multi-strain, competing
        // overdispersed reporting streams). Now mirrors pgas.rs
        // exactly.
        let mut probs: Vec<(usize, f64, Vec<f64>)> = Vec::new(); // (tr_idx, eff_rate, d_eff_rate/dθ)
        let mut total_rate = 0.0_f64;
        let mut total_rate_grad = vec![0.0; d];

        for &tr_idx in group {
            let rate = propensities[tr_idx];
            // gh#122: a sole-exit deterministic source member is a POINT MASS.
            // `step_one` records `count = clamp(round(rate*dt), 0, n_src)`; the
            // density is a point mass (no term) and its gradient is 0 (`round`
            // is piecewise-constant → d/dθ = 0 a.e.). A flow disagreeing with the
            // recorded count is an impossible trajectory (density 0 → -inf), and
            // that guard MUST match the value fn (`log_transition_density_substep`)
            // so the NUTS (energy, gradient) pair stays consistent. (Mixed sources
            // are rejected upstream, so a deterministic member is the sole exit.)
            if matches!(model.model.transitions[tr_idx].draw_method,
                        ir::transition::DrawMethod::Deterministic)
            {
                let expected = ((rate * dt).round() as i64).clamp(0, n_src) as u64;
                if flows[tr_idx] != expected {
                    return Ok((f64::NEG_INFINITY, vec![0.0; d]));
                }
                handled[tr_idx] = true;
                continue;
            }
            if rate <= crate::chain_binomial::RATE_EPSILON {
                handled[tr_idx] = true;
                continue;
            }
            let per_capita = rate / n_src as f64;

            // Compute d(rate)/dθ for each estimated parameter (index-keyed, no string lookup)
            let mut d_rate = vec![0.0; d];
            for (est_idx, entry) in &rate_grads_for_run[tr_idx] {
                d_rate[*est_idx] = eval_deriv_entry(entry, &ctx) / n_src as f64;
            }

            let (effective, d_effective) = if let ir::transition::DrawMethod::Overdispersed { .. } =
                &model.model.transitions[tr_idx].draw_method
            {
                // Consume one gamma per overdispersed transition —
                // matches step_one's gamma_used.push(...) and
                // pgas.rs's gamma_idx += 1 inside the per-transition
                // Overdispersed arm.
                let g = if gamma_idx < gammas.len() { gammas[gamma_idx] } else { 1.0 };
                gamma_idx += 1;
                let eff = per_capita * g;
                let d_eff: Vec<f64> = d_rate.iter().map(|&dr| dr * g).collect();
                (eff, d_eff)
            } else {
                (per_capita, d_rate)
            };

            total_rate += effective;
            for i in 0..d { total_rate_grad[i] += d_effective[i]; }
            probs.push((tr_idx, effective, d_effective));
        }

        if total_rate <= crate::chain_binomial::RATE_EPSILON || probs.is_empty() { continue; }

        // Total exits: Binom(n_exit; n_src, p_total).
        //
        // Im17 in 2026-04-19 inference review: the clamp to
        // [BINOM_PROB_EPS, 1-BINOM_PROB_EPS] keeps `dbinom_dp` finite (otherwise
        // 1/0 → NaN) at the cost of accuracy right at the
        // boundary — the gradient becomes ~±1e15 and NUTS
        // divergences are expected there. Without the clamp the
        // sampler would hit NaN and ICE; with it the sampler sees
        // a huge-but-finite number, rejects the leapfrog step,
        // and adapts step size down. Tolerated behavior.
        // gh#audit-H3: stable (p, q) primitive (clamped for the
        // gradient form; q would be the right partner for the future
        // (n-k)/q gradient term but the current code computes
        // (n-k)/(1-p), which we keep — the clamp at least avoids the
        // worst cancellation).
        let (p_total, _q) = super::numerics::prob_q_from_rate_dt_clamped(total_rate, dt, BINOM_PROB_EPS);
        // gh#811: the value and the gradient must agree about the rejected
        // region. `binom_logpmf` scores a non-finite `p` as -inf (gh#810); the
        // hand-rolled `dbinom_dp` below would evaluate `k/NaN - (n-k)/(1-NaN)`
        // to NaN. That pair is the gh#197/gh#200 divergence class: the -inf
        // kills one particle, the NaN poisons the NUTS momentum update.
        //
        // Reachable, though not by the route first suspected. Parameters are
        // guarded at `propensity.rs:532` and computed propensities at `:651`,
        // so a NaN cannot arrive through the rate expression. It arrives
        // through the GAMMA multipliers, which `:160` above multiplies into the
        // rate with no finiteness check and which are REPLAYED from a stored
        // trajectory (`--resume`, `--init-state`) rather than freshly drawn.
        // Only `overdispersed()` transitions carry them.
        //
        // Same floor the branch below already returns for an impossible flow.
        if !p_total.is_finite() {
            return Ok((f64::NEG_INFINITY, vec![0.0; d]));
        }
        let n_exit: u64 = probs.iter().map(|&(tr_idx, _, _)| flows[tr_idx]).sum();
        log_p += binom_logpmf(n_exit, n_src as u64, p_total);

        // Gradient of binom_logpmf w.r.t. p_total:
        //   d/dp [k*ln(p) + (n-k)*ln(1-p)] = k/p - (n-k)/(1-p)
        let dbinom_dp = n_exit as f64 / p_total - (n_src as u64 - n_exit) as f64 / (1.0 - p_total);

        // dp_total/d(total_rate) = dt * exp(-total_rate * dt)
        let dp_dtotalrate = dt * (-total_rate * dt).exp();

        // Chain rule: d(binom)/dθ = dbinom_dp * dp_dtotalrate * d(total_rate)/dθ
        for i in 0..d {
            grad[i] += dbinom_dp * dp_dtotalrate * total_rate_grad[i];
        }

        // Split density: Binom(flow_k; remaining, p_split)
        let n_competing = probs.len();
        let mut remaining = n_exit;
        let mut rate_remaining = total_rate;
        let mut rate_remaining_grad = total_rate_grad.clone();

        for (k, &(tr_idx, eff_rate, ref d_eff_rate)) in probs.iter().enumerate() {
            handled[tr_idx] = true;
            if k == n_competing - 1 {
                if flows[tr_idx] != remaining {
                    return Ok((f64::NEG_INFINITY, vec![0.0; d]));
                }
                // Last category: no density contribution (remainder)
            } else if remaining > 0 && rate_remaining > 0.0 {
                let p_split = (eff_rate / rate_remaining).clamp(BINOM_PROB_EPS, 1.0 - BINOM_PROB_EPS);
                // gh#811, the split-draw sibling: same divergence, same floor.
                if !p_split.is_finite() {
                    return Ok((f64::NEG_INFINITY, vec![0.0; d]));
                }
                let flow_k = flows[tr_idx];
                log_p += binom_logpmf(flow_k, remaining, p_split);

                // Gradient of p_split = eff_rate / rate_remaining
                // d(p_split)/dθ = (d_eff * rate_rem - eff * d_rate_rem) / rate_rem²
                let dbinom_dp_split = flow_k as f64 / p_split
                    - (remaining - flow_k) as f64 / (1.0 - p_split);
                for i in 0..d {
                    let dp_split = (d_eff_rate[i] * rate_remaining
                        - eff_rate * rate_remaining_grad[i])
                        / (rate_remaining * rate_remaining);
                    grad[i] += dbinom_dp_split * dp_split;
                }

                remaining -= flow_k;
                rate_remaining -= eff_rate;
                for i in 0..d { rate_remaining_grad[i] -= d_eff_rate[i]; }
            } else if flows[tr_idx] > 0 {
                return Ok((f64::NEG_INFINITY, vec![0.0; d]));
            }
        }
    }

    // Ungrouped / inflow transitions: Poisson density (or deterministic
    // exact-count check). Mirrors the value fn's ungrouped loop in
    // `pgas::log_transition_density_substep` so the grad-path energy matches
    // `complete_data_loglik` (gh#200: a deterministic source-less inflow must
    // NOT be Poisson-scored; gh#3-ungrouped: skip on RATE_EPSILON, not 0.0).
    for (tr_idx, &rate) in propensities.iter().enumerate() {
        if handled[tr_idx] || rate <= crate::chain_binomial::RATE_EPSILON { continue; }
        let mean = rate * dt;
        let flow = flows[tr_idx] as f64;

        if matches!(model.model.transitions[tr_idx].draw_method,
                    ir::transition::DrawMethod::Deterministic) {
            // Deterministic: the flow is a fixed function of the rate, not a
            // Poisson draw. Exact-count guard, NO density term and NO gradient
            // (same as the value fn). A mismatch is an impossible trajectory.
            if flows[tr_idx] != mean.round() as u64 {
                return Ok((f64::NEG_INFINITY, vec![0.0; d]));
            }
            continue;
        }

        // log Poisson(k; λ) = k*ln(λ) - λ - lgamma(k+1)
        // d/dλ = k/λ - 1
        // dλ/dθ = d(rate)/dθ * dt
        log_p += crate::inference::obs_loglik::poisson_logpmf(flow, mean);

        for (est_idx, entry) in &rate_grads_for_run[tr_idx] {
            let d_rate = eval_deriv_entry(entry, &ctx);
            let d_mean = d_rate * dt;
            if mean > 0.0 {
                grad[*est_idx] += (flow / mean - 1.0) * d_mean;
            }
        }
    }

    Ok((log_p, grad))
}

/// Adds the gamma-multiplier density VALUE to `log_p` and returns its GRADIENT.
///
/// For each of the substep's overdispersed transitions it adds
/// `log Γ(g; dt/σ², σ²/dt)` directly into `*log_p` (gh#197 — the term was
/// previously absent from the grad-path energy) and accumulates its derivative
/// into the returned `grad`. The value goes through the shared
/// [`gamma_multiplier_log_density`] helper the value fn also uses, AND is added
/// in the same left-fold order (directly, not pre-summed), so the grad-path
/// energy matches `complete_data_loglik` BIT-EXACTLY for any number of gammas
/// per substep (a pre-summed `(g1+g2)` would differ by a ULP — f64 add is
/// non-associative).
///
/// For each overdispersed transition with rate > RATE_EPSILON, the recorded
/// `gammas[gamma_idx]` is the realised draw from Γ(g; dt/σ², σ²/dt). The
/// log-density is
///
///   log p(g; dt/σ², σ²/dt) = (shape-1)·ln(g) - g/scale - shape·ln(scale) - lgamma(shape)
///
/// where shape = dt/σ², scale = σ²/dt. Differentiating w.r.t. σ² and
/// chain-ruling through any estimated parameter `θ_k`:
///
///   d(shape)/d(σ²) = -dt/σ⁴            d(scale)/d(σ²) = 1/dt
///   d(log Γ)/d(shape) = ln(g) - ln(scale) - ψ(shape)        (ψ = digamma)
///   d(log Γ)/d(scale) = g/scale² - shape/scale
///   d(log Γ)/d(σ²)    = d(log Γ)/d(shape)·d(shape)/d(σ²)
///                     + d(log Γ)/d(scale)·d(scale)/d(σ²)
///   d(log Γ)/d(θ_k)   = d(log Γ)/d(σ²) · d(σ²)/d(θ_k)
///
/// Mirrors the gamma-density loop in `pgas::complete_data_loglik` exactly
/// (`pgas.rs:565-617`): same source_group iteration order, same gamma_idx
/// accounting (advance on rate > RATE_EPSILON ∧ not Deterministic ∧
/// Some(overdispersion)).
///
/// `estimated_to_model[i]` is the model-param index of the i-th estimated
/// parameter — used to look up `∂σ²/∂θ` in the compiler-emitted σ² gradient map
/// via the shared `eval_emitted_grad` seam (gh#180).
fn gamma_density_value_and_grad_substep(
    model: &CompiledModel,
    counts_before: &[i64],
    gammas: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    // gh#272 LICM: per-eval prologue staged at `complete_data_loglik_grad`.
    per_eval: Option<&[f64]>,
    estimated_to_model: &[usize],
    log_p: &mut f64,
) -> Result<Vec<f64>, SimError> {
    use crate::chain_binomial::RATE_EPSILON;

    let d = estimated_to_model.len();
    let mut grad = vec![0.0; d];
    if gammas.is_empty() {
        return Ok(grad);
    }

    let n_int = model.int_local_to_global.len();
    let mut int_s = IntState::new(n_int);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = RealState::new(model.real_local_to_global.len());

    let n_tr = model.model.transitions.len();
    let mut propensities = vec![0.0; n_tr];
    eval_propensities(model, &int_s, &real_s, params, t, dt, per_eval, &mut propensities)?;

    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, dt,
        projected: None, aux: None, int_float_override: None, per_eval,
    };

    let mut gamma_idx_local: usize = 0;
    for &(src_local, ref group) in &model.source_groups {
        let n_src = counts_before[src_local].max(0);
        if n_src == 0 { continue; }
        for &tr_idx in group {
            let rate = propensities[tr_idx];
            if rate <= RATE_EPSILON { continue; }
            if let ir::transition::DrawMethod::Deterministic = model.model.transitions[tr_idx].draw_method {
                continue;
            }
            if let Some(ref resolved_od) = model.resolved.overdispersion[tr_idx] {
                let sigma_sq = eval_resolved(resolved_od, &ctx);
                // σ²'s emitted `∂σ²/∂θ` map (in lockstep with `overdispersion`, so
                // `Some` here). Empty ⇒ every σ² derivative is a genuine zero.
                let od_grad = model.resolved.overdispersion_grad[tr_idx]
                    .as_deref()
                    .unwrap_or(&[]);
                if gamma_idx_local < gammas.len() && sigma_sq > OVERDISP_SIGMA_SQ_FLOOR {
                    let g = gammas[gamma_idx_local];
                    let shape = dt / sigma_sq;
                    let scale = sigma_sq / dt;

                    // VALUE (gh#197): the gamma-multiplier log-density — the term
                    // whose gradient is added below but which was previously
                    // absent from the grad-path energy. Added DIRECTLY to `log_p`
                    // (not pre-summed) so the fold order matches
                    // complete_data_loglik's left-fold `((td)+g1)+g2` exactly —
                    // f64 addition is non-associative, so a pre-summed
                    // `(g1+g2)` would differ by a ULP for multi-gamma substeps.
                    // Same helper + same condition as the value fn (not gated on
                    // g > 0.0; the helper floors ln(g)) → the spine oracle holds
                    // bit-exact for any number of gammas per substep.
                    *log_p += gamma_multiplier_log_density(shape, scale, g);

                    // GRADIENT: d/dθ log Γ(g; shape, scale).
                    if g > 0.0 {
                        let dlg_dshape = g.ln() - scale.ln() - digamma(shape);
                        let dlg_dscale = g / (scale * scale) - shape / scale;
                        let dshape_dsq = -dt / (sigma_sq * sigma_sq);
                        let dscale_dsq = 1.0 / dt;
                        let dlg_dsq = dlg_dshape * dshape_dsq + dlg_dscale * dscale_dsq;

                        for (est_idx, &model_idx) in estimated_to_model.iter().enumerate() {
                            let d_sigma_sq = eval_emitted_grad(od_grad, model_idx, &ctx);
                            grad[est_idx] += dlg_dsq * d_sigma_sq;
                        }
                    }
                }
                gamma_idx_local += 1;
            }
        }
    }
    Ok(grad)
}

/// Gradient of the complete-data log-likelihood over all substeps.
///
/// Returns (log_p, grad) summed over four terms (all wired as of gh#76):
/// 1. Initial-state density and its gradient, from the laws the model
///    DECLARES (`init { I ~ poisson(rate = I0) }`), through the shared seam.
/// 2. Transition rate-density gradient (via compiler-emitted `rate_grad` and
///    the binomial-chain-rule machinery in `log_transition_density_grad`).
/// 3. Gamma-multiplier-density gradient w.r.t. σ² (gh#20) — chain rule through
///    the compiler-emitted `∂σ²/∂θ` map via the shared `eval_emitted_grad` seam.
/// 4. Observation-density gradient w.r.t. obs-model params (gh#76, gh#180) —
///    per-distribution `∂logpmf/∂arg` helpers in `obs_loglik.rs` times the
///    compiler-emitted `∂arg/∂θ` (with a `DerivedExpr` projection inlined, so a
///    param reaching the observation through the projection is captured).
///
/// `estimated_to_model[i]` is the model-param index of the i-th estimated
/// parameter (the inverse of `model_to_estimated` used to build
/// `rate_grads_for_run`). Required by terms 3 and 4.
pub fn complete_data_loglik_grad(
    model: &CompiledModel,
    trajectory: &PGASTrajectory,
    params: &[f64],
    _observations: &[Observation],
    dt: f64,
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    d: usize,
    rate_grads_for_run: &[ResolvedGradMap],
    obs_at_substep: &super::pgas::ObsAtSubstep,
    estimated_to_model: &[usize],
) -> Result<(f64, Vec<f64>), SimError> {
    debug_assert_eq!(estimated_to_model.len(), d,
        "estimated_to_model length {} must match d={}", estimated_to_model.len(), d);

    let t_start = model.model.simulation.t_start;
    let n_substeps = trajectory.substeps.len();
    let n_tr = model.model.transitions.len();
    // gh#272 LICM: stage the per-eval prologue ONCE for this θ (`params` fixed for
    // the whole gradient evaluation) and thread it into every per-substep grad
    // eval. `None` ⇒ on-demand.
    let per_eval_scratch = crate::resolved_expr::stage_per_eval(model, params, t_start, dt);
    let per_eval = per_eval_scratch.as_deref();
    let mut log_p = 0.0;
    let mut grad = vec![0.0; d];

    // Initial-state density AND its gradient, from the SAME seam the value path
    // (`complete_data_loglik`) and the sampler (`csmc_as`) use. A law is a
    // sampler and a density and a gradient; taking two of the three from one
    // place and the third from another is how NUTS ends up with a gradient
    // identically zero on a coordinate the energy does depend on.
    //
    // `initial_state_logpdf_grad` is indexed by MODEL parameter; the NUTS
    // basis is the estimated set, so map it through `estimated_to_model` here
    // (the same projection the observation and σ² terms use below).
    {
        log_p += model.initial_state_logpdf(&trajectory.initial_counts, &[], params)?;
        let init_grad =
            model.initial_state_logpdf_grad(&trajectory.initial_counts, &[], params)?;
        for (i, &model_idx) in estimated_to_model.iter().enumerate() {
            grad[i] += init_grad[model_idx];
        }
    }

    let mut cum_flows = vec![0u64; n_tr];
    // Phase 2a: the gradient path mirrors the value path's per-stream `acc` bin
    // EXACTLY. If value bins per-stream while grad stays blanket, NUTS would
    // differentiate a different binning than the value objective accepts — a
    // silent bias. They MUST match (fold + reset_due in lockstep).
    let mut acc = vec![0u64; obs_model.n_interval_streams()];
    // Exact-tiling invariant (debug): records partition the run contiguously,
    // each duration in (0, dt]. Replaces the 2b snap invariant (rec.t0 ==
    // t_start+s·dt) a shortened exact substep violates. Value and gradient must
    // reconstruct the SAME (t0, dt_substep) — both now read rec, never s·dt.
    let mut prev_end = t_start;

    for s in 0..n_substeps {
        let rec = &trajectory.substeps[s];
        if cfg!(debug_assertions) {
            debug_assert!(rec.dt_substep > 0.0 && rec.dt_substep <= dt + 1e-9,
                "substep {s}: dt_substep {} not in (0, dt={dt}]", rec.dt_substep);
            debug_assert!((rec.t0 - prev_end).abs() < 1e-9,
                "substep {s}: t0 {} not contiguous with previous end {prev_end}", rec.t0);
            prev_end = rec.t0 + rec.dt_substep;
        }
        let t = rec.t0;
        let dt_s = rec.dt_substep;
        let counts_before = &rec.counts_before;

        let (td, td_grad) = log_transition_density_grad(
            model, counts_before, &rec.flows, &rec.gammas,
            params, t, dt_s, per_eval, d, rate_grads_for_run,
        )?;

        if !td.is_finite() {
            return Ok((f64::NEG_INFINITY, vec![0.0; d]));
        }
        log_p += td;
        for i in 0..d { grad[i] += td_grad[i]; }

        // gh#20: Gamma-multiplier density gradient.
        //
        // Adds d/dθ_k log Gamma(g; dt/σ², σ²/dt) for each gamma multiplier
        // recorded at this substep. Non-zero whenever σ² is — or depends on —
        // an estimated parameter (typical case: a parameter like `sigma_se`
        // appears directly as the σ² of an overdispersed transition).
        // gh#197: the gamma-multiplier density contributes to BOTH the energy
        // (`log_p`) and the gradient. Previously only the gradient was added, so
        // the NUTS energy was low by Σ log Γ — biasing the σ² posterior and
        // diverging from the value fn / MH / swap. The helper adds the value
        // straight into `log_p` (in the value fn's fold order) and returns grad.
        let gamma_grad = gamma_density_value_and_grad_substep(
            model, counts_before, &rec.gammas, params, t, dt_s, per_eval, estimated_to_model,
            &mut log_p,
        )?;
        for i in 0..d { grad[i] += gamma_grad[i]; }

        // Accumulate flows
        for (i, &f) in rec.flows.iter().enumerate() {
            cum_flows[i] += f;
        }

        // gh#76: observation density + its gradient.
        // Snapshot projections read post-step state from the trajectory record.
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            // FOLD (Phase 2a): close this interval's per-transition `cum_flows`
            // into the per-stream `acc` BEFORE scoring — EXACTLY mirroring the
            // value path. Both the loglik and its gradient read `acc`.
            obs_model.fold_into_acc(&cum_flows, &mut acc);
            log_p += obs_model.log_likelihood_from_flows_and_counts(
                &acc, &rec.counts_after, obs_idx, params);

            // Per-distribution gradient helpers in `obs_loglik.rs` give
            // d(log L)/d(mean), d(log L)/d(k), etc.; the per-stream method
            // chain-rules through the compiler-emitted `∂arg/∂θ` map (via the
            // shared `eval_emitted_grad` seam) to reach the estimated parameters.
            let obs_grad = obs_model.log_likelihood_grad_from_flows_and_counts(
                &acc, &rec.counts_after, obs_idx, params, estimated_to_model,
            );
            for i in 0..d { grad[i] += obs_grad[i]; }

            // `cum_flows` blanket-zeroed (unchanged); the per-stream `acc` bins
            // per-stream (mirrors value path's reset_due_acc).
            cum_flows.fill(0);
            obs_model.reset_due_acc(obs_idx, &mut acc);
        }
    }

    Ok((log_p, grad))
}
