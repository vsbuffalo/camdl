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

/// Probability-domain clamp to keep values strictly in (0,1) for
/// numerical stability in log(p) / log(1-p) computations.
/// Distinct from LOG_PROB_FLOOR which is a log-domain floor.
const PROB_CLAMP_EPS: f64 = 1e-15;
use crate::propensity::{eval_propensities, EvalCtx};
use crate::resolved_expr::{eval_resolved, ResolvedExpr};
use crate::state::{IntState, RealState};
use crate::inference::obs_loglik::binom_logpmf;
use crate::inference::pgas::{PGASTrajectory, IVPMapping};
use crate::inference::particle_filter::Observation;

/// Build a run-specific rate_grads table with estimated-param indices.
///
/// `rate_grads_indexed[tr_idx]` contains `(model_param_idx, expr)` pairs.
/// `model_to_estimated[model_idx]` maps a model param index to its position
/// in the estimated-param vector, or `None` if that param is not being
/// estimated in this run.
///
/// The returned table has estimated-param indices: `(est_idx, expr)`. Only
/// entries where `model_to_estimated[model_idx]` is `Some` are kept — fixed
/// parameters (not in the estimated set) are intentionally dropped.
pub fn resolve_rate_grad_for_run(
    rate_grads_indexed: &[Vec<(usize, ResolvedExpr)>],
    model_to_estimated: &[Option<usize>],
) -> Vec<Vec<(usize, ResolvedExpr)>> {
    rate_grads_indexed.iter()
        .map(|tr_grads| {
            tr_grads.iter()
                .filter_map(|&(model_idx, ref expr)| {
                    model_to_estimated.get(model_idx)
                        .and_then(|opt| *opt)
                        .map(|est_idx| (est_idx, expr.clone()))
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
/// `rate_grads_for_run` is pre-resolved via `resolve_rate_grad_for_run`:
/// each entry is `(estimated_param_idx, gradient_expr)`. No string lookup
/// happens in this function — the hot path is index-only.
pub fn log_transition_density_grad(
    model: &CompiledModel,
    counts_before: &[i64],
    flows: &[u64],
    gammas: &[f64],
    params: &[f64],
    t: f64,
    dt: f64,
    d: usize,
    rate_grads_for_run: &[Vec<(usize, ResolvedExpr)>],
) -> Result<(f64, Vec<f64>), SimError> {
    let n_int = model.int_local_to_global.len();
    let n_tr = model.model.transitions.len();

    let mut int_s = IntState::new(n_int);
    int_s.counts.copy_from_slice(counts_before);
    let real_s = RealState::new(model.real_local_to_global.len());

    let mut propensities = vec![0.0; n_tr];
    eval_propensities(model, &int_s, &real_s, params, t, &mut propensities)?;

    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params, t, projected: None, int_float_override: None,
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
            if rate <= crate::chain_binomial::RATE_EPSILON
                || matches!(model.model.transitions[tr_idx].draw_method,
                    ir::transition::DrawMethod::Deterministic)
            {
                handled[tr_idx] = true;
                continue;
            }
            let per_capita = rate / n_src as f64;

            // Compute d(rate)/dθ for each estimated parameter (index-keyed, no string lookup)
            let mut d_rate = vec![0.0; d];
            for &(est_idx, ref resolved_grad) in &rate_grads_for_run[tr_idx] {
                d_rate[est_idx] = eval_resolved(resolved_grad, &ctx) / n_src as f64;
            }

            let (effective, d_effective) = if let ir::transition::DrawMethod::Overdispersed(_) =
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
        // [1e-15, 1-1e-15] keeps `dbinom_dp` finite (otherwise
        // 1/0 → NaN) at the cost of accuracy right at the
        // boundary — the gradient becomes ~±1e15 and NUTS
        // divergences are expected there. Without the clamp the
        // sampler would hit NaN and ICE; with it the sampler sees
        // a huge-but-finite number, rejects the leapfrog step,
        // and adapts step size down. Tolerated behavior.
        let p_total = (1.0 - (-total_rate * dt).exp()).clamp(PROB_CLAMP_EPS, 1.0 - PROB_CLAMP_EPS);
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
                let p_split = (eff_rate / rate_remaining).clamp(PROB_CLAMP_EPS, 1.0 - PROB_CLAMP_EPS);
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

    // Ungrouped / inflow transitions (Poisson)
    for (tr_idx, &rate) in propensities.iter().enumerate() {
        if handled[tr_idx]
            || rate <= crate::chain_binomial::RATE_EPSILON
            || matches!(
                model.model.transitions[tr_idx].draw_method,
                ir::transition::DrawMethod::Deterministic
            )
        { continue; }
        let mean = rate * dt;
        let flow = flows[tr_idx] as f64;

        // log Poisson(k; λ) = k*ln(λ) - λ - lgamma(k+1)
        // d/dλ = k/λ - 1
        // dλ/dθ = d(rate)/dθ * dt
        log_p += crate::inference::obs_loglik::poisson_logpmf(flow, mean);

        for &(est_idx, ref resolved_grad) in &rate_grads_for_run[tr_idx] {
            let d_rate = eval_resolved(resolved_grad, &ctx);
            let d_mean = d_rate * dt;
            if mean > 0.0 {
                grad[est_idx] += (flow / mean - 1.0) * d_mean;
            }
        }
    }

    Ok((log_p, grad))
}

/// Gradient of the complete-data log-likelihood over all substeps.
///
/// Returns (log_p, grad) summed over transition densities + observation densities.
/// Observation model gradient is zero when obs model params (rho, psi) are fixed.
pub fn complete_data_loglik_grad(
    model: &CompiledModel,
    trajectory: &PGASTrajectory,
    params: &[f64],
    _observations: &[Observation],
    dt: f64,
    obs_model: &super::multi_stream_obs::MultiStreamObsModel,
    ivp_mappings: &[IVPMapping],
    d: usize,
    rate_grads_for_run: &[Vec<(usize, ResolvedExpr)>],
    obs_at_substep: &super::pgas::ObsAtSubstep,
) -> Result<(f64, Vec<f64>), SimError> {
    let t_start = model.model.simulation.t_start;
    let n_substeps = trajectory.substeps.len();
    let n_tr = model.model.transitions.len();
    let mut log_p = 0.0;
    let mut grad = vec![0.0; d];

    // Initial state density gradient: d/dθ log Binom(S₀; N₀, s0)
    // N₀ is the per-patch population (not global) for stratified models.
    if !ivp_mappings.is_empty() {
        for ivp in ivp_mappings {
            let count = trajectory.initial_counts[ivp.compartment_idx] as u64;
            let frac = params[ivp.model_param_idx].clamp(1e-10, 1.0 - 1e-10);
            let patch_pop = super::pgas::patch_population(model, &trajectory.initial_counts, ivp.compartment_idx);
            log_p += binom_logpmf(count, patch_pop as u64, frac);

            // d/d(frac) log Binom(count; N, frac) = count/frac - (N-count)/(1-frac)
            let dbinom_dfrac = count as f64 / frac
                - (patch_pop as u64 - count) as f64 / (1.0 - frac);
            grad[ivp.param_idx] += dbinom_dfrac;
        }
    }

    let mut cum_flows = vec![0u64; n_tr];

    for s in 0..n_substeps {
        let t = t_start + s as f64 * dt;
        let counts_before = &trajectory.substeps[s].counts_before;
        let rec = &trajectory.substeps[s];

        let (td, td_grad) = log_transition_density_grad(
            model, counts_before, &rec.flows, &rec.gammas,
            params, t, dt, d, rate_grads_for_run,
        )?;

        if !td.is_finite() {
            return Ok((f64::NEG_INFINITY, vec![0.0; d]));
        }
        log_p += td;
        for i in 0..d { grad[i] += td_grad[i]; }

        // Gamma density gradient: d/dθ log Gamma(g; dt/σ², σ²/dt).
        // Currently zero because σ² is typically a constant (not estimated).
        // When σ² depends on estimated params, this needs sigma_sq_grad
        // expressions from the compiler. The LL includes the gamma term
        // (re-enabled after fixing per-transition gamma indexing), but since
        // it's constant w.r.t. θ for typical models, the gradient is zero
        // and the objective/gradient agreement holds.

        // Accumulate flows
        for (i, &f) in rec.flows.iter().enumerate() {
            cum_flows[i] += f;
        }

        // Observation density (gradient is zero when obs params are fixed).
        // Snapshot projections read post-step state from the trajectory record.
        if let Some(&obs_idx) = obs_at_substep.get(&s) {
            log_p += obs_model.log_likelihood_from_flows_and_counts(
                &cum_flows, &rec.counts_after, obs_idx, params);
            cum_flows.fill(0);
        }
    }

    Ok((log_p, grad))
}
