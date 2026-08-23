//! `nuts` on `ode` — gradient-based Bayesian sampling of the deterministic ODE
//! likelihood (gh#275 Phase 2).
//!
//! With [`det_grad`](crate::inference::ode_grad::det_grad) in hand, the sampler is
//! nearly free: the NUTS core ([`nuts_step`], [`DualAveraging`], [`MassMatrix`])
//! is unchanged, and the posterior target is the standard composition
//!
//! ```text
//! log p(z) = log p(y | θ(z)) + Σ_i [ log π(θ_i(z)) + log|dθ_i/dz_i| ]
//! ∇_z      = ∇_θ log p · dθ/dz + Σ_i [ ∇_z log π + d/dz log|dθ/dz| ]
//! ```
//!
//! where the data term `(log p(y|θ), ∇_θ log p)` is [`det_grad`], and the prior +
//! change-of-variables terms reuse PGAS's authorities verbatim
//! ([`prior_log_density_and_grad_z`](crate::inference::pgas::prior_log_density_and_grad_z),
//! `EstimatedParam::{transform_deriv, log_jacobian, jacobian_grad}`) — so the
//! ODE-NUTS target and the PGAS-NUTS target cannot diverge in how they build the
//! posterior. `nuts` is a DETERMINISTIC-likelihood method: on a stochastic backend
//! gradient-NUTS lives inside PGAS, so this path is `ode`-only (routed by the
//! method registry + the §1h capability gate, which `det_grad` runs).

use crate::compiled_model::CompiledModel;
use crate::error::SimError;
use crate::inference::nuts::{nuts_step, MassMetric, NUTSConfig, WarmupAdapter};
use crate::inference::ode_grad::det_grad;
use crate::inference::pgas::prior_log_density_and_grad_z;
use crate::inference::pmmh::Prior;
use crate::inference::types::EstimatedParam;
use crate::inference::MultiStreamObsModel;
use crate::rng::StatefulRng;

/// ODE-NUTS run configuration.
pub struct OdeNutsConfig {
    /// Warm-up (adaptation) iterations — step size via dual averaging + mass
    /// matrix via warm-up moments; discarded, not kept as draws.
    pub n_warmup: usize,
    /// Post-warmup posterior draws to keep.
    pub n_samples: usize,
    /// Maximum NUTS tree depth (doublings).
    pub max_tree_depth: usize,
    /// Target mean acceptance probability for dual averaging (Stan default 0.8).
    pub target_accept: f64,
    /// Initial leapfrog step size (dual averaging adapts from here).
    pub init_step_size: f64,
    /// Mass-matrix adaptation strategy (see [`MassMetric`]). Default `Diagonal`.
    pub metric: MassMetric,
    /// Fixed RK4 step for the ODE integration inside `det_grad`.
    pub dt: f64,
    /// Coarse RK4 step for the unscored warm-up `[t_start, first_obs)` (gh#396
    /// follow-on). `burnin_dt <= dt` disables it (fine step throughout); a larger
    /// value integrates the transient in big steps (state + sensitivity together,
    /// so the gradient stays consistent) for the burn-in speed-up.
    pub burnin_dt: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for OdeNutsConfig {
    fn default() -> Self {
        OdeNutsConfig {
            n_warmup: 500,
            n_samples: 500,
            max_tree_depth: 10,
            target_accept: 0.8,
            init_step_size: 0.1,
            metric: MassMetric::Diagonal,
            dt: 1.0,
            burnin_dt: 1.0, // = dt ⇒ off
            seed: 0,
        }
    }
}

/// ODE-NUTS result.
pub struct OdeNutsResult {
    /// Post-warmup posterior draws on the NATURAL scale — one row per sample,
    /// columns in `estimated` order.
    pub samples: Vec<Vec<f64>>,
    /// Data log-likelihood `log p(y | θ)` at each kept sample (trace column).
    pub sample_loglik: Vec<f64>,
    /// Log-posterior (data + prior + Jacobian) at each kept sample (trace column).
    pub sample_logpost: Vec<f64>,
    /// Whether each kept sample was a divergent transition (trace diagnostic).
    pub sample_divergent: Vec<bool>,
    /// Number of divergent transitions among the KEPT samples (the E-BFMI /
    /// boundary canary — a nonzero count means the step is too large or the
    /// geometry is pathological).
    pub n_divergent: usize,
    /// Mean acceptance probability over the sampling phase.
    pub mean_accept: f64,
    /// The adapted step size carried out of warm-up.
    pub step_size: f64,
    /// Mean NUTS tree depth over the sampling phase — the anisotropy canary. A
    /// value near `max_tree_depth` means the sampler is building maximal trees
    /// (each a full augmented-ODE solve per leapfrog step), the signature of a
    /// posterior the metric has not tamed; a healthy fit sits well below the cap.
    pub mean_tree_depth: f64,
    /// Sampling draws that hit the `max_tree_depth` cap (a U-turn was never found
    /// within the allowed doublings). Nonzero ⇒ slow + biased; raise the cap or
    /// improve the metric/reparameterization.
    pub max_depth_hits: usize,
    /// Warm-up sweeps that hit the `max_tree_depth` cap. Counted separately
    /// because a chain stuck at max depth *through warm-up* (each an expensive
    /// augmented-ODE solve) is the "hung chain" symptom — and it never reaches
    /// sampling, so `max_depth_hits` alone would report a healthy-looking zero.
    pub warmup_max_depth_hits: usize,
}

/// Which phase a [`NutsProgress`] tick comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NutsPhase {
    Warmup,
    Sampling,
}

/// Per-iteration snapshot handed to a [`NutsProgress`] callback — enough for a
/// runner to stream a live trace row and for a watcher to distinguish a
/// slow-gradient run (small `tree_depth`) from a max-tree-depth blow-up
/// (`tree_depth` pinned at the cap).
pub struct NutsIter {
    pub phase: NutsPhase,
    /// 0-based index within the phase.
    pub iter: usize,
    /// Total iterations in this phase (`n_warmup` or `n_samples`).
    pub total: usize,
    /// NUTS tree depth reached this iteration (leapfrog steps ≈ `2^tree_depth`).
    pub tree_depth: usize,
    pub divergent: bool,
    pub step_size: f64,
    pub log_posterior: f64,
    /// Data log-likelihood at the current draw (sampling phase; `NaN` in warm-up
    /// where the per-iter data loglik is not separately recomputed).
    pub loglik: f64,
    /// Current draw on the natural scale, in `estimated` order.
    pub params_natural: Vec<f64>,
}

/// Optional per-iteration progress callback for a long ODE-NUTS run — the
/// ODE-NUTS analogue of if2's
/// [`ProgressCallback`](crate::inference::if2::ProgressCallback). A runner uses
/// it to stream per-draw trace rows (so `chain_N/trace.tsv` fills as the run
/// proceeds instead of appearing only at the end) and to print a throttled
/// progress line. Carries `tree_depth`, which if2's callback has no analogue of
/// and which is the key ODE-NUTS diagnostic, so it is a sibling seam, not a
/// reuse of if2's tuple.
///
/// No `Sync` bound: the callback fires serially within a single run (the leapfrog
/// loop is sequential); cross-chain parallelism happens above this call, each
/// chain owning its own callback and trace writer.
pub type NutsProgress<'a> = Option<&'a dyn Fn(&NutsIter)>;

/// Sample `p(θ | y) ∝ p(y | θ, ODE skeleton) · π(θ)` with NUTS.
///
/// `params_base` is the full model parameter vector (fixed params at their values,
/// estimated params at their initial values). `estimated` carries each estimated
/// parameter's model index, transform, and starting value; `priors[i]` is the
/// prior for `estimated[i]`. Refuses (via `det_grad`'s §1h gate) any model the ODE
/// gradient cannot handle soundly, before sampling begins.
pub fn run_ode_nuts(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    params_base: &[f64],
    estimated: &[EstimatedParam],
    priors: &[Prior],
    config: &OdeNutsConfig,
) -> Result<OdeNutsResult, SimError> {
    run_ode_nuts_with_progress(
        compiled, obs_model, obs_times, params_base, estimated, priors, config, None,
    )
}

/// As [`run_ode_nuts`], with an optional per-iteration [`NutsProgress`] callback
/// (trace streaming + a throttled progress line for long runs).
#[allow(clippy::too_many_arguments)]
pub fn run_ode_nuts_with_progress(
    compiled: &CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    params_base: &[f64],
    estimated: &[EstimatedParam],
    priors: &[Prior],
    config: &OdeNutsConfig,
    on_iter: NutsProgress,
) -> Result<OdeNutsResult, SimError> {
    let d = estimated.len();
    if priors.len() != d {
        return Err(SimError::Validation(format!(
            "run_ode_nuts: {} priors for {} estimated parameters",
            priors.len(),
            d
        )));
    }
    let estimated_to_model: Vec<usize> = estimated.iter().map(|e| e.index).collect();

    // The z-space posterior target: data term from det_grad, prior + Jacobian from
    // the shared PGAS authorities. det_grad runs the §1h capability gate on every
    // call — a static model scan, negligible beside the ODE solve — so an
    // unsupported model is refused up front on the first evaluation. Returns
    // `(log_posterior, grad_z, data_loglik)`: the data loglik `ll` is the trace's
    // `log_likelihood` column and comes free from the same solve that produced the
    // gradient — so we never re-solve the ODE just to fill that column.
    let target = |z: &[f64]| -> Result<(f64, Vec<f64>, f64), SimError> {
        let mut params = params_base.to_vec();
        for (i, ep) in estimated.iter().enumerate() {
            params[ep.index] = ep.from_transformed(z[i]);
        }
        let (ll, ll_grad_theta) = det_grad(
            compiled, obs_model, obs_times, config.dt, config.burnin_dt, &params,
            &estimated_to_model,
        )?;

        let mut log_p = ll;
        let mut grad_z = vec![0.0; d];
        for i in 0..d {
            let z_i = z[i];
            let theta = params[estimated[i].index];
            let dtheta_dz = estimated[i].transform_deriv(z_i);
            // data term (chain rule to z)
            grad_z[i] += ll_grad_theta[i] * dtheta_dz;
            // prior (value + z-gradient), the SAME authority PGAS-NUTS uses
            let (prior_val, prior_grad_z) =
                prior_log_density_and_grad_z(&priors[i], &estimated[i], theta, z_i);
            log_p += prior_val;
            grad_z[i] += prior_grad_z;
            // change of variables
            log_p += estimated[i].log_jacobian(z_i);
            grad_z[i] += estimated[i].jacobian_grad(z_i);
        }
        Ok((log_p, grad_z, ll))
    };

    // A finite-fallback wrapper: a non-finite target (a model that blew up at this
    // θ, or a gate refusal on the first call) becomes `-inf`, steering the sampler
    // away rather than crashing. The gate error is surfaced by the up-front probe
    // below. Carries the data loglik through for the accept-path trace column.
    // BOTH the value AND every gradient component must be finite: a coarse/stiff
    // step can blow the (signed, unclamped) sensitivity to inf/NaN while the
    // clamped value stays finite; passing that gradient to the leapfrog integrator
    // poisons the momentum (NaN trajectory) instead of yielding a clean rejection.
    let target_or_neg_inf = |z: &[f64]| -> (f64, Vec<f64>, f64) {
        match target(z) {
            Ok((lp, g, ll)) if lp.is_finite() && g.iter().all(|x| x.is_finite()) => (lp, g, ll),
            _ => (f64::NEG_INFINITY, vec![0.0; d], f64::NEG_INFINITY),
        }
    };

    // The NUTS core takes a plain `Fn -> (log_p, grad)`; adapt by dropping the
    // data-loglik component (the core never needs it).
    let nuts_target = |z: &[f64]| -> (f64, Vec<f64>) {
        let (lp, g, _) = target_or_neg_inf(z);
        (lp, g)
    };

    // z at the estimated parameters' starting values.
    let mut z: Vec<f64> = estimated.iter().map(|e| e.to_transformed(e.initial)).collect();

    // Probe once so a capability-gate refusal (or a non-finite start) is a real
    // error, not a silent all-divergent run. `cur_loglik` (the trace's data-loglik
    // column) is maintained from here on out of the accept-path solve.
    let (mut log_p, mut grad, mut cur_loglik) = target(&z)?;
    if !log_p.is_finite() {
        return Err(SimError::Validation(
            "run_ode_nuts: the posterior is not finite at the initial parameters — the \
             model blew up or the data is incompatible with the starting point. Check the \
             initial values and bounds."
                .to_string(),
        ));
    }

    let mut rng = StatefulRng::new(config.seed);

    // Natural-scale parameter vector at the current z (for progress rows).
    let natural = |z: &[f64]| -> Vec<f64> {
        estimated.iter().enumerate().map(|(i, ep)| ep.from_transformed(z[i])).collect()
    };

    // Stan-style windowed warm-up (see [`WarmupAdapter`]): step-size dual
    // averaging every sweep, with the mass matrix re-estimated over expanding
    // windows so the covariance converges to the true posterior geometry instead
    // of being frozen once from a poorly-mixed identity-mass phase. The adapter
    // owns the schedule/Welford/dual-averaging; the loop just runs `nuts_step`
    // under its current (step, metric) and feeds back the drawn position.
    let mut adapter = WarmupAdapter::new(
        config.metric, d, config.n_warmup, config.init_step_size, config.target_accept,
    );
    let mut warmup_max_depth_hits = 0usize;
    for sweep in 0..config.n_warmup {
        let cfg = NUTSConfig {
            max_tree_depth: config.max_tree_depth,
            step_size: adapter.step_size(),
            mass_matrix: adapter.mass().clone(),
        };
        let r = nuts_step(&z, log_p, &grad, &cfg, &nuts_target, &mut rng);
        if r.tree_depth >= config.max_tree_depth {
            warmup_max_depth_hits += 1;
        }
        if r.accepted {
            z = r.params;
            log_p = r.log_posterior;
            // One augmented solve at the accepted z yields both the gradient (for
            // the next step) and the data loglik (kept current for sampling).
            let (_, g, ll) = target_or_neg_inf(&z);
            grad = g;
            cur_loglik = ll;
        }

        let froze = adapter.observe(sweep, &z, r.mean_accept_prob);
        if froze && log::log_enabled!(log::Level::Info) {
            let sds = adapter.metric_sd();
            let named: Vec<String> = estimated
                .iter()
                .zip(&sds)
                .map(|(ep, sd)| format!("{}={:.4}", ep.name, sd))
                .collect();
            log::info!(target: "nuts",
                "metric re-estimated at warmup {}/{}: z-sd [{}]",
                sweep + 1, config.n_warmup, named.join(", "));
        }

        if let Some(cb) = on_iter {
            cb(&NutsIter {
                phase: NutsPhase::Warmup,
                iter: sweep,
                total: config.n_warmup,
                tree_depth: r.tree_depth,
                divergent: r.divergent,
                step_size: adapter.step_size(),
                log_posterior: log_p,
                loglik: f64::NAN,
                params_natural: natural(&z),
            });
        }
    }
    adapter.finalize();
    let step_size = adapter.step_size();
    let mass = adapter.into_mass();

    // Sampling. `cur_loglik` (the trace's data-loglik column) was left current by
    // the last warm-up accept and is refreshed from the accept-path solve below —
    // no separate ODE solve just to fill that column.
    let cfg = NUTSConfig { max_tree_depth: config.max_tree_depth, step_size, mass_matrix: mass };
    let mut samples: Vec<Vec<f64>> = Vec::with_capacity(config.n_samples);
    let mut sample_loglik: Vec<f64> = Vec::with_capacity(config.n_samples);
    let mut sample_logpost: Vec<f64> = Vec::with_capacity(config.n_samples);
    let mut sample_divergent: Vec<bool> = Vec::with_capacity(config.n_samples);
    let mut n_divergent = 0usize;
    let mut accept_sum = 0.0f64;
    let mut tree_depth_sum = 0usize;
    let mut max_depth_hits = 0usize;
    for i in 0..config.n_samples {
        let r = nuts_step(&z, log_p, &grad, &cfg, &nuts_target, &mut rng);
        accept_sum += r.mean_accept_prob;
        tree_depth_sum += r.tree_depth;
        if r.tree_depth >= config.max_tree_depth {
            max_depth_hits += 1;
        }
        let divergent = r.divergent;
        if divergent {
            n_divergent += 1;
        }
        if r.accepted {
            z = r.params;
            log_p = r.log_posterior;
            // One solve → gradient (for the next step) + data loglik (trace column).
            let (_, g, ll) = target_or_neg_inf(&z);
            grad = g;
            cur_loglik = ll;
        }
        // Record the natural-scale parameter vector for this draw + diagnostics.
        let params_nat = natural(&z);
        if let Some(cb) = on_iter {
            cb(&NutsIter {
                phase: NutsPhase::Sampling,
                iter: i,
                total: config.n_samples,
                tree_depth: r.tree_depth,
                divergent,
                step_size,
                log_posterior: log_p,
                loglik: cur_loglik,
                params_natural: params_nat.clone(),
            });
        }
        samples.push(params_nat);
        sample_loglik.push(cur_loglik);
        sample_logpost.push(log_p);
        sample_divergent.push(divergent);
    }

    Ok(OdeNutsResult {
        samples,
        sample_loglik,
        sample_logpost,
        sample_divergent,
        n_divergent,
        mean_accept: if config.n_samples > 0 { accept_sum / config.n_samples as f64 } else { 0.0 },
        step_size,
        mean_tree_depth: if config.n_samples > 0 {
            tree_depth_sum as f64 / config.n_samples as f64
        } else {
            0.0
        },
        max_depth_hits,
        warmup_max_depth_hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::multi_stream_obs::{StreamProjection, StreamSpec};
    use crate::inference::types::Transform;
    use crate::inference::{dense_cells, BoundObs};
    use ir::deriv::DerivEntry;
    use ir::expr::{ConstExpr, Expr, ParamExpr, ProjectedExpr};
    use ir::observation::{Likelihood, NegBinomialLikelihood};
    use ir::Diffable;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// SEIR with an incidence stream `negbin(mean = incidence, dispersion = k)` and
    /// explicit init — the same shape the det_grad oracle validates, here used to
    /// recover a known `beta` from synthetic incidence data.
    fn seir_incidence_model() -> ir::Model {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = std::path::PathBuf::from(&manifest)
            .join("../../../ocaml/golden/seir_observations.ir.json");
        let mut model: ir::Model =
            ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        model.initial_conditions = ir::model::InitialConditions::Explicit(HashMap::from([
            ("S".to_string(), 9990.0),
            ("E".to_string(), 0.0),
            ("I".to_string(), 10.0),
            ("R".to_string(), 0.0),
        ]));
        model.observations.retain(|o| o.name == "weekly_cases");
        for om in &mut model.observations {
            om.likelihood = Likelihood::NegBinomial(NegBinomialLikelihood {
                mean: Diffable {
                    expr: Expr::Projected(ProjectedExpr { projected: () }),
                    grad: HashMap::new(),
                    // mean IS projected → ∂mean/∂projected = 1 (what the compiler emits
                    // for a bare `Projected` argument).
                    proj_grad: Some(DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 }))),
                },
                dispersion: Diffable {
                    expr: Expr::Param(ParamExpr { param: "k".to_string() }),
                    grad: HashMap::from([(
                        "k".to_string(),
                        DerivEntry::Grad(Expr::Const(ConstExpr { value: 1.0 })),
                    )]),
                    proj_grad: None,
                },
            });
        }
        model.simulation.t_end = 60.0;
        model
    }

    fn compiled(true_beta: f64) -> (Arc<CompiledModel>, Vec<f64>) {
        let mut model = seir_incidence_model();
        for p in &mut model.parameters {
            let v = match p.name.as_str() {
                "beta" => true_beta,
                "sigma" => 0.2, "gamma" => 0.1, "k" => 40.0,
                "rho" => 0.5, "p_detect" => 0.8, "N0" => 10000.0, "I0" => 10.0,
                _ => 0.5,
            };
            p.value = p.value.with_value(v);
        }
        let cm = Arc::new(CompiledModel::new(model).unwrap());
        let params: Vec<f64> = cm
            .model
            .parameters
            .iter()
            .map(|p| p.value.resolved_value().unwrap())
            .collect();
        (cm, params)
    }

    /// End-to-end: `nuts` on `ode` recovers a known `beta` from synthetic weekly
    /// incidence data. Data is generated at the true `beta` (so it is informative
    /// and identifies `beta`); NUTS starts from a wrong `beta` and must return a
    /// posterior mean near the truth, with no divergences.
    #[test]
    fn ode_nuts_recovers_known_beta_from_incidence() {
        let true_beta = 0.9;
        let (cm, true_params) = compiled(true_beta);
        let dt = 1.0;
        let obs_times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();

        // Synthesize incidence data at the true params (near the mode).
        let beta_idx = cm.param_index["beta"];
        let (int_s0, _) = cm.initial_state_mean(&true_params).unwrap();
        let seed = vec![0.0; int_s0.counts.len()];
        let recs = crate::ode::integrate_obs_sensitivity(
            &cm, &true_params, &[beta_idx], &seed, &crate::config::OdeConfig {
                t_start: 0.0, t_end: 60.0, dt,
            }, &obs_times, dt,
        )
        .unwrap();
        let infection_idx = cm.model.transitions.iter().position(|t| t.name == "infection").unwrap();
        let data: Vec<f64> = recs.iter().map(|r| r.inc[infection_idx].round()).collect();
        assert!(data.iter().sum::<f64>() > 100.0, "incidence data must be substantial");

        // Build the obs model.
        let om = cm.model.observations[0].clone();
        let projection = StreamProjection::from_ir(&om.projection, &cm, &om.name).unwrap();
        let spec = StreamSpec {
            projection,
            ir_model: om,
            observations: dense_cells(data),
            obs_times: obs_times.clone(),
            aux: vec![],
        };
        let obs_model =
            MultiStreamObsModel::new(BoundObs::bind(vec![spec]).unwrap().0, cm.clone()).unwrap();

        // Estimate beta only, starting from a wrong value; flat prior, log transform.
        let estimated = vec![EstimatedParam {
            name: "beta".to_string(),
            index: beta_idx,
            initial: 0.4, // wrong start
            rw_sd: 0.0,
            transform: Transform::Log { lo: 0.05, hi: 5.0 },
            lower: 0.05,
            upper: 5.0,
            rw_sd_auto: false,
            ivp: false,
        }];
        let priors = vec![Prior::Fixed(crate::inference::prior::Density::Flat)];

        // Start the full param vector at the (correct) fixed values but the WRONG beta.
        let mut base = true_params.clone();
        base[beta_idx] = 0.4;

        let config = OdeNutsConfig {
            n_warmup: 150,
            n_samples: 250,
            max_tree_depth: 7,
            target_accept: 0.8,
            init_step_size: 0.2,
            metric: MassMetric::Diagonal,
            dt,
            burnin_dt: dt, // off
            seed: 20260707,
        };
        let result =
            run_ode_nuts(&cm, &obs_model, &obs_times, &base, &estimated, &priors, &config).unwrap();

        let post_mean: f64 =
            result.samples.iter().map(|s| s[0]).sum::<f64>() / result.samples.len() as f64;
        eprintln!(
            "ode-nuts: true beta = {true_beta}, posterior mean = {post_mean:.4}, \
             divergences = {}, mean_accept = {:.3}, step = {:.4}",
            result.n_divergent, result.mean_accept, result.step_size
        );

        assert!(
            (post_mean - true_beta).abs() < 0.12,
            "posterior mean beta {post_mean:.4} did not recover the truth {true_beta} \
             — the ODE-NUTS gradient/target is not steering the sampler correctly"
        );
        assert_eq!(
            result.n_divergent, 0,
            "a well-specified 1D recovery should have no divergences (got {})",
            result.n_divergent
        );
    }

    /// Diagonal-metric warm-up adapts to an anisotropic posterior. On a
    /// 2-parameter fit (`beta`, `gamma`) whose z-posteriors have different
    /// widths, an unadapted (`Unit`) metric must size its single step to the
    /// tightest direction, so it steps small; `Diagonal` rescales each parameter
    /// and takes a much larger step at the *same* acceptance. This is the
    /// mechanism behind the ODE-NUTS fix (gh#275; garki friction F21): on the real
    /// Garki posterior the diagonal metric took 7.3× larger steps (0.32 vs 0.044)
    /// at accept≈0.91 and freed the wide-posterior parameter `a2` that identity
    /// mass left stuck at its bound. The A/B *is* the regression guard: if
    /// adaptation silently fell back to identity, `Diagonal`'s step would match
    /// `Unit`'s and the step-enlargement assertion would fail.
    #[test]
    fn diagonal_metric_enlarges_step_on_anisotropic_posterior() {
        let true_beta = 0.9;
        let (cm, true_params) = compiled(true_beta);
        let dt = 1.0;
        let obs_times: Vec<f64> = (1..=8).map(|w| (w * 7) as f64).collect();
        let beta_idx = cm.param_index["beta"];
        let gamma_idx = cm.param_index["gamma"];

        // Synthesize incidence data at the true params.
        let (int_s0, _) = cm.initial_state_mean(&true_params).unwrap();
        let seed = vec![0.0; int_s0.counts.len()];
        let recs = crate::ode::integrate_obs_sensitivity(
            &cm, &true_params, &[beta_idx], &seed,
            &crate::config::OdeConfig { t_start: 0.0, t_end: 60.0, dt }, &obs_times, dt,
        )
        .unwrap();
        let inf_idx = cm.model.transitions.iter().position(|t| t.name == "infection").unwrap();
        let data: Vec<f64> = recs.iter().map(|r| r.inc[inf_idx].round()).collect();

        let om = cm.model.observations[0].clone();
        let projection = StreamProjection::from_ir(&om.projection, &cm, &om.name).unwrap();
        let spec = StreamSpec {
            projection, ir_model: om, observations: dense_cells(data),
            obs_times: obs_times.clone(), aux: vec![],
        };
        let obs_model =
            MultiStreamObsModel::new(BoundObs::bind(vec![spec]).unwrap().0, cm.clone()).unwrap();

        // Estimate beta AND gamma — both interior-identified (no boundary railing),
        // with differently-scaled z-posteriors: the anisotropy the metric adapts to.
        let estimated = vec![
            EstimatedParam {
                name: "beta".to_string(), index: beta_idx, initial: 0.6, rw_sd: 0.0,
                transform: Transform::Log { lo: 0.05, hi: 5.0 }, lower: 0.05, upper: 5.0,
                rw_sd_auto: false, ivp: false,
            },
            EstimatedParam {
                name: "gamma".to_string(), index: gamma_idx, initial: 0.2, rw_sd: 0.0,
                transform: Transform::Log { lo: 0.01, hi: 1.0 }, lower: 0.01, upper: 1.0,
                rw_sd_auto: false, ivp: false,
            },
        ];
        let priors = vec![
            Prior::Fixed(crate::inference::prior::Density::Flat),
            Prior::Fixed(crate::inference::prior::Density::Flat),
        ];
        let mut base = true_params.clone();
        base[beta_idx] = 0.6;
        base[gamma_idx] = 0.2;

        let run = |metric: MassMetric| {
            let config = OdeNutsConfig {
                n_warmup: 300, n_samples: 200, max_tree_depth: 10,
                target_accept: 0.8, init_step_size: 0.2, metric, dt, burnin_dt: dt, seed: 20260707,
            };
            run_ode_nuts(&cm, &obs_model, &obs_times, &base, &estimated, &priors, &config).unwrap()
        };
        let unit = run(MassMetric::Unit);
        let diag = run(MassMetric::Diagonal);
        let beta_mean =
            |r: &OdeNutsResult| r.samples.iter().map(|s| s[0]).sum::<f64>() / r.samples.len() as f64;
        eprintln!(
            "anisotropy A/B: unit step={:.4} depth={:.2} accept={:.2} | \
             diag step={:.4} depth={:.2} accept={:.2} | beta unit={:.3} diag={:.3}",
            unit.step_size, unit.mean_tree_depth, unit.mean_accept,
            diag.step_size, diag.mean_tree_depth, diag.mean_accept,
            beta_mean(&unit), beta_mean(&diag),
        );

        // The mechanism: the diagonal metric lets NUTS take a meaningfully larger
        // step than identity mass at comparable acceptance. If adaptation silently
        // reverted to identity, the two steps would coincide and this would fail.
        assert!(
            diag.step_size > unit.step_size * 1.3,
            "diagonal metric should enlarge the adapted step vs unit \
             (unit {:.4} vs diag {:.4}) — mass adaptation is not live",
            unit.step_size, diag.step_size,
        );
        // Both metrics recover the well-identified beta (no correctness regression).
        assert!(
            (beta_mean(&diag) - true_beta).abs() < 0.15,
            "diagonal-metric fit must recover beta (got {:.3})", beta_mean(&diag),
        );
    }
}
