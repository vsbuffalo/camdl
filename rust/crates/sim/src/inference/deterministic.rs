//! Deterministic-skeleton inference via NLopt.
//!
//! Per gh#40 / proposal `2026-05-02-ode-backend-deterministic-inference.md`:
//! `camdl profile --backend ode` replaces the per-cell IF2 inner
//! reoptimisation with a deterministic ODE forward sim + NLopt
//! optimisation. The likelihood evaluated here is
//! `p(y | θ, deterministic skeleton)` — *not* the chain-binomial
//! (`p(y | θ)`) likelihood. In low-noise regimes (large per-cell
//! populations, near-deterministic trajectories) the two converge; in
//! high-noise regimes they don't. See the proposal's "Two likelihoods"
//! framing — that wording must surface in user-facing help text.
//!
//! NLopt's success-state semantics (proposal §"NLopt success-state
//! semantics"):
//!   * `Success`, `XtolReached`, `FtolReached` → converged
//!   * `MaxEvalReached`                        → soft failure (returned,
//!                                               but `converged = false`)
//!   * any `FailState` (`Failure`, `InvalidArgs`, …) → hard error
//!
//! The objective closure runs `OdeSim` once per call, scores the
//! resulting trajectory against a `MultiStreamObsModel`, and returns
//! `-loglik` (NLopt minimises). A failed sim (`SimError`) maps to
//! `f64::INFINITY` rather than a panic — numerically broken parameter
//! tuples are infeasible, not bugs.

use std::sync::Arc;

use nlopt::{Algorithm, Nlopt, SuccessState, Target};

use crate::compiled_model::CompiledModel;
use crate::config::{OdeConfig, SimConfig};
use crate::ode::OdeSim;
use crate::simulate::Simulate;
use crate::state::Trajectory;

use super::multi_stream_obs::MultiStreamObsModel;
use super::types::EstimatedParam;

// ── Algorithm selection ───────────────────────────────────────────────────────

/// Which NLopt algorithm to use for the deterministic inner optimisation.
///
/// Default is `Sbplx` (NLopt's `LN_SBPLX`, a robust Nelder-Mead variant)
/// — proposal §"Algorithm choice" justifies this: compartmental
/// likelihoods are smooth in the interior of the parameter box but
/// non-smooth at boundaries (degenerate states, event-timing kinks).
/// BOBYQA's quadratic trust region fails badly when smoothness breaks.
/// Sbplx is slower on truly smooth problems but doesn't fail
/// catastrophically.
#[derive(Clone, Copy, Debug)]
pub enum DetAlgorithm {
    /// `LN_SBPLX` — robust Nelder-Mead variant. Recommended default.
    Sbplx,
    /// `LN_BOBYQA` — derivative-free quadratic. Faster on smooth
    /// objectives; can fail catastrophically when smoothness breaks.
    Bobyqa,
    /// `LN_COBYLA` — supports active linear-inequality constraints.
    Cobyla,
    /// `GN_ISRES` — global, multi-modal "is this the basin?" pass. Slow.
    Isres,
    /// `GN_CRS2_LM` — global, controlled random search. Faster than ISRES.
    Crs2,
}

impl DetAlgorithm {
    /// Canonical CLI string (matches the user-facing `--optimizer` flag).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sbplx  => "sbplx",
            Self::Bobyqa => "bobyqa",
            Self::Cobyla => "cobyla",
            Self::Isres  => "isres",
            Self::Crs2   => "crs2",
        }
    }
}

impl std::str::FromStr for DetAlgorithm {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "sbplx"  => Ok(Self::Sbplx),
            "bobyqa" => Ok(Self::Bobyqa),
            "cobyla" => Ok(Self::Cobyla),
            "isres"  => Ok(Self::Isres),
            "crs2" | "crs2_lm" | "crs2-lm" => Ok(Self::Crs2),
            other => Err(format!(
                "unknown --optimizer '{}' (expected one of: \
                 sbplx, bobyqa, cobyla, isres, crs2)", other)),
        }
    }
}

impl From<DetAlgorithm> for Algorithm {
    fn from(a: DetAlgorithm) -> Self {
        match a {
            DetAlgorithm::Sbplx  => Algorithm::Sbplx,
            DetAlgorithm::Bobyqa => Algorithm::Bobyqa,
            DetAlgorithm::Cobyla => Algorithm::Cobyla,
            DetAlgorithm::Isres  => Algorithm::Isres,
            DetAlgorithm::Crs2   => Algorithm::Crs2Lm,
        }
    }
}

// ── Config and result types ───────────────────────────────────────────────────

/// Configuration for `optimize_det`.
#[derive(Clone, Debug)]
pub struct DetOptConfig {
    /// Which NLopt algorithm to use. `Sbplx` is the recommended default.
    pub algorithm: DetAlgorithm,
    /// `xtol_rel`: stop when relative step in x falls below this. NLopt
    /// default is essentially never; we use 1e-4 by default at the call
    /// site.
    pub xtol_rel: f64,
    /// `ftol_rel`: optional relative loglik-stagnation stop. `None` →
    /// not set (NLopt picks).
    pub ftol_rel: Option<f64>,
    /// `maxeval`: hard cap on objective evaluations. Hitting this is a
    /// soft failure (`converged = false`).
    pub max_evals: usize,
    /// Forward-simulation config for `OdeSim`.
    pub ode: OdeConfig,
}

impl Default for DetOptConfig {
    fn default() -> Self {
        Self {
            algorithm: DetAlgorithm::Sbplx,
            xtol_rel: 1e-4,
            ftol_rel: None,
            max_evals: 500,
            ode: OdeConfig { t_start: 0.0, t_end: 0.0, dt: 1.0 },
        }
    }
}

/// Result of one deterministic-chain optimisation.
#[derive(Clone, Debug)]
pub struct DetOptResult {
    /// Optimised parameter values, in the order of the `estimated`
    /// slice passed to `optimize_det`. Caller is responsible for
    /// scattering back into the full parameter vector.
    pub params: Vec<f64>,
    /// Log-likelihood at the optimum (note: `loglik`, not `-loglik` —
    /// callers that want NLopt's reported minimised value should negate
    /// this themselves).
    pub loglik: f64,
    /// Number of objective evaluations.
    pub n_evals: usize,
    /// NLopt success state (or `None` if NLopt returned a `FailState`
    /// — in which case `converged = false` and the caller should treat
    /// this as a hard failure).
    pub status: Option<SuccessState>,
    /// `true` iff the run finished cleanly: status ∈ {Success,
    /// XtolReached, FtolReached}. `MaxEvalReached` and any FailState
    /// → `false`.
    pub converged: bool,
}

// ── Trajectory → loglik scoring ───────────────────────────────────────────────

/// Score a deterministic `Trajectory` against a `MultiStreamObsModel`
/// by walking the snapshots in order and matching each obs time.
///
/// Contract: the trajectory must contain a snapshot whose `t` equals
/// `obs_time(idx)` (within `t_tol = 1e-6`) for every obs idx in
/// `0..obs_model.n_observations()`. This is the same alignment that
/// the IF2/PF path implicitly relies on — the simulation must produce
/// outputs at the observation times. If the user's model has a
/// mismatched output schedule, this returns `f64::NEG_INFINITY` and
/// the optimiser will move away from that region (in practice the
/// caller should set up the OdeConfig so output_times line up; for
/// the camdl profile dispatch the IR's `output.times` is presumed to
/// already match the data file's times via the standard convention).
pub fn score_trajectory(
    obs_model: &MultiStreamObsModel,
    traj: &Trajectory,
    params: &[f64],
) -> f64 {
    use crate::inference::traits::ObservationModel;
    let n_obs = obs_model.n_observations();
    if n_obs == 0 { return 0.0; }

    let t_tol: f64 = 1e-6;
    let mut total: f64 = 0.0;
    let mut snap_cursor: usize = 0;

    for obs_idx in 0..n_obs {
        let target_t = obs_model.obs_time(obs_idx);
        // Advance cursor to the first snapshot whose t >= target_t - t_tol.
        while snap_cursor < traj.snapshots.len()
            && traj.snapshots[snap_cursor].t < target_t - t_tol
        {
            snap_cursor += 1;
        }
        if snap_cursor >= traj.snapshots.len() {
            return f64::NEG_INFINITY;
        }
        let snap = &traj.snapshots[snap_cursor];
        if (snap.t - target_t).abs() > t_tol {
            // No snapshot at this obs time — model output schedule is
            // mismatched against the data. The optimiser will see
            // -inf and move away.
            return f64::NEG_INFINITY;
        }
        // The snapshot's `flows` field carries the cumulative flow
        // counts since the previous snapshot, which is exactly the
        // per-interval incidence the FlowSum projection wants. The
        // counts slice is the integer compartment state at the
        // observation instant — what IntCompSum / Expr need.
        let cum_flows: Vec<u64> = snap.flows.counts.clone();
        total += obs_model.log_likelihood_from_flows_and_counts(
            &cum_flows, &snap.int_state.counts, obs_idx, params,
        );
    }
    total
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Run NLopt against a deterministic ODE-backed objective.
///
/// `base_params` is the full model parameter vector (focal params
/// already pinned by the caller). `estimated` is the subset of
/// parameter slots to optimise; `estimated[i].lower / upper` give the
/// box-constraint bounds. Initial values come from
/// `base_params[estimated[i].index]`.
///
/// Returns `Err` only on hard NLopt failures (FailState other than
/// MaxEvalReached, or constructor errors). MaxEvalReached is reported
/// via `DetOptResult { converged: false, ... }` so callers can apply
/// their own gate.
pub fn optimize_det(
    compiled: Arc<CompiledModel>,
    obs_model: Arc<MultiStreamObsModel>,
    base_params: &[f64],
    estimated: &[EstimatedParam],
    config: &DetOptConfig,
) -> Result<DetOptResult, String> {
    use std::cell::RefCell;

    if estimated.is_empty() {
        // Nothing to optimise — just score the deterministic trajectory
        // at base_params and report it. No NLopt invocation; n_evals=1.
        let traj = OdeSim
            .run(&compiled, base_params, 0, &SimConfig::Ode(config.ode.clone()))
            .map_err(|e| format!("ODE forward sim failed at base_params: {:?}", e))?;
        let loglik = score_trajectory(&obs_model, &traj, base_params);
        return Ok(DetOptResult {
            params: vec![],
            loglik,
            n_evals: 1,
            status: Some(SuccessState::Success),
            converged: true,
        });
    }

    // Validate bounds before the optimiser sees them — NLopt errors
    // out at optimize-time on inverted bounds, but a clear pre-flight
    // diagnostic is cheaper for the user than a `nlopt::FailState`
    // with no parameter name attached. NLopt's local derivative-free
    // algorithms (Sbplx, Bobyqa, Cobyla) accept ±∞ as "unbounded";
    // the global algorithms (Isres, Crs2) do not. We surface the
    // distinction so the user gets a clean diagnostic when --optimizer
    // isres/crs2 is passed against a model whose bounds the IR
    // populates as `[0, ∞]` by default (typical for unbounded rate
    // parameters declared without explicit `bounds: ...`).
    let global_optimizer = matches!(
        config.algorithm,
        DetAlgorithm::Isres | DetAlgorithm::Crs2,
    );
    for spec in estimated {
        if spec.lower >= spec.upper {
            return Err(format!(
                "deterministic optimiser: parameter '{}' has \
                 lower={} >= upper={} — bounds must be strictly \
                 increasing",
                spec.name, spec.lower, spec.upper,
            ));
        }
        if global_optimizer && !(spec.lower.is_finite() && spec.upper.is_finite()) {
            return Err(format!(
                "deterministic optimiser ({:?}): parameter '{}' has \
                 non-finite bounds (lower={}, upper={}); global \
                 algorithms require finite [lower, upper]. Declare \
                 explicit bounds in the model, pass --fixed {} to \
                 hold this parameter constant, or switch to a local \
                 optimiser (sbplx, bobyqa, cobyla)",
                config.algorithm, spec.name, spec.lower, spec.upper, spec.name,
            ));
        }
    }

    let n = estimated.len();
    let lower: Vec<f64> = estimated.iter().map(|e| e.lower).collect();
    let upper: Vec<f64> = estimated.iter().map(|e| e.upper).collect();

    // Initial point from base_params; clamp into [lower, upper] in
    // case the caller passed a focal-pinned vector that nudged outside
    // a nuisance bound (rare but well-defined).
    let mut x: Vec<f64> = estimated.iter()
        .map(|e| base_params[e.index].clamp(e.lower, e.upper))
        .collect();

    // Eval counter lives in user_data — NLopt 0.8's Rust wrapper has
    // no `get_numevals()` shim around `nlopt_get_numevals`, so we
    // count ourselves. RefCell because the closure can be called
    // multiple times by NLopt's internals.
    struct UserData {
        counter: RefCell<usize>,
    }
    let user_data = UserData { counter: RefCell::new(0) };

    // Capture immutable shared state by clone (Arc) so the closure is
    // 'static-ish. The objective never mutates `compiled`,
    // `obs_model`, `base_params`, `estimated`, or `config.ode`.
    let compiled_obj = Arc::clone(&compiled);
    let obs_model_obj = Arc::clone(&obs_model);
    let base_params_owned: Vec<f64> = base_params.to_vec();
    let estimated_owned: Vec<EstimatedParam> = estimated.to_vec();
    let ode_cfg = config.ode.clone();

    let objective = move |theta: &[f64], _grad: Option<&mut [f64]>, ud: &mut UserData| -> f64 {
        *ud.counter.borrow_mut() += 1;
        let mut params = base_params_owned.clone();
        for (i, e) in estimated_owned.iter().enumerate() {
            params[e.index] = theta[i];
        }
        match OdeSim.run(&compiled_obj, &params, 0, &SimConfig::Ode(ode_cfg.clone())) {
            Ok(traj) => {
                let ll = score_trajectory(&obs_model_obj, &traj, &params);
                if ll.is_finite() { -ll } else { f64::INFINITY }
            }
            Err(_) => f64::INFINITY,
        }
    };

    let mut opt = Nlopt::new(
        config.algorithm.into(),
        n,
        objective,
        Target::Minimize,
        user_data,
    );
    opt.set_lower_bounds(&lower)
        .map_err(|e| format!("nlopt set_lower_bounds failed: {:?}", e))?;
    opt.set_upper_bounds(&upper)
        .map_err(|e| format!("nlopt set_upper_bounds failed: {:?}", e))?;
    // Initial step: NLopt's local derivative-free algorithms (Sbplx,
    // Bobyqa, Cobyla) default the first probe step to `(upper - lower)
    // / 4`, which is +∞ when either bound is non-finite. Provide a
    // sensible default per dimension: 10% of bound width if both
    // finite, otherwise 10% of the initial value (or 0.1 if x = 0).
    // This keeps the optimiser exploring at a reasonable scale even
    // when the model declares an unbounded rate parameter.
    let initial_step: Vec<f64> = estimated.iter().enumerate().map(|(i, spec)| {
        if spec.lower.is_finite() && spec.upper.is_finite() {
            ((spec.upper - spec.lower) * 0.1).max(1e-6)
        } else {
            (x[i].abs() * 0.1).max(0.1)
        }
    }).collect();
    opt.set_initial_step(&initial_step)
        .map_err(|e| format!("nlopt set_initial_step failed: {:?}", e))?;
    opt.set_xtol_rel(config.xtol_rel)
        .map_err(|e| format!("nlopt set_xtol_rel failed: {:?}", e))?;
    if let Some(ftol) = config.ftol_rel {
        opt.set_ftol_rel(ftol)
            .map_err(|e| format!("nlopt set_ftol_rel failed: {:?}", e))?;
    }
    opt.set_maxeval(config.max_evals as u32)
        .map_err(|e| format!("nlopt set_maxeval failed: {:?}", e))?;

    let outcome = opt.optimize(&mut x);
    // Recover the eval counter before consuming `opt` ends the closure.
    let recovered = opt.recover_user_data();
    let n_evals = *recovered.counter.borrow();

    match outcome {
        Ok((status, neg_loglik)) => {
            let converged = matches!(
                status,
                SuccessState::Success
                    | SuccessState::XtolReached
                    | SuccessState::FtolReached
                    | SuccessState::StopValReached,
            );
            Ok(DetOptResult {
                params: x,
                loglik: -neg_loglik,
                n_evals,
                status: Some(status),
                converged,
            })
        }
        Err((fail, neg_loglik)) => {
            // Per proposal §"NLopt success-state semantics", the
            // distinction we surface is converged-vs-not. Hard
            // FailStates other than MaxEvalReached (which lives in
            // `SuccessState` in this crate's enum split, not here)
            // bubble up as Err so the caller can decide whether to
            // retry from a different start. The most common cause of
            // a generic `Failure` here is the objective returning
            // NaN/Inf at every probed point — flag that case
            // explicitly so the user knows to widen bounds or change
            // the start.
            let hint = if matches!(fail, nlopt::FailState::Failure) {
                " (most often: every probed θ produced an infeasible \
                 / non-finite loglik — try widening bounds or a \
                 different starting point)"
            } else { "" };
            Err(format!(
                "nlopt optimisation failed: {:?} (best -loglik so far \
                 = {}){}",
                fail, neg_loglik, hint,
            ))
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::Transform;
    use crate::inference::multi_stream_obs::{StreamProjection, StreamSpec};
    use ir::{
        expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr, PopSumExpr, ProjectedExpr},
        model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
        observation::{
            Likelihood, ObservationModel as IrObservationModel, ObservationSchedule, PoissonLikelihood,
            Projection, RegularSchedule,
        },
        parameter::Parameter,
        transition::{Transition, StoichiometryEntry, DrawMethod},
        Model,
    };
    use std::collections::HashMap;

    /// Build a tiny SIR model with a Poisson-on-incidence obs stream.
    /// Used both to generate "data" via OdeSim and to test that
    /// `optimize_det` recovers the MLE.
    fn sir_model(beta: f64, gamma: f64) -> Model {
        let times: Vec<f64> = (0..=20).map(|i| i as f64).collect();
        Model {
            name: "sir_det_test".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
                Compartment { name: "R".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![
                Transition {
                    name: "infection".into(),
                    stoichiometry: vec![
                        StoichiometryEntry("S".into(), -1),
                        StoichiometryEntry("I".into(), 1),
                    ],
                    rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Div,
                        left: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                            op: BinOp::Mul,
                            left: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                                op: BinOp::Mul,
                                left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                                right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                            }})),
                            right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                        }})),
                        right: Box::new(Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] })),
                    }}),
                    metadata: None,
                    draw_method: DrawMethod::Poisson, rate_grad: Default::default(),
                },
                Transition {
                    name: "recovery".into(),
                    stoichiometry: vec![
                        StoichiometryEntry("I".into(), -1),
                        StoichiometryEntry("R".into(), 1),
                    ],
                    rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                    }}),
                    metadata: None,
                    draw_method: DrawMethod::Poisson, rate_grad: Default::default(),
                },
            ],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![IrObservationModel {
                name: "cases".into(),
                data_stream: "cases".into(),
                schedule: ObservationSchedule::Regular(RegularSchedule {
                    start: 1.0, step: 1.0, end: 20.0,
                }),
                projection: Projection::CumulativeFlow("infection".into()),
                likelihood: Likelihood::Poisson(PoissonLikelihood {
                    rate: Expr::Projected(ProjectedExpr { projected: () }),
                }),
            }],
            parameters: vec![
                Parameter { name: "beta".into(),  value: Some(beta),  bounds: Some((0.05, 3.0)),  prior: None, transform: None, initial_value: None, param_kind: None, param_dim: None, hierarchical: None },
                Parameter { name: "gamma".into(), value: Some(gamma), bounds: Some((0.05, 1.0)), prior: None, transform: None, initial_value: None, param_kind: None, param_dim: None, hierarchical: None },
            ],
            initial_conditions: InitialConditions::Explicit({
                let mut m = HashMap::new();
                m.insert("S".into(), 9990.0);
                m.insert("I".into(), 10.0);
                m
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(times),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 20.0,
                time_semantics: "continuous".into(),
                dt: Some(0.1), rng_seed: Some(0),
            },
            presets: vec![],
            model_structure: None, balance: None,
        }
    }

    /// Run OdeSim and project per-obs incidence — used as the ground-truth
    /// "data" for the MLE-recovery test. Returns one count per obs time
    /// (rounded to integer), matching what `score_trajectory` expects.
    fn deterministic_data(compiled: &CompiledModel) -> Vec<f64> {
        let cfg = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: 20.0, dt: 0.1 });
        let traj = OdeSim.run(compiled, &compiled.default_params, 0, &cfg).unwrap();
        // Skip the t=0 snapshot (no incidence yet), take the next n_obs.
        // The Poisson likelihood expects integer-shaped data.
        traj.snapshots.iter()
            .filter(|s| s.t > 1e-9)
            .take(20)
            .map(|s| s.flows.counts[0] as f64)
            .collect()
    }

    fn build_obs_model(compiled: Arc<CompiledModel>, data: Vec<f64>) -> MultiStreamObsModel {
        let obs_times: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let stream_spec = StreamSpec {
            projection: StreamProjection::FlowSum(vec![0]),  // infection flow
            ir_model: compiled.model.observations[0].clone(),
            observations: data,
            obs_times,
        };
        MultiStreamObsModel::new(vec![stream_spec], compiled).unwrap()
    }

    #[test]
    fn recovers_mle_on_synthetic_ode_data() {
        // Generate data at the true (β, γ), then start the optimiser
        // from a dispersed point and verify it walks back. We
        // optimise β only (with γ fixed at truth) — joint (β, γ)
        // recovery on the SIR Poisson-incidence likelihood has a
        // strong R₀=β/γ ridge that pomp practitioners sidestep by
        // either profiling on R₀ or jointly fitting both with a tight
        // observation-noise constraint; for unit-test purposes the
        // single-parameter recovery is the right scope (it's what
        // `optimize_det` will be doing in profile's per-cell role
        // anyway, where focal params are pinned).
        let true_beta = 0.6;
        let true_gamma = 0.2;
        let model = sir_model(true_beta, true_gamma);
        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let data = deterministic_data(&compiled);
        // Sanity: the data should have non-trivial epidemic dynamics.
        assert!(data.iter().sum::<f64>() > 100.0, "synthetic data is degenerate: {:?}", data);

        let obs_model = Arc::new(build_obs_model(Arc::clone(&compiled), data));

        let beta_idx = compiled.param_index["beta"];
        let gamma_idx = compiled.param_index["gamma"];
        let mut start_params = compiled.default_params.clone();
        start_params[beta_idx] = 1.2;             // wrong by 2x
        start_params[gamma_idx] = true_gamma;     // pinned at truth

        let estimated = vec![
            EstimatedParam {
                name: "beta".into(), index: beta_idx, initial: 1.2,
                rw_sd: 0.0, transform: Transform::None,
                lower: 0.05, upper: 3.0, ivp: false, rw_sd_auto: false,
            },
        ];
        let config = DetOptConfig {
            algorithm: DetAlgorithm::Sbplx,
            xtol_rel: 1e-7,
            ftol_rel: None,
            max_evals: 5000,
            ode: OdeConfig { t_start: 0.0, t_end: 20.0, dt: 0.1 },
        };

        // Reference loglik at the true (β, γ) — sanity check that the
        // synthetic data actually has its peak there before we ask the
        // optimiser to find it.
        let mut true_p = compiled.default_params.clone();
        true_p[beta_idx] = true_beta;
        true_p[gamma_idx] = true_gamma;
        let true_traj = OdeSim.run(&compiled, &true_p, 0,
            &SimConfig::Ode(config.ode.clone())).unwrap();
        let true_ll = score_trajectory(&obs_model, &true_traj, &true_p);
        eprintln!("loglik at true (β={}, γ={}) = {:.3}", true_beta, true_gamma, true_ll);

        let result = optimize_det(
            Arc::clone(&compiled), Arc::clone(&obs_model),
            &start_params, &estimated, &config,
        ).unwrap();

        let beta_hat = result.params[0];
        eprintln!("det-MLE: β={:.4} (true {}), loglik={:.3}, \
                   n_evals={}, status={:?}, converged={}",
            beta_hat, true_beta,
            result.loglik, result.n_evals, result.status, result.converged);
        assert!(result.converged, "Sbplx should converge on a smooth synthetic problem");
        // Loglik must be at least as good as the truth — finite-sample
        // MLE on integer-rounded incidence may not exactly equal the
        // data-generating β, but it cannot fall meaningfully short.
        assert!(result.loglik >= true_ll - 1.0,
            "optimiser found loglik {:.3} but truth gives {:.3} \
             — failed to walk back to the basin",
            result.loglik, true_ll);
        assert!((beta_hat - true_beta).abs() < 0.02,
            "β not recovered: got {:.4}, true {}", beta_hat, true_beta);
    }

    #[test]
    fn loglik_responds_to_param_changes() {
        // Sanity check that score_trajectory responds to β. If two
        // different β values give the same trajectory the optimiser
        // can never distinguish them — and that would be a pre-flight
        // bug, not a Sbplx bug.
        let model = sir_model(0.6, 0.2);
        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let data = deterministic_data(&compiled);
        let obs_model = Arc::new(build_obs_model(Arc::clone(&compiled), data));

        let beta_idx = compiled.param_index["beta"];
        let cfg = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: 20.0, dt: 0.1 });

        let mut p_a = compiled.default_params.clone();
        p_a[beta_idx] = 0.6;
        let mut p_b = compiled.default_params.clone();
        p_b[beta_idx] = 1.2;

        let traj_a = OdeSim.run(&compiled, &p_a, 0, &cfg).unwrap();
        let traj_b = OdeSim.run(&compiled, &p_b, 0, &cfg).unwrap();

        eprintln!("traj_a snapshots: {}, traj_b snapshots: {}",
            traj_a.snapshots.len(), traj_b.snapshots.len());
        for i in [0usize, 1, 5, 10, 19].iter().filter(|&&i| i < traj_a.snapshots.len()) {
            eprintln!("  t={}: traj_a flows={:?} counts={:?} | traj_b flows={:?} counts={:?}",
                traj_a.snapshots[*i].t,
                traj_a.snapshots[*i].flows.counts,
                traj_a.snapshots[*i].int_state.counts,
                traj_b.snapshots[*i].flows.counts,
                traj_b.snapshots[*i].int_state.counts);
        }
        let ll_a = score_trajectory(&obs_model, &traj_a, &p_a);
        let ll_b = score_trajectory(&obs_model, &traj_b, &p_b);
        eprintln!("loglik β=0.6: {}, loglik β=1.2: {}", ll_a, ll_b);
        assert_ne!(ll_a, ll_b, "loglik must respond to β");
    }

    #[test]
    fn objective_is_deterministic() {
        // NLopt assumes a deterministic objective. The same θ evaluated
        // twice must give bitwise-identical loglik.
        let model = sir_model(0.5, 0.2);
        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let data = deterministic_data(&compiled);
        let obs_model = Arc::new(build_obs_model(Arc::clone(&compiled), data));

        let cfg = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: 20.0, dt: 0.1 });
        let p = compiled.default_params.clone();
        let traj_a = OdeSim.run(&compiled, &p, 0, &cfg).unwrap();
        let traj_b = OdeSim.run(&compiled, &p, 12345, &cfg).unwrap();
        let ll_a = score_trajectory(&obs_model, &traj_a, &p);
        let ll_b = score_trajectory(&obs_model, &traj_b, &p);
        assert_eq!(ll_a.to_bits(), ll_b.to_bits(),
            "ODE forward sim must be byte-identical regardless of seed; \
             loglik differs ({} vs {})", ll_a, ll_b);
    }

    #[test]
    fn maxeval_is_soft_failure() {
        // Cap evaluations far below what the real problem needs,
        // verify status reports MaxEvalReached and converged=false.
        let model = sir_model(0.6, 0.2);
        let compiled = Arc::new(CompiledModel::new(model).unwrap());
        let data = deterministic_data(&compiled);
        let obs_model = Arc::new(build_obs_model(Arc::clone(&compiled), data));

        let beta_idx = compiled.param_index["beta"];
        let gamma_idx = compiled.param_index["gamma"];
        let mut start_params = compiled.default_params.clone();
        start_params[beta_idx] = 1.5;
        start_params[gamma_idx] = 0.5;

        let estimated = vec![
            EstimatedParam {
                name: "beta".into(), index: beta_idx, initial: 1.5,
                rw_sd: 0.0, transform: Transform::None,
                lower: 0.05, upper: 3.0, ivp: false, rw_sd_auto: false,
            },
            EstimatedParam {
                name: "gamma".into(), index: gamma_idx, initial: 0.5,
                rw_sd: 0.0, transform: Transform::None,
                lower: 0.05, upper: 1.0, ivp: false, rw_sd_auto: false,
            },
        ];
        let config = DetOptConfig {
            algorithm: DetAlgorithm::Sbplx,
            xtol_rel: 1e-12,  // unreachable
            ftol_rel: None,
            max_evals: 5,    // hits cap immediately
            ode: OdeConfig { t_start: 0.0, t_end: 20.0, dt: 0.1 },
        };

        let result = optimize_det(
            Arc::clone(&compiled), Arc::clone(&obs_model),
            &start_params, &estimated, &config,
        ).unwrap();
        assert!(matches!(result.status, Some(SuccessState::MaxEvalReached)),
            "expected MaxEvalReached, got {:?}", result.status);
        assert!(!result.converged,
            "MaxEvalReached must surface as converged=false");
        assert!(result.n_evals <= 6,
            "should have stopped near the eval cap, got {}", result.n_evals);
    }
}
