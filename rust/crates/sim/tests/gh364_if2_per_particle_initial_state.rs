//! gh#364 — every IF2 particle must draw its OWN initial state from its OWN
//! perturbed θ, otherwise a pure initial-condition parameter (declared
//! `perturb_only_at_t0`) is
//! unestimatable: the swarm has zero initial-state spread, the weights carry
//! no selection pressure on it, and the reported mean does an unselected
//! random walk.
//!
//! Normative source, Ionides, Nguyen, Atchadé, Stoev & King (2015),
//! "Inference for dynamic and latent variable models via iterated, perturbed
//! Bayes maps", PNAS 112(3):719–724, doi:10.1073/pnas.1410597112, Algorithm 1
//! (IF2), per-iteration preamble:
//!
//! ```text
//!   Θ^{F,m}_{0,j} ~ h_0(θ | Θ^{m-1}_j; σ_m)          for j in 1:J   [perturb]
//!   X^{F,m}_{0,j} ~ f_{X_0}(x_0; Θ^{F,m}_{0,j})      for j in 1:J   [per-particle x₀]
//! ```
//!
//! The `j` subscript on Θ inside `f_{X_0}` is the whole point: particle `j`'s
//! initial state comes from particle `j`'s perturbed parameters. pomp does
//! the same — `pomp:::mif2_pfilter` (`R/mif2.R`, pomp 6.4.0.2,
//! kingaa/pomp@0eaf3c01) calls `rinit(object, params=tparams)` where `tparams`
//! is the `npars × Np` matrix that `randwalk_perturbation` (`src/mif2.c`) has
//! just jittered column-by-column, one column per particle.
//!
//! Before the fix camdl evaluated the initial state once at `current_params`
//! ONCE from the iteration-mean θ and copied the result to every particle.
//!
//! Two tests:
//!   1. `if2_initial_state_uses_each_particles_own_theta` — exact, no
//!      statistics: a mock process stamps the θ it was initialised from into
//!      the state; the obs model checks it against the particle's own θ.
//!   2. `pure_ic_ivp_param_is_identified` — the harm, on a real
//!      chain-binomial SIR whose `i0` is a pure IC parameter (it appears in
//!      `initial_conditions` and in no rate and in no observation): IF2 must
//!      recover it, and the weights must exert selection pressure on it
//!      (weighted-variance ratio < 1).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ir::{
    expr::{BinOp, ConstExpr, Expr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    chain_binomial::{step_one, StepScratch},
    compiled_model::CompiledModel,
    error::SimError,
    inference::{
        if2::{run_if2, EstimatedParam, IF2Config, Transform},
        obs_loglik::negbin_logpmf,
        traits::{ObservationModel, ProcessModel},
        ChainBinomialProcess, ParticleState,
    },
    rng::StatefulRng,
};

// ── Test 1: exact per-particle x₀ provenance ────────────────────────────────

const IVP_IDX: usize = 0;
const N_OBS_MOCK: usize = 3;

fn stamp(x: f64) -> i64 {
    x.to_bits() as i64
}
fn unstamp(v: i64) -> f64 {
    f64::from_bits(v as u64)
}

/// Mock process whose initial state is a pure function of an IC-only
/// parameter and whose `step` is a no-op — the analogue of a model where
/// `i0` sets x₀ and never re-enters a rate. `counts[0]` carries the θ that
/// produced the initial state; `step` must not disturb it.
struct ICStampProcess;

impl ProcessModel for ICStampProcess {
    type State = ParticleState;
    type Scratch = ();

    fn n_compartments(&self) -> usize {
        1
    }
    fn n_transitions(&self) -> usize {
        1
    }
    fn initial_state_draw(
        &self, params: &[f64], _rng: &mut StatefulRng,
    ) -> Result<ParticleState, SimError> {
        let mut s = ParticleState::new(1, 1, 0);
        s.counts[0] = stamp(params[IVP_IDX]);
        Ok(s)
    }
    fn step(
        &self,
        _state: &mut ParticleState,
        _params: &[f64],
        _t: f64,
        _dt: f64,
        _per_eval: Option<&[f64]>,
        _rng: &mut StatefulRng,
        _scratch: &mut (),
        _due_effects: &[usize],
    ) -> Result<(), SimError> {
        Ok(())
    }
    fn new_scratch(&self) {}
}

/// Records (θ that produced x₀, θ this particle carries) at every scoring.
struct ICWitnessObs {
    pairs: Mutex<Vec<(f64, f64)>>,
}

impl ObservationModel<ParticleState> for ICWitnessObs {
    fn log_likelihood(&self, state: &ParticleState, _obs_idx: usize, params: &[f64]) -> f64 {
        self.pairs
            .lock()
            .unwrap()
            .push((unstamp(state.counts[0]), params[IVP_IDX]));
        0.0
    }
    fn n_observations(&self) -> usize {
        N_OBS_MOCK
    }
    fn obs_time(&self, obs_idx: usize) -> f64 {
        (obs_idx + 1) as f64
    }
}

#[test]
fn if2_initial_state_uses_each_particles_own_theta() {
    let process = ICStampProcess;
    let obs = ICWitnessObs {
        pairs: Mutex::new(Vec::new()),
    };

    let base_params = vec![0.05_f64];
    let if2_params = vec![EstimatedParam {
        name: "i0".into(),
        index: IVP_IDX,
        initial: 0.05,
        rw_sd: 0.3,
        transform: Transform::Logit { lo: 0.001, hi: 0.5 },
        lower: 0.001,
        upper: 0.5,
        // Pure initial-condition parameter: perturbed at t=0 only, exactly
        // like pomp's `ivp()` entry in `rw.sd`. camdl spells the flag
        // `perturb_only_at_t0` — it is a perturbation schedule, nothing more.
        perturb_only_at_t0: true,
        rw_sd_auto: false,
    }];

    let config = IF2Config {
        n_particles: 16,
        n_iterations: 3,
        cooling_fraction: 0.9,
        cooling_target_iters: 50,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    run_if2(&process, &obs, &base_params, &if2_params, &config, 7).expect("mock IF2 run");

    let pairs = obs.pairs.into_inner().unwrap();
    assert_eq!(
        pairs.len(),
        config.n_particles * N_OBS_MOCK * config.n_iterations
    );

    // Negative control: a `perturb_only_at_t0` parameter is perturbed at
    // t=0 only, so the
    // spread must come from that single perturbation. If it were zero, the
    // equality below would hold vacuously.
    let distinct = pairs
        .iter()
        .map(|&(_, theta)| theta.to_bits())
        .collect::<std::collections::HashSet<_>>();
    assert!(
        distinct.len() > 1,
        "vacuous test: the t=0 perturbation produced no per-particle spread"
    );

    let mismatches = pairs.iter().filter(|&&(from_x0, own)| from_x0 != own).count();
    assert_eq!(
        mismatches,
        0,
        "gh#364: {}/{} particles carry an initial state generated from a θ that is \
         not their own — IF2 evaluated the initial state once from the swarm mean. \
         Algorithm 1 requires X^F_{{0,j}} ~ f_{{X_0}}(·; Θ^F_{{0,j}}). \
         First offender: x₀ from θ={:.12}, particle θ={:.12}",
        mismatches,
        pairs.len(),
        pairs.iter().find(|&&(a, b)| a != b).map(|p| p.0).unwrap_or(f64::NAN),
        pairs.iter().find(|&&(a, b)| a != b).map(|p| p.1).unwrap_or(f64::NAN),
    );
}

// ── Test 2: a pure-IC `perturb_only_at_t0` param must be identified ────────

const N_POP: f64 = 10_000.0;
const TRUE_I0: f64 = 0.006; // I₀ = 60
// I₀ = 100 — 1.67× HIGH, and deliberately on the far side of the truth from
// the midpoint of the declared bounds (0.0253). A `perturb_only_at_t0`
// parameter perturbed
// symmetrically on the logit scale but averaged back on the natural scale
// picks up a Jensen drift toward that midpoint, which is data-free: it happens
// whether or not the weights carry any information. Starting BELOW the truth
// would let that drift walk the estimate through the answer and pass a broken
// filter. Starting above makes the drift push away from the truth, so only
// genuine selection can close the gap.
const START_I0: f64 = 0.010;
const N_DAYS: usize = 20; // exponential-growth window (R₀ = 2, no depletion)
const NEGBIN_K: f64 = 5.0; // observation dispersion (Var = μ + μ²/k)

/// SIR whose initial conditions are parameterized by `i0` (the initially
/// infectious fraction). `i0` appears ONLY in `initial_conditions` — no
/// transition rate mentions it, and the observation model below does not use
/// it either. It is therefore a *pure* IC parameter: the only channel through
/// which the data can inform it is the spread of x₀ across the swarm.
fn pure_ic_sir_model() -> (CompiledModel, Vec<f64>) {
    let n = Expr::const_(N_POP);
    // beta * S * I / N
    let infection = Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(
            BinOp::Mul,
            Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
            Expr::pop("I"),
        ),
        n.clone(),
    );
    // gamma * I
    let recovery = Expr::bin_op(BinOp::Mul, Expr::param("gamma"), Expr::pop("I"));

    let model = Model {
        ic_grad: Default::default(),
        name: "sir_pure_ic_ivp".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: infection,
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            },
            Transition {
                rate_state_grad: Default::default(),
                name: "recovery".into(),
                stoichiometry: vec![
                    StoichiometryEntry("I".into(), -1),
                    StoichiometryEntry("R".into(), 1),
                ],
                rate: recovery,
                metadata: None,
                draw_method: DrawMethod::Poisson,
                rate_grad: Default::default(),
                lineage: None,
            },
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter {
                name: "beta".into(),
                value: ir::parameter::ParamValue::Estimated {
                    init: Some(0.4),
                    bounds: Some((0.01, 2.0)),
                    prior: ir::parameter::PriorSpec::Flat,
                    transform: ir::parameter::Transform::Identity,
                },
                param_kind: None,
                param_dim: None,
            },
            Parameter {
                name: "gamma".into(),
                value: ir::parameter::ParamValue::Estimated {
                    init: Some(0.2),
                    bounds: Some((0.01, 1.0)),
                    prior: ir::parameter::PriorSpec::Flat,
                    transform: ir::parameter::Transform::Identity,
                },
                param_kind: None,
                param_dim: None,
            },
            Parameter {
                name: "i0".into(),
                value: ir::parameter::ParamValue::Estimated {
                    init: Some(TRUE_I0),
                    bounds: Some((0.0005, 0.05)),
                    prior: ir::parameter::PriorSpec::Flat,
                    transform: ir::parameter::Transform::Identity,
                },
                param_kind: None,
                param_dim: None,
            },
        ],
        initial_conditions: InitialConditions::Parameterized(HashMap::from([
            (
                "S".to_string(),
                Expr::bin_op(
                    BinOp::Sub,
                    Expr::const_(N_POP),
                    Expr::bin_op(BinOp::Mul, Expr::param("i0"), Expr::const_(N_POP)),
                ),
            ),
            (
                "I".to_string(),
                Expr::bin_op(BinOp::Mul, Expr::param("i0"), Expr::const_(N_POP)),
            ),
            ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
        ])),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 40.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 40.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Daily PREVALENCE (compartment `I`) observed with NegBin(k = NEGBIN_K).
///
/// Prevalence rather than incidence, and a wide dispersion, on purpose: in the
/// exponential-growth window `I(t) ≈ I₀ e^{(β−γ)t}`, so the whole series scales
/// with `i0` — informative — while the between-particle spread stays small
/// relative to the observation noise, which keeps ESS healthy on BOTH arms.
/// (Tight daily-incidence observations put a mis-started swarm deep in the
/// likelihood tail, where the filter degenerates before it can be shown to
/// have learned nothing — a red for the wrong reason.)
///
/// The likelihood does NOT read `params` — `i0` reaches the data only through
/// x₀, which is exactly the channel gh#364 severed.
struct NegBinPrevalence {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for NegBinPrevalence {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        let projected = (state.counts[1] as f64).max(0.1);
        negbin_logpmf(self.observations[obs_idx], projected, NEGBIN_K)
    }
    fn n_observations(&self) -> usize {
        self.observations.len()
    }
    fn obs_time(&self, obs_idx: usize) -> f64 {
        self.obs_times[obs_idx]
    }
}

/// One synthetic epidemic: daily prevalence over `N_DAYS`.
fn generate_data(compiled: &CompiledModel, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StatefulRng::new(11);
    let n_int = compiled.int_local_to_global.len();
    let n_tr = compiled.model.transitions.len();
    let mut state = ParticleState::new(n_int, n_tr, 0);
    let (init, _) = compiled.initial_state_mean(params).unwrap();
    state.counts.copy_from_slice(&init.counts);

    let mut scratch = StepScratch::new(compiled);
    let mut real = sim::state::RealState::new(compiled.real_local_to_global.len());
    let (mut times, mut values) = (Vec::new(), Vec::new());
    let mut t = 0.0;
    while t < N_DAYS as f64 {
        let fire_steps = compiled.resolve_fire_steps(1.0, params);
        sim::effects::due_effects(compiled, &fire_steps, t + 1.0, 1.0, &mut scratch.effect_batch);
        step_one(
            compiled,
            &mut state.counts,
            &mut state.flow_accumulators,
            &mut real,
            params,
            t,
            1.0,
            None,
            &mut rng,
            &mut scratch,
        )
        .unwrap();
        t += 1.0;
        times.push(t);
        values.push(state.counts[1] as f64);
        state.reset_flows();
    }
    (times, values)
}

#[test]
fn pure_ic_ivp_param_is_identified() {
    let (compiled, true_params) = pure_ic_sir_model();
    assert_eq!(
        compiled.initial_state_mean(&true_params).unwrap().0.counts[1],
        (TRUE_I0 * N_POP) as i64,
        "fixture sanity: i0 must actually drive I₀"
    );
    let (obs_times, obs_values) = generate_data(&compiled, &true_params);

    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());
    let obs_model = NegBinPrevalence { observations: obs_values, obs_times };

    let mut start_params = true_params.clone();
    start_params[2] = START_I0;

    // Estimate ONLY the pure-IC parameter; beta and gamma are held at truth so
    // the test isolates the IC channel.
    let if2_params = vec![EstimatedParam {
        name: "i0".into(),
        index: 2,
        initial: START_I0,
        // Natural scale (`EstimatedParam::transformed_sd` divides by dθ/dz):
        // jitter i0 by ±0.003, i.e. ±30 initial infectious out of N = 10 000.
        rw_sd: 0.003,
        transform: Transform::Logit { lo: 0.0005, hi: 0.05 },
        lower: 0.0005,
        upper: 0.05,
        perturb_only_at_t0: true,
        rw_sd_auto: false,
    }];

    let config = IF2Config {
        n_particles: 600,
        n_iterations: 30,
        cooling_fraction: 0.5,
        cooling_target_iters: 30,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let result = run_if2(&process, &obs_model, &start_params, &if2_params, &config, 3)
        .expect("IF2 run");

    let i0_hat = result.mle[2];
    let trace: Vec<f64> = result.iterations.iter().map(|it| it.param_means[2]).collect();
    // Weighted-variance ratio Var_w(θ)/Var_u(θ), averaged over observations and
    // over the run. Weights that carry no information about θ leave the cloud
    // untouched and this sits at ≈ 1; genuine selection pushes it below 1.
    // Reported, not asserted: at low ESS it can dip below 1 for reasons that
    // have nothing to do with selection on θ, so it is a diagnostic here and
    // the recovery assertion below is the test.
    let wvr: f64 = result
        .iterations
        .iter()
        .map(|it| it.param_diag[0].weighted_var_ratio)
        .sum::<f64>()
        / result.iterations.len() as f64;

    eprintln!(
        "gh#364: i0 start={:.5} true={:.5} estimate={:.5}; mean weighted_var_ratio={:.4}",
        START_I0, TRUE_I0, i0_hat, wvr
    );
    eprintln!(
        "gh#364: per-iteration i0 filter mean: {}",
        trace.iter().map(|v| format!("{:.5}", v)).collect::<Vec<_>>().join(" ")
    );

    // The failing mode this pins: with one shared x₀ the weights are
    // independent of each particle's i0 (`weighted_var_ratio` ≈ 1 above), so
    // resampling is a blind subsample and nothing pulls the filter mean toward
    // the truth. What motion remains is the data-free logit/natural-scale
    // Jensen drift, which pushes i0 UP toward the midpoint of its bounds —
    // away from TRUE_I0 < START_I0. Closing this gap requires selection.
    assert!(
        (i0_hat - TRUE_I0).abs() < 0.3 * TRUE_I0,
        "gh#364: IF2 did not recover the pure-IC perturb_only_at_t0 param: \
         started at {:.5}, \
         truth {:.5}, estimate {:.5}. Every particle shared one initial state \
         evaluated from the swarm mean, so the weights could not select on i0 \
         and its filter mean did an unselected random walk. \
         (mean weighted-variance ratio {:.4}; trace above)",
        START_I0,
        TRUE_I0,
        i0_hat,
        wvr
    );
}
