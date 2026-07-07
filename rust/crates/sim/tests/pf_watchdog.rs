//! gh#110 — particle filter degeneracy watchdog integration test.
//!
//! Construct a contrived SIR chain-binomial model where the
//! dynamics blow up (R0 ~ 50) and feed it observations that the
//! likelihood cannot reconcile. ESS collapses within a handful of
//! observation windows. The watchdog must return
//! `Err(SimError::PFDegenerate { kind: EssCollapsed, .. })` and not
//! hang past a generous wall-clock budget.
//!
//! The acceptance criterion on gh#110 is explicit: this kind of
//! pathology must surface within ~5 seconds, not 30+ minutes.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ir::{
    expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr, ConstExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry, DrawMethod},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    error::{PFDegenerateKind, SimError},
    inference::{
        if2::{run_if2, IF2Config, EstimatedParam, Transform},
        obs_loglik::poisson_logpmf,
        particle_filter::bootstrap_filter,
        ChainBinomialProcess,
        traits::{ObservationModel, ProcessModel, SMCConfig},
        ParticleState,
    },
    rng::StatefulRng,
};

/// Observe compartment index 2 (I) with Poisson likelihood. Used to
/// build a deliberately mis-specified problem where the model predicts
/// a huge epidemic and the data is flat zero.
struct PoissonOnIObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for PoissonOnIObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        // Observe I (index 2). Clamp tiny mean to avoid -inf so the
        // failure mode is ESS collapse, not All-Particles-Dead via
        // a -inf swarm. The watchdog must catch ESS collapse first.
        let predicted = (state.counts[2] as f64).max(0.1);
        poisson_logpmf(self.observations[obs_idx], predicted)
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { 1 }
    fn sample(&self, _: &ParticleState, _: usize, _: &[f64], _: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _: &ParticleState, _: usize, _: &[f64]) -> Vec<f64> { vec![] }
}

/// Pathological SIR: R0 = beta/gamma ≈ 50 with N=1000 and only S0
/// initial pop. Every particle's I count explodes within a few days;
/// observations of "I = 0" are then astronomically unlikely under
/// the simulator. The PF re-weights kill all but one particle per
/// window, ESS goes to ~1, and stays there.
fn pathological_sir_model() -> (CompiledModel, Vec<f64>) {
    let beta = 5.0;   // contacts/day
    let gamma = 0.1;  // 1/recovery_days → R0 = 50
    let n_pop = 1000.0;

    let mut ic = HashMap::new();
    ic.insert("S".into(), n_pop - 1.0);
    ic.insert("I".into(), 1.0);
    ic.insert("R".into(), 0.0);

    // beta * S * I / N
    let infection_rate = Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
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
        right: Box::new(Expr::Const(ConstExpr { value: n_pop })),
    }});
    // gamma * I
    let recovery_rate = Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
        op: BinOp::Mul,
        left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
    }});

    let model = Model {
        ic_grad: Default::default(),
        name: "pathological_sir_pf".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: infection_rate,
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
                rate: recovery_rate,
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
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: beta }, param_kind: None, param_dim: None },
            Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Fixed { value: gamma }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit(ic),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 50.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 50.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// gh#110 acceptance: a contrived chain-binomial test model that
/// triggers ESS collapse fast must return `Err(SimError::PFDegenerate
/// { kind: EssCollapsed })` from `bootstrap_filter` within 5 seconds
/// of test wall-clock, NOT a hang.
#[test]
fn bootstrap_filter_bails_on_ess_collapse() {
    let (compiled, params) = pathological_sir_model();
    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());

    // 50 daily observations of "I = 0" against a model whose I count
    // hits hundreds by day 5. Every particle is astronomically
    // unlikely after the first few obs; ESS collapses immediately.
    let obs_times: Vec<f64> = (1..=50).map(|k| k as f64).collect();
    let observations: Vec<f64> = vec![0.0; 50];
    let obs_model = PoissonOnIObs { observations, obs_times };

    let config = SMCConfig {
        n_particles: 200, dt: 1.0, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let t0 = Instant::now();
    let res = bootstrap_filter(&process, &obs_model, &params, &config, 42);
    let elapsed = t0.elapsed();

    // Acceptance criterion: <5s, not a hang.
    assert!(elapsed.as_secs() < 5,
        "watchdog must bail within 5s; took {:?}", elapsed);

    match res {
        Err(SimError::PFDegenerate { kind, obs_window, elapsed_s: _ }) => {
            // The specific kind we expect on this pathology is
            // EssCollapsed. AllParticlesDead is also acceptable (limit case of
            // the same collapse). (gh#241 removed the wall-clock fallback; the
            // ESS detector is now the only statistical bail.)
            match kind {
                PFDegenerateKind::EssCollapsed { last_ess } => {
                    assert!(last_ess.iter().all(|&e| e <= sim::inference::degeneracy::ESS_FLOOR),
                        "ESS history at bail must all be at or below the floor: {:?}", last_ess);
                }
                PFDegenerateKind::AllParticlesDead => {
                    // Limit case of ESS collapse — acceptable.
                }
                PFDegenerateKind::IterationBudgetExceeded { .. } => {
                    panic!("expected EssCollapsed (or AllParticlesDead); the \
                            iteration budget fired instead, which is impossible \
                            on this dt=1.0 fixture and means the budget is \
                            mis-sized. obs_window={}", obs_window);
                }
            }
            assert!(obs_window < obs_model.n_observations(),
                "obs_window must be a valid index into the obs series");
        }
        Err(other) => panic!("expected SimError::PFDegenerate, got {:?}", other),
        Ok(r) => panic!(
            "expected PFDegenerate error; PF returned loglik={} with ESS trace {:?}",
            r.log_likelihood, r.ess_trace,
        ),
    }
}

/// gh#110 acceptance: an IF2 chain with a bad init (R₀ ≈ 50 here)
/// must return `Err(SimError::PFDegenerate)` from
/// `run_if2_with_progress` within 5 seconds, not a hang. The shared
/// `check_pf_degeneracy` helper is wired into IF2's inner per-iter
/// PF loop independently of `bootstrap_filter` (IF2 doesn't call it).
#[test]
fn if2_bails_on_ess_collapse() {
    let (compiled, params) = pathological_sir_model();
    let compiled = Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());

    // Same pathology: huge R₀ vs flat-zero observations.
    let obs_times: Vec<f64> = (1..=50).map(|k| k as f64).collect();
    let observations: Vec<f64> = vec![0.0; 50];
    let obs_model = PoissonOnIObs { observations, obs_times };

    // Estimate beta. IF2 will perturb it but the init is already
    // pathological — the very first iteration's PF eval should
    // collapse before cooling has a chance to bring things down.
    let beta_idx = compiled.param_index["beta"];
    let if2_params = vec![EstimatedParam {
        index: beta_idx,
        name: "beta".into(),
        initial: 5.0,
        rw_sd: 0.1,
        rw_sd_auto: false,
        transform: Transform::Log { lo: 0.01, hi: 100.0 },
        lower: 0.01,
        upper: 100.0,
        ivp: false,
    }];

    let config = IF2Config {
        n_particles: 200,
        n_iterations: 3,
        cooling_fraction: 0.5,
        cooling_target_iters: 10,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let t0 = Instant::now();
    let res = run_if2(&process, &obs_model, &params, &if2_params, &config, 42);
    let elapsed = t0.elapsed();

    assert!(elapsed.as_secs() < 5,
        "IF2 watchdog must bail within 5s; took {:?}", elapsed);

    match res {
        Err(SimError::PFDegenerate { kind, .. }) => {
            // EssCollapsed or AllParticlesDead is acceptable — the
            // deterministic ESS detector must fire in IF2's inner loop.
            assert!(
                matches!(kind, PFDegenerateKind::EssCollapsed { .. }
                              | PFDegenerateKind::AllParticlesDead),
                "expected EssCollapsed/AllParticlesDead; got {:?} (means IF2's \
                 per-iter ESS detector didn't fire)",
                kind);
        }
        Err(other) => panic!("expected SimError::PFDegenerate, got {:?}", other),
        Ok(_) => panic!("IF2 returned Ok on a pathological model; watchdog didn't fire"),
    }
}

/// Healthy SIR (R₀ = 2: beta = 0.5, gamma = 0.25, N = 1000). Runs to
/// completion without tripping any watchdog — used by the
/// parallel-determinism pin below.
fn healthy_sir_model() -> (CompiledModel, Vec<f64>) {
    let beta = 0.5;
    let gamma = 0.25;
    let n_pop = 1000.0;

    let mut ic = HashMap::new();
    ic.insert("S".into(), n_pop - 1.0);
    ic.insert("I".into(), 1.0);
    ic.insert("R".into(), 0.0);

    let infection_rate = Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
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
        right: Box::new(Expr::Const(ConstExpr { value: n_pop })),
    }});
    let recovery_rate = Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
        op: BinOp::Mul,
        left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
    }});

    let model = Model {
        ic_grad: Default::default(),
        name: "healthy_sir_pf".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                rate_state_grad: Default::default(),
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: infection_rate,
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
                rate: recovery_rate,
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
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: beta }, param_kind: None, param_dim: None },
            Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Fixed { value: gamma }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit(ic),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 30.0]),
            format: "tsv".into(), trajectory: true, observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 30.0, time_semantics: "continuous".into(),
            dt: Some(1.0), rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// gh#147 (M3.1). Parallel-determinism pin. The new pre-window
/// compute-budget guard computes its cost as a closed-form scalar
/// (`n_particles · ceil((obs_time − t)/dt)`), NOT a per-particle
/// reduction, so it cannot introduce thread-order dependence. This test
/// runs the *same* IF2 fit inside a 1-thread and an 8-thread rayon pool
/// and asserts the parameter trajectory is bit-identical — the property
/// content addressing requires (a fit's θ̂ must be a pure function of its
/// inputs, independent of `--parallel`). The fit completes in
/// milliseconds, far under the wall-clock floor, so that
/// machine-speed-dependent watchdog never fires either.
#[test]
fn if2_theta_hat_is_identical_across_thread_counts() {
    let (compiled, params) = healthy_sir_model();
    let compiled = Arc::new(compiled);

    // Early-epidemic growth curve for I; consistent enough with R₀ = 2
    // that ESS stays healthy across the (short) series.
    let obs_times: Vec<f64> = (1..=12).map(|k| k as f64).collect();
    let observations: Vec<f64> = vec![2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 19.0, 28.0, 40.0, 57.0, 80.0, 110.0];

    let beta_idx = compiled.param_index["beta"];
    let if2_params = vec![EstimatedParam {
        index: beta_idx,
        name: "beta".into(),
        initial: 0.5,
        rw_sd: 0.05,
        rw_sd_auto: false,
        transform: Transform::Log { lo: 0.01, hi: 5.0 },
        lower: 0.01,
        upper: 5.0,
        ivp: false,
    }];
    let config = IF2Config {
        n_particles: 100,
        n_iterations: 4,
        cooling_fraction: 0.5,
        cooling_target_iters: 10,
        dt: 1.0,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    // Run the identical fit under two different rayon pool sizes. `install`
    // scopes the pool for the closure, and the filter's `par_iter` picks it
    // up — so this genuinely exercises 1-way vs 8-way parallelism.
    let run = |threads: usize| -> (Vec<f64>, Vec<(Vec<f64>, f64)>) {
        let process = ChainBinomialProcess::new(compiled.clone());
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let res = pool.install(|| {
            run_if2(&process, &obs_model_for(&observations, &obs_times), &params, &if2_params, &config, 7)
        }).expect("healthy fit must complete (no watchdog)");
        let trace = res.iterations.iter()
            .map(|it| (it.param_means.clone(), it.if2_perturbed_loglik))
            .collect();
        (res.mle, trace)
    };

    let (mle_1, trace_1) = run(1);
    let (mle_8, trace_8) = run(8);

    assert_eq!(mle_1, mle_8,
        "θ̂ (mle) must be bit-identical across --parallel 1 vs 8; got {:?} vs {:?}",
        mle_1, mle_8);
    assert_eq!(trace_1.len(), trace_8.len());
    for (i, (a, b)) in trace_1.iter().zip(trace_8.iter()).enumerate() {
        assert_eq!(a, b,
            "iteration {} (param_means, perturbed_loglik) must be bit-identical \
             across thread counts; got {:?} vs {:?}", i, a, b);
    }
    // Non-vacuous guard: the perturbed loglik must actually be finite (the
    // fit ran a real filter, not an all-collapsed swarm that trivially
    // agrees at -inf on both runs).
    assert!(trace_1.iter().all(|(_, ll)| ll.is_finite()),
        "perturbed logliks must be finite (real filter ran): {:?}",
        trace_1.iter().map(|(_, ll)| *ll).collect::<Vec<_>>());
}

/// Helper so the determinism test can build an obs model inline twice
/// (the model is consumed by value per run).
fn obs_model_for(observations: &[f64], obs_times: &[f64]) -> PoissonOnIObs {
    PoissonOnIObs {
        observations: observations.to_vec(),
        obs_times: obs_times.to_vec(),
    }
}

/// gh#147 (M3.1). A process spy that counts `step()` calls so a test can
/// observe *whether the substep loop ran at all*. The `cap` is a
/// stand-in for "this loop would otherwise run unboundedly": once the
/// cumulative step count exceeds it, `step()` returns a non-recoverable
/// error so a filter *without* the pre-window guard terminates fast
/// (returning the cap error) instead of hanging on a sub-nanosecond dt.
/// A filter *with* the pre-window guard never calls `step()` at all on a
/// budget-busting window, so `steps()` stays 0 — that is the property
/// these tests pin (pre-window placement, not post-window).
struct CountingProcess {
    n_int: usize,
    n_tr: usize,
    steps: AtomicUsize,
    cap: usize,
}

impl CountingProcess {
    fn new(n_int: usize, n_tr: usize, cap: usize) -> Self {
        CountingProcess { n_int, n_tr, steps: AtomicUsize::new(0), cap }
    }
    fn steps(&self) -> usize {
        self.steps.load(Ordering::Relaxed)
    }
}

impl ProcessModel for CountingProcess {
    type State = ParticleState;
    type Scratch = ();

    fn n_compartments(&self) -> usize { self.n_int }
    fn n_transitions(&self) -> usize { self.n_tr }

    fn initial_state(&self, _params: &[f64]) -> Result<ParticleState, SimError> {
        // Mock process: `acc` sized 0 (the filter resizes from the obs model).
        Ok(ParticleState::new(self.n_int, self.n_tr, 0))
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
        _due_interventions: &[usize],
    ) -> Result<(), SimError> {
        let n = self.steps.fetch_add(1, Ordering::Relaxed);
        if n >= self.cap {
            // Stand-in for "would run forever": terminate the no-guard run
            // deterministically. `Validation` is NOT per-particle-recoverable,
            // so `bootstrap_filter`/`run_if2` propagate it (rather than
            // marking the particle dead and looping again).
            return Err(SimError::Validation(
                "CountingProcess step cap hit — the substep loop ran (would otherwise be unbounded)".into(),
            ));
        }
        Ok(())
    }

    fn new_scratch(&self) {}
}

/// Observation model that supplies only a time grid (constant likelihood,
/// no streams). Enough to drive the per-window loop; the placement tests
/// bail before any likelihood is evaluated.
struct TimeGridObs {
    obs_times: Vec<f64>,
}

impl ObservationModel<ParticleState> for TimeGridObs {
    fn log_likelihood(&self, _: &ParticleState, _: usize, _: &[f64]) -> f64 { 0.0 }
    fn n_observations(&self) -> usize { self.obs_times.len() }
    fn obs_time(&self, i: usize) -> f64 { self.obs_times[i] }
    fn n_streams(&self) -> usize { 0 }
    fn sample(&self, _: &ParticleState, _: usize, _: &[f64], _: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _: &ParticleState, _: usize, _: &[f64]) -> Vec<f64> { vec![] }
}

/// gh#147 (M3.1). The deterministic substep-budget guard must fire
/// BEFORE the per-window propagation loop, not after it. With a
/// sub-nanosecond dt the first window's cost
/// (`n_particles · ceil((obs_time − t)/dt)`) exceeds `ITER_BUDGET`, so a
/// correctly-placed guard bails with `PFIterationBudget` having run
/// ZERO substeps. A guard placed after the loop (the old watchdog site)
/// would first run the loop — billions of steps — and hang; the spy's
/// step cap stands in for that, turning the would-be hang into the
/// wrong error (`Validation`) with a non-zero step count.
#[test]
fn bootstrap_filter_iteration_budget_aborts_pre_window() {
    let n_particles = 20;
    let dt = 1e-9;
    let obs_times: Vec<f64> = vec![1.0, 2.0, 3.0];
    // Sanity: the first window genuinely busts the fixed budget.
    let cost = sim::inference::degeneracy::window_substep_cost(n_particles, 0.0, obs_times[0], dt);
    assert!(cost > sim::inference::degeneracy::ITER_BUDGET,
        "test fixture must bust the budget: cost {} vs ITER_BUDGET {}",
        cost, sim::inference::degeneracy::ITER_BUDGET);

    let process = CountingProcess::new(3, 1, 256);
    let obs_model = TimeGridObs { obs_times };
    let config = SMCConfig {
        n_particles, dt, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let t0 = Instant::now();
    let res = bootstrap_filter(&process, &obs_model, &[], &config, 42);
    let elapsed = t0.elapsed();

    assert!(elapsed.as_secs() < 5, "must bail fast, not hang; took {:?}", elapsed);
    match res {
        Err(SimError::PFIterationBudget { obs_window, attempted_substeps, budget_substeps }) => {
            assert_eq!(obs_window, 0, "must bail on the first (budget-busting) window");
            assert!(attempted_substeps > budget_substeps,
                "attempted {} must exceed budget {}", attempted_substeps, budget_substeps);
            assert_eq!(budget_substeps, sim::inference::degeneracy::ITER_BUDGET);
        }
        Err(other) => panic!("expected PFIterationBudget, got {:?}", other),
        Ok(_) => panic!("expected PFIterationBudget; PF returned Ok (the substep loop must have run)"),
    }
    // The load-bearing assertion: the guard fired PRE-window, so the
    // substep loop never executed. A post-window check would have run it.
    assert_eq!(process.steps(), 0,
        "iteration-budget guard must abort BEFORE the substep loop runs any step");
}

/// gh#147 (M3.1). Same pre-window placement property for IF2's inner
/// per-iteration PF loop (the spec applies the guard to both filters).
#[test]
fn if2_iteration_budget_aborts_pre_window() {
    let n_particles = 20;
    let dt = 1e-9;
    let obs_times: Vec<f64> = vec![1.0, 2.0, 3.0];
    let cost = sim::inference::degeneracy::window_substep_cost(n_particles, 0.0, obs_times[0], dt);
    assert!(cost > sim::inference::degeneracy::ITER_BUDGET);

    let process = CountingProcess::new(3, 1, 256);
    let obs_model = TimeGridObs { obs_times };
    let params = vec![1.0];
    let if2_params = vec![EstimatedParam {
        index: 0,
        name: "p".into(),
        initial: 1.0,
        rw_sd: 0.1,
        rw_sd_auto: false,
        transform: Transform::Log { lo: 0.01, hi: 100.0 },
        lower: 0.01,
        upper: 100.0,
        ivp: false,
    }];
    let config = IF2Config {
        n_particles,
        n_iterations: 3,
        cooling_fraction: 0.5,
        cooling_target_iters: 10,
        dt,
        t_start: 0.0,
        simplex_groups: vec![],
        skip_first_obs_from_loglik: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let t0 = Instant::now();
    let res = run_if2(&process, &obs_model, &params, &if2_params, &config, 42);
    let elapsed = t0.elapsed();

    assert!(elapsed.as_secs() < 5, "IF2 must bail fast, not hang; took {:?}", elapsed);
    match res {
        Err(SimError::PFIterationBudget { obs_window, .. }) => {
            assert_eq!(obs_window, 0);
        }
        Err(other) => panic!("expected PFIterationBudget, got {:?}", other),
        Ok(_) => panic!("expected PFIterationBudget; IF2 returned Ok (the substep loop must have run)"),
    }
    assert_eq!(process.steps(), 0,
        "IF2 iteration-budget guard must abort BEFORE the substep loop runs any step");
}
