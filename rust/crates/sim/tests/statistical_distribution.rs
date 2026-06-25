//! Statistical distribution tests (§A.2). These are marked #[ignore] for nightly CI only.

use std::collections::HashMap;
use ir::{
    expr::{ConstExpr, Expr, ParamExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{Transition, StoichiometryEntry},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim,
};

/// Test adapter for the retired `apply_interventions_at`: derive the due batch
/// at `t` (grid_dt) and apply the scheduled interventions. Byte-identical.
fn apply_interventions_at(
    t: f64,
    model: &CompiledModel,
    fire_steps: &[std::collections::BTreeSet<i64>],
    dt: f64,
    grid_dt: f64,
    int_s: &mut sim::state::IntState,
    real_s: &mut sim::state::RealState,
    params: &[f64],
) -> Result<bool, sim::error::SimError> {
    let mut batch = sim::schedule::EffectBatch::default();
    sim::effects::due_effects(model, fire_steps, t, grid_dt, &mut batch);
    sim::intervention::apply_effect_batch(t, model, &batch.intervention_idx, dt, int_s, real_s, params)
}

fn load_golden(name: &str) -> ir::Model {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = std::path::PathBuf::from(&manifest)
        .join("../../../ir/golden")
        .join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("could not read {:?}", path));
    ir::from_str(&contents).unwrap()
}

/// Pure death process: I(t=10) should follow Binomial(100, exp(-0.1*10)) = Binomial(100, exp(-1)).
/// Test: mean and variance of I(10) over 2000 seeds.
#[test]
#[ignore = "statistical test: run with --ignored in nightly CI"]
fn test_pure_death_distribution() {
    let model = load_golden("pure_death");
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: 0.0,
        t_end: 10.0,
        output_dt: None,
    });

    let mut samples: Vec<f64> = Vec::new();
    for seed in 0..2000u64 {
        let traj = GillespieSim.run(&compiled, &params, seed, &config).unwrap();
        if let Some(last) = traj.snapshots.last() {
            samples.push(last.int_state.counts[0] as f64);
        }
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

    // N=1000, mu=0.1, t=10: E[N(10)] = 1000 * exp(-1) ≈ 367.88
    // Var = N * p * (1-p) where p = exp(-mu*t) = exp(-1)
    let n0 = 1000.0;
    let p = (-1.0_f64).exp();
    let expected_mean = n0 * p;
    let expected_var = n0 * p * (1.0 - p);

    assert!(
        (mean - expected_mean).abs() < 5.0,
        "pure death mean wrong: got {:.2}, expected {:.2}", mean, expected_mean
    );
    assert!(
        (var - expected_var).abs() < 15.0,
        "pure death variance wrong: got {:.2}, expected {:.2}", var, expected_var
    );
}

/// Two-state equilibrium: E[A] = N * k2/(k1+k2) = 50 * 0.7 = 35.
#[test]
#[ignore = "statistical test: run with --ignored in nightly CI"]
fn test_two_state_equilibrium() {
    let model = load_golden("two_state");
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: 0.0,
        t_end: 100.0, // run to equilibrium
        output_dt: None,
    });

    let mut a_samples: Vec<f64> = Vec::new();
    for seed in 0..5000u64 {
        let traj = GillespieSim.run(&compiled, &params, seed, &config).unwrap();
        if let Some(last) = traj.snapshots.last() {
            // A is first compartment
            a_samples.push(last.int_state.counts[0] as f64);
        }
    }

    let n = a_samples.len() as f64;
    let mean = a_samples.iter().sum::<f64>() / n;
    // A↔B: alpha=0.5 (A→B), beta_r=0.3 (B→A), N=100
    // E[A] = N * beta_r / (alpha + beta_r) = 100 * 0.3 / 0.8 = 37.5
    let expected_mean = 100.0 * 0.3 / (0.5 + 0.3);

    assert!(
        (mean - expected_mean).abs() < 1.5,
        "two-state equilibrium mean wrong: got {:.2}, expected {:.2}", mean, expected_mean
    );
}

/// Gillespie on immigration-death process: constant birth rate λ, per-capita death rate μ.
/// Stationary distribution is Poisson(λ/μ).
/// Here we use the pure_death golden model with birth added. But building state-dependent
/// rates from IR is complex, so we use the two_state model as a proxy: its equilibrium
/// distribution is known (Binomial), and we already test its mean above.
/// Instead, we validate the PURE DEATH (no birth) distribution more precisely,
/// checking that the variance matches the binomial prediction.
#[test]
#[ignore = "statistical test: run with --ignored in nightly CI"]
fn test_pure_death_variance() {
    let model = load_golden("pure_death");
    let compiled = CompiledModel::new(model.clone()).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::Gillespie(GillespieConfig {
        t_start: 0.0,
        t_end: 10.0,
        output_dt: None,
    });

    let mut samples: Vec<f64> = Vec::new();
    for seed in 0..5000u64 {
        let traj = GillespieSim.run(&compiled, &params, seed, &config).unwrap();
        if let Some(last) = traj.snapshots.last() {
            samples.push(last.int_state.counts[0] as f64);
        }
    }

    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

    // Binomial(1000, exp(-1)): mean ≈ 367.9, var = 1000 * p * (1-p) ≈ 232.5
    let p = (-1.0_f64).exp();
    let expected_mean = 1000.0 * p;
    let expected_var = 1000.0 * p * (1.0 - p);

    // With 5000 samples, SE of mean ≈ sqrt(232.5/5000) ≈ 0.22, so 3σ ≈ 0.65
    assert!(
        (mean - expected_mean).abs() < 2.0,
        "pure death mean: got {:.3}, expected {:.3}", mean, expected_mean
    );
    // SE of variance ≈ var * sqrt(2/(n-1)) ≈ 232.5 * 0.02 ≈ 4.65
    assert!(
        (var - expected_var).abs() < 15.0,
        "pure death variance: got {:.3}, expected {:.3}", var, expected_var
    );
}

// ── Overdispersion variance validation ──────────────────────────────────

/// Validate that overdispersed() produces the correct variance.
/// Single transition with known rate and σ², run many chain-binomial steps,
/// check empirical Var[count] ≈ mean + mean² · σ² / dt.
#[test]
#[ignore = "statistical test: run with --ignored in nightly CI"]
fn test_overdispersion_variance_chain_binomial() {
    // S → I at rate beta*S with overdispersion sigma_sq.
    // With S=10000 and beta=0.01, propensity = 100, mean per step (dt=1) = 100.
    // Var should be: 100 + 100² × 0.5 / 1.0 = 100 + 5000 = 5100
    // Without overdispersion: Var = 100 (Poisson).
    use ir::expr::{BinOpExpr, BinOpWrap, BinOp, PopExpr};

    let model = Model {
        name: "od_test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            Transition {
                name: "infection".into(),
                stoichiometry: vec![
                    StoichiometryEntry("S".into(), -1),
                    StoichiometryEntry("I".into(), 1),
                ],
                rate: Expr::BinOp(BinOpWrap {
                    bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                    },
                }),
                metadata: None,
                draw_method: ir::transition::DrawMethod::Overdispersed(
                    Expr::Param(ParamExpr { param: "sigma_sq".into() })),
                rate_grad: Default::default(), lineage: None,
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
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: 0.01 }, param_kind: None, param_dim: None, doc: None },
            Parameter { name: "sigma_sq".into(), value: ir::parameter::ParamValue::Fixed { value: 0.5 }, param_kind: None, param_dim: None, doc: None },
        ],
        // Start with S=10000, I=0. After one dt=1 step, about 100 events.
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new(); m.insert("S".into(), 10000.0); m.insert("I".into(), 0.0); m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start: 0.0,
        t_end: 1.0,
        dt: 1.0,
    });

    // Collect infection counts from one chain-binomial step across many seeds
    let mut counts: Vec<f64> = Vec::new();
    for seed in 0..10000u64 {
        let traj = ChainBinomialSim.run(&compiled, &params, seed, &config).unwrap();
        let last = traj.snapshots.last().unwrap();
        // Infections = flow_infection (index 0)
        counts.push(last.flows.as_int()[0] as f64);
    }

    let n = counts.len() as f64;
    let mean = counts.iter().sum::<f64>() / n;
    let var = counts.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);

    // Expected: mean ≈ 100, var ≈ 100 + 100²×0.5 = 5100
    let expected_mean = 100.0;
    let expected_var = expected_mean + expected_mean * expected_mean * 0.5;

    assert!(
        (mean - expected_mean).abs() < 5.0,
        "overdispersion mean: got {:.1}, expected {:.1}", mean, expected_mean
    );
    // Variance has high SE: SE(var) ≈ var * sqrt(2/n) ≈ 5100 * 0.014 ≈ 72
    assert!(
        (var - expected_var).abs() < 500.0,
        "overdispersion variance: got {:.0}, expected {:.0} (mean+mean²×σ²)", var, expected_var
    );
    // Also verify it's much larger than Poisson variance
    assert!(
        var > 500.0,
        "overdispersion variance {} should be >> Poisson variance {}", var, expected_mean
    );
}

// ── Intervention edge cases ─────────────────────────────────────────────

#[test]
fn test_fraction_transfer_edge_cases() {
    use ir::intervention::{Action, FractionTransfer, Intervention, InterventionSchedule};

    let make_model = |frac: f64| -> (CompiledModel, Vec<f64>) {
        let iv = Intervention {
            name: "vacc".into(),
            base_name: None,
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![1.0])),
            kind: ir::intervention::InterventionKind::Scenario, actions: vec![Action::FractionTransfer(FractionTransfer {
                src: "S".into(), dst: "V".into(),
                fraction: Expr::Const(ConstExpr { value: frac }),
            })],
        };
        let model = Model {
            name: "test".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None, origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "V".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![iv],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![],
            initial_conditions: InitialConditions::Parameterized(HashMap::new()),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 2.0,
                time_semantics: "continuous".into(), dt: None, rng_seed: Some(42),
                integrator: Default::default(),
            },
            presets: vec![],
            model_structure: None, balance: None, identity_tracked_compartments: vec![],
        };
        let compiled = CompiledModel::new(model).unwrap();
        let params = compiled.default_params.clone();
        (compiled, params)
    };

    // fraction=1.0, S=100 → transfer all
    {
        let (model, _) = make_model(1.0);
        let mut int_s = sim::state::IntState::from_vec(vec![100, 0]);
        let mut real_s = sim::state::RealState::new(0);
        apply_interventions_at(1.0, &model, &model.resolve_fire_steps(1.0, &[]), 1.0, 1.0, &mut int_s, &mut real_s, &[]).unwrap();
        assert_eq!(int_s.counts[0], 0, "frac=1.0: S should be 0");
        assert_eq!(int_s.counts[1], 100, "frac=1.0: V should be 100");
    }

    // fraction=0.0, S=100 → transfer nothing
    {
        let (model, _) = make_model(0.0);
        let mut int_s = sim::state::IntState::from_vec(vec![100, 0]);
        let mut real_s = sim::state::RealState::new(0);
        apply_interventions_at(1.0, &model, &model.resolve_fire_steps(1.0, &[]), 1.0, 1.0, &mut int_s, &mut real_s, &[]).unwrap();
        assert_eq!(int_s.counts[0], 100, "frac=0.0: S should stay 100");
        assert_eq!(int_s.counts[1], 0, "frac=0.0: V should stay 0");
    }

    // fraction=0.8, S=1 → floor(0.8) = 0
    {
        let (model, _) = make_model(0.8);
        let mut int_s = sim::state::IntState::from_vec(vec![1, 0]);
        let mut real_s = sim::state::RealState::new(0);
        apply_interventions_at(1.0, &model, &model.resolve_fire_steps(1.0, &[]), 1.0, 1.0, &mut int_s, &mut real_s, &[]).unwrap();
        assert_eq!(int_s.counts[0], 1, "frac=0.8, S=1: floor(0.8)=0, no transfer");
        assert_eq!(int_s.counts[1], 0);
    }

    // fraction=0.8, S=0 → no crash, no transfer
    {
        let (model, _) = make_model(0.8);
        let mut int_s = sim::state::IntState::from_vec(vec![0, 0]);
        let mut real_s = sim::state::RealState::new(0);
        apply_interventions_at(1.0, &model, &model.resolve_fire_steps(1.0, &[]), 1.0, 1.0, &mut int_s, &mut real_s, &[]).unwrap();
        assert_eq!(int_s.counts[0], 0, "frac=0.8, S=0: no crash");
        assert_eq!(int_s.counts[1], 0);
    }
}
