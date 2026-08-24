//! Byte-identical INFERENCE-loglik baseline gate — the refactor ratchet for the
//! likelihood path, sibling to `gate_trajectory_baseline.rs` (which ratchets
//! forward trajectories and never runs the filter).
//!
//! For each reference model — built in-memory with an observation block — run the
//! bootstrap particle filter at fixed (params, seed, particles, dt) against a
//! fixed observation vector and assert the marginal log-likelihood matches a
//! committed baseline. This is the gate the unified-timeline refactor (the merged
//! schedule, the consolidated filter loop, the substep lifecycle) must not move
//! silently: a 1-ULP shift in the scored loglik fails here, where the forward
//! trajectory gate would not notice (it scores nothing).
//!
//! The reference is the validated `sir` recovery case
//! (`tests/recovery/cases/sir`, see its README): the book getting-started SIR
//! scored at the planted truth against the committed seed-1 synthetic weekly-case
//! series. The model is rebuilt in-memory here (rather than loaded from a
//! generated IR fixture that would go stale on a schema bump) — the same style as
//! `likelihood_path_parity.rs` / `particle_filter.rs`.
//!
//! Baselines are machine/toolchain-specific (libm `exp`/`lgamma` differ by ULPs
//! across platforms) — a development ratchet: capture on the dev machine, run
//! before/after each refactor phase on the same machine. Re-capture with
//!   CAMDL_CAPTURE_BASELINE=1 cargo test -p sim --test gate_inference_baseline -- --nocapture
//! and paste the printed table into BASELINES.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr, PopSumExpr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{
        Likelihood, NegBinomialLikelihood, ObservationModel as IrObs, ObservationSchedule,
        Projection,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        BoundObs,
        dense_cells,
        multi_stream_obs::{MultiStreamObsModel, StreamProjection, StreamSpec},
        particle_filter::bootstrap_filter,
        traits::SMCConfig,
        ChainBinomialProcess,
    },
};

const SEED: u64 = 42;
const N_PARTICLES: usize = 8000;
const DT: f64 = 1.0;

// ── expr / model builders ──────────────────────────────────────────────────
fn p(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn c(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn mul(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(a), right: Box::new(b) } })
}
fn div(a: Expr, b: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op: BinOp::Div, left: Box::new(a), right: Box::new(b) } })
}
fn param(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ir::parameter::ParamValue::Fixed { value: value }, param_kind: None, param_dim: None }
}
fn transition(name: &str, sto: Vec<StoichiometryEntry>, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(), stoichiometry: sto, rate, metadata: None,
        draw_method: DrawMethod::Poisson, rate_grad: Default::default(), lineage: None,
    }
}

/// The `sir` recovery reference: closed SIR, weekly incidence(infection) ~
/// NegBin(mean = rho·proj, r = k), at truth (β=0.4, γ=0.15, ρ=0.6, k=10),
/// scored against the committed seed-1 synthetic weekly-case series, with the
/// observations placed at `obs_times`. Exercises the FlowSum (incidence)
/// projection + chain-binomial dynamics through the full bootstrap filter.
fn build_sir(obs_times: Vec<f64>) -> (MultiStreamObsModel, Arc<CompiledModel>, Vec<f64>) {
    let n = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    let m = Model {
        ic_grad: Default::default(),
        name: "sir_weekly_negbin".into(),
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
            transition(
                "infection",
                vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                div(mul(mul(p("beta"), c("S")), c("I")), n),
            ),
            transition(
                "recovery",
                vec![StoichiometryEntry("I".into(), -1), StoichiometryEntry("R".into(), 1)],
                mul(p("gamma"), c("I")),
            ),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![IrObs {
            name: "weekly_cases".into(),
            source: "weekly_cases".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "weekly_cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "weekly_cases".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            projection: Projection::CumulativeFlow("infection".into()),
            projection_state_grad: Default::default(),
            likelihood: Likelihood::NegBinomial(NegBinomialLikelihood {
                mean: ir::Diffable::new(mul(p("rho"), Expr::Projected(ProjectedExpr { projected: () }))),
                dispersion: ir::Diffable::new(p("k")),
            }),
        }],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![param("beta", 0.4), param("gamma", 0.15), param("rho", 0.6), param("k", 10.0)],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("S".into(), 9990.0);
            h.insert("I".into(), 10.0);
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 80.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 80.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };
    let compiled = Arc::new(CompiledModel::new(m).unwrap());
    let inf = compiled.model.transitions.iter().position(|t| t.name == "infection").unwrap();
    let spec = StreamSpec {
        ir_model: compiled.model.observations[0].clone(),
        projection: StreamProjection::FlowSum(vec![inf]),
        // seed-1 synthetic weekly reported cases (see the sir case README).
        observations: dense_cells(vec![16.0, 166.0, 626.0, 1303.0, 1260.0, 1023.0, 327.0, 91.0, 58.0, 6.0, 2.0]),
        obs_times,
        aux: vec![],
    };
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![spec]).unwrap().0, compiled.clone()).unwrap();
    let params = compiled.default_params.clone();
    (obs_model, compiled, params)
}

/// On-grid weekly observations (the validated `sir` recovery case).
fn sir_incidence() -> (MultiStreamObsModel, Arc<CompiledModel>, Vec<f64>) {
    build_sir(vec![7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0, 63.0, 70.0, 77.0])
}

/// Same model + data, observations at OFF-GRID times (dt=1): the PF lands on each
/// via a shortened final substep. Pins the off-grid likelihood path the on-grid
/// corpus cannot reach — the round-2 review's vacuity gap, and exactly the path
/// the unified-timeline refactor's snap-vs-exact policy changes.
fn sir_incidence_offgrid() -> (MultiStreamObsModel, Arc<CompiledModel>, Vec<f64>) {
    build_sir(vec![7.3, 14.6, 21.9, 29.2, 36.5, 43.8, 51.1, 58.4, 65.7, 73.0, 79.3])
}

type RefBuilder = fn() -> (MultiStreamObsModel, Arc<CompiledModel>, Vec<f64>);
const REFERENCES: &[(&str, RefBuilder)] = &[
    ("sir_incidence_truth", sir_incidence),
    ("sir_incidence_offgrid", sir_incidence_offgrid),
];

/// Committed baselines: (name) -> PF marginal log-likelihood, captured on the dev
/// machine. Re-capture with CAMDL_CAPTURE_BASELINE=1 (see the module header).
const BASELINES: &[(&str, f64)] = &[
    ("sir_incidence_truth", -5.94512991469047165e1),
    ("sir_incidence_offgrid", -5.97885420281019435e1),
];

fn run(builder: RefBuilder) -> f64 {
    let (obs_model, compiled, params) = builder();
    let process = ChainBinomialProcess::new(compiled);
    let config = SMCConfig {
        n_particles: N_PARTICLES,
        dt: DT,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter(&process, &obs_model, &params, &config, SEED)
        .unwrap()
        .log_likelihood
}

#[test]
fn inference_loglik_is_deterministic() {
    // Precondition for a byte-identical baseline: same seed → identical loglik.
    let a = run(sir_incidence);
    let b = run(sir_incidence);
    assert_eq!(a.to_bits(), b.to_bits(), "PF loglik must be deterministic at a fixed seed (got {a} vs {b})");
}

#[test]
fn inference_loglik_baselines_hold() {
    let capture = std::env::var("CAMDL_CAPTURE_BASELINE").is_ok();
    let mut captured: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    for (name, builder) in REFERENCES {
        let ll = run(*builder);
        assert!(ll.is_finite(), "{name}: PF loglik must be finite, got {ll}");
        if capture {
            captured.push(format!("    (\"{name}\", {ll:.17e}),"));
            continue;
        }
        match BASELINES.iter().find(|(n, _)| n == name).map(|(_, v)| *v) {
            Some(expected) => assert_eq!(
                ll.to_bits(),
                expected.to_bits(),
                "INFERENCE LOGLIK CHANGED for {name}: a refactor moved the scored \
                 log-likelihood (got {ll:.17e}, expected {expected:.17e})"
            ),
            None => missing.push(name.to_string()),
        }
    }

    if capture {
        eprintln!("\n// <<CAPTURED-BASELINES>> — paste into BASELINES:");
        for line in &captured {
            eprintln!("{line}");
        }
        return;
    }
    assert!(
        missing.is_empty(),
        "no baseline for: {missing:?} — run with CAMDL_CAPTURE_BASELINE=1 and paste the table"
    );
}
