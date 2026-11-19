//! `init {}` is evaluated at the start of the run, not at `t = 0`.
//!
//! The language spec says initial conditions are evaluated at `t_start`
//! (§15.1, §17), the time `simulate { from }` sets. Every evaluation in the
//! initial-state seam (`initial_state_mean`, `_draw`, `_logpdf`,
//! `_logpdf_grad`, `initial_state_continuous`, `ic_grad_seed`) built its
//! evaluation context with `t: 0.0`, so an entry that reads `t` or a forcing
//! took its value at day 0 for a run that starts later. A port of He, Ionides
//! & King (2010), whose `init {}` scales the initial fractions by the
//! population forcing on the first day, started from the population at day 0
//! instead of at its own start.
//!
//! Each test starts the run at day 50 and seeds `A = t`: the seeded value must
//! be 50 on every path.

use ir::{
    expr::Expr,
    model::{
        Compartment, CompartmentKind, InitCountLaw, InitSpec, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::PoissonLikelihood,
    parameter::{ParamValue, Parameter},
    Diffable, Model,
};
use sim::compiled_model::CompiledModel;

const T_START: f64 = 50.0;

/// Two integer compartments `A`, `B`, no transitions, a run starting at
/// `t_start`, and whatever `init` entries the caller supplies.
fn model(t_start: f64, init: Vec<(&str, InitSpec)>) -> Model {
    Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "init_at_t_start".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "A".into(), kind: CompartmentKind::Integer },
            Compartment { name: "B".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "A0".into(),
            value: ParamValue::Fixed { value: 500.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions(
            init.into_iter().map(|(k, s)| (k.to_string(), s)).collect(),
        ),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![t_start]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start,
            t_end: t_start + 1.0,
            time_semantics: "continuous".into(),
            dt: None,
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

fn det(e: Expr) -> InitSpec {
    InitSpec::Deterministic(e)
}

/// `B ~ poisson(rate = t)`.
fn poisson_at_t() -> InitSpec {
    InitSpec::Count(InitCountLaw::Poisson(PoissonLikelihood { rate: Diffable::new(Expr::time()) }))
}

#[test]
fn a_deterministic_entry_reads_t_at_the_run_start() {
    let compiled = CompiledModel::new(model(T_START, vec![("A", det(Expr::time()))])).expect("build");
    let params = compiled.default_params.clone();

    let (int_s, _) = compiled.initial_state_mean(&params).expect("mean");
    assert_eq!(int_s.counts[0], 50, "A = t must be the run's start, 50, not 0");

    let (int_c, _) = compiled.initial_state_continuous(&params).expect("continuous");
    assert_eq!(int_c[0], 50.0, "the ODE path must evaluate A = t at the run's start too");

    // A run starting at 0 is unchanged.
    let at_zero = CompiledModel::new(model(0.0, vec![("A", det(Expr::time()))])).expect("build");
    let (z, _) = at_zero.initial_state_mean(&at_zero.default_params.clone()).expect("mean");
    assert_eq!(z.counts[0], 0, "a run from t = 0 seeds A = 0, as before");
}

#[test]
fn a_law_entry_reads_t_at_the_run_start_in_its_draw_and_its_density() {
    let compiled = CompiledModel::new(model(
        T_START,
        vec![("A", det(Expr::time())), ("B", poisson_at_t())],
    ))
    .expect("build");
    let params = compiled.default_params.clone();

    // The draw: a Poisson(50) mean over many draws, far from Poisson(0) = 0.
    let mut rng = sim::rng::StatefulRng::new(7);
    let n = 2000;
    let mut sum = 0i64;
    for _ in 0..n {
        let (int_s, _) = compiled.initial_state_draw(&params, &mut rng).expect("draw");
        assert_eq!(int_s.counts[0], 50, "the deterministic entry in a law-bearing block reads t = 50");
        sum += int_s.counts[1];
    }
    let mean = sum as f64 / n as f64;
    assert!((mean - 50.0).abs() < 1.0, "B ~ poisson(rate = t) must draw around 50, got mean {mean}");

    // The density of x0 = (50, 50) is the Poisson(50) pmf at 50.
    let ll = compiled.initial_state_logpdf(&[50, 50], &[], &params).expect("logpdf");
    let expected = 50.0 * 50f64.ln() - 50.0 - (1..=50).map(|k| (k as f64).ln()).sum::<f64>();
    assert!((ll - expected).abs() < 1e-9, "log p(B = 50) must be the Poisson(50) pmf: got {ll}, want {expected}");
}
