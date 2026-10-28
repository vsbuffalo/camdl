//! The `bspline` forcing kind at runtime (proposal
//! 2026-10-27-estimable-bsplines §4.3, §4.4), on hand-built IR: the compiled
//! forcing evaluates the clamped span of `sim::bspline`, holds its end values
//! outside the window, applies `link = log` as exp of the basis sum, reads its
//! coefficients live, and refuses malformed knots at build, naming the forcing.

use ir::{
    expr::Expr,
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::{ParamValue, Parameter},
    time_func::{BSpline, BSplineKnots, SplineCoefs, SplineLink, TimeFuncKind, TimeFunction},
    Model,
};
use sim::bspline::{span, Knots};
use sim::compiled_model::CompiledModel;
use sim::resolved_expr::{eval_resolved, ResolvedExpr};

/// One compartment, one forcing `f`, one fixed parameter `v` (= 1).
fn model_with(kind: TimeFuncKind) -> Model {
    Model {
        ic_grad: Default::default(),
        name: "bspline".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![TimeFunction { name: "f".into(), kind, dim: (0, 0), lag: None, data_source: None }],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "v".into(),
            value: ParamValue::Fixed { value: 1.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::default(),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: None, rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

fn bspline(knots: BSplineKnots, degree: u32, coefs: Vec<Expr>, link: SplineLink) -> TimeFuncKind {
    TimeFuncKind::BSpline(BSpline { knots, degree, coefs: SplineCoefs::Explicit { coefs }, link })
}

fn consts(cs: &[f64]) -> Vec<Expr> {
    cs.iter().map(|&c| Expr::const_(c)).collect()
}

/// The forcing's value at `t` with parameter `v` set to `v`.
fn forcing_at(model: &CompiledModel, v: f64, t: f64) -> f64 {
    let (int_s, real_s) = model.initial_state_mean(&model.default_params).unwrap();
    let params = [v];
    let ctx = sim::propensity::EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params: &params,
        t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    eval_resolved(&ResolvedExpr::TimeFunc(0), &ctx)
}

/// Σ cₖ Bₖ(t) straight from the span evaluator.
fn spline_sum(knots: &Knots, p: usize, t: f64, c: &[f64]) -> f64 {
    let s = span(knots, p, t);
    (0..=p).map(|r| s.values[r] * c[s.first + r]).sum()
}

const C6: [f64; 6] = [0.3, 0.45, 0.6, 0.35, 0.2, 0.25];

/// Uniform window [10, 90], n_basis 6, cubic: the forcing is the span sum bit
/// for bit, interpolates c₀ at a and c₅ at b, and holds those values outside.
#[test]
fn uniform_window_evaluates_the_span_and_holds_outside() {
    let model = CompiledModel::new(model_with(bspline(
        BSplineKnots::Uniform { lower: Expr::const_(10.0), upper: Expr::const_(90.0), n_intervals: 3 },
        3,
        consts(&C6),
        SplineLink::Identity,
    )))
    .unwrap();
    let knots = Knots::uniform(10.0, 90.0, 3).unwrap();
    for i in 0..=400 {
        let t = -20.0 + 140.0 * i as f64 / 400.0;
        assert_eq!(forcing_at(&model, 1.0, t).to_bits(), spline_sum(&knots, 3, t, &C6).to_bits(), "t={t}");
    }
    assert!((forcing_at(&model, 1.0, 10.0) - C6[0]).abs() <= 1e-15);
    assert!((forcing_at(&model, 1.0, 90.0) - C6[5]).abs() <= 1e-15);
    for t in [-1e6, 0.0, 9.999] {
        assert_eq!(forcing_at(&model, 1.0, t), forcing_at(&model, 1.0, 10.0), "held before the window, t={t}");
    }
    for t in [90.001, 120.0, 1e6] {
        assert_eq!(forcing_at(&model, 1.0, t), forcing_at(&model, 1.0, 90.0), "held after the window, t={t}");
    }
}

/// Explicit, irregular breakpoints at degree 2.
#[test]
fn explicit_breakpoints_evaluate_the_span() {
    let bps = [0.0, 10.0, 25.0, 30.0, 80.0];
    let c = [0.3, 0.45, 0.6, 0.35, 0.2, 0.25];
    let model = CompiledModel::new(model_with(bspline(
        BSplineKnots::Explicit { breakpoints: consts(&bps) },
        2,
        consts(&c),
        SplineLink::Identity,
    )))
    .unwrap();
    let knots = Knots::explicit(bps.to_vec()).unwrap();
    for i in 0..=400 {
        let t = -5.0 + 95.0 * i as f64 / 400.0;
        assert_eq!(forcing_at(&model, 1.0, t).to_bits(), spline_sum(&knots, 2, t, &c).to_bits(), "t={t}");
    }
}

/// `link = log`: the value is exp of the identity-link value, inside the
/// window and in the held regions, to 1e-15 relative.
#[test]
fn log_link_is_exp_of_the_basis_sum() {
    let knots = || BSplineKnots::Uniform { lower: Expr::const_(0.0), upper: Expr::const_(60.0), n_intervals: 3 };
    let g = CompiledModel::new(model_with(bspline(knots(), 3, consts(&C6), SplineLink::Identity))).unwrap();
    let f = CompiledModel::new(model_with(bspline(knots(), 3, consts(&C6), SplineLink::Log))).unwrap();
    for i in 0..=300 {
        let t = -10.0 + 90.0 * i as f64 / 300.0;
        let (lf, lg) = (forcing_at(&f, 1.0, t), forcing_at(&g, 1.0, t).exp());
        assert!(((lf - lg) / lg).abs() <= 1e-15, "t={t}: {lf} vs exp(g) {lg}");
    }
}

/// A coefficient naming a parameter is live: the model builds once and the
/// value moves with the parameter in the params slice.
#[test]
fn a_parameter_coefficient_is_live() {
    let mut cs = consts(&C6);
    cs[2] = Expr::param("v");
    let model = CompiledModel::new(model_with(bspline(
        BSplineKnots::Uniform { lower: Expr::const_(0.0), upper: Expr::const_(60.0), n_intervals: 3 },
        3,
        cs,
        SplineLink::Identity,
    )))
    .unwrap();
    let knots = Knots::uniform(0.0, 60.0, 3).unwrap();
    for v in [0.6, 2.0, -1.0] {
        let mut c = C6;
        c[2] = v;
        for t in [5.0, 21.0, 33.3, 59.0] {
            assert_eq!(forcing_at(&model, v, t).to_bits(), spline_sum(&knots, 3, t, &c).to_bits(), "v={v} t={t}");
        }
    }
}

fn build_error(kind: TimeFuncKind) -> String {
    CompiledModel::new(model_with(kind)).err().expect("must be refused").to_string()
}

#[test]
fn malformed_bsplines_are_refused_naming_the_forcing() {
    let uni = |a: f64, b: f64, n| BSplineKnots::Uniform { lower: Expr::const_(a), upper: Expr::const_(b), n_intervals: n };
    let cases = [
        (bspline(uni(50.0, 10.0, 3), 3, consts(&C6), SplineLink::Identity), "must have a < b"),
        (
            bspline(BSplineKnots::Explicit { breakpoints: consts(&[0.0, 30.0, 20.0, 80.0]) }, 3, consts(&C6), SplineLink::Identity),
            "breakpoints must increase",
        ),
        (bspline(uni(0.0, 60.0, 3), 3, consts(&C6[..5]), SplineLink::Identity), "coefs length (5) must equal"),
        (bspline(uni(0.0, 60.0, 3), 6, consts(&[1.0; 9]), SplineLink::Identity), "degree (6) must be at most 5"),
    ];
    for (kind, want) in cases {
        let err = build_error(kind);
        assert!(err.contains("bspline forcing 'f'") && err.contains(want), "expected {want:?}, got: {err}");
    }
    // A parameter in a knot is structural.
    let err = build_error(bspline(
        BSplineKnots::Uniform { lower: Expr::const_(0.0), upper: Expr::param("v"), n_intervals: 3 },
        3,
        consts(&C6),
        SplineLink::Identity,
    ));
    assert!(err.contains("structural") && err.contains("'v'"), "{err}");
}

/// `link` applies to `periodic_spline` too (§4.4): exp of the identity value.
#[test]
fn periodic_spline_log_link_is_exp_of_the_basis_sum() {
    use ir::time_func::PeriodicSpline;
    let ps = |link| {
        TimeFuncKind::PeriodicSpline(PeriodicSpline {
            period: Expr::const_(7.0),
            n_basis: 6,
            degree: 3,
            coefs: consts(&C6),
            link,
        })
    };
    let g = CompiledModel::new(model_with(ps(SplineLink::Identity))).unwrap();
    let f = CompiledModel::new(model_with(ps(SplineLink::Log))).unwrap();
    for i in 0..=200 {
        let t = -7.0 + 21.0 * i as f64 / 200.0;
        let (lf, lg) = (forcing_at(&f, 1.0, t), forcing_at(&g, 1.0, t).exp());
        assert!(((lf - lg) / lg).abs() <= 1e-15, "t={t}: {lf} vs exp(g) {lg}");
    }
}
