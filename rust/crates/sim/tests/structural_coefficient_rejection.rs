//! Structural forcing data (interpolation knots, the cubic-spline basis,
//! periodic-spline coefficients, piecewise steps) is precomputed `f64` and
//! cannot be a live parameter — its value feeds a sorted knot table or a
//! construction-time solve, so it cannot vary per step. A param-referencing
//! entry must be rejected at `CompiledModel::new` with a clear error, never
//! silently baked to its default value (the freeze). Proposal
//! `2026-06-09-const-parametric-forcing.md` §3 ("reject estimated spline
//! knots").
//!
//! Counterpart to the OCaml `autodiff` E600 floor: the OCaml compiler rejects
//! such a model at `camdlc` time (no gradient), and this Rust check is the
//! defense-in-depth at IR-load time for a hand-built or future-emitted IR that
//! reaches the runtime with a structural param.

use std::collections::HashMap;
use ir::{
    expr::Expr,
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::{ParamValue, Parameter},
    time_func::{Interpolated, InterpMethod, PeriodicSpline, Periodic, TimeFuncKind, TimeFunction},
    Model,
};
use sim::compiled_model::CompiledModel;

/// Minimal one-compartment model carrying a single forcing `kind`, with `v`
/// declared as a (fixed-valued) parameter so the model can build far enough to
/// process the forcing.
fn model_with_forcing(kind: TimeFuncKind) -> Model {
    Model {
        name: "structural".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![TimeFunction { name: "f".into(), kind, dim: (0, 0) }],
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
        initial_conditions: InitialConditions::Parameterized(HashMap::new()),
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
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![],
    }
}

fn assert_structural_rejection(kind: TimeFuncKind, what: &str) {
    let err = CompiledModel::new(model_with_forcing(kind))
        .err()
        .unwrap_or_else(|| panic!("{what}: expected a structural-rejection error, model built"));
    let msg = format!("{err}");
    assert!(
        msg.contains("structural") && msg.contains("'v'"),
        "{what}: error should name the parameter and call it structural; got: {msg}"
    );
}

#[test]
fn cubic_spline_knot_value_param_is_rejected() {
    // Interpolated + Spline → a cubic-spline basis (construction-time Thomas
    // solve); a parameter in a knot value cannot be live.
    assert_structural_rejection(
        TimeFuncKind::Interpolated(Interpolated {
            times: vec![Expr::const_(0.0), Expr::const_(1.0), Expr::const_(2.0)],
            values: vec![Expr::const_(0.0), Expr::param("v"), Expr::const_(0.0)],
            method: InterpMethod::Spline,
        }),
        "cubic-spline knot value",
    );
}

#[test]
fn linear_interp_knot_value_param_is_rejected() {
    assert_structural_rejection(
        TimeFuncKind::Interpolated(Interpolated {
            times: vec![Expr::const_(0.0), Expr::const_(1.0)],
            values: vec![Expr::param("v"), Expr::const_(1.0)],
            method: InterpMethod::Linear,
        }),
        "linear interpolation knot value",
    );
}

#[test]
fn periodic_spline_coef_param_is_rejected() {
    // de Boor basis coefficients are structural; a parameter coef cannot be live.
    assert_structural_rejection(
        TimeFuncKind::PeriodicSpline(PeriodicSpline {
            period: Expr::const_(4.0),
            n_basis: 6,
            degree: 3,
            coefs: vec![
                Expr::const_(1.0), Expr::param("v"), Expr::const_(1.0),
                Expr::const_(1.0), Expr::const_(1.0), Expr::const_(1.0),
            ],
        }),
        "periodic-spline coefficient",
    );
}

#[test]
fn periodic_step_value_param_builds_live() {
    // Sanity counterpart: a Periodic forcing IS a scalar-coefficient kind (value
    // half), so a parameter step value is live, not rejected — it builds. (Its
    // NUTS gradient is the separate, guarded concern; gh#215.)
    let built = CompiledModel::new(model_with_forcing(TimeFuncKind::Periodic(Periodic {
        period: Expr::const_(7.0),
        values: vec![Expr::param("v"), Expr::const_(1.0)],
    })));
    assert!(built.is_ok(),
        "a periodic step value referencing a parameter is live (value half), not a \
         structural rejection: {:?}", built.err());
}
