//! gh#314: a forcing's optional `lag` shifts evaluation to `t − lag`.
//!
//! The shift lives in `propensity.rs`'s `Expr::TimeFunc` arm — a single shared
//! shift applied uniformly across every forcing kind. These tests drive that
//! arm directly via `eval_expr` on a built `CompiledModel`, comparing a lagged
//! forcing's value at `t` to the *un-lagged* forcing's value at `t − lag`:
//!
//!   lagged(t)  ==  unlagged(t − lag)     for any lag (Dirac/point delay).
//!
//! Interpolated is the clean case (a known piecewise-affine shape). Three lag
//! forms are covered: a literal duration (`Const`), `lag = 0` (identity), and a
//! `lag` parameter resolved live from the params slice.

use std::collections::HashMap;
use ir::{
    expr::{Expr, TimeFuncRef, TimeFuncWrap},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::{ParamValue, Parameter},
    time_func::{InterpMethod, Interpolated, TimeFuncKind, TimeFunction},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    flat_eval::{build, eval_flat, scratch_capacity, FlatCache, FlatVm},
    propensity::{eval_expr, interpolated_value, EvalCtx},
    resolved_expr::eval_resolved,
    state::{IntState, RealState},
};

// A piecewise-affine interpolated forcing (the clean shape to shift).
const XS: [f64; 6] = [0.0, 1.0, 3.0, 5.0, 8.0, 10.0];
const YS: [f64; 6] = [2.0, 3.5, 2.8, 4.2, 3.1, 5.0];

fn interpolated_kind() -> TimeFuncKind {
    TimeFuncKind::Interpolated(Interpolated {
        times: XS.iter().map(|&x| Expr::const_(x)).collect(),
        values: YS.iter().map(|&y| Expr::const_(y)).collect(),
        method: InterpMethod::Linear,
    })
}

fn param(name: &str, value: f64) -> Parameter {
    Parameter {
        name: name.into(),
        value: ParamValue::Fixed { value },
        param_kind: None,
        param_dim: None,
    }
}

/// Minimal model carrying a single interpolated forcing `vc` with the given
/// `lag`, plus any parameters the lag references.
fn model_with_lag(lag: Option<Expr>, params: Vec<Parameter>) -> Model {
    Model {
        ic_grad: Default::default(),
        name: "lag_test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![TimeFunction {
            name: "vc".into(),
            kind: interpolated_kind(),
            dim: (0, 0), // 'ratio
            lag,
        }],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: params,
        initial_conditions: InitialConditions::Parameterized(HashMap::new()),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 20.0,
            time_semantics: "continuous".into(),
            dt: None,
            rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

/// Evaluate the forcing `vc` at simulation time `t` through the full
/// `Expr::TimeFunc` dispatch (where the lag shift lives).
fn eval_vc(model: &CompiledModel, params: &[f64], t: f64) -> f64 {
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::TimeFunc(TimeFuncWrap { time_func: TimeFuncRef { name: "vc".into() } });
    let ctx = EvalCtx {
        model, int_s: &int_s, real_s: &real_s, params,
        t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
    };
    eval_expr(&expr, &ctx).expect("forcing eval must not error")
}

/// A literal `lag = τ` evaluates the forcing at `t − τ`: `lagged(t)` must equal
/// the raw interpolation at `t − τ` for every probe time.
#[test]
fn literal_lag_shifts_evaluation_by_tau() {
    let tau = 5.0;
    let model = CompiledModel::new(model_with_lag(Some(Expr::const_(tau)), vec![])).unwrap();
    for &t in &[0.0, 2.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 15.0] {
        let lagged = eval_vc(&model, &[], t);
        let expected = interpolated_value(&XS, &YS, t - tau);
        assert!(
            (lagged - expected).abs() < 1e-12,
            "lagged vc({t}) = {lagged}, expected unlagged vc({}) = {expected}",
            t - tau
        );
    }
}

/// `lag = 0` is the identity: a zero-lag forcing is byte-identical to the raw
/// interpolation at `t`, AND identical to a forcing declared without `lag`.
#[test]
fn zero_lag_is_identity() {
    let lagged = CompiledModel::new(model_with_lag(Some(Expr::const_(0.0)), vec![])).unwrap();
    let no_lag = CompiledModel::new(model_with_lag(None, vec![])).unwrap();
    for &t in &[0.0, 2.0, 4.0, 6.5, 8.0, 10.0, 13.0] {
        let raw = interpolated_value(&XS, &YS, t);
        let z = eval_vc(&lagged, &[], t);
        let n = eval_vc(&no_lag, &[], t);
        assert_eq!(z, raw, "lag = 0 must equal the raw interpolation at t = {t}");
        assert_eq!(z, n, "lag = 0 must equal no-lag at t = {t}");
    }
}

/// A `lag` PARAMETER (`lag = tau`) is resolved live from the params slice: the
/// same compiled model, evaluated with two different `tau` values, shifts by
/// each. This is the lag-as-parameter case (a primary motivation) on a
/// gradient-free path (direct evaluation).
#[test]
fn param_lag_resolves_live() {
    let model = CompiledModel::new(
        model_with_lag(Some(Expr::param("tau")), vec![param("tau", 0.0)]),
    ).unwrap();
    let tau_idx = model.param_index["tau"];

    for &tau in &[0.0, 3.0, 5.0, 7.5] {
        let mut p = model.default_params.clone();
        p[tau_idx] = tau;
        for &t in &[2.0, 5.0, 8.0, 11.0, 14.0] {
            let lagged = eval_vc(&model, &p, t);
            let expected = interpolated_value(&XS, &YS, t - tau);
            assert!(
                (lagged - expected).abs() < 1e-12,
                "param lag tau = {tau}: vc({t}) = {lagged}, expected vc({}) = {expected}",
                t - tau
            );
        }
    }
}

/// The flat-bytecode propensity path (`CAMDL_EVAL_FLAT`, gh#209) must apply the
/// SAME lag shift as the recursive `eval_resolved` path — else a lagged forcing
/// is silently wrong under flat eval. The generic byte-identity gate
/// (`flat_eval_byte_identity.rs`) iterates `ir/golden`, none of which carry a
/// lag, so this pins the lagged case explicitly: a transition rate that reads
/// `vc` evaluates bit-identically through both paths.
#[test]
fn flat_eval_matches_standard_path_for_lagged_forcing() {
    // Model with one transition whose rate is the lagged forcing `vc`.
    let mut model = model_with_lag(Some(Expr::const_(5.0)), vec![]);
    model.transitions = vec![Transition {
        rate_state_grad: Default::default(),
        name: "self".into(),
        stoichiometry: vec![StoichiometryEntry("S".into(), 0)],
        rate: Expr::TimeFunc(TimeFuncWrap { time_func: TimeFuncRef { name: "vc".into() } }),
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: HashMap::new(),
        lineage: None,
    }];
    let cm = CompiledModel::new(model).unwrap();
    let params = cm.default_params.clone();

    let vm: FlatVm = build(&cm.resolved.rates, &cm.resolved.bindings);
    let cap = scratch_capacity(&vm);
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);

    for &t in &[0.0, 2.0, 5.0, 8.0, 13.0] {
        let ctx = EvalCtx {
            model: &cm, int_s: &int_s, real_s: &real_s, params: &params,
            t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        let standard = eval_resolved(&cm.resolved.rates[0], &ctx);
        let mut scratch: Vec<f64> = Vec::with_capacity(cap + 16);
        let mut cache = FlatCache::new(vm.n_bindings);
        let flat = eval_flat(&vm, &vm.rates[0], &ctx, &mut scratch, &mut cache);
        assert_eq!(
            standard.to_bits(), flat.to_bits(),
            "flat vs standard lagged vc @t={t}: standard={standard}, flat={flat}"
        );
        // And both must equal the un-lagged interpolation at t − 5.
        let expected = interpolated_value(&XS, &YS, t - 5.0);
        assert!((standard - expected).abs() < 1e-12);
    }
}
