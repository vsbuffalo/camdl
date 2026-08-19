//! Unit tests for the expression evaluator (§A.4).

use std::collections::HashMap;
use ir::{
    expr::{
        BinOp, BinOpExpr, BinOpWrap,
        CondExpr, CondWrap,
        ConstExpr, DtExpr, Expr,
        ParamExpr, PopExpr, PopSumExpr,
        TimeExpr,
        UnOp, UnOpExpr, UnOpWrap,
    },
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    Model,
    parameter::Parameter,
};
use sim::{
    compiled_model::CompiledModel,
    propensity::{eval_expr, EvalCtx},
    state::{IntState, RealState},
};

fn minimal_model(compartments: Vec<Compartment>, params: Vec<Parameter>) -> Model {
    Model {
        ic_grad: Default::default(),
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments,
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
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
            t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: None,
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

fn param(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ir::parameter::ParamValue::Fixed { value: value }, param_kind: None, param_dim: None }
}

// ── gh#127 (#12): runtime out-of-range table lookup returns Err, never panics ──
//
// A NON-CONSTANT table index (state/param-dependent) is not statically
// range-checkable, so it reaches the runtime evaluator. Previously the fast
// path (eval_resolved) `panic!`d on an out-of-range index under
// OobPolicy::Error, tearing down the whole process — one bad particle could
// crash an entire inference run. The fix routes it through the existing typed-
// error boundary (eval_propensities): it must return a SimError, not panic.
// ── Serialization for the degenerate-rate tests (gh#481) ─────────────────────
//
// `sim::eval_stats::allow_degenerate_rates` is a PROCESS-GLOBAL, and `cargo
// test` runs the tests in a binary on parallel threads, so the tests below
// clobber each other: one sets it `true` while another is mid-assertion that a
// degenerate rate is *rejected*, and that one fails with the value allowed
// through. Hold this lock for the whole of any test that touches the flag.
//
// The flag itself is deliberately left global — it is set once from the CLI at
// startup and read by rayon workers, so making it thread-local would change
// runtime semantics to fix a test-harness problem.
static DEGENERATE_RATES_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// Poison-tolerant: a genuine failure in one of these tests must not cascade
// into six more by poisoning the mutex.
fn degenerate_rates_guard() -> std::sync::MutexGuard<'static, ()> {
    DEGENERATE_RATES_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn test_runtime_oob_table_lookup_returns_err_not_panic() {
    use ir::{
        expr::{TableLookupExpr, TableLookupWrap},
        table::{OobPolicy, Table, TableSource},
        transition::{StoichiometryEntry, Transition},
    };
    use sim::propensity::eval_propensities;

    // Model: one compartment `S`, one transition whose rate is `kernel[S]`.
    // The table `kernel` has 2 cells (valid indices {0, 1}); S starts at 5, so
    // the lookup index is out of range — but the index references state, so
    // validate() cannot reject it statically (verified separately in the ir
    // crate's table_lookup_nonconstant_index_is_not_range_checked).
    let mut m = minimal_model(vec![int_comp("S")], vec![]);
    m.tables.push(Table {
        name: "kernel".into(),
        source: TableSource::Inline {
            values: vec![Expr::Const(ConstExpr { value: 1.0 }),
                         Expr::Const(ConstExpr { value: 2.0 })],
        },
        out_of_bounds: OobPolicy::Error,
        cell_kind: None,
    });
    m.transitions.push(Transition {
        rate_state_grad: Default::default(),
        name: "leave_S".into(),
        stoichiometry: vec![StoichiometryEntry("S".into(), -1)],
        rate: Expr::TableLookup(TableLookupWrap {
            table_lookup: TableLookupExpr {
                table: "kernel".into(),
                indices: vec![Expr::Pop(PopExpr { pop: "S".into() })],
            },
        }),
        metadata: None,
        draw_method: Default::default(),
        rate_grad: HashMap::new(),
        lineage: None,
    });

    let model = CompiledModel::new(m).expect("model with state-dependent table index must compile");
    let int_s = IntState::from_vec(vec![5]); // S = 5 → index 5, out of range for a 2-cell table
    let real_s = RealState::new(0);
    let mut out = Vec::new();
    // Must be Err (a typed SimError), NOT a panic. Against the buggy code this
    // call panics inside eval_resolved (red); the fix makes it return Err.
    let res = eval_propensities(&model, &int_s, &real_s, &[], 0.0, 1.0, None, &mut out);
    assert!(
        res.is_err(),
        "out-of-range table lookup must return Err, not panic or a value; got {:?}",
        res
    );
    // The error must name the table — error quality is a feature (CLAUDE.md).
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("kernel"),
        "table-lookup OOB error should name the offending table 'kernel'; got: {msg}"
    );
}

#[test]
fn test_const() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Const(ConstExpr { value: 3.14 });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 3.14).abs() < 1e-12);
}

#[test]
fn test_param() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S")],
        vec![param("beta", 0.5)],
    )).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let params = vec![0.5f64];
    let expr = Expr::Param(ParamExpr { param: "beta".into() });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &params, t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 0.5).abs() < 1e-12);
}

#[test]
fn test_pop_integer() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("I"), int_comp("S")],
        vec![],
    )).unwrap();
    let mut int_s = IntState::new(2);
    int_s.counts[0] = 42; // I is first
    let real_s = RealState::new(0);
    let expr = Expr::Pop(PopExpr { pop: "I".into() });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 42.0).abs() < 1e-12);
}

#[test]
fn test_pop_sum() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S"), int_comp("I"), int_comp("R")],
        vec![],
    )).unwrap();
    let int_s = IntState::from_vec(vec![100, 20, 30]);
    let real_s = RealState::new(0);
    let expr = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 150.0).abs() < 1e-12);
}

#[test]
fn test_time() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Time(TimeExpr { time: () });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 7.5, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 7.5).abs() < 1e-12);
}

#[test]
fn test_dt_evaluates_to_ctx_dt() {
    // gh#54: Expr::Dt should read EvalCtx.dt at runtime.
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Dt(DtExpr { dt: () });
    for &dt in &[1.0_f64, 0.5, 0.25, 0.1, 7.0] {
        let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt, projected: None, aux: None, int_float_override: None, per_eval: None };
        let result = eval_expr(&expr, &ctx).unwrap();
        assert!((result - dt).abs() < 1e-12, "dt={} got {}", dt, result);
    }
}

#[test]
fn test_dt_serde_roundtrip() {
    // gh#54: {"dt": null} ↔ Expr::Dt(DtExpr { dt: () }).
    let original = Expr::dt();
    let json = serde_json::to_string(&original).unwrap();
    assert_eq!(json, r#"{"dt":null}"#);
    let parsed: Expr = serde_json::from_str(&json).unwrap();
    assert_eq!(original, parsed);
}

// gh#58: trig primitives

fn eval_unop(op: UnOp, arg: f64) -> f64 {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr { op, arg: Box::new(Expr::Const(ConstExpr { value: arg })) },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 0.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    eval_expr(&expr, &ctx).unwrap()
}

#[test]
fn test_sin_known_points() {
    assert!((eval_unop(UnOp::Sin, 0.0) - 0.0).abs() < 1e-12);
    assert!((eval_unop(UnOp::Sin, std::f64::consts::FRAC_PI_2) - 1.0).abs() < 1e-12);
    assert!((eval_unop(UnOp::Sin, std::f64::consts::PI) - 0.0).abs() < 1e-12);
}

#[test]
fn test_cos_known_points() {
    assert!((eval_unop(UnOp::Cos, 0.0) - 1.0).abs() < 1e-12);
    assert!((eval_unop(UnOp::Cos, std::f64::consts::FRAC_PI_2) - 0.0).abs() < 1e-12);
    assert!((eval_unop(UnOp::Cos, std::f64::consts::PI) - (-1.0)).abs() < 1e-12);
}

#[test]
fn test_tanh_known_points() {
    assert!((eval_unop(UnOp::Tanh, 0.0) - 0.0).abs() < 1e-12);
    // tanh(∞) → 1; large finite arg approximates well
    assert!((eval_unop(UnOp::Tanh, 100.0) - 1.0).abs() < 1e-12);
    assert!((eval_unop(UnOp::Tanh, -100.0) - (-1.0)).abs() < 1e-12);
}

#[test]
fn test_trig_serde_roundtrip() {
    for op in [UnOp::Sin, UnOp::Cos, UnOp::Tanh] {
        let original = Expr::UnOp(UnOpWrap {
            un_op: UnOpExpr { op: op.clone(), arg: Box::new(Expr::Const(ConstExpr { value: 1.5 })) },
        });
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Expr = serde_json::from_str(&json).unwrap();
        assert_eq!(original, parsed, "round-trip failed for {:?}", op);
    }
}

#[test]
fn test_binop_add() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Add,
            left: Box::new(Expr::Const(ConstExpr { value: 3.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 4.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 7.0).abs() < 1e-12);
}

#[test]
fn test_binop_mul() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Mul,
            left: Box::new(Expr::Const(ConstExpr { value: 6.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 7.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 42.0).abs() < 1e-12);
}

#[test]
fn test_binop_div() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Div,
            left: Box::new(Expr::Const(ConstExpr { value: 10.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 3.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 10.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_div_by_zero_errors_by_default() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    // gh#audit-C6 / S1: division by zero used to silently return 0.0
    // (wrapped in Ok(_)) — masking malformed rate expressions. Now
    // it returns SimError::NumericalCollapse{DivByZero} by default;
    // the legacy Ok(0.0) is only opt-in via --allow-degenerate-rates.
    use sim::{CollapseKind, SimError};
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Div,
            left: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 0.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_expr(&expr, &ctx).unwrap_err();
    assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::DivByZero, .. }),
        "Div by zero must produce NumericalCollapse{{DivByZero}}, got {:?}", err);

    // Legacy silent-zero behaviour is still available under opt-in.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let r = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(r, 0.0, "with --allow-degenerate-rates, div-by-zero returns 0.0");
    sim::eval_stats::set_allow_degenerate_rates(false); // reset for other tests
}

#[test]
fn test_unop_exp() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Exp,
            arg: Box::new(Expr::Const(ConstExpr { value: 1.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_unop_neg() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Neg,
            arg: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    assert!((eval_expr(&expr, &ctx).unwrap() - (-5.0)).abs() < 1e-12);
}

#[test]
fn test_unop_log() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Log,
            arg: Box::new(Expr::Const(ConstExpr { value: std::f64::consts::E })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    assert!((eval_expr(&expr, &ctx).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_unop_sqrt() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Sqrt,
            arg: Box::new(Expr::Const(ConstExpr { value: 16.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    assert!((eval_expr(&expr, &ctx).unwrap() - 4.0).abs() < 1e-12);
}

#[test]
fn test_unop_abs() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Abs,
            arg: Box::new(Expr::Const(ConstExpr { value: -7.5 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    assert!((eval_expr(&expr, &ctx).unwrap() - 7.5).abs() < 1e-12);
}

#[test]
fn test_unop_floor() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Floor,
            arg: Box::new(Expr::Const(ConstExpr { value: 3.7 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    assert!((eval_expr(&expr, &ctx).unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn test_unop_ceil() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr {
            op: UnOp::Ceil,
            arg: Box::new(Expr::Const(ConstExpr { value: 3.2 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    assert!((eval_expr(&expr, &ctx).unwrap() - 4.0).abs() < 1e-12);
}

#[test]
fn test_cond_pred_positive() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    // cond(1.0, 5.0, 0.0) → pred>0 → 5.0
    let expr = Expr::Cond(CondWrap {
        cond: CondExpr {
            pred: Box::new(Expr::Const(ConstExpr { value: 1.0 })),
            then: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
            else_: Box::new(Expr::Const(ConstExpr { value: 0.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 5.0);
}

#[test]
fn test_cond_pred_zero() {
    // pred=0 → falsy → else branch
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Cond(CondWrap {
        cond: CondExpr {
            pred: Box::new(Expr::Const(ConstExpr { value: 0.0 })),
            then: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
            else_: Box::new(Expr::Const(ConstExpr { value: 0.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_cond_pred_negative() {
    // pred<0 → falsy → else branch
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Cond(CondWrap {
        cond: CondExpr {
            pred: Box::new(Expr::Const(ConstExpr { value: -1.0 })),
            then: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
            else_: Box::new(Expr::Const(ConstExpr { value: 99.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 99.0);
}

#[test]
fn test_binop_gt_true() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Gt,
            left: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 3.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 1.0);
}

#[test]
fn test_binop_gt_false() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Gt,
            left: Box::new(Expr::Const(ConstExpr { value: 2.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_binop_eq_true() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Eq,
            left: Box::new(Expr::Const(ConstExpr { value: 4.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 4.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 1.0);
}

#[test]
fn test_binop_le() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    // 3 <= 3 → true (1.0)
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Le,
            left: Box::new(Expr::Const(ConstExpr { value: 3.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 3.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 1.0);
}

// ── NaN / edge-case guard tests ───────────────────────────────────────

#[test]
fn test_log_nonpositive_errors_by_default() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    // gh#audit-C6 / S1, item 17: `log(x ≤ 0)` is a domain error (no real
    // result), exactly like `sqrt(neg)` — it must route through the same typed
    // NumericalCollapse under the strict default, NOT silently return −inf.
    use sim::{CollapseKind, SimError};
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    for value in [-1.0_f64, 0.0] {
        let expr = Expr::UnOp(UnOpWrap {
            un_op: UnOpExpr { op: UnOp::Log, arg: Box::new(Expr::Const(ConstExpr { value })) },
        });
        sim::eval_stats::set_allow_degenerate_rates(false);
        let err = eval_expr(&expr, &ctx).unwrap_err();
        assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::LogNonPositive, .. }),
            "log({value}) must produce NumericalCollapse{{LogNonPositive}}, got {:?}", err);
        // --allow-degenerate-rates coerces the degenerate rate to 0, like sqrt.
        sim::eval_stats::set_allow_degenerate_rates(true);
        assert_eq!(eval_expr(&expr, &ctx).unwrap(), 0.0,
            "with --allow-degenerate-rates, log({value}) coerces to 0.0");
        sim::eval_stats::set_allow_degenerate_rates(false);
    }
}

#[test]
fn test_sqrt_negative_errors_by_default() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    // gh#audit-C6 / S1.
    use sim::{CollapseKind, SimError};
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr { op: UnOp::Sqrt, arg: Box::new(Expr::Const(ConstExpr { value: -4.0 })) },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_expr(&expr, &ctx).unwrap_err();
    assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::SqrtNegative, .. }),
        "Sqrt of negative must produce NumericalCollapse{{SqrtNegative}}, got {:?}", err);
    sim::eval_stats::set_allow_degenerate_rates(true);
    assert_eq!(eval_expr(&expr, &ctx).unwrap(), 0.0);
    sim::eval_stats::set_allow_degenerate_rates(false);
}

// item 17: the fast (pre-resolved) propensity path must reject a `log(x ≤ 0)`
// rate the same way `eval_expr` does — a −inf that used to slip past the
// `is_nan` boundary guard is now caught and surfaced as a typed collapse.
#[test]
fn test_log_nonpositive_rate_errors_via_eval_propensities() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    use ir::transition::{StoichiometryEntry, Transition};
    use sim::propensity::eval_propensities;

    // rate = log(S - 5); with S = 1 the argument is -4 → domain error.
    let mut m = minimal_model(vec![int_comp("S")], vec![]);
    m.transitions.push(Transition {
        rate_state_grad: Default::default(),
        name: "leave_S".into(),
        stoichiometry: vec![StoichiometryEntry("S".into(), -1)],
        rate: Expr::UnOp(UnOpWrap { un_op: UnOpExpr {
            op: UnOp::Log,
            arg: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                op: BinOp::Sub,
                left: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                right: Box::new(Expr::Const(ConstExpr { value: 5.0 })),
            }})),
        }}),
        metadata: None,
        draw_method: Default::default(),
        rate_grad: HashMap::new(),
        lineage: None,
    });
    let model = CompiledModel::new(m).unwrap();
    let int_s = IntState::from_vec(vec![1]); // S = 1 → log(-4)
    let real_s = RealState::new(0);
    let mut out = Vec::new();

    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_propensities(&model, &int_s, &real_s, &[], 0.0, 1.0, None, &mut out)
        .expect_err("log of a non-positive rate must hard-error under the strict default");
    // Must be a NumericalCollapse (the log −inf now routes through the is_finite
    // boundary guard), NOT the NegativePropensity the old −inf produced — that
    // distinction is what makes this assertion non-vacuous against the old code.
    assert!(matches!(err, sim::SimError::NumericalCollapse { .. }),
        "strict log(≤0) rate must be a NumericalCollapse, not NegativePropensity; got {err:?}");
    // --allow-degenerate-rates coerces the degenerate rate to a finite 0.
    sim::eval_stats::set_allow_degenerate_rates(true);
    eval_propensities(&model, &int_s, &real_s, &[], 0.0, 1.0, None, &mut out)
        .expect("with --allow-degenerate-rates the degenerate rate coerces to 0");
    assert_eq!(out, vec![0.0]);
    sim::eval_stats::set_allow_degenerate_rates(false);
}

// item 17: a resolved propensity that is +inf (an overflow that escaped the
// per-op guards, e.g. exp of a large argument) is not a usable rate. The
// `is_finite` boundary guard must reject it rather than push +inf.
#[test]
fn test_infinite_propensity_is_rejected() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    use ir::transition::{StoichiometryEntry, Transition};
    use sim::propensity::eval_propensities;

    // rate = exp(1000 * S); with S = 1 this is exp(1000) = +inf.
    let mut m = minimal_model(vec![int_comp("S")], vec![]);
    m.transitions.push(Transition {
        rate_state_grad: Default::default(),
        name: "leave_S".into(),
        stoichiometry: vec![StoichiometryEntry("S".into(), -1)],
        rate: Expr::UnOp(UnOpWrap { un_op: UnOpExpr {
            op: UnOp::Exp,
            arg: Box::new(Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                op: BinOp::Mul,
                left: Box::new(Expr::Const(ConstExpr { value: 1000.0 })),
                right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
            }})),
        }}),
        metadata: None,
        draw_method: Default::default(),
        rate_grad: HashMap::new(),
        lineage: None,
    });
    let model = CompiledModel::new(m).unwrap();
    let int_s = IntState::from_vec(vec![1]); // S = 1 → exp(1000) = +inf
    let real_s = RealState::new(0);
    let mut out = Vec::new();

    sim::eval_stats::set_allow_degenerate_rates(false);
    let res = eval_propensities(&model, &int_s, &real_s, &[], 0.0, 1.0, None, &mut out);
    assert!(res.is_err(),
        "a +inf resolved propensity must be rejected, not used as a rate; got {:?}", out);
    sim::eval_stats::set_allow_degenerate_rates(false);
}

#[test]
fn test_pow_negative_base_fractional_exp_errors_by_default() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    // gh#audit-C6 / S1.
    use sim::{CollapseKind, SimError};
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Pow,
            left: Box::new(Expr::Const(ConstExpr { value: -2.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: 0.5 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_expr(&expr, &ctx).unwrap_err();
    assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::PowNanInf, .. }),
        "(-2)^0.5 must produce NumericalCollapse{{PowNanInf}}, got {:?}", err);
    sim::eval_stats::set_allow_degenerate_rates(true);
    assert_eq!(eval_expr(&expr, &ctx).unwrap(), 0.0);
    sim::eval_stats::set_allow_degenerate_rates(false);
}

#[test]
fn test_pow_zero_to_negative_errors_by_default() {
    let _degenerate_rates_guard = degenerate_rates_guard();
    // gh#audit-C6 / S1.
    use sim::{CollapseKind, SimError};
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Pow,
            left: Box::new(Expr::Const(ConstExpr { value: 0.0 })),
            right: Box::new(Expr::Const(ConstExpr { value: -1.0 })),
        },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None };
    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_expr(&expr, &ctx).unwrap_err();
    assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::PowNanInf, .. }),
        "0^(-1) must produce NumericalCollapse{{PowNanInf}}, got {:?}", err);
    sim::eval_stats::set_allow_degenerate_rates(true);
    assert_eq!(eval_expr(&expr, &ctx).unwrap(), 0.0);
    sim::eval_stats::set_allow_degenerate_rates(false);
}
