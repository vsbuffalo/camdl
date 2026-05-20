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
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        compartments,
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
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
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    }
}

fn int_comp(name: &str) -> Compartment {
    Compartment { name: name.into(), kind: CompartmentKind::Integer }
}

fn param(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: Some(value), bounds: None, prior: None, transform: None, initial_value: None, param_kind: None, param_dim: None, hierarchical: None }
}

#[test]
fn test_const() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Const(ConstExpr { value: 3.14 });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &params, t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 150.0).abs() < 1e-12);
}

#[test]
fn test_time() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Time(TimeExpr { time: () });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 7.5, dt: 1.0, projected: None, int_float_override: None };
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
        let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 0.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!((result - 10.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_div_by_zero_errors_by_default() {
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert_eq!(result, 1.0);
}

// ── NaN / edge-case guard tests ───────────────────────────────────────

#[test]
fn test_log_negative_returns_neg_inf() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr { op: UnOp::Log, arg: Box::new(Expr::Const(ConstExpr { value: -1.0 })) },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
    let result = eval_expr(&expr, &ctx).unwrap();
    assert!(result.is_infinite() && result < 0.0, "log(-1) should be -inf, got {}", result);
}

#[test]
fn test_sqrt_negative_errors_by_default() {
    // gh#audit-C6 / S1.
    use sim::{CollapseKind, SimError};
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::UnOp(UnOpWrap {
        un_op: UnOpExpr { op: UnOp::Sqrt, arg: Box::new(Expr::Const(ConstExpr { value: -4.0 })) },
    });
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_expr(&expr, &ctx).unwrap_err();
    assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::SqrtNegative, .. }),
        "Sqrt of negative must produce NumericalCollapse{{SqrtNegative}}, got {:?}", err);
    sim::eval_stats::set_allow_degenerate_rates(true);
    assert_eq!(eval_expr(&expr, &ctx).unwrap(), 0.0);
    sim::eval_stats::set_allow_degenerate_rates(false);
}

#[test]
fn test_pow_negative_base_fractional_exp_errors_by_default() {
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
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
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: None, int_float_override: None };
    sim::eval_stats::set_allow_degenerate_rates(false);
    let err = eval_expr(&expr, &ctx).unwrap_err();
    assert!(matches!(err, SimError::NumericalCollapse { kind: CollapseKind::PowNanInf, .. }),
        "0^(-1) must produce NumericalCollapse{{PowNanInf}}, got {:?}", err);
    sim::eval_stats::set_allow_degenerate_rates(true);
    assert_eq!(eval_expr(&expr, &ctx).unwrap(), 0.0);
    sim::eval_stats::set_allow_degenerate_rates(false);
}
