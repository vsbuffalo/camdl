//! Tests for resolved expression resolution + evaluation.
//!
//! Invariant: eval_resolved(resolve_expr(e), ctx) == eval_expr(e, ctx)
//! for all Expr e and all valid EvalCtx ctx.

use std::collections::HashMap;
use ir::{
    expr::{
        BinOp,
        CondExpr, CondWrap,
        ConstExpr, Expr,
        ParamExpr, PopExpr, PopSumExpr,
        TimeExpr,
        UnOp,
    },
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    Model,
    parameter::Parameter,
};
use sim::{
    compiled_model::CompiledModel,
    propensity::{eval_expr, EvalCtx},
    resolved_expr::{resolve_expr, eval_resolved, eval_resolved_deriv, ResolveCtx},
    state::{IntState, RealState},
};

fn minimal_model(compartments: Vec<Compartment>, params: Vec<Parameter>) -> Model {
    Model {
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
    Parameter { name: name.into(), value: ir::parameter::ParamValue::Fixed { value: value }, param_kind: None, param_dim: None }
}

/// Build a ResolveCtx from a CompiledModel.
fn resolve_ctx_from(model: &CompiledModel) -> ResolveCtx<'_> {
    let table_meta: Vec<(ir::table::OobPolicy, usize)> = model.model.tables.iter()
        .zip(&model.table_values_cache)
        .map(|(t, cached)| (t.out_of_bounds.clone(), cached.len()))
        .collect();
    // Leak the table_meta so the borrow lives long enough.
    // This is fine in tests.
    let table_meta = Box::leak(Box::new(table_meta));
    ResolveCtx {
        comp_index: &model.comp_index,
        param_index: &model.param_index,
        time_func_index: &model.time_func_index,
        table_index: &model.table_index,
        global_to_int: &model.global_to_int,
        global_to_real: &model.global_to_real,
        binding_index: &model.binding_index,
        table_meta,
    }
}

/// gh#audit-C6 / S1: ALLOW_DEGENERATE_RATES is process-global atomic
/// state. Tests that toggle it must serialize via this mutex so a
/// parallel test doesn't see the flag in the wrong state mid-eval.
static EVAL_FLAG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Assert eval_resolved(resolve(expr)) == eval_expr(expr) for given context.
///
/// Enables --allow-degenerate-rates so the degenerate-arithmetic test
/// cases (div-by-zero etc.) take the legacy silent-zero path on both
/// sides. Without the flag, eval_expr returns
/// SimError::NumericalCollapse and eval_resolved returns NaN — they
/// agree on degenerate-handling semantics, but the comparison shape
/// (Result<f64> vs f64) makes equivalence assertion awkward. The
/// comparison test isn't the right place to assert hard-fail behavior;
/// expr_eval.rs::test_*_errors_by_default carries that load.
fn assert_resolved_matches(expr: &Expr, model: &CompiledModel, int_s: &IntState, real_s: &RealState, params: &[f64], t: f64) {
    let _guard = EVAL_FLAG_LOCK.lock().unwrap();
    sim::eval_stats::set_allow_degenerate_rates(true);
    let rctx = resolve_ctx_from(model);
    let resolved = resolve_expr(expr, &rctx).expect("resolve_expr failed");
    let ctx = EvalCtx { model, int_s, real_s, params, t, dt: 1.0, projected: None, aux: None, int_float_override: None };
    let expected = eval_expr(expr, &ctx).expect("eval_expr failed");
    let actual = eval_resolved(&resolved, &ctx);
    sim::eval_stats::set_allow_degenerate_rates(false);
    assert!(
        (expected - actual).abs() < 1e-12,
        "mismatch: eval_expr={}, eval_resolved={}", expected, actual
    );
}

#[test]
fn test_const() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Const(ConstExpr { value: 3.14 });
    assert_resolved_matches(&expr, &model, &int_s, &real_s, &[], 0.0);
}

#[test]
fn test_param() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S")],
        vec![param("beta", 0.5)],
    )).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Param(ParamExpr { param: "beta".into() });
    assert_resolved_matches(&expr, &model, &int_s, &real_s, &[0.5], 0.0);
}

#[test]
fn test_pop() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("I"), int_comp("S")],
        vec![],
    )).unwrap();
    let mut int_s = IntState::new(2);
    int_s.counts[0] = 42;
    int_s.counts[1] = 100;
    let real_s = RealState::new(0);
    let expr = Expr::Pop(PopExpr { pop: "I".into() });
    assert_resolved_matches(&expr, &model, &int_s, &real_s, &[], 0.0);
}

#[test]
fn test_pop_sum() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S"), int_comp("I"), int_comp("R")],
        vec![],
    )).unwrap();
    let mut int_s = IntState::new(3);
    int_s.counts[0] = 100;
    int_s.counts[1] = 50;
    int_s.counts[2] = 30;
    let real_s = RealState::new(0);
    let expr = Expr::PopSum(PopSumExpr { pop_sum: vec!["S".into(), "I".into(), "R".into()] });
    assert_resolved_matches(&expr, &model, &int_s, &real_s, &[], 0.0);
}

#[test]
fn test_time() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let expr = Expr::Time(TimeExpr { time: () });
    assert_resolved_matches(&expr, &model, &int_s, &real_s, &[], 7.5);
}

#[test]
fn test_binop_arithmetic() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S")],
        vec![param("a", 3.0), param("b", 4.0)],
    )).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let params = vec![3.0, 4.0];

    // a + b
    let add = Expr::bin_op(BinOp::Add, Expr::param("a"), Expr::param("b"));
    assert_resolved_matches(&add, &model, &int_s, &real_s, &params, 0.0);

    // a * b
    let mul = Expr::bin_op(BinOp::Mul, Expr::param("a"), Expr::param("b"));
    assert_resolved_matches(&mul, &model, &int_s, &real_s, &params, 0.0);

    // a / b
    let div = Expr::bin_op(BinOp::Div, Expr::param("a"), Expr::param("b"));
    assert_resolved_matches(&div, &model, &int_s, &real_s, &params, 0.0);

    // Div by zero
    let div_zero = Expr::bin_op(BinOp::Div, Expr::param("a"), Expr::const_(0.0));
    assert_resolved_matches(&div_zero, &model, &int_s, &real_s, &params, 0.0);
}

#[test]
fn test_unop() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S")],
        vec![param("x", 2.0)],
    )).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let params = vec![2.0];

    let exp = Expr::un_op(UnOp::Exp, Expr::param("x"));
    assert_resolved_matches(&exp, &model, &int_s, &real_s, &params, 0.0);

    let log = Expr::un_op(UnOp::Log, Expr::param("x"));
    assert_resolved_matches(&log, &model, &int_s, &real_s, &params, 0.0);

    let neg = Expr::un_op(UnOp::Neg, Expr::param("x"));
    assert_resolved_matches(&neg, &model, &int_s, &real_s, &params, 0.0);

    let sqrt = Expr::un_op(UnOp::Sqrt, Expr::param("x"));
    assert_resolved_matches(&sqrt, &model, &int_s, &real_s, &params, 0.0);
}

#[test]
fn test_cond() {
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S")],
        vec![param("x", 5.0)],
    )).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let params = vec![5.0];

    // if x > 3 then 1.0 else 0.0
    let cond = Expr::Cond(CondWrap {
        cond: CondExpr {
            pred: Box::new(Expr::bin_op(BinOp::Gt, Expr::param("x"), Expr::const_(3.0))),
            then: Box::new(Expr::const_(1.0)),
            else_: Box::new(Expr::const_(0.0)),
        },
    });
    assert_resolved_matches(&cond, &model, &int_s, &real_s, &params, 0.0);
}

#[test]
fn test_realistic_rate_expression() {
    // beta * S * I / (S + E + I + R) — a typical SIR force of infection
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S"), int_comp("E"), int_comp("I"), int_comp("R")],
        vec![param("beta", 0.3)],
    )).unwrap();
    let mut int_s = IntState::new(4);
    int_s.counts[0] = 990;  // S
    int_s.counts[1] = 5;    // E
    int_s.counts[2] = 5;    // I
    int_s.counts[3] = 0;    // R
    let real_s = RealState::new(0);
    let params = vec![0.3];

    let n = Expr::pop_sum(vec!["S".into(), "E".into(), "I".into(), "R".into()]);
    let expr = Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(
            BinOp::Mul,
            Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
            Expr::pop("I"),
        ),
        n,
    );
    assert_resolved_matches(&expr, &model, &int_s, &real_s, &params, 0.0);
}

#[test]
fn test_resolve_error_unknown_param() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let rctx = resolve_ctx_from(&model);
    let expr = Expr::Param(ParamExpr { param: "nonexistent".into() });
    assert!(resolve_expr(&expr, &rctx).is_err());
}

#[test]
fn test_resolve_error_unknown_compartment() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let rctx = resolve_ctx_from(&model);
    let expr = Expr::Pop(PopExpr { pop: "nonexistent".into() });
    assert!(resolve_expr(&expr, &rctx).is_err());
}

#[test]
fn test_derivative_matches() {
    // d/d(beta) of (beta * S * I / N) = S * I / N
    let model = CompiledModel::new(minimal_model(
        vec![int_comp("S"), int_comp("I"), int_comp("R")],
        vec![param("beta", 0.3), param("gamma", 0.1)],
    )).unwrap();
    let mut int_s = IntState::new(3);
    int_s.counts[0] = 990;
    int_s.counts[1] = 10;
    int_s.counts[2] = 0;
    let real_s = RealState::new(0);
    let params = vec![0.3, 0.1];

    let n = Expr::pop_sum(vec!["S".into(), "I".into(), "R".into()]);
    let expr = Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(
            BinOp::Mul,
            Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
            Expr::pop("I"),
        ),
        n,
    );

    let rctx = resolve_ctx_from(&model);
    let resolved = resolve_expr(&expr, &rctx).unwrap();
    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &params, t: 0.0, dt: 1.0, projected: None, aux: None, int_float_override: None };

    // beta is param index 0
    let old_deriv = sim::propensity::eval_expr_deriv(&expr, 0, &ctx);
    let new_deriv = eval_resolved_deriv(&resolved, 0, &ctx);
    assert!(
        (old_deriv - new_deriv).abs() < 1e-12,
        "derivative mismatch: eval_expr_deriv={}, eval_resolved_deriv={}", old_deriv, new_deriv
    );

    // gamma is param index 1 — derivative should be 0 (not in expression)
    let old_deriv2 = sim::propensity::eval_expr_deriv(&expr, 1, &ctx);
    let new_deriv2 = eval_resolved_deriv(&resolved, 1, &ctx);
    assert!(
        (old_deriv2 - new_deriv2).abs() < 1e-12,
        "derivative mismatch for gamma: eval_expr_deriv={}, eval_resolved_deriv={}", old_deriv2, new_deriv2
    );
}

#[test]
fn test_projected() {
    let model = CompiledModel::new(minimal_model(vec![int_comp("S")], vec![])).unwrap();
    let int_s = IntState::new(1);
    let real_s = RealState::new(0);
    let rctx = resolve_ctx_from(&model);

    let expr = Expr::Projected(ir::expr::ProjectedExpr { projected: () });
    let resolved = resolve_expr(&expr, &rctx).unwrap();

    let ctx = EvalCtx { model: &model, int_s: &int_s, real_s: &real_s, params: &[], t: 0.0, dt: 1.0, projected: Some(42.0), aux: None, int_float_override: None };
    let val = eval_resolved(&resolved, &ctx);
    assert!((val - 42.0).abs() < 1e-12);
}
