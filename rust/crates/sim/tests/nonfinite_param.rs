//! gh#81 Phase 2 regression: when a non-finite parameter (NaN / ±Inf)
//! reaches `eval_propensities`, the evaluator must surface a structured
//! `SimError::NonFiniteParameter { name, value, t }` error — not the
//! generic `NumericalCollapse { kind: DivByZero }` that fires downstream
//! once NaN propagates through `beta * S * I / N`. The variant must
//! classify as per-particle recoverable so PGAS / PMMH proposal
//! mechanisms can reject the proposal and continue.
//!
//! Phase 1 diagnosis traced the WA fit failure (`numerical collapse
//! (DivByZero) at t=-101` with `S=7.6e6`, `I=0`, `beta=NaN`) to a NUTS
//! leapfrog step that produced a NaN parameter. The downstream rate
//! evaluator then blamed the rate expression `beta*S*I/N`, hiding the
//! upstream proposal failure.

use std::path::Path;
use sim::{
    compiled_model::CompiledModel,
    error::SimError,
    propensity::eval_propensities,
    state::{IntState, RealState},
};

fn load_model(name: &str) -> ir::Model {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = Path::new(&manifest)
        .join("../../../ir/golden")
        .join(format!("{}.ir.json", name));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("could not read {}", path.display()));
    ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("failed to parse {}: {}", path.display(), e))
}

fn apply_baseline(model: &mut ir::Model) {
    if let Some(preset) = model.presets.first() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
}

/// At rate-eval entry the upstream condition (a parameter that is
/// already NaN before any expression touched it) is the actionable
/// fault. Asserting on the generic NumericalCollapse{DivByZero} that
/// fires downstream once NaN propagates would mis-diagnose: the rate
/// expression is innocent.
#[test]
fn nonfinite_param_in_rate_eval_returns_structured_error() {
    let mut model = load_model("sir_basic");
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    // Build a params slice with `beta = NaN`. `default_params` is in
    // `param_index` order — locate beta and corrupt that slot.
    let beta_idx = *compiled.param_index.get("beta").expect("sir_basic has beta");
    let mut params = compiled.default_params.clone();
    params[beta_idx] = f64::NAN;

    // State at t=-101 in the WA repro: S large, I=0. Doesn't matter for
    // the upstream check; the guard must fire before rate eval runs.
    let int_s = IntState::from_vec(vec![990, 10, 0]);
    let real_s = RealState::new(0);
    let mut propensities = Vec::new();

    let err = eval_propensities(
        &compiled, &int_s, &real_s, &params, -101.0, 1.0, None, &mut propensities,
    ).expect_err("non-finite param should surface a structured error");

    match err {
        SimError::NonFiniteParameter { name, value, t } => {
            assert_eq!(name, "beta", "expected beta as the offending param, got {}", name);
            assert!(value.is_nan(), "expected NaN, got {}", value);
            assert_eq!(t, -101.0, "expected t=-101, got {}", t);
        }
        other => panic!(
            "expected SimError::NonFiniteParameter, got {:?}.\n\
             Hint: the generic NumericalCollapse {{ DivByZero }} this used \
             to produce blamed the rate expression. The actual fault is \
             upstream — a NaN/Inf parameter reaching rate evaluation, \
             typically from a NUTS leapfrog or MH proposal.",
            other
        ),
    }
}

#[test]
fn nonfinite_param_pos_infinity_returns_structured_error() {
    let mut model = load_model("sir_basic");
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    let beta_idx = *compiled.param_index.get("beta").expect("sir_basic has beta");
    let mut params = compiled.default_params.clone();
    params[beta_idx] = f64::INFINITY;

    let int_s = IntState::from_vec(vec![990, 10, 0]);
    let real_s = RealState::new(0);
    let mut propensities = Vec::new();

    let err = eval_propensities(
        &compiled, &int_s, &real_s, &params, 0.0, 1.0, None, &mut propensities,
    ).expect_err("+Inf param should surface a structured error");

    match err {
        SimError::NonFiniteParameter { name, value, t: _ } => {
            assert_eq!(name, "beta");
            assert!(value.is_infinite() && value > 0.0, "expected +Inf, got {}", value);
        }
        other => panic!("expected NonFiniteParameter, got {:?}", other),
    }
}

#[test]
fn nonfinite_param_neg_infinity_returns_structured_error() {
    let mut model = load_model("sir_basic");
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();

    let beta_idx = *compiled.param_index.get("beta").expect("sir_basic has beta");
    let mut params = compiled.default_params.clone();
    params[beta_idx] = f64::NEG_INFINITY;

    let int_s = IntState::from_vec(vec![990, 10, 0]);
    let real_s = RealState::new(0);
    let mut propensities = Vec::new();

    let err = eval_propensities(
        &compiled, &int_s, &real_s, &params, 0.0, 1.0, None, &mut propensities,
    ).expect_err("-Inf param should surface a structured error");

    match err {
        SimError::NonFiniteParameter { name, value, t: _ } => {
            assert_eq!(name, "beta");
            assert!(value.is_infinite() && value < 0.0, "expected -Inf, got {}", value);
        }
        other => panic!("expected NonFiniteParameter, got {:?}", other),
    }
}

/// PGAS / PMMH proposal mechanisms (NUTS leapfrog, MH random-walk) can
/// produce NaN/Inf parameters when adaptation goes pathological. The
/// inference layer treats this as a per-particle / per-proposal failure:
/// reject and continue, not tear down the chain. The new variant must
/// classify under the same recoverable umbrella as NumericalCollapse.
#[test]
fn nonfinite_param_is_per_particle_recoverable() {
    let err = SimError::NonFiniteParameter {
        name: "beta".into(),
        value: f64::NAN,
        t: -101.0,
    };
    assert!(
        err.is_per_particle_recoverable(),
        "NonFiniteParameter must be per-particle recoverable so PGAS/PMMH \
         can reject the offending proposal rather than killing the chain."
    );

    // Sanity: structural errors (UnknownParameter, etc.) are NOT
    // recoverable. This is the contrast that pins the classifier's intent.
    let structural = SimError::UnknownParameter("ghost".into());
    assert!(!structural.is_per_particle_recoverable());
}

/// Sanity: finite params still flow through cleanly. The upstream guard
/// must not regress the happy path.
#[test]
fn finite_params_still_evaluate_normally() {
    let mut model = load_model("sir_basic");
    apply_baseline(&mut model);
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();

    let int_s = IntState::from_vec(vec![990, 10, 0]);
    let real_s = RealState::new(0);
    let mut propensities = Vec::new();

    eval_propensities(
        &compiled, &int_s, &real_s, &params, 0.0, 1.0, None, &mut propensities,
    ).expect("happy path: finite params should evaluate without error");

    assert!(!propensities.is_empty(), "expected non-empty propensities vector");
    for &p in &propensities {
        assert!(p.is_finite() && p >= 0.0, "propensity should be finite non-negative, got {}", p);
    }
}
