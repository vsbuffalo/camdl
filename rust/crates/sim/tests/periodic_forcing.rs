//! Validate periodic forcing bin lookup against hand-computed values.
//! An off-by-one in the bin index would shift the school calendar by one bin width.

use std::collections::HashMap;
use ir::{
    expr::{ConstExpr, Expr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    time_func::{Fourier, Periodic, PeriodicSpline, TimeFuncKind, TimeFunction},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    propensity::eval_time_func,
};

fn model_with_periodic(period: f64, values: Vec<f64>) -> CompiledModel {
    let tf = TimeFunction {
        name: "school".into(),
        kind: TimeFuncKind::Periodic(Periodic {
            period: Expr::Const(ConstExpr { value: period }),
            values: values.iter().map(|&v| Expr::Const(ConstExpr { value: v })).collect(),
        }),
        // GH #8: 'ratio dim — dimensionless multiplier (0/1 indicator here)
        dim: (0, 0),
    };
    let model = Model {
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![tf],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
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
            dt: None, rng_seed: Some(42),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    CompiledModel::new(model).unwrap()
}

#[test]
fn test_periodic_4_bins() {
    // Period=100, 4 bins of width 25: values [1, 2, 3, 4]
    // Bin 0: t ∈ [0, 25)  → 1
    // Bin 1: t ∈ [25, 50) → 2
    // Bin 2: t ∈ [50, 75) → 3
    // Bin 3: t ∈ [75, 100) → 4
    let cm = model_with_periodic(100.0, vec![1.0, 2.0, 3.0, 4.0]);
    let tf = &cm.time_func_cache[0].kind;

    let cases: Vec<(f64, f64)> = vec![
        (0.0, 1.0), (12.5, 1.0), (24.9, 1.0),
        (25.0, 2.0), (37.5, 2.0), (49.9, 2.0),
        (50.0, 3.0), (62.5, 3.0), (74.9, 3.0),
        (75.0, 4.0), (87.5, 4.0), (99.9, 4.0),
        // Wrapping: t=100 → phase=0 → bin 0
        (100.0, 1.0), (125.0, 2.0),
        // Negative time: t=-10 wraps to phase=90 → bin 3
        (-10.0, 4.0),
    ];

    for (t, expected) in &cases {
        let actual = eval_time_func(tf, *t);
        assert_eq!(
            actual, *expected,
            "periodic(t={}) = {}, expected {}", t, actual, expected
        );
    }
}

#[test]
fn test_periodic_school_calendar_52_weeks() {
    // He et al. school calendar: 52 bins over 365.25 days (7.024 days/bin)
    // Known values: bin 0 (day 0) = holiday, bin 2 (day ~14) = term, bin 15 (day ~105) = holiday
    let mut values = vec![0.0; 52];
    // Term weeks (1-indexed in He et al.): 2-14, 17-28, 37-43, 45-51
    // 0-indexed: 1-13, 16-27, 36-42, 44-50
    for i in 1..=13 { values[i] = 1.0; }
    for i in 16..=27 { values[i] = 1.0; }
    for i in 36..=42 { values[i] = 1.0; }
    for i in 44..=50 { values[i] = 1.0; }

    let cm = model_with_periodic(365.25, values.clone());
    let tf = &cm.time_func_cache[0].kind;
    let bin_width = 365.25 / 52.0;

    // Check each bin's midpoint
    for (i, &expected) in values.iter().enumerate() {
        let t = (i as f64 + 0.5) * bin_width; // midpoint of bin i
        let actual = eval_time_func(tf, t);
        assert_eq!(
            actual, expected,
            "school(t={:.1}, bin={}) = {}, expected {}", t, i, actual, expected
        );
    }

    // Check that day 0 is holiday (bin 0)
    assert_eq!(eval_time_func(tf, 0.0), 0.0, "day 0 should be holiday");

    // Check wrapping: year 2, same pattern
    let t_year2 = 365.25 + 2.0 * bin_width; // bin 2 of year 2
    assert_eq!(eval_time_func(tf, t_year2), 1.0, "year 2 bin 2 should be term");
}

#[test]
fn test_periodic_boundary_no_out_of_bounds() {
    // Edge case: t exactly at period boundary
    let cm = model_with_periodic(10.0, vec![1.0, 2.0]);
    let tf = &cm.time_func_cache[0].kind;

    // t=10.0 → phase=0.0 → bin 0
    assert_eq!(eval_time_func(tf, 10.0), 1.0);
    // t=5.0 → phase=5.0 → bin 1
    assert_eq!(eval_time_func(tf, 5.0), 2.0);
    // t=4.99999 → bin 0
    assert_eq!(eval_time_func(tf, 4.999), 1.0);
}

// ── gh#59: Fourier + PeriodicSpline ──────────────────────────────────────────

fn model_with_kind(kind: TimeFuncKind) -> CompiledModel {
    let tf = TimeFunction { name: "f".into(), kind, dim: (0, 0) };
    let model = Model {
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![tf],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
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
            dt: None, rng_seed: Some(42),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    CompiledModel::new(model).unwrap()
}

fn ec(v: f64) -> Expr { Expr::Const(ConstExpr { value: v }) }

#[test]
fn test_fourier_pure_cos_first_harmonic() {
    // (a_1, b_1) = (1, 0), period = 1.0:
    // f(t) = cos(2π t).
    //   t = 0    → 1
    //   t = 0.25 → 0
    //   t = 0.5  → -1
    //   t = 0.75 → 0
    //   t = 1.0  → 1   (periodicity)
    let cm = model_with_kind(TimeFuncKind::Fourier(Fourier {
        period: ec(1.0),
        harmonics: vec![(ec(1.0), ec(0.0))],
    }));
    let tf = &cm.time_func_cache[0].kind;
    let cases = [(0.0_f64, 1.0), (0.25, 0.0), (0.5, -1.0), (0.75, 0.0), (1.0, 1.0)];
    for (t, expected) in cases {
        let actual = eval_time_func(tf, t);
        assert!(
            (actual - expected).abs() < 1e-9,
            "fourier cos(2π·{}) = {}, expected {}", t, actual, expected
        );
    }
}

#[test]
fn test_fourier_pure_sin_second_harmonic() {
    // (a_1, b_1) = (0, 0), (a_2, b_2) = (0, 1), period = 1.0:
    // f(t) = sin(4π t).
    //   t = 0.125 → sin(π/2) = 1
    //   t = 0.25  → sin(π)   = 0
    //   t = 0.375 → sin(3π/2)= -1
    let cm = model_with_kind(TimeFuncKind::Fourier(Fourier {
        period: ec(1.0),
        harmonics: vec![(ec(0.0), ec(0.0)), (ec(0.0), ec(1.0))],
    }));
    let tf = &cm.time_func_cache[0].kind;
    assert!((eval_time_func(tf, 0.125) - 1.0).abs() < 1e-9);
    assert!((eval_time_func(tf, 0.25)  - 0.0).abs() < 1e-9);
    assert!((eval_time_func(tf, 0.375) + 1.0).abs() < 1e-9);
}

#[test]
fn test_fourier_zero_harmonics_returns_zero() {
    let cm = model_with_kind(TimeFuncKind::Fourier(Fourier {
        period: ec(365.25),
        harmonics: vec![],
    }));
    let tf = &cm.time_func_cache[0].kind;
    for &t in &[0.0_f64, 1.0, 100.0, -50.0] {
        assert_eq!(eval_time_func(tf, t), 0.0);
    }
}

#[test]
fn test_periodic_spline_partition_of_unity() {
    // gh#59 v2: with all coefs = 1, the B-spline sums to 1 everywhere
    // (partition of unity is preserved by the periodic wrap-fold).
    let cm = model_with_kind(TimeFuncKind::PeriodicSpline(PeriodicSpline {
        period:  ec(4.0),
        n_basis: 6,
        degree:  3,
        coefs:   vec![ec(1.0); 6],
    }));
    let tf = &cm.time_func_cache[0].kind;
    for t in [0.0_f64, 0.5, 1.0, 1.7, 2.0, 3.3, 3.99] {
        let v = eval_time_func(tf, t);
        assert!(
            (v - 1.0).abs() < 1e-12,
            "partition-of-unity violated at t={}: {}", t, v
        );
    }
}

#[test]
fn test_periodic_spline_wraps() {
    // gh#59 v2: f(t) == f(t + period).
    let cm = model_with_kind(TimeFuncKind::PeriodicSpline(PeriodicSpline {
        period:  ec(4.0),
        n_basis: 6,
        degree:  3,
        coefs:   vec![ec(0.7), ec(1.2), ec(0.9), ec(0.5), ec(1.1), ec(0.8)],
    }));
    let tf = &cm.time_func_cache[0].kind;
    for t in [0.123_f64, 1.5, 2.9, 3.6] {
        let a = eval_time_func(tf, t);
        let b = eval_time_func(tf, t + 4.0);
        let c = eval_time_func(tf, t - 4.0);
        assert!((a - b).abs() < 1e-12, "periodicity at t={}: a={} t+P={}", t, a, b);
        assert!((a - c).abs() < 1e-12, "periodicity at t={}: a={} t-P={}", t, a, c);
    }
}
