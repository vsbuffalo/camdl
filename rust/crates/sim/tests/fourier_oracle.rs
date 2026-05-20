//! Cross-validate camdl's Fourier forcing evaluator against numpy.
//!
//! Math is direct (`Σ_k a_k cos(2π k t/T) + b_k sin(...)`) so the
//! risk is transcription error rather than algorithm choice. The
//! numpy oracle still catches index-off-by-one, harmonic-shift, and
//! period-sign mistakes that a hand-computed test would miss.

use ir::{
    expr::{ConstExpr, Expr},
    time_func::Fourier,
    time_func::TimeFuncKind,
};
use sim::compiled_model::{CompiledModel, CompiledTimeFuncKind};
use sim::propensity::eval_time_func;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn read_tsv_pairs(path: &Path) -> Vec<(f64, f64)> {
    let f = File::open(path).unwrap_or_else(|e|
        panic!("could not open {}: {}", path.display(), e));
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line.unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("t\t") {
            continue;
        }
        let cols: Vec<&str> = trimmed.split('\t').collect();
        assert_eq!(cols.len(), 2, "expected 2 columns");
        out.push((cols[0].parse().unwrap(), cols[1].parse().unwrap()));
    }
    out
}

fn compile_fourier(period: f64, harmonics: &[(f64, f64)]) -> CompiledTimeFuncKind {
    let tf = ir::time_func::TimeFunction {
        name: "f".into(),
        kind: TimeFuncKind::Fourier(Fourier {
            period: Expr::Const(ConstExpr { value: period }),
            harmonics: harmonics.iter().map(|(a, b)| (
                Expr::Const(ConstExpr { value: *a }),
                Expr::Const(ConstExpr { value: *b }),
            )).collect(),
        }),
        dim: (0, 0),
    };
    let m = ir::Model {
        name: "test".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        compartments: vec![ir::model::Compartment {
            name: "S".into(),
            kind: ir::model::CompartmentKind::Integer,
        }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![tf],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
        initial_conditions: ir::model::InitialConditions::Parameterized(HashMap::new()),
        output: ir::model::OutputConfig {
            times: ir::model::OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: ir::model::SimulationConfig {
            t_start: 0.0, t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: None, rng_seed: Some(42),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![],
    };
    let cm = CompiledModel::new(m).unwrap();
    cm.time_func_cache[0].kind.clone()
}

#[test]
fn fourier_matches_numpy() {
    let path = Path::new("tests/fixtures/fourier_numpy.tsv");
    let pairs = read_tsv_pairs(path);
    assert!(!pairs.is_empty(), "fourier fixture is empty");

    // Must match the fixture generator's parameters
    // (scripts/gen_fourier_numpy_fixture.py).
    let tf = compile_fourier(365.25, &[(0.2, 0.1), (0.05, -0.07), (0.03, 0.02)]);

    let mut max_diff: f64 = 0.0;
    for (t, expected) in &pairs {
        let actual = eval_time_func(&tf, *t);
        let diff = (actual - expected).abs();
        if diff > max_diff { max_diff = diff; }
    }
    assert!(
        max_diff < 1e-12,
        "camdl Fourier vs numpy max |diff| = {:.3e}; threshold 1e-12",
        max_diff
    );
}
