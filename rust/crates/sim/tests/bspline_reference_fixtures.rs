//! camdl's B-spline basis against two independent references (proposal
//! `docs/dev/proposals/2026-10-27-estimable-bsplines.md` §8.1, R1):
//! scipy's `BSpline` and R's `splines::splineDesign`, for clamped knots over
//! three layouts (uniform from 0, uniform at absolute time [1000, 1168],
//! irregular) at degrees 0–5; and scipy for the periodic spline at degrees
//! 0–5. Points include the window ends, every breakpoint, points between and
//! points outside the window, where camdl holds the end value.
//!
//! The fixtures are oracles, not inputs: `make fixtures-bspline` regenerates
//! them byte for byte, and CI reads the committed files offline. The values
//! are O(1), so the tolerance is absolute, 1e-12.

use std::path::Path;

use sim::bspline::{basis, Knots};
use sim::periodic_bspline::PeriodicBasis;

const TOL: f64 = 1e-12;

/// The layouts the generators define (`LAYOUTS` in
/// `scripts/gen_bspline_scipy_fixtures.py`).
fn layout(name: &str) -> Knots {
    match name {
        "uniform" => Knots::uniform(0.0, 168.0, 9).unwrap(),
        "uniform_offset" => Knots::uniform(1000.0, 1168.0, 9).unwrap(),
        "irregular" => {
            Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap()
        }
        other => panic!("unknown layout {other:?}"),
    }
}

/// The data rows of a `#`-commented TSV with a header line, split on tabs.
fn rows(file: &str) -> Vec<Vec<String>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|l| l.split('\t').map(str::to_string).collect())
        .collect()
}

/// Max |camdl − reference| over every (layout, degree, t, k) row, and the
/// row count, so an empty fixture cannot pass.
fn clamped_max_error(file: &str) -> (f64, usize) {
    let rows = rows(file);
    let mut worst: f64 = 0.0;
    for r in &rows {
        let knots = layout(&r[0]);
        let p: usize = r[1].parse().unwrap();
        let t: f64 = r[2].parse().unwrap();
        let k: usize = r[3].parse().unwrap();
        let want: f64 = r[4].parse().unwrap();
        let got = basis(&knots, p, t, k);
        let err = (got - want).abs();
        assert!(err <= TOL, "{file}: {} p={p} t={t} k={k}: camdl {got} vs {want}", r[0]);
        worst = worst.max(err);
    }
    (worst, rows.len())
}

#[test]
fn clamped_basis_matches_scipy() {
    let (worst, n) = clamped_max_error("bspline_basis_scipy.tsv");
    assert!(n > 8000, "scipy fixture has {n} rows");
    eprintln!("clamped basis vs scipy: max |diff| = {worst:.3e} over {n} values");
}

#[test]
fn clamped_basis_matches_r_splines() {
    let (worst, n) = clamped_max_error("bspline_basis_splines.tsv");
    assert!(n > 8000, "splines fixture has {n} rows");
    eprintln!("clamped basis vs R splines: max |diff| = {worst:.3e} over {n} values");
}

#[test]
fn periodic_spline_matches_scipy_at_degrees_0_to_5() {
    let coefs = [0.7, 1.2, 0.9, 0.5, 1.1, 0.8, 1.3];
    let rows = rows("periodic_bspline_degrees_scipy.tsv");
    assert_eq!(rows.len(), 6 * 205);
    let mut worst: f64 = 0.0;
    for r in &rows {
        let p: usize = r[0].parse().unwrap();
        let t: f64 = r[1].parse().unwrap();
        let want: f64 = r[2].parse().unwrap();
        let got = PeriodicBasis::new(4.0, 7, p).unwrap().eval(t, |k| coefs[k]);
        let err = (got - want).abs();
        assert!(err <= TOL, "periodic p={p} t={t}: camdl {got} vs scipy {want}");
        worst = worst.max(err);
    }
    eprintln!("periodic spline vs scipy: max |diff| = {worst:.3e} over {} values", rows.len());
}

/// R1: the curvature penalty Ω (exact (p − 1)-point Gauss–Legendre on
/// camdl's A2.3 second derivatives) against scipy's second-derivative splines
/// integrated by adaptive quadrature, ≤ 1e-12·max|Ω| per layout and degree.
/// The reference is independent of both the rule and the recurrence.
#[test]
fn curvature_penalty_matches_scipy() {
    use sim::spline_penalty::penalty;
    let rows = rows("bspline_penalty_scipy.tsv");
    assert!(rows.len() > 300, "penalty fixture has {} rows", rows.len());
    let mut groups: std::collections::BTreeMap<(String, usize), Vec<(usize, usize, f64)>> = Default::default();
    for r in &rows {
        groups.entry((r[0].clone(), r[1].parse().unwrap()))
            .or_default()
            .push((r[2].parse().unwrap(), r[3].parse().unwrap(), r[4].parse().unwrap()));
    }
    for ((name, p), entries) in groups {
        let om = penalty(&layout(&name), p);
        let scale = om.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut worst: f64 = 0.0;
        for (i, j, want) in entries {
            worst = worst.max((om.at(i, j) - want).abs());
        }
        assert!(worst <= 1e-12 * scale, "{name} p={p}: max |Ω − scipy| = {worst:e}, max|Ω| = {scale:e}");
        eprintln!("Ω vs scipy {name} p={p}: {:.2e} of max|Ω|", worst / scale);
    }
}

/// R1 for `smooth`: the runtime's forcing values at fixed (a, b, τ, z)
/// against an independent numpy construction of w = a + b(ξ − t_m) +
/// (τ/κ)Mz (the proposal's Householder and Cholesky steps, κ = √(K − 2)·per^{3/2})
/// evaluated by scipy, with the lag and unit scale: `bspline_smooth_grad`'s
/// log-link `f_log` and identity `f_id`, 15 times each, inside, at knots, at
/// the ends and past them, ≤ 1e-12 relative. Pins κ and M in the value path,
/// which the gradient checks see only through consistency with the basis.
#[test]
fn smooth_values_match_numpy() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/fixtures/gradient/ir/bspline_smooth_grad.ir.json");
    let mut m = ir::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let z = [0.8, -0.5, 1.1, 0.2, -0.9, 0.4];
    let y = [0.6, -1.2, 0.3, 0.9];
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "a" => -1.5, "b" => 0.01, "tau" => 0.4, "c" => 0.3, "d" => 0.02, "s" => 0.7,
            n if n.starts_with("z_m") => z[n[3..].parse::<usize>().unwrap() - 1],
            n if n.starts_with("y_i") => y[n[3..].parse::<usize>().unwrap() - 1],
            n => panic!("unexpected parameter {n}"),
        };
        p.value = p.value.with_value(v);
    }
    let model = sim::compiled_model::CompiledModel::new(m).unwrap();
    let (int_s, real_s) = model.initial_state_mean(&model.default_params).unwrap();
    let rows = rows("bspline_smooth_values_numpy.tsv");
    assert_eq!(rows.len(), 30);
    let mut worst: f64 = 0.0;
    for r in &rows {
        let idx = model.model.time_functions.iter().position(|tf| tf.name == r[0]).unwrap();
        let (t, want): (f64, f64) = (r[1].parse().unwrap(), r[2].parse().unwrap());
        let ctx = sim::propensity::EvalCtx {
            model: &model, int_s: &int_s, real_s: &real_s, params: &model.default_params,
            t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        let got = sim::resolved_expr::eval_resolved(&sim::resolved_expr::ResolvedExpr::TimeFunc(idx), &ctx);
        let rel = (got - want).abs() / want.abs();
        worst = worst.max(rel);
        assert!(rel <= 1e-12, "{} t = {t}: camdl {got} vs numpy {want}", r[0]);
    }
    eprintln!("smooth values vs numpy: worst relative {worst:.2e}");
}
