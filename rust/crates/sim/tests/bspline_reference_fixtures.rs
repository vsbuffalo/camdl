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
use sim::periodic_bspline::eval_periodic_bspline;

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
        let p: u32 = r[0].parse().unwrap();
        let t: f64 = r[1].parse().unwrap();
        let want: f64 = r[2].parse().unwrap();
        let got = eval_periodic_bspline(t, 4.0, 7, p, &coefs);
        let err = (got - want).abs();
        assert!(err <= TOL, "periodic p={p} t={t}: camdl {got} vs scipy {want}");
        worst = worst.max(err);
    }
    eprintln!("periodic spline vs scipy: max |diff| = {worst:.3e} over {} values", rows.len());
}
