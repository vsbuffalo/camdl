//! The span evaluator against the naive Cox–de Boor recursion it replaced
//! (proposal 2026-10-27-estimable-bsplines §9 Stage 1,
//! `basis_matches_cox_de_boor_recursion`). The recursion evaluates each basis
//! function independently from its definition, so agreement checks the
//! triangle, the span lookup and the periodic fold together. The old code is
//! kept only as this oracle, in `tests/support/recursive_bspline.rs`.

#[path = "support/recursive_bspline.rs"]
mod recursive_bspline;

use recursive_bspline::{bspline_value, eval_periodic_bspline_recursive};
use sim::bspline::{span, Knots, MAX_DEGREE};
use sim::periodic_bspline::PeriodicBasis;

/// Clamped spans against the recursion on the full clamped knot vector, at
/// t ∈ [a, b) (the recursion's half-open degree-0 indicator gives 0 at b; the
/// end value is checked by `bspline::tests`). Values are O(1); 1e-12 absolute.
#[test]
fn basis_matches_cox_de_boor_recursion() {
    let layouts = [
        Knots::uniform(0.0, 168.0, 9).unwrap(),
        Knots::uniform(1000.0, 1168.0, 5).unwrap(),
        Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap(),
    ];
    let mut worst: f64 = 0.0;
    for k in &layouts {
        let (a, b) = k.window();
        let n = k.n_intervals();
        let xs: Vec<f64> = (0..=n).map(|i| breakpoint(k, i)).collect();
        for p in 0..=MAX_DEGREE {
            let mut u = vec![a; p];
            u.extend(&xs);
            u.extend(std::iter::repeat_n(b, p));
            for i in 0..2_000 {
                let t = a + (b - a) * i as f64 / 2_000.0;
                let s = span(k, p, t);
                for j in 0..n + p {
                    let want = bspline_value(t, j, p as u32, &u);
                    let got = if (s.first..=s.first + p).contains(&j) { s.values[j - s.first] } else { 0.0 };
                    let err = (got - want).abs();
                    assert!(err <= 1e-12, "{k:?} p={p} t={t} j={j}: span {got} vs recursion {want}");
                    worst = worst.max(err);
                }
            }
        }
    }
    eprintln!("clamped span vs recursion: max |diff| = {worst:.3e}");
}

fn breakpoint(k: &Knots, i: usize) -> f64 {
    match k {
        Knots::Uniform { a, h, .. } => a + i as f64 * h,
        Knots::Explicit(xs) => xs[i],
    }
}

/// The new periodic evaluator against the old one, every degree 0–5 and
/// every n_basis from p + 1 to 12, over several periods either side of zero.
/// The two sum in different orders, so they agree to rounding, not bitwise:
/// 1e-12 of the coefficient scale.
#[test]
fn periodic_matches_the_recursive_evaluator() {
    let mut worst: f64 = 0.0;
    for p in 0..=MAX_DEGREE {
        for nb in p + 1..=12 {
            for period in [1.0, 7.0, 365.25] {
                let c: Vec<f64> = (0..nb).map(|i| 0.3 + ((i * 7 + 3) % 11) as f64 / 5.0).collect();
                let scale = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let basis = PeriodicBasis::new(period, nb, p).unwrap();
                for i in 0..1_000 {
                    let t = period * (-3.0 + 7.0 * i as f64 / 1_000.0);
                    let want = eval_periodic_bspline_recursive(t, period, nb as u32, p as u32, &c);
                    let got = basis.eval(t, |k| c[k]);
                    let err = (got - want).abs();
                    assert!(err <= 1e-12 * scale, "p={p} nb={nb} period={period} t={t}: {got} vs {want}");
                    worst = worst.max(err / scale);
                }
            }
        }
    }
    eprintln!("periodic vs recursive evaluator: max |diff|/max|c| = {worst:.3e}");
}
