//! Before/after: the recursive, allocating periodic B-spline evaluator against
//! the span evaluator that replaced it (proposal
//! 2026-10-27-estimable-bsplines §4.1, §9 Stage 1).
//!
//!   cargo bench -p sim --bench spline_eval
//!
//! Custom main, no criterion harness, as `binomial_ab.rs`. `eval_ab` compares
//! two evaluators inside one build and cannot see this change, so both arms
//! are compiled here: the old evaluator from the test oracle
//! `tests/support/recursive_bspline.rs`, the new one from the library. Both
//! arms run in one process, interleaved per rep with the order alternating by
//! rep parity, over the same pseudo-random times; the median of 9 reps is
//! reported. It times the forcing evaluation with `f64` coefficients only, not
//! a simulation, so the ratio is an upper bound on any end-to-end effect.

#[path = "../tests/support/recursive_bspline.rs"]
mod recursive_bspline;

use std::hint::black_box;
use std::time::Instant;

use recursive_bspline::eval_periodic_bspline_recursive;
use sim::periodic_bspline::PeriodicBasis;

const CALLS: usize = 200_000;
const REPS: usize = 9;

/// Deterministic times over several periods either side of zero.
fn times(period: f64) -> Vec<f64> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..CALLS)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            period * (-3.0 + 7.0 * (s >> 11) as f64 / (1u64 << 53) as f64)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    println!("case\tn_basis\tdegree\told_ns\tnew_ns\tspeedup\tmax_abs_diff");
    for (period, nb, p) in [(365.25, 6, 3), (365.25, 12, 3), (7.0, 8, 2), (365.25, 20, 5)] {
        let c: Vec<f64> = (0..nb).map(|i| 0.3 + ((i * 7 + 3) % 11) as f64 / 5.0).collect();
        let ts = times(period);
        let basis = PeriodicBasis::new(period, nb, p).unwrap();
        let max_diff = ts
            .iter()
            .map(|&t| (basis.eval(t, |k| c[k]) - eval_periodic_bspline_recursive(t, period, nb as u32, p as u32, &c)).abs())
            .fold(0.0_f64, f64::max);

        let old_arm = || {
            let start = Instant::now();
            let mut acc = 0.0;
            for &t in &ts {
                acc += eval_periodic_bspline_recursive(black_box(t), period, nb as u32, p as u32, black_box(&c));
            }
            black_box(acc);
            start.elapsed().as_secs_f64() * 1e9 / CALLS as f64
        };
        let new_arm = || {
            let start = Instant::now();
            let mut acc = 0.0;
            for &t in &ts {
                let c = black_box(&c);
                acc += basis.eval(black_box(t), |k| c[k]);
            }
            black_box(acc);
            start.elapsed().as_secs_f64() * 1e9 / CALLS as f64
        };
        let (mut old, mut new) = (Vec::new(), Vec::new());
        for rep in 0..REPS {
            if rep % 2 == 0 {
                old.push(old_arm());
                new.push(new_arm());
            } else {
                new.push(new_arm());
                old.push(old_arm());
            }
        }
        let (o, n) = (median(old), median(new));
        println!("periodic\t{nb}\t{p}\t{o:.1}\t{n:.1}\t{:.1}x\t{max_diff:.1e}", o / n);
    }
}
