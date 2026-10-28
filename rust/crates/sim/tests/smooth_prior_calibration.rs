//! The committed calibration of the `smooth` scale prior (proposal
//! 2026-10-27-estimable-bsplines §6.5), reproduced by camdl's own code.
//!
//! `tests/recovery/cases/spline_seir/smooth_prior_calibration.tsv` is written
//! by `make -f tests/recovery/Makefile calibrate-smooth-prior` from scipy's
//! B-splines and numpy's QR and Cholesky. This test redraws the U = 0.5 rows
//! with `sim::spline_penalty`'s whitening, `sim::bspline`'s basis and an
//! independent random stream, and requires (a) the committed table to state
//! λ = −ln(0.05)/0.5 = 5.99 and to meet the §6.5 target on every knot set,
//! and (b) camdl's bands to agree with it within Monte Carlo error.
//!
//! The tolerance: the 2.5% quantile of log R from n draws has standard error
//! about √(q(1 − q)/n)/f(x_q) ≈ 0.045 at n = 4000 (the table's draws, log R's
//! prior SD ≈ 1.1) and ≈ 0.020 at n = 20000 (here); five combined standard
//! errors is 0.25 in log R. Two implementations of one formula, not two
//! measurements: agreement is evidence about the code, not about the prior.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Exp, Normal};
use sim::bspline::{span, Knots};
use sim::spline_penalty::{kappa, Whitening};

const DRAWS: usize = 20_000;
const U: f64 = 0.5;
const GAMMA: f64 = 0.125;
const T_REF: f64 = 7.0;
const A_SD: f64 = 0.6;
const B_SD: f64 = 0.005;
const REPORT_T: [f64; 6] = [0.0, 42.0, 84.0, 126.0, 168.0, 182.0];
/// Five combined Monte Carlo standard errors of a 2.5% / 97.5% quantile of
/// log R (module doc).
const LOG_R_TOL: f64 = 0.25;

fn knot_sets() -> Vec<(&'static str, Knots)> {
    vec![
        ("K8_uniform", Knots::uniform(0.0, 168.0, 5).unwrap()),
        ("K12_uniform", Knots::uniform(0.0, 168.0, 9).unwrap()),
        ("K20_uniform", Knots::uniform(0.0, 168.0, 17).unwrap()),
        ("K12_irregular", Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap()),
    ]
}

/// (knots, t or "all") → (λ, R_lo, R_hi, meets) for the U = 0.5 rows.
fn committed() -> std::collections::HashMap<(String, String), (f64, f64, f64, bool)> {
    let p = format!("{}/../../../tests/recovery/cases/spline_seir/smooth_prior_calibration.tsv", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(p).unwrap();
    let mut out = std::collections::HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')).skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        if f[1].parse::<f64>().unwrap() != U {
            continue;
        }
        let num = |i: usize| f[i].parse::<f64>().unwrap();
        out.insert((f[0].to_string(), f[3].to_string()), (num(2), num(4), num(5), f[6] == "1"));
    }
    out
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// The 95% prior band of log R(t) at every integer t in [0, 182].
fn bands(knots: &Knots, lambda: f64, rng: &mut ChaCha8Rng) -> Vec<(f64, f64, f64)> {
    let w = Whitening::new(knots, 3).unwrap();
    let (k, d) = (w.n_basis(), w.n_modes());
    let kap = kappa(k, T_REF);
    let a_d = Normal::new((1.2 * GAMMA).ln(), A_SD).unwrap();
    let b_d = Normal::new(0.0, B_SD).unwrap();
    let tau_d = Exp::new(lambda).unwrap();
    let z_d = Normal::new(0.0, 1.0).unwrap();
    let coefs: Vec<Vec<f64>> = (0..DRAWS).map(|_| {
        let (a, b, tau) = (a_d.sample(rng), b_d.sample(rng), tau_d.sample(rng));
        let z: Vec<f64> = (0..d).map(|_| z_d.sample(rng)).collect();
        (0..k).map(|i| {
            let mz: f64 = (0..d).map(|r| w.m.at(i, r) * z[r]).sum();
            a + b * (w.xi[i] - w.t_mid) + tau / kap * mz
        }).collect()
    }).collect();
    (0..=182).map(|ti| {
        let t = ti as f64;
        let s = span(knots, 3, t);
        let mut v: Vec<f64> = coefs.iter()
            .map(|c| (0..=3).map(|r| s.values[r] * c[s.first + r]).sum::<f64>() - GAMMA.ln())
            .collect();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap());
        (t, quantile(&v, 0.025), quantile(&v, 0.975))
    }).collect()
}

#[test]
fn the_committed_calibration_states_lambda_and_meets_the_target() {
    let table = committed();
    let lambda = -(0.05_f64).ln() / U;
    for (name, _) in knot_sets() {
        let (lam, lo, hi, meets) = table[&(name.to_string(), "all".to_string())];
        assert!((lam - lambda).abs() < 1e-12 && (lam * 100.0).round() / 100.0 == 5.99, "{name}: λ = {lam}");
        assert!(meets && lo >= 0.1 && hi <= 20.0, "{name}: U = 0.5 must meet the target, band [{lo}, {hi}]");
    }
}

#[test]
fn camdls_whitening_reproduces_the_committed_calibration() {
    let table = committed();
    let lambda = -(0.05_f64).ln() / U;
    let mut rng = ChaCha8Rng::seed_from_u64(0x5000_7a0b);
    let mut worst: f64 = 0.0;
    for (name, knots) in knot_sets() {
        let b = bands(&knots, lambda, &mut rng);
        let check = |key: &str, lo: f64, hi: f64, worst: &mut f64| {
            let (_, r_lo, r_hi, _) = table[&(name.to_string(), key.to_string())];
            let (e_lo, e_hi) = ((lo - r_lo.ln()).abs(), (hi - r_hi.ln()).abs());
            *worst = worst.max(e_lo).max(e_hi);
            assert!(e_lo <= LOG_R_TOL && e_hi <= LOG_R_TOL,
                "{name} t = {key}: camdl band [{:.3}, {:.3}] vs committed [{r_lo:.3}, {r_hi:.3}]", lo.exp(), hi.exp());
        };
        for &t in &REPORT_T {
            let &(_, lo, hi) = b.iter().find(|(tt, _, _)| *tt == t).unwrap();
            check(&format!("{t:?}"), lo, hi, &mut worst);
        }
        let lo = b.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        let hi = b.iter().map(|x| x.2).fold(f64::NEG_INFINITY, f64::max);
        check("all", lo, hi, &mut worst);
        // camdl's own band meets the §6.5 target at every t.
        assert!(b.iter().all(|&(_, lo, hi)| lo.exp() <= 0.5 && hi.exp() >= 3.0 && lo.exp() >= 0.1 && hi.exp() <= 20.0),
            "{name}: camdl's U = 0.5 band misses the target");
    }
    eprintln!("worst |log band − committed| = {worst:.3} (tolerance {LOG_R_TOL})");
}
