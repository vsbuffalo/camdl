//! Does one `nuts_step` leave its target invariant?
//!
//! Runs the NUTS kernel alone on targets whose moments are known in closed
//! form, at a fixed step size and a fixed mass matrix (no adaptation), so the
//! chain is a plain time-homogeneous Markov chain and its long-run averages
//! must match the exact moments. Any departure is non-stationarity in the
//! kernel itself, separated from every PGAS and ODE component.
//!
//! Targets (z is the sampler's coordinate):
//! - `gauss1`, `gauss2`, `gauss5`: z ~ N(0, I_d).
//! - `beta22_gauss`: z₁ = logit θ with θ ~ Beta(2, 2), so
//!   log p(z₁) = 2 log σ(z₁) + 2 log(1 − σ(z₁)) (prior + log-Jacobian),
//!   E[θ] = 1/2, Var[θ] = 1/20; z₂ ~ N(0, 1) independent.
//! - `corr2`: z ~ N(0, Σ) with unit variances and correlation 0.9, sampled
//!   under three mass matrices: the true dense covariance, a deliberately
//!   wrong dense one, and a diagonal one unequal to the variances. A wrong
//!   mass matrix leaves the target unchanged but makes every metric operation
//!   (momentum draw, kinetic energy, M⁻¹p in the leapfrog and the U-turn
//!   check) do real work, so an error in any of them shows up as a moment
//!   error.
//!
//! The statistics include second moments and cross-moments: a kernel can get
//! every mean right and still have the wrong spread or correlation.
//!
//! Each row is scored as a z-score on a batch-means Monte-Carlo standard error
//! (MCSE); the test fails if any |z| ≥ 5. With 200,000 draws that resolves a
//! bias of roughly 2% in a variance. The seed is fixed, so the test is
//! deterministic.

use sim::inference::nuts::{nuts_step, MassMatrix, NUTSConfig};
use sim::rng::StatefulRng;

const N_DRAWS: usize = 200_000;
const N_BURN: usize = 1_000;
const Z_FAIL: f64 = 5.0;

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Batch-means MCSE of the mean of `x`.
fn mcse(x: &[f64]) -> f64 {
    let n_batches = 200usize;
    let b = x.len() / n_batches;
    let means: Vec<f64> = (0..n_batches)
        .map(|k| x[k * b..(k + 1) * b].iter().sum::<f64>() / b as f64)
        .collect();
    let m = means.iter().sum::<f64>() / n_batches as f64;
    let v = means.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (n_batches - 1) as f64;
    (v / n_batches as f64).sqrt()
}

type Stat<'a> = (&'a str, &'a dyn Fn(&[f64]) -> f64, f64);

/// Runs the kernel and returns the largest |z| over `stats`.
fn run(
    label: &str,
    step: f64,
    mass: MassMatrix,
    lp: &dyn Fn(&[f64]) -> (f64, Vec<f64>),
    stats: &[Stat],
    d: usize,
) -> f64 {
    let cfg = NUTSConfig { max_tree_depth: 10, step_size: step, mass_matrix: mass };
    let mut rng = StatefulRng::new(20260923);
    let mut z = vec![0.1; d];
    let (mut l, mut g) = lp(&z);
    let mut traces: Vec<Vec<f64>> = vec![Vec::with_capacity(N_DRAWS); stats.len()];
    for it in 0..(N_DRAWS + N_BURN) {
        let r = nuts_step(&z, l, &g, &cfg, lp, &mut rng);
        z = r.params;
        (l, g) = lp(&z);
        if it >= N_BURN {
            for (k, (_, f, _)) in stats.iter().enumerate() {
                traces[k].push(f(&z));
            }
        }
    }
    let mut worst = 0.0f64;
    for (k, (name, _, exact)) in stats.iter().enumerate() {
        let x = &traces[k];
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let se = mcse(x);
        let zsc = (m - exact) / se;
        worst = worst.max(zsc.abs());
        eprintln!("{label:24} step={step}: {name:10} exact {exact:+.5} est {m:+.5} mcse {se:.5} z {zsc:+.2}");
    }
    worst
}

/// N(0, I_d).
fn std_normal(z: &[f64]) -> (f64, Vec<f64>) {
    (-0.5 * z.iter().map(|x| x * x).sum::<f64>(), z.iter().map(|x| -x).collect())
}

/// N(0, Σ) in 2-D with unit variances and correlation `rho`.
fn corr_normal(rho: f64) -> impl Fn(&[f64]) -> (f64, Vec<f64>) {
    move |z: &[f64]| {
        // Σ⁻¹ = [[1, −ρ], [−ρ, 1]] / (1 − ρ²)
        let c = 1.0 / (1.0 - rho * rho);
        let g0 = -c * (z[0] - rho * z[1]);
        let g1 = -c * (z[1] - rho * z[0]);
        (0.5 * (z[0] * g0 + z[1] * g1), vec![g0, g1])
    }
}

/// One-dimensional targets are included as controls: the gh#956 defect (a
/// subtree U-turn checked from a point outside the subtree) left them exact
/// while biasing every multi-dimensional row by ≈10–11% in variance.
#[test]
#[ignore = "red until gh#956: a NUTS subtree's U-turn is checked from a point outside the subtree"]
fn nuts_kernel_is_stationary_on_known_targets() {
    let mut worst = 0.0f64;

    for step in [0.4, 0.9] {
        worst = worst.max(run("gauss1", step, MassMatrix::identity(1), &std_normal, &[
            ("E[z]", &|z| z[0], 0.0),
            ("E[z^2]", &|z| z[0] * z[0], 1.0),
        ], 1));
        worst = worst.max(run("gauss2", step, MassMatrix::identity(2), &std_normal, &[
            ("E[z1^2]", &|z| z[0] * z[0], 1.0),
            ("E[z2^2]", &|z| z[1] * z[1], 1.0),
            ("E[z1 z2]", &|z| z[0] * z[1], 0.0),
        ], 2));
    }

    worst = worst.max(run("gauss5", 0.5, MassMatrix::identity(5), &std_normal, &[
        ("E[z1^2]", &|z| z[0] * z[0], 1.0),
        ("E[z3^2]", &|z| z[2] * z[2], 1.0),
        ("E[z5^2]", &|z| z[4] * z[4], 1.0),
        ("E[z1 z5]", &|z| z[0] * z[4], 0.0),
    ], 5));

    let beta_gauss = |z: &[f64]| -> (f64, Vec<f64>) {
        let s = sigmoid(z[0]);
        let lp = 2.0 * s.ln() + 2.0 * (1.0 - s).ln() - 0.5 * z[1] * z[1];
        // d/dz [2 log σ + 2 log(1−σ)] = 2(1−σ) − 2σ
        (lp, vec![2.0 * (1.0 - s) - 2.0 * s, -z[1]])
    };
    for step in [0.4, 0.9] {
        worst = worst.max(run("beta22_gauss", step, MassMatrix::identity(2), &beta_gauss, &[
            ("E[θ]", &|z| sigmoid(z[0]), 0.5),
            ("E[(θ-½)²]", &|z| (sigmoid(z[0]) - 0.5).powi(2), 0.05),
            ("E[z2^2]", &|z| z[1] * z[1], 1.0),
        ], 2));
    }

    let rho = 0.9;
    let corr = corr_normal(rho);
    let corr_stats: [Stat; 3] = [
        ("E[z1^2]", &|z| z[0] * z[0], 1.0),
        ("E[z2^2]", &|z| z[1] * z[1], 1.0),
        ("E[z1 z2]", &|z| z[0] * z[1], rho),
    ];
    let true_cov = [1.0, rho, rho, 1.0];
    // Positive definite (det = 0.75) and far from Σ: a different scale per
    // axis and the wrong sign of correlation.
    let wrong_cov = [2.0, -0.5, -0.5, 0.5];
    worst = worst.max(run("corr2 dense(true Σ)", 0.5,
        MassMatrix::dense_from_covariance(&true_cov, 2), &corr, &corr_stats, 2));
    worst = worst.max(run("corr2 dense(wrong Σ)", 0.15,
        MassMatrix::dense_from_covariance(&wrong_cov, 2), &corr, &corr_stats, 2));
    worst = worst.max(run("corr2 diag(4, 0.25)", 0.1,
        MassMatrix::diagonal(vec![4.0, 0.25]), &corr, &corr_stats, 2));

    assert!(worst < Z_FAIL, "NUTS kernel is not stationary on a known target: worst |z| = {worst:.2}");
}
