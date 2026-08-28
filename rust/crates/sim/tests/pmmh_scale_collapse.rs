//! Reproduction of the reported PMMH proposal-scale collapse.
//!
//! No particle filter, no model: `run_pmmh` takes the likelihood as a closure,
//! so the adaptation can be driven by a synthetic target whose exact posterior
//! is known. The target on the transformed scale is a standard normal in `D`
//! dimensions (`Transform::None` + `Density::Flat` ⇒ posterior = likelihood),
//! for which the optimal random-walk proposal SD is 2.38/√D and the optimal
//! acceptance rate is known analytically.
//!
//! Two arms that differ in exactly one thing:
//!
//!   * `sigma = 0` — the likelihood is evaluated exactly. This is an ordinary
//!     adaptive Metropolis run and is the control.
//!   * `sigma > 0` — the likelihood is replaced by an unbiased noisy estimate,
//!     `log L̂ = log L + σZ − σ²/2` with `Z ~ N(0,1)` drawn independently at
//!     every evaluation. `E[L̂] = L`, so this is a valid pseudo-marginal
//!     sampler and is the standard model of particle-filter likelihood noise
//!     (Pitt, Silva, Giordani & Kohn 2012, *J. Econometrics* 171(2):134–151;
//!     Doucet, Pitt, Deligiannidis & Kohn 2015, *Biometrika* 102(2):295–313).
//!
//! Everything else — seed, dimension, initial proposal SD, `adapt_start`,
//! iteration count — is identical between the arms.
//!
//! Reported statistic matches the field report: per 250-iteration block, the
//! acceptance rate and the median absolute accepted step in the first
//! coordinate, plus the two multiplicative pieces of the proposal scale that
//! `pmmh.rs` composes (the Robbins–Monro scalar λ and the Haario Cholesky
//! diagonal), read out of the serialised `AdaptiveProposal`.

use std::cell::RefCell;

use sim::error::SimError;
use sim::inference::{
    if2::{EstimatedParam, Transform},
    pmmh::{run_pmmh, PMMHConfig, Prior},
};
use sim::rng::StatefulRng;

const D: usize = 6;
const BLOCK: usize = 250;
const N_STEPS: usize = 5000;
const SEED: u64 = 20260827;

fn params() -> Vec<EstimatedParam> {
    (0..D)
        .map(|i| EstimatedParam {
            name: format!("p{i}"),
            index: i,
            initial: 0.0,
            rw_sd: 1.0,
            transform: Transform::None,
            lower: -50.0,
            upper: 50.0,
            rw_sd_auto: false,
            perturb_only_at_t0: false,
        })
        .collect()
}

/// Exact standard-normal log-density (unnormalised) in D dimensions.
fn exact_ll(p: &[f64]) -> f64 {
    -0.5 * p.iter().map(|x| x * x).sum::<f64>()
}

/// One recorded step of a run.
#[derive(Clone, Copy)]
struct Rec {
    accepted: bool,
    p0: f64,
}

/// State of the adaptation at the end of a run: the Robbins–Monro scalar λ
/// and the first diagonal entry of the Haario Cholesky factor L.
struct AdaptState {
    lambda: f64,
    chol00: f64,
    chol_valid: bool,
}

/// Run `n_steps` of PMMH against the synthetic target and return the per-step
/// record plus the end-of-run adaptation state.
///
/// The RNG is advanced only inside the sampling loop, so a run of length `n`
/// is a strict prefix of a run of length `m > n` with the same seed. Calling
/// this at successive block boundaries therefore reads λ and L along one
/// single chain, not across independent chains.
fn run(n_steps: usize, sigma: f64, init_sd: f64) -> (Vec<Rec>, AdaptState) {
    let if2_params = params();
    let priors: Vec<Prior> =
        (0..D).map(|_| Prior::Fixed(sim::inference::prior::Density::Flat)).collect();
    let base_params = vec![0.0f64; D];

    let eval_loglik = move |p: &[f64], seed: u64| -> Result<f64, SimError> {
        let exact = exact_ll(p);
        if sigma == 0.0 {
            Ok(exact)
        } else {
            // Unbiased on the natural scale: E[exp(σZ − σ²/2)] = 1.
            let mut r = StatefulRng::new(seed ^ 0xABCD_EF01_2345_6789);
            Ok(exact + sigma * r.normal() - 0.5 * sigma * sigma)
        }
    };

    let config = PMMHConfig {
        n_steps,
        n_particles: 0,
        dt: 1.0,
        // Unused here: the likelihood is a closure, not a particle filter, so
        // there is no observation grid to size correlated noise against.
        t_start: 0.0,
        proposal_sd: vec![init_sd; D],
        adapt: true,
        adapt_start: 300, // the shipped default (`default_pmmh_adapt_start`)
        thin: 1,
        burn_in: 0,
        rho: None,
        n_source_groups: 0,
    };

    let recs: RefCell<Vec<Rec>> = RefCell::new(Vec::with_capacity(n_steps));
    let on_step = |_step: usize, _ll: f64, accepted: bool, p: &[f64]| {
        recs.borrow_mut().push(Rec { accepted, p0: p[0] });
    };

    let result = run_pmmh(
        &if2_params,
        &priors,
        &base_params,
        &[],
        &config,
        &[],
        &eval_loglik,
        None,
        SEED,
        Some(&on_step),
        None,
        String::new(),
    )
    .unwrap();

    // `AdaptiveProposal` derives Serialize; its fields are private, so read
    // them through serde rather than widening the API for a test.
    let v = serde_json::to_value(result.resume_state.adaptive.as_ref().unwrap()).unwrap();
    let adapt = AdaptState {
        lambda: v["log_scale"].as_f64().unwrap().exp(),
        chol00: v["chol"][0].as_f64().unwrap(),
        chol_valid: v["chol_valid"].as_bool().unwrap(),
    };

    (recs.into_inner(), adapt)
}

/// Acceptance rate and median absolute accepted step in coordinate 0 over
/// `recs[lo..hi]`. Mirrors the field report's statistic exactly: the step is
/// the change in the *current* state, which is non-zero only on acceptance.
fn block_stats(recs: &[Rec], lo: usize, hi: usize) -> (f64, f64) {
    let n = hi - lo;
    let n_acc = recs[lo..hi].iter().filter(|r| r.accepted).count();
    let mut steps: Vec<f64> = (lo..hi)
        .filter(|&t| recs[t].accepted)
        .map(|t| (recs[t].p0 - recs[t - 1].p0).abs())
        .collect();
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if steps.is_empty() { f64::NAN } else { steps[steps.len() / 2] };
    (n_acc as f64 / n as f64, median)
}

/// Print the per-block table for one arm and return the block medians.
fn table(label: &str, sigma: f64, init_sd: f64) -> Vec<f64> {
    // Target acceptance the Robbins–Monro scalar is chasing (pmmh.rs
    // `target_accept`): 0.234 + 0.206/d.
    let target = 0.234 + 0.206 / D as f64;
    println!("\n=== {label}  (sigma={sigma}, init_sd={init_sd}, d={D}, target a*={target:.3}) ===");
    println!(
        "{:>10} {:>8} {:>12} {:>8} {:>12} {:>12} {:>12}",
        "block", "acc%", "med|step|", "ratio", "lambda", "chol[0][0]", "lambda*chol"
    );

    let mut medians = Vec::new();
    let mut prev_med = f64::NAN;
    let n_blocks = N_STEPS / BLOCK;
    for b in 0..n_blocks {
        let end = (b + 1) * BLOCK;
        let (recs, adapt) = run(end, sigma, init_sd);
        let lo = b * BLOCK;
        // Block 0 starts at index 1 so `recs[t-1]` is in range.
        let lo = if lo == 0 { 1 } else { lo };
        let (acc, med) = block_stats(&recs, lo, end);
        let ratio = if prev_med.is_nan() { f64::NAN } else { med / prev_med };
        let chol = if adapt.chol_valid { adapt.chol00 } else { init_sd };
        println!(
            "{:>4}-{:<5} {:>7.1}% {:>12.6} {:>8.3} {:>12.3e} {:>12.3e} {:>12.3e}",
            lo,
            end,
            acc * 100.0,
            med,
            ratio,
            adapt.lambda,
            chol,
            adapt.lambda * chol
        );
        medians.push(med);
        prev_med = med;
    }
    medians
}

/// Control arm: an exact likelihood. The adaptation must find and hold a
/// proposal scale near the theoretical optimum; the scale must not collapse.
#[test]
fn control_exact_likelihood_scale_does_not_collapse() {
    let medians = table("CONTROL: exact likelihood", 0.0, 3.0);
    let first = medians[1]; // block 250-500, after the initial correction
    let last = *medians.last().unwrap();
    println!(
        "control: first/last median accepted step = {first:.6} / {last:.6}  (ratio {:.3})",
        last / first
    );
    assert!(
        last > 0.2 * first,
        "control arm should hold its scale, but the median accepted step fell \
         from {first:.6} to {last:.6}"
    );
}

/// Reproduction arm: the same run with an unbiased *noisy* likelihood, the
/// only change. `sigma = 2` is a realistic particle-filter noise level for a
/// long national series (Doucet et al. 2015 recommend tuning the particle
/// count to sigma ≈ 1.0–1.7 *at the mode*; sigma is larger in the tails, which
/// is where a chain that has drifted spends its time).
#[test]
fn noisy_likelihood_collapses_proposal_scale() {
    let medians = table("REPRO: noisy likelihood (sigma=2)", 2.0, 3.0);
    let first = medians[1];
    let last = *medians.last().unwrap();
    println!(
        "repro: first/last median accepted step = {first:.6} / {last:.6}  (ratio {:.3e})",
        last / first
    );
    assert!(
        last > 0.2 * first,
        "PROPOSAL SCALE COLLAPSE: median accepted step fell from {first:.6} to \
         {last:.6} over the run"
    );
}

/// The collapse is not an artifact of a bad starting scale: start at the
/// theoretically optimal proposal SD 2.38/√d and it still collapses.
#[test]
fn noisy_likelihood_collapses_from_an_optimal_start() {
    let opt = 2.38 / (D as f64).sqrt();
    let medians = table("REPRO: noisy, optimal start", 2.0, opt);
    let first = medians[1];
    let last = *medians.last().unwrap();
    println!(
        "repro-opt: first/last median accepted step = {first:.6} / {last:.6}  (ratio {:.3e})",
        last / first
    );
    assert!(
        last > 0.2 * first,
        "PROPOSAL SCALE COLLAPSE from an optimal start: median accepted step \
         fell from {first:.6} to {last:.6}"
    );
}

/// The requested statement, made as an assertion on the real `run_pmmh` path:
/// find the 250-iteration blocks whose acceptance rate is **at or above** the
/// Robbins–Monro target, and check whether the proposal scale grew in them.
///
/// A correct adaptation cannot shrink the proposal in a block where it is
/// accepting more often than it wants to. Reported separately for the two
/// multiplicative pieces, because they behave differently: the Robbins–Monro
/// scalar λ does respond correctly to the acceptance signal, and the Haario
/// shape term ignores it entirely.
#[test]
fn scale_falls_in_blocks_where_acceptance_is_at_or_above_target() {
    let target = 0.234 + 0.206 / D as f64;
    let n_blocks = N_STEPS / BLOCK;

    let mut acc = Vec::new();
    let mut lambda = Vec::new();
    let mut chol = Vec::new();
    for b in 0..n_blocks {
        let end = (b + 1) * BLOCK;
        let (recs, adapt) = run(end, 2.0, 3.0);
        let lo = if b == 0 { 1 } else { b * BLOCK };
        acc.push(block_stats(&recs, lo, end).0);
        lambda.push(adapt.lambda);
        chol.push(adapt.chol00);
    }

    // The Haario shape term is a one-way ratchet: check every block, at every
    // acceptance rate.
    let shape_grew = (1..n_blocks).filter(|&b| chol[b] > chol[b - 1]).count();
    println!(
        "\nHaario shape term grew in {shape_grew} of {} blocks (acceptance ranged {:.1}%-{:.1}%)",
        n_blocks - 1,
        acc.iter().cloned().fold(f64::INFINITY, f64::min) * 100.0,
        acc.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 100.0,
    );

    // Blocks where the sampler is accepting at or above what it is aiming for.
    let mut above: Vec<usize> = Vec::new();
    for b in 1..n_blocks {
        if acc[b] >= target {
            above.push(b);
        }
    }
    println!("blocks at or above target a* = {target:.3}:");
    for &b in &above {
        println!(
            "  {:>4}-{:<5}  acc {:>5.1}%   lambda {:.3e} -> {:.3e} ({:+.1}%)   \
             chol {:.3e} -> {:.3e} ({:+.1}%)",
            b * BLOCK,
            (b + 1) * BLOCK,
            acc[b] * 100.0,
            lambda[b - 1],
            lambda[b],
            (lambda[b] / lambda[b - 1] - 1.0) * 100.0,
            chol[b - 1],
            chol[b],
            (chol[b] / chol[b - 1] - 1.0) * 100.0,
        );
    }

    assert_eq!(
        shape_grew,
        n_blocks - 1,
        "RATCHET: the Haario shape term failed to grow in {} of {} blocks, \
         including blocks accepting above the {target:.3} target — it does not \
         read the acceptance signal at all",
        n_blocks - 1 - shape_grew,
        n_blocks - 1,
    );
}

/// Question 5: the sustained-0% endpoint. As the likelihood noise grows, the
/// pseudo-marginal chain spends longer sojourns pinned at a state whose
/// likelihood estimate came out high by chance; the acceptance rate averaged
/// over a block falls further below the Robbins–Monro target, so λ shrinks
/// faster, so the chain is even less able to escape. Scan sigma and report the
/// acceptance rate and proposal scale in the final block.
#[test]
fn noise_level_scan_reaches_sustained_zero_acceptance() {
    println!(
        "\n{:>7} {:>14} {:>14} {:>12} {:>12}",
        "sigma", "acc% first 250", "acc% last 250", "lambda end", "chol end"
    );
    let mut zero_at = None;
    for &sigma in &[0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0] {
        let (recs, adapt) = run(N_STEPS, sigma, 3.0);
        let (acc_first, _) = block_stats(&recs, 1, BLOCK);
        let (acc_last, _) = block_stats(&recs, N_STEPS - BLOCK, N_STEPS);
        println!(
            "{sigma:>7.1} {:>13.1}% {:>13.1}% {:>12.3e} {:>12.3e}",
            acc_first * 100.0,
            acc_last * 100.0,
            adapt.lambda,
            adapt.chol00
        );
        if acc_last == 0.0 && zero_at.is_none() {
            zero_at = Some(sigma);
        }
    }
    assert!(
        zero_at.is_none(),
        "SUSTAINED 0% ACCEPTANCE reached at sigma = {:?}: the chain took no \
         accepted move in its final 250 iterations",
        zero_at.unwrap()
    );
}
