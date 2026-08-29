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
//! `adapt_stop`, iteration count — is identical between the arms.
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
    pmmh::{run_pmmh, scale_is_far_from_optimum, PMMHConfig, PMMHResumeState, Prior},
};
use sim::rng::StatefulRng;

const D: usize = 6;
const BLOCK: usize = 250;
const N_STEPS: usize = 5000;
/// End of the warm-up window: adaptation runs over the first half of the run
/// and the transition kernel is frozen for the second. Half is a stand-in for
/// the shipped policy, where the CLI sets `adapt_stop` to the burn-in it
/// discards. A fixed absolute boundary keeps the prefix property `run` relies
/// on: freezing changes no RNG draw, so a run of `n` steps is still a strict
/// prefix of a longer one at the same seed.
const ADAPT_STOP: usize = N_STEPS / 2;
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
    /// Steps the Robbins–Monro scale spent pinned at its lower bound, as
    /// `run_pmmh` counted them for its end-of-run warning.
    steps_at_scale_floor: usize,
    /// The end-of-run scale as `run_pmmh` reports it on `PMMHResult`, read
    /// back independently of `lambda` (which comes from the serialised
    /// `AdaptiveProposal`) so the two can be checked against each other.
    final_scale: f64,
}

/// Run `n_steps` of PMMH against the synthetic target and return the per-step
/// record plus the end-of-run adaptation state.
///
/// The RNG is advanced only inside the sampling loop, so a run of length `n`
/// is a strict prefix of a run of length `m > n` with the same seed. Calling
/// this at successive block boundaries therefore reads λ and L along one
/// single chain, not across independent chains.
fn run(n_steps: usize, sigma: f64, init_sd: f64) -> (Vec<Rec>, AdaptState) {
    let (recs, adapt, _) = run_from(n_steps, sigma, init_sd, None);
    (recs, adapt)
}

/// `run`, plus the chain state a continuation would be handed. `resume_from`
/// starts the chain at the checkpoint's step count instead of at 0, which is
/// how a `--resume` run reaches the sampling phase without re-walking warm-up.
fn run_from(
    n_steps: usize,
    sigma: f64,
    init_sd: f64,
    resume_from: Option<PMMHResumeState>,
) -> (Vec<Rec>, AdaptState, PMMHResumeState) {
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
        adapt_stop: ADAPT_STOP,
        thin: 1,
        burn_in: 0,
        rho: None,
        n_source_groups: 0, init_noise_width: 0,
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
        resume_from,
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
        steps_at_scale_floor: result.steps_at_scale_floor,
        final_scale: result.final_scale,
    };

    (recs.into_inner(), adapt, result.resume_state)
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
    let medians = table("control: exact likelihood", 0.0, 3.0);
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

/// What a noise-aware target must deliver at `sigma = 2`: the proposal scale
/// holds, rather than falling away over the run. `sigma = 2` is a realistic
/// particle-filter noise level for a long national series (Doucet et al. 2015
/// recommend tuning the particle count to sigma ≈ 1.0–1.7 *at the mode*; sigma
/// is larger in the tails, which is where a chain that has drifted spends its
/// time).
///
/// Parked, not skipped: the scale bound this branch adds cannot satisfy it. At
/// sigma = 2 the attainable acceptance is 2Φ(−sigma/√2) = 15.7%, below the
/// 26.8% target at d = 6, so the Robbins–Monro recursion has no root and no
/// bound on lambda puts the scale where this test asks. Measured directly:
/// forcing `target_accept` to 0.07 ends this run at lambda = 1.18 with Haario
/// diagonal 0.99 — no collapse at all — but the same target overshoots at
/// sigma = 0 and 1, which is why 0.07 cannot simply become the default.
#[test]
#[ignore = "specifies the noise-aware target acceptance (fix 2 of docs/dev/proposals/2026-08-28-pmmh-proposal-adaptation.md), which is not built. The scale bound cannot satisfy it: at these noise levels the target itself is unattainable, so no bound holds the scale where this asks."]
fn noise_aware_target_holds_the_scale_at_sigma_2() {
    let medians = table("noisy likelihood (sigma=2)", 2.0, 3.0);
    let first = medians[1];
    let last = *medians.last().unwrap();
    println!(
        "repro: first/last median accepted step = {first:.6} / {last:.6}  (ratio {:.3e})",
        last / first
    );
    assert!(
        last > 0.2 * first,
        "the proposal scale collapsed: median accepted step fell from \
         {first:.6} to {last:.6} over the run"
    );
}

/// The same requirement from the theoretically optimal starting proposal SD
/// 2.38/√d, which rules out "the run merely started in the wrong place" as an
/// explanation for the fall. Parked for the same reason as the test above.
#[test]
#[ignore = "specifies the noise-aware target acceptance (fix 2 of docs/dev/proposals/2026-08-28-pmmh-proposal-adaptation.md), which is not built. The scale bound cannot satisfy it: at these noise levels the target itself is unattainable, so no bound holds the scale where this asks."]
fn noise_aware_target_holds_the_scale_from_an_optimal_start() {
    let opt = 2.38 / (D as f64).sqrt();
    let medians = table("noisy likelihood, optimal start", 2.0, opt);
    let first = medians[1];
    let last = *medians.last().unwrap();
    println!(
        "repro-opt: first/last median accepted step = {first:.6} / {last:.6}  (ratio {:.3e})",
        last / first
    );
    assert!(
        last > 0.2 * first,
        "the proposal scale collapsed from an optimal start: median accepted \
         step fell from {first:.6} to {last:.6}"
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
///
/// Parked with the other three. This one is about the *shape* term rather than
/// λ, and the scale bound does not touch it: `update_cholesky`'s εI already
/// floors the shape SD at 1e-3 by construction, and what the shape term lacks
/// is any reference to acceptance at all, not a floor.
#[test]
#[ignore = "specifies the noise-aware target acceptance (fix 2 of docs/dev/proposals/2026-08-28-pmmh-proposal-adaptation.md), which is not built. The scale bound cannot satisfy it: at these noise levels the target itself is unattainable, so no bound holds the scale where this asks."]
fn noise_aware_target_grows_the_shape_term_in_every_block() {
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
        "the Haario shape term failed to grow in {} of {} blocks, including \
         blocks accepting above the {target:.3} target — it does not read the \
         acceptance signal at all",
        n_blocks - 1 - shape_grew,
        n_blocks - 1,
    );
}

/// The sustained-0% endpoint. As the likelihood noise grows, the
/// pseudo-marginal chain spends longer sojourns pinned at a state whose
/// likelihood estimate came out high by chance; the acceptance rate averaged
/// over a block falls further below the Robbins–Monro target, so λ shrinks
/// faster, so the chain is even less able to escape. Scan sigma and require
/// that no noise level leaves the chain taking no accepted move at all.
///
/// Parked with the other three: at sigma = 5 the attainable acceptance is
/// 4e-5 against a target of 0.2461 at d = 17, so this asks for behaviour no
/// bound on the scale can produce.
#[test]
#[ignore = "specifies the noise-aware target acceptance (fix 2 of docs/dev/proposals/2026-08-28-pmmh-proposal-adaptation.md), which is not built. The scale bound cannot satisfy it: at these noise levels the target itself is unattainable, so no bound holds the scale where this asks."]
fn noise_aware_target_avoids_sustained_zero_acceptance() {
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
        "sustained zero acceptance reached at sigma = {:?}: the chain took no \
         accepted move in its final 250 iterations",
        zero_at.unwrap()
    );
}

// ── What the scale bound delivers ──────────────────────────────────────────
//
// The four tests above specify a sampler that keeps exploring at these noise
// levels; the bound does not deliver that and cannot. What it does deliver is
// that the failure is bounded and announced instead of silent: the scale comes
// to rest on a floor rather than running to zero, and the run reports how long
// it sat there. Both are asserted here on the same synthetic target.

/// Without a bound the Robbins–Monro scale runs away: at sigma = 5 the
/// attainable acceptance is far below the target at every scale, so log lambda
/// drifts as −(a* − a)·T^0.4/0.4 and does not settle. With the bound it comes
/// to rest exactly on `LOG_SCALE_MIN` and stays there.
///
/// This is the difference between a proposal that is merely too narrow and one
/// that is numerically zero — at lambda = 5.9e-17 (measured before the bound)
/// every proposed move is identical to the current state.
#[test]
fn the_scale_comes_to_rest_on_its_bound_rather_than_running_to_zero() {
    let (_, adapt) = run(N_STEPS, 5.0, 3.0);
    let floor = sim::inference::pmmh::LOG_SCALE_MIN.exp();
    println!(
        "sigma=5: lambda = {:.3e}, bound = {floor:.3e}, steps at floor = {}",
        adapt.lambda, adapt.steps_at_scale_floor
    );
    assert!(
        adapt.lambda >= floor * (1.0 - 1e-9),
        "lambda must not fall below the bound: got {:.3e} against {floor:.3e}",
        adapt.lambda,
    );
    assert!(
        (adapt.lambda - floor).abs() < floor * 1e-9,
        "at sigma = 5 the adaptation should be pinned to the bound, not resting \
         above it: lambda = {:.3e}, bound = {floor:.3e}",
        adapt.lambda,
    );
}

/// A run that spent time on the floor did not explore its posterior, so the
/// condition has to leave the sampler as data rather than only as a line on
/// stderr — a wrapper that captures stdout, or a log nobody reads, would
/// otherwise turn a run-invalidating condition into a normal-looking result.
///
/// The control arm pins the other half: a chain that can reach its target never
/// touches the floor, so a non-zero count means what it says.
#[test]
fn a_run_pinned_at_the_bound_reports_how_long_it_sat_there() {
    let (_, noisy) = run(N_STEPS, 5.0, 3.0);
    assert!(
        noisy.steps_at_scale_floor > 0,
        "a sigma = 5 run ends pinned at the bound, so the floor-step count must \
         be non-zero; got {}",
        noisy.steps_at_scale_floor,
    );

    let (_, exact) = run(N_STEPS, 0.0, 3.0);
    println!(
        "steps at floor: sigma=5 {} of {N_STEPS}, sigma=0 {}",
        noisy.steps_at_scale_floor, exact.steps_at_scale_floor
    );
    assert_eq!(
        exact.steps_at_scale_floor, 0,
        "an exact likelihood reaches its target acceptance, so the bound must \
         never bind — this is the regime the deterministic `mh` sampler runs in",
    );
}

// ── What freezing at the end of warm-up delivers ───────────────────────────
//
// `adapt_scale` used to run from step 0 to the last step of the chain, so the
// drift it accumulated was bounded by the run length rather than by any
// adaptation budget, and the draws a run kept were produced while the proposal
// was still shrinking under them. Adaptation now stops at `adapt_stop`.

/// Both adapting quantities stop at the boundary, exactly: the run's last
/// 2,500 steps leave λ and the Haario factor bit-identical to what the first
/// 2,500 produced.
///
/// σ = 2 is the arm that makes this visible. It sits below the noise level at
/// which λ reaches its `LOG_SCALE_MIN` floor within the run (σ = 5 gets there
/// by step ~1,400, after which λ cannot move anyway), so an unfrozen chain
/// keeps driving λ down for the whole run: 4.788e-4 at step 2,500 against
/// 1.364e-4 at step 5,000, measured before this change.
#[test]
fn lambda_and_the_shape_term_stop_moving_at_the_end_of_warm_up() {
    let (_, at_boundary) = run(ADAPT_STOP, 2.0, 3.0);
    let (_, at_end) = run(N_STEPS, 2.0, 3.0);
    println!(
        "sigma=2: lambda {:.6e} at step {ADAPT_STOP} -> {:.6e} at step \
         {N_STEPS}; chol[0][0] {:.6e} -> {:.6e}",
        at_boundary.lambda, at_end.lambda, at_boundary.chol00, at_end.chol00,
    );
    assert_eq!(
        at_end.lambda.to_bits(),
        at_boundary.lambda.to_bits(),
        "the Robbins-Monro scale must not move after the warm-up boundary: \
         {:.6e} at step {ADAPT_STOP} became {:.6e} by step {N_STEPS}",
        at_boundary.lambda,
        at_end.lambda,
    );
    assert_eq!(
        at_end.chol00.to_bits(),
        at_boundary.chol00.to_bits(),
        "the Haario shape term must not move after the warm-up boundary: \
         {:.6e} at step {ADAPT_STOP} became {:.6e} by step {N_STEPS}",
        at_boundary.chol00,
        at_end.chol00,
    );
}

/// The boundary is an absolute step index, so a chain resumed from a
/// checkpoint past it is already in its sampling phase and adapts no further.
/// A resumed chain that kept adapting would be the same defect wearing a
/// different hat, and would additionally make the kernel a run samples under
/// depend on how many times the run was interrupted.
#[test]
fn a_chain_resumed_past_its_warm_up_does_not_adapt() {
    let (_, at_boundary, checkpoint) = run_from(ADAPT_STOP, 2.0, 3.0, None);
    let (_, after_resume, _) = run_from(N_STEPS, 2.0, 3.0, Some(checkpoint));
    println!(
        "resumed {ADAPT_STOP} -> {N_STEPS}: lambda {:.6e} -> {:.6e}, \
         chol[0][0] {:.6e} -> {:.6e}",
        at_boundary.lambda, after_resume.lambda, at_boundary.chol00, after_resume.chol00,
    );
    assert_eq!(
        after_resume.lambda.to_bits(),
        at_boundary.lambda.to_bits(),
        "a chain resumed at step {ADAPT_STOP} is past its warm-up, so lambda \
         must stay at {:.6e}; it moved to {:.6e}",
        at_boundary.lambda,
        after_resume.lambda,
    );
    assert_eq!(
        after_resume.chol00.to_bits(),
        at_boundary.chol00.to_bits(),
        "the shape term must be frozen across a resume too: {:.6e} became {:.6e}",
        at_boundary.chol00,
        after_resume.chol00,
    );
}

/// The scale a run finished with is carried on `PMMHResult` and flagged when
/// it is far from λ = 1 — the case the floor warning cannot see.
///
/// σ = 2 ends nowhere near the `LOG_SCALE_MIN` floor, so `steps_at_scale_floor`
/// is 0 and that warning stays silent, while the chain samples with a proposal
/// orders of magnitude narrower than the covariance it scales. That is the
/// silent failure: a run that completes, writes draws, and reports nothing
/// worse than a poor R-hat.
#[test]
fn the_end_of_run_scale_is_reported_and_flagged_when_far_from_one() {
    let (_, noisy) = run(N_STEPS, 2.0, 3.0);
    println!(
        "sigma=2: final_scale = {:.6e} (checkpointed lambda {:.6e}), steps at \
         floor {}",
        noisy.final_scale, noisy.lambda, noisy.steps_at_scale_floor,
    );
    assert_eq!(
        noisy.final_scale.to_bits(),
        noisy.lambda.to_bits(),
        "the reported scale must be the sampler's own: the result says {:.6e}, \
         the checkpointed AdaptiveProposal says {:.6e}",
        noisy.final_scale,
        noisy.lambda,
    );
    assert_eq!(
        noisy.steps_at_scale_floor, 0,
        "this arm never reaches the floor — that is what makes it the case the \
         floor warning misses",
    );
    assert!(
        scale_is_far_from_optimum(noisy.final_scale),
        "a run that ends at lambda = {:.6e} sampled far more narrowly than its \
         own covariance estimate and must be flagged",
        noisy.final_scale,
    );

    let (_, exact) = run(N_STEPS, 0.0, 3.0);
    println!("sigma=0: final_scale = {:.6e}", exact.final_scale);
    assert!(
        !scale_is_far_from_optimum(exact.final_scale),
        "an exact likelihood reaches its target acceptance and ends near \
         lambda = 1, so it must not be flagged; got {:.6e}",
        exact.final_scale,
    );
}
