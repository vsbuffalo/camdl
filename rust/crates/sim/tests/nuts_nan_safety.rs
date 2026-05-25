//! gh#81 Phase 2 — NUTS-side safety net.
//!
//! Phase 1 diagnosis traced the WA fit failure to a NUTS leapfrog step
//! that produced a NaN parameter, which was then committed to the chain
//! because:
//!
//! (a) build_tree's divergence detector `(h_new - h0).abs() > delta_max`
//!     evaluates to `false` when `h_new` is NaN (all NaN comparisons
//!     return false), so a NaN-energy proposal is NOT flagged divergent.
//!
//! (b) `nuts_step`'s accept criterion `z_proposal != current_z` is
//!     `true` even when `z_proposal` contains NaN (NaN != anything),
//!     so the NaN proposal is committed and downstream simulation
//!     blows up with a generic "rate expression collapse" error far
//!     from the actual fault site.
//!
//! Fix: treat non-finite energy as divergent at the build_tree leaf,
//! and reject any proposal whose log_p is non-finite or whose z
//! components are non-finite at the nuts_step accept boundary. The
//! upstream `log_prob_and_grad` is the right place to detect
//! non-finite parameters (it's already wired to return -Inf there);
//! the safety net here is a defense-in-depth against the case where
//! a leapfrog *itself* produces NaN before log_prob_and_grad is even
//! called with the bad point.

use sim::inference::nuts::{nuts_step, MassMatrix, NUTSConfig};
use sim::rng::StatefulRng;

/// log_prob_and_grad that returns a NaN gradient — simulating the case
/// where the model's gradient evaluator hit a numerical pathology. Under
/// the buggy NUTS, this propagates through leapfrog to produce NaN z,
/// and the NaN-energy proposal is silently accepted. Under the fix, NUTS
/// detects the non-finite outcome and rejects the proposal.
fn nan_grad_lp(z: &[f64]) -> (f64, Vec<f64>) {
    // Return finite log_p so h0 is finite, but NaN gradient. Leapfrog
    // will then produce NaN p_half, NaN z_new, and `log_prob_and_grad`
    // at NaN z is what we'd see next — but for testing the build_tree
    // detector we only need the energy at the proposal to be non-finite.
    let log_p = -0.5 * z.iter().map(|&x| x * x).sum::<f64>();
    let grad = vec![f64::NAN; z.len()];
    (log_p, grad)
}

/// log_prob_and_grad that returns NaN log_p when the param is "extreme"
/// (|z| > 100). Mimics the actual repro where the rate evaluator sees a
/// NaN/Inf param and the upstream now returns -Inf (after the gh#81
/// structured-error fix). The momentum can still push z out, and the
/// returned log_p tells NUTS the proposal is infeasible.
fn neg_inf_when_extreme(z: &[f64]) -> (f64, Vec<f64>) {
    if z.iter().any(|&x| !x.is_finite() || x.abs() > 100.0) {
        return (f64::NEG_INFINITY, vec![0.0; z.len()]);
    }
    let log_p = -0.5 * z.iter().map(|&x| x * x).sum::<f64>();
    let grad: Vec<f64> = z.iter().map(|&x| -x).collect();
    (log_p, grad)
}

/// Regression: NUTS must NOT commit a NaN-valued z proposal.
///
/// Pre-fix mechanism (Phase 1 verified):
///   - leapfrog with grad=NaN produces NaN p_half → NaN z_new.
///   - build_tree returns z_new (NaN) and log_p_new (NaN).
///   - divergence check `(NaN - h0).abs() > delta_max` is `NaN > 1000`
///     which evaluates to `false` (IEEE 754 NaN compares unordered).
///   - top-level nuts_step's `accepted = z_proposal != current_z` is
///     `true` (NaN != anything), so the NaN params are returned.
/// Downstream, the next call to log_prob_and_grad(NaN params) returns
/// (-Inf, zeros), and the chain reports a NumericalCollapse far from
/// the actual NUTS fault.
///
/// Post-fix invariant: when a NUTS step would commit a non-finite
/// param vector, it must instead either (a) report `divergent=true`
/// and `accepted=false`, or (b) return a finite `params` slice.
#[test]
fn nuts_step_rejects_nan_param_proposal() {
    let mut rng = StatefulRng::new(7);
    let z0 = vec![0.0, 0.0];
    let (log_p0, grad0) = nan_grad_lp(&z0);
    let config = NUTSConfig {
        max_tree_depth: 4,
        step_size: 0.5,
        mass_matrix: MassMatrix::identity(2),
    };

    let result = nuts_step(&z0, log_p0, &grad0, &config, &nan_grad_lp, &mut rng);

    // The result.params and log_posterior must be finite OR the step
    // must report a rejected/divergent move. Returning a NaN-valued
    // proposal as "accepted" is the bug.
    let all_finite = result.params.iter().all(|x| x.is_finite())
                  && result.log_posterior.is_finite();

    assert!(
        all_finite || !result.accepted,
        "NUTS committed a non-finite proposal: accepted={}, params={:?}, log_posterior={}. \
         The build_tree divergence detector must classify non-finite energies as divergent \
         and the top-level accept criterion must reject non-finite z proposals — otherwise \
         downstream simulation receives NaN parameters and produces misleading errors.",
        result.accepted, result.params, result.log_posterior
    );
}

/// Regression: NUTS divergence must be raised when the proposal energy
/// is non-finite. The internal `divergent` flag is what upstream uses
/// to count divergent transitions and (for adaptation) to back off step
/// size. NaN energies sneaking past as "non-divergent" misreports
/// chain health.
#[test]
fn nuts_step_flags_nonfinite_energy_as_divergent() {
    let mut rng = StatefulRng::new(11);
    let z0 = vec![0.0, 0.0];
    let (log_p0, grad0) = nan_grad_lp(&z0);
    let config = NUTSConfig {
        max_tree_depth: 4,
        step_size: 0.5,
        mass_matrix: MassMatrix::identity(2),
    };

    let result = nuts_step(&z0, log_p0, &grad0, &config, &nan_grad_lp, &mut rng);

    assert!(
        result.divergent,
        "NUTS did not flag divergent on a NaN-energy proposal. \
         `(NaN - h0).abs() > delta_max` evaluates to false, so the build_tree \
         leaf at the NaN point returns `divergent=false`, which then propagates \
         up as a non-divergent NUTS step. The fix should classify non-finite \
         energy as divergent."
    );
}

/// Sanity: NUTS still works correctly on a well-conditioned target.
/// The fix must not break the happy path — only reject the pathology.
#[test]
fn nuts_step_accepts_finite_proposals_on_clean_target() {
    let mut rng = StatefulRng::new(42);
    let z0 = vec![0.5, -0.3];
    let (log_p0, grad0) = neg_inf_when_extreme(&z0);
    let config = NUTSConfig {
        max_tree_depth: 6,
        step_size: 0.3,
        mass_matrix: MassMatrix::identity(2),
    };

    // 50 steps; at least one accepted move with finite params expected.
    let mut z = z0.clone();
    let mut log_p = log_p0;
    let mut grad = grad0.clone();
    let mut n_accepted = 0;
    let mut any_nonfinite = false;
    for _ in 0..50 {
        let result = nuts_step(&z, log_p, &grad, &config, &neg_inf_when_extreme, &mut rng);
        if !result.params.iter().all(|x| x.is_finite()) || !result.log_posterior.is_finite() {
            any_nonfinite = true;
        }
        if result.accepted {
            z = result.params;
            log_p = result.log_posterior;
            let (_, g) = neg_inf_when_extreme(&z);
            grad = g;
            n_accepted += 1;
        }
    }

    assert!(!any_nonfinite, "NUTS returned a non-finite proposal on a clean target");
    assert!(n_accepted >= 5, "NUTS accepted only {}/50 moves on a clean target — \
            the fix appears to over-reject", n_accepted);
}

/// When the upstream `log_prob_and_grad` returns -Inf at extreme z
/// (the post-gh#81-fix steady state), NUTS must reject those proposals
/// rather than committing the extreme z to the chain. This is the
/// real-world case: the rate evaluator already classifies NaN params
/// as -Inf via SimError::NonFiniteParameter → recoverable → -Inf
/// return at the log_prob_and_grad boundary in pgas.rs.
#[test]
fn nuts_step_rejects_when_log_p_is_neg_inf_at_proposal() {
    let mut rng = StatefulRng::new(99);
    // Start far from the centre so the leapfrog can push outside the
    // -100 ≤ z ≤ 100 feasible region within a few steps.
    let z0 = vec![90.0, -90.0];
    let (log_p0, grad0) = neg_inf_when_extreme(&z0);
    // Aggressive step size to force the leapfrog to reach |z|>100.
    let config = NUTSConfig {
        max_tree_depth: 5,
        step_size: 50.0,
        mass_matrix: MassMatrix::identity(2),
    };

    let mut n_committed_extreme = 0;
    let mut z = z0.clone();
    let mut log_p = log_p0;
    let mut grad = grad0.clone();
    for _ in 0..30 {
        let result = nuts_step(&z, log_p, &grad, &config, &neg_inf_when_extreme, &mut rng);
        if result.accepted {
            if !result.log_posterior.is_finite() {
                // The proposal was committed at a point where the
                // target says -Inf — this is the bug we're guarding
                // against. The fixed accept criterion should refuse.
                n_committed_extreme += 1;
            }
            z = result.params;
            log_p = result.log_posterior;
            let (_, g) = neg_inf_when_extreme(&z);
            grad = g;
        }
    }

    assert_eq!(
        n_committed_extreme, 0,
        "NUTS committed {} proposals where log_p = -Inf. The accept boundary \
         must reject non-finite log_p proposals; otherwise downstream code \
         sees a chain state with -Inf likelihood and any subsequent draw \
         depending on `current_log_p` produces NaN.", n_committed_extreme
    );
}
