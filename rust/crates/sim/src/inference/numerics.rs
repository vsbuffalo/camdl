//! gh#audit-H3. Numerically-stable primitives shared across inference
//! and simulation backends. The audit found four near-identical
//! `(1.0 - (-total_rate * dt).exp()).clamp(...)` sites
//! (chain_binomial.rs:304, pgas.rs:339,
//! pgas_grad.rs:165) that all hit the same catastrophic-cancellation
//! regime when `total_rate * dt ≫ 1` — the direct subtraction
//! `1 - exp(-large)` collapses to floating-point noise. Particle
//! weights and gradients in this regime become essentially noise,
//! pinning NUTS step-size adaptation low and biasing the posterior
//! toward parameter values that keep p_total in the interior
//! (smaller rates).
//!
//! `prob_q_from_rate_dt` returns `(p, q = 1 - p)` as a pair without
//! ever subtracting near-1 values:
//!
//!   q = exp(-total_rate * dt)        # tail-stable
//!   p = -expm1(-total_rate * dt)     # near-zero-stable when rate*dt → 0
//!
//! Both are computed independently to avoid any subtraction. The
//! pair is the right primitive for binomial split rates: callers
//! that need only `p` discard `q`; those that need both (e.g. the
//! gradient form `k/p - (n-k)/q`) get them without re-subtracting.
//!
//! Future work (proposal §6, deferred from this commit): thread the
//! `(p, q)` pair through `binom_logpmf` and the PGAS gradient form
//! so the gradient at extreme rates avoids `(n-k)/(1-p)` instability
//! too. The primitive extracted here is the foundation.

/// Compute (p, q) = (1 - exp(-r·dt), exp(-r·dt)) without
/// catastrophic cancellation. p+q = 1 by construction (modulo ULP).
///
/// `r` may be 0 (degenerate: returns (0, 1)) or +∞ (returns (1, 0)).
/// NaN inputs propagate through the `expm1`/`exp` pair as NaN; the
/// caller's NaN guard (e.g. eval_propensities post-eval check from
/// audit C6/S1) catches those.
#[inline]
pub fn prob_q_from_rate_dt(r: f64, dt: f64) -> (f64, f64) {
    let neg_x = -r * dt;
    let q = neg_x.exp();
    // -expm1(neg_x) = 1 - exp(neg_x) = p, computed as a single
    // intrinsic to avoid the cancellation when neg_x is small.
    let p = -(neg_x).exp_m1();
    (p, q)
}

/// Interior clamp for binomial split/exit probabilities so `log(p)` /
/// `log(1−p)` and their NUTS gradients stay finite. Shared by the PGAS
/// log-density (`pgas.rs`) and its gradient (`pgas_grad.rs`); they MUST use
/// the same value — a one-sided change makes the Hamiltonian
/// non-conservative (energy drift → spurious divergences). Distinct in
/// *concept* from the `1e-15` rate/step floors (`RATE_EPSILON`,
/// `MIN_STEP_EPS`) despite the shared magnitude.
pub const BINOM_PROB_EPS: f64 = 1e-15;

/// Variant that clamps `(p, q)` to a closed interval to avoid
/// degenerate weights in inference. Some PGAS / NUTS hot paths need
/// p strictly in (0, 1) (otherwise log(p) or log(q) is -inf and
/// the gradient is undefined). The clamp is symmetric: if p is
/// clamped up by `eps`, q is clamped down by `eps` so `p + q = 1`
/// holds within ULP.
#[inline]
pub fn prob_q_from_rate_dt_clamped(r: f64, dt: f64, eps: f64) -> (f64, f64) {
    let (p, _q) = prob_q_from_rate_dt(r, dt);
    let p_c = p.clamp(eps, 1.0 - eps);
    let q_c = (1.0 - p_c).max(eps);
    (p_c, q_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_sums_to_one_in_normal_regime() {
        let (p, q) = prob_q_from_rate_dt(0.5, 1.0);
        assert!((p + q - 1.0).abs() < 1e-15);
        // beats direct subtraction in the extreme regime
    }

    #[test]
    fn extreme_rate_does_not_collapse_to_zero_minus_one() {
        let (p, q) = prob_q_from_rate_dt(50.0, 1.0);
        assert!(p > 0.999);
        assert!(q > 0.0);  // exp(-50) ≈ 1.9e-22 — small but nonzero
        assert!(q < 1e-20);
    }

    #[test]
    fn small_rate_keeps_p_accurate() {
        // rate*dt = 1e-10 → p ≈ 1e-10 - 5e-21 (Taylor: 1 - exp(-x) =
        // x - x²/2 + ...). Direct subtraction `1 - exp(-1e-10)` loses
        // the leading-bits cancellation; expm1-based form keeps full
        // f64 precision (relative error < 4 ULP, ≈ 1e-15 of p).
        let (p, _q) = prob_q_from_rate_dt(1e-10, 1.0);
        assert!((p - 1e-10).abs() < 1e-20,
            "p = {} (expected ~1e-10), absolute error {} > 1e-20", p, (p - 1e-10).abs());
    }

    #[test]
    fn zero_rate_returns_zero_and_one() {
        let (p, q) = prob_q_from_rate_dt(0.0, 1.0);
        assert_eq!(p, 0.0);
        assert_eq!(q, 1.0);
    }

    #[test]
    fn clamped_variant_preserves_p_plus_q_eq_one() {
        let (p, q) = prob_q_from_rate_dt_clamped(50.0, 1.0, 1e-15);
        assert!((p + q - 1.0).abs() < 2.0 * 1e-15);
    }
}
