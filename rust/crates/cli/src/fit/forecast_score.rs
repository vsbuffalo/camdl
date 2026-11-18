//! The log scores of a fixed-origin forecast against its targets' held-out
//! values (D79; migration plan, Stage 6, "The log score").
//!
//! The forecast's predictive distribution at a target is the equal-weight
//! mixture over its `M` posterior draws by `N` particles (D53), so the
//! predictive density of a held-out value `y_t` is estimated by
//!
//! ```text
//! p̂(y_t) = (1 / MN) · Σ_{m,i} p(y_t | x_{m,i}, θ_m)
//! ```
//!
//! and its log score is `log p̂(y_t) = logsumexp_{m,i} ℓ_{m,i,t} − log(MN)`,
//! with `ℓ_{m,i,t}` the per-particle log-likelihood the filter's scoring
//! likelihood recorded ([`Forecast::held_out_log_liks`]). It is never an
//! average of per-draw log scores, which would score a different quantity
//! (by Jensen's inequality, one no larger).
//!
//! Two scores are kept apart and labelled apart:
//!
//! - the **marginal** log score of one (stream, target), above;
//! - the **joint** log score of several targets, which sums each sample's
//!   log-likelihoods over the targets before the log-sum-exp,
//!   `logsumexp_{m,i} Σ_t ℓ_{m,i,t} − log(MN)`. Particle `i` of draw `m` is
//!   one trajectory across every target (D54), so this is the log predictive
//!   density of the targets' values together; it is not the sum of their
//!   marginal scores.
//!
//! Nothing here is written to an artifact yet: the score artifact that will
//! carry these (gh#808) is Stage 6's.

#![cfg_attr(not(test), allow(dead_code))] // read by Stage 6's score artifact

use super::forecast_driver::Forecast;

/// Which of the two log scores a value is (D79).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogScoreKind {
    /// One (stream, target)'s held-out value against its predictive marginal.
    Marginal,
    /// Several targets' held-out values together, against the joint
    /// predictive of the samples' trajectories.
    Joint,
}

impl LogScoreKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            LogScoreKind::Marginal => "marginal_log_score",
            LogScoreKind::Joint => "joint_log_score",
        }
    }
}

/// A log score of a forecast.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LogScore {
    pub kind: LogScoreKind,
    /// The targets scored, as indices into the plan's targets: one for a
    /// marginal score.
    pub targets: Vec<usize>,
    /// `logsumexp − log(MN)` over the `n_samples = M·N` samples.
    pub log_score: f64,
    pub n_samples: usize,
}

/// `logsumexp(ll) − log(len)`: the log of the mean of `exp(ll)`. Refuses NaN
/// (a missing term must not be read as a score) and an empty set.
pub(crate) fn log_mean_exp(ll: &[f64]) -> Result<f64, String> {
    if ll.is_empty() {
        return Err("no samples to score".into());
    }
    if let Some(i) = ll.iter().position(|x| x.is_nan()) {
        return Err(format!("sample {i}'s log-likelihood is NaN"));
    }
    Ok(numerics::log_sum_exp(ll) - (ll.len() as f64).ln())
}

/// The marginal log score of every target that has a held-out value, in the
/// plan's order.
pub(crate) fn marginal_log_scores(fc: &Forecast) -> Result<Vec<LogScore>, String> {
    fc.held_out_log_liks.iter().enumerate()
        .filter_map(|(k, ll)| ll.as_ref().map(|ll| (k, ll)))
        .map(|(k, ll)| {
            check_samples(fc, k, ll)?;
            Ok(LogScore {
                kind: LogScoreKind::Marginal,
                targets: vec![k],
                log_score: log_mean_exp(ll).map_err(|e| describe(fc, k, &e))?,
                n_samples: ll.len(),
            })
        })
        .collect()
}

/// The joint log score of `targets` (indices into the plan's targets), each
/// of which must have a held-out value.
pub(crate) fn joint_log_score(fc: &Forecast, targets: &[usize]) -> Result<LogScore, String> {
    if targets.is_empty() {
        return Err("a joint log score needs at least one target".into());
    }
    let n = fc.n_draws * fc.n_particles;
    let mut sum = vec![0.0_f64; n];
    for &k in targets {
        let ll = fc.held_out_log_liks.get(k).and_then(Option::as_ref).ok_or_else(|| describe(
            fc, k, "it has no held-out value, so it cannot enter a joint score"))?;
        check_samples(fc, k, ll)?;
        for (s, l) in sum.iter_mut().zip(ll) {
            *s += l;
        }
    }
    let mut sorted = targets.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != targets.len() {
        return Err("a joint log score counts each target once".into());
    }
    Ok(LogScore {
        kind: LogScoreKind::Joint,
        targets: sorted,
        log_score: log_mean_exp(&sum)?,
        n_samples: n,
    })
}

fn check_samples(fc: &Forecast, k: usize, ll: &[f64]) -> Result<(), String> {
    let n = fc.n_draws * fc.n_particles;
    if ll.len() != n {
        return Err(describe(fc, k, &format!("{} log-likelihoods for {n} samples", ll.len())));
    }
    Ok(())
}

fn describe(fc: &Forecast, k: usize, why: &str) -> String {
    match fc.plan.targets().get(k) {
        Some(t) => format!("target of '{}' ending at {}: {why}", t.stream, t.measures.end()),
        None => format!("target {k}: {why}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The definition, computed independently: the log of the plain mean of
    /// the likelihoods, for values small enough to exponentiate directly.
    fn direct(ll: &[f64]) -> f64 {
        (ll.iter().map(|l| l.exp()).sum::<f64>() / ll.len() as f64).ln()
    }

    #[test]
    fn log_mean_exp_is_the_log_of_the_mean_likelihood() {
        let ll = [0.2_f64.ln(), 0.4_f64.ln(), 0.05_f64.ln(), 0.35_f64.ln()];
        assert!((log_mean_exp(&ll).unwrap() - 0.25_f64.ln()).abs() < 1e-15);
        // Far below exp's range, the shift keeps it exact: every term equal.
        assert!((log_mean_exp(&[-2000.0; 3]).unwrap() + 2000.0).abs() < 1e-12);
        assert_eq!(log_mean_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]).unwrap(), f64::NEG_INFINITY);
        assert!(log_mean_exp(&[0.0, f64::NAN]).is_err(), "NaN is refused, never read as 0");
        assert!(log_mean_exp(&[]).is_err());
        // It is not the mean of the log-likelihoods.
        let mean_log = ll.iter().sum::<f64>() / ll.len() as f64;
        assert!(log_mean_exp(&ll).unwrap() > mean_log + 0.1);
        assert!((log_mean_exp(&ll).unwrap() - direct(&ll)).abs() < 1e-15);
    }
}
