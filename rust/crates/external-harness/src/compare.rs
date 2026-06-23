//! Tolerance-aware comparison between camdl and reference summary stats.

use crate::manifest::{Check, CheckKind};
use crate::summary::{Summary, SummaryRow};

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub stat: String,
    pub kind_description: String,
    pub outcome: Outcome,
    /// Machine-readable detail for test-failure messages.
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    Fail,
}

pub fn run_check(
    stat: &str,
    check: &Check,
    camdl: &SummaryRow,
    reference: &SummaryRow,
) -> CheckResult {
    match &check.kind {
        CheckKind::Mean { tol_abs, tol_rel } => {
            let (pass, detail) = check_scalar(
                camdl.mean, reference.mean, *tol_abs, *tol_rel, "mean",
            );
            CheckResult {
                stat: stat.to_string(),
                kind_description: "mean".to_string(),
                outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                detail,
            }
        }
        CheckKind::Quantiles { q, tol_abs, tol_rel } => {
            let mut fails = Vec::new();
            for qv in q {
                let (cv, rv) = (pick_quantile(camdl, *qv), pick_quantile(reference, *qv));
                let (pass, detail) = check_scalar(
                    cv, rv, *tol_abs, *tol_rel, &format!("q{:.3}", qv),
                );
                if !pass { fails.push(detail); }
            }
            let outcome = if fails.is_empty() { Outcome::Pass } else { Outcome::Fail };
            CheckResult {
                stat: stat.to_string(),
                kind_description: format!("quantiles {:?}", q),
                outcome,
                detail: if fails.is_empty() {
                    format!("all {} quantiles within tolerance", q.len())
                } else {
                    fails.join("; ")
                },
            }
        }
        CheckKind::Value { tol_abs, tol_rel } => {
            // Scalar compare against the mean (the only single-value field
            // that makes sense for non-ensemble fixtures).
            let (pass, detail) = check_scalar(
                camdl.mean, reference.mean, *tol_abs, *tol_rel, "value",
            );
            CheckResult {
                stat: stat.to_string(),
                kind_description: "value".to_string(),
                outcome: if pass { Outcome::Pass } else { Outcome::Fail },
                detail,
            }
        }
        CheckKind::ProportionTest { alpha } => {
            // Two-sample z-test for proportions using mean as p_hat and
            // n as sample size. Both means must be in [0, 1].
            let p1 = camdl.mean.clamp(0.0, 1.0);
            let p2 = reference.mean.clamp(0.0, 1.0);
            let n1 = camdl.n as f64;
            let n2 = reference.n as f64;
            let p_pool = (p1 * n1 + p2 * n2) / (n1 + n2);
            let se = (p_pool * (1.0 - p_pool) * (1.0/n1 + 1.0/n2)).sqrt();
            let (outcome, detail) = if se == 0.0 {
                // Both sides are exactly 0 or exactly 1. Pass iff identical.
                if (p1 - p2).abs() < 1e-12 {
                    (Outcome::Pass, format!(
                        "proportions identical ({:.6}); se=0 (pooled p={:.6})", p1, p_pool
                    ))
                } else {
                    (Outcome::Fail, format!(
                        "proportions differ: camdl={:.6}, ref={:.6}; se=0 so any diff is significant",
                        p1, p2
                    ))
                }
            } else {
                let z = (p1 - p2).abs() / se;
                // Two-sided z threshold at alpha. Accurate enough for alpha
                // in [1e-4, 0.1] without pulling a cdf crate.
                let z_crit = z_two_sided(*alpha);
                let pass = z <= z_crit;
                (
                    if pass { Outcome::Pass } else { Outcome::Fail },
                    format!(
                        "proportions camdl={:.4} (n={}) ref={:.4} (n={}); \
                         z={:.3}, critical={:.3} at α={}",
                        p1, camdl.n, p2, reference.n, z, z_crit, alpha
                    ),
                )
            };
            CheckResult {
                stat: stat.to_string(),
                kind_description: "proportion-test".to_string(),
                outcome,
                detail,
            }
        }
        CheckKind::KsTest { alpha } => {
            // Not implemented yet. The summary format doesn't carry the
            // full ECDF; KS requires either raw samples or an ECDF
            // fixture. Defer to a follow-up: case_category = "kstest"
            // would need a separate fixture format.
            CheckResult {
                stat: stat.to_string(),
                kind_description: format!("ks-test α={}", alpha),
                outcome: Outcome::Fail,
                detail: "ks-test not yet implemented — summary format \
                         doesn't preserve ECDF".to_string(),
            }
        }
    }
}

pub fn run_all(
    expected: &crate::manifest::ExpectedManifest,
    camdl: &Summary,
    reference: &Summary,
) -> Vec<CheckResult> {
    expected.checks.iter().map(|(stat, check)| {
        let (cam, refr) = match (camdl.rows.get(stat), reference.rows.get(stat)) {
            (Some(c), Some(r)) => (c, r),
            (None, _) => return CheckResult {
                stat: stat.clone(),
                kind_description: describe_kind(&check.kind),
                outcome: Outcome::Fail,
                detail: format!("stat '{}' missing from camdl summary", stat),
            },
            (_, None) => return CheckResult {
                stat: stat.clone(),
                kind_description: describe_kind(&check.kind),
                outcome: Outcome::Fail,
                detail: format!("stat '{}' missing from reference summary", stat),
            },
        };
        run_check(stat, check, cam, refr)
    }).collect()
}

fn describe_kind(k: &CheckKind) -> String {
    match k {
        CheckKind::Mean {..}             => "mean".to_string(),
        CheckKind::Quantiles { q, .. }   => format!("quantiles {:?}", q),
        CheckKind::Value {..}            => "value".to_string(),
        CheckKind::ProportionTest { alpha } => format!("proportion-test α={}", alpha),
        CheckKind::KsTest { alpha }      => format!("ks-test α={}", alpha),
    }
}

fn pick_quantile(r: &SummaryRow, q: f64) -> f64 {
    if (q - 0.025).abs() < 1e-6 { r.q025 }
    else if (q - 0.5).abs() < 1e-6 { r.q500 }
    else if (q - 0.975).abs() < 1e-6 { r.q975 }
    else {
        // TODO: summary format stores only three quantiles. A future
        // revision can carry a configurable quantile set; for now, map
        // arbitrary q to the nearest stored one and warn.
        eprintln!("warning: summary only carries q025/q500/q975; {} treated as nearest", q);
        if q < 0.2625 { r.q025 }
        else if q > 0.7375 { r.q975 }
        else { r.q500 }
    }
}

/// Returns `(pass, human_detail)` for a scalar tolerance check.
/// When both tol_abs and tol_rel are set, either-passes is accepted
/// (inclusive OR) per principle #5 in the proposal.
fn check_scalar(
    actual: f64,
    expected: f64,
    tol_abs: Option<f64>,
    tol_rel: Option<f64>,
    label: &str,
) -> (bool, String) {
    let diff = (actual - expected).abs();
    let rel = if expected.abs() > 1e-30 { diff / expected.abs() } else { 0.0 };
    let abs_pass = tol_abs.map(|t| diff <= t);
    let rel_pass = tol_rel.map(|t| rel <= t);
    let pass = match (abs_pass, rel_pass) {
        (Some(a), Some(r)) => a || r,
        (Some(a), None)    => a,
        (None, Some(r))    => r,
        (None, None)       => return (false, format!(
            "{}: no tolerance set (tol_abs or tol_rel required)", label)),
    };
    let tol_str = match (tol_abs, tol_rel) {
        (Some(a), Some(r)) => format!("tol_abs={} OR tol_rel={}", a, r),
        (Some(a), None)    => format!("tol_abs={}", a),
        (None, Some(r))    => format!("tol_rel={}", r),
        _ => "".into(),
    };
    let detail = format!(
        "{}: camdl={:.6}, ref={:.6}, diff={:.3e} ({:.2}%); {}",
        label, actual, expected, diff, rel * 100.0, tol_str,
    );
    (pass, detail)
}

/// Two-sided z critical value for given alpha: z = Φ⁻¹(1 − alpha/2). Uses the
/// shared `numerics::normal_quantile` (Beasley–Springer–Moro) — previously a
/// local `inv_norm` copy that had drifted from the runtime's version (it
/// lacked the interior clamp, so it returned ±inf at p ∈ {0, 1}).
fn z_two_sided(alpha: f64) -> f64 {
    numerics::normal_quantile(1.0 - alpha / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_critical_known_values() {
        // α=0.05 two-sided ⇒ z ≈ 1.960
        assert!((z_two_sided(0.05) - 1.9600).abs() < 1e-3);
        // α=0.01 ⇒ z ≈ 2.576
        assert!((z_two_sided(0.01) - 2.5758).abs() < 1e-3);
    }

    #[test]
    fn scalar_rel_pass() {
        let (pass, _) = check_scalar(100.0, 102.0, None, Some(0.05), "mean");
        assert!(pass);
    }

    #[test]
    fn scalar_rel_fail() {
        let (pass, _) = check_scalar(100.0, 110.0, None, Some(0.05), "mean");
        assert!(!pass);
    }

    #[test]
    fn scalar_abs_passes_even_if_rel_would_fail() {
        // diff = 0.01, rel = 10% (big), abs = 0.01 (tiny) — either-passes
        let (pass, _) = check_scalar(0.1, 0.11, Some(0.02), Some(0.05), "mean");
        assert!(pass);
    }
}
