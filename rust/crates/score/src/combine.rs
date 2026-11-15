//! Combining prequential traces: the totals of a set of filter replicates.

use crate::trace::PrequentialTrace;

/// The replicate totals a derived trace records (schema 5): each replicate's
/// total elpd and per-stream sums, in replicate order, and the filter-noise
/// Monte-Carlo SE from the totals.
///
/// The SE is the log-mean-exp delta-method SE of the replicate TOTAL elpds
/// (pomp practice; the PF likelihood estimator is unbiased on the natural
/// scale). The headline elpd is the SUM of per-step log-mean-exp values —
/// the pointwise-consistent aggregate (its steps sum to it, and the paired Δ
/// machinery reads the steps); the two aggregates agree in expectation to
/// O(per-step variance), and the SE measures the same filter-noise scale
/// either way. `None` for a single replicate.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplicateTotals {
    pub elpd: Vec<f64>,
    pub by_stream: std::collections::BTreeMap<String, Vec<f64>>,
    pub mc_se: Option<f64>,
}

impl ReplicateTotals {
    pub fn of(replicates: &[PrequentialTrace]) -> Self {
        let elpd: Vec<f64> = replicates.iter().map(|t| t.elpd()).collect();
        let mut by_stream: std::collections::BTreeMap<String, Vec<f64>> =
            std::collections::BTreeMap::new();
        for t in replicates {
            for (stream, total) in t.elpd_by_stream() {
                by_stream.entry(stream).or_default().push(total);
            }
        }
        let mc_se = (elpd.len() >= 2)
            .then(|| numerics::logmeanexp_with_se(&elpd).1);
        ReplicateTotals { elpd, by_stream, mc_se }
    }
}
