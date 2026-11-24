//! The weighted interval score of a predictive distribution given by its
//! quantiles, and its three parts (Bracher, Ray, Gneiting & Reich 2021,
//! "Evaluating epidemic forecasts in an interval format", PLOS Comput. Biol.
//! 17(2), e1008618, §2.2).
//!
//! For `K` central intervals at levels `1 − α_k`, with lower and upper ends
//! `l_k` and `u_k`, and the median `m`,
//!
//! ```text
//! IS_α(l, u; y) = (u − l) + (2/α)(l − y)·1(y < l) + (2/α)(y − u)·1(y > u)
//! WIS(y)        = 1/(K + 1/2) · ( w_0·|y − m| + Σ_k w_k·IS_{α_k}(l_k, u_k; y) )
//! ```
//!
//! with `w_0 = 1/2` and `w_k = α_k / 2`. The score splits into the
//! **dispersion**, `1/(K + 1/2)·Σ_k w_k (u_k − l_k)`, and the penalties for
//! an observation **below** the intervals (the prediction too high,
//! "overprediction") and **above** them ("underprediction"), the median's
//! absolute error counted in whichever of the two its side gives. The three
//! sum to the score.
//!
//! [`HUB_LEVELS`] is the quantile set of the US COVID-19 Forecast Hub and of
//! CDC FluSight from 2021 (Cramer et al. 2022, PNAS 119(15), e2113561119;
//! Bracher et al. 2021, §2.2): the median and eleven central intervals, at
//! `α ∈ {0.02, 0.05, 0.1, 0.2, …, 0.9}`.

use serde::{Deserialize, Serialize};

/// The 23 quantile levels the forecast hubs collect, ascending.
pub const HUB_LEVELS: [f64; 23] = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
    0.9, 0.95, 0.975, 0.99,
];

/// One central interval: its `α` (the interval is at level `1 − α`) and its
/// ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    pub alpha: f64,
    pub lower: f64,
    pub upper: f64,
}

/// The weighted interval score and its parts; `dispersion + overprediction
/// + underprediction == wis` up to rounding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Wis {
    pub wis: f64,
    pub dispersion: f64,
    /// The penalty for an observation below the intervals and the median.
    pub overprediction: f64,
    /// The penalty for an observation above the intervals and the median.
    pub underprediction: f64,
}

/// The weighted interval score of the observation `y` against the median
/// `median` and the central `intervals`. Every `alpha` must lie in `(0, 1)`
/// and each interval's `lower` must not exceed its `upper`; `None` when one
/// does not, or when there is no interval.
pub fn weighted_interval_score(median: f64, intervals: &[Interval], y: f64) -> Option<Wis> {
    if intervals.is_empty()
        || intervals.iter().any(|i| !(i.alpha > 0.0 && i.alpha < 1.0) || i.lower > i.upper)
    {
        return None;
    }
    let scale = 1.0 / (intervals.len() as f64 + 0.5);
    let mut dispersion = 0.0;
    let mut over = 0.5 * (median - y).max(0.0);
    let mut under = 0.5 * (y - median).max(0.0);
    for i in intervals {
        let w = i.alpha / 2.0;
        dispersion += w * (i.upper - i.lower);
        over += w * (2.0 / i.alpha) * (i.lower - y).max(0.0);
        under += w * (2.0 / i.alpha) * (y - i.upper).max(0.0);
    }
    let (dispersion, overprediction, underprediction) = (scale * dispersion, scale * over, scale * under);
    Some(Wis { wis: dispersion + overprediction + underprediction, dispersion, overprediction, underprediction })
}

/// The median and central intervals that the quantiles `values` at the
/// ascending `levels` (which must contain 0.5 and be symmetric about it) give.
pub fn intervals_of(levels: &[f64], values: &[f64]) -> Option<(f64, Vec<Interval>)> {
    if levels.len() != values.len() || levels.len().is_multiple_of(2) {
        return None;
    }
    let mid = levels.len() / 2;
    if (levels[mid] - 0.5).abs() > 1e-12 {
        return None;
    }
    let mut intervals = Vec::with_capacity(mid);
    for k in 0..mid {
        let (lo, hi) = (levels[k], levels[levels.len() - 1 - k]);
        if (lo + hi - 1.0).abs() > 1e-12 {
            return None;
        }
        intervals.push(Interval { alpha: 2.0 * lo, lower: values[k], upper: values[levels.len() - 1 - k] });
    }
    Some((values[mid], intervals))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand computation by the formula of Bracher et al. (2021): median
    /// 10, the 50% interval [8, 12] and the 90% interval [5, 16], so
    /// `K = 2` and the weights are 1/2, 0.25 and 0.05.
    #[test]
    fn the_score_and_its_parts_equal_the_hand_computation() {
        let intervals = [
            Interval { alpha: 0.5, lower: 8.0, upper: 12.0 },
            Interval { alpha: 0.1, lower: 5.0, upper: 16.0 },
        ];
        let scale = 1.0 / 2.5;
        // y = 3: below every interval and the median.
        //   dispersion = 0.25·4 + 0.05·11 = 1.55
        //   over       = 0.5·7 + 0.25·(2/0.5)·5 + 0.05·(2/0.1)·2 = 3.5 + 5 + 2 = 10.5
        let w = weighted_interval_score(10.0, &intervals, 3.0).unwrap();
        assert!((w.dispersion - scale * 1.55).abs() < 1e-12, "{w:?}");
        assert!((w.overprediction - scale * 10.5).abs() < 1e-12, "{w:?}");
        assert_eq!(w.underprediction, 0.0);
        assert!((w.wis - scale * 12.05).abs() < 1e-12, "{w:?}");
        // y = 13: above the 50% interval and the median, inside the 90%.
        //   under = 0.5·3 + 0.25·4·1 = 2.5
        let w = weighted_interval_score(10.0, &intervals, 13.0).unwrap();
        assert!((w.underprediction - scale * 2.5).abs() < 1e-12, "{w:?}");
        assert_eq!(w.overprediction, 0.0);
        assert!((w.wis - scale * (1.55 + 2.5)).abs() < 1e-12);
        // y at the median: the dispersion alone.
        let w = weighted_interval_score(10.0, &intervals, 10.0).unwrap();
        assert!((w.wis - scale * 1.55).abs() < 1e-12);
    }

    /// The score is the sum of the pinball losses over the quantiles, scaled
    /// by `1/(K + 1/2)`: the quantile form of the same rule, computed
    /// independently.
    #[test]
    fn the_score_equals_the_scaled_sum_of_pinball_losses_on_the_hub_set() {
        let values: Vec<f64> = HUB_LEVELS.iter().map(|q| 100.0 * q * q + 3.0).collect();
        let (median, intervals) = intervals_of(&HUB_LEVELS, &values).unwrap();
        assert_eq!(intervals.len(), 11);
        assert!((intervals[0].alpha - 0.02).abs() < 1e-12 && (intervals[10].alpha - 0.9).abs() < 1e-12);
        for y in [-5.0, 0.0, 20.0, 28.0, 60.0, 200.0] {
            let pinball: f64 = HUB_LEVELS
                .iter()
                .zip(&values)
                .map(|(&tau, &q)| if y >= q { tau * (y - q) } else { (1.0 - tau) * (q - y) })
                .sum();
            let w = weighted_interval_score(median, &intervals, y).unwrap();
            assert!((w.wis - pinball / 11.5).abs() < 1e-10, "y = {y}: {} vs {}", w.wis, pinball / 11.5);
            assert!((w.dispersion + w.overprediction + w.underprediction - w.wis).abs() < 1e-12);
        }
    }

    #[test]
    fn malformed_intervals_are_refused() {
        assert!(weighted_interval_score(1.0, &[], 1.0).is_none());
        assert!(weighted_interval_score(1.0, &[Interval { alpha: 0.0, lower: 0.0, upper: 1.0 }], 1.0).is_none());
        assert!(weighted_interval_score(1.0, &[Interval { alpha: 0.5, lower: 2.0, upper: 1.0 }], 1.0).is_none());
        assert!(intervals_of(&[0.25, 0.75], &[1.0, 2.0]).is_none());
    }
}
