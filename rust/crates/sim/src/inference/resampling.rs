//! Systematic resampling for the particle filter.
//!
//! Lower variance than multinomial resampling, O(N), standard choice
//! for bootstrap filters (Carpenter, Clifford & Fearnhead 1999).

use crate::rng::StatefulRng;
use super::types::normalize_log_weights;

/// Systematic resampling. Returns indices of selected particles.
///
/// `log_weights` are unnormalized log-weights. Internally normalizes
/// via log-sum-exp to avoid overflow.
///
/// Algorithm: one uniform draw U ~ [0, 1/N), then select particle j
/// whenever the cumulative weight crosses U + i/N for i = 0..N-1.
/// This gives exactly N selected particles with probability proportional
/// to exp(log_weight).
pub fn systematic_resample(log_weights: &[f64], rng: &mut StatefulRng) -> Vec<usize> {
    if log_weights.is_empty() { return vec![]; }
    let weights = normalize_log_weights(log_weights);
    systematic_resample_core(&weights, rng.uniform())
}

/// Core systematic-resampling loop over **already-normalized** `weights`, given
/// a base uniform draw `u0 ∈ [0, 1)`: one uniform, evenly-spaced thresholds,
/// returning exactly `weights.len()` selected indices.
///
/// Shared by the rng-driven [`systematic_resample`] and the correlated PF's
/// `systematic_resample_fixed_u` (which supplies a *fixed* `u0` for
/// common-random-numbers coupling across paired runs) so the two selection
/// loops can never drift apart.
pub(crate) fn systematic_resample_core(weights: &[f64], u0: f64) -> Vec<usize> {
    let n = weights.len();
    if n == 0 { return vec![]; }
    let u = u0 / n as f64;
    let mut indices = Vec::with_capacity(n);
    let mut cumsum = 0.0;
    let mut j = 0;
    for i in 0..n {
        let threshold = u + i as f64 / n as f64;
        while j < n - 1 && cumsum + weights[j] < threshold {
            cumsum += weights[j];
            j += 1;
        }
        indices.push(j);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_systematic_resample_uniform_weights() {
        let mut rng = StatefulRng::new(42);
        let log_weights = vec![0.0; 100]; // all equal
        let indices = systematic_resample(&log_weights, &mut rng);
        assert_eq!(indices.len(), 100);
        // With uniform weights, each particle should be selected exactly once
        let mut counts = vec![0usize; 100];
        for &i in &indices { counts[i] += 1; }
        for &c in &counts {
            assert_eq!(c, 1, "uniform weights should give exactly 1 copy per particle");
        }
    }

    #[test]
    fn test_systematic_resample_degenerate() {
        let mut rng = StatefulRng::new(42);
        // One particle has all the weight
        let mut log_weights = vec![f64::NEG_INFINITY; 10];
        log_weights[3] = 0.0;
        let indices = systematic_resample(&log_weights, &mut rng);
        assert_eq!(indices.len(), 10);
        // All should select particle 3
        for &i in &indices {
            assert_eq!(i, 3, "degenerate weights should select only particle 3");
        }
    }

    #[test]
    fn test_systematic_resample_proportional() {
        let mut rng = StatefulRng::new(42);
        // Particle 0 has 3x the weight of particle 1
        let log_weights = vec![3.0_f64.ln(), 0.0, 0.0, 0.0];
        let indices = systematic_resample(&log_weights, &mut rng);
        assert_eq!(indices.len(), 4);
        let count_0 = indices.iter().filter(|&&i| i == 0).count();
        // Particle 0 should get ~2 copies (3/6 * 4 = 2)
        assert!((1..=3).contains(&count_0),
            "particle 0 (weight 3/6) got {} copies out of 4", count_0);
    }
}
