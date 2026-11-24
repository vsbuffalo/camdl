//! Simplex (barycentric) transform tests.
//!
//! Verifies: softmax round-trip, sum-to-1 invariant, numerical stability,
//! perturbation preserves simplex, edge cases (zero/tiny fractions).

use sim::inference::if2::ZSd;
use sim::inference::if2::SimplexGroup;
use sim::rng::StatefulRng;

fn rng() -> StatefulRng { StatefulRng::new(42) }

fn sds(v: &[f64]) -> Vec<ZSd> { v.iter().map(|&s| ZSd::new(s).unwrap()).collect() }

fn assert_sums_to_one(fracs: &[f64], msg: &str) {
    let sum: f64 = fracs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10,
        "{}: sum = {} (expected 1.0)", msg, sum);
}

fn assert_all_positive(fracs: &[f64], msg: &str) {
    for (i, &f) in fracs.iter().enumerate() {
        assert!(f > 0.0, "{}: frac[{}] = {} (must be > 0)", msg, i, f);
    }
}

// ── to_log_barycentric ──────────────────────────────────────────────────

#[test]
fn log_barycentric_of_uniform() {
    // Equal fractions → all log-ratios should be equal (= -ln(n))
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[0.1, 0.1, 0.1]),
    };
    let params = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
    let z = group.to_log_barycentric(&params);
    let expected = (1.0_f64 / 3.0).ln();
    for &zi in &z {
        assert!((zi - expected).abs() < 1e-10,
            "uniform fractions: z = {:.6}, expected {:.6}", zi, expected);
    }
}

#[test]
fn log_barycentric_of_he2010_fractions() {
    // He et al. initial fractions: s0=0.0297, e0=5.17e-5, i0=5.14e-5, r0=0.97
    let group = SimplexGroup {
        indices: vec![0, 1, 2, 3],
        sds: sds(&[0.1, 0.1, 0.1, 0.1]),
    };
    let mut params = vec![0.0297, 5.17e-5, 5.14e-5, 0.9703];
    // Normalize to exactly sum to 1 (avoids round-trip normalization mismatch)
    let sum: f64 = params.iter().sum();
    for p in &mut params { *p /= sum; }

    let z = group.to_log_barycentric(&params);

    // r0 should have the largest (least negative) log-ratio
    assert!(z[3] > z[0], "r0 log-ratio should be > s0");
    assert!(z[0] > z[1], "s0 log-ratio should be > e0");

    // Round-trip: softmax of log-ratios should recover fractions
    let recovered = SimplexGroup::from_log_barycentric(&z);
    for i in 0..4 {
        assert!((recovered[i] - params[i]).abs() < 1e-10,
            "round-trip failed for index {}: {} vs {}", i, recovered[i], params[i]);
    }
}

// ── from_log_barycentric (softmax) ──────────────────────────────────────

#[test]
fn softmax_sums_to_one() {
    let z = vec![1.0, 2.0, 3.0];
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax");
    assert_all_positive(&fracs, "softmax");
}

#[test]
fn softmax_of_zeros() {
    // All equal log-ratios → uniform
    let z = vec![0.0, 0.0, 0.0, 0.0];
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax(zeros)");
    for &f in &fracs {
        assert!((f - 0.25).abs() < 1e-10, "expected 0.25, got {}", f);
    }
}

#[test]
fn softmax_numerical_stability_large_positive() {
    // Extreme positive values — should not overflow
    let z = vec![300.0, 301.0, 299.0];
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax(large positive)");
    assert_all_positive(&fracs, "softmax(large positive)");
    // The middle value (301) should dominate
    assert!(fracs[1] > fracs[0], "exp(301) > exp(300)");
    assert!(fracs[1] > fracs[2], "exp(301) > exp(299)");
}

#[test]
fn softmax_numerical_stability_large_negative() {
    // Extreme negative values — should not underflow to all zeros
    let z = vec![-300.0, -301.0, -299.0];
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax(large negative)");
    assert_all_positive(&fracs, "softmax(large negative)");
}

#[test]
fn softmax_one_dominant() {
    // One value much larger → it gets ~1.0, others ~0
    let z = vec![-100.0, 100.0, -100.0];
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax(one dominant)");
    assert!(fracs[1] > 0.999, "dominant frac should be ~1.0, got {}", fracs[1]);
}

#[test]
fn softmax_two_elements() {
    let z = vec![0.0, 0.0];
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax(2 elements)");
    assert!((fracs[0] - 0.5).abs() < 1e-10);
    assert!((fracs[1] - 0.5).abs() < 1e-10);
}

#[test]
fn softmax_many_elements() {
    let z = vec![1.0; 100]; // 100 equal elements
    let fracs = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&fracs, "softmax(100 elements)");
    for &f in &fracs {
        assert!((f - 0.01).abs() < 1e-10, "expected 0.01, got {}", f);
    }
}

// ── Round-trip ──────────────────────────────────────────────────────────

#[test]
fn round_trip_preserves_fractions() {
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[0.1, 0.1, 0.1]),
    };
    // Various fraction vectors that sum to 1
    let test_cases = vec![
        vec![0.5, 0.3, 0.2],
        vec![0.99, 0.005, 0.005],
        vec![0.001, 0.001, 0.998],
        vec![1.0/3.0, 1.0/3.0, 1.0/3.0],
    ];

    for fracs in test_cases {
        let z = group.to_log_barycentric(&fracs);
        let recovered = SimplexGroup::from_log_barycentric(&z);
        for i in 0..fracs.len() {
            assert!((recovered[i] - fracs[i]).abs() < 1e-10,
                "round-trip failed: {:?} → {:?} → {:?}", fracs, z, recovered);
        }
    }
}

#[test]
fn round_trip_non_unit_sum() {
    // Fractions that don't sum to 1 → round-trip normalizes them
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[0.1, 0.1, 0.1]),
    };
    let params = vec![2.0, 3.0, 5.0]; // sum = 10
    let z = group.to_log_barycentric(&params);
    let recovered = SimplexGroup::from_log_barycentric(&z);
    assert_sums_to_one(&recovered, "normalized round-trip");
    // Proportions should be preserved: 0.2, 0.3, 0.5
    assert!((recovered[0] - 0.2).abs() < 1e-10);
    assert!((recovered[1] - 0.3).abs() < 1e-10);
    assert!((recovered[2] - 0.5).abs() < 1e-10);
}

// ── Perturbation ────────────────────────────────────────────────────────

#[test]
fn perturb_preserves_sum_to_one() {
    let group = SimplexGroup {
        indices: vec![0, 1, 2, 3],
        sds: sds(&[0.1, 0.1, 0.1, 0.1]),
    };
    let mut params = vec![0.25, 0.25, 0.25, 0.25];
    let mut rng = rng();

    // Run 100 perturbations — every one should sum to 1
    for i in 0..100 {
        group.perturb(&mut params, &mut rng, 1.0);
        let sum: f64 = group.indices.iter().map(|&j| params[j]).sum();
        assert!((sum - 1.0).abs() < 1e-10,
            "perturbation {}: sum = {} (expected 1.0)", i, sum);
        assert_all_positive(&params, &format!("perturbation {}", i));
    }
}

#[test]
fn perturb_with_cooling() {
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[0.5, 0.5, 0.5]), // large rw_sd
    };

    // Run with cooling = 0.01 (almost frozen) — fractions should barely change
    let original = vec![0.6, 0.3, 0.1];
    let mut params = original.clone();
    let mut rng = rng();

    for _ in 0..10 {
        group.perturb(&mut params, &mut rng, 0.01);
    }
    assert_sums_to_one(&params, "cooled perturbation");

    // With tiny cooling, the perturbation should be small
    for i in 0..3 {
        assert!((params[i] - original[i]).abs() < 0.05,
            "cooled perturbation changed frac[{}] too much: {} → {}",
            i, original[i], params[i]);
    }
}

#[test]
fn perturb_extreme_fractions() {
    // One fraction near 1, others near 0 — should not panic
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[0.1, 0.1, 0.1]),
    };
    let mut params = vec![0.999, 0.0005, 0.0005];
    let mut rng = rng();

    for i in 0..50 {
        group.perturb(&mut params, &mut rng, 1.0);
        assert_sums_to_one(&params, &format!("extreme perturbation {}", i));
        assert_all_positive(&params, &format!("extreme perturbation {}", i));
    }
}

#[test]
fn perturb_very_tiny_fractions() {
    // Fractions at 1e-6 scale — should not produce NaN
    let group = SimplexGroup {
        indices: vec![0, 1, 2, 3],
        sds: sds(&[0.01, 0.01, 0.01, 0.01]),
    };
    let mut params = vec![0.97, 0.02, 1e-6, 1e-6];
    // Normalize to sum to 1
    let sum: f64 = params.iter().sum();
    for p in &mut params { *p /= sum; }

    let mut rng = rng();
    for i in 0..50 {
        group.perturb(&mut params, &mut rng, 1.0);
        let sum: f64 = group.indices.iter().map(|&j| params[j]).sum();
        assert!((sum - 1.0).abs() < 1e-10,
            "tiny frac perturbation {}: sum = {}", i, sum);
        for &idx in &group.indices {
            assert!(params[idx].is_finite(),
                "tiny frac perturbation {}: NaN at index {}", i, idx);
        }
    }
}

#[test]
fn perturb_different_rw_sds() {
    // Different rw_sd per member — larger rw_sd should cause more movement
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[1.0, 0.01, 0.01]), // first member moves a lot
    };
    let start = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
    let mut rng = rng();

    let mut total_movement = [0.0; 3];
    let n_reps = 100;
    for _ in 0..n_reps {
        let mut params = start.clone();
        group.perturb(&mut params, &mut rng, 1.0);
        for i in 0..3 {
            total_movement[i] += (params[i] - start[i]).abs();
        }
    }

    let avg_0 = total_movement[0] / n_reps as f64;
    let avg_1 = total_movement[1] / n_reps as f64;

    // Member 0 (rw_sd=1.0) should move more than member 1 (rw_sd=0.01).
    // Softmax couples members, so the ratio won't match rw_sd ratio exactly,
    // but member 0 should still move noticeably more.
    assert!(avg_0 > avg_1,
        "expected member 0 to move more: avg_0={:.4}, avg_1={:.4}", avg_0, avg_1);
}

// ── Determinism ─────────────────────────────────────────────────────────

#[test]
fn perturb_deterministic_with_seed() {
    let group = SimplexGroup {
        indices: vec![0, 1, 2],
        sds: sds(&[0.1, 0.1, 0.1]),
    };

    let mut params1 = vec![0.5, 0.3, 0.2];
    let mut params2 = vec![0.5, 0.3, 0.2];
    let mut rng1 = StatefulRng::new(99);
    let mut rng2 = StatefulRng::new(99);

    group.perturb(&mut params1, &mut rng1, 0.5);
    group.perturb(&mut params2, &mut rng2, 0.5);

    for i in 0..3 {
        assert_eq!(params1[i], params2[i],
            "same seed should give same result: index {}", i);
    }
}

// ── Edge cases ──────────────────────────────────────────────────────────

#[test]
fn perturb_single_element_group() {
    // Degenerate: single-element simplex → always 1.0
    let group = SimplexGroup {
        indices: vec![0],
        sds: sds(&[0.1]),
    };
    let mut params = vec![1.0];
    let mut rng = rng();
    group.perturb(&mut params, &mut rng, 1.0);
    assert!((params[0] - 1.0).abs() < 1e-10, "single element should always be 1.0");
}

#[test]
fn perturb_with_extra_params() {
    // Simplex group is indices [1,2,3], but params vector has index 0 and 4 too
    let group = SimplexGroup {
        indices: vec![1, 2, 3],
        sds: sds(&[0.1, 0.1, 0.1]),
    };
    let mut params = vec![100.0, 0.5, 0.3, 0.2, 999.0];
    let mut rng = rng();

    group.perturb(&mut params, &mut rng, 1.0);

    // Non-group params should be untouched
    assert_eq!(params[0], 100.0, "non-group param 0 should be untouched");
    assert_eq!(params[4], 999.0, "non-group param 4 should be untouched");

    // Group params should sum to 1
    let group_sum: f64 = [params[1], params[2], params[3]].iter().sum();
    assert!((group_sum - 1.0).abs() < 1e-10,
        "group sum should be 1.0, got {}", group_sum);
}
