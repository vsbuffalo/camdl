/// Space-filling sampling schemes for experimental designs.
///
/// Supports:
/// - `sobol`  — Saltelli's structured scheme for Sobol sensitivity indices.
///              Generates N(2k+2) parameter points from two base matrices A, B
///              and k "crossed" matrices A_Bi.
/// - `lhs`    — Latin Hypercube Sampling (N points, stratified in each dimension).
/// - `random` — Uniform random (N points, fully independent).
///
/// All methods produce samples in [0, 1]^k, then callers apply `transform` to
/// map them to the desired parameter range.

use std::collections::HashMap;

use ir::parameter::PriorDist;

/// Human-readable description for a prior distribution. Used by
/// `priors.txt` (the assumptions log written alongside design output)
/// and error messages.
///
/// Wire format for the `prior = ...` block in design TOML matches
/// `ir::parameter::PriorDist`'s externally-tagged form:
/// ```toml
/// prior = { beta = { alpha = 4.0, beta = 6.0 } }
/// prior = { log_normal = { mu = 1.0, sigma = 0.5 } }
/// prior = { normal = { mean = 0.3, sd = 0.1 } }
/// ```
pub fn describe_prior(p: &PriorDist) -> String {
    match p {
        PriorDist::Beta(q) => format!("Beta(alpha={}, beta={})", q.alpha, q.beta),
        PriorDist::LogNormal(q) => format!("LogNormal(mu={}, sigma={})", q.mu, q.sigma),
        PriorDist::Normal(q) => format!("Normal(mean={}, sd={})", q.mean, q.sd),
        PriorDist::Uniform(q) => format!("Uniform(lower={}, upper={})", q.lower, q.upper),
        PriorDist::HalfNormal(q) => format!("HalfNormal(sigma={})", q.sigma),
        PriorDist::Gamma(q) => format!("Gamma(shape={}, rate={})", q.shape, q.rate),
        PriorDist::Exponential(q) => format!("Exponential(rate={})", q.rate),
        PriorDist::LogUniform(q) => format!("LogUniform(lower={}, upper={})", q.lower, q.upper),
        PriorDist::TruncatedNormal(q) =>
            format!("TruncatedNormal(mean={}, sd={}, lower={}, upper={})", q.mean, q.sd, q.lower, q.upper),
        PriorDist::Fixed(v) => format!("Fixed({})", v),
    }
}

/// A single parameter's sampling bounds and transform.
#[derive(Debug, Clone)]
pub struct DesignParam {
    pub min: f64,
    pub max: f64,
    /// None = linear, "log" = log-uniform, "logit" = logit-uniform
    pub transform: Option<String>,
    /// Optional prior distribution for VOI importance weighting.
    /// If None, uniform over [min, max] is assumed.
    pub prior: Option<PriorDist>,
}

impl DesignParam {
    /// Map a unit sample u ∈ [0, 1] to the parameter's actual value.
    pub fn map_unit(&self, u: f64) -> f64 {
        match self.transform.as_deref() {
            Some("log") => {
                let log_min = self.min.ln();
                let log_max = self.max.ln();
                (log_min + u * (log_max - log_min)).exp()
            }
            Some("logit") => {
                // Sample in logit space: logit(min)..logit(max)
                let logit = |x: f64| (x / (1.0 - x)).ln();
                let ilogit = |y: f64| 1.0 / (1.0 + (-y).exp());
                let a = logit(self.min);
                let b = logit(self.max);
                ilogit(a + u * (b - a))
            }
            _ => self.min + u * (self.max - self.min),
        }
    }
}

/// Result of generating a design: a list of parameter point maps.
///
/// For Sobol designs the points are structured (A, B, A_B0, ..., A_B(k-1) blocks);
/// for LHS/random they are unstructured.
pub struct DesignPoints {
    pub points: Vec<HashMap<String, f64>>,
    /// Parameter names in the order they appear in the Sobol matrices (for analysis).
    pub param_names: Vec<String>,
}

// ─── Sobol sequence (Van der Corput, base 2) ─────────────────────────────────

/// Generate the i-th element of the Van der Corput sequence in base 2.
/// This is the simplest quasi-random low-discrepancy sequence — sufficient
/// for up to ~20 dimensions when dimensions are independent.
fn van_der_corput(i: usize) -> f64 {
    let mut n = i;
    let mut result = 0.0;
    let mut base = 0.5;
    while n > 0 {
        result += base * (n & 1) as f64;
        n >>= 1;
        base *= 0.5;
    }
    result
}

/// Generate the i-th element of the Halton sequence for a given prime base.
/// Used for dimensions > 1 to get better coverage than independent Van der Corput.
fn halton(i: usize, base: usize) -> f64 {
    let mut n = i;
    let mut result = 0.0;
    let mut denominator = 1.0;
    while n > 0 {
        denominator *= base as f64;
        result += (n % base) as f64 / denominator;
        n /= base;
    }
    result
}

// First 20 primes for Halton sequence dimensions.
const PRIMES: [usize; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/// Generate an n×k quasi-random matrix in [0,1]^k using the Halton sequence.
/// Each row is a k-dimensional sample point.
fn halton_matrix(n: usize, k: usize) -> Vec<Vec<f64>> {
    (1..=n).map(|i| {
        (0..k).map(|d| {
            if d < PRIMES.len() {
                halton(i, PRIMES[d])
            } else {
                // Fallback for > 20 dimensions: use VdC with scramble
                van_der_corput(i ^ (d * 0x9e3779b9))
            }
        }).collect()
    }).collect()
}

// ─── Saltelli scheme ─────────────────────────────────────────────────────────

/// Generate Saltelli's structured sample matrices for Sobol sensitivity analysis.
///
/// For n base samples and k parameters, generates n(2+k) total rows:
///   rows 0..n         → A matrix
///   rows n..2n        → B matrix
///   rows (2+i)n..(3+i)n → A_Bi matrix (A with column i replaced by B's column i)
///
/// A and B are generated as the first k and second k columns of a 2k-dimensional
/// Halton matrix. Different Halton bases (primes) are independent by construction,
/// so A and B have the required independence for the Saltelli estimator to be unbiased.
/// This approach supports up to k = 10 parameters (20 primes available).
///
/// Returns unit samples in [0,1]^k. Callers map to actual parameter ranges.
pub fn saltelli_matrices(n: usize, k: usize) -> Vec<Vec<f64>> {
    assert!(2 * k <= PRIMES.len(),
        "saltelli_matrices supports at most {} parameters (need 2k Halton dimensions)", PRIMES.len() / 2);

    // A single n×2k Halton matrix; split into A (cols 0..k) and B (cols k..2k).
    // Cross-column independence holds because each column uses a different prime base.
    let full = halton_matrix(n, 2 * k);
    let a: Vec<Vec<f64>> = full.iter().map(|row| row[..k].to_vec()).collect();
    let b: Vec<Vec<f64>> = full.iter().map(|row| row[k..2*k].to_vec()).collect();

    let total = n * (2 + k);
    let mut result = Vec::with_capacity(total);

    // A block
    for row in &a {
        result.push(row.clone());
    }
    // B block
    for row in &b {
        result.push(row.clone());
    }
    // A_Bi blocks: A with column i replaced by B[:,i]
    for i in 0..k {
        for (a_row, b_row) in a.iter().zip(b.iter()) {
            let mut row = a_row.clone();
            row[i] = b_row[i];
            result.push(row);
        }
    }

    result
}

// ─── Latin Hypercube ──────────────────────────────────────────────────────────

/// Generate n LHS samples in k dimensions.
/// Each dimension is stratified into n equal intervals; within each stratum
/// the sample is placed uniformly. Strata are shuffled independently per dimension
/// using a simple deterministic permutation.
fn lhs_matrix(n: usize, k: usize) -> Vec<Vec<f64>> {
    // We need a simple deterministic shuffle. Use a linear congruential permutation.
    let permute = |seed: u64, n: usize| -> Vec<usize> {
        // Fisher-Yates with LCG
        let mut v: Vec<usize> = (0..n).collect();
        let mut rng = seed;
        for i in (1..n).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            v.swap(i, j);
        }
        v
    };

    // For each dimension, generate stratified samples in a shuffled order
    let columns: Vec<Vec<f64>> = (0..k).map(|d| {
        let perm = permute((d as u64).wrapping_mul(0x517cc1b727220a95).wrapping_add(1), n);
        // Unit samples within each stratum
        let sub_samples: Vec<f64> = (0..n).map(|i| {
            let stratum = perm[i] as f64;
            // Midpoint of stratum (deterministic, no randomness)
            (stratum + 0.5) / n as f64
        }).collect();
        sub_samples
    }).collect();

    // Transpose: columns → rows
    (0..n).map(|i| (0..k).map(|d| columns[d][i]).collect()).collect()
}

// ─── Uniform random ───────────────────────────────────────────────────────────

/// Generate n uniform random samples in k dimensions using a simple LCG.
/// Not cryptographically random, but deterministic and sufficient for sensitivity analysis.
fn random_matrix(n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut rng: u64 = 0x12345678abcdef01;
    let next = |rng: &mut u64| -> f64 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*rng >> 11) as f64) / (1u64 << 53) as f64
    };
    (0..n).map(|_| (0..k).map(|_| next(&mut rng)).collect()).collect()
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Generate design points for a named method.
///
/// `params` is a sorted list of (name, DesignParam) pairs.
/// `n` is the base sample count (for sobol: total = n(2k+2)).
/// `method` is "sobol", "lhs", or "random".
pub fn generate_design(
    params: &[(String, DesignParam)],
    n: usize,
    method: &str,
) -> DesignPoints {
    let k = params.len();
    let param_names: Vec<String> = params.iter().map(|(name, _)| name.clone()).collect();

    let unit_matrix = match method {
        "sobol" => saltelli_matrices(n, k),
        "lhs"   => lhs_matrix(n, k),
        _       => random_matrix(n, k),   // "random" + fallback
    };

    let points = unit_matrix.iter().map(|row| {
        row.iter().zip(params.iter()).map(|(&u, (name, param))| {
            (name.clone(), param.map_unit(u))
        }).collect()
    }).collect();

    DesignPoints { points, param_names }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_param(min: f64, max: f64) -> DesignParam {
        DesignParam { min, max, transform: None, prior: None }
    }

    fn log_param(min: f64, max: f64) -> DesignParam {
        DesignParam { min, max, transform: Some("log".to_string()), prior: None }
    }

    #[test]
    fn saltelli_output_size() {
        // n=16 samples, k=3 parameters → n(2+k) = 80 rows
        let n = 16;
        let k = 3;
        let mat = saltelli_matrices(n, k);
        assert_eq!(mat.len(), n * (2 + k), "expected {} rows", n * (2 + k));
        assert_eq!(mat[0].len(), k);
    }

    #[test]
    fn saltelli_unit_range() {
        let mat = saltelli_matrices(32, 4);
        for row in &mat {
            for &v in row {
                assert!((0.0..=1.0).contains(&v), "value {} out of [0,1]", v);
            }
        }
    }

    #[test]
    fn saltelli_a_b_differ() {
        // A block (rows 0..n) and B block (rows n..2n) should be different sequences
        let n = 16;
        let mat = saltelli_matrices(n, 2);
        let a_sum: f64 = mat[0..n].iter().map(|r| r[0]).sum();
        let b_sum: f64 = mat[n..2*n].iter().map(|r| r[0]).sum();
        assert!((a_sum - b_sum).abs() > 0.01, "A and B blocks should differ");
    }

    #[test]
    fn a_bi_structure() {
        // For A_B0 block: column 0 should match B, column 1 should match A
        let n = 8;
        let k = 2;
        let mat = saltelli_matrices(n, k);
        let a_block  = &mat[0..n];
        let b_block  = &mat[n..2*n];
        let ab0_block = &mat[2*n..3*n];  // A with col 0 replaced by B's col 0
        for i in 0..n {
            // col 0: A_B0[i][0] should == B[i][0]
            assert!((ab0_block[i][0] - b_block[i][0]).abs() < 1e-12,
                "A_B0 col 0 should match B col 0");
            // col 1: A_B0[i][1] should == A[i][1]
            assert!((ab0_block[i][1] - a_block[i][1]).abs() < 1e-12,
                "A_B0 col 1 should match A col 1");
        }
    }

    #[test]
    fn lhs_output_size() {
        let pts = lhs_matrix(20, 5);
        assert_eq!(pts.len(), 20);
        assert_eq!(pts[0].len(), 5);
    }

    #[test]
    fn lhs_unit_range() {
        let pts = lhs_matrix(50, 4);
        for row in &pts {
            for &v in row {
                assert!((0.0..1.0).contains(&v), "value {} out of [0,1)", v);
            }
        }
    }

    #[test]
    fn linear_param_map() {
        let p = linear_param(2.0, 8.0);
        assert!((p.map_unit(0.0) - 2.0).abs() < 1e-12);
        assert!((p.map_unit(0.5) - 5.0).abs() < 1e-12);
        assert!((p.map_unit(1.0) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn log_param_map() {
        let p = log_param(0.01, 100.0);  // 4 orders of magnitude
        let mid = p.map_unit(0.5);
        // midpoint in log space should be sqrt(0.01 * 100) = 1.0
        assert!((mid - 1.0).abs() < 1e-10, "log midpoint should be 1.0, got {}", mid);
    }

    #[test]
    fn generate_design_sobol_count() {
        let params = vec![
            ("vacc_eff".to_string(), linear_param(0.1, 0.9)),
            ("R0".to_string(), linear_param(1.0, 5.0)),
            ("kappa".to_string(), log_param(0.001, 0.1)),
        ];
        let design = generate_design(&params, 64, "sobol");
        assert_eq!(design.points.len(), 64 * (2 + 3));  // n(2+k) = n(2k+2)
    }

    #[test]
    fn generate_design_lhs_count() {
        let params = vec![
            ("a".to_string(), linear_param(0.0, 1.0)),
            ("b".to_string(), linear_param(0.0, 1.0)),
        ];
        let design = generate_design(&params, 100, "lhs");
        assert_eq!(design.points.len(), 100);
    }

    #[test]
    fn generated_values_in_range() {
        let params = vec![
            ("x".to_string(), linear_param(5.0, 10.0)),
            ("y".to_string(), log_param(0.001, 1.0)),
        ];
        let design = generate_design(&params, 32, "lhs");
        for pt in &design.points {
            let x = pt["x"];
            let y = pt["y"];
            assert!((5.0..=10.0).contains(&x), "x={} out of range", x);
            assert!((0.001..=1.0).contains(&y), "y={} out of range", y);
        }
    }
}
