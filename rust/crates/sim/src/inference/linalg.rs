//! Shared linear algebra utilities for inference.

/// Cholesky decomposition: A = L L^T.
///
/// Returns `L` (lower triangular, row-major) if `a` is positive definite, and
/// otherwise the first pivot that was not positive.
pub fn cholesky_lower(a: &[f64], d: usize) -> Result<Vec<f64>, NotPositiveDefinite> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let diag = a[i * d + i] - sum;
                if diag <= 0.0 {
                    return Err(NotPositiveDefinite { index: i, diag });
                }
                l[i * d + j] = diag.sqrt();
            } else {
                l[i * d + j] = (a[i * d + j] - sum) / l[j * d + j];
            }
        }
    }
    Ok(l)
}

/// The pivot at which [`cholesky_lower`] found its matrix not positive
/// definite: the diagonal entry `index`, at `diag` after subtracting the
/// factor's earlier columns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotPositiveDefinite {
    pub index: usize,
    pub diag: f64,
}

impl std::fmt::Display for NotPositiveDefinite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cholesky_lower: matrix not positive definite \
             (diag[{}] = {:.6e} after subtraction)",
            self.index, self.diag,
        )
    }
}
