//! Quantile reduction and numeric formatting — the shared substrate under every
//! banded artifact.
//!
//! These primitives are used by three consumers that have nothing to do with one
//! another: [`crate::quantity_output`] (generated quantities),
//! [`crate::fit::predict`] (posterior predictive bands) and
//! [`crate::fit::contrasts`] (counterfactual differences). They lived in
//! `fit/predict.rs` and were imported *by* the shared renderer — a consumer
//! owning the substrate its siblings depend on. Moving them to a neutral module
//! removes the inversion rather than relocating it.
//!
//! Proposal: `docs/dev/proposals/2026-08-11-scenario-banding-in-simulate.md` §3.6.

/// The default quantile levels and their column labels. A small fixed set —
/// `fill_between` wants columns, not a long-format `quantile` key.
pub const QUANTILE_LEVELS: &[(f64, &str)] =
    &[(0.05, "q05"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.95, "q95")];

/// Which random quantity of an observation row a band or a sample is of: the
/// `quantity` column of every prediction table and of `filtered_streams.tsv`.
///
/// The two have the same mean. The observation's variance is the
/// expectation's plus the mean of the observation model's own variance, so an
/// expectation band is narrower by the observation noise, and observed data
/// are compared with an observation band only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantity {
    /// `y`, a draw from the observation model: its spread carries the
    /// parameters, the latent state and the observation noise.
    Observation,
    /// `E[y | x, θ]`, the observation model's mean at one state and one
    /// parameter vector, with no observation noise drawn.
    Expectation,
}

impl Quantity {
    /// The value written into the `quantity` column.
    pub fn as_str(self) -> &'static str {
        match self {
            Quantity::Observation => "observation",
            Quantity::Expectation => "expectation",
        }
    }
}

// ── Pure numerics: the quantile reduction ──────────────────────────────────

/// Linear-interpolated quantile of `xs` at `q ∈ [0, 1]` (the numpy/`type-7`
/// rule). `xs` need not be sorted; a copy is sorted. Empty → `NaN`.
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

/// The full quantile band (`QUANTILE_LEVELS`) of one cell's draws. Rejects a
/// non-finite sample (a `NaN`/±∞ `y_rep` is an upstream bug, not a band to
/// publish) so a quietly-wrong band can never reach the artifact.
pub fn band(xs: &[f64]) -> Result<Vec<f64>, String> {
    if xs.iter().any(|x| !x.is_finite()) {
        return Err(format!(
            "non-finite predictive sample ({} draws, {} non-finite) — refusing to \
             quantile a NaN/±∞ y_rep",
            xs.len(),
            xs.iter().filter(|x| !x.is_finite()).count(),
        ));
    }
    Ok(QUANTILE_LEVELS.iter().map(|(q, _)| quantile(xs, *q)).collect())
}

// ── Numeric formatting ─────────────────────────────────────────────────────

/// Format a time: integral times as integers (`7`), else minimal decimal.
pub(crate) fn fmt_time(t: f64) -> String {
    if t.fract() == 0.0 && t.abs() < 1e15 {
        format!("{}", t as i64)
    } else {
        format!("{t}")
    }
}

/// Format a value: integral values as integers (count data reads cleanly),
/// else a decimal trimmed to ≤6 places (so an interpolated count quantile is
/// `204.9`, not `204.89999999999995`).
pub(crate) fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_linear_interpolation_matches_numpy() {
        // numpy.quantile([0,1,2,3,4], q) with default 'linear'.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&xs, 0.0), 0.0);
        assert_eq!(quantile(&xs, 1.0), 4.0);
        assert_eq!(quantile(&xs, 0.5), 2.0);
        assert_eq!(quantile(&xs, 0.25), 1.0);
        assert!((quantile(&xs, 0.05) - 0.2).abs() < 1e-12);
        assert!((quantile(&xs, 0.95) - 3.8).abs() < 1e-12);
    }

    #[test]
    fn quantile_sorts_unsorted_input_and_handles_edges() {
        assert!(quantile(&[], 0.5).is_nan());
        assert_eq!(quantile(&[7.0], 0.5), 7.0, "single value is its own quantile");
        // Unsorted input gives the same answer as sorted.
        assert_eq!(quantile(&[4.0, 0.0, 2.0, 1.0, 3.0], 0.5), 2.0);
    }

    #[test]
    fn band_returns_all_five_levels_in_order() {
        let xs: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let b = band(&xs).unwrap();
        assert_eq!(b.len(), 5);
        assert_eq!(b, vec![5.0, 25.0, 50.0, 75.0, 95.0]);
    }

    #[test]
    fn band_rejects_non_finite_samples() {
        // A NaN/±∞ y_rep is an upstream bug, not a band to publish.
        assert!(band(&[1.0, f64::NAN, 3.0]).is_err());
        assert!(band(&[1.0, f64::INFINITY]).is_err());
        assert!(band(&[1.0, 2.0, 3.0]).is_ok());
    }
}
