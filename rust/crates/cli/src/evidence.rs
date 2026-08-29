//! Evidence formatting — decibans alongside nats for model-comparison output.
//!
//! See `docs/dev/proposals/2026-04-23-evidence-in-decibans.md` for the full
//! rationale. Short version: model-comparison output should display
//! log-likelihood *differences* in decibans (dB, base-10 log-likelihood ratio
//! × 10) with the Jeffreys qualitative label, alongside the raw nats. This
//! captures the "evidence scale" framing Turing + Good + Jeffreys developed
//! and that Kass & Raftery (1995) use, while preserving nats as the primary
//! machine-readable unit for interop with pomp / Stan / NumPyro pipelines.
//!
//! Rule: dB appears only for *differences* between log-likelihoods (where the
//! Jacobian cancels and the value is a scale-free log-likelihood ratio), not
//! for raw absolute log-likelihoods (whose additive constant is arbitrary).
//!
//! Second rule: a tier is attached only to a difference that exceeds twice its
//! paired standard error ([`within_noise`]). The scale calibrates *magnitude*
//! and knows nothing about *resolution*, so an ungated tier can name a
//! magnitude the data cannot resolve — the same 6 nats reads "strong" whether
//! its se(Δ) is 0.3 or 30.
//!
//! Evidence scale (labels and thresholds). Attribution is split, and the
//! split matters — do not move labels without updating the citations:
//!
//!   0 – 5 dB    indeterminate   (odds 1:1 to 3:1)
//!   5 – 10 dB   substantial     (odds 3:1 to 10:1)
//!   10 – 15 dB  strong          (odds 10:1 to ~30:1)
//!   15 – 20 dB  very strong     (odds ~30:1 to 100:1)
//!   20 – 40 dB  decisive        (odds 100:1 to 10⁴:1)
//!   40+ dB      overwhelming    (odds > 10⁴:1)
//!
//! Tiers 1–5 (0/5/10/15/20 dB breakpoints) are **Jeffreys 1961**,
//! *Theory of Probability*, Appendix B. The Jeffreys original has
//! five tiers and the top one ("decisive") is unbounded: anything
//! above 20 dB is "decisive" in the historical scale.
//!
//! The 20–40 dB / 40+ dB split of Jeffreys' unbounded top tier into
//! "decisive" and "overwhelming" is a **camdl pedagogical
//! extension**, not Jeffreys' or anyone else's. The motivation is
//! that epi model comparisons on multi-year weekly data routinely
//! produce log-likelihood differences in the thousands of decibans —
//! Jeffreys' "decisive, 20 dB and up" qualitatively collapses the
//! "borderline significant" regime with the "10^150 times more
//! likely" regime, which teaches nothing about relative magnitude.
//! The 40 dB break is where cross-validation evidence ratios start
//! exceeding 10⁴:1 (log₁₀ BF > 4), which for typical epi likelihood
//! surfaces marks the point where "you have to work hard to
//! dismiss this" becomes "even an adversarial reviewer can't
//! reasonably dismiss this."
//!
//! Good (1950), *Probability and the Weighing of Evidence*, is the
//! primary source for decibans as a unit and the weight-of-evidence
//! framing; Jaynes (*Probability Theory: The Logic of Science*,
//! ch. 4) uses decibans extensively but does **not** publish a
//! labeled tier scale. Any attribution of "overwhelming" to Jaynes
//! is incorrect — the tier is camdl's own extension of Jeffreys.
//!
//! Kass & Raftery (1995, JASA) is the modern alternative: four
//! tiers at 2 ln(BF) thresholds of 2/6/10, which correspond to
//! 8.7/26.1/43.4 dB — not 5-dB boundaries, so the two scales are
//! not interchangeable without conversion.

/// Nats → decibans: 1 nat ≈ 4.342944819 dB.
pub const NATS_TO_DB: f64 = 10.0 / std::f64::consts::LN_10;

/// Sample standard deviation (N-1 denominator). Returns 0.0 for len<2.
pub fn sample_sd(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

/// `logmeanexp` plus a delta-method standard error of the combined
/// estimate. Returns `(log(mean(exp(xs))), SE)`.
///
/// Derivation: with `Y = log(M⁻¹ Σ exp(X_i))` and `L_i = exp(X_i)`,
/// `Y = log(L̄)`. By the delta method (d/dL log L = 1/L),
/// `Var(Y) ≈ Var(L̄) / L̄² = Var(L_i) / (M · L̄²)`, so
/// `SE(Y) = SD(L_i) / (L̄ · √M)`. Numerically stable form (max-shift):
/// the `exp(m)` factor cancels between SD and L̄.
///
/// This is the correct SE for log-mean-exp combining of replicate
/// particle-filter log-likelihoods (matches pomp's `pfilter` output;
/// see Ionides et al. 2015 PNAS supplement and the pomp manual on
/// `logmeanexp`). The naïve `sd(xs)/√M` is only correct for the
/// *arithmetic* mean of the logs, not for log-of-arithmetic-mean of
/// likelihoods, and underestimates `Var(Y)` whenever per-replicate
/// variance is non-trivial.
///
/// Returns:
/// - `(NEG_INFINITY, NaN)` for empty input.
/// - `(value, 0.0)` for a single-element input.
/// - `(value, NaN)` if the input contains NaN.
pub fn logmeanexp_with_se(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (f64::NEG_INFINITY, f64::NAN);
    }
    if xs.len() == 1 {
        return (xs[0], 0.0);
    }
    if xs.iter().any(|x| x.is_nan()) {
        return (f64::NAN, f64::NAN);
    }
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !m.is_finite() {
        // All -inf, or +inf in input: SE is undefined.
        return (m, f64::NAN);
    }
    let shifted: Vec<f64> = xs.iter().map(|x| (x - m).exp()).collect();
    let mean_shifted: f64 = shifted.iter().sum::<f64>() / xs.len() as f64;
    let logmeanexp_val = m + mean_shifted.ln();
    // Delta-method SE on the log scale: SD(L_i) / (L̄ · √M). The
    // exp(m) factor cancels between SD and L̄, so we compute SD on
    // the shifted values directly.
    let sd_shifted = sample_sd(&shifted);
    let se = sd_shifted / (mean_shifted * (xs.len() as f64).sqrt());
    (logmeanexp_val, se)
}

/// Jeffreys qualitative label for an evidence magnitude in decibans. The
/// input is the *signed* Δlog-lik in dB; the label describes the magnitude,
/// and the caller decides whether to prefix with "for" / "against" based on
/// the sign.
pub fn jeffreys_label(db: f64) -> &'static str {
    let m = db.abs();
    if      m <  5.0 { "indeterminate"  }
    else if m < 10.0 { "substantial"    }
    else if m < 15.0 { "strong"         }
    else if m < 20.0 { "very strong"    }
    else if m < 40.0 { "decisive"       }
    else             { "overwhelming"   }
}

/// Is a paired difference inside its own noise band, `|Δ| < 2·se(Δ)`?
///
/// `Δelpd` is a sum over the scored observations and `se(Δ)` is the paired
/// standard error of that sum, so `|Δ| < 2·se` says the difference is smaller
/// than the observation-to-observation scatter it was built from. A Jeffreys
/// word on such a difference reports a magnitude the data cannot resolve — the
/// tier is computed from `Δ` alone and knows nothing about its spread, so
/// without this gate a Δ of 6 nats reads "strong" whether its se is 0.3 or 30.
///
/// The two-SE cut is the conventional descriptive band, not a test: `se(Δ)`
/// itself rests on a normal approximation that is unreliable at small `T` and
/// when the models are similar (Sivula, Magnusson & Vehtari, arXiv:2008.10296).
/// It is used here only to decline to label, never to declare a winner.
///
/// A non-finite or zero `se` cannot establish a band, so it does not gate: the
/// answer is `false` and the tier is shown.
pub fn within_noise(delta_nats: f64, se_nats: f64) -> bool {
    se_nats.is_finite() && se_nats > 0.0 && delta_nats.abs() < 2.0 * se_nats
}

/// The filter-noise gate on the evidence verdict (Stage 4.2 of the
/// 2026-08-29 proposal): with replicate derives, each row's elpd carries a
/// Monte-Carlo SE from the particle filter's own randomness, and a `Δ`
/// within twice the pair's combined MC SE is evidence about the filters,
/// not the models — the PF log-likelihood estimator's noise (and its
/// model-dependent bias, Bérard–Del Moral–Doucet 2014) lives at exactly
/// this scale. `mc_se_delta` is `√(mc_a² + mc_b²)`; conservative under
/// common random numbers (shared seeds correlate the two errors, which
/// shrinks the true SE of the difference). `None` (no replicates) does
/// not gate.
pub fn within_filter_noise(delta_nats: f64, mc_se_delta: Option<f64>) -> bool {
    match mc_se_delta {
        Some(mc) => mc.is_finite() && mc > 0.0 && delta_nats.abs() < 2.0 * mc,
        None => false,
    }
}

/// The machine-readable evidence label for a Δlog-lik (nats), its paired
/// standard error, and (with replicate derives) the pair's combined
/// Monte-Carlo SE: `within_noise` when [`within_noise`] holds,
/// `within_filter_noise` when only the MC gate trips, else the magnitude
/// tier from [`jeffreys_label`] (direction is carried by the sign of
/// `delta_elpd`, which is reported alongside).
pub fn evidence_label(delta_nats: f64, se_nats: f64, mc_se_delta: Option<f64>) -> &'static str {
    if within_noise(delta_nats, se_nats) {
        "within_noise"
    } else if within_filter_noise(delta_nats, mc_se_delta) {
        "within_filter_noise"
    } else {
        jeffreys_label(delta_nats * NATS_TO_DB)
    }
}

/// The display cell for a Δlog-lik (nats) and its paired standard error:
/// the difference in decibans, then what may honestly be said about it.
///
/// The dB number is always shown — it is just a unit change on `Δ`. The word
/// after it is gated on the SE: inside the noise band the cell says
/// `within noise` and no tier, because the tier would name a magnitude the
/// data cannot resolve.
///
/// Outside the band the cell carries direction explicitly via a `for` /
/// `against` suffix. This closes a real readability bug — pre-fix,
/// `-32.3 dB, decisive` could be misread as "decisive evidence supporting this
/// model" when the negative sign actually means decisive evidence *against* it
/// (the baseline outscored it). The "indeterminate" tier keeps no direction
/// (the whole point of that tier is "we can't commit").
///
/// Examples:
/// - `evidence_cell(+27.3, 1.0)` → `"+118.6 dB, decisive for"`
/// - `evidence_cell(-7.45, 0.5, None)` → `"-32.4 dB, decisive against"`
/// - `evidence_cell(+0.5, 1.0)`  → `"+2.2 dB, within noise"`
pub fn evidence_cell(delta_nats: f64, se_nats: f64, mc_se_delta: Option<f64>) -> String {
    if !delta_nats.is_finite() {
        return "—".into();
    }
    let db = delta_nats * NATS_TO_DB;
    if within_noise(delta_nats, se_nats) {
        return format!("{:+.1} dB, within noise", db);
    }
    if within_filter_noise(delta_nats, mc_se_delta) {
        return format!("{:+.1} dB, within filter noise", db);
    }
    let tag = jeffreys_label(db);
    let labeled = if tag == "indeterminate" {
        // Below the substantial-evidence threshold the data don't
        // pick a side, so adding "for"/"against" would dress up
        // noise as a direction.
        tag.to_string()
    } else if db > 0.0 {
        format!("{} for", tag)
    } else {
        format!("{} against", tag)
    };
    format!("{:+.1} dB, {}", db, labeled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_factor_matches_10_over_ln10() {
        // 1 nat = 10 / ln(10) ≈ 4.342944819 dB.
        assert!((NATS_TO_DB - 4.342944819).abs() < 1e-9);
    }

    #[test]
    fn jeffreys_boundaries_symmetric() {
        // Each boundary is open on the lower side, closed on the upper:
        // < 5 → indeterminate; [5, 10) → substantial; etc.
        assert_eq!(jeffreys_label(0.0),     "indeterminate");
        assert_eq!(jeffreys_label(4.99),    "indeterminate");
        assert_eq!(jeffreys_label(5.0),     "substantial");
        assert_eq!(jeffreys_label(9.99),    "substantial");
        assert_eq!(jeffreys_label(10.0),    "strong");
        assert_eq!(jeffreys_label(14.99),   "strong");
        assert_eq!(jeffreys_label(15.0),    "very strong");
        assert_eq!(jeffreys_label(19.99),   "very strong");
        assert_eq!(jeffreys_label(20.0),    "decisive");
        assert_eq!(jeffreys_label(39.99),   "decisive");
        assert_eq!(jeffreys_label(40.0),    "overwhelming");
        assert_eq!(jeffreys_label(1000.0),  "overwhelming");
    }

    #[test]
    fn jeffreys_label_is_symmetric_in_sign() {
        // The label names magnitude only; the sign tells you which direction.
        for db in [6.0, 12.0, 17.0, 25.0, 50.0] {
            assert_eq!(jeffreys_label(db), jeffreys_label(-db),
                "label must be sign-symmetric at {} dB", db);
        }
    }

    /// Stage 4.2: the filter-noise gate. A Δ inside twice the pair's
    /// combined MC SE gets `within filter noise` — after the pairwise-SE
    /// gate (which wins when both trip), and only when replicates supplied
    /// an MC SE at all.
    #[test]
    fn filter_noise_gate_suppresses_the_verdict() {
        // Δ = 3 nats, tight pairwise se, but MC SE of 2 → filter noise.
        assert!(within_filter_noise(3.0, Some(2.0)));
        assert_eq!(evidence_label(3.0, 0.5, Some(2.0)), "within_filter_noise");
        assert!(evidence_cell(3.0, 0.5, Some(2.0)).contains("within filter noise"));
        // Small MC SE does not gate a large Δ.
        assert_eq!(evidence_label(3.0, 0.5, Some(0.1)), "strong");
        // No replication → no gate.
        assert!(!within_filter_noise(3.0, None));
        assert_eq!(evidence_label(3.0, 0.5, None), "strong");
        // The pairwise gate takes precedence when both trip.
        assert_eq!(evidence_label(3.0, 2.0, Some(2.0)), "within_noise");
        // Degenerate MC SE (NaN/0) does not gate.
        assert!(!within_filter_noise(3.0, Some(f64::NAN)));
        assert!(!within_filter_noise(3.0, Some(0.0)));
    }

    #[test]
    fn evidence_cell_positive_carries_for() {
        // Positive Δ → candidate model beats the baseline → "for".
        // se = 0.5 ⇒ the band is ±1.0 nats and 5.5 is well outside it.
        assert_eq!(evidence_cell(5.5, 0.5, None), "+23.9 dB, decisive for");
    }

    #[test]
    fn evidence_cell_negative_carries_against() {
        // Negative Δ → baseline beats the candidate → "against".
        // The motivating bug: pre-fix this read "−32.3 dB, decisive"
        // and a reader could miss the sign and conclude the model
        // was preferred.
        let cell = evidence_cell(-7.45, 0.5, None);
        // -7.45 nats × 4.342944819 ≈ -32.35 dB → rounds to -32.4.
        assert!(cell.starts_with("-32.4 dB"), "expected -32.4 dB prefix, got {}", cell);
        assert!(cell.ends_with("decisive against"),
            "expected `decisive against` suffix, got {}", cell);
    }

    #[test]
    fn evidence_cell_indeterminate_carries_no_direction() {
        // Below |5 dB| and outside the noise band, the tier is
        // "indeterminate" — we explicitly refuse to commit to a
        // direction, so adding for/against would dress up noise as
        // a finding.
        let cell = evidence_cell(0.5, 0.1, None);  // ≈ 2.2 dB, band ±0.2
        assert!(cell.contains("indeterminate"), "{cell}");
        assert!(!cell.contains("for"));
        assert!(!cell.contains("against"));
        // Same for negative-but-still-indeterminate.
        let cell = evidence_cell(-0.5, 0.1, None);
        assert!(cell.contains("indeterminate"), "{cell}");
        assert!(!cell.contains("for"));
        assert!(!cell.contains("against"));
    }

    /// The SE gate: the same Δ is a tier against a small SE and `within noise`
    /// against a large one. The dB number is unchanged — only what may be said
    /// about it changes.
    #[test]
    fn evidence_cell_is_gated_by_the_paired_se() {
        assert_eq!(evidence_cell(3.0, 0.5, None), "+13.0 dB, strong for");
        assert_eq!(evidence_cell(3.0, 2.0, None), "+13.0 dB, within noise");
        assert_eq!(evidence_label(3.0, 0.5, None), "strong");
        assert_eq!(evidence_label(3.0, 2.0, None), "within_noise");
        // The boundary is closed on the tier side: |Δ| = 2·se is not "within".
        assert!(!within_noise(3.0, 1.5));
        assert!(within_noise(3.0, 1.500001));
        // An SE that cannot describe a band does not gate.
        assert!(!within_noise(0.0, 0.0));
        assert!(!within_noise(0.1, f64::NAN));
        assert_eq!(evidence_label(0.1, f64::NAN, None), "indeterminate");
    }

    #[test]
    fn evidence_cell_handles_non_finite() {
        // NaN / ±∞ → "—" in the dB column, not a panic.
        assert_eq!(evidence_cell(f64::NAN, 1.0, None), "—");
        assert_eq!(evidence_cell(f64::INFINITY, 1.0, None), "—");
    }

    #[test]
    fn sample_sd_basic() {
        assert_eq!(sample_sd(&[]), 0.0);
        assert_eq!(sample_sd(&[5.0]), 0.0);
        // sd of [1, 2, 3, 4, 5] is sqrt(2.5) ≈ 1.5811 with N-1 denom.
        let got = sample_sd(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((got - 2.5f64.sqrt()).abs() < 1e-9, "got {}", got);
    }

    #[test]
    fn logmeanexp_with_se_constant_input_zero_se() {
        // Identical replicates → SE = 0 (no per-rep spread).
        let (val, se) = logmeanexp_with_se(&[-7.3, -7.3, -7.3, -7.3]);
        assert!((val - (-7.3)).abs() < 1e-12);
        assert!(se.abs() < 1e-12, "constant input should give SE = 0, got {}", se);
    }

    #[test]
    fn logmeanexp_with_se_is_numerically_stable() {
        // Naive exp of -1e6 underflows to 0; stable form must not.
        // log((e^-1e6 + e^0)/2) = log(1/2) ≈ -0.6931 (the -1e6 term
        // is negligible). Naive implementation would return -inf.
        let (val, _) = logmeanexp_with_se(&[-1e6, 0.0]);
        assert!((val - (-(2.0f64).ln())).abs() < 1e-9, "got {}", val);
    }

    #[test]
    fn logmeanexp_with_se_matches_pf_bias_example() {
        // Two PF reps ℓ = [-100, -98]. logmeanexp > arithmetic mean
        // (Jensen): log((e^-100 + e^-98)/2) = -98 + log((e^-2+1)/2).
        let xs = [-100.0, -98.0];
        let (val, _) = logmeanexp_with_se(&xs);
        let arith = (xs[0] + xs[1]) / 2.0;
        assert!(val > arith,
            "logmeanexp {} should exceed arithmetic mean {}", val, arith);
        let expected = -98.0 + ((1.0 + (-2f64).exp()) / 2.0).ln();
        assert!((val - expected).abs() < 1e-9, "got {}, expected {}", val, expected);
    }

    #[test]
    fn logmeanexp_with_se_known_two_replicate_case() {
        // Two replicates ℓ = [-100, -98]. After max-shift (m = -98):
        //   shifted = [exp(-2), 1] ≈ [0.13534, 1.0]
        //   mean_shifted = 0.56767
        //   logmeanexp = -98 + ln(0.56767) ≈ -98.5662
        // SE = sd(shifted) / (mean_shifted * sqrt(M)).
        let xs = [-100.0, -98.0];
        let (val, se) = logmeanexp_with_se(&xs);
        let expected_val = -98.0 + ((1.0 + (-2f64).exp()) / 2.0).ln();
        assert!((val - expected_val).abs() < 1e-9);

        let m = -98.0_f64;
        let shifted = [(-100.0_f64 - m).exp(), (-98.0_f64 - m).exp()];
        let mean_shifted: f64 = (shifted[0] + shifted[1]) / 2.0;
        let sd_shifted = sample_sd(&shifted);
        let expected_se = sd_shifted / (mean_shifted * 2f64.sqrt());
        assert!((se - expected_se).abs() < 1e-9, "got {}, expected {}", se, expected_se);
    }

    #[test]
    fn logmeanexp_with_se_matches_naive_in_small_variance_limit() {
        // When per-rep variance is small (σ ≪ 1 nat), the delta-method
        // SE of logmeanexp converges to sd(x)/√M to first order. This is
        // the regime users typically have post-IF2 (per-rep PF noise of
        // a fraction of a nat). The two SEs should agree closely.
        let xs = [-100.0, -100.05, -99.95, -100.0, -100.0];
        let (_, se) = logmeanexp_with_se(&xs);
        let naive = sample_sd(&xs) / (xs.len() as f64).sqrt();
        let rel_diff = (se - naive).abs() / naive;
        assert!(rel_diff < 0.05,
            "delta-method SE ({}) and naïve SE ({}) should agree within 5% \
             in small-variance regime (rel diff {})",
            se, naive, rel_diff);
    }

    #[test]
    fn logmeanexp_with_se_diverges_from_naive_when_variance_large() {
        // When per-rep variance is non-trivial (≥ 1 nat), the delta-
        // method SE diverges from sd(x)/√M. Direction is regime-
        // dependent (smaller for samples where a few replicates dominate
        // L̄; larger in symmetric-lognormal regimes). The point is that
        // the two formulas are NOT interchangeable — using sd(x)/√M as
        // the SE of logmeanexp is wrong outside the small-variance
        // limit. Just assert they disagree by > 5% so a future regression
        // would catch a fall-back to the naive formula.
        let xs = [-105.0, -100.0, -95.0];
        let (_, se) = logmeanexp_with_se(&xs);
        let naive = sample_sd(&xs) / (xs.len() as f64).sqrt();
        let rel_diff = (se - naive).abs() / naive;
        assert!(rel_diff > 0.05,
            "delta-method SE ({}) and naïve SE ({}) should disagree noticeably \
             outside the small-variance regime (rel diff {})",
            se, naive, rel_diff);
    }

    #[test]
    fn logmeanexp_with_se_edge_cases() {
        // Empty input: value is -inf, SE is NaN.
        let (v, s) = logmeanexp_with_se(&[]);
        assert!(v.is_infinite() && v.is_sign_negative());
        assert!(s.is_nan());
        // Single-element input: SE is exactly 0.
        let (v, s) = logmeanexp_with_se(&[5.0]);
        assert!((v - 5.0).abs() < 1e-12);
        assert!(s.abs() < 1e-12);
        // NaN propagates to both fields.
        let (v, s) = logmeanexp_with_se(&[f64::NAN, 0.0]);
        assert!(v.is_nan() && s.is_nan());
        // All -inf: value is -inf, SE is NaN (undefined).
        let (v, s) = logmeanexp_with_se(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert!(v.is_infinite() && v.is_sign_negative());
        assert!(s.is_nan());
    }

    #[test]
    fn boundary_evidence_from_proposal() {
        // Spot-check the Jeffreys table against the proposal's worked values,
        // each against an SE small enough that the noise gate does not fire.
        // 20 nats × 4.343 ≈ 86.9 dB, "overwhelming" (above 40 dB).
        assert!(evidence_cell(20.0, 1.0, None).contains("overwhelming"));
        // Small gap (0.5 nats ≈ 2.2 dB) should be indeterminate — noise floor.
        assert!(evidence_cell(0.5, 0.1, None).contains("indeterminate"));
        // 3-nat gap (~13 dB) should be "strong" — a few nats of difference
        // on a weekly-obs fit is already beyond anecdotal.
        assert!(evidence_cell(3.0, 0.5, None).contains("strong"));
    }
}
