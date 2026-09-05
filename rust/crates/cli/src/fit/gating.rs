//! Refine-stage convergence gates.
//!
//! Two gates protect against the "refine launders an unconverged
//! scout" failure mode documented in
//! `docs/dev/proposals/2026-04-19-refine-gates-scout-convergence.md`:
//!
//! - Gate 1 (pre-refine): scout's tail chain-agreement (Â) on every
//!   structural estimated parameter (every one not declared
//!   `perturb_only_at_t0`) must be below `gate.a_thresh`. If it isn't,
//!   refine refuses to start. Overridable via `--allow-nonconverged-scout`.
//!
//! - Gate 2 (post-refine): refine's best loglik must not regress
//!   below scout's by more than a tolerance ε. If it does, refine's
//!   output is rejected — this is a near-certain bug in the run
//!   itself, not a statistical choice, so there's no override.
//!
//! Both gates produce actionable error messages that name the failing
//! values AND suggest fixes.

use super::config_v2::GateConfig;
use super::state::FitState;
use crate::evidence::NATS_TO_DB;

/// Soft threshold: params between this and `gate.a_thresh` get a
/// prominent warning but refine still runs. When `gate.a_thresh ≤
/// A_SOFT` the SoftWarn band is empty (every above-soft Â also
/// exceeds the hard gate). Matches the existing scout diagnostic
/// colour-coding (red ≥ a_thresh, yellow A_SOFT..a_thresh,
/// green < A_SOFT).
pub const A_SOFT: f64 = 1.05;

/// Where one parameter's chain-agreement Â sits relative to **the gate that
/// will actually be applied to it**.
///
/// Derived from `GateConfig::a_thresh`, not from a literal, because every
/// renderer that spelled the band as a constant drifted from the gate the
/// moment `a_thresh` moved off that constant — and the default did move, to
/// 1.01. One `fit summary` printed `max Â = 1.030 ✗ (threshold 1.01)` in the
/// gate block and `Â=1.030 ✓` in the parameter table twenty lines below.
///
/// This is the **glyph** band and nothing else. The diagnostic severity ladder
/// — whether a finding is a warning or an error — is a different question with
/// a different answer (`DiagnosticKind::severity`), and merging the two would
/// make "the gate refuses this" and "this is an error-level finding" one knob
/// when they are two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgreementBand {
    /// Below [`A_SOFT`]: nothing to say.
    Pass,
    /// In `[A_SOFT, a_thresh)`: prominent warning, but refine still runs.
    /// Empty whenever `a_thresh <= A_SOFT`, which the 1.01 default is.
    SoftWarn,
    /// At or above `a_thresh` — the comparison
    /// [`check_scout_convergence`] refuses on.
    Fail,
    /// Â is not defined for this parameter: the within-chain variance `W`
    /// collapsed relative to the parameter's scale, so Â would diverge with no
    /// diagnostic meaning and `compute_chain_agreement` returns `NaN` (gh#45).
    ///
    /// **Not** a failure. Every comparison against `NaN` is false, so a band
    /// written as an if/else-if chain of `<` falls through to the failure arm
    /// and reports a parameter that was never assessed as one that failed. The
    /// compound gate's Δ_dB leg carries the verdict for these.
    NotAssessed,
}

impl AgreementBand {
    /// The plain glyph. Colour is the renderer's business — `fit summary`
    /// paints through its own `ok`/`warn`/`err`, the stage report through raw
    /// ANSI — so this returns the character alone.
    pub fn glyph(self) -> &'static str {
        match self {
            Self::Pass => "✓",
            Self::SoftWarn => "~",
            Self::Fail => "✗",
            Self::NotAssessed => "n/a",
        }
    }
}

impl GateConfig {
    /// Which band one parameter's Â falls in, under **this** gate.
    ///
    /// The single authority for the ✓/~/✗ on an Â, so a glyph cannot disagree
    /// with the verdict printed beside it. [`check_scout_convergence`] reduces
    /// through the same ladder.
    pub fn a_band(&self, a: f64) -> AgreementBand {
        if !a.is_finite() {
            AgreementBand::NotAssessed
        } else if a >= self.a_thresh {
            AgreementBand::Fail
        } else if a >= A_SOFT {
            AgreementBand::SoftWarn
        } else {
            AgreementBand::Pass
        }
    }
}

/// Minimum ε for Gate 2. Scout's noise floor on a typical PF-based
/// loglik estimator at reasonable particle counts. `epsilon` takes the
/// max of this and `2 * σ_scout_chains` so multi-modal scout runs (high
/// between-chain σ) get a proportionally wider tolerance.
pub const LOGLIK_EPSILON_MIN: f64 = 3.0;

/// Verdict from the pre-refine convergence check. `SoftWarn` callers
/// should print the named parameters prominently. `Hard` and
/// `DecibansSpread` callers should error unless the user passed
/// `--allow-nonconverged-scout`, in which case downgrade to a warning.
#[derive(Debug)]
pub enum ScoutGateVerdict {
    Ok,
    SoftWarn { param_agreement: Vec<(String, f64)> },
    Hard {
        /// All structural params with Â ≥ `gate.a_thresh`. Named and
        /// sorted worst-first so the error message leads with the
        /// most obvious failure.
        failing: Vec<(String, f64)>,
        /// Every structural Â, for the full diagnostic table.
        all_structural: Vec<(String, f64)>,
        /// Â for the `perturb_only_at_t0` (initial-state) params —
        /// reported but not gated.
        perturb_only_at_t0: Vec<(String, f64)>,
        /// Spread across scout's per-chain final logliks. A wide
        /// spread is the strongest signal of multi-modality.
        loglik_spread: f64,
    },
    /// New in §Proposal 3 (Step 8): chain agreement Â passed but the
    /// inter-chain CLEAN-EVAL log-likelihood spread (in decibans)
    /// exceeded `max(gate.decibans_thresh, 8 · max(SE) · NATS_TO_DB)`.
    /// Strong signal that chains landed in different basins even with
    /// per-parameter trajectories that look stable.
    DecibansSpread {
        delta_db: f64,
        threshold_db: f64,
        sigma_max: f64,
        chain_logliks: Vec<f64>,
    },
}

/// Check scout's fit_state for pre-refine convergence.
///
/// Compound gate (Step 8, proposal §Proposal 3):
///
/// 1. Chain-agreement Â on every structural (non-`perturb_only_at_t0`)
///    param must be `< gate.a_thresh`.
///    Failure → `Hard`. Â in `[A_SOFT, gate.a_thresh)` → `SoftWarn`.
/// 2. If both `chain_eval_logliks` and `chain_eval_ses` are populated
///    (≥ 2 entries), the inter-chain decibans-spread of clean-eval
///    log-likelihoods must be below
///    `max(gate.decibans_thresh, 8 · max(SE) · NATS_TO_DB)`. The
///    SE-aware floor prevents penalising chains whose Monte-Carlo
///    spread alone could explain the observed log-lik spread.
///    Failure → `DecibansSpread`.
///
/// Legacy fit_state files (no `tail_chain_agreement`) return `Ok` —
/// the caller is expected to warn and proceed. Same fall-through for
/// missing `chain_clean_*` (the decibans check simply isn't run).
pub fn check_scout_convergence(scout: &FitState, gate: &GateConfig) -> ScoutGateVerdict {
    // Absent tail_chain_agreement means legacy fit_state — can't gate.
    // Caller handles the warn-and-proceed branch.
    if scout.tail_chain_agreement.is_empty() {
        return ScoutGateVerdict::Ok;
    }

    let t0_set: std::collections::HashSet<&str> =
        scout.perturb_only_at_t0_params.iter().map(|s| s.as_str()).collect();
    let mut structural: Vec<(String, f64)> = Vec::new();
    let mut perturb_only_at_t0: Vec<(String, f64)> = Vec::new();
    for (name, &agreement) in &scout.tail_chain_agreement {
        if t0_set.contains(name.as_str()) {
            perturb_only_at_t0.push((name.clone(), agreement));
        } else {
            structural.push((name.clone(), agreement));
        }
    }
    structural.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    perturb_only_at_t0
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let worst = structural.iter().map(|(_, r)| *r)
        .fold(0.0_f64, f64::max);

    // Step 1 — Â check. Reduced through `GateConfig::a_band`, the same
    // authority every renderer uses, so a printed glyph and this verdict
    // cannot disagree about one number.
    match gate.a_band(worst) {
        AgreementBand::Fail => {
            let failing: Vec<(String, f64)> = structural.iter()
                .filter(|(_, r)| gate.a_band(*r) == AgreementBand::Fail)
                .cloned().collect();
            let loglik_spread = if scout.chain_logliks.len() >= 2 {
                let hi = scout.chain_logliks.iter().cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let lo = scout.chain_logliks.iter().cloned()
                    .fold(f64::INFINITY, f64::min);
                hi - lo
            } else { 0.0 };
            return ScoutGateVerdict::Hard {
                failing,
                all_structural: structural,
                perturb_only_at_t0,
                loglik_spread,
            };
        }
        AgreementBand::SoftWarn => {
            let warnable: Vec<(String, f64)> = structural.into_iter()
                .filter(|(_, r)| gate.a_band(*r) == AgreementBand::SoftWarn)
                .collect();
            return ScoutGateVerdict::SoftWarn { param_agreement: warnable };
        }
        // `worst` is a max over finite entries seeded at 0.0, so `NotAssessed`
        // is unreachable for it; either way there is nothing for the Â leg to
        // refuse and the decibans leg below carries the verdict.
        AgreementBand::Pass | AgreementBand::NotAssessed => {}
    }

    // Step 2 — decibans-spread check on clean-eval logliks. Skipped
    // when the scout is pre-§Proposal 1 and didn't write the new fields.
    if scout.chain_eval_logliks.len() >= 2
        && scout.chain_eval_ses.len() == scout.chain_eval_logliks.len()
    {
        let hi = scout.chain_eval_logliks.iter().cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let lo = scout.chain_eval_logliks.iter().cloned()
            .fold(f64::INFINITY, f64::min);
        let delta_db = (hi - lo) * NATS_TO_DB;

        let sigma_max = scout.chain_eval_ses.iter().cloned()
            .fold(0.0_f64, f64::max);
        let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
        let threshold_db = gate.decibans_thresh.max(se_floor_db);

        if delta_db >= threshold_db {
            return ScoutGateVerdict::DecibansSpread {
                delta_db,
                threshold_db,
                sigma_max,
                chain_logliks: scout.chain_eval_logliks.clone(),
            };
        }
    }

    ScoutGateVerdict::Ok
}

/// Compute the ε tolerance for Gate 2: `max(LOGLIK_EPSILON_MIN,
/// 2 · σ(scout.chain_logliks))`. A wider scout spread (more evidence
/// of multi-modality) gives refine proportionally more room.
pub fn loglik_regression_epsilon(scout_chain_logliks: &[f64]) -> f64 {
    if scout_chain_logliks.len() < 2 {
        return LOGLIK_EPSILON_MIN;
    }
    let n = scout_chain_logliks.len() as f64;
    let mean = scout_chain_logliks.iter().sum::<f64>() / n;
    let var = scout_chain_logliks.iter()
        .map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let two_sigma = 2.0 * var.sqrt();
    LOGLIK_EPSILON_MIN.max(two_sigma)
}

/// Check Gate 2: refine's best loglik must not be worse than scout's
/// by more than ε. Returns `Ok(())` on pass, `Err(msg)` with a
/// human-readable diagnosis naming both logliks, the delta, and ε.
pub fn check_loglik_regression(
    scout_best: f64,
    refine_best: f64,
    scout_chain_logliks: &[f64],
) -> Result<(), String> {
    let epsilon = loglik_regression_epsilon(scout_chain_logliks);
    let delta = refine_best - scout_best;
    if delta >= -epsilon {
        Ok(())
    } else {
        Err(format!(
            "refine regressed below scout.\n\n  \
             scout  best_loglik = {:.1}\n  \
             refine best_loglik = {:.1}   delta = {:+.1}, threshold ε = {:.1}\n\n  \
             Refine landed in a worse basin than scout found. This is a\n  \
             pipeline failure, not a user-facing knob — refine is supposed\n  \
             to polish scout's best, not regress from it. Possible causes:\n\n    \
             - scout was multi-modal and refine's starts_from filter picked\n      \
             top-K chains from the wrong basin (re-run with tighter bounds\n      \
             around scout's best chain)\n    \
             - refine cooling too aggressive given rw_sd; collapsed on the\n      \
             first accessible local maximum\n    \
             - the model or data changed between stages (hash mismatch —\n      \
             check run.json)\n\n  \
             scout/fit_state.toml is authoritative for \"what scout's best\n  \
             looked like.\" Investigate before re-running.",
            scout_best, refine_best, delta, epsilon))
    }
}

/// Render the DecibansSpread verdict as a human error message.
/// Names the spread, the threshold (and which limb of `max(...)` it
/// came from), and the per-chain logliks in nats and decibans so the
/// user can see whether one chain is the obvious outlier.
pub fn format_decibans_spread_verdict(
    delta_db: f64,
    threshold_db: f64,
    sigma_max: f64,
    chain_logliks: &[f64],
) -> String {
    let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
    let floor_source = if se_floor_db >= threshold_db {
        format!("8 · σ_max · NATS_TO_DB = 8 · {:.2} · {:.3} ≈ {:.1} dB",
            sigma_max, NATS_TO_DB, se_floor_db)
    } else {
        format!("user-configured floor decibans_thresh = {:.1} dB", threshold_db)
    };
    let mut msg = format!(
        "scout chains landed in different basins.\n\n  \
         clean-eval log-likelihood spread:\n    \
         Δℓ = {:.2} dB > threshold = {:.2} dB ({})\n\n  \
         Per-chain clean logliks (nats / dB from worst):\n",
        delta_db, threshold_db, floor_source);
    let lo = chain_logliks.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut sorted: Vec<(usize, f64)> = chain_logliks.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, ll) in &sorted {
        msg.push_str(&format!("    chain {:<2}  ℓ = {:>9.2}  ({:+.2} dB from worst)\n",
            i + 1, ll, (ll - lo) * NATS_TO_DB));
    }

    // Name the chains DRIVING the spread, so the fix points at a chain index —
    // not just "the spread is wide". Same robust seam the `fit summary` per-chain
    // table uses (a modified z-score on per-chain clean loglik, centred on the
    // median and scaled by the MAD), so co-stuck chains are all flagged rather
    // than masking each other. The worst-chain fallback is belt-and-suspenders
    // for a smooth spread with no single robust outlier.
    let scores = super::chain_diagnostics::chain_loglik_mod_zscores(chain_logliks);
    let flagged = super::chain_diagnostics::outlier_labels(&scores);
    if !flagged.is_empty() {
        msg.push_str(&format!(
            "\n  Outlier chains (|modified z| > {:.1} on per-chain clean loglik): {}\n",
            super::chain_diagnostics::CHAIN_LOGLIK_OUTLIER_MODZ, flagged.join(", ")));
    } else if let Some((worst_i, _)) = chain_logliks.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
    {
        msg.push_str(&format!(
            "\n  Chain driving the spread (lowest clean loglik): chain {}\n",
            worst_i + 1));
    }

    msg.push_str("\n  Pick one:\n    \
                  - re-run scout with more chains (the wider the spread,\n      \
                    the higher the chance one chain is in the right basin)\n    \
                  - inspect chain_evaluations.tsv to see which candidate\n      \
                    label dominated each chain — divergent labels are a\n      \
                    multimodality signal\n    \
                  - if the spread is genuinely Monte-Carlo noise, raise\n      \
                    [stages.scout.clean_eval] n_particles or n_replicates\n    \
                  - relax the gate via [stages.scout.gate].decibans_thresh\n      \
                    or pass --decibans-thresh on the next run\n\n  \
                  To proceed anyway:  camdl fit run fit.toml --allow-nonconverged-scout");
    msg
}

/// Render the Gate 1 Hard verdict as a human error message.
///
/// `gate` is the one this verdict came from, so every glyph in the table is
/// the same comparison the verdict is — not a literal that agrees only while
/// `a_thresh` happens to equal it.
pub fn format_hard_verdict(
    gate: &GateConfig,
    failing: &[(String, f64)],
    all_structural: &[(String, f64)],
    perturb_only_at_t0: &[(String, f64)],
    loglik_spread: f64,
    scout_best_loglik: f64,
    scout_best_chain_values: Option<&[(String, f64)]>,
) -> String {
    let mut msg = format!(
        "refine stage requires scout convergence.\n\n  \
         Scout tail Â (IF2 chain agreement over the last half of iterations), \
         threshold {:.2}:\n",
        gate.a_thresh);
    for (name, agreement) in all_structural {
        let band = gate.a_band(*agreement);
        let marker = if band == AgreementBand::Pass { " " } else { band.glyph() };
        msg.push_str(&format!("    {} {:<10} Â = {:>6.3}{}\n",
            marker, name, agreement,
            if band == AgreementBand::Fail {
                format!("   (>= {:.2})", gate.a_thresh)
            } else {
                String::new()
            }));
    }
    for (name, agreement) in perturb_only_at_t0 {
        msg.push_str(&format!(
            "      {:<10} Â = {:>6.3}   (perturb_only_at_t0 — not gated)\n",
            name, agreement));
    }
    if loglik_spread > 0.0 {
        msg.push_str(&format!("\n  Scout loglik spread: {:.1} (best chain loglik {:.1})\n",
            loglik_spread, scout_best_loglik));
    }
    if loglik_spread > LOGLIK_EPSILON_MIN * 3.0 {
        msg.push_str("  -> likelihood surface is almost certainly multi-modal.\n");
    }
    msg.push_str(&format!("\n  Failing: {}\n",
        failing.iter().map(|(n, r)| format!("{} (Â={:.2})", n, r))
            .collect::<Vec<_>>().join(", ")));
    msg.push_str("\n  Pick one:\n    \
                  - re-run scout with more chains or iterations\n    \
                  - narrow bounds to the basin scout's best chain found");
    if let Some(vals) = scout_best_chain_values {
        msg.push_str(":\n");
        for (name, value) in vals {
            msg.push_str(&format!("        {} ≈ {:.4}\n", name, value));
        }
        msg.push_str("      copy into [estimate.*] bounds / start values\n    ");
    } else {
        msg.push_str("\n    ");
    }
    msg.push_str("- mark weakly-identified initial-state params as \
                  `perturb_only_at_t0 = true`\n      \
                  (reported but not gated)\n\n  \
                  To run refine anyway (results may launder multi-modality):\n    \
                  camdl fit run fit.toml --allow-nonconverged-scout");
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(
        tail_chain_agreement: &[(&str, f64)],
        perturb_only_at_t0_params: &[&str],
        chain_logliks: &[f64],
        best_loglik: f64,
    ) -> FitState {
        FitState {
            stage: "scout".into(),
            seed: 1,
            timestamp: "2026-04-19T00:00:00Z".into(),
            input_hash: None, camdl_version: None,
            best_loglik,
            initial_loglik: f64::NEG_INFINITY,
            best_chain: 0,
            n_chains: chain_logliks.len().max(1),
            n_good_chains: None,
            start_values: Default::default(),
            rw_sd: Default::default(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: tail_chain_agreement.iter()
                .map(|(k, v)| (k.to_string(), *v)).collect(),
            perturb_only_at_t0_params: perturb_only_at_t0_params.iter()
                .map(|s| s.to_string()).collect(),
            chain_logliks: chain_logliks.to_vec(),
            chain_eval_logliks: vec![],
            chain_eval_ids: Vec::new(),
            chain_eval_ses: vec![],
            resolved_gate: None,
            resolved_loglik_eval: None,
            chain_init_source: None,
            dt_check: None,
            pf_noise: None,
        }
    }

    /// Legacy GateConfig matching the pre-§Proposal 3 thresholds —
    /// useful for tests that exercise the SoftWarn band and the
    /// initial-state exemption logic, both of which were defined before the new
    /// stricter default `a_thresh = 1.01`.
    fn legacy_gate() -> GateConfig {
        GateConfig { a_thresh: 1.10, decibans_thresh: f64::INFINITY }
    }

    #[test]
    fn hard_gate_fires_when_structural_agreement_exceeds_threshold() {
        let s = make_state(
            &[("beta", 3.5), ("gamma", 1.2), ("I0", 16.5)],
            &["I0"],
            &[-60.2, -62.5, -63.3, -64.5, -66.2, -68.7, -854.6],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Hard { failing, loglik_spread, .. } => {
                let names: Vec<&str> = failing.iter()
                    .map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"beta"),
                    "beta (Â=3.5) must fail the gate: {:?}", names);
                assert!(names.contains(&"gamma"),
                    "gamma (Â=1.2) must fail: {:?}", names);
                // The perturb_only_at_t0 param I0 must NOT appear in failing.
                assert!(!names.contains(&"I0"),
                    "I0 is perturb_only_at_t0 — must not be in failing: {:?}",
                    names);
                // Loglik spread computed: 854.6 − 60.2 = 794.4
                assert!((loglik_spread - 794.4).abs() < 0.1,
                    "loglik spread {:.1}, expected 794.4", loglik_spread);
            }
            other => panic!("expected Hard, got {:?}", other),
        }
    }

    #[test]
    fn perturb_only_at_t0_agreement_not_gated_even_when_extreme() {
        // All structural params are green; only the perturb_only_at_t0 params
        // have extreme Â. The check must pass — an initial-state parameter is
        // expected to be hard to identify.
        let s = make_state(
            &[("beta", 1.02), ("gamma", 1.01), ("I0", 16.5), ("R_init", 5.5)],
            &["I0", "R_init"],
            &[-60.2, -60.5],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Ok => (),
            other => panic!(
                "expected Ok (perturb_only_at_t0 exempt), got {:?}", other),
        }
    }

    #[test]
    fn soft_warn_between_thresholds() {
        let s = make_state(
            &[("beta", 1.07), ("gamma", 1.02)],
            &[],
            &[-60.2, -60.5],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::SoftWarn { param_agreement } => {
                let names: Vec<&str> = param_agreement.iter()
                    .map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"beta"));
                assert!(!names.contains(&"gamma"),
                    "gamma (1.02) is below soft threshold, shouldn't be warned");
            }
            other => panic!("expected SoftWarn, got {:?}", other),
        }
    }

    #[test]
    fn legacy_state_with_no_agreement_returns_ok() {
        // Absent tail_chain_agreement (legacy fit_state from pre-2026-04-19):
        // caller treats this as "unknown, warn and proceed" via the
        // Ok verdict.
        let s = make_state(&[], &[], &[-60.0], -60.0);
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Ok => (),
            other => panic!("legacy state → Ok, got {:?}", other),
        }
    }

    /// Step 8: under the new strict default (`a_thresh = 1.01`),
    /// structural Â of 1.05 is already a hard fail — even though the
    /// pre-§Proposal 3 gate would have only soft-warned at this value.
    /// This is the intended tightening; documented here so a future
    /// reader doesn't try to "fix" it.
    #[test]
    fn default_gate_is_strict_about_chain_agreement() {
        let s = make_state(
            &[("beta", 1.05), ("gamma", 1.001)],
            &[],
            &[-60.0, -60.5],
            -60.0,
        );
        match check_scout_convergence(&s, &GateConfig::default()) {
            ScoutGateVerdict::Hard { failing, .. } => {
                let names: Vec<&str> = failing.iter().map(|(n, _)| n.as_str()).collect();
                assert!(names.contains(&"beta"),
                    "default a_thresh = 1.01 must fail beta (Â=1.05): {:?}", names);
            }
            other => panic!("expected Hard under default gate, got {:?}", other),
        }
    }

    /// Step 8 — SE-aware floor case from the handoff: small SE means
    /// `8·σ_max·NATS_TO_DB` is below the user-configured floor, so the
    /// floor binds. A spread of 100 dB exceeds 30 dB → DecibansSpread.
    #[test]
    fn decibans_spread_fails_when_floor_is_binding() {
        // Spread = 100 dB → 100 / NATS_TO_DB ≈ 23.03 nats.
        let mut s = make_state(
            &[("beta", 1.001)],
            &[],
            &[-60.0, -60.0],   // legacy chain_logliks unused by gate 2
            -60.0,
        );
        s.chain_eval_logliks = vec![-60.0, -60.0 - 100.0 / NATS_TO_DB];
        s.chain_eval_ses = vec![0.5, 0.5];   // 8·0.5·NATS_TO_DB ≈ 17.4 dB < 30 dB

        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::DecibansSpread {
                delta_db, threshold_db, sigma_max, chain_logliks,
            } => {
                assert!((delta_db - 100.0).abs() < 0.5,
                    "delta_db ≈ 100 dB; got {}", delta_db);
                assert!((threshold_db - 30.0).abs() < 1e-9,
                    "30 dB floor must bind (8·σ·NATS_TO_DB ≈ 17.4 dB < 30); got {}",
                    threshold_db);
                assert!((sigma_max - 0.5).abs() < 1e-12);
                assert_eq!(chain_logliks.len(), 2);
            }
            other => panic!("expected DecibansSpread, got {:?}", other),
        }
    }

    /// Step 8 — SE-aware floor case from the handoff: large SE pushes
    /// the threshold above the user-configured floor (8·5·NATS_TO_DB
    /// ≈ 173.7 dB), so a 100 dB spread now passes.
    #[test]
    fn decibans_spread_passes_when_se_aware_floor_dominates() {
        let mut s = make_state(
            &[("beta", 1.001)],
            &[],
            &[-60.0, -60.0],
            -60.0,
        );
        s.chain_eval_logliks = vec![-60.0, -60.0 - 100.0 / NATS_TO_DB];
        s.chain_eval_ses = vec![5.0, 5.0];   // 8·5·NATS_TO_DB ≈ 173.7 dB

        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::Ok => (),
            other => panic!("expected Ok (SE-aware floor dominates), got {:?}", other),
        }
    }

    /// Step 8 — when clean-eval fields are absent (legacy fit_state
    /// or pre-§Proposal 1), the decibans check is skipped silently.
    /// The Â check still runs.
    #[test]
    fn missing_clean_eval_fields_skip_decibans_check() {
        let s = make_state(
            &[("beta", 1.001)],
            &[],
            &[-60.0, -200.0],   // huge legacy spread, but Step 2 doesn't use this
            -60.0,
        );
        match check_scout_convergence(&s, &GateConfig::default()) {
            ScoutGateVerdict::Ok => (),
            other => panic!("missing clean_eval fields → Ok; got {:?}", other),
        }
    }

    #[test]
    fn loglik_regression_fires_when_refine_below_scout() {
        // Scout best = -60.2; refine best = -76.3. Regression of 16.1.
        // Scout chain spread is wide (-60.2 to -68.7, σ ≈ 3), so
        // ε = max(3, 2·3) ≈ 6. Delta of -16.1 >> ε → error.
        let scout_lls = vec![-60.2, -62.5, -63.3, -64.5, -66.2, -68.7];
        let err = check_loglik_regression(-60.2, -76.3, &scout_lls)
            .expect_err("refine regressed far below scout");
        assert!(err.contains("-60.2") && err.contains("-76.3"),
            "error must name both logliks: {}", err);
        assert!(err.contains("regressed"),
            "error must use the word 'regressed': {}", err);
    }

    #[test]
    fn loglik_regression_tolerates_small_delta() {
        // Scout best = -60.2; refine best = -62.0. Delta 1.8 < ε (3).
        // Should pass — within the noise floor of the PF loglik.
        let scout_lls = vec![-60.2, -60.3, -60.1, -60.4];  // tight
        check_loglik_regression(-60.2, -62.0, &scout_lls)
            .expect("small regression within ε should pass");
    }

    #[test]
    fn loglik_regression_passes_when_refine_better() {
        // Refine improved on scout's best — always passes.
        let scout_lls = vec![-60.2, -62.5, -63.3];
        check_loglik_regression(-60.2, -58.0, &scout_lls)
            .expect("refine improvement must pass");
    }

    /// gh#406: the DecibansSpread message must NAME the outlier chain, not just
    /// report the aggregate spread. A single clear low outlier among six chains
    /// clears the modified-z threshold, so the "Outlier chains" line names it.
    #[test]
    fn decibans_message_names_a_clear_outlier_chain() {
        // Five chains near -60, chain 6 stuck at -300 (index 5 → "chain 6").
        let chain_logliks = vec![-60.0, -61.0, -59.0, -60.5, -60.2, -300.0];
        let msg = format_decibans_spread_verdict(240.0, 30.0, 0.5, &chain_logliks);
        assert!(
            msg.contains("Outlier chains"),
            "message must flag the outlier chain, not just the spread:\n{msg}"
        );
        assert!(
            msg.contains("Outlier chains (|modified z| > 3.5 on per-chain clean loglik): chain 6"),
            "message must name chain 6 as the outlier:\n{msg}"
        );
    }

    /// gh#406 (robust upgrade): two co-stuck chains among six are BOTH named now
    /// — the classic mean/SD z masked them, the robust median/MAD modified z does
    /// not. Good chains carry realistic jitter (real chain means never coincide).
    #[test]
    fn decibans_message_names_both_co_stuck_chains() {
        // Four good (jittered) near -60, two stuck near -400 (chains 5 and 6).
        let chain_logliks = vec![-60.0, -61.0, -59.0, -60.5, -400.0, -401.0];
        let msg = format_decibans_spread_verdict(340.0, 30.0, 0.5, &chain_logliks);
        assert!(msg.contains("Outlier chains"), "both co-stuck chains must be flagged:\n{msg}");
        assert!(msg.contains("chain 5"), "must name chain 5:\n{msg}");
        assert!(msg.contains("chain 6"), "must name chain 6:\n{msg}");
        assert!(
            !msg.contains("Chain driving the spread"),
            "the fallback must NOT fire when robust outliers are found:\n{msg}"
        );
    }

    /// gh#406: the belt-and-suspenders fallback. A smooth spread (a gradient of
    /// chains, no single robust outlier) still triggers the gate, so the message
    /// names the single worst chain driving the low end — always actionable.
    #[test]
    fn decibans_message_fallback_names_worst_on_smooth_spread() {
        let chain_logliks = vec![-60.0, -70.0, -80.0, -90.0, -100.0, -110.0];
        let scores = super::super::chain_diagnostics::chain_loglik_mod_zscores(&chain_logliks);
        assert!(
            scores.iter().all(|s| !s.is_outlier),
            "precondition: a smooth gradient has no single robust outlier: {:?}",
            scores.iter().map(|s| s.mod_z).collect::<Vec<_>>()
        );
        let msg = format_decibans_spread_verdict(150.0, 30.0, 0.5, &chain_logliks);
        assert!(
            msg.contains("Chain driving the spread (lowest clean loglik): chain 6"),
            "fallback must name the worst chain (-110 → chain 6):\n{msg}"
        );
        assert!(!msg.contains("Outlier chains"), "no robust outlier line on a smooth spread:\n{msg}");
    }

    #[test]
    fn epsilon_widens_with_scout_loglik_spread() {
        let tight = vec![-60.0, -60.1, -60.0, -59.9];
        let wide  = vec![-60.0, -70.0, -80.0, -55.0];
        let eps_tight = loglik_regression_epsilon(&tight);
        let eps_wide  = loglik_regression_epsilon(&wide);
        assert!(eps_wide > eps_tight * 2.0,
            "wider scout spread should give proportionally larger ε: \
             tight={:.2}, wide={:.2}", eps_tight, eps_wide);
        assert!(eps_tight >= LOGLIK_EPSILON_MIN,
            "ε must never drop below the floor: {}", eps_tight);
    }
}
