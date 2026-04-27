//! Refine-stage convergence gates.
//!
//! Two gates protect against the "refine launders an unconverged
//! scout" failure mode documented in
//! `docs/dev/proposals/2026-04-19-refine-gates-scout-convergence.md`:
//!
//! - Gate 1 (pre-refine): scout's tail chain-agreement (Â) on every
//!   non-IVP estimated parameter must be ≤ `A_HARD`. If it isn't, refine refuses to
//!   start. Overridable via `--allow-nonconverged-scout`.
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
use serde::{Deserialize, Serialize};

/// Legacy hard threshold for the chain-agreement Â check. Retained as
/// a documented constant because it informs the SoftWarn band's upper
/// bound when `gate.a_thresh` is configured loosely. Step 8 (proposal
/// §Proposal 3) shifts the *effective* hard threshold to
/// `gate.a_thresh` (default 1.01) — a deliberate tightening; see the
/// proposal.
pub const A_HARD: f64 = 1.10;

/// Soft threshold: params between this and `gate.a_thresh` get a
/// prominent warning but refine still runs. When `gate.a_thresh ≤
/// A_SOFT` the SoftWarn band is empty (every above-soft Â also
/// exceeds the hard gate). Matches the existing scout diagnostic
/// colour-coding (red ≥ a_thresh, yellow A_SOFT..a_thresh,
/// green < A_SOFT).
pub const A_SOFT: f64 = 1.05;

/// Multiplier for the SE-aware decibans floor:
///   threshold = max(decibans_thresh, SE_FLOOR_K * max(SE) * NATS_TO_DB)
/// k=8 is the proposal's recommended default — see
/// docs/dev/proposals/2026-04-24-if2-scout-findings-remediation.md §Proposal 3.
pub const SE_FLOOR_K: f64 = 8.0;

/// Minimum ε for Gate 2. Scout's noise floor on a typical PF-based
/// loglik estimator at reasonable particle counts. `epsilon` takes the
/// max of this and `2 * σ_scout_chains` so multi-modal scout runs (high
/// between-chain σ) get a proportionally wider tolerance.
pub const LOGLIK_EPSILON_MIN: f64 = 3.0;

/// Verdict from the pre-refine convergence check. `SoftWarn` callers
/// should print the named parameters prominently. `Hard` and
/// `DecibansSpread` callers should error unless the user passed
/// `--allow-nonconverged-scout`, in which case downgrade to a warning.
/// `LegacySkipped` means the prior stage's `fit_state.toml` predates
/// the tail-chain-agreement field; the caller must warn and proceed.
#[derive(Debug)]
pub enum ScoutGateVerdict {
    Ok,
    /// Prior state has no `tail_chain_agreement` — legacy fit_state
    /// from before chain-agreement tracking. Gate was skipped entirely;
    /// caller must warn the user rather than treating this as a pass.
    LegacySkipped,
    SoftWarn { param_agreement: Vec<(String, f64)> },
    Hard {
        /// All non-IVP params with Â ≥ `gate.a_thresh`. Named and
        /// sorted worst-first so the error message leads with the
        /// most obvious failure.
        failing: Vec<(String, f64)>,
        /// Every non-IVP Â, for the full diagnostic table.
        all_structural: Vec<(String, f64)>,
        /// IVP Â values (reported but not gated).
        ivp: Vec<(String, f64)>,
        /// Spread across chains using clean-eval logliks when available
        /// (more accurate than in-run logliks), falling back to
        /// `chain_logliks`. Wide spread signals multi-modality.
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
        /// Pre-computed SE-aware floor: `SE_FLOOR_K * sigma_max * NATS_TO_DB`.
        /// Carried here so display formatters don't recompute it.
        se_floor_db: f64,
        chain_logliks: Vec<f64>,
    },
}

/// Check scout's fit_state for pre-refine convergence.
///
/// Compound gate (Step 8, proposal §Proposal 3):
///
/// 1. Chain-agreement Â on every non-IVP param must be `< gate.a_thresh`.
///    Failure → `Hard`. Â in `[A_SOFT, gate.a_thresh)` → `SoftWarn`.
/// 2. If both `chain_clean_logliks` and `chain_clean_ses` are populated
///    (≥ 2 entries), the inter-chain decibans-spread of clean-eval
///    log-likelihoods must be below
///    `max(gate.decibans_thresh, 8 · max(SE) · NATS_TO_DB)`. The
///    SE-aware floor prevents penalising chains whose Monte-Carlo
///    spread alone could explain the observed log-lik spread.
///    Failure → `DecibansSpread`.
///
/// Legacy fit_state files (no `tail_chain_agreement`) return
/// `LegacySkipped` — the caller must warn the user rather than
/// treating it as a pass. Same fall-through for missing `chain_clean_*`
/// (the decibans check simply isn't run).
pub fn check_scout_convergence(scout: &FitState, gate: &GateConfig) -> ScoutGateVerdict {
    // Absent tail_chain_agreement means legacy fit_state — can't gate.
    // Return LegacySkipped so the caller can warn explicitly rather
    // than silently treating absent data as a convergence pass.
    if scout.tail_chain_agreement.is_empty() {
        return ScoutGateVerdict::LegacySkipped;
    }

    let ivp_set: std::collections::HashSet<&str> = scout.ivp_params.iter()
        .map(|s| s.as_str()).collect();
    let mut structural: Vec<(String, f64)> = Vec::new();
    let mut ivp: Vec<(String, f64)> = Vec::new();
    for (name, &agreement) in &scout.tail_chain_agreement {
        if ivp_set.contains(name.as_str()) {
            ivp.push((name.clone(), agreement));
        } else {
            structural.push((name.clone(), agreement));
        }
    }
    structural.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ivp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let worst = structural.iter().map(|(_, r)| *r)
        .fold(0.0_f64, f64::max);

    // Step 1 — Â check.
    if worst >= gate.a_thresh {
        let failing: Vec<(String, f64)> = structural.iter()
            .filter(|(_, r)| *r >= gate.a_thresh)
            .cloned().collect();
        // Prefer de-biased clean-eval spread (less noisy than in-run
        // logliks at typical particle counts). Fall back to chain_logliks
        // when clean-eval data is absent (legacy or pre-§Proposal 1 runs).
        let loglik_spread = if scout.chain_clean_logliks.len() >= 2 {
            let hi = scout.chain_clean_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = scout.chain_clean_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            hi - lo
        } else if scout.chain_logliks.len() >= 2 {
            let hi = scout.chain_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = scout.chain_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            hi - lo
        } else { 0.0 };
        return ScoutGateVerdict::Hard {
            failing,
            all_structural: structural,
            ivp,
            loglik_spread,
        };
    }
    if worst >= A_SOFT {
        // SoftWarn band only exists when gate.a_thresh > A_SOFT;
        // otherwise the previous branch consumed everything above
        // A_SOFT and we don't reach here.
        let warnable: Vec<(String, f64)> = structural.into_iter()
            .filter(|(_, r)| *r >= A_SOFT)
            .collect();
        return ScoutGateVerdict::SoftWarn { param_agreement: warnable };
    }

    // Step 2 — decibans-spread check on clean-eval logliks. Skipped
    // when the scout is pre-§Proposal 1 and didn't write the new fields.
    if scout.chain_clean_logliks.len() >= 2
        && scout.chain_clean_ses.len() == scout.chain_clean_logliks.len()
    {
        let hi = scout.chain_clean_logliks.iter().cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let lo = scout.chain_clean_logliks.iter().cloned()
            .fold(f64::INFINITY, f64::min);
        let delta_db = (hi - lo) * NATS_TO_DB;

        let sigma_max = scout.chain_clean_ses.iter().cloned()
            .fold(0.0_f64, f64::max);
        let se_floor_db = SE_FLOOR_K * sigma_max * NATS_TO_DB;
        let threshold_db = gate.decibans_thresh.max(se_floor_db);

        if delta_db >= threshold_db {
            return ScoutGateVerdict::DecibansSpread {
                delta_db,
                threshold_db,
                sigma_max,
                se_floor_db,
                chain_logliks: scout.chain_clean_logliks.clone(),
            };
        }
    }

    ScoutGateVerdict::Ok
}

/// Compute the ε tolerance for Gate 2: `max(LOGLIK_EPSILON_MIN,
/// 2 · σ(scout.chain_logliks))`. A wider scout spread (more evidence
/// of multi-modality) gives refine proportionally more room.
///
/// Uses **sample** standard deviation (Bessel-corrected, `/(n-1)`), not
/// population SD. For two chains this gives `|ll_1 - ll_2|`, which is
/// the natural pairwise spread — a sensible choice even at n=2.
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
///
/// `se_floor_db` must equal `SE_FLOOR_K * sigma_max * NATS_TO_DB` —
/// the caller (always `check_scout_convergence`) computes it once and
/// the `DecibansSpread` variant carries it, avoiding a third site for
/// this formula.
pub fn format_decibans_spread_verdict(
    delta_db: f64,
    threshold_db: f64,
    sigma_max: f64,
    se_floor_db: f64,
    chain_logliks: &[f64],
) -> String {
    let floor_source = if se_floor_db >= threshold_db {
        format!("{} · σ_max · NATS_TO_DB = {} · {:.2} · {:.3} ≈ {:.1} dB",
            SE_FLOOR_K as u64, SE_FLOOR_K as u64, sigma_max, NATS_TO_DB, se_floor_db)
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
pub fn format_hard_verdict(
    failing: &[(String, f64)],
    all_structural: &[(String, f64)],
    ivp: &[(String, f64)],
    loglik_spread: f64,
    scout_best_loglik: f64,
    scout_best_chain_values: Option<&[(String, f64)]>,
) -> String {
    let mut msg = String::from(
        "refine stage requires scout convergence.\n\n  \
         Scout tail Â (last half of iterations):\n");
    for (name, agreement) in all_structural {
        let marker = if *agreement > A_HARD { "✗" }
                     else if *agreement > A_SOFT { "~" }
                     else { " " };
        msg.push_str(&format!("    {} {:<10} Â = {:>6.3}{}\n",
            marker, name, agreement,
            if *agreement > A_HARD { "   (> 1.10)" } else { "" }));
    }
    for (name, agreement) in ivp {
        msg.push_str(&format!("      {:<10} Â = {:>6.3}   (ivp — not gated)\n",
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
    msg.push_str("- mark weakly-identified params as `ivp = true`\n      \
                  (reported but not gated)\n\n  \
                  Note: decibans-spread check skipped — fix Â first,\n  \
                  then re-run (you may see a second gate failure on spread).\n\n  \
                  To run refine anyway (results may launder multi-modality):\n    \
                  camdl fit run fit.toml --allow-nonconverged-scout");
    msg
}

// ── Post-run gate summary types (ClM3) ─────────────────────────────

/// Where the gate thresholds used in a `GateVerdictSummary` came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateThresholdSource {
    /// Read from `state.resolved_gate` — thresholds reflect what the
    /// run was actually judged against.
    Resolved,
    /// Legacy `fit_state.toml` without `resolved_gate`. Showing
    /// `GateConfig::default()` with a caveat.
    DefaultFallback,
}

/// Pre-computed gate verdict for a completed stage. Computed once from
/// a `FitState` by `compute_gate_verdict` and consumed by all rendering
/// paths (text, JSON, MD, LaTeX) so that the arithmetic lives in exactly
/// one place and can't diverge between formats (ClM3).
///
/// All `Option` fields are `None` when the decibans leg can't be
/// evaluated (missing `chain_clean_logliks` / `chain_clean_ses`).
/// `overall_pass = None` when data is incomplete.
#[derive(Debug, Clone)]
pub struct GateVerdictSummary {
    pub max_a: f64,
    pub max_a_param: Option<String>,
    pub a_thresh: f64,
    pub a_passes: bool,
    pub delta_db: Option<f64>,
    pub threshold_db: Option<f64>,
    pub sigma_max: Option<f64>,
    pub db_passes: Option<bool>,
    pub overall_pass: Option<bool>,
    pub threshold_source: GateThresholdSource,
}

/// Compute the gate verdict once from a `FitState`. All rendering paths
/// must call this rather than re-implementing the gate arithmetic.
pub fn compute_gate_verdict(state: &FitState) -> GateVerdictSummary {
    let (gate, threshold_source) = match &state.resolved_gate {
        Some(g) => (g.clone(), GateThresholdSource::Resolved),
        None    => (GateConfig::default(), GateThresholdSource::DefaultFallback),
    };

    let max_a = state.tail_chain_agreement.values().cloned()
        .fold(0.0_f64, f64::max);
    let max_a_param = state.tail_chain_agreement.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k.clone());
    let a_passes = max_a < gate.a_thresh;

    let (delta_db, threshold_db, sigma_max, db_passes) =
        if state.chain_clean_logliks.len() >= 2
            && state.chain_clean_ses.len() == state.chain_clean_logliks.len()
        {
            let hi = state.chain_clean_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = state.chain_clean_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let dd = (hi - lo) * NATS_TO_DB;
            let sm = state.chain_clean_ses.iter().cloned().fold(0.0_f64, f64::max);
            let floor = SE_FLOOR_K * sm * NATS_TO_DB;
            let td = gate.decibans_thresh.max(floor);
            (Some(dd), Some(td), Some(sm), Some(dd < td))
        } else {
            (None, None, None, None)
        };

    let overall_pass = db_passes.map(|p| p && a_passes);

    GateVerdictSummary {
        max_a, max_a_param, a_thresh: gate.a_thresh, a_passes,
        delta_db, threshold_db, sigma_max, db_passes, overall_pass,
        threshold_source,
    }
}

// ── Gate failure sidecar (M2 / M3) ─────────────────────────────────

/// Which gate blocked this stage — written to `gate_failure.toml` in
/// the blocked stage directory so `camdl fit summary` can show the
/// reason even when `fit_state.toml` is absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateFailureKind {
    /// Gate 1 Hard: max(Â) ≥ a_thresh on a non-IVP param.
    Gate1Hard,
    /// Gate 1 DecibansSpread: Â passed but inter-chain clean-eval
    /// loglik spread exceeded the SE-adaptive threshold.
    Gate1DecibansSpread,
    /// Gate 2 regression: stage best_loglik regressed below the prior
    /// stage's best_loglik by more than ε.
    Gate2Regression,
}

/// Sidecar written to a stage directory when a gate blocks it. Lets
/// `camdl fit summary` surface the failure reason without requiring the
/// user to have kept the `camdl fit run` terminal output.
#[derive(Debug, Serialize, Deserialize)]
pub struct GateFailureRecord {
    pub kind: GateFailureKind,
    pub prior_stage: Option<String>,
    pub timestamp: String,
    // Gate 1 Hard fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a_thresh: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_a: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_a_param: Option<String>,
    /// Failing params as "name=value" strings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failing_params: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loglik_spread: Option<f64>,
    // Gate 1 DecibansSpread fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta_db: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threshold_db: Option<f64>,
    // Gate 2 regression fields
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scout_best_loglik: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_best_loglik: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regression_epsilon: Option<f64>,
}

/// Write a `gate_failure.toml` sidecar into `dir`. Creates `dir` if it
/// doesn't exist (Gate 1 blocks before the stage dir is created).
/// Errors are non-fatal — if we can't write the sidecar the gate
/// message itself still appeared on stderr.
pub fn write_gate_failure(dir: &std::path::Path, record: &GateFailureRecord) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("warning: could not create {} for gate_failure.toml: {}", dir.display(), e);
        return;
    }
    let path = dir.join("gate_failure.toml");
    match toml::to_string_pretty(record) {
        Ok(body) => {
            let _ = std::fs::write(&path,
                format!("# gate_failure.toml — written by camdl on gate block\n{}", body));
        }
        Err(e) => eprintln!("warning: could not serialize gate_failure.toml: {}", e),
    }
}

/// Load a `gate_failure.toml` from a stage directory. Returns `None`
/// when the file is absent (stage completed normally or never started).
pub fn load_gate_failure(dir: &str) -> Option<GateFailureRecord> {
    let path = format!("{}/gate_failure.toml", dir);
    let body = std::fs::read_to_string(&path).ok()?;
    toml::from_str(&body).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_state(
        tail_chain_agreement: &[(&str, f64)],
        ivp_params: &[&str],
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
            start_values: HashMap::new(),
            rw_sd: HashMap::new(),
            loglik_type: Some("if2".into()),
            acceptance_rate: None,
            tail_chain_agreement: tail_chain_agreement.iter()
                .map(|(k, v)| (k.to_string(), *v)).collect(),
            ivp_params: ivp_params.iter().map(|s| s.to_string()).collect(),
            chain_logliks: chain_logliks.to_vec(),
            chain_clean_logliks: vec![],
            chain_clean_ses: vec![],
            soft_warn_params: vec![],
            resolved_gate: None,
            resolved_clean_eval: None,
        }
    }

    /// Legacy GateConfig matching the pre-§Proposal 3 thresholds —
    /// useful for tests that exercise the SoftWarn band and IVP
    /// exemption logic, both of which were defined before the new
    /// stricter default `a_thresh = 1.01`.
    fn legacy_gate() -> GateConfig {
        GateConfig { a_thresh: A_HARD, decibans_thresh: f64::INFINITY }
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
                // IVP param I0 must NOT appear in failing.
                assert!(!names.contains(&"I0"),
                    "I0 is ivp — must not be in failing: {:?}", names);
                // Loglik spread computed: 854.6 − 60.2 = 794.4
                assert!((loglik_spread - 794.4).abs() < 0.1,
                    "loglik spread {:.1}, expected 794.4", loglik_spread);
            }
            other => panic!("expected Hard, got {:?}", other),
        }
    }

    #[test]
    fn ivp_agreement_not_gated_even_when_extreme() {
        // All structural params are green; only IVP has extreme Â.
        // Gate should pass — IVP is expected to be hard to identify.
        let s = make_state(
            &[("beta", 1.02), ("gamma", 1.01), ("I0", 16.5), ("R_init", 5.5)],
            &["I0", "R_init"],
            &[-60.2, -60.5],
            -60.2,
        );
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::Ok => (),
            other => panic!("expected Ok (IVP exempt), got {:?}", other),
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
    fn legacy_state_with_no_agreement_returns_legacy_skipped() {
        // Absent tail_chain_agreement (legacy fit_state from pre-2026-04-19):
        // caller must warn and proceed — distinguished from a genuine Ok
        // by the LegacySkipped variant so the warning is explicit.
        let s = make_state(&[], &[], &[-60.0], -60.0);
        match check_scout_convergence(&s, &legacy_gate()) {
            ScoutGateVerdict::LegacySkipped => (),
            other => panic!("legacy state → LegacySkipped, got {:?}", other),
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
        s.chain_clean_logliks = vec![-60.0, -60.0 - 100.0 / NATS_TO_DB];
        s.chain_clean_ses = vec![0.5, 0.5];   // 8·0.5·NATS_TO_DB ≈ 17.4 dB < 30 dB

        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        match check_scout_convergence(&s, &gate) {
            ScoutGateVerdict::DecibansSpread {
                delta_db, threshold_db, sigma_max, se_floor_db, chain_logliks,
            } => {
                assert!((delta_db - 100.0).abs() < 0.5,
                    "delta_db ≈ 100 dB; got {}", delta_db);
                assert!((threshold_db - 30.0).abs() < 1e-9,
                    "30 dB floor must bind (8·σ·NATS_TO_DB ≈ 17.4 dB < 30); got {}",
                    threshold_db);
                assert!((sigma_max - 0.5).abs() < 1e-12);
                assert_eq!(chain_logliks.len(), 2);
                // se_floor_db = 8 * 0.5 * NATS_TO_DB ≈ 17.4
                let expected_floor = SE_FLOOR_K * 0.5 * NATS_TO_DB;
                assert!((se_floor_db - expected_floor).abs() < 1e-9,
                    "se_floor_db must equal SE_FLOOR_K*sigma*NATS_TO_DB: {} vs {}",
                    se_floor_db, expected_floor);
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
        s.chain_clean_logliks = vec![-60.0, -60.0 - 100.0 / NATS_TO_DB];
        s.chain_clean_ses = vec![5.0, 5.0];   // 8·5·NATS_TO_DB ≈ 173.7 dB

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

    // ── TsN3: compound-gate quadrant tests ─────────────────────────
    // All four (Â pass/fail) × (decibans pass/fail) combinations must
    // produce the right compute_gate_verdict output. Catches desync
    // between check_scout_convergence and compute_gate_verdict.

    fn make_state_with_clean_eval(
        a_val: f64, clean_spread_db: f64, clean_se: f64, gate: &GateConfig,
    ) -> (FitState, GateConfig) {
        let mut s = make_state(&[("beta", a_val)], &[], &[-60.0, -60.0], -60.0);
        s.resolved_gate = Some(gate.clone());
        let lo = -60.0_f64;
        let hi = lo + clean_spread_db / NATS_TO_DB;
        s.chain_clean_logliks = vec![lo, hi];
        s.chain_clean_ses = vec![clean_se, clean_se];
        (s, gate.clone())
    }

    /// Q1: Â passes, decibans passes → overall pass
    #[test]
    fn quadrant_a_pass_db_pass() {
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        // Â = 1.001 < 1.10; spread = 5 dB < 30 dB
        let (s, _) = make_state_with_clean_eval(1.001, 5.0, 0.5, &gate);
        let v = compute_gate_verdict(&s);
        assert!(v.a_passes, "Â must pass: a={}", v.max_a);
        assert_eq!(v.db_passes, Some(true));
        assert_eq!(v.overall_pass, Some(true));
    }

    /// Q2: Â fails, decibans passes → overall fail (Â is the failing leg)
    #[test]
    fn quadrant_a_fail_db_pass() {
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        // Â = 1.15 > 1.10; spread = 5 dB < 30 dB
        let (s, _) = make_state_with_clean_eval(1.15, 5.0, 0.5, &gate);
        let v = compute_gate_verdict(&s);
        assert!(!v.a_passes, "Â must fail: a={}", v.max_a);
        assert_eq!(v.db_passes, Some(true));
        assert_eq!(v.overall_pass, Some(false));
    }

    /// Q3: Â passes, decibans fails → overall fail (decibans is the failing leg)
    #[test]
    fn quadrant_a_pass_db_fail() {
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        // Â = 1.001 < 1.10; spread = 100 dB > 30 dB; SE floor < 30 dB
        let (s, _) = make_state_with_clean_eval(1.001, 100.0, 0.5, &gate);
        let v = compute_gate_verdict(&s);
        assert!(v.a_passes, "Â must pass: a={}", v.max_a);
        assert_eq!(v.db_passes, Some(false));
        assert_eq!(v.overall_pass, Some(false));
    }

    /// Q4: both Â and decibans fail → overall fail
    #[test]
    fn quadrant_a_fail_db_fail() {
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        // Â = 1.15 > 1.10; spread = 100 dB > 30 dB
        let (s, _) = make_state_with_clean_eval(1.15, 100.0, 0.5, &gate);
        let v = compute_gate_verdict(&s);
        assert!(!v.a_passes, "Â must fail: a={}", v.max_a);
        assert_eq!(v.db_passes, Some(false));
        assert_eq!(v.overall_pass, Some(false));
    }

    /// compute_gate_verdict is pure — calling it twice on the same
    /// FitState produces identical results.
    #[test]
    fn compute_gate_verdict_is_pure() {
        let gate = GateConfig { a_thresh: 1.10, decibans_thresh: 30.0 };
        let (s, _) = make_state_with_clean_eval(1.001, 5.0, 0.5, &gate);
        let v1 = compute_gate_verdict(&s);
        let v2 = compute_gate_verdict(&s);
        assert_eq!(v1.a_passes, v2.a_passes);
        assert_eq!(v1.db_passes, v2.db_passes);
        assert_eq!(v1.overall_pass, v2.overall_pass);
        assert!((v1.max_a - v2.max_a).abs() < 1e-12);
    }

    /// GateFailureRecord round-trips through TOML.
    #[test]
    fn gate_failure_record_round_trips_toml() {
        let rec = GateFailureRecord {
            kind: GateFailureKind::Gate1Hard,
            prior_stage: Some("scout".into()),
            timestamp: "2026-04-27T00:00:00Z".into(),
            a_thresh: Some(1.01),
            max_a: Some(3.5),
            max_a_param: Some("beta".into()),
            failing_params: vec!["beta=3.50".into()],
            loglik_spread: Some(794.4),
            delta_db: None,
            threshold_db: None,
            scout_best_loglik: None,
            stage_best_loglik: None,
            regression_epsilon: None,
        };
        let body = toml::to_string_pretty(&rec).expect("serialize");
        let loaded: GateFailureRecord = toml::from_str(&body).expect("deserialize");
        assert_eq!(loaded.kind, GateFailureKind::Gate1Hard);
        assert_eq!(loaded.prior_stage.as_deref(), Some("scout"));
        assert!((loaded.max_a.unwrap() - 3.5).abs() < 1e-9);
        assert_eq!(loaded.failing_params, vec!["beta=3.50"]);
    }
}
