//! Typed loaded interpretation of a completed fit-stage.
//!
//! One variant per inference method: each carries the typed payload its
//! method produces, so consumers pattern-match instead of stringly-dispatching
//! on the stage leaf's `inputs.method` tag.
//!
//! Three variants — pfilter is excluded by design (it's a CLI
//! evaluator on already-fixed parameters, never a fit-stage). See
//! `docs/dev/proposals/2026-04-28-fit-experiment-management.md` §2.
//!
//! Map fields use [`BTreeMap<String, _>`] end-to-end (never `HashMap`)
//! so `serde_json` produces lexicographically-ordered JSON output.
//! This is load-bearing for step 5's `summary ⊆ table` byte-equality
//! test (Deliverable C). A clippy lint or unit test guarding this
//! would be reasonable insurance — for now, the type definitions are
//! the authoritative constraint.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::fit::state::FitState;

/// One stage of a fit, typed by method. The variant carries the
/// payload appropriate to its inference method (point estimate +
/// gates for IF2; posterior summaries + R̂ for Bayesian; status +
/// per-chain spread for NLopt deterministic MLE).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "kebab-case")]
pub enum MethodResult {
    If2(If2StageResult),
    Pgas(PgasStageResult),
    Pmmh(PmmhStageResult),
    #[serde(rename = "nl-sbplx", alias = "nl-bobyqa")]
    Nlopt(NloptStageResult),
}

/// Compound scout-convergence gate verdict (the IF2 "is this stage
/// converged" answer). String projection used in
/// `table_row.gate_verdict`:
///
/// | variant   | string       |
/// |-----------|--------------|
/// | `Pass`    | `"pass"`     |
/// | `FailA`   | `"fail_a"`   |
/// | `FailDb`  | `"fail_db"`  |
/// | `FailBoth`| `"fail_both"`|
///
/// Bayesian rows render `"n/a"` because the IF2 gate doesn't apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    Pass,
    FailA,
    FailDb,
    FailBoth,
}

// String projection for the four variants is supplied by the
// `serde(rename_all = "snake_case")` derive: serializing a
// `GateVerdict` produces exactly `"pass"` / `"fail_a"` / `"fail_db"`
// / `"fail_both"`. A separate `as_str` method would be redundant
// with that and prone to drift; consumers call `serde_json::to_value`
// (or destructure on the variant) instead.

/// IF2-stage result: point estimate + convergence diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct If2StageResult {
    pub best_loglik: f64,
    pub best_chain: usize,
    /// Winner θ̂ from clean-eval. Estimated parameters only — fixed
    /// params live elsewhere (e.g. `final_params.toml` carries both).
    pub theta_hat: BTreeMap<String, f64>,
    /// Maximum chain-agreement Â over estimated params. **Â, not
    /// Gelman-Rubin R̂** — they're computed differently (Â is
    /// per-parameter chain-agreement on the IF2 trace's tail; R̂ is
    /// the standard MCMC convergence diagnostic). Renderers must not
    /// merge the two columns. See proposal §2.
    pub max_chain_agreement: f64,
    pub gate_verdict: GateVerdict,
    /// Particle-filter ESS evaluated at the clean-eval winner θ̂.
    /// `None` when the stage didn't compute one (e.g. clean-eval was
    /// disabled or the file is absent).
    pub ess_at_mle: Option<EssSummary>,
    pub n_chains: usize,
    pub n_iter: usize,
}

/// PF ESS summary at the IF2 winner θ̂. Three numbers; renderers can
/// pick whichever the table needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EssSummary {
    pub ess_min: f64,
    pub ess_mean: f64,
    /// Index of the observation step where ESS hit its minimum
    /// (1-based for human-readable display, matches
    /// `chain_evaluations.tsv` convention; `None` when ESS computation
    /// failed across the board).
    pub ess_min_step: Option<usize>,
}

/// PGAS-stage result: posterior approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PgasStageResult {
    /// Number of post-burn-in thinned posterior samples (across all
    /// chains, as written to `draws.tsv`).
    pub n_samples: usize,
    pub posterior_mean: BTreeMap<String, f64>,
    pub posterior_q025: BTreeMap<String, f64>,
    pub posterior_q975: BTreeMap<String, f64>,
    pub ess_per_param: BTreeMap<String, f64>,
    /// Maximum Gelman-Rubin R̂ over estimated params. **R̂, not the
    /// IF2 Â** — see the comment on `If2StageResult.max_chain_agreement`.
    pub max_rhat: f64,
    pub acceptance_per_param: BTreeMap<String, f64>,
    pub n_chains: usize,
}

/// NLopt-stage result: deterministic MLE on the ODE skeleton (Phase 1
/// of the ODE-inference proposal). Mirrors `If2StageResult`'s point-
/// estimate shape, with NLopt-specific convergence info (status and
/// per-param chain spread) replacing IF2's chain-agreement Â and
/// PF-ESS columns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NloptStageResult {
    pub best_loglik: f64,
    pub best_chain: usize,
    /// Winner θ̂ — estimated parameters only. Fixed params live in
    /// `mle_params.toml` (full vector with provenance).
    pub theta_hat: BTreeMap<String, f64>,
    /// Which NLopt algorithm produced this result: `"nl-sbplx"` or
    /// `"nl-bobyqa"`.
    pub algorithm: String,
    /// Number of converged (`Success` / `XtolReached` / `FtolReached`)
    /// chains. `n_chains - n_converged` includes both `MaxEvalReached`
    /// (soft failure) and `Failed` (hard error).
    pub n_converged: usize,
    /// Maximum per-param relative range across chains (range / bound
    /// width). NA when n_chains == 1.
    pub max_rel_range: f64,
    pub n_chains: usize,
}

/// PMMH-stage result: posterior approximation + scalar acceptance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PmmhStageResult {
    pub n_samples: usize,
    pub posterior_mean: BTreeMap<String, f64>,
    pub ess: BTreeMap<String, f64>,
    pub max_rhat: f64,
    /// Scalar across chains (mean of per-chain rates). PGAS reports
    /// per-parameter rates because its inner Gibbs proposes parameters
    /// one at a time; PMMH proposes the full vector each step.
    pub acceptance_rate: f64,
    pub map_loglik: f64,
    pub n_chains: usize,
}

/// Errors loading a `MethodResult` from a stage directory.
#[derive(Debug)]
pub enum MethodResultError {
    /// `run.json` named a method this ADT doesn't carry. New methods
    /// get added by extending the enum, which produces compile errors
    /// at every consumer that doesn't handle them yet — exactly what
    /// this typing was designed to surface.
    UnknownMethod {
        method: String,
        stage_dir: PathBuf,
    },
    /// A required artifact was missing or unreadable.
    Io {
        stage_dir: PathBuf,
        message: String,
    },
}

impl std::fmt::Display for MethodResultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MethodResultError::UnknownMethod { method, stage_dir } => write!(
                f,
                "unknown fit-stage method `{}` at {} (expected if2, pgas, \
                 pmmh, nl-sbplx, or nl-bobyqa)",
                method,
                stage_dir.display()
            ),
            MethodResultError::Io { stage_dir, message } => {
                write!(f, "loading {}: {}", stage_dir.display(), message)
            }
        }
    }
}

impl std::error::Error for MethodResultError {}

impl MethodResult {
    /// Dispatch on `method` and load the matching variant. Errors on
    /// unknown methods rather than silently producing a generic shape.
    pub fn load_from(stage_dir: &Path, method: &str) -> Result<Self, MethodResultError> {
        match method {
            "if2" => Ok(MethodResult::If2(If2StageResult::load(stage_dir)?)),
            "pgas" => Ok(MethodResult::Pgas(PgasStageResult::load(stage_dir)?)),
            "pmmh" => Ok(MethodResult::Pmmh(PmmhStageResult::load(stage_dir)?)),
            "nl-sbplx" | "nl-bobyqa" => Ok(MethodResult::Nlopt(
                NloptStageResult::load(stage_dir, method)?,
            )),
            unknown => Err(MethodResultError::UnknownMethod {
                method: unknown.to_string(),
                stage_dir: stage_dir.to_owned(),
            }),
        }
    }
}

/// Read the stage leaf's `inputs` map from its `run.json` (`runid::RunRecord`).
/// Convenience for loaders that need stage-level metadata (`n_chains`,
/// `algorithm`) recorded alongside the leaf. Errors with a typed
/// [`MethodResultError::Io`] on a missing / malformed file or a non-`FitStage`
/// record — the contract is "every fit-stage writes a `RunRecord` leaf whose
/// `inputs` carries its config".
fn read_stage_inputs(stage_dir: &Path) -> Result<serde_json::Value, MethodResultError> {
    let bytes = std::fs::read(stage_dir.join("run.json")).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("run.json: {}", e),
    })?;
    let rec: runid::RunRecord = serde_json::from_slice(&bytes).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("run.json: {}", e),
    })?;
    if rec.kind != runid::ArtifactKind::FitStage {
        return Err(MethodResultError::Io {
            stage_dir: stage_dir.to_owned(),
            message: "run.json is not a fit-stage".into(),
        });
    }
    Ok(rec.inputs)
}

/// `inputs.n_chains` as a `usize` (0 when absent).
fn inputs_n_chains(inputs: &serde_json::Value) -> usize {
    inputs.get("n_chains").and_then(|v| v.as_u64()).unwrap_or(0) as usize
}

// ── NloptStageResult ────────────────────────────────────────────────

impl NloptStageResult {
    pub fn load(stage_dir: &Path, method: &str) -> Result<Self, MethodResultError> {
        let n_chains = inputs_n_chains(&read_stage_inputs(stage_dir)?);
        let state = FitState::load(&stage_dir.to_string_lossy()).map_err(|e| {
            MethodResultError::Io {
                stage_dir: stage_dir.to_owned(),
                message: format!("fit_state.toml: {}", e),
            }
        })?;

        // theta_hat: estimated parameters only. NLopt's tail_chain_agreement
        // (per-param relative range) carries one entry per estimated param
        // — same trick the IF2 loader uses to filter to the estimated set.
        let theta_hat: BTreeMap<String, f64> = if state.tail_chain_agreement.is_empty() {
            state.start_values.iter().map(|(k, v)| (k.clone(), *v)).collect()
        } else {
            state
                .tail_chain_agreement
                .keys()
                .filter_map(|name| state.start_values.get(name).map(|v| (name.clone(), *v)))
                .collect()
        };

        let max_rel_range = state
            .tail_chain_agreement
            .values()
            .copied()
            .fold(0.0_f64, f64::max);

        Ok(NloptStageResult {
            best_loglik: state.best_loglik,
            best_chain: state.best_chain,
            theta_hat,
            algorithm: method.to_string(),
            n_converged: state.n_good_chains.unwrap_or(n_chains),
            max_rel_range,
            n_chains,
        })
    }
}

// ── If2StageResult ──────────────────────────────────────────────────

impl If2StageResult {
    pub fn load(stage_dir: &Path) -> Result<Self, MethodResultError> {
        let inputs = read_stage_inputs(stage_dir)?;
        let n_chains = inputs_n_chains(&inputs);

        let state = FitState::load(&stage_dir.to_string_lossy()).map_err(|e| {
            MethodResultError::Io {
                stage_dir: stage_dir.to_owned(),
                message: format!("fit_state.toml: {}", e),
            }
        })?;

        // theta_hat: estimated parameters only. We don't have direct
        // access to the [estimate] block from this side of the
        // pipeline, but `tail_chain_agreement` only contains estimated
        // params — we use that as the key set. When tail_chain_agreement
        // is empty (legacy file), fall back to the full start_values
        // map (which over-includes fixed params, but is the best we
        // can do without re-reading fit.toml).
        let theta_hat: BTreeMap<String, f64> = if state.tail_chain_agreement.is_empty() {
            state
                .start_values
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect()
        } else {
            state
                .tail_chain_agreement
                .keys()
                .filter_map(|name| state.start_values.get(name).map(|v| (name.clone(), *v)))
                .collect()
        };

        let max_chain_agreement = state
            .tail_chain_agreement
            .values()
            .copied()
            .fold(0.0_f64, f64::max);

        let gate_verdict = compute_if2_gate_verdict(&state);

        let ess_at_mle = read_ess_at_mle(stage_dir);

        let n_iter = inputs
            .get("algorithm")
            .and_then(|a| a.get("iterations"))
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);

        Ok(If2StageResult {
            best_loglik: state.best_loglik,
            best_chain: state.best_chain,
            theta_hat,
            max_chain_agreement,
            gate_verdict,
            ess_at_mle,
            n_chains,
            n_iter,
        })
    }
}

/// Apply the compound IF2 gate to the persisted FitState. Mirrors
/// `Formatter::gate_verdict_block` in `fit_summary.rs` but returns a
/// typed verdict instead of a rendered block. Falls back to
/// `GateConfig::default()` when `state.resolved_gate` is absent
/// (legacy file pre-Phase-3) — that mirrors the existing summary's
/// fallback. The `loglik_eval` data drives the decibans leg; absent →
/// the leg is inconclusive and we judge on Â alone.
fn compute_if2_gate_verdict(state: &FitState) -> GateVerdict {
    use crate::evidence::NATS_TO_DB;
    use crate::fit::config_v2::GateConfig;

    let gate = state
        .resolved_gate
        .clone()
        .unwrap_or_else(GateConfig::default);

    let max_a = state
        .tail_chain_agreement
        .values()
        .copied()
        .fold(0.0_f64, f64::max);
    let a_passes = max_a < gate.a_thresh;

    let db_passes = if state.chain_eval_logliks.len() >= 2
        && state.chain_eval_ses.len() == state.chain_eval_logliks.len()
    {
        let hi = state
            .chain_eval_logliks
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lo = state
            .chain_eval_logliks
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let delta_db = (hi - lo) * NATS_TO_DB;
        let sigma_max = state.chain_eval_ses.iter().copied().fold(0.0_f64, f64::max);
        let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
        let threshold_db = gate.decibans_thresh.max(se_floor_db);
        Some(delta_db < threshold_db)
    } else {
        None
    };

    match (a_passes, db_passes) {
        (true, Some(true)) | (true, None) => GateVerdict::Pass,
        (false, Some(false)) => GateVerdict::FailBoth,
        (false, _) => GateVerdict::FailA,
        (true, Some(false)) => GateVerdict::FailDb,
    }
}

/// Read ESS-at-MLE from `chain_evaluations.tsv` if present. The TSV
/// schema (set in `runner::write_clean_eval_tsv`):
/// `chain candidate loglik se ess_mean ess_min ess_min_step n_neg_inf_incr <param₁> ...`
///
/// We pick the row corresponding to the overall winner (max loglik
/// across all rows), since clean-eval re-scoring is done per
/// (chain × candidate) and the IF2's `best_chain` is the overall
/// max. Returns `None` if the file is absent or malformed.
fn read_ess_at_mle(stage_dir: &Path) -> Option<EssSummary> {
    let path = stage_dir.join("chain_evaluations.tsv");
    let contents = std::fs::read_to_string(&path).ok()?;
    let mut header: Option<Vec<String>> = None;
    let mut best_row: Option<(f64, EssSummary)> = None;
    for line in contents.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if header.is_none() {
            header = Some(cols.iter().map(|s| (*s).to_string()).collect());
            continue;
        }
        let h = header.as_ref().unwrap();
        let idx = |name: &str| -> Option<usize> { h.iter().position(|c| c == name) };
        let loglik = idx("loglik").and_then(|i| cols.get(i)).and_then(|s| s.parse::<f64>().ok())?;
        let ess_mean = idx("ess_mean").and_then(|i| cols.get(i)).and_then(|s| s.parse::<f64>().ok())?;
        let ess_min = idx("ess_min").and_then(|i| cols.get(i)).and_then(|s| s.parse::<f64>().ok())?;
        let ess_min_step = idx("ess_min_step")
            .and_then(|i| cols.get(i))
            .and_then(|s| s.parse::<i64>().ok())
            .map(|i| if i < 0 { None } else { Some(i as usize) })
            .unwrap_or(None);
        let summary = EssSummary {
            ess_min,
            ess_mean,
            ess_min_step,
        };
        match &best_row {
            Some((best_ll, _)) if loglik <= *best_ll => {}
            _ => best_row = Some((loglik, summary)),
        }
    }
    best_row.map(|(_, s)| s)
}

// ── PgasStageResult ─────────────────────────────────────────────────

impl PgasStageResult {
    pub fn load(stage_dir: &Path) -> Result<Self, MethodResultError> {
        let n_chains = inputs_n_chains(&read_stage_inputs(stage_dir)?);

        // Estimated parameter names from algorithm config — pgas
        // doesn't store them directly. Fall back to "every column in
        // draws.tsv" if not in algorithm.
        let summary = read_summary_json(stage_dir, "pgas_summary.json")?;

        let rhat_map: BTreeMap<String, f64> = summary
            .get("rhat")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                    .collect()
            })
            .unwrap_or_default();
        let ess_map: BTreeMap<String, f64> = summary
            .get("ess")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                    .collect()
            })
            .unwrap_or_default();

        // Posterior moments: average each estimated-param column in
        // draws.tsv. The estimated-param key set is rhat_map's keys
        // when present (rhat is per estimated param), else ess_map's.
        let est_names: Vec<String> = if !rhat_map.is_empty() {
            rhat_map.keys().cloned().collect()
        } else {
            ess_map.keys().cloned().collect()
        };
        let (n_samples, posterior_mean, posterior_q025, posterior_q975) =
            posterior_summaries(stage_dir, &est_names);

        // Acceptance per param: pgas writes acceptance_rates as
        // Vec<Vec<f64>> (n_chains × n_estimated). Aggregate to per-
        // param mean across chains.
        let acceptance_per_param: BTreeMap<String, f64> = {
            let raw = summary
                .get("acceptance_rates")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let n_chains = raw.len();
            if n_chains == 0 || est_names.is_empty() {
                BTreeMap::new()
            } else {
                let mut acc = vec![0.0_f64; est_names.len()];
                let mut counts = vec![0usize; est_names.len()];
                for chain in &raw {
                    let row = chain.as_array().cloned().unwrap_or_default();
                    for (i, v) in row.iter().enumerate() {
                        if i >= acc.len() {
                            break;
                        }
                        if let Some(x) = v.as_f64() {
                            acc[i] += x;
                            counts[i] += 1;
                        }
                    }
                }
                est_names
                    .iter()
                    .enumerate()
                    .map(|(i, name)| {
                        let denom = counts[i].max(1) as f64;
                        (name.clone(), acc[i] / denom)
                    })
                    .collect()
            }
        };

        let max_rhat = rhat_map.values().copied().fold(0.0_f64, f64::max);

        Ok(PgasStageResult {
            n_samples,
            posterior_mean,
            posterior_q025,
            posterior_q975,
            ess_per_param: ess_map,
            max_rhat,
            acceptance_per_param,
            n_chains,
        })
    }
}

// ── PmmhStageResult ─────────────────────────────────────────────────

impl PmmhStageResult {
    pub fn load(stage_dir: &Path) -> Result<Self, MethodResultError> {
        let n_chains = inputs_n_chains(&read_stage_inputs(stage_dir)?);
        let summary = read_summary_json(stage_dir, "pmmh_summary.json")?;

        let rhat_map: BTreeMap<String, f64> = summary
            .get("rhat")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                    .collect()
            })
            .unwrap_or_default();
        let ess_map: BTreeMap<String, f64> = summary
            .get("ess")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
                    .collect()
            })
            .unwrap_or_default();
        let est_names: Vec<String> = if !rhat_map.is_empty() {
            rhat_map.keys().cloned().collect()
        } else {
            ess_map.keys().cloned().collect()
        };
        let (n_samples, posterior_mean, _q025, _q975) =
            posterior_summaries(stage_dir, &est_names);

        // PMMH writes acceptance_rate as Vec<f64> (one per chain). The
        // table-row scalar is the mean across chains.
        let acceptance_rate = summary
            .get("acceptance_rate")
            .and_then(|v| v.as_array())
            .map(|a| {
                let v: Vec<f64> = a.iter().filter_map(|x| x.as_f64()).collect();
                if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f64>() / v.len() as f64
                }
            })
            .unwrap_or(0.0);
        let map_loglik = summary
            .get("map_loglik")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NEG_INFINITY);
        let max_rhat = rhat_map.values().copied().fold(0.0_f64, f64::max);

        Ok(PmmhStageResult {
            n_samples,
            posterior_mean,
            ess: ess_map,
            max_rhat,
            acceptance_rate,
            map_loglik,
            n_chains,
        })
    }
}

// ── shared helpers ──────────────────────────────────────────────────

/// Read a stage's summary JSON (`pgas_summary.json` /
/// `pmmh_summary.json`) into a serde value. These files are written
/// by the runners and persist scalar diagnostics that aren't in
/// fit_state.toml.
fn read_summary_json(
    stage_dir: &Path,
    filename: &str,
) -> Result<serde_json::Value, MethodResultError> {
    let path = stage_dir.join(filename);
    let contents = std::fs::read_to_string(&path).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("{}: {}", filename, e),
    })?;
    serde_json::from_str(&contents).map_err(|e| MethodResultError::Io {
        stage_dir: stage_dir.to_owned(),
        message: format!("{}: parse error: {}", filename, e),
    })
}

/// Read `<stage>/draws.tsv` and compute (n_samples, mean, q025, q975)
/// for each param in `est_names`. `draws.tsv` is the canonical
/// posterior-sample table written by both PGAS and PMMH (post-
/// burn-in, thinned, all params).
fn posterior_summaries(
    stage_dir: &Path,
    est_names: &[String],
) -> (
    usize,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
) {
    let path = stage_dir.join("draws.tsv");
    let mut mean = BTreeMap::new();
    let mut q025 = BTreeMap::new();
    let mut q975 = BTreeMap::new();
    let contents = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return (0, mean, q025, q975),
    };
    let mut lines = contents.lines();
    let header = match lines.next() {
        Some(h) => h,
        None => return (0, mean, q025, q975),
    };
    let cols: Vec<&str> = header.split('\t').collect();
    // Build column indices for the estimated-param subset.
    let mut col_idx: Vec<(String, usize)> = Vec::new();
    for name in est_names {
        if let Some(i) = cols.iter().position(|c| c == name) {
            col_idx.push((name.clone(), i));
        }
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); col_idx.len()];
    let mut n = 0_usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        for (k, (_, ci)) in col_idx.iter().enumerate() {
            if let Some(s) = fields.get(*ci) {
                if let Ok(v) = s.parse::<f64>() {
                    samples[k].push(v);
                }
            }
        }
        n += 1;
    }
    for (k, (name, _)) in col_idx.iter().enumerate() {
        let v = &mut samples[k];
        if v.is_empty() {
            continue;
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        mean.insert(name.clone(), m);
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pick = |q: f64| -> f64 {
            let idx = ((v.len() - 1) as f64 * q).round() as usize;
            v[idx.min(v.len() - 1)]
        };
        q025.insert(name.clone(), pick(0.025));
        q975.insert(name.clone(), pick(0.975));
    }
    (n, mean, q025, q975)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};
    use std::collections::HashMap;

    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir(tag: &str) -> TempDir {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "camdl_methodresult_{}_{}_{}",
            tag,
            std::process::id(),
            ns
        ));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    /// Write a `FitStage` `runid::RunRecord` leaf at `dir` carrying the
    /// `inputs` the loaders read (`method`, `n_chains`, `algorithm`).
    fn write_stage_run(dir: &Path, method: crate::run_meta::FitAlgorithm, n_chains: usize, algorithm: serde_json::Value) {
        std::fs::create_dir_all(dir).unwrap();
        let rec = serde_json::json!({
            "format_version": 1,
            "kind": "fit_stage",
            "run_id": "deadbeef".repeat(8),
            "hash_version": 1,
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {"name": "fit",   "label": "fit",   "hash": "f00d".repeat(16), "schema_version": 1},
                {"name": "stage", "label": "01-scout", "hash": "1fb03eee00000000000000000000000000000000000000000000000000000000", "schema_version": 1},
                {"name": "seed",  "label": "seed_1", "hash": "06cbd6b300000000000000000000000000000000000000000000000000000000", "schema_version": 1}
            ],
            "status": "completed",
            "artifacts": {},
            "inputs": {
                "stage": "scout",
                "method": method.as_str(),
                "backend": "chain_binomial",
                "seed": 1,
                "n_chains": n_chains,
                "algorithm": algorithm,
                "best_loglik": -100.0,
                "best_chain": 0
            },
            "provenance": {"created_at": "2026-04-27T00:00:00Z", "argv": ["camdl"]}
        });
        std::fs::write(dir.join("run.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    }

    fn synthetic_if2_state() -> FitState {
        let mut start_values = HashMap::new();
        start_values.insert("R0".into(), 56.8);
        start_values.insert("sigma".into(), 0.115);
        start_values.insert("N0".into(), 1000.0); // fixed param, no Â
        let mut agreement = HashMap::new();
        agreement.insert("R0".into(), 1.04);
        agreement.insert("sigma".into(), 1.01);
        FitState {
            stage: "scout".into(),
            seed: 42,
            timestamp: "2026-04-27T00:00:00Z".into(),
            input_hash: None,
            camdl_version: Some("0.1.0+test".into()),
            best_loglik: -3804.9,
            initial_loglik: -7000.0,
            best_chain: 1,
            n_chains: 4,
            n_good_chains: Some(4),
            start_values,
            rw_sd: HashMap::new(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: agreement,
            ivp_params: vec![],
            chain_logliks: vec![-3810.0, -3805.0, -3812.0, -3804.9],
            chain_eval_logliks: vec![-3810.0, -3805.0, -3812.0, -3804.9],
            chain_eval_ses: vec![1.0, 1.0, 1.0, 1.0],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            dt_check: None,
        }
    }

    #[test]
    fn loads_if2_stage_result_pass_verdict() {
        let tmp = tempdir("if2_pass");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            4,
            serde_json::json!({"method": "if2", "iterations": 50}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();

        let r = If2StageResult::load(dir).unwrap();
        assert_eq!(r.n_chains, 4);
        assert_eq!(r.n_iter, 50);
        assert_eq!(r.best_chain, 1);
        assert!((r.best_loglik - (-3804.9)).abs() < 1e-9);
        // theta_hat restricted to estimated params (those with Â).
        assert_eq!(r.theta_hat.len(), 2);
        assert!(r.theta_hat.contains_key("R0"));
        assert!(r.theta_hat.contains_key("sigma"));
        assert!(!r.theta_hat.contains_key("N0"),
            "fixed param N0 must not appear in theta_hat");
        // Â passes (max=1.04 < 1.01? actually 1.04 > 1.01, so this should FAIL Â).
        // Default a_thresh is 1.01, max Â = 1.04, so a_passes = false.
        // Decibans spread is small, db_passes = true.
        // Verdict: FailA.
        assert_eq!(r.gate_verdict, GateVerdict::FailA);
        assert!((r.max_chain_agreement - 1.04).abs() < 1e-9);
    }

    #[test]
    fn if2_pass_verdict_when_thresholds_clear() {
        let tmp = tempdir("if2_clean");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            4,
            serde_json::json!({"method": "if2", "iterations": 50}),
        );
        let mut state = synthetic_if2_state();
        state.tail_chain_agreement.insert("R0".into(), 1.005);
        state.tail_chain_agreement.insert("sigma".into(), 1.002);
        state.save(&dir.to_string_lossy()).unwrap();
        let r = If2StageResult::load(dir).unwrap();
        assert_eq!(r.gate_verdict, GateVerdict::Pass);
    }

    #[test]
    fn loads_if2_ess_from_chain_evaluations_tsv() {
        let tmp = tempdir("if2_ess");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            2,
            serde_json::json!({"method": "if2", "iterations": 10}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();
        std::fs::write(
            dir.join("chain_evaluations.tsv"),
            "# camdl 0.1.0+test\n\
             chain\tcandidate\tloglik\tse\tess_mean\tess_min\tess_min_step\tn_neg_inf_incr\tR0\n\
             1\tfinal_iter\t-3805.1\t1.0\t850.0\t412.0\t17\t0\t56.8\n\
             1\ttail_mean\t-3810.0\t1.0\t800.0\t300.0\t12\t0\t56.7\n\
             2\tfinal_iter\t-3812.0\t1.0\t820.0\t380.0\t15\t0\t57.0\n",
        )
        .unwrap();
        let r = If2StageResult::load(dir).unwrap();
        let ess = r.ess_at_mle.expect("ESS summary must be loaded");
        // Best (highest loglik) row is chain 1 / final_iter at -3805.1.
        assert!((ess.ess_min - 412.0).abs() < 1e-9);
        assert!((ess.ess_mean - 850.0).abs() < 1e-9);
        assert_eq!(ess.ess_min_step, Some(17));
    }

    #[test]
    fn ess_at_mle_none_when_file_absent() {
        let tmp = tempdir("if2_no_ess");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            2,
            serde_json::json!({"method": "if2", "iterations": 10}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();
        let r = If2StageResult::load(dir).unwrap();
        assert!(r.ess_at_mle.is_none());
    }

    #[test]
    fn loads_pgas_stage_result() {
        let tmp = tempdir("pgas");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::Pgas,
            2,
            serde_json::json!({"method": "pgas", "sweeps": 100}),
        );
        // pgas_summary.json, schema matched to pgas::write_summary.
        std::fs::write(
            dir.join("pgas_summary.json"),
            serde_json::to_string(&serde_json::json!({
                "stage": "pgas",
                "n_chains": 2,
                "acceptance_rates": [[0.32, 0.35], [0.28, 0.30]],
                "rhat": {"R0": 1.02, "sigma": 1.04},
                "ess": {"R0": 850.0, "sigma": 412.0}
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("draws.tsv"),
            "R0\tsigma\tN0\n\
             56.8\t0.115\t1000.0\n\
             57.1\t0.110\t1000.0\n\
             56.5\t0.118\t1000.0\n\
             57.3\t0.112\t1000.0\n",
        )
        .unwrap();

        let r = PgasStageResult::load(dir).unwrap();
        assert_eq!(r.n_chains, 2);
        assert_eq!(r.n_samples, 4);
        // BTreeMap order: alphabetic. R0 first, sigma second.
        let r0 = r.posterior_mean["R0"];
        assert!((r0 - (56.8 + 57.1 + 56.5 + 57.3) / 4.0).abs() < 1e-9);
        // R̂ map present, max captured.
        assert!((r.max_rhat - 1.04).abs() < 1e-9);
        // Acceptance per param: chain-mean. R0 col 0: (0.32 + 0.28)/2 = 0.30.
        assert!((r.acceptance_per_param["R0"] - 0.30).abs() < 1e-9);
        // ESS comes through.
        assert!((r.ess_per_param["sigma"] - 412.0).abs() < 1e-9);
    }

    #[test]
    fn loads_pmmh_stage_result() {
        let tmp = tempdir("pmmh");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::Pmmh,
            2,
            serde_json::json!({"method": "pmmh", "iterations": 50}),
        );
        std::fs::write(
            dir.join("pmmh_summary.json"),
            serde_json::to_string(&serde_json::json!({
                "stage": "pmmh",
                "n_chains": 2,
                "acceptance_rate": [0.20, 0.30],
                "rhat": {"R0": 1.03},
                "ess": {"R0": 600.0},
                "map_loglik": -3801.4,
                "map_chain": 1
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("draws.tsv"),
            "R0\tN0\n\
             57.0\t1000.0\n\
             57.5\t1000.0\n",
        )
        .unwrap();

        let r = PmmhStageResult::load(dir).unwrap();
        assert_eq!(r.n_chains, 2);
        assert_eq!(r.n_samples, 2);
        // Mean over the two posterior samples for R0.
        assert!((r.posterior_mean["R0"] - 57.25).abs() < 1e-9);
        assert!((r.acceptance_rate - 0.25).abs() < 1e-9);
        assert!((r.map_loglik - (-3801.4)).abs() < 1e-9);
        assert!((r.max_rhat - 1.03).abs() < 1e-9);
    }

    #[test]
    fn load_from_dispatches_on_method_string() {
        let tmp = tempdir("dispatch");
        let dir = tmp.path();
        write_stage_run(
            dir,
            crate::run_meta::FitAlgorithm::If2,
            2,
            serde_json::json!({"method": "if2", "iterations": 5}),
        );
        synthetic_if2_state()
            .save(&dir.to_string_lossy())
            .unwrap();
        let r = MethodResult::load_from(dir, "if2").unwrap();
        assert!(matches!(r, MethodResult::If2(_)));

        let err = MethodResult::load_from(dir, "if4").unwrap_err();
        match err {
            MethodResultError::UnknownMethod { method, .. } => assert_eq!(method, "if4"),
            other => panic!("expected UnknownMethod, got {:?}", other),
        }
    }

    /// The proposal pins `gate_verdict` strings to `pass` / `fail_a`
    /// / `fail_db` / `fail_both` (proposal §2). `serde_json` is the
    /// projection — assert the rendered scalar matches exactly so a
    /// future `rename_all` change to the enum doesn't silently shift
    /// the wire format.
    #[test]
    fn gate_verdict_serializes_to_proposal_strings() {
        for (variant, expected) in [
            (GateVerdict::Pass, "pass"),
            (GateVerdict::FailA, "fail_a"),
            (GateVerdict::FailDb, "fail_db"),
            (GateVerdict::FailBoth, "fail_both"),
        ] {
            let s = serde_json::to_value(variant).unwrap();
            assert_eq!(s, serde_json::Value::String(expected.into()));
        }
    }
}
