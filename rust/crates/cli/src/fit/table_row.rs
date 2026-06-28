//! `table_row` — the schema shared by `fit summary --format json` and
//! `fit table --format json`.
//!
//! One fit projects to one `TableRow`. `fit summary` embeds a
//! `table_row` block in its top-level JSON; `fit table` produces a
//! list of them. **Deliverable C** asserts these are byte-equal for
//! the same fit (`docs/dev/proposals/2026-04-28-fit-experiment-management.md` §3).
//!
//! Map fields (`params`, `ess_posterior`) use [`BTreeMap<String, _>`]
//! end-to-end so `serde_json` produces lex-ordered output. A
//! `HashMap` anywhere on this serialization graph would make the
//! byte-equality test flake on key-order changes.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use crate::fit::config_diff::ConfigDiff;
use crate::fit::fit_tree::{self, StageNode};
use crate::fit::fit_view::FitView;
use crate::fit::method_result::{
    EssSummary, GateVerdict, If2StageResult, MethodResult, MethodResultError,
    PgasStageResult, PmmhStageResult,
};

/// Schema discriminator. The proposal pins `name = "table_row"` and
/// `version = 1`. Field additions are non-breaking under v1; removals
/// or semantic changes require a v2 emitted side-by-side for one
/// minor release before v1 is dropped (proposal §3, Schema stability
/// post-ship).
pub const SCHEMA_NAME: &str = "table_row";
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TableRowSchema {
    pub name: String,
    pub version: u32,
}

impl TableRowSchema {
    pub fn current() -> Self {
        Self {
            name: SCHEMA_NAME.into(),
            version: SCHEMA_VERSION,
        }
    }
}

/// One fit, projected to a single row.
///
/// Field ordering matches proposal §3 (table_row schema v1) and is
/// deliberately preserved so JSON-pretty rendering reads top-to-bottom
/// in the documented order.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TableRow {
    pub schema: TableRowSchema,
    /// `fit_hash[..8]` — short, human-quotable.
    pub fit_id: String,
    /// Full 64-char fit-level hash (`FitView.fit_hash`).
    pub fit_hash: String,
    /// User-supplied display label, set at fit-run time via
    /// `--label "..."` or post-hoc via `camdl label <hash>
    /// "<text>"`. None when the user hasn't labelled the fit.
    pub label: Option<String>,
    /// fit.toml stem — the basename of `FitView.fit_toml_path` minus
    /// trailing `.fit.toml` / `.toml` extension.
    pub stem: String,
    pub model_identity: String,
    /// Stage names that completed under this fit, in declaration order
    /// (filtered by `FitView.stages_declared`). A multi-stage IF2 fit
    /// reads `["scout", "refine", "validate"]`; a PGAS-only fit reads
    /// `["pgas"]`. Stages that didn't complete are excluded.
    pub stages: Vec<String>,
    /// Method of the *terminal* completed stage (last in
    /// `stages_declared` that has a `run.json` on disk).
    pub method: String,
    pub config_diff_from_baseline: ConfigDiff,
    /// Method-uniform convergence boolean (proposal §3, `converged`):
    /// IF2 → `gate_verdict == Pass`; PGAS / PMMH → `max_rhat < 1.05`.
    pub converged: bool,
    /// IF2: one of `pass | fail_a | fail_db | fail_both`. PGAS / PMMH:
    /// `"n/a"` (the IF2 gate doesn't apply).
    pub gate_verdict: String,
    /// IF2 → best clean-eval loglik. PMMH → `map_loglik`. PGAS →
    /// `null` (no point estimate).
    pub best_loglik: Option<f64>,
    /// The class of `best_loglik` (gh#280): `complete_data` for PGAS's
    /// joint, a marginal kind otherwise. Always populated (derived from the
    /// typed `MethodResult`); present even for PGAS, whose `best_loglik` is
    /// `null`. `None` only on the error-placeholder row.
    pub loglik_type: Option<crate::fit::loglik::LoglikType>,
    /// Maximum chain-agreement Â (NOT Gelman-Rubin). IF2 only; null
    /// otherwise. See `If2StageResult.max_chain_agreement`.
    pub max_chain_agreement: Option<f64>,
    /// Maximum Gelman-Rubin R̂. PGAS / PMMH only; null otherwise.
    pub max_rhat: Option<f64>,
    /// Scalar acceptance rate. PMMH only — PGAS reports per-param
    /// acceptances which do not project to a uniform scalar; IF2 has
    /// no acceptance.
    pub acceptance_rate: Option<f64>,
    /// Particle-filter ESS at θ̂. IF2 only.
    pub ess_at_mle: Option<EssSummary>,
    /// Posterior chain ESS per parameter. PGAS / PMMH only.
    pub ess_posterior: Option<BTreeMap<String, f64>>,
    /// Estimated parameters. IF2 → θ̂ (clean-eval winner); PGAS / PMMH
    /// → posterior mean. The full estimated set is included; renderers
    /// truncate.
    pub params: BTreeMap<String, f64>,
    /// Loglik delta vs the best fit in the current scope. `0.0` when
    /// the row is alone in scope (single-fit `summary` or
    /// `--hash <h>` filter).
    pub delta_ll_vs_best: f64,
    /// Wall time since `Run.created_at` (seconds; `i64` because system
    /// clock skew can produce small negative deltas).
    pub age_seconds: i64,
    pub created_at: String,
    /// Step-7 stale flag. Today always `false`.
    pub stale: bool,
    pub stale_reason: Option<String>,
    /// Per-fit generated-quantity medians, keyed by quantity name.
    /// Populated only when `fit table --quantity <NAME>` is requested:
    /// each entry is the posterior median (q50 of the `as_fitted` row)
    /// of a scalar quantity from the model's `quantities {}` block,
    /// read from `<fit_dir>/quantities/<NAME>.tsv` (derived on demand
    /// via `fit predict` when absent). An uncomputable quantity for a
    /// row is simply absent from the map. A field addition is
    /// non-breaking under schema v1 (proposal §3); the
    /// `skip_serializing_if` empty-map guard keeps the no-quantity case
    /// byte-identical to the pre-existing serialization (so the
    /// `summary ⊆ table` parity invariant holds).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub quantities: BTreeMap<String, f64>,
}

/// Errors building a `TableRow` from a fit_dir.
#[derive(Debug)]
pub enum TableRowError {
    /// The fit_dir's top-level `run.json` couldn't be read.
    TopLevelRun {
        fit_dir: std::path::PathBuf,
        message: String,
    },
    /// The walker found no completed stages under this fit_dir.
    NoStages { fit_dir: std::path::PathBuf },
    /// Loading the terminal stage's `MethodResult` failed.
    Method(MethodResultError),
}

impl std::fmt::Display for TableRowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableRowError::TopLevelRun { fit_dir, message } => write!(
                f,
                "loading top-level run.json at {}: {}",
                fit_dir.display(),
                message
            ),
            TableRowError::NoStages { fit_dir } => write!(
                f,
                "no completed fit-stage runs found under {}",
                fit_dir.display()
            ),
            TableRowError::Method(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for TableRowError {}

impl From<MethodResultError> for TableRowError {
    fn from(e: MethodResultError) -> Self {
        TableRowError::Method(e)
    }
}

/// Build one `TableRow` for `fit_dir`. The `config_diff` argument is
/// computed by the caller (`fit table` picks a baseline from the
/// cohort; `fit summary` uses [`ConfigDiff::identity`]). `delta_ll`
/// and `now_unix` are also caller-supplied so the same row builder
/// powers single-fit and cohort callers without baking in scope
/// assumptions.
pub fn build_row(
    fit_dir: &Path,
    config_diff: ConfigDiff,
    delta_ll_vs_best: f64,
    now_unix: i64,
) -> Result<TableRow, TableRowError> {
    let view = FitView::read(fit_dir).ok_or_else(|| TableRowError::TopLevelRun {
        fit_dir: fit_dir.to_path_buf(),
        message: "not a fit segment (no FitStage leaves + sidecar)".into(),
    })?;

    let nodes = fit_tree::walk_fit_dir(fit_dir).map_err(|e| TableRowError::TopLevelRun {
        fit_dir: fit_dir.to_path_buf(),
        message: format!("walker: {}", e),
    })?;
    let terminal = pick_terminal_stage(&view, &nodes).ok_or_else(|| {
        TableRowError::NoStages {
            fit_dir: fit_dir.to_path_buf(),
        }
    })?;
    let completed_stages = completed_stage_names(&view, &nodes);

    let method = terminal.stage.method.as_str().to_string();
    let method_result = MethodResult::load_from(&terminal.stage_dir, &method)?;
    let mview = MethodView::from(&method_result);

    let stem = stem_from_fit_toml_path(&view.fit_toml_path);
    let created = parse_iso8601_to_unix(&view.created_at);
    let age_seconds = match created {
        Some(c) => now_unix - c,
        None => 0,
    };

    Ok(TableRow {
        schema: TableRowSchema::current(),
        fit_id: short_hash(&view.fit_hash),
        fit_hash: view.fit_hash.clone(),
        label: view.label.clone(),
        stem,
        model_identity: view.model_identity.clone(),
        stages: completed_stages,
        method,
        config_diff_from_baseline: config_diff,
        converged: mview.converged,
        gate_verdict: mview.gate_verdict,
        best_loglik: mview.best_loglik,
        loglik_type: Some(crate::fit::loglik::LoglikType::from(&method_result)),
        max_chain_agreement: mview.max_chain_agreement,
        max_rhat: mview.max_rhat,
        acceptance_rate: mview.acceptance_rate,
        ess_at_mle: mview.ess_at_mle,
        ess_posterior: mview.ess_posterior,
        params: mview.params,
        delta_ll_vs_best,
        age_seconds,
        created_at: view.created_at.clone(),
        stale: false,
        stale_reason: None,
        quantities: BTreeMap::new(),
    })
}

/// Method-projected view used internally to keep [`build_row`] flat
/// rather than carrying a giant pattern match inline.
struct MethodView {
    converged: bool,
    gate_verdict: String,
    best_loglik: Option<f64>,
    max_chain_agreement: Option<f64>,
    max_rhat: Option<f64>,
    acceptance_rate: Option<f64>,
    ess_at_mle: Option<EssSummary>,
    ess_posterior: Option<BTreeMap<String, f64>>,
    params: BTreeMap<String, f64>,
}

impl MethodView {
    fn from(r: &MethodResult) -> Self {
        match r {
            MethodResult::If2(if2) => Self::from_if2(if2),
            MethodResult::Pgas(pgas) => Self::from_pgas(pgas),
            MethodResult::Pmmh(pmmh) => Self::from_pmmh(pmmh),
            MethodResult::Nlopt(nl) => Self::from_nlopt(nl),
        }
    }

    fn from_nlopt(r: &super::method_result::NloptStageResult) -> Self {
        // Treat NLopt's two-leg gate as "all chains converged AND
        // max_rel_range below the proposal's first-pass threshold."
        // For single-chain runs the rel range is trivially 0.
        let converged = r.n_converged == r.n_chains
            && (r.n_chains < 2 || r.max_rel_range < 0.05);
        Self {
            converged,
            gate_verdict: if converged {
                "pass".into()
            } else if r.n_converged == r.n_chains {
                "fail_chain_agreement".into()
            } else {
                "fail_status".into()
            },
            best_loglik: Some(r.best_loglik),
            max_chain_agreement: if r.n_chains > 1 {
                Some(r.max_rel_range)
            } else {
                None
            },
            max_rhat: None,
            acceptance_rate: None,
            ess_at_mle: None,
            ess_posterior: None,
            params: r.theta_hat.clone(),
        }
    }

    fn from_if2(r: &If2StageResult) -> Self {
        let gate_str = serde_json::to_value(r.gate_verdict)
            .ok()
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            converged: matches!(r.gate_verdict, GateVerdict::Pass),
            gate_verdict: gate_str,
            best_loglik: Some(r.best_loglik),
            max_chain_agreement: Some(r.max_chain_agreement),
            max_rhat: None,
            acceptance_rate: None,
            ess_at_mle: r.ess_at_mle.clone(),
            ess_posterior: None,
            params: r.theta_hat.clone(),
        }
    }

    fn from_pgas(r: &PgasStageResult) -> Self {
        Self {
            converged: r.max_rhat < 1.05,
            gate_verdict: "n/a".into(),
            best_loglik: None,
            max_chain_agreement: None,
            max_rhat: Some(r.max_rhat),
            acceptance_rate: None,
            ess_at_mle: None,
            ess_posterior: Some(r.ess_per_param.clone()),
            params: r.posterior_mean.clone(),
        }
    }

    fn from_pmmh(r: &PmmhStageResult) -> Self {
        Self {
            converged: r.max_rhat < 1.05,
            gate_verdict: "n/a".into(),
            best_loglik: Some(r.map_loglik),
            max_chain_agreement: None,
            max_rhat: Some(r.max_rhat),
            acceptance_rate: Some(r.acceptance_rate),
            ess_at_mle: None,
            ess_posterior: Some(r.ess.clone()),
            params: r.posterior_mean.clone(),
        }
    }
}

/// Find the terminal completed stage. Walks `stages_declared` in
/// reverse and returns the first node whose stage name matches a
/// completed run. Picks Real over Synthetic, lowest `fit_seed`, then
/// lex-first stage_dir — same priority as
/// `fit_summary::resolve_if2_stage_dirs`.
fn pick_terminal_stage<'a>(
    view: &FitView,
    nodes: &'a [StageNode],
) -> Option<&'a StageNode> {
    for stage_name in view.stages_declared.iter().rev() {
        if let Some(node) = best_node_for_stage(stage_name, nodes) {
            return Some(node);
        }
    }
    // Falls back to whatever the walker found in lex order.
    nodes.first()
}

fn best_node_for_stage<'a>(stage_name: &str, nodes: &'a [StageNode]) -> Option<&'a StageNode> {
    use crate::fit::fit_tree::DataKind;
    let mut best: Option<(&'a StageNode, (u8, u64))> = None;
    for node in nodes {
        if node.stage.stage != stage_name {
            continue;
        }
        let rank: (u8, u64) = match &node.stage.axes {
            Some(axes) => {
                let kind_rank = match axes.data_kind {
                    DataKind::Real => 0,
                    DataKind::Synthetic { .. } => 1,
                };
                (kind_rank, axes.fit_seed)
            }
            None => (2, u64::MAX),
        };
        match best {
            Some((_, br)) if rank >= br => {}
            _ => best = Some((node, rank)),
        }
    }
    best.map(|(n, _)| n)
}

fn completed_stage_names(view: &FitView, nodes: &[StageNode]) -> Vec<String> {
    let completed: std::collections::HashSet<&str> =
        nodes.iter().map(|n| n.stage.stage.as_str()).collect();
    view.stages_declared
        .iter()
        .filter(|s| completed.contains(s.as_str()))
        .cloned()
        .collect()
}

fn short_hash(full: &str) -> String {
    full.chars().take(8).collect()
}

/// Strip the trailing `.fit.toml` or `.toml` extension off a fit.toml
/// path's basename. Falls back to the basename verbatim if it doesn't
/// end with either extension.
fn stem_from_fit_toml_path(path: &str) -> String {
    let p = Path::new(path);
    let name = p
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(path);
    if let Some(s) = name.strip_suffix(".fit.toml") {
        s.to_string()
    } else if let Some(s) = name.strip_suffix(".toml") {
        s.to_string()
    } else {
        name.to_string()
    }
}

/// Parse an ISO 8601 UTC timestamp ("2026-04-27T18:30:21Z") into a
/// Unix epoch second. Returns `None` on malformed input rather than
/// panicking — the caller (`age_seconds`) treats `None` as zero.
///
/// The runner emits exactly the format
/// `<YYYY>-<MM>-<DD>T<HH>:<MM>:<SS>Z`; this parser accepts that
/// strictly. A more permissive parser would invite drift.
fn parse_iso8601_to_unix(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    if bytes.len() != 20 || bytes[10] != b'T' || bytes[19] != b'Z' {
        return None;
    }
    let year: i32 = s[0..4].parse().ok()?;
    let month: u32 = s[5..7].parse().ok()?;
    let day: u32 = s[8..10].parse().ok()?;
    let hour: u32 = s[11..13].parse().ok()?;
    let minute: u32 = s[14..16].parse().ok()?;
    let second: u32 = s[17..19].parse().ok()?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let days = ir::caltime::unix_epoch_days(year as i64, month as i64, day as i64);
    Some(days * 86_400 + hour as i64 * 3600 + minute as i64 * 60 + second as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stem_strips_fit_toml_then_toml() {
        assert_eq!(stem_from_fit_toml_path("fit_he2010.fit.toml"), "fit_he2010");
        assert_eq!(stem_from_fit_toml_path("fit_he2010.toml"), "fit_he2010");
        assert_eq!(stem_from_fit_toml_path("models/fit_he2010.fit.toml"), "fit_he2010");
        assert_eq!(stem_from_fit_toml_path("oddname"), "oddname");
    }

    #[test]
    fn iso8601_round_trips_to_unix() {
        // 2026-04-27T18:30:21Z = 20570 days × 86400 + 18·3600 + 30·60 + 21
        // (independent check — `date -u -j -f "%FT%TZ" "2026-04-27T18:30:21Z" "+%s"`).
        assert_eq!(parse_iso8601_to_unix("2026-04-27T18:30:21Z"), Some(1_777_314_621));
        // Epoch.
        assert_eq!(parse_iso8601_to_unix("1970-01-01T00:00:00Z"), Some(0));
        // Leap-year edge: 2024-02-29 (Hinnant must accept Feb 29 in
        // leap years).
        assert_eq!(parse_iso8601_to_unix("2024-02-29T00:00:00Z"), Some(1_709_164_800));
        // Malformed → None, no panic.
        assert!(parse_iso8601_to_unix("garbage").is_none());
        assert!(parse_iso8601_to_unix("2026-04-27 18:30:21").is_none());
    }

    #[test]
    fn schema_pin_is_v1() {
        let s = TableRowSchema::current();
        assert_eq!(s.name, "table_row");
        assert_eq!(s.version, 1);
    }

    #[test]
    fn short_hash_takes_first_eight_chars() {
        assert_eq!(short_hash("0123456789abcdef0123456789abcdef"), "01234567");
        assert_eq!(short_hash("abc"), "abc");
    }
}
