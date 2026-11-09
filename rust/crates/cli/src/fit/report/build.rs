//! The one builder of a method's report.
//!
//! Every format `fit summary` writes, and the end of `fit run`, build a
//! [`MethodReport`] here from one [`MethodFacts`] (proposal
//! `docs/dev/proposals/2026-11-08-cli-reporting.md`, "The report").

use std::collections::BTreeMap;

use serde::Serialize;

use super::{Check, ChainWarmup, MethodFacts, Verdict};
use crate::chain_index::ChainIndex;
use crate::fit::config_v2::{GateConfig, LoglikEvalConfig};
use crate::fit::method_result::{If2Record, If2Run, MethodResult};
use crate::version;

/// Calendar context for date-rendering `instant`-kind estimands
/// (2026-05-22 calendar-time §6.7). Sourced once per `fit summary`
/// invocation from the fit dir's model (path via `FitView.model`),
/// then shared by every formatter. When the model declares no
/// `origin`, or cannot be loaded (moved path, parse error), rendering
/// degrades to numeric-only — `date_for` returns `None` and no
/// formatter changes shape.
#[derive(Debug, Clone, Default)]
pub(crate) struct CalendarContext {
    /// `origin` date string from the model (`Some` iff declared).
    pub(crate) origin: Option<String>,
    /// Model `time_unit` (`days`/`weeks`/`months`/`years`); the divisor
    /// for the internal-time ↔ date map.
    pub(crate) time_unit: String,
    /// Names of parameters declared `instant`-kind. Only these render
    /// as dates; `duration`-kind and all other kinds stay numeric.
    pub(crate) instant_params: std::collections::HashSet<String>,
}

impl CalendarContext {
    /// Compute the calendar date for a rendered parameter, or `None`
    /// when it is not an `instant` estimand or no `origin` is set.
    /// Single source of truth for all four formatters so the date map
    /// is applied identically (text / json / md / latex).
    pub(crate) fn date_for(&self, param: &str, value: f64) -> Option<String> {
        let origin = self.origin.as_deref()?;
        if !self.instant_params.contains(param) {
            return None;
        }
        // `internal_to_date` only errors on a malformed origin or an
        // unknown time_unit; both would have failed earlier loads, so a
        // None here is a safe numeric-only fallback rather than a crash.
        ir::caltime::internal_to_date(origin, value, &self.time_unit).ok()
    }

    /// gh#103 (H17): warn when the model declares `instant`-kind
    /// (calendar-date) parameters but no `origin`. Without an origin the
    /// internal-time ↔ date map is undefined, so these estimands render
    /// numeric-only and the intended calendar dates silently never
    /// appear. Returns `Some(msg)` for the caller to print to stderr;
    /// `None` when there is nothing to warn about (no instant params, or
    /// an origin is set so dates render).
    pub(crate) fn missing_origin_warning(&self) -> Option<String> {
        if self.origin.is_some() || self.instant_params.is_empty() {
            return None;
        }
        let mut names: Vec<&str> =
            self.instant_params.iter().map(|s| s.as_str()).collect();
        names.sort_unstable();
        Some(format!(
            "instant-kind parameter(s) {} render as calendar dates, but \
             the model declares no `origin` — they will be shown as raw \
             internal-time numbers instead.\n  \
             Add an `origin` to the model (e.g. `origin: 2020-01-01`) to \
             map internal time to dates.",
            names.join(", ")))
    }
}


/// Read a params TOML, returning a flat name→value map. Skips any
/// `[provenance]` section per `util::load_params_toml`'s convention.
pub(crate) fn read_param_values(path: &str) -> Option<std::collections::HashMap<String, f64>> {
    crate::util::load_params_toml(path).ok()
}

/// Two parameter dictionaries agree iff every shared key has values
/// matching to floating-point tolerance. Disjoint keys are treated as
/// "match" — `final_params.toml` and `mle_params.toml` legitimately
/// have non-overlapping fields (e.g. mle_params has more fixed params
/// rolled in). The shared subset is what would diverge under #16.
pub(crate) fn params_agree(
    a: &std::collections::HashMap<String, f64>,
    b: &std::collections::HashMap<String, f64>,
) -> bool {
    for (k, v) in a {
        if let Some(other) = b.get(k) {
            let scale = v.abs().max(other.abs()).max(1.0);
            if (v - other).abs() > 1e-9 * scale {
                return false;
            }
        }
    }
    true
}


/// Per-method report. Method-aware: IF2 methods populate the
/// `gate` / `parameters` / `chains` / `provenance` / `heuristic` blocks;
/// PGAS / PMMH methods set them all to `None` and lean on the typed
/// `method_result` payload, which carries posterior summaries / R̂ /
/// ESS for Bayesian methods.
#[derive(Debug, Clone, Serialize)]
pub struct MethodReport {
    pub name: String,
    /// Inference method: `"if2"`, `"pgas"`, or `"pmmh"`.
    pub method: String,
    pub n_chains: usize,
    /// IF2: best loglik-eval result. PMMH: `map_loglik`. PGAS: `None`
    /// (no point estimate).
    pub best_loglik: Option<f64>,
    /// The class of `best_loglik` (gh#280): `complete_data` for PGAS's
    /// joint value, a marginal kind otherwise. Derived from the typed
    /// `method_result`; `None` only when that failed to load.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loglik_type: Option<crate::fit::loglik::LoglikType>,
    pub initial_loglik: Option<f64>,
    pub camdl_version: Option<String>,
    /// Compound IF2 gate. `None` for Bayesian methods (the gate
    /// doesn't apply).
    pub gate: Option<GateReport>,
    /// IF2 estimated parameters with Â. Empty for Bayesian methods
    /// (consult `method_result` for posterior_mean / ess).
    pub parameters: Vec<ParameterReport>,
    /// IF2 per-chain loglik-eval table. Empty for Bayesian.
    pub chains: Vec<ChainReport>,
    /// IF2 provenance cross-check (final_params ↔ mle_params, etc.).
    /// `None` for Bayesian.
    pub provenance: Option<ProvenanceReport>,
    /// Typed `MethodResult` payload — pattern-match for headline
    /// scalars. Always populated for methods that loaded successfully.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method_result: Option<MethodResult>,
    /// Advisory IF2 interpretation block. `None` for Bayesian
    /// methods.
    #[serde(rename = "_heuristic")]
    pub heuristic: Option<HeuristicReport>,
    /// Calendar dates for `instant`-kind estimands (param name → ISO
    /// date), present when the model declares an `origin` (2026-05-22
    /// calendar-time §6.7). Covers IF2 point estimates and Bayesian
    /// posterior means alike. Additive field — empty (and omitted from
    /// JSON) for origin-less models, so existing consumers are
    /// unaffected; the md / latex renderers read it to annotate the
    /// posterior table value cell.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub param_dates: BTreeMap<String, String>,
    /// The checks the run made before sampling, as it showed them.
    pub checks: Vec<Check>,
    /// Each chain's adaptive warm-up (PMMH, `mh`); empty when nothing adapted.
    pub warmup: Vec<ChainWarmup>,
    /// The method's findings, one record per finding, as `diagnostics.json`
    /// holds them.
    pub findings: Vec<sim::inference::diagnostic::Diagnostic>,
    /// The verdict, from R̂ alone (`report::verdict`), so a consumer reads it
    /// rather than recomputing it.
    pub verdict: Verdict,
    /// The run's opening block (`run.json` `inputs.header`); absent for a leaf
    /// written before it was recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub header: Option<crate::fit::report::RunHeader>,
    /// Where the leaf landed and the method's wall time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored: Option<crate::fit::report::Stored>,
    /// Entries of `diagnostics.json` that could not be read, one sentence
    /// each; absent when every entry read. A finding missing from `findings`
    /// is named here, never silently dropped.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub findings_unreadable: Vec<String>,
    /// What the summary's text reads from the leaf beside the typed result.
    /// Not serialized: the JSON carries the typed result and the findings,
    /// and this is the text view's own read-back.
    #[serde(skip)]
    pub chain_detail: Option<super::ChainDetail>,
}

/// What a method's run came to. A run that fails has no `MethodResult`, and
/// still has a report: the header and findings, which exist before sampling
/// ends, and the error that stopped it. The checks the run showed are not
/// carried: the run's end has shown them already, and nothing else renders a
/// failed run.
#[derive(Debug, Clone)]
pub enum MethodOutcome {
    Completed(MethodReport),
    /// Every chain failed to start, no chain reached a finite log-likelihood,
    /// or the method returned an error.
    Failed {
        header: super::RunHeader,
        findings: Vec<sim::inference::diagnostic::Diagnostic>,
        /// The error as the command reports it.
        error: String,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    pub max_a_hat: f64,
    pub max_a_param: Option<String>,
    pub a_thresh: f64,
    pub a_passes: bool,
    pub delta_db: Option<f64>,
    pub threshold_db: Option<f64>,
    pub sigma_max: Option<f64>,
    pub db_passes: Option<bool>,
    pub overall_pass: Option<bool>,
    /// Parameters whose Â is unavailable (not finite). Any entry makes the
    /// verdict incomplete, which is not a pass.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub a_unavailable: Vec<String>,
    /// `"resolved"` when read from `state.resolved_gate`; `"default_fallback"`
    /// when fit_state.toml predates Phase 3 and we substituted
    /// `GateConfig::default()`. Critical signal for downstream readers.
    pub threshold_source: String,
    pub resolved_gate: Option<GateConfig>,
    pub resolved_loglik_eval: Option<LoglikEvalConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParameterReport {
    pub name: String,
    pub estimate: f64,
    pub chain_agreement: Option<f64>,
    /// Declared `perturb_only_at_t0 = true` in `[estimate]` — an
    /// initial-state parameter, perturbed at t=0 only under IF2.
    pub perturb_only_at_t0: bool,
    /// Calendar date of `estimate` for `instant`-kind params when the
    /// model declares an `origin` (2026-05-22 calendar-time §6.7).
    /// Additive field — `None`/omitted for non-instant params and
    /// origin-less models, so existing JSON consumers are unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimate_date: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChainReport {
    /// The chain's number — the `chain_N/` directory it names (gh#990), read
    /// from `FitState::chain_eval_ids`, never from the row position (gh#981).
    pub chain_id: ChainIndex,
    pub clean_loglik: f64,
    pub clean_se: f64,
    pub is_winner: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceReport {
    pub final_params_matches_mle_params: Option<bool>,
    pub fit_state_winner_matches_final_params: Option<bool>,
    pub stale_camdl_version: Option<String>,
    /// Whether `final_params.toml` and `mle_params.toml` were there to
    /// compare, so the text can name the absent one.
    #[serde(skip)]
    pub final_params_present: bool,
    #[serde(skip)]
    pub mle_params_present: bool,
}

impl ProvenanceReport {
    pub(crate) fn any_failed(&self) -> bool {
        matches!(self.final_params_matches_mle_params, Some(false))
            || matches!(self.fit_state_winner_matches_final_params, Some(false))
    }
}

impl MethodReport {
    /// Helper for `--strict`: returns whether this method's provenance
    /// cross-check (when applicable) reported any failure. Bayesian
    /// methods always return false (no provenance check applies).
    pub(crate) fn provenance_failed(&self) -> bool {
        self.provenance
            .as_ref()
            .map(|p| p.any_failed())
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct HeuristicReport {
    pub overall_status: String,
    pub interpretation: Option<String>,
}


/// The one builder of a [`MethodReport`]: every format and both producers
/// (the end of `fit run`, from memory; `fit summary`'s loader, from the leaf)
/// go through it. Exhaustive over [`MethodResult`], so a new method does not
/// compile until it has a report.
pub(crate) fn build_method_report(facts: MethodFacts, cal: &CalendarContext) -> MethodReport {
    let dir = facts.method_dir.to_string_lossy().into_owned();
    let mut report = match &facts.result {
        MethodResult::If2(if2) => if2_method_report(&facts.name, &dir, if2, cal),
        MethodResult::Pgas(_) | MethodResult::Pmmh(_) => {
            bayesian_method_report(&facts.name, &facts.method, Some(facts.result.clone()), cal)
        }
        MethodResult::Nlopt(r) => point_method_report(
            &facts.name, &facts.method, r.n_chains, Some(r.best_loglik), &facts.result,
            r.theta_hat.iter().filter_map(|(k, v)| cal.date_for(k, *v).map(|d| (k.clone(), d))).collect(),
        ),
        MethodResult::Pfilter(r) => point_method_report(
            &facts.name, &facts.method, 0, Some(r.ll_mean), &facts.result, BTreeMap::new(),
        ),
    };
    report.verdict = crate::fit::report::verdict(&facts.result);
    report.checks = facts.checks;
    report.warmup = facts.warmup;
    report.findings = facts.findings;
    report.header = facts.header;
    report.stored = facts.stored;
    report.findings_unreadable = facts.findings_unreadable;
    report.chain_detail = facts.chain_detail;
    report
}

/// The report of a method with no IF2 gate and no posterior: NLopt's
/// optimum, pfilter's likelihood at fixed parameters. The typed result
/// carries the content.
fn point_method_report(
    name: &str,
    method: &str,
    n_chains: usize,
    best_loglik: Option<f64>,
    result: &MethodResult,
    param_dates: BTreeMap<String, String>,
) -> MethodReport {
    MethodReport {
        name: name.to_string(),
        method: method.to_string(),
        n_chains,
        best_loglik,
        loglik_type: Some(crate::fit::loglik::LoglikType::from(result)),
        initial_loglik: None,
        camdl_version: Some(version::VERSION_SHORT.to_string()),
        gate: None,
        parameters: Vec::new(),
        chains: Vec::new(),
        provenance: None,
        method_result: Some(result.clone()),
        heuristic: None,
        param_dates,
        checks: Vec::new(),
        warmup: Vec::new(),
        findings: Vec::new(),
        verdict: Verdict::NotApplicable,
        header: None,
        stored: None,
        findings_unreadable: Vec::new(),
        chain_detail: None,
    }
}


/// Bayesian method report: only `method_result` carries content.
/// IF2-specific subfields default to None / empty.
pub(crate) fn bayesian_method_report(
    name: &str,
    method: &str,
    method_result: Option<MethodResult>,
    cal: &CalendarContext,
) -> MethodReport {
    let (n_chains, best_loglik) = match &method_result {
        Some(MethodResult::Pgas(r)) => (r.diagnostics.n_chains, None),
        Some(MethodResult::Pmmh(r)) => (r.diagnostics.n_chains, Some(r.map_loglik)),
        _ => (0, None),
    };
    // Date-annotate instant-kind posterior means (keyed off the
    // posterior mean, the value the md/latex tables render).
    let dates_from = |pm: &BTreeMap<String, f64>| -> BTreeMap<String, String> {
        pm.iter()
            .filter_map(|(name, mean)| cal.date_for(name, *mean).map(|d| (name.clone(), d)))
            .collect()
    };
    let param_dates: BTreeMap<String, String> = match &method_result {
        Some(MethodResult::Pgas(r)) => dates_from(&r.posterior_mean),
        Some(MethodResult::Pmmh(r)) => dates_from(&r.posterior_mean),
        _ => BTreeMap::new(),
    };
    let loglik_type = method_result.as_ref().map(crate::fit::loglik::LoglikType::from);
    MethodReport {
        name: name.to_string(),
        method: method.to_string(),
        n_chains,
        best_loglik,
        loglik_type,
        initial_loglik: None,
        camdl_version: Some(version::VERSION_SHORT.to_string()),
        gate: None,
        parameters: Vec::new(),
        chains: Vec::new(),
        provenance: None,
        method_result,
        heuristic: None,
        param_dates,
        checks: Vec::new(),
        warmup: Vec::new(),
        findings: Vec::new(),
        verdict: Verdict::NotApplicable,
        header: None,
        stored: None,
        findings_unreadable: Vec::new(),
        chain_detail: None,
    }
}

pub(crate) fn if2_method_report(
    method: &str,
    method_dir: &str,
    if2: &If2Run,
    cal: &CalendarContext,
) -> MethodReport {
    let state = &if2.record;
    let method_result = Some(MethodResult::If2(if2.clone()));
    // Gate analysis — the measurement `render::if2_gate_section` draws,
    // as structured data.
    let (gate_cfg, threshold_source) = match &state.resolved_gate {
        Some(g) => (g.clone(), "resolved".to_string()),
        None    => (GateConfig::default(), "default_fallback".to_string()),
    };
    // The one IF2 measurement the text block, `fit table` and the source
    // check also read (`gating::assess_if2`).
    let assessed = crate::fit::gating::assess_if2(state, &gate_cfg);
    let max_a = assessed.worst().map(|(_, a)| a).unwrap_or(0.0);
    let max_a_param = assessed.worst().map(|(n, _)| n);
    let a_passes = assessed.a_passes();
    let delta_db = assessed.spread.map(|leg| leg.delta_db);
    let threshold_db = assessed.spread.map(|leg| leg.threshold_db);
    let sigma_max = assessed.spread.map(|leg| leg.sigma_max);
    let db_passes = assessed.spread.map(|leg| leg.passes());
    // `None` when nothing was assessed (one chain): neither a pass nor a fail.
    let overall_pass = (!assessed.is_not_assessed()).then(|| assessed.passes());
    let a_unavailable = assessed.unavailable.clone();

    let gate = GateReport {
        max_a_hat: max_a,
        max_a_param,
        a_thresh: gate_cfg.a_thresh,
        a_passes,
        delta_db, threshold_db, sigma_max, db_passes, overall_pass, a_unavailable,
        threshold_source,
        resolved_gate: state.resolved_gate.clone(),
        resolved_loglik_eval: state.resolved_loglik_eval.clone(),
    };

    let parameters = if2_parameters(state, Some(std::path::Path::new(method_dir)), cal);
    let param_dates: BTreeMap<String, String> = parameters
        .iter()
        .filter_map(|p| p.estimate_date.clone().map(|d| (p.name.clone(), d)))
        .collect();

    // Chains. Labelled by `chain_eval_ids`, as the text table is: IF2 drops
    // PF-degenerate chains before the loglik-eval, so a row's position is not
    // a chain once any chain is missing. Without the ids (a fit_state written
    // before they existed) the table is omitted rather than mislabelled
    // (gh#981).
    let n = state.chain_eval_logliks.len();
    let best_idx = state.chain_eval_logliks.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    let chains: Vec<ChainReport> = if state.chain_eval_ids.len() == n {
        (0..n).map(|i| ChainReport {
            chain_id: state.chain_eval_ids[i],
            clean_loglik: state.chain_eval_logliks[i],
            clean_se: state.chain_eval_ses.get(i).copied().unwrap_or(f64::NAN),
            is_winner: Some(i) == best_idx,
        }).collect()
    } else {
        Vec::new()
    };

    let provenance = if2_provenance(state, method_dir);

    let overall_status = match overall_pass {
        Some(true)  => "pass".to_string(),
        Some(false) => "fail".to_string(),
        None        => "indeterminate".to_string(),
    };
    let interpretation = if !a_passes && db_passes == Some(false) {
        Some("chains disagree on basin (Â and decibans-spread both fail)".to_string())
    } else if !a_passes {
        Some("per-parameter chain agreement insufficient".to_string())
    } else if db_passes == Some(false) {
        Some("chains agree per-parameter but disagree on basin quality".to_string())
    } else {
        None
    };

    MethodReport {
        name: method.to_string(),
        method: "if2".into(),
        n_chains: state.n_chains,
        best_loglik: Some(if2.best_loglik),
        // Derive from the typed result; fall back to the method's recorded
        // kind so a legacy run with no loadable `method_result` still
        // labels its IF2 marginal rather than reading `unknown`.
        loglik_type: method_result
            .as_ref()
            .map(crate::fit::loglik::LoglikType::from)
            .or(state.loglik_type),
        initial_loglik: if state.initial_loglik.is_finite() {
            Some(state.initial_loglik)
        } else { None },
        camdl_version: state.camdl_version.clone(),
        gate: Some(gate),
        parameters,
        chains,
        provenance: Some(provenance),
        method_result,
        heuristic: Some(HeuristicReport { overall_status, interpretation }),
        param_dates,
        checks: Vec::new(),
        warmup: Vec::new(),
        findings: Vec::new(),
        verdict: Verdict::NotApplicable,
        header: None,
        stored: None,
        findings_unreadable: Vec::new(),
        chain_detail: None,
    }
}

/// The estimated parameters at the selected chain's θ̂, by name, with their Â
/// and calendar dates. Fixed parameters are left out: the estimated set is
/// Â's keys, else the fit's record of it, else every recorded value.
pub(crate) fn if2_parameters(
    state: &If2Record,
    method_dir: Option<&std::path::Path>,
    cal: &CalendarContext,
) -> Vec<ParameterReport> {
    let t0_set: std::collections::HashSet<&str> =
        state.perturb_only_at_t0_params.iter().map(|s| s.as_str()).collect();
    let mut keys: Vec<&String> = state.start_values.keys().collect();
    keys.sort();
    let estimated = crate::fit::method_result::estimated_names(method_dir, &state.tail_chain_agreement);
    keys.into_iter()
        .filter(|k| estimated.as_ref().is_none_or(|set| set.contains(k.as_str())))
        .map(|k| {
            let estimate = state.start_values[k];
            ParameterReport {
                name: k.clone(),
                estimate,
                chain_agreement: state.tail_chain_agreement.get(k).copied(),
                perturb_only_at_t0: t0_set.contains(k.as_str()),
                estimate_date: cal.date_for(k, estimate),
            }
        })
        .collect()
}

/// The IF2 provenance cross-check: `final_params.toml` against
/// `mle_params.toml`, and the recorded winner against `final_params.toml`.
pub(crate) fn if2_provenance(state: &If2Record, method_dir: &str) -> ProvenanceReport {
    let final_params = read_param_values(&format!("{}/final_params.toml", method_dir));
    let mle_params = read_param_values(&format!("{}/mle_params.toml", method_dir));
    let final_matches_mle = match (&final_params, &mle_params) {
        (Some(f), Some(m)) => Some(params_agree(f, m)),
        _ => None,
    };
    let state_matches_final = match &final_params {
        Some(f) if !state.start_values.is_empty() => {
            let mut ok = true;
            for (k, fv) in f {
                if let Some(sv) = state.start_values.get(k) {
                    if (sv - fv).abs() > 1e-9 * fv.abs().max(1.0) {
                        ok = false;
                        break;
                    }
                }
            }
            Some(ok)
        }
        _ => None,
    };
    let stale = match &state.camdl_version {
        Some(v) if v != version::VERSION_SHORT => Some(v.clone()),
        _ => None,
    };
    ProvenanceReport {
        final_params_matches_mle_params: final_matches_mle,
        fit_state_winner_matches_final_params: state_matches_final,
        stale_camdl_version: stale,
        final_params_present: final_params.is_some(),
        mle_params_present: mle_params.is_some(),
    }
}
