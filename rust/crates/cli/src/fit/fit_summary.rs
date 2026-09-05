//! `camdl fit summary` — single-fit interpretation surface.
//!
//! Reads a fit dir produced by `camdl fit run` and renders an
//! interpretation block per stage. The block content is method-aware:
//! IF2 stages render the compound-gate verdict, Â-keyed parameter
//! table, per-chain loglik-eval table, and filter-health provenance
//! cross-check; PGAS / PMMH stages render the posterior summary
//! (mean, R̂, ESS, acceptance) instead. Stage names come from the
//! walker (`fit_tree::walk_fit_dir`) — there is no hard-coded stage
//! list.
//!
//! Boundary rule: `status` answers "what's the state of my filesystem
//! / pipeline?", `summary` answers "what does this fit say?",
//! `compare` answers "which of these models predicts better?". Three
//! commands, three orthogonal jobs.
//!
//! See `docs/dev/proposals/2026-04-28-fit-experiment-management.md` §8
//! for the step-5 walker-consumer refactor; this file's prior shape
//! is preserved in git history at commit `d45c932` for context.

use crate::args::{FitSummaryArgs, FitSummaryFormat};
use crate::chain_selection::{warn_active_selection, ChainSelection, SubsetInfo};
use crate::evidence::NATS_TO_DB;
use crate::fit::config_diff::ConfigDiff;
use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};
use crate::fit::gating::AgreementBand;
use crate::fit::fit_tree::{self, DataKind};
use crate::fit::fit_view::FitView;
use crate::fit::method_result::{
    If2StageResult, MaxRhat, MethodResult, MinEss, NutsStageResult, PgasStageResult,
    PmmhStageResult, PosteriorDiagnostics, RhatBand, RHAT_CONVERGED_THRESHOLD, Stat,
};
use crate::fit::state::FitState;
use crate::fit::table_row::{self, TableRow};
use crate::version;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Versioned JSON schema. Bumped when fields are renamed / removed /
/// retyped; field additions are non-breaking and keep version stable.
const SCHEMA_VERSION: u32 = 1;

/// Recompute a Bayesian stage's diagnostics + posterior means over a
/// chain-filtered draws cloud, mutating `diag`/`posterior_mean` in place, and
/// return the [`SubsetInfo`] provenance.
///
/// The subset R̂ / ESS / mean are computed from the SAME per-chain sequences and
/// the SAME [`compute_rhat_ess`](crate::fit::runner::compute_rhat_ess) the fit
/// used at completion (the shared seam), applied to the retained chains only —
/// so `fit summary --exclude-chains` answers "what would the diagnostics be
/// without these chains?" with the fit's own estimator, not a parallel one.
fn recompute_over_subset(
    diag: &mut crate::fit::method_result::PosteriorDiagnostics,
    posterior_mean: &mut BTreeMap<String, f64>,
    stage_dir: &Path,
    selection: &ChainSelection,
) -> Result<SubsetInfo, String> {
    // Recompute R̂ / ESS over the retained chains for the estimated params — the
    // keys of `posterior_mean`, the exact set the renderer iterates, so the table
    // shape is unchanged. Routed through the one shared recompute
    // (`chain_selection::recompute_subset_diagnostics`) that `fit predict` also
    // calls, so summary and predict cannot disagree on the same fit + selection.
    let param_names: Vec<String> = posterior_mean.keys().cloned().collect();
    let sub = crate::chain_selection::recompute_subset_diagnostics(
        &stage_dir.join("draws.tsv"),
        selection,
        &param_names,
    )?;

    // Posterior means over the retained rows — summary-specific, so computed
    // here from the shared recompute's `kept` rows rather than in the shared fn.
    let mut new_mean = BTreeMap::new();
    for p in &param_names {
        let vals: Vec<f64> = sub.kept.iter().filter_map(|r| r.params.get(p).copied()).collect();
        let mean = if vals.is_empty() {
            f64::NAN
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        new_mean.insert(p.clone(), mean);
    }

    diag.per_param = sub.per_param;
    diag.n_samples = sub.n_samples;
    diag.n_chains = sub.n_chains;
    // `thin` and `wall_time_secs` are properties of the whole run, unchanged by
    // a read-side subset (ESS/iter and ESS/sec are then reported over the
    // subset's ESS against the same iteration / wall-clock denominators).
    *posterior_mean = new_mean;
    Ok(sub.info)
}

/// Mutate a Bayesian stage's typed result to the chain-subset diagnostics.
/// Errors (never silently no-ops) for a non-Bayesian stage — the caller only
/// invokes it on Bayesian stages.
fn apply_selection_to_typed(
    typed: &mut MethodResult,
    stage_dir: &Path,
    selection: &ChainSelection,
) -> Result<SubsetInfo, String> {
    match typed {
        MethodResult::Pgas(r) => {
            recompute_over_subset(&mut r.diagnostics, &mut r.posterior_mean, stage_dir, selection)
        }
        MethodResult::Pmmh(r) => {
            recompute_over_subset(&mut r.diagnostics, &mut r.posterior_mean, stage_dir, selection)
        }
        MethodResult::Nuts(r) => {
            recompute_over_subset(&mut r.diagnostics, &mut r.posterior_mean, stage_dir, selection)
        }
        _ => Err(
            "--exclude-chains applies only to Bayesian stages (PGAS / PMMH / NUTS)".to_string(),
        ),
    }
}

/// Print the read-side chain-selection advisory to STDERR (so it never pollutes
/// the stdout summary): the identifiability nudge FIRST (when the per-chain
/// outlier signal is strong — gh#406), then the loud, non-quietable exclusion
/// warning. The flag is the second thing the user reads, not the first.
fn chain_selection_advisory(
    stage_dir: &Path,
    info: &SubsetInfo,
    kind: super::loglik::LoglikType,
) {
    use super::chain_diagnostics as cd;
    if let Some(means) = cd::read_chain_mean_logliks(stage_dir, kind) {
        let scores = cd::chain_loglik_mod_zscores(&means.scored);
        let outliers = cd::outlier_labels(&scores);
        if !outliers.is_empty() {
            eprintln!(
                "\x1b[33mnote:\x1b[0m {} disagree strongly with the others on `{}` — before \
                 excluding, ask whether a parameter is unidentified (a flat likelihood ridge). \
                 The per-chain table below names them; the primary fix is the model, not the flag.",
                outliers.join(", "), means.scored_column
            );
        }
    }
    warn_active_selection(info);
}

/// Top-level entry point. Resolves `args.fit` (the fit handle) to its segment
/// directory, walks every completed fit-stage run, and dispatches to the right
/// formatter based on `--format` and `--params-only`. Exits with code 1 if the
/// handle does not resolve or the segment is empty; with code 1 in `--strict`
/// mode if any IF2 stage's provenance cross-check fails.
pub fn cmd_fit_summary(args: &FitSummaryArgs) {
    // Resolve the fit handle (@label / hash prefix / run-dir / fit.toml) → its
    // segment directory. summary operates on the directory; it needs no config.
    let segment = match crate::fit::handle::resolve_fit_segment(&args.fit) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    let dir = segment.to_string_lossy().into_owned();

    let strict = args.strict || ci_env_set();

    // The full set of completed stages, walker-discovered, in fit.toml
    // declaration order (see `discover_stages`). Validate `--stage`
    // against this set rather than a hard-coded constant.
    let discovered = discover_stages(Path::new(&dir));

    // gh#103 (H17): warn (once, to stderr) when the fit has instant-kind
    // calendar-date parameters but the model declares no `origin` — the
    // dates can't be rendered and the omission is otherwise silent.
    // stderr keeps the warning out of the JSON/stdout payload.
    if let Some(msg) = load_calendar_context(Path::new(&dir)).missing_origin_warning() {
        eprintln!("\x1b[33mwarning:\x1b[0m {}", msg);
    }

    if args.params_only {
        match dump_params_only(&dir, args.stage.as_deref(), &discovered) {
            Ok(s) => {
                print!("{}", s);
                return;
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }

    let selected: Vec<ResolvedStage> = match &args.stage {
        Some(name) => {
            let valid: Vec<&str> = discovered.iter().map(|r| r.stage.as_str()).collect();
            if !valid.contains(&name.as_str()) {
                eprintln!(
                    "error: unknown stage `{}`. Available: {}",
                    name,
                    if valid.is_empty() { "(none)".to_string() } else { valid.join(", ") }
                );
                std::process::exit(1);
            }
            discovered.iter().filter(|r| r.stage == *name).cloned().collect()
        }
        None => discovered.clone(),
    };

    // Parse `--exclude-chains` at the boundary into a typed selection. A
    // selection is meaningless without a Bayesian stage to subset — refuse it
    // rather than silently no-op (an optimizer-only fit has no chains-as-draws).
    let selection: Option<ChainSelection> = match args.exclude_chains.as_deref() {
        Some(raw) => match ChainSelection::parse_exclude(raw) {
            Ok(sel) => {
                let has_bayesian = selected.iter().any(|s| {
                    matches!(s.method.as_str(), "pgas" | "pmmh" | "nuts" | "mh")
                });
                if !has_bayesian {
                    eprintln!(
                        "error: --exclude-chains needs a Bayesian stage (PGAS / PMMH / NUTS) to \
                         subset; this fit's selected stage(s) have no posterior chains"
                    );
                    std::process::exit(1);
                }
                Some(sel)
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
        None => None,
    };

    match args.format {
        FitSummaryFormat::Text => format_text(&dir, args, &selected, strict, selection.as_ref()),
        FitSummaryFormat::Json => format_json(&dir, &selected, strict, selection.as_ref()),
        FitSummaryFormat::Md => format_md(&dir, &selected, strict, selection.as_ref()),
        FitSummaryFormat::Latex => format_latex(&dir, &selected, strict, selection.as_ref()),
    }
}

/// Calendar context for date-rendering `instant`-kind estimands
/// (2026-05-22 calendar-time §6.7). Sourced once per `fit summary`
/// invocation from the fit dir's model (path via `FitView.model`),
/// then shared by every formatter. When the model declares no
/// `origin`, or cannot be loaded (moved path, parse error), rendering
/// degrades to numeric-only — `date_for` returns `None` and no
/// formatter changes shape.
#[derive(Debug, Clone, Default)]
struct CalendarContext {
    /// `origin` date string from the model (`Some` iff declared).
    origin: Option<String>,
    /// Model `time_unit` (`days`/`weeks`/`months`/`years`); the divisor
    /// for the internal-time ↔ date map.
    time_unit: String,
    /// Names of parameters declared `instant`-kind. Only these render
    /// as dates; `duration`-kind and all other kinds stay numeric.
    instant_params: std::collections::HashSet<String>,
}

impl CalendarContext {
    /// Compute the calendar date for a rendered parameter, or `None`
    /// when it is not an `instant` estimand or no `origin` is set.
    /// Single source of truth for all four formatters so the date map
    /// is applied identically (text / json / md / latex).
    fn date_for(&self, param: &str, value: f64) -> Option<String> {
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
    fn missing_origin_warning(&self) -> Option<String> {
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

/// Load the calendar context from a fit dir: read the model path from the
/// fit-level view (`FitView.model`, sidecar-sourced), falling back to any stage
/// leaf's `provenance.source_paths`; load that model and extract `origin`,
/// `time_unit`, and the `instant`-kind parameter set. Any failure (no fit view,
/// model moved/unparseable) degrades to an empty context (numeric-only
/// rendering) — never panics.
fn load_calendar_context(fit_dir: &Path) -> CalendarContext {
    // Legacy fits carried the model path on the top-level `Fit` run.json. A
    // content-addressed fit (gh#147 M3.2) has no fit-wide record — the fit
    // level is a path segment — so recover the model path from any stage
    // leaf's `run.json` `provenance.source_paths`.
    let model_path = FitView::read(fit_dir)
        .map(|v| v.model)
        .filter(|m| !m.is_empty())
        .or_else(|| {
            crate::cas_read::walk_records(fit_dir)
                .into_iter()
                .find_map(|(_, rec)| rec.provenance.source_paths.first().cloned())
        });
    let Some(model_path) = model_path else {
        return CalendarContext::default();
    };
    let model = match crate::util::load_model(&model_path) {
        Ok((m, _)) => m,
        Err(_) => return CalendarContext::default(),
    };
    let instant_params = model
        .parameters
        .iter()
        .filter(|p| p.param_kind == Some(ir::parameter::ParamKind::Instant))
        .map(|p| p.name.clone())
        .collect();
    CalendarContext {
        origin: model.origin.clone(),
        time_unit: model.time_unit.clone(),
        instant_params,
    }
}

/// The fit's model `#'` documentation dictionary — the model's own header block
/// and its per-parameter docs, for the two legends at the top of the summary.
/// Uses the same model-path recovery as [`load_calendar_context`]. Empty when
/// the model can't be located or documents nothing, so an undocumented fit
/// prints no legend at all.
///
/// One loader for both, because they come from one compile: asking twice would
/// shell out to `camdlc` twice and let the two halves disagree about which
/// model they describe.
fn load_model_docs_for_fit(fit_dir: &Path) -> ir::ModelDocs {
    let model_path = FitView::read(fit_dir)
        .map(|v| v.model)
        .filter(|m| !m.is_empty())
        .or_else(|| {
            crate::cas_read::walk_records(fit_dir)
                .into_iter()
                .find_map(|(_, rec)| rec.provenance.source_paths.first().cloned())
        });
    let Some(model_path) = model_path else { return ir::ModelDocs::default() };
    // The envelope dictionary keys by base parameter name (`R0`, not
    // `R0_urban`), so a stratified family shows once — `BTreeMap` order is
    // deterministic.
    crate::util::load_model_docs(&model_path).unwrap_or_default()
}

/// One resolved fit-stage to render: stage name, on-disk directory,
/// and method (so consumers can pattern-match on `MethodResult` once
/// the typed payload is loaded).
#[derive(Debug, Clone)]
struct ResolvedStage {
    stage: String,
    method: String,
    stage_dir: PathBuf,
}

/// Walk the fit_dir and return one `ResolvedStage` per completed
/// stage. Order matches `FitView.stages_declared` (the execution order recovered
/// from the leaves' ordinal-prefixed stage labels); stages that didn't complete are
/// dropped; stages that completed but aren't in `stages_declared`
/// (shouldn't happen in v2 layouts, but the walker is permissive)
/// are appended at the end in walker order so they're still visible.
///
/// When the same stage name appears in multiple cells (synthetic
/// replicates, sweep cells), prefers Real over Synthetic, lowest
/// `fit_seed`, then lex-first stage_dir — same priority as
/// [`crate::fit::table_row::build_row`]'s terminal-stage picker.
fn discover_stages(fit_dir: &Path) -> Vec<ResolvedStage> {
    // Read the fit-level view for `stages_declared`; if missing, fall back to
    // walker-order (covers user-built non-canonical fits).
    let stages_declared: Vec<String> = FitView::read(fit_dir)
        .map(|v| v.stages_declared)
        .unwrap_or_default();
    let nodes = fit_tree::walk_fit_dir(fit_dir).unwrap_or_default();

    // Best (stage_name, method, dir) per stage name, by (data_kind,
    // fit_seed, stage_dir) priority.
    type Rank = (u8, u64, PathBuf);
    let mut best: BTreeMap<String, (String, Rank, PathBuf)> = BTreeMap::new();
    for node in &nodes {
        let (stage, method) = (node.stage.stage.clone(), node.stage.method.as_str().to_string());
        let rank: Rank = match &node.stage.axes {
            Some(axes) => {
                let kind_rank = match axes.data_kind {
                    DataKind::Real => 0,
                    DataKind::Synthetic { .. } => 1,
                };
                (kind_rank, axes.fit_seed, node.stage_dir.clone())
            }
            None => (2, u64::MAX, node.stage_dir.clone()),
        };
        best.entry(stage)
            .and_modify(|(meth, cur, dir)| {
                if rank < *cur {
                    *meth = method.clone();
                    *cur = rank.clone();
                    *dir = node.stage_dir.clone();
                }
            })
            .or_insert_with(|| (method.clone(), rank, node.stage_dir.clone()));
    }

    let mut out: Vec<ResolvedStage> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for name in &stages_declared {
        if let Some((method, _, dir)) = best.get(name) {
            out.push(ResolvedStage {
                stage: name.clone(),
                method: method.clone(),
                stage_dir: dir.clone(),
            });
            seen.insert(name.clone());
        }
    }
    for (name, (method, _, dir)) in best {
        if !seen.contains(&name) {
            out.push(ResolvedStage {
                stage: name,
                method,
                stage_dir: dir,
            });
        }
    }
    out
}

fn format_text(
    dir: &str,
    args: &FitSummaryArgs,
    stages: &[ResolvedStage],
    strict: bool,
    selection: Option<&ChainSelection>,
) {
    let use_color = should_use_color(args.no_color);
    let cal = load_calendar_context(Path::new(dir));
    let fmt = Formatter { use_color, cal };
    let mut had_provenance_failure = false;
    // Emit the loud selection advisory (nudge + warning) once, from the first
    // Bayesian stage that recomputes.
    let mut warned = false;

    print!("{}", fmt.fit_header(dir));

    let docs = load_model_docs_for_fit(Path::new(dir));

    // What this model IS, from its file-header `#'` block (gh#750) — so the
    // first question a summary raises ("a fit of what?") is answered without
    // opening the `.camdl`. One source line per output line, so a
    // `@base`/`@adds`/`@changes` lineage header stays readable.
    if let Some(d) = &docs.model {
        if let Some(t) = &d.text {
            println!();
            for line in t.lines() {
                println!("  {}", line);
            }
        }
        if let Some(r) = &d.reference {
            println!("  [{}]", r);
        }
    }

    // Parameter legend from the model's `#'` docs (symbol — description [ref]).
    // Shown only when the model documents at least one parameter, so it adds no
    // noise to undocumented fits.
    let documented: Vec<(String, ir::parameter::DocBlock)> =
        docs.parameters.into_iter().collect();
    if !documented.is_empty() {
        println!("\n  parameters");
        for (name, d) in &documented {
            let mut line = String::from("    ");
            if let Some(s) = &d.symbol {
                line.push_str(s);
                line.push_str("  ");
            }
            line.push_str(name);
            if let Some(t) = &d.text {
                line.push_str("  —  ");
                line.push_str(t);
            }
            if let Some(r) = &d.reference {
                line.push_str("  [");
                line.push_str(r);
                line.push(']');
            }
            println!("{}", line);
        }
    }

    let mut prev_loglik: Option<f64> = None;
    let mut prev_stage_name: Option<String> = None;
    for resolved in stages {
        let stage_dir_str = resolved.stage_dir.to_string_lossy().into_owned();
        let mut typed = match MethodResult::load_from(&resolved.stage_dir, &resolved.method) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "warning: cannot load {} ({}): {}",
                    stage_dir_str, resolved.method, e
                );
                continue;
            }
        };

        // The stage's log-likelihood class, derived once from the typed result
        // (the single authority). It selects the per-chain trace column the
        // outlier diagnostics may compare across chains (gh#667).
        let ll_kind = crate::fit::loglik::LoglikType::from(&typed);

        // Chain selection: recompute this Bayesian stage's diagnostics over the
        // retained chains before rendering (an IF2 / NLopt stage has no chains
        // and is left untouched). The advisory prints once.
        let mut subset_info: Option<SubsetInfo> = None;
        if let Some(sel) = selection {
            if matches!(
                typed,
                MethodResult::Pgas(_) | MethodResult::Pmmh(_) | MethodResult::Nuts(_)
            ) {
                match apply_selection_to_typed(&mut typed, &resolved.stage_dir, sel) {
                    Ok(info) => {
                        if !warned {
                            chain_selection_advisory(&resolved.stage_dir, &info, ll_kind);
                            warned = true;
                        }
                        subset_info = Some(info);
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        std::process::exit(1);
                    }
                }
            }
        }

        match &typed {
            MethodResult::If2(if2) => {
                // IF2 keeps the rich FitState rendering — gate, params
                // table, per-chain loglik-eval, provenance — because it
                // surfaces information the typed `If2StageResult`
                // doesn't (e.g. perturb_only_at_t0 markers, per-chain SE, raw
                // start_values). The typed payload is the source of
                // truth for headline scalars; FitState is the source
                // for the rendered tables.
                let state = match FitState::load(&stage_dir_str) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!(
                            "warning: cannot load {}/fit_state.toml: {}",
                            stage_dir_str, e
                        );
                        continue;
                    }
                };
                let block = fmt.stage_block(
                    &resolved.stage,
                    &stage_dir_str,
                    &state,
                    Some(if2),
                    prev_loglik,
                    prev_stage_name.as_deref(),
                );
                if block.provenance_failed {
                    had_provenance_failure = true;
                }
                print!("{}", block.text);
                prev_loglik = Some(state.best_loglik);
                prev_stage_name = Some(resolved.stage.clone());
            }
            MethodResult::Pgas(pgas) => {
                print!(
                    "{}",
                    fmt.bayesian_block(&resolved.stage, "pgas", &resolved.stage_dir, BayesianView::Pgas(pgas), subset_info.as_ref(), ll_kind)
                );
                prev_stage_name = Some(resolved.stage.clone());
                // Bayesian rows have no scalar best_loglik to chain
                // through; `prev_loglik` stays where it was.
            }
            MethodResult::Pmmh(pmmh) => {
                print!(
                    "{}",
                    fmt.bayesian_block(&resolved.stage, "pmmh", &resolved.stage_dir, BayesianView::Pmmh(pmmh), subset_info.as_ref(), ll_kind)
                );
                prev_loglik = Some(pmmh.map_loglik);
                prev_stage_name = Some(resolved.stage.clone());
            }
            MethodResult::Nuts(nuts) => {
                print!(
                    "{}",
                    fmt.bayesian_block(&resolved.stage, "nuts", &resolved.stage_dir, BayesianView::Nuts(nuts), subset_info.as_ref(), ll_kind)
                );
                prev_loglik = Some(nuts.map_loglik);
                prev_stage_name = Some(resolved.stage.clone());
            }
            MethodResult::Nlopt(r) => {
                // NLopt stages are point-estimate (like IF2) but with no
                // FitState-rendered IF2 gate to display. Print a compact
                // headline + the theta_hat table from the typed payload.
                println!("\n  stage: {} (algorithm = {})", resolved.stage, r.algorithm);
                println!(
                    "    loglik:   {:.2} ({})     converged chains: {}/{}",
                    r.best_loglik,
                    crate::fit::loglik::LoglikType::from(&typed).tag(),
                    r.n_converged, r.n_chains
                );
                if r.n_chains > 1 {
                    println!(
                        "    chain-agreement: max rel range = {:.2}% bound",
                        r.max_rel_range * 100.0
                    );
                }
                println!("    θ̂ ({} estimated params):", r.theta_hat.len());
                let w = name_col_width(r.theta_hat.keys().map(String::as_str), 14);
                for (k, v) in &r.theta_hat {
                    let shown = fit_name(k, w);
                    match fmt.cal.date_for(k, *v) {
                        Some(date) => println!("      {:<w$} = {}  ({})", shown, v, date),
                        None => println!("      {:<w$} = {}", shown, v),
                    }
                }
                prev_loglik = Some(r.best_loglik);
                prev_stage_name = Some(resolved.stage.clone());
            }
        }
    }

    if stages.is_empty() {
        println!("  (no completed stages found in {})", dir);
    }

    // gh#322: the keyed-joint (θ, X) forkable count — how many posterior draws
    // pair with a saved smoothed trajectory (or are deterministic, for ODE),
    // i.e. how many a counterfactual `compare`/contrast could fork. Shown only
    // for a posterior fit; an optimizer fit has no cloud, so `resolve_joint`
    // errors and the line is skipped.
    //
    // The cloud is resolved under this command's own `--exclude-chains`, so the
    // count describes the same posterior the rest of the summary reports —
    // never the full cloud under a header that says chains were dropped
    // (gh#695). The retained-chain scope is named on the line itself, because
    // "24/24 forkable" reads identically whether it is a subset or the whole.
    let joint = crate::posterior_draws::resolve_posterior_draws(dir, args.stage.as_deref())
        .map(|p| p.with_selection(selection.cloned()))
        .and_then(|p| crate::fit::joint::resolve_joint(&p));
    if let Ok(j) = joint {
        println!();
        println!("  {}", fmt.bold("(θ, X) forkability"));
        let note = if j.n_forkable == j.n_total {
            fmt.ok("(all draws)")
        } else {
            fmt.dim("(partial — only path-saved draws can be conditioned-forked)")
        };
        let scope = match &j.selection {
            Some(info) => {
                format!(" over the retained chains (chain(s) {} excluded)", info.excluded_csv())
            }
            None => String::new(),
        };
        println!("    forkable draws: {}/{}{}  {}", j.n_forkable, j.n_total, scope, note);
    }

    if strict && had_provenance_failure {
        eprintln!();
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn format_json(dir: &str, stages: &[ResolvedStage], strict: bool, selection: Option<&ChainSelection>) {
    let doc = build_summary_doc(dir, stages, selection);
    let any_failed = doc.stages.iter().any(|s| s.provenance_failed());
    let s = serde_json::to_string_pretty(&doc).expect("FitSummaryDoc must serialize");
    println!("{}", s);
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn format_md(dir: &str, stages: &[ResolvedStage], strict: bool, selection: Option<&ChainSelection>) {
    let doc = build_summary_doc(dir, stages, selection);
    let any_failed = doc.stages.iter().any(|s| s.provenance_failed());
    print!("{}", render_markdown(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn format_latex(dir: &str, stages: &[ResolvedStage], strict: bool, selection: Option<&ChainSelection>) {
    let doc = build_summary_doc(dir, stages, selection);
    let any_failed = doc.stages.iter().any(|s| s.provenance_failed());
    print!("{}", render_latex(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

enum BayesianView<'a> {
    Pgas(&'a PgasStageResult),
    Pmmh(&'a PmmhStageResult),
    Nuts(&'a NutsStageResult),
}

// ── Formatting layer ────────────────────────────────────────────────

struct Formatter {
    use_color: bool,
    /// Calendar context for date-rendering `instant`-kind estimands.
    /// Empty (no origin / no instant params) → numeric-only.
    cal: CalendarContext,
}

struct StageBlock {
    text: String,
    provenance_failed: bool,
}

/// Significant figures the posterior-mean column carries. Four separates the
/// values a reader compares across rows (`0.001854` against `240.8`) without
/// implying a precision the Monte-Carlo error does not support.
const POSTERIOR_MEAN_SIG_FIGS: usize = 4;

/// Widest a parameter-name column is allowed to grow before names are
/// ellipsized instead. Stratified names are built by suffixing (`I0_ituri`,
/// `phi_split_haut_uele`) and a deeply stratified model can reach a width no
/// terminal helps with; past this point a readable grid is worth more than a
/// complete name.
const NAME_COL_MAX: usize = 44;

/// The width a name column must have for every name to fit, bounded by
/// [`NAME_COL_MAX`] and never narrower than `min` (which keeps a short-named
/// model's table looking as it always did).
///
/// Rust's `{:14}` is a MINIMUM width: it pads a short name and passes a long
/// one through whole, shoving every later column right. Sizing the column to
/// the names actually present is what keeps the grid a grid.
fn name_col_width<'a>(names: impl Iterator<Item = &'a str>, min: usize) -> usize {
    names
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .clamp(min, NAME_COL_MAX)
}

/// `name` cut to `width` characters, ellipsized in the MIDDLE.
///
/// The head gives way, not the tail: a stratified parameter's distinguishing
/// part is its suffix (`..._nord_kivu`), so a tail-truncated name would
/// collapse every stratum of one parameter onto the same unreadable row.
fn fit_name(name: &str, width: usize) -> String {
    let n = name.chars().count();
    if n <= width {
        return name.to_string();
    }
    // One char for the ellipsis; the remainder splits with the extra going to
    // the tail, which is the half that identifies the row.
    let keep = width.saturating_sub(1);
    let head = keep / 2;
    let tail = keep - head;
    let chars: Vec<char> = name.chars().collect();
    let mut out: String = chars[..head].iter().collect();
    out.push('\u{2026}');
    out.extend(&chars[n - tail..]);
    out
}

/// A number at `sig` significant figures, in the shortest form that keeps
/// them -- the `%g` rule.
///
/// Fixed decimals give every parameter the same ABSOLUTE precision, which is
/// the wrong invariant when one is a coupling rate near `0.002` and the next
/// is a seed size near `241`: six decimals is four wasted columns on one and
/// three meaningless digits on the other. Significant figures give each the
/// same RELATIVE precision, so the column can be scanned.
fn sig_figs(v: f64, sig: usize) -> String {
    debug_assert!(sig >= 1, "at least one significant figure");
    if v.is_nan() {
        return "NaN".to_string();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    // `%g`'s own switch. Below the window the fixed form is mostly leading
    // zeros; above it, the fixed form would print MORE significant digits than
    // the budget claims (`6322125` asserts seven), which is the failure this
    // function exists to avoid.
    if exp < -4 || exp >= sig as i32 {
        return format!("{:.*e}", sig - 1, v);
    }
    let decimals = (sig as i32 - 1 - exp).max(0) as usize;
    format!("{v:.decimals$}")
}

/// The max-R̂ cell for the export formats: the number, or a word saying why
/// there isn't one. Never `0.000` — that is a real R̂ value and must not double
/// as "could not be computed" (review blocker 1).
fn max_rhat_cell(diag: &PosteriorDiagnostics) -> String {
    match diag.max_rhat_status() {
        MaxRhat::Reported(v) => format!("{v:.3}"),
        MaxRhat::Unassessable { params } =>
            format!("not computable for {} parameter(s) — NOT converged", params.len()),
        MaxRhat::NotApplicable { reason } => format!("not assessed ({})", reason.describe()),
        MaxRhat::NoParams => "not assessed".to_string(),
    }
}

impl Formatter {
    fn fit_header(&self, dir: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!("\n{}/\n", self.bold(dir)));
        s.push_str(&format!("  camdl {}\n\n", self.dim(version::VERSION_SHORT)));
        s
    }

    fn stage_block(
        &self,
        stage: &str,
        stage_dir: &str,
        state: &FitState,
        typed: Option<&If2StageResult>,
        prev_loglik: Option<f64>,
        prev_stage_name: Option<&str>,
    ) -> StageBlock {
        let mut s = String::new();

        s.push_str(&format!("══ {} {}\n",
            self.bold(stage),
            "═".repeat(74_usize.saturating_sub(stage.len()))));

        // Headline — type tag *after* the number (gh#280), so a scraper
        // reading `loglik=<num>` stops at the first non-numeric char.
        s.push_str(&format!("  best loglik:  {:.1} ({})",
            state.best_loglik,
            crate::fit::loglik::LoglikType::tag_or_unknown(state.loglik_type)));
        if !state.chain_eval_logliks.is_empty() {
            // The headline loglik is the cross-chain max of the
            // re-scored per-chain values (loglik-eval — see
            // crates/cli/src/fit/loglik_eval.rs for the pipeline,
            // and the per-chain table below for breakdown).
            s.push_str("  (loglik-eval, max across chains)");
        }
        s.push('\n');
        s.push_str(&format!("  chains:       {}\n", state.n_chains));
        if let Some(ref v) = state.camdl_version {
            if v != version::VERSION_SHORT {
                s.push_str(&format!("                {}\n",
                    self.warn(&format!("⚠ stale: produced by {}, current is {}",
                        v, version::VERSION_SHORT))));
            }
        }
        if let Some(prev) = prev_loglik {
            let delta = state.best_loglik - prev;
            let prev_label = prev_stage_name.unwrap_or("prev");
            let glyph = if delta >= 0.0 { self.ok("✓") } else { self.err("✗") };
            s.push_str(&format!(
                "  vs {}:    Δ = {:+.1} nats  {}\n",
                prev_label, delta, glyph));
        }
        s.push('\n');

        // Compound-gate verdict
        s.push_str(&self.gate_verdict_block(state));

        // Per-parameter table
        s.push_str(&self.parameter_table(state));

        // Per-chain loglik-eval table
        if !state.chain_eval_logliks.is_empty() {
            s.push_str(&self.chain_loglik_eval_table(state));
        }

        // ESS-at-MLE — surfaced from the typed payload only, not from
        // FitState (FitState doesn't carry it; the loader extracts it
        // from chain_evaluations.tsv).
        if let Some(if2) = typed {
            if let Some(ess) = &if2.ess_at_mle {
                s.push_str(&format!("  {}\n", self.bold("ESS at θ̂")));
                s.push_str(&format!(
                    "    min  = {:>8.0}{}    mean = {:>8.0}\n",
                    ess.ess_min,
                    match ess.ess_min_step {
                        Some(step) => format!("  (at obs step {})", step),
                        None => String::new(),
                    },
                    ess.ess_mean,
                ));
                s.push('\n');
            }
        }

        // Richardson dt-convergence verdict (gh#52). Rendered when
        // populated (every IF2 stage post-§Proposal-1 where dt_check
        // is enabled); legacy fit_state.toml or `enabled = false`
        // skips the block.
        if let Some(dt_check) = &state.dt_check {
            s.push_str(&self.dt_check_block(dt_check));
        }

        // Provenance cross-check (#16 fixture, every read)
        let prov = self.provenance_block(stage_dir, state);
        let provenance_failed = prov.failed;
        s.push_str(&prov.text);

        s.push('\n');
        StageBlock { text: s, provenance_failed }
    }

    /// Render the gh#52 Richardson dt-convergence verdict line +
    /// optional ladder. Pass case is one line; fail/marginal includes
    /// the ladder rows for context.
    fn dt_check_block(&self, dt_check: &super::dt_check::DtCheckResult) -> String {
        use super::dt_check::DtCheckVerdict;
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("dt-convergence at θ̂ (Richardson)")));
        let label = match dt_check.verdict {
            DtCheckVerdict::Pass     => self.ok("PASS"),
            DtCheckVerdict::Marginal => self.warn("MARGINAL"),
            DtCheckVerdict::Fail     => self.err("FAIL"),
            DtCheckVerdict::Skipped  => "skipped".to_string(),
        };
        s.push_str(&format!("    verdict: {}    ({})\n", label, dt_check.notes));
        // Ladder rows for non-pass verdicts; the pass case's
        // numbers are already in `notes` and the ladder is noise.
        if !matches!(dt_check.verdict, DtCheckVerdict::Pass | DtCheckVerdict::Skipped) {
            s.push_str(&format!(
                "    threshold τ = {:.2} nats  (SE-aware floor 4·σ_max = {:.2})\n",
                dt_check.threshold_nats, dt_check.threshold_se_aware_nats));
            for (i, rung) in dt_check.ladder.iter().enumerate() {
                let tag = if i == 0 { "(fit)" } else { "" };
                s.push_str(&format!(
                    "    dt = {:>7.4}   ll = {:>9.2} ± {:.2}   {}\n",
                    rung.dt, rung.loglik, rung.se, tag));
            }
            if dt_check.pf_se_inflation {
                s.push_str(&format!(
                    "    {} PF-SE inflated as dt halved (auxiliary signal).\n",
                    self.warn("⚠")));
            }
            if matches!(dt_check.verdict, DtCheckVerdict::Fail) {
                s.push_str("    Note: synthetic recovery shares this dt and \
                    cannot detect dt bias by itself.\n");
            }
        }
        s.push('\n');
        s
    }

    /// The gate config to render an Â against. Priority:
    ///   1. `state.resolved_gate` (Phase 3 — the value actually used)
    ///   2. `GateConfig::default()`, with a "(thresholds unknown)" caveat
    ///
    /// Shared by the gate block and the parameter table so the two cannot
    /// glyph one number against two different thresholds.
    fn resolve_gate(state: &FitState) -> (GateConfig, GateThresholdSource) {
        match &state.resolved_gate {
            Some(g) => (g.clone(), GateThresholdSource::Resolved),
            None => (GateConfig::default(), GateThresholdSource::DefaultFallback),
        }
    }

    /// One Â glyph, painted. The band comes from the gate; only the colour is
    /// this renderer's own.
    fn a_glyph(&self, gate: &GateConfig, a: f64) -> String {
        match gate.a_band(a) {
            AgreementBand::Pass => self.ok("✓"),
            AgreementBand::SoftWarn => self.warn("~"),
            AgreementBand::Fail => self.err("✗"),
            AgreementBand::NotAssessed => self.dim("n/a").to_string(),
        }
    }

    fn gate_verdict_block(&self, state: &FitState) -> String {
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("compound scout-convergence gate")));

        let (gate, threshold_source) = Self::resolve_gate(state);

        // Â leg
        let max_a = state.tail_chain_agreement.values().cloned()
            .fold(0.0_f64, f64::max);
        let max_a_param = state.tail_chain_agreement.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone()).unwrap_or_else(|| "—".into());
        let a_passes = max_a < gate.a_thresh;
        let a_glyph = if a_passes { self.ok("✓") } else { self.err("✗") };
        s.push_str(&format!(
            "    Â leg:           max Â = {:.3} ({})  {}  (threshold {:.2})\n",
            max_a, max_a_param, a_glyph, gate.a_thresh));

        // Decibans leg
        if state.chain_eval_logliks.len() >= 2
            && state.chain_eval_ses.len() == state.chain_eval_logliks.len()
        {
            let hi = state.chain_eval_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = state.chain_eval_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let delta_db = (hi - lo) * NATS_TO_DB;
            let sigma_max = state.chain_eval_ses.iter().cloned()
                .fold(0.0_f64, f64::max);
            let se_floor_db = 8.0 * sigma_max * NATS_TO_DB;
            let threshold_db = gate.decibans_thresh.max(se_floor_db);
            let db_passes = delta_db < threshold_db;
            let db_glyph = if db_passes { self.ok("✓") } else { self.err("✗") };
            s.push_str(&format!(
                "    decibans leg:    Δ = {:.1} dB / threshold {:.1} dB  {}  (σ_max={:.2})\n",
                delta_db, threshold_db, db_glyph, sigma_max));

            let overall_pass = a_passes && db_passes;
            let overall = if overall_pass {
                self.ok("✓ PASS")
            } else {
                self.err("✗ FAIL")
            };
            s.push_str(&format!("    overall:         {}\n", overall));
        } else {
            s.push_str(&format!("    decibans leg:    {} (loglik-eval data not present)\n",
                self.dim("—")));
        }

        match threshold_source {
            GateThresholdSource::Resolved => {}
            GateThresholdSource::DefaultFallback => {
                s.push_str(&format!("    {}\n", self.warn(
                    "(thresholds unknown — fit_state.toml predates Phase 3; \
                     showing GateConfig::default())"
                )));
            }
        }
        s.push('\n');
        s
    }

    fn parameter_table(&self, state: &FitState) -> String {
        let mut s = String::new();
        // The Â column is glyphed against the SAME gate the block above
        // reports, so one number cannot print ✗ there and ✓ here.
        let (gate, _) = Self::resolve_gate(state);
        s.push_str(&format!(
            "  {}  {}\n",
            self.bold("parameter estimates (loglik-eval, selected chain θ̂)"),
            self.dim(&format!("Â threshold {:.2}", gate.a_thresh))));
        if state.start_values.is_empty() {
            s.push_str(&format!("    {}\n", self.dim("(no start_values in fit_state.toml)")));
            s.push('\n');
            return s;
        }
        let t0_set: std::collections::HashSet<&str> =
            state.perturb_only_at_t0_params.iter()
            .map(|s| s.as_str()).collect();
        let mut keys: Vec<&String> = state.start_values.keys().collect();
        keys.sort();
        // Filter to params we have agreement data for (these are the
        // estimated ones); fixed params are noise here. Fall back to
        // showing everything if no agreement data.
        let est_keys: Vec<&String> = if state.tail_chain_agreement.is_empty() {
            keys.clone()
        } else {
            keys.iter().filter(|k| state.tail_chain_agreement.contains_key(k.as_str()))
                .copied().collect()
        };
        // Sized to the names present: `{:12}` was a minimum width, so a
        // longer name shifted the `=` and every column after it.
        let w = name_col_width(est_keys.iter().map(|k| k.as_str()), 12);
        for k in est_keys {
            let v = state.start_values[k];
            let agreement = state.tail_chain_agreement.get(k).copied();
            let agreement_str = match agreement {
                // A collapsed within-chain variance leaves no Â at all; say so
                // rather than printing `Â=NaN ✗`, which reads as a failure the
                // estimator never assessed (gh#45).
                // A collapsed within-chain variance leaves no Â at all; say so
                // rather than printing `Â=NaN ✗`, which reads as a failure the
                // estimator never assessed (gh#45).
                Some(r) if !r.is_finite() => {
                    self.dim("Â=n/a (W ≈ 0; rely on Δ_dB)").to_string()
                }
                Some(r) => format!("Â={:.3} {}", r, self.a_glyph(&gate, r)),
                None => self.dim("Â=—").to_string(),
            };
            let t0_marker = if t0_set.contains(k.as_str()) {
                format!(" {}", self.dim("(t0-only)"))
            } else {
                String::new()
            };
            let date_marker = match self.cal.date_for(k, v) {
                Some(date) => format!("  ({})", date),
                None => String::new(),
            };
            s.push_str(&format!("    {:w$} = {:<12.6}  {}{}{}\n",
                fit_name(k, w), v, agreement_str, t0_marker, date_marker));
        }
        s.push('\n');
        s
    }

    fn chain_loglik_eval_table(&self, state: &FitState) -> String {
        let mut s = String::new();
        let n = state.chain_eval_logliks.len();
        s.push_str(&format!("  {}\n",
            self.bold(&format!("per-chain loglik-eval ({} chains)", n))));
        let best_idx = state.chain_eval_logliks.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        // ESS columns added in Phase 2. Today fit_state.toml only
        // carries the per-chain ll + se; per-chain ESS lives in
        // <stage>_summary.json's `chains[]`. Phase 1 of summary keeps
        // the table simple and reads only what's already in
        // fit_state.toml; ESS surfacing in this table waits for a
        // Phase 4 follow-up that loads <stage>_summary.json. Note
        // here so a future reader doesn't think it was forgotten.
        s.push_str(&format!("    {:6} {:>12}   {:>6}\n", "chain", "loglik", "± se"));
        for i in 0..n {
            let ll = state.chain_eval_logliks[i];
            let se = state.chain_eval_ses.get(i).copied().unwrap_or(f64::NAN);
            // Marker for the chain whose θ̂ is reported as the MLE
            // (cross-chain argmax of clean re-scored loglik).
            // "selected" rather than "winner" — neutral, no competition
            // framing; describes the operation (the cross-chain argmax
            // selected this chain's θ̂ as the reported MLE).
            let marker = if Some(i) == best_idx {
                format!("  {}", self.ok("← selected"))
            } else {
                String::new()
            };
            s.push_str(&format!("    {:6} {:>12.2}   ± {:>4.2}{}\n",
                i + 1, ll, se, marker));
        }
        s.push('\n');
        s
    }

    /// Bayesian (PGAS / PMMH) stage block. The interpretation surface
    /// is different from IF2: posterior mean per parameter,
    /// Gelman-Rubin R̂, ESS, and (for PMMH only) a scalar acceptance
    /// rate. The IF2 compound gate doesn't apply; convergence keys on
    /// `max R̂ < 1.05`.
    /// `ll_kind` is the stage's log-likelihood class, derived once by the
    /// caller from the typed `MethodResult` (the single authority — see
    /// `loglik::LoglikType`). It decides which per-chain trace column the
    /// outlier table may compare (gh#667); it is not re-derived here, so the
    /// summary and every other surface cannot disagree about it.
    fn bayesian_block(&self, stage: &str, method: &str, stage_dir: &Path, view: BayesianView<'_>, subset: Option<&SubsetInfo>, ll_kind: super::loglik::LoglikType) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "══ {} {} {}\n",
            self.bold(stage),
            self.dim(&format!("[{}]", method)),
            "═".repeat(74_usize.saturating_sub(stage.len() + method.len() + 3))
        ));

        // Convergence + efficiency come from the shared diagnostics; the
        // method-specific extras (acceptance summary, MAP loglik) are pulled
        // per variant. Every Bayesian sampler routes through this one block.
        let (diag, posterior_mean, acceptance_summary, map_loglik) = match view {
            BayesianView::Pgas(r) => (&r.diagnostics, &r.posterior_mean, None::<f64>, None::<f64>),
            BayesianView::Pmmh(r) => (
                &r.diagnostics,
                &r.posterior_mean,
                Some(r.acceptance_rate),
                Some(r.map_loglik),
            ),
            // nuts realized-acceptance is a dual-averaging target, not an M-H
            // accept rate — omit it rather than mislabel; MAP loglik does apply.
            BayesianView::Nuts(r) => (&r.diagnostics, &r.posterior_mean, None::<f64>, Some(r.map_loglik)),
        };

        // Header: with an active chain selection, `diag.n_chains` is already the
        // RETAINED count (recomputed), so name the subset and what was dropped.
        match subset {
            Some(info) => s.push_str(&format!(
                "  chains:       {} of {}  (excluded {})\n",
                info.kept.len(),
                info.n_total,
                info.excluded_csv()
            )),
            None => s.push_str(&format!("  chains:       {}\n", diag.n_chains)),
        }
        s.push_str(&format!("  samples:      {}\n", diag.n_samples));
        if let Some(ll) = map_loglik {
            s.push_str(&format!("  MAP loglik:   {:.1}\n", ll));
        }
        s.push('\n');

        // Convergence: Gelman-Rubin R̂ (NOT IF2's Â — see
        // method_result.rs §`max_chain_agreement` vs §`max_rhat`).
        s.push_str(&format!("  {}\n", self.bold("posterior convergence")));
        // "R̂ could not be computed" and "R̂ was computed and it was fine" must
        // not share a rendering. Folding an empty map to 0.0 printed
        // `max R̂ = 0.000 ✓` for a fit where every parameter was refused — a
        // fit that could not be assessed certifying itself.
        match diag.max_rhat_status() {
            MaxRhat::Reported(v) => {
                // The same band the stage's own end-of-run block glyphs
                // against, so one R̂ cannot read ✓ when the fit finishes and ✗
                // here. The verdict is also spelled out: on a surface a
                // public-health decision is read off, a glyph should not have
                // to carry the whole meaning.
                let band = RhatBand::of(v);
                let glyph = match band {
                    RhatBand::Converged => self.ok(band.glyph()),
                    RhatBand::NotConverged => self.warn(band.glyph()),
                    RhatBand::Severe => self.err(band.glyph()),
                    RhatBand::NotAssessed => self.dim(band.glyph()).to_string(),
                };
                s.push_str(&format!(
                    "    max R̂ = {:.3}  {}  {}  (rank-normalized split R̂, threshold {})\n",
                    v, glyph, band.describe(), RHAT_CONVERGED_THRESHOLD
                ));
                // Above the band, say WHY. R̂ is `max(rhat_bulk, rhat_folded)`
                // and the two halves have different remedies: a location
                // disagreement is a warm-up/drift problem, a spread
                // disagreement points at per-chain effective diversity
                // (docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md).
                if v >= RHAT_CONVERGED_THRESHOLD {
                    for (name, p) in &diag.per_param {
                        if !matches!(p.rhat(), Stat::Value(x) if x >= RHAT_CONVERGED_THRESHOLD) {
                            continue;
                        }
                        match p.rhat_decomposition() {
                            Some(d) => s.push_str(&format!("      {name} — {d}\n")),
                            // A fit written before the two halves were stored
                            // has the headline and nothing to decompose.
                            None => {}
                        }
                        if let Some(e) = p.per_chain_ess() {
                            s.push_str(&format!("        {e}\n"));
                        }
                    }
                }
            }
            MaxRhat::Unassessable { params } => {
                s.push_str(&format!("    max R̂ = —   {}\n", self.err("✗")));
                s.push_str(&format!(
                    "      R̂ could not be computed for {} of the estimated parameters,\n",
                    params.len()
                ));
                s.push_str("      which is a sampler failure, not a missing number:\n");
                for name in params.iter().take(8) {
                    let why = diag
                        .per_param
                        .get(name)
                        .and_then(|p| p.why_no_rhat())
                        .unwrap_or_else(|| "no reason recorded".to_string());
                    s.push_str(&format!("        {name} — {why}\n"));
                }
                if params.len() > 8 {
                    s.push_str(&format!("        … and {} more\n", params.len() - 8));
                }
                s.push_str("      This fit is NOT converged.\n");
            }
            MaxRhat::NotApplicable { reason } => {
                s.push_str("    max R̂ = —   (not assessed)\n");
                s.push_str(&format!(
                    "      a between-chain statistic was never possible here: {}\n",
                    reason.describe()
                ));
            }
            MaxRhat::NoParams => {
                s.push_str("    max R̂ = —   (no parameter was assessed across chains)\n");
            }
        }
        if let Some(acc) = acceptance_summary {
            s.push_str(&format!("    acceptance = {:.3} (mean across chains)\n", acc));
        }
        // Both efficiency lines are the min-parameter ESS over a denominator, so
        // both stand or fall with that minimum being defined. It is defined only
        // when every parameter assessed across chains reports a pooled ESS: a
        // minimum over the reporting subset RISES as a fit gets worse, because
        // the badly-mixing parameters drop out and the survivors set it (gh#687).
        // When it is undefined, name the parameters that withhold it — the blank
        // is then the diagnosis rather than a gap the reader must interpret.
        match diag.min_ess_status() {
            MinEss::Reported(min_ess) => {
                // ESS/iteration — the ALGORITHM-comparison metric: min-parameter
                // ESS per raw sampling step. `n_samples` is KEPT (thinned) draws,
                // so `× thin` recovers the raw iterations, making this invariant
                // to thinning and iteration count. Hardware-independent: "this
                // sampler mixes N× better per step" holds on any machine.
                if let Some(epi) = diag.ess_per_iter() {
                    s.push_str(&format!(
                        "    ESS/iter = {:.3}  (min-param ESS {:.0} / {} raw sampling iters)\n",
                        epi, min_ess, diag.raw_iters()
                    ));
                }
                // ESS/second — the RUNTIME metric: min-parameter ESS per second of
                // wall-clock. Also thinning-invariant, but hardware/implementation-
                // dependent, so it estimates runtime-to-target on THIS machine
                // rather than comparing algorithms. `None` wall-time (older runs)
                // simply omits it.
                if let (Some(eps), Some(secs)) =
                    (diag.ess_per_sec(), diag.wall_time_secs.filter(|s| *s > 0.0))
                {
                    s.push_str(&format!(
                        "    ESS/sec  = {:.2}  (min-param ESS {:.0} / {:.1}s wall)\n",
                        eps, min_ess, secs
                    ));
                }
            }
            MinEss::Unreportable { missing, n_expected } => {
                s.push_str("    ESS/iter = —   ESS/sec = —   (efficiency not reportable)\n");
                s.push_str(&format!(
                    "      {} of {} parameters report no bulk ESS — either the estimator\n",
                    missing.len(),
                    n_expected
                ));
                s.push_str("      refused their draws outright, or the rank-transformed draws\n");
                s.push_str("      are constant so the autocorrelation is undefined. Each says\n");
                s.push_str("      which in the per-parameter table below:\n");
                // Wrap the names to a readable width. The per-parameter table
                // below marks them too, but a column of dashes reads as "not
                // applicable"; the reader should not have to derive the list.
                let mut line = String::new();
                for (i, name) in missing.iter().enumerate() {
                    let piece = if i + 1 == missing.len() {
                        name.clone()
                    } else {
                        format!("{}, ", name)
                    };
                    if !line.is_empty() && line.len() + piece.len() > 62 {
                        s.push_str(&format!("        {}\n", line.trim_end()));
                        line.clear();
                    }
                    line.push_str(&piece);
                }
                if !line.is_empty() {
                    s.push_str(&format!("        {}\n", line.trim_end()));
                }
                s.push_str(&format!(
                    "      A minimum over the {} that did report would rise as the fit got\n",
                    n_expected - missing.len()
                ));
                s.push_str("      worse, so no efficiency headline is given.\n");
            }
            // No parameter was assessed across chains — there was never an
            // efficiency line here, and nothing to explain.
            MinEss::NoParams => {}
        }
        s.push('\n');

        // Per-chain loglik outlier diagnostic (gh#406). R̂/ESS above say WHETHER
        // the chains agreed; this says WHICH chain didn't. Same for every
        // Bayesian sampler (mh/pmmh/pgas/nuts) — read from the per-chain traces,
        // on the column this sampler's loglik CLASS makes comparable (gh#667).
        s.push_str(&self.bayesian_chain_loglik_table(stage_dir, diag.n_chains, ll_kind));

        // Per-chain saved-vs-forkable latent paths (gh#727). PGAS only: it is
        // the one Bayesian stage that writes latent paths, so for PMMH / NUTS
        // there is no count to omit. A path is usable downstream only when its
        // sweep is also a retained draw, and the two rules that decide those
        // are independent — so the counts must be readable side by side.
        if matches!(view, BayesianView::Pgas(_)) {
            s.push_str(&self.saved_path_table(stage_dir));
            s.push_str(&self.filter_ess_table(stage_dir));
            s.push_str(&self.latent_path_table(stage_dir));
        }

        // Posterior parameter table.
        s.push_str(&format!("  {}\n", self.bold("posterior summary")));
        if posterior_mean.is_empty() {
            s.push_str(&format!("    {}\n", self.dim("(no posterior parameters)")));
        } else {
            // The column is sized to the names in THIS table. `{:14}` is a
            // minimum width, so a name longer than it pushes every later
            // column right and the grid stops being one.
            let w = name_col_width(posterior_mean.keys().map(String::as_str), 14);
            s.push_str(&format!(
                "    {:w$} {:>14} {:>10} {:>10} {:>8}\n",
                "param", "mean", "ESS bulk", "ESS tail", "R̂"
            ));
            for (name, mean) in posterior_mean.iter() {
                // Both encodings of "no pooled ESS" — an absent key on the
                // loaded path, a present NaN on the --exclude-chains recompute
                // — render as the same dash (gh#691). One fact, one rendering.
                let ess_str = format!("{:>10}", diag.ess_cell(name, "—"));
                let date_marker = match self.cal.date_for(name, *mean) {
                    Some(date) => format!("  ({})", date),
                    None => String::new(),
                };
                s.push_str(&format!(
                    "    {:w$} {:>14} {} {:>10} {:>8}{}\n",
                    fit_name(name, w),
                    sig_figs(*mean, POSTERIOR_MEAN_SIG_FIGS),
                    ess_str,
                    diag.ess_tail_cell(name, "—"),
                    diag.rhat_cell(name, "—"),
                    date_marker
                ));
            }
        }
        s.push('\n');
        s
    }

    /// One right-aligned per-chain log-likelihood cell. `-inf` (a chain stuck
    /// off the support, gh#608) is rendered loudly and distinctly from `—`
    /// (nothing readable) — softening either one hides the only signal there
    /// is.
    fn chain_loglik_cell(&self, v: f64, width: usize) -> String {
        if v.is_finite() {
            format!("{:>width$.2}", v, width = width)
        } else if v == f64::NEG_INFINITY {
            format!("{:>width$}", self.err("-inf"), width = width)
        } else {
            format!("{:>width$}", "—", width = width)
        }
    }

    /// Per-chain saved-vs-forkable latent paths for a PGAS stage (gh#727).
    ///
    /// A PGAS chain retains a posterior draw every `thin` sweeps and writes a
    /// latent path on every `draw_stride`-th of those draws, so every written
    /// path is one a consumer can join to a draw — `simulate --init-state fit`
    /// or a `last_obs`-anchored `quantities {}` entry. The two counts are
    /// reported side by side anyway: they are measured from what each chain
    /// wrote, so a shortfall is a real event (a record skipped as incoherent,
    /// a chain resumed partway in) and is named rather than left for the
    /// reader to subtract.
    ///
    /// Reads the block `pgas.rs` writes into `pgas_summary.json`, through that
    /// module's own reader, so producer and consumer cannot drift.
    fn saved_path_table(&self, stage_dir: &Path) -> String {
        let path = stage_dir.join(crate::run_meta::FitAlgorithm::Pgas.summary_filename());
        let Some(report) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
            .as_ref()
            .and_then(super::pgas::read_saved_path_counts)
        else {
            // A stage that recorded no such block: nothing to report and
            // nothing known to be missing.
            return String::new();
        };
        if report.per_chain.is_empty() {
            return String::new();
        }
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("saved latent paths")));
        let total_saved: u64 = report.per_chain.iter().map(|c| c.n_saved).sum();
        let total_forkable: u64 = report.per_chain.iter().map(|c| c.n_forkable).sum();
        s.push_str(&format!("    {:6} {:>10} {:>10} {:>10}\n",
            "chain", "written", "forkable", "unusable"));
        for (i, c) in report.per_chain.iter().enumerate() {
            let lost = c.n_saved.saturating_sub(c.n_forkable);
            // Pad before coloring — the ANSI bytes would otherwise count
            // toward the field width and break the column.
            let cell = format!("{:>10}", lost);
            let lost_cell = if lost == 0 { self.dim(&cell).to_string() } else { self.warn(&cell) };
            s.push_str(&format!("    {:6} {:>10} {:>10}{}\n",
                i + 1, c.n_saved, c.n_forkable, lost_cell));
        }
        if total_forkable < total_saved {
            s.push_str(&format!("    {}\n", self.warn(&format!(
                "{} of {} written paths cannot be joined to a posterior draw",
                total_saved - total_forkable, total_saved))));
            s.push_str(&format!("    {}\n", self.dim(&format!(
                "a path is written on every {} retained draw (thin {}), so every \
                 one should be joinable; the shortfall is paths that were \
                 written outside this stage's retained set or skipped as \
                 incoherent records",
                report.draw_stride.map_or("—".to_string(), |v| v.to_string()),
                report.thin))));
        }
        s.push('\n');
        s
    }

    /// The conditional filter's ESS at every observation (gh#685), read from
    /// the `filter_ess` block of `pgas_summary.json` through that module's
    /// own reader, so producer and consumer cannot drift. Quiet for a stage
    /// that wrote no block — one that predates it, or in which no sweep
    /// scored an observation. The stage-end block is printed as it was, with
    /// the starved observations marked.
    fn filter_ess_table(&self, stage_dir: &Path) -> String {
        use super::filter_ess::FilterEss;
        let path = stage_dir.join(crate::run_meta::FitAlgorithm::Pgas.summary_filename());
        let Some(fe) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
            .as_ref()
            .and_then(FilterEss::read)
        else {
            return String::new();
        };
        let mut s = String::new();
        for line in fe.report().trim_start_matches('\n').lines() {
            s.push_str("  ");
            s.push_str(line);
            s.push('\n');
        }
        if fe.n_starved > 0 {
            s.push_str(&format!("    {}\n", self.warn(&format!(
                "the path through {} starved observation(s) is drawn from a handful of \
                 particles every sweep; look at the data at t={} before the model",
                fe.n_starved,
                fe.worst.first().map_or("?".to_string(), |o| crate::quantile::fmt_time(o.time))))));
        }
        s.push('\n');
        s
    }

    /// Convergence of the latent path itself (gh#822): R̂/ESS of every state at
    /// every substep across the chains, recomputed here from the chains'
    /// `trajectories.tsv` rather than read from `pgas_summary.json`, so a
    /// stage that finished before the block existed reports it too and the
    /// number is always the one the paths on disk give. Chains without a saved
    /// path (a refused start) are skipped, as the stage skips them. Quiet when
    /// no chain saved a path; a refusal (one chain, fewer than four saved
    /// paths) is named.
    ///
    /// The per-cell table is the one the stage writes; when the stage did not
    /// (it predates the block), it is written now, once — the same bytes a
    /// re-run would leave — and the block says so.
    fn latent_path_table(&self, stage_dir: &Path) -> String {
        use super::latent_convergence::{latent_convergence, read_stage_paths, LATENT_CONVERGENCE_TSV};
        let mut s = String::new();
        let (chains, columns) = match read_stage_paths(stage_dir) {
            Ok(Some(read)) => read,
            Ok(None) => return s,
            Err(e) => {
                s.push_str(&format!("  {}\n", self.bold("latent-path convergence")));
                s.push_str(&format!("    {}\n\n", self.warn(&format!("not computed: {e}"))));
                return s;
            }
        };
        match latent_convergence(&chains, &columns) {
            Ok(lc) => {
                // The stage-end block, indented into this section.
                for line in lc.report().trim_start_matches('\n').lines() {
                    s.push_str("  ");
                    s.push_str(line);
                    s.push('\n');
                }
                let table = stage_dir.join(LATENT_CONVERGENCE_TSV);
                if !table.is_file() {
                    match lc.write_tsv(&table) {
                        Ok(()) => s.push_str(&format!("    {}\n", self.dim(&format!(
                            "the stage predates this block; {LATENT_CONVERGENCE_TSV} written now from its saved paths")))),
                        Err(e) => s.push_str(&format!("    {}\n", self.dim(&format!(
                            "the stage predates this block and the table could not be written: {e}")))),
                    }
                }
            }
            Err(e) => {
                s.push_str(&format!("  {}\n", self.bold("latent-path convergence")));
                s.push_str(&format!("    {}\n", self.dim(&format!("not computed: {e}"))));
            }
        }
        s.push('\n');
        s
    }

    /// Per-chain log-likelihood breakdown for a Bayesian stage (gh#406).
    /// Reads each `chain_N/trace.tsv`, computes the per-chain mean post-burn-in
    /// loglik and its robust modified z-score (median/MAD) against the
    /// between-chain spread, and flags the outliers by name — so a user with a
    /// minority of chains stuck in a side mode sees *which* chains without
    /// opening every trace by hand.
    ///
    /// `kind` decides WHICH trace column is compared, by name: `obs_ll`
    /// (`log p(y | X, θ)`) for PGAS, `log_likelihood` (`log p(y | θ)`) for
    /// pmmh / mh / nuts (gh#667). For PGAS the complete-data target and its
    /// transition term are shown alongside — the sampler's own objective, and
    /// the term whose spread makes a flat-ridge diagnosis obvious — but they
    /// are never ranked on. When no per-chain traces exist, says so rather
    /// than skipping.
    fn bayesian_chain_loglik_table(
        &self,
        stage_dir: &Path,
        n_chains_expected: usize,
        kind: super::loglik::LoglikType,
    ) -> String {
        use super::chain_diagnostics as cd;
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("per-chain log-likelihood")));

        let Some(means) = cd::read_chain_mean_logliks(stage_dir, kind) else {
            s.push_str(&format!("    {}\n\n", self.dim(
                "(per-chain traces unavailable — cannot break down by chain)")));
            return s;
        };
        // Whether a cross-chain comparison is possible at all. When it is not,
        // the reason is stated and the ranked table is skipped — but the
        // degeneracy screens below still run: they read `log_posterior` and
        // `draws.tsv`, so a -inf or point-mass chain must not go unreported
        // just because the comparison column is missing (gh#608, gh#635).
        let mut scores = Vec::new();
        if means.scored_column_absent {
            // No silent gap: a stage whose traces predate the column would
            // otherwise render as rows of dashes that read like broken chains.
            s.push_str(&format!("    {}\n", self.dim(&format!(
                "(traces carry no `{}` column — cannot compare chains on it)",
                means.scored_column))));
        } else if means.scored.len() < 2 {
            s.push_str(&format!("    {}\n", self.dim(
                "(need ≥2 chains with traces for a cross-chain outlier score)")));
        } else {
            if n_chains_expected != 0 && means.scored.len() != n_chains_expected {
                s.push_str(&format!("    {}\n", self.dim(&format!(
                    "(found traces for {} of {} chains)",
                    means.scored.len(), n_chains_expected))));
            }
            scores = cd::chain_loglik_mod_zscores(&means.scored);
            // gh#667: say what is being ranked, and why the other two columns
            // are present but not ranked. Without this the reader has no way to
            // know the table stopped scoring the sampler's own target.
            if means.complete_data.is_some() {
                s.push_str(&format!("    {}\n", self.dim(
                    "ranked on obs ll = log p(y | X): does this chain reproduce the DATA.")));
                s.push_str(&format!("    {}\n", self.dim(
                    "transition ll = log p(X | θ) and complete ll = log p(y, X | θ) are path")));
                s.push_str(&format!("    {}\n", self.dim(
                    "densities at each chain's OWN path — shown, not ranked (gh#667).")));
            }
            match &means.complete_data {
                Some(_) => s.push_str(&format!("    {:6} {:>14}  {:>7}  {:>14} {:>14}   {}\n",
                    "chain", "obs ll", "mod-z", "transition ll", "complete ll", "flag")),
                None => s.push_str(&format!("    {:6} {:>14}  {:>7}   {}\n",
                    "chain", "mean loglik", "mod-z", "flag")),
            }
            for (i, sc) in scores.iter().enumerate() {
                let flag = if sc.is_outlier {
                    self.err("← outlier")
                } else {
                    String::new()
                };
                // gh#608: distinguish "stuck at -inf" (a degenerate chain whose
                // draws contaminate the pooled numbers) from "no readable trace"
                // (—). The old rendering softened the one signal that existed.
                let ll = self.chain_loglik_cell(sc.mean_loglik, 14);
                // A non-finite mod-z (unreadable trace) renders `—`, never a fake 0.
                let z = if sc.mod_z.is_finite() {
                    format!("{:>7.2}", sc.mod_z)
                } else {
                    format!("{:>7}", "—")
                };
                match &means.complete_data {
                    Some(cd_means) => {
                        let trans = self.chain_loglik_cell(
                            cd_means.transition.get(i).copied().unwrap_or(f64::NAN), 14);
                        let complete = self.chain_loglik_cell(
                            cd_means.complete.get(i).copied().unwrap_or(f64::NAN), 14);
                        s.push_str(&format!("    {:6} {}  {}  {} {}   {}\n",
                            sc.chain, ll, z, trans, complete, flag));
                    }
                    None => s.push_str(&format!("    {:6} {}  {}   {}\n", sc.chain, ll, z, flag)),
                }
            }
        }

        // gh#608 (ebola F8): the stuck-state screen. A chain recording a
        // non-finite log-posterior as its CURRENT state is degenerate
        // (gh#607); the robust mod-z above cannot flag it (a -inf mean is
        // excluded from the centre/scale), so it gets its own loud line,
        // right where the reader is told the pooled numbers include it.
        if let Some(neginf) = cd::read_chain_neginf(stage_dir) {
            for d in neginf.iter().filter(|d| d.n_neginf > 0 && d.n_retained > 0) {
                s.push_str(&format!("    {}\n", self.err(&format!(
                    "⚠ chain {}: log-posterior -inf on {:.1}% of retained draws \
                     ({}/{}) — a DEGENERATE chain; its draws are in draws.tsv \
                     and every pooled number in this block. View without it: \
                     --exclude-chains <stage>:{} (exclusion stays explicit — \
                     gh#419).",
                    d.chain,
                    100.0 * d.n_neginf as f64 / d.n_retained as f64,
                    d.n_neginf, d.n_retained, d.chain,
                ))));
            }
        }

        // gh#635 (ebola item 1): a POINT-MASS chain — one distinct parameter
        // vector across all retained draws (zero accepted θ-moves) — evades
        // both the mod-z score and the −inf screen when its start has finite
        // density. Same loud treatment.
        if let Some(uniq) = cd::read_chain_unique_draws(stage_dir) {
            for u in uniq.iter().filter(|u| u.n_unique == 1 && u.n_draws > 1) {
                s.push_str(&format!("    {}\n", self.err(&format!(
                    "⚠ chain {}: ONE distinct parameter vector across {} retained \
                     draws (zero accepted moves) — a point-mass chain; its draws \
                     are in draws.tsv and every pooled number in this block. View \
                     without it: --exclude-chains <stage>:{} (exclusion stays \
                     explicit — gh#419).",
                    u.chain, u.n_draws, u.chain,
                ))));
            }
        }

        let flagged = cd::outlier_labels(&scores);
        if !flagged.is_empty() {
            // The one-line nudge: chains in a distinctly different part of the
            // likelihood surface is the near-unidentified-parameter (flat-ridge)
            // signature.
            s.push_str(&format!("    {}\n", self.warn(&format!(
                "⚠ chains disagree ({} — {} far from the rest) — is a parameter \
                 weakly identified? Inspect its per-chain posterior.",
                flagged.join(", "), means.scored_column))));
        }
        s.push('\n');
        s
    }

    fn provenance_block(&self, stage_dir: &str, state: &FitState)
        -> ProvenanceBlock
    {
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("provenance")));
        let final_path = format!("{}/final_params.toml", stage_dir);
        let mle_path = format!("{}/mle_params.toml", stage_dir);
        let mut failed = false;

        let final_params = read_param_values(&final_path);
        let mle_params = read_param_values(&mle_path);

        match (&final_params, &mle_params) {
            (Some(f), Some(m)) => {
                let agree = params_agree(f, m);
                if agree {
                    s.push_str(&format!("    final_params.toml ↔ mle_params.toml: {}\n",
                        self.ok("✓ params match")));
                } else {
                    s.push_str(&format!("    final_params.toml ↔ mle_params.toml: {}\n",
                        self.err("✗ DISAGREE — silent-wrong-answer (GH #16) class")));
                    failed = true;
                }
            }
            (None, _) => s.push_str(&format!("    final_params.toml: {}\n",
                self.dim("(absent)"))),
            (_, None) => s.push_str(&format!("    mle_params.toml:   {}\n",
                self.dim("(absent)"))),
        }

        // fit_state winner ↔ final_params
        if !state.start_values.is_empty() && final_params.is_some() {
            let f = final_params.as_ref().unwrap();
            let mut state_matches = true;
            for (k, fv) in f {
                if let Some(sv) = state.start_values.get(k) {
                    if (sv - fv).abs() > 1e-9 * fv.abs().max(1.0) {
                        state_matches = false;
                        break;
                    }
                }
            }
            if state_matches {
                s.push_str(&format!("    fit_state.toml ↔ final_params.toml:   {}\n",
                    self.ok("✓ params match")));
            } else {
                s.push_str(&format!("    fit_state.toml ↔ final_params.toml:   {}\n",
                    self.err("✗ DISAGREE — fit_state's start_values diverge from winner")));
                failed = true;
            }
        }

        ProvenanceBlock { text: s, failed }
    }

    // ── Colour helpers ──────────────────────────────────────────────

    fn wrap(&self, code: &str, s: &str) -> String {
        if self.use_color {
            format!("\x1b[{}m{}\x1b[0m", code, s)
        } else {
            s.to_string()
        }
    }
    fn bold(&self, s: &str)  -> String { self.wrap("1", s) }
    fn dim(&self, s: &str)   -> String { self.wrap("2", s) }
    fn ok(&self, s: &str)    -> String { self.wrap("32", s) }
    fn warn(&self, s: &str)  -> String { self.wrap("33", s) }
    fn err(&self, s: &str)   -> String { self.wrap("31", s) }
}

struct ProvenanceBlock {
    text: String,
    failed: bool,
}

enum GateThresholdSource {
    /// Read from `state.resolved_gate` (Phase 3 — what the run was
    /// actually judged against).
    Resolved,
    /// Legacy fit_state.toml — no resolved_gate. Showing
    /// `GateConfig::default()` with a caveat.
    DefaultFallback,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn ci_env_set() -> bool {
    matches!(std::env::var("CI").as_deref(), Ok("true") | Ok("1"))
}

/// Resolve color preference with the standard Unix precedence:
/// `--no-color` flag > `NO_COLOR` env (forces off; see no-color.org)
/// > `CLICOLOR_FORCE` env (forces on regardless of TTY; common
/// convention used by ls / grep / git when piped through `less -R`)
/// > TTY auto-detect.
///
/// Default behavior (no flag, no env): colored when stdout is a TTY,
/// plain text otherwise. Pipe to `less -R` with `CLICOLOR_FORCE=1`
/// to keep colors in a pager.
fn should_use_color(no_color_flag: bool) -> bool {
    if no_color_flag { return false; }
    if std::env::var("NO_COLOR").is_ok() { return false; }
    if std::env::var("CLICOLOR_FORCE").is_ok() { return true; }
    is_stdout_tty()
}

fn is_stdout_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdout().is_terminal()
}

/// Read a params TOML, returning a flat name→value map. Skips any
/// `[provenance]` section per `util::load_params_toml`'s convention.
fn read_param_values(path: &str) -> Option<std::collections::HashMap<String, f64>> {
    crate::util::load_params_toml(path).ok()
}

/// Two parameter dictionaries agree iff every shared key has values
/// matching to floating-point tolerance. Disjoint keys are treated as
/// "match" — `final_params.toml` and `mle_params.toml` legitimately
/// have non-overlapping fields (e.g. mle_params has more fixed params
/// rolled in). The shared subset is what would diverge under #16.
fn params_agree(
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

// ── Phase 4 / 5: structured doc + multi-format renderers ────────────

/// Structured fit-interpretation document. Serialized as `--format
/// json`; consumed by md / latex renderers. Stable schema versioned
/// via `schema.version`.
///
/// The top-level `table_row` field carries the cross-fit row schema
/// (`name = "table_row"`, `version = 1`) embedded verbatim — the
/// `summary ⊆ table` invariant in proposal §3 / Deliverable C asserts
/// it is byte-equal to a `fit table --hash <h> --format json` row.
#[derive(Debug, Clone, Serialize)]
pub struct FitSummaryDoc {
    pub schema: SchemaInfo,
    pub fit_dir: String,
    /// Embedded `table_row` block. The single source of truth for the
    /// cross-fit schema: any field added here must also appear in
    /// `fit table`'s row output, enforced by Deliverable C.
    pub table_row: TableRow,
    pub stages: Vec<StageReport>,
    /// Read-side chain selection (`--exclude-chains`), when active: the
    /// `{excluded, kept, n_total}` provenance for the subset the stages'
    /// diagnostics were recomputed over. Absent (omitted) for a full-cloud
    /// summary, so existing machine consumers are unaffected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_selection: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SchemaInfo {
    pub version: u32,
    pub camdl_version: String,
}

/// Per-stage report. Method-aware: IF2 stages populate the
/// `gate` / `parameters` / `chains` / `provenance` / `heuristic` blocks;
/// PGAS / PMMH stages set them all to `None` and lean on the typed
/// `method_result` payload, which carries posterior summaries / R̂ /
/// ESS for Bayesian methods.
#[derive(Debug, Clone, Serialize)]
pub struct StageReport {
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
    /// Compound IF2 gate. `None` for Bayesian stages (the gate
    /// doesn't apply).
    pub gate: Option<GateReport>,
    pub stage_progression: Option<StageProgression>,
    /// IF2 estimated parameters with Â. Empty for Bayesian stages
    /// (consult `method_result` for posterior_mean / ess).
    pub parameters: Vec<ParameterReport>,
    /// IF2 per-chain loglik-eval table. Empty for Bayesian.
    pub chains: Vec<ChainReport>,
    /// IF2 provenance cross-check (final_params ↔ mle_params, etc.).
    /// `None` for Bayesian.
    pub provenance: Option<ProvenanceReport>,
    /// Typed `MethodResult` payload — pattern-match for headline
    /// scalars. Always populated for stages that loaded successfully.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method_result: Option<MethodResult>,
    /// Advisory IF2 interpretation block. `None` for Bayesian
    /// stages.
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
    /// `"resolved"` when read from `state.resolved_gate`; `"default_fallback"`
    /// when fit_state.toml predates Phase 3 and we substituted
    /// `GateConfig::default()`. Critical signal for downstream readers.
    pub threshold_source: String,
    pub resolved_gate: Option<GateConfig>,
    pub resolved_loglik_eval: Option<LoglikEvalConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageProgression {
    pub previous_stage: String,
    pub previous_loglik: f64,
    pub delta_nats: f64,
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
    pub chain_id: usize,
    pub clean_loglik: f64,
    pub clean_se: f64,
    pub is_winner: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProvenanceReport {
    pub final_params_matches_mle_params: Option<bool>,
    pub fit_state_winner_matches_final_params: Option<bool>,
    pub stale_camdl_version: Option<String>,
}

impl ProvenanceReport {
    fn any_failed(&self) -> bool {
        matches!(self.final_params_matches_mle_params, Some(false))
            || matches!(self.fit_state_winner_matches_final_params, Some(false))
    }
}

impl StageReport {
    /// Helper for `--strict`: returns whether this stage's provenance
    /// cross-check (when applicable) reported any failure. Bayesian
    /// stages always return false (no provenance check applies).
    fn provenance_failed(&self) -> bool {
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

/// Walk the stage dirs and build a `FitSummaryDoc`. Used by JSON, MD,
/// and LaTeX formatters. Pure on its inputs (file system + the args
/// it was called with).
fn build_summary_doc(
    dir: &str,
    stages: &[ResolvedStage],
    selection: Option<&ChainSelection>,
) -> FitSummaryDoc {
    let cal = load_calendar_context(Path::new(dir));
    let mut stage_reports: Vec<StageReport> = Vec::new();
    let mut prev_loglik: Option<f64> = None;
    let mut prev_stage_name_owned: Option<String> = None;
    // The chain-selection provenance (stamped on the doc) + one-shot advisory.
    let mut chain_selection_json: Option<serde_json::Value> = None;
    let mut advised = false;
    for resolved in stages {
        let stage_dir_str = resolved.stage_dir.to_string_lossy().into_owned();
        let mut typed = MethodResult::load_from(&resolved.stage_dir, &resolved.method).ok();
        // Chain selection: recompute this Bayesian stage's diagnostics over the
        // retained chains before the report is built from the (now mutated)
        // typed payload — so JSON / MD / LaTeX all carry the subset diagnostics.
        if let Some(sel) = selection {
            if matches!(
                typed,
                Some(MethodResult::Pgas(_)) | Some(MethodResult::Pmmh(_)) | Some(MethodResult::Nuts(_))
            ) {
                if let Some(t) = typed.as_mut() {
                    // Same authority as the text path: the class comes from the
                    // typed result, never from a column position (gh#667).
                    let ll_kind = crate::fit::loglik::LoglikType::from(&*t);
                    match apply_selection_to_typed(t, &resolved.stage_dir, sel) {
                        Ok(info) => {
                            if !advised {
                                chain_selection_advisory(&resolved.stage_dir, &info, ll_kind);
                                advised = true;
                            }
                            chain_selection_json = Some(info.to_json());
                        }
                        Err(e) => {
                            eprintln!("error: {e}");
                            std::process::exit(1);
                        }
                    }
                }
            }
        }
        let prev_name = prev_stage_name_owned.as_deref();
        let report = match (&typed, resolved.method.as_str()) {
            (Some(MethodResult::If2(_)), _) => {
                let state = match FitState::load(&stage_dir_str) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let r = if2_stage_report(
                    &resolved.stage,
                    &stage_dir_str,
                    &state,
                    typed.clone(),
                    prev_loglik,
                    prev_name,
                    &cal,
                );
                prev_loglik = Some(state.best_loglik);
                r
            }
            (Some(MethodResult::Pgas(_)), _)
            | (Some(MethodResult::Pmmh(_)), _)
            | (Some(MethodResult::Nuts(_)), _) => {
                let r = bayesian_stage_report(
                    &resolved.stage,
                    &resolved.method,
                    typed.clone(),
                    &cal,
                );
                match &typed {
                    Some(MethodResult::Pmmh(p)) => prev_loglik = Some(p.map_loglik),
                    Some(MethodResult::Nuts(p)) => prev_loglik = Some(p.map_loglik),
                    _ => {}
                }
                r
            }
            _ => continue,
        };
        stage_reports.push(report);
        prev_stage_name_owned = Some(resolved.stage.clone());
    }

    // Build the table_row block. For single-fit summary the row is
    // alone in scope, so config_diff is the identity diff (empty
    // diff vs self) and `delta_ll_vs_best` is 0.0.
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let table_row = build_summary_table_row(Path::new(dir), now_unix);

    FitSummaryDoc {
        schema: SchemaInfo {
            version: SCHEMA_VERSION,
            camdl_version: version::VERSION_SHORT.to_string(),
        },
        fit_dir: dir.to_string(),
        table_row,
        stages: stage_reports,
        chain_selection: chain_selection_json,
    }
}

/// Project the fit_dir to a `TableRow`, using the identity diff and
/// `delta_ll_vs_best = 0.0` (single-fit scope). On error,
/// substitutes a placeholder row whose `fit_id` reflects the failure
/// so JSON consumers see something rather than a crash.
fn build_summary_table_row(fit_dir: &Path, now_unix: i64) -> TableRow {
    let fit_hash = FitView::read(fit_dir).map(|v| v.fit_hash).unwrap_or_default();
    let diff = ConfigDiff::identity(&fit_hash);
    match table_row::build_row(fit_dir, diff, 0.0, now_unix) {
        Ok(r) => r,
        Err(e) => {
            // Placeholder so the JSON shape stays stable on errors. The
            // schema discriminator is correct; everything else is
            // empty/zero. Loud-via-stderr so tests don't silently
            // pass on a broken fit.
            eprintln!("warning: cannot build table_row for {}: {}", fit_dir.display(), e);
            TableRow {
                schema: table_row::TableRowSchema::current(),
                fit_id: String::new(),
                fit_hash: String::new(),
                label: None,
                stem: String::new(),
                model_identity: String::new(),
                stages: Vec::new(),
                method: String::new(),
                config_diff_from_baseline: ConfigDiff::identity(""),
                converged: false,
                gate_verdict: "n/a".into(),
                best_loglik: None,
                loglik_type: None,
                max_chain_agreement: None,
                max_rhat: None,
                acceptance_rate: None,
                ess_at_mle: None,
                ess_posterior: None,
                ess_per_iter: None,
                ess_per_sec: None,
                params: BTreeMap::new(),
                delta_ll_vs_best: 0.0,
                age_seconds: 0,
                created_at: String::new(),
                stale: false,
                stale_reason: None,
                quantities: BTreeMap::new(),
            }
        }
    }
}

/// Bayesian stage report: only `method_result` carries content.
/// IF2-specific subfields default to None / empty.
fn bayesian_stage_report(
    stage: &str,
    method: &str,
    method_result: Option<MethodResult>,
    cal: &CalendarContext,
) -> StageReport {
    let (n_chains, best_loglik) = match &method_result {
        Some(MethodResult::Pgas(r)) => (r.diagnostics.n_chains, None),
        Some(MethodResult::Pmmh(r)) => (r.diagnostics.n_chains, Some(r.map_loglik)),
        Some(MethodResult::Nuts(r)) => (r.diagnostics.n_chains, Some(r.map_loglik)),
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
        Some(MethodResult::Nuts(r)) => dates_from(&r.posterior_mean),
        _ => BTreeMap::new(),
    };
    let loglik_type = method_result.as_ref().map(crate::fit::loglik::LoglikType::from);
    StageReport {
        name: stage.to_string(),
        method: method.to_string(),
        n_chains,
        best_loglik,
        loglik_type,
        initial_loglik: None,
        camdl_version: Some(version::VERSION_SHORT.to_string()),
        gate: None,
        stage_progression: None,
        parameters: Vec::new(),
        chains: Vec::new(),
        provenance: None,
        method_result,
        heuristic: None,
        param_dates,
    }
}

fn if2_stage_report(
    stage: &str,
    stage_dir: &str,
    state: &FitState,
    method_result: Option<MethodResult>,
    prev_loglik: Option<f64>,
    prev_stage_name: Option<&str>,
    cal: &CalendarContext,
) -> StageReport {
    // Gate analysis — same logic as Formatter::gate_verdict_block but
    // returning structured data instead of pre-formatted strings.
    let (gate_cfg, threshold_source) = match &state.resolved_gate {
        Some(g) => (g.clone(), "resolved".to_string()),
        None    => (GateConfig::default(), "default_fallback".to_string()),
    };
    let max_a = state.tail_chain_agreement.values().cloned()
        .fold(0.0_f64, f64::max);
    let max_a_param = state.tail_chain_agreement.iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(k, _)| k.clone());
    let a_passes = max_a < gate_cfg.a_thresh;

    let (delta_db, threshold_db, sigma_max, db_passes) =
        if state.chain_eval_logliks.len() >= 2
            && state.chain_eval_ses.len() == state.chain_eval_logliks.len()
        {
            let hi = state.chain_eval_logliks.iter().cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let lo = state.chain_eval_logliks.iter().cloned()
                .fold(f64::INFINITY, f64::min);
            let dd = (hi - lo) * NATS_TO_DB;
            let sm = state.chain_eval_ses.iter().cloned()
                .fold(0.0_f64, f64::max);
            let se_floor_db = 8.0 * sm * NATS_TO_DB;
            let td = gate_cfg.decibans_thresh.max(se_floor_db);
            (Some(dd), Some(td), Some(sm), Some(dd < td))
        } else {
            (None, None, None, None)
        };
    let overall_pass = db_passes.map(|p| p && a_passes);

    let gate = GateReport {
        max_a_hat: max_a,
        max_a_param,
        a_thresh: gate_cfg.a_thresh,
        a_passes,
        delta_db, threshold_db, sigma_max, db_passes, overall_pass,
        threshold_source,
        resolved_gate: state.resolved_gate.clone(),
        resolved_loglik_eval: state.resolved_loglik_eval.clone(),
    };

    // Parameters
    let t0_set: std::collections::HashSet<&str> =
        state.perturb_only_at_t0_params.iter()
        .map(|s| s.as_str()).collect();
    let mut keys: Vec<&String> = state.start_values.keys().collect();
    keys.sort();
    let est_keys: Vec<&String> = if state.tail_chain_agreement.is_empty() {
        keys.clone()
    } else {
        keys.iter().filter(|k| state.tail_chain_agreement.contains_key(k.as_str()))
            .copied().collect()
    };
    let parameters: Vec<ParameterReport> = est_keys.iter().map(|k| {
        let estimate = state.start_values[*k];
        ParameterReport {
            name: (*k).clone(),
            estimate,
            chain_agreement: state.tail_chain_agreement.get(*k).copied(),
            perturb_only_at_t0: t0_set.contains(k.as_str()),
            estimate_date: cal.date_for(k, estimate),
        }
    }).collect();
    let param_dates: BTreeMap<String, String> = parameters
        .iter()
        .filter_map(|p| p.estimate_date.clone().map(|d| (p.name.clone(), d)))
        .collect();

    // Chains
    let n = state.chain_eval_logliks.len();
    let best_idx = state.chain_eval_logliks.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    let chains: Vec<ChainReport> = (0..n).map(|i| ChainReport {
        chain_id: i + 1,
        clean_loglik: state.chain_eval_logliks[i],
        clean_se: state.chain_eval_ses.get(i).copied().unwrap_or(f64::NAN),
        is_winner: Some(i) == best_idx,
    }).collect();

    // Provenance
    let final_path = format!("{}/final_params.toml", stage_dir);
    let mle_path = format!("{}/mle_params.toml", stage_dir);
    let final_params = read_param_values(&final_path);
    let mle_params = read_param_values(&mle_path);
    let final_matches_mle = match (&final_params, &mle_params) {
        (Some(f), Some(m)) => Some(params_agree(f, m)),
        _ => None,
    };
    let state_matches_final = match &final_params {
        Some(f) if !state.start_values.is_empty() => {
            let mut ok = true;
            for (k, fv) in f {
                if let Some(sv) = state.start_values.get(k) {
                    let scale = fv.abs().max(1.0);
                    if (sv - fv).abs() > 1e-9 * scale {
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
    let provenance = ProvenanceReport {
        final_params_matches_mle_params: final_matches_mle,
        fit_state_winner_matches_final_params: state_matches_final,
        stale_camdl_version: stale,
    };

    let stage_progression = prev_loglik.zip(prev_stage_name).map(|(prev, prev_name)| StageProgression {
        previous_stage: prev_name.to_string(),
        previous_loglik: prev,
        delta_nats: state.best_loglik - prev,
    });

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

    StageReport {
        name: stage.to_string(),
        method: "if2".into(),
        n_chains: state.n_chains,
        best_loglik: Some(state.best_loglik),
        // Derive from the typed result; fall back to the stage's recorded
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
        stage_progression,
        parameters,
        chains,
        provenance: Some(provenance),
        method_result,
        heuristic: Some(HeuristicReport { overall_status, interpretation }),
        param_dates,
    }
}

/// Render a `FitSummaryDoc` as GitHub-flavoured Markdown. Tabular per
/// stage, code-fenced parameter tables. Suitable for embedding in book
/// chapters via `run_cli("camdl fit summary {dir} --format md", ...)`.
pub fn render_markdown(doc: &FitSummaryDoc) -> String {
    let mut s = String::new();
    s.push_str(&format!("# Fit summary: `{}`\n\n", doc.fit_dir));
    s.push_str(&format!("camdl `{}` (schema v{})\n\n",
        doc.schema.camdl_version, doc.schema.version));
    if let Some(cs) = &doc.chain_selection {
        s.push_str(&format!(
            "> **Chain subset** — diagnostics recomputed over {} of {} chains (excluded {}). \
             Post-hoc chain exclusion biases the posterior toward the retained mode.\n\n",
            cs["kept"].as_array().map(|a| a.len()).unwrap_or(0),
            cs["n_total"].as_u64().unwrap_or(0),
            render_id_csv(&cs["excluded"]),
        ));
    }
    if doc.stages.is_empty() {
        s.push_str("_(no MLE stages found)_\n");
        return s;
    }
    for stage in &doc.stages {
        s.push_str(&render_md_stage(stage));
    }
    s
}

fn render_md_stage(stage: &StageReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("## `{}` ({})\n\n", stage.name, stage.method));
    if let Some(ll) = stage.best_loglik {
        s.push_str(&format!("- best loglik: **{:.2}** ({})\n",
            ll, crate::fit::loglik::LoglikType::tag_or_unknown(stage.loglik_type)));
    }
    s.push_str(&format!("- chains: {}\n", stage.n_chains));
    if let Some(init) = stage.initial_loglik {
        s.push_str(&format!("- initial loglik: {:.2}\n", init));
    }
    if let Some(prog) = &stage.stage_progression {
        s.push_str(&format!("- vs `{}`: Δ = {:+.2} nats\n",
            prog.previous_stage, prog.delta_nats));
    }
    if let Some(prov) = &stage.provenance {
        if let Some(stale) = &prov.stale_camdl_version {
            s.push_str(&format!("- ⚠ stale: produced by camdl `{}`, current is `{}`\n",
                stale, version::VERSION_SHORT));
        }
    }
    s.push('\n');

    // IF2 gate verdict (only when populated).
    if let Some(gate) = &stage.gate {
        s.push_str("### Compound scout-convergence gate\n\n");
        s.push_str("| leg | value | threshold | pass? |\n|---|---|---|---|\n");
        let glyph = |b| if b { "✓" } else { "✗" };
        s.push_str(&format!("| Â (max over params{}) | {:.3} | {:.2} | {} |\n",
            gate.max_a_param.as_deref().map(|p| format!(", `{}`", p)).unwrap_or_default(),
            gate.max_a_hat, gate.a_thresh,
            glyph(gate.a_passes)));
        if let (Some(dd), Some(td), Some(p)) = (gate.delta_db, gate.threshold_db, gate.db_passes) {
            s.push_str(&format!("| decibans-spread | {:.1} dB | {:.1} dB | {} |\n",
                dd, td, glyph(p)));
        } else {
            s.push_str("| decibans-spread | _(no loglik-eval data)_ | — | — |\n");
        }
        s.push_str(&format!("\n**overall:** {}\n", match gate.overall_pass {
            Some(true)  => "✓ PASS",
            Some(false) => "✗ FAIL",
            None        => "(indeterminate)",
        }));
        if gate.threshold_source == "default_fallback" {
            s.push_str("\n> ⚠ thresholds unknown — fit_state.toml predates Phase 3; showing `GateConfig::default()`.\n");
        }
        s.push('\n');
    }

    // Method-specific posterior block.
    if let Some(MethodResult::Pgas(p)) = &stage.method_result {
        s.push_str(&format!("### Posterior summary (PGAS, max R̂ = {})\n\n",
            max_rhat_cell(&p.diagnostics)));
        s.push_str("| param | mean | q025 | q975 | ESS bulk | ESS tail | R̂ |\n|---|---|---|---|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let ess = p.diagnostics.ess_cell(name, "—");
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} | {} | {} | {} |\n", name, mean_cell, q025, q975, ess,
                p.diagnostics.ess_tail_cell(name, "—"), p.diagnostics.rhat_cell(name, "—")));
        }
        s.push('\n');
    } else if let Some(MethodResult::Pmmh(p)) = &stage.method_result {
        s.push_str(&format!(
            "### Posterior summary (PMMH, max R̂ = {}, acceptance = {:.3})\n\n",
            max_rhat_cell(&p.diagnostics), p.acceptance_rate
        ));
        s.push_str("| param | mean | ESS bulk | ESS tail | R̂ |\n|---|---|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let ess = p.diagnostics.ess_cell(name, "—");
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} | {} |\n", name, mean_cell, ess,
                p.diagnostics.ess_tail_cell(name, "—"), p.diagnostics.rhat_cell(name, "—")));
        }
        s.push('\n');
    } else if let Some(MethodResult::Nuts(p)) = &stage.method_result {
        s.push_str(&format!(
            "### Posterior summary (NUTS, max R̂ = {}, divergences = {})\n\n",
            max_rhat_cell(&p.diagnostics), p.n_divergent
        ));
        s.push_str("| param | mean | q025 | q975 | ESS bulk | ESS tail | R̂ |\n|---|---|---|---|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let ess = p.diagnostics.ess_cell(name, "—");
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} | {} | {} | {} |\n", name, mean_cell, q025, q975, ess,
                p.diagnostics.ess_tail_cell(name, "—"), p.diagnostics.rhat_cell(name, "—")));
        }
        s.push('\n');
    }

    // IF2 parameter table.
    if !stage.parameters.is_empty() {
        s.push_str("### Parameter estimates (loglik-eval, selected chain θ̂)\n\n");
        s.push_str("| name | estimate | Â | flags |\n|---|---|---|---|\n");
        for p in &stage.parameters {
            let a_str = p.chain_agreement
                .map(|r| format!("{:.3}", r))
                .unwrap_or_else(|| "—".into());
            let flag = if p.perturb_only_at_t0 { "t0-only" } else { "" };
            let est_cell = match &p.estimate_date {
                Some(date) => format!("{:.6} ({})", p.estimate, date),
                None => format!("{:.6}", p.estimate),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} |\n",
                p.name, est_cell, a_str, flag));
        }
        s.push('\n');
    }

    // IF2 per-chain loglik-eval.
    if !stage.chains.is_empty() {
        s.push_str(&format!("### Per-chain loglik-eval ({} chains)\n\n", stage.chains.len()));
        // Column / marker naming matches the text emitter and CLAUDE.md:
        // "loglik" / "selected", not "clean_ll" / "winner". The
        // ChainSummary.is_winner field name on the data side stays
        // (it's the data model); only the display label changed.
        s.push_str("| chain | loglik | ± se | selected |\n|---|---|---|---|\n");
        for c in &stage.chains {
            let mark = if c.is_winner { "★" } else { "" };
            s.push_str(&format!("| {} | {:.2} | {:.2} | {} |\n",
                c.chain_id, c.clean_loglik, c.clean_se, mark));
        }
        s.push('\n');
    }

    // IF2 provenance.
    if let Some(prov) = &stage.provenance {
        s.push_str("### Provenance\n\n");
        let prov_row = |label: &str, val: Option<bool>| {
            match val {
                Some(true)  => format!("- {}: ✓\n", label),
                Some(false) => format!("- {}: ✗ **DISAGREE**\n", label),
                None        => format!("- {}: _(absent)_\n", label),
            }
        };
        s.push_str(&prov_row("final_params.toml ↔ mle_params.toml",
            prov.final_params_matches_mle_params));
        s.push_str(&prov_row("fit_state.toml ↔ final_params.toml",
            prov.fit_state_winner_matches_final_params));
        s.push('\n');
    }
    s
}

/// Render a `FitSummaryDoc` as LaTeX `tabular` blocks per stage.
/// One section per stage with three tables: gate verdict, parameters,
/// per-chain loglik-eval. No preamble — the caller should embed inside
/// an existing document.
pub fn render_latex(doc: &FitSummaryDoc) -> String {
    let mut s = String::new();
    s.push_str(&format!("% camdl fit summary: {}\n", escape_latex(&doc.fit_dir)));
    s.push_str(&format!("% camdl {} schema v{}\n\n",
        doc.schema.camdl_version, doc.schema.version));
    if let Some(cs) = &doc.chain_selection {
        s.push_str(&format!(
            "% chain subset: diagnostics over {} of {} chains (excluded {}); \
             post-hoc exclusion biases toward the retained mode\n\n",
            cs["kept"].as_array().map(|a| a.len()).unwrap_or(0),
            cs["n_total"].as_u64().unwrap_or(0),
            render_id_csv(&cs["excluded"]),
        ));
    }
    for stage in &doc.stages {
        s.push_str(&render_latex_stage(stage));
    }
    s
}

/// Render a JSON array of chain ids as a `"3,5"` CSV for the header/provenance
/// lines. Empty array → empty string.
fn render_id_csv(v: &serde_json::Value) -> String {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64())
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default()
}

fn render_latex_stage(stage: &StageReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "\\subsection*{{Stage: \\texttt{{{}}} ({})}}\n\n",
        escape_latex(&stage.name),
        stage.method
    ));

    if let Some(ll) = stage.best_loglik {
        s.push_str(&format!(
            "Best log-likelihood: \\textbf{{{:.2}}} ({}); chains: {}\n\n",
            ll,
            crate::fit::loglik::LoglikType::tag_or_unknown(stage.loglik_type),
            stage.n_chains
        ));
    } else {
        s.push_str(&format!("chains: {}\n\n", stage.n_chains));
    }

    if let Some(gate) = &stage.gate {
        s.push_str("\\begin{tabular}{lrrl}\n");
        s.push_str("\\toprule\n");
        s.push_str("Leg & Value & Threshold & Pass? \\\\\n");
        s.push_str("\\midrule\n");
        let glyph = |b| if b { "$\\checkmark$" } else { "$\\times$" };
        s.push_str(&format!(
            "$\\hat A$ (max) & {:.3} & {:.2} & {} \\\\\n",
            gate.max_a_hat, gate.a_thresh, glyph(gate.a_passes)
        ));
        if let (Some(dd), Some(td), Some(p)) = (gate.delta_db, gate.threshold_db, gate.db_passes) {
            s.push_str(&format!(
                "Decibans-spread & {:.1} dB & {:.1} dB & {} \\\\\n",
                dd, td, glyph(p)
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }

    if !stage.parameters.is_empty() {
        s.push_str("\\begin{tabular}{lrrl}\n\\toprule\n");
        s.push_str("Parameter & Estimate & $\\hat A$ & Flags \\\\\n\\midrule\n");
        for p in &stage.parameters {
            let a_str = p
                .chain_agreement
                .map(|r| format!("{:.3}", r))
                .unwrap_or_else(|| "---".into());
            let flag = if p.perturb_only_at_t0 { "t0-only" } else { "" };
            let est_cell = match &p.estimate_date {
                Some(date) => format!("{:.6} ({})", p.estimate, date),
                None => format!("{:.6}", p.estimate),
            };
            s.push_str(&format!(
                "\\texttt{{{}}} & {} & {} & {} \\\\\n",
                escape_latex(&p.name),
                est_cell,
                a_str,
                flag
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }

    if !stage.chains.is_empty() {
        // Column / marker naming matches the text + md emitters: "loglik"
        // / "Selected", not "clean_ll" / "Winner".
        s.push_str("\\begin{tabular}{rrrc}\n\\toprule\n");
        s.push_str("Chain & loglik & $\\pm$ se & Selected \\\\\n\\midrule\n");
        for c in &stage.chains {
            let mark = if c.is_winner { "$\\star$" } else { "" };
            s.push_str(&format!(
                "{} & {:.2} & {:.2} & {} \\\\\n",
                c.chain_id, c.clean_loglik, c.clean_se, mark
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }

    // PGAS / PMMH posterior block.
    if let Some(MethodResult::Pgas(p)) = &stage.method_result {
        s.push_str(&format!(
            "Posterior summary (max $\\hat R$ = {}):\n\n",
            max_rhat_cell(&p.diagnostics)
        ));
        s.push_str("\\begin{tabular}{lrrrrrr}\n\\toprule\n");
        s.push_str("Parameter & Mean & $q_{0.025}$ & $q_{0.975}$ & ESS bulk & ESS tail & $\\hat R$ \\\\\n\\midrule\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "---".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "---".into());
            let ess = p.diagnostics.ess_cell(name, "---");
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!(
                "\\texttt{{{}}} & {} & {} & {} & {} & {} & {} \\\\\n",
                escape_latex(name), mean_cell, q025, q975, ess,
                p.diagnostics.ess_tail_cell(name, "---"),
                p.diagnostics.rhat_cell(name, "---"),
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    } else if let Some(MethodResult::Pmmh(p)) = &stage.method_result {
        s.push_str(&format!(
            "Posterior summary (max $\\hat R$ = {}; acceptance = {:.3}):\n\n",
            max_rhat_cell(&p.diagnostics), p.acceptance_rate
        ));
        s.push_str("\\begin{tabular}{lrrrr}\n\\toprule\n");
        s.push_str("Parameter & Mean & ESS bulk & ESS tail & $\\hat R$ \\\\\n\\midrule\n");
        for (name, mean) in &p.posterior_mean {
            let ess = p.diagnostics.ess_cell(name, "---");
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!(
                "\\texttt{{{}}} & {} & {} & {} & {} \\\\\n",
                escape_latex(name), mean_cell, ess,
                p.diagnostics.ess_tail_cell(name, "---"),
                p.diagnostics.rhat_cell(name, "---"),
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    } else if let Some(MethodResult::Nuts(p)) = &stage.method_result {
        s.push_str(&format!(
            "Posterior summary (max $\\hat R$ = {}; divergences = {}):\n\n",
            max_rhat_cell(&p.diagnostics), p.n_divergent
        ));
        s.push_str("\\begin{tabular}{lrrrrrr}\n\\toprule\n");
        s.push_str("Parameter & Mean & $q_{0.025}$ & $q_{0.975}$ & ESS bulk & ESS tail & $\\hat R$ \\\\\n\\midrule\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "---".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "---".into());
            let ess = p.diagnostics.ess_cell(name, "---");
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!(
                "\\texttt{{{}}} & {} & {} & {} & {} & {} & {} \\\\\n",
                escape_latex(name), mean_cell, q025, q975, ess,
                p.diagnostics.ess_tail_cell(name, "---"),
                p.diagnostics.rhat_cell(name, "---"),
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }

    s
}

/// Escape LaTeX-active characters in identifiers / paths. Minimal —
/// we don't escape `_` inside `\texttt{}` because LaTeX renders
/// `\texttt{foo_bar}` literally (the `_` in `\texttt` is allowed in
/// most modern LaTeX engines), but we replace `&`, `%`, `#`, `$`.
fn escape_latex(s: &str) -> String {
    s.chars().map(|c| match c {
        '&' => "\\&".into(),
        '%' => "\\%".into(),
        '#' => "\\#".into(),
        '$' => "\\$".into(),
        '_' => "\\_".into(),
        '{' => "\\{".into(),
        '}' => "\\}".into(),
        c   => c.to_string(),
    }).collect()
}

/// The winning stage's point estimate (θ̂) as a flat params TOML — the same
/// payload `fit summary --params-only` prints. A `pub(crate)` seam over
/// [`discover_stages`] + [`dump_params_only`] so `compare` can derive a
/// prequential at θ̂ from a sealed fit without touching the private
/// `ResolvedStage` type. `stage` selects a stage; `None` = the terminal stage.
pub(crate) fn winner_params_toml(
    segment: &Path,
    stage: Option<&str>,
) -> Result<String, String> {
    let dir = segment.to_string_lossy();
    let discovered = discover_stages(segment);
    dump_params_only(&dir, stage, &discovered)
}

/// Dump the chosen stage's winner params as a flat TOML, pipeable
/// into `camdl pfilter --params`. No header, no metadata, no
/// provenance — just `name = value` lines the standard params loader
/// will accept.
///
/// `stage_filter` selects an explicit stage (must be present in
/// `discovered`); when `None`, picks the *terminal* stage in
/// declaration order (`FitView.stages_declared` walked in reverse).
fn dump_params_only(
    dir: &str,
    stage_filter: Option<&str>,
    discovered: &[ResolvedStage],
) -> Result<String, String> {
    let target = match stage_filter {
        Some(name) => discovered
            .iter()
            .find(|r| r.stage == name)
            .cloned()
            .ok_or_else(|| {
                let avail: Vec<&str> = discovered.iter().map(|r| r.stage.as_str()).collect();
                format!(
                    "no completed `{}` stage found under {}. Available: {}",
                    name,
                    dir,
                    if avail.is_empty() {
                        "(none)".to_string()
                    } else {
                        avail.join(", ")
                    }
                )
            })?,
        None => discovered
            .iter()
            .next_back()
            .cloned()
            .ok_or_else(|| format!("no completed fit-stage runs found in {}", dir))?,
    };
    let target_stage = target.stage.clone();
    let stage_path = target.stage_dir.clone();
    let path = format!("{}/final_params.toml", stage_path.to_string_lossy());
    let params = crate::util::load_params_toml(&path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    let mut out = String::new();
    out.push_str(&format!("# camdl fit summary --params-only --stage {}\n", target_stage));
    out.push_str(&format!("# source: {}\n", path));
    out.push_str(&format!("# camdl: {}\n\n", version::VERSION_SHORT));
    for k in keys {
        let v = params[k];
        // Emit integers without a decimal so the loader returns the
        // expected value. format_param_value already handles this.
        out.push_str(&format!("{} = {}\n", k, crate::fit::runner::format_param_value(v)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};
    use crate::fit::loglik::LoglikType;
    use crate::fit::method_result::PosteriorDiagnostics;

    fn synthetic_fit_state() -> FitState {
        let mut start = std::collections::BTreeMap::new();
        start.insert("R0".into(),  56.0);
        start.insert("sigma".into(), 0.08);
        start.insert("gamma".into(), 0.08);
        let mut agreement = std::collections::BTreeMap::new();
        agreement.insert("R0".into(),    1.04);
        agreement.insert("sigma".into(), 1.01);
        agreement.insert("gamma".into(), 1.21);
        FitState {
            stage: "scout".into(),
            seed: 42,
            timestamp: "2026-04-25T00:00:00Z".into(),
            input_hash: Some("deadbeef".into()),
            camdl_version: Some(version::VERSION_SHORT.into()),
            best_loglik: -3804.9,
            initial_loglik: -7891.0,
            best_chain: 1,
            n_chains: 8,
            n_good_chains: Some(8),
            start_values: start,
            rw_sd: Default::default(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: agreement,
            perturb_only_at_t0_params: vec!["I0".into()],
            chain_logliks: vec![-3810.0; 8],
            chain_eval_logliks: vec![
                -3810.5, -3805.1, -3812.0, -3808.7,
                -3804.9, -3811.2, -3809.0, -3807.6,
            ],
            chain_eval_ses: vec![1.5, 1.2, 1.8, 1.4, 1.1, 1.6, 1.3, 1.5],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            dt_check: None,
            pf_noise: None,
        }
    }

    #[test]
    fn formatter_renders_pass_verdict_when_thresholds_clear() {
        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let block = fmt.gate_verdict_block(&state);

        // Â leg: max = 1.21 on gamma, threshold 1.01 → fail.
        // Spread is small (~7 nats) → decibans leg passes.
        // Overall: FAIL because Â leg fails.
        assert!(block.contains("Â leg:"));
        assert!(block.contains("max Â = 1.210 (gamma)"),
            "expected max Â call-out; got: {}", block);
        assert!(block.contains("decibans leg:"));
        assert!(block.contains("overall:"));
    }

    #[test]
    fn formatter_emits_caveat_when_resolved_gate_absent() {
        let mut state = synthetic_fit_state();
        state.resolved_gate = None;
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let block = fmt.gate_verdict_block(&state);
        assert!(block.contains("thresholds unknown"),
            "legacy fit_state without resolved_gate must surface caveat; got: {}",
            block);
    }

    /// One fit, one number, two glyphs.
    ///
    /// `gate_verdict_block` compares Â to the CONFIGURED `a_thresh` (default
    /// 1.01, and the value `check_scout_convergence` actually refuses on);
    /// `parameter_table` compared it to a bare literal 1.05. On this fixture
    /// R0's Â is 1.040 — above the gate, below the literal — so the gate block
    /// printed `max Â = 1.210 ✗ (threshold 1.01)` and the parameter table
    /// twenty lines below printed `Â=1.040 ✓` for a parameter of the same fit
    /// that the same gate refuses.
    #[test]
    fn the_parameter_table_glyph_agrees_with_the_gate_printed_above_it() {
        let state = synthetic_fit_state();
        let gate = state.resolved_gate.clone().expect("fixture resolves its gate");
        assert!(
            gate.a_thresh < 1.04,
            "fixture premise: Â = 1.040 must FAIL this gate ({})",
            gate.a_thresh
        );
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.parameter_table(&state);
        let r0 = table.lines().find(|l| l.contains("R0")).expect("an R0 row");
        assert!(
            !r0.contains('✓'),
            "R0 (Â=1.040) is refused by the a_thresh={} gate, so its row must \
             not print ✓:\n{r0}\nfull table:\n{table}",
            gate.a_thresh
        );
    }

    /// A parameter whose within-chain variance collapsed has NO Â — the G-R
    /// formula divides by it, so `compute_chain_agreement` returns NaN and the
    /// end-of-stage block prints "n/a (W ≈ 0; rely on Δ_dB)". The summary
    /// table compared that NaN to a literal, and every comparison against NaN
    /// is false, so it fell through to ✗ — reporting "this parameter failed"
    /// for a parameter that was never assessed.
    #[test]
    fn a_parameter_with_no_agreement_is_not_reported_as_failing() {
        let mut state = synthetic_fit_state();
        state.tail_chain_agreement.insert("sigma".into(), f64::NAN);
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.parameter_table(&state);
        let sigma = table.lines().find(|l| l.contains("sigma")).expect("a sigma row");
        assert!(
            !sigma.contains('✗'),
            "an unassessable Â is not a failure:\n{sigma}"
        );
        assert!(
            sigma.contains("n/a"),
            "and it must say so, naming why:\n{sigma}"
        );
    }

    #[test]
    fn parameter_table_filters_to_estimated_params() {
        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let block = fmt.parameter_table(&state);
        assert!(block.contains("R0"), "R0 row missing: {}", block);
        assert!(block.contains("Â=1.040"), "R0 agreement missing: {}", block);
        assert!(block.contains("sigma"));
        assert!(block.contains("gamma"));
    }

    #[test]
    fn ci_env_strict_auto_enable() {
        // Sanity check on the gate that triggers --strict from CI=true.
        // We can't toggle env vars in a thread-safe way during cargo
        // test, so just verify the helper reads the right values.
        std::env::remove_var("CI");
        assert!(!ci_env_set());
        std::env::set_var("CI", "true");
        assert!(ci_env_set());
        std::env::set_var("CI", "1");
        assert!(ci_env_set());
        std::env::set_var("CI", "false");
        assert!(!ci_env_set());
        std::env::remove_var("CI");
    }

    /// Provenance cross-check is the always-on diagnostic that turns
    /// the GH #16 silent-wrong-answer mode into a visible ✗ on every
    /// read. Test: write a fit dir where final_params.toml and
    /// mle_params.toml carry different R0 values; assert provenance
    /// block flags it.
    #[test]
    fn provenance_block_detects_mle_final_disagreement() {
        let dir = crate::test_support::unique_temp_dir("summary_prov");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("final_params.toml"),
            "R0 = 56.82\nsigma = 0.0791\n").unwrap();
        std::fs::write(dir.join("mle_params.toml"),
            "R0 = 81.45\nsigma = 0.0791\n").unwrap();

        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let prov = fmt.provenance_block(&dir.to_string_lossy(), &state);
        assert!(prov.failed, "must flag the disagreement: {}", prov.text);
        assert!(prov.text.contains("DISAGREE"),
            "must call out DISAGREE: {}", prov.text);
        assert!(prov.text.contains("#16"),
            "must reference the GH issue this guards against: {}",
            prov.text);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The withholding message described a mechanism gh#299 removed. It told
    /// the reader their parameters had "no pooled ESS — their chains disagree
    /// (R̂ above the pooling threshold), so per-chain ESS cannot be summed into
    /// an effective N". None of that is how camdl computes ESS any more: bulk
    /// ESS is the rank-normalized cross-chain statistic of Vehtari et al.
    /// (2021), it uses the between-chain variance instead of summing per-chain
    /// estimates, and no R̂ gate suppresses it. A reader who acted on that
    /// sentence would go looking for a threshold that does not exist.
    #[test]
    fn the_withholding_message_does_not_describe_a_retired_mechanism() {
        use std::collections::BTreeMap;
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let no_traces = std::path::Path::new("/nonexistent/stage_dir");
        let r = PgasStageResult {
            diagnostics: PosteriorDiagnostics {
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([("a2".to_string(), 1.01), ("tau".to_string(), 2.639)]),
                    BTreeMap::from([("a2".to_string(), 145.0)]),
                    BTreeMap::new(),
                ),
                n_samples: 500,
                thin: 1,
                wall_time_secs: Some(11.8),
                n_chains: 4,
            },
            posterior_mean: BTreeMap::from([("a2".to_string(), 0.5), ("tau".to_string(), 1.0)]),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        };
        let out = fmt.bayesian_block(
            "pgas", "pgas", no_traces, BayesianView::Pgas(&r), None,
            crate::fit::loglik::LoglikType::CompleteData,
        );
        assert!(
            !out.contains("pooling threshold"),
            "no R̂ gate suppresses bulk ESS — gh#299 removed it:\n{out}"
        );
        assert!(
            !out.contains("summed"),
            "bulk ESS is not a sum of per-chain estimates; it uses the \
             between-chain variance:\n{out}"
        );
        // It must still name the parameters that withhold the headline.
        assert!(out.contains("tau"), "and must still name who withholds it:\n{out}");
    }

    /// gh#687: when a parameter reports no bulk ESS, the minimum over the
    /// parameters that DID report rises as the fit gets worse. The block must
    /// print no efficiency number at all in that state,
    /// and must name the parameters that withhold it — the blank is the
    /// diagnosis. The control leg (a complete map) proves the withholding is
    /// conditional, not a renderer that lost its numbers.
    #[test]
    fn bayesian_block_withholds_efficiency_and_names_params_without_ess() {
        use std::collections::BTreeMap;
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let no_traces = std::path::Path::new("/nonexistent/stage_dir");
        let mk = |ess: BTreeMap<String, f64>| PgasStageResult {
            diagnostics: PosteriorDiagnostics {
                // Both assessed across chains; `tau` carries no bulk ESS,
                // the state this block has to withhold the headline for.
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([("a2".to_string(), 1.01), ("tau".to_string(), 2.639)]),
                    ess,
                    BTreeMap::new(),
                ),
                n_samples: 500,
                thin: 1,
                wall_time_secs: Some(11.8),
                n_chains: 4,
            },
            posterior_mean: BTreeMap::from([("a2".to_string(), 0.5), ("tau".to_string(), 1.0)]),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        };

        // `tau` reports no ESS → no efficiency headline, and `tau` is named.
        let gapped = mk(BTreeMap::from([("a2".to_string(), 145.0)]));
        let out = fmt.bayesian_block("posterior", "pgas", no_traces, BayesianView::Pgas(&gapped), None, LoglikType::CompleteData);
        assert!(
            !out.contains("min-param ESS"),
            "no min-param ESS may be printed over the reporting subset: {out}"
        );
        assert!(
            !out.contains("ESS/iter = 0.") && !out.contains("ESS/sec  = "),
            "neither efficiency ratio may carry a number here: {out}"
        );
        assert!(out.contains("ESS/iter = —"), "the withheld metric is shown as a dash: {out}");
        assert!(
            out.contains("1 of 2 parameters report no bulk ESS"),
            "the count of parameters withholding the metric is stated: {out}"
        );
        assert!(
            out.contains("\n        tau\n"),
            "the withholding parameter is named on its own line, not left to the \
             table's dashes: {out}"
        );

        // Control: with `tau`'s ESS present the same block reports both ratios —
        // 145 / 500 raw iters and 145 / 11.8 s off the slower of the two.
        let complete = mk(BTreeMap::from([("a2".to_string(), 145.0), ("tau".to_string(), 300.0)]));
        let ok = fmt.bayesian_block("posterior", "pgas", no_traces, BayesianView::Pgas(&complete), None, LoglikType::CompleteData);
        assert!(
            ok.contains("ESS/iter = 0.290  (min-param ESS 145 / 500 raw sampling iters)"),
            "a complete map still reports ESS/iter off the slowest param: {ok}"
        );
        assert!(
            ok.contains("ESS/sec  = 12.29"),
            "a complete map still reports ESS/sec: {ok}"
        );
        assert!(
            !ok.contains("not reportable"),
            "the withholding branch must not fire on a complete map: {ok}"
        );
    }

    /// gh#611. When a fit fails, `max R̂ = 6.571 ✗` says THAT it failed; the
    /// per-parameter table is where a reader finds out WHICH parameter — which
    /// is the whole diagnostic question at that moment. The table has an R̂
    /// column and an ESS column and `diagnostics.json` carries both values per
    /// parameter, but the R̂ cell was hard-coded empty, so every row read as a
    /// dash and the answer had to be recovered by parsing the JSON by hand.
    #[test]
    fn bayesian_block_fills_the_per_param_rhat_and_ess_columns() {
        use std::collections::BTreeMap;
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let no_traces = std::path::Path::new("/nonexistent/stage_dir");
        let r = PgasStageResult {
            diagnostics: PosteriorDiagnostics {
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([
                    ("a2".to_string(), 1.013),
                    ("tau".to_string(), 6.571),
                ]),
                    BTreeMap::from([
                    ("a2".to_string(), 145.0),
                    ("tau".to_string(), 42.0),
                    ("phi".to_string(), f64::NAN),
                ]),
                    BTreeMap::from([
                    ("a2".to_string(), 268.0),
                    ("tau".to_string(), f64::NAN),
                ]),
                ),
                n_samples: 500,
                thin: 1,
                wall_time_secs: Some(11.8),
                n_chains: 4,
            },
            posterior_mean: BTreeMap::from([
                ("a2".to_string(), 0.5),
                ("tau".to_string(), 1.0),
                ("phi".to_string(), 0.25),
            ]),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        };
        let out = fmt.bayesian_block(
            "posterior", "pgas", no_traces, BayesianView::Pgas(&r), None,
            LoglikType::CompleteData,
        );
        let row = |name: &str| -> String {
            out.lines()
                .find(|l| l.trim_start().starts_with(&format!("{name} ")))
                .unwrap_or_else(|| panic!("no `{name}` row in the table:\n{out}"))
                .to_string()
        };
        let tau = row("tau");
        assert!(tau.contains("6.571"),
            "the row for the parameter that failed must carry its R̂: {tau}");
        assert!(tau.contains("42"),
            "the row for the parameter that failed must carry its ESS: {tau}");
        // gh#691 item 2: the `--exclude-chains` recompute KEEPS the key and
        // stores NaN where the loaded path drops it, so the per-parameter cell
        // must render both encodings the same way. `phi` here is the
        // present-but-NaN form.
        let phi = row("phi");
        assert!(!phi.contains("NaN"),
            "an absent ESS is a dash, never the literal NaN: {phi}");
        let a2 = row("a2");
        assert!(a2.contains("1.013"), "every assessed parameter carries its R̂: {a2}");
        assert!(a2.contains("145"), "every assessed parameter carries its ESS: {a2}");
        assert!(a2.contains("268"),
            "bulk ESS alone does not say whether the interval endpoints mixed; \
             the tail column does: {a2}");
        assert!(!tau.contains("NaN"),
            "an undefined tail-ESS is a dash, never the literal NaN: {tau}");
    }

    #[test]
    fn bayesian_block_reports_ess_per_second_off_the_slowest_param() {
        use std::collections::BTreeMap;
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let mk = |wall: Option<f64>, n_samples: usize, thin: usize| PgasStageResult {
            diagnostics: PosteriorDiagnostics {
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([("a2".to_string(), 1.01), ("g".to_string(), 1.00)]),
                    BTreeMap::from([("a2".to_string(), 145.0), ("g".to_string(), 300.0)]),
                    BTreeMap::new(),
                ),
                n_samples,
                thin,
                wall_time_secs: wall,
                n_chains: 4,
            },
            posterior_mean: BTreeMap::from([("a2".to_string(), 0.5), ("g".to_string(), 1.0)]),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        };
        // No chain traces on disk for this in-memory result → the per-chain
        // table degrades to "unavailable"; the ESS lines under test are unaffected.
        let no_traces = std::path::Path::new("/nonexistent/stage_dir");
        // min-param ESS (145) / wall (11.8 s) = 12.29 ESS/sec — thinning-invariant.
        let with = fmt.bayesian_block("posterior", "pgas", no_traces, BayesianView::Pgas(&mk(Some(11.8), 500, 1)), None, LoglikType::CompleteData);
        assert!(
            with.contains("ESS/sec  = 12.29"),
            "must report ESS/sec off the slowest param (145/11.8): {with}"
        );
        // ESS/iteration = 145 / (n_samples 500 × thin 1) = 0.290, per raw sampling step.
        assert!(
            with.contains("ESS/iter = 0.290"),
            "must report ESS/iteration off raw sampling steps (145/500): {with}"
        );
        // Thinning-invariance: (n_samples 50 × thin 10) is the SAME 500 raw steps,
        // so ESS/iter is identical — the whole point.
        let thinned = fmt.bayesian_block("posterior", "pgas", no_traces, BayesianView::Pgas(&mk(Some(5.0), 50, 10)), None, LoglikType::CompleteData);
        assert!(
            thinned.contains("ESS/iter = 0.290"),
            "ESS/iter must be invariant to thinning (50×10 == 500 raw): {thinned}"
        );
        // No wall-time (older run) → no ESS/sec line, but ESS/iter still shows
        // (it needs only n_samples×thin, not wall-time).
        let without = fmt.bayesian_block("posterior", "pgas", no_traces, BayesianView::Pgas(&mk(None, 500, 1)), None, LoglikType::CompleteData);
        assert!(
            !without.contains("ESS/sec"),
            "no wall-time must omit the ESS/sec line: {without}"
        );
        assert!(
            without.contains("ESS/iter = 0.290"),
            "ESS/iter does not need wall-time and must still show: {without}"
        );
    }

    /// gh#406: the per-chain loglik table names the stuck chain in a Bayesian
    /// summary. Six-chain stage dir (five near -50, chain 6 stuck at -300) with
    /// per-chain traces + a draws.tsv manifest → the table flags chain 6 and the
    /// nudge fires.
    #[test]
    fn bayesian_chain_loglik_table_names_the_stuck_chain() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_diag");
        std::fs::create_dir_all(&dir).unwrap();
        let write_trace = |c: usize, kept: &str| {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            // Two warm-up rows (must be stripped by the last-K_c rule) + 3 kept.
            let body = format!(
                "step\tlog_likelihood\tlog_posterior\n1\t-900.0\t-905.0\n2\t-880.0\t-885.0\n{kept}");
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        };
        // Good chains carry realistic jitter (distinct means ≈ -50) so the robust
        // MAD is non-zero; chain 6 is the lone stuck chain at ≈ -300.
        write_trace(1, "3\t-50.0\t-52.0\n4\t-50.0\t-52.0\n5\t-50.0\t-52.0\n"); // mean -50.0
        write_trace(2, "3\t-50.5\t-52.0\n4\t-50.5\t-52.0\n5\t-50.5\t-52.0\n"); // mean -50.5
        write_trace(3, "3\t-49.5\t-51.0\n4\t-49.5\t-51.0\n5\t-49.5\t-51.0\n"); // mean -49.5
        write_trace(4, "3\t-50.2\t-52.0\n4\t-50.2\t-52.0\n5\t-50.2\t-52.0\n"); // mean -50.2
        write_trace(5, "3\t-49.8\t-51.0\n4\t-49.8\t-51.0\n5\t-49.8\t-51.0\n"); // mean -49.8
        write_trace(6, "3\t-300.0\t-302.0\n4\t-301.0\t-303.0\n5\t-299.0\t-301.0\n"); // stuck -300
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 6, LoglikType::Marginal);
        assert!(table.contains("per-chain log-likelihood"), "header present:\n{table}");
        assert!(table.contains("← outlier"), "stuck chain must be flagged:\n{table}");
        assert!(table.contains("chains disagree (chain 6"), "nudge must name chain 6:\n{table}");
        // The stuck chain's mean must reflect the post-burn-in draws (≈ -300),
        // not the stripped warm-up (≈ -900).
        assert!(table.contains("-300.00"), "chain 6 mean ≈ -300 (warm-up stripped):\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667: a PGAS stage's per-chain table ranks on `obs_ll`
    /// (`log p(y | X, θ)`), not on the `log_complete_data_ll` target. The
    /// fixture is built so the two rankings disagree: chain 3 has a hugely
    /// concentrated latent path (a high complete-data value) with an ordinary
    /// data fit, chain 6 fits the data ≈450 nats worse with an ordinary
    /// complete-data value. The table must flag chain 6 and leave chain 3 clean,
    /// while still SHOWING the complete-data target and its transition term.
    #[test]
    fn pgas_chain_table_ranks_on_obs_ll_and_shows_the_split() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_gh667");
        std::fs::create_dir_all(&dir).unwrap();
        // (transition_ll, obs_ll); complete = transition + obs.
        let chains = [
            (-2832.6, -952.0),
            (-2933.0, -952.8),
            (-800.0, -951.7),
            (-3100.9, -952.9),
            (-3172.6, -952.3),
            (-3000.0, -1400.0),
        ];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{}", i + 1));
            std::fs::create_dir_all(&cd).unwrap();
            let complete = trans + obs;
            let mut body = String::from(
                "sweep\tlog_complete_data_ll\tlog_posterior\ttransition_ll\tobs_ll\n");
            for s in 0..2 {
                body.push_str(&format!("{s}\t-9000\t-9100\t-8000\t-1000\n"));
            }
            for s in 2..5 {
                body.push_str(&format!("{s}\t{complete}\t{}\t{trans}\t{obs}\n", complete - 1.0));
            }
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        }
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.{c}{d}\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 6, LoglikType::CompleteData);

        // The flag lands on the chain that reproduces the data worst…
        let flagged: Vec<&str> = table
            .lines()
            .filter(|l| l.contains("← outlier"))
            .map(|l| l.trim_start().split_whitespace().next().unwrap_or(""))
            .collect();
        assert_eq!(flagged, vec!["6"],
            "only chain 6 (the bad DATA fit) may be flagged:\n{table}");
        assert!(table.contains("chains disagree (chain 6"),
            "the nudge must name chain 6:\n{table}");

        // …and the ranked column is obs_ll, not the complete-data target.
        assert!(table.contains("-1400.00"),
            "chain 6's scored value is its obs_ll (-1400.00):\n{table}");
        assert!(table.contains("-952.00"),
            "chain 1's scored value is its obs_ll (-952.00):\n{table}");

        // The complete-data target and its latent-path term stay VISIBLE — they
        // are the sampler's own objective, just not the ranking key.
        assert!(table.contains("-4400.00"),
            "chain 6's complete-data target must still be shown:\n{table}");
        assert!(table.contains("-1751.70"),
            "chain 3's complete-data target must still be shown:\n{table}");
        assert!(table.contains("-800.00"),
            "chain 3's transition_ll must still be shown:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667 must not cost gh#608/gh#635 their reach. A PGAS stage whose
    /// traces predate the `obs_ll` column cannot be compared chain-to-chain —
    /// the table says so by name rather than falling back to the position-1
    /// column — but the degeneracy screens read `log_posterior` and
    /// `draws.tsv`, so a −inf chain must still be named.
    #[test]
    fn missing_obs_ll_still_runs_the_degeneracy_screen() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_no_obs");
        std::fs::create_dir_all(&dir).unwrap();
        for c in 1..=3 {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            // No obs_ll / transition_ll columns at all. Chain 3 is degenerate.
            let rows = if c == 3 {
                "0\t-77.0\t-inf\n1\t-77.0\t-inf\n"
            } else {
                "0\t-77.0\t-79.0\n1\t-77.0\t-79.0\n"
            };
            std::fs::write(cd.join("trace.tsv"),
                format!("sweep\tlog_complete_data_ll\tlog_posterior\n{rows}")).unwrap();
        }
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..3 {
            for d in 0..2 {
                draws.push_str(&format!("{c}\t{d}\t0.{c}{d}\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 3, LoglikType::CompleteData);
        assert!(table.contains("no `obs_ll` column"),
            "the missing comparison column is named, not silently substituted:\n{table}");
        assert!(!table.contains("← outlier"),
            "nothing may be ranked without the comparison column:\n{table}");
        assert!(table.contains("DEGENERATE") && table.contains("chain 3"),
            "the -inf screen still names chain 3:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#608: a chain recording -inf log-posterior as its current state is
    /// flagged DEGENERATE next to the per-chain table — with the count, the
    /// contamination statement, and the explicit exclusion fix — and its mean
    /// renders as -inf, never the "missing data" em-dash.
    #[test]
    fn bayesian_chain_table_flags_neginf_chain_loudly() {
        let dir = crate::test_support::unique_temp_dir("summary_neginf_chain");
        std::fs::create_dir_all(&dir).unwrap();
        let write_trace = |c: usize, kept: &str| {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            let body = format!(
                "step\tlog_likelihood\tlog_posterior\n1\t-900.0\t-905.0\n{kept}");
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        };
        write_trace(1, "2\t-50.0\t-52.0\n3\t-50.5\t-52.5\n4\t-49.5\t-51.5\n");
        write_trace(2, "2\t-50.2\t-52.2\n3\t-50.1\t-52.1\n4\t-49.9\t-51.9\n");
        // Chain 3: stuck — two of three retained rows carry -inf as the
        // CURRENT state (the gh#607 shape).
        write_trace(3, "2\t-inf\t-inf\n3\t-inf\t-inf\n4\t-60.0\t-62.0\n");
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..3 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 3, LoglikType::Marginal);
        assert!(table.contains("-inf"),
            "the stuck chain's mean renders -inf, not the missing-data dash:\n{table}");
        assert!(table.contains("DEGENERATE"),
            "the stuck chain gets the loud flag:\n{table}");
        assert!(table.contains("chain 3") && table.contains("66.7%")
                && table.contains("(2/3)"),
            "the flag names the chain and the -inf fraction:\n{table}");
        assert!(table.contains("--exclude-chains"),
            "the flag names the explicit exclusion fix:\n{table}");
        assert!(table.contains("draws.tsv"),
            "the flag states the contamination (draws are pooled):\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#608 negative control: healthy chains draw no DEGENERATE flag.
    #[test]
    fn bayesian_chain_table_no_degenerate_flag_when_healthy() {
        let dir = crate::test_support::unique_temp_dir("summary_no_neginf");
        std::fs::create_dir_all(&dir).unwrap();
        for c in 1..=3 {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            std::fs::write(cd.join("trace.tsv"), format!(
                "step\tlog_likelihood\tlog_posterior\n1\t-5{c}.0\t-5{c}.5\n2\t-5{c}.1\t-5{c}.6\n"))
                .unwrap();
        }
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 3, LoglikType::Marginal);
        assert!(!table.contains("DEGENERATE"),
            "healthy chains must not be flagged:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#635: a chain with ONE distinct parameter vector across its retained
    /// draws (zero accepted moves, finite density — evades the −inf screen)
    /// draws the loud point-mass flag; mixing chains do not.
    #[test]
    fn bayesian_chain_table_flags_point_mass_chain() {
        let dir = crate::test_support::unique_temp_dir("summary_point_mass");
        std::fs::create_dir_all(&dir).unwrap();
        for c in 1..=3 {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            std::fs::write(cd.join("trace.tsv"), format!(
                "step\tlog_likelihood\tlog_posterior\n1\t-5{c}.0\t-5{c}.5\n2\t-5{c}.1\t-5{c}.6\n3\t-5{c}.2\t-5{c}.7\n"))
                .unwrap();
        }
        // Chains 0 and 1 mix (distinct vectors); chain 2 is frozen at one θ.
        let mut draws = String::from("chain\tdraw\tbeta\tgamma\n");
        for d in 0..3 {
            draws.push_str(&format!("0\t{d}\t0.{d}1\t0.2\n"));
            draws.push_str(&format!("1\t{d}\t0.{d}3\t0.2\n"));
            draws.push_str(&format!("2\t{d}\t0.55\t0.20\n")); // identical every draw
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 3, LoglikType::Marginal);
        assert!(table.contains("point-mass"),
            "the frozen chain gets the loud flag:\n{table}");
        assert!(table.contains("chain 3") && table.contains("across 3 retained"),
            "the flag names the chain and the draw count:\n{table}");
        assert!(table.contains("--exclude-chains"),
            "the flag names the explicit fix:\n{table}");
        // Exactly one flagged chain — mixing chains stay clean.
        assert_eq!(table.matches("point-mass").count(), 1,
            "only the frozen chain is flagged:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#406 negative control: a well-mixed six-chain stage flags no outlier and
    /// prints no "disagree" nudge — not a vacuous pass (the table still renders
    /// every chain with a finite z).
    #[test]
    fn bayesian_chain_loglik_table_clean_when_well_mixed() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_diag_clean");
        std::fs::create_dir_all(&dir).unwrap();
        for c in 1..=6 {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            // All chains ≈ -50, tiny spread.
            let jitter = (c as f64) * 0.1;
            std::fs::write(
                cd.join("trace.tsv"),
                format!("step\tlog_likelihood\tlog_posterior\n0\t{:.2}\t-52.0\n1\t{:.2}\t-52.0\n",
                    -50.0 + jitter, -50.1 + jitter),
            )
            .unwrap();
        }
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..2 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 6, LoglikType::Marginal);
        assert!(table.contains("per-chain log-likelihood"), "header present:\n{table}");
        assert!(!table.contains("← outlier"), "well-mixed must flag nothing:\n{table}");
        assert!(!table.contains("chains disagree"), "no nudge when well-mixed:\n{table}");
        // Not vacuous: all six chain rows render (each data row starts, after
        // its indent, with the chain number; the header row starts with "chain").
        let data_rows = table
            .lines()
            .filter(|l| l.trim_start().chars().next().is_some_and(|c| c.is_ascii_digit()))
            .count();
        assert_eq!(data_rows, 6, "six per-chain rows must render:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#406: a stage with no per-chain traces says so rather than silently
    /// omitting the section (no-silent-gap).
    #[test]
    fn bayesian_chain_loglik_table_reports_unavailable() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_diag_none");
        std::fs::create_dir_all(&dir).unwrap();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let table = fmt.bayesian_chain_loglik_table(&dir, 4, LoglikType::Marginal);
        assert!(table.contains("per-chain traces unavailable"),
            "must say traces are unavailable, not skip:\n{table}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── gh#727: saved-vs-forkable latent paths in the summary ────────────

    /// Write a `pgas_summary.json` carrying just the gh#727 `trajectories`
    /// block — the only key `saved_path_table` reads.
    fn write_traj_summary(
        dir: &std::path::Path,
        draw_stride: u64,
        thin: u64,
        per_chain: &[(u64, u64)],
    ) {
        let summary = serde_json::json!({
            "stage": "pgas",
            "thin": thin,
            "trajectories": {
                "draw_stride": draw_stride,
                "thin": thin,
                "n_saved": per_chain.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
                "n_forkable": per_chain.iter().map(|(_, b)| *b).collect::<Vec<_>>(),
            },
        });
        std::fs::write(dir.join("pgas_summary.json"),
            serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }

    /// gh#727: a shortfall between written and forkable latent paths lands in
    /// the rendered summary, per chain, with the total named. The path-save
    /// rule cannot produce one, so this is the regression detector: if the two
    /// rules ever decouple again, or a chain's records are skipped, the number
    /// is in the artifact rather than only on a consumer's stderr.
    #[test]
    fn saved_path_table_names_the_unusable_paths_per_chain() {
        let dir = crate::test_support::unique_temp_dir("summary_gh727_lossy");
        std::fs::create_dir_all(&dir).unwrap();
        // Two chains, one path every 28 retained draws at thin 5; chain 1 had
        // 200 of its 250 records skipped as incoherent, chain 2 none.
        write_traj_summary(&dir, 28, 5, &[(250, 50), (250, 250)]);
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let t = fmt.saved_path_table(&dir);
        assert!(t.contains("saved latent paths"), "header present:\n{t}");
        // Two per-chain rows carrying written / forkable / unusable.
        assert!(t.contains("     1        250         50       200\n"),
            "chain 1 row must show 250 written, 50 forkable, 200 unusable:\n{t}");
        assert!(t.contains("     2        250        250         0\n"),
            "chain 2 row must show nothing unusable:\n{t}");
        assert!(t.contains("200 of 500 written paths cannot be joined"),
            "the total shortfall must be stated:\n{t}");
        assert!(t.contains("every 28 retained draw"),
            "the rule the shortfall is against must be stated:\n{t}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Negative control, and the ordinary case: every written path landed on a
    /// retained draw, so the counts render and nothing claims a shortfall.
    #[test]
    fn saved_path_table_is_quiet_when_every_path_is_forkable() {
        let dir = crate::test_support::unique_temp_dir("summary_gh727_clean");
        std::fs::create_dir_all(&dir).unwrap();
        write_traj_summary(&dir, 36, 5, &[(200, 200), (200, 200)]);
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let t = fmt.saved_path_table(&dir);
        assert!(t.contains("saved latent paths"), "header still present:\n{t}");
        assert!(t.contains("     1        200        200         0\n"),
            "the counts still render, with nothing unusable:\n{t}");
        assert!(!t.contains("cannot be joined"),
            "no shortfall must be claimed:\n{t}");
        assert!(!t.contains("skipped as"),
            "a shortfall it does not have must not be explained:\n{t}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A stage whose summary carries no `trajectories` block (PMMH / NUTS, or a
    /// PGAS run predating gh#727) renders nothing rather than a row of dashes.
    #[test]
    fn saved_path_table_is_empty_without_the_block() {
        let dir = crate::test_support::unique_temp_dir("summary_gh727_absent");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("pgas_summary.json"), r#"{"stage":"pgas","thin":5}"#).unwrap();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        assert_eq!(fmt.saved_path_table(&dir), "");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── the rendered grid, against adversarial input ─────────────────────
    //
    // The defect these pin is INPUT-dependent: on a corpus of short names the
    // old code rendered correctly, so reading the format strings could not
    // show it. The fixture therefore carries what a real fit carries — a
    // stratified name past the old column width and means six orders of
    // magnitude apart — and the rendered text is compared whole.

    /// The block a `bayesian_block` renders under `header`, up to its first
    /// blank line. Comparing the block whole is the point: an alignment defect
    /// is a property of the WHOLE row, and a `contains` on one cell cannot see
    /// that the columns after it moved.
    fn block_under(text: &str, header: &str) -> String {
        let mut out = String::new();
        let mut inside = false;
        for line in text.lines() {
            if !inside {
                if line.trim() == header {
                    inside = true;
                    out.push_str(line);
                    out.push('\n');
                }
                continue;
            }
            if line.trim().is_empty() {
                break;
            }
            out.push_str(line);
            out.push('\n');
        }
        assert!(!out.is_empty(), "no `{header}` block in:\n{text}");
        out
    }

    fn wide_range_pgas_result() -> PgasStageResult {
        use std::collections::BTreeMap;
        PgasStageResult {
            diagnostics: PosteriorDiagnostics {
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([
                        ("I0_ituri".to_string(), 3.481),
                        ("N0_haut_uele".to_string(), 1.002),
                        ("iota".to_string(), 1.010),
                        ("kappa".to_string(), 3.412),
                        ("phi_split_haut_uele".to_string(), 1.194),
                    ]),
                    BTreeMap::from([
                        ("I0_ituri".to_string(), 9.0),
                        ("N0_haut_uele".to_string(), 1500.0),
                        ("iota".to_string(), 1200.0),
                        ("kappa".to_string(), 9.0),
                        ("phi_split_haut_uele".to_string(), 28.0),
                    ]),
                    BTreeMap::from([
                        ("I0_ituri".to_string(), 21.0),
                        ("N0_haut_uele".to_string(), 1800.0),
                        ("iota".to_string(), 1400.0),
                        ("kappa".to_string(), 17.0),
                        ("phi_split_haut_uele".to_string(), 317.0),
                    ]),
                ),
                n_samples: 4800,
                thin: 1,
                wall_time_secs: None,
                n_chains: 8,
            },
            // Six orders of magnitude between the smallest and largest mean.
            posterior_mean: BTreeMap::from([
                ("I0_ituri".to_string(), 240.759_840),
                ("N0_haut_uele".to_string(), 6_322_125.1),
                ("iota".to_string(), 1e-6),
                ("kappa".to_string(), 0.001_854_2),
                ("phi_split_haut_uele".to_string(), 59.681_847),
            ]),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        }
    }

    /// A parameter name longer than the column width must not shove the four
    /// numeric columns right, and the mean column must carry significant
    /// figures rather than six decimals for everything.
    ///
    /// `phi_split_haut_uele` is 19 characters against the old `{:14}`, which is
    /// a MINIMUM width — it padded short names and passed long ones through
    /// whole. `iota` at `1e-6` and `N0_haut_uele` at `6.3e6` are the two ends
    /// of the range six fixed decimals cannot serve.
    #[test]
    fn posterior_summary_grid_holds_under_long_names_and_a_wide_value_range() {
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let no_traces = std::path::Path::new("/nonexistent/stage_dir");
        let out = fmt.bayesian_block(
            "posterior", "pgas", no_traces,
            BayesianView::Pgas(&wide_range_pgas_result()), None,
            LoglikType::CompleteData,
        );
        let table = block_under(&out, "posterior summary");
        assert_eq!(table, concat!(
"  posterior summary\n",
"    param                         mean   ESS bulk   ESS tail       R\u{302}\n",
"    I0_ituri                     240.8          9         21    3.481\n",
"    N0_haut_uele               6.322e6       1500       1800    1.002\n",
"    iota                      1.000e-6       1200       1400    1.010\n",
"    kappa                     0.001854          9         17    3.412\n",
"    phi_split_haut_uele          59.68         28        317    1.194\n",
        ));
    }

    /// A name past [`NAME_COL_MAX`] is ellipsized in the MIDDLE, so the
    /// stratum suffix that distinguishes one row from the next survives, and
    /// the grid still holds.
    #[test]
    fn a_name_past_the_column_cap_keeps_its_stratum_suffix() {
        use std::collections::BTreeMap;
        let long_a = "incidence_reporting_probability_by_health_zone_ituri";
        let long_b = "incidence_reporting_probability_by_health_zone_nord_kivu";
        assert!(long_a.len() > NAME_COL_MAX && long_b.len() > NAME_COL_MAX);
        let mut r = wide_range_pgas_result();
        r.posterior_mean = BTreeMap::from([
            (long_a.to_string(), 0.31),
            (long_b.to_string(), 0.62),
        ]);
        r.diagnostics.per_param = crate::fit::method_result::per_param_from_maps(
            BTreeMap::from([(long_a.to_string(), 1.01), (long_b.to_string(), 1.02)]),
            BTreeMap::from([(long_a.to_string(), 100.0), (long_b.to_string(), 200.0)]),
            BTreeMap::from([(long_a.to_string(), 300.0), (long_b.to_string(), 400.0)]),
        );
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let no_traces = std::path::Path::new("/nonexistent/stage_dir");
        let out = fmt.bayesian_block(
            "posterior", "pgas", no_traces, BayesianView::Pgas(&r), None,
            LoglikType::CompleteData,
        );
        let table = block_under(&out, "posterior summary");
        for line in table.lines().skip(1) {
            assert_eq!(line.chars().count(), 4 + NAME_COL_MAX + 1 + 14 + 1 + 10 + 1 + 10 + 1 + 8,
                "every row is the same width: {line:?}");
        }
        assert!(table.contains("incidence_reporting_p\u{2026}"),
            "the head is what gives way:\n{table}");
        assert!(table.contains("_zone_ituri "),
            "the ituri row must still be identifiable by its suffix:\n{table}");
        assert!(table.contains("_zone_nord_kivu "),
            "the nord_kivu row must still be identifiable by its suffix:\n{table}");
    }

    #[test]
    fn provenance_block_passes_when_params_match() {
        let dir = crate::test_support::unique_temp_dir("summary_prov_ok");
        std::fs::create_dir_all(&dir).unwrap();
        // Values must match `synthetic_fit_state().start_values`
        // exactly — the second cross-check (fit_state ↔ final_params)
        // compares them.
        std::fs::write(dir.join("final_params.toml"),
            "R0 = 56.0\nsigma = 0.08\ngamma = 0.08\n").unwrap();
        std::fs::write(dir.join("mle_params.toml"),
            "R0 = 56.0\nsigma = 0.08\ngamma = 0.08\n").unwrap();
        let state = synthetic_fit_state();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let prov = fmt.provenance_block(&dir.to_string_lossy(), &state);
        assert!(!prov.failed, "must not flag when params match: {}", prov.text);
        assert!(prov.text.contains("✓"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Phase 4 / 5 tests ─────────────────────────────────────────

    /// Build a fit dir with one stage at the v2 layout
    /// (`<dir>/real/fit_1/<stage>/`), and write the top-level + stage
    /// `run.json` files so `fit_tree::walk_fit_dir` finds it. Mirror
    /// of what `camdl fit run` produces, minus the bits these tests
    /// don't need.
    fn make_fit_dir(stage: &str, state: &FitState, params: &[(&str, f64)])
        -> std::path::PathBuf
    {
        let dir = crate::test_support::unique_temp_dir(&format!("summary_format_{stage}"));
        std::fs::create_dir_all(&dir).unwrap();
        let parent_hash: String = "deadbeef".repeat(8);
        write_top_level_fit_run(&dir, &parent_hash);

        let stage_dir = dir.join("real").join("fit_1").join(stage);
        std::fs::create_dir_all(&stage_dir).unwrap();
        state.save(&stage_dir.to_string_lossy()).unwrap();
        write_stage_run(&stage_dir, &parent_hash, stage, crate::run_meta::FitAlgorithm::If2);

        // final_params.toml + mle_params.toml carrying matching values
        // so the provenance cross-check passes.
        let mut body = String::new();
        for (k, v) in params {
            body.push_str(&format!("{} = {}\n", k, v));
        }
        std::fs::write(stage_dir.join("final_params.toml"), &body).unwrap();
        std::fs::write(stage_dir.join("mle_params.toml"), &body).unwrap();
        dir
    }

    /// Write the fit-level provenance sidecar at the segment root so
    /// `FitView::read` treats `dir` as a well-formed fit. `parent_hash` is the
    /// `fit`-level hash the stage leaves also carry.
    fn write_top_level_fit_run(dir: &std::path::Path, _parent_hash: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("fit.meta.json"),
            r#"{"model_path":"sir.camdl","model_identity":"f00d","fit_toml_path":"fit.toml","estimated":["R0","sigma","gamma"]}"#,
        )
        .unwrap();
    }

    /// Write a `FitStage` `runid::RunRecord` leaf for `stage` under `stage_dir`.
    /// `parent_hash` seeds the shared `fit`-level hash; the `inputs` carry the
    /// per-stage numbers (`method`, `n_chains`, `best_loglik`, …) the views
    /// project. The `stage` LEVEL label gets an ordinal prefix (`01-scout`,
    /// `02-refine`, …) so `FitView::read` recovers execution order; `inputs.stage`
    /// keeps the bare name consumers read.
    fn write_stage_run(stage_dir: &std::path::Path, parent_hash: &str, stage: &str, method: crate::run_meta::FitAlgorithm) {
        std::fs::create_dir_all(stage_dir).unwrap();
        // Fixed pipeline order for the test stage names → `NN-` ordinal.
        let ord = match stage {
            "scout" | "mle" => 1,
            "refine"        => 2,
            "validate"      => 3,
            _               => 9,
        };
        let stage_label = format!("{ord:02}-{stage}");
        let fit_hash: String = parent_hash.chars().cycle().take(64).collect();
        let run_id: String = format!("{}-{}", parent_hash, stage)
            .chars()
            .filter(|c| c.is_ascii_hexdigit())
            .chain(std::iter::repeat('0'))
            .take(64)
            .collect();
        let algorithm = serde_json::json!({
            "algorithm": method.as_str(),
            "backend": "chain_binomial",
            "iterations": 50,
        });
        let rec = serde_json::json!({
            "format_version": 1,
            "kind": "fit_stage",
            "run_id": run_id,
            "hash_version": 1,
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {"name": "fit",   "label": "fit",  "hash": fit_hash, "schema_version": 1},
                {"name": "stage", "label": stage_label, "hash": "1fb03eee00000000000000000000000000000000000000000000000000000000", "schema_version": 1},
                {"name": "seed",  "label": "seed_1","hash": "06cbd6b300000000000000000000000000000000000000000000000000000000", "schema_version": 1}
            ],
            "status": "completed",
            "artifacts": {},
            "inputs": {
                "stage": stage,
                "method": method.as_str(),
                "backend": "chain_binomial",
                "seed": 1,
                "n_chains": 8,
                "algorithm": algorithm,
                "best_loglik": -3804.9,
                "best_chain": 1
            },
            "provenance": {"created_at": "2026-04-27T00:00:00Z", "argv": ["camdl"]}
        });
        std::fs::write(stage_dir.join("run.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    }

    /// JSON output is parseable, schema.version is 1, stage report
    /// fields match the FitState we constructed it from. Catches any
    /// future schema rename / removal that would break the book
    /// pipeline.
    #[test]
    fn json_format_round_trips_and_carries_schema_version() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let stages = discover_stages(&dir);
        let doc = build_summary_doc(&dir.to_string_lossy(), &stages, None);
        let json = serde_json::to_string_pretty(&doc).unwrap();
        assert!(json.contains("\"version\": 1"),
            "schema.version must be present and = 1: {}", json);
        // Reparse and pin the load-bearing fields.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["schema"]["version"], 1);
        assert_eq!(parsed["fit_dir"], dir.to_string_lossy().as_ref());
        assert_eq!(parsed["stages"][0]["name"], "scout");
        assert!((parsed["stages"][0]["best_loglik"].as_f64().unwrap() - (-3804.9)).abs() < 1e-6);
        // Heuristic block is namespaced.
        assert!(parsed["stages"][0]["_heuristic"]["overall_status"].is_string());
        // Provenance keys present.
        let prov = &parsed["stages"][0]["provenance"];
        assert_eq!(prov["final_params_matches_mle_params"], true);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Markdown output is well-formed: headings, gate-verdict table,
    /// parameter table. Spot-check critical lines.
    #[test]
    fn markdown_format_renders_gate_table_and_params() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let stages = discover_stages(&dir);
        let doc = build_summary_doc(&dir.to_string_lossy(), &stages, None);
        let md = render_markdown(&doc);
        assert!(md.contains("# Fit summary:"));
        assert!(md.contains("## `scout`"));
        assert!(md.contains("### Compound scout-convergence gate"));
        assert!(md.contains("| Â (max over params"));
        assert!(md.contains("### Parameter estimates"));
        assert!(md.contains("`R0`"));
        assert!(md.contains("### Per-chain loglik-eval"));
        assert!(md.contains("### Provenance"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn latex_format_renders_tabular_blocks() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let stages = discover_stages(&dir);
        let doc = build_summary_doc(&dir.to_string_lossy(), &stages, None);
        let tex = render_latex(&doc);
        // No preamble, but tabular blocks per stage.
        assert!(tex.contains("\\subsection*{Stage:"));
        assert!(tex.contains("\\begin{tabular}"));
        assert!(tex.contains("$\\hat A$"));
        assert!(tex.contains("\\bottomrule"));
        // No raw `&` — must be escaped.
        let r0_count = tex.matches(" & ").count();
        assert!(r0_count > 0, "tables must use & as column separator");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--params-only` emits a flat params TOML that the standard
    /// loader accepts. Pipes-through to `pfilter --params` are the
    /// load-bearing use case for Phase 5.
    #[test]
    fn params_only_emits_loadable_toml() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("scout", &state, &params);

        let stages = discover_stages(&dir);
        let s = dump_params_only(&dir.to_string_lossy(), Some("scout"), &stages).unwrap();
        // No metadata leaks at top level (the existing loader skips
        // `[provenance]`, but --params-only doesn't even emit it).
        assert!(!s.contains("[provenance]"),
            "params-only must not include the [provenance] block: {}", s);

        // Round-trip via the actual production loader.
        let tmp = dir.join("emitted.toml");
        std::fs::write(&tmp, &s).unwrap();
        let loaded = crate::util::load_params_toml(tmp.to_str().unwrap()).unwrap();
        assert!((loaded["R0"] - 56.0).abs() < 1e-9);
        assert!((loaded["sigma"] - 0.08).abs() < 1e-9);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn params_only_picks_terminal_stage_in_pipeline_order() {
        // Build a fit dir with both scout and refine at the v2
        // layout; --params-only without --stage should pick refine
        // (most refined).
        let state = synthetic_fit_state();
        let scout_params = [("R0", 56.0_f64)];
        let refine_params = [("R0", 56.5_f64)];
        let dir = make_fit_dir("scout", &state, &scout_params);
        // Add a refine stage at the v2 path real/fit_1/refine/.
        let parent_hash: String = "deadbeef".repeat(8);
        let refine_dir = dir.join("real").join("fit_1").join("refine");
        std::fs::create_dir_all(&refine_dir).unwrap();
        state.save(&refine_dir.to_string_lossy()).unwrap();
        write_stage_run(&refine_dir, &parent_hash, "refine", crate::run_meta::FitAlgorithm::If2);
        let mut body = String::new();
        for (k, v) in refine_params { body.push_str(&format!("{} = {}\n", k, v)); }
        std::fs::write(refine_dir.join("final_params.toml"), &body).unwrap();
        std::fs::write(refine_dir.join("mle_params.toml"), &body).unwrap();

        let stages = discover_stages(&dir);
        let s = dump_params_only(&dir.to_string_lossy(), None, &stages).unwrap();
        assert!(s.contains("--stage refine"),
            "no --stage filter must pick refine over scout: {}", s);
        assert!(s.contains("R0 = 56.5"),
            "must dump refine's params, not scout's: {}", s);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn params_only_errors_when_no_completed_stage() {
        let dir = crate::test_support::unique_temp_dir("summary_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let stages = discover_stages(&dir);
        let err = dump_params_only(&dir.to_string_lossy(), None, &stages).unwrap_err();
        assert!(err.contains("no completed fit-stage runs"));
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── dt-check verdict rendering (gh#52) ───────────────────────────

    fn dt_check_pass() -> super::super::dt_check::DtCheckResult {
        use super::super::dt_check::{compute_verdict, LadderEntry};
        compute_verdict(&[
            LadderEntry { dt: 0.1,  loglik: -58.7, se: 0.07 },
            LadderEntry { dt: 0.05, loglik: -59.3, se: 0.07 },
        ], 2.0)
    }

    fn dt_check_fail() -> super::super::dt_check::DtCheckResult {
        use super::super::dt_check::{compute_verdict, LadderEntry};
        compute_verdict(&[
            LadderEntry { dt: 1.0,  loglik: -62.6, se: 0.07 },
            LadderEntry { dt: 0.5,  loglik: -65.7, se: 0.24 },
            LadderEntry { dt: 0.25, loglik: -73.9, se: 0.63 },
        ], 2.0)
    }

    #[test]
    fn dt_check_pass_renders_one_line() {
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let block = fmt.dt_check_block(&dt_check_pass());
        assert!(block.contains("PASS"), "must call out pass verdict: {}", block);
        // No ladder rows on PASS — keep summary terse.
        assert!(!block.contains("dt = 0.0500"),
            "should not include ladder rows on pass: {}", block);
    }

    #[test]
    fn dt_check_fail_includes_ladder_and_synth_recovery_note() {
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let block = fmt.dt_check_block(&dt_check_fail());
        assert!(block.contains("FAIL"), "must call out fail: {}", block);
        // Ladder rows present so the user can see the drift.
        assert!(block.contains("dt = "), "ladder rendered: {}", block);
        // The load-bearing teaching point: synth recovery cannot
        // detect dt bias by itself.
        assert!(block.contains("synthetic recovery"),
            "must surface the synth-recovery note on FAIL: {}", block);
    }

    // ── Calendar-time date rendering for instant estimands (§6.7) ──────

    /// Calendar context with `tau` declared `instant`-kind under
    /// `origin = 2020-02-24`, `time_unit = days` (matches the
    /// `seed_timing_dated` fixture). `R0` is deliberately NOT an
    /// instant, to confirm only instant params are date-annotated.
    fn instant_cal() -> CalendarContext {
        CalendarContext {
            origin: Some("2020-02-24".into()),
            time_unit: "days".into(),
            instant_params: ["tau".to_string()].into_iter().collect(),
        }
    }

    #[test]
    fn date_for_renders_instant_with_origin_only() {
        let cal = instant_cal();
        // 2020-02-24 + 23 days = 2020-03-18.
        assert_eq!(cal.date_for("tau", 23.0).as_deref(), Some("2020-03-18"));
        // Non-instant param: no date even though origin is set.
        assert_eq!(cal.date_for("R0", 23.0), None);
        // Negative internal time (seed before origin) renders a date
        // before the origin — the seed-timing use case.
        assert_eq!(cal.date_for("tau", -4.0).as_deref(), Some("2020-02-20"));
    }

    #[test]
    fn date_for_numeric_only_without_origin() {
        // Same instant set, but no origin → numeric-only (no crash).
        let cal = CalendarContext {
            origin: None,
            time_unit: "days".into(),
            instant_params: ["tau".to_string()].into_iter().collect(),
        };
        assert_eq!(cal.date_for("tau", 23.0), None);
        // The fully-empty (default) context is also numeric-only.
        assert_eq!(CalendarContext::default().date_for("tau", 23.0), None);
    }

    // ── gh#103 (H17): missing-origin warning for instant params ────────

    #[test]
    fn missing_origin_warns_with_instant_params_and_no_origin() {
        // The triggering case: instant-kind params present, origin absent.
        let cal = CalendarContext {
            origin: None,
            time_unit: "days".into(),
            instant_params: ["tau".to_string(), "t_seed".to_string()]
                .into_iter().collect(),
        };
        let msg = cal.missing_origin_warning()
            .expect("instant params without origin must warn");
        assert!(msg.contains("tau") && msg.contains("t_seed"),
            "warning must name the instant params: {}", msg);
        assert!(msg.contains("origin"),
            "warning must point at the missing `origin`: {}", msg);
    }

    #[test]
    fn missing_origin_silent_when_origin_present() {
        // Control: instant params but origin set → dates render, no warning.
        assert!(instant_cal().missing_origin_warning().is_none(),
            "origin present — dates render, so no warning expected");
    }

    #[test]
    fn missing_origin_silent_when_no_instant_params() {
        // Control: no origin, but also no instant params → nothing to
        // render as a date, so nothing to warn about.
        let cal = CalendarContext {
            origin: None,
            time_unit: "days".into(),
            instant_params: Default::default(),
        };
        assert!(cal.missing_origin_warning().is_none(),
            "no instant params — no calendar rendering, so no warning");
        // The fully-default context (no origin, no instant params) is silent.
        assert!(CalendarContext::default().missing_origin_warning().is_none(),
            "default context has nothing to warn about");
    }

    /// FitState with an `instant`-kind estimand `tau` so the IF2
    /// parameter table exercises date annotation.
    fn instant_fit_state() -> FitState {
        let mut s = synthetic_fit_state();
        s.start_values.insert("tau".into(), 23.0);
        s.tail_chain_agreement.insert("tau".into(), 1.02);
        s
    }

    #[test]
    fn if2_text_parameter_table_annotates_instant_date() {
        let state = instant_fit_state();
        let fmt = Formatter { use_color: false, cal: instant_cal() };
        let block = fmt.parameter_table(&state);
        // tau row carries the calendar date; numeric value preserved.
        assert!(block.contains("tau"), "tau row missing: {}", block);
        assert!(block.contains("(2020-03-18)"),
            "instant date annotation missing from tau row: {}", block);
        // A non-instant param (R0) is NOT date-annotated.
        let r0_line = block.lines().find(|l| l.trim_start().starts_with("R0")).unwrap();
        assert!(!r0_line.contains("2020-"),
            "non-instant R0 must stay numeric: {}", r0_line);
    }

    #[test]
    fn if2_text_parameter_table_numeric_only_without_origin() {
        let state = instant_fit_state();
        // Default (no origin) context → numeric-only, no crash, no date.
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };
        let block = fmt.parameter_table(&state);
        assert!(block.contains("tau"), "tau row missing: {}", block);
        assert!(!block.contains("2020-"),
            "no date should appear without an origin: {}", block);
    }

    #[test]
    fn parameter_report_json_carries_estimate_date() {
        // An instant ParameterReport with origin set serializes an
        // `estimate_date` sibling field; a non-instant one omits it.
        let with_date = ParameterReport {
            name: "tau".into(),
            estimate: 23.0,
            chain_agreement: Some(1.02),
            perturb_only_at_t0: false,
            estimate_date: Some("2020-03-18".into()),
        };
        let json = serde_json::to_value(&with_date).unwrap();
        assert_eq!(json["estimate_date"], serde_json::json!("2020-03-18"));
        // Existing fields keep their names/shapes (additive-only).
        assert_eq!(json["name"], serde_json::json!("tau"));
        assert_eq!(json["estimate"], serde_json::json!(23.0));

        let without = ParameterReport {
            name: "R0".into(),
            estimate: 2.5,
            chain_agreement: Some(1.01),
            perturb_only_at_t0: false,
            estimate_date: None,
        };
        let json = serde_json::to_value(&without).unwrap();
        // skip_serializing_if = Option::is_none → field absent, not null.
        assert!(json.get("estimate_date").is_none(),
            "numeric param must omit estimate_date entirely: {}", json);
    }

    /// A `draws.tsv` with a `beta` param on `n_chains` chains, `per_chain` draws
    /// each. Chain `outlier` (0-based) sits far from the rest, so it dominates
    /// the between-chain spread → a large R̂ over all chains.
    fn write_outlier_draws(dir: &Path, n_chains: usize, per_chain: usize, outlier: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let mut s = String::from("chain\tdraw\tbeta\n");
        for c in 0..n_chains {
            for d in 0..per_chain {
                let jitter = ((d % 7) as f64 - 3.0) * 0.003;
                let beta = if c == outlier { 0.85 + jitter } else { 0.35 + jitter };
                s.push_str(&format!("{c}\t{d}\t{beta:.4}\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), s).unwrap();
    }

    /// The load-bearing numeric: `fit summary --exclude-chains` recomputes R̂/ESS
    /// over the retained chains with the fit's OWN estimator. Over all 4 chains
    /// the outlier inflates R̂ far past the 1.1 gate; dropping it collapses R̂ to
    /// ~1 and yields a finite (gated) ESS. The posterior mean also moves toward
    /// the retained (tight) chains.
    #[test]
    fn recompute_over_subset_drops_outlier_and_fixes_rhat() {
        let dir = crate::test_support::unique_temp_dir("summary_subset_recompute");
        write_outlier_draws(&dir, 4, 40, 3); // chains 1..4; chain 4 (0-based 3) stuck

        // Ground truth: over ALL four chains R̂ is large (the outlier disagrees).
        let all: Vec<Vec<f64>> = {
            let rows = crate::load_draws_tsv_keyed(&dir.join("draws.tsv").to_string_lossy()).unwrap();
            let mut by_chain: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for r in &rows {
                by_chain.entry(r.chain.unwrap()).or_default().push(r.params["beta"]);
            }
            by_chain.into_values().collect()
        };
        let rhat_all = crate::fit::runner::compute_rhat_ess(&all)
            .rank()
            .expect("four chains of equal length are scored")
            .rhat;
        assert!(rhat_all > 1.5, "over all chains the outlier inflates R̂: {rhat_all}");

        // Recompute over the subset (drop chain 4).
        let mut diag = PosteriorDiagnostics {
            per_param: crate::fit::method_result::per_param_from_maps(
                BTreeMap::from([("beta".to_string(), rhat_all)]),
                BTreeMap::from([("beta".to_string(), f64::NAN)]),
                BTreeMap::new(),
            ),
            n_samples: 160,
            thin: 1,
            wall_time_secs: Some(10.0),
            n_chains: 4,
        };
        let mut mean = BTreeMap::from([("beta".to_string(), 0.475)]); // mixed (incl. outlier)
        let sel = ChainSelection::parse_exclude("4").unwrap();
        let info = recompute_over_subset(&mut diag, &mut mean, &dir, &sel).unwrap();

        assert_eq!(info.kept, vec![1, 2, 3]);
        assert_eq!(info.excluded, vec![4]);
        assert_eq!(info.n_total, 4);
        assert_eq!(diag.n_chains, 3, "n_chains is now the retained count");
        assert_eq!(diag.n_samples, 120, "3 retained chains × 40 draws");
        let max_r = diag.max_rhat().expect("the retained subset is assessable");
        assert!(
            max_r < 1.1,
            "excluding the outlier collapses R̂ below the gate: {max_r}"
        );
        let ess = diag.min_ess().expect("ess present");
        assert!(ess.is_finite() && ess > 0.0, "subset ESS is finite + positive: {ess}");
        assert!(
            (mean["beta"] - 0.35).abs() < 0.02,
            "posterior mean moves onto the retained tight chains: {}",
            mean["beta"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    fn synthetic_pgas_result() -> MethodResult {
        MethodResult::Pgas(PgasStageResult {
            diagnostics: PosteriorDiagnostics {
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([("R0".to_string(), 1.02)]),
                    BTreeMap::new(),
                    BTreeMap::new(),
                ),
                n_samples: 100,
                thin: 1,
                wall_time_secs: None,
                n_chains: 4,
            },
            posterior_mean: BTreeMap::new(),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        })
    }

    fn synthetic_pmmh_result() -> MethodResult {
        MethodResult::Pmmh(PmmhStageResult {
            diagnostics: PosteriorDiagnostics {
                per_param: crate::fit::method_result::per_param_from_maps(
                    BTreeMap::from([("R0".to_string(), 1.03)]),
                    BTreeMap::new(),
                    BTreeMap::new(),
                ),
                n_samples: 100,
                thin: 1,
                wall_time_secs: None,
                n_chains: 4,
            },
            posterior_mean: BTreeMap::new(),
            acceptance_rate: 0.24,
            map_loglik: -3801.2,
        })
    }

    fn synthetic_if2_result() -> MethodResult {
        MethodResult::If2(If2StageResult {
            best_loglik: -3804.9,
            best_chain: 1,
            theta_hat: BTreeMap::new(),
            max_chain_agreement: 1.04,
            gate_verdict: crate::fit::method_result::GateVerdict::Pass,
            ess_at_mle: None,
            n_chains: 8,
            n_iter: 50,
        })
    }

    /// gh#280: every `StageReport` JSON object carries `loglik_type`,
    /// derived from the typed `method_result`. Fails on the pre-gh#280 code
    /// (the struct had no such field). The PGAS value MUST read
    /// `complete_data` even though its `best_loglik` is null — that is the
    /// joint-vs-marginal distinction an agent scrapes.
    #[test]
    fn stage_report_json_carries_loglik_type() {
        let cal = CalendarContext::default();

        let pgas = bayesian_stage_report("pgas", "pgas", Some(synthetic_pgas_result()), &cal);
        let j = serde_json::to_value(&pgas).unwrap();
        assert_eq!(j["loglik_type"], serde_json::json!("complete_data"),
            "PGAS stage must tag its joint loglik: {j}");
        assert!(j["best_loglik"].is_null(),
            "PGAS has no scalar best_loglik, yet still carries loglik_type: {j}");

        let pmmh = bayesian_stage_report("pmmh", "pmmh", Some(synthetic_pmmh_result()), &cal);
        let j = serde_json::to_value(&pmmh).unwrap();
        assert_eq!(j["loglik_type"], serde_json::json!("marginal"),
            "PMMH MAP loglik is marginal: {j}");

        // IF2 via the dedicated builder. No stage dir on disk → the
        // provenance reads tolerate absence; loglik_type comes from the
        // typed result.
        let if2 = if2_stage_report(
            "scout", "/nonexistent/stage", &synthetic_fit_state(),
            Some(synthetic_if2_result()), None, None, &cal,
        );
        let j = serde_json::to_value(&if2).unwrap();
        assert_eq!(j["loglik_type"], serde_json::json!("if2"),
            "IF2 stage must tag its marginal: {j}");
    }

    /// gh#280: human headlines label the loglik class, with the tag rendered
    /// *after* the number so a `loglik=<num>` scraper stops before it.
    #[test]
    fn headlines_label_the_loglik_class() {
        let cal = CalendarContext::default();
        let fmt = Formatter { use_color: false, cal: CalendarContext::default() };

        // Terminal IF2 headline.
        let block = fmt.stage_block(
            "scout", "/nonexistent/stage", &synthetic_fit_state(), None, None, None,
        );
        let line = block.text.lines().find(|l| l.contains("best loglik:"))
            .expect("IF2 block has a best-loglik headline");
        assert!(line.contains("(if2)"), "IF2 headline must carry (if2): {line}");
        let num_at = line.find(|c: char| c == '-' || c.is_ascii_digit()).unwrap();
        let tag_at = line.find("(if2)").unwrap();
        assert!(tag_at > num_at, "type tag must render after the number: {line}");

        // Markdown + LaTeX exports of a PMMH (marginal) stage.
        let pmmh = bayesian_stage_report("pmmh", "pmmh", Some(synthetic_pmmh_result()), &cal);
        let md = render_md_stage(&pmmh);
        assert!(md.contains("- best loglik:") && md.contains("(marginal)"),
            "md PMMH headline carries (marginal): {md}");
        let tex = render_latex_stage(&pmmh);
        assert!(tex.contains("Best log-likelihood") && tex.contains("(marginal)"),
            "latex PMMH headline carries (marginal): {tex}");
    }
}
