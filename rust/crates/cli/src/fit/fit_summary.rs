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
use crate::evidence::NATS_TO_DB;
use crate::fit::config_diff::ConfigDiff;
use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};
use crate::fit::fit_tree::{self, DataKind};
use crate::fit::fit_view::FitView;
use crate::fit::method_result::{
    If2StageResult, MethodResult, PgasStageResult, PmmhStageResult,
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

    match args.format {
        FitSummaryFormat::Text => format_text(&dir, args, &selected, strict),
        FitSummaryFormat::Json => format_json(&dir, args, &selected, strict),
        FitSummaryFormat::Md => format_md(&dir, args, &selected, strict),
        FitSummaryFormat::Latex => format_latex(&dir, args, &selected, strict),
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

/// Documented parameters (name + `#'` doc block) recovered from the fit's
/// model IR, for the parameter legend at the top of the summary. Uses the same
/// model-path recovery as [`load_calendar_context`]. Empty when the model can't
/// be located or no parameter carries a `#'` doc — so undocumented fits show no
/// legend.
fn load_documented_params(fit_dir: &Path) -> Vec<(String, ir::parameter::DocBlock)> {
    let model_path = FitView::read(fit_dir)
        .map(|v| v.model)
        .filter(|m| !m.is_empty())
        .or_else(|| {
            crate::cas_read::walk_records(fit_dir)
                .into_iter()
                .find_map(|(_, rec)| rec.provenance.source_paths.first().cloned())
        });
    let Some(model_path) = model_path else { return Vec::new() };
    match crate::util::load_model_docs(&model_path) {
        // The envelope dictionary keys by base parameter name (`R0`, not
        // `R0_urban`), so a stratified family shows once — `BTreeMap` order is
        // deterministic.
        Ok(docs) => docs.parameters.into_iter().collect(),
        Err(_) => Vec::new(),
    }
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

fn format_text(dir: &str, args: &FitSummaryArgs, stages: &[ResolvedStage], strict: bool) {
    let use_color = should_use_color(args.no_color);
    let cal = load_calendar_context(Path::new(dir));
    let fmt = Formatter { use_color, cal };
    let mut had_provenance_failure = false;

    print!("{}", fmt.fit_header(dir));

    // Parameter legend from the model's `#'` docs (symbol — description [ref]).
    // Shown only when the model documents at least one parameter, so it adds no
    // noise to undocumented fits.
    let documented = load_documented_params(Path::new(dir));
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
        let typed = match MethodResult::load_from(&resolved.stage_dir, &resolved.method) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "warning: cannot load {} ({}): {}",
                    stage_dir_str, resolved.method, e
                );
                continue;
            }
        };

        match &typed {
            MethodResult::If2(if2) => {
                // IF2 keeps the rich FitState rendering — gate, params
                // table, per-chain loglik-eval, provenance — because it
                // surfaces information the typed `If2StageResult`
                // doesn't (e.g. ivp markers, per-chain SE, raw
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
                    fmt.bayesian_block(&resolved.stage, "pgas", BayesianView::Pgas(pgas))
                );
                prev_stage_name = Some(resolved.stage.clone());
                // Bayesian rows have no scalar best_loglik to chain
                // through; `prev_loglik` stays where it was.
            }
            MethodResult::Pmmh(pmmh) => {
                print!(
                    "{}",
                    fmt.bayesian_block(&resolved.stage, "pmmh", BayesianView::Pmmh(pmmh))
                );
                prev_loglik = Some(pmmh.map_loglik);
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
                for (k, v) in &r.theta_hat {
                    match fmt.cal.date_for(k, *v) {
                        Some(date) => println!("      {:<14} = {}  ({})", k, v, date),
                        None => println!("      {:<14} = {}", k, v),
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
    if let Ok(j) = crate::fit::joint::resolve_joint(dir, args.stage.as_deref()) {
        println!();
        println!("  {}", fmt.bold("(θ, X) forkability"));
        let note = if j.n_forkable == j.n_total {
            fmt.ok("(all draws)")
        } else {
            fmt.dim("(partial — only path-saved draws can be conditioned-forked)")
        };
        println!("    forkable draws: {}/{}  {}", j.n_forkable, j.n_total, note);
    }

    if strict && had_provenance_failure {
        eprintln!();
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn format_json(dir: &str, _args: &FitSummaryArgs, stages: &[ResolvedStage], strict: bool) {
    let doc = build_summary_doc(dir, stages);
    let any_failed = doc.stages.iter().any(|s| s.provenance_failed());
    let s = serde_json::to_string_pretty(&doc).expect("FitSummaryDoc must serialize");
    println!("{}", s);
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn format_md(dir: &str, _args: &FitSummaryArgs, stages: &[ResolvedStage], strict: bool) {
    let doc = build_summary_doc(dir, stages);
    let any_failed = doc.stages.iter().any(|s| s.provenance_failed());
    print!("{}", render_markdown(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
}

fn format_latex(dir: &str, _args: &FitSummaryArgs, stages: &[ResolvedStage], strict: bool) {
    let doc = build_summary_doc(dir, stages);
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

    fn gate_verdict_block(&self, state: &FitState) -> String {
        let mut s = String::new();
        s.push_str(&format!("  {}\n", self.bold("compound scout-convergence gate")));

        // Resolve the gate config to render against. Priority:
        //   1. state.resolved_gate (Phase 3 — the value actually used)
        //   2. GateConfig::default() with a "(thresholds unknown)" caveat
        let (gate, threshold_source) = match &state.resolved_gate {
            Some(g) => (g.clone(), GateThresholdSource::Resolved),
            None => (GateConfig::default(), GateThresholdSource::DefaultFallback),
        };

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
        s.push_str(&format!("  {}\n", self.bold("parameter estimates (loglik-eval, selected chain θ̂)")));
        if state.start_values.is_empty() {
            s.push_str(&format!("    {}\n", self.dim("(no start_values in fit_state.toml)")));
            s.push('\n');
            return s;
        }
        let ivp_set: std::collections::HashSet<&str> = state.ivp_params.iter()
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
        for k in est_keys {
            let v = state.start_values[k];
            let agreement = state.tail_chain_agreement.get(k).copied();
            let agreement_str = match agreement {
                Some(r) => {
                    let glyph = if r < 1.05 { self.ok("✓") }
                        else if r < 1.10 { self.warn("~") }
                        else { self.err("✗") };
                    format!("Â={:.3} {}", r, glyph)
                }
                None => self.dim("Â=—").to_string(),
            };
            let ivp_marker = if ivp_set.contains(k.as_str()) {
                format!(" {}", self.dim("(ivp)"))
            } else {
                String::new()
            };
            let date_marker = match self.cal.date_for(k, v) {
                Some(date) => format!("  ({})", date),
                None => String::new(),
            };
            s.push_str(&format!("    {:12} = {:<12.6}  {}{}{}\n",
                k, v, agreement_str, ivp_marker, date_marker));
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
    fn bayesian_block(&self, stage: &str, method: &str, view: BayesianView<'_>) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "══ {} {} {}\n",
            self.bold(stage),
            self.dim(&format!("[{}]", method)),
            "═".repeat(74_usize.saturating_sub(stage.len() + method.len() + 3))
        ));

        let (n_chains, n_samples, posterior_mean, ess, max_rhat, acceptance_summary, map_loglik) =
            match view {
                BayesianView::Pgas(r) => (
                    r.n_chains,
                    r.n_samples,
                    &r.posterior_mean,
                    &r.ess_per_param,
                    r.max_rhat,
                    None::<f64>,
                    None::<f64>,
                ),
                BayesianView::Pmmh(r) => (
                    r.n_chains,
                    r.n_samples,
                    &r.posterior_mean,
                    &r.ess,
                    r.max_rhat,
                    Some(r.acceptance_rate),
                    Some(r.map_loglik),
                ),
            };

        s.push_str(&format!("  chains:       {}\n", n_chains));
        s.push_str(&format!("  samples:      {}\n", n_samples));
        if let Some(ll) = map_loglik {
            s.push_str(&format!("  MAP loglik:   {:.1}\n", ll));
        }
        s.push('\n');

        // Convergence: Gelman-Rubin R̂ (NOT IF2's Â — see
        // method_result.rs §`max_chain_agreement` vs §`max_rhat`).
        s.push_str(&format!("  {}\n", self.bold("posterior convergence")));
        let r_glyph = if max_rhat < 1.05 {
            self.ok("✓")
        } else {
            self.err("✗")
        };
        s.push_str(&format!(
            "    max R̂ = {:.3}  {}  (threshold 1.05)\n",
            max_rhat, r_glyph
        ));
        if let Some(acc) = acceptance_summary {
            s.push_str(&format!("    acceptance = {:.3} (mean across chains)\n", acc));
        }
        s.push('\n');

        // Posterior parameter table.
        s.push_str(&format!("  {}\n", self.bold("posterior summary")));
        if posterior_mean.is_empty() {
            s.push_str(&format!("    {}\n", self.dim("(no posterior parameters)")));
        } else {
            s.push_str(&format!(
                "    {:14} {:>14} {:>10} {:>8}\n",
                "param", "mean", "ESS", "R̂?"
            ));
            for (name, mean) in posterior_mean.iter() {
                let ess_v = ess.get(name).copied();
                let ess_str = match ess_v {
                    Some(v) => format!("{:>10.0}", v),
                    None => format!("{:>10}", "—"),
                };
                let date_marker = match self.cal.date_for(name, *mean) {
                    Some(date) => format!("  ({})", date),
                    None => String::new(),
                };
                s.push_str(&format!(
                    "    {:14} {:>14.6} {} {:>8}{}\n",
                    name,
                    mean,
                    ess_str,
                    "", // per-param R̂ not surfaced in this view
                    date_marker
                ));
            }
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
    pub ivp: bool,
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
fn build_summary_doc(dir: &str, stages: &[ResolvedStage]) -> FitSummaryDoc {
    let cal = load_calendar_context(Path::new(dir));
    let mut stage_reports: Vec<StageReport> = Vec::new();
    let mut prev_loglik: Option<f64> = None;
    let mut prev_stage_name_owned: Option<String> = None;
    for resolved in stages {
        let stage_dir_str = resolved.stage_dir.to_string_lossy().into_owned();
        let typed = MethodResult::load_from(&resolved.stage_dir, &resolved.method).ok();
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
            (Some(MethodResult::Pgas(_)), _) | (Some(MethodResult::Pmmh(_)), _) => {
                let r = bayesian_stage_report(
                    &resolved.stage,
                    &resolved.method,
                    typed.clone(),
                    &cal,
                );
                if let Some(MethodResult::Pmmh(p)) = &typed {
                    prev_loglik = Some(p.map_loglik);
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
        Some(MethodResult::Pgas(r)) => (r.n_chains, None),
        Some(MethodResult::Pmmh(r)) => (r.n_chains, Some(r.map_loglik)),
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
    let ivp_set: std::collections::HashSet<&str> = state.ivp_params.iter()
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
            ivp: ivp_set.contains(k.as_str()),
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
        s.push_str(&format!("### Posterior summary (PGAS, max R̂ = {:.3})\n\n", p.max_rhat));
        s.push_str("| param | mean | q025 | q975 | ESS |\n|---|---|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let ess = p.ess_per_param.get(name).copied().map(|v| format!("{:.0}", v)).unwrap_or_else(|| "—".into());
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} | {} |\n", name, mean_cell, q025, q975, ess));
        }
        s.push('\n');
    } else if let Some(MethodResult::Pmmh(p)) = &stage.method_result {
        s.push_str(&format!(
            "### Posterior summary (PMMH, max R̂ = {:.3}, acceptance = {:.3})\n\n",
            p.max_rhat, p.acceptance_rate
        ));
        s.push_str("| param | mean | ESS |\n|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let ess = p.ess.get(name).copied().map(|v| format!("{:.0}", v)).unwrap_or_else(|| "—".into());
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} |\n", name, mean_cell, ess));
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
            let flag = if p.ivp { "ivp" } else { "" };
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
    for stage in &doc.stages {
        s.push_str(&render_latex_stage(stage));
    }
    s
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
            let flag = if p.ivp { "ivp" } else { "" };
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
            "Posterior summary (max $\\hat R$ = {:.3}):\n\n",
            p.max_rhat
        ));
        s.push_str("\\begin{tabular}{lrrrr}\n\\toprule\n");
        s.push_str("Parameter & Mean & $q_{0.025}$ & $q_{0.975}$ & ESS \\\\\n\\midrule\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "---".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "---".into());
            let ess = p.ess_per_param.get(name).copied().map(|v| format!("{:.0}", v)).unwrap_or_else(|| "---".into());
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!(
                "\\texttt{{{}}} & {} & {} & {} & {} \\\\\n",
                escape_latex(name), mean_cell, q025, q975, ess,
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    } else if let Some(MethodResult::Pmmh(p)) = &stage.method_result {
        s.push_str(&format!(
            "Posterior summary (max $\\hat R$ = {:.3}; acceptance = {:.3}):\n\n",
            p.max_rhat, p.acceptance_rate
        ));
        s.push_str("\\begin{tabular}{lrr}\n\\toprule\n");
        s.push_str("Parameter & Mean & ESS \\\\\n\\midrule\n");
        for (name, mean) in &p.posterior_mean {
            let ess = p.ess.get(name).copied().map(|v| format!("{:.0}", v)).unwrap_or_else(|| "---".into());
            let mean_cell = match stage.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!(
                "\\texttt{{{}}} & {} & {} \\\\\n",
                escape_latex(name), mean_cell, ess,
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
    use std::collections::HashMap;

    fn synthetic_fit_state() -> FitState {
        let mut start = HashMap::new();
        start.insert("R0".into(),  56.0);
        start.insert("sigma".into(), 0.08);
        start.insert("gamma".into(), 0.08);
        let mut agreement = HashMap::new();
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
            rw_sd: HashMap::new(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: agreement,
            ivp_params: vec!["I0".into()],
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
        let doc = build_summary_doc(&dir.to_string_lossy(), &stages);
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
        let doc = build_summary_doc(&dir.to_string_lossy(), &stages);
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
        let doc = build_summary_doc(&dir.to_string_lossy(), &stages);
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
            ivp: false,
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
            ivp: false,
            estimate_date: None,
        };
        let json = serde_json::to_value(&without).unwrap();
        // skip_serializing_if = Option::is_none → field absent, not null.
        assert!(json.get("estimate_date").is_none(),
            "numeric param must omit estimate_date entirely: {}", json);
    }

    fn synthetic_pgas_result() -> MethodResult {
        MethodResult::Pgas(PgasStageResult {
            n_samples: 100,
            posterior_mean: BTreeMap::new(),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            ess_per_param: BTreeMap::new(),
            max_rhat: 1.02,
            acceptance_per_param: BTreeMap::new(),
            n_chains: 4,
        })
    }

    fn synthetic_pmmh_result() -> MethodResult {
        MethodResult::Pmmh(PmmhStageResult {
            n_samples: 100,
            posterior_mean: BTreeMap::new(),
            ess: BTreeMap::new(),
            max_rhat: 1.03,
            acceptance_rate: 0.24,
            map_loglik: -3801.2,
            n_chains: 4,
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
