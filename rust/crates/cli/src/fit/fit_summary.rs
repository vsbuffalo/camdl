//! `camdl fit summary` — single-fit interpretation surface.
//!
//! Reads a fit dir produced by `camdl fit run` and renders an
//! interpretation block per method. The block content is method-aware:
//! IF2 methods render the compound-gate verdict, Â-keyed parameter
//! table, per-chain loglik-eval table, and filter-health provenance
//! cross-check; PGAS / PMMH methods render the posterior summary
//! (mean, R̂, ESS, acceptance) instead. Method names come from the
//! walker (`fit_tree::walk_fit_dir`) — there is no hard-coded method
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

use crate::out::paint::{Painter, Stream, Style};
use crate::fit::report::{Check, ChainDetail, MethodFacts};
use crate::fit::report::export::{render_latex, render_markdown};
use crate::fit::report::render::*;
use crate::fit::report::build::{build_method_report, CalendarContext, MethodOutcome, MethodReport};
use crate::args::{FitSummaryArgs, FitSummaryFormat};
use crate::chain_selection::{warn_active_selection, ChainSelection, SubsetInfo};
use crate::fit::config_diff::ConfigDiff;
use crate::fit::fit_tree::{self, DataKind};
use crate::fit::fit_view::FitView;
use crate::fit::method_result::MethodResult;
use crate::fit::table_row::{self, TableRow};
use crate::version;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Versioned JSON schema. Bumped when fields are renamed / removed /
/// retyped; field additions are non-breaking and keep version stable.
///
/// v2 (gh#901): the per-method array is `methods`, not `stages`. The
/// `[stages]` → `[method]` split left the plural key spelling a word the fit
/// config, the store levels and the CLI no longer use, and the array is
/// machine-read (the book pipeline and `fit table`'s byte-equality check both
/// parse it).
///
/// v3: the per-method `method_progression` field is gone. It carried the
/// log-likelihood difference against the method leaf listed before it, which
/// compared successive stages of the removed `[stages]` pipeline; leaves under
/// one fit directory are now independent runs whose log-likelihoods can be of
/// different kinds, so the difference compared unlike quantities.
const SCHEMA_VERSION: u32 = 4;

/// Recompute a Bayesian method's diagnostics + posterior means over a
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
    method_dir: &Path,
    selection: &ChainSelection,
) -> Result<SubsetInfo, String> {
    // Recompute R̂ / ESS over the retained chains for the estimated params — the
    // keys of `posterior_mean`, the exact set the renderer iterates, so the table
    // shape is unchanged. Routed through the one shared recompute
    // (`chain_selection::recompute_subset_diagnostics`) that `fit predict` also
    // calls, so summary and predict cannot disagree on the same fit + selection.
    let param_names: Vec<String> = posterior_mean.keys().cloned().collect();
    let sub = crate::chain_selection::recompute_subset_diagnostics(
        &method_dir.join("draws.tsv"),
        selection,
        &param_names,
    )?;
    // A subset of chains that began at one point still began at one point:
    // the recomputed R̂ is no more informative than the stored one.
    let point = crate::fit::method_result::point_start(method_dir).map_err(|e| e.to_string())?;

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
    if let Some((n, starts)) = &point {
        diag.demote_point_start(*n, starts);
    }
    diag.n_samples = sub.n_samples;
    diag.n_chains = sub.n_chains;
    // `thin` and `wall_time_secs` are properties of the whole run, unchanged by
    // a read-side subset (ESS/iter and ESS/sec are then reported over the
    // subset's ESS against the same iteration / wall-clock denominators).
    *posterior_mean = new_mean;
    Ok(sub.info)
}

/// Mutate a Bayesian method's typed result to the chain-subset diagnostics.
/// Errors (never silently no-ops) for a non-Bayesian method — the caller only
/// invokes it on Bayesian methods.
fn apply_selection_to_typed(
    typed: &mut MethodResult,
    method_dir: &Path,
    selection: &ChainSelection,
) -> Result<SubsetInfo, String> {
    match typed {
        MethodResult::Pgas(r) => {
            recompute_over_subset(&mut r.diagnostics, &mut r.posterior_mean, method_dir, selection)
        }
        MethodResult::Pmmh(r) => {
            recompute_over_subset(&mut r.diagnostics, &mut r.posterior_mean, method_dir, selection)
        }
        _ => Err(
            "--exclude-chains applies only to Bayesian methods (PGAS / PMMH / MH)".to_string(),
        ),
    }
}

/// Print the read-side chain-selection advisory to STDERR (so it never pollutes
/// the stdout summary): the identifiability nudge FIRST (when the per-chain
/// outlier signal is strong — gh#406), then the loud, non-quietable exclusion
/// warning. The flag is the second thing the user reads, not the first.
fn chain_selection_advisory(
    method_dir: &Path,
    info: &SubsetInfo,
    kind: super::loglik::LoglikType,
    stderr: Painter,
) {
    use super::chain_diagnostics as cd;
    if let Some(means) = cd::read_chain_mean_logliks(method_dir, kind) {
        let scores = cd::chain_loglik_mod_zscores(&means.scored_by_chain());
        let outliers = cd::outlier_labels(&scores);
        if !outliers.is_empty() {
            eprintln!(
                "{n} {} disagree strongly with the others on `{}` — before \
                 excluding, ask whether a parameter is unidentified (a flat likelihood ridge). \
                 The per-chain table below names them; the primary fix is the model, not the flag.",
                outliers.join(", "), means.scored_column, n = stderr.paint("note:", Style::Warn)
            );
        }
    }
    warn_active_selection(info, stderr);
}

/// Top-level entry point. Resolves `args.fit` (the fit handle) to its segment
/// directory, walks every completed fit-method run, and dispatches to the right
/// formatter based on `--format` and `--params-only`.
///
/// Exits with code 1 when the handle does not resolve; when `--strict` is set
/// and any IF2 method's provenance cross-check fails; and when every method
/// leaf the walker found failed to load, so the rendered summary has nothing
/// behind it (gh#905, [`exit_if_nothing_loaded`]).
///
/// A handle whose directory holds no completed method leaf has nothing to
/// summarise and exits 1 (gh#896); a leaf directory summarises that one run.
pub fn cmd_fit_summary(args: &FitSummaryArgs) {
    // Resolve the fit handle (@label / hash prefix / run-dir / fit.toml) to
    // what it names. A run handle summarises that run; a problem folder in
    // which one method has several runs is refused with the runs listed,
    // rather than one chosen by directory order (gh#896). summary operates on
    // the directory; it needs no config.
    use crate::fit::handle::{self, FitTarget};
    let target = handle::resolve_target(&args.fit).and_then(|t| match t {
        FitTarget::Leaf(leaf) => Ok(leaf.dir),
        FitTarget::Segment(seg) => {
            handle::refuse_several_runs_of_one_method(&args.fit, &seg).map(|()| seg)
        }
    });
    let segment = match target {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    let dir = segment.to_string_lossy().into_owned();

    let strict = args.strict || ci_env_set();
    // Warnings and notes go to stderr under the same `--no-color` as the
    // summary on stdout.
    let stderr = Painter::for_stream(Stream::Stderr, args.no_color);

    // The full set of completed method leaves, walker-discovered, in label
    // order (see `discover_methods`).
    let discovered = discover_methods(Path::new(&dir));
    if discovered.is_empty() {
        eprintln!(
            "error: no completed method leaf under {dir}, so there is nothing to summarise.\n  \
             Pass a fit's directory, a leaf directory, @label, a run-id prefix, or its fit.toml."
        );
        std::process::exit(1);
    }

    // gh#103 (H17): warn (once, to stderr) when the fit has instant-kind
    // calendar-date parameters but the model declares no `origin` — the
    // dates can't be rendered and the omission is otherwise silent.
    // stderr keeps the warning out of the JSON/stdout payload.
    if let Some(msg) = load_calendar_context(Path::new(&dir)).missing_origin_warning() {
        eprintln!("{} {}", stderr.paint("warning:", Style::Warn), msg);
    }

    if args.params_only {
        match dump_params_only(&dir, &discovered) {
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

    let selected: Vec<ResolvedMethod> = discovered.clone();

    // Parse `--exclude-chains` at the boundary into a typed selection. A
    // selection is meaningless without a Bayesian method to subset — refuse it
    // rather than silently no-op (an optimizer-only fit has no chains-as-draws).
    let selection: Option<ChainSelection> = match args.exclude_chains.as_deref() {
        Some(raw) => match ChainSelection::parse_exclude(raw) {
            Ok(sel) => {
                let has_bayesian = selected.iter().any(|s| {
                    matches!(s.method.as_str(), "pgas" | "pmmh" | "mh")
                });
                if !has_bayesian {
                    eprintln!(
                        "error: --exclude-chains needs a Bayesian method (PGAS / PMMH / MH) to \
                         subset; this fit's selected method(s) have no posterior chains"
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

    let failures = match args.format {
        FitSummaryFormat::Text => format_text(&dir, args, &selected, strict, selection.as_ref(), stderr),
        FitSummaryFormat::Json => format_json(&dir, &selected, strict, selection.as_ref(), stderr),
        FitSummaryFormat::Md => format_md(&dir, &selected, strict, selection.as_ref(), stderr),
        FitSummaryFormat::Latex => format_latex(&dir, &selected, strict, selection.as_ref(), stderr),
    };

    // One exit decision for all four formats: an unreadable leaf is reported
    // the same way whatever was being rendered (gh#905).
    exit_if_nothing_loaded(&dir, &discovered, &failures);
}

/// Load the calendar context from a fit dir: read the model path from the
/// fit-level view (`FitView.model`, sidecar-sourced), falling back to any method
/// leaf's `provenance.source_paths`; load that model and extract `origin`,
/// `time_unit`, and the `instant`-kind parameter set. Any failure (no fit view,
/// model moved/unparseable) degrades to an empty context (numeric-only
/// rendering) — never panics.
fn load_calendar_context(fit_dir: &Path) -> CalendarContext {
    // Legacy fits carried the model path on the top-level `Fit` run.json. A
    // content-addressed fit (gh#147 M3.2) has no fit-wide record — the fit
    // level is a path segment — so recover the model path from any method
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

/// The fit's model, compiled from the source path its records name.
fn load_fit_envelope(fit_dir: &Path) -> Option<ir::IrEnvelope> {
    let model_path = FitView::read(fit_dir)
        .map(|v| v.model)
        .filter(|m| !m.is_empty())
        .or_else(|| {
            crate::cas_read::walk_records(fit_dir)
                .into_iter()
                .find_map(|(_, rec)| rec.provenance.source_paths.first().cloned())
        });
    crate::util::load_model_envelope(&model_path?).ok()
}

/// One resolved fit-method leaf to render: its name, on-disk directory,
/// and method (so consumers can pattern-match on `MethodResult` once
/// the typed payload is loaded).
#[derive(Debug, Clone)]
struct ResolvedMethod {
    name: String,
    method: String,
    method_dir: PathBuf,
}

/// A method leaf the walker found but whose stored result could not be read
/// back — a truncated `<method>_summary.json`, an unparseable `fit_state.toml`,
/// a file removed under the store.
///
/// Carried out of every formatter so the exit code is decided in one place
/// (gh#905), and serialized into `--format json` so a reader parsing the
/// document finds the cause of an empty `methods` array inside the document
/// rather than on a stderr stream it may not have captured.
#[derive(Debug, Clone, Serialize)]
pub struct LoadFailure {
    /// The method-leaf directory, spelled as the stderr warning spells it.
    pub leaf: String,
    /// The loader's own message: which file, and what was wrong with it.
    pub reason: String,
}

/// Whether the unreadable leaves leave the summary with nothing to stand on.
///
/// `fit summary` used to warn about a leaf it could not read, skip it, render
/// a summary with no method in it, and exit 0 — so "this fit has nothing to
/// say" and "this fit could not be read" were the same status code, and a
/// caller that checked it carried on (gh#905).
///
/// True when every leaf that was found failed. The single-leaf case gh#905
/// reports falls under the same rule: when the handle named one leaf, that
/// leaf failing IS every leaf failing. False when at least one leaf loaded —
/// the summary is a complete answer for those leaves, the per-leaf warning
/// names the ones it is missing, and the caller did get an answer.
fn nothing_loaded(discovered: &[ResolvedMethod], failures: &[LoadFailure]) -> bool {
    !discovered.is_empty() && failures.len() >= discovered.len()
}

/// Report the leaves that could not be read, and exit non-zero when they leave
/// no summary behind. Returns normally when at least one leaf loaded.
fn exit_if_nothing_loaded(dir: &str, discovered: &[ResolvedMethod], failures: &[LoadFailure]) {
    if !nothing_loaded(discovered, failures) {
        return;
    }
    // The rendered document is already on stdout; flush it before the process
    // ends so a reader piping stdout still gets what was produced.
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!();
    eprintln!(
        "error: no method leaf under {} could be read back, so there is nothing to \
         summarise ({} found, {} unreadable):",
        dir,
        discovered.len(),
        failures.len()
    );
    for f in failures {
        eprintln!("  {}: {}", f.leaf, f.reason);
    }
    std::process::exit(1);
}

/// Walk the fit_dir and return one `ResolvedMethod` per completed method
/// leaf. Order matches `FitView.methods_declared` (label order); leaves that
/// didn't complete are dropped; leaves that completed but aren't in
/// `methods_declared` (shouldn't happen in v2 layouts, but the walker is
/// permissive) are appended at the end in walker order so they're still
/// visible.
///
/// When the same method name appears in multiple cells (synthetic
/// replicates, sweep cells), prefers Real over Synthetic, lowest
/// `fit_seed`, then lex-first method_dir — same priority as
/// [`crate::fit::table_row::build_row`]'s terminal-method picker.
fn discover_methods(fit_dir: &Path) -> Vec<ResolvedMethod> {
    // Read the fit-level view for `methods_declared`; if missing, fall back to
    // walker-order (covers user-built non-canonical fits).
    let methods_declared: Vec<String> = FitView::read(fit_dir)
        .map(|v| v.methods_declared)
        .unwrap_or_default();
    let nodes = fit_tree::walk_fit_dir(fit_dir).unwrap_or_default();

    // Best (method_name, method, dir) per method name, by (data_kind,
    // fit_seed, method_dir) priority.
    type Rank = (u8, u64, PathBuf);
    let mut best: BTreeMap<String, (String, Rank, PathBuf)> = BTreeMap::new();
    for node in &nodes {
        let method = node.view.method.as_str().to_string();
        let name = method.clone();
        let rank: Rank = match &node.view.axes {
            Some(axes) => {
                let kind_rank = match axes.data_kind {
                    DataKind::Real => 0,
                    DataKind::Synthetic { .. } => 1,
                };
                (kind_rank, axes.fit_seed, node.method_dir.clone())
            }
            None => (2, u64::MAX, node.method_dir.clone()),
        };
        best.entry(name)
            .and_modify(|(meth, cur, dir)| {
                if rank < *cur {
                    *meth = method.clone();
                    *cur = rank.clone();
                    *dir = node.method_dir.clone();
                }
            })
            .or_insert_with(|| (method.clone(), rank, node.method_dir.clone()));
    }

    let mut out: Vec<ResolvedMethod> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for name in &methods_declared {
        if let Some((method, _, dir)) = best.get(name) {
            out.push(ResolvedMethod {
                name: name.clone(),
                method: method.clone(),
                method_dir: dir.clone(),
            });
            seen.insert(name.clone());
        }
    }
    for (name, (method, _, dir)) in best {
        if !seen.contains(&name) {
            out.push(ResolvedMethod {
                name,
                method,
                method_dir: dir,
            });
        }
    }
    out
}

/// `fit summary`'s text: every method leaf loaded into its report, each
/// drawn by `render_report` under the summary view, with the fit's own lines
/// around them.
fn format_text(
    dir: &str,
    args: &FitSummaryArgs,
    methods: &[ResolvedMethod],
    strict: bool,
    selection: Option<&ChainSelection>,
    stderr: Painter,
) -> Vec<LoadFailure> {
    let mut failures: Vec<LoadFailure> = Vec::new();
    let painter = Painter::for_stream(Stream::Stdout, args.no_color);
    let cal = load_calendar_context(Path::new(dir));
    let mut had_provenance_failure = false;
    // Emit the loud selection advisory (nudge + warning) once, from the first
    // Bayesian method that recomputes.
    let mut warned = false;

    // One compile of the fit's model: its `#'` docs and constants for the
    // legends, its observation streams for the method reports' checks.
    let envelope = load_fit_envelope(Path::new(dir));
    let model = envelope.as_ref().map(|env| &env.model);
    let (docs, constants) = envelope.as_ref()
        .map(|env| (env.docs.clone(), env.model.constants.clone()))
        .unwrap_or_default();
    let has_legend = !docs.parameters.is_empty() || !constants.is_empty();
    print_lines(painter, &summary_head(dir, &docs, &constants, args.parameters));

    // The camdl version is a fact about the FIT, so it opens the first method's
    // identity line and is not repeated on the others'. A fit with no method at
    // all still has to report it, below.
    let mut first = true;

    for resolved in methods {
        let version_lead = std::mem::replace(&mut first, false);
        let method_dir_str = resolved.method_dir.to_string_lossy().into_owned();
        // A leaf whose chain directories count from 1 is refused before
        // anything on it is read: its per-chain table would label chain_N/ as
        // chain N while its draws.tsv keys the same chain N - 1 (gh#990).
        if let Err(e) = crate::chain_index::refuse_one_based_layout(&resolved.method_dir) {
            eprintln!("warning: cannot load {} ({}): {}", method_dir_str, resolved.method, e);
            failures.push(LoadFailure { leaf: method_dir_str.clone(), reason: e });
            continue;
        }
        let mut typed = match MethodResult::load_from(&resolved.method_dir, &resolved.method) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "warning: cannot load {} ({}): {}",
                    method_dir_str, resolved.method, e
                );
                failures.push(LoadFailure {
                    leaf: method_dir_str.clone(),
                    reason: e.to_string(),
                });
                continue;
            }
        };

        // The method's log-likelihood class, derived once from the typed result
        // (the single authority). It selects the per-chain trace column the
        // outlier diagnostics may compare across chains (gh#667).
        let ll_kind = crate::fit::loglik::LoglikType::from(&typed);

        // Chain selection: recompute this Bayesian method's diagnostics over the
        // retained chains before rendering (an IF2 / NLopt method has no chains
        // and is left untouched). The advisory prints once.
        let mut subset_info: Option<SubsetInfo> = None;
        if let Some(sel) = selection {
            if matches!(typed, MethodResult::Pgas(_) | MethodResult::Pmmh(_)) {
                match apply_selection_to_typed(&mut typed, &resolved.method_dir, sel) {
                    Ok(info) => {
                        if !warned {
                            chain_selection_advisory(&resolved.method_dir, &info, ll_kind, stderr);
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

        // Everything the text reads from the leaf beside the typed result,
        // read once, here.
        let chain_detail = match &typed {
            MethodResult::Pgas(_) | MethodResult::Pmmh(_) => Some(ChainDetail::gather(
                &resolved.method_dir,
                ll_kind,
                matches!(typed, MethodResult::Pgas(_)),
                subset_info,
                forkability(dir, &resolved.name, selection),
            )),
            MethodResult::If2(_) => Some(ChainDetail::default()),
            MethodResult::Nlopt(_) | MethodResult::Pfilter(_) => None,
        };
        let report = match load_method_facts(resolved, typed, model, chain_detail) {
            Ok(facts) => build_method_report(facts, &cal),
            Err(e) => {
                eprintln!("warning: cannot load {} ({}): {}", method_dir_str, resolved.method, e);
                failures.push(LoadFailure { leaf: method_dir_str.clone(), reason: e });
                continue;
            }
        };
        had_provenance_failure |= report.provenance_failed();
        let view = View::Summary(SummaryView { explain: args.explain, version_lead });
        print_lines(painter, &render_report(&MethodOutcome::Completed(report), view));
    }

    if methods.is_empty() {
        print_lines(painter, &summary_no_methods(dir));
    }
    print_lines(painter, &summary_foot(has_legend, args.parameters, args.explain));

    if strict && had_provenance_failure {
        eprintln!();
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }

    failures
}

/// Write `lines` to stdout through `painter`. A write that fails is fatal, as
/// it is for `print!`.
fn print_lines(painter: Painter, lines: &[crate::out::paint::Line]) {
    if let Err(e) = painter.write(&mut std::io::stdout().lock(), lines) {
        panic!("failed printing to stdout: {e}");
    }
}

/// gh#322: the keyed-joint (θ, X) forkable count for one method — how many of
/// its posterior draws pair with a saved smoothed trajectory (or are
/// deterministic, for ODE), i.e. how many a counterfactual `compare`/contrast
/// could fork. `None` for a method with no posterior cloud.
///
/// The cloud is resolved under this command's own `--exclude-chains`, so the
/// count describes the same posterior the rest of the summary reports — never
/// the full cloud under a header that says chains were dropped (gh#695). The
/// retained-chain scope is named on the line itself, because "24/24 forkable"
/// reads identically whether it is a subset or the whole.
fn forkability(
    dir: &str,
    method: &str,
    selection: Option<&ChainSelection>,
) -> Option<crate::fit::report::Forkable> {
    crate::posterior_draws::resolve_posterior_draws(dir, Some(method))
        .map_err(String::from)
        .map(|p| p.with_selection(selection.cloned()))
        .and_then(|p| crate::fit::joint::resolve_joint(&p))
        .ok()
        .map(|j| crate::fit::report::Forkable { n_forkable: j.n_forkable, n_total: j.n_total, selection: j.selection })
}

fn format_json(dir: &str, methods: &[ResolvedMethod], strict: bool, selection: Option<&ChainSelection>, stderr: Painter) -> Vec<LoadFailure> {
    let doc = build_summary_doc(dir, methods, selection, stderr);
    let any_failed = doc.methods.iter().any(|s| s.provenance_failed());
    let s = serde_json::to_string_pretty(&doc).expect("FitSummaryDoc must serialize");
    println!("{}", s);
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
    doc.failures
}

fn format_md(dir: &str, methods: &[ResolvedMethod], strict: bool, selection: Option<&ChainSelection>, stderr: Painter) -> Vec<LoadFailure> {
    let doc = build_summary_doc(dir, methods, selection, stderr);
    let any_failed = doc.methods.iter().any(|s| s.provenance_failed());
    print!("{}", render_markdown(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
    doc.failures
}

fn format_latex(dir: &str, methods: &[ResolvedMethod], strict: bool, selection: Option<&ChainSelection>, stderr: Painter) -> Vec<LoadFailure> {
    let doc = build_summary_doc(dir, methods, selection, stderr);
    let any_failed = doc.methods.iter().any(|s| s.provenance_failed());
    print!("{}", render_latex(&doc));
    if strict && any_failed {
        eprintln!("error: provenance cross-checks failed (--strict).");
        std::process::exit(1);
    }
    doc.failures
}

// ── Formatting layer ────────────────────────────────────────────────

// ── Helpers ─────────────────────────────────────────────────────────

fn ci_env_set() -> bool {
    matches!(std::env::var("CI").as_deref(), Ok("true") | Ok("1"))
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
    pub methods: Vec<MethodReport>,
    /// Method leaves the walker found but could not read back (gh#905). Always
    /// present, empty when every leaf loaded — so a consumer that sees an empty
    /// `methods` array can tell "this fit has no completed method" from "this
    /// fit could not be read", without parsing stderr.
    pub failures: Vec<LoadFailure>,
    /// Read-side chain selection (`--exclude-chains`), when active: the
    /// `{excluded, kept, n_total}` provenance for the subset the methods'
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

/// Walk the method dirs and build a `FitSummaryDoc`. Used by JSON, MD,
/// and LaTeX formatters. Pure on its inputs (file system + the args
/// it was called with).
fn build_summary_doc(
    dir: &str,
    methods: &[ResolvedMethod],
    selection: Option<&ChainSelection>,
    stderr: Painter,
) -> FitSummaryDoc {
    let cal = load_calendar_context(Path::new(dir));
    let model = load_fit_envelope(Path::new(dir)).map(|env| env.model);
    let mut method_reports: Vec<MethodReport> = Vec::new();
    // The chain-selection provenance (stamped on the doc) + one-shot advisory.
    let mut chain_selection_json: Option<serde_json::Value> = None;
    let mut failures: Vec<LoadFailure> = Vec::new();
    let mut advised = false;
    for resolved in methods {
        let method_dir_str = resolved.method_dir.to_string_lossy().into_owned();
        // The same refusal the text loop makes, before anything is read: the
        // per-chain rows and the `--exclude-chains` advisory would otherwise
        // label chain_N/ as chain N while draws.tsv keys it N - 1 (gh#990).
        if let Err(e) = crate::chain_index::refuse_one_based_layout(&resolved.method_dir) {
            eprintln!("warning: cannot load {} ({}): {}", method_dir_str, resolved.method, e);
            failures.push(LoadFailure { leaf: method_dir_str.clone(), reason: e });
            continue;
        }
        let typed_loaded = MethodResult::load_from(&resolved.method_dir, &resolved.method);
        // Same warning the text formatter prints. `.ok()` alone left a machine
        // reader with an empty `methods` array and nothing at all saying why
        // (gh#905).
        if let Err(e) = &typed_loaded {
            eprintln!(
                "warning: cannot load {} ({}): {}",
                method_dir_str, resolved.method, e
            );
            failures.push(LoadFailure {
                leaf: method_dir_str.clone(),
                reason: e.to_string(),
            });
        }
        let mut typed = typed_loaded.ok();
        // Chain selection: recompute this Bayesian method's diagnostics over the
        // retained chains before the report is built from the (now mutated)
        // typed payload — so JSON / MD / LaTeX all carry the subset diagnostics.
        if let Some(sel) = selection {
            if matches!(
                typed,
                Some(MethodResult::Pgas(_)) | Some(MethodResult::Pmmh(_))
            ) {
                if let Some(t) = typed.as_mut() {
                    // Same authority as the text path: the class comes from the
                    // typed result, never from a column position (gh#667).
                    let ll_kind = crate::fit::loglik::LoglikType::from(&*t);
                    match apply_selection_to_typed(t, &resolved.method_dir, sel) {
                        Ok(info) => {
                            if !advised {
                                chain_selection_advisory(&resolved.method_dir, &info, ll_kind, stderr);
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
        // A leaf whose typed result did not load is already in `failures`.
        let Some(typed) = typed else { continue };
        match load_method_facts(resolved, typed, model.as_ref(), None) {
            Ok(facts) => method_reports.push(build_method_report(facts, &cal)),
            Err(e) => {
                eprintln!("warning: cannot load {} ({}): {}", method_dir_str, resolved.method, e);
                failures.push(LoadFailure { leaf: method_dir_str.clone(), reason: e });
            }
        }
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
        methods: method_reports,
        failures,
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
                methods: Vec::new(),
                method: String::new(),
                run_id: String::new(),
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

/// `fit summary`'s producer of [`MethodFacts`]: everything the report needs,
/// read from the method leaf. `result` is the typed result already loaded
/// (and, under `--exclude-chains`, recomputed over the kept chains); `model`
/// is the fit's model, for the observation-stream checks.
fn load_method_facts(
    resolved: &ResolvedMethod,
    result: MethodResult,
    model: Option<&ir::Model>,
    mut chain_detail: Option<ChainDetail>,
) -> Result<MethodFacts, String> {
    let dir = &resolved.method_dir;
    let inputs: serde_json::Value = std::fs::read(dir.join("run.json")).ok()
        .and_then(|b| serde_json::from_slice::<runid::RunRecord>(&b).ok())
        .map(|r| r.inputs)
        .unwrap_or(serde_json::Value::Null);
    let header: Option<crate::fit::report::RunHeader> = inputs.get("header")
        .and_then(|h| serde_json::from_value(h.clone()).ok());
    let method = resolved.method.as_str();
    let backend = inputs.get("backend").and_then(|v| v.as_str()).unwrap_or("");

    // In the order the run shows them: the method's caveat, the starts note,
    // the noise check, the observation streams.
    let mut checks = Vec::new();
    if let Some(note) = crate::fit::methods::status_note_for(method, backend) {
        // PMMH records the series length it said its caveat with
        // (`pmmh_summary.json` `plan.series_len`).
        let series_len = (method == "pmmh").then(|| {
            std::fs::read_to_string(dir.join("pmmh_summary.json")).ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v["plan"]["series_len"].as_u64())
                .map(|n| n as usize)
                .filter(|&n| n > 0)
        }).flatten();
        checks.push(Check::MethodCaveat { method: method.to_string(), note: note.to_string(), series_len });
    }
    let unused = inputs.get("unused_starts");
    let params: Vec<String> = unused.and_then(|u| u.get("params")).and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if !params.is_empty() {
        let mode = unused.and_then(|u| u.get("rule")).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let note = crate::fit::report::unused_starts_note(&mode, &params, matches!(method, "pmmh" | "mh"));
        checks.push(Check::Starts { mode, params, note });
    }
    if let MethodResult::Pmmh(r) = &result {
        if let Some(check) = r.pf_noise {
            checks.push(Check::FilterNoise { check });
        }
    }
    if matches!(method, "pmmh" | "mh" | "pgas") {
        if let Some(m) = model {
            checks.extend(crate::fit::report::obs_stream_checks(m));
        }
    }

    let warmup = match method {
        "pmmh" | "mh" => {
            let algo = crate::fit::methods::parse_algorithm(method).expect("pmmh and mh are registered");
            crate::fit::report::read_warmup(&dir.join(algo.summary_filename()))
        }
        _ => Vec::new(),
    };
    let read = crate::fit::report::read_findings(dir)?;
    if let Some(detail) = chain_detail.as_mut() {
        detail.source_run = inputs.get("starts_source")
            .and_then(|s| s.get("run_id"))
            .and_then(|v| v.as_str())
            .map(String::from);
    }
    let stored = Some(crate::fit::report::Stored {
        leaf: crate::fit::report::display_path(dir),
        total_seconds: inputs.get("wall_time_seconds").and_then(|v| v.as_f64()),
    });
    Ok(MethodFacts {
        name: resolved.name.clone(),
        method: resolved.method.clone(),
        method_dir: dir.clone(),
        header,
        result,
        checks,
        warmup,
        findings: read.findings,
        findings_unreadable: read.unreadable,
        stored,
        chain_detail,
    })
}

/// The winning method's point estimate (θ̂) as a flat params TOML — the same
/// payload `fit summary --params-only` prints. A `pub(crate)` seam over
/// [`discover_methods`] + [`dump_params_only`] so `compare` can derive a
/// prequential at θ̂ from a sealed fit without touching the private
/// `ResolvedMethod` type. The last completed method leaf's winner.
pub(crate) fn winner_params_toml(segment: &Path) -> Result<String, String> {
    let dir = segment.to_string_lossy();
    let discovered = discover_methods(segment);
    dump_params_only(&dir, &discovered)
}

/// Dump the chosen method's winner params as a flat TOML, pipeable
/// into `camdl pfilter --params`. No header, no metadata, no
/// provenance — just `name = value` lines the standard params loader
/// will accept.
///
/// Reads the last leaf `discover_methods` returned, i.e. the last in
/// declaration order (`FitView.methods_declared`).
fn dump_params_only(
    dir: &str,
    discovered: &[ResolvedMethod],
) -> Result<String, String> {
    let target = discovered
        .iter()
        .next_back()
        .cloned()
        .ok_or_else(|| format!("no completed method leaves found in {}", dir))?;
    let target_method = target.name.clone();
    let method_path = target.method_dir.clone();
    let path = format!("{}/final_params.toml", method_path.to_string_lossy());
    let params = crate::util::load_params_toml(&path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    let mut out = String::new();
    out.push_str(&format!("# camdl fit summary --params-only ({})\n", target_method));
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
    use crate::chain_index::ChainIndex;
    use crate::fit::method_result::{If2Record, If2Run, PgasRun, PmmhRun};
    use crate::fit::report::RefusedChains;
    use crate::fit::report::export::{render_latex_method, render_md_method};
    use crate::fit::state::FitState;
    use crate::fit::report::build::{
        bayesian_method_report, if2_method_report, if2_parameters, if2_provenance, ParameterReport,
    };
    use crate::fit::report::render::seeded_from_line as render_seeded_from_line;
    use crate::fit::method_result::Stat;

    /// gh#663: the chain-count text for a method that refused chains. Nothing
    /// refused reads as the plain count (`None`); a refusal names the ids; a
    /// `n_good_chains` with no refusal record behind it says so rather than
    /// silently picking one source.
    #[test]
    fn refused_chains_ran_of() {
        let none = RefusedChains { records: Vec::new(), unreadable: Vec::new() };
        assert_eq!(none.ran_of(8, None), None);
        let two = RefusedChains {
            records: vec![(ChainIndex(3), "a".into()), (ChainIndex(5), "b".into())],
            unreadable: Vec::new(),
        };
        assert_eq!(two.ran_of(8, Some(6)).as_deref(),
            Some("6 of 8 chains ran — refused at start: 3, 5"));
        // IF2 records no `n_good_chains`; the records alone give the count.
        assert_eq!(two.ran_of(8, None).as_deref(),
            Some("6 of 8 chains ran — refused at start: 3, 5"));
        let s = none.ran_of(8, Some(6)).unwrap();
        assert!(s.starts_with("6 of 8 chains ran")
            && s.contains("2 chain(s) not accounted for"), "{s}");
    }

    /// A method directory holding `diagnostics.json` with `raw` as its body.
    fn method_with_diagnostics(tag: &str, raw: &str) -> std::path::PathBuf {
        let dir = crate::test_support::unique_temp_dir(tag);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diagnostics.json"), raw).unwrap();
        dir
    }

    /// The defect: a fit whose chains froze at different values has R̂ = +∞,
    /// and its `RhatHigh` finding made `diagnostics.json` unreadable — so the
    /// chain the same run refused at its start went unlisted, and the summary
    /// said it could not look. The file here is written by the collector the
    /// samplers use, so it is the file a fit leaves behind.
    #[test]
    fn a_refused_chain_is_listed_beside_an_infinite_rhat() {
        use sim::inference::diagnostic::{DiagnosticCollector, DiagnosticKind};
        let c = DiagnosticCollector::new("pmmh");
        c.push(DiagnosticKind::RhatHigh {
            param: "beta".into(), rhat: f64::INFINITY, threshold: 1.1,
        });
        c.push(DiagnosticKind::BadInit {
            chain_id: 2,
            params: std::collections::BTreeMap::from([("beta".to_string(), 4.8)]),
            reason: "no start of 10 could be scored".into(),
            attempts: Vec::new(),
        });
        let dir = crate::test_support::unique_temp_dir("summary_inf_rhat_refused");
        std::fs::create_dir_all(&dir).unwrap();
        c.write_json(dir.join("diagnostics.json").to_str().unwrap()).unwrap();

        let read = crate::fit::report::read_findings(&dir).unwrap();
        let refused = RefusedChains::from_findings(&read.findings, &read.unreadable);
        let lines = plain_formatter().refused_chains_lines(&refused);
        let ran = refused.ran_of(3, Some(2));
        let _ = std::fs::remove_dir_all(&dir);

        assert!(!lines.contains("could not be read"),
            "a file the current writer produced must read:\n{lines}");
        assert!(lines.contains("refused chain 2: no start of 10 could be scored"),
            "the refused chain is listed:\n{lines}");
        assert_eq!(ran.as_deref(), Some("2 of 3 chains ran — refused at start: 2"));
    }

    /// A `diagnostics.json` written before non-finite numbers were spelled
    /// out carries `null` where its R̂ was `+∞`. That entry cannot be read
    /// honestly — `null` was written for `+∞`, `−∞` and `NaN` alike — so it is
    /// refused by name; but the refusal is the ENTRY's, and the chain the same
    /// file records as refused is still listed. The `null`s an attempt carries
    /// for a summary it could not compute are values, not damage, and must
    /// not be blamed.
    #[test]
    fn a_legacy_null_refuses_its_entry_by_field_and_keeps_the_rest_of_the_file() {
        let raw = r#"[
  {
    "kind": { "type": "rhat_high", "param": "beta", "rhat": null, "threshold": 1.1 },
    "severity": "error",
    "message": "rank-normalized split R̂ for 'beta' is inf (threshold 1.100).",
    "method": "pmmh",
    "timestamp": "2026-10-20T00:00:00Z"
  },
  {
    "kind": { "type": "acceptance_rate_unhealthy", "rate": null, "param": null, "kernel": "random_walk" },
    "severity": "error",
    "message": "chain acceptance rate NaN% is outside healthy range [15%, 50%] (random-walk MH).",
    "method": "pmmh",
    "timestamp": "2026-10-20T00:00:00Z"
  },
  {
    "kind": {
      "type": "bad_init",
      "chain_id": 2,
      "params": { "beta": 4.8 },
      "reason": "no start of 10 could be scored",
      "attempts": [ {
        "stream": "cases", "time": 7.0, "date": null,
        "cell": { "state": "scored", "y_obs": 18.0 },
        "projected_max": null, "projected_median": null,
        "n_projected_zero": 0, "n_projected_nan": 0, "n_neg_inf": 0,
        "neg_inf_causes": [], "n_live": 0, "n_dead": 30, "n_particles": 30
      } ]
    },
    "severity": "error",
    "message": "Chain 2 starting parameters were pathological — skipped.",
    "method": "pmmh",
    "timestamp": "2026-10-20T00:00:00Z"
  }
]"#;
        let dir = method_with_diagnostics("summary_legacy_null", raw);
        let read = crate::fit::report::read_findings(&dir).unwrap();
        let refused = RefusedChains::from_findings(&read.findings, &read.unreadable);
        let lines = plain_formatter().refused_chains_lines(&refused);
        let _ = std::fs::remove_dir_all(&dir);

        assert!(lines.contains("refused chain 2: no start of 10 could be scored"),
            "the refused chain is listed from the entry that does read:\n{lines}");
        assert!(lines.contains("rhat_high") && lines.contains("`rhat`"),
            "the unreadable entry is named, with its field:\n{lines}");
        assert!(lines.contains("written before"),
            "and the reader is told the file predates the fix:\n{lines}");
        assert!(lines.contains("acceptance_rate_unhealthy") && lines.contains("`rate` is null"),
            "every unreadable entry is named, each with its own field:\n{lines}");
        for legit in ["projected_max", "projected_median", "date", "`param`"] {
            assert!(!lines.contains(legit),
                "a null that is a value ({legit}) is not blamed:\n{lines}");
        }
    }
    use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};
    use crate::fit::loglik::LoglikType;
    use crate::fit::method_result::PosteriorDiagnostics;

    /// The text view under test in nearly every case: no calendar, no
    /// `--explain`. Its methods render one section each, as plain text (no
    /// colour, so a block can be compared as text).
    struct TextView {
        cal: CalendarContext,
        explain: bool,
    }

    /// What the provenance section printed, and whether it failed.
    struct Provenance {
        text: String,
        failed: bool,
    }

    impl TextView {
        fn refused_chains_lines(&self, refused: &RefusedChains) -> String {
            text(&refused_lines(refused))
        }
        fn chain_loglik_eval_table(&self, state: &If2Record) -> String {
            text(&if2_chain_eval_table(state))
        }
        fn gate_verdict_block(&self, state: &If2Record) -> String {
            text(&if2_gate_section(state))
        }
        fn parameter_table(&self, state: &If2Record, method_dir: Option<&Path>) -> String {
            text(&if2_parameter_table(state, &if2_parameters(state, method_dir, &self.cal)))
        }
        fn provenance_block(&self, method_dir: &str, state: &If2Record) -> Provenance {
            let prov = if2_provenance(state, method_dir);
            Provenance { text: text(&provenance_section(&prov)), failed: prov.any_failed() }
        }
        fn dt_check_block(&self, dt_check: &super::super::dt_check::DtCheckResult) -> String {
            text(&dt_check_section(dt_check))
        }
        fn seeded_from_line(
            &self,
            source: Option<&str>,
            kind: Option<crate::fit::starts::ChainStartsKind>,
            source_run: Option<&str>,
        ) -> String {
            text(&[render_seeded_from_line(source, kind, source_run)])
        }
        fn method_block(&self, lead: &str, name: &str, method_dir: &str, if2: &If2Run) -> Provenance {
            let report = if2_method_report(name, method_dir, if2, &self.cal);
            Provenance {
                text: text(&if2_block(!lead.is_empty(), &report, if2)),
                failed: report.provenance_failed(),
            }
        }
    }

    fn plain_formatter() -> TextView {
        TextView { cal: CalendarContext::default(), explain: false }
    }

    /// The same view with `--explain` on.
    fn explaining_formatter() -> TextView {
        TextView { cal: CalendarContext::default(), explain: true }
    }

    /// Rendered lines as plain text, each with its newline.
    fn text(lines: &[crate::out::paint::Line]) -> String {
        lines.iter().map(|l| l.text() + "\n").collect()
    }


    /// Everything the renderer put under the `name` section rule, up to the
    /// next rule or the end of the block — the rule's own dash count excluded,
    /// since it is a function of the name's length and carries nothing.
    ///
    /// Comparing a section whole is the point: an alignment defect is a
    /// property of the WHOLE row, and a `contains` on one cell cannot see that
    /// the columns after it moved.
    fn section_of(text: &str, name: &str) -> String {
        let head = format!("── {} ", name);
        let mut out = String::new();
        let (mut inside, mut found) = (false, false);
        for line in text.lines() {
            if line.starts_with("── ") {
                if inside {
                    break;
                }
                inside = line.starts_with(&head);
                found |= inside;
                continue;
            }
            if inside {
                out.push_str(line);
                out.push('\n');
            }
        }
        assert!(found, "no `{name}` section in:\n{text}");
        // The blank line before the next rule belongs to the SEPARATOR, not to
        // the section — every section but the last would otherwise carry one
        // and the expectations would differ for no reason a reader can see.
        match out.strip_suffix("\n\n") {
            Some(trimmed) => format!("{trimmed}\n"),
            None => out,
        }
    }

    /// The `chains` section for a method on disk. `ChainDetail::gather` is asked
    /// for the PGAS reads unconditionally: a non-PGAS fixture writes neither a
    /// `pgas_summary.json` nor a `trajectories.tsv`, so they find nothing and
    /// the section renders exactly as it would for that sampler.
    fn chain_table(
        fmt: &TextView,
        dir: &std::path::Path,
        n_chains: usize,
        kind: LoglikType,
    ) -> String {
        let facts = ChainDetail::gather(dir, kind, true, None, None);
        text(&chains_section(n_chains, &facts, fmt.explain))
    }

    /// Which Bayesian result a [`block`] renders.
    enum BayesianView<'a> {
        Pgas(&'a PgasRun),
        Pmmh(&'a PmmhRun),
    }

    /// `bayesian_block` for a test that cares only about the rendered sections:
    /// no version lead, no resolved forkability.
    fn block(
        fmt: &TextView,
        name: &str,
        method: &str,
        method_dir: &std::path::Path,
        view: BayesianView<'_>,
        ll_kind: LoglikType,
    ) -> String {
        let (result, is_pgas) = match view {
            BayesianView::Pgas(r) => (MethodResult::Pgas(r.clone()), true),
            BayesianView::Pmmh(r) => (MethodResult::Pmmh(r.clone()), false),
        };
        let mut report = bayesian_method_report(name, method, Some(result), &fmt.cal);
        report.chain_detail = Some(ChainDetail::gather(method_dir, ll_kind, is_pgas, None, None));
        text(&bayesian_block(false, &report, fmt.explain))
    }

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
            method: "scout".into(),
            seed: 42,
            timestamp: "2026-04-25T00:00:00Z".into(),
            input_hash: Some("deadbeef".into()),
            camdl_version: Some(version::VERSION_SHORT.into()),
            best_loglik: -3804.9,
            initial_loglik: -7891.0,
            best_chain: ChainIndex(1),
            n_chains: 8,
            n_good_chains: Some(8),
            start_values: start,
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: agreement,
            perturb_only_at_t0_params: vec!["I0".into()],
            chain_logliks: vec![-3810.0; 8],
            chain_eval_logliks: vec![
                -3810.5, -3805.1, -3812.0, -3808.7,
                -3804.9, -3811.2, -3809.0, -3807.6,
            ],
            chain_eval_ids: ChainIndex::all(8).collect(),
            chain_eval_ses: vec![1.5, 1.2, 1.8, 1.4, 1.1, 1.6, 1.3, 1.5],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            chain_starts_kind: Some(crate::fit::starts::ChainStartsKind::Spread),
            dt_check: None,
            pf_noise: None,
            optimizer_convergence: None,
        }
    }

    /// The per-chain loglik-eval table labelled row `i` as chain `i`.
    /// `ChainResults::chain_eval_logliks` sorts by chain id and then discards
    /// it, and IF2 drops PF-degenerate chains before that point, so on a fit
    /// where a chain was dropped the positions are not chain numbers -- and
    /// the `<- selected` marker, an argmax over positions, named the wrong
    /// chain. Contiguous ids hide this, which is why the existing fixture
    /// never caught it.
    #[test]
    fn the_loglik_eval_table_names_chains_not_row_positions() {
        let fmt = plain_formatter();
        let mut state = synthetic_fit_state();
        // Chains 1, 4 and 6 died in the filter and never reached the eval.
        state.chain_eval_ids = [0, 2, 3, 5, 7].map(ChainIndex).to_vec();
        state.chain_eval_logliks = vec![-3810.5, -3812.0, -3808.7, -3804.9, -3807.6];
        state.chain_eval_ses = vec![1.5, 1.8, 1.4, 1.1, 1.5];
        let out = fmt.chain_loglik_eval_table(&If2Record::from_state(&state));

        for id in ["    0 ", "    2 ", "    3 ", "    5 ", "    7 "] {
            assert!(out.contains(id), "surviving chain {id:?} must be named:\n{out}");
        }
        assert!(!out.contains("    4 "), "chain 4 died and must not appear:\n{out}");

        // The best loglik (-3804.9) is chain 5, the FOURTH row. Keyed on
        // position the marker would have said chain 3.
        let marked: Vec<&str> = out.lines().filter(|l| l.contains("selected")).collect();
        assert_eq!(marked.len(), 1, "exactly one chain is selected:\n{out}");
        assert!(marked[0].trim_start().starts_with("5 "),
            "the marker belongs to chain 5, not row position 3:\n{}", marked[0]);
    }

    /// A fit_state written before `chain_eval_ids` existed carries none. The
    /// table is omitted rather than falling back to positions -- the
    /// mislabelling the field exists to prevent.
    #[test]
    fn the_loglik_eval_table_is_omitted_without_ids_rather_than_mislabelled() {
        let fmt = plain_formatter();
        let mut state = synthetic_fit_state();
        state.chain_eval_ids = Vec::new();
        let out = fmt.chain_loglik_eval_table(&If2Record::from_state(&state));
        assert!(out.contains("predates per-chain ids"),
            "the omission is stated and explained:\n{out}");
        assert!(!out.contains("selected"),
            "and no chain is marked on evidence we cannot attribute:\n{out}");
    }

    /// gh#981: the json / md / latex per-chain table is built from the same
    /// `chain_eval_ids` the text table reads, not from row positions, so on a
    /// fit where a chain was dropped the `selected` marker names the chain
    /// whose `chain_N/` directory holds the reported estimate.
    #[test]
    fn the_structured_chain_table_names_chains_not_row_positions() {
        let mut state = synthetic_fit_state();
        // Chains 1, 4 and 6 died in the filter and never reached the eval.
        state.chain_eval_ids = [0, 2, 3, 5, 7].map(ChainIndex).to_vec();
        state.chain_eval_logliks = vec![-3810.5, -3812.0, -3808.7, -3804.9, -3807.6];
        state.chain_eval_ses = vec![1.5, 1.8, 1.4, 1.1, 1.5];
        let report = if2_method_report("scout", "/nonexistent", &If2Run::from_state(Path::new("/nonexistent"), &state, state.n_chains, 0), &CalendarContext::default());
        let ids: Vec<ChainIndex> = report.chains.iter().map(|c| c.chain_id).collect();
        assert_eq!(ids, [0, 2, 3, 5, 7].map(ChainIndex).to_vec(),
            "rows carry the surviving chains' own numbers");
        let selected: Vec<ChainIndex> =
            report.chains.iter().filter(|c| c.is_winner).map(|c| c.chain_id).collect();
        assert_eq!(selected, vec![ChainIndex(5)], "the best loglik (-3804.9) is chain 5, not row 3");
    }

    /// The structured report (`--format json|md|latex`) carries the same IF2
    /// verdict the text block and `fit table` do: an initial-state Â failure
    /// fails the Â leg, and a leaf without clean-eval data still gets an
    /// overall verdict, judged on Â alone, as `fit table`'s `gate_verdict` is.
    #[test]
    fn structured_gate_report_is_the_same_verdict_as_the_table() {
        let mut state = synthetic_fit_state();
        state.tail_chain_agreement =
            [("R0".to_string(), 1.001), ("sigma".to_string(), 1.002), ("I0".to_string(), 16.5)]
                .into_iter().collect();
        let report = if2_method_report("scout", "/nonexistent", &If2Run::from_state(Path::new("/nonexistent"), &state, state.n_chains, 0), &CalendarContext::default());
        let gate = report.gate.expect("an IF2 report has a gate");
        assert!(!gate.a_passes, "the initial-state Â is gated: {gate:?}");
        assert_eq!(gate.max_a_param.as_deref(), Some("I0"));
        assert_eq!(gate.overall_pass, Some(false));

        state.tail_chain_agreement.remove("I0");
        state.chain_eval_logliks.clear();
        state.chain_eval_ses.clear();
        let report = if2_method_report("scout", "/nonexistent", &If2Run::from_state(Path::new("/nonexistent"), &state, state.n_chains, 0), &CalendarContext::default());
        let gate = report.gate.expect("an IF2 report has a gate");
        assert_eq!(gate.db_passes, None, "no clean-eval data, no decibans leg");
        assert_eq!(gate.overall_pass, Some(true),
            "judged on Â alone, as fit table's gate_verdict is");
    }

    /// gh#981, the other half: without the ids the structured table is
    /// omitted, as the text table is, rather than labelled by position.
    #[test]
    fn the_structured_chain_table_is_omitted_without_ids_rather_than_mislabelled() {
        let mut state = synthetic_fit_state();
        state.chain_eval_ids = Vec::new();
        let report = if2_method_report("scout", "/nonexistent", &If2Run::from_state(Path::new("/nonexistent"), &state, state.n_chains, 0), &CalendarContext::default());
        assert!(report.chains.is_empty(),
            "no ids, no rows — a position is not a chain number: {:?}", report.chains);
    }

    #[test]
    fn formatter_renders_pass_verdict_when_thresholds_clear() {
        let state = synthetic_fit_state();
        let fmt = plain_formatter();
        let block = fmt.gate_verdict_block(&If2Record::from_state(&state));

        // Â leg: max = 1.21 on gamma, threshold 1.01 → fail.
        // Spread is small (~7 nats) → decibans leg passes.
        // Overall: FAIL because Â leg fails.
        assert!(block.contains("Â leg:"));
        assert!(block.contains("max Â = 1.210 (gamma)"),
            "expected max Â call-out; got: {}", block);
        assert!(block.contains("decibans leg:"));
        assert!(block.contains("overall:"));
    }

    /// The gate block reads the same assessment the source check does: an
    /// initial-state Â failure fails the Â leg and the overall verdict, and
    /// the dynamics-only sub-verdict is reported beside it.
    #[test]
    fn gate_block_gates_a_t0_parameter_and_reports_the_dynamics_sub_verdict() {
        let mut state = synthetic_fit_state();
        state.tail_chain_agreement =
            [("R0".to_string(), 1.001), ("sigma".to_string(), 1.002), ("I0".to_string(), 16.5)]
                .into_iter().collect();
        let block = plain_formatter().gate_verdict_block(&If2Record::from_state(&state));
        assert!(block.contains("max Â = 16.500 (I0)"), "{block}");
        assert!(block.contains("dynamics only (without perturb_only_at_t0): ✓"), "{block}");
        assert!(block.contains("FAIL"), "{block}");
    }

    /// One chain: no Â and no decibans spread. The block must not say PASS.
    #[test]
    fn a_one_chain_gate_block_says_not_assessed() {
        let mut state = synthetic_fit_state();
        state.n_chains = 1;
        state.tail_chain_agreement.clear();
        state.chain_eval_logliks = vec![-3804.9];
        state.chain_eval_ses = vec![1.0];
        let block = plain_formatter().gate_verdict_block(&If2Record::from_state(&state));
        assert!(block.contains("NOT ASSESSED") && !block.contains("PASS"), "{block}");
    }

    #[test]
    fn formatter_emits_caveat_when_resolved_gate_absent() {
        let mut state = synthetic_fit_state();
        state.resolved_gate = None;
        let fmt = plain_formatter();
        let block = fmt.gate_verdict_block(&If2Record::from_state(&state));
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
        let fmt = plain_formatter();
        let table = fmt.parameter_table(&If2Record::from_state(&state), None);
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
    /// end-of-run block prints "Â unavailable (W ≈ 0) — verdict incomplete". The summary
    /// table compared that NaN to a literal, and every comparison against NaN
    /// is false, so it fell through to ✗ — reporting "this parameter failed"
    /// for a parameter that was never assessed.
    #[test]
    fn a_parameter_with_no_agreement_is_not_reported_as_failing() {
        let mut state = synthetic_fit_state();
        state.tail_chain_agreement.insert("sigma".into(), f64::NAN);
        let fmt = plain_formatter();
        let table = fmt.parameter_table(&If2Record::from_state(&state), None);
        let sigma = table.lines().find(|l| l.contains("sigma")).expect("a sigma row");
        assert!(
            !sigma.contains('✗'),
            "an unassessable Â is not a failure:\n{sigma}"
        );
        assert!(
            sigma.contains("unavailable — verdict incomplete"),
            "and it must say so, and what it does to the verdict:\n{sigma}"
        );
    }

    /// A population-sized estimate must not push its own row's Â out of the
    /// column every other row's sits in.
    ///
    /// `{:<12.6}` bounds the decimals but not the integer part, so a value at
    /// or above `1e6` renders fourteen characters into a twelve-wide field and
    /// shifts everything after it on that row alone. Every value in this
    /// repository's fixtures is a single digit or smaller, which is exactly why
    /// the overflow survived: it is invisible until a parameter is a
    /// population, a case count, or a seed size.
    #[test]
    fn a_population_sized_estimate_does_not_shift_the_agreement_column() {
        let mut state = synthetic_fit_state();
        state.start_values.insert("N".into(), 2_400_000.0);
        state.tail_chain_agreement.insert("N".into(), 1.02);
        let fmt = plain_formatter();
        let table = fmt.parameter_table(&If2Record::from_state(&state), None);
        let cols: Vec<usize> = table
            .lines()
            .filter(|l| l.contains("Â="))
            .map(|l| l.split("Â=").next().unwrap().chars().count())
            .collect();
        assert_eq!(cols.len(), 4, "one row per estimated parameter:\n{table}");
        assert!(
            cols.iter().all(|c| *c == cols[0]),
            "every Â starts in the same column, whatever the magnitudes: {cols:?}\n{table}"
        );
        assert!(
            table.contains("2400000.000000"),
            "and the value is printed whole — a number cannot be ellipsized \
             the way a name can:\n{table}"
        );
    }

    #[test]
    fn parameter_table_filters_to_estimated_params() {
        let state = synthetic_fit_state();
        let fmt = plain_formatter();
        let block = fmt.parameter_table(&If2Record::from_state(&state), None);
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
        let fmt = plain_formatter();
        let prov = fmt.provenance_block(&dir.to_string_lossy(), &If2Record::from_state(&state));
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
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = PgasRun {
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
        let out = block(&fmt, "pgas", "pgas", no_traces, BayesianView::Pgas(&r),
            crate::fit::loglik::LoglikType::CompleteData);
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
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let mk = |ess: BTreeMap<String, f64>| PgasRun {
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
        let out = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&gapped), LoglikType::CompleteData);
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
            out.contains("\n      tau\n"),
            "the withholding parameter is named on its own line, not left to the \
             table's dashes: {out}"
        );

        // Control: with `tau`'s ESS present the same block reports both ratios —
        // 145 / 500 raw iters and 145 / 11.8 s off the slower of the two.
        let complete = mk(BTreeMap::from([("a2".to_string(), 145.0), ("tau".to_string(), 300.0)]));
        let ok = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&complete), LoglikType::CompleteData);
        assert!(
            ok.contains("ESS/iter = 0.290  (min-param ESS 145 / 500 raw sampling iters)"),
            "a complete map still reports ESS/iter off the slowest param: {ok}"
        );
        assert!(
            ok.contains("ESS/sec  = 12.3"),
            "a complete map still reports ESS/sec: {ok}"
        );
        assert!(
            !ok.contains("not reportable"),
            "the withholding branch must not fire on a complete map: {ok}"
        );
    }

    /// Both efficiency ratios must still carry information on a badly-mixing
    /// fit — the one whose reader most needs them.
    ///
    /// Each is an ESS over a count that grows as the fit is pushed harder, so
    /// both shrink exactly when mixing is worst. A min-parameter ESS of 8 over
    /// 48,000 raw sampling iterations is `1.67e-4`, which `{:.3}` printed as
    /// `0.000`; over a twenty-minute run it is `6.67e-3` per second, which
    /// `{:.2}` printed as `0.01`. Two fits an order of magnitude apart then
    /// render identically, and one of them renders as nothing at all.
    #[test]
    fn the_efficiency_headline_stays_informative_at_a_tiny_ess() {
        use std::collections::BTreeMap;
        let diag = |n_samples: usize| PosteriorDiagnostics {
            per_param: crate::fit::method_result::per_param_from_maps(
                BTreeMap::from([("beta".to_string(), 2.639)]),
                BTreeMap::from([("beta".to_string(), 8.0)]),
                BTreeMap::new(),
            ),
            n_samples,
            thin: 1,
            wall_time_secs: Some(1200.0),
            n_chains: 4,
        };

        // 8 / 48,000 iterations and 8 / 1,200 s: three significant figures at
        // both magnitudes, where fixed decimals had none and one.
        let out = text(&efficiency_lines(&diag(48_000)));
        assert!(
            out.contains("ESS/iter = 0.000167  (min-param ESS 8 / 48000 raw sampling iters)"),
            "the per-iteration ratio keeps its digits at 1.67e-4:\n{out}"
        );
        assert!(
            out.contains("ESS/sec  = 0.00667  (min-param ESS 8 / 1200.0s wall)"),
            "and so does the per-second ratio at 6.67e-3:\n{out}"
        );

        // Ten times longer, ten times worse: the two must be distinguishable,
        // and below 1e-4 the fixed form is mostly leading zeros, so it switches
        // to scientific rather than growing them.
        let worse = text(&efficiency_lines(&diag(480_000)));
        assert!(
            worse.contains("ESS/iter = 1.67e-5"),
            "an order of magnitude worse reads as an order of magnitude worse:\n{worse}"
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
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = PgasRun {
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
        let out = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&r),
            LoglikType::CompleteData);
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
        let fmt = plain_formatter();
        let mk = |wall: Option<f64>, n_samples: usize, thin: usize| PgasRun {
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
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        // min-param ESS (145) / wall (11.8 s) = 12.3 ESS/sec — thinning-invariant.
        let with = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&mk(Some(11.8), 500, 1)), LoglikType::CompleteData);
        assert!(
            with.contains("ESS/sec  = 12.3"),
            "must report ESS/sec off the slowest param (145/11.8): {with}"
        );
        // ESS/iteration = 145 / (n_samples 500 × thin 1) = 0.290, per raw sampling step.
        assert!(
            with.contains("ESS/iter = 0.290"),
            "must report ESS/iteration off raw sampling steps (145/500): {with}"
        );
        // Thinning-invariance: (n_samples 50 × thin 10) is the SAME 500 raw steps,
        // so ESS/iter is identical — the whole point.
        let thinned = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&mk(Some(5.0), 50, 10)), LoglikType::CompleteData);
        assert!(
            thinned.contains("ESS/iter = 0.290"),
            "ESS/iter must be invariant to thinning (50×10 == 500 raw): {thinned}"
        );
        // No wall-time (older run) → no ESS/sec line, but ESS/iter still shows
        // (it needs only n_samples×thin, not wall-time).
        let without = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&mk(None, 500, 1)), LoglikType::CompleteData);
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
    /// summary. Six-chain method dir (five near -50, chain 5 stuck at -300) with
    /// per-chain traces + a draws.tsv manifest → the table flags chain 5 and the
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
        // MAD is non-zero; chain 5 is the lone stuck chain at ≈ -300.
        write_trace(0, "3\t-50.0\t-52.0\n4\t-50.0\t-52.0\n5\t-50.0\t-52.0\n"); // mean -50.0
        write_trace(1, "3\t-50.5\t-52.0\n4\t-50.5\t-52.0\n5\t-50.5\t-52.0\n"); // mean -50.5
        write_trace(2, "3\t-49.5\t-51.0\n4\t-49.5\t-51.0\n5\t-49.5\t-51.0\n"); // mean -49.5
        write_trace(3, "3\t-50.2\t-52.0\n4\t-50.2\t-52.0\n5\t-50.2\t-52.0\n"); // mean -50.2
        write_trace(4, "3\t-49.8\t-51.0\n4\t-49.8\t-51.0\n5\t-49.8\t-51.0\n"); // mean -49.8
        write_trace(5, "3\t-300.0\t-302.0\n4\t-301.0\t-303.0\n5\t-299.0\t-301.0\n"); // stuck -300
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..6 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 6, LoglikType::Marginal);
        assert!(table.contains("── chains "), "the section rule is present:\n{table}");
        assert!(table.contains("← outlier"), "stuck chain must be flagged:\n{table}");
        assert!(table.contains("chains disagree (chain 5"), "nudge must name chain 5:\n{table}");
        // The stuck chain's mean must reflect the post-burn-in draws (≈ -300),
        // not the stripped warm-up (≈ -900).
        assert!(table.contains("-300.00"), "chain 5 mean ≈ -300 (warm-up stripped):\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667: a PGAS method's per-chain table ranks on `obs_ll`
    /// (`log p(y | X, θ)`), not on the `log_complete_data_ll` target. The
    /// fixture is built so the two rankings disagree: chain 2 has a hugely
    /// concentrated latent path (a high complete-data value) with an ordinary
    /// data fit, chain 5 fits the data ≈450 nats worse with an ordinary
    /// complete-data value. The table must flag chain 5 and leave chain 2 clean,
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
            let cd = dir.join(format!("chain_{i}"));
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

        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 6, LoglikType::CompleteData);

        // The flag lands on the chain that reproduces the data worst…
        let flagged: Vec<&str> = table
            .lines()
            .filter(|l| l.contains("← outlier"))
            .map(|l| l.split_whitespace().next().unwrap_or(""))
            .collect();
        assert_eq!(flagged, vec!["5"],
            "only chain 5 (the bad DATA fit) may be flagged:\n{table}");
        assert!(table.contains("chains disagree (chain 5"),
            "the nudge must name chain 5:\n{table}");

        // …and the ranked column is obs_ll, not the complete-data target.
        assert!(table.contains("-1400.00"),
            "chain 5's scored value is its obs_ll (-1400.00):\n{table}");
        assert!(table.contains("-952.00"),
            "chain 0's scored value is its obs_ll (-952.00):\n{table}");

        // The complete-data target and its latent-path term stay VISIBLE — they
        // are the sampler's own objective, just not the ranking key.
        assert!(table.contains("-4400.00"),
            "chain 5's complete-data target must still be shown:\n{table}");
        assert!(table.contains("-1751.70"),
            "chain 2's complete-data target must still be shown:\n{table}");
        assert!(table.contains("-800.00"),
            "chain 3's transition_ll must still be shown:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#667 must not cost gh#608/gh#635 their reach. A PGAS method whose
    /// traces predate the `obs_ll` column cannot be compared chain-to-chain —
    /// the table says so by name rather than falling back to the position-1
    /// column — but the degeneracy screens read `log_posterior` and
    /// `draws.tsv`, so a −inf chain must still be named.
    #[test]
    fn missing_obs_ll_still_runs_the_degeneracy_screen() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_no_obs");
        std::fs::create_dir_all(&dir).unwrap();
        for c in 0..3 {
            let cd = dir.join(format!("chain_{c}"));
            std::fs::create_dir_all(&cd).unwrap();
            // No obs_ll / transition_ll columns at all. Chain 2 is degenerate.
            let rows = if c == 2 {
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

        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 3, LoglikType::CompleteData);
        assert!(table.contains("no `obs_ll` column"),
            "the missing comparison column is named, not silently substituted:\n{table}");
        assert!(!table.contains("← outlier"),
            "nothing may be ranked without the comparison column:\n{table}");
        assert!(table.contains("DEGENERATE") && table.contains("chain 2"),
            "the -inf screen still names chain 2:\n{table}");

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
        write_trace(0, "2\t-50.0\t-52.0\n3\t-50.5\t-52.5\n4\t-49.5\t-51.5\n");
        write_trace(1, "2\t-50.2\t-52.2\n3\t-50.1\t-52.1\n4\t-49.9\t-51.9\n");
        // Chain 2: stuck — two of three retained rows carry -inf as the
        // CURRENT state (the gh#607 shape).
        write_trace(2, "2\t-inf\t-inf\n3\t-inf\t-inf\n4\t-60.0\t-62.0\n");
        let mut draws = String::from("chain\tdraw\tbeta\n");
        for c in 0..3 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.5\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();

        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 3, LoglikType::Marginal);
        assert!(table.contains("-inf"),
            "the stuck chain's mean renders -inf, not the missing-data dash:\n{table}");
        assert!(table.contains("DEGENERATE"),
            "the stuck chain gets the loud flag:\n{table}");
        assert!(table.contains("chain 2") && table.contains("66.7%")
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
        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 3, LoglikType::Marginal);
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
        for c in 0..3 {
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

        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 3, LoglikType::Marginal);
        assert!(table.contains("point-mass"),
            "the frozen chain gets the loud flag:\n{table}");
        assert!(table.contains("chain 2") && table.contains("across 3 retained"),
            "the flag names the chain and the draw count:\n{table}");
        assert!(table.contains("--exclude-chains"),
            "the flag names the explicit fix:\n{table}");
        // Exactly one flagged chain — mixing chains stay clean.
        assert_eq!(table.matches("point-mass").count(), 1,
            "only the frozen chain is flagged:\n{table}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// gh#406 negative control: a well-mixed six-chain method flags no outlier and
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

        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 6, LoglikType::Marginal);
        assert!(table.contains("── chains "), "the section rule is present:\n{table}");
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

    /// gh#406: a method with no per-chain traces says so rather than silently
    /// omitting the section (no-silent-gap).
    #[test]
    fn bayesian_chain_loglik_table_reports_unavailable() {
        let dir = crate::test_support::unique_temp_dir("summary_chain_diag_none");
        std::fs::create_dir_all(&dir).unwrap();
        let fmt = plain_formatter();
        let table = chain_table(&fmt, &dir, 4, LoglikType::Marginal);
        assert!(table.contains("per-chain traces unavailable"),
            "must say traces are unavailable, not skip:\n{table}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── gh#727: saved-vs-forkable latent paths in the summary ────────────

    /// Write a `pgas_summary.json` carrying just the gh#727 `trajectories`
    /// block — the only key `saved_path_table` reads. `per_chain` rows are
    /// `(chain id, written, forkable)`; the id is written because a method that
    /// refused a chain at its start has survivors whose ids are not `1..n`.
    fn write_traj_summary(
        dir: &std::path::Path,
        draw_stride: u64,
        thin: u64,
        per_chain: &[(u64, u64, u64)],
    ) {
        let summary = serde_json::json!({
            "method": "pgas",
            "thin": thin,
            "trajectories": {
                "draw_stride": draw_stride,
                "thin": thin,
                "chain_id": per_chain.iter().map(|(c, _, _)| *c).collect::<Vec<_>>(),
                "n_saved": per_chain.iter().map(|(_, a, _)| *a).collect::<Vec<_>>(),
                "n_forkable": per_chain.iter().map(|(_, _, b)| *b).collect::<Vec<_>>(),
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
        write_traj_summary(&dir, 28, 5, &[(1, 250, 50), (2, 250, 250)]);
        let fmt = plain_formatter();
        let out = chain_table(&fmt, &dir, 2, LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CHAINS), concat!(
"  (per-chain traces unavailable \u{2014} cannot break down by chain)\n",
"  chain   paths   forkable\n",
"      1     250         50\n",
"      2     250        250\n",
"  200 of 500 written paths cannot be joined to a posterior draw\n",
"  a path is written on every 28 retained draw (thin 5), so every one should\n",
"  be joinable; the shortfall is paths that were written outside this leaf's\n",
"  retained set or skipped as incoherent records\n",
        ), "the counts survive the absence of the traces, and the shortfall is named");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Negative control, and the ordinary case: every written path landed on a
    /// retained draw, so the counts render and nothing claims a shortfall.
    #[test]
    fn saved_path_table_is_quiet_when_every_path_is_forkable() {
        let dir = crate::test_support::unique_temp_dir("summary_gh727_clean");
        std::fs::create_dir_all(&dir).unwrap();
        write_traj_summary(&dir, 36, 5, &[(1, 200, 200), (2, 200, 200)]);
        let fmt = plain_formatter();
        let out = chain_table(&fmt, &dir, 2, LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CHAINS), concat!(
"  (per-chain traces unavailable \u{2014} cannot break down by chain)\n",
"  chain   paths   forkable\n",
"      1     200        200\n",
"      2     200        200\n",
        ), "the counts render and NOTHING claims a shortfall this method \
            does not have");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A method whose summary carries no `trajectories` block (PMMH / MH, or a
    /// PGAS run predating gh#727) renders nothing rather than a row of dashes.
    #[test]
    fn saved_path_table_is_empty_without_the_block() {
        let dir = crate::test_support::unique_temp_dir("summary_gh727_absent");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("pgas_summary.json"), r#"{"method":"pgas","thin":5}"#).unwrap();
        let fmt = plain_formatter();
        let out = chain_table(&fmt, &dir, 2, LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CHAINS),
"  (per-chain traces unavailable \u{2014} cannot break down by chain)\n",
            "no `trajectories` block means no path columns and no note about \
             them \u{2014} there is nothing this method failed to report");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ── the `seeded from` header (proposal 2026-09-08 §3.4) ──────────────

    /// Every leaf says where its chains began and whether they began apart;
    /// a leaf written before the fields existed says `unknown`, never a guess.
    #[test]
    fn seeded_from_line_names_the_rule_and_its_kind() {
        use crate::fit::starts::ChainStartsKind;
        let fmt = plain_formatter();
        assert_eq!(
            fmt.seeded_from_line(Some("from_mle @mle"), Some(ChainStartsKind::Point), None),
            "  seeded from:  from_mle @mle (every chain at one point)\n"
        );
        assert_eq!(
            fmt.seeded_from_line(Some("from_prior"), Some(ChainStartsKind::Spread), None),
            "  seeded from:  from_prior (one independent draw per chain)\n"
        );
        assert_eq!(fmt.seeded_from_line(Some("lhs"), None, None), "  seeded from:  lhs\n");
        assert_eq!(fmt.seeded_from_line(None, None, None), "  seeded from:  unknown\n");
        // gh#896: a stored-fit source names the run it read beside the handle.
        assert_eq!(
            fmt.seeded_from_line(
                Some("from_mle @mle"), Some(ChainStartsKind::Point), Some(&"ab12".repeat(16))),
            "  seeded from:  from_mle @mle, run ab12ab12ab12 (every chain at one point)\n"
        );
    }

    // ── the redesigned text layout ───────────────────────────────────────
    //
    // Every case here compares a rendered SECTION whole rather than probing it
    // with `contains`. The defects this layout exists to fix are properties of
    // the whole block — what comes first, what order the rows are in, which
    // columns sit beside which — and none of them is visible one cell at a
    // time.

    /// A PGAS method whose parameters differ in R̂, in both halves of it, and in
    /// per-chain mixing — the shape every table below is read against.
    ///
    /// Alphabetically the order is `alpha, beta, gamma, tau`; by R̂ it is
    /// `tau, beta, alpha, gamma`, so a table that lost the sort cannot pass by
    /// coincidence. `beta` is the one parameter whose FOLDED half is the larger
    /// — the chains agree on where it sits and disagree on its spread — which
    /// is the distinction the `worse` column exists to carry.
    fn layout_pgas_result() -> PgasRun {
        use crate::fit::method_result::ParamConvergence;
        use std::collections::BTreeMap;
        let scored = |rhat: f64, bulk: f64, folded: f64, classic: f64,
                      ess: f64, tail: f64, per_chain: Vec<f64>| {
            ParamConvergence::Scored {
                rhat: Stat::Value(rhat),
                rhat_bulk: Stat::Value(bulk),
                rhat_folded: Stat::Value(folded),
                rhat_classic: Stat::Value(classic),
                ess_bulk: Stat::Value(ess),
                ess_tail: Stat::Value(tail),
                ess_per_chain: per_chain,
                all_chains_frozen: false,
                rhat_withheld: None,
            }
        };
        PgasRun {
            diagnostics: PosteriorDiagnostics {
                per_param: BTreeMap::from([
                    ("alpha".to_string(),
                        scored(1.412, 1.412, 1.109, 2.870, 41.0, 88.0, vec![39.0, 4.0, 38.0, 26.0])),
                    ("beta".to_string(),
                        scored(2.104, 1.330, 2.104, 9.512, 12.0, 19.0, vec![9.0, 3.0, 40.0, 7.0])),
                    ("gamma".to_string(),
                        scored(1.008, 1.008, 1.002, 1.011, 1450.0, 1610.0, vec![380.0, 402.0, 331.0, 337.0])),
                    ("tau".to_string(),
                        scored(4.870, 4.870, 4.524, 28.662, 8.0, 10.0, vec![39.0, 4.0, 38.0, 26.0])),
                ]),
                n_samples: 4000,
                thin: 1,
                wall_time_secs: Some(200.0),
                n_chains: 4,
            },
            posterior_mean: BTreeMap::from([
                ("alpha".to_string(), 0.235_134),
                ("beta".to_string(), 6.178_211),
                ("gamma".to_string(), 0.480_575),
                ("tau".to_string(), 240.759_84),
            ]),
            posterior_q025: BTreeMap::new(),
            posterior_q975: BTreeMap::new(),
            acceptance_per_param: BTreeMap::new(),
        }
    }

    /// The verdict comes before every table, and the sections come in reading
    /// order: the estimates, then what qualifies them.
    ///
    /// This is the whole point of the redesign. Convergence detail used to
    /// open the block — sixty-five lines of it on the ebola province fit — and
    /// the posterior summary came last, so the number a reader opened the
    /// summary for was line 147 of 182.
    #[test]
    fn the_verdict_comes_first_and_the_sections_in_reading_order() {
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = layout_pgas_result();
        let out = block(&fmt, "posterior", "pgas", no_traces,
            BayesianView::Pgas(&r), LoglikType::CompleteData);

        let rules: Vec<&str> = out
            .lines()
            .filter(|l| l.starts_with("── "))
            .map(|l| l.split(" ─").next().unwrap().trim_start_matches("── "))
            .collect();
        assert_eq!(rules, vec![SECTION_POSTERIOR, SECTION_CONVERGENCE, SECTION_CHAINS,
            SECTION_LATENT]);

        // The verdict is above the first rule, and carries the parameter that
        // set it — a reader who stops after two lines has the answer and knows
        // where to look for the rest.
        let head: Vec<&str> = out.lines().take_while(|l| !l.starts_with("── ")).collect();
        assert_eq!(head, vec![
            "  posterior · pgas · 4 chains · 4000 draws",
            // No fit_state.toml under the fixture: the header says so rather
            // than guessing a rule.
            "  seeded from:  unknown",
            "",
            // 4.870 is past the severe band, so the verdict names what that
            // means rather than leaving the glyph to carry it.
            "  ✗ NOT converged — the chains are in different places",
            "    max R̂ 4.870 (tau), threshold 1.05",
            "    3 of 4 parameters above threshold",
            "",
        ]);
    }

    /// The posterior table is ordered by R̂ descending, so the parameter that
    /// failed is the first row rather than wherever the alphabet put it.
    #[test]
    fn the_posterior_table_is_ordered_by_rhat_not_by_name() {
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = layout_pgas_result();
        let out = block(&fmt, "posterior", "pgas", no_traces,
            BayesianView::Pgas(&r), LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_POSTERIOR), concat!(
"  param                    mean   ESS bulk   ESS tail        R̂\n",
"  tau                     240.8          8         10    4.870\n",
"  beta                    6.178         12         19    2.104\n",
"  alpha                  0.2351         41         88    1.412\n",
"  gamma                  0.4806       1450       1610    1.008\n",
        ));
    }

    /// The posterior table's header sits over the columns it names, `R̂`
    /// included.
    ///
    /// `{:>8}` pads to a `char` count, and `R̂` is `R` followed by U+0302
    /// COMBINING CIRCUMFLEX ACCENT — two `char`s in one terminal cell — so that
    /// header cell was seven cells wide against a column of eight-cell numbers,
    /// and the label sat one cell left of what it labels. Measured in cells,
    /// not in `char`s: a `chars().count()` check reports the broken header as
    /// the same width as its rows, which is exactly how this survived.
    #[test]
    fn the_posterior_header_sits_over_its_columns_in_display_cells() {
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = layout_pgas_result();
        let out = block(&fmt, "posterior", "pgas", no_traces,
            BayesianView::Pgas(&r), LoglikType::CompleteData);
        let table = section_of(&out, SECTION_POSTERIOR);
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines.len() > 1, "header plus at least one row:\n{table}");
        let widths: Vec<usize> = lines.iter().map(|l| display_width(l)).collect();
        assert!(
            widths.iter().all(|w| *w == widths[0]),
            "the header is the same number of cells wide as its rows: {widths:?}\n{table}"
        );
        // The premise, asserted so the check above cannot go vacuous: the
        // header is the line whose `char` count overstates the space it takes.
        let header = lines[0];
        assert_eq!(
            header.chars().count(), widths[0] + 1,
            "the accent makes the header one `char` wider than it prints:\n{header}"
        );
    }

    /// The convergence table carries both halves of R̂, the classic statistic,
    /// which half is worse, and the per-chain ESS — in the same order the
    /// posterior table used, so the two can be read across.
    ///
    /// `beta` is the row that matters: its folded half is the larger, so the
    /// chains agree on where it sits and disagree on its spread. That is a
    /// different problem with a different fix, and only this column says so.
    #[test]
    fn the_convergence_table_names_which_half_is_worse_per_parameter() {
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = layout_pgas_result();
        let out = block(&fmt, "posterior", "pgas", no_traces,
            BayesianView::Pgas(&r), LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CONVERGENCE), concat!(
"  param              bulk    folded   classic   worse  per-chain ESS\n",
"  tau               4.870     4.524    28.662   bulk   39 4 38 26\n",
"  beta              1.330     2.104     9.512   folded 9 3 40 7\n",
"  alpha             1.412     1.109     2.870   bulk   39 4 38 26\n",
"  gamma             1.008     1.002     1.011   bulk   380 402 331 337\n",
"  ESS/iter = 0.00200  (min-param ESS 8 / 4000 raw sampling iters)\n",
"  ESS/sec  = 0.0400  (min-param ESS 8 / 200.0s wall)\n",
        ));
    }

    /// The per-chain log-likelihood and the saved-latent-path counts are ONE
    /// table. They are keyed on the same thing — the chain — and answer halves
    /// of one question: which chain disagrees, and how many forkable paths came
    /// from it. They were two tables, several lines apart, for no reason but
    /// the order they were written in.
    #[test]
    fn the_chains_table_carries_the_path_counts_in_the_same_row() {
        let dir = crate::test_support::unique_temp_dir("summary_chains_merged");
        std::fs::create_dir_all(&dir).unwrap();
        // (transition_ll, obs_ll); chain 2 fits the data far worse.
        let chains = [(-2832.6, -952.0), (-2933.0, -952.8), (-3000.0, -1400.0)];
        for (i, (trans, obs)) in chains.iter().enumerate() {
            let cd = dir.join(format!("chain_{i}"));
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
        for c in 0..3 {
            for d in 0..3 {
                draws.push_str(&format!("{c}\t{d}\t0.{c}{d}\n"));
            }
        }
        std::fs::write(dir.join("draws.tsv"), draws).unwrap();
        write_traj_summary(&dir, 40, 1, &[(0, 10, 10), (1, 10, 8), (2, 10, 10)]);

        let fmt = plain_formatter();
        let out = chain_table(&fmt, &dir, 3, LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CHAINS), concat!(
"  chain         obs ll    mod-z   transition ll    complete ll   paths   forkable   flag\n",
"      0        -952.00     0.67        -2832.60       -3784.60      10         10   \n",
"      1        -952.80     0.00        -2933.00       -3885.80      10          8   \n",
"      2       -1400.00  -377.05        -3000.00       -4400.00      10         10   ← outlier\n",
"  2 of 30 written paths cannot be joined to a posterior draw\n",
"  a path is written on every 40 retained draw (thin 1), so every one should\n",
"  be joinable; the shortfall is paths that were written outside this leaf's\n",
"  retained set or skipped as incoherent records\n",
"  ⚠ chains disagree (chain 2 — obs_ll far from the rest) — is a parameter\n",
"  weakly identified? Inspect its per-chain posterior.\n",
        ));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A method that saves no latent paths — PMMH here, and MH the same — has
    /// no `latent paths` section at all, rather than one saying it has nothing.
    /// The remaining three still read as a layout.
    #[test]
    fn a_method_without_pgas_has_no_latent_path_section() {
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let pgas = layout_pgas_result();
        let r = PmmhRun {
            diagnostics: pgas.diagnostics.clone(),
            posterior_mean: pgas.posterior_mean.clone(),
            acceptance_rate: 0.234,
            acceptance_per_chain: BTreeMap::new(),
            map_loglik: -1234.5,
            pf_noise: None,
        };
        let out = block(&fmt, "posterior", "pmmh", no_traces,
            BayesianView::Pmmh(&r), LoglikType::Marginal);

        let rules: Vec<&str> = out
            .lines()
            .filter(|l| l.starts_with("── "))
            .map(|l| l.split(" ─").next().unwrap().trim_start_matches("── "))
            .collect();
        assert_eq!(rules, vec![SECTION_POSTERIOR, SECTION_CONVERGENCE, SECTION_CHAINS],
            "no `latent paths` rule for a sampler that writes none:\n{out}");
        assert!(!out.contains("chains saved paths"),
            "and nothing on the identity line claims any:\n{out}");
        assert!(out.contains("loglik -1234.5 at the highest log-posterior draw \
                              (natural-scale prior, no Jacobian)"),
            "the identity line still carries what this sampler does report:\n{out}");
        // The acceptance rate belongs with the other efficiency numbers.
        assert!(section_of(&out, SECTION_CONVERGENCE)
            .contains("  acceptance = 0.234 (mean across chains)\n"));
    }

    /// The parameter legend is off by default and complete under
    /// `--parameters` — symbol, name, prose and citation, exactly as the model
    /// documents them.
    #[test]
    fn the_parameter_legend_is_absent_by_default_and_whole_under_the_flag() {
        use std::collections::BTreeMap;
        let docs: BTreeMap<String, ir::parameter::DocBlock> = BTreeMap::from([
            ("p_fatal".to_string(), ir::parameter::DocBlock {
                symbol: Some("p_f".to_string()),
                text: Some("The infection fatality ratio.".to_string()),
                reference: Some("MacNeil et al. 2010, doi:10.3201/eid1612.100627".to_string()),
            }),
            ("sigma".to_string(), ir::parameter::DocBlock {
                symbol: None,
                text: Some("Latent progression.".to_string()),
                reference: None,
            }),
        ]);
        assert_eq!(parameter_legend(&docs, false), "",
            "the legend is scholarly reference material, not the answer the \
             command is run for");
        assert_eq!(parameter_legend(&docs, true), concat!(
"\n  parameters\n",
"    p_f  p_fatal  —  The infection fatality ratio.  [MacNeil et al. 2010, doi:10.3201/eid1612.100627]\n",
"    sigma  —  Latent progression.\n",
        ));
        // A model that documents nothing has no legend to gate.
        assert_eq!(parameter_legend(&BTreeMap::new(), true), "");
    }

    /// A model's constants are listed under their own heading, every one of
    /// them with its value and kind, its prose when it has any, and only
    /// under `--parameters`.
    #[test]
    fn the_constant_legend_lists_every_constant_under_the_flag() {
        use std::collections::BTreeMap;
        let constants = vec![
            ir::parameter::Constant { name: "omega".into(), value: 0.01,
                                      kind: ir::parameter::ParamKind::Rate },
            ir::parameter::Constant { name: "iota".into(), value: 1e-6,
                                      kind: ir::parameter::ParamKind::Count },
        ];
        let docs: BTreeMap<String, ir::parameter::DocBlock> = BTreeMap::from([
            ("omega".to_string(), ir::parameter::DocBlock {
                symbol: Some("ω".to_string()),
                text: Some("Waning of immunity.".to_string()),
                reference: None,
            }),
        ]);
        assert_eq!(constant_legend(&constants, &docs, false), "");
        assert_eq!(constant_legend(&constants, &docs, true), concat!(
"\n  constants (typed `let`s: part of the model, not parameters)\n",
"    ω  omega = 0.01 (rate)  —  Waning of immunity.\n",
"    iota = 0.000001 (count)\n",
        ));
        assert_eq!(constant_legend(&[], &docs, true), "");
    }

    /// `--explain` defines each term exactly ONCE, under the section it is
    /// primarily read in, and adds nothing anywhere else.
    ///
    /// R̂ is a column in three sections; if each explained it, the flag would
    /// add more noise than the legend it replaced. `posterior` therefore
    /// carries no prose of its own — every column it has is defined under
    /// `convergence`.
    #[test]
    fn explain_defines_each_term_once_where_it_is_read() {
        let fmt = explaining_formatter();
        let plain = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = layout_pgas_result();
        let view = || BayesianView::Pgas(&r);
        let with = block(&fmt, "posterior", "pgas", no_traces, view(),
            LoglikType::CompleteData);
        let without = block(&plain, "posterior", "pgas", no_traces, view(),
            LoglikType::CompleteData);

        // Each definition appears once in the whole output, and each under its
        // own section — never repeated where the same term recurs as a column.
        for (section, text) in EXPLAIN {
            let first = wrap(text, SECTION_WIDTH - 6).remove(0);
            assert_eq!(with.matches(first.as_str()).count(), 1,
                "`{section}` prose must appear exactly once in the whole \
                 output:\n{with}");
            assert!(section_of(&with, section).contains(&first),
                "`{section}` prose must sit under `{section}`:\n{with}");
            assert!(!without.contains(first.as_str()),
                "and nothing of it may leak into the default output:\n{without}");
        }

        // `posterior` defines nothing of its own — R̂, ESS bulk and ESS tail are
        // all defined under `convergence`, one section below.
        assert_eq!(section_of(&with, SECTION_POSTERIOR),
            section_of(&without, SECTION_POSTERIOR),
            "the flag must not add prose to a section that defines no term");

        // The flag only ADDS. Every line of the default output survives it.
        for line in without.lines().filter(|l| !l.trim().is_empty()) {
            assert!(with.contains(line), "`--explain` may not remove `{line}`");
        }
    }

    /// Under `--exclude-chains` the `chains` table and the outlier count keep
    /// their FULL scope, and both say so.
    ///
    /// R̂, ESS and the posterior means above them are recomputed over the
    /// retained chains; the per-chain table is read from the traces on disk,
    /// which the selection does not filter, and the modified z-score is scaled
    /// by all of them. Listing every chain is deliberate — inspecting the
    /// chain you dropped is usually why you dropped it — but standing
    /// unlabelled beside recomputed numbers it reads as though it shared their
    /// scope. That is the silent-wrong reading this labelling exists to stop.
    #[test]
    fn an_excluded_chain_still_appears_and_the_scope_is_named() {
        let dir = crate::test_support::unique_temp_dir("summary_excl_scope");
        std::fs::create_dir_all(&dir).unwrap();
        for (i, ll) in [-50.0_f64, -50.2, -300.0].iter().enumerate() {
            let cd = dir.join(format!("chain_{i}"));
            std::fs::create_dir_all(&cd).unwrap();
            let mut body = String::from("sweep\tlog_likelihood\tlog_posterior\n");
            for sw in 0..5 {
                body.push_str(&format!("{sw}\t{ll}\t{}\n", ll - 1.0));
            }
            std::fs::write(cd.join("trace.tsv"), body).unwrap();
        }
        let info = SubsetInfo {
            excluded: vec![ChainIndex(2)],
            kept: vec![ChainIndex(0), ChainIndex(1)],
            n_total: 3,
        };
        let facts = ChainDetail::gather(&dir, LoglikType::Marginal, false, Some(info), None);
        let section = text(&chains_section(2, &facts, false));

        assert!(section.contains("including the excluded 2"),
            "the table must say whose rows it is still showing:\n{section}");
        assert!(section.contains("the outlier score is over all of them"),
            "and that the score was not recomputed over the subset:\n{section}");
        assert!(section.lines().any(|l| l.trim_start().starts_with("2 ")),
            "chain 2 keeps its row — dropping it would hide the evidence for \
             the exclusion:\n{section}");
        assert!(!section.contains("found traces for 3 of 2 chains"),
            "and the unfiltered/retained mismatch must not render as a count \
             that reads like a defect:\n{section}");

        // Without a selection the same method says none of it.
        let facts = ChainDetail::gather(&dir, LoglikType::Marginal, false, None, None);
        let plain = text(&chains_section(3, &facts, false));
        assert!(!plain.contains("excluded"), "no scope note when nothing was \
            excluded:\n{plain}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The two flags are independent: together the output carries the legend
    /// AND each section's prose, and offers neither back.
    #[test]
    fn the_two_flags_compose() {
        use std::collections::BTreeMap;
        let docs: BTreeMap<String, ir::parameter::DocBlock> = BTreeMap::from([
            ("sigma".to_string(), ir::parameter::DocBlock {
                symbol: None,
                text: Some("Latent progression.".to_string()),
                reference: None,
            }),
        ]);
        let fmt = explaining_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let r = layout_pgas_result();
        let out = format!("{}{}", parameter_legend(&docs, true),
            block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&r),
                LoglikType::CompleteData));

        assert!(out.contains("    sigma  —  Latent progression.\n"),
            "the legend is present:\n{out}");
        for (_, text) in EXPLAIN {
            let first = wrap(text, SECTION_WIDTH - 6).remove(0);
            assert_eq!(out.matches(first.as_str()).count(), 1,
                "each section's prose is present, once:\n{out}");
        }
    }

    // ── the rendered grid, against adversarial input ─────────────────────
    //
    // Both defects these pin are INPUT-dependent: on a corpus of short names
    // and contiguous chains the old code rendered correctly, so reading the
    // format strings could not show either. The fixture therefore carries what
    // a real fit carries — a stratified name past the old column width, means
    // six orders of magnitude apart, and a chain set whose surviving ids are
    // not `1..n` — and the rendered text is compared whole.

    fn wide_range_pgas_result() -> PgasRun {
        use std::collections::BTreeMap;
        PgasRun {
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
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let out = block(&fmt, "posterior", "pgas", no_traces,
            BayesianView::Pgas(&wide_range_pgas_result()), LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_POSTERIOR), concat!(
"  param                         mean   ESS bulk   ESS tail        R\u{302}\n",
"  I0_ituri                     240.8          9         21    3.481\n",
"  kappa                     0.001854          9         17    3.412\n",
"  phi_split_haut_uele          59.68         28        317    1.194\n",
"  iota                      1.000e-6       1200       1400    1.010\n",
"  N0_haut_uele               6.322e6       1500       1800    1.002\n",
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
        let fmt = plain_formatter();
        let no_traces = std::path::Path::new("/nonexistent/method_dir");
        let out = block(&fmt, "posterior", "pgas", no_traces, BayesianView::Pgas(&r),
            LoglikType::CompleteData);
        let table = section_of(&out, SECTION_POSTERIOR);
        for line in table.lines().skip(1) {
            assert_eq!(line.chars().count(), 2 + NAME_COL_MAX + 1 + 14 + 1 + 10 + 1 + 10 + 1 + 8,
                "every row is the same width: {line:?}");
        }
        assert!(table.contains("incidence_reporting_p\u{2026}"),
            "the head is what gives way:\n{table}");
        assert!(table.contains("_zone_ituri "),
            "the ituri row must still be identifiable by its suffix:\n{table}");
        assert!(table.contains("_zone_nord_kivu "),
            "the nord_kivu row must still be identifiable by its suffix:\n{table}");
    }

    /// gh#727 follow-up: the `chain` column of the saved-path table is the
    /// CHAIN, not the row number.
    ///
    /// The two coincide whenever every chain survives its start, which is why
    /// the defect was invisible on the test corpus. Here chains 4, 7, 9 and 13
    /// are the survivors (gh#607 refused the rest), so a positional index
    /// would print 1..4 and send a reader inspecting "chain 1" to a chain that
    /// contributed nothing — while the latent-path block a few lines below
    /// prints the true ids.
    #[test]
    fn saved_path_table_names_the_chain_not_the_row() {
        let dir = crate::test_support::unique_temp_dir("summary_saved_path_ids");
        std::fs::create_dir_all(&dir).unwrap();
        write_traj_summary(&dir, 600, 1,
            &[(4, 10, 10), (7, 10, 10), (9, 10, 10), (13, 10, 10)]);
        let fmt = plain_formatter();
        let out = chain_table(&fmt, &dir, 4, LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CHAINS), concat!(
"  (per-chain traces unavailable \u{2014} cannot break down by chain)\n",
"  chain   paths   forkable\n",
"      4      10         10\n",
"      7      10         10\n",
"      9      10         10\n",
"     13      10         10\n",
        ), "the ids are the artifact\'s own, never 1..4");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A `trajectories` block written before the chain ids were recorded
    /// cannot say which chain a row is, and the row's position is not an
    /// answer. It is read as no report at all rather than rendered against a
    /// positional guess, so an existing fit loses this table until it is
    /// re-run. That is the deliberate cost of never printing a wrong chain id;
    /// pinned here so no positional fallback creeps back in.
    #[test]
    fn saved_path_table_is_empty_when_the_block_has_no_chain_ids() {
        let dir = crate::test_support::unique_temp_dir("summary_saved_path_no_ids");
        std::fs::create_dir_all(&dir).unwrap();
        let summary = serde_json::json!({
            "method": "pgas",
            "thin": 1,
            "trajectories": {
                "draw_stride": 600,
                "thin": 1,
                "n_saved": [10, 10],
                "n_forkable": [10, 10],
            },
        });
        std::fs::write(dir.join("pgas_summary.json"),
            serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        let fmt = plain_formatter();
        let out = chain_table(&fmt, &dir, 2, LoglikType::CompleteData);
        assert_eq!(section_of(&out, SECTION_CHAINS),
"  (per-chain traces unavailable \u{2014} cannot break down by chain)\n",
            "no positional fallback may fill the columns");
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
        let fmt = plain_formatter();
        let prov = fmt.provenance_block(&dir.to_string_lossy(), &If2Record::from_state(&state));
        assert!(!prov.failed, "must not flag when params match: {}", prov.text);
        assert!(prov.text.contains("✓"));

        std::fs::remove_dir_all(&dir).ok();
    }

    // ── Phase 4 / 5 tests ─────────────────────────────────────────

    /// Build a fit dir with one method at the v2 layout
    /// (`<dir>/real/fit_1/<method>/`), and write the top-level + method
    /// `run.json` files so `fit_tree::walk_fit_dir` finds it. Mirror
    /// of what `camdl fit run` produces, minus the bits these tests
    /// don't need.
    fn make_fit_dir(method: &str, state: &FitState, params: &[(&str, f64)])
        -> std::path::PathBuf
    {
        let dir = crate::test_support::unique_temp_dir(&format!("summary_format_{method}"));
        std::fs::create_dir_all(&dir).unwrap();
        let parent_hash: String = "deadbeef".repeat(8);
        write_top_level_fit_run(&dir, &parent_hash);

        let method_dir = dir.join("real").join("fit_1").join(method);
        std::fs::create_dir_all(&method_dir).unwrap();
        state.save(&method_dir.to_string_lossy()).unwrap();
        write_method_run(&method_dir, &parent_hash, method, crate::run_meta::FitAlgorithm::If2);

        // final_params.toml + mle_params.toml carrying matching values
        // so the provenance cross-check passes.
        let mut body = String::new();
        for (k, v) in params {
            body.push_str(&format!("{} = {}\n", k, v));
        }
        std::fs::write(method_dir.join("final_params.toml"), &body).unwrap();
        std::fs::write(method_dir.join("mle_params.toml"), &body).unwrap();
        dir
    }

    /// Write the fit-level provenance sidecar at the segment root so
    /// `FitView::read` treats `dir` as a well-formed fit. `parent_hash` is the
    /// `fit`-level hash the method leaves also carry.
    fn write_top_level_fit_run(dir: &std::path::Path, _parent_hash: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join("fit.meta.json"),
            r#"{"model_path":"sir.camdl","model_identity":"f00d","fit_toml_path":"fit.toml","estimated":["R0","sigma","gamma"]}"#,
        )
        .unwrap();
    }

    /// Write a `FitMethod` `runid::RunRecord` leaf for `method` under `method_dir`.
    /// `parent_hash` seeds the shared `fit`-level hash; the `inputs` carry the
    /// per-leaf numbers (`method`, `n_chains`, `best_loglik`, …) the views
    /// project. The `method` LEVEL label is the leaf's name, and it is the
    /// same string as `inputs.method`.
    fn write_method_run(method_dir: &std::path::Path, parent_hash: &str, name: &str, method: crate::run_meta::FitAlgorithm) {
        std::fs::create_dir_all(method_dir).unwrap();
        let method_label = name.to_string();
        let fit_hash: String = parent_hash.chars().cycle().take(64).collect();
        let run_id: String = format!("{}-{}", parent_hash, name)
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
            "kind": "fit_method",
            "run_id": run_id,
            "hash_version": 1,
            "ir_version": "0.7",
            "engine_version": "0.1.0+test",
            "levels": [
                {"name": "fit",   "label": "fit",  "hash": fit_hash, "schema_version": 1},
                {"name": "method", "label": method_label, "hash": "1fb03eee00000000000000000000000000000000000000000000000000000000", "schema_version": 1},
                {"name": "seed",  "label": "seed_1","hash": "06cbd6b300000000000000000000000000000000000000000000000000000000", "schema_version": 1}
            ],
            "status": "completed",
            "artifacts": {},
            "inputs": {
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
        std::fs::write(method_dir.join("run.json"), serde_json::to_string(&rec).unwrap()).unwrap();
    }

    /// JSON output is parseable, `schema.version` is the current one, and the
    /// per-method report fields match the `FitState` we constructed it from.
    /// Catches any future schema rename / removal that would break the book
    /// pipeline.
    ///
    /// gh#901: the per-method array is `methods`, and `methods` is absent — a
    /// consumer keyed on the old word must break loudly at the tag, not read
    /// an array that quietly stopped being there.
    ///
    /// v3: `method_progression` is gone. A Δ against the leaf listed before
    /// this one compared runs whose log-likelihoods may be of different kinds.
    #[test]
    fn json_format_round_trips_and_carries_schema_version() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("if2", &state, &params);

        let methods = discover_methods(&dir);
        let doc = build_summary_doc(&dir.to_string_lossy(), &methods, None, Painter::plain());
        let json = serde_json::to_string_pretty(&doc).unwrap();
        assert!(json.contains("\"version\": 4"),
            "schema.version must be present and = 4: {}", json);
        // Reparse and pin the load-bearing fields.
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["schema"]["version"], 4);
        assert_eq!(parsed["fit_dir"], dir.to_string_lossy().as_ref());
        assert_eq!(parsed["methods"][0]["name"], "if2");
        assert!(parsed.get("stages").is_none(),
            "the old `stages` key must be gone, not kept beside `methods`: {json}");
        assert!(parsed["methods"][0].get("method_progression").is_none(),
            "no method is compared against the leaf listed before it: {json}");
        assert!((parsed["methods"][0]["best_loglik"].as_f64().unwrap() - (-3804.9)).abs() < 1e-6);
        // Heuristic block is namespaced.
        assert!(parsed["methods"][0]["_heuristic"]["overall_status"].is_string());
        // Provenance keys present.
        let prov = &parsed["methods"][0]["provenance"];
        assert_eq!(prov["final_params_matches_mle_params"], true);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Markdown output is well-formed: headings, gate-verdict table,
    /// parameter table. Spot-check critical lines.
    #[test]
    fn markdown_format_renders_gate_table_and_params() {
        let state = synthetic_fit_state();
        let params = [("R0", 56.0_f64), ("sigma", 0.08), ("gamma", 0.08)];
        let dir = make_fit_dir("if2", &state, &params);

        let methods = discover_methods(&dir);
        let doc = build_summary_doc(&dir.to_string_lossy(), &methods, None, Painter::plain());
        let md = render_markdown(&doc);
        assert!(md.contains("# Fit summary:"));
        assert!(md.contains("## `if2`"));
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
        let dir = make_fit_dir("if2", &state, &params);

        let methods = discover_methods(&dir);
        let doc = build_summary_doc(&dir.to_string_lossy(), &methods, None, Painter::plain());
        let tex = render_latex(&doc);
        // No preamble, but tabular blocks per method.
        assert!(tex.contains("\\subsection*{Method:"));
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
        let dir = make_fit_dir("if2", &state, &params);

        let methods = discover_methods(&dir);
        let s = dump_params_only(&dir.to_string_lossy(), &methods).unwrap();
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
    fn params_only_errors_when_no_completed_method() {
        let dir = crate::test_support::unique_temp_dir("summary_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let methods = discover_methods(&dir);
        let err = dump_params_only(&dir.to_string_lossy(), &methods).unwrap_err();
        assert!(err.contains("no completed method leaves"), "{err}");
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
        let fmt = plain_formatter();
        let block = fmt.dt_check_block(&dt_check_pass());
        assert!(block.contains("PASS"), "must call out pass verdict: {}", block);
        // No ladder rows on PASS — keep summary terse.
        assert!(!block.contains("dt = 0.0500"),
            "should not include ladder rows on pass: {}", block);
    }

    #[test]
    fn dt_check_fail_includes_ladder_and_synth_recovery_note() {
        let fmt = plain_formatter();
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
        let fmt = TextView { cal: instant_cal(), explain: false };
        let block = fmt.parameter_table(&If2Record::from_state(&state), None);
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
        let fmt = plain_formatter();
        let block = fmt.parameter_table(&If2Record::from_state(&state), None);
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
        write_outlier_draws(&dir, 4, 40, 3); // chains 0..3; chain 3 stuck

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

        // Recompute over the subset (drop chain 3).
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
        let sel = ChainSelection::parse_exclude("3").unwrap();
        let info = recompute_over_subset(&mut diag, &mut mean, &dir, &sel).unwrap();

        assert_eq!(info.kept, vec![ChainIndex(0), ChainIndex(1), ChainIndex(2)]);
        assert_eq!(info.excluded, vec![ChainIndex(3)]);
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
        MethodResult::Pgas(PgasRun {
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
        MethodResult::Pmmh(PmmhRun {
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
            acceptance_per_chain: BTreeMap::new(),
            map_loglik: -3801.2,
            pf_noise: None,
        })
    }

    fn synthetic_if2_result() -> MethodResult {
        MethodResult::If2(If2Run {
            best_loglik: -3804.9,
            best_chain: ChainIndex(1),
            theta_hat: BTreeMap::new(),
            max_chain_agreement: Some(1.04),
            gate_verdict: crate::fit::method_result::GateVerdict::Pass,
            ess_at_mle: None,
            n_chains: 8,
            n_iter: 50,
            record: If2Record::from_state(&synthetic_fit_state()),
        })
    }

    /// gh#280: every `MethodReport` JSON object carries `loglik_type`,
    /// derived from the typed `method_result`. Fails on the pre-gh#280 code
    /// (the struct had no such field). The PGAS value MUST read
    /// `complete_data` even though its `best_loglik` is null — that is the
    /// joint-vs-marginal distinction an agent scrapes.
    #[test]
    fn method_report_json_carries_loglik_type() {
        let cal = CalendarContext::default();

        let pgas = bayesian_method_report("pgas", "pgas", Some(synthetic_pgas_result()), &cal);
        let j = serde_json::to_value(&pgas).unwrap();
        assert_eq!(j["loglik_type"], serde_json::json!("complete_data"),
            "PGAS method must tag its joint loglik: {j}");
        assert!(j["best_loglik"].is_null(),
            "PGAS has no scalar best_loglik, yet still carries loglik_type: {j}");

        let pmmh = bayesian_method_report("pmmh", "pmmh", Some(synthetic_pmmh_result()), &cal);
        let j = serde_json::to_value(&pmmh).unwrap();
        assert_eq!(j["loglik_type"], serde_json::json!("marginal"),
            "PMMH MAP loglik is marginal: {j}");

        // IF2 via the dedicated builder. No method dir on disk → the
        // provenance reads tolerate absence; loglik_type comes from the
        // typed result.
        let if2 = if2_method_report(
            "scout", "/nonexistent/method",
            &match synthetic_if2_result() { MethodResult::If2(r) => r, _ => unreachable!() }, &cal,
        );
        let j = serde_json::to_value(&if2).unwrap();
        assert_eq!(j["loglik_type"], serde_json::json!("if2"),
            "IF2 method must tag its marginal: {j}");
    }

    /// gh#280: human headlines label the loglik class, with the tag rendered
    /// *after* the number so a `loglik=<num>` scraper stops before it.
    #[test]
    fn headlines_label_the_loglik_class() {
        let cal = CalendarContext::default();
        let fmt = plain_formatter();

        // Terminal IF2 headline.
        let block = fmt.method_block(
            "", "scout", "/nonexistent/method",
            &If2Run::from_state(Path::new("/nonexistent/method"), &synthetic_fit_state(), 4, 0),
        );
        let line = block.text.lines().find(|l| l.contains("best loglik:"))
            .expect("IF2 block has a best-loglik headline");
        assert!(line.contains("(if2)"), "IF2 headline must carry (if2): {line}");
        let num_at = line.find(|c: char| c == '-' || c.is_ascii_digit()).unwrap();
        let tag_at = line.find("(if2)").unwrap();
        assert!(tag_at > num_at, "type tag must render after the number: {line}");

        // Markdown + LaTeX exports of a PMMH (marginal) method.
        let pmmh = bayesian_method_report("pmmh", "pmmh", Some(synthetic_pmmh_result()), &cal);
        let md = render_md_method(&pmmh);
        assert!(md.contains("- best loglik:") && md.contains("(marginal)"),
            "md PMMH headline carries (marginal): {md}");
        let tex = render_latex_method(&pmmh);
        assert!(tex.contains("Best log-likelihood") && tex.contains("(marginal)"),
            "latex PMMH headline carries (marginal): {tex}");
    }

    // ─── Unreadable method leaves (gh#905) ────────────────────────────

    fn leaf(name: &str) -> ResolvedMethod {
        ResolvedMethod {
            name: name.to_string(),
            method: "pgas".to_string(),
            method_dir: PathBuf::from(format!("/nonexistent/{name}")),
        }
    }

    fn failed(name: &str) -> LoadFailure {
        LoadFailure {
            leaf: format!("/nonexistent/{name}"),
            reason: "pgas_summary.json: parse error".to_string(),
        }
    }

    /// gh#905. The exit rule, at the boundary that matters: a summary with no
    /// leaf behind it is a failure, a summary missing one of several leaves is
    /// not.
    ///
    /// The mixed case is pinned here rather than end to end because a single
    /// `fit run` produces one method leaf per segment, so a segment holding a
    /// readable leaf beside an unreadable one is not something the e2e fixture
    /// can build. Exiting non-zero there would be the worse regression of the
    /// two: it would fail a summary that answered the question for every leaf
    /// it could read.
    #[test]
    fn only_a_summary_with_no_leaf_behind_it_is_a_failure() {
        let one = [leaf("a")];
        let three = [leaf("a"), leaf("b"), leaf("c")];

        assert!(nothing_loaded(&one, &[failed("a")]),
            "the one leaf the handle named failed — there is no summary");
        assert!(nothing_loaded(&three, &[failed("a"), failed("b"), failed("c")]),
            "every leaf failed — there is no summary");

        assert!(!nothing_loaded(&three, &[failed("b")]),
            "two of three leaves rendered; the warning names the third");
        assert!(!nothing_loaded(&three, &[failed("a"), failed("c")]),
            "one leaf rendered, so the summary still answers the question");
        assert!(!nothing_loaded(&one, &[]),
            "nothing failed");

        // A segment with no completed leaf at all is a different report — it
        // prints "(no completed methods found)" and is not a load failure.
        assert!(!nothing_loaded(&[], &[]),
            "an empty segment is not an unreadable one");
    }
}
