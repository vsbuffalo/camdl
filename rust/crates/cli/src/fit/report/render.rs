//! A method's report as lines ([`render_report`]), section by section, and the
//! fit-level lines `fit summary` puts around its methods. Never writes: the
//! caller hands the lines to the stream's painter.

use std::collections::BTreeMap;

use sim::inference::diagnostic::{fmt_rate, AcceptanceKernel};

use super::build::{MethodOutcome, MethodReport, ParameterReport, ProvenanceReport};
use super::{ChainDetail, ChainWarmup, RefusedChains, RunHeader};
use crate::fit::chain_diagnostics::ChainLoglikOutcome;
use crate::fit::config_v2::GateConfig;
use crate::fit::method_result::{If2Record, If2Run, MethodResult, NloptRun, PfilterRun};
use crate::fit::method_result::{PosteriorDiagnostics, Stat};
use crate::out::paint::{Line, Span, Style};

/// The acceptance table at the end of a PMMH or `mh` run: one row per chain
/// in chain order, its acceptance rate against the kernel's healthy band, and
/// beside it the scale its draws were proposed under and how many warm-up
/// steps the scale sat at its floor. A chain with no warm-up record (nothing
/// adapted) shows `—` there.
pub fn acceptance_table(
    rates: &BTreeMap<usize, f64>,
    warmup: &[ChainWarmup],
    kernel: AcceptanceKernel,
) -> Vec<Line> {
    let (lo, hi) = kernel.healthy_band();
    let mut lines = vec![
        Line::default(),
        Line(vec![Span::plain("acceptance rates:")]),
        Line(vec![Span::plain(format!(
            "  {:<5}  {:>10} {:<10}  {:>16}  {:>14}",
            "chain", "acceptance", "", "λ end of warm-up", "steps at floor"))]),
    ];
    for (&chain, &rate) in rates {
        let (glyph, style) = if rate < lo {
            ("✗ too low", Style::Err)
        } else if rate > hi {
            ("~ high", Style::Warn)
        } else {
            ("✓", Style::Ok)
        };
        let (lambda, floor) = match warmup.iter().find(|w| w.chain == chain) {
            Some(w) => (format!("{:.3e}", w.final_scale),
                format!("{} / {}", w.steps_at_scale_floor, w.warmup_steps)),
            None => ("—".to_string(), "—".to_string()),
        };
        lines.push(Line(vec![
            Span::plain(format!("  {:<5}  {:>10} ", chain, fmt_rate(rate))),
            Span::styled(format!("{glyph:<10}"), style),
            Span::plain(format!("  {lambda:>16}  {floor:>14}")),
        ]));
    }
    lines
}

// ─── Lines, spans and the shared section furniture ────────────────────────────

/// A line of one plain span.
fn plain(text: impl Into<String>) -> Line {
    Line(vec![Span::plain(text)])
}

/// A line of `indent` followed by `text` in `style`.
fn indented(indent: &str, text: impl Into<String>, style: Style) -> Line {
    Line(vec![Span::plain(indent), Span::styled(text, style)])
}

/// `text` right-aligned in a field `width` characters wide, in `style`: the
/// padding is plain, so it is measured on the text and not on any escape
/// code the painter adds.
fn right_styled(text: &str, width: usize, style: Style) -> Vec<Span> {
    vec![
        Span::plain(" ".repeat(width.saturating_sub(text.chars().count()))),
        Span::styled(text, style),
    ]
}

/// A section rule, `── name ─────…` to [`SECTION_WIDTH`], after a blank line.
///
/// One horizontal character and nothing else. A box — corners, verticals,
/// junctions — has to stay aligned to read as a box, and a terminal narrower
/// than the table breaks it into rubble; a rule that is simply too long just
/// wraps, and the section is still findable.
pub(crate) fn section_rule(name: &str) -> Vec<Line> {
    let used = 3 + name.chars().count() + 1;
    vec![
        Line::default(),
        Line(vec![
            Span::styled("──", Style::Dim),
            Span::plain(" "),
            Span::styled(name, Style::Bold),
            Span::plain(" "),
            Span::styled("─".repeat(SECTION_WIDTH.saturating_sub(used)), Style::Dim),
        ]),
    ]
}

/// A note under a table, wrapped to the section width and indented with the
/// rows, in `style`.
///
/// Diagnostics here run to two hundred characters — the shortfall
/// explanation, the outlier nudge, the reason a column is missing — and an
/// unwrapped one is folded by the terminal at whatever column it happens to
/// be, mid-word, which is how a finding stops being read.
pub(crate) fn note(style: Style, text: &str) -> Vec<Line> {
    wrap(text, SECTION_WIDTH - 2).into_iter().map(|l| indented("  ", l, style)).collect()
}

/// The `--explain` block for `section`, or nothing — because the flag is
/// off, or because this section defines no term of its own.
///
/// Rendered dim and indented past the table's own rows, so it reads as
/// subordinate to the numbers rather than as another row of them.
pub(crate) fn explain(on: bool, section: &str) -> Vec<Line> {
    if !on {
        return Vec::new();
    }
    let Some((_, text)) = EXPLAIN.iter().find(|(k, _)| *k == section) else {
        return Vec::new();
    };
    let mut out = vec![Line::default()];
    out.extend(wrap(text, SECTION_WIDTH - 6).into_iter().map(|l| indented("      ", l, Style::Dim)));
    out
}

/// The lines of a block some other module renders as text, each a plain line.
fn text_lines(text: &str) -> Vec<Line> {
    text.lines().map(plain).collect()
}

// ─── The report, rendered whole ───────────────────────────────────────────────

/// Which sections a view shows. Each section is one function; a view is a
/// list of them, so two views share every section they both show.
#[derive(Clone, Copy, Debug)]
pub enum View {
    /// `fit summary`'s text.
    Summary(SummaryView),
    /// The block `fit run` ends a method with, on stderr.
    RunEnd,
}

/// The choices `fit summary`'s text makes per method.
#[derive(Clone, Copy, Debug, Default)]
pub struct SummaryView {
    /// `--explain`: each section's terms defined under its table.
    pub explain: bool,
    /// The camdl version opens this method's identity line: it is a fact
    /// about the fit, said once, on the first method.
    pub version_lead: bool,
}

/// One method's outcome as lines. Pure: reads only the outcome.
pub fn render_report(outcome: &MethodOutcome, view: View) -> Vec<Line> {
    match (view, outcome) {
        (View::Summary(v), MethodOutcome::Completed(report)) => match &report.method_result {
            Some(MethodResult::If2(if2)) => if2_block(v.version_lead, report, if2),
            Some(MethodResult::Pgas(_) | MethodResult::Pmmh(_)) => {
                bayesian_block(v.version_lead, report, v.explain)
            }
            Some(MethodResult::Nlopt(r)) => nlopt_block(v.version_lead, report, r),
            Some(MethodResult::Pfilter(r)) => pfilter_block(v.version_lead, r),
            None => Vec::new(),
        },
        (View::Summary(v), MethodOutcome::Failed { header, findings, error, .. }) => {
            failed_block(v.version_lead, header, findings, error)
        }
        (View::RunEnd, MethodOutcome::Completed(report)) => run_end_block(report),
        // The run's opening block is on screen already and the error is the
        // command's to print (`Out::error`): what is left is the findings the
        // run recorded before it stopped.
        (View::RunEnd, MethodOutcome::Failed { findings, .. }) => super::findings::render(
            findings,
            sim::inference::diagnostic::HintContext { chains_completed: None },
        ),
    }
}

// ─── The end of `fit run` ─────────────────────────────────────────────────────

/// What `fit run` prints when a method completes: for a Bayesian method the
/// verdict as `fit summary` gives it, the R̂ legend, the acceptance table and
/// the R̂ table; for PGAS then its path-renewal, filter-ESS, latent-path and
/// cross-chain blocks; for every method its findings; for PMMH, `mh` and PGAS
/// the cost of sampling and the best draw's log-likelihood. The `stored`
/// milestone follows it (`Out::milestone`).
fn run_end_block(report: &MethodReport) -> Vec<Line> {
    use sim::inference::diagnostic::HintContext;
    let default_detail = ChainDetail::default();
    let detail = report.chain_detail.as_ref().unwrap_or(&default_detail);
    let refused = RefusedChains::from_findings(&report.findings, &report.findings_unreadable);
    let mut out = Vec::new();
    let completed = match &report.method_result {
        Some(MethodResult::Pmmh(r)) => {
            out.extend(verdict_section(&r.diagnostics, detail));
            out.extend(rhat_legend());
            out.extend(acceptance_table(&r.acceptance_per_chain, &report.warmup, crate::fit::pmmh::THETA_MOVE_KERNEL));
            if r.diagnostics.n_chains > 1 {
                out.extend(rhat_run_table(&r.diagnostics));
            }
            r.diagnostics.n_chains.saturating_sub(refused.records.len())
        }
        Some(MethodResult::Pgas(r)) => {
            out.extend(verdict_section(&r.diagnostics, detail));
            out.extend(rhat_legend());
            out.extend(pgas_acceptance_table(r));
            if r.diagnostics.n_chains > 1 {
                out.extend(rhat_run_table(&r.diagnostics));
            }
            out.extend(pgas_run_blocks(detail));
            r.diagnostics.n_chains.saturating_sub(refused.records.len())
        }
        Some(MethodResult::If2(r)) => r.record.n_good_chains.unwrap_or(r.record.n_chains),
        Some(MethodResult::Nlopt(r)) => r.n_chains,
        Some(MethodResult::Pfilter(_)) | None => 0,
    };
    out.extend(super::findings::render(&report.findings, HintContext { chains_completed: Some(completed) }));
    if let Some(MethodResult::Pmmh(r)) = &report.method_result {
        if let Some(cost) = &r.cost {
            out.push(Line::default());
            out.push(plain(crate::fit::pmmh::completion_line(&report.method, report.method == "mh", cost)));
        }
        if let Some(chain) = r.map_chain {
            out.push(plain(format!(
                "  loglik at the highest log-posterior draw (natural-scale prior, no Jacobian): {:.1} (chain {})",
                r.map_loglik, chain)));
        }
    }
    if let Some(MethodResult::Pgas(r)) = &report.method_result {
        if let Some(secs) = r.sampling_seconds {
            out.push(Line::default());
            out.push(plain(format!("{} complete in {:.1}s", report.method, secs)));
        }
        if let (Some(chain), Some(ll)) = (r.best_chain, r.best_loglik) {
            out.push(plain(format!("  best complete-data ll: {:.1} (chain {})", ll, chain)));
        }
    }
    out
}

/// PGAS's blocks between its R̂ table and its findings, each from the record
/// the run wrote: trajectory renewal by tenth of the series and the filter's
/// ESS (`pgas_summary.json`), latent-path convergence over the saved paths
/// (`chain_*/trajectories.tsv`), and the cross-chain compatibility numbers
/// (`cross_chain_compat.json`). A block whose record is absent is not drawn;
/// latent-path convergence that could not be computed says why.
fn pgas_run_blocks(detail: &ChainDetail) -> Vec<Line> {
    let mut out = Vec::new();
    if let Some(pr) = &detail.path_renewal {
        out.extend(text_lines(&pr.report()));
    }
    if let Some(fe) = &detail.filter_ess {
        out.extend(text_lines(&fe.report()));
    }
    let not_computed = |e: &dyn std::fmt::Display| plain(format!("  latent-path convergence not computed: {e}"));
    match &detail.latent {
        Some(Ok(lc)) => out.extend(text_lines(&lc.report())),
        Some(Err(e)) => out.push(not_computed(e)),
        // No chain saved a path: the reason the computation over none gives.
        None => match crate::fit::latent_convergence::latent_convergence(&[], &[]) {
            Err(e) => out.push(not_computed(&e)),
            Ok(lc) => out.extend(text_lines(&lc.report())),
        },
    }
    if let Some(cc) = &detail.cross_chain {
        out.extend(text_lines(&cc.report()));
    }
    out
}

/// What the four R̂ thresholds mean, said once under the verdict, so the
/// `threshold 1.05` above the tables and the `threshold 1.100` in the findings
/// read as two bars with two jobs rather than as a contradiction.
pub(crate) fn rhat_legend() -> Vec<Line> {
    use crate::fit::method_result::RHAT_CONVERGED_THRESHOLD;
    use crate::fit::runner::RHAT_REPORT_THRESHOLD;
    use sim::inference::diagnostic::CONVERGENCE_ERROR_SEVERITY;
    vec![
        plain(format!(
            "  R̂: below {RHAT_CONVERGED_THRESHOLD} counts as converged; above {RHAT_REPORT_THRESHOLD} \
             draws a finding, a warning up to {CONVERGENCE_ERROR_SEVERITY}")),
        plain("  and an error above it."),
    ]
}

/// A statistic as the run's R̂ table prints it: `∞` as `inf`, an undefined
/// value as `NaN`.
fn stat_f64(s: Stat) -> f64 {
    match s {
        Stat::Value(v) => v,
        Stat::Infinite => f64::INFINITY,
        Stat::Undefined => f64::NAN,
    }
}

/// The R̂ table at the end of a run: one row per parameter with its R̂, the
/// certification glyph and its ESS against the draws; under any parameter not
/// certified converged, the two halves of its R̂ and each chain's ESS.
///
/// The glyph is the certification band `fit summary` uses (`RhatBand`), never
/// greener than the finding `RHAT_REPORT_THRESHOLD` draws. When the run's
/// shape withholds R̂ — every chain started at one point (gh#890) — the table
/// says so and shows the ESS alone: no number, no glyph, no decomposition.
pub(crate) fn rhat_run_table(diag: &PosteriorDiagnostics) -> Vec<Line> {
    use crate::fit::method_result::{ParamConvergence, RhatBand, RHAT_CONVERGED_THRESHOLD};
    use crate::fit::runner::RHAT_REPORT_THRESHOLD;
    let withheld = diag.per_param.values().find_map(|p| match p {
        ParamConvergence::Scored { rhat_withheld: Some(e), .. } => Some(e),
        _ => None,
    });
    let mut out = vec![Line::default(), plain(match withheld {
        Some(why) => format!("R̂ — not assessed: {why}. ESS:"),
        None => format!(
            "R̂ (rank-normalized split, Vehtari et al. 2021) / ESS — R̂ threshold {RHAT_CONVERGED_THRESHOLD}:"),
    })];
    for (name, p) in &diag.per_param {
        let ParamConvergence::Scored { rhat, ess_bulk, ess_tail, rhat_withheld, all_chains_frozen, .. } = p else {
            let why = p.why_no_rhat().unwrap_or_default();
            out.push(plain(format!("  {:12} not reported — {}", name, why)));
            continue;
        };
        let ess = stat_f64(*ess_bulk);
        let tail = match ess_tail.finite() {
            Some(t) => format!("{t:.0}"),
            None => "—".to_string(),
        };
        let share = if diag.n_samples == 0 { 0.0 } else { ess / diag.n_samples as f64 };
        let ess_text = format!("ESS bulk={:.0} tail={} ({:.1}% of {} draws)",
            ess, tail, 100.0 * share, diag.n_samples);
        if rhat_withheld.is_some() {
            out.push(plain(format!("  {:12} {}", name, ess_text)));
            continue;
        }
        // Every chain frozen at its own value has no finite R̂, and the
        // summary records it as that refusal rather than as a number: the
        // worst band, with no number to print.
        let (r_text, band, drew_finding) = match (*rhat, *all_chains_frozen) {
            (Stat::Undefined, true) => ("—".to_string(), RhatBand::Severe, true),
            (stat, _) => {
                let r = stat_f64(stat);
                (format!("{r:.3}"), RhatBand::of(r), r > RHAT_REPORT_THRESHOLD)
            }
        };
        let glyph = match band.glyph_with_finding(drew_finding) {
            "✓" => Span::styled("✓", Style::Ok),
            "~" => Span::styled("~", Style::Warn),
            "✗" => Span::styled("✗", Style::Err),
            other => Span::plain(other),
        };
        out.push(Line(vec![
            Span::plain(format!("  {:12} Rhat={} ", name, r_text)),
            glyph,
            Span::plain(format!(" {ess_text}")),
        ]));
        // The decomposition follows the glyph, not the finding: it is the
        // "so what do I do" line for anything not certified converged.
        if band != RhatBand::Converged || drew_finding {
            if let Some(driver) = p.rhat_driver() {
                out.push(plain(format!(
                    "  {:12}   R̂ = max(bulk {:.3}, folded {:.3}), classic {}; \
                     the {} half is larger — {}",
                    "", stat_f64(p.rhat_bulk()), stat_f64(p.rhat_folded()),
                    p.rhat_classic().cell(3, "—"), driver.half(), driver.describe())));
            }
            // Whether each chain mixes inside its own mode, which no
            // cross-chain statistic can say.
            let per_chain = p.ess_per_chain();
            if per_chain.len() >= 2 {
                let cells: Vec<String> = per_chain.iter()
                    .map(|e| if e.is_finite() { format!("{e:.0}") } else { "—".into() })
                    .collect();
                out.push(plain(format!("  {:12}   per-chain ESS [{}]", "", cells.join(", "))));
            }
        }
    }
    out
}

/// PGAS's acceptance table: each chain's acceptance per parameter, against
/// the kernel's healthy band. A leaf written before the chains were recorded
/// beside their rates has no table.
fn pgas_acceptance_table(r: &crate::fit::method_result::PgasRun) -> Vec<Line> {
    if r.acceptance_per_chain.is_empty() {
        return Vec::new();
    }
    let kernel = r.acceptance_kernel.unwrap_or(AcceptanceKernel::RandomWalk);
    let (lo, hi) = kernel.healthy_band();
    let nuts = kernel == AcceptanceKernel::Nuts;
    let mut out = vec![Line::default(), plain(format!(
        "acceptance rates{}:", if nuts { " (NUTS block; ~80% is the target)" } else { "" }))];
    for (chain, rates) in &r.acceptance_per_chain {
        let mut spans = vec![Span::plain(format!("  chain {}: ", chain))];
        for (i, (name, &rate)) in rates.iter().enumerate() {
            let style = if rate < lo { Style::Err } else if rate > hi { Style::Warn } else { Style::Ok };
            if i > 0 {
                spans.push(Span::plain(" "));
            }
            spans.push(Span::plain(format!("  {name}=")));
            spans.push(Span::styled(format!("{:.0}%", rate * 100.0), style));
        }
        out.push(Line(spans));
    }
    out
}

/// An NLopt method's block: a point estimate with no IF2 gate, so a compact
/// headline and the θ̂ table from the typed result.
fn nlopt_block(version_lead: bool, report: &MethodReport, r: &NloptRun) -> Vec<Line> {
    let mut out = vec![Line::default(), identity_line(version_lead, &r.algorithm, &[])];
    out.push(plain(format!(
        "    loglik:   {:.2} ({})     converged chains: {}/{}",
        r.best_loglik,
        crate::fit::loglik::LoglikType::tag_or_unknown(report.loglik_type),
        r.n_converged, r.n_chains
    )));
    match r.chain_agreement() {
        Some(range) => out.push(plain(format!(
            "    chain-agreement: max rel range = {:.2}% bound", range * 100.0))),
        None if r.n_chains > 1 => out.push(plain(
            "    chain-agreement: not computed (fewer than two converged chains)")),
        None => {}
    }
    out.push(plain(format!("    θ̂ ({} estimated params):", r.theta_hat.len())));
    let w = name_col_width(r.theta_hat.keys().map(String::as_str), 14);
    for (k, v) in &r.theta_hat {
        let shown = fit_name(k, w);
        out.push(plain(match report.param_dates.get(k) {
            Some(date) => format!("      {:<w$} = {}  ({})", shown, v, date),
            None => format!("      {:<w$} = {}", shown, v),
        }));
    }
    out
}

/// A pfilter method's block: the likelihood at fixed parameters, the mean
/// and spread over its replicate filters, as the run printed them.
fn pfilter_block(version_lead: bool, r: &PfilterRun) -> Vec<Line> {
    vec![
        Line::default(),
        identity_line(version_lead, "pfilter", &[]),
        plain(format!("    loglik = {:.1} ± {:.1} ({} reps, {} particles)",
            r.ll_mean, r.ll_sd, r.n_replicates, r.n_particles)),
    ]
}

/// A failed method's block: the run's opening block as `fit run` prints it,
/// the findings recorded before it stopped, and the error that stopped it.
fn failed_block(
    version_lead: bool,
    header: &RunHeader,
    findings: &[sim::inference::diagnostic::Diagnostic],
    error: &str,
) -> Vec<Line> {
    let mut out = vec![Line::default()];
    if version_lead {
        out.push(Line(vec![Span::plain("  camdl "), Span::styled(crate::version::VERSION_SHORT, Style::Dim)]));
    }
    out.push(plain(format!("fit: {} (method={})", header.config, header.method)));
    out.push(plain(format!("  model:    {}", header.model)));
    out.push(plain(format!("  estimate: {}", header.estimate.join(", "))));
    out.push(plain(format!("  fixed:    {}", header.fixed.join(", "))));
    out.push(plain(format!("  starts:   {}", header.starts)));
    out.push(plain(format!("  output:   {}", header.output)));
    out.extend(super::findings::render(
        findings,
        sim::inference::diagnostic::HintContext { chains_completed: None },
    ));
    out.push(Line(vec![Span::styled("error:", Style::Err), Span::plain(format!(" {error}"))]));
    out
}

// ─── The fit around the methods (`fit summary`) ───────────────────────────────

/// What `fit summary` shows above its first method: the fit's directory,
/// what the model is (its file-header `#'` block, gh#750, one source line per
/// output line so a `@base`/`@adds`/`@changes` header stays readable), and,
/// under `--parameters`, the parameter and constant legends.
pub(crate) fn summary_head(
    dir: &str,
    docs: &ir::ModelDocs,
    constants: &[ir::parameter::Constant],
    parameters: bool,
) -> Vec<Line> {
    let mut out = vec![
        Line::default(),
        Line(vec![Span::styled(dir, Style::Bold), Span::plain("/")]),
    ];
    if let Some(d) = &docs.model {
        if let Some(t) = &d.text {
            out.push(Line::default());
            out.extend(t.lines().map(|l| plain(format!("  {l}"))));
        }
        if let Some(r) = &d.reference {
            out.push(plain(format!("  [{r}]")));
        }
    }
    out.extend(text_lines(&parameter_legend(&docs.parameters, parameters)));
    out.extend(text_lines(&constant_legend(constants, &docs.constants, parameters)));
    out
}

/// The line for a fit with no completed method: the version, which no
/// method line carried, and the absence itself.
pub(crate) fn summary_no_methods(dir: &str) -> Vec<Line> {
    vec![
        Line(vec![Span::plain("  camdl "), Span::styled(crate::version::VERSION_SHORT, Style::Dim)]),
        plain(format!("  (no completed methods found in {dir})")),
    ]
}

/// The two flags that add to `fit summary`'s output, named where a reader has
/// just finished reading it and is deciding what to ask next. Each is offered
/// only while it would change what is on screen.
pub(crate) fn summary_foot(has_legend: bool, parameters: bool, explain: bool) -> Vec<Line> {
    let mut offers: Vec<&str> = Vec::new();
    if has_legend && !parameters {
        offers.push("--parameters for priors and provenance");
    }
    if !explain {
        offers.push("--explain for how to read this");
    }
    if offers.is_empty() {
        return Vec::new();
    }
    vec![Line::default(), indented("  ", offers.join(" · "), Style::Dim)]
}

// ─── What every method block opens with ───────────────────────────────────────

/// One method's identity line: what ran, with what, over how much.
///
/// `version_lead` puts the camdl version first: it is a fact about the fit,
/// so it opens the first method's line and is not repeated on the others'.
/// `extra` is whatever the method alone can say — for PGAS, how many chains
/// left a latent path behind.
pub(crate) fn identity_line(version_lead: bool, method: &str, extra: &[String]) -> Line {
    let mut spans = vec![Span::plain("  ")];
    if version_lead {
        spans.push(Span::plain("camdl "));
        spans.push(Span::styled(crate::version::VERSION_SHORT, Style::Dim));
        spans.push(Span::plain(" · "));
    }
    spans.push(Span::styled(method, Style::Bold));
    for e in extra {
        spans.push(Span::plain(format!(" · {e}")));
    }
    Line(spans)
}

/// One line per chain refused at its start, each carrying the refusal's own
/// reason — for PGAS the log-likelihood breakdown (transition, observation,
/// ivp, prior), which says which term was impossible — after one line for
/// each part of `diagnostics.json` that could not be read. Empty when nothing
/// was refused.
pub(crate) fn refused_lines(refused: &RefusedChains) -> Vec<Line> {
    let mut out: Vec<Line> = refused.unreadable.iter()
        .map(|e| indented("  ", e.as_str(), Style::Warn))
        .collect();
    for (chain, reason) in &refused.records {
        out.push(Line(vec![
            Span::plain("  "),
            Span::styled("refused", Style::Warn),
            Span::plain(format!(" chain {chain}: {reason}")),
        ]));
    }
    out
}

/// Where the chains began — the rule as written and whether it spread them —
/// under every leaf's identity line, so the reader of an R̂ (or an Â) knows
/// what it can and cannot say. A leaf written before the fields existed says
/// `unknown` rather than a guess. `source_run` is the run id a stored-fit
/// source resolved to when this run started (gh#896): the handle is how the
/// rule was written, the run id is what it read.
pub(crate) fn seeded_from_line(
    source: Option<&str>,
    kind: Option<crate::fit::starts::ChainStartsKind>,
    source_run: Option<&str>,
) -> Line {
    use crate::fit::starts::ChainStartsKind;
    let named;
    let source = match (source, source_run) {
        (Some(src), Some(id)) => {
            named = format!("{src}, run {}", &id[..12.min(id.len())]);
            Some(named.as_str())
        }
        (src, _) => src,
    };
    let what = match (source, kind) {
        (Some(src), Some(k)) => format!(
            "{} ({})",
            src,
            match k {
                ChainStartsKind::Spread => "one independent draw per chain",
                ChainStartsKind::Point => "every chain at one point",
            }
        ),
        (Some(src), None) => src.to_string(),
        (None, _) => "unknown".to_string(),
    };
    plain(format!("  seeded from:  {}", what))
}

// ─── The Bayesian block ───────────────────────────────────────────────────────

/// A PGAS, PMMH or `mh` method's block in `fit summary`.
///
/// Reading order is the whole design: what ran, then the verdict, then the
/// estimates, then the diagnostics that qualify them. Nothing is hidden to
/// achieve the order — every section carries its full table.
pub(crate) fn bayesian_block(version_lead: bool, report: &MethodReport, explain_on: bool) -> Vec<Line> {
    // Convergence and efficiency come from the shared diagnostics; the
    // method-specific extras (acceptance, the highest-draw loglik) per variant.
    let (method, diag, posterior_mean, acceptance_summary, map_loglik) = match &report.method_result {
        Some(MethodResult::Pgas(r)) => ("pgas", &r.diagnostics, &r.posterior_mean, None, None),
        Some(MethodResult::Pmmh(r)) => (
            "pmmh", &r.diagnostics, &r.posterior_mean, Some(r.acceptance_rate), Some(r.map_loglik),
        ),
        _ => return Vec::new(),
    };
    let default_detail = ChainDetail::default();
    let detail = report.chain_detail.as_ref().unwrap_or(&default_detail);
    let refused = RefusedChains::from_findings(&report.findings, &report.findings_unreadable);

    // Identity. With an active chain selection `diag.n_chains` is already the
    // RETAINED count (recomputed), so name the subset and what was dropped
    // rather than printing a total that is no longer the total. gh#663: a
    // chain refused at its start is named here, from its durable record,
    // rather than folded into a chain count that reads as a normal run.
    let ran_of = detail.chain_starts.as_ref()
        .and_then(|st| refused.ran_of(st.n_chains, st.n_good_chains));
    let mut extra = vec![match (&detail.subset, &ran_of) {
        (Some(info), _) => format!(
            "{} of {} chains (excluded {})",
            info.kept.len(), info.n_total, info.excluded_csv()
        ),
        (None, Some(text)) => text.clone(),
        (None, None) => format!("{} chains", diag.n_chains),
    }];
    extra.push(format!("{} draws", diag.n_samples));
    if detail.n_chains_with_paths > 0 {
        extra.push(format!("{} chains saved paths", detail.n_chains_with_paths));
    }
    if let Some(ll) = map_loglik {
        // The log-likelihood of the visited draw that maximised log-likelihood
        // + log prior, the prior on the natural scale with no transform
        // Jacobian: not a MAP estimate, and not an MLE. Said as what it is.
        extra.push(format!(
            "loglik {:.1} at the highest log-posterior draw (natural-scale prior, no Jacobian)",
            ll));
    }
    let mut out = vec![identity_line(version_lead, method, &extra)];
    out.extend(refused_lines(&refused));
    out.push(seeded_from_line(
        detail.chain_starts.as_ref().and_then(|st| st.source.as_deref()),
        detail.chain_starts.as_ref().and_then(|st| st.kind),
        detail.source_run.as_deref(),
    ));
    out.extend(verdict_section(diag, detail));
    out.extend(posterior_section(diag, posterior_mean, &report.param_dates, explain_on));
    out.extend(convergence_section(diag, acceptance_summary, explain_on));
    out.extend(chains_section(
        diag.n_chains, diag.frozen_chains.as_deref().unwrap_or(&[]), detail, explain_on));
    if method == "pgas" {
        out.extend(latent_paths_section(detail, explain_on));
    }
    out
}

// ─── The IF2 block ────────────────────────────────────────────────────────────

/// An IF2 method's block in `fit summary`: the compound gate's verdict, the
/// Â-keyed parameter table, the per-chain loglik-eval table, the filter's ESS
/// at the estimate, the dt check and the provenance cross-check.
pub(crate) fn if2_block(version_lead: bool, report: &MethodReport, if2: &If2Run) -> Vec<Line> {
    let state = &if2.record;
    let refused = RefusedChains::from_findings(&report.findings, &report.findings_unreadable);
    let source_run = report.chain_detail.as_ref().and_then(|d| d.source_run.as_deref());
    let mut out = vec![identity_line(
        version_lead, "if2",
        &[refused
            .ran_of(state.n_chains, state.n_good_chains)
            .unwrap_or_else(|| format!("{} chains", state.n_chains))],
    )];
    out.extend(refused_lines(&refused));
    out.push(seeded_from_line(state.chain_init_source.as_deref(), state.chain_starts_kind, source_run));
    out.push(Line::default());

    // Headline — type tag *after* the number (gh#280), so a scraper reading
    // `loglik=<num>` stops at the first non-numeric char. With loglik-eval
    // data the headline is the cross-chain max of the re-scored per-chain
    // values (the per-chain table below breaks it down).
    let mut headline = format!("  best loglik:  {:.1} ({})",
        if2.best_loglik,
        crate::fit::loglik::LoglikType::tag_or_unknown(state.loglik_type));
    if !state.chain_eval_logliks.is_empty() {
        headline.push_str("  (loglik-eval, max across chains)");
    }
    out.push(plain(headline));
    if let Some(ref v) = state.camdl_version {
        if v != crate::version::VERSION_SHORT {
            out.push(indented("                ",
                format!("⚠ stale: produced by {}, current is {}", v, crate::version::VERSION_SHORT),
                Style::Warn));
        }
    }
    out.push(Line::default());

    out.extend(if2_gate_section(state));
    out.extend(if2_parameter_table(state, &report.parameters));
    if !state.chain_eval_logliks.is_empty() {
        out.extend(if2_chain_eval_table(state));
    }
    // ESS-at-MLE: read from chain_evaluations.tsv with the typed result.
    if let Some(ess) = &if2.ess_at_mle {
        out.push(indented("  ", "ESS at θ̂", Style::Bold));
        out.push(plain(format!(
            "    min  = {:>8.0}{}    mean = {:>8.0}",
            ess.ess_min,
            match ess.ess_min_step {
                Some(step) => format!("  (at obs step {})", step),
                None => String::new(),
            },
            ess.ess_mean,
        )));
        out.push(Line::default());
    }
    // Richardson dt-convergence verdict (gh#52), when the method ran it.
    if let Some(dt_check) = &state.dt_check {
        out.extend(dt_check_section(dt_check));
    }
    if let Some(prov) = &report.provenance {
        out.extend(provenance_section(prov));
    }
    out.push(Line::default());
    out
}

/// The gate config an Â is judged against: `resolved_gate`, the value the run
/// used, else `GateConfig::default()` with a caveat. Shared by the gate
/// section and the parameter table so the two cannot glyph one number
/// against two different thresholds.
fn resolve_gate(state: &If2Record) -> (GateConfig, bool) {
    match &state.resolved_gate {
        Some(g) => (g.clone(), true),
        None => (GateConfig::default(), false),
    }
}

/// One Â glyph. The band comes from the gate; only the style is the
/// renderer's own.
fn a_glyph(gate: &GateConfig, a: f64) -> Span {
    use crate::fit::gating::AgreementBand;
    match gate.a_band(a) {
        AgreementBand::Pass => Span::styled("✓", Style::Ok),
        AgreementBand::SoftWarn => Span::styled("~", Style::Warn),
        AgreementBand::Fail => Span::styled("✗", Style::Err),
        AgreementBand::NotAssessed => Span::styled("n/a", Style::Dim),
    }
}

fn pass_glyph(passes: bool) -> Span {
    if passes { Span::styled("✓", Style::Ok) } else { Span::styled("✗", Style::Err) }
}

/// The compound convergence gate: the Â leg, the decibans leg and the
/// overall verdict, from the one measurement the source check and `fit
/// table` read (`gating::assess_if2`), so the three cannot disagree.
pub(crate) fn if2_gate_section(state: &If2Record) -> Vec<Line> {
    let mut out = vec![indented("  ", "compound convergence gate", Style::Bold)];
    let (gate, resolved) = resolve_gate(state);
    let assessed = crate::fit::gating::assess_if2(state, &gate);

    // Â leg — over every estimated parameter, initial-state ones included;
    // the dynamics-only sub-verdict is reported beside it.
    let (max_a, max_a_param) = assessed.worst()
        .map(|(n, a)| (a, n))
        .unwrap_or((0.0, "—".into()));
    out.push(Line(vec![
        Span::plain(format!("    Â leg:           max Â = {:.3} ({})  ", max_a, max_a_param)),
        pass_glyph(assessed.a_passes()),
        Span::plain(format!("  (threshold {:.2})", gate.a_thresh)),
    ]));
    if !assessed.unavailable.is_empty() {
        out.push(indented("    ", format!("Â unavailable (not finite), so not a pass: {}",
            assessed.unavailable.join(", ")), Style::Warn));
    }
    if !assessed.perturb_only_at_t0.is_empty() {
        let dynamics = if assessed.dynamics_a_passes(&gate) { "✓" } else { "✗" };
        out.push(indented("    ",
            format!("dynamics only (without perturb_only_at_t0): {dynamics}"), Style::Dim));
    }

    match assessed.spread {
        Some(leg) => out.push(Line(vec![
            Span::plain(format!("    decibans leg:    Δ = {:.1} dB / threshold {:.1} dB  ",
                leg.delta_db, leg.threshold_db)),
            pass_glyph(leg.passes()),
            Span::plain(format!("  (σ_max={:.2})", leg.sigma_max)),
        ])),
        None => out.push(Line(vec![
            Span::plain("    decibans leg:    "),
            Span::styled("—", Style::Dim),
            Span::plain(" (loglik-eval data not present)"),
        ])),
    }
    let overall = if assessed.is_not_assessed() {
        Span::styled("— NOT ASSESSED (one chain: nothing to compare between chains)", Style::Warn)
    } else if let (true, Some(lost)) = (assessed.is_incomplete(), assessed.lost_chains) {
        Span::styled(format!(
            "— INCOMPLETE ({} of {} chains lost mid-run; not a pass)",
            lost.lost(), lost.started), Style::Warn)
    } else if assessed.is_incomplete() {
        Span::styled("— INCOMPLETE (not a pass)", Style::Warn)
    } else if assessed.passes() {
        Span::styled("✓ PASS", Style::Ok)
    } else {
        Span::styled("✗ FAIL", Style::Err)
    };
    out.push(Line(vec![Span::plain("    overall:         "), overall]));
    if !resolved {
        out.push(indented("    ",
            "(thresholds unknown — fit_state.toml predates Phase 3; \
             showing GateConfig::default())", Style::Warn));
    }
    out.push(Line::default());
    out
}

/// The estimated parameters at the selected chain's θ̂, each with its Â
/// glyphed against the same gate the section above reports, so one number
/// cannot print ✗ there and ✓ here. `parameters` are the report's rows: the
/// estimated parameters only, by name, with their dates.
pub(crate) fn if2_parameter_table(state: &If2Record, parameters: &[ParameterReport]) -> Vec<Line> {
    let (gate, _) = resolve_gate(state);
    let mut out = vec![Line(vec![
        Span::plain("  "),
        Span::styled("parameter estimates (loglik-eval, selected chain θ̂)", Style::Bold),
        Span::plain("  "),
        Span::styled(format!("Â threshold {:.2}", gate.a_thresh), Style::Dim),
    ])];
    if state.start_values.is_empty() {
        out.push(indented("    ", "(no start_values in fit_state.toml)", Style::Dim));
        out.push(Line::default());
        return out;
    }
    // Sized to the names present: `{:12}` is a minimum width, so a longer
    // name shifted the `=` and every column after it.
    let w = name_col_width(parameters.iter().map(|p| p.name.as_str()), 12);
    // Rendered once so the value column can be sized to them too. `{:.6}`
    // bounds the decimals but not the integer part, so a population-sized
    // estimate (1e6 renders 14 characters) would overrun a fixed field and
    // shift the Â column and everything after it on that row alone.
    let vals: Vec<String> = parameters.iter().map(|p| format!("{:.6}", p.estimate)).collect();
    let vw = num_col_width(vals.iter().map(|v| v.as_str()), 12);
    for (p, val) in parameters.iter().zip(&vals) {
        let mut row = vec![Span::plain(format!("    {:w$} = {:<vw$}  ", fit_name(&p.name, w), val))];
        match p.chain_agreement {
            // A collapsed within-chain variance leaves no Â at all; say so
            // rather than printing `Â=NaN ✗`, which reads as a failure the
            // estimator never assessed (gh#45).
            Some(r) if !r.is_finite() => {
                row.push(Span::styled("Â unavailable — verdict incomplete", Style::Dim));
            }
            Some(r) => {
                row.push(Span::plain(format!("Â={:.3} ", r)));
                row.push(a_glyph(&gate, r));
            }
            None => row.push(Span::styled("Â=—", Style::Dim)),
        }
        if p.perturb_only_at_t0 {
            row.push(Span::plain(" "));
            row.push(Span::styled("(t0-only)", Style::Dim));
        }
        if let Some(date) = &p.estimate_date {
            row.push(Span::plain(format!("  ({})", date)));
        }
        out.push(Line(row));
    }
    out.push(Line::default());
    out
}

/// The per-chain loglik-eval table, labelled by the chain ids the record
/// carries and never by row position: IF2 drops PF-degenerate chains before
/// the eval, so row `i` is not chain `i` on any fit where a chain was
/// dropped. A record written before `chain_eval_ids` existed gets no table,
/// and says why (alpha posture: no fallback).
pub(crate) fn if2_chain_eval_table(state: &If2Record) -> Vec<Line> {
    let n = state.chain_eval_logliks.len();
    let mut out = vec![indented("  ", format!("per-chain loglik-eval ({} chains)", n), Style::Bold)];
    let best_idx = state.chain_eval_logliks.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    if state.chain_eval_ids.len() != n {
        out.push(indented("    ",
            "(omitted: this fit_state predates per-chain ids, and a row \
             position is not a chain number once a chain has been dropped)", Style::Dim));
        out.push(Line::default());
        return out;
    }
    out.push(plain(format!("    {:6} {:>12}   {:>6}", "chain", "loglik", "± se")));
    for i in 0..n {
        let ll = state.chain_eval_logliks[i];
        let se = state.chain_eval_ses.get(i).copied().unwrap_or(f64::NAN);
        let mut row = vec![Span::plain(format!("    {:6} {:>12.2}   ± {:>4.2}",
            state.chain_eval_ids[i], ll, se))];
        // The chain whose θ̂ is reported as the MLE (cross-chain argmax of the
        // re-scored loglik): "selected", describing the operation.
        if Some(i) == best_idx {
            row.push(Span::plain("  "));
            row.push(Span::styled("← selected", Style::Ok));
        }
        out.push(Line(row));
    }
    out.push(Line::default());
    out
}

/// The Richardson dt-convergence verdict (gh#52). A pass is one line; a
/// marginal or failing verdict adds the ladder that decided it.
pub(crate) fn dt_check_section(dt_check: &crate::fit::dt_check::DtCheckResult) -> Vec<Line> {
    use crate::fit::dt_check::DtCheckVerdict;
    let mut out = vec![indented("  ", "dt-convergence at θ̂ (Richardson)", Style::Bold)];
    let label = match dt_check.verdict {
        DtCheckVerdict::Pass => Span::styled("PASS", Style::Ok),
        DtCheckVerdict::Marginal => Span::styled("MARGINAL", Style::Warn),
        DtCheckVerdict::Fail => Span::styled("FAIL", Style::Err),
        DtCheckVerdict::Skipped => Span::plain("skipped"),
    };
    out.push(Line(vec![
        Span::plain("    verdict: "),
        label,
        Span::plain(format!("    ({})", dt_check.notes)),
    ]));
    // The pass case's numbers are already in `notes`; the ladder is noise.
    if !matches!(dt_check.verdict, DtCheckVerdict::Pass | DtCheckVerdict::Skipped) {
        out.push(plain(format!(
            "    threshold τ = {:.2} nats  (SE-aware floor 4·σ_max = {:.2})",
            dt_check.threshold_nats, dt_check.threshold_se_aware_nats)));
        for (i, rung) in dt_check.ladder.iter().enumerate() {
            let tag = if i == 0 { "(fit)" } else { "" };
            out.push(plain(format!(
                "    dt = {:>7.4}   ll = {:>9.2} ± {:.2}   {}",
                rung.dt, rung.loglik, rung.se, tag)));
        }
        if dt_check.pf_se_inflation {
            out.push(Line(vec![
                Span::plain("    "),
                Span::styled("⚠", Style::Warn),
                Span::plain(" PF-SE inflated as dt halved (auxiliary signal)."),
            ]));
        }
        if matches!(dt_check.verdict, DtCheckVerdict::Fail) {
            out.push(plain("    Note: synthetic recovery shares this dt and \
                cannot detect dt bias by itself."));
        }
    }
    out.push(Line::default());
    out
}

/// The provenance cross-check: `final_params.toml` against
/// `mle_params.toml`, and the recorded winner against `final_params.toml`
/// (the GH #16 silent-wrong-answer class).
pub(crate) fn provenance_section(prov: &ProvenanceReport) -> Vec<Line> {
    let mut out = vec![indented("  ", "provenance", Style::Bold)];
    match (prov.final_params_present, prov.mle_params_present) {
        (true, true) => out.push(Line(vec![
            Span::plain("    final_params.toml ↔ mle_params.toml: "),
            if prov.final_params_matches_mle_params == Some(true) {
                Span::styled("✓ params match", Style::Ok)
            } else {
                Span::styled("✗ DISAGREE — silent-wrong-answer (GH #16) class", Style::Err)
            },
        ])),
        (false, _) => out.push(Line(vec![
            Span::plain("    final_params.toml: "),
            Span::styled("(absent)", Style::Dim),
        ])),
        (_, false) => out.push(Line(vec![
            Span::plain("    mle_params.toml:   "),
            Span::styled("(absent)", Style::Dim),
        ])),
    }
    if let Some(matches) = prov.fit_state_winner_matches_final_params {
        out.push(Line(vec![
            Span::plain("    fit_state.toml ↔ final_params.toml:   "),
            if matches {
                Span::styled("✓ params match", Style::Ok)
            } else {
                Span::styled("✗ DISAGREE — fit_state's start_values diverge from winner", Style::Err)
            },
        ]));
    }
    out
}


// ─── The Bayesian block's sections ────────────────────────────────────────────

/// The answer, before any table: did this fit converge, and what are the
/// two or three numbers that qualify the answer.
///
/// Every fact on the second and third lines is also a column somewhere
/// below — the point is that a reader who stops here has not been misled,
/// and a reader who continues knows what to look for.
pub(crate) fn verdict_section(diag: &PosteriorDiagnostics, detail: &ChainDetail) -> Vec<Line> {
    use crate::fit::method_result::{MaxRhat, ParamConvergence, RhatBand, RHAT_CONVERGED_THRESHOLD};
    let mut out = vec![Line::default()];
    match diag.max_rhat_status() {
        MaxRhat::Reported(v) => {
            // The same band the method's own end-of-run block glyphs
            // against, so one R̂ cannot read ✓ when the fit finishes and ✗
            // here. The verdict is also spelled out: on a surface a
            // public-health decision is read off, a glyph should not have
            // to carry the whole meaning.
            let band = RhatBand::of(v);
            let style = match band {
                RhatBand::Converged => Style::Ok,
                RhatBand::NotConverged => Style::Warn,
                RhatBand::Severe => Style::Err,
                RhatBand::NotAssessed => Style::Dim,
            };
            out.push(Line(vec![
                Span::plain("  "),
                Span::styled(band.glyph(), style),
                Span::plain(format!(" {}", band.describe())),
            ]));
            let carrier = match worst_rhat_param(diag) {
                Some((name, _)) => format!(" ({})", name),
                None => String::new(),
            };
            out.push(plain(format!(
                "    max R̂ {:.3}{}, threshold {}", v, carrier, RHAT_CONVERGED_THRESHOLD)));
            let n_above = diag
                .per_param
                .values()
                .filter(|p| match p.rhat() {
                    Stat::Value(x) => x >= RHAT_CONVERGED_THRESHOLD,
                    Stat::Infinite => true,
                    Stat::Undefined => false,
                })
                .count();
            out.extend(verdict_facts(
                Some(format!("{} of {} parameters above threshold", n_above, diag.per_param.len())),
                detail,
            ));
        }
        MaxRhat::ChainsFrozen { chains } => {
            out.push(Line(vec![
                Span::plain("  "),
                Span::styled("✗", Style::Err),
                Span::plain(" NOT converged"),
            ]));
            let what = if chains.len() == 1 {
                "its retained draws are one parameter vector, a point mass in every pooled \
                 number below"
            } else {
                "the retained draws of each are one parameter vector, a point mass in every \
                 pooled number below"
            };
            out.push(plain(format!(
                "    {}: {what}", crate::fit::method_result::frozen_chains_clause(&chains))));
            out.extend(verdict_facts(None, detail));
        }
        MaxRhat::Unassessable { params } => {
            out.push(Line(vec![
                Span::plain("  "),
                Span::styled("✗", Style::Err),
                Span::plain(" NOT converged"),
            ]));
            out.push(plain(format!(
                "    R̂ is unavailable for {} of {} parameters (a sampler failure, or a \
                 summary that does not record it), so the fit is not certified",
                params.len(),
                diag.per_param.len()
            )));
            out.extend(verdict_facts(
                Some("each parameter's reason is under `convergence`".to_string()),
                detail,
            ));
        }
        MaxRhat::NotApplicable { reason } => {
            out.push(not_assessed_headline());
            // The refusal with its numbers, when the leaf carries them:
            // "all 4 chains started at one point (starts = from_mle @mle)"
            // says what the bare classification cannot.
            let why = diag.per_param.values().find_map(|p| match p {
                ParamConvergence::Scored { rhat_withheld: Some(e), .. }
                | ParamConvergence::NotScored { detail: Some(e), .. } => Some(e.to_string()),
                _ => None,
            });
            out.push(plain(match why {
                Some(d) => format!("    R̂ — not assessed: {}", d),
                None => format!(
                    "    a between-chain statistic was never possible here: {}",
                    reason.describe()
                ),
            }));
            out.extend(verdict_facts(None, detail));
        }
        MaxRhat::NoParams => {
            out.push(not_assessed_headline());
            out.push(plain("    no parameter was assessed across chains"));
            out.extend(verdict_facts(None, detail));
        }
    }
    out
}

fn not_assessed_headline() -> Line {
    Line(vec![Span::plain("  "), Span::styled("—", Style::Dim), Span::plain(" not assessed")])
}

/// The verdict's third line: the counts that qualify it, dot-separated.
///
/// A fact with nothing to report is dropped rather than printed as a zero —
/// "no chain differs" and "the chains were never compared" are different
/// statements and must not share a rendering.
///
/// Under `--exclude-chains` its job is also to say what the chain count is
/// not. R̂ and the parameter counts on the lines above are recomputed over
/// the retained chains; the log-likelihood comparison is the one the run
/// recorded over every chain, which the selection does not filter. Standing
/// unlabelled beside recomputed numbers, it would read as though it shared
/// their scope.
fn verdict_facts(lead: Option<String>, detail: &ChainDetail) -> Vec<Line> {
    let mut parts: Vec<String> = lead.into_iter().collect();
    match &detail.chain_logliks {
        None => parts.push("chains' log-likelihood comparison not recorded".to_string()),
        Some(a) if a.compared() => {
        let scores = &a.chains;
        let n = scores.iter().filter(|s| s.is_flagged()).count();
        let n_assessed = scores.iter().filter(|s| s.test().is_some()).count();
        let scope = match detail.subset {
            Some(_) => " (over every chain, not the retained subset)",
            None => "",
        };
        let n_all = scores.len();
        parts.push(match (n_assessed, n) {
            (0, _) => format!("chains' log-likelihoods not assessed{scope}"),
            (a, 0) if a == n_all => format!("chains' log-likelihoods agree{scope}"),
            (a, 0) => format!("{a} of {n_all} chains' log-likelihoods compared, none differs{scope}"),
            (a, 1) if a == n_all => format!("1 chain's log-likelihood differs{scope}"),
            (a, n) if a == n_all => format!("{n} chains' log-likelihoods differ{scope}"),
            (a, 1) => format!("{a} of {n_all} chains' log-likelihoods compared, 1 differs{scope}"),
            (a, n) => format!("{a} of {n_all} chains' log-likelihoods compared, {n} differ{scope}"),
        });
        }
        Some(_) => {}
    }
    if let Some(j) = &detail.forkable {
        let scope = match &j.selection {
            Some(info) => format!(" over the retained chains (chain(s) {} excluded)",
                info.excluded_csv()),
            None => String::new(),
        };
        parts.push(format!("{}/{} forkable{}", j.n_forkable, j.n_total, scope));
    }
    // Packed at the `·` separators, never inside a fact: word-wrapping this
    // line splits `80/48000 forkable` across two of them, which reads as a
    // number and then an orphaned word.
    let mut out = Vec::new();
    let mut line = String::new();
    for part in parts {
        if !line.is_empty() && line.chars().count() + 3 + part.chars().count() > SECTION_WIDTH - 2 {
            out.push(indented("    ", std::mem::take(&mut line), Style::Dim));
        }
        if !line.is_empty() {
            line.push_str(" · ");
        }
        line.push_str(&part);
    }
    if !line.is_empty() {
        out.push(indented("    ", line, Style::Dim));
    }
    out
}

/// The estimates, worst R̂ first. `dates` carries the calendar date of each
/// `instant`-kind estimate (`MethodReport::param_dates`).
pub(crate) fn posterior_section(
    diag: &PosteriorDiagnostics,
    posterior_mean: &BTreeMap<String, f64>,
    dates: &BTreeMap<String, String>,
    explain_on: bool,
) -> Vec<Line> {
    let mut out = section_rule(SECTION_POSTERIOR);
    if posterior_mean.is_empty() {
        out.push(indented("  ", "(no posterior parameters)", Style::Dim));
        return out;
    }
    // The column is sized to the names in THIS table. `{:14}` is a minimum
    // width, so a name longer than it pushes every later column right and the
    // grid stops being one.
    let w = name_col_width(posterior_mean.keys().map(String::as_str), 14);
    // Every header cell but the last is plain ASCII, where `{:>N}` pads to the
    // right column. `R̂` is not: it is two `char`s in one cell, so `{:>8}` gave
    // it a seven-cell field and the label sat one cell left of the eight-cell
    // column of numbers it names.
    out.push(plain(format!(
        "  {:w$} {:>14} {:>10} {:>10} {}",
        "param", "mean", "ESS bulk", "ESS tail", right_align("R̂", 8)
    )));
    for name in by_rhat_desc(diag, posterior_mean.keys().map(String::as_str)) {
        let mean = posterior_mean[name];
        // Both encodings of "no pooled ESS" — an absent key on the loaded
        // path, a present NaN on the --exclude-chains recompute — render as
        // the same dash (gh#691). One fact, one rendering.
        let date_marker = match dates.get(name) {
            Some(date) => format!("  ({})", date),
            None => String::new(),
        };
        out.push(plain(format!(
            "  {:w$} {:>14} {:>10} {:>10} {:>8}{}",
            fit_name(name, w),
            sig_figs(mean, POSTERIOR_MEAN_SIG_FIGS),
            diag.ess_cell(name, "—"),
            diag.ess_tail_cell(name, "—"),
            diag.rhat_cell(name, "—"),
            date_marker
        )));
    }
    out.extend(explain(explain_on, SECTION_POSTERIOR));
    out
}

/// Both halves of R̂, the classic statistic, which half is worse, and the
/// per-chain ESS — one row per parameter, in the same order the posterior
/// table uses.
///
/// Every parameter gets a row, not only the failing ones. A reader comparing
/// a failing parameter against a healthy one needs both on screen, and the
/// healthy rows are what make the failing gap legible.
pub(crate) fn convergence_section(
    diag: &PosteriorDiagnostics,
    acceptance_summary: Option<f64>,
    explain_on: bool,
) -> Vec<Line> {
    let mut out = section_rule(SECTION_CONVERGENCE);
    if diag.per_param.is_empty() {
        out.push(indented("  ", "(no parameter was assessed across chains)", Style::Dim));
    } else {
        let w = name_col_width(diag.per_param.keys().map(String::as_str), 14);
        out.push(plain(format!(
            "  {:w$} {:>8} {:>9} {:>9}   {:<6} {}",
            "param", "bulk", "folded", "classic", "worse", "per-chain ESS"
        )));
        for name in by_rhat_desc(diag, diag.per_param.keys().map(String::as_str)) {
            let p = &diag.per_param[name];
            let worse = match p.rhat_driver() {
                Some(d) => d.half().to_string(),
                None => "—".to_string(),
            };
            let per_chain: String = p
                .ess_per_chain()
                .iter()
                .map(|e| if e.is_finite() { format!("{e:.0}") } else { "—".to_string() })
                .collect::<Vec<_>>()
                .join(" ");
            out.push(plain(format!(
                "  {:w$} {:>8} {:>9} {:>9}   {:<6} {}",
                fit_name(name, w),
                p.rhat_bulk().cell(3, "—"),
                p.rhat_folded().cell(3, "—"),
                p.rhat_classic().cell(3, "—"),
                worse,
                per_chain,
            )));
        }
        // A parameter carrying no usable R̂ has a row of dashes above; the
        // reason it has one is the finding, and it is only readable here.
        for (name, p) in &diag.per_param {
            if let Some(why) = p.why_no_rhat() {
                out.push(indented("  ", format!("{name} — {why}"), Style::Warn));
            }
        }
    }
    if let Some(acc) = acceptance_summary {
        out.push(plain(format!("  acceptance = {:.3} (mean across chains)", acc)));
    }
    out.extend(efficiency_lines(diag));
    out.extend(explain(explain_on, SECTION_CONVERGENCE));
    out
}

/// ESS per raw sampling iteration and per wall-clock second — or the reason
/// neither can be reported.
///
/// Both are the min-parameter ESS over a denominator, so both stand or fall
/// with that minimum being defined. It is defined only when every parameter
/// assessed across chains reports a pooled ESS: a minimum over the reporting
/// subset RISES as a fit gets worse, because the badly-mixing parameters drop
/// out and the survivors set it (gh#687). When it is undefined, name the
/// parameters that withhold it — the blank is then the diagnosis rather than
/// a gap the reader must interpret.
pub(crate) fn efficiency_lines(diag: &PosteriorDiagnostics) -> Vec<Line> {
    use crate::fit::method_result::MinEss;
    let mut out = Vec::new();
    match diag.min_ess_status() {
        MinEss::Reported(min_ess) => {
            // ESS/iteration — the ALGORITHM-comparison metric: min-parameter
            // ESS per raw sampling step. `n_samples` is KEPT (thinned) draws,
            // so `× thin` recovers the raw iterations, making this invariant
            // to thinning and iteration count. Hardware-independent: "this
            // sampler mixes N× better per step" holds on any machine.
            if let Some(epi) = diag.ess_per_iter() {
                out.push(plain(format!(
                    "  ESS/iter = {}  (min-param ESS {:.0} / {} raw sampling iters)",
                    sig_figs(epi, EFFICIENCY_SIG_FIGS), min_ess, diag.raw_iters()
                )));
            }
            // ESS/second — the RUNTIME metric: min-parameter ESS per second of
            // wall-clock. Also thinning-invariant, but hardware/implementation-
            // dependent, so it estimates runtime-to-target on THIS machine
            // rather than comparing algorithms. `None` wall-time (older runs)
            // simply omits it.
            if let (Some(eps), Some(secs)) =
                (diag.ess_per_sec(), diag.wall_time_secs.filter(|s| *s > 0.0))
            {
                out.push(plain(format!(
                    "  ESS/sec  = {}  (min-param ESS {:.0} / {:.1}s wall)",
                    sig_figs(eps, EFFICIENCY_SIG_FIGS), min_ess, secs
                )));
            }
        }
        MinEss::Unreportable { missing, n_expected } => {
            out.push(plain("  ESS/iter = —   ESS/sec = —   (efficiency not reportable)"));
            out.push(plain(format!(
                "    {} of {} parameters report no bulk ESS — either the estimator",
                missing.len(),
                n_expected
            )));
            out.push(plain("    refused their draws outright, or the rank-transformed draws"));
            out.push(plain("    are constant so the autocorrelation is undefined. Each says"));
            out.push(plain("    which in the per-parameter table above:"));
            // Wrap the names to a readable width. The per-parameter table
            // above marks them too, but a column of dashes reads as "not
            // applicable"; the reader should not have to derive the list.
            for line in wrap(&missing.join(", "), 62) {
                out.push(plain(format!("      {}", line)));
            }
            out.push(plain(format!(
                "    A minimum over the {} that did report would rise as the fit got",
                n_expected - missing.len()
            )));
            out.push(plain("    worse, so no efficiency headline is given."));
        }
        // No parameter was assessed across chains — there was never an
        // efficiency line here, and nothing to explain.
        MinEss::NoParams => {}
    }
    out
}

/// Per-chain log-likelihood, its comparison with the other chains, and — for
/// PGAS — how many latent paths each chain left behind.
///
/// One table, because both halves are keyed on the chain id and answer halves
/// of one question: chain 7 differs from the rest, and chain 7 is also where
/// ten of the forkable paths came from.
///
/// The scored column is decided by the method's log-likelihood class, by
/// name: `obs_ll` (`log p(y | X, θ)`) for PGAS, `log_likelihood`
/// (`log p(y | θ)`) for pmmh / mh (gh#667). For PGAS the complete-data target
/// and its transition term are shown alongside — the sampler's own objective,
/// and the term whose spread makes a flat-ridge diagnosis obvious — but they
/// are never ranked on. When no per-chain traces exist, says so rather than
/// skipping.
pub(crate) fn chains_section(
    n_chains_expected: usize,
    frozen_chains: &[crate::chain_index::ChainIndex],
    detail: &ChainDetail,
    explain_on: bool,
) -> Vec<Line> {
    let mut out = section_rule(SECTION_CHAINS);

    // `--exclude-chains` recomputes the posterior diagnostics over the
    // retained chains, but this table is read from the per-chain traces on
    // disk and the selection does not filter them. Every excluded chain still
    // has a row and is still compared with the others. That is
    // deliberate — inspecting the chain you dropped is the reason you dropped
    // it — but it must be SAID, or the rows read as though they shared the
    // recomputed scope of the tables above.
    if let Some(info) = &detail.subset {
        out.extend(note(Style::Dim, &format!(
            "(every chain is listed, including the excluded {} — this table \
             is the comparison the run recorded, and the chains are compared \
             over all of them)",
            info.excluded_csv())));
    }

    // Per-chain path counts, keyed by the chain id the artifact records —
    // never by a row's position, which is not an answer once a chain has
    // been refused at its start (gh#607).
    let paths: BTreeMap<u64, (u64, u64)> = detail
        .saved_paths
        .as_ref()
        .map(|r| r.per_chain.iter().map(|c| (c.chain, (c.n_saved, c.n_forkable))).collect())
        .unwrap_or_default();
    let show_paths = !paths.is_empty();

    // Whether a cross-chain comparison is possible at all. When it is not,
    // the reason is stated and the ranked table is skipped — but the
    // degeneracy screens below still run: the -inf screen reads
    // `log_posterior` and the frozen chains are the run's record, so neither
    // goes unreported because the comparison is missing (gh#608, gh#1032).
    let agreement = detail.chain_logliks.as_ref();
    if agreement.is_none() {
        // A leaf written before the run recorded the comparison: said, never
        // recomputed here and never rendered as chains that agree.
        out.push(indented("  ", "(no per-chain log-likelihood comparison recorded in this leaf)",
            Style::Dim));
        out.extend(path_only_table(&paths));
    } else if let Some(a) = agreement.filter(|a| a.column_absent) {
        // No silent gap: a method whose traces predate the column would
        // otherwise render as rows of dashes that read like broken chains.
        out.push(indented("  ", format!(
            "(traces carry no `{}` column — cannot compare chains on it)",
            a.column), Style::Dim));
        out.extend(path_only_table(&paths));
    } else if agreement.is_some_and(|a| a.chains.len() < 2) {
        out.push(indented("  ", "(need ≥2 chains with traces to compare chains)", Style::Dim));
        out.extend(path_only_table(&paths));
    } else if let Some(a) = agreement {
        // Under a selection `n_chains_expected` is the RETAINED count while
        // the rows are every chain on disk, so the comparison would read
        // "found traces for 16 of 6 chains". The note above already says why
        // the two differ; this line only makes sense unfiltered.
        if detail.subset.is_none() && n_chains_expected != 0
            && a.chains.len() != n_chains_expected
        {
            out.push(indented("  ", format!(
                "(found traces for {} of {} chains)",
                a.chains.len(), n_chains_expected), Style::Dim));
        }
        let path_head = if show_paths {
            format!(" {:>7} {:>10}", "paths", "forkable")
        } else {
            String::new()
        };
        out.push(plain(match &a.complete_data {
            Some(_) => format!("  {:5} {:>14}  {:>8}  {:>6}  {:>14} {:>14}{}   {}",
                "chain", "obs ll", "Δ others", "Z", "transition ll", "complete ll", path_head, "flag"),
            None => format!("  {:5} {:>14}  {:>8}  {:>6}{}   {}",
                "chain", "mean loglik", "Δ others", "Z", path_head, "flag"),
        }));
        for (i, sc) in a.chains.iter().enumerate() {
            // A chain not assessed renders `—`, never a fake 0.
            let z = match sc.test() {
                Some(t) => format!("{:>+8.2}  {:>6.2}", t.diff, t.z),
                None => format!("{:>8}  {:>6}", "—", "—"),
            };
            let path_cells = if show_paths {
                match paths.get(&(sc.chain.0 as u64)) {
                    Some((n_saved, n_forkable)) => format!(" {:>7} {:>10}", n_saved, n_forkable),
                    None => format!(" {:>7} {:>10}", "—", "—"),
                }
            } else {
                String::new()
            };
            let mut row = vec![Span::plain(format!("  {:5} ", sc.chain))];
            // gh#608: distinguish "stuck at -inf" (a degenerate chain whose
            // draws contaminate the pooled numbers) from "no readable trace"
            // (—).
            row.extend(chain_loglik_cell(sc.mean_loglik, 14));
            row.push(Span::plain(format!("  {z}")));
            if let Some(cd_means) = &a.complete_data {
                row.push(Span::plain("  "));
                row.extend(chain_loglik_cell(cd_means.transition.get(i).copied().unwrap_or(f64::NAN), 14));
                row.push(Span::plain(" "));
                row.extend(chain_loglik_cell(cd_means.complete.get(i).copied().unwrap_or(f64::NAN), 14));
            }
            row.push(Span::plain(format!("{path_cells}   ")));
            match &sc.outcome {
                ChainLoglikOutcome::Assessed(t) if t.flagged => {
                    row.push(Span::styled("← differs", Style::Warn));
                }
                ChainLoglikOutcome::Assessed(_) => {}
                ChainLoglikOutcome::NotAssessed(_) => {
                    row.push(Span::styled("not assessed", Style::Dim));
                }
            }
            out.push(Line(row));
        }
        // Why the chains not assessed were not, one line per reason.
        let mut reasons: Vec<(String, Vec<String>)> = Vec::new();
        for sc in &a.chains {
            if let ChainLoglikOutcome::NotAssessed(why) = &sc.outcome {
                let why = why.describe();
                match reasons.iter_mut().find(|(w, _)| *w == why) {
                    Some((_, chains)) => chains.push(sc.chain.to_string()),
                    None => reasons.push((why, vec![sc.chain.to_string()])),
                }
            }
        }
        for (why, chains) in reasons {
            let who = match chains.len() {
                1 => format!("chain {}", chains[0]),
                _ => format!("chains {}", chains.join(", ")),
            };
            out.extend(note(Style::Dim, &format!("{who} not compared: {why}.")));
        }
    }

    out.extend(saved_path_notes(detail, show_paths));

    // gh#608 (ebola F8): the stuck-state screen. A chain recording a
    // non-finite log-posterior as its CURRENT state is degenerate (gh#607);
    // the comparison above cannot assess it (a -inf mean has no standard
    // error), so it gets its own loud line, right where the reader is
    // told the pooled numbers include it.
    if let Some(neginf) = &detail.neginf {
        for d in neginf.iter().filter(|d| d.n_neginf > 0 && d.n_retained > 0) {
            out.extend(note(Style::Err, &format!(
                "⚠ chain {}: log-posterior -inf on {:.1}% of retained draws \
                 ({}/{}) — a DEGENERATE chain; its draws are in draws.tsv \
                 and every pooled number in this block. View without it: \
                 --exclude-chains {} (exclusion stays explicit — \
                 gh#419).",
                d.chain,
                100.0 * d.n_neginf as f64 / d.n_retained as f64,
                d.n_neginf, d.n_retained, d.chain,
            )));
        }
    }

    // gh#635, gh#1032: a chain that took no step after burn-in — one
    // parameter vector across its retained draws — is not assessed by the
    // comparison (one effective draw) and evades the −inf screen when its
    // point has finite density. The run recorded it (`frozen_chains`); the
    // same loud treatment, read from that record.
    for c in frozen_chains {
        out.extend(note(Style::Err, &format!(
            "⚠ chain {c}: took no step after burn-in — its retained draws are one \
             parameter vector, a point mass in draws.tsv and in every pooled number \
             in this block. View without it: --exclude-chains {c} (exclusion stays \
             explicit — gh#419).",
        )));
    }

    // What was observed, and what it can indicate — never a cause stated as
    // fact. R̂ and ESS above are the convergence verdict; this corroborates.
    let observed: Vec<String> = detail.chain_logliks.as_ref()
        .map(crate::fit::chain_diagnostics::describe_flagged)
        .unwrap_or_default();
    for o in &observed {
        out.extend(note(Style::Warn, &format!("⚠ {o}.")));
    }
    if !observed.is_empty() {
        out.extend(note(Style::Warn,
            "This can indicate a chain sampling another region of the posterior (a side \
             mode, or a ridge along a weakly identified parameter) or one that has not \
             mixed; compare its per-chain posterior with the others'."));
    }
    out.extend(explain(explain_on, SECTION_CHAINS));
    out
}

/// The `paths` / `forkable` columns on their own, for a method whose chains
/// could not be ranked.
///
/// The two halves of the `chains` table come from different artifacts — the
/// per-chain traces and the method summary's `trajectories` block — and
/// either can be missing without the other. When the ranking half is, these
/// counts are still a fact about each chain, and dropping them because their
/// neighbours are absent would hide data the section holds.
fn path_only_table(paths: &BTreeMap<u64, (u64, u64)>) -> Vec<Line> {
    if paths.is_empty() {
        return Vec::new();
    }
    let mut out = vec![plain(format!("  {:5} {:>7} {:>10}", "chain", "paths", "forkable"))];
    for (chain, (n_saved, n_forkable)) in paths {
        out.push(plain(format!("  {:5} {:>7} {:>10}", chain, n_saved, n_forkable)));
    }
    out
}

/// What the `paths` / `forkable` columns could not say, and why.
///
/// A PGAS chain retains a posterior draw every `thin` sweeps and writes a
/// latent path on every `draw_stride`-th of those draws, so every written
/// path should be one a consumer can join to a draw — `simulate --init-state
/// fit` or a `last_obs`-anchored `quantities {}` entry. A shortfall is a real
/// event (a record skipped as incoherent, a chain resumed partway in) and is
/// named rather than left for the reader to subtract.
fn saved_path_notes(detail: &ChainDetail, show_paths: bool) -> Vec<Line> {
    if !show_paths {
        // A method that saved paths but cannot say which chain each row
        // belongs to. The columns are omitted rather than filled against a
        // positional guess, and the omission is stated (gh#727 follow-up).
        if detail.n_chains_with_paths > 0 {
            return note(Style::Dim,
                "(no per-chain path counts: this leaf's `trajectories` block \
                 predates the chain ids that key them, and a row's position is \
                 not an answer — re-run the method to recover the columns)");
        }
        return Vec::new();
    }
    let Some(report) = &detail.saved_paths else { return Vec::new() };
    let total_saved: u64 = report.per_chain.iter().map(|c| c.n_saved).sum();
    let total_forkable: u64 = report.per_chain.iter().map(|c| c.n_forkable).sum();
    let mut out = Vec::new();
    if total_forkable < total_saved {
        out.extend(note(Style::Warn, &format!(
            "{} of {} written paths cannot be joined to a posterior draw",
            total_saved - total_forkable, total_saved)));
        out.extend(note(Style::Dim, &format!(
            "a path is written on every {} retained draw (thin {}), so every \
             one should be joinable; the shortfall is paths that were \
             written outside this leaf's retained set or skipped as \
             incoherent records",
            report.draw_stride.map_or("—".to_string(), |v| v.to_string()),
            report.thin)));
    }
    out
}

/// Whether the sampled latent paths themselves mixed — the particle filter's
/// ESS at each observation, then R̂ and the frozen-cell fractions over the
/// paths on disk.
///
/// Both are about X rather than θ, which is why they share a section: the
/// first says how well each observation was resolved, the second whether the
/// resolved paths agree across chains.
pub(crate) fn latent_paths_section(detail: &ChainDetail, explain_on: bool) -> Vec<Line> {
    use crate::fit::latent_convergence::LATENT_CONVERGENCE_TSV;
    let mut out = section_rule(SECTION_LATENT);
    out.extend(filter_ess_lines(detail));
    match &detail.latent {
        None => out.push(indented("  ", "(this leaf saved no latent path)", Style::Dim)),
        Some(Err(e)) => out.extend(note(Style::Warn, &format!("not computed: {e}"))),
        Some(Ok(lc)) => {
            // The headline the section rule has already half-said, compact:
            // which chains, how many paths each, over what shape.
            out.push(plain(format!(
                "  chains {} × {} paths · {} substeps × {} columns",
                lc.chain_ids.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "),
                lc.n_draws,
                lc.n_substeps,
                lc.columns.len(),
            )));
            out.extend(text_lines(&lc.bins_table()));
            out.extend(text_lines(&lc.findings()));
            // The per-cell table is the one the method writes; a leaf that
            // predates it had it written by the read, and the reader is told.
            match &detail.latent_table_written {
                Some(Ok(())) => out.push(indented("  ", format!(
                    "the leaf predates this block; {LATENT_CONVERGENCE_TSV} written now from its saved paths"), Style::Dim)),
                Some(Err(e)) => out.push(indented("  ", format!(
                    "the leaf predates this block and the table could not be written: {e}"), Style::Dim)),
                None => {}
            }
        }
    }
    out.extend(explain(explain_on, SECTION_LATENT));
    out
}

/// The conditional filter's ESS at every observation (gh#685), from the
/// `filter_ess` block of `pgas_summary.json`. Quiet for a method that wrote
/// no block — one that predates it, or in which no sweep scored an
/// observation.
fn filter_ess_lines(detail: &ChainDetail) -> Vec<Line> {
    let Some(fe) = &detail.filter_ess else { return Vec::new() };
    // Indented into the section with the rows below it; `FilterEss::report`
    // is written for the end-of-run print, which has no section to sit in.
    let mut out: Vec<Line> = fe.report().trim_start_matches('\n').lines()
        .map(|l| plain(format!("  {l}")))
        .collect();
    if fe.n_starved > 0 {
        out.extend(note(Style::Warn, &format!(
            "the path through {} starved observation(s) is drawn from a handful of \
             particles every sweep; look at the data at t={} before the model",
            fe.n_starved,
            fe.worst.first().map_or("?".to_string(), |o| crate::quantile::fmt_time(o.time)))));
    }
    out
}

/// One right-aligned per-chain log-likelihood cell. `-inf` (a chain stuck
/// off the support, gh#608) is rendered loudly and distinctly from `—`
/// (nothing readable) — softening either one hides the only signal there is.
fn chain_loglik_cell(v: f64, width: usize) -> Vec<Span> {
    if v.is_finite() {
        vec![Span::plain(format!("{:>width$.2}", v))]
    } else if v == f64::NEG_INFINITY {
        right_styled("-inf", width, Style::Err)
    } else {
        vec![Span::plain(format!("{:>width$}", "—"))]
    }
}


// ─── Text layout shared by the sections ───────────────────────────────────────

/// Significant figures the posterior-mean column carries. Four separates the
/// values a reader compares across rows (`0.001854` against `240.8`) without
/// implying a precision the Monte-Carlo error does not support.
///
/// The export formats (`--format md` / `latex` / `json`) deliberately do NOT
/// share this: they stay at a fixed `{:.6}`. A terminal column is read by eye
/// and scanned across rows, so relative precision and a narrow column win;
/// an export is parsed or typeset downstream, where a fixed number of decimals
/// is easier to align, diff and re-read than a width that varies per value.
/// The two are different jobs, not a drift to reconcile — do not "fix" one to
/// match the other.
pub(crate) const POSTERIOR_MEAN_SIG_FIGS: usize = 4;

/// Significant figures the two efficiency headlines carry.
///
/// Fixed decimals were the wrong invariant here for a sharper reason than in
/// the mean column: the quantity is a ratio whose numerator is an ESS and whose
/// denominator is an iteration or a second count, so it gets SMALLER exactly as
/// the fit gets worse. A min-parameter ESS of 8 over 48,000 raw iterations is
/// 1.7e-4, which `{:.3}` renders `0.000` — the headline reads as "nothing" on
/// the one fit whose reader most needs the number, and two such fits an order
/// of magnitude apart render identically. Significant figures keep three digits
/// at any magnitude, and [`sig_figs`] falls back to scientific notation below
/// `1e-4` rather than growing a run of leading zeros.
pub(crate) const EFFICIENCY_SIG_FIGS: usize = 3;

/// Column the section rules and the `--explain` prose wrap to. Wide enough for
/// the per-chain log-likelihood table, narrow enough to survive an 80-column
/// terminal without wrapping.
pub(crate) const SECTION_WIDTH: usize = 76;

/// Widest a parameter-name column is allowed to grow before names are
/// ellipsized instead. Stratified names are built by suffixing (`I0_ituri`,
/// `phi_split_haut_uele`) and a deeply stratified model can reach a width no
/// terminal helps with; past this point a readable grid is worth more than a
/// complete name.
pub(crate) const NAME_COL_MAX: usize = 44;

/// The width a name column must have for every name to fit, bounded by
/// [`NAME_COL_MAX`] and never narrower than `min` (which keeps a short-named
/// model's table looking as it always did).
///
/// Rust's `{:14}` is a MINIMUM width: it pads a short name and passes a long
/// one through whole, shoving every later column right. Sizing the column to
/// the names actually present is what keeps the grid a grid.
pub(crate) fn name_col_width<'a>(names: impl Iterator<Item = &'a str>, min: usize) -> usize {
    names
        .map(|n| n.chars().count())
        .max()
        .unwrap_or(0)
        .clamp(min, NAME_COL_MAX)
}

/// The width a column of already-rendered numbers must have for every one of
/// them to fit, never narrower than `min`.
///
/// The numeric twin of [`name_col_width`], and it has no upper bound for the
/// same reason that one does: a name can be ellipsized and still identify its
/// row, a number cannot be shortened without changing what it says. Callers
/// render each value once and size the column to those strings — `{:<12.6}`
/// bounds the decimals but not the integer part, so a value at or above `1e6`
/// is fourteen characters in a twelve-wide field and moves every column after
/// it on that row while its neighbours stay put.
pub(crate) fn num_col_width<'a>(rendered: impl Iterator<Item = &'a str>, min: usize) -> usize {
    rendered.map(|v| v.chars().count()).max().unwrap_or(0).max(min)
}

/// Codepoint ranges [`display_width`] treats as occupying no cell of their own:
/// the combining diacritical marks (U+0302, the accent in `R̂`) and the
/// combining marks for symbols.
const COMBINING_MARKS: [(char, char); 2] = [('\u{0300}', '\u{036F}'), ('\u{20D0}', '\u{20F0}')];

/// How many terminal cells `s` occupies.
///
/// `chars().count()` — which is what Rust's own `{:>8}` pads by — is the wrong
/// measure wherever a string carries a combining mark: `R̂` is `R` followed by
/// U+0302 COMBINING CIRCUMFLEX ACCENT, two `char`s the terminal composes into
/// one cell, so a field padded by `char` count comes out a cell short of the
/// column it is meant to fill. Subtracting the combining marks is the whole of
/// the correction these tables need — their labels are ASCII plus a diacritic,
/// with nothing double-width in them — so a full Unicode width table would be a
/// dependency bought for a handful of fixed strings.
pub(crate) fn display_width(s: &str) -> usize {
    s.chars()
        .filter(|c| !COMBINING_MARKS.iter().any(|(lo, hi)| c >= lo && c <= hi))
        .count()
}

/// `s` right-aligned in a field `width` display cells wide: `{:>width$}` with
/// [`display_width`] in place of `chars().count()`.
///
/// Reach for it wherever a padded cell can carry a combining mark, and leave
/// `{:>N}` where the cell is plain ASCII. A hand-placed compensating space
/// would do the same thing here and stop doing it the next time the string
/// changes.
pub(crate) fn right_align(s: &str, width: usize) -> String {
    " ".repeat(width.saturating_sub(display_width(s))) + s
}

/// `name` cut to `width` characters, ellipsized in the MIDDLE.
///
/// The head gives way, not the tail: a stratified parameter's distinguishing
/// part is its suffix (`..._nord_kivu`), so a tail-truncated name would
/// collapse every stratum of one parameter onto the same unreadable row.
pub(crate) fn fit_name(name: &str, width: usize) -> String {
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
pub(crate) fn sig_figs(v: f64, sig: usize) -> String {
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

/// `--explain` prose, one entry per section, keyed by the section's own name.
///
/// A term is defined ONCE, in the section it is primarily read in, and never
/// repeated: R̂ appears as a column in `posterior` and again in `latent paths`,
/// but it is `convergence` that shows both its halves side by side, so that is
/// where it is explained. `posterior` therefore has no entry — every column it
/// carries is defined under `convergence`.
///
/// Written as one paragraph each; [`wrap`] lays them out at the rendering
/// width, so re-flowing here changes nothing on screen.
pub(crate) const EXPLAIN: &[(&str, &str)] = &[
    (
        SECTION_CONVERGENCE,
        "R̂ compares within-chain to between-chain variance; 1.0 is perfect \
         agreement. Rank-normalized, so a heavy tail cannot inflate it. Two \
         halves: `bulk` worse means the chains disagree about where the \
         posterior sits; `folded` worse means they agree on location and \
         disagree on spread. `classic` is the unnormalized statistic, shown \
         because a large gap to it is itself informative — it says the \
         disagreement lives in the tails.",
    ),
    (
        SECTION_CHAINS,
        "Compared on `obs ll` = log p(y | X): does this chain reproduce the data. \
         Transition and complete ll are path densities at each chain's own \
         path, so they are shown but not compared. `Δ others` is the chain's \
         mean minus the pooled mean of the other chains, in nats; `Z` divides \
         it by its Monte Carlo standard error, from each chain's own trace \
         (sd / √ESS). A chain is flagged when Z exceeds Φ⁻¹(1 − 0.001/2k) for \
         k chains (3.66 for four), so a fit whose chains agree is flagged once \
         in a thousand; a flagged chain is set aside and the rest compared \
         again. A chain with fewer than 10 effective draws is not assessed. \
         `forkable` \
         counts draws whose latent path was saved, so a counterfactual can be \
         branched from them; only path-saved draws qualify.",
    ),
    (
        SECTION_LATENT,
        "`frozen-disagree`: cells where every chain sits constant at its own \
         value — each chain holds a single draw, so R̂ is undefined rather than \
         large. `chains frozen`: over the non-constant cells, the fraction of \
         chains that never moved; a cell counts as mixed once one chain moves \
         once, so this is the honest denominator.",
    ),
];

/// Section names. Constants because [`EXPLAIN`] is keyed on them: a renamed
/// section that silently lost its prose would be invisible.
pub(crate) const SECTION_POSTERIOR: &str = "posterior";
pub(crate) const SECTION_CONVERGENCE: &str = "convergence";
pub(crate) const SECTION_CHAINS: &str = "chains";
pub(crate) const SECTION_LATENT: &str = "latent paths";

/// `text` greedily wrapped to `width` columns, one `String` per line.
///
/// Deliberately simple: whitespace-separated, no hyphenation, no re-flowing of
/// a word longer than the width (it takes its own over-long line rather than
/// being cut, because the over-long words here are identifiers). Deterministic,
/// so a rendered block can be asserted whole.
pub(crate) fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        // `chars().count()`, not `len()`: `R̂` and `—` are multi-byte.
        let w = word.chars().count();
        if !line.is_empty() && line.chars().count() + 1 + w > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// The model's parameter legend from its `#'` docs — symbol, name, the prose,
/// and the citation — or nothing.
///
/// Nothing unless `wanted`. The legend is the priors, their citations and
/// their caveats: reference material a modeller reads deliberately, and not
/// the answer `fit summary` is run to get. On the ebola province model it is
/// thirty-six lines carrying two DOIs, all of it above the first number, which
/// is why it moved behind `--parameters` rather than staying the first thing
/// on screen.
pub(crate) fn parameter_legend(
    documented: &BTreeMap<String, ir::parameter::DocBlock>,
    wanted: bool,
) -> String {
    if !wanted || documented.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n  parameters\n");
    for (name, d) in documented {
        s.push_str("    ");
        if let Some(sym) = &d.symbol {
            s.push_str(sym);
            s.push_str("  ");
        }
        s.push_str(name);
        if let Some(t) = &d.text {
            s.push_str("  —  ");
            s.push_str(t);
        }
        if let Some(r) = &d.reference {
            s.push_str("  [");
            s.push_str(r);
            s.push(']');
        }
        s.push('\n');
    }
    s
}

/// The model's constants — each typed `let` with a constant body, with its
/// value, its declared kind and its `#'` prose — or nothing.
///
/// A constant is part of the model, not a parameter: no fit estimated or held
/// it, so it appears in no method's table. It is listed here, after the
/// parameter legend and under the same flag, because its value is an
/// assumption the fit rests on. Every constant is listed, documented or not.
pub(crate) fn constant_legend(
    constants: &[ir::parameter::Constant],
    documented: &BTreeMap<String, ir::parameter::DocBlock>,
    wanted: bool,
) -> String {
    if !wanted || constants.is_empty() {
        return String::new();
    }
    let mut s = String::from("\n  constants (typed `let`s: part of the model, not parameters)\n");
    for c in constants {
        s.push_str("    ");
        let d = documented.get(&c.name);
        if let Some(sym) = d.and_then(|d| d.symbol.as_ref()) {
            s.push_str(sym);
            s.push_str("  ");
        }
        s.push_str(&format!("{} = {} ({})", c.name, c.value, c.kind.as_str()));
        if let Some(t) = d.and_then(|d| d.text.as_ref()) {
            s.push_str("  —  ");
            s.push_str(t);
        }
        if let Some(r) = d.and_then(|d| d.reference.as_ref()) {
            s.push_str("  [");
            s.push_str(r);
            s.push(']');
        }
        s.push('\n');
    }
    s
}

/// Parameter names worst-R̂ first — the order every per-parameter table in the
/// text format uses.
///
/// Alphabetical order buries the parameter a reader opened the summary to find
/// among the healthy ones; on the ebola province fit the worst of thirty sat
/// sixth. Parameters carrying no R̂ sort last: they are a different finding,
/// and `convergence` names each one's reason under its table. Ties break
/// alphabetically, so the order is stable between runs of the same fit.
pub(crate) fn by_rhat_desc<'a>(
    diag: &PosteriorDiagnostics,
    names: impl Iterator<Item = &'a str>,
) -> Vec<&'a str> {
    let mut v: Vec<&str> = names.collect();
    v.sort_by(|a, b| {
        // `(has_no_rhat, value)`: `∞` is a real answer and the worst one, so it
        // sorts above every finite R̂ rather than with the unscored.
        let key = |n: &str| match diag.per_param.get(n).map(|p| p.rhat()) {
            Some(Stat::Infinite) => (0u8, f64::INFINITY),
            Some(Stat::Value(x)) => (0, x),
            _ => (1, 0.0),
        };
        let (ka, va) = key(a);
        let (kb, vb) = key(b);
        ka.cmp(&kb)
            .then(vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.cmp(b))
    });
    v
}

/// The parameter carrying the worst R̂, and that R̂ — the name
/// [`PosteriorDiagnostics::max_rhat_status`] computes the value of but does not
/// return, so the verdict can say WHICH parameter failed in the same breath as
/// saying that one did.
///
/// The same filter as that method's `Reported` arm: finite values only. Ties
/// keep the first in name order, so the verdict is stable between runs.
pub(crate) fn worst_rhat_param(diag: &PosteriorDiagnostics) -> Option<(&str, f64)> {
    let mut best: Option<(&str, f64)> = None;
    for (name, p) in &diag.per_param {
        let Some(v) = p.rhat().finite() else { continue };
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((name.as_str(), v)),
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_row_per_chain_with_its_warmup() {
        let rates = BTreeMap::from([(0, 0.0005), (2, 0.396)]);
        let warmup = vec![ChainWarmup {
            chain: 0, warmup_steps: 1000, steps_at_scale_floor: 241,
            final_scale: 4.54e-5, target_accept: 0.337,
        }];
        let lines = acceptance_table(&rates, &warmup, AcceptanceKernel::RandomWalk);
        let text: Vec<String> = lines.iter().map(Line::text).collect();
        assert_eq!(text[1], "acceptance rates:");
        assert!(text[3].starts_with("  0") && text[3].contains("0.050% ✗ too low")
            && text[3].contains("4.540e-5") && text[3].ends_with("241 / 1000"), "{}", text[3]);
        assert!(text[4].starts_with("  2") && text[4].contains("39.6% ✓")
            && text[4].ends_with("—"), "{}", text[4]);
    }

    /// A failed run's report: its opening block, the findings recorded before
    /// it stopped, and the error, in that order; the checks travel with it.
    #[test]
    fn a_failed_outcome_shows_its_header_findings_and_error() {
        use sim::inference::diagnostic::{DiagnosticCollector, DiagnosticKind};
        let c = DiagnosticCollector::new("pmmh");
        c.push(DiagnosticKind::BadInit {
            chain_id: 0,
            params: BTreeMap::from([("beta".to_string(), 0.5)]),
            reason: "no start of 10 could be scored".into(),
            attempts: Vec::new(),
        });
        let header = RunHeader {
            config: "fail.toml".into(), method: "pmmh".into(), model: "sir.camdl".into(),
            estimate: vec!["beta".into(), "gamma".into()], fixed: vec!["rho".into()],
            starts: "single (every chain at one point)".into(), output: "results/fits/fail-1".into(),
        };
        let outcome = MethodOutcome::Failed {
            header, findings: c.snapshot(),
            error: "all 2 chains failed init-eval with PFDegenerate".into(),
        };
        // At the end of `fit run` the header is on screen already and the
        // error is `Out::error`'s: the block is the findings alone.
        let run_end: Vec<String> = render_report(&outcome, View::RunEnd)
            .iter().map(Line::text).collect();
        assert!(run_end.iter().any(|l| l.starts_with("── diagnostics")), "{run_end:#?}");
        assert!(!run_end.iter().any(|l| l.contains("fit: fail.toml") || l.contains("error: all 2")),
            "{run_end:#?}");
        let text: Vec<String> = render_report(&outcome, View::Summary(SummaryView::default()))
            .iter().map(Line::text).collect();
        let at = |needle: &str| text.iter().position(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no `{needle}` in {text:#?}"));
        assert!(at("fit: fail.toml (method=pmmh)") < at("estimate: beta, gamma"));
        assert!(at("estimate: beta, gamma") < at("── diagnostics"));
        assert!(at("── diagnostics") < at("error: all 2 chains failed"));
    }
}
