//! `fit summary --format md` and `--format latex`: the fit's summary
//! document as Markdown and as LaTeX tables, from the same reports the text
//! and JSON formats show.

use super::build::MethodReport;
use crate::fit::fit_summary::FitSummaryDoc;
use crate::fit::method_result::{MaxRhat, MethodResult, PosteriorDiagnostics};
use crate::version;

/// The max-R̂ cell for the export formats: the number, or a word saying why
/// there isn't one. Never `0.000` — that is a real R̂ value and must not double
/// as "could not be computed" (review blocker 1).
pub(crate) fn max_rhat_cell(diag: &PosteriorDiagnostics) -> String {
    match diag.max_rhat_status() {
        MaxRhat::Reported(v) => format!("{v:.3}"),
        MaxRhat::ChainsFrozen { chains } => format!(
            "NOT converged — {}", crate::fit::method_result::frozen_chains_clause(&chains)),
        MaxRhat::Unassessable { params } =>
            format!("not computable for {} parameter(s) — NOT converged", params.len()),
        MaxRhat::NotApplicable { reason } => format!("not assessed ({})", reason.describe()),
        MaxRhat::NoParams => "not assessed".to_string(),
    }
}

/// Render a `FitSummaryDoc` as GitHub-flavoured Markdown. Tabular per
/// method, code-fenced parameter tables. Suitable for embedding in book
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
    // Every leaf, completed or not, when there is more than one, so no run
    // is read for the reader and a failed one is not left out (gh#896).
    if doc.leaves.len() > 1 || doc.leaves.iter().any(|l| l.error.is_some() || l.verdict.is_none()) {
        s.push_str("| method | status | verdict | leaf |\n|---|---|---|---|\n");
        for l in &doc.leaves {
            let verdict = match &l.error {
                Some(e) => format!("{} — {}", l.verdict_text, e),
                None => l.verdict_text.clone(),
            };
            s.push_str(&format!("| `{}` | {} | {} | `{}` |\n",
                l.method, crate::fit::fit_summary::status_word(l.status), verdict.replace('|', "\\|"), l.leaf));
        }
        s.push('\n');
    }
    if doc.methods.is_empty() {
        s.push_str("_(no completed methods found)_\n");
        return s;
    }
    for method in &doc.methods {
        s.push_str(&render_md_method(method));
    }
    s
}


pub(crate) fn render_md_method(method: &MethodReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("## `{}` ({})\n\n", method.name, method.method));
    if let Some(ll) = method.best_loglik {
        s.push_str(&format!("- best loglik: **{:.2}** ({})\n",
            ll, crate::fit::loglik::LoglikType::tag_or_unknown(method.loglik_type)));
    }
    s.push_str(&format!("- chains: {}\n", method.n_chains));
    if let Some(init) = method.initial_loglik {
        s.push_str(&format!("- initial loglik: {:.2}\n", init));
    }
    if let Some(prov) = &method.provenance {
        if let Some(stale) = &prov.stale_camdl_version {
            s.push_str(&format!("- ⚠ stale: produced by camdl `{}`, current is `{}`\n",
                stale, version::VERSION_SHORT));
        }
    }
    s.push('\n');

    // IF2 gate verdict (only when populated).
    if let Some(gate) = &method.gate {
        s.push_str("### Compound convergence gate\n\n");
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
    if let Some(MethodResult::Pgas(p)) = &method.method_result {
        s.push_str(&format!("### Posterior summary (PGAS, max R̂ = {})\n\n",
            max_rhat_cell(&p.diagnostics)));
        s.push_str("| param | mean | q025 | q975 | ESS bulk | ESS tail | R̂ |\n|---|---|---|---|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let q025 = p.posterior_q025.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let q975 = p.posterior_q975.get(name).copied().map(|v| format!("{:.4}", v)).unwrap_or_else(|| "—".into());
            let ess = p.diagnostics.ess_cell(name, "—");
            let mean_cell = match method.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} | {} | {} | {} |\n", name, mean_cell, q025, q975, ess,
                p.diagnostics.ess_tail_cell(name, "—"), p.diagnostics.rhat_cell(name, "—")));
        }
        s.push('\n');
    } else if let Some(MethodResult::Pmmh(p)) = &method.method_result {
        s.push_str(&format!(
            "### Posterior summary (PMMH, max R̂ = {}, acceptance = {:.3})\n\n",
            max_rhat_cell(&p.diagnostics), p.acceptance_rate
        ));
        s.push_str("| param | mean | ESS bulk | ESS tail | R̂ |\n|---|---|---|---|---|\n");
        for (name, mean) in &p.posterior_mean {
            let ess = p.diagnostics.ess_cell(name, "—");
            let mean_cell = match method.param_dates.get(name) {
                Some(date) => format!("{:.6} ({})", mean, date),
                None => format!("{:.6}", mean),
            };
            s.push_str(&format!("| `{}` | {} | {} | {} | {} |\n", name, mean_cell, ess,
                p.diagnostics.ess_tail_cell(name, "—"), p.diagnostics.rhat_cell(name, "—")));
        }
        s.push('\n');
    }

    // IF2 parameter table.
    if !method.parameters.is_empty() {
        s.push_str("### Parameter estimates (loglik-eval, selected chain θ̂)\n\n");
        s.push_str("| name | estimate | Â | flags |\n|---|---|---|---|\n");
        for p in &method.parameters {
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
    if !method.chains.is_empty() {
        s.push_str(&format!("### Per-chain loglik-eval ({} chains)\n\n", method.chains.len()));
        // Column / marker naming matches the text emitter and CLAUDE.md:
        // "loglik" / "selected", not "clean_ll" / "winner". The
        // ChainSummary.is_winner field name on the data side stays
        // (it's the data model); only the display label changed.
        s.push_str("| chain | loglik | ± se | selected |\n|---|---|---|---|\n");
        for c in &method.chains {
            let mark = if c.is_winner { "★" } else { "" };
            s.push_str(&format!("| {} | {:.2} | {:.2} | {} |\n",
                c.chain_id, c.clean_loglik, c.clean_se, mark));
        }
        s.push('\n');
    }

    // IF2 provenance.
    if let Some(prov) = &method.provenance {
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

/// Render a `FitSummaryDoc` as LaTeX `tabular` blocks per method.
/// One section per method with three tables: gate verdict, parameters,
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
    if doc.leaves.len() > 1 || doc.leaves.iter().any(|l| l.error.is_some() || l.verdict.is_none()) {
        for l in &doc.leaves {
            s.push_str(&format!("% run: {} {} ({}) {}\n",
                l.method, crate::fit::fit_summary::status_word(l.status), l.verdict_text, l.leaf));
        }
        s.push('\n');
    }
    for method in &doc.methods {
        s.push_str(&render_latex_method(method));
    }
    s
}

/// Render a JSON array of chain ids as a `"3,5"` CSV for the header/provenance
/// lines. Empty array → empty string.
pub(crate) fn render_id_csv(v: &serde_json::Value) -> String {
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

pub(crate) fn render_latex_method(method: &MethodReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "\\subsection*{{Method: \\texttt{{{}}} ({})}}\n\n",
        escape_latex(&method.name),
        method.method
    ));

    if let Some(ll) = method.best_loglik {
        s.push_str(&format!(
            "Best log-likelihood: \\textbf{{{:.2}}} ({}); chains: {}\n\n",
            ll,
            crate::fit::loglik::LoglikType::tag_or_unknown(method.loglik_type),
            method.n_chains
        ));
    } else {
        s.push_str(&format!("chains: {}\n\n", method.n_chains));
    }

    if let Some(gate) = &method.gate {
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

    if !method.parameters.is_empty() {
        s.push_str("\\begin{tabular}{lrrl}\n\\toprule\n");
        s.push_str("Parameter & Estimate & $\\hat A$ & Flags \\\\\n\\midrule\n");
        for p in &method.parameters {
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

    if !method.chains.is_empty() {
        // Column / marker naming matches the text + md emitters: "loglik"
        // / "Selected", not "clean_ll" / "Winner".
        s.push_str("\\begin{tabular}{rrrc}\n\\toprule\n");
        s.push_str("Chain & loglik & $\\pm$ se & Selected \\\\\n\\midrule\n");
        for c in &method.chains {
            let mark = if c.is_winner { "$\\star$" } else { "" };
            s.push_str(&format!(
                "{} & {:.2} & {:.2} & {} \\\\\n",
                c.chain_id, c.clean_loglik, c.clean_se, mark
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n\n");
    }

    // PGAS / PMMH posterior block.
    if let Some(MethodResult::Pgas(p)) = &method.method_result {
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
            let mean_cell = match method.param_dates.get(name) {
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
    } else if let Some(MethodResult::Pmmh(p)) = &method.method_result {
        s.push_str(&format!(
            "Posterior summary (max $\\hat R$ = {}; acceptance = {:.3}):\n\n",
            max_rhat_cell(&p.diagnostics), p.acceptance_rate
        ));
        s.push_str("\\begin{tabular}{lrrrr}\n\\toprule\n");
        s.push_str("Parameter & Mean & ESS bulk & ESS tail & $\\hat R$ \\\\\n\\midrule\n");
        for (name, mean) in &p.posterior_mean {
            let ess = p.diagnostics.ess_cell(name, "---");
            let mean_cell = match method.param_dates.get(name) {
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
    }

    s
}

/// Escape LaTeX-active characters in identifiers / paths. Minimal —
/// we don't escape `_` inside `\texttt{}` because LaTeX renders
/// `\texttt{foo_bar}` literally (the `_` in `\texttt` is allowed in
/// most modern LaTeX engines), but we replace `&`, `%`, `#`, `$`.
pub(crate) fn escape_latex(s: &str) -> String {
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

