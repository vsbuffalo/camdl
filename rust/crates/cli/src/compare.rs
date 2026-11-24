//! `camdl compare`: the scores of several models, differenced against a
//! baseline, as a table, markdown or JSON.
//!
//! Each compared input is read as a [`PrequentialTrace`] per group of rows:
//! a stored score, prediction or fit run through [`stored`], which finds or
//! computes the stored score and reads its rows of one mode, one group per
//! role and, for a forecast, per lead; or a `prequential.json`, read as it is
//! (one group). The groups of the inputs are compared in order, one table
//! each, and no table pools rows of different roles, leads, modes, streams or
//! bases.
//!
//! Within a table every cross-row check runs once ([`preflight`], the checks
//! of `score::preflight`): rows predicted in different modes, of different
//! roles or leads, forecasts of different reach or covariate statements, on
//! different predictive bases, conditioned differently (a stored fit's
//! verified training rows against a fixed point's declared ones), or scoring
//! different observed values are refused, each naming the field. A table then
//! reports per row its elpd, Δelpd against the baseline with the paired SE
//! and an evidence label, and per stream its CRPS, WIS, PIT coverage and 50%
//! coverage, with Δcrps and Δwis.

use crate::chain_selection::ChainSelection;
use crate::fit::stream_bindings::StreamBindings;
use serde::Deserialize;
use score::trace::{
    Conditioning, PredictiveBasis, PrequentialTrace, PrequentialWarning, StreamMean,
};
use score::evidence::{paired_delta, paired_diffs};

mod stored;
use score::preflight::{
    self as pf, CheckId, CheckOutcome, CheckResult, ConditioningMode, Finding,
    NotCheckedReason, Override, Qualification, Qualified, Refusal, SourceDigests, Unchecked, Unhashed,
    UnhashedData, Waiver,
};
use std::path::{Path, PathBuf};

/// Scored steps below which `se(Δ)`'s normal approximation is not to be leaned
/// on. A round number, not a threshold with a derivation behind it — the
/// approximation degrades continuously, and the caveat exists to stop a reader
/// treating `Δ ± se` as a test at the window sizes an outbreak analysis
/// actually has.
const SE_NORMAL_APPROX_MIN_T: usize = 100;

/// The caveat on `se(Δ)` at a short scoring window, or `None`.
///
/// `se(Δ) = √(T · Var_t(Δ_t))` is a normal approximation to the sampling
/// distribution of the summed difference. Sivula, Magnusson & Vehtari
/// (arXiv:2008.10296) show it is unreliable in two regimes that co-occur in
/// practice: small T, and models making similar predictions — precisely the
/// case where a reader most wants the SE to settle the question. Descriptive,
/// not a test.
fn se_caveat(n_scored: usize) -> Option<String> {
    (n_scored < SE_NORMAL_APPROX_MIN_T).then(|| format!(
        "se(Δ) is a normal approximation, unreliable at small T (here T = \
         {n_scored}) and when the models predict alike (Sivula, Magnusson & \
         Vehtari, arXiv:2008.10296) — read it as descriptive, not as a test."
    ))
}

/// Printed once under any rendering that shows an evidence label.
///
/// The tier names ("strong", "decisive") come from Jeffreys' scale for **Bayes
/// factors** — marginal likelihoods with the parameters integrated out. What
/// the evidence column holds is a plug-in predictive likelihood ratio at a
/// single θ̂ fit to the same observations, which is a different quantity: it
/// carries no parameter-uncertainty penalty and no Occam factor. The tier is a
/// reading aid for magnitude, and the reader is told so rather than left to
/// infer a Bayes-factor interpretation from the vocabulary.
const JEFFREYS_SCALE_NOTE: &str =
    "The Jeffreys tiers calibrate Bayes factors; these are in-sample \
     likelihood ratios, not Bayes factors.";

/// The last line of every human rendering (gh#806).
///
/// The footer's job is to state what is specific to THIS comparison — how many
/// steps were scored, what θ saw, the numbers the noise gates read. The
/// definitions behind the column names are the same on every render, so they
/// live in one place that can hold citations and derivations, and the footer
/// names the two ways to reach it.
const METHODS_POINTER: &str =
    "Definitions and caveats: `camdl compare --explain` \
     (or `camdl docs model-comparison`).";

/// What θ saw, in one line, from the traces' `Conditioning` stamp.
///
/// Held-out and in-sample scores answer different questions, and the answer
/// changes how a Δelpd should be read — an in-sample Δ tilts toward the more
/// flexible model, a held-out one does not. Until now the stamp reached only
/// the JSON, so the table a reader actually looks at never said which it was.
/// One line, not per row: the conditioning preflight refuses a cohort whose
/// rows disagree, so the baseline's stamp is the cohort's.
fn conditioning_line(trace: &PrequentialTrace) -> String {
    match &trace.conditioning {
        Conditioning::InSample =>
            "Conditioning: in-sample — θ fit to all scored data; Δelpd favors \
             the more flexible model.".to_string(),
        Conditioning::HoldOutTail { train_end, .. } => format!(
            "Conditioning: held-out tail (train_end = {train_end}) — θ sealed \
             before the scored window."),
        Conditioning::Unknown =>
            "Conditioning: unknown — no record of the rows θ was fitted to.".to_string(),
        Conditioning::HoldOutDeclared { trained_through } => format!(
            "Conditioning: held out as declared (θ declared fitted through t = \
             {trained_through}), not verified."),
    }
}

/// What predictive the scores come from, in one line, from the traces'
/// `PredictiveBasis` stamp (gh#312); `None` for the particle filter's, the
/// basis of every trace before the ODE path existed, which the JSON states
/// per row. One line, not per row: the basis preflight refuses a cohort whose
/// rows disagree, so the baseline's stamp is the cohort's.
fn basis_line(trace: &PrequentialTrace) -> Option<&'static str> {
    match trace.basis {
        PredictiveBasis::Filtered => None,
        PredictiveBasis::OdeFreeForward { .. } => Some(
            "Predictive: ODE free-forward — the deterministic skeleton run from \
             t_start at each θ, never re-anchored to data; each step scored by \
             the exact observation density, not a filter estimate."),
        PredictiveBasis::SimulatedFreeForward => Some(
            "Predictive: simulated free-forward — one stochastic trajectory per \
             θ run from t_start, never re-anchored to data."),
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)] // gh#241 G3: reject typo'd compare.toml keys instead of silently dropping
struct CompareToml {
    baseline: Option<String>,
    metrics: Option<Vec<String>>,
    format: Option<String>,
    /// `--streams`: score each row on these observation streams only.
    streams: Option<Vec<String>>,
    #[serde(rename = "model")]
    models: Vec<CompareModelEntry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompareModelEntry {
    name: String,
    path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format { Table, Md, Json }

/// Per-model row in the comparison table.
#[derive(Clone)]
struct Row {
    name: String,
    path: String,
    trace: PrequentialTrace,
    data: DataIdentity,
    /// The observation streams this row's model declares and its fit bound,
    /// as far as the input says.
    streams: RowStreams,
    /// What the row's scores are of besides its trace.
    facts: ScoreFacts,
}

/// What a row's scores are of beyond its trace: the mode its rows were
/// predicted in, their role and lead, and a forecast's reach and covariate
/// statement. Two rows on which one of these differs are not differenced
/// (the checks `Mode`, `Role`, `Lead`, `Reach` and `Covariates`).
#[derive(Debug, Clone, PartialEq)]
struct ScoreFacts {
    /// `free_forward`, `one_step` or `forecast`.
    mode: String,
    /// `trained`, `held_out`, `future` or `unknown`.
    role: String,
    /// A forecast's lead, as `scored.tsv` writes it; `None` outside a
    /// forecast.
    lead: Option<String>,
    /// A forecast's reach as written (`--horizon`); `None` outside a
    /// forecast.
    reach: Option<String>,
    /// Whether a forecast read a covariate stated `realised`, observed after
    /// its origin.
    realised: bool,
    /// Whether the row was read from a stored score; the facts of a
    /// `prequential.json` are read off its trace.
    stored: bool,
    /// The first row end at which several streams are scored and the score
    /// holds no joint log score of them (free-forward scores hold none): the
    /// row's joint step scores are not defined there, and only a table of
    /// one stream (`--streams`) compares it.
    no_joint: Option<f64>,
}

impl ScoreFacts {
    /// The facts of a `prequential.json`, read off its trace: a particle
    /// filter's trace is one-step and an ODE or simulated one free-forward,
    /// and its conditioning says its rows' role.
    fn of_trace(trace: &PrequentialTrace) -> ScoreFacts {
        ScoreFacts {
            mode: match trace.basis {
                PredictiveBasis::Filtered => "one_step",
                PredictiveBasis::OdeFreeForward { .. } | PredictiveBasis::SimulatedFreeForward => "free_forward",
            }
            .to_string(),
            role: match trace.conditioning {
                Conditioning::InSample => "trained",
                Conditioning::HoldOutTail { .. } | Conditioning::HoldOutDeclared { .. } => "held_out",
                Conditioning::Unknown => "unknown",
            }
            .to_string(),
            lead: None,
            reach: None,
            realised: false,
            stored: false,
            no_joint: None,
        }
    }
}

impl Row {
    /// This row re-scored on one leaf alone
    /// ([`score::trace::project_onto_stream`]); the refusal
    /// names the row.
    fn onto(&self, leaf: &str) -> Result<Row, String> {
        let trace = score::trace::project_onto_stream(&self.trace, leaf)
            .map_err(|e| format!("'{}': {e}", self.name))?;
        Ok(Row { trace, ..self.clone() })
    }

    /// The leaves this row's fit bound, sorted ([`StreamBindings::leaves`]),
    /// or `None` when the input does not say (an explicit
    /// `prequential.json`).
    fn assimilated(&self) -> Option<Vec<String>> {
        match &self.streams {
            RowStreams::Derived { bindings, .. } =>
                Some(bindings.leaves().map(|(l, _)| l.to_string()).collect()),
            RowStreams::Explicit => None,
        }
    }
}

/// What a row's input says about the observation streams behind its trace.
#[derive(Clone)]
enum RowStreams {
    /// A derived row: every leaf the fit's model declares with its source,
    /// in the model's order, and the fit's [`StreamBindings`].
    Derived { declared: Vec<(String, String)>, bindings: StreamBindings },
    /// An explicit `prequential.json`: the trace names the leaves it scored
    /// and nothing else — not the model's sources, not what the fit bound.
    Explicit,
}

/// The leaves `--streams NAME[,NAME...]` selects across `rows`, in the order
/// first resolved (names in the order given, each name's leaves in each
/// row's order).
///
/// A name is a leaf or a source of a derived row's model; a source selects
/// every leaf the model expands it to, in the model's order. For an explicit
/// `prequential.json` row, which records leaf names only, a name must be a
/// leaf its trace scored. A name must resolve in every row, and a refusal
/// lists what that row offers. Rows whose models expand a source differently
/// contribute the union; a row that then lacks one of the selected leaves is
/// refused when it is projected onto it ([`Row::onto`]).
fn resolve_selection(names: &[String], rows: &[Row]) -> Result<Vec<String>, String> {
    if names.is_empty() {
        return Err("--streams names no stream".into());
    }
    let mut selected: Vec<String> = Vec::new();
    for name in names {
        for r in rows {
            let leaves: Vec<&str> = match &r.streams {
                RowStreams::Derived { declared, .. } => {
                    let as_leaf: Vec<&str> = declared.iter()
                        .filter(|(l, _)| l == name).map(|(l, _)| l.as_str()).collect();
                    if !as_leaf.is_empty() {
                        as_leaf
                    } else {
                        declared.iter().filter(|(_, s)| s == name)
                            .map(|(l, _)| l.as_str()).collect()
                    }
                }
                RowStreams::Explicit => r.trace.streams().into_iter()
                    .filter(|l| l == name).collect(),
            };
            if leaves.is_empty() {
                return Err(match &r.streams {
                    RowStreams::Derived { declared, .. } => {
                        let mut sources: Vec<&str> = Vec::new();
                        for (_, s) in declared {
                            if !sources.contains(&s.as_str()) { sources.push(s); }
                        }
                        format!(
                            "--streams: '{name}' is neither an observation stream nor a \
                             data source of the model of '{}'. Its streams: {}. Its \
                             sources: {}.",
                            r.name,
                            declared.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>().join(", "),
                            sources.join(", "))
                    }
                    RowStreams::Explicit => format!(
                        "--streams: '{name}' is not a stream '{}' scored. It is read as an \
                         explicit prequential.json, which records the streams it scored and \
                         not the model's data sources, so a name must be one of them: {}.",
                        r.name, r.trace.streams().join(", ")),
                });
            }
            for l in leaves {
                if !selected.iter().any(|s| s == l) {
                    selected.push(l.to_string());
                }
            }
        }
    }
    Ok(selected)
}

/// What a compared row can say about the observations it was scored against
/// (gh#713).
///
/// Two fits bound to different data produce a Δelpd that mixes a model
/// difference with a data difference, and neither the table nor the trace says
/// so — the trace records scores and times, never which bytes it scored. A
/// row read from a stored score records, per observation source, the SHA-256
/// of the rows its prediction conditioned on: a stored fit's training rows,
/// or a fixed point's stored data table, both read through the store.
#[derive(Clone)]
enum DataIdentity {
    /// The digests of the rows conditioned on, per source.
    Digests(RecordedData),
    /// Nothing to check, and why — an explicit `prequential.json` (a trace
    /// records no data identity). The reason is printed, because a check that
    /// silently covers half the cohort is worse than one the reader knows the
    /// scope of.
    Unknown(String),
}

/// A row's data digests, per observation source.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RecordedData {
    /// Source → the hash of the data bound to it.
    training: std::collections::BTreeMap<String, String>,
    /// Sources bound by the fit whose hash cannot be read from its record, and
    /// why. A check involving one of these is reported as not made, never
    /// passed.
    unresolved: std::collections::BTreeMap<String, String>,
}

/// Hex characters in a full SHA-256.
const FULL_DIGEST_LEN: usize = 64;

/// Row names for positional inputs: the file basename, extended leftward with
/// parent directory components until every name is distinct (gh#806).
///
/// The convention that names a row by its basename renders `a/fit.toml
/// b/fit.toml` as two rows both called `fit.toml` — the arrangement a
/// per-variant directory layout produces, so the collision is the common case
/// rather than a corner one. There is nothing in such a table to say which row
/// is which, and `--baseline` / `--exclude-chains` match the displayed name, so
/// an ambiguous name is not only unreadable but unaddressable.
///
/// Only the colliding names grow, and only far enough to separate them, so a
/// cohort with distinct basenames renders exactly as before. Identical paths
/// stay identical — that is one fit listed twice, which `parse_cohort_exclude`
/// rejects by name as ambiguous.
///
/// Runs at name-assignment time, before `parse_cohort_exclude` reads the names,
/// so a per-fit `--exclude-chains @NAME:IDS` token targets the name the table
/// shows.
fn positional_model_names(paths: &[String]) -> Vec<String> {
    // Root, prefix and `.` components carry no identity; `..` does.
    let components = |p: &String| -> Vec<String> {
        use std::path::Component;
        std::path::Path::new(p).components().filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            Component::ParentDir => Some("..".to_string()),
            _ => None,
        }).collect()
    };
    let comps: Vec<Vec<String>> = paths.iter().map(components).collect();
    let name_at = |i: usize, depth: usize| -> String {
        let c = &comps[i];
        if c.is_empty() { return paths[i].clone(); }
        c[c.len() - depth.min(c.len())..].join("/")
    };

    let mut depth: Vec<usize> = vec![1; paths.len()];
    loop {
        let mut by_name: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..paths.len() {
            by_name.entry(name_at(i, depth[i])).or_default().push(i);
        }
        let mut grew = false;
        for group in by_name.values().filter(|g| g.len() > 1) {
            for &i in group {
                if depth[i] < comps[i].len() {
                    depth[i] += 1;
                    grew = true;
                }
            }
        }
        // Every collision is either separated or unseparable (identical paths).
        if !grew { break; }
    }
    (0..paths.len()).map(|i| name_at(i, depth[i])).collect()
}

#[expect(clippy::disallowed_methods, reason = "reads the `--config` compare.toml that lists the rows to compare; what compare stores is the predictions and scores of those rows, whose identities hold their own inputs")]
pub(crate) fn cmd_compare(a: &crate::args::CompareArgs, compile: crate::util::CompileSettings) {
    // The forcing-extrapolation policy this command's evaluations run under,
    // installed for the rest of the command.
    let forcing =
        crate::forcing_extrapolation::ForcingPolicy::new(a.forcing_extrapolation.allow_forcing_extrapolation);
    let _forcing_route = forcing.install();
    // compare has no `--no-color`: its painters follow the environment and
    // whether each stream is a terminal.
    let out = crate::out::Out::new(false);
    // `--explain` serves the methods page the footer points at, and exits.
    // It is answered before any input is resolved: the reader who needs the
    // definitions is usually the one who has not assembled a comparison yet,
    // and requiring two model paths to read a definition is the friction the
    // pointer exists to remove. Given alongside model paths it still only
    // prints — one contract, no mode where the guide and a table interleave.
    if a.explain {
        out.report_raw(format_args!("{}", crate::docs::topic_body("model-comparison")
            .expect("the model-comparison docs topic is compiled into the binary")));
        return;
    }

    let config_path: Option<String> = a.config.clone();
    let baseline: Option<String> = a.baseline.clone();
    let positional: Vec<String> = a.paths.clone();
    let request = stored::Request {
        mode: a.mode,
        horizon: a.horizon.as_deref(),
        draws: a.draws,
        particles: a.particles,
        seed: a.seed,
        replicates: a.replicates,
        in_sample: a.in_sample,
        data: &a.data,
        data_held_out: &a.data_held_out,
        allow_forcing_extrapolation: a.forcing_extrapolation.allow_forcing_extrapolation,
        allow_mismatched_filter_settings: a.allow_mismatched_filter_settings,
        compile,
    };
    let metrics_cli: Option<Vec<String>> = a.metrics.as_ref().map(|s|
        s.split(',').map(|t| t.trim().to_string()).collect());
    let format = match a.format.as_str() {
        "table" => Format::Table,
        "md"    => Format::Md,
        "json"  => Format::Json,
        other   => {
            out.fail(format!("error: --format must be table|md|json (got '{}')", other));
        }
    };

    // Resolve model list: CLI positional > compare.toml
    let (models, cfg_baseline, cfg_metrics, cfg_format, cfg_streams) = if let Some(path) = config_path {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            out.fail(format!("error: cannot read {}: {}", path, e));
        });
        let cfg: CompareToml = toml::from_str(&text).unwrap_or_else(|e| {
            out.fail(format!("error: parsing {}: {}", path, e));
        });
        let fmt = cfg.format.as_deref().and_then(|f| match f {
            "table" => Some(Format::Table),
            "md"    => Some(Format::Md),
            "json"  => Some(Format::Json),
            _       => None,
        });
        (cfg.models, cfg.baseline, cfg.metrics, fmt, cfg.streams)
    } else if positional.len() >= 2 {
        let names = positional_model_names(&positional);
        let models = positional.iter().zip(names).map(|(p, name)| CompareModelEntry {
            name,
            path: p.clone(),
        }).collect();
        (models, None, None, None, None)
    } else {
        out.fail("error: compare requires either --config FILE or ≥2 stored scores, predictions or fit runs \
                  (@label, run id or folder)");
    };

    if models.len() < 2 {
        out.fail(format!("error: compare needs ≥2 models; got {}", models.len()));
    }

    // gh#417/gh#418: chain exclusion resolved PER FIT, bound to each model's
    // name. `@a:4` targets one fit; bare `3,4` is cohort-wide (every fit). Parsed
    // and name-validated at the boundary, before anything is predicted. An
    // input that is not a fit run has no posterior cloud to filter, so the
    // selection is inert there.
    let model_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let cohort = parse_cohort_selection(&a.chains, &a.exclude_chains, &model_names).unwrap_or_else(|e| {
        out.fail(format!("error: {e}"));
    });
    cohort.warn(&out);

    // Each input resolved, and the inputs checked together, before anything
    // is predicted or scored.
    let inputs: Vec<(CompareModelEntry, Input)> = models.into_iter().map(|m| {
        let input = resolve_input(&m.path, &request).unwrap_or_else(|e| {
            out.fail(format!("error loading the scores of '{}' at '{}': {}", m.name, m.path, e));
        });
        (m, input)
    }).collect();
    let prepared: Vec<(&str, &stored::Prepared)> = inputs.iter().filter_map(|(m, i)| match i {
        Input::Stored(p) => Some((m.name.as_str(), p)),
        Input::Explicit(_) => None,
    }).collect();
    stored::preflight(&prepared, &request).unwrap_or_else(|e| refuse(&out, format!("error: {e}")));
    // Each input's groups of rows, each input loaded once however many
    // streams `--streams` selects.
    let loaded: Vec<(CompareModelEntry, Vec<Loaded>)> = inputs.into_iter().map(|(m, input)| {
        let selection = cohort.for_fit(&m.name);
        let groups = match input {
            Input::Explicit(groups) => Ok(groups),
            Input::Stored(p) => stored::load(&out, p, &request, selection),
        };
        let groups = groups.unwrap_or_else(|e| {
            out.fail(format!("error loading the scores of '{}' at '{}': {}", m.name, m.path, e));
        });
        (m, groups)
    }).collect();
    let groups = grouped(loaded).unwrap_or_else(|e| refuse(&out, e));

    // #295 Ask 1 — the optimism caveat, the miscalibration flags and the trace
    // warnings are emitted by the RENDERERS (`warning_lines`), not here, so they
    // travel with the artifact: a markdown table pasted into a report and a JSON
    // document read by a script carry the same caveats the terminal shows.

    let waivers = Waivers::from_args(a);
    let baseline = baseline.or(cfg_baseline);
    let metrics_chosen = metrics_cli.or(cfg_metrics).unwrap_or_else(|| {
        ["elpd", "crps", "wis", "pit_cov90", "cov50"].map(String::from).to_vec()
    });
    let fmt_final = cfg_format.unwrap_or(format);
    let render = |t: &Table| match fmt_final {
        Format::Json  => render_json(&t.rows, t.base_idx, &metrics_chosen, &t.checks),
        Format::Md    => render_md(&t.rows, t.base_idx, &metrics_chosen, &t.checks),
        Format::Table => render_table(&t.rows, t.base_idx, &metrics_chosen, &t.checks),
    };

    let selected: Option<Vec<String>> = a.streams.as_ref()
        .map(|s| s.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect())
        .or(cfg_streams);
    let Some(names) = selected else {
        // One joint table per group of rows.
        let mut refusals: Vec<String> = Vec::new();
        let tables: Vec<Table> = groups.into_iter().filter_map(|rows| {
            if let Some(e) = no_joint_refusal(&rows) {
                refusals.push(e);
                return None;
            }
            prepare_table(&out, rows, &waivers, baseline.as_deref())
                .map_err(|e| refusals.extend(e)).ok()
        }).collect();
        if !refusals.is_empty() {
            refuse(&out, refusals.join("\n"));
        }
        if let Some(path) = &a.pointwise {
            let [t] = tables.as_slice() else {
                out.fail(format!(
                    "error: --pointwise writes the rows of one table, and this comparison has {} (one per lead); \
                     compare one lead's scores, or read each score's scored.tsv", tables.len()));
            };
            match write_pointwise(&t.rows, t.base_idx, path) {
                Ok(PointwiseWritten { joint: (jp, nj), streams: (sp, ns) }) => out.say(format_args!(
                    "wrote {nj} joint pointwise rows to {} and {ns} per-stream rows to {}",
                    jp.display(), sp.display())),
                Err(e) => {
                    out.fail(format!("error: {e}"));
                }
            }
        }
        let rendered = match (fmt_final, tables.as_slice()) {
            (Format::Json, [one]) => render(one),
            (Format::Json, _) => render_json_tables(&tables, None, &metrics_chosen),
            (Format::Md | Format::Table, _) => tables.iter().map(render).collect::<Vec<_>>().join("\n"),
        };
        out.report_raw(rendered);
        return;
    };

    // `--streams`: one table per group and selected leaf, every row re-scored
    // on that leaf alone. A refusal for any leaf refuses the command; every
    // table's refusals are printed, each naming its leaf.
    let mut leaves: Vec<String> = Vec::new();
    let mut tables: Vec<Table> = Vec::new();
    let mut refusals: Vec<String> = Vec::new();
    for rows in &groups {
        let mine = resolve_selection(&names, rows).unwrap_or_else(|e| {
            out.fail(format!("error: {e}"));
        });
        match stream_tables(&out, rows, &mine, &waivers, baseline.as_deref()) {
            Ok(t) => tables.extend(t),
            Err(e) => refusals.extend(e),
        }
        for l in mine {
            if !leaves.contains(&l) {
                leaves.push(l);
            }
        }
    }
    if !refusals.is_empty() {
        refuse(&out, refusals.join("\n"));
    }
    if let Some(path) = &a.pointwise {
        match write_pointwise_streams(&tables, &leaves, path) {
            Ok((sp, ns)) => out.say(format_args!(
                "wrote {ns} per-stream pointwise rows to {}; no joint file was written to {}, \
                 because --streams scores each stream alone", sp.display(), path.display())),
            Err(e) => {
                out.fail(format!("error: {e}"));
            }
        }
    }
    // One table per stream, and nothing that combines them.
    let rendered = match fmt_final {
        Format::Json => render_json_tables(&tables, Some(&leaves), &metrics_chosen),
        Format::Md | Format::Table => tables.iter().map(render).collect::<Vec<_>>().join("\n"),
    };
    out.report_raw(rendered);
}

/// The inputs' groups of rows as one list of rows per table: the `k`-th group
/// of every input in table `k`. Every input must hold the same number of
/// groups; when they do not, the refusal names each input's leads (a
/// forecast's groups are its leads).
fn grouped(loaded: Vec<(CompareModelEntry, Vec<Loaded>)>) -> Result<Vec<Vec<Row>>, String> {
    let counts: Vec<usize> = loaded.iter().map(|(_, g)| g.len()).collect();
    if counts.windows(2).any(|w| w[0] != w[1]) {
        let each: Vec<String> = loaded.iter().map(|(m, g)| {
            let leads: Vec<String> = g.iter().map(|l| l.facts.lead.clone().unwrap_or_else(|| "—".into())).collect();
            format!("'{}' at {}", m.name, leads.join(", "))
        }).collect();
        return Err(format!(
            "error: the compared scores hold rows at different leads: {}.\n       A forecast's rows at one lead \
             are compared only with rows at that lead; forecast every model to the same --horizon.",
            each.join("; ")));
    }
    let n = counts.first().copied().unwrap_or(0);
    let mut tables: Vec<Vec<Row>> = (0..n).map(|_| Vec::new()).collect();
    for (m, groups) in loaded {
        for (k, Loaded { trace, data, streams, facts }) in groups.into_iter().enumerate() {
            tables[k].push(Row { name: m.name.clone(), path: m.path.clone(), trace, data, streams, facts });
        }
    }
    Ok(tables)
}

/// The refusal of a joint table whose rows hold no joint log score at some
/// time (a free-forward score of several streams observed together), or
/// `None`.
fn no_joint_refusal(rows: &[Row]) -> Option<String> {
    let missing: Vec<String> = rows.iter()
        .filter_map(|r| r.facts.no_joint.map(|t| format!("'{}' at t = {t}", r.name)))
        .collect();
    (!missing.is_empty()).then(|| {
        let mut streams: Vec<&str> = rows.iter().flat_map(|r| r.trace.streams()).collect();
        streams.sort();
        streams.dedup();
        format!(
            "error: a joint score of the streams observed together is not defined for {}: a {} score holds each \
             stream's own score and no joint one there.\n       Compare one stream at a time with --streams {}.",
            missing.join(", "), rows[0].facts.mode, streams.join(","))
    })
}

/// One comparison table, checked and ready to render: its rows, the
/// baseline's index among them, and the outcome of every cross-row check.
struct Table {
    rows: Vec<Row>,
    base_idx: usize,
    checks: Preflight,
}

/// Run the cross-row checks over `rows` and choose the baseline: `Err` with
/// every refusal, in check order, when any check fails and is not waived.
fn prepare_table(
    out: &crate::out::Out,
    rows: Vec<Row>,
    waivers: &Waivers,
    baseline: Option<&str>,
) -> Result<Table, Vec<String>> {
    // Every cross-row check, once: all of them run, and the refusals return
    // here together; what renders states the outcome, overrides included,
    // from this one value (`Preflight`).
    let checks = preflight(&rows, waivers);
    let refusals = checks.refusals();
    if !refusals.is_empty() {
        return Err(refusals.iter().map(|r| refusal_message(r, &rows)).collect());
    }

    // Baseline: explicit > cfg > argmax elpd.
    let baseline_name = baseline.map(str::to_string).unwrap_or_else(|| {
        let best = rows.iter()
            .max_by(|a, b| a.trace.elpd().partial_cmp(&b.trace.elpd())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        best.name.clone()
    });
    let base_idx = rows.iter().position(|r| r.name == baseline_name)
        .unwrap_or_else(|| {
            out.fail(format!("error: baseline '{}' not found among models: {:?}",
                baseline_name,
                rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>()));
        });
    Ok(Table { rows, base_idx, checks })
}

/// One checked table per leaf, every row re-scored on that leaf alone, or
/// `Err` with every table's refusals, each naming its leaf, in leaf order. A
/// row that cannot be projected onto a leaf is that leaf's refusal.
fn stream_tables(
    out: &crate::out::Out,
    rows: &[Row],
    leaves: &[String],
    waivers: &Waivers,
    baseline: Option<&str>,
) -> Result<Vec<Table>, Vec<String>> {
    let mut refusals: Vec<String> = Vec::new();
    let tables: Vec<Table> = leaves.iter().filter_map(|leaf| {
        let projected: Vec<Row> = match rows.iter().map(|r| r.onto(leaf)).collect() {
            Ok(p) => p,
            Err(e) => {
                refusals.push(for_stream(leaf, format!("error: {e}")));
                return None;
            }
        };
        match prepare_table(out, projected, waivers, baseline) {
            Ok(t) => Some(t),
            Err(e) => {
                refusals.extend(e.into_iter().map(|m| for_stream(leaf, m)));
                None
            }
        }
    }).collect();
    if refusals.is_empty() { Ok(tables) } else { Err(refusals) }
}

/// A refusal of one `--streams` table, naming the stream it was for.
fn for_stream(leaf: &str, refusal: String) -> String {
    match refusal.strip_prefix("error: ") {
        Some(rest) => format!("error: stream '{leaf}': {rest}"),
        None => format!("stream '{leaf}': {refusal}"),
    }
}

/// Whether two held-out rows sealed training at the same boundary.
///
/// A `train_end` is a fit's training extent (the latest end among its
/// training rows) read back from its sidecar, so two fits whose training
/// ends at the same row can differ by float noise (a date resolved through
/// a different path, a value round-tripped through text). The holdout code
/// decides "the same time at the boundary" with [`crate::fit::runner::TIME_TIE_EPS`]; this uses the same
/// tolerance, so a cohort the fits treat as one window is not refused here
/// as two.
fn same_sealed_boundary(a: f64, b: f64) -> bool {
    (a - b).abs() <= crate::fit::runner::TIME_TIE_EPS
}

/// A comparison refused because its rows are not commensurable: exit 2, as
/// every such refusal always has, distinct from the 1 of an input error.
fn refuse(out: &crate::out::Out, message: String) -> ! {
    out.error(crate::out::CommandError { code: crate::out::ExitCode(2), message, hint: None }).exit()
}

// ── the cross-row checks, computed once (D13) ───────────────────────────

/// The waivers the command line gave.
#[derive(Debug, Clone, Copy, Default)]
struct Waivers {
    mismatched_horizon: bool,
    mixed_provenance: bool,
    data_mismatch: bool,
    mismatched_filter_settings: bool,
    in_sample_ablation: bool,
}

impl Waivers {
    fn from_args(a: &crate::args::CompareArgs) -> Self {
        Waivers {
            mismatched_horizon: a.allow_mismatched_horizon,
            mixed_provenance: a.allow_mixed_provenance,
            data_mismatch: a.allow_data_mismatch,
            mismatched_filter_settings: a.allow_mismatched_filter_settings,
            in_sample_ablation: a.allow_in_sample_ablation,
        }
    }

    fn allows(&self, w: Waiver) -> bool {
        match w {
            Waiver::MismatchedHorizon => self.mismatched_horizon,
            Waiver::MixedProvenance => self.mixed_provenance,
            Waiver::DataMismatch => self.data_mismatch,
            Waiver::MismatchedFilterSettings => self.mismatched_filter_settings,
            Waiver::InSampleAblation => self.in_sample_ablation,
        }
    }
}

/// The result of every preflight check over one table's rows, computed once
/// by [`preflight`] and read by every renderer, so no format can state a
/// different outcome from another, or from the refusals.
#[derive(Debug, Clone, Default, PartialEq)]
struct Preflight {
    /// One result per [`CheckId`], in declaration order.
    results: Vec<CheckResult>,
}

impl Preflight {
    fn result(&self, check: CheckId) -> Option<&CheckResult> {
        self.results.iter().find(|r| r.check == check)
    }

    /// The failed checks that were not waived, in check order.
    fn refusals(&self) -> Vec<Refusal> {
        self.results.iter().filter_map(CheckResult::refusal).collect()
    }

    /// The waivers that were given and needed, in check order.
    fn overrides(&self) -> Vec<Override> {
        self.results.iter().filter_map(CheckResult::override_).collect()
    }

    /// The finding of `check` when it failed and its waiver let it through.
    fn waived_finding(&self, check: CheckId) -> Option<&Finding> {
        match &self.result(check)?.outcome {
            CheckOutcome::Failed { finding, waived: true } => Some(finding),
            _ => None,
        }
    }

    /// Whether the rows scored different numbers of observations, so no Δ is
    /// formed (the comparison rendered under `--allow-mismatched-horizon`).
    fn horizons_differ(&self) -> bool {
        self.waived_finding(CheckId::ScoredSteps).is_some()
    }
}

/// Each distinct assimilated set among `rows`, with the rows that have it, in
/// the order first seen: "'full' assimilated cases, deaths; 'cases_only'
/// assimilated cases". A row whose set is not recorded is its own entry, since
/// an unknown set is not known to equal any other.
fn assimilated_sets(rows: &[Row]) -> Vec<String> {
    let mut groups: Vec<(Option<Vec<String>>, Vec<&str>)> = Vec::new();
    for r in rows {
        let set = r.assimilated();
        match groups.iter_mut().find(|(s, _)| s.is_some() && *s == set) {
            Some((_, names)) => names.push(&r.name),
            None => groups.push((set, vec![&r.name])),
        }
    }
    groups.into_iter().map(|(set, names)| {
        let who = names.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ");
        match set {
            Some(s) => format!("{who} assimilated {}", s.join(", ")),
            None => format!("{who} (not recorded: an explicit prequential.json)"),
        }
    }).collect()
}

/// The leaf a `--streams` table scores, from its rows' traces; `None` for a
/// joint comparison. The pairing check has shown the rows agree on it.
fn scored_leaf(rows: &[Row]) -> Option<&str> {
    rows.first()?.trace.scored_streams.as_ref()?.first().map(String::as_str)
}

/// The footer line of a `--streams` table whose rows assimilated different
/// streams, listing each distinct set with its rows; `None` otherwise.
fn assimilated_line(rows: &[Row]) -> Option<String> {
    (scored_leaf(rows).is_some() && assimilated_sets_differ(rows))
        .then(|| format!("Assimilated: {}.", assimilated_sets(rows).join("; ")))
}

/// The sentence a horizon or time refusal of a `--streams` table adds when
/// its rows assimilated different streams and start scoring the leaf at
/// different times; `None` otherwise.
fn first_step_note(rows: &[Row]) -> Option<&'static str> {
    scored_leaf(rows)?;
    let firsts: Vec<f64> = rows.iter().filter_map(|r| r.trace.steps.first().map(|s| s.t)).collect();
    (assimilated_sets_differ(rows) && firsts.windows(2).any(|w| w[0] != w[1])).then_some(
        "These fits assimilated different streams and start scoring this stream at \
         different times: a fit that conditions on its first observation (`ic_free`) \
         starts scoring one step later on its own time axis, and that axis depends on \
         which streams it assimilated.")
}

/// Whether `rows` assimilated different streams: some row's set is not
/// recorded, or two recorded sets differ.
fn assimilated_sets_differ(rows: &[Row]) -> bool {
    let sets: Vec<Option<Vec<String>>> = rows.iter().map(Row::assimilated).collect();
    sets.iter().any(Option::is_none) || sets.windows(2).any(|w| w[0] != w[1])
}

/// A trace's predictive basis, as the basis refusal names it (gh#312).
fn basis_kind(b: &PredictiveBasis) -> String {
    match b {
        PredictiveBasis::Filtered =>
            "filtered (a particle filter's one-step-ahead estimate)".to_string(),
        PredictiveBasis::OdeFreeForward { samples_per_draw } => format!(
            "ode_free_forward (the ODE skeleton run from t_start at each θ, scored \
             exactly; {samples_per_draw} observation-model draws per θ)"),
        PredictiveBasis::SimulatedFreeForward =>
            "simulated_free_forward (one stochastic trajectory per θ from t_start)".to_string(),
    }
}

fn provenance_kind(p: &score::trace::Provenance) -> String {
    use score::trace::Provenance;
    match p {
        Provenance::PlugIn => "plug_in (a single θ)".to_string(),
        Provenance::Posterior { n_draws } => format!("posterior (mixture over {n_draws} draws)"),
    }
}

fn conditioning_kind(c: &ConditioningMode) -> String {
    match c {
        ConditioningMode::InSample => "in_sample (θ saw the scored window)".to_string(),
        ConditioningMode::HoldOutTail { train_end } =>
            format!("hold_out_tail (θ sealed at t = {train_end})"),
        ConditioningMode::Unknown => "unknown (no record of the rows θ was fitted to)".to_string(),
        ConditioningMode::HoldOutDeclared { trained_through } =>
            format!("hold_out_declared (θ declared fitted through t = {trained_through}, not verified)"),
    }
}

/// Run every cross-row check over the loaded rows, in [`CheckId`] order, and
/// record each result. A check that reads what an earlier check established
/// ([`prerequisite`]) is recorded as not evaluated when that check failed
/// unwaived or was itself not evaluated. A failed check is waived when
/// `waived` gives its flag.
fn preflight(rows: &[Row], waived: &Waivers) -> Preflight {
    let mut results: Vec<CheckResult> = Vec::with_capacity(CheckId::ALL.len());
    for check in CheckId::ALL {
        let blocked = prerequisite(check, rows).filter(|p| {
            !results.iter().find(|r| r.check == *p)
                .expect("a prerequisite precedes the check that reads it")
                .established()
        });
        let mut result = match blocked {
            Some(because) => CheckResult {
                check,
                outcome: CheckOutcome::NotEvaluated { because },
                not_checked: Vec::new(),
                qualified: Vec::new(),
                waiver: check.waiver(),
            },
            None => evaluate(check, rows, &results),
        };
        let allowed = result.waiver.is_some_and(|w| waived.allows(w));
        if let CheckOutcome::Failed { waived, .. } = &mut result.outcome {
            *waived = allowed;
        }
        results.push(result);
    }
    Preflight { results }
}

/// The earlier check whose result `check` reads, if any.
///
/// - `TrainingBoundary` compares boundaries among held-out rows, and
///   `InSampleAblation` reads the one conditioning the rows share: both
///   need `Conditioning`.
/// - `Pairing` pairs steps by position when the rows scored the same number
///   of steps and joins them by time when they did not (under the waiver):
///   it needs `ScoredSteps`.
/// - `SharedData`, when a row is re-scored on one leaf, compares the hashes
///   of the leaf's source, the same for every row only once `Pairing` has
///   shown the rows are scored on the same leaves.
fn prerequisite(check: CheckId, rows: &[Row]) -> Option<CheckId> {
    match check {
        CheckId::TrainingBoundary | CheckId::InSampleAblation => Some(CheckId::Conditioning),
        CheckId::Pairing => Some(CheckId::ScoredSteps),
        CheckId::SharedData if rows.iter().any(|r| r.trace.scored_streams.is_some()) =>
            Some(CheckId::Pairing),
        _ => None,
    }
}

/// One check over `rows`; `earlier` holds the results of the checks before
/// it. The outcome of a failed check is not yet waived.
fn evaluate(check: CheckId, rows: &[Row], earlier: &[CheckResult]) -> CheckResult {
    match check {
        CheckId::Basis => check_basis(rows),
        CheckId::Conditioning => check_conditioning(rows),
        CheckId::TrainingBoundary => check_training_boundary(rows),
        CheckId::InSampleAblation => check_in_sample_ablation(rows),
        CheckId::Provenance => check_provenance(rows),
        CheckId::ScoredSteps => check_scored_steps(rows),
        CheckId::Pairing => {
            // With the step counts differing (and waived: otherwise this
            // check is not evaluated), steps are joined by time.
            let horizons_differ = earlier.iter().any(|r| r.check == CheckId::ScoredSteps
                && matches!(r.outcome, CheckOutcome::Failed { .. }));
            check_pairing(rows, horizons_differ)
        }
        CheckId::SharedData => check_shared_data(rows),
        CheckId::FilterSettings => check_filter_settings(rows),
        CheckId::Mode => fact_check(CheckId::Mode, rows, |f| f.mode.clone(), |groups| Finding::ModeDiffers { groups }),
        CheckId::Role => fact_check(CheckId::Role, rows, |f| f.role.clone(), |groups| Finding::RoleDiffers { groups }),
        CheckId::Lead => fact_check(CheckId::Lead, rows, |f| f.lead.clone(), |groups| Finding::LeadDiffers { groups }),
        CheckId::Reach => fact_check(CheckId::Reach, rows, |f| f.reach.clone(), |groups| Finding::ReachDiffers { groups }),
        CheckId::Covariates =>
            fact_check(CheckId::Covariates, rows, |f| f.realised, |groups| Finding::CovariatesDiffer { groups }),
        CheckId::ScoredRows => check_scored_rows(rows),
    }
}

/// A check of one [`ScoreFacts`] field: the rows partitioned by its value,
/// failed when they take more than one.
fn fact_check<V: Clone + Ord>(
    check: CheckId,
    rows: &[Row],
    value: impl Fn(&ScoreFacts) -> V,
    finding: impl Fn(pf::Groups<V>) -> Finding,
) -> CheckResult {
    let groups = pf::partition(by_row(rows, |r| value(&r.facts)), |a, b| a.cmp(b), |a, b| a == b);
    let failure = (groups.len() > 1).then(|| finding(groups));
    simple_result(check, failure, rows.len() >= 2)
}

/// The rows scored the same observed value of each stream at each time two
/// or more of them scored it: a score's outcomes are its own, so two rows
/// whose outcome files differ in a held-out row pair by time and stream and
/// differ only here. The first time and stream at which they differ is the
/// finding.
fn check_scored_rows(rows: &[Row]) -> CheckResult {
    let mut times: Vec<f64> = rows.iter().flat_map(|r| r.trace.steps.iter().map(|s| s.t)).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mut failure = None;
    'times: for t in times {
        let at: Vec<(&Row, &score::trace::PrequentialStep)> = rows.iter()
            .filter_map(|r| r.trace.steps.iter().find(|s| s.t == t).map(|s| (r, s)))
            .collect();
        let mut streams: Vec<&str> = at.iter().flat_map(|(_, s)| s.per_stream.iter().map(|x| x.stream.as_str())).collect();
        streams.sort();
        streams.dedup();
        for stream in streams {
            let items: Vec<(f64, String)> = at.iter().filter_map(|(r, s)| {
                s.per_stream.iter().find(|x| x.stream == stream).map(|x| (x.y_obs, r.name.clone()))
            }).collect();
            let groups = pf::partition(items, |a, b| a.total_cmp(b), |a, b| a.total_cmp(b).is_eq());
            if groups.len() > 1 {
                failure = Some(Finding::ScoredValuesDiffer { t, stream: stream.to_string(), groups });
                break 'times;
            }
        }
    }
    simple_result(CheckId::ScoredRows, failure, rows.len() >= 2)
}

/// A result with nothing not checked or qualified: failed on `failure`,
/// passed when the members compared, and nothing to compare otherwise.
fn simple_result(check: CheckId, failure: Option<Finding>, comparable: bool) -> CheckResult {
    CheckResult {
        check,
        outcome: outcome(failure, comparable),
        not_checked: Vec::new(),
        qualified: Vec::new(),
        waiver: check.waiver(),
    }
}

fn outcome(failure: Option<Finding>, comparable: bool) -> CheckOutcome {
    match failure {
        Some(finding) => CheckOutcome::Failed { finding, waived: false },
        None if comparable => CheckOutcome::Passed,
        None => CheckOutcome::NothingToCompare,
    }
}

/// `rows` as (value, name) items for [`score::preflight::partition`].
fn by_row<V>(rows: &[Row], value: impl Fn(&Row) -> V) -> Vec<(V, String)> {
    rows.iter().map(|r| (value(r), r.name.clone())).collect()
}

/// Whether the groups' values take more than one variant.
fn variants_differ<V>(groups: &[(V, Vec<String>)]) -> bool {
    groups.windows(2).any(|w| std::mem::discriminant(&w[0].0) != std::mem::discriminant(&w[1].0))
}

/// gh#312: rows on different predictive bases are never differenced. A
/// filter's score is a Monte-Carlo estimate whose log is biased downward by
/// an amount that depends on the model and the particle count; an ODE
/// trace's is exact. Their Δ would carry that bias as if it were a model
/// difference, and no setting makes the two estimators agree, so there is
/// no override.
fn check_basis(rows: &[Row]) -> CheckResult {
    let key = |b: &PredictiveBasis| match b {
        PredictiveBasis::Filtered => (0, 0),
        PredictiveBasis::OdeFreeForward { samples_per_draw } => (1, *samples_per_draw),
        PredictiveBasis::SimulatedFreeForward => (2, 0),
    };
    let groups = pf::partition(by_row(rows, |r| r.trace.basis),
        |a, b| key(a).cmp(&key(b)), |a, b| a == b);
    let failure = variants_differ(&groups).then_some(Finding::BasisDiffers { groups });
    simple_result(CheckId::Basis, failure, rows.len() >= 2)
}

/// Fairness, part zero (gh#585, Stage 3.4): rows with different
/// conditioning modes are never differenced. An in-sample score (θ saw the
/// scored window) and a held-out score (θ sealed before it) answer
/// different questions; their Δ is meaningless however displayed, so there
/// is no override: the fix is to make the modes agree.
fn check_conditioning(rows: &[Row]) -> CheckResult {
    let mode = |r: &Row| match &r.trace.conditioning {
        Conditioning::InSample => ConditioningMode::InSample,
        Conditioning::HoldOutTail { train_end, .. } =>
            ConditioningMode::HoldOutTail { train_end: *train_end },
        Conditioning::Unknown => ConditioningMode::Unknown,
        Conditioning::HoldOutDeclared { trained_through } =>
            ConditioningMode::HoldOutDeclared { trained_through: *trained_through },
    };
    let order = |a: &ConditioningMode, b: &ConditioningMode| match (a, b) {
        (ConditioningMode::InSample, ConditioningMode::InSample) => std::cmp::Ordering::Equal,
        (ConditioningMode::InSample, _) => std::cmp::Ordering::Less,
        (_, ConditioningMode::InSample) => std::cmp::Ordering::Greater,
        (ConditioningMode::HoldOutTail { train_end: x }, ConditioningMode::HoldOutTail { train_end: y }) =>
            x.total_cmp(y),
        (ConditioningMode::HoldOutTail { .. }, _) => std::cmp::Ordering::Less,
        (_, ConditioningMode::HoldOutTail { .. }) => std::cmp::Ordering::Greater,
        (ConditioningMode::Unknown, ConditioningMode::Unknown) => std::cmp::Ordering::Equal,
        (ConditioningMode::Unknown, _) => std::cmp::Ordering::Less,
        (_, ConditioningMode::Unknown) => std::cmp::Ordering::Greater,
        (ConditioningMode::HoldOutDeclared { trained_through: x }, ConditioningMode::HoldOutDeclared { trained_through: y }) =>
            x.total_cmp(y),
    };
    let groups = pf::partition(by_row(rows, mode), order, |a, b| a == b);
    let failure = variants_differ(&groups).then_some(Finding::ConditioningDiffers { groups });
    simple_result(CheckId::Conditioning, failure, rows.len() >= 2)
}

/// All held-out: the sealed boundaries must also agree, or the rows score
/// different windows (the pairing check would refuse anyway, but this names
/// the actual cause). Boundaries within [`same_sealed_boundary`] of a
/// group's smallest are one group.
fn check_training_boundary(rows: &[Row]) -> CheckResult {
    let held: Vec<(f64, String)> = rows.iter().filter_map(|r| match &r.trace.conditioning {
        Conditioning::HoldOutTail { train_end, .. } => Some((*train_end, r.name.clone())),
        Conditioning::InSample | Conditioning::Unknown | Conditioning::HoldOutDeclared { .. } => None,
    }).collect();
    let comparable = held.len() >= 2;
    let groups = pf::partition(held, |a, b| a.total_cmp(b), |a, b| same_sealed_boundary(*a, *b));
    let failure = (groups.len() > 1).then_some(Finding::TrainingBoundaryDiffers { groups });
    simple_result(CheckId::TrainingBoundary, failure, comparable)
}

/// A table re-scored on one leaf (`--streams`) whose rows' fits assimilated
/// different streams is the stream-ablation experiment. In-sample it is
/// biased by construction toward the fit trained on fewer streams; held
/// out, neither fit's θ̂ was tuned to the scored rows. The rows share one
/// conditioning mode (the check's prerequisite).
fn check_in_sample_ablation(rows: &[Row]) -> CheckResult {
    let rescored = rows.iter().any(|r| r.trace.scored_streams.is_some());
    let in_sample = rows.iter().all(|r| r.trace.conditioning == Conditioning::InSample);
    let failure = (rescored && in_sample && rows.len() >= 2 && assimilated_sets_differ(rows))
        .then(|| Finding::InSampleAblation {
            groups: pf::partition(by_row(rows, Row::assimilated), |a, b| a.cmp(b),
                // An unknown set is not known to equal any other.
                |a, b| a.is_some() && a == b),
        });
    simple_result(CheckId::InSampleAblation, failure, rows.len() >= 2)
}

/// Provenance preflight (Stage 4.1): a plug-in trace (under-dispersed: one
/// θ) against a posterior mixture is a Δ between predictives of different
/// dispersion. Overridable, unlike conditioning: an optimizer fit has no
/// posterior and comparing it against a Bayesian fit is a legitimate ask, as
/// long as the reader chose it.
fn check_provenance(rows: &[Row]) -> CheckResult {
    use score::trace::Provenance;
    let key = |p: &Provenance| match p {
        Provenance::PlugIn => (0, 0),
        Provenance::Posterior { n_draws } => (1, *n_draws),
    };
    let groups = pf::partition(by_row(rows, |r| r.trace.provenance),
        |a, b| key(a).cmp(&key(b)), |a, b| a == b);
    let failure = variants_differ(&groups).then_some(Finding::ProvenanceDiffers { groups });
    simple_result(CheckId::Provenance, failure, rows.len() >= 2)
}

/// Fairness: T_score (the number of scored steps) must agree across rows
/// unless overridden.
fn check_scored_steps(rows: &[Row]) -> CheckResult {
    let groups = pf::partition(by_row(rows, |r| r.trace.n_scored()), |a, b| a.cmp(b), |a, b| a == b);
    let failure = (groups.len() > 1).then_some(Finding::ScoredStepsDiffer { groups });
    simple_result(CheckId::ScoredSteps, failure, rows.len() >= 2)
}

/// Fairness, part two: equal T_score is not an equal observation axis.
/// `paired_delta` zips steps by index, so two traces scoring the same number
/// of observations at different times (a hole in one series, a different
/// `t0`) produce a Δelpd, an se(Δ) and a deciban verdict computed across two
/// different axes, rendered as a confident answer. Not overridable by
/// `--allow-mismatched-horizon`: that flag says "render the Δ columns as '—'
/// because the horizons differ", whereas this is a pairing that is
/// meaningless however it is displayed.
///
/// With the horizons differing (and the flag set: otherwise this check is
/// not evaluated), the flag waives the times only one row scored, and
/// nothing else: at every time two or more rows scored, the stream-set
/// (gh#570) and coverage (gh#833) checks still hold.
fn check_pairing(rows: &[Row], horizons_differ: bool) -> CheckResult {
    simple_result(CheckId::Pairing, pairing_finding(rows, horizons_differ), rows.len() >= 2)
}

/// The first way `rows` fail to pair, or `None`.
///
/// The rows must score the same quantity: all jointly over every leaf their
/// fit observed, or all re-scored on the same leaves
/// (`PrequentialTrace::scored_streams`); a step's log score means a different
/// quantity otherwise, even where the steps' streams agree. Then, step by
/// step (by position when `by_time` is false, at each time two or more rows
/// scored when it is true), the rows must be at the same time and pass
/// [`step_finding`].
fn pairing_finding(rows: &[Row], by_time: bool) -> Option<Finding> {
    let quantity = pf::partition(by_row(rows, |r| r.trace.scored_streams.clone()),
        |a, b| a.cmp(b), |a, b| a == b);
    if quantity.len() > 1 {
        return Some(Finding::ScoredQuantityDiffers { groups: quantity });
    }
    if !by_time {
        let n = rows.iter().map(|r| r.trace.steps.len()).min().unwrap_or(0);
        for k in 0..n {
            let times = pf::partition(by_row(rows, |r| r.trace.steps[k].t),
                |a, b| a.total_cmp(b), |a, b| a == b);
            if times.len() > 1 {
                return Some(Finding::ObservationTimesDiffer { step: k + 1, groups: times });
            }
            let at: Vec<(&Row, &score::trace::PrequentialStep)> =
                rows.iter().map(|r| (r, &r.trace.steps[k])).collect();
            if let Some(f) = step_finding(&at) {
                return Some(f);
            }
        }
        return None;
    }
    // Step times come from the same observation grid, so a shared time is
    // the same value; this is the equality the equal-horizon path pairs on.
    let mut times: Vec<f64> = rows.iter().flat_map(|r| r.trace.steps.iter().map(|s| s.t)).collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    for t in times {
        let at: Vec<(&Row, &score::trace::PrequentialStep)> = rows.iter()
            .filter_map(|r| r.trace.steps.iter().find(|s| s.t == t).map(|s| (r, s)))
            .collect();
        if at.len() >= 2 {
            if let Some(f) = step_finding(&at) {
                return Some(f);
            }
        }
    }
    None
}

/// The rows' steps at one observation time scored the same streams, and the
/// same quantity for each stream.
///
/// A v1 trace carries no per-stream breakdown at all (an empty stream set).
/// Two such traces have nothing to check and pass; one of each cannot be
/// checked, and an unverifiable stream set is not a verified one.
///
/// The coverage half (gh#833): two fits of one file that differ only in what
/// a row is declared to cover (a gapped window against a merged one, a
/// stream read as an instant against the same name read as a flow) score at
/// the same times and produce different likelihoods, and nothing else in the
/// pipeline can tell that apart from a model difference. An `Unrecorded`
/// coverage (a trace older than schema 4) is treated as the v1 case is.
fn step_finding(at: &[(&Row, &score::trace::PrequentialStep)]) -> Option<Finding> {
    use score::trace::Coverage;
    let t = at.first()?.1.t;
    let streams_of = |s: &score::trace::PrequentialStep| -> Vec<String> {
        step_streams(s).into_iter().map(str::to_string).collect()
    };
    let sets = pf::partition(at.iter().map(|(r, s)| (streams_of(s), r.name.clone())).collect(),
        |a, b| a.cmp(b), |a, b| a == b);
    if sets.len() > 1 {
        // The streams every one of these rows scores anywhere: `--streams`
        // compares the rows on those, each alone.
        let mut shared: Vec<String> = at[0].0.trace.streams().into_iter().map(str::to_string).collect();
        for (r, _) in &at[1..] {
            let theirs = r.trace.streams();
            shared.retain(|s| theirs.contains(&s.as_str()));
        }
        shared.sort();
        return Some(Finding::StreamSetsDiffer { t, groups: sets, shared_streams: shared });
    }
    let key = |c: &Coverage| match c {
        Coverage::Unrecorded => (0, 0.0, 0.0),
        Coverage::Instant => (1, 0.0, 0.0),
        Coverage::Interval { start, stop } => (2, *start, *stop),
    };
    let order = |a: &Coverage, b: &Coverage| {
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1)).then(ka.2.total_cmp(&kb.2))
    };
    for stream in &sets[0].0 {
        let items: Vec<(Coverage, String)> = at.iter().filter_map(|(r, s)| {
            s.per_stream.iter().find(|x| &x.stream == stream).map(|x| (x.coverage, r.name.clone()))
        }).collect();
        let groups = pf::partition(items, order, |a, b| a == b);
        if groups.len() > 1 {
            return Some(Finding::CoverageDiffers { t, stream: stream.clone(), groups });
        }
    }
    None
}

/// Fairness, part three (gh#713): the compared fits must have been bound to
/// the same observations. Two fits on different data produce a Δelpd that is
/// part model difference and part data difference, with nothing in the
/// table to say which. Overridable, unlike the pairing check: comparing a fit
/// on revised counts against the fit on the original ones is a real thing
/// to want, as long as the reader knows the Δ is confounded.
///
/// `compare @a @b` re-filters each fit at its own θ̂ and differences the
/// scores. If the two fits read different data (a corrected case series, a
/// stream added between runs, a file edited in place) the difference carries
/// both effects and the table attributes all of it to the models. Nothing
/// else in the pipeline catches this: each fit is internally consistent, and
/// the derived traces agree on T_score, times and stream names, because
/// those come from the same model.
///
/// Per source, the rows that record a hash for it or bind it not at all are
/// compared pair by pair, each pair on the shorter of its two hashes (an
/// older sidecar records 8 characters of the digest a newer one records in
/// full), and partitioned by [`hash_groups`]; an empty hash names nothing and
/// matches nothing, and a row that binds the source beside one that does not
/// is a failure. A row with no data identity, and a source a row could not
/// resolve, take no part and are listed as not checked. The rows whose
/// comparison rested on a short hash are qualified.
///
/// Rows re-scored on one leaf (`--streams`) compare only the hashes of that
/// leaf's source, and an absent hash for it is a failed check, not a pass;
/// the pairing check (this check's prerequisite then) has established that
/// every row is scored on the same leaves.
fn check_shared_data(rows: &[Row]) -> CheckResult {
    let leaves = rows.iter().find_map(|r| r.trace.scored_streams.as_ref());
    let mut unhashed: Vec<UnhashedData> = Vec::new();
    let entries: Vec<(&str, DataIdentity)> = rows.iter().map(|r| match leaves {
        None => (r.name.as_str(), r.data.clone()),
        Some(leaves) => {
            let (d, missing) = data_for_streams(r, leaves);
            unhashed.extend(missing.into_iter()
                .map(|what| UnhashedData { member: r.name.clone(), what }));
            (r.name.as_str(), d)
        }
    }).collect();
    pf::sort_by_member(&mut unhashed, |u| &u.member);

    let mut not_checked: Vec<Unchecked> = Vec::new();
    for (name, d) in &entries {
        let unchecked = |reason| Unchecked { member: name.to_string(), reason };
        match d {
            DataIdentity::Digests(rec) => {
                for (source, why) in &rec.unresolved {
                    not_checked.push(unchecked(NotCheckedReason::SourceUnresolved {
                        source: source.clone(), why: why.clone() }));
                }
            }
            DataIdentity::Unknown(why) =>
                not_checked.push(unchecked(NotCheckedReason::NoDataIdentity { why: why.clone() })),
        }
    }
    pf::sort_by_member(&mut not_checked, |u| &u.member);

    let known: Vec<(&str, &RecordedData)> = entries.iter().filter_map(|(name, d)| match d {
        DataIdentity::Digests(rec) => Some((*name, rec)),
        DataIdentity::Unknown(_) => None,
    }).collect();
    let mut sources: Vec<SourceDigests> = Vec::new();
    let mut prefix_only: Vec<String> = Vec::new();
    let all: std::collections::BTreeSet<&String> =
        known.iter().flat_map(|(_, rec)| rec.training.keys()).collect();
    for source in all {
        let taking_part: Vec<(Option<String>, String)> = known.iter()
            .filter(|(_, rec)| !rec.unresolved.contains_key(source))
            .map(|(name, rec)| (rec.training.get(source).map(|h| h.to_ascii_lowercase()), name.to_string()))
            .collect();
        let (groups, short) = hash_groups(taking_part);
        prefix_only.extend(short);
        if groups.len() > 1 {
            sources.push(SourceDigests { source: source.clone(), groups });
        }
    }
    let failure = (!sources.is_empty() || !unhashed.is_empty())
        .then_some(Finding::DataDiffers { sources, unhashed });
    let comparable = known.len() >= 2 || failure.is_some();
    prefix_only.sort();
    prefix_only.dedup();
    CheckResult {
        check: CheckId::SharedData,
        outcome: outcome(failure, comparable),
        not_checked,
        qualified: prefix_only.into_iter()
            .map(|member| Qualified { member, how: Qualification::HashPrefixOnly }).collect(),
        waiver: CheckId::SharedData.waiver(),
    }
}

/// One source's recorded hashes (lowercased; `None` where the member records
/// none) partitioned for [`check_shared_data`], and the members whose
/// comparison rested on a hash shorter than a full SHA-256.
///
/// Two hashes match when one is a non-empty prefix of the other: each pair is
/// compared on the shorter of its two lengths, so a short hash in one member
/// does not shorten the comparison of two full ones. The source passes when
/// every two recorded hashes match and no member lacks one beside a member
/// that has one, that is when there is one group:
///
/// - a hash that is a proper prefix of no other recorded hash is a group's
///   value, and the members recording it are the group;
/// - a shorter hash that is the prefix of exactly one such value joins that
///   group; one that is the prefix of two or more (which differ, so the
///   source fails) is shown on its own value, as its own group;
/// - an empty hash names nothing and matches nothing, not even another
///   empty one: each such member is its own group;
/// - the members recording no hash are one group, under `None`.
///
/// A member is returned as compared on a prefix when it records a non-empty
/// hash shorter than [`FULL_DIGEST_LEN`] and some other member records a
/// non-empty hash for the source.
fn hash_groups(items: Vec<(Option<String>, String)>) -> (pf::Groups<Option<String>>, Vec<String>) {
    let recorded: std::collections::BTreeSet<&str> =
        items.iter().filter_map(|(h, _)| h.as_deref()).filter(|h| !h.is_empty()).collect();
    let longest: Vec<&str> = recorded.iter().copied()
        .filter(|h| !recorded.iter().any(|o| o != h && o.starts_with(h)))
        .collect();
    let with_hash = items.iter().filter(|(h, _)| h.as_deref().is_some_and(|h| !h.is_empty())).count();
    let short: Vec<String> = items.iter()
        .filter(|(h, _)| h.as_deref().is_some_and(|h| !h.is_empty() && h.len() < FULL_DIGEST_LEN))
        .filter(|_| with_hash >= 2)
        .map(|(_, name)| name.clone())
        .collect();
    let keyed: Vec<(Option<String>, String)> = items.iter().map(|(h, name)| {
        let key = h.as_deref().map(|h| {
            let extensions: Vec<&str> = longest.iter().copied()
                .filter(|l| !h.is_empty() && l.starts_with(h)).collect();
            match extensions.as_slice() {
                [only] => only.to_string(),
                _ => h.to_string(),
            }
        });
        (key, name.clone())
    }).collect();
    let groups = pf::partition(keyed, |a, b| a.cmp(b), |a, b| a == b && a.as_deref() != Some(""));
    (groups, short)
}

/// Two rows scored at different particle or replicate counts differ partly
/// by the filters. Overridable: comparing a cheap explicit trace against a
/// derived row is a legitimate ask once the reader knows. Only rows that
/// record their settings take part; the others are listed as not checked.
fn check_filter_settings(rows: &[Row]) -> CheckResult {
    let recorded: Vec<(&str, &score::trace::Derivation)> = rows.iter()
        .filter_map(|r| r.trace.derivation.as_ref().map(|d| (r.name.as_str(), d)))
        .collect();
    let counts = |of: fn(&score::trace::Derivation) -> usize| pf::partition(
        recorded.iter().map(|(name, d)| (of(d), name.to_string())).collect(),
        |a, b| a.cmp(b), |a, b| a == b);
    let particles = counts(|d| d.particles);
    let replicates = counts(|d| d.replicates);
    let failure = (particles.len() > 1 || replicates.len() > 1)
        .then_some(Finding::FilterSettingsDiffer { particles, replicates });
    let mut not_checked: Vec<Unchecked> = rows.iter()
        .filter(|r| r.trace.derivation.is_none())
        .map(|r| Unchecked {
            member: r.name.clone(),
            reason: if r.trace.schema_version < 5 {
                NotCheckedReason::TraceBeforeSchema5
            } else {
                NotCheckedReason::NoDerivationBlock
            },
        })
        .collect();
    pf::sort_by_member(&mut not_checked, |u| &u.member);
    CheckResult {
        check: CheckId::FilterSettings,
        outcome: outcome(failure, recorded.len() >= 2),
        not_checked,
        qualified: Vec::new(),
        waiver: CheckId::FilterSettings.waiver(),
    }
}

// ── the sentences the checks' results are rendered as ─────────────────

/// `'a', 'b'`.
fn quoted(names: &[String]) -> String {
    names.iter().map(|n| format!("'{n}'")).collect::<Vec<_>>().join(", ")
}

/// `'a'`, `'a' and 'b'`, `'a', 'b' and 'c'`.
fn quoted_and(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => format!("'{one}'"),
        [rest @ .., last] => format!("{} and '{last}'", quoted(rest)),
    }
}

fn plain(names: &[String]) -> String {
    names.join(", ")
}

/// The finding as one statement, the form an override and the JSON state it
/// in. For the checks no flag waives, the statement of the refusal.
fn finding_text(f: &Finding) -> String {
    match f {
        Finding::BasisDiffers { groups } => format!("the compared traces mix predictive bases: {}",
            groups.iter().map(|(b, m)| format!("{}: {}", plain(m), basis_kind(b)))
                .collect::<Vec<_>>().join("; ")),
        Finding::ConditioningDiffers { groups } => format!(
            "the compared traces mix conditioning modes: {}",
            groups.iter().map(|(c, m)| format!("{}: {}", plain(m), conditioning_kind(c)))
                .collect::<Vec<_>>().join("; ")),
        Finding::TrainingBoundaryDiffers { groups } => format!(
            "the compared fits seal training at different boundaries: {}",
            groups.iter().map(|(t, m)| format!("{} at t = {t}", quoted(m)))
                .collect::<Vec<_>>().join("; ")),
        Finding::InSampleAblation { groups } => format!(
            "the compared fits assimilated different streams and are scored in-sample: {}",
            groups.iter().map(|(set, m)| match set {
                Some(s) => format!("{} assimilated {}", quoted(m), s.join(", ")),
                None => format!("{} (not recorded: an explicit prequential.json)", quoted(m)),
            }).collect::<Vec<_>>().join("; ")),
        Finding::ProvenanceDiffers { groups } => format!(
            "the compared traces mix provenance kinds: {}",
            groups.iter().map(|(p, m)| format!("{} {}", quoted(m), provenance_kind(p)))
                .collect::<Vec<_>>().join("; ")),
        Finding::ScoredStepsDiffer { groups } => format!("T_score differs across models: {}",
            groups.iter().map(|(n, m)| format!("{} {n}", quoted(m)))
                .collect::<Vec<_>>().join("; ")),
        Finding::ScoredQuantityDiffers { groups } => {
            let what = |set: &Option<Vec<String>>, m: &[String]| {
                let (is, fit) = if m.len() == 1 { ("is", "its fit") } else { ("are", "their fits") };
                match set {
                    None => format!("{} {is} scored jointly over every stream {fit} observed", quoted(m)),
                    Some(s) => format!("{} {is} re-scored on {} alone", quoted(m), s.join(", ")),
                }
            };
            format!("{} score different quantities: {}.\n       \
                     Compare traces scored the same way.",
                quoted_and(&f.members()),
                groups.iter().map(|(s, m)| what(s, m)).collect::<Vec<_>>().join(", and "))
        }
        Finding::ObservationTimesDiffer { step, groups } => format!(
            "{} are not on the same observation axis: scored step {step} is {}.\n       \
             Δelpd pairs observations by position, so differencing these \
             would compare unlike times.\n       \
             Re-score both models on the same observation set \
             (check for a hole in one series, a different t0, or a \
             different `covers` declaration — a period is scored at its \
             stop, so `day(t)` scores one bucket after `closing_at`).",
            quoted_and(&f.members()),
            groups.iter().map(|(t, m)| format!("t={t} in {}", quoted_and(m)))
                .collect::<Vec<_>>().join(" and ")),
        Finding::StreamSetsDiffer { t, groups, shared_streams } => {
            let set = |s: &[String]| format!("{{{}}}", s.join(", "));
            let detail = match groups.iter().find(|(s, _)| s.is_empty()) {
                Some((_, without)) => {
                    let others: Vec<&(Vec<String>, Vec<String>)> =
                        groups.iter().filter(|(s, _)| !s.is_empty()).collect();
                    let with: Vec<String> = others.iter().flat_map(|(_, m)| m.clone()).collect();
                    format!(
                        "{} has no per-stream breakdown at that step, so its \
                         joint score cannot be shown to cover the same streams as \
                         {} ({}).\n       \
                         Re-derive {} — a v1 `prequential.json` predates \
                         the per-stream breakdown.",
                        quoted_and(without), quoted_and(&with),
                        others.iter().map(|(s, _)| set(s)).collect::<Vec<_>>().join(", "),
                        quoted_and(without))
                }
                None => {
                    let hint = if shared_streams.is_empty() {
                        String::new()
                    } else {
                        format!("\n       To compare them on the streams they share, each scored \
                                 alone, pass --streams {}.", shared_streams.join(","))
                    };
                    format!(
                        "{}.\n       \
                         The step's log score is the JOINT score over those streams, \
                         so differencing them compares unlike quantities.\n       \
                         Re-score both models on the same set of observation streams.{hint}",
                        groups.iter().enumerate().map(|(i, (s, m))| if i == 0 {
                            format!("{} scored {} there", quoted(m), set(s))
                        } else {
                            format!("{} scored {}", quoted(m), set(s))
                        }).collect::<Vec<_>>().join(" and "))
                }
            };
            format!("{} did not score the same streams at t={t}: {detail}", quoted_and(&f.members()))
        }
        Finding::CoverageDiffers { t, stream, groups } => {
            use score::trace::Coverage;
            match groups.iter().find(|(c, _)| *c == Coverage::Unrecorded) {
                Some((_, without)) => {
                    let others: Vec<&(Coverage, Vec<String>)> =
                        groups.iter().filter(|(c, _)| *c != Coverage::Unrecorded).collect();
                    let with: Vec<String> = others.iter().flat_map(|(_, m)| m.clone()).collect();
                    format!(
                        "{} cannot be shown to score the same quantity for '{stream}' at \
                         t={t}: {} does not record what its rows cover, so its score \
                         cannot be shown to span the same window as {} ({}).\n       \
                         Re-derive {} — a `prequential.json` older than schema 4 \
                         predates the coverage record.",
                        quoted_and(&f.members()), quoted_and(without), quoted_and(&with),
                        others.iter().map(|(c, _)| fmt_coverage(*c)).collect::<Vec<_>>().join(", "),
                        quoted_and(without))
                }
                None => format!(
                    "{} score '{stream}' over different windows at t={t}: {}.\n       \
                     The two were scored at the same time but not against the same quantity — \
                     a different `covers` declaration or different window columns — and \
                     Δelpd would report that as a model difference.\n       \
                     Re-fit both with the same declaration of what each row covers.",
                    quoted_and(&f.members()),
                    groups.iter().map(|(c, m)| format!("{} in {}", fmt_coverage(*c), quoted_and(m)))
                        .collect::<Vec<_>>().join(" and ")),
            }
        }
        Finding::DataDiffers { sources, unhashed } => {
            let missing = (!unhashed.is_empty()).then(|| format!(
                "no hash is recorded for the data this table scores: {}",
                unhashed.iter().map(|u| match &u.what {
                    Unhashed::StreamNotBound { stream } => format!(
                        "'{}' does not bind stream '{stream}', so no hash of its data is recorded",
                        u.member),
                    Unhashed::NoHashForSource { source } =>
                        format!("'{}' records no hash for source '{source}'", u.member),
                }).collect::<Vec<_>>().join("; ")));
            let differing = (!sources.is_empty()).then(|| {
                let mut members: Vec<String> =
                    sources.iter().flat_map(|s| s.groups.iter().flat_map(|(_, m)| m.clone())).collect();
                members.sort();
                members.dedup();
                format!("{} were fit to different observed data: {}", quoted_and(&members),
                    sources.iter().map(source_digests_text).collect::<Vec<_>>().join("; "))
            });
            match (missing, differing) {
                (Some(a), Some(b)) => format!("{a}; and {b}"),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => String::new(),
            }
        }
        Finding::FilterSettingsDiffer { particles, replicates } => {
            let dimension = |label: &str, g: &[(usize, Vec<String>)]| (g.len() > 1).then(|| {
                g.iter().map(|(v, m)| format!("{label} {v} in {}", quoted_and(m)))
                    .collect::<Vec<_>>().join(" vs ")
            });
            format!("the compared traces were scored with different particle-filter settings: {}",
                [dimension("particles", particles), dimension("replicates", replicates)]
                    .into_iter().flatten().collect::<Vec<_>>().join("; "))
        }
        Finding::ModeDiffers { groups } => format!("the compared scores' rows were predicted in different modes \
            (mode): {}", groups.iter().map(|(m, n)| format!("{} {m}", quoted_and(n))).collect::<Vec<_>>().join("; ")),
        Finding::RoleDiffers { groups } => format!("the compared scores' rows have different roles (role): {}",
            groups.iter().map(|(r, n)| format!("{} {r}", quoted_and(n))).collect::<Vec<_>>().join("; ")),
        Finding::LeadDiffers { groups } => format!("the compared scores' rows are at different leads (lead): {}",
            groups.iter().map(|(l, n)| format!("{} {}", quoted_and(n), l.as_deref().unwrap_or("none")))
                .collect::<Vec<_>>().join("; ")),
        Finding::ReachDiffers { groups } => format!("the compared forecasts reach different distances past their \
            origin (reach): {}", groups.iter().map(|(r, n)| format!("{} {}", quoted_and(n),
                r.as_deref().map_or("none (not a forecast)".to_string(), |r| format!("{r:?}"))))
                .collect::<Vec<_>>().join("; ")),
        Finding::CovariatesDiffer { groups } => format!("the compared forecasts state their covariates after the \
            origin differently (covariates): {}", groups.iter().map(|(realised, n)| format!("{} {}", quoted_and(n),
                if *realised { "read covariates observed after the origin (realised)" }
                else { "read no covariate stated realised" }))
                .collect::<Vec<_>>().join("; ")),
        Finding::ScoredValuesDiffer { t, stream, groups } => format!("the compared scores scored different observed \
            values (scored rows): '{stream}' at t = {t} is {}",
            groups.iter().map(|(y, n)| format!("{y} in {}", quoted_and(n))).collect::<Vec<_>>().join(" and ")),
    }
}

/// One source of a data mismatch: `'cases' (aaaaaaaa in 'a', bbbbbbbb in
/// 'b')`, or `'east' (bound by 'b' only)`.
fn source_digests_text(s: &SourceDigests) -> String {
    let bound: Vec<&(Option<String>, Vec<String>)> =
        s.groups.iter().filter(|(h, _)| h.is_some()).collect();
    let unbound: Vec<String> =
        s.groups.iter().filter(|(h, _)| h.is_none()).flat_map(|(_, m)| m.clone()).collect();
    let detail = match (bound.as_slice(), unbound.is_empty()) {
        ([(_, only)], false) => format!("bound by {} only", quoted_and(only)),
        _ => {
            let hashes = bound.iter()
                .map(|(h, m)| format!("{} in {}", short_digest(h.as_deref().unwrap_or("")), quoted_and(m)))
                .collect::<Vec<_>>().join(", ");
            if unbound.is_empty() { hashes } else { format!("{hashes}; not bound by {}", quoted_and(&unbound)) }
        }
    };
    format!("'{}' ({detail})", s.source)
}

/// The finding and its consequence as one statement.
fn finding_note(f: &Finding, w: Waiver) -> String {
    format!("{}. {}", finding_text(f), w.consequence())
}

/// The bound-data statement: the finding, its consequence and the remedy.
/// The refusal and the `--allow-data-mismatch` JSON field share it.
fn shared_data_statement(f: &Finding) -> String {
    format!("{}.\n       {}\n       Re-fit both on the same \
        observations, or pass --allow-data-mismatch to render the comparison as \
        confounded.", finding_text(f), Waiver::DataMismatch.consequence())
}

/// What `compare` prints for one refusal: the finding, what it does to the
/// comparison and how to proceed. `rows` are the table's, for the sentence
/// a `--streams` table's horizon or time refusal adds.
fn refusal_message(r: &Refusal, rows: &[Row]) -> String {
    let first_step = first_step_note(rows).map(|n| format!("\n       {n}")).unwrap_or_default();
    let lines = |head: &str, groups: Vec<String>| {
        let mut m = format!("error: {head}:");
        for g in groups {
            m.push_str(&format!("\n       {g}"));
        }
        m
    };
    match &r.finding {
        Finding::BasisDiffers { groups } => format!("{}\n       A particle filter's log score is a \
                   Monte-Carlo estimate, biased downward by an amount that depends on the model \
                   and the particle count; an ODE trace's is exact. Their Δelpd would mix that \
                   bias into the model difference. Compare ODE fits with ODE fits, and filtered \
                   fits with filtered fits.",
            lines("the compared traces mix predictive bases", groups.iter()
                .map(|(b, m)| format!("{}: {}", plain(m), basis_kind(b))).collect())),
        Finding::ConditioningDiffers { groups } => format!("{}\n       Declare the same holdout \
                   in every compared fit, or pass --in-sample to score all of them in-sample.",
            lines("the compared traces mix conditioning modes", groups.iter()
                .map(|(c, m)| format!("{}: {}", plain(m), conditioning_kind(c))).collect())),
        Finding::TrainingBoundaryDiffers { .. } => format!("error: {}\n       Held-out scores \
                   over different windows are not commensurable; declare the same \
                   holdout_after in every compared fit.", finding_text(&r.finding)),
        Finding::InSampleAblation { .. } => format!("error: {}.\n       {}\n       Declare the \
                   same holdout_after in every compared fit, so the rows are scored held-out, \
                   or pass --allow-in-sample-ablation to render anyway.",
            finding_text(&r.finding), Waiver::InSampleAblation.consequence()),
        Finding::ProvenanceDiffers { groups } => format!("{}\n       {}\n       Compare \
                   posteriors with posteriors and single points with single points, or pass \
                   --allow-mixed-provenance to render anyway.",
            lines("the compared traces mix provenance kinds", groups.iter()
                .map(|(p, m)| format!("{}: {}", plain(m), provenance_kind(p))).collect()),
            Waiver::MixedProvenance.consequence()),
        Finding::ScoredStepsDiffer { .. } => format!("error: {}\n       Δelpd and Δcrps are not \
                   commensurable.\n       Pass --allow-mismatched-horizon to render \
                   (uncomparable Δ columns → '—').{first_step}", finding_text(&r.finding)),
        Finding::ScoredQuantityDiffers { .. }
        | Finding::ObservationTimesDiffer { .. }
        | Finding::StreamSetsDiffer { .. }
        | Finding::CoverageDiffers { .. } =>
            format!("error: {}{first_step}", finding_text(&r.finding)),
        Finding::DataDiffers { .. } => format!("error: {}", shared_data_statement(&r.finding)),
        Finding::FilterSettingsDiffer { .. } => format!("error: {}\n       Pass compare the \
                   --particles and --replicates of the rows it reads as written (a prequential.json), \
                   so every prediction it makes matches them, or pass \
                   --allow-mismatched-filter-settings to render anyway.",
            finding_note(&r.finding, Waiver::MismatchedFilterSettings)),
        Finding::ModeDiffers { .. } => format!("error: {}.\n       A free-forward, a one-step and a \
                   forecast score answer different questions; compare scores of one mode (--mode).",
            finding_text(&r.finding)),
        Finding::RoleDiffers { .. } => format!("error: {}.\n       Scores of rows the parameters were \
                   fitted to and of rows they were not are never differenced; compare the same rows of \
                   every model (--in-sample for the training rows, or the same held-out rows).",
            finding_text(&r.finding)),
        Finding::LeadDiffers { .. } => format!("error: {}.\n       A score at one lead and one at \
                   another are different measurements; forecast every model to the same --horizon.",
            finding_text(&r.finding)),
        Finding::ReachDiffers { .. } => format!("error: {}.\n       Forecast every model to the same \
                   --horizon.", finding_text(&r.finding)),
        Finding::CovariatesDiffer { .. } => format!("error: {}.\n       A forecast that read covariates \
                   observed after its origin is an after-the-fact evaluation, and its scores flatter the \
                   model; state the covariates of every forecast alike (`camdl predict --covariate`).",
            finding_text(&r.finding)),
        Finding::ScoredValuesDiffer { .. } => format!("error: {}.\n       {}\n       Score every model \
                   against the same outcome files (--data), or pass --allow-data-mismatch to render the \
                   comparison as confounded.", finding_text(&r.finding), Waiver::DataMismatch.consequence()),
    }
}

/// The one line every human rendering prints for an override.
fn override_line(o: &Override) -> String {
    format!("⚠ {} was set for {}: {}", o.waiver.flag(), quoted(&o.finding.members()),
        finding_note(&o.finding, o.waiver))
}

fn override_json(o: &Override) -> serde_json::Value {
    serde_json::json!({
        "flag": o.waiver.flag(),
        "rows": o.finding.members(),
        "finding": finding_text(&o.finding),
        "consequence": o.waiver.consequence(),
    })
}

/// A check's entries in the order of the rows they name, each row's own
/// entries in the order the check recorded them.
fn in_row_order<'a, T>(entries: &'a [T], member: impl Fn(&T) -> &str, rows: &[Row]) -> Vec<&'a T> {
    let mut out: Vec<&T> = entries.iter().collect();
    out.sort_by_key(|e| rows.iter().position(|r| r.name == member(e)).unwrap_or(usize::MAX));
    out
}

fn not_checked<'a>(checks: &'a Preflight, check: CheckId, rows: &[Row]) -> Vec<&'a Unchecked> {
    checks.result(check)
        .map(|r| in_row_order(&r.not_checked, |u| &u.member, rows))
        .unwrap_or_default()
}

/// The rows and sources the bound-data check could not cover, and why, or
/// `None` when every row carried digests. Stated even when the check itself
/// fails, so the scope of the check is known.
fn data_unchecked_note(checks: &Preflight, rows: &[Row]) -> Option<String> {
    let skipped: Vec<String> = not_checked(checks, CheckId::SharedData, rows).into_iter()
        .map(|u| match &u.reason {
            NotCheckedReason::NoDataIdentity { why } => format!("'{}' ({why})", u.member),
            NotCheckedReason::SourceUnresolved { source, why } =>
                format!("'{}' stream '{source}' ({why})", u.member),
            other => format!("'{}' ({other:?})", u.member),
        }).collect();
    (!skipped.is_empty()).then(|| format!(
        "not checked for a shared observation set: {}. \
         A difference against such a row may be a data difference, not a model \
         difference.",
        skipped.join("; "),
    ))
}

/// The rows whose data the shared-data check compared on an 8-character
/// prefix, or `None`. Every renderer states it: a difference beyond the
/// prefix is not detected.
fn data_prefix_note(checks: &Preflight, rows: &[Row]) -> Option<String> {
    let qualified: Vec<Qualified> = checks.result(CheckId::SharedData)
        .map(|r| r.qualified.clone())
        .unwrap_or_default();
    let mut names: Vec<String> = in_row_order(&qualified, |q| &q.member, rows).into_iter()
        .map(|q| q.member.clone()).collect();
    names.dedup();
    (!names.is_empty()).then(|| format!(
        "data compared on 8 characters of the SHA-256 for {}: its fit.meta.json was \
         written before full digests were recorded, so the shared-data check \
         for that row compares the first 8 characters only.",
        quoted(&names)))
}

/// The rows whose trace records no filter settings, so the settings check
/// could not cover them, or `None`.
fn filter_settings_unchecked_note(checks: &Preflight, rows: &[Row]) -> Option<String> {
    let unrecorded: Vec<String> = not_checked(checks, CheckId::FilterSettings, rows).into_iter()
        .map(|u| match &u.reason {
            NotCheckedReason::TraceBeforeSchema5 => format!(
                "filter settings not recorded for '{}' (trace older than schema 5); not checked",
                u.member),
            NotCheckedReason::NoDerivationBlock => format!(
                "filter settings not recorded for '{}' (its producer wrote no derivation block); \
                 not checked", u.member),
            other => format!("filter settings not recorded for '{}' ({other:?}); not checked",
                u.member),
        }).collect();
    (!unrecorded.is_empty()).then(|| unrecorded.join("; "))
}

// ── the pointwise Δelpd vector (gh#706) ──────────────────────────────────

/// One emitted row: a candidate's log score at one observation, the baseline's
/// at the same observation, and their difference.
///
/// `log_score` and `baseline_log_score` are `Option` because a stream one model
/// scored and the other did not is a real and load-bearing case — gh#570's
/// failure mode, where an elpd difference is quietly taken across two different
/// stream sets. Here it shows as an empty cell and an empty difference instead
/// of being summed into a scalar.
#[derive(Debug, Clone, PartialEq)]
struct PointwiseRow {
    model: String,
    baseline: String,
    t: f64,
    log_score: Option<f64>,
    baseline_log_score: Option<f64>,
}

impl PointwiseRow {
    fn delta(&self) -> Option<f64> {
        Some(self.log_score? - self.baseline_log_score?)
    }
}

/// One stream's own row: that stream's scores at one observation time.
#[derive(Debug, Clone, PartialEq)]
struct StreamPointwiseRow {
    stream: String,
    row: PointwiseRow,
}

/// The pointwise projection of a comparison, in its two files: the joint
/// scores, and each stream's own. A stream's score is not a share of the joint
/// one — the joint is the log of a joint predictive, not a sum of the
/// per-stream logs — so the two never share a column.
#[derive(Debug, Clone, PartialEq, Default)]
struct Pointwise {
    joint: Vec<PointwiseRow>,
    streams: Vec<StreamPointwiseRow>,
}

fn fmt_coverage(c: score::trace::Coverage) -> String {
    use score::trace::Coverage;
    match c {
        Coverage::Unrecorded => "unrecorded".into(),
        Coverage::Instant => "an instant reading".into(),
        Coverage::Interval { start, stop } => format!("the flow over [{start}, {stop})"),
    }
}

/// A row's data identity as a table scored on `leaves` alone checks it: the
/// recorded hashes of those leaves' sources (found through the row's
/// [`StreamBindings::source_of`]), with every other source set aside as not
/// compared, so [`check_shared_data`] skips it and lists it as not checked.
/// Also returns each selected leaf the row does not bind and each selected
/// source for which the row carries digests but no hash at all: an absent
/// hash must not read as a matching one, so the check fails on it.
///
/// A row with no data identity is returned as it is. So is an explicit
/// `prequential.json`, which binds nothing.
fn data_for_streams(r: &Row, leaves: &[String]) -> (DataIdentity, Vec<Unhashed>) {
    let (DataIdentity::Digests(rec), RowStreams::Derived { bindings, .. }) = (&r.data, &r.streams)
    else {
        return (r.data.clone(), Vec::new());
    };
    let mut sources: Vec<&str> = Vec::new();
    let mut missing: Vec<Unhashed> = Vec::new();
    for leaf in leaves {
        match bindings.source_of(leaf) {
            Some(s) => if !sources.contains(&s) { sources.push(s) },
            None => missing.push(Unhashed::StreamNotBound { stream: leaf.clone() }),
        }
    }
    let not_compared = format!("not compared: this table scores {} alone", leaves.join(", "));
    let mut out = RecordedData::default();
    let split = |all: &std::collections::BTreeMap<String, String>,
                 keep: &mut std::collections::BTreeMap<String, String>,
                 set_aside: &mut std::collections::BTreeMap<String, String>| {
        for (source, h) in all {
            if sources.contains(&source.as_str()) {
                keep.insert(source.clone(), h.clone());
            } else {
                set_aside.insert(source.clone(), not_compared.clone());
            }
        }
    };
    split(&rec.training, &mut out.training, &mut out.unresolved);
    for (source, why) in &rec.unresolved {
        if sources.contains(&source.as_str()) {
            out.unresolved.insert(source.clone(), why.clone());
        }
    }
    // Refused by the caller, and set aside here so the pairwise comparison
    // does not report the same absence a second time as a source one row
    // alone bound.
    for s in &sources {
        if !out.training.contains_key(*s) && !out.unresolved.contains_key(*s) {
            missing.push(Unhashed::NoHashForSource { source: s.to_string() });
            out.unresolved.insert(s.to_string(), "the fit recorded no hash for it".into());
        }
    }
    (DataIdentity::Digests(out), missing)
}

/// A content hash abbreviated for a message: enough to tell two apart, short
/// enough to read. Character-wise, so a short test or hand-written digest does
/// not panic on a byte slice.
fn short_digest(hex: &str) -> String {
    hex.chars().take(8).collect()
}

/// The set of stream names scored at one step. Empty for a v1 trace, which has
/// no per-stream breakdown at all.
fn step_streams(s: &score::trace::PrequentialStep)
    -> std::collections::BTreeSet<&str>
{
    s.per_stream.iter().map(|x| x.stream.as_str()).collect()
}

/// Project every candidate against the baseline, step by step and stream by
/// stream.
///
/// The preflight ([`check_pairing`]) has already established
/// that paired steps carry the same `t`; the check is repeated here because
/// this function is also the one that would turn a mismatch into a plot that
/// looks fine, and it must not depend on a caller having run the preflight.
fn pointwise_rows(rows: &[Row], base_idx: usize) -> Result<Pointwise, String> {
    let base = &rows[base_idx];
    let mut out = Pointwise::default();
    for (i, r) in rows.iter().enumerate() {
        if i == base_idx {
            continue;
        }
        if r.trace.n_scored() != base.trace.n_scored() {
            return Err(format!(
                "'{}' scored {} observations and baseline '{}' scored {} — a \
                 pointwise difference needs the same observations on both sides",
                r.name, r.trace.n_scored(), base.name, base.trace.n_scored(),
            ));
        }
        for (step, bstep) in r.trace.steps.iter().zip(&base.trace.steps) {
            if step.t != bstep.t {
                return Err(format!(
                    "'{}' and baseline '{}' disagree on the observation time of a \
                     paired step ({} vs {}) — the two traces are not on the same \
                     observation axis",
                    r.name, base.name, step.t, bstep.t,
                ));
            }
            out.joint.push(PointwiseRow {
                model: r.name.clone(),
                baseline: base.name.clone(),
                t: step.t,
                log_score: Some(step.log_score),
                baseline_log_score: Some(bstep.log_score),
            });
            // The union of both sides' streams, in the candidate's order then
            // any the baseline alone scored — so a stream missing from either
            // side is visible rather than dropped.
            let mut names: Vec<&str> =
                step.per_stream.iter().map(|s| s.stream.as_str()).collect();
            for s in &bstep.per_stream {
                if !names.contains(&s.stream.as_str()) {
                    names.push(s.stream.as_str());
                }
            }
            let find = |ss: &[score::trace::StreamScore], n: &str| {
                ss.iter().find(|s| s.stream == n).map(|s| s.log_score)
            };
            for n in names {
                out.streams.push(StreamPointwiseRow {
                    stream: n.to_string(),
                    row: PointwiseRow {
                        model: r.name.clone(),
                        baseline: base.name.clone(),
                        t: step.t,
                        log_score: find(&step.per_stream, n),
                        baseline_log_score: find(&bstep.per_stream, n),
                    },
                });
            }
        }
    }
    Ok(out)
}

/// A score cell. An absent score is an EMPTY cell, never `NaN` or `0` — the
/// reader must be able to tell "this model did not score here" from "this
/// model scored zero here".
fn pointwise_cell(v: Option<f64>) -> String {
    v.map(|x| format!("{:.10}", x)).unwrap_or_default()
}

/// The joint file: one row per candidate × scored step. `streams_name` is the
/// sibling file the header points at.
fn render_pointwise_joint_tsv(rows: &[PointwiseRow], streams_name: &str) -> String {
    let mut s = String::new();
    s.push_str("# camdl compare --pointwise: per-observation joint log predictive scores\n");
    s.push_str("# log_score is the log of the joint one-step-ahead predictive density of \
                every stream observed at t; summed over t it is the row's elpd.\n");
    s.push_str("# delta_log_score = log_score - baseline_log_score, in nats.\n");
    s.push_str(&format!("# Each stream's own score is in {streams_name}.\n"));
    s.push_str("model\tbaseline\tt\tlog_score\tbaseline_log_score\tdelta_log_score\n");
    for r in rows {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            r.model, r.baseline, fmt_pointwise_time(r.t),
            pointwise_cell(r.log_score), pointwise_cell(r.baseline_log_score),
            pointwise_cell(r.delta()),
        ));
    }
    s
}

/// What the per-stream pointwise file sits beside, which its header states.
enum StreamsFile<'a> {
    /// The joint file of the same comparison, by name.
    BesideJoint(&'a str),
    /// No joint file: `--streams` scored each of these streams alone.
    Selected(&'a [String]),
}

/// The per-stream file: one row per candidate × scored step × stream either
/// side scored there.
fn render_pointwise_streams_tsv(rows: &[StreamPointwiseRow], about: StreamsFile) -> String {
    let mut s = String::new();
    match about {
        StreamsFile::BesideJoint(joint_name) => {
            s.push_str("# camdl compare --pointwise: per-observation log predictive scores, \
                        one stream at a time\n");
            s.push_str("# log_score is the stream's own one-step-ahead log predictive density. \
                        These rows do not sum to the joint scores in ");
            s.push_str(joint_name);
            s.push_str(": a joint score is the log of a joint predictive, not a sum of \
                        per-stream logs.\n");
        }
        StreamsFile::Selected(leaves) => {
            s.push_str(&format!("# camdl compare --pointwise --streams {}: per-observation \
                log predictive scores on the selected streams only, one stream at a time\n",
                leaves.join(",")));
            s.push_str("# log_score is the stream's own one-step-ahead log predictive density, \
                        each row re-scored on that stream alone. No joint file is written: \
                        a sum of these rows over streams is not a joint score.\n");
        }
    }
    s.push_str("# delta_log_score = log_score - baseline_log_score, in nats.\n");
    s.push_str("# An empty cell means that side did not score that stream at \
                that time.\n");
    s.push_str("model\tbaseline\tt\tstream\tlog_score\tbaseline_log_score\tdelta_log_score\n");
    for StreamPointwiseRow { stream, row: r } in rows {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.model, r.baseline, fmt_pointwise_time(r.t), stream,
            pointwise_cell(r.log_score), pointwise_cell(r.baseline_log_score),
            pointwise_cell(r.delta()),
        ));
    }
    s
}

/// Observation times join the rows back to the data, so they are written the
/// way the observation axis reads: `14`, not `14.0000000000`.
fn fmt_pointwise_time(t: f64) -> String {
    if t.fract() == 0.0 && t.abs() < 1e15 {
        format!("{}", t as i64)
    } else {
        format!("{}", t)
    }
}

/// What `--pointwise` wrote: each file and its row count.
struct PointwiseWritten {
    joint: (PathBuf, usize),
    streams: (PathBuf, usize),
}

/// Write the joint rows to `path` and the per-stream rows to its
/// [`streams_sibling`]. Both are rendered before either is written, so a
/// refusal leaves neither file behind.
fn write_pointwise(rows: &[Row], base_idx: usize, path: &Path) -> Result<PointwiseWritten, String> {
    let pw = pointwise_rows(rows, base_idx)?;
    let streams_path = crate::prequential_out::streams_sibling(path);
    let file_name = |p: &Path| p.file_name().map_or_else(
        || p.display().to_string(), |n| n.to_string_lossy().into_owned());
    let joint_text = render_pointwise_joint_tsv(&pw.joint, &file_name(&streams_path));
    let streams_text = render_pointwise_streams_tsv(&pw.streams,
        StreamsFile::BesideJoint(&file_name(path)));
    for (p, text) in [(path, &joint_text), (streams_path.as_path(), &streams_text)] {
        std::fs::write(p, text).map_err(|e| format!("cannot write {}: {e}", p.display()))?;
    }
    Ok(PointwiseWritten {
        joint: (path.to_path_buf(), pw.joint.len()),
        streams: (streams_path, pw.streams.len()),
    })
}

/// Under `--streams`: write each table's per-stream rows, table by table, to
/// the [`streams_sibling`] of `path`, and no joint file. Every table's rows are
/// rendered before the file is written, so a refusal leaves no file behind.
fn write_pointwise_streams(tables: &[Table], leaves: &[String], path: &Path)
    -> Result<(PathBuf, usize), String>
{
    let mut rows: Vec<StreamPointwiseRow> = Vec::new();
    for t in tables {
        rows.extend(pointwise_rows(&t.rows, t.base_idx)?.streams);
    }
    let streams_path = crate::prequential_out::streams_sibling(path);
    std::fs::write(&streams_path, render_pointwise_streams_tsv(&rows, StreamsFile::Selected(leaves)))
        .map_err(|e| format!("cannot write {}: {e}", streams_path.display()))?;
    Ok((streams_path, rows.len()))
}

/// One group of rows of one compared input: what [`Row`] carries besides its
/// name.
struct Loaded {
    trace: PrequentialTrace,
    data: DataIdentity,
    streams: RowStreams,
    facts: ScoreFacts,
}

/// One comparison input, resolved.
enum Input {
    /// A `prequential.json` (a `.json` file, or a directory holding
    /// `prequential.json`), read as it is, as one group.
    Explicit(Vec<Loaded>),
    /// A stored score, prediction or fit run, read through [`stored::load`].
    Stored(stored::Prepared),
}

fn resolve_input(path: &str, request: &stored::Request) -> Result<Input, String> {
    if let Some(trace) = try_load_explicit_prequential(path)? {
        // A trace records scores and times, never which observations produced
        // them — so this row cannot take part in the data-identity check, and
        // re-reads no data file.
        let facts = ScoreFacts::of_trace(&trace);
        return Ok(Input::Explicit(vec![Loaded {
            trace,
            data: DataIdentity::Unknown("read as an explicit prequential.json, which records no bound data".into()),
            streams: RowStreams::Explicit,
            facts,
        }]));
    }
    stored::prepare(path, request).map(Input::Stored)
}

/// A row's filter-noise Monte-Carlo SE of its elpd, as its trace records it
/// (schema 5, `derivation.mc_se_elpd`); `None` without replication or for a
/// trace that records no derivation. An explicit `prequential.json` that
/// carries the block takes part in the filter-noise gate like a derived row.
fn row_mc_se(row: &Row) -> Option<f64> {
    row.trace.derivation.as_ref().and_then(|d| d.mc_se_elpd)
}

/// The pair's combined filter-noise SE for a Δ against the baseline,
/// `√(mc_a² + mc_b²)` (§3.4 remedy 4); `None` unless both rows carry a
/// replicate-derived MC SE. Conservative under common random numbers.
fn mc_se_delta(row: &Row, base: &Row) -> Option<f64> {
    match (row_mc_se(row), row_mc_se(base)) {
        (Some(a), Some(b)) => Some((a * a + b * b).sqrt()),
        _ => None,
    }
}

/// The note naming each non-baseline row whose evidence label was not checked
/// against filter noise because only one side of its pair with the baseline
/// carries a filter-noise MC SE, or `None`.
///
/// [`mc_se_delta`] needs both sides; with one, the "within filter noise" gate
/// is skipped and the label reads as if it had passed. Neither side having
/// one is the documented cheap mode (no replication requested) and says
/// nothing new; both having one is the checked case.
fn filter_noise_unassessed_note(rows: &[Row], base_idx: usize) -> Option<String> {
    let base = &rows[base_idx];
    let one_sided: Vec<String> = rows.iter().enumerate()
        .filter(|(i, r)| *i != base_idx && row_mc_se(r).is_some() != row_mc_se(base).is_some())
        .map(|(_, r)| {
            let (with, without) = if row_mc_se(r).is_some() { (&r.name, &base.name) }
                                  else { (&base.name, &r.name) };
            format!("'{}' (filter-noise SE for '{with}' only; '{without}' has none)", r.name)
        })
        .collect();
    (!one_sided.is_empty()).then(|| format!(
        "evidence label not checked against filter noise for {}: the \
         within-filter-noise gate needs a replicate-derived MC SE on both rows \
         of the pair. Re-derive both rows with --replicates ≥ 2 to check it.",
        one_sided.join("; "),
    ))
}

/// Try to read `path` as an explicit prequential artifact. Returns `Ok(None)`
/// (the input is then a stored result) when it is not one: a non-existent
/// path, a `@label` / hash prefix, a `.toml`, or a directory with no
/// `prequential.json`. Only a `.json` file or a `<dir>/prequential.json` is
/// read — so a `fit.toml` handle is never mis-parsed as a trace.
#[expect(clippy::disallowed_methods, reason = "reads a prequential.json the user names as a row, which compare reads as it is and stores nothing from")]
fn try_load_explicit_prequential(path: &str) -> Result<Option<PrequentialTrace>, String> {
    let p = Path::new(path);
    let json_path = if p.is_dir() {
        let jp = p.join("prequential.json");
        if !jp.exists() {
            return Ok(None);
        }
        jp
    } else if p.extension().map(|e| e.eq_ignore_ascii_case("json")).unwrap_or(false) {
        if !p.exists() {
            return Ok(None);
        }
        p.to_path_buf()
    } else {
        return Ok(None);
    };
    let text = std::fs::read_to_string(&json_path)
        .map_err(|e| format!("{}: {}", json_path.display(), e))?;
    let trace = serde_json::from_str::<PrequentialTrace>(&text)
        .map_err(|e| format!("parsing {}: {}", json_path.display(), e))?;
    // Every producer this build has writes the current schema; a trace below
    // it was written by an older build and cannot say whether a forcing was
    // held outside its range while it was scored.
    if trace.schema_version < score::trace::SCHEMA_RECORDING_HELD_FORCINGS {
        return Err(format!(
            "{}: this prequential trace has schema_version {}, written before held forcings \
             were recorded (schema {}), so whether a forcing was held outside its range while \
             it was scored is not known.\n  Re-run the fit that wrote it (`camdl fit run \
             <config>`), or, for a trace file, the `camdl pfilter --save-prequential` that \
             wrote it.",
            json_path.display(), trace.schema_version,
            score::trace::SCHEMA_RECORDING_HELD_FORCINGS));
    }
    Ok(Some(trace))
}

/// A row's line for a forcing held while its trace was scored
/// ([`PrequentialWarning::ForcingHeld`]), in the words of the run's own warning
/// (`OutsideRange`'s `Display`); `None` for any other warning.
fn held_forcing_line(row: &str, w: &PrequentialWarning) -> Option<String> {
    let PrequentialWarning::ForcingHeld { forcing, defined_on: [lo, hi], side, held_value_at, bound, lagged } = w else {
        return None;
    };
    Some(format!(
        "⚠ {row}: scored with forcing `{forcing}` held outside its range: it is defined on \
         t = {lo} to t = {hi}{}, and was evaluated {}, where its value at t = {held_value_at} \
         was held (allow_forcing_extrapolation){}",
        match bound {
            Some(score::trace::ForcingBound::Data) => " (its data)",
            Some(score::trace::ForcingBound::Window) => " (its window)",
            None => "",
        },
        match side {
            score::trace::ForcingSide::Before => "before it",
            score::trace::ForcingSide::After => "after it",
        },
        // The clause the run's own warning adds (`OutsideRange::lag_clause`).
        if *lagged == Some(true) { "; the forcing has a lag, so these are times of its argument, t − lag" } else { "" },
    ))
}

/// Chain exclusion for the compared cohort, resolved per fit by name. A fit
/// absent from the map keeps all its chains. Built from repeatable
/// `--exclude-chains` tokens: `@fit:ids` targets one fit; bare `ids` apply
/// cohort-wide (every fit). `cohort_wide` only tunes the warning wording.
#[derive(Debug, Default)]
struct CohortChainSelection {
    per_fit: std::collections::BTreeMap<String, ChainSelection>,
    cohort_wide: bool,
}

impl CohortChainSelection {
    /// This fit's drop set, or `None` (keep all its chains).
    fn for_fit(&self, name: &str) -> Option<&ChainSelection> {
        self.per_fit.get(name)
    }

    /// The loud, non-quietable bias warning, once, before derivations run.
    /// Cohort-wide reuses the shared `each fit` phrasing; the per-fit form names
    /// each targeted fit and states the bias caveat a single time.
    fn warn(&self, out: &crate::out::Out) {
        if self.per_fit.is_empty() {
            return;
        }
        if self.cohort_wide {
            // Every entry carries the same drop set — warn once.
            if let Some(sel) = self.per_fit.values().next() {
                sel.warn_requested(out, );
            }
        } else {
            for (fit, sel) in &self.per_fit {
                sel.warn_requested_for_fit(out, fit);
            }
            crate::chain_selection::eprint_bias_caveat(out, );
        }
    }
}

/// Parse the repeatable `--chains` (keep) and `--exclude-chains` (drop)
/// tokens, bound to the resolved model `names`. Each token is `@fit:ids` /
/// `fit:ids` (one fit) or bare `ids` (cohort-wide, applied to every fit).
///
/// Mixing the two forms is rejected as ambiguous, across both flags: a bare
/// value in either is refused when either holds a per-fit value. The two flags
/// may name different fits; one fit named in both, or both flags given
/// cohort-wide (which names every fit in both), is refused. A named fit must
/// match EXACTLY ONE model — an unknown or ambiguous name is a hard error, so a
/// per-fit token can never silently bind nowhere or to the wrong fit.
fn parse_cohort_selection(
    keep: &[String],
    drop: &[String],
    names: &[String],
) -> Result<CohortChainSelection, String> {
    use crate::chain_selection::Polarity;
    let all = || keep.iter().chain(drop.iter());
    if all().next().is_none() {
        return Ok(CohortChainSelection::default());
    }
    let has_keyed = all().any(|t| t.contains(':'));
    let has_bare = all().any(|t| !t.contains(':'));
    if has_keyed && has_bare {
        return Err(
            "--chains / --exclude-chains: mixing cohort-wide ids and per-fit `@fit:ids` is \
             ambiguous; give either bare ids (applied to every fit) or @fit:ids (per fit), \
             not both"
                .to_string(),
        );
    }

    let mut per_fit: std::collections::BTreeMap<String, ChainSelection> =
        std::collections::BTreeMap::new();

    if has_bare {
        if !keep.is_empty() && !drop.is_empty() {
            return Err(
                "--chains and --exclude-chains are both cohort-wide, so both name every fit; \
                 give one selection per fit"
                    .to_string(),
            );
        }
        // Cohort-wide: union the bare tokens into one selection, applied to all.
        let (tokens, polarity) =
            if keep.is_empty() { (drop, Polarity::Drop) } else { (keep, Polarity::Keep) };
        let sel = ChainSelection::parse(&tokens.join(","), polarity, polarity.read_flag())?;
        for name in names {
            per_fit.insert(name.clone(), sel.clone());
        }
        return Ok(CohortChainSelection { per_fit, cohort_wide: true });
    }

    // Per-fit: `fit:ids`. Split on the LAST ':' so a fit name may contain one.
    // The name is matched VERBATIM against the compared fits' names — a
    // handle-referenced fit is named `@a`, a path-referenced one `ctl_rm.toml`.
    // The `@` is part of a handle name, NOT a per-fit sigil to prepend.
    for (tokens, polarity) in [(keep, Polarity::Keep), (drop, Polarity::Drop)] {
        let flag = polarity.read_flag();
        for tok in tokens {
            let (fit, ids) = tok
                .rsplit_once(':')
                .expect("every token has ':' in the per-fit branch");
            if fit.is_empty() {
                return Err(format!("{flag} '{tok}': missing fit name before ':'"));
            }
            let n_match = names.iter().filter(|n| n.as_str() == fit).count();
            if n_match == 0 {
                // Common slip: the `--help` example uses a handle name `@a`, so a
                // user prepends a spurious `@` to a path-named fit. If dropping it
                // would match, say so rather than a bare "no such fit".
                let hint = fit
                    .strip_prefix('@')
                    .filter(|stripped| names.iter().any(|n| n == stripped))
                    .map(|stripped| {
                        format!(
                            " — did you mean '{stripped}'? the leading `@` names a run-store \
                             handle; a fit given by path is named without it"
                        )
                    })
                    .unwrap_or_default();
                return Err(format!(
                    "{flag}: no compared fit named '{fit}' (fits: {}){hint}",
                    names.join(", ")
                ));
            }
            if n_match > 1 {
                return Err(format!(
                    "{flag}: fit name '{fit}' is ambiguous ({n_match} compared fits \
                     share it) — give each `[[model]]` a unique name via --config"
                ));
            }
            let sel = ChainSelection::parse(ids, polarity, &format!("{flag} {fit}"))?;
            if let Some(prior) = per_fit.insert(fit.to_string(), sel) {
                return Err(if prior.polarity() == polarity {
                    format!("{flag}: fit '{fit}' listed more than once")
                } else {
                    format!(
                        "fit '{fit}' is named by both --chains and --exclude-chains; give \
                         one selection per fit"
                    )
                });
            }
        }
    }
    Ok(CohortChainSelection { per_fit, cohort_wide: false })
}

/// The reason a row has no CRPS or calibration figure when its trace scored
/// steps but carries no per-stream breakdown.
const NO_STREAM_BREAKDOWN: &str = "trace predates the per-stream breakdown";

/// One stream's accuracy and calibration in one row.
#[derive(Debug, Clone, PartialEq)]
struct StreamAccuracy {
    stream: String,
    /// The mean CRPS over the steps that scored the stream; `n` is the
    /// number of scored observations.
    mean_crps: StreamMean,
    /// The mean per-observation CRPS difference against the baseline, or
    /// `None` for the baseline itself and for a stream the two rows did not
    /// score at the same times (a mean over different observations is not a
    /// paired difference).
    delta_mean_crps: Option<f64>,
    /// The share of the stream's PIT values inside the central 90%.
    pit_cov90: StreamMean,
    /// The mean weighted interval score, and its paired difference against
    /// the baseline as for `delta_mean_crps`; `None` for a trace that records
    /// no WIS (a prequential.json).
    mean_wis: Option<StreamMean>,
    delta_mean_wis: Option<f64>,
    /// The share of the stream's observations inside the central 50%
    /// interval; `None` for a trace that records no interval membership.
    cov50: Option<StreamMean>,
}

/// The accuracy columns a rendering shows: those the metrics ask for, and,
/// of WIS and the 50% coverage, only those some row records.
#[derive(Debug, Clone, Copy)]
struct Want {
    crps: bool,
    wis: bool,
    pit: bool,
    cov50: bool,
}

impl Want {
    fn of(metrics: &[String], accuracy: &[Accuracy]) -> Want {
        let asked = |names: &[&str]| metrics.iter().any(|m| names.contains(&m.as_str()));
        let recorded = |f: fn(&StreamAccuracy) -> bool| accuracy.iter().any(|a| match a {
            Accuracy::PerStream(v) => v.iter().any(f),
            Accuracy::Unavailable(_) => false,
        });
        Want {
            crps: asked(&["crps"]),
            wis: asked(&["wis"]) && recorded(|s| s.mean_wis.is_some()),
            pit: asked(&["pit_cov90", "pit"]),
            cov50: asked(&["cov50"]) && recorded(|s| s.cov50.is_some()),
        }
    }

    fn any(self) -> bool {
        self.crps || self.wis || self.pit || self.cov50
    }
}

/// What a row can say about accuracy and calibration: per stream, or nothing
/// and why. There is no combined figure: a CRPS is in the units of one
/// observed quantity, and the streams together are not one.
#[derive(Debug, Clone, PartialEq)]
enum Accuracy {
    Unavailable(&'static str),
    PerStream(Vec<StreamAccuracy>),
}

/// `row`'s per-stream accuracy, with each Δ formed against `base`.
fn row_accuracy(row: &Row, base: &Row) -> Accuracy {
    let streams = row.trace.streams();
    if streams.is_empty() {
        return Accuracy::Unavailable(if row.trace.steps.is_empty() {
            "no scored steps"
        } else {
            NO_STREAM_BREAKDOWN
        });
    }
    let is_base = std::ptr::eq(row, base);
    // The mean paired difference of one per-observation figure against the
    // baseline, over the observations both scored at the same times.
    let delta = |stream: &str, of: fn(&score::trace::StreamScore) -> Option<f64>| -> Option<f64> {
        if is_base {
            return None;
        }
        let a: Vec<(f64, Option<f64>)> = row.trace.stream_scores(stream).map(|(t, s)| (t, of(s))).collect();
        let b: Vec<(f64, Option<f64>)> = base.trace.stream_scores(stream).map(|(t, s)| (t, of(s))).collect();
        let same_times = a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.0 == y.0);
        let diffs: Vec<f64> = a.iter().zip(&b)
            .filter_map(|(x, y)| Some(x.1? - y.1?))
            .filter(|d| d.is_finite())
            .collect();
        (same_times && !diffs.is_empty()).then(|| diffs.iter().sum::<f64>() / diffs.len() as f64)
    };
    Accuracy::PerStream(streams.into_iter().map(|stream| {
        StreamAccuracy {
            stream: stream.to_string(),
            mean_crps: row.trace.mean_crps(stream),
            delta_mean_crps: delta(stream, |s| Some(s.crps)),
            pit_cov90: row.trace.pit_coverage(stream, 0.90),
            mean_wis: row.trace.mean_wis(stream),
            delta_mean_wis: delta(stream, |s| s.wis),
            cov50: row.trace.coverage_50(stream),
        }
    }).collect())
}

/// The one stream every row scored, when each scored exactly one and it is
/// the same: the case in which the table keeps its CRPS and PIT columns.
fn shared_single_stream(accuracy: &[Accuracy]) -> Option<&str> {
    let mut only: Option<&str> = None;
    for a in accuracy {
        match a {
            Accuracy::PerStream(v) if v.len() == 1 => match only {
                None => only = Some(&v[0].stream),
                Some(s) if s == v[0].stream => {}
                Some(_) => return None,
            },
            _ => return None,
        }
    }
    only
}

/// The comparison's per-stream accuracy, row by row (in input order), and the
/// shared single stream if there is one.
fn accuracy_view(rows: &[Row], base_idx: usize) -> (Vec<Accuracy>, Option<String>) {
    let acc: Vec<Accuracy> = rows.iter().map(|r| row_accuracy(r, &rows[base_idx])).collect();
    let single = shared_single_stream(&acc).map(str::to_string);
    (acc, single)
}

/// The streams the rows scored, in the order they first appear.
fn accuracy_streams(accuracy: &[Accuracy]) -> Vec<&str> {
    let mut out: Vec<&str> = Vec::new();
    for a in accuracy {
        if let Accuracy::PerStream(v) = a {
            for s in v {
                if !out.contains(&s.stream.as_str()) { out.push(&s.stream); }
            }
        }
    }
    out
}

/// Render order: ascending Δelpd (worst → best), the baseline at its natural
/// Δelpd = 0 slot, so the best-supported row lands at the bottom where the
/// reader's eye ends. Under mismatched horizons no Δ exists to sort by, so the
/// input order stands.
fn render_order(rows: &[Row], base_idx: usize, t_mismatch: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    if t_mismatch {
        return idx;
    }
    let base = &rows[base_idx].trace;
    idx.sort_by(|&a, &b| {
        let da = if a == base_idx { 0.0 } else { paired_delta(&rows[a].trace, base).0 };
        let db = if b == base_idx { 0.0 } else { paired_delta(&rows[b].trace, base).0 };
        // NaN delta → place row at the top (worst-case ranking).
        match (da.is_nan(), db.is_nan()) {
            (true, true)  => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal),
        }
    });
    idx
}

/// The cells of one stream × row line, in the per-stream block's column
/// order (`n`, then the wanted metrics), or the shared columns of the
/// single-stream table (`want_n = false`).
fn accuracy_cells(sa: &StreamAccuracy, want_n: bool, want: Want, delta_shown: bool) -> Vec<String> {
    let mut cells = Vec::new();
    let delta = |d: Option<f64>| match (delta_shown, d) {
        (true, Some(d)) => format!("{:+.3}", d),
        _ => "—".into(),
    };
    if want_n { cells.push(sa.mean_crps.n.to_string()); }
    if want.crps {
        cells.push(format!("{:.3}", sa.mean_crps.value));
        cells.push(delta(sa.delta_mean_crps));
    }
    if want.wis {
        cells.push(sa.mean_wis.map_or("—".into(), |m| format!("{:.3}", m.value)));
        cells.push(delta(sa.delta_mean_wis));
    }
    if want.pit { cells.push(format!("{:.2}", sa.pit_cov90.value)); }
    if want.cov50 { cells.push(sa.cov50.map_or("—".into(), |m| format!("{:.2}", m.value))); }
    cells
}

/// The accuracy columns of the single-stream table's header.
fn accuracy_columns(want: Want) -> Vec<String> {
    let mut h: Vec<String> = Vec::new();
    if want.crps { h.push("crps".into()); h.push("Δcrps".into()); }
    if want.wis { h.push("wis".into()); h.push("Δwis".into()); }
    if want.pit { h.push("PIT_cov90".into()); }
    if want.cov50 { h.push("cov50".into()); }
    h
}

/// The per-stream block's header, with the wanted metric columns.
fn accuracy_header(want: Want) -> Vec<String> {
    let mut h: Vec<String> = vec!["stream".into(), "model".into(), "n".into()];
    if want.crps { h.push("mean CRPS".into()); h.push("Δ mean CRPS".into()); }
    if want.wis { h.push("mean WIS".into()); h.push("Δ mean WIS".into()); }
    if want.pit { h.push("PIT_cov90".into()); }
    if want.cov50 { h.push("cov50".into()); }
    h
}

/// The per-stream block's body: one line per stream × row, rows in render
/// order, `None` when there is nothing to show (no wanted metric, or no row
/// with a per-stream breakdown).
fn accuracy_block(rows: &[Row], order: &[usize], accuracy: &[Accuracy],
                  want: Want, t_mismatch: bool) -> Option<Vec<Vec<String>>> {
    if !want.any() { return None; }
    let streams = accuracy_streams(accuracy);
    if streams.is_empty() { return None; }
    let mut body = Vec::new();
    for stream in streams {
        for &i in order {
            let Accuracy::PerStream(v) = &accuracy[i] else { continue };
            let Some(sa) = v.iter().find(|s| s.stream == stream) else { continue };
            let mut line = vec![stream.to_string(), rows[i].name.clone()];
            line.extend(accuracy_cells(sa, true, want, !t_mismatch));
            body.push(line);
        }
    }
    Some(body)
}

/// The columns after `Model` that say what a table's rows are of: `role`
/// when its rows were read from stored scores, and `lead` when they are a
/// forecast's. Every row of one table shares them (the `Role` and `Lead`
/// checks).
fn fact_columns(rows: &[Row]) -> Vec<String> {
    let mut h = Vec::new();
    if rows.iter().any(|r| r.facts.stored) { h.push("role".to_string()); }
    if rows.iter().any(|r| r.facts.lead.is_some()) { h.push("lead".to_string()); }
    h
}

fn fact_cells(rows: &[Row], r: &Row) -> Vec<String> {
    let mut c = Vec::new();
    if rows.iter().any(|r| r.facts.stored) { c.push(r.facts.role.clone()); }
    if rows.iter().any(|r| r.facts.lead.is_some()) { c.push(r.facts.lead.clone().unwrap_or_else(|| "—".into())); }
    c
}

/// The `⚠` line for one stream of a row whose 90% predictive interval covers
/// far less than nominal — the plug-in overconfidence tell. `None` when
/// coverage is fine or not computable. Evaluated per stream, so one stream's
/// coverage cannot mask another's.
fn pit_coverage_warning(row: &str, sa: &StreamAccuracy) -> Option<String> {
    let cov = sa.pit_cov90.value;
    (cov < 0.70).then(|| format!(
        "'{row}' stream '{}': PIT 90%-coverage {:.2} (nominal 0.90) — likely overconfident.",
        sa.stream, cov))
}

/// The note for a row stream whose CRPS or PIT values include non-finite
/// ones, which the per-stream figures leave out.
fn nonfinite_note(row: &str, sa: &StreamAccuracy) -> Option<String> {
    let (c, p) = (sa.mean_crps.n_nonfinite, sa.pit_cov90.n_nonfinite);
    (c + p > 0).then(|| format!(
        "'{row}' stream '{}': {c} non-finite CRPS and {p} non-finite PIT value(s) of {} \
         left out of its per-stream figures.", sa.stream, sa.mean_crps.n))
}

/// The evidence cell of `row` against `base`, as [`score::evidence::evidence_cell`]
/// gives it, and for two rows read from stored filtered scores of which one
/// carries no filter-noise MC SE (`--replicates 1`), a verdict outside the
/// se(Δ) band says the filter noise was not measured: the within-filter-noise
/// gate did not run, so the tier is not checked against it.
fn evidence_text(row: &Row, base: &Row, d: f64, se: f64) -> String {
    let mc = mc_se_delta(row, base);
    let cell = score::evidence::evidence_cell(d, se, mc);
    let unmeasured = mc.is_none()
        && row.facts.stored && base.facts.stored
        && row.trace.basis == PredictiveBasis::Filtered
        && score::evidence::evidence_label(d, se, mc) != "within_noise";
    match unmeasured {
        true => format!("{cell}, filter noise not measured"),
        false => cell,
    }
}

/// Format the likelihood ratio `exp(Δelpd)` for display: the candidate's
/// in-sample plug-in predictive likelihood over the baseline's, on the same
/// scored observations. `LR = 1` means "tied with the baseline"; `LR = 100`
/// means the candidate assigned the observed series 100× the predictive
/// likelihood the baseline did; `LR = 0.01`, one hundredth of it.
///
/// The ratio is computed at a θ̂ fit to the very observations it scores, so it
/// is optimistic in level and biased toward the more flexible model — read it
/// with `se(Δ)` and the caveats under the table, not on its own.
///
/// The value ranges over many orders of magnitude: compact decimal inside the
/// readable band [0.001, 1000], scientific notation outside it.
fn fmt_lr(lr: f64) -> String {
    if !lr.is_finite() { return "—".into(); }
    if lr == 0.0 { return "0".into(); }
    if !(0.001..1000.0).contains(&lr) {
        format!("{:.2e}", lr)
    } else {
        format!("{:.3}", lr)
    }
}

/// Render the comparison as an aligned text table. Returns the whole rendering
/// (including its trailing newline) rather than printing, so every line the
/// reader sees — footers, caveats, warnings — is assertable in a unit test.
fn render_table(rows: &[Row], base_idx: usize, metrics: &[String], checks: &Preflight) -> String {
    let t_mismatch = checks.horizons_differ();
    let (accuracy, single) = accuracy_view(rows, base_idx);
    let want = Want::of(metrics, &accuracy);

    // Δelpd (nats) is the primary machine-readable column; "evidence"
    // (decibans + Jeffreys label) is the human-interpretable alongside —
    // see docs/dev/proposals/2026-04-23-evidence-in-decibans.md §Scope.
    // CRPS and PIT columns only when every row scored the same one stream;
    // otherwise they are per stream, in the block below the table.
    let mut header = vec!["Model".to_string()];
    header.extend(fact_columns(rows));
    header.extend(["T_score", "elpd", "Δelpd", "LR", "se(Δ)", "evidence"].map(String::from));
    if single.is_some() {
        header.extend(accuracy_columns(want));
    }

    let base = &rows[base_idx].trace;
    let order = render_order(rows, base_idx, t_mismatch);
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for &i in &order {
        let r = &rows[i];
        let elpd = r.trace.elpd();
        let mut row = vec![r.name.clone()];
        row.extend(fact_cells(rows, r));
        row.extend([format!("{}", r.trace.n_scored()), format!("{:.2}", elpd)]);
        if i == base_idx || t_mismatch {
            row.push("—".into());   // Δelpd
            row.push("—".into());   // LR
            row.push("—".into());   // se(Δ)
            row.push("—".into());   // evidence (dB + Jeffreys label)
        } else {
            let (d, se) = paired_delta(&r.trace, base);
            row.push(format!("{:+.2}", d));
            row.push(fmt_lr(d.exp()));
            row.push(format!("{:.2}", se));
            row.push(evidence_text(r, &rows[base_idx], d, se));
        }
        if let (Some(_), Accuracy::PerStream(v)) = (&single, &accuracy[i]) {
            row.extend(accuracy_cells(&v[0], false, want, i != base_idx && !t_mismatch));
        }
        body.push(row);
    }

    let mut out = String::new();
    if let Some(leaf) = scored_leaf(rows) {
        out.push_str(&format!("Stream: {leaf} — every row scored on this stream alone.\n\n"));
    }
    push_aligned(&mut out, &header, &body);
    if single.is_none() {
        if let Some(block) = accuracy_block(rows, &order, &accuracy, want, t_mismatch) {
            out.push_str("\nPer-stream accuracy and calibration:\n");
            push_aligned(&mut out, &accuracy_header(want), &block);
        }
    }

    // The footer states what is specific to this comparison — the scored
    // window, the baseline, what θ saw, and the numbers behind the noise
    // gates — and points at the methods page for everything that is the same
    // on every render (gh#806). The static explanations it used to carry are
    // still in the JSON (`evidence_scale_note`, `se_caveat`), which is the
    // machine surface and loses nothing by repeating itself.
    out.push('\n');
    // `t0` indexes each fit's own time axis, which differs between fits that
    // assimilated different streams, so a re-scored table does not print it.
    match scored_leaf(rows) {
        None => out.push_str(&format!("Scored steps: {} (t0={}).  Baseline: {}.\n",
            base.n_scored(), base.t0, rows[base_idx].name)),
        Some(_) => out.push_str(&format!("Scored steps: {}.  Baseline: {}.\n",
            base.n_scored(), rows[base_idx].name)),
    }
    if !t_mismatch {
        out.push_str("Sorted by Δelpd ascending — best-supported model at the bottom.\n");
    }
    out.push_str(&conditioning_line(base));
    out.push('\n');
    if let Some(line) = basis_line(base) {
        out.push_str(line);
        out.push('\n');
    }
    if let Some(line) = assimilated_line(rows) {
        out.push_str(&line);
        out.push('\n');
    }
    for line in warning_lines(rows, base_idx, checks, &accuracy) {
        out.push_str(&line);
        out.push('\n');
    }
    out.push_str(METHODS_POINTER);
    out.push('\n');
    out
}

/// Append an aligned text table — header, rule, body — to `out`.
fn push_aligned(out: &mut String, header: &[String], body: &[Vec<String>]) {
    let widths: Vec<usize> = (0..header.len()).map(|c| {
        let h = header[c].chars().count();
        let b = body.iter().map(|r| r[c].chars().count()).max().unwrap_or(0);
        h.max(b)
    }).collect();
    let total: usize = widths.iter().sum::<usize>() + 3 * (widths.len() - 1);
    out.push_str(&fmt_table_row(header, &widths));
    out.push_str(&format!("{}\n", "─".repeat(total)));
    for row in body {
        out.push_str(&fmt_table_row(row, &widths));
    }
}

/// Everything below the table that the reader must not miss: each override
/// the comparison needed and what the checks could not cover (both from
/// `checks`, never recomputed here), the miscalibration flags, each trace's own
/// warnings, and the optimism caveat.
///
/// One function, called by every renderer, because the failure this guards is
/// asymmetry: the table warned about a miscalibrated model and the markdown
/// someone pasted into a report did not. A new warning added here appears in
/// every format by construction.
fn warning_lines(rows: &[Row], base_idx: usize, checks: &Preflight, accuracy: &[Accuracy])
    -> Vec<String>
{
    // Each override that was needed, in one line: the flag, the rows, what
    // differed and what that does to the table. The terminal session that
    // granted it is not where the table is read a week later.
    let mut lines: Vec<String> = checks.overrides().iter().map(override_line).collect();
    // What the checks could not cover.
    for note in [data_unchecked_note(checks, rows), data_prefix_note(checks, rows),
                 filter_settings_unchecked_note(checks, rows)]
        .into_iter().flatten()
    {
        lines.push(format!("ⓘ {note}"));
    }
    // PIT warnings — flag clear miscalibration, stream by stream — and the
    // values the per-stream figures could not use.
    let mut unavailable: Vec<(&str, &str)> = Vec::new();
    for (r, a) in rows.iter().zip(accuracy) {
        match a {
            Accuracy::PerStream(v) => {
                for sa in v {
                    if let Some(w) = pit_coverage_warning(&r.name, sa) {
                        lines.push(format!("⚠ {w}"));
                    }
                }
                for sa in v {
                    if let Some(n) = nonfinite_note(&r.name, sa) {
                        lines.push(format!("ⓘ {n}"));
                    }
                }
            }
            Accuracy::Unavailable(why) => unavailable.push((&r.name, why)),
        }
    }
    if !unavailable.is_empty() {
        let named: Vec<String> = unavailable.iter()
            .map(|(name, why)| format!("'{name}' ({why})")).collect();
        lines.push(format!("ⓘ CRPS and PIT calibration not available for {}.", named.join(", ")));
    }
    // Propagate trace-level warnings.
    for r in rows {
        for w in &r.trace.warnings {
            lines.push(held_forcing_line(&r.name, w).unwrap_or_else(|| format!("ⓘ {}: {:?}", r.name, w)));
        }
    }
    // §3.4 remedy 4: print each row's filter-noise MC SE so the reader
    // sees the scale the "within filter noise" gate reads. The numbers are
    // this comparison's; what they are and where they come from is the
    // methods page's job (gh#806).
    let with_mc: Vec<String> = rows.iter()
        .filter_map(|r| row_mc_se(r).map(|se| format!("{} ±{:.2}", r.name, se)))
        .collect();
    if !with_mc.is_empty() {
        lines.push(format!(
            "note: filter-noise MC SE of elpd: {}",
            with_mc.join(", ")));
    }
    if let Some(note) = filter_noise_unassessed_note(rows, base_idx) {
        lines.push(format!("ⓘ {note}"));
    }
    // Stage 4.3 (§3.4 remedy 2): the serial dependence the HAC se(Δ)
    // exists for — printed so the reader sees how far from iid the
    // per-step differentials are. Same split: the value here, the
    // Newey–West/HLN construction on the methods page.
    let lag1: Vec<String> = rows.iter().enumerate()
        .filter(|(i, _)| *i != base_idx)
        .filter_map(|(_, r)| {
            let d = paired_diffs(&r.trace, &rows[base_idx].trace)?;
            let a1 = score::evidence::lag1_autocorrelation(&d);
            a1.is_finite().then(|| format!("{} {:+.2}", r.name, a1))
        })
        .collect();
    if !lag1.is_empty() {
        lines.push(format!(
            "note: lag-1 autocorrelation of the per-step Δelpd vs baseline: {}",
            lag1.join(", ")));
    }
    if let Some(caveat) = optimism_caveat(rows) {
        lines.push(format!("note: {caveat}"));
    }
    lines
}

/// The optimism caveat carried by the compared traces (#295) — plug-in and/or
/// in-sample, and what that does to the level and to the differences. Today
/// every trace is both, so this always fires; when posterior / LFO traces
/// exist, the per-trace tag drives it.
fn optimism_caveat(rows: &[Row]) -> Option<String> {
    rows.iter().find_map(|r| r.trace.optimism_caveat())
}

fn fmt_table_row(cells: &[String], widths: &[usize]) -> String {
    let parts: Vec<String> = cells.iter().zip(widths)
        .map(|(c, w)| format!("{:>width$}", c, width = w))
        .collect();
    // Left-align the first column (model name) for readability.
    let mut out = String::new();
    for (i, (c, w)) in cells.iter().zip(widths).enumerate() {
        if i == 0 {
            out.push_str(&format!("{:<width$}", c, width = w));
        } else {
            out.push_str("   ");
            out.push_str(&parts[i]);
        }
    }
    out.push('\n');
    out
}

/// Render the comparison as a markdown table. Returns the rendering rather than
/// printing it, for the same reason [`render_table`] does.
fn render_md(rows: &[Row], base_idx: usize, metrics: &[String], checks: &Preflight) -> String {
    let t_mismatch = checks.horizons_differ();
    let (accuracy, single) = accuracy_view(rows, base_idx);
    let want = Want::of(metrics, &accuracy);
    let md_table = |out: &mut String, header: &[String], body: &[Vec<String>]| {
        out.push_str(&format!("| {} |\n", header.join(" | ")));
        out.push_str(&format!("|{}|\n",
            header.iter().map(|_| "---").collect::<Vec<_>>().join("|")));
        for cells in body {
            out.push_str(&format!("| {} |\n", cells.join(" | ")));
        }
    };

    let mut header: Vec<String> = vec!["Model".to_string()];
    header.extend(fact_columns(rows));
    header.extend(["T_score", "elpd", "Δelpd", "LR", "se(Δ)", "evidence"].map(String::from));
    if single.is_some() {
        header.extend(accuracy_columns(want));
    }

    let base = &rows[base_idx].trace;
    // Same render order as the table renderer.
    let order = render_order(rows, base_idx, t_mismatch);
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for &i in &order {
        let r = &rows[i];
        let mut cells: Vec<String> = vec![r.name.clone()];
        cells.extend(fact_cells(rows, r));
        cells.extend([format!("{}", r.trace.n_scored()), format!("{:.2}", r.trace.elpd())]);
        if i == base_idx || t_mismatch {
            cells.push("—".into());  // Δelpd
            cells.push("—".into());  // LR
            cells.push("—".into());  // se(Δ)
            cells.push("—".into());  // evidence
        } else {
            let (d, se) = paired_delta(&r.trace, base);
            cells.push(format!("{:+.2}", d));
            cells.push(fmt_lr(d.exp()));
            cells.push(format!("{:.2}", se));
            cells.push(evidence_text(r, &rows[base_idx], d, se));
        }
        if let (Some(_), Accuracy::PerStream(v)) = (&single, &accuracy[i]) {
            cells.extend(accuracy_cells(&v[0], false, want, i != base_idx && !t_mismatch));
        }
        body.push(cells);
    }
    let mut out = String::new();
    if let Some(leaf) = scored_leaf(rows) {
        out.push_str(&format!("### Stream `{leaf}`\n\n_Every row scored on this stream alone._\n\n"));
    }
    md_table(&mut out, &header, &body);
    if single.is_none() {
        if let Some(block) = accuracy_block(rows, &order, &accuracy, want, t_mismatch) {
            out.push_str("\n**Per-stream accuracy and calibration**\n\n");
            md_table(&mut out, &accuracy_header(want), &block);
        }
    }
    // Same footer as the text table (gh#806), in markdown: the sort order,
    // the conditioning stamp, the comparison-specific notes, and the pointer
    // at the methods page.
    if !t_mismatch {
        out.push('\n');
        out.push_str("_Sorted by Δelpd ascending — best-supported model at the bottom._\n");
    }
    out.push('\n');
    out.push_str(&format!("_{}_\n", conditioning_line(base)));
    if let Some(line) = basis_line(base) {
        out.push_str(&format!("\n_{line}_\n"));
    }
    if let Some(line) = assimilated_line(rows) {
        out.push_str(&format!("\n_{line}_\n"));
    }
    // The same block the text table prints, as a markdown list so each warning
    // stays its own line when the document is rendered.
    let warnings = warning_lines(rows, base_idx, checks, &accuracy);
    if !warnings.is_empty() {
        out.push('\n');
        for line in warnings {
            out.push_str(&format!("- {line}\n"));
        }
    }
    out.push('\n');
    out.push_str(&format!("_{METHODS_POINTER}_\n"));
    out
}

/// Render the comparison as JSON. Returns the document rather than printing it.
fn render_json(rows: &[Row], base_idx: usize, metrics: &[String], checks: &Preflight) -> String {
    format!("{}\n", serde_json::to_string_pretty(&json_document(rows, base_idx, metrics, checks))
        .unwrap())
}

/// The JSON document of several tables: one [`json_document`] per table, in
/// order, and under `--streams` the selected streams in order, each table
/// naming its stream. Nothing else: no field combines tables.
fn render_json_tables(tables: &[Table], leaves: Option<&[String]>, metrics: &[String]) -> String {
    let mut out = serde_json::json!({
        "tables": tables.iter()
            .map(|t| json_document(&t.rows, t.base_idx, metrics, &t.checks))
            .collect::<Vec<_>>(),
    });
    if let Some(leaves) = leaves {
        out["streams_scored"] = serde_json::json!(leaves);
    }
    format!("{}\n", serde_json::to_string_pretty(&out).unwrap())
}

/// The comparison as a JSON value. A `--streams` table also names the stream
/// it scores (`stream`) and, per row, the leaves its fit assimilated
/// (`streams_assimilated`, null when the input does not record them).
fn json_document(rows: &[Row], base_idx: usize, metrics: &[String], checks: &Preflight)
    -> serde_json::Value
{
    use serde_json::json;
    let scored = scored_leaf(rows);
    let base = &rows[base_idx].trace;
    let t_mismatch = checks.horizons_differ();
    let (accuracy, single) = accuracy_view(rows, base_idx);
    let entries: Vec<serde_json::Value> = rows.iter().enumerate().map(|(i, r)| {
        let (d_elpd, se_elpd) = if i == base_idx { (f64::NAN, f64::NAN) }
            else { paired_delta(&r.trace, base) };
        // Each stream's accuracy and calibration; the Δ is suppressed under
        // mismatched horizons, as the table's is.
        let stream_json = |sa: &StreamAccuracy| {
            let mut j = json!({
                "stream": sa.stream,
                "n": sa.mean_crps.n,
                "mean_crps": option_finite(sa.mean_crps.value),
                "crps_nonfinite": sa.mean_crps.n_nonfinite,
                "delta_mean_crps": sa.delta_mean_crps.filter(|_| !t_mismatch)
                    .map_or(serde_json::Value::Null, option_finite),
                "pit_cov90": option_finite(sa.pit_cov90.value),
                "pit_nonfinite": sa.pit_cov90.n_nonfinite,
                "pit_cov90_warning": pit_coverage_warning(&r.name, sa),
            });
            // A stored score's figures, absent from a prequential.json's row.
            if let Some(w) = sa.mean_wis {
                j["mean_wis"] = option_finite(w.value);
                j["delta_mean_wis"] = sa.delta_mean_wis.filter(|_| !t_mismatch)
                    .map_or(serde_json::Value::Null, option_finite);
            }
            if let Some(c) = sa.cov50 {
                j["cov50"] = option_finite(c.value);
            }
            j
        };
        let (per_stream, unavailable): (Vec<serde_json::Value>, Option<&str>) = match &accuracy[i] {
            Accuracy::PerStream(v) => (v.iter().map(stream_json).collect(), None),
            Accuracy::Unavailable(why) => (Vec::new(), Some(why)),
        };
        // The row-level figures are the single-stream case's, and null
        // otherwise: there is no combined CRPS or PIT across streams.
        let only = match (&single, &accuracy[i]) {
            (Some(_), Accuracy::PerStream(v)) => per_stream.first().cloned()
                .map(|j| (j, &v[0])),
            _ => None,
        };
        let field = |k: &str| only.as_ref().map_or(serde_json::Value::Null, |(j, _)| j[k].clone());
        let lr = if d_elpd.is_finite() { d_elpd.exp() } else { f64::NAN };
        // Evidence: Δelpd (nats) → decibans + a label. Derived field for
        // human-interpretable consumption; nats remain the primary
        // machine-readable quantity (delta_elpd). The label is
        // `within_noise` when |Δ| < 2·se(Δ), else the Jeffreys tier —
        // consumers reading the tier without the SE get the same gate the
        // table shows. See
        // docs/dev/proposals/2026-04-23-evidence-in-decibans.md.
        let (d_elpd_db, evidence_label) = if d_elpd.is_finite() {
            let db = d_elpd * score::evidence::NATS_TO_DB;
            (option_finite(db),
             serde_json::json!(score::evidence::evidence_label(d_elpd, se_elpd,
                 mc_se_delta(r, &rows[base_idx]))))
        } else {
            (serde_json::Value::Null, serde_json::Value::Null)
        };
        let mut entry = json!({
            "name": r.name,
            "path": r.path,
            "t_score": r.trace.n_scored(),
            "elpd": r.trace.elpd(),
            "delta_elpd": option_finite(d_elpd),
            "delta_elpd_db": d_elpd_db,
            "evidence_label": evidence_label,
            "lr": option_finite(lr),
            "se_delta_elpd": option_finite(se_elpd),
            // Filter-noise Monte-Carlo SE from replicate derives (§3.4
            // remedy 4): this row's own, and the pair's combined SE the
            // evidence gate reads. Null without replication.
            "mc_se_elpd": row_mc_se(r).map_or(serde_json::Value::Null,
                option_finite),
            "mc_se_delta_elpd": mc_se_delta(r, &rows[base_idx])
                .map_or(serde_json::Value::Null, option_finite),
            // Stage 4.3: serial dependence of the per-step Δelpd — the
            // reason se_delta_elpd is a Newey–West HAC SE (HLN-corrected,
            // t_{T−1} reference). Null for the baseline row.
            "lag1_autocorr_delta_elpd": if i == base_idx {
                serde_json::Value::Null
            } else {
                paired_diffs(&r.trace, base)
                    .map_or(serde_json::Value::Null, |d| option_finite(
                        score::evidence::lag1_autocorrelation(&d)))
            },
            "mean_crps": field("mean_crps"),
            "delta_mean_crps": field("delta_mean_crps"),
            "pit_cov90": field("pit_cov90"),
            // Accuracy and calibration, one entry per stream the row scored;
            // empty, with the reason, for a trace with no per-stream breakdown.
            "per_stream": per_stream,
            "per_stream_unavailable": unavailable,
            // How this row's score was built and what it saw: the two optimism
            // axes (#295), per row because a cohort may mix them once posterior
            // / LFO traces exist.
            "provenance": r.trace.provenance,
            "conditioning": r.trace.conditioning,
            // Which predictive the scores come from (gh#312): a particle
            // filter's estimate, or the ODE skeleton's exact density.
            "basis": r.trace.basis,
            // How the trace was computed (schema 5); null for an older trace.
            "derivation": r.trace.derivation,
            // The trace's own warnings and the miscalibration flag, verbatim
            // from the same source the table prints — a script reading the JSON
            // must not have to re-derive them from `pit_cov90`.
            "warnings": r.trace.warnings,
            "pit_cov90_warning": field("pit_cov90_warning"),
        });
        if scored.is_some() {
            entry["streams_assimilated"] = json!(r.assimilated());
        }
        // What a stored score's rows are of; a prequential.json's row says it
        // through `basis` and `conditioning`.
        if r.facts.stored {
            entry["mode"] = json!(r.facts.mode);
            entry["role"] = json!(r.facts.role);
            entry["lead"] = json!(r.facts.lead);
            entry["reach"] = json!(r.facts.reach);
            entry["covariates_realised"] = json!(r.facts.realised);
            // Whether the evidence label was checked against filter noise:
            // both rows of the pair carry an MC SE (`--replicates` ≥ 2).
            entry["filter_noise_measured"] = match i == base_idx {
                true => serde_json::Value::Null,
                false => json!(mc_se_delta(r, &rows[base_idx]).is_some()),
            };
        }
        entry
    }).collect();
    let any_evidence = rows.iter().enumerate().any(|(i, r)| {
        i != base_idx && paired_delta(&r.trace, base).0.is_finite()
    });
    let mut out = json!({
        "baseline": rows[base_idx].name,
        "metrics": metrics,
        // The same footnote the table and markdown renderers print: a consumer
        // reading `evidence_label` out of the JSON must get the caveat that
        // travels with it.
        "evidence_scale_note": any_evidence.then_some(JEFFREYS_SCALE_NOTE),
        // The `se_delta_elpd` column's own caveat at a short scoring window.
        "se_caveat": se_caveat(base.n_scored()),
        // Every waiver the comparison needed to render: the flag, the rows it
        // concerned, what differed, and what it does to the reading. Empty
        // when every cross-row check passed.
        "overrides": checks.overrides().iter().map(override_json).collect::<Vec<_>>(),
        // The bound-data check (gh#713): present only under
        // --allow-data-mismatch, and the rows the check could not cover.
        "data_mismatch": checks.waived_finding(CheckId::SharedData).map(shared_data_statement),
        "data_unchecked": data_unchecked_note(checks, rows),
        // Rows whose recorded hashes are 8 characters (an older fit.meta.json):
        // the shared-data check compared them on that prefix.
        "data_compared_on_prefix": data_prefix_note(checks, rows),
        // Particle-filter settings that differ across rows (present only under
        // --allow-mismatched-filter-settings), and the rows that record none.
        "filter_settings_mismatch": checks.waived_finding(CheckId::FilterSettings)
            .map(|f| finding_note(f, Waiver::MismatchedFilterSettings)),
        "filter_settings_unchecked": filter_settings_unchecked_note(checks, rows),
        // Rows whose evidence label skipped the within-filter-noise gate
        // because only one side of the pair carries an MC SE.
        "filter_noise_unassessed": filter_noise_unassessed_note(rows, base_idx),
        // #295: the plug-in / in-sample caveat, `null` only when every compared
        // trace is honest on both axes.
        "optimism_caveat": optimism_caveat(rows),
        "rows": entries,
    });
    if let Some(leaf) = scored {
        out["stream"] = json!(leaf);
    }
    out
}

fn option_finite(x: f64) -> serde_json::Value {
    if x.is_finite() { serde_json::json!(x) } else { serde_json::Value::Null }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stored trace below schema 7 cannot say whether a forcing was held
    /// while it was scored, so compare refuses it, naming the file and the
    /// commands that write a current one; a schema-7 trace reads.
    #[test]
    fn a_trace_below_schema_7_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let trace = |v: u32| format!(r#"{{"schema_version": {v}, "t0": 0, "provenance": "plug_in",
            "conditioning": "in_sample", "steps": [{{"t": 7.0, "log_score": -1.0, "ess": 100.0}}],
            "warnings": []}}"#);
        let path = tmp.path().join("prequential.json");
        std::fs::write(&path, trace(6)).unwrap();
        let err = try_load_explicit_prequential(&path.to_string_lossy()).unwrap_err();
        assert!(err.contains(&path.display().to_string()), "{err}");
        assert!(err.contains("schema_version 6")
            && err.contains("written before held forcings were recorded"), "{err}");
        assert!(err.contains("camdl fit run <config>")
            && err.contains("camdl pfilter --save-prequential"), "{err}");
        std::fs::write(&path, trace(7)).unwrap();
        assert!(try_load_explicit_prequential(&path.to_string_lossy()).unwrap().is_some());
    }

    /// A held forcing's row line says what bounds its range and, for a
    /// lagged forcing, that the times are of its argument, as the run's own
    /// warning does; an unrecorded lag adds nothing.
    #[test]
    fn a_held_forcing_line_carries_its_bound_and_lag() {
        let held = |bound, lagged| PrequentialWarning::ForcingHeld {
            forcing: "mult".into(),
            defined_on: [0.0, 50.0],
            side: score::trace::ForcingSide::After,
            held_value_at: 50.0,
            bound,
            lagged,
        };
        let line = held_forcing_line("a", &held(Some(score::trace::ForcingBound::Window), Some(true))).unwrap();
        assert!(line.contains("t = 0 to t = 50 (its window)") && line.contains("t − lag"), "{line}");
        let line = held_forcing_line("a", &held(None, None)).unwrap();
        assert!(!line.contains("its window") && !line.contains("lag"), "{line}");
        assert_eq!(held_forcing_line("a", &PrequentialWarning::StartsAtPrior), None);
    }

    /// What `cmd_compare` would do with `rows` under `waivers`: the outcome
    /// every renderer reads, or every refusal as the command prints them.
    fn outcome_of(rows: &[Row], waivers: &Waivers) -> Result<Preflight, String> {
        let checks = preflight(rows, waivers);
        let refusals = checks.refusals();
        if refusals.is_empty() {
            Ok(checks)
        } else {
            Err(refusals.iter().map(|r| refusal_message(r, rows)).collect::<Vec<_>>().join("\n"))
        }
    }

    /// The pairing check over rows of equal horizons, as a refusal's text.
    fn axis(rows: &[Row]) -> Result<(), String> {
        pairing_finding(rows, false).map_or(Ok(()), |f| Err(finding_text(&f)))
    }

    /// The pairing check over rows of differing horizons, joined by time.
    fn shared_times(rows: &[Row]) -> Result<(), String> {
        pairing_finding(rows, true).map_or(Ok(()), |f| Err(finding_text(&f)))
    }

    /// gh#312: a particle filter's log score is a Monte-Carlo estimate biased
    /// downward by an amount that depends on the model and the particle count,
    /// and an ODE trace's is exact, so a Δelpd between them would carry that
    /// bias as a model difference. The pair is refused whatever is waived; two
    /// rows on one basis are not; and the basis is stated in every format.
    #[test]
    fn traces_on_different_predictive_bases_are_never_differenced() {
        let ode_basis = PredictiveBasis::OdeFreeForward { samples_per_draw: 100 };
        let mut rows = vec![
            row_at("pf", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("ode", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        rows[1].trace.basis = ode_basis;
        let err = outcome_of(&rows, &every_waiver())
            .expect_err("no waiver lets a filtered and an ODE trace be differenced");
        assert!(err.contains("mix predictive bases"), "{err}");
        assert!(err.contains("pf: filtered") && err.contains("ode: ode_free_forward"),
            "names each row's basis: {err}");

        rows[0].trace.basis = ode_basis;
        let checks = outcome_of(&rows, &Waivers::default()).expect("two ODE rows compare");
        let metrics = vec!["elpd".to_string()];
        let ode_line = "Predictive: ODE free-forward";
        assert!(render_table(&rows, 0, &metrics, &checks).contains(ode_line));
        assert!(render_md(&rows, 0, &metrics, &checks).contains(ode_line));
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checks)).unwrap();
        for r in v["rows"].as_array().unwrap() {
            assert_eq!(r["basis"], serde_json::json!({"ode_free_forward": {"samples_per_draw": 100}}),
                "{v}");
        }

        // A filtered cohort's table is as it was; its JSON states the basis.
        let pf = vec![row_at("a", &[7.0], &[-6.0]), row_at("b", &[7.0], &[-5.0])];
        let checks = outcome_of(&pf, &Waivers::default()).unwrap();
        assert!(!render_table(&pf, 0, &metrics, &checks).contains("Predictive:"));
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&pf, 0, &metrics, &checks)).unwrap();
        assert_eq!(v["rows"][0]["basis"], "filtered", "{v}");
    }

    /// The exponentiated Δelpd column reports an in-sample likelihood ratio
    /// against the baseline. It is not an e-value: an e-value is a non-negative
    /// statistic whose expectation under the null is at most 1, which licenses
    /// an anytime-valid reading ("1/E is a p-value bound"). Nothing here
    /// establishes that — θ is fit to the same data the ratio is computed on —
    /// so the name is `LR` in every format, and the word appears nowhere.
    #[test]
    fn the_ratio_column_is_named_lr_in_every_format() {
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        let metrics = vec!["elpd".to_string()];

        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(table.contains("LR"), "the table names the column LR:\n{table}");
        assert!(!table.contains("E_T"), "and never E_T:\n{table}");

        let md = render_md(&rows, 0, &metrics, &checked(&rows));
        assert!(md.contains("| LR |"), "the md header names the column LR:\n{md}");
        assert!(!md.contains("E_T"), "and never E_T:\n{md}");

        let json = render_json(&rows, 0, &metrics, &checked(&rows));
        assert!(json.contains("\"lr\""), "the JSON key is `lr`:\n{json}");
        assert!(!json.contains("\"e_t\""), "and never `e_t`:\n{json}");
    }

    /// A Δelpd smaller than twice its own paired standard error is not evidence
    /// of anything, and a Jeffreys word attached to it reads as a finding. Here
    /// Δ = +1.00 nats with se(Δ) = 1.00: the tier would have been
    /// "indeterminate" at ±4.3 dB, but even "indeterminate" is a claim about a
    /// quantity we cannot resolve. Every format says so instead.
    #[test]
    fn a_delta_inside_the_paired_se_band_prints_within_noise() {
        // diffs = [+1, 0, 0] → Δ = 1.00, se = √(3·Var) = 1.00, so |Δ| < 2·se.
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        let metrics = vec!["elpd".to_string()];

        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(table.contains("within noise"),
            "a Δ inside the noise band is labelled `within noise`:\n{table}");
        assert!(!table.contains("indeterminate"),
            "and carries no Jeffreys tier:\n{table}");

        let md = render_md(&rows, 0, &metrics, &checked(&rows));
        assert!(md.contains("within noise"), "the md table agrees:\n{md}");
        assert!(!md.contains("indeterminate"), "{md}");

        let json = render_json(&rows, 0, &metrics, &checked(&rows));
        assert!(json.contains("\"evidence_label\": \"within_noise\""),
            "the JSON label is `within_noise`, not a tier:\n{json}");
    }

    /// The negative control: a Δ well outside the noise band keeps its tier, so
    /// the gate cannot be satisfied by silencing every label. diffs = [3, 3, 4]
    /// → Δ = 10 nats, se = 1.00.
    #[test]
    fn a_delta_beyond_the_paired_se_band_keeps_its_tier() {
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-3.0, -3.0, -2.0]),
        ];
        let metrics = vec!["elpd".to_string()];

        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(table.contains("overwhelming for"),
            "10 nats ≈ 43 dB against se = 1.00 keeps its tier:\n{table}");
        assert!(!table.contains("within noise"), "{table}");

        let json = render_json(&rows, 0, &metrics, &checked(&rows));
        assert!(json.contains("\"evidence_label\": \"overwhelming\""),
            "the JSON label is the tier:\n{json}");
    }

    /// gh#706. `paired_delta` pairs steps BY INDEX, which is safe for the
    /// scalar only because the T_score preflight refuses mismatched horizons.
    /// Equal horizons are not equal observation axes: two traces can score the
    /// same NUMBER of observations at different times (a hole in one series, a
    /// different `t0`). Differencing those is not a comparison, and a pointwise
    /// file is exactly where it would become a plot that looks fine — so it is
    /// refused by name here.
    #[test]
    fn pointwise_refuses_traces_on_different_observation_axes() {
        let base = || row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]);
        let b = row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]);
        let ok = pointwise_rows(&[b, base()], 1).expect("aligned traces score");
        assert_eq!(ok.joint.len(), 3, "one joint row per step");
        assert!(ok.streams.is_empty(), "no per-stream data here");
        assert!((ok.joint[0].delta().unwrap() - 1.0).abs() < 1e-12);

        // Same count, different times.
        let shifted = row_at("b", &[7.0, 14.0, 28.0], &[-5.0, -6.0, -6.0]);
        let err = pointwise_rows(&[shifted, base()], 1)
            .expect_err("a shifted observation axis must be refused");
        assert!(err.contains("observation axis") && err.contains("21") && err.contains("28"),
            "the refusal must name both times: {err}");

        // Different counts.
        let short = row_at("b", &[7.0, 14.0], &[-5.0, -6.0]);
        let err = pointwise_rows(&[short, base()], 1)
            .expect_err("a shorter trace must be refused");
        assert!(err.contains("same observations"), "{err}");
    }

    /// An absent score is an EMPTY cell, never `NaN` and never `0` — a reader
    /// must be able to tell "did not score here" from "scored zero here", and
    /// zero is a perfectly ordinary log score.
    #[test]
    fn pointwise_tsv_writes_an_absent_score_as_an_empty_cell() {
        let r = StreamPointwiseRow {
            stream: "east".into(),
            row: PointwiseRow {
                model: "cand".into(),
                baseline: "base".into(),
                t: 14.0,
                log_score: Some(-1.0),
                baseline_log_score: None,
            },
        };
        assert_eq!(r.row.delta(), None, "no difference without both sides");
        let tsv = render_pointwise_streams_tsv(&[r], StreamsFile::BesideJoint("scores.tsv"));
        let data = tsv.lines().find(|l| l.starts_with("cand")).expect("a data row");
        let cells: Vec<&str> = data.split('\t').collect();
        assert_eq!(cells.len(), 7, "seven columns: {cells:?}");
        assert_eq!(cells[2], "14", "integral times are written as integers");
        assert_eq!(cells[3], "east");
        assert_eq!(cells[5], "", "the absent baseline score is empty: {cells:?}");
        assert_eq!(cells[6], "", "the absent difference is empty: {cells:?}");
        assert!(!tsv.contains("NaN"), "never NaN: {tsv}");
    }

    /// D16: the joint scores and the per-stream scores go to two files, and
    /// neither has a `scope` column. Each header says what its `log_score` is;
    /// the per-stream file says its rows do not sum to the joint file's.
    #[test]
    fn pointwise_writes_the_joint_and_the_per_stream_scores_to_two_files() {
        let t = [7.0, 14.0];
        let base = row_with_streams("base", &t, &["north", "south"]);
        let mut cand = row_with_streams("cand", &t, &["north", "south"]);
        cand.trace.steps[1].log_score = -4.0;
        cand.trace.steps[1].per_stream[0].log_score = -1.0;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scores.tsv");
        let w = write_pointwise(&[cand, base], 1, &path).expect("writes");
        assert_eq!(w.joint, (path.clone(), 2));
        assert_eq!(w.streams, (dir.path().join("scores.streams.tsv"), 4));

        let joint = std::fs::read_to_string(&path).unwrap();
        let header = joint.lines().find(|l| !l.starts_with('#')).unwrap();
        assert_eq!(header, "model\tbaseline\tt\tlog_score\tbaseline_log_score\tdelta_log_score");
        assert!(joint.contains("# log_score is the log of the joint"), "{joint}");
        assert!(joint.contains("scores.streams.tsv"), "names its sibling: {joint}");
        let rows: Vec<&str> = joint.lines().filter(|l| l.starts_with("cand")).collect();
        assert_eq!(rows[1].split('\t').nth(5), Some("2.0000000000"), "{joint}");

        let streams = std::fs::read_to_string(dir.path().join("scores.streams.tsv")).unwrap();
        let header = streams.lines().find(|l| !l.starts_with('#')).unwrap();
        assert_eq!(header,
            "model\tbaseline\tt\tstream\tlog_score\tbaseline_log_score\tdelta_log_score");
        assert!(streams.contains("do not sum to the joint scores in scores.tsv"), "{streams}");
        let north14 = streams.lines()
            .find(|l| l.starts_with("cand\tbase\t14\tnorth")).expect("north at 14");
        assert_eq!(north14.split('\t').nth(6), Some("2.0000000000"), "{streams}");
        for text in [&joint, &streams] {
            assert!(!text.contains("scope"), "no scope column: {text}");
        }
    }

    /// gh#295 / the "one format, one warning" rule: a warning or caveat that
    /// the table prints and another format drops is a silent-wrong answer for
    /// whoever reads that other format. The markdown table is what goes into a
    /// report and the JSON is what a script reads — a miscalibration flag, a
    /// trace warning, and the optimism caveat must reach all three.
    #[test]
    fn no_output_format_drops_a_warning_or_the_optimism_caveat() {
        use score::trace::PrequentialWarning;
        let mut rows = vec![
            row_with_streams("a", &[7.0, 14.0, 21.0], &["cases"]),
            row_with_streams("b", &[7.0, 14.0, 21.0], &["cases"]),
        ];
        rows[1].trace.steps[0].log_score = -5.0;
        // Row `b`: every observation lands outside its own 90% predictive
        // interval (PIT 0.99), and the filter saved no predictive samples.
        for s in &mut rows[1].trace.steps {
            s.per_stream[0].pit = 0.99;
        }
        rows[1].trace.warnings.push(PrequentialWarning::SamplesNotSaved);
        let metrics = vec!["elpd".to_string(), "pit_cov90".to_string()];

        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        let md = render_md(&rows, 0, &metrics, &checked(&rows));
        let json = render_json(&rows, 0, &metrics, &checked(&rows));

        for (fmt, text) in [("table", &table), ("md", &md)] {
            assert!(text.contains("PIT 90%-coverage"),
                "the {fmt} rendering carries the miscalibration flag:\n{text}");
            assert!(text.contains("SamplesNotSaved"),
                "the {fmt} rendering carries the trace warning:\n{text}");
            assert!(text.contains("not a leave-future-out forecast score"),
                "the {fmt} rendering carries the optimism caveat:\n{text}");
        }

        let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(v["rows"][0]["conditioning"], "in_sample",
            "each row states how θ was conditioned:\n{json}");
        assert_eq!(v["rows"][0]["provenance"], "plug_in",
            "and how the predictive was built:\n{json}");
        assert!(v["optimism_caveat"].as_str()
                .is_some_and(|s| s.contains("not a leave-future-out forecast score")),
            "the caveat is a top-level field:\n{json}");
        assert_eq!(v["rows"][1]["warnings"][0]["kind"], "samples_not_saved",
            "the trace's own warnings ride with its row:\n{json}");
        assert!(v["rows"][1]["pit_cov90_warning"].as_str()
                .is_some_and(|s| s.contains("PIT 90%-coverage")),
            "and the miscalibration flag the table prints:\n{json}");
        assert!(v["rows"][0]["pit_cov90_warning"].is_null(),
            "a well-calibrated row carries no flag:\n{json}");
    }

    /// gh#570. Equal T_score and equal times still permit two traces to have
    /// scored DIFFERENT streams at those times — a national fit scoring two
    /// districts against one scoring a single district. The joint `log_score`
    /// is a cross-stream quantity, so differencing them subtracts a 2-stream
    /// joint from a 1-stream joint and calls the gap evidence about the model.
    /// The preflight refuses the pair, naming what differs.
    #[test]
    fn refuses_traces_that_scored_different_stream_sets() {
        let t = [7.0, 14.0, 21.0];

        // Same streams at every step → the ordinary case still compares.
        let a = row_with_streams("a", &t, &["north", "south"]);
        let b = row_with_streams("b", &t, &["south", "north"]);
        axis(&[a, b])
            .expect("the same stream set in a different order is the same set");

        // Different sets → refused, naming the step, both sets, both traces.
        let a = row_with_streams("a", &t, &["north", "south"]);
        let b = row_with_streams("b", &t, &["north"]);
        let err = axis(&[a, b])
            .expect_err("a 2-stream joint may not be differenced against a 1-stream one");
        assert!(err.contains("south"), "names the stream that differs: {err}");
        assert!(err.contains("'a'") && err.contains("'b'"),
            "names both traces: {err}");
        assert!(err.contains("t=7"), "and the step where they diverge: {err}");

        // Two legacy v1 traces (no per-stream breakdown at all) carry no
        // information to check, so they are allowed through.
        let a = row_at("a", &t, &[-6.0, -6.0, -6.0]);
        let b = row_at("b", &t, &[-6.0, -6.0, -6.0]);
        axis(&[a, b])
            .expect("two traces with no per-stream breakdown are not a mismatch");

        // One with a breakdown and one without: the sets cannot be compared, so
        // the difference cannot be shown to be like-for-like. Refused, and the
        // message says which side is missing the breakdown rather than naming a
        // phantom stream mismatch.
        let a = row_with_streams("a", &t, &["north", "south"]);
        let b = row_at("b", &t, &[-6.0, -6.0, -6.0]);
        let err = axis(&[a, b])
            .expect_err("an unverifiable stream set is not a verified one");
        assert!(err.contains("no per-stream breakdown"),
            "the message names the real problem: {err}");
    }

    /// gh#713: `compare @a @b` says nothing about whether the two fits read the
    /// same observations. They are separately consistent, and their derived
    /// traces agree on T_score, times and stream names — those come from the
    /// model, not the data — so a comparison against a revised case series
    /// renders as a clean model verdict. The fits' recorded data hashes are the
    /// only place the difference shows.
    #[test]
    fn refuses_fits_bound_to_different_observed_data() {
        let same_a = digests(&[("north", "aaa"), ("south", "bbb")]);
        let same_b = digests(&[("north", "aaa"), ("south", "bbb")]);
        failure(&check_shared_data(&data_rows(&[("a", &same_a), ("b", &same_b)])))
            .expect("fits bound to the same bytes compare");

        // One stream's file differs → refused, naming the stream and both fits.
        let revised = digests(&[("north", "aaa"), ("south", "ccc")]);
        let err = failure(&check_shared_data(&data_rows(&[("a", &same_a), ("b", &revised)])))
            .expect_err("a differing data file must be refused");
        assert!(err.contains("south"), "names the stream that differs: {err}");
        assert!(!err.contains("north"), "and not the one that agrees: {err}");
        assert!(err.contains("'a'") && err.contains("'b'"), "names both fits: {err}");
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        rows[0].data = digests(&[("north", "aaa"), ("south", "bbb")]);
        rows[1].data = revised;
        let refusal = outcome_of(&rows, &Waivers::default())
            .expect_err("refused without the flag");
        assert!(refusal.contains("south") && refusal.contains("--allow-data-mismatch"),
            "and how to proceed deliberately: {refusal}");

        // A stream one fit bound and the other did not is the same failure.
        let extra = digests(&[("north", "aaa"), ("south", "bbb"), ("east", "ddd")]);
        let err = failure(&check_shared_data(&data_rows(&[("a", &same_a), ("b", &extra)])))
            .expect_err("an extra bound stream must be refused");
        assert!(err.contains("east"), "{err}");

        // Rows with no data identity are not evidence either way: they neither
        // refuse the comparison nor pass silently.
        let unknown = DataIdentity::Unknown("read as an explicit prequential.json".into());
        let rows = data_rows(&[("a", &same_a), ("t", &unknown)]);
        let result = check_shared_data(&rows);
        failure(&result).expect("a row with no data identity cannot contradict anything");
        let note = data_unchecked_note(&Preflight { results: vec![result] }, &rows)
            .expect("but it is named");
        assert!(note.contains("'t'") && note.contains("prequential.json"), "{note}");
        assert!(!note.contains("'a'"), "the checked row is not in the note: {note}");
        let rows = data_rows(&[("a", &same_a), ("b", &same_b)]);
        let result = check_shared_data(&rows);
        assert!(data_unchecked_note(&Preflight { results: vec![result] }, &rows).is_none(),
            "no note when every row was checked");
    }

    /// A 64-character hash: `head` padded with `f`.
    fn full_hash(head: &str) -> String {
        format!("{head:f<64}")
    }

    /// Every order of three items.
    fn orders<T: Clone>(v: &[T; 3]) -> Vec<Vec<T>> {
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]].iter()
            .map(|o| o.iter().map(|&i| v[i].clone()).collect()).collect()
    }

    fn qualified_members(r: &CheckResult) -> Vec<&str> {
        r.qualified.iter().map(|q| q.member.as_str()).collect()
    }

    /// Recorded hashes are compared pair by pair, each pair on the shorter
    /// of its two lengths: a short hash in one row (an older fit record holds
    /// 8 characters) does not shorten the comparison of two full ones. `a`
    /// and `b` differ after a shared 8-character prefix and are refused in
    /// every order of the rows; `c`, the 8-character prefix of both, matches
    /// either and is the only row whose comparison rested on a short hash.
    #[test]
    fn two_full_hashes_that_differ_after_a_short_ones_prefix_are_refused() {
        let (a, b) = (full_hash("aaaaaaaa1"), full_hash("aaaaaaaa2"));
        let ids = [("a", digests(&[("cases", a.as_str())])),
                   ("b", digests(&[("cases", b.as_str())])),
                   ("c", digests(&[("cases", "aaaaaaaa")]))];
        let waived = Waivers { data_mismatch: true, ..Default::default() };
        for order in orders(&ids) {
            let rows = data_rows(&order.iter().map(|(n, d)| (*n, d)).collect::<Vec<_>>());
            let names: Vec<&str> = order.iter().map(|(n, _)| *n).collect();
            let r = check_shared_data(&rows);
            // The full hashes differ, so `c` (a prefix of both) is shown on
            // its own value, which sorts first.
            let want = Finding::DataDiffers {
                sources: vec![SourceDigests { source: "cases".into(),
                    groups: vec![(Some("aaaaaaaa".into()), vec!["c".into()]),
                                 (Some(a.clone()), vec!["a".into()]),
                                 (Some(b.clone()), vec!["b".into()])] }],
                unhashed: vec![],
            };
            assert_eq!(r.outcome, CheckOutcome::Failed { finding: want, waived: false }, "{names:?}");
            assert_eq!(qualified_members(&r), ["c"], "{names:?}");
            let refusal = outcome_of(&rows, &Waivers::default()).expect_err("refused");
            assert!(refusal.contains("different observed data")
                    && refusal.contains("--allow-data-mismatch"), "{names:?}: {refusal}");
            let checks = outcome_of(&rows, &waived).expect("renders under the waiver");
            assert_eq!(checks.overrides().iter().map(|o| o.waiver).collect::<Vec<_>>(),
                [Waiver::DataMismatch], "{names:?}: the waiver is recorded");
        }

        // A full hash and its 8-character prefix match; the short row is
        // qualified, and only it.
        let rows = data_rows(&[("a", &ids[0].1), ("c", &ids[2].1)]);
        let r = check_shared_data(&rows);
        assert_eq!(r.outcome, CheckOutcome::Passed, "{r:?}");
        assert_eq!(qualified_members(&r), ["c"]);

        // Two rows with the same full hash and the short one pass.
        let a2 = digests(&[("cases", a.to_ascii_uppercase().as_str())]);
        for order in orders(&[("a", ids[0].1.clone()), ("a2", a2), ("c", ids[2].1.clone())]) {
            let rows = data_rows(&order.iter().map(|(n, d)| (*n, d)).collect::<Vec<_>>());
            let r = check_shared_data(&rows);
            assert_eq!(r.outcome, CheckOutcome::Passed, "{r:?}");
            assert_eq!(qualified_members(&r), ["c"]);
        }

        // An empty recorded hash matches nothing: not a full hash, not a short
        // one, not another empty one.
        let empty = digests(&[("cases", "")]);
        for other in [&ids[0].1, &ids[2].1, &empty] {
            let r = check_shared_data(&data_rows(&[("e", &empty), ("x", other)]));
            assert!(matches!(r.outcome, CheckOutcome::Failed { .. }), "{r:?}");
            assert!(r.qualified.is_empty(), "no comparison rested on a short hash: {r:?}");
        }
    }

    /// The same cases on a table re-scored on one stream: only the scored
    /// stream's source is compared, pair by pair.
    #[test]
    fn on_one_stream_two_full_hashes_that_differ_after_a_short_ones_prefix_are_refused() {
        let (a, b) = (full_hash("aaaaaaaa1"), full_hash("aaaaaaaa2"));
        let deaths = full_hash("dddddddd");
        let row = |name: &str, cases: &str| {
            let mut r = derived_row(name, &CASES_DEATHS, &["cases", "deaths"]);
            r.data = digests(&[("cases", cases), ("deaths", deaths.as_str())]);
            r.onto("cases").unwrap()
        };
        let waived = Waivers { data_mismatch: true, ..Default::default() };
        let rows3 = [row("a", &a), row("b", &b), row("c", "aaaaaaaa")];
        for rows in orders(&rows3) {
            let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
            let checks = preflight(&rows, &Waivers::default());
            let r = checks.result(CheckId::SharedData).unwrap();
            assert!(matches!(&r.outcome, CheckOutcome::Failed { finding: Finding::DataDiffers {
                sources, .. }, waived: false } if sources.len() == 1 && sources[0].groups.len() == 3),
                "{names:?}: {r:?}");
            assert_eq!(qualified_members(r), ["c"], "{names:?}");
            let refusal = outcome_of(&rows, &Waivers::default()).expect_err("refused");
            assert!(refusal.contains("--allow-data-mismatch"), "{names:?}: {refusal}");
            let checks = outcome_of(&rows, &waived).expect("renders under the waiver");
            assert_eq!(checks.overrides().iter().map(|o| o.waiver).collect::<Vec<_>>(),
                [Waiver::DataMismatch], "{names:?}");
        }
        let pass = |rows: &[Row]| {
            let checks = outcome_of(rows, &Waivers::default()).expect("passes");
            let r = checks.result(CheckId::SharedData).unwrap().clone();
            assert_eq!(r.outcome, CheckOutcome::Passed, "{r:?}");
            assert_eq!(qualified_members(&r), ["c"]);
        };
        pass(&[row("a", &a), row("c", "aaaaaaaa")]);
        for rows in orders(&[row("a", &a), row("a2", &a), row("c", "aaaaaaaa")]) {
            pass(&rows);
        }
        let r = preflight(&[row("a", &a), row("e", "")], &Waivers::default());
        assert!(matches!(r.result(CheckId::SharedData).unwrap().outcome,
            CheckOutcome::Failed { .. }), "an empty hash matches nothing: {r:?}");
    }

    /// Rows that differ only in the data identity given, in that order.
    fn data_rows(entries: &[(&str, &DataIdentity)]) -> Vec<Row> {
        entries.iter().map(|(name, d)| Row {
            data: (*d).clone(),
            ..row_at(name, &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0])
        }).collect()
    }

    /// A check's result as the old `Result`: `Err` with the finding's text
    /// when it failed.
    fn failure(r: &CheckResult) -> Result<(), String> {
        match &r.outcome {
            CheckOutcome::Failed { finding, .. } => Err(finding_text(finding)),
            _ => Ok(()),
        }
    }

    /// A failed check's finding with its consequence, as an override states
    /// it, or `None` when it did not fail.
    fn waived_note(r: &CheckResult) -> Option<String> {
        match &r.outcome {
            CheckOutcome::Failed { finding, .. } =>
                Some(finding_note(finding, r.waiver.expect("a waivable check"))),
            _ => None,
        }
    }

    /// Two fits sealed at the same observation can record boundaries that
    /// differ by float noise; those are one window, not two. A real
    /// difference between boundaries is still a refusal.
    #[test]
    fn sealed_boundaries_agree_up_to_the_holdout_time_tolerance() {
        assert!(same_sealed_boundary(60.0, 60.0 + 1e-12),
            "float noise at the boundary is the same observation time");
        assert!(same_sealed_boundary(60.0 + 1e-12, 60.0), "symmetric");
        assert!(!same_sealed_boundary(21.0, 21.5),
            "half a day apart is two different training windows");
    }

    /// `--allow-mismatched-horizon` waives the T_score check and nothing else.
    /// Two traces of different lengths still refuse when, at a time both
    /// scored, they scored different stream sets (gh#570) or the same stream
    /// over different windows (gh#833). Times only one trace scored are what
    /// the flag waives.
    #[test]
    fn mismatched_horizons_still_check_streams_and_coverage_at_shared_times() {
        use score::trace::Coverage;
        let iv = |s: f64, e: f64| Coverage::Interval { start: s, stop: e };

        // (i) Different stream sets at shared t = 14.
        let short = row_with_streams("short", &[7.0, 14.0], &["north"]);
        let long = row_with_streams("long", &[7.0, 14.0, 21.0], &["north", "south"]);
        let err = shared_times(&[short, long])
            .expect_err("different stream sets at a shared time are refused");
        assert!(err.contains("did not score the same streams") && err.contains("south")
                && err.contains("t=7"), "{err}");

        // (ii) Same stream, different coverage at shared t = 14; the long
        // trace's extra step at 21 is not consulted.
        let short = row_covering("short", &[7.0, 14.0], "cases",
            &[iv(0.0, 7.0), iv(12.0, 14.0)]);
        let long = row_covering("long", &[7.0, 14.0, 21.0], "cases",
            &[iv(0.0, 7.0), iv(7.0, 14.0), iv(14.0, 21.0)]);
        let err = shared_times(&[short, long])
            .expect_err("different coverage at a shared time is refused");
        assert!(err.contains("over different windows") && err.contains("t=14"), "{err}");

        // Shared times joined by t, not position: the short trace's t = 14
        // pairs with the long trace's t = 14 although it sits at index 0.
        let short = row_covering("short", &[14.0], "cases", &[iv(12.0, 14.0)]);
        let long = row_covering("long", &[7.0, 14.0], "cases", &[iv(0.0, 7.0), iv(7.0, 14.0)]);
        assert!(shared_times(&[short, long]).is_err(), "joined by time");

        // (iii) Same streams and coverage at every shared time: it renders,
        // with the Δ columns suppressed as before.
        let short = row_covering("short", &[7.0, 14.0], "cases", &[iv(0.0, 7.0), iv(7.0, 14.0)]);
        let long = row_covering("long", &[7.0, 14.0, 21.0], "cases",
            &[iv(0.0, 7.0), iv(7.0, 14.0), iv(14.0, 21.0)]);
        let rows = vec![short, long];
        shared_times(&rows).expect("agreeing shared times render");
        let table = render_table(&rows, 0, &["elpd".to_string()], &checked(&rows));
        assert!(table.contains("Δ columns suppressed"), "{table}");
    }

    fn derivation(particles: usize, replicates: usize)
        -> score::trace::Derivation
    {
        score::trace::Derivation {
            particles, replicates, dt: 1.0, seed: 1,
            theta: score::trace::ThetaSource::OptimizerWinner,
            excluded_chains: vec![], enabled: vec![], disabled: vec![],
            replicate_elpd: vec![0.0; replicates],
            replicate_elpd_by_stream: Default::default(),
            mc_se_elpd: None,
            overrides: vec![],
        }
    }

    /// The totals a replicated derive stores in its trace: each replicate's
    /// elpd and per-stream sums in replicate order, and the filter-noise SE
    /// as the log-mean-exp SE of exactly those totals. One replicate has no
    /// SE. (The per-stream sums are not expected to add up to the joint.)
    #[test]
    fn replicate_totals_are_stored_in_order_with_their_filter_noise_se() {
        let t = [7.0, 14.0, 21.0];
        let mut a = row_with_streams("a", &t, &["north", "south"]).trace;
        let mut b = row_with_streams("b", &t, &["north", "south"]).trace;
        a.steps[0].log_score = -4.0;
        b.steps[2].per_stream[1].log_score = -5.0;

        let totals = score::combine::ReplicateTotals::of(&[a.clone(), b.clone()]);
        assert_eq!(totals.elpd, vec![a.elpd(), b.elpd()]);
        assert_eq!(totals.elpd, vec![-16.0, -18.0]);
        assert_eq!(totals.by_stream["north"], vec![-9.0, -9.0]);
        assert_eq!(totals.by_stream["south"], vec![-9.0, -11.0]);
        assert_eq!(totals.mc_se,
            Some(numerics::logmeanexp_with_se(&totals.elpd).1),
            "the SE is the log-mean-exp SE of the stored totals");
        assert!(totals.mc_se.is_some_and(|se| se > 0.0));

        let one = score::combine::ReplicateTotals::of(&[a]);
        assert_eq!(one.elpd.len(), 1);
        assert_eq!(one.by_stream["north"].len(), 1);
        assert_eq!(one.mc_se, None, "a single replicate has no filter-noise SE");
    }

    /// Give a test row the filter-noise SE a replicated derive records, or
    /// (for `None`) no derivation block at all, as an explicit trace has.
    fn set_mc_se(row: &mut Row, mc: Option<f64>) {
        row.trace.derivation = mc.map(|se| score::trace::Derivation {
            mc_se_elpd: Some(se),
            ..derivation(1000, 5)
        });
    }

    /// Two rows scored at different particle or replicate counts differ
    /// partly by the filters — the log of a PF likelihood estimate is biased
    /// downward, more so at fewer particles — so the pair is named with both
    /// values; equal settings say nothing. A rendering under the override
    /// states the mismatch in every format.
    #[test]
    fn rows_scored_with_different_filter_settings_are_named_in_every_format() {
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        rows[0].trace.derivation = Some(derivation(1000, 5));
        rows[1].trace.derivation = Some(derivation(1000, 5));
        assert!(waived_note(&check_filter_settings(&rows)).is_none(), "equal settings");
        assert!(filter_settings_unchecked_note(&Preflight { results: vec![check_filter_settings(&rows)] }, &rows).is_none(), "both recorded");

        rows[1].trace.derivation = Some(derivation(200, 5));
        let note = waived_note(&check_filter_settings(&rows)).expect("particles differ");
        assert!(note.contains("'a'") && note.contains("'b'")
                && note.contains("particles 1000") && note.contains("particles 200"),
            "names both rows and both values: {note}");
        assert!(note.contains("biased downward"), "and why it matters: {note}");
        assert!(!note.contains("replicates 5 vs"), "replicates agree: {note}");

        rows[1].trace.derivation = Some(derivation(1000, 1));
        let note = waived_note(&check_filter_settings(&rows)).expect("replicates differ");
        assert!(note.contains("replicates 5") && note.contains("replicates 1"), "{note}");

        let metrics = vec!["elpd".to_string()];
        for (fmt, text) in [("table", render_table(&rows, 0, &metrics, &checked(&rows))),
                            ("md", render_md(&rows, 0, &metrics, &checked(&rows)))] {
            assert!(text.contains("different particle-filter settings")
                    && text.contains("--allow-mismatched-filter-settings was set"),
                "the {fmt} rendering states the mismatch:\n{text}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert!(v["filter_settings_mismatch"].as_str()
                .is_some_and(|s| s.contains("replicates 1")), "{v}");
        assert_eq!(v["rows"][1]["derivation"]["replicates"], serde_json::json!(1),
            "each row carries its own block: {v}");
    }

    /// A trace older than schema 5 records no filter settings. It is never
    /// passed silently: it reads, takes no part in the mismatch check, and
    /// is named as not checked; its JSON row carries a null block.
    #[test]
    fn a_trace_older_than_schema_5_is_reported_as_not_checked() {
        let old: PrequentialTrace = serde_json::from_str(r#"{
            "schema_version": 4, "t0": 0, "provenance": "plug_in",
            "conditioning": "in_sample",
            "steps": [{"t": 7.0, "y_obs": 1.0, "y_pred_samples": [],
                       "log_score": -1.0, "crps": 0.5, "pit": 0.5, "ess": 100.0}],
            "warnings": []
        }"#).expect("a schema-4 trace still reads");
        assert!(old.derivation.is_none());
        let mut rows = vec![
            row_at("new", &[7.0], &[-1.5]),
            row_at("old", &[7.0], &[-1.0]),
        ];
        rows[0].trace.derivation = Some(derivation(1000, 5));
        rows[1].trace = old;

        assert!(waived_note(&check_filter_settings(&rows)).is_none(),
            "an unrecorded side cannot contradict anything");
        let note = filter_settings_unchecked_note(&Preflight { results: vec![check_filter_settings(&rows)] }, &rows).expect("old is unchecked");
        assert!(note.contains(
            "filter settings not recorded for 'old' (trace older than schema 5); not checked"),
            "{note}");
        assert!(!note.contains("'new'"), "{note}");

        let metrics = vec!["elpd".to_string()];
        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(table.contains("filter settings not recorded for 'old'"), "{table}");
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert!(v["filter_settings_unchecked"].as_str().is_some_and(|s| s.contains("'old'")),
            "{v}");
        assert!(v["rows"][1]["derivation"].is_null(), "{v}");
        assert!(v["filter_settings_mismatch"].is_null(), "{v}");
    }

    /// The "within filter noise" gate needs a filter-noise MC SE on both rows
    /// of a pair. A derived row with replication against an explicit
    /// prequential.json (no SE) used to skip the gate without saying so; the
    /// note names the row in every format. Neither side having one (no
    /// replication requested) or both having one is not noted.
    #[test]
    fn a_pair_with_filter_noise_on_one_side_only_is_named_in_every_format() {
        let metrics = vec!["elpd".to_string()];
        let mk = |mc: [Option<f64>; 3]| {
            let mut rows = vec![
                row_at("base", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
                row_at("half", &[7.0, 14.0, 21.0], &[-3.0, -3.0, -2.0]),
                row_at("both", &[7.0, 14.0, 21.0], &[-3.0, -3.0, -2.0]),
            ];
            for (r, m) in rows.iter_mut().zip(mc) { set_mc_se(r, m); }
            rows
        };

        // Baseline and `both` replicated, `half` read as-is: only `half`.
        let rows = mk([Some(0.3), None, Some(0.4)]);
        let note = filter_noise_unassessed_note(&rows, 0).expect("half is unassessed");
        assert!(note.contains("'half'") && !note.contains("'both'"), "{note}");
        assert!(note.contains("not checked against filter noise"), "{note}");
        for (fmt, text) in [("table", render_table(&rows, 0, &metrics, &checked(&rows))),
                            ("md", render_md(&rows, 0, &metrics, &checked(&rows)))] {
            assert!(text.contains("not checked against filter noise") && text.contains("'half'"),
                "the {fmt} rendering carries the note:\n{text}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert!(v["filter_noise_unassessed"].as_str().is_some_and(|s| s.contains("'half'")),
            "the JSON carries it as a field:\n{v}");

        // The SE on the baseline side only is the same gap, for every pair.
        let rows = mk([Some(0.3), None, None]);
        let note = filter_noise_unassessed_note(&rows, 0).expect("both pairs one-sided");
        assert!(note.contains("'half'") && note.contains("'both'"), "{note}");
        // And the candidate side only.
        let rows = mk([None, Some(0.3), None]);
        let note = filter_noise_unassessed_note(&rows, 0).expect("half is one-sided");
        assert!(note.contains("'half'") && !note.contains("'both'"), "{note}");

        // No replication anywhere, and replication everywhere: nothing to say.
        assert!(filter_noise_unassessed_note(&mk([None, None, None]), 0).is_none());
        let all = mk([Some(0.3), Some(0.2), Some(0.4)]);
        assert!(filter_noise_unassessed_note(&all, 0).is_none());
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&all, 0, &metrics, &checked(&all))).unwrap();
        assert!(v["filter_noise_unassessed"].is_null(), "{v}");
    }

    /// `se(Δ)` is √(T·Var) of the per-observation differences — a normal
    /// approximation whose accuracy depends on T and on how alike the two
    /// models are. It stays a per-comparison fact in the JSON (`se_caveat`,
    /// populated only below 100 scored steps); the sentence explaining it left
    /// the human footer for the methods page (gh#806).
    #[test]
    fn the_small_t_se_caveat_is_a_json_field_not_a_footer_sentence() {
        let metrics = vec!["elpd".to_string()];
        let short = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];

        let json = render_json(&short, 0, &metrics, &checked(&short));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["se_caveat"].as_str().is_some_and(|s| s.contains("2008.10296")),
            "the JSON carries the caveat as a field:\n{json}");

        let table = render_table(&short, 0, &metrics, &checked(&short));
        assert!(!table.contains("2008.10296"),
            "the citation belongs to the methods page now:\n{table}");
        let md = render_md(&short, 0, &metrics, &checked(&short));
        assert!(!md.contains("2008.10296"), "{md}");

        // A long scoring window does not carry it at all: the field must mean
        // something when it is populated.
        let times: Vec<f64> = (1..=120).map(|i| i as f64).collect();
        let long = vec![
            row_at("a", &times, &vec![-6.0; 120]),
            row_at("b", &times, &vec![-5.9; 120]),
        ];
        let json = render_json(&long, 0, &metrics, &checked(&long));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["se_caveat"].is_null(),
            "120 scored steps is not a small-T window:\n{json}");
    }

    /// gh#806. The footer states what is specific to THIS comparison and
    /// points at the methods page for the rest. Both human renderings end at
    /// the pointer; neither repeats the Jeffreys-scale note, the small-T
    /// citation, or the Newey–West/HLN construction — and the JSON, the
    /// machine surface, keeps every field it had.
    #[test]
    fn the_human_footer_points_at_the_methods_page_and_the_json_keeps_its_fields() {
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        let metrics = vec!["elpd".to_string()];

        for (fmt, out) in [("table", render_table(&rows, 0, &metrics, &checked(&rows))),
                           ("md",    render_md(&rows, 0, &metrics, &checked(&rows)))] {
            assert!(out.contains("camdl compare --explain"),
                "the {fmt} footer names where the definitions are:\n{out}");
            assert!(out.contains("camdl docs model-comparison"),
                "and the offline topic:\n{out}");
            assert!(!out.contains("not Bayes factors"),
                "the {fmt} footer no longer carries the Jeffreys-scale \
                 sentence:\n{out}");
            assert!(!out.contains("Newey"),
                "nor the HAC construction:\n{out}");
            assert!(!out.contains("Harvey"), "{out}");
            // The numeric note it replaced still prints its value.
            assert!(out.contains("lag-1 autocorrelation of the per-step Δelpd"),
                "the lag-1 number survives the collapse:\n{out}");
        }

        let json = render_json(&rows, 0, &metrics, &checked(&rows));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["evidence_scale_note"].as_str()
                .is_some_and(|s| s.contains("not Bayes factors")),
            "the JSON keeps the Jeffreys-scale note verbatim:\n{json}");
        assert!(v["se_caveat"].as_str().is_some_and(|s| s.contains("2008.10296")),
            "and the small-T caveat:\n{json}");
        assert!(v["rows"][1]["lag1_autocorr_delta_elpd"].is_number(),
            "and the per-row lag-1 field:\n{json}");
    }

    /// gh#806. A comparison never said what θ had seen: `hold_out_tail`'s
    /// sealed boundary reached the JSON only, so a held-out table and an
    /// in-sample one were indistinguishable at a glance — while their Δelpd
    /// answer different questions. Both human renderings state it, once.
    #[test]
    fn the_footer_states_the_conditioning_the_rows_were_scored_under() {
        let metrics = vec!["elpd".to_string()];
        let in_sample = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        let table = render_table(&in_sample, 0, &metrics, &checked(&in_sample));
        assert!(table.contains("Conditioning: in-sample"),
            "the in-sample stamp reaches the table:\n{table}");
        assert_eq!(table.matches("Conditioning:").count(), 1,
            "stated once for the cohort, not once per row:\n{table}");
        let md = render_md(&in_sample, 0, &metrics, &checked(&in_sample));
        assert!(md.contains("Conditioning: in-sample"), "{md}");

        // Held out at t = 42: the boundary is the fact a reader needs, so the
        // line carries the number, not just the mode.
        let mut held = in_sample;
        for r in &mut held {
            r.trace.conditioning = Conditioning::HoldOutTail {
                train_end: 42.0,
                theta_source: "results/fits/sir-abc12345".into(),
            };
        }
        let table = render_table(&held, 0, &metrics, &checked(&held));
        assert!(table.contains("Conditioning: held-out tail (train_end = 42)"),
            "the sealed boundary is named:\n{table}");
        assert!(!table.contains("in-sample"), "{table}");
        let md = render_md(&held, 0, &metrics, &checked(&held));
        assert!(md.contains("train_end = 42"), "{md}");
    }

    /// A comparison rendered in spite of a data mismatch (the reader passed
    /// `--allow-data-mismatch`) must carry the confound in the artifact. The
    /// terminal session that granted the override is not where the table is
    /// read a week later.
    #[test]
    fn an_overridden_data_mismatch_is_stated_in_every_format() {
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        rows[0].data = digests(&[("cases", "aaaaaaaa11")]);
        rows[1].data = digests(&[("cases", "bbbbbbbb22")]);
        let metrics = vec!["elpd".to_string()];

        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(table.contains("different observed data") && table.contains("cases"),
            "the text table states the confound:\n{table}");
        let md = render_md(&rows, 0, &metrics, &checked(&rows));
        assert!(md.contains("different observed data"), "and the markdown:\n{md}");

        let json = render_json(&rows, 0, &metrics, &checked(&rows));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["data_mismatch"].as_str().is_some_and(|s| s.contains("cases")),
            "and the JSON, as a field:\n{json}");
        assert!(v["data_unchecked"].is_null(),
            "both rows carried digests, so nothing was skipped:\n{json}");

        // Matching data: no confound line anywhere.
        rows[1].data = digests(&[("cases", "aaaaaaaa11")]);
        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(!table.contains("different observed data"), "{table}");
        let json = render_json(&rows, 0, &metrics, &checked(&rows));
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["data_mismatch"].is_null(), "{json}");
    }

    /// Every waiver given to every check, as a test renders a comparison the
    /// command would have rendered.
    fn every_waiver() -> Waivers {
        Waivers {
            mismatched_horizon: true, mixed_provenance: true, data_mismatch: true,
            mismatched_filter_settings: true, in_sample_ablation: true,
        }
    }

    /// The preflight outcome `cmd_compare` would hand the renderers for
    /// `rows`, every flag given.
    fn checked(rows: &[Row]) -> Preflight {
        outcome_of(rows, &every_waiver()).expect("every check is waived")
    }

    /// Two clean rows, then the one change that makes `w` necessary.
    fn rows_needing(w: Waiver) -> Vec<Row> {
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        match w {
            Waiver::MismatchedHorizon => rows[1] = row_at("b", &[7.0, 14.0], &[-5.0, -6.0]),
            Waiver::MixedProvenance => rows[1].trace.provenance =
                score::trace::Provenance::Posterior { n_draws: 4 },
            Waiver::DataMismatch => {
                rows[0].data = digests(&[("cases", "aaaaaaaa11")]);
                rows[1].data = digests(&[("cases", "bbbbbbbb22")]);
            }
            Waiver::MismatchedFilterSettings => {
                rows[0].trace.derivation = Some(derivation(1000, 5));
                rows[1].trace.derivation = Some(derivation(200, 5));
            }
            // Fits of different stream sets, both re-scored on the one they
            // share, in-sample.
            Waiver::InSampleAblation => rows = vec![
                derived_row("a", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap(),
                derived_row("b", &CASES_DEATHS, &["cases"]).onto("cases").unwrap(),
            ],
        }
        rows
    }

    const ALL_WAIVERS: [Waiver; 5] = [
        Waiver::MismatchedHorizon, Waiver::MixedProvenance, Waiver::DataMismatch,
        Waiver::MismatchedFilterSettings, Waiver::InSampleAblation,
    ];

    /// D13: a comparison that renders only because a flag waived a failed
    /// check states that override once, in every format: the flag, the rows
    /// it concerned and what it does to the reading. Without the flag the
    /// same rows are refused.
    #[test]
    fn every_needed_override_is_stated_once_in_every_format() {
        let metrics = vec!["elpd".to_string()];
        for w in ALL_WAIVERS {
            let rows = rows_needing(w);
            assert!(outcome_of(&rows, &Waivers::default()).is_err(),
                "{w:?}: refused without its flag");
            let checks = checked(&rows);
            assert_eq!(checks.overrides().iter().map(|o| o.waiver).collect::<Vec<_>>(), vec![w],
                "{w:?}: exactly the needed override is listed");

            for (fmt, text) in [("table", render_table(&rows, 0, &metrics, &checks)),
                                ("md", render_md(&rows, 0, &metrics, &checks))] {
                let lines: Vec<&str> = text.lines()
                    .filter(|l| l.contains(w.flag())).collect();
                assert_eq!(lines.len(), 1, "{w:?}: the {fmt} rendering states it once:\n{text}");
                assert!(lines[0].contains(w.consequence()) && lines[0].contains("'b'"),
                    "{w:?}: with the rows and the consequence:\n{text}");
            }
            let v: serde_json::Value =
                serde_json::from_str(&render_json(&rows, 0, &metrics, &checks)).unwrap();
            let o = v["overrides"].as_array().expect("an array");
            assert_eq!(o.len(), 1, "{w:?}: {v}");
            assert_eq!(o[0]["flag"], w.flag(), "{v}");
            assert_eq!(o[0]["consequence"], w.consequence(), "{v}");
            assert!(o[0]["rows"].as_array().unwrap().contains(&serde_json::json!("b")), "{v}");
        }
    }

    /// A flag given to a comparison whose check passes waived nothing, and
    /// nothing says it did.
    #[test]
    fn a_flag_given_but_not_needed_lists_nothing() {
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        let checks = checked(&rows);
        assert!(checks.overrides().is_empty(), "{checks:?}");
        let metrics = vec!["elpd".to_string()];
        for text in [render_table(&rows, 0, &metrics, &checks), render_md(&rows, 0, &metrics, &checks)] {
            assert!(!text.contains("was set"), "{text}");
            for w in ALL_WAIVERS {
                assert!(!text.contains(w.flag()), "{text}");
            }
        }
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checks)).unwrap();
        assert_eq!(v["overrides"], serde_json::json!([]), "an empty array, not null: {v}");
    }

    /// The renderers state the preflight's outcome and recompute nothing: a
    /// JSON document for a mismatched-horizon comparison says why its Δ
    /// fields are null, and rows whose data differ, rendered with an outcome
    /// that records no mismatch, are not reported as mismatched by any format.
    #[test]
    fn the_renderers_state_the_preflight_outcome_and_recompute_no_check() {
        let metrics = vec!["elpd".to_string()];
        let rows = rows_needing(Waiver::MismatchedHorizon);
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert!(v["rows"][1]["delta_elpd"].is_null(), "{v}");
        assert!(v["overrides"][0]["consequence"].as_str()
                .is_some_and(|s| s.contains("null in the JSON")),
            "the document says why its Δ fields are null: {v}");

        let rows = rows_needing(Waiver::DataMismatch);
        let silent = Preflight::default();
        let table = render_table(&rows, 0, &metrics, &silent);
        let md = render_md(&rows, 0, &metrics, &silent);
        assert!(!table.contains("different observed data") && !md.contains("different observed data"),
            "a renderer that re-ran the data check would print it:\n{table}\n{md}");
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &silent)).unwrap();
        assert!(v["data_mismatch"].is_null() && v["overrides"] == serde_json::json!([]), "{v}");
    }

    /// Two single-stream traces as `prequential.json` writes them (schema 4,
    /// the step-level summed fields alongside the stream's own, equal for one
    /// stream with no tied PIT).
    fn single_stream_rows() -> Vec<Row> {
        let trace = |scores: [(f64, f64, f64); 4]| -> PrequentialTrace {
            let steps: Vec<String> = scores.iter().enumerate().map(|(k, (ls, crps, pit))| {
                let t = 7.0 * (k + 1) as f64;
                format!(r#"{{"t": {t}, "y_obs": 10.0, "y_pred_samples": [], "log_score": {ls},
                    "crps": {crps}, "pit": {pit}, "ess": 900.0,
                    "per_stream": [{{"stream": "cases", "y_obs": 10.0, "y_pred_samples": [],
                        "log_score": {ls}, "crps": {crps}, "pit": {pit},
                        "coverage": {{"kind": "interval", "start": {s}, "stop": {t}}}}}]}}"#,
                    s = t - 7.0)
            }).collect();
            serde_json::from_str(&format!(r#"{{"schema_version": 4, "t0": 0,
                "provenance": "plug_in", "conditioning": "in_sample",
                "steps": [{}], "warnings": []}}"#, steps.join(","))).unwrap()
        };
        let row = |name: &str, trace: PrequentialTrace| Row {
            name: name.into(), path: name.into(),
            facts: ScoreFacts::of_trace(&trace), trace,
            data: DataIdentity::Unknown("a test fixture".into()),
            streams: RowStreams::Explicit,
        };
        vec![
            row("a", trace([(-3.0, 2.5, 0.40), (-4.0, 3.0, 0.97), (-2.5, 1.5, 0.55), (-3.5, 2.0, 0.30)])),
            row("b", trace([(-2.0, 1.0, 0.60), (-3.0, 2.0, 0.45), (-2.0, 1.25, 0.02), (-3.0, 1.5, 0.70)])),
        ]
    }

    /// The rendering of [`single_stream_rows`] before CRPS and calibration
    /// became per-stream, captured from the code at 961a6a0a.
    const SINGLE_STREAM_TABLE: &str = r#"Model   T_score     elpd   Δelpd       LR   se(Δ)               evidence    crps    Δcrps   PIT_cov90
─────────────────────────────────────────────────────────────────────────────────────────────────────
a             4   -13.00       —        —       —                      —   2.250        —        0.75
b             4   -10.00   +3.00   20.086    0.65   +13.0 dB, strong for   1.438   -0.812        0.75

Scored steps: 4 (t0=0).  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
ⓘ not checked for a shared observation set: 'a' (a test fixture); 'b' (a test fixture). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b +0.25
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const SINGLE_STREAM_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence | crps | Δcrps | PIT_cov90 |
|---|---|---|---|---|---|---|---|---|---|
| a | 4 | -13.00 | — | — | — | — | 2.250 | — | 0.75 |
| b | 4 | -10.00 | +3.00 | 20.086 | 0.65 | +13.0 dB, strong for | 1.438 | -0.812 | 0.75 |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⓘ not checked for a shared observation set: 'a' (a test fixture); 'b' (a test fixture). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b +0.25
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;

    /// D14: when every row scored exactly one stream, and the same one, the
    /// table keeps its `crps`, `Δcrps` and `PIT_cov90` columns, with the same
    /// numbers as before CRPS and calibration became per-stream, and the JSON
    /// row keeps its row-level fields alongside the per-stream entry.
    #[test]
    fn a_single_stream_comparison_renders_as_before() {
        let rows = single_stream_rows();
        let metrics = vec!["elpd".to_string(), "crps".to_string(), "pit_cov90".to_string()];
        let checks = checked(&rows);
        assert_eq!(render_table(&rows, 0, &metrics, &checks), SINGLE_STREAM_TABLE);
        assert_eq!(render_md(&rows, 0, &metrics, &checks), SINGLE_STREAM_MD);
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checks)).unwrap();
        let (a, b) = (&v["rows"][0], &v["rows"][1]);
        assert_eq!((a["mean_crps"].as_f64(), b["mean_crps"].as_f64()), (Some(2.25), Some(1.4375)));
        assert_eq!(b["delta_mean_crps"].as_f64(), Some(-0.8125));
        assert!(a["delta_mean_crps"].is_null());
        assert_eq!((a["pit_cov90"].as_f64(), b["pit_cov90"].as_f64()), (Some(0.75), Some(0.75)));
        assert_eq!(b["per_stream"], serde_json::json!([{
            "stream": "cases", "n": 4, "mean_crps": 1.4375, "crps_nonfinite": 0,
            "delta_mean_crps": -0.8125, "pit_cov90": 0.75, "pit_nonfinite": 0,
            "pit_cov90_warning": null,
        }]), "{v}");
    }

    /// Two rows scoring `north` and `south`, read back through a
    /// `prequential.json` whose steps also carry an older trace's summed
    /// fields with sentinels (CRPS 77.7, PIT 0.5) that no output may show: the
    /// summed predictive is not an observed quantity. `north` is calibrated
    /// (PIT 0.5 throughout); `south` is not (PIT 0.99), in row `b` only.
    fn two_stream_rows() -> Vec<Row> {
        let t = [7.0, 14.0, 21.0, 28.0];
        let mut a = row_with_streams("a", &t, &["north", "south"]);
        let mut b = row_with_streams("b", &t, &["north", "south"]);
        for (k, (sa, sb)) in a.trace.steps.iter_mut().zip(&mut b.trace.steps).enumerate() {
            sa.per_stream[0].crps = 2.0;
            sa.per_stream[1].crps = 4.0 + k as f64;
            sb.per_stream[0].crps = 1.0;
            sb.per_stream[1].crps = 3.0;
            sb.per_stream[1].pit = 0.99;
            sb.log_score = -5.0;
        }
        for r in [&mut a, &mut b] {
            let mut v = serde_json::to_value(&r.trace).unwrap();
            for step in v["steps"].as_array_mut().unwrap() {
                step["crps"] = serde_json::json!(77.7);
                step["pit"] = serde_json::json!(0.5);
                step["y_obs"] = serde_json::json!(77.7);
            }
            r.trace = serde_json::from_value(v).expect("the older on-disk form reads");
        }
        vec![a, b]
    }

    /// D14: a comparison of multi-stream fits drops the CRPS and PIT columns
    /// from the main table and reports them per stream × model, in every
    /// format, with the calibration warning naming the stream — and no output
    /// carries a figure that combines the streams.
    #[test]
    fn a_multi_stream_comparison_reports_accuracy_per_stream_only() {
        let rows = two_stream_rows();
        let metrics = vec!["elpd".to_string(), "crps".to_string(), "pit_cov90".to_string()];
        let checks = checked(&rows);
        let table = render_table(&rows, 0, &metrics, &checks);
        let md = render_md(&rows, 0, &metrics, &checks);
        let json = render_json(&rows, 0, &metrics, &checks);

        for (fmt, text) in [("table", &table), ("md", &md)] {
            let header = text.lines().next().unwrap();
            assert!(!header.contains("crps") && !header.contains("PIT_cov90"),
                "the {fmt} main table drops the summed columns:\n{text}");
            assert!(text.contains("Per-stream accuracy and calibration"), "{fmt}:\n{text}");
            assert!(!text.contains("77.7"),
                "no {fmt} figure combines the streams:\n{text}");
            let line = |stream: &str, model: &str| -> String {
                text.lines()
                    .find(|l| {
                        let cells: Vec<&str> = l.split(['|', ' ']).filter(|c| !c.is_empty()).collect();
                        cells.first() == Some(&stream) && cells.get(1) == Some(&model)
                    })
                    .unwrap_or_else(|| panic!("no {stream} × {model} line in {fmt}:\n{text}"))
                    .to_string()
            };
            let cells = |l: String| -> Vec<String> {
                l.split(['|', ' ']).filter(|c| !c.is_empty()).map(String::from).collect()
            };
            // stream, model, n, mean CRPS, Δ mean CRPS, PIT_cov90
            assert_eq!(cells(line("north", "a")), ["north", "a", "4", "2.000", "—", "1.00"], "{fmt}");
            assert_eq!(cells(line("north", "b")), ["north", "b", "4", "1.000", "-1.000", "1.00"], "{fmt}");
            assert_eq!(cells(line("south", "a")), ["south", "a", "4", "5.500", "—", "1.00"], "{fmt}");
            assert_eq!(cells(line("south", "b")), ["south", "b", "4", "3.000", "-2.500", "0.00"], "{fmt}");
            let warn: Vec<&str> = text.lines().filter(|l| l.contains("PIT 90%-coverage")).collect();
            assert_eq!(warn.len(), 1, "one stream is miscalibrated:\n{text}");
            assert!(warn[0].contains("'b'") && warn[0].contains("'south'"),
                "and the warning names it: {}", warn[0]);
        }

        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(!json.contains("77.7"), "{json}");
        for r in v["rows"].as_array().unwrap() {
            for k in ["mean_crps", "delta_mean_crps", "pit_cov90", "pit_cov90_warning"] {
                assert!(r[k].is_null(), "no row-level `{k}` for a multi-stream row: {r}");
            }
        }
        let south_b = &v["rows"][1]["per_stream"][1];
        assert_eq!(south_b["stream"], "south");
        assert_eq!(south_b["n"], 4);
        assert_eq!(south_b["mean_crps"].as_f64(), Some(3.0));
        assert_eq!(south_b["delta_mean_crps"].as_f64(), Some(-2.5));
        assert_eq!(south_b["pit_cov90"].as_f64(), Some(0.0));
        assert!(south_b["pit_cov90_warning"].as_str().is_some_and(|w| w.contains("'south'")), "{v}");
        assert!(v["rows"][1]["per_stream"][0]["pit_cov90_warning"].is_null(), "north is calibrated: {v}");
        assert!(v["rows"][0]["per_stream"][0]["delta_mean_crps"].is_null(), "the baseline: {v}");
    }

    /// The Δ mean CRPS of a stream is formed only over the same times on both
    /// sides; a stream the two rows scored at different times gets none.
    #[test]
    fn a_stream_scored_at_different_times_gets_no_delta_crps() {
        let mut rows = two_stream_rows();
        // `b` loses its `south` score at t = 14 (a hole in that series).
        rows[1].trace.steps[1].per_stream.truncate(1);
        let acc = row_accuracy(&rows[1], &rows[0]);
        let Accuracy::PerStream(streams) = acc else { panic!("{acc:?}") };
        let south = streams.iter().find(|s| s.stream == "south").unwrap();
        assert_eq!(south.mean_crps.n, 3);
        assert_eq!(south.delta_mean_crps, None);
        let north = streams.iter().find(|s| s.stream == "north").unwrap();
        assert_eq!(north.delta_mean_crps, Some(-1.0));
    }

    /// A trace without a per-stream breakdown (schema 1) still compares on
    /// elpd; its CRPS and calibration are reported as not available, with the
    /// reason, never as a number.
    #[test]
    fn a_trace_without_a_stream_breakdown_reports_no_crps_or_calibration() {
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        let metrics = vec!["elpd".to_string(), "crps".to_string(), "pit_cov90".to_string()];
        let checks = checked(&rows);
        for text in [render_table(&rows, 0, &metrics, &checks), render_md(&rows, 0, &metrics, &checks)] {
            assert!(text.contains("-18.00") && text.contains("+1.00"), "elpd still compares:\n{text}");
            assert!(!text.lines().next().unwrap().contains("crps"), "{text}");
            assert!(text.contains("trace predates the per-stream breakdown"), "{text}");
            assert!(!text.contains("1.000"), "the step-level CRPS is not shown:\n{text}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checks)).unwrap();
        for r in v["rows"].as_array().unwrap() {
            assert!(r["mean_crps"].is_null() && r["pit_cov90"].is_null(), "{r}");
            assert_eq!(r["per_stream"], serde_json::json!([]));
            assert_eq!(r["per_stream_unavailable"], "trace predates the per-stream breakdown");
        }
    }

    fn digests(entries: &[(&str, &str)]) -> DataIdentity {
        DataIdentity::Digests(RecordedData {
            training: entries.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            ..Default::default()
        })
    }

    /// A trace with joint scores at the given times and no per-stream breakdown.
    fn row_at(name: &str, times: &[f64], log_scores: &[f64]) -> Row {
        use score::trace::{Conditioning, PrequentialStep, Provenance};
        let steps = times.iter().zip(log_scores).map(|(&t, &ls)| PrequentialStep {
            t,
            log_score: ls,
            ess: 900.0,
            per_stream: Vec::new(),
        }).collect();
        let trace = PrequentialTrace {
            schema_version: 1,
            t0: 1,
            provenance: Provenance::PlugIn,
            basis: score::trace::PredictiveBasis::Filtered,
            conditioning: Conditioning::InSample,
            steps,
            warnings: Vec::new(),
            score_from: None,
            pit_randomization_seed: None,
            derivation: None, scored_streams: None,
        };
        Row {
            name: name.to_string(),
            path: name.to_string(),
            streams: RowStreams::Explicit,
            data: DataIdentity::Unknown("a test fixture, built from scores alone".into()),
            facts: ScoreFacts::of_trace(&trace),
            trace,
        }
    }

    /// A trace scoring the named streams at every one of `times`.
    fn row_with_streams(name: &str, times: &[f64], streams: &[&str]) -> Row {
        use score::trace::StreamScore;
        let log_scores = vec![-6.0; times.len()];
        let mut row = row_at(name, times, &log_scores);
        for step in &mut row.trace.steps {
            step.per_stream = streams.iter().map(|s| StreamScore {
                stream: (*s).to_string(),
                y_obs: 5.0,
                y_pred_samples: Vec::new(),
                log_score: -3.0,
                crps: 0.5,
                pit: 0.5,
                interval: Default::default(),
                coverage: Default::default(),
                wis: None,
                in_50: None,
            }).collect();
        }
        row
    }

    /// A trace scoring one stream at every one of `times`, recording what each
    /// value covered (`coverage[k]` at `times[k]`).
    fn row_covering(
        name: &str,
        times: &[f64],
        stream: &str,
        coverage: &[score::trace::Coverage],
    ) -> Row {
        assert_eq!(times.len(), coverage.len());
        let mut row = row_with_streams(name, times, &[stream]);
        for (step, c) in row.trace.steps.iter_mut().zip(coverage) {
            step.per_stream[0].coverage = *c;
        }
        row
    }

    /// gh#833. Equal times and equal stream sets still permit two traces to
    /// have scored DIFFERENT QUANTITIES: the same file read with its windows
    /// gapped in one fit and merged in the other, or a stream read as an
    /// instant in one and as a flow in the other. The times agree, the names
    /// agree, the data hashes agree; only what each value covered differs, and
    /// Δelpd would report the difference as evidence about the models.
    #[test]
    fn refuses_traces_that_scored_the_same_stream_over_different_windows() {
        use score::trace::Coverage;
        let iv = |s: f64, e: f64| Coverage::Interval { start: s, stop: e };
        // Rows closing at 4, 6, 7. Gapped: [5,6) with (4,5) covered by
        // nothing. Merged: [4,6) as one two-day row. Same stops, same count of
        // scored steps — the ambiguity a missing row used to hide.
        let t = [4.0, 6.0, 7.0];
        let gapped = [iv(3.0, 4.0), iv(5.0, 6.0), iv(6.0, 7.0)];
        let merged = [iv(3.0, 4.0), iv(4.0, 6.0), iv(6.0, 7.0)];

        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_covering("b", &t, "cases", &gapped);
        axis(&[a, b]).expect("identical windows compare");

        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_covering("b", &t, "cases", &merged);
        let err = axis(&[a, b])
            .expect_err("a gapped reading may not be differenced against a merged one");
        assert!(err.contains("[5, 6)") && err.contains("[4, 6)"),
            "names both windows: {err}");
        assert!(err.contains("'a'") && err.contains("'b'") && err.contains("'cases'"),
            "names both traces and the stream: {err}");
        assert!(err.contains("t=6"), "and the step where they diverge: {err}");
        assert!(err.contains("covers"), "and points at the declaration: {err}");

        // The same name read as an instant in one trace and a flow in the
        // other is not the same quantity either.
        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_covering("b", &t, "cases", &[Coverage::Instant; 3]);
        let err = axis(&[a, b])
            .expect_err("an instant reading is not a flow");
        assert!(err.contains("instant") && err.contains("[3, 4)"), "{err}");

        // Two traces that never recorded coverage carry nothing to check.
        let a = row_with_streams("a", &t, &["cases"]);
        let b = row_with_streams("b", &t, &["cases"]);
        axis(&[a, b])
            .expect("two traces with no coverage record are not a mismatch");

        // One recorded, one not: unverifiable, and the message says which
        // side to re-derive rather than inventing a window mismatch.
        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_with_streams("b", &t, &["cases"]);
        let err = axis(&[a, b])
            .expect_err("an unverifiable window is not a verified one");
        assert!(err.contains("does not record what its rows cover") && err.contains("Re-derive 'b'"),
            "{err}");
    }

    /// gh#241 G3: `deny_unknown_fields` — a typo'd compare.toml key must ERROR.
    #[test]
    fn compare_toml_rejects_unknown_keys() {
        let ok = "baseline = \"a\"\n[[model]]\nname = \"a\"\npath = \"p\"\n";
        assert!(toml::from_str::<CompareToml>(ok).is_ok(), "valid compare.toml must parse");

        let bad_top = "baselne = \"a\"\n[[model]]\nname = \"a\"\npath = \"p\"\n"; // typo: baselne
        assert!(
            toml::from_str::<CompareToml>(bad_top).is_err(),
            "a typo'd top-level key must be rejected"
        );

        let bad_model = "[[model]]\nname = \"a\"\npath = \"p\"\nlabel = \"x\"\n"; // unknown: label
        assert!(
            toml::from_str::<CompareToml>(bad_model).is_err(),
            "an unknown [[model]] key must be rejected"
        );
    }

    /// gh#806. Naming a positional row by its basename renders
    /// `a/fit.toml b/fit.toml` as two rows both called `fit.toml` — the
    /// per-variant directory layout produces exactly that, so it is the common
    /// arrangement, not a corner case. Colliding names grow leftward until
    /// distinct; names that already differ are untouched.
    #[test]
    fn positional_names_grow_only_until_the_collision_is_resolved() {
        let names = |ps: &[&str]| positional_model_names(
            &ps.iter().map(|s| s.to_string()).collect::<Vec<_>>());

        // The collision: one parent component separates them.
        assert_eq!(names(&["a/fit.toml", "b/fit.toml"]),
            vec!["a/fit.toml", "b/fit.toml"]);

        // Distinct basenames render as before — the whole path is not shown
        // just because some other cohort needed it.
        assert_eq!(names(&["a/ctl_rm.toml", "b/ctl_bb.toml"]),
            vec!["ctl_rm.toml", "ctl_bb.toml"]);

        // Only the colliding pair grows.
        assert_eq!(names(&["x/fit.toml", "y/fit.toml", "z/other.toml"]),
            vec!["x/fit.toml", "y/fit.toml", "other.toml"]);

        // Growth stops at the first component that separates them: the shared
        // `runs/` prefix above it is never shown.
        assert_eq!(names(&["runs/a/fit.toml", "runs/b/fit.toml"]),
            vec!["a/fit.toml", "b/fit.toml"]);

        // When the distinguishing component is further up, it keeps going.
        assert_eq!(names(&["v1/run/fit.toml", "v2/run/fit.toml"]),
            vec!["v1/run/fit.toml", "v2/run/fit.toml"]);

        // A handle or bare name has no parent to grow into and is left alone.
        assert_eq!(names(&["@a", "@b"]), vec!["@a", "@b"]);
    }

    /// The disambiguation runs at name-assignment time, BEFORE
    /// `parse_cohort_exclude` reads the names — so a per-fit
    /// `--exclude-chains NAME:IDS` token matches the name the table shows.
    /// Assigning names after parsing would have made the grown rows
    /// unaddressable.
    #[test]
    fn a_disambiguated_name_is_the_name_exclude_chains_matches() {
        let names = positional_model_names(&[
            "a/fit.toml".to_string(), "b/fit.toml".to_string()]);
        let c = parse_cohort_selection(&[], &["a/fit.toml:3".to_string()], &names).unwrap();
        assert_eq!(c.for_fit("a/fit.toml").unwrap().ids_csv(), "3");
        assert!(c.for_fit("b/fit.toml").is_none(), "the other fit keeps all chains");

        // The bare basename is no longer a fit name, and says so.
        let e = parse_cohort_selection(&[], &["fit.toml:3".to_string()], &names).unwrap_err();
        assert!(e.contains("no compared fit named 'fit.toml'")
            && e.contains("a/fit.toml"),
            "the error names the fits as displayed: {e}");
    }

    /// gh#418: per-fit `@fit:ids` binds to one fit; bare ids are cohort-wide.
    #[test]
    fn parse_cohort_exclude_forms() {
        // Fit names are matched VERBATIM. A handle-referenced fit carries the
        // `@` in its name (`@a`); a path-referenced one does not (`ctl_rm.toml`).
        let names = vec!["@a".to_string(), "@b".to_string()];

        // Empty → no filtering.
        assert!(parse_cohort_selection(&[], &[], &names).unwrap().per_fit.is_empty());

        // Bare ids → cohort-wide: every fit gets the same drop set.
        let c = parse_cohort_selection(&[], &["3,4".to_string()], &names).unwrap();
        assert!(c.cohort_wide);
        assert_eq!(c.for_fit("@a").unwrap().ids_csv(), "3,4");
        assert_eq!(c.for_fit("@b").unwrap().ids_csv(), "3,4");

        // Per-fit → ONLY the named fit is filtered; the other keeps all chains.
        let c = parse_cohort_selection(&[], &["@a:2".to_string()], &names).unwrap();
        assert!(!c.cohort_wide);
        assert_eq!(c.for_fit("@a").unwrap().ids_csv(), "2");
        assert!(c.for_fit("@b").is_none(), "@b keeps all chains");

        // A path-named fit (no `@`) is matched by its bare name.
        let paths = vec!["ctl_rm.toml".to_string(), "ctl_bb.toml".to_string()];
        let c = parse_cohort_selection(&[], &["ctl_rm.toml:4".to_string()], &paths).unwrap();
        assert_eq!(c.for_fit("ctl_rm.toml").unwrap().ids_csv(), "4");
        assert!(c.for_fit("ctl_bb.toml").is_none());

        // Multiple per-fit tokens, each its own drop set.
        let c =
            parse_cohort_selection(&[], &["@a:2".to_string(), "@b:5,6".to_string()], &names).unwrap();
        assert_eq!(c.for_fit("@a").unwrap().ids_csv(), "2");
        assert_eq!(c.for_fit("@b").unwrap().ids_csv(), "5,6");
    }

    #[test]
    fn parse_cohort_exclude_hints_spurious_at_sigil() {
        // A path-named fit targeted with a spurious `@` (the common slip garki
        // hit): the error names the fix rather than a bare "no such fit".
        let names = vec!["ctl_rm.toml".to_string(), "ctl_bb.toml".to_string()];
        let e = parse_cohort_selection(&[], &["@ctl_rm.toml:4".to_string()], &names).unwrap_err();
        assert!(e.contains("did you mean 'ctl_rm.toml'"), "{e}");
    }

    /// A per-fit token that cannot bind to exactly one model is a hard error —
    /// never a silent no-op or a wrong-fit bind.
    #[test]
    fn parse_cohort_exclude_rejects_bad_forms() {
        let names = vec!["@a".to_string(), "@b".to_string()];

        // Unknown fit name → error naming the available fits (verbatim, so the
        // message names `@z` — and there is no `z` to hint, so no "did you mean").
        let e = parse_cohort_selection(&[], &["@z:3".to_string()], &names).unwrap_err();
        assert!(e.contains("no compared fit named '@z'") && e.contains("@a"), "{e}");
        assert!(!e.contains("did you mean"), "no hint when the stripped name also misses: {e}");

        // Mixing bare (cohort) and per-fit tokens → rejected as ambiguous.
        assert!(parse_cohort_selection(&[], &["3".to_string(), "@a:4".to_string()], &names).is_err());

        // Ambiguous name (two compared models share it) → rejected.
        let dup = vec!["scout".to_string(), "scout".to_string()];
        let e = parse_cohort_selection(&[], &["scout:3".to_string()], &dup).unwrap_err();
        assert!(e.contains("ambiguous"), "{e}");

        // Same fit listed twice → rejected.
        assert!(
            parse_cohort_selection(&[], &["@a:2".to_string(), "@a:3".to_string()], &names).is_err()
        );

        // Missing fit name before ':' → rejected.
        assert!(parse_cohort_selection(&[], &[":3".to_string()], &names).is_err());
    }

    /// `--chains` and `--exclude-chains` together: different fits are fine,
    /// one fit in both is refused, and a bare value in either is refused when
    /// either has a per-fit value.
    #[test]
    fn compare_chain_flag_combinations() {
        use crate::chain_selection::Polarity;
        let names = vec!["@a".to_string(), "@b".to_string()];
        let v = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<_>>();

        let c = parse_cohort_selection(&v(&["@a:0,1"]), &v(&["@b:4"]), &names).unwrap();
        assert_eq!(c.for_fit("@a").unwrap().polarity(), Polarity::Keep);
        assert_eq!(c.for_fit("@a").unwrap().ids_csv(), "0,1");
        assert_eq!(c.for_fit("@b").unwrap().polarity(), Polarity::Drop);
        assert_eq!(c.for_fit("@b").unwrap().ids_csv(), "4");

        let e = parse_cohort_selection(&v(&["@a:0,1"]), &v(&["@a:4"]), &names).unwrap_err();
        assert!(e.contains("named by both --chains and --exclude-chains"), "{e}");

        for (keep, drop) in [(v(&["0,1"]), v(&["@b:4"])), (v(&["@a:0"]), v(&["4"]))] {
            let e = parse_cohort_selection(&keep, &drop, &names).unwrap_err();
            assert!(e.contains("mixing cohort-wide ids and per-fit"), "{keep:?} {drop:?}: {e}");
        }

        let e = parse_cohort_selection(&v(&["0,1"]), &v(&["4"]), &names).unwrap_err();
        assert!(e.contains("both cohort-wide"), "{e}");

        // Cohort-wide keep applies to every fit.
        let c = parse_cohort_selection(&v(&["0,1"]), &[], &names).unwrap();
        assert!(c.cohort_wide);
        assert_eq!(c.for_fit("@b").unwrap().polarity(), Polarity::Keep);

        // A repeated id within one value is refused, named by fit.
        let e = parse_cohort_selection(&v(&["@a:1,1"]), &[], &names).unwrap_err();
        assert!(e.contains("--chains @a lists chain 1 twice"), "{e}");
    }

    // ── --streams ───────────────────────────────────────────────────────

    /// A model declaring these (leaf, source) observation streams: a golden
    /// model's first stream copied under each name, which is all
    /// [`StreamBindings::resolve`] reads of it.
    fn model_declaring(declared: &[(&str, &str)]) -> ir::Model {
        let path = format!("{}/../../../ocaml/golden/seir_observations.ir.json",
            env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let mut model: ir::Model = serde_json::from_value(v["model"].clone()).unwrap();
        let template = model.observations[0].clone();
        model.observations = declared.iter().map(|(l, s)| {
            let mut o = template.clone();
            o.name = l.to_string();
            o.source = s.to_string();
            o
        }).collect();
        model
    }

    /// A derived row whose model declares these (leaf, source) pairs and whose
    /// fit binds the sources of the leaves it scores, at 7, 14, 21.
    fn derived_row(name: &str, declared: &[(&str, &str)], scored: &[&str]) -> Row {
        let mut r = row_with_streams(name, &[7.0, 14.0, 21.0], scored);
        let mut data = String::from("[observations]\n");
        let mut bound: Vec<&str> = Vec::new();
        for (l, s) in declared {
            if scored.contains(l) && !bound.contains(s) {
                bound.push(s);
                data.push_str(&format!("{s} = \"{s}.tsv\"\n"));
            }
        }
        let spec: crate::fit::config_v2::DataSpec = toml::from_str(&data).unwrap();
        r.streams = RowStreams::Derived {
            declared: declared.iter().map(|(l, s)| (l.to_string(), s.to_string())).collect(),
            bindings: StreamBindings::resolve(&model_declaring(declared), &spec).unwrap(),
        };
        r
    }

    const PATCHES: [(&str, &str); 3] =
        [("cases_south", "cases"), ("cases_north", "cases"), ("deaths", "deaths")];

    /// A `--streams` name is a leaf, or a source standing for its leaves in
    /// the model's order; it must resolve in every row, and the refusal lists
    /// what the row offers. An explicit trace records leaves only.
    #[test]
    fn streams_names_resolve_to_leaves_in_every_row() {
        let full = derived_row("full", &PATCHES, &["cases_south", "cases_north", "deaths"]);
        let cases_only = derived_row("cases_only", &PATCHES, &["cases_south", "cases_north"]);
        let rows = [full.clone(), cases_only.clone()];
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();

        assert_eq!(resolve_selection(&names(&["cases"]), &rows).unwrap(),
            ["cases_south", "cases_north"], "a source selects its leaves in the model's order");
        assert_eq!(resolve_selection(&names(&["cases_north", "cases"]), &rows).unwrap(),
            ["cases_north", "cases_south"], "names in the order given, each leaf once");
        assert_eq!(resolve_selection(&names(&["deaths"]), &rows).unwrap(), ["deaths"],
            "a name the model declares resolves even where the fit did not bind it");

        let err = resolve_selection(&names(&["hospital"]), &rows).unwrap_err();
        assert!(err.contains("'hospital'") && err.contains("'full'")
                && err.contains("cases_south, cases_north, deaths")
                && err.contains("sources: cases, deaths"), "{err}");

        // An explicit trace names the leaves it scored and nothing more.
        let mut explicit = row_with_streams("pq", &[7.0, 14.0, 21.0], &["cases_south", "cases_north"]);
        explicit.streams = RowStreams::Explicit;
        let rows = [full, explicit];
        assert_eq!(resolve_selection(&names(&["cases_north"]), &rows).unwrap(), ["cases_north"]);
        let err = resolve_selection(&names(&["cases"]), &rows).unwrap_err();
        assert!(err.contains("'pq'") && err.contains("explicit prequential.json")
                && err.contains("cases_south, cases_north"), "{err}");
        assert!(resolve_selection(&[], &rows).is_err());

        // Models that expand a source differently contribute the union.
        let other = derived_row("other", &[("cases", "cases")], &["cases"]);
        let rows = [cases_only, other];
        assert_eq!(resolve_selection(&names(&["cases"]), &rows).unwrap(),
            ["cases_south", "cases_north", "cases"]);
    }

    /// A row re-scored on one leaf is that leaf's projection; a row that does
    /// not score the leaf is refused, naming the row and the leaf.
    #[test]
    fn a_row_projected_onto_a_leaf_it_never_scored_is_refused_by_name() {
        let r = derived_row("cases_only", &PATCHES, &["cases_south", "cases_north"]);
        let err = r.onto("deaths").map(|_| ()).unwrap_err();
        assert!(err.starts_with("'cases_only': ") && err.contains("'deaths'"), "{err}");
        let p = r.onto("cases_north").unwrap();
        assert_eq!(p.trace.streams(), ["cases_north"]);
        assert_eq!(p.trace.elpd(), -9.0, "three steps of the leaf's own -3");
        assert_eq!(p.name, "cases_only");
        assert_eq!(for_stream("deaths", "error: a refusal".into()),
            "error: stream 'deaths': a refusal");
    }

    /// The per-leaf filter-noise SE of a replicated derive equals the
    /// log-mean-exp SE of the replicates' own projected totals — computed
    /// here by hand from each replicate trace — not the joint SE.
    #[test]
    fn the_per_leaf_filter_noise_se_is_that_of_the_projected_replicates() {
        let t = [7.0, 14.0, 21.0];
        let reps: Vec<PrequentialTrace> = [(-3.0, -1.0), (-3.5, -2.0), (-2.5, -1.5)].iter()
            .map(|&(north, south)| {
                let mut tr = row_with_streams("r", &t, &["north", "south"]).trace;
                for (k, s) in tr.steps.iter_mut().enumerate() {
                    s.per_stream[0].log_score = north - k as f64 * 0.25;
                    s.per_stream[1].log_score = south * (k + 1) as f64;
                    s.log_score = s.per_stream[0].log_score + s.per_stream[1].log_score - 0.1;
                }
                tr
            }).collect();
        // What a replicated compare records on the combined trace.
        let totals = score::combine::ReplicateTotals::of(&reps);
        let mut combined = stored::combine_replicates(&reps).unwrap();
        let mut d = derivation(100, reps.len());
        d.replicate_elpd = totals.elpd.clone();
        d.replicate_elpd_by_stream = totals.by_stream.clone();
        d.mc_se_elpd = totals.mc_se;
        combined.derivation = Some(d);
        let mut row = row_with_streams("a", &t, &[]);
        row.trace = combined;

        for leaf in ["north", "south"] {
            let by_hand: Vec<f64> = reps.iter()
                .map(|rep| score::trace::project_onto_stream(rep, leaf).unwrap().elpd())
                .collect();
            let want = numerics::logmeanexp_with_se(&by_hand).1;
            let got = row_mc_se(&row.onto(leaf).unwrap());
            assert_eq!(got, Some(want), "{leaf}");
            assert_ne!(got, totals.mc_se, "{leaf}: not the joint SE");
        }
    }

    /// A projected trace and a joint trace are refused as a pair even where
    /// their steps agree: one step's log score is a leaf's own, the other's
    /// the joint score over what its fit observed.
    #[test]
    fn a_projected_trace_is_not_compared_with_a_joint_one() {
        let joint = row_with_streams("joint", &[7.0, 14.0, 21.0], &["cases"]);
        let projected = joint.onto("cases").unwrap();
        let mut projected = Row { name: "projected".into(), ..projected };
        projected.trace.steps.iter_mut().for_each(|s| s.log_score = -6.0);
        let err = axis(&[joint.clone(), projected.clone()]).unwrap_err();
        assert!(err.contains("'joint' is scored jointly") && err.contains("'projected' is re-scored on cases alone"),
            "{err}");
        // Under mismatched horizons the same check holds at shared times.
        assert!(shared_times(&[joint.clone(), projected]).is_err());
        let both = [joint.onto("cases").unwrap(), joint.onto("cases").unwrap()];
        axis(&both).expect("two projections onto one leaf compare");
    }

    const CASES_DEATHS: [(&str, &str); 2] = [("cases", "cases"), ("deaths", "deaths")];

    /// Two fits of cases and deaths whose `deaths` files differ and whose
    /// `cases` files agree: the joint comparison is refused as fitted to
    /// different data; on `cases` alone only the `cases` hash is compared, the
    /// `deaths` one is named as not compared, and no format reports a mismatch.
    #[test]
    fn a_table_on_one_stream_compares_only_that_streams_data() {
        let mut rows = vec![
            derived_row("a", &CASES_DEATHS, &["cases", "deaths"]),
            derived_row("b", &CASES_DEATHS, &["cases", "deaths"]),
        ];
        rows[0].data = digests(&[("cases", "aaaaaaaa"), ("deaths", "dddddddd")]);
        rows[1].data = digests(&[("cases", "aaaaaaaa"), ("deaths", "eeeeeeee")]);
        let refusal = outcome_of(&rows, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("different observed data") && refusal.contains("'deaths'"),
            "{refusal}");

        let on_cases: Vec<Row> = rows.iter().map(|r| r.onto("cases").unwrap()).collect();
        let checks = outcome_of(&on_cases, &Waivers::default())
            .expect("the cases files agree, and only they are compared");
        assert!(checks.overrides().is_empty() && checks.waived_finding(CheckId::SharedData).is_none(), "{checks:?}");
        let note = data_unchecked_note(&checks, &on_cases).unwrap_or_default();
        assert!(note.contains("'a' stream 'deaths' (not compared: this table scores cases alone)")
                && note.contains("'b' stream 'deaths'"), "the set-aside source is named: {note}");
        let metrics = vec!["elpd".to_string()];
        for text in [render_table(&on_cases, 0, &metrics, &checks),
                     render_md(&on_cases, 0, &metrics, &checks),
                     render_json(&on_cases, 0, &metrics, &checks)] {
            assert!(!text.contains("different observed data") && !text.contains("--allow-data-mismatch"),
                "{text}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&on_cases, 0, &metrics, &checks)).unwrap();
        assert!(v["data_mismatch"].is_null(), "{v}");

        // On `deaths` alone the differing files are compared, and refused.
        let on_deaths: Vec<Row> = rows.iter().map(|r| r.onto("deaths").unwrap()).collect();
        let refusal = outcome_of(&on_deaths, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'deaths'") && refusal.contains("dddddddd"), "{refusal}");
    }

    /// A row that records digests but none for the source the table scores is
    /// refused: an absent hash is not a matching one. The usual waiver renders
    /// it, stated as the override.
    #[test]
    fn a_row_without_a_hash_for_the_scored_stream_is_refused() {
        let mut rows = [
            derived_row("a", &CASES_DEATHS, &["cases", "deaths"]),
            derived_row("b", &CASES_DEATHS, &["cases", "deaths"]),
        ];
        rows[0].data = digests(&[("cases", "aaaaaaaa"), ("deaths", "dddddddd")]);
        rows[1].data = digests(&[("deaths", "dddddddd")]);
        let on_cases: Vec<Row> = rows.iter().map(|r| r.onto("cases").unwrap()).collect();
        let refusal = outcome_of(&on_cases, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'b' records no hash for source 'cases'")
                && refusal.contains("--allow-data-mismatch"), "{refusal}");
        let waived = Waivers { data_mismatch: true, ..Default::default() };
        let checks = outcome_of(&on_cases, &waived).unwrap();
        assert_eq!(checks.overrides().len(), 1);
        assert_eq!(checks.overrides()[0].finding.members(), ["b"]);

        // Both rows lacking it is no better: nothing matched.
        rows[0].data = digests(&[("deaths", "dddddddd")]);
        let on_cases: Vec<Row> = rows.iter().map(|r| r.onto("cases").unwrap()).collect();
        let refusal = outcome_of(&on_cases, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'a' records no hash") && refusal.contains("'b' records no hash"),
            "{refusal}");
    }

    /// Under `--streams`, a row that cannot be projected onto a later leaf is
    /// one more refusal: the refusals of the earlier leaves' tables are
    /// printed with it, each naming its stream.
    #[test]
    fn an_unprojectable_row_is_refused_with_the_other_tables_refusals() {
        let mut full = derived_row("full", &CASES_DEATHS, &["cases", "deaths"]);
        let mut cases_only = derived_row("cases_only", &CASES_DEATHS, &["cases"]);
        full.data = digests(&[("cases", "aaaaaaaa")]);
        cases_only.data = digests(&[("cases", "bbbbbbbb")]);
        let leaves = ["cases".to_string(), "deaths".to_string()];
        let refusals = match stream_tables(&crate::out::Out::new(false), &[full, cases_only],
                &leaves, &Waivers::default(), None) {
            Err(e) => e,
            Ok(_) => panic!("refused"),
        };
        let at = |needle: &str| refusals.iter().position(|m| m.contains(needle))
            .unwrap_or_else(|| panic!("{needle:?} in {refusals:#?}"));
        let data = at("different observed data");
        assert!(refusals[data].starts_with("error: stream 'cases': "), "{refusals:#?}");
        let unprojectable = at("'cases_only': ");
        assert!(refusals[unprojectable].starts_with("error: stream 'deaths': ")
                && refusals[unprojectable].contains("'deaths'"), "{refusals:#?}");
        assert!(data < unprojectable, "in leaf order: {refusals:#?}");
    }

    /// The stream-ablation experiment scored in-sample is biased by
    /// construction, so a `--streams` table whose rows assimilated different
    /// stream sets is refused unless the rows are scored held-out. The refusal
    /// names each row's set and the remedy.
    #[test]
    fn fits_that_assimilated_different_streams_are_refused_in_sample() {
        let full = derived_row("full", &CASES_DEATHS, &["cases", "deaths"]);
        let cases_only = derived_row("cases_only", &CASES_DEATHS, &["cases"]);
        assert_eq!(full.assimilated(), Some(vec!["cases".to_string(), "deaths".to_string()]));
        let on_cases = [full.onto("cases").unwrap(), cases_only.onto("cases").unwrap()];
        let refusal = outcome_of(&on_cases, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("assimilated different streams")
                && refusal.contains("'full' assimilated cases, deaths")
                && refusal.contains("'cases_only' assimilated cases")
                && refusal.contains("holdout_after")
                && refusal.contains("--allow-in-sample-ablation")
                && refusal.contains("fewer"), "{refusal}");

        // The same set: no refusal.
        let twin = derived_row("twin", &CASES_DEATHS, &["cases", "deaths"]);
        outcome_of(&[full.onto("cases").unwrap(), twin.onto("cases").unwrap()], &Waivers::default())
            .expect("rows that assimilated the same streams compare in-sample");

        // Held out: no refusal.
        let held = |r: Row| {
            let mut p = r.onto("cases").unwrap();
            p.trace.conditioning = Conditioning::HoldOutTail {
                train_end: 3.0, theta_source: r.name.clone() };
            p
        };
        let checks = outcome_of(&[held(full.clone()), held(cases_only.clone())], &Waivers::default())
            .expect("held-out rows need no flag");
        assert!(checks.overrides().is_empty(), "{checks:?}");

        // An explicit trace does not record what its fit assimilated, and
        // counts as different even from a row with the same streams.
        let mut explicit = full.clone();
        explicit.name = "pq".into();
        explicit.streams = RowStreams::Explicit;
        assert_eq!(explicit.assimilated(), None);
        let refusal = outcome_of(&[full.onto("cases").unwrap(), explicit.onto("cases").unwrap()],
                                &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'pq' (not recorded: an explicit prequential.json)"), "{refusal}");
        // Two unknown sets are not known to be equal either.
        let mut explicit_too = explicit.clone();
        explicit_too.name = "pq2".into();
        let refusal = outcome_of(&[explicit.onto("cases").unwrap(), explicit_too.onto("cases").unwrap()],
                                &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'pq' (not recorded") && refusal.contains("'pq2' (not recorded"),
            "each unknown set is its own entry: {refusal}");

        // A joint comparison is not an ablation: the check is not made there.
        let twin_joint = [full.clone(), twin.clone()];
        outcome_of(&twin_joint, &Waivers::default()).expect("a joint comparison is untouched");
    }

    /// The JSON document of [`two_stream_rows`], captured from the renderer
    /// before `--streams` changed it. Without `--streams` the document must
    /// stay byte for byte what it was.
    const TWO_STREAM_JSON: &str = r#"{
  "baseline": "a",
  "data_compared_on_prefix": null,
  "data_mismatch": null,
  "data_unchecked": "not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.",
  "evidence_scale_note": "The Jeffreys tiers calibrate Bayes factors; these are in-sample likelihood ratios, not Bayes factors.",
  "filter_noise_unassessed": null,
  "filter_settings_mismatch": null,
  "filter_settings_unchecked": "filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked",
  "metrics": [
    "elpd",
    "crps",
    "pit_cov90"
  ],
  "optimism_caveat": "scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).",
  "overrides": [],
  "rows": [
    {
      "basis": "filtered",
      "conditioning": "in_sample",
      "delta_elpd": null,
      "delta_elpd_db": null,
      "delta_mean_crps": null,
      "derivation": null,
      "elpd": -24.0,
      "evidence_label": null,
      "lag1_autocorr_delta_elpd": null,
      "lr": null,
      "mc_se_delta_elpd": null,
      "mc_se_elpd": null,
      "mean_crps": null,
      "name": "a",
      "path": "a",
      "per_stream": [
        {
          "crps_nonfinite": 0,
          "delta_mean_crps": null,
          "mean_crps": 2.0,
          "n": 4,
          "pit_cov90": 1.0,
          "pit_cov90_warning": null,
          "pit_nonfinite": 0,
          "stream": "north"
        },
        {
          "crps_nonfinite": 0,
          "delta_mean_crps": null,
          "mean_crps": 5.5,
          "n": 4,
          "pit_cov90": 1.0,
          "pit_cov90_warning": null,
          "pit_nonfinite": 0,
          "stream": "south"
        }
      ],
      "per_stream_unavailable": null,
      "pit_cov90": null,
      "pit_cov90_warning": null,
      "provenance": "plug_in",
      "se_delta_elpd": null,
      "t_score": 4,
      "warnings": []
    },
    {
      "basis": "filtered",
      "conditioning": "in_sample",
      "delta_elpd": 4.0,
      "delta_elpd_db": 17.37177927613007,
      "delta_mean_crps": null,
      "derivation": null,
      "elpd": -20.0,
      "evidence_label": "very strong",
      "lag1_autocorr_delta_elpd": null,
      "lr": 54.598150033144236,
      "mc_se_delta_elpd": null,
      "mc_se_elpd": null,
      "mean_crps": null,
      "name": "b",
      "path": "b",
      "per_stream": [
        {
          "crps_nonfinite": 0,
          "delta_mean_crps": -1.0,
          "mean_crps": 1.0,
          "n": 4,
          "pit_cov90": 1.0,
          "pit_cov90_warning": null,
          "pit_nonfinite": 0,
          "stream": "north"
        },
        {
          "crps_nonfinite": 0,
          "delta_mean_crps": -2.5,
          "mean_crps": 3.0,
          "n": 4,
          "pit_cov90": 0.0,
          "pit_cov90_warning": "'b' stream 'south': PIT 90%-coverage 0.00 (nominal 0.90) — likely overconfident.",
          "pit_nonfinite": 0,
          "stream": "south"
        }
      ],
      "per_stream_unavailable": null,
      "pit_cov90": null,
      "pit_cov90_warning": null,
      "provenance": "plug_in",
      "se_delta_elpd": 0.0,
      "t_score": 4,
      "warnings": []
    }
  ],
  "se_caveat": "se(Δ) is a normal approximation, unreliable at small T (here T = 4) and when the models predict alike (Sivula, Magnusson & Vehtari, arXiv:2008.10296) — read it as descriptive, not as a test."
}
"#;

    #[test]
    fn without_streams_the_json_document_is_unchanged() {
        let rows = two_stream_rows();
        let metrics = vec!["elpd".to_string(), "crps".to_string(), "pit_cov90".to_string()];
        assert_eq!(render_json(&rows, 0, &metrics, &checked(&rows)), TWO_STREAM_JSON);
    }

    /// A `--streams` table, rendered: rows of different stream sets re-scored
    /// on `cases`, held out so the comparison needs no waiver.
    fn ablation_table() -> Table {
        let held = |r: Row| {
            let mut p = r.onto("cases").unwrap();
            p.trace.conditioning = Conditioning::HoldOutTail {
                train_end: 3.0, theta_source: r.name.clone() };
            p
        };
        let rows = vec![
            held(derived_row("full", &CASES_DEATHS, &["cases", "deaths"])),
            held(derived_row("cases_only", &CASES_DEATHS, &["cases"])),
        ];
        let checks = outcome_of(&rows, &Waivers::default()).unwrap();
        Table { rows, base_idx: 0, checks }
    }

    /// Every format states the stream a `--streams` table scores and what
    /// each row's fit assimilated; the text footer drops `t0`, which indexes
    /// each fit's own time axis; and nothing in any format combines streams.
    #[test]
    fn every_format_states_the_scored_stream_and_each_rows_assimilated_set() {
        let t = ablation_table();
        let metrics = vec!["elpd".to_string(), "crps".to_string(), "pit_cov90".to_string()];
        let table = render_table(&t.rows, t.base_idx, &metrics, &t.checks);
        let md = render_md(&t.rows, t.base_idx, &metrics, &t.checks);
        assert!(table.starts_with("Stream: cases — every row scored on this stream alone.\n\n"),
            "{table}");
        assert!(md.starts_with("### Stream `cases`\n\n_Every row scored on this stream alone._\n\n"),
            "{md}");
        for (fmt, text) in [("table", &table), ("md", &md)] {
            assert!(text.contains("Assimilated: 'full' assimilated cases, deaths; \
                                   'cases_only' assimilated cases."), "{fmt}:\n{text}");
            assert!(text.contains("crps") && text.contains("PIT_cov90"),
                "a one-stream table keeps its accuracy columns:\n{text}");
        }
        assert!(table.contains("Scored steps: 3.  Baseline: full.\n") && !table.contains("t0="),
            "{table}");

        // Rows that assimilated the same set: no list.
        let same = vec![derived_row("a", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap(),
                        derived_row("b", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap()];
        let checks = outcome_of(&same, &Waivers::default()).unwrap();
        assert!(!render_table(&same, 0, &metrics, &checks).contains("Assimilated:"));

        let doc = json_document(&t.rows, t.base_idx, &metrics, &t.checks);
        assert_eq!(doc["stream"], "cases", "{doc}");
        assert_eq!(doc["rows"][0]["streams_assimilated"], serde_json::json!(["cases", "deaths"]));
        assert_eq!(doc["rows"][1]["streams_assimilated"], serde_json::json!(["cases"]));
        let mut explicit = t.rows[1].clone();
        explicit.streams = RowStreams::Explicit;
        let rows = vec![t.rows[0].clone(), explicit];
        let doc = json_document(&rows, 0, &metrics, &t.checks);
        assert!(doc["rows"][1]["streams_assimilated"].is_null(), "unknown is null: {doc}");

        // The command's document: the selection, and one table per stream, and
        // nothing beside them.
        let whole: serde_json::Value = serde_json::from_str(
            &render_json_tables(&[ablation_table(), ablation_table()], Some(&["cases".into(), "deaths".into()]), &metrics)).unwrap();
        let keys: Vec<&String> = whole.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["streams_scored", "tables"], "{whole}");
        assert_eq!(whole["streams_scored"], serde_json::json!(["cases", "deaths"]));
        assert_eq!(whole["tables"].as_array().unwrap().len(), 2);
        assert_eq!(whole["tables"][0]["stream"], "cases");

        // Without --streams none of this appears.
        let rows = two_stream_rows();
        let doc = json_document(&rows, 0, &metrics, &checked(&rows));
        assert!(doc.get("stream").is_none() && doc["rows"][0].get("streams_assimilated").is_none());
        assert!(!render_table(&rows, 0, &metrics, &checked(&rows)).starts_with("Stream:"));
    }

    /// Under `--streams` the pointwise output is the per-stream file alone,
    /// and its header says so; no row combines streams.
    #[test]
    fn pointwise_under_streams_writes_the_per_stream_file_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scores.tsv");
        let (written, n) = write_pointwise_streams(&[ablation_table()], &["cases".into()], &path)
            .unwrap();
        assert_eq!(written, dir.path().join("scores.streams.tsv"));
        assert!(!path.exists(), "no joint file under --streams");
        let text = std::fs::read_to_string(&written).unwrap();
        assert!(text.contains("# camdl compare --pointwise --streams cases:")
                && text.contains("No joint file is written"), "{text}");
        assert!(!text.contains("do not sum to the joint scores"), "{text}");
        assert_eq!(n, 3, "one candidate × three steps");
        let rows: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).skip(1).collect();
        assert!(rows.iter().all(|l| l.split('\t').nth(3) == Some("cases")), "{text}");
    }

    /// Without `--streams`, the stream-set refusal names the option and the
    /// streams the two rows share; it does not suggest the horizon waiver.
    #[test]
    fn the_stream_set_refusal_names_streams_and_the_shared_leaves() {
        let t = [7.0, 14.0, 21.0];
        let full = row_with_streams("full", &t, &["cases", "deaths"]);
        let cases_only = row_with_streams("cases_only", &t, &["cases"]);
        let err = axis(&[full.clone(), cases_only]).unwrap_err();
        assert!(err.contains("--streams cases"), "{err}");
        assert!(!err.contains("--allow-mismatched-horizon"), "{err}");
        // Nothing shared, nothing suggested.
        let hosp = row_with_streams("hosp", &t, &["hospital"]);
        let err = axis(&[full, hosp]).unwrap_err();
        assert!(!err.contains("--streams"), "{err}");
    }

    /// Under `--streams`, when rows that assimilated different sets start
    /// scoring the leaf at different times, the refusal says that a fit
    /// conditioning on its first observation starts one step later.
    #[test]
    fn a_first_step_mismatch_between_ablated_fits_names_ic_free() {
        let full = derived_row("full", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap();
        let mut late = derived_row("cases_only", &CASES_DEATHS, &["cases"]).onto("cases").unwrap();
        late.trace.steps.remove(0);
        let waived = Waivers { in_sample_ablation: true, ..Default::default() };
        let err = outcome_of(&[full.clone(), late.clone()], &waived).unwrap_err();
        assert!(err.contains("T_score") && err.contains("ic_free") && err.contains("one step later"),
            "{err}");
        // The same count of steps at different times: the axis check speaks.
        let mut early = full.clone();
        early.trace.steps.pop();
        let err = outcome_of(&[early.clone(), late.clone()], &waived).unwrap_err();
        assert!(err.contains("not on the same observation axis") && err.contains("ic_free"), "{err}");
        // Rows that assimilated the same set get no such sentence.
        let mut twin = derived_row("twin", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap();
        twin.trace.steps.remove(0);
        let err = outcome_of(&[early, twin], &waived).unwrap_err();
        assert!(!err.contains("ic_free"), "{err}");
    }

    // ── pinned renderings ───────────────────────────────────────────────

    // The text and markdown renderings of the in-memory fixtures, held as
    // committed text so that a commit moving the scoring code between crates
    // is seen to change no output. Captured from the code at 868585f7. A
    // change to any of them is a change to what `compare` prints, and is the
    // subject of the commit that makes it. `single_stream_rows` is pinned by
    // `a_single_stream_comparison_renders_as_before`, and the JSON of
    // `two_stream_rows` by `without_streams_the_json_document_is_unchanged`.
    const TWO_STREAM_TABLE: &str = r#"Model   T_score     elpd   Δelpd       LR   se(Δ)                    evidence
─────────────────────────────────────────────────────────────────────────────
a             4   -24.00       —        —       —                           —
b             4   -20.00   +4.00   54.598    0.00   +17.4 dB, very strong for

Per-stream accuracy and calibration:
stream   model   n   mean CRPS   Δ mean CRPS   PIT_cov90
────────────────────────────────────────────────────────
north        a   4       2.000             —        1.00
north        b   4       1.000        -1.000        1.00
south        a   4       5.500             —        1.00
south        b   4       3.000        -2.500        0.00

Scored steps: 4 (t0=1).  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
⚠ 'b' stream 'south': PIT 90%-coverage 0.00 (nominal 0.90) — likely overconfident.
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const TWO_STREAM_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| a | 4 | -24.00 | — | — | — | — |
| b | 4 | -20.00 | +4.00 | 54.598 | 0.00 | +17.4 dB, very strong for |

**Per-stream accuracy and calibration**

| stream | model | n | mean CRPS | Δ mean CRPS | PIT_cov90 |
|---|---|---|---|---|---|
| north | a | 4 | 2.000 | — | 1.00 |
| north | b | 4 | 1.000 | -1.000 | 1.00 |
| south | a | 4 | 5.500 | — | 1.00 |
| south | b | 4 | 3.000 | -2.500 | 0.00 |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- ⚠ 'b' stream 'south': PIT 90%-coverage 0.00 (nominal 0.90) — likely overconfident.
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const ABLATION_TABLE: &str = r#"Stream: cases — every row scored on this stream alone.

Model        T_score    elpd   Δelpd      LR   se(Δ)                 evidence    crps    Δcrps   PIT_cov90
──────────────────────────────────────────────────────────────────────────────────────────────────────────
full               3   -9.00       —       —       —                        —   0.500        —        1.00
cases_only         3   -9.00   +0.00   1.000    0.00   +0.0 dB, indeterminate   0.500   +0.000        1.00

Scored steps: 3.  Baseline: full.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: held-out tail (train_end = 3) — θ sealed before the scored window.
Assimilated: 'full' assimilated cases, deaths; 'cases_only' assimilated cases.
ⓘ not checked for a shared observation set: 'full' (a test fixture, built from scores alone); 'cases_only' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'full' (trace older than schema 5); not checked; filter settings not recorded for 'cases_only' (trace older than schema 5); not checked
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const ABLATION_MD: &str = r#"### Stream `cases`

_Every row scored on this stream alone._

| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence | crps | Δcrps | PIT_cov90 |
|---|---|---|---|---|---|---|---|---|---|
| full | 3 | -9.00 | — | — | — | — | 0.500 | — | 1.00 |
| cases_only | 3 | -9.00 | +0.00 | 1.000 | 0.00 | +0.0 dB, indeterminate | 0.500 | +0.000 | 1.00 |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: held-out tail (train_end = 3) — θ sealed before the scored window._

_Assimilated: 'full' assimilated cases, deaths; 'cases_only' assimilated cases._

- ⓘ not checked for a shared observation set: 'full' (a test fixture, built from scores alone); 'cases_only' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'full' (trace older than schema 5); not checked; filter settings not recorded for 'cases_only' (trace older than schema 5); not checked
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const NONFINITE_TABLE: &str = r#"Model   T_score     elpd   Δelpd   LR   se(Δ)   evidence
────────────────────────────────────────────────────────
b             3     -inf    -inf    0    0.00          —
a             3   -18.00       —    —       —          —

Scored steps: 3 (t0=1).  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const NONFINITE_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| b | 3 | -inf | -inf | 0 | 0.00 | — |
| a | 3 | -18.00 | — | — | — | — |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const WAIVED_MISMATCHED_HORIZON_TABLE: &str = r#"Model   T_score     elpd   Δelpd   LR   se(Δ)   evidence
────────────────────────────────────────────────────────
a             3   -18.00       —    —       —          —
b             2   -11.00       —    —       —          —

Scored steps: 3 (t0=1).  Baseline: a.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
⚠ --allow-mismatched-horizon was set for 'a', 'b': T_score differs across models: 'b' 2; 'a' 3. Δ columns suppressed ('—' in the table, null in the JSON): a Δ pairs two rows' scores observation by observation, and these rows scored different numbers of observations.
ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const WAIVED_MISMATCHED_HORIZON_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| a | 3 | -18.00 | — | — | — | — |
| b | 2 | -11.00 | — | — | — | — |

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⚠ --allow-mismatched-horizon was set for 'a', 'b': T_score differs across models: 'b' 2; 'a' 3. Δ columns suppressed ('—' in the table, null in the JSON): a Δ pairs two rows' scores observation by observation, and these rows scored different numbers of observations.
- ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const WAIVED_MIXED_PROVENANCE_TABLE: &str = r#"Model   T_score     elpd   Δelpd      LR   se(Δ)                evidence
────────────────────────────────────────────────────────────────────────
a             3   -18.00       —       —       —                       —
b             3   -17.00   +1.00   2.718    0.91   +4.3 dB, within noise

Scored steps: 3 (t0=1).  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
⚠ --allow-mixed-provenance was set for 'a', 'b': the compared traces mix provenance kinds: 'a' plug_in (a single θ); 'b' posterior (mixture over 4 draws). A plug-in predictive is under-dispersed relative to a posterior mixture, so the Δ confounds model difference with dispersion treatment.
ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b -0.17
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const WAIVED_MIXED_PROVENANCE_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| a | 3 | -18.00 | — | — | — | — |
| b | 3 | -17.00 | +1.00 | 2.718 | 0.91 | +4.3 dB, within noise |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⚠ --allow-mixed-provenance was set for 'a', 'b': the compared traces mix provenance kinds: 'a' plug_in (a single θ); 'b' posterior (mixture over 4 draws). A plug-in predictive is under-dispersed relative to a posterior mixture, so the Δ confounds model difference with dispersion treatment.
- ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
- note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b -0.17
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const WAIVED_DATA_MISMATCH_TABLE: &str = r#"Model   T_score     elpd   Δelpd      LR   se(Δ)                evidence
────────────────────────────────────────────────────────────────────────
a             3   -18.00       —       —       —                       —
b             3   -17.00   +1.00   2.718    0.91   +4.3 dB, within noise

Scored steps: 3 (t0=1).  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
⚠ --allow-data-mismatch was set for 'a', 'b': 'a' and 'b' were fit to different observed data: 'cases' (aaaaaaaa in 'a', bbbbbbbb in 'b'). Δelpd across them is part model difference and part data difference, and the table cannot separate the two.
ⓘ data compared on 8 characters of the SHA-256 for 'a', 'b': its fit.meta.json was written before full digests were recorded, so the shared-data check for that row compares the first 8 characters only.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b -0.17
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const WAIVED_DATA_MISMATCH_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| a | 3 | -18.00 | — | — | — | — |
| b | 3 | -17.00 | +1.00 | 2.718 | 0.91 | +4.3 dB, within noise |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⚠ --allow-data-mismatch was set for 'a', 'b': 'a' and 'b' were fit to different observed data: 'cases' (aaaaaaaa in 'a', bbbbbbbb in 'b'). Δelpd across them is part model difference and part data difference, and the table cannot separate the two.
- ⓘ data compared on 8 characters of the SHA-256 for 'a', 'b': its fit.meta.json was written before full digests were recorded, so the shared-data check for that row compares the first 8 characters only.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
- note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b -0.17
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const WAIVED_MISMATCHED_FILTER_SETTINGS_TABLE: &str = r#"Model   T_score     elpd   Δelpd      LR   se(Δ)                evidence
────────────────────────────────────────────────────────────────────────
a             3   -18.00       —       —       —                       —
b             3   -17.00   +1.00   2.718    0.91   +4.3 dB, within noise

Scored steps: 3 (t0=1).  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
⚠ --allow-mismatched-filter-settings was set for 'a', 'b': the compared traces were scored with different particle-filter settings: particles 200 in 'b' vs particles 1000 in 'a'. The log of a particle-filter likelihood estimate is biased downward, and by more at fewer particles (and fewer replicates combined), so part of Δelpd would measure the filters rather than the models.
ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b -0.17
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const WAIVED_MISMATCHED_FILTER_SETTINGS_MD: &str = r#"| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| a | 3 | -18.00 | — | — | — | — |
| b | 3 | -17.00 | +1.00 | 2.718 | 0.91 | +4.3 dB, within noise |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

- ⚠ --allow-mismatched-filter-settings was set for 'a', 'b': the compared traces were scored with different particle-filter settings: particles 200 in 'b' vs particles 1000 in 'a'. The log of a particle-filter likelihood estimate is biased downward, and by more at fewer particles (and fewer replicates combined), so part of Δelpd would measure the filters rather than the models.
- ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ CRPS and PIT calibration not available for 'a' (trace predates the per-stream breakdown), 'b' (trace predates the per-stream breakdown).
- note: lag-1 autocorrelation of the per-step Δelpd vs baseline: b -0.17
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;
    const WAIVED_IN_SAMPLE_ABLATION_TABLE: &str = r#"Stream: cases — every row scored on this stream alone.

Model   T_score    elpd   Δelpd      LR   se(Δ)                 evidence
────────────────────────────────────────────────────────────────────────
a             3   -9.00       —       —       —                        —
b             3   -9.00   +0.00   1.000    0.00   +0.0 dB, indeterminate

Scored steps: 3.  Baseline: a.
Sorted by Δelpd ascending — best-supported model at the bottom.
Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model.
Assimilated: 'a' assimilated cases, deaths; 'b' assimilated cases.
⚠ --allow-in-sample-ablation was set for 'a', 'b': the compared fits assimilated different streams and are scored in-sample: 'b' assimilated cases; 'a' assimilated cases, deaths. Scored in-sample, Δelpd leans toward the fit that assimilated fewer streams: its θ̂ was chosen to fit the shared stream alone, while the other fit's traded that stream off against the rest, so on its own training data it fits it no better and usually slightly worse, whatever the extra streams contribute to prediction.
ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).
Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`).
"#;
    const WAIVED_IN_SAMPLE_ABLATION_MD: &str = r#"### Stream `cases`

_Every row scored on this stream alone._

| Model | T_score | elpd | Δelpd | LR | se(Δ) | evidence |
|---|---|---|---|---|---|---|
| a | 3 | -9.00 | — | — | — | — |
| b | 3 | -9.00 | +0.00 | 1.000 | 0.00 | +0.0 dB, indeterminate |

_Sorted by Δelpd ascending — best-supported model at the bottom._

_Conditioning: in-sample — θ fit to all scored data; Δelpd favors the more flexible model._

_Assimilated: 'a' assimilated cases, deaths; 'b' assimilated cases._

- ⚠ --allow-in-sample-ablation was set for 'a', 'b': the compared fits assimilated different streams and are scored in-sample: 'b' assimilated cases; 'a' assimilated cases, deaths. Scored in-sample, Δelpd leans toward the fit that assimilated fewer streams: its θ̂ was chosen to fit the shared stream alone, while the other fit's traded that stream off against the rest, so on its own training data it fits it no better and usually slightly worse, whatever the extra streams contribute to prediction.
- ⓘ not checked for a shared observation set: 'a' (a test fixture, built from scores alone); 'b' (a test fixture, built from scores alone). A difference against such a row may be a data difference, not a model difference.
- ⓘ filter settings not recorded for 'a' (trace older than schema 5); not checked; filter settings not recorded for 'b' (trace older than schema 5); not checked
- note: scores are plug-in (scored at a single θ, dropping parameter uncertainty → under-dispersed); and in-sample (θ fit to all the data → optimistic in absolute level, by roughly the effective number of parameters fit, which does not cancel in a Δelpd and so biases differences toward the more flexible model) — not a leave-future-out forecast score. See LFO (#295).

_Definitions and caveats: `camdl compare --explain` (or `camdl docs model-comparison`)._
"#;

    /// Both renderings of one comparison, in the order the pins are held.
    fn rendered(rows: &[Row], base_idx: usize, metrics: &[&str], checks: &Preflight)
        -> (String, String)
    {
        let metrics: Vec<String> = metrics.iter().map(|m| m.to_string()).collect();
        (render_table(rows, base_idx, &metrics, checks), render_md(rows, base_idx, &metrics, checks))
    }

    const ALL_METRICS: [&str; 3] = ["elpd", "crps", "pit_cov90"];

    #[test]
    fn the_two_stream_comparison_renders_as_pinned() {
        let rows = two_stream_rows();
        let (table, md) = rendered(&rows, 0, &ALL_METRICS, &checked(&rows));
        assert_eq!(table, TWO_STREAM_TABLE);
        assert_eq!(md, TWO_STREAM_MD);
    }

    #[test]
    fn the_stream_ablation_table_renders_as_pinned() {
        let t = ablation_table();
        let (table, md) = rendered(&t.rows, t.base_idx, &ALL_METRICS, &t.checks);
        assert_eq!(table, ABLATION_TABLE);
        assert_eq!(md, ABLATION_MD);
    }

    /// One comparison per waiver, each rendered under the flag it needs.
    #[test]
    fn each_waived_comparison_renders_as_pinned() {
        let pins = [
            (Waiver::MismatchedHorizon, WAIVED_MISMATCHED_HORIZON_TABLE, WAIVED_MISMATCHED_HORIZON_MD),
            (Waiver::MixedProvenance, WAIVED_MIXED_PROVENANCE_TABLE, WAIVED_MIXED_PROVENANCE_MD),
            (Waiver::DataMismatch, WAIVED_DATA_MISMATCH_TABLE, WAIVED_DATA_MISMATCH_MD),
            (Waiver::MismatchedFilterSettings, WAIVED_MISMATCHED_FILTER_SETTINGS_TABLE,
             WAIVED_MISMATCHED_FILTER_SETTINGS_MD),
            (Waiver::InSampleAblation, WAIVED_IN_SAMPLE_ABLATION_TABLE, WAIVED_IN_SAMPLE_ABLATION_MD),
        ];
        assert_eq!(pins.map(|p| p.0), ALL_WAIVERS, "one pin per waiver");
        for (w, want_table, want_md) in pins {
            let rows = rows_needing(w);
            let (table, md) = rendered(&rows, 0, &["elpd"], &checked(&rows));
            assert_eq!(table, want_table, "{w:?}: the table");
            assert_eq!(md, want_md, "{w:?}: the markdown");
        }
    }

    /// Row `b` scored one step at −∞. This pins what the renderers print
    /// today: `-inf` for its elpd and Δelpd, `0` for the LR, `0.00` for
    /// se(Δ), `—` for the evidence, and no line below the table naming the
    /// non-finite score.
    #[test]
    fn a_row_with_a_non_finite_score_renders_as_pinned() {
        let rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, f64::NEG_INFINITY, -6.0]),
        ];
        let checks = outcome_of(&rows, &Waivers::default()).expect("no check refuses it");
        let (table, md) = rendered(&rows, 0, &["elpd"], &checks);
        assert_eq!(table, NONFINITE_TABLE);
        assert_eq!(md, NONFINITE_MD);
    }

    // ── every check evaluated and recorded (proposal step 6) ─────────────

    /// Three rows failing three independent checks: mixed provenance,
    /// differing filter settings and differing bound data.
    fn three_failures() -> Vec<Row> {
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
            row_at("c", &[7.0, 14.0, 21.0], &[-5.5, -6.0, -6.0]),
        ];
        rows[1].trace.provenance = score::trace::Provenance::Posterior { n_draws: 4 };
        rows[0].trace.derivation = Some(derivation(1000, 5));
        rows[1].trace.derivation = Some(derivation(200, 5));
        rows[2].trace.derivation = Some(derivation(1000, 5));
        rows[0].data = digests(&[("cases", "aaaaaaaa11")]);
        rows[1].data = digests(&[("cases", "aaaaaaaa11")]);
        rows[2].data = digests(&[("cases", "bbbbbbbb22")]);
        rows
    }

    /// Rows whose conditioning modes differ, one of them held out.
    fn mixed_conditioning() -> Vec<Row> {
        let mut rows = rows_needing(Waiver::InSampleAblation);
        rows[1].trace.conditioning = Conditioning::HoldOutTail {
            train_end: 3.0, theta_source: "b".into() };
        rows
    }

    /// Rows re-scored on one stream that fail to pair: the same count of
    /// steps at different times.
    fn unpaired_on_one_stream() -> Vec<Row> {
        let full = derived_row("full", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap();
        let mut late = derived_row("late", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap();
        late.trace.steps[2].t = 28.0;
        vec![full, late]
    }

    /// Every fixture the tests below run over: passing, each waiver needed,
    /// several failures, and the dependency cases.
    fn fixtures() -> Vec<(String, Vec<Row>)> {
        let mut out = vec![
            ("passing".to_string(), vec![
                row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
                row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0])]),
            ("single stream".into(), single_stream_rows()),
            ("two streams".into(), two_stream_rows()),
            ("ablation".into(), ablation_table().rows),
            ("three failures".into(), three_failures()),
            ("mixed conditioning".into(), mixed_conditioning()),
            ("unpaired on one stream".into(), unpaired_on_one_stream()),
        ];
        for w in ALL_WAIVERS {
            out.push((format!("{w:?}"), rows_needing(w)));
        }
        out
    }

    /// Guarantee 2: one result per check, in declaration order, whatever
    /// fails and whatever is waived. The order and the count are stated
    /// here, not read from `CheckId::ALL`: the match is exhaustive, so a new
    /// check fails to compile until it is placed, and a check missing from
    /// the results, repeated or out of order breaks the sequence.
    #[test]
    fn every_check_has_one_result_in_declaration_order() {
        let position = |c: CheckId| match c {
            CheckId::Basis => 0,
            CheckId::Conditioning => 1,
            CheckId::TrainingBoundary => 2,
            CheckId::InSampleAblation => 3,
            CheckId::Provenance => 4,
            CheckId::ScoredSteps => 5,
            CheckId::Pairing => 6,
            CheckId::SharedData => 7,
            CheckId::FilterSettings => 8,
            CheckId::Mode => 9,
            CheckId::Role => 10,
            CheckId::Lead => 11,
            CheckId::Reach => 12,
            CheckId::Covariates => 13,
            CheckId::ScoredRows => 14,
        };
        let declared: Vec<usize> = (0..15).collect();
        for (name, rows) in fixtures() {
            for waivers in [Waivers::default(), every_waiver()] {
                let got: Vec<usize> =
                    preflight(&rows, &waivers).results.iter().map(|r| position(r.check)).collect();
                assert_eq!(got, declared, "{name}, {waivers:?}");
            }
        }
    }

    /// Test 18: three independent failures are three refusals, and the
    /// command prints all three, in check order, where it printed the first.
    #[test]
    fn three_independent_failures_are_refused_together() {
        let rows = three_failures();
        let refusals = preflight(&rows, &Waivers::default()).refusals();
        assert_eq!(refusals.iter().map(|r| r.check).collect::<Vec<_>>(),
            [CheckId::Provenance, CheckId::SharedData, CheckId::FilterSettings]);
        let message = match prepare_table(&crate::out::Out::new(false), rows, &Waivers::default(), None) {
            Err(m) => m.join("\n"),
            Ok(_) => panic!("refused"),
        };
        assert_eq!(message.matches("error: ").count(), 3, "{message}");
        let at = |needle: &str| message.find(needle)
            .unwrap_or_else(|| panic!("{needle:?} in:\n{message}"));
        assert!(at("mix provenance kinds") < at("different observed data")
                && at("different observed data") < at("particle-filter settings"),
            "in check order:\n{message}");
        for flag in ["--allow-mixed-provenance", "--allow-data-mismatch",
                     "--allow-mismatched-filter-settings"] {
            assert!(message.contains(flag), "each with its remedy: {flag}\n{message}");
        }
    }

    /// Test 19: a check that reads what a failed check established is
    /// recorded as not evaluated, naming that check.
    #[test]
    fn a_check_whose_prerequisite_failed_is_not_evaluated() {
        let outcome_of_check = |checks: &Preflight, c: CheckId| {
            checks.result(c).map(|r| r.outcome.clone())
        };
        let checks = preflight(&mixed_conditioning(), &every_waiver());
        assert!(matches!(checks.result(CheckId::Conditioning).unwrap().outcome,
            CheckOutcome::Failed { waived: false, .. }));
        for c in [CheckId::InSampleAblation, CheckId::TrainingBoundary] {
            assert_eq!(outcome_of_check(&checks, c),
                Some(CheckOutcome::NotEvaluated { because: CheckId::Conditioning }), "{c:?}");
        }

        let checks = preflight(&unpaired_on_one_stream(), &every_waiver());
        assert!(matches!(&checks.result(CheckId::Pairing).unwrap().outcome,
            CheckOutcome::Failed { finding: Finding::ObservationTimesDiffer { step: 3, .. }, waived: false }),
            "{checks:?}");
        assert_eq!(outcome_of_check(&checks, CheckId::SharedData),
            Some(CheckOutcome::NotEvaluated { because: CheckId::Pairing }));
        // The same rows scored jointly: the data check does not need pairing.
        let joint: Vec<Row> = unpaired_on_one_stream().into_iter()
            .map(|mut r| { r.trace.scored_streams = None; r }).collect();
        let checks = preflight(&joint, &every_waiver());
        assert!(!matches!(checks.result(CheckId::SharedData).unwrap().outcome,
            CheckOutcome::NotEvaluated { .. }), "{checks:?}");

        // Step counts differing without the waiver: pairing is not evaluated;
        // with it, pairing joins the steps by time.
        let rows = rows_needing(Waiver::MismatchedHorizon);
        assert_eq!(outcome_of_check(&preflight(&rows, &Waivers::default()), CheckId::Pairing),
            Some(CheckOutcome::NotEvaluated { because: CheckId::ScoredSteps }));
        assert_eq!(outcome_of_check(&preflight(&rows, &every_waiver()), CheckId::Pairing),
            Some(CheckOutcome::Passed));
    }

    fn unchecked_members(r: &CheckResult) -> Vec<&str> {
        r.not_checked.iter().map(|u| u.member.as_str()).collect()
    }

    /// Guarantee 3 and test 20: a row lacking the fact a check reads is
    /// listed as not checked whatever the outcome over the other rows. Two
    /// rows that differ and one that carries nothing: a failure over the two,
    /// and the third not checked; for the bound-data and filter-settings
    /// checks.
    #[test]
    fn a_failed_check_still_lists_the_rows_it_could_not_check() {
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
            row_at("c", &[7.0, 14.0, 21.0], &[-5.5, -6.0, -6.0]),
        ];
        rows[0].data = digests(&[("cases", "aaaaaaaa")]);
        rows[1].data = digests(&[("cases", "bbbbbbbb")]);
        let r = check_shared_data(&rows);
        assert!(matches!(r.outcome, CheckOutcome::Failed { .. }), "{r:?}");
        assert_eq!(unchecked_members(&r), ["c"]);

        // Filter settings: `a` and `b` differ, `c` records none.
        rows[0].trace.derivation = Some(derivation(1000, 5));
        rows[1].trace.derivation = Some(derivation(200, 5));
        let r = check_filter_settings(&rows);
        assert!(matches!(r.outcome, CheckOutcome::Failed { .. }), "{r:?}");
        assert_eq!(r.not_checked, [Unchecked { member: "c".into(),
            reason: NotCheckedReason::TraceBeforeSchema5 }]);

        // Every format still states the rows not checked when the checks
        // fail and are waived.
        let checks = checked(&rows);
        for note in [data_unchecked_note(&checks, &rows), filter_settings_unchecked_note(&checks, &rows)] {
            assert!(note.as_deref().is_some_and(|n| n.contains("'c'")), "{note:?}");
        }
    }

    /// Two rows that differ only in their predictive basis.
    fn mixed_bases() -> Vec<Row> {
        let mut rows = rows_needing(Waiver::MixedProvenance);
        rows[1].trace.provenance = rows[0].trace.provenance;
        rows[1].trace.basis = PredictiveBasis::OdeFreeForward { samples_per_draw: 100 };
        rows
    }

    /// Held-out rows sealing training at these boundaries.
    fn held_out_at(ends: &[f64]) -> Vec<Row> {
        let t = [7.0, 14.0, 21.0];
        ends.iter().zip(["x", "y", "z"]).map(|(&end, name)| {
            let mut r = row_at(name, &t, &[-1.0; 3]);
            r.trace.conditioning = Conditioning::HoldOutTail { train_end: end, theta_source: "f".into() };
            r
        }).collect()
    }

    /// The flag `w` alone.
    fn only(w: Waiver) -> Waivers {
        let none = Waivers::default();
        match w {
            Waiver::MismatchedHorizon => Waivers { mismatched_horizon: true, ..none },
            Waiver::MixedProvenance => Waivers { mixed_provenance: true, ..none },
            Waiver::DataMismatch => Waivers { data_mismatch: true, ..none },
            Waiver::MismatchedFilterSettings => Waivers { mismatched_filter_settings: true, ..none },
            Waiver::InSampleAblation => Waivers { in_sample_ablation: true, ..none },
        }
    }

    /// Guarantee 4: an override is listed when, and only when, its check
    /// failed and its waiver was given. Per fixture, stated here rather than
    /// recomputed: the waivers it needs, and the checks it fails that have no
    /// waiver (conditioning, basis, training boundary, pairing), which stay
    /// refused when every waiver is given.
    #[test]
    fn overrides_are_exactly_the_waived_failures() {
        use CheckId as C;
        use Waiver as W;
        for c in [C::Basis, C::Conditioning, C::TrainingBoundary, C::Pairing] {
            assert_eq!(c.waiver(), None, "{c:?} has no waiver");
        }
        let mut cases: Vec<(String, Vec<Row>, Vec<Waiver>, Vec<CheckId>)> = vec![
            ("passing".into(), fixtures()[0].1.clone(), vec![], vec![]),
            ("single stream".into(), single_stream_rows(), vec![], vec![]),
            ("two streams".into(), two_stream_rows(), vec![], vec![]),
            ("ablation".into(), ablation_table().rows, vec![], vec![]),
            ("three failures".into(), three_failures(),
                vec![W::MixedProvenance, W::DataMismatch, W::MismatchedFilterSettings], vec![]),
            ("mixed conditioning".into(), mixed_conditioning(), vec![], vec![C::Conditioning]),
            ("unpaired on one stream".into(), unpaired_on_one_stream(), vec![], vec![C::Pairing]),
            ("mixed bases".into(), mixed_bases(), vec![], vec![C::Basis]),
            ("boundaries".into(), held_out_at(&[3.0, 5.0]), vec![], vec![C::TrainingBoundary]),
        ];
        for w in ALL_WAIVERS {
            cases.push((format!("{w:?}"), rows_needing(w), vec![w], vec![]));
        }
        let waivers_of = |o: &[Override]| o.iter().map(|o| o.waiver).collect::<Vec<_>>();
        let refused = |c: &Preflight| c.refusals().iter().map(|r| r.check).collect::<Vec<_>>();
        for (name, rows, needed, unwaivable) in cases {
            let all = preflight(&rows, &every_waiver());
            assert_eq!(waivers_of(&all.overrides()), needed, "{name}: every flag given");
            assert_eq!(refused(&all), unwaivable, "{name}: refused whatever is waived");
            for w in ALL_WAIVERS {
                let one = preflight(&rows, &only(w));
                let want: Vec<Waiver> = needed.iter().copied().filter(|n| *n == w).collect();
                assert_eq!(waivers_of(&one.overrides()), want, "{name}: {w:?} alone");
            }
            assert!(preflight(&rows, &Waivers::default()).overrides().is_empty(), "{name}: none");
        }
    }

    /// Differing training boundaries are refused, with no waiver to give.
    /// Boundaries are grouped within the time tolerance of a group's
    /// smallest, so three boundaries each within the tolerance of the next
    /// but not of the smallest are refused in every order, where the code
    /// comparing adjacent rows passed them in that order; three within the
    /// tolerance of the smallest pass.
    #[test]
    fn differing_training_boundaries_are_refused_whatever_is_waived() {
        let eps = crate::fit::runner::TIME_TIE_EPS;
        assert_eq!(eps, 1e-9, "the cases below are written for this tolerance");
        let boundary = |rows: &[Row], w: &Waivers| preflight(rows, w)
            .result(CheckId::TrainingBoundary).unwrap().outcome.clone();
        let rows = held_out_at(&[3.0, 5.0]);
        for w in [Waivers::default(), every_waiver()] {
            assert!(matches!(boundary(&rows, &w), CheckOutcome::Failed { waived: false, .. }),
                "{w:?}");
            let refusal = outcome_of(&rows, &w).expect_err("refused");
            assert!(refusal.contains("seal training at different boundaries"), "{refusal}");
        }
        for ends in orders(&[60.0, 60.0 + 0.8e-9, 60.0 + 1.6e-9]) {
            assert!(matches!(boundary(&held_out_at(&ends), &every_waiver()),
                CheckOutcome::Failed { waived: false, .. }), "{ends:?}");
        }
        for ends in orders(&[60.0, 60.0 + 0.5e-9, 60.0 + 0.9e-9]) {
            assert_eq!(boundary(&held_out_at(&ends), &Waivers::default()), CheckOutcome::Passed,
                "{ends:?}");
        }
    }

    /// An override's JSON `rows` are every member of its finding, sorted by
    /// name: for a data mismatch over three rows, all three, the two that
    /// agree included.
    #[test]
    fn a_waived_data_mismatch_lists_every_row_of_the_finding() {
        let mut rows = vec![
            row_at("c", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("a", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.5, -6.0, -6.0]),
        ];
        rows[0].data = digests(&[("cases", "aaaaaaaa11")]);
        rows[1].data = digests(&[("cases", "bbbbbbbb22")]);
        rows[2].data = digests(&[("cases", "aaaaaaaa11")]);
        let checks = outcome_of(&rows, &Waivers { data_mismatch: true, ..Default::default() })
            .expect("waived");
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &["elpd".to_string()], &checks)).unwrap();
        assert_eq!(v["overrides"].as_array().map(Vec::len), Some(1), "{v}");
        assert_eq!(v["overrides"][0]["flag"], "--allow-data-mismatch");
        assert_eq!(v["overrides"][0]["rows"], serde_json::json!(["a", "b", "c"]), "{v}");
    }

    /// Guarantee 11, for the checks: permuting the rows changes no result,
    /// because a finding partitions the rows by value.
    #[test]
    fn permuting_the_rows_changes_no_check_result() {
        use score::trace::Coverage;
        let mut fixtures = fixtures();
        // Three rows that pair badly on stream sets, on coverage, and three
        // held-out rows two of which seal at one boundary up to float noise.
        let t = [7.0, 14.0, 21.0];
        fixtures.push(("stream sets".into(), vec![row_with_streams("x", &t, &["north"]),
            row_with_streams("y", &t, &["north", "south"]), row_with_streams("z", &t, &["north"])]));
        let iv = |s: f64, e: f64| Coverage::Interval { start: s, stop: e };
        fixtures.push(("coverage".into(), vec![
            row_covering("x", &t, "cases", &[iv(0.0, 7.0), iv(7.0, 14.0), iv(14.0, 21.0)]),
            row_covering("y", &t, "cases", &[iv(0.0, 7.0), iv(8.0, 14.0), iv(14.0, 21.0)]),
            row_covering("z", &t, "cases", &[iv(0.0, 7.0), iv(7.0, 14.0), iv(14.0, 21.0)])]));
        let mut boundaries = vec![row_at("x", &t, &[-1.0; 3]), row_at("y", &t, &[-1.0; 3]),
                                  row_at("z", &t, &[-1.0; 3])];
        for (r, end) in boundaries.iter_mut().zip([3.0, 3.0 + 1e-12, 5.0]) {
            r.trace.conditioning = Conditioning::HoldOutTail { train_end: end, theta_source: "f".into() };
        }
        fixtures.push(("boundaries".into(), boundaries));
        for (name, rows) in fixtures {
            let want = preflight(&rows, &every_waiver());
            for shift in 1..rows.len() {
                let mut permuted = rows.clone();
                permuted.rotate_left(shift);
                assert_eq!(preflight(&permuted, &every_waiver()), want, "{name}, rotated {shift}");
                permuted.reverse();
                assert_eq!(preflight(&permuted, &every_waiver()), want, "{name}, reversed");
            }
        }
    }

    /// Two rows differing in one fact of their scores are refused, the
    /// refusal naming the field: the mode, the role, the lead, the reach and
    /// the covariate statement, whatever is waived; a held-out score of a
    /// stored fit's verified training rows against a fixed point's declared
    /// ones; and different observed values at one time and stream, which
    /// only `--allow-data-mismatch` lets through.
    #[test]
    fn each_score_fact_that_differs_is_refused_naming_the_field() {
        let t = [7.0, 14.0, 21.0];
        let stored = |name: &str| {
            let mut r = row_with_streams(name, &t, &["cases"]);
            r.facts = ScoreFacts { stored: true, ..r.facts.clone() };
            r
        };
        let base = || vec![stored("a"), stored("b")];
        assert!(outcome_of(&base(), &Waivers::default()).is_ok(), "two equal rows compare");
        type Change = Box<dyn Fn(&mut ScoreFacts)>;
        let cases: Vec<(&str, Change)> = vec![
            ("(mode)", Box::new(|f| f.mode = "free_forward".into())),
            ("(role)", Box::new(|f| f.role = "held_out".into())),
            ("(lead)", Box::new(|f| f.lead = Some("7".into()))),
            ("(reach)", Box::new(|f| f.reach = Some("2 weeks".into()))),
            ("(covariates)", Box::new(|f| f.realised = true)),
        ];
        for (field, change) in &cases {
            let mut rows = base();
            change(&mut rows[1].facts);
            for waivers in [Waivers::default(), every_waiver()] {
                let err = outcome_of(&rows, &waivers).expect_err(field);
                assert!(err.contains(field) && err.contains("'a'") && err.contains("'b'"), "{field}: {err}");
            }
        }

        let mut rows = base();
        rows[0].trace.conditioning = Conditioning::HoldOutTail { train_end: 3.0, theta_source: "f".into() };
        rows[1].trace.conditioning = Conditioning::HoldOutDeclared { trained_through: 3.0 };
        let err = outcome_of(&rows, &every_waiver()).expect_err("verified against declared");
        assert!(err.contains("hold_out_tail") && err.contains("hold_out_declared"), "{err}");

        let mut rows = base();
        rows[1].trace.steps[1].per_stream[0].y_obs = 6.0;
        let err = outcome_of(&rows, &Waivers::default()).expect_err("other observed values");
        assert!(err.contains("(scored rows)") && err.contains("'cases' at t = 14") && err.contains("6 in 'b'"), "{err}");
        let waived = Waivers { data_mismatch: true, ..Waivers::default() };
        assert!(outcome_of(&rows, &waived).is_ok(), "--allow-data-mismatch renders it");
    }

    /// A stored score's row shows its role, and a forecast's its lead, beside
    /// the model, and its WIS and 50% coverage beside the CRPS and PIT; a
    /// prequential.json's row shows none of them.
    #[test]
    fn a_stored_scores_table_has_role_lead_wis_and_cov50() {
        let t = [7.0, 14.0, 21.0];
        let mut rows = vec![row_with_streams("a", &t, &["cases"]), row_with_streams("b", &t, &["cases"])];
        let metrics: Vec<String> = ["elpd", "crps", "wis", "pit_cov90", "cov50"].map(String::from).to_vec();
        let plain = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(!plain.contains("role") && !plain.contains("wis") && !plain.contains("cov50"), "{plain}");
        for (k, r) in rows.iter_mut().enumerate() {
            r.facts = ScoreFacts {
                mode: "forecast".into(), role: "future".into(), lead: Some("7".into()),
                reach: Some("1 weeks".into()), stored: true, ..r.facts.clone()
            };
            for s in &mut r.trace.steps {
                s.per_stream[0].wis = Some(1.0 + k as f64);
                s.per_stream[0].in_50 = Some(k == 0);
            }
        }
        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        let header = table.lines().next().unwrap();
        let cols: Vec<&str> = header.split_whitespace().collect();
        assert_eq!(cols, ["Model", "role", "lead", "T_score", "elpd", "Δelpd", "LR", "se(Δ)", "evidence",
            "crps", "Δcrps", "wis", "Δwis", "PIT_cov90", "cov50"], "{table}");
        let b = table.lines().find(|l| l.starts_with("b ")).unwrap();
        assert!(b.contains("future") && b.contains(" 7 ") && b.contains("+1.000") && b.contains("0.00"), "{table}");
        let md = render_md(&rows, 0, &metrics, &checked(&rows));
        assert!(md.contains("| Model | role | lead | T_score"), "{md}");
        let v: serde_json::Value = serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert_eq!(v["rows"][1]["per_stream"][0]["delta_mean_wis"], serde_json::json!(1.0), "{v}");
        assert_eq!(v["rows"][1]["role"], "future");
        assert_eq!(v["rows"][1]["lead"], "7");
    }

    /// Between two rows of stored filtered scores, a verdict outside the
    /// se(Δ) band says the filter noise was not measured when a row carries
    /// no filter-noise SE (`--replicates 1`), and carries no such note when
    /// both do; a prequential.json's row is rendered as before.
    #[test]
    fn an_unmeasured_filter_noise_is_said_in_the_evidence_cell() {
        let t = [7.0, 14.0, 21.0];
        let mut rows = vec![row_at("a", &t, &[-6.0, -6.0, -6.0]), row_at("b", &t, &[-1.0, -1.5, -0.5])];
        let metrics = vec!["elpd".to_string()];
        let plain = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(!plain.contains("filter noise not measured"), "{plain}");
        for r in &mut rows {
            r.facts.stored = true;
        }
        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(table.contains("overwhelming for, filter noise not measured"), "{table}");
        let v: serde_json::Value = serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert_eq!(v["rows"][1]["filter_noise_measured"], false, "{v}");
        for r in &mut rows {
            set_mc_se(r, Some(0.01));
        }
        let table = render_table(&rows, 0, &metrics, &checked(&rows));
        assert!(!table.contains("filter noise not measured"), "{table}");
    }
}
