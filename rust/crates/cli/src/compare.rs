//! `camdl compare` — multi-model prequential comparison table.
//!
//! Consumes `prequential.json` artifacts written by `fit run` (PFilter
//! stage) or `camdl pfilter --save-prequential`. Computes Δelpd / Δcrps
//! with paired pointwise SEs, refuses structurally unfair comparisons
//! (T_score mismatch by default), renders as table / markdown / JSON.
//!
//! See docs/dev/proposals/2026-04-20-prequential-evaluation.md §8.
//!
//! Scope (Part I):
//!   - baseline-centered Δelpd + paired SE
//!   - Δcrps + PIT 90% coverage column
//!   - T_score fairness preflight (override: --allow-mismatched-horizon)
//!   - observation-axis and stream-set preflights (gh#570; not overridable —
//!     a difference across unlike observations is meaningless however shown)
//!   - bound-data preflight (gh#713; override: --allow-data-mismatch)
//!   - backend preflight on the derive path (gh#729; no override — gh#312)
//!   - formats: table (default), md, json
//!   - compare.toml for reproducible multi-model specs
//! Out of scope (Part II): betting mode, CAS ref resolution, obs-model
//!   preflight, stacking, plotting.

use crate::chain_selection::ChainSelection;
use crate::fit::handle::ResolvedFit;
use crate::posterior_draws::PosteriorDrawsRef;
use serde::Deserialize;
use sim::inference::prequential::PrequentialTrace;
use std::path::{Path, PathBuf};

/// Default particle count for auto-deriving a prequential from a fit handle.
/// Applied uniformly across the compared fits (see [`DeriveSettings`]).
pub(crate) const DEFAULT_DERIVE_PARTICLES: usize = 1000;

/// Default filter seed for auto-deriving a prequential from a fit handle.
pub(crate) const DEFAULT_DERIVE_SEED: u64 = 1;

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

/// Settings applied uniformly to every fit handle whose prequential is
/// auto-derived, so T_score and the scores stay commensurable across the
/// compared fits. An explicit `prequential.json` input ignores these (it is
/// read as-is, at whatever particles/seed produced it).
#[derive(Debug, Clone, Copy)]
struct DeriveSettings {
    particles: usize,
    seed: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)] // gh#241 G3: reject typo'd compare.toml keys instead of silently dropping
struct CompareToml {
    baseline: Option<String>,
    metrics: Option<Vec<String>>,
    format: Option<String>,
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
struct Row {
    name: String,
    path: String,
    trace: PrequentialTrace,
    data: DataIdentity,
}

/// What a compared row can say about the observations it was scored against
/// (gh#713).
///
/// Two fits bound to different data produce a Δelpd that mixes a model
/// difference with a data difference, and neither the table nor the trace says
/// so — the trace records scores and times, never which bytes it scored. The
/// fit's own provenance sidecar (`fit.meta.json`, [`crate::run_meta::FitSidecar`])
/// does: `data_hashes` is stream name → SHA-256 of that stream's file, recorded
/// when the fit ran. Read from there rather than re-hashing, so the check
/// answers "were these fits bound to the same bytes?" using the same record the
/// fit's own identity was built from.
enum DataIdentity {
    /// Stream name → SHA-256 hex of the file's bytes, from the fit's sidecar.
    Digests(std::collections::BTreeMap<String, String>),
    /// Nothing to check, and why — an explicit `prequential.json` (a trace
    /// records no data identity), or a fit whose sidecar is absent. The reason
    /// is printed, because a check that silently covers half the cohort is
    /// worse than one the reader knows the scope of.
    Unknown(String),
}

pub fn cmd_compare(a: &crate::args::CompareArgs) {
    let config_path: Option<String> = a.config.clone();
    let baseline: Option<String> = a.baseline.clone();
    let allow_mismatched_horizon = a.allow_mismatched_horizon;
    let positional: Vec<String> = a.paths.clone();
    let derive = DeriveSettings { particles: a.particles, seed: a.seed };
    let metrics_cli: Option<Vec<String>> = a.metrics.as_ref().map(|s|
        s.split(',').map(|t| t.trim().to_string()).collect());
    let format = match a.format.as_str() {
        "table" => Format::Table,
        "md"    => Format::Md,
        "json"  => Format::Json,
        other   => {
            eprintln!("error: --format must be table|md|json (got '{}')", other);
            std::process::exit(1);
        }
    };

    // Resolve model list: CLI positional > compare.toml
    let (models, cfg_baseline, cfg_metrics, cfg_format) = if let Some(path) = config_path {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            eprintln!("error: cannot read {}: {}", path, e);
            std::process::exit(1);
        });
        let cfg: CompareToml = toml::from_str(&text).unwrap_or_else(|e| {
            eprintln!("error: parsing {}: {}", path, e);
            std::process::exit(1);
        });
        let fmt = cfg.format.as_deref().and_then(|f| match f {
            "table" => Some(Format::Table),
            "md"    => Some(Format::Md),
            "json"  => Some(Format::Json),
            _       => None,
        });
        (cfg.models, cfg.baseline, cfg.metrics, fmt)
    } else if positional.len() >= 2 {
        let models = positional.iter().map(|p| CompareModelEntry {
            name: std::path::Path::new(p).file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.clone()),
            path: p.clone(),
        }).collect();
        (models, None, None, None)
    } else {
        eprintln!("error: compare requires either --config FILE or ≥2 fit stage paths");
        std::process::exit(1);
    };

    if models.len() < 2 {
        eprintln!("error: compare needs ≥2 models; got {}", models.len());
        std::process::exit(1);
    }

    // gh#417/gh#418: chain exclusion resolved PER FIT, bound to each model's
    // name. `@a:4` targets one fit; bare `3,4` is cohort-wide (every fit). Parsed
    // and name-validated at the boundary, before any derivation runs. A fit with
    // no posterior cloud (an optimizer fit, or an explicit prequential.json) has
    // nothing to filter, so the selection is inert there.
    let model_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let cohort = parse_cohort_exclude(&a.exclude_chains, &model_names).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    cohort.warn();

    // Load traces, each filtered by its own fit's selection (if any).
    let rows: Vec<Row> = models.into_iter().map(|m| {
        let selection = cohort.for_fit(&m.name);
        let (trace, data) = load_trace(&m.path, derive, selection).unwrap_or_else(|e| {
            eprintln!("error loading trace for '{}' at '{}': {}", m.name, m.path, e);
            std::process::exit(1);
        });
        Row { name: m.name, path: m.path, trace, data }
    }).collect();

    // Fairness: T_score (n_scored) must agree across rows unless overridden.
    let t_scores: Vec<usize> = rows.iter().map(|r| r.trace.n_scored()).collect();
    let t_ref = t_scores[0];
    let t_mismatch = t_scores.iter().any(|&t| t != t_ref);
    if t_mismatch && !allow_mismatched_horizon {
        eprintln!("error: T_score differs across models: {:?}", t_scores);
        eprintln!("       Δelpd and Δcrps are not commensurable.");
        eprintln!("       Pass --allow-mismatched-horizon to render (uncomparable Δ columns → '—').");
        std::process::exit(2);
    }

    // Fairness, part two: equal T_score is not an equal observation AXIS.
    // `paired_delta` zips steps by index, so two traces scoring the same NUMBER
    // of observations at different TIMES — a hole in one series, a different
    // `t0` — produce a Δelpd, an se(Δ) and a deciban verdict computed across
    // two different axes, rendered as a confident answer. This is not
    // overridable by `--allow-mismatched-horizon`: that flag says "render the Δ
    // columns as '—' because the horizons differ", whereas this is a pairing
    // that is meaningless however it is displayed.
    if !t_mismatch {
        if let Err(e) = check_shared_observation_axis(&rows) {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }

    // Fairness, part three (gh#713): the compared fits must have been bound to
    // the same observations. Two fits on different data produce a Δelpd that is
    // part model difference and part data difference, with nothing in the table
    // to say which. Overridable, unlike the axis checks: comparing a fit on
    // revised counts against the fit on the original ones is a real thing to
    // want, as long as the reader knows the Δ is confounded.
    // The note naming what the check covered, and the confound when it is
    // overridden, are emitted by the renderers (`warning_lines`) so they travel
    // with the artifact; only the refusal is a terminal-side error.
    let data_entries: Vec<(&str, &DataIdentity)> =
        rows.iter().map(|r| (r.name.as_str(), &r.data)).collect();
    if let Err(e) = check_shared_data(&data_entries) {
        if !a.allow_data_mismatch {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    }

    // #295 Ask 1 — the optimism caveat, the miscalibration flags and the trace
    // warnings are emitted by the RENDERERS (`warning_lines`), not here, so they
    // travel with the artifact: a markdown table pasted into a report and a JSON
    // document read by a script carry the same caveats the terminal shows.

    // Baseline: explicit > cfg > argmax elpd.
    let baseline_name = baseline.or(cfg_baseline).unwrap_or_else(|| {
        let best = rows.iter()
            .max_by(|a, b| a.trace.elpd().partial_cmp(&b.trace.elpd())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        best.name.clone()
    });
    let base_idx = rows.iter().position(|r| r.name == baseline_name)
        .unwrap_or_else(|| {
            eprintln!("error: baseline '{}' not found among models: {:?}",
                baseline_name,
                rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>());
            std::process::exit(1);
        });

    let metrics_chosen = metrics_cli.or(cfg_metrics)
        .unwrap_or_else(|| vec!["elpd".into(), "crps".into(), "pit_cov90".into()]);
    let fmt_final = cfg_format.unwrap_or(format);

    if let Some(path) = &a.pointwise {
        match write_pointwise(&rows, base_idx, path) {
            Ok(n) => eprintln!("wrote {n} pointwise rows to {}", path.display()),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }

    let out = match fmt_final {
        Format::Json  => render_json(&rows, base_idx, &metrics_chosen),
        Format::Md    => render_md(&rows, base_idx, &metrics_chosen, t_mismatch),
        Format::Table => render_table(&rows, base_idx, &metrics_chosen, t_mismatch),
    };
    print!("{out}");
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
    /// `joint` for the cross-stream score, `stream` for one stream's own.
    scope: &'static str,
    /// The stream name; empty for `scope = joint`. A separate column rather
    /// than a sentinel in the name column, so no real stream can collide with
    /// it.
    stream: String,
    log_score: Option<f64>,
    baseline_log_score: Option<f64>,
}

impl PointwiseRow {
    fn delta(&self) -> Option<f64> {
        Some(self.log_score? - self.baseline_log_score?)
    }
}

/// Every pair of traces that will be differenced must sit on the same
/// observation axis — same count, same times, and the same STREAMS at each of
/// those times.
///
/// `paired_delta` zips by index. That is safe only if index `k` means the same
/// observation on both sides, which `n_scored()` alone does not establish: a
/// hole in one series or a different `t0` gives two traces of equal length at
/// different times. Differencing those is not a comparison at any level of
/// display, so this runs in the preflight rather than in one renderer.
///
/// Equal times are not enough either (gh#570). A step's `log_score` is the
/// JOINT score across the streams scored there, so a trace covering two
/// districts and one covering a single district can agree on every `t` and
/// still be scoring different quantities; their difference then reads as
/// evidence about the models when it is mostly the second district's absence.
///
/// The message names the traces AND what differs: this refuses pairs that
/// previously rendered, and "not comparable" with no detail is not actionable
/// for someone reading it against a live outbreak.
fn check_shared_observation_axis(rows: &[Row]) -> Result<(), String> {
    let Some(first) = rows.first() else { return Ok(()) };
    for r in rows.iter().skip(1) {
        for (k, (a, b)) in first.trace.steps.iter().zip(&r.trace.steps).enumerate() {
            if a.t != b.t {
                return Err(format!(
                    "'{}' and '{}' are not on the same observation axis: scored \
                     step {} is t={} in '{}' and t={} in '{}'.\n       \
                     Δelpd pairs observations by position, so differencing these \
                     would compare unlike times.\n       \
                     Re-score both models on the same observation set \
                     (check for a hole in one series, or a different t0).",
                    first.name, r.name, k + 1, a.t, first.name, b.t, r.name,
                ));
            }
            let (sa, sb) = (step_streams(a), step_streams(b));
            if sa == sb {
                continue;
            }
            // A v1 trace carries no `per_stream` at all. Two such traces have
            // nothing to check and are allowed through; one of each cannot be
            // checked, and an unverifiable stream set is not a verified one.
            let detail = if sa.is_empty() || sb.is_empty() {
                let (with, without) = if sa.is_empty() { (&r.name, &first.name) }
                                      else { (&first.name, &r.name) };
                format!(
                    "'{without}' has no per-stream breakdown at that step, so its \
                     joint score cannot be shown to cover the same streams as \
                     '{with}' ({}).\n       \
                     Re-derive '{without}' — a v1 `prequential.json` predates \
                     the per-stream breakdown.",
                    fmt_stream_set(if sa.is_empty() { &sb } else { &sa }),
                )
            } else {
                format!(
                    "'{}' scored {} there and '{}' scored {}.\n       \
                     The step's log score is the JOINT score over those streams, \
                     so differencing them compares unlike quantities.\n       \
                     Re-score both models on the same set of observation streams.",
                    first.name, fmt_stream_set(&sa), r.name, fmt_stream_set(&sb),
                )
            };
            return Err(format!(
                "'{}' and '{}' did not score the same streams at t={}: {detail}",
                first.name, r.name, a.t,
            ));
        }
    }
    Ok(())
}

/// Refuse a comparison whose fits were bound to different observed data
/// (gh#713).
///
/// `compare @a @b` re-filters each fit at its own θ̂ and differences the
/// scores. If the two fits read different data — a corrected case series, a
/// stream added between runs, a file edited in place — the difference carries
/// both effects and the table attributes all of it to the models. Nothing else
/// in the pipeline catches this: each fit is internally consistent, and the
/// derived traces agree on T_score, times and stream names, because those come
/// from the same model.
///
/// Rows with no data identity ([`DataIdentity::Unknown`]) are not evidence
/// either way and are skipped; [`unchecked_data_note`] names them.
fn check_shared_data(entries: &[(&str, &DataIdentity)]) -> Result<(), String> {
    use std::collections::BTreeSet;
    let known: Vec<(&str, &std::collections::BTreeMap<String, String>)> = entries
        .iter()
        .filter_map(|(name, d)| match d {
            DataIdentity::Digests(m) => Some((*name, m)),
            DataIdentity::Unknown(_) => None,
        })
        .collect();
    // Equality is transitive, so every row is checked against the first that
    // carries digests.
    let Some(&(ref_name, ref_map)) = known.first() else { return Ok(()) };
    for &(name, map) in known.iter().skip(1) {
        let mut streams: BTreeSet<&str> = ref_map.keys().map(|s| s.as_str()).collect();
        streams.extend(map.keys().map(|s| s.as_str()));
        let differing: Vec<String> = streams.iter().filter_map(|s| {
            match (ref_map.get(*s), map.get(*s)) {
                (Some(a), Some(b)) if a == b => None,
                (Some(a), Some(b)) => Some(format!(
                    "'{s}' ({} in '{ref_name}', {} in '{name}')",
                    short_digest(a), short_digest(b))),
                (Some(_), None) => Some(format!("'{s}' (bound by '{ref_name}' only)")),
                (None, Some(_)) => Some(format!("'{s}' (bound by '{name}' only)")),
                (None, None) => None,
            }
        }).collect();
        if !differing.is_empty() {
            return Err(format!(
                "'{ref_name}' and '{name}' were fit to different observed data: \
                 {}.\n       \
                 Δelpd across them is part model difference and part data \
                 difference, and the table cannot separate the two.\n       \
                 Re-fit both on the same observations, or pass \
                 --allow-data-mismatch to render the comparison as confounded.",
                differing.join("; "),
            ));
        }
    }
    Ok(())
}

/// A content hash abbreviated for a message: enough to tell two apart, short
/// enough to read. Character-wise, so a short test or hand-written digest does
/// not panic on a byte slice.
fn short_digest(hex: &str) -> String {
    hex.chars().take(8).collect()
}

/// The note naming the rows excluded from the data check and why, or `None`
/// when every row carried digests. Separate from [`check_shared_data`] so the
/// scope of the check is stated even when the check itself refuses.
fn unchecked_data_note(entries: &[(&str, &DataIdentity)]) -> Option<String> {
    let skipped: Vec<String> = entries.iter().filter_map(|(name, d)| match d {
        DataIdentity::Digests(_) => None,
        DataIdentity::Unknown(why) => Some(format!("'{name}' ({why})")),
    }).collect();
    (!skipped.is_empty()).then(|| format!(
        "not checked for a shared observation set: {}. \
         A difference against such a row may be a data difference, not a model \
         difference.",
        skipped.join("; "),
    ))
}

/// The set of stream names scored at one step. Empty for a v1 trace, which has
/// no per-stream breakdown at all.
fn step_streams(s: &sim::inference::prequential::PrequentialStep)
    -> std::collections::BTreeSet<&str>
{
    s.per_stream.iter().map(|x| x.stream.as_str()).collect()
}

fn fmt_stream_set(s: &std::collections::BTreeSet<&str>) -> String {
    format!("{{{}}}", s.iter().copied().collect::<Vec<_>>().join(", "))
}

/// Project every candidate against the baseline, step by step and stream by
/// stream.
///
/// The preflight ([`check_shared_observation_axis`]) has already established
/// that paired steps carry the same `t`; the check is repeated here because
/// this function is also the one that would turn a mismatch into a plot that
/// looks fine, and it must not depend on a caller having run the preflight.
fn pointwise_rows(rows: &[Row], base_idx: usize) -> Result<Vec<PointwiseRow>, String> {
    let base = &rows[base_idx];
    let mut out = Vec::new();
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
            out.push(PointwiseRow {
                model: r.name.clone(),
                baseline: base.name.clone(),
                t: step.t,
                scope: "joint",
                stream: String::new(),
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
            let find = |ss: &[sim::inference::prequential::StreamScore], n: &str| {
                ss.iter().find(|s| s.stream == n).map(|s| s.log_score)
            };
            for n in names {
                out.push(PointwiseRow {
                    model: r.name.clone(),
                    baseline: base.name.clone(),
                    t: step.t,
                    scope: "stream",
                    stream: n.to_string(),
                    log_score: find(&step.per_stream, n),
                    baseline_log_score: find(&bstep.per_stream, n),
                });
            }
        }
    }
    Ok(out)
}

/// Render the pointwise rows as a TSV. An absent score is an EMPTY cell, never
/// `NaN` or `0` — the reader must be able to tell "this model did not score
/// here" from "this model scored zero here".
fn render_pointwise_tsv(rows: &[PointwiseRow]) -> String {
    let cell = |v: Option<f64>| v.map(|x| format!("{:.10}", x)).unwrap_or_default();
    let mut s = String::new();
    s.push_str("# camdl compare --pointwise: per-observation log predictive scores\n");
    s.push_str("# delta_log_score = log_score - baseline_log_score, in nats.\n");
    s.push_str("# scope=joint is the cross-stream score; scope=stream is one \
                stream's own.\n");
    s.push_str("# An empty cell means that side did not score that stream at \
                that time.\n");
    s.push_str("model\tbaseline\tt\tscope\tstream\tlog_score\t\
                baseline_log_score\tdelta_log_score\n");
    for r in rows {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.model, r.baseline, fmt_pointwise_time(r.t), r.scope, r.stream,
            cell(r.log_score), cell(r.baseline_log_score), cell(r.delta()),
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

fn write_pointwise(rows: &[Row], base_idx: usize, path: &Path) -> Result<usize, String> {
    let pw = pointwise_rows(rows, base_idx)?;
    let text = render_pointwise_tsv(&pw);
    std::fs::write(path, text)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(pw.len())
}

/// Resolve a single comparison input to a `PrequentialTrace`. Two paths:
///
/// 1. **Explicit prequential** (kept, tried first): a `.json` file that exists,
///    or a directory holding `prequential.json` — read+parsed as-is. Preserves
///    `pfilter --save-prequential` and stage-dir inputs; the derive settings do
///    not touch it.
/// 2. **Fit handle** (Phase 2a): `@label` / hash prefix / run dir / `fit.toml`.
///    The prequential is DERIVED by invoking the canonical `camdl pfilter` at
///    the fit's sealed θ̂ — never by reimplementing the filter (the obs-model
///    assembly is already triplicated; a fourth copy is forbidden).
fn load_trace(
    path: &str,
    derive: DeriveSettings,
    selection: Option<&ChainSelection>,
) -> Result<(PrequentialTrace, DataIdentity), String> {
    if let Some(trace) = try_load_explicit_prequential(path)? {
        // A trace records scores and times, never which observations produced
        // them — so this row cannot take part in the data-identity check.
        return Ok((trace, DataIdentity::Unknown(
            "read as an explicit prequential.json, which records no bound data".into())));
    }
    // Not an explicit prequential → treat as a fit handle and derive.
    match crate::fit::handle::resolve_fit(path) {
        Ok(resolved) => {
            let data = fit_data_identity(&resolved);
            derive_prequential(&resolved, derive, selection).map(|t| (t, data))
        }
        Err(resolve_err) => Err(format!(
            "'{path}' is neither a prequential trace nor a resolvable fit handle.\n  \
             - as a fit handle: {resolve_err}\n  \
             - as a prequential path: no prequential.json at '{path}' (or \
             '{path}/prequential.json') — run `camdl pfilter --save-prequential` \
             or `camdl fit run` with a pfilter stage to generate one."
        )),
    }
}

/// Refuse to derive a prequential for a fit that did not run on the backend the
/// derive path filters with (gh#729).
///
/// [`derive_prequential`] invokes `camdl pfilter`, a bootstrap particle filter
/// over the chain-binomial forward process. That is the right filter for a
/// chain-binomial fit and the wrong one for any other: an ODE fit's θ̂ describes
/// a deterministic trajectory, and scoring it through a stochastic
/// chain-binomial predictive produces a number that is neither the fit's own
/// likelihood nor a like-for-like predictive score — while the table presents
/// it as that fit's prequential. A label would not fix it; a reader told to
/// discount a column will forget to.
///
/// No override: the honest ODE prequential is gh#312. Until then an ODE fit can
/// still be compared by supplying an explicit `prequential.json` scored under
/// its own backend.
fn check_derivable_backend(
    backend: crate::run_meta::InferenceBackend,
    segment: &Path,
) -> Result<(), String> {
    use crate::run_meta::InferenceBackend;
    match backend {
        InferenceBackend::ChainBinomial => Ok(()),
        other => Err(format!(
            "the fit at {} ran its terminal stage on the `{}` backend, but \
             `compare` derives a prequential by invoking `camdl pfilter`, a \
             bootstrap particle filter over the `chain_binomial` process.\n  \
             The derived scores would come from a different forward process \
             than the one the fit used, and the table would present them as \
             this fit's own.\n  \
             Score it under its own backend and pass the resulting \
             prequential.json instead. The derived path for `{}` fits is \
             tracked in gh#312 (ODE prequential).",
            segment.display(), other.as_str(), other.as_str(),
        )),
    }
}

/// The per-stream data digests a resolved fit recorded when it ran, read from
/// its provenance sidecar (`fit.meta.json`). Not re-hashed here: the files may
/// have moved or changed since, and the question is what the FIT was bound to.
///
/// The sidecar is derived provenance rather than a source of truth, which is
/// the right strength for a preflight — it is a faithful projection of inputs
/// already hashed into the fit's identity. A fit without one (an older run, a
/// hand-assembled directory) is reported as unchecked rather than refused.
fn fit_data_identity(resolved: &ResolvedFit) -> DataIdentity {
    match crate::run_meta::read_fit_sidecar(&resolved.segment) {
        Some(s) if !s.data_hashes.is_empty() => DataIdentity::Digests(s.data_hashes),
        Some(_) => DataIdentity::Unknown(
            "its fit.meta.json records no data hashes".into()),
        None => DataIdentity::Unknown(format!(
            "no fit.meta.json under {}", resolved.segment.display())),
    }
}

/// Try to read `path` as an explicit prequential artifact. Returns `Ok(None)`
/// (fall through to fit-handle derivation) when it is not one: a non-existent
/// path, a `@label` / hash prefix, a `.toml`, or a directory with no
/// `prequential.json`. Only a `.json` file or a `<dir>/prequential.json` is
/// read — so a `fit.toml` handle is never mis-parsed as a trace.
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
    Ok(Some(trace))
}

/// Derive a prequential trace from a sealed fit by invoking the canonical
/// `camdl pfilter` at the fit's winning θ̂. This routes through the one
/// production filter + obs-model assembly rather than rebuilding it, so a
/// derived trace can never silently diverge from a hand-run `pfilter`.
///
/// Steps: (a) θ̂ from the winning stage as a flat params TOML; (b) the model
/// (archived IR if present, else the loose source the config names); (c) the
/// fit's data streams as `--data NAME=PATH`; (d) run `pfilter
/// --save-prequential` to a temp stem; (e) read back the `{stem}.json` trace.
/// All temp files live in the system temp dir and are best-effort cleaned up.
fn derive_prequential(
    resolved: &ResolvedFit,
    derive: DeriveSettings,
    selection: Option<&ChainSelection>,
) -> Result<PrequentialTrace, String> {
    let segment = &resolved.segment;
    let config = &resolved.config;

    // gh#729: the filter this shells out to is chain-binomial. The θ̂ below
    // comes from the TERMINAL stage (both `resolve_posterior_draws` and
    // `winner_params_toml` pick it), so that stage's backend is the one the
    // derived score must match.
    if let Some(terminal) = config.stages.values().last() {
        check_derivable_backend(terminal.backend(), segment)?;
    }

    // (a) θ̂ — the plug-in point the prequential is scored at. Routed through
    // the draws-cloud authority (`resolve_posterior_draws`), NOT a per-method
    // point-estimate file: a Bayesian fit (PGAS/PMMH/MH) writes no
    // `final_params.toml` — its θ̂ is the posterior MEAN over `draws.tsv`; only
    // an optimizer fit (IF2/NLopt) has a single winner file. The headline
    // `compare @pgas_a @pgas_b` workflow used to dead-end on the missing file.
    let params_toml = point_estimate_params_toml(segment, selection)?;

    // (b) model — prefer the self-contained archived IR (Phase 1a), else the
    // loose source the config names (recompiled by pfilter). config.model.camdl
    // is already absolute (resolved against the fit.toml dir at load).
    let archived = segment.join("model.ir.json");
    let model_path: PathBuf = if archived.is_file() {
        archived
    } else {
        PathBuf::from(&config.model.camdl)
    };

    // (c) data — stream → absolute path. The config's data paths were made
    // absolute (relative to the original fit.toml) at load, so they pass to the
    // child unchanged.
    let data = config.data.as_ref().ok_or_else(|| format!(
        "fit at {} has no [data] block — there are no observations to score, so a \
         prequential trace cannot be derived.\n  Provide an explicit \
         prequential.json, or compare fits that bind data.",
        segment.display()))?;
    let streams = resolve_streams(data, &model_path)?;
    if streams.is_empty() {
        return Err(format!(
            "fit at {} has a [data] block with no observation streams.",
            segment.display()));
    }

    // Temp files: unique per process + fit segment so concurrent compares don't
    // collide. STEM has no `.` so pfilter's `{stem}.json` / `{stem}.tsv` are
    // unambiguous.
    let seg_slug = segment
        .file_name()
        .map(|s| s.to_string_lossy().replace(|c: char| !c.is_ascii_alphanumeric(), "_"))
        .unwrap_or_else(|| "fit".into());
    let base = std::env::temp_dir()
        .join(format!("camdl_compare_{}_{}", std::process::id(), seg_slug));
    let theta_path = base.with_extension("theta.toml");
    let preq_stem = base.to_string_lossy().into_owned() + "_preq";
    let preq_json = format!("{preq_stem}.json");
    let preq_tsv = format!("{preq_stem}.tsv");
    let cleanup = || {
        let _ = std::fs::remove_file(&theta_path);
        let _ = std::fs::remove_file(&preq_json);
        let _ = std::fs::remove_file(&preq_tsv);
    };

    if let Err(e) = std::fs::write(&theta_path, &params_toml) {
        cleanup();
        return Err(format!("writing temp params {}: {}", theta_path.display(), e));
    }

    // (d) run the canonical filter.
    let exe = std::env::current_exe()
        .map_err(|e| { cleanup(); format!("cannot locate the running camdl binary: {e}") })?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("pfilter")
        .arg(&model_path)
        .arg("--params").arg(&theta_path)
        .arg("--save-prequential").arg(&preq_stem)
        .arg("--particles").arg(derive.particles.to_string())
        .arg("--seed").arg(derive.seed.to_string())
        .env("CAMDL_SKIP_VERSION_CHECK", "1");
    for (name, abs_path) in &streams {
        cmd.arg("--data").arg(format!("{name}={abs_path}"));
    }
    // gh#634: forward the fit's conditioning window, or the derived
    // prequential scores a window the fit never scored (and pfilter's W329
    // guard then recommends setting condition_from — which the fit toml
    // already sets, sending the user to the wrong layer). The CLI flag
    // grammar is the transport (gh#621): a bare spec is the all-streams
    // default, LABEL=SPEC a per-stream shadow.
    if let Some(cond) = &config.condition_from {
        use crate::fit::config_v2::{ConditionFrom, CONDITION_FROM_DEFAULT_KEY};
        match cond {
            ConditionFrom::All(spec) => {
                cmd.arg("--condition-from").arg(spec);
            }
            ConditionFrom::PerStream(map) => {
                for (label, spec) in map {
                    if label == CONDITION_FROM_DEFAULT_KEY {
                        cmd.arg("--condition-from").arg(spec);
                    } else {
                        cmd.arg("--condition-from").arg(format!("{label}={spec}"));
                    }
                }
            }
        }
    }
    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => { cleanup(); return Err(format!("spawning `camdl pfilter`: {e}")); }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let code = output.status.code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".into());
        cleanup();
        return Err(format!(
            "deriving prequential via `camdl pfilter` failed (exit {code}) for fit {}:\n{}",
            segment.display(),
            stderr.trim()));
    }

    // (e) read back the trace.
    let text = match std::fs::read_to_string(&preq_json) {
        Ok(t) => t,
        Err(e) => { cleanup(); return Err(format!(
            "`camdl pfilter` succeeded but its prequential output {preq_json} could not be read: {e}")); }
    };
    let trace = serde_json::from_str::<PrequentialTrace>(&text)
        .map_err(|e| format!("parsing derived prequential {preq_json}: {e}"));
    cleanup();
    trace
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
    fn warn(&self) {
        if self.per_fit.is_empty() {
            return;
        }
        if self.cohort_wide {
            // Every entry carries the same drop set — warn once.
            if let Some(sel) = self.per_fit.values().next() {
                sel.warn_requested();
            }
        } else {
            for (fit, sel) in &self.per_fit {
                sel.warn_requested_for_fit(fit);
            }
            crate::chain_selection::eprint_bias_caveat();
        }
    }
}

/// Parse repeatable `--exclude-chains` tokens, bound to the resolved model
/// `names`. Each token is `@fit:ids` / `fit:ids` (one fit) or bare `ids`
/// (cohort-wide, applied to every fit). Mixing the two forms is rejected as
/// ambiguous. A named fit must match EXACTLY ONE model — an unknown or ambiguous
/// name is a hard error, so a per-fit token can never silently bind nowhere or
/// to the wrong fit.
fn parse_cohort_exclude(
    tokens: &[String],
    names: &[String],
) -> Result<CohortChainSelection, String> {
    if tokens.is_empty() {
        return Ok(CohortChainSelection::default());
    }
    let has_keyed = tokens.iter().any(|t| t.contains(':'));
    let has_bare = tokens.iter().any(|t| !t.contains(':'));
    if has_keyed && has_bare {
        return Err(
            "--exclude-chains: mixing cohort-wide ids and per-fit `@fit:ids` is ambiguous; \
             give either bare ids (dropped from every fit) or @fit:ids (per fit), not both"
                .to_string(),
        );
    }

    let mut per_fit = std::collections::BTreeMap::new();

    if has_bare {
        // Cohort-wide: union the bare tokens into one selection, applied to all.
        let sel = ChainSelection::parse_exclude(&tokens.join(","))
            .map_err(|e| format!("--exclude-chains: {e}"))?;
        for name in names {
            per_fit.insert(name.clone(), sel.clone());
        }
        return Ok(CohortChainSelection { per_fit, cohort_wide: true });
    }

    // Per-fit: `fit:ids`. Split on the LAST ':' so a fit name may contain one.
    // The name is matched VERBATIM against the compared fits' names — a
    // handle-referenced fit is named `@a`, a path-referenced one `ctl_rm.toml`.
    // The `@` is part of a handle name, NOT a per-fit sigil to prepend.
    for tok in tokens {
        let (fit, ids) = tok
            .rsplit_once(':')
            .expect("every token has ':' in the per-fit branch");
        if fit.is_empty() {
            return Err(format!("--exclude-chains '{tok}': missing fit name before ':'"));
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
                "--exclude-chains: no compared fit named '{fit}' (fits: {}){hint}",
                names.join(", ")
            ));
        }
        if n_match > 1 {
            return Err(format!(
                "--exclude-chains: fit name '{fit}' is ambiguous ({n_match} compared fits \
                 share it) — give each `[[model]]` a unique name via --config"
            ));
        }
        let sel = ChainSelection::parse_exclude(ids)
            .map_err(|e| format!("--exclude-chains @{fit}: {e}"))?;
        if per_fit.insert(fit.to_string(), sel).is_some() {
            return Err(format!("--exclude-chains: fit '{fit}' listed more than once"));
        }
    }
    Ok(CohortChainSelection { per_fit, cohort_wide: false })
}

/// θ̂ for the prequential as a flat params TOML, routed through the draws-cloud
/// authority. A Bayesian fit (its terminal stage wrote a `draws.tsv`) plugs in
/// the posterior MEAN over the cloud; an optimizer fit (no cloud) plugs in its
/// winner file (`final_params.toml`, via `winner_params_toml`). This is the
/// "resolve by artifact, not by method name" rule — the headline `compare
/// @pgas_a @pgas_b` Bayesian comparison was previously dead-ending on the
/// IF2-only `final_params.toml`.
///
/// A `--exclude-chains` selection (gh#417) is attached to the resolved draws
/// ref, so the posterior mean is taken over the RETAINED cloud — the same
/// filter `fit predict`/`fit summary` apply. An optimizer fit has no cloud, so
/// the selection has nothing to filter there.
fn point_estimate_params_toml(
    segment: &Path,
    selection: Option<&ChainSelection>,
) -> Result<String, String> {
    match crate::posterior_draws::resolve_posterior_draws(&segment.to_string_lossy(), None) {
        Ok(pdraws) => posterior_mean_params_toml(&pdraws.with_selection(selection.cloned())),
        // No posterior cloud → an optimizer fit; its θ̂ is the single winner.
        Err(_) => crate::fit::fit_summary::winner_params_toml(segment, None),
    }
}

/// The posterior MEAN of every parameter column in a fit's draws cloud as a flat
/// params TOML — the plug-in point a prequential is scored at for a Bayesian
/// fit. Every model parameter is a column (estimated + fixed); a fixed column is
/// constant, so its mean is the fixed value. Columns are emitted in sorted order
/// for determinism.
///
/// Reads through the shared draws authority ([`PosteriorDrawsRef`]) so an
/// attached `--exclude-chains` selection is applied ONCE, here, before the mean
/// — the same seam `fit predict`/`fit summary` use, rather than a second raw
/// draws reader. The `(chain, draw)` key columns are dropped by the keyed
/// loader, never emitted as parameters (gh#322).
fn posterior_mean_params_toml(pref: &PosteriorDrawsRef) -> Result<String, String> {
    let (rows, _info) = pref.load_params_with_info()?;
    let n = rows.len();
    if n == 0 {
        return Err(format!("no posterior draws in {}", pref.draws_path.display()));
    }
    // Sum each parameter over the retained rows in file order — one accumulator
    // per name, so the intra-row (HashMap) order is irrelevant; the BTreeMap
    // keeps the emit order sorted and deterministic.
    let mut sums: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    for row in &rows {
        for (name, v) in row {
            *sums.entry(name.clone()).or_insert(0.0) += *v;
        }
    }
    let mut out = String::new();
    out.push_str("# camdl compare: posterior-mean point estimate (θ̂) for the prequential\n");
    out.push_str(&format!("# source: {} ({n} draws)\n\n", pref.draws_path.display()));
    for (name, sum) in &sums {
        out.push_str(&format!(
            "{} = {}\n",
            name,
            crate::fit::runner::format_param_value(sum / n as f64)
        ));
    }
    Ok(out)
}

/// Resolve a fit's `[data]` spec to a stream-name → absolute-path map for
/// `--data NAME=PATH`. The per-stream `observations` map is used directly; the
/// single-file shorthand (`file = "..."`) is expanded via the existing
/// [`DataSpec::effective_observations`] seam, which needs the model's declared
/// observation-stream names (loaded from `model_path`).
fn resolve_streams(
    data: &crate::fit::config_v2::DataSpec,
    model_path: &Path,
) -> Result<indexmap::IndexMap<String, String>, String> {
    if !data.observations.is_empty() {
        return Ok(data.observations.clone());
    }
    if data.file.is_some() {
        let (model, _) = crate::util::load_model(&model_path.to_string_lossy())
            .map_err(|e| format!(
                "loading model {} to expand the single-file [data] shorthand: {e}",
                model_path.display()))?;
        let names: Vec<String> = model.observations.iter().map(|o| o.name.clone()).collect();
        return data.effective_observations(&names);
    }
    Err("the fit's [data] block has neither `observations` nor `file`".to_string())
}

/// Paired Δ = sum_t (a_t − b_t); paired SE = sqrt(T · Var_t(a_t − b_t)).
/// Returns (delta, se) or (NaN, NaN) if horizons mismatch.
fn paired_delta(a: &PrequentialTrace, b: &PrequentialTrace, field: Field)
    -> (f64, f64)
{
    if a.n_scored() != b.n_scored() || a.n_scored() == 0 {
        return (f64::NAN, f64::NAN);
    }
    let diffs: Vec<f64> = a.steps.iter().zip(&b.steps)
        .map(|(x, y)| match field {
            Field::LogScore => x.log_score - y.log_score,
            Field::Crps     => x.crps - y.crps,
        })
        .collect();
    let t = diffs.len() as f64;
    let delta: f64 = diffs.iter().sum();
    let mean = delta / t;
    let var = if t > 1.0 {
        diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (t - 1.0)
    } else { 0.0 };
    let se = (t * var).sqrt();
    (delta, se)
}

#[derive(Copy, Clone)]
enum Field { LogScore, Crps }

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
    if lr >= 1000.0 || lr < 0.001 {
        format!("{:.2e}", lr)
    } else {
        format!("{:.3}", lr)
    }
}

/// Render the comparison as an aligned text table. Returns the whole rendering
/// (including its trailing newline) rather than printing, so every line the
/// reader sees — footers, caveats, warnings — is assertable in a unit test.
fn render_table(rows: &[Row], base_idx: usize, metrics: &[String], t_mismatch: bool) -> String {
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");

    // Δelpd (nats) is the primary machine-readable column; "evidence"
    // (decibans + Jeffreys label) is the human-interpretable alongside —
    // see docs/dev/proposals/2026-04-23-evidence-in-decibans.md §Scope.
    let mut header = vec!["Model".to_string(), "T_score".into(), "elpd".into(),
        "Δelpd".into(), "LR".into(), "se(Δ)".into(), "evidence".into()];
    if want_crps { header.push("crps".into()); header.push("Δcrps".into()); }
    if want_pit  { header.push("PIT_cov90".into()); }

    let base = &rows[base_idx].trace;
    // Render order: ascending Δelpd (worst → best). Baseline drops in
    // at its natural Δelpd = 0 slot. Best candidate lands at the
    // bottom — reader's eye sees the recommended model last. When
    // T_score mismatches block per-row deltas, we can't sort
    // meaningfully, so leave the input order alone in that path.
    let order: Vec<usize> = if t_mismatch {
        (0..rows.len()).collect()
    } else {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| {
            let da = if a == base_idx { 0.0 }
                else { paired_delta(&rows[a].trace, base, Field::LogScore).0 };
            let db = if b == base_idx { 0.0 }
                else { paired_delta(&rows[b].trace, base, Field::LogScore).0 };
            // NaN delta → place row at the top (worst-case ranking).
            match (da.is_nan(), db.is_nan()) {
                (true, true)  => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        idx
    };
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    let mut any_evidence = false;
    for &i in &order {
        let r = &rows[i];
        let elpd = r.trace.elpd();
        let mut row = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", elpd),
        ];
        if i == base_idx {
            row.push("—".into());   // Δelpd
            row.push("—".into());   // LR
            row.push("—".into());   // se(Δ)
            row.push("—".into());   // evidence (dB + Jeffreys label)
        } else if t_mismatch {
            row.push("—".into());
            row.push("—".into());
            row.push("—".into());
            row.push("—".into());
        } else {
            let (d, se) = paired_delta(&r.trace, base, Field::LogScore);
            row.push(format!("{:+.2}", d));
            row.push(fmt_lr(d.exp()));
            row.push(format!("{:.2}", se));
            row.push(crate::evidence::evidence_cell(d, se));
            any_evidence |= d.is_finite();
        }
        if want_crps {
            row.push(format!("{:.3}", r.trace.mean_crps()));
            if i == base_idx || t_mismatch {
                row.push("—".into());
            } else {
                let (d, _) = paired_delta(&r.trace, base, Field::Crps);
                let mean_diff = d / r.trace.n_scored() as f64;
                row.push(format!("{:+.3}", mean_diff));
            }
        }
        if want_pit {
            row.push(format!("{:.2}", r.trace.pit_coverage(0.90)));
        }
        body.push(row);
    }

    let widths: Vec<usize> = (0..header.len()).map(|c| {
        let h = header[c].chars().count();
        let b = body.iter().map(|r| r[c].chars().count()).max().unwrap_or(0);
        h.max(b)
    }).collect();

    let sep = |cols: &[usize]| -> String {
        let total: usize = cols.iter().sum::<usize>() + 3 * (cols.len() - 1);
        "─".repeat(total)
    };

    let mut out = String::new();
    out.push_str(&fmt_table_row(&header, &widths));
    out.push_str(&format!("{}\n", sep(&widths)));
    for row in &body {
        out.push_str(&fmt_table_row(row, &widths));
    }

    out.push('\n');
    out.push_str(&format!("Scored steps: {} (t0={}).  Baseline: {}.\n",
        base.n_scored(), base.t0, rows[base_idx].name));
    if !t_mismatch {
        out.push_str("Sorted by Δelpd ascending — best-supported model at the bottom.\n");
    }
    if any_evidence {
        out.push_str(JEFFREYS_SCALE_NOTE);
        out.push('\n');
        if let Some(caveat) = se_caveat(base.n_scored()) {
            out.push_str(&caveat);
            out.push('\n');
        }
    }
    for line in warning_lines(rows, t_mismatch) {
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// Everything below the table that the reader must not miss: the suppressed-Δ
/// notice, an overridden data mismatch and what the data check could not cover,
/// the miscalibration flags, each trace's own warnings, and the optimism
/// caveat.
///
/// One function, called by every renderer, because the failure this guards is
/// asymmetry: the table warned about a miscalibrated model and the markdown
/// someone pasted into a report did not. A new warning added here appears in
/// every format by construction.
fn warning_lines(rows: &[Row], t_mismatch: bool) -> Vec<String> {
    let mut lines = Vec::new();
    if t_mismatch {
        lines.push("⚠ T_score differs across models — Δ columns suppressed \
            (--allow-mismatched-horizon was set).".to_string());
    }
    // A comparison that reaches a renderer with a failed data check was
    // rendered under `--allow-data-mismatch`; the confound belongs in the
    // artifact, not only in the terminal session that asked for it.
    let data_entries: Vec<(&str, &DataIdentity)> =
        rows.iter().map(|r| (r.name.as_str(), &r.data)).collect();
    if let Err(e) = check_shared_data(&data_entries) {
        lines.push(format!("⚠ {e}"));
    }
    if let Some(note) = unchecked_data_note(&data_entries) {
        lines.push(format!("ⓘ {note}"));
    }
    // PIT warnings — flag clear miscalibration.
    for r in rows {
        if let Some(w) = pit_coverage_warning(r) {
            lines.push(format!("⚠ {w}"));
        }
    }
    // Propagate trace-level warnings.
    for r in rows {
        for w in &r.trace.warnings {
            lines.push(format!("ⓘ {}: {:?}", r.name, w));
        }
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

/// The `⚠` line for a row whose 90% predictive interval covers far less than
/// nominal — the plug-in overconfidence tell. `None` when coverage is fine.
/// Shared by every renderer: a warning one format prints and another drops is
/// the failure mode this exists to prevent.
fn pit_coverage_warning(r: &Row) -> Option<String> {
    let cov = r.trace.pit_coverage(0.90);
    (cov < 0.70).then(|| format!(
        "{}: PIT 90%-coverage {:.2} (nominal 0.90) — likely overconfident.",
        r.name, cov))
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
fn render_md(rows: &[Row], base_idx: usize, metrics: &[String], t_mismatch: bool) -> String {
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");

    let mut out = String::new();
    let mut header = vec!["Model", "T_score", "elpd", "Δelpd", "LR", "se(Δ)", "evidence"];
    if want_crps { header.push("crps"); header.push("Δcrps"); }
    if want_pit  { header.push("PIT_cov90"); }
    out.push_str(&format!("| {} |\n", header.join(" | ")));
    out.push_str(&format!("|{}|\n",
        header.iter().map(|_| "---").collect::<Vec<_>>().join("|")));

    let base = &rows[base_idx].trace;
    // Same render order as the table renderer: ascending Δelpd
    // (best-supported last). When T_score mismatches block per-row
    // deltas, leave input order alone.
    let order: Vec<usize> = if t_mismatch {
        (0..rows.len()).collect()
    } else {
        let mut idx: Vec<usize> = (0..rows.len()).collect();
        idx.sort_by(|&a, &b| {
            let da = if a == base_idx { 0.0 }
                else { paired_delta(&rows[a].trace, base, Field::LogScore).0 };
            let db = if b == base_idx { 0.0 }
                else { paired_delta(&rows[b].trace, base, Field::LogScore).0 };
            match (da.is_nan(), db.is_nan()) {
                (true, true)  => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        idx
    };
    let mut any_evidence = false;
    for &i in &order {
        let r = &rows[i];
        let mut cells: Vec<String> = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", r.trace.elpd()),
        ];
        if i == base_idx || t_mismatch {
            cells.push("—".into());  // Δelpd
            cells.push("—".into());  // LR
            cells.push("—".into());  // se(Δ)
            cells.push("—".into());  // evidence
        } else {
            let (d, se) = paired_delta(&r.trace, base, Field::LogScore);
            cells.push(format!("{:+.2}", d));
            cells.push(fmt_lr(d.exp()));
            cells.push(format!("{:.2}", se));
            cells.push(crate::evidence::evidence_cell(d, se));
            any_evidence |= d.is_finite();
        }
        if want_crps {
            cells.push(format!("{:.3}", r.trace.mean_crps()));
            if i == base_idx || t_mismatch {
                cells.push("—".into());
            } else {
                let (d, _) = paired_delta(&r.trace, base, Field::Crps);
                let mean_diff = d / r.trace.n_scored() as f64;
                cells.push(format!("{:+.3}", mean_diff));
            }
        }
        if want_pit { cells.push(format!("{:.2}", r.trace.pit_coverage(0.90))); }
        out.push_str(&format!("| {} |\n", cells.join(" | ")));
    }
    if !t_mismatch {
        out.push('\n');
        out.push_str("_Sorted by Δelpd ascending — best-supported model at the bottom._\n");
    }
    if any_evidence {
        out.push('\n');
        out.push_str(&format!("_{JEFFREYS_SCALE_NOTE}_\n"));
        if let Some(caveat) = se_caveat(base.n_scored()) {
            out.push('\n');
            out.push_str(&format!("_{caveat}_\n"));
        }
    }
    // The same block the text table prints, as a markdown list so each warning
    // stays its own line when the document is rendered.
    let warnings = warning_lines(rows, t_mismatch);
    if !warnings.is_empty() {
        out.push('\n');
        for line in warnings {
            out.push_str(&format!("- {line}\n"));
        }
    }
    out
}

/// Render the comparison as JSON. Returns the document rather than printing it.
fn render_json(rows: &[Row], base_idx: usize, metrics: &[String]) -> String {
    use serde_json::json;
    let base = &rows[base_idx].trace;
    let entries: Vec<serde_json::Value> = rows.iter().enumerate().map(|(i, r)| {
        let (d_elpd, se_elpd) = if i == base_idx { (f64::NAN, f64::NAN) }
            else { paired_delta(&r.trace, base, Field::LogScore) };
        let (d_crps, _) = if i == base_idx { (f64::NAN, f64::NAN) }
            else { paired_delta(&r.trace, base, Field::Crps) };
        let mean_dcrps = if r.trace.n_scored() == 0 { f64::NAN }
            else { d_crps / r.trace.n_scored() as f64 };
        let lr = if d_elpd.is_finite() { d_elpd.exp() } else { f64::NAN };
        // Evidence: Δelpd (nats) → decibans + a label. Derived field for
        // human-interpretable consumption; nats remain the primary
        // machine-readable quantity (delta_elpd). The label is
        // `within_noise` when |Δ| < 2·se(Δ), else the Jeffreys tier —
        // consumers reading the tier without the SE get the same gate the
        // table shows. See
        // docs/dev/proposals/2026-04-23-evidence-in-decibans.md.
        let (d_elpd_db, evidence_label) = if d_elpd.is_finite() {
            let db = d_elpd * crate::evidence::NATS_TO_DB;
            (option_finite(db),
             serde_json::json!(crate::evidence::evidence_label(d_elpd, se_elpd)))
        } else {
            (serde_json::Value::Null, serde_json::Value::Null)
        };
        json!({
            "name": r.name,
            "path": r.path,
            "t_score": r.trace.n_scored(),
            "elpd": r.trace.elpd(),
            "delta_elpd": option_finite(d_elpd),
            "delta_elpd_db": d_elpd_db,
            "evidence_label": evidence_label,
            "lr": option_finite(lr),
            "se_delta_elpd": option_finite(se_elpd),
            "mean_crps": r.trace.mean_crps(),
            "delta_mean_crps": option_finite(mean_dcrps),
            "pit_cov90": r.trace.pit_coverage(0.90),
            // How this row's score was built and what it saw: the two optimism
            // axes (#295), per row because a cohort may mix them once posterior
            // / LFO traces exist.
            "provenance": r.trace.provenance,
            "conditioning": r.trace.conditioning,
            // The trace's own warnings and the miscalibration flag, verbatim
            // from the same source the table prints — a script reading the JSON
            // must not have to re-derive them from `pit_cov90`.
            "warnings": r.trace.warnings,
            "pit_cov90_warning": pit_coverage_warning(r),
        })
    }).collect();
    let any_evidence = rows.iter().enumerate().any(|(i, r)| {
        i != base_idx && paired_delta(&r.trace, base, Field::LogScore).0.is_finite()
    });
    let data_entries: Vec<(&str, &DataIdentity)> =
        rows.iter().map(|r| (r.name.as_str(), &r.data)).collect();
    let out = json!({
        "baseline": rows[base_idx].name,
        "metrics": metrics,
        // The same footnote the table and markdown renderers print: a consumer
        // reading `evidence_label` out of the JSON must get the caveat that
        // travels with it.
        "evidence_scale_note": any_evidence.then_some(JEFFREYS_SCALE_NOTE),
        // The `se_delta_elpd` column's own caveat at a short scoring window.
        "se_caveat": se_caveat(base.n_scored()),
        // The bound-data check (gh#713), as the renderers state it: a document
        // present here at all was rendered under --allow-data-mismatch, and the
        // note says which rows the check could not cover.
        "data_mismatch": check_shared_data(&data_entries).err(),
        "data_unchecked": unchecked_data_note(&data_entries),
        // #295: the plug-in / in-sample caveat, `null` only when every compared
        // trace is honest on both axes.
        "optimism_caveat": optimism_caveat(rows),
        "rows": entries,
    });
    format!("{}\n", serde_json::to_string_pretty(&out).unwrap())
}

fn option_finite(x: f64) -> serde_json::Value {
    if x.is_finite() { serde_json::json!(x) } else { serde_json::Value::Null }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let table = render_table(&rows, 0, &metrics, false);
        assert!(table.contains("LR"), "the table names the column LR:\n{table}");
        assert!(!table.contains("E_T"), "and never E_T:\n{table}");

        let md = render_md(&rows, 0, &metrics, false);
        assert!(md.contains("| LR |"), "the md header names the column LR:\n{md}");
        assert!(!md.contains("E_T"), "and never E_T:\n{md}");

        let json = render_json(&rows, 0, &metrics);
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

        let table = render_table(&rows, 0, &metrics, false);
        assert!(table.contains("within noise"),
            "a Δ inside the noise band is labelled `within noise`:\n{table}");
        assert!(!table.contains("indeterminate"),
            "and carries no Jeffreys tier:\n{table}");
        assert!(table.contains("not Bayes factors"),
            "a table showing evidence labels states what the Jeffreys scale \
             calibrates:\n{table}");

        let md = render_md(&rows, 0, &metrics, false);
        assert!(md.contains("within noise"), "the md table agrees:\n{md}");
        assert!(!md.contains("indeterminate"), "{md}");
        assert!(md.contains("not Bayes factors"), "{md}");

        let json = render_json(&rows, 0, &metrics);
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

        let table = render_table(&rows, 0, &metrics, false);
        assert!(table.contains("overwhelming for"),
            "10 nats ≈ 43 dB against se = 1.00 keeps its tier:\n{table}");
        assert!(!table.contains("within noise"), "{table}");

        let json = render_json(&rows, 0, &metrics);
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
        assert_eq!(ok.len(), 3, "one joint row per step (no per-stream data here)");
        assert!((ok[0].delta().unwrap() - 1.0).abs() < 1e-12);

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
        let r = PointwiseRow {
            model: "cand".into(),
            baseline: "base".into(),
            t: 14.0,
            scope: "stream",
            stream: "east".into(),
            log_score: Some(-1.0),
            baseline_log_score: None,
        };
        assert_eq!(r.delta(), None, "no difference without both sides");
        let tsv = render_pointwise_tsv(&[r]);
        let data = tsv.lines().find(|l| l.starts_with("cand")).expect("a data row");
        let cells: Vec<&str> = data.split('\t').collect();
        assert_eq!(cells.len(), 8, "eight columns: {cells:?}");
        assert_eq!(cells[2], "14", "integral times are written as integers");
        assert_eq!(cells[6], "", "the absent baseline score is empty: {cells:?}");
        assert_eq!(cells[7], "", "the absent difference is empty: {cells:?}");
        assert!(!tsv.contains("NaN"), "never NaN: {tsv}");
    }

    /// gh#295 / the "one format, one warning" rule: a warning or caveat that
    /// the table prints and another format drops is a silent-wrong answer for
    /// whoever reads that other format. The markdown table is what goes into a
    /// report and the JSON is what a script reads — a miscalibration flag, a
    /// trace warning, and the optimism caveat must reach all three.
    #[test]
    fn no_output_format_drops_a_warning_or_the_optimism_caveat() {
        use sim::inference::prequential::PrequentialWarning;
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        // Row `b`: every observation lands outside its own 90% predictive
        // interval (PIT 0.99), and the filter saved no predictive samples.
        for s in &mut rows[1].trace.steps {
            s.pit = 0.99;
        }
        rows[1].trace.warnings.push(PrequentialWarning::SamplesNotSaved);
        let metrics = vec!["elpd".to_string(), "pit_cov90".to_string()];

        let table = render_table(&rows, 0, &metrics, false);
        let md = render_md(&rows, 0, &metrics, false);
        let json = render_json(&rows, 0, &metrics);

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
        check_shared_observation_axis(&[a, b])
            .expect("the same stream set in a different order is the same set");

        // Different sets → refused, naming the step, both sets, both traces.
        let a = row_with_streams("a", &t, &["north", "south"]);
        let b = row_with_streams("b", &t, &["north"]);
        let err = check_shared_observation_axis(&[a, b])
            .expect_err("a 2-stream joint may not be differenced against a 1-stream one");
        assert!(err.contains("south"), "names the stream that differs: {err}");
        assert!(err.contains("'a'") && err.contains("'b'"),
            "names both traces: {err}");
        assert!(err.contains("t=7"), "and the step where they diverge: {err}");

        // Two legacy v1 traces (no per-stream breakdown at all) carry no
        // information to check, so they are allowed through.
        let a = row_at("a", &t, &[-6.0, -6.0, -6.0]);
        let b = row_at("b", &t, &[-6.0, -6.0, -6.0]);
        check_shared_observation_axis(&[a, b])
            .expect("two traces with no per-stream breakdown are not a mismatch");

        // One with a breakdown and one without: the sets cannot be compared, so
        // the difference cannot be shown to be like-for-like. Refused, and the
        // message says which side is missing the breakdown rather than naming a
        // phantom stream mismatch.
        let a = row_with_streams("a", &t, &["north", "south"]);
        let b = row_at("b", &t, &[-6.0, -6.0, -6.0]);
        let err = check_shared_observation_axis(&[a, b])
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
        check_shared_data(&[("a", &same_a), ("b", &same_b)])
            .expect("fits bound to the same bytes compare");

        // One stream's file differs → refused, naming the stream and both fits.
        let revised = digests(&[("north", "aaa"), ("south", "ccc")]);
        let err = check_shared_data(&[("a", &same_a), ("b", &revised)])
            .expect_err("a differing data file must be refused");
        assert!(err.contains("south"), "names the stream that differs: {err}");
        assert!(!err.contains("north"), "and not the one that agrees: {err}");
        assert!(err.contains("'a'") && err.contains("'b'"), "names both fits: {err}");
        assert!(err.contains("--allow-data-mismatch"),
            "and how to proceed deliberately: {err}");

        // A stream one fit bound and the other did not is the same failure.
        let extra = digests(&[("north", "aaa"), ("south", "bbb"), ("east", "ddd")]);
        let err = check_shared_data(&[("a", &same_a), ("b", &extra)])
            .expect_err("an extra bound stream must be refused");
        assert!(err.contains("east"), "{err}");

        // Rows with no data identity are not evidence either way: they neither
        // refuse the comparison nor pass silently.
        let unknown = DataIdentity::Unknown("read as an explicit prequential.json".into());
        check_shared_data(&[("a", &same_a), ("t", &unknown)])
            .expect("a row with no data identity cannot contradict anything");
        let note = unchecked_data_note(&[("a", &same_a), ("t", &unknown)])
            .expect("but it is named");
        assert!(note.contains("'t'") && note.contains("prequential.json"), "{note}");
        assert!(!note.contains("'a'"), "the checked row is not in the note: {note}");
        assert!(unchecked_data_note(&[("a", &same_a), ("b", &same_b)]).is_none(),
            "no note when every row was checked");
    }

    /// gh#729: the derive path shells out to `camdl pfilter`, which runs a
    /// chain-binomial bootstrap particle filter whatever the fit ran on. For an
    /// ODE fit that means the "prequential" is produced by a different forward
    /// process than the one whose θ̂ it plugs in — a stochastic filter scoring a
    /// deterministic model's parameters — and the table labels it as that fit's
    /// score. Refused rather than labelled, since a comparison the reader must
    /// know to discount is one they will forget to discount.
    #[test]
    fn refuses_to_derive_a_prequential_under_a_backend_the_fit_did_not_use() {
        use crate::run_meta::InferenceBackend;
        let seg = Path::new("results/fits/sir-abc12345");

        check_derivable_backend(InferenceBackend::ChainBinomial, seg)
            .expect("the filter and the fit agree — this is the ordinary path");

        let err = check_derivable_backend(InferenceBackend::Ode, seg)
            .expect_err("an ODE fit's prequential may not be derived by a \
                         chain-binomial filter");
        assert!(err.contains("ode"), "names the backend the fit ran on: {err}");
        assert!(err.contains("chain_binomial"),
            "and the one the derived score would come from: {err}");
        assert!(err.contains("gh#312"), "and where the fix is tracked: {err}");
        assert!(err.contains("sir-abc12345"), "and which fit: {err}");
    }

    /// `se(Δ)` is √(T·Var) of the per-observation differences — a normal
    /// approximation whose accuracy depends on T and on how alike the two
    /// models are. At a short scoring window it is not a quantity to read a
    /// verdict from, and the table says so once, not once per row.
    #[test]
    fn a_short_scoring_window_carries_the_se_caveat() {
        let metrics = vec!["elpd".to_string()];
        let short = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];

        let table = render_table(&short, 0, &metrics, false);
        assert_eq!(table.matches("arXiv:2008.10296").count(), 1,
            "the caveat fires once per render, not once per row:\n{table}");
        let md = render_md(&short, 0, &metrics, false);
        assert_eq!(md.matches("arXiv:2008.10296").count(), 1, "{md}");
        let json = render_json(&short, 0, &metrics);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["se_caveat"].as_str().is_some_and(|s| s.contains("2008.10296")),
            "the JSON carries it as a field:\n{json}");

        // A long scoring window does not: the caveat must mean something when
        // it appears.
        let times: Vec<f64> = (1..=120).map(|i| i as f64).collect();
        let long = vec![
            row_at("a", &times, &vec![-6.0; 120]),
            row_at("b", &times, &vec![-5.9; 120]),
        ];
        let table = render_table(&long, 0, &metrics, false);
        assert!(!table.contains("2008.10296"),
            "120 scored steps is not a small-T window:\n{table}");
        let json = render_json(&long, 0, &metrics);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["se_caveat"].is_null(), "{json}");
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

        let table = render_table(&rows, 0, &metrics, false);
        assert!(table.contains("different observed data") && table.contains("cases"),
            "the text table states the confound:\n{table}");
        let md = render_md(&rows, 0, &metrics, false);
        assert!(md.contains("different observed data"), "and the markdown:\n{md}");

        let json = render_json(&rows, 0, &metrics);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["data_mismatch"].as_str().is_some_and(|s| s.contains("cases")),
            "and the JSON, as a field:\n{json}");
        assert!(v["data_unchecked"].is_null(),
            "both rows carried digests, so nothing was skipped:\n{json}");

        // Matching data: no confound line anywhere.
        rows[1].data = digests(&[("cases", "aaaaaaaa11")]);
        let table = render_table(&rows, 0, &metrics, false);
        assert!(!table.contains("different observed data"), "{table}");
        let json = render_json(&rows, 0, &metrics);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v["data_mismatch"].is_null(), "{json}");
    }

    fn digests(entries: &[(&str, &str)]) -> DataIdentity {
        DataIdentity::Digests(entries.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect())
    }

    /// A trace with joint scores at the given times and no per-stream breakdown.
    fn row_at(name: &str, times: &[f64], log_scores: &[f64]) -> Row {
        use sim::inference::prequential::{Conditioning, PrequentialStep, Provenance};
        let steps = times.iter().zip(log_scores).map(|(&t, &ls)| PrequentialStep {
            t,
            y_obs: 10.0,
            y_pred_samples: Vec::new(),
            log_score: ls,
            crps: 1.0,
            pit: 0.5,
            ess: 900.0,
            interval: Default::default(),
            per_stream: Vec::new(),
        }).collect();
        Row {
            name: name.to_string(),
            path: name.to_string(),
            data: DataIdentity::Unknown("a test fixture, built from scores alone".into()),
            trace: PrequentialTrace {
                schema_version: 1,
                t0: 1,
                provenance: Provenance::PlugIn,
                conditioning: Conditioning::InSample,
                steps,
                warnings: Vec::new(),
                pit_randomization_seed: None,
            },
        }
    }

    /// A trace scoring the named streams at every one of `times`.
    fn row_with_streams(name: &str, times: &[f64], streams: &[&str]) -> Row {
        use sim::inference::prequential::StreamScore;
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
            }).collect();
        }
        row
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

    /// gh#417: `compare --exclude-chains` must drop the named chains from the
    /// posterior cloud BEFORE deriving θ̂ — the plug-in point the prequential is
    /// scored at. A stuck chain shifts the posterior mean, so the subset θ̂ must
    /// differ from the all-chain θ̂ and equal the mean over the retained chains
    /// (θ̂ changes ⇒ elpd/CRPS change: a correctness knob, not cosmetic).
    #[test]
    fn posterior_mean_excludes_selected_chains() {
        use crate::chain_selection::ChainSelection;
        use crate::posterior_draws::PosteriorDrawsRef;

        // Two chains in draws.tsv (`chain` is 0-based on disk). Chain id 1 is
        // mixed at beta = 1.0; chain id 2 is stuck at beta = 9.0 (never moved,
        // 0% acceptance).
        let dir = tempfile::tempdir().unwrap();
        let draws_path = dir.path().join("draws.tsv");
        let mut s = String::from("chain\tdraw\tbeta\n");
        for d in 0..4 {
            s.push_str(&format!("0\t{d}\t1.0\n"));
        }
        for d in 0..4 {
            s.push_str(&format!("1\t{d}\t9.0\n"));
        }
        std::fs::write(&draws_path, s).unwrap();

        let pref = PosteriorDrawsRef {
            stage: "pgas".into(),
            draws_path,
            method: None,
            backend: None,
            chain_selection: None,
        };

        let beta_of = |toml_text: &str| -> f64 {
            toml_text
                .lines()
                .find_map(|l| l.strip_prefix("beta = "))
                .and_then(|v| v.trim().parse::<f64>().ok())
                .unwrap_or_else(|| panic!("no beta in θ̂:\n{toml_text}"))
        };

        // All chains: mean beta = (1.0·4 + 9.0·4)/8 = 5.0 — the stuck chain
        // drags θ̂ up.
        let all = posterior_mean_params_toml(&pref).unwrap();
        assert_eq!(beta_of(&all), 5.0, "all-chain θ̂ pools both chains");

        // Drop the stuck chain 2 → mean beta = 1.0 over the retained chain.
        let sel = ChainSelection::parse_exclude("2").unwrap();
        let sub =
            posterior_mean_params_toml(&pref.clone().with_selection(Some(sel))).unwrap();
        assert_eq!(beta_of(&sub), 1.0, "subset θ̂ is the mean over the retained chain");
        assert_ne!(
            beta_of(&all),
            beta_of(&sub),
            "excluding a stuck chain must change θ̂ (hence the prequential score)"
        );
    }

    /// gh#418: per-fit `@fit:ids` binds to one fit; bare ids are cohort-wide.
    #[test]
    fn parse_cohort_exclude_forms() {
        // Fit names are matched VERBATIM. A handle-referenced fit carries the
        // `@` in its name (`@a`); a path-referenced one does not (`ctl_rm.toml`).
        let names = vec!["@a".to_string(), "@b".to_string()];

        // Empty → no filtering.
        assert!(parse_cohort_exclude(&[], &names).unwrap().per_fit.is_empty());

        // Bare ids → cohort-wide: every fit gets the same drop set.
        let c = parse_cohort_exclude(&["3,4".to_string()], &names).unwrap();
        assert!(c.cohort_wide);
        assert_eq!(c.for_fit("@a").unwrap().excluded_csv(), "3,4");
        assert_eq!(c.for_fit("@b").unwrap().excluded_csv(), "3,4");

        // Per-fit → ONLY the named fit is filtered; the other keeps all chains.
        let c = parse_cohort_exclude(&["@a:2".to_string()], &names).unwrap();
        assert!(!c.cohort_wide);
        assert_eq!(c.for_fit("@a").unwrap().excluded_csv(), "2");
        assert!(c.for_fit("@b").is_none(), "@b keeps all chains");

        // A path-named fit (no `@`) is matched by its bare name.
        let paths = vec!["ctl_rm.toml".to_string(), "ctl_bb.toml".to_string()];
        let c = parse_cohort_exclude(&["ctl_rm.toml:4".to_string()], &paths).unwrap();
        assert_eq!(c.for_fit("ctl_rm.toml").unwrap().excluded_csv(), "4");
        assert!(c.for_fit("ctl_bb.toml").is_none());

        // Multiple per-fit tokens, each its own drop set.
        let c =
            parse_cohort_exclude(&["@a:2".to_string(), "@b:5,6".to_string()], &names).unwrap();
        assert_eq!(c.for_fit("@a").unwrap().excluded_csv(), "2");
        assert_eq!(c.for_fit("@b").unwrap().excluded_csv(), "5,6");
    }

    #[test]
    fn parse_cohort_exclude_hints_spurious_at_sigil() {
        // A path-named fit targeted with a spurious `@` (the common slip garki
        // hit): the error names the fix rather than a bare "no such fit".
        let names = vec!["ctl_rm.toml".to_string(), "ctl_bb.toml".to_string()];
        let e = parse_cohort_exclude(&["@ctl_rm.toml:4".to_string()], &names).unwrap_err();
        assert!(e.contains("did you mean 'ctl_rm.toml'"), "{e}");
    }

    /// A per-fit token that cannot bind to exactly one model is a hard error —
    /// never a silent no-op or a wrong-fit bind.
    #[test]
    fn parse_cohort_exclude_rejects_bad_forms() {
        let names = vec!["@a".to_string(), "@b".to_string()];

        // Unknown fit name → error naming the available fits (verbatim, so the
        // message names `@z` — and there is no `z` to hint, so no "did you mean").
        let e = parse_cohort_exclude(&["@z:3".to_string()], &names).unwrap_err();
        assert!(e.contains("no compared fit named '@z'") && e.contains("@a"), "{e}");
        assert!(!e.contains("did you mean"), "no hint when the stripped name also misses: {e}");

        // Mixing bare (cohort) and per-fit tokens → rejected as ambiguous.
        assert!(parse_cohort_exclude(&["3".to_string(), "@a:4".to_string()], &names).is_err());

        // Ambiguous name (two compared models share it) → rejected.
        let dup = vec!["scout".to_string(), "scout".to_string()];
        let e = parse_cohort_exclude(&["scout:3".to_string()], &dup).unwrap_err();
        assert!(e.contains("ambiguous"), "{e}");

        // Same fit listed twice → rejected.
        assert!(
            parse_cohort_exclude(&["@a:2".to_string(), "@a:3".to_string()], &names).is_err()
        );

        // Missing fit name before ':' → rejected.
        assert!(parse_cohort_exclude(&[":3".to_string()], &names).is_err());
    }
}

