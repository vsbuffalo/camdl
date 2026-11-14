//! `camdl compare` — multi-model prequential comparison table.
//!
//! Consumes `prequential.json` artifacts written by `fit run` (PFilter
//! method) or `camdl pfilter --save-prequential`. Computes Δelpd / Δcrps
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
//!   - changed-data preflight: a derived row's training file must still hash
//!     to what its fit recorded (override: --allow-changed-data)
//!   - backend preflight on the derive path (gh#729; no override — gh#312)
//!   - formats: table (default), md, json
//!   - compare.toml for reproducible multi-model specs
//!
//! Out of scope (Part II): betting mode, CAS ref resolution, obs-model
//! preflight, stacking, plotting.

use crate::chain_selection::ChainSelection;
use crate::fit::handle::ResolvedFit;
use crate::fit::stream_bindings::StreamBindings;
use crate::posterior_draws::PosteriorDrawsRef;
use serde::Deserialize;
use sim::inference::prequential::{
    Conditioning, PrequentialTrace, StreamMean, ThetaSource, TraceOverride,
};
use std::path::{Path, PathBuf};

/// Default particle count for auto-deriving a prequential from a fit handle.
/// Applied uniformly across the compared fits (see [`DeriveSettings`]).
pub(crate) const DEFAULT_DERIVE_PARTICLES: usize = 1000;

/// Default filter seed for auto-deriving a prequential from a fit handle.
pub(crate) const DEFAULT_DERIVE_SEED: u64 = 1;

/// Default mixture size for the posterior predictive (§3.6, decision 9 of
/// the 2026-08-29 proposal): one filter pass per thinned draw. `--draws 1`
/// is the documented cheap mode (plug-in at the posterior mean).
pub(crate) const DEFAULT_DERIVE_DRAWS: usize = 64;

/// Default replicate count for the filter-noise Monte-Carlo SE (§3.4
/// remedy 4, decision 9). `--replicates 1` is the documented cheap mode
/// (no replication, no MC SE).
pub(crate) const DEFAULT_DERIVE_REPLICATES: usize = 5;

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
    }
}

/// Settings applied uniformly to every fit handle whose prequential is
/// auto-derived, so T_score and the scores stay commensurable across the
/// compared fits. An explicit `prequential.json` input ignores these (it is
/// read as-is, at whatever particles/seed produced it).
#[derive(Debug, Clone, Copy)]
struct DeriveSettings {
    particles: usize,
    seed: u64,
    /// `--in-sample` (gh#585): score the full series at θ̂ even for a fit
    /// that declares a holdout, stamped `in_sample` with its caveat.
    force_in_sample: bool,
    /// `--draws` (§3.6): mixture size for a Bayesian fit's predictive;
    /// 1 = plug-in at the posterior mean.
    draws: usize,
    /// `--replicates` (§3.4 remedy 4): filter replicates per fit at seeds
    /// `seed..seed+R`, combined per step by log-mean-exp; the replicate
    /// totals give the Monte-Carlo (filter-noise) SE. 1 = no replication.
    replicates: usize,
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
    /// Whether the training files this row scored are the bytes its fit was
    /// fitted to; empty for an explicit prequential.json.
    freshness: DataFreshness,
    /// The observation streams this row's model declares and its fit bound,
    /// as far as the input says.
    streams: RowStreams,
}

impl Row {
    /// This row re-scored on one leaf alone
    /// ([`sim::inference::prequential::project_onto_stream`]); the refusal
    /// names the row.
    fn onto(&self, leaf: &str) -> Result<Row, String> {
        let trace = sim::inference::prequential::project_onto_stream(&self.trace, leaf)
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
/// so — the trace records scores and times, never which bytes it scored. The
/// fit's own provenance sidecar (`fit.meta.json`, [`crate::run_meta::FitSidecar`])
/// does: `data_hashes` and `holdout_hashes` record, per observation source, the
/// SHA-256 of the training and `[data.holdout]` files bound to it, the digests
/// the fit's run identity was built from. Read from there rather than
/// re-hashing, so the check answers "were these fits bound to the same bytes?"
/// with what each fit read when it ran.
#[derive(Clone)]
enum DataIdentity {
    /// What the fit's sidecar records for each source it binds.
    Digests(RecordedData),
    /// Nothing to check, and why — an explicit `prequential.json` (a trace
    /// records no data identity), or a fit whose sidecar is absent. The reason
    /// is printed, because a check that silently covers half the cohort is
    /// worse than one the reader knows the scope of.
    Unknown(String),
}

/// A fit's recorded data hashes, read per observation source through the fit's
/// [`StreamBindings`] ([`recorded_by_source`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RecordedData {
    /// Source → the hash the fit recorded for the file bound to it.
    training: std::collections::BTreeMap<String, String>,
    /// Sources bound by the fit whose hash cannot be read from its record, and
    /// why. A check involving one of these is reported as not made, never
    /// passed.
    unresolved: std::collections::BTreeMap<String, String>,
    /// Source → the hash the fit recorded for its `[data.holdout]` file. Read
    /// only for a row scored held out on those files.
    holdout: std::collections::BTreeMap<String, String>,
    /// Holdout sources the row scores whose hash the record lacks, and why.
    holdout_unresolved: std::collections::BTreeMap<String, String>,
}

impl RecordedData {
    /// Whether any hash this row's checks compare is shorter than a full
    /// SHA-256: a sidecar written before full digests were recorded holds 8
    /// characters, and every comparison involving it is made on those.
    fn has_short_hash(&self) -> bool {
        self.training.values().chain(self.holdout.values()).any(|h| h.len() < FULL_DIGEST_LEN)
    }
}

/// Hex characters in a full SHA-256.
const FULL_DIGEST_LEN: usize = 64;

/// Whether two recorded hashes name the same bytes, compared on the length of
/// the shorter: an older sidecar records 8 characters of the digest a newer one
/// records in full. An empty hash names nothing and matches nothing.
fn same_digest(a: &str, b: &str) -> bool {
    let n = a.len().min(b.len());
    n > 0 && a.is_char_boundary(n) && b.is_char_boundary(n)
        && a[..n].eq_ignore_ascii_case(&b[..n])
}

/// Read a sidecar's `data_hashes` per source of `bindings`, and, for a row
/// scored held out on `[data.holdout]` files, its `holdout_hashes`.
///
/// A sidecar keys `data_hashes` by source. One written before that, by a fit
/// using the one-file shorthand, keys it by leaf. Such a record is read
/// through the bindings where it names one file unambiguously, that is when
/// every leaf of the source records the same hash; otherwise the source is
/// unresolved, with the reason.
fn recorded_by_source(
    side: &crate::run_meta::FitSidecar,
    bindings: &StreamBindings,
    scores_holdout_files: bool,
) -> RecordedData {
    let record = &side.data_hashes;
    let mut out = RecordedData::default();
    if scores_holdout_files {
        for source in bindings.holdout_by_source().keys() {
            match side.holdout_hashes.get(source) {
                Some(h) => { out.holdout.insert(source.clone(), h.clone()); }
                None => { out.holdout_unresolved.insert(source.clone(),
                    "the fit recorded no hash for its holdout file (its fit.meta.json \
                     was written before holdout files were recorded)".into()); }
            }
        }
    }
    for source in bindings.training_by_source().keys() {
        if let Some(h) = record.get(source) {
            out.training.insert(source.clone(), h.clone());
            continue;
        }
        let leaves = bindings.leaves_of(source);
        let per_leaf: Vec<(&str, &str)> = leaves.iter()
            .filter_map(|l| record.get(*l).map(|h| (*l, h.as_str())))
            .collect();
        let distinct: std::collections::BTreeSet<&str> = per_leaf.iter().map(|(_, h)| *h).collect();
        if !per_leaf.is_empty() && per_leaf.len() == leaves.len() && distinct.len() == 1 {
            out.training.insert(source.clone(), per_leaf[0].1.to_string());
        } else if per_leaf.is_empty() {
            out.unresolved.insert(source.clone(), "the fit recorded no hash for it".into());
        } else {
            out.unresolved.insert(source.clone(), format!(
                "the fit recorded hashes per leaf ({}) that do not name one file for \
                 the leaves {}",
                per_leaf.iter().map(|(l, h)| format!("{l} = {}", short_digest(h)))
                    .collect::<Vec<_>>().join(", "),
                leaves.join(", ")));
        }
    }
    out
}

/// What one training file `compare` is about to score says against the hash
/// the fit recorded for its stream when it ran.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FileCheck {
    /// The bytes are the ones the fit was fitted to.
    Matches,
    /// The file was edited after the fit ran. Short hashes, for the message.
    Changed { recorded: String, current: String },
    /// Nothing to compare against, and why.
    NotChecked(String),
}

/// Compare a freshly computed full SHA-256 of a file's bytes with the hash a
/// fit recorded for it. The record is the full 64-character digest, or its
/// first 8 characters in a sidecar written before full digests were recorded
/// ([`data_prefix_note`] says which rows), so the comparison is by prefix, and
/// an empty or absent record is not a match.
fn check_file_against_record(current_sha256: &str, recorded: Option<&str>) -> FileCheck {
    match recorded {
        None => FileCheck::NotChecked("the fit recorded no hash for it".into()),
        Some("") =>
            FileCheck::NotChecked("the fit recorded an empty hash for it".into()),
        Some(r) if current_sha256.starts_with(&r.to_ascii_lowercase()) => FileCheck::Matches,
        Some(r) => FileCheck::Changed {
            recorded: short_digest(r),
            current: short_digest(current_sha256),
        },
    }
}

/// One file of a derived row that could not be shown to be the one the fit
/// was fitted to: either it changed, or there was nothing to check it against.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FileFinding {
    stream: String,
    path: String,
    check: FileCheck,
}

/// Whether the files a derived row scores are the bytes its fit was fitted
/// to, computed once when the row is loaded and read by the refusal and
/// every renderer.
///
/// `compare` re-reads each training file from its path. The shared-data
/// check ([`check_shared_data`]) reads only the hashes recorded at fit time,
/// so a file edited in place after the fit would be scored at a θ̂ fitted to
/// other bytes, with nothing said. An explicit `prequential.json` row
/// re-reads nothing, so it has nothing to check here (and is already named
/// by [`unchecked_data_note`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct DataFreshness {
    /// Every file that changed or could not be checked; files that match are
    /// not listed.
    findings: Vec<FileFinding>,
    /// The whole row could not be checked, and why (no sidecar, or a sidecar
    /// with no data hashes).
    row_unchecked: Option<String>,
}

/// Hash every training file a derived row is about to score (`streams`,
/// source → path, [`StreamBindings::training_by_source`]) and compare each
/// with the sidecar's record for its source (see [`DataFreshness`]).
fn check_training_freshness(
    streams: &indexmap::IndexMap<String, String>,
    recorded: &DataIdentity,
) -> DataFreshness {
    let record = match recorded {
        DataIdentity::Digests(m) => m,
        DataIdentity::Unknown(why) => return DataFreshness {
            findings: Vec::new(),
            row_unchecked: Some(why.clone()),
        },
    };
    let findings = streams.iter().filter_map(|(stream, path)| {
        let check = match (record.unresolved.get(stream), std::fs::read(path)) {
            (Some(why), _) => FileCheck::NotChecked(why.clone()),
            (None, Ok(bytes)) => check_file_against_record(
                &crate::hashing::sha256_hex(&bytes),
                record.training.get(stream).map(|s| s.as_str())),
            (None, Err(e)) => FileCheck::NotChecked(format!("it could not be read: {e}")),
        };
        (check != FileCheck::Matches).then(|| FileFinding {
            stream: stream.to_string(), path: path.to_string(), check,
        })
    }).collect();
    DataFreshness { findings, row_unchecked: None }
}

/// The rows scored on data changed since their fit ran, and which files, or
/// `None`. The refusal and the `--allow-changed-data` renderings share it.
fn changed_data(rows: &[Row]) -> Option<Mismatch> {
    let mut concerned: Vec<String> = Vec::new();
    let changed: Vec<String> = rows.iter().flat_map(|r| {
        r.freshness.findings.iter().filter_map(move |f| match &f.check {
            FileCheck::Changed { recorded, current } => Some((r.name.clone(), format!(
                "'{}' stream '{}' ({}): the fit recorded {recorded}, the file now \
                 hashes to {current}",
                r.name, f.stream, f.path))),
            _ => None,
        })
    }).map(|(name, text)| {
        if !concerned.contains(&name) { concerned.push(name); }
        text
    }).collect();
    (!changed.is_empty()).then(|| Mismatch {
        rows: concerned,
        finding: format!("data changed after the fit ran: {}", changed.join("; ")),
    })
}

/// The scope of the changed-data check: every derived file, or derived row,
/// that could not be compared with a recorded hash, and why. `None` when
/// every scored file was checked.
fn data_change_unchecked_note(rows: &[Row]) -> Option<String> {
    let mut skipped: Vec<String> = Vec::new();
    for r in rows {
        if let Some(why) = &r.freshness.row_unchecked {
            skipped.push(format!("'{}' ({why})", r.name));
        }
        for f in &r.freshness.findings {
            if let FileCheck::NotChecked(why) = &f.check {
                skipped.push(format!("'{}' stream '{}' ({}): {why}", r.name, f.stream, f.path));
            }
        }
    }
    (!skipped.is_empty()).then(|| format!(
        "not checked for changes since the fit ran: {}.", skipped.join("; ")))
}

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

pub fn cmd_compare(a: &crate::args::CompareArgs) {
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
    let derive = DeriveSettings {
        particles: a.particles,
        seed: a.seed,
        force_in_sample: a.in_sample,
        draws: a.draws.max(1),
        replicates: a.replicates.max(1),
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
        out.fail("error: compare requires either --config FILE or ≥2 fit leaf paths");
    };

    if models.len() < 2 {
        out.fail(format!("error: compare needs ≥2 models; got {}", models.len()));
    }

    // gh#417/gh#418: chain exclusion resolved PER FIT, bound to each model's
    // name. `@a:4` targets one fit; bare `3,4` is cohort-wide (every fit). Parsed
    // and name-validated at the boundary, before any derivation runs. A fit with
    // no posterior cloud (an optimizer fit, or an explicit prequential.json) has
    // nothing to filter, so the selection is inert there.
    let model_names: Vec<String> = models.iter().map(|m| m.name.clone()).collect();
    let cohort = parse_cohort_exclude(&a.exclude_chains, &model_names).unwrap_or_else(|e| {
        out.fail(format!("error: {e}"));
    });
    cohort.warn(&out);

    // Load traces, each filtered by its own fit's selection (if any). Each
    // input is loaded once, however many streams `--streams` selects.
    let rows: Vec<Row> = models.into_iter().map(|m| {
        let selection = cohort.for_fit(&m.name);
        let Loaded { trace, data, freshness, streams } =
            load_trace(&out, &m.path, derive, selection).unwrap_or_else(|e| {
                out.fail(format!("error loading trace for '{}' at '{}': {}", m.name, m.path, e));
            });
        Row { name: m.name, path: m.path, trace, data, freshness, streams }
    }).collect();

    // #295 Ask 1 — the optimism caveat, the miscalibration flags and the trace
    // warnings are emitted by the RENDERERS (`warning_lines`), not here, so they
    // travel with the artifact: a markdown table pasted into a report and a JSON
    // document read by a script carry the same caveats the terminal shows.

    let waivers = Waivers::from_args(a);
    let baseline = baseline.or(cfg_baseline);
    let metrics_chosen = metrics_cli.or(cfg_metrics)
        .unwrap_or_else(|| vec!["elpd".into(), "crps".into(), "pit_cov90".into()]);
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
        let t = prepare_table(&out, rows, &waivers, baseline.as_deref())
            .unwrap_or_else(|e| refuse(&out, e));
        if let Some(path) = &a.pointwise {
            match write_pointwise(&t.rows, t.base_idx, path) {
                Ok(PointwiseWritten { joint: (jp, nj), streams: (sp, ns) }) => out.say(format_args!(
                    "wrote {nj} joint pointwise rows to {} and {ns} per-stream rows to {}",
                    jp.display(), sp.display())),
                Err(e) => {
                    out.fail(format!("error: {e}"));
                }
            }
        }
        out.report_raw(render(&t));
        return;
    };

    // `--streams`: one table per selected leaf, every row re-scored on that
    // leaf alone. A refusal for any leaf refuses the command, naming the leaf.
    let leaves = resolve_selection(&names, &rows).unwrap_or_else(|e| {
        out.fail(format!("error: {e}"));
    });
    let tables: Vec<Table> = leaves.iter().map(|leaf| {
        let projected: Vec<Row> = rows.iter().map(|r| r.onto(leaf))
            .collect::<Result<_, _>>()
            .unwrap_or_else(|e| refuse(&out, for_stream(leaf, format!("error: {e}"))));
        prepare_table(&out, projected, &waivers, baseline.as_deref())
            .unwrap_or_else(|e| refuse(&out, for_stream(leaf, e)))
    }).collect();
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
        Format::Json => render_json_tables(&tables, &leaves, &metrics_chosen),
        Format::Md | Format::Table => tables.iter().map(render).collect::<Vec<_>>().join("\n"),
    };
    out.report_raw(rendered);
}

/// One comparison table, checked and ready to render: its rows, the
/// baseline's index among them, and the outcome of every cross-row check.
struct Table {
    rows: Vec<Row>,
    base_idx: usize,
    checks: Preflight,
}

/// Run the cross-row checks over `rows` and choose the baseline: `Err` with
/// the refusal when a check fails and is not waived.
fn prepare_table(
    out: &crate::out::Out,
    mut rows: Vec<Row>,
    waivers: &Waivers,
    baseline: Option<&str>,
) -> Result<Table, String> {
    // Every cross-row check, once: a refusal returns here; what renders states
    // the outcome, overrides included, from this one value (`Preflight`).
    let checks = preflight(&rows, waivers)?;

    // Rendered under `--allow-changed-data`: each trace scored on a changed
    // file records which streams' files it was.
    for r in &mut rows {
        let streams: Vec<String> = r.freshness.findings.iter()
            .filter(|f| matches!(f.check, FileCheck::Changed { .. }))
            .map(|f| f.stream.clone())
            .collect();
        if let (false, Some(d)) = (streams.is_empty(), r.trace.derivation.as_mut()) {
            d.overrides.push(TraceOverride::ChangedData { streams });
        }
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

/// A refusal of one `--streams` table, naming the stream it was for.
fn for_stream(leaf: &str, refusal: String) -> String {
    match refusal.strip_prefix("error: ") {
        Some(rest) => format!("error: stream '{leaf}': {rest}"),
        None => format!("stream '{leaf}': {refusal}"),
    }
}

/// Whether two held-out rows sealed training at the same boundary.
///
/// A `train_end` is an observation time read back from each fit's sidecar,
/// so two fits cut at the same observation can differ by float noise (a
/// date resolved through a different path, a value round-tripped through
/// text). The holdout code decides "the same observation time at the
/// boundary" with [`crate::fit::runner::TIME_TIE_EPS`]; this uses the same
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

/// A cross-row check that a `compare` flag lets a comparison render through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Waiver {
    MismatchedHorizon,
    MixedProvenance,
    DataMismatch,
    ChangedData,
    MismatchedFilterSettings,
    InSampleAblation,
}

impl Waiver {
    fn flag(self) -> &'static str {
        match self {
            Waiver::MismatchedHorizon => "--allow-mismatched-horizon",
            Waiver::MixedProvenance => "--allow-mixed-provenance",
            Waiver::DataMismatch => "--allow-data-mismatch",
            Waiver::ChangedData => "--allow-changed-data",
            Waiver::MismatchedFilterSettings => "--allow-mismatched-filter-settings",
            Waiver::InSampleAblation => "--allow-in-sample-ablation",
        }
    }

    /// What the failed check does to the reading of the table, in one
    /// sentence. The refusal and every rendering under the flag share it.
    fn consequence(self) -> &'static str {
        match self {
            Waiver::MismatchedHorizon =>
                "Δ columns suppressed ('—' in the table, null in the JSON): a Δ pairs \
                 two rows' scores observation by observation, and these rows scored \
                 different numbers of observations.",
            Waiver::MixedProvenance =>
                "A plug-in predictive is under-dispersed relative to a posterior \
                 mixture, so the Δ confounds model difference with dispersion treatment.",
            Waiver::DataMismatch =>
                "Δelpd across them is part model difference and part data difference, \
                 and the table cannot separate the two.",
            Waiver::ChangedData =>
                "`compare` re-reads these files, so the row is scored on data its θ̂ \
                 was not fitted to.",
            Waiver::MismatchedFilterSettings =>
                "The log of a particle-filter likelihood estimate is biased downward, \
                 and by more at fewer particles (and fewer replicates combined), so \
                 part of Δelpd would measure the filters rather than the models.",
            Waiver::InSampleAblation =>
                "Scored in-sample, Δelpd leans toward the fit that assimilated fewer \
                 streams: its θ̂ was chosen to fit the shared stream alone, while the \
                 other fit's traded that stream off against the rest, so on its own \
                 training data it fits it no better and usually slightly worse, \
                 whatever the extra streams contribute to prediction.",
        }
    }
}

/// The waivers the command line gave.
#[derive(Debug, Clone, Copy, Default)]
struct Waivers {
    mismatched_horizon: bool,
    mixed_provenance: bool,
    data_mismatch: bool,
    changed_data: bool,
    mismatched_filter_settings: bool,
    in_sample_ablation: bool,
}

impl Waivers {
    fn from_args(a: &crate::args::CompareArgs) -> Self {
        Waivers {
            mismatched_horizon: a.allow_mismatched_horizon,
            mixed_provenance: a.allow_mixed_provenance,
            data_mismatch: a.allow_data_mismatch,
            changed_data: a.allow_changed_data,
            mismatched_filter_settings: a.allow_mismatched_filter_settings,
            in_sample_ablation: a.allow_in_sample_ablation,
        }
    }

    fn allows(&self, w: Waiver) -> bool {
        match w {
            Waiver::MismatchedHorizon => self.mismatched_horizon,
            Waiver::MixedProvenance => self.mixed_provenance,
            Waiver::DataMismatch => self.data_mismatch,
            Waiver::ChangedData => self.changed_data,
            Waiver::MismatchedFilterSettings => self.mismatched_filter_settings,
            Waiver::InSampleAblation => self.in_sample_ablation,
        }
    }
}

/// A failed cross-row check: the rows it concerns and what differed.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Mismatch {
    rows: Vec<String>,
    finding: String,
}

impl Mismatch {
    /// The finding and its consequence as one statement.
    fn note(&self, w: Waiver) -> String {
        format!("{}. {}", self.finding, w.consequence())
    }
}

/// A waiver that was given and needed: the check would have refused.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Override {
    waiver: Waiver,
    mismatch: Mismatch,
}

impl Override {
    /// The one line every human rendering prints for it.
    fn line(&self) -> String {
        let rows: Vec<String> = self.mismatch.rows.iter().map(|r| format!("'{r}'")).collect();
        format!("⚠ {} was set for {}: {}", self.waiver.flag(), rows.join(", "),
            self.mismatch.note(self.waiver))
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "flag": self.waiver.flag(),
            "rows": self.mismatch.rows,
            "finding": self.mismatch.finding,
            "consequence": self.waiver.consequence(),
        })
    }
}

/// The outcome of every cross-row check, computed once by [`preflight`] and
/// read by every renderer, so no format can state a different outcome from
/// another, or from the refusal.
#[derive(Debug, Clone, Default)]
struct Preflight {
    /// The waivers that were given and needed, in check order.
    overrides: Vec<Override>,
    /// The bound-data refusal text, present only under its override.
    data_mismatch: Option<String>,
    /// What the bound-data check could not cover.
    data_unchecked: Option<String>,
    /// The changed-data statement, present only under its override.
    data_changed: Option<String>,
    /// What the changed-data check could not cover.
    data_change_unchecked: Option<String>,
    /// Rows whose data hashes were compared on an 8-character prefix.
    data_compared_on_prefix: Option<String>,
    /// The filter-settings statement, present only under its override.
    filter_settings_mismatch: Option<String>,
    /// Rows whose trace records no filter settings.
    filter_settings_unchecked: Option<String>,
}

impl Preflight {
    /// Record `w` as needed, or refuse with `refusal` when it was not given.
    fn waive(&mut self, waived: &Waivers, w: Waiver, m: Mismatch, refusal: String)
        -> Result<(), String>
    {
        if !waived.allows(w) {
            return Err(refusal);
        }
        self.overrides.push(Override { waiver: w, mismatch: m });
        Ok(())
    }

    fn overridden(&self, w: Waiver) -> bool {
        self.overrides.iter().any(|o| o.waiver == w)
    }

    /// Whether the rows scored different numbers of observations, so no Δ is
    /// formed (the comparison rendered under `--allow-mismatched-horizon`).
    fn horizons_differ(&self) -> bool {
        self.overridden(Waiver::MismatchedHorizon)
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

/// The in-sample stream ablation, or `None`: rows re-scored on the same leaves
/// (`--streams`), stamped in-sample, whose fits assimilated different streams.
fn in_sample_ablation(rows: &[Row]) -> Option<Mismatch> {
    let first = rows.first()?;
    let rescored = first.trace.scored_streams.is_some();
    let in_sample = first.trace.conditioning == Conditioning::InSample;
    (rescored && in_sample && rows.len() >= 2 && assimilated_sets_differ(rows)).then(|| Mismatch {
        rows: rows.iter().map(|r| r.name.clone()).collect(),
        finding: format!("the compared fits assimilated different streams and are scored \
            in-sample: {}", assimilated_sets(rows).join("; ")),
    })
}

fn provenance_kind(p: &sim::inference::prequential::Provenance) -> String {
    use sim::inference::prequential::Provenance;
    match p {
        Provenance::PlugIn => "plug_in (a single θ)".to_string(),
        Provenance::Posterior { n_draws } => format!("posterior (mixture over {n_draws} draws)"),
    }
}

/// Run every cross-row check over the loaded rows: `Err` with the refusal
/// message when a check fails and `waived` does not allow it, otherwise the
/// outcome every renderer states.
fn preflight(rows: &[Row], waived: &Waivers) -> Result<Preflight, String> {
    use std::mem::discriminant;
    let mut checks = Preflight::default();
    let names: Vec<String> = rows.iter().map(|r| r.name.clone()).collect();

    // Fairness, part zero (gh#585, Stage 3.4): rows with different
    // conditioning modes are never differenced. An in-sample score (θ saw
    // the scored window) and a held-out score (θ sealed before it) answer
    // different questions; their Δ is meaningless however displayed, so
    // there is no override — the fix is to make the modes agree.
    let mixed = rows.windows(2).any(|w|
        discriminant(&w[0].trace.conditioning) != discriminant(&w[1].trace.conditioning));
    if mixed {
        let mut message = "error: the compared traces mix conditioning modes:".to_string();
        for r in rows {
            let kind = match &r.trace.conditioning {
                Conditioning::InSample => "in_sample (θ saw the scored window)".to_string(),
                Conditioning::HoldOutTail { train_end, .. } =>
                    format!("hold_out_tail (θ sealed at t = {train_end})"),
            };
            message.push_str(&format!("\n       {}: {}", r.name, kind));
        }
        message.push_str("\n       Declare the same holdout in every compared fit, or pass \
                   --in-sample to score all of them in-sample.");
        return Err(message);
    }
    // All held-out: the sealed boundaries must also agree, or the rows
    // score different windows (the axis check would refuse anyway, but
    // this names the actual cause).
    let ends: Vec<f64> = rows.iter().filter_map(|r| match &r.trace.conditioning {
        Conditioning::HoldOutTail { train_end, .. } => Some(*train_end),
        _ => None,
    }).collect();
    if ends.windows(2).any(|w| !same_sealed_boundary(w[0], w[1])) {
        return Err(format!("error: the compared fits seal training at different \
                   boundaries: {ends:?}\n       Held-out scores over different windows are not \
                   commensurable; declare the same holdout_after in every \
                   compared fit."));
    }

    // A table re-scored on one leaf (`--streams`) whose rows' fits assimilated
    // different streams is the stream-ablation experiment. In-sample it is
    // biased by construction toward the fit trained on fewer streams; held
    // out, neither fit's θ̂ was tuned to the scored rows. All rows share one
    // conditioning mode by now, so the first row's is the cohort's.
    if let Some(m) = in_sample_ablation(rows) {
        let message = format!("error: {}.\n       {}\n       Declare the same \
            holdout_after in every compared fit, so the rows are scored held-out, or \
            pass --allow-in-sample-ablation to render anyway.",
            m.finding, Waiver::InSampleAblation.consequence());
        checks.waive(waived, Waiver::InSampleAblation, m, message)?;
    }

    // Provenance preflight (Stage 4.1): a plug-in trace (under-dispersed —
    // one θ) against a posterior mixture is a Δ between predictives of
    // different dispersion. Overridable, unlike conditioning: an optimizer
    // fit has no posterior and comparing it against a Bayesian fit is a
    // legitimate ask — as long as the reader chose it.
    let mixed = rows.windows(2).any(|w|
        discriminant(&w[0].trace.provenance) != discriminant(&w[1].trace.provenance));
    if mixed {
        let kinds: Vec<String> = rows.iter()
            .map(|r| format!("'{}' {}", r.name, provenance_kind(&r.trace.provenance)))
            .collect();
        let mut message = "error: the compared traces mix provenance kinds:".to_string();
        for r in rows {
            message.push_str(&format!("\n       {}: {}", r.name, provenance_kind(&r.trace.provenance)));
        }
        message.push_str(&format!("\n       {}\n       Pass --draws 1 to score every fit \
                   plug-in, or --allow-mixed-provenance to render anyway.",
                   Waiver::MixedProvenance.consequence()));
        checks.waive(waived, Waiver::MixedProvenance, Mismatch {
            rows: names.clone(),
            finding: format!("the compared traces mix provenance kinds: {}", kinds.join("; ")),
        }, message)?;
    }

    // Under `--streams`, rows that assimilated different streams may start
    // scoring the leaf at different times for a reason the refusals below
    // would not otherwise name.
    let first_step = first_step_note(rows).map(|n| format!("\n       {n}")).unwrap_or_default();

    // Fairness: T_score (n_scored) must agree across rows unless overridden.
    let t_scores: Vec<usize> = rows.iter().map(|r| r.trace.n_scored()).collect();
    let t_mismatch = t_scores.iter().any(|&t| t != t_scores[0]);
    if t_mismatch {
        let per_row: Vec<String> = rows.iter()
            .map(|r| format!("'{}' {}", r.name, r.trace.n_scored()))
            .collect();
        checks.waive(waived, Waiver::MismatchedHorizon, Mismatch {
            rows: names.clone(),
            finding: format!("T_score differs across models: {}", per_row.join(", ")),
        }, format!("error: T_score differs across models: {:?}\n       \
            Δelpd and Δcrps are not commensurable.\n       \
            Pass --allow-mismatched-horizon to render (uncomparable Δ columns → '—').{first_step}",
            t_scores))?;
    }

    // Fairness, part two: equal T_score is not an equal observation AXIS.
    // `paired_delta` zips steps by index, so two traces scoring the same NUMBER
    // of observations at different TIMES — a hole in one series, a different
    // `t0` — produce a Δelpd, an se(Δ) and a deciban verdict computed across
    // two different axes, rendered as a confident answer. This is not
    // overridable by `--allow-mismatched-horizon`: that flag says "render the Δ
    // columns as '—' because the horizons differ", whereas this is a pairing
    // that is meaningless however it is displayed.
    //
    // With the horizons differing (and the flag set — otherwise refused
    // above), the flag waives the times only one row scored, and nothing
    // else: at every time both rows scored, the stream-set (gh#570) and
    // coverage (gh#833) checks still hold.
    let pairing = if t_mismatch {
        check_shared_times(rows)
    } else {
        check_shared_observation_axis(rows)
    };
    pairing.map_err(|e| format!("error: {e}{first_step}"))?;

    // Fairness, part three (gh#713): the compared fits must have been bound to
    // the same observations. Two fits on different data produce a Δelpd that is
    // part model difference and part data difference, with nothing in the table
    // to say which. Overridable, unlike the axis checks: comparing a fit on
    // revised counts against the fit on the original ones is a real thing to
    // want, as long as the reader knows the Δ is confounded.
    //
    // Rows re-scored on one leaf (`--streams`) compare only the hashes of that
    // leaf's source, and an absent hash for it is a failed check, not a pass;
    // the pairing check above has established that every row is scored on the
    // same leaves.
    let restricted: Vec<DataIdentity>;
    let mut absent: Option<Mismatch> = None;
    let data_entries: Vec<(&str, &DataIdentity)> = match rows.first()
        .and_then(|r| r.trace.scored_streams.as_ref())
    {
        None => rows.iter().map(|r| (r.name.as_str(), &r.data)).collect(),
        Some(leaves) => {
            let mut missing: Vec<String> = Vec::new();
            let mut concerned: Vec<String> = Vec::new();
            restricted = rows.iter().map(|r| {
                let (d, m) = data_for_streams(r, leaves);
                if !m.is_empty() { concerned.push(r.name.clone()); }
                missing.extend(m);
                d
            }).collect();
            absent = (!missing.is_empty()).then(|| Mismatch {
                rows: concerned,
                finding: format!("no hash is recorded for the data this table scores: {}",
                    missing.join("; ")),
            });
            rows.iter().map(|r| r.name.as_str()).zip(&restricted).collect()
        }
    };
    let data_mismatch = match (absent, check_shared_data(&data_entries).err()) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m),
        (Some(a), Some(b)) => Some(Mismatch {
            rows: a.rows.iter().chain(&b.rows).fold(Vec::new(), |mut v, r| {
                if !v.contains(r) { v.push(r.clone()); }
                v
            }),
            finding: format!("{}; and {}", a.finding, b.finding),
        }),
    };
    if let Some(m) = data_mismatch {
        let message = format!("{}.\n       {}\n       Re-fit both on the same \
            observations, or pass --allow-data-mismatch to render the comparison as \
            confounded.", m.finding, Waiver::DataMismatch.consequence());
        checks.waive(waived, Waiver::DataMismatch, m, format!("error: {message}"))?;
        checks.data_mismatch = Some(message);
    }
    checks.data_unchecked = unchecked_data_note(&data_entries);

    // A derived row's training file edited after its fit ran is scored at a
    // θ̂ fitted to other bytes. The same class as the shared-data refusal, and
    // overridable for the same reason.
    if let Some(m) = changed_data(rows) {
        let note = m.note(Waiver::ChangedData);
        checks.waive(waived, Waiver::ChangedData, m, format!(
            "error: {note}\n       Re-run the fit on the current file, restore the \
             file the fit read, or pass --allow-changed-data to render anyway."))?;
        checks.data_changed = Some(note);
    }
    checks.data_change_unchecked = data_change_unchecked_note(rows);
    checks.data_compared_on_prefix = data_prefix_note(rows);

    // Two rows scored at different particle or replicate counts differ partly
    // by the filters. Overridable: comparing a cheap explicit trace against a
    // derived row is a legitimate ask once the reader knows.
    if let Some(m) = filter_settings_mismatch(rows) {
        let note = m.note(Waiver::MismatchedFilterSettings);
        checks.waive(waived, Waiver::MismatchedFilterSettings, m, format!(
            "error: {note}\n       Re-derive every row with the same --particles and \
             --replicates, or pass --allow-mismatched-filter-settings to render \
             anyway."))?;
        checks.filter_settings_mismatch = Some(note);
    }
    checks.filter_settings_unchecked = filter_settings_unchecked_note(rows);
    Ok(checks)
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
        check_scored_streams(first, r)?;
        for (k, (a, b)) in first.trace.steps.iter().zip(&r.trace.steps).enumerate() {
            if a.t != b.t {
                return Err(format!(
                    "'{}' and '{}' are not on the same observation axis: scored \
                     step {} is t={} in '{}' and t={} in '{}'.\n       \
                     Δelpd pairs observations by position, so differencing these \
                     would compare unlike times.\n       \
                     Re-score both models on the same observation set \
                     (check for a hole in one series, a different t0, or a \
                     different `covers` declaration — a period is scored at its \
                     stop, so `day(t)` scores one bucket after `closing_at`).",
                    first.name, r.name, k + 1, a.t, first.name, b.t, r.name,
                ));
            }
            check_step_pair(first, r, a, b)?;
        }
    }
    Ok(())
}

/// Two traces are scored on the same thing: both jointly over every leaf their
/// fit observed, or both re-scored on the same leaves
/// (`PrequentialTrace::scored_streams`). A step's log score means a different
/// quantity on each side otherwise, even where the steps' streams agree — a
/// one-stream trace scored jointly and a projection onto that stream look the
/// same step by step.
fn check_scored_streams(first: &Row, r: &Row) -> Result<(), String> {
    if first.trace.scored_streams == r.trace.scored_streams {
        return Ok(());
    }
    let what = |row: &Row| match &row.trace.scored_streams {
        None => format!("'{}' is scored jointly over every stream its fit observed", row.name),
        Some(s) => format!("'{}' is re-scored on {} alone", row.name, s.join(", ")),
    };
    Err(format!(
        "'{}' and '{}' score different quantities: {}, and {}.\n       \
         Compare traces scored the same way.",
        first.name, r.name, what(first), what(r)))
}

/// Two rows' steps at one observation time scored the same streams, and the
/// same quantity for each stream ([`check_shared_coverage`]). The per-step
/// half of the pairing checks, shared by the equal-horizon path (steps paired
/// by position, after their times are checked equal) and the
/// mismatched-horizon path (steps joined by time).
fn check_step_pair(
    first: &Row,
    r: &Row,
    a: &sim::inference::prequential::PrequentialStep,
    b: &sim::inference::prequential::PrequentialStep,
) -> Result<(), String> {
    let (sa, sb) = (step_streams(a), step_streams(b));
    if sa == sb {
        return check_shared_coverage(first, r, a, b);
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
        // The streams both traces score anywhere, in the first trace's order:
        // `--streams` compares the rows on those, each alone.
        let other = r.trace.streams();
        let shared: Vec<&str> = first.trace.streams().into_iter()
            .filter(|s| other.contains(s)).collect();
        let hint = if shared.is_empty() {
            String::new()
        } else {
            format!("\n       To compare them on the streams they share, each scored \
                     alone, pass --streams {}.", shared.join(","))
        };
        format!(
            "'{}' scored {} there and '{}' scored {}.\n       \
             The step's log score is the JOINT score over those streams, \
             so differencing them compares unlike quantities.\n       \
             Re-score both models on the same set of observation streams.{hint}",
            first.name, fmt_stream_set(&sa), r.name, fmt_stream_set(&sb),
        )
    };
    Err(format!(
        "'{}' and '{}' did not score the same streams at t={}: {detail}",
        first.name, r.name, a.t,
    ))
}

/// The pairing checks for rows whose horizons differ, rendered under
/// `--allow-mismatched-horizon`.
///
/// The flag waives the T_score check — the times only one row scored — and
/// nothing else. At every observation time both rows scored, steps joined by
/// `t` rather than by position (their positions differ once one series is
/// longer), the rows must still have scored the same streams and the same
/// quantity per stream, with the same refusals as the equal-horizon path.
fn check_shared_times(rows: &[Row]) -> Result<(), String> {
    let Some(first) = rows.first() else { return Ok(()) };
    for r in rows.iter().skip(1) {
        check_scored_streams(first, r)?;
        for a in &first.trace.steps {
            // Step times come from the same observation grid, so a shared
            // time is the same value; this is the equality the
            // equal-horizon path pairs on.
            if let Some(b) = r.trace.steps.iter().find(|b| b.t == a.t) {
                check_step_pair(first, r, a, b)?;
            }
        }
    }
    Ok(())
}

/// The stream sets agree at this step; do the values scored agree on what they
/// COVER? (gh#833.) Two fits of one file that differ only in what a row is
/// declared to cover — a gapped window against a merged one, a stream read as
/// an instant against the same name read as a flow — score at the same times
/// and produce different likelihoods, and nothing else in the pipeline can tell
/// that apart from a model difference: the times agree, the stream names agree,
/// the data hashes agree. Only the per-stream coverage record does not.
///
/// An `Unrecorded` side (a trace older than schema 4) is treated as the v1
/// per-stream case is: two such traces have nothing to check and pass, one of
/// each cannot be checked and is refused, because an unverifiable window is not
/// a verified one.
fn check_shared_coverage(
    first: &Row,
    r: &Row,
    a: &sim::inference::prequential::PrequentialStep,
    b: &sim::inference::prequential::PrequentialStep,
) -> Result<(), String> {
    use sim::inference::Coverage;
    for sa in &a.per_stream {
        let Some(sb) = b.per_stream.iter().find(|s| s.stream == sa.stream) else { continue };
        if sa.coverage == sb.coverage {
            continue;
        }
        if sa.coverage == Coverage::Unrecorded || sb.coverage == Coverage::Unrecorded {
            let (without, with, known) = if sa.coverage == Coverage::Unrecorded {
                (&first.name, &r.name, sb.coverage)
            } else {
                (&r.name, &first.name, sa.coverage)
            };
            return Err(format!(
                "'{}' and '{}' cannot be shown to score the same quantity for '{}' at \
                 t={}: '{without}' does not record what its rows cover, so its score \
                 cannot be shown to span the same window as '{with}' ({}).\n       \
                 Re-derive '{without}' — a `prequential.json` older than schema 4 \
                 predates the coverage record.",
                first.name, r.name, sa.stream, a.t, fmt_coverage(known),
            ));
        }
        return Err(format!(
            "'{}' and '{}' score '{}' over different windows at t={}: {} in '{}' and \
             {} in '{}'.\n       \
             The two were scored at the same time but not against the same quantity — \
             a different `covers` declaration or different window columns — and \
             Δelpd would report that as a model difference.\n       \
             Re-fit both with the same declaration of what each row covers.",
            first.name, r.name, sa.stream, a.t,
            fmt_coverage(sa.coverage), first.name, fmt_coverage(sb.coverage), r.name,
        ));
    }
    Ok(())
}

fn fmt_coverage(c: sim::inference::Coverage) -> String {
    use sim::inference::Coverage;
    match c {
        Coverage::Unrecorded => "unrecorded".into(),
        Coverage::Instant => "an instant reading".into(),
        Coverage::Interval { start, stop } => format!("the flow over [{start}, {stop})"),
    }
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
/// either way and are skipped, as is a source either row could not resolve
/// ([`RecordedData::unresolved`]); [`unchecked_data_note`] names both.
fn check_shared_data(entries: &[(&str, &DataIdentity)]) -> Result<(), Mismatch> {
    use std::collections::BTreeSet;
    let known: Vec<(&str, &RecordedData)> = entries
        .iter()
        .filter_map(|(name, d)| match d {
            DataIdentity::Digests(m) => Some((*name, m)),
            DataIdentity::Unknown(_) => None,
        })
        .collect();
    // Equality is transitive, so every row is checked against the first that
    // carries digests.
    let Some(&(ref_name, ref_rec)) = known.first() else { return Ok(()) };
    for &(name, rec) in known.iter().skip(1) {
        // Training files, then the `[data.holdout]` files a held-out row
        // scores; a source either row could not resolve takes no part.
        let kinds = [
            ("", &ref_rec.training, &rec.training, &ref_rec.unresolved, &rec.unresolved),
            (" holdout", &ref_rec.holdout, &rec.holdout,
             &ref_rec.holdout_unresolved, &rec.holdout_unresolved),
        ];
        let differing: Vec<String> = kinds.iter().flat_map(|(kind, ref_map, map, ref_un, un)| {
            let mut streams: BTreeSet<&str> = ref_map.keys().map(|s| s.as_str()).collect();
            streams.extend(map.keys().map(|s| s.as_str()));
            streams.into_iter().filter_map(move |s| {
                if ref_un.contains_key(s) || un.contains_key(s) {
                    return None;
                }
                match (ref_map.get(s), map.get(s)) {
                    (Some(a), Some(b)) if same_digest(a, b) => None,
                    (Some(a), Some(b)) => Some(format!(
                        "'{s}'{kind} ({} in '{ref_name}', {} in '{name}')",
                        short_digest(a), short_digest(b))),
                    (Some(_), None) => Some(format!("'{s}'{kind} (bound by '{ref_name}' only)")),
                    (None, Some(_)) => Some(format!("'{s}'{kind} (bound by '{name}' only)")),
                    (None, None) => None,
                }
            }).collect::<Vec<_>>()
        }).collect();
        if !differing.is_empty() {
            return Err(Mismatch {
                rows: vec![ref_name.to_string(), name.to_string()],
                finding: format!("'{ref_name}' and '{name}' were fit to different observed \
                    data: {}", differing.join("; ")),
            });
        }
    }
    Ok(())
}

/// A row's data identity as a table scored on `leaves` alone checks it: the
/// recorded hashes of those leaves' sources (found through the row's
/// [`StreamBindings::source_of`]), with every other source set aside as not
/// compared, so [`check_shared_data`] skips it and [`unchecked_data_note`]
/// names it. Also returns, one line each, the selected sources for which the
/// row carries digests but no hash at all: an absent hash must not read as a
/// matching one, so the caller refuses it.
///
/// A row with no data identity is returned as it is. So is an explicit
/// `prequential.json`, which binds nothing.
fn data_for_streams(r: &Row, leaves: &[String]) -> (DataIdentity, Vec<String>) {
    let (DataIdentity::Digests(rec), RowStreams::Derived { bindings, .. }) = (&r.data, &r.streams)
    else {
        return (r.data.clone(), Vec::new());
    };
    let mut sources: Vec<&str> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for leaf in leaves {
        match bindings.source_of(leaf) {
            Some(s) => if !sources.contains(&s) { sources.push(s) },
            None => missing.push(format!(
                "'{}' does not bind stream '{leaf}', so no hash of its data is recorded",
                r.name)),
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
    split(&rec.holdout, &mut out.holdout, &mut out.holdout_unresolved);
    for (source, why) in &rec.unresolved {
        if sources.contains(&source.as_str()) {
            out.unresolved.insert(source.clone(), why.clone());
        }
    }
    for (source, why) in &rec.holdout_unresolved {
        if sources.contains(&source.as_str()) {
            out.holdout_unresolved.insert(source.clone(), why.clone());
        }
    }
    // Refused by the caller, and set aside here so the pairwise comparison
    // does not report the same absence a second time as a source one row
    // alone bound.
    for s in &sources {
        if !out.training.contains_key(*s) && !out.unresolved.contains_key(*s) {
            missing.push(format!("'{}' records no hash for source '{s}'", r.name));
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

/// The note naming the rows excluded from the data check and why, or `None`
/// when every row carried digests. Separate from [`check_shared_data`] so the
/// scope of the check is stated even when the check itself refuses.
fn unchecked_data_note(entries: &[(&str, &DataIdentity)]) -> Option<String> {
    let skipped: Vec<String> = entries.iter().flat_map(|(name, d)| match d {
        DataIdentity::Digests(rec) => rec.unresolved.iter()
            .map(|(source, why)| format!("'{name}' stream '{source}' ({why})"))
            .chain(rec.holdout_unresolved.iter()
                .map(|(source, why)| format!("'{name}' holdout '{source}' ({why})")))
            .collect::<Vec<_>>(),
        DataIdentity::Unknown(why) => vec![format!("'{name}' ({why})")],
    }).collect();
    (!skipped.is_empty()).then(|| format!(
        "not checked for a shared observation set: {}. \
         A difference against such a row may be a data difference, not a model \
         difference.",
        skipped.join("; "),
    ))
}

/// The rows whose recorded data hashes are shorter than a full SHA-256, so the
/// shared-data and changed-data checks compared them on that prefix, or `None`.
/// Every renderer states it: a difference beyond the prefix is not detected.
fn data_prefix_note(rows: &[Row]) -> Option<String> {
    let short: Vec<String> = rows.iter()
        .filter(|r| matches!(&r.data, DataIdentity::Digests(rec) if rec.has_short_hash()))
        .map(|r| format!("'{}'", r.name))
        .collect();
    (!short.is_empty()).then(|| format!(
        "data compared on 8 characters of the SHA-256 for {}: its fit.meta.json was \
         written before full digests were recorded, so the shared-data and \
         changed-data checks for that row compare the first 8 characters only.",
        short.join(", ")))
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
            let find = |ss: &[sim::inference::prequential::StreamScore], n: &str| {
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

/// One compared input, loaded: what [`Row`] carries besides its name.
struct Loaded {
    trace: PrequentialTrace,
    data: DataIdentity,
    freshness: DataFreshness,
    streams: RowStreams,
}

/// What [`derive_prequential`] produced for one fit.
struct Derived {
    trace: PrequentialTrace,
    data: DataIdentity,
    freshness: DataFreshness,
    streams: RowStreams,
}

/// Resolve a single comparison input to a `PrequentialTrace`. Two paths:
///
/// 1. **Explicit prequential** (kept, tried first): a `.json` file that exists,
///    or a directory holding `prequential.json` — read+parsed as-is. Preserves
///    `pfilter --save-prequential` and method-dir inputs; the derive settings do
///    not touch it.
/// 2. **Fit handle** (Phase 2a): `@label` / hash prefix / run dir / `fit.toml`.
///    The prequential is DERIVED by invoking the canonical `camdl pfilter` at
///    the fit's sealed θ̂ — never by reimplementing the filter (the obs-model
///    assembly is already triplicated; a fourth copy is forbidden).
fn load_trace(
    out: &crate::out::Out,
    path: &str,
    derive: DeriveSettings,
    selection: Option<&ChainSelection>,
) -> Result<Loaded, String> {
    if let Some(trace) = try_load_explicit_prequential(path)? {
        // A trace records scores and times, never which observations produced
        // them — so this row cannot take part in the data-identity check, and
        // carries no replicate-derived filter-noise SE. It re-reads no data
        // file, so there is nothing to check for changes since a fit.
        return Ok(Loaded {
            trace,
            data: DataIdentity::Unknown(
                "read as an explicit prequential.json, which records no bound data".into()),
            freshness: DataFreshness::default(),
            streams: RowStreams::Explicit,
        });
    }
    // Not an explicit prequential → treat as a fit handle and derive.
    match crate::fit::handle::resolve_fit_run(path) {
        Ok(resolved) => derive_prequential(out, &resolved, derive, selection)
            .map(|d| Loaded {
                trace: d.trace, data: d.data, freshness: d.freshness, streams: d.streams,
            }),
        Err(resolve_err) => Err(format!(
            "'{path}' is neither a prequential trace nor a resolvable fit handle.\n  \
             - as a fit handle: {resolve_err}\n  \
             - as a prequential path: no prequential.json at '{path}' (or \
             '{path}/prequential.json') — run `camdl pfilter --save-prequential` \
             or `camdl fit run` with a pfilter method to generate one."
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
            "the fit at {} ran its terminal method on the `{}` backend, but \
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

/// A resolved fit's sealed holdout, checked for the `HoldOutTail` stamp
/// (gh#585; §3.7.3 of the 2026-08-29 honest-predictive-evaluation
/// proposal). Detection enforces (a) and (b) of the non-leakage gate:
///
/// - (a) the fit's sealed config — hashed into its identity — declares a
///   holdout (`holdout_after` or `[data.holdout]`);
/// - (b) `fit.meta.json` carries the *applied* training window, the
///   positive proof written by the engine that actually truncated
///   training. A version predicate cannot serve here (the recorded
///   `camdl_version`'s git-hash suffix is unordered); the record can —
///   a pre-gh#585 fit parsed the declaration without applying it, so it
///   can never carry the record, and is refused with the re-run fix.
///
/// (c) — every scored time strictly after `train_end` — is checked on the
/// derived trace before stamping ([`derive_prequential`]).
struct SealedHoldout {
    train_end: f64,
    /// source → holdout file path for explicit `[data.holdout]` fits; the
    /// scored series is training ∪ holdout, merged per source. Empty for
    /// temporal `holdout_after` fits, whose training files already carry
    /// the full series (truncation happened at fit load, not in the file).
    holdout_files: Vec<(String, String)>,
}

fn detect_sealed_holdout(
    resolved: &ResolvedFit,
    bindings: &StreamBindings,
) -> Result<Option<SealedHoldout>, String> {
    let data = resolved.config.problem.data.as_ref();
    let declares = data
        .map(|d| d.holdout_after.is_some() || d.holdout.is_some())
        .unwrap_or(false);
    if !declares {
        return Ok(None);
    }
    let window = crate::run_meta::read_fit_sidecar(&resolved.segment)?
        .and_then(|s| s.training_window);
    let Some(window) = window else {
        return Err(format!(
            "the fit at {} declares a holdout but its fit.meta.json carries \
             no applied training window — it was produced before the holdout \
             declaration was actually applied (gh#585), so its θ̂ saw the \
             full series and a \"held-out\" score would be a leak presented \
             as honest.\n  Re-run the fit (its identity re-keys; the new run \
             trains on the truncated series and records the window), or pass \
             --in-sample to score it as the in-sample trace it is.",
            resolved.segment.display()));
    };
    let holdout_files = bindings.holdout_by_source().iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok(Some(SealedHoldout { train_end: window.train_end, holdout_files }))
}

/// For an explicit `[data.holdout]` fit, refuse the `HoldOutTail` stamp if
/// any bound training file has an observation after the recorded
/// `train_end`. The derived trace scores everything after `train_end`, so
/// such a row — data θ̂ was fitted to — would be scored as held out. A fit
/// run before `train_end` covered every bound stream records exactly this
/// shape (a stream without a holdout file training past another stream's
/// held-out rows), and so does a training file extended after the fit.
///
/// The training files are read through the fit's own loader, so times
/// (dates included) resolve as the fit resolved them. A `holdout_after`
/// fit is exempt: its training files legitimately run past the cutoff,
/// which the fit applied at load.
fn check_training_ends_by_train_end(
    h: &SealedHoldout,
    bindings: &StreamBindings,
    model: &ir::Model,
    dt: f64,
    segment: &Path,
) -> Result<(), String> {
    if h.holdout_files.is_empty() {
        return Ok(());
    }
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let mut late: Vec<String> = Vec::new();
    for (leaf, source) in bindings.leaves() {
        let path = &bindings.training_by_source()[source];
        let Some(obs) = model.observations.iter().find(|o| o.name == leaf) else { continue };
        let siblings: Vec<&ir::observation::ObservationModel> = model.observations.iter()
            .filter(|o| o.source == obs.source)
            .collect();
        let (rows, _cells, _aux) =
            crate::fit::runner::load_observations(path, obs, &siblings, dt, &time_opts)?;
        let last = rows.iter().map(|o| o.time).fold(f64::NEG_INFINITY, f64::max);
        let entry = format!("'{}' (training data runs to t = {last})", obs.source);
        if last > h.train_end + crate::fit::runner::TIME_TIE_EPS && !late.contains(&entry) {
            late.push(entry);
        }
    }
    if late.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the fit at {} trained on data inside the window `compare` would \
         score as held out: its recorded training window ends at t = {}, but \
         stream {} has training observations after that time. Scoring it \
         would present in-sample scores as held-out.\n  Re-run the fit (it \
         now requires every held-out row to lie after the training data of \
         every stream), or pass --in-sample to score it as the in-sample \
         trace it is.",
        segment.display(), h.train_end, late.join(", ")))
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

/// The rows scored with different particle-filter settings — particle count
/// or replicate count — and how they differ, or `None`. Only rows that record
/// their settings take part; [`filter_settings_unchecked_note`] names the
/// others. The refusal and the `--allow-mismatched-filter-settings`
/// renderings share it.
fn filter_settings_mismatch(rows: &[Row]) -> Option<Mismatch> {
    let recorded: Vec<(&str, &sim::inference::prequential::Derivation)> = rows.iter()
        .filter_map(|r| r.trace.derivation.as_ref().map(|d| (r.name.as_str(), d)))
        .collect();
    // Equality is transitive, so every row is checked against the first that
    // records its settings.
    let (ref_name, ref_d) = *recorded.first()?;
    let mut concerned = vec![ref_name.to_string()];
    let differing: Vec<String> = recorded.iter().skip(1).filter_map(|&(name, d)| {
        let mut parts = Vec::new();
        if d.particles != ref_d.particles {
            parts.push(format!("particles {} vs particles {}", ref_d.particles, d.particles));
        }
        if d.replicates != ref_d.replicates {
            parts.push(format!("replicates {} vs replicates {}", ref_d.replicates, d.replicates));
        }
        (!parts.is_empty()).then(|| {
            concerned.push(name.to_string());
            format!("'{ref_name}' and '{name}': {}", parts.join(", "))
        })
    }).collect();
    (!differing.is_empty()).then(|| Mismatch {
        rows: concerned,
        finding: format!("the compared traces were scored with different \
            particle-filter settings: {}", differing.join("; ")),
    })
}

/// The rows whose trace records no filter settings, so the settings check
/// could not cover them, or `None`.
fn filter_settings_unchecked_note(rows: &[Row]) -> Option<String> {
    let unrecorded: Vec<String> = rows.iter()
        .filter(|r| r.trace.derivation.is_none())
        .map(|r| if r.trace.schema_version < 5 {
            format!("filter settings not recorded for '{}' (trace older than schema 5); \
                     not checked", r.name)
        } else {
            format!("filter settings not recorded for '{}' (its producer wrote no \
                     derivation block); not checked", r.name)
        })
        .collect();
    (!unrecorded.is_empty()).then(|| unrecorded.join("; "))
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

/// The data digests a resolved fit recorded when it ran, read per source of its
/// `bindings` from its provenance sidecar (`fit.meta.json`): training files
/// always, `[data.holdout]` files when the row scores them
/// (`scores_holdout_files`). Not re-hashed here: the files may have moved or
/// changed since, and the question is what the FIT was bound to.
///
/// The sidecar records the very digests the fit's run identity folded
/// ([`crate::fit::stream_bindings::StreamBindings::digests`]); a sidecar
/// written before that holds 8-character training hashes and no holdout
/// hashes, which [`recorded_by_source`] and [`data_prefix_note`] account for. A fit without one (a
/// hand-assembled directory), or whose sidecar cannot be read, is reported as
/// unchecked rather than refused.
fn fit_data_identity(
    resolved: &ResolvedFit,
    bindings: &StreamBindings,
    scores_holdout_files: bool,
) -> DataIdentity {
    match crate::run_meta::read_fit_sidecar(&resolved.segment) {
        Ok(Some(s)) if !s.data_hashes.is_empty() =>
            DataIdentity::Digests(recorded_by_source(&s, bindings, scores_holdout_files)),
        Ok(Some(_)) => DataIdentity::Unknown(
            "its fit.meta.json records no data hashes".into()),
        Ok(None) => DataIdentity::Unknown(format!(
            "no fit.meta.json under {}", resolved.segment.display())),
        Err(e) => DataIdentity::Unknown(e),
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

/// Why one `camdl pfilter` pass of [`derive_prequential`] failed.
enum PassError {
    /// The filter ran and failed at this θ ([`crate::pfilter::FILTER_FAILED`]):
    /// a statement about the draw.
    FilterRejected(String),
    /// Anything else (the fit's inputs, the temp files, the child process):
    /// it would fail at every θ, so it says nothing about the draw.
    Other(String),
}

/// Derive a prequential trace from a sealed fit by invoking the canonical
/// `camdl pfilter` at the fit's winning θ̂. This routes through the one
/// production filter + obs-model assembly rather than rebuilding it, so a
/// derived trace can never silently diverge from a hand-run `pfilter`.
///
/// Steps: (a) θ̂ from the winning method as a flat params TOML; (b) the model
/// (archived IR if present, else the loose source the config names); (c) the
/// fit's data, as `--data LEAF=PATH` for every leaf it binds; (d) run `pfilter
/// --save-prequential` to a temp stem; (e) read back the `{stem}.json` trace.
/// All temp files live in the system temp dir and are best-effort cleaned up.
fn derive_prequential(
    out: &crate::out::Out,
    resolved: &ResolvedFit,
    derive: DeriveSettings,
    selection: Option<&ChainSelection>,
) -> Result<Derived, String> {
    let segment = &resolved.segment;
    // The run θ̂ comes from: the leaf the handle named, else the folder's.
    let run_dir = resolved.run_dir();
    let config = &resolved.config.problem;

    // gh#729: the filter this shells out to is chain-binomial. The θ̂ below
    // comes from the fit's method leaf (both `resolve_posterior_draws` and
    // `winner_params_toml` pick it), so that method's backend is the one the
    // derived score must match. A named run's backend is its own record's;
    // the archived config describes the folder's latest run, which may not
    // be it (gh#896).
    let backend = match &resolved.leaf {
        Some(leaf) => crate::fit::fit_view::FitView::read(leaf)
            .and_then(|v| v.methods.first().map(|s| s.backend)),
        None => resolved.config.inference.method.as_ref().map(|m| m.algorithm.backend()),
    };
    if let Some(backend) = backend {
        check_derivable_backend(backend, run_dir)?;
    }

    // (a) θ — what the prequential is scored at. With `--draws M ≥ 2` and a
    // posterior cloud, the honest choice: M thinned draws, one filter pass
    // each, densities mixed and samples pooled downstream (§3.6, Stage 4.1).
    // `--draws 1`, or an optimizer fit (no cloud), falls back to the plug-in
    // point — routed through the draws-cloud authority
    // (`resolve_posterior_draws`), NOT a per-method point-estimate file: a
    // Bayesian fit (PGAS/PMMH/MH) writes no `final_params.toml` — its θ̂ is
    // the posterior MEAN over `draws.tsv`; only an optimizer fit (IF2/NLopt)
    // has a single winner file.
    let point = |run_dir| -> Result<(Vec<String>, ThetaSource), String> {
        let (toml, source) = point_estimate_params_toml(run_dir, selection)?;
        Ok((vec![toml], source))
    };
    let (theta_tomls, theta_source): (Vec<String>, ThetaSource) = if derive.draws >= 2 {
        match thinned_draws_params_tomls(run_dir, selection, derive.draws)? {
            Some(thinned) => thinned,
            None => point(run_dir)?,
        }
    } else {
        point(run_dir)?
    };
    let is_mixture = theta_tomls.len() >= 2;

    // (b) model — prefer the self-contained archived IR (Phase 1a), else the
    // loose source the config names (recompiled by pfilter). config.model.camdl
    // is already absolute (resolved against the fit.toml dir at load).
    let archived = segment.join("model.ir.json");
    let model_path: PathBuf = if archived.is_file() {
        archived
    } else {
        PathBuf::from(&config.model.camdl)
    };

    // The fit's own intervention selection: the filter scores the fitted
    // model, so it runs the interventions `fit run` ran. Passed as the
    // resolved `--enable`/`--disable` lists rather than `--scenario`, which
    // `pfilter` refuses beside the `--params` θ̂ (gh#194); the lists are the
    // whole of a fit scenario's effect (`util::FitBase`).
    let fit_base = crate::util::FitBase::of(config)?;
    let (model, _) = crate::util::load_model(&model_path.to_string_lossy())?;
    // The step the fit ran at: the filter re-scores the data through the
    // fitted process, so it integrates at the fit's dt — not the model's
    // declared step or 1.0, which is all `pfilter` would see without it.
    let fitted_dt = crate::fit::fitted_model::stored_fit_simulator(resolved)?.dt;
    let (base_enable, base_disable): (Vec<String>, Vec<String>) = if fit_base.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        fit_base.lists(&model).map_err(|e| e.to_string())?
    };

    // (c) data — source → absolute path, and the leaves each source binds.
    // The config's data paths were made absolute (relative to the original
    // fit.toml) at load, so they pass to the child unchanged.
    let data = config.data.as_ref().ok_or_else(|| format!(
        "fit at {} has no [data] block — there are no observations to score, so a \
         prequential trace cannot be derived.\n  Provide an explicit \
         prequential.json, or compare fits that bind data.",
        segment.display()))?;
    let bindings = StreamBindings::resolve(&model, data)?;
    let mut streams = bindings.training_by_source().clone();
    if streams.is_empty() {
        return Err(format!(
            "fit at {} has a [data] block with no observation streams.",
            segment.display()));
    }
    // gh#585 (Stage 3.4): a fit that declares a holdout is scored held-out
    // by default — the filter runs over the FULL series at the sealed θ̂
    // and scores only past the training boundary. `--in-sample` opts out.
    let holdout = if derive.force_in_sample {
        None
    } else {
        detect_sealed_holdout(resolved, &bindings)?
    };
    let scores_holdout_files = holdout.as_ref().is_some_and(|h| !h.holdout_files.is_empty());
    let recorded_data = fit_data_identity(resolved, &bindings, scores_holdout_files);
    // The files about to be scored, against the hashes the fit recorded for
    // them — read before the holdout merge below replaces any path with a
    // temp file.
    let mut freshness = check_training_freshness(&streams, &recorded_data);
    if let (Some(h), DataIdentity::Digests(rec)) = (&holdout, &recorded_data) {
        for (source, path) in &h.holdout_files {
            let check = match (rec.holdout_unresolved.get(source), std::fs::read(path)) {
                (Some(why), _) => FileCheck::NotChecked(why.clone()),
                (None, Ok(bytes)) => check_file_against_record(
                    &crate::hashing::sha256_hex(&bytes),
                    rec.holdout.get(source).map(String::as_str)),
                (None, Err(e)) => FileCheck::NotChecked(format!("it could not be read: {e}")),
            };
            if check != FileCheck::Matches {
                freshness.findings.push(FileFinding {
                    stream: format!("{source} (holdout)"), path: path.clone(), check,
                });
            }
        }
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

    if let Some(h) = &holdout {
        check_training_ends_by_train_end(h, &bindings, &model, fitted_dt, segment)?;
    }

    // Explicit `[data.holdout]` fits: the scored series is training ∪
    // holdout, merged per stream into a temp file (temporal `holdout_after`
    // fits need no merge — their training files already carry the full
    // series; the fit truncated at load, not in the file).
    let mut merged_paths: Vec<PathBuf> = Vec::new();
    if let Some(h) = &holdout {
        for (source, holdout_path) in &h.holdout_files {
            // `StreamBindings::resolve` refused a holdout source with no
            // training file.
            let train_path = streams[source].clone();
            let merged = base.with_extension(format!("{source}.merged.tsv"));
            merge_holdout_file(&train_path, holdout_path, &merged)?;
            streams.insert(source.clone(), merged.to_string_lossy().into_owned());
            merged_paths.push(merged);
        }
    }

    // (d) run the canonical filter — once per θ per replicate. Within a
    // replicate every pass shares the filter seed and score window, so
    // mixture components differ ONLY in θ; replicates differ ONLY in seed
    // (common random numbers across draws and across compared fits).
    let exe = std::env::current_exe()
        .map_err(|e| format!("cannot locate the running camdl binary: {e}"))?;
    let run_pass = |pass: usize, params_toml: &str, seed: u64| -> Result<PrequentialTrace, PassError> {
        let other = PassError::Other;
        let theta_path = base.with_extension(format!("theta{pass}.toml"));
        let preq_stem = format!("{}_preq{pass}", base.to_string_lossy());
        let preq_json = format!("{preq_stem}.json");
        let preq_tsv = format!("{preq_stem}.tsv");
        let cleanup = || {
            let _ = std::fs::remove_file(&theta_path);
            let _ = std::fs::remove_file(&preq_json);
            let _ = std::fs::remove_file(&preq_tsv);
        };
        if let Err(e) = std::fs::write(&theta_path, params_toml) {
            cleanup();
            return Err(other(format!("writing temp params {}: {}", theta_path.display(), e)));
        }
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("pfilter")
            .arg(&model_path)
            .arg("--params").arg(&theta_path)
            .arg("--save-prequential").arg(&preq_stem)
            .arg("--particles").arg(derive.particles.to_string())
            .arg("--seed").arg(seed.to_string())
            .arg("--dt").arg(fitted_dt.to_string())
            // The trace is the product; a stored log-likelihood leaf under
            // the caller's output root is not.
            .arg("--no-store")
            .env("CAMDL_SKIP_VERSION_CHECK", "1");
        // `pfilter --data NAME=PATH` binds NAME as a leaf name (or the
        // prefix of a family of leaf names), not as a source, so every leaf
        // of a source is named with its source's file.
        for (source, abs_path) in &streams {
            for leaf in bindings.leaves_of(source) {
                cmd.arg("--data").arg(format!("{leaf}={abs_path}"));
            }
        }
        for name in &base_enable {
            cmd.arg("--enable").arg(name);
        }
        for name in &base_disable {
            cmd.arg("--disable").arg(name);
        }
        // gh#585 (Stage 3.4): score only past the sealed training boundary.
        if let Some(h) = &holdout {
            cmd.arg("--score-from").arg(h.train_end.to_string());
        }
        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => { cleanup(); return Err(other(format!("spawning `camdl pfilter`: {e}"))); }
        };
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let code = output.status.code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "signal".into());
            cleanup();
            let message = format!(
                "deriving prequential via `camdl pfilter` failed (exit {code}) for fit {}:\n{}",
                segment.display(),
                stderr.trim());
            // Only a filter that ran and failed at θ says anything about the
            // draw; a missing data file or a version mismatch fails every θ.
            return Err(if output.status.code() == Some(crate::pfilter::FILTER_FAILED.0) {
                PassError::FilterRejected(message)
            } else {
                PassError::Other(message)
            });
        }
        // (e) read back the trace.
        let text = match std::fs::read_to_string(&preq_json) {
            Ok(t) => t,
            Err(e) => { cleanup(); return Err(other(format!(
                "`camdl pfilter` succeeded but its prequential output {preq_json} could not be read: {e}"))); }
        };
        let trace = serde_json::from_str::<PrequentialTrace>(&text)
            .map_err(|e| other(format!("parsing derived prequential {preq_json}: {e}")));
        cleanup();
        trace
    };

    let n_theta = theta_tomls.len();
    if is_mixture || derive.replicates >= 2 {
        out.say(format_args!(
            "compare: {}: {} filter passes ({} draw(s) × {} replicate(s), \
             {} particles each; --draws 1 / --replicates 1 for the cheap modes)",
            seg_slug, n_theta * derive.replicates, n_theta, derive.replicates,
            derive.particles));
    }
    let mut replicate_traces: Vec<PrequentialTrace> =
        Vec::with_capacity(derive.replicates);
    for rep in 0..derive.replicates {
        // §3.4 remedy 4: replicate r reruns everything at seed + r.
        let seed_r = derive.seed + rep as u64;
        let mut component_traces = Vec::with_capacity(n_theta);
        for (i, params_toml) in theta_tomls.iter().enumerate() {
            let r = run_pass(rep * n_theta + i, params_toml, seed_r).map_err(|e| match e {
                PassError::FilterRejected(e) if is_mixture => {
                    // A degenerate mixture component is informative, not
                    // repairable: dropping it would bias the mixture toward
                    // well-behaved θ, misrepresenting the predictive.
                    format!(
                        "mixture component {} of {} (a retained posterior draw) \
                         could not be scored — the posterior holds θ the filter \
                         cannot track through this data.\n  If a stuck/divergent \
                         chain put it there, drop that chain with \
                         --exclude-chains; --draws 1 scores the plug-in posterior \
                         mean instead.\n  {}",
                        i + 1, n_theta, e)
                }
                PassError::FilterRejected(e) | PassError::Other(e) => e,
            });
            if r.is_err() {
                for p in &merged_paths { let _ = std::fs::remove_file(p); }
            }
            component_traces.push(r?);
        }
        let t = if is_mixture {
            combine_mixture(component_traces, seed_r)?
        } else {
            component_traces.into_iter().next().expect("one pass ran")
        };
        replicate_traces.push(t);
    }
    for p in &merged_paths { let _ = std::fs::remove_file(p); }

    // The filter-noise MC SE, from the replicate TOTAL elpds combined by
    // log-mean-exp with its delta-method SE (pomp practice; the PF
    // likelihood estimator is unbiased on the natural scale). The headline
    // elpd below is the SUM of per-step log-mean-exp values — the
    // pointwise-consistent aggregate (its steps sum to it, and the paired
    // Δ machinery reads the steps); the two aggregates agree in
    // expectation to O(per-step variance), and the SE measures the same
    // filter-noise scale either way.
    let totals = ReplicateTotals::of(&replicate_traces);
    let mut trace = if replicate_traces.len() >= 2 {
        combine_scored_traces(replicate_traces, derive.seed)?
    } else {
        replicate_traces.into_iter().next().expect("one replicate ran")
    };
    // Schema 5: each pass recorded its own single run; the trace as combined
    // is what this derivation did. The excluded chains are recorded only
    // where they applied — a θ drawn from a posterior cloud.
    let excluded_chains = match theta_source {
        ThetaSource::PosteriorMean { .. } | ThetaSource::PosteriorDraws { .. } =>
            selection.map(|s| s.excluded().iter().map(|c| c.0).collect()).unwrap_or_default(),
        ThetaSource::Supplied | ThetaSource::OptimizerWinner => Vec::new(),
    };
    let derivation = trace.derivation.as_mut().ok_or_else(|| format!(
        "internal error: `camdl pfilter` wrote a trace for {} with no derivation \
         block", segment.display()))?;
    derivation.replicates = derive.replicates;
    derivation.theta = theta_source;
    derivation.excluded_chains = excluded_chains;
    derivation.replicate_elpd = totals.elpd;
    derivation.replicate_elpd_by_stream = totals.by_stream;
    derivation.mc_se_elpd = totals.mc_se;
    // `--in-sample` waived this fit's sealed holdout: the trace says so.
    let declares_holdout = config.data.as_ref()
        .is_some_and(|d| d.holdout_after.is_some() || d.holdout.is_some());
    if derive.force_in_sample && declares_holdout {
        derivation.overrides.push(TraceOverride::InSampleForced);
    }

    // gh#585 (Stage 3.5): §3.7.3(c) — the `HoldOutTail` stamp is granted
    // only after checking, on the derived trace itself, that every scored
    // time lies strictly after the sealed training window. (a) and (b)
    // were established by `detect_sealed_holdout`. The guarantee is
    // precisely "no data leakage into θ": it cannot certify that the
    // modeler chose the model family without seeing the full curve.
    if let Some(h) = holdout {
        if trace.steps.is_empty() {
            return Err(format!(
                "fit at {}: the held-out window (t > {}) contains no scored \
                 observations — nothing lies past the training boundary in \
                 the bound data.",
                segment.display(), h.train_end));
        }
        if let Some(bad) = trace.steps.iter().find(|s| s.t <= h.train_end) {
            return Err(format!(
                "fit at {}: refusing the hold_out_tail stamp — scored \
                 observation at t = {} does not lie strictly after the \
                 sealed training window (train_end = {}).",
                segment.display(), bad.t, h.train_end));
        }
        trace.conditioning = Conditioning::HoldOutTail {
            train_end: h.train_end,
            theta_source: segment.file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| segment.display().to_string()),
        };
    }
    let streams = RowStreams::Derived {
        declared: model.observations.iter().map(|o| (o.name.clone(), o.source.clone())).collect(),
        bindings,
    };
    Ok(Derived { trace, data: recorded_data, freshness, streams })
}

/// The replicate totals a derived trace records (schema 5): each replicate's
/// total elpd and per-stream sums, in replicate order, and the filter-noise
/// Monte-Carlo SE from the totals.
///
/// The SE is the log-mean-exp delta-method SE of the replicate TOTAL elpds
/// (pomp practice; the PF likelihood estimator is unbiased on the natural
/// scale). The headline elpd is the SUM of per-step log-mean-exp values —
/// the pointwise-consistent aggregate (its steps sum to it, and the paired Δ
/// machinery reads the steps); the two aggregates agree in expectation to
/// O(per-step variance), and the SE measures the same filter-noise scale
/// either way. `None` for a single replicate.
#[derive(Debug, Clone, PartialEq)]
struct ReplicateTotals {
    elpd: Vec<f64>,
    by_stream: std::collections::BTreeMap<String, Vec<f64>>,
    mc_se: Option<f64>,
}

impl ReplicateTotals {
    fn of(replicates: &[PrequentialTrace]) -> Self {
        let elpd: Vec<f64> = replicates.iter().map(|t| t.elpd()).collect();
        let mut by_stream: std::collections::BTreeMap<String, Vec<f64>> =
            std::collections::BTreeMap::new();
        for t in replicates {
            for (stream, total) in t.elpd_by_stream() {
                by_stream.entry(stream).or_default().push(total);
            }
        }
        let mc_se = (elpd.len() >= 2)
            .then(|| sim::inference::types::logmeanexp_with_se(&elpd).1);
        ReplicateTotals { elpd, by_stream, mc_se }
    }
}

/// Combine per-draw filter traces into the posterior-mixture predictive
/// (§3.6 of the 2026-08-29 proposal; Stage 4.1).
///
/// The honest Bayesian predictive is the mixture over draws
/// `p̂(y_t|·) = (1/M) Σ_m p̂(y_t|·,θ_m)`, so per-step log scores combine
/// as `logsumexp_m − log M` over the per-draw *densities* — averaging the
/// per-draw log scores instead is a different, strictly lower number
/// (Jensen) that scores each θ separately rather than the model's actual
/// predictive. Stamped `Provenance::Posterior { n_draws }`.
fn combine_mixture(
    traces: Vec<PrequentialTrace>,
    pit_seed: u64,
) -> Result<PrequentialTrace, String> {
    use sim::inference::prequential::Provenance;
    let m = traces.len();
    let mut trace = combine_scored_traces(traces, pit_seed)?;
    trace.provenance = Provenance::Posterior { n_draws: m };
    Ok(trace)
}

/// The shared per-step combination behind [`combine_mixture`] (per-draw
/// predictives, Stage 4.1) and replicate derives (per-seed re-estimates
/// of ONE predictive, Stage 4.2). The arithmetic is the same log-mean-exp
/// either way; the meaning differs — a mixture density over draws vs the
/// unbiased-on-the-natural-scale combination of replicate PF density
/// estimates (pomp practice) — and the caller stamps the provenance
/// accordingly.
///
/// Per step: the joint log score = log-mean-exp over components; per
/// stream, the log score likewise, and the stream's predictive samples
/// pooled (the pooled cloud is a draw from the combined predictive), so its
/// fair CRPS, randomized PIT (fresh seeded draws) and interval are computed
/// on the pool; ESS is the MIN across components (the weakest effective
/// support); warnings deduplicated by kind.
///
/// Every component ran with the same data and scoring boundary, so the
/// step axes must be identical — anything else is a derive bug, not a
/// user error.
fn combine_scored_traces(
    traces: Vec<PrequentialTrace>,
    pit_seed: u64,
) -> Result<PrequentialTrace, String> {
    use sim::inference::prequential::{
        crps_sample_fair, pit_sample_randomized, PredInterval, PrequentialStep,
        StreamScore, PIT_RNG_STREAM,
    };
    let m = traces.len();
    assert!(m >= 2, "combine_scored_traces needs >= 2 components");
    let first = &traces[0];
    for t in &traces[1..] {
        let same_axis = t.steps.len() == first.steps.len()
            && t.steps.iter().zip(&first.steps).all(|(a, b)| {
                a.t == b.t
                    && a.per_stream.len() == b.per_stream.len()
                    && a.per_stream.iter().zip(&b.per_stream)
                        .all(|(x, y)| x.stream == y.stream && x.coverage == y.coverage)
            });
        if !same_axis {
            return Err(
                "internal error: mixture components scored different \
                 observation axes — same data, seed and boundary must give \
                 identical steps".into());
        }
    }

    // Fresh randomized-PIT draws for the pooled predictives, consumed in
    // build_trace's order (step by step, one per stream) for consistency.
    let mut pit_rng = sim::rng::StatefulRng::new_stream(pit_seed, PIT_RNG_STREAM);

    let mut steps = Vec::with_capacity(first.steps.len());
    for si in 0..first.steps.len() {
        let ls: Vec<f64> = traces.iter().map(|t| t.steps[si].log_score).collect();
        let (log_score, _) = sim::inference::types::logmeanexp_with_se(&ls);
        let ess = traces.iter().map(|t| t.steps[si].ess)
            .fold(f64::INFINITY, f64::min);

        let mut per_stream = Vec::with_capacity(first.steps[si].per_stream.len());
        for pi in 0..first.steps[si].per_stream.len() {
            let ss0 = &first.steps[si].per_stream[pi];
            let ls_s: Vec<f64> = traces.iter()
                .map(|t| t.steps[si].per_stream[pi].log_score).collect();
            let (log_score_s, _) = sim::inference::types::logmeanexp_with_se(&ls_s);
            let mut pooled_s: Vec<f64> = Vec::new();
            for t in &traces {
                pooled_s.extend_from_slice(&t.steps[si].per_stream[pi].y_pred_samples);
            }
            per_stream.push(StreamScore {
                stream: ss0.stream.clone(),
                y_obs: ss0.y_obs,
                log_score: log_score_s,
                crps: crps_sample_fair(&pooled_s, ss0.y_obs),
                pit: pit_sample_randomized(&pooled_s, ss0.y_obs, pit_rng.uniform()),
                interval: PredInterval::from_samples(&pooled_s),
                y_pred_samples: pooled_s,
                coverage: ss0.coverage,
            });
        }

        steps.push(PrequentialStep {
            t: first.steps[si].t,
            log_score, ess,
            per_stream,
        });
    }

    // Union of warnings, one per kind (the counts differ per component;
    // the first instance is representative).
    let mut warnings = Vec::new();
    for t in &traces {
        for w in &t.warnings {
            if !warnings.iter().any(|have|
                std::mem::discriminant(have) == std::mem::discriminant(w)) {
                warnings.push(w.clone());
            }
        }
    }

    Ok(PrequentialTrace {
        schema_version: first.schema_version,
        t0: first.t0,
        provenance: first.provenance,
        conditioning: first.conditioning.clone(),
        steps,
        warnings,
        score_from: first.score_from,
        pit_randomization_seed: Some(pit_seed),
        derivation: first.derivation.clone(),
        scored_streams: first.scored_streams.clone(),
    })
}

/// Up to `m` evenly thinned posterior draws as flat params TOMLs — the
/// mixture components of §3.6. `Ok(None)` when the fit has no posterior
/// cloud (an optimizer fit: a mixture is impossible, the plug-in winner
/// is the fallback). Thinning is an even stride over the retained
/// (chain-filtered) rows in file order; fewer rows than `m` uses them
/// all. Reads through the shared draws authority so `--exclude-chains`
/// applies once, at the same seam every other consumer uses.
fn thinned_draws_params_tomls(
    segment: &Path,
    selection: Option<&ChainSelection>,
    m: usize,
) -> Result<Option<(Vec<String>, ThetaSource)>, String> {
    let pdraws = match crate::posterior_draws::resolve_posterior_draws(
        &segment.to_string_lossy(), None,
    ) {
        Ok(p) => p.with_selection(selection.cloned()),
        // No cloud: the caller scores the point estimate instead. Any other
        // failure (a leaf refused for its chain numbering, gh#990) is an error,
        // not a fallback.
        Err(crate::posterior_draws::PosteriorDrawsError::NoPosterior(_)) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let (rows, _info) = pdraws.load_params_with_info()?;
    let n = rows.len();
    if n == 0 {
        return Err(format!("no posterior draws in {}", pdraws.draws_path.display()));
    }
    let m_eff = m.min(n);
    let mut out = Vec::with_capacity(m_eff);
    for i in 0..m_eff {
        let row = &rows[i * n / m_eff];
        let sorted: std::collections::BTreeMap<&String, &f64> = row.iter().collect();
        let mut s = String::new();
        s.push_str("# camdl compare: thinned posterior draw (mixture component)\n\n");
        for (name, v) in sorted {
            s.push_str(&format!(
                "{} = {}\n", name,
                crate::fit::runner::format_param_value(*v)));
        }
        out.push(s);
    }
    Ok(Some((out, ThetaSource::PosteriorDraws { n_used: m_eff, n_available: n })))
}

/// Merge a training file and its held-out tail into one scored series
/// (gh#585): compare's held-out derivation filters over training ∪
/// holdout. The two files must share a header line (the holdout file is
/// the same schema by declaration); holdout rows are strictly after the
/// training tail (validated when the fit ran), so appending preserves
/// time order.
fn merge_holdout_file(train: &str, holdout: &str, out: &Path) -> Result<(), String> {
    let t = std::fs::read_to_string(train)
        .map_err(|e| format!("reading training data {train}: {e}"))?;
    let h = std::fs::read_to_string(holdout)
        .map_err(|e| format!("reading holdout data {holdout}: {e}"))?;
    let t_header = t.lines().next().unwrap_or("");
    let h_header = h.lines().next().unwrap_or("");
    if t_header != h_header {
        return Err(format!(
            "[data.holdout] file {holdout} has a different header than its \
             training file {train}:\n  training: {t_header}\n  holdout:  {h_header}"));
    }
    let mut merged = String::with_capacity(t.len() + h.len());
    merged.push_str(t.trim_end());
    merged.push('\n');
    for line in h.lines().skip(1) {
        merged.push_str(line);
        merged.push('\n');
    }
    std::fs::write(out, merged)
        .map_err(|e| format!("writing merged series {}: {e}", out.display()))
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
/// authority. A Bayesian fit (its terminal method wrote a `draws.tsv`) plugs in
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
) -> Result<(String, ThetaSource), String> {
    match crate::posterior_draws::resolve_posterior_draws(&segment.to_string_lossy(), None) {
        Ok(pdraws) => posterior_mean_params_toml(&pdraws.with_selection(selection.cloned()))
            .map(|(toml, n_draws)| (toml, ThetaSource::PosteriorMean { n_draws })),
        // No posterior cloud → an optimizer fit; its θ̂ is the single winner.
        Err(crate::posterior_draws::PosteriorDrawsError::NoPosterior(_)) => {
            crate::fit::fit_summary::winner_params_toml(segment)
                .map(|toml| (toml, ThetaSource::OptimizerWinner))
        }
        // Anything else — a leaf refused for counting its chains from 1
        // (gh#990) — must not be scored at a winner instead.
        Err(e) => Err(e.into()),
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
fn posterior_mean_params_toml(pref: &PosteriorDrawsRef) -> Result<(String, usize), String> {
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
    Ok((out, n))
}

/// Paired Δ = sum_t (a_t − b_t); paired SE = the Newey–West HAC standard
/// error of that sum with the Harvey–Leybourne–Newbold small-sample
/// inflation (Stage 4.3 of the 2026-08-29 proposal, §3.4 remedy 2).
///
/// Loss differentials are autocorrelated exactly when a model is
/// misspecified — the interesting case — so the iid `√(T·Var_t(d_t))` is
/// anti-conservative there. The Diebold–Mariano standard: a Bartlett
/// long-run variance at the pinned `sandwich`-default lag
/// (`evidence::newey_west_lag`), the SE divided by the HLN factor
/// `√((T−1)/T)`, with `t_{T−1}` as the reference distribution (stated in
/// the footnote; the evidence gate stays the descriptive 2·se band).
/// At lag 0 / no autocorrelation this is the iid SE with the biased
/// variance. Returns (delta, se) or (NaN, NaN) if horizons mismatch.
fn paired_delta(a: &PrequentialTrace, b: &PrequentialTrace) -> (f64, f64) {
    let diffs = match paired_diffs(a, b) {
        Some(d) => d,
        None => return (f64::NAN, f64::NAN),
    };
    let delta: f64 = diffs.iter().sum();
    let se = crate::evidence::newey_west_se_of_sum(&diffs)
        / crate::evidence::hln_factor(diffs.len());
    (delta, se)
}

/// The per-step loss differentials `d_t = a_t − b_t`, or `None` on a
/// horizon mismatch — the shared input of the paired SE and the lag-1
/// serial-dependence diagnostic.
fn paired_diffs(a: &PrequentialTrace, b: &PrequentialTrace) -> Option<Vec<f64>> {
    if a.n_scored() != b.n_scored() || a.n_scored() == 0 {
        return None;
    }
    Some(a.steps.iter().zip(&b.steps)
        .map(|(x, y)| x.log_score - y.log_score)
        .collect())
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
    Accuracy::PerStream(streams.into_iter().map(|stream| {
        let delta_mean_crps = (!is_base).then(|| {
            let a: Vec<(f64, f64)> = row.trace.stream_scores(stream).map(|(t, s)| (t, s.crps)).collect();
            let b: Vec<(f64, f64)> = base.trace.stream_scores(stream).map(|(t, s)| (t, s.crps)).collect();
            let same_times = a.len() == b.len() && a.iter().zip(&b).all(|(x, y)| x.0 == y.0);
            let diffs: Vec<f64> = a.iter().zip(&b)
                .map(|(x, y)| x.1 - y.1)
                .filter(|d| d.is_finite())
                .collect();
            (same_times && !diffs.is_empty())
                .then(|| diffs.iter().sum::<f64>() / diffs.len() as f64)
        }).flatten();
        StreamAccuracy {
            stream: stream.to_string(),
            mean_crps: row.trace.mean_crps(stream),
            delta_mean_crps,
            pit_cov90: row.trace.pit_coverage(stream, 0.90),
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
fn accuracy_cells(sa: &StreamAccuracy, want_n: bool, want_crps: bool, want_pit: bool,
                  delta_shown: bool) -> Vec<String> {
    let mut cells = Vec::new();
    if want_n { cells.push(sa.mean_crps.n.to_string()); }
    if want_crps {
        cells.push(format!("{:.3}", sa.mean_crps.value));
        cells.push(match (delta_shown, sa.delta_mean_crps) {
            (true, Some(d)) => format!("{:+.3}", d),
            _ => "—".into(),
        });
    }
    if want_pit { cells.push(format!("{:.2}", sa.pit_cov90.value)); }
    cells
}

/// The per-stream block's header, with the wanted metric columns.
fn accuracy_header(want_crps: bool, want_pit: bool) -> Vec<String> {
    let mut h: Vec<String> = vec!["stream".into(), "model".into(), "n".into()];
    if want_crps { h.push("mean CRPS".into()); h.push("Δ mean CRPS".into()); }
    if want_pit { h.push("PIT_cov90".into()); }
    h
}

/// The per-stream block's body: one line per stream × row, rows in render
/// order, `None` when there is nothing to show (no wanted metric, or no row
/// with a per-stream breakdown).
fn accuracy_block(rows: &[Row], order: &[usize], accuracy: &[Accuracy],
                  want_crps: bool, want_pit: bool, t_mismatch: bool) -> Option<Vec<Vec<String>>> {
    if !(want_crps || want_pit) { return None; }
    let streams = accuracy_streams(accuracy);
    if streams.is_empty() { return None; }
    let mut body = Vec::new();
    for stream in streams {
        for &i in order {
            let Accuracy::PerStream(v) = &accuracy[i] else { continue };
            let Some(sa) = v.iter().find(|s| s.stream == stream) else { continue };
            let mut line = vec![stream.to_string(), rows[i].name.clone()];
            line.extend(accuracy_cells(sa, true, want_crps, want_pit, !t_mismatch));
            body.push(line);
        }
    }
    Some(body)
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
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");
    let (accuracy, single) = accuracy_view(rows, base_idx);

    // Δelpd (nats) is the primary machine-readable column; "evidence"
    // (decibans + Jeffreys label) is the human-interpretable alongside —
    // see docs/dev/proposals/2026-04-23-evidence-in-decibans.md §Scope.
    // CRPS and PIT columns only when every row scored the same one stream;
    // otherwise they are per stream, in the block below the table.
    let mut header = vec!["Model".to_string(), "T_score".into(), "elpd".into(),
        "Δelpd".into(), "LR".into(), "se(Δ)".into(), "evidence".into()];
    if single.is_some() {
        if want_crps { header.push("crps".into()); header.push("Δcrps".into()); }
        if want_pit  { header.push("PIT_cov90".into()); }
    }

    let base = &rows[base_idx].trace;
    let order = render_order(rows, base_idx, t_mismatch);
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
    for &i in &order {
        let r = &rows[i];
        let elpd = r.trace.elpd();
        let mut row = vec![
            r.name.clone(),
            format!("{}", r.trace.n_scored()),
            format!("{:.2}", elpd),
        ];
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
            row.push(crate::evidence::evidence_cell(d, se, mc_se_delta(r, &rows[base_idx])));
        }
        if let (Some(_), Accuracy::PerStream(v)) = (&single, &accuracy[i]) {
            row.extend(accuracy_cells(&v[0], false, want_crps, want_pit,
                i != base_idx && !t_mismatch));
        }
        body.push(row);
    }

    let mut out = String::new();
    if let Some(leaf) = scored_leaf(rows) {
        out.push_str(&format!("Stream: {leaf} — every row scored on this stream alone.\n\n"));
    }
    push_aligned(&mut out, &header, &body);
    if single.is_none() {
        if let Some(block) = accuracy_block(rows, &order, &accuracy, want_crps, want_pit, t_mismatch) {
            out.push_str("\nPer-stream accuracy and calibration:\n");
            push_aligned(&mut out, &accuracy_header(want_crps, want_pit), &block);
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
    let mut lines: Vec<String> = checks.overrides.iter().map(Override::line).collect();
    // What the checks could not cover.
    for note in [&checks.data_unchecked, &checks.data_change_unchecked,
                 &checks.data_compared_on_prefix, &checks.filter_settings_unchecked]
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
            lines.push(format!("ⓘ {}: {:?}", r.name, w));
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
            let a1 = crate::evidence::lag1_autocorrelation(&d);
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
    let want_crps = metrics.iter().any(|m| m == "crps");
    let want_pit  = metrics.iter().any(|m| m == "pit_cov90" || m == "pit");
    let (accuracy, single) = accuracy_view(rows, base_idx);
    let md_table = |out: &mut String, header: &[String], body: &[Vec<String>]| {
        out.push_str(&format!("| {} |\n", header.join(" | ")));
        out.push_str(&format!("|{}|\n",
            header.iter().map(|_| "---").collect::<Vec<_>>().join("|")));
        for cells in body {
            out.push_str(&format!("| {} |\n", cells.join(" | ")));
        }
    };

    let mut header: Vec<String> = ["Model", "T_score", "elpd", "Δelpd", "LR", "se(Δ)", "evidence"]
        .iter().map(|h| h.to_string()).collect();
    if single.is_some() {
        if want_crps { header.push("crps".into()); header.push("Δcrps".into()); }
        if want_pit  { header.push("PIT_cov90".into()); }
    }

    let base = &rows[base_idx].trace;
    // Same render order as the table renderer.
    let order = render_order(rows, base_idx, t_mismatch);
    let mut body: Vec<Vec<String>> = Vec::with_capacity(rows.len());
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
            let (d, se) = paired_delta(&r.trace, base);
            cells.push(format!("{:+.2}", d));
            cells.push(fmt_lr(d.exp()));
            cells.push(format!("{:.2}", se));
            cells.push(crate::evidence::evidence_cell(d, se, mc_se_delta(r, &rows[base_idx])));
        }
        if let (Some(_), Accuracy::PerStream(v)) = (&single, &accuracy[i]) {
            cells.extend(accuracy_cells(&v[0], false, want_crps, want_pit,
                i != base_idx && !t_mismatch));
        }
        body.push(cells);
    }
    let mut out = String::new();
    if let Some(leaf) = scored_leaf(rows) {
        out.push_str(&format!("### Stream `{leaf}`\n\n_Every row scored on this stream alone._\n\n"));
    }
    md_table(&mut out, &header, &body);
    if single.is_none() {
        if let Some(block) = accuracy_block(rows, &order, &accuracy, want_crps, want_pit, t_mismatch) {
            out.push_str("\n**Per-stream accuracy and calibration**\n\n");
            md_table(&mut out, &accuracy_header(want_crps, want_pit), &block);
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

/// The `--streams` JSON document: the selected streams in order, and one
/// [`json_document`] per stream, each naming its stream. Nothing else: no
/// field combines streams.
fn render_json_tables(tables: &[Table], leaves: &[String], metrics: &[String]) -> String {
    let out = serde_json::json!({
        "streams_scored": leaves,
        "tables": tables.iter()
            .map(|t| json_document(&t.rows, t.base_idx, metrics, &t.checks))
            .collect::<Vec<_>>(),
    });
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
        let stream_json = |sa: &StreamAccuracy| json!({
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
            let db = d_elpd * crate::evidence::NATS_TO_DB;
            (option_finite(db),
             serde_json::json!(crate::evidence::evidence_label(d_elpd, se_elpd,
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
                        crate::evidence::lag1_autocorrelation(&d)))
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
        "overrides": checks.overrides.iter().map(Override::to_json).collect::<Vec<_>>(),
        // The bound-data check (gh#713): present only under
        // --allow-data-mismatch, and the rows the check could not cover.
        "data_mismatch": checks.data_mismatch,
        "data_unchecked": checks.data_unchecked,
        // A training file edited after its fit ran (present only under
        // --allow-changed-data), and the files that could not be checked.
        "data_changed": checks.data_changed,
        "data_change_unchecked": checks.data_change_unchecked,
        // Rows whose recorded hashes are 8 characters (an older fit.meta.json):
        // both data checks compared them on that prefix.
        "data_compared_on_prefix": checks.data_compared_on_prefix,
        // Particle-filter settings that differ across rows (present only under
        // --allow-mismatched-filter-settings), and the rows that record none.
        "filter_settings_mismatch": checks.filter_settings_mismatch,
        "filter_settings_unchecked": checks.filter_settings_unchecked,
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

    /// §3.6 (Stage 4.1): the mixture combines per-draw *densities*, pools
    /// each stream's samples, takes the min ESS, and stamps
    /// `Posterior { n_draws }`.
    #[test]
    fn combine_mixture_averages_densities_and_pools_samples() {
        use sim::inference::prequential::{
            crps_sample_fair, PredInterval, PrequentialStep, PrequentialTrace,
            PrequentialWarning, Provenance, StreamScore,
        };
        let component = |ls: f64, samples: Vec<f64>, ess: f64| PrequentialTrace {
            schema_version: 3,
            t0: 0,
            provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample,
            steps: vec![PrequentialStep {
                t: 7.0,
                log_score: ls,
                ess,
                per_stream: vec![StreamScore {
                    stream: "cases".into(),
                    y_obs: 3.0,
                    interval: PredInterval::from_samples(&samples),
                    y_pred_samples: samples,
                    log_score: ls,
                    crps: 0.0,
                    pit: 0.0,
                    coverage: Default::default(),
                }],
            }],
            warnings: vec![PrequentialWarning::StartsAtPrior],
            score_from: None,
            pit_randomization_seed: Some(1),
            derivation: None, scored_streams: None,
        };
        let a = component(0.2_f64.ln(), vec![1.0, 2.0], 40.0);
        let b = component(0.4_f64.ln(), vec![3.0, 4.0], 25.0);
        let mixed = combine_mixture(vec![a, b], 9).expect("same axis combines");

        // Density average, NOT mean of logs: log((0.2 + 0.4)/2) = log(0.3).
        let got = mixed.steps[0].log_score;
        let want = 0.3_f64.ln();
        assert!((got - want).abs() < 1e-12,
            "mixture log score must be log-mean-density: {got} vs {want}");
        let mean_of_logs = (0.2_f64.ln() + 0.4_f64.ln()) / 2.0;
        assert!(got > mean_of_logs,
            "Jensen: the mixture score is strictly above the mean of logs");
        assert!((mixed.steps[0].per_stream[0].log_score - want).abs() < 1e-12,
            "and so is the stream's own");

        // Pooled cloud, min ESS, CRPS over the pool.
        let s = &mixed.steps[0].per_stream[0];
        assert_eq!(s.y_pred_samples, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mixed.steps[0].ess, 25.0);
        let want_crps = crps_sample_fair(&[1.0, 2.0, 3.0, 4.0], 3.0);
        assert!((s.crps - want_crps).abs() < 1e-12);

        // Stamps and carried metadata.
        assert_eq!(mixed.provenance, Provenance::Posterior { n_draws: 2 });
        assert_eq!(mixed.conditioning, Conditioning::InSample);
        assert_eq!(mixed.pit_randomization_seed, Some(9));
        assert_eq!(mixed.warnings.len(), 1, "duplicate warnings deduped by kind");
    }

    #[test]
    fn combine_mixture_refuses_mismatched_axes() {
        use sim::inference::prequential::{PrequentialStep, PrequentialTrace, Provenance};
        let trace_at = |t: f64| PrequentialTrace {
            schema_version: 3,
            t0: 0,
            provenance: Provenance::PlugIn,
            conditioning: Conditioning::InSample,
            steps: vec![PrequentialStep {
                t,
                log_score: -1.0,
                ess: 10.0,
                per_stream: vec![],
            }],
            warnings: vec![],
            score_from: None,
            pit_randomization_seed: None,
            derivation: None, scored_streams: None,
        };
        let err = combine_mixture(vec![trace_at(7.0), trace_at(14.0)], 1)
            .expect_err("different step times must refuse");
        assert!(err.contains("different observation axes"), "{err}");
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
        use sim::inference::prequential::PrequentialWarning;
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
            .expect_err("a differing data file must be refused").finding;
        assert!(err.contains("south"), "names the stream that differs: {err}");
        assert!(!err.contains("north"), "and not the one that agrees: {err}");
        assert!(err.contains("'a'") && err.contains("'b'"), "names both fits: {err}");
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        rows[0].data = digests(&[("north", "aaa"), ("south", "bbb")]);
        rows[1].data = revised;
        let refusal = preflight(&rows, &Waivers::default())
            .expect_err("refused without the flag");
        assert!(refusal.contains("south") && refusal.contains("--allow-data-mismatch"),
            "and how to proceed deliberately: {refusal}");

        // A stream one fit bound and the other did not is the same failure.
        let extra = digests(&[("north", "aaa"), ("south", "bbb"), ("east", "ddd")]);
        let err = check_shared_data(&[("a", &same_a), ("b", &extra)])
            .expect_err("an extra bound stream must be refused").finding;
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
        use sim::inference::Coverage;
        let iv = |s: f64, e: f64| Coverage::Interval { start: s, stop: e };

        // (i) Different stream sets at shared t = 14.
        let short = row_with_streams("short", &[7.0, 14.0], &["north"]);
        let long = row_with_streams("long", &[7.0, 14.0, 21.0], &["north", "south"]);
        let err = check_shared_times(&[short, long])
            .expect_err("different stream sets at a shared time are refused");
        assert!(err.contains("did not score the same streams") && err.contains("south")
                && err.contains("t=7"), "{err}");

        // (ii) Same stream, different coverage at shared t = 14; the long
        // trace's extra step at 21 is not consulted.
        let short = row_covering("short", &[7.0, 14.0], "cases",
            &[iv(0.0, 7.0), iv(12.0, 14.0)]);
        let long = row_covering("long", &[7.0, 14.0, 21.0], "cases",
            &[iv(0.0, 7.0), iv(7.0, 14.0), iv(14.0, 21.0)]);
        let err = check_shared_times(&[short, long])
            .expect_err("different coverage at a shared time is refused");
        assert!(err.contains("over different windows") && err.contains("t=14"), "{err}");

        // Shared times joined by t, not position: the short trace's t = 14
        // pairs with the long trace's t = 14 although it sits at index 0.
        let short = row_covering("short", &[14.0], "cases", &[iv(12.0, 14.0)]);
        let long = row_covering("long", &[7.0, 14.0], "cases", &[iv(0.0, 7.0), iv(7.0, 14.0)]);
        assert!(check_shared_times(&[short, long]).is_err(), "joined by time");

        // (iii) Same streams and coverage at every shared time: it renders,
        // with the Δ columns suppressed as before.
        let short = row_covering("short", &[7.0, 14.0], "cases", &[iv(0.0, 7.0), iv(7.0, 14.0)]);
        let long = row_covering("long", &[7.0, 14.0, 21.0], "cases",
            &[iv(0.0, 7.0), iv(7.0, 14.0), iv(14.0, 21.0)]);
        let rows = vec![short, long];
        check_shared_times(&rows).expect("agreeing shared times render");
        let table = render_table(&rows, 0, &["elpd".to_string()], &checked(&rows));
        assert!(table.contains("Δ columns suppressed"), "{table}");
    }

    /// A training file's fresh full SHA-256 against the hash its fit
    /// recorded: equal bytes match whether the record is 8 or 64 hex
    /// characters, different bytes are a change naming both short hashes,
    /// and no record is "not checked", never a silent pass.
    #[test]
    fn a_training_file_is_checked_against_its_recorded_hash_by_prefix() {
        let fresh = crate::hashing::sha256_hex(b"time\tcases\n1\t2\n");
        let other = crate::hashing::sha256_hex(b"time\tcases\n1\t3\n");

        assert_eq!(check_file_against_record(&fresh, Some(&fresh)), FileCheck::Matches,
            "a 64-character record of the same bytes matches");
        assert_eq!(check_file_against_record(&fresh, Some(&fresh[..8])), FileCheck::Matches,
            "an 8-character record (today's sidecar) matches by prefix");
        assert_eq!(check_file_against_record(&fresh, Some(&fresh[..8].to_uppercase())),
            FileCheck::Matches, "hex case is not a difference");

        for recorded in [&other[..], &other[..8]] {
            assert_eq!(check_file_against_record(&fresh, Some(recorded)),
                FileCheck::Changed { recorded: other[..8].to_string(),
                                     current: fresh[..8].to_string() },
                "different bytes are a change, at either record length");
        }

        assert!(matches!(check_file_against_record(&fresh, None),
            FileCheck::NotChecked(why) if why.contains("recorded no hash")));
        assert!(matches!(check_file_against_record(&fresh, Some("")),
            FileCheck::NotChecked(_)), "an empty record is not a match");
    }

    /// The freshness of a derived row, end to end over real files: the
    /// keys are sources, a source the sidecar lacks is
    /// "not checked", a row with no sidecar hashes is not checked as a
    /// whole, and the notes say each in every format.
    #[test]
    fn training_freshness_names_changed_and_unchecked_files_in_every_format() {
        let dir = tempfile::tempdir().unwrap();
        let north = dir.path().join("north.tsv");
        let south = dir.path().join("south.tsv");
        std::fs::write(&north, "time\tcases\n1\t2\n").unwrap();
        std::fs::write(&south, "time\tcases\n1\t5\n").unwrap();
        let streams: indexmap::IndexMap<String, String> = [
            ("north".to_string(), north.to_string_lossy().into_owned()),
            ("south".to_string(), south.to_string_lossy().into_owned()),
        ].into_iter().collect();

        // Both recorded and unchanged: nothing to say.
        let south_hash = crate::hashing::sha256_hex(b"time\tcases\n1\t5\n");
        let north_hash = crate::hashing::sha256_hex(b"time\tcases\n1\t2\n");
        let clean = check_training_freshness(&streams,
            &digests(&[("north", &north_hash[..8]), ("south", &south_hash[..8])]));
        assert_eq!(clean, DataFreshness::default());

        // `south` edited since the fit; `north` has no record.
        let fresh = check_training_freshness(&streams,
            &digests(&[("south", "deadbeef")]));
        assert_eq!(fresh.findings.len(), 2, "{fresh:?}");
        let mut rows = vec![
            row_at("a", &[7.0, 14.0, 21.0], &[-6.0, -6.0, -6.0]),
            row_at("b", &[7.0, 14.0, 21.0], &[-5.0, -6.0, -6.0]),
        ];
        rows[0].freshness = fresh;
        rows[1].freshness = check_training_freshness(&streams,
            &DataIdentity::Unknown("its fit.meta.json records no data hashes".into()));

        let changed = changed_data(&rows).map(|m| m.note(Waiver::ChangedData)).expect("south changed");
        assert!(changed.contains("'a' stream 'south'") && changed.contains("deadbeef")
                && changed.contains(&south_hash[..8]) && !changed.contains("north"),
            "{changed}");
        let unchecked = data_change_unchecked_note(&rows).expect("north and b unchecked");
        assert!(unchecked.contains("'a' stream 'north'")
                && unchecked.contains("recorded no hash")
                && unchecked.contains("'b' (its fit.meta.json records no data hashes)"),
            "{unchecked}");

        let metrics = vec!["elpd".to_string()];
        for (fmt, text) in [("table", render_table(&rows, 0, &metrics, &checked(&rows))),
                            ("md", render_md(&rows, 0, &metrics, &checked(&rows)))] {
            assert!(text.contains("changed after the fit ran")
                    && text.contains("--allow-changed-data was set"),
                "the {fmt} rendering states the change:\n{text}");
            assert!(text.contains("not checked for changes since the fit ran"),
                "and what was not checked:\n{text}");
        }
        let v: serde_json::Value =
            serde_json::from_str(&render_json(&rows, 0, &metrics, &checked(&rows))).unwrap();
        assert!(v["data_changed"].as_str().is_some_and(|s| s.contains("south")), "{v}");
        assert!(v["data_change_unchecked"].as_str().is_some_and(|s| s.contains("north")), "{v}");
    }

    fn derivation(particles: usize, replicates: usize)
        -> sim::inference::prequential::Derivation
    {
        sim::inference::prequential::Derivation {
            particles, replicates, dt: 1.0, seed: 1,
            theta: sim::inference::prequential::ThetaSource::OptimizerWinner,
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

        let totals = ReplicateTotals::of(&[a.clone(), b.clone()]);
        assert_eq!(totals.elpd, vec![a.elpd(), b.elpd()]);
        assert_eq!(totals.elpd, vec![-16.0, -18.0]);
        assert_eq!(totals.by_stream["north"], vec![-9.0, -9.0]);
        assert_eq!(totals.by_stream["south"], vec![-9.0, -11.0]);
        assert_eq!(totals.mc_se,
            Some(sim::inference::types::logmeanexp_with_se(&totals.elpd).1),
            "the SE is the log-mean-exp SE of the stored totals");
        assert!(totals.mc_se.is_some_and(|se| se > 0.0));

        let one = ReplicateTotals::of(&[a]);
        assert_eq!(one.elpd.len(), 1);
        assert_eq!(one.by_stream["north"].len(), 1);
        assert_eq!(one.mc_se, None, "a single replicate has no filter-noise SE");
    }

    /// Give a test row the filter-noise SE a replicated derive records, or
    /// (for `None`) no derivation block at all, as an explicit trace has.
    fn set_mc_se(row: &mut Row, mc: Option<f64>) {
        row.trace.derivation = mc.map(|se| sim::inference::prequential::Derivation {
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
        assert!(filter_settings_mismatch(&rows).map(|m| m.note(Waiver::MismatchedFilterSettings)).is_none(), "equal settings");
        assert!(filter_settings_unchecked_note(&rows).is_none(), "both recorded");

        rows[1].trace.derivation = Some(derivation(200, 5));
        let note = filter_settings_mismatch(&rows).map(|m| m.note(Waiver::MismatchedFilterSettings)).expect("particles differ");
        assert!(note.contains("'a'") && note.contains("'b'")
                && note.contains("particles 1000") && note.contains("particles 200"),
            "names both rows and both values: {note}");
        assert!(note.contains("biased downward"), "and why it matters: {note}");
        assert!(!note.contains("replicates 5 vs"), "replicates agree: {note}");

        rows[1].trace.derivation = Some(derivation(1000, 1));
        let note = filter_settings_mismatch(&rows).map(|m| m.note(Waiver::MismatchedFilterSettings)).expect("replicates differ");
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

        assert!(filter_settings_mismatch(&rows).map(|m| m.note(Waiver::MismatchedFilterSettings)).is_none(),
            "an unrecorded side cannot contradict anything");
        let note = filter_settings_unchecked_note(&rows).expect("old is unchecked");
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
            changed_data: true, mismatched_filter_settings: true, in_sample_ablation: true,
        }
    }

    /// The preflight outcome `cmd_compare` would hand the renderers for
    /// `rows`, every flag given.
    fn checked(rows: &[Row]) -> Preflight {
        preflight(rows, &every_waiver()).expect("every check is waived")
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
                sim::inference::prequential::Provenance::Posterior { n_draws: 4 },
            Waiver::DataMismatch => {
                rows[0].data = digests(&[("cases", "aaaaaaaa11")]);
                rows[1].data = digests(&[("cases", "bbbbbbbb22")]);
            }
            Waiver::ChangedData => rows[1].freshness.findings.push(FileFinding {
                stream: "cases".into(), path: "cases.tsv".into(),
                check: FileCheck::Changed { recorded: "aaaaaaaa".into(), current: "bbbbbbbb".into() },
            }),
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

    const ALL_WAIVERS: [Waiver; 6] = [
        Waiver::MismatchedHorizon, Waiver::MixedProvenance, Waiver::DataMismatch,
        Waiver::ChangedData, Waiver::MismatchedFilterSettings, Waiver::InSampleAblation,
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
            assert!(preflight(&rows, &Waivers::default()).is_err(),
                "{w:?}: refused without its flag");
            let checks = checked(&rows);
            assert_eq!(checks.overrides.iter().map(|o| o.waiver).collect::<Vec<_>>(), vec![w],
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
        assert!(checks.overrides.is_empty(), "{checks:?}");
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
            name: name.into(), path: name.into(), trace,
            data: DataIdentity::Unknown("a test fixture".into()),
            freshness: DataFreshness::default(),
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
        use sim::inference::prequential::{Conditioning, PrequentialStep, Provenance};
        let steps = times.iter().zip(log_scores).map(|(&t, &ls)| PrequentialStep {
            t,
            log_score: ls,
            ess: 900.0,
            per_stream: Vec::new(),
        }).collect();
        Row {
            name: name.to_string(),
            path: name.to_string(),
            freshness: DataFreshness::default(),
            streams: RowStreams::Explicit,
            data: DataIdentity::Unknown("a test fixture, built from scores alone".into()),
            trace: PrequentialTrace {
                schema_version: 1,
                t0: 1,
                provenance: Provenance::PlugIn,
                conditioning: Conditioning::InSample,
                steps,
                warnings: Vec::new(),
                score_from: None,
                pit_randomization_seed: None,
                derivation: None, scored_streams: None,
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
                coverage: Default::default(),
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
        coverage: &[sim::inference::Coverage],
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
        use sim::inference::Coverage;
        let iv = |s: f64, e: f64| Coverage::Interval { start: s, stop: e };
        // Rows closing at 4, 6, 7. Gapped: [5,6) with (4,5) covered by
        // nothing. Merged: [4,6) as one two-day row. Same stops, same count of
        // scored steps — the ambiguity a missing row used to hide.
        let t = [4.0, 6.0, 7.0];
        let gapped = [iv(3.0, 4.0), iv(5.0, 6.0), iv(6.0, 7.0)];
        let merged = [iv(3.0, 4.0), iv(4.0, 6.0), iv(6.0, 7.0)];

        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_covering("b", &t, "cases", &gapped);
        check_shared_observation_axis(&[a, b]).expect("identical windows compare");

        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_covering("b", &t, "cases", &merged);
        let err = check_shared_observation_axis(&[a, b])
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
        let err = check_shared_observation_axis(&[a, b])
            .expect_err("an instant reading is not a flow");
        assert!(err.contains("instant") && err.contains("[3, 4)"), "{err}");

        // Two traces that never recorded coverage carry nothing to check.
        let a = row_with_streams("a", &t, &["cases"]);
        let b = row_with_streams("b", &t, &["cases"]);
        check_shared_observation_axis(&[a, b])
            .expect("two traces with no coverage record are not a mismatch");

        // One recorded, one not: unverifiable, and the message says which
        // side to re-derive rather than inventing a window mismatch.
        let a = row_covering("a", &t, "cases", &gapped);
        let b = row_with_streams("b", &t, &["cases"]);
        let err = check_shared_observation_axis(&[a, b])
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
            name: "pgas".into(),
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
        let (all, n_all) = posterior_mean_params_toml(&pref).unwrap();
        assert_eq!(beta_of(&all), 5.0, "all-chain θ̂ pools both chains");
        assert_eq!(n_all, 8, "the mean is over all 8 draws");

        // Drop the stuck chain 1 → mean beta = 1.0 over the retained chain.
        let sel = ChainSelection::parse_exclude("1").unwrap();
        let (sub, n_sub) =
            posterior_mean_params_toml(&pref.clone().with_selection(Some(sel))).unwrap();
        assert_eq!(beta_of(&sub), 1.0, "subset θ̂ is the mean over the retained chain");
        assert_eq!(n_sub, 4, "and over the 4 retained draws");
        assert_ne!(
            beta_of(&all),
            beta_of(&sub),
            "excluding a stuck chain must change θ̂ (hence the prequential score)"
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
        let c = parse_cohort_exclude(&["a/fit.toml:3".to_string()], &names).unwrap();
        assert_eq!(c.for_fit("a/fit.toml").unwrap().excluded_csv(), "3");
        assert!(c.for_fit("b/fit.toml").is_none(), "the other fit keeps all chains");

        // The bare basename is no longer a fit name, and says so.
        let e = parse_cohort_exclude(&["fit.toml:3".to_string()], &names).unwrap_err();
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
        // What derive_prequential records on the combined trace.
        let totals = ReplicateTotals::of(&reps);
        let mut combined = combine_scored_traces(reps.clone(), 1).unwrap();
        let mut d = derivation(100, reps.len());
        d.replicate_elpd = totals.elpd.clone();
        d.replicate_elpd_by_stream = totals.by_stream.clone();
        d.mc_se_elpd = totals.mc_se;
        combined.derivation = Some(d);
        let mut row = row_with_streams("a", &t, &[]);
        row.trace = combined;

        for leaf in ["north", "south"] {
            let by_hand: Vec<f64> = reps.iter()
                .map(|rep| sim::inference::prequential::project_onto_stream(rep, leaf).unwrap().elpd())
                .collect();
            let want = sim::inference::types::logmeanexp_with_se(&by_hand).1;
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
        let err = check_shared_observation_axis(&[joint.clone(), projected.clone()]).unwrap_err();
        assert!(err.contains("'joint' is scored jointly") && err.contains("'projected' is re-scored on cases alone"),
            "{err}");
        // Under mismatched horizons the same check holds at shared times.
        assert!(check_shared_times(&[joint.clone(), projected]).is_err());
        let both = [joint.onto("cases").unwrap(), joint.onto("cases").unwrap()];
        check_shared_observation_axis(&both).expect("two projections onto one leaf compare");
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
        let refusal = preflight(&rows, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("different observed data") && refusal.contains("'deaths'"),
            "{refusal}");

        let on_cases: Vec<Row> = rows.iter().map(|r| r.onto("cases").unwrap()).collect();
        let checks = preflight(&on_cases, &Waivers::default())
            .expect("the cases files agree, and only they are compared");
        assert!(checks.overrides.is_empty() && checks.data_mismatch.is_none(), "{checks:?}");
        let note = checks.data_unchecked.as_deref().unwrap_or_default();
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
        let refusal = preflight(&on_deaths, &Waivers::default()).unwrap_err();
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
        let refusal = preflight(&on_cases, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'b' records no hash for source 'cases'")
                && refusal.contains("--allow-data-mismatch"), "{refusal}");
        let waived = Waivers { data_mismatch: true, ..Default::default() };
        let checks = preflight(&on_cases, &waived).unwrap();
        assert_eq!(checks.overrides.len(), 1);
        assert_eq!(checks.overrides[0].mismatch.rows, ["b"]);

        // Both rows lacking it is no better: nothing matched.
        rows[0].data = digests(&[("deaths", "dddddddd")]);
        let on_cases: Vec<Row> = rows.iter().map(|r| r.onto("cases").unwrap()).collect();
        let refusal = preflight(&on_cases, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'a' records no hash") && refusal.contains("'b' records no hash"),
            "{refusal}");
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
        let refusal = preflight(&on_cases, &Waivers::default()).unwrap_err();
        assert!(refusal.contains("assimilated different streams")
                && refusal.contains("'full' assimilated cases, deaths")
                && refusal.contains("'cases_only' assimilated cases")
                && refusal.contains("holdout_after")
                && refusal.contains("--allow-in-sample-ablation")
                && refusal.contains("fewer"), "{refusal}");

        // The same set: no refusal.
        let twin = derived_row("twin", &CASES_DEATHS, &["cases", "deaths"]);
        preflight(&[full.onto("cases").unwrap(), twin.onto("cases").unwrap()], &Waivers::default())
            .expect("rows that assimilated the same streams compare in-sample");

        // Held out: no refusal.
        let held = |r: Row| {
            let mut p = r.onto("cases").unwrap();
            p.trace.conditioning = Conditioning::HoldOutTail {
                train_end: 3.0, theta_source: r.name.clone() };
            p
        };
        let checks = preflight(&[held(full.clone()), held(cases_only.clone())], &Waivers::default())
            .expect("held-out rows need no flag");
        assert!(checks.overrides.is_empty(), "{checks:?}");

        // An explicit trace does not record what its fit assimilated, and
        // counts as different even from a row with the same streams.
        let mut explicit = full.clone();
        explicit.name = "pq".into();
        explicit.streams = RowStreams::Explicit;
        assert_eq!(explicit.assimilated(), None);
        let refusal = preflight(&[full.onto("cases").unwrap(), explicit.onto("cases").unwrap()],
                                &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'pq' (not recorded: an explicit prequential.json)"), "{refusal}");
        // Two unknown sets are not known to be equal either.
        let mut explicit_too = explicit.clone();
        explicit_too.name = "pq2".into();
        let refusal = preflight(&[explicit.onto("cases").unwrap(), explicit_too.onto("cases").unwrap()],
                                &Waivers::default()).unwrap_err();
        assert!(refusal.contains("'pq' (not recorded") && refusal.contains("'pq2' (not recorded"),
            "each unknown set is its own entry: {refusal}");

        // A joint comparison is not an ablation: the check is not made there.
        let twin_joint = [full.clone(), twin.clone()];
        preflight(&twin_joint, &Waivers::default()).expect("a joint comparison is untouched");
    }

    /// The JSON document of [`two_stream_rows`], captured from the renderer
    /// before `--streams` changed it. Without `--streams` the document must
    /// stay byte for byte what it was.
    const TWO_STREAM_JSON: &str = r#"{
  "baseline": "a",
  "data_change_unchecked": null,
  "data_changed": null,
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
        let checks = preflight(&rows, &Waivers::default()).unwrap();
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
        let checks = preflight(&same, &Waivers::default()).unwrap();
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
            &render_json_tables(&[ablation_table(), ablation_table()], &["cases".into(), "deaths".into()], &metrics)).unwrap();
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
        let err = check_shared_observation_axis(&[full.clone(), cases_only]).unwrap_err();
        assert!(err.contains("--streams cases"), "{err}");
        assert!(!err.contains("--allow-mismatched-horizon"), "{err}");
        // Nothing shared, nothing suggested.
        let hosp = row_with_streams("hosp", &t, &["hospital"]);
        let err = check_shared_observation_axis(&[full, hosp]).unwrap_err();
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
        let err = preflight(&[full.clone(), late.clone()], &waived).unwrap_err();
        assert!(err.contains("T_score") && err.contains("ic_free") && err.contains("one step later"),
            "{err}");
        // The same count of steps at different times: the axis check speaks.
        let mut early = full.clone();
        early.trace.steps.pop();
        let err = preflight(&[early.clone(), late.clone()], &waived).unwrap_err();
        assert!(err.contains("not on the same observation axis") && err.contains("ic_free"), "{err}");
        // Rows that assimilated the same set get no such sentence.
        let mut twin = derived_row("twin", &CASES_DEATHS, &["cases", "deaths"]).onto("cases").unwrap();
        twin.trace.steps.remove(0);
        let err = preflight(&[early, twin], &waived).unwrap_err();
        assert!(!err.contains("ic_free"), "{err}");
    }
}

