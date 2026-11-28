//! `camdl score PREDICTION --data STREAM=FILE ...`: a stored prediction scored
//! against the observed outcomes of the rows it predicted, stored inside the
//! prediction at `scores/score-<id>/`
//! (`docs/dev/proposals/2026-11-23-prediction-and-scores.md`, "`camdl
//! score`"; the identity is [`crate::fit_store::ScoreIdentity`]).
//!
//! The command takes the prediction's handle and the outcome files, and
//! nothing else: everything about the prediction is in its identity, from
//! which [`crate::fit::predict::repose`] poses it again. Before anything is
//! computed, each outcome file is read by the loader the prediction read its
//! data with, and every row the prediction conditioned on must be present in
//! it and equal by row digest; the first row that differs is refused. The
//! rows after them are the outcomes the score's identity holds.
//!
//! The score then computes the prediction again with the outcomes bound
//! ([`crate::fit::predict::recompute`]). Binding an outcome changes no
//! weight and draws no random number, so the prediction computed must be the
//! stored one: its run id equal, and every file it writes equal byte for byte.
//! A difference is refused as an internal error before any score is written.
//! The scored rows are then reduced by the rules of [`crate::fit::scoring`]
//! into the files of one of two layouts, which share their first columns:
//! for a free-forward or one-step prediction, which predicts the data's own
//! rows, `scored_rows.tsv`, `summary_rows.tsv`, `joint_rows.tsv` and
//! `joint_summary_rows.tsv`; for a forecast, which predicts targets after its
//! origin, `scored_targets.tsv`, `summary_targets.tsv`, `joint_targets.tsv`
//! and `joint_summary_targets.tsv`. Either is written with `score.json`
//! ([`ScoreFile`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::fit::observed_rows::ObservedRow;
use crate::fit::predict::RowRole;
use crate::fit::scoring::{CellKind, DataRowMode, JointKind, OutcomeBinding, RowScore, ScoreMode, ScoreRules, Scored};
use crate::result_files::fit::ScoreFile;
use crate::result_files::ResultFile;
use crate::quantile::{fmt_time, fmt_value};

/// The central interval levels a score reports the coverage of.
const INTERVALS: (f64, f64) = (0.5, 0.9);

/// The `schema` tag of `score.json`.
pub(crate) const SCORE_SCHEMA: &str = "camdl.score/v2";

/// `camdl score PREDICTION`.
pub(crate) fn cmd_score(inv: crate::args::ScoreInvocation) {
    let out = crate::out::Out::new(false);
    let (declared, execution, exports, request, given) = store::Inputs::split(inv).into_parts();
    let (args,) = declared.into_parts();
    let (prediction, data, held_out) = args.into_parts();
    let (exports,) = exports.into_parts();
    let (output, manifest, link_root) = exports.into_parts();
    let (request,) = request.into_parts();
    let (force, label, root_flag) = request.into_parts();
    let link_label = label.clone();
    let request = crate::camdl_store::store_request(label, force, root_flag).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    if let Some(dir) = &output {
        if let Err(e) = crate::util::refuse_missing_export_dir("-o", dir) {
            out.fail(format!("error: {e}"));
        }
    }
    let link_root = crate::link_tree::link_root(link_root.as_deref());
    let bindings = crate::bindings::claim(&out, manifest.as_deref(), link_root, link_label.as_deref());
    // A prediction named by its folder is in the store holding the folder.
    let root = crate::fit_store::store_holding(Path::new(&prediction)).unwrap_or_else(|| crate::camdl_store::root_of(&request));
    crate::fit_store::set_reader_root(root.clone());
    let store = crate::camdl_store::open_existing(&root)
        .unwrap_or_else(|| out.fail(format!("error: {prediction} names a stored prediction, and there is no run store at {}", root.display())));
    let info = store::RunInfo::new(&given, &execution, &request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    match score(&out, &store, &prediction, &data, &held_out, &info, &request) {
        Ok(stored) => {
            let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            if let Some(dir) = &output {
                export(&out, &stored, dir);
            }
            crate::bindings::write(&out, &bindings, &store, &invocation);
        }
        Err(e) => out.fail(format!("error: {e}")),
    }
}

/// The score of the prediction `handle` names against the outcome files
/// `data` (`STREAM=FILE`), stored in `store`: reused when stored, computed
/// otherwise.
#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
fn score(
    out: &crate::out::Out,
    store: &store::Store,
    handle: &str,
    data: &[String],
    held_out: &[String],
    info: &store::RunInfo,
    request: &store::StoreRequest,
) -> Result<store::Stored<crate::fit_store::Score>, String> {
    let parsed: store::Handle = handle.parse().map_err(|e: store::Error| e.to_string())?;
    let prediction: store::Stored<crate::fit_store::Prediction> =
        store.resolve(&store::RunRef::new(parsed)).map_err(|e| format!("{handle}: {e}"))?;
    let outcomes = Outcomes::Files { specs: data, held_out, skip_unbound: false, shown: &BTreeMap::new() };
    score_prediction(out, store, &prediction, outcomes, info, request)
}

/// Where a score's outcomes come from.
#[derive(Clone, Copy)]
pub(crate) enum Outcomes<'a> {
    /// Outcome files, `STREAM=FILE`: `specs` whole files, beginning with the
    /// rows the prediction conditioned on (`--data`), and `held_out` files of
    /// the rows after them alone (`--data-held-out`). With `skip_unbound`, a
    /// file of a stream the prediction does not bind is left out rather than
    /// refused, for a set of files given once for several predictions.
    /// `shown` maps a file's path to the path the outcome table's labels
    /// record for it, where that is not the path itself.
    Files { specs: &'a [String], held_out: &'a [String], skip_unbound: bool, shown: &'a BTreeMap<String, String> },
    /// The rows the prediction conditioned on and no other: a score of the
    /// conditioning rows alone, which reads no file.
    Conditioned,
}

/// The score of `prediction`, stored in `store`, against `outcomes`: reused
/// when stored, computed otherwise. `info` records the invocation and
/// `request` is the store request a fixed point's inputs are stored under.
#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
pub(crate) fn score_prediction(
    out: &crate::out::Out,
    store: &store::Store,
    prediction: &store::Stored<crate::fit_store::Prediction>,
    outcomes: Outcomes,
    info: &store::RunInfo,
    request: &store::StoreRequest,
) -> Result<store::Stored<crate::fit_store::Score>, String> {
    use crate::fit_store::{Score, ScoreIdentity, ScoreSettings};
    let no_labels = BTreeMap::new();
    let (whole, held_out, skip_unbound, shown): (&[String], &[String], bool, &BTreeMap<String, String>) = match outcomes {
        Outcomes::Files { specs, held_out, skip_unbound, shown } => (specs, held_out, skip_unbound, shown),
        Outcomes::Conditioned => (&[], &[], false, &no_labels),
    };
    // A held-out file is read as the whole file it completes: the stored rows
    // the prediction conditioned on, then its rows. The two spellings of one
    // outcome so give one file, and one score.
    let merged_dir = tempfile::Builder::new()
        .prefix("camdl-score-held-out-")
        .tempdir()
        .map_err(|e| format!("cannot make a scratch directory: {e}"))?;
    let mut completed: Vec<String> = whole.to_vec();
    // The path each outcome table's labels record: the file the user named,
    // never the scratch file a held-out file is completed into.
    let mut recorded: BTreeMap<String, String> = shown.clone();
    if !held_out.is_empty() {
        let posed = crate::fit::predict::repose(out, store, prediction, info, request, &BTreeMap::new(), &|_, _, _| Ok(()))?;
        let (model_ir, _) = posed.model_and_problem();
        for (spec, given) in complete_held_out(
            model_ir,
            posed.config_dt,
            &posed.conditioned,
            whole,
            held_out,
            skip_unbound,
            merged_dir.path(),
        )? {
            let path = spec.split_once('=').map(|(_, p)| p.to_string()).unwrap_or_default();
            recorded.insert(path, shown.get(&given).cloned().unwrap_or(given));
            completed.push(spec);
        }
    }
    let given: &[String] = &completed;
    let mut files = BTreeMap::new();
    for spec in given {
        let (name, path) = spec.split_once('=').ok_or_else(|| format!("--data {spec:?}: expected STREAM=FILE"))?;
        if files.insert(name.to_string(), path.to_string()).is_some() {
            return Err(format!("--data names stream `{name}` twice"));
        }
    }
    // The outcome files are checked against the stored conditioned lines
    // before the prediction is posed again: a fixed point binds them there.
    let check = |model: &Path, dt: f64, conditioned: &BTreeMap<String, String>| {
        let data = bound_specs(model, conditioned, given, skip_unbound)?;
        bind_outcomes(model, dt, conditioned, &data, None, &recorded).map(|_| ())
    };
    let reposed = crate::fit::predict::repose(out, store, prediction, info, request, &files, &check)?;
    let model_ir = reposed.model_and_problem().0;
    let data: Vec<String> = match outcomes {
        Outcomes::Files { .. } => bound_specs(model_ir, &reposed.conditioned, given, skip_unbound)?,
        Outcomes::Conditioned => reposed.conditioned.iter().map(|(source, path)| format!("{source}={path}")).collect(),
    };
    let (binding, outcomes) =
        bind_outcomes(model_ir, reposed.config_dt, &reposed.conditioned, &data, Some((store, info)), &recorded)?;
    let settings: serde_json::Value = serde_json::from_str(&prediction.identity().options.settings).map_err(|e| e.to_string())?;
    let pit_seed = settings["seed"].as_u64().ok_or("internal error: the prediction's options hold no seed")?;
    let finite = |x: f64| store::Finite::new(x).map_err(|e| e.to_string());
    let identity = ScoreIdentity {
        prediction: prediction.dep_all(),
        outcomes,
        settings: ScoreSettings { intervals: (finite(INTERVALS.0)?, finite(INTERVALS.1)?), pit_seed },
    };
    let files = if is_forecast(&settings) { ScoreFile::TARGETS } else { ScoreFile::DATA_ROWS };
    let run = match crate::camdl_store::begin::<Score>(store, &identity, &(), info, request.reuse(), files).map_err(|e| e.to_string())? {
        store::Begin::Reused(stored) => {
            out.say(format_args!("score: reused the stored score {}", stored.folder().display()));
            return Ok(stored);
        }
        store::Begin::Busy(owner) => return Err(store::Error::InProgress(owner).to_string()),
        store::Begin::Claimed(run) => run,
    };
    let partition = training_partition(prediction)?;
    let mut recomputed = crate::fit::predict::recompute(out, &reposed, info, &binding)?;
    check_recomputed(prediction, &recomputed)?;
    // A stream the prediction failed to compute has no row to score; its
    // report names it, and so does the score.
    for (name, bytes) in &recomputed.files {
        if !name.contains('/') && name.starts_with("report") && name.ends_with(".json") {
            let (failures, _) = crate::fit::failures::read_report(bytes)?;
            for f in failures {
                recomputed.scored.notes.push(format!("the prediction did not compute every object, so this is not scored: {}", f.describe()));
            }
        }
    }
    let (model, _) = crate::util::load_ir(&reposed.model_and_problem().0.to_string_lossy())?;
    let calendar = ir::caltime::CalendarAxis::from_model(&model).map_err(|e| e.to_string())?;
    let rules = ScoreRules { intervals: INTERVALS, pit_seed };
    let last_scored_end = recomputed.scored.cells.iter().map(|c| c.stop).fold(f64::NEG_INFINITY, f64::max);
    let labels = Labels::of(prediction, &settings, partition, last_scored_end, recomputed.scored.plug_in);
    let rendered = render(&recomputed.scored, &rules, &labels, &calendar, &recomputed.forcings_held)?;
    for (file, text) in rendered.files {
        let name = file.stored();
        run.result_file(&name).and_then(|h| h.write_all(text.as_bytes())).map_err(|e| format!("{name}: {e}"))?;
    }
    let stored = run.finish().map_err(|e| e.to_string())?;
    out.say(format_args!("score: {} row(s) scored, stored at {}", rendered.n_rows, stored.folder().display()));
    for note in &recomputed.scored.notes {
        out.say(format_args!("score: {note}"));
    }
    for w in &recomputed.forcings_held {
        if let score::trace::PrequentialWarning::ForcingHeld { forcing, held_value_at, .. } = w {
            out.say(format_args!(
                "score: forcing `{forcing}` was held outside its range, at its value at t = {held_value_at} \
                 (allow_forcing_extrapolation)"
            ));
        }
    }
    Ok(stored)
}

/// Refuse a recomputed prediction that is not the stored one: a run id or
/// a file that differs means the score would describe another prediction.
fn check_recomputed(
    stored: &store::Stored<crate::fit_store::Prediction>,
    recomputed: &crate::fit::predict::Recomputed,
) -> Result<(), String> {
    if recomputed.id != stored.id().key() {
        let record = serde_json::to_value(stored.identity()).map_err(|e| e.to_string())?;
        return Err(format!(
            "internal error: the prediction computed again from its stored identity has run id {}, not the \
             stored prediction's {} (they first differ at `{}`); no score was written. Please report this.",
            recomputed.id.short12(),
            stored.id().key().short12(),
            first_difference(&record, &recomputed.record, String::new()).unwrap_or_default()
        ));
    }
    let mut files = BTreeMap::new();
    for name in stored.result_files() {
        files.insert(name.to_string(), crate::camdl_store::read_stored(stored, name)?);
    }
    same_files(&files, &recomputed.files)
}

/// Refuse recomputed files `recomputed` that are not the stored files
/// `stored` byte for byte, naming the first that differs, is missing or is
/// extra.
fn same_files(stored: &BTreeMap<String, Vec<u8>>, recomputed: &BTreeMap<String, Vec<u8>>) -> Result<(), String> {
    for (name, bytes) in stored {
        match recomputed.get(name) {
            Some(b) if b == bytes => {}
            Some(_) => {
                return Err(format!(
                    "internal error: the prediction computed again with its outcomes bound writes `{name}` \
                     differently from the stored prediction; binding an outcome must change no prediction. \
                     No score was written. Please report this."
                ))
            }
            None => return Err(format!("internal error: the prediction computed again writes no `{name}`")),
        }
    }
    if let Some(extra) = recomputed.keys().find(|f| !stored.contains_key(*f)) {
        return Err(format!("internal error: the prediction computed again writes `{extra}`, which the stored one does not hold"));
    }
    Ok(())
}

/// The path of the first value at which two JSON documents differ.
fn first_difference(a: &serde_json::Value, b: &serde_json::Value, at: String) -> Option<String> {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            // A dependency's build is recorded and not hashed.
            let keys: std::collections::BTreeSet<&String> = x.keys().chain(y.keys()).filter(|k| *k != "build").collect();
            keys.into_iter().find_map(|k| match (x.get(k), y.get(k)) {
                (Some(p), Some(q)) => first_difference(p, q, format!("{at}.{k}")),
                _ => Some(format!("{at}.{k}")),
            })
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            x.iter().zip(y).enumerate().find_map(|(i, (p, q))| first_difference(p, q, format!("{at}[{i}]")))
        }
        _ => (a != b).then(|| format!("{at}: {a} vs {b}")),
    }
}

/// Each held-out file `held_out` (`STREAM=FILE`, `--data-held-out`) of a
/// source the prediction binds, written into `dir` as the whole file it
/// completes and returned as a `STREAM=FILE` of that file: the stored lines the
/// prediction conditioned on (`conditioned`, by source), then the held-out
/// file's lines after its header, beside the path of the held-out file it
/// completes. Refused: a stream named by both `--data`
/// (`whole`) and `--data-held-out`; a header that differs from the conditioned
/// file's; and a held-out row of a stream that ends at or before that
/// stream's last conditioning row, naming the first.
#[expect(clippy::disallowed_methods, reason = "reads each `--data-held-out` file, declared `#[input(identity)]` on `ScoreArgs` and on compare's arguments, and the copy of the stored conditioned lines `repose` wrote, to complete it into the whole outcome file whose lines after the conditioning rows the score's identity holds")]
fn complete_held_out(
    model_ir: &Path,
    config_dt: f64,
    conditioned: &BTreeMap<String, String>,
    whole: &[String],
    held_out: &[String],
    skip_unbound: bool,
    dir: &Path,
) -> Result<Vec<(String, String)>, String> {
    let (model, _) = crate::util::load_ir(&model_ir.to_string_lossy())?;
    let dt = crate::dt_rule::resolved_dt(model.simulation.dt, config_dt);
    let opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let source_of = |name: &str| -> Option<String> {
        if conditioned.contains_key(name) {
            return Some(name.to_string());
        }
        model.observations.iter().find(|o| o.name == name).map(|o| o.source.clone()).filter(|s| conditioned.contains_key(s))
    };
    let whole_sources: Vec<String> =
        whole.iter().filter_map(|spec| spec.split_once('=').and_then(|(n, _)| source_of(n))).collect();
    let mut out = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for spec in held_out {
        let (name, path) =
            spec.split_once('=').ok_or_else(|| format!("--data-held-out {spec:?}: expected STREAM=FILE"))?;
        let Some(source) = source_of(name) else {
            if skip_unbound {
                continue;
            }
            return Err(format!(
                "--data-held-out {name}=…: the prediction binds no stream source `{name}`; it binds {}",
                conditioned.keys().map(|s| format!("`{s}`")).collect::<Vec<_>>().join(", ")
            ));
        };
        if whole_sources.contains(&source) {
            return Err(format!(
                "--data and --data-held-out both give the outcomes of stream `{source}`: --data is a whole file, \
                 beginning with the rows the prediction conditioned on, and --data-held-out the rows after them \
                 alone; give one"
            ));
        }
        if seen.contains(&source) {
            return Err(format!("--data-held-out names stream `{source}` twice"));
        }
        seen.push(source.clone());
        let held = std::fs::read(path).map_err(|e| format!("--data-held-out {spec}: cannot read {path}: {e}"))?;
        let base = std::fs::read(&conditioned[&source]).map_err(|e| format!("the conditioned rows of `{source}`: {e}"))?;
        let header = |b: &[u8]| -> Vec<u8> {
            b.split(|c| *c == b'\n').next().unwrap_or_default().strip_suffix(b"\r").map(<[u8]>::to_vec).unwrap_or_else(|| b.split(|c| *c == b'\n').next().unwrap_or_default().to_vec())
        };
        if header(&held) != header(&base) {
            return Err(format!(
                "--data-held-out {spec}: its header is `{}`, and the file the prediction conditioned on has `{}`; \
                 a held-out file has the columns of the stream's data file",
                String::from_utf8_lossy(&header(&held)),
                String::from_utf8_lossy(&header(&base))
            ));
        }
        // Every held-out row ends after the stream's last conditioning row.
        for o in model.observations.iter().filter(|o| o.source == source) {
            let siblings: Vec<&ir::observation::ObservationModel> =
                model.observations.iter().filter(|x| x.source == o.source).collect();
            let before = crate::fit::runner::load_observations(&conditioned[&source], o, &siblings, &opts)?;
            let after = crate::fit::runner::load_observations(path, o, &siblings, &opts)
                .map_err(|e| format!("--data-held-out {spec}: {e}"))?;
            let Some(last) = before.last().map(|r| r.measures.end()) else { continue };
            if let Some((i, r)) =
                after.iter().enumerate().find(|(_, r)| r.measures.end() <= last + crate::fit::runner::TIME_TIE_EPS)
            {
                return Err(format!(
                    "--data-held-out {spec}: row {} of '{}' (time {}) ends at t = {}, at or before the last row the \
                     prediction conditioned on (t = {}); a held-out file holds only the rows after them",
                    i + 1,
                    o.name,
                    fmt_time(r.label),
                    fmt_time(r.measures.end()),
                    fmt_time(last)
                ));
            }
        }
        let mut bytes = base.clone();
        if !bytes.ends_with(b"\n") {
            bytes.push(b'\n');
        }
        let body = held.splitn(2, |c| *c == b'\n').nth(1).unwrap_or_default();
        bytes.extend_from_slice(body);
        let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("tsv");
        let completed = dir.join(format!("{source}.{ext}"));
        std::fs::write(&completed, bytes).map_err(|e| format!("cannot write {}: {e}", completed.display()))?;
        out.push((format!("{source}={}", completed.display()), path.to_string()));
    }
    Ok(out)
}

/// The outcome files `given` (`STREAM=FILE`) the prediction binds: with
/// `skip_unbound`, the files of the streams the model at `model_ir` does not
/// read from a source of `conditioned` are left out; without it every file is
/// kept, and [`bind_outcomes`] refuses one it cannot bind.
fn bound_specs(
    model_ir: &Path,
    conditioned: &BTreeMap<String, String>,
    given: &[String],
    skip_unbound: bool,
) -> Result<Vec<String>, String> {
    if !skip_unbound {
        return Ok(given.to_vec());
    }
    let (model, _) = crate::util::load_ir(&model_ir.to_string_lossy())?;
    Ok(given
        .iter()
        .filter(|spec| {
            let name = spec.split_once('=').map_or(spec.as_str(), |(n, _)| n);
            conditioned.contains_key(name)
                || model.observations.iter().any(|o| o.name == name && conditioned.contains_key(&o.source))
        })
        .cloned()
        .collect())
}

/// Read each outcome file for the stream source it names, and check that it
/// holds every row the prediction conditioned on (`conditioned`, the stored
/// conditioned lines by source, read under the model at `model_ir` and the
/// written step `config_dt`). With a store, also store the lines after them
/// as the score's outcome tables, each labelled with the path `recorded`
/// gives for its file, or the file's own.
#[expect(clippy::disallowed_methods, reason = "reads each `--data` outcome file, declared `#[input(identity)]` on `ScoreArgs`, whose lines after the conditioning rows are stored as the `DataTable` the score's identity holds")]
#[allow(clippy::type_complexity, clippy::result_large_err)]
fn bind_outcomes(
    model_ir: &Path,
    config_dt: f64,
    conditioned: &BTreeMap<String, String>,
    data: &[String],
    store: Option<(&store::Store, &store::RunInfo)>,
    recorded: &BTreeMap<String, String>,
) -> Result<(OutcomeBinding, BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>), String> {
    let (model, _) = crate::util::load_ir(&model_ir.to_string_lossy())?;
    let dt = crate::dt_rule::resolved_dt(model.simulation.dt, config_dt);
    let opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let load = |path: &str, leaf: &str| -> Result<Vec<ObservedRow>, String> {
        let obs = model.observations.iter().find(|o| o.name == leaf).ok_or_else(|| format!("internal error: no stream `{leaf}`"))?;
        let siblings: Vec<&ir::observation::ObservationModel> = model.observations.iter().filter(|o| o.source == obs.source).collect();
        crate::fit::runner::load_observations(path, obs, &siblings, &opts)
    };
    let sources: Vec<&String> = conditioned.keys().collect();
    let mut binding = OutcomeBinding::default();
    let mut tables = BTreeMap::new();
    for spec in data {
        let (name, path) = spec.split_once('=').ok_or_else(|| format!("--data {spec:?}: expected STREAM=FILE"))?;
        // A source, or one of the observation blocks that read it.
        let leaf_source = model.observations.iter().find(|o| o.name == name).map(|o| o.source.as_str());
        let source = match conditioned.get_key_value(name).or_else(|| leaf_source.and_then(|s| conditioned.get_key_value(s))) {
            Some((s, _)) => s.clone(),
            None => {
                return Err(format!(
                    "--data {name}=…: the prediction binds no stream source `{name}`; it binds {}",
                    sources.iter().map(|s| format!("`{s}`")).collect::<Vec<_>>().join(", ")
                ))
            }
        };
        if binding.files.contains_key(&source) {
            return Err(format!("--data names stream `{source}` twice"));
        }
        if !Path::new(path).is_file() {
            return Err(format!("--data {spec}: cannot read {path}: it does not exist"));
        }
        let conditioned_path = &conditioned[&source];
        let leaves: Vec<&str> = model.observations.iter().filter(|o| o.source == source).map(|o| o.name.as_str()).collect();
        let mut counts = BTreeMap::new();
        for leaf in &leaves {
            let conditioned = load(conditioned_path, leaf)?;
            let all = load(path, leaf).map_err(|e| format!("--data {spec}: {e}"))?;
            check_agreement(leaf, &conditioned, &all, path)?;
            counts.insert(leaf.to_string(), conditioned.len());
            binding.conditioning.insert(leaf.to_string(), conditioned.len());
            binding.rows.insert(leaf.to_string(), all);
        }
        binding.files.insert(source.clone(), path.to_string());
        let Some((store, info)) = store else { continue };
        let bytes = std::fs::read(path).map_err(|e| format!("--data {spec}: cannot read {path}: {e}"))?;
        let ext = Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("tsv").to_string();
        let after = crate::fit_store::lines_after_conditioning(&bytes, &ext, &source, &leaves, &counts, &load)?;
        let table = store::Hashed::new(<crate::camdl_store::Table as store::Content>::from_bytes(&after).map_err(|e| e.to_string())?);
        let shown = recorded.get(path).map_or(path, String::as_str);
        let attributes = crate::camdl_store::TableAttributes { stream: source.clone(), path: shown.to_string() };
        let dep = store
            .put::<crate::camdl_store::DataTable>(table, &attributes, &store::Provenance::none(), info)
            .and_then(|s| s.dep(<crate::camdl_store::DataTable as store::ContentArtifact>::FILE))
            .map_err(|e| format!("storing the outcomes of `{source}`: {e}"))?;
        tables.insert(source.clone(), dep);
    }
    Ok((binding, tables))
}

/// Refuse an outcome file of `leaf` whose rows do not begin with the rows the
/// prediction conditioned on, naming the first row that differs.
fn check_agreement(leaf: &str, conditioned: &[ObservedRow], all: &[ObservedRow], path: &str) -> Result<(), String> {
    use crate::fit::training_partition::row_digest;
    for (i, row) in conditioned.iter().enumerate() {
        let Some(given) = all.get(i) else {
            return Err(format!(
                "the outcome file {path} holds {} row(s) of '{leaf}', and the prediction conditioned on {}: \
                 row {} (time {}) is missing. An outcome file holds every row the prediction conditioned \
                 on, unchanged, and the rows after them.",
                all.len(),
                conditioned.len(),
                i + 1,
                fmt_time(row.label)
            ));
        };
        if row_digest(std::slice::from_ref(row))? != row_digest(std::slice::from_ref(given))? {
            return Err(format!(
                "the outcome file {path} differs from the rows the prediction conditioned on: row {} of '{leaf}' \
                 (time {}) is {} in it and {} in the prediction's data. Its held-out rows would belong to another \
                 series than the one the prediction conditioned on; score with the file the prediction was given, \
                 extended by the rows after it.",
                i + 1,
                fmt_time(row.label),
                describe_cell(given),
                describe_cell(row)
            ));
        }
    }
    Ok(())
}

fn describe_cell(row: &ObservedRow) -> String {
    match row.cell {
        None => "NA".to_string(),
        Some(sim::inference::ObsCell::Scalar(v) | sim::inference::ObsCell::HeldOut(v)) => {
            format!("{} at {}", fmt_value(v), fmt_time(row.label))
        }
    }
}

/// `-o DIR`: a copy of each of the score's result files.
fn export(out: &crate::out::Out, stored: &store::Stored<crate::fit_store::Score>, dir: &Path) {
    let fail = |e: String| -> ! { out.fail(format!("error: -o {}: {e}", dir.display())) };
    for name in stored.result_files() {
        let bytes = crate::camdl_store::read_stored(stored, name).unwrap_or_else(|e| fail(e));
        let path: PathBuf = dir.join(name);
        std::fs::write(&path, bytes).unwrap_or_else(|e| fail(format!("cannot write {}: {e}", path.display())));
    }
    out.say(format_args!("score: exported the score's files to {}", dir.display()));
}

/// What every row of a score is labelled with, read off the prediction.
struct Labels {
    prediction: String,
    /// `posterior`, `winner` (an optimizer run's winning point, with
    /// verified conditioning) or `fixed_point` (a point given by a file).
    parameters: &'static str,
    conditioning: score::trace::Conditioning,
    /// The statements of the table forcings a forecast read after its origin.
    covariates: BTreeMap<String, String>,
    settings: serde_json::Value,
}

impl Labels {
    fn of(
        prediction: &store::Stored<crate::fit_store::Prediction>,
        settings: &serde_json::Value,
        partition: Option<TrainingPartition>,
        last_scored_end: f64,
        plug_in: bool,
    ) -> Labels {
        use crate::fit_store::ParameterSourceRecord;
        let id = prediction.identity();
        let (parameters, conditioning) = match &id.source {
            // A fit that held nothing out still scores held out the outcome
            // rows past its data: the parameters saw none of them.
            // An optimizer run's winner is one point, `winner`; its training
            // rows are the run's, so its conditioning is verified as a
            // sampler's is.
            ParameterSourceRecord::Posterior { draws, .. } => (
                if plug_in { "winner" } else { "posterior" },
                match partition {
                    Some(TrainingPartition { held_out: held, train_end })
                        if held || last_scored_end > train_end + crate::fit::runner::TIME_TIE_EPS =>
                    {
                        score::trace::Conditioning::HoldOutTail { train_end, theta_source: draws.id().key().to_hex() }
                    }
                    _ => score::trace::Conditioning::InSample,
                },
            ),
            ParameterSourceRecord::FixedPoint { trained_through, .. } => (
                "fixed_point",
                match trained_through {
                    Some(t) => score::trace::Conditioning::HoldOutDeclared { trained_through: t.get() },
                    None => score::trace::Conditioning::Unknown,
                },
            ),
        };
        let covariates = settings["forecast"]["covariates"]
            .as_object()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or_default().to_string())).collect())
            .unwrap_or_default();
        Labels { prediction: prediction.id().key().to_hex(), parameters, conditioning, covariates, settings: settings.clone() }
    }

    /// The label of every score of a forecast read under a `realised`
    /// statement.
    fn realised(&self) -> bool {
        self.covariates.values().any(|s| s == "realised")
    }
}

/// The training partition a prediction's manifest records: whether the fit
/// held rows out, and the end of its last training row.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TrainingPartition {
    held_out: bool,
    train_end: f64,
}

/// The training partition the stored prediction's manifests record, read
/// from its first `predictive*.json` or `forecast*.json` in name order, or
/// `None` when it stores neither: a prediction that emitted no predictive
/// or forecast table writes no such manifest, and has no row to score.
///
/// Every such manifest must be one this build writes. One that is not JSON,
/// carries another `schema` tag, or records no `training_rows` with a rule
/// and an extent is refused, naming the file: a posterior prediction's rows
/// are labelled held out or in sample from those rows, and without them
/// every row would be labelled in sample.
fn training_partition(prediction: &store::Stored<crate::fit_store::Prediction>) -> Result<Option<TrainingPartition>, String> {
    use crate::result_files::fit::{PredictionFile, PredictionPart};
    let mut found = None;
    for name in prediction.result_files() {
        let expected = match PredictionFile::parse(name) {
            Some(PredictionFile::Manifest { part: PredictionPart::Predictive, .. }) => crate::fit::predict::PREDICTIVE_SCHEMA,
            Some(PredictionFile::Manifest { part: PredictionPart::Forecast, .. }) => crate::fit::forecast_output::FORECAST_SCHEMA,
            _ => continue,
        };
        let bytes = crate::camdl_store::read_stored(prediction, name)?;
        let partition = partition_of(&bytes, expected).map_err(|why| {
            format!(
                "the stored prediction {}'s `{name}` {why}, so the rows its parameters were trained on are not known; \
                 make the prediction again with `camdl predict` and score that",
                prediction.id().key().short12()
            )
        })?;
        found.get_or_insert(partition);
    }
    Ok(found)
}

/// The training partition one manifest's `bytes` record, refused with the
/// reason when they are not a manifest tagged `expected` that records its
/// `training_rows`.
fn partition_of(bytes: &[u8], expected: &str) -> Result<TrainingPartition, String> {
    let doc: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("cannot be read as JSON ({e})"))?;
    match doc["schema"].as_str() {
        Some(tag) if tag == expected => {}
        Some(tag) => return Err(format!("is tagged `{tag}`, where this build reads `{expected}`")),
        None => return Err(format!("carries no `schema` tag, where this build reads `{expected}`")),
    }
    let rows = doc.get("training_rows").ok_or("records no `training_rows`")?;
    let kind = rows["rule"]["kind"].as_str().ok_or("records `training_rows` with no `rule.kind`")?;
    let train_end = rows["extent"]["t"].as_f64().ok_or("records `training_rows` with no `extent.t`")?;
    Ok(TrainingPartition { held_out: kind != "all", train_end })
}

/// A score's files, rendered, each with its contents.
struct Rendered {
    files: Vec<(ScoreFile, String)>,
    n_rows: usize,
}

/// The columns every scored row has.
struct CoreLine<'a> {
    scenario: &'a str,
    sweep: &'a [(String, f64)],
    /// The observation block the row is of.
    stream: String,
    /// The data source the block reads.
    source: String,
    stratum: &'a [(String, String)],
    start: f64,
    stop: f64,
    role: RowRole,
    observed: Option<f64>,
    log_score: f64,
    /// The sample rules, when some draw's state reached the row alive.
    rules: Option<RowScore>,
    /// No draw reached the row: it is counted in its summary's `n_unscored`
    /// and written to no other file.
    unscored: bool,
}

/// One stored score's rows: a score holds one kind.
enum ScoredLines<'a> {
    DataRows(Vec<DataRowLine<'a>>),
    Targets(Vec<TargetLine<'a>>),
}

/// A scored row of the data, predicted free-forward or one step ahead.
struct DataRowLine<'a> {
    core: CoreLine<'a>,
    mode: DataRowMode,
}

/// A scored forecast target.
struct TargetLine<'a> {
    core: CoreLine<'a>,
    origin: f64,
    periods_ahead: i64,
    lead: f64,
}

/// One joint row: the log score of several streams' rows or targets
/// together.
struct JointLine<'a> {
    scenario: &'a str,
    sweep: &'a [(String, f64)],
    start: f64,
    stop: f64,
    role: RowRole,
    /// The streams scored together, in the order the prediction lists them.
    streams: &'a [String],
    kind: JointKind,
    log_score: f64,
    unscored: bool,
}

/// What a score's rows are predictions of, read off the prediction's mode:
/// forecast targets for a forecast, data rows otherwise.
fn is_forecast(settings: &serde_json::Value) -> bool {
    settings["mode"].as_str() == Some("forecast")
}

#[derive(Default)]
struct Sums {
    n: usize,
    n_unscored: usize,
    /// Rows among `n` with the sample rules.
    n_rules: usize,
    log_score: f64,
    crps: f64,
    wis: f64,
    inside: (usize, usize),
}

impl Sums {
    fn add(&mut self, log_score: f64, rules: Option<&RowScore>, unscored: bool) {
        if unscored {
            self.n_unscored += 1;
            return;
        }
        self.n += 1;
        self.log_score += log_score;
        if let Some(r) = rules {
            self.n_rules += 1;
            self.crps += r.crps;
            self.wis += r.wis.wis;
            self.inside.0 += usize::from(r.inside.0);
            self.inside.1 += usize::from(r.inside.1);
        }
    }

    /// `n`, `n_unscored`, `log_score_sum` and `log_score_mean`.
    fn log_score_cells(&self) -> [String; 4] {
        let some = |x: f64| if self.n > 0 { num(x) } else { String::new() };
        [self.n.to_string(), self.n_unscored.to_string(), some(self.log_score), some(self.log_score / self.n as f64)]
    }

    /// `crps_mean`, `wis_mean`, `coverage_50` and `coverage_90`.
    fn rule_cells(&self) -> [String; 4] {
        if self.n_rules == 0 {
            return Default::default();
        }
        let k = self.n_rules as f64;
        [num(self.crps / k), num(self.wis / k), num(self.inside.0 as f64 / k), num(self.inside.1 as f64 / k)]
    }
}

/// Summary sums by key, in the order each key first appears.
#[derive(Default)]
struct Grouped {
    order: Vec<Vec<String>>,
    sums: BTreeMap<Vec<String>, Sums>,
}

impl Grouped {
    fn entry(&mut self, key: Vec<String>) -> &mut Sums {
        if !self.sums.contains_key(&key) {
            self.order.push(key.clone());
        }
        self.sums.entry(key).or_default()
    }
}

/// A number as a score file writes it: empty when it is NaN or +∞.
fn num(x: f64) -> String {
    if x.is_finite() || x == f64::NEG_INFINITY { format!("{x}") } else { String::new() }
}

/// What the rows of one score share in how they are written: the sweep
/// parameters and stratum dimensions any row has, and the calendar.
struct Columns<'a> {
    sweep_names: Vec<String>,
    dims: Vec<String>,
    calendar: &'a ir::caltime::CalendarAxis,
}

impl Columns<'_> {
    fn dated(&self) -> bool {
        self.calendar.origin.is_some()
    }

    fn date(&self, t: f64) -> String {
        self.calendar.date_of(t).map(ir::caltime::RenderedInstant::into_string).unwrap_or_default()
    }

    /// `scenario` and the `sweep:<p>` columns.
    fn arm_header(&self) -> Vec<String> {
        std::iter::once("scenario".to_string()).chain(self.sweep_names.iter().map(|p| format!("sweep:{p}"))).collect()
    }

    fn arm_cells(&self, scenario: &str, sweep: &[(String, f64)]) -> Vec<String> {
        std::iter::once(scenario.to_string())
            .chain(
                self.sweep_names
                    .iter()
                    .map(|p| sweep.iter().find(|(q, _)| q == p).map(|(_, v)| fmt_value(*v)).unwrap_or_default()),
            )
            .collect()
    }

    /// The arm, then `stream`, `source` and the dimensions: the part of a
    /// row's summary key every kind shares.
    fn stream_header(&self) -> Vec<String> {
        let mut h = self.arm_header();
        h.extend(["stream", "source"].map(String::from));
        h.extend(self.dims.iter().cloned());
        h
    }

    fn stream_cells(&self, l: &CoreLine) -> Vec<String> {
        let mut row = self.arm_cells(l.scenario, l.sweep);
        row.extend([l.stream.clone(), l.source.clone()]);
        row.extend(
            self.dims.iter().map(|d| l.stratum.iter().find(|(e, _)| e == d).map(|(_, v)| v.clone()).unwrap_or_default()),
        );
        row
    }

    /// `t_start`, `t_stop` and, in a dated model, `date_start` and
    /// `date_stop`.
    fn period_header(&self) -> Vec<String> {
        let mut h: Vec<String> = ["t_start", "t_stop"].map(String::from).to_vec();
        if self.dated() {
            h.extend(["date_start", "date_stop"].map(String::from));
        }
        h
    }

    fn period_cells(&self, start: f64, stop: f64) -> Vec<String> {
        let mut row = vec![fmt_time(start), fmt_time(stop)];
        if self.dated() {
            row.extend([self.date(start), self.date(stop)]);
        }
        row
    }

    /// The shared core of `scored_rows.tsv` and `scored_targets.tsv`.
    fn core_header(&self) -> Vec<String> {
        let mut h = self.stream_header();
        h.extend(self.period_header());
        h.extend(
            [
                "role", "parameters", "quantity", "mean", "q05", "q25", "q50", "q75", "q95", "observed", "log_score", "crps", "wis",
                "wis_dispersion", "wis_over", "wis_under", "pit", "in_50", "in_90",
            ]
            .map(String::from),
        );
        h
    }

    fn core_cells(&self, l: &CoreLine, parameters: &str) -> Vec<String> {
        let mut row = self.stream_cells(l);
        row.extend(self.period_cells(l.start, l.stop));
        row.push(l.role.as_str().into());
        row.push(parameters.into());
        match &l.rules {
            Some(s) => {
                row.push(crate::quantile::Quantity::Observation.as_str().into());
                row.push(fmt_value(s.mean));
                row.extend(s.quantiles.iter().map(|q| fmt_value(*q)));
                row.push(fmt_value(l.observed.unwrap_or(f64::NAN)));
                row.push(num(l.log_score));
                row.push(num(s.crps));
                row.push(num(s.wis.wis));
                row.push(num(s.wis.dispersion));
                row.push(num(s.wis.overprediction));
                row.push(num(s.wis.underprediction));
                row.push(num(s.pit));
                row.push(u8::from(s.inside.0).to_string());
                row.push(u8::from(s.inside.1).to_string());
            }
            None => {
                row.extend(vec![String::new(); 7]);
                row.push(l.observed.map(fmt_value).unwrap_or_default());
                row.push(num(l.log_score));
                row.extend(vec![String::new(); 8]);
            }
        }
        row
    }
}

fn tsv(header: &[String], rows: &[Vec<String>]) -> String {
    let mut out = header.join("\t") + "\n";
    for r in rows {
        out += &(r.join("\t") + "\n");
    }
    out
}

/// The rows of `scored` as the lines of a score of one kind, `forecast`
/// choosing which, and its joint lines. A row or joint of the other kind is
/// an internal error.
fn lines_of<'a>(scored: &'a Scored, rules: &ScoreRules, forecast: bool) -> Result<(ScoredLines<'a>, Vec<JointLine<'a>>), String> {
    use sim::inference::prequential::PIT_RNG_STREAM;
    let mut pit_rng = sim::rng::StatefulRng::new_stream(rules.pit_seed, PIT_RNG_STREAM);
    let mut lines = if forecast { ScoredLines::Targets(Vec::new()) } else { ScoredLines::DataRows(Vec::new()) };
    for c in &scored.cells {
        // One uniform per row, scored or not, so a row's PIT draw does not
        // depend on whether another row was reached.
        let v = pit_rng.uniform();
        let (log_score, s) = match (c.log_liks.is_empty(), c.samples.is_empty()) {
            (true, _) => (f64::NAN, None),
            (false, true) => (crate::fit::scoring::log_mean_exp(&c.log_liks)?, None),
            (false, false) => {
                let s = crate::fit::scoring::score_cell(c, rules, v)?;
                (s.log_score, Some(s))
            }
        };
        let core = CoreLine {
            scenario: &c.scenario,
            sweep: &c.sweep,
            stream: c.leaf.clone(),
            source: c.source.clone(),
            stratum: &c.stratum,
            start: c.start,
            stop: c.stop,
            role: c.role,
            observed: Some(c.observed),
            log_score,
            rules: s,
            unscored: c.log_liks.is_empty(),
        };
        match (&mut lines, c.kind) {
            (ScoredLines::DataRows(v), CellKind::DataRow { mode }) => v.push(DataRowLine { core, mode }),
            (ScoredLines::Targets(v), CellKind::Target { origin, periods_ahead, lead }) => {
                v.push(TargetLine { core, origin, periods_ahead, lead })
            }
            (_, kind) => {
                return Err(format!(
                    "internal error: a {} row of '{}' in the score of a {} prediction",
                    kind.mode().as_str(),
                    c.leaf,
                    if forecast { "forecast" } else { "free-forward or one-step" }
                ))
            }
        }
    }
    let mut joints = Vec::with_capacity(scored.joints.len());
    for j in &scored.joints {
        if matches!(j.kind, JointKind::Target { .. }) != forecast {
            return Err(format!("internal error: a {} joint row in the score of another kind", j.kind.mode().as_str()));
        }
        let log_score = match j.log_liks.is_empty() {
            true => f64::NAN,
            false => crate::fit::scoring::log_mean_exp(&j.log_liks)
                .map_err(|e| format!("the joint log score at t = {}: {e}", fmt_time(j.stop)))?,
        };
        joints.push(JointLine {
            scenario: &j.scenario,
            sweep: &j.sweep,
            start: j.start,
            stop: j.stop,
            role: j.role,
            streams: &j.streams,
            kind: j.kind,
            log_score,
            unscored: j.log_liks.is_empty(),
        });
    }
    Ok((lines, joints))
}

fn render(
    scored: &Scored,
    rules: &ScoreRules,
    labels: &Labels,
    calendar: &ir::caltime::CalendarAxis,
    forcings_held: &[score::trace::PrequentialWarning],
) -> Result<Rendered, String> {
    let forecast = is_forecast(&labels.settings);
    let (lines, joints) = lines_of(scored, rules, forecast)?;
    let cores: Vec<&CoreLine> = match &lines {
        ScoredLines::DataRows(v) => v.iter().map(|l| &l.core).collect(),
        ScoredLines::Targets(v) => v.iter().map(|l| &l.core).collect(),
    };
    // Every dimension any row is stratified by, and every swept parameter, in
    // order of first appearance.
    let mut dims: Vec<String> = Vec::new();
    let mut sweep_names: Vec<String> = Vec::new();
    for l in &cores {
        for (d, _) in l.stratum {
            if !dims.contains(d) {
                dims.push(d.clone());
            }
        }
    }
    for sweep in cores.iter().map(|l| l.sweep).chain(joints.iter().map(|j| j.sweep)) {
        for (p, _) in sweep {
            if !sweep_names.contains(p) {
                sweep_names.push(p.clone());
            }
        }
    }
    let cols = Columns { sweep_names, dims, calendar };
    let score_header = ["n", "n_unscored", "log_score_sum", "log_score_mean", "crps_mean", "wis_mean", "coverage_50", "coverage_90"];
    let joint_score_header = ["n", "n_unscored", "log_score_sum", "log_score_mean"];
    let mut scored_rows: Vec<Vec<String>> = Vec::new();
    let mut groups = Grouped::default();
    let mut joint_rows: Vec<Vec<String>> = Vec::new();
    let mut joint_groups = Grouped::default();
    let (files, summary_key): ([ScoreFile; 4], [&str; 2]) = match &lines {
        ScoredLines::DataRows(v) => {
            for l in v {
                let mut key = cols.stream_cells(&l.core);
                key.extend([l.mode.as_str().to_string(), l.core.role.as_str().to_string()]);
                groups.entry(key).add(l.core.log_score, l.core.rules.as_ref(), l.core.unscored);
                if l.core.unscored {
                    continue;
                }
                let mut row = cols.core_cells(&l.core, labels.parameters);
                row.push(l.mode.as_str().into());
                scored_rows.push(row);
            }
            for j in &joints {
                let mut key = cols.arm_cells(j.scenario, j.sweep);
                key.push(j.role.as_str().into());
                joint_groups.entry(key).add(j.log_score, None, j.unscored);
                if j.unscored {
                    continue;
                }
                let mut row = cols.arm_cells(j.scenario, j.sweep);
                row.extend(cols.period_cells(j.start, j.stop));
                row.extend([j.role.as_str().to_string(), j.streams.join(","), num(j.log_score)]);
                joint_rows.push(row);
            }
            (
                [ScoreFile::ScoredRows, ScoreFile::SummaryRows, ScoreFile::JointRows, ScoreFile::JointSummaryRows],
                ["mode", "role"],
            )
        }
        ScoredLines::Targets(v) => {
            for l in v {
                let mut key = cols.stream_cells(&l.core);
                key.extend([l.core.role.as_str().to_string(), fmt_time(l.lead)]);
                groups.entry(key).add(l.core.log_score, l.core.rules.as_ref(), l.core.unscored);
                if l.core.unscored {
                    continue;
                }
                let mut row = cols.core_cells(&l.core, labels.parameters);
                row.extend([fmt_time(l.origin), l.periods_ahead.to_string(), fmt_time(l.lead)]);
                scored_rows.push(row);
            }
            for j in &joints {
                let JointKind::Target { origin, lead } = j.kind else { unreachable!("checked by lines_of") };
                let lead = lead.map(fmt_time).unwrap_or_default();
                let mut key = cols.arm_cells(j.scenario, j.sweep);
                key.extend([j.role.as_str().to_string(), lead.clone()]);
                joint_groups.entry(key).add(j.log_score, None, j.unscored);
                if j.unscored {
                    continue;
                }
                let mut row = cols.arm_cells(j.scenario, j.sweep);
                row.extend(cols.period_cells(j.start, j.stop));
                row.extend([j.role.as_str().to_string(), j.streams.join(","), fmt_time(origin), lead, num(j.log_score)]);
                joint_rows.push(row);
            }
            (
                [ScoreFile::ScoredTargets, ScoreFile::SummaryTargets, ScoreFile::JointTargets, ScoreFile::JointSummaryTargets],
                ["role", "lead"],
            )
        }
    };
    let mut scored_header = cols.core_header();
    let mut joint_header = cols.arm_header();
    joint_header.extend(cols.period_header());
    joint_header.extend(["role", "streams"].map(String::from));
    match &lines {
        ScoredLines::DataRows(_) => {
            scored_header.push("mode".into());
        }
        ScoredLines::Targets(_) => {
            scored_header.extend(["origin", "periods_ahead", "lead"].map(String::from));
            joint_header.extend(["origin", "lead"].map(String::from));
        }
    }
    joint_header.push("log_score".into());
    let mut summary_header = cols.stream_header();
    summary_header.extend(summary_key.map(String::from));
    summary_header.extend(score_header.map(String::from));
    let summary_rows: Vec<Vec<String>> = groups
        .order
        .iter()
        .map(|key| {
            let s = &groups.sums[key];
            key.iter().cloned().chain(s.log_score_cells()).chain(s.rule_cells()).collect()
        })
        .collect();
    let mut joint_summary_header = cols.arm_header();
    joint_summary_header.push("role".into());
    if forecast {
        joint_summary_header.push("lead".into());
    }
    joint_summary_header.extend(joint_score_header.map(String::from));
    let joint_summary_rows: Vec<Vec<String>> = joint_groups
        .order
        .iter()
        .map(|key| key.iter().cloned().chain(joint_groups.sums[key].log_score_cells()).collect())
        .collect();

    let held_out = cores.iter().any(|l| l.role == RowRole::HeldOut);
    let manifest = manifest(scored, labels, rules, held_out, forcings_held)?;
    Ok(Rendered {
        files: vec![
            (files[0], tsv(&scored_header, &scored_rows)),
            (files[1], tsv(&summary_header, &summary_rows)),
            (files[2], tsv(&joint_header, &joint_rows)),
            (files[3], tsv(&joint_summary_header, &joint_summary_rows)),
            (ScoreFile::Score, manifest),
        ],
        n_rows: scored.cells.len(),
    })
}

/// `score.json`: the parameter source, each mode's basis, the conditioning,
/// the reach and particle count, the rules' settings, each stream's scored
/// rows and their digest, and the notes.
fn manifest(
    scored: &Scored,
    labels: &Labels,
    rules: &ScoreRules,
    held_out: bool,
    forcings_held: &[score::trace::PrequentialWarning],
) -> Result<String, String> {
    let s = &labels.settings;
    let particles = s["particles"].as_u64().map(|n| n as usize);
    let posterior = labels.parameters == "posterior";
    let mut modes: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    for c in &scored.cells {
        let mode = c.kind.mode();
        if modes.contains_key(mode.as_str()) {
            continue;
        }
        let per_draw = match mode {
            ScoreMode::FreeForward => 1,
            ScoreMode::OneStep | ScoreMode::Forecast => particles.unwrap_or(1).max(1),
        };
        let n_draws = c.samples.len() / per_draw;
        let provenance = if posterior { score::trace::Provenance::Posterior { n_draws } } else { score::trace::Provenance::PlugIn };
        let basis = match mode {
            ScoreMode::OneStep | ScoreMode::Forecast => score::trace::PredictiveBasis::Filtered,
            ScoreMode::FreeForward if scored.ode => {
                score::trace::PredictiveBasis::OdeFreeForward { samples_per_draw: 1 }
            }
            ScoreMode::FreeForward => score::trace::PredictiveBasis::SimulatedFreeForward,
        };
        modes.insert(mode.as_str(), serde_json::json!({
            "provenance": provenance,
            "basis": basis,
            "n_draws": n_draws,
            "samples_per_draw": per_draw,
        }));
    }
    // Per (stream, mode): the scored rows and a digest of what was scored,
    // each row's stream, period and observed value.
    let mut digests: BTreeMap<(String, &'static str), (usize, String)> = BTreeMap::new();
    for c in &scored.cells {
        let e = digests.entry((c.leaf.clone(), c.kind.mode().as_str())).or_insert((0, String::new()));
        e.0 += 1;
        e.1 += &format!("{}\t{:016x}\t{:016x}\t{:016x}\t{}\n", c.leaf, c.start.to_bits(), c.stop.to_bits(), c.observed.to_bits(), c.scenario);
    }
    let mut notes = scored.notes.clone();
    let single: Vec<&str> = {
        let mut v: Vec<&str> = scored.cells.iter().filter(|c| c.samples.len() == 1).map(|c| c.leaf.as_str()).collect();
        v.sort();
        v.dedup();
        v
    };
    if !single.is_empty() {
        notes.push(format!(
            "the rows of {} have one predictive sample each, so their CRPS, WIS, PIT and coverage come from a \
             single sample",
            single.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", ")
        ));
    }
    // Per mode, each set of streams a joint row's density is of.
    let mut joint_streams: BTreeMap<&'static str, std::collections::BTreeSet<Vec<String>>> = BTreeMap::new();
    for j in &scored.joints {
        let mut set = j.streams.clone();
        set.sort();
        set.dedup();
        joint_streams.entry(j.kind.mode().as_str()).or_default().insert(set);
    }
    let mut labels_out: Vec<&str> = Vec::new();
    if labels.realised() {
        labels_out.push("conditional on covariates observed after the origin");
    }
    if matches!(labels.conditioning, score::trace::Conditioning::HoldOutDeclared { .. }) && held_out {
        labels_out.push("held out as declared, not verified");
    }
    let mut doc = serde_json::json!({
        "schema": SCORE_SCHEMA,
        "prediction": labels.prediction,
        "parameters": labels.parameters,
        "conditioning": labels.conditioning,
        "modes": modes,
        "reach": s["reach"],
        "particles": particles,
        "seed": s["seed"],
        "rules": {
            "log_score": "logsumexp over every sample of the observation log density of the outcome, minus log(samples)",
            "crps": "fair sample estimator (Ferro 2014)",
            "wis_levels": score::interval::HUB_LEVELS,
            "intervals": [rules.intervals.0, rules.intervals.1],
            "pit": "randomized within the jump of a count",
            "pit_seed": rules.pit_seed,
        },
        "covariates": labels.covariates,
        "labels": labels_out,
        "scored": digests.iter().map(|((stream, mode), (n, text))| serde_json::json!({
            "stream": stream, "mode": mode, "rows": n, "digest": crate::hashing::sha256_hex(text.as_bytes()),
        })).collect::<Vec<_>>(),
        "joint_streams": joint_streams,
        "notes": notes,
        // Each forcing held outside its range while the prediction was
        // computed again (`allow_forcing_extrapolation`).
        "forcings_held": forcings_held,
    });
    crate::fit::predict::stamp_engine(&mut doc);
    serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    /// The recomputed prediction is refused unless it writes the stored
    /// files byte for byte: a file with other bytes, a file missing and a
    /// file more are each named.
    #[test]
    fn a_recomputed_prediction_must_write_the_stored_files() {
        let files = |pairs: &[(&str, &str)]| -> BTreeMap<String, Vec<u8>> {
            pairs.iter().map(|(n, b)| (n.to_string(), b.as_bytes().to_vec())).collect()
        };
        let stored = files(&[("forecast/cases.tsv", "a\n1\n"), ("forecast.json", "{}")]);
        assert!(super::same_files(&stored, &stored.clone()).is_ok());
        let other = files(&[("forecast/cases.tsv", "a\n2\n"), ("forecast.json", "{}")]);
        let e = super::same_files(&stored, &other).unwrap_err();
        assert!(e.contains("writes `forecast/cases.tsv` differently"), "{e}");
        let missing = files(&[("forecast.json", "{}")]);
        assert!(super::same_files(&stored, &missing).unwrap_err().contains("writes no `forecast/cases.tsv`"));
        let extra = files(&[("forecast/cases.tsv", "a\n1\n"), ("forecast.json", "{}"), ("more.tsv", "")]);
        assert!(super::same_files(&stored, &extra).unwrap_err().contains("writes `more.tsv`"));
    }

    /// A manifest's training partition is read only from a manifest this
    /// build writes: one that is not JSON, carries another tag, or records no
    /// `training_rows` is refused with the reason.
    #[test]
    fn a_training_partition_is_read_only_from_a_current_manifest() {
        use super::{partition_of, TrainingPartition};
        let tag = crate::fit::forecast_output::FORECAST_SCHEMA;
        let held = serde_json::json!({
            "schema": tag,
            "training_rows": { "rule": { "kind": "holdout_after" }, "extent": { "t": 24.0 } },
        });
        let bytes = |v: &serde_json::Value| serde_json::to_vec(v).unwrap();
        assert_eq!(partition_of(&bytes(&held), tag), Ok(TrainingPartition { held_out: true, train_end: 24.0 }));
        let mut all = held.clone();
        all["training_rows"]["rule"]["kind"] = "all".into();
        assert_eq!(partition_of(&bytes(&all), tag), Ok(TrainingPartition { held_out: false, train_end: 24.0 }));

        assert!(partition_of(b"{ not json", tag).unwrap_err().contains("cannot be read as JSON"));
        let mut old = held.clone();
        old["schema"] = "camdl.forecast/v5".into();
        assert!(partition_of(&bytes(&old), tag).unwrap_err().contains("is tagged `camdl.forecast/v5`"));
        let mut untagged = held.clone();
        untagged.as_object_mut().unwrap().remove("schema");
        assert!(partition_of(&bytes(&untagged), tag).unwrap_err().contains("no `schema` tag"));
        let mut renamed = held.clone();
        let rows = renamed.as_object_mut().unwrap().remove("training_rows").unwrap();
        renamed["parameter_set"] = rows;
        assert_eq!(partition_of(&bytes(&renamed), tag).unwrap_err(), "records no `training_rows`");
        let mut no_extent = held;
        no_extent["training_rows"].as_object_mut().unwrap().remove("extent");
        assert!(partition_of(&bytes(&no_extent), tag).unwrap_err().contains("no `extent.t`"));
    }
}
