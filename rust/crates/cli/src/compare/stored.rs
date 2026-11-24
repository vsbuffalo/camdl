//! The rows `camdl compare` reads from stored scores
//! (`docs/dev/proposals/2026-11-23-prediction-and-scores.md`, "`compare`
//! reads stored scores").
//!
//! A compared input is a stored score, a stored prediction or a stored fit
//! run. Each input is first resolved ([`prepare`]) and the inputs checked
//! together before anything is computed ([`preflight`]): every stream an
//! outcome file names is bound by some input, and every stored prediction
//! read in a filtered mode ran its filters at compare's particle count. Then
//! a fit run is predicted in the mode compare asks for, with compare's draws
//! (a posterior's only; an optimizer run is predicted at its winner),
//! particles and seed, by the computation `camdl predict` runs
//! ([`crate::fit::predict::prediction_of_run`]), and a prediction is scored
//! by the computation `camdl score` runs
//! ([`crate::score_command::score_prediction`]). Each is reused when it is
//! stored and stored when it is computed, so a second compare of the same
//! inputs computes nothing. Nothing here runs a model or a child process.
//!
//! A score's outcomes are, in order of preference, the `--data` and
//! `--data-held-out` files; the rows the prediction conditioned on, when only
//! those are compared (a fit that held nothing out, or `--in-sample`); or the
//! one stored score of the prediction that holds rows past them. A stored fit
//! keeps its held-out rows' dates and not their values, so its held-out rows
//! are scored only from outcome files or a stored score.
//!
//! Replicates: the prediction of an input is posed again at the seeds after
//! its own (a fit run's at compare's seed, a stored prediction's or score's at
//! the prediction's), each stored, and each scored against the outcome rows of
//! the first score. An ODE free-forward score does not depend on the seed and
//! is predicted once.
//!
//! The rows of one mode are read back from the score's `scored.tsv` and
//! grouped by role (and, for a forecast, by lead), and each group becomes one
//! [`PrequentialTrace`]: a step per row end, its log score the score's
//! `joint` row there (the one stream's own when one stream is observed
//! there), and each stream's own scores beside it. Nothing is recomputed:
//! every number a trace holds is one the score stored, except that
//! replicates combine the log scores of several stored scores.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use score::trace::{
    Conditioning, Coverage, Derivation, PredInterval, PredictiveBasis, PrequentialStep, PrequentialTrace,
    Provenance, StreamScore, ThetaSource, TraceOverride,
};

use crate::args::types::PredictModeArg;
use crate::chain_selection::ChainSelection;
use crate::fit::stream_bindings::StreamBindings;
use crate::fit_store::{OpenedFit, ParameterSourceRecord, Prediction, Score};

use super::{DataIdentity, Loaded, RecordedData, RowStreams, ScoreFacts};

/// The label `score.json` gives a forecast read under a `realised` covariate
/// statement.
const REALISED_LABEL: &str = "conditional on covariates observed after the origin";

/// The predictions compare makes of each input when `--replicates` is not
/// given: enough for a filter-noise SE, and so the within-filter-noise check,
/// by default.
pub(super) const DEFAULT_REPLICATES: usize = 5;

/// The particle count of the filters of every prediction compare makes when
/// `--particles` is not given.
pub(crate) const DEFAULT_DERIVE_PARTICLES: usize = 1000;

/// What compare asks of every stored input: the options of the predictions
/// it makes and the rows it compares.
pub(super) struct Request<'a> {
    /// `--mode`; `None` is the default of each input.
    pub mode: Option<PredictModeArg>,
    /// `--horizon`, a forecast's reach.
    pub horizon: Option<&'a str>,
    /// `--draws`, for a fit run with a posterior.
    pub draws: Option<usize>,
    /// `--particles`; `None` is [`DEFAULT_DERIVE_PARTICLES`].
    pub particles: Option<usize>,
    pub seed: Option<u64>,
    /// `--replicates`; `None` is [`DEFAULT_REPLICATES`].
    pub replicates: Option<usize>,
    pub in_sample: bool,
    /// `--data STREAM=FILE`, for every prediction that binds the stream.
    pub data: &'a [String],
    /// `--data-held-out STREAM=FILE`, likewise.
    pub data_held_out: &'a [String],
    pub allow_forcing_extrapolation: bool,
    /// `--allow-mismatched-filter-settings`.
    pub allow_mismatched_filter_settings: bool,
    /// The compile settings of `--no-licm` and `--no-ir-cache`.
    pub compile: crate::util::CompileSettings,
}

impl Request<'_> {
    /// The particle count of every filter compare runs.
    fn particles(&self) -> usize {
        self.particles.unwrap_or(DEFAULT_DERIVE_PARTICLES)
    }

    fn outcome_files(&self) -> bool {
        !self.data.is_empty() || !self.data_held_out.is_empty()
    }
}

/// What a compared handle names.
enum Input {
    Score(store::Stored<Score>),
    Prediction(store::Stored<Prediction>),
    Run(Arc<OpenedFit>),
}

/// A compared input resolved, before anything is predicted or scored.
pub(super) struct Prepared {
    text: String,
    store: store::Store,
    root: PathBuf,
    input: Input,
    /// The mode its rows are read in.
    mode: PredictModeArg,
    /// The names an outcome file's `STREAM` may give for it: each stream
    /// source its prediction conditions on, and each observation block that
    /// reads one.
    binds: Vec<String>,
    /// For a stored prediction or score read in a filtered mode, the particle
    /// count its prediction's filters ran at.
    stored_particles: Option<usize>,
    /// Whether a fit run is an optimizer's, predicted at its winner.
    optimizer: bool,
}

/// Resolve the input `text` names, reading nothing it does not need to check
/// the inputs together.
pub(super) fn prepare(text: &str, req: &Request) -> Result<Prepared, String> {
    let root = crate::fit_store::store_holding(Path::new(text)).unwrap_or_else(crate::fit_store::reader_root);
    let store = crate::camdl_store::open_existing(&root)
        .ok_or_else(|| format!("'{text}' names a stored result, and there is no run store at {}", root.display()))?;
    let input = resolve(&store, text)?;
    let (mode, binds, stored_particles, optimizer) = match &input {
        Input::Run(fit) => (mode_of_run(fit, req)?, binds_of_fit(fit)?, None, is_optimizer(fit)),
        Input::Score(s) => {
            let p = prediction_of(&store, s, text)?;
            let mode = mode_of_prediction(&p, req)?;
            (mode, binds_of_prediction(&store, &p)?, particles_of(&p, mode)?, false)
        }
        Input::Prediction(p) => {
            let mode = mode_of_prediction(p, req)?;
            (mode, binds_of_prediction(&store, p)?, particles_of(p, mode)?, false)
        }
    };
    Ok(Prepared { text: text.to_string(), store, root, input, mode, binds, stored_particles, optimizer })
}

/// Refuse inputs that cannot be compared, before any is predicted or scored:
/// an outcome file whose stream no input binds, and a stored prediction read
/// in a filtered mode whose filters ran at a particle count other than
/// compare's, `--particles` or, when compare predicts a fit run, its default;
/// with neither, stored predictions that ran at different counts (unless
/// `--allow-mismatched-filter-settings`). `inputs` pairs
/// each input with its row name.
pub(super) fn preflight(inputs: &[(&str, &Prepared)], req: &Request) -> Result<(), String> {
    for (flag, specs) in [("--data", req.data), ("--data-held-out", req.data_held_out)] {
        for spec in specs {
            let name = spec.split_once('=').map_or(spec.as_str(), |(n, _)| n);
            if !inputs.iter().any(|(_, p)| p.binds.iter().any(|b| b == name)) {
                let mut bound: Vec<&str> = inputs.iter().flat_map(|(_, p)| p.binds.iter().map(String::as_str)).collect();
                bound.sort();
                bound.dedup();
                return Err(format!(
                    "{flag} {name}=…: no compared prediction binds a stream or stream source `{name}`, so the file \
                     would give no outcome; they bind {}",
                    bound.iter().map(|b| format!("`{b}`")).collect::<Vec<_>>().join(", ")
                ));
            }
        }
    }
    if req.allow_mismatched_filter_settings {
        return Ok(());
    }
    let filtered: Vec<(&str, &Prepared)> =
        inputs.iter().copied().filter(|(_, p)| p.mode != PredictModeArg::FreeForward).collect();
    // The count every filter must run at: `--particles`, or compare's default
    // when it predicts a fit run; with neither, the stored inputs' own count,
    // which they must share.
    let predicts = filtered.iter().any(|(_, p)| matches!(p.input, Input::Run(_)));
    let resolved = req.particles.or(predicts.then(|| req.particles()));
    let mut stored: Vec<usize> = filtered.iter().filter_map(|(_, p)| p.stored_particles).collect();
    stored.sort_unstable();
    stored.dedup();
    let agree = match resolved {
        Some(n) => stored.iter().all(|m| *m == n),
        None => stored.len() <= 1,
    };
    if agree {
        return Ok(());
    }
    let each: Vec<String> = filtered
        .iter()
        .map(|(name, p)| match (&p.input, p.stored_particles) {
            (Input::Run(_), _) => format!("'{name}' {} (a fit run, which compare predicts)", req.particles()),
            (_, Some(m)) => format!("'{name}' {m} (stored)"),
            (_, None) => format!("'{name}' (stored, with no filter)"),
        })
        .collect();
    let hint = match stored.as_slice() {
        [m] => format!("pass --particles {m}, so every prediction compare makes runs its filters at the stored count"),
        _ => "predict the stored inputs again at one count (`camdl predict … --particles N`) and pass the same \
              --particles N here"
            .to_string(),
    };
    Err(format!(
        "the compared predictions ran their particle filters at different particle counts: {}. The log of a \
         particle-filter likelihood estimate is biased downward, and by more at fewer particles, so part of \
         Δelpd would measure the filters rather than the models.\n       {hint}, or pass \
         --allow-mismatched-filter-settings to compare them anyway.",
        each.join(", ")
    ))
}

/// The groups of rows the input `prepared` contributes to the comparison, in
/// lead order: one for a one-step or free-forward score, one per lead for a
/// forecast. `selection` is the input's chain selection.
pub(super) fn load(
    out: &crate::out::Out,
    prepared: Prepared,
    req: &Request,
    selection: Option<&ChainSelection>,
) -> Result<Vec<Loaded>, String> {
    let Prepared { text, store, root, input, mode, optimizer, .. } = prepared;
    let text = text.as_str();
    let request = crate::camdl_store::store_request(None, false, Some(root.clone()));
    let info = store::RunInfo::new(&store::Given::__from_fields(serde_json::Map::new()), &(), &request)
        .map_err(|e| e.to_string())?;
    let wanted = req.replicates.unwrap_or(DEFAULT_REPLICATES).max(1);
    // The first prediction and its score; then, for every later replicate,
    // the prediction posed again at the next seed.
    let (first, again): (store::Stored<Score>, Box<dyn Fn(usize) -> Result<store::Stored<Prediction>, String> + '_>) =
        match input {
            Input::Run(fit) => {
                let own = fit_settings(&fit)?;
                if let Some(note) =
                    own.turned_on_note(own.with_flags(false, req.allow_forcing_extrapolation), text, "compare scores it")
                {
                    out.say(format_args!("{note}"));
                }
                let args = predict_args(text, mode, req, 0, selection, optimizer);
                let p = crate::fit::predict::prediction_of_run(out, &args, Arc::clone(&fit), req.compile, &info)?;
                let first = score_of(out, &store, &p, mode, req, &info, &request, text)?;
                let info = &info;
                let again = move |r: usize| {
                    let args = predict_args(text, mode, req, r, selection, optimizer);
                    crate::fit::predict::prediction_of_run(out, &args, Arc::clone(&fit), req.compile, info)
                };
                (first, Box::new(again))
            }
            Input::Prediction(p) => {
                let first = score_of(out, &store, &p, mode, req, &info, &request, text)?;
                (first, reposer(out, &store, p, &info, &request, req)?)
            }
            Input::Score(s) => {
                let p = prediction_of(&store, &s, text)?;
                (s, reposer(out, &store, p, &info, &request, req)?)
            }
        };
    let n = match deterministic(&first, mode)? {
        true => {
            if wanted >= 2 {
                out.say(format_args!(
                    "compare: '{text}': an ODE free-forward score does not depend on the seed, so its prediction is \
                     scored once, not {wanted} times"
                ));
            }
            1
        }
        false => wanted,
    };
    let mut scores = vec![first];
    for r in 1..n {
        let p = again(r)?;
        scores.push(score_against(out, &store, &p, &scores[0], &info, &request)?);
    }
    drop(again);
    drop(info);
    let replicated = scores.len() >= 2;
    let prediction = prediction_of(&store, &scores[0], text)?;
    let context = Context::of(&store, &prediction)?;
    let mut replicate_groups: Vec<Vec<Group>> = Vec::new();
    for s in &scores {
        replicate_groups.push(groups_of(&store, s, req, &context, text)?);
    }
    let first = replicate_groups.remove(0);
    // A score's notes are the score's, the same for every group of it.
    for note in first.first().map(|g| g.notes.as_slice()).unwrap_or_default() {
        out.say(format_args!("compare: '{text}': {note}"));
    }
    let mut loaded = Vec::with_capacity(first.len());
    for (k, g) in first.into_iter().enumerate() {
        let mut traces = vec![g.trace.clone()];
        for gs in &replicate_groups {
            let o = gs.get(k).ok_or("internal error: the replicate scores of one prediction hold different leads")?;
            traces.push(o.trace.clone());
        }
        let totals = score::combine::ReplicateTotals::of(&traces);
        let mut trace = if replicated { combine_replicates(&traces)? } else { g.trace };
        if let Some(d) = trace.derivation.as_mut() {
            d.replicates = traces.len();
            d.replicate_elpd = totals.elpd;
            d.replicate_elpd_by_stream = totals.by_stream;
            d.mc_se_elpd = totals.mc_se;
        }
        loaded.push(Loaded {
            trace,
            data: context.data.clone(),
            streams: context.streams.clone(),
            facts: g.facts,
        });
    }
    if replicated {
        out.say(format_args!(
            "compare: '{text}': {} replicate scores; each row's log score is the log-mean-exp of theirs, and its \
             CRPS, WIS, PIT and coverage are the first replicate's",
            scores.len()
        ));
    }
    Ok(loaded)
}

/// The predictions of the stored prediction `p` at the seeds after its own:
/// replicate `r` is `p` posed again from its identity with its seed plus
/// `r`, every other option its own, stored as `camdl predict` stores it.
fn reposer<'a>(
    out: &'a crate::out::Out,
    store: &'a store::Store,
    p: store::Stored<Prediction>,
    info: &'a store::RunInfo,
    request: &'a store::StoreRequest,
    req: &'a Request,
) -> Result<Box<dyn Fn(usize) -> Result<store::Stored<Prediction>, String> + 'a>, String> {
    let seed = settings_of(&p)?["seed"].as_u64().ok_or("internal error: the prediction's options hold no seed")?;
    Ok(Box::new(move |r: usize| {
        let mut posed = crate::fit::predict::repose(out, store, &p, info, request, &BTreeMap::new(), &|_, _, _| Ok(()))?;
        posed.args.flags.seed = Some(seed + r as u64);
        crate::fit::predict::prediction_of(out, &posed.args, posed.source.clone(), req.compile, info)
    }))
}

/// Whether `s`'s rows of `mode` are an ODE free-forward score, which is the
/// same at every seed.
fn deterministic(s: &store::Stored<Score>, mode: PredictModeArg) -> Result<bool, String> {
    let manifest: serde_json::Value = serde_json::from_slice(&crate::camdl_store::read_stored(s, "score.json")?)
        .map_err(|e| format!("{}: score.json: {e}", s.folder().display()))?;
    let basis = &manifest["modes"][mode_name(mode)]["basis"];
    Ok(basis == "ode_free_forward" || basis.get("ode_free_forward").is_some())
}

/// The score of `p` against the outcome rows `first` scored: each of its
/// outcome tables given as a held-out file, labelled with the stored table's
/// path.
fn score_against(
    out: &crate::out::Out,
    store: &store::Store,
    p: &store::Stored<Prediction>,
    first: &store::Stored<Score>,
    info: &store::RunInfo,
    request: &store::StoreRequest,
) -> Result<store::Stored<Score>, String> {
    let scratch = tempfile::Builder::new()
        .prefix("camdl-compare-outcomes-")
        .tempdir()
        .map_err(|e| format!("cannot make a scratch directory: {e}"))?;
    let mut held_out = Vec::new();
    let mut shown = BTreeMap::new();
    for (source, dep) in &first.identity().outcomes {
        let t: store::Stored<crate::camdl_store::DataTable> = store
            .get(dep.id())
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("the outcome table {} the score names is not stored", dep.id().key().short12()))?;
        let file = <crate::camdl_store::DataTable as store::ContentArtifact>::FILE;
        let bytes = crate::camdl_store::read_stored(&t, file)?;
        let header = bytes.split(|c| *c == b'\n').next().unwrap_or_default();
        let ext = if header.contains(&b'\t') || !header.contains(&b',') { "tsv" } else { "csv" };
        let path = scratch.path().join(format!("{source}.{ext}"));
        std::fs::write(&path, &bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let path = path.to_string_lossy().into_owned();
        shown.insert(path.clone(), t.folder().join(file).to_string_lossy().into_owned());
        held_out.push(format!("{source}={path}"));
    }
    let outcomes =
        crate::score_command::Outcomes::Files { specs: &[], held_out: &held_out, skip_unbound: false, shown: &shown };
    crate::score_command::score_prediction(out, store, p, outcomes, info, request)
}

/// The prediction the score `s` scores.
fn prediction_of(store: &store::Store, s: &store::Stored<Score>, text: &str) -> Result<store::Stored<Prediction>, String> {
    store
        .get(s.identity().prediction.id())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("'{text}': the prediction its score scores is not stored"))
}

/// The stored result `text` names: a score, a prediction or a fit run. A
/// handle that names no score or prediction is read as a fit run, whose
/// reader names what the handle is when it is none (a fit file, a problem).
fn resolve(store: &store::Store, text: &str) -> Result<Input, String> {
    if let Ok(handle) = text.parse::<store::Handle>() {
        if let Ok(s) = store.resolve::<Score>(&store::RunRef::new(handle.clone())) {
            return Ok(Input::Score(s));
        }
        if let Ok(p) = store.resolve::<Prediction>(&store::RunRef::new(handle)) {
            return Ok(Input::Prediction(p));
        }
    }
    crate::fit_store::opened_run(text).map(Input::Run).map_err(|e| {
        format!("'{text}' is neither a prequential.json nor a stored score, prediction or fit run: {e}")
    })
}

/// Whether the fit run `fit` is an optimizer's (IF2, NLopt), whose
/// prediction is of its one winning point.
fn is_optimizer(fit: &OpenedFit) -> bool {
    fit.leaf
        .as_ref()
        .and_then(|leaf| crate::fit::fit_view::FitView::read(leaf))
        .and_then(|v| v.methods.first().map(|m| m.method.is_optimizer()))
        .unwrap_or(false)
}

/// The names an outcome file may give for a stream of a prediction whose
/// model is `model`, conditioned on the stream sources `sources`.
fn bound_names<'a>(model: &ir::Model, sources: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut names: Vec<String> = sources.cloned().collect();
    for o in &model.observations {
        if names.contains(&o.source) && !names.contains(&o.name) {
            names.push(o.name.clone());
        }
    }
    names
}

fn binds_of_fit(fit: &OpenedFit) -> Result<Vec<String>, String> {
    let (model, _) = crate::util::load_ir(&fit.model_ir.to_string_lossy())?;
    let spec = fit.config.problem.data_spec().map_err(|e| format!("the fit binds no data: {e}"))?;
    let bindings = StreamBindings::resolve(&model, spec)?;
    Ok(bound_names(&model, bindings.training_by_source().keys()))
}

fn binds_of_prediction(store: &store::Store, p: &store::Stored<Prediction>) -> Result<Vec<String>, String> {
    if let Some(fit) = fit_of(store, p)? {
        return binds_of_fit(&fit);
    }
    let id = p.identity();
    let ParameterSourceRecord::FixedPoint { model, .. } = &id.source else { unreachable!("not a posterior") };
    let envelope = crate::fit_store::stored_envelope(store, model, "model")?;
    Ok(bound_names(&envelope.model, id.data.keys()))
}

/// The particle count of the filters of the stored prediction `p`, when the
/// mode it is read in runs one.
fn particles_of(p: &store::Stored<Prediction>, mode: PredictModeArg) -> Result<Option<usize>, String> {
    if mode == PredictModeArg::FreeForward {
        return Ok(None);
    }
    Ok(settings_of(p)?["particles"].as_u64().map(|n| n as usize))
}

/// The mode a fit run is predicted in: `--mode`, or a forecast when
/// `--horizon` is given, or one-step on a backend that runs a particle
/// filter and free-forward on the others.
fn mode_of_run(fit: &OpenedFit, req: &Request) -> Result<PredictModeArg, String> {
    let mode = match (req.mode, req.horizon) {
        (Some(m), _) => m,
        (None, Some(_)) => PredictModeArg::Forecast,
        (None, None) => {
            let backend = match &fit.leaf {
                Some(leaf) => crate::fit::fit_view::FitView::read(leaf).and_then(|v| v.methods.first().map(|s| s.backend)),
                None => fit.config.inference.method.as_ref().map(|m| m.algorithm.backend()),
            };
            match backend {
                Some(crate::run_meta::InferenceBackend::ChainBinomial) | None => PredictModeArg::OneStep,
                Some(_) => PredictModeArg::FreeForward,
            }
        }
    };
    match (mode, req.horizon) {
        (PredictModeArg::Forecast, None) => Err(
            "--mode forecast needs --horizon DURATION, the reach of each forecast from the end of its fit's \
             training rows (e.g. --horizon \"4 weeks\")"
                .into(),
        ),
        (PredictModeArg::FreeForward | PredictModeArg::OneStep, Some(h)) => {
            Err(format!("--horizon {h:?} sets a forecast's reach, and --mode {} is not a forecast", mode_name(mode)))
        }
        _ => Ok(mode),
    }
}

/// The mode a stored prediction's score is read in: `--mode`, else the
/// prediction's own when it made one, else one-step when it holds one and
/// free-forward otherwise.
fn mode_of_prediction(p: &store::Stored<Prediction>, req: &Request) -> Result<PredictModeArg, String> {
    let settings = settings_of(p)?;
    let own = match settings["mode"].as_str() {
        Some("free_forward") => Some(PredictModeArg::FreeForward),
        Some("one_step") => Some(PredictModeArg::OneStep),
        Some("forecast") => Some(PredictModeArg::Forecast),
        _ => None,
    };
    Ok(match (req.mode, own) {
        (Some(m), _) => m,
        (None, Some(m)) => m,
        (None, None) => match settings["particles"].is_null() {
            true => PredictModeArg::FreeForward,
            false => PredictModeArg::OneStep,
        },
    })
}

fn mode_name(m: PredictModeArg) -> &'static str {
    match m {
        PredictModeArg::FreeForward => "free_forward",
        PredictModeArg::OneStep => "one_step",
        PredictModeArg::Forecast => "forecast",
    }
}

fn settings_of(p: &store::Stored<Prediction>) -> Result<serde_json::Value, String> {
    serde_json::from_str(&p.identity().options.settings).map_err(|e| format!("the prediction's options: {e}"))
}

/// The `camdl predict` compare runs for replicate `r` of the fit run `text`:
/// `--draws` for a posterior (an optimizer run is predicted at its winner,
/// once), and compare's particle count wherever a filter runs.
fn predict_args(
    text: &str,
    mode: PredictModeArg,
    req: &Request,
    r: usize,
    selection: Option<&ChainSelection>,
    optimizer: bool,
) -> crate::args::PredictArgs {
    let mut flags = crate::args::PredictFlags {
        mode: Some(mode),
        horizon: (mode == PredictModeArg::Forecast).then(|| req.horizon.map(str::to_string)).flatten(),
        n_draws: if optimizer { None } else { req.draws },
        // Replicate 0 is the prediction `camdl predict` makes at `--seed`.
        seed: match r {
            0 => req.seed,
            _ => Some(req.seed.unwrap_or(1) + r as u64),
        },
        particles: (mode != PredictModeArg::FreeForward).then(|| req.particles()),
        ..Default::default()
    };
    flags.forcing_extrapolation.allow_forcing_extrapolation = req.allow_forcing_extrapolation;
    let (exclude, keep) = match selection {
        Some(s) if s.polarity() == crate::chain_selection::Polarity::Drop => (Some(s.ids_csv()), None),
        Some(s) => (None, Some(s.ids_csv())),
        None => (None, None),
    };
    crate::args::PredictArgs {
        source: Some(text.to_string()),
        flags,
        posterior: crate::args::PosteriorFlags { by_chain: false, exclude_chains: exclude, chains: keep, realised_design: false },
        fixed_point: crate::args::FixedPointFlags::default(),
        _removed_fit: None,
    }
}

/// The score of `p` whose rows compare reads: against the `--data` files;
/// else, when only the conditioning rows are compared, against them; else
/// the one stored score of `p` that holds rows past them.
#[allow(clippy::too_many_arguments)]
fn score_of(
    out: &crate::out::Out,
    store: &store::Store,
    p: &store::Stored<Prediction>,
    mode: PredictModeArg,
    req: &Request,
    info: &store::RunInfo,
    request: &store::StoreRequest,
    text: &str,
) -> Result<store::Stored<Score>, String> {
    use crate::score_command::{score_prediction, Outcomes};
    if req.outcome_files() {
        let outcomes =
            Outcomes::Files { specs: req.data, held_out: req.data_held_out, skip_unbound: true, shown: &BTreeMap::new() };
        return score_prediction(out, store, p, outcomes, info, request);
    }
    let held = held_out_rows(store, p)?;
    let past_conditioning = mode == PredictModeArg::Forecast || (!req.in_sample && !held.is_empty());
    if !past_conditioning {
        return score_prediction(out, store, p, Outcomes::Conditioned, info, request);
    }
    let found: Vec<store::Stored<Score>> = stored_scores(store, p)?
        .into_iter()
        .filter(|s| scored_rows(s).is_ok_and(|rows| rows.iter().any(|r| r["mode"] == mode_name(mode) && r["role"] != "trained")))
        .collect();
    match found.len() {
        1 => Ok(found.into_iter().next().expect("one")),
        0 => Err(format!(
            "'{text}': {} past the rows its prediction conditioned on ({}), and a stored fit keeps those rows' \
             dates, not their values, so compare needs their outcomes: pass --data STREAM=FILE with the files the \
             fit was given, or --data-held-out STREAM=FILE with their held-out rows alone, or score the prediction {} with `camdl score`{}",
            if mode == PredictModeArg::Forecast { "a forecast's targets lie" } else { "the held-out rows lie" },
            if held.is_empty() { "none held out".to_string() } else { held.join(", ") },
            p.folder().display(),
            if mode == PredictModeArg::Forecast { "" } else { ", or pass --in-sample to compare the training rows" },
        )),
        n => Err(format!(
            "'{text}': its prediction {} has {n} stored scores against different outcomes ({}); name the outcome \
             files with --data STREAM=FILE, or compare the score you mean by its handle",
            p.folder().display(),
            found.iter().map(|s| s.folder().display().to_string()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Each stream that held rows out of the training rows `p`'s parameters
/// were fitted to, as `stream: n`; for a fixed point declared fitted through
/// `--holdout-after`, the declaration.
fn held_out_rows(store: &store::Store, p: &store::Stored<Prediction>) -> Result<Vec<String>, String> {
    match &p.identity().source {
        ParameterSourceRecord::Posterior { .. } => {
            let fit = fit_of(store, p)?.expect("a posterior prediction is of a fit run");
            Ok(fit.fit.held_out_rows().into_iter().map(|(s, n)| format!("{s}: {n}")).collect())
        }
        ParameterSourceRecord::FixedPoint { trained_through, .. } => Ok(trained_through
            .iter()
            .map(|t| format!("every row after the declared --holdout-after {}", crate::quantile::fmt_time(t.get())))
            .collect()),
    }
}

/// The fit run a posterior prediction's draws came from; `None` for a fixed
/// point.
fn fit_of(store: &store::Store, p: &store::Stored<Prediction>) -> Result<Option<Arc<OpenedFit>>, String> {
    let ParameterSourceRecord::Posterior { draws, .. } = &p.identity().source else { return Ok(None) };
    let run: store::Stored<crate::fit_store::FitRun> = store
        .get(draws.id())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("the fit run {} the prediction's draws came from is not stored", draws.id().key().short12()))?;
    crate::fit_store::opened_run(&run.folder().to_string_lossy()).map(Some)
}

/// Every completed score stored inside the prediction `p`.
fn stored_scores(store: &store::Store, p: &store::Stored<Prediction>) -> Result<Vec<store::Stored<Score>>, String> {
    let dir = p.folder().join("scores");
    let Ok(entries) = std::fs::read_dir(&dir) else { return Ok(Vec::new()) };
    let mut folders: Vec<std::path::PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    folders.sort();
    let mut found = Vec::new();
    for f in folders {
        match store.resolve::<Score>(&store::RunRef::new(store::Handle::Path(f))) {
            Ok(s) => found.push(s),
            Err(store::Error::NotFound { .. }) => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(found)
}

type Line = BTreeMap<String, String>;

/// The rows of a score's `scored.tsv`, each by column name.
fn scored_rows(s: &store::Stored<Score>) -> Result<Vec<Line>, String> {
    let bytes = crate::camdl_store::read_stored(s, "scored.tsv")?;
    let text = String::from_utf8(bytes).map_err(|e| format!("{}: scored.tsv: {e}", s.folder().display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
    Ok(lines
        .filter(|l| !l.is_empty())
        .map(|l| header.iter().map(|h| h.to_string()).zip(l.split('\t').map(str::to_string)).collect())
        .collect())
}

/// What every row of a prediction's scores shares: the data it conditioned
/// on, and the streams its model declares and binds.
struct Context {
    data: DataIdentity,
    streams: RowStreams,
    /// The fit run's folder, for a posterior.
    fit_folder: Option<String>,
    /// The fit run's retained posterior draws, for a posterior.
    n_available: Option<usize>,
    /// Whether the fit run is an optimizer's, predicted at its winner.
    winner: bool,
    /// Whether the fit itself was made under `allow_forcing_extrapolation`;
    /// `None` for a fixed point.
    fit_allows_forcing: Option<bool>,
    /// The step the prediction ran at: the dt rule over the model's own step
    /// and the problem's (a fit's `[config] dt`, a fixed point's `--dt`).
    dt: f64,
}

impl Context {
    fn of(store: &store::Store, p: &store::Stored<Prediction>) -> Result<Context, String> {
        let mut recorded = RecordedData::default();
        let fit_allows_forcing: Option<bool>;
        let config_dt: f64;
        let (model, data, fit_folder, n_available) = match fit_of(store, p)? {
            Some(fit) => {
                fit_allows_forcing = Some(fit_settings(&fit)?.allow_forcing_extrapolation);
                config_dt = fit.config.problem.config.dt;
                for (source, input) in &fit.fit.training {
                    recorded.training.insert(source.clone(), crate::hashing::sha256_hex(&input.bytes));
                }
                let (model, _) = crate::util::load_ir(&fit.model_ir.to_string_lossy())?;
                let spec = fit.config.problem.data_spec().map_err(|e| format!("the fit binds no data: {e}"))?.clone();
                let n = match is_optimizer(&fit) {
                    true => None,
                    false => Some(posterior_size(&fit, &settings_of(p)?)?),
                };
                (model, spec, Some(crate::fit::report::display_path(fit.fit.problem.folder())), n)
            }
            None => return Self::of_fixed_point(store, p),
        };
        let bindings = StreamBindings::resolve(&model, &data)?;
        let declared = model.observations.iter().map(|o| (o.name.clone(), o.source.clone())).collect();
        Ok(Context {
            data: DataIdentity::Digests(recorded),
            streams: RowStreams::Derived { declared, bindings },
            winner: fit_folder.is_some() && n_available.is_none(),
            fit_allows_forcing,
            dt: crate::dt_rule::resolved_dt(model.simulation.dt, config_dt),
            fit_folder,
            n_available,
        })
    }

    /// A fixed point's: the data tables its identity holds, and the streams
    /// of its stored model, which bind every block of each table's source.
    fn of_fixed_point(store: &store::Store, p: &store::Stored<Prediction>) -> Result<Context, String> {
        let id = p.identity();
        let ParameterSourceRecord::FixedPoint { model, .. } = &id.source else { unreachable!("not a posterior") };
        let record = serde_json::to_value(&id.source).map_err(|e| e.to_string())?;
        let config_dt = record["FixedPoint"]["numerics"]["dt"].as_f64().unwrap_or(f64::NAN);
        let mut recorded = RecordedData::default();
        for (source, dep) in &id.data {
            let t: store::Stored<crate::camdl_store::DataTable> = store
                .get(dep.id())
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("the data table {} the prediction names is not stored", dep.id().key().short12()))?;
            let bytes = crate::camdl_store::read_stored(&t, <crate::camdl_store::DataTable as store::ContentArtifact>::FILE)?;
            recorded.training.insert(source.clone(), crate::hashing::sha256_hex(&bytes));
        }
        let envelope = crate::fit_store::stored_envelope(store, model, "model")?;
        let declared: Vec<(String, String)> =
            envelope.model.observations.iter().map(|o| (o.name.clone(), o.source.clone())).collect();
        // Every source the point was given data for, bound by name; the
        // bindings say which blocks read each source, not where its file is.
        let observations: serde_json::Map<String, serde_json::Value> =
            id.data.keys().map(|source| (source.clone(), serde_json::Value::String(format!("{source}.tsv")))).collect();
        let spec: crate::fit::config_v2::DataSpec = serde_json::from_value(serde_json::json!({ "observations": observations }))
            .map_err(|e| format!("internal error: a fixed point's data binding: {e}"))?;
        let bindings = StreamBindings::resolve(&envelope.model, &spec)?;
        Ok(Context {
            data: DataIdentity::Digests(recorded),
            streams: RowStreams::Derived { declared, bindings },
            fit_folder: None,
            n_available: None,
            winner: false,
            fit_allows_forcing: None,
            dt: crate::dt_rule::resolved_dt(envelope.model.simulation.dt, config_dt),
        })
    }
}

/// The settings a stored fit was made under (`[config]`
/// `allow_degenerate_rates`, `allow_forcing_extrapolation`).
fn fit_settings(fit: &OpenedFit) -> Result<crate::fit::fitted_model::FittedSettings, String> {
    let (model, _) = crate::util::load_ir(&fit.model_ir.to_string_lossy())?;
    Ok(crate::fit::fitted_model::stored_fit_simulator(fit, &model)?.settings)
}

/// The posterior draws a fit run's prediction drew from: its retained draws
/// after the prediction's chain selection.
fn posterior_size(fit: &OpenedFit, settings: &serde_json::Value) -> Result<usize, String> {
    let leaf = fit.leaf.as_ref().ok_or("a prediction is made from a run, and the handle names a problem")?;
    let pdraws =
        crate::posterior_draws::resolve_posterior_draws(&leaf.to_string_lossy(), None).map_err(|e| -> String { e.into() })?;
    let excluded: Vec<usize> =
        settings["excluded_chains"].as_array().map(|a| a.iter().filter_map(|c| c.as_u64().map(|c| c as usize)).collect()).unwrap_or_default();
    let selection = match excluded.is_empty() {
        true => None,
        false => Some(ChainSelection::parse(
            &excluded.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(","),
            crate::chain_selection::Polarity::Drop,
            "--exclude-chains",
        )?),
    };
    let (rows, _) = pdraws.with_selection(selection).load_params_with_info()?;
    Ok(rows.len())
}

/// One role (and, for a forecast, one lead) of one mode of a score, as a
/// trace and the facts the checks compare.
struct Group {
    trace: PrequentialTrace,
    facts: ScoreFacts,
    notes: Vec<String>,
}

/// The groups of `s` that compare reads under `req`, in lead order.
fn groups_of(
    store: &store::Store,
    s: &store::Stored<Score>,
    req: &Request,
    context: &Context,
    text: &str,
) -> Result<Vec<Group>, String> {
    let manifest: serde_json::Value = serde_json::from_slice(&crate::camdl_store::read_stored(s, "score.json")?)
        .map_err(|e| format!("{}: score.json: {e}", s.folder().display()))?;
    let all = scored_rows(s)?;
    let modes: Vec<&str> = {
        let mut m: Vec<&str> = all.iter().map(|r| r["mode"].as_str()).collect();
        m.sort();
        m.dedup();
        m
    };
    let mode = match req.mode {
        Some(m) => mode_name(m),
        None if req.horizon.is_some() => "forecast",
        None => ["one_step", "free_forward", "forecast"].into_iter().find(|m| modes.contains(m)).unwrap_or("one_step"),
    };
    let rows: Vec<&Line> = all.iter().filter(|r| r["mode"] == mode).collect();
    if rows.is_empty() {
        return Err(format!(
            "'{text}': its score {} holds no {mode} rows; it holds {}",
            s.folder().display(),
            if modes.is_empty() { "none".to_string() } else { modes.join(", ") }
        ));
    }
    if let Some(col) = all.first().and_then(|r| r.keys().find(|k| k.starts_with("sweep:")).cloned()) {
        return Err(format!("'{text}': its score is of a sweep (`{col}`), and compare compares one parameter point"));
    }
    // One scenario: the prediction's arm with no overlay when it made several.
    let mut scenarios: Vec<&str> = rows.iter().map(|r| r["scenario"].as_str()).collect();
    scenarios.sort();
    scenarios.dedup();
    let scenario = match scenarios.as_slice() {
        [one] => one.to_string(),
        many => match many.iter().find(|s| **s == crate::args::FITTED || **s == crate::sim_job::IMPLICIT_ARM_LABEL) {
            Some(s) => s.to_string(),
            None => return Err(format!("'{text}': its score holds the scenarios {}; compare compares one", many.join(", "))),
        },
    };
    let rows: Vec<&Line> = rows.into_iter().filter(|r| r["scenario"] == scenario).collect();
    let roles: Vec<&str> = {
        let mut v: Vec<&str> = rows.iter().map(|r| r["role"].as_str()).collect();
        v.sort();
        v.dedup();
        v
    };
    let role = if mode == "forecast" {
        if req.in_sample {
            return Err("--in-sample compares the training rows, and a forecast predicts none".into());
        }
        "future"
    } else if req.in_sample {
        "trained"
    } else if roles.contains(&"held_out") {
        "held_out"
    } else if roles.contains(&"unknown") {
        "unknown"
    } else if req.outcome_files() {
        return Err(format!(
            "'{text}': the outcome files give no {mode} row past the rows its prediction conditioned on, so its \
             score {} holds only those rows; give files that extend past them, or pass --in-sample to compare \
             the training rows",
            s.folder().display()
        ));
    } else {
        "trained"
    };
    let rows: Vec<&Line> = rows.into_iter().filter(|r| r["role"] == role).collect();
    if rows.iter().all(|r| r["stream"] == "joint") {
        return Err(format!(
            "'{text}': its score holds no {mode} rows of role {role}; their roles are {}",
            roles.join(", ")
        ));
    }
    let mode_record = &manifest["modes"][mode];
    let provenance: Provenance =
        serde_json::from_value(mode_record["provenance"].clone()).map_err(|e| format!("score.json: provenance: {e}"))?;
    let basis: PredictiveBasis =
        serde_json::from_value(mode_record["basis"].clone()).map_err(|e| format!("score.json: basis: {e}"))?;
    let stamped: Conditioning =
        serde_json::from_value(manifest["conditioning"].clone()).map_err(|e| format!("score.json: conditioning: {e}"))?;
    let conditioning = match role {
        "trained" => Conditioning::InSample,
        "unknown" => Conditioning::Unknown,
        _ => match &stamped {
            Conditioning::HoldOutTail { train_end, .. } => Conditioning::HoldOutTail {
                train_end: *train_end,
                theta_source: context.fit_folder.clone().unwrap_or_default(),
            },
            other => other.clone(),
        },
    };
    // The options the prediction was made under, which its identity holds.
    let prediction: store::Stored<Prediction> = store
        .get(s.identity().prediction.id())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("'{text}': the prediction its score scores is not stored"))?;
    let settings = settings_of(&prediction)?;
    let held_out_fit = !matches!(stamped, Conditioning::InSample | Conditioning::Unknown);
    let mut overrides = Vec::new();
    if role == "trained" && held_out_fit {
        overrides.push(TraceOverride::InSampleForced);
    }
    // The prediction's flag turned the setting on for a fit made without it.
    if settings["allow_forcing_extrapolation"].as_bool() == Some(true) && context.fit_allows_forcing == Some(false) {
        overrides.push(TraceOverride::ForcingExtrapolationAllowed);
    }
    let strings = |v: &serde_json::Value| -> Vec<String> {
        match v {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        }
    };
    let n_draws = mode_record["n_draws"].as_u64().unwrap_or(1) as usize;
    let derivation = Derivation {
        particles: manifest["particles"].as_u64().unwrap_or(0) as usize,
        replicates: 1,
        dt: context.dt,
        seed: settings["seed"].as_u64().unwrap_or(0),
        theta: match (context.n_available, context.winner) {
            (Some(n_available), _) => ThetaSource::PosteriorDraws { n_used: n_draws, n_available },
            (None, true) => ThetaSource::OptimizerWinner,
            (None, false) => ThetaSource::Supplied,
        },
        excluded_chains: settings["excluded_chains"]
            .as_array()
            .map(|a| a.iter().filter_map(|c| c.as_u64().map(|c| c as usize)).collect())
            .unwrap_or_default(),
        enabled: strings(&settings["enable"]),
        disabled: strings(&settings["disable"]),
        replicate_elpd: Vec::new(),
        replicate_elpd_by_stream: BTreeMap::new(),
        mc_se_elpd: None,
        overrides,
    };
    let forcings_held: Vec<score::trace::PrequentialWarning> = match manifest.get("forcings_held") {
        Some(v) => serde_json::from_value(v.clone()).map_err(|e| format!("score.json: forcings_held: {e}"))?,
        None => Vec::new(),
    };
    let realised = manifest["labels"].as_array().is_some_and(|l| l.iter().any(|x| x == REALISED_LABEL));
    let reach = manifest["reach"].as_str().map(str::to_string);
    let mut notes: Vec<String> =
        manifest["notes"].as_array().map(|n| n.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default();
    notes.dedup();
    // A forecast's rows by lead; the joint row over every target of the
    // forecast, which has no lead, is not a row of any lead.
    let mut leads: Vec<(f64, String)> = Vec::new();
    if mode == "forecast" {
        for r in &rows {
            if let (false, Ok(l)) = (r["lead"].is_empty(), r["lead"].parse::<f64>()) {
                if !leads.iter().any(|(_, s)| s == &r["lead"]) {
                    leads.push((l, r["lead"].clone()));
                }
            }
        }
        leads.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let lead_keys: Vec<Option<String>> = match mode {
        "forecast" => leads.into_iter().map(|(_, s)| Some(s)).collect(),
        _ => vec![None],
    };
    let mut groups = Vec::with_capacity(lead_keys.len());
    for lead in lead_keys {
        let mine: Vec<&Line> = rows.iter().copied().filter(|r| lead.as_ref().is_none_or(|l| &r["lead"] == l)).collect();
        let (steps, no_joint) = steps_of(&mine, mode != "free_forward")?;
        let mut trace = PrequentialTrace {
            schema_version: score::trace::SCHEMA_RECORDING_HELD_FORCINGS,
            t0: 0,
            provenance,
            conditioning: conditioning.clone(),
            basis,
            steps,
            warnings: forcings_held.clone(),
            score_from: match &conditioning {
                Conditioning::HoldOutTail { train_end, .. } => Some(*train_end),
                _ => None,
            },
            pit_randomization_seed: manifest["rules"]["pit_seed"].as_u64(),
            derivation: Some(derivation.clone()),
            scored_streams: None,
        };
        let totals = score::combine::ReplicateTotals::of(std::slice::from_ref(&trace));
        if let Some(d) = trace.derivation.as_mut() {
            d.replicate_elpd = totals.elpd;
            d.replicate_elpd_by_stream = totals.by_stream;
        }
        groups.push(Group {
            trace,
            facts: ScoreFacts {
                mode: mode.to_string(),
                role: role.to_string(),
                lead,
                reach: reach.clone(),
                realised,
                stored: true,
                no_joint,
            },
            notes: notes.clone(),
        });
    }
    Ok(groups)
}

/// One step per row end of `rows` (one role, and one lead for a forecast),
/// in time order: each stream's own scores, and the step's log score, the
/// `joint` row at that end when `joint` is read and one is there, else the
/// one stream's own. The second value is the first end at which several
/// streams are scored with no joint row, whose step score is NaN.
fn steps_of(rows: &[&Line], joint: bool) -> Result<(Vec<PrequentialStep>, Option<f64>), String> {
    let num = |r: &Line, col: &str| -> Result<f64, String> {
        let v = r.get(col).map(String::as_str).unwrap_or("");
        match v {
            "" => Ok(f64::NAN),
            v => v.parse::<f64>().map_err(|e| format!("scored.tsv: `{col}` = {v:?}: {e}")),
        }
    };
    let mut ends: Vec<f64> = Vec::new();
    for r in rows {
        let t = num(r, "t_stop")?;
        if !ends.contains(&t) {
            ends.push(t);
        }
    }
    ends.sort_by(f64::total_cmp);
    let mut steps = Vec::with_capacity(ends.len());
    let mut no_joint = None;
    for t in ends {
        let at: Vec<&&Line> = rows.iter().filter(|r| num(r, "t_stop").is_ok_and(|x| x == t)).collect();
        let mut per_stream = Vec::new();
        let mut joint_score = None;
        for r in &at {
            if r["stream"] == "joint" {
                joint_score = Some(num(r, "log_score")?);
                continue;
            }
            let start = num(r, "t_start")?;
            per_stream.push(StreamScore {
                stream: r["stream"].clone(),
                y_obs: num(r, "observed")?,
                y_pred_samples: Vec::new(),
                log_score: num(r, "log_score")?,
                crps: num(r, "crps")?,
                pit: num(r, "pit")?,
                interval: PredInterval {
                    q05: num(r, "q05")?,
                    q25: num(r, "q25")?,
                    q50: num(r, "q50")?,
                    q75: num(r, "q75")?,
                    q95: num(r, "q95")?,
                },
                coverage: if start == t { Coverage::Instant } else { Coverage::Interval { start, stop: t } },
                wis: Some(num(r, "wis")?).filter(|w| !w.is_nan()),
                in_50: match r.get("in_50").map(String::as_str) {
                    Some("1") => Some(true),
                    Some("0") => Some(false),
                    _ => None,
                },
            });
        }
        if per_stream.is_empty() {
            continue;
        }
        per_stream.sort_by(|a, b| a.stream.cmp(&b.stream));
        let log_score = match (joint.then_some(joint_score).flatten(), per_stream.as_slice()) {
            (Some(j), _) => j,
            (None, [one]) => one.log_score,
            (None, _) => {
                no_joint.get_or_insert(t);
                f64::NAN
            }
        };
        steps.push(PrequentialStep { t, log_score, ess: f64::NAN, per_stream });
    }
    Ok((steps, no_joint))
}

/// Replicate traces of one group combined: each step's log score, and each
/// stream's, the log-mean-exp of the replicates'; every other figure the
/// first replicate's, since a score keeps no predictive samples to pool.
pub(super) fn combine_replicates(traces: &[PrequentialTrace]) -> Result<PrequentialTrace, String> {
    let first = &traces[0];
    let same = traces.iter().all(|t| {
        t.steps.len() == first.steps.len()
            && t.steps.iter().zip(&first.steps).all(|(a, b)| {
                a.t == b.t && a.per_stream.len() == b.per_stream.len()
                    && a.per_stream.iter().zip(&b.per_stream).all(|(x, y)| x.stream == y.stream)
            })
    });
    if !same {
        return Err("internal error: the replicate scores of one prediction scored different rows".into());
    }
    let mut out = first.clone();
    for (k, step) in out.steps.iter_mut().enumerate() {
        let ls: Vec<f64> = traces.iter().map(|t| t.steps[k].log_score).collect();
        step.log_score = numerics::logmeanexp_with_se(&ls).0;
        for (j, ss) in step.per_stream.iter_mut().enumerate() {
            let ls: Vec<f64> = traces.iter().map(|t| t.steps[k].per_stream[j].log_score).collect();
            ss.log_score = numerics::logmeanexp_with_se(&ls).0;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(pairs: &[(&str, &str)]) -> Line {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn stream_row(stream: &str, start: &str, stop: &str, y: &str, ls: &str) -> Line {
        line(&[
            ("stream", stream), ("t_start", start), ("t_stop", stop), ("observed", y), ("log_score", ls),
            ("crps", "0.5"), ("pit", "0.25"), ("q05", "1"), ("q25", "2"), ("q50", "3"), ("q75", "4"), ("q95", "5"),
            ("wis", "0.75"), ("in_50", "1"),
        ])
    }

    /// Each row end is one step: its score is the joint row there when the
    /// mode has joint rows, the one stream's own when one stream is scored
    /// there, and NaN, reported, when several are and no joint row is; each
    /// stream's own figures are the score's, an instant row's coverage an
    /// instant and a period's its window.
    #[test]
    fn a_score_row_end_is_a_step_scored_by_its_joint_row() {
        let rows = [
            stream_row("cases", "6", "7", "3", "-1.5"),
            stream_row("survey", "7", "7", "2", "-0.5"),
            line(&[("stream", "joint"), ("t_stop", "7"), ("log_score", "-1.75")]),
            stream_row("cases", "7", "8", "4", "-2"),
        ];
        let refs: Vec<&Line> = rows.iter().collect();
        let (steps, no_joint) = steps_of(&refs, true).unwrap();
        assert_eq!(no_joint, None);
        assert_eq!(steps.len(), 2);
        assert_eq!((steps[0].t, steps[0].log_score), (7.0, -1.75), "the joint row's score");
        assert_eq!((steps[1].t, steps[1].log_score), (8.0, -2.0), "the one stream's own");
        let cases = &steps[0].per_stream[0];
        assert_eq!((cases.stream.as_str(), cases.y_obs, cases.crps, cases.wis, cases.in_50), ("cases", 3.0, 0.5, Some(0.75), Some(true)));
        assert_eq!(cases.coverage, Coverage::Interval { start: 6.0, stop: 7.0 });
        assert_eq!(steps[0].per_stream[1].coverage, Coverage::Instant);
        assert_eq!(cases.interval.q95, 5.0);

        let (steps, no_joint) = steps_of(&refs, false).unwrap();
        assert_eq!(no_joint, Some(7.0), "free-forward reads no joint row");
        assert!(steps[0].log_score.is_nan());
    }

    /// Replicates combine each step's score, and each stream's, by
    /// log-mean-exp, the log of the mean density and not the mean of logs;
    /// every other figure is the first replicate's.
    #[test]
    fn replicates_combine_their_log_scores_by_log_mean_exp() {
        let rows = [stream_row("cases", "6", "7", "3", "")];
        let refs: Vec<&Line> = rows.iter().collect();
        let trace = |ls: f64, crps: f64| {
            let (mut steps, _) = steps_of(&refs, true).unwrap();
            steps[0].log_score = ls;
            steps[0].per_stream[0].log_score = ls;
            steps[0].per_stream[0].crps = crps;
            PrequentialTrace {
                schema_version: score::trace::SCHEMA_RECORDING_HELD_FORCINGS, t0: 0, provenance: Provenance::PlugIn,
                conditioning: Conditioning::InSample, basis: PredictiveBasis::Filtered, steps, warnings: Vec::new(),
                score_from: None, pit_randomization_seed: None, derivation: None, scored_streams: None,
            }
        };
        let combined = combine_replicates(&[trace(0.2_f64.ln(), 1.0), trace(0.4_f64.ln(), 9.0)]).unwrap();
        assert!((combined.steps[0].log_score - 0.3_f64.ln()).abs() < 1e-12, "{}", combined.steps[0].log_score);
        assert!((combined.steps[0].per_stream[0].log_score - 0.3_f64.ln()).abs() < 1e-12);
        assert_eq!(combined.steps[0].per_stream[0].crps, 1.0, "the first replicate's");
        let mut other = trace(-1.0, 1.0);
        other.steps[0].t = 8.0;
        assert!(combine_replicates(&[trace(-1.0, 1.0), other]).is_err(), "replicates of other rows");
    }
}
