//! A fit on the run store: the fit problem, the runs on it, and the
//! predictions made from a run (`docs/dev/proposals/2026-11-20-run-store-api.md`,
//! "Worked example: `camdl fit run`" and "Predictions").
//!
//! A [`FitProblem`] is written once and shared by every run on it. It holds
//! what a fit is fitted to and how the problem is posed: the training rows of
//! each bound data file, the dates of the held-out rows (never their values),
//! the external tables, the estimated set with its bounds and priors, the
//! fixed values, the settings, the scenario selection and `ic_free`. A
//! [`FitRun`] sits inside its problem at `runs/`, grouped by the method's
//! settings and named by the seed typed; its identity holds the method, the
//! per-parameter perturbation (`rw_sd`, `perturb_only_at_t0`, `start`), the
//! seed, where its chains start and its length. A [`Prediction`] from a
//! posterior sits inside the run whose draws it used, at `predict/`.
//!
//! ```text
//! <build>/fit/<model>-…/                       FitProblem
//!   problem.toml  partition.json  inputs/<source>.<ext>  held_out/<stream>.tsv
//!   inputs/tables/<name>.<ext>  model.camdl
//!   runs/<algorithm>-…/seed_<n>-…/             FitRun
//!     method.toml  …
//!     predict/<mode>-…/                        Prediction
//! ```
//!
//! Every reader of a stored fit reads it through [`read_fit`], which gives
//! the compile the problem was posed on, the [`config_v2::FitConfig`] the
//! problem (and the run, when one is named) was resolved to, and the bytes of
//! each stored input, all read through the store. `problem.toml` is written in
//! `fit.toml` syntax from the identity alone, so it is a function of the
//! problem's identity, and it is parsed back by the production parser before
//! it is stored.
//!
//! Each training file is stored with only the lines that hold a training row
//! of one of its streams, and the lines that give no stream a row, every column
//! kept; when every stream trains on all its rows, the stored file is the
//! source's bytes. The loader itself says which stream each line gives a row
//! to. The result is checked by binding the stored bytes again: each stream
//! must read exactly its training rows, by count and row digest, or the
//! problem is refused. The rebuilt problem therefore names no `holdout_after`
//! and no `[data.holdout]`; which rows trained is `partition.json`.

use std::collections::BTreeMap;
use std::path::Path;

use indexmap::IndexMap;
use store::{Artifact, Content, Dep, Finite, Handle, Hashed, Kinds, Layout, Placed, Primary, Run, RunRef, Store, Stored, WriteMode};

use crate::camdl_store::{CompiledModel, DataTable, ProcessSeed, ResolvedParams, Table};
use crate::fit::config_v2::{self, EstimateSpecV2, FitConfig, Method, Problem};
use crate::fit::observed_rows::{Measures, ObservedRow};
use crate::fit::training_partition::row_digest;
use crate::run_meta::TrainingPartitionRecord;

// ── Values a fit's identities hold ───────────────────────────────────────────

/// One stored input file: its bytes and the extension it is stored under,
/// which the data loader reads.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_input_file", version = 1)]
pub(crate) struct InputFile {
    pub ext: String,
    pub content: Hashed<Table>,
}

impl InputFile {
    fn bytes(&self) -> Vec<u8> {
        self.content.get().to_bytes()
    }
}

/// The held-out rows of one stream, without their values: per row, the file
/// it is in (`training`, past the training prefix, or `holdout`, a
/// `[data.holdout]` file), its position in that file, its label and what it
/// measures. A forecast names its targets from these; their outcomes enter
/// only a score.
pub(crate) struct HeldOutDates {
    bytes: Vec<u8>,
}

impl HeldOutDates {
    const HEADER: &'static str = "file\tposition\tlabel\tmeasures\tstart\tend\n";

    fn of(rows: &[(&'static str, &ObservedRow)]) -> Self {
        let mut text = String::from(Self::HEADER);
        for (file, r) in rows {
            let kind = match r.measures {
                Measures::Instant(_) => "instant",
                Measures::Period(_) => "period",
            };
            text.push_str(&format!(
                "{file}\t{}\t{}\t{kind}\t{}\t{}\n",
                r.position,
                r.label,
                r.measures.start(),
                r.measures.end()
            ));
        }
        HeldOutDates { bytes: text.into_bytes() }
    }
}

impl Content for HeldOutDates {
    fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    fn from_bytes(bytes: &[u8]) -> store::Result<Self> {
        Ok(HeldOutDates { bytes: bytes.to_vec() })
    }
}

/// How the training rows were decided, under the partition rule's version.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_partition", version = 1)]
pub(crate) struct Partition {
    pub rule_version: u16,
    pub rule: PartitionRule,
}

#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_partition_rule", version = 1)]
pub(crate) enum PartitionRule {
    /// No holdout: every row trains.
    All,
    /// `holdout_after`, resolved to model time: a row trains if its end is at
    /// or before `at`.
    Cutoff { at: Finite },
    /// `[data.holdout]`: every row of the training files trains.
    SeparateFiles,
}

/// One estimated parameter as the problem poses it.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_estimated", version = 1)]
pub(crate) struct Estimated {
    pub name: String,
    pub bounds: Option<(Finite, Finite)>,
    /// The `prior` entry, as canonical JSON of its parsed form.
    pub prior: Option<String>,
}

/// The estimated set, in the order of the file's `[estimate]` table, which
/// camdl reads in order.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_estimate", version = 1)]
pub(crate) struct Estimate {
    pub parameters: Vec<Estimated>,
}

/// One estimated parameter's settings that belong to the run.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_perturbation_entry", version = 1)]
pub(crate) struct Perturbation {
    pub name: String,
    pub rw_sd: Option<Finite>,
    pub perturb_only_at_t0: bool,
    pub start: Option<Finite>,
}

/// `rw_sd`, `perturb_only_at_t0` and `start` of every estimated parameter, in
/// the order of `[estimate]`. They sit in `[estimate]` in the file and change
/// only how a run moves, so two runs that differ only in them share a problem.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_perturbation", version = 1)]
pub(crate) struct PerturbationSettings {
    pub parameters: Vec<Perturbation>,
}

/// `[config]`, its default filled in: the step, and the two policies.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_settings", version = 1)]
pub(crate) struct FitSettings {
    pub dt: Finite,
    pub allow_degenerate_rates: bool,
    pub allow_forcing_extrapolation: bool,
}

/// The scenario selection as given. A named scenario both selects
/// interventions and sets parameter values, and every reader applies it to
/// its own compile of the model, so the selection is held as written.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_scenario", version = 1)]
pub(crate) struct ScenarioRequest {
    pub scenario: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
}

// ── The fit problem ──────────────────────────────────────────────────────────

/// Everything that poses a fit problem: what is fitted, to which rows, with
/// which parameters free. No method setting is here.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_problem", version = 1)]
pub(crate) struct FitProblemIdentity {
    /// The compile rewritten from the data (anchored times, bound tables).
    pub model: Dep<CompiledModel>,
    /// Each bound data file's training lines, by the source it is bound to,
    /// in binding order.
    pub training: Vec<(String, InputFile)>,
    /// Each stream's held-out rows without their values; empty when nothing
    /// is held out.
    pub held_out: BTreeMap<String, Hashed<HeldOutDates>>,
    pub partition: Partition,
    /// Tables bound with `external("NAME")`, by name.
    pub tables: BTreeMap<String, InputFile>,
    pub estimate: Estimate,
    /// `[fixed]`, resolved: a `from_file` read and a `from_scenario` expanded.
    pub fixed: ResolvedParams,
    pub simplex_groups: Vec<Vec<String>>,
    pub settings: FitSettings,
    pub scenario: ScenarioRequest,
    pub ic_free: bool,
}

/// The model file's stem and, in a sweep, the point's values as typed.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProblemLabels {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sweep_point: Option<String>,
}

/// A fit problem, written once.
pub(crate) struct FitProblem;

impl Artifact for FitProblem {
    const KIND: &'static str = "fit";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = FitProblemIdentity;
    type Labels = ProblemLabels;
}

impl Primary for FitProblem {
    fn layout(_id: &FitProblemIdentity, labels: &ProblemLabels) -> Placed {
        Layout::in_kind().result(&labels.model)
    }
}

// ── The fit run ──────────────────────────────────────────────────────────────

/// The method's settings without its length (PGAS `sweeps`, PMMH and MH
/// `iterations`) and without `starts`, which [`StartSource`] holds.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_method", version = 1)]
pub(crate) struct MethodSettings {
    /// The `algorithm` key.
    pub algorithm: String,
    /// The rest of `[method]`, as canonical JSON of its parsed form.
    pub settings: String,
}

/// What a start rule reads besides the problem.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_start_input", version = 1)]
pub(crate) enum StartInput {
    /// A rule that reads nothing: the prior, a spread, the declared values.
    None,
    /// `from_mle` or `from_posterior`: the part of a stored run's file the
    /// starts are read from.
    Run { source: Dep<FitRun> },
    /// `from_params`: the file's bytes.
    File { file: InputFile },
}

/// Where a run's chains start: the rule, and what it reads.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_starts", version = 1)]
pub(crate) struct StartSource {
    /// The rule's identity value (`ChainStarts::identity_value`), as
    /// canonical JSON.
    pub rule: String,
    pub input: StartInput,
}

/// Everything about a run except how long it is. A resume must match this
/// exactly.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_chain", version = 1)]
pub(crate) struct ChainIdentity {
    pub problem: Dep<FitProblem>,
    pub method: MethodSettings,
    pub perturbation: PerturbationSettings,
    /// The compile with the state Jacobian; only `nl-lbfgs` on the ODE backend.
    pub jacobian_model: Option<Dep<CompiledModel>>,
    pub seed: ProcessSeed,
    pub starts: StartSource,
}

/// One run on a problem.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_run", version = 1)]
pub(crate) struct FitRunIdentity {
    pub chain: ChainIdentity,
    /// Sweeps or iterations, for a method that can be extended; 0 otherwise.
    pub length: u64,
    /// The base run's carried chain files, for a resumed run.
    pub resumed_from: Option<Dep<FitRun>>,
}

/// The method's name and the seed the user typed.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct RunLabels {
    pub method: String,
    pub seed: u64,
}

/// One fit run, written in place.
pub(crate) struct FitRun;

impl Artifact for FitRun {
    const KIND: &'static str = "fit_run";
    const WRITE: WriteMode = WriteMode::InPlace;
    type Identity = FitRunIdentity;
    type Labels = RunLabels;
}

impl Primary for FitRun {
    fn layout(id: &FitRunIdentity, labels: &RunLabels) -> Placed {
        Layout::inside(id.chain.problem.id(), "runs")
            .group(&labels.method, &id.chain.method)
            .result(&format!("seed_{}", labels.seed))
    }
}

impl MethodSettings {
    /// The settings of `method` and its length. Its `starts` must be resolved,
    /// and a value that is NaN or infinite is refused: JSON writes either as
    /// `null`, so two such methods would otherwise share a key.
    pub fn of(method: &Method) -> Result<(MethodSettings, u64), String> {
        crate::fit::cas::ensure_finite(method)?;
        let whole = serde_json::to_value(method).map_err(|e| format!("cannot encode [method]: {e}"))?;
        let algorithm = whole
            .get("algorithm")
            .and_then(|a| a.as_str())
            .ok_or("internal error: [method] encodes without its algorithm")?
            .to_string();
        let length = match method.algorithm.extension_key() {
            Some(key) => whole
                .get(key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("internal error: [method] encodes without `{key}`"))?,
            None => 0,
        };
        let mut exclude = vec!["starts"];
        exclude.extend(method.algorithm.extension_key());
        let rest = crate::fit::cas::serialize_minus(method, &exclude)?;
        Ok((MethodSettings { algorithm, settings: canonical_json(&rest)? }, length))
    }
}

impl StartSource {
    /// The rule `method` starts its chains under, reading `input`.
    /// A NaN or infinite value in the method is refused, as for
    /// [`MethodSettings::of`].
    pub fn of(method: &Method, input: StartInput) -> Result<StartSource, String> {
        crate::fit::cas::ensure_finite(method)?;
        let rule = method.starts()?.identity_value();
        Ok(StartSource { rule: canonical_json(&rule)?, input })
    }
}

// ── Predictions ──────────────────────────────────────────────────────────────

/// What a prediction predicts.
#[derive(store::Identity, Clone, Copy, Debug, PartialEq, Eq)]
#[identity(tag = "predict_mode", version = 1)]
pub(crate) enum PredictMode {
    All,
    FreeForward,
    OneStep,
    Forecast,
}

impl PredictMode {
    pub fn label(self) -> &'static str {
        match self {
            PredictMode::All => "all",
            PredictMode::FreeForward => "free_forward",
            PredictMode::OneStep => "one_step",
            PredictMode::Forecast => "forecast",
        }
    }
}

/// How a prediction is made, besides where its parameters come from.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "predict_options", version = 1)]
pub(crate) struct PredictOptions {
    pub mode: PredictMode,
    /// A forecast's reach past its origin, in model time.
    pub reach: Option<Finite>,
    pub draws: u64,
    pub seed: ProcessSeed,
    /// The streams predicted; empty for every eligible stream.
    pub streams: Vec<String>,
    /// The arms, each a scenario selection, in the order asked for.
    pub scenarios: Vec<ScenarioRequest>,
    /// Sweep arms: a parameter and its values.
    pub sweep: Vec<(String, Vec<Finite>)>,
    pub by_chain: bool,
    pub excluded_chains: Vec<u64>,
    pub particles: Option<u64>,
    pub allow_forcing_extrapolation: bool,
}

/// Where a prediction's parameters come from.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "parameter_source", version = 1)]
pub(crate) enum ParameterSource {
    /// A fit run's draws, and the problem they were fitted on.
    Posterior { draws: Dep<FitRun>, problem: Dep<FitProblem> },
    /// One point, with no parameter uncertainty.
    FixedPoint { model: Dep<CompiledModel>, params: ResolvedParams, trained_through: Option<Finite> },
}

/// One prediction.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "prediction", version = 1)]
pub(crate) struct PredictIdentity {
    pub source: ParameterSource,
    /// The data the filter conditions on, by stream; never held-out outcomes.
    pub data: BTreeMap<String, Dep<DataTable>>,
    pub options: PredictOptions,
}

/// The mode, the scenario names as typed, and the model file's stem.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct PredictionLabels {
    pub mode: String,
    pub scenarios: Vec<String>,
    pub model: String,
}

/// One prediction, written once.
pub(crate) struct Prediction;

impl Artifact for Prediction {
    const KIND: &'static str = "prediction";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = PredictIdentity;
    type Labels = PredictionLabels;
}

impl Primary for Prediction {
    fn layout(id: &PredictIdentity, labels: &PredictionLabels) -> Placed {
        match &id.source {
            ParameterSource::Posterior { draws, .. } => Layout::inside(draws.id(), "predict"),
            ParameterSource::FixedPoint { model, .. } => Layout::in_kind().group(&labels.model, model),
        }
        .result(&labels.mode)
    }
}

/// Adds every kind of this module to `kinds`.
pub(crate) fn register(kinds: Kinds) -> Kinds {
    kinds.register::<FitProblem>().register::<FitRun>().register::<Prediction>()
}

// ── Resolving a problem ──────────────────────────────────────────────────────

/// The model file a problem was compiled from, kept as its provenance file.
pub(crate) struct ModelSourceFile {
    pub bytes: Vec<u8>,
    /// `true` for camdl source, `false` for compiled IR JSON.
    pub camdl: bool,
}

impl ModelSourceFile {
    fn name(&self) -> &'static str {
        if self.camdl {
            "model.camdl"
        } else {
            "model.ir.json"
        }
    }
}

/// A fit problem resolved against its model: the identity, and what writing
/// it needs besides.
pub(crate) struct ResolvedProblem {
    pub identity: FitProblemIdentity,
    pub labels: ProblemLabels,
    /// The model `identity.model` names, as compiled.
    model: ir::Model,
    source: ModelSourceFile,
    partition: TrainingPartitionRecord,
}

/// The stored problem's files.
pub(crate) const PROBLEM_FILE: &str = "problem.toml";
pub(crate) const PARTITION_FILE: &str = "partition.json";
/// The run's merged `[method]` and its `[estimate]` perturbation keys.
pub(crate) const METHOD_FILE: &str = "method.toml";

fn training_file(source: &str, ext: &str) -> String {
    format!("inputs/{source}.{ext}")
}

fn table_file(name: &str, ext: &str) -> String {
    format!("inputs/tables/{name}.{ext}")
}

fn held_out_file(stream: &str) -> String {
    format!("held_out/{stream}.tsv")
}

fn extension(path: &str) -> String {
    Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("tsv").to_string()
}

fn finite(x: f64, what: &str) -> Result<Finite, String> {
    Finite::new(x).map_err(|e| format!("{what}: {e}"))
}

fn canonical_json(v: &serde_json::Value) -> Result<String, String> {
    serde_json::to_string(&crate::fit::cas::canonical(v)).map_err(|e| e.to_string())
}

fn read_file(path: &str, what: &str) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("cannot read {what} ({path}): {e}"))
}

/// The time options every data file of `model` is read under.
fn time_opts(model: &ir::Model, dt: f64) -> crate::caltime_load::TimeOpts<'_> {
    crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    }
}

/// Resolve `problem` against `model`, the compile `model_dep` names
/// (anchored times and bound tables written in).
///
/// `problem` must bind its data through `[data]`: generated datasets are
/// resolved after they are written, as data. Paths in `problem` are read as
/// given, so they must already be anchored.
pub(crate) fn resolve_problem(
    problem: &Problem,
    model: &ir::Model,
    model_dep: Dep<CompiledModel>,
    source: ModelSourceFile,
    labels: ProblemLabels,
) -> Result<ResolvedProblem, String> {
    let Problem {
        model: _,
        data,
        synthetic,
        simplex_groups,
        output_dir: _,
        estimate,
        fixed,
        config,
        simulate: _,
        scenario,
        enable,
        disable,
        ic_free,
        provenance: _,
        compiled_ir: _,
    } = problem;
    if model_dep.digest() != crate::camdl_store::model_part_digest(model) {
        return Err("internal error: the model given is not the stored compile the problem names".into());
    }
    if synthetic.is_some() {
        return Err("internal error: a [synthetic] problem is resolved once its datasets are written as data".into());
    }
    let data = data.as_ref().ok_or("a fit problem needs a [data] table")?;
    let config_v2::DataSpec { file: _, observations: _, holdout_after, holdout: _, tables } = data;

    let dt = crate::dt_rule::resolved_dt(model.simulation.dt, config.dt);
    let opts = time_opts(model, dt);
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)?;
    if bindings.training_by_source().is_empty() {
        return Err("fit.toml [data] binds no observation stream".into());
    }
    let obs_of = |leaf: &str| model.observations.iter().find(|o| o.name == leaf);
    let siblings_of = |source: &str| -> Vec<&ir::observation::ObservationModel> {
        model.observations.iter().filter(|o| o.source == source).collect()
    };
    let load = |path: &str, leaf: &str| -> Result<Vec<ObservedRow>, String> {
        let obs = obs_of(leaf).ok_or_else(|| format!("internal error: no stream `{leaf}`"))?;
        crate::fit::runner::load_observations(path, obs, &siblings_of(&obs.source), &opts)
    };

    // Every bound stream's rows, as the fit reads them.
    let mut rows: BTreeMap<String, Vec<ObservedRow>> = BTreeMap::new();
    for (leaf, source) in bindings.leaves() {
        let path = &bindings.training_by_source()[source];
        rows.insert(leaf.to_string(), load(path, leaf)?);
    }

    // Which rows train.
    use crate::fit::training_partition::WholeFiles;
    let holdout_files = bindings.holdout_by_source();
    let (partition, rule) = match holdout_after {
        Some(spec) => {
            let p = crate::fit::runner::cutoff_partition(spec, &rows, model)?;
            let at = match p.rule() {
                crate::fit::training_partition::PartitionRule::Cutoff { at } => at,
                _ => return Err("internal error: a cutoff partition without its cutoff".into()),
            };
            (p, PartitionRule::Cutoff { at: finite(at, "holdout_after")? })
        }
        None if !holdout_files.is_empty() => {
            let p = crate::fit::runner::whole_partition(&rows, WholeFiles::SeparateHoldoutFiles)?;
            let obs: Vec<&ir::observation::ObservationModel> = rows.keys().filter_map(|l| obs_of(l)).collect();
            crate::fit::runner::check_holdout_files(&p, &rows, &obs, holdout_files, model, &opts)?;
            (p, PartitionRule::SeparateFiles)
        }
        None => (crate::fit::runner::whole_partition(&rows, WholeFiles::NoHoldout)?, PartitionRule::All),
    };
    let calendar = ir::caltime::CalendarAxis::from_model(model).map_err(|e| e.to_string())?;
    let record = partition.record(&calendar);

    // Each training file, with only its training lines.
    let mut training = Vec::new();
    for (source, path) in bindings.training_by_source() {
        let leaves = bindings.leaves_of(source);
        let ext = extension(path);
        let bytes = read_file(path, &format!("data file '{source}'"))?;
        let cut = cut_to_training(&bytes, &ext, source, &leaves, &record, &rows, &load)?;
        let content = Hashed::new(Table::from_bytes(&cut).map_err(|e| e.to_string())?);
        training.push((source.clone(), InputFile { ext, content }));
    }

    // The held-out rows' dates, per stream.
    let mut held_out = BTreeMap::new();
    if !matches!(rule, PartitionRule::All) {
        for (leaf, source) in bindings.leaves() {
            let mut dates: Vec<(&'static str, ObservedRow)> =
                rows[leaf][partition.rows(leaf)..].iter().map(|r| ("training", r.clone())).collect();
            if let Some(path) = holdout_files.get(source) {
                dates.extend(load(path, leaf)?.into_iter().map(|r| ("holdout", r)));
            }
            let refs: Vec<(&'static str, &ObservedRow)> = dates.iter().map(|(f, r)| (*f, r)).collect();
            held_out.insert(leaf.to_string(), Hashed::new(HeldOutDates::of(&refs)));
        }
    }

    let tables = tables
        .iter()
        .map(|(name, path)| {
            let bytes = read_file(path, &format!("external table '{name}'"))?;
            let content = Hashed::new(Table::from_bytes(&bytes).map_err(|e| e.to_string())?);
            Ok((name.clone(), InputFile { ext: extension(path), content }))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;

    let mut fixed_cfg = fixed.clone();
    fixed_cfg.expand_from_scenario(model, estimate)?;
    let fixed = ResolvedParams::new(fixed_cfg.resolve_with_model(model)?)?;
    let (estimate, _) = split_estimate(estimate)?;
    let config_v2::FitBackendConfig { dt: _, allow_degenerate_rates, allow_forcing_extrapolation } = config;

    let identity = FitProblemIdentity {
        model: model_dep,
        training,
        held_out,
        partition: Partition { rule_version: record.rule_version, rule },
        tables,
        estimate,
        fixed,
        simplex_groups: simplex_groups.iter().map(|g| g.params.clone()).collect(),
        settings: FitSettings {
            dt: finite(dt, "[config] dt")?,
            allow_degenerate_rates: *allow_degenerate_rates,
            allow_forcing_extrapolation: *allow_forcing_extrapolation,
        },
        scenario: ScenarioRequest { scenario: scenario.clone(), enable: enable.clone(), disable: disable.clone() },
        ic_free: ic_free.unwrap_or(false),
    };
    Ok(ResolvedProblem { identity, labels, model: model.clone(), source, partition: record })
}

/// The problem's half of `[estimate]`, and the settings that belong to the run.
pub(crate) fn split_estimate(
    estimate: &IndexMap<String, EstimateSpecV2>,
) -> Result<(Estimate, PerturbationSettings), String> {
    let mut problem = Vec::new();
    let mut run = Vec::new();
    for (name, spec) in estimate {
        let EstimateSpecV2 { bounds, prior, perturb_only_at_t0, rw_sd, start } = spec;
        let what = |key: &str| format!("[estimate.{name}] {key}");
        let bounds = match bounds {
            Some((lo, hi)) => Some((finite(*lo, &what("bounds"))?, finite(*hi, &what("bounds"))?)),
            None => None,
        };
        let prior = match prior {
            Some(p) => Some(canonical_json(&serde_json::to_value(p).map_err(|e| e.to_string())?)?),
            None => None,
        };
        problem.push(Estimated { name: name.clone(), bounds, prior });
        run.push(Perturbation {
            name: name.clone(),
            rw_sd: rw_sd.map(|x| finite(x, &what("rw_sd"))).transpose()?,
            perturb_only_at_t0: *perturb_only_at_t0,
            start: start.map(|x| finite(x, &what("start"))).transpose()?,
        });
    }
    Ok((Estimate { parameters: problem }, PerturbationSettings { parameters: run }))
}

/// The lines of `bytes`, each with its line terminator.
fn lines_of(bytes: &[u8]) -> Vec<&[u8]> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'\n' {
            lines.push(&bytes[start..=i]);
            start = i + 1;
        }
    }
    if start < bytes.len() {
        lines.push(&bytes[start..]);
    }
    lines
}

/// `bytes`, a data file read by the streams `leaves`, with only the lines
/// that hold a training row of some stream, and every line that gives no
/// stream a row; the whole file when every stream trains on all its rows.
///
/// Which stream a line gives a row to is read by the loader itself: each data
/// line is bound alone, under the lines that precede the first data line, and
/// the rows each stream reads from it are counted. A stream's rows are its
/// lines in file order, so its first `n` such lines are its training rows, in
/// whatever order the file interleaves its streams. The result is refused
/// unless binding it gives each stream exactly its training rows, by count and
/// row digest, which fails only when streams of one file train on different
/// numbers of its lines.
fn cut_to_training(
    bytes: &[u8],
    ext: &str,
    source: &str,
    leaves: &[&str],
    record: &TrainingPartitionRecord,
    rows: &BTreeMap<String, Vec<ObservedRow>>,
    load: &dyn Fn(&str, &str) -> Result<Vec<ObservedRow>, String>,
) -> Result<Vec<u8>, String> {
    let prefix = |leaf: &str| record.streams.iter().find(|p| p.stream == leaf).map(|p| (p.rows, p.digest.clone()));
    let mut wanted = Vec::new();
    for leaf in leaves {
        wanted.push((*leaf, prefix(leaf).ok_or_else(|| format!("internal error: no training rows for `{leaf}`"))?));
    }
    if wanted.iter().all(|(leaf, (n, _))| *n == rows[*leaf].len()) {
        return Ok(bytes.to_vec());
    }
    let dir = tempfile::tempdir().map_err(|e| format!("cannot make a temporary directory: {e}"))?;
    let path = dir.path().join(format!("cut.{ext}"));
    let path_text = path.to_string_lossy().into_owned();
    let bind = |content: &[u8]| -> Result<Vec<Vec<ObservedRow>>, String> {
        std::fs::write(&path, content).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        wanted.iter().map(|(leaf, _)| load(&path_text, leaf)).collect()
    };
    let lines = lines_of(bytes);
    // The lines before the first data line: the fewest leading lines the
    // loader reads as a file, which give no stream a row.
    let head = (0..lines.len())
        .find(|&h| bind(&lines[..=h].concat()).is_ok_and(|got| got.iter().all(Vec::is_empty)))
        .ok_or_else(|| format!("data file '{source}' has no header the loader reads"))?;
    let mut kept: Vec<u8> = lines[..=head].concat();
    let mut seen = vec![0usize; wanted.len()];
    for line in &lines[head + 1..] {
        let mut probe = lines[..=head].concat();
        probe.extend_from_slice(line);
        let got = bind(&probe)?;
        let mut keep = got.iter().all(Vec::is_empty);
        for (i, g) in got.iter().enumerate() {
            if g.len() > 1 {
                return Err(format!("internal error: one line of data file '{source}' gave {} rows of one stream", g.len()));
            }
            if g.len() == 1 {
                keep |= seen[i] < wanted[i].1 .0;
                seen[i] += 1;
            }
        }
        if keep {
            kept.extend_from_slice(line);
        }
    }
    for (g, (leaf, (n, digest))) in bind(&kept)?.iter().zip(&wanted) {
        if g.len() != *n || row_digest(g)?.to_hex() != *digest {
            return Err(format!(
                "the streams that read data file '{source}' train on different numbers of its lines ('{leaf}' \
                 trains on {n} of its {} rows), so the file cannot be stored with only training rows; bind '{leaf}' \
                 to a file of its own",
                rows[*leaf].len()
            ));
        }
    }
    Ok(kept)
}

// ── The stored files ─────────────────────────────────────────────────────────

/// A TOML key, bare when it can be.
fn toml_key(key: &str) -> String {
    if !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        key.to_string()
    } else {
        toml::Value::String(key.to_string()).to_string()
    }
}

/// The text of `problem.toml`: the problem `id` poses, in `fit.toml` syntax,
/// with every path relative to the problem's folder. `model` is the compile
/// `id.model` names, whose start is written as `[simulate] from`.
fn problem_toml(id: &FitProblemIdentity, model: &ir::Model, source: &ModelSourceFile) -> Result<String, String> {
    use toml::Value;
    let FitProblemIdentity {
        model: _,
        training,
        held_out: _,
        partition: _,
        tables,
        estimate,
        fixed,
        simplex_groups,
        settings,
        scenario,
        ic_free,
    } = id;
    let mut doc = toml::Table::new();
    doc.insert("model".into(), Value::Table([("camdl".to_string(), Value::String(source.name().into()))].into_iter().collect()));
    doc.insert(
        "fixed".into(),
        Value::Table(fixed.values.iter().map(|(k, v)| (k.clone(), Value::Float(v.get()))).collect()),
    );
    let FitSettings { dt, allow_degenerate_rates, allow_forcing_extrapolation } = settings;
    doc.insert(
        "config".into(),
        Value::Table(
            [
                ("dt".to_string(), Value::Float(dt.get())),
                ("allow_degenerate_rates".to_string(), Value::Boolean(*allow_degenerate_rates)),
                ("allow_forcing_extrapolation".to_string(), Value::Boolean(*allow_forcing_extrapolation)),
            ]
            .into_iter()
            .collect(),
        ),
    );
    doc.insert(
        "simulate".into(),
        Value::Table([("from".to_string(), Value::Float(model.simulation.t_start))].into_iter().collect()),
    );
    let ScenarioRequest { scenario, enable, disable } = scenario;
    if let Some(s) = scenario {
        doc.insert("scenario".into(), Value::String(s.clone()));
    }
    let strings = |v: &[String]| Value::Array(v.iter().cloned().map(Value::String).collect());
    if !enable.is_empty() {
        doc.insert("enable".into(), strings(enable));
    }
    if !disable.is_empty() {
        doc.insert("disable".into(), strings(disable));
    }
    doc.insert("ic_free".into(), Value::Boolean(*ic_free));
    if !simplex_groups.is_empty() {
        doc.insert(
            "simplex_groups".into(),
            Value::Array(
                simplex_groups
                    .iter()
                    .map(|g| Value::Table([("params".to_string(), strings(g))].into_iter().collect()))
                    .collect(),
            ),
        );
    }
    if !tables.is_empty() {
        let t: toml::Table =
            tables.iter().map(|(name, f)| (name.clone(), Value::String(table_file(name, &f.ext)))).collect();
        doc.insert("data".into(), Value::Table([("tables".to_string(), Value::Table(t))].into_iter().collect()));
    }
    let mut text = String::from(
        "# problem.toml: the fit problem this folder holds, in fit.toml syntax. Each data file is\n\
         # stored with its training rows only; which rows trained is partition.json.\n",
    );
    text.push_str(&toml::to_string(&doc).map_err(|e| format!("cannot write problem.toml: {e}"))?);
    // The two tables camdl reads in order are written in order.
    text.push_str("\n[data.observations]\n");
    for (source, f) in training {
        text.push_str(&format!("{} = {}\n", toml_key(source), Value::String(training_file(source, &f.ext))));
    }
    text.push_str("\n[estimate]\n");
    for e in &estimate.parameters {
        let mut t = toml::Table::new();
        if let Some((lo, hi)) = e.bounds {
            t.insert("bounds".into(), Value::Array(vec![Value::Float(lo.get()), Value::Float(hi.get())]));
        }
        if let Some(prior) = &e.prior {
            let spec: config_v2::EstimatePriorSpec = serde_json::from_str(prior).map_err(|e| e.to_string())?;
            t.insert("prior".into(), Value::try_from(&spec).map_err(|e| format!("cannot write a prior: {e}"))?);
        }
        text.push_str(&format!("{} = {}\n", toml_key(&e.name), Value::Table(t)));
    }
    Ok(text)
}

/// Write a claimed problem's files: `problem.toml`, `partition.json`, each
/// training file, each table and each stream's held-out dates as result
/// files, and the model file as a provenance file. `problem.toml` is refused
/// unless the production parser reads it back.
pub(crate) fn write_problem(run: &mut Run<FitProblem>, problem: &ResolvedProblem) -> Result<(), String> {
    let id = &problem.identity;
    let text = problem_toml(id, &problem.model, &problem.source)?;
    FitConfig::from_toml_str_named(&text, PROBLEM_FILE)
        .map_err(|e| format!("internal error: the problem.toml this fit would store does not parse back: {e}"))?;
    let e = |e: store::Error| e.to_string();
    run.result_file(PROBLEM_FILE).map_err(e)?.write_all(text.as_bytes()).map_err(e)?;
    let partition = serde_json::to_vec_pretty(&problem.partition).map_err(|e| e.to_string())?;
    run.result_file(PARTITION_FILE).map_err(e)?.write_all(&partition).map_err(e)?;
    for (source, f) in &id.training {
        run.result_file(&training_file(source, &f.ext)).map_err(e)?.write_all(&f.bytes()).map_err(e)?;
    }
    for (name, f) in &id.tables {
        run.result_file(&table_file(name, &f.ext)).map_err(e)?.write_all(&f.bytes()).map_err(e)?;
    }
    for (stream, dates) in &id.held_out {
        run.result_file(&held_out_file(stream)).map_err(e)?.write_all(&dates.get().to_bytes()).map_err(e)?;
    }
    run.provenance_file(problem.source.name()).map_err(e)?.write_all(&problem.source.bytes).map_err(e)?;
    Ok(())
}

/// One estimated parameter's run settings, as `method.toml` writes them.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PerturbationEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rw_sd: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    perturb_only_at_t0: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    start: Option<f64>,
}

/// What `method.toml` holds.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodToml {
    method: Method,
    #[serde(default)]
    estimate: BTreeMap<String, PerturbationEntry>,
}

/// The text of `method.toml`: `method`, after the flags were applied, and
/// each estimated parameter's run settings. A sourced `starts` is written as
/// `method` holds it, so the caller names its source as the run stores it.
fn method_toml(method: &Method, perturbation: &PerturbationSettings) -> Result<String, String> {
    let estimate = perturbation
        .parameters
        .iter()
        .map(|p| {
            let entry = PerturbationEntry {
                rw_sd: p.rw_sd.map(Finite::get),
                perturb_only_at_t0: p.perturb_only_at_t0,
                start: p.start.map(Finite::get),
            };
            (p.name.clone(), entry)
        })
        .collect();
    let doc = MethodToml { method: method.clone(), estimate };
    let text = toml::to_string(&doc).map_err(|e| format!("cannot write method.toml: {e}"))?;
    let back: MethodToml =
        toml::from_str(&text).map_err(|e| format!("internal error: method.toml does not parse back: {e}"))?;
    let json = |m: &Method| serde_json::to_value(m).map_err(|e| e.to_string());
    if json(&back.method)? != json(method)? {
        return Err("internal error: method.toml reads back to another method".into());
    }
    Ok(format!("# method.toml: this run's [method] and the run settings of [estimate].\n{text}"))
}

/// Write a claimed run's `method.toml`.
pub(crate) fn write_method(run: &mut Run<FitRun>, method: &Method) -> Result<(), String> {
    let text = method_toml(method, &run.identity().chain.perturbation)?;
    let e = |e: store::Error| e.to_string();
    run.result_file(METHOD_FILE).map_err(e)?.write_all(text.as_bytes()).map_err(e)
}

// ── Reading a stored fit ─────────────────────────────────────────────────────

/// One stored input file of a problem, read through the store.
pub(crate) struct StoredInput {
    /// The file's name in the problem, which `config` names it by.
    pub name: String,
    /// The extension the loader reads it under.
    pub ext: String,
    /// The bytes, checked against the file's digest.
    pub bytes: Vec<u8>,
}

/// What a fit handle names: a problem, or one run and its problem, with
/// every input the problem holds read through the store.
pub(crate) struct StoredFit {
    pub problem: Stored<FitProblem>,
    pub run: Option<Stored<FitRun>>,
    /// The compile the problem was posed on. A reader runs this, never the
    /// model source, which is a provenance file.
    pub model: Dep<CompiledModel>,
    /// `problem.toml`, with `method.toml` when `run` is `Some`, parsed. Its
    /// `[model]` and `[data]` entries are the names of stored files, not
    /// paths: the files are `training` and `tables`, the model is `model`.
    pub config: FitConfig,
    /// Each bound source's training file, in binding order.
    pub training: Vec<(String, StoredInput)>,
    /// Each external table, by name.
    pub tables: BTreeMap<String, StoredInput>,
    /// Each stream's held-out rows without their values, the bytes of
    /// `held_out/<stream>.tsv`; empty when nothing is held out.
    pub held_out: BTreeMap<String, Vec<u8>>,
    /// Which rows trained, from `partition.json`.
    pub partition: TrainingPartitionRecord,
}

impl StoredFit {
    /// The problem with each input written to `dir` and its `[data]` naming
    /// those copies, for the data loader, which reads a file by its path. The
    /// copies are the checked bytes of [`Self::training`] and [`Self::tables`].
    pub fn bind_in(&self, dir: &Path) -> Result<Problem, String> {
        let write = |input: &StoredInput| -> Result<String, String> {
            let path = dir.join(&input.name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            std::fs::write(&path, &input.bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            Ok(path.to_string_lossy().into_owned())
        };
        let mut problem = self.config.problem.clone();
        let data = problem.data.as_mut().ok_or("internal error: a stored problem with no [data]")?;
        data.observations = self
            .training
            .iter()
            .map(|(source, input)| Ok((source.clone(), write(input)?)))
            .collect::<Result<_, String>>()?;
        data.tables =
            self.tables.iter().map(|(name, input)| Ok((name.clone(), write(input)?))).collect::<Result<_, String>>()?;
        Ok(problem)
    }
}

use crate::camdl_store::read_stored;

/// Reads the fit `handle` names: `@label`, an id prefix or a result folder.
/// A `FitRun` handle gives the run and its problem; a `FitProblem` handle
/// gives the problem alone. A label that names several runs is refused with
/// them listed. Every file is read through the store, which checks it
/// against its digest.
pub(crate) fn read_fit(store: &Store, handle: &Handle) -> Result<StoredFit, String> {
    let run = match store.resolve::<FitRun>(&RunRef::new(handle.clone())) {
        Ok(r) => Some(r),
        Err(store::Error::WrongKind { found, .. }) if found == FitProblem::KIND => None,
        Err(e) => return Err(e.to_string()),
    };
    let problem = match &run {
        Some(r) => {
            let id = r.identity().chain.problem.id();
            store
                .get(id)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("{handle}: the run's problem {} is not stored", id.key().to_hex()))?
        }
        None => store.resolve::<FitProblem>(&RunRef::new(handle.clone())).map_err(|e| e.to_string())?,
    };
    let text = String::from_utf8(read_stored(&problem, PROBLEM_FILE)?).map_err(|e| format!("{PROBLEM_FILE}: {e}"))?;
    let mut config = FitConfig::from_toml_str_named(&text, PROBLEM_FILE)?;
    let partition: TrainingPartitionRecord = serde_json::from_slice(&read_stored(&problem, PARTITION_FILE)?)
        .map_err(|e| format!("{PARTITION_FILE}: {e}"))?;
    let record = problem.identity();
    let input = |name: String, ext: &str| -> Result<StoredInput, String> {
        let bytes = read_stored(&problem, &name)?;
        Ok(StoredInput { name, ext: ext.to_string(), bytes })
    };
    let training = record
        .training
        .iter()
        .map(|(source, f)| Ok((source.clone(), input(training_file(source, &f.ext), &f.ext)?)))
        .collect::<Result<Vec<_>, String>>()?;
    let tables = record
        .tables
        .iter()
        .map(|(name, f)| Ok((name.clone(), input(table_file(name, &f.ext), &f.ext)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let held_out = record
        .held_out
        .keys()
        .map(|stream| Ok((stream.clone(), read_stored(&problem, &held_out_file(stream))?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let model = record.model.clone();
    if let Some(r) = &run {
        let text = String::from_utf8(read_stored(r, METHOD_FILE)?).map_err(|e| format!("{METHOD_FILE}: {e}"))?;
        let MethodToml { method, estimate } = toml::from_str(&text).map_err(|e| format!("{METHOD_FILE}: {e}"))?;
        for (name, p) in estimate {
            let spec = config
                .problem
                .estimate
                .get_mut(&name)
                .ok_or_else(|| format!("{METHOD_FILE}: `{name}` is not estimated in the run's problem"))?;
            let PerturbationEntry { rw_sd, perturb_only_at_t0, start } = p;
            spec.rw_sd = rw_sd;
            spec.perturb_only_at_t0 = perturb_only_at_t0;
            spec.start = start;
        }
        config.inference.method = Some(method);
    }
    Ok(StoredFit { problem, run, model, config, training, tables, held_out, partition })
}

#[cfg(test)]
mod tests;
