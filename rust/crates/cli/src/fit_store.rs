//! A fit on the run store: the fit problem, the runs on it, and the
//! predictions made from a run (`docs/dev/proposals/2026-11-20-run-store-api.md`,
//! "Worked example: `camdl fit run`" and "Predictions").
//!
//! A [`FitProblem`] is written once and shared by every run on it. It holds
//! what a fit is fitted to and how the problem is posed: the training rows of
//! each bound data file, the dates of the held-out rows (never their values),
//! the external tables, the estimated set with its bounds and priors, the
//! fixed values, the settings, the scenario selection and `ic_free`. A
//! [`FitRun`] sits inside its problem at `runs/`, grouped by its
//! [`MethodIdentity`] and named by the seed typed; its identity holds the
//! method with each estimated parameter's resolved step, the seed, where its
//! chains start (with `[estimate].start`) and its length. A [`Prediction`] from a
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
use store::{Artifact, Content, ContentArtifact, Dep, Finite, Handle, Hashed, Key, Kinds, Layout, Placed, Primary, Run, RunRef, Store, Stored, WriteMode};

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
    pub(crate) fn bytes(&self) -> Vec<u8> {
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
    /// For a problem whose data a `[synthetic]` table generated: what
    /// generated it. `None` for a problem posed on data files.
    pub synthetic: Option<SyntheticOrigin>,
}

/// How a `[synthetic]` table generated a problem's data: the true parameters
/// file's bytes, the dataset's simulation seed, and the table's other
/// settings (its scenario and backend) as canonical JSON. The truth sets the
/// data, and the readers that judge an estimate against it read it here.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_synthetic", version = 1)]
pub(crate) struct SyntheticOrigin {
    pub truth: InputFile,
    pub sim_seed: u64,
    pub generation: String,
}

/// The stored copy of a synthetic problem's true parameters file.
pub(crate) const SYNTHETIC_TRUTH_FILE: &str = "synthetic/truth.toml";

/// The model file's stem and, in a sweep, the point's values as typed.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct ProblemAttributes {
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
    type Attributes = ProblemAttributes;
}

impl Primary for FitProblem {
    fn layout(_id: &FitProblemIdentity, labels: &ProblemAttributes) -> Placed {
        Layout::in_kind().result(&labels.model)
    }
}

// ── The fit run ──────────────────────────────────────────────────────────────

mod method;
pub(crate) use method::{MethodIdentity, PerturbationIdentity};

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

/// Where a run's chains start: the rule, what it reads, and the starts
/// `[estimate]` writes.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_starts", version = 2)]
pub(crate) struct StartSource {
    /// The rule's identity value (`ChainStarts::identity_value`), as
    /// canonical JSON.
    pub rule: String,
    pub input: StartInput,
    /// `[estimate].start`, by name: the base point the rule draws around.
    /// It sits in `[estimate]` in the file and changes only where a run
    /// begins, so two runs that differ only in it share a problem.
    pub written: BTreeMap<String, Finite>,
}

/// Everything about a run except how long it is. A resume must match this
/// exactly.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_chain", version = 2)]
pub(crate) struct ChainIdentity {
    pub problem: Dep<FitProblem>,
    pub method: MethodIdentity,
    /// The compile with the state Jacobian; only `nl-lbfgs` on the ODE backend.
    pub jacobian_model: Option<Dep<CompiledModel>>,
    pub seed: ProcessSeed,
    pub starts: StartSource,
}

/// One run on a problem.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "fit_run", version = 3)]
pub(crate) struct FitRunIdentity {
    pub chain: ChainIdentity,
    /// Sweeps or iterations, for a method that can be extended; 0 otherwise.
    pub length: u64,
    /// The base run's carried chain files, for a resumed run.
    pub resumed_from: Option<Dep<FitRun>>,
}

/// The method's name and the seed the user typed, and the compile's
/// quantities part the invocation that named the run compiled. Labels are
/// recorded per invocation, so an invocation that reuses a run with edited
/// quantities records its own: the quantities do not change the run, and a
/// prediction evaluates the ones the command that names the run compiled.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct RunAttributes {
    pub method: String,
    pub seed: u64,
    /// The run id (hex) of the `Quantities` content the invocation stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantities: Option<String>,
}

/// One fit run, written in place.
pub(crate) struct FitRun;

impl Artifact for FitRun {
    const KIND: &'static str = "fit_run";
    const WRITE: WriteMode = WriteMode::InPlace;
    type Identity = FitRunIdentity;
    type Attributes = RunAttributes;
}

impl Primary for FitRun {
    fn layout(id: &FitRunIdentity, labels: &RunAttributes) -> Placed {
        Layout::inside(id.chain.problem.id(), "runs")
            .group(&labels.method, &id.chain.method)
            .result(&format!("seed_{}", labels.seed))
    }
}

impl StartSource {
    /// The rule `method` starts its chains under, reading `input`, around
    /// the starts `estimate` writes. A NaN or infinite value in the method
    /// or a start is refused.
    pub fn of(
        method: &Method,
        estimate: &IndexMap<String, EstimateSpecV2>,
        input: StartInput,
    ) -> Result<StartSource, String> {
        crate::canonical_json::ensure_finite(method)?;
        let rule = method.starts()?.identity_value();
        let written = estimate
            .iter()
            .filter_map(|(name, spec)| spec.start.map(|s| (name, s)))
            .map(|(name, s)| Ok((name.clone(), finite(s, &format!("[estimate.{name}] start"))?)))
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        Ok(StartSource { rule: canonical_json(&rule)?, input, written })
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
    pub const fn label(self) -> &'static str {
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
    /// Every option that changes what is written — the reach, the draws, the
    /// seed, the streams, the scenario and sweep arms, the chain selection,
    /// the particle count, a forecast's targets and design mode — as canonical JSON of the options
    /// `camdl predict` resolved. A new option is in it unless it is named as
    /// left out there.
    pub settings: String,
    /// The design files a forecast reads its targets' design values from
    /// (`--design SOURCE=FILE`), by stream source; empty without one.
    pub design: BTreeMap<String, Dep<crate::camdl_store::DataTable>>,
}

/// The numerics a fixed-point prediction runs under: what a posterior
/// prediction reads from its fit problem's settings. The ODE integrator is
/// not here: `--integrator` is written into the compiled model before it is
/// keyed, as `simulate` writes it.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "predict_numerics", version = 1)]
pub(crate) struct Numerics {
    pub backend: crate::sim_store::Backend,
    /// The step, by the one dt rule (`crate::dt_rule`).
    pub dt: Finite,
    /// The chain-binomial draws' sampler; BTPE on the other backends.
    pub binomial: crate::pfilter_store::Binomial,
    /// A collapsing rate is 0 instead of an error.
    pub allow_degenerate_rates: bool,
}

/// Where a prediction's parameters come from.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "parameter_source", version = 1)]
pub(crate) enum ParameterSource {
    /// A fit run's draws, and the problem they were fitted on.
    Posterior { draws: Dep<FitRun>, problem: Dep<FitProblem> },
    /// One point given on the command line, repeated once per draw: the
    /// bands carry process and observation noise and no parameter
    /// uncertainty.
    FixedPoint {
        /// The compile rewritten from the data (anchored times, bound
        /// tables, `--integrator`).
        model: Dep<CompiledModel>,
        /// Every parameter's value: the model's, then each `--params` file,
        /// then each `--param`.
        params: ResolvedParams,
        numerics: Numerics,
        /// `--holdout-after` in model time: the rows the point was fitted to
        /// end here. `None` when every bound row is conditioned on.
        trained_through: Option<Finite>,
    },
}

/// One prediction.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "prediction", version = 3)]
pub(crate) struct PredictIdentity {
    pub source: ParameterSource,
    /// The data the filter conditions on, by stream; never held-out outcomes.
    pub data: BTreeMap<String, Dep<DataTable>>,
    /// The model's quantities and contrasts the prediction evaluates: the
    /// part of the compile the fit's problem names in its lineage. Two models
    /// that differ only here pose one fit problem and make two predictions.
    pub quantities: Option<Dep<crate::camdl_store::Quantities>>,
    pub options: PredictOptions,
}

/// The mode, the scenario names as typed, and the model file's stem.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct PredictionAttributes {
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
    type Attributes = PredictionAttributes;
}

impl Primary for Prediction {
    fn layout(id: &PredictIdentity, labels: &PredictionAttributes) -> Placed {
        match &id.source {
            ParameterSource::Posterior { draws, .. } => Layout::inside(draws.id(), "predict"),
            ParameterSource::FixedPoint { model, .. } => Layout::in_kind().group(&labels.model, model),
        }
        .result(&labels.mode)
    }
}

// ── Scores ───────────────────────────────────────────────────────────────────

/// The settings of a score that its rules do not fix: the levels of the two
/// central intervals whose coverage it reports, and the seed of the uniform
/// draws that randomize the PIT within a count's jump. Which rules are
/// computed is fixed by the score's identity version.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "score_settings", version = 1)]
pub(crate) struct ScoreSettings {
    /// The central interval levels, 0.5 and 0.9.
    pub intervals: (Finite, Finite),
    pub pit_seed: u64,
}

/// One score: a stored prediction scored against the outcomes of the rows it
/// predicted. The outcomes enter here and nowhere in the prediction, so a
/// prediction scored with two sets of outcomes is one prediction with two
/// scores.
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "score", version = 1)]
pub(crate) struct ScoreIdentity {
    /// The prediction scored; its identity holds the parameter source, the
    /// mode, the reach and everything the prediction read.
    pub prediction: Dep<Prediction>,
    /// Per stream source, the outcome rows scored: the lines of the bound
    /// file after the prediction's conditioning rows, as a stored data table.
    pub outcomes: BTreeMap<String, Dep<DataTable>>,
    pub settings: ScoreSettings,
}

/// One score, written once, inside the prediction it scores.
pub(crate) struct Score;

impl Artifact for Score {
    const KIND: &'static str = "score";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = ScoreIdentity;
    type Attributes = ();
}

impl Primary for Score {
    fn layout(id: &ScoreIdentity, _labels: &()) -> Placed {
        Layout::inside(id.prediction.id(), "scores").result("score")
    }
}

/// Adds every kind of this module to `kinds`.
pub(crate) fn register(kinds: Kinds) -> Kinds {
    kinds.register::<FitProblem>().register::<FitRun>().register::<Prediction>().register::<Score>()
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
    /// The model `identity.model` names, as compiled.
    model: ir::Model,
    source: ModelSourceFile,
    partition: TrainingPartitionRecord,
    /// A real-data fit's `[synthetic]` table (`fit recovery`'s truth), kept
    /// as provenance files outside the identity.
    recovery: Option<RecoveryTruth>,
}

impl ResolvedProblem {
    /// Which rows trained, as `partition.json` records it.
    pub(crate) fn partition(&self) -> &TrainingPartitionRecord {
        &self.partition
    }
}

/// Where a problem's observation data came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DataOrigin {
    /// Files the fit file names. A `[synthetic]` table beside them is the
    /// truth `fit recovery` compares estimates with, not the data's origin.
    Files,
    /// One dataset the fit file's `[synthetic]` table generated, at one seed.
    Generated,
}

/// A real-data fit's `[synthetic]` table, as the problem's provenance: the
/// table in `fit.toml` syntax, naming the stored truth file, and that file's
/// bytes.
struct RecoveryTruth {
    table: String,
    truth: Vec<u8>,
}

/// A real-data problem's `[synthetic]` table, a provenance file.
pub(crate) const RECOVERY_TABLE_FILE: &str = "synthetic.toml";

/// The stored problem's files.
pub(crate) const PROBLEM_FILE: &str = "problem.toml";
pub(crate) const PARTITION_FILE: &str = "partition.json";
/// The run's merged `[method]`, with its resolved step tables.
pub(crate) const METHOD_FILE: &str = "method.toml";

pub(crate) fn training_file(source: &str, ext: &str) -> String {
    format!("inputs/{source}.{ext}")
}

pub(crate) fn table_file(name: &str, ext: &str) -> String {
    format!("inputs/tables/{name}.{ext}")
}

pub(crate) fn held_out_file(stream: &str) -> String {
    format!("held_out/{stream}.tsv")
}

fn extension(path: &str) -> String {
    Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("tsv").to_string()
}

fn finite(x: f64, what: &str) -> Result<Finite, String> {
    Finite::new(x).map_err(|e| format!("{what}: {e}"))
}

fn canonical_json(v: &serde_json::Value) -> Result<String, String> {
    serde_json::to_string(&crate::canonical_json::canonical(v)).map_err(|e| e.to_string())
}

#[expect(clippy::disallowed_methods, reason = "reads a data or table file the fit config names, for `resolve_problem`, which hashes the bytes into the problem's identity")]
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
#[expect(clippy::disallowed_methods, reason = "reads the `[synthetic]` truth file the fit config names: its bytes enter `SyntheticOrigin` in the identity for a generated dataset, and are kept as provenance outside the identity for a real-data fit, whose result they do not change")]
pub(crate) fn resolve_problem(
    problem: &Problem,
    model: &ir::Model,
    model_dep: Dep<CompiledModel>,
    source: ModelSourceFile,
    origin: DataOrigin,
) -> Result<ResolvedProblem, String> {
    let Problem {
        model: _,
        data,
        synthetic,
        simplex_groups,
        store: _,
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
        compiled_digest: _,
    } = problem;
    if model_dep.digest() != crate::camdl_store::model_part_digest(model) {
        return Err("internal error: the model given is not the stored compile the problem names".into());
    }
    let mut recovery = None;
    let synthetic = match (synthetic, origin) {
        (None, _) => None,
        (Some(spec), DataOrigin::Files) => {
            let bytes = std::fs::read(&spec.true_params)
                .map_err(|e| format!("cannot read [synthetic] true_params {}: {e}", spec.true_params))?;
            let mut stored = spec.clone();
            stored.true_params = SYNTHETIC_TRUTH_FILE.into();
            let mut doc = toml::Table::new();
            doc.insert("synthetic".into(), toml::Value::try_from(&stored).map_err(|e| e.to_string())?);
            recovery = Some(RecoveryTruth { table: toml::to_string(&doc).map_err(|e| e.to_string())?, truth: bytes });
            None
        }
        (Some(spec), DataOrigin::Generated) => {
            let seeds = spec.sim_seeds.to_vec()?;
            let [sim_seed] = seeds[..] else {
                return Err("internal error: a synthetic problem is one dataset, generated at one seed".into());
            };
            let bytes = std::fs::read(&spec.true_params)
                .map_err(|e| format!("cannot read [synthetic] true_params {}: {e}", spec.true_params))?;
            let content = Hashed::new(<Table as store::Content>::from_bytes(&bytes).map_err(|e| e.to_string())?);
            let mut generation = serde_json::to_value(spec).map_err(|e| e.to_string())?;
            if let Some(o) = generation.as_object_mut() {
                for k in ["true_params", "sim_seeds", "datasets"] {
                    o.remove(k);
                }
                // An unset setting is left out, as `problem.toml` writes it.
                o.retain(|_, v| !v.is_null());
            }
            Some(SyntheticOrigin {
                truth: InputFile { ext: "toml".into(), content },
                sim_seed,
                generation: crate::canonical_json::canonical(&generation).to_string(),
            })
        }
    };
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
        if !Path::new(path).is_file() {
            return Err(format!("cannot read data file {path}, bound to `{source}`: it does not exist"));
        }
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
    let estimate = problem_estimate(estimate)?;
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
        synthetic,
    };
    Ok(ResolvedProblem { identity, model: model.clone(), source, partition: record, recovery })
}

/// The problem's half of `[estimate]`: each parameter's bounds and prior.
/// Its `start` is the run's ([`StartSource`]).
pub(crate) fn problem_estimate(estimate: &IndexMap<String, EstimateSpecV2>) -> Result<Estimate, String> {
    let mut problem = Vec::new();
    for (name, spec) in estimate {
        let EstimateSpecV2 { bounds, prior, start: _ } = spec;
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
    }
    Ok(Estimate { parameters: problem })
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
    let names: Vec<&str> = wanted.iter().map(|(leaf, _)| *leaf).collect();
    let kept = cut_lines(bytes, ext, source, &names, &|i, seen| seen < wanted[i].1 .0, load)?;
    let dir = tempfile::tempdir().map_err(|e| format!("cannot make a temporary directory: {e}"))?;
    let path = dir.path().join(format!("cut.{ext}"));
    let path_text = path.to_string_lossy().into_owned();
    std::fs::write(&path, &kept).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    let bound: Vec<Vec<ObservedRow>> = wanted.iter().map(|(leaf, _)| load(&path_text, leaf)).collect::<Result<_, _>>()?;
    for (g, (leaf, (n, digest))) in bound.iter().zip(&wanted) {
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

/// `bytes`, a data file read by the streams `leaves`, with its header, every
/// line that gives no stream a row, and each line that gives stream `i` its
/// `seen`-th row (counted from 0) when `keep(i, seen)` holds.
///
/// Which stream a line gives a row to is read by the loader itself: each data
/// line is bound alone, under the lines that precede the first data line, and
/// the rows each stream reads from it are counted.
fn cut_lines(
    bytes: &[u8],
    ext: &str,
    source: &str,
    leaves: &[&str],
    keep: &dyn Fn(usize, usize) -> bool,
    load: &dyn Fn(&str, &str) -> Result<Vec<ObservedRow>, String>,
) -> Result<Vec<u8>, String> {
    let dir = tempfile::tempdir().map_err(|e| format!("cannot make a temporary directory: {e}"))?;
    let path = dir.path().join(format!("cut.{ext}"));
    let path_text = path.to_string_lossy().into_owned();
    let bind = |content: &[u8]| -> Result<Vec<Vec<ObservedRow>>, String> {
        std::fs::write(&path, content).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        leaves.iter().map(|leaf| load(&path_text, leaf)).collect()
    };
    let lines = lines_of(bytes);
    // The lines before the first data line: the fewest leading lines the
    // loader reads as a file, which give no stream a row.
    let head = (0..lines.len())
        .find(|&h| bind(&lines[..=h].concat()).is_ok_and(|got| got.iter().all(Vec::is_empty)))
        .ok_or_else(|| format!("data file '{source}' has no header the loader reads"))?;
    let mut kept: Vec<u8> = lines[..=head].concat();
    let mut seen = vec![0usize; leaves.len()];
    for line in &lines[head + 1..] {
        let mut probe = lines[..=head].concat();
        probe.extend_from_slice(line);
        let got = bind(&probe)?;
        let mut keep_line = got.iter().all(Vec::is_empty);
        for (i, g) in got.iter().enumerate() {
            if g.len() > 1 {
                return Err(format!("internal error: one line of data file '{source}' gave {} rows of one stream", g.len()));
            }
            if g.len() == 1 {
                keep_line |= keep(i, seen[i]);
                seen[i] += 1;
            }
        }
        if keep_line {
            kept.extend_from_slice(line);
        }
    }
    Ok(kept)
}

/// `bytes`, the outcome file of the stream source `source` read by `leaves`,
/// with its header and the lines after each leaf's first `conditioning[leaf]`
/// rows: the outcomes a score reads beyond what its prediction conditioned on.
/// `load(path, leaf)` reads one leaf's rows of a file.
pub(crate) fn lines_after_conditioning(
    bytes: &[u8],
    ext: &str,
    source: &str,
    leaves: &[&str],
    conditioning: &BTreeMap<String, usize>,
    load: &dyn Fn(&str, &str) -> Result<Vec<ObservedRow>, String>,
) -> Result<Vec<u8>, String> {
    let n: Vec<usize> = leaves.iter().map(|l| conditioning.get(*l).copied().unwrap_or(0)).collect();
    cut_lines(bytes, ext, source, leaves, &|i, seen| seen >= n[i], load)
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
        synthetic,
    } = id;
    let mut doc = toml::Table::new();
    if let Some(SyntheticOrigin { truth: _, sim_seed, generation }) = synthetic {
        let mut t: toml::Table = serde_json::from_str::<toml::Table>(generation).map_err(|e| e.to_string())?;
        t.insert("true_params".into(), Value::String(SYNTHETIC_TRUTH_FILE.into()));
        t.insert("sim_seeds".into(), Value::Array(vec![Value::Integer(*sim_seed as i64)]));
        doc.insert("synthetic".into(), Value::Table(t));
    }
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
    use crate::result_files::fit::ProblemFile;
    let id = &problem.identity;
    let text = problem_toml(id, &problem.model, &problem.source)?;
    FitConfig::from_toml_str_named(&text, PROBLEM_FILE)
        .map_err(|e| format!("internal error: the problem.toml this fit would store does not parse back: {e}"))?;
    let partition = serde_json::to_vec_pretty(&problem.partition).map_err(|e| e.to_string())?;
    for file in problem_files(problem) {
        let bytes: Vec<u8> = match &file {
            ProblemFile::Problem => text.as_bytes().to_vec(),
            ProblemFile::Partition => partition.clone(),
            ProblemFile::Training { source, .. } => {
                id.training.iter().find(|(s, _)| s == source).expect("a declared training file").1.bytes()
            }
            ProblemFile::Table { name, .. } => id.tables[name].bytes(),
            ProblemFile::HeldOut { stream } => id.held_out[stream].get().to_bytes(),
            ProblemFile::SyntheticTruth => id.synthetic.as_ref().expect("a declared synthetic truth").truth.bytes(),
            ProblemFile::RecoveryTruth => problem.recovery.as_ref().expect("a declared recovery truth").truth.clone(),
            ProblemFile::RecoveryTable => problem.recovery.as_ref().expect("a declared recovery table").table.as_bytes().to_vec(),
            ProblemFile::ModelSource { .. } => problem.source.bytes.clone(),
        };
        let e = |e: store::Error| e.to_string();
        crate::result_files::create(run, &file).map_err(e)?.write_all(&bytes).map_err(e)?;
    }
    Ok(())
}

/// The files a problem's run stores: `problem.toml`, `partition.json`, each
/// training file, each table, each stream's held-out dates, a generated
/// dataset's truth, a real-data fit's `[synthetic]` table and its truth
/// (provenance), and the model file (provenance), in the order
/// [`write_problem`] writes them.
pub(crate) fn problem_files(problem: &ResolvedProblem) -> Vec<crate::result_files::fit::ProblemFile> {
    use crate::result_files::fit::ProblemFile;
    let id = &problem.identity;
    let mut v = vec![ProblemFile::Problem, ProblemFile::Partition];
    v.extend(id.training.iter().map(|(source, f)| ProblemFile::Training { source: source.clone(), ext: f.ext.clone() }));
    v.extend(id.tables.iter().map(|(name, f)| ProblemFile::Table { name: name.clone(), ext: f.ext.clone() }));
    v.extend(id.held_out.keys().map(|stream| ProblemFile::HeldOut { stream: stream.clone() }));
    if id.synthetic.is_some() {
        v.push(ProblemFile::SyntheticTruth);
    }
    if problem.recovery.is_some() {
        v.push(ProblemFile::RecoveryTruth);
        v.push(ProblemFile::RecoveryTable);
    }
    v.push(ProblemFile::ModelSource { camdl: problem.source.camdl });
    v
}

/// One estimated parameter's start, as `method.toml` writes it.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StartEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    start: Option<f64>,
}

/// What `method.toml` holds.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodToml {
    method: Method,
    #[serde(default)]
    estimate: BTreeMap<String, StartEntry>,
}

/// `method` with its step table replaced by the resolved steps of `id`, every
/// estimated parameter's value in z: `[method.perturbation]` under IF2,
/// `[method.proposal]` under PMMH and MH.
fn with_resolved_steps(method: &Method, id: &MethodIdentity) -> Result<Method, String> {
    use config_v2::{Algorithm as A, PerturbationSpec};
    use sim::inference::types::ZSd;
    let z = |f: Finite| ZSd::new(f.get()).map_err(|e| format!("internal error: a stored step {e}"));
    let mut out = method.clone();
    match (&mut out.algorithm, id) {
        (A::IF2 { perturbation, .. }, MethodIdentity::If2(i)) => {
            let mut table = IndexMap::new();
            for (name, p) in &i.perturbation {
                let initial_state = p.at == method::PerturbAtIdentity::TimeZeroOnly;
                table.insert(name.clone(), PerturbationSpec { sd: Some(z(p.sd)?), initial_state });
            }
            for (name, sd) in i.simplex.iter().flatten() {
                table.insert(name.clone(), PerturbationSpec { sd: Some(z(*sd)?), initial_state: false });
            }
            table.sort_keys();
            *perturbation = table;
        }
        (A::PMMH { proposal, .. }, MethodIdentity::Pmmh(method::PmmhIdentity { proposal: ids, .. }))
        | (A::Mh { proposal, .. }, MethodIdentity::Mh(method::MhIdentity { proposal: ids, .. })) => {
            *proposal = ids.iter().map(|(n, sd)| Ok((n.clone(), z(*sd)?))).collect::<Result<_, String>>()?;
        }
        _ => {}
    }
    Ok(out)
}

/// The text of `method.toml`, a function of the run's identity: `method`,
/// after the flags were applied, with every estimated parameter's resolved
/// step in its table, and `[estimate]` with the starts the file writes. A
/// sourced `starts` is written as `method` holds it, so the caller names its
/// source as the run stores it.
fn method_toml(method: &Method, chain: &ChainIdentity) -> Result<String, String> {
    let method = &with_resolved_steps(method, &chain.method)?;
    let estimate =
        chain.starts.written.iter().map(|(name, s)| (name.clone(), StartEntry { start: Some(s.get()) })).collect();
    let doc = MethodToml { method: method.clone(), estimate };
    let text = toml::to_string(&doc).map_err(|e| format!("cannot write method.toml: {e}"))?;
    let back: MethodToml =
        toml::from_str(&text).map_err(|e| format!("internal error: method.toml does not parse back: {e}"))?;
    let json = |m: &Method| serde_json::to_value(m).map_err(|e| e.to_string());
    if json(&back.method)? != json(method)? {
        return Err("internal error: method.toml reads back to another method".into());
    }
    Ok(format!("# method.toml: this run's [method], with every parameter's resolved step, and the starts of [estimate].\n{text}"))
}

/// Reads `method.toml`. A run stored before step sizes moved to `[method]`
/// writes `rw_sd` and `perturb_only_at_t0` in its `[estimate]`; they are set
/// aside with one notice, since the run's identity was computed when it
/// ran and reading it computes nothing.
fn read_method_toml(text: &str) -> Result<MethodToml, String> {
    let mut raw: toml::Value = toml::from_str(text).map_err(|e| format!("{METHOD_FILE}: {e}"))?;
    if let Some(notice) = config_v2::set_aside_legacy_step_keys(&mut raw, &format!("this run's {METHOD_FILE}")) {
        crate::out::detached().say(format_args!("{notice}"));
    }
    raw.try_into().map_err(|e| format!("{METHOD_FILE}: {e}"))
}

/// Write a claimed run's `method.toml`.
pub(crate) fn write_method(run: &mut Run<FitRun>, method: &Method) -> Result<(), String> {
    let text = method_toml(method, &run.identity().chain)?;
    let e = |e: store::Error| e.to_string();
    run.result_file(METHOD_FILE).map_err(e)?.write_all(text.as_bytes()).map_err(e)
}

// ── Reading a stored fit ─────────────────────────────────────────────────────

/// One stored input file of a problem, read through the store.
pub(crate) struct StoredInput {
    /// The file's name in the problem, which `config` names it by.
    pub name: String,
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
    /// A synthetic problem's true parameters file.
    pub synthetic_truth: Option<StoredInput>,
}

impl StoredFit {
    /// Each stream that held rows out of training, with how many: the rows of
    /// its `held_out/<stream>.tsv`. Empty when the problem holds none out.
    pub fn held_out_rows(&self) -> Vec<(String, usize)> {
        self.held_out
            .iter()
            .map(|(stream, bytes)| (stream.clone(), bytes.split(|b| *b == b'\n').skip(1).filter(|l| !l.is_empty()).count()))
            .filter(|(_, n)| *n > 0)
            .collect()
    }

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
        if let (Some(spec), Some(truth)) = (problem.synthetic.as_mut(), &self.synthetic_truth) {
            spec.true_params = write(truth)?;
        }
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
            match store.get(id).map_err(|e| e.to_string())? {
                Some(p) => p,
                None => {
                    let why = match store.inspect(id) {
                        Ok(Some(e)) => format!("is in {}: {}", crate::fit::report::display_path(e.folder()), entry_state(e.state())),
                        _ => "is not stored".to_string(),
                    };
                    return Err(format!("{handle}: the run's problem {} {why}", id.key().short12()));
                }
            }
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
        let _ = ext;
        Ok(StoredInput { name, bytes })
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
    let provenance = |name: &str| -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut problem.open_provenance(name).map_err(|e| e.to_string())?, &mut bytes)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    };
    let synthetic_truth = match &record.synthetic {
        Some(_) => Some(input(SYNTHETIC_TRUTH_FILE.to_string(), "toml")?),
        // A real-data fit's recovery truth, kept as provenance.
        None if problem.provenance_files().contains(&RECOVERY_TABLE_FILE) => {
            let table = String::from_utf8(provenance(RECOVERY_TABLE_FILE)?).map_err(|e| e.to_string())?;
            let doc: toml::Table = table.parse().map_err(|e| format!("{RECOVERY_TABLE_FILE}: {e}"))?;
            let spec: config_v2::SyntheticSpec = doc
                .get("synthetic")
                .cloned()
                .ok_or_else(|| format!("{RECOVERY_TABLE_FILE}: no [synthetic] table"))?
                .try_into()
                .map_err(|e| format!("{RECOVERY_TABLE_FILE}: {e}"))?;
            config.problem.synthetic = Some(spec);
            Some(StoredInput { name: SYNTHETIC_TRUTH_FILE.to_string(), bytes: provenance(SYNTHETIC_TRUTH_FILE)? })
        }
        None => None,
    };
    let model = record.model.clone();
    if let Some(r) = &run {
        let text = String::from_utf8(read_stored(r, METHOD_FILE)?).map_err(|e| format!("{METHOD_FILE}: {e}"))?;
        let MethodToml { method, estimate } = read_method_toml(&text)?;
        for (name, p) in estimate {
            let spec = config
                .problem
                .estimate
                .get_mut(&name)
                .ok_or_else(|| format!("{METHOD_FILE}: `{name}` is not estimated in the run's problem"))?;
            let StartEntry { start } = p;
            spec.start = start;
        }
        config.inference.method = Some(method);
    }
    Ok(StoredFit { problem, run, model, config, training, tables, held_out, partition, synthetic_truth })
}

// ── A stored run's files, for readers that read by path ─────────────────────

/// What a run was asked to do and what it found, as the method's readers
/// read it (`method_result`, `fit summary`): the method, the seed, the
/// header the run printed, the starts and their source, and the best
/// log-likelihood and chain. A provenance file: it names the fit file as
/// typed.
pub(crate) const RUN_INPUTS_FILE: &str = "run_inputs.json";

/// The fit's provenance as the run resolved it (`run_meta::FitSidecar`):
/// where each prior came from, every parameter's value and source, the
/// partition into estimated and fixed. A provenance file.
pub(crate) const FIT_META_FILE: &str = "fit.meta.json";

/// The fit file the run was given, as read.
pub(crate) const FIT_TOML_FILE: &str = "fit.toml";

/// A stored run's result and provenance files, each read through the store
/// and checked against its digest, copied into a scratch directory that is
/// removed when this is dropped. For the readers of a run's files (the method
/// result, `fit summary`, `compare`, a sourced start), which read them by
/// path; the scratch copy is never a path into the store.
pub(crate) struct RunFiles {
    dir: tempfile::TempDir,
}

/// Each copy of a stored fit's files, with the path in the store it stands
/// for, so a reader that shows a path shows the store and never the copy. A
/// file's entry comes before its directory's, and the first match is used.
static SHOWN: std::sync::Mutex<Vec<(std::path::PathBuf, std::path::PathBuf)>> = std::sync::Mutex::new(Vec::new());

/// `p`, a path inside a copy of a stored run's files, as the same path in the
/// run's folder; any other path unchanged.
pub(crate) fn shown(p: &Path) -> std::path::PathBuf {
    let copies = SHOWN.lock().unwrap_or_else(|e| e.into_inner());
    for (copy, folder) in copies.iter() {
        if let Ok(rest) = p.strip_prefix(copy) {
            return folder.join(rest);
        }
    }
    p.to_path_buf()
}

/// Shows `copy` as `stored`, ahead of every mapping already registered.
fn show_as(copy: &Path, stored: &Path) {
    SHOWN.lock().unwrap_or_else(|e| e.into_inner()).insert(0, (copy.to_path_buf(), stored.to_path_buf()));
}

impl RunFiles {
    pub fn of(run: &Stored<FitRun>) -> Result<RunFiles, String> {
        let dir = tempfile::Builder::new()
            .prefix("camdl-run-")
            .tempdir()
            .map_err(|e| format!("cannot make a scratch directory: {e}"))?;
        SHOWN.lock().unwrap_or_else(|e| e.into_inner()).push((dir.path().to_path_buf(), run.folder().to_path_buf()));
        let write = |name: &str, bytes: &[u8]| -> Result<(), String> {
            let path = dir.path().join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
            }
            std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
        };
        for name in run.result_files() {
            write(name, &read_stored(run, name)?)?;
        }
        for name in run.provenance_files() {
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut run.open_provenance(name).map_err(|e| e.to_string())?, &mut bytes)
                .map_err(|e| format!("reading {name}: {e}"))?;
            write(name, &bytes)?;
        }
        Ok(RunFiles { dir })
    }

    /// The scratch directory, laid out as the run's folder is.
    pub fn path(&self) -> &Path {
        self.dir.path()
    }
}

/// The inputs `dir`, a run's folder or a copy of it, records
/// ([`RUN_INPUTS_FILE`]).
#[expect(clippy::disallowed_methods, reason = "reads `run_inputs.json` from the directory its caller passes: a scratch copy of a stored run's files read out through the store (`RunFiles` for `--resume`, an `OpenedFit` run copy), or the folder of the run this process's `fit run` is writing, read back")]
pub(crate) fn read_run_inputs(dir: &Path) -> Result<serde_json::Value, String> {
    let path = dir.join(RUN_INPUTS_FILE);
    let bytes = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// A stored fit opened for a reader that reads it by path: the fit as
/// [`read_fit`] gives it, its problem with `[data]` and `[model]` naming
/// checked copies in a scratch directory, the stored compile as an IR file
/// there, and, when the handle named a run, the run's files copied out. The
/// scratch directory is removed when this is dropped.
pub(crate) struct OpenedFit {
    pub fit: StoredFit,
    /// The stored compile, with the model's name, description and `#'`
    /// documentation, which are not part of it, read back from its
    /// provenance, as an IR file in the scratch directory.
    pub model_ir: std::path::PathBuf,
    /// The run's files, with `model.ir.json` beside them; `None` for a
    /// problem handle.
    pub run: Option<RunFiles>,
    /// The problem's labels, from the invocation that stored it.
    pub labels: Option<ProblemAttributes>,
    /// The directory holding `model.ir.json`: the copy of the run's files,
    /// or for a problem handle the scratch directory.
    pub segment: std::path::PathBuf,
    /// The copy of the run's files; `None` for a problem handle.
    pub leaf: Option<std::path::PathBuf>,
    /// The config the problem (and the run, when one is named) was resolved
    /// to, with every path naming a copy in the scratch directory.
    pub config: FitConfig,
    /// The compile's quantities and contrasts, which the problem names in its
    /// lineage; restored into [`Self::model_ir`].
    pub quantities: Option<Dep<crate::camdl_store::Quantities>>,
    /// The presentation the invocation the handle names recorded for
    /// [`StoredFit::model`] ([`crate::camdl_store::shown_for`]), whose
    /// description and documentation are restored into [`Self::model_ir`].
    pub shown: crate::camdl_store::Shown,
    /// The handle the fit was opened by, as parsed.
    pub handle: Handle,
    _scratch: tempfile::TempDir,
}

impl OpenedFit {
    /// The model file the problem keeps (`model.camdl`, or the IR it was
    /// given), copied out of the store into the scratch directory under its
    /// stored name, for a reader that compiles against the model's source.
    pub fn model_source(&self) -> Result<std::path::PathBuf, String> {
        let problem = &self.fit.problem;
        let name = problem
            .provenance_files()
            .into_iter()
            .find(|f| f.starts_with("model."))
            .ok_or("the stored problem keeps no model file")?
            .to_string();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut problem.open_provenance(&name).map_err(|e| e.to_string())?, &mut bytes)
            .map_err(|e| e.to_string())?;
        let path = self._scratch.path().join("source").join(&name);
        std::fs::create_dir_all(path.parent().expect("a file in the scratch directory")).map_err(|e| e.to_string())?;
        std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        show_as(&path, &problem.folder().join(&name));
        Ok(path)
    }

    /// The directory a reader reads the run from: the copy of the run's
    /// files.
    pub fn run_dir(&self) -> Option<&Path> {
        self.run.as_ref().map(RunFiles::path)
    }

    /// [`Self::config`], cloned.
    pub fn config(&self) -> FitConfig {
        self.config.clone()
    }

    /// The model's name: the stem of the file the problem was posed on.
    pub fn model_name(&self) -> String {
        self.labels.as_ref().map(|l| l.model.clone()).unwrap_or_else(|| "model".to_string())
    }

    /// Records in the invocation `info` that this fit's compiled model was
    /// run from the model file [`Self::shown`] presents, for a command that
    /// stores a result read from this fit: a reader of that result then finds
    /// the presentation through the command's own invocation. Nothing is
    /// recorded when no presentation was.
    pub fn record_presentation(&self, store: &Store, info: &store::RunInfo) -> Result<(), String> {
        record_presentation(store, info, &self.fit.model, &self.shown, &self.model_name())
    }

    /// This fit, reported through another model file, which the caller has
    /// checked compiles to the model the fit ran on apart from its quantities
    /// and contrasts. `quantities`, what that file's compile produced, and
    /// `shown`, its presentation, replace the recorded ones: in
    /// [`Self::quantities`], in [`Self::shown`] (so [`Self::record_presentation`]
    /// records the file) and in the IR files a reader loads.
    pub fn reported_through(
        mut self,
        store: &Store,
        quantities: Dep<crate::camdl_store::Quantities>,
        content: &crate::camdl_store::QuantitiesContent,
        shown: crate::camdl_store::Shown,
    ) -> Result<Self, String> {
        let mut envelope = stored_envelope(store, &self.fit.model, &self.model_name())?;
        shown.restore(&self.handle, &mut envelope);
        envelope.model.quantities = content.quantities()?;
        envelope.model.contrasts = content.contrasts()?;
        let json = serde_json::to_string(&envelope).map_err(|e| e.to_string())?;
        let mut files = vec![self.model_ir.clone()];
        files.extend(self.run.as_ref().map(|r| r.path().join("model.ir.json")));
        for path in files {
            std::fs::write(&path, &json).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        self.quantities = Some(quantities);
        self.shown = shown;
        Ok(self)
    }
}

/// Records in the invocation `info` that the stored compiled model `model`,
/// named `name`, was run from the model file the presentation `shown` names
/// presents. Nothing is recorded when it names none.
pub(crate) fn record_presentation(
    store: &Store,
    info: &store::RunInfo,
    model: &Dep<CompiledModel>,
    shown: &crate::camdl_store::Shown,
    name: &str,
) -> Result<(), String> {
    let Some(hex) = shown.run_id() else { return Ok(()) };
    crate::camdl_store::presentation::link_presentation(store, info, model, crate::camdl_store::Presentation::Stored(hex), name)
}

/// The compiled model a stored result names, as an IR envelope: the model
/// part, named `name`. The description and the documentation are a
/// presentation's, which [`crate::camdl_store::Shown::restore`] restores;
/// without it the envelope has neither.
pub(crate) fn stored_envelope(store: &Store, dep: &Dep<CompiledModel>, name: &str) -> Result<ir::IrEnvelope, String> {
    let stored: Stored<CompiledModel> = store
        .get(dep.id())
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("the compiled model {} is not stored", dep.id().key().short12()))?;
    let content = read_stored(&stored, CompiledModel::FILE)?;
    if Key::digest_bytes(&content) != dep.digest() {
        return Err(format!("the stored compiled model {} is not the one the fit names", dep.id().key().short12()));
    }
    let part = <crate::camdl_store::CompiledModelContent as Content>::from_bytes(&content).map_err(|e| e.to_string())?;
    let mut model = part.model().clone();
    model.name = name.to_string();
    Ok(ir::IrEnvelope::wrap(model, None, declared::DeclaredSection::not_recorded()))
}

/// The quantities part the invocation that `handle` names recorded for
/// `run`: for `@label`, the invocation holding the label; for an id or a
/// folder, the one every invocation that named the run recorded. An id or
/// folder whose invocations recorded different quantities is refused, since
/// which the user means is not said. `None` when no invocation recorded one.
fn run_quantities(
    store: &Store,
    handle: &Handle,
    text: &str,
    run: &Stored<FitRun>,
) -> Result<Option<(Dep<crate::camdl_store::Quantities>, crate::camdl_store::QuantitiesContent)>, String> {
    let recorded = |inv: &store::Invocation| -> Option<String> {
        inv.members::<FitRun>().ok()?.into_iter().find(|(id, _)| id == run.id()).and_then(|(_, l)| l.quantities)
    };
    let hex = match handle {
        Handle::Label { .. } => store.invocation(handle).ok().as_ref().and_then(recorded),
        _ => {
            let mut seen: Vec<(u64, String)> = Vec::new();
            for inv in store.invocations().iter() {
                if let Some(q) = recorded(inv) {
                    seen.push((inv.seq(), q));
                }
            }
            let distinct: std::collections::BTreeSet<&String> = seen.iter().map(|(_, q)| q).collect();
            if distinct.len() > 1 {
                let list: Vec<String> = seen
                    .iter()
                    .map(|(seq, q)| format!("    invocation {seq}: quantities {}", &q[..12.min(q.len())]))
                    .collect();
                return Err(format!(
                    "{text} names a run that invocations compiled with different quantities blocks:\n{}\n  \
                     Name the run by the @label of the `fit run` whose model's quantities you mean \
                     (`fit run --label`), so the prediction evaluates those.",
                    list.join("\n")
                ));
            }
            seen.pop().map(|(_, q)| q)
        }
    };
    let Some(hex) = hex else { return Ok(None) };
    use crate::camdl_store::Quantities;
    let stored: Stored<Quantities> = store
        .resolve(&RunRef::new(Handle::IdPrefix(hex.clone())))
        .map_err(|e| format!("the run's quantities {}: {e}", &hex[..12.min(hex.len())]))?;
    let dep = stored.dep(Quantities::FILE).map_err(|e| e.to_string())?;
    let bytes = read_stored(&stored, Quantities::FILE)?;
    let content = <crate::camdl_store::QuantitiesContent as Content>::from_bytes(&bytes).map_err(|e| e.to_string())?;
    Ok(Some((dep, content)))
}

/// The quantities and contrasts `problem`'s lineage names, read and checked
/// against the digest the lineage recorded; `None` for a problem whose
/// lineage names none.
fn problem_quantities(
    store: &Store,
    problem: &Stored<FitProblem>,
) -> Result<Option<(Dep<crate::camdl_store::Quantities>, crate::camdl_store::QuantitiesContent)>, String> {
    use crate::camdl_store::Quantities;
    let Some((run_id, digest)) = problem.lineage().iter().find_map(|l| match l {
        store::Lineage::Result { kind, run_id, digest, .. } if kind == Quantities::KIND => Some((*run_id, *digest)),
        _ => None,
    }) else {
        return Ok(None);
    };
    let stored: Stored<Quantities> = store
        .resolve(&RunRef::new(Handle::IdPrefix(run_id.to_hex())))
        .map_err(|e| format!("the fit's quantities {}: {e}", run_id.short12()))?;
    let dep = stored.dep(Quantities::FILE).map_err(|e| e.to_string())?;
    if dep.digest() != digest {
        return Err(format!("the stored quantities {} are not the ones the fit names", run_id.short12()));
    }
    let bytes = read_stored(&stored, Quantities::FILE)?;
    let content = <crate::camdl_store::QuantitiesContent as Content>::from_bytes(&bytes).map_err(|e| e.to_string())?;
    Ok(Some((dep, content)))
}

/// The labels the invocation that stored `problem` gave it.
fn problem_labels(store: &Store, problem: &Stored<FitProblem>) -> Option<ProblemAttributes> {
    let inv = store.invocation(&Handle::IdPrefix(problem.id().key().to_hex())).ok()?;
    inv.members::<FitProblem>().ok()?.into_iter().find(|(id, _)| id == problem.id()).map(|(_, l)| l)
}

/// Opens the fit `text` names in `store` for a reader: `@label`, a run-id or
/// problem-id prefix, or a result's folder. A fit file is never a handle:
/// it is refused, with the runs it produced listed, by [`resolve_run`]'s
/// rule.
pub(crate) fn open_fit(store: &Store, text: &str) -> Result<OpenedFit, String> {
    if is_config_file(text) {
        let source = store::SourceBytes::read(text).map_err(|e| e.to_string())?;
        return Err(not_a_run(store, &source));
    }
    let handle = parse_handle(text)?;
    let fit = read_fit(store, &handle).map_err(|e| {
        not_completed(store, text)
            .or_else(|| match &handle {
                Handle::Label { name, back } if *back > 0 => past_the_history(store, text, name),
                _ => None,
            })
            .unwrap_or(e)
    })?;
    say_resolution(store, text, &fit);
    let scratch = tempfile::Builder::new()
        .prefix("camdl-fit-")
        .tempdir()
        .map_err(|e| format!("cannot make a scratch directory: {e}"))?;
    let labels = problem_labels(store, &fit.problem);
    let name = labels.as_ref().map(|l| l.model.clone()).unwrap_or_else(|| "model".to_string());
    let mut envelope = stored_envelope(store, &fit.model, &name)?;
    let result = fit.run.as_ref().map(|r| r.id().key()).unwrap_or_else(|| fit.problem.id().key());
    let shown = crate::camdl_store::shown_for(store, &handle, result, fit.model.id().key());
    shown.restore(&handle, &mut envelope);
    let quantities = match &fit.run {
        Some(run) => match run_quantities(store, &handle, text, run)? {
            Some(q) => Some(q),
            None => problem_quantities(store, &fit.problem)?,
        },
        None => problem_quantities(store, &fit.problem)?,
    };
    if let Some((_, content)) = &quantities {
        envelope.model.quantities = content.quantities()?;
        envelope.model.contrasts = content.contrasts()?;
    }
    let quantities = quantities.map(|(dep, _)| dep);
    let json = serde_json::to_string(&envelope).map_err(|e| e.to_string())?;
    let model_ir = scratch.path().join("model.ir.json");
    std::fs::write(&model_ir, &json).map_err(|e| format!("cannot write {}: {e}", model_ir.display()))?;
    let mut problem = fit.bind_in(&scratch.path().join("problem"))?;
    problem.model.camdl = model_ir.to_string_lossy().into_owned();
    problem.compiled_ir = Some(problem.model.camdl.clone());
    let run = match &fit.run {
        Some(r) => {
            let files = RunFiles::of(r)?;
            std::fs::write(files.path().join("model.ir.json"), &json).map_err(|e| e.to_string())?;
            Some(files)
        }
        None => None,
    };
    // The copies show as the store's files: the problem's inputs as its
    // folder, and the compile as the model file the problem keeps, which a
    // user can run again.
    show_as(&scratch.path().join("problem"), fit.problem.folder());
    if let Some(given) = fit.problem.provenance_files().into_iter().find(|f| f.starts_with("model.")) {
        let stored_model = fit.problem.folder().join(given);
        show_as(&model_ir, &stored_model);
        if let Some(r) = &run {
            show_as(&r.path().join("model.ir.json"), &stored_model);
        }
    }
    let config = FitConfig { problem, inference: fit.config.inference.clone() };
    let leaf = run.as_ref().map(|r| r.path().to_path_buf());
    let segment = leaf.clone().unwrap_or_else(|| scratch.path().to_path_buf());
    Ok(OpenedFit { fit, model_ir, run, labels, segment, leaf, config, quantities, shown, handle, _scratch: scratch })
}

/// The run store a reader of a stored fit opens: the one the command stores
/// in, set by the command before it reads a fit; else `CAMDL_STORE`, else the
/// default.
static READER_ROOT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// Name the run store this command reads stored fits from. The first call
/// wins.
pub(crate) fn set_reader_root(root: std::path::PathBuf) {
    let _ = READER_ROOT.set(root);
}

/// The run store this command reads stored fits from.
pub(crate) fn reader_root() -> std::path::PathBuf {
    READER_ROOT.get().cloned().unwrap_or_else(crate::camdl_store::read_root)
}

/// The root of the run store `path` lies in: the nearest directory at or
/// above it holding a `store.json`. `None` for a path in no store.
pub(crate) fn store_holding(path: &Path) -> Option<std::path::PathBuf> {
    let abs = std::fs::canonicalize(path).ok()?;
    abs.ancestors().find(|d| d.join("store.json").is_file()).map(Path::to_path_buf)
}

type OpenedCache = std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<OpenedFit>>>;

/// The fit `text` names, opened once per command in [`reader_root`]'s store
/// and shared by every reader of it.
pub(crate) fn opened(text: &str) -> Result<std::sync::Arc<OpenedFit>, String> {
    static CACHE: std::sync::OnceLock<OpenedCache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(o) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(text) {
        return Ok(std::sync::Arc::clone(o));
    }
    let o = std::sync::Arc::new(open_unshared(text)?);
    cache.lock().unwrap_or_else(|e| e.into_inner()).insert(text.to_string(), std::sync::Arc::clone(&o));
    Ok(o)
}

/// The fit `text` names, opened in the store [`opened`] reads it from, for a
/// reader that changes what it opened and so does not share it.
pub(crate) fn open_unshared(text: &str) -> Result<OpenedFit, String> {
    // A result's folder names its own store.
    let root = store_holding(Path::new(text)).unwrap_or_else(reader_root);
    let Some(store) = crate::camdl_store::open_existing(&root) else {
        if is_config_file(text) {
            return Err(format!(
                "{text} is a fit configuration, and a configuration never names a run; there is \
                 no run store at {}, so no run has been made from it. Run `camdl fit run {text}`, \
                 then name the run by its @label, its id or its folder.",
                root.display()
            ));
        }
        return Err(format!("{text} names a stored fit, and there is no run store at {}", root.display()));
    };
    open_fit(&store, text)
}

/// [`opened`] for a reader that reads one run: a problem handle is refused.
pub(crate) fn opened_run(text: &str) -> Result<std::sync::Arc<OpenedFit>, String> {
    let o = opened(text)?;
    refuse_problem_handle(text, &o)?;
    Ok(o)
}

/// [`open_unshared`] for a reader that reads one run: a problem handle is
/// refused.
pub(crate) fn open_run_unshared(text: &str) -> Result<OpenedFit, String> {
    let o = open_unshared(text)?;
    refuse_problem_handle(text, &o)?;
    Ok(o)
}

/// Refuses `o`, opened by `text`, when it names a fit problem and not one of
/// its runs.
fn refuse_problem_handle(text: &str, o: &OpenedFit) -> Result<(), String> {
    if o.run.is_none() {
        let store = crate::camdl_store::open_existing(&store_holding(Path::new(text)).unwrap_or_else(reader_root));
        return Err(match store {
            Some(store) => not_one_run_of(&store, text, o.fit.problem.id()),
            None => format!("{text} names a fit problem, and this reads one run of it; name the run"),
        });
    }
    Ok(())
}

/// The fit problem a `--fit` argument names, for a reader of the problem
/// alone (`pfilter`, `design lhs`, `profile`): a fit file, read in process and
/// stored nowhere, or a stored fit, opened through the store.
pub(crate) struct FitArg {
    /// The argument as typed, for messages.
    pub handle: String,
    /// The problem (and the run's method, for a run handle). For a stored
    /// fit every path in it names a checked copy in a scratch directory.
    pub config: FitConfig,
    /// The text `config` was parsed from: the fit file, or the stored
    /// problem's `problem.toml`.
    text: String,
    /// The stored fit, kept open so its copies stay; `None` for a fit file.
    pub stored: Option<Box<OpenedFit>>,
}

impl FitArg {
    /// A `--fit` argument: a fit file when it names one, else a handle in
    /// `store`.
    #[expect(clippy::disallowed_methods, reason = "reads the fit file `--fit` names, declared `#[input(identity)]` on the `pfilter` and `profile` arguments; a stored fit is read through `open_fit` instead")]
    pub fn read(text: &str, store: Option<&Store>) -> Result<FitArg, String> {
        if is_config_file(text) {
            let path = std::path::PathBuf::from(text);
            let body = std::fs::read_to_string(&path).map_err(|e| format!("cannot read fit config {text}: {e}"))?;
            let config = FitConfig::parse_anchored_at(&body, &path, &path)
                .map_err(|e| format!("cannot load fit config {text}: {e}"))?;
            return Ok(FitArg {
                handle: text.to_string(),
                config,
                text: body,
                stored: None,
            });
        }
        let store = store.ok_or_else(|| format!("{text} names a stored fit, and there is no run store"))?;
        let opened = open_fit(store, text)?;
        let body = String::from_utf8(read_stored(&opened.fit.problem, PROBLEM_FILE)?)
            .map_err(|e| format!("{PROBLEM_FILE}: {e}"))?;
        Ok(FitArg {
            handle: text.to_string(),
            config: opened.config(),
            text: body,
            stored: Some(Box::new(opened)),
        })
    }

    /// The text [`Self::config`] was parsed from, for a reader that asks what
    /// the config writes (`[config] dt`).
    pub fn config_text(&self) -> &str {
        &self.text
    }

    /// The argument's file name, for messages.
    pub fn name(&self) -> String {
        Path::new(&self.handle)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.handle.clone())
    }

    /// The stored problem, as a dependency; `None` for a fit file.
    pub fn problem_dep(&self) -> Option<Dep<FitProblem>> {
        self.stored.as_ref().map(|o| o.fit.problem.dep_all())
    }

    /// The binomial sampler the fit's method draws its chain-binomial
    /// binomials with: a stored run's recorded sampler, else its method's
    /// `[method] binomial`. `None` when the fit names no method, or its
    /// method runs on another backend, which draws no binomials.
    pub fn binomial(&self) -> Result<Option<sim::rng::BinomialAlgorithm>, String> {
        let Some(method) = self.config.inference.method.as_ref() else { return Ok(None) };
        if crate::args::types::ForwardBackend::from(method.algorithm.backend())
            != crate::args::types::ForwardBackend::ChainBinomial
        {
            return Ok(None);
        }
        let configured = method.algorithm.binomial();
        match self.stored.as_ref().and_then(|o| o.run_dir()) {
            Some(run) => crate::fit::forecast_output::fit_binomial(run, Some(configured)).map(Some),
            None => Ok(Some(configured)),
        }
    }
}

/// The handle a reader was given, parsed. A configuration file is never a
/// handle: a fit file names a problem and, through its invocations, the runs
/// it produced.
pub(crate) fn parse_handle(text: &str) -> Result<Handle, String> {
    // A label's earlier holder is `@<label>~<N>`, `N` a whole number.
    if let Some((name, back)) = text.strip_prefix('@').and_then(|r| r.rsplit_once('~')) {
        if back.is_empty() || !back.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!(
                "{text}: a label names its latest holder as @{name}, and an earlier one as \
                 @<label>~<N>, N a whole number counting back (@{name}~1 is the one before)"
            ));
        }
    }
    text.parse::<Handle>().map_err(|e| format!("{text}: {e}"))
}

/// The invocations that hold or held label `name`, as the store counts them
/// (`@name` first, then `@name~1`, ...): each one's sequence number and the
/// fit runs it named.
fn label_history(store: &Store, name: &str) -> Vec<(u64, Vec<store::Key>)> {
    store
        .label_holders(name)
        .into_iter()
        .map(|inv| {
            let runs = inv.members::<FitRun>().map(|m| m.into_iter().map(|(id, _)| id.key()).collect()).unwrap_or_default();
            (inv.seq(), runs)
        })
        .collect()
}

/// The refusal of `@name~back` past the end of the label's history, listing
/// it.
fn past_the_history(store: &Store, text: &str, name: &str) -> Option<String> {
    let held = label_history(store, name);
    if held.is_empty() {
        return None;
    }
    let mut msg = format!("{text}: @{name} has {} holder(s); counting back, they are:", held.len());
    for (i, (seq, runs)) in held.iter().enumerate() {
        let ids: Vec<String> = runs.iter().map(|k| k.short12()).collect();
        let form = if i == 0 { format!("@{name}") } else { format!("@{name}~{i}") };
        msg.push_str(&format!("\n    {form:<12} invocation {seq}  run {}", ids.join(", ")));
    }
    Some(msg)
}

/// Whether `text` names a fit configuration file rather than a stored result.
pub(crate) fn is_config_file(text: &str) -> bool {
    let p = Path::new(text);
    p.extension().is_some_and(|e| e.eq_ignore_ascii_case("toml")) || (p.is_file() && !p.is_dir())
}

/// The one completed fit run `text` names in `store`: `@label`, a run-id
/// prefix or the run's folder. A fit file is refused with the runs its
/// invocations produced, newest first, and a handle naming a problem or
/// several runs is refused with them listed.
pub(crate) fn resolve_run(store: &Store, text: &str) -> Result<Stored<FitRun>, String> {
    if is_config_file(text) {
        let source = store::SourceBytes::read(text).map_err(|e| e.to_string())?;
        return Err(not_a_run(store, &source));
    }
    let handle = parse_handle(text)?;
    match store.resolve::<FitRun>(&RunRef::new(handle)) {
        Ok(run) => Ok(run),
        Err(store::Error::WrongKind { found, .. }) if found == FitProblem::KIND => {
            match store.resolve::<FitProblem>(&RunRef::new(parse_handle(text)?)) {
                Ok(problem) => Err(not_one_run_of(store, text, problem.id())),
                Err(_) => Err(format!("{text} names a fit problem, and this reads one run of it; name the run")),
            }
        }
        Err(e) => Err(not_completed(store, text).unwrap_or_else(|| e.to_string())),
    }
}

/// Why the fit run or problem `text` names is not one a reader can read, when
/// the store holds it in another state than completed: its folder, its state
/// and the reason the store recorded. `None` when `text` names no such result.
pub(crate) fn not_completed(store: &Store, text: &str) -> Option<String> {
    let handle = parse_handle(text).ok()?;
    let (kind, folder, state) = match store.resolve_entry::<FitRun>(&RunRef::new(handle.clone())) {
        Ok(e) => ("fit run", e.folder().to_path_buf(), e.state().clone()),
        Err(_) => {
            let e = store.resolve_entry::<FitProblem>(&RunRef::new(handle)).ok()?;
            ("fit problem", e.folder().to_path_buf(), e.state().clone())
        }
    };
    if matches!(state, store::EntryState::Completed) {
        return None;
    }
    let why = entry_state(&state);
    Some(format!(
        "{text} names a {kind} that did not complete and is not read: {} {why}",
        crate::fit::report::display_path(&folder)
    ))
}

/// For a handle that is a label, the line saying which result it names, on
/// stderr: `@post → run 69fda5 (<folder>)`.
pub(crate) fn say_resolution(store: &Store, text: &str, fit: &StoredFit) {
    let Ok(Handle::Label { name, back }) = text.parse::<Handle>() else { return };
    // The holders the label had before the one named.
    let earlier = label_history(store, &name).len().saturating_sub(1 + back as usize);
    let tail = match earlier {
        0 => String::new(),
        1 => " (1 earlier holder)".to_string(),
        n => format!(" ({n} earlier holders)"),
    };
    let (kind, key, folder) = match &fit.run {
        Some(r) => ("run", r.id().key(), r.folder().to_path_buf()),
        None => ("problem", fit.problem.id().key(), fit.problem.folder().to_path_buf()),
    };
    crate::out::detached().say(format_args!(
        "{text} → {kind} {} ({}){tail}",
        &key.to_hex()[..6],
        crate::fit::report::display_path(&folder)
    ));
}

/// A result's state other than completed, as a refusal says it.
fn entry_state(state: &store::EntryState) -> String {
    match state {
        store::EntryState::Completed => "completed".to_string(),
        store::EntryState::Failed { operation, reason } => format!(
            "failed{}{}",
            operation.as_ref().map(|o| format!(" while {o}")).unwrap_or_default(),
            reason.as_ref().map(|r| format!(": {r}")).unwrap_or_default()
        ),
        store::EntryState::Interrupted { .. } => "was interrupted: its process ended before it completed".to_string(),
        store::EntryState::Running { .. } => "is still running".to_string(),
        store::EntryState::Damaged { what } => format!("is damaged: {what}"),
    }
}

/// The refusal of a problem handle offered where one run is needed: the
/// problem's stored runs, each with its folder. A problem names no one run,
/// even when it holds one, since a later run of it would change the answer.
fn not_one_run_of(store: &Store, text: &str, problem: &store::Id<FitProblem>) -> String {
    let mut runs: Vec<(store::Key, std::path::PathBuf)> = Vec::new();
    for inv in store.invocations().iter() {
        let Ok(members) = inv.members::<FitRun>() else { continue };
        for (id, _) in members {
            if runs.iter().any(|(k, _)| *k == id.key()) {
                continue;
            }
            if let Ok(Some(run)) = store.get(&id) {
                if run.identity().chain.problem.id() == problem {
                    runs.push((id.key(), run.folder().to_path_buf()));
                }
            }
        }
    }
    let mut msg = format!("{text} names a fit problem, and this reads one run of it.");
    if runs.is_empty() {
        msg.push_str("\n  No run of it is stored.");
    } else {
        msg.push_str(" Its runs:");
        for (key, folder) in &runs {
            msg.push_str(&format!("\n    {}  {}", key.short12(), folder.display()));
        }
        msg.push_str("\n  Name one by its id or folder, or by `@label` (`fit run --label`).");
    }
    msg
}

/// The refusal of a fit file offered where one run is needed: the runs the
/// file's invocations produced, newest first, each with its folder.
pub(crate) fn not_a_run(store: &Store, source: &store::SourceBytes) -> String {
    let runs = store.runs_of_configuration(source.digest());
    let mut text = format!(
        "{} is a fit configuration, and a configuration never names a run: one file makes \
         several runs whenever its data, its seeds, a sweep or the build change.",
        source.path().display()
    );
    let mut listed = 0;
    for id in &runs {
        if let Ok(run) = store.resolve::<FitRun>(&RunRef::new(Handle::IdPrefix(id.to_hex()))) {
            text.push_str(&format!("\n    {}  {}", id.short12(), run.folder().display()));
            listed += 1;
        }
    }
    if listed == 0 {
        text.push_str("\n  No run of it is stored. Run `camdl fit run` on it, then name the run by \
                       `@label` (`--label`) or by its id.");
    } else {
        text.push_str("\n  Name one by its id or folder, or by `@label` (`fit run --label`).");
    }
    text
}

/// For a test that writes a run's folder by hand: the run's inputs a
/// record's `inputs` carries, with its run id and its problem's (the first
/// level's hash), as `run_inputs.json` beside it.
#[cfg(test)]
pub(crate) fn write_test_inputs(dir: &Path, record: &serde_json::Value) {
    let mut inputs = record["inputs"].clone();
    if let Some(o) = inputs.as_object_mut() {
        o.insert("run_id".into(), record["run_id"].clone());
        if let Some(h) = record.pointer("/levels/0/hash") {
            o.insert("problem_id".into(), h.clone());
        }
    }
    std::fs::write(dir.join(RUN_INPUTS_FILE), serde_json::to_vec(&inputs).unwrap()).unwrap();
}

#[cfg(test)]
mod tests;
