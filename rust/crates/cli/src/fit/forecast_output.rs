//! `camdl predict --mode forecast`: the fixed-origin forecast of a fit,
//! from the end of its training rows out to `--horizon`
//! (`docs/engineering/forecasting.md` §1.2, §3.5; migration plan, Stage 5).
//!
//! Reads the fit's recorded training partition (verified against the data),
//! builds the [`ForecastPlan`], runs the [`fixed_origin_forecast`] driver over
//! the posterior draws, and writes the forecast artifact:
//!
//! - `forecast/<stream>.tsv`: one row per target, with its period, mode,
//!   periods ahead, lead time, quantity, quantiles and mean;
//! - `forecast-samples.tsv`: joint samples, each a complete trajectory across
//!   every target (D54), with the quantity each value is a draw of;
//! - `forecast.json`: the plan (origin, reach, targets and where each stream's
//!   came from), the training rows and the conditioning rows, and the
//!   calendar.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::fit::failures::{DeterministicFailure, Site, SkipReason, SkippedStream};
use crate::fit::forecast_arms::{build_arms, ArmInputs, ArmLabel, Difference};
use crate::fit::predict_streams::StreamSelection;
use crate::fit::forecast_driver::{extend_run, fixed_origin_forecast, DriverConfig, Forecast, HeldOutValues};
use crate::fit::forecast_covariates::{classify, Covariates, ReachContext};
use crate::fit::forecast_plan::{parse_reach, DesignInput, ForecastPlan, HeldOut, PlanStream, RowFile};
use crate::fit::observed_rows::{Measures, ObservedRow};
use crate::fit::predict::PosteriorDraws;
use crate::fit::training_partition::TrainingPartition;
use crate::quantile::{band, fmt_time, fmt_value, Quantity, QUANTILE_LEVELS};

/// Joint samples written per posterior draw, evenly strided over its
/// particles. Every particle is a draw from the same predictive (equal weights
/// after the origin), so a stride is a sample of them; the bands use all.
pub(crate) const FORECAST_SAMPLES_PER_DRAW: usize = 5;

/// A posterior draw the forecast ran: its row of the fit's `draws.tsv` and
/// the `(chain, draw)` key written there, when the cloud carries one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DrawRef {
    pub cloud_row: usize,
    pub key: Option<(usize, usize)>,
}

impl DrawRef {
    /// How a message names the draw.
    pub(crate) fn describe(&self) -> String {
        match self.key {
            Some((c, d)) => format!(
                "posterior draw (chain {c}, draw {d}), row {} of draws.tsv", self.cloud_row),
            None => format!("posterior draw at row {} of draws.tsv", self.cloud_row),
        }
    }
}

/// The `schema` tag of `forecast.json`.
pub(crate) const FORECAST_SCHEMA: &str = "camdl.forecast/v6";

/// The file [`write_manifest`] writes in a forecast's directory.
pub(crate) const FORECAST_MANIFEST: &str = "forecast.json";

/// What `camdl show` states of one forecast, read back from its
/// `forecast.json`: where it starts, how far it reaches, which chains its
/// draws came from and how many particles each draw's filter ran.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct ForecastSummary {
    pub origin: SummaryPoint,
    pub horizon: SummaryHorizon,
    pub calendar: ir::caltime::CalendarAxis,
    pub n_draws: usize,
    pub particles: crate::fit::predict::PredictParticles,
    /// Absent when the full posterior cloud was used.
    #[serde(default)]
    pub chain_selection: Option<SummaryChains>,
}

/// A time `forecast.json` records: model time, and its date under a
/// calendar origin.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct SummaryPoint {
    pub t: f64,
    pub date: Option<String>,
}

/// `forecast.json`'s `horizon`: `--horizon` as written, and its length in
/// model time units.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct SummaryHorizon {
    pub as_written: String,
    pub duration: f64,
}

/// `forecast.json`'s `chain_selection`.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub(crate) struct SummaryChains {
    pub excluded: Vec<usize>,
    pub kept: Vec<usize>,
    pub n_total: usize,
}

impl ForecastSummary {
    /// The forecast `bytes` (a `forecast.json`) describe; an error for bytes
    /// that are not one, or carry a schema other than [`FORECAST_SCHEMA`].
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, String> {
        let doc: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("not JSON: {e}"))?;
        match doc.get("schema").and_then(|s| s.as_str()) {
            Some(FORECAST_SCHEMA) => {}
            other => return Err(format!(
                "schema {}, not {FORECAST_SCHEMA}, which this camdl reads",
                other.map_or_else(|| "missing".to_string(), |s| format!("'{s}'")))),
        }
        serde_json::from_value(doc).map_err(|e| e.to_string())
    }

    /// The summary as `(label, value)` lines.
    pub(crate) fn lines(&self) -> Vec<(&'static str, String)> {
        let origin = match &self.origin.date {
            Some(d) => format!("t = {} ({d})", fmt_time(self.origin.t)),
            None => format!("t = {}", fmt_time(self.origin.t)),
        };
        let horizon = format!("{} ({} {})",
            self.horizon.as_written, fmt_time(self.horizon.duration), self.calendar.time_unit.as_str());
        let ids = |v: &[usize]| v.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
        let chains = match &self.chain_selection {
            None => format!("all ({} posterior draws)", self.n_draws),
            Some(s) => format!("kept {}, left out {}, of {} ({} posterior draws)",
                ids(&s.kept), ids(&s.excluded), s.n_total, self.n_draws),
        };
        let particles = format!("{} per draw", self.particles.n);
        vec![("origin", origin), ("horizon", horizon), ("chains", chains), ("particles", particles)]
    }
}

/// The joint samples file `write_tables` writes beside it.
pub(crate) const FORECAST_SAMPLES_FILE: &str = "forecast-samples.tsv";

/// The directory of the per-stream forecast tables, `forecast/<stream>.tsv`.
pub(crate) const FORECAST_TABLES_DIR: &str = "forecast";


/// Why a fixed point cannot be forecast on `backend`; `None` on
/// chain-binomial, the one backend whose forecast exists.
pub(crate) fn fixed_point_forecast_refusal(backend: crate::args::types::ForwardBackend) -> Option<String> {
    use crate::args::types::ForwardBackend;
    match backend {
        ForwardBackend::ChainBinomial => None,
        ForwardBackend::Ode => Some(
            "a fixed-origin forecast on the ODE backend (the cumulative-flow difference on the \
             trajectory) is unsupported; forecast on --backend chain_binomial, or use \
             --mode free_forward."
                .to_string(),
        ),
        ForwardBackend::Gillespie => Some(
            "a forecast filters the conditioned rows, and the Gillespie backend runs no particle \
             filter; forecast on --backend chain_binomial, or use --mode free_forward."
                .to_string(),
        ),
    }
}

/// Refuse what forecast mode does not support, before any work: the backends
/// and options of the Stage 5 capability matrix that have no forecast yet.
pub(crate) fn refuse_unsupported(
    args: &crate::args::PredictArgs,
    posterior: &PosteriorDraws,
) -> Result<(), String> {
    use crate::args::types::ForwardBackend;
    let leaf = &posterior.name;
    if posterior.treatment() == crate::fit::predict::TreatmentKind::PlugIn {
        if let Some(why) = fixed_point_forecast_refusal(posterior.backend) {
            return Err(why);
        }
    }
    match posterior.backend {
        ForwardBackend::ChainBinomial => {}
        ForwardBackend::Ode => return Err(format!(
            "the '{leaf}' method leaf ran on the ODE backend; a fixed-origin forecast of an \
             ODE posterior (the cumulative-flow difference on each draw's trajectory) is not \
             available yet. A chain-binomial posterior (PMMH, PGAS) can be forecast; for an \
             ODE fit use --mode free_forward.")),
        ForwardBackend::Gillespie => return Err(format!(
            "the '{leaf}' method leaf ran on the Gillespie backend, which is not a fitting \
             backend, so there is nothing to forecast.")),
    }
    // A stored fit holds its held-out rows' dates and never their values,
    // which enter only a score; `--realised-design` reads the held-out rows'
    // design values.
    if args.posterior.realised_design {
        return Err("--realised-design reads the design values of the fit's held-out rows, and a stored \
                    fit holds only their dates: held-out values enter only a score, which the run store \
                    computes as its own result (step 7 of the run-store proposal, scores). Supply the \
                    design with --design SOURCE=FILE.".to_string());
    }
    let mut asked: Vec<&str> = Vec::new();
    if !args.flags.enable.is_empty() { asked.push("--enable"); }
    if !args.flags.disable.is_empty() { asked.push("--disable"); }
    if args.posterior.by_chain { asked.push("--by-chain"); }
    if args.posterior.model.is_some() { asked.push("--model"); }
    if !asked.is_empty() {
        return Err(format!(
            "{} cannot be combined with --mode forecast yet: a forecast writes no quantities \
             table and no per-chain band, and its arms are the fitted model and the named \
             scenarios. To forecast with interventions switched on or off, declare a \
             scenario that does it and pass --scenario; or run --mode free_forward.",
            asked.join(", ")));
    }
    Ok(())
}

/// The binomial sampler the fit run whose files `leaf` holds drew with
/// (gh#747), so the forecast continues the fit's filter on the same sampler:
/// the run's recorded `binomial` (`run_inputs.json`), else `configured`, the
/// method's own, for a method that records none.
pub(crate) fn fit_binomial(
    leaf: &std::path::Path,
    configured: Option<sim::rng::BinomialAlgorithm>,
) -> Result<sim::rng::BinomialAlgorithm, String> {
    use sim::rng::BinomialAlgorithm;
    let path = leaf.join(crate::fit_store::RUN_INPUTS_FILE);
    let inputs = crate::fit_store::read_run_inputs(leaf)?;
    match (inputs.get("binomial").filter(|v| !v.is_null()), configured) {
        (Some(v), _) => serde_json::from_value::<BinomialAlgorithm>(v.clone()).map_err(|e| format!(
            "{}: the recorded binomial sampler {v} is not one camdl knows: {e}", path.display())),
        (None, Some(c)) => Ok(c),
        (None, None) => Err(format!(
            "{} records no binomial sampler and the fit names no method; re-run the fit \
             so the forecast can continue its filter on the same sampler.", path.display())),
    }
}

/// What the forecast needs from `run_predict`.
pub(crate) struct ForecastJob<'a> {
    pub args: &'a crate::args::PredictArgs,
    /// The fitted model, observation anchors resolved.
    pub model: &'a ir::Model,
    /// The same model with every parameter seeded from a draw, so it compiles.
    pub seeded: &'a ir::Model,
    pub compiled: &'a Arc<sim::CompiledModel>,
    pub config: &'a crate::fit::config_v2::Problem,
    pub fit_base: &'a crate::util::FitBase,
    pub posterior: &'a PosteriorDraws,
    /// Where the rows the parameters were fitted to are recorded.
    pub partition: &'a crate::fit::training_partition::RecordedPartition,
    /// The directory the forecast's files are written into.
    pub out_dir: &'a std::path::Path,
    /// Each stream's held-out rows without their values, as the stored
    /// problem holds them (`held_out/<stream>.tsv`): the rows a target is
    /// matched to.
    pub held_out: &'a BTreeMap<String, Vec<u8>>,
    pub run_id: &'a str,
    /// The options as resolved before the plan: the forecast adds its design
    /// inputs (D81) and addresses its directory from the result.
    pub options: &'a crate::fit::predict::PredictOptions,
    pub calendar: &'a ir::caltime::CalendarAxis,
    pub schema: Option<&'a crate::run_meta::ObsSchema>,
    pub dt: f64,
    pub seed: u64,
    /// The binomial sampler the fit's method ran (gh#747).
    pub binomial: sim::rng::BinomialAlgorithm,
    /// The streams this forecast covers, skips and fails before any work (D76).
    pub selection: &'a StreamSelection,
    /// `camdl predict`'s scenario list, the fitted model first (D50).
    pub scenarios: &'a [crate::sim_job::ScenarioRef],
    /// The `--sweep` grid, one entry per point; empty without `--sweep`.
    pub sweep_points: &'a [Vec<(String, f64)>],
    /// The filters' particle count per draw.
    pub particles: crate::fit::predict::PredictParticles,
    /// The name of the arm with no overlay: `fitted`, or `(none)` for a
    /// fixed point.
    pub baseline: &'a str,
    /// The outcomes `camdl score` binds: each target's value is bound as a
    /// held-out cell, which only the filter's scoring likelihood reads.
    /// `None` for `camdl predict`, which binds none.
    pub binding: Option<&'a crate::fit::scoring::OutcomeBinding>,
}

/// What [`run_forecast`] computed: the directory and the files written, the
/// streams that failed and were skipped, the options the forecast is keyed
/// by, and, when outcomes were bound, the scored targets.
pub(crate) struct ForecastRun {
    pub out_dir: PathBuf,
    pub written: Vec<PathBuf>,
    pub failures: Vec<DeterministicFailure>,
    pub skipped: Vec<SkippedStream>,
    pub options: crate::fit::predict::PredictOptions,
    pub scored: crate::fit::scoring::Scored,
}

/// The held-out rows of one stream, from the stored problem's
/// `held_out/<stream>.tsv`: each row's file, position, label and what it
/// measures, and no value.
pub(crate) fn held_out_rows(bytes: &[u8]) -> Result<Vec<HeldOut>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut rows = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let cols: Vec<&str> = line.split('\t').collect();
        let [file, position, label, measures, start, end] = cols[..] else {
            return Err(format!("expected six columns, found {}: {line}", cols.len()));
        };
        let num = |x: &str| x.parse::<f64>().map_err(|e| format!("{x}: {e}"));
        let measures = match measures {
            "instant" => Measures::Instant(num(start)?),
            "period" => Measures::Period(sim::inference::multi_stream_obs::Period::new(num(start)?, num(end)?)?),
            other => return Err(format!("unknown measure `{other}`")),
        };
        let file = match file {
            "training" => RowFile::Training,
            "holdout" => RowFile::Holdout,
            other => return Err(format!("unknown file `{other}`")),
        };
        rows.push(HeldOut {
            file,
            row: ObservedRow {
                position: position.parse().map_err(|e| format!("{position}: {e}"))?,
                label: num(label)?,
                measures,
                temporal: Vec::new(),
                cell: None,
                aux: Vec::new(),
                design: Vec::new(),
            },
        });
    }
    Ok(rows)
}

/// One `--design` file as `forecast.json` records it: the stream it was
/// given for and the SHA-256 of its bytes, never the path typed.
#[derive(Debug, Clone, serde::Serialize)]
struct DesignFile {
    stream: String,
    sha256: String,
}

/// Each `--design SOURCE=FILE`, read for every leaf whose source (or name) is
/// SOURCE (D81): the leaf's design rows, by leaf name, and each file's
/// content digest.
#[expect(clippy::disallowed_methods, reason = "reads each `--design` file, declared `#[input(identity)]` on `PredictFlags`, which `store_prediction` stores as a `DataTable` the prediction's identity depends on")]
fn prospective_design(
    model: &ir::Model,
    specs: &[String],
    time_opts: &crate::caltime_load::TimeOpts,
) -> Result<(BTreeMap<String, Vec<ObservedRow>>, Vec<DesignFile>), String> {
    let mut out = BTreeMap::new();
    let mut files = Vec::with_capacity(specs.len());
    for spec in specs {
        let (name, path) = spec.split_once('=').ok_or_else(|| format!(
            "--design {spec:?}: expected SOURCE=FILE, a stream and its design file"))?;
        let bytes = std::fs::read(path).map_err(|e| format!("--design {spec:?}: cannot read {path}: {e}"))?;
        files.push(DesignFile { stream: name.to_string(), sha256: crate::hashing::sha256_hex(&bytes) });
        let leaves: Vec<&ir::observation::ObservationModel> = model.observations.iter()
            .filter(|o| o.source == name || o.name == name)
            .collect();
        if leaves.is_empty() {
            return Err(format!(
                "--design {spec:?}: the model has no observation stream '{name}'"));
        }
        for o in leaves {
            if out.insert(o.name.clone(), crate::fit::observed_rows::design_rows(path, o, time_opts)?).is_some() {
                return Err(format!("--design names stream '{}' twice", o.name));
            }
        }
    }
    Ok((out, files))
}

/// The options a forecast's directory is keyed by: `base`, with what the
/// plan contributes to the predictions' inputs and the statements of the
/// table forcings it reads after its origin.
pub(crate) fn forecast_options(
    base: &crate::fit::predict::PredictOptions,
    plan: &ForecastPlan,
    covariates: &Covariates,
) -> crate::fit::predict::PredictOptions {
    crate::fit::predict::PredictOptions {
        forecast: Some(crate::fit::predict::ForecastKey {
            targets: plan.targets_digest(),
            design_mode: plan.design_mode(),
            covariates: covariates.key(),
        }),
        ..base.clone()
    }
}

/// Run the forecast and write its artifact. Returns the directory it was
/// written to (keyed by the options with the forecast's design inputs), the
/// files written, the streams that failed (asked for and unbound, or refused a
/// forecast) and the streams skipped; the caller records the last two in
/// `report.json`.
pub(crate) fn run_forecast(out: &crate::out::Out, job: ForecastJob) -> Result<ForecastRun, String> {
    let model = job.model;
    let raw = job.args.flags.horizon.as_deref().expect("forecast mode carries --horizon");
    let reach = parse_reach(raw, &model.time_unit, model.origin.as_deref())?;
    let data = job.config.data_spec()
        .map_err(|e| format!("this fit has no [data] block to read its training rows from: {e}"))?;
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)
        .map_err(|e| format!("resolving [data]: {e}"))?;
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt: job.dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let mut streams = crate::fit::runner::resolve_and_load_obs_streams(
        model, job.compiled, bindings.training_by_source(), &time_opts)?;

    // The training rows: the rows the fit recorded, verified against the data.
    let rows: BTreeMap<String, Vec<ObservedRow>> =
        streams.iter().map(|s| (s.name.clone(), s.rows.clone())).collect();
    let (record, partition) = TrainingPartition::read(job.partition, &rows)?;

    // Each stream's held-out rows, from the stored problem's dates: they only
    // name a target's held-out row, and are never filtered. A stored fit
    // holds no held-out value, so no target is scored.
    let mut held_out: BTreeMap<String, Vec<HeldOut>> = BTreeMap::new();
    for s in &streams {
        let rows = match job.held_out.get(&s.name) {
            Some(bytes) => held_out_rows(bytes).map_err(|e| format!("held_out/{}.tsv: {e}", s.name))?,
            None => Vec::new(),
        };
        held_out.insert(s.name.clone(), rows);
    }
    if job.binding.is_none() && held_out.values().any(|r| !r.is_empty()) {
        out.say("predict: a forecast from a stored fit computes no held-out score: the fit holds its \
                 held-out rows' dates and not their values, which enter only a score (step 7 of the \
                 run-store proposal)");
    }
    for s in &mut streams {
        s.truncate_rows(partition.rows(&s.name));
    }

    // The plan: targets for the eligible streams (D76); every bound stream
    // conditions the state. A stream with no bound data is not among
    // `streams` at all: it has no rows to continue and no stop in the filter
    // where its running total could start, so the selection has already
    // skipped it or, when `--stream` asked for it, failed it.
    let selection = job.selection;
    // D81: where a design-reading stream's target design values come from.
    let (design_files, design_digests) = prospective_design(model, &job.args.flags.design, &time_opts)?;
    let design_of = |name: &str| match (job.args.posterior.realised_design, design_files.get(name)) {
        (true, _) => DesignInput::ExPost,
        (false, Some(rows)) => DesignInput::Prospective(rows),
        (false, None) => DesignInput::NotSupplied,
    };
    let plan_streams: Vec<PlanStream> = streams.iter()
        .filter(|s| selection.includes(&s.name))
        .map(|s| PlanStream {
            obs: &s.obs_model_ir, training: &s.rows, held_out: &held_out[&s.name], design: design_of(&s.name),
        })
        .collect();
    if let Some(name) = design_files.keys().find(|n| !plan_streams.iter().any(|p| &p.obs.name == *n)) {
        return Err(format!(
            "--design supplies stream '{name}', which this forecast does not predict (it binds \
             no rows, or --stream left it out)"));
    }
    if let Some(o) = model.observations.iter()
        .find(|o| selection.includes(&o.name) && !streams.iter().any(|s| s.name == o.name))
    {
        return Err(format!(
            "internal error: stream '{}' was selected as eligible but its rows were not loaded",
            o.name));
    }
    let plan = ForecastPlan::new(partition, reach, &plan_streams).map_err(|e| e.to_string())?;
    // The table forcings the forecast reads after its origin, each with its
    // statement; refused here, before any filter runs, when one has none.
    let mut covariates = Covariates::resolve(
        &classify(model, job.compiled, plan.origin(), plan.origin() + plan.horizon()),
        &job.args.flags.covariate,
        &ReachContext {
            origin: plan.origin(),
            reach_end: plan.origin() + plan.horizon(),
            as_written: raw,
            calendar: job.calendar,
        },
    )?;
    // The table forcings whose values before the origin, which the filter
    // reads, depend on rows after it: a label beside any statement, said
    // once. It enters no key and changes no value.
    {
        let origin = plan.origin();
        let cut = crate::run_meta::TimePoint {
            t: origin,
            date: job.calendar.date_of(origin).map(ir::caltime::RenderedInstant::into_string),
        };
        let at = cut.date.clone().unwrap_or_else(|| format!("t = {}", crate::quantile::fmt_time(origin)));
        for d in crate::fit::forecast_covariates::pre_cut_dependence(model, job.compiled, origin, model.simulation.t_start) {
            out.say(format_args!("predict: note: {}", d.note(&at, "the forecast origin")));
            covariates.interpolation.push(d.record(cut.clone()));
        }
    }
    // Each target's held-out value (D79), bound as a held-out cell: the
    // filter's scoring likelihood evaluates it and nothing else reads it, so
    // the predictions below do not depend on it.
    let held_out_values = match job.binding {
        Some(b) => HeldOutValues::from_outcomes(&plan, b, model)?,
        None => HeldOutValues::from_rows(&plan, &held_out, model)?,
    };
    // The address (D81, D86): the options with the design inputs that
    // determine the predictions. The held-out outcomes are not among them.
    let options = forecast_options(job.options, &plan, &covariates);
    let address_key = options.address_key(job.run_id);
    let out_dir = job.out_dir.to_path_buf();
    if let Some(mode) = plan.design_mode() {
        out.say(format_args!("predict: forecast design values: {}", mode.label()));
    }
    for line in covariates.describe() {
        out.say(format_args!("predict: {line}"));
    }
    // An eligible stream the plan refused is a failure; one with no target
    // within the reach is skipped, unless `--stream` asked for it.
    let mut failures: Vec<DeterministicFailure> = selection.failures.clone();
    failures.extend(plan.refusals().iter().map(|r| DeterministicFailure::EvaluationFailed {
        at: Site::ForecastStream(r.stream().to_string()),
        reason: r.to_string(),
    }));
    let mut skipped: Vec<SkippedStream> = selection.skipped.clone();
    for p in &plan_streams {
        let name = &p.obs.name;
        let produced = plan.sources().iter().any(|(s, _)| s == name);
        let refused = plan.refusals().iter().any(|r| r.stream() == name);
        if produced || refused {
            continue;
        }
        if selection.was_requested(name) {
            failures.push(DeterministicFailure::EvaluationFailed {
                at: Site::ForecastStream(name.clone()),
                reason: format!(
                    "--stream asked for it, but {}; lengthen --horizon to reach its next period",
                    SkipReason::NoTargetWithinHorizon.describe()),
            });
        } else {
            skipped.push(SkippedStream { stream: name.clone(), reason: SkipReason::NoTargetWithinHorizon });
        }
    }

    // The fitted model, its run extended to the latest target (D49).
    let mut fitted = job.seeded.clone();
    job.fit_base.filter(&mut fitted)?;
    extend_run(&mut fitted, &plan);
    if plan.run_end() > model.simulation.t_end {
        out.say(format_args!(
            "predict: the forecast runs to t = {}, past the model's `simulate {{ to }}` \
             (t = {}), with interventions planned to that end; nothing else is extended.",
            plan.run_end(), model.simulation.t_end));
    }
    let mut compiled = sim::CompiledModel::new(fitted)
        .map_err(|e| format!("compiling the fitted model for the forecast: {e:?}"))?
        .with_eval_settings(job.compiled.eval);
    covariates.apply_holds(&mut compiled, plan.origin())?;
    let compiled = Arc::new(compiled);

    // The posterior draws, subsampled as the other modes do.
    let cap = job.args.flags.n_draws.unwrap_or(crate::fit::predict::DEFAULT_PREDICT_DRAWS);
    let idx = crate::fit::predict::subsample_indices(job.posterior.n_draws(), cap);
    if idx.len() < job.posterior.n_draws() {
        out.say(format_args!(
            "predict: forecast — subsampling {} of {} posterior draws (raise with --n-draws)",
            idx.len(), job.posterior.n_draws()));
    }
    // Each draw is named by its row of draws.tsv, which under a chain
    // selection is not its place `i` in the retained cloud.
    // A fixed point's copies are named by their place among the copies.
    let draw_refs: Vec<DrawRef> = match (job.posterior.keys(), job.posterior.treatment()) {
        (Some(keys), _) => idx.iter().map(|&i| DrawRef { cloud_row: keys.rows[i], key: keys.per_draw[i] }).collect(),
        (None, crate::fit::predict::TreatmentKind::PlugIn) => {
            idx.iter().map(|&i| DrawRef { cloud_row: i, key: None }).collect()
        }
        (None, crate::fit::predict::TreatmentKind::Posterior) => {
            return Err("internal error: the posterior cloud carries no draws.tsv rows to name its draws by".into())
        }
    };
    let param_sets: Vec<Vec<f64>> = idx.iter()
        .map(|&i| {
            let mut p = compiled.default_params.clone();
            for (name, &v) in &job.posterior.draws()[i] {
                if let Some(&k) = compiled.param_index.get(name.as_str()) {
                    p[k] = v;
                }
            }
            p
        })
        .collect();
    // The arms (D50): the fitted model, then each scenario and sweep point,
    // each refused here, before any filtering, unless it differs from the
    // fitted model only at or after the origin.
    let rows: Vec<&indexmap::IndexMap<String, f64>> = idx.iter().map(|&i| &job.posterior.draws()[i]).collect();
    let arms = build_arms(ArmInputs {
        model,
        fit_base: job.fit_base,
        scenarios: job.scenarios,
        sweep_points: job.sweep_points,
        draws: &rows,
        baseline: &compiled,
        baseline_params: &param_sets,
        plan: &plan,
        covariates: &covariates,
        dt: job.dt,
    })?;
    let config = DriverConfig {
        n_particles: job.particles.n,
        dt: job.dt,
        base_seed: job.seed,
        record_expected: false,
        // Arms are compared before the origin only when there are arms.
        trace_pre_origin: arms.len() > 1,
        binomial: job.binomial,
    };
    // Every arm runs the same draws with the same seeds; each repeats the
    // filtering of the training rows (filtering once and forking at the
    // origin is deferred, spec Part 7).
    let mut forecasts: Vec<ArmForecast> = Vec::with_capacity(arms.len());
    for arm in arms {
        let name_arm = |s: String| match arm.label.is_baseline() {
            true => s,
            false => format!("{s} (forecast arm {})", arm.label.describe()),
        };
        let fc = fixed_origin_forecast(plan.clone(), &streams, &held_out_values, arm.compiled.clone(), &arm.param_sets, config)
            .map_err(|f| name_arm(f.describe(|d| match job.posterior.treatment() {
                crate::fit::predict::TreatmentKind::PlugIn => format!("copy {} of the fixed point", draw_refs[d].cloud_row),
                crate::fit::predict::TreatmentKind::Posterior => draw_refs[d].describe(),
            })))?;
        if let Some(base) = forecasts.first() {
            check_shared_past(&base.fc, &fc, &arm.label, &draw_refs)?;
        }
        forecasts.push(ArmForecast { label: arm.label, difference: arm.difference, fc });
    }

    let mut written = write_tables(&forecasts, model, job.schema, &out_dir, &draw_refs)?;
    written.push(write_manifest(&forecasts, &job, ManifestInputs {
        record: &record, reach_as_written: raw, skipped: &skipped, failures: &failures,
        draw_refs: &draw_refs, design_files: &design_digests, out_dir: &out_dir, address_key: &address_key,
        options: &options, covariates: &covariates,
    })?);
    let fc = &forecasts[0].fc;
    out.say(format_args!(
        "predict: forecast from t = {} (the end of the training rows) to t = {}: {} \
         target(s) on {} stream(s), {} draws × {} particles, {} arm(s)",
        fc.plan.origin(), fc.plan.origin() + fc.plan.horizon(), fc.plan.targets().len(),
        fc.plan.sources().len(), fc.n_draws, fc.n_particles, forecasts.len()));
    let scored = match job.binding {
        Some(_) => scored_targets(&forecasts, &held_out_values, model),
        None => crate::fit::scoring::Scored::default(),
    };
    Ok(ForecastRun { out_dir, written, failures, skipped, options, scored })
}

/// Each arm's targets that hold an outcome, as scored cells: every sample's
/// draw and its log density of the outcome, which the driver recorded at the
/// target's end before anything could be assimilated there. Beside them, per
/// arm, the joint log density of the targets ending at each time, and of
/// every scored target of the forecast together: one sample is one particle's
/// trajectory across every target (D54), so each joint sums its log densities
/// over the targets before the samples are averaged.
fn scored_targets(arms: &[ArmForecast], values: &HeldOutValues, model: &ir::Model) -> crate::fit::scoring::Scored {
    use crate::fit::scoring::{JointCell, ScoreMode, ScoredCell, Scored};
    let mut scored = Scored::default();
    let fc0 = &arms[0].fc;
    for (k, t) in fc0.plan.targets().iter().enumerate() {
        match (values.at(k), values.not_scored(k)) {
            (Some(_), _) => {}
            (None, Some(why)) => scored.notes.push(format!(
                "the target of '{}' ending at {} is not scored: {why}", t.stream, fmt_time(t.measures.end()))),
            (None, None) => scored.notes.push(format!(
                "the target of '{}' ending at {} is not scored: no outcome row measures its period",
                t.stream, fmt_time(t.measures.end()))),
        }
    }
    let origin = fc0.plan.origin();
    for arm in arms {
        let fc = &arm.fc;
        let n = fc.n_draws * fc.n_particles;
        let joint = |ks: &[usize], start: f64, stop: f64, periods_ahead: Option<i64>, lead: Option<f64>| {
            let mut sum = vec![0.0_f64; n];
            for &k in ks {
                if let Some(ll) = &fc.held_out_log_liks[k] {
                    sum.iter_mut().zip(ll).for_each(|(s, l)| *s += l);
                }
            }
            JointCell {
                scenario: arm.label.scenario.clone(),
                sweep: arm.label.sweep.clone(),
                mode: ScoreMode::Forecast,
                origin: Some(origin),
                periods_ahead,
                lead,
                start,
                stop,
                role: crate::fit::predict::RowRole::Future,
                streams: ks.iter().map(|&k| fc.plan.targets()[k].stream.clone()).collect(),
                log_liks: sum,
            }
        };
        let mut by_end: Vec<(f64, Vec<usize>)> = Vec::new();
        for (k, t) in fc.plan.targets().iter().enumerate() {
            let (Some(y), Some(ll)) = (values.at(k), &fc.held_out_log_liks[k]) else { continue };
            let o = leaf(model, &t.stream);
            scored.cells.push(ScoredCell {
                scenario: arm.label.scenario.clone(),
                sweep: arm.label.sweep.clone(),
                leaf: t.stream.clone(),
                source: o.source.clone(),
                stratum: stratum_of(o),
                start: t.measures.start(),
                stop: t.measures.end(),
                mode: ScoreMode::Forecast,
                origin: Some(origin),
                periods_ahead: Some(t.periods_ahead),
                lead: Some(t.lead),
                role: crate::fit::predict::RowRole::Future,
                observed: y,
                samples: fc.draws[k].clone(),
                log_liks: ll.clone(),
            });
            match by_end.last_mut() {
                Some((end, ks)) if (*end - t.measures.end()).abs() <= crate::fit::runner::TIME_TIE_EPS => ks.push(k),
                _ => by_end.push((t.measures.end(), vec![k])),
            }
        }
        for (end, ks) in &by_end {
            // The targets ending at one time share their lead; `periods_ahead`
            // counts within one stream and is left out of the joint.
            let lead = fc.plan.targets()[ks[0]].lead;
            let start = ks.iter().map(|&k| fc.plan.targets()[k].measures.start()).fold(f64::INFINITY, f64::min);
            scored.joints.push(joint(ks, start, *end, None, Some(lead)));
        }
        let all: Vec<usize> = by_end.iter().flat_map(|(_, ks)| ks.iter().copied()).collect();
        if let Some((last, _)) = by_end.last() {
            scored.joints.push(joint(&all, origin, *last, None, None));
        }
    }
    scored
}

/// One arm's forecast.
pub(crate) struct ArmForecast {
    pub label: ArmLabel,
    /// `None` for the baseline.
    pub difference: Option<Difference>,
    pub fc: Forecast,
}

/// An arm must share the baseline's filter before the origin bit for bit
/// (D50): the same model with the same seeds over the same rows. The static
/// check in [`build_arms`] admits only arms whose differences act at or after
/// the origin; this reads the filters' observation records before the origin
/// (log-likelihood increments, ESS, and each particle's per-stream
/// log-likelihoods and predictive draws), so an arm that differs there by a
/// route that check does not see fails the forecast rather than being
/// written. It does not hash the latent states: a pre-origin difference that
/// no observation reads before the origin is not caught here.
pub(crate) fn check_shared_past(base: &Forecast, arm: &Forecast, label: &ArmLabel, draw_refs: &[DrawRef]) -> Result<(), String> {
    if base.pre_origin.len() != base.n_draws || arm.pre_origin.len() != arm.n_draws {
        return Err(format!(
            "internal error: the forecast arm {} cannot be checked against the fitted model: \
             a filter recorded no trace before the origin",
            label.describe()));
    }
    for (d, (b, a)) in base.pre_origin.iter().zip(&arm.pre_origin).enumerate() {
        if let Some((time, what)) = b.first_difference(a) {
            let at = time.map(|t| format!(" at t = {t}")).unwrap_or_default();
            return Err(format!(
                "the forecast arm {} departs from the fitted model before the forecast origin \
                 t = {}: on {}, {what} differs{at}. Every arm must share the filter through the \
                 training rows; no arm was written. This arm passed the check of what it \
                 changes, so the difference reached the filter by a route that check does not \
                 see: please report it.",
                label.describe(), base.plan.origin(), draw_refs[d].describe()));
        }
    }
    Ok(())
}

fn leaf<'m>(model: &'m ir::Model, name: &str) -> &'m ir::observation::ObservationModel {
    model.observations.iter().find(|o| o.name == name)
        .expect("a target's stream is one of the model's observation blocks")
}

fn stratum_of(o: &ir::observation::ObservationModel) -> Vec<(String, String)> {
    o.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect()
}

/// The swept parameters, in the order `--sweep` named them: the `sweep:<p>`
/// columns of every forecast table.
fn sweep_names(arms: &[ArmForecast]) -> Vec<String> {
    arms.iter().find(|a| !a.label.sweep.is_empty())
        .map(|a| a.label.sweep.iter().map(|(p, _)| p.clone()).collect())
        .unwrap_or_default()
}

/// The `scenario` and `sweep:<p>` cells of an arm's rows; the baseline's
/// sweep cells are empty.
fn arm_cells(label: &ArmLabel, sweep: &[String]) -> Vec<String> {
    std::iter::once(label.scenario.clone())
        .chain(sweep.iter().map(|p| label.sweep.iter().find(|(q, _)| q == p)
            .map(|(_, v)| fmt_value(*v)).unwrap_or_default()))
        .collect()
}

fn arm_header(sweep: &[String]) -> Vec<String> {
    std::iter::once("scenario".to_string()).chain(sweep.iter().map(|p| format!("sweep:{p}"))).collect()
}

/// `forecast/<source>.tsv` per logical stream, and `forecast-samples.tsv`,
/// each arm's rows in turn under a leading `scenario` column (and a
/// `sweep:<p>` column per swept parameter).
fn write_tables(
    arms: &[ArmForecast],
    model: &ir::Model,
    schema: Option<&crate::run_meta::ObsSchema>,
    out_dir: &std::path::Path,
    draw_refs: &[DrawRef],
) -> Result<Vec<PathBuf>, String> {
    let sweep = sweep_names(arms);
    let fc0 = &arms[0].fc;
    let targets = fc0.plan.targets();
    // Logical stream → its targets' indices, in the plan's order.
    let mut by_source: indexmap::IndexMap<String, Vec<usize>> = indexmap::IndexMap::new();
    for (k, t) in targets.iter().enumerate() {
        by_source.entry(leaf(model, &t.stream).source.clone()).or_default().push(k);
    }
    let mut written = Vec::new();
    for (source, ks) in &by_source {
        let first = leaf(model, &targets[ks[0]].stream);
        let dims = crate::fit::predict::index_dims_for(schema, source, &stratum_of(first).into_iter().map(|(d, _)| d).collect::<Vec<_>>());
        let mut header: Vec<String> = arm_header(&sweep);
        header.extend(dims.iter().cloned());
        header.extend(["t_start", "t_stop", "mode", "periods_ahead", "lead", "quantity", "mean"]
            .map(String::from));
        header.extend(QUANTILE_LEVELS.iter().map(|(_, l)| l.to_string()));
        header.push("n_samples".into());
        let mut tsv = header.join("\t") + "\n";
        for (arm, &k) in arms.iter().flat_map(|a| ks.iter().map(move |k| (a, k))) {
            let t = &targets[k];
            let stratum = stratum_of(leaf(model, &t.stream));
            let ys = &arm.fc.draws[k];
            let qs = band(ys).map_err(|e| format!(
                "forecast arm {}, '{}' ending at {}: {e}", arm.label.describe(), t.stream, t.measures.end()))?;
            let mean = ys.iter().sum::<f64>() / ys.len() as f64;
            let mut cells: Vec<String> = arm_cells(&arm.label, &sweep);
            cells.extend(dims.iter()
                .map(|d| stratum.iter().find(|(dd, _)| dd == d).map(|(_, l)| l.clone()).unwrap_or_default()));
            cells.push(fmt_time(t.measures.start()));
            cells.push(fmt_time(t.measures.end()));
            cells.push(FORECAST_MODE.to_string());
            cells.push(t.periods_ahead.to_string());
            cells.push(fmt_time(t.lead));
            cells.push(FORECAST_QUANTITY.as_str().to_string());
            cells.push(fmt_value(mean));
            cells.extend(qs.iter().map(|q| fmt_value(*q)));
            cells.push(ys.len().to_string());
            tsv += &(cells.join("\t") + "\n");
        }
        written.push(crate::fit::predict::write_tsv(out_dir, FORECAST_TABLES_DIR, source, &tsv)?);
    }

    // Joint samples: per draw, `FORECAST_SAMPLES_PER_DRAW` particles evenly
    // strided; each sample's values across every target are one particle's.
    // Sample `s` is the same draw and particle index in every arm.
    let n = fc0.n_particles;
    let per = FORECAST_SAMPLES_PER_DRAW.min(n);
    let mut tsv = arm_header(&sweep).join("\t")
        + "\tsample\tchain\tdraw\tcloud_row\tparticle\tstream\tt_start\tt_stop\tperiods_ahead\tquantity\tvalue\n";
    for arm in arms {
        let lead = arm_cells(&arm.label, &sweep).join("\t");
        let mut sample = 0usize;
        for (d, r) in draw_refs.iter().enumerate().take(arm.fc.n_draws) {
            let (chain, draw) = match r.key {
                Some((c, i)) => (c.to_string(), i.to_string()),
                None => (String::new(), String::new()),
            };
            for j in 0..per {
                let p = j * n / per;
                for (k, t) in targets.iter().enumerate() {
                    tsv += &format!("{lead}\t{sample}\t{chain}\t{draw}\t{}\t{p}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                        r.cloud_row, t.stream, fmt_time(t.measures.start()), fmt_time(t.measures.end()),
                        t.periods_ahead, FORECAST_QUANTITY.as_str(), fmt_value(arm.fc.draws[k][d * n + p]));
                }
                sample += 1;
            }
        }
    }
    let path = out_dir.join(FORECAST_SAMPLES_FILE);
    std::fs::write(&path, tsv).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    written.push(path);
    if arms.len() > 1 {
        written.push(write_contrasts(arms, out_dir, draw_refs, &sweep)?);
    }
    Ok(written)
}

/// The `mode` cell of every row of `forecast/<stream>.tsv`.
const FORECAST_MODE: &str = crate::fit_store::PredictMode::Forecast.label();

/// What the forecast's bands and samples are of: the driver draws each
/// target's value from the observation model at each particle.
const FORECAST_QUANTITY: Quantity = Quantity::Observation;

/// The file of [`write_contrasts`].
pub(crate) const FORECAST_CONTRASTS_FILE: &str = "forecast-contrasts.tsv";

/// `forecast-contrasts.tsv` (D83): per scenario or sweep arm and target, the
/// arm's predictive mean, the fitted model's, their difference `delta` and
/// the paired Monte Carlo standard error of `delta`
/// ([`crate::fit::forecast_contrast`]); no band of paired differences.
fn write_contrasts(
    arms: &[ArmForecast],
    out_dir: &std::path::Path,
    draw_refs: &[DrawRef],
    sweep: &[String],
) -> Result<PathBuf, String> {
    use crate::fit::forecast_contrast::{paired_mean_difference, per_draw_differences};
    let base = &arms[0].fc;
    let n = base.n_particles;
    let chains: Vec<Option<usize>> = draw_refs.iter().take(base.n_draws).map(|r| r.key.map(|k| k.0)).collect();
    let mean = |ys: &[f64]| ys.iter().sum::<f64>() / ys.len() as f64;
    let mut tsv = arm_header(sweep).join("\t")
        + "\tstream\tt_start\tt_stop\tperiods_ahead\tlead\tmean\tmean_fitted\tdelta\tdelta_mcse\tess\tn_draws\tmcse_note\n";
    for arm in &arms[1..] {
        let lead = arm_cells(&arm.label, sweep).join("\t");
        for (k, t) in base.plan.targets().iter().enumerate() {
            let d = per_draw_differences(&arm.fc.draws[k], &base.draws[k], n);
            let r = paired_mean_difference(&d, &chains);
            let (se, note) = match &r.mcse {
                Ok(se) => (fmt_value(*se), String::new()),
                Err(why) => (String::new(), why.describe()),
            };
            tsv += &format!("{lead}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{se}\t{}\t{}\t{note}\n",
                t.stream, fmt_time(t.measures.start()), fmt_time(t.measures.end()), t.periods_ahead, fmt_time(t.lead),
                fmt_value(mean(&arm.fc.draws[k])), fmt_value(mean(&base.draws[k])), fmt_value(r.delta),
                if r.ess.is_finite() { fmt_value(r.ess) } else { String::new() }, d.len());
        }
    }
    let path = out_dir.join(FORECAST_CONTRASTS_FILE);
    std::fs::write(&path, tsv).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

/// What `forecast.json` is written from, beside the forecast itself.
struct ManifestInputs<'a> {
    record: &'a crate::run_meta::TrainingPartitionRecord,
    reach_as_written: &'a str,
    skipped: &'a [SkippedStream],
    failures: &'a [DeterministicFailure],
    draw_refs: &'a [DrawRef],
    /// The `--design` files, by content digest.
    design_files: &'a [DesignFile],
    out_dir: &'a std::path::Path,
    /// The key of `out_dir` ([`crate::fit::predict::PredictOptions::address_key`]).
    address_key: &'a str,
    /// The options `address_key` hashes, with the design inputs (D81).
    options: &'a crate::fit::predict::PredictOptions,
    /// The statements of the table forcings read after the origin.
    covariates: &'a Covariates,
}

/// `forecast.json`: what the forecast predicted and what conditioned it.
fn write_manifest(arms: &[ArmForecast], job: &ForecastJob, m: ManifestInputs) -> Result<PathBuf, String> {
    let ManifestInputs {
        record, reach_as_written, skipped, failures, draw_refs, design_files, out_dir, address_key, options,
        covariates,
    } = m;
    let fc = &arms[0].fc;
    let point = |t: f64| serde_json::json!({
        "t": t,
        "date": job.calendar.date_of(t).map(ir::caltime::RenderedInstant::into_string),
    });
    let targets: Vec<serde_json::Value> = fc.plan.targets().iter().map(|t| {
        let (kind, start) = match t.measures {
            Measures::Period(p) => ("period", p.start()),
            Measures::Instant(x) => ("instant", x),
        };
        serde_json::json!({
            "stream": t.stream,
            "measures": kind,
            "start": point(start),
            "end": point(t.measures.end()),
            "periods_ahead": t.periods_ahead,
            "lead": t.lead,
            // D81: the design values the prediction read; absent for a
            // stream whose likelihood reads none.
            "design": (!t.design.is_empty()).then(|| t.design.iter()
                .map(|(c, v)| (c.clone(), serde_json::json!(v)))
                .collect::<serde_json::Map<String, serde_json::Value>>()),
        })
    }).collect();
    let model_end = job.model.simulation.t_end;
    let per_draw = FORECAST_SAMPLES_PER_DRAW.min(fc.n_particles);
    let mut doc = serde_json::json!({
        "schema": FORECAST_SCHEMA,
        "run_id": job.run_id,
        "address_key": address_key,
        "options": options,
        "calendar": job.calendar.to_json(),
        "mode": "fixed_origin",
        "treatment": job.posterior.treatment().label(),
        "method": job.posterior.method.map(|m| m.as_str()),
        "origin": point(fc.plan.origin()),
        "horizon": { "as_written": reach_as_written, "duration": fc.plan.horizon() },
        // D49: the run goes to the latest target end, past the model's
        // `simulate { to }` when that is earlier, with interventions planned
        // to that end; nothing else is extended.
        "run_extent": {
            "model_end": point(model_end),
            "run_end": point(fc.plan.run_end().max(model_end)),
            "extended": fc.plan.run_end() > model_end,
        },
        "training_rows": record,
        "conditioning_rows": {
            "conditioning": "filtered",
            "through": point(fc.state.through()),
            "streams": fc.state.rows().iter().map(|p| serde_json::json!({
                "stream": p.stream, "rows": p.rows, "digest": p.digest.to_hex(),
            })).collect::<Vec<_>>(),
        },
        "target_sources": fc.plan.sources().iter()
            .map(|(s, src)| serde_json::json!({ "stream": s, "source": src.describe() }))
            .collect::<Vec<_>>(),
        // D76: every stream this forecast covers is in exactly one of these (all
        // the declared streams, or those --stream names), so a reader
        // need not infer completeness from the exit status. `produced` holds
        // the streams with targets, `skipped` the ones left out without
        // failing the forecast (each with its reason), `failed` the ones that
        // make its exit status 1.
        "produced": fc.plan.sources().iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(),
        "skipped": skipped,
        "failed": crate::fit::failures::failed_streams(failures),
        "n_draws": fc.n_draws,
        // The posterior draws run, in the order of `forecast-samples.tsv`:
        // each one's row of draws.tsv and its (chain, draw) key there.
        "draws": draw_refs.iter().map(|r| serde_json::json!({
            "cloud_row": r.cloud_row,
            "chain": r.key.map(|k| k.0),
            "draw": r.key.map(|k| k.1),
        })).collect::<Vec<_>>(),
        "n_particles": fc.n_particles,
        // The filters' particle count per draw, as `{"n": N}`.
        "particles": job.particles,
        // D85: per draw, in the order of `draws`, the effective sample size
        // of its filter's weights at its last assimilation before the origin,
        // before resampling, with that time and ESS/N. Every arm shares it.
        // It characterises the Monte Carlo reliability of the state the
        // forecast starts from; it is not a count of distinct ancestors and
        // says nothing of the forecast's accuracy.
        "filter_diagnostics": {
            "n_particles": fc.n_particles,
            "last_assimilation": fc.last_assimilation.iter().zip(draw_refs).map(|(a, r)| serde_json::json!({
                "cloud_row": r.cloud_row,
                "t": a.map(|a| a.time),
                "ess": a.map(|a| a.ess),
                "ess_over_n": a.map(|a| a.ess / fc.n_particles as f64),
            })).collect::<Vec<_>>(),
            "distinct_parents_at_final_resampling": serde_json::Value::Null,
            "distinct_parents_note": "not recorded: the bootstrap filter returns its resampling \
                indices only when it records the full ancestry, which the forecast does not",
        },
        "binomial": fc.binomial,
        // What `forecast-samples.tsv` holds, and what the bands are over.
        "samples": {
            "file": FORECAST_SAMPLES_FILE,
            "per_draw": per_draw,
            "of_particles": fc.n_particles,
            "description": format!(
                "{per_draw} joint samples per posterior draw, the particles j·{n}/{per_draw} \
                 (j = 0…{last}) of its {n}; each sample is one particle's trajectory across \
                 every target. The bands and means in forecast/ use all {n} particles of \
                 every draw.",
                n = fc.n_particles, last = per_draw.saturating_sub(1)),
        },
        // D50: the fitted model, then each scenario and sweep arm, in the
        // order of the tables' rows. Every arm ran the same draws with the
        // same seeds and shares the baseline's filter through the training
        // rows; `first_difference` is the earliest time an arm acts
        // differently, at or after the origin (null when nothing it changes
        // acts within the run).
        "arms": arms.iter().map(|a| serde_json::json!({
            "scenario": a.label.scenario,
            "sweep": a.label.sweep.iter().map(|(p, v)| (p.clone(), serde_json::json!(v)))
                .collect::<serde_json::Map<String, serde_json::Value>>(),
            "baseline": a.label.is_baseline(),
            "first_difference": a.difference.as_ref().and_then(|d| d.first).map(point),
            "interventions_changed": a.difference.as_ref().map(|d| d.interventions.clone()).unwrap_or_default(),
            "parameters_changed": a.difference.as_ref().map(|d| d.parameters.clone()).unwrap_or_default(),
        })).collect::<Vec<_>>(),
        "targets": targets,
        // D81: where the design-reading streams' design values came from, and
        // what the forecast therefore is; null when no target reads one.
        "design": fc.plan.design_mode().map(|mode| serde_json::json!({
            "mode": mode,
            "label": mode.label(),
            "streams": fc.plan.targets().iter().filter(|t| !t.design.is_empty())
                .map(|t| t.stream.as_str()).collect::<std::collections::BTreeSet<_>>(),
            "files": design_files,
        })),
    });
    // The table forcings read after the origin, their statements and the
    // notes on piecewise forcings the reach passes; absent when there are
    // none.
    if let Some(block) = covariates.manifest(fc.plan.origin(), point) {
        doc["covariates"] = block;
    }
    // D86: the same `chain_selection` block `predictive.json` carries, absent
    // when the full cloud was used.
    if let Some(info) = job.posterior.selection() {
        doc["chain_selection"] = info.to_json();
    }
    // D86: the chains left out of the cloud are always stated, `[]` for the
    // full cloud. The address key hashes the options with the field absent
    // in that case, so the full cloud's key is unchanged.
    doc["options"]["excluded_chains"] =
        serde_json::json!(options.excluded_chains.clone().unwrap_or_default());
    // The engine that made the forecast, as `run.json` records it.
    crate::fit::predict::stamp_engine(&mut doc);
    // D83: with more than one arm, how the arms' random numbers are coupled
    // and what `forecast-contrasts.tsv` holds.
    if arms.len() > 1 {
        // What the contrasts are taken against: the fitted model, or for a
        // fixed point the `(none)` arm, the point with no scenario.
        let base_arm = if job.baseline == crate::args::FITTED {
            "the fitted model's".to_string()
        } else {
            format!("the `{}` arm's", job.baseline)
        };
        doc["coupling"] = serde_json::json!({
            "method": "shared_seed_per_particle_streams",
            "description": format!("posterior draw m runs with the same seed in every arm, and every \
                particle draws from its own random stream derived from that seed, so the arms \
                share the filter through the training rows (checked bit for bit) and particle \
                i of draw m starts every arm's forecast from the same state. After the origin \
                an arm's streams stay aligned with {base_arm} only until its dynamics \
                differ; how strongly the arms are coupled after that is a property of the \
                implementation, not of the scenarios."),
        });
        doc["contrasts"] = serde_json::json!({
            "file": FORECAST_CONTRASTS_FILE,
            "baseline": job.baseline,
            "delta": format!("the mean of the arm's predictive draws at the target minus {base_arm} \
                (all particles of all draws)"),
            "delta_mcse": {
                "kind": "monte_carlo_standard_error",
                "description": "the Monte Carlo standard error of delta: how far it would move \
                    were the forecast rerun with other random numbers and other posterior draws. \
                    It is not the predictive uncertainty of either arm, nor the uncertainty of \
                    the scenario's effect.",
                "method": format!("per posterior draw, the arm's mean over its particles minus \
                    {base_arm}; per chain, the sample variance of those differences over the \
                    effective sample size of a variant of Geyer's initial-positive-sequence \
                    estimator (the chain's draws in order); chains combined as \
                    independent, weighted by their share of the draws"),
                "min_ess_per_chain": crate::fit::forecast_contrast::CONTRAST_MIN_ESS,
            },
            "paired_band": "none: a band of paired differences A[s] − B[s] would be set by the \
                coupling of the arms' random numbers, which the scenario models do not identify",
        });
    }
    let path = out_dir.join(FORECAST_MANIFEST);
    let text = serde_json::to_string_pretty(&doc).map_err(|e| format!("serializing forecast.json: {e}"))?;
    std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::forecast_driver::ForecastFailure;

    /// A failure names the posterior draw by its row of draws.tsv and its
    /// (chain, draw) key, not by its place among the parameter sets the
    /// driver was handed.
    #[test]
    fn a_failure_names_the_posterior_draw_by_its_key() {
        let refs = [DrawRef { cloud_row: 0, key: Some((0, 3)) }, DrawRef { cloud_row: 40, key: Some((1, 7)) }];
        let f = ForecastFailure { draw: Some(1), time: Some(26.0), cause: "particle 4 died".into() };
        let msg = f.describe(|d| refs[d].describe());
        assert!(msg.contains("posterior draw (chain 1, draw 7), row 40 of draws.tsv at t = 26"), "{msg}");
        let unkeyed = DrawRef { cloud_row: 12, key: None };
        assert_eq!(unkeyed.describe(), "posterior draw at row 12 of draws.tsv");
    }

    /// The fit's sampler is the run's record, else the method's own.
    #[test]
    fn the_fits_sampler_is_read_from_the_leaf() {
        use sim::rng::BinomialAlgorithm::{Btpe, Btrs};
        let leaf = |json: &str| {
            let dir = crate::test_support::unique_temp_dir("fit_binomial");
            std::fs::create_dir_all(&dir).unwrap();
            crate::fit_store::write_test_inputs(&dir, &serde_json::from_str::<serde_json::Value>(json).unwrap());
            dir
        };
        let recorded = leaf(r#"{"inputs": {"binomial": "btrs"}}"#);
        assert_eq!(fit_binomial(&recorded, Some(Btpe)).unwrap(), Btrs, "the record wins over fit.toml");
        let plain = leaf(r#"{"inputs": {}}"#);
        assert_eq!(fit_binomial(&plain, Some(Btrs)).unwrap(), Btrs, "the method's own");
        assert!(fit_binomial(&plain, None).is_err());
        let bad = leaf(r#"{"inputs": {"binomial": "fast"}}"#);
        assert!(fit_binomial(&bad, Some(Btpe)).is_err());
    }
}
