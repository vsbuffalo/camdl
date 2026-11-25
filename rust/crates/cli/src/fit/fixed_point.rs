//! `camdl predict MODEL`: a prediction from one point given on the command
//! line, `ParameterSource::FixedPoint`
//! (`docs/dev/proposals/2026-11-20-run-store-api.md`, "`camdl predict` from a
//! fixed point").
//!
//! [`resolve`] poses the prediction as a fit problem with nothing estimated
//! would pose it, and stores what its identity depends on before anything is
//! computed: the model compiled and rewritten from the data (its anchored
//! times, its bound tables, `--integrator`), and each bound stream's
//! conditioned lines, cut by the rule a fit problem's `inputs/` are
//! (`fit_store::resolve_problem`). Every row is parsed to find the cut; a
//! held-out row is then neither stored nor used: the problem the prediction
//! runs on binds copies of the conditioned lines, so every row it reads
//! trained.
//!
//! The point is resolved by the parameter resolver as `pfilter` resolves it
//! (`docs/camdl-run-spec.md` §1.3): the model's values, each `--params` file
//! in order, then each `--param`. A scenario arm applies on top of it, as it
//! applies on top of a posterior draw.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use store::{ContentArtifact, Dep, Finite};

use crate::camdl_store::{CompiledModel, DataTable, Quantities, ResolvedParams};
use crate::fit::config_v2::{DataSpec, FitBackendConfig, FixedParams, ModelRef, Problem, TimeSpecToml};
use crate::fit_store::Numerics;

/// A fixed-point prediction resolved against its model and data, with what
/// its identity depends on stored.
pub(crate) struct FixedPoint {
    /// The root of the store the prediction is stored in.
    pub store_root: PathBuf,
    pub model: Dep<CompiledModel>,
    /// The compile's quantities and contrasts, which the prediction
    /// evaluates.
    pub quantities: Dep<Quantities>,
    pub params: ResolvedParams,
    /// The point, every model parameter's value.
    pub point: IndexMap<String, f64>,
    pub numerics: Numerics,
    /// The simulator the prediction runs on, resolved under the fitting rule.
    pub simulator: crate::simulator::Simulator,
    pub trained_through: Option<Finite>,
    /// Each bound stream's conditioned lines, by the source they are bound
    /// to.
    pub data: BTreeMap<String, Dep<DataTable>>,
    /// The problem the prediction runs on: `[data]` binds copies of the
    /// conditioned lines and holds nothing out.
    pub problem: Problem,
    /// The keyed compile as an IR file, which every cell of the engine loads.
    pub model_ir: PathBuf,
    /// The model file given, which a `--quantities` vocabulary compiles
    /// against.
    pub model_source: PathBuf,
    /// The model file's stem.
    pub model_name: String,
    /// Which rows trained, cut by `--holdout-after`.
    pub partition: crate::run_meta::TrainingPartitionRecord,
    _scratch: tempfile::TempDir,
}

/// Whether `source`, `camdl predict`'s positional argument, names a model
/// file rather than a stored fit: a `.camdl` or `.ir.json` path, as
/// `simulate` and `pfilter` accept.
pub(crate) fn names_a_model(source: &str) -> bool {
    source.ends_with(".camdl") || source.ends_with(".ir.json")
}

/// Resolve the fixed-point prediction of the model at `model_path` that
/// `flags` pose, storing the compiled model, its quantities and each stream's
/// conditioned lines in the store `request` names (`--store`, else
/// `CAMDL_STORE`, else `./camdl-store`), under `info`. `reposed` is the
/// presentation of a model file camdl wrote itself from a stored prediction,
/// which the compile is stored with; `None` for a model file the user gave,
/// whose own presentation is rendered and stored.
#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
#[expect(clippy::disallowed_methods, reason = "reads the model file `source` names (declared `#[input(identity)]` on `PredictArgs`) and the IR camdlc compiled from it, stored as the `CompiledModel` the prediction's identity holds, and writes scratch copies of the stored conditioned lines for the data loader")]
pub(crate) fn resolve(
    out: &crate::out::Out,
    model_path: &str,
    flags: &crate::args::FixedPointFlags,
    allow_forcing_extrapolation: bool,
    compile: crate::util::CompileSettings,
    info: &store::RunInfo,
    request: &store::StoreRequest,
    reposed: Option<crate::camdl_store::Presentation>,
) -> Result<FixedPoint, String> {
    use crate::args::types::ForwardBackend;
    if flags.data.is_empty() {
        return Err("a prediction from a model file needs its observations: --data PATH (one stream), or \
                    --data NAME=PATH for each stream. The streams it predicts, and the rows its bands are \
                    drawn at, are the bound streams' rows; a free-forward run with no data is \
                    `camdl simulate --obs`."
            .into());
    }
    if !Path::new(model_path).is_file() {
        return Err(format!("cannot read the model {model_path}: it does not exist"));
    }
    let scratch = tempfile::Builder::new()
        .prefix("camdl-predict-point-")
        .tempdir()
        .map_err(|e| format!("cannot make a scratch directory: {e}"))?;

    // The compile, and the streams `--data` binds.
    let (ir_path, _ir_tmp) = crate::util::resolve_ir_path(model_path, false, None, compile)?;
    let (model_in, _) = crate::util::load_ir(&ir_path)?;
    let leaf_names: Vec<String> = model_in.observations.iter().map(|o| o.name.clone()).collect();
    let bound = crate::util::resolve_data_specs(&flags.data, &leaf_names, flags.obs.as_deref())?;
    let observations = bind_by_source(&model_in, &bound)?;
    let (simulator, _) = crate::simulator::Simulator::resolve(
        crate::simulator::ModelSimulate::of(&model_in.simulation),
        crate::simulator::WrittenSimulator {
            backend: flags.backend,
            dt: flags.dt,
            config_dt: None,
            integrator: flags.integrator,
            binomial: flags.binomial,
        },
        crate::simulator::Basis::Fitting {
            backend: ForwardBackend::ChainBinomial,
            run: "this prediction",
            file: crate::dt_rule::FitFile::Flag,
        },
    )?;
    let dt = simulator.dt().get();

    let holdout_after = flags.holdout_after.as_deref().map(|raw| match raw.trim().parse::<f64>() {
        Ok(v) => TimeSpecToml::Num(v),
        Err(_) => TimeSpecToml::Spec(raw.trim().to_string()),
    });
    let problem = Problem {
        model: ModelRef { camdl: model_path.to_string() },
        data: Some(DataSpec {
            file: None,
            observations,
            holdout_after,
            holdout: None,
            tables: flags.table.iter().map(|t| (t.name.clone(), t.path.to_string_lossy().into_owned())).collect(),
        }),
        synthetic: None,
        simplex_groups: Vec::new(),
        store: None,
        estimate: IndexMap::new(),
        fixed: FixedParams { from_file: None, from_scenario: None, values: IndexMap::new() },
        config: FitBackendConfig { dt, allow_degenerate_rates: flags.allow_degenerate_rates, allow_forcing_extrapolation },
        simulate: None,
        scenario: None,
        enable: Vec::new(),
        disable: Vec::new(),
        ic_free: None,
        provenance: None,
        compiled_ir: None,
        compiled_digest: None,
    };
    if let Ok(data) = problem.data_spec() {
        crate::fit::stream_bindings::StreamBindings::resolve(&model_in, data)?;
    }
    check_time_format(&model_in, &bound, dt, flags.time_format)?;

    // The model rewritten from the data, as `fit run` rewrites it, then given
    // the simulator's integrator, as `simulate` gives it: the compile the
    // identity holds and the engine runs.
    let ir_path = crate::fit::resolve_anchors_into_temp_ir(Some(out), &ir_path, &problem)?;
    let (ir_path, _tables) = crate::fit::bind_tables_into_temp_ir(&ir_path, &problem)
        .map_err(|e| format!("--table: {e}"))?;
    let mut envelope = crate::util::load_ir_envelope(&ir_path)?;
    simulator.apply_to(&mut envelope.model);
    let model = envelope.model.clone();
    let model_ir = scratch.path().join("model.ir.json");
    std::fs::write(&model_ir, serde_json::to_string(&envelope).map_err(|e| e.to_string())?)
        .map_err(|e| format!("cannot write {}: {e}", model_ir.display()))?;

    // The point, resolved before anything is stored, so a refused point stores
    // nothing.
    let point = resolve_point(out, &model, flags)?;
    let params = ResolvedParams::new(point.iter().map(|(k, v)| (k.clone(), *v)))?;

    let store = crate::camdl_store::open(request)
        .map_err(|e| format!("the run store at {}: {e}", crate::camdl_store::root_of(request).display()))?;
    let source_bytes = std::fs::read(model_path).map_err(|e| format!("cannot read {model_path}: {e}"))?;
    let model_name = crate::hashing::path_stem_slug(model_path).unwrap_or_else(|| "model".to_string());
    let parts = crate::camdl_store::CompiledParts::split(
        envelope,
        model_name.clone(),
        match reposed {
            None => crate::camdl_store::ModelSource::read(&source_bytes, Path::new(model_path)),
            Some(_) => crate::camdl_store::ModelSource {
                bytes: &source_bytes,
                path: Path::new(model_path),
                rendered: crate::camdl_store::Rendered::nothing(),
            },
        },
    )?;
    let crate::camdl_store::CompiledParts { model: model_part, quantities, presentation, labels: model_labels, provenance } = parts;
    let presentation = reposed.unwrap_or(crate::camdl_store::Presentation::New(presentation));
    let quantities = store
        .put::<Quantities>(quantities, &model_labels, &store::Provenance::none(), info)
        .and_then(|s| s.dep(Quantities::FILE))
        .map_err(|e| format!("storing the quantities: {e}"))?;
    let model_dep = crate::camdl_store::put_compiled(&store, info, model_part, presentation, &model_labels, &provenance)
        .map_err(|e| format!("storing the compiled model: {e}"))?;

    // The conditioned lines of each bound file, cut by the fit problem's rule.
    let resolved = crate::fit_store::resolve_problem(
        &problem,
        &model,
        model_dep.clone(),
        crate::fit_store::ModelSourceFile { bytes: source_bytes, camdl: crate::util::model_is_camdl_source(model_path) },
        crate::fit_store::DataOrigin::Files,
    )?;
    let trained_through = match &resolved.identity.partition.rule {
        crate::fit_store::PartitionRule::Cutoff { at } => Some(*at),
        _ => None,
    };
    let mut data = BTreeMap::new();
    let mut conditioned: IndexMap<String, String> = IndexMap::new();
    let given = &problem.data.as_ref().expect("posed with [data]").observations;
    for (source, file) in &resolved.identity.training {
        let labels = crate::camdl_store::TableLabels {
            stream: source.clone(),
            path: given.get(source).cloned().unwrap_or_default(),
        };
        let dep = store
            .put::<DataTable>(file.content.clone(), &labels, &store::Provenance::none(), info)
            .and_then(|s| s.dep(DataTable::FILE))
            .map_err(|e| format!("storing the data table of `{source}`: {e}"))?;
        data.insert(source.clone(), dep);
        let copy = scratch.path().join("data").join(format!("{source}.{}", file.ext));
        std::fs::create_dir_all(copy.parent().expect("a file in the scratch directory")).map_err(|e| e.to_string())?;
        std::fs::write(&copy, file.bytes()).map_err(|e| format!("cannot write {}: {e}", copy.display()))?;
        conditioned.insert(source.clone(), copy.to_string_lossy().into_owned());
    }
    let mut posed = problem.clone();
    let d = posed.data.as_mut().expect("posed with [data]");
    d.observations = conditioned;
    d.holdout_after = None;

    let numerics = Numerics {
        backend: crate::sim_store::Backend::of(simulator.backend()),
        dt: simulator.dt(),
        binomial: simulator.binomial().into(),
        allow_degenerate_rates: flags.allow_degenerate_rates,
    };
    Ok(FixedPoint {
        store_root: store.root().to_path_buf(),
        model: model_dep,
        quantities,
        params,
        point,
        numerics,
        simulator,
        trained_through,
        data,
        problem: posed,
        model_ir,
        model_source: PathBuf::from(model_path),
        model_name,
        partition: resolved.partition().clone(),
        _scratch: scratch,
    })
}

/// `--data`'s bindings, one leaf each, as the source each leaf reads and the
/// file bound to it: a stream is bound as a whole, so every leaf of a source
/// must be bound, and to one file.
fn bind_by_source(model: &ir::Model, bound: &[(String, PathBuf)]) -> Result<IndexMap<String, String>, String> {
    let mut by_source: IndexMap<String, String> = IndexMap::new();
    for (leaf, path) in bound {
        let obs = model
            .observations
            .iter()
            .find(|o| &o.name == leaf)
            .ok_or_else(|| format!("internal error: --data bound `{leaf}`, which the model does not declare"))?;
        let path = path.to_string_lossy().into_owned();
        match by_source.get(&obs.source) {
            Some(p) if *p != path => {
                return Err(format!(
                    "--data binds the leaves of stream '{}' to two files ({p} and {path}); a stream is bound to \
                     one file: --data {}=FILE",
                    obs.source, obs.source
                ))
            }
            _ => {
                by_source.insert(obs.source.clone(), path);
            }
        }
    }
    for o in &model.observations {
        if by_source.contains_key(&o.source) && !bound.iter().any(|(l, _)| l == &o.name) {
            return Err(format!(
                "--data binds part of stream '{}' and not its leaf '{}'; a stream is bound as a whole: --data {}=FILE",
                o.source, o.name, o.source
            ));
        }
    }
    Ok(by_source)
}

/// Refuse a `--time-format` that a bound file's time column does not hold;
/// this check is all the flag does. Every later read detects the format from
/// the whole column, which agrees with any format the column holds, so the
/// flag changes no result, and it does not silence W326, which those reads
/// give for a numeric column under a model with an `origin`.
fn check_time_format(
    model: &ir::Model,
    bound: &[(String, PathBuf)],
    dt: f64,
    format: crate::caltime_load::TimeFormat,
) -> Result<(), String> {
    log::debug!("predict: --data checked under --time-format {format:?}");
    if format == crate::caltime_load::TimeFormat::Auto {
        return Ok(());
    }
    let opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format,
    };
    for (leaf, path) in bound {
        let obs = model.observations.iter().find(|o| &o.name == leaf).expect("a bound leaf is declared");
        let siblings: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().filter(|o| o.source == obs.source).collect();
        crate::fit::runner::load_observations(&path.to_string_lossy(), obs, &siblings, &opts)
            .map_err(|e| format!("--data {leaf}: {e}"))?;
    }
    Ok(())
}

/// The point: every parameter of `model` after the model's own values, each
/// `--params` file and each `--param`. A parameter with no value after every
/// source is refused, and so is a `--param` the model does not declare.
fn resolve_point(
    out: &crate::out::Out,
    model: &ir::Model,
    flags: &crate::args::FixedPointFlags,
) -> Result<IndexMap<String, f64>, String> {
    use crate::params_resolver::{emit_notices, resolve_parameters, FlagStyle, ParameterInputs};
    let fixed_cli: Vec<(String, f64)> = flags.param.iter().map(|p| (p.name.clone(), p.value)).collect();
    // `model` has its tables bound already.
    let external = crate::external_inputs::ExternalBindings::default();
    let resolved = resolve_parameters(ParameterInputs {
        model,
        scenario: None,
        adhoc_enable: &[],
        adhoc_disable: &[],
        scenario_inline_name: None,
        scenario_inline_set: &[],
        scenario_inline_scale: &[],
        point_overrides: &[],
        fixed_cli: &fixed_cli,
        fixed_files: &flags.params,
        swept: &[],
        fit_toml_fixed: &IndexMap::new(),
        fit_toml_estimate: &indexmap::IndexSet::new(),
        external: &external,
    })
    .map_err(|e| e.with_flags(FlagStyle::Params).to_string())?;
    emit_notices(out, &[&resolved]);
    let mut point = IndexMap::new();
    let mut missing = Vec::new();
    for p in &resolved.model.parameters {
        match p.value.resolved_value() {
            Some(v) => {
                point.insert(p.name.clone(), v);
            }
            None => missing.push(p.name.as_str()),
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "parameter(s) {} have no value after the model, --params and --param; give each one, in a \
             --params file or as --param NAME=VALUE",
            missing.join(", ")
        ));
    }
    Ok(point)
}
