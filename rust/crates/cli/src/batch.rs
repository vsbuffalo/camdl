//! `batch run FILE` subcommand — the multi-scenario / sweep runner
//! behind `camdl batch run`. Writes content-addressed output under
//! `<output_dir>/sims/…` in the same factored `runid` layout `--cas` uses
//! for single runs (model / config / params / scenario / seed levels), so
//! `camdl list/show/cat` browse both uniformly.
//!
//! ## Schema note: batch TOML is v1
//!
//! The field names used here (`[config]`, `[[scenario]]`, `[sweep]`,
//! `[design.*]`) are standalone and pre-date the v2 run-system types
//! (`SimulateJob`, `SweepSpec`, `Seeds` in `fit/config_v2.rs`). A future
//! version will align the schema with v2 for consistency across the
//! single-run and batch paths.
//!
//! External tooling should NOT assume the current field names survive
//! unchanged. If you're building tooling against this schema and need
//! a migration window, open an issue.

use std::collections::HashMap;
use serde::Deserialize;

use crate::util::{load_params_toml, resolve_ir_path};
use crate::sampling::{generate_design, describe_prior, DesignParam};
use ir::parameter::PriorDist;
use crate::cas_sink::{cells_progress, CasSink, ResolvedEntry};

// gh#audit-H13: build a SCOPED local rayon pool from `--parallel` and run the
// engine (sweep or design) inside `pool.install(...)`. The earlier code used
// `rayon::ThreadPoolBuilder::new().num_threads(parallel).build_global()`, but by
// the time these commands run the global pool is already initialised, so
// `build_global` returned AlreadyInitialized (swallowed by `let _ = …`) and the
// default all-core pool ran regardless of `--parallel`. A scoped pool is
// order-independent: nested rayon work (the engine's `into_par_iter`) inherits
// the surrounding pool. A value of 0 means "use rayon's default" (all logical
// cores) — leave the pool unset and run on the global pool. Same shape as
// pfilter.rs / profile.rs / survey.rs (f7bde701).
fn build_parallel_pool(parallel: usize) -> Option<rayon::ThreadPool> {
    if parallel > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build()
                .unwrap_or_else(|e| {
                    eprintln!("error: failed to build thread pool (--parallel {}): {}", parallel, e);
                    std::process::exit(1);
                }),
        )
    } else {
        None
    }
}

/// Run a closure on the scoped pool if one was built, else on the global pool
/// (parallel == 0). Generic over the closure so borrowed captures keep it
/// `Send` for `ThreadPool::install`.
fn run_pooled<R: Send>(pool: &Option<rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

// ─── TOML schema (v1 — see module-level doc) ─────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)] // gh#241 G3: a typo'd batch.toml key must error, not silently drop
struct ExperimentToml {
    config: ConfigSection,
    #[serde(default)]
    scenario: Vec<ScenarioEntry>,
    #[serde(default)]
    sweep: HashMap<String, SweepSpec>,
    #[serde(default)]
    design: HashMap<String, DesignBlock>,
    #[serde(default)]
    obs: ObsSection,
    /// gh#156: trajectory output view (`--output-every` / `--no-flows` /
    /// `--columns`) applied to every cell — the same shared clap+serde struct
    /// the `simulate` CLI flattens.
    #[serde(default)]
    output: crate::args::OutputView,
}

/// `[obs]` section — synthetic observation output for the batch ensemble.
/// `enabled = true` samples each run's observation streams and writes them
/// into the CAS obs subtree (`seed_N/obs/{obs_hash}-{obs_seed}/<stream>.tsv`,
/// the designed layout from `cas/mod.rs`). Resolves CLI review finding #4.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ObsSection {
    #[serde(default)]
    enabled: bool,
}

// ─── Design specification ─────────────────────────────────────────────────────

/// A named experimental design block (`[design.NAME]`).
/// Represents a named belief state: parameter ranges + sampling method.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DesignBlock {
    method: String,   // "sobol" | "lhs" | "random"
    n: usize,
    #[serde(default)]
    parameters: HashMap<String, DesignParamToml>,
}

/// Per-parameter specification within a design block.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DesignParamToml {
    range: RangeMinMax,
    #[serde(default)]
    transform: Option<String>,   // "log" | "logit" | None (linear)
    #[serde(default)]
    prior: Option<PriorDist>,    // prior distribution for VOI importance weighting
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeMinMax {
    min: f64,
    max: f64,
}

// ─── Sweep specification ─────────────────────────────────────────────────────

/// One swept parameter's value specification.
/// TOML forms:
///   vacc_eff = [0.1, 0.3, 0.5]
///   vacc_eff = { linspace = { min = 0.1, max = 0.9, n = 9 } }
///   kappa    = { logspace = { min = 0.001, max = 0.1, n = 5 } }
///   R0       = { range = { min = 1.0, max = 5.0, step = 0.5 } }
///
/// Parsed by hand ([`SweepSpec::try_from`]) rather than as an untagged enum,
/// so a spec is refused when it is read rather than expanded wrongly: a key its
/// form does not have, a `range` written as an array (which filled `min`/`max`
/// and dropped a `step` written beside it, gh#594), and every value that would
/// expand to no points, to NaN, or — a `step` that is not positive — to an
/// unbounded loop.
#[derive(Debug, Deserialize)]
#[serde(try_from = "toml::Value")]
enum SweepSpec {
    List(Vec<f64>),
    Linspace { linspace: LinspaceSpec },
    Logspace  { logspace: LinspaceSpec },
    Range     { range: RangeSpec },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LinspaceSpec {
    min: f64,
    max: f64,
    n: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RangeSpec {
    min: f64,
    max: f64,
    #[serde(default = "default_step")]
    step: f64,
}

fn default_step() -> f64 { 1.0 }

const SWEEP_FORMS: &str = "a list `[a, b, …]`, `{ linspace = { min, max, n } }`, \
    `{ logspace = { min, max, n } }`, or `{ range = { min, max, step } }`";

impl TryFrom<toml::Value> for SweepSpec {
    type Error = String;

    fn try_from(v: toml::Value) -> Result<Self, String> {
        use toml::Value;
        let finite = |what: &str, x: f64| -> Result<f64, String> {
            if x.is_finite() { Ok(x) } else { Err(format!("{what} must be finite, got {x}")) }
        };
        let spec = match v {
            Value::Array(items) => {
                if items.is_empty() {
                    return Err("an empty list sweeps no values, so the batch would \
                                run no cells".to_string());
                }
                let vals = items.iter()
                    .map(|x| match x {
                        Value::Float(f) => finite("a list value", *f),
                        Value::Integer(i) => Ok(*i as f64),
                        other => Err(format!("a list value must be a number, got {other}")),
                    })
                    .collect::<Result<Vec<f64>, String>>()?;
                SweepSpec::List(vals)
            }
            Value::Table(t) => {
                let mut keys = t.keys();
                let (Some(form), None) = (keys.next(), keys.next()) else {
                    return Err(format!(
                        "a sweep spec is one of {SWEEP_FORMS}; this table has the keys \
                         {:?}", t.keys().collect::<Vec<_>>()));
                };
                let body = t[form].clone();
                if !body.is_table() {
                    return Err(format!(
                        "`{form}` takes a table ({SWEEP_FORMS}), not {}", body.type_str()));
                }
                match form.as_str() {
                    "linspace" | "logspace" => {
                        let s: LinspaceSpec = body.try_into()
                            .map_err(|e| format!("`{form}`: {e}"))?;
                        finite(&format!("`{form}` min"), s.min)?;
                        finite(&format!("`{form}` max"), s.max)?;
                        if s.n == 0 {
                            return Err(format!("`{form}` n = 0 sweeps no values, so the \
                                                batch would run no cells"));
                        }
                        if form == "logspace" && (s.min <= 0.0 || s.max <= 0.0) {
                            return Err(format!(
                                "`logspace` min and max must be positive (it spaces their \
                                 logarithms), got min = {}, max = {}", s.min, s.max));
                        }
                        if form == "linspace" {
                            SweepSpec::Linspace { linspace: s }
                        } else {
                            SweepSpec::Logspace { logspace: s }
                        }
                    }
                    "range" => {
                        let s: RangeSpec = body.try_into().map_err(|e| format!("`range`: {e}"))?;
                        finite("`range` min", s.min)?;
                        finite("`range` max", s.max)?;
                        if !s.step.is_finite() || s.step <= 0.0 {
                            return Err(format!(
                                "`range` step must be a positive, finite number, got {} — \
                                 a step that does not advance from min never reaches max",
                                s.step));
                        }
                        if s.max < s.min {
                            return Err(format!(
                                "`range` max ({}) is below min ({}), so it sweeps no values",
                                s.max, s.min));
                        }
                        SweepSpec::Range { range: s }
                    }
                    other => return Err(format!(
                        "unknown sweep form `{other}`; a sweep spec is one of {SWEEP_FORMS}")),
                }
            }
            other => return Err(format!(
                "a sweep spec is one of {SWEEP_FORMS}, not {}", other.type_str())),
        };
        Ok(spec)
    }
}

impl SweepSpec {
    /// Expand to a concrete vector of values.
    fn expand(&self) -> Vec<f64> {
        match self {
            SweepSpec::List(v) => v.clone(),
            SweepSpec::Linspace { linspace: s } => {
                if s.n == 1 {
                    return vec![s.min];
                }
                (0..s.n).map(|i| s.min + (s.max - s.min) * i as f64 / (s.n - 1) as f64).collect()
            }
            SweepSpec::Logspace { logspace: s } => {
                if s.n == 1 {
                    return vec![s.min];
                }
                let log_min = s.min.ln();
                let log_max = s.max.ln();
                (0..s.n).map(|i| {
                    let t = i as f64 / (s.n - 1) as f64;
                    (log_min + (log_max - log_min) * t).exp()
                }).collect()
            }
            SweepSpec::Range { range: s } => {
                let mut v = Vec::new();
                let mut x = s.min;
                while x <= s.max + 1e-12 * s.step.abs() {
                    v.push(x);
                    x += s.step;
                }
                v
            }
        }
    }
}

/// Expand the full `[sweep]` section into a list of parameter override maps.
/// If no sweep keys are defined, returns a single empty map (one "null" point).
fn expand_sweep(sweep: &HashMap<String, SweepSpec>) -> Vec<HashMap<String, f64>> {
    if sweep.is_empty() {
        return vec![HashMap::new()];
    }
    // Sort keys for deterministic ordering
    let mut keys: Vec<&String> = sweep.keys().collect();
    keys.sort();
    let values: Vec<Vec<f64>> = keys.iter().map(|k| sweep[*k].expand()).collect();

    // Cartesian product via fold
    let mut points: Vec<HashMap<String, f64>> = vec![HashMap::new()];
    for (key, vals) in keys.iter().zip(values.iter()) {
        let mut next = Vec::with_capacity(points.len() * vals.len());
        for point in &points {
            for &v in vals {
                let mut p = point.clone();
                p.insert(key.to_string(), v);
                next.push(p);
            }
        }
        points = next;
    }
    points
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigSection {
    model: String,
    #[serde(default)]
    params: Option<String>,
    #[serde(default)]
    geo: Option<String>,
    #[serde(default = "default_backend")]
    backend: crate::args::types::ForwardBackend,
    #[serde(default = "default_dt")]
    dt: f64,
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_parallel")]
    parallel: usize,
    #[serde(default)]
    seeds: SeedsSection,
}

fn default_backend() -> crate::args::types::ForwardBackend {
    crate::args::types::ForwardBackend::ChainBinomial
}
fn default_dt() -> f64 { 1.0 }
fn default_output_dir() -> String { crate::run_paths::DEFAULT_OUTPUT_ROOT.to_string() }
fn default_parallel() -> usize { 1 }

use crate::util::model_is_camdl_source;

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SeedsSection {
    from: Option<u64>,
    to:   Option<u64>,
    list: Option<Vec<u64>>,
    n:    Option<u64>,
    start: Option<u64>,
}

impl SeedsSection {
    fn resolve(&self) -> Result<Vec<u64>, String> {
        if let Some(ref list) = self.list {
            return Ok(list.clone());
        }
        if let Some(n) = self.n {
            let start = self.start.unwrap_or(1);
            return Ok((start..start + n).collect());
        }
        if let (Some(from), Some(to)) = (self.from, self.to) {
            return Ok((from..=to).collect());
        }
        Ok(vec![1])
    }
}

/// Per-scenario specification as parsed from the experiment TOML.
#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ScenarioEntry {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, f64>,
    #[serde(default)]
    pub enable: Vec<String>,
    #[serde(default)]
    pub disable: Vec<String>,
}

/// Resolve every `[[scenario]]` entry against the model's preset names
/// using the shared [`crate::sim_job::resolve_scenario_ref`] semantics.
/// Errors (unknown name, preset+inline collision) are user-facing and
/// fatal — the batch must not silently run a mislabeled scenario.
fn resolve_batch_scenarios(
    entries: &[ScenarioEntry],
    model: &ir::Model,
) -> Result<Vec<ResolvedEntry>, String> {
    use crate::sim_job::{resolve_scenario_ref, ResolvedScenario, ScenarioRef};
    let preset_names: Vec<String> =
        model.presets.iter().map(|p| p.name.clone()).collect();

    entries
        .iter()
        .map(|sc| {
            let params_im: indexmap::IndexMap<String, f64> =
                sc.params.iter().map(|(k, v)| (k.clone(), *v)).collect();
            let sref = ScenarioRef::Inline {
                name: sc.name.clone(),
                enable: sc.enable.clone(),
                disable: sc.disable.clone(),
                params: params_im,
            };
            match resolve_scenario_ref(&sref, &preset_names)? {
                ResolvedScenario::Preset { name } => {
                    // Composed, via the single horizon authority — see the twin
                    // site in `main.rs` (gh#561).
                    let t_end = crate::params_resolver::composed_preset_t_end(model, &name)
                        .map_err(|e| e.to_string())?;
                    ResolvedEntry::from_preset(model, &name, t_end)
                }
                ResolvedScenario::Adhoc { name, enable, disable, params } => {
                    Ok(ResolvedEntry {
                        name,
                        route: None,
                        enable,
                        disable,
                        params: params.into_iter().collect(),
                        scale: Vec::new(),
                        // An ad-hoc patch has no horizon: `simulate { to }` is
                        // a model-file declaration with no CLI spelling.
                        t_end: None,
                    })
                }
            }
        })
        .collect()
}

// ─── The manifest's model ───────────────────────────────────────────────

/// The manifest's `model` as `batch run` executes it. The temp files the IR
/// lives in are held here, so the paths stay valid while this is alive.
struct ManifestModel {
    /// The IR the engine loads and the cells' identity is resolved from.
    ir_path: String,
    ir_json: String,
    _compiled: Option<std::path::PathBuf>,
    _every: Option<tempfile::NamedTempFile>,
}

/// Load the manifest's model the one way `batch run` and `batch status` share:
/// `.camdl` source compiled (a `.ir.json` taken as is), then `[output] every`
/// lowered into the IR. Status predicts which leaves exist from the identity
/// this IR resolves to, so a second loader — one that skipped the compile, or
/// the cadence — would count against a different identity (gh#592).
fn load_manifest_model(model_path: &str, output_every: Option<f64>) -> Result<ManifestModel, String> {
    // Batch runs are forward simulations — no state-Jacobian consumer, so
    // compile lean (`needs_state_grad = false`, gh#439 A2).
    let (compiled_path, compiled) = resolve_ir_path(model_path, false)?;
    // gh#156: `[output] every` rewrites the compiled IR's output schedule once,
    // so both the engine and the CAS identity (loaded from this path) see the
    // overridden cadence.
    let (ir_path, every) =
        crate::util::rematerialize_with_output_every(&compiled_path, output_every)?;
    let ir_json = std::fs::read_to_string(&ir_path)
        .map_err(|e| format!("cannot read {}: {}", ir_path, e))?;
    Ok(ManifestModel { ir_path, ir_json, _compiled: compiled, _every: every })
}

/// The `SimulateJob` a batch manifest runs: its model IR and params file,
/// and none of the CLI-only knobs (`--to`, `--init-state`, anchors,
/// `--integrator`, `--param`, `--param-vec`, `--table`). The one constructor
/// `batch run`, the `[design.*]` path, and the dry-run / `batch status`
/// predictions share, so a prediction builds each cell's `SimRun` from the job
/// the run would build (`engine::build_cell_sim_run`).
///
/// Refuses a scenario that touches a swept parameter
/// ([`crate::engine::check_scenario_coordinate_collision`], gh#572). The check
/// lives here as well as in `engine::run_job` because the dry-run and status
/// commands plan cells without calling `run_job`; every batch command obtains
/// its job from this constructor, so none can plan a grid `batch run` would
/// refuse.
#[allow(clippy::too_many_arguments)]
fn batch_job(
    ir_path: &str,
    params_file: Option<&str>,
    backend: crate::args::types::ForwardBackend,
    dt: f64,
    source: crate::sim_job::ParamSource,
    scenarios: Vec<crate::sim_job::ScenarioRef>,
    seeds: &[u64],
    parallel: usize,
) -> Result<crate::sim_job::SimulateJob, String> {
    let job = crate::sim_job::SimulateJob {
        model: ir_path.to_string(),
        params_files: params_file.map(|p| vec![p.to_string()]).unwrap_or_default(),
        backend,
        dt,
        integrator: None, // batch uses the model's declared integrator (no CLI override)
        source,
        scenarios,
        // gh#626: batch TOML has no `to` key (deliberate; CLI-only override).
        t_end_override: None,
        // Nor forecast-row boundaries: batch records the model's own schedule.
        required_output_times: Vec::new(),
        // gh#641: nor an `init_state` key, for the same reason.
        init_state: None,
        // gh#616: `batch run` has no data-binding surface, so an anchored
        // model is refused at `CompiledModel::new` rather than resolved.
        obs_anchors: None,
        // Batch seeds are always explicit (range / count / list).
        seeds: crate::sim_job::Seeds::Explicit(seeds.to_vec()),
        cli_overrides: Vec::new(),
        // Nor a fit config to sample priors from.
        fit_fixed: indexmap::IndexMap::new(),
        set_vec_entries: Vec::new(),
        table_files: Vec::new(),
        // batch keeps its CAS-per-cell obs ensemble (CasSink), not the
        // combined-file ObsOutput modes — leave None here.
        obs: crate::sim_job::ObsOutput::None,
        parallel,
    };
    crate::engine::check_scenario_coordinate_collision(&job)?;
    Ok(job)
}

/// The engine's scenario references for a manifest's resolved scenarios: a
/// resolved preset → `ScenarioRef::Named` (the params_resolver preset path);
/// an ad-hoc patch → `ScenarioRef::Inline`.
fn scenario_refs(resolved: &[ResolvedEntry]) -> Vec<crate::sim_job::ScenarioRef> {
    use crate::sim_job::ScenarioRef;
    resolved.iter().map(|r| match &r.route {
        Some(preset_name) => ScenarioRef::Named(preset_name.clone()),
        None => ScenarioRef::Inline {
            name: r.name.clone(),
            enable: r.enable.clone(),
            disable: r.disable.clone(),
            params: r.params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        },
    }).collect()
}

/// The parameter source for a manifest's `[sweep]`: a non-empty sweep → a
/// `Sweep` over its expanded points; an empty one → `Point` (the single null
/// point). Batch replicates ride on the explicit `seeds` list, so the engine
/// uses that length (not `replicates`) for the rep count.
fn sweep_source(
    sweep: &HashMap<String, SweepSpec>,
    points: &[HashMap<String, f64>],
) -> crate::sim_job::ParamSource {
    use crate::sim_job::ParamSource;
    if sweep.is_empty() {
        ParamSource::Point { replicates: 1 }
    } else {
        ParamSource::Sweep {
            points: points.iter()
                .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
                .collect(),
            replicates: 1,
        }
    }
}

// ─── cmd_batch_run ──────────────────────────────────────────────────────

pub fn cmd_batch_run(a: &crate::args::BatchArgs) {
    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();  // gh#audit-H5
    sim::eval_stats::set_allow_degenerate_rates(a.allow_degenerate_rates);  // gh#audit-C6
    let toml_path = a.file.to_string_lossy().into_owned();

    let toml_src = std::fs::read_to_string(&toml_path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", toml_path, e);
        std::process::exit(1);
    });
    let exp: ExperimentToml = toml::from_str(&toml_src).unwrap_or_else(|e| {
        eprintln!("error: TOML parse error in {}: {}", toml_path, e);
        std::process::exit(1);
    });

    let output_dir = a.output_dir.as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| exp.config.output_dir.clone());
    let parallel = a.parallel.unwrap_or(exp.config.parallel);
    let backend    = exp.config.backend;
    let dt         = exp.config.dt;
    let model_path = exp.config.model.clone();

    let loaded = load_manifest_model(&model_path, exp.output.every).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let ir_path_resolved = loaded.ir_path.clone();
    let ir_json = loaded.ir_json.clone();

    let base_params: HashMap<String, f64> = if let Some(ref pf) = exp.config.params {
        // Surface fit-provenance status for the params file: a verified MLE
        // file gets a green check; one edited since inference produced it gets
        // a hash-mismatch warning. A standalone (no-header) params file is
        // silent. Side-effect only — the per-cell leaves carry the params
        // identity in their `params` level.
        match crate::fit::provenance::verify_content_hash(pf) {
            Ok(crate::fit::provenance::ContentVerification::Valid) => {
                eprintln!("params: {} \x1b[32m✓ provenance verified\x1b[0m", pf);
            }
            Ok(crate::fit::provenance::ContentVerification::Modified { declared, computed }) => {
                eprintln!("\x1b[33mwarning: params file {} has been modified since inference produced it.\x1b[0m", pf);
                eprintln!("  Content hash mismatch: expected {}, got {}", declared, computed);
            }
            _ => {
                // No provenance header — standalone params file, that's fine.
            }
        }
        load_params_toml(pf).unwrap_or_else(|e| {
            eprintln!("error: cannot load params {}: {}", pf, e);
            std::process::exit(1);
        })
    } else {
        HashMap::new()
    };

    let seeds = exp.config.seeds.resolve().unwrap_or_else(|e| {
        eprintln!("error resolving seeds: {}", e);
        std::process::exit(1);
    });

    // Validate [sweep] and [design.*] are mutually exclusive.
    if !exp.sweep.is_empty() && !exp.design.is_empty() {
        eprintln!("error: [sweep] and [design.*] are mutually exclusive.");
        eprintln!("  [sweep] — deterministic grid for specific parameter values");
        eprintln!("  [design.*] — space-filling for sensitivity/VOI analysis");
        eprintln!("  Use one or the other in a single experiment file.");
        std::process::exit(1);
    }

    let params_file_opt = exp.config.params.clone();

    // ── Shared setup (both the [sweep]/plain path and the [design.*] path) ──
    //
    // Parse the model, resolve the `[output]` column view, resolve every
    // `[[scenario]]` against the model's presets, and validate `[obs]`. The
    // design path routes through the SAME engine + `CasSink` machinery as the
    // normal sweep flow (gh#241 PR E2), so it needs the identical resolved
    // inputs — a design cell and a normal sim cell with the same
    // params/scenario/seed must resolve to the same `Sim` identity and dedupe.

    // Resolve each [[scenario]] against the model's scenarios{} presets
    // (CLI review #3). A name matching a preset routes through the same
    // params_resolver preset path simulate --scenario uses; a name with
    // inline patches is ad-hoc; an unknown name with no patches is a hard
    // error. The model is parsed from the IR JSON already in hand.
    let batch_model: ir::Model = ir::from_str(&ir_json).unwrap_or_else(|e| {
        eprintln!("error: cannot parse model IR for scenario resolution: {}", e);
        std::process::exit(1);
    });

    // gh#156: resolve + validate the `[output]` column view against the model,
    // once. Folded into each cell's CAS identity and used by the leaf writer.
    // The design path now threads the same column view through `CasSink`, so
    // `no_flows` / `columns` are honored for design experiments too (no special
    // restriction — the normal flow's writer is the only writer now).
    let output_cols = crate::util::OutputColumns::resolve(&exp.output, &batch_model)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

    let raw_scenarios: Vec<ScenarioEntry> = if exp.scenario.is_empty() {
        vec![ScenarioEntry { name: "baseline".to_string(), params: HashMap::new(), enable: vec![], disable: vec![] }]
    } else {
        exp.scenario.clone()
    };
    let resolved_scenarios = resolve_batch_scenarios(&raw_scenarios, &batch_model)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

    // [obs] enabled (CLI review #4). When set, each run's observation
    // streams are sampled and written into the CAS obs subtree. Validate
    // up front that the model declares observation blocks — a silent
    // no-op here would be a "looks like it works but produced nothing"
    // trap.
    let obs_enabled = exp.obs.enabled;
    if obs_enabled && batch_model.observations.is_empty() {
        eprintln!("error: [obs] enabled = true but the model declares no \
                   observations {{}} block — nothing to sample.");
        std::process::exit(1);
    }
    // gh#829: the `obs/` subtree draws every stream with no data columns, so
    // a stream whose likelihood reads one is refused before any cell runs.
    if obs_enabled {
        for o in &batch_model.observations {
            if let Err(e) = crate::obs_emit::check_data_columns_supplied(
                o, crate::obs_emit::ColumnSupply::Nothing,
            ) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }

    // The store's model directory and the leaf's provenance name the model
    // the manifest points at, never the IR it resolved to: for `.camdl`
    // source that is the IR cache's content hash, or `camdl_<pid>` with the
    // cache off, so the same run_id would land in a different directory per
    // entry point and per invocation (gh#583 item E). `simulate` threads its
    // original path the same way (`build_simulate_cas_sink`'s `display_path`).
    let model_stem = crate::hashing::path_stem_slug(&model_path);

    // Validate [sweep] and [design.*] are mutually exclusive.
    if !exp.sweep.is_empty() && !exp.design.is_empty() {
        eprintln!("error: [sweep] and [design.*] are mutually exclusive.");
        eprintln!("  [sweep] — deterministic grid for specific parameter values");
        eprintln!("  [design.*] — space-filling for sensitivity/VOI analysis");
        eprintln!("  Use one or the other in a single experiment file.");
        std::process::exit(1);
    }

    // [design.*]: each block's generated points are routed through the SAME
    // engine + `CasSink` flow as a [sweep] (the points ARE sweep points). The
    // experiment-side metadata (`parameter_points.tsv`, `priors.txt`) is
    // written per block; the sim leaves are canonical `ArtifactKind::Sim`
    // leaves under `<output>/sims/`, deduping against identical normal sims.
    if !exp.design.is_empty() {
        run_design_experiment(
            &exp.design,
            &resolved_scenarios,
            &batch_model,
            &ir_path_resolved,
            &model_path,
            model_stem.as_deref(),
            &output_dir,
            &params_file_opt,
            backend,
            dt,
            a.allow_degenerate_rates,
            &output_cols,
            obs_enabled,
            a.force,
            a.dry_run,
            parallel,
            &seeds,
        );
        return;
    }

    // Expand [sweep] into parameter points (empty sweep → one null point).
    let sweep_points = expand_sweep(&exp.sweep);
    let has_sweep = !exp.sweep.is_empty();

    let runs_dir = format!("{}/sims", output_dir);

    // The cell grid the engine will run: |points| × |scenarios| × |seeds|.
    // `expand_sweep` returns one null point for an empty sweep, so the count
    // is exact for both the sweep and no-sweep cases.
    let total = sweep_points.len() * resolved_scenarios.len() * seeds.len();

    // ── Build the SimulateJob (before the dry-run branch) ────────────────────
    //
    // `batch run` is a thin TOML front-end over `engine::run_job` — the
    // SAME engine `camdl simulate` uses (run-spec §3.1). The per-cell seed
    // arithmetic and SimRun construction are shared; the CAS-tree output
    // shape lives in `CasSink`, which resolves each cell's identity via
    // `resolve::resolve_trajectory` so the on-disk layout / content-hashes
    // match the `simulate` path exactly. Built before the dry-run branch so a
    // dry run refuses what the run would refuse (a scenario touching a swept
    // parameter, gh#572) and predicts cells from the same job.
    let job = batch_job(
        &ir_path_resolved, params_file_opt.as_deref(), backend, dt,
        sweep_source(&exp.sweep, &sweep_points), scenario_refs(&resolved_scenarios),
        &seeds, parallel,
    ).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    if a.dry_run {
        print_batch_dry_run(
            &model_path, &batch_model, model_stem.as_deref(),
            backend, dt, &output_dir, parallel,
            &resolved_scenarios, &sweep_points, &seeds, &base_params,
            exp.config.params.as_deref(), &runs_dir, a.allow_degenerate_rates,
            &output_cols, a.force, &job,
        );
        return;
    }

    std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| {
        eprintln!("error: cannot create output dir {}: {}", runs_dir, e);
        std::process::exit(1);
    });

    std::fs::write(format!("{}/model.ir.json", output_dir), &ir_json).unwrap_or_else(|e| {
        eprintln!("warning: could not write model.ir.json: {}", e);
    });

    // Archive the model's display render and its structured flow graph beside
    // the IR so a viewer (camdl-watch) can show the model's math and draw its
    // compartmental diagram without recompiling. Best-effort — a render failure
    // warns and skips, never aborts the run.
    //
    // gh#496: "the model was given as compiled IR" is NOT one of those
    // failures. `camdlc render` takes `.camdl` source, `resolve_ir_path`
    // accepts source or a compiled `.ir.json`, so `exp.config.model` may be
    // either — and shelling out at an `.ir.json` reported camdlc's words for an
    // expected, benign condition:
    //
    //   warning: could not render model for archive: camdlc render failed:
    //   camdlc render: parse error in .../sir_basic.ir.json
    //
    // which reads as "your IR file is malformed." Decide it here instead, and
    // say what actually happened. (`fit/mod.rs` uses `config.model.camdl`,
    // which is source by construction, so it is unaffected.)
    if model_is_camdl_source(&model_path) {
        match crate::util::render_model_json(std::path::Path::new(&model_path)) {
            Ok(json) => {
                if let Err(e) = std::fs::write(format!("{}/model.render.json", output_dir), &json) {
                    eprintln!("warning: could not write model.render.json: {}", e);
                }
            }
            Err(e) => eprintln!("warning: could not render model for archive: {}", e),
        }

        match crate::util::render_model_graph_json(std::path::Path::new(&model_path)) {
            Ok(json) => {
                if let Err(e) = std::fs::write(format!("{}/model.graph.json", output_dir), &json) {
                    eprintln!("warning: could not write model.graph.json: {}", e);
                }
            }
            Err(e) => eprintln!("warning: could not render model graph for archive: {}", e),
        }
    } else {
        eprintln!(
            "note: model given as compiled IR; skipping model.render.json / \
             model.graph.json (pass the .camdl source to archive the display render)"
        );
    }

    // Copy any boundary GeoJSON into the output tree as a sibling artifact
    // (a map viewer reads `<output>/geo/boundaries.geojson`).
    if let Some(ref geo_src) = exp.config.geo {
        let geo_dest = format!("{}/geo/boundaries.geojson", output_dir);
        if let Err(e) = std::fs::create_dir_all(format!("{}/geo", output_dir))
            .and_then(|_| std::fs::copy(geo_src, &geo_dest))
        {
            eprintln!("warning: could not copy geo file '{}': {}", geo_src, e);
        }
    }

    if has_sweep {
        eprintln!("Sweep: {} parameter points", sweep_points.len());
        for (i, pt) in sweep_points.iter().enumerate().take(3) {
            let desc: Vec<String> = pt.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
            eprintln!("  point {}: {}", i, desc.join(", "));
        }
        if sweep_points.len() > 3 {
            eprintln!("  ... ({} more)", sweep_points.len() - 3);
        }
    }

    // gh#audit-H13: scope the engine's parallelism to `--parallel`. The engine
    // (engine.rs) does `(0..n).into_par_iter()` on the surrounding pool gated by
    // `grid.parallel > 1`; running `run_job` inside `pool.install(...)` makes
    // that par_iter use the scoped pool. See `build_parallel_pool`.
    let pool = build_parallel_pool(parallel);

    let mut sink = CasSink {
        resolved_scenarios: resolved_scenarios.clone(),
        model_path: model_path.clone(),
        model_stem: model_stem.clone(),
        base_model: batch_model.clone(),
        backend,
        dt,
        allow_degenerate_rates: a.allow_degenerate_rates,
        output_cols: output_cols.clone(),
        runs_dir: runs_dir.clone(),
        obs_enabled,
        // `batch run` has no `--emit-every` surface (gh#656 is scoped to
        // `simulate` and `fit run`); an experiment TOML declares its own
        // cadence in the model.
        emit_every: None,
        force: a.force,
        total,
        counter: 0,
        completed_runs: Vec::new(),
        errors: Vec::new(),
        label: None,
        fit_dep: Vec::new(),
        progress: cells_progress(total, "batch run"),
    };

    run_pooled(&pool, || crate::engine::run_job(&job, &mut sink)).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let errors = sink.errors;
    let completed_runs = sink.completed_runs;
    if !errors.is_empty() {
        eprintln!("Errors encountered:");
        for e in &errors { eprintln!("  {}", e); }
    }

    let completed = completed_runs.len();
    // No batch-level index file: each completed cell is its own
    // `runid::RunRecord` leaf under `sims/` (the system of record). Enumerate
    // a sweep with `camdl list` / the derived `index.json`, not a manifest.
    eprintln!("Done: {}/{} runs completed. Leaves under {}/sims/",
        completed, total, output_dir);
    if !errors.is_empty() { std::process::exit(1); }
}

impl CasSink {
    /// A throwaway `CasSink` for dry-run / status prediction. It is used ONLY
    /// for `cell_resolve` (the canonical identity/path resolver) and `root()`,
    /// so the identity-relevant fields are populated from the same resolved
    /// inputs the real run path uses; the run-only fields (`model_path`,
    /// `total`/`counter`, `completed_runs`, `progress`, …) are inert.
    fn probe(
        resolved_scenarios: &[ResolvedEntry],
        base_model: &ir::Model,
        model_stem: Option<&str>,
        backend: crate::args::types::ForwardBackend,
        dt: f64,
        allow_degenerate_rates: bool,
        output_cols: &crate::util::OutputColumns,
        runs_dir: &str,
        force: bool,
    ) -> Self {
        CasSink {
            resolved_scenarios: resolved_scenarios.to_vec(),
            model_path: String::new(),
            model_stem: model_stem.map(|s| s.to_string()),
            base_model: base_model.clone(),
            backend,
            dt,
            allow_degenerate_rates,
            output_cols: output_cols.clone(),
            runs_dir: runs_dir.to_string(),
            obs_enabled: false,
            emit_every: None,
            force,
            total: 0,
            counter: 0,
            completed_runs: Vec::new(),
            errors: Vec::new(),
            label: None,
            fit_dep: Vec::new(),
            progress: None,
        }
    }

    /// Resolve every (point, scenario, seed) cell's real `Sim` identity + path
    /// and classify each as a CAS cache hit/miss via `store.lookup` — the SAME
    /// `cell_resolve` + `LeafIdentity` gate `should_run` uses on the real run
    /// path, so a dry-run / status prediction reports exactly what `run` would
    /// resolve. No simulation, no files written.
    ///
    /// `points` is the expanded `[sweep]`/`[design]` grid (one null point for an
    /// empty sweep). Scenario routing matches the engine job: a resolved preset
    /// → `ScenarioRef::Named`, an ad-hoc patch → `Inline`; the sweep point is
    /// the cell's `point_overrides`. Each cell's `SimRun` is built from `job` by
    /// `engine::build_cell_sim_run` — the constructor the engine uses — so the
    /// predicted identity reads the inputs the run would. A cell whose identity
    /// fails to resolve is counted as a miss (it surfaces as an error on the
    /// real run path).
    fn predict_cells(
        &self,
        job: &crate::sim_job::SimulateJob,
        points: &[indexmap::IndexMap<String, f64>],
        seeds: &[u64],
    ) -> Vec<CellPrediction> {
        let store = runid::FsCasStore::new(self.root());
        let table_files: HashMap<String, String> = job.table_files.iter().cloned().collect();
        let mut out =
            Vec::with_capacity(points.len() * self.resolved_scenarios.len() * seeds.len());
        for (point_idx, point_overrides) in points.iter().enumerate() {
            for sref in &self.resolved_scenarios {
                let scenario = match &sref.route {
                    Some(preset) => crate::sim_job::ScenarioRef::Named(preset.clone()),
                    None => crate::sim_job::ScenarioRef::Inline {
                        name: sref.name.clone(),
                        enable: sref.enable.clone(),
                        disable: sref.disable.clone(),
                        params: sref.params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                    },
                };
                for &seed in seeds {
                    let sim_run = crate::engine::build_cell_sim_run(
                        job, &scenario, point_overrides, &table_files, seed, point_idx, 0,
                    );
                    let spec = crate::engine::CellSpec {
                        run_idx: 0,
                        point_idx,
                        scenario: scenario.clone(),
                        point_overrides: point_overrides.clone(),
                        process_seed: seed,
                        obs_seed: seed ^ crate::util::SEED_MIX_OBS,
                        sim_run,
                    };
                    match self.cell_resolve(&spec) {
                        Ok((rt, dir, rel)) => {
                            let hit = !self.force
                                && matches!(
                                    store.lookup(&dir, &runid::LeafIdentity::new(rt.run_id)),
                                    runid::Lookup::Hit(_)
                                );
                            out.push(CellPrediction {
                                scenario: sref.name.clone(),
                                seed,
                                rel,
                                hit,
                            });
                        }
                        // No cache key without a resolve → treat as a miss; the
                        // error surfaces in `merge_cell` on the real run path.
                        Err(_) => out.push(CellPrediction {
                            scenario: sref.name.clone(),
                            seed,
                            rel: String::new(),
                            hit: false,
                        }),
                    }
                }
            }
        }
        out
    }
}

/// One cell's resolved CAS prediction for a dry-run / `batch status` summary:
/// the cell coordinates (scenario, seed), its store-relative `Sim` path, and
/// whether it is already a committed leaf (a `store.lookup` hit). Identity is
/// resolved through `CasSink::cell_resolve` — the same path a real `run` takes.
struct CellPrediction {
    scenario: String,
    seed: u64,
    rel: String,
    hit: bool,
}

// ─── Design experiment execution ─────────────────────────────────────────────

/// Run a design-based experiment (VOI/sensitivity analysis).
///
/// For each named design:
///   1. Generate parameter points via the specified method (sobol/lhs/random).
///   2. Write the experiment-side metadata —
///      `{output_dir}/designs/{design}/parameter_points.tsv` (point_id → param
///      values) and `priors.txt` — the bridge a downstream `camdl voi` uses to
///      recover which point each leaf belongs to.
///   3. Route the block's points through the SAME engine + `CasSink` flow the
///      normal sweep uses (gh#241 PR E2): each (point, scenario, seed) cell
///      becomes a canonical `ArtifactKind::Sim` leaf under `{output_dir}/sims/`,
///      deduping against identical normal sims. A design point's values enter
///      the `params` identity level (via `cell_resolve`); the design
///      method/name/point_idx are NOT in the leaf identity — they live only in
///      `parameter_points.tsv`.
///
/// Multiple design blocks → one engine pass per block (a fresh `CasSink` per
/// block so hit-accounting and the progress bar are per-block correct).
#[allow(clippy::too_many_arguments)]
fn run_design_experiment(
    designs: &HashMap<String, DesignBlock>,
    resolved_scenarios: &[ResolvedEntry],
    batch_model: &ir::Model,
    ir_path: &str,
    model_path: &str,
    model_stem: Option<&str>,
    output_dir: &str,
    params_file_opt: &Option<String>,
    backend: crate::args::types::ForwardBackend,
    dt: f64,
    allow_degenerate_rates: bool,
    output_cols: &crate::util::OutputColumns,
    obs_enabled: bool,
    force: bool,
    dry_run: bool,
    parallel: usize,
    seeds: &[u64],
) {
    use crate::sim_job::ParamSource;

    // The CAS store root is the canonical output root: design sim leaves share
    // the `<output>/sims/` tree with normal `batch`/`simulate` cells so an
    // identical cell dedupes to the same leaf (same run_id, same path).
    let runs_dir = format!("{}/sims", output_dir);

    // Scenario routing for the engine job — identical to the normal flow.
    let job_scenarios = scenario_refs(resolved_scenarios);

    // Sort design names for deterministic output.
    let mut design_names: Vec<&String> = designs.keys().collect();
    design_names.sort();

    for design_name in &design_names {
        let block = &designs[*design_name];
        eprintln!("Design '{}': method={} n={} parameters={}",
            design_name, block.method, block.n, block.parameters.len());

        // Build sorted parameter list.
        let mut param_names: Vec<&String> = block.parameters.keys().collect();
        param_names.sort();
        let params: Vec<(String, DesignParam)> = param_names.iter().map(|name| {
            let p = &block.parameters[*name];
            ((*name).clone(), DesignParam {
                min: p.range.min,
                max: p.range.max,
                transform: p.transform.clone(),
                prior: p.prior.clone(),
            })
        }).collect();

        // Generate design points.
        let design_result = generate_design(&params, block.n, &block.method);
        let n_points = design_result.points.len();
        eprintln!("  Generated {} parameter points", n_points);

        // The design points ARE sweep points: each is a name→value override
        // map of the same shape `expand_sweep` produces, run through the SAME
        // `SimulateJob` + `CasSink` the normal flow uses. Built (and checked
        // for scenario collisions, gh#572) before any design metadata is
        // written, so a refused design — dry run or not — leaves nothing behind.
        let points: Vec<indexmap::IndexMap<String, f64>> = design_result.points.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect();
        let job = batch_job(
            ir_path, params_file_opt.as_deref(), backend, dt,
            ParamSource::Sweep { points: points.clone(), replicates: 1 },
            job_scenarios.clone(), seeds, parallel,
        ).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

        // ── Experiment-side metadata (preserved exactly) ──────────────────
        let design_dir = format!("{}/designs/{}", output_dir, design_name);
        std::fs::create_dir_all(&design_dir).unwrap_or_else(|e| {
            eprintln!("error: cannot create design dir {}: {}", design_dir, e);
            std::process::exit(1);
        });

        let pts_path = format!("{}/parameter_points.tsv", design_dir);
        let mut pts_tsv = String::new();
        // Header: point_id + sorted param names
        pts_tsv.push_str("point_id");
        for name in &design_result.param_names {
            pts_tsv.push('\t');
            pts_tsv.push_str(name);
        }
        pts_tsv.push('\n');
        for (i, pt) in design_result.points.iter().enumerate() {
            pts_tsv.push_str(&i.to_string());
            for name in &design_result.param_names {
                pts_tsv.push('\t');
                pts_tsv.push_str(&format!("{:.8}", pt[name]));
            }
            pts_tsv.push('\n');
        }
        std::fs::write(&pts_path, &pts_tsv).unwrap_or_else(|e| {
            eprintln!("warning: could not write {}: {}", pts_path, e);
        });
        eprintln!("  Wrote {}", pts_path);

        // Write priors.txt if any parameter has a prior specification.
        let priors_txt = build_priors_txt(&params);
        if let Some(txt) = priors_txt {
            let priors_path = format!("{}/priors.txt", design_dir);
            let _ = std::fs::write(&priors_path, txt);
        }

        let total = points.len() * resolved_scenarios.len() * seeds.len();

        // Dry-run: resolve every cell's real CAS identity and count
        // `store.lookup` hits (the same path `CasSink::should_run` takes) —
        // never the legacy hash/marker check. No simulation, no files written.
        if dry_run {
            print_design_dry_run(
                design_name, &points, resolved_scenarios, batch_model,
                model_stem,
                &job,
                backend, dt, allow_degenerate_rates, output_cols, &runs_dir, seeds, force,
            );
            continue;
        }

        std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| {
            eprintln!("error: cannot create runs dir {}: {}", runs_dir, e);
            std::process::exit(1);
        });

        let mut sink = CasSink {
            resolved_scenarios: resolved_scenarios.to_vec(),
            model_path: model_path.to_string(),
            model_stem: model_stem.map(|s| s.to_string()),
            base_model: batch_model.clone(),
            backend,
            dt,
            allow_degenerate_rates,
            output_cols: output_cols.clone(),
            runs_dir: runs_dir.clone(),
            obs_enabled,
            emit_every: None,
            force,
            total,
            counter: 0,
            completed_runs: Vec::new(),
            errors: Vec::new(),
            label: None,
            fit_dep: Vec::new(),
            progress: cells_progress(total, format!("design '{}'", design_name)),
        };

        // gh#audit-H13: scope the engine's parallelism to `--parallel`.
        let pool = build_parallel_pool(parallel);
        run_pooled(&pool, || crate::engine::run_job(&job, &mut sink)).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

        if !sink.errors.is_empty() {
            eprintln!("Errors encountered in design '{}':", design_name);
            for e in &sink.errors { eprintln!("  {}", e); }
        }
        eprintln!("Design '{}' complete: {}/{} cells. Leaves under {}/.",
            design_name, sink.completed_runs.len(), total, runs_dir);
    }
}

/// Dry-run summary for one design block: resolve every (point, scenario, seed)
/// cell's real CAS identity (`Sim` run_id + path) and count `store.lookup`
/// hits — the SAME identity/path `CasSink::should_run` uses, NOT the legacy
/// `plan_runs` hash/marker. No simulation, no files written.
#[allow(clippy::too_many_arguments)]
fn print_design_dry_run(
    design_name: &str,
    points: &[indexmap::IndexMap<String, f64>],
    resolved_scenarios: &[ResolvedEntry],
    batch_model: &ir::Model,
    model_stem: Option<&str>,
    job: &crate::sim_job::SimulateJob,
    backend: crate::args::types::ForwardBackend,
    dt: f64,
    allow_degenerate_rates: bool,
    output_cols: &crate::util::OutputColumns,
    runs_dir: &str,
    seeds: &[u64],
    force: bool,
) {
    let probe = CasSink::probe(
        resolved_scenarios, batch_model, model_stem,
        backend, dt, allow_degenerate_rates, output_cols, runs_dir, force,
    );
    let predictions = probe.predict_cells(job, points, seeds);
    let hits = predictions.iter().filter(|c| c.hit).count();
    let misses = predictions.len() - hits;

    eprintln!("  Design '{}' (dry run): {} points × {} scenarios × {} seeds = {} cells",
        design_name, points.len(), resolved_scenarios.len(), seeds.len(),
        predictions.len());
    eprintln!("    {} cache hits  → skipped", hits);
    eprintln!("    {} cache misses → would simulate", misses);
    eprintln!("  Output paths (sims/<run_path>):");
    for cell in predictions.iter().take(6) {
        eprintln!("    [{}] {}", if cell.hit { "hit " } else { "miss" }, cell.rel);
    }
    if predictions.len() > 6 {
        eprintln!("    ... ({} more)", predictions.len() - 6);
    }
}

// ─── Prior spec helpers ───────────────────────────────────────────────────────

/// Build human-readable priors.txt content for a design's parameter list.
/// Returns None if no parameters have prior specifications.
fn build_priors_txt(params: &[(String, DesignParam)]) -> Option<String> {
    let with_priors: Vec<&(String, DesignParam)> = params.iter()
        .filter(|(_, p)| p.prior.is_some())
        .collect();
    if with_priors.is_empty() {
        return None;
    }
    let mut txt = String::from("Parameter priors:\n\n");
    for (name, param) in params {
        let prior_desc = match &param.prior {
            Some(p) => describe_prior(p),
            None => "Uniform (no prior specified)".to_string(),
        };
        let transform_desc = match param.transform.as_deref() {
            Some("log") => " [log-uniform sampling]",
            Some("logit") => " [logit-uniform sampling]",
            _ => "",
        };
        txt.push_str(&format!("  {}: {} over [{}, {}]{}\n",
            name, prior_desc, param.min, param.max, transform_desc));
    }
    txt.push('\n');
    txt.push_str("These priors are used by the VOI tool (camdl voi run) for importance\n");
    txt.push_str("weighting. If no prior is specified for a parameter, uniform is assumed.\n");
    Some(txt)
}

// ─── cmd_batch_status ───────────────────────────────────────────────────

pub fn cmd_batch_status(a: &crate::args::BatchStatusArgs) {
    let toml_path = a.file.to_string_lossy().into_owned();

    let toml_src = std::fs::read_to_string(&toml_path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", toml_path, e);
        std::process::exit(1);
    });
    let exp: ExperimentToml = toml::from_str(&toml_src).unwrap_or_else(|e| {
        eprintln!("error: TOML parse error in {}: {}", toml_path, e);
        std::process::exit(1);
    });

    let output_dir = exp.config.output_dir.clone();

    // Status is derived live from (a) the batch.toml plan and (b) the
    // on-disk leaf tree — there is no batch-level manifest. We resolve every
    // cell's real `Sim` identity exactly as `batch run` would (`cell_resolve`)
    // and count which already have a committed leaf via `store.lookup` — the
    // same CAS gate `should_run` uses, never a legacy hash/marker prediction.
    println!("Experiment status for: {}", toml_path);
    println!("  Model:      {}", exp.config.model);
    println!("  Output dir: {}", output_dir);

    let seeds = exp.config.seeds.resolve().unwrap_or_default();

    // The model exactly as `batch run` loads it, so every cell resolves to the
    // identity the run would give it. A model that cannot be loaded is an
    // error, not a sweep that has not started.
    let loaded = load_manifest_model(&exp.config.model, exp.output.every).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let ir_json = &loaded.ir_json;

    // Parse the model and resolve every `[[scenario]]` against its presets, so
    // the prediction resolves each cell's identity exactly as the run path
    // does (CLI review #3). A parse/resolution failure means the model itself
    // is broken — report and bail rather than mis-predict from raw scenarios.
    let batch_model: ir::Model = ir::from_str(ir_json).unwrap_or_else(|e| {
        eprintln!("error: cannot parse model IR for {}: {}", exp.config.model, e);
        std::process::exit(1);
    });
    let raw_scenarios: Vec<ScenarioEntry> = if exp.scenario.is_empty() {
        vec![ScenarioEntry { name: "baseline".to_string(), params: HashMap::new(), enable: vec![], disable: vec![] }]
    } else {
        exp.scenario
    };
    let resolved_scenarios = match resolve_batch_scenarios(&raw_scenarios, &batch_model) {
        Ok(r) => r,
        Err(e) => {
            println!("  (cannot resolve scenarios: {})", e);
            println!("  Run 'camdl batch run {}' to start.", toml_path);
            return;
        }
    };
    let scenario_names: Vec<String> = resolved_scenarios.iter().map(|s| s.name.clone()).collect();
    println!("  Scenarios:  {}", scenario_names.join(", "));
    println!("  Seeds:      {} total ({}..={})",
        seeds.len(),
        seeds.first().copied().unwrap_or(0),
        seeds.last().copied().unwrap_or(0));

    // Resolve the `[output]` column view (folded into each cell's `config`
    // identity, so the prediction must use the same view the run path does).
    let output_cols = match crate::util::OutputColumns::resolve(&exp.output, &batch_model) {
        Ok(c) => c,
        Err(e) => {
            println!("  (cannot resolve [output] view: {})", e);
            println!("  Run 'camdl batch run {}' to start.", toml_path);
            return;
        }
    };

    let model_stem = crate::hashing::path_stem_slug(&exp.config.model);
    let runs_dir = format!("{}/sims", output_dir);
    let sweep_points = expand_sweep(&exp.sweep);
    let points: Vec<indexmap::IndexMap<String, f64>> = sweep_points
        .iter()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
        .collect();

    let probe = CasSink::probe(
        &resolved_scenarios, &batch_model, model_stem.as_deref(),
        exp.config.backend, exp.config.dt, false, &output_cols, &runs_dir, false,
    );
    // The job `batch run` would execute: a manifest `batch run` refuses (a
    // scenario touching a swept parameter, gh#572) is refused here too.
    let job = batch_job(
        &loaded.ir_path, exp.config.params.as_deref(), exp.config.backend, exp.config.dt,
        sweep_source(&exp.sweep, &sweep_points), scenario_refs(&resolved_scenarios),
        &seeds, 1,
    ).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let predictions = probe.predict_cells(&job, &points, &seeds);
    let live_hits = predictions.iter().filter(|c| c.hit).count();
    println!("  Completed:  {}/{} leaves present", live_hits, predictions.len());

    // List the first few cells still to run (scenario / seed / path), so a
    // resumed `batch run` is predictable from `batch status`.
    let remaining: Vec<&CellPrediction> = predictions.iter().filter(|c| !c.hit).collect();
    if !remaining.is_empty() {
        println!("  Remaining:  {} cell(s) to run:", remaining.len());
        for c in remaining.iter().take(6) {
            // `rel` is store-relative and already begins `sims/`.
            println!("    scenario={} seed={}  → {}/{}", c.scenario, c.seed, output_dir, c.rel);
        }
        if remaining.len() > 6 {
            println!("    ... ({} more)", remaining.len() - 6);
        }
    }
    if live_hits == 0 {
        println!("  Run 'camdl batch run {}' to start.", toml_path);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Print the resolved sweep grid + cache summary for `batch run --dry-run`.
/// Does not simulate. Format mirrors the single-run `--dry-run` idiom in
/// main.rs: header block, per-item tables, totals.
///
/// The cache status + output paths are resolved through `CasSink::predict_cells`
/// — the SAME `cell_resolve` + `store.lookup` gate the real run path uses — so a
/// dry-run reports exactly what `run` would resolve.
#[allow(clippy::too_many_arguments)]
fn print_batch_dry_run(
    model_path: &str,
    batch_model: &ir::Model,
    model_stem: Option<&str>,
    backend: crate::args::types::ForwardBackend,
    dt: f64,
    output_dir: &str,
    parallel: usize,
    scenarios: &[ResolvedEntry],
    sweep_points: &[HashMap<String, f64>],
    seeds: &[u64],
    base_params: &HashMap<String, f64>,
    params_file: Option<&str>,
    runs_dir: &str,
    allow_degenerate_rates: bool,
    output_cols: &crate::util::OutputColumns,
    force: bool,
    job: &crate::sim_job::SimulateJob,
) {
    // Resolve every cell's real CAS identity + hit/miss exactly as the run path
    // would (`CasSink::cell_resolve` + `store.lookup`). `expand_sweep` returns
    // one null point for an empty sweep, so the grid count is exact either way.
    let points: Vec<indexmap::IndexMap<String, f64>> = if sweep_points.is_empty() {
        vec![indexmap::IndexMap::new()]
    } else {
        sweep_points.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect()
    };
    let probe = CasSink::probe(
        scenarios, batch_model, model_stem,
        backend, dt, allow_degenerate_rates, output_cols, runs_dir, force,
    );
    let predictions = probe.predict_cells(job, &points, seeds);

    eprintln!("camdl batch run (dry run)");
    eprintln!();
    eprintln!("  model:       {}", model_path);
    eprintln!("  backend:     {}", backend);
    eprintln!("  dt:          {}", dt);
    eprintln!("  output_dir:  {}", output_dir);
    eprintln!("  parallel:    {}", parallel);
    eprintln!();

    // Scenarios. For a named preset, show the RESOLVED enable/disable/set
    // read off the model (CLI review #3): pre-fix this printed
    // "baseline (baseline)", implying a resolution that never happened.
    eprintln!("Scenarios ({}):", scenarios.len());
    for sc in scenarios {
        let route_tag = match &sc.route {
            Some(_) => "[model preset]",
            None => "[ad-hoc]",
        };
        let marker = if sc.enable.is_empty() && sc.disable.is_empty() && sc.params.is_empty()
            && sc.scale.is_empty()
        {
            "(no patch — baseline identity)".to_string()
        } else {
            let mut parts = Vec::new();
            if !sc.enable.is_empty()  { parts.push(format!("enable={}",  sc.enable.join(","))); }
            if !sc.disable.is_empty() { parts.push(format!("disable={}", sc.disable.join(","))); }
            if !sc.params.is_empty() {
                let mut ks: Vec<&String> = sc.params.keys().collect();
                ks.sort();
                let kv: Vec<String> = ks.iter().map(|k| format!("{}={}", k, sc.params[*k])).collect();
                parts.push(format!("set={{{}}}", kv.join(", ")));
            }
            if !sc.scale.is_empty() {
                let kv: Vec<String> = sc.scale.iter().map(|(k, f)| format!("{k}={f}")).collect();
                parts.push(format!("scale={{{}}}", kv.join(", ")));
            }
            parts.join(" ")
        };
        eprintln!("  {:24} {} {}", sc.name, route_tag, marker);
    }
    eprintln!();

    // Sweep grid with per-point provenance
    let total_runs = predictions.len();
    let n_pts = sweep_points.len().max(1);
    eprintln!(
        "Sweep grid ({} points × {} scenarios × {} seeds = {} runs):",
        n_pts, scenarios.len(), seeds.len(), total_runs,
    );
    eprintln!();

    let src_label = |name: &str, in_sweep: bool, scenario: Option<&str>| -> String {
        if in_sweep {
            "sweep override".to_string()
        } else if let Some(sn) = scenario {
            format!("scenario '{}' set", sn)
        } else if let Some(pf) = params_file.filter(|_| base_params.contains_key(name)) {
            format!("params file: {}", pf)
        } else if base_params.contains_key(name) {
            "TOML default".to_string()
        } else {
            "model default".to_string()
        }
    };

    // Show every sweep point, each as a compact table. For points > 0,
    // only list the keys that differ from point 0 (keeps wide sweeps
    // readable when most params are constant).
    let effective_points: &[HashMap<String, f64>] = if sweep_points.is_empty() {
        &[] // no sweep → one implicit null point, handled separately
    } else {
        sweep_points
    };

    if effective_points.is_empty() {
        // No sweep: just the baseline param set
        eprintln!("  (no [sweep] — single parameter point)");
        let mut keys: Vec<&String> = base_params.keys().collect();
        keys.sort();
        for k in keys {
            eprintln!("    {:20} = {:<12}  {}", k, base_params[k], src_label(k, false, None));
        }
    } else {
        // Compute union of all keys that ever vary across sweep points.
        let mut varying_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for pt in effective_points {
            for k in pt.keys() { varying_keys.insert(k.clone()); }
        }

        for (i, pt) in effective_points.iter().enumerate() {
            eprintln!("  point {}:", i);
            if i == 0 {
                // Full table for the first point
                let mut union_keys: std::collections::BTreeSet<String> =
                    base_params.keys().cloned().collect();
                for k in pt.keys() { union_keys.insert(k.clone()); }
                for k in &union_keys {
                    let (v, in_sweep) = match pt.get(k) {
                        Some(v) => (*v, true),
                        None    => (*base_params.get(k).unwrap_or(&f64::NAN), false),
                    };
                    eprintln!("    {:20} = {:<12}  {}", k, v, src_label(k, in_sweep, None));
                }
            } else {
                // Subsequent points: only show varying-keys that differ
                for k in &varying_keys {
                    if let Some(v) = pt.get(k) {
                        eprintln!("    {:20} = {:<12}  sweep override", k, v);
                    }
                }
            }
            eprintln!();
        }
    }

    // Cache status
    let hits   = predictions.iter().filter(|c| c.hit).count();
    let misses = predictions.len() - hits;
    eprintln!("Cache status:");
    eprintln!("  {} cache hits  → skipped", hits);
    eprintln!("  {} cache misses → would simulate", misses);
    eprintln!();

    // Output destinations — the content-addressed relative path each cell
    // would land in (first few), resolved via the same `cell_resolve` the run
    // path uses (no simulation, no files written).
    eprintln!("Output paths (sims/<run_path>):");
    for cell in predictions.iter().take(6) {
        eprintln!("  [{}] {}", if cell.hit { "hit " } else { "miss" }, cell.rel);
    }
    if predictions.len() > 6 {
        eprintln!("  ... ({} more)", predictions.len() - 6);
    }
    eprintln!();
    eprintln!("(dry run — no simulation, no files written.)");
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard: batch's default output dir must equal the
    /// canonical root used by simulate / fit / voi. Before 2026-04-21
    /// these diverged — batch wrote to `./output/` while everything
    /// else wrote to `./results/` — so `camdl list` silently skipped
    /// batch output. See commit fixing `default_output_dir`.
    #[test]
    fn batch_default_output_dir_matches_canonical_root() {
        assert_eq!(default_output_dir(), crate::run_paths::DEFAULT_OUTPUT_ROOT);
    }

    /// gh#241 G3: `deny_unknown_fields` — a typo'd batch.toml key must ERROR,
    /// not silently drop (and so neither apply nor reach the CAS hash).
    /// gh#496: the render archive shells out to `camdlc render`, which accepts
    /// `.camdl` SOURCE only — while `resolve_ir_path` accepts source or a
    /// compiled `.ir.json`, so `exp.config.model` may be either. Running it at
    /// an `.ir.json` fails, and the failure surfaced in run logs as
    ///
    ///   warning: could not render model for archive: camdlc render failed:
    ///   camdlc render: parse error in .../sir_basic.ir.json
    ///
    /// which reads as "your IR file is malformed" for an expected, benign
    /// condition. Verified against the real binary:
    ///
    ///   $ camdlc render ocaml/golden/sir_basic.ir.json
    ///   camdlc render: parse error in ocaml/golden/sir_basic.ir.json
    ///   $ echo $?
    ///   1
    ///
    /// The predicate must use the same test `resolve_ir_path` applies, or the
    /// two disagree about what a "model path" is.
    #[test]
    fn ir_model_path_is_not_camdl_source() {
        assert!(model_is_camdl_source("sir.camdl"));
        assert!(model_is_camdl_source("models/nested/sir.camdl"));

        assert!(!model_is_camdl_source("ocaml/golden/sir_basic.ir.json"));
        assert!(!model_is_camdl_source("sir.ir.json"));
        // A directory named like the extension is not a source file.
        assert!(!model_is_camdl_source("some.camdl/model.ir.json"));
    }

    #[test]
    fn experiment_toml_rejects_unknown_keys() {
        let ok = "[config]\nmodel = \"m.camdl\"\n";
        assert!(toml::from_str::<ExperimentToml>(ok).is_ok(), "minimal valid config must parse");

        let bad_top = "[config]\nmodel = \"m.camdl\"\n[bogus]\nx = 1\n";
        assert!(
            toml::from_str::<ExperimentToml>(bad_top).is_err(),
            "an unknown top-level table must be rejected"
        );

        let bad_cfg = "[config]\nmodel = \"m.camdl\"\nparallell = 4\n"; // typo: parallell
        assert!(
            toml::from_str::<ExperimentToml>(bad_cfg).is_err(),
            "a typo'd [config] key must be rejected"
        );

        let bad_seeds = "[config]\nmodel = \"m.camdl\"\n[config.seeds]\nfromm = 1\n"; // typo: fromm
        assert!(
            toml::from_str::<ExperimentToml>(bad_seeds).is_err(),
            "a typo'd [config.seeds] key must be rejected"
        );

        let bad_scen = "[config]\nmodel = \"m.camdl\"\n[[scenario]]\nname = \"s\"\nenabel = []\n"; // typo: enabel
        assert!(
            toml::from_str::<ExperimentToml>(bad_scen).is_err(),
            "a typo'd [[scenario]] key must be rejected"
        );
    }

    // ── sweep expansion ──────────────────────────────────────────────────────

    #[test]
    fn sweep_linspace_expansion() {
        let mut sweep = HashMap::new();
        sweep.insert("x".to_string(), SweepSpec::Linspace {
            linspace: LinspaceSpec { min: 0.0, max: 1.0, n: 5 }
        });
        let points = expand_sweep(&sweep);
        assert_eq!(points.len(), 5);
        let vals: Vec<f64> = points.iter().map(|p| p["x"]).collect();
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[2] - 0.5).abs() < 1e-10);
        assert!((vals[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sweep_list_expansion() {
        let mut sweep = HashMap::new();
        sweep.insert("y".to_string(), SweepSpec::List(vec![1.0, 2.0, 4.0]));
        let points = expand_sweep(&sweep);
        assert_eq!(points.len(), 3);
        let vals: Vec<f64> = points.iter().map(|p| p["y"]).collect();
        assert_eq!(vals, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn sweep_cartesian_product() {
        let mut sweep = HashMap::new();
        sweep.insert("a".to_string(), SweepSpec::List(vec![1.0, 2.0]));
        sweep.insert("b".to_string(), SweepSpec::List(vec![10.0, 20.0]));
        let points = expand_sweep(&sweep);
        assert_eq!(points.len(), 4, "2 × 2 = 4");
    }

    #[test]
    fn sweep_empty_returns_one_null_point() {
        let sweep = HashMap::new();
        let points = expand_sweep(&sweep);
        assert_eq!(points.len(), 1);
        assert!(points[0].is_empty());
    }

    /// Parse a manifest whose `[sweep]` has the one entry `mu = <spec>`.
    fn parse_sweep(spec: &str) -> Result<ExperimentToml, String> {
        toml::from_str::<ExperimentToml>(&format!(
            "[config]\nmodel = \"m.camdl\"\n\n[sweep]\nmu = {spec}\n"))
            .map_err(|e| e.to_string())
    }

    /// The documented forms parse, and expand to what they say.
    #[test]
    fn sweep_documented_forms_parse_and_expand() {
        let exp = |spec: &str| {
            parse_sweep(spec).unwrap_or_else(|e| panic!("{spec}: {e}")).sweep["mu"].expand()
        };
        assert_eq!(exp("[0.1, 0.3, 0.5]"), vec![0.1, 0.3, 0.5]);
        assert_eq!(exp("[1, 2]"), vec![1.0, 2.0], "integer list entries are numbers");
        assert_eq!(exp("{ linspace = { min = 0.0, max = 1.0, n = 3 } }"), vec![0.0, 0.5, 1.0]);
        assert_eq!(exp("{ logspace = { min = 1.0, max = 100.0, n = 3 } }").len(), 3);
        assert_eq!(exp("{ range = { min = 1.0, max = 5.0, step = 0.5 } }").len(), 9);
        assert_eq!(exp("{ range = { min = 1.0, max = 3.0 } }"), vec![1.0, 2.0, 3.0],
            "step defaults to 1");
        assert_eq!(exp("{ range = { min = 1, max = 3 } }"), vec![1.0, 2.0, 3.0],
            "integer bounds are numbers");
    }

    /// gh#594 item 3. A `range` whose `step` is not positive never advanced
    /// the loop variable, so expansion pushed points until the process was
    /// killed (the gh#257 class). It is refused when the manifest is parsed —
    /// before any expansion is attempted, so this test cannot hang.
    #[test]
    fn a_range_with_a_non_positive_or_non_finite_step_is_refused_at_parse() {
        for step in ["0.0", "-0.5", "nan", "inf"] {
            let err = parse_sweep(&format!("{{ range = {{ min = 0.1, max = 0.3, step = {step} }} }}"))
                .err()
                .unwrap_or_else(|| panic!("step = {step} must be refused"));
            assert!(err.contains("step"), "step = {step}: the error names the key: {err}");
        }
    }

    /// gh#594 item 3. `range = [min, max]` with the step written beside it
    /// parsed as a one-point sweep: the array filled `min`/`max`, `step`
    /// defaulted to 1, and the outer `step` key was dropped as unknown. Keys a
    /// form does not have, and the array spelling of a table, are refused.
    #[test]
    fn a_sweep_spec_with_a_key_or_shape_its_form_does_not_have_is_refused() {
        for spec in [
            "{ range = [0.1, 0.3], step = 0.1 }",
            "{ range = [0.1, 0.3, 0.1] }",
            "{ range = { min = 0.1, max = 0.3, stepp = 0.1 } }",
            "{ linspace = { min = 0.0, max = 1.0, n = 3, step = 0.5 } }",
            "{ linspace = { min = 0.0, max = 1.0, n = 3 }, n = 4 }",
            "{ grid = [1.0, 2.0] }",
        ] {
            assert!(parse_sweep(spec).is_err(), "{spec} must be refused, not parsed");
        }
    }

    /// The other ways a sweep spec silently expands to no points or to NaN:
    /// an empty list, `n = 0`, a `range` whose `max` is below its `min`, a
    /// non-positive `logspace` bound. Each would run zero cells, or cells at
    /// NaN, with exit 0.
    #[test]
    fn a_sweep_spec_that_expands_to_nothing_or_nan_is_refused() {
        for spec in [
            "[]",
            "[1.0, nan]",
            "{ linspace = { min = 0.0, max = 1.0, n = 0 } }",
            "{ linspace = { min = 0.0, max = inf, n = 3 } }",
            "{ logspace = { min = 0.0, max = 1.0, n = 3 } }",
            "{ range = { min = 3.0, max = 1.0, step = 1.0 } }",
        ] {
            assert!(parse_sweep(spec).is_err(), "{spec} must be refused, not parsed");
        }
    }
}
