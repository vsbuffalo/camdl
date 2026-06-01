//! `batch run FILE` subcommand — the multi-scenario / sweep runner
//! behind `camdl batch run`. Writes content-addressed output to
//! `<output_dir>/sims/{sim_hash}/{scen_slug}-{scen_hash}/seed_{n}/` in the
//! same layout `--cas` uses for single runs, so `camdl list/show/cat`
//! browse both uniformly.
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use crate::util::{run_simulation, write_traj_tsv, load_params_toml, resolve_ir_path, SimRun};
use crate::hashing::{model_hash, sim_hash, scen_hash, canonical_params};
use crate::sampling::{generate_design, describe_prior, DesignParam};
use ir::parameter::PriorDist;
use crate::cas;
use crate::version;

// ─── TOML schema (v1 — see module-level doc) ─────────────────────────────────

#[derive(Debug, Deserialize)]
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
}

/// `[obs]` section — synthetic observation output for the batch ensemble.
/// `enabled = true` samples each run's observation streams and writes them
/// into the CAS obs subtree (`seed_N/obs/{obs_hash}-{obs_seed}/<stream>.tsv`,
/// the designed layout from `cas/mod.rs`). Resolves CLI review finding #4.
#[derive(Debug, Deserialize, Default)]
struct ObsSection {
    #[serde(default)]
    enabled: bool,
}

// ─── Design specification ─────────────────────────────────────────────────────

/// A named experimental design block (`[design.NAME]`).
/// Represents a named belief state: parameter ranges + sampling method.
#[derive(Debug, Deserialize)]
struct DesignBlock {
    method: String,   // "sobol" | "lhs" | "random"
    n: usize,
    #[serde(default)]
    parameters: HashMap<String, DesignParamToml>,
}

/// Per-parameter specification within a design block.
#[derive(Debug, Deserialize)]
struct DesignParamToml {
    range: RangeMinMax,
    #[serde(default)]
    transform: Option<String>,   // "log" | "logit" | None (linear)
    #[serde(default)]
    prior: Option<PriorDist>,    // prior distribution for VOI importance weighting
}

#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SweepSpec {
    List(Vec<f64>),
    Linspace { linspace: LinspaceSpec },
    Logspace  { logspace: LinspaceSpec },
    Range     { range: RangeSpec },
}

#[derive(Debug, Deserialize)]
struct LinspaceSpec {
    min: f64,
    max: f64,
    n: usize,
}

#[derive(Debug, Deserialize)]
struct RangeSpec {
    min: f64,
    max: f64,
    #[serde(default = "default_step")]
    step: f64,
}

fn default_step() -> f64 { 1.0 }

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
struct ConfigSection {
    model: String,
    #[serde(default)]
    params: Option<String>,
    #[serde(default)]
    geo: Option<String>,
    #[serde(default = "default_backend")]
    backend: crate::args::types::Backend,
    #[serde(default = "default_dt")]
    dt: f64,
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_parallel")]
    parallel: usize,
    #[serde(default)]
    seeds: SeedsSection,
}

fn default_backend() -> crate::args::types::Backend {
    crate::args::types::Backend::ChainBinomial
}
fn default_dt() -> f64 { 1.0 }
fn default_output_dir() -> String { crate::run_paths::DEFAULT_OUTPUT_ROOT.to_string() }
fn default_parallel() -> usize { 1 }

#[derive(Debug, Deserialize, Default)]
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
pub struct ScenarioEntry {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, f64>,
    #[serde(default)]
    pub enable: Vec<String>,
    #[serde(default)]
    pub disable: Vec<String>,
}

/// A `[[scenario]]` entry after resolution against the model's
/// `scenarios{}` presets (CLI review finding #3).
///
/// `route` decides how the per-run `SimRun` is built:
///   - `Some(preset_name)` → route through the named-preset branch of
///     `params_resolver` (`scenario_name = Some(...)`), exactly as
///     `simulate --scenario` does. The model preset is the source of
///     truth for params/enable/disable/scale/compose.
///   - `None` → ad-hoc patch (`scenario_name = None`) using the inline
///     enable/disable/params.
///
/// `enable`/`disable`/`params` are the *hash-relevant* delta — for a
/// preset they are read off the preset (mirroring `prepare_cas_ctx`) so
/// the CAS path stays consistent with the single-run `--cas` layout.
#[derive(Debug, Clone)]
pub struct ResolvedEntry {
    pub name: String,
    pub route: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
    pub params: HashMap<String, f64>,
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
                    // Hash-relevant delta = the preset's own
                    // enable/disable/params, matching prepare_cas_ctx
                    // so batch and single-run --cas agree on layout.
                    let preset = model.presets.iter()
                        .find(|p| p.name == name)
                        .expect("resolve_scenario_ref confirmed preset exists");
                    Ok(ResolvedEntry {
                        name,
                        route: Some(preset.name.clone()),
                        enable: preset.enable.clone(),
                        disable: preset.disable.clone(),
                        params: preset.params.clone(),
                    })
                }
                ResolvedScenario::Adhoc { name, enable, disable, params } => {
                    Ok(ResolvedEntry {
                        name,
                        route: None,
                        enable,
                        disable,
                        params: params.into_iter().collect(),
                    })
                }
            }
        })
        .collect()
}

// ─── Run planning ─────────────────────────────────────────────────────────────

/// Whether a planned run should be skipped (cache hit) or executed (cache miss).
#[derive(Debug, PartialEq)]
pub enum RunDecision {
    /// traj.tsv already exists and --force was not set; cached result will be reused.
    CacheHit,
    /// traj.tsv is absent or --force was set; this run must be executed.
    CacheMiss,
}

/// A fully-resolved description of one (sweep_point, scenario, seed) run,
/// including its cache decision. Produced by `plan_runs` before any simulation
/// is started.
#[derive(Debug)]
pub struct RunPlan {
    pub scenario: String,
    pub seed: u64,
    /// Sweep parameter overrides for this run (empty if no sweep).
    pub sweep_overrides: HashMap<String, f64>,
    /// Index of the design/sweep point (0-based). Used by design experiments
    /// to write run.json so analyze can recover point_id without hashing.
    pub point_idx: usize,
    /// Path relative to runs/: {sim_hash_8}/{scenario_slug}-{scen_hash_8}/seed_{seed}
    pub run_path: String,
    /// Absolute path to the run directory.
    pub run_dir: String,
    pub decision: RunDecision,
}

/// Classify every (sweep_point, scenario, seed) triple as CacheHit or CacheMiss
/// by inspecting the filesystem. Does not simulate or write anything.
///
/// `sweep_points` is a list of parameter override maps from `[sweep]`. Pass
/// `&[HashMap::new()]` (one empty map) when there is no sweep.
///
/// `shash` must be the full 64-char hex sim_hash; only the first 8 chars are
/// used in paths. `runs_dir` is the absolute path to the runs/ subdirectory.
pub fn plan_runs(
    scenarios: &[ScenarioEntry],
    sweep_points: &[HashMap<String, f64>],
    seeds: &[u64],
    shash: &str,
    model_stem: Option<&str>,
    runs_dir: &str,
    force: bool,
) -> Vec<RunPlan> {
    let effective_points: &[HashMap<String, f64>] = if sweep_points.is_empty() {
        &[HashMap::new()]
    } else {
        sweep_points
    };

    let mut plans = Vec::with_capacity(effective_points.len() * scenarios.len() * seeds.len());
    for (pt_idx, sweep) in effective_points.iter().enumerate() {
        for sc in scenarios {
            // Merge sweep overrides into scenario params for hashing
            let mut merged_params = sc.params.clone();
            merged_params.extend(sweep.iter().map(|(k, v)| (k.clone(), *v)));

            let sc_hash = scen_hash(&sc.enable, &sc.disable, &merged_params);
            for &seed in seeds {
                // `runs_dir` is already the `<root>/sims` subtree; the
                // `sim_run_rel` helper produces the trailing three-segment
                // relative path (stem-<sim_hash>/slug-<scen_hash>/seed_N)
                // and stays byte-identical to the single-run --cas path.
                let run_path = crate::run_paths::sim_run_rel(
                    model_stem, shash, &sc.name, &sc_hash, seed,
                );
                let run_dir  = format!("{}/{}", runs_dir, run_path);
                let traj_exists = std::path::Path::new(&format!("{}/traj.tsv", run_dir)).exists();
                let decision = if !force && traj_exists {
                    RunDecision::CacheHit
                } else {
                    RunDecision::CacheMiss
                };
                plans.push(RunPlan {
                    scenario: sc.name.clone(),
                    seed,
                    sweep_overrides: sweep.clone(),
                    point_idx: pt_idx,
                    run_path,
                    run_dir,
                    decision,
                });
            }
        }
    }
    plans
}

// ─── Manifest / run metadata ─────────────────────────────────────────────────

// RunMeta is the shared cas::RunMeta — both single-run `--cas` and
// batch `--batch` write the same schema so `camdl list/show/cat` reads
// both uniformly.

/// Minimal descriptor for one completed run, included in manifest.json.
/// The web app uses run_path to construct the URL: /runs/{run_path}/traj.tsv
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunEntry {
    scenario: String,
    seed: u64,
    run_path: String,
    /// Mirrors RunMeta.sweep_point — convenient for aggregating without
    /// reading every run.json.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    sweep_point: HashMap<String, f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ParamsProvenance {
    source: String,
    content_hash: Option<String>,
    input_hash: Option<String>,
    verified: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    model: String,
    scenarios: Vec<String>,
    seeds: Vec<u64>,
    total_runs: usize,
    completed: usize,
    output_dir: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    geo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    params_provenance: Option<ParamsProvenance>,
    /// Completed runs. run_path is relative to runs/ and used by the web app
    /// to fetch trajectories: GET /runs/{run_path}/traj.tsv
    runs: Vec<RunEntry>,
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

    let (ir_path_resolved, _tmpfile) = resolve_ir_path(&model_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let ir_json = std::fs::read_to_string(&ir_path_resolved).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", ir_path_resolved, e);
        std::process::exit(1);
    });
    let mhash = model_hash(&ir_json);

    let mut params_provenance: Option<ParamsProvenance> = None;
    let base_params: HashMap<String, f64> = if let Some(ref pf) = exp.config.params {
        // Check provenance if the params file has a content hash header
        let prov = match crate::fit::provenance::verify_content_hash(pf) {
            Ok(crate::fit::provenance::ContentVerification::Valid) => {
                eprintln!("params: {} \x1b[32m✓ provenance verified\x1b[0m", pf);
                // Extract input_hash from comment header
                let input_hash = std::fs::read_to_string(pf).ok()
                    .and_then(|s| s.lines()
                        .find(|l| l.starts_with("# Input hash:"))
                        .and_then(|l| l.split_whitespace().nth(3))
                        .map(|s| s.to_string()));
                let content_hash = std::fs::read_to_string(pf).ok()
                    .and_then(|s| s.lines()
                        .find(|l| l.starts_with("# Content hash:"))
                        .and_then(|l| l.split_whitespace().nth(3))
                        .map(|s| s.to_string()));
                Some(ParamsProvenance {
                    source: pf.clone(),
                    content_hash,
                    input_hash,
                    verified: true,
                })
            }
            Ok(crate::fit::provenance::ContentVerification::Modified { declared, computed }) => {
                eprintln!("\x1b[33mwarning: params file {} has been modified since inference produced it.\x1b[0m", pf);
                eprintln!("  Content hash mismatch: expected {}, got {}", declared, computed);
                Some(ParamsProvenance {
                    source: pf.clone(),
                    content_hash: Some(computed),
                    input_hash: None,
                    verified: false,
                })
            }
            _ => {
                // No provenance header — standalone params file, that's fine
                None
            }
        };
        params_provenance = prov;
        load_params_toml(pf).unwrap_or_else(|e| {
            eprintln!("error: cannot load params {}: {}", pf, e);
            std::process::exit(1);
        })
    } else {
        HashMap::new()
    };
    let shash = sim_hash(&mhash, &canonical_params(&base_params), backend.as_str(), dt);

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

    // Expand [design.*] blocks into parameter points (writes parameter_points.tsv per design).
    if !exp.design.is_empty() {
        // Resolve scenarios before consuming exp
        let scenarios: Vec<ScenarioEntry> = if exp.scenario.is_empty() {
            vec![ScenarioEntry { name: "baseline".to_string(), params: HashMap::new(), enable: vec![], disable: vec![] }]
        } else {
            exp.scenario.clone()
        };
        run_design_experiment(scenarios, exp.design, &ir_path_resolved, &output_dir, &shash,
                              backend, dt, a.force, parallel, &params_file_opt, &seeds);
        return;
    }

    // Expand [sweep] into parameter points (empty sweep → one null point).
    let sweep_points = expand_sweep(&exp.sweep);
    let has_sweep = !exp.sweep.is_empty();

    let raw_scenarios: Vec<ScenarioEntry> = if exp.scenario.is_empty() {
        vec![ScenarioEntry { name: "baseline".to_string(), params: HashMap::new(), enable: vec![], disable: vec![] }]
    } else {
        exp.scenario
    };

    // Resolve each [[scenario]] against the model's scenarios{} presets
    // (CLI review #3). A name matching a preset routes through the same
    // params_resolver preset path simulate --scenario uses; a name with
    // inline patches is ad-hoc; an unknown name with no patches is a hard
    // error. The model is parsed from the IR JSON already in hand.
    let batch_model: ir::Model = ir::from_str(&ir_json).unwrap_or_else(|e| {
        eprintln!("error: cannot parse model IR for scenario resolution: {}", e);
        std::process::exit(1);
    });
    let resolved_scenarios = resolve_batch_scenarios(&raw_scenarios, &batch_model)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    // Hash-only view: enable/disable/params are the resolved delta so the
    // CAS path is byte-identical to single-run --cas for preset scenarios.
    let scenarios: Vec<ScenarioEntry> = resolved_scenarios.iter().map(|r| ScenarioEntry {
        name: r.name.clone(),
        params: r.params.clone(),
        enable: r.enable.clone(),
        disable: r.disable.clone(),
    }).collect();

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

    let runs_dir = format!("{}/sims", output_dir);

    let model_stem = crate::hashing::path_stem_slug(&ir_path_resolved);
    let plans = plan_runs(&scenarios, &sweep_points, &seeds, &shash,
        model_stem.as_deref(), &runs_dir, a.force);
    let total = plans.len();

    if a.dry_run {
        print_batch_dry_run(
            &model_path, backend, dt, &output_dir, parallel,
            &resolved_scenarios, &sweep_points, &seeds, &base_params,
            exp.config.params.as_deref(), &plans,
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

    let geo_url: Option<String> = if let Some(ref geo_src) = exp.config.geo {
        let geo_dest = format!("{}/geo/boundaries.geojson", output_dir);
        match std::fs::create_dir_all(format!("{}/geo", output_dir))
            .and_then(|_| std::fs::copy(geo_src, &geo_dest))
        {
            Ok(_) => Some("geo/boundaries.geojson".to_string()),
            Err(e) => { eprintln!("warning: could not copy geo file '{}': {}", geo_src, e); None }
        }
    } else {
        None
    };

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

    let scenario_names: Vec<String> = scenarios.iter().map(|s| s.name.clone()).collect();

    if parallel > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build_global();
    }

    // ── Build the SimulateJob and route through the unified engine ──────────
    //
    // `batch run` is now a thin TOML front-end over `engine::run_job` — the
    // SAME engine `camdl simulate` uses (run-spec §3.1). The per-cell seed
    // arithmetic and SimRun construction are shared; the CAS-tree output
    // shape lives in `CasSink`, which reuses the existing `SimulateInputs`
    // hashing so the on-disk layout / content-hashes stay byte-identical to
    // the pre-unification batch path.
    //
    // Scenario routing: a resolved preset → `ScenarioRef::Named` (the
    // params_resolver preset path); an ad-hoc patch → `ScenarioRef::Inline`.
    use crate::sim_job::{ParamSource, ScenarioRef, Seeds, SimulateJob};
    let job_scenarios: Vec<ScenarioRef> = resolved_scenarios.iter().map(|r| {
        match &r.route {
            Some(preset_name) => ScenarioRef::Named(preset_name.clone()),
            None => ScenarioRef::Inline {
                name: r.name.clone(),
                enable: r.enable.clone(),
                disable: r.disable.clone(),
                params: r.params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            },
        }
    }).collect();

    // ParamSource: a non-empty [sweep] → Sweep over the expanded points; an
    // empty sweep → Point (the single null point). Batch base params come
    // from the params file (M layer); sweep points override per cell.
    let source = if has_sweep {
        let points: Vec<indexmap::IndexMap<String, f64>> = sweep_points.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect();
        ParamSource::Sweep { points }
    } else {
        ParamSource::Point
    };

    let job = SimulateJob {
        model: ir_path_resolved.clone(),
        params_files: params_file_opt.as_ref().map(|p| vec![p.clone()]).unwrap_or_default(),
        backend,
        dt,
        source,
        scenarios: job_scenarios,
        // Batch seeds are always explicit (range / count / list).
        seeds: Seeds::Explicit(seeds.clone()),
        cli_overrides: Vec::new(),
        set_vec_entries: Vec::new(),
        table_files: Vec::new(),
        // batch keeps its CAS-per-cell obs ensemble (CasSink), not the
        // combined-file ObsOutput modes — leave None here.
        obs: crate::sim_job::ObsOutput::None,
        parallel,
    };

    let mut sink = CasSink {
        resolved_scenarios: resolved_scenarios.clone(),
        model_path: ir_path_resolved.clone(),
        model_stem: model_stem.clone(),
        base_model: batch_model.clone(),
        base_params: base_params.clone(),
        backend,
        dt,
        allow_degenerate_rates: a.allow_degenerate_rates,
        runs_dir: runs_dir.clone(),
        obs_enabled,
        force: a.force,
        total,
        counter: 0,
        completed_runs: Vec::new(),
        errors: Vec::new(),
    };

    crate::engine::run_job(&job, &mut sink).unwrap_or_else(|e| {
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
    let manifest = Manifest {
        model: model_path,
        scenarios: scenario_names,
        seeds,
        total_runs: total,
        completed,
        output_dir: output_dir.clone(),
        geo: geo_url,
        params_provenance,
        runs: completed_runs,
    };
    // Manifest lives under `sims/` so the output root contains only
    // the two subtree roots (sims/, fits/) plus optional geo/. Was
    // `<output>/manifest.json` before 2026-04-19.
    let manifest_path = format!("{}/sims/manifest.json", output_dir);
    if let Some(parent) = std::path::Path::new(&manifest_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap_or_default())
        .unwrap_or_else(|e| eprintln!("warning: could not write manifest.json: {}", e));

    eprintln!("Done: {}/{} runs completed. Manifest: {}", completed, total, manifest_path);
    if !errors.is_empty() { std::process::exit(1); }
}

// ─── CasSink: batch's content-addressed output strategy ───────────────────────

/// `RunSink` for `camdl batch run`: writes each cell into the
/// content-addressed CAS tree (`sims/<sim>/<scen>-<scen_hash>/seed_N/`),
/// reusing the existing [`crate::cas::sim_inputs::SimulateInputs`] hashing
/// so the on-disk layout, `run.json` content-hash, and `kind.sweep_point`
/// are byte-identical to the pre-unification batch path. Cache hits are
/// skipped via `should_run` (the engine never simulates them).
struct CasSink {
    /// Resolved `[[scenario]]` entries, looked up by name for the
    /// hash-relevant enable/disable/params delta.
    resolved_scenarios: Vec<ResolvedEntry>,
    model_path: String,
    model_stem: Option<String>,
    /// The **base** model — its whole-IR digest is the (constant-across-cells)
    /// model level. Never `cell.model` (which has scenario + sweep applied).
    base_model: ir::Model,
    /// Resolved base parameter values; per cell, the sweep point is layered on
    /// top into the `params` level (a resolved value, not the scenario delta).
    base_params: HashMap<String, f64>,
    backend: crate::args::types::Backend,
    dt: f64,
    allow_degenerate_rates: bool,
    /// Absolute `<output>/sims` subtree.
    runs_dir: String,
    obs_enabled: bool,
    force: bool,
    total: usize,
    counter: usize,
    completed_runs: Vec<RunEntry>,
    errors: Vec<String>,
}

impl CasSink {
    /// The store root (`<output>`, the parent of `<output>/sims`).
    fn root(&self) -> std::path::PathBuf {
        std::path::Path::new(&self.runs_dir)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    /// Resolve a cell's trajectory identity + factored store path.
    ///
    /// The model level is the **base** model digest (constant across the
    /// sweep — never `cell.model`, which has scenario + sweep applied). The
    /// sweep point goes into the `params` level (base params ∪ sweep, a
    /// resolved value — not the scenario delta, not the model hash). The
    /// named scenario's enable/disable/params is the `scenario` level.
    /// `process_seed` is engine-resolved (batch seeds are explicit).
    fn cell_resolve(
        &self,
        spec: &crate::engine::CellSpec,
    ) -> Result<(crate::resolve::ResolvedTrajectory, std::path::PathBuf, String), String> {
        let name = spec.scenario.name();
        let resolved = self.resolved_scenarios.iter()
            .find(|s| s.name == name || s.route.as_deref() == Some(name))
            .ok_or_else(|| format!("cell scenario '{}' not among resolved batch scenarios", name))?;

        let mut params = self.base_params.clone();
        for (k, v) in &spec.point_overrides {
            params.insert(k.clone(), *v);
        }

        let scenario_label = if resolved.name.is_empty() { "baseline" } else { resolved.name.as_str() };
        let param_label = if spec.point_overrides.is_empty() {
            "base".to_string()
        } else {
            let mut sw: Vec<(&String, &f64)> = spec.point_overrides.iter().collect();
            sw.sort_by(|a, b| a.0.cmp(b.0));
            sw.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join("_")
        };

        let ctx = crate::resolve::TrajectoryCtx {
            model: &self.base_model,
            model_stem: self.model_stem.as_deref().unwrap_or("model"),
            ir_version: ir::IR_VERSION.trim(),
            engine_version: version::VERSION_SHORT,
            backend: self.backend,
            dt: self.dt,
            t_start: self.base_model.simulation.t_start,
            t_end: self.base_model.simulation.t_end,
            output: &self.base_model.output.times,
            allow_degenerate_rates: self.allow_degenerate_rates,
            base_params: &params,
            table_digests: Vec::new(),
            enable: &resolved.enable,
            disable: &resolved.disable,
            scen_params: &resolved.params,
            param_label: &param_label,
            scenario_label,
            base_seed: spec.process_seed,
            process_seed: spec.process_seed,
        };
        let rt = crate::resolve::resolve_trajectory(&ctx).map_err(|e| format!("resolve error: {e}"))?;
        let root = self.root();
        let dir = runid::store_path(&root, runid::ArtifactKind::Sim, &rt.levels);
        let rel = dir.strip_prefix(&root).unwrap_or(&dir).to_string_lossy().into_owned();
        Ok((rt, dir, rel))
    }
}

impl crate::engine::RunSink for CasSink {
    fn should_run(&mut self, spec: &crate::engine::CellSpec) -> bool {
        if self.force {
            return true;
        }
        match self.cell_resolve(spec) {
            Ok((rt, dir, _)) => {
                let store = runid::FsCasStore::new(self.root());
                !matches!(
                    store.lookup(&dir, &runid::LeafIdentity::new(rt.run_id)),
                    runid::Lookup::Hit(_)
                )
            }
            // No cache key without a resolve → run; the error surfaces in merge_cell.
            Err(_) => true,
        }
    }

    fn on_skip(&mut self, spec: &crate::engine::CellSpec) {
        let rel = self.cell_resolve(spec).map(|(_, _, rel)| rel).unwrap_or_default();
        let name = spec.scenario.name().to_string();
        self.counter += 1;
        eprintln!("[{}/{}] scenario={} seed={} (skipped — cache hit)",
            self.counter, self.total, name, spec.process_seed);
        self.completed_runs.push(RunEntry {
            scenario: name,
            seed: spec.process_seed,
            run_path: rel,
            sweep_point: spec.point_overrides.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        });
    }

    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let spec = &cell.spec;
        let name = spec.scenario.name().to_string();

        let (rt, dir, rel) = match self.cell_resolve(spec) {
            Ok(x) => x,
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: {}", name, spec.process_seed, e));
                return Ok(());
            }
        };

        let traj_bytes = crate::util::traj_tsv_bytes(&cell.model, &cell.traj, true);

        // Declare the obs ensemble as a child sub-artifact so the leaf's
        // exact-set doesn't orphan the `obs/` subdir (written after the
        // trajectory commits). M2-interim: the obs child is recorded + its
        // streams written via the existing ensemble writer; a full
        // obs-child RunRecord identity is a follow-up.
        let mut children: std::collections::BTreeMap<String, Vec<runid::ContentHash>> =
            std::collections::BTreeMap::new();
        let has_obs = self.obs_enabled && !cell.model.observations.is_empty();
        if has_obs {
            let obs_seed = spec.process_seed ^ crate::util::SEED_MIX_OBS;
            let obs_json = serde_json::to_string(&cell.model.observations).unwrap_or_default();
            let obs_hash = crate::hashing::sha256_hex(obs_json.as_bytes());
            let obs_id = runid::ContentHash::digest_bytes(
                format!("{}:{}:{}", rt.run_id.to_hex(), obs_seed, obs_hash).as_bytes(),
            );
            children.insert("obs".to_string(), vec![obs_id]);
        }

        let record = runid::RunRecord {
            format_version: runid::FORMAT_VERSION,
            kind: runid::ArtifactKind::Sim,
            run_id: rt.run_id,
            hash_version: runid::HASH_VERSION,
            ir_version: ir::IR_VERSION.trim().to_string(),
            engine_version: version::VERSION_SHORT.to_string(),
            levels: rt.levels,
            deps: Vec::new(),
            status: runid::RunStatus::Running,
            artifacts: Default::default(),
            children,
            inputs: serde_json::Value::Null,
            provenance: runid::Provenance {
                argv: std::env::args().collect(),
                created_at: Some(cas::iso8601_utc(std::time::SystemTime::now())),
                source_paths: vec![self.model_path.clone()],
                camdl_version: Some(version::VERSION_SHORT.to_string()),
                ..Default::default()
            },
        };

        let mut artifacts = runid::Artifacts::new();
        artifacts.insert("traj.tsv", traj_bytes);
        let store = runid::FsCasStore::new(self.root());
        let dest = match store.commit_atomic(&dir, record, artifacts) {
            Ok(d) => d,
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: commit failed: {}",
                    name, spec.process_seed, e));
                return Ok(());
            }
        };

        // Obs ensemble: written into the committed leaf's `obs/` child. A
        // failure here is non-fatal — children are independent, so a missing
        // obs child never staleens the parent trajectory.
        if has_obs {
            if let Err(e) = write_obs_into_cas(&dest, &cell.model, &cell.traj, spec.process_seed) {
                self.errors.push(format!("scenario={} seed={}: obs ensemble: {}",
                    name, spec.process_seed, e));
            }
        }

        self.counter += 1;
        eprintln!("[{}/{}] scenario={} seed={}", self.counter, self.total, name, spec.process_seed);
        self.completed_runs.push(RunEntry {
            scenario: name,
            seed: spec.process_seed,
            run_path: rel,
            sweep_point: spec.point_overrides.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        });
        Ok(())
    }
}

// ─── Design experiment execution ─────────────────────────────────────────────

/// Run a design-based experiment (VOI/sensitivity analysis).
///
/// For each named design:
///   1. Generate parameter points via the specified method (sobol/lhs/random)
///   2. Write `{output_dir}/designs/{design}/parameter_points.tsv`
///   3. Run all (point, scenario, seed) combinations
///   4. Collect summary outputs → `outputs.tsv` (consumed downstream by `camdl voi`)
#[allow(clippy::too_many_arguments)]
fn run_design_experiment(
    scenarios: Vec<ScenarioEntry>,
    designs: HashMap<String, DesignBlock>,
    ir_path: &str,
    output_dir: &str,
    shash: &str,
    backend: crate::args::types::Backend,
    dt: f64,
    force: bool,
    parallel: usize,
    params_file_opt: &Option<String>,
    seeds: &[u64],
) {
    // Sort design names for deterministic output
    let mut design_names: Vec<&String> = designs.keys().collect();
    design_names.sort();

    for design_name in &design_names {
        let block = &designs[*design_name];
        eprintln!("Design '{}': method={} n={} parameters={}",
            design_name, block.method, block.n, block.parameters.len());

        // Build sorted parameter list
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

        // Generate design points
        let design_result = generate_design(&params, block.n, &block.method);
        let n_points = design_result.points.len();
        eprintln!("  Generated {} parameter points", n_points);

        // Write parameter_points.tsv
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

        // Write priors.txt if any parameter has a prior specification
        let priors_txt = build_priors_txt(&params);
        if let Some(txt) = priors_txt {
            let priors_path = format!("{}/priors.txt", design_dir);
            let _ = std::fs::write(&priors_path, txt);
        }

        // Run all (point, scenario, seed) combinations
        let runs_dir = format!("{}/designs/{}/sims", output_dir, design_name);
        std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| {
            eprintln!("error: cannot create runs dir {}: {}", runs_dir, e);
            std::process::exit(1);
        });

        // Annotate each point with its index for run.json
        let sweep_points = &design_result.points;
        let design_stem = crate::hashing::path_stem_slug(ir_path);
        let plans = plan_runs(&scenarios, sweep_points, seeds, shash,
            design_stem.as_deref(), &runs_dir, force);
        let total = plans.len();
        let counter = Arc::new(AtomicUsize::new(0));

        if parallel > 0 {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build_global();
        }

        {
            plans.par_iter().for_each(|plan| {
                if plan.decision == RunDecision::CacheHit {
                    counter.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                let sc = scenarios.iter().find(|s| s.name == plan.scenario).unwrap();
                let mut overrides_map: HashMap<String, f64> = plan.sweep_overrides.clone();
                overrides_map.extend(sc.params.iter().map(|(k, v)| (k.clone(), *v)));

                let sim_run = SimRun {
                    ir_path: ir_path.to_string(),
                    params_files: params_file_opt.as_ref().map(|p| vec![p.clone()]).unwrap_or_default(),
                    overrides: overrides_map,
                    scenario_name: None,
                    adhoc_enable: sc.enable.clone(),
                    adhoc_disable: sc.disable.clone(),
                    backend,
                    dt,
                    seed: plan.seed,
                    ..Default::default()
                };

                match run_simulation(&sim_run) {
                    Err(e) => {
                        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!("[{}/{}] design={} scenario={} seed={} ERROR: {}",
                            n, total, design_name, plan.scenario, plan.seed, e);
                    }
                    Ok((traj, model)) => {
                        if let Err(e) = std::fs::create_dir_all(&plan.run_dir) {
                            eprintln!("error: cannot create {}: {}", plan.run_dir, e);
                            return;
                        }
                        if let Err(e) = write_traj_tsv(&format!("{}/traj.tsv", plan.run_dir), &model, &traj, true) {
                            eprintln!("error: cannot write traj.tsv in {}: {}", plan.run_dir, e);
                            return;
                        }
                        // Write run.json so summarize can recover (point_id, scenario, seed)
                        // without parsing directory names.
                        let run_json = format!(
                            "{{\"design_point_index\":{},\"scenario\":{},\"seed\":{}}}\n",
                            plan.point_idx,
                            serde_json::to_string(&plan.scenario).unwrap_or_default(),
                            plan.seed,
                        );
                        let _ = std::fs::write(format!("{}/run.json", plan.run_dir), run_json);
                        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                        eprintln!("[{}/{}] design={} scenario={} seed={}", n, total, design_name, plan.scenario, plan.seed);
                    }
                }
            });
        }
        eprintln!("Design '{}' complete.", design_name);
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

// ─── Observation ensemble writer (CLI review #4) ───────────────────────────

/// Sample synthetic observations for one completed run and write them into
/// the CAS obs subtree: `run_dir/obs/{obs_hash[:8]}-{obs_seed}/<stream>.tsv`
/// plus an `obs.json` provenance file (the layout designed in
/// `cas/mod.rs`).
///
/// Reuses the *same* sampling primitives as `simulate --obs`
/// (`compile_obs_sample_pf`, `project_all_obs_times`, `obs_schedule_times`,
/// `snap_at`) and the canonical obs RNG derivation
/// (`process_seed ^ SEED_MIX_OBS`), so a given seed produces the same
/// observation bytes regardless of which entry point generated them. One
/// file per stream means multi-cadence streams are handled correctly with
/// no single-file kludge.
///
/// `process_seed` is the run's simulation seed; in this single-realization
/// path it *is* the `obs_seed` recorded in the directory name (run-spec
/// §"Gating risk": a future `[obs] replicates = K` fans the obs layer by
/// mixing K distinct obs seeds, leaving the trajectory RNG untouched).
fn write_obs_into_cas(
    run_dir: &std::path::Path,
    model: &ir::Model,
    traj: &sim::Trajectory,
    process_seed: u64,
) -> Result<(), String> {
    use std::io::Write;

    if model.observations.is_empty() {
        return Ok(());
    }

    // obs_hash = hash of the resolved observation blocks only (run-spec:
    // changing a reporting parameter re-samples obs without invalidating
    // the cached trajectory). Canonical JSON of model.observations.
    let obs_json = serde_json::to_string(&model.observations)
        .map_err(|e| format!("cannot serialize observations for hashing: {}", e))?;
    let obs_hash = crate::hashing::sha256_hex(obs_json.as_bytes());
    let obs_seed = process_seed ^ crate::util::SEED_MIX_OBS;

    let obs_dir = run_dir.join("obs").join(format!(
        "{}-{}", &obs_hash[..8.min(obs_hash.len())], obs_seed,
    ));
    std::fs::create_dir_all(&obs_dir)
        .map_err(|e| format!("cannot create {}: {}", obs_dir.display(), e))?;

    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model.clone())
            .map_err(|e| format!("model compile error for obs: {:?}", e))?,
    );
    let params = compiled.default_params.clone();
    // One obs RNG, consumed in declaration order across streams — the same
    // order simulate's --obs loop uses (main.rs).
    let mut obs_rng = sim::rng::StatefulRng::new(obs_seed);

    let mut stream_names: Vec<String> = Vec::new();
    for obs_ir in &model.observations {
        let sampler = sim::inference::obs_model::compile_obs_sample_pf(
            obs_ir, compiled.clone(), &params,
        );
        let obs_times = crate::obs_schedule_times(
            &obs_ir.schedule,
            model.simulation.t_start,
            model.simulation.t_end,
        );
        let projected = crate::project_all_obs_times(traj, obs_ir, model, &obs_times);

        let path = obs_dir.join(format!("{}.tsv", obs_ir.name));
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&path)
                .map_err(|e| format!("cannot create {}: {}", path.display(), e))?,
        );
        writeln!(out, "time\t{}", obs_ir.name).map_err(|e| e.to_string())?;
        for (ti, &obs_t) in obs_times.iter().enumerate() {
            let snap = crate::snap_at(traj, obs_t);
            let draw = sampler(projected[ti], obs_t, &snap.int_state.counts, &mut obs_rng);
            if draw == draw.round() && draw.abs() < 1e15 {
                writeln!(out, "{}\t{}", obs_t, draw as i64).map_err(|e| e.to_string())?;
            } else {
                writeln!(out, "{}\t{:.6}", obs_t, draw).map_err(|e| e.to_string())?;
            }
        }
        out.flush().map_err(|e| e.to_string())?;
        stream_names.push(obs_ir.name.clone());
    }

    // obs.json provenance: the inputs that produced this obs draw.
    let obs_meta = serde_json::json!({
        "obs_hash": obs_hash,
        "obs_seed": obs_seed,
        "process_seed": process_seed,
        "streams": stream_names,
        "version": version::VERSION_SHORT,
    });
    std::fs::write(
        obs_dir.join("obs.json"),
        serde_json::to_string_pretty(&obs_meta).unwrap_or_default(),
    )
    .map_err(|e| format!("cannot write obs.json: {}", e))?;

    Ok(())
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

    let output_dir   = exp.config.output_dir.clone();
    let manifest_path = format!("{}/sims/manifest.json", output_dir);

    if let Ok(src) = std::fs::read_to_string(&manifest_path) {
        if let Ok(manifest) = serde_json::from_str::<Manifest>(&src) {
            println!("Experiment status for: {}", toml_path);
            println!("  Model:      {}", manifest.model);
            println!("  Output dir: {}", manifest.output_dir);
            println!("  Scenarios:  {}", manifest.scenarios.join(", "));
            println!("  Seeds:      {} total ({:?}..={:?})",
                manifest.seeds.len(),
                manifest.seeds.first().unwrap_or(&0),
                manifest.seeds.last().unwrap_or(&0));
            println!("  Completed:  {}/{}", manifest.completed, manifest.total_runs);

            if let Ok(ir_json) = std::fs::read_to_string(&exp.config.model) {
                let mhash   = model_hash(&ir_json);
                let base_params: HashMap<String, f64> = exp.config.params.as_ref()
                    .and_then(|p| load_params_toml(p).ok())
                    .unwrap_or_default();
                let shash   = sim_hash(&mhash, &canonical_params(&base_params), exp.config.backend.as_str(), exp.config.dt);
                let raw_scenarios: Vec<ScenarioEntry> = if exp.scenario.is_empty() {
                    vec![ScenarioEntry { name: "baseline".to_string(), params: HashMap::new(), enable: vec![], disable: vec![] }]
                } else {
                    exp.scenario
                };
                // Resolve presets so the cache-hit count uses the same
                // scen_hash the run path was written under (CLI review #3).
                let scenarios: Vec<ScenarioEntry> = match ir::from_str(&ir_json) {
                    Ok(model) => match resolve_batch_scenarios(&raw_scenarios, &model) {
                        Ok(resolved) => resolved.iter().map(|r| ScenarioEntry {
                            name: r.name.clone(),
                            params: r.params.clone(),
                            enable: r.enable.clone(),
                            disable: r.disable.clone(),
                        }).collect(),
                        Err(_) => raw_scenarios,
                    },
                    Err(_) => raw_scenarios,
                };
                let seeds   = exp.config.seeds.resolve().unwrap_or_default();
                let sweep_points = expand_sweep(&exp.sweep);
                let runs_dir = format!("{}/sims", output_dir);
                let cache_stem = crate::hashing::path_stem_slug(&exp.config.model);
                let plans   = plan_runs(&scenarios, &sweep_points, &seeds, &shash,
                    cache_stem.as_deref(), &runs_dir, false);
                let live_hits = plans.iter().filter(|p| p.decision == RunDecision::CacheHit).count();
                println!("  Live count: {}/{} traj.tsv files present", live_hits, plans.len());
            }
            return;
        }
    }

    println!("Experiment status for: {}", toml_path);
    println!("  No manifest.json found at {}", manifest_path);
    println!("  Run 'camdl batch run {}' to start.", toml_path);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Print the resolved sweep grid + cache summary for `batch run --dry-run`.
/// Does not simulate. Format mirrors the single-run `--dry-run` idiom in
/// main.rs: header block, per-item tables, totals.
#[allow(clippy::too_many_arguments)]
fn print_batch_dry_run(
    model_path: &str,
    backend: crate::args::types::Backend,
    dt: f64,
    output_dir: &str,
    parallel: usize,
    scenarios: &[ResolvedEntry],
    sweep_points: &[HashMap<String, f64>],
    seeds: &[u64],
    base_params: &HashMap<String, f64>,
    params_file: Option<&str>,
    plans: &[RunPlan],
) {
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
        let marker = if sc.enable.is_empty() && sc.disable.is_empty() && sc.params.is_empty() {
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
            parts.join(" ")
        };
        eprintln!("  {:24} {} {}", sc.name, route_tag, marker);
    }
    eprintln!();

    // Sweep grid with per-point provenance
    let total_runs = plans.len();
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
        } else if params_file.is_some() && base_params.contains_key(name) {
            format!("params file: {}", params_file.unwrap())
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
    let hits    = plans.iter().filter(|p| p.decision == RunDecision::CacheHit).count();
    let misses  = plans.iter().filter(|p| p.decision == RunDecision::CacheMiss).count();
    eprintln!("Cache status:");
    eprintln!("  {} cache hits  → skipped", hits);
    eprintln!("  {} cache misses → would simulate", misses);
    eprintln!();

    // Output destinations — the content-addressed relative path each cell
    // would land in (first few). Confirms the sim×scen×seed hashing the
    // CasSink uses without running anything.
    eprintln!("Output paths (sims/<run_path>):");
    for plan in plans.iter().take(6) {
        let tag = match plan.decision {
            RunDecision::CacheHit  => "hit ",
            RunDecision::CacheMiss => "miss",
        };
        eprintln!("  [{}] {}", tag, plan.run_path);
    }
    if plans.len() > 6 {
        eprintln!("  ... ({} more)", plans.len() - 6);
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

    fn sc(name: &str) -> ScenarioEntry {
        ScenarioEntry { name: name.to_string(), params: HashMap::new(), enable: vec![], disable: vec![] }
    }

    fn sc_enable(name: &str, enables: &[&str]) -> ScenarioEntry {
        ScenarioEntry {
            name: name.to_string(),
            params: HashMap::new(),
            enable: enables.iter().map(|s| s.to_string()).collect(),
            disable: vec![],
        }
    }

    fn sc_params(name: &str, kv: &[(&str, f64)]) -> ScenarioEntry {
        ScenarioEntry {
            name: name.to_string(),
            params: kv.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            enable: vec![],
            disable: vec![],
        }
    }

    fn seed_traj(run_dir: &str) {
        std::fs::create_dir_all(run_dir).unwrap();
        std::fs::write(format!("{}/traj.tsv", run_dir), "t\n").unwrap();
    }

    fn no_sweep() -> Vec<HashMap<String, f64>> { vec![HashMap::new()] }

    fn sweep1(kv: &[(&str, f64)]) -> Vec<HashMap<String, f64>> {
        vec![kv.iter().map(|(k, v)| (k.to_string(), *v)).collect()]
    }

    // ── basic classification ─────────────────────────────────────────────────

    #[test]
    fn all_miss_on_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let plans = plan_runs(&[sc("baseline"), sc("with_sia")], &no_sweep(), &[1, 2, 3],
            "aaaa1111bbbb2222", None, dir.path().to_str().unwrap(), false);
        assert_eq!(plans.len(), 6);
        assert!(plans.iter().all(|p| p.decision == RunDecision::CacheMiss));
    }

    #[test]
    fn hit_when_traj_exists() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        // First pass to learn the path
        let plans = plan_runs(&[sc("baseline")], &no_sweep(), &[1, 2], "aaaa1111bbbb2222", None, runs_dir, false);
        seed_traj(&plans[0].run_dir); // seed 1 only
        // Re-classify
        let plans = plan_runs(&[sc("baseline")], &no_sweep(), &[1, 2], "aaaa1111bbbb2222", None, runs_dir, false);
        assert_eq!(plans[0].decision, RunDecision::CacheHit,  "seed 1 should be a hit");
        assert_eq!(plans[1].decision, RunDecision::CacheMiss, "seed 2 should be a miss");
    }

    #[test]
    fn force_ignores_existing_traj() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        let plans = plan_runs(&[sc("baseline")], &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        seed_traj(&plans[0].run_dir);
        let plans = plan_runs(&[sc("baseline")], &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, true);
        assert_eq!(plans[0].decision, RunDecision::CacheMiss);
    }

    // ── sim_hash invalidation ────────────────────────────────────────────────

    #[test]
    fn sim_hash_change_invalidates_all() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        // Populate under old sim_hash
        let old = plan_runs(&[sc("baseline")], &no_sweep(), &[1, 2], "aaaa1111bbbb2222", None, runs_dir, false);
        for p in &old { seed_traj(&p.run_dir); }
        // New sim_hash → different tier, all miss
        let new = plan_runs(&[sc("baseline")], &no_sweep(), &[1, 2], "cccc3333dddd4444", None, runs_dir, false);
        assert!(new.iter().all(|p| p.decision == RunDecision::CacheMiss));
        // Old paths unchanged
        for p in &old {
            assert!(std::path::Path::new(&format!("{}/traj.tsv", p.run_dir)).exists());
        }
    }

    // ── scen_hash invalidation ───────────────────────────────────────────────

    #[test]
    fn scen_change_invalidates_only_that_scenario() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        let scenarios = vec![sc("baseline"), sc_enable("with_sia", &["sia_r1"])];
        // Populate all runs
        let plans = plan_runs(&scenarios, &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        for p in &plans { seed_traj(&p.run_dir); }
        // Change only with_sia's enable list
        let new_scenarios = vec![sc("baseline"), sc_enable("with_sia", &["sia_r1", "sia_r2"])];
        let new = plan_runs(&new_scenarios, &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        let baseline = new.iter().find(|p| p.scenario == "baseline").unwrap();
        let with_sia = new.iter().find(|p| p.scenario == "with_sia").unwrap();
        assert_eq!(baseline.decision, RunDecision::CacheHit,  "baseline must be reused");
        assert_eq!(with_sia.decision, RunDecision::CacheMiss, "with_sia must be invalidated");
    }

    #[test]
    fn scen_param_change_invalidates_only_that_scenario() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        let scenarios = vec![sc("baseline"), sc_params("variant", &[("vacc_frac", 0.7)])];
        let plans = plan_runs(&scenarios, &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        for p in &plans { seed_traj(&p.run_dir); }
        let new_scenarios = vec![sc("baseline"), sc_params("variant", &[("vacc_frac", 0.9)])];
        let new = plan_runs(&new_scenarios, &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        assert_eq!(new.iter().find(|p| p.scenario == "baseline").unwrap().decision, RunDecision::CacheHit);
        assert_eq!(new.iter().find(|p| p.scenario == "variant").unwrap().decision, RunDecision::CacheMiss);
    }

    // ── seed extension ───────────────────────────────────────────────────────

    #[test]
    fn adding_seeds_reuses_existing() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        // Populate seeds 1-3
        let plans = plan_runs(&[sc("baseline")], &no_sweep(), &[1, 2, 3], "aaaa1111bbbb2222", None, runs_dir, false);
        for p in &plans { seed_traj(&p.run_dir); }
        // Extend to seeds 1-5
        let plans = plan_runs(&[sc("baseline")], &no_sweep(), &[1, 2, 3, 4, 5], "aaaa1111bbbb2222", None, runs_dir, false);
        let (hits, misses): (Vec<_>, Vec<_>) = plans.iter()
            .partition(|p| p.decision == RunDecision::CacheHit);
        assert_eq!(hits.len(), 3,   "seeds 1-3 must be reused");
        assert_eq!(misses.len(), 2, "seeds 4-5 must be new");
        let miss_seeds: Vec<u64> = misses.iter().map(|p| p.seed).collect();
        assert!(miss_seeds.contains(&4) && miss_seeds.contains(&5));
    }

    // ── run_path structure ───────────────────────────────────────────────────

    #[test]
    fn run_path_format() {
        let dir = tempfile::tempdir().unwrap();
        let plans = plan_runs(&[sc("with sia!")], &no_sweep(), &[42], "aaaa1111bbbb2222", None, dir.path().to_str().unwrap(), false);
        // sim_hash_8 / slug-scen_hash_8 / seed_N
        let parts: Vec<&str> = plans[0].run_path.splitn(3, '/').collect();
        assert_eq!(parts[0], "aaaa1111",            "sim_hash_8");
        assert!(parts[1].starts_with("with_sia_"),  "slug must sanitize spaces and '!'");
        assert_eq!(parts[2], "seed_42",             "seed component");
    }

    #[test]
    fn rename_scenario_same_semantics_same_scen_hash() {
        // Two scenarios with identical overrides but different names share the same
        // scen_hash suffix — demonstrating that renaming doesn't create a new cache entry
        // for semantically identical runs.
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        let p1 = plan_runs(&[sc_enable("old_name", &["sia"])], &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        let p2 = plan_runs(&[sc_enable("new_name", &["sia"])], &no_sweep(), &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        // Slugs differ but scen_hash_8 (embedded in dir name) is identical
        let hash1: &str = p1[0].run_path.split('/').nth(1).unwrap().split_once('-').unwrap().1;
        let hash2: &str = p2[0].run_path.split('/').nth(1).unwrap().split_once('-').unwrap().1;
        assert_eq!(hash1, hash2, "same enables/params → same scen_hash_8");
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

    #[test]
    fn sweep_changes_scen_hash() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().to_str().unwrap();
        let pt1 = sweep1(&[("vacc_eff", 0.3)]);
        let pt2 = sweep1(&[("vacc_eff", 0.7)]);
        let p1 = plan_runs(&[sc("baseline")], &pt1, &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        let p2 = plan_runs(&[sc("baseline")], &pt2, &[1], "aaaa1111bbbb2222", None, runs_dir, false);
        // Different sweep values → different scen_hash → different directories
        assert_ne!(p1[0].run_path, p2[0].run_path, "distinct sweep points must produce distinct paths");
    }

    #[test]
    fn sweep_count_correct() {
        let dir = tempfile::tempdir().unwrap();
        let mut sweep = HashMap::new();
        sweep.insert("x".to_string(), SweepSpec::Linspace {
            linspace: LinspaceSpec { min: 0.0, max: 1.0, n: 5 }
        });
        let points = expand_sweep(&sweep);
        // 5 sweep × 2 scenarios × 3 seeds = 30
        let plans = plan_runs(&[sc("baseline"), sc("with_sia")], &points, &[1, 2, 3],
            "aaaa1111bbbb2222", None, dir.path().to_str().unwrap(), false);
        assert_eq!(plans.len(), 30);
    }
}
