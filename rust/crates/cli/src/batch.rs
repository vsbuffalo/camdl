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
use crate::version;

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
    /// The scenario's own simulation horizon (`scenarios { x { simulate { to
    /// = … } } }`), when it declares one — otherwise the cell inherits
    /// `model.simulation.t_end` (gh#561).
    ///
    /// Identity-relevant: it changes the stored trajectory, so it is folded
    /// into the cell's `config` level by [`CasSink::cell_resolve`]. A model
    /// preset carries whatever it declared; an inline ad-hoc scenario is
    /// always `None`, because a horizon is a model-file declaration and there
    /// is no CLI spelling for one.
    pub t_end: Option<f64>,
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
                    // Composed, via the single horizon authority — see the twin
                    // site in `main.rs` (gh#561).
                    let t_end = crate::params_resolver::composed_preset_t_end(model, &name)
                        .map_err(|e| e.to_string())?;
                    Ok(ResolvedEntry {
                        name,
                        route: Some(preset.name.clone()),
                        enable: preset.enable.clone(),
                        disable: preset.disable.clone(),
                        params: preset.params.clone(),
                        t_end,
                    })
                }
                ResolvedScenario::Adhoc { name, enable, disable, params } => {
                    Ok(ResolvedEntry {
                        name,
                        route: None,
                        enable,
                        disable,
                        params: params.into_iter().collect(),
                        // An ad-hoc patch has no horizon: `simulate { to }` is
                        // a model-file declaration with no CLI spelling.
                        t_end: None,
                    })
                }
            }
        })
        .collect()
}

// ─── Run metadata ─────────────────────────────────────────────────────────────

/// Descriptor for one completed cell, accumulated on `CasSink.completed_runs`
/// and used by the post-run leaf summary (`report_cas_leaves`). Each cell is
/// independently a `runid::RunRecord` leaf on disk; this is the in-memory
/// tally, not a file. Only the store-relative path is reported.
#[derive(Debug, Clone)]
pub(crate) struct RunEntry {
    pub(crate) run_path: String,
    /// The leaf's `run_id` — the ensemble (simulate, multi-cell) records these
    /// as its `deps` and folds them into its `grid` identity. `None` only if
    /// identity resolution failed (the cell is then also on `errors`).
    pub(crate) run_id: Option<runid::ContentHash>,
    /// SHA-256 of the cell's `traj.tsv` (the artifact the ensemble's combined
    /// TSV is derived from). Pins which upstream file the dep consumed.
    pub(crate) traj_digest: Option<runid::ContentHash>,
    /// Scenario label, resolved process seed, and 0-based draw index — the
    /// per-cell coordinates the ensemble's `grid` digest sorts over (alongside
    /// the cell `run_id`).
    pub(crate) scenario: String,
    pub(crate) process_seed: u64,
    pub(crate) draw_idx: usize,
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

    // Batch runs are forward simulations — no state-Jacobian consumer, so compile
    // lean (`needs_state_grad = false`, gh#439 A2).
    let (ir_path_resolved, _tmpfile) = resolve_ir_path(&model_path, false).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    // gh#156: `[output] every` rewrites the compiled IR's output schedule once,
    // so both the engine and the CAS identity (loaded from this path) see the
    // overridden cadence.
    let (ir_path_resolved, _every_tmp) =
        crate::util::rematerialize_with_output_every(&ir_path_resolved, exp.output.every)
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
    let ir_json = std::fs::read_to_string(&ir_path_resolved).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", ir_path_resolved, e);
        std::process::exit(1);
    });

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

    let model_stem = crate::hashing::path_stem_slug(&ir_path_resolved);

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
            &base_params,
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

    if a.dry_run {
        print_batch_dry_run(
            &model_path, &batch_model, model_stem.as_deref(),
            backend, dt, &output_dir, parallel,
            &resolved_scenarios, &sweep_points, &seeds, &base_params,
            exp.config.params.as_deref(), &runs_dir, a.allow_degenerate_rates,
            &output_cols, a.force,
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

    // ── Build the SimulateJob and route through the unified engine ──────────
    //
    // `batch run` is a thin TOML front-end over `engine::run_job` — the
    // SAME engine `camdl simulate` uses (run-spec §3.1). The per-cell seed
    // arithmetic and SimRun construction are shared; the CAS-tree output
    // shape lives in `CasSink`, which resolves each cell's identity via
    // `resolve::resolve_trajectory` so the on-disk layout / content-hashes
    // match the `simulate` path exactly.
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
        // Batch replicates ride on the explicit `seeds` list, so the engine
        // uses that length (not `replicates`) for the rep count.
        ParamSource::Sweep { points, replicates: 1 }
    } else {
        ParamSource::Point { replicates: 1 }
    };

    let job = SimulateJob {
        model: ir_path_resolved.clone(),
        params_files: params_file_opt.as_ref().map(|p| vec![p.clone()]).unwrap_or_default(),
        backend,
        dt,
        integrator: None, // batch uses the model's declared integrator (no CLI override)
        source,
        scenarios: job_scenarios,
        // gh#626: batch TOML has no `to` key (deliberate; CLI-only override).
        t_end_override: None,
        // gh#641: nor an `init_state` key, for the same reason.
        init_state: None,
        // gh#616: `batch run` has no data-binding surface, so an anchored
        // model is refused at `CompiledModel::new` rather than resolved.
        obs_anchors: None,
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
        // Batch TOML has no `--table` mechanism (its `[config]` carries no
        // table field); embedded tables ride the whole-IR model digest.
        table_files: HashMap::new(),
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

// ─── CasSink: batch's content-addressed output strategy ───────────────────────

/// `RunSink` for `camdl batch run`: writes each cell into the content-addressed
/// store under `sims/` (the `runid` factored sim layout), resolving each cell's
/// identity via [`crate::resolve::resolve_trajectory`] so the path, `run.json`
/// `run_id`, and recorded sweep point match the `simulate` path exactly. Cache
/// hits are skipped via `should_run` (the engine never simulates them).
pub(crate) struct CasSink {
    /// Resolved `[[scenario]]` entries, looked up by name for the
    /// hash-relevant enable/disable/params delta.
    pub(crate) resolved_scenarios: Vec<ResolvedEntry>,
    pub(crate) model_path: String,
    pub(crate) model_stem: Option<String>,
    /// The **base** model — its whole-IR digest is the (constant-across-cells)
    /// model level. Never `cell.model` (which has scenario + sweep applied).
    pub(crate) base_model: ir::Model,
    /// Resolved base parameter values; per cell, the sweep point is layered on
    /// top into the `params` level (a resolved value, not the scenario delta).
    pub(crate) base_params: HashMap<String, f64>,
    /// External `--table NAME=PATH` overrides (name → file path). The table
    /// reference in the IR is identity-inert on its own; the file's *content*
    /// is what enters the run_id, so each resolve reads the bytes and folds
    /// their digest into the `params` level (`ResolvedParams.tables`). Empty
    /// for batch (its TOML has no `--table` mechanism); populated by
    /// `simulate --table`. An edit to a `--table` file re-keys the run, so a
    /// changed `matrix.tsv` cannot serve a stale cached trajectory.
    pub(crate) table_files: HashMap<String, String>,
    pub(crate) backend: crate::args::types::ForwardBackend,
    pub(crate) dt: f64,
    pub(crate) allow_degenerate_rates: bool,
    /// Output view (gh#156): the resolved `--no-flows` / `--columns` filter,
    /// applied to both this cell's leaf `traj.tsv` (writer) and its `config`
    /// identity (`cell_resolve`). Default = full output. `--output-every` is
    /// not here — it is lowered into the model schedule upstream.
    pub(crate) output_cols: crate::util::OutputColumns,
    /// Absolute `<output>/sims` subtree.
    pub(crate) runs_dir: String,
    pub(crate) obs_enabled: bool,
    /// gh#656: `simulate --emit-every`, the per-stream emission-cadence
    /// override. It sets the emitted times AND (via [`obs_subtree_hash`]) the
    /// obs subtree's address, so two cadences are two artifacts under one
    /// trajectory leaf. `None` for `batch run`, which has no such key, and for
    /// every `simulate` without the flag — those key exactly as they always did.
    pub(crate) emit_every: Option<crate::emit_every::EmitEvery>,
    pub(crate) force: bool,
    pub(crate) total: usize,
    pub(crate) counter: usize,
    pub(crate) completed_runs: Vec<RunEntry>,
    pub(crate) errors: Vec<String>,
    /// User-supplied display label, recorded on each leaf's
    /// `RunRecord.provenance.label`. `simulate --label` sets this; batch
    /// leaves it `None`.
    pub(crate) label: Option<String>,
    /// Upstream-fit lineage recorded on each leaf's `RunRecord.deps`. Set
    /// when `simulate --params` consumes a fit-MLE params file carrying a
    /// `[provenance]` block (so the sim records which fit it came from);
    /// empty for batch and for standalone (non-fit) params. Provenance only —
    /// a sim's identity is its factored levels (resolve_trajectory), never
    /// `deps`, so this does not change the sim's run_id or store path.
    pub(crate) fit_dep: Vec<runid::inputs::ArtifactRef>,
    /// The overall cells progress bar (advanced once per committed or
    /// cache-hit cell, finished on the last). `Some` only for a multi-cell run
    /// (`total > 1`) — a LONE `simulate` gets the engine's per-timestep bar
    /// instead, so its overall bar would be a redundant `1/1`. `None` is the
    /// inert case (single cell, or `--progress none`/tests). Honours the
    /// `--progress` mode via [`crate::progress::Task`].
    pub(crate) progress: Option<crate::progress::Task>,
}

/// The overall cells bar for a multi-cell run, or `None` for a single cell
/// (the engine's per-timestep bar covers that) or when no bar applies. The
/// `Task` itself honours `--progress none`/`plain` internally.
pub(crate) fn cells_progress(total: usize, label: impl Into<String>) -> Option<crate::progress::Task> {
    (total > 1).then(|| crate::progress::Reporter::new().task(total as u64, label, "cells"))
}

impl CasSink {
    /// Advance the overall cells bar by one cell (committed or cache-hit) and
    /// finish it once the last cell lands. The bar replaces the old
    /// `[i/total] scenario=… seed=…` per-cell stderr lines; the per-cell
    /// detail is intentionally dropped in favour of a clean `pos/len` bar
    /// (drill into individuals with `camdl list --kind sim`). No-op when
    /// `progress` is `None` (single cell / `--progress none` / tests).
    fn advance_progress(&mut self) {
        if let Some(t) = self.progress.as_mut() {
            t.inc(1);
        }
        if self.counter >= self.total {
            if let Some(t) = self.progress.take() {
                t.finish();
            }
        }
    }

    /// The store root (`<output>`, the parent of `<output>/sims`).
    fn root(&self) -> std::path::PathBuf {
        std::path::Path::new(&self.runs_dir)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."))
    }

    /// Content digests of the external `--table` files, sorted by table name,
    /// for folding into the `params` level (`ResolvedParams.tables`). Each
    /// file's *bytes* are hashed (never its path), so a regenerated or edited
    /// table re-keys the run and a stale cached trajectory is never served.
    ///
    /// Mirrors `profile_cas::data_digests`: sort by name, `digest_bytes` per
    /// file (= `ContentHash::from_hex(sha256_hex(bytes))`). Empty `table_files`
    /// → empty vec, identical to the no-`--table` path (so a plain `simulate`
    /// is unaffected).
    fn table_digests(&self) -> Result<Vec<runid::inputs::DataDigest>, String> {
        let mut names: Vec<&String> = self.table_files.keys().collect();
        names.sort();
        names
            .iter()
            .map(|name| {
                let path = &self.table_files[*name];
                let bytes = std::fs::read(path)
                    .map_err(|e| format!("cannot read --table file '{}' ({}): {}", name, path, e))?;
                Ok(runid::inputs::DataDigest(runid::ContentHash::digest_bytes(&bytes)))
            })
            .collect()
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
        let param_label = params_path_label(&spec.point_overrides);

        let ctx = crate::resolve::TrajectoryCtx {
            model: &self.base_model,
            model_stem: self.model_stem.as_deref().unwrap_or("model"),
            ir_version: ir::IR_VERSION.trim(),
            engine_version: version::VERSION_SHORT,
            backend: self.backend,
            dt: self.dt,
            // gh#641: the cell's EFFECTIVE start, not the base model's. A run
            // seeded from a filtered state begins at that state's origin and
            // therefore covers a different window — the same argument the
            // horizon override makes below, from the other end.
            t_start: spec
                .sim_run
                .init_state
                .as_ref()
                .map(|i| i.origin_t)
                .unwrap_or(self.base_model.simulation.t_start),
            // gh#561: the cell's EFFECTIVE horizon, not the base model's. A
            // scenario declaring its own `simulate { to }` produces a different
            // trajectory, so it must re-key (`feedback: count-in-the-key`) —
            // otherwise a horizon menu whose scenarios carry no `set`/`enable`
            // delta resolves every cell to one `run_id` and the store serves
            // one trajectory for all of them.
            //
            // This folds the RESOLVED VALUE into the existing `SimConfig::t_end`
            // rather than adding a field to the scenario level, so it costs
            // nothing for models that don't use the feature: a preset with no
            // `to`, or one equal to the model horizon, hashes exactly as before.
            t_end: resolved.t_end.unwrap_or(self.base_model.simulation.t_end),
            output: &self.base_model.output.times,
            allow_degenerate_rates: self.allow_degenerate_rates,
            no_flows: self.output_cols.no_flows,
            columns: &self.output_cols.allow,
            base_params: &params,
            table_digests: self.table_digests()?,
            enable: &resolved.enable,
            disable: &resolved.disable,
            scen_params: &resolved.params,
            param_label: &param_label,
            scenario_label,
            base_seed: spec.process_seed,
            process_seed: spec.process_seed,
            // gh#641 / gh#697: the state this cell restored. The origin
            // ENSEMBLE's content digest and the restored ROW both key the run —
            // a re-filtered / re-fitted ensemble under an unchanged model must
            // not serve the cached forecast, and two cells sharing a
            // `process_seed` (`--seeds 7,7`) must not collide while starting
            // from different rows. `None` for every `batch run` cell (no
            // `init_state` key — CLI-only, like `to`).
            init_state: spec.sim_run.init_state.as_ref().map(|i| {
                runid::inputs::InitStateDigest {
                    ensemble: runid::inputs::DataDigest(i.ensemble_digest),
                    row: i.row,
                }
            }),
        };
        let rt = crate::resolve::resolve_trajectory(&ctx).map_err(|e| format!("resolve error: {e}"))?;
        let root = self.root();
        let dir = runid::store_path(&root, runid::ArtifactKind::Sim, &rt.levels);
        let rel = dir.strip_prefix(&root).unwrap_or(&dir).to_string_lossy().into_owned();
        Ok((rt, dir, rel))
    }

    /// A throwaway `CasSink` for dry-run / status prediction. It is used ONLY
    /// for `cell_resolve` (the canonical identity/path resolver) and `root()`,
    /// so the identity-relevant fields are populated from the same resolved
    /// inputs the real run path uses; the run-only fields (`model_path`,
    /// `total`/`counter`, `completed_runs`, `progress`, …) are inert.
    fn probe(
        resolved_scenarios: &[ResolvedEntry],
        base_model: &ir::Model,
        model_stem: Option<&str>,
        base_params: &HashMap<String, f64>,
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
            base_params: base_params.clone(),
            table_files: HashMap::new(),
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
    /// the cell's `point_overrides`. A cell whose identity fails to resolve is
    /// counted as a miss (it surfaces as an error on the real run path).
    fn predict_cells(
        &self,
        points: &[indexmap::IndexMap<String, f64>],
        seeds: &[u64],
    ) -> Vec<CellPrediction> {
        let store = runid::FsCasStore::new(self.root());
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
                    let spec = crate::engine::CellSpec {
                        run_idx: 0,
                        point_idx,
                        scenario: scenario.clone(),
                        point_overrides: point_overrides.clone(),
                        process_seed: seed,
                        obs_seed: seed ^ crate::util::SEED_MIX_OBS,
                        sim_run: crate::util::SimRun::default(),
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

/// Human-readable provenance label for the `params` path level.
///
/// Sparse overrides — a scenario sweep point like `beta=0.3_gamma=0.1` — render
/// as a readable, key-sorted `k=v` join so `ls results/sims/` stays skimmable. A
/// *full* parameter vector (a `--draws` row over a stratified model) would blow a
/// single path component past NAME_MAX (gh#169), so it collapses to a short
/// `draws` tag. Identity is the level's content hash, never this label (see
/// `resolve.rs`: label and hash are separate inputs to `level()`), so the
/// collapse is lossless — the full drawn values live in `run.json`.
fn params_path_label(overrides: &indexmap::IndexMap<String, f64>) -> String {
    if overrides.is_empty() {
        return "base".to_string();
    }
    let mut sorted: Vec<(&String, &f64)> = overrides.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let joined = sorted
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join("_");
    // Readable when sparse; a full draw vector is too long for a path
    // component, so tag it `draws` (the level hash disambiguates the values).
    const LABEL_CAP: usize = 96;
    if joined.len() <= LABEL_CAP {
        joined
    } else {
        "draws".to_string()
    }
}

#[cfg(test)]
mod gh169_params_label {
    use super::params_path_label;
    use indexmap::IndexMap;

    fn im(pairs: &[(&str, f64)]) -> IndexMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn empty_overrides_is_base() {
        assert_eq!(params_path_label(&IndexMap::new()), "base");
    }

    #[test]
    fn sparse_scenario_overrides_stay_readable() {
        // a scenario sweep point: a couple of params -> readable, key-sorted join
        assert_eq!(
            params_path_label(&im(&[("gamma", 0.1), ("beta", 0.3)])),
            "beta=0.3_gamma=0.1"
        );
    }

    #[test]
    fn full_draw_vector_collapses_to_short_tag() {
        // a --draws row over a stratified model: the whole vector is overridden,
        // which previously rendered as one 600+ byte path component -> ENAMETOOLONG (gh#169).
        let overrides: IndexMap<String, f64> = (0..40)
            .map(|i| (format!("beta_age{}_patch{}", i % 2, i), 0.3125))
            .collect();
        let label = params_path_label(&overrides);
        assert_eq!(label, "draws", "a full draw vector must collapse to a short tag");
        // the rendered path component `{label}-{hash8}` must fit in NAME_MAX (255)
        assert!(label.len() + 1 + 8 <= 255);
    }
}

impl crate::engine::RunSink for CasSink {
    fn should_run(&mut self, spec: &crate::engine::CellSpec) -> bool {
        match self.cell_resolve(spec) {
            Ok((rt, _dir, _)) => {
                let store = runid::FsCasStore::new(self.root());
                let artifact = crate::resolve::ResolvedArtifact {
                    kind: runid::ArtifactKind::Sim,
                    levels: rt.levels.clone(),
                    run_id: rt.run_id,
                    display_inputs: serde_json::Value::Null,
                };
                let policy = if self.force {
                    crate::resolve::WritePolicy::Force
                } else {
                    crate::resolve::WritePolicy::Reuse
                };
                // Through the one policy seam. Under Force the incumbent is
                // displaced (quarantined) HERE, so the recompute lands on a
                // clean leaf — previously force only skipped this check and
                // the recomputed bytes were discarded at commit as an
                // already-completed no-op, which is why `--force` could not
                // change a stored artifact.
                let root = self.root();
                match crate::resolve::check_reuse(&store, &root, &artifact, policy) {
                    Ok(crate::resolve::ReuseVerdict::CacheHit { dir }) => {
                        // A trajectory hit is not enough when observations were
                        // asked for: the obs subtree is a CHILD, written only
                        // by `merge_cell`, so skipping a cached cell also skips
                        // its obs. `batch --obs` over a store populated by
                        // obs-free runs therefore produced no observations at
                        // all, silently — the user asked for a thing and got
                        // nothing, with a "cached" line implying otherwise.
                        //
                        // The check is PRESENCE of the obs subtree, not its
                        // exact address: `should_run` has only `base_model`,
                        // while `merge_cell` writes from `cell.model` (scenario
                        // + sweep applied). Recomputing the subtree hash from
                        // the wrong model would reintroduce the
                        // divergent-resolution class this arc exists to remove.
                        // Residual, tracked separately: a leaf holding SOME
                        // obs subtree but not this cadence's still skips.
                        // `true` here means RUN. Run when observations were
                        // requested and this leaf has none; otherwise the
                        // trajectory hit stands and the cell is skipped.
                        self.obs_enabled
                            && !self.base_model.observations.is_empty()
                            && !dir.join("obs").is_dir()
                    }
                    Ok(crate::resolve::ReuseVerdict::MustRun) => true,
                    // Cache check failed → run; the error surfaces in merge_cell.
                    Err(_) => true,
                }
            }
            // No cache key without a resolve → run; the error surfaces in merge_cell.
            Err(_) => true,
        }
    }

    fn on_skip(&mut self, spec: &crate::engine::CellSpec) {
        // A cache hit still contributes to a multi-cell ensemble's deps, so
        // recover the leaf's run_id (from the resolve) and its traj.tsv digest
        // (from the existing leaf's run.json, recorded at first commit).
        let (rel, run_id, traj_digest) = match self.cell_resolve(spec) {
            Ok((rt, dir, rel)) => {
                let digest = read_traj_digest(&dir);
                // Keep an explicit `--label` current even on a pure cache-hit
                // skip (same reasoning as the commit path; no-op when None).
                if let Some(ref label) = self.label {
                    let _ = ensure_provenance_label(&dir, label);
                }
                (rel, Some(rt.run_id), digest)
            }
            Err(_) => (String::new(), None, None),
        };
        let name = spec.scenario.name().to_string();
        self.counter += 1;
        self.advance_progress();
        self.completed_runs.push(RunEntry {
            run_path: rel,
            run_id,
            traj_digest,
            scenario: name,
            process_seed: spec.process_seed,
            draw_idx: spec.point_idx,
        });
    }

    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let spec = &cell.spec;
        let name = spec.scenario.name().to_string();

        let (rt, _dir, rel) = match self.cell_resolve(spec) {
            Ok(x) => x,
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: {}", name, spec.process_seed, e));
                return Ok(());
            }
        };

        let cols = self.output_cols.cols(&cell.model);
        let traj_bytes = crate::util::traj_tsv_bytes(&cell.traj, &cols);
        let traj_digest = runid::ContentHash::digest_bytes(&traj_bytes);

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
            // The SAME hash `write_obs_into_cas` names the subtree with — a
            // declared child that did not match the directory written beside it
            // would be provenance pointing at nothing.
            let obs_hash = match obs_subtree_hash(
                &cell.model.observations, self.emit_every.as_ref(),
            ) {
                Ok(h) => h,
                Err(e) => {
                    self.counter += 1;
                    self.errors.push(format!("scenario={} seed={}: {}",
                        name, spec.process_seed, e));
                    return Ok(());
                }
            };
            let obs_id = runid::ContentHash::digest_bytes(
                format!("{}:{}:{}", rt.run_id.to_hex(), obs_seed, obs_hash).as_bytes(),
            );
            children.insert("obs".to_string(), vec![obs_id]);
        }

        // Atomic write through the one resolved-writer seam (gh#241 PR D).
        // Identity (`Sim` / `levels` / `run_id`) is copied verbatim from the
        // resolved trajectory; the obs child + the fit dep ride in the record.
        let resolved_artifact = crate::resolve::ResolvedArtifact {
            kind: runid::ArtifactKind::Sim,
            levels: rt.levels.clone(),
            run_id: rt.run_id,
            display_inputs: serde_json::Value::Null,
        };
        let mut meta = crate::resolve::RecordMeta::new(
            ir::IR_VERSION.trim(), self.model_path.clone(), self.label.clone())
            .with_deps(self.fit_dep.clone())
            .with_children(children);

        let mut artifacts = runid::Artifacts::new();
        artifacts.insert("traj.tsv", traj_bytes);
        // `simulate --event-log`: the recorded event log is a first-class
        // artifact in THIS leaf, alongside `traj.tsv`, declared in the
        // exact-set (so the on-disk `event_log.tsv` path is a valid input to
        // `lineage realize`). The recorder is passive (Tier 2a) so the run_id
        // is unchanged — a freshly-committed leaf a plain `simulate` would
        // write simply gains one more artifact. Recording into a leaf that
        // already exists WITHOUT the log is handled below by `augment`, which
        // adds it in place; the store gained that door precisely because this
        // case used to lose the log. (This comment previously said `--force`
        // or a fresh identity was needed — neither was true: forcing
        // re-commits into the same discard.)
        // Kept alongside the staged set so they can be re-added to an
        // ALREADY-COMPLETED leaf below: a cache hit discards the staged
        // artifacts wholesale, which is how `--event-log` against an existing
        // leaf silently lost its log.
        let mut optional_artifacts: Vec<(&'static str, Vec<u8>)> = Vec::new();
        if let Some(ref el) = cell.event_log {
            match sim::lineage::event_log_io::to_tsv_bytes(el) {
                Ok(bytes) => {
                    optional_artifacts.push(("event_log.tsv", bytes.clone()));
                    artifacts.insert("event_log.tsv", bytes);
                }
                Err(e) => {
                    self.counter += 1;
                    self.errors.push(format!("scenario={} seed={}: event log serialize: {:?}",
                        name, spec.process_seed, e));
                    return Ok(());
                }
            }
        }
        // gh#204: a run with an active reactive policy declares its firing log
        // as a first-class artifact in THIS leaf alongside `traj.tsv` — present
        // even with zero firings (`Some` carries that), absent when the model
        // has no active reactive policy. The agenda's realized-obs draws run on
        // a dedicated RNG salt off the run seed, so the trajectory bytes and the
        // run_id are unchanged: a leaf a plain `simulate` would write simply
        // gains one more declared artifact, never an optional-on-cache-hit one.
        if let Some(ref firings) = cell.traj.reactive_log {
            let bytes = sim::reactive::format_reactive_log(firings).into_bytes();
            optional_artifacts.push(("reactive_log.tsv", bytes.clone()));
            artifacts.insert("reactive_log.tsv", bytes);
        }
        // Declare the tabular outputs' column schema in run.json — classify the
        // in-memory artifact headers (they're committed atomically, not yet on
        // disk). Recorded, not hashed.
        meta.output_schema = crate::output_schema::sim_output_schema(&artifacts.files);
        let root = self.root();
        let store = runid::FsCasStore::new(&root);
        // The commit half of the force policy. `should_run` covers callers
        // that skip cached cells, but `simulate` deliberately runs every cell
        // (its `-o` mirror needs them all) and reaches commit with a cache hit
        // in place — where the staged bytes were silently discarded, so
        // `--force` could never change a stored artifact. Displacing here
        // (quarantine, not delete) gives every commit path the same one
        // meaning of force, and is a no-op when nothing is stored.
        if self.force {
            if let Err(e) = crate::resolve::check_reuse(
                &store, &root, &resolved_artifact, crate::resolve::WritePolicy::Force,
            ) {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: force displace failed: {}",
                    name, spec.process_seed, e));
                return Ok(());
            }
        }
        let dest = match crate::resolve::begin_resolved_write(
            &store, &root, &resolved_artifact, &meta,
            crate::resolve::WriteMode::Atomic(artifacts),
        ) {
            Ok(crate::resolve::ResolvedWrite::Committed(d)) => d,
            Ok(crate::resolve::ResolvedWrite::Streaming(_)) => {
                unreachable!("Atomic write mode never returns a streaming claim")
            }
            Err(e) => {
                self.counter += 1;
                self.errors.push(format!("scenario={} seed={}: commit failed: {}",
                    name, spec.process_seed, e));
                return Ok(());
            }
        };

        // A cache hit at commit returns the INCUMBENT leaf and discards
        // everything we staged, so an optional artifact this run recorded but
        // the stored leaf lacks — `--event-log` against a leaf first written
        // without it — would simply vanish. (The old comment here claimed
        // `--force` was the workaround; it was not: that path re-commits and
        // reaches the same discard.) `augment` adds it under the leaf's lock;
        // it is a no-op when the bytes already match, which is the fresh-commit
        // case, and it refuses divergent bytes for the same reason the commit
        // path does.
        for (artifact, bytes) in &optional_artifacts {
            if let Err(e) = store.augment(
                &dest, &runid::LeafIdentity::new(rt.run_id), artifact, bytes)
            {
                self.errors.push(format!(
                    "scenario={} seed={}: recording {} into {}: {}",
                    name, spec.process_seed, artifact, dest.display(), e));
            }
        }

        // Obs ensemble: written into the committed leaf's `obs/` child. A
        // failure here is non-fatal — children are independent, so a missing
        // obs child never staleens the parent trajectory.
        if has_obs {
            // gh#641: a restarted cell's obs axis begins at the forecast
            // origin, exactly as the `-o`/`--obs` mirror's does. `None` for
            // every ordinary cell — and for every `batch run` cell, which has
            // no `init_state` key.
            let restart_origin = spec.sim_run.init_state.as_ref().map(|i| i.origin_t);
            if let Err(e) = write_obs_into_cas(
                &dest, &cell.model, &cell.traj, spec.process_seed, restart_origin,
                self.emit_every.as_ref(),
            ) {
                self.errors.push(format!("scenario={} seed={}: obs ensemble: {}",
                    name, spec.process_seed, e));
            }
        }

        // A `--label` must land on the leaf even when `commit_atomic` resolved
        // to an existing (cache-hit) leaf whose `run.json` carried a different
        // or absent label — the content-addressed dedup must not silently drop
        // the user's explicit label. `provenance` is metadata, not identity, so
        // this never changes the run_id (same mechanism as `camdl label`). A
        // no-op on a fresh commit, whose record already carries `self.label`.
        if let Some(ref label) = self.label {
            if let Err(e) = ensure_provenance_label(&dest, label) {
                self.errors.push(format!("scenario={} seed={}: label update: {}",
                    name, spec.process_seed, e));
            }
        }

        self.counter += 1;
        self.advance_progress();
        self.completed_runs.push(RunEntry {
            run_path: rel,
            run_id: Some(rt.run_id),
            traj_digest: Some(traj_digest),
            scenario: name,
            process_seed: spec.process_seed,
            draw_idx: spec.point_idx,
        });
        Ok(())
    }
}

/// Read a committed leaf's `traj.tsv` SHA-256 from its `run.json` artifact
/// manifest (recorded at commit). Used by the cache-hit path (`on_skip`) to
/// recover the dep digest without re-reading the trajectory bytes.
fn read_traj_digest(dir: &std::path::Path) -> Option<runid::ContentHash> {
    let bytes = std::fs::read(dir.join("run.json")).ok()?;
    let rec: runid::RunRecord = serde_json::from_slice(&bytes).ok()?;
    rec.artifacts.get("traj.tsv").map(|c| c.digest)
}

/// Ensure a leaf's `run.json` carries `provenance.label == Some(label)`,
/// rewriting it atomically (tmp + rename) only when it differs. `provenance`
/// is metadata, not identity — the run_id and exact-set are untouched — so this
/// is the same in-place relabel `camdl label` performs, applied at write time
/// so an explicit `--label` survives a CAS cache hit. Idempotent: a no-op when
/// the leaf already has the requested label (e.g. a fresh commit).
fn ensure_provenance_label(leaf_dir: &std::path::Path, label: &str) -> Result<(), String> {
    let path = leaf_dir.join("run.json");
    let txt = std::fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {}", path.display(), e))?;
    let mut rec: runid::RunRecord = serde_json::from_str(&txt)
        .map_err(|e| format!("parse {}: {}", path.display(), e))?;
    if rec.provenance.label.as_deref() == Some(label) {
        return Ok(()); // already current — no rewrite
    }
    rec.provenance.label = Some(label.to_string());
    let tmp = leaf_dir.join("run.json.tmp");
    let json = serde_json::to_string_pretty(&rec)
        .map_err(|e| format!("serialize run.json: {}", e))?;
    std::fs::write(&tmp, json)
        .and_then(|_| std::fs::rename(&tmp, &path))
        .map_err(|e| format!("write {}: {}", path.display(), e))
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
    base_params: &HashMap<String, f64>,
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
    use crate::sim_job::{ParamSource, ScenarioRef, Seeds, SimulateJob};

    // The CAS store root is the canonical output root: design sim leaves share
    // the `<output>/sims/` tree with normal `batch`/`simulate` cells so an
    // identical cell dedupes to the same leaf (same run_id, same path).
    let runs_dir = format!("{}/sims", output_dir);

    // Scenario routing for the engine job — identical to the normal flow: a
    // resolved preset → `ScenarioRef::Named`; an ad-hoc patch → `Inline`.
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

        // ── Run the block's points as a sweep over the unified engine ─────
        //
        // The design points ARE sweep points: each is a name→value override
        // map of the same shape `expand_sweep` produces. Build a `ParamSource::
        // Sweep` and the SAME `SimulateJob` + `CasSink` the normal flow uses.
        let points: Vec<indexmap::IndexMap<String, f64>> = design_result.points.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect();

        let total = points.len() * resolved_scenarios.len() * seeds.len();

        // Dry-run: resolve every cell's real CAS identity and count
        // `store.lookup` hits (the same path `CasSink::should_run` takes) —
        // never the legacy hash/marker check. No simulation, no files written.
        if dry_run {
            print_design_dry_run(
                design_name, &points, resolved_scenarios, batch_model,
                model_stem, base_params, backend, dt, allow_degenerate_rates,
                output_cols, &runs_dir, seeds, force,
            );
            continue;
        }

        std::fs::create_dir_all(&runs_dir).unwrap_or_else(|e| {
            eprintln!("error: cannot create runs dir {}: {}", runs_dir, e);
            std::process::exit(1);
        });

        let job = SimulateJob {
            model: ir_path.to_string(),
            params_files: params_file_opt.as_ref().map(|p| vec![p.clone()]).unwrap_or_default(),
            backend,
            dt,
            integrator: None,
            source: ParamSource::Sweep { points, replicates: 1 },
            scenarios: job_scenarios.clone(),
        // gh#626: batch TOML has no `to` key (deliberate; CLI-only override).
        t_end_override: None,
        // gh#641: nor an `init_state` key, for the same reason.
        init_state: None,
        // gh#616: `batch run` has no data-binding surface, so an anchored
        // model is refused at `CompiledModel::new` rather than resolved.
        obs_anchors: None,
            seeds: Seeds::Explicit(seeds.to_vec()),
            cli_overrides: Vec::new(),
            set_vec_entries: Vec::new(),
            table_files: Vec::new(),
            obs: crate::sim_job::ObsOutput::None,
            parallel,
        };

        let mut sink = CasSink {
            resolved_scenarios: resolved_scenarios.to_vec(),
            model_path: model_path.to_string(),
            model_stem: model_stem.map(|s| s.to_string()),
            base_model: batch_model.clone(),
            base_params: base_params.clone(),
            table_files: HashMap::new(),
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
    base_params: &HashMap<String, f64>,
    backend: crate::args::types::ForwardBackend,
    dt: f64,
    allow_degenerate_rates: bool,
    output_cols: &crate::util::OutputColumns,
    runs_dir: &str,
    seeds: &[u64],
    force: bool,
) {
    let probe = CasSink::probe(
        resolved_scenarios, batch_model, model_stem, base_params,
        backend, dt, allow_degenerate_rates, output_cols, runs_dir, force,
    );
    let predictions = probe.predict_cells(points, seeds);
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
/// The obs subtree's identity component.
///
/// A hash of the resolved observation blocks (run-spec: changing a reporting
/// parameter re-samples obs without invalidating the cached trajectory), plus
/// any `--emit-every` override (gh#656) — which changes WHICH times the subtree
/// carries, so two cadences must not address the same directory (the
/// count-in-the-key rule).
///
/// One function, because the hash is computed twice for every cell — once to
/// declare the `obs` child in `run.json`, once to name the subtree directory —
/// and a disagreement between the two would declare one child while writing
/// another. Byte-identical to the historical `sha256(json(observations))`
/// whenever no override is in play, so an unflagged run addresses exactly the
/// subtree it always did.
pub(crate) fn obs_subtree_hash(
    observations: &[ir::observation::ObservationModel],
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<String, String> {
    let mut buf = serde_json::to_string(observations)
        .map_err(|e| format!("cannot serialize observations for hashing: {}", e))?;
    if let Some(e) = emit {
        buf.push_str(&e.identity_repr());
    }
    Ok(crate::hashing::sha256_hex(buf.as_bytes()))
}

fn write_obs_into_cas(
    run_dir: &std::path::Path,
    model: &ir::Model,
    traj: &sim::Trajectory,
    process_seed: u64,
    // gh#641: `Some(T)` when this cell was restarted from a filtered state at
    // `T`; the obs axis then begins there, because the trajectory does. `None`
    // for every ordinary cell. NOTE this function serves `simulate` cells as
    // well as `batch run` ones — the obs subtree is written for both.
    restart_origin: Option<f64>,
    // gh#656: the `--emit-every` override — it sets the emitted times AND names
    // the subtree (via `obs_subtree_hash`), so the two cadences coexist under one
    // trajectory leaf instead of one overwriting the other.
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<(), String> {
    use std::io::Write;

    if model.observations.is_empty() {
        return Ok(());
    }

    let obs_hash = obs_subtree_hash(&model.observations, emit)?;
    let obs_seed = process_seed ^ crate::util::SEED_MIX_OBS;

    // Preflight EVERY stream before creating any file (gh#589 review).
    //
    // The per-stream loop below validates and writes in the same iteration, so
    // an aligned stream followed by a misaligned one would leave the first
    // stream's `.tsv` on disk while the error aborts before `obs.json` is
    // written — a child directory that exists, is incomplete, and carries no
    // provenance to say so. Validation is cheap and pure, so hoisting it turns
    // a partial write into no write.
    for obs_ir in &model.observations {
        // Must use the SAME horizon the write loop below uses — a preflight that
        // validates a different set of times than gets written is worse than no
        // preflight (gh#561 + gh#589).
        let times =
            crate::obs_emit_schedule_times(obs_ir, restart_origin, model.simulation.t_end, emit)?;
        // `None`: an emit_schedule-driven synthetic stream binds no data, so
        // there is no conditioning window to open the first bin at (gh#702).
        crate::project_all_obs_times(traj, obs_ir, model, &times, None)?;
    }

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
        // The cell's own horizon (a per-scenario `simulate { to }` has already
        // moved `model.simulation.t_end`), so the CAS `obs/` subtree never
        // carries rows past the end of the trajectory beside it (gh#561).
        let obs_times =
            crate::obs_emit_schedule_times(obs_ir, restart_origin, model.simulation.t_end, emit)?;
        // `None` — emit_schedule-driven, no data to condition on (gh#702).
        let projected =
            crate::project_all_obs_times(traj, obs_ir, model, &obs_times, None)?;

        let path = obs_dir.join(format!("{}.tsv", obs_ir.name));
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&path)
                .map_err(|e| format!("cannot create {}: {}", path.display(), e))?,
        );
        writeln!(out, "time\t{}", obs_ir.name).map_err(|e| e.to_string())?;
        for (ti, &obs_t) in obs_times.iter().enumerate() {
            let snap = crate::snap_at(traj, obs_t);
            let draw = sampler(projected[ti], obs_t, &snap.int_state.counts, &[], &mut obs_rng);
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

    let ir_json = match std::fs::read_to_string(&exp.config.model) {
        Ok(j) => j,
        Err(e) => {
            println!("  (cannot read model {}: {})", exp.config.model, e);
            println!("  Run 'camdl batch run {}' to start.", toml_path);
            return;
        }
    };

    let base_params: HashMap<String, f64> = exp.config.params.as_ref()
        .and_then(|p| load_params_toml(p).ok())
        .unwrap_or_default();

    // Parse the model and resolve every `[[scenario]]` against its presets, so
    // the prediction resolves each cell's identity exactly as the run path
    // does (CLI review #3). A parse/resolution failure means the model itself
    // is broken — report and bail rather than mis-predict from raw scenarios.
    let batch_model: ir::Model = match ir::from_str(&ir_json) {
        Ok(m) => m,
        Err(e) => {
            println!("  (cannot parse model {}: {})", exp.config.model, e);
            println!("  Run 'camdl batch run {}' to start.", toml_path);
            return;
        }
    };
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
    let points: Vec<indexmap::IndexMap<String, f64>> = expand_sweep(&exp.sweep)
        .iter()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
        .collect();

    let probe = CasSink::probe(
        &resolved_scenarios, &batch_model, model_stem.as_deref(), &base_params,
        exp.config.backend, exp.config.dt, false, &output_cols, &runs_dir, false,
    );
    let predictions = probe.predict_cells(&points, &seeds);
    let live_hits = predictions.iter().filter(|c| c.hit).count();
    println!("  Completed:  {}/{} leaves present", live_hits, predictions.len());

    // List the first few cells still to run (scenario / seed / path), so a
    // resumed `batch run` is predictable from `batch status`.
    let remaining: Vec<&CellPrediction> = predictions.iter().filter(|c| !c.hit).collect();
    if !remaining.is_empty() {
        println!("  Remaining:  {} cell(s) to run:", remaining.len());
        for c in remaining.iter().take(6) {
            println!("    scenario={} seed={}  → {}/{}", c.scenario, c.seed, runs_dir, c.rel);
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
        scenarios, batch_model, model_stem, base_params,
        backend, dt, allow_degenerate_rates, output_cols, runs_dir, force,
    );
    let predictions = probe.predict_cells(&points, seeds);

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

    // ── --table content folds into the run identity (Q1) ─────────────────────

    use ir::model::{
        InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule,
        SimulationConfig,
    };

    fn tiny_model() -> Model {
        Model {
            ic_grad: Default::default(),
            name: "sir".into(),
            version: "1".into(),
            time_unit: "days".into(),
            description: None,
            origin: None,
            origin_rata_die: None,
            compartments: vec![],
            transitions: vec![],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![],
            parameters: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            initial_conditions: InitialConditions::default(),
            output: OutputConfig {
                times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0,
                t_end: 100.0,
                time_semantics: "continuous".into(),
                dt: Some(1.0),
                rng_seed: None,
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }

    /// A `CasSink` whose only non-default identity input is the `--table` map,
    /// so a run_id difference between two such sinks is attributable to the
    /// table content alone.
    fn sink_with_tables(runs_dir: &str, table_files: HashMap<String, String>) -> CasSink {
        CasSink {
            resolved_scenarios: vec![ResolvedEntry {
                name: "baseline".to_string(),
                route: None,
                enable: vec![],
                disable: vec![],
                params: HashMap::new(),
                t_end: None,
            }],
            model_path: "model.ir.json".to_string(),
            model_stem: Some("sir".to_string()),
            base_model: tiny_model(),
            base_params: HashMap::new(),
            table_files,
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            allow_degenerate_rates: false,
            output_cols: crate::util::OutputColumns::default(),
            runs_dir: runs_dir.to_string(),
            obs_enabled: false,
            emit_every: None,
            force: false,
            total: 1,
            counter: 0,
            completed_runs: Vec::new(),
            errors: Vec::new(),
            label: None,
            fit_dep: Vec::new(),
            progress: None,
        }
    }

    fn baseline_spec() -> crate::engine::CellSpec {
        crate::engine::CellSpec {
            run_idx: 0,
            point_idx: 0,
            scenario: crate::sim_job::ScenarioRef::Inline {
                name: "baseline".to_string(),
                enable: vec![],
                disable: vec![],
                params: indexmap::IndexMap::new(),
            },
            point_overrides: indexmap::IndexMap::new(),
            process_seed: 1,
            obs_seed: 1 ^ crate::util::SEED_MIX_OBS,
            sim_run: crate::util::SimRun::default(),
        }
    }

    /// `--table NAME=PATH` is a runtime override: the IR carries only the
    /// table reference, so the file's *content* must enter the run_id. Two
    /// tables with the same NAME but different CONTENT must resolve to
    /// DIFFERENT sim run_ids (else a changed `matrix.tsv` silently serves the
    /// stale cached trajectory); identical content → the same run_id
    /// (count-in-the-key for tables). Exercises the shared `CasSink`, so the
    /// fix covers both `simulate` and `batch`.
    #[test]
    fn table_content_folds_into_sim_run_id() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();

        let path_a = dir.path().join("matrix.tsv");
        let path_b = dir.path().join("matrix2.tsv");
        let path_c = dir.path().join("matrix3.tsv");
        std::fs::write(&path_a, b"1.0\t2.0\n3.0\t4.0\n").unwrap();
        std::fs::write(&path_b, b"9.9\t8.8\n7.7\t6.6\n").unwrap(); // different content
        std::fs::write(&path_c, b"1.0\t2.0\n3.0\t4.0\n").unwrap(); // same content as A

        let spec = baseline_spec();

        let mk = |p: &std::path::Path| {
            let mut tf = HashMap::new();
            tf.insert("contact".to_string(), p.to_str().unwrap().to_string());
            sink_with_tables(runs_dir, tf)
        };

        let (rt_a, _, _) = mk(&path_a).cell_resolve(&spec).unwrap();
        let (rt_b, _, _) = mk(&path_b).cell_resolve(&spec).unwrap();
        let (rt_c, _, _) = mk(&path_c).cell_resolve(&spec).unwrap();

        // Negative control: with NO --table, the run_id is yet another value,
        // and it must differ from the with-table run_ids (so the table digest
        // is genuinely folded in, not silently dropped).
        let (rt_none, _, _) = sink_with_tables(runs_dir, HashMap::new())
            .cell_resolve(&spec)
            .unwrap();

        assert_ne!(
            rt_a.run_id, rt_b.run_id,
            "different --table content (same name) MUST produce different sim run_ids"
        );
        assert_eq!(
            rt_a.run_id, rt_c.run_id,
            "identical --table content MUST produce the same sim run_id"
        );
        assert_ne!(
            rt_a.run_id, rt_none.run_id,
            "a --table override must change the run_id vs no table (digest folded in)"
        );

        // The difference is isolated to the params level (tables live there),
        // never the model/config/scenario/seed levels.
        assert_eq!(rt_a.levels[0].hash, rt_b.levels[0].hash, "model level unchanged");
        assert_eq!(rt_a.levels[1].hash, rt_b.levels[1].hash, "config level unchanged");
        assert_ne!(rt_a.levels[2].hash, rt_b.levels[2].hash, "params level (tables) must differ");
        assert_eq!(rt_a.levels[3].hash, rt_b.levels[3].hash, "scenario level unchanged");
        assert_eq!(rt_a.levels[4].hash, rt_b.levels[4].hash, "seed level unchanged");
    }

    /// A `CellSpec` carrying a single param-point override (a sweep point /
    /// design point), at process seed 1, baseline scenario.
    fn spec_with_override(k: &str, v: f64) -> crate::engine::CellSpec {
        let mut overrides = indexmap::IndexMap::new();
        overrides.insert(k.to_string(), v);
        crate::engine::CellSpec {
            run_idx: 0,
            point_idx: 0,
            scenario: crate::sim_job::ScenarioRef::Inline {
                name: "baseline".to_string(),
                enable: vec![],
                disable: vec![],
                params: indexmap::IndexMap::new(),
            },
            point_overrides: overrides,
            process_seed: 1,
            obs_seed: 1 ^ crate::util::SEED_MIX_OBS,
            sim_run: crate::util::SimRun::default(),
        }
    }

    /// gh#241 PR E2 — THE dedupe proof. A `[design.*]` cell and a normal
    /// `[sweep]` / `simulate` sim cell with the SAME params/scenario/seed must
    /// resolve to the SAME `Sim` `run_id` and the SAME store path. The design
    /// path routes its generated points through the same `CasSink`/
    /// `cell_resolve` as the normal sweep (the point value enters the `params`
    /// identity level), so an identical cell collapses to one canonical leaf
    /// rather than two parallel copies under `designs/` vs `sims/`.
    ///
    /// The design method/name/point_idx are NOT in the identity: this test
    /// resolves a bare point override with no design label anywhere in the
    /// resolved inputs, which is exactly what `cell_resolve` sees for both
    /// entry points.
    #[test]
    fn design_cell_dedupes_with_normal_sim_cell() {
        let dir = tempfile::tempdir().unwrap();
        let runs_dir = dir.path().join("sims");
        let runs_dir = runs_dir.to_str().unwrap();

        // "Design" cell: point value 0.3 for `mu`, supplied as a design point.
        let design_spec = spec_with_override("mu", 0.3);
        // "Normal sweep" cell: the SAME point value 0.3 for `mu`.
        let sweep_spec = spec_with_override("mu", 0.3);
        // A DIFFERENT point value — must NOT dedupe.
        let other_spec = spec_with_override("mu", 0.7);

        // Both paths build the same `CasSink` shape (same base model, root,
        // backend, dt, no tables) — only the spec's point override differs.
        let (rt_design, dir_design, rel_design) =
            sink_with_tables(runs_dir, HashMap::new()).cell_resolve(&design_spec).unwrap();
        let (rt_sweep, dir_sweep, rel_sweep) =
            sink_with_tables(runs_dir, HashMap::new()).cell_resolve(&sweep_spec).unwrap();
        let (rt_other, _, _) =
            sink_with_tables(runs_dir, HashMap::new()).cell_resolve(&other_spec).unwrap();

        assert_eq!(
            rt_design.run_id, rt_sweep.run_id,
            "an identical design cell and normal sim cell MUST share one run_id (dedupe)"
        );
        assert_eq!(
            dir_design, dir_sweep,
            "the identical cell must resolve to the SAME store path (one leaf on disk)"
        );
        assert_eq!(rel_design, rel_sweep, "and the same store-relative path");

        assert_ne!(
            rt_design.run_id, rt_other.run_id,
            "a different design-point value must resolve to a different run_id"
        );
    }
}
