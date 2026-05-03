//! `camdl survey` — likelihood-landscape diagnostic.
//!
//! Draws N Latin-hypercube points across declared parameter bounds,
//! evaluates the marginal log-likelihood at each point via a particle
//! filter (default) or a single deterministic trajectory (opt-in), and
//! writes a TSV ready for visualization. Optional `--render` produces
//! a self-contained interactive HTML pair-plot.
//!
//! This is a **diagnostic tool**, not a fitting routine. It does not
//! produce an MLE. See
//! `docs/dev/proposals/2026-05-03-survey-subcommand.md`.
//!
//! ## CAS layout
//!
//! ```text
//! <root>/surveys/<stem>-<hash[:8]>/
//!   run.json            # RunKind::Survey(SurveyMeta)
//!   landscape.tsv       # primary artifact (always)
//!   summary.json        # SE distribution, top-K stats, dimensionality info
//!   landscape.html      # interactive pair-plot (only when --render)
//! ```
//!
//! Reuse paths:
//! - LHS sampling via `fit::init::build_chain_starts` (scale-aware)
//! - Bounds resolution via `fit::runner::build_if2_params_from_specs`
//! - PF eval via `sim::inference::particle_filter::bootstrap_filter`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        particle_filter::{bootstrap_filter, Observation, PFilterResult},
        ChainBinomialProcess, MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
        traits::{ObservationModel, SMCConfig},
        types::{log_sum_exp, EstimatedParam, ParticleState},
    },
};

use crate::cas::typed::{hash_canonical, CasInputs, ContentHash};
use crate::run_meta::{Run, RunKind, RunStatus, SurveyEvalMethod, SurveyMeta};

// ─── SurveyInputs (CAS) ──────────────────────────────────────────────────────

/// Typed CAS inputs for a single survey run. Mirrors `ProfileInputs`'s
/// pattern: every content-bearing input contributes to `content_hash`,
/// presentation hints (model_path, stem) appear in the path but not
/// the hash.
#[derive(Clone, Debug)]
pub struct SurveyInputs {
    /// Display-only model path. Recorded in `SurveyMeta.model`.
    pub model_path: String,
    /// Slugified stem from the model path.
    pub stem: Option<String>,
    /// Full SHA-256 of the IR JSON.
    pub model_hash: String,
    /// Per-stream data file content hashes — keyed by observation
    /// stream name. Content-only (gh#39): editing a TSV invalidates
    /// the cache.
    pub data_hashes: HashMap<String, String>,
    /// LHS box: parameter name → (lo, hi).
    pub bounds: HashMap<String, (f64, f64)>,
    /// Order of estimated parameters — drives TSV column order.
    pub estimated: Vec<String>,
    /// Resolved fixed parameters (name → value).
    pub fixed: HashMap<String, f64>,
    /// Named scenario applied before survey (`None` = baseline).
    pub scenario: Option<String>,
    pub n_points: usize,
    pub eval_method: SurveyEvalMethod,
    pub eval_particles: usize,
    pub eval_replicates: usize,
    pub seed: u64,
}

impl SurveyInputs {
    /// Canonical-form content hash. The fields included determine the
    /// cache key:
    ///
    /// - `model` — IR bytes (model_hash)
    /// - `data` — concatenated per-stream data hashes
    /// - `bounds` — sorted name=lo:hi list
    /// - `estimated` — param order (drives LHS dim assignment)
    /// - `fixed` — sorted name=value list
    /// - `scenario` — name (or empty)
    /// - `n_points`, `eval_method`, `eval_particles`,
    ///   `eval_replicates`, `seed`
    ///
    /// Bounds, fixed, and data_hashes are sorted by name before
    /// hashing so HashMap iteration order doesn't perturb the hash.
    pub fn canonical_hash(&self) -> ContentHash {
        let bounds_canonical = canonical_bounds_string(&self.bounds);
        let fixed_canonical = canonical_fixed_string(&self.fixed);
        let data_canonical = canonical_data_string(&self.data_hashes);
        let eval_canonical = format!(
            "method={};particles={};replicates={}",
            self.eval_method.as_str(),
            self.eval_particles,
            self.eval_replicates,
        );
        let n_points_str = self.n_points.to_string();
        let seed_str = self.seed.to_string();
        let scenario_ref = self.scenario.as_deref().unwrap_or("");
        let estimated_canonical = self.estimated.join(",");
        hash_canonical(&[
            ("model",      &self.model_hash),
            ("data",       &data_canonical),
            ("bounds",     &bounds_canonical),
            ("estimated",  &estimated_canonical),
            ("fixed",      &fixed_canonical),
            ("scenario",   scenario_ref),
            ("n_points",   &n_points_str),
            ("eval",       &eval_canonical),
            ("seed",       &seed_str),
        ])
    }
}

/// Sort `bounds` by parameter name and serialize as
/// `name1=lo1:hi1;name2=lo2:hi2;…`. Stable across HashMap iteration.
fn canonical_bounds_string(bounds: &HashMap<String, (f64, f64)>) -> String {
    let mut entries: Vec<(&String, &(f64, f64))> = bounds.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.iter()
        .map(|(k, (lo, hi))| format!("{}={}:{}", k, lo, hi))
        .collect::<Vec<_>>()
        .join(";")
}

fn canonical_fixed_string(fixed: &HashMap<String, f64>) -> String {
    let mut entries: Vec<(&String, &f64)> = fixed.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(";")
}

fn canonical_data_string(data: &HashMap<String, String>) -> String {
    let mut entries: Vec<(&String, &String)> = data.iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join(";")
}

impl CasInputs for SurveyInputs {
    fn content_hash(&self) -> ContentHash {
        self.canonical_hash()
    }

    fn cas_path(&self, root: &Path) -> PathBuf {
        let h = self.content_hash();
        let dirname = match &self.stem {
            Some(s) if !s.is_empty() => format!("{}-{}", s, h.short()),
            _ => h.short().to_string(),
        };
        root.join("surveys").join(dirname)
    }

    fn run_kind(&self) -> RunKind {
        RunKind::Survey(SurveyMeta {
            model:           self.model_path.clone(),
            model_hash:      self.model_hash.clone(),
            data_hashes:     self.data_hashes.clone(),
            bounds:          self.bounds.clone(),
            n_points:        self.n_points,
            eval_method:     self.eval_method,
            eval_particles:  self.eval_particles,
            eval_replicates: self.eval_replicates,
            seed:            self.seed,
            fixed:           self.fixed.clone(),
            scenario:        self.scenario.clone(),
            estimated:       self.estimated.clone(),
        })
    }
}

// ─── Resolved input payload ──────────────────────────────────────────────────
//
// The fit-aware (`--fit`) and inline (`--estimate` / `--data`) input
// modes converge here: a single `ResolvedSurveyInputs` carrying the
// loaded model, the EstimatedParam specs (with bounds resolved
// fit.toml > model), the per-stream observation data and the scenario
// / fixed context. Everything past this point is mode-agnostic.

struct ResolvedSurveyInputs {
    /// IR JSON bytes — used to compute `model_hash`.
    model_ir_json: String,
    /// Compiled model (Arc to share across rayon threads).
    compiled: Arc<CompiledModel>,
    /// Default parameter vector (post-fixed/scenario apply).
    base_params: Vec<f64>,
    /// EstimatedParam vector with resolved bounds — drives LHS.
    estimated: Vec<EstimatedParam>,
    /// Resolved IR observation models, in declaration order. The
    /// survey scores against ALL of them simultaneously (matches the
    /// fit-side multi-stream loglik convention).
    obs_models: Vec<ir::observation::ObservationModel>,
    /// Per-stream observations, aligned to `obs_models`.
    per_stream_obs: Vec<Vec<Observation>>,
    /// Per-stream data file content hashes, keyed by stream name.
    data_hashes: HashMap<String, String>,
    /// Resolved fixed params (name → value).
    fixed: HashMap<String, f64>,
    /// Named scenario applied (`None` = baseline).
    scenario: Option<String>,
}

// ─── cmd_survey entry point ──────────────────────────────────────────────────

pub fn cmd_survey(a: &crate::args::SurveyArgs) {
    // Validate input mode mutual exclusion at the boundary.
    if a.fit.is_none() {
        if a.data.is_none() {
            eprintln!(
                "error: camdl survey requires either --fit FIT.toml \
                 (fit-aware mode) or --data DATA.tsv with --estimate \
                 NAME=LO:HI flags (inline mode).\n\
                 Got neither.");
            std::process::exit(1);
        }
        if a.estimate.is_empty() {
            eprintln!(
                "error: --data {} given without any --estimate flags. \
                 Pass --estimate NAME=LO:HI for each parameter to vary \
                 across the LHS box (repeat for multiple parameters).",
                a.data.as_ref().unwrap().display());
            std::process::exit(1);
        }
    }

    if a.eval_replicates == 0 {
        eprintln!("error: --eval-replicates must be >= 1 (got 0).");
        std::process::exit(1);
    }
    if a.eval_particles == 0 && a.eval == SurveyEvalMethod::Pfilter {
        eprintln!("error: --eval pfilter requires --eval-particles >= 1.");
        std::process::exit(1);
    }
    if a.n_points == 0 {
        eprintln!("error: --n-points must be >= 1 (got 0).");
        std::process::exit(1);
    }

    let label_arg: Option<String> = match a.label.as_deref() {
        Some(raw) => match crate::fit::validate_label(raw) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("error: invalid --label: {}", e);
                std::process::exit(1);
            }
        },
        None => None,
    };

    // Configure rayon parallelism (best-effort; if a global pool is
    // already configured, this is a no-op).
    if a.parallel > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(a.parallel).build_global();
    }

    let resolved = match resolve_survey_inputs(a) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    // Curse-of-dim warnings (proposal §"Runtime warnings").
    let d = resolved.estimated.len();
    if d > 10 {
        eprintln!(
            "warning: surveying {} parameters; pair-plots become \
             hard to interpret past d ~= 8. Consider `camdl profile` \
             for higher-dimensional identifiability questions, or \
             restricting [estimate] to a focal subset.", d);
    } else if d > 6 {
        eprintln!(
            "note: surveying {} parameters; pair-plot 2D marginals \
             project a {}-D joint distribution. Concentrations in \
             one panel may reflect tight conditioning on parameters \
             not visible in that view.", d, d);
    }
    if d > 0 {
        let coverage_floor = 50.0 * (d as f64) * (d as f64);
        if (a.n_points as f64) < coverage_floor {
            eprintln!(
                "note: --n-points {} is below the rule-of-thumb \
                 coverage floor of n_points/d^2 >= 50 (d={}, \
                 recommended >= {}). Consider --n-points {} for \
                 adequate pair-plot resolution.",
                a.n_points, d, coverage_floor as usize, coverage_floor as usize);
        }
    }
    if a.eval == SurveyEvalMethod::Simulate {
        eprintln!(
            "warning: --eval simulate uses a single deterministic \
             trajectory per LHS point. This is a 1-sample MC estimator \
             of p(y|theta) — biased toward 'lucky outliers' when \
             process noise is non-trivial (Andrieu & Roberts 2009; \
             Doucet et al. 2015). Use --eval pfilter unless the \
             model is known-deterministic.");
    }

    // Build typed CAS inputs.
    let bounds_map: HashMap<String, (f64, f64)> = resolved.estimated.iter()
        .map(|ep| (ep.name.clone(), (ep.lower, ep.upper)))
        .collect();
    let estimated_names: Vec<String> = resolved.estimated.iter()
        .map(|ep| ep.name.clone()).collect();
    let stem = crate::hashing::path_stem_slug(&a.model.to_string_lossy());
    let inputs = SurveyInputs {
        model_path:      a.model.to_string_lossy().into_owned(),
        stem:            stem.clone(),
        model_hash:      crate::hashing::model_hash(&resolved.model_ir_json),
        data_hashes:     resolved.data_hashes.clone(),
        bounds:          bounds_map,
        estimated:       estimated_names,
        fixed:           resolved.fixed.clone(),
        scenario:        resolved.scenario.clone(),
        n_points:        a.n_points,
        eval_method:     a.eval,
        eval_particles:  a.eval_particles,
        eval_replicates: a.eval_replicates,
        seed:            a.seed,
    };

    let output_root = crate::run_paths::output_root(
        a.output.as_ref().map(|p| p.to_string_lossy().into_owned()).as_deref(),
        None,
    );
    let run_dir = inputs.cas_path(&output_root);
    if let Err(e) = std::fs::create_dir_all(&run_dir) {
        eprintln!("error: cannot create {}: {}", run_dir.display(), e);
        std::process::exit(1);
    }

    // Cache hit short-circuit (force=false, prior landscape.tsv exists,
    // hash matches). The TSV is the authoritative artifact: if it exists
    // and the run.json hash matches, the survey is done.
    let landscape_path = run_dir.join("landscape.tsv");
    let summary_path = run_dir.join("summary.json");
    let html_path = run_dir.join("landscape.html");
    let expected_hash = inputs.content_hash().full().to_string();
    if !a.force {
        match Run::check_cache(&run_dir, &expected_hash) {
            crate::run_meta::CacheStatus::Hit if landscape_path.exists() => {
                eprintln!("survey: cache hit at {}", run_dir.display());
                if a.render && !html_path.exists() {
                    eprintln!(
                        "  rendering --render HTML from cached landscape.tsv …");
                    if let Err(e) = render_landscape_html(
                        &landscape_path, &html_path, &inputs)
                    {
                        eprintln!("warning: HTML render failed: {}", e);
                    }
                }
                return;
            }
            _ => {}
        }
    }

    let argv: Vec<String> = std::env::args().collect();
    // Write a Running run.json so a crash mid-survey leaves a
    // discoverable trace.
    let mut run = inputs.to_run(crate::version::VERSION_SHORT.to_string(), argv);
    run.label = label_arg;
    if let Err(e) = run.write(&run_dir) {
        eprintln!("warning: could not write initial run.json: {}", e);
    }

    eprintln!("survey: {} ({} points, eval={})",
        run_dir.display(), a.n_points, a.eval);

    let t0 = std::time::Instant::now();

    // ── LHS sampling ────────────────────────────────────────────────
    //
    // gh#42's `build_chain_starts` is the scale-aware sampler. LHS
    // requires n >= 2; reject n_points = 1 upstream so the call here
    // doesn't degenerate to "just use base_params".
    let lhs_starts = crate::fit::init::build_chain_starts(
        crate::fit::init::InitMethod::Lhs,
        &resolved.estimated,
        a.n_points,
        a.seed,
    ).unwrap_or_else(|| {
        // n_points < 2 already rejected; this is unreachable today.
        eprintln!("internal error: LHS sampler returned None at n_points={}", a.n_points);
        std::process::exit(1);
    });

    // ── Parallel evaluation loop ────────────────────────────────────
    let process = Arc::new(ChainBinomialProcess::new(resolved.compiled.clone()));
    let dt = resolved.compiled.model.simulation.t_start;
    let dt = if dt.is_finite() { 1.0 } else { 1.0 };
    let _ = dt; // dt is configured via SMCConfig below; survey doesn't expose it.
    let smc_dt = 1.0_f64;
    let t_start = resolved.compiled.model.simulation.t_start;

    let obs_model: Arc<dyn ObservationModel<ParticleState> + Send + Sync> = {
        let obs_times: Vec<f64> = resolved.per_stream_obs.first()
            .map(|v| v.iter().map(|o| o.time).collect())
            .unwrap_or_default();
        let mut stream_specs = Vec::with_capacity(resolved.obs_models.len());
        for (obs, stream_obs) in resolved.obs_models.iter().zip(resolved.per_stream_obs.iter()) {
            let projection = sim::inference::multi_stream_obs::StreamProjection::from_ir(
                &obs.projection, &resolved.compiled, &obs.name,
            ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
            stream_specs.push(StreamSpec {
                projection,
                ir_model: obs.clone(),
                observations: stream_obs.iter().map(|o| o.value).collect(),
                obs_times: obs_times.clone(),
            });
        }
        Arc::new(MultiStreamObsModel::new(stream_specs, resolved.compiled.clone())
            .unwrap_or_else(|e| {
                eprintln!("error: observation model construction failed: {:?}", e);
                std::process::exit(1);
            }))
    };

    let progress = std::sync::atomic::AtomicUsize::new(0);
    let total = a.n_points;
    let progress_step = (total / 20).max(1);
    let rows: Vec<LandscapeRow> = lhs_starts.par_iter().enumerate()
        .map(|(point_id, draw)| {
            // Build the full parameter vector: base_params overwritten
            // at each estimated index. Fixed params are already baked
            // into base_params (resolve_survey_inputs).
            let mut params = resolved.base_params.clone();
            for spec in draw {
                params[spec.index] = spec.initial;
            }
            let row = match a.eval {
                SurveyEvalMethod::Pfilter => eval_point_pfilter(
                    &process, obs_model.as_ref(),
                    &params, &resolved.estimated, draw,
                    a.eval_particles, a.eval_replicates,
                    smc_dt, t_start, a.seed, point_id,
                ),
                SurveyEvalMethod::Simulate => eval_point_simulate(
                    &process, obs_model.as_ref(),
                    &params, &resolved.estimated, draw,
                    smc_dt, t_start, a.seed, point_id,
                ),
            };
            let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if done % progress_step == 0 || done == total {
                eprintln!("  survey progress: {} / {} points", done, total);
            }
            row
        })
        .collect();

    // ── TSV writer (sorted by loglik desc) ──────────────────────────
    let mut sorted = rows;
    sorted.sort_by(|a, b| {
        // -inf goes to the bottom; NaN treated as -inf for sort stability.
        let av = if a.loglik.is_nan() { f64::NEG_INFINITY } else { a.loglik };
        let bv = if b.loglik.is_nan() { f64::NEG_INFINITY } else { b.loglik };
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Err(e) = write_landscape_tsv(
        &landscape_path, &sorted, &resolved.estimated, a.eval,
        &expected_hash) {
        eprintln!("error writing landscape.tsv: {}", e);
        std::process::exit(1);
    }
    eprintln!("survey: wrote {} ({} rows)", landscape_path.display(), sorted.len());

    // ── SE-distribution warning (proposal §"Runtime warnings") ──────
    if a.eval == SurveyEvalMethod::Pfilter {
        emit_se_warning(&sorted);
    }

    // ── summary.json ────────────────────────────────────────────────
    if let Err(e) = write_summary_json(&summary_path, &sorted, a, d) {
        eprintln!("warning: could not write summary.json: {}", e);
    }

    // ── Render HTML if requested ────────────────────────────────────
    if a.render {
        if let Err(e) = render_landscape_html(&landscape_path, &html_path, &inputs) {
            eprintln!("warning: HTML render failed: {}", e);
        } else {
            eprintln!("survey: wrote {}", html_path.display());
        }
    }

    // ── Patch run.json with Completed status ────────────────────────
    let elapsed = t0.elapsed().as_secs_f64();
    run.status = RunStatus::Completed { wall_time_seconds: elapsed };
    if let Err(e) = run.write(&run_dir) {
        eprintln!("warning: could not patch run.json: {}", e);
    }
}

// ─── Resolution ──────────────────────────────────────────────────────────────

fn resolve_survey_inputs(a: &crate::args::SurveyArgs)
    -> Result<ResolvedSurveyInputs, String>
{
    use crate::fit::config_v2::FitConfigV2;
    use crate::fit::runner::{build_if2_params_from_specs, ParamSpec};

    let model_path = a.model.to_string_lossy().into_owned();

    if let Some(fit_path) = a.fit.as_ref() {
        // Fit-aware mode: load fit.toml; pull bounds from [estimate],
        // data from [data], fixed from [fixed], scenario from top.
        let fit_path_str = fit_path.to_string_lossy().into_owned();
        let mut config = FitConfigV2::load(&fit_path_str)?;
        // Make scenario+enable+disable mutual exclusion explicit
        // (matches fit::runner::FitRunConfig::build).
        if config.scenario.is_some() && (!config.enable.is_empty() || !config.disable.is_empty()) {
            return Err(
                "fit.toml: `scenario` is mutually exclusive with `enable`/`disable`. \
                 Use one approach.".into());
        }

        // Load model from fit.toml's `model.camdl` (already path-
        // resolved by FitConfigV2::load).
        let (mut model, model_ir_json) = crate::util::load_model(&config.model.camdl)?;

        // Apply scenario.
        let (enable_list, disable_list) = if let Some(ref name) = config.scenario {
            let preset = model.presets.iter().find(|p| p.name == *name).cloned()
                .ok_or_else(|| format!("scenario '{}' not found in model", name))?;
            for p in &mut model.parameters {
                if let Some(&v) = preset.params.get(&p.name) { p.value = Some(v); }
            }
            (preset.enable, preset.disable)
        } else {
            (config.enable.clone(), config.disable.clone())
        };
        crate::util::apply_scenario_filter(&mut model, &enable_list, &disable_list)?;

        // Resolve [fixed] (file load, scenario lookup, inline overlay).
        config.fixed.expand_from_scenario(&model)?;
        let fixed_resolved = config.fixed.resolve_with_model(&model)?;

        // Apply estimate.start and fixed values to model so the
        // base_params built from compiled.default_params has the right
        // numbers in the non-LHS slots.
        for (name, spec) in &config.estimate {
            if let Some(p) = model.parameters.iter_mut().find(|p| p.name == *name) {
                if p.value.is_none() {
                    let v = spec.start.unwrap_or_else(|| {
                        let (lo, hi) = spec.bounds;
                        if lo > 0.0 && hi > 0.0 { (lo * hi).sqrt() }
                        else { 0.5 * (lo + hi) }
                    });
                    p.value = Some(v);
                }
            }
        }
        for (name, &v) in &fixed_resolved {
            if let Some(p) = model.parameters.iter_mut().find(|p| p.name == *name) {
                if p.value.is_none() { p.value = Some(v); }
            }
        }
        crate::util::validate_parameter_values(&model)?;

        let compiled = Arc::new(CompiledModel::new(model.clone())
            .map_err(|e| format!("compile error: {:?}", e))?);
        let mut base_params = compiled.default_params.clone();
        for (name, spec) in &config.estimate {
            if let Some(start) = spec.start {
                if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                    base_params[idx] = start;
                }
            }
        }
        for (name, &v) in &fixed_resolved {
            if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                base_params[idx] = v;
            }
        }

        // Build EstimatedParam vector from [estimate]'s bounds, in
        // declaration order — the fit-toml-bounds-within-model-bounds
        // check is in `build_if2_params_from_specs`.
        let specs: Vec<ParamSpec> = config.estimate.iter()
            .map(|(name, spec)| ParamSpec {
                name: name.clone(),
                rw_sd: spec.rw_sd,
                transform: spec.transform.as_ref().map(|t| t.as_str().to_string()),
                ivp: spec.ivp,
                bounds: Some(spec.bounds),
            })
            .collect();
        let estimated = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)?;

        // Load observations from [data] and hash bytes per stream.
        let data_spec = config.data_spec()?;
        let model_obs_names: Vec<String> = model.observations.iter()
            .map(|o| o.name.clone()).collect();
        let effective = data_spec.effective_observations(&model_obs_names)?;
        if effective.is_empty() {
            return Err("fit.toml [data] resolves to zero observation streams.".into());
        }
        // Sort by name so order is canonical.
        let mut entries: Vec<(&String, &String)> = effective.iter().collect();
        entries.sort_by_key(|(k, _)| k.as_str());

        let mut obs_models = Vec::new();
        let mut per_stream_obs = Vec::new();
        let mut data_hashes: HashMap<String, String> = HashMap::new();
        let mut canonical_times: Option<Vec<f64>> = None;
        for (stream_name, data_path) in &entries {
            let obs_model_ir = model.observations.iter()
                .find(|o| o.name == **stream_name).cloned()
                .ok_or_else(|| format!(
                    "no observation block named '{}' in model", stream_name))?;
            let observations = load_observations_from_tsv(data_path, stream_name)?;
            let times: Vec<f64> = observations.iter().map(|o| o.time).collect();
            match &canonical_times {
                None => canonical_times = Some(times),
                Some(ct) => {
                    if ct.len() != times.len()
                        || ct.iter().zip(&times).any(|(a, b)| (a - b).abs() > 1e-9) {
                        return Err(format!(
                            "observation times for stream '{}' differ from the first \
                             stream; all streams must share identical schedules.",
                            stream_name));
                    }
                }
            }
            let bytes = std::fs::read(data_path)
                .map_err(|e| format!("cannot read data file '{}': {}", data_path, e))?;
            data_hashes.insert((*stream_name).clone(), crate::hashing::sha256_hex(&bytes));
            obs_models.push(obs_model_ir);
            per_stream_obs.push(observations);
        }

        Ok(ResolvedSurveyInputs {
            model_ir_json,
            compiled,
            base_params,
            estimated,
            obs_models,
            per_stream_obs,
            data_hashes,
            fixed: fixed_resolved.into_iter().collect(),
            scenario: config.scenario,
        })
    } else {
        // Inline mode: --estimate flags + --data (already validated).
        let data_path = a.data.as_ref().unwrap().to_string_lossy().into_owned();
        let (mut model, model_ir_json) = crate::util::load_model(&model_path)?;

        // Apply scenario if specified.
        let mut enable_list = Vec::new();
        let mut disable_list = Vec::new();
        if let Some(ref name) = a.scenario {
            let preset = model.presets.iter().find(|p| p.name == *name).cloned()
                .ok_or_else(|| format!("scenario '{}' not found in model", name))?;
            for p in &mut model.parameters {
                if let Some(&v) = preset.params.get(&p.name) { p.value = Some(v); }
            }
            enable_list = preset.enable;
            disable_list = preset.disable;
        }
        crate::util::apply_scenario_filter(&mut model, &enable_list, &disable_list)?;

        // Apply --fixed overrides to model parameter values.
        let fixed_map: HashMap<String, f64> = a.fixed.iter()
            .map(|p| (p.name.clone(), p.value)).collect();
        for (name, &v) in &fixed_map {
            if let Some(p) = model.parameters.iter_mut().find(|p| p.name == *name) {
                p.value = Some(v);
            }
        }
        crate::util::validate_parameter_values(&model)?;

        let compiled = Arc::new(CompiledModel::new(model.clone())
            .map_err(|e| format!("compile error: {:?}", e))?);
        let mut base_params = compiled.default_params.clone();
        for (name, &v) in &fixed_map {
            if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                base_params[idx] = v;
            }
        }

        // Build EstimatedParam vector from --estimate flags.
        let specs: Vec<ParamSpec> = a.estimate.iter().map(|e| ParamSpec {
            name: e.name.clone(),
            rw_sd: None,
            transform: None,
            ivp: false,
            bounds: Some((e.lo, e.hi)),
        }).collect();
        let estimated = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)?;

        // Inline mode: data is a single file. If the model has one
        // observation, score against it; otherwise treat the file as
        // a wide TSV with one column per declared stream.
        if model.observations.is_empty() {
            return Err("model declares no observations; survey requires \
                an observation block to score against.".into());
        }
        let mut obs_models = Vec::new();
        let mut per_stream_obs = Vec::new();
        let mut data_hashes: HashMap<String, String> = HashMap::new();
        let bytes = std::fs::read(&data_path)
            .map_err(|e| format!("cannot read --data file '{}': {}", data_path, e))?;
        let data_hash = crate::hashing::sha256_hex(&bytes);

        // Sort observation names for canonical ordering.
        let mut sorted_obs: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().collect();
        sorted_obs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut canonical_times: Option<Vec<f64>> = None;
        for obs in sorted_obs {
            let observations = load_observations_from_tsv(&data_path, &obs.name)?;
            let times: Vec<f64> = observations.iter().map(|o| o.time).collect();
            match &canonical_times {
                None => canonical_times = Some(times),
                Some(ct) => {
                    if ct.len() != times.len()
                        || ct.iter().zip(&times).any(|(x, y)| (x - y).abs() > 1e-9) {
                        return Err(format!(
                            "observation times for stream '{}' differ from the first \
                             stream; all streams must share identical schedules.",
                            obs.name));
                    }
                }
            }
            data_hashes.insert(obs.name.clone(), data_hash.clone());
            obs_models.push(obs.clone());
            per_stream_obs.push(observations);
        }

        Ok(ResolvedSurveyInputs {
            model_ir_json,
            compiled,
            base_params,
            estimated,
            obs_models,
            per_stream_obs,
            data_hashes,
            fixed: fixed_map,
            scenario: a.scenario.clone(),
        })
    }
}

/// Load (time, value) pairs from a TSV column. Mirrors profile's
/// load helper: by-name lookup with fallback to column 1 for 2-column
/// TSVs.
fn load_observations_from_tsv(path: &str, column: &str)
    -> Result<Vec<Observation>, String>
{
    let by_name = crate::pfilter::load_data_tsv_column(path, column);
    let raw = match by_name {
        Ok(v) => v,
        Err(_) => crate::pfilter::load_data_tsv_pub(path)?,
    };
    Ok(raw.into_iter().map(|o| Observation { time: o.time, value: o.value }).collect())
}

// ─── Per-point evaluation ────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct LandscapeRow {
    point_id: usize,
    /// Parameter values at the natural scale, in `estimated` order.
    param_values: Vec<f64>,
    loglik: f64,
    loglik_se: f64,
    /// Mean ESS across observation times. NaN when --eval simulate.
    mean_ess: f64,
    n_replicates: usize,
}

#[allow(clippy::too_many_arguments)]
fn eval_point_pfilter(
    process: &ChainBinomialProcess,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    params: &[f64],
    estimated: &[EstimatedParam],
    draw: &[EstimatedParam],
    n_particles: usize,
    n_replicates: usize,
    dt: f64,
    t_start: f64,
    seed_base: u64,
    point_id: usize,
) -> LandscapeRow {
    // Per-point per-replicate seed, derived from (seed_base, point_id, rep).
    let mut log_liks: Vec<f64> = Vec::with_capacity(n_replicates);
    let mut ess_values: Vec<f64> = Vec::new();
    for rep in 0..n_replicates {
        let seed = derive_point_seed(seed_base, point_id, rep);
        let cfg = SMCConfig {
            n_particles,
            dt,
            t_start,
            skip_first_obs_from_loglik: false,
            record_ancestry: false,
            record_prequential: false,
        };
        match bootstrap_filter(process, obs_model, params, &cfg, seed) {
            Ok(PFilterResult { log_likelihood, ess_trace, .. }) => {
                log_liks.push(log_likelihood);
                if !ess_trace.is_empty() {
                    let mean = ess_trace.iter().sum::<f64>() / ess_trace.len() as f64;
                    ess_values.push(mean);
                }
            }
            Err(_) => {
                log_liks.push(f64::NEG_INFINITY);
            }
        }
    }
    let n = log_liks.len() as f64;
    let log_n = n.ln();
    let logmeanexp = log_sum_exp(&log_liks) - log_n;
    // Replicate SE on the natural log-likelihood scale.
    let se = if log_liks.iter().any(|x| !x.is_finite()) || log_liks.len() < 2 {
        0.0
    } else {
        let mean = log_liks.iter().sum::<f64>() / n;
        let var = log_liks.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        (var / n).sqrt()
    };
    let mean_ess = if ess_values.is_empty() {
        f64::NAN
    } else {
        ess_values.iter().sum::<f64>() / ess_values.len() as f64
    };
    LandscapeRow {
        point_id,
        param_values: estimated.iter()
            .map(|spec| draw.iter().find(|d| d.index == spec.index)
                .map(|d| d.initial)
                .unwrap_or(params[spec.index]))
            .collect(),
        loglik: logmeanexp,
        loglik_se: se,
        mean_ess,
        n_replicates,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_point_simulate(
    process: &ChainBinomialProcess,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    params: &[f64],
    estimated: &[EstimatedParam],
    draw: &[EstimatedParam],
    dt: f64,
    t_start: f64,
    seed_base: u64,
    point_id: usize,
) -> LandscapeRow {
    // "Single deterministic trajectory" eval is implemented as a 1-particle
    // bootstrap filter: cheap, exercises the same propensity machinery as
    // the PF path, and degenerates to a single trajectory's loglik. SE
    // is undefined (no replicates) — set to 0.0 per proposal §"TSV output".
    let cfg = SMCConfig {
        n_particles: 1,
        dt,
        t_start,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
    };
    let seed = derive_point_seed(seed_base, point_id, 0);
    let loglik = match bootstrap_filter(process, obs_model, params, &cfg, seed) {
        Ok(r) => r.log_likelihood,
        Err(_) => f64::NEG_INFINITY,
    };
    LandscapeRow {
        point_id,
        param_values: estimated.iter()
            .map(|spec| draw.iter().find(|d| d.index == spec.index)
                .map(|d| d.initial)
                .unwrap_or(params[spec.index]))
            .collect(),
        loglik,
        loglik_se: 0.0,
        mean_ess: f64::NAN,
        n_replicates: 1,
    }
}

/// Per-(point, rep) seed mixer. ChaCha8 maps seeds to streams uniformly,
/// so any pairwise-distinct mixing is fine; we pick one inspired by
/// the rest of camdl's seed derivation (golden-ratio constants).
fn derive_point_seed(base: u64, point_id: usize, rep: usize) -> u64 {
    const SEED_MIX_POINT: u64 = 0x9e37_79b9_7f4a_7c15;
    const SEED_MIX_REP:   u64 = 0x517c_c1b7_2722_0a95;
    base ^ (point_id as u64).wrapping_mul(SEED_MIX_POINT)
         ^ (rep as u64).wrapping_mul(SEED_MIX_REP)
}

// ─── TSV writer ──────────────────────────────────────────────────────────────

fn write_landscape_tsv(
    path: &Path,
    rows: &[LandscapeRow],
    estimated: &[EstimatedParam],
    eval: SurveyEvalMethod,
    run_hash: &str,
) -> std::io::Result<()> {
    use std::io::Write as _;
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        // Comment header (TSV consumers tolerant of leading `#` lines).
        writeln!(f, "# camdl survey landscape; run_hash={}; version={}",
            run_hash, crate::version::VERSION_SHORT)?;
        writeln!(f, "# eval={}; n_points={}", eval.as_str(), rows.len())?;
        // Header row: param columns, then loglik / loglik_se /
        // (mean_ess if pfilter) / n_replicates / point_id.
        let mut cols: Vec<String> = estimated.iter().map(|ep| ep.name.clone()).collect();
        cols.push("loglik".into());
        cols.push("loglik_se".into());
        if eval == SurveyEvalMethod::Pfilter {
            cols.push("mean_ess".into());
        }
        cols.push("n_replicates".into());
        cols.push("point_id".into());
        writeln!(f, "{}", cols.join("\t"))?;
        for r in rows {
            let mut fields: Vec<String> = r.param_values.iter()
                .map(|v| format_float(*v)).collect();
            fields.push(format_float(r.loglik));
            fields.push(format_float(r.loglik_se));
            if eval == SurveyEvalMethod::Pfilter {
                fields.push(format_float(r.mean_ess));
            }
            fields.push(r.n_replicates.to_string());
            fields.push(r.point_id.to_string());
            writeln!(f, "{}", fields.join("\t"))?;
        }
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn format_float(v: f64) -> String {
    if v.is_nan() { "NaN".into() }
    else if v == f64::INFINITY  { "Inf".into() }
    else if v == f64::NEG_INFINITY { "-Inf".into() }
    else { format!("{}", v) }
}

// ─── SE warning ──────────────────────────────────────────────────────────────

fn emit_se_warning(rows: &[LandscapeRow]) {
    // Doucet et al. 2015 (Biometrika): per-point loglik SE > ~1.7 nats
    // makes pseudo-marginal MCMC ranks unreliable. Survey isn't doing
    // PMMH but the same bar applies to ranking N points.
    const DOUCET: f64 = 1.7;
    let finite_se: Vec<f64> = rows.iter()
        .map(|r| r.loglik_se)
        .filter(|s| s.is_finite()).collect();
    if finite_se.is_empty() { return; }
    let n = finite_se.len();
    let above = finite_se.iter().filter(|&&s| s > DOUCET).count();
    let pct = 100.0 * (above as f64) / (n as f64);
    if pct > 25.0 {
        eprintln!(
            "warning: {:.0}% of survey points have loglik_se > {} nats — \
             ranks for those points are unreliable. Consider:\n  \
             --eval-replicates 5  (3x compute, ~sqrt(5/3) variance reduction)\n  \
             --eval-particles 500 (2.5x compute, lower per-replicate variance)",
            pct, DOUCET);
    }
}

// ─── summary.json ────────────────────────────────────────────────────────────

fn write_summary_json(
    path: &Path,
    rows: &[LandscapeRow],
    a: &crate::args::SurveyArgs,
    d: usize,
) -> std::io::Result<()> {
    let finite_lls: Vec<f64> = rows.iter()
        .map(|r| r.loglik)
        .filter(|x| x.is_finite()).collect();
    let top_loglik = finite_lls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let top_loglik = if top_loglik == f64::NEG_INFINITY { None } else { Some(top_loglik) };

    let finite_ses: Vec<f64> = rows.iter()
        .map(|r| r.loglik_se)
        .filter(|x| x.is_finite()).collect();
    let se_q = quartiles(&finite_ses);

    // Top-K (default 5) param-value ranges, just for the summary
    // (visualization is via the HTML).
    let top_k = 5;
    let top_rows: Vec<&LandscapeRow> = rows.iter().take(top_k).collect();

    let summary = serde_json::json!({
        "n_points": rows.len(),
        "dimensions": d,
        "eval_method": a.eval.as_str(),
        "eval_particles": a.eval_particles,
        "eval_replicates": a.eval_replicates,
        "seed": a.seed,
        "top_loglik": top_loglik,
        "loglik_se_quartiles": se_q,
        "top_k_count": top_rows.len(),
        "n_finite_loglik": finite_lls.len(),
    });
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&summary)
        .map_err(std::io::Error::other)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn quartiles(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let pick = |q: f64| -> f64 {
        let idx = ((n as f64 - 1.0) * q).round() as usize;
        v[idx.min(n - 1)]
    };
    serde_json::json!({
        "min":    v[0],
        "q25":    pick(0.25),
        "median": pick(0.50),
        "q75":    pick(0.75),
        "max":    v[n - 1],
        "n":      n,
    })
}

// ─── HTML rendering (stub — fleshed out in landscape_html commit) ───────────

fn render_landscape_html(
    _landscape_path: &Path,
    html_path: &Path,
    inputs: &SurveyInputs,
) -> Result<(), String> {
    crate::landscape_html::render(html_path, inputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs() -> SurveyInputs {
        let mut bounds = HashMap::new();
        bounds.insert("beta".into(),  (0.001_f64, 1.0_f64));
        bounds.insert("gamma".into(), (0.01_f64,  0.5_f64));
        let mut data_hashes = HashMap::new();
        data_hashes.insert("cases".into(), "deadbeef".repeat(8));
        SurveyInputs {
            model_path:      "sir.camdl".into(),
            stem:            Some("sir".into()),
            model_hash:      "f00d".repeat(16),
            data_hashes,
            bounds,
            estimated:       vec!["beta".into(), "gamma".into()],
            fixed:           HashMap::new(),
            scenario:        None,
            n_points:        100,
            eval_method:     SurveyEvalMethod::Pfilter,
            eval_particles:  200,
            eval_replicates: 3,
            seed:            42,
        }
    }

    #[test]
    fn canonical_hash_is_deterministic() {
        let a = sample_inputs().canonical_hash();
        let b = sample_inputs().canonical_hash();
        assert_eq!(a, b, "same inputs must produce the same hash");
        assert_eq!(a.full().len(), 64);
    }

    #[test]
    fn canonical_hash_invariant_under_hashmap_order() {
        let mut s1 = sample_inputs();
        let mut s2 = sample_inputs();
        s1.bounds.clear();
        s1.bounds.insert("gamma".into(), (0.01, 0.5));
        s1.bounds.insert("beta".into(),  (0.001, 1.0));
        s2.bounds.clear();
        s2.bounds.insert("beta".into(),  (0.001, 1.0));
        s2.bounds.insert("gamma".into(), (0.01, 0.5));
        assert_eq!(s1.canonical_hash(), s2.canonical_hash());
    }

    #[test]
    fn different_bounds_change_the_hash() {
        let s_a = sample_inputs();
        let mut s_b = sample_inputs();
        s_b.bounds.insert("beta".into(), (0.005, 1.0));
        assert_ne!(s_a.canonical_hash(), s_b.canonical_hash(),
            "bounds must contribute to the cache key");
    }

    #[test]
    fn different_eval_config_changes_the_hash() {
        let s_a = sample_inputs();
        let mut s_b = sample_inputs();
        s_b.eval_particles = 500;
        assert_ne!(s_a.canonical_hash(), s_b.canonical_hash());
        let mut s_c = sample_inputs();
        s_c.eval_replicates = 5;
        assert_ne!(s_a.canonical_hash(), s_c.canonical_hash());
        let mut s_d = sample_inputs();
        s_d.eval_method = SurveyEvalMethod::Simulate;
        assert_ne!(s_a.canonical_hash(), s_d.canonical_hash());
    }

    #[test]
    fn different_data_contents_change_the_hash() {
        let s_a = sample_inputs();
        let mut s_b = sample_inputs();
        s_b.data_hashes.insert("cases".into(), "00000000".repeat(8));
        assert_ne!(s_a.canonical_hash(), s_b.canonical_hash(),
            "data file contents (digest) must contribute to the cache key");
    }

    #[test]
    fn different_seed_changes_the_hash() {
        let s_a = sample_inputs();
        let mut s_b = sample_inputs();
        s_b.seed = 7;
        assert_ne!(s_a.canonical_hash(), s_b.canonical_hash());
    }

    #[test]
    fn cas_path_uses_stem_plus_short_hash() {
        let s = sample_inputs();
        let p = s.cas_path(Path::new("/results"));
        let h = s.content_hash();
        assert_eq!(p, Path::new("/results/surveys").join(format!("sir-{}", h.short())));
    }

    #[test]
    fn run_kind_round_trips_through_serde() {
        let s = sample_inputs();
        let kind = s.run_kind();
        let json = serde_json::to_string(&kind).unwrap();
        assert!(json.contains(r#""kind":"survey""#));
        let parsed: RunKind = serde_json::from_str(&json).unwrap();
        match parsed {
            RunKind::Survey(m) => {
                assert_eq!(m.estimated, vec!["beta", "gamma"]);
                assert_eq!(m.n_points, 100);
            }
            _ => panic!("expected Survey kind"),
        }
    }

    #[test]
    fn landscape_tsv_header_includes_estimated_and_diagnostic_columns() {
        // Column order: estimated names (in declaration order), then
        // loglik, loglik_se, mean_ess (pfilter only), n_replicates, point_id.
        use sim::inference::types::{Transform, EstimatedParam};
        let estimated = vec![
            EstimatedParam {
                name: "beta".into(), index: 0, initial: 0.5, rw_sd: 0.1,
                transform: Transform::None, lower: 0.0, upper: 1.0,
                rw_sd_auto: false, ivp: false,
            },
            EstimatedParam {
                name: "gamma".into(), index: 1, initial: 0.2, rw_sd: 0.1,
                transform: Transform::None, lower: 0.01, upper: 0.5,
                rw_sd_auto: false, ivp: false,
            },
        ];
        let rows = vec![
            LandscapeRow {
                point_id: 0,
                param_values: vec![0.3, 0.15],
                loglik: -123.4,
                loglik_se: 0.5,
                mean_ess: 180.0,
                n_replicates: 3,
            },
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("landscape.tsv");
        write_landscape_tsv(&path, &rows, &estimated, SurveyEvalMethod::Pfilter, "deadbeef")
            .unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        // First two lines are comments.
        assert!(lines[0].starts_with("# camdl survey"));
        assert!(lines[1].starts_with("# eval="));
        // Header: beta, gamma, loglik, loglik_se, mean_ess, n_replicates, point_id
        let header: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(header,
            vec!["beta", "gamma", "loglik", "loglik_se", "mean_ess", "n_replicates", "point_id"]);
        // Data row.
        let row: Vec<&str> = lines[3].split('\t').collect();
        assert_eq!(row.len(), 7);
        assert_eq!(row[6], "0");
    }

    #[test]
    fn landscape_tsv_simulate_omits_mean_ess() {
        use sim::inference::types::{Transform, EstimatedParam};
        let estimated = vec![EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.5, rw_sd: 0.1,
            transform: Transform::None, lower: 0.0, upper: 1.0,
            rw_sd_auto: false, ivp: false,
        }];
        let rows = vec![
            LandscapeRow {
                point_id: 0,
                param_values: vec![0.3],
                loglik: -123.4,
                loglik_se: 0.0,
                mean_ess: f64::NAN,
                n_replicates: 1,
            },
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("landscape.tsv");
        write_landscape_tsv(&path, &rows, &estimated, SurveyEvalMethod::Simulate, "h")
            .unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        let header: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(header, vec!["beta", "loglik", "loglik_se", "n_replicates", "point_id"]);
    }

    #[test]
    fn quartiles_handles_small_inputs() {
        // Empty → null.
        assert!(quartiles(&[]).is_null());
        // Single value.
        let q = quartiles(&[1.0]);
        assert_eq!(q.get("min").and_then(|v| v.as_f64()), Some(1.0));
        assert_eq!(q.get("max").and_then(|v| v.as_f64()), Some(1.0));
        // Standard 5-number summary on a known sequence.
        let q = quartiles(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(q.get("median").and_then(|v| v.as_f64()), Some(3.0));
    }

    #[test]
    fn point_seed_distinguishes_points_and_reps() {
        // (point, rep) → distinct seeds. Identical (point, rep) →
        // identical seeds (deterministic).
        let s_a = derive_point_seed(42, 0, 0);
        let s_b = derive_point_seed(42, 0, 0);
        assert_eq!(s_a, s_b);
        assert_ne!(derive_point_seed(42, 0, 0), derive_point_seed(42, 0, 1));
        assert_ne!(derive_point_seed(42, 0, 0), derive_point_seed(42, 1, 0));
        assert_ne!(derive_point_seed(42, 0, 0), derive_point_seed(43, 0, 0));
    }
}
