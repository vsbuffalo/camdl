//! `camdl fit predict` — the free-forward posterior predictive verb, plus the
//! type system that makes a misread predictive object unrepresentable.
//!
//! A predictive varies on two independent axes the dangerous bugs leave
//! implicit: the **horizon** (what is predicted) and the **parameter
//! treatment** (how parameter uncertainty is handled). The second is the one
//! that bites — a free-forward band run at one plug-in parameter is
//! artificially narrow, and if it is read as a posterior band the gap is
//! mistaken for science. So both axes are typed and travel with every band,
//! and the band-building path only ever runs on a real posterior cloud.
//!
//! v1 ships exactly one cell — `FreeForward × Posterior`. Optimizer fits
//! (IF2 / NLopt), which produce no draws cloud, are refused with an actionable
//! message rather than silently plugged in.

use std::io::Write;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::posterior_draws;
use crate::run_meta::{FitAlgorithm, ObsSchema};

// ── The two axes, as types ─────────────────────────────────────────────────

/// What is predicted. v1 emits only `FreeForward`; `OneStepAhead` / `KStep…`
/// append as the folding lands (gh#269). A single-variant enum today, but it
/// makes the `horizon` artifact column type-safe and the axis explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Horizon {
    /// Run the fitted model forward from the start and see what it generates.
    FreeForward,
}

impl Horizon {
    /// The legible value written into the `horizon` artifact column.
    pub fn as_str(&self) -> &'static str {
        match self {
            Horizon::FreeForward => "free_forward",
        }
    }
}

/// Whether a band carries its full parameter uncertainty or pretends the
/// parameters are known. The safety-critical axis: the band-building path only
/// accepts [`ParamTreatment::Posterior`], which can only be constructed from a
/// real [`PosteriorDraws`] cloud — so a posterior band over a single point is
/// unrepresentable, and a plug-in band is always explicitly labelled.
#[derive(Debug, Clone)]
pub enum ParamTreatment {
    /// Average over the whole posterior cloud — the band carries full
    /// parameter uncertainty.
    Posterior(PosteriorDraws),
    /// A single best-guess parameter — a *narrower* band. v1 never builds one
    /// (it refuses point-estimate fits); the variant carries the method/stage
    /// the refusal names. The future plug-in cell resolves the point's
    /// parameter vector here.
    PlugIn {
        method: Option<FitAlgorithm>,
        stage: String,
    },
}

impl ParamTreatment {
    /// The legible value written into the `treatment` artifact column.
    pub fn label(&self) -> &'static str {
        match self {
            ParamTreatment::Posterior(_) => "posterior",
            ParamTreatment::PlugIn { .. } => "plug_in",
        }
    }
}

/// Every band carries its own convergence number, copied from the producing
/// stage's summary, so a band is never silent about whether its fit settled.
/// v1 records the number; it does not gate on it (the refusal policy is the
/// deferred guardrail).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceStatus {
    /// The stage reported a Gelman–Rubin R̂ / ESS summary.
    Reported { rhat_max: f64, ess_min: f64 },
    /// No per-stage R̂ available (e.g. a single-chain stage).
    NotAssessed,
}

impl ConvergenceStatus {
    /// The value written into the `rhat_max` column — the empty string when
    /// not assessed, so a consumer can tell "converged at 1.01" from "unknown".
    pub fn rhat_max_cell(&self) -> String {
        match self {
            ConvergenceStatus::Reported { rhat_max, .. } => format!("{rhat_max:.4}"),
            ConvergenceStatus::NotAssessed => String::new(),
        }
    }
}

// ── What a fit produced, resolved by artifact ──────────────────────────────

/// A resolved posterior draws cloud plus the provenance a band needs to label
/// itself honestly.
#[derive(Debug, Clone)]
pub struct PosteriorDraws {
    /// One complete parameter vector per draw (estimated + fixed columns).
    pub draws: Vec<IndexMap<String, f64>>,
    /// Bare stage name the cloud came from.
    pub stage: String,
    /// The inference algorithm, when recoverable.
    pub method: Option<FitAlgorithm>,
    /// The stage's convergence summary.
    pub convergence: ConvergenceStatus,
}

/// What a fit actually produced — resolved once, at the boundary, **by
/// artifact** (does the chosen stage have a draws cloud?), not by method name.
#[derive(Debug, Clone)]
pub enum FitResult {
    /// A Bayesian stage wrote a posterior draws cloud.
    Posterior(PosteriorDraws),
    /// An optimizer stage (IF2 / NLopt): one best point, no cloud. v1 cannot
    /// draw a band from it; carries the method/stage for the refusal message.
    PointEstimate {
        method: Option<FitAlgorithm>,
        stage: String,
    },
}

impl FitResult {
    /// Map a fit to how a predictive would treat its parameters. The total
    /// match is where the safety property falls out: a `Posterior` fit's draws
    /// flow into `Posterior` treatment; a `PointEstimate` can only ever be
    /// `PlugIn`.
    pub fn into_treatment(self) -> ParamTreatment {
        match self {
            FitResult::Posterior(pd) => ParamTreatment::Posterior(pd),
            FitResult::PointEstimate { method, stage } => {
                ParamTreatment::PlugIn { method, stage }
            }
        }
    }
}

// ── The output object ──────────────────────────────────────────────────────

/// A predictive object: its horizon, how it treated parameters, its
/// convergence, and the per-`(time, stratum)` quantile bands. None of the
/// three axes is optional — you cannot build a `Predictive` without stating
/// all of them, so none can be guessed downstream.
#[derive(Debug, Clone)]
pub struct Predictive {
    pub horizon: Horizon,
    pub treatment: ParamTreatment,
    pub convergence: ConvergenceStatus,
    pub streams: Vec<StreamBands>,
}

/// The quantile bands for one logical stream, faceted by its index dimensions.
#[derive(Debug, Clone)]
pub struct StreamBands {
    /// Logical stream name (the data-source key).
    pub source: String,
    /// The dimensions this stream is stratified over, the artifact's key
    /// columns (`[]` for a national series).
    pub index_dims: Vec<String>,
    /// One row per `(time, stratum)`.
    pub rows: Vec<BandRow>,
}

/// One predictive cell: a time, its stratum (dim → level), and the quantiles
/// of `y_rep` across draws at that cell.
#[derive(Debug, Clone)]
pub struct BandRow {
    pub time: f64,
    /// `(dim, level)` pairs, aligned to the stream's `index_dims`.
    pub stratum: Vec<(String, String)>,
    /// Quantile values, aligned to [`QUANTILE_LEVELS`].
    pub quantiles: Vec<f64>,
}

/// One observed series cell: a time, its stratum, the recorded value (`None`
/// for a hole — a scheduled-but-missing observation).
#[derive(Debug, Clone)]
pub struct ObservedRow {
    pub time: f64,
    pub stratum: Vec<(String, String)>,
    pub value: Option<f64>,
}

/// The default quantile levels and their column labels. A small fixed set —
/// `fill_between` wants columns, not a long-format `quantile` key.
pub const QUANTILE_LEVELS: &[(f64, &str)] =
    &[(0.05, "q05"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.95, "q95")];

// ── Pure numerics: the quantile reduction ──────────────────────────────────

/// Linear-interpolated quantile of `xs` at `q ∈ [0, 1]` (the numpy/`type-7`
/// rule). `xs` need not be sorted; a copy is sorted. Empty → `NaN`.
pub fn quantile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if v.len() == 1 {
        return v[0];
    }
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    v[lo] * (1.0 - frac) + v[hi] * frac
}

/// The full quantile band (`QUANTILE_LEVELS`) of one cell's draws.
pub fn band(xs: &[f64]) -> Vec<f64> {
    QUANTILE_LEVELS.iter().map(|(q, _)| quantile(xs, *q)).collect()
}

// ── Rendering the tidy artifact ────────────────────────────────────────────

/// Render `predictive/<stream>.tsv`: `time | <dims…> | horizon | treatment |
/// rhat_max | q05 … q95`. Tidy, plot-ready; the two axes are columns so a new
/// predictive cell is more rows, never new consumer code.
pub fn render_predictive_tsv(
    stream: &StreamBands,
    horizon: Horizon,
    treatment: &ParamTreatment,
    convergence: ConvergenceStatus,
) -> String {
    let mut out = String::new();
    // Header.
    out.push_str("time");
    for d in &stream.index_dims {
        out.push('\t');
        out.push_str(d);
    }
    out.push_str("\thorizon\ttreatment\trhat_max");
    for (_, label) in QUANTILE_LEVELS {
        out.push('\t');
        out.push_str(label);
    }
    out.push('\n');

    let rhat = convergence.rhat_max_cell();
    for row in &stream.rows {
        out.push_str(&fmt_time(row.time));
        for dim in &stream.index_dims {
            out.push('\t');
            out.push_str(level_for(&row.stratum, dim));
        }
        out.push('\t');
        out.push_str(horizon.as_str());
        out.push('\t');
        out.push_str(treatment.label());
        out.push('\t');
        out.push_str(&rhat);
        for q in &row.quantiles {
            out.push('\t');
            out.push_str(&fmt_value(*q));
        }
        out.push('\n');
    }
    out
}

/// Render `observed/<stream>.tsv`: `time | <dims…> | value`. The observed half
/// of the panel — a derived series in the same tidy keys as `predictive`, so a
/// panel renders from a join on `(time, <dims>)`. Holes render as an empty
/// `value` cell (scheduled but not observed), distinct from an observed zero.
pub fn render_observed_tsv(index_dims: &[String], rows: &[ObservedRow]) -> String {
    let mut out = String::new();
    out.push_str("time");
    for d in index_dims {
        out.push('\t');
        out.push_str(d);
    }
    out.push_str("\tvalue\n");
    for row in rows {
        out.push_str(&fmt_time(row.time));
        for dim in index_dims {
            out.push('\t');
            out.push_str(level_for(&row.stratum, dim));
        }
        out.push('\t');
        match row.value {
            Some(v) => out.push_str(&fmt_value(v)),
            None => {} // hole → empty cell
        }
        out.push('\n');
    }
    out
}

fn level_for<'a>(stratum: &'a [(String, String)], dim: &str) -> &'a str {
    stratum
        .iter()
        .find(|(d, _)| d == dim)
        .map(|(_, l)| l.as_str())
        .unwrap_or("")
}

/// Format a time: integral times as integers (`7`), else minimal decimal.
fn fmt_time(t: f64) -> String {
    if t.fract() == 0.0 && t.abs() < 1e15 {
        format!("{}", t as i64)
    } else {
        format!("{t}")
    }
}

/// Format a value: integral values as integers (count data reads cleanly),
/// else a decimal trimmed to ≤6 places (so an interpolated count quantile is
/// `204.9`, not `204.89999999999995`).
fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.6}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Write a tidy file, creating parent directories. Returns the path written.
pub fn write_tsv(dir: &Path, sub: &str, stream: &str, content: &str) -> Result<PathBuf, String> {
    let d = dir.join(sub);
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot create {}: {e}", d.display()))?;
    let path = d.join(format!("{stream}.tsv"));
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    f.write_all(content.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    Ok(path)
}

// ── Schema lookup: leaf → (logical stream, index dims) ─────────────────────

/// Look up a logical stream's index dimensions from the run schema, falling
/// back to the dims present on the leaf strata if the schema is absent.
pub fn index_dims_for(schema: Option<&ObsSchema>, source: &str, leaf_dims: &[String]) -> Vec<String> {
    schema
        .and_then(|s| s.streams.iter().find(|d| d.name == source))
        .map(|d| d.index_dims.clone())
        .unwrap_or_else(|| leaf_dims.to_vec())
}

// ── Resolving a fit into a FitResult, by artifact ──────────────────────────

impl FitResult {
    /// Resolve a fit results directory into a [`FitResult`] — by artifact.
    /// A stage that wrote a `draws.tsv` → [`FitResult::Posterior`]; an
    /// optimizer-only fit (no cloud) → [`FitResult::PointEstimate`]; nothing
    /// resolvable → the resolver's actionable error.
    pub fn resolve(segment: &Path, stage: Option<&str>) -> Result<FitResult, String> {
        let seg_str = segment.to_str().ok_or("fit path is not valid UTF-8")?;
        match posterior_draws::resolve_posterior_draws(seg_str, stage) {
            Ok(pref) => {
                let rows = crate::load_draws_tsv(&pref.draws_path.to_string_lossy())?;
                let draws: Vec<IndexMap<String, f64>> = rows
                    .into_iter()
                    .map(|m| m.into_iter().collect())
                    .collect();
                let stage_dir = pref
                    .draws_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| segment.to_path_buf());
                let convergence = read_convergence(&stage_dir, pref.method);
                Ok(FitResult::Posterior(PosteriorDraws {
                    draws,
                    stage: pref.stage,
                    method: pref.method,
                    convergence,
                }))
            }
            // No cloud: classify as a point-estimate fit if there are stages,
            // else surface the resolver's "not a fit / no posterior" error.
            Err(e) => {
                if let Some(view) = crate::fit::fit_view::FitView::read(segment) {
                    if let Some(terminal) = view.stages.last() {
                        return Ok(FitResult::PointEstimate {
                            method: Some(terminal.method),
                            stage: terminal.stage.clone(),
                        });
                    }
                }
                Err(e)
            }
        }
    }
}

/// Read a Bayesian stage's convergence summary (`pgas_summary.json` /
/// `pmmh_summary.json`): `max` over its R̂ map, `min` over its ESS map. Returns
/// [`ConvergenceStatus::NotAssessed`] when no summary or no R̂ is present (a
/// single-chain stage), so a band is never silently "converged".
fn read_convergence(stage_dir: &Path, method: Option<FitAlgorithm>) -> ConvergenceStatus {
    let file = match method {
        Some(FitAlgorithm::Pmmh) | Some(FitAlgorithm::Mh) => "pmmh_summary.json",
        _ => "pgas_summary.json",
    };
    let try_read = |name: &str| -> Option<ConvergenceStatus> {
        let bytes = std::fs::read(stage_dir.join(name)).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        let rhat_max = v.get("rhat")?.as_object()?.values()
            .filter_map(|x| x.as_f64())
            .fold(f64::NEG_INFINITY, f64::max);
        let ess_min = v.get("ess").and_then(|e| e.as_object()).map(|o| {
            o.values().filter_map(|x| x.as_f64()).fold(f64::INFINITY, f64::min)
        }).unwrap_or(f64::INFINITY);
        if rhat_max.is_finite() {
            Some(ConvergenceStatus::Reported { rhat_max, ess_min })
        } else {
            None
        }
    };
    // Try the method's summary, then the other (a stage dir we couldn't name).
    try_read(file)
        .or_else(|| try_read("pgas_summary.json"))
        .or_else(|| try_read("pmmh_summary.json"))
        .unwrap_or(ConvergenceStatus::NotAssessed)
}

// ── Resolving a config to its run segment ──────────────────────────────────

/// Resolve a fit reference to its run segment directory.
///
/// A directory is used as the segment directly. A `fit.toml` config is matched
/// to its run by `fit_toml_hash` (the sha256 the sidecar records) across the
/// run store — the proposal's "a config resolves to its run; error and list if
/// it maps to several" rule, without recomputing the CAS identity.
fn resolve_segment(fit_ref: &Path) -> Result<(PathBuf, crate::fit::config_v2::FitConfigV2), String> {
    use crate::fit::config_v2::FitConfigV2;
    if fit_ref.is_dir() {
        let cfg_path = fit_ref.join("fit.toml.original");
        let config = FitConfigV2::load(&cfg_path.to_string_lossy()).map_err(|e| {
            format!(
                "{} is a fit directory but its archived config could not be read: {e}\n  \
                 (expected {})",
                fit_ref.display(),
                cfg_path.display()
            )
        })?;
        return Ok((fit_ref.to_path_buf(), config));
    }

    // A config TOML: hash it, match against the run store's sidecars.
    let bytes = std::fs::read(fit_ref)
        .map_err(|e| format!("cannot read fit config {}: {e}", fit_ref.display()))?;
    let toml_hash = crate::hashing::sha256_hex(&bytes);
    let config = FitConfigV2::load(&fit_ref.to_string_lossy())
        .map_err(|e| format!("cannot load fit config {}: {e}", fit_ref.display()))?;
    let cas_root = crate::run_paths::output_root(None, config.output_dir.as_deref());
    let fits_dir = cas_root.join("fits");
    let mut matches: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&fits_dir) {
        for entry in entries.flatten() {
            let seg = entry.path();
            if !seg.is_dir() {
                continue;
            }
            if let Some(side) = crate::run_meta::read_fit_sidecar(&seg) {
                if side.fit_toml_hash == toml_hash {
                    matches.push(seg);
                }
            }
        }
    }
    matches.sort();
    match matches.len() {
        0 => Err(format!(
            "no completed fit found for {} under {}.\n  \
             Run `camdl fit run {}` first.",
            fit_ref.display(),
            fits_dir.display(),
            fit_ref.display()
        )),
        1 => Ok((matches.remove(0), config)),
        _ => {
            let list = matches.iter().map(|p| format!("    {}", p.display()))
                .collect::<Vec<_>>().join("\n");
            Err(format!(
                "{} resolves to {} runs:\n{}\n  \
                 Pass one of these run directories directly.",
                fit_ref.display(),
                matches.len(),
                list
            ))
        }
    }
}

// ── The engine sink: sample y_rep per draw at the observed cadence ─────────

/// A [`RunSink`] that samples `y_rep` for every fit leaf at the observed times,
/// for each draw (= cell), accumulating per `(leaf, time)` across draws. The
/// quantile reduction runs after all cells merge.
struct PredictiveSink {
    compiled: std::sync::Arc<sim::CompiledModel>,
    /// Per leaf (in `model.observations` order): the observation times to score.
    leaf_times: Vec<Vec<f64>>,
    /// `samples[leaf][time_idx]` = the `y_rep` values across draws.
    samples: Vec<Vec<Vec<f64>>>,
}

impl crate::engine::RunSink for PredictiveSink {
    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let model = &cell.model;
        // The draw's parameter vector — base defaults overlaid with this cell's
        // overrides. Load-bearing: the observation likelihood (e.g. a reporting
        // rate) reads the DRAW's parameters, not the model defaults, so the
        // posterior predictive carries observation-parameter uncertainty too.
        let mut params = self.compiled.default_params.clone();
        for (name, value) in &cell.spec.point_overrides {
            if let Some(&idx) = self.compiled.param_index.get(name.as_str()) {
                params[idx] = *value;
            }
        }
        let mut obs_rng = sim::rng::StatefulRng::new(cell.spec.obs_seed);
        for (si, obs_ir) in model.observations.iter().enumerate() {
            let times = &self.leaf_times[si];
            if times.is_empty() {
                continue;
            }
            let sampler = sim::inference::obs_model::compile_obs_sample_pf(
                obs_ir,
                self.compiled.clone(),
                &params,
            );
            let projected = crate::project_all_obs_times(&cell.traj, obs_ir, model, times);
            for (ti, &t) in times.iter().enumerate() {
                let snap = crate::snap_at(&cell.traj, t);
                let y = sampler(projected[ti], t, &snap.int_state.counts, &mut obs_rng);
                self.samples[si][ti].push(y);
            }
        }
        Ok(())
    }
}

// ── The verb ───────────────────────────────────────────────────────────────

/// `camdl fit predict` — write the free-forward posterior predictive artifact.
pub fn cmd_fit_predict(args: &crate::args::FitPredictArgs) {
    match run_predict(args) {
        Ok(paths) => {
            for p in paths {
                println!("wrote {}", p.display());
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// One fit leaf's loaded observed series: its logical stream, stratum, times,
/// and observed cells (`None` = a hole).
struct LeafObs {
    source: String,
    stratum: Vec<(String, String)>,
    times: Vec<f64>,
    observed: Vec<Option<f64>>,
}

fn run_predict(args: &crate::args::FitPredictArgs) -> Result<Vec<PathBuf>, String> {
    // 1. Resolve the run segment + its config.
    let (segment, config) = resolve_segment(args.fit()?)?;

    // 2. Resolve the posterior — by artifact. A point-estimate fit is refused.
    let fit_result = FitResult::resolve(&segment, args.stage.as_deref())?;
    let treatment = fit_result.into_treatment();
    let posterior = match treatment {
        ParamTreatment::Posterior(pd) => pd,
        ParamTreatment::PlugIn { method, stage } => return Err(plugin_refusal(method, &stage)),
    };

    // 3. Compile the model (same recipe the fit runner uses).
    let (compiled_ir, _ir_tmp) = crate::util::resolve_ir_path(&config.model.camdl)?;
    let (model, _) = crate::util::load_model(&compiled_ir)?;
    let dt = model.simulation.dt.unwrap_or(1.0);

    // 4. Load the observed data per leaf (the cadence + the observed half).
    let leaves = load_leaf_obs(&model, &config, dt, args.stream.as_deref())?;
    if leaves.is_empty() {
        return Err("no observation streams to predict — check that the model's \
                    observation sources are bound to data in the fit config, and that \
                    --stream (if given) names a real stream".into());
    }

    // 5. Drive the engine over the draws; sample y_rep at the observed times.
    // The standalone CompiledModel (used by the sink's observation sampler)
    // needs concrete parameter values to compile — an estimated parameter has
    // only a prior in the model. Seed it with the first draw (which carries
    // every parameter); the sink overrides per draw, so these values only
    // satisfy compilation. The engine compiles its own per-cell model from the
    // draw, independently.
    let mut model_for_obs = model.clone();
    if let Some(first) = posterior.draws.first() {
        for p in &mut model_for_obs.parameters {
            if let Some(&v) = first.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model_for_obs)
            .map_err(|e| format!("compiling model for prediction: {e:?}"))?,
    );
    // Leaf order = model.observations order; map the (possibly filtered) leaves
    // back onto that order for the sink.
    let leaf_times: Vec<Vec<f64>> = model
        .observations
        .iter()
        .map(|o| {
            leaves
                .iter()
                .find(|l| leaf_matches(o, l))
                .map(|l| l.times.clone())
                .unwrap_or_default()
        })
        .collect();
    let samples_init: Vec<Vec<Vec<f64>>> = leaf_times
        .iter()
        .map(|ts| vec![Vec::new(); ts.len()])
        .collect();
    let mut sink = PredictiveSink { compiled: compiled.clone(), leaf_times, samples: samples_init };

    let seed = args.seed.unwrap_or(1);
    let job = crate::sim_job::SimulateJob {
        model: compiled_ir.clone(),
        params_files: vec![],
        backend: posterior_backend(&model),
        dt,
        integrator: None,
        source: crate::sim_job::ParamSource::Draws {
            rows: posterior.draws.clone(),
            replicates: 1,
        },
        // A single no-op baseline scenario, matching `simulate`'s no-`--scenario`
        // path (an empty list would default to a named "baseline" preset lookup).
        scenarios: vec![crate::sim_job::ScenarioRef::Inline {
            name: "baseline".to_string(),
            enable: vec![],
            disable: vec![],
            params: IndexMap::new(),
        }],
        seeds: crate::sim_job::Seeds::Single(seed),
        cli_overrides: vec![],
        set_vec_entries: vec![],
        table_files: vec![],
        obs: crate::sim_job::ObsOutput::None,
        parallel: 1,
    };
    crate::engine::run_job(&job, &mut sink)?;

    // 6. Quantile-reduce and assemble the typed Predictive, then write.
    let schema = crate::run_meta::read_fit_sidecar(&segment).and_then(|s| s.schema);
    let predictive = assemble_predictive(
        &model,
        &sink,
        &leaves,
        Horizon::FreeForward,
        ParamTreatment::Posterior(posterior.clone()),
        posterior.convergence,
        schema.as_ref(),
    );

    let mut written = Vec::new();
    for stream in &predictive.streams {
        let pred_tsv = render_predictive_tsv(
            stream,
            predictive.horizon,
            &predictive.treatment,
            predictive.convergence,
        );
        written.push(write_tsv(&segment, "predictive", &stream.source, &pred_tsv)?);
    }
    // The observed half, grouped by logical stream.
    for (source, index_dims, rows) in observed_by_stream(&leaves, schema.as_ref()) {
        let obs_tsv = render_observed_tsv(&index_dims, &rows);
        written.push(write_tsv(&segment, "observed", &source, &obs_tsv)?);
    }
    let method_label = posterior.method.map(|m| m.as_str()).unwrap_or("posterior");
    eprintln!(
        "fit predict: horizon=free_forward treatment=posterior, {} stream(s), \
         {} draws from {} stage '{}'",
        predictive.streams.len(),
        posterior.draws.len(),
        method_label,
        posterior.stage,
    );
    Ok(written)
}

/// Whether a model observation leaf corresponds to a loaded `LeafObs` (matched
/// by logical source + stratum).
fn leaf_matches(o: &ir::observation::ObservationModel, l: &LeafObs) -> bool {
    o.source == l.source
        && o.stratum.len() == l.stratum.len()
        && o.stratum.iter().all(|sk| {
            l.stratum.iter().any(|(d, lvl)| *d == sk.dim && *lvl == sk.level)
        })
}

/// The forward backend for the predictive sim: the model's declared backend.
/// Chain-binomial is the inference default; the model's `simulation` block
/// names it.
fn posterior_backend(_model: &ir::Model) -> crate::args::types::ForwardBackend {
    // The fit ran chain_binomial unless the model is real-valued (ODE). v1
    // predicts on chain_binomial, matching the dominant inference backend;
    // an ODE-backed predictive is a follow-up.
    crate::args::types::ForwardBackend::ChainBinomial
}

/// Load each (filtered) observation leaf's observed series.
fn load_leaf_obs(
    model: &ir::Model,
    config: &crate::fit::config_v2::FitConfigV2,
    dt: f64,
    stream_filter: Option<&str>,
) -> Result<Vec<LeafObs>, String> {
    let data = config
        .data_spec()
        .map_err(|e| format!("this fit has no [data] block to read the observed series from: {e}"))?;
    let model_obs_names: Vec<String> = model.observations.iter().map(|o| o.name.clone()).collect();
    let effective = data
        .effective_observations(&model_obs_names)
        .map_err(|e| format!("resolving [data.observations]: {e}"))?;
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };

    let mut out = Vec::new();
    for obs_model in &model.observations {
        if !stream_selected(obs_model, stream_filter) {
            continue;
        }
        let Some(data_path) = effective.get(&obs_model.source) else {
            continue; // source not bound to data — skip (e.g. a fit-only diagnostic stream)
        };
        let siblings: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().filter(|o| o.source == obs_model.source).collect();
        let (obs, cells, _aux) =
            crate::fit::runner::load_observations(data_path, obs_model, &siblings, dt, &time_opts)?;
        let times: Vec<f64> = obs.iter().map(|o| o.time).collect();
        let observed: Vec<Option<f64>> = cells
            .iter()
            .map(|c| c.as_ref().map(|cell| match cell {
                sim::inference::ObsCell::Scalar(v) => *v,
            }))
            .collect();
        let stratum: Vec<(String, String)> =
            obs_model.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect();
        out.push(LeafObs { source: obs_model.source.clone(), stratum, times, observed });
    }
    Ok(out)
}

/// Whether a leaf passes the `--stream` filter — matches the logical source or
/// the expanded leaf name (the proposal's "accepts either name").
fn stream_selected(o: &ir::observation::ObservationModel, filter: Option<&str>) -> bool {
    match filter {
        None => true,
        Some(name) => o.source == name || o.name == name,
    }
}

/// Quantile-reduce the accumulated samples into the typed [`Predictive`],
/// grouping leaves by logical stream.
fn assemble_predictive(
    model: &ir::Model,
    sink: &PredictiveSink,
    leaves: &[LeafObs],
    horizon: Horizon,
    treatment: ParamTreatment,
    convergence: ConvergenceStatus,
    schema: Option<&ObsSchema>,
) -> Predictive {
    // Group leaf indices by logical source, preserving first-appearance order.
    let mut order: Vec<String> = Vec::new();
    let mut by_source: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (si, obs_ir) in model.observations.iter().enumerate() {
        // Only streams we actually loaded (filtered + bound).
        if !leaves.iter().any(|l| leaf_matches(obs_ir, l)) {
            continue;
        }
        by_source.entry(obs_ir.source.clone()).or_insert_with(|| {
            order.push(obs_ir.source.clone());
            Vec::new()
        }).push(si);
    }

    let mut streams = Vec::new();
    for source in &order {
        let leaf_idxs = &by_source[source];
        let leaf_dims: Vec<String> = model.observations[leaf_idxs[0]]
            .stratum.iter().map(|k| k.dim.clone()).collect();
        let index_dims = index_dims_for(schema, source, &leaf_dims);
        let mut rows = Vec::new();
        for &si in leaf_idxs {
            let stratum: Vec<(String, String)> = model.observations[si]
                .stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect();
            for (ti, draws_at_t) in sink.samples[si].iter().enumerate() {
                rows.push(BandRow {
                    time: sink.leaf_times[si][ti],
                    stratum: stratum.clone(),
                    quantiles: band(draws_at_t),
                });
            }
        }
        streams.push(StreamBands { source: source.clone(), index_dims, rows });
    }

    Predictive { horizon, treatment, convergence, streams }
}

/// Group the observed leaves into `(source, index_dims, rows)` for the observed
/// artifact, mirroring the predictive grouping.
fn observed_by_stream(
    leaves: &[LeafObs],
    schema: Option<&ObsSchema>,
) -> Vec<(String, Vec<String>, Vec<ObservedRow>)> {
    let mut order: Vec<String> = Vec::new();
    let mut by_source: IndexMap<String, Vec<ObservedRow>> = IndexMap::new();
    let mut dims_of: IndexMap<String, Vec<String>> = IndexMap::new();
    for leaf in leaves {
        let leaf_dims: Vec<String> = leaf.stratum.iter().map(|(d, _)| d.clone()).collect();
        let entry = by_source.entry(leaf.source.clone()).or_insert_with(|| {
            order.push(leaf.source.clone());
            dims_of.insert(leaf.source.clone(), index_dims_for(schema, &leaf.source, &leaf_dims));
            Vec::new()
        });
        for (ti, &time) in leaf.times.iter().enumerate() {
            entry.push(ObservedRow {
                time,
                stratum: leaf.stratum.clone(),
                value: leaf.observed[ti],
            });
        }
    }
    order
        .into_iter()
        .map(|s| {
            let dims = dims_of.get(&s).cloned().unwrap_or_default();
            let rows = by_source.get(&s).cloned().unwrap_or_default();
            (s, dims, rows)
        })
        .collect()
}

/// The actionable refusal for a point-estimate fit (the proposal's message).
fn plugin_refusal(method: Option<FitAlgorithm>, stage: &str) -> String {
    let m = method.map(|m| m.as_str()).unwrap_or("optimizer");
    format!(
        "stage '{stage}' is an optimizer fit ({m}) — it returns a single best-fit \
         parameter set, not a distribution, so there is no posterior band to draw.\n  \
         Get those parameters and run a plug-in forward simulation instead:\n    \
         camdl fit summary <run> --params-only > params.toml\n    \
         camdl simulate <model> --params params.toml --obs-only-dir out/\n  \
         (A labelled plug-in predictive is a future cell; v1 emits posterior bands only.)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_linear_interpolation_matches_numpy() {
        // numpy.quantile([0,1,2,3,4], q) with default 'linear'.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile(&xs, 0.0), 0.0);
        assert_eq!(quantile(&xs, 1.0), 4.0);
        assert_eq!(quantile(&xs, 0.5), 2.0);
        assert_eq!(quantile(&xs, 0.25), 1.0);
        assert!((quantile(&xs, 0.05) - 0.2).abs() < 1e-12);
        assert!((quantile(&xs, 0.95) - 3.8).abs() < 1e-12);
    }

    #[test]
    fn quantile_sorts_unsorted_input_and_handles_edges() {
        assert!(quantile(&[], 0.5).is_nan());
        assert_eq!(quantile(&[7.0], 0.5), 7.0, "single value is its own quantile");
        // Unsorted input gives the same answer as sorted.
        assert_eq!(quantile(&[4.0, 0.0, 2.0, 1.0, 3.0], 0.5), 2.0);
    }

    #[test]
    fn band_returns_all_five_levels_in_order() {
        let xs: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let b = band(&xs);
        assert_eq!(b.len(), 5);
        assert_eq!(b, vec![5.0, 25.0, 50.0, 75.0, 95.0]);
    }

    fn stratum(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(d, l)| (d.to_string(), l.to_string())).collect()
    }

    #[test]
    fn predictive_tsv_has_typed_axis_columns_and_one_row_per_cell() {
        let stream = StreamBands {
            source: "onset".into(),
            index_dims: vec!["patch".into()],
            rows: vec![
                BandRow { time: 7.0, stratum: stratum(&[("patch", "Bo")]),
                          quantiles: vec![0.0, 1.0, 3.0, 6.0, 12.0] },
                BandRow { time: 7.0, stratum: stratum(&[("patch", "Bombali")]),
                          quantiles: vec![0.0, 0.0, 1.0, 3.0, 7.0] },
            ],
        };
        let treatment = ParamTreatment::Posterior(PosteriorDraws {
            draws: vec![], stage: "pgas".into(), method: Some(FitAlgorithm::Pgas),
            convergence: ConvergenceStatus::Reported { rhat_max: 1.01, ess_min: 420.0 },
        });
        let tsv = render_predictive_tsv(
            &stream, Horizon::FreeForward, &treatment,
            ConvergenceStatus::Reported { rhat_max: 1.01, ess_min: 420.0 },
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0], "time\tpatch\thorizon\ttreatment\trhat_max\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "7\tBo\tfree_forward\tposterior\t1.0100\t0\t1\t3\t6\t12");
        assert_eq!(lines[2], "7\tBombali\tfree_forward\tposterior\t1.0100\t0\t0\t1\t3\t7");
        assert_eq!(lines.len(), 3, "header + one row per (time, stratum)");
    }

    #[test]
    fn predictive_tsv_national_series_has_no_dim_columns() {
        let stream = StreamBands {
            source: "cases".into(),
            index_dims: vec![],
            rows: vec![BandRow { time: 1.0, stratum: vec![], quantiles: vec![1.0, 2.0, 3.0, 4.0, 5.0] }],
        };
        let treatment = ParamTreatment::Posterior(PosteriorDraws {
            draws: vec![], stage: "pgas".into(), method: None,
            convergence: ConvergenceStatus::NotAssessed,
        });
        let tsv = render_predictive_tsv(&stream, Horizon::FreeForward, &treatment, ConvergenceStatus::NotAssessed);
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        // No dim column; not-assessed rhat is an empty cell, not a fabricated value.
        assert_eq!(lines[0], "time\thorizon\ttreatment\trhat_max\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "1\tfree_forward\tposterior\t\t1\t2\t3\t4\t5");
    }

    #[test]
    fn observed_tsv_renders_holes_as_empty_cells() {
        let rows = vec![
            ObservedRow { time: 1.0, stratum: stratum(&[("patch", "Bo")]), value: Some(3.0) },
            ObservedRow { time: 2.0, stratum: stratum(&[("patch", "Bo")]), value: None },
            ObservedRow { time: 1.0, stratum: stratum(&[("patch", "Bombali")]), value: Some(0.0) },
        ];
        let tsv = render_observed_tsv(&["patch".to_string()], &rows);
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0], "time\tpatch\tvalue");
        assert_eq!(lines[1], "1\tBo\t3");
        assert_eq!(lines[2], "2\tBo\t", "a hole is an empty value cell, not 0");
        assert_eq!(lines[3], "1\tBombali\t0", "an observed zero is a 0, distinct from a hole");
    }

    #[test]
    fn treatment_and_horizon_labels_are_legible() {
        assert_eq!(Horizon::FreeForward.as_str(), "free_forward");
        let post = ParamTreatment::Posterior(PosteriorDraws {
            draws: vec![], stage: "s".into(), method: None,
            convergence: ConvergenceStatus::NotAssessed,
        });
        assert_eq!(post.label(), "posterior");
        let plug = ParamTreatment::PlugIn { method: Some(FitAlgorithm::If2), stage: "scout".into() };
        assert_eq!(plug.label(), "plug_in");
    }

    #[test]
    fn point_estimate_fit_maps_to_plugin_treatment() {
        // The safety property, behaviourally: an optimizer fit can only ever
        // become a PlugIn treatment — never Posterior.
        let fit = FitResult::PointEstimate { method: Some(FitAlgorithm::If2), stage: "scout".into() };
        match fit.into_treatment() {
            ParamTreatment::PlugIn { method, stage } => {
                assert_eq!(method, Some(FitAlgorithm::If2));
                assert_eq!(stage, "scout");
            }
            ParamTreatment::Posterior(_) => panic!("a point estimate must not become a posterior band"),
        }
    }

    #[test]
    fn posterior_fit_maps_to_posterior_treatment() {
        let pd = PosteriorDraws {
            draws: vec![IndexMap::from([("beta".to_string(), 0.5)])],
            stage: "pgas".into(), method: Some(FitAlgorithm::Pgas),
            convergence: ConvergenceStatus::Reported { rhat_max: 1.02, ess_min: 300.0 },
        };
        match FitResult::Posterior(pd).into_treatment() {
            ParamTreatment::Posterior(d) => assert_eq!(d.draws.len(), 1),
            ParamTreatment::PlugIn { .. } => panic!("a posterior fit must keep its cloud"),
        }
    }

    #[test]
    fn index_dims_prefers_schema_then_leaf() {
        let schema = ObsSchema {
            dimensions: std::collections::BTreeMap::new(),
            streams: vec![crate::run_meta::StreamDescriptor {
                name: "onset".into(),
                index_dims: vec!["patch".into()],
                value_column: "onset".into(),
                value_kind: Some("count".into()),
                likelihood: "poisson".into(),
            }],
        };
        assert_eq!(index_dims_for(Some(&schema), "onset", &[]), vec!["patch".to_string()]);
        // Unknown stream falls back to the leaf dims.
        assert_eq!(index_dims_for(Some(&schema), "other", &["age".to_string()]), vec!["age".to_string()]);
        assert_eq!(index_dims_for(None, "onset", &["patch".to_string()]), vec!["patch".to_string()]);
    }
}
