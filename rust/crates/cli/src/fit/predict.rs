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

/// What is predicted. The `horizon` artifact column is this axis, typed so a
/// one-step band is never read as a free-forward one (or vice versa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Horizon {
    /// Run the fitted model forward from the start and see what it generates —
    /// `p(y_t | θ)`, never re-anchored to data. The generative check.
    FreeForward,
    /// `p(y_t | y_{1:t-1})` — the one-step-ahead posterior predictive: re-run a
    /// bootstrap filter over the data per posterior draw, sample `ỹ` from the
    /// propagated (pre-reweight) particles at each observation time, pool over
    /// (particles × draws). The honest short-horizon forecast object.
    OneStepAhead,
}

impl Horizon {
    /// The legible value written into the `horizon` artifact column.
    pub fn as_str(&self) -> &'static str {
        match self {
            Horizon::FreeForward => "free_forward",
            Horizon::OneStepAhead => "one_step",
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
    /// The label without the payload — what the output object carries, so the
    /// (potentially large) draw cloud never has to be cloned into a `Predictive`.
    pub fn kind(&self) -> TreatmentKind {
        match self {
            ParamTreatment::Posterior(_) => TreatmentKind::Posterior,
            ParamTreatment::PlugIn { .. } => TreatmentKind::PlugIn,
        }
    }
}

/// The treatment axis as a payload-free label — the value the artifact carries
/// and renders. Kept separate from [`ParamTreatment`] (which carries the cloud
/// during dispatch) so a built [`Predictive`] never holds the draw payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreatmentKind {
    Posterior,
    PlugIn,
}

impl TreatmentKind {
    /// The legible value written into the `treatment` artifact column.
    pub fn label(&self) -> &'static str {
        match self {
            TreatmentKind::Posterior => "posterior",
            TreatmentKind::PlugIn => "plug_in",
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

    /// The value written into the `ess_min` column — empty when not assessed or
    /// when no finite ESS was reported (a single-chain / ESS-less summary).
    pub fn ess_min_cell(&self) -> String {
        match self {
            ConvergenceStatus::Reported { ess_min, .. } if ess_min.is_finite() => {
                // ESS is an effective count; render it cleanly.
                if ess_min.fract() == 0.0 { format!("{}", *ess_min as i64) } else { format!("{ess_min:.1}") }
            }
            _ => String::new(),
        }
    }
}

// ── What a fit produced, resolved by artifact ──────────────────────────────

/// A resolved posterior draws cloud plus the provenance a band needs to label
/// itself honestly. The cloud is non-empty **by construction**: the only way to
/// build one is [`PosteriorDraws::new`], which rejects an empty draw set — so a
/// `ParamTreatment::Posterior` / `FitResult::Posterior` can never carry a
/// zero-draw cloud that would quantile to `NaN`.
#[derive(Debug, Clone)]
pub struct PosteriorDraws {
    /// One complete parameter vector per draw (estimated + fixed columns).
    /// Private: the non-empty invariant lives in [`PosteriorDraws::new`].
    draws: Vec<IndexMap<String, f64>>,
    /// Bare stage name the cloud came from.
    pub stage: String,
    /// The inference algorithm, when recoverable.
    pub method: Option<FitAlgorithm>,
    /// The forward simulator the producing stage ran on — the predictive
    /// replays on the SAME backend the fit used, not a hardcoded default.
    pub backend: crate::args::types::ForwardBackend,
    /// The stage's convergence summary.
    pub convergence: ConvergenceStatus,
}

impl PosteriorDraws {
    /// Build a posterior cloud, rejecting an empty draw set. This is the only
    /// constructor, so an empty "posterior" band is unrepresentable.
    pub fn new(
        draws: Vec<IndexMap<String, f64>>,
        stage: String,
        method: Option<FitAlgorithm>,
        backend: crate::args::types::ForwardBackend,
        convergence: ConvergenceStatus,
    ) -> Result<Self, String> {
        if draws.is_empty() {
            return Err(format!(
                "posterior stage '{stage}' has zero draws — cannot build a posterior \
                 predictive band (check that the stage's draws.tsv is non-empty and \
                 that burn-in did not discard every sweep)"
            ));
        }
        Ok(PosteriorDraws { draws, stage, method, backend, convergence })
    }

    /// The draw cloud (read-only).
    pub fn draws(&self) -> &[IndexMap<String, f64>] {
        &self.draws
    }

    /// Number of draws — always ≥ 1.
    pub fn n_draws(&self) -> usize {
        self.draws.len()
    }
}

// ── The filterable-fit witness (the one-step horizon's type gate) ───────────

/// Proof that a fit's draws can drive a particle filter — the only constructor
/// is [`FilterableFit::from_posterior`], which gates on the backend. The
/// one-step producer takes this BY VALUE/reference, so it is unreachable for a
/// non-filterable (ODE / Gillespie) fit: the gate is in the type, not a runtime
/// `if` the caller can forget. Free-forward takes a plain [`PosteriorDraws`] and
/// runs on any backend; one-step requires this witness.
pub struct FilterableFit {
    /// Private — the only way in is `from_posterior`, which proves the backend
    /// is filterable. [`FilterableFit::draws`] reads it back out.
    draws: PosteriorDraws,
}

/// Why a fit's posterior draws cannot drive a particle filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotFilterable {
    /// ODE: deterministic given θ, so `p(x_t | y_{1:t-1}, θ) = δ(x_t(θ))` and
    /// the one-step predictive reduces to the observation model at the
    /// deterministic state — identical to free-forward. A separate one-step band
    /// would be a relabelled duplicate.
    Deterministic,
    /// Gillespie: not an inference backend, so a fit never has Gillespie draws.
    /// Handled for exhaustiveness only.
    NotAnInferenceBackend,
}

impl FilterableFit {
    /// Construct the witness iff the fit ran on a filterable (chain-binomial)
    /// backend. ODE / Gillespie are typed out via [`NotFilterable`].
    pub fn from_posterior(d: PosteriorDraws) -> Result<Self, NotFilterable> {
        use crate::args::types::ForwardBackend;
        match d.backend {
            ForwardBackend::ChainBinomial => Ok(FilterableFit { draws: d }),
            ForwardBackend::Ode => Err(NotFilterable::Deterministic),
            ForwardBackend::Gillespie => Err(NotFilterable::NotAnInferenceBackend),
        }
    }

    /// The posterior cloud this witness proves is filterable.
    pub fn draws(&self) -> &PosteriorDraws {
        &self.draws
    }
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

/// The full quantile band (`QUANTILE_LEVELS`) of one cell's draws. Rejects a
/// non-finite sample (a `NaN`/±∞ `y_rep` is an upstream bug, not a band to
/// publish) so a quietly-wrong band can never reach the artifact.
pub fn band(xs: &[f64]) -> Result<Vec<f64>, String> {
    if xs.iter().any(|x| !x.is_finite()) {
        return Err(format!(
            "non-finite predictive sample ({} draws, {} non-finite) — refusing to \
             quantile a NaN/±∞ y_rep",
            xs.len(),
            xs.iter().filter(|x| !x.is_finite()).count(),
        ));
    }
    Ok(QUANTILE_LEVELS.iter().map(|(q, _)| quantile(xs, *q)).collect())
}

// ── Rendering the tidy artifact ────────────────────────────────────────────

/// One horizon's contribution to a `predictive/<stream>.tsv`: its rows, plus the
/// labels they carry (the horizon axis, the treatment, the convergence, the draw
/// count). Stacking several of these under one header is how a chain-binomial fit
/// writes both `free_forward` and `one_step` rows into the same file.
pub struct PredictiveSection<'a> {
    pub horizon: Horizon,
    pub treatment: TreatmentKind,
    pub convergence: ConvergenceStatus,
    pub n_draws: usize,
    pub rows: &'a [BandRow],
}

/// Render `predictive/<stream>.tsv`: `time | <dims…> | horizon | treatment |
/// rhat_max | ess_min | n_draws | q05 … q95`. Tidy, plot-ready; the axes and
/// the convergence channel are columns so a new predictive cell — a new horizon,
/// a new treatment — is more rows, never new consumer code. Several
/// [`PredictiveSection`]s (one per horizon) stack under the single header.
pub fn render_predictive_tsv_sections(
    index_dims: &[String],
    sections: &[PredictiveSection],
) -> String {
    let mut out = String::new();
    // Header.
    out.push_str("time");
    for d in index_dims {
        out.push('\t');
        out.push_str(d);
    }
    out.push_str("\thorizon\ttreatment\trhat_max\tess_min\tn_draws");
    for (_, label) in QUANTILE_LEVELS {
        out.push('\t');
        out.push_str(label);
    }
    out.push('\n');

    for section in sections {
        let rhat = section.convergence.rhat_max_cell();
        let ess = section.convergence.ess_min_cell();
        let n = section.n_draws.to_string();
        for row in section.rows {
            out.push_str(&fmt_time(row.time));
            for dim in index_dims {
                out.push('\t');
                out.push_str(level_for(&row.stratum, dim));
            }
            out.push('\t');
            out.push_str(section.horizon.as_str());
            out.push('\t');
            out.push_str(section.treatment.label());
            out.push('\t');
            out.push_str(&rhat);
            out.push('\t');
            out.push_str(&ess);
            out.push('\t');
            out.push_str(&n);
            for q in &row.quantiles {
                out.push('\t');
                out.push_str(&fmt_value(*q));
            }
            out.push('\n');
        }
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
pub(crate) fn fmt_time(t: f64) -> String {
    if t.fract() == 0.0 && t.abs() < 1e15 {
        format!("{}", t as i64)
    } else {
        format!("{t}")
    }
}

/// Format a value: integral values as integers (count data reads cleanly),
/// else a decimal trimmed to ≤6 places (so an interpolated count quantile is
/// `204.9`, not `204.89999999999995`).
pub(crate) fn fmt_value(v: f64) -> String {
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
                // Replay on the SAME backend the stage ran on; default only when
                // a bare stage dir was passed (no fit view to read it from).
                let backend = pref
                    .backend
                    .map(crate::args::types::ForwardBackend::from)
                    .unwrap_or(crate::args::types::ForwardBackend::ChainBinomial);
                Ok(FitResult::Posterior(PosteriorDraws::new(
                    draws,
                    pref.stage,
                    pref.method,
                    backend,
                    convergence,
                )?))
            }
            // No cloud: classify as a point-estimate fit. Report the stage the
            // user asked for (`--stage`), not the terminal one, so the refusal
            // names the right stage.
            Err(e) => {
                if let Some(view) = crate::fit::fit_view::FitView::read(segment) {
                    let chosen = match stage {
                        Some(want) => view.stages.iter().find(|s| s.stage == want),
                        None => view.stages.last(),
                    };
                    if let Some(s) = chosen {
                        return Ok(FitResult::PointEstimate {
                            method: Some(s.method),
                            stage: s.stage.clone(),
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
    /// The generated-quantities evaluator, `Some` iff the model declares a
    /// `quantities {}` block. Composed alongside the obs-sample accumulator (same
    /// draw, same params) — not a second [`RunSink`].
    quant_eval: Option<sim::quantity::QuantityEvaluator>,
    /// One inner `Vec` per draw (= cell): each quantity leaf's value, in
    /// `model.quantities` order. Retains derived values, never the trajectory.
    quant_draws: Vec<Vec<sim::quantity::QuantityResult>>,
    /// The trajectory snapshot times, captured once (every draw shares the output
    /// cadence) — the time axis a series quantity bands against.
    quant_times: Vec<f64>,
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
        // Capture the per-draw y_sim per stream for `observations.<stream>`
        // quantities — the SAME draws the predictive output uses, no redraw.
        // Skip the capture entirely when only state quantities are present.
        let want_obs = self
            .quant_eval
            .as_ref()
            .is_some_and(|e| e.references_observations());
        let mut obs_set =
            sim::quantity::ObsSeriesSet { streams: std::collections::HashMap::new() };
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
            let mut stream_vals: Vec<f64> =
                if want_obs { Vec::with_capacity(times.len()) } else { Vec::new() };
            for (ti, &t) in times.iter().enumerate() {
                let snap = crate::snap_at(&cell.traj, t);
                let y = sampler(projected[ti], t, &snap.int_state.counts, &mut obs_rng);
                self.samples[si][ti].push(y);
                if want_obs {
                    stream_vals.push(y);
                }
            }
            if want_obs {
                // Key by the stream's declared `name` — what `observations.<name>`
                // in the DSL references (v1.1 is unstratified, so name == base).
                obs_set.streams.insert(obs_ir.name.clone(), (times.clone(), stream_vals));
            }
        }

        // Generated quantities: fold this draw's trajectory + the just-drawn y_sim
        // into its per-quantity values, using the SAME resolved params + draws as
        // the predictive output above.
        if let Some(eval) = &self.quant_eval {
            let results = eval.eval_draw(&params, &cell.traj, &self.compiled, Some(&obs_set));
            if self.quant_times.is_empty() {
                self.quant_times = cell.traj.snapshots.iter().map(|s| s.t).collect();
            }
            self.quant_draws.push(results);
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
    // The label the artifact carries, derived from the treatment before we
    // unwrap the cloud (v1 only ever reaches `posterior` here, but the label is
    // read off the treatment, not hardcoded).
    let treatment_kind = treatment.kind();
    let posterior = match treatment {
        ParamTreatment::Posterior(pd) => pd,
        ParamTreatment::PlugIn { method, stage } => return Err(plugin_refusal(method, &stage)),
    };

    // 2b. Resolve which horizon(s) to emit. The one-step horizon is gated by a
    // backend witness ([`FilterableFit`]); an explicit `--horizon one_step` on a
    // non-filterable fit is a hard error, while the default ("all applicable")
    // silently skips it for those backends. `one_step_fit` is `Some` exactly when
    // the one-step producer will run.
    use crate::args::types::HorizonArg;
    let want_free_forward =
        matches!(args.horizon, None | Some(HorizonArg::FreeForward));
    let one_step_fit: Option<FilterableFit> = match args.horizon {
        Some(HorizonArg::FreeForward) => None,
        Some(HorizonArg::OneStep) => {
            // EXPLICIT request: a non-filterable fit is a hard error with a redirect.
            Some(FilterableFit::from_posterior(posterior.clone()).map_err(|why| {
                one_step_refusal(why, posterior.method, &posterior.stage)
            })?)
        }
        None => {
            // DEFAULT "all applicable": run one-step where the backend supports
            // it; for ODE / Gillespie it is simply not applicable (no error).
            FilterableFit::from_posterior(posterior.clone()).ok()
        }
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

    // 5. Validate the draw schema BEFORE simulating: every draw column must be a
    // real model parameter (an unknown name is a stale/mis-keyed draw, never
    // silently dropped), and every model parameter must be covered (so none is
    // silently defaulted). The cloud is non-empty by construction, so [0] is safe.
    {
        use std::collections::HashSet;
        let model_params: HashSet<&str> =
            model.parameters.iter().map(|p| p.name.as_str()).collect();
        let draw_keys: HashSet<&str> =
            posterior.draws()[0].keys().map(|s| s.as_str()).collect();
        let mut unknown: Vec<&str> =
            draw_keys.iter().filter(|k| !model_params.contains(*k)).copied().collect();
        let mut missing: Vec<&str> =
            model_params.iter().filter(|p| !draw_keys.contains(*p)).copied().collect();
        unknown.sort();
        missing.sort();
        if !unknown.is_empty() {
            return Err(format!(
                "posterior draws contain parameter(s) the model does not declare: {} \
                 (the draws were produced against a different model)",
                unknown.join(", ")
            ));
        }
        if !missing.is_empty() {
            return Err(format!(
                "posterior draws do not cover model parameter(s): {} \
                 (every parameter must be present so none is silently defaulted)",
                missing.join(", ")
            ));
        }
    }

    // The standalone CompiledModel (used by the free-forward sink's observation
    // sampler AND the one-step filter) needs concrete parameter values to compile
    // — an estimated parameter has only a prior in the model. Seed it with the
    // first draw (which carries every parameter); each producer overrides per
    // draw, so these values only satisfy compilation.
    let mut model_for_obs = model.clone();
    if let Some(first) = posterior.draws().first() {
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

    let schema = crate::run_meta::read_fit_sidecar(&segment).and_then(|s| s.schema);
    let seed = args.seed.unwrap_or(1);

    // Build the generated-quantities evaluator once (the IR is fixed across draws);
    // `None` when the model declares no `quantities {}` block. The evaluator drives
    // the free-forward sink — the always-fresh path that has a trajectory in hand.
    let quant_eval: Option<sim::quantity::QuantityEvaluator> = if !model.quantities.is_empty() {
        Some(
            sim::quantity::QuantityEvaluator::new(&model.quantities, compiled.as_ref())
                .map_err(|e| format!("building quantity evaluator: {e}"))?,
        )
    } else {
        None
    };
    // The rendered quantity sidecars + manifest, filled after the free-forward pass.
    let mut quantity_outputs: Vec<(String, String)> = Vec::new();
    let mut quantity_manifest: Option<String> = None;

    // ── Free-forward horizon: drive the engine over the draws; sample y_rep at
    // the observed times. Run only when free-forward is requested.
    let free_forward: Option<Vec<StreamBands>> = if want_free_forward {
        // Leaf order = model.observations order; map the (possibly filtered)
        // leaves back onto that order for the sink.
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
        let mut sink = PredictiveSink {
            compiled: compiled.clone(),
            leaf_times,
            samples: samples_init,
            quant_eval,
            quant_draws: Vec::new(),
            quant_times: Vec::new(),
        };

        let job = crate::sim_job::SimulateJob {
            model: compiled_ir.clone(),
            params_files: vec![],
            // Replay on the SAME forward simulator the fit used (chain_binomial /
            // ode), resolved from the stage — never a hardcoded default.
            backend: posterior.backend,
            dt,
            integrator: None,
            source: crate::sim_job::ParamSource::Draws {
                rows: posterior.draws().to_vec(),
                replicates: 1,
            },
            // A single no-op baseline scenario, matching `simulate`'s
            // no-`--scenario` path (an empty list would default to a named
            // "baseline" preset lookup).
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

        // Band the accumulated per-draw quantity values into sidecars + a manifest.
        if !model.quantities.is_empty() {
            let (outs, manifest) = crate::quantity_output::render_quantities(
                &model.quantities,
                &sink.quant_draws,
                &sink.quant_times,
                crate::quantity_output::Mode::Banded,
            )?;
            quantity_outputs = outs;
            quantity_manifest = Some(manifest);
        }

        Some(assemble_predictive(&model, &sink, &leaves, schema.as_ref())?)
    } else {
        None
    };

    // ── One-step horizon: per-draw bootstrap filter over the data, pooled. Runs
    // only when the witness was built (a filterable fit and the horizon wanted).
    let n_draws_cap = args.n_draws.unwrap_or(DEFAULT_ONE_STEP_DRAWS);
    let one_step: Option<(Vec<StreamBands>, usize)> = match &one_step_fit {
        Some(fit) => Some(one_step_bands(
            compiled.clone(),
            &model,
            &config,
            dt,
            args.stream.as_deref(),
            fit,
            n_draws_cap,
            seed,
            schema.as_ref(),
        )?),
        None => None,
    };

    // 6. Write the predictive artifact — both horizons stacked into one file per
    // logical stream (the typed `horizon` column distinguishes the rows). The
    // observed half follows.
    let mut written = Vec::new();
    let one_step_streams: &[StreamBands] = one_step.as_ref().map(|(s, _)| s.as_slice()).unwrap_or(&[]);
    let one_step_n = one_step.as_ref().map(|(_, n)| *n).unwrap_or(0);

    // Union of source names across both horizons, preserving free-forward order
    // first, then any one-step-only sources.
    let mut sources: Vec<String> = Vec::new();
    if let Some(ff) = &free_forward {
        for s in ff {
            if !sources.contains(&s.source) {
                sources.push(s.source.clone());
            }
        }
    }
    for s in one_step_streams {
        if !sources.contains(&s.source) {
            sources.push(s.source.clone());
        }
    }

    for source in &sources {
        let ff_stream = free_forward.as_ref().and_then(|ff| ff.iter().find(|s| &s.source == source));
        let os_stream = one_step_streams.iter().find(|s| &s.source == source);
        // index_dims is the same across horizons (same schema/leaf); take it from
        // whichever section is present.
        let index_dims = ff_stream
            .map(|s| s.index_dims.clone())
            .or_else(|| os_stream.map(|s| s.index_dims.clone()))
            .unwrap_or_default();

        let mut sections: Vec<PredictiveSection> = Vec::new();
        if let Some(s) = ff_stream {
            sections.push(PredictiveSection {
                horizon: Horizon::FreeForward,
                treatment: treatment_kind,
                convergence: posterior.convergence,
                n_draws: posterior.n_draws(),
                rows: &s.rows,
            });
        }
        if let Some(s) = os_stream {
            sections.push(PredictiveSection {
                horizon: Horizon::OneStepAhead,
                treatment: treatment_kind,
                convergence: posterior.convergence,
                n_draws: one_step_n,
                rows: &s.rows,
            });
        }
        let pred_tsv = render_predictive_tsv_sections(&index_dims, &sections);
        written.push(write_tsv(&segment, "predictive", source, &pred_tsv)?);
    }
    // The observed half, grouped by logical stream.
    for (source, index_dims, rows) in observed_by_stream(&leaves, schema.as_ref()) {
        let obs_tsv = render_observed_tsv(&index_dims, &rows);
        written.push(write_tsv(&segment, "observed", &source, &obs_tsv)?);
    }
    // Generated quantities: one sidecar TSV per logical quantity + a manifest.
    // These are NOT in the run_id-keyed CAS leaf — a regenerated sidecar beside
    // `predictive/`/`observed/`, overwritten in place.
    for (name, content) in &quantity_outputs {
        written.push(write_tsv(&segment, "quantities", name, content)?);
    }
    if let Some(manifest) = &quantity_manifest {
        let path = segment.join("quantities.json");
        std::fs::write(&path, manifest)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        written.push(path);
    }
    let method_label = posterior.method.map(|m| m.as_str()).unwrap_or("posterior");
    let mut horizons: Vec<String> = Vec::new();
    if free_forward.is_some() {
        horizons.push("free_forward".to_string());
    }
    if one_step.is_some() {
        horizons.push(format!("one_step({one_step_n} draws)"));
    }
    eprintln!(
        "fit predict: horizon={} treatment=posterior, {} stream(s), \
         {} draws from {} stage '{}'",
        horizons.join("+"),
        sources.len(),
        posterior.n_draws(),
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

/// Quantile-reduce the accumulated free-forward samples into per-stream bands,
/// grouping leaves by logical stream. The horizon/treatment/convergence/n_draws
/// labels are applied at render time (each [`PredictiveSection`] carries them),
/// so this returns only the bands.
fn assemble_predictive(
    model: &ir::Model,
    sink: &PredictiveSink,
    leaves: &[LeafObs],
    schema: Option<&ObsSchema>,
) -> Result<Vec<StreamBands>, String> {
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
                let quantiles = band(draws_at_t).map_err(|e| {
                    format!("stream '{source}' at t={}: {e}", sink.leaf_times[si][ti])
                })?;
                rows.push(BandRow {
                    time: sink.leaf_times[si][ti],
                    stratum: stratum.clone(),
                    quantiles,
                });
            }
        }
        streams.push(StreamBands { source: source.clone(), index_dims, rows });
    }

    Ok(streams)
}

// ── The one-step-ahead posterior predictive producer ───────────────────────

/// Default posterior-cloud subsample for the one-step horizon. The band pools
/// `draws × n_particles` samples per cell, so a few hundred draws saturate
/// q05…q95; the full fit cloud at fit-grade N is never run silently.
const DEFAULT_ONE_STEP_DRAWS: usize = 200;

/// Particle count for the one-step prediction filter. It need not match the
/// fit's N — the band pools across draws too, and every particle's `ỹ` is
/// kept (the recorder retains them), so a modest N yields a dense band cheaply.
const ONE_STEP_N_PARTICLES: usize = 500;

/// Build the one-step-ahead posterior predictive bands: for each (subsampled)
/// posterior draw θ, run a bootstrap filter over the data with
/// `record_prequential = true`, capturing the per-particle one-step predictive
/// samples `ỹ ∼ p(y | x_t, θ)` at each observation time (the particles are
/// distributed as `p(x_t | y_{1:t-1}, θ)` at that point). Pool over
/// (particles × draws) per `(stream-leaf, time)`, quantile, and group by logical
/// source + stratum exactly like the free-forward path. Horizon = one_step.
///
/// `n_draws_used` (out) is the subsample count actually filtered, for the
/// `n_draws` artifact column.
fn one_step_bands(
    compiled: std::sync::Arc<sim::CompiledModel>,
    model: &ir::Model,
    config: &crate::fit::config_v2::FitConfigV2,
    dt: f64,
    stream_filter: Option<&str>,
    fit: &FilterableFit,
    n_draws_cap: usize,
    base_seed: u64,
    schema: Option<&ObsSchema>,
) -> Result<(Vec<StreamBands>, usize), String> {
    use sim::inference::{
        bootstrap_filter, multi_stream_obs::StreamProjection, multi_stream_obs::StreamSpec,
        traits::SMCConfig, BoundObs, ChainBinomialProcess, MultiStreamObsModel,
    };

    // ── Build the filter obs model ONCE. This mirrors `pfilter.rs:386-421` and
    // `fit::runner::build_obs_model`: it is the THIRD copy of this obs-model
    // assembly (pfilter.rs, runner, here). Consolidating the three onto a shared
    // builder is a follow-up — NOT done here (the two existing copies are left
    // untouched). Each bound leaf reuses `runner::load_observations` for its
    // cells + cadence, then `StreamProjection::from_ir` + `bind` + the
    // multi-stream model, the same way the fit's pfilter stage does.
    let data = config.data_spec().map_err(|e| {
        format!("this fit has no [data] block to read the observed series from: {e}")
    })?;
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

    // Bound leaves, in `model.observations` order, so `stream_idx` (the index
    // into the obs model's `stream_names()`) maps back to its IR leaf here.
    let mut bound_leaves: Vec<&ir::observation::ObservationModel> = Vec::new();
    let mut specs: Vec<StreamSpec> = Vec::new();
    for obs_model in &model.observations {
        if !stream_selected(obs_model, stream_filter) {
            continue;
        }
        let Some(data_path) = effective.get(&obs_model.source) else {
            continue; // source not bound to data — skip (a fit-only diagnostic stream)
        };
        let siblings: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().filter(|o| o.source == obs_model.source).collect();
        let (obs, cells, aux) =
            crate::fit::runner::load_observations(data_path, obs_model, &siblings, dt, &time_opts)?;
        let projection = StreamProjection::from_ir(&obs_model.projection, &compiled, &obs_model.name)?;
        let times: Vec<f64> = obs.iter().map(|o| o.time).collect();
        specs.push(StreamSpec {
            projection,
            ir_model: obs_model.clone(),
            observations: cells,
            obs_times: times,
            aux,
        });
        bound_leaves.push(obs_model);
    }
    if specs.is_empty() {
        return Err("no observation streams to predict — check that the model's \
                    observation sources are bound to data in the fit config, and that \
                    --stream (if given) names a real stream"
            .into());
    }

    let (bound, _report) = BoundObs::bind(specs)
        .map_err(|report| format!("observation data invalid:\n{}", report.render()))?;
    let obs_model = MultiStreamObsModel::new(bound, compiled.clone())
        .map_err(|e| format!("observation model construction failed: {e:?}"))?;

    // ── Build the process ONCE.
    let process = ChainBinomialProcess::new(compiled.clone());

    // ── Subsample the posterior cloud (never silently run the full cloud).
    let draws = fit.draws().draws();
    let total = draws.len();
    let n_used = n_draws_cap.min(total).max(1);
    let chosen: Vec<&IndexMap<String, f64>> = if n_used >= total {
        draws.iter().collect()
    } else {
        // Evenly-spaced subsample across the cloud.
        (0..n_used)
            .map(|i| {
                let idx = (i * total) / n_used;
                &draws[idx]
            })
            .collect()
    };
    if n_used < total {
        eprintln!(
            "fit predict: one_step horizon — subsampling {n_used} of {total} posterior \
             draws (raise with --n-draws)"
        );
    }

    let smc_config = SMCConfig {
        n_particles: ONE_STEP_N_PARTICLES,
        dt,
        t_start: compiled.model.simulation.t_start,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: true,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    // ── Pool per (stream_idx, obs_idx): all particle samples across all draws.
    // `pooled[stream_idx][obs_idx]` accumulates ỹ over (particles × draws); the
    // stream/time axes come from the FIRST result (identical across draws — same
    // obs model). NaN entries (a not-scheduled stream at a union time) are
    // dropped, mirroring the prequential capture's filter.
    let mut pooled: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut stream_names: Vec<String> = Vec::new();
    let mut obs_times: Vec<f64> = Vec::new();

    for (draw_idx, draw) in chosen.iter().enumerate() {
        // The draw's parameter vector — base defaults overlaid by name (the
        // survey.rs:722-734 idiom). The cloud is schema-validated upstream, so
        // every model parameter is present.
        let mut params = compiled.default_params.clone();
        for (name, &value) in draw.iter() {
            if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                params[idx] = value;
            }
        }
        // Distinct, reproducible per-draw seed: mix the draw index into the base
        // seed so each filter pass has its own RNG stream and the whole run is
        // deterministic given `base_seed`.
        let seed = base_seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(draw_idx as u64 + 1));
        let result = bootstrap_filter(&process, &obs_model, &params, &smc_config, seed)
            .map_err(|e| format!("one-step filter failed on draw {draw_idx}: {e:?}"))?;
        let preq = result.prequential.ok_or_else(|| {
            "one-step filter did not record prequential samples (record_prequential was \
             requested but the result is empty — internal error)"
                .to_string()
        })?;

        if pooled.is_empty() {
            stream_names = preq.stream_names.clone();
            obs_times = preq.obs_times.clone();
            pooled = vec![vec![Vec::new(); obs_times.len()]; stream_names.len()];
        }

        // `per_stream_samples[obs_idx][stream_idx][particle]`.
        for (obs_idx, per_stream) in preq.per_stream_samples.iter().enumerate() {
            for (stream_idx, particles) in per_stream.iter().enumerate() {
                for &y in particles {
                    if y.is_finite() {
                        pooled[stream_idx][obs_idx].push(y);
                    }
                }
            }
        }
    }

    // ── Group leaves by logical source (first-appearance order), exactly like
    // `assemble_predictive`, and build one_step BandRows. `stream_names[si]` is
    // the leaf `obs.name`; map it back to the bound leaf for its source/stratum.
    let leaf_of_name: std::collections::HashMap<&str, &ir::observation::ObservationModel> =
        bound_leaves.iter().map(|o| (o.name.as_str(), *o)).collect();

    let mut order: Vec<String> = Vec::new();
    let mut by_source: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (si, name) in stream_names.iter().enumerate() {
        let leaf = leaf_of_name.get(name.as_str()).ok_or_else(|| {
            format!("one-step: filter stream '{name}' has no matching bound leaf (internal error)")
        })?;
        by_source.entry(leaf.source.clone()).or_insert_with(|| {
            order.push(leaf.source.clone());
            Vec::new()
        }).push(si);
    }

    let mut streams = Vec::new();
    for source in &order {
        let leaf_idxs = &by_source[source];
        let first_leaf = leaf_of_name[stream_names[leaf_idxs[0]].as_str()];
        let leaf_dims: Vec<String> = first_leaf.stratum.iter().map(|k| k.dim.clone()).collect();
        let index_dims = index_dims_for(schema, source, &leaf_dims);
        let mut rows = Vec::new();
        for &si in leaf_idxs {
            let leaf = leaf_of_name[stream_names[si].as_str()];
            let stratum: Vec<(String, String)> =
                leaf.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect();
            for (ti, &t) in obs_times.iter().enumerate() {
                let cell = &pooled[si][ti];
                if cell.is_empty() {
                    // This stream is not scheduled at this union time (all NaN,
                    // dropped) — emit no row for it (multi-cadence).
                    continue;
                }
                let quantiles = band(cell)
                    .map_err(|e| format!("stream '{source}' (one_step) at t={t}: {e}"))?;
                rows.push(BandRow { time: t, stratum: stratum.clone(), quantiles });
            }
        }
        streams.push(StreamBands { source: source.clone(), index_dims, rows });
    }

    Ok((streams, n_used))
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

/// The actionable refusal for an explicit `--horizon one_step` on a fit whose
/// backend cannot drive a particle filter. ODE's one-step predictive is
/// identical to free-forward (deterministic given θ), so the redirect points the
/// user there rather than emit a relabelled-identical band.
fn one_step_refusal(why: NotFilterable, method: Option<FitAlgorithm>, stage: &str) -> String {
    let m = method.map(|m| m.as_str()).unwrap_or("posterior");
    match why {
        NotFilterable::Deterministic => format!(
            "stage '{stage}' ({m}) ran on the ODE backend, which is deterministic given \
             the parameters — its one-step-ahead predictive p(y_t | y_{{1:t-1}}) reduces \
             to the observation model at the deterministic state, identical to the \
             free-forward band. There is no separate one-step object to emit.\n  \
             Use the free-forward horizon instead:\n    \
             camdl fit predict --fit <run> --horizon free_forward"
        ),
        NotFilterable::NotAnInferenceBackend => format!(
            "stage '{stage}' ({m}) ran on the Gillespie backend, which is not an \
             inference backend — a fit never produces Gillespie posterior draws, so a \
             one-step-ahead predictive is not defined. This should not occur; please \
             report it."
        ),
    }
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
        let b = band(&xs).unwrap();
        assert_eq!(b.len(), 5);
        assert_eq!(b, vec![5.0, 25.0, 50.0, 75.0, 95.0]);
    }

    #[test]
    fn band_rejects_non_finite_samples() {
        // A NaN/±∞ y_rep is an upstream bug, not a band to publish.
        assert!(band(&[1.0, f64::NAN, 3.0]).is_err());
        assert!(band(&[1.0, f64::INFINITY]).is_err());
        assert!(band(&[1.0, 2.0, 3.0]).is_ok());
    }

    fn cb() -> crate::args::types::ForwardBackend {
        crate::args::types::ForwardBackend::ChainBinomial
    }

    fn stratum(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(d, l)| (d.to_string(), l.to_string())).collect()
    }

    /// A one-draw posterior cloud on the given backend, for the witness tests.
    fn posterior_on(backend: crate::args::types::ForwardBackend) -> PosteriorDraws {
        PosteriorDraws::new(
            vec![IndexMap::from([("beta".to_string(), 0.5)])],
            "pgas".into(),
            Some(FitAlgorithm::Pgas),
            backend,
            ConvergenceStatus::NotAssessed,
        )
        .unwrap()
    }

    #[test]
    fn one_step_horizon_label_is_legible() {
        assert_eq!(Horizon::OneStepAhead.as_str(), "one_step");
        assert_eq!(Horizon::FreeForward.as_str(), "free_forward");
    }

    #[test]
    fn filterable_fit_gates_on_backend() {
        use crate::args::types::ForwardBackend;

        // chain_binomial → the witness constructs (one-step runs).
        let cb_fit = FilterableFit::from_posterior(posterior_on(ForwardBackend::ChainBinomial));
        assert!(cb_fit.is_ok(), "a chain-binomial fit is filterable");
        // The witness carries the cloud back out.
        assert_eq!(cb_fit.unwrap().draws().n_draws(), 1);

        // ODE → Deterministic (one-step ≡ free-forward; redirect, not a band).
        assert_eq!(
            FilterableFit::from_posterior(posterior_on(ForwardBackend::Ode)).err(),
            Some(NotFilterable::Deterministic),
            "an ODE fit is not filterable: its one-step is the free-forward band"
        );

        // Gillespie → NotAnInferenceBackend (exhaustiveness).
        assert_eq!(
            FilterableFit::from_posterior(posterior_on(ForwardBackend::Gillespie)).err(),
            Some(NotFilterable::NotAnInferenceBackend),
            "Gillespie is not an inference backend"
        );
    }

    #[test]
    fn one_step_refusal_for_ode_redirects_to_free_forward() {
        let msg = one_step_refusal(NotFilterable::Deterministic, Some(FitAlgorithm::Mh), "posterior");
        assert!(msg.contains("ODE"), "names the backend: {msg}");
        assert!(msg.contains("--horizon free_forward"), "redirects to free-forward: {msg}");
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
        let tsv = render_predictive_tsv_sections(
            &stream.index_dims,
            &[PredictiveSection {
                horizon: Horizon::FreeForward,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::Reported { rhat_max: 1.01, ess_min: 420.0 },
                n_draws: 40,
                rows: &stream.rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0],
            "time\tpatch\thorizon\ttreatment\trhat_max\tess_min\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "7\tBo\tfree_forward\tposterior\t1.0100\t420\t40\t0\t1\t3\t6\t12");
        assert_eq!(lines[2], "7\tBombali\tfree_forward\tposterior\t1.0100\t420\t40\t0\t0\t1\t3\t7");
        assert_eq!(lines.len(), 3, "header + one row per (time, stratum)");
    }

    #[test]
    fn predictive_tsv_national_series_unassessed_convergence_is_empty() {
        let stream = StreamBands {
            source: "cases".into(),
            index_dims: vec![],
            rows: vec![BandRow { time: 1.0, stratum: vec![], quantiles: vec![1.0, 2.0, 3.0, 4.0, 5.0] }],
        };
        let tsv = render_predictive_tsv_sections(
            &stream.index_dims,
            &[PredictiveSection {
                horizon: Horizon::FreeForward,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::NotAssessed,
                n_draws: 12,
                rows: &stream.rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        // No dim column; not-assessed rhat/ess are empty cells, not fabricated
        // values; n_draws is still carried.
        assert_eq!(lines[0], "time\thorizon\ttreatment\trhat_max\tess_min\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "1\tfree_forward\tposterior\t\t\t12\t1\t2\t3\t4\t5");
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
        assert_eq!(TreatmentKind::Posterior.label(), "posterior");
        assert_eq!(TreatmentKind::PlugIn.label(), "plug_in");
        let plug = ParamTreatment::PlugIn { method: Some(FitAlgorithm::If2), stage: "scout".into() };
        assert_eq!(plug.kind(), TreatmentKind::PlugIn);
    }

    #[test]
    fn posterior_draws_new_rejects_empty_cloud() {
        // #2/#3: an empty posterior cloud is unrepresentable — the only
        // constructor refuses it, so no NaN band can be built.
        let r = PosteriorDraws::new(
            vec![], "pgas".into(), Some(FitAlgorithm::Pgas), cb(),
            ConvergenceStatus::NotAssessed,
        );
        assert!(r.is_err(), "empty cloud must be rejected at construction");
        assert!(r.unwrap_err().contains("zero draws"));
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
        let pd = PosteriorDraws::new(
            vec![IndexMap::from([("beta".to_string(), 0.5)])],
            "pgas".into(), Some(FitAlgorithm::Pgas), cb(),
            ConvergenceStatus::Reported { rhat_max: 1.02, ess_min: 300.0 },
        ).unwrap();
        match FitResult::Posterior(pd).into_treatment() {
            ParamTreatment::Posterior(d) => assert_eq!(d.n_draws(), 1),
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
