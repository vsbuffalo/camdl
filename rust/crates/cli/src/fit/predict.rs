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

use crate::chain_selection::{warn_active_selection, ChainSelection, SubsetInfo};
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
///
/// This IS the sealed-fit proposal's Phase 4 "can't-drop-uncertainty"
/// invariant, enforced here by illegal-states-unrepresentable typing rather
/// than a runtime check. [`PosteriorDraws`] is that proposal's `Ensemble` minus
/// two additive fields — per-parameter provenance and the latent trajectory —
/// which the keyed-joint `(θ, X)` output supplies (see
/// `docs/dev/proposals/2026-06-28-keyed-joint-param-trajectory-output.md`).
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
    /// The read-side chain selection that produced this cloud, when one was
    /// active (`--exclude-chains`). `None` for a full-cloud posterior. Carried
    /// so the predictive artifact can stamp its provenance (`predictive.json`
    /// `chain_selection`) — a chain-subset band is never mistakable for a
    /// full-cloud one.
    pub selection: Option<SubsetInfo>,
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
        Ok(PosteriorDraws { draws, stage, method, backend, convergence, selection: None })
    }

    /// Attach the chain-selection provenance (`--exclude-chains`) to the cloud.
    pub fn with_selection_info(mut self, info: Option<SubsetInfo>) -> Self {
        self.selection = info;
        self
    }

    /// The chain-selection provenance, when a selection was active.
    pub fn selection(&self) -> Option<&SubsetInfo> {
        self.selection.as_ref()
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
    /// No posterior draws cloud resolved for the chosen stage — so there is no
    /// band to draw. Either a genuine optimizer stage (IF2 / NLopt: one best
    /// point, no cloud) or a Bayesian sampler whose `draws.tsv` is not yet
    /// present (incomplete / still running). The method (when known) tells the
    /// two apart, so the refusal never mislabels a sampler an optimizer (gh#343).
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

/// One free-forward design cell: a sweep coordinate × scenario, with the banded
/// streams that cell produced. The sweep axis lives here (in `run_predict`'s
/// loop), NOT in the scenario-keyed [`PredictiveSink`] — a fresh sink runs per
/// sweep-point, and each `(scenario, accumulator)` it yields becomes one cell.
struct FreeForwardCell {
    /// This cell's sweep coordinate: `(param, value)` per swept parameter, in
    /// sorted-name order. EMPTY when no `--sweep`.
    sweep: Vec<(String, f64)>,
    /// The scenario name this cell ran under (`fitted` for the no-overlay row).
    scenario: String,
    /// The banded predictive streams for this cell.
    bands: Vec<StreamBands>,
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

/// One (scenario, horizon) contribution to a `predictive/<stream>.tsv`: its rows,
/// plus the labels they carry (the scenario overlay axis, the horizon axis, the
/// treatment, the convergence, the draw count). Stacking several of these under
/// one header is how `fit predict` writes every scenario's `free_forward` rows
/// plus the `one_step` rows into the same file — the `scenario` and `horizon`
/// columns distinguish them.
pub struct PredictiveSection<'a> {
    /// The overlay axis value: a scenario name, or `fitted` for the no-overlay
    /// (fitted-model) rows. ALWAYS present (the leading column), the way
    /// `horizon`/`treatment` are.
    pub scenario: String,
    /// This cell's sweep coordinate: one `(param, value)` per swept parameter,
    /// in sorted-name order. EMPTY when no `--sweep` (and on the sweep-agnostic
    /// one-step section), so no `sweep:<param>` column is emitted and the header
    /// stays byte-identical to the no-sweep path.
    pub sweep: Vec<(String, f64)>,
    pub horizon: Horizon,
    pub treatment: TreatmentKind,
    pub convergence: ConvergenceStatus,
    pub n_draws: usize,
    pub rows: &'a [BandRow],
}

/// Render `predictive/<stream>.tsv`: `scenario | time | <dims…> | horizon |
/// treatment | rhat_max | ess_min | n_draws | q05 … q95`. Tidy, plot-ready; the
/// axes and the convergence channel are columns so a new predictive cell — a new
/// scenario, a new horizon, a new treatment — is more rows, never new consumer
/// code. Several [`PredictiveSection`]s (one per (scenario, horizon)) stack under
/// the single header.
pub fn render_predictive_tsv_sections(
    index_dims: &[String],
    sections: &[PredictiveSection],
) -> String {
    // The swept parameter names for the `sweep:<param>` columns: taken from the
    // first section that declares any (all free-forward sections share them; the
    // sweep-agnostic one-step section has none). EMPTY when no `--sweep` → no
    // sweep columns → byte-identical header.
    let sweep_names: Vec<String> = sections
        .iter()
        .find(|s| !s.sweep.is_empty())
        .map(|s| s.sweep.iter().map(|(n, _)| n.clone()).collect())
        .unwrap_or_default();

    let mut out = String::new();
    // Header — `scenario` leads (the overlay axis), then the `sweep:<param>`
    // columns, then everything else. Both are always present in the layout.
    out.push_str("scenario");
    for n in &sweep_names {
        out.push_str("\tsweep:");
        out.push_str(n);
    }
    out.push_str("\ttime");
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
            out.push_str(&section.scenario);
            // This section's swept values, aligned to `sweep_names`. A section
            // with no sweep coordinate (the one-step rows) leaves each cell empty.
            for name in &sweep_names {
                out.push('\t');
                if let Some((_, v)) = section.sweep.iter().find(|(n, _)| n == name) {
                    out.push_str(&fmt_value(*v));
                }
            }
            out.push('\t');
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
    ///
    /// `selection` (`--exclude-chains`) is applied to the cloud through the
    /// draws authority ([`PosteriorDrawsRef::load_params_with_info`]) — the one
    /// place a chain filter meets a cloud — and its provenance is carried onto
    /// the returned [`PosteriorDraws`].
    pub fn resolve(
        segment: &Path,
        stage: Option<&str>,
        selection: Option<ChainSelection>,
    ) -> Result<FitResult, String> {
        let seg_str = segment.to_str().ok_or("fit path is not valid UTF-8")?;
        match posterior_draws::resolve_posterior_draws(seg_str, stage) {
            Ok(pref) => {
                let pref = pref.with_selection(selection);
                let (rows, sel_info) = pref.load_params_with_info()?;
                let draws: Vec<IndexMap<String, f64>> = rows
                    .into_iter()
                    .map(|m| m.into_iter().collect())
                    .collect();
                let stage_dir = pref
                    .draws_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| segment.to_path_buf());
                // Convergence for the band. With NO chain selection this is the
                // stage's stored full-cloud summary. With `--exclude-chains`
                // active the stored summary describes the WHOLE cloud (including
                // the dropped chains), so it must not label a subset band — a
                // band drawn from the clean chains would carry a "did not
                // converge" verdict about a subset that did. Recompute R̂ / ESS
                // over the RETAINED chains through the same seam `fit summary`
                // uses, so the two agree on the same fit + selection (gh#409).
                let convergence = match &pref.chain_selection {
                    Some(sel) => subset_convergence(&pref.draws_path, sel)?,
                    None => read_convergence(&stage_dir, pref.method),
                };
                // Replay on the SAME backend the stage ran on; default only when
                // a bare stage dir was passed (no fit view to read it from).
                let backend = pref
                    .backend
                    .map(crate::args::types::ForwardBackend::from)
                    .unwrap_or(crate::args::types::ForwardBackend::ChainBinomial);
                Ok(FitResult::Posterior(
                    PosteriorDraws::new(
                        draws,
                        pref.stage,
                        pref.method,
                        backend,
                        convergence,
                    )?
                    .with_selection_info(sel_info),
                ))
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

/// Read a Bayesian stage's convergence summary (`<algorithm>_summary.json`):
/// `max` over its R̂ map, `min` over its ESS map. Returns
/// [`ConvergenceStatus::NotAssessed`] when no summary or no R̂ is present (a
/// single-chain stage), so a band is never silently "converged".
fn read_convergence(stage_dir: &Path, method: Option<FitAlgorithm>) -> ConvergenceStatus {
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
    // Each method reads its OWN `<algorithm>_summary.json` via the naming seam —
    // no cross-name fallback. A missing summary (an unnamed stage dir, or a
    // pre-rename mh fit that stored `pmmh_summary.json`) is NotAssessed, never
    // read under a sibling's name. The mh-summary rename is a clean break: a fit
    // produced before it must be re-run to be assessed here.
    method
        .map(|m| m.summary_filename())
        .as_deref()
        .and_then(try_read)
        .unwrap_or(ConvergenceStatus::NotAssessed)
}

/// Recompute a band's convergence over the RETAINED chains of a
/// `--exclude-chains` selection, so a chain-subset predictive band is labelled
/// with the R̂ / ESS of the subset it was drawn from — never the stored
/// full-cloud summary (which includes the dropped chains). Uses the ONE shared
/// recompute seam ([`crate::chain_selection::recompute_subset_diagnostics`] →
/// `runner::compute_rhat_ess` over the same `apply_keyed` filter) that
/// `fit summary --exclude-chains` uses, so the two agree on the same fit +
/// selection (gh#409).
///
/// Scores every param column (`None`); fixed columns yield a non-finite R̂ / ESS
/// that the `max` / `min` skip. Reduces to max R̂ / min ESS the same way
/// [`crate::fit::method_result::PosteriorDiagnostics::max_rhat`] /
/// [`min_ess`](crate::fit::method_result::PosteriorDiagnostics::min_ess) do —
/// [`ConvergenceStatus::NotAssessed`] when no param has a finite R̂ (e.g. a
/// single retained chain), rather than a falsely-converged band.
fn subset_convergence(
    draws_path: &Path,
    selection: &ChainSelection,
) -> Result<ConvergenceStatus, String> {
    let sub = crate::chain_selection::recompute_subset_diagnostics(draws_path, selection, None)?;
    let rhat_max = sub.rhat_per_param.values().copied().fold(f64::NEG_INFINITY, f64::max);
    if !rhat_max.is_finite() {
        return Ok(ConvergenceStatus::NotAssessed);
    }
    // Min ESS over the scored params; non-finite entries (non-finite-R̂ / R̂ > 1.1
    // params) are skipped by `f64::min`, matching `min_ess`. `None` (no params)
    // cannot occur here — a finite `rhat_max` proves at least one scored param.
    let ess_min = sub.ess_per_param.values().copied().reduce(f64::min).unwrap_or(f64::INFINITY);
    Ok(ConvergenceStatus::Reported { rhat_max, ess_min })
}

// ── The engine sink: sample y_rep per draw at the observed cadence ─────────

/// One scenario's accumulated free-forward output: the per-`(leaf, time)`
/// `y_rep` samples across that scenario's draws, plus the per-draw quantity
/// values and the trajectory snapshot grid. The engine runs every draw of one
/// scenario before the next (scenario is the outermost loop), so a cell merges
/// into its scenario's accumulator keyed by [`crate::engine::CellSpec::scenario`].
struct ScenarioAccum {
    /// `samples[leaf][time_idx]` = the `y_rep` values across this scenario's draws.
    samples: Vec<Vec<Vec<f64>>>,
    /// One inner `Vec` per draw: each quantity leaf's value, in
    /// `model.quantities` order. Empty when the model declares no quantities.
    quant_draws: Vec<Vec<sim::quantity::QuantityResult>>,
    /// The trajectory snapshot times, captured once per scenario (every draw
    /// shares the output cadence) — the time axis a series quantity bands against.
    quant_times: Vec<f64>,
}

/// A [`RunSink`] that samples `y_rep` for every fit leaf at the observed times,
/// for each draw (= cell), accumulating per `(scenario, leaf, time)` across
/// draws. The quantile reduction runs per scenario after all cells merge.
struct PredictiveSink {
    compiled: std::sync::Arc<sim::CompiledModel>,
    /// Per leaf (in `model.observations` order): the observation times to score.
    leaf_times: Vec<Vec<f64>>,
    /// Per leaf, per time (aligned with `leaf_times`): the observed auxiliary
    /// columns carried forward into the predictive draw (a data-supplied
    /// binomial denominator `n = n_examined`). Empty inner vec = no aux at that
    /// obs time (the likelihood's denominator then resolves data-free).
    leaf_aux: Vec<Vec<Vec<(String, f64)>>>,
    /// The generated-quantities evaluator, `Some` iff the model declares a
    /// `quantities {}` block. Composed alongside the obs-sample accumulator (same
    /// draw, same params) — not a second [`RunSink`]. Held behind an `Arc` so a
    /// fresh sink per sweep-point shares the one evaluator without rebuilding it.
    quant_eval: Option<std::sync::Arc<sim::quantity::QuantityEvaluator>>,
    /// Scenario name → its accumulator. Insertion order (= the engine's canonical
    /// `scenario → point → rep` order, scenario outermost) is preserved so the
    /// rendered files list scenarios in CLI order.
    by_scenario: IndexMap<String, ScenarioAccum>,
}

impl PredictiveSink {
    /// Get (or lazily create) the accumulator for `scenario`, sized to the leaf
    /// cadence.
    fn accum_for(&mut self, scenario: &str) -> &mut ScenarioAccum {
        let leaf_times = &self.leaf_times;
        self.by_scenario.entry(scenario.to_string()).or_insert_with(|| ScenarioAccum {
            samples: leaf_times.iter().map(|ts| vec![Vec::new(); ts.len()]).collect(),
            quant_draws: Vec::new(),
            quant_times: Vec::new(),
        })
    }
}

impl crate::engine::RunSink for PredictiveSink {
    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let model = &cell.model;
        // The cell's parameter vector for the observation sampler. Load-bearing on
        // TWO axes: (1) it must read the DRAW's parameters (e.g. an estimated
        // reporting rate), so the posterior predictive carries observation-
        // parameter uncertainty; (2) it must read the SCENARIO's overlay (e.g.
        // `set = { rho = 0.3 }`), so a counterfactual that changes an
        // observation-only parameter actually shifts the predictive bands. The
        // engine resolved BOTH into `cell.model.parameters` (the scenario via
        // `apply_scenario_filter`/`params_resolver`, the draw via the cell's
        // point_overrides), so we read the resolved values from there — the single
        // source of truth — rather than re-deriving from default_params + the draw
        // (which would silently drop the scenario overlay, identical bands across
        // scenarios for an obs-only parameter).
        let mut params = self.compiled.default_params.clone();
        for p in &model.parameters {
            if let (Some(&idx), Some(v)) =
                (self.compiled.param_index.get(p.name.as_str()), p.value.resolved_value())
            {
                params[idx] = v;
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
        // Sample y_rep into a per-leaf scratch first, then commit into this
        // cell's scenario accumulator (so the borrow of `self.compiled` /
        // `self.quant_eval` does not overlap the `&mut self` accumulator borrow).
        let scenario_name = cell.spec.scenario.name().to_string();
        let mut leaf_y: Vec<Vec<f64>> = vec![Vec::new(); model.observations.len()];
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
            let leaf_aux = &self.leaf_aux[si];
            let mut stream_vals: Vec<f64> = Vec::with_capacity(times.len());
            for (ti, &t) in times.iter().enumerate() {
                let snap = crate::snap_at(&cell.traj, t);
                // Carry the OBSERVED aux (survey denominator) at this obs time
                // into the draw; `&[]` when the leaf has no aux (aligned 1:1 with
                // `times`, so a wrong index is impossible).
                let aux: &[(String, f64)] = leaf_aux.get(ti).map(|v| v.as_slice()).unwrap_or(&[]);
                let y = sampler(projected[ti], t, &snap.int_state.counts, aux, &mut obs_rng);
                stream_vals.push(y);
            }
            if want_obs {
                // Key by the stream's declared `name` — what `observations.<name>`
                // in the DSL references (v1.1 is unstratified, so name == base).
                obs_set.streams.insert(obs_ir.name.clone(), (times.clone(), stream_vals.clone()));
            }
            leaf_y[si] = stream_vals;
        }

        // Generated quantities: fold this draw's trajectory + the just-drawn y_sim
        // into its per-quantity values, using the SAME resolved params + draws as
        // the predictive output above.
        let quant_results = self
            .quant_eval
            .as_ref()
            .map(|eval| eval.eval_draw(&params, &cell.traj, &self.compiled, Some(&obs_set)));
        let snapshot_times: Vec<f64> = if quant_results.is_some() {
            cell.traj.snapshots.iter().map(|s| s.t).collect()
        } else {
            Vec::new()
        };

        // Commit into the cell's scenario accumulator.
        let acc = self.accum_for(&scenario_name);
        for (si, ys) in leaf_y.into_iter().enumerate() {
            for (ti, y) in ys.into_iter().enumerate() {
                acc.samples[si][ti].push(y);
            }
        }
        if let Some(results) = quant_results {
            if acc.quant_times.is_empty() {
                acc.quant_times = snapshot_times;
            }
            acc.quant_draws.push(results);
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
/// observed cells (`None` = a hole), and per-observation auxiliary columns
/// (e.g. a survey denominator `n = n_examined`) aligned 1:1 with `times`.
struct LeafObs {
    source: String,
    stratum: Vec<(String, String)>,
    times: Vec<f64>,
    observed: Vec<Option<f64>>,
    /// `aux[i]` = the auxiliary `(column, value)` pairs observed at `times[i]`.
    /// Carried into the free-forward predictive so a data-supplied denominator
    /// (`binomial(n = n_examined, …)`) draws `y_rep ~ binomial(n_examined, p̂)`
    /// rather than `binomial(0, …) = 0`.
    aux: Vec<Vec<(String, f64)>>,
}

/// Expand the `--sweep` specs into Cartesian grid cells, each a sorted-by-name
/// `(param, value)` list. No specs → a single empty cell, so the free-forward
/// production runs exactly once (byte-identical to the no-sweep path). Mirrors
/// [`crate::batch`]'s `expand_sweep`, but yields a sorted-name coordinate so the
/// `sweep:<param>` columns (and the manifest sweep object) are deterministic.
fn expand_predict_sweep(specs: &[crate::args::types::SweepSpec]) -> Vec<Vec<(String, f64)>> {
    if specs.is_empty() {
        return vec![Vec::new()];
    }
    // Sort by parameter name for a deterministic column/axis order.
    let mut sorted: Vec<&crate::args::types::SweepSpec> = specs.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    let mut cells: Vec<Vec<(String, f64)>> = vec![Vec::new()];
    for spec in sorted {
        let values = spec.grid.expand();
        let mut next = Vec::with_capacity(cells.len() * values.len());
        for cell in &cells {
            for &v in &values {
                let mut c = cell.clone();
                c.push((spec.name.clone(), v));
                next.push(c);
            }
        }
        cells = next;
    }
    cells
}

fn run_predict(args: &crate::args::FitPredictArgs) -> Result<Vec<PathBuf>, String> {
    // 1. Resolve the fit handle (@label / hash prefix / run-dir / fit.toml) →
    //    its segment + config.
    let crate::fit::handle::ResolvedFit { segment, config } =
        crate::fit::handle::resolve_fit(args.fit()?).map_err(|e| e.to_string())?;

    // 2. Resolve the posterior — by artifact. A point-estimate fit is refused.
    //    `--exclude-chains` is parsed at the boundary into a typed selection and
    //    applied to the cloud through the draws authority (one filter, once).
    let selection = args
        .exclude_chains
        .as_deref()
        .map(ChainSelection::parse_exclude)
        .transpose()?;
    let fit_result = FitResult::resolve(&segment, args.stage.as_deref(), selection)?;
    let treatment = fit_result.into_treatment();
    // The label the artifact carries, derived from the treatment before we
    // unwrap the cloud (v1 only ever reaches `posterior` here, but the label is
    // read off the treatment, not hardcoded).
    let treatment_kind = treatment.kind();
    let posterior = match treatment {
        ParamTreatment::Posterior(pd) => pd,
        ParamTreatment::PlugIn { method, stage } => return Err(plugin_refusal(method, &stage)),
    };

    // A chain selection actually dropped chains — warn loudly (non-quietable),
    // naming the dropped chains and the bias direction, before any output.
    if let Some(info) = posterior.selection() {
        warn_active_selection(info);
    }

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

    // 3. Resolve the model IR. Prefer the IR archived in the fit leaf at
    //    `fit run` time (gh#322 — a self-contained, portable run); fall back to
    //    recompiling the loose `.camdl` recorded in the config only when the
    //    archive is absent (a run produced before IR archival landed). The
    //    archived IR is itself a compiled-IR path, so the engine's per-cell
    //    recompile (`job.model`, below) loads it directly.
    let archived_ir = segment.join("model.ir.json");
    let (compiled_ir, _ir_tmp): (String, Option<std::path::PathBuf>) = if archived_ir.is_file() {
        (archived_ir.to_string_lossy().into_owned(), None)
    } else {
        // `fit predict` replays forward trajectories from stored draws — it never
        // recomputes an ODE gradient, so compile lean (`needs_state_grad = false`,
        // gh#439 A2).
        crate::util::resolve_ir_path(&config.model.camdl, false)?
    };
    let (model, _) = crate::util::load_model(&compiled_ir)?;
    // One calendar block for every sidecar manifest this run writes
    // (predictive / observed / quantities), read once off the model.
    let calendar = io::CalendarMeta::from_model(&model);
    let dt = model.simulation.dt.unwrap_or(1.0);

    // 3b. Parse the prospective scenario overlay (reusing simulate's ScenarioRef
    // surface). No `--scenario` → a single `fitted` row (the fitted model, no
    // overlay). Validate each ref against the model's presets up front so an
    // unknown `--scenario NAME` errors with the available presets, BEFORE any
    // simulation runs (the resolver's actionable message).
    let scenario_refs = args.scenario_refs()?;
    let preset_names: Vec<String> =
        model.presets.iter().map(|p| p.name.clone()).collect();
    for sref in &scenario_refs {
        crate::sim_job::resolve_scenario_ref(sref, &preset_names)?;
    }
    // Layer 1 supports param-overlay scenarios cleanly; an intervention-toggling
    // scenario (enable/disable) replays correctly through the engine — the engine
    // recompiles the model per cell with the scenario's intervention set applied
    // (`apply_scenario_filter`) and simulates from t_start, so the schedule is
    // re-seated by construction (NOT a resume-from-saved-state path, so the gh#216
    // inference hazard does not apply to a forward replay). Should a future change
    // make a toggle unsupported, route it through the capability/validation path
    // (a loud error), never a silent baseline replay — see Guard 2.

    // 3c. Expand the parameter sweep into Cartesian grid cells, each a sorted
    // `(param, value)` list. No `--sweep` → a single empty cell, so the
    // free-forward production runs exactly once, byte-identical to the no-sweep
    // path. A swept value rides in the SAME draw/sweep tier as a draw (it
    // OVERRIDES the swept parameter in each draw row), so the resolver applies it
    // below the scenario tier — a scenario still wins over a sweep.
    let sweep_points: Vec<Vec<(String, f64)>> = expand_predict_sweep(&args.sweep);
    let swept_params: std::collections::BTreeSet<String> =
        args.sweep.iter().map(|s| s.name.clone()).collect();

    // Each swept parameter must be a real model parameter, named at most once —
    // a sweep over an undeclared (or duplicated) name would vary a `sweep:<param>`
    // column while the dynamics never move (a silent no-op).
    {
        use std::collections::HashSet;
        if swept_params.len() != args.sweep.len() {
            return Err(
                "--sweep names the same parameter more than once; sweep each \
                 parameter at most once (the grid is the Cartesian product of \
                 distinct parameters)"
                    .to_string(),
            );
        }
        let model_params: HashSet<&str> =
            model.parameters.iter().map(|p| p.name.as_str()).collect();
        let unknown: Vec<&str> = swept_params
            .iter()
            .map(|s| s.as_str())
            .filter(|p| !model_params.contains(p))
            .collect();
        if !unknown.is_empty() {
            return Err(format!(
                "--sweep names parameter(s) the model does not declare: {} \
                 (a sweep must vary a real model parameter)",
                unknown.join(", ")
            ));
        }
    }

    // 3d. A scenario and a sweep on the SAME parameter is a hard error: the
    // scenario PINS the parameter (winning over the draw/sweep tier) while the
    // sweep VARIES it — applying both at once is contradictory (the scenario would
    // silently override every sweep cell, collapsing the grid). The two guards
    // (this one and the engine's explicit-`--draws` guard) share one footprint
    // (`scenario_param_footprint`) so they cannot disagree. Runs BEFORE any
    // simulation.
    if !swept_params.is_empty() {
        for sref in &scenario_refs {
            let footprint = crate::params_resolver::scenario_param_footprint(&model, sref)?;
            let mut clash: Vec<&str> = footprint
                .iter()
                .map(|k| k.as_str())
                .filter(|k| swept_params.contains(*k))
                .collect();
            if !clash.is_empty() {
                clash.sort();
                clash.dedup();
                let param_list = clash.join(", ");
                let swept_list: Vec<&str> = swept_params.iter().map(|s| s.as_str()).collect();
                return Err(format!(
                    "scenario '{scen}' pins parameter(s) [{param_list}] that --sweep \
                     also varies (sweep over [{swept}]). A scenario sets/scales these \
                     parameters and wins over the sweep, so pinning them via the \
                     scenario and varying them via --sweep at once is contradictory.\n  \
                     Fix: drop [{param_list}] from one side — pin the parameter via the \
                     scenario, OR vary it via --sweep, not both.",
                    scen = sref.name(),
                    swept = swept_list.join(", "),
                ));
            }
        }
    }

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
    // Held behind an `Arc` so each sweep-point's fresh sink shares the one
    // evaluator without rebuilding it.
    let quant_eval: Option<std::sync::Arc<sim::quantity::QuantityEvaluator>> =
        if !model.quantities.is_empty() {
            Some(std::sync::Arc::new(
                sim::quantity::QuantityEvaluator::new(&model.quantities, compiled.as_ref())
                    .map_err(|e| format!("building quantity evaluator: {e}"))?,
            ))
        } else {
            None
        };
    // The rendered quantity sidecars (per logical quantity, all design cells
    // stacked) + the merged manifest, filled after the free-forward pass.
    let mut quantity_outputs: Vec<(String, String)> = Vec::new();
    let mut quantity_manifest: Option<String> = None;
    // The free-forward bands, one [`FreeForwardCell`] per (sweep-point × scenario)
    // in engine canonical order. `None` when the free-forward horizon was not
    // requested.
    let mut free_forward: Option<Vec<FreeForwardCell>> = None;
    // Count of posterior draws the free-forward horizon actually replayed (the
    // `--n-draws` subsample, or the full cloud when it is ≤ the cap). Carried
    // onto each free-forward band section's `n_draws` diagnostic. Stays 0 when
    // the horizon was not requested (that band section then never runs).
    let mut ff_n_draws: usize = 0;

    // ── Free-forward horizon: replay the posterior forward under each scenario.
    //
    // The scenario is the OVERLAY on the fitted parameters: its `set`/`scale` must
    // WIN over the draw (a counterfactual `set = { rho = 0.3 }` has to override the
    // fitted rho), and its `enable`/`disable` must re-seat the intervention set.
    // The resolver now enforces exactly this precedence (spec §1.3): a posterior
    // draw routes through the DRAW/SWEEP tier (below scenario), so a scenario
    // `set`/`scale` wins over the draw automatically. We therefore hand the engine
    // the UNMODIFIED draw rows plus the original scenario reference — a `Named`
    // preset replays the preset's `set`/`scale`/`enable`/`disable` from the model
    // (the resolver's preset path), and an ad-hoc `Inline` ref applies its inline
    // `set` + intervention toggle. No hand-folding of `set`/`scale` into draws (a
    // `scale` folded per draw AND re-applied by the resolver would double-apply,
    // e.g. ×1.5 → ×2.25); the resolver is the single precedence authority.
    //
    // Paired-seed CRN holds across scenarios: each per-scenario job has the same
    // `total_runs` and the same seed, so `process_seed_for` derives identical
    // per-draw seeds — the scenarios' pre-divergence trajectories are coupled.
    if want_free_forward {
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
        // Observed aux per leaf, in the SAME model.observations order + the same
        // per-leaf time alignment as `leaf_times` (both cloned from the matched
        // `LeafObs`), so `leaf_aux[si][ti]` is the survey denominator at
        // `leaf_times[si][ti]`.
        let leaf_aux: Vec<Vec<Vec<(String, f64)>>> = model
            .observations
            .iter()
            .map(|o| {
                leaves
                    .iter()
                    .find(|l| leaf_matches(o, l))
                    .map(|l| l.aux.clone())
                    .unwrap_or_default()
            })
            .collect();

        // The free-forward cells (one per sweep-point × scenario, engine canonical
        // order), plus the stacked quantity bodies + merged manifest entries — all
        // accumulated across the sweep grid. quantity name → accumulated TSV body
        // (header written once, from the first design cell's render).
        let mut ff_cells: Vec<FreeForwardCell> = Vec::new();
        let mut quant_bodies: IndexMap<String, String> = IndexMap::new();
        let mut quant_manifest_entries: Vec<serde_json::Value> = Vec::new();

        // ── Honor --n-draws on free-forward: an even, deterministic subsample of
        // the whole posterior cloud (gh#387). Without it the free-forward path
        // replays EVERY draw single-threaded, so a long-burn-in ODE fit
        // (thousands of ~seconds-each solves) never finishes and no artifact is
        // written. Same knob + default + strided pick the one-step horizon uses,
        // so both horizons subsample identically. Computed ONCE — the subsample
        // is scenario/sweep-independent.
        let ff_cap = args.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS);
        let ff_draws = subsample_draws(posterior.draws(), ff_cap);
        ff_n_draws = ff_draws.len();
        if ff_n_draws < posterior.n_draws() {
            eprintln!(
                "fit predict: free_forward horizon — subsampling {ff_n_draws} of {} \
                 posterior draws (raise with --n-draws)",
                posterior.n_draws()
            );
        }
        // Fan the (draws × scenarios) replay grid across Rayon. The engine seeds
        // each cell by its planned `point_idx`/`rep` (`process_seed_for`),
        // independent of execution order, so parallelism never perturbs a
        // trajectory (engine.rs) — the bands are byte-identical to a sequential
        // replay of the same subsample. `fit predict` has no thread-budget flag,
        // so default to the machine width; `RAYON_NUM_THREADS` still caps the
        // global pool.
        let ff_parallel = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

        for sweep_pt in &sweep_points {
            // A FRESH sink per sweep-point keeps the sink scenario-keyed (no sink
            // rewrite); the sweep axis lives in this loop, not the sink. The
            // evaluator is shared via the `Arc` clone.
            let mut sink = PredictiveSink {
                compiled: compiled.clone(),
                leaf_times: leaf_times.clone(),
                leaf_aux: leaf_aux.clone(),
                quant_eval: quant_eval.clone(),
                by_scenario: IndexMap::new(),
            };

            for sref in &scenario_refs {
                // Draw rows for this sweep cell: each posterior draw with the swept
                // parameters OVERWRITTEN to this cell's grid values (the draw
                // supplies every other parameter). The swept value lands in the
                // SAME draw/sweep tier as the draw (`point_overrides`), so the
                // resolver applies the scenario's `set`/`scale`/`enable`/`disable`
                // ON TOP — exactly once — and a scenario still wins over the sweep.
                // No per-draw folding of `set`/`scale` (that would double-apply).
                let rows: Vec<IndexMap<String, f64>> = ff_draws
                    .iter()
                    .map(|d| {
                        let mut r = (*d).clone();
                        for (name, val) in sweep_pt {
                            r.insert(name.clone(), *val);
                        }
                        r
                    })
                    .collect();
                let job = crate::sim_job::SimulateJob {
                    model: compiled_ir.clone(),
                    params_files: vec![],
                    // Replay on the SAME forward simulator the fit used
                    // (chain_binomial / ode), resolved from the stage — never a
                    // hardcoded default.
                    backend: posterior.backend,
                    dt,
                    integrator: None,
                    // Generated posterior draws (not a user-authored file), so a
                    // scenario simply wins over a draw/sweep column — no collision
                    // error (the scenario×sweep guard above already rejected a
                    // same-parameter clash).
                    source: crate::sim_job::ParamSource::Draws {
                        rows,
                        replicates: 1,
                        explicit_file: None,
                    },
                    // The original scenario reference drives the engine's scenario
                    // tier; the scenario NAME is carried for the sink's per-scenario
                    // partition (`cell.spec.scenario.name()`).
                    scenarios: vec![sref.clone()],
                    seeds: crate::sim_job::Seeds::Single(seed),
                    cli_overrides: vec![],
                    set_vec_entries: vec![],
                    table_files: vec![],
                    obs: crate::sim_job::ObsOutput::None,
                    parallel: ff_parallel,
                };
                crate::engine::run_job(&job, &mut sink)?;
            }

            // Per (this sweep-point × scenario): band the predictive samples +
            // (if present) the quantity draws, stacking quantity rows for every
            // design cell under one header per logical quantity and merging the
            // manifests. Scenario order = the sink's insertion order (engine
            // canonical order, scenario outermost).
            for (scenario_name, accum) in &sink.by_scenario {
                let coords = crate::quantity_output::DesignCoords {
                    scenario: Some(scenario_name),
                    sweep: sweep_pt,
                };
                if !model.quantities.is_empty() {
                    let (outs, manifest) = crate::quantity_output::render_quantities(
                        &model.quantities,
                        &accum.quant_draws,
                        &accum.quant_times,
                        crate::quantity_output::Mode::Banded,
                        coords,
                        &calendar,
                    )?;
                    for (name, content) in outs {
                        // First design cell for this quantity: keep its header +
                        // rows. Subsequent cells: append only the data rows (drop
                        // the repeated header line) so all cells stack under one
                        // header.
                        match quant_bodies.entry(name) {
                            indexmap::map::Entry::Vacant(e) => {
                                e.insert(content);
                            }
                            indexmap::map::Entry::Occupied(mut e) => {
                                let body: String =
                                    content.split_inclusive('\n').skip(1).collect();
                                e.get_mut().push_str(&body);
                            }
                        }
                    }
                    let m: serde_json::Value = serde_json::from_str(&manifest)
                        .map_err(|e| format!("parsing quantities manifest: {e}"))?;
                    if let Some(arr) = m["quantities"].as_array() {
                        quant_manifest_entries.extend(arr.iter().cloned());
                    }
                }
                ff_cells.push(FreeForwardCell {
                    sweep: sweep_pt.clone(),
                    scenario: scenario_name.clone(),
                    bands: assemble_predictive(
                        &model, accum, &leaf_times, &leaves, schema.as_ref(),
                    )?,
                });
            }
        }

        if !model.quantities.is_empty() {
            quantity_outputs = quant_bodies.into_iter().collect();
            let merged = serde_json::json!({
                "schema": "camdl.quantities/v1",
                "quantities": quant_manifest_entries,
            });
            quantity_manifest = Some(
                serde_json::to_string_pretty(&merged)
                    .map_err(|e| format!("serializing quantities manifest: {e}"))?,
            );
        }
        free_forward = Some(ff_cells);
    }

    // ── One-step horizon: per-draw bootstrap filter over the data, pooled. Runs
    // only when the witness was built (a filterable fit and the horizon wanted).
    let n_draws_cap = args.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS);
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

    // 6. Write the predictive artifact — every scenario's free-forward rows plus
    // the one-step rows stacked into one file per logical stream (the leading
    // `scenario` column and the typed `horizon` column distinguish the rows). The
    // observed half follows.
    //
    // The one-step horizon is scenario-AGNOSTIC: it filters the OBSERVED data
    // through the fitted model, so it carries no overlay — applying a counter-
    // factual scenario to a filter over the actual data is ill-defined (you would
    // condition the modified model on data the unmodified model generated). It is
    // emitted once, tagged `fitted`, alongside every scenario's free-forward
    // rows. The `scenario`/`horizon` columns keep the file tidy.
    let mut written = Vec::new();
    let one_step_streams: &[StreamBands] = one_step.as_ref().map(|(s, _)| s.as_slice()).unwrap_or(&[]);
    let one_step_n = one_step.as_ref().map(|(_, n)| *n).unwrap_or(0);

    // Union of source names across all (sweep × scenario) free-forward streams +
    // the one-step streams, preserving free-forward order first.
    let mut sources: Vec<String> = Vec::new();
    if let Some(ff) = &free_forward {
        for cell in ff {
            for s in &cell.bands {
                if !sources.contains(&s.source) {
                    sources.push(s.source.clone());
                }
            }
        }
    }
    for s in one_step_streams {
        if !sources.contains(&s.source) {
            sources.push(s.source.clone());
        }
    }

    // The `sweep:<param>` coordinate columns, in the same sorted-name order the
    // predictive header uses (empty without `--sweep`). Computed once — uniform
    // across streams.
    let sweep_col_names: Vec<String> = {
        let mut names: Vec<String> = args.sweep.iter().map(|s| s.name.clone()).collect();
        names.sort();
        names
    };
    // `predictive.json`: the per-stream join contract — which columns are
    // coordinates (group-by keys) vs the band, plus the value kind and the band
    // quantiles — so a downstream reader joins without reverse-engineering
    // headers. Net-new sibling of `quantities.json`.
    let mut predictive_manifest_entries: Vec<serde_json::Value> = Vec::new();

    for source in &sources {
        // Each free-forward design cell's StreamBands for this source (in
        // sweep × scenario order), plus the one-step StreamBands (sweep- and
        // scenario-agnostic).
        let ff_for_source: Vec<(&FreeForwardCell, &StreamBands)> = free_forward
            .as_ref()
            .map(|ff| {
                ff.iter()
                    .filter_map(|cell| {
                        cell.bands
                            .iter()
                            .find(|s| &s.source == source)
                            .map(|s| (cell, s))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let os_stream = one_step_streams.iter().find(|s| &s.source == source);
        // index_dims is the same across design cells/horizons (same schema/leaf);
        // take it from whichever section is present.
        let index_dims = ff_for_source
            .first()
            .map(|(_, s)| s.index_dims.clone())
            .or_else(|| os_stream.map(|s| s.index_dims.clone()))
            .unwrap_or_default();

        let mut sections: Vec<PredictiveSection> = Vec::new();
        for (cell, s) in &ff_for_source {
            sections.push(PredictiveSection {
                scenario: cell.scenario.clone(),
                sweep: cell.sweep.clone(),
                horizon: Horizon::FreeForward,
                treatment: treatment_kind,
                convergence: posterior.convergence,
                n_draws: ff_n_draws,
                rows: &s.rows,
            });
        }
        if let Some(s) = os_stream {
            sections.push(PredictiveSection {
                scenario: crate::args::FITTED.to_string(),
                // The one-step horizon is sweep-agnostic (it filters the OBSERVED
                // data through the fitted model, so a swept-parameter overlay is
                // ill-defined) — empty sweep ⇒ empty `sweep:<param>` cells.
                sweep: Vec::new(),
                horizon: Horizon::OneStepAhead,
                treatment: treatment_kind,
                convergence: posterior.convergence,
                n_draws: one_step_n,
                rows: &s.rows,
            });
        }
        // Record this stream's join contract for `predictive.json`. Coordinate
        // columns (the group-by keys) in header order: `scenario`, the
        // `sweep:<param>` columns, `time`, the stratum dims, `horizon`,
        // `treatment`. The band columns are the quantile labels; `rhat_max` /
        // `ess_min` / `n_draws` are per-cell diagnostics. `value_kind` is the
        // observation's likelihood family (the nature of the banded value).
        let mut coordinates: Vec<String> = vec!["scenario".to_string()];
        coordinates.extend(sweep_col_names.iter().map(|n| format!("sweep:{n}")));
        coordinates.push("time".to_string());
        coordinates.extend(index_dims.iter().cloned());
        coordinates.push("horizon".to_string());
        coordinates.push("treatment".to_string());
        let value_kind = model
            .observations
            .iter()
            .find(|o| &o.name == source || &o.source == source)
            .map(|o| o.likelihood.name())
            .unwrap_or("count");
        predictive_manifest_entries.push(serde_json::json!({
            "name": source,
            "file": format!("predictive/{source}.tsv"),
            "value_kind": value_kind,
            "coordinates": coordinates,
            "diagnostics": ["rhat_max", "ess_min", "n_draws"],
            "band": QUANTILE_LEVELS.iter().map(|(_, l)| *l).collect::<Vec<_>>(),
            "quantiles": QUANTILE_LEVELS.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
        }));

        let pred_tsv = render_predictive_tsv_sections(&index_dims, &sections);
        written.push(write_tsv(&segment, "predictive", source, &pred_tsv)?);
    }
    // `predictive.json`: the per-stream join contract beside the predictive
    // TSVs — a sibling of `quantities.json`, NOT in the run_id-keyed CAS leaf
    // (regenerated, overwritten in place). Written whenever any predictive
    // stream was emitted.
    if !predictive_manifest_entries.is_empty() {
        let mut manifest = serde_json::json!({
            "schema": "camdl.predictive/v1",
            "calendar": calendar.to_json(),
            "streams": predictive_manifest_entries,
        });
        // Provenance: a chain-subset predictive records the selection alongside
        // the streams, so a chain-subset artifact is never mistakable for a
        // full-cloud one. Absent (no key) when the full cloud was used.
        if let Some(info) = posterior.selection() {
            manifest["chain_selection"] = info.to_json();
        }
        let path = segment.join("predictive.json");
        let text = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("serializing predictive manifest: {e}"))?;
        std::fs::write(&path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        written.push(path);
    }

    // The observed half, grouped by logical stream, plus an `observed.json`
    // manifest carrying the calendar semantics for the `time` column — a sibling
    // of `predictive.json`, NOT in the run_id-keyed CAS leaf (regenerated,
    // overwritten in place).
    let mut observed_manifest_entries: Vec<serde_json::Value> = Vec::new();
    for (source, index_dims, rows) in observed_by_stream(&leaves, schema.as_ref()) {
        let obs_tsv = render_observed_tsv(&index_dims, &rows);
        written.push(write_tsv(&segment, "observed", &source, &obs_tsv)?);
        let file = format!("observed/{source}.tsv");
        let mut coordinates: Vec<String> = vec!["time".to_string()];
        coordinates.extend(index_dims.iter().cloned());
        observed_manifest_entries.push(serde_json::json!({
            "name": source,
            "file": file,
            "coordinates": coordinates,
            "value_column": "value",
        }));
    }
    if !observed_manifest_entries.is_empty() {
        let manifest = serde_json::json!({
            "schema": "camdl.observed/v1",
            "calendar": calendar.to_json(),
            "streams": observed_manifest_entries,
        });
        let path = segment.join("observed.json");
        let text = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("serializing observed manifest: {e}"))?;
        std::fs::write(&path, text)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        written.push(path);
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
    // Counterfactual contrasts: auto-emitted when the model declares any
    // `contrasts {}`. The two-arm replay reducer forks each forkable posterior
    // draw from its smoothed X(T*) and bands the difference into
    // `contrasts/<name>.tsv`. A model with no `contrasts {}` is byte-identical
    // (this is a no-op). A non-forkable / ODE fit emits no file and a located note.
    if !model.contrasts.is_empty() {
        let paths = crate::fit::contrasts::emit_contrasts(
            &segment,
            args.stage.as_deref(),
            &model,
            posterior.backend,
            seed,
        )?;
        written.extend(paths);
    }
    let method_label = posterior.method.map(|m| m.as_str()).unwrap_or("posterior");
    let mut horizons: Vec<String> = Vec::new();
    if free_forward.is_some() {
        horizons.push("free_forward".to_string());
    }
    if one_step.is_some() {
        horizons.push(format!("one_step({one_step_n} draws)"));
    }
    let scenario_labels: Vec<&str> = scenario_refs.iter().map(|s| s.name()).collect();
    eprintln!(
        "fit predict: horizon={} treatment=posterior, {} scenario(s) [{}], {} stream(s), \
         {} draws from {} stage '{}'",
        horizons.join("+"),
        scenario_labels.len(),
        scenario_labels.join(", "),
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
        let (obs, cells, aux) =
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
        out.push(LeafObs { source: obs_model.source.clone(), stratum, times, observed, aux });
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

/// Quantile-reduce ONE scenario's accumulated free-forward samples into per-stream
/// bands, grouping leaves by logical stream. The scenario/horizon/treatment/
/// convergence/n_draws labels are applied at render time (each
/// [`PredictiveSection`] carries them), so this returns only the bands.
fn assemble_predictive(
    model: &ir::Model,
    accum: &ScenarioAccum,
    leaf_times: &[Vec<f64>],
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
            for (ti, draws_at_t) in accum.samples[si].iter().enumerate() {
                let quantiles = band(draws_at_t).map_err(|e| {
                    format!("stream '{source}' at t={}: {e}", leaf_times[si][ti])
                })?;
                rows.push(BandRow {
                    time: leaf_times[si][ti],
                    stratum: stratum.clone(),
                    quantiles,
                });
            }
        }
        streams.push(StreamBands { source: source.clone(), index_dims, rows });
    }

    Ok(streams)
}

// ── Posterior-cloud subsampling (shared by both horizons) ───────────────────

/// Default posterior-cloud subsample cap for `fit predict`, shared by both
/// horizons. Neither needs the full fit-grade cloud: the one-step band pools
/// `draws × n_particles` per cell and the free-forward band pools one forward
/// replay per draw, so a few hundred draws saturate q05…q95. The full cloud is
/// never replayed silently — a full free-forward replay of a long-burn-in ODE
/// fit is thousands of ~seconds-each solves (gh#387).
const DEFAULT_PREDICT_DRAWS: usize = 200;

/// Evenly-spaced, deterministic subsample of a posterior cloud down to `cap`
/// draws (the whole cloud when `cap >= len`, always at least one). Both horizons
/// cap the cloud through this one seam, so a fit-grade cloud is never silently
/// replayed at full size. The pick is STRIDED across the whole cloud
/// (`idx = i * total / n_used`), never `take(cap)` of the front — a front-take
/// would bias the band toward early sweeps / a single chain. Chosen draws are
/// returned in cloud order.
fn subsample_draws(draws: &[IndexMap<String, f64>], cap: usize) -> Vec<&IndexMap<String, f64>> {
    let total = draws.len();
    let n_used = cap.min(total).max(1);
    if n_used >= total {
        draws.iter().collect()
    } else {
        (0..n_used).map(|i| &draws[(i * total) / n_used]).collect()
    }
}

// ── The one-step-ahead posterior predictive producer ───────────────────────

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
        bootstrap_filter, multi_stream_obs::StreamProjection,
        traits::SMCConfig, BoundObs, ChainBinomialProcess, MultiStreamObsModel,
    };
    use crate::fit::runner::ObsStream;

    // ── Build the filter obs model ONCE. Resolution here is predict-specific
    // (skips a fit-only diagnostic stream whose source is unbound; honours the
    // `--stream` filter; iterates in `model.observations` order so `stream_idx`
    // maps back to its IR leaf via `bound_leaves`), so it does NOT route through
    // `resolve_and_load_obs_streams`. But the `ObsStream -> StreamSpec` assembly
    // (the bug-prone shared part) IS routed through the single shared builder
    // `stream_specs_from_obs_streams`, the same one `fit run` and `pfilter` use.
    // Each bound leaf reuses `runner::load_observations` for its cells + cadence.
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
    let mut obs_streams: Vec<ObsStream> = Vec::new();
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
        obs_streams.push(ObsStream {
            name: obs_model.name.clone(),
            projection,
            obs_model_ir: obs_model.clone(),
            data: obs,
            cells,
            aux,
        });
        bound_leaves.push(obs_model);
    }
    if obs_streams.is_empty() {
        return Err("no observation streams to predict — check that the model's \
                    observation sources are bound to data in the fit config, and that \
                    --stream (if given) names a real stream"
            .into());
    }

    let specs = crate::fit::runner::stream_specs_from_obs_streams(&obs_streams);
    let (bound, _report) = BoundObs::bind(specs)
        .map_err(|report| format!("observation data invalid:\n{}", report.render()))?;
    let obs_model = MultiStreamObsModel::new(bound, compiled.clone())
        .map_err(|e| format!("observation model construction failed: {e:?}"))?;

    // gh#191 review (defense-in-depth): one_step runs the chain_binomial particle
    // filter, which does not advance real-valued compartments. A chain_binomial
    // posterior can only exist for a model that already passed
    // check_model_capabilities at fit time (withholding REAL_COMPARTMENTS +
    // rejecting multi-source, gh#121), so this makes the invariant explicit rather
    // than relying purely on the FilterableFit type witness.
    crate::fit::methods::check_model_capabilities(
        crate::run_meta::InferenceBackend::ChainBinomial, &compiled)?;

    // ── Build the process ONCE.
    let process = ChainBinomialProcess::new(compiled.clone());

    // ── Subsample the posterior cloud (never silently run the full cloud).
    let draws = fit.draws().draws();
    let total = draws.len();
    let chosen = subsample_draws(draws, n_draws_cap);
    let n_used = chosen.len();
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

/// The actionable refusal when a fit reaches `predict` with no posterior band to
/// draw. Two genuinely different causes, told apart by the stage's method
/// (gh#343): a Bayesian sampler (PGAS / PMMH / MH) that simply has not written
/// its `draws.tsv` yet — incomplete or still running — versus a real optimizer
/// (IF2 / NLopt) that returns a single point and never has a band. Framing a
/// sampler as an "optimizer fit" misdirected the user of the *default* Bayesian
/// method to the plug-in workflow, discarding the posterior it does have.
fn plugin_refusal(method: Option<FitAlgorithm>, stage: &str) -> String {
    if let Some(m) = method.filter(|m| m.is_posterior_sampler()) {
        return format!(
            "stage '{stage}' ({m}) is a Bayesian posterior sampler, but it has not \
             written its posterior draws (draws.tsv) — that file is written at stage \
             completion, so the fit is likely incomplete or still running.\n  \
             Let the fit finish (or re-run it); a completed {m} stage has a full \
             posterior cloud, and `camdl fit predict` will draw the band with no \
             extra flags."
        );
    }
    if let Some(m) = method.filter(|m| m.is_optimizer()) {
        return format!(
            "stage '{stage}' is an optimizer fit ({m}) — it returns a single best-fit \
             parameter set, not a distribution, so there is no posterior band to draw.\n  \
             Get those parameters and run a plug-in forward simulation instead:\n    \
             camdl fit summary <run> --params-only > params.toml\n    \
             camdl simulate <model> --params params.toml --obs-only-dir out/\n  \
             (A labelled plug-in predictive is a future cell; v1 emits posterior bands only.)"
        );
    }
    // Neither a posterior sampler nor an optimizer (a likelihood-eval stage such
    // as pfilter, or an unrecognized method): this stage kind yields no posterior
    // distribution, so there is nothing to band — but it is NOT an optimizer fit.
    let m = method.map(|m| m.as_str()).unwrap_or("this stage");
    format!(
        "stage '{stage}' ({m}) produced no posterior draws and is not a posterior \
         sampler, so there is no band to draw for it."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_convergence_finds_mh_summary_for_mh_method() {
        // An mh(-ODE) stage writes `mh_summary.json` (NOT `pmmh_summary.json`),
        // so read_convergence must resolve it via the algorithm's own name. On
        // the pre-fix code — which searched only pgas/pmmh — this returned
        // NotAssessed, silently dropping the mh stage's convergence.
        let dir = std::env::temp_dir().join("camdl_read_convergence_mh_summary_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("mh_summary.json"),
            r#"{"rhat": {"beta": 1.03}, "ess": {"beta": 250.0}}"#,
        )
        .unwrap();

        match read_convergence(&dir, Some(FitAlgorithm::Mh)) {
            ConvergenceStatus::Reported { rhat_max, ess_min } => {
                assert!((rhat_max - 1.03).abs() < 1e-9, "rhat_max from mh_summary.json");
                assert!((ess_min - 250.0).abs() < 1e-9, "ess_min from mh_summary.json");
            }
            _ => panic!("mh stage must resolve its own mh_summary.json, not NotAssessed"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_convergence_clean_break_no_pmmh_fallback() {
        // Clean break: a pre-rename mh fit (only `pmmh_summary.json` on disk) is
        // NOT silently read under the old name — it is NotAssessed, so the fit
        // must be re-run. Asserts the cross-name fallback was removed.
        let dir = std::env::temp_dir().join("camdl_read_convergence_clean_break_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("pmmh_summary.json"),
            r#"{"rhat": {"beta": 1.03}, "ess": {"beta": 250.0}}"#,
        )
        .unwrap();
        assert!(
            matches!(
                read_convergence(&dir, Some(FitAlgorithm::Mh)),
                ConvergenceStatus::NotAssessed
            ),
            "an mh fit must not fall back to pmmh_summary.json (clean break)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

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

    /// A cloud of `n` draws tagged by a single `id` column = its cloud index, so
    /// the subsample's chosen indices are recoverable from the returned draws.
    fn cloud(n: usize) -> Vec<IndexMap<String, f64>> {
        (0..n).map(|i| IndexMap::from([("id".to_string(), i as f64)])).collect()
    }

    #[test]
    fn subsample_draws_is_strided_not_front_biased() {
        // 80 draws capped to 10: the pick must SPAN the whole cloud (stride 8),
        // not `take(10)` of the front (which would return ids 0..10 — the bias
        // this guards against: early sweeps / a single chain).
        let c = cloud(80);
        let chosen: Vec<usize> =
            subsample_draws(&c, 10).iter().map(|d| d["id"] as usize).collect();
        assert_eq!(chosen, vec![0, 8, 16, 24, 32, 40, 48, 56, 64, 72]);
        assert_ne!(chosen, (0..10).collect::<Vec<_>>(), "must not be a front-take");
        // The span reaches the tail third of the cloud (front-take never would).
        assert!(*chosen.last().unwrap() >= 2 * 80 / 3, "subsample must reach the cloud tail");
    }

    #[test]
    fn subsample_draws_caps_and_floors() {
        // cap ≥ len → the whole cloud, in order.
        let c = cloud(5);
        assert_eq!(subsample_draws(&c, 200).len(), 5);
        assert_eq!(subsample_draws(&c, 5).len(), 5);
        // cap < len → exactly cap draws.
        assert_eq!(subsample_draws(&c, 3).len(), 3);
        // cap 0 floors to one draw (an empty band is unrepresentable).
        assert_eq!(subsample_draws(&c, 0).len(), 1);
    }

    #[test]
    fn default_predict_draws_is_200() {
        // Both horizons default to this cap; pin it so a silent drift is a test
        // failure, not a surprise 9-hour predict.
        assert_eq!(DEFAULT_PREDICT_DRAWS, 200);
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
    fn refusal_for_a_posterior_sampler_is_not_an_optimizer_message() {
        // gh#343: a Bayesian sampler stage (PGAS / PMMH / MH) that reaches the
        // refusal — its draws.tsv is absent (incomplete / still running) — must
        // NOT be called an "optimizer fit". PGAS is the *default* Bayesian path;
        // the misclassification made the blessed posterior-predictive verb
        // unusable for it. The refusal must name the missing draws, never the
        // plug-in-simulate optimizer workflow (which discards the posterior).
        for m in [FitAlgorithm::Pgas, FitAlgorithm::Pmmh, FitAlgorithm::Mh] {
            let msg = plugin_refusal(Some(m), "posterior");
            assert!(
                !msg.contains("optimizer"),
                "a {m} stage must not be described as an optimizer fit; got: {msg}"
            );
            assert!(
                msg.contains("draws.tsv"),
                "the refusal should name the missing posterior draws (draws.tsv); got: {msg}"
            );
            assert!(
                !msg.contains("--params-only"),
                "a sampler is not redirected to the plug-in point-estimate workflow; got: {msg}"
            );
        }
    }

    #[test]
    fn refusal_for_an_optimizer_still_names_the_plug_in_workflow() {
        // The optimizer refusal is preserved: IF2 / NLopt genuinely have no band,
        // so the plug-in-simulate workflow is the right redirect (no over-correction).
        for m in [FitAlgorithm::If2, FitAlgorithm::NlSbplx, FitAlgorithm::NlBobyqa] {
            let msg = plugin_refusal(Some(m), "scout");
            assert!(msg.contains("optimizer fit"), "a {m} stage is an optimizer fit; got: {msg}");
            assert!(
                msg.contains("--params-only"),
                "the optimizer refusal points at the plug-in workflow; got: {msg}"
            );
        }
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
                scenario: "fitted".to_string(),
                sweep: Vec::new(),
                horizon: Horizon::FreeForward,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::Reported { rhat_max: 1.01, ess_min: 420.0 },
                n_draws: 40,
                rows: &stream.rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0],
            "scenario\ttime\tpatch\thorizon\ttreatment\trhat_max\tess_min\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "fitted\t7\tBo\tfree_forward\tposterior\t1.0100\t420\t40\t0\t1\t3\t6\t12");
        assert_eq!(lines[2], "fitted\t7\tBombali\tfree_forward\tposterior\t1.0100\t420\t40\t0\t0\t1\t3\t7");
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
                scenario: "fitted".to_string(),
                sweep: Vec::new(),
                horizon: Horizon::FreeForward,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::NotAssessed,
                n_draws: 12,
                rows: &stream.rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        // No dim column; not-assessed rhat/ess are empty cells, not fabricated
        // values; n_draws is still carried. Scenario leads.
        assert_eq!(lines[0], "scenario\ttime\thorizon\ttreatment\trhat_max\tess_min\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "fitted\t1\tfree_forward\tposterior\t\t\t12\t1\t2\t3\t4\t5");
    }

    #[test]
    fn predictive_tsv_sweep_columns_lead_after_scenario_and_one_step_is_blank() {
        // Two free-forward sweep cells (k=8, k=12) plus a sweep-agnostic one-step
        // section: the `sweep:k` column follows `scenario`, free-forward rows carry
        // the cell's swept value, and the one-step rows leave it blank.
        let conv = ConvergenceStatus::Reported { rhat_max: 1.0, ess_min: 100.0 };
        let rows = vec![BandRow { time: 7.0, stratum: vec![], quantiles: vec![1.0, 2.0, 3.0, 4.0, 5.0] }];
        let tsv = render_predictive_tsv_sections(
            &[],
            &[
                PredictiveSection {
                    scenario: "fitted".to_string(),
                    sweep: vec![("k".to_string(), 8.0)],
                    horizon: Horizon::FreeForward,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
                PredictiveSection {
                    scenario: "fitted".to_string(),
                    sweep: vec![("k".to_string(), 12.0)],
                    horizon: Horizon::FreeForward,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
                PredictiveSection {
                    scenario: "fitted".to_string(),
                    sweep: Vec::new(), // one-step is sweep-agnostic
                    horizon: Horizon::OneStepAhead,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
            ],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(
            lines[0],
            "scenario\tsweep:k\ttime\thorizon\ttreatment\trhat_max\tess_min\tn_draws\tq05\tq25\tq50\tq75\tq95",
            "sweep:k column follows scenario"
        );
        assert_eq!(lines[1], "fitted\t8\t7\tfree_forward\tposterior\t1.0000\t100\t10\t1\t2\t3\t4\t5");
        assert_eq!(lines[2], "fitted\t12\t7\tfree_forward\tposterior\t1.0000\t100\t10\t1\t2\t3\t4\t5");
        // One-step row: the sweep:k cell is blank (empty), not a fabricated value.
        assert_eq!(lines[3], "fitted\t\t7\tone_step\tposterior\t1.0000\t100\t10\t1\t2\t3\t4\t5");
    }

    #[test]
    fn expand_predict_sweep_empty_is_one_null_cell_and_cartesian_is_sorted() {
        use crate::args::types::{Grid, SweepSpec};
        // No specs → exactly one empty cell (the no-sweep single-pass path).
        assert_eq!(expand_predict_sweep(&[]), vec![Vec::<(String, f64)>::new()]);
        // Two params → Cartesian product, each cell sorted by param name.
        let specs = vec![
            SweepSpec { name: "rho".to_string(), grid: Grid::List(vec![0.3, 0.5]) },
            SweepSpec { name: "k".to_string(), grid: Grid::List(vec![8.0]) },
        ];
        let cells = expand_predict_sweep(&specs);
        assert_eq!(
            cells,
            vec![
                vec![("k".to_string(), 8.0), ("rho".to_string(), 0.3)],
                vec![("k".to_string(), 8.0), ("rho".to_string(), 0.5)],
            ],
            "k sorts before rho; the grid is the Cartesian product"
        );
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
