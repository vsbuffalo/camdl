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

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::quantile::{band, fmt_time, fmt_value, QUANTILE_LEVELS};

use crate::chain_selection::{warn_active_selection, ChainSelection, SubsetInfo};
use crate::posterior_draws;
use crate::fit::method_result::{MaxRhat, PosteriorDiagnostics};
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

/// Which chains a band was drawn from — the `chain` column `--by-chain` adds
/// (gh#794).
///
/// The pooled band is the default and stays first-class; a per-chain band is an
/// addition beside it, tagged rather than filed separately, the same way
/// `--scenario` tags its arms with a leading `scenario` column and `--sweep`
/// with `sweep:<param>`. So a run with `--by-chain` is more rows in the same
/// file, never a second file tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainLabel {
    /// Every retained chain, pooled — what the file carries with no
    /// `--by-chain`, and what a reader who does nothing gets.
    All,
    /// One chain's draws alone. Carried 0-based; rendered 1-based to match the
    /// `chain_N/` directories, the `fit summary` per-chain table and
    /// `--exclude-chains`.
    One(usize),
}

impl ChainLabel {
    /// The value written into the `chain` column.
    pub fn as_cell(&self) -> String {
        match self {
            ChainLabel::All => "all".to_string(),
            ChainLabel::One(c) => (c + 1).to_string(),
        }
    }
}

/// The producing stage's own convergence numbers, copied from its summary, so
/// a band is never silent about whether the fit behind it settled. It records
/// the number; it does not reject a band on it (the refusal policy is the
/// deferred guardrail).
///
/// This is **provenance about the fit**, not a statement about any row: it is
/// the worst parameter's R̂ over the whole stage, repeated identically on every
/// row of every stream. That is why the columns it writes are named
/// `fit_rhat_max` / `fit_ess_min` (gh#794) — under their former names
/// `rhat_max` / `ess_min` they read as the R̂ of the predicted value beside
/// them, which they are not. The columns that do describe the row are
/// [`BandRow::mean_conv`] and [`BandRow::pred_conv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceStatus {
    /// The stage reported a Gelman–Rubin R̂ / ESS summary.
    Reported { rhat_max: f64, ess_min: f64 },
    /// No per-stage R̂ available (e.g. a single-chain stage).
    NotAssessed,
}

impl ConvergenceStatus {
    /// The value written into the `fit_rhat_max` column — the empty string when
    /// not assessed, so a consumer can tell "converged at 1.01" from "unknown".
    pub fn fit_rhat_max_cell(&self) -> String {
        match self {
            ConvergenceStatus::Reported { rhat_max, .. } => format!("{rhat_max:.4}"),
            ConvergenceStatus::NotAssessed => String::new(),
        }
    }

    /// The value written into the `fit_ess_min` column — empty when not
    /// assessed or when no finite ESS was reported (a single-chain / ESS-less
    /// summary).
    pub fn fit_ess_min_cell(&self) -> String {
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
    /// Each draw's posterior key + the stage it came from — `None` when the
    /// cloud was built without them (the test constructors). See [`DrawKeys`].
    keys: Option<DrawKeys>,
}

/// Each posterior draw's `(chain, draw)` key and the stage directory its saved
/// smoothing path would live in (gh#722).
///
/// The key travels WITH the cloud rather than being re-read beside it, because
/// a `value_at(..., last_obs)` has to read the path belonging to the draw it is
/// reported for: a shuffled or offset pairing still bands plausibly, and the
/// band is the thing a situation report quotes.
#[derive(Debug, Clone)]
pub struct DrawKeys {
    /// The stage directory holding `chain_<N+1>/trajectories.tsv`.
    pub stage_dir: PathBuf,
    /// Per draw, in cloud order: its `(chain, draw)` key, or `None` for a
    /// param-only `draws.tsv` with no key columns.
    pub per_draw: Vec<Option<(usize, usize)>>,
}

/// Which draws actually have a smoothing path ON DISK — [`DrawKeys`] narrowed
/// to the forkable subset, with its size, so the count reported and the paths
/// openable are the same fact.
#[derive(Debug, Clone)]
pub struct SavedPaths {
    pub stage_dir: PathBuf,
    /// Per draw, in cloud order: the key of its saved path, `None` when that
    /// draw has none. The trajectory save stride and `thin` need not agree, so
    /// this is a genuine subset (gh#727).
    pub per_draw: Vec<Option<(usize, usize)>>,
    /// Draws with a saved path, over `per_draw.len()`.
    pub n_saved: usize,
}

impl DrawKeys {
    /// Intersect the cloud's keys with the keys that actually have a saved
    /// path. One `read_to_string` pass per `chain_*/trajectories.tsv`, so this
    /// runs only when a quantity needs a conditioned read — a `fit predict`
    /// with no in-window `value_at` pays nothing.
    pub fn resolve_saved(&self) -> SavedPaths {
        let on_disk = crate::fit::joint::trajectory_keys(&self.stage_dir);
        let per_draw: Vec<Option<(usize, usize)>> = self
            .per_draw
            .iter()
            .map(|k| k.filter(|key| on_disk.contains(key)))
            .collect();
        let n_saved = per_draw.iter().filter(|k| k.is_some()).count();
        SavedPaths { stage_dir: self.stage_dir.clone(), per_draw, n_saved }
    }
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
        Ok(PosteriorDraws {
            draws,
            stage,
            method,
            backend,
            convergence,
            selection: None,
            keys: None,
        })
    }

    /// Attach the chain-selection provenance (`--exclude-chains`) to the cloud.
    pub fn with_selection_info(mut self, info: Option<SubsetInfo>) -> Self {
        self.selection = info;
        self
    }

    /// Attach the per-draw posterior keys. Rejects a list that is not 1:1 with
    /// the cloud — an off-by-one here pairs draw *i*'s parameters with draw
    /// *i+1*'s inferred state and still produces a plausible band.
    pub fn with_keys(mut self, keys: DrawKeys) -> Result<Self, String> {
        if keys.per_draw.len() != self.draws.len() {
            return Err(format!(
                "internal: {} posterior keys for {} draws — the key list must be 1:1 \
                 with the cloud",
                keys.per_draw.len(),
                self.draws.len()
            ));
        }
        self.keys = Some(keys);
        Ok(self)
    }

    /// Each draw's `(chain, draw)` key + its stage dir, when the cloud carries
    /// them.
    pub fn keys(&self) -> Option<&DrawKeys> {
        self.keys.as_ref()
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
    /// One row per `(time, stratum)`, pooled over every retained chain — the
    /// default band, and the only one without `--by-chain`.
    pub rows: Vec<BandRow>,
    /// `--by-chain` only: the same rows banded from ONE chain's draws, keyed by
    /// 0-based chain id and ordered by it. EMPTY without the flag, so the
    /// rendered file is byte-identical to the pooled-only path (gh#794).
    ///
    /// A per-chain band carries no `rhat_*` / `ess_*` cell: those compare
    /// chains, and one chain has nothing to compare against. The pooled row is
    /// where the comparison lives.
    pub per_chain: Vec<(usize, Vec<BandRow>)>,
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
    /// R̂ / bulk-ESS of *this cell's* **latent expected value** across chains —
    /// "do the chains agree about the expected trajectory here?" (gh#794). The
    /// diagnostic to act on: it is the one an overdispersed observation model
    /// cannot dilute. `None` when the reduction was refused (see
    /// [`crate::fit::row_convergence::row_convergence`]).
    pub mean_conv: Option<crate::fit::row_convergence::RowConvergence>,
    /// R̂ / bulk-ESS of *this cell's* **predictive draws** across chains — "do the
    /// chains give the same predictive distribution?". Legitimate when the
    /// reported interval is genuinely dominated by irreducible observation
    /// noise, and the weaker of the two: the noise lands in the within-chain
    /// variance and pulls it toward 1.
    pub pred_conv: Option<crate::fit::row_convergence::RowConvergence>,
}

/// One observed series cell: a time, its stratum, the recorded value (`None`
/// for a hole — a scheduled-but-missing observation).
#[derive(Debug, Clone)]
pub struct ObservedRow {
    pub time: f64,
    pub stratum: Vec<(String, String)>,
    pub value: Option<f64>,
}


// ── Rendering the tidy artifact ────────────────────────────────────────────

/// One (scenario, horizon) contribution to a `predictive/<stream>.tsv`: its rows,
/// plus the labels they carry (the scenario overlay axis, the horizon axis, the
/// treatment, the convergence, the draw count). Stacking several of these under
/// one header is how `fit predict` writes every scenario's `free_forward` rows
/// plus the `one_step` rows into the same file — the `scenario` and `horizon`
/// columns distinguish them.
pub struct PredictiveSection<'a> {
    /// Which chains this section's band was drawn from (gh#794). The `chain`
    /// column is emitted only when some section carries a
    /// [`ChainLabel::One`] — so a run without `--by-chain` has no such column
    /// and renders byte-identically, exactly as `sweep:<param>` appears only
    /// under `--sweep`.
    pub chain: ChainLabel,
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
/// treatment | fit_rhat_max | fit_ess_min | rhat_mean | ess_mean | rhat_pred |
/// ess_pred |
/// n_draws | q05 … q95`. Tidy, plot-ready; the axes and the convergence channels
/// are columns so a new predictive cell — a new scenario, a new horizon, a new
/// treatment — is more rows, never new consumer code. Several
/// [`PredictiveSection`]s (one per (scenario, horizon)) stack under the single
/// header.
///
/// Two convergence channels sit side by side and must not be confused (gh#794).
/// `fit_rhat_max`/`fit_ess_min` are the *producing stage's* worst-parameter numbers,
/// constant down the file — provenance. `rhat_mean`/`ess_mean` and
/// `rhat_pred`/`ess_pred` describe *this row*: the first over the latent expected
/// value, the second over the predictive draws.
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

    // The `chain` column exists only when a per-chain band is present
    // (`--by-chain`). Data-driven, like `sweep_names` above: no flag plumbed
    // into the renderer, and no column when there is nothing to put in it.
    let by_chain = sections.iter().any(|s| s.chain != ChainLabel::All);

    let mut out = String::new();
    // Header — `chain` leads when present (the outermost partition of the
    // draws: every other coordinate is nested inside it), then `scenario` (the
    // overlay axis), then the `sweep:<param>` columns, then everything else.
    if by_chain {
        out.push_str("chain\t");
    }
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
    out.push_str(
        "\thorizon\ttreatment\tfit_rhat_max\tfit_ess_min\
         \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws",
    );
    for (_, label) in QUANTILE_LEVELS {
        out.push('\t');
        out.push_str(label);
    }
    out.push('\n');

    for section in sections {
        let rhat = section.convergence.fit_rhat_max_cell();
        let ess = section.convergence.fit_ess_min_cell();
        let n = section.n_draws.to_string();
        let chain_cell = section.chain.as_cell();
        for row in section.rows {
            if by_chain {
                out.push_str(&chain_cell);
                out.push('\t');
            }
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
            // The per-row channels (gh#794): the mean one first, because it is
            // the one to act on.
            use crate::fit::row_convergence::RowConvergence;
            out.push('\t');
            out.push_str(&RowConvergence::rhat_cell(row.mean_conv.as_ref()));
            out.push('\t');
            out.push_str(&RowConvergence::ess_cell(row.mean_conv.as_ref()));
            out.push('\t');
            out.push_str(&RowConvergence::rhat_cell(row.pred_conv.as_ref()));
            out.push('\t');
            out.push_str(&RowConvergence::ess_cell(row.pred_conv.as_ref()));
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
                // Keyed, not param-only: the `(chain, draw)` key locates the
                // draw's saved smoothing path, and building both out of ONE
                // pass over the same rows is what keeps them paired (gh#722).
                let (rows, sel_info) = pref.load_keyed_with_info()?;
                let mut draws: Vec<IndexMap<String, f64>> = Vec::with_capacity(rows.len());
                let mut keys: Vec<Option<(usize, usize)>> = Vec::with_capacity(rows.len());
                for r in rows {
                    keys.push(match (r.chain, r.draw) {
                        (Some(c), Some(d)) => Some((c, d)),
                        _ => None,
                    });
                    draws.push(r.params.into_iter().collect());
                }
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
                    Some(sel) => subset_convergence(segment, &pref.draws_path, sel)?,
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
                    .with_selection_info(sel_info)
                    .with_keys(DrawKeys { stage_dir, per_draw: keys })?,
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
/// `max` over its R̂ map, and the min-parameter ESS **through the shared
/// classification** ([`min_ess_over`]) rather than a bare fold. Returns
/// [`ConvergenceStatus::NotAssessed`] when no summary or no R̂ is present (a
/// single-chain stage), so a band is never silently "converged".
fn read_convergence(stage_dir: &Path, method: Option<FitAlgorithm>) -> ConvergenceStatus {
    let try_read = |name: &str| -> Option<ConvergenceStatus> {
        let bytes = std::fs::read(stage_dir.join(name)).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        // Reduce through the SAME classification `fit summary` uses, over the
        // same per-parameter type, read by the same reader, so a band and the
        // summary cannot disagree about one fit (gh#409). A stored summary is
        // just another producer of that map.
        let diag = PosteriorDiagnostics {
            per_param: crate::fit::method_result::ConvergenceMaps::read(&v).per_param(),
            n_samples: 0,
            thin: 1,
            wall_time_secs: None,
            n_chains: 0,
        };
        match diag.max_rhat_status() {
            MaxRhat::Reported(rhat_max) => Some(ConvergenceStatus::Reported {
                rhat_max,
                ess_min: diag.min_ess().unwrap_or(f64::NAN),
            }),
            _ => None,
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
    segment: &Path,
    draws_path: &Path,
    selection: &ChainSelection,
) -> Result<ConvergenceStatus, String> {
    // The ESTIMATED parameter names, from the fit's own sidecar. `draws.tsv`
    // also carries the model's pinned parameters, constant by construction, and
    // scoring those and dropping the non-finite results is what hid a FROZEN
    // estimated parameter behind the same filter that legitimately hides a
    // pinned one. Without the sidecar the two cannot be told apart, so the band
    // says "not assessed" rather than inventing a verdict.
    let Some(estimated) = crate::run_meta::read_fit_sidecar(segment).map(|s| s.estimated) else {
        return Ok(ConvergenceStatus::NotAssessed);
    };
    let sub =
        crate::chain_selection::recompute_subset_diagnostics(draws_path, selection, &estimated)?;

    // Reduce through the SAME classification `fit summary` uses, so the two
    // cannot disagree about one fit + selection (gh#409). A band whose R̂ is
    // unassessable — a sampler failure, not a missing number — carries no
    // number rather than the max over the parameters that happened to work.
    let diag = crate::fit::method_result::PosteriorDiagnostics {
        per_param: sub.per_param.clone(),
        n_samples: sub.n_samples,
        thin: 1,
        wall_time_secs: None,
        n_chains: sub.n_chains,
    };
    match diag.max_rhat_status() {
        crate::fit::method_result::MaxRhat::Reported(rhat_max) => {
            Ok(ConvergenceStatus::Reported {
                rhat_max,
                ess_min: diag.min_ess().unwrap_or(f64::NAN),
            })
        }
        _ => Ok(ConvergenceStatus::NotAssessed),
    }
}

// ── The engine sink: sample y_rep per draw on the emission grid ────────────

/// One scenario's accumulated free-forward output: the per-`(leaf, time)`
/// `y_rep` samples across that scenario's draws, plus the per-draw quantity
/// values and the trajectory snapshot grid. The engine runs every draw of one
/// scenario before the next (scenario is the outermost loop), so a cell merges
/// into its scenario's accumulator keyed by [`crate::engine::CellSpec::scenario`].
struct ScenarioAccum {
    /// `samples[leaf][time_idx]` = the `y_rep` values across this scenario's draws.
    samples: Vec<Vec<Vec<f64>>>,
    /// `means[leaf][time_idx]` = `E[y | x_t, θ]` for the *same* draws, in the same
    /// order — the latent expected value the observation distribution is centred
    /// on, before observation noise. The operand of `rhat_mean` (gh#794): an R̂
    /// over `samples` buries the chains' disagreement in the observation noise.
    means: Vec<Vec<Vec<f64>>>,
    /// The chain each accumulated draw came from, pushed once per merged cell so
    /// it is positionally aligned with `samples`/`means`/`quant_draws` by
    /// construction rather than by re-deriving the subsample stride (gh#794).
    draw_chain: crate::fit::row_convergence::ChainOfDraw,
    /// One inner `Vec` per draw: each quantity leaf's value, in
    /// `model.quantities` order. Empty when the model declares no quantities.
    quant_draws: Vec<Vec<sim::quantity::QuantityResult>>,
    /// The trajectory snapshot times, captured once per scenario (every draw
    /// shares the output cadence) — the time axis a series quantity bands against.
    quant_times: Vec<f64>,
    /// Draws in this scenario whose saved smoothing path was opened — the
    /// denominator an in-window `value_at` band is actually over (gh#722). The
    /// forkable subset is REPORTED, never silently substituted.
    n_conditioned: usize,
}

/// Does this design cell read its in-window `value_at` quantities off the
/// smoothing path (gh#722)? The ONE rule — the sink construction and the
/// manifest tag both fold through it, so the artifact cannot claim one object
/// while the number came from another.
///
/// Three conditions: the fit carries per-draw path locators at all; the cell is
/// the arm the smoothing path belongs to (the no-overlay `fitted` arm — see
/// [`ConditionedSource::scenario`]); and no sweep value overrides a parameter,
/// which would make the cell replay a different model too.
///
/// Deliberately NOT conditioned on `n_saved > 0`: a fit that saved no path
/// still routes, and every draw then censors. That is the loud, empty band the
/// caller has already announced — not a quiet fall back to the replay.
fn conditioned_here(
    saved: &Option<SavedPaths>,
    conditioned_scenario: &Option<String>,
    sweep_pt: &[(String, f64)],
    scenario: &str,
) -> bool {
    saved.is_some()
        && conditioned_scenario.as_deref() == Some(scenario)
        && sweep_pt.is_empty()
}

/// What the free-forward sink reads a [`sim::quantity::QuantityPath::Smoothed`]
/// quantity off (gh#722). `None` on the sink ⇒ no conditioned read is in play
/// for this design cell and every quantity folds the replay, unchanged.
struct ConditionedSource {
    /// The scenario whose cells read conditioned. Only the no-overlay `fitted`
    /// arm qualifies: a scenario or a sweep point replays a DIFFERENT model,
    /// and the smoothing path was inferred under the fitted one — there is no
    /// conditioned path for a counterfactual, and reusing the fitted one would
    /// print the same number under every arm.
    scenario: String,
    /// Per free-forward draw index (`CellSpec::point_idx`, which indexes the
    /// subsample the engine was handed): the `(chain, draw)` key of its saved
    /// path, `None` when that draw is outside the forkable subset.
    per_draw: Vec<Option<(usize, usize)>>,
    /// Every path `per_draw` names, keyed by `(chain, draw)`, read up front —
    /// ONE pass per `chain_<N+1>/trajectories.tsv`, not one per draw (see
    /// [`ConditionedSource::load`]).
    paths: std::collections::HashMap<(usize, usize), sim::Trajectory>,
}

impl ConditionedSource {
    /// Read every saved path this design cell will need, one pass per chain
    /// file.
    ///
    /// The reporting configuration is hundreds of posterior draws over a
    /// handful of chains, and a `trajectories.tsv` at substep resolution is
    /// large. Reading it per draw is `n_draws × file_size`: 300 draws over a
    /// 190 MB fixture took 35 s of pure re-scanning. Grouping the keys by the
    /// file that holds them makes it `n_chains × file_size`.
    ///
    /// The cost is peak memory: the forkable subset's paths are resident at
    /// once rather than one at a time. That is the same shape as what the
    /// engine already does — `run_job` collects every cell's forward trajectory
    /// before the merge phase — but not the same size, because a saved
    /// smoothing path is at SUBSTEP resolution while a forward trajectory is at
    /// the output cadence. Order of magnitude: the parsed paths run about
    /// twice the on-disk size of the `trajectories.tsv` files they came from.
    /// If that ever binds, the fix is to stream chain by chain (the merge phase
    /// is in canonical `point_idx` order, and a PGAS cloud is chain-major), not
    /// to go back to reading the file once per draw.
    fn load(
        scenario: String,
        per_draw: Vec<Option<(usize, usize)>>,
        stage_dir: &Path,
        model: &ir::Model,
    ) -> Result<Self, String> {
        // No incidence columns: only STATE quantities route to the smoothing
        // path, and requiring `inc_<stream>` here would refuse a file that has
        // none.
        let columns = io::trajectories::TrajColumnSpec::from_model(model, &[]);
        // Group the keys by the file that holds them. The on-disk chain dir is
        // 1-based (`chain_{N+1}`); the in-file `chain` column is the 0-based key.
        let mut by_chain: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for key in per_draw.iter().flatten() {
            by_chain.entry(key.0).or_default().push(*key);
        }
        let mut paths = std::collections::HashMap::new();
        for (chain, keys) in by_chain {
            let traj_path =
                stage_dir.join(format!("chain_{}", chain + 1)).join("trajectories.tsv");
            let read = io::trajectories::read_trajectories(&traj_path, &columns, &keys)
                .map_err(|e| {
                    format!(
                        "reading the smoothing paths of chain {chain} to evaluate a \
                         quantity anchored inside the observed record: {e}"
                    )
                })?;
            // The `inc_<stream>` sidecar is empty by construction here (no
            // incidence columns requested); the state path is the operand.
            paths.extend(read.into_iter().map(|(k, (traj, _inc))| (k, traj)));
        }
        Ok(ConditionedSource { scenario, per_draw, paths })
    }
}

/// A [`RunSink`] that samples `y_rep` for every fit leaf on its emission grid,
/// for each draw (= cell), accumulating per `(scenario, leaf, time)` across
/// draws. The quantile reduction runs per scenario after all cells merge.
struct PredictiveSink {
    compiled: std::sync::Arc<sim::CompiledModel>,
    /// Per leaf (in `model.observations` order): the times to emit at — the
    /// leaf's observation times, then (gh#696) the forecast times continuing
    /// its cadence to the model horizon. NOT the observed-data axis: `last_obs`
    /// anchors are resolved off the observation times alone.
    leaf_times: Vec<Vec<f64>>,
    /// Per leaf, per time (aligned with the OBSERVED prefix of `leaf_times`,
    /// which is why the forecast tail is only ever built for a leaf whose
    /// likelihood reads no data column): the observed auxiliary
    /// columns carried forward into the predictive draw (a data-supplied
    /// binomial denominator `n = n_examined`). Empty inner vec = no aux at that
    /// obs time (the likelihood's denominator then resolves data-free).
    leaf_aux: Vec<Vec<Vec<(String, f64)>>>,
    /// Per leaf: the fit's `condition_from` boundary, where it has one — the
    /// time the likelihood resets that stream's incidence accumulator at, and
    /// therefore where the FIRST emitted bin opens (gh#702). `None` for a
    /// stream with no conditioning; ignored by a prevalence projection.
    leaf_window_start: Vec<Option<f64>>,
    /// The generated-quantities evaluator, `Some` iff the model declares a
    /// `quantities {}` block. Composed alongside the obs-sample accumulator (same
    /// draw, same params) — not a second [`RunSink`]. Held behind an `Arc` so a
    /// fresh sink per sweep-point shares the one evaluator without rebuilding it.
    quant_eval: Option<std::sync::Arc<sim::quantity::QuantityEvaluator>>,
    /// The run's resolved observation window — the two ends of `leaf_times`,
    /// folded once by the caller so every consumer in this command (here and the
    /// contrast reducer) anchors `value_at` to the same pair.
    obs_anchors: Option<sim::quantity::ObsAnchorTimes>,
    /// Where an in-window `value_at` is read from (gh#722). `Some` only when
    /// the model declares one AND this sink's design cell is the no-overlay
    /// fitted arm.
    conditioned: Option<ConditionedSource>,
    /// Per free-forward draw index (`CellSpec::point_idx`, which indexes the
    /// subsample the engine was handed): the chain that draw came from, `None`
    /// when the cloud's `draws.tsv` carries no chain column. The partition every
    /// per-row R̂ reduces over (gh#794).
    chain_of_point: Vec<Option<usize>>,
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
            means: leaf_times.iter().map(|ts| vec![Vec::new(); ts.len()]).collect(),
            draw_chain: crate::fit::row_convergence::ChainOfDraw::default(),
            quant_draws: Vec::new(),
            quant_times: Vec::new(),
            n_conditioned: 0,
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
        // The *same* draws' latent expected values, `E[y | x_t, θ]` before
        // observation noise (gh#794). Accumulated beside `leaf_y` so the two are
        // the same draw at the same time with the same resolved parameters.
        let mut leaf_mean: Vec<Vec<f64>> = vec![Vec::new(); model.observations.len()];
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
            // Consumes no RNG, so it cannot perturb the paired-seed replay the
            // sampler above drives.
            let meaner = sim::inference::obs_model::compile_obs_mean_pf(
                obs_ir,
                self.compiled.clone(),
                &params,
            );
            // The first bin opens where the LIKELIHOOD opened it — at this
            // stream's conditioning boundary, not at the model origin (gh#702).
            let projected = crate::project_all_obs_times(
                &cell.traj, obs_ir, model, times, self.leaf_window_start[si],
            )?;
            let leaf_aux = &self.leaf_aux[si];
            let mut stream_vals: Vec<f64> = Vec::with_capacity(times.len());
            let mut stream_means: Vec<f64> = Vec::with_capacity(times.len());
            for (ti, &t) in times.iter().enumerate() {
                let snap = crate::snap_at(&cell.traj, t);
                // Carry the OBSERVED aux (survey denominator) at this obs time
                // into the draw; `&[]` when the leaf has no aux (aligned 1:1 with
                // `times`, so a wrong index is impossible).
                let aux: &[(String, f64)] = leaf_aux.get(ti).map(|v| v.as_slice()).unwrap_or(&[]);
                let y = sampler(projected[ti], t, &snap.int_state.counts, aux, &mut obs_rng);
                stream_vals.push(y);
                stream_means.push(meaner(projected[ti], t, &snap.int_state.counts, aux));
            }
            if want_obs {
                // Key by the stream's declared `name` — what `observations.<name>`
                // in the DSL references (v1.1 is unstratified, so name == base).
                obs_set.streams.insert(obs_ir.name.clone(), (times.clone(), stream_vals.clone()));
            }
            leaf_y[si] = stream_vals;
            leaf_mean[si] = stream_means;
        }

        // gh#722: a quantity anchored at or before `last_obs` is read off this
        // draw's CONDITIONED smoothing path, not off the replay above. Only the
        // no-overlay fitted arm has one (see [`ConditionedSource::scenario`]),
        // and only the forkable subset of draws — a draw outside it censors
        // rather than silently borrowing the replay's answer. The paths were
        // all read up front (`ConditionedSource::load`), so this is a lookup,
        // not a scan of the chain's `trajectories.tsv`.
        let conditioned_path: Option<&sim::Trajectory> = match &self.conditioned {
            Some(src) if src.scenario == scenario_name => {
                match src.per_draw.get(cell.spec.point_idx).copied().flatten() {
                    Some((chain, draw)) => Some(src.paths.get(&(chain, draw)).ok_or_else(
                        || {
                            format!(
                                "internal: the smoothing path for (chain {chain}, draw \
                                 {draw}) was not among the paths read for this run — the \
                                 preloaded set must cover every keyed draw"
                            )
                        },
                    )?),
                    None => None,
                }
            }
            _ => None,
        };
        let conditioned = match (&self.conditioned, &conditioned_path) {
            (Some(src), _) if src.scenario != scenario_name => {
                sim::quantity::ConditionedRead::Off
            }
            (Some(_), Some(p)) => sim::quantity::ConditionedRead::Saved(p),
            (Some(_), None) => sim::quantity::ConditionedRead::NotSaved,
            (None, _) => sim::quantity::ConditionedRead::Off,
        };
        let read_conditioned = matches!(conditioned, sim::quantity::ConditionedRead::Saved(_));

        // Generated quantities: fold this draw's trajectory + the just-drawn y_sim
        // into its per-quantity values, using the SAME resolved params + draws as
        // the predictive output above. The `value_at` anchors are the two ends of
        // the OBSERVED data axis — the min and max over the leaves' observation
        // times, which predict carries for the predicted-vs-observed join.
        let quant_results = self.quant_eval.as_ref().map(|eval| {
            eval.eval_draw(
                &params,
                &cell.traj,
                conditioned,
                &self.compiled,
                Some(&obs_set),
                self.obs_anchors,
            )
        });
        let snapshot_times: Vec<f64> = if quant_results.is_some() {
            cell.traj.snapshots.iter().map(|s| s.t).collect()
        } else {
            Vec::new()
        };

        // Commit into the cell's scenario accumulator.
        let chain_of_this_draw =
            self.chain_of_point.get(cell.spec.point_idx).copied().flatten();
        let acc = self.accum_for(&scenario_name);
        acc.draw_chain.0.push(chain_of_this_draw);
        for (si, ys) in leaf_y.into_iter().enumerate() {
            for (ti, y) in ys.into_iter().enumerate() {
                acc.samples[si][ti].push(y);
            }
        }
        for (si, ms) in leaf_mean.into_iter().enumerate() {
            for (ti, m) in ms.into_iter().enumerate() {
                acc.means[si][ti].push(m);
            }
        }
        if let Some(results) = quant_results {
            if acc.quant_times.is_empty() {
                acc.quant_times = snapshot_times;
            }
            acc.quant_draws.push(results);
            if read_conditioned {
                acc.n_conditioned += 1;
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
    let fit_result = FitResult::resolve(&segment, args.stage.as_deref(), selection.clone())?;
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
    let (mut model, _) = crate::util::load_model(&compiled_ir)?;

    // 3a. `--quantities FILE` (proposal 2026-08-19): report this fit with a
    //     supplied reporting vocabulary in place of the model's own. This is the
    //     reason the feature exists — the archived IR above is what makes a fit
    //     self-contained AND what makes correcting a quantity in the source have
    //     no effect on a fit that already ran.
    //
    //     The vocabulary is compiled against the model SOURCE, not the archived
    //     IR, and the source is required. A `let` that mentions a parameter
    //     (`let R0 = beta / gamma`) is INLINED at its use sites and appears
    //     nowhere in the compiled IR, so a vocabulary resolved against the IR
    //     would reject `R0` as undeclared against a model that plainly declares
    //     it — a wrong answer of exactly the plausible-looking kind.
    //
    //     Safety comes from a hash, not from trust: `model_ir_hash` EXCLUDES
    //     `quantities` (runid's `ir_quantities_excluded_from_hash`), so the
    //     recompiled model and the archived one must hash equal. If they do the
    //     source is the same model that was fit, modulo exactly the reporting
    //     layer we are replacing, and transplanting its quantities is sound. If
    //     they differ the source has drifted and we refuse — the alternative is
    //     reporting a fit through formulas written for a different model.
    let vocabulary: Option<crate::quantities_file::QuantitiesOverride> =
        args.quantities.as_deref().map(crate::quantities_file::QuantitiesOverride::load).transpose()?;
    if let Some(v) = &vocabulary {
        model.quantities = quantities_from_vocabulary(&model, &config.model.camdl, v)?;
    }

    // gh#616: `fit predict` HAS the fit's data, so it resolves the model's
    // observation anchors rather than refusing them — and it must do so HERE,
    // before the horizon guard below and before any compile. Inheriting an
    // unresolved horizon would emit a single-snapshot "forecast" that reads as a
    // plausible plateau at exit 0.
    //
    // Resolving before `refuse_scenario_horizon` is also what makes that guard
    // meaningful: with a model horizon of `last_obs + 4 'weeks` and a scenario's
    // of `last_obs + 8 'weeks`, the comparison is between two REAL numbers that
    // differ, so the scenario's window is refused rather than silently dropped.
    // Two anchors that resolve to the SAME time still compare equal and are
    // allowed, which is the existing no-op precedent.
    let resolved_obs_anchors: Option<ir::anchor::ObsAnchorTimes> =
        if crate::obs_anchor::model_is_anchored(&model) {
            let dt0 = model.simulation.dt.unwrap_or(1.0);
            let (first, last) =
                crate::obs_anchors_from_config(&model, &config, dt0).map_err(|e| {
                    format!("resolving this model's observation anchors from the fit's data: {e}")
                })?;
            let w = ir::anchor::ObsAnchorTimes { first, last };
            let moved = crate::obs_anchor::substitute(&mut model, w)?;
            crate::obs_anchor::report(&moved, &model);
            Some(w)
        } else {
            None
        };
    let model = model;
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
    // gh#561: every cell this command runs is handed `t_end_override: None`, so
    // it replays at the MODEL's horizon; a scenario's own `simulate { to }`
    // cannot move this command's window. Honouring it here would be a no-op, and
    // silently ignoring a declared horizon is the exact bug gh#561 is about.
    // Refuse, and name the two things that do work.
    //
    // gh#696 extended the free-forward band OUT TO that model horizon; it did
    // not make a per-scenario horizon reachable, so this refusal stands
    // unchanged and must keep firing.
    //
    // Only a genuine difference from the model horizon is refused: a preset
    // restating the run horizon is a no-op and keeps working.
    for sref in &scenario_refs {
        crate::util::refuse_scenario_horizon(
            &model, Some(sref.name()), "fit predict",
            "the predictive replays every scenario at the model's own horizon, \
             which is one window for the whole run",
        )?;
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

    // The observed-data axis, per leaf in `model.observations` order — the
    // cadence the predictive is emitted at, and the axis a
    // `value_at(..., last_obs)` quantity anchors to. Read once here, above the
    // horizon blocks, because BOTH the free-forward sink and the contrast
    // reducer need it and the contrast reducer runs whichever horizon was asked
    // for.
    let leaf_times: Vec<Vec<f64>> = model
        .observations
        .iter()
        .map(|o| {
            leaves.iter().find(|l| leaf_matches(o, l)).map(|l| l.times.clone()).unwrap_or_default()
        })
        .collect();
    // The run's resolved observation window (gh#694). ONE window for the whole
    // command: the ordinary `quantities/` sidecar and the contrast arms both
    // fold through it, so they cannot disagree about what `last_obs` meant for
    // this fit. `None` iff no leaf carries an observation time — the callers
    // below then have nothing to anchor to and say so.
    let quantity_obs_anchors: Option<sim::quantity::ObsAnchorTimes> =
        sim::quantity::ObsAnchorTimes::of_times(leaf_times.iter().flatten().copied());
    // Per leaf, the conditioning boundary this fit scored from (gh#702). Read
    // off `leaf_times` — the OBSERVED axis — because `condition_from`'s
    // relative form is anchored to each stream's own first observation, which
    // is what the fit anchored it to.
    let leaf_window_start = leaf_condition_boundaries(&model, &config, &leaf_times, dt)?;

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
    // The same count split by chain — the denominator a `--by-chain` section
    // reports, since a per-chain band is over that chain's draws alone (gh#794).
    // Empty when the horizon was not requested or the cloud carries no chain
    // keys.
    let mut ff_draws_per_chain: BTreeMap<usize, usize> = BTreeMap::new();

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
        // A `value_at(..., last_obs)` quantity anchors to the observed-data
        // axis; if no leaf carries any observation time the anchor is
        // unresolvable — refuse loudly rather than silently censoring every
        // draw (proposal 2026-08-17).
        if let Some(eval) = quant_eval.as_deref() {
            if eval.references_obs_anchor() && quantity_obs_anchors.is_none() {
                return Err(format!(
                    "quantity `{}` reads `value_at` at an observation anchor \
                     (`last_obs` / `first_obs`, with or without an offset), but \
                     this fit binds no observation times to anchor to.",
                    eval.obs_anchor_quantity_names().join("`, `"),
                ));
            }
        }
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

        // ── The free-forward emission grid (gh#696) ────────────────────────
        //
        // The trajectory is integrated to the model horizon — `quantities/`
        // carries rows out there and always has. Only the emission time list
        // was restricted to the observed times, so a scenario's projected curve
        // in OBSERVABLE units existed inside this same run and was discarded.
        // Past the last observation each stream continues on its own reporting
        // cadence, over the trajectory's own snapshot grid, to the horizon.
        //
        // This is the FREE-FORWARD grid only. The one-step band is
        // `p(y_t | y_{1:t-1})` — data-conditioned by definition, so it has
        // nothing to say past the data and keeps its own observed-time axis
        // (`one_step_bands`, untouched below).
        //
        // `leaf_times` itself is NOT extended: it is also the observed-data
        // axis that `value_at(..., last_obs)` anchors to (`quantity_obs_anchors`,
        // read above) and that the contrast reducer folds through. `last_obs`
        // means the last OBSERVATION, not the last emitted row.
        let horizon_output_times: Vec<f64> =
            sim::output::output_times(&model.output.times, model.simulation.t_end);
        let mut ff_emit_times: Vec<Vec<f64>> = leaf_times.clone();
        // One note per logical stream, not per stratum leaf.
        let mut noted: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (si, obs_ir) in model.observations.iter().enumerate() {
            let times = &ff_emit_times[si];
            if times.is_empty() {
                continue; // stream not bound to data (or filtered out)
            }
            let last_obs = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            // Negated `>`, not `<=`: an unresolved horizon arrives as NaN and
            // must fall through to "no forecast window", never to "extend".
            if !(model.simulation.t_end > last_obs) {
                continue; // no forecast window — byte-identical to before
            }
            // A likelihood whose arguments read an observation data column
            // (`binomial(n = tested)`, a person-time offset) has NO value for
            // that column past the data: `compile_obs_sample_pf` resolves an
            // unavailable aux to 0, so a binomial denominator becomes 0 and
            // every forecast draw is 0. An identically-zero ribbon presented as
            // a projection is the worst outcome available here, so the stream is
            // omitted from the extended grid and the omission is announced —
            // silence would read as "this model has no horizon".
            let aux_cols = crate::pfilter::stream_aux_columns(obs_ir);
            if !aux_cols.is_empty() {
                if noted.insert(obs_ir.source.clone()) {
                    eprintln!(
                        "fit predict: stream '{}' stops at its last observation \
                         (t = {last_obs}) even though the model horizon is {}: its \
                         likelihood reads the observed data column(s) [{}], which \
                         have no value past the data. Projecting it would draw \
                         against a zero denominator and report an identically-zero \
                         band as a forecast.\n  \
                         Fix: to project this stream past the data, express the \
                         denominator in the model (a parameter or a compartment \
                         sum) rather than reading it from the data file.",
                        obs_ir.source,
                        model.simulation.t_end,
                        aux_cols.join(", "),
                    );
                }
                continue;
            }
            let extension = forecast_times(times, &horizon_output_times);
            if extension.is_empty() {
                if noted.insert(obs_ir.source.clone()) {
                    eprintln!(
                        "fit predict: stream '{}' stops at its last observation \
                         (t = {last_obs}) even though the model horizon is {}: no \
                         trajectory output time past the data continues this \
                         stream's observed cadence, so there is no grid to project \
                         onto.\n  \
                         Fix: widen `output {{ trajectories {{ ... }} }}` so the \
                         forecast window carries output times on the stream's \
                         reporting cadence.",
                        obs_ir.source,
                        model.simulation.t_end,
                    );
                }
                continue;
            }
            ff_emit_times[si].extend(extension);
        }

        // The free-forward cells (one per sweep-point × scenario, engine canonical
        // order), plus the stacked quantity files — accumulated across the whole
        // sweep grid, since one design cell is a (sweep point, scenario) pair and
        // the sink is rebuilt per sweep point.
        let mut ff_cells: Vec<FreeForwardCell> = Vec::new();
        let mut stacked = crate::quantity_output::StackedQuantities::new(
            crate::quantity_output::Mode::Banded,
        );

        // ── Honor --n-draws on free-forward: an even, deterministic subsample of
        // the whole posterior cloud (gh#387). Without it the free-forward path
        // replays EVERY draw single-threaded, so a long-burn-in ODE fit
        // (thousands of ~seconds-each solves) never finishes and no artifact is
        // written. Same knob + default + strided pick the one-step horizon uses,
        // so both horizons subsample identically. Computed ONCE — the subsample
        // is scenario/sweep-independent.
        let ff_cap = args.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS);
        let ff_idx = subsample_indices(posterior.n_draws(), ff_cap);
        let ff_draws: Vec<&IndexMap<String, f64>> =
            ff_idx.iter().map(|&i| &posterior.draws()[i]).collect();
        ff_n_draws = ff_draws.len();
        // The chain behind each replayed draw, picked by the *same* indices as the
        // draws themselves — the partition the per-row R̂ reduces over (gh#794).
        // Every entry is `None` when the cloud's `draws.tsv` has no chain column,
        // and the per-row columns are then empty rather than invented.
        let ff_chain_of_point: Vec<Option<usize>> = match posterior.keys() {
            Some(k) => ff_idx.iter().map(|&i| k.per_draw[i].map(|(c, _)| c)).collect(),
            None => vec![None; ff_idx.len()],
        };
        for c in ff_chain_of_point.iter().flatten() {
            *ff_draws_per_chain.entry(*c).or_insert(0) += 1;
        }
        if ff_chain_of_point.iter().all(Option::is_none) {
            eprintln!(
                "fit predict: this fit's draws.tsv carries no chain column, so the \
                 per-row convergence columns (rhat_mean / ess_mean / rhat_pred / \
                 ess_pred) are left empty — a between-chain statistic needs to know \
                 which chain each draw came from."
            );
            if args.by_chain {
                eprintln!(
                    "fit predict: --by-chain has nothing to split on for the same \
                     reason, so no `chain` column is written and the file carries \
                     the pooled band only."
                );
            }
        }
        if ff_n_draws < posterior.n_draws() {
            eprintln!(
                "fit predict: free_forward horizon — subsampling {ff_n_draws} of {} \
                 posterior draws (raise with --n-draws)",
                posterior.n_draws()
            );
        }

        // ── The conditioned read for in-window quantities (gh#722) ─────────
        //
        // A `value_at` anchored at or before `last_obs` is a retrospective
        // estimand: the observations covering it are what answers it, so it is
        // folded over the draw's saved smoothing path `p(x | y, θ)` rather than
        // over a fresh unconditioned replay from `init {}`. Classified per
        // QUANTITY, once, so a band is never a mixture of two objects.
        let quant_paths: Vec<sim::quantity::QuantityPath> = quant_eval
            .as_deref()
            .map(|e| e.eval_paths(quantity_obs_anchors))
            .unwrap_or_default();
        let any_smoothed =
            quant_paths.iter().any(|p| *p == sim::quantity::QuantityPath::Smoothed);
        // Named, not silent: an `observations.<stream>` reduction anchored
        // inside the record has the same defect, and no saved path carries a
        // y_sim draw to fix it with.
        if let Some(eval) = quant_eval.as_deref() {
            let unconditioned = eval.quantity_names_on(
                sim::quantity::QuantityPath::ReplayUnconditioned,
                quantity_obs_anchors,
            );
            if !unconditioned.is_empty() {
                eprintln!(
                    "fit predict: quantity `{}` reduces `observations.<stream>` at an \
                     anchor inside the observed record, and is reported on the \
                     free-forward replay. The saved smoothing path carries the \
                     conditioned projection (`inc_<stream>`, a mean), not a draw from \
                     it, so there is nothing conditioned to sample the observation \
                     from (gh#722).\n  \
                     Fix: express the quantity over latent state \
                     (`value_at(<state expr>, last_obs)`), which IS read on the \
                     smoothing path.",
                    unconditioned.join("`, `"),
                );
            }
        }
        // The saved subset is resolved only when a quantity needs it — the scan
        // reads every `chain_*/trajectories.tsv`, and a predict with no
        // in-window `value_at` must not pay for it.
        let ff_saved: Option<SavedPaths> = if any_smoothed {
            posterior.keys().map(|k| {
                let saved = k.resolve_saved();
                SavedPaths {
                    stage_dir: saved.stage_dir,
                    per_draw: ff_idx.iter().map(|&i| saved.per_draw[i]).collect(),
                    n_saved: ff_idx.iter().filter(|&&i| saved.per_draw[i].is_some()).count(),
                }
            })
        } else {
            None
        };
        // The scenario whose cells read conditioned: the no-overlay `fitted`
        // arm only, and only when no `--enable`/`--disable` rides on it (that
        // makes it a counterfactual too).
        let conditioned_scenario: Option<String> =
            if args.enable.is_empty() && args.disable.is_empty() {
                Some(crate::args::FITTED.to_string())
            } else {
                None
            };
        if any_smoothed {
            let names = quant_eval
                .as_deref()
                .map(|e| {
                    e.quantity_names_on(
                        sim::quantity::QuantityPath::Smoothed,
                        quantity_obs_anchors,
                    )
                    .join("`, `")
                })
                .unwrap_or_default();
            match (&ff_saved, &conditioned_scenario) {
                (Some(s), Some(_)) if s.n_saved > 0 => {
                    eprintln!(
                        "fit predict: quantity `{names}` is anchored at or before \
                         last_obs — reported on the conditioned smoothing path \
                         p(x|y), not on the free-forward replay (gh#722). \
                         {}/{} replayed draws have a saved path; the other {} are \
                         censored for these quantities, never substituted from the \
                         replay.",
                        s.n_saved,
                        ff_n_draws,
                        ff_n_draws - s.n_saved,
                    );
                }
                (_, None) => {
                    eprintln!(
                        "fit predict: quantity `{names}` is anchored at or before \
                         last_obs, but `--enable`/`--disable` makes every arm of this \
                         run a counterfactual, for which no conditioned path exists. \
                         They are reported on the free-forward replay, which ignores \
                         the observations they are anchored inside (gh#722)."
                    );
                }
                _ => {
                    eprintln!(
                        "fit predict: quantity `{names}` is anchored at or before \
                         last_obs, but this fit saved no latent path for any replayed \
                         draw — there is nothing conditioned to read them on. They are \
                         reported as fully censored rather than taken from the \
                         free-forward replay, which ignores every observation they are \
                         anchored inside (gh#722).\n  \
                         Fix: re-fit with `n_trajectories` set on the posterior stage \
                         (PGAS/PMMH save the smoothing paths), then re-run \
                         `fit predict`."
                    );
                }
            }
        }
        // A counterfactual arm keeps the replay, and says so once — its rows sit
        // in the same file as the fitted arm's, under a `scenario` column, so a
        // reader comparing them must know they are two different objects.
        if any_smoothed && scenario_refs.iter().any(|s| s.name() != crate::args::FITTED) {
            eprintln!(
                "fit predict: the scenario arms report their anchored quantities on \
                 their OWN free-forward replay — the smoothing path was inferred under \
                 the fitted model, and the data a counterfactual would have generated \
                 do not exist. `quantities.json` tags each entry with `evaluated_on`."
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
            //
            // A sweep point OVERRIDES a parameter, so its cells replay a
            // different model than the one the smoothing path was inferred
            // under: only the un-swept design cell reads conditioned (gh#722).
            let conditioned = match (&ff_saved, &conditioned_scenario) {
                (Some(saved), Some(scenario))
                    if conditioned_here(&ff_saved, &conditioned_scenario, sweep_pt, scenario) =>
                {
                    Some(ConditionedSource::load(
                        scenario.clone(),
                        saved.per_draw.clone(),
                        &saved.stage_dir,
                        &model,
                    )?)
                }
                _ => None,
            };
            let mut sink = PredictiveSink {
                compiled: compiled.clone(),
                leaf_times: ff_emit_times.clone(),
                leaf_aux: leaf_aux.clone(),
                leaf_window_start: leaf_window_start.clone(),
                quant_eval: quant_eval.clone(),
                obs_anchors: quantity_obs_anchors,
                conditioned,
                chain_of_point: ff_chain_of_point.clone(),
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
                    // gh#626: the predictive window comes from the data.
                    t_end_override: None,
                    // gh#641: the predictive replays from the model's init {} at
                    // each posterior draw; a filtered-state restart is a
                    // `simulate --init-state` surface, not a `fit predict` one.
                    init_state: None,
                    // gh#616: the engine re-loads the model from the ARCHIVED IR
                    // path per cell, which still carries the unresolved anchors —
                    // so the resolved window has to travel with the job, not just
                    // live in the copy predict substituted above. Same window,
                    // so the per-cell model matches the one this command
                    // validated.
                    obs_anchors: resolved_obs_anchors,
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
                    // One design cell = this (sweep point, scenario). The shared
                    // stacker owns the header-drop and the manifest merge.
                    // gh#722: a scenario cell folded the replay for EVERY
                    // quantity, so its manifest entries must say `replay` —
                    // the routing tag is per design cell, not per model.
                    let cell_paths: Vec<sim::quantity::QuantityPath> =
                        if conditioned_here(&ff_saved, &conditioned_scenario, sweep_pt, scenario_name)
                        {
                            quant_paths.clone()
                        } else {
                            quant_paths
                                .iter()
                                .map(|p| match p {
                                    sim::quantity::QuantityPath::Smoothed => {
                                        sim::quantity::QuantityPath::Replay
                                    }
                                    other => *other,
                                })
                                .collect()
                        };
                    stacked.push_group(
                        &model.quantities,
                        coords,
                        &accum.quant_draws,
                        &accum.quant_times,
                        Some(crate::quantity_output::EvaluatedOn {
                            paths: &cell_paths,
                            n_conditioned: accum.n_conditioned,
                        }),
                        // The same chain partition the predictive rows reduce
                        // over, so a `quantities/` row and a `predictive/` row
                        // from one fit describe the same chains (gh#794).
                        Some(&accum.draw_chain),
                        &calendar,
                    )?;
                }
                ff_cells.push(FreeForwardCell {
                    sweep: sweep_pt.clone(),
                    scenario: scenario_name.clone(),
                    bands: assemble_predictive(
                        &model, accum, &ff_emit_times, &leaves, schema.as_ref(),
                        args.by_chain,
                    )?,
                });
            }
        }

        if !stacked.is_empty() {
            let (outs, manifest) = stacked.finish(&calendar)?;
            quantity_outputs = outs;
            quantity_manifest = Some(manifest);
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

    // WHERE the predictive lands is keyed by the chain selection it was banded
    // over (gh#795): the full cloud keeps `predictive/` + `predictive.json`,
    // a `--exclude-chains 3,5` subset writes `predictive-excl3,5/` +
    // `predictive-excl3,5.json`. A subset is a different posterior — its own
    // warning says so — and writing it at the pooled address REPLACED the run's
    // canonical predictive with a cherry-picked one, silently, with only a
    // `chain_selection` stamp inside the file it had already overwritten.
    let predictive_sub =
        crate::chain_selection::artifact_name("predictive", selection.as_ref());

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
            // The pooled band first — the default object, and the one a reader
            // who ignores the `chain` column gets.
            sections.push(PredictiveSection {
                chain: ChainLabel::All,
                scenario: cell.scenario.clone(),
                sweep: cell.sweep.clone(),
                horizon: Horizon::FreeForward,
                treatment: treatment_kind,
                convergence: posterior.convergence,
                n_draws: ff_n_draws,
                rows: &s.rows,
            });
            // Then one section per chain (`--by-chain`; empty otherwise). Each
            // reports its OWN draw count: a per-chain band is over that chain's
            // draws, and carrying the pooled `n_draws` there would overstate the
            // denominator by the number of chains.
            for (chain, chain_rows) in &s.per_chain {
                sections.push(PredictiveSection {
                    chain: ChainLabel::One(*chain),
                    scenario: cell.scenario.clone(),
                    sweep: cell.sweep.clone(),
                    horizon: Horizon::FreeForward,
                    treatment: treatment_kind,
                    convergence: posterior.convergence,
                    n_draws: ff_draws_per_chain.get(chain).copied().unwrap_or(0),
                    rows: chain_rows,
                });
            }
        }
        if let Some(s) = os_stream {
            sections.push(PredictiveSection {
                // The one-step cell pools over filter particles as well as
                // draws, so it has no per-chain decomposition to offer: its rows
                // stay `all` even under `--by-chain`.
                chain: ChainLabel::All,
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
        // columns (the group-by keys) in header order: the `chain` column when
        // `--by-chain` wrote one, `scenario`, the `sweep:<param>` columns,
        // `time`, the stratum dims, `horizon`, `treatment`. The band columns are
        // the quantile labels; the diagnostics
        // are the stage-provenance pair (`fit_rhat_max` / `fit_ess_min`, constant down
        // the file), the per-row pairs (`rhat_mean` / `ess_mean` over the latent
        // expected value, `rhat_pred` / `ess_pred` over the predictive draws —
        // gh#794), and `n_draws`. `value_kind` is the observation's likelihood
        // family (the nature of the banded value).
        let mut coordinates: Vec<String> = Vec::new();
        if sections.iter().any(|s| s.chain != ChainLabel::All) {
            coordinates.push("chain".to_string());
        }
        coordinates.push("scenario".to_string());
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
            // The manifest's declared location is the keyed one, or a consumer
            // that follows it lands back on the pooled artifact.
            "file": format!("{predictive_sub}/{source}.tsv"),
            "value_kind": value_kind,
            "coordinates": coordinates,
            "diagnostics": [
                "fit_rhat_max", "fit_ess_min",
                "rhat_mean", "ess_mean", "rhat_pred", "ess_pred",
                "n_draws",
            ],
            "band": QUANTILE_LEVELS.iter().map(|(_, l)| *l).collect::<Vec<_>>(),
            "quantiles": QUANTILE_LEVELS.iter().map(|(q, _)| *q).collect::<Vec<_>>(),
        }));

        let pred_tsv = render_predictive_tsv_sections(&index_dims, &sections);
        written.push(write_tsv(&segment, &predictive_sub, source, &pred_tsv)?);
    }
    // `predictive.json`: the per-stream join contract beside the predictive
    // TSVs — a sibling of `quantities.json`, NOT in the run_id-keyed CAS leaf
    // (regenerated, overwritten in place). Written whenever any predictive
    // stream was emitted.
    if !predictive_manifest_entries.is_empty() {
        let mut manifest = serde_json::json!({
            // The tag is the only thing telling a consumer which column
            // contract a stored artifact was written under, so every change to
            // the column set bumps it.
            //
            // v1: `rhat_max`/`ess_min` carried classic Gelman-Rubin R̂ and a
            //     Geyer per-chain sum.
            // v2 (gh#84): the same two column NAMES, now the rank-normalized
            //     split R̂ and bulk-ESS of Vehtari et al. (2021), with `ess_min`
            //     withheld whenever any assessed parameter has no pooled ESS
            //     rather than silently minimizing over the ones that do. Same
            //     names, different statistics — which is exactly why the tag
            //     has to be keyed on.
            // v3 (gh#794): those two are renamed `fit_rhat_max`/`fit_ess_min`,
            //     because they describe the *fit* and not the row they sit on,
            //     and the per-row `rhat_mean`/`ess_mean`/`rhat_pred`/`ess_pred`
            //     channels join them.
            "schema": "camdl.predictive/v3",
            "calendar": calendar.to_json(),
            "streams": predictive_manifest_entries,
        });
        // Provenance: a chain-subset predictive records the selection alongside
        // the streams, so a chain-subset artifact is never mistakable for a
        // full-cloud one. Absent (no key) when the full cloud was used. The
        // ADDRESS (`predictive_sub`, above) is what keeps the two artifacts from
        // colliding; this stamp is what names the selection once you have one.
        if let Some(info) = posterior.selection() {
            manifest["chain_selection"] = info.to_json();
        }
        let path = segment.join(format!("{predictive_sub}.json"));
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
    //
    // WHICH sidecar, though, is keyed by the reporting vocabulary that produced
    // it (proposal 2026-08-19): the model's own block keeps writing
    // `quantities/`, a `--quantities` vocabulary writes `quantities-<key8>/`
    // with a matching manifest. Two vocabularies applied to one fit are two
    // different tables; sharing one address would overwrite the first and leave
    // no way to tell which formulas produced the survivor.
    //
    // A quantity is read off the posterior cloud, so the CHAIN SELECTION keys it
    // for exactly the same reason (gh#795) — a chain-subset table is a different
    // table. The two keys are independent and compose:
    // `quantities-<key8>-excl3,5/`.
    let quantities_sub = crate::chain_selection::artifact_name(
        &crate::quantities_file::quantities_dir_name(vocabulary.as_ref()),
        selection.as_ref(),
    );
    for (name, content) in &quantity_outputs {
        written.push(write_tsv(&segment, &quantities_sub, name, content)?);
    }
    if let Some(manifest) = &quantity_manifest {
        let manifest =
            crate::quantities_file::stamp_provenance(manifest, vocabulary.as_ref())?;
        let path = segment.join(format!("{quantities_sub}.json"));
        std::fs::write(&path, &manifest)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        written.push(path);
    }
    // Counterfactual contrasts: auto-emitted when the model declares any
    // `contrasts {}`. The two-arm replay reducer forks each forkable posterior
    // draw from its smoothed X(T*) and bands the difference into
    // `contrasts/<name>.tsv`. A model with no `contrasts {}` is byte-identical
    // (this is a no-op). A non-forkable / ODE fit emits no file and a located note.
    //
    // The SAME `--exclude-chains` selection the free-forward cloud was resolved
    // under is passed here: a contrast bands over the cloud this run's manifest
    // describes, never over a chain the manifest says was dropped (gh#695).
    if !model.contrasts.is_empty() {
        let paths = crate::fit::contrasts::emit_contrasts(
            &segment,
            args.stage.as_deref(),
            selection.as_ref(),
            &model,
            posterior.backend,
            seed,
            quantity_obs_anchors,
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

/// Compile `vocabulary` against the fit's model SOURCE and return the quantity
/// IR to transplant onto `archived` — refusing unless the source is still the
/// same model the fit ran on.
///
/// The equality test is `runid::inputs::model_ir_hash`, which excludes
/// `quantities` (and `contrasts`). That is what makes it the right test rather
/// than merely a convenient one: it asks "is this the same model apart from its
/// reporting layer?", which is exactly the question a reporting-layer swap has
/// to answer yes to. It is also gradient-independent (runid SV=2), so the lean
/// recompile here and a full-Jacobian fit compile still agree.
fn quantities_from_vocabulary(
    archived: &ir::Model,
    model_source: &str,
    vocabulary: &crate::quantities_file::QuantitiesOverride,
) -> Result<Vec<ir::quantity::Quantity>, String> {
    if !crate::util::model_is_camdl_source(model_source) {
        return Err(format!(
            "--quantities needs this fit's model SOURCE, but the fit records `{model_source}`, \
             which is compiled IR. A reporting vocabulary is resolved against the model's \
             symbols at compile time, and a `let` that mentions a parameter is inlined away \
             in the IR, so it cannot be applied to a compiled model."
        ));
    }
    if !std::path::Path::new(model_source).is_file() {
        return Err(format!(
            "--quantities needs this fit's model source `{model_source}`, which is not \
             readable from here. The fit itself is self-contained (it archived its compiled \
             IR), but a reporting vocabulary has to be compiled against the model's symbols."
        ));
    }
    // Lean compile: `fit predict` replays forward trajectories and never
    // recomputes an ODE gradient (gh#439 A2). Identity is gradient-independent,
    // so this does not disturb the hash comparison below.
    let (ir_path, _tmp) = crate::util::resolve_ir_path_with_quantities(
        model_source, false, Some(vocabulary))?;
    let (recompiled, _) = crate::util::load_model(&ir_path)?;
    let archived_h = runid::inputs::model_ir_hash(archived);
    let recompiled_h = runid::inputs::model_ir_hash(&recompiled);
    if archived_h != recompiled_h {
        return Err(format!(
            "this fit's model source `{model_source}` is no longer the model the fit ran on, \
             so a reporting vocabulary cannot be checked against it (archived {}, source {}). \
             Quantities are excluded from that hash, so this difference is a real change to \
             the dynamics, observations, or parameters — not the vocabulary. Re-run the fit \
             against the current source, or check out the source the fit used.",
            &archived_h.to_hex()[..8],
            &recompiled_h.to_hex()[..8],
        ));
    }
    Ok(recompiled.quantities)
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

/// Each leaf's conditioning boundary, in `model.observations` order and aligned
/// 1:1 with `leaf_times`: the time the FIT reset that stream's incidence
/// accumulator at, or `None` for a stream with no `condition_from` (and for one
/// not bound to data at all, which has no window and emits nothing).
///
/// gh#702. A `condition_from` fit simulates `[t_start, cond_from)` as warm-up
/// and scores its first incidence datum over `(cond_from, first_obs]`. The
/// predictive is plotted against that same datum, so it must report the same
/// interval — the free-forward projection would otherwise report
/// `(t_start, first_obs]`, folding the whole warm-up into the first row.
///
/// Resolution routes through [`crate::fit::runner::stream_condition_window`],
/// the same per-stream resolver `fit run` / `pfilter` / `profile` reach through
/// `apply_conditioning_windows`, so the fit and its predictive cannot disagree
/// about where a stream's first bin opens. What predict does NOT do is prepend
/// the reset-only hole row: here the observation times are also the emitted
/// axis and the axis `value_at(..., first_obs / last_obs)` anchors to
/// (`main.rs`'s anchor resolution deliberately folds over the raw streams for
/// the same reason), and a synthetic row must shift neither.
///
/// Label validation is not repeated here: it is a property of the whole spec
/// against the whole bound stream set, already enforced when the fit ran, and
/// re-running it under a `--stream` filter would reject a shadow naming a
/// stream this invocation merely filtered out.
fn leaf_condition_boundaries(
    model: &ir::Model,
    config: &crate::fit::config_v2::FitConfigV2,
    leaf_times: &[Vec<f64>],
    dt: f64,
) -> Result<Vec<Option<f64>>, String> {
    let spec = config.condition_from.as_ref();
    if spec.is_none() {
        return Ok(vec![None; model.observations.len()]);
    }
    let t_start = model.simulation.t_start;
    model
        .observations
        .iter()
        .enumerate()
        .map(|(si, o)| {
            // A stream with no observation times is unbound or filtered out: it
            // emits nothing, so it has no first bin to open.
            let first_obs = leaf_times[si].iter().copied().fold(f64::INFINITY, f64::min);
            if !first_obs.is_finite() {
                return Ok(None);
            }
            crate::fit::runner::stream_condition_window(
                spec, &o.source, &o.name, first_obs, model, t_start, dt,
            )
            .map(|w| w.boundary())
        })
        .collect()
}

/// Whether a leaf passes the `--stream` filter — matches the logical source or
/// the expanded leaf name (the proposal's "accepts either name").
fn stream_selected(o: &ir::observation::ObservationModel, filter: Option<&str>) -> bool {
    match filter {
        None => true,
        Some(name) => o.source == name || o.name == name,
    }
}

// ── The free-forward forecast grid (gh#696) ────────────────────────────────

/// How closely a trajectory output time must sit on the continued reporting
/// cadence to be that cadence's next forecast time. Relative, so float
/// accumulation over a long horizon does not drop a row; far below any real
/// cadence, so two adjacent output times are never confusable.
const CADENCE_MATCH_REL_TOL: f64 = 1e-6;

/// The forecast times for one observation leaf: an UNBROKEN continuation of the
/// stream's own reporting cadence past its last observation, over the
/// trajectory's snapshot grid, out to the model horizon. Empty when there is no
/// forecast window, no cadence to continue, or nothing on the grid to continue
/// onto.
///
/// The continuation stops at the first cadence step the snapshot grid does not
/// carry, rather than skipping it: on an incidence stream the emitted value is
/// the flow accumulated since the PREVIOUS emitted time, so a skipped step
/// would widen one row's interval — a lone spike in a column of otherwise
/// uniform counts.
///
/// **Why the observed cadence and not every output time.** An incidence stream
/// reports the flow accumulated since the previous emitted time
/// ([`crate::project_all_obs_times`] differences the cumulative flow), so
/// emitting weekly data and then daily forecast rows would put 1-day counts and
/// 7-day counts in one column — a sevenfold cliff at the seam that reads as a
/// collapse rather than a change of units. Continuing the cadence keeps every
/// row in the column the same quantity.
///
/// **Why not `emit_schedule`.** It is optional — a fit-only model omits it
/// entirely (spec §16.4: the data file's `time` column drives the fit) — and
/// where it is present it is the SIMULATE-side cadence, which is free to
/// disagree with the data the fit actually holds. The observed times are always
/// present here and are what the rest of `fit predict` is keyed to, so they are
/// the single authority; `output_times` (the trajectory's own snapshot grid, the
/// grid `quantities/` is written on) supplies the candidates, which is what
/// makes the forecast band and the quantity sidecars agree by construction.
///
/// `output_times` must be the model's snapshot grid: every emitted time has to
/// be one, or the projection would read a stale snapshot
/// (`check_obs_times_on_snapshot_grid`, gh#589). Filtering candidates OUT of
/// that grid is what makes that guard unreachable from here.
fn forecast_times(obs_times: &[f64], output_times: &[f64]) -> Vec<f64> {
    // Two observations are the fewest that define a spacing to continue.
    if obs_times.len() < 2 {
        return Vec::new();
    }
    let last_obs = obs_times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !last_obs.is_finite() {
        return Vec::new();
    }
    let gaps: Vec<f64> =
        obs_times.windows(2).map(|w| w[1] - w[0]).filter(|g| *g > 0.0).collect();
    if gaps.is_empty() {
        return Vec::new();
    }
    // Negated `>`, not `<=`: `modal_value` returns NaN when it finds no positive
    // gap, and NaN must fall through to "no cadence", not to "cadence is fine".
    let cadence = crate::util::modal_value(&gaps);
    if !(cadence > 0.0) {
        return Vec::new();
    }
    let Some(&grid_end) = output_times.last() else {
        return Vec::new();
    };
    // Walk the cadence forward, taking the snapshot time that sits on each step.
    // The EXACT grid value is what is emitted (not the accumulated `last_obs +
    // k·cadence`), so the projection's snapshot lookup gets a bit-identical
    // time and never falls back to the previous snapshot.
    let mut out: Vec<f64> = Vec::new();
    let mut k = 1.0_f64;
    loop {
        let want = last_obs + k * cadence;
        let tol = CADENCE_MATCH_REL_TOL * want.abs().max(cadence);
        if want > grid_end + tol {
            break;
        }
        match output_times.iter().copied().find(|o| (o - want).abs() <= tol) {
            Some(on_grid) => out.push(on_grid),
            None => break, // the cadence has left the snapshot grid
        }
        k += 1.0;
    }
    out
}

/// Quantile-reduce ONE scenario's accumulated free-forward samples into per-stream
/// bands, grouping leaves by logical stream. The scenario/horizon/treatment/
/// convergence/n_draws labels are applied at render time (each
/// [`PredictiveSection`] carries them), so this returns only the bands.
///
/// `by_chain` additionally bands each chain's own draws into
/// [`StreamBands::per_chain`] (gh#794). The pooled band is computed and
/// returned identically either way — a per-chain band is an addition beside it,
/// never a replacement.
fn assemble_predictive(
    model: &ir::Model,
    accum: &ScenarioAccum,
    emit_times: &[Vec<f64>],
    leaves: &[LeafObs],
    schema: Option<&ObsSchema>,
    by_chain: bool,
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
        // chain id → its rows, in the same (leaf, time) order as `rows`.
        let mut per_chain: BTreeMap<usize, Vec<BandRow>> = BTreeMap::new();
        for &si in leaf_idxs {
            let stratum: Vec<(String, String)> = model.observations[si]
                .stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect();
            for (ti, draws_at_t) in accum.samples[si].iter().enumerate() {
                let t = emit_times[si][ti];
                rows.push(
                    band_row(
                        t,
                        stratum.clone(),
                        CellDraws {
                            predictive: draws_at_t,
                            mean: &accum.means[si][ti],
                        },
                        &accum.draw_chain,
                    )
                    .map_err(|e| format!("stream '{source}' at t={t}: {e}"))?,
                );
                if !by_chain {
                    continue;
                }
                // One band per chain, over that chain's draws ALONE — no
                // truncation to a common length (that is an R̂ precondition,
                // not a banding one), and no convergence cells (a between-chain
                // statistic has nothing to compare a single chain against).
                let Some(split) =
                    crate::fit::row_convergence::group_by_chain(draws_at_t, &accum.draw_chain)
                else {
                    continue;
                };
                for (chain, chain_draws) in split {
                    per_chain.entry(chain).or_default().push(BandRow {
                        time: t,
                        stratum: stratum.clone(),
                        quantiles: band(&chain_draws).map_err(|e| {
                            format!("stream '{source}' at t={t}, chain {}: {e}", chain + 1)
                        })?,
                        mean_conv: None,
                        pred_conv: None,
                    });
                }
            }
        }
        streams.push(StreamBands {
            source: source.clone(),
            index_dims,
            rows,
            per_chain: per_chain.into_iter().collect(),
        });
    }

    Ok(streams)
}

/// The two per-draw series one free-forward predictive cell carries, named so
/// the operand each convergence channel reduces is explicit at the call site
/// (gh#794).
struct CellDraws<'a> {
    /// `y_rep` at this cell, one per draw — the posterior predictive draw, with
    /// observation noise.
    predictive: &'a [f64],
    /// `E[y | x_t, θ]` for the *same* draws in the same order — the latent
    /// expected value, before observation noise.
    mean: &'a [f64],
}

/// Band one predictive cell and attach its two per-row convergence channels.
///
/// The quantiles pool over every draw; the convergence numbers run the *same*
/// draws through the chain-grouped reduction. The channel split is the whole
/// point of gh#794 and it lives here, in one place: `rhat_mean` reduces
/// [`CellDraws::mean`] and `rhat_pred` reduces [`CellDraws::predictive`].
/// Reducing the predictive draws where the mean belongs is not a near-miss —
/// observation noise inflates the within-chain variance and drags R̂ toward 1,
/// so the diagnostic reports a forecast the chains disagree fourfold about as
/// sound.
fn band_row(
    time: f64,
    stratum: Vec<(String, String)>,
    draws: CellDraws<'_>,
    chains: &crate::fit::row_convergence::ChainOfDraw,
) -> Result<BandRow, String> {
    use crate::fit::row_convergence::row_convergence;
    let quantiles = band(draws.predictive)?;
    Ok(BandRow {
        time,
        stratum,
        quantiles,
        mean_conv: row_convergence(draws.mean, chains),
        pred_conv: row_convergence(draws.predictive, chains),
    })
}

// ── Posterior-cloud subsampling (shared by both horizons) ───────────────────

/// Default posterior-cloud subsample cap for `fit predict`, shared by both
/// horizons. Neither needs the full fit-grade cloud: the one-step band pools
/// `draws × n_particles` per cell and the free-forward band pools one forward
/// replay per draw, so a few hundred draws saturate q05…q95. The full cloud is
/// never replayed silently — a full free-forward replay of a long-burn-in ODE
/// fit is thousands of ~seconds-each solves (gh#387).
pub(crate) const DEFAULT_PREDICT_DRAWS: usize = 200;

/// Evenly-spaced, deterministic subsample of a posterior cloud down to `cap`
/// draws (the whole cloud when `cap >= len`, always at least one). Both horizons
/// cap the cloud through this one seam, so a fit-grade cloud is never silently
/// replayed at full size. The pick is STRIDED across the whole cloud
/// (`idx = i * total / n_used`), never `take(cap)` of the front — a front-take
/// would bias the band toward early sweeps / a single chain. Chosen draws are
/// returned in cloud order.
pub(crate) fn subsample_draws<T>(draws: &[T], cap: usize) -> Vec<&T> {
    subsample_indices(draws.len(), cap).into_iter().map(|i| &draws[i]).collect()
}

/// The subsample as INDICES into the cloud. Everything a draw carries that is
/// not in its parameter row — its `(chain, draw)` key, and so the saved
/// smoothing path a `value_at(..., last_obs)` reads (gh#722) — has to be picked
/// by the SAME indices, and re-deriving the stride at the second call site is
/// how the two silently drift apart.
pub(crate) fn subsample_indices(total: usize, cap: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    let n_used = cap.min(total).max(1);
    if n_used >= total {
        (0..total).collect()
    } else {
        (0..n_used).map(|i| (i * total) / n_used).collect()
    }
}

// ── The one-step-ahead posterior predictive producer ───────────────────────

/// Particle count for the one-step prediction filter. It need not match the
/// fit's N — the band pools across draws too, and every particle's `ỹ` is
/// kept (the recorder retains them), so a modest N yields a dense band cheaply.
const ONE_STEP_N_PARTICLES: usize = 500;

/// The pooled one-step predictive: per-(stream-leaf, obs-time) particle
/// samples accumulated across posterior draws, plus how many draws
/// contributed.
#[cfg_attr(test, derive(Debug))]
struct PooledOneStep {
    /// Leaf stream names, from the first successful filter result (identical
    /// across draws — same obs model).
    stream_names: Vec<String>,
    /// Union observation-time axis, same provenance.
    obs_times: Vec<f64>,
    /// `pooled[stream_idx][obs_idx]` = ỹ over (particles × draws). NaN entries
    /// (a not-scheduled stream at a union time) are dropped, mirroring the
    /// prequential capture's filter.
    pooled: Vec<Vec<Vec<f64>>>,
    /// Draws that contributed (= param_sets.len() − degenerate-skipped).
    n_pooled: usize,
}

/// Filter every posterior draw and pool the per-(stream, time) one-step
/// predictive samples. The filter call is injected (`run_filter(draw_idx,
/// params, seed)`) so the skip policy below is unit-testable without a fit
/// on disk; production passes `bootstrap_filter`.
///
/// Skip policy (gh#620): a draw whose filter bails with
/// `SimError::PFDegenerate` — the statistical ESS-collapse /
/// all-particles-dead bail — is skipped and counted, never fatal. A small
/// tail of pathological draws is expected in a converging posterior, and one
/// such draw must not abort the whole predictive. This mirrors the settled
/// treatment of the same error everywhere else it occurs: PMMH's init-eval
/// pushes a BadInit diagnostic and skips the chain (`pmmh.rs`), IF2 skips
/// the one bad chain and continues (`runner.rs`), and the PF eval closures
/// report −∞ so MH rejects the proposal cleanly (`sim/src/error.rs` on the
/// variant). The deliberate contrast carries over too: any OTHER error —
/// compute budget, model eval — stays fatal and aborts on first occurrence,
/// because it trips identically for every draw and 200 identical skips
/// would only bury it.
///
/// Skips are loud, never silent: one stderr summary line with the count and
/// the first failure, and the returned `n_pooled` flows into the artifact's
/// `n_draws` column, so the band never claims more draws than it used. If
/// EVERY draw degenerates there is nothing to pool and the run errors.
fn pool_one_step_draws(
    param_sets: &[Vec<f64>],
    base_seed: u64,
    mut run_filter: impl FnMut(
        usize,
        &[f64],
        u64,
    ) -> Result<sim::inference::particle_filter::PFilterResult, sim::SimError>,
) -> Result<PooledOneStep, String> {
    let mut pooled: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut stream_names: Vec<String> = Vec::new();
    let mut obs_times: Vec<f64> = Vec::new();
    let mut skipped: Vec<(usize, String)> = Vec::new();

    for (draw_idx, params) in param_sets.iter().enumerate() {
        // Distinct, reproducible per-draw seed: mix the draw index into the
        // base seed so each filter pass has its own RNG stream and the whole
        // run is deterministic given `base_seed`. Keyed on the index, so a
        // skipped draw does not shift the seeds of the draws after it.
        let seed = base_seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(draw_idx as u64 + 1));
        let result = match run_filter(draw_idx, params, seed) {
            Ok(r) => r,
            Err(e @ sim::SimError::PFDegenerate { .. }) => {
                skipped.push((draw_idx, e.to_string()));
                continue;
            }
            Err(e) => {
                return Err(format!("one-step filter failed on draw {draw_idx}: {e:?}"));
            }
        };
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

    let n_total = param_sets.len();
    if skipped.len() == n_total && n_total > 0 {
        let (first_idx, first_err) = &skipped[0];
        return Err(format!(
            "one_step horizon: the one-step filter degenerated on ALL {n_total} posterior \
             draws — there is nothing to pool. First failure (draw {first_idx}): \
             {first_err}"
        ));
    }
    if !skipped.is_empty() {
        let (first_idx, first_err) = &skipped[0];
        eprintln!(
            "fit predict: one_step horizon — skipped {}/{} posterior draws whose one-step \
             filter degenerated (a small tail of degenerate draws is expected in a \
             converging posterior; first: draw {first_idx}: {first_err}); bands pool the \
             remaining {} draws",
            skipped.len(),
            n_total,
            n_total - skipped.len(),
        );
    }
    Ok(PooledOneStep { stream_names, obs_times, pooled, n_pooled: n_total - skipped.len() })
}

/// Build the one-step-ahead posterior predictive bands: for each (subsampled)
/// posterior draw θ, run a bootstrap filter over the data with
/// `record_prequential = true`, capturing the per-particle one-step predictive
/// samples `ỹ ∼ p(y | x_t, θ)` at each observation time (the particles are
/// distributed as `p(x_t | y_{1:t-1}, θ)` at that point). Pool over
/// (particles × draws) per `(stream-leaf, time)`, quantile, and group by logical
/// source + stratum exactly like the free-forward path. Horizon = one_step.
///
/// `n_draws_used` (out) is the number of draws actually pooled — the
/// subsample minus any degenerate-skipped draws (gh#620) — for the
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

    // ── The conditioning window, into the FILTER (gh#702) ──────────────────
    //
    // The one-step band is `p(y_t | y_{1:t-1})`, drawn from the filter's own
    // accumulator — so unlike the free-forward path there is nothing to reseed
    // after the fact: the filter has to be handed the same leading reset-only
    // hole the fit's likelihood was, or its first predictive is an incidence
    // bin that has been accumulating since `t_start`. Without it the band is
    // wrong at the first observation AND the particle weights there are
    // computed against the wrong bin, so the resampled cloud carries the error
    // forward.
    //
    // Resolution routes through the shared per-stream resolver, so the fit and
    // its predictive cannot disagree about where a stream's first bin opens.
    // Label validation and the W329 enforcer stay at the fit
    // (`apply_conditioning_windows`): they judge the whole spec against the
    // whole bound stream set, which a `--stream`-filtered predict does not have.
    let mut boundary_by_leaf: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    if config.condition_from.is_some() {
        let t_start = compiled.model.simulation.t_start;
        for s in obs_streams.iter_mut() {
            let first_obs_s = s.data.iter().map(|o| o.time).fold(f64::INFINITY, f64::min);
            if !first_obs_s.is_finite() {
                continue;
            }
            let window = crate::fit::runner::stream_condition_window(
                config.condition_from.as_ref(),
                &s.obs_model_ir.source,
                &s.name,
                first_obs_s,
                model,
                t_start,
                dt,
            )?;
            if let Some(cond_from) = window.boundary() {
                // The same three-line prepend `apply_conditioning_windows`
                // makes: `cells` is authoritative for scoring, and the `data`
                // row's 0.0 is a never-read placeholder.
                s.data.insert(
                    0,
                    sim::inference::if2::Observation { time: cond_from, value: 0.0 },
                );
                s.cells.insert(0, None);
                s.aux.insert(0, Vec::new());
                boundary_by_leaf.insert(s.name.clone(), cond_from);
            }
        }
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

    // Each draw's parameter vector — base defaults overlaid by name (the
    // survey.rs:722-734 idiom). The cloud is schema-validated upstream, so
    // every model parameter is present.
    let param_sets: Vec<Vec<f64>> = chosen
        .iter()
        .map(|draw| {
            let mut params = compiled.default_params.clone();
            for (name, &value) in draw.iter() {
                if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                    params[idx] = value;
                }
            }
            params
        })
        .collect();

    let PooledOneStep { stream_names, obs_times, pooled, n_pooled } =
        pool_one_step_draws(&param_sets, base_seed, |_, params, seed| {
            bootstrap_filter(&process, &obs_model, params, &smc_config, seed)
        })?;

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
            // This leaf's conditioning boundary, when it has one (gh#702).
            let boundary = boundary_by_leaf.get(leaf.name.as_str()).copied();
            for (ti, &t) in obs_times.iter().enumerate() {
                let cell = &pooled[si][ti];
                if cell.is_empty() {
                    // This stream is not scheduled at this union time (all NaN,
                    // dropped) — emit no row for it (multi-cadence).
                    continue;
                }
                if boundary.is_some_and(|b| (t - b).abs() < 1e-9) {
                    // The conditioning boundary is a RESET, not an observation:
                    // the filter is scheduled there (that is how the bin
                    // reopens) and therefore drew a sample, but the bin it
                    // predicts is the discarded warm-up and there is no
                    // observed row to plot it against. Per leaf, so a sibling
                    // stream genuinely observed at this union time keeps its
                    // row (gh#702).
                    continue;
                }
                let quantiles = band(cell)
                    .map_err(|e| format!("stream '{source}' (one_step) at t={t}: {e}"))?;
                // No per-row R̂ on the one-step horizon (gh#794). Its cell is a
                // pool over (posterior draws × filter particles), so the
                // sequence a chain contributes is not the posterior chain and an
                // ESS computed from its autocorrelation would be inflated by the
                // particles. Deferred rather than approximated: *both* channels
                // stay empty, so a reader cannot pick up the diluted one by
                // accident. Follow-up: gh#798.
                rows.push(BandRow {
                    time: t,
                    stratum: stratum.clone(),
                    quantiles,
                    mean_conv: None,
                    pred_conv: None,
                });
            }
        }
        // No per-chain decomposition on the one-step horizon: the cell pools
        // over (posterior draws × filter particles), so a "chain's band" here
        // would not be the same object the free-forward per-chain band is.
        streams.push(StreamBands {
            source: source.clone(),
            index_dims,
            rows,
            per_chain: Vec::new(),
        });
    }

    Ok((streams, n_pooled))
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
    use crate::fit::row_convergence::ChainOfDraw;

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

    /// gh#691 / gh#687. A predictive band's `ess_min` must not be a minimum
    /// over the parameters that happened to report one. A parameter with no
    /// bulk ESS serializes as JSON `null` and drops out of the map, so a `min`
    /// that skips it is a minimum over the REPORTING SUBSET, and it RISES as
    /// the fit gets worse: the
    /// badly-mixing parameters leave the map and the well-mixing survivors set
    /// the value. Measured on the summary headline as a 13x inversion between
    /// two runs of one model differing only in particle count (gh#687); this is
    /// the same reduce, on the band label.
    /// A stage whose R̂ was REFUSED must not have its predictive band labelled
    /// with a reported convergence status. The refusals live in the summary's
    /// `rhat_not_reported`, and reading only the numeric maps dropped them —
    /// silently promoting "we could not assess this parameter" to "assessed,
    /// and here is the max over the ones that were".
    #[test]
    fn read_convergence_is_not_reported_when_the_summary_carries_a_refusal() {
        let dir = std::env::temp_dir().join("camdl_read_convergence_refusal_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // `tau` never moved — a sampler pathology, not a missing number.
        std::fs::write(
            dir.join("pgas_summary.json"),
            r#"{"rhat": {"a2": 1.01},
                "ess":  {"a2": 145.0},
                "rhat_not_reported": {"tau": "constant_draws"}}"#,
        )
        .unwrap();
        match read_convergence(&dir, Some(FitAlgorithm::Pgas)) {
            ConvergenceStatus::Reported { rhat_max, .. } => panic!(
                "a fit with a refused parameter must not report a band off the \
                 parameters that survived; got max R̂ = {rhat_max}"
            ),
            _ => {}
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_convergence_withholds_ess_min_when_a_param_reports_none() {
        let dir = std::env::temp_dir().join("camdl_read_convergence_partial_ess_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // `tau` is assessed across chains (finite R̂) and carries no ESS.
        // `a2` mixed and carries 145.
        std::fs::write(
            dir.join("pgas_summary.json"),
            r#"{"rhat": {"a2": 1.01, "tau": 2.639},
                "ess":  {"a2": 145.0, "tau": null}}"#,
        )
        .unwrap();
        match read_convergence(&dir, Some(FitAlgorithm::Pgas)) {
            ConvergenceStatus::Reported { rhat_max, ess_min } => {
                assert!((rhat_max - 2.639).abs() < 1e-9,
                    "the worst R̂ is still reported: {rhat_max}");
                assert!(!ess_min.is_finite(),
                    "ess_min must be withheld while `tau` reports none — 145 is \
                     the minimum over the converged subset, not over the fit: \
                     got {ess_min}");
            }
            other => panic!("R̂ is assessable here, so the band is Reported: {other:?}"),
        }
        // Control: with every assessed parameter reporting, the minimum is real.
        std::fs::write(
            dir.join("pgas_summary.json"),
            r#"{"rhat": {"a2": 1.01, "tau": 2.639},
                "ess":  {"a2": 145.0, "tau": 9.0}}"#,
        )
        .unwrap();
        match read_convergence(&dir, Some(FitAlgorithm::Pgas)) {
            ConvergenceStatus::Reported { ess_min, .. } => assert!((ess_min - 9.0).abs() < 1e-9,
                "a complete map reports the slowest parameter: {ess_min}"),
            other => panic!("expected Reported, got {other:?}"),
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

    // ── The free-forward forecast grid (gh#696) ─────────────────────────────

    /// A regular output/snapshot grid `start, start+step, …, end`.
    fn grid(start: f64, step: f64, end: f64) -> Vec<f64> {
        let mut out = Vec::new();
        let mut t = start;
        while t <= end + step * 1e-9 {
            out.push(t);
            t += step;
        }
        out
    }

    #[test]
    fn forecast_continues_the_observed_cadence_to_the_horizon() {
        // Weekly data to t=56 on a DAILY snapshot grid running to t=80. The
        // forecast rows must be weekly (63, 70, 77) — not daily. A daily
        // continuation would put 1-day incidence counts in the same column as
        // 7-day ones and show a sevenfold cliff at the seam.
        let obs = vec![7.0, 14.0, 21.0, 28.0, 35.0, 42.0, 49.0, 56.0];
        assert_eq!(forecast_times(&obs, &grid(0.0, 1.0, 80.0)), vec![63.0, 70.0, 77.0]);
    }

    #[test]
    fn forecast_is_empty_when_the_horizon_does_not_run_past_the_data() {
        // The snapshot grid ends AT the last observation — nothing to project
        // onto, so the emitted grid is exactly the observed one (the change is
        // a no-op for every model whose horizon is its data).
        let obs = vec![7.0, 14.0, 21.0, 28.0];
        assert!(forecast_times(&obs, &grid(0.0, 1.0, 28.0)).is_empty());
    }

    #[test]
    fn forecast_survives_holes_in_the_observed_series() {
        // A daily series missing three days: the gaps are 1, 1, 2, 1, 3, 1 — the
        // MODE is 1, which is the cadence the series actually reports on. A
        // mean or median gap would drift off the reporting grid and admit
        // nothing.
        let obs = vec![1.0, 2.0, 3.0, 5.0, 6.0, 9.0, 10.0];
        assert_eq!(forecast_times(&obs, &grid(0.0, 1.0, 14.0)),
                   vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn forecast_admits_nothing_off_the_reporting_cadence() {
        // Weekly data (last observation t=52) against a coarse `at = [...]`
        // output grid. The next weekly step, 59, is not a snapshot time — so
        // nothing is emitted. In particular t=80 must NOT be picked up merely
        // because 80 − 52 = 28 happens to be four cadences out: emitting it
        // alone would put a 28-day accumulation in a column of 7-day counts.
        let obs = vec![10.0, 17.0, 24.0, 31.0, 38.0, 45.0, 52.0];
        let sparse_output = vec![0.0, 20.0, 40.0, 60.0, 80.0];
        assert!(
            forecast_times(&obs, &sparse_output).is_empty(),
            "an isolated cadence multiple is not a continuation"
        );
    }

    #[test]
    fn forecast_stops_at_the_first_missing_cadence_step() {
        // Weekly data to t=14 on a daily grid with t=28 punched out (an
        // `at = [...]` list). 21 is available; 28 is not. The continuation ends
        // at 21 rather than resuming at 35 with a doubled interval.
        let obs = vec![7.0, 14.0];
        let mut output = grid(0.0, 7.0, 49.0);
        output.retain(|t| *t != 28.0);
        assert_eq!(forecast_times(&obs, &output), vec![21.0]);
    }

    #[test]
    fn forecast_needs_two_observations_to_have_a_cadence() {
        // One observation defines no spacing to continue; do not guess one.
        assert!(forecast_times(&[30.0], &grid(0.0, 1.0, 80.0)).is_empty());
        assert!(forecast_times(&[], &grid(0.0, 1.0, 80.0)).is_empty());
    }

    #[test]
    fn forecast_times_land_on_the_snapshot_grid() {
        // Every emitted time must be an output time — the projection reads the
        // snapshot at (or before) it, so an off-grid time would silently report
        // a stale state (gh#589). Filtering candidates out of the grid is what
        // makes that guard unreachable from here; pin the property.
        let obs = vec![4.0, 8.0, 12.0, 16.0];
        let output = grid(0.0, 2.0, 40.0);
        let fc = forecast_times(&obs, &output);
        assert_eq!(fc, vec![20.0, 24.0, 28.0, 32.0, 36.0, 40.0]);
        for t in &fc {
            assert!(output.iter().any(|o| (o - t).abs() < 1e-12),
                    "forecast time {t} must be an output time");
        }
    }

    // ── pool_one_step_draws skip policy (gh#620, ebola F11) ─────────────────

    use sim::error::PFDegenerateKind;
    use sim::inference::particle_filter::{PFilterResult, PrequentialRecorded};

    /// A minimal successful filter result: one stream, two obs times, two
    /// particles per (time, stream), samples = `base` and `base + 1`.
    fn fake_pf_result(base: f64) -> PFilterResult {
        let per_time = |b: f64| vec![vec![b, b + 1.0]]; // [stream][particle]
        PFilterResult {
            log_likelihood: -10.0,
            ess_trace: vec![100.0, 100.0],
            logw_var_trace: vec![0.1, 0.1],
            ll_increments: vec![-5.0, -5.0],
            predictions: None,
            final_states: None,
            ancestry: None,
            prequential: Some(PrequentialRecorded {
                obs_times: vec![7.0, 14.0],
                log_liks: vec![vec![-5.0, -5.0], vec![-5.0, -5.0]],
                y_pred_samples: vec![vec![base, base + 1.0], vec![base, base + 1.0]],
                stream_names: vec!["cases".into()],
                per_stream_log_liks: vec![vec![vec![-5.0, -5.0]], vec![vec![-5.0, -5.0]]],
                per_stream_samples: vec![per_time(base), per_time(base)],
            }),
        }
    }

    fn degenerate() -> sim::SimError {
        sim::SimError::PFDegenerate {
            kind: PFDegenerateKind::EssCollapsed { last_ess: vec![1.0, 1.0, 1.0] },
            obs_window: 3,
            elapsed_s: 0.5,
        }
    }

    #[test]
    fn one_step_pool_skips_degenerate_draw_and_counts() {
        // Draw 1 of 3 degenerates: it is skipped, the other two pool, and
        // n_pooled says 2 — the artifact must not claim 3 draws (F11: one
        // pathological draw of 200 used to abort the whole predictive).
        let param_sets = vec![vec![0.0]; 3];
        let out = pool_one_step_draws(&param_sets, 42, |draw_idx, _params, _seed| {
            if draw_idx == 1 { Err(degenerate()) } else { Ok(fake_pf_result(10.0)) }
        })
        .expect("a single degenerate draw must not abort the pool");
        assert_eq!(out.n_pooled, 2);
        assert_eq!(out.stream_names, vec!["cases".to_string()]);
        assert_eq!(out.obs_times, vec![7.0, 14.0]);
        // 2 surviving draws × 2 particles per (stream, time).
        assert_eq!(out.pooled[0][0].len(), 4);
        assert_eq!(out.pooled[0][1].len(), 4);
    }

    #[test]
    fn one_step_pool_aborts_on_structural_error() {
        // A non-degeneracy error (here a numerical collapse) trips identically
        // for every draw — it must abort on first occurrence, naming the draw,
        // not be skipped 200 times.
        let param_sets = vec![vec![0.0]; 3];
        let err = pool_one_step_draws(&param_sets, 42, |draw_idx, _params, _seed| {
            if draw_idx == 1 {
                Err(sim::SimError::NumericalCollapse { kind: sim::CollapseKind::UnOpNan, t: 3.0 })
            } else {
                Ok(fake_pf_result(10.0))
            }
        })
        .expect_err("structural errors must stay fatal");
        assert!(err.contains("draw 1"), "fatal error must name the draw: {err}");
    }

    #[test]
    fn one_step_pool_errors_when_all_draws_degenerate() {
        // Nothing pooled → no band to emit; the run must error loudly, not
        // return an empty artifact.
        let param_sets = vec![vec![0.0]; 3];
        let err = pool_one_step_draws(&param_sets, 42, |_, _, _| Err(degenerate()))
            .expect_err("an all-degenerate pool must error");
        assert!(err.contains("ALL 3"), "must say every draw degenerated: {err}");
    }

    #[test]
    fn one_step_pool_seeds_keyed_on_draw_index() {
        // A skipped draw must not shift later draws' seeds — reproducibility
        // of the pooled band is keyed on (base_seed, draw index) alone.
        use std::cell::RefCell;
        let param_sets = vec![vec![0.0]; 3];
        let seeds_with_skip = RefCell::new(Vec::new());
        let _ = pool_one_step_draws(&param_sets, 42, |draw_idx, _params, seed| {
            seeds_with_skip.borrow_mut().push((draw_idx, seed));
            if draw_idx == 0 { Err(degenerate()) } else { Ok(fake_pf_result(10.0)) }
        });
        let seeds_no_skip = RefCell::new(Vec::new());
        let _ = pool_one_step_draws(&param_sets, 42, |draw_idx, _params, seed| {
            seeds_no_skip.borrow_mut().push((draw_idx, seed));
            Ok(fake_pf_result(10.0))
        });
        assert_eq!(*seeds_with_skip.borrow(), *seeds_no_skip.borrow());
    }

    fn cb() -> crate::args::types::ForwardBackend {
        crate::args::types::ForwardBackend::ChainBinomial
    }

    fn stratum(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(d, l)| (d.to_string(), l.to_string())).collect()
    }

    /// A band row with no per-row convergence — the layout tests assert the
    /// column POSITIONS, and an unassessed row renders those four cells empty.
    fn bare_row(time: f64, stratum: Vec<(String, String)>, quantiles: Vec<f64>) -> BandRow {
        BandRow { time, stratum, quantiles, mean_conv: None, pred_conv: None }
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

    // ── gh#794: the per-row convergence channels ───────────────────────────
    //
    // The fixture below is the reason the feature has two columns instead of
    // one. It reproduces the five-chain Ebola forecast the issue measured: the
    // chains disagree FOURFOLD about the expected trajectory at +56 days (chain
    // medians 93 … 372 cases/day), while each chain's own negative-binomial
    // predictive band is two to three times wider than that disagreement. An R̂
    // over the predictive draws sits near 1 and calls the forecast sound; an R̂
    // over the latent mean does not.

    /// Sixteen fixed standardized offsets (mean 0, unit scale), so the fixture
    /// is deterministic and the two channels differ only in the *scale* the
    /// offsets are applied at — never in the draw pattern.
    ///
    /// Sign-alternating rather than sorted, and balanced within each half, so
    /// neither half of a chain drifts from the other. A sorted sequence would
    /// hand split-R̂ a within-chain trend and inflate *both* channels — an
    /// artifact of the fixture, not of the observation noise this test is about.
    const OFFSETS: [f64; 16] = [
        -1.53, 1.53, -0.89, 0.89, -0.49, 0.49, -0.16, 0.16,
        1.15, -1.15, 0.67, -0.67, 0.32, -0.32, 0.0, 0.0,
    ];

    /// The measured chain medians of the eight-week forecast (issue gh#794).
    const CHAIN_LEVELS: [f64; 5] = [93.0, 125.0, 140.0, 155.0, 372.0];

    /// The fixture: per-draw latent means and predictive draws for one cell,
    /// plus the chain each draw came from.
    ///
    /// `mean_sd` is the within-chain spread of the expected trajectory (the
    /// parameter uncertainty inside one chain); `obs_sd` is the extra spread the
    /// observation model adds on top of it. The predictive draw is the mean plus
    /// its own noise, so the two series are the same object before and after the
    /// observation model — exactly what the production path accumulates.
    fn diluted_forecast_cell(mean_sd: f64, obs_sd: f64) -> (Vec<f64>, Vec<f64>, ChainOfDraw) {
        let mut means = Vec::new();
        let mut preds = Vec::new();
        let mut chains = Vec::new();
        for (c, level) in CHAIN_LEVELS.iter().enumerate() {
            for (i, z) in OFFSETS.iter().enumerate() {
                let m = level + mean_sd * z;
                means.push(m);
                // A *different* fixed offset for the observation noise (rotated by
                // the chain and the draw), so the noise is not a rescaling of
                // the parameter spread.
                let w = OFFSETS[(i + 5 * c + 3) % OFFSETS.len()];
                preds.push((m + obs_sd * w).max(0.0));
                chains.push(Some(c));
            }
        }
        (means, preds, ChainOfDraw(chains))
    }

    /// *The* test for gh#794: on chains that genuinely disagree about the
    /// trajectory, under an observation model dispersed enough to hide it,
    /// `rhat_mean` is large and `rhat_pred` is near 1.
    ///
    /// If this ever passes with the two agreeing, the fixture has stopped
    /// exercising the point of the feature.
    #[test]
    fn rhat_mean_catches_a_disagreement_that_rhat_pred_is_diluted_past() {
        // Within-chain parameter spread 12; observation noise 230 on top, giving
        // a within-chain predictive spread of ~230 against a between-chain
        // spread of ~100 — the issue's measured 0.37 ratio.
        let (means, preds, chains) = diluted_forecast_cell(12.0, 230.0);
        let row = band_row(
            56.0,
            vec![],
            CellDraws { predictive: &preds, mean: &means },
            &chains,
        )
        .expect("the cell bands");

        let mean_conv = row.mean_conv.expect("5 chains x 16 draws is assessable");
        let pred_conv = row.pred_conv.expect("5 chains x 16 draws is assessable");
        let (rhat_mean, rhat_pred) = (mean_conv.rhat, pred_conv.rhat);

        // Measured on this fixture: rhat_mean 2.7445, rhat_pred 1.0825.
        assert!(
            rhat_mean > 2.5,
            "the chains disagree fourfold about the expected trajectory, so \
             rhat_mean must flag it; got {rhat_mean:.4}"
        );
        assert!(
            rhat_pred < 1.15,
            "the observation noise hides that disagreement from a draw-based \
             R̂, which is why rhat_pred is the weaker column; got {rhat_pred:.4}"
        );
        // The gap is the finding, not either number alone. A change that made
        // both columns read the same operand would collapse it.
        assert!(
            rhat_mean - rhat_pred > 1.0,
            "rhat_mean {rhat_mean:.4} and rhat_pred {rhat_pred:.4} must come \
             apart on this fixture — if they agree, neither column is telling \
             the user anything the other does not"
        );
        // The dilution inflates the effective sample size the same way: 11.5
        // against 62.8 here. A user reading `ess_pred` alone would believe the
        // forecast rests on five times the information it does.
        assert!(
            pred_conv.ess > 4.0 * mean_conv.ess,
            "ess_pred {:.1} must be far larger than ess_mean {:.1} — the noise \
             that hides the disagreement also reads as independent information",
            pred_conv.ess,
            mean_conv.ess
        );
    }

    /// The other direction: when the observation model adds nothing, the two
    /// channels agree. This is what makes the test above a statement about
    /// DILUTION rather than about the fixture's arithmetic.
    #[test]
    fn the_two_channels_agree_when_the_observation_model_adds_no_noise() {
        let (means, preds, chains) = diluted_forecast_cell(12.0, 0.0);
        let row = band_row(56.0, vec![], CellDraws { predictive: &preds, mean: &means }, &chains)
            .expect("the cell bands");
        let rhat_mean = row.mean_conv.expect("assessable").rhat;
        let rhat_pred = row.pred_conv.expect("assessable").rhat;
        assert!(
            rhat_mean > 2.0 && rhat_pred > 2.0,
            "with no observation noise both channels see the same disagreement: \
             mean {rhat_mean:.4}, pred {rhat_pred:.4}"
        );
    }

    /// The columns land in the header positions the manifest advertises, and an
    /// assessed row renders real numbers there.
    #[test]
    fn per_row_convergence_columns_render_between_the_stage_stamp_and_n_draws() {
        let (means, preds, chains) = diluted_forecast_cell(12.0, 230.0);
        let row = band_row(56.0, vec![], CellDraws { predictive: &preds, mean: &means }, &chains)
            .expect("the cell bands");
        let rows = vec![row];
        let tsv = render_predictive_tsv_sections(
            &[],
            &[PredictiveSection {
                chain: ChainLabel::All,
                scenario: "fitted".to_string(),
                sweep: Vec::new(),
                horizon: Horizon::FreeForward,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::Reported { rhat_max: 2.7863, ess_min: 5.8 },
                n_draws: 80,
                rows: &rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        let header: Vec<&str> = lines[0].split('\t').collect();
        let cells: Vec<&str> = lines[1].split('\t').collect();
        let at = |name: &str| header.iter().position(|h| *h == name)
            .unwrap_or_else(|| panic!("no `{name}` column in {header:?}"));
        // Provenance pair, then the two per-row pairs, then n_draws.
        assert!(at("fit_rhat_max") < at("rhat_mean"));
        assert!(at("ess_mean") < at("rhat_pred"));
        assert!(at("ess_pred") < at("n_draws"));
        // The stage stamp is the fit's worst parameter; the per-row numbers are
        // this row's. They are different numbers, which is the whole point.
        assert_eq!(cells[at("fit_rhat_max")], "2.7863");
        assert_ne!(cells[at("rhat_mean")], "", "an assessed row reports rhat_mean");
        assert_ne!(cells[at("rhat_pred")], "", "an assessed row reports rhat_pred");
        assert_ne!(
            cells[at("rhat_mean")], cells[at("fit_rhat_max")],
            "the per-row R̂ must not be the stage's provenance stamp"
        );
    }

    #[test]
    fn a_cloud_with_no_chain_keys_leaves_the_per_row_columns_empty() {
        let (means, preds, _) = diluted_forecast_cell(12.0, 230.0);
        let unkeyed = ChainOfDraw(vec![None; means.len()]);
        let row = band_row(56.0, vec![], CellDraws { predictive: &preds, mean: &means }, &unkeyed)
            .expect("the cell still bands");
        assert!(row.mean_conv.is_none(), "no chain column ⇒ no between-chain statistic");
        assert!(row.pred_conv.is_none());
        // The band itself is unaffected: the quantiles still pool every draw.
        assert_eq!(row.quantiles.len(), QUANTILE_LEVELS.len());
    }

    // ── gh#794: `--by-chain` ────────────────────────────────────────────────

    /// The `chain` column appears only when a per-chain section is present, and
    /// then it leads. Without one the rendered bytes are unchanged — the
    /// property `--by-chain` has to keep, since every existing consumer reads
    /// the no-flag file.
    #[test]
    fn the_chain_column_is_absent_until_a_per_chain_section_exists() {
        let rows = vec![bare_row(7.0, vec![], vec![1.0, 2.0, 3.0, 4.0, 5.0])];
        let conv = ConvergenceStatus::Reported { rhat_max: 1.0, ess_min: 100.0 };
        let pooled = PredictiveSection {
            chain: ChainLabel::All,
            scenario: "fitted".to_string(),
            sweep: Vec::new(),
            horizon: Horizon::FreeForward,
            treatment: TreatmentKind::Posterior,
            convergence: conv,
            n_draws: 40,
            rows: &rows,
        };
        let without = render_predictive_tsv_sections(&[], std::slice::from_ref(&pooled));
        assert!(
            !without.lines().next().unwrap().starts_with("chain\t"),
            "no per-chain section ⇒ no `chain` column: {}",
            without.lines().next().unwrap()
        );

        let per_chain = PredictiveSection { chain: ChainLabel::One(2), n_draws: 20, ..pooled };
        // Rebuild the pooled section (it was consumed by the struct update).
        let pooled = PredictiveSection {
            chain: ChainLabel::All,
            scenario: "fitted".to_string(),
            sweep: Vec::new(),
            horizon: Horizon::FreeForward,
            treatment: TreatmentKind::Posterior,
            convergence: conv,
            n_draws: 40,
            rows: &rows,
        };
        let with = render_predictive_tsv_sections(&[], &[pooled, per_chain]);
        let lines: Vec<&str> = with.trim_end().lines().collect();
        assert!(lines[0].starts_with("chain\tscenario\t"), "chain leads: {}", lines[0]);
        assert!(lines[1].starts_with("all\tfitted\t"), "the pooled row is `all`: {}", lines[1]);
        // 0-based internally, 1-based in the artifact — matching `chain_N/` and
        // `--exclude-chains`.
        assert!(lines[2].starts_with("3\tfitted\t"), "chain 2 renders as 3: {}", lines[2]);
        // A per-chain section reports its own denominator, not the pooled one.
        let hdr: Vec<&str> = lines[0].split('\t').collect();
        let n = hdr.iter().position(|h| *h == "n_draws").unwrap();
        assert_eq!(lines[1].split('\t').nth(n), Some("40"));
        assert_eq!(lines[2].split('\t').nth(n), Some("20"));
    }

    /// Everything except the leading `chain` cell is unchanged by the flag, so a
    /// consumer that drops the column recovers the pooled file exactly.
    #[test]
    fn by_chain_only_prepends_a_cell_to_the_rows_that_were_already_there() {
        let rows = vec![
            bare_row(7.0, vec![], vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            bare_row(14.0, vec![], vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        ];
        let conv = ConvergenceStatus::NotAssessed;
        let mk = |chain| PredictiveSection {
            chain,
            scenario: "fitted".to_string(),
            sweep: Vec::new(),
            horizon: Horizon::FreeForward,
            treatment: TreatmentKind::Posterior,
            convergence: conv,
            n_draws: 40,
            rows: &rows,
        };
        let pooled_only = render_predictive_tsv_sections(&[], &[mk(ChainLabel::All)]);
        let with_chain =
            render_predictive_tsv_sections(&[], &[mk(ChainLabel::All), mk(ChainLabel::One(0))]);
        let stripped: String = with_chain
            .lines()
            .filter(|l| !l.starts_with("1\t")) // drop the per-chain rows
            .map(|l| l.split_once('\t').unwrap().1) // drop the `chain` cell
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!(
            stripped, pooled_only,
            "dropping the `chain` column and its per-chain rows must reproduce \
             the pooled file byte for byte"
        );
    }

    /// A per-chain band carries no convergence cells: R̂ compares chains, and a
    /// single chain has nothing to compare against. Publishing a number there
    /// would be the same category error the whole issue is about.
    #[test]
    fn a_per_chain_row_reports_no_between_chain_statistic() {
        let (means, preds, chains) = diluted_forecast_cell(12.0, 230.0);
        let pooled =
            band_row(56.0, vec![], CellDraws { predictive: &preds, mean: &means }, &chains)
                .expect("the pooled cell bands");
        assert!(pooled.mean_conv.is_some(), "the pooled row does carry one");

        // What `assemble_predictive` builds for a per-chain row.
        let one_chain = BandRow {
            time: 56.0,
            stratum: vec![],
            quantiles: pooled.quantiles.clone(),
            mean_conv: None,
            pred_conv: None,
        };
        let rows = vec![one_chain];
        let tsv = render_predictive_tsv_sections(
            &[],
            &[PredictiveSection {
                chain: ChainLabel::One(0),
                scenario: "fitted".to_string(),
                sweep: Vec::new(),
                horizon: Horizon::FreeForward,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::NotAssessed,
                n_draws: 16,
                rows: &rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        let hdr: Vec<&str> = lines[0].split('\t').collect();
        let cells: Vec<&str> = lines[1].split('\t').collect();
        for name in ["rhat_mean", "ess_mean", "rhat_pred", "ess_pred"] {
            let i = hdr.iter().position(|h| *h == name).unwrap();
            assert_eq!(cells[i], "", "`{name}` must be empty on a single-chain band");
        }
    }

    #[test]
    fn chain_label_renders_one_based_and_pooled_as_all() {
        assert_eq!(ChainLabel::All.as_cell(), "all");
        assert_eq!(ChainLabel::One(0).as_cell(), "1");
        assert_eq!(ChainLabel::One(7).as_cell(), "8");
    }

    #[test]
    fn predictive_tsv_has_typed_axis_columns_and_one_row_per_cell() {
        let stream = StreamBands {
            source: "onset".into(),
            index_dims: vec!["patch".into()],
            per_chain: Vec::new(),
            rows: vec![
                bare_row(7.0, stratum(&[("patch", "Bo")]), vec![0.0, 1.0, 3.0, 6.0, 12.0]),
                bare_row(7.0, stratum(&[("patch", "Bombali")]), vec![0.0, 0.0, 1.0, 3.0, 7.0]),
            ],
        };
        let tsv = render_predictive_tsv_sections(
            &stream.index_dims,
            &[PredictiveSection {
                chain: ChainLabel::All,
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
            "scenario\ttime\tpatch\thorizon\ttreatment\tfit_rhat_max\tfit_ess_min\
             \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1],
            "fitted\t7\tBo\tfree_forward\tposterior\t1.0100\t420\t\t\t\t\t40\t0\t1\t3\t6\t12");
        assert_eq!(lines[2],
            "fitted\t7\tBombali\tfree_forward\tposterior\t1.0100\t420\t\t\t\t\t40\t0\t0\t1\t3\t7");
        assert_eq!(lines.len(), 3, "header + one row per (time, stratum)");
    }

    #[test]
    fn predictive_tsv_national_series_unassessed_convergence_is_empty() {
        let stream = StreamBands {
            source: "cases".into(),
            index_dims: vec![],
            per_chain: Vec::new(),
            rows: vec![bare_row(1.0, vec![], vec![1.0, 2.0, 3.0, 4.0, 5.0])],
        };
        let tsv = render_predictive_tsv_sections(
            &stream.index_dims,
            &[PredictiveSection {
                chain: ChainLabel::All,
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
        assert_eq!(lines[0],
            "scenario\ttime\thorizon\ttreatment\tfit_rhat_max\tfit_ess_min\
             \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "fitted\t1\tfree_forward\tposterior\t\t\t\t\t\t\t12\t1\t2\t3\t4\t5");
    }

    #[test]
    fn predictive_tsv_sweep_columns_lead_after_scenario_and_one_step_is_blank() {
        // Two free-forward sweep cells (k=8, k=12) plus a sweep-agnostic one-step
        // section: the `sweep:k` column follows `scenario`, free-forward rows carry
        // the cell's swept value, and the one-step rows leave it blank.
        let conv = ConvergenceStatus::Reported { rhat_max: 1.0, ess_min: 100.0 };
        let rows = vec![bare_row(7.0, vec![], vec![1.0, 2.0, 3.0, 4.0, 5.0])];
        let tsv = render_predictive_tsv_sections(
            &[],
            &[
                PredictiveSection {
                    chain: ChainLabel::All,
                    scenario: "fitted".to_string(),
                    sweep: vec![("k".to_string(), 8.0)],
                    horizon: Horizon::FreeForward,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
                PredictiveSection {
                    chain: ChainLabel::All,
                    scenario: "fitted".to_string(),
                    sweep: vec![("k".to_string(), 12.0)],
                    horizon: Horizon::FreeForward,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
                PredictiveSection {
                    chain: ChainLabel::All,
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
            "scenario\tsweep:k\ttime\thorizon\ttreatment\tfit_rhat_max\tfit_ess_min\
             \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95",
            "sweep:k column follows scenario"
        );
        assert_eq!(lines[1],
            "fitted\t8\t7\tfree_forward\tposterior\t1.0000\t100\t\t\t\t\t10\t1\t2\t3\t4\t5");
        assert_eq!(lines[2],
            "fitted\t12\t7\tfree_forward\tposterior\t1.0000\t100\t\t\t\t\t10\t1\t2\t3\t4\t5");
        // One-step row: the sweep:k cell is blank (empty), not a fabricated value.
        assert_eq!(lines[3],
            "fitted\t\t7\tone_step\tposterior\t1.0000\t100\t\t\t\t\t10\t1\t2\t3\t4\t5");
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
