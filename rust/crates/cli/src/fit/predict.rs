//! `camdl predict` — the free-forward posterior predictive verb, plus the
//! type system that makes a misread predictive object unrepresentable.
//!
//! A predictive varies on two independent axes the dangerous bugs leave
//! implicit: the **horizon** (what is predicted) and the **parameter
//! treatment** (how parameter uncertainty is handled). The second is the one
//! that bites — a free-forward band run at one plug-in parameter is
//! artificially narrow, and if it is read as a posterior band the gap is
//! mistaken for science. So both axes are typed and travel with every band:
//! the cloud a band is drawn from is built either from a fit run's draws
//! ([`PosteriorDraws::new`], treatment `posterior`) or from copies of one point
//! given with a model file ([`PosteriorDraws::plug_in`], treatment `plug_in`),
//! and the label is read off the cloud.
//!
//! Where the parameters come from is a [`PredictSource`]; everything else is
//! one predictor. A fit run of an optimizer (IF2 / NLopt), which produces no
//! draws cloud, is refused with the command that predicts at its estimate.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;

use crate::quantile::{band, fmt_time, fmt_value, Quantity, QUANTILE_LEVELS};

use score::convergence::ChainIndex;
use crate::chain_selection::{warn_active_selection, ChainSelection, SubsetInfo};
use crate::fit::failures::{DeterministicFailure, Site, SkippedStream};
use crate::fit::predict_streams::{select_streams, StreamSelection};
use crate::posterior_draws;
use crate::fit::method_result::PosteriorDiagnostics;
use score::convergence::MaxRhat;
use crate::run_meta::{FitAlgorithm, ObsSchema};

// ── The two axes, as types ─────────────────────────────────────────────────

/// What is predicted: the mode of a band in `predictive/<stream>.tsv`, its
/// `mode` column, typed so a one-step band is never read as a free-forward
/// one (or vice versa).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveMode {
    /// Run the fitted model forward from the start and see what it generates —
    /// `p(y_t | θ)`, never re-anchored to data. The generative check.
    FreeForward,
    /// `p(y_t | y_{1:t-1})` — the one-step-ahead posterior predictive: re-run a
    /// bootstrap filter over the data per posterior draw, sample `ỹ` from the
    /// propagated (pre-reweight) particles at each observation time, pool over
    /// (particles × draws). The honest short-horizon forecast object.
    OneStepAhead,
}

impl PredictiveMode {
    /// The legible value written into the `mode` artifact column.
    pub fn as_str(&self) -> &'static str {
        match self {
            PredictiveMode::FreeForward => "free_forward",
            PredictiveMode::OneStepAhead => "one_step",
        }
    }
}

/// The part a predictive row played in the fit — the `role` artifact column
/// (`docs/dev/temporal-architecture.md` §5.3). A one-step band covers every
/// data row, and the parameters behind it were fitted to the `Trained` ones
/// alone, so a `HeldOut` row's band is out of sample in its parameters while
/// its state has been filtered through every earlier row (gh#1079). The
/// spec's third role, `target`, belongs to a forecast's targets (each with a
/// horizon and a lead) and is not written here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowRole {
    /// A row of the fit's training prefix of its stream (an `NA` row
    /// included: it holds its position in the prefix).
    Trained,
    /// A row of the bound file after that prefix: the fit held it out.
    HeldOut,
    /// This prediction read no data row for this period: a free-forward row
    /// past the last row of the stream's bound file. Under `[data.holdout]`
    /// that includes the periods the holdout file covers, which is not read.
    Future,
    /// A data row of a fixed-point prediction given no `--holdout-after`:
    /// nothing records which rows the point was fitted to, so the row may
    /// have trained it or not.
    Unknown,
}

impl RowRole {
    /// The legible value written into the `role` artifact column.
    pub fn as_str(&self) -> &'static str {
        match self {
            RowRole::Trained => "trained",
            RowRole::HeldOut => "held_out",
            RowRole::Future => "future",
            RowRole::Unknown => "unknown",
        }
    }

    /// The role of the data row at `position` in a stream whose first
    /// `trained` rows are the fit's training prefix. Membership is the row's
    /// position, as the partition records it; no time is compared with a
    /// cutoff here.
    pub fn of_data_row(position: usize, trained: usize) -> Self {
        if position < trained { RowRole::Trained } else { RowRole::HeldOut }
    }
}

/// The fit's parameter set as `camdl predict` reads it: the training partition
/// the fit recorded, verified against the bound files as they are now.
struct ParameterSet {
    /// The record as the fit wrote it; `predictive.json` carries it as
    /// `forecast.json` does.
    record: crate::run_meta::TrainingPartitionRecord,
    partition: crate::fit::training_partition::TrainingPartition,
    /// The sources with a `[data.holdout]` file, sorted. Those files are not
    /// read here, so none of their rows is predicted.
    holdout_sources: Vec<String>,
    /// The labels (the times rows are emitted at) of each stream's rows past
    /// the training prefix of its training file, from the stored problem's
    /// `held_out/`. A stored problem keeps only its training rows as data, so
    /// a predicted row past them at one of these times is a held-out row.
    held_out_times: BTreeMap<String, Vec<f64>>,
    /// `true` for a fixed point given no `--holdout-after`, whose partition
    /// marks every row trained only because nothing says otherwise: its data
    /// rows are then [`RowRole::Unknown`]. A stored fit's partition, and a
    /// fixed point's `--holdout-after`, say which rows trained.
    training_unknown: bool,
}

impl ParameterSet {
    /// Read the partition `recorded` names and check it against
    /// the rows of every stream the fit bound (not only the streams this
    /// predict covers: the recorded training extent spans them all).
    fn read(
        model: &ir::Model,
        compiled: &sim::CompiledModel,
        config: &crate::fit::config_v2::Problem,
        dt: f64,
        recorded: &crate::fit::training_partition::RecordedPartition,
        held_out: &BTreeMap<String, Vec<u8>>,
    ) -> Result<Self, String> {
        let data = config.data_spec().map_err(|e| {
            format!("this fit has no [data] block to read its training rows from: {e}")
        })?;
        let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)
            .map_err(|e| format!("resolving [data]: {e}"))?;
        let time_opts = crate::caltime_load::TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt,
            t_start: model.simulation.t_start,
            format: crate::caltime_load::TimeFormat::Auto,
        };
        let rows: BTreeMap<String, Vec<crate::fit::observed_rows::ObservedRow>> =
            crate::fit::runner::resolve_and_load_obs_streams(
                model, compiled, bindings.training_by_source(), &time_opts)?
            .into_iter()
            .map(|s| (s.name, s.rows))
            .collect();
        let (record, partition) =
            crate::fit::training_partition::TrainingPartition::read(recorded, &rows)?;
        let mut holdout_sources: Vec<String> = bindings.holdout_by_source().keys().cloned().collect();
        let mut held_out_times: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (stream, bytes) in held_out {
            for h in crate::fit::forecast_output::held_out_rows(bytes)? {
                if h.file == crate::fit::forecast_plan::RowFile::Holdout {
                    if let Some(o) = model.observations.iter().find(|o| &o.name == stream) {
                        holdout_sources.push(o.source.clone());
                    }
                }
                // A `[data.holdout]` file is not read here, so a row in a
                // period it covers stays `future`; a row past the training
                // prefix of the training file is held out.
                if h.file == crate::fit::forecast_plan::RowFile::Training {
                    held_out_times.entry(stream.clone()).or_default().push(h.row.label);
                }
            }
        }
        holdout_sources.sort();
        holdout_sources.dedup();
        let training_unknown = matches!(
            recorded,
            crate::fit::training_partition::RecordedPartition::Given(r)
                if r.rule == crate::run_meta::PartitionRuleRecord::All
        );
        Ok(ParameterSet { record, partition, holdout_sources, held_out_times, training_unknown })
    }

    /// The role of a predicted row past the stream's data rows, at time `t`:
    /// held out when the stored problem holds a row of `stream` there, else
    /// future.
    fn role_past_data(&self, stream: &str, t: f64) -> RowRole {
        let held = self.held_out_times.get(stream).is_some_and(|ts| ts.iter().any(|h| (h - t).abs() <= 1e-9 * h.abs().max(1.0)));
        if held { RowRole::HeldOut } else { RowRole::Future }
    }

    /// The role of the data row at `position` of the leaf stream `stream`.
    fn role(&self, stream: &str, position: usize) -> RowRole {
        if self.training_unknown {
            return RowRole::Unknown;
        }
        RowRole::of_data_row(position, self.partition.rows(stream))
    }
}

/// One stream's part of the one-step state set: the rows the filter walked,
/// and how many of them the fit held out.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct FilteredStream {
    stream: String,
    /// Every row of the stream's bound file; an `NA` row is a row.
    rows: usize,
    /// The rows after the fit's training prefix.
    held_out_rows: usize,
    /// The row digest of `rows`, as a partition records a prefix's.
    digest: String,
}

/// What a stored fit run gives a band: its posterior cloud, or, for an
/// optimizer's run, nothing to band, which is refused. A fixed point given
/// with a model file is not a fit run and does not pass through here: its
/// cloud is [`PosteriorDraws::plug_in`], labelled `plug_in`.
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
    /// An optimizer run's winning point (IF2, NLopt): predicted once, as a
    /// fixed point is, labelled `plug_in`. Its training rows are the run's
    /// problem's, so its held-out rows are as verified as a sampler's.
    Winner {
        method: FitAlgorithm,
        name: String,
        backend: crate::args::types::ForwardBackend,
    },
    /// A fit run with no draws cloud that is not an optimizer's (a sampler
    /// whose draws are not written yet); the variant carries the method and
    /// the name the refusal gives.
    PlugIn {
        method: Option<FitAlgorithm>,
        name: String,
    },
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
    /// One chain's draws alone, by the number its `chain_N/` directory, the
    /// `fit summary` per-chain table, `--exclude-chains` and `draws.tsv`'s
    /// `chain` column all carry (gh#990).
    One(ChainIndex),
}

impl ChainLabel {
    /// The value written into the `chain` column.
    pub fn as_cell(&self) -> String {
        match self {
            ChainLabel::All => "all".to_string(),
            ChainLabel::One(c) => c.to_string(),
        }
    }
}

/// The producing method's own convergence numbers, copied from its summary, so
/// a band is never silent about whether the fit behind it settled. It records
/// the number; it does not reject a band on it (the refusal policy is the
/// deferred guardrail).
///
/// This is **provenance about the fit**, not a statement about any row: it is
/// the worst parameter's R̂ over the whole method, repeated identically on every
/// row of every stream. That is why the columns it writes are named
/// `fit_rhat_max` / `fit_ess_min` (gh#794) — under their former names
/// `rhat_max` / `ess_min` they read as the R̂ of the predicted value beside
/// them, which they are not. The columns that do describe the row are
/// [`BandRow::mean_conv`] and [`BandRow::pred_conv`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceStatus {
    /// The method reported a Gelman–Rubin R̂ / ESS summary.
    Reported { rhat_max: f64, ess_min: f64 },
    /// No per-method R̂ available (e.g. a single-chain method).
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

/// The cloud a band is drawn from, plus the provenance a band needs to label
/// itself honestly: a fit run's posterior draws, or the copies of one fixed
/// point ([`Self::plug_in`]), told apart by [`Self::treatment`]. The cloud is
/// non-empty **by construction**: each constructor rejects an empty draw set — so a
/// `ParamTreatment::Posterior` / `FitResult::Posterior` can never carry a
/// zero-draw cloud that would quantile to `NaN`.
#[derive(Debug, Clone)]
pub struct PosteriorDraws {
    /// One complete parameter vector per draw (estimated + fixed columns).
    /// Private: the non-empty invariant lives in [`PosteriorDraws::new`].
    draws: Vec<IndexMap<String, f64>>,
    /// Bare method name the cloud came from.
    pub name: String,
    /// The inference algorithm, when recoverable.
    pub method: Option<FitAlgorithm>,
    /// The forward simulator the producing method ran on — the predictive
    /// replays on the SAME backend the fit used, not a hardcoded default.
    pub backend: crate::args::types::ForwardBackend,
    /// The method's convergence summary.
    pub convergence: ConvergenceStatus,
    /// The read-side chain selection that produced this cloud, when one was
    /// active (`--exclude-chains`). `None` for a full-cloud posterior. Carried
    /// so the predictive artifact can stamp its provenance (`predictive.json`
    /// `chain_selection`) — a chain-subset band is never mistakable for a
    /// full-cloud one.
    pub selection: Option<SubsetInfo>,
    /// Each draw's posterior key + the method it came from — `None` when the
    /// cloud was built without them (the test constructors). See [`DrawKeys`].
    keys: Option<DrawKeys>,
    /// `Posterior` for a fit's cloud; `PlugIn` for the copies of one point
    /// ([`Self::plug_in`]). Every band drawn from the cloud carries it.
    treatment: TreatmentKind,
}

/// Each posterior draw's `(chain, draw)` key and the method directory its saved
/// smoothing path would live in (gh#722).
///
/// The key travels WITH the cloud rather than being re-read beside it, because
/// a `value_at(..., last_obs)` has to read the path belonging to the draw it is
/// reported for: a shuffled or offset pairing still bands plausibly, and the
/// band is the thing a situation report quotes.
#[derive(Debug, Clone)]
pub struct DrawKeys {
    /// The method directory holding `chain_<N+1>/trajectories.tsv`.
    pub method_dir: PathBuf,
    /// Per draw, in cloud order: its `(chain, draw)` key, or `None` for a
    /// param-only `draws.tsv` with no key columns.
    pub per_draw: Vec<Option<(usize, usize)>>,
    /// Per draw, in cloud order: its row of `draws.tsv` (0-based, the header
    /// not counted). Under a chain selection this is not the draw's place in
    /// the cloud, which holds only the retained rows; an artifact that names
    /// a draw by row reads it here.
    pub rows: Vec<usize>,
}

/// Which draws actually have a smoothing path ON DISK — [`DrawKeys`] narrowed
/// to the forkable subset, with its size, so the count reported and the paths
/// openable are the same fact.
#[derive(Debug, Clone)]
pub struct SavedPaths {
    pub method_dir: PathBuf,
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
    /// runs only when a quantity needs a conditioned read — a `camdl predict`
    /// with no in-window `value_at` pays nothing.
    pub fn resolve_saved(&self) -> SavedPaths {
        let on_disk = crate::fit::joint::trajectory_keys(&self.method_dir);
        let per_draw: Vec<Option<(usize, usize)>> = self
            .per_draw
            .iter()
            .map(|k| k.filter(|key| on_disk.contains(key)))
            .collect();
        let n_saved = per_draw.iter().filter(|k| k.is_some()).count();
        SavedPaths { method_dir: self.method_dir.clone(), per_draw, n_saved }
    }
}

impl PosteriorDraws {
    /// Build a posterior cloud, rejecting an empty draw set. This is the only
    /// constructor, so an empty "posterior" band is unrepresentable.
    pub fn new(
        draws: Vec<IndexMap<String, f64>>,
        name: String,
        method: Option<FitAlgorithm>,
        backend: crate::args::types::ForwardBackend,
        convergence: ConvergenceStatus,
    ) -> Result<Self, String> {
        if draws.is_empty() {
            return Err(format!(
                "posterior method '{name}' has zero draws — cannot build a posterior \
                 predictive band (check that the method's draws.tsv is non-empty and \
                 that burn-in did not discard every sweep)"
            ));
        }
        Ok(PosteriorDraws {
            draws,
            name,
            method,
            backend,
            convergence,
            selection: None,
            keys: None,
            treatment: TreatmentKind::Posterior,
        })
    }

    /// `copies` copies of one point, labelled [`TreatmentKind::PlugIn`]: what
    /// a fixed-point prediction pools over in place of a posterior. Each copy
    /// runs under its own seed, derived from its place as a posterior draw's
    /// is, so the bands carry process and observation noise and no parameter
    /// uncertainty. The cloud carries no chain keys, so the columns that
    /// compare chains or count effective draws are empty. `copies` must be at
    /// least one.
    pub fn plug_in(
        point: IndexMap<String, f64>,
        copies: usize,
        backend: crate::args::types::ForwardBackend,
    ) -> Result<Self, String> {
        if copies == 0 {
            return Err("--n-draws must be at least 1: it is the number of copies of the point the bands pool over".into());
        }
        Ok(PosteriorDraws {
            draws: vec![point; copies],
            name: "fixed point".to_string(),
            method: None,
            backend,
            convergence: ConvergenceStatus::NotAssessed,
            selection: None,
            keys: None,
            treatment: TreatmentKind::PlugIn,
        })
    }

    /// How the cloud treats the parameters: a posterior, or one point.
    pub fn treatment(&self) -> TreatmentKind {
        self.treatment
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
        if keys.per_draw.len() != self.draws.len() || keys.rows.len() != self.draws.len() {
            return Err(format!(
                "internal: {} posterior keys and {} draws.tsv rows for {} draws — the key \
                 lists must be 1:1 with the cloud",
                keys.per_draw.len(),
                keys.rows.len(),
                self.draws.len()
            ));
        }
        self.keys = Some(keys);
        Ok(self)
    }

    /// Each draw's `(chain, draw)` key + its method dir, when the cloud carries
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
/// artifact** (does the chosen method have a draws cloud?), not by method name.
#[derive(Debug, Clone)]
pub enum FitResult {
    /// A Bayesian method wrote a posterior draws cloud.
    Posterior(PosteriorDraws),
    /// No posterior draws cloud resolved for the chosen method — so there is no
    /// band to draw. Either a genuine optimizer method (IF2 / NLopt: one best
    /// point, no cloud) or a Bayesian sampler whose `draws.tsv` is not yet
    /// present (incomplete / still running). The method (when known) tells the
    /// two apart, so the refusal never mislabels a sampler an optimizer (gh#343).
    PointEstimate {
        method: Option<FitAlgorithm>,
        name: String,
        /// The backend the run's method ran on, when its view records it.
        backend: Option<crate::args::types::ForwardBackend>,
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
            FitResult::PointEstimate { method: Some(m), name, backend } if m.is_optimizer() => ParamTreatment::Winner {
                method: m,
                name,
                backend: backend.unwrap_or(crate::args::types::ForwardBackend::ChainBinomial),
            },
            FitResult::PointEstimate { method, name, .. } => ParamTreatment::PlugIn { method, name },
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
    /// The part this row played in the fit (the `role` column).
    pub role: RowRole,
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

/// One (scenario, mode) contribution to a `predictive/<stream>.tsv`: its rows,
/// plus the labels they carry (the scenario overlay axis, the mode axis, the
/// treatment, the quantity, the convergence, the draw count). Stacking several
/// of these under one header is how `camdl predict` writes every scenario's
/// `free_forward` rows plus the `one_step` rows into the same file — the
/// `scenario` and `mode` columns distinguish them.
pub struct PredictiveSection<'a> {
    /// Which chains this section's band was drawn from (gh#794). The `chain`
    /// column is emitted only when some section carries a
    /// [`ChainLabel::One`] — so a run without `--by-chain` has no such column
    /// and renders byte-identically, exactly as `sweep:<param>` appears only
    /// under `--sweep`.
    pub chain: ChainLabel,
    /// The overlay axis value: a scenario name, or `fitted` for the no-overlay
    /// (fitted-model) rows. ALWAYS present (the leading column), the way
    /// `mode`/`treatment` are.
    pub scenario: String,
    /// This cell's sweep coordinate: one `(param, value)` per swept parameter,
    /// in sorted-name order. EMPTY when no `--sweep` (and on the sweep-agnostic
    /// one-step section), so no `sweep:<param>` column is emitted and the header
    /// stays byte-identical to the no-sweep path.
    pub sweep: Vec<(String, f64)>,
    pub mode: PredictiveMode,
    pub treatment: TreatmentKind,
    /// What the band is of.
    pub quantity: Quantity,
    pub convergence: ConvergenceStatus,
    pub n_draws: usize,
    pub rows: &'a [BandRow],
}

/// Render `predictive/<stream>.tsv`: `scenario | time | <dims…> | mode |
/// horizon | treatment | role | quantity | fit_rhat_max | fit_ess_min |
/// rhat_mean | ess_mean | rhat_pred | ess_pred | n_draws | q05 … q95`. Tidy,
/// plot-ready; the axes and the convergence channels are columns so a new
/// predictive cell — a new scenario, a new mode, a new treatment — is more
/// rows, never new consumer code. Several [`PredictiveSection`]s (one per
/// (scenario, mode)) stack under the single header. `horizon`, a forecast
/// target's position after its origin, is empty: neither mode written here
/// has an origin.
///
/// Two convergence channels sit side by side and must not be confused (gh#794).
/// `fit_rhat_max`/`fit_ess_min` are the *producing method's* worst-parameter numbers,
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
        "\tmode\thorizon\ttreatment\trole\tquantity\tfit_rhat_max\tfit_ess_min\
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
            out.push_str(section.mode.as_str());
            // `horizon`: empty, as no row here has a forecast origin.
            out.push('\t');
            out.push('\t');
            out.push_str(section.treatment.label());
            out.push('\t');
            out.push_str(row.role.as_str());
            out.push('\t');
            out.push_str(section.quantity.as_str());
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
        // A `None` leaves the cell empty — that is how a hole is written.
        if let Some(v) = row.value {
            out.push_str(&fmt_value(v));
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
    /// A method that wrote a `draws.tsv` → [`FitResult::Posterior`]; an
    /// optimizer-only fit (no cloud) → [`FitResult::PointEstimate`]; nothing
    /// resolvable → the resolver's actionable error.
    ///
    /// `selection` (`--exclude-chains`) is applied to the cloud through the
    /// draws authority ([`PosteriorDrawsRef::load_params_with_info`]) — the one
    /// place a chain filter meets a cloud — and its provenance is carried onto
    /// the returned [`PosteriorDraws`].
    ///
    /// `run_dir` is where the run is read from — a named leaf, or `segment`
    /// itself — and `segment` is the problem folder whose sidecar names the
    /// estimated parameters.
    pub fn resolve(
        out: &crate::out::Out,
        segment: &Path,
        run_dir: &Path,
        selection: Option<ChainSelection>,
    ) -> Result<FitResult, String> {
        let seg_str = run_dir.to_str().ok_or("fit path is not valid UTF-8")?;
        match posterior_draws::resolve_posterior_draws(seg_str, None) {
            Ok(pref) => {
                let pref = pref.with_selection(selection);
                // Keyed, not param-only: the `(chain, draw)` key locates the
                // draw's saved smoothing path, and building both out of ONE
                // pass over the same rows is what keeps them paired (gh#722).
                let (rows, sel_info) = pref.load_keyed_with_info()?;
                let mut draws: Vec<IndexMap<String, f64>> = Vec::with_capacity(rows.len());
                let mut keys: Vec<Option<(usize, usize)>> = Vec::with_capacity(rows.len());
                let mut cloud_rows: Vec<usize> = Vec::with_capacity(rows.len());
                for r in rows {
                    cloud_rows.push(r.row);
                    keys.push(match (r.chain, r.draw) {
                        (Some(c), Some(d)) => Some((c, d)),
                        _ => None,
                    });
                    draws.push(r.params.into_iter().collect());
                }
                let method_dir = pref
                    .draws_path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| run_dir.to_path_buf());
                // Convergence for the band. With NO chain selection this is the
                // method's stored full-cloud summary. With `--exclude-chains`
                // active the stored summary describes the WHOLE cloud (including
                // the dropped chains), so it must not label a subset band — a
                // band drawn from the clean chains would carry a "did not
                // converge" verdict about a subset that did. Recompute R̂ / ESS
                // over the RETAINED chains through the same seam `fit summary`
                // uses, so the two agree on the same fit + selection (gh#409).
                let convergence = match &pref.chain_selection {
                    Some(sel) => subset_convergence(segment, &pref.draws_path, sel)?,
                    None => read_convergence(out, &method_dir, pref.method),
                };
                // Replay on the SAME backend the method ran on; default only when
                // a bare method dir was passed (no fit view to read it from).
                let backend = pref
                    .backend
                    .map(crate::args::types::ForwardBackend::from)
                    .unwrap_or(crate::args::types::ForwardBackend::ChainBinomial);
                Ok(FitResult::Posterior(
                    PosteriorDraws::new(
                        draws,
                        pref.name,
                        pref.method,
                        backend,
                        convergence,
                    )?
                    .with_selection_info(sel_info)
                    .with_keys(DrawKeys { method_dir, per_draw: keys, rows: cloud_rows })?,
                ))
            }
            // No cloud: classify as a point-estimate fit, naming the method
            // the refusal is about. Only `NoPosterior` means that; any other
            // failure — a leaf refused for counting its chains from 1
            // (gh#990) among them — is the answer, not a reason to reclassify.
            Err(posterior_draws::PosteriorDrawsError::Other(e)) => Err(e),
            Err(posterior_draws::PosteriorDrawsError::NoPosterior(e)) => {
                // `run_dir` is one run (`resolve_fit_run`), so the view holds
                // one leaf; more than one is never resolved by picking.
                if let Some(view) = crate::fit::fit_view::FitView::read(run_dir) {
                    match view.methods.as_slice() {
                        [s] => {
                            return Ok(FitResult::PointEstimate {
                                method: Some(s.method),
                                name: s.method.as_str().to_string(),
                                backend: Some(crate::args::types::ForwardBackend::from(s.backend)),
                            })
                        }
                        [] => {}
                        several => {
                            let list: Vec<String> = several.iter()
                                .map(|s| format!("  {} {}", s.method.as_str(), s.method_dir.display()))
                                .collect();
                            return Err(format!(
                                "{} holds {} runs and this reads one; pass the run's directory:\n{}",
                                run_dir.display(), several.len(), list.join("\n")));
                        }
                    }
                }
                Err(e)
            }
        }
    }
}

/// Read a Bayesian method's convergence summary (`<algorithm>_summary.json`):
/// `max` over its R̂ map, and the min-parameter ESS **through the shared
/// classification** ([`min_ess_over`]) rather than a bare fold. Returns
/// [`ConvergenceStatus::NotAssessed`] when no summary or no R̂ is present (a
/// single-chain method), so a band is never silently "converged".
#[expect(clippy::disallowed_methods, reason = "reads a method's summary JSON from the `fit_store::RunFiles` copy of the stored run the prediction is made from")]
fn read_convergence(out: &crate::out::Out, method_dir: &Path, method: Option<FitAlgorithm>) -> ConvergenceStatus {
    // Chains that began at one point agree by construction; the leaf's R̂
    // is not a verdict (proposal 2026-09-08, §3.4), so the band carries no
    // number — the same demotion `fit summary` applies.
    // An unreadable fit_state.toml is not assessed either: never "converged".
    if !matches!(crate::fit::method_result::point_start(method_dir), Ok(None)) {
        return ConvergenceStatus::NotAssessed;
    }
    let try_read = |name: &str| -> Option<ConvergenceStatus> {
        let bytes = std::fs::read(method_dir.join(name)).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        // gh#728: a summary of another vintage holds `rhat` and `ess` keys
        // that are different estimators under the same names, so labelling a
        // band with them would be labelling something else. Say so once and
        // report the band as not assessed rather than carrying the number.
        if let Err(why) = crate::run_meta::check_fit_summary_schema(&v, name) {
            out.say(format_args!("warning: {why}"));
            return None;
        }
        // Reduce through the SAME classification `fit summary` uses, over the
        // same per-parameter type, read by the same reader, so a band and the
        // summary cannot disagree about one fit (gh#409). A stored summary is
        // just another producer of that map.
        let maps = crate::fit::method_result::ConvergenceMaps::read(&v);
        let diag = PosteriorDiagnostics {
            per_param: maps.per_param(),
            n_samples: 0,
            thin: 1,
            wall_time_secs: None,
            n_chains: 0,
            frozen_chains: maps.frozen_chains(),
            chain_logliks: None,
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
    // no cross-name fallback. A missing summary (an unnamed method dir, or a
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
    let Some(estimated) = crate::run_meta::read_fit_sidecar(segment)?.map(|s| s.estimated) else {
        return Ok(ConvergenceStatus::NotAssessed);
    };
    // A subset of chains that began at one point still began at one point.
    if let Some(leaf) = draws_path.parent() {
        // An unreadable fit_state.toml is not assessed either.
        if !matches!(crate::fit::method_result::point_start(leaf), Ok(None)) {
            return Ok(ConvergenceStatus::NotAssessed);
        }
    }
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
        frozen_chains: Some(sub.frozen_chains.clone()),
        chain_logliks: None,
    };
    match diag.max_rhat_status() {
        score::convergence::MaxRhat::Reported(rhat_max) => {
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
    /// `log_liks[leaf][time_idx]`: under a score, each draw's observation log
    /// density of the row's observed value at the state its `y_rep` was drawn
    /// from, aligned with `samples`; empty where no value is scored.
    log_liks: Vec<Vec<Vec<f64>>>,
}

/// What a score binds into the free-forward sink: per leaf (in
/// `model.observations` order), per emitted time, the observed value scored
/// there and the auxiliary columns its density reads; `None` where no
/// outcome is known.
#[derive(Clone)]
struct FreeForwardOutcomes {
    observed: Vec<Vec<Option<(f64, Vec<(String, f64)>)>>>,
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
        method_dir: &Path,
        model: &ir::Model,
    ) -> Result<Self, String> {
        // No incidence columns: only STATE quantities route to the smoothing
        // path, and requiring `inc_<stream>` here would refuse a file that has
        // none.
        let columns = io::trajectories::TrajColumnSpec::from_model(model, &[]);
        // Group the keys by the file that holds them: the in-file `chain` key
        // is the number of the `chain_N/` directory (gh#990).
        let mut by_chain: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
        for key in per_draw.iter().flatten() {
            by_chain.entry(key.0).or_default().push(*key);
        }
        let mut paths = std::collections::HashMap::new();
        for (chain, keys) in by_chain {
            let traj_path = ChainIndex(chain).dir_in(method_dir).join("trajectories.tsv");
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
    /// The horizon the model *declares* (`simulate { to }`, resolved). The
    /// emission grid may run past it — a forecast row whose declared period
    /// closes beyond the horizon has a snapshot recorded at that close
    /// (`required_output_times` on the job) — but `quantities {}` is reported
    /// over the declared window, so the evaluator reads a trajectory cut back
    /// to this time. Equal to the trajectory's own end whenever nothing was
    /// extended, and then the cut is not taken at all.
    quantity_t_end: f64,
    /// Snapshots predict asked for that the model's declared output schedule
    /// does not record (`required_output_times` on the job). They exist so the
    /// observation projection can read a row's boundaries; `quantities {}` is
    /// reported on the declared output grid, so the evaluator reads the replay
    /// with these removed (their interval flows folded into the next recorded
    /// snapshot, which keeps every cumulative flow it reads unchanged). Empty
    /// when nothing was added, and then no copy is taken.
    extra_snapshot_times: Vec<f64>,
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
    /// Per leaf, per emitted time (aligned 1:1 with `leaf_times`): what the
    /// row covers — the observed rows per the stream's bound declaration, the
    /// forecast tail per the same declaration continued. The likelihood scored
    /// exactly these periods, so the predictive's first bin opens where the
    /// likelihood's did (gh#702) and never reads the warm-up (gh#833).
    leaf_coverages: Vec<Vec<(f64, score::trace::Coverage)>>,
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
    /// The binomial sampler the fit ran (D78): `y_rep`'s binomial draws use it,
    /// as the replay's transitions do.
    binomial: sim::rng::BinomialAlgorithm,
    /// Scenario name → its accumulator. Insertion order (= the engine's canonical
    /// `scenario → point → rep` order, scenario outermost) is preserved so the
    /// rendered files list scenarios in CLI order.
    by_scenario: IndexMap<String, ScenarioAccum>,
    /// A score's outcomes; `None` for `camdl predict`. Reading them draws no
    /// random number, so the `y_rep` draws are the same with or without.
    outcomes: Option<FreeForwardOutcomes>,
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
            log_liks: leaf_times.iter().map(|ts| vec![Vec::new(); ts.len()]).collect(),
        })
    }
}

impl crate::engine::RunSink for PredictiveSink {
    fn merge_cell(&mut self, cell: &crate::engine::CellResult) -> Result<(), String> {
        let model = cell.model();
        // The cell's parameter vector for the observation sampler. Load-bearing on
        // TWO axes: (1) it must read the DRAW's parameters (e.g. an estimated
        // reporting rate), so the posterior predictive carries observation-
        // parameter uncertainty; (2) it must read the SCENARIO's overlay (e.g.
        // `set = { rho = 0.3 }`), so a counterfactual that changes an
        // observation-only parameter actually shifts the predictive bands. The
        // engine resolved BOTH into `cell.model().parameters` (the scenario via
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
        let mut obs_rng = sim::rng::StatefulRng::new(cell.spec.obs_seed).with_binomial(self.binomial);
        // Sample y_rep into a per-leaf scratch first, then commit into this
        // cell's scenario accumulator (so the borrow of `self.compiled` /
        // `self.quant_eval` does not overlap the `&mut self` accumulator borrow).
        let scenario_name = cell.spec.scenario.name().to_string();
        let mut leaf_y: Vec<Vec<f64>> = vec![Vec::new(); model.observations.len()];
        // The *same* draws' latent expected values, `E[y | x_t, θ]` before
        // observation noise (gh#794). Accumulated beside `leaf_y` so the two are
        // the same draw at the same time with the same resolved parameters.
        let mut leaf_mean: Vec<Vec<f64>> = vec![Vec::new(); model.observations.len()];
        // Under a score: `(time index, log density)` of each scored row.
        let mut leaf_ll: Vec<Vec<(usize, f64)>> = vec![Vec::new(); model.observations.len()];
        for (si, obs_ir) in model.observations.iter().enumerate() {
            let times = &self.leaf_times[si];
            if times.is_empty() {
                continue;
            }
            let sampler = sim::inference::obs_model::compile_obs_sample_pf(
                obs_ir,
                self.compiled.clone(),
                &params,
            ).map_err(|e| e.to_string())?;
            // Consumes no RNG, so it cannot perturb the paired-seed replay the
            // sampler above drives.
            let meaner = sim::inference::obs_model::compile_obs_mean_pf(
                obs_ir,
                self.compiled.clone(),
                &params,
            ).map_err(|e| e.to_string())?;
            // Each row is projected over the period the LIKELIHOOD scored it
            // over — the stream's declared coverage — so the first bin opens
            // where the likelihood's did, not at the model origin (gh#702).
            let projected = crate::project_coverages(
                &cell.traj, obs_ir, model, &self.leaf_coverages[si], self.compiled.eval,
            )?;
            let leaf_aux = &self.leaf_aux[si];
            // Where each row is scored: its period's stop, or the instant it
            // reads. The state and the time its likelihood arguments are
            // evaluated at are read there, as the fit scores it, never at the
            // row's label (a `starting_on` row labelled 21 is scored at 28).
            let scored_at: Vec<f64> = self.leaf_coverages[si].iter()
                .map(|&(t, cov)| match cov {
                    score::trace::Coverage::Interval { stop, .. } => stop,
                    score::trace::Coverage::Instant
                    | score::trace::Coverage::Unrecorded => t,
                })
                .collect();
            // The density of a scored row's value, by the likelihood the
            // sampler draws from, at the same state and parameters.
            let density = match &self.outcomes {
                Some(o) if o.observed[si].iter().any(Option::is_some) => Some(
                    sim::inference::obs_model::compile_obs_loglik_pf(obs_ir, self.compiled.clone(), &params)
                        .map_err(|e| e.to_string())?,
                ),
                _ => None,
            };
            let mut stream_vals: Vec<f64> = Vec::with_capacity(times.len());
            let mut stream_means: Vec<f64> = Vec::with_capacity(times.len());
            for (ti, &at) in scored_at.iter().enumerate() {
                let snap = crate::snap_at(&cell.traj, at);
                // Carry the OBSERVED aux (survey denominator) at this obs time
                // into the draw; `&[]` when the leaf has no aux (aligned 1:1 with
                // `times`, so a wrong index is impossible).
                let aux: &[(String, f64)] = leaf_aux.get(ti).map(|v| v.as_slice()).unwrap_or(&[]);
                let y = sampler(projected[ti], at, &snap.int_state.counts, aux, &mut obs_rng);
                stream_vals.push(y);
                stream_means.push(meaner(projected[ti], at, &snap.int_state.counts, aux));
                if let (Some(density), Some(o)) = (&density, &self.outcomes) {
                    if let Some((y_obs, y_aux)) = &o.observed[si][ti] {
                        leaf_ll[si].push((ti, density(*y_obs, projected[ti], at, &snap.int_state.counts, y_aux)));
                    }
                }
            }
            if want_obs {
                // Key by the stream's declared `name` — what `observations.<name>`
                // in the DSL references (v1.1 is unstratified, so name == base).
                // Each draw is named by its row's label, as the predictive
                // table names it (which time names a row: D69).
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
        //
        // The emission grid may have been widened past the *declared* horizon so
        // that a forecast row's closing boundary is a recorded snapshot. That
        // widening is bookkeeping for the observation projection alone: a
        // quantity is reported over the window `simulate { to }` declares, so a
        // series must not gain a row past it and a reduction (a `max`, a first
        // crossing) must not see a day the model does not declare. Cut the
        // replay back before the evaluator reads it — a clone taken only when
        // the grid actually ran past the horizon and the model declares
        // quantities at all, so the ordinary predict pays nothing.
        let quant_traj: Option<sim::Trajectory> = match self.quant_eval {
            Some(_) => {
                let keep = cell
                    .traj
                    .snapshots
                    .iter()
                    .take_while(|s| s.t <= self.quantity_t_end + crate::OBS_SNAP_EPS)
                    .count();
                let has_extra = cell.traj.snapshots[..keep].iter().any(|s| {
                    self.extra_snapshot_times
                        .iter()
                        .any(|e| (e - s.t).abs() <= crate::OBS_SNAP_EPS)
                });
                if has_extra {
                    Some(declared_grid_replay(&cell.traj, keep, &self.extra_snapshot_times))
                } else {
                    (keep < cell.traj.snapshots.len()).then(|| {
                        let mut cut = cell.traj.clone();
                        cut.snapshots.truncate(keep);
                        cut
                    })
                }
            }
            None => None,
        };
        let quant_traj: &sim::Trajectory = quant_traj.as_ref().unwrap_or(&cell.traj);
        let quant_results = self.quant_eval.as_ref().map(|eval| {
            eval.eval_draw(
                &params,
                quant_traj,
                conditioned,
                &self.compiled,
                Some(&obs_set),
                self.obs_anchors,
            )
        });
        let snapshot_times: Vec<f64> = if quant_results.is_some() {
            quant_traj.snapshots.iter().map(|s| s.t).collect()
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
        for (si, lls) in leaf_ll.into_iter().enumerate() {
            for (ti, l) in lls {
                acc.log_liks[si][ti].push(l);
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

/// The first `keep` snapshots of `traj` with the snapshots at `extra` times
/// removed, each removed snapshot's interval flows folded into the next
/// retained snapshot. A snapshot carries the flow since the previous one, so
/// this leaves every cumulative flow at a retained time exactly as it was; the
/// state at a retained time is its own. A removed snapshot after the last
/// retained one has no snapshot to fold into and is dropped (the run then ends
/// at its last declared output time, as it would have without the extras).
fn declared_grid_replay(
    traj: &sim::Trajectory,
    keep: usize,
    extra: &[f64],
) -> sim::Trajectory {
    use sim::state::Flows;
    let is_extra = |t: f64| extra.iter().any(|e| (e - t).abs() <= crate::OBS_SNAP_EPS);
    let mut snapshots: Vec<sim::state::Snapshot> = Vec::new();
    let mut pending: Option<Flows> = None;
    for s in &traj.snapshots[..keep] {
        let mut flows = s.flows.clone();
        if let Some(p) = pending.take() {
            match (&mut flows, &p) {
                (Flows::Int(a), Flows::Int(b)) => {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += *y)
                }
                (Flows::Real(a), Flows::Real(b)) => {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += *y)
                }
                _ => unreachable!("one trajectory carries one flow kind"),
            }
        }
        if is_extra(s.t) {
            pending = Some(flows);
        } else {
            let mut kept = s.clone();
            kept.flows = flows;
            snapshots.push(kept);
        }
    }
    let mut out = traj.clone();
    out.snapshots = snapshots;
    out
}

// ── The verb ───────────────────────────────────────────────────────────────

/// What one `camdl predict` produced: the files it wrote, and what refused while
/// it did.
///
/// The two are independent. A run can write a complete one-step artifact and
/// still have a non-empty `failures` list, which is the whole point: the exit
/// status fails closed on the failure while the object that computed is kept
/// (proposal 2026-09-08 §3.5, §8 item 15).
struct PredictOutcome {
    written: Vec<PathBuf>,
    failures: Vec<DeterministicFailure>,
    /// Streams left out without failing the run (D76): named on stderr and in
    /// `report.json`, and no effect on the exit status.
    skipped: Vec<SkippedStream>,
    /// Where the files were written, to be stored as the prediction.
    stored: ToStore,
    /// The rows whose outcomes `camdl score` bound, as the computation saw
    /// them; empty for `camdl predict`, which binds none.
    scored: crate::fit::scoring::Scored,
}

/// Where `camdl predict` takes a prediction's parameters from: the
/// positional argument, resolved.
#[derive(Clone)]
pub(crate) enum PredictSource {
    /// A stored fit run's posterior draws; the prediction is stored inside
    /// the run.
    Posterior(std::sync::Arc<crate::fit_store::OpenedFit>),
    /// One point given on the command line, with its model, data and
    /// numerics stored; the prediction is stored under `prediction/`.
    FixedPoint(std::sync::Arc<crate::fit::fixed_point::FixedPoint>),
}

impl PredictSource {
    /// The name of the scenario arm with no overlay: `fitted` for a
    /// posterior, and `(none)`, as in `simulate`, for a fixed point, which
    /// was not fitted here.
    fn no_overlay(&self) -> &'static str {
        match self {
            PredictSource::Posterior(_) => crate::args::FITTED,
            PredictSource::FixedPoint(_) => crate::sim_job::IMPLICIT_ARM_LABEL,
        }
    }

    /// The store the prediction is stored in: the one holding the run, which
    /// a folder handle names whatever `CAMDL_STORE` is; or the one a fixed
    /// point's model and data were stored in.
    fn store(&self) -> Result<store::Store, String> {
        let root = match self {
            PredictSource::Posterior(fit) => crate::fit_store::store_holding(fit.fit.problem.folder())
                .unwrap_or_else(crate::fit_store::reader_root),
            PredictSource::FixedPoint(p) => p.store_root.clone(),
        };
        crate::camdl_store::open_existing(&root).ok_or_else(|| format!("no run store at {}", root.display()))
    }

    /// The identity and labels of the prediction from this source under
    /// `options`, with the `--design` tables `design`.
    fn identity(
        &self,
        options: &PredictOptions,
        design: BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>,
    ) -> Result<(crate::fit_store::PredictIdentity, crate::fit_store::PredictionLabels), String> {
        match self {
            PredictSource::Posterior(fit) => prediction_identity(fit, options, design),
            PredictSource::FixedPoint(p) => {
                let source = crate::fit_store::ParameterSource::FixedPoint {
                    model: p.model.clone(),
                    params: p.params.clone(),
                    numerics: p.numerics.clone(),
                    trained_through: p.trained_through,
                };
                identity_from(source, p.data.clone(), Some(p.quantities.clone()), options, design, p.model_name.clone())
            }
        }
    }

    /// The prediction from this source under `options` when it is stored and
    /// completed, reused under `info`; `None` when it is not.
    /// Only a mode other than a forecast is looked up: its options are known
    /// before anything is computed, and it reads no design file.
    fn reuse(
        &self,
        options: &PredictOptions,
        info: &store::RunInfo,
    ) -> Result<Option<store::Stored<crate::fit_store::Prediction>>, String> {
        use crate::fit_store::Prediction;
        let store = self.store()?;
        let (identity, labels) = self.identity(options, BTreeMap::new())?;
        let planned = store.plan::<Prediction>(&[(identity.clone(), labels.clone())]).map_err(|e| e.to_string())?;
        let completed = match planned.first() {
            Some(p) => store.get(&p.id).map_err(|e| e.to_string())?.is_some(),
            None => false,
        };
        if !completed {
            return Ok(None);
        }
        match store.begin::<Prediction>(&identity, &labels, info, store::Reuse::IfCompleted).map_err(|e| e.to_string())? {
            store::Begin::Reused(stored) => Ok(Some(stored)),
            store::Begin::Busy(owner) => Err(store::Error::InProgress(owner).to_string()),
            // Completed a moment ago and gone now: compute it.
            store::Begin::Claimed(run) => {
                drop(run);
                Ok(None)
            }
        }
    }
}

/// What `camdl predict` stores its files as: a prediction inside the run whose
/// draws it used, or under `prediction/` for a fixed point.
struct ToStore {
    source: PredictSource,
    /// The scratch directory the files were written into.
    dir: tempfile::TempDir,
    /// The options that change what is written, as keyed.
    options: PredictOptions,
    /// The `--design SOURCE=FILE` arguments, as given.
    design: Vec<String>,
}

/// `camdl predict RUN|MODEL`: the prediction from the stored fit run `args`
/// names, stored inside that run, or from the fixed point `args` gives with
/// a model file, stored under `prediction/`; `info` records the invocation.
/// Each file written is named on stderr; a skipped stream is named and does
/// not fail the command; a deterministic failure fails it after every object
/// that computed is stored. A prediction already stored is reused without
/// computing it again, except a forecast, whose identity holds its targets.
pub(crate) fn predict(
    args: &crate::args::PredictArgs,
    compile: crate::util::CompileSettings,
    info: store::RunInfo,
    request: &store::StoreRequest,
    output: Option<&Path>,
    bindings: &crate::bindings::Bindings,
) {
    // The forcing-extrapolation policy this command's evaluations run under,
    // installed for the rest of the command.
    let forcing =
        crate::forcing_extrapolation::ForcingPolicy::new(args.flags.forcing_extrapolation.allow_forcing_extrapolation);
    let _forcing_route = forcing.install();
    // predict has no `--no-color`: its painters follow the environment
    // and whether each stream is a terminal.
    let out = crate::out::Out::new(false);
    let source = resolve_source(&out, args, compile, &info, request).unwrap_or_else(|e| out.fail(format!("error: {e}")));
    match run_predict(&out, args, compile, source, &info, request.reuse(), None) {
        Ok(Outcome::Reused { stored, source }) => {
            let store = source.store().unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            out.say(format_args!("predict: reused the stored prediction {}", stored.folder().display()));
            export(&out, &stored, output);
            // A stored prediction is as complete as when it was computed: the
            // failures its report records fail this command as they failed
            // that one.
            let (failures, skipped) = stored_report(&stored).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            for s in &skipped {
                out.say(format_args!("predict: {}", s.describe()));
            }
            if !failures.is_empty() {
                out.say(format_args!(
                    "predict: {} deterministic failure(s); this stored predictive is incomplete:",
                    failures.len()
                ));
                for f in &failures {
                    out.say(format_args!("  ✗ {}", f.describe()));
                }
                out.fail(format!(
                    "predict: the stored prediction holds {} file(s) — every object that computed — \
                     and `report.json` lists what did not. Exit status 1.",
                    stored.result_files().len()
                ));
            }
            crate::bindings::write(&out, bindings, &store, &invocation);
        }
        Ok(Outcome::Computed(outcome)) => {
            let stored = store_prediction(&outcome.stored, &info, request.reuse())
                .unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let store = outcome.stored.source.store().unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let invocation = info.finish(&store).unwrap_or_else(|e| out.fail(format!("error: {e}")));
            let folder = stored.folder().to_path_buf();
            export(&out, &stored, output);
            // What it wrote is the command's account of itself, not data:
            // stderr, like every other command's `wrote` lines.
            for p in &outcome.written {
                let name = p.strip_prefix(outcome.stored.dir.path()).unwrap_or(p);
                out.say(format_args!("wrote {}", folder.join(name).display()));
            }
            // A skipped stream was never eligible: it is named so the reader
            // knows the artifact leaves it out, and it does not fail the run.
            for s in &outcome.skipped {
                out.say(format_args!("predict: {}", s.describe()));
            }
            if !outcome.failures.is_empty() {
                // What failed, then what was written — in that order, because
                // the first is why the exit status is 1 and the second is what
                // the reader still has.
                out.say(format_args!(
                    "predict: {} deterministic failure(s); this predictive is incomplete:",
                    outcome.failures.len()
                ));
                for f in &outcome.failures {
                    out.say(format_args!("  ✗ {}", f.describe()));
                }
                out.fail(format!(
                    "predict: wrote {} file(s) anyway — every object that computed is on \
                     disk, and `report.json` lists what did not. Exit status 1.",
                    outcome.written.len()
                ));
            }
            crate::bindings::write(&out, bindings, &store, &invocation);
        }
        Err(e) => {
            out.fail(format!("error: {e}"));
        }
    }
}

/// The refusals of a fixed-point prediction that its flags alone decide, made
/// before anything is compiled or stored: a mode the backend cannot run,
/// `--particles` where no filter runs, `--design` outside a forecast,
/// `--binomial` on a backend that draws no binomials, and `--integrator` on a
/// backend other than ODE. The predictor makes the
/// same checks again on the cloud, for both sources.
fn refuse_fixed_point_flags(args: &crate::args::PredictArgs) -> Result<(), String> {
    use crate::args::types::{ForwardBackend, PredictModeArg};
    let mode = predict_mode(args)?;
    let backend = args.fixed_point.backend.unwrap_or(ForwardBackend::ChainBinomial);
    let filterable = backend == ForwardBackend::ChainBinomial;
    match (mode, backend) {
        (Some(PredictModeArg::OneStep), ForwardBackend::Ode) => {
            return Err(fixed_point_one_step_refusal(NotFilterable::Deterministic))
        }
        (Some(PredictModeArg::OneStep), ForwardBackend::Gillespie) => {
            return Err(fixed_point_one_step_refusal(NotFilterable::NotAnInferenceBackend))
        }
        (Some(PredictModeArg::Forecast), _) => {
            if let Some(why) = crate::fit::forecast_output::fixed_point_forecast_refusal(backend) {
                return Err(why);
            }
        }
        _ => {}
    }
    let runs_filter = matches!(mode, Some(PredictModeArg::OneStep | PredictModeArg::Forecast)) || (mode.is_none() && filterable);
    if args.flags.particles.is_some() && !runs_filter {
        return Err(format!(
            "--particles sets the particle count of the one-step band's and the forecast's \
             filters, and this predict runs neither ({}).",
            match mode {
                Some(PredictModeArg::FreeForward) => "--mode free_forward".to_string(),
                _ => format!("the {backend} backend has no one-step band"),
            }
        ));
    }
    if !args.flags.design.is_empty() && mode != Some(PredictModeArg::Forecast) {
        return Err("--design supplies the design values of a forecast's targets, and this predict is not a \
                    forecast; pass --horizon DURATION (--mode forecast)."
            .to_string());
    }
    refuse_covariate_outside_forecast(args, mode)?;
    if args.fixed_point.binomial.is_some() && !filterable {
        return Err(format!(
            "--binomial sets the sampler of the chain-binomial draws, and this prediction runs on the {backend} \
             backend, which draws none."
        ));
    }
    crate::fit::fitted_model::refuse_integrator_off_ode(args.fixed_point.integrator, backend, "this prediction")
}

/// Refuse `--covariate` on a prediction that is not a forecast: only a
/// forecast has an origin after which a forcing's values can be stated.
fn refuse_covariate_outside_forecast(
    args: &crate::args::PredictArgs,
    mode: Option<crate::args::types::PredictModeArg>,
) -> Result<(), String> {
    match args.flags.covariate.first() {
        Some(c) if mode != Some(crate::args::types::PredictModeArg::Forecast) => Err(format!(
            "--covariate {}={} states what a forcing's values after a forecast's origin are, and this \
             predict is not a forecast; pass --horizon DURATION (--mode forecast).",
            c.forcing,
            c.statement.as_str()
        )),
        _ => Ok(()),
    }
}

/// The failures and skipped streams the stored prediction's report records:
/// its top-level `report*.json` (`report.json`, or the report of a chain
/// selection, `report-excl<ids>.json`).
fn stored_report(
    stored: &store::Stored<crate::fit_store::Prediction>,
) -> Result<(Vec<DeterministicFailure>, Vec<SkippedStream>), String> {
    let name = stored
        .result_files()
        .into_iter()
        .find(|f| !f.contains('/') && f.starts_with("report") && f.ends_with(".json"))
        .ok_or_else(|| format!("the stored prediction {} holds no report", stored.folder().display()))?;
    crate::fit::failures::read_report(&crate::camdl_store::read_stored(stored, name)?)
}

/// `-o DIR`: a copy of each of the stored prediction's result files at
/// `DIR/<its path in the folder>`. `DIR` is made when its parent exists.
fn export(out: &crate::out::Out, stored: &store::Stored<crate::fit_store::Prediction>, dir: Option<&Path>) {
    let Some(dir) = dir else { return };
    let fail = |e: String| -> ! { out.fail(format!("error: -o {}: {e}", dir.display())) };
    for name in stored.result_files() {
        let bytes = crate::camdl_store::read_stored(stored, name).unwrap_or_else(|e| fail(e));
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|e| fail(format!("cannot create {}: {e}", parent.display())));
        }
        std::fs::write(&path, bytes).unwrap_or_else(|e| fail(format!("cannot write {}: {e}", path.display())));
    }
    out.say(format_args!("predict: exported the prediction's files to {}", dir.display()));
}

/// The source the positional argument names: a model file is a fixed point,
/// resolved and its inputs stored under `info`; anything else is a handle of
/// a stored fit run.
fn resolve_source(
    out: &crate::out::Out,
    args: &crate::args::PredictArgs,
    compile: crate::util::CompileSettings,
    info: &store::RunInfo,
    request: &store::StoreRequest,
) -> Result<PredictSource, String> {
    predict_mode(args)?;
    let text = args.source.as_deref().ok_or(
        "a stored fit run or a model file is required: a run's @label, a run-id prefix or its folder, or a \
         .camdl or .ir.json model",
    )?;
    if crate::fit::fixed_point::names_a_model(text) {
        refuse_fixed_point_flags(args)?;
        let point = crate::fit::fixed_point::resolve(
            out,
            text,
            &args.fixed_point,
            args.flags.forcing_extrapolation.allow_forcing_extrapolation,
            compile,
            info,
            request,
            None,
        )?;
        return Ok(PredictSource::FixedPoint(std::sync::Arc::new(point)));
    }
    Ok(PredictSource::Posterior(crate::fit_store::opened_run(text)?))
}

/// Store the prediction `p` describes: its files are the result files, and
/// its identity is where its parameters came from and the options. A
/// prediction already stored is reused, and the files just written are
/// dropped. `reuse` is the store request's: `--force` computes it again.
#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
#[expect(clippy::disallowed_methods, reason = "reads each `--design` file, declared `#[input(identity)]` on `PredictFlags`, to store it as a `DataTable` the prediction's identity depends on, and the files this process wrote in its scratch directory, to store them as the prediction's result files")]
fn store_prediction(
    p: &ToStore,
    info: &store::RunInfo,
    reuse: store::Reuse,
) -> Result<store::Stored<crate::fit_store::Prediction>, String> {
    use crate::fit_store::Prediction;
    let store = p.source.store()?;
    let design = design_deps(&store, &p.design, info)?;
    let (identity, labels) = p.source.identity(&p.options, design)?;
    let declared = crate::result_files::fit::PredictionFile::of_dir(p.dir.path())?;
    let stored = match crate::camdl_store::begin::<Prediction>(&store, &identity, &labels, info, reuse, declared)
        .map_err(|e| e.to_string())?
    {
        store::Begin::Reused(stored) => stored,
        store::Begin::Busy(owner) => return Err(store::Error::InProgress(owner).to_string()),
        store::Begin::Claimed(run) => {
            let mut files: Vec<PathBuf> = Vec::new();
            let mut stack = vec![p.dir.path().to_path_buf()];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).map_err(|e| format!("cannot read {}: {e}", d.display()))? {
                    let path = e.map_err(|e| e.to_string())?.path();
                    if path.is_dir() { stack.push(path) } else { files.push(path) }
                }
            }
            files.sort();
            for f in files {
                let name = f.strip_prefix(p.dir.path()).expect("under the scratch directory").to_string_lossy().into_owned();
                let bytes = std::fs::read(&f).map_err(|e| format!("cannot read {}: {e}", f.display()))?;
                run.result_file(&name).and_then(|h| h.write_all(&bytes)).map_err(|e| format!("{name}: {e}"))?;
            }
            run.finish().map_err(|e| e.to_string())?
        }
    };
    Ok(stored)
}

/// The prediction `args` asks for from the stored fit run `fit`, reused when
/// stored and computed and stored otherwise, under `info`, which the caller
/// finishes. A prediction that computed with a deterministic failure is
/// stored, as `camdl predict` stores it, and refused here, naming what
/// failed: a score of it would leave those objects out.
pub(crate) fn prediction_of_run(
    out: &crate::out::Out,
    args: &crate::args::PredictArgs,
    fit: std::sync::Arc<crate::fit_store::OpenedFit>,
    compile: crate::util::CompileSettings,
    info: &store::RunInfo,
) -> Result<store::Stored<crate::fit_store::Prediction>, String> {
    prediction_of(out, args, PredictSource::Posterior(fit), compile, info)
}

/// The prediction `args` asks for from `source`, a fit run or a fixed point,
/// as [`prediction_of_run`] makes it of a fit run.
pub(crate) fn prediction_of(
    out: &crate::out::Out,
    args: &crate::args::PredictArgs,
    source: PredictSource,
    compile: crate::util::CompileSettings,
    info: &store::RunInfo,
) -> Result<store::Stored<crate::fit_store::Prediction>, String> {
    // What a forcing held is reported by the score of the prediction, which
    // computes it again.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::recording(
        args.flags.forcing_extrapolation.allow_forcing_extrapolation,
    );
    let _forcing_route = forcing.install();
    let stored = match run_predict(out, args, compile, source, info, store::Reuse::IfCompleted, None)? {
        Outcome::Reused { stored, .. } => stored,
        Outcome::Computed(o) => store_prediction(&o.stored, info, store::Reuse::IfCompleted)?,
    };
    let (failures, _) = stored_report(&stored)?;
    if let Some(f) = failures.first() {
        return Err(format!(
            "the prediction {} did not compute every object ({} failure(s); the first: {}), so its score \
             would leave them out",
            stored.folder().display(),
            failures.len(),
            f.describe()
        ));
    }
    Ok(stored)
}

/// Each `--design SOURCE=FILE` file, stored in `store` under `info` as a data
/// table the prediction's identity depends on, by stream source.
#[expect(clippy::disallowed_methods, reason = "reads each `--design` file, declared `#[input(identity)]` on `PredictFlags`, to store it as a `DataTable` the prediction's identity depends on")]
#[allow(clippy::result_large_err)] // the store's error carries the paths a person acts on, returned once
fn design_deps(
    store: &store::Store,
    specs: &[String],
    info: &store::RunInfo,
) -> Result<BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>, String> {
    let mut design = BTreeMap::new();
    for spec in specs {
        let (source, path) = spec.split_once('=').ok_or_else(|| format!("--design {spec:?}: expected SOURCE=FILE"))?;
        let bytes = std::fs::read(path).map_err(|e| format!("--design {spec:?}: cannot read {path}: {e}"))?;
        let table = store::Hashed::new(
            <crate::camdl_store::Table as store::Content>::from_bytes(&bytes).map_err(|e| e.to_string())?,
        );
        let labels = crate::camdl_store::TableLabels { stream: source.to_string(), path: path.to_string() };
        let dep = store
            .put::<crate::camdl_store::DataTable>(table, &labels, &store::Provenance::none(), info)
            .and_then(|s| s.dep(<crate::camdl_store::DataTable as store::ContentArtifact>::FILE))
            .map_err(|e| format!("storing the design file of `{source}`: {e}"))?;
        design.insert(source.to_string(), dep);
    }
    Ok(design)
}

/// A stored prediction posed again from its identity alone: the arguments
/// and the parameter source that compute it. The files the arguments name
/// (a fixed point's model, point and data, and any design file) are copies
/// of the stored ones, in `scratch`.
pub(crate) struct Reposed {
    pub args: crate::args::PredictArgs,
    pub source: PredictSource,
    /// Per stream source, a copy of the stored lines the prediction
    /// conditions on: a stored fit problem's training lines, or a fixed
    /// point's stored data table.
    pub conditioned: BTreeMap<String, String>,
    /// The step the problem writes (`[config] dt`, a fixed point's `--dt`),
    /// which the dt rule resolves against the model's own.
    pub config_dt: f64,
    _scratch: tempfile::TempDir,
}

impl Reposed {
    /// The compiled model the prediction runs, as an IR file, and the
    /// problem whose `[data]` binds its streams.
    pub(crate) fn model_and_problem(&self) -> (&Path, &crate::fit::config_v2::Problem) {
        match &self.source {
            PredictSource::Posterior(fit) => (fit.model_ir.as_path(), &fit.config.problem),
            PredictSource::FixedPoint(p) => (p.model_ir.as_path(), &p.problem),
        }
    }
}

/// Pose the prediction `stored` again from its identity, its options and its
/// stored inputs, for `camdl score`. Its run store is `store`; a fixed
/// point's inputs are stored again under `info` in the store `request`
/// names, where they are already stored. A fixed point declared fitted
/// through `--holdout-after` binds the outcome file `outcomes` gives for a
/// stream source in place of its stored data, which holds no row after the
/// declaration and so cannot be cut by it; the conditioned lines it yields
/// must be the stored ones, which the prediction's run id then checks. A
/// prediction made with a `--quantities` vocabulary is refused: its identity
/// keeps the vocabulary's digest, not the file.
pub(crate) fn repose(
    out: &crate::out::Out,
    store: &store::Store,
    stored: &store::Stored<crate::fit_store::Prediction>,
    info: &store::RunInfo,
    request: &store::StoreRequest,
    outcomes: &BTreeMap<String, String>,
    check: &dyn Fn(&Path, f64, &BTreeMap<String, String>) -> Result<(), String>,
) -> Result<Reposed, String> {
    use crate::args::types::{CovariateSpec, ForwardBackend, Grid, PredictModeArg, SweepSpec};
    let mut conditioned: BTreeMap<String, String> = BTreeMap::new();
    let id = stored.identity();
    let settings: serde_json::Value =
        serde_json::from_str(&id.options.settings).map_err(|e| format!("the prediction's options: {e}"))?;
    let bad = |what: &str| format!("internal error: the prediction's options hold no valid `{what}`");
    if settings.get("quantities").is_some_and(|q| !q.is_null()) {
        return Err("this prediction evaluated a `--quantities` vocabulary, and its identity keeps the \
                    vocabulary's digest, not the file, so `score` cannot compute it again. Predict \
                    without --quantities and score that prediction."
            .into());
    }
    let strings = |v: &serde_json::Value| -> Vec<String> {
        match v {
            serde_json::Value::String(s) => vec![s.clone()],
            serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        }
    };
    let scratch = tempfile::Builder::new()
        .prefix("camdl-score-")
        .tempdir()
        .map_err(|e| format!("cannot make a scratch directory: {e}"))?;
    let write = |rel: &str, bytes: &[u8]| -> Result<String, String> {
        let path = scratch.path().join(rel);
        std::fs::create_dir_all(path.parent().expect("a file in the scratch directory")).map_err(|e| e.to_string())?;
        std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(path.to_string_lossy().into_owned())
    };
    let table_bytes = |dep: &store::Dep<crate::camdl_store::DataTable>| -> Result<Vec<u8>, String> {
        let t: store::Stored<crate::camdl_store::DataTable> = store
            .get(dep.id())
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("the data table {} the prediction names is not stored", dep.id().key().short12()))?;
        crate::camdl_store::read_stored(&t, <crate::camdl_store::DataTable as store::ContentArtifact>::FILE)
    };

    let mut flags = crate::args::PredictFlags {
        mode: match settings["mode"].as_str() {
            Some("all") => None,
            Some("free_forward") => Some(PredictModeArg::FreeForward),
            Some("one_step") => Some(PredictModeArg::OneStep),
            Some("forecast") => Some(PredictModeArg::Forecast),
            _ => return Err(bad("mode")),
        },
        horizon: settings["reach"].as_str().map(str::to_string),
        n_draws: Some(settings["n_draws"].as_u64().ok_or_else(|| bad("n_draws"))? as usize),
        seed: Some(settings["seed"].as_u64().ok_or_else(|| bad("seed"))?),
        stream: strings(&settings["stream"]),
        scenarios: strings(&settings["scenarios"]),
        enable: strings(&settings["enable"]),
        disable: strings(&settings["disable"]),
        particles: settings["particles"].as_u64().map(|n| n as usize),
        ..Default::default()
    };
    flags.forcing_extrapolation.allow_forcing_extrapolation =
        settings["allow_forcing_extrapolation"].as_bool().ok_or_else(|| bad("allow_forcing_extrapolation"))?;
    for s in settings["sweep"].as_array().ok_or_else(|| bad("sweep"))? {
        let name = s[0].as_str().ok_or_else(|| bad("sweep"))?.to_string();
        let values: Vec<f64> = s[1].as_array().ok_or_else(|| bad("sweep"))?.iter().filter_map(|v| v.as_f64()).collect();
        flags.sweep.push(SweepSpec { name, grid: Grid::List(values) });
    }
    if let Some(cov) = settings["forecast"]["covariates"].as_object() {
        for (forcing, statement) in cov {
            let statement = statement.as_str().ok_or_else(|| bad("forecast.covariates"))?;
            flags.covariate.push(format!("{forcing}={statement}").parse::<CovariateSpec>()?);
        }
    }
    for (source, dep) in &id.options.design {
        let path = write(&format!("design/{source}.tsv"), &table_bytes(dep)?)?;
        flags.design.push(format!("{source}={path}"));
    }
    let posterior = crate::args::PosteriorFlags {
        by_chain: settings["by_chain"].as_bool().unwrap_or(false),
        exclude_chains: settings["excluded_chains"].as_array().map(|ids| {
            ids.iter().filter_map(|c| c.as_u64().map(|c| c.to_string())).collect::<Vec<_>>().join(",")
        }),
        chains: None,
        realised_design: settings["forecast"]["design_mode"].as_str() == Some("ex_post"),
    };

    let (text, source, fixed_point) = match &id.source {
        crate::fit_store::ParameterSourceRecord::Posterior { draws, .. } => {
            let run: store::Stored<crate::fit_store::FitRun> = store
                .get(draws.id())
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("the fit run {} the prediction's draws came from is not stored", draws.id().key().short12()))?;
            let text = run.folder().to_string_lossy().into_owned();
            let fit = crate::fit_store::opened_run(&text)?;
            let spec = fit.config.problem.data_spec().map_err(|e| format!("the fit binds no data: {e}"))?;
            let (model, _) = crate::util::load_ir(&fit.model_ir.to_string_lossy())?;
            conditioned = crate::fit::stream_bindings::StreamBindings::resolve(&model, spec)?
                .training_by_source()
                .iter()
                .map(|(s, p)| (s.clone(), p.clone()))
                .collect();
            check(&fit.model_ir, fit.config.problem.config.dt, &conditioned)?;
            (text, PredictSource::Posterior(fit), crate::args::FixedPointFlags::default())
        }
        crate::fit_store::ParameterSourceRecord::FixedPoint { model, trained_through, .. } => {
            let record = serde_json::to_value(&id.source).map_err(|e| e.to_string())?;
            let point = &record["FixedPoint"];
            let name = store
                .invocation(&store::Handle::IdPrefix(stored.id().key().to_hex()))
                .ok()
                .and_then(|inv| inv.members::<crate::fit_store::Prediction>().ok())
                .and_then(|m| m.into_iter().find(|(i, _)| i == stored.id()).map(|(_, l)| l.model))
                .unwrap_or_else(|| "model".to_string());
            let mut envelope = crate::fit_store::stored_envelope(store, model, &name)?;
            let handle = store::Handle::IdPrefix(stored.id().key().to_hex());
            crate::camdl_store::present(store, &handle, model, &mut envelope);
            let reposed = match crate::camdl_store::recorded_presentation(store, &handle, model) {
                Some(hex) => crate::camdl_store::Presentation::Stored(hex),
                None => crate::camdl_store::Presentation::Unknown,
            };
            if let Some(q) = &id.quantities {
                let qs: store::Stored<crate::camdl_store::Quantities> = store
                    .get(q.id())
                    .map_err(|e| e.to_string())?
                    .ok_or("the prediction's quantities are not stored")?;
                let bytes = crate::camdl_store::read_stored(&qs, <crate::camdl_store::Quantities as store::ContentArtifact>::FILE)?;
                let content = <crate::camdl_store::QuantitiesContent as store::Content>::from_bytes(&bytes).map_err(|e| e.to_string())?;
                envelope.model.quantities = content.quantities()?;
                envelope.model.contrasts = content.contrasts()?;
            }
            let model_path = write(
                &format!("{name}.ir.json"),
                serde_json::to_string(&envelope).map_err(|e| e.to_string())?.as_bytes(),
            )?;
            let mut values = toml::Table::new();
            for (p, v) in point["params"]["values"].as_object().ok_or_else(|| bad("params"))? {
                values.insert(p.clone(), toml::Value::Float(v.as_f64().ok_or_else(|| bad("params"))?));
            }
            let params = write("point.toml", toml::to_string(&values).map_err(|e| e.to_string())?.as_bytes())?;
            let numerics = &point["numerics"];
            let backend = match numerics["backend"].as_str() {
                Some("ChainBinomial") => ForwardBackend::ChainBinomial,
                Some("Gillespie") => ForwardBackend::Gillespie,
                Some("Ode") => ForwardBackend::Ode,
                _ => return Err(bad("numerics.backend")),
            };
            let binomial = match numerics["binomial"].as_str() {
                Some("Btpe") => sim::rng::BinomialAlgorithm::Btpe,
                Some("Btrs") => sim::rng::BinomialAlgorithm::Btrs,
                _ => return Err(bad("numerics.binomial")),
            };
            let mut data = Vec::new();
            for (source, dep) in &id.data {
                let path = write(&format!("data/{source}.tsv"), &table_bytes(dep)?)?;
                conditioned.insert(source.clone(), path.clone());
                // The outcome file, named by the source or by a block of it.
                let outcome = outcomes.get(source).or_else(|| {
                    envelope.model.observations.iter().filter(|o| &o.source == source).find_map(|o| outcomes.get(&o.name))
                });
                let bound = match (trained_through, outcome) {
                    (Some(_), Some(file)) => file.clone(),
                    _ => path,
                };
                // `--data` binds leaves: every block reading the source, to
                // its one file, as the prediction's own `--data` did.
                for o in envelope.model.observations.iter().filter(|o| &o.source == source) {
                    data.push(crate::args::types::DataSpec::Named { name: o.name.clone(), path: PathBuf::from(&bound) });
                }
            }
            check(Path::new(&model_path), numerics["dt"].as_f64().ok_or_else(|| bad("numerics.dt"))?, &conditioned)?;
            let fixed_point = crate::args::FixedPointFlags {
                params: vec![PathBuf::from(params)],
                data,
                holdout_after: trained_through.map(|t| format!("{:?}", t.get())),
                backend: Some(backend),
                dt: Some(numerics["dt"].as_f64().ok_or_else(|| bad("numerics.dt"))?),
                binomial: (backend == ForwardBackend::ChainBinomial).then_some(binomial),
                allow_degenerate_rates: numerics["allow_degenerate_rates"].as_bool().ok_or_else(|| bad("numerics"))?,
                ..Default::default()
            };
            let point = crate::fit::fixed_point::resolve(
                out,
                &model_path,
                &fixed_point,
                flags.forcing_extrapolation.allow_forcing_extrapolation,
                crate::util::CompileSettings::resolve(false, false),
                info,
                request,
                Some(reposed),
            )?;
            (model_path, PredictSource::FixedPoint(std::sync::Arc::new(point)), fixed_point)
        }
    };
    let args = crate::args::PredictArgs { source: Some(text), flags, posterior, fixed_point, _removed_fit: None };
    let config_dt = match &source {
        PredictSource::Posterior(fit) => fit.config.problem.config.dt,
        PredictSource::FixedPoint(p) => p.problem.config.dt,
    };
    Ok(Reposed { args, source, conditioned, config_dt, _scratch: scratch })
}

/// A stored prediction computed again with outcomes bound.
pub(crate) struct Recomputed {
    /// The run id of the prediction computed: the stored one's, when it was
    /// posed again faithfully.
    pub id: store::Key,
    /// Its identity's record form, to name where it departs from the stored
    /// prediction's.
    pub record: serde_json::Value,
    /// Every file it wrote, by its path in the prediction's folder.
    pub files: BTreeMap<String, Vec<u8>>,
    pub scored: crate::fit::scoring::Scored,
    /// Each forcing the computation held outside its range, under
    /// `--allow-forcing-extrapolation`.
    pub forcings_held: Vec<score::trace::PrequentialWarning>,
}

/// Compute the prediction `reposed` poses again with `binding` bound, into a
/// scratch directory, under `info`.
#[expect(clippy::disallowed_methods, reason = "reads back the files this process wrote in its scratch directory, which `camdl score` compares with the stored prediction's")]
pub(crate) fn recompute(
    out: &crate::out::Out,
    reposed: &Reposed,
    info: &store::RunInfo,
    binding: &crate::fit::scoring::OutcomeBinding,
) -> Result<Recomputed, String> {
    // A forcing held is recorded in the score, not said as it is met.
    let forcing = crate::forcing_extrapolation::ForcingPolicy::recording(
        reposed.args.flags.forcing_extrapolation.allow_forcing_extrapolation,
    );
    let _forcing_route = forcing.install();
    let compile = crate::util::CompileSettings::resolve(false, false);
    let outcome = run_predict(out, &reposed.args, compile, reposed.source.clone(), info, store::Reuse::Force, Some(binding))?;
    let Outcome::Computed(o) = outcome else {
        return Err("internal error: a score's prediction was reused rather than computed".into());
    };
    let store = o.stored.source.store()?;
    let design = design_deps(&store, &o.stored.design, info)?;
    let (identity, labels) = o.stored.source.identity(&o.stored.options, design)?;
    let planned = store
        .plan::<crate::fit_store::Prediction>(&[(identity.clone(), labels)])
        .map_err(|e| e.to_string())?;
    let id = planned.first().ok_or("internal error: the store planned no prediction")?.id.key();
    let mut files = BTreeMap::new();
    let mut stack = vec![o.stored.dir.path().to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("cannot read {}: {e}", d.display()))? {
            let path = e.map_err(|e| e.to_string())?.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let name = path.strip_prefix(o.stored.dir.path()).expect("under the scratch directory").to_string_lossy().into_owned();
                files.insert(name, std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?);
            }
        }
    }
    Ok(Recomputed {
        id,
        record: serde_json::to_value(store::Identity::record(&identity)).map_err(|e| e.to_string())?,
        files,
        scored: o.scored,
        forcings_held: forcing.trace_warnings(),
    })
}

/// The identity and labels of the prediction of the run `fit` names under
/// `options`: the run's draws and its problem, and every option that changes
/// what is written.
pub(crate) fn prediction_identity(
    fit: &crate::fit_store::OpenedFit,
    options: &PredictOptions,
    design: std::collections::BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>,
) -> Result<(crate::fit_store::PredictIdentity, crate::fit_store::PredictionLabels), String> {
    let run = fit.fit.run.as_ref().ok_or("a prediction is made from a run, and the handle names a problem")?;
    identity_from(
        crate::fit_store::ParameterSource::Posterior { draws: run.dep_all(), problem: fit.fit.problem.dep_all() },
        std::collections::BTreeMap::new(),
        fit.quantities.clone(),
        options,
        design,
        fit.model_name(),
    )
}

/// The identity and labels of a prediction whose parameters come from
/// `source`, conditioned on `data`, evaluating `quantities`, under `options`
/// with the `--design` tables `design`; `model` is the model file's stem.
fn identity_from(
    source: crate::fit_store::ParameterSource,
    data: std::collections::BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>,
    quantities: Option<store::Dep<crate::camdl_store::Quantities>>,
    options: &PredictOptions,
    design: std::collections::BTreeMap<String, store::Dep<crate::camdl_store::DataTable>>,
    model: String,
) -> Result<(crate::fit_store::PredictIdentity, crate::fit_store::PredictionLabels), String> {
    use crate::fit_store::{PredictIdentity, PredictMode, PredictionLabels};
    let mode = match options.mode {
        "free_forward" => PredictMode::FreeForward,
        "one_step" => PredictMode::OneStep,
        "forecast" => PredictMode::Forecast,
        _ => PredictMode::All,
    };
    let settings = serde_json::to_value(options).map_err(|e| e.to_string())?;
    crate::canonical_json::ensure_finite(options)?;
    let identity = PredictIdentity {
        source,
        data,
        quantities,
        options: crate::fit_store::PredictOptions {
            mode,
            settings: serde_json::to_string(&crate::canonical_json::canonical(&settings)).map_err(|e| e.to_string())?,
            design,
        },
    };
    let labels = PredictionLabels { mode: mode.label().to_string(), scenarios: options.scenarios.clone(), model };
    Ok((identity, labels))
}

/// The stored prediction of the run `fit` names under `options`, if one is
/// stored in `store`.
pub(crate) fn stored_prediction(
    store: &store::Store,
    fit: &crate::fit_store::OpenedFit,
    options: &PredictOptions,
) -> Result<Option<store::Stored<crate::fit_store::Prediction>>, String> {
    // A table's derived prediction reads no design file.
    let (identity, labels) = prediction_identity(fit, options, std::collections::BTreeMap::new())?;
    let planned = store.plan::<crate::fit_store::Prediction>(&[(identity, labels)]).map_err(|e| e.to_string())?;
    match planned.first() {
        Some(p) => store.get(&p.id).map_err(|e| e.to_string()),
        None => Ok(None),
    }
}

/// One fit leaf's loaded observed series: its logical stream, stratum, times,
/// observed cells (`None` = a hole), and per-observation auxiliary columns
/// (e.g. a survey denominator `n = n_examined`) aligned 1:1 with `times`.
struct LeafObs {
    source: String,
    stratum: Vec<(String, String)>,
    times: Vec<f64>,
    /// What each observed row covers, per the stream's declaration — the same
    /// schedule the fit's likelihood scored, so the predictive's first bin
    /// opens where the likelihood's did (gh#702, gh#833).
    stream_times: sim::inference::StreamTimes,
    observed: Vec<Option<f64>>,
    /// `aux[i]` = the auxiliary `(column, value)` pairs observed at `times[i]`.
    /// Carried into the free-forward predictive so a data-supplied denominator
    /// (`binomial(n = n_examined, …)`) draws `y_rep ~ binomial(n_examined, p̂)`
    /// rather than `binomial(0, …) = 0`.
    aux: Vec<Vec<(String, f64)>>,
}

/// The predictive `--mode` and `--horizon` ask for: `None` is every
/// applicable one. `--horizon` alone means the forecast; the forecast needs a
/// reach; and a reach means nothing to another mode.
fn predict_mode(args: &crate::args::PredictArgs)
    -> Result<Option<crate::args::types::PredictModeArg>, String>
{
    use crate::args::types::PredictModeArg;
    // `--horizon` once named the mode; answer that spelling with the flag
    // that does it now, rather than with a malformed-duration error.
    if let Some(h) = args.flags.horizon.as_deref() {
        let mode = h.trim().replace('-', "_");
        if matches!(mode.as_str(), "free_forward" | "one_step" | "forecast") {
            return Err(format!(
                "--horizon sets a forecast's reach (a duration such as \"4 weeks\"); the \
                 predictive mode is chosen with --mode {mode}."));
        }
    }
    match (args.flags.mode, args.flags.horizon.as_deref()) {
        (None, Some(_)) => Ok(Some(PredictModeArg::Forecast)),
        (Some(PredictModeArg::Forecast), None) => Err(
            "--mode forecast needs --horizon DURATION, the reach of the forecast from the \
             end of the fit's training rows (e.g. --horizon \"4 weeks\").".into()),
        (Some(m @ (PredictModeArg::FreeForward | PredictModeArg::OneStep)), Some(h)) => Err(format!(
            "--horizon {h:?} sets the reach of a forecast, but --mode {} is not one; use \
             --mode forecast (or omit --mode).",
            if m == PredictModeArg::FreeForward { "free_forward" } else { "one_step" })),
        (m, _) => Ok(m),
    }
}

/// What `camdl predict` did: computed a prediction whose files wait to be
/// stored, or found it already stored.
enum Outcome {
    Computed(PredictOutcome),
    Reused { stored: store::Stored<crate::fit_store::Prediction>, source: PredictSource },
}

/// What a prediction reads from where its parameters come from, besides the
/// parameters: the problem the parameters were fitted on (its model, its
/// data, its settings), the sampler, the rows that trained, and the IR the
/// engine runs.
struct SourceReads {
    /// The problem half a replay reads: model, data, fixed, estimate.
    config: crate::fit::config_v2::Problem,
    /// The binomial sampler every simulation of the prediction draws with:
    /// the fit run's, or a fixed point's `--binomial`.
    binomial: sim::rng::BinomialAlgorithm,
    /// What the manifests name as the source the bands replay: the fit run's
    /// id, or a fixed point's key.
    run_id: String,
    /// The PMMH fit's own filter's particle count, the filters' default; `None`
    /// for another method and for a fixed point.
    fit_pmmh_particles: Option<usize>,
    selection: Option<ChainSelection>,
    /// The posterior cloud, or the point's copies.
    posterior: PosteriorDraws,
    /// The compiled IR every cell of the engine loads.
    compiled_ir: String,
    /// The digest of the stored compiled model the prediction's identity
    /// names: the fit problem's, or the fixed point's.
    stored_model: store::Key,
    _ir_tmp: Option<PathBuf>,
    /// The copy of the run's files, which the contrasts fork from; `None`
    /// for a fixed point.
    run_dir: Option<PathBuf>,
    partition: crate::fit::training_partition::RecordedPartition,
    /// Each stream's held-out rows without their values; empty for a fixed
    /// point, which keeps none.
    held_out: BTreeMap<String, Vec<u8>>,
    schema: Option<ObsSchema>,
}

impl SourceReads {
    /// Read the fit run `fit` holds, or the fixed point `p`.
    fn of(
        out: &crate::out::Out,
        args: &crate::args::PredictArgs,
        compile: crate::util::CompileSettings,
        source: &PredictSource,
    ) -> Result<SourceReads, String> {
        let fit = match source {
            PredictSource::Posterior(fit) => fit,
            PredictSource::FixedPoint(p) => {
                let point = PosteriorDraws::plug_in(
                    p.point.clone(),
                    args.flags.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS),
                    p.backend,
                )?;
                let (identity, _) = source.identity(&PredictOptions::defaults(), BTreeMap::new())?;
                return Ok(SourceReads {
                    config: p.problem.clone(),
                    binomial: p.binomial,
                    // The point, its model, its numerics and its data: the
                    // prediction's identity without its options.
                    run_id: store::Identity::key(&(identity.source, identity.data)).to_hex(),
                    fit_pmmh_particles: None,
                    selection: None,
                    posterior: point,
                    compiled_ir: p.model_ir.to_string_lossy().into_owned(),
                    stored_model: p.model.digest(),
                    _ir_tmp: None,
                    run_dir: None,
                    partition: crate::fit::training_partition::RecordedPartition::Given(p.partition.clone()),
                    held_out: BTreeMap::new(),
                    schema: Some(ObsSchema::from_model(&crate::util::load_ir(&p.model_ir.to_string_lossy())?.0)),
                });
            }
        };
        // 1. The run the handle names (`@label`, a run-id prefix, its folder),
        //    read through the store into a copy of its files.
        let run_dir = fit.run_dir().expect("a run handle has its files").to_path_buf();
        let segment = fit.segment.clone();
        let config = fit.config.clone();
        // The sampler the fit's method ran, as written in fit.toml; the leaf's own
        // record, read in forecast mode, takes precedence (gh#747).
        let configured_binomial = config.method().ok().map(|m| m.algorithm.binomial());
        // D78: every simulation this command runs — the forecast's filters, the
        // one-step filters, the free-forward replay and its `y_rep` draws, the
        // contrast forks — draws with the binomial sampler the fit ran, read from
        // its leaf. The guarantee is the same sampler and recorded configuration,
        // not the fit's realised random numbers.
        let binomial = crate::fit::forecast_output::fit_binomial(&run_dir, configured_binomial)?;
        // gh#896: everything predict writes goes under a directory keyed by the
        // run it read, so a second run's predict cannot overwrite this one's and
        // every file is attributable to one run.
        let run_id: String = fit.fit.run.as_ref().expect("a run handle names a run").id().key().to_hex();
        // D85: the filters' particle count, the PMMH fit's own by default.
        let fit_pmmh_particles = fit_pmmh_particles(&run_dir)?;
        // The problem half is all a replay reads: model, data, fixed, estimate.
        let config = config.problem;

        // 2. Resolve the posterior — by artifact. A point-estimate fit is refused.
        //    `--exclude-chains` is parsed at the boundary into a typed selection and
        //    applied to the cloud through the draws authority (one filter, once).
        let selection =
            ChainSelection::from_flags(args.posterior.chains.as_deref(), args.posterior.exclude_chains.as_deref())?;
        let fit_result = FitResult::resolve(out, &segment, &run_dir, selection.clone())?;
        let posterior = match fit_result.into_treatment() {
            ParamTreatment::Posterior(pd) => pd,
            ParamTreatment::Winner { method, name, backend } => winner_point(args, &run_dir, method, name, backend)?,
            ParamTreatment::PlugIn { method, name } => {
                return Err(plugin_refusal(method, &name, args.source.as_deref().unwrap_or("<run>")))
            }
        };

        // A chain selection actually dropped chains — warn loudly (non-quietable),
        // naming the dropped chains and the bias direction, before any output.
        if let (Some(info), Some(sel)) = (posterior.selection(), selection.as_ref()) {
            warn_active_selection(out, sel.polarity(), info, crate::out::paint::stderr());
        }

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
            // `camdl predict` replays forward trajectories from stored draws — it never
            // recomputes an ODE gradient, so compile lean (`needs_state_grad = false`,
            // gh#439 A2).
            crate::util::resolve_ir_path(&config.model.camdl, false, None, compile)?
        };
        let schema = crate::run_meta::read_fit_sidecar(&segment)?.and_then(|s| s.schema);
        Ok(SourceReads {
            config,
            binomial,
            run_id,
            fit_pmmh_particles,
            selection,
            posterior,
            compiled_ir,
            stored_model: fit.fit.model.digest(),
            _ir_tmp,
            run_dir: Some(run_dir),
            partition: crate::fit::training_partition::RecordedPartition::FitSegment(segment),
            held_out: fit.fit.held_out.clone(),
            schema,
        })
    }
}

/// The one point an optimizer run (`method`, on `backend`) predicts from: its
/// winner, read as `fit summary --params-only` prints it, as one copy
/// labelled `plug_in`. `--n-draws` above one is refused: there is one point.
fn winner_point(
    args: &crate::args::PredictArgs,
    run_dir: &Path,
    method: FitAlgorithm,
    name: String,
    backend: crate::args::types::ForwardBackend,
) -> Result<PosteriorDraws, String> {
    if let Some(n) = args.flags.n_draws.filter(|n| *n > 1) {
        return Err(format!(
            "--n-draws {n}: the '{name}' run is an optimizer fit ({method}), which returns one best-fit point and \
             no posterior, so it is predicted from that one point, once; omit --n-draws or pass --n-draws 1"
        ));
    }
    let text = crate::fit::fit_summary::winner_params_toml(run_dir)?;
    let table: toml::Table = text.parse().map_err(|e| format!("the winning point of the '{name}' run: {e}"))?;
    let mut point = IndexMap::new();
    for (k, v) in &table {
        let x = match v {
            toml::Value::Float(f) => *f,
            toml::Value::Integer(i) => *i as f64,
            other => return Err(format!("the winning point of the '{name}' run: '{k}' is not a number: {other}")),
        };
        point.insert(k.clone(), x);
    }
    let mut pd = PosteriorDraws::plug_in(point, 1, backend)?;
    pd.name = name;
    pd.method = Some(method);
    Ok(pd)
}

/// The prediction `args` pose from `source`, computed into a scratch
/// directory, or found stored under `info` when `reuse` allows it.
///
/// `binding` is `camdl score`'s: the outcomes of the rows the prediction
/// predicts. Bound, they change no weight and draw no random number, so the
/// files written are the prediction's own; the rows they score come back in
/// [`PredictOutcome::scored`]. A one-step pass then continues through the
/// rows after the conditioning ones, predicting and assimilating each
/// (chain-binomial only), and its bands for those rows are returned there and
/// written to no file.
fn run_predict(
    out: &crate::out::Out,
    args: &crate::args::PredictArgs,
    compile: crate::util::CompileSettings,
    source: PredictSource,
    info: &store::RunInfo,
    reuse: store::Reuse,
    binding: Option<&crate::fit::scoring::OutcomeBinding>,
) -> Result<Outcome, String> {
    let SourceReads {
        config,
        binomial,
        run_id,
        fit_pmmh_particles,
        selection,
        posterior,
        compiled_ir,
        stored_model,
        _ir_tmp,
        run_dir,
        partition,
        held_out,
        schema,
    } = SourceReads::of(out, args, compile, &source)?;
    // An optimizer run's winner is one copy: its options record one draw.
    let one_point: crate::args::PredictArgs;
    let args = match (&source, posterior.treatment()) {
        (PredictSource::Posterior(_), TreatmentKind::PlugIn) => {
            one_point = crate::args::PredictArgs {
                flags: crate::args::PredictFlags { n_draws: Some(1), ..args.flags.clone() },
                ..args.clone()
            };
            &one_point
        }
        _ => args,
    };
    let no_overlay = source.no_overlay();
    // Every file is written into a scratch directory, then stored as one
    // prediction.
    let scratch = tempfile::Builder::new().prefix("camdl-predict-").tempdir()
        .map_err(|e| format!("cannot make a scratch directory: {e}"))?;
    let particles = PredictParticles::resolve(args.flags.particles, fit_pmmh_particles)?;
    // The label the artifact carries, read off the cloud: `posterior` for a
    // fit's draws, `plug_in` for a fixed point's copies.
    let treatment_kind = posterior.treatment();

    // 2b. Resolve which predictive(s) to emit. The one-step horizon is gated by a
    // backend witness ([`FilterableFit`]); an explicit `--mode one_step` on a
    // non-filterable fit is a hard error, while the default ("all applicable")
    // silently skips it for those backends. `one_step_fit` is `Some` exactly when
    // the one-step producer will run. The forecast mode runs neither.
    use crate::args::types::PredictModeArg;
    let mode = predict_mode(args)?;
    let want_free_forward = matches!(mode, None | Some(PredictModeArg::FreeForward));
    let one_step_fit: Option<FilterableFit> = match mode {
        Some(PredictModeArg::FreeForward | PredictModeArg::Forecast) => None,
        Some(PredictModeArg::OneStep) => {
            // EXPLICIT request: a non-filterable fit is a hard error with a redirect.
            Some(FilterableFit::from_posterior(posterior.clone()).map_err(|why| match treatment_kind {
                TreatmentKind::Posterior => one_step_refusal(why, posterior.method, &posterior.name),
                TreatmentKind::PlugIn => fixed_point_one_step_refusal(why),
            })?)
        }
        None => {
            // DEFAULT "all applicable": run one-step where the backend supports
            // it; for ODE / Gillespie it is simply not applicable (no error).
            FilterableFit::from_posterior(posterior.clone()).ok()
        }
    };
    if mode == Some(PredictModeArg::Forecast) {
        crate::fit::forecast_output::refuse_unsupported(args, &posterior)?;
    }
    // D85: a filter runs for the one-step band and for the forecast (whose
    // backend the line above has checked). Its particle count is in the
    // address whenever one runs; where none runs, `--particles` would be
    // silently ignored, so it is refused.
    let runs_filter = one_step_fit.is_some() || mode == Some(PredictModeArg::Forecast);
    if args.flags.particles.is_some() && !runs_filter {
        return Err(format!(
            "--particles sets the particle count of the one-step band's and the forecast's \
             filters, and this predict runs neither ({}).",
            match (mode, treatment_kind) {
                (Some(PredictModeArg::FreeForward), _) => "--mode free_forward".to_string(),
                (_, TreatmentKind::PlugIn) => format!("the {} backend has no one-step band", posterior.backend),
                _ => format!("the '{}' leaf ran on a backend with no one-step band", posterior.name),
            }));
    }

    let (mut model, _) = crate::util::load_ir(&compiled_ir)?;
    // Every engine below runs the model at `compiled_ir`, so it must be the
    // stored compile the prediction's identity names.
    crate::camdl_store::require_stored_model(&model, stored_model, "prediction", &compiled_ir)?;

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
    //     `quantities` (`ir::model_hash`'s `ir_quantities_excluded_from_hash`), so the
    //     recompiled model and the archived one must hash equal. If they do the
    //     source is the same model that was fit, modulo exactly the reporting
    //     layer we are replacing, and transplanting its quantities is sound. If
    //     they differ the source has drifted and we refuse — the alternative is
    //     reporting a fit through formulas written for a different model.
    let vocabulary: Option<crate::quantities_file::QuantitiesOverride> =
        args.flags.quantities.as_deref().map(crate::quantities_file::QuantitiesOverride::load).transpose()?;
    // The options that key the output directory. A forecast's gain its design
    // inputs (D81) once its plan is built, so the forecast addresses its own
    // directory (`run_forecast`); every other mode's is addressed below,
    // after the forecast branch.
    let mut options = PredictOptions {
        particles: runs_filter.then_some(particles.n),
        ..PredictOptions::of(args, posterior.selection(), vocabulary.as_ref())
    };
    // With no --mode, the modes that apply to the fit's backend, resolved
    // before the options are keyed: where the one-step band does not apply,
    // the default computes the free-forward band alone, and is keyed as
    // `--mode free_forward` is.
    if mode.is_none() && one_step_fit.is_none() {
        options.mode = "free_forward";
    }
    if (!args.flags.design.is_empty() || args.posterior.realised_design) && mode != Some(PredictModeArg::Forecast) {
        return Err(format!(
            "{} supplies the design values of a forecast's targets, and this predict is not a \
             forecast; pass --horizon DURATION (--mode forecast).",
            if args.posterior.realised_design { "--realised-design" } else { "--design" }));
    }
    refuse_covariate_outside_forecast(args, mode)?;
    if let Some(v) = &vocabulary {
        // The vocabulary compiles against the model file the problem keeps.
        let mut posed = config.clone();
        posed.model.camdl = match &source {
            PredictSource::Posterior(fit) => fit.model_source()?,
            PredictSource::FixedPoint(p) => p.model_source.clone(),
        }
        .to_string_lossy()
        .into_owned();
        model.quantities =
            quantities_from_vocabulary(&model, &posed, v, compile, args.fixed_point.integrator, treatment_kind)?;
    }

    // gh#616: `camdl predict` HAS the fit's data, so it resolves the model's
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
            // The window the fit itself resolved, read the same way.
            Some(crate::obs_anchor::fit_window(&model, &config).map_err(|e| {
                format!("resolving this model's observation anchors from the fit's data: {e}")
            })?)
        } else {
            None
        };
    // The fit's `[simulate] from` and the anchors, through the one resolution
    // `fit run` used. A stored fit's archived model is already resolved (its
    // start a number), so this is a no-op there and the fit's start replays.
    if crate::obs_anchor::fit_resolves(&model, &config) {
        let moved = crate::obs_anchor::resolve_for_fit(&mut model, &config).map_err(|e| {
            format!("resolving this model's start and observation anchors from the fit's data: {e}")
        })?;
        crate::obs_anchor::report(out, &moved, &model);
    }
    // The start the fit ran from, which every cell's re-load of the IR is
    // moved to (`util::load_run_model`): a no-op on an archived model, whose
    // start is already this number.
    let fitted_t_start = model.simulation.t_start;
    let model = model;
    // A fit made under `allow_forcing_extrapolation` is predicted under it:
    // the predictive replays the fit's own model over its own horizon.
    // The fit's policy replaces the command's own for the rest of the
    // predict. A forcing the command's policy refused before this point has
    // already ended the command, so only what is evaluated from here on
    // differs.
    let _fit_forcing_route = (config.config.allow_forcing_extrapolation
        && !args.flags.forcing_extrapolation.allow_forcing_extrapolation)
        .then(|| {
            crate::forcing_extrapolation::ForcingPolicy::installed()
                .map(|p| p.allowing())
                .unwrap_or_else(|| crate::forcing_extrapolation::ForcingPolicy::new(true))
                .install()
        });
    // …and under its `allow_degenerate_rates`: a rate that collapsed to 0 in
    // the fit collapses to 0 in every replay of it (the one-step filters, the
    // free-forward replay, the forecast, the contrast forks), rather than
    // failing the replay of a model the fit ran. Carried on every model this
    // command compiles: `compiled` below and the models derived from it, and
    // the free-forward job's cells (`SimulateJob::eval`).
    let eval = crate::env::eval_settings(config.config.allow_degenerate_rates);
    // One calendar block for every sidecar manifest this run writes
    // (predictive / observed / quantities), read once off the model.
    let calendar = ir::caltime::CalendarAxis::from_model(&model).map_err(|e| e.to_string())?;
    // The step this command walks, for both horizons: the step the FIT ran
    // at, by the one dt rule (`crate::dt_rule`) — the model's declared dt, else
    // the fit's `[config] dt`. The one-step band re-filters the observed
    // series through the fitted model, and the free-forward replay runs the
    // fitted model forward; either one on another step would be a different
    // process than the posterior describes (gh#708). The two grids used to be
    // separate and could disagree; under the rule they are one, so the notice
    // that named the difference is gone.
    let (dt, one_step_dt) = predict_step_grids(model.simulation.dt, config.config.dt);

    // 3b. Parse the prospective scenario overlay (reusing simulate's ScenarioRef
    // surface). No `--scenario` → a single `fitted` row (the fitted model, no
    // overlay). Validate each ref against the model's presets up front so an
    // unknown `--scenario NAME` errors with the available presets, BEFORE any
    // simulation runs (the resolver's actionable message).
    let scenario_refs = args.flags.scenario_refs_named(no_overlay)?;
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
    //
    // WHERE the refusal lands depends on what this run was asked for, because
    // the scenario overlay is a free-forward thing only — the one-step band
    // filters the OBSERVED data through the fitted model and carries no overlay
    // at all. When the tail is being built, this is a deterministic failure of
    // the tail, recorded below and reported in `report.json`; when it is not
    // (`--mode one_step`), the scenario could never have been honoured by
    // anything this run produces, so it is a plain usage error with no artifact
    // to salvage.
    let scenario_horizon_refusal: Option<String> = scenario_refs.iter().find_map(|sref| {
        crate::util::refuse_scenario_horizon(
            &model, Some(sref.name()), "predict",
            "the predictive replays every scenario at the model's own horizon, \
             which is one window for the whole run",
        )
        .err()
    });
    if let Some(reason) = &scenario_horizon_refusal {
        if !want_free_forward {
            return Err(reason.clone());
        }
    }
    // The fitted model is the model the fit ran: its own `scenario` /
    // `enable` / `disable` (from the archived config) is the base of every
    // arm — the `fitted` row replays exactly it, a `--scenario` /
    // `--enable` / `--disable` arm applies its own changes on top, and the
    // one-step filter runs it. Said once, so a reader of the bands knows
    // which interventions the `fitted` row carries.
    let fit_base = crate::util::FitBase::of(&config)?;
    if !fit_base.is_empty() {
        out.say(format_args!(
            "predict: the fitted model runs the fit's own {}; every scenario \
             arm applies its changes on top of it",
            fit_base.describe()
        ));
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
    // `(param, value)` list — the one expansion `simulate --sweep` uses too
    // (`crate::sweep`), so both verbs give the same `point_id` to the same
    // flags. No `--sweep` → a single empty cell, so the free-forward production
    // runs exactly once, byte-identical to the no-sweep path. A swept value
    // rides in the SAME draw/sweep tier as a draw (it OVERRIDES the swept
    // parameter in each draw row), so the resolver applies it below the
    // scenario tier. A scenario touching a swept parameter is refused (3d); on
    // the parameters it does touch, the scenario's tier applies.
    let sweep_points: Vec<Vec<(String, f64)>> = crate::sweep::expand_sweep_specs(&args.flags.sweep);
    let swept_params: std::collections::BTreeSet<String> =
        args.flags.sweep.iter().map(|s| s.name.clone()).collect();

    // Each swept parameter must be a real model parameter, named at most once —
    // a sweep over an undeclared (or duplicated) name would vary a `sweep:<param>`
    // column while the dynamics never move (a silent no-op).
    crate::sweep::check_swept_names(&args.flags.sweep, &model)?;

    // 3d. A scenario and a sweep on the SAME parameter is a hard error: the
    // scenario's `set` would override (its `scale` rescale) every sweep value,
    // so the grid would not vary that parameter (gh#572). Worded by the same
    // formatter as the engine's guard, over the same footprint
    // (`scenario_param_footprint`), so the commands cannot disagree. Checked
    // here rather than by the engine because predict folds the sweep into the
    // draw rows, which the engine sees as generated draws. Runs BEFORE any
    // simulation.
    if !swept_params.is_empty() {
        let varied = crate::params_resolver::coordinate_values(
            sweep_points.iter().flat_map(|cell| cell.iter().map(|(k, v)| (k.as_str(), *v))),
        );
        let mut refusals: Vec<String> = Vec::new();
        for sref in &scenario_refs {
            if let Some(msg) = crate::params_resolver::scenario_coordinate_collision(
                &model,
                sref,
                crate::params_resolver::UserCoordinate::Sweep,
                &varied,
            )? {
                refusals.push(msg);
            }
        }
        if !refusals.is_empty() {
            return Err(refusals.join("\n\nerror: "));
        }
    }

    // 4. Validate the draw schema BEFORE simulating: every draw column must be a
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
    let model_for_one_step = model_for_obs.clone();
    let compiled = std::sync::Arc::new(
        sim::CompiledModel::new(model_for_obs)
            .map_err(|e| format!("compiling model for prediction: {e:?}"))?
            .with_eval_settings(eval),
    );

    // D76: the streams this predict covers. A stream the fit bound data to is
    // eligible; one with no data is skipped and named unless `--stream` asked
    // for it, in which case it is a failure. One rule for every mode.
    let streams_in: StreamSelection = {
        let data = config.data_spec()
            .map_err(|e| format!("this fit has no [data] block to read the observed series from: {e}"))?;
        let bindings = crate::fit::stream_bindings::StreamBindings::resolve(&model, data)
            .map_err(|e| format!("resolving [data]: {e}"))?;
        let bound = bindings.training_by_source();
        let site = match mode {
            Some(PredictModeArg::Forecast) => Site::ForecastStream,
            _ => Site::Stream,
        };
        select_streams(&model.observations, |o| bound.contains_key(&o.source), &args.flags.stream, site)?
    };

    // The forecast mode: the fixed-origin forecast and nothing else.
    if mode == Some(PredictModeArg::Forecast) {
        let crate::fit::forecast_output::ForecastRun {
            out_dir, mut written, failures, skipped, options: forecast_options, scored,
        } = crate::fit::forecast_output::run_forecast(
            out,
            crate::fit::forecast_output::ForecastJob {
                args,
                model: &model,
                seeded: &model_for_one_step,
                compiled: &compiled,
                config: &config,
                fit_base: &fit_base,
                posterior: &posterior,
                partition: &partition,
                out_dir: scratch.path(),
                held_out: &held_out,
                run_id: &run_id,
                options: &options,
                calendar: &calendar,
                schema: schema.as_ref(),
                dt: one_step_dt,
                seed: args.flags.seed.unwrap_or(1),
                binomial,
                selection: &streams_in,
                scenarios: &scenario_refs,
                sweep_points: if args.flags.sweep.is_empty() { &[] } else { &sweep_points },
                particles,
                baseline: no_overlay,
                binding,
            },
        )?;
        written.push(crate::fit::failures::write_report(
            &out_dir,
            // D86: the directory is keyed by the chain selection; the name
            // carries none.
            "report",
            &failures,
            &skipped,
        )?);
        return Ok(Outcome::Computed(PredictOutcome {
            written, failures, skipped,
            stored: ToStore { source: source.clone(), dir: scratch, options: forecast_options, design: args.flags.design.clone() },
            scored: crate::fit::scoring::Scored { plug_in: treatment_kind == TreatmentKind::PlugIn, ..scored },
        }));
    }

    // A prediction already stored is reused, before anything is computed:
    // every option that keys it is resolved here. A score computes the
    // prediction again whether or not it is stored.
    if reuse == store::Reuse::IfCompleted && binding.is_none() {
        if let Some(stored) = source.reuse(&options, info)? {
            return Ok(Outcome::Reused { stored, source });
        }
    }

    // The output directory, keyed by every input that changes what is written
    // there: the run, the engine, and the options as resolved (the chain
    // selection against this fit's chains, the vocabulary by its bytes). Two
    // predictions from different inputs never share it.
    let address_key = options.address_key(&run_id);
    let out_dir = scratch.path().to_path_buf();

    // 5. Load the observed data per leaf (the cadence, the observed half, and
    // what each row covers — resolved against the compiled model's projections).
    // Every eligible leaf is bound, so none is empty here.
    let leaves = load_leaf_obs(&model, &compiled, &config, dt, &streams_in)?;
    // The parameter set (gh#1079): which rows the fit's parameters were fitted
    // to, read from the fit's record and checked against the files. Every
    // predictive row's `role` is read off it, and a fit that recorded none is
    // refused here, before anything is computed.
    let parameter_set = ParameterSet::read(&model, &compiled, &config, dt, &partition, &held_out)?;
    let seed = args.flags.seed.unwrap_or(1);

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
    // this fit. It is the fit's own window, read by the reader the model's
    // anchors use (`obs_anchor::fit_window`): the observed window of the
    // training rows (D40), so held-out rows appended to the file move no
    // `value_at(..., last_obs)`. `None` iff no leaf carries an observation
    // time — the callers below then have nothing to anchor to and say so.
    let quantity_obs_anchors: Option<sim::quantity::ObsAnchorTimes> =
        match leaf_times.iter().all(Vec::is_empty) {
            true => None,
            false => Some(crate::obs_anchor::fit_window(&model, &config).map_err(|e| format!(
                "resolving the observed window `last_obs` / `first_obs` refer to: {e}"))?),
        };

    // What refused. Non-empty ⇒ the artifact this run writes is incomplete, the
    // failures are named in `report.json`, and the exit status is 1 — but every
    // object that DID compute is still written (proposal §3.5, §8 item 15).
    // A stream `--stream` asked for that cannot be produced is the first entry.
    let mut failures: Vec<DeterministicFailure> = streams_in.failures.clone();

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
    // Under a score, the free-forward rows whose outcome is known.
    let mut ff_scored = crate::fit::scoring::Scored::default();

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
        // The whole free-forward tail is produced as ONE recoverable unit
        // (proposal §3.5, §8 item 15). Every refusal inside it — a scenario
        // window this verb cannot honour, an unresolvable observation anchor, a
        // replay that errored — is a fact about the tail, and the one-step band
        // below is a different object: data-conditioned, never reaching the
        // horizon, and already complete by the time any of this runs. So the
        // error is caught, recorded against `Site::FreeForward`, and the run
        // carries on to write what it has; the exit status is still 1.
        //
        // The closure exists only to give the `?`s inside a place to land. It
        // is called immediately, so every mutable capture is released on this
        // statement and the error branch below can reset them.
        let outcome = (|| -> Result<(), String> {
            // gh#561: the scenario horizon this command cannot honour, resolved
            // above. It blocks the tail and nothing else.
            if let Some(reason) = &scenario_horizon_refusal {
                return Err(reason.clone());
            }
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
                // NaN arm explicit: an unresolved horizon arrives as NaN and
                // must fall through to "no forecast window", never to "extend"
                // — and `t_end <= last_obs` alone is false for NaN.
                let t_end = model.simulation.t_end;
                if t_end.is_nan() || last_obs.is_nan() || t_end <= last_obs {
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
                        out.say(format_args!(
                            "predict: stream '{}' stops at its last observation \
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
                        ));
                    }
                    continue;
                }
                let extension = forecast_times(times, &horizon_output_times);
                if extension.is_empty() {
                    if noted.insert(obs_ir.source.clone()) {
                        out.say(format_args!(
                            "predict: stream '{}' stops at its last observation \
                             (t = {last_obs}) even though the model horizon is {}: no \
                             trajectory output time past the data continues this \
                             stream's observed cadence, so there is no grid to project \
                             onto.\n  \
                             Fix: widen `output {{ trajectories {{ ... }} }}` so the \
                             forecast window carries output times on the stream's \
                             reporting cadence.",
                            obs_ir.source,
                            model.simulation.t_end,
                        ));
                    }
                    continue;
                }
                ff_emit_times[si].extend(extension);
            }

            // What each emitted row covers (gh#833): the observed prefix per the
            // stream's bound declaration — the very periods the likelihood scored
            // — and the forecast tail per the same declaration continued.
            let leaf_coverages: Vec<Vec<(f64, score::trace::Coverage)>> = model
                .observations
                .iter()
                .enumerate()
                .map(|(si, o)| {
                    let n_obs = leaf_times[si].len();
                    let observed = leaves.iter().find(|l| leaf_matches(o, l));
                    leaf_row_coverages(
                        o,
                        observed.map(|l| &l.stream_times),
                        &ff_emit_times[si][..n_obs],
                        &ff_emit_times[si][n_obs..],
                        model.simulation.t_start,
                    )
                })
                .collect();

            // ── The snapshots the projection reads ─────────────────────────────
            //
            // A row's value is read off the recorded trajectory, not off
            // integrator state: a stock at the row's label, or, for a flow, the
            // difference of the recorded cumulative flow at the row's two
            // boundaries (`project_coverages`). Each of those times must
            // therefore be a recorded output time. The fit does not need that —
            // the particle filter scores each row from integrator state at its
            // data time — so a data time between output points (data every 2.5
            // days against the default 1-day output) fits fine and would then
            // fail here. Predict integrates its own trajectories per draw, so it
            // decides what they record: it asks for a snapshot at every time its
            // projection reads (`required_output_times` on the job), changing no
            // stored fit and no run identity. Those extra snapshots are internal
            // to the projection — `quantities/` is read off the trajectory cut
            // back to the declared output grid (`PredictiveSink::extra_snapshot_times`).
            //
            // Two kinds of time:
            //
            // - Every label and every interval boundary of every row, observed
            //   or forecast, that is not already on the declared grid and lies
            //   inside the declared window. (Observed rows past the declared
            //   horizon are left to `check_obs_times_on_snapshot_grid`, which
            //   refuses them rather than silently extending the run.)
            // - The *close* of the last forecast row, which can sit past the
            //   horizon: a uniform `covers` form with a closing offset puts it
            //   past the label — `ending_on(time, 7 'days)` closes the row
            //   labelled `t` at `t + 1`, and the last forecast label sits at the
            //   horizon, so the close sits one day beyond it, where nothing was
            //   integrated. Without a snapshot there the only fix left to the
            //   modeller is to move `simulate { to }` by one day, which re-keys
            //   the model and orphans the fit
            //   (2026-09-08-workflow-first-fit-config §1.5, §3.5).
            //
            // A windowed stream (`window_start`/`window_stop`) takes the contiguous
            // continuation `[previous stop, label)` in `leaf_row_coverages`, so its
            // tail closes at its label and this never fires — the list stays empty,
            // and an empty list changes nothing about the run.
            let mut required_output_times: Vec<f64> = Vec::new();
            let mut extended: std::collections::HashSet<String> =
                std::collections::HashSet::new();
            for (si, obs_ir) in model.observations.iter().enumerate() {
                let n_obs = leaf_times[si].len();
                if ff_emit_times[si].len() <= n_obs {
                    continue; // no forecast tail — nothing past the data to close
                }
                // An `Instant` row reads the state at its label, which is an
                // emitted time and therefore already on the grid.
                let Some(&(label, score::trace::Coverage::Interval { start, stop })) =
                    leaf_coverages[si].last()
                else {
                    continue;
                };
                if horizon_output_times.iter().any(|o| (o - stop).abs() <= crate::OBS_SNAP_EPS) {
                    continue; // the schedule already records the close
                }
                required_output_times.push(stop);
                // One note per logical stream, not per stratum leaf.
                if extended.insert(obs_ir.source.clone()) {
                    out.say(format_args!(
                        "predict: stream '{}': the last forecast row (t = {label}) \
                         covers [{start}, {stop}) under its `covers` declaration, \
                         which closes past the declared horizon t = {}. Integrating \
                         to t = {stop} and recording a snapshot there, so the row's \
                         closing cumulative flow can be read; `simulate {{ to }}` \
                         keeps its declared value and the fit is unchanged.",
                        obs_ir.source,
                        model.simulation.t_end,
                    ));
                }
            }
            // Every other time the projection reads, inside the declared window.
            let t_start_decl = model.simulation.t_start;
            let t_end_decl = model.simulation.t_end;
            let mut n_between_grid = 0_usize;
            for rows in &leaf_coverages {
                for &(label, cov) in rows {
                    let reads: [Option<f64>; 2] = match cov {
                        // `Unrecorded` is a prequential-trace artefact; a row built
                        // here never carries it, and its label is the one time known.
                        score::trace::Coverage::Instant
                        | score::trace::Coverage::Unrecorded => [Some(label), None],
                        score::trace::Coverage::Interval { start, stop } => {
                            [Some(start), Some(stop)]
                        }
                    };
                    for t in reads.into_iter().flatten() {
                        let inside = t.is_finite()
                            && t > t_start_decl + crate::OBS_SNAP_EPS
                            && t <= t_end_decl + crate::OBS_SNAP_EPS;
                        let on_grid = horizon_output_times
                            .iter()
                            .any(|o| (o - t).abs() <= crate::OBS_SNAP_EPS);
                        if inside && !on_grid {
                            required_output_times.push(t);
                            n_between_grid += 1;
                        }
                    }
                }
            }
            if n_between_grid > 0 {
                out.say("predict: recording extra trajectory snapshots at the \
                     observation and row-boundary times that fall between the \
                     model's output times, so each row can be projected off the \
                     trajectory the fit scored from integrator state. They are \
                     internal to the projection: `quantities/` stays on the \
                     declared output grid, and the fit is unchanged.");
            }
            required_output_times.sort_by(f64::total_cmp);
            required_output_times.dedup_by(|a, b| (*a - *b).abs() <= crate::OBS_SNAP_EPS);
            // What a quantity must not see: every snapshot the declared grid
            // did not record. (The closing boundary past the horizon is also
            // cut by `quantity_t_end`.)
            let extra_snapshot_times: Vec<f64> = required_output_times.clone();

            // A score's outcomes at the emitted rows: a data row's own value,
            // and the outcome row past the data that measures a tail row's
            // period or instant. An outcome row the grid emits no row for is
            // not scored, and is named.
            let ff_outcomes: Option<FreeForwardOutcomes> = match binding {
                None => None,
                Some(b) => {
                    let mut observed: Vec<Vec<Option<(f64, Vec<(String, f64)>)>>> = Vec::new();
                    for (si, o) in model.observations.iter().enumerate() {
                        let n_obs = leaf_times[si].len();
                        let leaf = leaves.iter().find(|l| leaf_matches(o, l));
                        let mut cells: Vec<Option<(f64, Vec<(String, f64)>)>> = (0..ff_emit_times[si].len())
                            .map(|ti| match (ti < n_obs, leaf) {
                                (true, Some(l)) => l.observed[ti].map(|y| (y, leaf_aux[si].get(ti).cloned().unwrap_or_default())),
                                _ => None,
                            })
                            .collect();
                        let mut unmatched = 0usize;
                        for r in b.after_conditioning(&o.name) {
                            let Some(sim::inference::ObsCell::Scalar(y)) = r.cell else { continue };
                            let hit = (n_obs..ff_emit_times[si].len()).find(|&ti| match (leaf_coverages[si][ti].1, r.measures) {
                                (score::trace::Coverage::Interval { start, stop }, crate::fit::observed_rows::Measures::Period(p)) => {
                                    (start - p.start()).abs() <= crate::OBS_SNAP_EPS && (stop - p.stop()).abs() <= crate::OBS_SNAP_EPS
                                }
                                (score::trace::Coverage::Instant, crate::fit::observed_rows::Measures::Instant(t)) => {
                                    (ff_emit_times[si][ti] - t).abs() <= crate::OBS_SNAP_EPS
                                }
                                _ => false,
                            });
                            match hit {
                                Some(ti) => cells[ti] = Some((y, r.aux.clone())),
                                None => unmatched += 1,
                            }
                        }
                        if unmatched > 0 {
                            ff_scored.notes.push(format!(
                                "{unmatched} outcome row(s) of '{}' past the prediction's data are not scored in \
                                 free_forward mode: the prediction emitted no row measuring their period",
                                o.name
                            ));
                        }
                        observed.push(cells);
                    }
                    Some(FreeForwardOutcomes { observed })
                }
            };

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
            let ff_cap = args.flags.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS);
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
            if treatment_kind == TreatmentKind::Posterior && ff_chain_of_point.iter().all(Option::is_none) {
                out.say("predict: this fit's draws.tsv carries no chain column, so the \
                     per-row convergence columns (rhat_mean / ess_mean / rhat_pred / \
                     ess_pred) are left empty — a between-chain statistic needs to know \
                     which chain each draw came from.");
                if args.posterior.by_chain {
                    out.say("predict: --by-chain has nothing to split on for the same \
                         reason, so no `chain` column is written and the file carries \
                         the pooled band only.");
                }
            }
            if ff_n_draws < posterior.n_draws() {
                out.say(format_args!(
                    "predict: free_forward mode — subsampling {ff_n_draws} of {} \
                     posterior draws (raise with --n-draws)",
                    posterior.n_draws()
                ));
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
                quant_paths.contains(&sim::quantity::QuantityPath::Smoothed);
            // Named, not silent: an `observations.<stream>` reduction anchored
            // inside the record has the same defect, and no saved path carries a
            // y_sim draw to fix it with.
            if let Some(eval) = quant_eval.as_deref() {
                let unconditioned = eval.quantity_names_on(
                    sim::quantity::QuantityPath::ReplayUnconditioned,
                    quantity_obs_anchors,
                );
                if !unconditioned.is_empty() {
                    out.say(format_args!(
                        "predict: quantity `{}` reduces `observations.<stream>` at an \
                         anchor inside the observed record, and is reported on the \
                         free-forward replay. The saved smoothing path carries the \
                         conditioned projection (`inc_<stream>`, a mean), not a draw from \
                         it, so there is nothing conditioned to sample the observation \
                         from (gh#722).\n  \
                         Fix: express the quantity over latent state \
                         (`value_at(<state expr>, last_obs)`), which IS read on the \
                         smoothing path.",
                        unconditioned.join("`, `"),
                    ));
                }
            }
            // The saved subset is resolved only when a quantity needs it — the scan
            // reads every `chain_*/trajectories.tsv`, and a predict with no
            // in-window `value_at` must not pay for it.
            let ff_saved: Option<SavedPaths> = if any_smoothed {
                posterior.keys().map(|k| {
                    let saved = k.resolve_saved();
                    SavedPaths {
                        method_dir: saved.method_dir,
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
                if args.flags.enable.is_empty() && args.flags.disable.is_empty() {
                    Some(no_overlay.to_string())
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
                        out.say(format_args!(
                            "predict: quantity `{names}` is anchored at or before \
                             last_obs — reported on the conditioned smoothing path \
                             p(x|y), not on the free-forward replay (gh#722). \
                             {}/{} replayed draws have a saved path; the other {} are \
                             censored for these quantities, never substituted from the \
                             replay.",
                            s.n_saved,
                            ff_n_draws,
                            ff_n_draws - s.n_saved,
                        ));
                    }
                    (_, None) => {
                        out.say(format_args!(
                            "predict: quantity `{names}` is anchored at or before \
                             last_obs, but `--enable`/`--disable` makes every arm of this \
                             run a counterfactual, for which no conditioned path exists. \
                             They are reported on the free-forward replay, which ignores \
                             the observations they are anchored inside (gh#722)."
                        ));
                    }
                    _ if treatment_kind == TreatmentKind::PlugIn => {
                        out.say(format_args!(
                            "predict: quantity `{names}` is anchored at or before \
                             last_obs, and a fixed-point prediction has no smoothing path \
                             to read it on. It is reported as fully censored rather than \
                             taken from the free-forward replay, which ignores every \
                             observation it is anchored inside."
                        ));
                    }
                    _ => {
                        out.say(format_args!(
                            "predict: quantity `{names}` is anchored at or before \
                             last_obs, but this fit saved no latent path for any replayed \
                             draw — there is nothing conditioned to read them on. They are \
                             reported as fully censored rather than taken from the \
                             free-forward replay, which ignores every observation they are \
                             anchored inside (gh#722).\n  \
                             Fix: re-fit with a PGAS method and `n_trajectories` > 0 \
                             (only PGAS saves the smoothing paths; PMMH and \
                             particle-filter methods write none), then re-run \
                             `camdl predict`."
                        ));
                    }
                }
            }
            // A counterfactual arm keeps the replay, and says so once — its rows sit
            // in the same file as the fitted arm's, under a `scenario` column, so a
            // reader comparing them must know they are two different objects.
            if any_smoothed && scenario_refs.iter().any(|s| s.name() != no_overlay) {
                out.say("predict: the scenario arms report their anchored quantities on \
                     their OWN free-forward replay — the smoothing path was inferred under \
                     the fitted model, and the data a counterfactual would have generated \
                     do not exist. `quantities.json` tags each entry with `evaluated_on`.");
            }
            // Fan the (draws × scenarios) replay grid across Rayon. The engine seeds
            // each cell by its planned `point_idx`/`rep` (`process_seed_for`),
            // independent of execution order, so parallelism never perturbs a
            // trajectory (engine.rs) — the bands are byte-identical to a sequential
            // replay of the same subsample. `camdl predict` has no thread-budget flag,
            // so default to the machine width; `RAYON_NUM_THREADS` still caps the
            // global pool.
            let ff_parallel = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

            // A swept value is written into each draw row (below), so the
            // resolver sees one draw/sweep-tier value and records no override:
            // the replacement of the draws' values is reported here instead,
            // once per swept parameter over every replayed draw.
            out.notices(&crate::params_resolver::sweep_over_draws_notices(
                ff_draws.iter().copied(), &sweep_points,
            ));

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
                            &saved.method_dir,
                            &model,
                        )?)
                    }
                    _ => None,
                };
                let mut sink = PredictiveSink {
                    compiled: compiled.clone(),
                    quantity_t_end: model.simulation.t_end,
                    extra_snapshot_times: extra_snapshot_times.clone(),
                    leaf_times: ff_emit_times.clone(),
                    leaf_aux: leaf_aux.clone(),
                    leaf_coverages: leaf_coverages.clone(),
                    quant_eval: quant_eval.clone(),
                    obs_anchors: quantity_obs_anchors,
                    conditioned,
                    chain_of_point: ff_chain_of_point.clone(),
                    binomial,
                    by_scenario: IndexMap::new(),
                    outcomes: ff_outcomes.clone(),
                };

                for sref in &scenario_refs {
                    // Draw rows for this sweep cell: each posterior draw with the swept
                    // parameters OVERWRITTEN to this cell's grid values (the draw
                    // supplies every other parameter). The swept value lands in the
                    // SAME draw/sweep tier as the draw (`point_overrides`), so the
                    // resolver applies the scenario's `set`/`scale`/`enable`/`disable`
                    // ON TOP — exactly once. 3d refused any scenario touching a swept
                    // parameter, so the two never meet on one parameter.
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
                        // (chain_binomial / ode), resolved from the method — never a
                        // hardcoded default.
                        backend: posterior.backend,
                        dt,
                        // D78: the sampler the fit ran.
                        binomial,
                        eval,
                        integrator: None,
                        // Generated posterior draws (not a user-authored file), so
                        // the engine's collision guard does not check them: a scenario
                        // over posterior draws is the counterfactual predict runs. The
                        // swept columns folded into these rows were checked by 3d.
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
                        t_start_override: Some(fitted_t_start),
                        // The forecast rows' closing boundaries, so the
                        // projection finds a recorded snapshot at each. Empty
                        // unless a declared period closes off the recorded grid,
                        // and an empty list leaves the run untouched.
                        required_output_times: required_output_times.clone(),
                        // gh#641: the predictive replays from the model's init {} at
                        // each posterior draw; a filtered-state restart is a
                        // `simulate --init-state` surface, not a `camdl predict` one.
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
                        // The posterior rows carry the fit's [fixed] columns
                        // themselves; no separate tier-2 block.
                        fit_fixed: indexmap::IndexMap::new(),
                        // Every arm runs on the fit's own intervention selection.
                        fit_base: fit_base.clone(),
                        set_vec_entries: vec![],
                        table_files: vec![],
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
                            &model, accum, &ff_emit_times, &leaves, &parameter_set,
                            schema.as_ref(), args.posterior.by_chain,
                        )?,
                    });
                    if let Some(o) = &ff_outcomes {
                        ff_scored.cells.extend(free_forward_scored(
                            &model, accum, o, &ff_emit_times, &leaf_coverages, &leaf_times, &parameter_set,
                            scenario_name, sweep_pt,
                        ));
                    }
                }
            }

            if !stacked.is_empty() {
                let rendered = stacked.finish(&calendar)?;
                // gh#715: one reporting expression that produced a NaN/±∞ is a
                // failure of THAT entry. It drops out of the table and is named
                // with the draw that produced it; every other quantity, and
                // every stream artifact below, is still written.
                for f in &rendered.failures {
                    // The fact here; the reason travels with the failure and is
                    // printed once, at the end, beside the exit status.
                    out.say(format_args!(
                        "predict: quantity `{}` refused and is dropped from the \
                         table — the reason is below and in report.json.",
                        f.name
                    ));
                    failures.push(DeterministicFailure::NonFinite {
                        at: Site::Quantity(f.name.clone()),
                        draw: f.draw,
                        reason: f.reason(),
                    });
                }
                quantity_outputs = rendered.files;
                quantity_manifest = Some(rendered.manifest);
            }
            free_forward = Some(ff_cells);
            Ok(())
        })();
        if let Err(reason) = outcome {
            // Only the fact, here: the reason travels with the failure and is
            // printed once, at the end, beside the exit status it explains.
            out.say("predict: the free-forward tail refused — continuing with what is \
                 computable; the reason is below and in report.json.");
            failures.push(DeterministicFailure::EvaluationFailed {
                at: Site::FreeForward,
                reason,
            });
            // Half a tail is worse than none: the cells that did complete were
            // banded against a design the failing cell was part of, and a
            // quantities table missing one scenario reads as a table over the
            // scenarios it has. Drop everything this horizon produced, keep
            // everything the one-step half is about to.
            free_forward = None;
            quantity_outputs.clear();
            quantity_manifest = None;
            ff_n_draws = 0;
            ff_draws_per_chain.clear();
            ff_scored = crate::fit::scoring::Scored::default();
            if binding.is_some() {
                ff_scored.notes.push("the free-forward tail refused, so no free_forward row is scored".into());
            }
        }
    }

    // ── One-step horizon: per-draw bootstrap filter over the data, pooled. Runs
    // only when the witness was built (a filterable fit and the horizon wanted).
    let n_draws_cap = args.flags.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS);
    let one_step: Option<OneStepBands> = match &one_step_fit {
        Some(fit) => Some(one_step_bands(
            out,
            // The fitted model — the fit's intervention selection applied —
            // not `compiled`, which carries every declared intervention so
            // the free-forward sink can read any arm's trajectory.
            //
            // A score's pass continues through the outcome rows past the
            // conditioning ones, so its run reaches the last of them.
            fitted_compiled(&extended_to(&model_for_one_step, binding), &fit_base, eval)?,
            &model,
            &config,
            // gh#708: the step the fit's likelihood walked (the one dt rule).
            // The one-step band re-filters the observed series through the
            // fitted model; under `chain_binomial` the step sets the transition
            // probability `1 - exp(-rate·dt)`, so another dt would be a
            // different process plotted against the data.
            one_step_dt,
            &streams_in,
            binomial,
            fit,
            n_draws_cap,
            particles.n,
            seed,
            schema.as_ref(),
            args.posterior.by_chain,
            &parameter_set,
            binding,
        )?),
        None => None,
    };
    let mut scored = ff_scored;
    if let Some(o) = &one_step {
        let mut s = o.scored.clone();
        for c in &mut s.cells {
            c.scenario = no_overlay.to_string();
        }
        for j in &mut s.joints {
            j.scenario = no_overlay.to_string();
        }
        scored.extend(s);
    }

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
    let one_step_streams: &[StreamBands] =
        one_step.as_ref().map(|o| o.streams.as_slice()).unwrap_or(&[]);
    let one_step_n = one_step.as_ref().map(|o| o.n_pooled).unwrap_or(0);
    let one_step_n_by_chain: &BTreeMap<usize, usize> =
        one_step.as_ref().map(|o| &o.n_pooled_by_chain).unwrap_or(&EMPTY_CHAIN_COUNTS);

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
        let mut names: Vec<String> = args.flags.sweep.iter().map(|s| s.name.clone()).collect();
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
    // a `--exclude-chains 3,5` subset writes `predictive-excl3_5/` +
    // `predictive-excl3_5.json`. A subset is a different posterior — its own
    // warning says so — and writing it at the pooled address REPLACED the run's
    // canonical predictive with a cherry-picked one, silently, with only a
    // `chain_selection` stamp inside the file it had already overwritten.
    let predictive_sub =
        crate::chain_selection::artifact_name("predictive", posterior.selection());

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
                mode: PredictiveMode::FreeForward,
                quantity: Quantity::Observation,
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
                    chain: ChainLabel::One(ChainIndex(*chain)),
                    scenario: cell.scenario.clone(),
                    sweep: cell.sweep.clone(),
                    mode: PredictiveMode::FreeForward,
                    quantity: Quantity::Observation,
                    treatment: treatment_kind,
                    convergence: posterior.convergence,
                    n_draws: ff_draws_per_chain.get(chain).copied().unwrap_or(0),
                    rows: chain_rows,
                });
            }
        }
        if let Some(s) = os_stream {
            sections.push(PredictiveSection {
                chain: ChainLabel::All,
                scenario: no_overlay.to_string(),
                // The one-step horizon is sweep-agnostic (it filters the OBSERVED
                // data through the fitted model, so a swept-parameter overlay is
                // ill-defined) — empty sweep ⇒ empty `sweep:<param>` cells.
                sweep: Vec::new(),
                mode: PredictiveMode::OneStepAhead,
                quantity: Quantity::Observation,
                treatment: treatment_kind,
                convergence: posterior.convergence,
                n_draws: one_step_n,
                rows: &s.rows,
            });
            // Then one section per chain (`--by-chain`; empty otherwise), each
            // reporting the draws it actually pooled — which is smaller than the
            // chain's share of the subsample when some of its draws degenerated.
            for (chain, chain_rows) in &s.per_chain {
                sections.push(PredictiveSection {
                    chain: ChainLabel::One(ChainIndex(*chain)),
                    scenario: no_overlay.to_string(),
                    sweep: Vec::new(),
                    mode: PredictiveMode::OneStepAhead,
                    quantity: Quantity::Observation,
                    treatment: treatment_kind,
                    convergence: posterior.convergence,
                    n_draws: one_step_n_by_chain.get(chain).copied().unwrap_or(0),
                    rows: chain_rows,
                });
            }
        }
        // Record this stream's join contract for `predictive.json`. Coordinate
        // columns (the group-by keys) in header order: the `chain` column when
        // `--by-chain` wrote one, `scenario`, the `sweep:<param>` columns,
        // `time`, the stratum dims, `mode`, `horizon`, `treatment`, `role`,
        // `quantity`. The band columns are
        // the quantile labels; the diagnostics
        // are the method-provenance pair (`fit_rhat_max` / `fit_ess_min`, constant down
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
        coordinates.push("mode".to_string());
        coordinates.push("horizon".to_string());
        coordinates.push("treatment".to_string());
        coordinates.push("role".to_string());
        coordinates.push("quantity".to_string());
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
        written.push(write_tsv(&out_dir, &predictive_sub, source, &pred_tsv)?);
    }
    // `predictive.json`: the per-stream join contract beside the predictive
    // TSVs — a sibling of `quantities.json`, NOT in the run_id-keyed CAS leaf
    // (regenerated, overwritten in place). Written whenever any predictive
    // stream was emitted.
    if !predictive_manifest_entries.is_empty() {
        let mut state_sets = serde_json::Map::new();
        if free_forward.is_some() {
            // A free-forward band runs the fitted model from its start and is
            // never conditioned on a row.
            state_sets.insert(
                PredictiveMode::FreeForward.as_str().to_string(),
                serde_json::json!({ "conditioning": "none", "streams": [] }),
            );
        }
        if let Some(o) = &one_step {
            // A one-step band at a row is drawn from a state filtered through
            // every earlier row of every stream in `streams` (the streams
            // this predict covers): per stream, `rows` walked, of which
            // `held_out_rows` lie after the fit's training prefix. The counts
            // say whether held-out rows were filtered through; no prose
            // restates them.
            state_sets.insert(
                PredictiveMode::OneStepAhead.as_str().to_string(),
                serde_json::json!({ "conditioning": "filtered", "streams": &o.filtered }),
            );
        }
        let mut manifest = serde_json::json!({
            // The tag names the column contract the file was written under;
            // every change to the column set bumps it. The versions are
            // listed in docs/camdl-run-spec.md §2.2.2.
            "schema": "camdl.predictive/v5",
            // gh#896: the run whose draws these bands replay, and the options
            // that shaped them (they also key this directory's address).
            "run_id": run_id,
            "address_key": address_key,
            "options": options,
            "calendar": calendar.to_json(),
            // D78: the binomial sampler every simulation of this predict drew
            // with — the fit's.
            "binomial": binomial,
            // The one-step band's particle count per draw, as `{"n": N}`;
            // absent when no filter ran.
            "particles": one_step_fit.is_some().then_some(particles),
            // The two information sets (`docs/dev/temporal-architecture.md`
            // §2.4). The parameter set is the fit's recorded training
            // partition, as `forecast.json` carries it: the rule, the training
            // extent and each stream's training rows. A row's `role` is its
            // position against that stream's `rows`.
            "parameter_set": &parameter_set.record,
            // The state set of each section written, keyed by its `mode`
            // value: the rows the latent state behind a band was conditioned
            // on.
            "state_sets": state_sets,
            // `[data.holdout]` files are not read by the free-forward or the
            // one-step mode: their rows are neither filtered through nor
            // predicted as data rows, so a free-forward row in a period they
            // cover is `future`. `holdout_after` and `[data.holdout]`
            // describing one split therefore do not yet give one record
            // (gh#1309). `null` without `[data.holdout]`.
            "holdout_files": (!parameter_set.holdout_sources.is_empty()).then(|| serde_json::json!({
                "read": false, "sources": &parameter_set.holdout_sources,
            })),
            "streams": predictive_manifest_entries,
            // D76: which declared streams (leaves) this artifact holds, which it
            // left out and why, and which failed, so a reader need not infer
            // completeness from the exit status.
            "produced": model.observations.iter()
                .filter(|o| streams_in.includes(&o.name) && sources.contains(&o.source))
                .map(|o| o.name.as_str())
                .collect::<Vec<_>>(),
            "skipped": &streams_in.skipped,
            "failed": crate::fit::failures::failed_streams(&failures),
        });
        // Provenance: a chain-subset predictive records the selection alongside
        // the streams, so a chain-subset artifact is never mistakable for a
        // full-cloud one. Absent (no key) when the full cloud was used. The
        // ADDRESS (`predictive_sub`, above) is what keeps the two artifacts from
        // colliding; this stamp is what names the selection once you have one.
        if let Some(info) = posterior.selection() {
            manifest["chain_selection"] = info.to_json();
        }
        stamp_engine(&mut manifest);
        let path = out_dir.join(format!("{predictive_sub}.json"));
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
        written.push(write_tsv(&out_dir, "observed", &source, &obs_tsv)?);
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
        let mut manifest = serde_json::json!({
            "schema": "camdl.observed/v1",
            "run_id": run_id,
            "address_key": address_key,
            "options": options,
            "calendar": calendar.to_json(),
            "streams": observed_manifest_entries,
        });
        stamp_engine(&mut manifest);
        let path = out_dir.join("observed.json");
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
    // `quantities-<key8>-excl3_5/`.
    let quantities_sub = crate::chain_selection::artifact_name(
        &crate::quantities_file::stored_quantities_dir_name(vocabulary.as_ref()),
        posterior.selection(),
    );
    for (name, content) in &quantity_outputs {
        written.push(write_tsv(&out_dir, &quantities_sub, name, content)?);
    }
    if let Some(manifest) = &quantity_manifest {
        let manifest =
            crate::quantities_file::stamp_provenance(manifest, vocabulary.as_ref())?;
        let mut manifest: serde_json::Value = serde_json::from_str(&manifest)
            .map_err(|e| format!("parsing the quantities manifest: {e}"))?;
        stamp_engine(&mut manifest);
        let manifest = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("serializing the quantities manifest: {e}"))?;
        // The manifest keeps the name `quantities_dir_name` gives, a file the
        // store allows at the top level.
        let manifest_name = crate::chain_selection::artifact_name(
            &crate::quantities_file::quantities_dir_name(vocabulary.as_ref()),
            posterior.selection(),
        );
        let path = out_dir.join(format!("{manifest_name}.json"));
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
    if let (false, None) = (model.contrasts.is_empty(), &run_dir) {
        out.say(format_args!(
            "predict: skipping {} contrast(s): a contrast forks each posterior draw from its \
             saved smoothing path, and a fixed-point prediction has none.",
            model.contrasts.len()
        ));
    }
    if let (false, Some(run_dir)) = (model.contrasts.is_empty(), &run_dir) {
        let paths = crate::fit::contrasts::emit_contrasts(
            out,
            &out_dir,
            run_dir,
            None,
            selection.as_ref(),
            &model,
            &fit_base,
            posterior.backend,
            seed,
            quantity_obs_anchors,
            config.config.dt,
            binomial,
            eval,
        )?;
        // The manifest of the contrast tables written, which the store's
        // file set for the prediction is declared from: one entry per
        // contrast that forked, none for one skipped with a note.
        let entries: Vec<serde_json::Value> = paths
            .iter()
            .map(|p| {
                let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let file = p.strip_prefix(&out_dir).unwrap_or(p).to_string_lossy().into_owned();
                serde_json::json!({ "name": name, "file": file })
            })
            .collect();
        let manifest = serde_json::json!({ "schema": "camdl.contrasts/v1", "contrasts": entries });
        let manifest_name = crate::chain_selection::artifact_name("contrasts", posterior.selection());
        let path = out_dir.join(format!("{manifest_name}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?)
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        written.extend(paths);
        written.push(path);
    }
    // `report.json`: what this run computed and what refused, beside the
    // predictive it describes. Keyed by the chain selection for the same reason
    // the predictive is (gh#795) — a subset predictive is a different object,
    // and so is the record of what failed while producing it.
    written.push(crate::fit::failures::write_report(
        &out_dir,
        &crate::chain_selection::artifact_name("report", posterior.selection()),
        &failures,
        &streams_in.skipped,
    )?);

    // gh#1079: a one-step band at a held-out row is not a forecast of it.
    if let Some(o) = &one_step {
        let held: Vec<String> = o.filtered.iter()
            .filter(|f| f.held_out_rows > 0)
            .map(|f| format!("{} of {} rows in '{}'", f.held_out_rows, f.rows, f.stream))
            .collect();
        if !held.is_empty() {
            // The streams the filter walked: under `--stream`, not every
            // stream the fit bound.
            let walked: Vec<String> = o.filtered.iter().map(|f| format!("'{}'", f.stream)).collect();
            out.say(format_args!(
                "predict: the one_step rows include held-out rows ({}): their bands use \
                 parameters fitted without them and a state filtered through every earlier \
                 row of {}, held-out rows included (the `role` column marks them).",
                held.join(", "), walked.join(", ")));
        }
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
    out.say(format_args!(
        "predict: mode={} treatment={}, {} scenario(s) [{}], {} stream(s), \
         {} draws from {} method '{}'",
        horizons.join("+"),
        treatment_kind.label(),
        scenario_labels.len(),
        scenario_labels.join(", "),
        sources.len(),
        posterior.n_draws(),
        method_label,
        posterior.name,
    ));
    Ok(Outcome::Computed(PredictOutcome {
        written, failures, skipped: streams_in.skipped,
        stored: ToStore { source, dir: scratch, options, design: args.flags.design.clone() },
        scored: crate::fit::scoring::Scored {
            ode: posterior.backend == crate::args::types::ForwardBackend::Ode,
            plug_in: treatment_kind == TreatmentKind::PlugIn,
            ..scored
        },
    }))
}

/// One free-forward design cell's rows whose outcome is known, as scored
/// cells: each draw's `y_rep` there and its log density of the outcome at
/// the same state. A data row keeps its role; a row past the data is held
/// out, or of unknown role for a fixed point with no `--holdout-after`.
#[allow(clippy::too_many_arguments)]
fn free_forward_scored(
    model: &ir::Model,
    accum: &ScenarioAccum,
    outcomes: &FreeForwardOutcomes,
    emit_times: &[Vec<f64>],
    coverages: &[Vec<(f64, score::trace::Coverage)>],
    leaf_times: &[Vec<f64>],
    parameter_set: &ParameterSet,
    scenario: &str,
    sweep: &[(String, f64)],
) -> Vec<crate::fit::scoring::ScoredCell> {
    use crate::fit::scoring::{ScoreMode, ScoredCell};
    let mut cells = Vec::new();
    for (si, o) in model.observations.iter().enumerate() {
        for (ti, cell) in outcomes.observed[si].iter().enumerate() {
            let Some((y, _)) = cell else { continue };
            let (start, stop) = match coverages[si][ti].1 {
                score::trace::Coverage::Interval { start, stop } => (start, stop),
                score::trace::Coverage::Instant | score::trace::Coverage::Unrecorded => (emit_times[si][ti], emit_times[si][ti]),
            };
            let role = if ti < leaf_times[si].len() {
                parameter_set.role(&o.name, ti)
            } else if parameter_set.training_unknown {
                RowRole::Unknown
            } else {
                RowRole::HeldOut
            };
            cells.push(ScoredCell {
                scenario: scenario.to_string(),
                sweep: sweep.to_vec(),
                leaf: o.name.clone(),
                source: o.source.clone(),
                stratum: o.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect(),
                start,
                stop,
                mode: ScoreMode::FreeForward,
                origin: None,
                horizon: None,
                lead: None,
                role,
                observed: *y,
                samples: accum.samples[si][ti].clone(),
                log_liks: accum.log_liks[si][ti].clone(),
            });
        }
    }
    cells
}

/// Compile `vocabulary` against the fit's model SOURCE and return the quantity
/// IR to transplant onto `archived` — refusing unless the source is still the
/// same model the fit ran on.
///
/// The equality test is [`crate::fit::fitted_model::compare_to_fitted`] —
/// `ir::model_hash::model_ir_hash`, which excludes `quantities` (and
/// `contrasts`), the comparison every replay of a fit through a given model
/// shares. That is what makes it the right test rather
/// than merely a convenient one: it asks "is this the same model apart from its
/// reporting layer?", which is exactly the question a reporting-layer swap has
/// to answer yes to. It is also gradient-independent (model hash SV=2), so the lean
/// recompile here and a full-Jacobian fit compile still agree.
fn quantities_from_vocabulary(
    archived: &ir::Model,
    config: &crate::fit::config_v2::Problem,
    vocabulary: &crate::quantities_file::QuantitiesOverride,
    compile: crate::util::CompileSettings,
    integrator: Option<crate::args::types::IntegratorArg>,
    treatment: TreatmentKind,
) -> Result<Vec<ir::quantity::Quantity>, String> {
    let model_source: &str = &config.model.camdl;
    // What the messages call the model the prediction runs: a fit's, or the
    // model file a fixed point was given with.
    let whose = match treatment {
        TreatmentKind::Posterior => "this fit's model",
        TreatmentKind::PlugIn => "the fixed point's model",
    };
    // Messages name the model file in the stored problem, not its scratch copy.
    let shown = crate::fit_store::shown(std::path::Path::new(model_source));
    if !crate::util::model_is_camdl_source(model_source) {
        return Err(format!(
            "--quantities needs {whose} SOURCE, but the prediction runs `{}`, \
             which is compiled IR. A reporting vocabulary is resolved against the model's \
             symbols at compile time, and a `let` that mentions a parameter is inlined away \
             in the IR, so it cannot be applied to a compiled model.",
            shown.display()
        ));
    }
    if !std::path::Path::new(model_source).is_file() {
        return Err(format!(
            "--quantities needs {whose} source `{}`, which is not readable from here. A \
             reporting vocabulary has to be compiled against the model's symbols.",
            shown.display()
        ));
    }
    // Lean compile: `camdl predict` replays forward trajectories and never
    // recomputes an ODE gradient (gh#439 A2). Identity is gradient-independent,
    // so this does not disturb the hash comparison below.
    let (ir_path, _tmp) = crate::util::resolve_ir_path(model_source, false, Some(vocabulary), compile)?;
    let (mut recompiled, _) = crate::util::load_ir(&ir_path)?;
    // A fixed point's compile carries `--integrator`, written in before it
    // was keyed; the source compiled here is given the same.
    crate::util::apply_integrator_override(&mut recompiled, integrator);
    use crate::fit::fitted_model::{as_fitted, compare_to_fitted, short, without_hoisting, ModelAgreement};
    // The archived IR is the model AFTER `fit run` resolved its observation
    // anchors; the recompiled source still carries them. Both go through the
    // fit's own resolution before the comparison — a no-op on the archived IR,
    // and on the pre-archive fallback (`archived` recompiled from source) it
    // resolves both sides alike. Both are compared with their hoisted
    // subexpressions written back in place, so the fit's compile and this
    // one agree whether loop-invariant code motion ran in either.
    let resolve = |m: &ir::Model| {
        as_fitted(m, config).map(|(m, _)| m).and_then(|m| without_hoisting(&m)).map_err(|e| {
            format!(
                "--quantities: resolving the observation anchors of {whose} against its \
                 data, to compare the source with the model the prediction runs, failed: {e}"
            )
        })
    };
    let recompiled = resolve(&recompiled)?;
    if let ModelAgreement::Differs { fitted: archived_h, given: recompiled_h } =
        compare_to_fitted(&resolve(archived)?, &recompiled)
    {
        return Err(match treatment {
            TreatmentKind::Posterior => format!(
                "this fit's model source `{}` is no longer the model the fit ran on, \
                 so a reporting vocabulary cannot be checked against it (archived {}, source {}). \
                 Quantities are excluded from that hash, so this difference is a real change to \
                 the dynamics, observations, or parameters — not the vocabulary. Re-run the fit \
                 against the current source, or check out the source the fit used.",
                shown.display(),
                short(&archived_h),
                short(&recompiled_h),
            ),
            TreatmentKind::PlugIn => format!(
                "the model file `{}` compiled with the vocabulary is not the model the fixed point \
                 runs (model {}, with the vocabulary {}). Quantities are excluded from that hash, so \
                 the file changed while the prediction ran; run it again.",
                shown.display(),
                short(&archived_h),
                short(&recompiled_h),
            ),
        });
    }
    // The quantities transplanted refer to no hoisted binding, so they read
    // the same in the archived model whatever it hoisted.
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

/// Load the observed series of each leaf `selection` makes eligible.
fn load_leaf_obs(
    model: &ir::Model,
    compiled: &sim::CompiledModel,
    config: &crate::fit::config_v2::Problem,
    dt: f64,
    selection: &StreamSelection,
) -> Result<Vec<LeafObs>, String> {
    let data = config
        .data_spec()
        .map_err(|e| format!("this fit has no [data] block to read the observed series from: {e}"))?;
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)
        .map_err(|e| format!("resolving [data]: {e}"))?;
    let effective = bindings.training_by_source();
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };

    let mut out = Vec::new();
    for obs_model in &model.observations {
        if !selection.includes(&obs_model.name) {
            continue; // not eligible: skipped or failed, and named by the caller (D76)
        }
        let data_path = bound_path(effective, obs_model)?;
        let siblings: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().filter(|o| o.source == obs_model.source).collect();
        let rows =
            crate::fit::runner::load_observations(data_path, obs_model, &siblings, &time_opts)?;
        // Resolved only to refuse a projection the compiled model cannot
        // resolve; the rows already carry what they measure.
        sim::inference::multi_stream_obs::StreamProjection::from_ir(
            &obs_model.projection, compiled, &obs_model.name,
        )?;
        let stream_times =
            crate::pfilter::stream_times_of_rows(obs_model, &rows, data_path, &time_opts)?;
        let (obs, cells, aux) = crate::fit::runner::observation_columns(&rows);
        let times: Vec<f64> = obs.iter().map(|o| o.time).collect();
        let observed: Vec<Option<f64>> = cells
            .iter()
            .map(|c| c.as_ref().map(|cell| match cell {
                // The observed series as the file wrote it; a row read from a file
                // holds no held-out cell, and its value is what it is either way.
                sim::inference::ObsCell::Scalar(v) | sim::inference::ObsCell::HeldOut(v) => *v,
            }))
            .collect();
        let stratum: Vec<(String, String)> =
            obs_model.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect();
        out.push(LeafObs {
            source: obs_model.source.clone(), stratum, times, stream_times, observed, aux,
        });
    }
    Ok(out)
}

/// What each emitted row of a leaf covers, aligned 1:1 with the leaf's emitted
/// times (gh#833). The observed rows are read off the stream's bound schedule —
/// the periods the fit's likelihood scored, so the predictive's first bin opens
/// exactly where the likelihood's did (gh#702) rather than at the model origin.
/// The forecast tail continues the same declaration: a uniform form assigns each
/// extrapolated label its period; per-row window columns, having no rule to
/// extrapolate, continue as contiguous windows closing at each label. A leaf
/// with no bound data has no rows.
fn leaf_row_coverages(
    obs_ir: &ir::observation::ObservationModel,
    observed: Option<&sim::inference::StreamTimes>,
    observed_times: &[f64],
    forecast_times: &[f64],
    t_start: f64,
) -> Vec<(f64, score::trace::Coverage)> {
    use ir::observation::TemporalKind;
    use score::trace::Coverage;
    let mut rows: Vec<(f64, Coverage)> = match observed {
        Some(st) => observed_times.iter().enumerate()
            .map(|(k, &t)| (t, st.coverage(k)))
            .collect(),
        None => Vec::new(),
    };
    let mut prev_stop = rows.last().map(|(t, _)| *t).unwrap_or(t_start);
    for &t in forecast_times {
        let coverage = match obs_ir.projection.temporal_kind() {
            TemporalKind::Instant => Coverage::Instant,
            // A uniform form assigns the label its period. Per-row windows
            // have no rule to extrapolate, so the forecast continues
            // contiguously. (An interval stream with no declaration never
            // binds — the loader refuses it — so `covers: None` is not
            // reached here; it takes the contiguous reading, which is the only
            // one with nothing to consult.)
            TemporalKind::Interval => match obs_ir.covers.as_ref().and_then(|c| c.period_of(t)) {
                Some((start, stop)) => Coverage::Interval { start, stop },
                None => Coverage::Interval { start: prev_stop, stop: t },
            },
        };
        prev_stop = t;
        rows.push((t, coverage));
    }
    rows
}

/// The data file bound to an eligible leaf's source. An eligible leaf is bound
/// by construction ([`select_streams`]), so a miss is an internal error.
fn bound_path<'a>(
    bound: &'a IndexMap<String, String>,
    o: &ir::observation::ObservationModel,
) -> Result<&'a String, String> {
    bound.get(&o.source).ok_or_else(|| format!(
        "internal error: stream '{}' was selected as eligible but its source '{}' is bound \
         to no data", o.name, o.source))
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
/// present here and are what the rest of `camdl predict` is keyed to, so they are
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
    // NaN arm explicit: `modal_value` returns NaN when it finds no positive
    // gap, and NaN must fall through to "no cadence", not to "cadence is fine"
    // — which `cadence <= 0.0` alone would do, being false for NaN.
    let cadence = crate::util::modal_value(&gaps);
    if cadence.is_nan() || cadence <= 0.0 {
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
    parameter_set: &ParameterSet,
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
            // A leaf's emitted times are its data rows, in file order, then
            // the forecast tail (`run_predict`'s `ff_emit_times`): a row's
            // index among them is its position in the stream until the data
            // runs out.
            let leaf_name = &model.observations[si].name;
            let n_data_rows = leaves
                .iter()
                .find(|l| leaf_matches(&model.observations[si], l))
                .map_or(0, |l| l.times.len());
            for (ti, draws_at_t) in accum.samples[si].iter().enumerate() {
                let t = emit_times[si][ti];
                let role = if ti < n_data_rows {
                    parameter_set.role(leaf_name, ti)
                } else {
                    parameter_set.role_past_data(leaf_name, t)
                };
                rows.push(
                    band_row(
                        t,
                        role,
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
                        role,
                        stratum: stratum.clone(),
                        quantiles: band(&chain_draws).map_err(|e| {
                            format!("stream '{source}' at t={t}, chain {}: {e}", ChainIndex(chain))
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
    role: RowRole,
    stratum: Vec<(String, String)>,
    draws: CellDraws<'_>,
    chains: &crate::fit::row_convergence::ChainOfDraw,
) -> Result<BandRow, String> {
    use crate::fit::row_convergence::row_convergence;
    let quantiles = band(draws.predictive)?;
    Ok(BandRow {
        time,
        role,
        stratum,
        quantiles,
        mean_conv: row_convergence(draws.mean, chains),
        pred_conv: row_convergence(draws.predictive, chains),
    })
}

// ── Posterior-cloud subsampling (shared by both horizons) ───────────────────

/// Default posterior-cloud subsample cap for `camdl predict`, shared by both
/// horizons. Neither needs the full fit-grade cloud: the one-step band pools
/// `draws × n_particles` per cell and the free-forward band pools one forward
/// replay per draw, so a few hundred draws saturate q05…q95. The full cloud is
/// never replayed silently — a full free-forward replay of a long-burn-in ODE
/// fit is thousands of ~seconds-each solves (gh#387).
pub(crate) const DEFAULT_PREDICT_DRAWS: usize = 200;

/// Every input that changes what `camdl predict` writes, beyond the run it
/// reads, resolved to the values the predict used. Together with the run and
/// the engine they key the output directory ([`Self::address_key`]), and they
/// are recorded in its manifests, so two predictions computed from different
/// inputs never share a directory and a reader can tell which one it holds.
///
/// The `PredictArgs` flags, and where each one goes:
///
/// - the run handle (`RUN`): the run it resolves to, keyed by its
///   full run id in [`Self::address_key`] (the spelling of the handle is not).
/// - `--mode`, `--horizon` (as `reach`), `--n-draws`, `--seed`, `--stream`,
///   `--scenario`, `--enable`, `--disable`, `--sweep`, `--by-chain`: fields.
/// - `--exclude-chains` / `--chains`: `excluded_chains`, the chains dropped
///   from the fit's cloud, so the two spellings of one selection are one key.
/// - `--quantities FILE`: `quantities`, the SHA-256 of the file's bytes.
/// - `--allow-forcing-extrapolation`: `allow_forcing_extrapolation`, as given;
///   the fit's own `[config]` setting is part of the run.
///
/// The fit's other settings (its config, its sampler, its
/// `allow_degenerate_rates`) are read from the run, so its run id covers them.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct PredictOptions {
    pub mode: &'static str,
    /// The forecast's `--horizon` as written; absent outside forecast mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reach: Option<String>,
    pub n_draws: usize,
    pub seed: u64,
    /// `--stream`, as a set: `null` when absent, the bare name when there is
    /// one, a list when there are several.
    #[serde(serialize_with = "serialize_streams")]
    pub stream: Vec<String>,
    pub scenarios: Vec<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
    pub sweep: Vec<(String, Vec<f64>)>,
    pub by_chain: bool,
    /// The chains `--exclude-chains` / `--chains` dropped, sorted; absent when
    /// the whole cloud was used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub excluded_chains: Option<Vec<usize>>,
    /// The SHA-256 of the `--quantities` file; absent for the model's own
    /// `quantities {}` block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantities: Option<String>,
    /// `--allow-forcing-extrapolation` as given on the command line.
    pub allow_forcing_extrapolation: bool,
    /// The filters' particle count (D85), present whenever a filter runs
    /// (the one-step band or the forecast). Where it came from is recorded
    /// in the manifest, not here, so `--particles` equal to the default
    /// addresses the same output as no flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub particles: Option<usize>,
    /// What a forecast's plan contributes to its predictions' inputs: the
    /// target set (with each target's design values) and the design mode.
    /// Absent outside forecast mode. Held-out outcomes never enter it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forecast: Option<ForecastKey>,
}

/// The plan's half of a forecast's address (D80, D81).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub(crate) struct ForecastKey {
    /// [`crate::fit::forecast_plan::ForecastPlan::targets_digest`]: every
    /// target's stream, period or instant, and design values.
    pub targets: String,
    /// Where the design values came from, when any target reads one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub design_mode: Option<crate::fit::forecast_plan::DesignMode>,
    /// What each data-table forcing the forecast reads after its origin is
    /// stated to be there (`--covariate`), by forcing. Empty, and absent from
    /// the encoding, when the forecast reads no table forcing after its
    /// origin.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub covariates: BTreeMap<String, crate::args::types::CovariateStatement>,
}

/// What [`PredictOptions::address_key`] hashes: the run, the engine that
/// computes the prediction, and the options.
#[derive(serde::Serialize)]
struct PredictAddress<'a> {
    run_id: &'a str,
    engine: &'a str,
    options: &'a PredictOptions,
}

/// Hex digits of the address key: 128 bits.
pub(crate) const PREDICT_KEY_HEX: usize = 32;

impl PredictOptions {
    /// What `camdl predict <run>` with no options uses.
    pub(crate) fn defaults() -> Self {
        PredictOptions {
            mode: "all",
            reach: None,
            n_draws: DEFAULT_PREDICT_DRAWS,
            seed: 1,
            stream: Vec::new(),
            scenarios: Vec::new(),
            enable: Vec::new(),
            disable: Vec::new(),
            sweep: Vec::new(),
            by_chain: false,
            excluded_chains: None,
            quantities: None,
            allow_forcing_extrapolation: false,
            particles: None,
            forecast: None,
        }
    }

    /// The free-forward horizon alone over the cloud `excluded` leaves, every
    /// other option at its default — what `fit table --quantity` derives with.
    pub(crate) fn free_forward(excluded: Option<&crate::chain_selection::SubsetInfo>) -> Self {
        PredictOptions {
            mode: "free_forward",
            excluded_chains: excluded_chain_ids(excluded),
            ..Self::defaults()
        }
    }

    /// The options of `args`, with the chain selection as resolved against the
    /// fit (`excluded`) and the `--quantities` vocabulary as loaded.
    pub(crate) fn of(
        args: &crate::args::PredictArgs,
        excluded: Option<&crate::chain_selection::SubsetInfo>,
        vocabulary: Option<&crate::quantities_file::QuantitiesOverride>,
    ) -> Self {
        use crate::args::types::PredictModeArg;
        PredictOptions {
            // An invalid combination is refused by [`predict_mode`] before
            // any output is addressed; this reads the same flags.
            mode: match (args.flags.mode, args.flags.horizon.is_some()) {
                (None, false) => "all",
                (Some(PredictModeArg::FreeForward), _) => "free_forward",
                (Some(PredictModeArg::OneStep), _) => "one_step",
                (Some(PredictModeArg::Forecast), _) | (None, true) => "forecast",
            },
            reach: args.flags.horizon.clone(),
            n_draws: args.flags.n_draws.unwrap_or(DEFAULT_PREDICT_DRAWS),
            seed: args.flags.seed.unwrap_or(1),
            stream: args.flags.stream.clone(),
            scenarios: args.flags.scenarios.clone(),
            enable: args.flags.enable.clone(),
            disable: args.flags.disable.clone(),
            sweep: args.flags.sweep.iter().map(|s| (s.name.clone(), s.grid.expand())).collect(),
            by_chain: args.posterior.by_chain,
            excluded_chains: excluded_chain_ids(excluded),
            quantities: vocabulary.map(|v| v.digest.clone()),
            allow_forcing_extrapolation: args.flags.forcing_extrapolation.allow_forcing_extrapolation,
            particles: None,
            forecast: None,
        }
        .normalised()
    }

    /// The options in their canonical form: the stream, scenario, enable and
    /// disable lists as sets (sorted), so the order they were spelled in does
    /// not key a different address.
    pub(crate) fn normalised(mut self) -> Self {
        self.stream.sort();
        self.stream.dedup();
        self.scenarios.sort();
        self.enable.sort();
        self.disable.sort();
        self
    }

    /// The address key of a prediction of `run_id` with these options by this
    /// engine: the first [`PREDICT_KEY_HEX`] hex digits (128 bits) of the
    /// SHA-256 of [`PredictAddress`]'s JSON. The engine is the key of the
    /// ambient identity every result on the store is grouped by
    /// ([`crate::camdl_store::ambient`]): the build fingerprint
    /// [`crate::version::ENGINE_ID`] and the environment switches that can
    /// change a result. So a build with uncommitted changes never reads or
    /// overwrites another build's prediction, and neither does a prediction
    /// made under a switch (`CAMDL_EVAL_UNRESOLVED`, …) one made without it.
    pub(crate) fn address_key(&self, run_id: &str) -> String {
        self.address_key_by(run_id, &prediction_engine_key())
    }

    fn address_key_by(&self, run_id: &str, engine: &str) -> String {
        let json = serde_json::to_string(&PredictAddress { run_id, engine, options: self })
            .expect("PredictOptions serializes");
        crate::hashing::sha256_hex(json.as_bytes())[..PREDICT_KEY_HEX].to_string()
    }
}

/// The key of the engine a prediction is made by: this build and the
/// environment switches that can change a result, as the store keys them
/// ([`crate::camdl_store::ambient`]).
fn prediction_engine_key() -> String {
    match crate::camdl_store::ambient() {
        store::Build::Known(e) | store::Build::Incomplete(e) => store::Identity::key(&e).to_hex(),
    }
}

/// Record on a prediction's manifest `doc` the engine that made it, spelled as
/// `run.json` spells it: `engine_version`, the build fingerprint, and
/// `engine_switches`, the switches that were on (absent when none was). The
/// address key folds the same identity, so two predictions that differ only in
/// the engine are told apart inside their files as well as by their
/// directories.
pub(crate) fn stamp_engine(doc: &mut serde_json::Value) {
    doc["engine_version"] = serde_json::Value::String(crate::version::ENGINE_ID.to_string());
    let switches = crate::env::get().engine_switches();
    if !switches.is_empty() {
        doc["engine_switches"] = serde_json::to_value(&switches).expect("an engine switch serializes");
    }
}

/// The chain ids a resolved selection dropped, in order.
fn excluded_chain_ids(excluded: Option<&crate::chain_selection::SubsetInfo>) -> Option<Vec<usize>> {
    excluded.map(|i| i.excluded.iter().map(|c| c.0).collect())
}

/// [`PredictOptions::stream`]: `null`, a bare name, or a list of several.
fn serialize_streams<S: serde::Serializer>(streams: &[String], s: S) -> Result<S::Ok, S::Error> {
    use serde::Serialize;
    match streams {
        [] => s.serialize_none(),
        [one] => one.serialize(s),
        many => many.serialize(s),
    }
}

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

/// The `n_draws`-by-chain map when no one-step horizon ran — a borrowable
/// empty, so the render path has one type rather than an `Option` per lookup.
static EMPTY_CHAIN_COUNTS: BTreeMap<usize, usize> = BTreeMap::new();

// ── The one-step-ahead posterior predictive producer ───────────────────────

/// The particle count of `camdl predict`'s filters when neither `--particles`
/// nor the fit gives one (D85): the fit is not PMMH (PGAS's conditional SMC
/// count is not a bootstrap-filter count).
pub(crate) const DEFAULT_PREDICT_PARTICLES: usize = 500;

/// The particle count of a PMMH fit's own filter, read from the leaf's
/// `run.json` (`inputs.algorithm`, written by `fit run`), as the forecast
/// reads the fit's binomial sampler there; `None` for any other method (D85).
pub(crate) fn fit_pmmh_particles(leaf: &Path) -> Result<Option<usize>, String> {
    let path = leaf.join(crate::fit_store::RUN_INPUTS_FILE);
    let inputs = crate::fit_store::read_run_inputs(leaf)?;
    let algorithm = inputs.get("algorithm");
    if algorithm.and_then(|a| a.get("algorithm")).and_then(|a| a.as_str()) != Some("pmmh") {
        return Ok(None);
    }
    algorithm.and_then(|a| a.get("particles")).and_then(|n| n.as_u64())
        .map(|n| Some(n as usize))
        .ok_or_else(|| format!(
            "{} records a PMMH fit but not its particle count; re-run the fit, or pass \
             --particles", path.display()))
}

/// The particle count every filter `camdl predict` runs per posterior draw
/// uses (the one-step band's and the forecast's), as resolved. Where the
/// count came from is not recorded: `--particles` equal to the default is the
/// same computation, under one identity, so a record of the flag would make
/// the stored bytes depend on which invocation computed them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PredictParticles {
    pub n: usize,
}

impl PredictParticles {
    /// `--particles` when given, else the PMMH fit's count, else
    /// [`DEFAULT_PREDICT_PARTICLES`].
    pub(crate) fn resolve(flag: Option<usize>, fit_pmmh: Option<usize>) -> Result<Self, String> {
        match (flag, fit_pmmh) {
            (Some(0), _) => Err("--particles must be at least 1".into()),
            (Some(n), _) | (None, Some(n)) => Ok(PredictParticles { n }),
            (None, None) => Ok(PredictParticles { n: DEFAULT_PREDICT_PARTICLES }),
        }
    }
}

/// One chain's share of the one-step pool, plus the draw accounting a per-chain
/// band has to report honestly (gh#794).
#[cfg_attr(test, derive(Debug))]
struct ChainPool {
    /// `pooled[stream_idx][obs_idx]` = ỹ over (particles × this chain's draws).
    pooled: Vec<Vec<Vec<f64>>>,
    /// Draws of this chain that actually contributed a filter pass.
    n_pooled: usize,
    /// Draws of this chain the subsample offered. Greater than `n_pooled` when
    /// some of them degenerated; the gap is what stops a thinned band from
    /// reading as a full one.
    n_offered: usize,
}

/// The one-step predictive: per-(stream-leaf, obs-time) particle samples,
/// partitioned by the chain each draw came from, plus how many draws
/// contributed.
///
/// The partition is always built, not only under `--by-chain`, so there is one
/// accumulation path rather than two that could drift. The pooled band is
/// [`PooledOneStep::cell`], the concatenation of every chain's samples at that
/// cell — the same multiset a single flat pool would hold. `quantile` sorts a
/// copy of its input ([`crate::quantile::quantile`]), so the concatenation
/// order cannot change the band; the pooled rows are byte-identical to the
/// pre-partition ones either way.
#[cfg_attr(test, derive(Debug))]
struct PooledOneStep {
    /// Leaf stream names, from the first successful filter result (identical
    /// across draws — same obs model).
    stream_names: Vec<String>,
    /// Union observation-time axis, same provenance.
    obs_times: Vec<f64>,
    /// Chain id → its pool. `None` is the whole cloud, used when `draws.tsv`
    /// carries no chain column and no chain can be named; a keyed cloud never
    /// mixes the two.
    by_chain: BTreeMap<Option<usize>, ChainPool>,
    /// Draws that contributed (= param_sets.len() − degenerate-skipped).
    n_pooled: usize,
}

impl PooledOneStep {
    /// Every chain's ỹ at one `(stream, obs)` cell, concatenated — the operand
    /// of the pooled band. NaN entries were already dropped at accumulation.
    fn cell(&self, si: usize, ti: usize) -> Vec<f64> {
        let n: usize = self.by_chain.values().map(|c| c.pooled[si][ti].len()).sum();
        let mut out = Vec::with_capacity(n);
        for c in self.by_chain.values() {
            out.extend_from_slice(&c.pooled[si][ti]);
        }
        out
    }

    /// The chains that can be banded on their own: keyed, and with at least one
    /// draw that survived its filter. An unkeyed cloud yields none, and neither
    /// does a chain every one of whose draws degenerated — banding an empty
    /// cell would publish `NaN` quantiles.
    fn bandable_chains(&self) -> Vec<usize> {
        self.by_chain
            .iter()
            .filter_map(|(k, c)| k.filter(|_| c.n_pooled > 0))
            .collect()
    }
}

/// What the one-step producer yields: the banded streams plus the draw
/// accounting the artifact reports.
struct OneStepBands {
    /// One entry per logical stream, each carrying the pooled rows and (under
    /// `--by-chain`) that stream's per-chain rows.
    streams: Vec<StreamBands>,
    /// Draws behind the pooled `all` rows.
    n_pooled: usize,
    /// Chain id -> draws behind that chain's own rows. Empty without
    /// `--by-chain`. A chain that lost some of its draws to a degenerate filter
    /// appears here with what it actually pooled, so its `n_draws` cell never
    /// claims a full band; a chain that lost all of them is absent.
    n_pooled_by_chain: BTreeMap<usize, usize>,
    /// The state set: per leaf stream, sorted by name, the rows every band's
    /// filter walked and how many of them the fit held out.
    filtered: Vec<FilteredStream>,
    /// With outcomes bound, every row the pass predicted whose outcome is
    /// known, the rows after the conditioning ones included; empty otherwise.
    scored: crate::fit::scoring::Scored,
}

/// One draw's one-step record that a score reads, `[stop][stream][particle]`:
/// the predictive draws, and the per-stream and joint log densities of the
/// observed values at the particles they were drawn from.
struct DrawRecord {
    samples: Vec<Vec<Vec<f64>>>,
    log_liks: Vec<Vec<Vec<f64>>>,
    joint: Vec<Vec<f64>>,
    /// The stop at which the draw's filter degenerated, after the
    /// conditioning rows; `None` when it ran to the end. Its record at that
    /// stop and after is not read.
    degenerated: Option<usize>,
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
    out: &crate::out::Out,
    param_sets: &[Vec<f64>],
    chain_of_draw: &[Option<usize>],
    base_seed: u64,
    mut run_filter: impl FnMut(
        usize,
        &[f64],
        u64,
    ) -> Result<sim::inference::particle_filter::PFilterResult, sim::SimError>,
) -> Result<PooledOneStep, String> {
    let mut by_chain: BTreeMap<Option<usize>, ChainPool> = BTreeMap::new();
    let mut stream_names: Vec<String> = Vec::new();
    let mut obs_times: Vec<f64> = Vec::new();
    let mut skipped: Vec<(usize, String)> = Vec::new();
    // Draws each chain was *offered*, counted before any filter runs, so the gap
    // to what it contributed is available even for a chain that lost every one.
    let mut offered: BTreeMap<Option<usize>, usize> = BTreeMap::new();
    for i in 0..param_sets.len() {
        *offered.entry(chain_of_draw.get(i).copied().flatten()).or_insert(0) += 1;
    }

    for (draw_idx, params) in param_sets.iter().enumerate() {
        // Distinct, reproducible per-draw seed: mix the draw index into the
        // base seed so each filter pass has its own RNG stream and the whole
        // run is deterministic given `base_seed`. Keyed on the index, so a
        // skipped draw does not shift the seeds of the draws after it.
        let seed = crate::fit::forecast_driver::draw_seed(base_seed, draw_idx);
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

        if stream_names.is_empty() {
            stream_names = preq.stream_names.clone();
            obs_times = preq.obs_times.clone();
        }

        // This draw's chain gets its own accumulator, created on first use and
        // sized to the axes the first successful filter established.
        let chain = chain_of_draw.get(draw_idx).copied().flatten();
        let n_offered = offered.get(&chain).copied().unwrap_or(0);
        let pool = by_chain.entry(chain).or_insert_with(|| ChainPool {
            pooled: vec![vec![Vec::new(); obs_times.len()]; stream_names.len()],
            n_pooled: 0,
            n_offered,
        });
        pool.n_pooled += 1;

        // `per_stream_samples[obs_idx][stream_idx][particle]`.
        for (obs_idx, per_stream) in preq.per_stream_samples.iter().enumerate() {
            for (stream_idx, particles) in per_stream.iter().enumerate() {
                for &y in particles {
                    if y.is_finite() {
                        pool.pooled[stream_idx][obs_idx].push(y);
                    }
                }
            }
        }
    }

    let n_total = param_sets.len();
    if skipped.len() == n_total && n_total > 0 {
        let (first_idx, first_err) = &skipped[0];
        return Err(format!(
            "one_step mode: the one-step filter degenerated on ALL {n_total} posterior \
             draws — there is nothing to pool. First failure (draw {first_idx}): \
             {first_err}"
        ));
    }
    if !skipped.is_empty() {
        let (first_idx, first_err) = &skipped[0];
        out.say(format_args!(
            "predict: one_step mode — skipped {}/{} posterior draws whose one-step \
             filter degenerated (a small tail of degenerate draws is expected in a \
             converging posterior; first: draw {first_idx}: {first_err}); bands pool the \
             remaining {} draws",
            skipped.len(),
            n_total,
            n_total - skipped.len(),
        ));
    }
    // A chain that lost *some* of its draws still bands, but its `n_draws` cell
    // reports what it actually pooled -- and the loss is named here, because a
    // reader comparing chains has to know that a narrower band may be a thinner
    // sample rather than a more certain chain. A chain that lost *every* draw is
    // absent from `by_chain` entirely (nothing was inserted for it), so it
    // yields no band; say so rather than let it vanish silently.
    for (chain, offered_here) in &offered {
        let Some(id) = chain else { continue };
        // Read the denominator off the pool where one exists, so the number in
        // the message is the number the pool actually carries; a chain that
        // contributed nothing has no pool, and falls back to what it was given.
        let (n_pooled, n_offered) = match by_chain.get(chain) {
            Some(c) => (c.n_pooled, c.n_offered),
            None => (0, *offered_here),
        };
        if n_pooled == n_offered {
            continue;
        }
        if n_pooled == 0 {
            out.say(format_args!(
                "predict: one_step mode -- chain {} contributed no band: all \
                 {n_offered} of its replayed draws degenerated. It is omitted from the \
                 per-chain rows rather than banded over nothing.",
                ChainIndex(*id)
            ));
        } else {
            out.say(format_args!(
                "predict: one_step mode -- chain {} bands over {n_pooled} of the \
                 {n_offered} draws it was given ({} degenerated); its rows report \
                 n_draws = {n_pooled}, so a narrower band there may be a thinner \
                 sample rather than a more certain chain.",
                ChainIndex(*id),
                n_offered - n_pooled,
            ));
        }
    }

    Ok(PooledOneStep {
        stream_names,
        obs_times,
        by_chain,
        n_pooled: n_total - skipped.len(),
    })
}

/// Build the one-step-ahead posterior predictive bands: for each (subsampled)
/// posterior draw θ, run a bootstrap filter over the data with
/// `record_prequential = true`, capturing the per-particle one-step predictive
/// samples `ỹ ∼ p(y | x_t, θ)` at each observation time (the particles are
/// distributed as `p(x_t | y_{1:t-1}, θ)` at that point). Pool over
/// (particles × draws) per `(stream-leaf, time)`, quantile, and group by logical
/// source + stratum exactly like the free-forward path. Mode = one_step.
///
/// [`OneStepBands::n_pooled`] is the number of draws actually pooled — the
/// subsample minus any degenerate-skipped draws (gh#620) — for the `n_draws`
/// artifact column.
///
/// `by_chain` additionally bands each chain's own draws into
/// [`StreamBands::per_chain`]. That is a *grouping* of the same reduction, not a
/// decomposition of the particle contribution: each chain's band pools its own
/// draws x particles exactly as the pooled band pools all of them, so the
/// particle pooling affects a per-chain band's width the same way it already
/// affects the pooled one. (What is deferred to gh#798 is R-hat/ESS on one-step
/// rows, which would need the particle contribution separated; the per-chain
/// rows carry no convergence cells for the same reason the free-forward ones do
/// not -- a between-chain statistic has nothing to compare one chain against.)
#[allow(clippy::too_many_arguments)]
/// `camdl predict`'s step grids, `(free_forward, one_step)`: both the step
/// the fit ran at (`dt_rule::resolved_dt`), so neither horizon replays a
/// different process than the fit. Kept as a pair so the two call sites
/// still name which grid they walk.
fn predict_step_grids(model_dt: Option<f64>, config_dt: f64) -> (f64, f64) {
    let fitted = crate::dt_rule::resolved_dt(model_dt, config_dt);
    (fitted, fitted)
}

/// `model`, its run extent raised to the end of the last outcome row
/// `binding` holds, so a score's one-step pass admits every row it predicts.
/// Nothing else is extended; without a binding, or when the model already
/// reaches the rows, `model` as it is.
fn extended_to(model: &ir::Model, binding: Option<&crate::fit::scoring::OutcomeBinding>) -> ir::Model {
    let mut m = model.clone();
    let end = binding
        .map(|b| b.rows.values().flatten().map(|r| r.measures.end()).fold(f64::NEG_INFINITY, f64::max))
        .unwrap_or(f64::NEG_INFINITY);
    if end > m.simulation.t_end {
        m.simulation.t_end = end;
    }
    m
}

/// The fitted model compiled for the one-step filter: `model` (parameters
/// already seeded) with the fit's intervention selection applied. The filter
/// re-scores the observed series, so it must run the interventions the fit's
/// likelihood ran — no more (every declared toggleable intervention, which an
/// unfiltered compile would keep) and no fewer.
fn fitted_compiled(
    model: &ir::Model,
    fit_base: &crate::util::FitBase,
    eval: sim::EvalSettings,
) -> Result<std::sync::Arc<sim::CompiledModel>, String> {
    let mut fitted = model.clone();
    fit_base.filter(&mut fitted)?;
    Ok(std::sync::Arc::new(
        sim::CompiledModel::new(fitted)
            .map_err(|e| format!("compiling the fitted model for the one-step filter: {e:?}"))?
            .with_eval_settings(eval),
    ))
}

fn one_step_bands(
    out: &crate::out::Out,
    compiled: std::sync::Arc<sim::CompiledModel>,
    model: &ir::Model,
    config: &crate::fit::config_v2::Problem,
    dt: f64,
    selection: &StreamSelection,
    binomial: sim::rng::BinomialAlgorithm,
    fit: &FilterableFit,
    n_draws_cap: usize,
    n_particles: usize,
    base_seed: u64,
    schema: Option<&ObsSchema>,
    by_chain: bool,
    parameter_set: &ParameterSet,
    binding: Option<&crate::fit::scoring::OutcomeBinding>,
) -> Result<OneStepBands, String> {
    use sim::inference::{
        bootstrap_filter, multi_stream_obs::StreamProjection,
        traits::SMCConfig, BoundObs, ChainBinomialProcess, MultiStreamObsModel,
    };
    use crate::fit::runner::ObsStream;

    // ── Build the filter obs model ONCE. Resolution here is predict-specific
    // (reads only the leaves `selection` makes eligible — D76; iterates in `model.observations` order so `stream_idx`
    // maps back to its IR leaf via `bound_leaves`), so it does NOT route through
    // `resolve_and_load_obs_streams`. But the `ObsStream -> StreamSpec` assembly
    // (the bug-prone shared part) IS routed through the single shared builder
    // `stream_specs_from_obs_streams`, the same one `fit run` and `pfilter` use.
    // Each bound leaf reuses `runner::load_observations` for its cells + cadence.
    let data = config.data_spec().map_err(|e| {
        format!("this fit has no [data] block to read the observed series from: {e}")
    })?;
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)
        .map_err(|e| format!("resolving [data]: {e}"))?;
    let effective = bindings.training_by_source();
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
    // Per leaf, how many of its rows the prediction conditions on: every row
    // unless a score bound outcomes, whose rows past these the pass goes on
    // to predict and assimilate.
    let mut n_conditioning: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for obs_model in &model.observations {
        if !selection.includes(&obs_model.name) {
            continue; // not eligible: skipped or failed, and named by the caller (D76)
        }
        let mut data_path = bound_path(effective, obs_model)?;
        let siblings: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().filter(|o| o.source == obs_model.source).collect();
        let mut rows =
            crate::fit::runner::load_observations(data_path, obs_model, &siblings, &time_opts)?;
        n_conditioning.insert(obs_model.name.clone(), rows.len());
        if let Some((all, file)) = binding.and_then(|b| Some((b.rows.get(&obs_model.name)?, b.files.get(&obs_model.source)?))) {
            // The outcome file's first rows are these, by row digest: the
            // score checked that before binding it.
            let n = rows.len();
            if all.len() < n
                || crate::fit::training_partition::row_digest(&all[..n])? != crate::fit::training_partition::row_digest(&rows)?
            {
                return Err(format!(
                    "internal error: the outcome rows of '{}' do not begin with the {n} rows the prediction \
                     conditions on", obs_model.name));
            }
            rows = all.clone();
            data_path = file;
        }
        let projection = StreamProjection::from_ir(&obs_model.projection, &compiled, &obs_model.name)?;
        let times = crate::pfilter::stream_times_of_rows(obs_model, &rows, data_path, &time_opts)?;
        let (obs, cells, aux) = crate::fit::runner::observation_columns(&rows);
        obs_streams.push(ObsStream {
            name: obs_model.name.clone(),
            projection,
            obs_model_ir: obs_model.clone(),
            data: obs,
            cells,
            aux,
            times,
            rows,
        });
        bound_leaves.push(obs_model);
    }
    if obs_streams.is_empty() {
        return Err("no observation streams to predict — check that the model's \
                    observation sources are bound to data in the fit config, and that \
                    --stream (if given) names a real stream"
            .into());
    }

    // The state set (gh#1079): the filter below walks every row of every
    // stream here, the held-out ones included, so that is what each band's
    // state was conditioned on. And where each row's value is assimilated
    // (the end of what it measures), to find a filter stop's row again.
    // A band is written for the conditioning rows alone, so the state set is
    // theirs whether or not a score continued the pass past them.
    let mut filtered: Vec<FilteredStream> = obs_streams
        .iter()
        .map(|s| {
            let rows = &s.rows[..n_conditioning[&s.name]];
            Ok(FilteredStream {
                stream: s.name.clone(),
                rows: rows.len(),
                held_out_rows: rows.len().saturating_sub(parameter_set.partition.rows(&s.name)),
                digest: crate::fit::training_partition::row_digest(rows)?.to_hex(),
            })
        })
        .collect::<Result<_, String>>()?;
    filtered.sort_by(|a, b| a.stream.cmp(&b.stream));
    let row_ends: std::collections::HashMap<&str, Vec<f64>> = obs_streams
        .iter()
        .map(|s| (s.name.as_str(), s.rows.iter().map(|r| r.measures.end()).collect()))
        .collect();

    // The one-step band is `p(y_t | y_{1:t-1})`, drawn from the filter's own
    // accumulator. Each stream's `times` carries its declared periods, so the
    // filter resets every bin exactly where the fit's likelihood did — the
    // first one at its declared start, never accumulating from `t_start`
    // (gh#702, gh#833). Nothing to hand the filter beyond the streams.
    let specs = crate::fit::runner::stream_specs_from_obs_streams(&obs_streams);
    let (bound, _report) = BoundObs::bind(compiled.model.simulation.t_start, specs)
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
    //
    // By index, not by value: the chain each draw came from lives beside the
    // cloud (`PosteriorDraws::keys`) and has to be picked by the *same* indices,
    // which is exactly what `subsample_indices` exists for. Re-deriving the
    // stride at a second call site is how the two silently drift apart.
    let draws = fit.draws().draws();
    let total = draws.len();
    let idx = subsample_indices(total, n_draws_cap);
    let chosen: Vec<&IndexMap<String, f64>> = idx.iter().map(|&i| &draws[i]).collect();
    let n_used = chosen.len();
    if n_used < total {
        out.say(format_args!(
            "predict: one_step mode — subsampling {n_used} of {total} posterior \
             draws (raise with --n-draws)"
        ));
    }
    // The chain behind each replayed draw. Every entry is `None` when the
    // cloud's `draws.tsv` carries no chain column; the pool then has a single
    // unkeyed bucket and `--by-chain` writes no per-chain rows.
    let chain_of_draw: Vec<Option<usize>> = match fit.draws().keys() {
        Some(k) => idx.iter().map(|&i| k.per_draw[i].map(|(c, _)| c)).collect(),
        None => vec![None; idx.len()],
    };

    let smc_config = SMCConfig {
        n_particles,
        dt,
        t_start: compiled.model.simulation.t_start,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, path_recording: Default::default(),
        record_prequential: true,
        record_predictions: false, // gh#520: one-step-ahead draws come from `.prequential`, not `.predictions`
        record_stream_means: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        // D78: the sampler the fit ran, as the forecast does. Both samplers
        // draw exactly Binomial(n, p), but from different random numbers, so a
        // replay of the fit's filter on another sampler is a different
        // computation from the one the fit's record describes.
        binomial,
    };

    // Each draw's parameter vector — base defaults overlaid by name. The cloud is schema-validated upstream, so
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

    // The latest end among the conditioning rows: a score's pass assimilates
    // every row up to it exactly as the prediction did, and goes on past it.
    let conditioning_end = obs_streams
        .iter()
        .flat_map(|s| s.rows[..n_conditioning[&s.name]].iter().map(|r| r.measures.end()))
        .fold(f64::NEG_INFINITY, f64::max);
    if let Some(s) = obs_streams.iter().find(|s| {
        s.rows[n_conditioning[&s.name]..].iter().any(|r| r.measures.end() <= conditioning_end + crate::OBS_SNAP_EPS)
    }) {
        // A stored fit and `--holdout-after` cut a stream by a row's end, and
        // the loader refuses rows out of time order, so this is not reached
        // from a file.
        return Err(format!(
            "internal error: a row of '{}' after the ones the prediction conditions on ends at or before \
             t = {conditioning_end}, the end of the last conditioning row: the one-step pass would assimilate \
             it before rows the prediction predicted, and so change their bands",
            s.name
        ));
    }
    // Each draw's record, kept for a score; `None` for a draw the filter
    // skipped, which is left out of the pool and the score alike.
    let mut records: Vec<Option<DrawRecord>> = (0..param_sets.len()).map(|_| None).collect();
    let pool = pool_one_step_draws(out, &param_sets, &chain_of_draw, base_seed, |d, params, seed| {
        let mut result = bootstrap_filter(&process, &obs_model, params, &smc_config, seed);
        if binding.is_none() {
            return result;
        }
        let mut degenerated = None;
        // A draw that degenerates before the end of the conditioning rows
        // degenerated in the prediction too, and is skipped by both. One that
        // degenerates at a stop `w` after it was pooled by the prediction: its
        // filter is run again with every outcome from `w` on a hole, which
        // leaves every stop in place and assimilates nothing from `w`, so its
        // record before `w` is the first run's and the prediction's bands are
        // unchanged. The score reads that record before `w` only.
        if let Err(sim::SimError::PFDegenerate { obs_window, .. }) = &result {
            let w = *obs_window;
            let t_w = obs_model.obs_time(w);
            if t_w > conditioning_end + crate::OBS_SNAP_EPS {
                let mut holed = obs_streams.clone();
                for s in &mut holed {
                    let n = n_conditioning[&s.name];
                    for i in n..s.rows.len() {
                        if s.rows[i].measures.end() >= t_w - crate::OBS_SNAP_EPS {
                            s.rows[i].cell = None;
                            s.rows[i].aux.clear();
                            s.cells[i] = None;
                            s.aux[i].clear();
                        }
                    }
                }
                let specs = crate::fit::runner::stream_specs_from_obs_streams(&holed);
                let (bound, _) = BoundObs::bind(compiled.model.simulation.t_start, specs)
                    .map_err(|r| sim::SimError::Validation(r.render()))?;
                let holed_model = MultiStreamObsModel::new(bound, compiled.clone())?;
                result = bootstrap_filter(&process, &holed_model, params, &smc_config, seed);
                degenerated = Some(w);
            }
        }
        if let Ok(r) = &mut result {
            if let Some(p) = r.prequential.as_mut() {
                records[d] = Some(DrawRecord {
                    samples: p.per_stream_samples.clone(),
                    log_liks: std::mem::take(&mut p.per_stream_log_liks),
                    joint: std::mem::take(&mut p.log_liks),
                    degenerated,
                });
            }
        }
        result
    })?;
    let PooledOneStep { stream_names, obs_times, n_pooled, .. } = &pool;
    let (stream_names, obs_times, n_pooled) = (stream_names.clone(), obs_times.clone(), *n_pooled);
    // The chains that can be banded on their own, and the draw count each will
    // report. Empty without `--by-chain`, and empty on an unkeyed cloud.
    let chain_ids: Vec<usize> = if by_chain { pool.bandable_chains() } else { Vec::new() };
    if by_chain && chain_ids.is_empty() {
        out.say("predict: one_step mode — --by-chain wrote no per-chain rows: this \
             fit's draws.tsv carries no chain column, so no chain can be named.");
    }
    let n_pooled_by_chain: BTreeMap<usize, usize> = chain_ids
        .iter()
        .map(|c| (*c, pool.by_chain[&Some(*c)].n_pooled))
        .collect();

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
        // chain id -> its rows, in the same (leaf, time) order as `rows`.
        let mut per_chain: BTreeMap<usize, Vec<BandRow>> = BTreeMap::new();
        for &si in leaf_idxs {
            let leaf = leaf_of_name[stream_names[si].as_str()];
            let stratum: Vec<(String, String)> =
                leaf.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect();
            for (ti, &t) in obs_times.iter().enumerate() {
                let cell = pool.cell(si, ti);
                if cell.is_empty() {
                    // This stream is not scheduled at this union time (all NaN,
                    // dropped) — emit no row for it (multi-cadence). A declared
                    // period's opening boundary is such a time: the filter
                    // stops there to reset the bin, nothing is observed, and
                    // there is no row to plot against (gh#702, gh#833).
                    continue;
                }
                // The row this stop assimilates, by its position in the
                // stream: the filter stops for a row at the end of what the
                // row measures, and a stream's ends strictly increase.
                let position = row_ends[stream_names[si].as_str()]
                    .iter()
                    .position(|end| (end - t).abs() <= crate::OBS_SNAP_EPS)
                    .ok_or_else(|| format!(
                        "internal error: the one-step filter predicted stream '{}' at t = {t}, \
                         where none of its {} rows ends, so the row's role cannot be read; \
                         please report this with the stream name and the time",
                        stream_names[si], row_ends[stream_names[si].as_str()].len()))?;
                // A row past the conditioning ones is predicted only by a
                // score's pass, whose bands are the score's.
                if position >= n_conditioning[stream_names[si].as_str()] {
                    continue;
                }
                let quantiles = band(&cell)
                    .map_err(|e| format!("stream '{source}' (one_step) at t={t}: {e}"))?;
                let role = parameter_set.role(&stream_names[si], position);
                // No per-row R̂ on the one-step horizon (gh#794). Its cell is a
                // pool over (posterior draws × filter particles), so the
                // sequence a chain contributes is not the posterior chain and an
                // ESS computed from its autocorrelation would be inflated by the
                // particles. Deferred rather than approximated: *both* channels
                // stay empty, so a reader cannot pick up the diluted one by
                // accident. Follow-up: gh#798.
                rows.push(BandRow {
                    time: t,
                    role,
                    stratum: stratum.clone(),
                    quantiles,
                    mean_conv: None,
                    pred_conv: None,
                });
                // One band per chain over that chain's own draws x particles
                // (`--by-chain`). This is the in-sample half of the question:
                // does each chain explain the observed record, as against the
                // free-forward rows' does each chain project the same future.
                // A chain not scheduled at this union time contributes nothing
                // and is skipped, the same way the pooled cell is.
                for &c in &chain_ids {
                    let cc = &pool.by_chain[&Some(c)].pooled[si][ti];
                    if cc.is_empty() {
                        continue;
                    }
                    per_chain.entry(c).or_default().push(BandRow {
                        time: t,
                        role,
                        stratum: stratum.clone(),
                        quantiles: band(cc).map_err(|e| {
                            format!("stream '{source}' (one_step) at t={t}, chain {}: {e}", ChainIndex(c))
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

    let scored = match binding {
        Some(_) => one_step_scored(&obs_streams, &leaf_of_name, &stream_names, &obs_times, &records, parameter_set)?,
        None => crate::fit::scoring::Scored::default(),
    };
    Ok(OneStepBands { streams, n_pooled, n_pooled_by_chain, filtered, scored })
}

/// Every row the one-step pass predicted whose value is observed, as a scored
/// cell over the draws the pool kept, and per stop the joint log density of
/// every stream observed there, which is the log of the filter's incremental
/// likelihood at that stop for each particle.
fn one_step_scored(
    obs_streams: &[crate::fit::runner::ObsStream],
    leaf_of_name: &std::collections::HashMap<&str, &ir::observation::ObservationModel>,
    stream_names: &[String],
    obs_times: &[f64],
    records: &[Option<DrawRecord>],
    parameter_set: &ParameterSet,
) -> Result<crate::fit::scoring::Scored, String> {
    use crate::fit::scoring::{JointCell, ScoreMode, ScoredCell, Scored};
    use crate::quantile::fmt_time;
    // The draws the pool kept, by their index among the draws offered.
    let kept: Vec<(usize, &DrawRecord)> = records.iter().enumerate().filter_map(|(d, r)| r.as_ref().map(|r| (d, r))).collect();
    let mut scored = Scored::default();
    for (d, r) in &kept {
        if let Some(w) = r.degenerated {
            scored.notes.push(format!(
                "the one-step filter of draw {d} degenerated at t = {}, after the conditioning rows: its density of \
                 each row ending there is zero, and it scores no row after",
                fmt_time(obs_times[w])
            ));
        }
    }
    let contribution = draw_contribution;
    // Per stop: the cells scored there, by index into `scored.cells`.
    let mut at_stop: Vec<Vec<usize>> = vec![Vec::new(); obs_times.len()];
    for (si, name) in stream_names.iter().enumerate() {
        let leaf = leaf_of_name[name.as_str()];
        let rows = &obs_streams.iter().find(|s| &s.name == name).expect("a filter stream is a bound stream").rows;
        let mut holes = 0usize;
        let mut fewer_rows = 0usize;
        for (position, row) in rows.iter().enumerate() {
            let end = row.measures.end();
            let Some(ti) = obs_times.iter().position(|t| (t - end).abs() <= crate::OBS_SNAP_EPS) else {
                return Err(format!("internal error: no stop of the one-step filter at the end of row {} of '{name}'", position + 1));
            };
            let y = match row.cell {
                Some(sim::inference::ObsCell::Scalar(v)) => v,
                _ => {
                    holes += 1;
                    continue;
                }
            };
            let mut samples = Vec::new();
            let mut log_liks = Vec::new();
            let mut fewer = false;
            for (d, r) in &kept {
                match contribution(r, ti, Some(si)) {
                    Some((ll, ys)) => {
                        fewer |= ys.is_empty();
                        log_liks.extend(ll);
                        samples.extend(ys);
                    }
                    None => {
                        fewer = true;
                        scored.notes.push(format!("the row of '{name}' ending at t = {} is unscored for draw {d}", fmt_time(end)));
                    }
                }
            }
            if fewer {
                fewer_rows += 1;
            }
            let lead = match row.measures {
                crate::fit::observed_rows::Measures::Period(p) => Some(p.stop() - p.start()),
                crate::fit::observed_rows::Measures::Instant(t) => position.checked_sub(1).map(|p| t - rows[p].measures.end()),
            };
            at_stop[ti].push(scored.cells.len());
            scored.cells.push(ScoredCell {
                scenario: String::new(),
                sweep: Vec::new(),
                leaf: name.clone(),
                source: leaf.source.clone(),
                stratum: leaf.stratum.iter().map(|k| (k.dim.clone(), k.level.clone())).collect(),
                start: row.measures.start(),
                stop: end,
                mode: ScoreMode::OneStep,
                origin: None,
                horizon: None,
                lead,
                role: parameter_set.role(name, position),
                observed: y,
                samples,
                log_liks,
            });
        }
        if holes > 0 {
            scored.notes.push(format!("{holes} row(s) of '{name}' hold no observed value (NA) and are not scored in one_step mode"));
        }
        if fewer_rows > 0 {
            scored.notes.push(format!(
                "{fewer_rows} one_step row(s) of '{name}' take their CRPS, WIS, PIT and coverage from the samples of \
                 fewer than all draws: a draw whose filter degenerated, or a particle the filter killed, has none"
            ));
        }
    }
    for (ti, cells) in at_stop.iter().enumerate() {
        let Some(&first) = cells.first() else { continue };
        let role = scored.cells[first].role;
        if cells.iter().any(|&c| scored.cells[c].role != role) {
            scored.notes.push(format!(
                "no joint one-step log score at t = {}: the streams observed there have different roles",
                crate::quantile::fmt_time(obs_times[ti])
            ));
            continue;
        }
        let lead = scored.cells[first].lead;
        let leads_agree = cells.iter().all(|&c| scored.cells[c].lead == lead);
        scored.joints.push(JointCell {
            scenario: String::new(),
            sweep: Vec::new(),
            mode: ScoreMode::OneStep,
            origin: None,
            horizon: None,
            lead: if leads_agree { lead } else { None },
            start: cells.iter().map(|&c| scored.cells[c].start).fold(f64::INFINITY, f64::min),
            stop: obs_times[ti],
            role,
            streams: cells.iter().map(|&c| scored.cells[c].leaf.clone()).collect(),
            log_liks: kept.iter().filter_map(|(_, r)| contribution(r, ti, None)).flat_map(|(ll, _)| ll).collect(),
        });
    }
    Ok(scored)
}

/// What one draw's one-step record contributes at stop `ti` to the row of
/// stream `si` (`Some`) or to the joint (`None`): the log density of the
/// outcome at each particle, and the predictive samples. A particle the
/// filter killed has density zero and no sample. At the stop where the
/// draw's filter degenerated every particle has density zero and no sample,
/// and after it the draw contributes nothing (`None`).
fn draw_contribution(r: &DrawRecord, ti: usize, si: Option<usize>) -> Option<(Vec<f64>, Vec<f64>)> {
    match r.degenerated {
        Some(w) if ti > w => return None,
        Some(w) if ti == w => return Some((vec![f64::NEG_INFINITY; r.joint[ti].len()], Vec::new())),
        _ => {}
    }
    let dead = sim::inference::prequential::dead_particles(&r.joint[ti], &r.log_liks[ti]);
    let ll: Vec<f64> = match si {
        Some(si) => r.log_liks[ti][si].clone(),
        None => r.joint[ti].clone(),
    };
    let ll = ll.into_iter().zip(&dead).map(|(l, &x)| if x { f64::NEG_INFINITY } else { l }).collect();
    let samples = match si {
        Some(si) => r.samples[ti][si].iter().zip(&dead).filter(|(_, &x)| !x).map(|(&y, _)| y).collect(),
        None => Vec::new(),
    };
    Some((ll, samples))
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

/// How a refusal names the leaf it is about: the leaf's own label, which after
/// the `[stages]` → `[method]` split is the method's name. Falls back to the
/// typed method, then to a neutral word, so a directory passed with no
/// `run.json` produces "the posterior method leaf" rather than "the '' …".
fn leaf_name(leaf: &str, method: Option<FitAlgorithm>) -> String {
    if !leaf.is_empty() {
        return leaf.to_string();
    }
    method.map(|m| m.as_str()).unwrap_or("posterior").to_string()
}

/// The actionable refusal for an explicit `--mode one_step` on a fit whose
/// backend cannot drive a particle filter. ODE's one-step predictive is
/// identical to free-forward (deterministic given θ), so the redirect points the
/// user there rather than emit a relabelled-identical band.
fn one_step_refusal(why: NotFilterable, method: Option<FitAlgorithm>, leaf: &str) -> String {
    // A leaf's label IS its method's name, so naming both would read
    // `'pgas' (pgas)`. Name it once, falling back to the typed method when a
    // directory was passed whose `run.json` did not say.
    let leaf = leaf_name(leaf, method);
    match why {
        NotFilterable::Deterministic => format!(
            "the '{leaf}' method leaf ran on the ODE backend, which is deterministic \
             given the parameters — its one-step-ahead predictive p(y_t | y_{{1:t-1}}) \
             reduces to the observation model at the deterministic state, identical to \
             the free-forward band. There is no separate one-step object to emit.\n  \
             Use the free-forward horizon instead:\n    \
             camdl predict <run> --mode free_forward"
        ),
        NotFilterable::NotAnInferenceBackend => format!(
            "the '{leaf}' method leaf ran on the Gillespie backend, which is not an \
             inference backend — a fit never produces Gillespie posterior draws, so a \
             one-step-ahead predictive is not defined. This should not occur; please \
             report it."
        ),
    }
}

/// The refusal of `--mode one_step` from a fixed point on a backend that runs
/// no particle filter, for the reason [`one_step_refusal`] gives a fit.
fn fixed_point_one_step_refusal(why: NotFilterable) -> String {
    match why {
        NotFilterable::Deterministic => "the ODE backend is deterministic given the parameters, so its \
             one-step-ahead predictive p(y_t | y_{1:t-1}) reduces to the observation model at the \
             deterministic state, identical to the free-forward band. There is no separate one-step \
             object to emit; use --mode free_forward."
            .to_string(),
        NotFilterable::NotAnInferenceBackend => "the Gillespie backend runs no particle filter, so a \
             one-step-ahead predictive is not computed on it; use --backend chain_binomial, or \
             --mode free_forward."
            .to_string(),
    }
}

/// The actionable refusal when a fit reaches `predict` with no posterior band to
/// draw. Two genuinely different causes, told apart by the leaf's method
/// (gh#343): a Bayesian sampler (PGAS / PMMH / MH) that simply has not written
/// its `draws.tsv` yet — incomplete or still running — versus a real optimizer
/// (IF2 / NLopt) that returns a single point and never has a band. Framing a
/// sampler as an "optimizer fit" misdirected the user of the *default* Bayesian
/// method to the plug-in workflow, discarding the posterior it does have.
fn plugin_refusal(method: Option<FitAlgorithm>, leaf: &str, handle: &str) -> String {
    let leaf = leaf_name(leaf, method);
    if let Some(m) = method.filter(|m| m.is_posterior_sampler()) {
        return format!(
            "the '{leaf}' method leaf is a Bayesian posterior sampler, but it has not \
             written its posterior draws (draws.tsv) — that file is written when the \
             method finishes, so the fit is likely incomplete or still running.\n  \
             Let the fit finish (or re-run it); a completed {m} leaf has a full \
             posterior cloud, and `camdl predict` will draw the band with no \
             extra flags."
        );
    }
    if let Some(m) = method.filter(|m| m.is_optimizer()) {
        return format!(
            "the '{leaf}' method leaf is an optimizer fit ({m}) — it returns a single \
             best-fit parameter set, not a distribution, so there is no posterior band \
             to draw.\n  \
             Write its estimate as a parameters file, and give that file to a command \
             that takes --params:\n    \
             camdl fit summary {handle} --params-only > params.toml\n    \
             camdl simulate <model> --params params.toml --obs-only-dir out/\n    \
             camdl predict <model> --params params.toml --data <stream>=<file>   \
             (bands at that point, treatment plug_in)"
        );
    }
    // Neither a posterior sampler nor an optimizer (a likelihood-eval method
    // such as pfilter, or an unrecognized one): this kind yields no posterior
    // distribution, so there is nothing to band — but it is NOT an optimizer fit.
    format!(
        "the '{leaf}' method leaf produced no posterior draws and is not a posterior \
         sampler, so there is no band to draw for it."
    )
}

#[cfg(test)]
mod tests {
    /// A run id and engine for the pinned keys, so they do not move with the
    /// build's version.
    const RUN: &str = "1fde5903ef4cf3f8ce25eb0661a8c042e9b6baaaa126997b2ab721793e515481";
    const OTHER_RUN: &str = "2fde5903ef4cf3f8ce25eb0661a8c042e9b6baaaa126997b2ab721793e515481";
    const ENGINE: &str = "0.0.0+test";
    const DEFAULTS_KEY: &str = "7840bfbd0dc6f1b96657e6cf30954059";
    const FREE_FORWARD_KEY: &str = "c3d82086913c29f5a729e920a61aff64";
    const SCENARIOS_AB_KEY: &str = "efe665fc26e51684ba9ac7f8f2e5a688";
    const STREAM_ONSET_KEY: &str = "0fd2a1e70cb0f7a22152c0f284e4d821";

    /// One draw's record of one stream over two stops: three particles, the
    /// second killed by the filter at stop 0 (joint weight −∞, its
    /// per-stream term read off its dead state and finite).
    fn record(degenerated: Option<usize>) -> super::DrawRecord {
        super::DrawRecord {
            samples: vec![vec![vec![3.0, 100.0, 4.0]], vec![vec![5.0, 6.0, 7.0]]],
            log_liks: vec![vec![vec![-1.0, -0.5, -2.0]], vec![vec![-1.5, -1.5, -1.5]]],
            joint: vec![vec![-1.0, f64::NEG_INFINITY, -2.0], vec![-1.5, -1.5, -1.5]],
            degenerated,
        }
    }

    /// A particle the filter killed has density zero in its stream as in the
    /// joint, so with one stream the stream's log score is the joint one, and
    /// its draw is no sample.
    #[test]
    fn a_dead_particle_scores_as_the_joint_does() {
        let r = record(None);
        let (stream, samples) = super::draw_contribution(&r, 0, Some(0)).unwrap();
        let (joint, _) = super::draw_contribution(&r, 0, None).unwrap();
        assert_eq!(stream, joint);
        assert_eq!(samples, vec![3.0, 4.0]);
        let lme = crate::fit::scoring::log_mean_exp;
        assert_eq!(lme(&stream).unwrap(), lme(&joint).unwrap());
    }

    /// A draw whose filter degenerated at a stop has density zero at every
    /// particle there and no sample, and contributes nothing after; the
    /// row's score over two draws stays finite while the other draw gives the
    /// outcome positive density, and is −∞ when both degenerated.
    #[test]
    fn a_degenerated_draw_has_density_zero_at_its_stop_and_nothing_after() {
        let (ll, samples) = super::draw_contribution(&record(Some(0)), 0, Some(0)).unwrap();
        assert_eq!(ll, vec![f64::NEG_INFINITY; 3]);
        assert!(samples.is_empty());
        assert!(super::draw_contribution(&record(Some(0)), 1, Some(0)).is_none());
        let lme = crate::fit::scoring::log_mean_exp;
        let other = super::draw_contribution(&record(None), 0, Some(0)).unwrap().0;
        let both: Vec<f64> = other.iter().chain(&ll).copied().collect();
        let s = lme(&both).unwrap();
        assert!(s.is_finite(), "one of two draws degenerated: {s}");
        assert!((s - (lme(&other).unwrap() - 2.0_f64.ln())).abs() < 1e-12, "the mean over all six particles");
        assert_eq!(lme(&[ll.clone(), ll].concat()).unwrap(), f64::NEG_INFINITY);
    }

    /// gh#896: a predict's address key is pinned. It names stored outputs, so a
    /// change to it — a reordered or renamed field of `PredictOptions`, a new
    /// default — orphans every keyed predict and must be a deliberate edit of
    /// these values, not a side effect.
    #[test]
    fn the_predict_address_key_is_pinned() {
        let key = |o: &PredictOptions| o.address_key_by(RUN, ENGINE);
        assert_eq!(key(&PredictOptions::defaults()), DEFAULTS_KEY);
        assert_eq!(key(&PredictOptions::free_forward(None)), FREE_FORWARD_KEY);
        assert_eq!(FREE_FORWARD_KEY.len(), PREDICT_KEY_HEX, "128 bits of hex");
        let scen = |names: &[&str]| PredictOptions {
            scenarios: names.iter().map(|n| n.to_string()).collect(),
            ..PredictOptions::defaults()
        }
        .normalised();
        assert_eq!(key(&scen(&["b", "a"])), key(&scen(&["a", "b"])),
            "the scenario set, not its spelling order, keys the address");
        assert_eq!(key(&scen(&["b", "a"])), SCENARIOS_AB_KEY);
        let streams = |names: &[&str]| PredictOptions {
            stream: names.iter().map(|n| n.to_string()).collect(),
            ..PredictOptions::defaults()
        }
        .normalised();
        assert!(serde_json::to_string(&streams(&["onset"])).unwrap().contains(r#""stream":"onset""#));
        assert!(serde_json::to_string(&streams(&[])).unwrap().contains(r#""stream":null"#));
        assert_eq!(key(&streams(&["onset"])), STREAM_ONSET_KEY);
        assert_eq!(key(&streams(&["b", "a"])), key(&streams(&["a", "b", "a"])),
            "the stream set, not its spelling, keys the address");
    }

    /// Every input that changes the output moves the key: the run, the engine,
    /// and each option `PredictOptions` carries for it — among them the chain
    /// selection, the `--quantities` vocabulary and
    /// `--allow-forcing-extrapolation`.
    #[test]
    fn every_output_changing_input_moves_the_predict_address_key() {
        let base = PredictOptions::free_forward(None);
        let k = base.address_key_by(RUN, ENGINE);
        assert_ne!(base.address_key_by(OTHER_RUN, ENGINE), k, "the run");
        assert_ne!(base.address_key_by(RUN, "0.0.1+test"), k, "the engine");
        let varied = [
            ("mode", PredictOptions { mode: "one_step", ..base.clone() }),
            ("reach", PredictOptions { reach: Some("2 weeks".into()), ..base.clone() }),
            ("n_draws", PredictOptions { n_draws: 7, ..base.clone() }),
            ("seed", PredictOptions { seed: 2, ..base.clone() }),
            ("stream", PredictOptions { stream: vec!["a".into()], ..base.clone() }),
            ("scenarios", PredictOptions { scenarios: vec!["a".into()], ..base.clone() }),
            ("enable", PredictOptions { enable: vec!["a".into()], ..base.clone() }),
            ("disable", PredictOptions { disable: vec!["a".into()], ..base.clone() }),
            ("sweep", PredictOptions { sweep: vec![("q".into(), vec![1.0])], ..base.clone() }),
            ("by_chain", PredictOptions { by_chain: true, ..base.clone() }),
            ("excluded_chains", PredictOptions { excluded_chains: Some(vec![1]), ..base.clone() }),
            ("quantities", PredictOptions { quantities: Some("ab".repeat(32)), ..base.clone() }),
            ("allow_forcing_extrapolation",
             PredictOptions { allow_forcing_extrapolation: true, ..base.clone() }),
            ("forecast", PredictOptions {
                forecast: Some(ForecastKey { targets: "cd".repeat(32), design_mode: None, covariates: BTreeMap::new() }),
                ..base.clone()
            }),
        ];
        let forecast_with = |design_mode, targets: &str, covariates: &[(&str, crate::args::types::CovariateStatement)]| {
            PredictOptions {
                forecast: Some(ForecastKey {
                    targets: targets.to_string(),
                    design_mode,
                    covariates: covariates.iter().map(|(f, s)| (f.to_string(), *s)).collect(),
                }),
                ..base.clone()
            }
            .address_key_by(RUN, ENGINE)
        };
        let forecast = |design_mode, targets: &str| forecast_with(design_mode, targets, &[]);
        use crate::fit::forecast_plan::DesignMode::{ExPost, Prospective};
        assert_ne!(forecast(Some(ExPost), "ab"), forecast(Some(Prospective), "ab"), "the design mode keys the address");
        assert_ne!(forecast(None, "ab"), forecast(Some(Prospective), "ab"));
        assert_ne!(forecast(None, "ab"), forecast(None, "ac"), "and the target set");
        {
            use crate::args::types::CovariateStatement::{Hold, Prospective as Stated, Realised};
            let keys = [
                forecast(None, "ab"),
                forecast_with(None, "ab", &[("pop", Stated)]),
                forecast_with(None, "ab", &[("pop", Realised)]),
                forecast_with(None, "ab", &[("pop", Hold)]),
                forecast_with(None, "ab", &[("births", Hold)]),
            ];
            let distinct: std::collections::BTreeSet<&String> = keys.iter().collect();
            assert_eq!(distinct.len(), keys.len(), "each statement, and the forcing it is for, keys the address");
            // A forecast that reads no table forcing after its origin states
            // nothing, and its key's encoding has no field for it.
            let none = ForecastKey { targets: "ab".into(), design_mode: None, covariates: BTreeMap::new() };
            assert_eq!(serde_json::to_string(&none).unwrap(), r#"{"targets":"ab"}"#);
        }
        for (what, o) in &varied {
            assert_ne!(o.address_key_by(RUN, ENGINE), k, "{what} must move the key");
        }
        assert_ne!(
            PredictOptions { excluded_chains: Some(vec![1]), ..base.clone() }.address_key_by(RUN, ENGINE),
            PredictOptions { excluded_chains: Some(vec![0]), ..base.clone() }.address_key_by(RUN, ENGINE),
            "two selections are two keys");
    }

    use super::*;

    /// `camdl predict` reads the observed series of every leaf a fit binds. Under
    /// the one-file shorthand on a model whose `cases[p in patch]` block
    /// expands to `cases_urban`/`cases_rural` (both of source `cases`), it
    /// looked each leaf's file up by source in a map keyed by leaf, found
    /// nothing, and skipped both leaves as unbound: the predictive was drawn
    /// with no observed series beside it, and nothing said so.
    #[test]
    fn observed_leaves_are_read_under_the_shorthand_on_an_indexed_block() {
        let tmp = tempfile::tempdir().unwrap();
        let data = tmp.path().join("cases_long.tsv");
        std::fs::write(&data, "time\tpatch\tcases\n7\turban\t12\n7\trural\t3\n\
                               14\turban\t18\n14\trural\t6\n").unwrap();
        let path = format!("{}/../../../ocaml/golden/sir_two_patch_long_obs.ir.json",
            env!("CARGO_MANIFEST_DIR"));
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut model: ir::Model = serde_json::from_value(v["model"].clone()).unwrap();
        for p in &mut model.parameters {
            p.value = p.value.with_value(0.1);
        }
        let compiled = sim::CompiledModel::new(model.clone()).unwrap();
        let config = crate::fit::config_v2::FitConfig::from_toml_str(&format!(
            "[model]\ncamdl = \"m.ir.json\"\n[data]\nfile = \"{}\"\n\
             [estimate]\nbeta = {{ bounds = [0.01, 0.5] }}\n[fixed]\nrho = 0.5\n\
             [method]\nalgorithm = \"if2\"\nbackend = \"chain_binomial\"\n\
             chains = 1\nparticles = 10\niterations = 1\ncooling = 0.7\n",
            data.display())).unwrap().problem;
        // The selection the verb builds: both leaves must count as bound.
        let bindings = crate::fit::stream_bindings::StreamBindings::resolve(
            &model, config.data_spec().unwrap()).unwrap();
        let bound = bindings.training_by_source();
        let streams = select_streams(
            &model.observations, |o| bound.contains_key(&o.source), &[], Site::Stream).unwrap();
        assert!(streams.skipped.is_empty(), "{:?}", streams.skipped);
        let leaves = load_leaf_obs(&model, &compiled, &config, 1.0, &streams).unwrap();
        let read: Vec<(String, Vec<f64>)> = leaves.iter()
            .map(|l| (l.stratum[0].1.clone(), l.times.clone())).collect();
        assert_eq!(read, vec![("urban".to_string(), vec![7.0, 14.0]),
                              ("rural".to_string(), vec![7.0, 14.0])],
            "both leaves' observed series are read");
    }

    /// The one-step band walks the step the fit ran at: under the one dt rule
    /// that is the model's declared dt whenever it has one — the fit's
    /// `[config] dt` (a default 1.0 here) used to win.
    #[test]
    fn the_one_step_band_walks_the_models_dt() {
        assert_eq!(predict_step_grids(Some(0.5), 1.0), (0.5, 0.5));
        assert_eq!(predict_step_grids(None, 0.25), (0.25, 0.25));
        // No model dt, `[config] dt = 7`: the fit ran at 7, so the free-forward
        // replay steps at 7 too, not at 1.0.
        assert_eq!(predict_step_grids(None, 7.0).0, 7.0, "free-forward at the fitted step");
    }
    use crate::fit::row_convergence::ChainOfDraw;

    #[test]
    fn read_convergence_finds_mh_summary_for_mh_method() {
        // An mh(-ODE) method writes `mh_summary.json` (NOT `pmmh_summary.json`),
        // so read_convergence must resolve it via the algorithm's own name. On
        // the pre-fix code — which searched only pgas/pmmh — this returned
        // NotAssessed, silently dropping the mh method's convergence.
        let dir = std::env::temp_dir().join("camdl_read_convergence_mh_summary_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("mh_summary.json"),
            r#"{"schema": "camdl.fit-summary/v2", "rhat": {"beta": 1.03}, "ess": {"beta": 250.0}}"#,
        )
        .unwrap();

        match read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Mh)) {
            ConvergenceStatus::Reported { rhat_max, ess_min } => {
                assert!((rhat_max - 1.03).abs() < 1e-9, "rhat_max from mh_summary.json");
                assert!((ess_min - 250.0).abs() < 1e-9, "ess_min from mh_summary.json");
            }
            _ => panic!("mh method must resolve its own mh_summary.json, not NotAssessed"),
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
    /// A method whose R̂ was REFUSED must not have its predictive band labelled
    /// with a reported convergence status. The refusals live in the summary's
    /// `rhat_not_reported`, and reading only the numeric maps dropped them —
    /// silently promoting "we could not assess this parameter" to "assessed,
    /// and here is the max over the ones that were".
    /// Proposal 2026-09-08 §3.4: a band drawn from a fit whose chains all
    /// began at one point carries no convergence stamp — the same demotion
    /// `fit summary` applies, so the two cannot disagree about one fit.
    #[test]
    fn read_convergence_is_not_assessed_for_a_point_start() {
        let dir = std::env::temp_dir().join("camdl_read_convergence_point_start_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("pgas_summary.json"),
            r#"{"schema": "camdl.fit-summary/v2", "rhat": {"beta": 1.001}, "ess": {"beta": 900.0}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("fit_state.toml"),
            "method = \"pgas\"\nseed = 1\ntimestamp = \"2026-01-01T00:00:00Z\"\n\
             best_loglik = -10.0\ninitial_loglik = -20.0\nbest_chain = 1\nn_chains = 4\n\
             chain_init_source = \"from_mle @mle\"\nchain_starts_kind = \"point\"\n\n\
             [start_values]\n\n[rw_sd]\n",
        )
        .unwrap();
        assert!(matches!(
            read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Pgas)),
            ConvergenceStatus::NotAssessed
        ), "chains that began together agree by construction; the band must not be stamped");
        // The spread counterpart of the same leaf reports its R̂.
        let state = std::fs::read_to_string(dir.join("fit_state.toml")).unwrap()
            .replace("chain_starts_kind = \"point\"", "chain_starts_kind = \"spread\"");
        std::fs::write(dir.join("fit_state.toml"), state).unwrap();
        assert!(matches!(
            read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Pgas)),
            ConvergenceStatus::Reported { .. }
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_convergence_is_not_reported_when_the_summary_carries_a_refusal() {
        let dir = std::env::temp_dir().join("camdl_read_convergence_refusal_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // `tau` never moved — a sampler pathology, not a missing number.
        std::fs::write(
            dir.join("pgas_summary.json"),
            r#"{"schema": "camdl.fit-summary/v2", "rhat": {"a2": 1.01},
                "ess":  {"a2": 145.0},
                "rhat_not_reported": {"tau": "constant_draws"}}"#,
        )
        .unwrap();
        if let ConvergenceStatus::Reported { rhat_max, .. } =
            read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Pgas))
        {
            panic!(
                "a fit with a refused parameter must not report a band off the \
                 parameters that survived; got max R̂ = {rhat_max}"
            );
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
            r#"{"schema": "camdl.fit-summary/v2", "rhat": {"a2": 1.01, "tau": 2.639},
                "ess":  {"a2": 145.0, "tau": null}}"#,
        )
        .unwrap();
        match read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Pgas)) {
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
            r#"{"schema": "camdl.fit-summary/v2", "rhat": {"a2": 1.01, "tau": 2.639},
                "ess":  {"a2": 145.0, "tau": 9.0}}"#,
        )
        .unwrap();
        match read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Pgas)) {
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
            r#"{"schema": "camdl.fit-summary/v2", "rhat": {"beta": 1.03}, "ess": {"beta": 250.0}}"#,
        )
        .unwrap();
        assert!(
            matches!(
                read_convergence(&crate::out::Out::new(true), &dir, Some(FitAlgorithm::Mh)),
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
            stream_means: None,
            prequential: Some(PrequentialRecorded {
                obs_times: vec![7.0, 14.0],
                log_liks: vec![vec![-5.0, -5.0], vec![-5.0, -5.0]],
                stream_names: vec!["cases".into()],
                per_stream_held_out_log_liks: Vec::new(),
                per_stream_log_liks: vec![vec![vec![-5.0, -5.0]], vec![vec![-5.0, -5.0]]],
                per_stream_samples: vec![per_time(base), per_time(base)],
            }),
            collapse: None,
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
        let out = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &unkeyed(&param_sets), 42, |draw_idx, _params, _seed| {
            if draw_idx == 1 { Err(degenerate()) } else { Ok(fake_pf_result(10.0)) }
        })
        .expect("a single degenerate draw must not abort the pool");
        assert_eq!(out.n_pooled, 2);
        assert_eq!(out.stream_names, vec!["cases".to_string()]);
        assert_eq!(out.obs_times, vec![7.0, 14.0]);
        // 2 surviving draws × 2 particles per (stream, time).
        assert_eq!(out.cell(0, 0).len(), 4);
        assert_eq!(out.cell(0, 1).len(), 4);
        // No chain column ⇒ one unkeyed bucket and nothing to band per chain.
        assert!(out.bandable_chains().is_empty());
    }

    /// Every draw's chain, for a cloud that carries no chain column — what the
    /// pooling path sees when `draws.tsv` predates keyed draws.
    fn unkeyed(param_sets: &[Vec<f64>]) -> Vec<Option<usize>> {
        vec![None; param_sets.len()]
    }

    /// `chains[i]` = chain of draw `i`, 0-based.
    fn keyed(chains: &[usize]) -> Vec<Option<usize>> {
        chains.iter().map(|c| Some(*c)).collect()
    }

    /// gh#794 one-step: the pool is partitioned by chain, and the pooled band's
    /// operand is the same multiset it was before the partition.
    #[test]
    fn one_step_pool_partitions_by_chain_and_the_pooled_cell_is_unchanged() {
        // Six draws, three per chain, two particles each.
        let param_sets = vec![vec![0.0]; 6];
        let chains = keyed(&[0, 0, 0, 1, 1, 1]);
        let out = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &chains, 42, |_, _, _| Ok(fake_pf_result(10.0)))
            .expect("nothing degenerates here");

        assert_eq!(out.bandable_chains(), vec![0, 1]);
        assert_eq!(out.n_pooled, 6);
        // Each chain holds its own three draws × two particles.
        for c in [0usize, 1] {
            assert_eq!(out.by_chain[&Some(c)].pooled[0][0].len(), 6);
            assert_eq!(out.by_chain[&Some(c)].n_pooled, 3);
            assert_eq!(out.by_chain[&Some(c)].n_offered, 3);
        }
        // And the pooled cell is exactly their union — the same multiset a flat
        // pool held, which is what keeps the `all` rows byte-identical (the band
        // sorts a copy, so order cannot matter).
        let mut pooled = out.cell(0, 0);
        let mut union: Vec<f64> = out.by_chain[&Some(0)].pooled[0][0].clone();
        union.extend_from_slice(&out.by_chain[&Some(1)].pooled[0][0]);
        pooled.sort_by(|a, b| a.partial_cmp(b).unwrap());
        union.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(pooled, union);
        assert_eq!(pooled.len(), 12);
    }

    /// A chain that lost SOME of its draws still bands, and reports what it
    /// pooled rather than what it was given. A chain that lost ALL of them
    /// yields no band at all — banding an empty cell would publish `NaN`
    /// quantiles, which reads as a band.
    #[test]
    fn a_partially_degenerate_chain_reports_its_real_count_and_a_dead_one_is_omitted() {
        // Chain 0: draws 0,1,2 — draw 1 degenerates. Chain 1: draws 3,4 — both
        // degenerate. Chain 2: draw 5 — survives.
        let param_sets = vec![vec![0.0]; 6];
        let chains = keyed(&[0, 0, 0, 1, 1, 2]);
        let out = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &chains, 42, |draw_idx, _, _| {
            if matches!(draw_idx, 1 | 3 | 4) { Err(degenerate()) } else { Ok(fake_pf_result(10.0)) }
        })
        .expect("a partially degenerate pool must not abort");

        // Chain 1 contributed nothing, so it cannot be banded: an empty cell
        // would quantile to NaN and render as a band over nothing.
        assert_eq!(out.bandable_chains(), vec![0, 2]);
        assert!(!out.by_chain.contains_key(&Some(1)));
        // Chain 0 bands, and its count is the 2 it pooled, not the 3 it was
        // offered — that gap is what the `n_draws` cell has to show.
        assert_eq!(out.by_chain[&Some(0)].n_pooled, 2);
        assert_eq!(out.by_chain[&Some(0)].n_offered, 3);
        assert_eq!(out.by_chain[&Some(2)].n_pooled, 1);
        // The pooled band still sees every surviving draw of every chain.
        assert_eq!(out.n_pooled, 3);
        assert_eq!(out.cell(0, 0).len(), 6); // 3 surviving draws × 2 particles
    }

    #[test]
    fn one_step_pool_aborts_on_structural_error() {
        // A non-degeneracy error (here a numerical collapse) trips identically
        // for every draw — it must abort on first occurrence, naming the draw,
        // not be skipped 200 times.
        let param_sets = vec![vec![0.0]; 3];
        let err = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &unkeyed(&param_sets), 42, |draw_idx, _params, _seed| {
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
        let err = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &unkeyed(&param_sets), 42, |_, _, _| Err(degenerate()))
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
        let _ = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &unkeyed(&param_sets), 42, |draw_idx, _params, seed| {
            seeds_with_skip.borrow_mut().push((draw_idx, seed));
            if draw_idx == 0 { Err(degenerate()) } else { Ok(fake_pf_result(10.0)) }
        });
        let seeds_no_skip = RefCell::new(Vec::new());
        let _ = pool_one_step_draws(&crate::out::Out::new(true), &param_sets, &unkeyed(&param_sets), 42, |draw_idx, _params, seed| {
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
        BandRow { time, role: RowRole::Trained, stratum, quantiles, mean_conv: None, pred_conv: None }
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
        assert_eq!(PredictiveMode::OneStepAhead.as_str(), "one_step");
        assert_eq!(PredictiveMode::FreeForward.as_str(), "free_forward");
    }

    /// gh#1079: a data row's role is its position against the stream's
    /// training prefix, and the three roles are written as the spec names them.
    #[test]
    fn a_data_rows_role_is_its_position_against_the_training_prefix() {
        assert_eq!(RowRole::of_data_row(0, 3), RowRole::Trained);
        assert_eq!(RowRole::of_data_row(2, 3), RowRole::Trained);
        assert_eq!(RowRole::of_data_row(3, 3), RowRole::HeldOut);
        assert_eq!(RowRole::of_data_row(0, 0), RowRole::HeldOut);
        assert_eq!(
            [RowRole::Trained, RowRole::HeldOut, RowRole::Future].map(|r| r.as_str()),
            ["trained", "held_out", "future"],
        );
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
        assert!(msg.contains("--mode free_forward"), "redirects to free-forward: {msg}");
    }

    #[test]
    fn refusal_for_a_posterior_sampler_is_not_an_optimizer_message() {
        // gh#343: a Bayesian sampler method (PGAS / PMMH / MH) that reaches the
        // refusal — its draws.tsv is absent (incomplete / still running) — must
        // NOT be called an "optimizer fit". PGAS is the *default* Bayesian path;
        // the misclassification made the blessed posterior-predictive verb
        // unusable for it. The refusal must name the missing draws, never the
        // plug-in-simulate optimizer workflow (which discards the posterior).
        for m in [FitAlgorithm::Pgas, FitAlgorithm::Pmmh, FitAlgorithm::Mh] {
            let msg = plugin_refusal(Some(m), "posterior", "@post");
            assert!(
                !msg.contains("optimizer"),
                "a {m} leaf must not be described as an optimizer fit; got: {msg}"
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
            let msg = plugin_refusal(Some(m), "scout", "@scout");
            assert!(msg.contains("optimizer fit"), "a {m} leaf is an optimizer fit; got: {msg}");
            assert!(
                msg.contains("camdl fit summary @scout --params-only") && msg.contains("--params params.toml"),
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
            RowRole::Trained,
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
        let row = band_row(56.0, RowRole::Trained, vec![], CellDraws { predictive: &preds, mean: &means }, &chains)
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
    fn per_row_convergence_columns_render_between_the_method_stamp_and_n_draws() {
        let (means, preds, chains) = diluted_forecast_cell(12.0, 230.0);
        let row = band_row(56.0, RowRole::Trained, vec![], CellDraws { predictive: &preds, mean: &means }, &chains)
            .expect("the cell bands");
        let rows = vec![row];
        let tsv = render_predictive_tsv_sections(
            &[],
            &[PredictiveSection {
                chain: ChainLabel::All,
                scenario: "fitted".to_string(),
                sweep: Vec::new(),
                mode: PredictiveMode::FreeForward,
                quantity: Quantity::Observation,
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
        // The method stamp is the fit's worst parameter; the per-row numbers are
        // this row's. They are different numbers, which is the whole point.
        assert_eq!(cells[at("fit_rhat_max")], "2.7863");
        assert_ne!(cells[at("rhat_mean")], "", "an assessed row reports rhat_mean");
        assert_ne!(cells[at("rhat_pred")], "", "an assessed row reports rhat_pred");
        assert_ne!(
            cells[at("rhat_mean")], cells[at("fit_rhat_max")],
            "the per-row R̂ must not be the method's provenance stamp"
        );
    }

    #[test]
    fn a_cloud_with_no_chain_keys_leaves_the_per_row_columns_empty() {
        let (means, preds, _) = diluted_forecast_cell(12.0, 230.0);
        let unkeyed = ChainOfDraw(vec![None; means.len()]);
        let row = band_row(56.0, RowRole::Trained, vec![], CellDraws { predictive: &preds, mean: &means }, &unkeyed)
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
            mode: PredictiveMode::FreeForward,
            quantity: Quantity::Observation,
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

        let per_chain =
            PredictiveSection { chain: ChainLabel::One(ChainIndex(2)), n_draws: 20, ..pooled };
        // Rebuild the pooled section (it was consumed by the struct update).
        let pooled = PredictiveSection {
            chain: ChainLabel::All,
            scenario: "fitted".to_string(),
            sweep: Vec::new(),
            mode: PredictiveMode::FreeForward,
            quantity: Quantity::Observation,
            treatment: TreatmentKind::Posterior,
            convergence: conv,
            n_draws: 40,
            rows: &rows,
        };
        let with = render_predictive_tsv_sections(&[], &[pooled, per_chain]);
        let lines: Vec<&str> = with.trim_end().lines().collect();
        assert!(lines[0].starts_with("chain\tscenario\t"), "chain leads: {}", lines[0]);
        assert!(lines[1].starts_with("all\tfitted\t"), "the pooled row is `all`: {}", lines[1]);
        // The chain's number, as `chain_N/` and `--exclude-chains` carry it.
        assert!(lines[2].starts_with("2\tfitted\t"), "chain 2 renders as 2: {}", lines[2]);
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
            mode: PredictiveMode::FreeForward,
            quantity: Quantity::Observation,
            treatment: TreatmentKind::Posterior,
            convergence: conv,
            n_draws: 40,
            rows: &rows,
        };
        let pooled_only = render_predictive_tsv_sections(&[], &[mk(ChainLabel::All)]);
        let with_chain =
            render_predictive_tsv_sections(&[], &[mk(ChainLabel::All), mk(ChainLabel::One(ChainIndex(0)))]);
        let stripped: String = with_chain
            .lines()
            .filter(|l| !l.starts_with("0\t")) // drop the per-chain rows
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
            band_row(56.0, RowRole::Trained, vec![], CellDraws { predictive: &preds, mean: &means }, &chains)
                .expect("the pooled cell bands");
        assert!(pooled.mean_conv.is_some(), "the pooled row does carry one");

        // What `assemble_predictive` builds for a per-chain row.
        let one_chain = BandRow {
            time: 56.0,
            role: RowRole::Trained,
            stratum: vec![],
            quantiles: pooled.quantiles.clone(),
            mean_conv: None,
            pred_conv: None,
        };
        let rows = vec![one_chain];
        let tsv = render_predictive_tsv_sections(
            &[],
            &[PredictiveSection {
                chain: ChainLabel::One(ChainIndex(0)),
                scenario: "fitted".to_string(),
                sweep: Vec::new(),
                mode: PredictiveMode::FreeForward,
                quantity: Quantity::Observation,
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
    fn chain_label_renders_the_chain_number_and_pooled_as_all() {
        assert_eq!(ChainLabel::All.as_cell(), "all");
        assert_eq!(ChainLabel::One(ChainIndex(0)).as_cell(), "0");
        assert_eq!(ChainLabel::One(ChainIndex(7)).as_cell(), "7");
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
                mode: PredictiveMode::FreeForward,
                quantity: Quantity::Observation,
                treatment: TreatmentKind::Posterior,
                convergence: ConvergenceStatus::Reported { rhat_max: 1.01, ess_min: 420.0 },
                n_draws: 40,
                rows: &stream.rows,
            }],
        );
        let lines: Vec<&str> = tsv.trim_end().lines().collect();
        assert_eq!(lines[0],
            "scenario\ttime\tpatch\tmode\thorizon\ttreatment\trole\tquantity\tfit_rhat_max\tfit_ess_min\
             \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1],
            "fitted\t7\tBo\tfree_forward\t\tposterior\ttrained\tobservation\t1.0100\t420\t\t\t\t\t40\t0\t1\t3\t6\t12");
        assert_eq!(lines[2],
            "fitted\t7\tBombali\tfree_forward\t\tposterior\ttrained\tobservation\t1.0100\t420\t\t\t\t\t40\t0\t0\t1\t3\t7");
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
                mode: PredictiveMode::FreeForward,
                quantity: Quantity::Observation,
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
            "scenario\ttime\tmode\thorizon\ttreatment\trole\tquantity\tfit_rhat_max\tfit_ess_min\
             \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95");
        assert_eq!(lines[1], "fitted\t1\tfree_forward\t\tposterior\ttrained\tobservation\t\t\t\t\t\t\t12\t1\t2\t3\t4\t5");
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
                    mode: PredictiveMode::FreeForward,
                    quantity: Quantity::Observation,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
                PredictiveSection {
                    chain: ChainLabel::All,
                    scenario: "fitted".to_string(),
                    sweep: vec![("k".to_string(), 12.0)],
                    mode: PredictiveMode::FreeForward,
                    quantity: Quantity::Observation,
                    treatment: TreatmentKind::Posterior,
                    convergence: conv,
                    n_draws: 10,
                    rows: &rows,
                },
                PredictiveSection {
                    chain: ChainLabel::All,
                    scenario: "fitted".to_string(),
                    sweep: Vec::new(), // one-step is sweep-agnostic
                    mode: PredictiveMode::OneStepAhead,
                    quantity: Quantity::Observation,
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
            "scenario\tsweep:k\ttime\tmode\thorizon\ttreatment\trole\tquantity\tfit_rhat_max\tfit_ess_min\
             \trhat_mean\tess_mean\trhat_pred\tess_pred\tn_draws\tq05\tq25\tq50\tq75\tq95",
            "sweep:k column follows scenario"
        );
        assert_eq!(lines[1],
            "fitted\t8\t7\tfree_forward\t\tposterior\ttrained\tobservation\t1.0000\t100\t\t\t\t\t10\t1\t2\t3\t4\t5");
        assert_eq!(lines[2],
            "fitted\t12\t7\tfree_forward\t\tposterior\ttrained\tobservation\t1.0000\t100\t\t\t\t\t10\t1\t2\t3\t4\t5");
        // One-step row: the sweep:k cell is blank (empty), not a fabricated value.
        assert_eq!(lines[3],
            "fitted\t\t7\tone_step\t\tposterior\ttrained\tobservation\t1.0000\t100\t\t\t\t\t10\t1\t2\t3\t4\t5");
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
    fn treatment_and_mode_labels_are_legible() {
        assert_eq!(PredictiveMode::FreeForward.as_str(), "free_forward");
        assert_eq!(TreatmentKind::Posterior.label(), "posterior");
        assert_eq!(TreatmentKind::PlugIn.label(), "plug_in");
        let point: IndexMap<String, f64> = [("beta".to_string(), 0.5)].into_iter().collect();
        let plug = PosteriorDraws::plug_in(point, 3, cb()).unwrap();
        assert_eq!(plug.treatment(), TreatmentKind::PlugIn);
        assert_eq!(plug.n_draws(), 3);
        assert_eq!(plug.convergence, ConvergenceStatus::NotAssessed);
        assert!(plug.keys().is_none(), "a point's copies carry no chain keys");
        assert!(PosteriorDraws::plug_in(IndexMap::new(), 0, cb()).is_err(), "no copies is refused");
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
        let fit = FitResult::PointEstimate { method: Some(FitAlgorithm::If2), name: "scout".into(), backend: None };
        match fit.into_treatment() {
            ParamTreatment::Winner { method, name, .. } => {
                assert_eq!(method, FitAlgorithm::If2);
                assert_eq!(name, "scout");
            }
            other => panic!("an optimizer's point must become its winner, not {other:?}"),
        }
        // A sampler with no draws yet stays refused.
        let fit = FitResult::PointEstimate { method: Some(FitAlgorithm::Pgas), name: "pgas".into(), backend: None };
        assert!(matches!(fit.into_treatment(), ParamTreatment::PlugIn { .. }));
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
            ParamTreatment::PlugIn { .. } | ParamTreatment::Winner { .. } => panic!("a posterior fit must keep its cloud"),
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

    /// gh#589: removing the snapshots predict added for the projection leaves
    /// every retained snapshot's state as it was and folds the removed
    /// intervals' flows forward, so the cumulative flow at a retained time is
    /// the same with or without the extras.
    #[test]
    fn dropping_extra_snapshots_folds_their_flows_into_the_next_one() {
        use sim::state::{Flows, FlowVec, IntState, RealState, Snapshot};
        let snap = |t: f64, s: i64, f: u64| Snapshot {
            t,
            int_state: IntState::from_vec(vec![s]),
            real_state: RealState::new(0),
            flows: Flows::Int(FlowVec::from_vec(vec![f]).counts),
        };
        let mut traj = sim::Trajectory::new();
        traj.snapshots = vec![
            snap(0.0, 10, 0),
            snap(1.0, 9, 1),
            snap(2.0, 7, 2), // extra
            snap(2.5, 6, 1), // extra
            snap(3.0, 5, 1),
            snap(4.0, 4, 1), // extra, after the last retained one within `keep`
        ];
        let out = declared_grid_replay(&traj, 6, &[2.0, 2.5, 4.0]);
        let got: Vec<(f64, i64, u64)> = out
            .snapshots
            .iter()
            .map(|s| (s.t, s.int_state.counts[0], s.flows.as_int()[0]))
            .collect();
        assert_eq!(got, vec![(0.0, 10, 0), (1.0, 9, 1), (3.0, 5, 4)]);
        let cum = |tr: &sim::Trajectory, t: f64| -> u64 {
            tr.snapshots.iter().take_while(|s| s.t <= t).map(|s| s.flows.as_int()[0]).sum()
        };
        for t in [0.0, 1.0, 3.0] {
            assert_eq!(cum(&out, t), cum(&traj, t), "cumulative flow at t = {t}");
        }
    }
}
