//! Typed inference diagnostics — machine-readable, severity-classified,
//! serializable to JSON for downstream tooling (camdl-book, camdl-vignettes,
//! CI pipelines) to consume programmatically.
//!
//! Call sites push `DiagnosticKind` variants; the collector handles
//! rendering, severity, hints, and serialization.

use serde::{Serialize, Deserialize};

/// Severity level for inference diagnostics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

/// A typed diagnostic emitted during inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Diagnostic {
    pub kind: DiagnosticKind,
    pub severity: Severity,
    pub message: String,
    pub stage: String,
    pub timestamp: String,
}

/// The θ-move kernel behind an acceptance rate (gh#631). Bands differ:
/// random-walk MH targets ~[15%, 50%]; NUTS dual-averaging targets ~0.8,
/// healthy ≈ [60%, 95%] — applying the RW band to NUTS reported every
/// well-tuned fit as `severity: error`, burying real failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceptanceKernel {
    RandomWalk,
    Nuts,
}

impl AcceptanceKernel {
    /// The healthy acceptance band `(lo, hi)`, inclusive at both ends.
    ///
    /// **The only place these numbers live.** They were previously spelled
    /// three ways — `[15%, 50%]` in the message a user reads, `[10%, 50%]` in
    /// the predicate that decides whether to emit it, and a third pair in
    /// `severity()` — so a random-walk chain accepting 12% was inside the
    /// applied band and outside the published one, and passed silently.
    ///
    /// Random-walk MH: optimal scaling for a high-dimensional target is 0.234
    /// (Roberts, Gelman & Gilks 1997, _Ann. Appl. Probab._ 7(1):110-120), and
    /// `[0.15, 0.50]` is the band around it that camdl publishes in
    /// `docs/workflow.md`. NUTS: dual averaging targets 0.8 by construction, so
    /// a rate of 0.9-0.99 is the sampler working, not failing (gh#631).
    pub fn healthy_band(self) -> (f64, f64) {
        match self {
            Self::RandomWalk => (0.15, 0.50),
            Self::Nuts => (0.60, 0.95),
        }
    }

    /// The band as it appears in a message, naming the kernel it belongs to.
    pub fn band_label(self) -> String {
        let (lo, hi) = self.healthy_band();
        let who = match self {
            Self::RandomWalk => "random-walk MH",
            Self::Nuts => "NUTS block; ~80% is the target",
        };
        format!("[{:.0}%, {:.0}%] ({})", lo * 100.0, hi * 100.0, who)
    }
}

/// The diagnostic an acceptance `rate` deserves under `kernel`, or `None` when
/// it is inside that kernel's healthy band.
///
/// Every emitter routes through this rather than re-deriving the comparison:
/// a band that is decided in one place and rendered from another is how the
/// NUTS false-firing of gh#631 and the 10%-vs-15% discrepancy of gh#299 item 3
/// both arose.
pub fn acceptance_diagnostic(
    rate: f64,
    param: Option<String>,
    kernel: AcceptanceKernel,
) -> Option<DiagnosticKind> {
    let (lo, hi) = kernel.healthy_band();
    (!(lo..=hi).contains(&rate))
        .then_some(DiagnosticKind::AcceptanceRateUnhealthy { rate, param, kernel })
}

/// Machine-readable diagnostic classification.
///
/// Each variant carries exactly the data needed for programmatic decisions.
/// The variant name is the stable identifier that downstream tooling and
/// CI pipelines should match on.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DiagnosticKind {
    // ── Convergence ──────────────────────────────────────────────
    RhatHigh {
        param: String,
        rhat: f64,
        threshold: f64,
    },
    ChainDiverged {
        chain_id: usize,
        n_chains: usize,
    },
    MultimodalLikelihood {
        ll_spread: f64,
        max_chain_agreement: f64,
    },
    ConvergenceIncomplete {
        max_chain_agreement: f64,
        n_unconverged: usize,
        n_total: usize,
    },

    // ── ESS / Particle Filter ────────────────────────────────────
    LowESS {
        obs_time: f64,
        ess: f64,
        n_particles: usize,
        ess_fraction: f64,
    },
    LowESSAtMLE {
        ess_mean: f64,
        ess_min: f64,
        n_particles: usize,
    },
    InitialLoglikInfinite,
    /// gh#110. The chain's *initial* PF evaluation at its starting
    /// θ returned NEG_INFINITY (typically via Err(PFDegenerate)
    /// collapsing through run_quick_pfilter_with_dt's Err→-∞ path).
    /// The chain is skipped — other chains in a multi-chain run
    /// continue normally. The reason string carries the upstream
    /// diagnostic (e.g. "ESS collapsed at obs 7 after 0.4s") so
    /// the user can tell which init was pathological.
    BadInit {
        /// **1-based**, matching the `chain_N/` directories, the stderr
        /// refusal, `chain_starts.tsv`'s `chain_id` column and
        /// `progress.json`'s per-chain rows (gh#781). The samplers count
        /// chains from zero internally and add one when they build this
        /// record, which is the artifact a reader joins the others against.
        chain_id: usize,
        /// Estimated parameter name → starting value on the natural
        /// scale, exactly as offered to the inference engine.
        params: std::collections::BTreeMap<String, f64>,
        /// One-line cause from the upstream PFDegenerateKind /
        /// fallback message. Surface in the diagnostic so the user
        /// can correlate with chain_starts.tsv.
        reason: String,
        /// What the swarm managed at the observation that lost support,
        /// one record per declared stream, keyed by stream name and
        /// observation time. The structured half of `reason`, which is
        /// rendered from these same fields — so a consumer reads them
        /// instead of regexing the prose. Empty for a chain whose start
        /// failed some other way (a PF degeneracy, an all-dead swarm):
        /// no observation was scored, so there is nothing to report.
        attempts: Vec<crate::inference::obs_attempt::StreamAttempt>,
    },

    // ── NUTS ─────────────────────────────────────────────────────
    MaxTreeDepthHits {
        n_hits: usize,
        n_sweeps: usize,
        pct: f64,
        max_depth: usize,
    },
    DivergentTransitions {
        n_divergent: usize,
        n_sweeps: usize,
    },

    // ── PGAS ─────────────────────────────────────────────────────
    DegenerateAncestorSampling {
        pct: f64,
        n_degenerate: usize,
        n_substeps: usize,
    },
    LowTrajectoryRenewal {
        renewal: f64,
    },
    /// gh#791. Renewal is concentrated at the END of the series: the per-bin
    /// profile rises steeply from the first bin to the last.
    ///
    /// A **different** finding from [`Self::LowTrajectoryRenewal`], which keys
    /// on the aggregate. The aggregate is a weighted mean over the bins, and
    /// its late terms are high in most runs, so a run whose early bins sit at
    /// 0.03 can still average a third and never trip the aggregate rule. Keyed
    /// on the SHAPE (`last_bin − first_bin`) rather than the level, so a run
    /// that renews poorly but uniformly in time draws the aggregate finding
    /// instead: that is a different failure with a different remedy.
    ///
    /// # What this finding does NOT claim
    ///
    /// **It does not name a cause.** The gradient reads only the two end bins,
    /// so it cannot distinguish the two shapes that produce a steep rise, and
    /// [`Self::render`] therefore describes the shape and hands the reader the
    /// discriminator rather than asserting a mechanism:
    ///
    /// - a **flat, near-zero early region followed by a step** — the coalesced
    ///   genealogy, measured at 0.03-0.07 across six bins on the gh#791 Ebola
    ///   runs;
    /// - a **smooth monotone ramp** — the ordinary finite coalescence depth of
    ///   a long series, which fires with a perfectly respectable prefix. The
    ///   repository's own `tests/fixtures/polio_afp_es` fires at prefix 0.449
    ///   on a 0.06 → 0.31 → 0.53 → … → 0.99 ramp with no flat region, and
    ///   `sir_T160_N40` at prefix 0.618. Firing is defensible on both; a 45% or
    ///   62% prefix is not a path "held at the reference", and a message saying
    ///   so would contradict the number printed beside it.
    PathRenewalCoalesced {
        /// Mean renewal over the first half of the bins.
        prefix: f64,
        /// `last_bin − first_bin`. Near 0 when renewal is uniform in time,
        /// large when it is concentrated late. Reads only the two end bins —
        /// see the type doc for what it therefore cannot say.
        gradient: f64,
        /// Renewal in the first bin — the earliest tenth of the series, where
        /// the initial condition and the earliest dynamics live.
        first_bin: f64,
        /// Renewal in the last bin.
        last_bin: f64,
        /// The aggregate `trajectory_renewal` this profile resolves, so the
        /// message can say what reading the aggregate alone would have given.
        aggregate: f64,
        /// Bins in the profile, and in the `prefix` mean — carried so the
        /// message states the span rather than assuming the reader knows it.
        n_bins: usize,
        /// Leading bins the `prefix` mean spans.
        n_prefix_bins: usize,
    },
    /// gh#783. Sweeps in which every particle scored zero observation density
    /// at some observation window, so the filter weight vector there could not
    /// be sampled. Distinct from `DegenerateAncestorSampling`, which is about
    /// the ANCESTOR weights: that one says no particle could reach the
    /// reference, this one says no particle could explain the data.
    ///
    /// Reported rather than fatal — `pgas::WeightCollapse` carries the argument
    /// — so this is the only place a run says the sweeps happened.
    FilterWeightCollapse {
        /// Sweeps with at least one collapsed observation window.
        n_sweeps: usize,
        /// Sweeps examined, so the rate is readable.
        n_total_sweeps: usize,
        /// Collapsed observation windows summed over those sweeps.
        n_windows: usize,
    },
    /// gh#685. Observations at which the conditional filter's effective
    /// sample size, averaged over the retained sweeps, is below a small
    /// fraction of the particle count. Distinct from `FilterWeightCollapse`:
    /// there every weight was zero, here the weights are finite and one or a
    /// few particles carry all of the mass — which `min_alive` reads as the
    /// full swarm. The following resample copies those few particles into
    /// every slot, so the path through that observation is drawn from a
    /// handful of candidates, sweep after sweep.
    ///
    /// One finding per stage over the pooled profile; the threshold and its
    /// argument live with the accumulator (`cli::fit::filter_ess`).
    FilterStarved {
        /// Observations whose mean ESS is below the bar.
        n_starved: usize,
        /// Observations scored by at least one retained sweep.
        n_obs: usize,
        /// Particles per sweep.
        n_particles: usize,
        /// The bar: the fraction of `n_particles` a mean ESS is under.
        starved_below: f64,
        /// The worst observation's time, mean ESS and smallest ESS.
        worst_time: f64,
        worst_mean: f64,
        worst_min: f64,
    },
    GammaDensityDisabled {
        reason: String,
    },

    // ── PMMH ─────────────────────────────────────────────────────
    AcceptanceRateUnhealthy {
        rate: f64,
        param: Option<String>,
        /// Which θ-move kernel produced the rate (gh#631): the healthy band is
        /// kernel-specific — [15%, 50%] for random-walk MH, [60%, 95%] for a
        /// NUTS block (≈0.8 is the TARGET there, not a failure). Serialized so
        /// diagnostics.json readers can key on it too.
        kernel: AcceptanceKernel,
    },

    // ── Parameters ───────────────────────────────────────────────
    ParamNearBound {
        param: String,
        value: f64,
        bound: f64,
        bound_type: String,
    },
    ProfileCIUnbounded {
        param: String,
        direction: String,
    },
    FlatProfile {
        param: String,
        curvature: f64,
    },
    AutoRwSd {
        param: String,
        rw_sd: f64,
    },
    CompressedLogitPosition {
        param: String,
        z: f64,
    },
    AutoRwSdNoConsensus {
        n_good: usize,
        n_total: usize,
    },

    // ── Cooling / IF2 ────────────────────────────────────────────
    CoolingExhausted {
        exhausted_at_iter: usize,
        total_iters: usize,
        rw_fraction_at_exhaustion: f64,
    },

    // ── Observation Model ────────────────────────────────────────
    ObsModelMismatch {
        obs_time: f64,
        observed: f64,
        predicted_mean: f64,
        n_sigma: f64,
    },
    ZeroRateNonzeroFlow {
        transition: String,
        flow: u64,
    },

    // ── Tempering ────────────────────────────────────────────────
    LowSwapRate {
        rung_i: usize,
        rung_j: usize,
        beta_i: f64,
        beta_j: f64,
        rate: f64,
    },

    // ── Resume ───────────────────────────────────────────────────
    ResumeConfigMismatch {
        expected: String,
        found: String,
    },
    ResumeParamMissing {
        param: String,
    },
}

/// The chain-agreement value above which a `RhatHigh` / `ConvergenceIncomplete`
/// finding is an **error** rather than a warning.
///
/// This is the SEVERITY ladder, and it is deliberately not the band a renderer
/// glyphs against (`method_result::RhatBand`, keyed on the threshold camdl
/// certifies at) nor the band the refine gate refuses at
/// (`GateConfig::a_thresh`). "May this fit be reported as converged", "does
/// this finding stop a run", and "which glyph goes in this cell" are three
/// questions; one constant answering all three would move all three together.
pub const CONVERGENCE_ERROR_SEVERITY: f64 = 1.5;

/// Below this magnitude a non-zero starting value is printed in scientific
/// notation. `{:.4}` leaves under one significant digit of anything smaller,
/// which is the range the rates in these models live in.
const START_VALUE_SMALL: f64 = 1e-3;

/// At or above this magnitude a starting value is printed in scientific
/// notation. `{:.4}` on a population-sized number is six digits of integer
/// part followed by four decimals nobody set.
const START_VALUE_LARGE: f64 = 1e5;

/// A chain's starting value as the `BadInit` line should print it: scientific
/// notation where four fixed decimals would hide the number, plain decimals
/// otherwise.
///
/// `{:.4}` alone is not a rendering of a rate. Three chains that started at
/// `6.678e-5`, `1.375e-4` and `5.922e-5` each printed `0.0001` — the same
/// string as one another and as the true value — on the one line a reader
/// consults to judge whether the start was sane. An ordinary bad-start
/// refusal therefore read as a refusal at the truth, and cost gh#876 its
/// whole investigation (gh#880).
fn format_start_value(v: f64) -> String {
    let mag = v.abs();
    // A NaN `mag` takes the exponential branch here where the two-comparison
    // form took the decimal one; both render NaN and ±inf identically, so the
    // output is unchanged.
    if v != 0.0 && !(START_VALUE_SMALL..START_VALUE_LARGE).contains(&mag) {
        format!("{:.4e}", v)
    } else {
        format!("{:.4}", v)
    }
}

/// What a hint needs to know about the run around a finding, as opposed to
/// about the finding itself.
///
/// Advice that is wrong for the run is worse than no advice. The `BadInit`
/// list told the reader of a run in which every chain was refused to "treat
/// the surviving chains as the result", three times over, with no surviving
/// chain to treat (gh#880). Anything a hint asserts about the run is read
/// from here, and the default asserts nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HintContext {
    /// Chains that finished and are in the pooled result, where the caller
    /// knows. `None` means the caller has no count to offer, and a hint that
    /// would depend on it is withheld rather than guessed.
    pub chains_completed: Option<usize>,
}

impl DiagnosticKind {
    pub fn severity(&self) -> Severity {
        match self {
            Self::InitialLoglikInfinite => Severity::Error,
            Self::BadInit { .. } => Severity::Error,
            Self::RhatHigh { rhat, .. } if *rhat > CONVERGENCE_ERROR_SEVERITY => Severity::Error,
            Self::RhatHigh { .. } => Severity::Warning,
            Self::ConvergenceIncomplete { max_chain_agreement, .. }
                if *max_chain_agreement > CONVERGENCE_ERROR_SEVERITY => Severity::Error,
            Self::ConvergenceIncomplete { .. } => Severity::Warning,
            Self::DivergentTransitions { .. } => Severity::Warning,
            Self::LowESS { ess_fraction, .. } if *ess_fraction < 0.05 => Severity::Error,
            Self::LowESS { .. } => Severity::Warning,
            Self::LowESSAtMLE { ess_min, n_particles, .. }
                if *ess_min < (*n_particles as f64 * 0.05) => Severity::Error,
            Self::LowESSAtMLE { .. } => Severity::Warning,
            Self::MaxTreeDepthHits { pct, .. } if *pct > 50.0 => Severity::Error,
            Self::MaxTreeDepthHits { .. } => Severity::Warning,
            Self::AutoRwSd { .. } => Severity::Info,
            Self::AutoRwSdNoConsensus { .. } => Severity::Warning,
            Self::CompressedLogitPosition { .. } => Severity::Warning,
            Self::ZeroRateNonzeroFlow { .. } => Severity::Warning,
            Self::ResumeConfigMismatch { .. } => Severity::Error,
            Self::ResumeParamMissing { .. } => Severity::Warning,
            Self::LowSwapRate { rate, .. } if *rate < 0.01 => Severity::Error,
            Self::LowSwapRate { .. } => Severity::Warning,
            Self::AcceptanceRateUnhealthy { rate, kernel: AcceptanceKernel::RandomWalk, .. }
                if *rate < 0.05 || *rate > 0.80 => Severity::Error,
            Self::AcceptanceRateUnhealthy { rate, kernel: AcceptanceKernel::Nuts, .. }
                if *rate < 0.30 || *rate > 0.99 => Severity::Error,
            Self::AcceptanceRateUnhealthy { .. } => Severity::Warning,
            // gh#791: never an error, at any gradient. The threshold behind it
            // is a midpoint of the statistic's own range and not a calibrated
            // bar (see `cli::fit::path_renewal::COALESCENCE_GRADIENT`), so it
            // must not be able to stop a run — it is a pointer at the profile,
            // which is the actual diagnostic.
            Self::PathRenewalCoalesced { .. } => Severity::Warning,
            _ => Severity::Warning,
        }
    }

    pub fn render(&self) -> String {
        match self {
            // `{:.3}` on the threshold, not `{:.1}`: every value in the band
            // camdl cares about — 1.01, 1.05, 1.1 — rounds to "1.0" at one
            // decimal, so the reader could not tell which bar was applied.
            // Name the statistic too: R̂ has been the rank-normalized split
            // statistic of Vehtari et al. (2021) since gh#84, not the classic
            // Gelman & Rubin one, and the two disagree by a third on a
            // drifting-chain fit.
            Self::RhatHigh { param, rhat, threshold } =>
                format!("rank-normalized split R̂ for '{}' is {:.3} \
                         (threshold {:.3}). Chain estimates have not \
                         converged.", param, rhat, threshold),
            Self::ChainDiverged { chain_id, n_chains } =>
                format!("Chain {} of {} diverged from the others (MLE outside 3×MAD).",
                    chain_id, n_chains),
            Self::MultimodalLikelihood { ll_spread, max_chain_agreement } =>
                format!("Likelihood surface may be multimodal: \
                         loglik spread={:.1}, max Â={:.2}.", ll_spread, max_chain_agreement),
            Self::ConvergenceIncomplete { max_chain_agreement, n_unconverged, n_total } =>
                format!("{}/{} parameters have Â > 1.1 (max {:.2}).",
                    n_unconverged, n_total, max_chain_agreement),
            Self::LowESS { obs_time, ess, n_particles, .. } =>
                format!("ESS dropped to {:.0}/{} at t={:.0}.",
                    ess, n_particles, obs_time),
            Self::LowESSAtMLE { ess_mean, ess_min, n_particles } =>
                format!("ESS at MLE: mean={:.0}, min={:.0}/{}.",
                    ess_mean, ess_min, n_particles),
            Self::InitialLoglikInfinite =>
                "Initial log-likelihood is -inf at starting parameters.".into(),
            Self::BadInit { chain_id, params, reason, .. } => {
                let pretty = params.iter()
                    .map(|(k, v)| format!("{}={}", k, format_start_value(*v)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Chain {} starting parameters were pathological — skipped. \
                     Reason: {}. Init: [{}].",
                    chain_id, reason, pretty,
                )
            }
            Self::MaxTreeDepthHits { n_hits, n_sweeps, max_depth, .. } =>
                format!("{}/{} sweeps ({:.0}%) hit max tree depth {}.",
                    n_hits, n_sweeps,
                    *n_hits as f64 / *n_sweeps as f64 * 100.0, max_depth),
            Self::DivergentTransitions { n_divergent, n_sweeps } =>
                format!("{} divergent transitions in {} sweeps.",
                    n_divergent, n_sweeps),
            Self::DegenerateAncestorSampling { pct, .. } =>
                format!("Ancestor sampling degenerate at {:.1}% of substeps.", pct),
            Self::LowTrajectoryRenewal { renewal } =>
                format!("Trajectory renewal is {:.1}% — CSMC may not be mixing.",
                    renewal * 100.0),
            // Describes the shape and stops. Naming a mechanism here would be
            // false on the runs that fire with a healthy prefix — the polio
            // fixture fires at prefix 0.449 — and the contradiction would be
            // visible against the prefix printed in the same sentence.
            Self::PathRenewalCoalesced {
                prefix, gradient, first_bin, last_bin, aggregate, n_bins, n_prefix_bins,
            } =>
                format!(
                    "Trajectory renewal is concentrated at the end of the series: the \
                     first 1/{n_bins} of it renews in {:.1}% of sweeps and the last in \
                     {:.1}% (gradient {:.2}), while the aggregate reads {:.1}%. The mean \
                     over the first {n_prefix_bins} of {n_bins} bins is {:.1}%. The \
                     gradient reads only the two end bins, so it does not say which \
                     shape produced the rise: an early region flat and near zero is a \
                     coalesced conditional-SMC genealogy holding the early path at the \
                     reference, whereas a smooth monotone ramp is the ordinary finite \
                     coalescence depth of a long series. Read the profile.",
                    first_bin * 100.0, last_bin * 100.0, gradient, aggregate * 100.0,
                    prefix * 100.0),
            Self::FilterWeightCollapse { n_sweeps, n_total_sweeps, n_windows } =>
                format!("{}/{} sweeps had an observation window where every particle \
                         scored zero density ({} windows in total). The filter found no \
                         trajectory explaining the data at those parameters, and the \
                         sweep returned the reference.",
                    n_sweeps, n_total_sweeps, n_windows),
            Self::FilterStarved {
                n_starved, n_obs, n_particles, starved_below, worst_time, worst_mean, worst_min,
            } =>
                format!("At {}/{} observations the filter's mean ESS over the retained \
                         sweeps is below {:.0} of {} particles; the worst is t={:.0}: \
                         mean ESS {:.1}, minimum {:.1}. The resample there draws every \
                         particle from that handful, so the path through the \
                         observation is not being explored — and min_alive does not \
                         show it, because the other weights are finite.",
                    n_starved, n_obs, starved_below, n_particles,
                    worst_time, worst_mean, worst_min),
            Self::GammaDensityDisabled { reason } =>
                format!("Gamma density disabled: {}", reason),
            Self::AcceptanceRateUnhealthy { rate, param, kernel } => {
                let target = if param.is_some() { "parameter" } else { "chain" };
                format!("{} acceptance rate {:.1}% is outside healthy range {}.",
                    target, rate * 100.0, kernel.band_label())
            }
            Self::ParamNearBound { param, value, bound, bound_type } =>
                format!("'{}' = {:.4} is near {} bound {:.4}.",
                    param, value, bound_type, bound),
            Self::ProfileCIUnbounded { param, direction } =>
                format!("Profile CI for '{}' is unbounded {}.", param, direction),
            Self::FlatProfile { param, .. } =>
                format!("Profile for '{}' is flat — parameter may not be identifiable.",
                    param),
            Self::AutoRwSd { param, rw_sd } =>
                format!("Auto rw_sd for '{}': {:.6}.", param, rw_sd),
            Self::CompressedLogitPosition { param, z } =>
                format!("'{}' logit position |z|={:.1} > 2 — effective perturbation reduced.",
                    param, z.abs()),
            Self::AutoRwSdNoConsensus { n_good, n_total } =>
                format!("Auto rw_sd: only {}/{} chains agree — no consensus.",
                    n_good, n_total),
            Self::CoolingExhausted { exhausted_at_iter, total_iters, .. } =>
                format!("Cooling exhausted at iteration {}/{} — \
                         perturbations are near-zero for remaining iterations.",
                    exhausted_at_iter, total_iters),
            Self::ObsModelMismatch { obs_time, observed, predicted_mean, n_sigma } =>
                format!("Obs at t={:.0}: observed={:.0}, predicted={:.0} ({:.1}σ away).",
                    obs_time, observed, predicted_mean, n_sigma),
            Self::ZeroRateNonzeroFlow { transition, flow } =>
                format!("Transition '{}' has rate=0 but flow={}. Add iota.",
                    transition, flow),
            Self::LowSwapRate { beta_i, beta_j, rate, .. } =>
                format!("Tempering swap rate B={:.2}↔{:.2}: {:.1}%.",
                    beta_i, beta_j, rate * 100.0),
            Self::ResumeConfigMismatch { .. } =>
                "Resume config hash mismatch — model/data/priors changed.".into(),
            Self::ResumeParamMissing { param } =>
                format!("Parameter '{}' not found in resume state.", param),
        }
    }

    /// Advice to print under this finding, given what `ctx` says about the run
    /// it came from. A hint whose truth depends on the run and cannot be
    /// established from `ctx` is omitted.
    pub fn hints(&self, ctx: HintContext) -> Vec<&'static str> {
        match self {
            Self::LowESSAtMLE { .. } => vec![
                "Increase particles",
                "Estimate overdispersion (sigma_se) if fixed",
                "Check observation model matches data scale",
            ],
            Self::MultimodalLikelihood { .. } => vec![
                "Run more chains to sample both basins",
                "Set start values near the known basin",
                "Narrow parameter bounds to exclude the wrong basin",
            ],
            Self::InitialLoglikInfinite => vec![
                "Check starting values are within parameter bounds",
                "Run with --verbosity debug for per-substep diagnostics",
            ],
            Self::BadInit { .. } => {
                let mut hints = vec![
                    "Inspect chain_starts.tsv to see which init was used",
                    "If using survey_top_k, the survey may be putting \
                     bound-pinned points into the top-K; consider --init lhs",
                ];
                // Only when a chain actually finished. A run in which every
                // chain was refused printed this line once per refusal,
                // pointing at survivors that did not exist (gh#880).
                if ctx.chains_completed.is_some_and(|n| n >= 1) {
                    hints.push(
                        "Other chains in this run completed normally; treat \
                         the surviving chains as the result");
                }
                hints
            }
            Self::MaxTreeDepthHits { .. } => vec![
                "Increase max_treedepth in [pgas] config",
                "Consider reparameterizing correlated parameters",
            ],
            Self::DivergentTransitions { .. } => vec![
                "Reduce NUTS step size",
                "Reparameterize (e.g., non-centered parameterization)",
            ],
            Self::ZeroRateNonzeroFlow { .. } => vec![
                "Add a seeding term (iota) to the rate expression",
            ],
            Self::LowSwapRate { .. } => vec![
                "Add more temperature rungs (denser ladder)",
                "The LL gap between basins may be too large for tempering",
            ],
            Self::CompressedLogitPosition { .. } => vec![
                "Widen parameter bounds if scientifically justified",
                "Use a different transform (e.g., log instead of logit)",
            ],
            Self::PathRenewalCoalesced { .. } => vec![
                "Read the whole profile before concluding anything: \
                 `path_renewal.bins` in pgas_summary.json, or `renewal_b0 … \
                 renewal_b9` in each chain's trace.tsv, one column per tenth of \
                 the substep series. Flat and near zero across the early bins is \
                 a coalesced genealogy; a smooth monotone ramp is not",
                "If the early bins are flat and near zero, raise the particle \
                 count. On a matched probe that changed nothing else, four times \
                 the particles roughly tripled renewal over the early bins, so \
                 the frozen prefix there was particle-limited",
                "After changing the particle count, re-read the PROFILE and the \
                 AGGREGATE, not the gradient. Holding model, data and sweeps \
                 fixed while raising N 100 → 400 → 1600 on one model family moved \
                 the gradient 0.81 → 0.86 → 0.90 — the wrong way — while the \
                 aggregate improved. The gradient is a shape, not a progress bar",
                "Read `as_accept` beside it — it says whether the ancestor-sampling \
                 splice is contributing to renewal at all, or whether the profile \
                 is coming from the filter alone",
            ],
            Self::FilterWeightCollapse { .. } => vec![
                "Read collapsed_windows and min_alive in the chain's trace.tsv \
                 to see which sweeps searched and which did not",
                "A zero-density window is usually an observation the model \
                 cannot reach: check for a projection of exactly 0 scored \
                 against a positive count",
                "Increase particles so the swarm can reach the observation",
            ],
            Self::FilterStarved { .. } => vec![
                "Read filter_ess.tsv in the stage directory: one row per chain \
                 and observation, mean and minimum ESS over the retained sweeps. \
                 The trough names the data row",
                "Look at that observation in the data before touching the model: \
                 a count re-issued as zero, a revision, a stream switching \
                 definition mid-series. Starvation at one observation is usually \
                 a data point the model cannot reach, not a particle budget",
                "If the observation is real, the observation model is too tight \
                 there: an overdispersed or zero-inflated stream, or a wider \
                 reporting noise, gives the particles a density to survive on",
                "More particles help only when the ESS scales with them. Compare \
                 the mean ESS at that observation across two particle counts; if \
                 it stays at a handful, the swarm is not the limit",
            ],
            _ => vec![],
        }
    }
}

/// Accumulates diagnostics during an inference run.
/// Thread-safe via Mutex.
pub struct DiagnosticCollector {
    diagnostics: std::sync::Mutex<Vec<Diagnostic>>,
    stage: String,
}

impl DiagnosticCollector {
    pub fn new(stage: &str) -> Self {
        DiagnosticCollector {
            diagnostics: std::sync::Mutex::new(Vec::new()),
            stage: stage.into(),
        }
    }

    pub fn push(&self, kind: DiagnosticKind) {
        let severity = kind.severity();
        let message = kind.render();
        let diag = Diagnostic {
            kind,
            severity,
            message,
            stage: self.stage.clone(),
            timestamp: chrono_now(),
        };
        self.diagnostics.lock().unwrap().push(diag);
    }

    pub fn drain(&self) -> Vec<Diagnostic> {
        std::mem::take(&mut *self.diagnostics.lock().unwrap())
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.lock().unwrap().iter()
            .any(|d| d.severity == Severity::Error)
    }

    pub fn has_warnings(&self) -> bool {
        self.diagnostics.lock().unwrap().iter()
            .any(|d| d.severity != Severity::Info)
    }

    /// Render all diagnostics to stderr with ANSI coloring.
    ///
    /// `ctx` carries what the hints need to know about the run — the caller is
    /// the only one who knows how many chains finished — and
    /// [`HintContext::default`] is the honest answer when it does not.
    pub fn render_to_stderr(&self, ctx: HintContext) {
        let diags = self.diagnostics.lock().unwrap();
        if diags.is_empty() { return; }

        eprintln!("\n── diagnostics ──────────────────────────────────────");
        for d in diags.iter() {
            let icon = match d.severity {
                Severity::Info    => "\x1b[34mi\x1b[0m",
                Severity::Warning => "\x1b[33m!\x1b[0m",
                Severity::Error   => "\x1b[31mx\x1b[0m",
            };
            eprintln!("  {} {}", icon, d.message);
            for hint in d.kind.hints(ctx) {
                eprintln!("    -> {}", hint);
            }
        }
        let n_err = diags.iter().filter(|d| d.severity == Severity::Error).count();
        let n_warn = diags.iter().filter(|d| d.severity == Severity::Warning).count();
        let n_info = diags.iter().filter(|d| d.severity == Severity::Info).count();
        eprintln!("  {} error(s), {} warning(s), {} info", n_err, n_warn, n_info);
    }

    /// Write diagnostics to a JSON file.
    pub fn write_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(
            &*self.diagnostics.lock().unwrap()
        )?;
        std::fs::write(path, json)
    }
}

fn chrono_now() -> String {
    // ISO 8601 timestamp without a chrono dependency. The civil-date
    // arithmetic is the canonical proleptic-Gregorian one in `ir::caltime`
    // (`civil_from_unix_epoch_days`); only the HH:MM:SS split is local.
    // Im23 in 2026-04-19 inference review batch 3.
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let day_secs = secs % 86400;
    let hour = day_secs / 3600;
    let minute = (day_secs % 3600) / 60;
    let second = day_secs % 60;

    let (y, m, d) = ir::caltime::civil_from_unix_epoch_days((secs / 86400) as i64);

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m, d, hour, minute, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bad_init(param: &str, value: f64) -> DiagnosticKind {
        DiagnosticKind::BadInit {
            chain_id: 1,
            params: std::collections::BTreeMap::from([(param.to_string(), value)]),
            reason: "ESS collapsed".into(),
            attempts: Vec::new(),
        }
    }

    /// gh#880. The three chains of gh#876 started at 6.678e-5, 1.375e-4 and
    /// 5.922e-5 — a 33-41% spread — and `{:.4}` printed `mu=0.0001` for every
    /// one of them, and for the true value they were being compared against.
    /// The line exists to let a reader judge the start, so three different
    /// starts must read as three different numbers.
    #[test]
    fn three_nearby_small_rates_render_as_three_different_values() {
        let rendered: Vec<String> = [6.678e-5, 1.375e-4, 5.922e-5]
            .iter()
            .map(|&v| bad_init("mu", v).render())
            .collect();
        for (i, a) in rendered.iter().enumerate() {
            for b in rendered.iter().skip(i + 1) {
                assert_ne!(a, b,
                    "two chains that started at different rates must not \
                     render identically:\n  {a}\n  {b}");
            }
            assert!(!a.contains("mu=0.0001"),
                "a rate below 1e-3 must not collapse to four fixed \
                 decimals: {a}");
        }
        assert!(rendered[0].contains("mu=6.6780e-5"),
            "the value the chain ran from must be legible: {}", rendered[0]);
    }

    /// gh#781: the record's `chain_id` is the number the prose prints and the
    /// number `chain_starts.tsv`, `progress.json` and the `chain_N/`
    /// directories use. A renderer that adds one to the stored field would
    /// leave the JSON a reader parses one behind the sentence beside it.
    #[test]
    fn the_rendered_chain_number_is_the_recorded_one() {
        let d = DiagnosticKind::BadInit {
            chain_id: 4,
            params: std::collections::BTreeMap::from([("mu".to_string(), 0.3)]),
            reason: "ESS collapsed".into(),
            attempts: Vec::new(),
        };
        assert!(d.render().starts_with("Chain 4 "),
            "the prose must name the chain the record names: {}", d.render());
    }

    /// The same rule must leave ordinary-magnitude values alone — a
    /// probability or a fraction reads worse in scientific notation, and the
    /// only reason to move a value is that fixed decimals would hide it.
    #[test]
    fn an_ordinary_magnitude_start_keeps_four_fixed_decimals() {
        let m = bad_init("rho", 0.3).render();
        assert!(m.contains("rho=0.3000"),
            "0.3 must render as 0.3000, not in scientific notation: {m}");
        // Zero is not "below 1e-3" for this purpose: `0.0000e0` is noise.
        let m = bad_init("iota", 0.0).render();
        assert!(m.contains("iota=0.0000"), "zero renders plainly: {m}");
    }

    /// gh#880. In the run that motivated this, no chain completed and the
    /// hint told the reader three times to "treat the surviving chains as the
    /// result". The claim is about the run, not the finding, so it is made
    /// only when the caller says at least one chain finished.
    #[test]
    fn the_surviving_chains_hint_needs_a_surviving_chain() {
        let bad = bad_init("mu", 1e-4);
        let survivors = |h: &[&'static str]| {
            h.iter().any(|s| s.contains("surviving chains"))
        };
        assert!(!survivors(&bad.hints(HintContext { chains_completed: Some(0) })),
            "no chain completed — the hint must not claim survivors");
        assert!(!survivors(&bad.hints(HintContext::default())),
            "an unknown chain count is not evidence of a survivor");
        assert!(survivors(&bad.hints(HintContext { chains_completed: Some(1) })),
            "one chain completed — the hint is the right advice and must fire");
        // The advice that does not depend on the run is unconditional.
        for ctx in [HintContext::default(), HintContext { chains_completed: Some(0) }] {
            assert!(bad.hints(ctx).iter().any(|h| h.contains("chain_starts.tsv")),
                "the chain_starts.tsv pointer holds whatever the run did");
        }
    }

    /// The threshold in a `RhatHigh` message was formatted `{:.1}`, so every
    /// value in the band camdl actually cares about rendered the same: 1.01,
    /// 1.05 and 1.1 all printed as "1.0". A reader could not tell which bar
    /// the finding was drawn against, and "1.0" is not a threshold anything
    /// applies. The message must also name WHICH statistic — R̂ has been the
    /// rank-normalized split statistic of Vehtari et al. (2021) since gh#84,
    /// not the classic Gelman & Rubin one, and the two disagree by a third on
    /// a drifting-chain fit.
    #[test]
    fn the_rhat_finding_names_its_statistic_and_its_threshold_exactly() {
        let m = DiagnosticKind::RhatHigh {
            param: "beta".into(), rhat: 1.0295, threshold: 1.01,
        }.render();
        assert!(
            m.contains("1.01"),
            "the threshold applied must be legible, not rounded to 1.0: {m}"
        );
        assert!(
            !m.contains("threshold 1.0)") && !m.contains("threshold 1.0."),
            "and must not round a 1.01 bar to 1.0: {m}"
        );
        assert!(
            m.contains("rank-normalized"),
            "and must name which R̂ statistic it is: {m}"
        );
    }

    /// gh#631: the healthy band is kernel-specific. 0.83 acceptance is a
    /// well-tuned NUTS block (no error, in-band → this kind is not even
    /// constructed by the emitter; when constructed near the edges it warns)
    /// and simultaneously a broken random-walk chain (error).
    #[test]
    fn acceptance_severity_keys_on_kernel() {
        let nuts_ok = DiagnosticKind::AcceptanceRateUnhealthy {
            rate: 0.83, param: None, kernel: AcceptanceKernel::Nuts,
        };
        assert_eq!(nuts_ok.severity(), Severity::Warning,
            "0.83 under NUTS is at worst a warning, never an error");
        let rw_bad = DiagnosticKind::AcceptanceRateUnhealthy {
            rate: 0.83, param: None, kernel: AcceptanceKernel::RandomWalk,
        };
        assert_eq!(rw_bad.severity(), Severity::Error,
            "0.83 under random-walk MH is the >0.80 error band");
        let nuts_collapsed = DiagnosticKind::AcceptanceRateUnhealthy {
            rate: 0.004, param: None, kernel: AcceptanceKernel::Nuts,
        };
        assert_eq!(nuts_collapsed.severity(), Severity::Error,
            "0.4% under NUTS is the genuinely-stuck error band");
    }

    #[test]
    fn acceptance_message_names_the_kernel_band() {
        let m = DiagnosticKind::AcceptanceRateUnhealthy {
            rate: 0.83, param: Some("r_eff".into()), kernel: AcceptanceKernel::Nuts,
        }.render();
        assert!(m.contains("[60%, 95%]") && m.contains("NUTS"),
            "NUTS message names its own band: {m}");
        let m = DiagnosticKind::AcceptanceRateUnhealthy {
            rate: 0.83, param: None, kernel: AcceptanceKernel::RandomWalk,
        }.render();
        assert!(m.contains("[15%, 50%]"), "RW message keeps its band: {m}");
    }

    /// gh#299 item 3. The band a user is told about and the band the emitter
    /// applies must be the same band. They were not: the message read
    /// `[15%, 50%]` while both emitters compared against 0.10, so a random-walk
    /// chain accepting 12% was silently in-band while being told, if it ever
    /// tripped, that 15% was the floor.
    #[test]
    fn the_published_band_is_the_band_that_fires() {
        for kernel in [AcceptanceKernel::RandomWalk, AcceptanceKernel::Nuts] {
            let (lo, hi) = kernel.healthy_band();
            let msg = DiagnosticKind::AcceptanceRateUnhealthy {
                rate: lo - 0.01, param: None, kernel,
            }.render();
            assert!(msg.contains(&format!("[{:.0}%, {:.0}%]", lo * 100.0, hi * 100.0)),
                "{kernel:?}: message must name the band that fires: {msg}");
            // Just inside both ends: silent. Just outside: reported.
            assert!(acceptance_diagnostic(lo, None, kernel).is_none(),
                "{kernel:?}: the lower edge is healthy");
            assert!(acceptance_diagnostic(hi, None, kernel).is_none(),
                "{kernel:?}: the upper edge is healthy");
            assert!(acceptance_diagnostic(lo - 1e-6, None, kernel).is_some(),
                "{kernel:?}: below the lower edge must be reported");
            assert!(acceptance_diagnostic(hi + 1e-6, None, kernel).is_some(),
                "{kernel:?}: above the upper edge must be reported");
        }
        assert_eq!(AcceptanceKernel::RandomWalk.healthy_band(), (0.15, 0.50),
            "the random-walk band camdl publishes starts at 15%, not 10%");
    }

    /// The gh#299 item 3 regression proper: NUTS runs at 0.90-0.99 and targets
    /// ~0.8. Under the random-walk band every one of those is a finding —
    /// which is what buried the genuinely fatal 0.4%-stuck chain of gh#607 in
    /// forty identical-looking entries per run.
    #[test]
    fn a_well_tuned_nuts_block_draws_no_finding_but_a_random_walk_one_would() {
        for rate in [0.80, 0.87, 0.92, 0.95] {
            assert!(
                acceptance_diagnostic(rate, None, AcceptanceKernel::Nuts).is_none(),
                "{rate} is a healthy NUTS block and must not be reported"
            );
            assert!(
                acceptance_diagnostic(rate, None, AcceptanceKernel::RandomWalk).is_some(),
                "fixture premise: {rate} IS outside the random-walk band"
            );
        }
        // A collapsed NUTS kernel still lands, and still as an error.
        let stuck = acceptance_diagnostic(0.004, None, AcceptanceKernel::Nuts)
            .expect("0.4% under NUTS is a finding");
        assert_eq!(stuck.severity(), Severity::Error);
    }
}
