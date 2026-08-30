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
        .then(|| DiagnosticKind::AcceptanceRateUnhealthy { rate, param, kernel })
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
        chain_id: usize,
        /// Estimated parameter name → starting value on the natural
        /// scale, exactly as offered to the inference engine.
        params: std::collections::BTreeMap<String, f64>,
        /// One-line cause from the upstream PFDegenerateKind /
        /// fallback message. Surface in the diagnostic so the user
        /// can correlate with chain_starts.tsv.
        reason: String,
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
            Self::BadInit { chain_id, params, reason } => {
                let pretty = params.iter()
                    .map(|(k, v)| format!("{}={:.4}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Chain {} starting parameters were pathological — skipped. \
                     Reason: {}. Init: [{}].",
                    chain_id + 1, reason, pretty,
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
            Self::FilterWeightCollapse { n_sweeps, n_total_sweeps, n_windows } =>
                format!("{}/{} sweeps had an observation window where every particle \
                         scored zero density ({} windows in total). The filter found no \
                         trajectory explaining the data at those parameters, and the \
                         sweep returned the reference.",
                    n_sweeps, n_total_sweeps, n_windows),
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

    pub fn hints(&self) -> Vec<&'static str> {
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
            Self::BadInit { .. } => vec![
                "Inspect chain_starts.tsv to see which init was used",
                "If using survey_top_k, the survey may be putting \
                 bound-pinned points into the top-K; consider --init lhs",
                "Other chains in this run completed normally; treat \
                 the surviving chains as the result",
            ],
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
            Self::FilterWeightCollapse { .. } => vec![
                "Read collapsed_windows and min_alive in the chain's trace.tsv \
                 to see which sweeps searched and which did not",
                "A zero-density window is usually an observation the model \
                 cannot reach: check for a projection of exactly 0 scored \
                 against a positive count",
                "Increase particles so the swarm can reach the observation",
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
    pub fn render_to_stderr(&self) {
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
            for hint in d.kind.hints() {
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
