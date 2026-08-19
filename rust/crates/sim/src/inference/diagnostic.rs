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

impl DiagnosticKind {
    pub fn severity(&self) -> Severity {
        match self {
            Self::InitialLoglikInfinite => Severity::Error,
            Self::BadInit { .. } => Severity::Error,
            Self::RhatHigh { rhat, .. } if *rhat > 1.5 => Severity::Error,
            Self::RhatHigh { .. } => Severity::Warning,
            Self::ConvergenceIncomplete { max_chain_agreement, .. } if *max_chain_agreement > 1.5 => Severity::Error,
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
            Self::RhatHigh { param, rhat, threshold } =>
                format!("Rhat for '{}' is {:.3} (threshold {:.1}). \
                         Chain estimates have not converged.", param, rhat, threshold),
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
            Self::GammaDensityDisabled { reason } =>
                format!("Gamma density disabled: {}", reason),
            Self::AcceptanceRateUnhealthy { rate, param, kernel } => {
                let target = if param.is_some() { "parameter" } else { "chain" };
                let band = match kernel {
                    AcceptanceKernel::RandomWalk => "[15%, 50%] (random-walk MH)",
                    AcceptanceKernel::Nuts => "[60%, 95%] (NUTS block; ~80% is the target)",
                };
                format!("{} acceptance rate {:.1}% is outside healthy range {}.",
                    target, rate * 100.0, band)
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
}
