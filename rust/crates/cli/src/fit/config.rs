//! `fit.toml` parsing and validation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Top-level fit.toml structure.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitToml {
    pub fit: FitSection,
    pub data: HashMap<String, String>,
    /// Holdout data for out-of-sample validation. Keys must match [data] keys.
    /// Scout/refine never see holdout data. Validate runs PF on train + holdout
    /// and reports separate logliks.
    pub holdout: Option<HashMap<String, String>>,
    pub config: FitConfigSection,
    pub estimate: HashMap<String, EstimateSpec>,
    pub fixed: HashMap<String, toml::Value>,
    /// Optional per-stage configuration. Omitted sections use defaults.
    pub scout: Option<StageConfig>,
    pub refine: Option<StageConfig>,
    pub validate: Option<ValidateConfig>,
    /// PMMH posterior sampling configuration.
    pub pmmh: Option<PMMHSampleConfig>,
    /// PGAS posterior sampling configuration.
    pub pgas: Option<PGASSampleConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitSection {
    pub model: String,
    pub output_dir: String,
    /// RNG seed. CLI --seed overrides this.
    pub seed: Option<u64>,
}

/// Per-stage configuration for scout and refine.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageConfig {
    pub chains: Option<usize>,
    pub particles: Option<usize>,
    pub iterations: Option<usize>,
    pub cooling: Option<f64>,
    /// Multiply all rw_sd values by this factor. Default 1.0.
    pub rw_sd_scale: Option<f64>,
    /// Number of chains seeded near start values (rest are random).
    /// Default 1 when any parameter has start, 0 otherwise.
    pub start_chains: Option<usize>,
}

/// Validate stage configuration (includes pfilter settings).
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ValidateConfig {
    pub chains: Option<usize>,
    pub particles: Option<usize>,
    pub iterations: Option<usize>,
    pub cooling: Option<f64>,
    pub rw_sd_scale: Option<f64>,
    /// Particle count for the final precise pfilter at the MLE.
    pub pfilter_particles: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfigSection {
    pub backend: String,
    pub dt: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EstimateSpec {
    /// Random walk SD on natural scale. If omitted, auto-computed from bounds.
    pub rw_sd: Option<f64>,
    #[serde(default)]
    pub ivp: bool,
    pub start: Option<f64>,
    pub transform: Option<String>,
    pub bounds: Option<(f64, f64)>,
    /// Prior distribution string. Supported:
    ///   "lognormal(mu, sigma)" → TransformedNormal on log scale
    ///   "normal(mu, sigma)"    → Normal on natural scale
    ///   omitted                → Flat (improper uniform)
    pub prior: Option<String>,
}

/// PMMH posterior sampling configuration.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PMMHSampleConfig {
    pub chains: Option<usize>,
    pub steps: Option<usize>,
    pub particles: Option<usize>,
    pub burn_in: Option<usize>,
    pub thin: Option<usize>,
    pub adapt: Option<bool>,
    pub adapt_start: Option<usize>,
    /// Directory containing fit_state.toml from a prior IF2 run.
    /// Used to seed proposal covariance from scout chain spread.
    pub proposal_from: Option<String>,
    /// Crank-Nicolson correlation for correlated pseudo-marginal MCMC.
    /// Default: 0.99. Set to None or 0.0 for vanilla (independent) PMMH.
    pub rho: Option<f64>,
}

/// PGAS (Particle Gibbs with Ancestor Sampling) configuration.
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PGASSampleConfig {
    pub chains: Option<usize>,
    pub sweeps: Option<usize>,
    pub particles: Option<usize>,
    pub burn_in: Option<usize>,
    pub thin: Option<usize>,
    /// Directory containing fit_state.toml from a prior stage.
    pub starts_from: Option<String>,
    /// Number of posterior trajectory samples to save per chain.
    /// Evenly spaced across post-burn-in sweeps. Default: 200.
    /// Set to 0 to disable trajectory output.
    pub n_trajectories: Option<usize>,
    /// Temperature ladder for parallel tempering (replica exchange).
    /// Each entry is a β value. First entry must be 1.0 (cold chain).
    /// Example: `[1.0, 0.7, 0.4, 0.15]` runs 4 rungs per chain.
    /// Default: no tempering (single rung).
    pub tempering: Option<Vec<f64>>,
}

impl FitToml {
    pub fn load(path: &str) -> Result<Self, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        let fit: FitToml = toml::from_str(&contents)
            .map_err(|e| format!("parse error in {}: {}", path, e))?;

        // Validate holdout keys match data keys
        if let Some(ref holdout) = fit.holdout {
            for key in holdout.keys() {
                if !fit.data.contains_key(key) {
                    return Err(format!(
                        "[holdout] key '{}' does not match any [data] key.\n\
                         Available data streams: {}",
                        key, fit.data.keys().cloned().collect::<Vec<_>>().join(", ")
                    ));
                }
            }
        }

        Ok(fit)
    }

    /// Exhaustive partition check: every model parameter must be in [estimate] or [fixed].
    pub fn validate_partition(&self, model_params: &[String]) -> Result<(), String> {
        let estimated: std::collections::HashSet<&str> =
            self.estimate.keys().map(|s| s.as_str()).collect();
        let fixed: std::collections::HashSet<&str> =
            self.fixed.keys().map(|s| s.as_str()).collect();

        // Check overlap
        let overlap: Vec<&str> = estimated.intersection(&fixed).copied().collect();
        if !overlap.is_empty() {
            return Err(format!(
                "Parameters in both [estimate] and [fixed]: {}\n\
                 Each parameter must appear in exactly one section.",
                overlap.join(", ")
            ));
        }

        // Check missing
        let missing: Vec<&str> = model_params.iter()
            .map(|s| s.as_str())
            .filter(|name| !estimated.contains(name) && !fixed.contains(name))
            .collect();
        if !missing.is_empty() {
            let suggestions: Vec<String> = missing.iter().map(|name| {
                format!("  [estimate]: {} = {{ rw_sd = 0.01 }}\n  [fixed]:    {} = true", name, name)
            }).collect();
            return Err(format!(
                "Parameters not assigned in fit.toml: {}\n\
                 Every model parameter must appear in [estimate] or [fixed].\n\n{}",
                missing.join(", "),
                suggestions.join("\n")
            ));
        }

        // Check extra (in fit.toml but not in model)
        let model_set: std::collections::HashSet<&str> =
            model_params.iter().map(|s| s.as_str()).collect();
        let extra_est: Vec<&str> = estimated.iter()
            .filter(|n| !model_set.contains(**n)).copied().collect();
        let extra_fix: Vec<&str> = fixed.iter()
            .filter(|n| !model_set.contains(**n)).copied().collect();
        if !extra_est.is_empty() || !extra_fix.is_empty() {
            let mut extra = extra_est;
            extra.extend(extra_fix);
            return Err(format!(
                "Parameters in fit.toml but not in model: {}\n\
                 Check for typos.",
                extra.join(", ")
            ));
        }

        Ok(())
    }

    /// Check that fit bounds are within model bounds.
    pub fn validate_bounds(&self, model: &ir::Model) -> Result<(), String> {
        for (name, spec) in &self.estimate {
            if let Some((fit_lo, fit_hi)) = spec.bounds {
                if let Some(ir_param) = model.parameters.iter().find(|p| p.name == *name) {
                    if let Some((model_lo, model_hi)) = ir_param.bounds {
                        if fit_lo < model_lo || fit_hi > model_hi {
                            return Err(format!(
                                "fit bound [{}, {}] for '{}' extends beyond model bound [{}, {}].\n\
                                 Fit search bounds must be within model structural bounds.",
                                fit_lo, fit_hi, name, model_lo, model_hi
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
