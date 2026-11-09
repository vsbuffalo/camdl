//! The typed record of what a fit method's run shows.
//!
//! Every fact the end of `fit run` prints is a value here before it is text,
//! and is recorded in the method leaf so `fit summary` can rebuild it from the
//! files (proposal `docs/dev/proposals/2026-11-08-cli-reporting.md`, "The
//! report"). This module holds the record types; each says where it is stored.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The run's opening block, as `fit run` prints it under `fit:`. Recorded in
/// `run.json`'s `inputs.header` (display inputs, never hashed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunHeader {
    /// The config file as given on the command line.
    pub config: String,
    /// The method's name (`pmmh`, `nl-sbplx`, …).
    pub method: String,
    /// The model path as the config writes it.
    pub model: String,
    /// Estimated parameters, in the config's order.
    pub estimate: Vec<String>,
    /// Fixed parameters, in the order the config resolves them.
    pub fixed: Vec<String>,
    /// The starts rule and why it applies, as the `starts:` line reads
    /// (`StartsResolution::describe`).
    pub starts: String,
    /// The fit directory, relative to the working directory the run was
    /// started from when it is under it ([`display_path`]).
    pub output: String,
}

/// A path for display: relative to the working directory when it is under
/// it, otherwise absolute. Relative is what a reader can paste back into a
/// command from where they stand; a path outside the working directory is
/// absolute so it cannot be read against the wrong base (gh#507).
pub fn display_path(p: &Path) -> String {
    let abs = crate::run_paths::display_absolute(p);
    match std::env::current_dir() {
        Ok(cwd) => match abs.strip_prefix(&cwd) {
            Ok(rel) if !rel.as_os_str().is_empty() => rel.display().to_string(),
            _ => abs.display().to_string(),
        },
        Err(_) => abs.display().to_string(),
    }
}

/// One chain's adaptive warm-up, as it ended. Recorded in
/// `pmmh_summary.json` (or `mh_summary.json`) under `warmup`; empty when
/// `adapt = false`, where nothing adapts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainWarmup {
    /// The chain's number, as its `chain_N/` directory carries it.
    pub chain: usize,
    /// Steps on which the Robbins–Monro scale adapted.
    pub warmup_steps: usize,
    /// Of those, steps on which it sat at its lower bound.
    pub steps_at_scale_floor: usize,
    /// The scale λ the kept draws were proposed under.
    #[serde(with = "sim::inference::json_f64")]
    pub final_scale: f64,
    /// The acceptance rate the adaptation chased.
    #[serde(with = "sim::inference::json_f64")]
    pub target_accept: f64,
}

impl ChainWarmup {
    /// The record of a chain resumed from `base`: the resumed call adapts only
    /// over whatever warm-up `base` had left, so its steps add to the base's
    /// and the scale it ends at is the one its draws were taken under.
    pub fn continued_from(base: &ChainWarmup, this_call: &ChainWarmup) -> ChainWarmup {
        ChainWarmup {
            chain: this_call.chain,
            warmup_steps: base.warmup_steps + this_call.warmup_steps,
            steps_at_scale_floor: base.steps_at_scale_floor + this_call.steps_at_scale_floor,
            final_scale: this_call.final_scale,
            target_accept: this_call.target_accept,
        }
    }
}

/// How a PMMH or `mh` method sampled. Recorded in its summary JSON under
/// `plan`; the same values the `pmmh:` line prints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingPlan {
    pub n_chains: usize,
    pub n_steps: usize,
    /// Particles per likelihood evaluation; 0 on the deterministic `mh` path.
    pub n_particles: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub adapt: bool,
    /// The initial proposal SD per estimated parameter, on the transformed
    /// scale.
    pub proposal_sd: BTreeMap<String, f64>,
}

/// What a PMMH or `mh` method's sampling cost. Recorded in its summary JSON
/// under `cost`; the values the `complete in` line prints.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingCost {
    /// Wall time of the chains, start of the first to end of the last.
    #[serde(with = "sim::inference::json_f64")]
    pub sampling_seconds: f64,
    /// Likelihood evaluations, as the completion line counts them.
    pub evaluations: usize,
    /// Wall time per evaluation within one chain, the chains running in
    /// parallel.
    #[serde(with = "sim::inference::json_f64")]
    pub ms_per_evaluation: f64,
}

/// The `warmup` record a method's summary JSON holds, or empty when the file
/// or the field is absent (a leaf written before it was recorded).
pub fn read_warmup(summary_path: &Path) -> Vec<ChainWarmup> {
    #[derive(Deserialize)]
    struct WithWarmup {
        #[serde(default)]
        warmup: Vec<ChainWarmup>,
    }
    std::fs::read_to_string(summary_path)
        .ok()
        .and_then(|t| serde_json::from_str::<WithWarmup>(&t).ok())
        .map(|w| w.warmup)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_resumed_chain_adds_its_warmup_to_the_base() {
        let base = ChainWarmup {
            chain: 2, warmup_steps: 600, steps_at_scale_floor: 40,
            final_scale: 0.5, target_accept: 0.337,
        };
        let resumed = ChainWarmup {
            chain: 2, warmup_steps: 400, steps_at_scale_floor: 10,
            final_scale: 0.4, target_accept: 0.337,
        };
        let c = ChainWarmup::continued_from(&base, &resumed);
        assert_eq!((c.warmup_steps, c.steps_at_scale_floor), (1000, 50));
        assert_eq!(c.final_scale, 0.4);
    }

    #[test]
    fn a_summary_without_a_warmup_record_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("pmmh_summary.json");
        std::fs::write(&p, r#"{"method": "pmmh"}"#).unwrap();
        assert!(read_warmup(&p).is_empty());
        assert!(read_warmup(&dir.path().join("absent.json")).is_empty());
    }
}
