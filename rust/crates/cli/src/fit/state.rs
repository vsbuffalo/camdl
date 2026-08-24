//! `fit_state.toml` — inter-stage handoff file.

use serde::{Deserialize, Serialize};
// gh#519: `fit_state.toml` is a serialized artifact whose digest is recorded
// in `run.json`, so its byte layout has to be a function of its contents and
// nothing else. `HashMap`'s iteration order is seeded randomly PER PROCESS, so
// these three fields serialized in a different order on every run — two
// otherwise-identical fits produced files with the same numbers on shuffled
// lines, and therefore different digests. `BTreeMap` sorts by key, which is
// deterministic by construction rather than by an upstream insertion order
// that could itself come from a `HashMap`. Deserialization is unaffected: TOML
// tables are unordered on read, so existing `fit_state.toml` files load
// unchanged.
use std::collections::BTreeMap;

use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};

#[derive(Debug, Serialize, Deserialize)]
pub struct FitState {
    pub stage: String,
    pub seed: u64,
    pub timestamp: String,
    /// Input hash identifying the computation that produced this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_hash: Option<String>,
    /// camdl version that produced this state (e.g. "0.1.0+ce78a5e").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camdl_version: Option<String>,
    pub best_loglik: f64,
    pub initial_loglik: f64,
    pub best_chain: usize,
    pub n_chains: usize,
    pub n_good_chains: Option<usize>,
    pub start_values: BTreeMap<String, f64>,
    pub rw_sd: BTreeMap<String, f64>,
    /// What kind of log-likelihood is in `best_loglik` (gh#280). Serializes
    /// to the same `snake_case` tags this field carried as a free string
    /// before, so legacy `fit_state.toml` files deserialize unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loglik_type: Option<crate::fit::loglik::LoglikType>,
    /// Overall acceptance rate of the best chain (PGAS/PMMH only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_rate: Option<f64>,

    /// Per-parameter tail chain-agreement Â (last half of iterations).
    /// Populated at end-of-stage so downstream stages (notably refine)
    /// can gate on scout's convergence without re-running. Absent in
    /// legacy fit_state.toml files — downstream readers must treat
    /// absence as "unknown, proceed with warning," not "converged."
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tail_chain_agreement: BTreeMap<String, f64>,

    /// Names of estimated parameters declared `perturb_only_at_t0 = true`.
    /// Refine's Â check exempts these — an initial-state parameter is
    /// expected to be harder to identify and shouldn't block the pipeline
    /// when structural convergence is fine. Stored here (not re-derived
    /// from fit.toml in the downstream) so the two can't drift.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub perturb_only_at_t0_params: Vec<String>,

    /// Per-chain final log-likelihoods (the full distribution behind
    /// `best_loglik`). Refine's post-run loglik-regression gate uses
    /// the spread to compute its tolerance ε. Short vector; cheap to
    /// serialise.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_logliks: Vec<f64>,

    /// Per-chain CLEAN-EVAL log-likelihoods — the de-biased combined
    /// score for each chain's winning candidate, in chain-id order. New
    /// in proposal §Proposal 1 (Step 8); the compound scout-convergence
    /// gate uses these together with `chain_eval_ses` to compute an
    /// SE-aware decibans-spread threshold. Absent in pre-§Proposal 1
    /// fit_state files; the gate falls back to the chain-agreement-only
    /// check when this is empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_eval_logliks: Vec<f64>,

    /// Per-chain clean-eval standard errors, parallel to
    /// `chain_eval_logliks`. `max(SE)` drives the SE-aware decibans
    /// floor: noisier chains get proportionally more tolerance before
    /// the spread gate fires. New in §Proposal 1 (Step 8).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_eval_ses: Vec<f64>,

    /// Resolved compound-gate configuration as it was applied at the
    /// end of this stage. "Resolved" = the value that was actually in
    /// force at runtime, after the priority chain `CLI flag >
    /// fit.toml [stages.<stage>.gate] > GateConfig::default()`
    /// collapsed.
    ///
    /// Persisted so `camdl fit summary` can render the verdict line
    /// against the threshold the run was actually judged by — not
    /// against whatever `fit.toml` happens to say at summary-time
    /// (which may have been edited since the run) and not against
    /// `GateConfig::default()` (which may differ from the CLI override
    /// the user passed). Without this, summary's verdict label is
    /// silently a fiction whenever overrides were in play. See the
    /// 2026-04-25 fit-summary-command proposal §Phase 3.
    ///
    /// `None` on legacy fit_state.toml files written before this field
    /// existed — summary should render with a "(thresholds unknown)"
    /// caveat in that case rather than silently substituting defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_gate: Option<GateConfig>,

    /// Resolved clean-eval configuration as it was applied at the end
    /// of this stage. Same priority chain and persistence rationale as
    /// `resolved_gate` — what particle count and replicate count were
    /// actually used to compute the per-chain `chain_eval_logliks`
    /// and `chain_eval_ses` above. Without this, a reader can't
    /// reproduce the clean-eval exactly. See proposal §Phase 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_loglik_eval: Option<LoglikEvalConfig>,

    /// Provenance of this stage's chain starts (gh#51). One of:
    /// `single`, `uniform`, `lhs`, or `survey:<full-hash>:top-<K>`
    /// when `init = "survey_top_k"`. Surfaced as a one-line
    /// header in `camdl fit summary` ("seeded from: <source>") so
    /// the survey → fit linkage is visible without parsing
    /// `chain_starts.tsv`.
    ///
    /// `None` on legacy fit_state.toml files written before this
    /// field existed; summary renders such fits with `seeded from:
    /// unknown` rather than substituting a default that would
    /// silently misrepresent provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_init_source: Option<String>,

    /// Post-fit Richardson dt-convergence check at θ̂ (gh#52).
    /// `None` on legacy fit_state.toml files or stages where
    /// `dt_check.enabled = false`. `Some(_)` carries the full
    /// ladder + verdict + notes; `camdl fit summary` reads this and
    /// renders the verdict line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt_check: Option<crate::fit::dt_check::DtCheckResult>,
}

impl FitState {
    pub fn load(dir: &str) -> Result<Self, String> {
        let path = format!("{}/fit_state.toml", dir);
        let contents = std::fs::read_to_string(&path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        toml::from_str(&contents)
            .map_err(|e| format!("parse error in {}: {}", path, e))
    }

    pub fn save(&self, dir: &str) -> Result<(), String> {
        let path = format!("{}/fit_state.toml", dir);
        let body = toml::to_string_pretty(self)
            .map_err(|e| format!("serialize error: {}", e))?;
        let contents = format!("# Generated by {}\n{}", crate::version::VERSION, body);
        std::fs::write(&path, contents)
            .map_err(|e| format!("cannot write {}: {}", path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gh#519: `fit_state.toml`'s byte layout must be a function of its
    /// contents alone. It was not — `start_values` and `rw_sd` were
    /// `HashMap`s, whose iteration order is seeded randomly per process, so
    /// two otherwise-identical fits wrote the same numbers on shuffled lines
    /// and got different digests (which `run.json` records).
    ///
    /// Asserting "two serializations agree" would NOT catch a revert: within
    /// one process a `HashMap` iterates the same key set the same way
    /// regardless of insertion order. So assert the emitted keys are SORTED,
    /// which only a sorted map guarantees. Eight keys chosen so that a
    /// `HashMap` landing on sorted order by chance is a ~1-in-40000 event.
    #[test]
    fn serialized_parameter_keys_are_in_sorted_order() {
        let mut st = synthetic_state();
        st.start_values = BTreeMap::new();
        // Inserted in deliberately non-alphabetical order.
        for (k, v) in [("rho", 0.6), ("beta", 0.3), ("sigma_se", 0.05),
                       ("gamma", 0.15), ("k", 20.0), ("iota", 1.0),
                       ("psi", 0.4), ("N0", 1e6)] {
            st.start_values.insert(k.into(), v);
        }
        let body = toml::to_string_pretty(&st).expect("serialize");

        let keys: Vec<&str> = body.lines()
            .skip_while(|l| !l.starts_with("[start_values]"))
            .skip(1)
            .take_while(|l| !l.trim().is_empty() && !l.starts_with('['))
            .filter_map(|l| l.split('=').next().map(str::trim))
            .collect();
        assert_eq!(keys.len(), 8, "expected all 8 params in [start_values]: {keys:?}");

        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted,
            "fit_state.toml's parameter keys must serialize in sorted order — \
             an unordered map here makes the file, and the digest run.json \
             records for it, differ between identical runs (gh#519). Got: {keys:?}");
    }

    fn synthetic_state() -> FitState {
        let mut tail_chain_agreement = BTreeMap::new();
        tail_chain_agreement.insert("beta".into(), 1.02);
        tail_chain_agreement.insert("gamma".into(), 1.07);
        FitState {
            stage: "scout".into(),
            seed: 42,
            timestamp: "2026-04-24T00:00:00Z".into(),
            input_hash: Some("deadbeef".into()),
            camdl_version: Some("0.1.0+test".into()),
            best_loglik: -123.45,
            initial_loglik: -200.0,
            best_chain: 1,
            n_chains: 2,
            n_good_chains: Some(2),
            start_values: BTreeMap::from([("beta".into(), 0.8)]),
            rw_sd: BTreeMap::new(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement,
            perturb_only_at_t0_params: vec!["s0".into()],
            chain_logliks: vec![-130.0, -123.45],
            chain_eval_logliks: vec![-128.7, -123.1],
            chain_eval_ses: vec![1.5, 0.8],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            dt_check: None,
        }
    }

    /// fit_state.toml round-trips through save/load with the new
    /// Step-8/9 fields populated. Catches schema regressions where a
    /// rename or type change would break inter-stage handoff.
    #[test]
    fn fit_state_round_trip_with_clean_eval_fields() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_state_rt_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_string_lossy().into_owned();

        let s = synthetic_state();
        s.save(&dir_str).unwrap();
        let loaded = FitState::load(&dir_str).unwrap();

        assert_eq!(loaded.chain_eval_logliks, vec![-128.7, -123.1]);
        assert_eq!(loaded.chain_eval_ses, vec![1.5, 0.8]);
        assert_eq!(loaded.chain_logliks, vec![-130.0, -123.45]);
        assert_eq!(loaded.tail_chain_agreement.get("beta").copied(), Some(1.02));
        assert_eq!(loaded.perturb_only_at_t0_params, vec!["s0".to_string()]);
        // Phase 3: resolved gate / clean-eval persisted with the
        // verdict so summary can report against the threshold the
        // run was actually judged by, not whatever fit.toml says
        // at summary-time.
        let gate = loaded.resolved_gate.as_ref().expect("resolved_gate persisted");
        assert!((gate.a_thresh - 1.01).abs() < 1e-12);
        assert!((gate.decibans_thresh - 30.0).abs() < 1e-12);
        let ce = loaded.resolved_loglik_eval.as_ref()
            .expect("resolved_loglik_eval persisted");
        assert_eq!(ce.n_particles, 4000);
        assert_eq!(ce.n_replicates, 8);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Legacy fit_state.toml files written before Phase 3 lacked
    /// `resolved_gate` and `resolved_loglik_eval`. Loading must succeed
    /// and surface them as `None`, so summary can render with a
    /// "(thresholds unknown)" caveat instead of silently substituting
    /// defaults. This is the contract the proposal's "honest
    /// reporting over round-trip fidelity" choice rests on.
    #[test]
    fn fit_state_loads_legacy_file_with_no_resolved_config() {
        let dir = std::env::temp_dir().join(format!(
            "camdl_state_legacy_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fit_state.toml");
        // Legacy contents — fields that existed before Phase 3 (no
        // resolved_gate / resolved_loglik_eval block). Mirrors what a
        // pre-2026-04-25 fit_state.toml looked like.
        std::fs::write(&path, r#"
stage = "scout"
seed = 42
timestamp = "2026-04-20T00:00:00Z"
best_loglik = -123.45
initial_loglik = -200.0
best_chain = 1
n_chains = 2

[start_values]
beta = 0.8

[rw_sd]
"#).unwrap();
        let loaded = FitState::load(dir.to_str().unwrap()).unwrap();
        assert!(loaded.resolved_gate.is_none(),
            "legacy file must surface resolved_gate as None, not silently default");
        assert!(loaded.resolved_loglik_eval.is_none(),
            "legacy file must surface resolved_loglik_eval as None");
        std::fs::remove_dir_all(&dir).ok();
    }
}
