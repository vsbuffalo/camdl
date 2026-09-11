//! `fit_state.toml` — inter-method handoff file.

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
    /// The inference method that wrote this leaf (`if2`, `pgas`,
    /// `pmmh`, `mh`, `nuts`, `nl-sbplx`, `nl-bobyqa`) — the label of the
    /// `method` store level the file sits under.
    pub method: String,
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
    /// Populated when the method finishes, so a downstream method
    /// (notably refine) can gate on scout's convergence without re-running. Absent in
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

    /// The chain ids `chain_eval_logliks` and `chain_eval_ses` belong to,
    /// parallel to both. Written because those two are built by
    /// `ChainResults::chain_eval_logliks`, which sorts by chain id and then
    /// DISCARDS it — and IF2 drops PF-degenerate chains before that point, so
    /// a position in the vector is not a chain number. Without this the
    /// summary's per-chain table labelled row `i` as "chain i + 1", which put
    /// the `<- selected` marker on the wrong chain whenever any chain had been
    /// dropped. Empty in fit_state files written before this field existed;
    /// the table is omitted rather than mislabelled when it is missing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_eval_ids: Vec<usize>,

    /// Per-chain clean-eval standard errors, parallel to
    /// `chain_eval_logliks`. `max(SE)` drives the SE-aware decibans
    /// floor: noisier chains get proportionally more tolerance before
    /// the spread gate fires. New in §Proposal 1 (Step 8).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_eval_ses: Vec<f64>,

    /// Resolved compound-gate configuration as it was applied at the
    /// end of this method's run. "Resolved" = the value that was actually
    /// in force at runtime, after the priority chain `CLI flag >
    /// fit.toml [method.gate] > GateConfig::default()` collapsed.
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
    /// of this method's run. Same priority chain and persistence rationale as
    /// `resolved_gate` — what particle count and replicate count were
    /// actually used to compute the per-chain `chain_eval_logliks`
    /// and `chain_eval_ses` above. Without this, a reader can't
    /// reproduce the clean-eval exactly. See proposal §Phase 3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_loglik_eval: Option<LoglikEvalConfig>,

    /// The `starts` rule that supplied this leaf's chain starts, as written
    /// with its source (`from_prior`, `from_mle @scout`,
    /// `from_params theta.toml`). Surfaced as the one-line `seeded from`
    /// header in `camdl fit summary`, so where the chains began is visible
    /// without parsing `chain_starts.tsv`.
    ///
    /// `None` on a fit_state.toml written before this field existed; summary
    /// renders such fits with `seeded from: unknown` rather than substituting
    /// a default that would silently misrepresent provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_init_source: Option<String>,

    /// Whether that rule gave each chain its own start (`spread`) or put every
    /// chain at one point (`point`) — the fact the between-chain R̂ needs
    /// (proposal 2026-09-08, §3.4). A multi-chain sampler leaf whose kind is
    /// `point` is read with R̂ not assessed for every parameter, never as a
    /// pass. `None` on a fit_state.toml written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_starts_kind: Option<crate::fit::starts::ChainStartsKind>,

    /// Post-fit Richardson dt-convergence check at θ̂ (gh#52).
    /// `None` on legacy fit_state.toml files or methods where
    /// `dt_check.enabled = false`. `Some(_)` carries the full
    /// ladder + verdict + notes; `camdl fit summary` reads this and
    /// renders the verdict line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dt_check: Option<crate::fit::dt_check::DtCheckResult>,

    /// Measured likelihood noise at the base θ for a pseudo-marginal method
    /// (gh#764): the spread of a single `log L̂`, the spread of the difference
    /// that enters the Metropolis ratio, and the particle and pair counts they
    /// were measured at. `σ` scales as `1/√N`, so a spread stored without its
    /// particle count is not a number anyone can act on.
    ///
    /// This is the one number that decides whether a pseudo-marginal chain can
    /// reach its target acceptance rate at all. It was computed at preflight,
    /// printed to stderr and dropped, so diagnosing a stuck chain meant
    /// re-running an expensive fit to read a value the first run had already
    /// produced.
    ///
    /// `None` on IF2/PGAS/NUTS runs, on the deterministic `mh` method (an ODE
    /// likelihood has no filter noise), on any run whose base θ or proposed
    /// θ' the filter ruled out, and on `fit_state.toml` files written before
    /// this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pf_noise: Option<crate::fit::pf_noise::PfNoiseCheck>,
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
            method: "scout".into(),
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
            chain_eval_ids: vec![1, 2],
            chain_eval_ses: vec![1.5, 0.8],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            chain_starts_kind: Some(crate::fit::starts::ChainStartsKind::Spread),
            dt_check: None,
            pf_noise: None,
        }
    }

    /// fit_state.toml round-trips through save/load with the new
    /// Step-8/9 fields populated. Catches schema regressions where a
    /// rename or type change would break the inter-method handoff.
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
method = "scout"
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

    /// The kind round-trips as its snake_case name and is absent on a state
    /// that never had it, so a pre-split leaf reads back as `None` — not as
    /// a guessed `spread`.
    #[test]
    fn chain_starts_kind_round_trips_and_is_optional() {
        let mut state = synthetic_state();
        state.chain_starts_kind = Some(crate::fit::starts::ChainStartsKind::Point);
        let body = toml::to_string_pretty(&state).unwrap();
        assert!(body.contains("chain_starts_kind = \"point\""), "{body}");
        let back: FitState = toml::from_str(&body).unwrap();
        assert_eq!(back.chain_starts_kind, Some(crate::fit::starts::ChainStartsKind::Point));

        let legacy = body.replace("chain_starts_kind = \"point\"\n", "");
        let back: FitState = toml::from_str(&legacy).unwrap();
        assert_eq!(back.chain_starts_kind, None, "a leaf without the field says nothing");
    }

    /// gh#901: the file names the inference method under the key the rest of
    /// the tool uses. `fit_state.toml` is the inter-method handoff file — a
    /// `from_mle` start reads it — and it spelled the method `stage` after the
    /// store levels, the config and the CLI had all moved to `method`. The
    /// assertion is two-sided: the new key is emitted *and* the old one is
    /// absent, so the rename cannot be half-reverted.
    #[test]
    fn fit_state_names_the_method_and_never_a_stage() {
        let body = toml::to_string_pretty(&synthetic_state()).expect("serialize");
        assert!(body.contains("method = \"scout\""),
            "fit_state.toml must name the method under `method`:\n{body}");
        assert!(!body.contains("stage = "),
            "and must not spell it `stage`, which no other surface does:\n{body}");
        // The reader is the other half of the handoff: a file written with the
        // new key must load, and one written with the old key must not quietly
        // parse into a default.
        let back: FitState = toml::from_str(&body).expect("round-trips");
        assert_eq!(back.method, "scout");
        let old = body.replace("method = \"scout\"", "stage = \"scout\"");
        assert!(toml::from_str::<FitState>(&old).is_err(),
            "a file spelling the key `stage` is not read (no compatibility \
             alias, VERSIONING.md alpha posture)");
    }
}
