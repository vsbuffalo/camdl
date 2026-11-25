//! `fit_state.toml` — a fit method's result record.
//!
//! Written once per method leaf when the method finishes, opening with its
//! schema tag ([`FIT_STATE_SCHEMA`]). It is read in one of two ways.
//! [`FitState::load_input`] reads it as the input to a new computation: a
//! `from_mle` start in `fit run` or `camdl profile`, the convergence check on
//! a warm-start source (`gating.rs`), a resumed PGAS chain's base, and the
//! runner's read-back of the record it wrote. It refuses a file of any other
//! schema. [`FitState::load_for_inspection`] reads it for display: the typed
//! method results (`method_result.rs`: `If2Run`, `PmmhRun`, …) that
//! `fit summary`, `fit table` and `predict` are built from, and the
//! method report. It reads a file of another schema with a notice.

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

use score::convergence::ChainIndex;
use crate::fit::config_v2::{LoglikEvalConfig, GateConfig};

#[derive(Debug, Serialize, Deserialize)]
pub struct FitState {
    /// The inference method that wrote this leaf (`if2`, `pgas`,
    /// `pmmh`, `mh`, `nuts`, `nl-sbplx`, `nl-bobyqa`) — the label of the
    /// `method` store level the file sits under.
    pub method: String,
    pub seed: u64,
    /// Input hash identifying the computation that produced this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_hash: Option<String>,
    /// camdl version that produced this state (e.g. "0.1.0+ce78a5e").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camdl_version: Option<String>,
    pub best_loglik: f64,
    pub initial_loglik: f64,
    /// The chain that produced `best_loglik`: the number of the `chain_N/`
    /// directory beside this file, the number the method's own stderr line
    /// prints, and the number `mle_params.toml`'s `[provenance] chain`
    /// carries — one number, counted from 0 like every chain index (gh#990),
    /// so `chain_<best_chain>/` opens with no arithmetic.
    pub best_chain: ChainIndex,
    pub n_chains: usize,
    pub n_good_chains: Option<usize>,
    pub start_values: BTreeMap<String, f64>,
    /// What kind of log-likelihood is in `best_loglik` (gh#280). Serializes
    /// to the same `snake_case` tags this field carried as a free string
    /// before, so legacy `fit_state.toml` files deserialize unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loglik_type: Option<crate::fit::loglik::LoglikType>,
    /// Overall acceptance rate of the best chain (PGAS/PMMH only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acceptance_rate: Option<f64>,

    /// Per-parameter tail chain-agreement Â (last half of iterations).
    /// Read by the convergence check on a starts source (`gating.rs`) and by
    /// `If2Run`. Absent in old fit_state.toml files — readers must treat
    /// absence as "unknown", not "converged."
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tail_chain_agreement: BTreeMap<String, f64>,

    /// Names of estimated parameters declared `perturb_only_at_t0 = true`.
    /// Their Â is gated like every other estimate's; the names let a
    /// renderer report the dynamics-only sub-verdict beside the overall one.
    /// Stored here (not re-derived from fit.toml in the downstream) so the
    /// two can't drift.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub initial_state_params: Vec<String>,

    /// Per-chain final log-likelihoods (the full distribution behind
    /// `best_loglik`). The convergence check on a starts source reads their
    /// spread (`gating.rs`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_logliks: Vec<f64>,

    /// Per-chain CLEAN-EVAL log-likelihoods — the de-biased combined
    /// score for each chain's winning candidate, in chain-id order. The
    /// compound IF2 convergence check uses these together with
    /// `chain_eval_ses` to compute an SE-aware decibans-spread threshold;
    /// with this empty it falls back to the chain-agreement-only check.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_eval_logliks: Vec<f64>,

    /// The chains `chain_eval_logliks` and `chain_eval_ses` belong to,
    /// parallel to both — the same numbers as `best_chain` and the `chain_N/`
    /// directories. Written because those two are built by
    /// `ChainResults::chain_eval_logliks`, which sorts by chain id and then
    /// DISCARDS it — and IF2 drops PF-degenerate chains before that point, so
    /// a position in the vector is not a chain number. Without this the
    /// summary's per-chain table labelled rows by position, which put the
    /// `<- selected` marker on the wrong chain whenever any chain had been
    /// dropped. Empty in fit_state files written before this field existed;
    /// the table is omitted rather than mislabelled when it is missing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chain_eval_ids: Vec<ChainIndex>,

    /// Per-chain clean-eval standard errors, parallel to
    /// `chain_eval_logliks`. `max(SE)` drives the SE-aware decibans
    /// floor: noisier chains get proportionally more tolerance before
    /// the spread check fails.
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

    /// An nl-* optimizer's own convergence verdict, as it judged its chains
    /// at the end of the run (`nlopt_method::check_convergence`). The verdict
    /// needs per-chain values and the optimizer tolerance, which the rest of
    /// this file does not keep, so it is recorded rather than re-derived; the
    /// warm-start source check and `fit table` read it. `None` on every other
    /// method, and on nl-* leaves written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub optimizer_convergence: Option<OptimizerConvergence>,

    /// IF2: each chain's clean-evaluation log-likelihood compared with the
    /// other chains' over its standard error (gh#406, gh#1008), computed once
    /// at the end of the run from `chain_eval_ids`, `chain_eval_logliks` and
    /// `chain_eval_ses`; the warm-start source check reads it. Last, so its
    /// tables follow every plain value in the file. `None` on every other
    /// method, on a run whose clean evaluations name no chains, and on a leaf
    /// written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_loglik_agreement: Option<crate::fit::chain_diagnostics::ChainLoglikAgreement>,
}

/// The two-leg verdict an nl-* optimizer reaches on its chains: the chain
/// agreement leg fails when some parameter's spread across chains is both
/// wide relative to its bounds and above the optimizer's numerical floor; the
/// decibans leg fails when the converged chains' log-likelihoods spread past
/// `gate.decibans_thresh`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptimizerConvergence {
    /// Chains that ended `Success` / `XtolReached` / `FtolReached`.
    pub n_converged: usize,
    /// Chains that ran out of evaluations (soft failure).
    pub n_maxeval: usize,
    /// Chains that failed outright.
    pub n_failed: usize,
    /// Largest per-parameter range across chains over the bound width.
    pub max_rel_range: f64,
    /// Largest per-parameter range across chains, natural units.
    pub max_abs_range: f64,
    pub chain_agreement_failed: bool,
    /// `max − min` log-likelihood across converged chains, nats.
    pub delta_loglik: f64,
    pub decibans_failed: bool,
}

impl OptimizerConvergence {
    /// Every chain converged, and both legs pass.
    pub fn passes(&self) -> bool {
        self.all_chains_converged() && !self.chain_agreement_failed && !self.decibans_failed
    }

    /// No chain ran out of evaluations or failed.
    pub fn all_chains_converged(&self) -> bool {
        self.n_maxeval == 0 && self.n_failed == 0
    }

    /// The verdict as `fit table`'s `gate_verdict` column spells it.
    pub fn gate_verdict(&self) -> &'static str {
        if !self.all_chains_converged() {
            return "fail_status";
        }
        // One converged chain: neither leg had two chains to compare.
        if self.n_converged < 2 {
            return "not_assessed";
        }
        match (self.chain_agreement_failed, self.decibans_failed) {
            (false, false) => "pass",
            (true, false) => "fail_chain_agreement",
            (false, true) => "fail_db",
            (true, true) => "fail_both",
        }
    }
}

/// The file a run's [`FitState`] is stored in.
pub(crate) const FIT_STATE_FILE: &str = "fit_state.toml";

/// The schema tag of the `fit_state.toml` this build writes: the value of the
/// file's top-level `schema` key. A file without the key was written before
/// the tag existed.
pub(crate) const FIT_STATE_SCHEMA: &str = "camdl.fit-state/1";

/// Keys a `fit_state.toml` without a schema tag may carry that this build
/// does not read under that name: `rw_sd` is not read at all, and
/// `perturb_only_at_t0_params` is the initial-state list's old key.
const LEGACY_KEYS: [&str; 2] = ["rw_sd", "perturb_only_at_t0_params"];

/// A `fit_state.toml` read for display, with the notice owed to its reader
/// when its schema is not the one this build writes.
#[derive(Debug)]
pub struct Inspected {
    pub state: FitState,
    /// `Some` when the file carries no schema tag or a tag other than
    /// [`FIT_STATE_SCHEMA`]: one line naming the build that wrote it, its
    /// tag, and the keys this build does not read under their own name.
    pub notice: Option<crate::out::notice::Notice>,
}

impl Inspected {
    /// The state, after saying its notice on stderr at most once per file in
    /// this process. `fit summary` reads one run's state through several
    /// typed results, and its reader is told once.
    pub fn said(self) -> FitState {
        static SAID: std::sync::Mutex<std::collections::BTreeSet<String>> =
            std::sync::Mutex::new(std::collections::BTreeSet::new());
        if let Some(n) = &self.notice {
            let first = SAID.lock().unwrap_or_else(|e| e.into_inner()).insert(n.message.clone());
            if first {
                crate::out::detached().notice(n);
            }
        }
        self.state
    }
}

/// `<dir>/fit_state.toml` parsed as a TOML table, with its path for messages.
/// A run whose chain directories count from 1 is refused first: its
/// `best_chain` and `chain_eval_ids` count from 1 too, and reading them as
/// counted from 0 would name the wrong chain everywhere the file is rendered.
#[expect(clippy::disallowed_methods, reason = "reads `fit_state.toml` from a method directory: the folder of the run this process's `fit run` is writing, read back, or a `fit_store::RunFiles` copy of a stored run read out through the store")]
fn read_table(dir: &str) -> Result<(String, toml::Table), String> {
    crate::chain_index::refuse_one_based_layout(std::path::Path::new(dir))?;
    let path = format!("{dir}/{FIT_STATE_FILE}");
    let contents = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let table: toml::Table = toml::from_str(&contents)
        .map_err(|e| format!("parse error in {}: {}", path, e))?;
    Ok((path, table))
}

/// The file's schema tag, or `None` when it has none.
fn schema_of(table: &toml::Table) -> Option<&str> {
    table.get("schema").and_then(|v| v.as_str())
}

/// A schema tag as a message names it.
fn describe_schema(tag: Option<&str>) -> String {
    match tag {
        Some(t) => format!("schema `{t}`"),
        None => format!("no schema tag, before {FIT_STATE_SCHEMA}"),
    }
}

/// The build a file records in `camdl_version`, as a message names it.
fn producer_of(table: &toml::Table) -> String {
    match table.get("camdl_version").and_then(|v| v.as_str()) {
        Some(v) => format!("camdl {v}"),
        None => "a camdl that did not record its version".to_string(),
    }
}

fn state_of(path: &str, table: toml::Table) -> Result<FitState, String> {
    table.try_into().map_err(|e| format!("parse error in {}: {}", path, e))
}

impl FitState {
    /// Reads `<dir>/fit_state.toml` as the input to a new computation: the
    /// point a `from_mle` start reads, the verdict the warm-start source
    /// check reads, the base of a resumed PGAS chain, or the record a runner
    /// reads back after writing it.
    ///
    /// A file whose schema tag is not [`FIT_STATE_SCHEMA`], including one
    /// with no tag, is refused, naming the build that wrote it and its tag,
    /// and saying to run the fit again: its keys are not certain to mean
    /// what this build's keys mean, and a computation started from a misread
    /// record gives a wrong answer that looks right. A file that cannot be
    /// read or parsed is an error.
    pub fn load_input(dir: &str) -> Result<Self, String> {
        let (path, table) = read_table(dir)?;
        let tag = schema_of(&table);
        if tag != Some(FIT_STATE_SCHEMA) {
            return Err(format!(
                "{path} was written by {} with {}, which this build does not read \
                 as an input. Run that fit again with this build and name the new \
                 run; `camdl fit summary` still shows the old one.",
                producer_of(&table),
                describe_schema(tag),
            ));
        }
        state_of(&path, table)
    }

    /// Reads `<dir>/fit_state.toml` for display: `fit summary`, `fit table`,
    /// a method report, and `predict`'s check of how the chains started.
    ///
    /// A file [`load_input`](Self::load_input) refuses is read as far as its
    /// keys allow, with a notice naming the build that wrote it, its tag and
    /// the keys this build does not read under their own name. In a file
    /// without a tag, the initial-state list is read under its old key,
    /// `perturb_only_at_t0_params`. That list decides only which parameters
    /// the dynamics-only sub-verdict leaves out, never the overall verdict.
    /// A file that cannot be read or parsed is an error.
    pub fn load_for_inspection(dir: &str) -> Result<Inspected, String> {
        let (path, mut table) = read_table(dir)?;
        let tag = schema_of(&table).map(str::to_owned);
        if tag.as_deref() == Some(FIT_STATE_SCHEMA) {
            return Ok(Inspected { state: state_of(&path, table)?, notice: None });
        }
        let unread: Vec<&str> = LEGACY_KEYS.iter().copied().filter(|k| table.contains_key(*k)).collect();
        let producer = producer_of(&table);
        if tag.is_none() && !table.contains_key("initial_state_params") {
            if let Some(list) = table.remove("perturb_only_at_t0_params") {
                table.insert("initial_state_params".into(), list);
            }
        }
        let keys = if unread.is_empty() {
            String::new()
        } else {
            format!(" Keys this build does not read under their own name: {}.", unread.join(", "))
        };
        let notice = crate::out::notice::Notice::info(
            "fit-state-schema",
            format!(
                "{path} was written by {producer} with {}. It is shown as far as this \
                 build reads it, and is refused as an input to a new fit.{keys}",
                describe_schema(tag.as_deref()),
            ),
        );
        Ok(Inspected { state: state_of(&path, table)?, notice: Some(notice) })
    }

    /// Writes the state as the run's `fit_state.toml`, its schema tag
    /// ([`FIT_STATE_SCHEMA`]) first.
    pub fn save(&self, files: &dyn crate::fit::output::RunWriter) -> Result<(), String> {
        let body = toml::to_string_pretty(self)
            .map_err(|e| format!("serialize error: {}", e))?;
        let contents = format!(
            "# Generated by {}\nschema = \"{FIT_STATE_SCHEMA}\"\n{}",
            crate::version::VERSION_SHORT,
            body
        );
        files.put(FIT_STATE_FILE, contents.as_bytes())
            .map_err(|e| format!("cannot write {FIT_STATE_FILE}: {e}"))
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
            input_hash: Some("deadbeef".into()),
            camdl_version: Some("0.1.0+test".into()),
            best_loglik: -123.45,
            initial_loglik: -200.0,
            best_chain: ChainIndex(1),
            n_chains: 2,
            n_good_chains: Some(2),
            start_values: BTreeMap::from([("beta".into(), 0.8)]),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement,
            initial_state_params: vec!["s0".into()],
            chain_logliks: vec![-130.0, -123.45],
            chain_eval_logliks: vec![-128.7, -123.1],
            chain_eval_ids: vec![ChainIndex(0), ChainIndex(1)],
            chain_eval_ses: vec![1.5, 0.8],
            resolved_gate: Some(GateConfig::default()),
            resolved_loglik_eval: Some(LoglikEvalConfig::default()),
            chain_init_source: Some("lhs".into()),
            chain_starts_kind: Some(crate::fit::starts::ChainStartsKind::Spread),
            dt_check: None,
            pf_noise: None,
            optimizer_convergence: None,
            chain_loglik_agreement: None,
        }
    }

    /// fit_state.toml round-trips through save/load with the new
    /// clean-eval fields populated. Catches schema regressions where a
    /// rename or type change would break a reader of the record.
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
        s.save(&crate::fit::output::DirWriter::new(&dir)).unwrap();
        let loaded = FitState::load_input(&dir_str).unwrap();

        assert_eq!(loaded.chain_eval_logliks, vec![-128.7, -123.1]);
        assert_eq!(loaded.chain_eval_ses, vec![1.5, 0.8]);
        assert_eq!(loaded.chain_logliks, vec![-130.0, -123.45]);
        assert_eq!(loaded.tail_chain_agreement.get("beta").copied(), Some(1.02));
        assert_eq!(loaded.initial_state_params, vec!["s0".to_string()]);
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
        let loaded = FitState::load_for_inspection(dir.to_str().unwrap()).unwrap().state;
        assert!(loaded.resolved_gate.is_none(),
            "legacy file must surface resolved_gate as None, not silently default");
        assert!(loaded.resolved_loglik_eval.is_none(),
            "legacy file must surface resolved_loglik_eval as None");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `fit_state.toml` records only what the method ran with or produced.
    /// Its `[rw_sd]` table recorded neither: nl-* copied the run config's
    /// derived step into it (nl-* never perturbs), IF2 wrote a MAD-based
    /// "calibration" it never ran with, PGAS/PMMH wrote it empty, and no
    /// reader consumed it — a leftover of the removed `[stages]` scout →
    /// refine handoff. The table is gone from what is written; a leaf written
    /// before still loads, because `FitState` does not deny unknown fields
    /// and the stale table is simply not read.
    #[test]
    fn fit_state_writes_no_rw_sd_and_still_loads_one_that_has_it() {
        let body = toml::to_string_pretty(&synthetic_state()).expect("serialize");
        assert!(!body.contains("rw_sd"),
            "fit_state.toml must not record an rw_sd no method ran with:\n{body}");
        let legacy = format!("{body}\n[rw_sd]\nbeta = 0.05\n");
        let back: FitState = toml::from_str(&legacy)
            .expect("a leaf written with the old [rw_sd] table still loads");
        assert_eq!(back.start_values.get("beta").copied(), Some(0.8));
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
    /// the tool uses. A `from_mle` start reads `fit_state.toml`, and it spelled the method `method` after the
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
        // The reader is the other half: a file written with the
        // new key must load, and one written with the old key must not quietly
        // parse into a default.
        let back: FitState = toml::from_str(&body).expect("round-trips");
        assert_eq!(back.method, "scout");
        let old = body.replace("method = \"scout\"", "stage = \"scout\"");
        assert!(toml::from_str::<FitState>(&old).is_err(),
            "a file spelling the key `method` is not read (no compatibility \
             alias, VERSIONING.md alpha posture)");
    }

    /// An IF2 `fit_state.toml` as a build before the schema tag wrote it: no
    /// `schema` key, the initial-state list under its old name, and the
    /// `[rw_sd]` table this build no longer reads.
    const UNTAGGED_IF2: &str = r#"# Generated by 0.1.0+oldbuild
method = "if2"
seed = 3
camdl_version = "0.1.0+oldbuild"
best_loglik = -100.0
initial_loglik = -150.0
best_chain = 0
n_chains = 2
perturb_only_at_t0_params = ["I0"]

[start_values]
beta = 0.5
I0 = 4.0

[tail_chain_agreement]
beta = 1.0
I0 = 1.3

[rw_sd]
beta = 0.02
"#;

    /// A state written by this build opens with its schema tag, and reads
    /// back as an input with nothing to say.
    #[test]
    fn a_written_state_carries_the_schema_tag_and_is_an_input() {
        let dir = tempfile::tempdir().unwrap();
        synthetic_state().save(&crate::fit::output::DirWriter::new(dir.path())).unwrap();
        let text = std::fs::read_to_string(dir.path().join(FIT_STATE_FILE)).unwrap();
        assert_eq!(text.lines().nth(1), Some("schema = \"camdl.fit-state/1\""), "{text}");
        let d = dir.path().to_string_lossy();
        FitState::load_input(&d).expect("a tagged state is an input");
        let seen = FitState::load_for_inspection(&d).unwrap();
        assert!(seen.notice.is_none(), "{:?}", seen.notice);
        assert_eq!(seen.state.initial_state_params, vec!["s0".to_string()]);
    }

    /// An untagged state is refused as an input, naming the build that wrote
    /// it, its missing tag and what to do. A `from_mle` start, the
    /// warm-start check, a PGAS resume and the runner's read-back all read
    /// through this function.
    #[test]
    fn an_untagged_state_is_refused_as_an_input() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(FIT_STATE_FILE), UNTAGGED_IF2).unwrap();
        let err = FitState::load_input(&dir.path().to_string_lossy())
            .expect_err("an untagged state is not an input");
        for want in ["camdl 0.1.0+oldbuild", "no schema tag, before camdl.fit-state/1", "Run that fit again"] {
            assert!(err.contains(want), "the refusal names `{want}`: {err}");
        }

        std::fs::write(
            dir.path().join(FIT_STATE_FILE),
            UNTAGGED_IF2.replace("method = \"if2\"", "schema = \"camdl.fit-state/9\"\nmethod = \"if2\""),
        ).unwrap();
        let err = FitState::load_input(&dir.path().to_string_lossy())
            .expect_err("a state of another schema is not an input");
        assert!(err.contains("schema `camdl.fit-state/9`"), "{err}");
    }

    /// The same untagged state is read for display, with a notice naming its
    /// build and the keys this build does not read under their own name, and
    /// with its initial-state list read from the old key.
    #[test]
    fn an_untagged_state_is_read_for_inspection_with_a_notice() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(FIT_STATE_FILE), UNTAGGED_IF2).unwrap();
        let seen = FitState::load_for_inspection(&dir.path().to_string_lossy())
            .expect("an untagged state is read for inspection");
        assert_eq!(seen.state.start_values.get("I0").copied(), Some(4.0));
        assert_eq!(seen.state.initial_state_params, vec!["I0".to_string()],
            "the old key still names the parameters the dynamics-only sub-verdict leaves out");
        let notice = seen.notice.expect("an untagged state is read with a notice");
        for want in ["camdl 0.1.0+oldbuild", "no schema tag", "perturb_only_at_t0_params", "rw_sd"] {
            assert!(notice.message.contains(want), "the notice names `{want}`: {}", notice.message);
        }
    }
}
