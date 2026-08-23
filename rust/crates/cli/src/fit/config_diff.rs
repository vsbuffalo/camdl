//! Structured fit-config diff engine.
//!
//! Computes a typed diff between two `FitConfigV2` instances —
//! ([estimate], [fixed], bounds, priors, data hashes, stages) — for
//! the `table_row.config_diff_from_baseline` field. JSON consumers
//! need a structured shape (free-form text loses information); the
//! text renderer in `fit table` projects the same struct
//! deterministically.
//!
//! See `docs/dev/proposals/2026-04-28-fit-experiment-management.md` §4.
//!
//! **Parser reuse.** This module never re-implements fit.toml parsing;
//! it always loads via [`config_v2::FitConfigV2::load`]. Two parsers
//! diverging silently on edge cases (transform aliases, prior syntax,
//! default filling) is exactly the drift class this proposal exists to
//! prevent.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::fit::config_v2::{FitConfigV2, PriorDist, Stage};
use crate::fit::fit_view::FitView;

/// Structured diff of one fit's config relative to a baseline. Map
/// fields use `BTreeMap` end-to-end so JSON serialization is
/// lex-ordered (load-bearing for the `summary ⊆ table` byte-equality
/// test in Deliverable C).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConfigDiff {
    /// Hash of the baseline fit. `None` only when no baseline was
    /// available (e.g. computing the diff for the baseline against
    /// itself produces all-empty fields and the baseline_hash equals
    /// the fit's own hash; the explicit `None` case is reserved for
    /// callers that don't pick a baseline at all).
    pub baseline_hash: Option<String>,
    /// True iff the underlying camdl model IR hash differs. When true,
    /// scalar comparisons (best_loglik, R0, etc.) are misleading; the
    /// text renderer says `(model changed; comparison limited)`.
    pub model_changed: bool,
    /// Parameters that moved into `[estimate]` (or appeared new).
    pub estimate_added: Vec<String>,
    /// Parameters that left `[estimate]`.
    pub estimate_removed: Vec<String>,
    /// Parameters that moved into `[fixed]` (or appeared new).
    pub fixed_added: Vec<String>,
    /// Parameters that left `[fixed]`.
    pub fixed_removed: Vec<String>,
    /// Parameters whose `[estimate.<name>].bounds` tuple changed.
    pub bounds_changed: Vec<BoundsChange>,
    /// Parameters whose declared `prior` differs (after canonical
    /// rendering — see [`format_prior`]). Add ↔ remove ↔ retype all
    /// fall under this collection.
    pub priors_changed: Vec<PriorChange>,
    /// Per-stream data file hash differences.
    pub data_hashes: DataHashesDiff,
    /// Stage-level diff (added / removed names plus per-stage settings
    /// changes).
    pub stages_changed: StagesChanged,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BoundsChange {
    pub param: String,
    /// `None` means the previous fit.toml omitted explicit bounds and
    /// fell back to the model file's parameters block. After bounds
    /// became optional in `[estimate.X]`, omit-vs-explicit is itself
    /// a meaningful change (e.g. omit → explicit narrowing).
    pub from: Option<(f64, f64)>,
    pub to: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PriorChange {
    pub param: String,
    /// Canonical prior string (output of [`format_prior`]). `None`
    /// means no prior was declared (only meaningful for MLE-only
    /// parameters; Bayesian fits require a prior).
    pub from: Option<String>,
    pub to: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct DataHashesDiff {
    /// Streams present in this fit but not in the baseline.
    pub added: Vec<String>,
    /// Streams present in the baseline but not in this fit.
    pub removed: Vec<String>,
    /// Streams in both fits whose content hashes differ.
    pub modified: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct StagesChanged {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    /// Per-stage settings changes — one entry per (stage, key) tuple
    /// whose value changed. The shape is intentionally flat (a list of
    /// settings deltas) rather than nested maps; renderers project
    /// however they want.
    pub settings_changed: Vec<StageSettingsChange>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct StageSettingsChange {
    pub stage: String,
    pub key: String,
    /// Pre-image as a JSON value (numbers stay numeric, strings stay
    /// stringy). Stored as `serde_json::Value` so consumers can
    /// type-introspect.
    pub from: serde_json::Value,
    pub to: serde_json::Value,
}

impl ConfigDiff {
    /// Diff of a fit against itself — every `_added`/`_removed`
    /// vector empty, no scalar changes, baseline_hash = self's hash.
    /// Used by `fit summary` (single-fit) and by `fit table` when the
    /// baseline-selection policy resolves to the same row (e.g.
    /// `--hash <h>` filtering to one row).
    pub fn identity(self_hash: &str) -> Self {
        ConfigDiff {
            baseline_hash: Some(self_hash.to_string()),
            model_changed: false,
            estimate_added: Vec::new(),
            estimate_removed: Vec::new(),
            fixed_added: Vec::new(),
            fixed_removed: Vec::new(),
            bounds_changed: Vec::new(),
            priors_changed: Vec::new(),
            data_hashes: DataHashesDiff::default(),
            stages_changed: StagesChanged::default(),
        }
    }

    /// Compare `this` against `baseline`. Both arguments are already
    /// parsed via [`FitConfigV2::load`] — callers that have only paths
    /// should call [`compare_paths`] instead, which loads then
    /// dispatches here.
    ///
    /// `model_changed` requires the caller to supply each fit's
    /// `model_identity` from its [`FitView`] (the `FitConfigV2` itself only
    /// references the model file; the canonical hash lives on the fit-level
    /// view / sidecar).
    pub fn compare(
        this: &FitConfigV2,
        baseline: &FitConfigV2,
        this_meta: &FitView,
        baseline_meta: &FitView,
    ) -> Self {
        let this_est: BTreeSet<&str> =
            this.estimate.keys().map(|s| s.as_str()).collect();
        let base_est: BTreeSet<&str> =
            baseline.estimate.keys().map(|s| s.as_str()).collect();

        let this_fix = this.fixed.resolve().unwrap_or_default();
        let base_fix = baseline.fixed.resolve().unwrap_or_default();
        let this_fix_keys: BTreeSet<&str> =
            this_fix.keys().map(|s| s.as_str()).collect();
        let base_fix_keys: BTreeSet<&str> =
            base_fix.keys().map(|s| s.as_str()).collect();

        let estimate_added: Vec<String> = this_est
            .difference(&base_est)
            .map(|s| s.to_string())
            .collect();
        let estimate_removed: Vec<String> = base_est
            .difference(&this_est)
            .map(|s| s.to_string())
            .collect();
        let fixed_added: Vec<String> = this_fix_keys
            .difference(&base_fix_keys)
            .map(|s| s.to_string())
            .collect();
        let fixed_removed: Vec<String> = base_fix_keys
            .difference(&this_fix_keys)
            .map(|s| s.to_string())
            .collect();

        let mut bounds_changed = Vec::new();
        for name in this_est.intersection(&base_est) {
            let tb = this.estimate[*name].bounds;
            let bb = baseline.estimate[*name].bounds;
            // Bounds Option-equality: omit↔omit unchanged; explicit↔omit
            // is a change; explicit↔explicit compares exact tuple.
            let differ = match (tb, bb) {
                (None, None) => false,
                (Some(t), Some(b)) => (t.0 - b.0).abs() > 0.0 || (t.1 - b.1).abs() > 0.0,
                _ => true,
            };
            if differ {
                bounds_changed.push(BoundsChange {
                    param: (*name).to_string(),
                    from: bb,
                    to: tb,
                });
            }
        }

        let mut priors_changed = Vec::new();
        let estimate_union: BTreeSet<&str> =
            this_est.union(&base_est).copied().collect();
        for name in &estimate_union {
            let tp = this.estimate.get(*name).and_then(|e| e.prior.as_ref());
            let bp = baseline.estimate.get(*name).and_then(|e| e.prior.as_ref());
            let tp_str = tp.map(format_prior);
            let bp_str = bp.map(format_prior);
            if tp_str != bp_str {
                priors_changed.push(PriorChange {
                    param: (*name).to_string(),
                    from: bp_str,
                    to: tp_str,
                });
            }
        }

        let data_hashes =
            diff_data_hashes(&this_meta.data_hashes, &baseline_meta.data_hashes);
        let stages_changed = diff_stages(&this.stages, &baseline.stages);

        ConfigDiff {
            baseline_hash: Some(baseline_meta_hash(baseline_meta)),
            model_changed: this_meta.model_identity != baseline_meta.model_identity,
            estimate_added,
            estimate_removed,
            fixed_added,
            fixed_removed,
            bounds_changed,
            priors_changed,
            data_hashes,
            stages_changed,
        }
    }
}

/// Canonical string projection for a [`PriorDist`]. Intentionally
/// terse — `log_normal(mu=..., sigma=...)`, `normal(mean=..., sd=...)`,
/// `beta(alpha=..., beta=...)`, `uniform(lower=..., upper=...)`,
/// `half_normal(sigma=...)`, `gamma(shape=..., rate=...)`,
/// `exponential(rate=...)`, `fixed(...)`, or `flat()` for the gh#75
/// explicit flat-prior opt-in.
/// The format is stable across versions because `priors_changed`
/// equality compares strings.
pub fn format_prior(spec: &crate::fit::config_v2::EstimatePriorSpec) -> String {
    use crate::fit::config_v2::EstimatePriorSpec;
    match spec {
        EstimatePriorSpec::Flat { .. } => "flat()".to_string(),
        EstimatePriorSpec::UniformOverBounds { .. } => "uniform(over bounds)".to_string(),
        EstimatePriorSpec::Dist(p) => match p {
            PriorDist::LogNormal(q) =>
                format!("log_normal(mu={}, sigma={})", q.mu, q.sigma),
            PriorDist::Normal(q) =>
                format!("normal(mean={}, sd={})", q.mean, q.sd),
            PriorDist::Beta(q) =>
                format!("beta(alpha={}, beta={})", q.alpha, q.beta),
            PriorDist::Uniform(q) =>
                format!("uniform(lower={}, upper={})", q.lower, q.upper),
            PriorDist::HalfNormal(q) =>
                format!("half_normal(sigma={})", q.sigma),
            PriorDist::Gamma(q) =>
                format!("gamma(shape={}, rate={})", q.shape, q.rate),
            PriorDist::Exponential(q) =>
                format!("exponential(rate={})", q.rate),
            PriorDist::LogUniform(q) =>
                format!("log_uniform(lower={}, upper={})", q.lower, q.upper),
            PriorDist::TruncatedNormal(q) =>
                format!("truncated_normal(mean={}, sd={}, lower={}, upper={})", q.mean, q.sd, q.lower, q.upper),
            PriorDist::Fixed(v) =>
                format!("fixed({})", v),
        }
    }
}

fn baseline_meta_hash(meta: &FitView) -> String {
    // The view carries the fit's input hash via fit_toml_hash (the canonical
    // hash for a v2 fit.toml); the *fit* hash itself is `fit_hash`. In practice
    // the consumer (`fit table`) passes a pre-resolved baseline hash via
    // `ConfigDiff::with_baseline_hash`, so this fallback is rarely the value
    // surfaced.
    meta.fit_toml_hash.clone()
}

impl ConfigDiff {
    /// Override the `baseline_hash` field after construction. Used by
    /// `fit table` to put the actual fit_hash on the diff rather than
    /// the fit_toml_hash that [`compare`] defaults to.
    pub fn with_baseline_hash(mut self, hash: String) -> Self {
        self.baseline_hash = Some(hash);
        self
    }
}

fn diff_data_hashes(
    this: &std::collections::HashMap<String, String>,
    baseline: &std::collections::HashMap<String, String>,
) -> DataHashesDiff {
    let this_keys: BTreeSet<&str> = this.keys().map(|s| s.as_str()).collect();
    let base_keys: BTreeSet<&str> = baseline.keys().map(|s| s.as_str()).collect();

    let added: Vec<String> = this_keys
        .difference(&base_keys)
        .map(|s| s.to_string())
        .collect();
    let removed: Vec<String> = base_keys
        .difference(&this_keys)
        .map(|s| s.to_string())
        .collect();
    let mut modified = Vec::new();
    for name in this_keys.intersection(&base_keys) {
        if this.get(*name) != baseline.get(*name) {
            modified.push((*name).to_string());
        }
    }
    DataHashesDiff {
        added,
        removed,
        modified,
    }
}

fn diff_stages(
    this: &indexmap::IndexMap<String, Stage>,
    baseline: &indexmap::IndexMap<String, Stage>,
) -> StagesChanged {
    let this_keys: BTreeSet<&str> = this.keys().map(|s| s.as_str()).collect();
    let base_keys: BTreeSet<&str> = baseline.keys().map(|s| s.as_str()).collect();

    let added: Vec<String> = this_keys
        .difference(&base_keys)
        .map(|s| s.to_string())
        .collect();
    let removed: Vec<String> = base_keys
        .difference(&this_keys)
        .map(|s| s.to_string())
        .collect();

    let mut settings_changed = Vec::new();
    for name in this_keys.intersection(&base_keys) {
        let ts = stage_settings_map(&this[*name]);
        let bs = stage_settings_map(&baseline[*name]);
        let key_union: BTreeSet<&str> =
            ts.keys().chain(bs.keys()).map(|s| s.as_str()).collect();
        for key in key_union {
            let from_v = bs.get(key).cloned().unwrap_or(serde_json::Value::Null);
            let to_v = ts.get(key).cloned().unwrap_or(serde_json::Value::Null);
            if from_v != to_v {
                settings_changed.push(StageSettingsChange {
                    stage: (*name).to_string(),
                    key: key.to_string(),
                    from: from_v,
                    to: to_v,
                });
            }
        }
    }
    StagesChanged {
        added,
        removed,
        settings_changed,
    }
}

/// Project a `Stage` into a flat key→value settings map, derived from the
/// stage's own serialization (dotted paths for nested tables) — the same
/// move `validate_stage_keys` makes to get its allowed-key set. The
/// previous hand-maintained per-variant projection listed a SUBSET of each
/// variant's fields, so `fit diff` reported "no settings changed" for two
/// configs differing in anything swept into its `..` (tempering, use_nuts,
/// dt_check, every init selector, …) — a silent wrong answer from a
/// provenance surface. Deriving from serialization means the key set can
/// never drift from the enum; keys carry the TOML-side spellings
/// (`init_mle`, `init`), which is what the user wrote and diffs against.
fn stage_settings_map(stage: &Stage) -> BTreeMap<String, serde_json::Value> {
    let mut m = BTreeMap::new();
    // The serde tag puts `algorithm` in the map alongside every field;
    // `backend` is an ordinary field on all variants.
    let v = serde_json::to_value(stage).unwrap_or(serde_json::Value::Null);
    flatten_settings("", &v, &mut m);
    m
}

/// Recursively flatten a JSON object into dotted-path keys. Non-object
/// leaves (numbers, strings, bools, nulls, arrays) are inserted as-is —
/// arrays (e.g. `tempering`) diff as whole values, which is the readable
/// grain for a settings row.
fn flatten_settings(
    prefix: &str,
    v: &serde_json::Value,
    m: &mut BTreeMap<String, serde_json::Value>,
) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                let key = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_settings(&key, val, m);
            }
        }
        leaf => {
            m.insert(prefix.to_string(), leaf.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::config_v2::FitConfigV2;
    use std::collections::HashMap;

    fn fitmeta(model_identity: &str) -> FitView {
        FitView {
            fit_hash: "f".repeat(64),
            engine_version: "0.1.0+test".into(),
            created_at: "2026-04-27T00:00:00Z".into(),
            argv: Vec::new(),
            label: None,
            model: "sir.camdl".into(),
            model_identity: model_identity.into(),
            fit_toml_path: "fit.toml".into(),
            fit_toml_hash: "h".repeat(64),
            data_hashes: HashMap::new(),
            estimated: Vec::new(),
            fixed: HashMap::new(),
            resolved_priors: Vec::new(),
            parameters_provenance: Default::default(),
            stages_declared: Vec::new(),
            stages: Vec::new(),
        }
    }

    fn parse(s: &str) -> FitConfigV2 {
        toml::from_str(s).expect("toml parse")
    }

    const BASELINE_TOML: &str = r#"
        [model]
        camdl = "sir.camdl"

        [data]
        observations = { cases = "cases.tsv" }

        [estimate.R0]
        bounds = [1.0, 100.0]

        [estimate.sigma]
        bounds = [0.01, 0.5]
        prior = { log_normal = { mu = -2.0, sigma = 0.5 } }

        [fixed]
        N0 = 1000.0

        [stages.scout]
        algorithm = "if2"
        backend = "chain_binomial"
        chains = 4
        particles = 500
        iterations = 50
        cooling = 0.7

        [stages.refine]
        algorithm = "if2"
        backend = "chain_binomial"
        chains = 4
        particles = 1000
        iterations = 100
        cooling = 0.5
    "#;

    #[test]
    fn identity_diff_is_all_empty() {
        let diff = ConfigDiff::identity("abcd1234");
        assert_eq!(diff.baseline_hash.as_deref(), Some("abcd1234"));
        assert!(!diff.model_changed);
        assert!(diff.estimate_added.is_empty());
        assert!(diff.estimate_removed.is_empty());
        assert!(diff.bounds_changed.is_empty());
        assert!(diff.priors_changed.is_empty());
    }

    #[test]
    fn detects_estimate_to_fixed_move() {
        let baseline = parse(BASELINE_TOML);
        let mut variant_str = BASELINE_TOML.replace(
            "[fixed]\n        N0 = 1000.0",
            "[fixed]\n        N0 = 1000.0\n        sigma = 0.08",
        );
        variant_str = variant_str.replace(
            "[estimate.sigma]\n        bounds = [0.01, 0.5]\n        prior = { log_normal = { mu = -2.0, sigma = 0.5 } }\n",
            "",
        );
        let variant = parse(&variant_str);
        let diff = ConfigDiff::compare(
            &variant,
            &baseline,
            &fitmeta("modelA"),
            &fitmeta("modelA"),
        );
        assert_eq!(diff.estimate_removed, vec!["sigma".to_string()]);
        assert_eq!(diff.fixed_added, vec!["sigma".to_string()]);
        assert!(diff.estimate_added.is_empty());
    }

    #[test]
    fn detects_bounds_change() {
        let baseline = parse(BASELINE_TOML);
        let variant_str = BASELINE_TOML.replace("[1.0, 100.0]", "[40.0, 80.0]");
        let variant = parse(&variant_str);
        let diff = ConfigDiff::compare(
            &variant,
            &baseline,
            &fitmeta("modelA"),
            &fitmeta("modelA"),
        );
        assert_eq!(diff.bounds_changed.len(), 1);
        let bc = &diff.bounds_changed[0];
        assert_eq!(bc.param, "R0");
        assert_eq!(bc.from, Some((1.0, 100.0)));
        assert_eq!(bc.to, Some((40.0, 80.0)));
    }

    #[test]
    fn detects_prior_change() {
        let baseline = parse(BASELINE_TOML);
        // Add a prior on R0 (was none).
        let variant_str = BASELINE_TOML.replace(
            "[estimate.R0]\n        bounds = [1.0, 100.0]",
            "[estimate.R0]\n        bounds = [1.0, 100.0]\n        prior = { log_normal = { mu = 4.0, sigma = 0.4 } }",
        );
        let variant = parse(&variant_str);
        let diff = ConfigDiff::compare(
            &variant,
            &baseline,
            &fitmeta("modelA"),
            &fitmeta("modelA"),
        );
        assert_eq!(diff.priors_changed.len(), 1);
        let pc = &diff.priors_changed[0];
        assert_eq!(pc.param, "R0");
        assert_eq!(pc.from, None);
        assert_eq!(pc.to.as_deref(), Some("log_normal(mu=4, sigma=0.4)"));
    }

    #[test]
    fn detects_stage_added_and_settings_changed() {
        let baseline = parse(BASELINE_TOML);
        let variant_str = BASELINE_TOML.replace(
            "[stages.refine]\n        algorithm = \"if2\"\n        backend = \"chain_binomial\"\n        chains = 4\n        particles = 1000\n        iterations = 100\n        cooling = 0.5",
            "[stages.refine]\n        algorithm = \"if2\"\n        backend = \"chain_binomial\"\n        chains = 8\n        particles = 1000\n        iterations = 100\n        cooling = 0.5\n\n        [stages.validate]\n        algorithm = \"if2\"\n        backend = \"chain_binomial\"\n        chains = 4\n        particles = 5000\n        iterations = 20\n        cooling = 0.9",
        );
        let variant = parse(&variant_str);
        let diff = ConfigDiff::compare(
            &variant,
            &baseline,
            &fitmeta("modelA"),
            &fitmeta("modelA"),
        );
        assert_eq!(diff.stages_changed.added, vec!["validate".to_string()]);
        assert!(diff.stages_changed.removed.is_empty());
        // refine.chains: 4 → 8
        let chains_chg = diff
            .stages_changed
            .settings_changed
            .iter()
            .find(|s| s.stage == "refine" && s.key == "chains")
            .expect("refine.chains delta missing");
        assert_eq!(chains_chg.from, serde_json::json!(4));
        assert_eq!(chains_chg.to, serde_json::json!(8));
    }

    /// The settings map must cover EVERY stage field, not a hand-picked
    /// subset: the old per-variant projection swallowed `tempering`,
    /// `use_nuts`, the init selectors and more with `..`, so a diff over
    /// any of them reported "no settings changed". Derived-from-
    /// serialization can't drift from the enum; this pins the fields the
    /// old code demonstrably missed.
    #[test]
    fn settings_map_covers_fields_the_old_projection_swallowed() {
        let cfg: FitConfigV2 = toml::from_str(r#"
        [model]
        camdl = "m.camdl"

        [estimate]
        beta = { bounds = [0.01, 2.0], prior = { log_normal = { mu = 0.0, sigma = 1.0 } } }

        [fixed]
        N0 = 1000000

        [stages.posterior]
        algorithm  = "pgas"
        backend    = "chain_binomial"
        chains     = 2
        particles  = 100
        sweeps     = 10
        tempering  = [1.0, 0.5]
        "#).expect("toml parse");
        let m = stage_settings_map(&cfg.stages["posterior"]);
        for key in ["algorithm", "backend", "chains", "tempering", "use_nuts", "init", "init_mle"] {
            assert!(m.contains_key(key),
                "settings map must carry '{key}' — a fit diff over it \
                 previously reported no change; keys present: {:?}",
                m.keys().collect::<Vec<_>>());
        }
        // And a diff over one of the previously-swallowed fields yields a row.
        let mut hot = cfg.stages["posterior"].clone();
        if let Stage::PGAS { tempering, .. } = &mut hot { *tempering = vec![1.0, 0.7, 0.4]; }
        let hot_map = stage_settings_map(&hot);
        assert_ne!(m.get("tempering"), hot_map.get("tempering"),
            "a tempering change must be visible to fit diff");
    }

    #[test]
    fn model_changed_requires_distinct_model_identities() {
        let baseline = parse(BASELINE_TOML);
        let diff_same = ConfigDiff::compare(
            &baseline,
            &baseline,
            &fitmeta("modelA"),
            &fitmeta("modelA"),
        );
        assert!(!diff_same.model_changed);
        let diff_changed = ConfigDiff::compare(
            &baseline,
            &baseline,
            &fitmeta("modelB"),
            &fitmeta("modelA"),
        );
        assert!(diff_changed.model_changed);
    }

    #[test]
    fn data_hashes_added_removed_modified() {
        let mut base = fitmeta("modelA");
        base.data_hashes.insert("cases".into(), "h1".into());
        base.data_hashes.insert("deaths".into(), "h2".into());
        let mut this = fitmeta("modelA");
        this.data_hashes.insert("cases".into(), "h1prime".into()); // modified
        this.data_hashes.insert("hospital".into(), "h3".into());   // added
        // "deaths" removed.
        let cfg = parse(BASELINE_TOML);
        let diff = ConfigDiff::compare(&cfg, &cfg, &this, &base);
        assert_eq!(diff.data_hashes.added, vec!["hospital".to_string()]);
        assert_eq!(diff.data_hashes.removed, vec!["deaths".to_string()]);
        assert_eq!(diff.data_hashes.modified, vec!["cases".to_string()]);
    }

    /// Identity diff serializes deterministically — empty vectors and
    /// `model_changed: false`. The test exists so any later schema
    /// change visible to consumers fails loudly.
    #[test]
    fn identity_diff_json_shape_is_stable() {
        let diff = ConfigDiff::identity("abc");
        let json = serde_json::to_value(&diff).unwrap();
        assert_eq!(json["baseline_hash"], "abc");
        assert_eq!(json["model_changed"], false);
        assert_eq!(json["estimate_added"], serde_json::json!([]));
        assert_eq!(json["data_hashes"]["modified"], serde_json::json!([]));
        assert_eq!(json["stages_changed"]["settings_changed"], serde_json::json!([]));
    }
}
