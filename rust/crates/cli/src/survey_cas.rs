//! gh#147 (M3.3). The survey CAS identity: map a resolved likelihood-landscape
//! survey into the `runid` factored levels (`model` / `config` / `box` /
//! `seed`) and its leaf `run_id`.
//!
//! A `camdl survey` is a likelihood-landscape diagnostic: N Latin-hypercube
//! points across the declared parameter bounds, each scored. It is a single
//! leaf (one `landscape.tsv`), no grid — the N points are *within* the leaf,
//! not an axis.
//!
//! Factoring (mirrors [`crate::pfilter_cas`] / [`crate::resolve`]):
//!   - **model** — the pure model IR digest.
//!   - **config** — the eval setup + problem context: `eval_method`,
//!     `eval_particles`, `eval_replicates`, the observed data digests, the
//!     fixed params, and the scenario. The eval count knobs are
//!     identity-bearing — they change the stored landscape loglik values, so
//!     they live in the key (the n_trajectories / n_replicates collision
//!     class; the old `SurveyInputs::content_hash` folded them in too).
//!   - **box** — the LHS sampling box: the estimated-param bounds + n_points
//!     (what region, how densely).
//!   - **seed** — the resolved LHS / PF base seed.

use runid::inputs::{EngineIdentity, ModelDigest, Seed};
use runid::{run_id, ArtifactKind, ContentHash, LevelId};

use crate::fit::cas::canonical_config_hash;

/// A fully-resolved survey leaf: the four factored levels (in path order) and
/// the leaf `run_id` composed from their hashes.
pub struct ResolvedSurvey {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_survey`], all resolved by the caller.
pub struct SurveyCtx<'a> {
    pub model: &'a ir::Model,
    pub ir_version: &'a str,
    /// The engine the model level folds: the build and the engine switches
    /// set (`env::Environment::engine_identity`). A survey run with a switch
    /// never shares a leaf with a default survey, whose key is the build's alone.
    pub engine: &'a EngineIdentity,
    /// Provenance label for the `model` path segment (the model stem).
    pub stem: &'a str,
    /// `(stream name, sha256-hex)` — already-computed data digests.
    pub data: &'a [(String, String)],
    /// Resolved eval method name (`"pfilter"` / `"simulate"`).
    pub eval_method: &'a str,
    pub eval_particles: u32,
    pub eval_replicates: u32,
    /// The likelihood each point is scored under (`ic_free` from the fit
    /// config). It changes every stored loglik, so it is identity.
    pub objective: crate::fit::config_v2::LikelihoodObjective,
    /// `--allow-forcing-extrapolation` or the fit config's `[config]
    /// allow_forcing_extrapolation`: the run may hold a forcing's end value
    /// outside its range (D44), so it is identity.
    pub allow_forcing_extrapolation: bool,
    /// `--allow-degenerate-rates` or the fit config's `[config]
    /// allow_degenerate_rates`: a collapsing rate is 0 instead of failing the
    /// trajectory, which changes the stored logliks, so it is identity.
    pub allow_degenerate_rates: bool,
    /// Under the fit config's `holdout_after`, how many leading rows of each
    /// stream train (by stream name); every later row was left out of every
    /// point's likelihood, so it is identity.
    pub training_rows: Option<&'a [(String, usize)]>,
    /// The LHS box: `(param, lo, hi)`, the estimated-param bounds.
    pub bounds: &'a [(String, f64, f64)],
    /// Fixed `(param, value)` (excluded from the LHS box).
    pub fixed: &'a [(String, f64)],
    /// Scenario name, if any.
    pub scenario: Option<&'a str>,
    pub n_points: u32,
    /// The resolved LHS / PF base seed (the `seed` level hashes this).
    pub seed: u64,
}

use crate::fit::cas::{data_digests, level, structural_level_hash};

/// The `config` level: the eval setup and problem context a landscape is
/// computed under. A struct rather than a `json!` literal so the level is
/// include-by-default — see `PfilterConfigLevel` for the full argument.
#[derive(serde::Serialize)]
struct SurveyConfigLevel<'a> {
    eval_method: &'a str,
    eval_particles: u32,
    eval_replicates: u32,
    data: &'a [(&'a str, &'a str)],
    fixed: &'a [(&'a str, f64)],
    scenario: Option<&'a str>,
    /// Omitted when full, so a survey that never set `ic_free` keeps its key.
    #[serde(skip_serializing_if = "crate::fit::config_v2::LikelihoodObjective::is_full")]
    objective: crate::fit::config_v2::LikelihoodObjective,
    /// Omitted when off, so the level of a run without the override is the
    /// one the literal pins.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    allow_forcing_extrapolation: bool,
    /// Omitted when off, so a survey without the setting keeps the key the
    /// literal pins (gh#1235).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    allow_degenerate_rates: bool,
    /// The training rows a `holdout_after` cutoff leaves each stream. The
    /// data and fixed values are keyed already; this is what says the rows
    /// after them were cut. Omitted without one (gh#1059).
    #[serde(skip_serializing_if = "Option::is_none")]
    training_rows: Option<&'a [(String, usize)]>,
}

/// The `box` level: the LHS sampling spec.
#[derive(serde::Serialize)]
struct SurveyBoxLevel<'a> {
    bounds: &'a [(&'a str, f64, f64)],
    n_points: u32,
}

/// Resolve a survey leaf's identity: the four factored levels and the `run_id`.
pub fn resolve_survey(ctx: &SurveyCtx) -> Result<ResolvedSurvey, String> {
    // config level — eval setup + problem context. The eval count knobs are
    // identity-bearing (they change the stored landscape).
    let _ = data_digests(ctx.data)?;
    let mut data_sorted: Vec<(&str, &str)> =
        ctx.data.iter().map(|(n, h)| (n.as_str(), h.as_str())).collect();
    data_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut fixed_sorted: Vec<(&str, f64)> =
        ctx.fixed.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    fixed_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let config_level = SurveyConfigLevel {
        eval_method: ctx.eval_method,
        eval_particles: ctx.eval_particles,
        eval_replicates: ctx.eval_replicates,
        data: &data_sorted,
        fixed: &fixed_sorted,
        scenario: ctx.scenario,
        objective: ctx.objective,
        allow_forcing_extrapolation: ctx.allow_forcing_extrapolation,
        allow_degenerate_rates: ctx.allow_degenerate_rates,
        training_rows: ctx.training_rows,
    };

    // box level — the LHS sampling spec.
    let mut bounds_sorted: Vec<(&str, f64, f64)> =
        ctx.bounds.iter().map(|(n, lo, hi)| (n.as_str(), *lo, *hi)).collect();
    bounds_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let box_level = SurveyBoxLevel { bounds: &bounds_sorted, n_points: ctx.n_points };

    let model_digest = ModelDigest::from_model(
        ctx.model,
        ctx.ir_version.to_string(),
        ctx.engine.clone(),
    );
    let seed = Seed { process_seed: ctx.seed, base_seed: ctx.seed };

    let config_label = format!(
        "{}-P{}-r{}", ctx.eval_method, ctx.eval_particles, ctx.eval_replicates);
    let box_label = format!("box-n{}", ctx.n_points);

    let levels = vec![
        level("model", ctx.stem, structural_level_hash(&model_digest)),
        level("config", &config_label, canonical_config_hash(&config_level, &[])?),
        level("box", &box_label, canonical_config_hash(&box_level, &[])?),
        level("seed", &format!("seed_{}", ctx.seed), structural_level_hash(&seed)),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::Survey, &level_hashes);
    Ok(ResolvedSurvey { levels, run_id: rid })
}

#[cfg(test)]
mod tests {
    use super::*;
    use runid::ContentAddressed;

    fn h(byte: u8) -> String {
        ContentHash::digest_bytes(&[byte]).to_hex()
    }

    // The `config` and `box` levels are `digest_value` over the same JSON
    // blobs `resolve_survey` builds; unit-testing those digests pins the
    // collision-freeness of the identity components without an `ir::Model`
    // fixture (mirrors `pfilter_cas::tests`).

    fn config_level(method: &str, particles: u32, replicates: u32,
                    data: &[(&str, &str)], scenario: Option<&str>) -> crate::fit::cas::LevelHash {
        let mut d: Vec<(&str, &str)> = data.to_vec();
        d.sort_by(|a, b| a.0.cmp(b.0));
        canonical_config_hash(&serde_json::json!({
            "eval_method": method, "eval_particles": particles,
            "eval_replicates": replicates, "data": d,
            "fixed": Vec::<(&str, f64)>::new(), "scenario": scenario,
        }), &[]).unwrap()
    }

    fn box_level(bounds: &[(&str, f64, f64)], n_points: u32) -> crate::fit::cas::LevelHash {
        let mut b: Vec<(&str, f64, f64)> = bounds.to_vec();
        b.sort_by(|a, x| a.0.cmp(x.0));
        canonical_config_hash(&serde_json::json!({ "bounds": b, "n_points": n_points }), &[]).unwrap()
    }

    /// Byte-neutrality of the struct rewrite: `SurveyConfigLevel` /
    /// `SurveyBoxLevel` must digest EXACTLY as the `json!` literals they
    /// replaced — the `config_level` / `box_level` helpers above ARE those
    /// literals — or every stored survey leaf silently re-keys.
    #[test]
    fn levels_are_byte_identical_to_the_literals_they_replaced() {
        let h1 = h(1);
        let data: Vec<(&str, &str)> = vec![("cases", h1.as_str())];
        let fixed: Vec<(&str, f64)> = Vec::new();
        let cfg = SurveyConfigLevel {
            eval_method: "pfilter", eval_particles: 200, eval_replicates: 1,
            data: &data, fixed: &fixed, scenario: None,
            objective: crate::fit::config_v2::LikelihoodObjective::Full,
            allow_forcing_extrapolation: false,
            allow_degenerate_rates: false,
            training_rows: None,
        };
        assert_eq!(canonical_config_hash(&cfg, &[]).unwrap(),
                   config_level("pfilter", 200, 1, &[("cases", &h1)], None),
                   "the config struct must reproduce the literal's digest");

        let bounds: Vec<(&str, f64, f64)> = vec![("beta", 0.01, 2.0)];
        let bx = SurveyBoxLevel { bounds: &bounds, n_points: 64 };
        assert_eq!(canonical_config_hash(&bx, &[]).unwrap(),
                   box_level(&[("beta", 0.01, 2.0)], 64),
                   "the box struct must reproduce the literal's digest");
    }

    /// The likelihood objective is identity: a survey of an `ic_free` fit
    /// stores different logliks from one of the same problem without it, so
    /// the two must not share a leaf. The full objective is omitted, so a
    /// survey that never set `ic_free` keeps the key the literal pins.
    #[test]
    fn conditional_objective_rekeys_the_config_level() {
        use crate::fit::config_v2::LikelihoodObjective;
        let h1 = h(1);
        let data: Vec<(&str, &str)> = vec![("cases", h1.as_str())];
        let fixed: Vec<(&str, f64)> = Vec::new();
        let level = |objective| canonical_config_hash(&SurveyConfigLevel {
            eval_method: "pfilter", eval_particles: 200, eval_replicates: 1,
            data: &data, fixed: &fixed, scenario: None, objective, training_rows: None, allow_forcing_extrapolation: false,
            allow_degenerate_rates: false,
        }, &[]).unwrap();
        let literal = config_level("pfilter", 200, 1, &[("cases", &h1)], None);
        assert_eq!(level(LikelihoodObjective::Full), literal,
            "the full objective must leave the key unchanged");
        assert_ne!(level(LikelihoodObjective::ConditionalOnFirstObservation), literal,
            "the conditional objective must change the config hash");
    }

    /// The training rows a `holdout_after` cutoff leaves are identity: a
    /// survey of a holdout config scores fewer rows than one of the same files
    /// without it. Without one the level is the literal (gh#1059).
    #[test]
    fn a_holdout_after_cutoff_rekeys_the_config_level() {
        let h1 = h(1);
        let data: Vec<(&str, &str)> = vec![("cases", h1.as_str())];
        let fixed: Vec<(&str, f64)> = Vec::new();
        let level = |training_rows| canonical_config_hash(&SurveyConfigLevel {
            eval_method: "pfilter", eval_particles: 200, eval_replicates: 1,
            data: &data, fixed: &fixed, scenario: None,
            objective: crate::fit::config_v2::LikelihoodObjective::Full, training_rows,
            allow_forcing_extrapolation: false, allow_degenerate_rates: false,
        }, &[]).unwrap();
        let literal = config_level("pfilter", 200, 1, &[("cases", &h1)], None);
        let four = [("cases".to_string(), 4)];
        let five = [("cases".to_string(), 5)];
        assert_eq!(level(None), literal, "no cutoff must leave the key unchanged");
        assert_ne!(level(Some(&four)), literal, "a cutoff must change the config hash");
        assert_ne!(level(Some(&four)), level(Some(&five)), "the training rows are keyed");
    }

    /// The forcing-extrapolation override is identity (D44); off, the level
    /// is the literal.
    #[test]
    fn the_forcing_override_rekeys_the_config_level() {
        let h1 = h(1);
        let data: Vec<(&str, &str)> = vec![("cases", h1.as_str())];
        let fixed: Vec<(&str, f64)> = Vec::new();
        let level = |allow_forcing_extrapolation| canonical_config_hash(&SurveyConfigLevel {
            eval_method: "pfilter", eval_particles: 200, eval_replicates: 1,
            data: &data, fixed: &fixed, scenario: None,
            objective: crate::fit::config_v2::LikelihoodObjective::Full, training_rows: None,
            allow_forcing_extrapolation, allow_degenerate_rates: false,
        }, &[]).unwrap();
        let literal = config_level("pfilter", 200, 1, &[("cases", &h1)], None);
        assert_eq!(level(false), literal, "off leaves the key unchanged");
        assert_ne!(level(true), literal, "the override must change the config hash");
    }

    /// The degenerate-rate setting is identity: a collapsing rate is 0 under
    /// it and fails the trajectory without it, so the two store different
    /// logliks (gh#1235). Off, the level is the literal, so no stored survey
    /// leaf re-keys.
    #[test]
    fn allow_degenerate_rates_rekeys_the_config_level() {
        let h1 = h(1);
        let data: Vec<(&str, &str)> = vec![("cases", h1.as_str())];
        let fixed: Vec<(&str, f64)> = Vec::new();
        let level = |allow_degenerate_rates| canonical_config_hash(&SurveyConfigLevel {
            eval_method: "pfilter", eval_particles: 200, eval_replicates: 1,
            data: &data, fixed: &fixed, scenario: None,
            objective: crate::fit::config_v2::LikelihoodObjective::Full, training_rows: None,
            allow_forcing_extrapolation: false, allow_degenerate_rates,
        }, &[]).unwrap();
        let literal = config_level("pfilter", 200, 1, &[("cases", &h1)], None);
        assert_eq!(level(false), literal, "off leaves the key unchanged");
        assert_ne!(level(true), literal, "the setting must change the config hash");
    }

    /// The eval count knobs are identity-bearing — `eval_method`,
    /// `eval_particles`, `eval_replicates` each change the stored landscape,
    /// so each must change the `config` hash (the count-in-the-key class).
    #[test]
    fn eval_config_participates_via_config() {
        let h1 = h(1);
        let base = config_level("pfilter", 200, 1, &[("cases", &h1)], None);
        assert_ne!(base, config_level("simulate", 200, 1, &[("cases", &h1)], None),
            "eval_method must change the config hash");
        assert_ne!(base, config_level("pfilter", 5000, 1, &[("cases", &h1)], None),
            "eval_particles must change the config hash");
        assert_ne!(base, config_level("pfilter", 200, 3, &[("cases", &h1)], None),
            "eval_replicates must change the config hash");
        assert_ne!(base, config_level("pfilter", 200, 1, &[("cases", &h(2))], None),
            "data must change the config hash");
        assert_eq!(base, config_level("pfilter", 200, 1, &[("cases", &h1)], None), "stable");
    }

    /// The LHS box distinguishes leaves: different bounds or n_points → a
    /// distinct `box` hash; identical is stable + order-independent.
    #[test]
    fn box_distinguishes_bounds_and_n_points() {
        let a = box_level(&[("beta", 0.1, 0.9)], 100);
        assert_ne!(a, box_level(&[("beta", 0.1, 0.5)], 100), "different bounds → distinct box");
        assert_ne!(a, box_level(&[("beta", 0.1, 0.9)], 200), "different n_points → distinct box");
        assert_eq!(a, box_level(&[("beta", 0.1, 0.9)], 100), "stable");
        assert_eq!(
            box_level(&[("beta", 0.1, 0.9), ("gamma", 0.0, 1.0)], 100),
            box_level(&[("gamma", 0.0, 1.0), ("beta", 0.1, 0.9)], 100),
            "bound order must not change the box hash"
        );
    }

    /// The `seed` level hashes the resolved seed.
    #[test]
    fn seed_level_distinguishes() {
        let s7 = Seed { process_seed: 7, base_seed: 7 }.content_hash();
        let s8 = Seed { process_seed: 8, base_seed: 8 }.content_hash();
        assert_ne!(s7, s8, "distinct resolved seeds must hash distinctly");
    }
}
