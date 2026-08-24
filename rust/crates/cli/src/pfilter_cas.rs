//! gh#147 (M3.3). The pfilter-eval CAS identity: map a resolved standalone
//! particle-filter evaluation into the `runid` factored levels
//! (`model` / `config` / `params` / `seed`) and its leaf `run_id`.
//!
//! A `camdl pfilter` eval *scores* a model at fixed params against observed
//! data — it does not estimate. It is a single leaf, no grid: the
//! `--replicates` are averaged within the leaf (loglik mean ± sd), not an
//! axis, so there is no cross-cell roll-up.
//!
//! Factoring: the observed `data` digests fold into the `config` level (the
//! scoring setup — particles + dt + obs-block + flow selection + the data the
//! loglik is computed against), keeping the `model` level the pure model IR.
//! Mirrors [`crate::resolve`] (sim) and [`crate::profile_cas`].

use runid::inputs::{EngineVersion, ModelDigest, Seed};
use runid::{
    run_id, ArtifactKind, ContentAddressed, ContentHash, LevelId,
};

use crate::fit::cas::canonical_config_hash;

/// A fully-resolved pfilter-eval leaf: the four factored levels (in path
/// order) and the leaf `run_id` composed from their hashes.
pub struct ResolvedPfilter {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// Inputs to [`resolve_pfilter`], all resolved by the caller.
pub struct PfilterCtx<'a> {
    pub model: &'a ir::Model,
    pub ir_version: &'a str,
    pub engine_version: &'a str,
    /// Provenance label for the `model` path segment (the model stem).
    pub stem: &'a str,
    /// `(stream name, sha256-hex)` — already-computed data digests (the same
    /// function as [`ContentHash::digest_bytes`], so `from_hex` reproduces the
    /// file-byte digest without a re-read).
    pub data: &'a [(String, String)],
    /// The resolved `(param, value)` point being scored.
    pub params: &'a [(String, f64)],
    pub particles: u32,
    /// Number of independent PF replicates averaged into the stored loglik.
    /// Identity-bearing: the replicate count changes the stored value, so it
    /// folds into the `config` level (cf. the n_trajectories collision class).
    pub replicates: u32,
    pub dt: f64,
    /// Obs-block override name. Vestigial since the `--flow` / `--obs-model`
    /// override was removed (projections now always come from `observations {}`);
    /// always `""`. Retained in the hashed context so a pre-removal `--flow`-free
    /// `run_id` is unchanged (removing the field would re-key every pfilter leaf).
    /// The resolved conditioning spec in force (`--condition-from`, else the
    /// `--fit` toml's `condition_from`), or `None` when nothing conditions.
    /// Identity-bearing: the window decides WHICH observations are scored, so
    /// it changes the stored loglik (2026-08-23 audit). The raw spec is
    /// hashed rather than the per-stream resolved times because the inputs
    /// that resolve it — the data bytes, `dt` and the model — are already in
    /// this leaf's identity, so spec + those pin the window uniquely.
    /// `None` is omitted from the config blob entirely, so an unconditioned
    /// pfilter keys exactly as before this field existed.
    pub condition_from: Option<&'a crate::fit::config_v2::ConditionFrom>,
    pub obs_block: &'a str,
    /// Flow-override transition indices. Vestigial (see `obs_block`); always
    /// empty. Kept so the content-addressed `run_id` of a `--flow`-free run is
    /// bit-stable across the override's removal.
    pub flow_indices: &'a [u32],
    /// The resolved pfilter seed (the `seed` level hashes this).
    pub seed: u64,
}

use crate::fit::cas::{data_digests, level};

/// The `config` level's identity: the scoring setup a pfilter log-likelihood
/// is computed under.
///
/// A STRUCT, not a `json!` literal, so the level is include-by-default: add a
/// field here and it is hashed, where the literal this replaced hashed only
/// what someone remembered to list (that omission is how `--condition-from`
/// stayed out of the key). Field names and shapes reproduce the literal
/// exactly, so existing pfilter leaves keep their `run_id`s — pinned by
/// `config_level_is_byte_identical_to_the_literal_it_replaced`.
#[derive(serde::Serialize)]
struct PfilterConfigLevel<'a> {
    particles: u32,
    replicates: u32,
    dt: f64,
    obs_block: &'a str,
    flow_indices: &'a [u32],
    data: &'a [(&'a str, &'a str)],
    /// Omitted entirely when nothing conditions, so an unconditioned run keys
    /// exactly as it did before this field existed.
    #[serde(skip_serializing_if = "Option::is_none")]
    condition_from: Option<&'a crate::fit::config_v2::ConditionFrom>,
}

/// Resolve a pfilter-eval leaf's identity: the four factored levels and the
/// `run_id` derived from their hashes.
pub fn resolve_pfilter(ctx: &PfilterCtx) -> Result<ResolvedPfilter, String> {
    // The scored point. Sorted for order-independence; finiteness-gated
    // before hashing (serde_json silently nulls non-finite floats, which
    // would collide distinct points).
    let mut params_sorted: Vec<(&str, f64)> =
        ctx.params.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    params_sorted.sort_by(|a, b| a.0.cmp(b.0));
    // The `params` level is the sorted (name, value) list itself — a
    // sequence, with no field set to forget, so it needs no wrapper struct
    // (and wrapping it would re-key: an object is not an array).

    // The scoring setup, with the observed data folded in (guardrail: the
    // model level stays the pure IR; the data the loglik is computed against
    // lives here). Validate the data hashes are well-formed.
    let _ = data_digests(ctx.data)?;
    let mut data_sorted: Vec<(&str, &str)> =
        ctx.data.iter().map(|(n, h)| (n.as_str(), h.as_str())).collect();
    data_sorted.sort_by(|a, b| a.0.cmp(b.0));
    let config_level = PfilterConfigLevel {
        particles: ctx.particles,
        replicates: ctx.replicates,
        dt: ctx.dt,
        obs_block: ctx.obs_block,
        flow_indices: ctx.flow_indices,
        data: &data_sorted,
        condition_from: ctx.condition_from,
    };

    let model_digest = ModelDigest::from_model(
        ctx.model,
        ctx.ir_version.to_string(),
        EngineVersion(ctx.engine_version.to_string()),
    );
    let seed = Seed { process_seed: ctx.seed, base_seed: ctx.seed };

    let config_label = format!("pf-N{}-r{}-dt{}", ctx.particles, ctx.replicates, fmt_dt(ctx.dt));

    let levels = vec![
        level("model", ctx.stem, model_digest.content_hash()),
        level("config", &config_label, canonical_config_hash(&config_level, &[])?),
        level("params", "params", canonical_config_hash(&params_sorted, &[])?),
        level("seed", &format!("seed_{}", ctx.seed), seed.content_hash()),
    ];
    let level_hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    let rid = run_id(ArtifactKind::Pfilter, &level_hashes);
    Ok(ResolvedPfilter { levels, run_id: rid })
}

/// Compact `dt` rendering for the `config` segment label (`1`, `0.5`, …).
fn fmt_dt(dt: f64) -> String {
    if (dt.round() - dt).abs() < 1e-9 {
        format!("{}", dt.round() as i64)
    } else {
        format!("{}", dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::cas::{digest_value, ensure_finite};

    fn h(byte: u8) -> String {
        ContentHash::digest_bytes(&[byte]).to_hex()
    }

    // The `params` and `config` levels are `digest_value` over the same JSON
    // blobs `resolve_pfilter` builds. Unit-testing those level digests pins
    // the collision-freeness of the identity components without a full
    // `ir::Model` fixture (the `model` level is `ModelDigest::from_model`,
    // covered by the integration round-trip; mirrors `profile_cas::tests`).

    fn params_level(params: &[(&str, f64)]) -> ContentHash {
        let mut s: Vec<(&str, f64)> = params.to_vec();
        s.sort_by(|a, b| a.0.cmp(b.0));
        digest_value(&serde_json::json!(s))
    }

    fn config_level(particles: u32, replicates: u32, dt: f64, obs: &str, data: &[(&str, &str)]) -> ContentHash {
        let mut d: Vec<(&str, &str)> = data.to_vec();
        d.sort_by(|a, b| a.0.cmp(b.0));
        digest_value(&serde_json::json!({
            "particles": particles, "replicates": replicates, "dt": dt,
            "obs_block": obs, "flow_indices": Vec::<u32>::new(), "data": d,
        }))
    }

    /// The same blob with a conditioning spec folded in, mirroring
    /// `resolve_pfilter`'s insert-only-when-Some.
    fn config_level_conditioned(
        particles: u32, replicates: u32, dt: f64, obs: &str, data: &[(&str, &str)],
        cond: &crate::fit::config_v2::ConditionFrom,
    ) -> ContentHash {
        let mut d: Vec<(&str, &str)> = data.to_vec();
        d.sort_by(|a, b| a.0.cmp(b.0));
        let mut blob = serde_json::json!({
            "particles": particles, "replicates": replicates, "dt": dt,
            "obs_block": obs, "flow_indices": Vec::<u32>::new(), "data": d,
        });
        blob["condition_from"] = serde_json::to_value(cond).unwrap();
        digest_value(&blob)
    }

    /// The conditioning window decides WHICH observations are scored, so two
    /// windows produce different logliks and must not share a `run_id`
    /// (2026-08-23 audit: `--condition-from` reached the scoring but not the
    /// key, so the store kept the first window's loglik.toml for both).
    /// Crucially, an UNCONDITIONED run must key exactly as it did before the
    /// field existed — the spec is omitted from the blob entirely.
    #[test]
    fn conditioning_window_is_in_the_key_and_absence_is_hash_neutral() {
        use crate::fit::config_v2::ConditionFrom;
        let hh = h(1);
        let data = [("cases", hh.as_str())];
        let plain = config_level(100, 1, 1.0, "", &data);

        let a = config_level_conditioned(100, 1, 1.0, "", &data,
            &ConditionFrom::All("first_obs - 3 'days".into()));
        let b = config_level_conditioned(100, 1, 1.0, "", &data,
            &ConditionFrom::All("first_obs - 10 'days".into()));
        assert_ne!(a, b, "two conditioning windows must key differently");
        assert_ne!(plain, a, "conditioned must not share the unconditioned key");
        assert_eq!(a, config_level_conditioned(100, 1, 1.0, "", &data,
            &ConditionFrom::All("first_obs - 3 'days".into())),
            "the same window must be stable");

        // Hash-neutrality: the unconditioned blob has no condition_from key at
        // all, so its digest is what it was before the field was added.
        assert_eq!(plain, digest_value(&serde_json::json!({
            "particles": 100u32, "replicates": 1u32, "dt": 1.0,
            "obs_block": "", "flow_indices": Vec::<u32>::new(),
            "data": vec![("cases", hh.as_str())],
        })), "an unconditioned pfilter must key exactly as before");
    }

    /// Byte-neutrality of the struct rewrite: `PfilterConfigLevel` must digest
    /// EXACTLY as the hand-built `json!` literal it replaced, or every stored
    /// pfilter leaf silently re-keys. `digest_value` canonicalizes (sorts keys
    /// recursively), so equality holds as long as the key set and values match
    /// — this pins that rather than assuming it.
    #[test]
    fn config_level_is_byte_identical_to_the_literal_it_replaced() {
        let hh = h(1);
        let data: Vec<(&str, &str)> = vec![("cases", hh.as_str())];
        let flow: Vec<u32> = Vec::new();

        // The struct, as production builds it.
        let lvl = PfilterConfigLevel {
            particles: 100, replicates: 1, dt: 1.0,
            obs_block: "", flow_indices: &flow, data: &data,
            condition_from: None,
        };
        // The literal, verbatim from before the rewrite.
        let literal = serde_json::json!({
            "particles": 100u32,
            "replicates": 1u32,
            "dt": 1.0,
            "obs_block": "",
            "flow_indices": Vec::<u32>::new(),
            "data": vec![("cases", hh.as_str())],
        });
        assert_eq!(canonical_config_hash(&lvl, &[]).unwrap(), digest_value(&literal),
            "the struct must reproduce the literal's digest — otherwise every \
             stored pfilter leaf re-keys silently");

        // And with conditioning in force, matching the insert-when-Some form.
        let cond = crate::fit::config_v2::ConditionFrom::All("first_obs - 3 'days".into());
        let lvl_c = PfilterConfigLevel {
            particles: 100, replicates: 1, dt: 1.0,
            obs_block: "", flow_indices: &flow, data: &data,
            condition_from: Some(&cond),
        };
        let mut literal_c = literal.clone();
        literal_c["condition_from"] = serde_json::to_value(&cond).unwrap();
        assert_eq!(canonical_config_hash(&lvl_c, &[]).unwrap(), digest_value(&literal_c));
        assert_ne!(canonical_config_hash(&lvl, &[]).unwrap(),
                   canonical_config_hash(&lvl_c, &[]).unwrap(),
                   "conditioning must still change the key");
    }

    /// A non-finite scored value must be REFUSED, not hashed. `json!` maps
    /// NaN and ±Inf alike to `Null`, so a point containing either would
    /// otherwise hash identically to the other — and to a literal null —
    /// silently colliding distinct evaluations (2026-08-23 audit: the gate
    /// ran on the already-built blob and could never fire).
    #[test]
    fn non_finite_scored_values_are_refused_before_hashing() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let pt = vec![("beta", bad)];
            assert!(ensure_finite(&pt).is_err(),
                "a non-finite scored value ({bad}) must be refused");
        }
        assert!(ensure_finite(&vec![("beta", 0.3)]).is_ok(), "finite points pass");
        // The hazard the gate exists to prevent: without it these collide.
        assert_eq!(
            digest_value(&serde_json::json!(vec![("beta", f64::NAN)])),
            digest_value(&serde_json::json!(vec![("beta", f64::INFINITY)])),
            "json! nulls both — which is exactly why the gate must run on the \
             raw values, before the blob is built"
        );
    }

    /// `dt` is the config level's only float and rides the same gate.
    #[test]
    fn non_finite_dt_is_refused_before_hashing() {
        assert!(ensure_finite(&f64::NAN).is_err(), "a NaN dt must be refused");
        assert!(ensure_finite(&1.0_f64).is_ok(), "a finite dt passes");
    }

    /// The scored point distinguishes leaves: two param vectors produce
    /// distinct `params`-level hashes; an identical point is stable; and the
    /// blob is order-independent.
    #[test]
    fn scored_point_distinguishes_and_is_stable() {
        let a = params_level(&[("beta", 0.3)]);
        let b = params_level(&[("beta", 0.5)]);
        assert_ne!(a, b, "distinct scored points must produce distinct params hashes");
        assert_eq!(a, params_level(&[("beta", 0.3)]), "the same point must be stable");
        assert_eq!(
            params_level(&[("beta", 0.3), ("gamma", 0.1)]),
            params_level(&[("gamma", 0.1), ("beta", 0.3)]),
            "param order must not change the hash"
        );
    }

    /// Data, particle count, and replicate count all participate in identity
    /// via the `config` level (data is folded in, not its own level). The
    /// replicate count is the priority-zero case: `--replicates N` averages N
    /// PF runs into the stored loglik, so it MUST be in the key (the
    /// n_trajectories collision class).
    #[test]
    fn data_particles_replicates_participate_via_config() {
        let h1 = h(1);
        let h2 = h(2);
        let base = config_level(100, 1, 1.0, "cases", &[("cases", &h1)]);
        let diff_data = config_level(100, 1, 1.0, "cases", &[("cases", &h2)]);
        let diff_particles = config_level(200, 1, 1.0, "cases", &[("cases", &h1)]);
        let diff_replicates = config_level(100, 3, 1.0, "cases", &[("cases", &h1)]);
        assert_ne!(base, diff_data, "different data must change the config hash");
        assert_ne!(base, diff_particles, "different particle count must change the config hash");
        assert_ne!(base, diff_replicates,
            "different replicate count must change the config hash — the stored \
             loglik depends on it (n_trajectories collision class)");
        assert_eq!(base, config_level(100, 1, 1.0, "cases", &[("cases", &h1)]), "stable");
    }

    /// The `seed` level hashes the resolved seed.
    #[test]
    fn seed_level_distinguishes() {
        let s7 = Seed { process_seed: 7, base_seed: 7 }.content_hash();
        let s8 = Seed { process_seed: 8, base_seed: 8 }.content_hash();
        assert_ne!(s7, s8, "distinct resolved seeds must hash distinctly");
    }
}
