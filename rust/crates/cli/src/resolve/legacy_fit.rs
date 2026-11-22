//! The fit identity of the old store, kept only for the golden run-id pins
//! in `resolve/tests.rs` (`all_kind_identities`, `cas_identity_pins`), which
//! record it. No command keys a fit this way: `fit run` stores on the run
//! store ([`crate::fit_store`]). It is deleted with the fit rows of those pins.

use runid::inputs::{ArtifactRef, Deps, EngineIdentity, FitDigest, MethodConfig, MethodLevel, ModelDigest, Seed};
use runid::{run_id, ArtifactKind, ContentHash, LevelId};

use crate::fit::config_v2::{Algorithm, Method, Problem};
use crate::profile_cas::{canonical_config_hash, level, structural_level_hash};

pub(crate) struct ResolvedFitMethod {
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
}

/// The inputs of the old fit identity. The data the problem binds is held
/// at none: every pin was recorded with no bound file.
pub(crate) struct FitMethodCtx<'a> {
    pub model: &'a ir::Model,
    pub fit_stem: &'a str,
    pub ir_version: &'a str,
    pub engine: &'a EngineIdentity,
    pub problem: &'a Problem,
    pub method: &'a Method,
    pub seed: u64,
    pub deps: Vec<ArtifactRef>,
}

fn digest_value(v: &serde_json::Value) -> ContentHash {
    let bytes = serde_json::to_vec(&crate::canonical_json::canonical(v)).expect("canonical JSON serializes");
    ContentHash::digest_bytes(&bytes)
}

const FIT_PROVENANCE_KEYS: &[&str] = &["output_dir", "provenance", "data.tables", "simulate.from"];

fn target_length(a: &Algorithm) -> u64 {
    match a {
        Algorithm::IF2 { iterations, .. } => *iterations as u64,
        Algorithm::PGAS { sweeps, .. } => *sweeps as u64,
        Algorithm::PMMH { iterations, .. } => *iterations as u64,
        Algorithm::Mh { iterations, .. } => *iterations as u64,
        Algorithm::PFilter { .. } | Algorithm::NlSbplx(_) | Algorithm::NlBobyqa(_) | Algorithm::NlLbfgs(_) => 0,
    }
}

fn reads_gradients(a: &Algorithm) -> bool {
    match a {
        Algorithm::PGAS { use_nuts, .. } => *use_nuts,
        Algorithm::NlLbfgs(_) => true,
        _ => false,
    }
}

fn n_trajectories(a: &Algorithm) -> u64 {
    match a {
        Algorithm::PGAS { n_trajectories, .. } => *n_trajectories as u64,
        _ => 0,
    }
}

fn identity_payload(m: &Method) -> serde_json::Value {
    let mut exclude: Vec<&str> = m.algorithm.extension_key().into_iter().collect();
    if matches!(m.algorithm, Algorithm::PGAS { .. }) {
        exclude.push("n_trajectories");
    }
    let mut payload = crate::canonical_json::serialize_minus(m, &exclude).unwrap_or_else(|_| serde_json::json!({}));
    if let (Some(starts), Some(obj)) = (&m.starts, payload.as_object_mut()) {
        obj.insert("starts".to_string(), starts.identity_value());
    }
    payload
}

fn method_config_hash(method: &Method, gradients: Option<&ContentHash>) -> Result<ContentHash, String> {
    method.starts()?;
    crate::canonical_json::ensure_finite(method)?;
    let mut v = serde_json::json!({
        "identity": identity_payload(method),
        "n_trajectories": n_trajectories(&method.algorithm),
    });
    if let Some(g) = gradients {
        v["gradients"] = serde_json::Value::String(g.to_hex().to_string());
    }
    Ok(digest_value(&v))
}

fn fit_level_digest(model: &ir::Model, ir_version: &str, engine: &EngineIdentity, problem: &Problem) -> Result<FitDigest, String> {
    Ok(FitDigest {
        model: ModelDigest::from_model(model, ir_version.to_string(), engine.clone()),
        data: Vec::new(),
        holdout_data: Vec::new(),
        tables: Vec::new(),
        fit_toml: canonical_config_hash(problem, FIT_PROVENANCE_KEYS)?.into_inner(),
        engine: engine.clone(),
        partition_rule_version: crate::fit::training_partition::PARTITION_RULE_VERSION,
    })
}

pub(crate) fn resolve_fit_method(ctx: &FitMethodCtx) -> Result<ResolvedFitMethod, String> {
    let fit = fit_level_digest(ctx.model, ctx.ir_version, ctx.engine, ctx.problem)?;
    let gradients = reads_gradients(&ctx.method.algorithm).then(|| runid::inputs::model_ir_hash_with_gradients(ctx.model));
    let method_config = MethodConfig {
        config: method_config_hash(ctx.method, gradients.as_ref())?,
        obs_block: String::new(),
        flow_indices: Vec::new(),
        target_length: target_length(&ctx.method.algorithm),
    };
    let method_level = MethodLevel { config: method_config, deps: Deps(ctx.deps.clone()) };
    let seed = Seed { process_seed: ctx.seed, base_seed: ctx.seed };
    let levels = vec![
        level("fit", ctx.fit_stem, structural_level_hash(&fit)),
        level("method", ctx.method.algorithm.method_name(), structural_level_hash(&method_level)),
        level("seed", &format!("seed_{}", ctx.seed), structural_level_hash(&seed)),
    ];
    let hashes: Vec<ContentHash> = levels.iter().map(|l| l.hash).collect();
    Ok(ResolvedFitMethod { run_id: run_id(ArtifactKind::FitMethod, &hashes), levels })
}

/// The old store's reuse answer for a fit leaf, for the fail-closed reuse
/// test of `resolve/tests.rs`.
pub(crate) enum ReuseVerdict {
    CacheHit,
    MustRun,
}

/// Whether the old store would reuse the leaf `resolved` names under
/// `policy` for a build whose identity is `engine`: a build whose identity is
/// incomplete never reuses, and the leaf it would have read is quarantined.
pub(crate) fn check_reuse_with(
    store: &runid::FsCasStore,
    root: &std::path::Path,
    resolved: &super::ResolvedArtifact,
    policy: super::WritePolicy,
    engine: crate::version::EngineReuse,
) -> Result<ReuseVerdict, runid::CasError> {
    let policy = super::engine_write_policy(policy, engine);
    let found = runid::find_leaf(root, resolved.kind, &resolved.levels);
    if policy == super::WritePolicy::Force {
        let dir = found.clone().unwrap_or_else(|| runid::store_path(root, resolved.kind, &resolved.levels));
        store.displace(&dir, &runid::LeafIdentity::new(resolved.run_id))?;
        return Ok(ReuseVerdict::MustRun);
    }
    Ok(match found {
        Some(_) => ReuseVerdict::CacheHit,
        None => ReuseVerdict::MustRun,
    })
}
