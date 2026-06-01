//! `camdl profile` — profile likelihood via parallel IF2 runs.
//!
//! For one or more focal parameters, fix them at a grid of values and
//! run IF2 to maximise over the remaining parameters at each grid
//! point. The profile likelihood shows how the MLE changes as you move
//! the focal parameter(s) — revealing identifiability, confidence
//! intervals, and parameter interactions. 2D profiles (two `--sweep`
//! flags) produce a likelihood surface suitable for contour plotting.
//!
//! ## CAS integration (2026-04-24 rewrite)
//!
//! Every profile is laid out as a `ReplicateSet` umbrella over its
//! seeds — N=1 is the trivial case, N>1 is the IF2-stochastic-
//! sensitivity sweep. Every (grid_point × start) under each seed is
//! a cacheable mini-fit:
//!
//! ```text
//! <root>/profiles/<stem>-<umbrella_hash[:8]>/
//!   run.json                                    # RunKind::ReplicateSet { child_kind: "profile" }
//!   summary.tsv                                 # cross-seed aggregate (1 row at N=1)
//!   replicates/
//!     seed_<n>/
//!       run.json                                # RunKind::Profile (per-seed)
//!       profile.tsv                             # per-seed rollup
//!       points/
//!         {point_idx:05d}/
//!           focal.toml                          # pinned focal values
//!           start_{start_idx}/
//!             run.json                          # RunKind::FitStage
//!             mle.toml                          # MLE at this start
//! ```
//!
//! Each `start_{k}/run.json` is written atomically (tmp-then-rename);
//! crash mid-IF2 leaves no run.json and the next invocation reruns
//! that start. Completed starts are preserved bit-for-bit. The rollup
//! is rewritten atomically after every completion, so it's always
//! current-as-of-last-finished-start.
//!
//! Design: docs/dev/proposals/2026-04-24-profile-cas-integration.md.
//! Supersedes GH #15's streaming-TSV + --resume approach.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        if2::{run_if2, IF2Config, Observation},
        pmmh::{run_pmmh, PMMHConfig, Prior},
        ChainBinomialProcess, MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
    },
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::cas::typed::{
    self, CasInputs, ContentHash, ReplicateSet, hash_canonical,
};
use crate::run_meta::{FitStageMeta, GridAxis, ProfileMeta, Run, RunKind, RunStatus};
use crate::run_paths::{
    output_root, profile_point_dir, profile_point_start_dir,
};

/// Per-cell optimizer choice. `--algorithm if2 --backend chain_binomial`
/// (the default) keeps the historical PF-based per-cell MLE; the new
/// `--algorithm nl-* --backend ode` paths run NLopt deterministic MLE
/// against `compute_ode_loglik`. See gh#47 for the dispatch rationale
/// and the proposal for the (algorithm, backend) matrix.
#[derive(Debug, Clone, Copy)]
enum ProfileAlgo {
    If2,
    Pmmh,
    Nlopt(sim::inference::deterministic::NloptAlgorithm),
}

impl ProfileAlgo {
    fn method_kind(self) -> crate::run_meta::MethodKind {
        match self {
            ProfileAlgo::If2  => crate::run_meta::MethodKind::If2,
            ProfileAlgo::Pmmh => crate::run_meta::MethodKind::Pmmh,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx) =>
                crate::run_meta::MethodKind::NlSbplx,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa) =>
                crate::run_meta::MethodKind::NlBobyqa,
        }
    }
    fn backend(self) -> crate::run_meta::Backend {
        match self {
            ProfileAlgo::If2      => crate::run_meta::Backend::ChainBinomial,
            ProfileAlgo::Pmmh     => crate::run_meta::Backend::ChainBinomial,
            ProfileAlgo::Nlopt(_) => crate::run_meta::Backend::Ode,
        }
    }
}

// Observation family resolution lives in `crate::util::resolve_data_specs`
// (gh#90). Profile's previous `resolve_obs_family` (gh#38) is subsumed
// there: `--data PATH --obs <family-root>` matches every IR obs whose
// name starts with `<root>_`, and `--data NAME=PATH` (gh#90 named form)
// also expands NAME as a family root. Same semantics, single dispatch.

// ─── ProfileInputs ───────────────────────────────────────────────────────────

/// Typed CAS inputs for a single-realization profile run. The struct
/// carries every content-bearing input (model, base params, focal
/// grid, fixed list, IF2 hyperparams, starts_from lineage, seed) plus
/// presentation hints (model_path, stem). Ephemeral inputs (parallel,
/// progress, output mirror) live on `ProfileArgs` and don't appear here.
///
/// `inner_hash` excludes seed and is the umbrella hash for a multi-seed
/// `ReplicateSet`. `content_hash` (the trait method) includes seed via
/// `compose_with_replicate(inner_hash, "seed", seed)` — so a standalone
/// `--seed N` invocation and one child of a `--seeds 1,N,...` set hit
/// the same cache key.
#[derive(Clone, Debug)]
pub struct ProfileInputs {
    /// Display-only model path. Recorded in `ProfileMeta.model`.
    pub model_path: String,
    /// Slugified stem from the model path; used as the `<stem>-<hash>`
    /// directory prefix.
    pub stem: Option<String>,
    /// Full SHA-256 of the IR JSON.
    pub model_hash: String,
    /// Canonical-form hash of the base parameter vector.
    pub base_params_hash: String,
    /// Per-stream SHA-256 of each bound `--data` file's bytes. gh#90
    /// extends the gh#39 single-file hash to multi-stream: every
    /// (stream_name, content_hash) pair the resolver bound participates
    /// in the cache key. Sorted by stream name before hashing for
    /// deterministic order independent of how the user spelled
    /// `--data NAME=PATH` on the command line.
    ///
    /// Content-only — paths are not part of these hashes, so two users
    /// with the same TSVs at different filesystem locations share a
    /// cache entry, while two users with different TSVs at the same
    /// path do not. gh#39: editing any bound stream's data file in
    /// place must invalidate the cache.
    pub data_hashes: Vec<(String, String)>,
    /// Focal grid (one axis per `--sweep` flag).
    pub focal_grid: Vec<GridAxis>,
    /// Fixed parameters (`--fixed`): excluded from IF2 estimation. Order
    /// doesn't matter; sorted before hashing.
    pub fixed: Vec<String>,
    /// `--obs <NAME>` argument as resolved against the IR. Either an
    /// exact stream name (single-stream profile) or a family root that
    /// expanded to N>1 concrete streams (joint multi-stream profile).
    /// Empty string when the model has exactly one observation and
    /// `--obs` was omitted. gh#38: this **must** be in the cache key —
    /// switching `--obs cases` ↔ `--obs cases_p1` changes the loglik
    /// scale by orders of magnitude (5 streams summed vs 1).
    pub obs_family: String,
    /// IF2 hyperparameter set.
    pub if2_config: ProfileIf2Config,
    /// gh#89: per-cell algorithm name (`if2`, `pmmh`, `nl-sbplx`, …).
    /// Resolved against the methods registry at dispatch time. Part of
    /// the cache key because switching algorithms with otherwise
    /// identical inputs (same particles, iterations, etc.) is a real
    /// content change — re-running `--algorithm if2 → pmmh` MUST
    /// produce a fresh cache entry, not return the IF2 result.
    pub algorithm: String,
    /// gh#89: PMMH steps per cell (`--pmmh-steps`). Bumping this is
    /// the canonical "give it more budget" knob — must invalidate the
    /// cache so the higher-budget run actually computes.
    pub pmmh_steps: usize,
    /// gh#89: PMMH particles per PF evaluation (`--pmmh-particles`).
    /// Same rationale as `pmmh_steps`.
    pub pmmh_particles: usize,
    /// gh#89: Crank-Nicolson correlation for CPM-MCMC (`--pmmh-rho`).
    /// `None` = vanilla PMMH (rho ≤ 0 at the CLI); `Some(r)` for
    /// 0 < r < 1. Affects MCMC mixing dynamics, so toggling on/off
    /// or changing the value is a content distinction.
    pub pmmh_rho: Option<f64>,
    /// Hash of an upstream stage's content this profile starts from.
    /// `None` for standalone profile invocations.
    pub starts_from_lineage: Option<String>,
    /// SHA-256 of the `--fit <toml>` file's bytes (gh#73). Part of
    /// the CAS key so re-running with a different fit toml against
    /// the same model produces a different cache dir.
    pub fit_toml_hash: Option<String>,
    /// Per-parameter prior-resolution audit (gh#73). Recorded into
    /// `run.json` via `ProfileMeta.resolved_priors`; included in the
    /// hash so that switching the prior source for any parameter
    /// (e.g. user adds a `~` to the model file) invalidates the
    /// cached profile.
    pub resolved_priors: Vec<(String, String)>,
    /// Diagnostic warnings the user suppressed (gh#73). Recorded but
    /// NOT part of the CAS hash — suppression is metadata, not a
    /// content distinction; two otherwise-identical profiles should
    /// hit the same cache regardless of which one suppressed the
    /// warning.
    pub suppressed_warnings: Vec<String>,
    /// Per-seed: the actual seed value. `inner_hash` excludes this;
    /// `content_hash` (trait method) includes it.
    pub seed: u64,
    /// gh#83/gh#85 step 9: per-parameter resolver provenance for the
    /// profile-level run.json. NOT part of the CAS hash — provenance
    /// is metadata about *how* a run was specified, not what was
    /// computed; identical content produces identical CAS keys
    /// regardless of which `--fixed-file` shape the user used.
    pub parameters_provenance: std::collections::HashMap<
        String, crate::run_meta::ParameterProvenance>,
    /// gh#83/gh#85 step 9: per-start init provenance. NOT part of the
    /// CAS hash; the per-cell `FitStage` children carry their own.
    pub init_provenance: Option<crate::run_meta::InitProvenance>,
}

#[derive(Clone, Debug)]
pub struct ProfileIf2Config {
    pub n_particles:  usize,
    pub n_iterations: usize,
    pub cooling:      f64,
    pub dt:           f64,
    pub n_starts:     usize,
}

impl ProfileInputs {
    /// Hash of all content fields *except* seed. Used as the
    /// inner_hash of a `ReplicateSet` umbrella when running multi-seed.
    pub fn inner_hash(&self) -> ContentHash {
        let grid_canonical = serde_json::to_string(&self.focal_grid).unwrap_or_default();
        let mut fixed_sorted = self.fixed.clone();
        fixed_sorted.sort();
        let if2 = format!(
            "particles={};iterations={};cooling={};dt={};starts={}",
            self.if2_config.n_particles, self.if2_config.n_iterations,
            self.if2_config.cooling, self.if2_config.dt, self.if2_config.n_starts,
        );
        // gh#89: the PMMH-specific budget knobs. Encoded as a single
        // canonical string so the canonical-keys vec stays flat. `rho`
        // is serialised as either `off` (None) or its f64 repr so
        // toggling the CPM mode invalidates the cache.
        let pmmh = format!(
            "steps={};particles={};rho={}",
            self.pmmh_steps, self.pmmh_particles,
            self.pmmh_rho.map(|r| r.to_string()).unwrap_or_else(|| "off".into()),
        );
        // gh#73: canonicalize the resolved-prior table for hashing.
        // Sort by param name so the order the resolver emitted them
        // (declaration order today) doesn't leak into the cache key —
        // adding a new estimated parameter is a real change, but
        // re-ordering an existing list is not.
        let mut priors_sorted = self.resolved_priors.clone();
        priors_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let priors_canonical: String = priors_sorted.iter()
            .map(|(n, s)| format!("{}={}", n, s))
            .collect::<Vec<_>>().join(",");
        // gh#90: every bound stream contributes its (name,
        // content_hash) pair. Sort by stream name so CLI ordering
        // (`--data cases=... --data deaths=...` vs `--data deaths=...
        // --data cases=...`) doesn't move the cache key, and so two
        // profiles with the same N streams in any order hit the same
        // entry. Concatenated with a NUL separator that can't appear
        // in a stream name to avoid `cases:hashA,deaths` colliding
        // with `cases:hashA-deaths:` style sneaky merges.
        let mut data_pairs_sorted = self.data_hashes.clone();
        data_pairs_sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let data_canonical: String = data_pairs_sorted.iter()
            .map(|(n, h)| format!("{}={}", n, h))
            .collect::<Vec<_>>().join("\x00");
        hash_canonical(&[
            ("model",       &self.model_hash),
            ("base_params", &self.base_params_hash),
            ("focal_grid",  &grid_canonical),
            ("fixed",       &fixed_sorted.join(",")),
            ("obs_family",  &self.obs_family),
            ("algorithm",   &self.algorithm),
            ("if2",         &if2),
            ("pmmh",        &pmmh),
            ("starts_from", self.starts_from_lineage.as_deref().unwrap_or("")),
            ("data",        &data_canonical),
            ("fit_toml",    self.fit_toml_hash.as_deref().unwrap_or("")),
            ("priors",      &priors_canonical),
        ])
    }
}

impl CasInputs for ProfileInputs {
    fn content_hash(&self) -> ContentHash {
        // Per-seed leaf hash. Composes with `seed` so the same value
        // is obtained whether the run was invoked standalone or as
        // one child of a multi-seed ReplicateSet.
        typed::compose_with_replicate(
            &self.inner_hash(), "seed", &self.seed.to_string(),
        )
    }

    fn cas_path(&self, root: &Path) -> PathBuf {
        let h = self.content_hash();
        let dirname = match &self.stem {
            Some(s) if !s.is_empty() => format!("{}-{}", s, h.short()),
            _ => h.short().to_string(),
        };
        root.join("profiles").join(dirname)
    }

    fn run_kind(&self) -> RunKind {
        let total_jobs = self.focal_grid.iter()
            .map(|g| g.values.len()).product::<usize>()
            * self.if2_config.n_starts;
        // The if2_config_hash and base_params_hash fields on
        // ProfileMeta are diagnostic; ProfileInputs.content_hash() is
        // the authoritative cache key. Keeping the meta fields for
        // human inspection in `camdl show`.
        let if2_canonical = format!(
            "particles={};iterations={};cooling={};dt={};starts={}",
            self.if2_config.n_particles, self.if2_config.n_iterations,
            self.if2_config.cooling, self.if2_config.dt, self.if2_config.n_starts,
        );
        let if2_config_hash = ContentHash::from_bytes(if2_canonical.as_bytes())
            .full().to_string();
        RunKind::Profile(ProfileMeta {
            model:            self.model_path.clone(),
            model_hash:       self.model_hash.clone(),
            focal_params:     self.focal_grid.iter().map(|g| g.param.clone()).collect(),
            grid:             self.focal_grid.clone(),
            n_starts:         self.if2_config.n_starts,
            if2_config_hash,
            base_params_hash: self.base_params_hash.clone(),
            seed_base:        self.seed,
            total_jobs,
            fit_toml_hash:    self.fit_toml_hash.clone(),
            resolved_priors:  self.resolved_priors.iter().map(|(n, s)| {
                crate::run_meta::ResolvedPriorEntry {
                    param: n.clone(), source: s.clone(),
                }
            }).collect(),
            suppressed_warnings: self.suppressed_warnings.clone(),
            // gh#83/gh#85 step 9: provenance threaded in post-CAS
            // by the run-finalization layer (which has the
            // `ResolvedParameters` / `ChainStarts` in scope).
            parameters_provenance: self.parameters_provenance.clone(),
            init_provenance: self.init_provenance.clone(),
        })
    }
}

pub fn cmd_profile(a: &crate::args::ProfileArgs) {
    crate::args::apply_pf_wallclock_env(&a.inference);  // gh#133
    // Validate (algorithm, backend) early, before any expensive setup.
    let algo_name = a.algorithm.as_deref().unwrap_or("if2");
    let backend_name = a.backend.as_deref().unwrap_or("chain_binomial");
    if let Err(msg) = crate::fit::methods::validate_combo(algo_name, backend_name) {
        eprintln!("error: {}", msg);
        std::process::exit(1);
    }
    let profile_algo = match algo_name {
        "if2"       => ProfileAlgo::If2,
        "pmmh"      => ProfileAlgo::Pmmh,
        "nl-sbplx"  => ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx),
        "nl-bobyqa" => ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa),
        other => {
            eprintln!(
                "error: --algorithm = \"{}\" is not yet supported for `camdl profile`. \
                 Currently supported: if2 (chain_binomial), pmmh (chain_binomial), \
                 nl-sbplx (ode), nl-bobyqa (ode).",
                other
            );
            std::process::exit(1);
        }
    };
    // PMMH on profile defaults to chain_binomial (matches `fit run --algorithm pmmh`).
    // The `pmmh + ode` combination isn't supported on the profile path; reject early.
    if matches!(profile_algo, ProfileAlgo::Pmmh) && backend_name == "ode" {
        eprintln!(
            "error: --algorithm pmmh requires --backend chain_binomial. \
             PMMH wraps a particle filter inside an MH step; under the ODE \
             backend the PF wrapping is degenerate (1-particle, exact) and \
             the algorithm collapses to vanilla MH. Re-run with \
             `--backend chain_binomial`."
        );
        std::process::exit(1);
    }

    let ir_path = a.model.to_string_lossy().into_owned();
    let n_particles = a.inference.particles;
    let n_iterations = a.iterations;
    let n_starts = a.starts;
    let cooling = a.cooling;
    let dt = a.inference.dt;
    let seed_base = a.inference.seed;
    let parallel = a.inference.parallel;
    // PMMH per-cell knobs (ignored when --algorithm is not pmmh).
    let pmmh_steps = a.pmmh_steps;
    let pmmh_particles = a.pmmh_particles;
    // Per spec: `--pmmh-rho 0.0` (or any non-positive) disables CPM.
    let pmmh_rho_opt: Option<f64> = if a.pmmh_rho > 0.0 {
        if !(a.pmmh_rho < 1.0) {
            eprintln!(
                "error: --pmmh-rho = {} must be in [0, 1). Use 0.0 (or negative) \
                 to disable CPM and run vanilla PMMH.",
                a.pmmh_rho
            );
            std::process::exit(1);
        }
        Some(a.pmmh_rho)
    } else {
        None
    };
    let output_tsv_path: Option<String> = a.output.as_ref().map(|p| p.to_string_lossy().into_owned());
    let scenario_name = a.scenario.scenario.clone();
    let flow_name = a.flow.flow.clone();
    let label_arg: Option<String> = match a.label.as_deref() {
        Some(raw) => match crate::fit::validate_label(raw) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("error: invalid --label: {}", e);
                std::process::exit(1);
            }
        },
        None => None,
    };
    // M-1 break per docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md
    // §"Migration": fail loudly on removed flags before any work.
    a.model_overrides.check_removed_flags("profile");
    // `--fixed NAME=VALUE` → `fixed_cli`, `--fixed-file <toml>`
    // (repeatable, layered) → `fixed_files`. Both feed into the
    // unified resolver; the fit.toml [fixed] block is still
    // pre-processed via FixedParams::expand_from_scenario +
    // resolve_with_model and supplied as `fit_toml_fixed`.
    let fixed_files_vec: Vec<std::path::PathBuf> = a.model_overrides.fixed_files.clone();
    let fixed_cli_vec: Vec<(String, f64)> = a.model_overrides.fixed_cli.iter()
        .map(|p| (p.name.clone(), p.value))
        .collect();
    let _overrides: HashMap<String, f64> = fixed_cli_vec.iter().cloned().collect();
    // Construct the full InitMethod (with payload) from the CLI tag
    // + companion path flags. This is the post-parse step that turns
    // `--init from_posterior --posterior <path>` into a typed
    // `InitMethod::FromPosterior { source: ... }`.
    let init_method: crate::fit::init::InitMethod = a.init.to_init_method(
        a.posterior.as_ref(),
        a.mle.as_ref(),
        a.init_params.as_ref(),
    ).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });

    let focal_names: Vec<String> = a.sweep.iter().map(|s| s.name.clone()).collect();

    struct FocalGrid { name: String, values: Vec<f64>, param_idx: usize }
    let mut focal_grids: Vec<FocalGrid> = Vec::new();

    let rw_sd = a.rw_sd.as_ref().unwrap_or_else(|| {
        eprintln!("error: --rw-sd required (e.g., --rw-sd \"sigma=0.01\" or --rw-sd auto)");
        std::process::exit(1);
    });
    let rw_sd_auto = matches!(rw_sd, crate::args::types::RwSd::Auto);
    let rw_sd_map_raw: HashMap<String, Option<f64>> = match rw_sd {
        crate::args::types::RwSd::Auto => HashMap::new(),
        crate::args::types::RwSd::Map(m) => m.clone(),
    };

    // Load model (pre-resolution). `mut` because the gh#34
    // `[estimate].start` fall-back below seeds values into
    // `model_pre.parameters[i].value` before the resolver call —
    // same pattern as `fit/runner.rs:144-180`.
    let (mut model_pre, model_json) = crate::util::load_model(&ir_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // ── Optional --fit toml resolution (gh#73) ──────────────────────
    //
    // When `--fit <path.toml>` is supplied, the fit-toml is the source
    // of truth for priors, bounds, and the default fixed list — same
    // shape `camdl fit run` reads. The toml's `[fixed]` block is
    // pre-processed against the model so scenario-driven fixed values
    // come through; the resulting IndexMap is fed to the unified
    // resolver as `fit_toml_fixed` (per 2026-05-25 CLI UX rev 2).
    //
    // The estimate IndexMap returned here is the canonical prior
    // source used downstream by `resolve_priors_with_precedence`. When
    // `--fit` is absent the map is empty and the resolver falls
    // through to model-IR priors (tier 2) for every parameter.
    let (fit_estimate, fit_toml_hash, fit_toml_fixed_indexmap):
        (indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
         Option<String>,
         indexmap::IndexMap<String, f64>) = if let Some(fit_path) = a.fit.as_ref() {
        let fit_path_str = fit_path.to_string_lossy().into_owned();
        let fit_cfg = crate::fit::config_v2::FitConfigV2::load(&fit_path_str)
            .unwrap_or_else(|e| {
                eprintln!("error: failed to load --fit toml '{}': {}",
                    fit_path_str, e);
                std::process::exit(1);
            });
        // Resolve [fixed] (file load, scenario, inline overlay) the
        // same way `camdl survey --fit` and `camdl fit run` do.
        let mut fixed_cfg = fit_cfg.fixed.clone();
        fixed_cfg.expand_from_scenario(&model_pre)
            .unwrap_or_else(|e| {
                eprintln!("error: --fit toml [fixed].expand_from_scenario: {}", e);
                std::process::exit(1);
            });
        let fixed_resolved = fixed_cfg.resolve_with_model(&model_pre)
            .unwrap_or_else(|e| {
                eprintln!("error: --fit toml [fixed].resolve_with_model: {}", e);
                std::process::exit(1);
            });
        // Hash the bytes for provenance. Path-independent — only the
        // contents participate, matching the `data_hash` convention.
        let bytes = std::fs::read(fit_path)
            .unwrap_or_else(|e| {
                eprintln!("error: cannot read --fit toml '{}': {}",
                    fit_path_str, e);
                std::process::exit(1);
            });
        let hash = crate::hashing::sha256_hex(&bytes);
        eprintln!("profile: using --fit '{}' for priors / bounds / [fixed]",
            fit_path_str);
        (fit_cfg.estimate, Some(hash), fixed_resolved)
    } else {
        (indexmap::IndexMap::new(), None, indexmap::IndexMap::new())
    };

    // Init-mode → resolver bridge: `--init from_params --params <toml>`
    // and `--init from_mle --mle <path>` both load a single-point
    // parameter file. The user's mental model is that the file's
    // values are authoritative for the parameters named in it — both
    // as the resolver's base AND as the chain starting point.
    // Without this seeding, the resolver fires `UnsetRequired` on any
    // parameter that has no DSL default + no `[fixed]` entry, even
    // when the user has explicitly named a file containing the value.
    // See `seed_params_from_init_method` for the design rationale.
    seed_params_from_init_method(&mut model_pre.parameters, &init_method)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // gh#34 [estimate].start fall-back. Apply BEFORE the resolver so
    // that params listed in fit-toml `[estimate]` with an explicit
    // `start = ...` carry that value past the resolver's
    // `UnsetRequired` check, exactly the way `camdl fit run` does
    // (see `fit/runner.rs:144-180`). Without this, a user writing
    //   [estimate]
    //   beta = { bounds = [0.01, 5.0], start = 0.5143 }
    // and no DSL default for `beta` would hit the resolver's
    // `UnsetRequired` error reading "no model default, no scenario,
    // no --fit toml entry, ..." — the [estimate].start was silently
    // ignored. Scope: explicit `spec.start` only (no
    // bounds-uniform fallback for profile yet — the focal sweep
    // overrides the focal param's value per-cell anyway, and the
    // non-focal params are usually pinned via `--fixed`).
    for (name, spec) in &fit_estimate {
        if let Some(p) = model_pre.parameters.iter_mut().find(|p| p.name == *name) {
            if p.value.is_none() {
                if let Some(start) = spec.start {
                    p.value = Some(start);
                }
            }
        }
    }

    // Run the unified resolver.
    let fit_toml_estimate: indexmap::IndexSet<String> =
        fit_estimate.keys().cloned().collect();
    let table_files_resolver: HashMap<String, std::path::PathBuf> = HashMap::new();
    let resolved = crate::params_resolver::resolve_parameters(
        crate::params_resolver::ParameterInputs {
            model: &model_pre,
            scenario: scenario_name.as_deref(),
            adhoc_enable: &a.scenario.enable,
            adhoc_disable: &a.scenario.disable,
            fixed_cli: &fixed_cli_vec,
            fixed_files: &fixed_files_vec,
            fit_toml_fixed: &fit_toml_fixed_indexmap,
            fit_toml_estimate: &fit_toml_estimate,
            table_files: &table_files_resolver,
        },
    ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    crate::params_resolver::print_warnings(&resolved);
    let model = resolved.model.clone();

    // Maintain the downstream `fit_toml_fixed: HashMap` view that the
    // rest of profile.rs reads (focal-vs-fixed conflict check, etc.).
    let fit_toml_fixed: HashMap<String, f64> = fit_toml_fixed_indexmap
        .iter().map(|(k, &v)| (k.clone(), v)).collect();

    let compiled = Arc::new(CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { eprintln!("{:?}", e); std::process::exit(1); }));
    let base_params = compiled.default_params.clone();

    // Reject overdispersed models on the ODE backend (and any other
    // backend / model-capability mismatch). The `validate_combo` call
    // above is structural-only; this check sees the actual model.
    if let Err(msg) = crate::fit::methods::check_model_capabilities(
        backend_name, &compiled,
    ) {
        eprintln!("error: {}", msg);
        std::process::exit(1);
    }

    // ── Resolve --data and --obs against the IR's observation list ──
    //
    // gh#90: polymorphic `--data` is the primary multi-stream surface.
    //   --data PATH         → single-stream, optionally narrowed by --obs.
    //   --data NAME=PATH    → multi-stream, joint scoring of every bound
    //                         stream (repeatable; one flag per stream).
    //
    // gh#38 family resolution survives: in the single-PATH form,
    // `--obs <root>` matches every IR obs whose name starts with
    // `<root>_`, so a single wide TSV covers an indexed
    // `cases[s,a]` family with one `--data <file>` flag plus
    // `--obs cases`.
    //
    // fit-toml fallback: when no `--data` flags supplied AND `--fit`
    // is, read `[data]` / `[data.observations]` from the toml. CLI
    // flags always win when both are supplied.
    let obs_name_arg = a.flow.obs.clone();
    let model_obs_names: Vec<String> = model.observations.iter()
        .map(|o| o.name.clone()).collect();
    let cli_data_specs: Vec<crate::args::types::DataSpec> = a.data.clone();
    let bound_streams: Vec<(String, std::path::PathBuf)> = if cli_data_specs.is_empty() {
        if let Some(fit_path) = a.fit.as_ref() {
            eprintln!("profile: no --data flags supplied, reading bindings \
                from --fit toml [data.observations]");
            crate::pfilter::load_data_observations_from_fit_toml(
                fit_path.as_path(), &model_obs_names,
            ).unwrap_or_else(|e| {
                eprintln!("error: --fit toml fallback for --data: {}", e);
                std::process::exit(1);
            })
        } else {
            eprintln!("error: --data is required. Use `--data PATH` for a \
                single-stream model, `--data NAME=PATH` (repeatable) for a \
                multi-stream model, or `--fit FOO.toml` with a \
                [data.observations] section.");
            std::process::exit(1);
        }
    } else {
        if a.fit.is_some() {
            eprintln!("profile: --data on CLI overrides --fit toml [data.observations]");
        }
        crate::util::resolve_data_specs(&cli_data_specs, &model_obs_names, obs_name_arg.as_deref())
            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
    };

    if bound_streams.is_empty() {
        eprintln!("error: zero streams resolved from --data / --fit toml.");
        std::process::exit(1);
    }

    // Resolved IR observation models, one per bound stream. Family
    // expansion happened inside the resolver; here every bound name
    // is an exact leaf match.
    let resolved_obs: Vec<&ir::observation::ObservationModel> = {
        let mut v = Vec::with_capacity(bound_streams.len());
        for (sname, _) in &bound_streams {
            match model.observations.iter().find(|o| &o.name == sname) {
                Some(o) => v.push(o),
                None => {
                    eprintln!("error: bound stream '{}' has no matching IR \
                        observation block (resolver bug). Available: {}",
                        sname,
                        model.observations.iter().map(|o| o.name.as_str())
                            .collect::<Vec<_>>().join(", "));
                    std::process::exit(1);
                }
            }
        }
        v
    };

    if resolved_obs.len() > 1 && flow_name.is_some() {
        eprintln!(
            "error: --flow <NAME> is incompatible with multi-stream --data \
             ({} streams bound). --flow rewrites a single stream's projection; \
             for multi-stream profile each stream uses its own IR projection.",
            resolved_obs.len(),
        );
        std::process::exit(1);
    }

    if resolved_obs.len() > 1 {
        eprintln!(
            "profile: {} streams bound (joint loglik = sum across all): {}",
            resolved_obs.len(),
            resolved_obs.iter().map(|o| o.name.as_str())
                .collect::<Vec<_>>().join(", "),
        );
    } else {
        eprintln!("profile: using observation model '{}' from IR",
            resolved_obs[0].name);
    }

    // gh#90: silent-wrong-answer warning. If the model declares N>1
    // observation blocks but only M<N are bound, the unbound streams
    // contribute zero to the likelihood — the result looks plausible
    // but is methodologically wrong (those parameters fall back to
    // priors). Fires whether the user used --data PATH --obs NAME
    // (intentional single-stream subset) or named pairs covering a
    // subset of the model's blocks.
    {
        let bound_names: Vec<String> = bound_streams.iter()
            .map(|(n, _)| n.clone()).collect();
        if let Some(w) = crate::util::format_unbound_streams_warning(
            "profile", &model_obs_names, &bound_names,
        ) {
            eprint!("{}", w);
        }
    }

    // Per-stream load: each binding picks its column from the bound
    // file by stream name. For a single-stream binding fall back to
    // the 2-col TSV loader when the file has no matching column —
    // preserves the legacy (`time,value`) schema.
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: compiled.model.simulation.t_start,
        format: a.inference.time_format,
    };

    let mut per_stream_obs: Vec<Vec<Observation>> = Vec::with_capacity(bound_streams.len());
    let mut canonical_times: Option<Vec<f64>> = None;
    let n_streams = bound_streams.len();
    for (sname, spath) in &bound_streams {
        let path_str = spath.to_string_lossy().into_owned();
        let result = if n_streams == 1 {
            crate::pfilter::load_data_tsv_column(&path_str, sname, &time_opts)
                .or_else(|_| crate::pfilter::load_data_tsv_pub(&path_str, &time_opts))
        } else {
            crate::pfilter::load_data_tsv_column(&path_str, sname, &time_opts)
        };
        let stream_obs: Vec<Observation> = match result {
            Ok(v) => v.into_iter().map(|o| Observation { time: o.time, value: o.value }).collect(),
            Err(e) => {
                eprintln!("error: cannot load data column '{}' from {}: {}",
                    sname, path_str, e);
                std::process::exit(1);
            }
        };
        let times: Vec<f64> = stream_obs.iter().map(|o| o.time).collect();
        match &canonical_times {
            None => canonical_times = Some(times),
            Some(ct) => {
                if ct.len() != times.len()
                    || ct.iter().zip(&times).any(|(a, b)| (a - b).abs() > 1e-9)
                {
                    eprintln!(
                        "error: observation times for stream '{}' differ from \
                         the first resolved stream. All streams in a profile \
                         must share identical observation times.",
                        sname,
                    );
                    std::process::exit(1);
                }
            }
        }
        per_stream_obs.push(stream_obs);
    }

    // First stream's obs vector is the canonical schedule; downstream
    // code reads it for `obs_times` only.
    let observations: Vec<Observation> = per_stream_obs[0].clone();
    let observations = Arc::new(observations);

    let flow_indices = crate::util::resolve_flow_indices(&model, flow_name.as_deref())
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    let flow_indices = Arc::new(flow_indices);

    for sw in &a.sweep {
        let idx = compiled.param_index.get(sw.name.as_str()).copied()
            .unwrap_or_else(|| {
                eprintln!("focal parameter '{}' not found", sw.name);
                std::process::exit(1);
            });
        focal_grids.push(FocalGrid {
            name: sw.name.clone(),
            values: sw.grid.expand(),
            param_idx: idx,
        });
    }

    // Build the resolved fixed set: CLI `--fixed` wins for collisions
    // with `--fit toml`'s `[fixed]` (per gh#73 §2 precedence). The
    // toml's `[fixed]` is the *artifact's* default; the CLI flag is
    // the per-invocation override.
    //
    // Per 2026-05-25 CLI UX rev 2, `--fixed NAME=VALUE` carries
    // *both* a value (handled by the resolver above) and the kick-out
    // assertion (this set). Names from `--fixed-file` are pulled in
    // via the resolver's provenance — they kick out of [estimate]
    // identically.
    let cli_fixed_names: std::collections::HashSet<String> =
        a.model_overrides.fixed_cli.iter().map(|p| p.name.clone()).collect();
    let mut fixed_names: std::collections::HashSet<String> = cli_fixed_names.clone();
    for k in fit_toml_fixed.keys() {
        fixed_names.insert(k.clone());
    }

    // Focal-vs-fixed conflict check (gh#73 §2): a parameter cannot
    // simultaneously be the sweep axis and a fixed value. Surface
    // *which* source declared the fixed entry so the user knows where
    // to remove it.
    for sw in &focal_names {
        let in_cli = cli_fixed_names.contains(sw);
        let in_toml = fit_toml_fixed.contains_key(sw);
        if in_cli || in_toml {
            let source = match (in_cli, in_toml) {
                (true, true)  => "both `--fixed` and the fit toml's [fixed] block",
                (true, false) => "`--fixed`",
                (false, true) => "the fit toml's [fixed] block",
                (false, false) => unreachable!(),
            };
            eprintln!(
                "error: parameter '{}' is in both `--sweep` and {}. \
                 A swept parameter is pinned per cell at the sweep value; \
                 listing it as fixed is contradictory. Drop it from {}.",
                sw, source,
                if in_cli && !in_toml { "`--fixed`" }
                else if in_toml && !in_cli { "the fit toml's [fixed]" }
                else { "both" },
            );
            std::process::exit(1);
        }
    }
    let exclude: std::collections::HashSet<String> = focal_names.iter()
        .chain(fixed_names.iter()).cloned().collect();

    let param_names_to_estimate: Vec<String> = if rw_sd_auto {
        model.parameters.iter()
            .filter(|p| !exclude.contains(&p.name))
            .filter(|p| compiled.param_index.contains_key(p.name.as_str()))
            .map(|p| p.name.clone())
            .collect()
    } else {
        rw_sd_map_raw.keys()
            .filter(|name| !exclude.contains(*name))
            .cloned()
            .collect()
    };

    // Specs honour the fit-toml `[estimate]` block when supplied
    // (gh#73): bounds, transform, and ivp flow through to
    // `build_if2_params_from_specs`'s fit-toml-bounds-within-model
    // resolver. `rw_sd` and `transform` from CLI still win when both
    // sides declare them (CLI `--rw-sd` is the per-invocation
    // override; fit toml's `rw_sd` is the artifact default).
    let specs: Vec<crate::fit::runner::ParamSpec> = param_names_to_estimate.iter().map(|name| {
        let from_fit = fit_estimate.get(name);
        crate::fit::runner::ParamSpec {
            name: name.clone(),
            rw_sd: rw_sd_map_raw.get(name).and_then(|v| *v)
                .or_else(|| from_fit.and_then(|e| e.rw_sd)),
            transform: from_fit.and_then(|e| e.transform.as_ref().map(|t| t.as_str().to_string())),
            ivp: from_fit.map(|e| e.ivp).unwrap_or(false),
            bounds: from_fit.and_then(|e| e.bounds),
        }
    }).collect();

    let if2_params = crate::fit::runner::build_if2_params_from_specs(
        &model, &compiled, &base_params, &specs,
    ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    let if2_params = Arc::new(if2_params);

    // gh#118: focal params are pinned at grid values and excluded
    // from `if2_params`, but their priors still contribute to the
    // joint log-posterior. Build a parallel spec/prior set for the
    // focal parameters so we can add the focal-prior offset to the
    // emitted `log_posterior` column (matching `camdl fit run`'s
    // definition: log_post = log_lik + Σ log_prior(all estimated)).
    let focal_specs_for_priors: Vec<crate::fit::runner::ParamSpec> =
        focal_names.iter().map(|name| {
            let from_fit = fit_estimate.get(name);
            crate::fit::runner::ParamSpec {
                name: name.clone(),
                rw_sd: rw_sd_map_raw.get(name).and_then(|v| *v)
                    .or_else(|| from_fit.and_then(|e| e.rw_sd)),
                transform: from_fit.and_then(|e|
                    e.transform.as_ref().map(|t| t.as_str().to_string())),
                ivp: from_fit.map(|e| e.ivp).unwrap_or(false),
                bounds: from_fit.and_then(|e| e.bounds),
            }
        }).collect();
    let focal_if2_params: Vec<sim::inference::if2::EstimatedParam> =
        crate::fit::runner::build_if2_params_from_specs(
            &model, &compiled, &base_params, &focal_specs_for_priors,
        ).unwrap_or_else(|e| {
            eprintln!("error: building focal-param specs: {}", e);
            std::process::exit(1);
        });
    let focal_if2_params = Arc::new(focal_if2_params);
    let focal_priors: Vec<sim::inference::prior::Prior> =
        crate::fit::priors_precedence::resolve_priors_with_precedence(
            &focal_names, &fit_estimate, &model,
        ).into_iter().map(|r| r.prior).collect();
    let focal_priors = Arc::new(focal_priors);

    // ── Prior resolution (gh#73) ─────────────────────────────────────
    //
    // Resolve priors once, up front, for two reasons:
    //   (a) Surface the flat-fallback warning (or its suppression)
    //       BEFORE the parallel per-cell loop kicks off — the warning
    //       is a property of the configuration, not of any single
    //       cell.
    //   (b) Record the per-parameter source (fit_toml / model_ir /
    //       flat_fallback) into the per-seed `run.json` so the CAS
    //       provenance captures which knob controlled each parameter.
    //
    // The per-cell PMMH branch *re-resolves* against the same
    // (fit_estimate, model) inputs to get the typed `Prior` values it
    // needs; that re-resolution is byte-identical to this one (same
    // call into `fit::runner::resolve_prior`). Resolving here as well
    // makes the warning/provenance flow independent of whether the
    // PMMH branch ever runs — keeps the diagnostic surface uniform
    // across `--algorithm` choices and supports a future extension to
    // honour priors on IF2 too (which currently ignores them by
    // design but should still surface what *would* be ignored).
    let resolved_priors: Vec<crate::fit::priors_precedence::ResolvedPrior> =
        crate::fit::priors_precedence::resolve_priors_with_precedence(
            &if2_params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &fit_estimate,
            &model,
        );
    if matches!(profile_algo, ProfileAlgo::Pmmh) && !a.suppress_warnings {
        if let Some(w) = crate::fit::priors_precedence::format_flat_fallback_warning(
            &resolved_priors, a.fit.is_some(),
        ) {
            eprint!("{}", w);
        }
    }
    // Suppression is loud — recorded into provenance even when the
    // user passed `--suppress-warnings` so reviewers can audit the
    // waiver. The `suppressed_warnings` field is empty when nothing
    // was suppressed.
    let suppressed_warnings: Vec<String> =
        if a.suppress_warnings
            && resolved_priors.iter()
                .any(|r| r.source == crate::fit::priors_precedence::PriorSource::FlatFallback)
        {
            vec!["profile_flat_prior_fallback".to_string()]
        } else {
            Vec::new()
        };

    // Per-start init draws across non-focal estimated params (gh#42).
    // Computed once, reused at every grid cell so start_idx=k means the
    // same draw across cells (lets the per-start TSV rows be compared
    // cell-to-cell). `None` → every start uses `if2_params` directly
    // (Single mode; or `--starts 1`).
    if init_method == crate::fit::init::InitMethod::SurveyTopK {
        eprintln!("error: --init survey_top_k is not yet supported on \
            `camdl profile`; v2 supports it on IF2 / PMMH / PGAS \
            `camdl fit` stages, profile (and NLopt) are deferred to v3 \
            (see gh#51 §\"Stage scope\"). Workaround: use --init lhs.");
        std::process::exit(1);
    }
    // Step 7 warm-start variants: dispatch through the new
    // `chain_starts::draw_chain_starts` entry point. The CLI break
    // wires `--posterior` / `--mle` / `--params` companion flags via
    // `InitModeTag::to_init_method` (step 7).
    let per_start_params: Option<Arc<Vec<Vec<sim::inference::if2::EstimatedParam>>>> =
        match &init_method {
            crate::fit::init::InitMethod::FromPrior
            | crate::fit::init::InitMethod::FromPosterior { .. }
            | crate::fit::init::InitMethod::FromMle       { .. }
            | crate::fit::init::InitMethod::FromParams    { .. } => {
                let starts = crate::fit::chain_starts::draw_chain_starts(
                    &resolved, &init_method, n_starts, seed_base,
                ).unwrap_or_else(|e| {
                    eprintln!("error: profile --init {}: {}", init_method, e);
                    std::process::exit(1);
                });
                Some(Arc::new(starts.to_estimated_params(&if2_params)))
            }
            _ => crate::fit::init::build_chain_starts(
                    init_method.clone(), &if2_params, n_starts, seed_base,
                ).map(Arc::new),
        };

    let process = Arc::new(ChainBinomialProcess::new(compiled.clone(), dt));
    // Build one StreamSpec per resolved IR observation. For
    // single-stream profiles `--flow <name>` overrides the IR
    // projection (forces incidence over the named transition family);
    // for multi-stream we always use each stream's IR projection
    // (the `--flow` + multi-stream combination was already rejected
    // upstream).
    // Concrete `Arc<MultiStreamObsModel>` — the IF2 per-cell call site
    // accepts `&dyn ObservationModel<ParticleState>` (auto-coerced from
    // `&MultiStreamObsModel`); the NLopt path needs the concrete type
    // for `optimize_cell` (`compute_ode_loglik` reads
    // `log_likelihood_from_flows_and_counts` directly).
    let obs_times_vec: Vec<f64> = observations.iter().map(|o| o.time).collect();
    let obs_model_obj: Arc<MultiStreamObsModel> = {
        let mut stream_specs = Vec::with_capacity(resolved_obs.len());
        for (obs, stream_obs) in resolved_obs.iter().zip(per_stream_obs.iter()) {
            let projection = if resolved_obs.len() == 1 && flow_name.is_some() {
                sim::inference::multi_stream_obs::StreamProjection::FlowSum(
                    flow_indices.to_vec(),
                )
            } else {
                sim::inference::multi_stream_obs::StreamProjection::from_ir(
                    &obs.projection, &compiled, &obs.name,
                ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
            };
            stream_specs.push(StreamSpec {
                projection,
                ir_model: (*obs).clone(),
                observations: stream_obs.iter().map(|o| o.value).collect(),
                obs_times: obs_times_vec.clone(),
            });
        }
        Arc::new(MultiStreamObsModel::new(stream_specs, compiled.clone())
            .unwrap_or_else(|e| {
                eprintln!("error: observation model construction failed: {:?}", e);
                std::process::exit(1);
            }))
    };

    // Build Cartesian product of all focal grids.
    let mut grid_points: Vec<Vec<(usize, f64)>> = vec![vec![]];
    for fg in &focal_grids {
        let mut expanded = Vec::new();
        for existing in &grid_points {
            for &val in &fg.values {
                let mut point = existing.clone();
                point.push((fg.param_idx, val));
                expanded.push(point);
            }
        }
        grid_points = expanded;
    }

    // ── Build typed CAS inputs ─────────────────────────────────────────
    //
    // ProfileInputs encapsulates every content-bearing input. inner_hash
    // (seed-free) drives the multi-seed umbrella; per-seed content_hash
    // = compose_with_replicate(inner, "seed", seed) — same as a
    // standalone --seed N invocation, so cache lookup is uniform.
    let model_hash = crate::hashing::model_hash(&model_json);
    let base_params_hash = {
        let mut lines: Vec<String> = model.parameters.iter()
            .map(|p| format!("{}={}", p.name,
                p.value.unwrap_or(base_params[compiled.param_index[p.name.as_str()]])))
            .collect();
        lines.sort();
        ContentHash::from_bytes(lines.join("\n").as_bytes()).full().to_string()
    };
    let grid_spec: Vec<GridAxis> = focal_grids.iter().map(|fg| GridAxis {
        param: fg.name.clone(),
        values: fg.values.clone(),
    }).collect();

    // Resolve seeds. --seeds wins; default is the single --seed.
    let seeds: Vec<u64> = match &a.seeds {
        Some(spec) => spec.expand(),
        None => vec![seed_base],
    };
    if seeds.is_empty() {
        eprintln!("error: --seeds expanded to empty list");
        std::process::exit(1);
    }

    let argv: Vec<String> = std::env::args().collect();
    let root = output_root(None, None);
    let stem = crate::hashing::path_stem_slug(&ir_path);

    // gh#38: obs_family is the resolved canonical name we used to pick
    // the IR observation set. For an explicit `--obs`, it's the
    // user-supplied name. For an implicit single-stream model it's the
    // sole IR observation's name (so two profiles on the same model
    // with one obs and the same params still hit the cache).
    let obs_family_key = obs_name_arg.clone()
        .unwrap_or_else(|| resolved_obs[0].name.clone());

    // gh#39 + gh#90: hash every bound stream's data file bytes at
    // launch. Each (stream_name, content_hash) pair participates in
    // the CAS key (sorted by stream name) so:
    //   - editing any one stream's data file invalidates the cache
    //     (gh#39 invariant generalised to multi-stream),
    //   - adding or removing a stream binding invalidates the cache
    //     (gh#90: switching --data cases=... → --data cases=...
    //     --data deaths=... is a real content change),
    //   - reordering --data flags on the CLI does NOT invalidate the
    //     cache (sort by name).
    // De-duplicate identical file paths: when a wide TSV serves
    // multiple streams (`--data cases=wide.tsv --data deaths=wide.tsv`
    // or family-root single-file expansion), each stream still records
    // its own (name, hash) pair so adding a stream invalidates even
    // when the file bytes are unchanged.
    let mut data_hashes: Vec<(String, String)> = Vec::with_capacity(bound_streams.len());
    let mut file_cache: HashMap<std::path::PathBuf, String> = HashMap::new();
    for (sname, spath) in &bound_streams {
        let h = if let Some(h) = file_cache.get(spath) {
            h.clone()
        } else {
            let bytes = std::fs::read(spath).unwrap_or_else(|e| {
                eprintln!("error: cannot read --data file '{}': {}",
                    spath.display(), e);
                std::process::exit(1);
            });
            let h = crate::hashing::sha256_hex(&bytes);
            file_cache.insert(spath.clone(), h.clone());
            h
        };
        data_hashes.push((sname.clone(), h));
    }

    // gh#73: the per-parameter source list is stored in the same
    // order the resolver emitted (declaration order of estimated
    // params). The `inner_hash` sort key normalises this to a
    // canonical form for caching; we keep the natural order here so
    // `run.json` is human-friendly.
    let resolved_priors_kv: Vec<(String, String)> = resolved_priors.iter()
        .map(|r| {
            let source_str = match r.source {
                crate::fit::priors_precedence::PriorSource::FitToml      => "fit_toml",
                crate::fit::priors_precedence::PriorSource::ModelIr      => "model_ir",
                crate::fit::priors_precedence::PriorSource::FlatFallback => "flat_fallback",
                // gh#75: explicit flat-prior opt-in via `prior = { flat = {} }`.
                // Not reachable on the profile path today (the resolver
                // gets its fit-toml priors from `--fit`, which doesn't yet
                // surface explicit flat — but the resolver's output is
                // shared, so the match must be exhaustive).
                crate::fit::priors_precedence::PriorSource::FlatExplicit => "flat_explicit",
            };
            (r.param.clone(), source_str.to_string())
        })
        .collect();
    // The CLI `--fixed` set is the union of CLI flags + fit-toml's
    // [fixed] block (CLI wins per spec for collisions, but both
    // contribute to the canonical fixed list). Sort for stable hashing.
    let mut fixed_for_cas: Vec<String> = fixed_names.iter().cloned().collect();
    fixed_for_cas.sort();

    // gh#83/gh#85 step 9: build parameters_provenance from the
    // unified resolver output (populated upstream). Each entry maps a
    // resolved parameter name to the value + source + role + audit
    // fields. NOT part of the CAS hash (provenance is metadata).
    let parameters_provenance: std::collections::HashMap<
        String, crate::run_meta::ParameterProvenance> =
        resolved.params.iter()
            .map(|rp| (rp.name.clone(),
                 crate::run_meta::ParameterProvenance::from_resolved(rp)))
            .collect();
    // Per-start init provenance: ChainStarts uses the `--init` mode
    // as its method tag. For legacy modes we still emit one entry so
    // `init_provenance.method` is never absent on a profile run.
    let init_provenance: Option<crate::run_meta::InitProvenance> =
        per_start_params.as_ref().map(|psp| {
            // Build a ChainStarts-shaped view from the existing
            // `EstimatedParam` per-start vectors, using the
            // best-available InitSource tag for each chain.
            let starts: Vec<crate::fit::chain_starts::ChainStart> =
                psp.iter().enumerate().map(|(chain_id, per_start)| {
                    let values: std::collections::HashMap<String, f64> =
                        per_start.iter()
                            .map(|spec| (spec.name.clone(), spec.initial))
                            .collect();
                    let source = match &init_method {
                        crate::fit::init::InitMethod::FromPrior =>
                            crate::fit::chain_starts::InitSource::PriorDraw {
                                seed: crate::util::derive_chain_seed(
                                    seeds[0], chain_id),
                            },
                        crate::fit::init::InitMethod::FromPosterior {
                            source: crate::fit::init::PosteriorSource::DrawsTsv(p),
                        } | crate::fit::init::InitMethod::FromPosterior {
                            source: crate::fit::init::PosteriorSource::FitDir(p),
                        } => crate::fit::chain_starts::InitSource::PosteriorRow {
                            row: chain_id, path: p.clone(),
                        },
                        crate::fit::init::InitMethod::FromMle {
                            source: crate::fit::init::MleSource::File(p),
                        } | crate::fit::init::InitMethod::FromMle {
                            source: crate::fit::init::MleSource::FitDir(p),
                        } => crate::fit::chain_starts::InitSource::MlePoint {
                            path: p.clone(),
                        },
                        crate::fit::init::InitMethod::FromParams { path } =>
                            crate::fit::chain_starts::InitSource::ParamsPoint {
                                path: path.clone(),
                            },
                        crate::fit::init::InitMethod::Lhs =>
                            crate::fit::chain_starts::InitSource::LhsCell {
                                row: chain_id,
                            },
                        crate::fit::init::InitMethod::Uniform =>
                            crate::fit::chain_starts::InitSource::UniformDraw {
                                seed: crate::util::derive_chain_seed(
                                    seeds[0], chain_id),
                            },
                        crate::fit::init::InitMethod::Single |
                        crate::fit::init::InitMethod::SurveyTopK =>
                            crate::fit::chain_starts::InitSource::SeededBase,
                    };
                    crate::fit::chain_starts::ChainStart {
                        chain_id, values, source,
                    }
                }).collect();
            let cs = crate::fit::chain_starts::ChainStarts {
                starts, method: init_method.clone(),
            };
            crate::run_meta::InitProvenance::from_chain_starts(&cs)
        });

    let template_inputs = ProfileInputs {
        model_path: ir_path.clone(),
        stem: stem.clone(),
        model_hash: model_hash.clone(),
        base_params_hash,
        data_hashes,
        focal_grid: grid_spec,
        fixed: fixed_for_cas,
        obs_family: obs_family_key,
        if2_config: ProfileIf2Config {
            n_particles, n_iterations, cooling, dt, n_starts,
        },
        // gh#89: cache-key inputs for non-IF2 algorithms.
        algorithm: format!("{:?}", profile_algo).to_lowercase(),
        pmmh_steps,
        pmmh_particles,
        pmmh_rho: pmmh_rho_opt,
        starts_from_lineage: None,
        fit_toml_hash: fit_toml_hash.clone(),
        resolved_priors: resolved_priors_kv,
        suppressed_warnings: suppressed_warnings.clone(),
        seed: seeds[0],   // overwritten per-seed below
        parameters_provenance,
        init_provenance,
    };

    // ── Layout: every profile is a ReplicateSet umbrella (N=1 trivially).
    // The single-seed case is just the degenerate replicate-set; the
    // disk layout, run.json schema, and resolution path are uniform.
    let replicate_set = ReplicateSet {
        inner_hash: template_inputs.inner_hash(),
        dim_name:   "seed".to_string(),
        keys:       seeds.iter().map(|s| format!("seed_{}", s)).collect(),
        child_kind: "profile".to_string(),
    };
    let umbrella_dir: PathBuf = {
        let parent_hash = replicate_set.parent_hash();
        let dirname = match &stem {
            Some(s) if !s.is_empty() => format!("{}-{}", s, parent_hash.short()),
            _ => parent_hash.short().to_string(),
        };
        let dir = root.join("profiles").join(dirname);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("error: cannot create {}: {}", dir.display(), e);
            std::process::exit(1);
        }
        let umbrella_run = Run {
            hash:              parent_hash.full().to_string(),
            version:           crate::version::VERSION_SHORT.to_string(),
            created_at:        crate::cas::iso8601_utc(std::time::SystemTime::now()),
            argv:              argv.clone(),
            status: RunStatus::Running,
            label:             label_arg.clone(),
            kind:              replicate_set.run_kind(),
        };
        if let Err(e) = umbrella_run.write(&dir) {
            eprintln!("warning: could not write umbrella run.json: {}", e);
        }
        eprintln!("profile ({} replicate{}): {}",
            seeds.len(),
            if seeds.len() == 1 { "" } else { "s" },
            dir.display());
        dir
    };

    // Per-seed directories + content hashes (the latter populates
    // FitStageMeta.parent_profile_hash on each leaf start_run.json).
    let mut seed_dirs: Vec<PathBuf> = Vec::with_capacity(seeds.len());
    let mut per_seed_hashes: Vec<String> = Vec::with_capacity(seeds.len());
    for &seed in &seeds {
        let inputs_seed = ProfileInputs { seed, ..template_inputs.clone() };
        let dir = replicate_set.child_dir(&umbrella_dir, &format!("seed_{}", seed));
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("error: cannot create {}: {}", dir.display(), e);
            std::process::exit(1);
        }
        let profile_run = Run {
            hash:              inputs_seed.content_hash().full().to_string(),
            version:           crate::version::VERSION_SHORT.to_string(),
            created_at:        crate::cas::iso8601_utc(std::time::SystemTime::now()),
            argv:              argv.clone(),
            status: RunStatus::Running,
            label:             None,
            kind:              inputs_seed.run_kind(),
        };
        if let Err(e) = profile_run.write(&dir) {
            eprintln!("warning: could not write profile run.json: {}", e);
        }
        // focal.toml per grid point inside this seed's tree.
        for (gi, point) in grid_points.iter().enumerate() {
            let point_dir = profile_point_dir(&dir, gi);
            if let Err(e) = std::fs::create_dir_all(&point_dir) {
                eprintln!("warning: cannot create {}: {}", point_dir.display(), e);
                continue;
            }
            let focal_toml_path = point_dir.join("focal.toml");
            if focal_toml_path.exists() { continue; }
            let mut body = String::from("# Pinned focal parameter values for this grid point.\n\n");
            for (fg, &(_, val)) in focal_grids.iter().zip(point.iter()) {
                body.push_str(&format!("{} = {}\n", fg.name, val));
            }
            let _ = std::fs::write(&focal_toml_path, body);
        }
        per_seed_hashes.push(inputs_seed.content_hash().full().to_string());
        seed_dirs.push(dir);
    }

    let total_jobs = grid_points.len() * n_starts * seeds.len();
    let dim_str = focal_grids.iter()
        .map(|fg| format!("{}={}", fg.name, fg.values.len()))
        .collect::<Vec<_>>().join(" × ");
    // Banner shape preserved from the IF2/NLopt era; PMMH lands the
    // accurate per-cell budget through a second one-line summary
    // below so we don't reshape this format for callers that grep
    // "IF2 runs". (PMMH banner is additive.)
    eprintln!("profile: {} grid ({}) × {} starts × {} seeds = {} IF2 runs ({} particles × {} iter each)",
        grid_points.len(), dim_str, n_starts, seeds.len(), total_jobs,
        n_particles, n_iterations);
    if matches!(profile_algo, ProfileAlgo::Pmmh) {
        eprintln!("profile: PMMH per cell = {} particles × {} MCMC steps (rho = {})",
            pmmh_particles, pmmh_steps,
            pmmh_rho_opt.map(|r| r.to_string()).unwrap_or_else(|| "off".into()));
    }

    // ── Progress + cache scan ─────────────────────────────────────────
    let mp = MultiProgress::with_draw_target(crate::progress::draw_target());
    let overall_style = ProgressStyle::with_template(
        "  {prefix:>12} {bar:40.cyan/dim} {pos:>3}/{len:3} {msg}"
    ).unwrap().progress_chars("━╸─");
    let overall_pb = mp.add(ProgressBar::new(total_jobs as u64));
    overall_pb.set_style(overall_style);
    overall_pb.set_prefix("profile");
    let plain = crate::progress::is_plain();
    let progress_throttle = Mutex::new(crate::progress::Throttle::default());
    if plain {
        log::info!("profile: {} grid points × {} starts × {} seeds = {} jobs",
            grid_points.len(), n_starts, seeds.len(), total_jobs);
    }

    // Job tuple: (seed_idx, grid_idx, start_idx). Cache hit if the
    // start_dir under this seed's profile tree has a parseable run.json.
    let jobs: Vec<(usize, usize, usize)> = (0..seeds.len())
        .flat_map(|seed_idx| (0..grid_points.len())
            .flat_map(move |gi| (0..n_starts).map(move |si| (seed_idx, gi, si))))
        .collect();

    let mut cached: Vec<(usize, usize, usize)> = Vec::new();
    let mut remaining: Vec<(usize, usize, usize)> = Vec::new();
    for &(seed_idx, gi, si) in &jobs {
        let start_dir = profile_point_start_dir(&seed_dirs[seed_idx], gi, si);
        if Run::read(&start_dir).is_ok() {
            cached.push((seed_idx, gi, si));
        } else {
            remaining.push((seed_idx, gi, si));
        }
    }
    if !cached.is_empty() {
        eprintln!("profile: {} of {} starts already cached — resuming",
            cached.len(), total_jobs);
        overall_pb.inc(cached.len() as u64);
    }

    if parallel > 0 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(parallel)
            .build_global();
    }

    // Throttled rollup rewrites: per-seed profile.tsv (1s throttle) and
    // (multi-seed only) the cross-seed summary.tsv (2s throttle, since
    // it reads N seeds' rollups). Last-completion-wins.
    let rollup_throttle = Mutex::new(std::time::Instant::now()
        - std::time::Duration::from_secs(10));
    let summary_throttle = Mutex::new(std::time::Instant::now()
        - std::time::Duration::from_secs(10));
    let start_time = std::time::Instant::now();

    let focal_names_ordered: Vec<String> =
        focal_grids.iter().map(|fg| fg.name.clone()).collect();

    // ── Run remaining jobs in parallel ──────────────────────────────
    remaining.par_iter().for_each(|&(seed_idx, grid_idx, start_idx)| {
        let process = Arc::clone(&process);
        let obs_model_obj = Arc::clone(&obs_model_obj);
        let if2_params = Arc::clone(&if2_params);
        let focal_if2_params = Arc::clone(&focal_if2_params);
        let focal_priors = Arc::clone(&focal_priors);
        let focal_values: Vec<f64> = grid_points[grid_idx].iter().map(|&(_, v)| v).collect();
        let seed = seeds[seed_idx];

        // Pin focal parameters
        let mut params = base_params.clone();
        for &(idx, val) in &grid_points[grid_idx] {
            params[idx] = val;
        }

        let job_seed = seed ^ (grid_idx as u64 * 1000 + start_idx as u64);
        let job_t0 = std::time::Instant::now();

        // gh#42: when --init lhs/uniform, each start uses its own draw
        // across the non-focal estimated params; otherwise (Single, or
        // --starts 1) every start shares `if2_params`. The focal params
        // are already pinned in `params` above, so the LHS draw on
        // non-focal indices doesn't perturb the grid point.
        let per_start_specs: &[sim::inference::if2::EstimatedParam] =
            per_start_params.as_ref()
                .map(|psp| psp[start_idx].as_slice())
                .unwrap_or(if2_params.as_slice());

        // Dispatch on algorithm. IF2 keeps the historical PF-driven
        // per-cell MLE; PMMH runs a per-cell MCMC chain and extracts
        // the MAP as the cell's MLE (the prompt's --algorithm pmmh
        // path for 2-D seed-timing profiles); NLopt branches into the
        // deterministic-MLE path (gh#47) — same `compute_ode_loglik`
        // closure that fit.toml stages call, with the focal grid value
        // pinned in `params` and optimization confined to the non-focal
        // estimated indices.
        //
        // gh#74 Option B: each branch also populates a
        // `PerStartDiagnostics` carrying the convergence info this
        // start produced (acceptance rate for PMMH, cooling-end state
        // for IF2, completion flag for everyone). The rollup phase
        // aggregates these across K starts → per-cell columns.
        let mut diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        // gh#109: per-cell return carries final_log_posterior alongside
        // final_loglik. Only the PMMH branch populates it (PMMHResult
        // already exposes map_log_posterior); IF2/NLopt are point-MLE
        // optimisers with no posterior concept, so the field is NaN
        // for those algorithms. Downstream (render_mle_toml, rollup
        // TSV) skip-when-NaN so the addition is transparent for
        // existing IF2/NLopt profiles.
        let (final_loglik, final_log_posterior, mle_params): (f64, f64, Vec<f64>) = match profile_algo {
            ProfileAlgo::If2 => {
                let config = IF2Config {
                    n_particles, n_iterations,
                    cooling_fraction: cooling, cooling_target_iters: n_iterations, dt,
                    t_start: process.compiled.model.simulation.t_start,
                    simplex_groups: vec![],
                    skip_first_obs_from_loglik: false,
                    pf_wallclock_disabled: false,
                };
                let result = run_if2(
                    &*process, &*obs_model_obj, &params, per_start_specs, &config, job_seed,
                );
                // gh#74 Option B: per-start IF2 diagnostics. Populate
                // the trace + cooling end-state on success; on Err the
                // engine returned, leave `diag` at defaults so the
                // aggregator records "not completed" for this start.
                diag.algo = Some(crate::profile_diagnostics::DiagAlgo::If2);
                match result {
                    Ok(r) => {
                        // Clean-eval re-pass at the swarm-mean MLE. IF2's
                        // `final_loglik` is the perturbed-params loglik
                        // from the last cooling iteration
                        // (`if2.rs:540: final_loglik:
                        // last_iter.if2_perturbed_loglik`), NOT the
                        // loglik at `r.mle` (the swarm mean). Without
                        // this re-pass, mle.toml reports a loglik that
                        // isn't reproducible by `camdl pfilter` at the
                        // saved MLE — historically a ~40-nat extraction
                        // bias on noisy PF runs, and finite-vs-(-inf)
                        // catastrophic on PF-degenerate models (events
                        // blocks, GH #68). Fit-run already does this via
                        // `run_quick_pfilter` (cf. runner.rs:1208-1224);
                        // this mirrors the same pattern for profile's
                        // per-cell path.
                        let smc_config = sim::inference::traits::SMCConfig {
                            n_particles: n_particles.min(500),
                            dt,
                            t_start: process.compiled.model.simulation.t_start,
                            skip_first_obs_from_loglik: false,
                            record_ancestry: false,
                            record_prequential: false,
                            pf_wallclock_disabled: false,
                        };
                        // Distinct seed from the IF2 inner run so the
                        // clean-eval PF doesn't reuse IF2's last
                        // resample-RNG state. Adding 1 is sufficient —
                        // ChaCha8 (StatefulRng) produces uncorrelated
                        // sequences for seeds 1 apart.
                        let clean_seed = job_seed.wrapping_add(1);
                        let true_ll = sim::inference::bootstrap_filter(
                            &*process, &*obs_model_obj, &r.mle, &smc_config, clean_seed,
                        ).map(|res| res.log_likelihood)
                          .unwrap_or(f64::NEG_INFINITY);
                        // Trace = per-iteration `if2_perturbed_loglik`.
                        // The `IF2IterResult.loglik` field is NaN here
                        // (the post-hoc clean-PF re-eval only runs at
                        // the swarm mean, not per iter — see
                        // if2.rs:121-129), so we surface the engine's
                        // own perturbed-loglik trace. It's documented
                        // as "NOT useful for model assessment" but is
                        // a valid signal for chain-level agreement
                        // across the K starts, which is exactly what
                        // Rhat measures.
                        let trace: Vec<f64> = r.iterations.iter()
                            .map(|it| it.if2_perturbed_loglik)
                            .collect();
                        let last_iter = r.iterations.last();
                        diag.completed = true_ll.is_finite();
                        diag.iterations_used = last_iter.map(|it| it.iteration + 1);
                        // cooling_final: mean across estimated params
                        // of the final iter's effective_rw_sd (the
                        // *actual* ending perturbation SD per
                        // ParamIterDiag.effective_rw_sd at
                        // if2.rs:147). Scalar summary of "how cool
                        // the perturbation got." Mean-across-params
                        // is fine — a model with a heterogeneous
                        // rw_sd has each param cooled by the same
                        // schedule factor, so the spread between
                        // params is mostly init-dependent and the
                        // mean is a representative single number.
                        diag.cooling_final = last_iter.and_then(|it| {
                            if it.param_diag.is_empty() { None }
                            else {
                                let s: f64 = it.param_diag.iter()
                                    .map(|p| p.effective_rw_sd).sum();
                                Some(s / it.param_diag.len() as f64)
                            }
                        });
                        diag.loglik_trace = trace;
                        // IF2 is a point-MLE optimiser; no posterior.
                        (true_ll, f64::NAN, r.mle)
                    }
                    Err(_) => {
                        // diag.completed stays false; iterations_used /
                        // cooling_final / loglik_trace stay at defaults
                        // — the aggregator records NaN for this start's
                        // contribution.
                        (f64::NEG_INFINITY, f64::NAN, params.clone())
                    }
                }
            }
            ProfileAlgo::Pmmh => {
                // Per-cell PMMH: short MCMC chain over non-focal
                // estimated params, focal pinned via `params`. The MAP
                // of the chain is the cell's reported MLE — matches
                // PMMHResult.map_params, which is the highest-posterior
                // sample seen (consistent with how fit/pmmh.rs reports
                // MAP at end of chain).
                //
                // Priors (gh#73): resolved via the precedence chain
                //   --fit toml > model IR (`~` syntax) > Prior::Flat.
                // The resolution is shared with `fit/runner::resolve_prior`
                // so behaviour matches `camdl fit run`. Pre-fix this
                // path hardcoded Prior::Flat, silently downgrading the
                // declared posterior semantics to MLE.
                let proposal_sd: Vec<f64> = per_start_specs.iter()
                    .map(|p| p.transformed_sd(p.rw_sd, p.initial) * 5.0)
                    .collect();
                let cell_names: Vec<String> = per_start_specs.iter()
                    .map(|p| p.name.clone())
                    .collect();
                let resolved = crate::fit::priors_precedence::resolve_priors_with_precedence(
                    &cell_names, &fit_estimate, &model,
                );
                let priors: Vec<Prior> = resolved.iter()
                    .map(|r| r.prior.clone())
                    .collect();
                let pmmh_config = PMMHConfig {
                    n_steps: pmmh_steps,
                    n_particles: pmmh_particles,
                    dt,
                    proposal_sd,
                    adapt: true,
                    adapt_start: 50,
                    thin: 1,
                    burn_in: 100,
                    rho: pmmh_rho_opt,
                    n_source_groups: compiled.source_groups.len(),
                };

                // PF process kernel + obs model for this cell. PMMH on
                // profile is chain_binomial-only (rejected upstream for
                // --backend ode), so wire ChainBinomialProcess directly.
                let pf_process = ChainBinomialProcess::new(
                    compiled.clone(), pmmh_config.dt,
                );
                let pf_obs_model = Arc::clone(&obs_model_obj);
                let smc_cfg = sim::inference::traits::SMCConfig {
                    n_particles: pmmh_config.n_particles,
                    dt: pmmh_config.dt,
                    t_start: compiled.model.simulation.t_start,
                    skip_first_obs_from_loglik: false,
                    record_ancestry: false,
                    record_prequential: false,
                    pf_wallclock_disabled: false,
                };

                let eval_loglik = |theta: &[f64], pf_seed: u64| -> f64 {
                    match sim::inference::bootstrap_filter(
                        &pf_process, &*pf_obs_model, theta, &smc_cfg, pf_seed,
                    ) {
                        Ok(r) => r.log_likelihood,
                        Err(_) => f64::NEG_INFINITY,
                    }
                };

                // Correlated-PF evaluator (only used when rho is set).
                // Mirrors fit/pmmh.rs's eval_correlated.
                let eval_correlated: Option<Box<dyn Fn(
                    &[f64],
                    &sim::inference::correlated_pf::PFRandomState,
                ) -> f64>> = if pmmh_config.rho.is_some() {
                    let pf_process2 = ChainBinomialProcess::new(
                        compiled.clone(), pmmh_config.dt,
                    );
                    let pf_obs_model2 = Arc::clone(&obs_model_obj);
                    let smc_cfg2 = smc_cfg.clone();
                    let cell_seed = job_seed;
                    Some(Box::new(move |theta: &[f64], randoms| -> f64 {
                        match sim::inference::correlated_pf::bootstrap_filter_correlated(
                            &pf_process2, &*pf_obs_model2, theta,
                            &smc_cfg2, randoms, cell_seed,
                        ) {
                            Ok(r) => r.log_likelihood,
                            Err(_) => f64::NEG_INFINITY,
                        }
                    }))
                } else {
                    None
                };
                let eval_corr_ref: Option<&sim::inference::pmmh::CorrelatedEvalFn> =
                    eval_correlated.as_deref();

                // Profile has no hierarchical prior surface, so
                // `param_names` can be empty (avoids the
                // base_params/param_names length check inside run_pmmh
                // when no Hierarchical priors are present).
                let result = run_pmmh(
                    per_start_specs,
                    &priors,
                    &params,
                    &[],
                    &pmmh_config,
                    &observations,
                    &eval_loglik,
                    eval_corr_ref,
                    job_seed,
                    None,
                    None,
                    String::new(),
                );
                // gh#74 Option B: PMMH diagnostics from the engine's
                // returned result. `acceptance_rate` is already
                // post-burn-in (see pmmh.rs:508-514). Trace carries
                // the per-step PF loglik — the same value the chain
                // uses for MH acceptance.
                diag.algo = Some(crate::profile_diagnostics::DiagAlgo::Pmmh);
                diag.acc_rate = if result.acceptance_rate.is_finite() {
                    Some(result.acceptance_rate)
                } else { None };
                diag.loglik_trace = result.steps.iter()
                    .map(|s| s.log_likelihood)
                    .collect();
                // gh#109: per-step joint log-posterior (log_likelihood
                // + log_prior). Together with the loglik trace this
                // lets the rollup compute the prior-contribution
                // delta and lets the user compare a profile likelihood
                // vs profile posterior.
                //
                // gh#118: `s.log_prior` from the PMMH engine sums over
                // the nuisance (estimated) params only — focal params
                // are pinned and excluded from PMMH's estimation set.
                // Add the focal-prior contribution at the cell's fixed
                // values so the column matches `camdl fit run`'s
                // joint-posterior definition. The offset is constant
                // within the cell.
                let focal_log_prior_offset = compute_focal_log_prior_offset(
                    &focal_if2_params, &focal_priors, &focal_values,
                );
                diag.log_posterior_trace = result.steps.iter()
                    .map(|s| s.log_likelihood + s.log_prior + focal_log_prior_offset)
                    .collect();
                let best_ll = result.steps.iter()
                    .map(|s| s.log_likelihood)
                    .fold(f64::NEG_INFINITY, f64::max);
                // Report the MAP point's loglik for the rollup if it
                // dominates the per-sample max (typical: map_loglik
                // is the recorded best). Otherwise fall back to the
                // per-sample max so the cell still reports a finite
                // value when MAP tracking missed an early high-ll
                // sample.
                let final_ll = if result.map_loglik.is_finite() {
                    result.map_loglik.max(best_ll)
                } else if best_ll.is_finite() {
                    best_ll
                } else {
                    f64::NEG_INFINITY
                };
                diag.completed = final_ll.is_finite();
                // gh#109: PMMH carries the MAP joint log-posterior
                // alongside the MLE-by-loglik. NaN-guard mirrors the
                // final_ll computation above so a non-finite chain
                // doesn't leak Inf into mle.toml.
                //
                // gh#118: add the focal-prior offset so the MAP
                // log_posterior matches `camdl fit run`'s definition.
                // If the offset itself is NEG_INFINITY (focal value
                // outside its prior's support), the sum is non-finite
                // and the is_finite guard surfaces NaN — semantically
                // correct (cell at zero prior probability).
                let final_lp = if result.map_log_posterior.is_finite() {
                    let corrected = result.map_log_posterior + focal_log_prior_offset;
                    if corrected.is_finite() { corrected } else { f64::NAN }
                } else {
                    f64::NAN
                };
                (final_ll, final_lp, result.map_params)
            }
            ProfileAlgo::Nlopt(nlopt_algo) => {
                // Per-cell starting point: copy `params` (focal pinned)
                // and overwrite the non-focal estimated slots with the
                // per-start LHS draws. `per_start_specs` contains only
                // non-focal estimated specs, so the focal pin survives.
                let mut full_start = params.clone();
                for spec in per_start_specs {
                    full_start[spec.index] = spec.initial;
                }
                let est_indices: Vec<usize> =
                    per_start_specs.iter().map(|p| p.index).collect();
                let bounds: Vec<(f64, f64)> =
                    per_start_specs.iter().map(|p| (p.lower, p.upper)).collect();
                let ode_dt = compiled.model.simulation.dt.unwrap_or(dt);
                // Per-cell NLopt knobs: looser than fit.toml stages
                // because profile is landscape exploration, not a
                // single converged MLE. With 11+ free params on a
                // typhoid-class model each ODE solve runs ~100ms-1s,
                // so a 1500-eval budget is ~5 min per cell — fast
                // enough for routine 1D profiles, expensive enough to
                // converge on the ridge. Tunable later via dedicated
                // `--nlopt-tolerance` / `--nlopt-max-evals` flags
                // when downstream tuning surfaces a need.
                let tolerance = 1e-4;
                let max_evals = 1500;
                diag.algo = Some(crate::profile_diagnostics::DiagAlgo::Nlopt);
                match crate::fit::nlopt_stage::optimize_cell(
                    nlopt_algo,
                    &compiled,
                    &obs_model_obj,
                    &obs_times_vec,
                    ode_dt,
                    &bounds,
                    &est_indices,
                    &full_start,
                    tolerance,
                    max_evals,
                ) {
                    Ok(r) => {
                        // Reproject the optimizer-space result back into
                        // a full param vector (focal pinned, non-focal
                        // optimized, fixed at base).
                        let mut mle = full_start.clone();
                        for (slot, &model_idx) in est_indices.iter().enumerate() {
                            mle[model_idx] = r.params[slot];
                        }
                        diag.completed = r.loglik.is_finite();
                        // NLopt is a point-MLE optimiser; no posterior.
                        (r.loglik, f64::NAN, mle)
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: nlopt optimize_cell failed at grid {}/start {}: {}",
                            grid_idx, start_idx, e
                        );
                        (f64::NEG_INFINITY, f64::NAN, params.clone())
                    }
                }
            }
        };
        let elapsed = job_t0.elapsed().as_secs_f64();

        let seed_dir = &seed_dirs[seed_idx];
        let start_dir = profile_point_start_dir(seed_dir, grid_idx, start_idx);
        if let Err(e) = std::fs::create_dir_all(&start_dir) {
            eprintln!("warning: cannot create {}: {}", start_dir.display(), e);
            return;
        }

        let mle_toml = render_mle_toml(&if2_params, &focal_values,
            &focal_grids.iter().map(|fg| fg.name.as_str()).collect::<Vec<_>>(),
            &mle_params, final_loglik, final_log_posterior, &diag);
        let _ = std::fs::write(start_dir.join("mle.toml"), mle_toml);

        // Per-start run.json. parent_profile_hash references THIS
        // seed's profile content hash (not the umbrella's), so leaves
        // walk back to their per-seed parent regardless of single- vs
        // multi-seed layout.
        let parent_profile_hash = &per_seed_hashes[seed_idx];
        let start_hash_input = format!(
            "{}|point={}|start={}|seed={}",
            parent_profile_hash, grid_idx, start_idx, job_seed,
        );
        let start_hash = ContentHash::from_bytes(start_hash_input.as_bytes())
            .full().to_string();
        let start_run = Run {
            hash: start_hash,
            version: crate::version::VERSION_SHORT.to_string(),
            created_at: crate::cas::iso8601_utc(std::time::SystemTime::now()),
            argv: argv.clone(),
            status: RunStatus::Completed { wall_time_seconds: elapsed },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: String::new(),
                stage: profile_algo.method_kind().as_str().to_string(),
                method: profile_algo.method_kind(),
                backend: profile_algo.backend(),
                seed: job_seed,
                n_chains: 1,
                algorithm: match profile_algo {
                    ProfileAlgo::If2 => serde_json::json!({
                        "particles":  n_particles,
                        "iterations": n_iterations,
                        "cooling":    cooling,
                        "dt":         dt,
                    }),
                    ProfileAlgo::Pmmh => serde_json::json!({
                        "steps":     pmmh_steps,
                        "particles": pmmh_particles,
                        "rho":       pmmh_rho_opt,
                        "dt":        dt,
                    }),
                    ProfileAlgo::Nlopt(_) => serde_json::json!({
                        "tolerance": 1e-4,
                        "max_evals": 1500,
                        "dt":        compiled.model.simulation.dt.unwrap_or(dt),
                    }),
                },
                best_loglik: if final_loglik.is_finite() { Some(final_loglik) } else { None },
                best_chain:  Some(0),
                starts_from: None,
                derived_from: None,
                parent_profile_hash: Some(parent_profile_hash.clone()),
                profile_point_idx:   Some(grid_idx),
                profile_start_idx:   Some(start_idx),
                // gh#83/gh#85 step 9: per-cell provenance is the
                // *parent* profile's responsibility — the cell run
                // here only carries the grid-point payload.
                parameters_provenance: Default::default(),
                init_provenance: None,
            }),
        };
        if let Err(e) = start_run.write(&start_dir) {
            eprintln!("warning: could not write {}/run.json: {}",
                start_dir.display(), e);
        }

        // Progress tick.
        overall_pb.inc(1);
        if plain {
            let done = overall_pb.position();
            let ready = progress_throttle.lock()
                .map(|mut t| t.ready()).unwrap_or(true);
            if ready || done == total_jobs as u64 {
                log::info!("profile: {}/{} jobs complete", done, total_jobs);
            }
        }

        // Per-seed rollup (throttled).
        let should_rewrite = {
            let mut last = rollup_throttle.lock().unwrap();
            let now = std::time::Instant::now();
            if now.duration_since(*last) >= std::time::Duration::from_secs(1) {
                *last = now;
                true
            } else { false }
        };
        if should_rewrite {
            if let Err(e) = rewrite_rollup(seed_dir, &focal_names_ordered,
                &if2_params, grid_points.len()) {
                eprintln!("warning: rollup rewrite failed: {}", e);
            }
        }

        // Cross-seed summary (throttled). For N=1 the aggregate is
        // the trivial copy of the single seed's profile.tsv with
        // zero-width spread columns — still written so the umbrella's
        // summary.tsv is the universal user-facing artifact.
        let should_summary = {
            let mut last = summary_throttle.lock().unwrap();
            let now = std::time::Instant::now();
            if now.duration_since(*last) >= std::time::Duration::from_secs(2) {
                *last = now;
                true
            } else { false }
        };
        if should_summary {
            if let Err(e) = write_cross_seed_summary(
                &umbrella_dir, &seed_dirs, &focal_names_ordered, &if2_params)
            {
                eprintln!("warning: summary rewrite failed: {}", e);
            }
        }
    });

    overall_pb.finish_with_message("done");

    // Final per-seed rollup rewrites + cross-seed summary (unthrottled).
    for seed_dir in &seed_dirs {
        if let Err(e) = rewrite_rollup(seed_dir, &focal_names_ordered,
            &if2_params, grid_points.len())
        {
            eprintln!("warning: final rollup rewrite failed: {}", e);
        }
    }
    if let Err(e) = write_cross_seed_summary(
        &umbrella_dir, &seed_dirs, &focal_names_ordered, &if2_params)
    {
        eprintln!("warning: final summary rewrite failed: {}", e);
    }

    // Patch each per-seed (and umbrella) run.json with total wall time.
    let total_wall = start_time.elapsed().as_secs_f64();
    for seed_dir in &seed_dirs {
        if let Ok(mut pr) = Run::read(seed_dir) {
            pr.status = RunStatus::Completed { wall_time_seconds: total_wall };
            let _ = pr.write(seed_dir);
        }
    }
    if let Ok(mut pr) = Run::read(&umbrella_dir) {
        pr.status = RunStatus::Completed { wall_time_seconds: total_wall };
        let _ = pr.write(&umbrella_dir);
    }

    // Mirror the user-facing TSV: the umbrella's summary.tsv is the
    // universal artifact — for N=1 it's a one-row aggregate of the
    // single seed; for N>1 it's the cross-seed sensitivity summary.
    let mirror_src = umbrella_dir.join("summary.tsv");
    if let Some(ref path) = output_tsv_path {
        if mirror_src.exists() {
            match std::fs::copy(&mirror_src, path) {
                Ok(_) => eprintln!("written to {}", path),
                Err(e) => eprintln!("warning: could not copy {} to {}: {}",
                    mirror_src.display(), path, e),
            }
        }
    } else {
        eprintln!("output: {}", mirror_src.display());
    }
}

/// Render a per-start MLE TOML file. Human-readable; also the format
/// `rewrite_rollup` reads back to reconstruct the rollup.
///
/// Seed `model.parameters[i].value` from an init-mode companion file
/// **before** the resolver runs, so single-point init modes
/// (`--init from_params --params <toml>` and `--init from_mle --mle
/// <path>`) deliver their values to Phase 2 (resolver) in addition
/// to Phase 3 (chain init). Without this bridge, the resolver fires
/// `UnsetRequired` for any parameter that has no DSL default + no
/// `[fixed]` entry, even when the user has explicitly named a file
/// containing the value.
///
/// **File wins (aggressive)**: if the user typed
/// `--init from_params --params start.toml` with `beta = 0.5`, and the
/// model's DSL also declares `beta = 0.3` as a default, beta resolves
/// to 0.5. The user explicitly named the file as authoritative;
/// silently preferring the DSL default would be a footgun.
///
/// **Per-chain-varying init modes** (`from_prior`, `from_posterior`)
/// don't seed model_pre because there's no single value to seed.
/// Users of those modes need to also supply a base-value source
/// (`--fit` with `[estimate].start`, or `--fixed-file`). Documented
/// behaviour, not a hidden constraint.
///
/// Returns `Ok(())` for init modes that don't seed (single, uniform,
/// lhs, from_prior, from_posterior, survey_top_k) — the caller can
/// blindly call this and let the resolver handle errors downstream.
///
/// Takes `&mut Vec<Parameter>` rather than `&mut Model` because the
/// helper only ever touches `model.parameters[i].value`. Narrower
/// surface = simpler tests + less coupling to the rest of the IR.
fn seed_params_from_init_method(
    params: &mut Vec<ir::parameter::Parameter>,
    init_method: &crate::fit::init::InitMethod,
) -> Result<(), String> {
    use crate::fit::init::{InitMethod, MleSource};
    let file_values: HashMap<String, f64> = match init_method {
        InitMethod::FromParams { path } => {
            crate::util::load_params_toml(&path.to_string_lossy())
                .map_err(|e| format!(
                    "loading --params for --init from_params: {}", e))?
        }
        InitMethod::FromMle { source } => {
            let path = match source {
                MleSource::File(p) => p.clone(),
                MleSource::FitDir(dir) => {
                    let mle = dir.join("mle.toml");
                    if mle.is_file() { mle }
                    else {
                        let final_p = dir.join("final_params.toml");
                        if final_p.is_file() { final_p }
                        else {
                            return Err(format!(
                                "--init from_mle: neither {}/mle.toml nor \
                                 {}/final_params.toml exists",
                                dir.display(), dir.display()));
                        }
                    }
                }
            };
            crate::fit::chain_starts::load_mle_toml(&path)
                .map_err(|e| format!(
                    "loading --mle for --init from_mle: {:?}", e))?
        }
        _ => return Ok(()),
    };
    for p in params.iter_mut() {
        if let Some(&v) = file_values.get(&p.name) {
            p.value = Some(v);
        }
    }
    Ok(())
}

/// gh#118: focal-parameter prior contribution at the cell's fixed
/// focal values. Profile pins focal params at grid values and
/// estimates only the nuisance set, so the PMMH engine's
/// `log_prior` sum covers nuisance priors only. To make the
/// emitted `log_posterior` column comparable to `camdl fit run`'s
/// `log_posterior` (which sums priors over the full estimated
/// set), add this offset.
///
/// The offset is constant within a cell (focal values are fixed by
/// definition) and additive: corrected `log_posterior = log_lik +
/// nuisance_log_prior + focal_log_prior_offset`.
///
/// Returns `NEG_INFINITY` when any focal value sits outside its
/// prior's support (e.g. a uniform prior whose declared bounds
/// don't cover the cell's grid value). That's semantically correct
/// — the cell has zero prior probability — and propagates to a NaN
/// in the rendered `mle.toml` via the existing `is_finite()` guard.
///
/// Hierarchical priors on focal params are not currently supported
/// (the env-free `Prior::log_density` falls back to `NEG_INFINITY`
/// for the Hierarchical variant per `prior.rs:143`). In practice
/// focal params are typically simple univariate distributions
/// (uniform / normal / log_normal); the rare case of a
/// hierarchical focal prior would need a separate code path that
/// passes a `NamedParams` env into `log_density_env`.
fn compute_focal_log_prior_offset(
    focal_specs:  &[sim::inference::if2::EstimatedParam],
    focal_priors: &[sim::inference::prior::Prior],
    focal_values: &[f64],
) -> f64 {
    debug_assert_eq!(focal_specs.len(), focal_priors.len(),
        "focal_specs and focal_priors must align by index");
    debug_assert_eq!(focal_specs.len(), focal_values.len(),
        "focal_specs and focal_values must align by index");
    focal_specs.iter()
        .zip(focal_priors.iter())
        .zip(focal_values.iter())
        .map(|((spec, prior), &val)| {
            let z = spec.to_transformed(val);
            prior.log_density(val, z)
        })
        .sum()
}

/// gh#74 Option B: emits a trailing `[diagnostics]` block holding the
/// per-start convergence record. Older `mle.toml` files (pre-gh#74)
/// have no `[diagnostics]` block; the rollup tolerates the absence
/// and renders NaN for that start's contribution to the per-cell
/// aggregate. Cached CAS dirs from older runs therefore continue to
/// roll up, just without diagnostics columns populated.
fn render_mle_toml(
    if2_params: &[sim::inference::if2::EstimatedParam],
    focal_values: &[f64],
    focal_names: &[&str],
    mle: &[f64],
    final_loglik: f64,
    final_log_posterior: f64,
    diag: &crate::profile_diagnostics::PerStartDiagnostics,
) -> String {
    let mut body = String::new();
    body.push_str("# Per-start MLE for one profile grid point.\n\n");
    body.push_str(&format!("final_loglik = {}\n", final_loglik));
    // gh#109: PMMH cells carry the MAP joint log-posterior. Other
    // algorithms (IF2, NLopt) write NaN here; the rollup reads-or-NaN
    // and the TSV column reads NaN as missing.
    if final_log_posterior.is_finite() {
        body.push_str(&format!("final_log_posterior = {}\n", final_log_posterior));
    } else if final_log_posterior.is_nan() {
        // Skip — keep mle.toml tight for IF2/NLopt cells.
    } else {
        body.push_str("final_log_posterior = -inf\n");
    }
    body.push('\n');
    body.push_str("[focal]\n");
    for (name, v) in focal_names.iter().zip(focal_values.iter()) {
        body.push_str(&format!("{} = {}\n", name, v));
    }
    body.push_str("\n[mle]\n");
    for spec in if2_params.iter() {
        body.push_str(&format!("{} = {}\n", spec.name, mle[spec.index]));
    }
    body.push_str("\n[diagnostics]\n");
    body.push_str(&diag.to_toml_fragment());
    body
}

/// Scan the per-start CAS tree and rewrite `profile.tsv` as the
/// derived rollup. One row per grid point, each row the winning start
/// (max final_loglik) across `n_starts`. Written atomically via
/// tmp-then-rename so concurrent rollups (from racing threads) never
/// expose a truncated intermediate.
///
/// gh#74 Option B: each row gains seven per-cell diagnostic columns
/// (`DIAG_COLUMNS`). Aggregated by `CellDiagnostics::aggregate` from
/// the K per-start `[diagnostics]` blocks parsed back out of each
/// `mle.toml`.
fn rewrite_rollup(
    profile_dir: &Path,
    focal_names: &[String],
    if2_params: &[sim::inference::if2::EstimatedParam],
    n_grid_points: usize,
) -> std::io::Result<()> {
    // For each grid point, find the winning start by scanning its
    // start_{k}/ subdirs for mle.toml. If no starts have finished yet
    // for this point, skip the row (partial rollup — consumers see
    // only completed points).
    let mut rows: Vec<RollupRow> = Vec::new();
    for gi in 0..n_grid_points {
        let point_dir = profile_point_dir(profile_dir, gi);
        let Ok(dir_iter) = std::fs::read_dir(&point_dir) else { continue; };

        let mut best: Option<ParsedMle> = None;
        let mut wall_time_sum: f64 = 0.0;
        let mut best_start: Option<usize> = None;
        // gh#74 Option B: collect per-start diagnostics + final logliks
        // so the rollup can aggregate them into per-cell columns. The
        // vectors are keyed by start_idx via the parallel `(starts,
        // finals)` arrays — order doesn't matter for the aggregator
        // (it's symmetric across starts).
        let mut starts_diag: Vec<crate::profile_diagnostics::PerStartDiagnostics>
            = Vec::new();
        let mut starts_final_ll: Vec<f64> = Vec::new();
        for entry in dir_iter.flatten() {
            let fname = entry.file_name();
            let name = fname.to_string_lossy();
            let Some(start_idx_str) = name.strip_prefix("start_") else { continue; };
            let Ok(start_idx) = start_idx_str.parse::<usize>() else { continue; };
            let start_dir = entry.path();

            // Use run.json's wall time for summation. Skip starts
            // with missing/broken run.json or still-running starts —
            // they're incomplete.
            let Ok(start_run) = Run::read(&start_dir) else { continue; };
            let Some(t) = start_run.status.wall_time_seconds() else { continue; };
            wall_time_sum += t;

            let mle_path = start_dir.join("mle.toml");
            let Ok(mle_text) = std::fs::read_to_string(&mle_path) else { continue; };
            let Some(parsed) = parse_mle_toml(&mle_text, if2_params, focal_names) else { continue; };
            // Re-parse the same TOML doc for the diagnostics table.
            // Cheap (the file is small); keeps `parse_mle_toml` focused
            // on its existing surface.
            let doc: toml::Value = match toml::from_str(&mle_text) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let diag = crate::profile_diagnostics::PerStartDiagnostics::from_toml(&doc);
            starts_diag.push(diag);
            starts_final_ll.push(parsed.final_loglik);

            match &best {
                Some(b) if parsed.final_loglik <= b.final_loglik => {}
                _ => {
                    best = Some(parsed);
                    best_start = Some(start_idx);
                }
            }
        }

        if let (Some(best), Some(best_start)) = (best, best_start) {
            let _ = gi;  // order preserved by outer loop; field elided.
            let cell_diag = crate::profile_diagnostics::CellDiagnostics::aggregate(
                &starts_diag, &starts_final_ll,
            );
            rows.push(RollupRow {
                focal_values: best.focal_values,
                best_loglik: best.final_loglik,
                best_log_posterior: best.final_log_posterior,
                best_start_idx: best_start,
                mle: best.mle,
                wall_time_sum,
                diag: cell_diag,
            });
        }
    }

    // Render.
    let mut body = String::new();
    body.push_str(&format!("# {}\n", crate::version::VERSION));
    body.push_str(&format!("# total_points={} completed={}\n",
        n_grid_points, rows.len()));
    for name in focal_names { body.push_str(&format!("{}\t", name)); }
    // gh#109: `best_log_posterior` sits next to `best_loglik` for
    // natural grouping. Header-based parsers (the camdl-book scripts
    // grep by column name) are unaffected; position-based parsers
    // shift by one column (already brittle pre-change).
    body.push_str("best_loglik\tbest_log_posterior\tbest_start_idx\twall_time_seconds");
    for spec in if2_params.iter() { body.push_str(&format!("\t{}", spec.name)); }
    for c in crate::profile_diagnostics::DIAG_COLUMNS {
        body.push_str(&format!("\t{}", c));
    }
    body.push('\n');
    for row in &rows {
        for v in &row.focal_values { body.push_str(&format!("{:.4}\t", v)); }
        // NaN-as-text: `nan` is the TSV-friendly form (Python pandas,
        // R read.table, jq all parse this). On IF2/NLopt cells the
        // column reads `nan`; on PMMH cells it reads a real f64.
        let lp_field = if row.best_log_posterior.is_finite() {
            format!("{:.4}", row.best_log_posterior)
        } else {
            "nan".to_string()
        };
        body.push_str(&format!("{:.4}\t{}\t{}\t{:.3}",
            row.best_loglik, lp_field, row.best_start_idx, row.wall_time_sum));
        for spec in if2_params.iter() {
            body.push_str(&format!("\t{:.6}", row.mle[spec.index]));
        }
        body.push('\t');
        body.push_str(&row.diag.render_tsv_row());
        body.push('\n');
    }

    // Atomic write.
    let final_path = profile_dir.join("profile.tsv");
    let tmp_path = profile_dir.join("profile.tsv.tmp");
    std::fs::write(&tmp_path, body)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

struct RollupRow {
    focal_values: Vec<f64>,
    best_loglik: f64,
    /// gh#109: MAP joint log-posterior for the winning start. NaN on
    /// IF2 / NLopt cells (point-MLE algorithms with no posterior),
    /// and on cells whose mle.toml predates gh#109 (cached CAS dirs).
    best_log_posterior: f64,
    best_start_idx: usize,
    mle: Vec<f64>,
    wall_time_sum: f64,
    /// Per-cell aggregate diagnostics (gh#74 Option B).
    diag: crate::profile_diagnostics::CellDiagnostics,
}

struct ParsedMle {
    final_loglik: f64,
    /// gh#109: MAP joint log-posterior. NaN when the per-start
    /// mle.toml omits the field (older / non-PMMH cells).
    final_log_posterior: f64,
    focal_values: Vec<f64>,
    mle: Vec<f64>,
}

fn parse_mle_toml(
    text: &str,
    if2_params: &[sim::inference::if2::EstimatedParam],
    focal_names: &[String],
) -> Option<ParsedMle> {
    let doc: toml::Value = toml::from_str(text).ok()?;
    let final_loglik = toml_as_f64(doc.get("final_loglik")?)?;
    // gh#109: optional — present on PMMH cells, omitted on
    // IF2/NLopt. Missing → NaN; the rollup TSV column reads NaN
    // as missing.
    let final_log_posterior = doc.get("final_log_posterior")
        .and_then(toml_as_f64)
        .unwrap_or(f64::NAN);
    let focal = doc.get("focal")?.as_table()?;
    let mle = doc.get("mle")?.as_table()?;

    // Extract focal values in the caller's declared order (the column
    // order of the rollup TSV header), not in TOML key order.
    let mut focal_values: Vec<f64> = Vec::with_capacity(focal_names.len());
    for name in focal_names {
        let v = focal.get(name).and_then(toml_as_f64)?;
        focal_values.push(v);
    }

    let mle_len = if2_params.iter().map(|s| s.index).max().unwrap_or(0) + 1;
    let mut mle_values: Vec<f64> = vec![0.0; mle_len];
    for spec in if2_params.iter() {
        if let Some(v) = mle.get(&spec.name).and_then(toml_as_f64) {
            if mle_values.len() <= spec.index {
                mle_values.resize(spec.index + 1, 0.0);
            }
            mle_values[spec.index] = v;
        }
    }

    Some(ParsedMle { final_loglik, final_log_posterior, focal_values, mle: mle_values })
}

/// Accept TOML numeric values whether they serialised as Integer
/// (`R0 = 50`) or Float (`R0 = 50.0`). `toml::Value::as_float()`
/// returns `None` for Integers, which would silently drop any focal
/// value that happened to be a whole number.
fn toml_as_f64(v: &toml::Value) -> Option<f64> {
    match v {
        toml::Value::Float(f)   => Some(*f),
        toml::Value::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

/// Cross-seed aggregator. Reads each per-seed `profile.tsv` and emits
/// `summary.tsv` at the umbrella directory: one row per grid point.
///
/// Schema (per gh#30 — option A):
///
/// * Bare-name columns (`loglik`, `<param>`) are always present and
///   carry the central value: the single per-seed value when
///   n_seeds=1, the mean across seeds when n_seeds>1. A reader doing
///   `df["loglik"]` and `df["R0"]` works identically in both cases —
///   the n_seeds=1 case (the common case for first-time profiles, the
///   camdl-book chapters, and "what does the surface look like"
///   checks) doesn't have to learn the multi-seed schema to read the
///   single value back.
/// * Spread-diagnostic columns (`loglik_sd / _min / _max`,
///   `<param>_sd`) are emitted *additively* and only when n_seeds>1,
///   where they describe stochastic IF2 instability across replicate
///   chains. High `loglik_sd` at a grid point flags an untrustworthy
///   conditional MLE.
///
/// Header preserves bare/`_sd` adjacency (`R0  R0_sd  alpha
/// alpha_sd`) so per-parameter pairs read together. The per-cell
/// finite-seed count is inlined as elevated `loglik_sd` rather than a
/// separate column; users who need the raw count can read the
/// per-seed `replicates/seed_*/profile.tsv` files.
///
/// Atomic write (tmp-then-rename) so concurrent throttled rewrites
/// from the rayon pool can't expose a half-written summary.
fn write_cross_seed_summary(
    umbrella_dir: &Path,
    seed_dirs: &[PathBuf],
    focal_names: &[String],
    if2_params: &[sim::inference::if2::EstimatedParam],
) -> std::io::Result<()> {
    use std::collections::BTreeMap;
    use crate::profile_diagnostics::{CellDiagnostics, DIAG_COLUMNS};

    // Per-cell sample bag. For each focal-key (canonicalised TSV
    // column string) we accumulate a (best_loglik, mle_vec,
    // per_seed_diag) tuple per seed. The per-seed diag is parsed out
    // of the trailing diagnostic columns of each row of the per-seed
    // profile.tsv (gh#74 Option B); cross-seed averaging happens in
    // the render loop below via `CellDiagnostics::average_across_seeds`.
    struct PerSeedSample {
        loglik:    f64,
        /// gh#109: MAP joint log-posterior. NaN when the per-seed
        /// profile.tsv's cell shows `nan` (IF2/NLopt or pre-gh#109
        /// cached cells).
        lp:        f64,
        mle:       Vec<f64>,
        diag:      CellDiagnostics,
    }
    let mut by_grid: BTreeMap<Vec<String>, Vec<PerSeedSample>> = BTreeMap::new();
    let mle_len = if2_params.iter().map(|s| s.index).max().unwrap_or(0) + 1;

    for seed_dir in seed_dirs {
        let path = seed_dir.join("profile.tsv");
        let Ok(text) = std::fs::read_to_string(&path) else { continue; };
        for line in text.lines() {
            if line.starts_with('#') { continue; }
            let cols: Vec<&str> = line.split('\t').collect();
            // Header row uses literal column names; skip it.
            if cols.get(focal_names.len()).copied() == Some("best_loglik") {
                continue;
            }
            // Layout (gh#74 + gh#109): focal_1 ... focal_N |
            //   best_loglik | best_log_posterior | best_start_idx |
            //   wall_time_seconds | mle_param_1 ... mle_param_M |
            //   acc_rate_avg | acc_rate_min | loglik_spread_starts |
            //   loglik_rhat_starts | starts_n_completed |
            //   iterations_used | cooling_final
            let base_cols = focal_names.len() + 4 + if2_params.len();
            if cols.len() < base_cols { continue; }

            let focal_key: Vec<String> = cols[..focal_names.len()]
                .iter().map(|s| s.trim().to_string()).collect();
            let Ok(best_loglik) = cols[focal_names.len()].parse::<f64>() else { continue; };
            // gh#109: best_log_posterior in the next column. `nan`
            // for non-PMMH cells or cached pre-gh#109 rollups (the
            // rollup writer emits the literal "nan" for those).
            let best_lp = parse_summary_cell(cols[focal_names.len() + 1]);

            let mle_start = focal_names.len() + 4;
            let mut mle = vec![f64::NAN; mle_len];
            for (i, spec) in if2_params.iter().enumerate() {
                if let Some(s) = cols.get(mle_start + i) {
                    if let Ok(v) = s.parse::<f64>() {
                        if spec.index < mle.len() {
                            mle[spec.index] = v;
                        }
                    }
                }
            }

            // Diagnostic columns sit at fixed offsets after the MLE
            // block. Parse them positionally — DIAG_COLUMNS pins the
            // order. Missing cells (older `profile.tsv` from before
            // gh#74) parse as NaN / 0, which the aggregator then
            // surfaces as NaN at the summary level.
            let diag_start = base_cols;
            let parse_at = |off: usize| -> f64 {
                cols.get(diag_start + off)
                    .map(|s| parse_summary_cell(s)).unwrap_or(f64::NAN)
            };
            let starts_n_completed = cols.get(diag_start + 4)
                .and_then(|s| s.trim().parse::<usize>().ok())
                .unwrap_or(0);
            let diag = CellDiagnostics {
                acc_rate_avg:        parse_at(0),
                acc_rate_min:        parse_at(1),
                loglik_spread_starts: parse_at(2),
                loglik_rhat_starts:   parse_at(3),
                starts_n_completed,
                iterations_used:      parse_at(5),
                cooling_final:        parse_at(6),
            };

            by_grid.entry(focal_key).or_default().push(PerSeedSample {
                loglik: best_loglik, lp: best_lp, mle, diag,
            });
        }
    }

    let n_seeds = seed_dirs.len();
    let multi_seed = n_seeds > 1;

    let mut body = String::new();
    body.push_str(&format!("# {} cross-seed summary across {} seed{}\n",
        crate::version::VERSION, n_seeds, if multi_seed { "s" } else { "" }));
    body.push_str(&format!("# n_grid_points={} n_seeds={}\n",
        by_grid.len(), n_seeds));

    // Header: focal | loglik [| loglik_sd loglik_min loglik_max] |
    //         <param_1> [| <param_1>_sd] | <param_2> [| <param_2>_sd] | ...
    //         | <diagnostic_1> | <diagnostic_2> | ... (gh#74 Option B,
    //         always appended; per-algorithm columns NaN where the
    //         algorithm has no value).
    for name in focal_names { body.push_str(&format!("{}\t", name)); }
    body.push_str("loglik");
    if multi_seed {
        body.push_str("\tloglik_sd\tloglik_min\tloglik_max");
    }
    // gh#109: log_posterior column (and multi-seed spread) sits
    // alongside loglik so the cross-seed summary mirrors the
    // per-seed profile.tsv layout.
    body.push_str("\tlog_posterior");
    if multi_seed {
        body.push_str("\tlog_posterior_sd\tlog_posterior_min\tlog_posterior_max");
    }
    for spec in if2_params.iter() {
        body.push_str(&format!("\t{}", spec.name));
        if multi_seed {
            body.push_str(&format!("\t{}_sd", spec.name));
        }
    }
    for c in DIAG_COLUMNS {
        body.push_str(&format!("\t{}", c));
    }
    body.push('\n');

    for (focal_key, samples) in &by_grid {
        for v in focal_key { body.push_str(&format!("{}\t", v)); }
        let logliks: Vec<f64> = samples.iter().map(|s| s.loglik)
            .filter(|x| x.is_finite()).collect();
        let (mean_ll, sd_ll, min_ll, max_ll) = summary_stats(&logliks);
        body.push_str(&format!("{:.4}", mean_ll));
        if multi_seed {
            body.push_str(&format!("\t{:.4}\t{:.4}\t{:.4}", sd_ll, min_ll, max_ll));
        }
        // gh#109: log_posterior cross-seed aggregate. Same shape as
        // loglik. When every seed's sample is NaN (e.g. all IF2 cells),
        // emit "nan" so the column is round-trippable.
        let lps: Vec<f64> = samples.iter().map(|s| s.lp)
            .filter(|x| x.is_finite()).collect();
        let (mean_lp, sd_lp, min_lp, max_lp) = summary_stats(&lps);
        if mean_lp.is_finite() {
            body.push_str(&format!("\t{:.4}", mean_lp));
        } else {
            body.push_str("\tnan");
        }
        if multi_seed {
            if sd_lp.is_finite() || mean_lp.is_finite() {
                body.push_str(&format!("\t{:.4}\t{:.4}\t{:.4}", sd_lp, min_lp, max_lp));
            } else {
                body.push_str("\tnan\tnan\tnan");
            }
        }
        for spec in if2_params.iter() {
            let vals: Vec<f64> = samples.iter()
                .filter_map(|s| s.mle.get(spec.index).copied())
                .filter(|x| x.is_finite()).collect();
            let (m, s, _, _) = summary_stats(&vals);
            body.push_str(&format!("\t{:.6}", m));
            if multi_seed {
                body.push_str(&format!("\t{:.6}", s));
            }
        }
        // Cross-seed aggregate of the per-seed diagnostic blocks.
        // For n_seeds=1 this is a passthrough; for n>1 we average
        // each diagnostic across seeds (starts_n_completed sums).
        let per_seed_diags: Vec<CellDiagnostics> =
            samples.iter().map(|s| s.diag.clone()).collect();
        let agg = CellDiagnostics::average_across_seeds(&per_seed_diags);
        body.push('\t');
        body.push_str(&agg.render_tsv_row());
        body.push('\n');
    }

    let final_path = umbrella_dir.join("summary.tsv");
    let tmp_path = umbrella_dir.join("summary.tsv.tmp");
    std::fs::write(&tmp_path, body)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
}

/// Parse a per-seed `profile.tsv` diagnostic cell into f64. Accepts
/// `NaN`/`Inf`/`-Inf` plus standard numerics; everything else falls
/// through to NaN so a malformed row doesn't poison the rollup.
fn parse_summary_cell(s: &str) -> f64 {
    match s.trim() {
        "NaN"  => f64::NAN,
        "Inf"  => f64::INFINITY,
        "-Inf" => f64::NEG_INFINITY,
        v      => v.parse::<f64>().unwrap_or(f64::NAN),
    }
}

/// (mean, sd, min, max) of a slice. Empty input returns NaN/0/inf/-inf;
/// callers should treat NaN as "no data" not "zero data."
fn summary_stats(xs: &[f64]) -> (f64, f64, f64, f64) {
    if xs.is_empty() {
        return (f64::NAN, 0.0, f64::INFINITY, f64::NEG_INFINITY);
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let sd = if xs.len() > 1 {
        (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else { 0.0 };
    let min = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, sd, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::if2::EstimatedParam;
    use sim::inference::types::Transform;

    fn estimated(name: &str, index: usize) -> EstimatedParam {
        EstimatedParam {
            name: name.into(), index, initial: 0.0, rw_sd: 0.0,
            transform: Transform::None, lower: 0.0, upper: 1.0,
            rw_sd_auto: false, ivp: false,
        }
    }

    /// Helper: write a per-seed `profile.tsv` in the expected
    /// pre-aggregation layout (matches what the in-process profile
    /// driver emits on disk, line 929-930 of this file).
    fn write_per_seed_profile(
        seed_dir: &std::path::Path,
        focal_names: &[&str],
        rows: &[(Vec<f64>, f64, Vec<f64>)],  // (focal vals, best_loglik, mle vals)
    ) {
        std::fs::create_dir_all(seed_dir).unwrap();
        let mut s = String::new();
        for n in focal_names { s.push_str(&format!("{}\t", n)); }
        // gh#109: best_log_posterior column sits between best_loglik
        // and best_start_idx in the per-seed profile.tsv. Test
        // helpers emit `nan` for the new column (the test rows
        // model IF2/legacy cells — no posterior); the gh#109-
        // specific tests below override with explicit values.
        s.push_str("best_loglik\tbest_log_posterior\tbest_start_idx\twall_time_seconds");
        for i in 0..rows[0].2.len() { s.push_str(&format!("\tparam_{}", i)); }
        s.push('\n');
        for (focal, ll, mle) in rows {
            for v in focal { s.push_str(&format!("{}\t", v)); }
            s.push_str(&format!("{:.4}\tnan\t0\t0.0", ll));
            for v in mle { s.push_str(&format!("\t{:.6}", v)); }
            s.push('\n');
        }
        std::fs::write(seed_dir.join("profile.tsv"), s).unwrap();
    }

    fn data_lines(text: &str) -> Vec<&str> {
        text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).collect()
    }

    /// gh#118 regression: `log_posterior` in profile output was built
    /// from the nuisance-only prior sum, silently excluding focal-
    /// parameter prior contributions. The expected behaviour is that
    /// `log_posterior = log_likelihood + Σ log_prior(ALL estimated
    /// parameters)`, matching `camdl fit run`'s definition. This
    /// regression pins the focal-prior offset computation against the
    /// hand-computed values in the gh#118 forensic table on the
    /// wa_weak seed-timing model.
    #[test]
    fn focal_log_prior_offset_matches_gh118_forensic_table() {
        use sim::inference::prior::Prior;

        // From gh#118: model has two focal params
        //   tau:    instant ~ uniform(lower=-86, upper=0); pinned at -5
        //   n_seed: count   ~ log_normal(mu=log 5, sigma=1); pinned at 100
        let tau_spec = EstimatedParam {
            name: "tau".into(), index: 0, initial: -5.0, rw_sd: 0.0,
            transform: Transform::None,
            lower: -86.0, upper: 0.0,
            rw_sd_auto: false, ivp: false,
        };
        let n_seed_spec = EstimatedParam {
            name: "n_seed".into(), index: 1, initial: 100.0, rw_sd: 0.0,
            transform: Transform::Log { lo: 1.0, hi: 1000.0 },
            lower: 1.0, upper: 1000.0,
            rw_sd_auto: false, ivp: false,
        };
        let tau_prior = Prior::Uniform { lower: -86.0, upper: 0.0 };
        let n_seed_prior = Prior::TransformedNormal {
            mean: 5.0_f64.ln(), sd: 1.0,
        };

        let offset = compute_focal_log_prior_offset(
            &[tau_spec, n_seed_spec],
            &[tau_prior, n_seed_prior],
            &[-5.0, 100.0],
        );

        // Hand-computed reference values from gh#118 (issue body §"Evidence"):
        //   log_prior(τ uniform[-86, 0] at -5)         = -ln(86)
        //                                              ≈ -4.4543
        //   log_prior(n_seed LogN(log 5, 1) at 100)
        //     = -ln(100) - ½ ln(2π) - ½ ((ln 100 - ln 5)/1)²
        //     = -4.6052 - 0.9189 - 4.4878
        //     ≈ -10.0119
        //   total ≈ -14.4662
        let expected = -(86.0_f64.ln())
                     + (-(100.0_f64.ln())
                        - 0.5 * (2.0 * std::f64::consts::PI).ln()
                        - 0.5 * ((100.0_f64.ln() - 5.0_f64.ln()) / 1.0).powi(2));
        assert!((offset - expected).abs() < 1e-9,
            "offset {} does not match analytic expected {}", offset, expected);
        // Sanity-check the analytic value lines up with the issue's
        // reported gap (+14.467 with 3-decimal precision in the issue).
        assert!((offset - (-14.4662)).abs() < 1e-3,
            "offset {} does not match gh#118 forensic table -14.4662", offset);
    }

    /// Focal value outside a Uniform prior's support produces
    /// -∞ — the cell is at zero prior probability. The wiring must
    /// surface this as a non-finite log_posterior (NaN sentinel in
    /// the emitted mle.toml), not a spuriously finite number.
    #[test]
    fn focal_log_prior_offset_out_of_support_is_neg_infinity() {
        use sim::inference::prior::Prior;

        let spec = EstimatedParam {
            name: "tau".into(), index: 0, initial: -100.0, rw_sd: 0.0,
            transform: Transform::None,
            lower: -86.0, upper: 0.0,
            rw_sd_auto: false, ivp: false,
        };
        let prior = Prior::Uniform { lower: -86.0, upper: 0.0 };
        let offset = compute_focal_log_prior_offset(
            &[spec], &[prior], &[-100.0],
        );
        assert!(offset.is_infinite() && offset < 0.0,
            "expected NEG_INFINITY, got {}", offset);
    }

    /// Empty focal set (1D-profile-on-the-flat-case or
    /// no-focal-priors path): offset is exactly zero.
    #[test]
    fn focal_log_prior_offset_empty_is_zero() {
        let offset = compute_focal_log_prior_offset(&[], &[], &[]);
        assert_eq!(offset, 0.0);
    }

    #[test]
    fn n1_summary_uses_bare_names_only() {
        // gh#30 option A, n=1 (the common case): the schema is
        //   <focal>  loglik  <param_1>  <param_2>  ... | <diag_cols>
        // No `_sd` / `_min` / `_max` — there's no aggregation to
        // describe. Diagnostic columns (gh#74 Option B) are appended
        // after the bare param block in `DIAG_COLUMNS` order.
        let tmp = tempfile::tempdir().unwrap();
        let umbrella = tmp.path();
        let seed_dir = umbrella.join("replicates").join("seed_1");
        write_per_seed_profile(
            &seed_dir,
            &["s0"],
            &[
                (vec![0.10], -42.5, vec![1.5, 0.3]),
                (vec![0.20], -38.1, vec![1.7, 0.4]),
            ],
        );
        let if2 = vec![estimated("R0", 0), estimated("alpha", 1)];
        write_cross_seed_summary(umbrella, &[seed_dir], &["s0".into()], &if2).unwrap();

        let text = std::fs::read_to_string(umbrella.join("summary.tsv")).unwrap();
        let lines = data_lines(&text);
        let header = lines[0];
        let cols: Vec<&str> = header.split('\t').collect();
        // gh#109: `log_posterior` sits next to `loglik` in the cross-seed
        // summary, mirroring the per-seed profile.tsv layout.
        let mut expected = vec!["s0", "loglik", "log_posterior", "R0", "alpha"];
        for c in crate::profile_diagnostics::DIAG_COLUMNS { expected.push(c); }
        assert_eq!(cols, expected,
            "n=1 schema must be focal + bare loglik + bare log_posterior \
             + bare params + diagnostics; got {:?}", cols);

        // Cross-seed spread columns must not leak through (gh#30
        // option A); the diagnostic columns are unrelated and are
        // expected to be present.
        for forbidden in &["loglik_sd", "loglik_min", "loglik_max",
                           "R0_sd", "alpha_sd", "n_seeds",
                           "mean_loglik", "max_loglik", "R0_mean", "alpha_mean"] {
            assert!(!header.contains(forbidden),
                "n=1 header must not contain {:?}: {}", forbidden, header);
        }

        assert_eq!(lines.len(), 3, "expected header + 2 grid rows: {:?}", lines);
        // gh#109: row_cols = focal(1) + loglik(1) + log_posterior(1) +
        // params(2) + DIAG_COLUMNS(7) = 12. Previously 11 before
        // log_posterior column.
        let row_cols = 5 + crate::profile_diagnostics::DIAG_COLUMNS.len();
        assert_eq!(lines[1].split('\t').count(), row_cols);
        assert_eq!(lines[2].split('\t').count(), row_cols);
    }

    #[test]
    fn multi_seed_summary_appends_spread_columns() {
        // gh#30 option A, n>1: bare names stay, `_sd / _min / _max`
        // are appended additively. Bare loglik = mean across seeds;
        // bare param = mean across seeds. gh#74 Option B diagnostic
        // columns are appended after the spread columns.
        let tmp = tempfile::tempdir().unwrap();
        let umbrella = tmp.path();
        let mut seed_dirs = Vec::new();
        for (idx, ll_offset, r0_off) in
            [(1usize, 0.0_f64, 0.0_f64), (2, 0.5, 0.05), (3, -0.5, -0.05)]
        {
            let seed_dir = umbrella.join("replicates").join(format!("seed_{}", idx));
            write_per_seed_profile(
                &seed_dir,
                &["s0"],
                &[
                    (vec![0.10], -42.5 + ll_offset, vec![1.5 + r0_off, 0.3]),
                    (vec![0.20], -38.1 + ll_offset, vec![1.7 + r0_off, 0.4]),
                ],
            );
            seed_dirs.push(seed_dir);
        }
        let if2 = vec![estimated("R0", 0), estimated("alpha", 1)];
        write_cross_seed_summary(umbrella, &seed_dirs, &["s0".into()], &if2).unwrap();

        let text = std::fs::read_to_string(umbrella.join("summary.tsv")).unwrap();
        let lines = data_lines(&text);
        let cols: Vec<&str> = lines[0].split('\t').collect();
        let mut expected = vec![
            "s0", "loglik", "loglik_sd", "loglik_min", "loglik_max",
            // gh#109: log_posterior aggregate sits between loglik
            // group and per-param group.
            "log_posterior", "log_posterior_sd", "log_posterior_min", "log_posterior_max",
            "R0", "R0_sd", "alpha", "alpha_sd",
        ];
        for c in crate::profile_diagnostics::DIAG_COLUMNS { expected.push(c); }
        assert_eq!(cols, expected, "n>1 schema: {:?}", cols);

        // Bare `loglik` value is the mean across seeds at the first
        // grid cell: mean(-42.5, -42.0, -43.0) = -42.5
        let row1: Vec<&str> = lines[1].split('\t').collect();
        let bare_loglik: f64 = row1[1].parse().unwrap();
        assert!((bare_loglik - (-42.5)).abs() < 1e-3,
            "bare loglik should be the cross-seed mean, got {}", bare_loglik);
    }

    // ── gh#39 data-file content hashing ─────────────────────────────────
    //
    // Note: the gh#38 `resolve_obs_family` standalone helper and its
    // tests moved to `crate::util::resolve_data_specs` (gh#90), where
    // single-PATH + --obs and named-pair forms share one dispatch.
    // Equivalence tests for family expansion live in
    // `util.rs::gh90_resolver_tests`.

    /// Construct a `ProfileInputs` with all content fields fixed to
    /// stable placeholders. Caller overrides only the field under
    /// test (typically `data_hashes`) so cross-test comparisons are
    /// crisp. The single-element `data_hashes` mirrors the
    /// single-stream profile case; multi-stream tests append more.
    fn fixture_inputs(data_hash: &str) -> ProfileInputs {
        ProfileInputs {
            model_path: "model.camdl".into(),
            stem: Some("model".into()),
            model_hash: "deadbeef".repeat(8),
            base_params_hash: "cafef00d".repeat(8),
            data_hashes: vec![("cases".into(), data_hash.to_string())],
            focal_grid: vec![GridAxis {
                param: "R0".into(),
                values: vec![1.5, 2.0, 2.5],
            }],
            fixed: vec![],
            obs_family: "cases".into(),
            if2_config: ProfileIf2Config {
                n_particles: 100, n_iterations: 50, cooling: 0.5, dt: 1.0, n_starts: 4,
            },
            // gh#89: stable defaults so fixture-based tests for the
            // *other* fields don't accidentally vary on these.
            algorithm: "if2".into(),
            pmmh_steps: 500,
            pmmh_particles: 500,
            pmmh_rho: Some(0.99),
            starts_from_lineage: None,
            fit_toml_hash: None,
            resolved_priors: vec![],
            suppressed_warnings: vec![],
            seed: 1,
            parameters_provenance: Default::default(),
            init_provenance: None,
        }
    }

    #[test]
    fn inner_hash_same_data_same_hash() {
        // Sanity: two identical input sets produce identical hashes.
        let h_data = crate::hashing::sha256_hex(b"time\tvalue\n1\t5\n2\t7\n");
        let a = fixture_inputs(&h_data);
        let b = fixture_inputs(&h_data);
        assert_eq!(a.inner_hash().full(), b.inner_hash().full());
    }

    #[test]
    fn inner_hash_different_data_different_hash() {
        // gh#39 core fix: changing the bytes the user supplied as
        // `--data` MUST invalidate the cache. Two TSVs with the same
        // shape but different observation values must hash differently.
        let h_a = crate::hashing::sha256_hex(b"time\tvalue\n1\t5\n2\t7\n");
        let h_b = crate::hashing::sha256_hex(b"time\tvalue\n1\t8\n2\t9\n");
        assert_ne!(h_a, h_b, "sanity: distinct bytes must hash differently");
        let a = fixture_inputs(&h_a);
        let b = fixture_inputs(&h_b);
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "editing --data file bytes must invalidate the profile CAS \
             key (gh#39); otherwise the cache silently returns stale \
             logliks against the old observations");
    }

    #[test]
    fn inner_hash_data_via_different_paths_same_hash() {
        // Locks the "content not path" invariant: two users with
        // identical TSVs at different filesystem paths must share a
        // cache entry. Implemented by hashing only the bytes of
        // `--data` at construction time, never the path string.
        let tmp = tempfile::tempdir().unwrap();
        let body = b"time\tvalue\n1\t5\n2\t7\n";
        let path_a = tmp.path().join("dir_a/cases.tsv");
        let path_b = tmp.path().join("dir_b/cases.tsv");
        std::fs::create_dir_all(path_a.parent().unwrap()).unwrap();
        std::fs::create_dir_all(path_b.parent().unwrap()).unwrap();
        std::fs::write(&path_a, body).unwrap();
        std::fs::write(&path_b, body).unwrap();

        // Hash exactly the way `cmd_profile` does at launch.
        let h_a = crate::hashing::sha256_hex(&std::fs::read(&path_a).unwrap());
        let h_b = crate::hashing::sha256_hex(&std::fs::read(&path_b).unwrap());
        assert_eq!(h_a, h_b,
            "same TSV bytes at different paths must hash identically");

        let a = fixture_inputs(&h_a);
        let b = fixture_inputs(&h_b);
        assert_eq!(a.inner_hash().full(), b.inner_hash().full(),
            "two profiles with identical content but different --data \
             paths must share a cache entry (path is not part of the \
             hash, only bytes are)");
    }

    // ── gh#89: pmmh-* knobs AND --algorithm must be in the cache key ──
    //
    // Symptom that triggered this: a user re-runs `camdl profile` with
    // `--pmmh-steps 5000` after a first run at `--pmmh-steps 1000`
    // (everything else identical, including seed). The CAS layer sees
    // the existing cached results and "resumes" them — silently
    // returning the lower-steps results instead of computing fresh.
    //
    // Project-stakes consideration: silently returning stale samples
    // when the user explicitly bumped the MCMC budget is a
    // public-health-grade footgun. The user thinks they got tighter
    // posterior coverage and got the old run instead. These tests
    // pin the invariants.

    #[test]
    fn inner_hash_changes_when_pmmh_steps_changes() {
        // The gh#89 minimal repro. Two ProfileInputs differing ONLY in
        // pmmh_steps must hash to different CAS keys.
        let h_data = crate::hashing::sha256_hex(b"time\tvalue\n1\t5\n2\t7\n");
        let a = fixture_inputs(&h_data);
        let b = ProfileInputs { pmmh_steps: a.pmmh_steps * 5, ..a.clone() };
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "bumping --pmmh-steps MUST invalidate the cache (gh#89); \
             otherwise the user silently gets the lower-steps results");
    }

    #[test]
    fn inner_hash_changes_when_pmmh_particles_changes() {
        // Same shape as --pmmh-steps: this is a budget knob that
        // materially affects PF noise and thus MCMC mixing.
        let h_data = crate::hashing::sha256_hex(b"time\tvalue\n1\t5\n2\t7\n");
        let a = fixture_inputs(&h_data);
        let b = ProfileInputs { pmmh_particles: a.pmmh_particles * 2, ..a.clone() };
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "bumping --pmmh-particles MUST invalidate the cache");
    }

    #[test]
    fn inner_hash_changes_when_pmmh_rho_changes() {
        // Crank-Nicolson correlation; affects MCMC mixing dynamics.
        // Off (None) vs on (Some(0.99)) is a content change.
        let h_data = crate::hashing::sha256_hex(b"time\tvalue\n1\t5\n2\t7\n");
        let a = ProfileInputs { pmmh_rho: None,        ..fixture_inputs(&h_data) };
        let b = ProfileInputs { pmmh_rho: Some(0.99),  ..fixture_inputs(&h_data) };
        let c = ProfileInputs { pmmh_rho: Some(0.50),  ..fixture_inputs(&h_data) };
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "toggling --pmmh-rho on/off MUST invalidate the cache");
        assert_ne!(b.inner_hash().full(), c.inner_hash().full(),
            "changing --pmmh-rho value MUST invalidate the cache");
    }

    #[test]
    fn inner_hash_changes_when_algorithm_changes() {
        // Same-class bug, surfaced while writing gh#89: --algorithm
        // selects between IF2, PMMH, NL-Sbplx, etc. A user who runs
        // `--algorithm if2` then re-runs `--algorithm pmmh` with the
        // same particle/iter counts would silently hit the if2 cache
        // (or the pmmh cache, depending on which ran first). The
        // resolved algorithm string must be in the canonical key.
        let h_data = crate::hashing::sha256_hex(b"time\tvalue\n1\t5\n2\t7\n");
        let a = ProfileInputs { algorithm: "if2".into(),       ..fixture_inputs(&h_data) };
        let b = ProfileInputs { algorithm: "pmmh".into(),      ..fixture_inputs(&h_data) };
        let c = ProfileInputs { algorithm: "nl-sbplx".into(),  ..fixture_inputs(&h_data) };
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "switching --algorithm if2 → pmmh MUST invalidate the cache");
        assert_ne!(b.inner_hash().full(), c.inner_hash().full(),
            "switching --algorithm pmmh → nl-sbplx MUST invalidate the cache");
    }

    #[test]
    fn inner_hash_data_field_is_load_bearing() {
        // Cross-check against the canonical-key implementation: with
        // every other field fixed, varying only `data_hashes[0]` must
        // move the inner_hash. This catches a future refactor that
        // accidentally drops the `("data", ...)` entry from the
        // canonical-keys vector.
        let a = fixture_inputs(&"a".repeat(64));
        let b = fixture_inputs(&"b".repeat(64));
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "data_hashes must be wired into inner_hash's canonical keys");
    }

    // ── gh#90 multi-stream cache key invariants ──────────────────────
    //
    // The data_hashes Vec replaces the gh#39 single data_hash to
    // carry every bound stream's (name, content_hash) pair into the
    // cache key. These tests pin the invariants that the dispatch
    // requires.

    #[test]
    fn profile_inner_hash_changes_when_stream_set_changes() {
        // Adding a stream to the bound set is a real content change
        // — the profile that scores `cases + deaths` jointly is NOT
        // the same as the one that scores only `cases`.
        let h_cases  = "a".repeat(64);
        let h_deaths = "b".repeat(64);
        let one = fixture_inputs(&h_cases);
        let two = ProfileInputs {
            data_hashes: vec![
                ("cases".into(),  h_cases.clone()),
                ("deaths".into(), h_deaths.clone()),
            ],
            ..fixture_inputs(&h_cases)
        };
        assert_ne!(one.inner_hash().full(), two.inner_hash().full(),
            "adding a bound stream MUST invalidate the profile CAS \
             key (gh#90); otherwise the cache silently returns the \
             1-stream loglik for what the user thinks is a 2-stream \
             joint score");
    }

    #[test]
    fn profile_inner_hash_changes_when_stream_data_content_changes() {
        // Per-stream gh#39 invariant: editing the bytes of any bound
        // stream's data file must invalidate the cache, including
        // streams beyond the first.
        let h_cases  = "a".repeat(64);
        let h_deaths_v1 = "b".repeat(64);
        let h_deaths_v2 = "c".repeat(64);
        let a = ProfileInputs {
            data_hashes: vec![
                ("cases".into(),  h_cases.clone()),
                ("deaths".into(), h_deaths_v1.clone()),
            ],
            ..fixture_inputs(&h_cases)
        };
        let b = ProfileInputs {
            data_hashes: vec![
                ("cases".into(),  h_cases.clone()),
                ("deaths".into(), h_deaths_v2.clone()),
            ],
            ..fixture_inputs(&h_cases)
        };
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "editing the bytes of any bound stream's --data file MUST \
             invalidate the cache (gh#39 generalised to multi-stream \
             per gh#90)");
    }

    #[test]
    fn profile_inner_hash_invariant_under_stream_reordering() {
        // The CLI ordering of --data NAME=PATH flags is presentation,
        // not content — `--data cases=... --data deaths=...` and
        // `--data deaths=... --data cases=...` must hit the same
        // cache entry. The inner_hash sort-by-name normalises this.
        let h_cases  = "a".repeat(64);
        let h_deaths = "b".repeat(64);
        let a = ProfileInputs {
            data_hashes: vec![
                ("cases".into(),  h_cases.clone()),
                ("deaths".into(), h_deaths.clone()),
            ],
            ..fixture_inputs(&h_cases)
        };
        let b = ProfileInputs {
            data_hashes: vec![
                ("deaths".into(), h_deaths.clone()),
                ("cases".into(),  h_cases.clone()),
            ],
            ..fixture_inputs(&h_cases)
        };
        assert_eq!(a.inner_hash().full(), b.inner_hash().full(),
            "reordering --data NAME=PATH flags must NOT change the \
             cache key — that's presentation, not content");
    }

    // ── seed_params_from_init_method: Phase 3 → Phase 2 bridge ────────
    //
    // Bug 3 from the post-CLI-UX-rev-2 regression report: the unified
    // resolver fires `UnsetRequired` when a parameter has no DSL
    // default + no `[fixed]` entry, even when the user has
    // explicitly named a single-point init file (`--init from_params
    // --params <toml>` or `--init from_mle --mle <path>`).
    //
    // These tests pin the fix: the helper loads the file BEFORE the
    // resolver runs and seeds matching parameters' values. File wins
    // over DSL default (aggressive — user explicitly named the file).

    fn build_params(specs: &[(&str, Option<f64>)]) -> Vec<ir::parameter::Parameter> {
        // Use serde so we don't track every Parameter field as the IR
        // schema evolves. Each entry is a minimal {name, value} JSON
        // object; serde_json fills the rest from
        // `#[serde(default)]` on the Parameter struct.
        specs.iter().map(|(name, value)| {
            let body = match value {
                Some(v) => format!(r#"{{"name":"{}","value":{}}}"#, name, v),
                None    => format!(r#"{{"name":"{}","value":null}}"#, name),
            };
            serde_json::from_str::<ir::parameter::Parameter>(&body)
                .expect("Parameter fixture must parse")
        }).collect()
    }

    fn write_toml(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn seed_from_params_populates_unset_params() {
        // Bug 3 minimal repro: model has no DSL default for beta;
        // --init from_params --params start.toml supplies beta=0.5;
        // the helper seeds model.parameters[beta].value = 0.5 so the
        // resolver doesn't fire UnsetRequired.
        let tmp = tempfile::tempdir().unwrap();
        let toml_path = write_toml(tmp.path(), "start.toml",
            "beta = 0.5\ngamma = 0.1\n");
        let mut params = build_params(&[
            ("beta",  None),  // no DSL default
            ("gamma", None),
        ]);
        let init = crate::fit::init::InitMethod::FromParams { path: toml_path };
        seed_params_from_init_method(&mut params, &init).unwrap();
        let beta_val  = params.iter()
            .find(|p| p.name == "beta").unwrap().value;
        let gamma_val = params.iter()
            .find(|p| p.name == "gamma").unwrap().value;
        assert_eq!(beta_val,  Some(0.5));
        assert_eq!(gamma_val, Some(0.1));
    }

    #[test]
    fn seed_from_params_overrides_dsl_default_file_wins() {
        // Aggressive seeding: if the model has `beta = 0.3` and the
        // user passes --init from_params --params with `beta = 0.5`,
        // the file wins. User explicitly named the file as the
        // authoritative source.
        let tmp = tempfile::tempdir().unwrap();
        let toml_path = write_toml(tmp.path(), "start.toml", "beta = 0.5\n");
        let mut params = build_params(&[
            ("beta", Some(0.3)),  // DSL default that the file should override
        ]);
        let init = crate::fit::init::InitMethod::FromParams { path: toml_path };
        seed_params_from_init_method(&mut params, &init).unwrap();
        let beta_val = params.iter()
            .find(|p| p.name == "beta").unwrap().value;
        assert_eq!(beta_val, Some(0.5),
            "file value must win over DSL default — user named the file");
    }

    #[test]
    fn seed_from_mle_resolves_fitdir_to_mle_toml() {
        // --init from_mle --mle <fit-dir>: helper auto-resolves to
        // <dir>/mle.toml. Verifies the [mle] section is parsed
        // (mle.toml shape, distinct from flat params.toml).
        let tmp = tempfile::tempdir().unwrap();
        let mle_dir = tmp.path().join("fit_results");
        std::fs::create_dir_all(&mle_dir).unwrap();
        write_toml(&mle_dir, "mle.toml",
            "final_loglik = -311.13\n\n[focal]\nR0 = 25\n\n[mle]\nbeta = 0.5\n");
        let mut params = build_params(&[
            ("beta", None),
        ]);
        let source = crate::fit::init::MleSource::FitDir(mle_dir);
        let init = crate::fit::init::InitMethod::FromMle { source };
        seed_params_from_init_method(&mut params, &init).unwrap();
        let beta_val = params.iter()
            .find(|p| p.name == "beta").unwrap().value;
        assert_eq!(beta_val, Some(0.5));
    }

    #[test]
    fn seed_noop_for_per_chain_varying_init_modes() {
        // from_prior / from_posterior have per-chain-varying values
        // — there's no single value to seed into model_pre. The
        // helper must be a no-op for these modes (users are expected
        // to supply a separate base-value source).
        let mut params = build_params(&[
            ("beta", Some(0.3)),
        ]);
        let original_beta = params.iter()
            .find(|p| p.name == "beta").unwrap().value;
        let init_prior = crate::fit::init::InitMethod::FromPrior;
        seed_params_from_init_method(&mut params, &init_prior).unwrap();
        assert_eq!(params.iter()
            .find(|p| p.name == "beta").unwrap().value, original_beta,
            "from_prior must NOT seed model_pre — values are per-chain");

        let init_lhs = crate::fit::init::InitMethod::Lhs;
        seed_params_from_init_method(&mut params, &init_lhs).unwrap();
        assert_eq!(params.iter()
            .find(|p| p.name == "beta").unwrap().value, original_beta,
            "Lhs must NOT seed model_pre");
    }

    // ── gh#109: log_posterior surfaces in mle.toml + profile TSV ──────
    //
    // PMMHResult exposes map_log_posterior alongside map_loglik; the
    // pre-gh#109 profile pipeline dropped the posterior at the per-cell
    // closure. These tests pin the new plumbing through render_mle_toml,
    // parse_mle_toml, and the rollup TSV header.

    fn estimated_param(name: &str, index: usize) -> sim::inference::if2::EstimatedParam {
        sim::inference::if2::EstimatedParam {
            name: name.into(), index,
            initial: 0.5, rw_sd: 0.1,
            transform: sim::inference::types::Transform::None,
            lower: 0.0, upper: 1.0,
            ivp: false, rw_sd_auto: false,
        }
    }

    #[test]
    fn render_mle_toml_emits_final_log_posterior_when_finite() {
        // PMMH cell: both fields written, transparent to a TOML round-trip.
        let if2 = vec![estimated_param("beta", 0), estimated_param("gamma", 1)];
        let diag = crate::profile_diagnostics::PerStartDiagnostics {
            algo: Some(crate::profile_diagnostics::DiagAlgo::Pmmh),
            completed: true,
            acc_rate: Some(0.3),
            iterations_used: None,
            cooling_final: None,
            loglik_trace:        vec![-12.0, -11.5, -11.0],
            log_posterior_trace: vec![-10.5, -10.0, -9.5],
        };
        let body = render_mle_toml(
            &if2, &[25.0], &["R0"], &[0.42, 0.10],
            -11.0, -9.5, &diag,
        );
        assert!(body.contains("final_loglik = -11"),
            "final_loglik must appear: {}", body);
        assert!(body.contains("final_log_posterior = -9.5"),
            "final_log_posterior must appear with the supplied value: {}", body);
        assert!(body.contains("log_posterior_trace ="),
            "[diagnostics] log_posterior_trace must appear: {}", body);
    }

    #[test]
    fn render_mle_toml_omits_final_log_posterior_when_nan() {
        // IF2 / NLopt cell: no posterior concept. Field should be
        // skipped entirely (mle.toml stays tight; readers fall back
        // to NaN at parse time).
        let if2 = vec![estimated_param("beta", 0)];
        let diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        let body = render_mle_toml(
            &if2, &[25.0], &["R0"], &[0.42],
            -11.0, f64::NAN, &diag,
        );
        assert!(body.contains("final_loglik = -11"));
        assert!(!body.contains("final_log_posterior"),
            "NaN log_posterior must be omitted, not emitted as 'nan': {}", body);
    }

    #[test]
    fn parse_mle_toml_reads_final_log_posterior() {
        // Round-trip: render + parse must preserve the field.
        let if2 = vec![estimated_param("beta", 0)];
        let diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        let body = render_mle_toml(
            &if2, &[25.0], &["R0"], &[0.42],
            -11.0, -9.5, &diag,
        );
        let focal_names = vec!["R0".to_string()];
        let parsed = parse_mle_toml(&body, &if2, &focal_names)
            .expect("synthetic mle.toml must parse");
        assert!((parsed.final_loglik - (-11.0)).abs() < 1e-9);
        assert!((parsed.final_log_posterior - (-9.5)).abs() < 1e-9);
    }

    #[test]
    fn parse_mle_toml_legacy_without_log_posterior_returns_nan() {
        // Cached CAS dirs from before gh#109: mle.toml has no
        // final_log_posterior field. Parser must accept and return
        // NaN so the rollup degrades gracefully (NaN column in TSV).
        let body = "final_loglik = -11.0\n\n\
                    [focal]\nR0 = 25\n\n\
                    [mle]\nbeta = 0.42\n";
        let if2 = vec![estimated_param("beta", 0)];
        let focal_names = vec!["R0".to_string()];
        let parsed = parse_mle_toml(body, &if2, &focal_names)
            .expect("legacy mle.toml must parse");
        assert!((parsed.final_loglik - (-11.0)).abs() < 1e-9);
        assert!(parsed.final_log_posterior.is_nan(),
            "legacy mle.toml without final_log_posterior must parse as NaN; \
             got {}", parsed.final_log_posterior);
    }
}

