//! Unified run-metadata ADT for the `output/` tree.
//!
//! One `Run` type with a `kind: RunKind` discriminator covers every
//! result camdl produces — simulate runs, top-level fits, and
//! per-stage fits — under one schema. Replaced the parallel
//! `cas::RunMeta` and `fit::provenance::StageProvenance` structs that
//! had ~80 % field overlap with drifting names (version vs
//! camdl_version, created_at vs timestamp, etc.); both are now gone.
//!
//! See `docs/dev/proposals/2026-04-19-unified-output-tree.md` for the
//! original design.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadata written to `run.json` at the top of every content-hashed
/// run directory. Shared fields live at the top level; kind-specific
/// fields are inside `kind`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Content hash for this run, full 64-char hex. Scope depends on
    /// `kind`:
    ///   - `Simulate`: hash of (sim_hash, scen_hash, seed).
    ///   - `Fit`: seed-independent content hash of
    ///     (fit.toml, model IR, data files).
    ///   - `FitStage`: stage-scope config hash from `fit_stage_hash`
    ///     (includes stage algorithm + seed).
    /// The 8-char prefix appears in the filesystem path.
    pub hash: String,
    /// camdl version at write time (e.g. "0.1.0+abc1234").
    pub version: String,
    /// ISO 8601 UTC timestamp at completion.
    pub created_at: String,
    /// Original argv that produced this run — `camdl show <hash>`
    /// prints it back for reproducibility.
    pub argv: Vec<String>,
    /// Run lifecycle state. `Running` is set at the first write
    /// (run.json appears as soon as we know the directory layout, so
    /// cancellations / crashes leave a discoverable trace);
    /// `Completed { wall_time_seconds }` replaces it at end-of-run.
    /// Replaces the prior sentinel `wall_time_seconds == 0.0`-means-
    /// running pattern.
    pub status: RunStatus,
    /// User-supplied display label. Optional — validated against
    /// `^[a-zA-Z0-9 ,._-]{1,64}$` after trim. Set at run-time via
    /// `--label` or post-hoc via `camdl label <hash> "<text>"`.
    /// Surfaced in `camdl list`, `camdl fit table`, `camdl show` to
    /// help disambiguate iterations of the same operation that share
    /// the same stem (e.g. multiple `fit_he2010-XXXXXXXX` directories
    /// with different bounds / priors). Applies uniformly to all
    /// `RunKind` variants — sims, fits, profiles, replicate-sets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Kind-specific payload.
    pub kind: RunKind,
}

/// Run lifecycle state. Two states:
///
/// - `Running` — the run.json was written but the run is still
///   executing (or crashed before patching). Wall time isn't known.
/// - `Completed { wall_time_seconds }` — the runner finished and
///   patched run.json with the elapsed time.
///
/// Replaces a prior sentinel pattern (`wall_time_seconds == 0.0` ⇒
/// running). Wire format is the snake-case enum:
///
///     "status": "running"
///     "status": { "completed": { "wall_time_seconds": 42.5 } }
///
/// Test fixtures and renderers should not assume `Completed` —
/// `cmd_label`, `camdl show`, etc. handle both states.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed { wall_time_seconds: f64 },
}

impl RunStatus {
    /// `true` iff the run is still executing (or crashed before
    /// finalizing).
    pub fn is_running(&self) -> bool {
        matches!(self, RunStatus::Running)
    }

    /// Wall time in seconds when complete; `None` while running.
    pub fn wall_time_seconds(&self) -> Option<f64> {
        match self {
            RunStatus::Completed { wall_time_seconds } => Some(*wall_time_seconds),
            RunStatus::Running => None,
        }
    }
}

/// Tagged union over the three result shapes. `serde(tag = "kind")`
/// emits a `"kind": "simulate"` (etc.) field in the JSON, so
/// `camdl list` can discriminate without needing to know the directory
/// layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RunKind {
    /// One simulate invocation. The directory contains `traj.tsv` and
    /// optional `obs/<obs_hash>-<obs_seed>/` subdirectories.
    Simulate(SimulateMeta),
    /// A complete fit (potentially multi-stage). The directory
    /// contains per-stage subdirectories, each with its own
    /// stage-level `Run` whose kind is `FitStage`.
    Fit(FitMeta),
    /// One stage of a fit. The directory is a child of a `Fit` run,
    /// at `<fit_dir>/real/fit_<seed>/<stage>/` or
    /// `<fit_dir>/synthetic/ds_NN/fit_<seed>/<stage>/`.
    FitStage(FitStageMeta),
    /// A profile-likelihood scan: Cartesian product over N focal
    /// parameter axes × `n_starts` independent IF2 mini-fits per
    /// grid point. The directory contains `profile.tsv` (derived
    /// rollup) and `points/{idx:05d}/start_{k}/` subtrees, where
    /// each `start_{k}/` is itself a `FitStage` run. See
    /// docs/dev/proposals/2026-04-24-profile-cas-integration.md.
    Profile(ProfileMeta),
    /// A *group* of single-realization runs that share an inner
    /// content (everything except a varying replicate dimension)
    /// and differ only on one input — typically `seed` for
    /// stochastic-method sensitivity, `dataset_idx` for
    /// synthetic-data fits. The directory contains an aggregate
    /// `summary.tsv` and a `replicates/<key>/` subdir per child;
    /// each child has its own `run.json` of the underlying kind.
    /// See docs/dev/proposals/2026-04-28-cas-typed-runs-and-profile-stages.md.
    ReplicateSet(crate::cas::typed::ReplicateSetMeta),
    /// A likelihood-landscape diagnostic: N Latin-hypercube points
    /// across declared parameter bounds, evaluated via PF or single
    /// simulation, written to `landscape.tsv`. NOT a fitting routine
    /// — produces no MLE artifact. See
    /// docs/dev/proposals/2026-05-03-survey-subcommand.md.
    Survey(SurveyMeta),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateMeta {
    /// Model file path or name (display only — not a hash input).
    pub model: String,
    /// Full model IR hash (64 hex chars).
    pub model_hash: String,
    /// Named scenario or "baseline".
    pub scenario: String,
    /// Simulation config hash: model + base params + backend + dt + version.
    pub sim_hash: String,
    /// Scenario delta hash: enable/disable/overrides.
    pub scen_hash: String,
    pub seed: u64,
    pub backend: crate::args::types::Backend,
    pub dt: f64,
    /// Sweep-point param values (empty for single-run `--cas`).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub sweep_point: HashMap<String, f64>,
    /// Full Run.hash of the fit whose `mle_params.toml` was passed to
    /// `camdl simulate --params`, when applicable. Populates a
    /// sim → fit provenance link for `camdl list` / `camdl show` to
    /// surface. See `docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_fit_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitMeta {
    /// Model file path (display only — not a hash input).
    pub model: String,
    /// Structural model IR hash.
    pub model_hash: String,
    /// Path to the fit.toml that produced this fit.
    pub fit_toml_path: String,
    /// Hash of the fit.toml bytes — canonical-form hash from
    /// `FitConfigV2::fit_content_hash`.
    pub fit_toml_hash: String,
    /// Per-stream data file hashes.
    pub data_hashes: HashMap<String, String>,
    /// Names of parameters declared in `[estimate]`.
    pub estimated: Vec<String>,
    /// Resolved fixed params (name → numeric value).
    pub fixed: HashMap<String, f64>,
    /// Stage names declared in fit.toml, in execution order.
    pub stages_declared: Vec<String>,
    /// IC-free inference flag (see 2026-04-18 proposal).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ic_free: bool,
}

/// Inference method tag — discriminator-only enum. Used in
/// `FitStageMeta.method` and surfaced by `Stage::method_kind()`. Wire
/// format matches the legacy strings (`"if2"` / `"pgas"` / `"pmmh"`
/// / `"pfilter"`) so existing run.json files round-trip unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MethodKind {
    If2,
    Pgas,
    Pmmh,
    Pfilter,
}

impl MethodKind {
    /// String form matching the legacy `FitStageMeta.method` value
    /// and the `Stage::FOO` serde tag.
    pub fn as_str(self) -> &'static str {
        match self {
            MethodKind::If2     => "if2",
            MethodKind::Pgas    => "pgas",
            MethodKind::Pmmh    => "pmmh",
            MethodKind::Pfilter => "pfilter",
        }
    }
}

impl std::fmt::Display for MethodKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitStageMeta {
    /// Hash of the parent fit (matches the `hash` on the enclosing
    /// `Run` of kind `Fit`). Enables walking from a stage back to its
    /// parent without relying on directory-layout inference.
    pub fit_hash: String,
    /// Stage name within the fit (e.g. "scout", "refine").
    pub stage: String,
    /// Stage method discriminator. Wire format: `"if2" | "pgas" |
    /// "pmmh" | "pfilter"`.
    pub method: MethodKind,
    // NB: the stage's own content hash lives in the enclosing
    // `Run.hash` field (a FitStage run hashes exactly its stage-scope
    // inputs). Previously FitStageMeta carried a duplicate
    // `stage_hash: String` — removed to collapse the two-source-of-
    // truth smell.
    pub seed: u64,
    pub n_chains: usize,
    /// Stage-specific algorithm settings (chains, particles, cooling,
    /// etc.). A `serde_json::Value` keeps the shape open — each method
    /// (if2/pgas/pmmh/pfilter) has a different parameter set, and the
    /// human-readable record doesn't need a typed schema.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub algorithm: serde_json::Value,
    /// Best loglik across chains; `None` if the stage didn't compute
    /// one (e.g. a pure-diagnostic pass).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_loglik: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_chain: Option<usize>,
    /// Reference to a parent stage this one started from, if any
    /// (e.g. refine → scout). Absent when the stage has no predecessor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub starts_from: Option<StartsFromRef>,
    /// Path to the upstream fit dir this fit was derived from
    /// (`camdl fit derive` workflows). Free-form string — the consumer
    /// treats this as a display hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<String>,
    /// Parent profile hash, when this FitStage is a grid-point × start
    /// child of a `RunKind::Profile`. Absent for standalone fit stages.
    /// Optional to preserve round-trip compatibility with existing
    /// fit-stage run.json files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_profile_hash: Option<String>,
    /// Grid-point index within the parent profile (flat index over
    /// the Cartesian product of focal axes). Set iff
    /// `parent_profile_hash` is set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_point_idx: Option<usize>,
    /// Start index within this grid point (0..n_starts).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_start_idx: Option<usize>,
}

/// Metadata for a `RunKind::Profile` run. The shape mirrors pomp's
/// and pfilter's convention of fanning out mini-fits over a grid;
/// every child start is a `FitStage` carrying its own seed and MLE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileMeta {
    /// Model file path (display only).
    pub model: String,
    /// Full model IR hash.
    pub model_hash: String,
    /// Ordered focal params. Order determines column order in the
    /// rollup TSV and is part of the profile-level hash.
    pub focal_params: Vec<String>,
    /// One axis per focal param, each with an explicit value list
    /// (mirroring the `--sweep NAME=V1,V2,...` CLI surface).
    pub grid: Vec<GridAxis>,
    /// Independent IF2 starts per grid point.
    pub n_starts: usize,
    /// Hash of the IF2 stage config (iterations, particles, cooling, dt).
    pub if2_config_hash: String,
    /// Hash of the base parameter vector (before focal-param pinning).
    pub base_params_hash: String,
    /// Seed base. Per-start seeds derive as a function of this +
    /// point_idx + start_idx.
    pub seed_base: u64,
    /// Total (grid_size × n_starts). Display only.
    pub total_jobs: usize,
}

/// One axis of a profile grid. `values` is the explicit list the
/// user supplied via `--sweep NAME=V1,V2,...`; the CLI parser
/// already splits on commas and converts to f64.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridAxis {
    pub param: String,
    pub values: Vec<f64>,
}

/// How `camdl survey` evaluates the marginal log-likelihood at each
/// LHS point. The default is `Pfilter` (handles process noise via a
/// PMMH-style MC estimator); `Simulate` is an opt-in fast path for
/// known-deterministic models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
#[clap(rename_all = "lowercase")]
pub enum SurveyEvalMethod {
    /// Bootstrap particle filter, K replicates → logmeanexp combiner.
    /// Estimates p(y|θ) under the chain-binomial process; the safe
    /// default for inference-grade likelihood evaluation. Doucet et
    /// al. 2015 (Biometrika) gives the rule for trustworthy ranks:
    /// per-point loglik SE ≤ ~1.7 nats.
    Pfilter,
    /// Single deterministic simulation per point. 1-sample MC estimator
    /// of the same quantity; cheap (~10× faster than Pfilter at
    /// modest particles/replicates) but biased toward "lucky outliers"
    /// when process noise is non-trivial. Andrieu & Roberts 2009 frame
    /// the failure mode.
    Simulate,
}

impl SurveyEvalMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            SurveyEvalMethod::Pfilter  => "pfilter",
            SurveyEvalMethod::Simulate => "simulate",
        }
    }
}

impl std::fmt::Display for SurveyEvalMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata for a `RunKind::Survey` run. Mirrors `ProfileMeta`'s
/// shape: model + content hashes + the canonical-hashed inputs that
/// determine the LHS box (`bounds`), the evaluator config
/// (`eval_method`/`eval_particles`/`eval_replicates`), the seed, and
/// any fixed / scenario context used to populate the rest of the
/// parameter vector at each LHS point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurveyMeta {
    /// Model file path (display only — not a hash input).
    pub model: String,
    /// Full SHA-256 of the IR JSON.
    pub model_hash: String,
    /// Per-stream data file content hashes. Same shape as `FitMeta` —
    /// content-only, so editing the TSV invalidates the cache (gh#39).
    pub data_hashes: HashMap<String, String>,
    /// LHS box: parameter name → (lo, hi). Resolved per the proposal's
    /// "fit.toml > model" priority. Canonical-hashed by sorting names.
    pub bounds: HashMap<String, (f64, f64)>,
    /// Number of LHS points evaluated.
    pub n_points: usize,
    /// PF or single-simulate.
    pub eval_method: SurveyEvalMethod,
    /// PF particle count (per replicate). Diagnostic when
    /// `eval_method = Simulate`.
    pub eval_particles: usize,
    /// PF replicates per LHS point (logmeanexp combiner). Diagnostic
    /// when `eval_method = Simulate` (always 1 in that case).
    pub eval_replicates: usize,
    /// LHS / PF base seed.
    pub seed: u64,
    /// Resolved fixed parameters (name → value). Canonical-hashed.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub fixed: HashMap<String, f64>,
    /// Named scenario applied before the survey. `None` for the
    /// baseline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scenario: Option<String>,
    /// Names of estimated parameters in declaration order — drives
    /// the column order of `landscape.tsv`. Bounds key + this slice
    /// together fix the LHS layout deterministically.
    pub estimated: Vec<String>,
}

/// Stable reference to a parent stage. Uses the stage *name* plus its
/// content hash, not a filesystem path — so the reference survives any
/// tree reorganisation. The path is a cache-lookup concern the caller
/// reconstructs via `run_paths::fit_stage_dir`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartsFromRef {
    pub stage: String,
    /// Content hash of the upstream stage, if its `run.json` could be
    /// read at write time. `None` when the upstream directory had no
    /// readable run.json — absent rather than empty so the provenance
    /// chain doesn't silently corrupt into "has a parent, but we don't
    /// know its hash."
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_hash: Option<String>,
}

/// Unified cache status: result of comparing an expected content hash
/// against the `run.json` in a directory. Applies to both simulate
/// and fit-stage runs.
#[derive(Debug, Clone)]
pub enum CacheStatus {
    /// Run directory exists and its stored hash matches the expected
    /// hash; caller can read results from `run_dir`.
    Hit,
    /// Directory exists but the stored hash differs from the expected
    /// one. Typically triggers a re-run with a warning.
    Stale { stored: String, current: String },
    /// No `run.json` at the expected location; cache miss.
    Miss,
}

impl Run {
    /// Write `run.json` inside `dir`. Creates parent directories.
    /// Write `run.json` atomically: write to `run.json.tmp`, then
    /// rename. POSIX rename within the same filesystem is atomic —
    /// readers either see the complete new file or nothing at all,
    /// never a half-written / truncated JSON. The invariant this
    /// preserves: if `run.json` exists, every sibling artifact was
    /// written before this rename succeeded, so `run.json`'s mere
    /// presence is an authoritative "stage completed" marker.
    ///
    /// Hardening proposal ship-now #3 — replaces a previous plain
    /// `fs::write` that left a crash window in which sibling
    /// artifacts (mle_params.toml, fit_state.toml) were already on
    /// disk but run.json hadn't been written yet, making partial
    /// stages look complete to any reader of the sibling files.
    pub fn write(&self, dir: &std::path::Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let json = serde_json::to_string_pretty(self)
            .map_err(std::io::Error::other)?;
        let tmp = dir.join("run.json.tmp");
        let final_path = dir.join("run.json");
        // Write tmp + rename. If anything fails mid-write, tmp may be
        // left behind — harmless because readers don't look at it,
        // and the next successful write overwrites it.
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, &final_path)
    }

    /// Read `run.json` from `dir`. Returns a serde error kinds in
    /// `ErrorKind::InvalidData` if the file exists but doesn't match
    /// the schema — a sign the directory was written by an older
    /// camdl version or a different tool.
    pub fn read(dir: &std::path::Path) -> std::io::Result<Run> {
        let contents = std::fs::read_to_string(dir.join("run.json"))?;
        serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Check whether `dir` has a `run.json` whose `hash` matches
    /// `expected_hash`. Replaces the sim-side `has_cached_traj` +
    /// `RunMeta.sim_hash` pair and the fit-side provenance.json check
    /// with one uniform code path.
    pub fn check_cache(dir: &std::path::Path, expected_hash: &str) -> CacheStatus {
        match Self::read(dir) {
            Ok(run) if run.hash == expected_hash =>
                CacheStatus::Hit,
            Ok(run) => CacheStatus::Stale {
                stored: run.hash,
                current: expected_hash.to_string(),
            },
            Err(_) => CacheStatus::Miss,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_simulate_run() -> Run {
        Run {
            hash: "abc12345def6789000000000000000000000000000000000000000000000000".into(),
            version: "0.1.0+test".into(),
            created_at: "2026-04-19T12:00:00Z".into(),
            argv: vec!["camdl".into(), "simulate".into(), "sir.camdl".into()],
            status: RunStatus::Completed { wall_time_seconds: 1.23 },
            label: None,
            kind: RunKind::Simulate(SimulateMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                scenario: "baseline".into(),
                sim_hash: "abc12345".into(),
                scen_hash: "def67890".into(),
                seed: 42,
                backend: crate::args::types::Backend::Gillespie,
                dt: 1.0,
                sweep_point: HashMap::new(),
                from_fit_hash: None,
            }),
        }
    }

    fn sample_fit_run() -> Run {
        Run {
            hash: "deadbeef".repeat(8),
            version: "0.1.0+test".into(),
            created_at: "2026-04-19T12:00:00Z".into(),
            argv: vec!["camdl".into(), "fit".into(), "run".into(), "fit.toml".into()],
            status: RunStatus::Completed { wall_time_seconds: 42.0 },
            label: None,
            kind: RunKind::Fit(FitMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                fit_toml_path: "fit.toml".into(),
                fit_toml_hash: "cafebabe".into(),
                data_hashes: {
                    let mut m = HashMap::new();
                    m.insert("cases".into(), "d4ta".repeat(2));
                    m
                },
                estimated: vec!["beta".into(), "gamma".into()],
                fixed: {
                    let mut m = HashMap::new();
                    m.insert("N0".into(), 1000.0);
                    m
                },
                stages_declared: vec!["scout".into(), "refine".into()],
                ic_free: false,
            }),
        }
    }

    fn sample_fit_stage_run() -> Run {
        Run {
            hash: "ae123456".repeat(8),
            version: "0.1.0+test".into(),
            created_at: "2026-04-19T12:00:00Z".into(),
            argv: vec!["camdl".into(), "fit".into(), "run".into(),
                       "fit.toml".into(), "--stage".into(), "refine".into()],
            status: RunStatus::Completed { wall_time_seconds: 10.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "deadbeef".repeat(8),
                stage: "refine".into(),
                method: MethodKind::If2,
                seed: 42,
                n_chains: 4,
                algorithm: serde_json::Value::Null,
                best_loglik: Some(-56.7),
                best_chain: Some(1),
                starts_from: Some(StartsFromRef {
                    stage: "scout".into(),
                    stage_hash: Some("beef1234".repeat(8)),
                }),
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
            }),
        }
    }

    #[test]
    fn simulate_run_roundtrip() {
        let r = sample_simulate_run();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"simulate""#),
            "kind discriminator missing from JSON: {}", json);
        let parsed: Run = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.hash, r.hash);
        match parsed.kind {
            RunKind::Simulate(m) => assert_eq!(m.seed, 42),
            _ => panic!("expected Simulate"),
        }
    }

    #[test]
    fn fit_run_roundtrip() {
        let r = sample_fit_run();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"fit""#));
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::Fit(m) => {
                assert_eq!(m.estimated.len(), 2);
                assert_eq!(m.stages_declared, vec!["scout", "refine"]);
            }
            _ => panic!("expected Fit"),
        }
    }

    #[test]
    fn fit_stage_run_roundtrip() {
        let r = sample_fit_stage_run();
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"fit-stage""#));
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::FitStage(m) => {
                assert_eq!(m.stage, "refine");
                assert_eq!(m.best_loglik, Some(-56.7));
                assert!(m.starts_from.is_some());
            }
            _ => panic!("expected FitStage"),
        }
    }

    #[test]
    fn write_read_roundtrip() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_run_meta_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        let r = sample_fit_run();
        r.write(&tmp).unwrap();
        let read = Run::read(&tmp).unwrap();
        assert_eq!(read.hash, r.hash);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn check_cache_hit_stale_miss() {
        let tmp = std::env::temp_dir().join(format!(
            "camdl_cache_status_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        let r = sample_simulate_run();
        let stored_hash = r.hash.clone();
        r.write(&tmp).unwrap();

        // Miss before write.
        let empty = tmp.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(matches!(Run::check_cache(&empty, &stored_hash), CacheStatus::Miss));

        // Hit with matching hash.
        assert!(matches!(
            Run::check_cache(&tmp, &stored_hash),
            CacheStatus::Hit
        ));

        // Stale when hash differs.
        match Run::check_cache(&tmp, "different_hash") {
            CacheStatus::Stale { stored, current } => {
                assert_eq!(stored, stored_hash);
                assert_eq!(current, "different_hash");
            }
            other => panic!("expected Stale, got {:?}", other),
        }

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn fit_stage_back_pointer_matches_parent_fit() {
        // The FitStageMeta.fit_hash field is how a stage references
        // its parent fit. If the two ever drift, stages become un-
        // attributable — guard against that by constructing both with
        // the same hash string and round-tripping the stage through
        // JSON, asserting the field survives.
        let parent = sample_fit_run();
        let parent_hash = parent.hash.clone();
        let stage = Run {
            hash: "stage0000".repeat(8),
            version: "v".into(),
            created_at: "t".into(),
            argv: vec![],
            status: RunStatus::Completed { wall_time_seconds: 1.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: parent_hash.clone(),
                stage: "scout".into(),
                method: MethodKind::If2,
                seed: 1,
                n_chains: 4,
                algorithm: serde_json::Value::Null,
                best_loglik: None,
                best_chain: None,
                starts_from: None,
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
            }),
        };
        let json = serde_json::to_string(&stage).unwrap();
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::FitStage(m) => assert_eq!(m.fit_hash, parent_hash),
            _ => panic!("expected FitStage"),
        }
    }

    #[test]
    fn atomic_write_leaves_no_tmp_after_success() {
        // On a clean run, run.json.tmp should not remain. Regression
        // guard: if we ever forget the rename, tmp would be left behind
        // and this test catches it.
        let tmp = std::env::temp_dir().join(format!(
            "camdl_atomic_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        let r = sample_simulate_run();
        r.write(&tmp).unwrap();
        assert!(tmp.join("run.json").exists(), "final run.json must exist");
        assert!(!tmp.join("run.json.tmp").exists(),
            "run.json.tmp must not be left behind after successful write");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn atomic_write_mid_crash_leaves_no_visible_run_json() {
        // Simulate the crash window: write the .tmp but never rename.
        // A reader should NOT see run.json — the invariant is "if
        // run.json exists, the write completed."
        let tmp = std::env::temp_dir().join(format!(
            "camdl_crash_{}_{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .unwrap().as_nanos()));
        std::fs::create_dir_all(&tmp).unwrap();
        // Manually create a run.json.tmp to simulate a crashed write.
        std::fs::write(tmp.join("run.json.tmp"),
            r#"{"partial": "data", "oops": "crash"}"#).unwrap();
        // No run.json should exist.
        assert!(!tmp.join("run.json").exists(),
            "crashed write: run.json must not be visible");
        // A reader's check_cache should report Miss, not a malformed
        // run.
        match Run::check_cache(&tmp, "any-hash") {
            CacheStatus::Miss => {}
            other => panic!("expected Miss (no run.json), got {:?}", other),
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn modern_stage_hash_omits_the_field_when_none() {
        // New writes omit the field entirely (skip_serializing_if).
        // Regression guard against someone re-introducing empty-string
        // writes.
        let r = Run {
            hash: "x".repeat(64), version: "v".into(), created_at: "t".into(),
            argv: vec![], status: RunStatus::Running, label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "f".repeat(64),
                stage: "refine".into(), method: MethodKind::If2,
                seed: 1, n_chains: 1,
                algorithm: serde_json::Value::Null,
                best_loglik: None, best_chain: None,
                starts_from: Some(StartsFromRef {
                    stage: "scout".into(), stage_hash: None,
                }),
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        // The `stage_hash` key should NOT appear when it's None —
        // otherwise the schema is silently round-tripping an empty
        // value.
        assert!(!json.contains("\"stage_hash\""),
            "None stage_hash should be omitted entirely, got {}", json);
    }

    #[test]
    fn optional_fields_skip_when_empty() {
        // Regression guard: serde(skip_serializing_if) on best_loglik,
        // best_chain, starts_from, sweep_point, ic_free. An empty
        // field shouldn't bloat the JSON.
        let r = Run {
            hash: "x".repeat(64), version: "v".into(),
            created_at: "t".into(), argv: vec![], status: RunStatus::Running,
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "f".repeat(64),
                stage: "mle".into(), method: MethodKind::If2,
                seed: 1, n_chains: 1,
                algorithm: serde_json::Value::Null,
                best_loglik: None, best_chain: None, starts_from: None,
                derived_from: None,
                parent_profile_hash: None,
                profile_point_idx: None,
                profile_start_idx: None,
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("best_loglik"));
        assert!(!json.contains("best_chain"));
        assert!(!json.contains("starts_from"));
        assert!(!json.contains("parent_profile_hash"));
    }

    #[test]
    fn profile_run_roundtrip() {
        let r = Run {
            hash: "a".repeat(64),
            version: "v".into(),
            created_at: "2026-04-24T00:00:00Z".into(),
            argv: vec!["camdl".into(), "profile".into()],
            status: RunStatus::Completed { wall_time_seconds: 3600.0 },
            label: None,
            kind: RunKind::Profile(ProfileMeta {
                model: "he2010_london.camdl".into(),
                model_hash: "b".repeat(64),
                focal_params: vec!["R0".into(), "gamma".into()],
                grid: vec![
                    GridAxis { param: "R0".into(), values: vec![40.0, 50.0, 60.0] },
                    GridAxis { param: "gamma".into(), values: vec![0.1, 0.2] },
                ],
                n_starts: 3,
                if2_config_hash: "c".repeat(64),
                base_params_hash: "d".repeat(64),
                seed_base: 42,
                total_jobs: 18,
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"profile""#),
            "kind discriminator missing from JSON: {}", json);
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::Profile(m) => {
                assert_eq!(m.focal_params, vec!["R0", "gamma"]);
                assert_eq!(m.grid.len(), 2);
                assert_eq!(m.grid[0].values, vec![40.0, 50.0, 60.0]);
                assert_eq!(m.total_jobs, 18);
            }
            _ => panic!("expected Profile kind"),
        }
    }

    #[test]
    fn survey_run_roundtrip() {
        let mut bounds = HashMap::new();
        bounds.insert("beta".into(), (0.001_f64, 1.0_f64));
        bounds.insert("gamma".into(), (0.01_f64, 0.5_f64));
        let mut data_hashes = HashMap::new();
        data_hashes.insert("cases".into(), "0123abcd".repeat(8));
        let r = Run {
            hash: "fa".repeat(32),
            version: "v".into(),
            created_at: "2026-05-03T00:00:00Z".into(),
            argv: vec!["camdl".into(), "survey".into(), "sir.camdl".into()],
            status: RunStatus::Completed { wall_time_seconds: 12.5 },
            label: None,
            kind: RunKind::Survey(SurveyMeta {
                model: "sir.camdl".into(),
                model_hash: "f00d".repeat(16),
                data_hashes,
                bounds,
                n_points: 1000,
                eval_method: SurveyEvalMethod::Pfilter,
                eval_particles: 200,
                eval_replicates: 3,
                seed: 42,
                fixed: HashMap::new(),
                scenario: None,
                estimated: vec!["beta".into(), "gamma".into()],
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains(r#""kind":"survey""#),
            "kind discriminator missing from JSON: {}", json);
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::Survey(m) => {
                assert_eq!(m.n_points, 1000);
                assert_eq!(m.eval_method, SurveyEvalMethod::Pfilter);
                assert_eq!(m.eval_particles, 200);
                assert_eq!(m.eval_replicates, 3);
                assert_eq!(m.estimated, vec!["beta", "gamma"]);
            }
            _ => panic!("expected Survey"),
        }
    }

    #[test]
    fn survey_eval_method_serializes_lowercase() {
        let p = SurveyEvalMethod::Pfilter;
        assert_eq!(serde_json::to_string(&p).unwrap(), r#""pfilter""#);
        let s = SurveyEvalMethod::Simulate;
        assert_eq!(serde_json::to_string(&s).unwrap(), r#""simulate""#);
    }

    #[test]
    fn fit_stage_with_profile_backref_roundtrips() {
        // A grid-point × start child under a profile: FitStageMeta with
        // parent_profile_hash + profile_point_idx + profile_start_idx
        // populated. Verifies the optional fields round-trip correctly.
        let r = Run {
            hash: "e".repeat(64),
            version: "v".into(),
            created_at: "2026-04-24T00:00:00Z".into(),
            argv: vec!["camdl".into(), "profile".into()],
            status: RunStatus::Completed { wall_time_seconds: 120.0 },
            label: None,
            kind: RunKind::FitStage(FitStageMeta {
                fit_hash: "".into(),    // no parent Fit; parent is a Profile
                stage: "if2".into(),
                method: MethodKind::If2,
                seed: 142,
                n_chains: 1,
                algorithm: serde_json::Value::Null,
                best_loglik: Some(-5827.35),
                best_chain: Some(0),
                starts_from: None,
                derived_from: None,
                parent_profile_hash: Some("f".repeat(64)),
                profile_point_idx: Some(7),
                profile_start_idx: Some(2),
            }),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("parent_profile_hash"));
        assert!(json.contains("profile_point_idx"));
        assert!(json.contains("profile_start_idx"));
        let parsed: Run = serde_json::from_str(&json).unwrap();
        match parsed.kind {
            RunKind::FitStage(m) => {
                assert_eq!(m.parent_profile_hash, Some("f".repeat(64)));
                assert_eq!(m.profile_point_idx, Some(7));
                assert_eq!(m.profile_start_idx, Some(2));
            }
            _ => panic!("expected FitStage"),
        }
    }

    #[test]
    fn run_status_running_serializes_as_lowercase_string() {
        let s = RunStatus::Running;
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#""running""#);
        let parsed: RunStatus = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_running());
        assert_eq!(parsed.wall_time_seconds(), None);
    }

    #[test]
    fn run_status_completed_carries_wall_time() {
        let s = RunStatus::Completed { wall_time_seconds: 42.5 };
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, r#"{"completed":{"wall_time_seconds":42.5}}"#);
        let parsed: RunStatus = serde_json::from_str(&json).unwrap();
        assert!(!parsed.is_running());
        assert_eq!(parsed.wall_time_seconds(), Some(42.5));
    }

    #[test]
    fn run_status_round_trips_inside_run() {
        // The status field round-trips inside a full Run wrapper, so
        // run.json writes can read back into the typed RunStatus.
        let r = sample_simulate_run();
        let json = serde_json::to_string(&r).unwrap();
        let parsed: Run = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed.status,
            RunStatus::Completed { wall_time_seconds }
            if (wall_time_seconds - 1.23).abs() < 1e-9));
    }
}
