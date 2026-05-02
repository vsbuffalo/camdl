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
        ParticleState,
        ChainBinomialProcess, MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
        traits::ObservationModel,
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

// ─── Observation family resolution ───────────────────────────────────────────

/// Resolve `--obs <name>` against the IR's observation list.
///
/// gh#38: profile must walk the *full* indexed observation family and
/// sum log-likelihood across every concrete stream that descended from
/// it — matching `camdl fit run`. Previously profile silently scored
/// only the first IR observation entry, producing a profile loglik
/// ~5 orders of magnitude off what the user thought they were
/// plotting on a 15-cell stratified family like typhoid's
/// `cases[s in setting, a in age]`.
///
/// Resolution rules:
///
/// 1. If `obs_name` is supplied:
///    a. Exact match against an IR obs name → single-stream profile.
///       (Family-name lookup never sees this — it's the leaf of an
///        already-expanded family.)
///    b. Otherwise treat the name as a family root and match every
///       IR obs whose name starts with `<name>_` (the OCaml expander
///       names indexed observations as `<family>_<idx1>_<idx2>...`).
///    c. No matches → `Err`.
/// 2. If `obs_name` is `None`:
///    a. Exactly one IR observation → use it.
///    b. Multiple → `Err` listing available names.
///    c. Zero → `Err`.
///
/// Returns the borrowed IR observations, in IR declaration order. The
/// returned slice is non-empty on `Ok`.
pub(crate) fn resolve_obs_family<'a>(
    observations: &'a [ir::observation::ObservationModel],
    obs_name: Option<&str>,
) -> Result<Vec<&'a ir::observation::ObservationModel>, String> {
    match obs_name {
        Some(name) => {
            // 1a. Exact match (single stream).
            let exact: Vec<_> = observations.iter().filter(|o| o.name == name).collect();
            if !exact.is_empty() {
                return Ok(exact);
            }
            // 1b. Family match.
            let prefix = format!("{}_", name);
            let family: Vec<_> = observations.iter()
                .filter(|o| o.name.starts_with(&prefix))
                .collect();
            if family.is_empty() {
                let avail = observations.iter().map(|o| o.name.as_str())
                    .collect::<Vec<_>>().join(", ");
                return Err(format!(
                    "error: --obs '{}' matches no observation in the IR.\n\
                     Available observation names: {}",
                    name, avail));
            }
            Ok(family)
        }
        None => {
            if observations.is_empty() {
                Err("error: model has no observations block".to_string())
            } else if observations.len() == 1 {
                Ok(vec![&observations[0]])
            } else {
                let avail = observations.iter().map(|o| o.name.as_str())
                    .collect::<Vec<_>>().join(", ");
                Err(format!(
                    "error: model declares {} observation streams. Pass `--obs <NAME>` \
                     to select one (exact stream name) or a family root (e.g. `cases` \
                     for an indexed `cases[s,a]` block — sums all expanded streams).\n\
                     Available: {}",
                    observations.len(), avail))
            }
        }
    }
}

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
    /// Full SHA-256 of the `--data` file's bytes. Content-only — the
    /// path is not part of this hash, so two users with the same TSV
    /// at different paths share a cache entry, while two users with
    /// different TSVs at the same path do not (gh#39: editing the data
    /// file in place must invalidate the cache, not silently return the
    /// previous run's logliks against the old observations).
    pub data_hash: String,
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
    /// Hash of an upstream stage's content this profile starts from.
    /// `None` for standalone profile invocations.
    pub starts_from_lineage: Option<String>,
    /// Per-seed: the actual seed value. `inner_hash` excludes this;
    /// `content_hash` (trait method) includes it.
    pub seed: u64,
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
        hash_canonical(&[
            ("model",       &self.model_hash),
            ("base_params", &self.base_params_hash),
            ("focal_grid",  &grid_canonical),
            ("fixed",       &fixed_sorted.join(",")),
            ("obs_family",  &self.obs_family),
            ("if2",         &if2),
            ("starts_from", self.starts_from_lineage.as_deref().unwrap_or("")),
            ("data",        &self.data_hash),
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
        })
    }
}

pub fn cmd_profile(a: &crate::args::ProfileArgs) {
    // ── Backend / optimizer cross-flag validation (gh#40) ──────────
    //
    // CLAUDE.md "no loose semantics": never silently accept a flag
    // combination whose meaning is ambiguous. `--optimizer` is only
    // meaningful with `--backend ode` — passing it under
    // `--backend chain_binomial` is a configuration error, not a
    // hint we should silently honour or ignore.
    use crate::args::{ProfileBackend, ProfileOptimizer};
    if matches!(a.backend, ProfileBackend::ChainBinomial) && a.optimizer.is_some() {
        eprintln!(
            "error: --optimizer is only meaningful with --backend ode \
             (passed --optimizer {:?} but --backend chain_binomial). \
             Drop --optimizer, or pass --backend ode if you intended \
             the deterministic ODE-backed profile.",
            a.optimizer.unwrap_or(ProfileOptimizer::Sbplx),
        );
        std::process::exit(1);
    }
    let ir_path = a.model.to_string_lossy().into_owned();
    let data_path = a.data.to_string_lossy().into_owned();
    let n_particles = a.inference.particles;
    let n_iterations = a.iterations;
    let n_starts = a.starts;
    let cooling = a.cooling;
    let dt = a.inference.dt;
    let seed_base = a.inference.seed;
    let parallel = a.inference.parallel;
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
    let overrides: HashMap<String, f64> = a.model_overrides.param.iter()
        .map(|p| (p.name.clone(), p.value))
        .collect();

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

    // Load model
    let (mut model, model_json) = crate::util::load_model(&ir_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    for pf in &a.model_overrides.params {
        crate::util::apply_params_file(&mut model, &pf.to_string_lossy())
            .unwrap_or_else(|e| { eprintln!("{}", e); std::process::exit(1); });
    }
    if let Some(ref name) = scenario_name {
        if let Some(preset) = model.presets.iter().find(|p| p.name == *name) {
            for p in &mut model.parameters {
                if let Some(&v) = preset.params.get(&p.name) { p.value = Some(v); }
            }
        }
    }
    for p in &mut model.parameters {
        if let Some(&v) = overrides.get(&p.name) { p.value = Some(v); }
    }

    // Bounds + finite-value check after all override paths resolved (gh#31).
    crate::util::validate_parameter_values(&model)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    let compiled = Arc::new(CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { eprintln!("{:?}", e); std::process::exit(1); }));
    let base_params = compiled.default_params.clone();

    // ── Resolve --obs against the IR's observation list ─────────────
    //
    // gh#38: profile must walk the *full* indexed observation family
    // and sum log-likelihood across every concrete stream that
    // descended from it — matching `camdl fit run`. Previously this
    // path silently scored only the first IR observation (e.g.
    // `cases_medium_a02`), producing a profile loglik ~5 orders of
    // magnitude off what the user thought they were plotting on a
    // 15-cell stratified family like typhoid's
    // `cases[s in setting, a in age]`.
    //
    // Resolution rules (mirrors `pfilter` for the single-stream case
    // and `fit run` for the family case):
    //
    // 1. If `--obs <name>` is supplied:
    //    a. Exact match against an IR obs name → single-stream profile.
    //       (Family-name lookup never sees this — it's the leaf of an
    //        already-expanded family.)
    //    b. Otherwise treat `<name>` as a family root and match every
    //       IR obs whose name starts with `<name>_` (the OCaml expander
    //       names indexed observations as `<family>_<idx1>_<idx2>...`).
    //    c. No matches → hard error.
    // 2. If `--obs` is omitted:
    //    a. Exactly one IR observation → use it.
    //    b. Multiple → error, list available names + family roots.
    //    c. Zero → error (model declares no observations block).
    //
    // `--flow <name>` is only meaningful for single-stream profiles
    // (it overrides the obs model's projection to a custom flow sum).
    // It's incompatible with a multi-stream resolution because each
    // stream has its own per-stratum projection.
    let obs_name_arg = a.flow.obs.clone();
    let resolved_obs: Vec<&ir::observation::ObservationModel> =
        resolve_obs_family(&model.observations, obs_name_arg.as_deref())
            .unwrap_or_else(|e| { eprintln!("{}", e); std::process::exit(1); });

    if resolved_obs.len() > 1 && flow_name.is_some() {
        eprintln!(
            "error: --flow <NAME> is incompatible with a multi-stream observation \
             family. `--obs '{}'` resolved to {} concrete streams (each has its own \
             per-stratum projection); `--flow` only makes sense when scoring a single \
             stream against a custom flow override.",
            obs_name_arg.as_deref().unwrap_or(""), resolved_obs.len(),
        );
        std::process::exit(1);
    }

    if resolved_obs.len() > 1 {
        eprintln!(
            "profile: --obs '{}' resolved to {} expanded streams \
             (joint loglik = sum across all)",
            obs_name_arg.as_deref().unwrap_or(""), resolved_obs.len(),
        );
    } else {
        eprintln!("profile: using observation model '{}' from IR",
            resolved_obs[0].name);
    }

    // Load each stream's column from the data TSV. The lookup keys
    // on `obs.name` to match the rest of the toolchain:
    //
    // * `camdl fit run` (runner.rs:271) loads per stream by the model
    //   observation's `name`, not its `data_stream`. Profile must
    //   agree so a `--data <file>.tsv` produced by `camdl simulate
    //   --obs-only` (which writes columns by `name`) reads back
    //   identically under both commands.
    // * The IR's `data_stream` field is preserved as the *declarer's*
    //   intended source-file column (see ocaml/lib/compiler/expander.ml
    //   ~line 2958), but for the runtime data-loading path the
    //   `name`/`data_stream` distinction is dead code today — fit
    //   uses `name`, simulate writes by `name`, and the wide-TSV
    //   convention assumed by indexed-obs families (e.g. typhoid's
    //   `cases_<setting>_<age>`) makes `name == data_stream`
    //   anyway when no explicit override was declared.
    //
    // Fallback for single-stream models: if the user supplied a 2-col
    // (time, value) TSV with a non-matching column name, accept it
    // via `load_data_tsv_pub` — same behaviour profile had before the
    // multi-stream rewrite, and matches what `camdl pfilter` does.
    // For multi-stream resolution every column must match by name;
    // no ambiguity is allowed.
    let load_stream_obs = |column: &str| -> Vec<Observation> {
        let result = if resolved_obs.len() == 1 {
            // Single stream: try by-name first, fall back to first
            // value column if the TSV has only (time, value).
            crate::pfilter::load_data_tsv_column(&data_path, column)
                .or_else(|_| crate::pfilter::load_data_tsv_pub(&data_path))
        } else {
            crate::pfilter::load_data_tsv_column(&data_path, column)
        };
        match result {
            Ok(v) => v.into_iter().map(|o| Observation { time: o.time, value: o.value }).collect(),
            Err(e) => {
                eprintln!("error: cannot load data column '{}' from {}: {}",
                    column, data_path, e);
                std::process::exit(1);
            }
        }
    };

    let mut per_stream_obs: Vec<Vec<Observation>> = Vec::with_capacity(resolved_obs.len());
    let mut canonical_times: Option<Vec<f64>> = None;
    for obs in &resolved_obs {
        let stream_obs = load_stream_obs(&obs.name);
        let times: Vec<f64> = stream_obs.iter().map(|o| o.time).collect();
        match &canonical_times {
            None => canonical_times = Some(times),
            Some(ct) => {
                if ct.len() != times.len()
                    || ct.iter().zip(&times).any(|(a, b)| (a - b).abs() > 1e-9)
                {
                    eprintln!(
                        "error: observation times for stream '{}' differ from the first \
                         resolved stream. All streams in a profile family must share \
                         identical observation times.",
                        obs.name
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

    let fixed_names: std::collections::HashSet<String> = a.fixed.iter().cloned().collect();
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

    let specs: Vec<crate::fit::runner::ParamSpec> = param_names_to_estimate.iter().map(|name| {
        crate::fit::runner::ParamSpec {
            name: name.clone(),
            rw_sd: rw_sd_map_raw.get(name).and_then(|v| *v),
            transform: None,
            ivp: false,
        }
    }).collect();

    let if2_params = crate::fit::runner::build_if2_params_from_specs(
        &model, &compiled, &base_params, &specs,
    ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    let if2_params = Arc::new(if2_params);

    let process = Arc::new(ChainBinomialProcess::new(compiled.clone()));
    // Build one StreamSpec per resolved IR observation. For
    // single-stream profiles `--flow <name>` overrides the IR
    // projection (forces incidence over the named transition family);
    // for multi-stream we always use each stream's IR projection
    // (the `--flow` + multi-stream combination was already rejected
    // upstream).
    let obs_model_obj: Arc<dyn ObservationModel<ParticleState> + Send + Sync> = {
        let obs_times: Vec<f64> = observations.iter().map(|o| o.time).collect();
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
                obs_times: obs_times.clone(),
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

    // gh#39: hash the --data file's bytes once at launch. The previous
    // CAS key omitted observation data entirely, so a user editing
    // `cases.tsv` in place silently got the prior run's logliks back
    // (correct shape, wrong likelihood). Mirrors the fit-side pattern
    // (see `FitConfigV2::fit_content_hash`). Path-independent: only
    // the bytes participate, so moving the file or pointing two
    // commands at copies of the same TSV still produces a cache hit.
    let data_bytes = std::fs::read(&data_path).unwrap_or_else(|e| {
        eprintln!("error: cannot read --data file '{}': {}", data_path, e);
        std::process::exit(1);
    });
    let data_hash = crate::hashing::sha256_hex(&data_bytes);

    let template_inputs = ProfileInputs {
        model_path: ir_path.clone(),
        stem: stem.clone(),
        model_hash: model_hash.clone(),
        base_params_hash,
        data_hash,
        focal_grid: grid_spec,
        fixed: a.fixed.clone(),
        obs_family: obs_family_key,
        if2_config: ProfileIf2Config {
            n_particles, n_iterations, cooling, dt, n_starts,
        },
        starts_from_lineage: None,
        seed: seeds[0],   // overwritten per-seed below
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
    eprintln!("profile: {} grid ({}) × {} starts × {} seeds = {} IF2 runs ({} particles × {} iter each)",
        grid_points.len(), dim_str, n_starts, seeds.len(), total_jobs,
        n_particles, n_iterations);

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
        let focal_values: Vec<f64> = grid_points[grid_idx].iter().map(|&(_, v)| v).collect();
        let seed = seeds[seed_idx];

        // Pin focal parameters
        let mut params = base_params.clone();
        for &(idx, val) in &grid_points[grid_idx] {
            params[idx] = val;
        }

        let config = IF2Config {
            n_particles, n_iterations,
            cooling_fraction: cooling, cooling_target_iters: n_iterations, dt,
            t_start: process.compiled.model.simulation.t_start,
            simplex_groups: vec![],
            skip_first_obs_from_loglik: false,
        };
        // job_seed derives from this REPLICATE's seed, not a global
        // seed_base — different seeds in --seeds drive distinct IF2
        // noise (the whole point of multi-seed sensitivity).
        let job_seed = seed ^ (grid_idx as u64 * 1000 + start_idx as u64);
        let job_t0 = std::time::Instant::now();

        let result = run_if2(
            &*process, &*obs_model_obj, &params, &if2_params, &config, job_seed,
        );
        let elapsed = job_t0.elapsed().as_secs_f64();

        let seed_dir = &seed_dirs[seed_idx];
        let start_dir = profile_point_start_dir(seed_dir, grid_idx, start_idx);
        if let Err(e) = std::fs::create_dir_all(&start_dir) {
            eprintln!("warning: cannot create {}: {}", start_dir.display(), e);
            return;
        }

        let (final_loglik, mle_params): (f64, Vec<f64>) = match result {
            Ok(r) => (r.final_loglik, r.mle),
            Err(_) => (f64::NEG_INFINITY, params.clone()),
        };

        let mle_toml = render_mle_toml(&if2_params, &focal_values,
            &focal_grids.iter().map(|fg| fg.name.as_str()).collect::<Vec<_>>(),
            &mle_params, final_loglik);
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
                stage: "if2".to_string(),
                method: crate::run_meta::MethodKind::If2,
                seed: job_seed,
                n_chains: 1,
                algorithm: serde_json::json!({
                    "particles":  n_particles,
                    "iterations": n_iterations,
                    "cooling":    cooling,
                    "dt":         dt,
                }),
                best_loglik: if final_loglik.is_finite() { Some(final_loglik) } else { None },
                best_chain:  Some(0),
                starts_from: None,
                derived_from: None,
                parent_profile_hash: Some(parent_profile_hash.clone()),
                profile_point_idx:   Some(grid_idx),
                profile_start_idx:   Some(start_idx),
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
fn render_mle_toml(
    if2_params: &[sim::inference::if2::EstimatedParam],
    focal_values: &[f64],
    focal_names: &[&str],
    mle: &[f64],
    final_loglik: f64,
) -> String {
    let mut body = String::new();
    body.push_str("# Per-start MLE for one profile grid point.\n\n");
    body.push_str(&format!("final_loglik = {}\n\n", final_loglik));
    body.push_str("[focal]\n");
    for (name, v) in focal_names.iter().zip(focal_values.iter()) {
        body.push_str(&format!("{} = {}\n", name, v));
    }
    body.push_str("\n[mle]\n");
    for spec in if2_params.iter() {
        body.push_str(&format!("{} = {}\n", spec.name, mle[spec.index]));
    }
    body
}

/// Scan the per-start CAS tree and rewrite `profile.tsv` as the
/// derived rollup. One row per grid point, each row the winning start
/// (max final_loglik) across `n_starts`. Written atomically via
/// tmp-then-rename so concurrent rollups (from racing threads) never
/// expose a truncated intermediate.
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
            rows.push(RollupRow {
                focal_values: best.focal_values,
                best_loglik: best.final_loglik,
                best_start_idx: best_start,
                mle: best.mle,
                wall_time_sum,
            });
        }
    }

    // Render.
    let mut body = String::new();
    body.push_str(&format!("# {}\n", crate::version::VERSION));
    body.push_str(&format!("# total_points={} completed={}\n",
        n_grid_points, rows.len()));
    for name in focal_names { body.push_str(&format!("{}\t", name)); }
    body.push_str("best_loglik\tbest_start_idx\twall_time_seconds");
    for spec in if2_params.iter() { body.push_str(&format!("\t{}", spec.name)); }
    body.push('\n');
    for row in &rows {
        for v in &row.focal_values { body.push_str(&format!("{:.4}\t", v)); }
        body.push_str(&format!("{:.4}\t{}\t{:.3}",
            row.best_loglik, row.best_start_idx, row.wall_time_sum));
        for spec in if2_params.iter() {
            body.push_str(&format!("\t{:.6}", row.mle[spec.index]));
        }
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
    best_start_idx: usize,
    mle: Vec<f64>,
    wall_time_sum: f64,
}

struct ParsedMle {
    final_loglik: f64,
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

    Some(ParsedMle { final_loglik, focal_values, mle: mle_values })
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

    // Map: focal-values key (as the canonical TSV-column strings, so
    // grid points with identical floats group together regardless of
    // formatting) → list of (best_loglik, mle_vec) per seed.
    let mut by_grid: BTreeMap<Vec<String>, Vec<(f64, Vec<f64>)>> = BTreeMap::new();
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
            // Layout: focal_1 ... focal_N | best_loglik | best_start_idx |
            //         wall_time_seconds | mle_param_1 ... mle_param_M
            if cols.len() < focal_names.len() + 3 + if2_params.len() { continue; }

            let focal_key: Vec<String> = cols[..focal_names.len()]
                .iter().map(|s| s.trim().to_string()).collect();
            let Ok(best_loglik) = cols[focal_names.len()].parse::<f64>() else { continue; };

            let mle_start = focal_names.len() + 3;
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
            by_grid.entry(focal_key).or_default().push((best_loglik, mle));
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
    for name in focal_names { body.push_str(&format!("{}\t", name)); }
    body.push_str("loglik");
    if multi_seed {
        body.push_str("\tloglik_sd\tloglik_min\tloglik_max");
    }
    for spec in if2_params.iter() {
        body.push_str(&format!("\t{}", spec.name));
        if multi_seed {
            body.push_str(&format!("\t{}_sd", spec.name));
        }
    }
    body.push('\n');

    for (focal_key, samples) in &by_grid {
        for v in focal_key { body.push_str(&format!("{}\t", v)); }
        let logliks: Vec<f64> = samples.iter().map(|(ll, _)| *ll)
            .filter(|x| x.is_finite()).collect();
        let (mean_ll, sd_ll, min_ll, max_ll) = summary_stats(&logliks);
        body.push_str(&format!("{:.4}", mean_ll));
        if multi_seed {
            body.push_str(&format!("\t{:.4}\t{:.4}\t{:.4}", sd_ll, min_ll, max_ll));
        }
        for spec in if2_params.iter() {
            let vals: Vec<f64> = samples.iter()
                .filter_map(|(_, mle)| mle.get(spec.index).copied())
                .filter(|x| x.is_finite()).collect();
            let (m, s, _, _) = summary_stats(&vals);
            body.push_str(&format!("\t{:.6}", m));
            if multi_seed {
                body.push_str(&format!("\t{:.6}", s));
            }
        }
        body.push('\n');
    }

    let final_path = umbrella_dir.join("summary.tsv");
    let tmp_path = umbrella_dir.join("summary.tsv.tmp");
    std::fs::write(&tmp_path, body)?;
    std::fs::rename(&tmp_path, &final_path)?;
    Ok(())
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
        s.push_str("best_loglik\tbest_start_idx\twall_time_seconds");
        for i in 0..rows[0].2.len() { s.push_str(&format!("\tparam_{}", i)); }
        s.push('\n');
        for (focal, ll, mle) in rows {
            for v in focal { s.push_str(&format!("{}\t", v)); }
            s.push_str(&format!("{:.4}\t0\t0.0", ll));
            for v in mle { s.push_str(&format!("\t{:.6}", v)); }
            s.push('\n');
        }
        std::fs::write(seed_dir.join("profile.tsv"), s).unwrap();
    }

    fn data_lines(text: &str) -> Vec<&str> {
        text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()).collect()
    }

    #[test]
    fn n1_summary_uses_bare_names_only() {
        // gh#30 option A, n=1 (the common case): the schema is
        //   <focal>  loglik  <param_1>  <param_2>  ...
        // No `_sd` / `_min` / `_max` — there's no aggregation to
        // describe.
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
        assert_eq!(cols, vec!["s0", "loglik", "R0", "alpha"],
            "n=1 schema must be focal + bare loglik + bare params; got {:?}", cols);

        // No spread columns must leak through.
        for forbidden in &["loglik_sd", "loglik_min", "loglik_max",
                           "R0_sd", "alpha_sd", "n_seeds",
                           "mean_loglik", "max_loglik", "R0_mean", "alpha_mean"] {
            assert!(!header.contains(forbidden),
                "n=1 header must not contain {:?}: {}", forbidden, header);
        }

        // Two data rows, four columns each, no all-zero `_sd` chaff.
        assert_eq!(lines.len(), 3, "expected header + 2 grid rows: {:?}", lines);
        assert_eq!(lines[1].split('\t').count(), 4);
        assert_eq!(lines[2].split('\t').count(), 4);
    }

    #[test]
    fn multi_seed_summary_appends_spread_columns() {
        // gh#30 option A, n>1: bare names stay, `_sd / _min / _max`
        // are appended additively. Bare loglik = mean across seeds;
        // bare param = mean across seeds.
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
        // Bare/_sd adjacency for params; `_sd / _min / _max` appended
        // after the bare loglik.
        assert_eq!(cols, vec![
            "s0", "loglik", "loglik_sd", "loglik_min", "loglik_max",
            "R0", "R0_sd", "alpha", "alpha_sd",
        ], "n>1 schema: {:?}", cols);

        // Bare `loglik` value is the mean across seeds at the first
        // grid cell: mean(-42.5, -42.0, -43.0) = -42.5
        let row1: Vec<&str> = lines[1].split('\t').collect();
        let bare_loglik: f64 = row1[1].parse().unwrap();
        assert!((bare_loglik - (-42.5)).abs() < 1e-3,
            "bare loglik should be the cross-seed mean, got {}", bare_loglik);
    }

    // ── gh#38 obs-family resolution ─────────────────────────────────

    /// Build a synthetic `ObservationModel` for resolution tests.
    /// Likelihood/projection/schedule fields are placeholders — only
    /// `name` is exercised by `resolve_obs_family`.
    fn make_obs(name: &str) -> ir::observation::ObservationModel {
        use ir::observation::{
            Likelihood, ObservationSchedule, PoissonLikelihood,
            Projection, RegularSchedule,
        };
        use ir::expr::Expr;
        ir::observation::ObservationModel {
            name: name.to_string(),
            data_stream: name.to_string(),
            schedule: ObservationSchedule::Regular(RegularSchedule {
                start: 0.0, step: 1.0, end: 10.0,
            }),
            projection: Projection::CumulativeFlow("flow".to_string()),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: Expr::Const(ir::expr::ConstExpr { value: 1.0 }),
            }),
        }
    }

    #[test]
    fn resolve_obs_family_root_matches_full_indexed_expansion() {
        // gh#38 core: passing the family root resolves to ALL expanded
        // streams (not just the first). The OCaml expander emits names
        // like `<family>_<idx1>_<idx2>...`.
        let obs = vec![
            make_obs("cases_medium_a02"),
            make_obs("cases_medium_a25"),
            make_obs("cases_high_a02"),
            make_obs("cases_high_a25"),
            make_obs("cases_veryhigh_a02"),
        ];
        let resolved = resolve_obs_family(&obs, Some("cases")).unwrap();
        assert_eq!(resolved.len(), 5,
            "family root 'cases' must match all 5 expanded streams, got {}",
            resolved.len());
        let names: Vec<&str> = resolved.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, vec![
            "cases_medium_a02", "cases_medium_a25",
            "cases_high_a02", "cases_high_a25", "cases_veryhigh_a02",
        ]);
    }

    #[test]
    fn resolve_obs_family_exact_match_takes_precedence_over_family() {
        // 1a: an exact match against an expanded leaf must short-circuit
        // — passing `cases_medium_a02` should give back exactly that
        // stream, not also the other `cases_medium_a02_*` siblings if
        // the model happened to declare any.
        let obs = vec![
            make_obs("cases_medium_a02"),
            make_obs("cases_medium_a02_extra"),
            make_obs("cases_medium_a25"),
        ];
        let resolved = resolve_obs_family(&obs, Some("cases_medium_a02")).unwrap();
        assert_eq!(resolved.len(), 1,
            "exact match must take precedence over family expansion");
        assert_eq!(resolved[0].name, "cases_medium_a02");
    }

    #[test]
    fn resolve_obs_family_unknown_name_errors() {
        let obs = vec![make_obs("cases_a02"), make_obs("cases_a25")];
        let err = resolve_obs_family(&obs, Some("deaths")).unwrap_err();
        assert!(err.contains("deaths") && err.contains("Available"),
            "error should name the unknown obs and list available, got: {}", err);
    }

    #[test]
    fn resolve_obs_family_no_arg_picks_unique_obs() {
        let obs = vec![make_obs("reported_cases")];
        let resolved = resolve_obs_family(&obs, None).unwrap();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "reported_cases");
    }

    #[test]
    fn resolve_obs_family_no_arg_with_multiple_errors() {
        // No --obs + multiple observations is the failure mode that
        // the gh#38 fix is closing: previously profile silently picked
        // the first one and reported "using observation model
        // 'cases_medium_a02' from IR", off by orders of magnitude.
        let obs = vec![
            make_obs("cases_a02"),
            make_obs("cases_a25"),
        ];
        let err = resolve_obs_family(&obs, None).unwrap_err();
        assert!(err.contains("Pass `--obs"),
            "error should prompt user to pass --obs, got: {}", err);
        assert!(err.contains("cases_a02") && err.contains("cases_a25"),
            "error should list available streams, got: {}", err);
    }

    #[test]
    fn resolve_obs_family_no_observations_errors() {
        let obs: Vec<ir::observation::ObservationModel> = vec![];
        assert!(resolve_obs_family(&obs, None).is_err());
        assert!(resolve_obs_family(&obs, Some("cases")).is_err());
    }

    // ── gh#39 data-file content hashing ─────────────────────────────────

    /// Construct a `ProfileInputs` with all content fields fixed to
    /// stable placeholders. Caller overrides only the field under test
    /// (typically `data_hash`) so cross-test comparisons are crisp.
    fn fixture_inputs(data_hash: &str) -> ProfileInputs {
        ProfileInputs {
            model_path: "model.camdl".into(),
            stem: Some("model".into()),
            model_hash: "deadbeef".repeat(8),
            base_params_hash: "cafef00d".repeat(8),
            data_hash: data_hash.to_string(),
            focal_grid: vec![GridAxis {
                param: "R0".into(),
                values: vec![1.5, 2.0, 2.5],
            }],
            fixed: vec![],
            obs_family: "cases".into(),
            if2_config: ProfileIf2Config {
                n_particles: 100, n_iterations: 50, cooling: 0.5, dt: 1.0, n_starts: 4,
            },
            starts_from_lineage: None,
            seed: 1,
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

    #[test]
    fn inner_hash_data_field_is_load_bearing() {
        // Cross-check against the canonical-key implementation: with
        // every other field fixed, varying only `data_hash` must move
        // the inner_hash. This catches a future refactor that
        // accidentally drops the `("data", ...)` entry from the
        // canonical-keys vector.
        let a = fixture_inputs(&"a".repeat(64));
        let b = fixture_inputs(&"b".repeat(64));
        assert_ne!(a.inner_hash().full(), b.inner_hash().full(),
            "data_hash must be wired into inner_hash's canonical keys");
    }
}

