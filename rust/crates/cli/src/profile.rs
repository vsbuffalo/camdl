//! `camdl profile` — profile likelihood via parallel IF2 runs.
//!
//! For one or more focal parameters, fix them at a grid of values and
//! run IF2 to maximise over the remaining parameters at each grid
//! point. The profile likelihood shows how the MLE changes as you move
//! the focal parameter(s) — revealing identifiability, confidence
//! intervals, and parameter interactions. 2D profiles (two `--sweep`
//! flags) produce a likelihood surface suitable for contour plotting.
//!
//! ## Content-addressed layout
//!
//! Each `(grid point × seed × start)` is an independent content-
//! addressed mini-fit, keyed by the five factored `runid` levels
//! (`profile` / `point` / `stage` / `seed` / `start`) resolved in
//! [`crate::profile_cas`]. The leaf is an `ArtifactKind::ProfilePoint` run
//! under `profiles/`; each path segment is `{label}-{hash8}`:
//!
//! ```text
//! results/profiles/{stem}-{h8}/{point}-{h8}/{stage}-{h8}/seed_{n}-{h8}/start_{k}-{h8}/
//!   run.json                                # RunRecord (kind = profile_point)
//!   mle.toml                                # MLE at this (point, seed, start)
//! ```
//!
//! Leaves are written via the store's streaming claim (staging dir +
//! atomic finalize): a crash mid-IF2 leaves no finalized `run.json`,
//! so the next invocation reruns only that leaf while completed leaves
//! are reused bit-for-bit. The per-seed profile curve and the
//! cross-seed summary are cross-leaf aggregates with no single home in
//! the factored tree — they are rebuilt from the derived index.

use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        if2::{run_if2, IF2Config, Observation},
        pmmh::{run_pmmh, PMMHConfig, Prior},
        BoundObs, ChainBinomialProcess, MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
    },
};
use std::collections::HashMap;
use std::sync::Arc;

use crate::cas::typed::ContentHash;
use crate::run_paths::output_root;

/// Burn-in discarded from each per-cell PMMH chain on the profile path.
/// Fixed (not user-tunable) here; the cell only reports its MAP sample,
/// so the chain just needs to clear transient. Named so the early
/// `pmmh_steps > burn_in` validation and the `PMMHConfig` site below
/// share one source of truth and cannot drift.
const PROFILE_PMMH_BURN_IN: usize = 100;

/// gh#102 (H10): reject a per-cell PMMH chain length that the fixed
/// burn-in would consume entirely. With `steps <= burn_in` every
/// sample is discarded, leaving zero post-burn-in records — the cell
/// silently yields empty diagnostics rather than a chain to summarise.
/// Pure (no I/O, no exit) so it is unit-testable; `cmd_profile` maps
/// the `Err` to a stderr message + `exit(1)`.
fn validate_profile_pmmh_steps(steps: usize, burn_in: usize) -> Result<(), String> {
    if steps <= burn_in {
        return Err(format!(
            "--pmmh-steps = {} must exceed the per-cell burn-in ({}). \
             Profile PMMH discards the first {} samples of each cell's \
             chain; with steps <= burn-in no post-burn-in samples survive \
             and the cell yields empty diagnostics. Re-run with \
             `--pmmh-steps {}` or more.",
            steps, burn_in, burn_in, burn_in + 1));
    }
    Ok(())
}

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
}

// Observation family resolution lives in `crate::util::resolve_data_specs`
// (gh#90). Profile's previous `resolve_obs_family` (gh#38) is subsumed
// there: `--data PATH --obs <family-root>` matches every IR obs whose
// name starts with `<root>_`, and `--data NAME=PATH` (gh#90 named form)
// also expands NAME as a family root. Same semantics, single dispatch.


pub fn cmd_profile(a: &crate::args::ProfileArgs) {
    // Validate (algorithm, backend) early, before any expensive setup.
    let algo_name = a.algorithm.as_deref().unwrap_or("if2");
    let backend_name = a.backend.as_deref().unwrap_or("chain_binomial");
    if let Err(msg) = crate::fit::methods::validate_combo(algo_name, backend_name) {
        eprintln!("error: {}", msg);
        std::process::exit(1);
    }
    // Registry-driven caveat for Beta/Experimental methods (e.g. pmmh, nl-*).
    crate::fit::methods::emit_status_banner(algo_name, backend_name);
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
    // gh#102 (H10): each per-cell PMMH chain discards a fixed burn-in.
    // Reject steps <= burn_in early, before any setup (see
    // validate_profile_pmmh_steps for the rationale).
    if matches!(profile_algo, ProfileAlgo::Pmmh) {
        if let Err(msg) = validate_profile_pmmh_steps(a.pmmh_steps, PROFILE_PMMH_BURN_IN) {
            eprintln!("error: {}", msg);
            std::process::exit(1);
        }
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
        fixed_cfg.expand_from_scenario(&model_pre, &fit_cfg.estimate)
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
            if p.value.resolved_value().is_none() {
                if let Some(start) = spec.start {
                    p.value = p.value.with_value(start);
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
    for (sname, spath) in &bound_streams {
        let path_str = spath.to_string_lossy().into_owned();
        // Strict by-name binding — no positional fallback (G1). The declared
        // `time` column is the axis (by-name-time flip) and `scored` is the
        // value; both must match the file header exactly (a typo'd/wrong-cased
        // header is a located error, not a silent positional bind).
        let obs_block = model.observations.iter()
            .find(|o| &o.name == sname)
            .unwrap_or_else(|| {
                eprintln!("error: no observation block named '{}'", sname);
                std::process::exit(1);
            });
        let time_col = crate::pfilter::obs_time_column(obs_block).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        let result = crate::pfilter::load_data_tsv_column(
            &path_str, time_col, &obs_block.scored, &time_opts);
        let stream_obs: Vec<Observation> = match result {
            Ok(v) => v.into_iter().map(|o| Observation { time: o.time, value: o.value }).collect(),
            Err(e) => {
                eprintln!("error: cannot load data column '{}' from {}: {}",
                    sname, path_str, e);
                std::process::exit(1);
            }
        };
        per_stream_obs.push(stream_obs);
    }

    // Canonical schedule: the sorted-unique UNION of every stream's observation
    // times (multi-cadence, proposal 2026-06-10 §3.3). Downstream code reads it
    // for `obs_times` (the substep grid + the ODE-MLE / PMMH consumers); `bind`
    // re-merges each stream's own schedule to this union and records per-stream
    // `at_union` membership. The old "must share identical observation times"
    // guard was the no-silent-gaps stance for machinery that did not yet exist.
    let observations: Vec<Observation> = {
        let mut times: Vec<f64> = per_stream_obs.iter()
            .flat_map(|obs| obs.iter().map(|o| o.time))
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).expect("observation times are finite"));
        times.dedup();
        times.into_iter().map(|time| Observation { time, value: 0.0 }).collect()
    };
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

    let process = Arc::new(ChainBinomialProcess::new(compiled.clone()));
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

    // gh#193 preflight: correlated PMMH (CPM, rho > 0) pre-draws a fixed-size
    // noise block per observation window, so it requires a (near-)uniform obs
    // grid. The check is θ-independent (obs grid only), so run it ONCE here and
    // fail with the actionable message — otherwise a genuine non-uniform grid
    // surfaces as a silently-swallowed all-(-inf) profile (every per-cell PF
    // eval maps the filter Err to -inf). A leading window coinciding with
    // t_start is fine (scored at the initial state); see validate_cpm_obs_grid.
    if pmmh_rho_opt.is_some() {
        if let Err(e) = sim::inference::correlated_pf::validate_cpm_obs_grid(
            &obs_times_vec, compiled.model.simulation.t_start, dt,
        ) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }

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
            // profile is dense + aux-free in v1 (survey denominators are a
            // `camdl fit` / `pfilter` feature; profile rejects NA upstream).
            // Multi-cadence (§3.3): feed each stream its OWN schedule (from
            // `stream_obs`), NOT the union `obs_times_vec`. `bind` re-merges to
            // the union and records per-stream `at_union` membership; the dense
            // cells have len == this stream's own obs_times.len().
            stream_specs.push(StreamSpec::dense(
                projection,
                (*obs).clone(),
                sim::inference::dense_cells(
                    stream_obs.iter().map(|o| o.value).collect()),
                stream_obs.iter().map(|o| o.time).collect(),
            ));
        }
        let (bound, _report) = BoundObs::bind(stream_specs).unwrap_or_else(|report| {
            eprintln!("error: observation data invalid:\n{}", report.render());
            std::process::exit(1);
        });
        Arc::new(MultiStreamObsModel::new(bound, compiled.clone())
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

    // ── Resolve content-bearing inputs ─────────────────────────────────
    //
    // Each profile leaf's identity is resolved per (point, seed, start)
    // by `profile_cas::resolve_profile_point`; the values gathered here
    // (model + data hashes, base params, fixed list, priors, fit.toml,
    // method config) are its content-bearing inputs.
    let model_hash = crate::hashing::model_hash(&model_json);
    let base_params_hash = {
        let mut lines: Vec<String> = model.parameters.iter()
            .map(|p| format!("{}={}", p.name,
                p.value.resolved_value().unwrap_or(base_params[compiled.param_index[p.name.as_str()]])))
            .collect();
        lines.sort();
        ContentHash::from_bytes(lines.join("\n").as_bytes()).full().to_string()
    };
    // Resolve seeds. --seeds wins; default is the single --seed.
    let seeds: Vec<u64> = match &a.seeds {
        Some(spec) => spec.expand(),
        None => vec![seed_base],
    };
    if seeds.is_empty() {
        eprintln!("error: --seeds expanded to empty list");
        std::process::exit(1);
    }

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

    // Run-level provenance recorded into every profile-point leaf's
    // `RunRecord.inputs` — display payload, NOT identity-bearing, so it
    // does not affect the leaf `run_id`. Folds forward the audit data
    // the old per-run `ProfileMeta` carried: per-parameter resolution
    // provenance (gh#83/gh#85), per-chain init provenance, and the loud
    // `--suppress-warnings` waiver trail.
    let run_provenance_json = serde_json::json!({
        "parameters_provenance":
            serde_json::to_value(&parameters_provenance).unwrap_or(serde_json::Value::Null),
        "init_provenance":
            serde_json::to_value(&init_provenance).unwrap_or(serde_json::Value::Null),
        "suppressed_warnings":
            serde_json::to_value(&suppressed_warnings).unwrap_or(serde_json::Value::Null),
    });

    // gh#89: lowercased algorithm tag — a cache-key input for the
    // method `stage` and the `stage`-segment display label.
    let algorithm = format!("{:?}", profile_algo).to_lowercase();

    // ── gh#147 (M3.3): content-addressed profile-point identity inputs ──
    // A profile point is a CAS leaf at `profiles/<base>/<point>/<stage>/
    // <seed>/<start>/`. The base is a path segment (no base-level record).
    // profile-base = the inference *problem*, with the focal GRID and the
    // method config EXCLUDED (guardrail 1) — the grid rides `point`, the
    // method `stage`. The base fit's `starts_from` rides the base as a dep.
    let mut fixed_blob = fixed_for_cas.clone();
    fixed_blob.sort();
    let mut priors_blob = resolved_priors_kv.clone();
    priors_blob.sort_by(|a, b| a.0.cmp(&b.0));
    let base_config_blob = serde_json::json!({
        "base_params": base_params_hash,
        "fixed":       fixed_blob,
        "obs_family":  obs_family_key,
        "fit_toml":    fit_toml_hash,
        "priors":      priors_blob,
    });
    let method_config_blob = serde_json::json!({
        "algorithm": algorithm,
        "if2": { "particles": n_particles, "iterations": n_iterations,
                 "cooling": cooling, "dt": dt, "starts": n_starts },
        "pmmh": { "steps": pmmh_steps, "particles": pmmh_particles, "rho": pmmh_rho_opt },
    });
    // The base fit's `starts_from` lineage would fold into the base as
    // a dep (guardrail 3-base). `camdl profile` does not currently
    // thread a base-fit lineage, so the dep list is empty; when it
    // does, push the resolved `FitStage` ArtifactRef here.
    let profile_deps: Vec<runid::inputs::ArtifactRef> = Vec::new();
    let store = runid::FsCasStore::new(&root);
    let ir_version_str = ir::IR_VERSION.trim().to_string();
    let stem_label = stem.clone().unwrap_or_else(|| "profile".to_string());

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
    // One overall bar over all (point × seed × start) jobs, ticked from the
    // parallel loop (`Task` is `Send + Sync`). The Reporter honors --progress
    // (Pretty=bar, Plain=throttled `profile pos/len` log lines, None=silent).
    // No per-tick metric: a best profile-loglik isn't tracked here (each cell
    // computes its own `final_loglik`; surfacing a global best would mean a
    // shared accumulator that the bar deliberately avoids).
    let bar = crate::progress::Reporter::new().task(total_jobs as u64, "profile", "jobs");

    // ── gh#147 (M3.3): pre-resolve every job's CAS identity ──────────
    // Job tuple (seed_idx, grid_idx, start_idx). The grid lives in the
    // `point` level + the method in `stage`; (seed, point, start) pins the
    // job's RNG deterministically (job_seed below).
    let jobs: Vec<(usize, usize, usize)> = (0..seeds.len())
        .flat_map(|seed_idx| (0..grid_points.len())
            .flat_map(move |gi| (0..n_starts).map(move |si| (seed_idx, gi, si))))
        .collect();

    let resolve_pt = |gi: usize, si: usize, seed: u64|
        -> Result<crate::profile_cas::ResolvedProfilePoint, String>
    {
        let focal: Vec<(String, f64)> = focal_grids.iter()
            .zip(grid_points[gi].iter())
            .map(|(fg, &(_, v))| (fg.name.clone(), v))
            .collect();
        crate::profile_cas::resolve_profile_point(&crate::profile_cas::ProfilePointCtx {
            model: &model,
            ir_version: &ir_version_str,
            engine_version: crate::version::VERSION_SHORT,
            stem: &stem_label,
            method_name: &algorithm,
            data: &data_hashes,
            base_config: &base_config_blob,
            method_config: &method_config_blob,
            focal: &focal,
            seed,
            start_index: si as u32,
            deps: profile_deps.clone(),
        })
    };

    // (seed_idx, gi, si, resolved, cas_path) per job, resolved sequentially.
    let resolved_jobs: Vec<(usize, usize, usize,
        crate::profile_cas::ResolvedProfilePoint, std::path::PathBuf)> =
        jobs.iter().map(|&(seed_idx, gi, si)| {
            let resolved = resolve_pt(gi, si, seeds[seed_idx]).unwrap_or_else(|e| {
                eprintln!("error: profile-point identity: {}", e);
                std::process::exit(1);
            });
            let cas_path = runid::store_path(
                &root, runid::ArtifactKind::ProfilePoint, &resolved.levels);
            (seed_idx, gi, si, resolved, cas_path)
        }).collect();

    // The profile-base segment is shared across all jobs — a path segment
    // with no base-level record (guardrail 2). Write the provenance sidecar
    // there once (guardrail 4: resolved_priors/estimated/data_hashes carried
    // with no silent default; not identity-bearing).
    if let Some((_, _, _, resolved0, _)) = resolved_jobs.first() {
        let base_seg = runid::store_path(
            &root, runid::ArtifactKind::ProfilePoint, &resolved0.levels[..1]);
        let sidecar = crate::run_meta::FitSidecar {
            label: label_arg.clone(),
            model_path: ir_path.clone(),
            model_hash: model_hash.clone(),
            fit_toml_path: a.fit.as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
            fit_toml_hash: fit_toml_hash.clone().unwrap_or_default(),
            data_hashes: data_hashes.iter().cloned().collect(),
            estimated: resolved_priors_kv.iter()
                .map(|(n, _)| n.clone()).collect(),
            resolved_priors: resolved_priors_kv.iter()
                .map(|(n, s)| crate::run_meta::ResolvedPriorEntry {
                    param: n.clone(), source: s.clone() })
                .collect(),
            ..Default::default()
        };
        // Archive the producing `fit.toml` (best-effort; `write_fit_sidecar`
        // skips when the path isn't a file, i.e. a CLI-only profile).
        let archive_src = a.fit.clone().unwrap_or_else(|| std::path::PathBuf::from(""));
        if let Err(e) = crate::run_meta::write_fit_sidecar(
            &base_seg, &archive_src, &sidecar)
        {
            eprintln!("warning: cannot write profile sidecar {}: {}",
                base_seg.display(), e);
        }
    }

    // Cache scan: a job is cached when its leaf already exists. `cached` /
    // `remaining` hold indices into `resolved_jobs`.
    let mut cached: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = Vec::new();
    for (ji, (_, _, _, resolved, cas_path)) in resolved_jobs.iter().enumerate() {
        if matches!(
            store.lookup(cas_path, &runid::LeafIdentity::new(resolved.run_id)),
            runid::Lookup::Hit(_)
        ) {
            cached.push(ji);
        } else {
            remaining.push(ji);
        }
    }
    if !cached.is_empty() {
        eprintln!("profile: {} of {} starts already cached — resuming",
            cached.len(), total_jobs);
        bar.inc(cached.len() as u64);
    }

    // gh#audit-H13: --parallel / CAMDL_PARALLEL throttles the rayon thread
    // budget. Build a SCOPED local pool and run the parallel job sweep inside
    // `pool.install(...)`. The earlier fix used `build_global`, but by the
    // time profile reaches here the global pool is already initialised, so
    // `build_global` returns AlreadyInitialized and is ignored — the default
    // all-core pool ran regardless of --parallel. A scoped pool is
    // order-independent. parallel == 0 means "use rayon's default" (all
    // logical cores): leave the pool unset and run on the global pool.
    let prof_pool: Option<rayon::ThreadPool> = if parallel > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build()
                .unwrap_or_else(|e| {
                    eprintln!("error: failed to build thread pool (--parallel {}): {}", parallel, e);
                    std::process::exit(1);
                }),
        )
    } else {
        None
    };

    // Throttled rollup rewrites: per-seed profile.tsv (1s throttle) and
    // (multi-seed only) the cross-seed summary.tsv (2s throttle, since
    // it reads N seeds' rollups). Last-completion-wins.

    // ── Run remaining jobs in parallel ──────────────────────────────
    let run_sweep = || {
    remaining.par_iter().for_each(|&ji| {
        let (seed_idx, grid_idx, start_idx, ref resolved_pt_id, ref cas_path) = resolved_jobs[ji];
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
                    max_substeps: a.inference.pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET), // gh#241
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
                            max_substeps: a.inference.pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET), // gh#241
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
                    burn_in: PROFILE_PMMH_BURN_IN,
                    rho: pmmh_rho_opt,
                    n_source_groups: compiled.source_groups.len(),
                };

                // PF process kernel + obs model for this cell. PMMH on
                // profile is chain_binomial-only (rejected upstream for
                // --backend ode), so wire ChainBinomialProcess directly.
                let pf_process = ChainBinomialProcess::new(compiled.clone());
                let pf_obs_model = Arc::clone(&obs_model_obj);
                let smc_cfg = sim::inference::traits::SMCConfig {
                    n_particles: pmmh_config.n_particles,
                    dt: pmmh_config.dt,
                    t_start: compiled.model.simulation.t_start,
                    skip_first_obs_from_loglik: false,
                    record_ancestry: false,
                    record_prequential: false,
                    max_substeps: a.inference.pf_max_substeps.unwrap_or(sim::inference::degeneracy::ITER_BUDGET), // gh#241
                };

                // gh#224: structural failures surface; a degenerate/recoverable
                // PF run is a ruled-out θ (−∞).
                let eval_loglik = |theta: &[f64], pf_seed: u64| -> Result<f64, sim::error::SimError> {
                    match sim::inference::bootstrap_filter(
                        &pf_process, &*pf_obs_model, theta, &smc_cfg, pf_seed,
                    ) {
                        Ok(r) => Ok(r.log_likelihood),
                        Err(e) if e.is_structural() => Err(e),
                        Err(_) => Ok(f64::NEG_INFINITY),
                    }
                };

                // Correlated-PF evaluator (only used when rho is set).
                // Mirrors fit/pmmh.rs's eval_correlated.
                let eval_correlated: Option<Box<dyn Fn(
                    &[f64],
                    &sim::inference::correlated_pf::PFRandomState,
                ) -> Result<f64, sim::error::SimError>>> = if pmmh_config.rho.is_some() {
                    let pf_process2 = ChainBinomialProcess::new(compiled.clone());
                    let pf_obs_model2 = Arc::clone(&obs_model_obj);
                    let smc_cfg2 = smc_cfg.clone();
                    let cell_seed = job_seed;
                    Some(Box::new(move |theta: &[f64], randoms| -> Result<f64, sim::error::SimError> {
                        match sim::inference::correlated_pf::bootstrap_filter_correlated(
                            &pf_process2, &*pf_obs_model2, theta,
                            &smc_cfg2, randoms, cell_seed,
                        ) {
                            Ok(r) => Ok(r.log_likelihood),
                            Err(e) if e.is_structural() => Err(e),
                            Err(_) => Ok(f64::NEG_INFINITY),
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
                // gh#224: a ruled-out θ is rejected internally as −∞; an `Err`
                // is a structural failure (model/config can't run). Every cell
                // would hit it identically, so abort the whole profile run
                // rather than silently emitting degenerate per-cell MLEs.
                let result = match run_pmmh(
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
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("error: profile PMMH cell (grid {}, start {}) \
                            failed with structural error: {}", grid_idx, start_idx, e);
                        std::process::exit(1);
                    }
                };
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
                // gh#97: report the loglik that belongs to the saved
                // MLE params. `mle_params = result.map_params` (below),
                // and `pmmh.rs:478-483` sets `map_loglik` in lockstep
                // with `map_params` — they are coherent by construction.
                // The previous `result.map_loglik.max(best_ll)` reported
                // the per-sample max loglik, which under any non-flat
                // prior comes from a DIFFERENT θ than the MAP (the
                // loglik-maximizing step is not the posterior-maximizing
                // step). That paired a loglik from one θ with params
                // from another — the same bug class f52d1ecd fixed on
                // the IF2 path. If `map_loglik` is non-finite (the
                // initial PF eval never reached a finite loglik), report
                // it honestly; the renderer's `is_finite()` guard writes
                // `-inf` rather than masking it.
                let final_ll = result.map_loglik;
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

        // gh#147 (M3.3): claim the CAS leaf, write the cell's mle.toml there,
        // finalize. The profile-base is never written (path segment only);
        // each (point, stage, seed, start) is its own leaf. The display
        // payload (method/algorithm/loglik) is recorded in `inputs`, not
        // hashed.
        let algorithm_payload = match profile_algo {
            ProfileAlgo::If2 => serde_json::json!({
                "particles": n_particles, "iterations": n_iterations,
                "cooling": cooling, "dt": dt }),
            ProfileAlgo::Pmmh => serde_json::json!({
                "steps": pmmh_steps, "particles": pmmh_particles,
                "rho": pmmh_rho_opt, "dt": dt }),
            ProfileAlgo::Nlopt(_) => serde_json::json!({
                "tolerance": 1e-4, "max_evals": 1500,
                "dt": compiled.model.simulation.dt.unwrap_or(dt) }),
        };
        let inputs_json = serde_json::json!({
            "method": profile_algo.method_kind().as_str(),
            "seed": job_seed,
            "n_chains": 1,
            "algorithm": algorithm_payload,
            "grid_point": grid_idx,
            "start": start_idx,
            "best_loglik": if final_loglik.is_finite() { Some(final_loglik) } else { None },
            "wall_time_seconds": elapsed,
            "provenance": run_provenance_json.clone(),
        });
        let running = crate::profile_cas::build_profile_point_record(
            resolved_pt_id, &profile_deps, &ir_version_str,
            runid::RunStatus::Running, serde_json::Value::Null, &ir_path);
        let claim = match store.claim_streaming(cas_path, running) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("warning: claim profile point {}: {}", cas_path.display(), e);
                return;
            }
        };
        let start_dir = claim.dir().to_path_buf();

        let mle_toml = render_mle_toml(&if2_params, &focal_values,
            &focal_grids.iter().map(|fg| fg.name.as_str()).collect::<Vec<_>>(),
            &mle_params, final_loglik, final_log_posterior, &diag);
        let _ = std::fs::write(start_dir.join("mle.toml"), mle_toml);

        let completed = crate::profile_cas::build_profile_point_record(
            resolved_pt_id, &profile_deps, &ir_version_str,
            runid::RunStatus::Completed, inputs_json, &ir_path);
        if let Err(e) = claim.finalize(completed) {
            eprintln!("warning: finalize profile point {}: {}", cas_path.display(), e);
        }

        // Passive progress tick. `Task` handles Pretty (redraw) / Plain
        // (throttled `profile pos/len` log line) / None (no-op) internally.
        bar.inc(1);

    });
    };
    match &prof_pool {
        Some(pool) => pool.install(run_sweep),
        None => run_sweep(),
    }

    bar.finish();

    // gh#147 (M3.3): the per-seed profile.tsv curve + the cross-seed
    // summary.tsv are cross-point / cross-seed aggregates with no home in the
    // base/point/stage/seed/start tree — they are M4-derived views over the
    // per-cell leaves (same as the grid summary). gh#154 restores them via
    // `reindex`. Until then a profile run writes per-cell leaves but no curve.
    let n_cells = resolved_jobs.len();
    eprintln!(
        "profile: {} cell{} written under {}; the profile curve / summary is \
         a derived view (gh#154 / reindex) and lands in M4.",
        n_cells, if n_cells == 1 { "" } else { "s" },
        root.join("profiles").display());
    if let Some(ref path) = output_tsv_path {
        eprintln!(
            "note: --output {} not written — the profile curve is an M4 derived \
             view (gh#154); the per-cell leaves carry each point's loglik.",
            path);
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
            p.value = p.value.with_value(v);
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

#[cfg(test)]
mod tests {
    use super::*;
    use sim::inference::if2::EstimatedParam;
    use sim::inference::types::Transform;

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

    // ── gh#102 (H10): profile-PMMH steps-vs-burn-in guard ─────────────

    #[test]
    fn profile_pmmh_steps_below_burn_in_rejected() {
        // steps < burn_in: every sample is burned → empty diagnostics.
        let err = validate_profile_pmmh_steps(50, PROFILE_PMMH_BURN_IN)
            .expect_err("steps below burn-in must be rejected");
        assert!(err.contains("50") && err.contains("100"),
            "error must report the offending steps and the burn-in: {}", err);
        assert!(err.contains("burn-in") && err.contains("--pmmh-steps"),
            "error must name the cause and the flag to fix: {}", err);
        // Suggested fix is burn_in + 1.
        assert!(err.contains("101"),
            "error must suggest a working steps value (burn_in + 1): {}", err);
    }

    #[test]
    fn profile_pmmh_steps_equal_burn_in_rejected() {
        // Boundary: steps == burn_in is still degenerate (0 post-burn-in).
        assert!(validate_profile_pmmh_steps(PROFILE_PMMH_BURN_IN, PROFILE_PMMH_BURN_IN)
            .is_err(),
            "steps == burn-in leaves zero post-burn-in samples — must reject");
    }

    #[test]
    fn profile_pmmh_steps_above_burn_in_ok() {
        // Control: steps > burn_in leaves a non-empty post-burn-in chain.
        assert!(validate_profile_pmmh_steps(PROFILE_PMMH_BURN_IN + 1, PROFILE_PMMH_BURN_IN)
            .is_ok(),
            "steps just above burn-in must pass");
        assert!(validate_profile_pmmh_steps(5000, PROFILE_PMMH_BURN_IN).is_ok(),
            "a realistic chain length must pass");
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
            // `value` is the typed ParamValue ADT (gh#191): a concrete number
            // is Fixed, absent is Required.
            let body = match value {
                Some(v) => format!(
                    r#"{{"name":"{}","value":{{"mode":"fixed","value":{}}}}}"#, name, v),
                None => format!(
                    r#"{{"name":"{}","value":{{"mode":"required"}}}}"#, name),
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
            .find(|p| p.name == "beta").unwrap().value.resolved_value();
        let gamma_val = params.iter()
            .find(|p| p.name == "gamma").unwrap().value.resolved_value();
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
            .find(|p| p.name == "beta").unwrap().value.resolved_value();
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
            .find(|p| p.name == "beta").unwrap().value.resolved_value();
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
            .find(|p| p.name == "beta").unwrap().value.resolved_value();
        let init_prior = crate::fit::init::InitMethod::FromPrior;
        seed_params_from_init_method(&mut params, &init_prior).unwrap();
        assert_eq!(params.iter()
            .find(|p| p.name == "beta").unwrap().value.resolved_value(), original_beta,
            "from_prior must NOT seed model_pre — values are per-chain");

        let init_lhs = crate::fit::init::InitMethod::Lhs;
        seed_params_from_init_method(&mut params, &init_lhs).unwrap();
        assert_eq!(params.iter()
            .find(|p| p.name == "beta").unwrap().value.resolved_value(), original_beta,
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

}

