//! `camdl profile` — profile likelihood via parallel per-cell optimization
//! (IF2 or deterministic nl-* MLE, per `--algorithm`).
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

use sim::{
    compiled_model::CompiledModel,
    inference::{
        if2::{run_if2, IF2Config},
        particle_filter::Observation,
        BoundObs, ChainBinomialProcess, MultiStreamObsModel,
    },
};
use std::collections::HashMap;
use std::sync::Arc;

use crate::cas::typed::ContentHash;
use crate::run_paths::output_root;

/// Why `camdl profile` has no per-cell PMMH, and what to run instead.
///
/// A profile is the maximum over the nuisance parameters λ of the
/// likelihood L(ψ, λ) at each grid value of the focal parameter ψ, so each
/// cell is an optimisation. A per-cell MCMC chain with ψ pinned samples
/// p(λ | ψ, y) correctly, but the number it reported — the log-likelihood
/// at the highest-posterior sample the chain visited — is neither that
/// maximum (under an informative prior the MAP sample sits below it, by an
/// amount that depends on how well a short chain explored) nor the marginal
/// posterior p(ψ | y), which needs each cell's marginal likelihood
/// p(y | ψ). Named once so the `--algorithm pmmh` refusal and the removed
/// `--pmmh-*` flags give the same answer.
const PROFILE_PMMH_REMOVED: &str = "per-cell PMMH was removed from `camdl profile`. \
    A profile likelihood is a maximum over the other parameters at each grid \
    value, so each cell is fitted by maximisation: use `--algorithm if2` \
    (chain_binomial) or an `nl-*` optimizer (`nl-sbplx`, `nl-bobyqa`, \
    `nl-lbfgs`; `--backend ode`). The per-cell chain reported the \
    log-likelihood at its highest-posterior sample, which is neither the \
    profile likelihood nor the marginal posterior of the swept parameter. For \
    a posterior of that parameter, fit once with `camdl fit run` (PMMH or \
    PGAS) and read its marginal from the draws.";

/// Per-cell optimizer choice. `--algorithm if2 --backend chain_binomial`
/// (the default) keeps the historical PF-based per-cell MLE; the new
/// `--algorithm nl-* --backend ode` paths run NLopt deterministic MLE
/// against `compute_ode_loglik`. See gh#47 for the dispatch rationale
/// and the proposal for the (algorithm, backend) matrix.
#[derive(Debug, Clone, Copy)]
enum ProfileAlgo {
    If2,
    Nlopt(sim::inference::deterministic::NloptAlgorithm),
}

impl ProfileAlgo {
    fn method_kind(self) -> crate::run_meta::FitAlgorithm {
        match self {
            ProfileAlgo::If2  => crate::run_meta::FitAlgorithm::If2,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx) =>
                crate::run_meta::FitAlgorithm::NlSbplx,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa) =>
                crate::run_meta::FitAlgorithm::NlBobyqa,
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs) =>
                crate::run_meta::FitAlgorithm::NlLbfgs,
        }
    }
}

// Observation family resolution lives in `crate::util::resolve_data_specs`
// (gh#90). Profile's previous `resolve_obs_family` (gh#38) is subsumed
// there: `--data PATH --obs <family-root>` matches every IR obs whose
// name starts with `<root>_`, and `--data NAME=PATH` (gh#90 named form)
// also expands NAME as a family root. Same semantics, single dispatch.

/// Normalized coordinates of grid cell `gi` — each focal value mapped to
/// `[0, 1]` over its axis's grid range, so the nearest-to-best ordering is
/// scale-fair across axes with very different magnitudes. Ordering only; it
/// never touches a loglik or a leaf.
fn profile_cell_norm(
    grid_points: &[Vec<(usize, f64)>],
    axis_ranges: &[(f64, f64)],
    gi: usize,
) -> Vec<f64> {
    grid_points[gi].iter().enumerate().map(|(k, &(_, v))| {
        let (mn, mx) = axis_ranges[k];
        if (mx - mn).abs() < 1e-12 { 0.0 } else { (v - mn) / (mx - mn) }
    }).collect()
}

/// Squared Euclidean distance between two normalized cell coordinate vectors
/// (see [`profile_cell_norm`]). Squared — ordering only needs the monotone
/// comparison.
fn profile_cell_dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// What to pass when `profile` has no observation data bound. Named once so the
/// anchored refusal and the ordinary missing-data refusal cannot drift apart.
const PROFILE_SUPPLY: &str = "pass `--data PATH` (single-stream), \
    `--data NAME=PATH` (repeatable, multi-stream), or `--fit FOO.toml` with a \
    [data.observations] section.";

/// The `base` level's identity: the inference PROBLEM a profile is computed
/// over — which model, data, fixed values, priors and scoring window.
///
/// A struct rather than a `json!` literal so the level is include-by-default:
/// add a field here and it is hashed. The literal this replaced was
/// exclude-by-default, and a scoring-window setting once fell through it —
/// present in the computation, absent from the key, so a rerun under a
/// different window was served the previous landscape.
#[derive(serde::Serialize)]
struct ProfileBaseLevel<'a> {
    base_params: &'a str,
    fixed: &'a [String],
    obs_family: &'a str,
    fit_toml: &'a Option<String>,
    priors: &'a [(String, String)],
    /// The likelihood every cell scores. It changes each stored loglik, so it
    /// is identity. Omitted when full, so a profile that never set `ic_free`
    /// keeps its key.
    #[serde(skip_serializing_if = "crate::fit::config_v2::LikelihoodObjective::is_full")]
    objective: crate::fit::config_v2::LikelihoodObjective,
}

/// The `stage` level's identity: the METHOD, i.e. how each cell is fitted.
#[derive(serde::Serialize)]
struct ProfileMethodLevel<'a> {
    algorithm: &'a str,
    if2: ProfileIf2Knobs,
    /// Resolved per-parameter perturbation magnitudes, sorted by name.
    rw_sd: &'a [(&'a str, Option<f64>)],
    /// `--rw-sd auto` derives magnitudes from the model rather than the flag,
    /// so it is its own discriminator rather than a value.
    rw_sd_auto: bool,
    init: &'a crate::fit::starts::ChainStarts,
    pf_max_substeps: Option<u64>,
    /// The `[[simplex_groups]]` IF2 perturbs jointly, member names in order.
    /// Omitted when empty, so a profile without groups keeps its key.
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    simplex_groups: &'a [Vec<String>],
}

#[derive(serde::Serialize)]
struct ProfileIf2Knobs {
    particles: usize,
    iterations: usize,
    cooling: f64,
    dt: f64,
    starts: usize,
}

pub fn cmd_profile(a: &crate::args::ProfileArgs) {
    // The `--pmmh-*` flags went with per-cell PMMH. They are trapped so a
    // stale script is answered with the reason rather than clap's
    // "unexpected argument", before any file is read.
    for (flag, given) in [
        ("--pmmh-steps", a._removed_pmmh_steps.as_deref()),
        ("--pmmh-particles", a._removed_pmmh_particles.as_deref()),
        ("--pmmh-rho", a._removed_pmmh_rho.as_deref()),
    ] {
        if let Some(raw) = given {
            eprintln!("error: {flag} {raw} is no longer accepted: {PROFILE_PMMH_REMOVED}");
            std::process::exit(1);
        }
    }
    // Parse the CLI strings into the typed registry entry (the string boundary);
    // all downstream dispatch reads the typed FitAlgorithm / InferenceBackend.
    let algo_name = a.algorithm.as_deref().unwrap_or("if2");
    let backend_name = a.backend.as_deref().unwrap_or("chain_binomial");
    // Refused on any backend, before the registry's own (algorithm, backend)
    // check, so `--algorithm pmmh --backend ode` gets the same answer.
    if crate::fit::methods::parse_algorithm(algo_name)
        == Some(crate::run_meta::FitAlgorithm::Pmmh)
    {
        eprintln!("error: --algorithm pmmh is not accepted: {PROFILE_PMMH_REMOVED}");
        std::process::exit(1);
    }
    let method = match crate::fit::methods::parse_combo(algo_name, backend_name) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("error: {}", msg);
            std::process::exit(1);
        }
    };
    let profile_algo = match method.algorithm {
        crate::run_meta::FitAlgorithm::If2      => ProfileAlgo::If2,
        crate::run_meta::FitAlgorithm::NlSbplx  =>
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx),
        crate::run_meta::FitAlgorithm::NlBobyqa =>
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa),
        // gh#275: the same per-cell `optimize_cell` the derivative-free NLopt
        // algorithms use, taking the gradient branch. `profile` reaches it
        // without further wiring because it loads the model through
        // `util::load_model`, which always compiles the full IR — the WrtPop
        // state-Jacobian the forward sensitivity reads is present, unlike the
        // `fit run` path, which compiles lean unless the method asks for it.
        crate::run_meta::FitAlgorithm::NlLbfgs =>
            ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs),
        other => {
            eprintln!(
                "error: --algorithm = \"{}\" is not yet supported for `camdl profile`. \
                 Currently supported: if2 (chain_binomial), \
                 nl-sbplx (ode), nl-bobyqa (ode), nl-lbfgs (ode).",
                other
            );
            std::process::exit(1);
        }
    };
    // Registry-driven caveat for methods carrying a status_note (Beta methods,
    // plus Stable methods with usage guidance — e.g. nl-*). After the profile
    // dispatch, so a refused algorithm is not first shown a caveat.
    crate::fit::methods::emit_status_banner(method.algorithm, method.backend);

    let ir_path = a.model.to_string_lossy().into_owned();
    let n_particles = a.inference.particles;
    let n_iterations = a.iterations;
    let n_starts = a.n_starts;
    let cooling = a.cooling;
    let dt = a.inference.dt;
    let seed_base = a.inference.seed;
    let parallel = a.inference.parallel;
    let output_tsv_path: Option<String> = a.output.as_ref().map(|p| p.to_string_lossy().into_owned());
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
    // gh#889: the `--init` family, answered with the same `--starts`
    // replacements `fit run` gives, from the same builder.
    if let Some(msg) = crate::fit::removed_flags_message(
        "profile",
        crate::fit::starts_family_removed_lines(
            a._removed_init.as_deref(),
            a._removed_posterior.as_deref(),
            a._removed_mle.as_deref(),
            a._removed_params.as_deref(),
        ),
    ) {
        eprintln!("error: {msg}");
        std::process::exit(2);
    }
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
    // Where each cell's chains begin, as `--starts <spec>` states it — the
    // same rule grammar and the same typed `ChainStarts` `fit run` takes
    // (gh#889). How many chains is `--n-starts`, a count.
    let init_method: crate::fit::starts::ChainStarts = a.starts.clone();

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
    // The model before any value is written into it, for the partition.
    let model_declared = model_pre.clone();

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
    let (fit_estimate, fit_toml_hash, fit_toml_fixed_indexmap, fit_scenario, fit_simplex_groups,
         objective, fit_fixed_via):
        (indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
         Option<String>,
         indexmap::IndexMap<String, f64>,
         (Option<String>, Vec<String>, Vec<String>),
         Vec<crate::fit::config_v2::SimplexGroup>,
         crate::fit::config_v2::LikelihoodObjective,
         Result<Vec<(String, crate::roles::FitFixedVia)>, String>)
        = if let Some(fit_path) = a.fit.as_ref() {
        let fit_path_str = fit_path.to_string_lossy().into_owned();
        let fit_cfg = crate::fit::config_v2::Problem::load(&fit_path_str)
            .unwrap_or_else(|e| {
                eprintln!("error: failed to load --fit toml '{}': {}",
                    fit_path_str, e);
                std::process::exit(1);
            });
        // How each [fixed] name reached the config, for the partition.
        let fixed_via = fit_cfg.fixed.declarations(&model_pre);
        // Resolve [fixed] (file load, scenario, inline overlay) the
        // same way `camdl survey --fit` and `camdl fit run` do.
        let mut fixed_cfg = fit_cfg.fixed.clone();
        fixed_cfg.expand_from_scenario(&model_pre, &fit_cfg.estimate)
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
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
        // The likelihood every cell scores is the fit's own (`ic_free`), read
        // through the same function `fit run` uses, so a profile curve is on
        // the scale of the fit it profiles.
        let objective = fit_cfg.likelihood_objective();
        (fit_cfg.estimate, Some(hash), fixed_resolved,
         (fit_cfg.scenario, fit_cfg.enable, fit_cfg.disable),
         fit_cfg.simplex_groups, objective, fixed_via)
    } else {
        (indexmap::IndexMap::new(), None, indexmap::IndexMap::new(),
         (None, Vec::new(), Vec::new()), Vec::new(),
         crate::fit::config_v2::LikelihoodObjective::Full, Ok(Vec::new()))
    };

    // The scenario this profile runs under: the fit config's top-level
    // `scenario`/`enable`/`disable` combined with the CLI flags by the same
    // rule `survey --fit` uses.
    let crate::util::ScenarioSelection {
        scenario: scenario_name,
        enable: adhoc_enable,
        disable: adhoc_disable,
    } = crate::util::combine_fit_scenario(
        crate::util::ScenarioChoice {
            scenario: a.scenario.scenario.as_deref(),
            enable: &a.scenario.enable,
            disable: &a.scenario.disable,
        },
        crate::util::ScenarioChoice {
            scenario: fit_scenario.0.as_deref(),
            enable: &fit_scenario.1,
            disable: &fit_scenario.2,
        },
    ).unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); });
    // The rule `fit run` applies to the same file: a scenario (from the fit
    // config or `--scenario`) that moves a [fixed] parameter to another value
    // is refused, not resolved.
    if let (Some(scenario), Some(fit_path)) = (scenario_name.as_deref(), a.fit.as_ref()) {
        match crate::params_resolver::scenario_fixed_collision(
            &model_pre, scenario, &fit_toml_fixed_indexmap, &fit_path.to_string_lossy(),
        ) {
            Ok(None) => {}
            Ok(Some(msg)) | Err(msg) => {
                eprintln!("error: {msg}");
                std::process::exit(1);
            }
        }
    }

    // A single-point starts rule (`--starts from_params=<toml>`,
    // `--starts from_mle=<fit>`) loads one parameter point. Its values are
    // the base value of every parameter they name — so a parameter the
    // profile neither fixes nor estimates is HELD at the file's value — and
    // the chain-start draw below reads the same rule again to start the
    // estimated (nuisance) parameters there. They enter the resolver as
    // their own tier, just above the model default
    // (`ValueSource::Starts`), so a later tier replacing one is reported and
    // recorded. Without them the resolver would fire `UnsetRequired` on a
    // parameter with no DSL default even though the user named a file
    // carrying it. See `starts_values_from_init_method`.
    let starts_values: Option<(crate::params_resolver::StartsOrigin, Vec<(String, f64)>)> =
        starts_values_from_init_method(&init_method)
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
    let mut seeded: HashMap<String, crate::params_resolver::ValueSource> = HashMap::new();
    for (name, spec) in &fit_estimate {
        if let Some(p) = model_pre.parameters.iter_mut().find(|p| p.name == *name) {
            if p.value.resolved_value().is_none() {
                if let Some(start) = spec.start {
                    p.value = p.value.with_value(start);
                    seeded.insert(name.clone(),
                        crate::params_resolver::ValueSource::EstimateStart { drawn: false });
                }
            }
        }
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
    //
    // This sits BEFORE the resolver and the compile because the observation
    // anchors below fold over exactly these bindings (gh#616), and
    // `CompiledModel::new` refuses a model that still carries one. Binding is a
    // pure read of the args plus `model.observations`, which neither the
    // `[estimate].start` seeding above nor the resolver below touches.
    //
    // `None` means nothing was bound at all (no `--data`, no `--fit`). That is
    // reported LATER, where it always was — after the compile — so the
    // capability gate keeps first say on an invocation that has a more specific
    // problem than a missing flag.
    let obs_name_arg = a.stream.obs.clone();
    let model_obs_names: Vec<String> = model_pre.observations.iter()
        .map(|o| o.name.clone()).collect();
    let cli_data_specs: Vec<crate::args::types::DataSpec> = a.data.clone();
    let bound_streams: Option<Vec<(String, std::path::PathBuf)>> = if cli_data_specs.is_empty() {
        a.fit.as_ref().map(|fit_path| {
            eprintln!("profile: no --data flags supplied, reading bindings \
                from --fit toml [data.observations]");
            crate::pfilter::load_data_observations_from_fit_toml(
                fit_path.as_path(), &model_obs_names,
            ).unwrap_or_else(|e| {
                eprintln!("error: --fit toml fallback for --data: {}", e);
                std::process::exit(1);
            })
        })
    } else {
        if a.fit.is_some() {
            eprintln!("profile: --data on CLI overrides --fit toml [data.observations]");
        }
        Some(crate::util::resolve_data_specs(&cli_data_specs, &model_obs_names, obs_name_arg.as_deref())
            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); }))
    };

    // gh#616: profile binds observation data, so it RESOLVES a model's
    // observation anchors instead of letting `CompiledModel::new` refuse them.
    // The window is folded from the bindings just resolved, so it is the same
    // window every profile point scores.
    //
    // Everything downstream reads the substituted model: `model_pre` is the
    // resolver's input and `resolved.model` is what compiles, and this path has
    // exactly one `load_model` and one `CompiledModel::new`. A second fresh load
    // of the compiled IR would re-introduce the unresolved marker (gh#616
    // regression, commit 7af5c9fa).
    //
    // With nothing bound, an ANCHORED model is refused here rather than
    // deferred: it has no horizon at all, so nothing later can be computed from
    // it. An unanchored model falls through untouched.
    let moved_any_anchor = match &bound_streams {
        Some(bound) => crate::obs_anchor::resolve_from_bindings(&mut model_pre, bound, dt)
            .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); }),
        None => {
            if let Some(msg) =
                crate::obs_anchor::refuse_without_data(&model_pre, PROFILE_SUPPLY)
            {
                eprintln!("error: {msg}");
                std::process::exit(1);
            }
            false
        }
    };
    // The run is content-addressed by the IR TEXT (`model_identity_from_ir`),
    // not by `model_pre`, so the substitution has to reach the text too — else
    // two data vintages that fork a forcing differently would share a
    // `model_identity`. Re-emitted only when something actually moved, so an
    // unanchored model's bytes are untouched.
    //
    // The re-emitted text carries the starts rule's values, as it did when
    // they were written into `model_pre` before the resolver: they now enter
    // the resolver as their own tier instead, and are seeded into this copy
    // only so the model identity of an anchored profile does not move.
    let model_json = if moved_any_anchor {
        let mut ident_model = model_pre.clone();
        if let Some((_, values)) = &starts_values {
            for p in &mut ident_model.parameters {
                if let Some((_, v)) = values.iter().find(|(n, _)| *n == p.name) {
                    p.value = p.value.with_value(*v);
                }
            }
        }
        ir::to_string_pretty(&ident_model).unwrap_or_else(|e| {
            eprintln!("error: re-emitting the anchor-resolved IR: {e}");
            std::process::exit(1);
        })
    } else {
        model_json
    };

    // gh#561: every profile point runs the filter, whose window is the
    // observation times — the model horizon is never read here. A scenario's
    // own `simulate { to }` therefore moves nothing, so refuse it rather than
    // discard it silently (the same guard as `pfilter` and `fit predict`).
    //
    // It runs AFTER the substitution above so it compares two RESOLVED numbers:
    // an anchored model and an anchored scenario that resolve to the same time
    // are a no-op and must pass, and two that differ must be named with times a
    // reader can check against the data.
    if let Err(e) = crate::util::refuse_scenario_horizon(
        &model_pre, scenario_name.as_deref(), "profile",
        "each profile point scores through a particle filter at the observation \
         times, so the window comes from the data, not the model horizon",
    ) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    // Run the unified resolver.
    let fit_toml_estimate: indexmap::IndexSet<String> =
        fit_estimate.keys().cloned().collect();
    let table_files_resolver: HashMap<String, std::path::PathBuf> = HashMap::new();
    let resolved = crate::params_resolver::resolve_parameters_with_starts(
        crate::params_resolver::ParameterInputs {
            model: &model_pre,
            scenario: scenario_name.as_deref(),
            adhoc_enable: &adhoc_enable,
            adhoc_disable: &adhoc_disable,
            scenario_inline_name: None,
            scenario_inline_set: &[],
            scenario_inline_scale: &[],
            point_overrides: &[],
            fixed_cli: &fixed_cli_vec,
            fixed_files: &fixed_files_vec,
            fit_toml_fixed: &fit_toml_fixed_indexmap,
            fit_toml_estimate: &fit_toml_estimate,
            table_files: &table_files_resolver,
        },
        starts_values.as_ref().map(|(origin, values)|
            crate::params_resolver::StartsValues { origin, values }),
    ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    let mut resolved = resolved;
    resolved.attribute_seeded_starts(&seeded);
    crate::params_resolver::emit_notices(&[&resolved]);
    // The grid writes each focal value straight over the parameter, so a
    // scenario's value for a focal parameter is replaced at every point: said
    // once here, for the run, not per grid point.
    let focal_notices: crate::notice::Notices = match scenario_name.as_deref() {
        Some(scenario) => {
            let focal: Vec<&str> = focal_names.iter().map(|s| s.as_str()).collect();
            crate::params_resolver::scenario_focal_notices(&model_pre, scenario, &focal)
                .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); })
                .into_iter()
                .collect()
        }
        None => crate::notice::Notices::default(),
    };
    focal_notices.emit_stderr();
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
        method.backend, &compiled,
    ) {
        eprintln!("error: {}", msg);
        std::process::exit(1);
    }

    // A conditional objective (`ic_free = true`) is admitted per method by the
    // check `fit run` applies: IF2 drops y₁'s term, the deterministic ODE
    // likelihood cannot, so an `nl-*` profile of such a fit is refused rather
    // than run on the full likelihood under the fit's name.
    if objective.skips_first_observation() {
        let init_law = if compiled.has_init_law {
            crate::fit::methods::InitLaw::Declared
        } else {
            crate::fit::methods::InitLaw::Absent
        };
        if let Err(msg) =
            crate::fit::methods::validate_ic_free(profile_algo.method_kind(), init_law)
        {
            eprintln!("error: the --fit config sets ic_free = true. {msg}");
            std::process::exit(1);
        }
        eprintln!("profile: ic_free = true — every cell scores log p(y₂:T | y₁, θ), \
            the fit's conditional likelihood");
    }

    // The missing-data refusal, reported here rather than where the binding was
    // resolved, so every gate above keeps first say (see the binding block).
    let bound_streams = bound_streams.unwrap_or_else(|| {
        eprintln!("error: profile needs observation data to score: {PROFILE_SUPPLY}");
        std::process::exit(1);
    });
    if bound_streams.is_empty() {
        eprintln!("error: zero streams resolved from --data / --fit toml.");
        std::process::exit(1);
    }

    // Resolve the bound observation streams (BY SOURCE) and load each one's
    // per-observation values + aux via the single shared seam that `fit run`
    // and `pfilter` also route through (`fit::runner::resolve_and_load_obs_streams`).
    // The seam filters `model.observations` by `source`, dispatches long-form vs
    // wide loading (holes + aux), resolves each projection, and runs the per-
    // stream origin / first-window guards — so a stratified family bound by its
    // ROOT (`[data.observations] prevalence = FILE`, source `prevalence`) fans
    // out to every leaf exactly as `fit run` binds it. `bound_streams` (the
    // CLI/toml key-space bindings) is mapped to the by-source `effective` map at
    // the boundary; it is retained below only for the CAS data-hash, whose
    // key-space (leaf name / family root) must not change. The typo guard (a
    // binding key matching no source or name) lives in the adapter.
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: compiled.model.simulation.t_start,
        format: a.inference.time_format,
    };
    let effective = crate::fit::runner::data_bindings_to_effective(&model, &bound_streams)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    let streams = crate::fit::runner::resolve_and_load_obs_streams(
        &model, &compiled, &effective, dt, &time_opts,
    ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    if streams.len() > 1 {
        eprintln!(
            "profile: {} streams bound (joint loglik = sum across all): {}",
            streams.len(),
            streams.iter().map(|s| s.name.as_str())
                .collect::<Vec<_>>().join(", "),
        );
    } else {
        eprintln!("profile: using observation model '{}' from IR",
            streams[0].name);
    }

    // gh#90: silent-wrong-answer warning. If the model declares N>1 observation
    // blocks but only M<N are bound, the unbound streams contribute zero to the
    // likelihood — the result looks plausible but is methodologically wrong. The
    // bound set is the RESOLVED leaf names (a family root fans out to its
    // leaves), not the raw binding keys — else an indexed family bound by its
    // root would look entirely unbound and false-warn.
    {
        let bound_names: Vec<String> = streams.iter()
            .map(|s| s.name.clone()).collect();
        if let Some(w) = crate::util::format_unbound_streams_warning(
            "profile", &model_obs_names, &bound_names,
        ) {
            eprint!("{}", w);
        }
    }

    // Canonical schedule: the UNION axis the bound observation model scores
    // along — every stream's scoring boundaries plus every declared period's
    // opening boundary inside the run (multi-cadence, proposal 2026-06-10
    // §3.3; periods, gh#833). Downstream code reads it for `obs_times` (the
    // substep grid + the ODE-MLE / PMMH consumers, which index the obs model
    // by position along it), so it is taken FROM `bind` — the same call that
    // builds the model below — never rebuilt from the rows' labels, which
    // agree with the boundaries only under `closing_at`.
    let observations: Vec<Observation> = {
        let specs = crate::fit::runner::stream_specs_from_obs_streams(&streams);
        let (bound, _report) = BoundObs::bind(compiled.model.simulation.t_start, specs)
            .unwrap_or_else(|report| {
                eprintln!("error: observation data invalid:\n{}", report.render());
                std::process::exit(1);
            });
        bound.times().iter().map(|&time| Observation { time, value: 0.0 }).collect()
    };
    let observations = Arc::new(observations);

    for sw in &a.sweep {
        let idx = compiled.param_index.get(sw.name.as_str()).copied()
            .unwrap_or_else(|| {
                eprintln!("focal parameter '{}' not found{}", sw.name,
                    crate::util::constant_hints(&compiled.model, [sw.name.as_str()]));
                std::process::exit(1);
            });
        let values = sw.grid.expand();
        // A grid value outside the parameter's range would be written straight
        // over the parameter at that cell, where no other check sees it (a
        // `--starts` value goes through the resolver's bounds check; a sweep
        // value does not). The range is the model's (`resolved_bounds`, read
        // off the IR as loaded), narrowed by the fit config's `[estimate]`
        // bounds when it gives them.
        let model_bounds = model_pre.parameters.iter()
            .find(|p| p.name == sw.name)
            .and_then(crate::params_resolver::resolved_bounds);
        let fit_bounds = fit_estimate.get(&sw.name).and_then(|e| e.bounds);
        let bounds = match (model_bounds, fit_bounds) {
            (Some((mlo, mhi)), Some((flo, fhi))) => Some((mlo.max(flo), mhi.min(fhi))),
            (m, f) => f.or(m),
        };
        if let Some((lo, hi)) = bounds {
            let outside: Vec<String> = values.iter()
                .filter(|&&v| !(lo..=hi).contains(&v))
                .map(|v| v.to_string())
                .collect();
            if !outside.is_empty() {
                eprintln!(
                    "error: --sweep {}: value(s) {} outside the parameter's bounds [{}, {}]{}. \
                     Sweep only inside the bounds.",
                    sw.name, outside.join(", "), lo, hi,
                    if fit_bounds.is_some() { " (the model's, narrowed by --fit [estimate])" }
                    else { " (the model's)" },
                );
                std::process::exit(1);
            }
        }
        focal_grids.push(FocalGrid {
            name: sw.name.clone(),
            values,
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
    // identically, and `--rw-sd auto` never estimates them. A file name a
    // later tier (a scenario, `--fixed`) re-valued is still pinned: the
    // override record keeps the file as the replaced source. A focal name
    // the file also carries stays focal (the sweep writes over it per cell).
    let cli_fixed_names: std::collections::HashSet<String> =
        a.model_overrides.fixed_cli.iter().map(|p| p.name.clone()).collect();
    let mut fixed_names: std::collections::HashSet<String> = cli_fixed_names.clone();
    for k in fit_toml_fixed.keys() {
        fixed_names.insert(k.clone());
    }
    let from_fixed_file = |s: &crate::params_resolver::ValueSource|
        matches!(s, crate::params_resolver::ValueSource::FixedFile { .. });
    for rp in &resolved.params {
        let named_by_file = from_fixed_file(&rp.source)
            || rp.overrides.iter().any(|o| from_fixed_file(&o.replaced.0));
        if named_by_file && !focal_names.contains(&rp.name) {
            fixed_names.insert(rp.name.clone());
        }
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

    // The profile's partition: today's selection, translated into role
    // declarations by the one adapter that reads it. Recorded, not acted on.
    let profile_roles: Result<crate::roles::Partition, String> =
        legacy_role_inputs(LegacySelection {
            estimated: &param_names_to_estimate,
            focal: &focal_names,
            fit_estimate: &fit_estimate,
            fit_fixed: fit_fixed_via,
            cli_fixed: &a.model_overrides.fixed_cli.iter()
                .map(|p| p.name.clone()).collect::<Vec<_>>(),
            fixed_files: &fixed_files_vec,
            model: &model_declared,
        })
        .and_then(|decls| crate::roles::resolve_roles(
            &decls.inputs(&model_declared, &fit_simplex_groups), &crate::roles::PROFILE,
        ).map_err(|e| e.to_string()));

    // Specs honour the fit-toml `[estimate]` block when supplied
    // (gh#73): bounds, transform, and perturb_only_at_t0 flow through to
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
            perturb_only_at_t0:
                from_fit.map(|e| e.perturb_only_at_t0).unwrap_or(false),
            bounds: from_fit.and_then(|e| e.bounds),
        }
    }).collect();

    let if2_params = crate::fit::runner::build_if2_params_from_specs(
        &model, &compiled, &base_params, &specs,
    ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    // Each IF2 cell perturbs by `rw_sd`, and a derived one must be able to
    // move the parameter.
    if matches!(profile_algo, ProfileAlgo::If2) {
        if let Err(e) = crate::fit::runner::refuse_degenerate_auto_rw_sd(
            &if2_params, "`--rw-sd NAME=SD`, or `rw_sd = …` in the fit config's [estimate]",
        ) {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }

    // `nl-lbfgs` takes each cell's optimum with the ODE forward-sensitivity
    // gradient, so ask the gradient capability gate once, here, rather than
    // letting every cell in the grid discover the same refusal and report it as
    // a failed cell. The estimated set is the non-focal free parameters, which
    // is exactly what `optimize_cell` will differentiate with respect to —
    // pinned focal parameters never enter it. (`fit run` asks the same question
    // in `FitConfig::validate_gradient_capability`.)
    if matches!(
        profile_algo,
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs)
    ) {
        let estimated: std::collections::HashSet<&str> =
            if2_params.iter().map(|p| p.name.as_str()).collect();
        if let Err(e) = sim::inference::gradient_capability::preflight_gradient_ode(
            &compiled, &base_params, &estimated,
        ) {
            eprintln!(
                "error: --algorithm nl-lbfgs takes each cell's optimum with the ODE \
                 forward-sensitivity gradient, and the gradient capability check refuses \
                 this model. The check's reason below still names `nuts`, the removed ODE \
                 sampler, for the same requirement (gh#917):\n\n  {e}\n\n  \
                 `--algorithm nl-sbplx` profiles the same ODE marginal likelihood without \
                 a gradient."
            );
            std::process::exit(1);
        }
    }

    // The fit config's `[[simplex_groups]]`: members that must sum to one.
    // IF2 keeps the constraint by perturbing them jointly on the log-ratio
    // scale, so every member has to be one of this profile's estimated
    // parameters — a swept or pinned member would leave the others free to
    // drift off the simplex. The nl-* optimizers do not honour the constraint
    // (as under `fit run`), which is said rather than dropped silently.
    let simplex_groups: Vec<sim::inference::if2::SimplexGroup> = if fit_simplex_groups.is_empty() {
        Vec::new()
    } else {
        for (gi, group) in fit_simplex_groups.iter().enumerate() {
            for name in &group.params {
                if !if2_params.iter().any(|p| p.name == *name) {
                    let why = if focal_names.contains(name) { "it is swept (--sweep)" }
                        else if fixed_names.contains(name) { "it is pinned" }
                        else { "it is not in --rw-sd" };
                    eprintln!(
                        "error: simplex_groups[{gi}] member '{name}' is not estimated by this \
                         profile: {why}. The members of a simplex group sum to one and are \
                         estimated jointly, so every member must be estimated: profile a \
                         parameter outside the group, or drop the group from the fit config.");
                    std::process::exit(1);
                }
            }
        }
        if !matches!(profile_algo, ProfileAlgo::If2) {
            eprintln!("warning: the fit config declares simplex_groups, but `{}` does not \
                honour the simplex constraint: members are optimised independently.",
                format!("{:?}", profile_algo).to_lowercase());
        }
        crate::fit::runner::resolve_simplex_groups(
            &fit_simplex_groups, &fit_estimate, &compiled.param_index, &if2_params,
        ).unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); })
    };
    // Member names per group, in order, for the method identity.
    let simplex_names: Vec<Vec<String>> =
        fit_simplex_groups.iter().map(|g| g.params.clone()).collect();

    let if2_params = Arc::new(if2_params);

    // ── Prior resolution (gh#73) ─────────────────────────────────────
    //
    // IF2 and the nl-* optimizers maximise the likelihood and read no prior.
    // The per-parameter source (fit_toml / model_ir / flat_fallback) is still
    // resolved once, up front, because it is provenance: it is recorded in the
    // profile-base sidecar and folded into the base identity (it is also what
    // keys the estimated *set*).
    let resolved_priors: Vec<crate::fit::priors_precedence::ResolvedPrior> =
        crate::fit::priors_precedence::resolve_priors_with_precedence(
            &if2_params.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            &fit_estimate,
            &model,
        );
    // gh#369: a `--fit` prior that displaces a model `~` prior, said once
    // before the grid (as `fit run` does). Over every `[estimate]` entry,
    // not only the non-focal set: the override holds wherever a prior is read.
    let prior_overrides = crate::fit::priors_precedence::prior_overrides(
        &fit_estimate.keys().cloned().collect::<Vec<_>>(), &fit_estimate, &model);
    if let Some(w) =
        crate::fit::priors_precedence::format_prior_override_warning(&prior_overrides)
    {
        eprint!("{}", w);
    }

    // Per-start init draws across non-focal estimated params (gh#42).
    // Computed once, reused at every grid cell so start_idx=k means the
    // same draw across cells (lets the per-start TSV rows be compared
    // cell-to-cell). `None` → every start uses `if2_params` directly
    // (Single mode; or `--starts 1`).
    // Every rule draws through the one seam the fit runners use
    // (`chain_starts::draw_chain_starts`), with a sourced rule's handle
    // resolved once. A source that did not converge is a warning here, not a
    // refusal: a profile is exploratory and makes no claim about the fit it
    // starts from.
    let resolved_starts = crate::fit::chain_starts::resolve_starts(&init_method, true)
        .unwrap_or_else(|e| {
            eprintln!("error: profile --starts {}: {}", init_method, e);
            std::process::exit(1);
        });
    let drawn_starts = {
        let priors: Vec<(String, sim::inference::pmmh::Prior)> = if2_params
            .iter()
            .map(|s| (s.name.clone(), crate::fit::runner::resolve_prior(&s.name, &fit_estimate, &resolved.model).0))
            .collect();
        let ctx = crate::fit::chain_starts::StartContext {
            resolved: &resolved,
            base_specs: &if2_params,
            priors: &priors,
        };
        crate::fit::chain_starts::draw_chain_starts(&ctx, &resolved_starts, n_starts, seed_base)
            .unwrap_or_else(|e| {
                eprintln!("error: profile --starts {}: {}", init_method, e);
                std::process::exit(1);
            })
    };
    let per_start_params: Option<Arc<Vec<Vec<sim::inference::if2::EstimatedParam>>>> =
        Some(Arc::new(drawn_starts.to_estimated_params(&if2_params)));

    let process = Arc::new(ChainBinomialProcess::new(compiled.clone()));
    // Build the multi-stream observation model from the loaded `ObsStream`s via
    // the single shared builder. Each stream's projection + likelihood come from
    // its `observations { }` block (the modern observation system, exactly as
    // `fit run` / `pfilter` resolve them) — there is no `--flow` override.
    // Concrete `Arc<MultiStreamObsModel>` — the IF2 per-cell call site
    // accepts `&dyn ObservationModel<ParticleState>` (auto-coerced from
    // `&MultiStreamObsModel`); the NLopt path needs the concrete type
    // for `optimize_cell` (`compute_ode_loglik` reads
    // `log_likelihood_from_flows_and_counts` directly).
    let obs_times_vec: Vec<f64> = observations.iter().map(|o| o.time).collect();

    let obs_model_obj: Arc<MultiStreamObsModel> = {
        // The `ObsStream -> StreamSpec` mapping is the single shared builder
        // (`stream_specs_from_obs_streams`): authoritative per-grid-time cells
        // (holes = `None`, skipped in the likelihood) + survey denominators
        // (`aux`), each stream fed its OWN schedule; `bind` re-merges to the
        // union and validates aux — identical to how `fit run` and `pfilter`
        // build their obs model, so an indexed survey family scores identically.
        let stream_specs = crate::fit::runner::stream_specs_from_obs_streams(&streams);
        let (bound, report) = BoundObs::bind(compiled.model.simulation.t_start, stream_specs)
            .unwrap_or_else(|report| {
            eprintln!("error: observation data invalid:\n{}", report.render());
            std::process::exit(1);
        });

        // gh#812: a non-fatal bind report was previously discarded, so the
        // Warn channel was write-only. Surface it — a zero denominator scores
        // as zero, and the user must be told which rows and why.
        if !report.findings().is_empty() {
            eprintln!("{}", report.render());
        }
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
    let model_identity = crate::resolve::model_identity_from_ir(&model_json);
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
        .unwrap_or_else(|| streams[0].name.clone());

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
    //
    // The resolver knows only `[estimate]` and the pins, so on its own it
    // records every parameter of a CLI-only profile as `fixed`. The roles
    // this profile actually gives them are set here: a swept parameter is
    // `focal` (pinned per cell at its grid value; the value recorded is the
    // base value the grid writes over), and every parameter IF2 or the
    // optimizer moves is `estimated`.
    let mut parameters_provenance =
        crate::run_meta::ParameterProvenance::of_all(&resolved.params);
    for (name, prov) in parameters_provenance.iter_mut() {
        if focal_names.contains(name) {
            prov.role = "focal".to_string();
        } else if if2_params.iter().any(|p| p.name == *name) {
            prov.role = "estimated".to_string();
        }
    }
    // Per-start init provenance, straight from the drawn starts.
    let init_provenance: Option<crate::run_meta::InitProvenance> =
        Some(crate::run_meta::InitProvenance::from_chain_starts(&drawn_starts));

    // Run-level provenance recorded into every profile-point leaf's
    // `RunRecord.inputs` — display payload, NOT identity-bearing, so it
    // does not affect the leaf `run_id`. Folds forward the audit data
    // the old per-run `ProfileMeta` carried: per-parameter resolution
    // provenance (gh#83/gh#85) and per-chain init provenance.
    let run_provenance_json = serde_json::json!({
        "parameters_provenance":
            serde_json::to_value(&parameters_provenance).unwrap_or(serde_json::Value::Null),
        "notices":
            serde_json::to_value(
                crate::params_resolver::notices(&[&resolved]).iter()
                    .chain(focal_notices.iter())
                    .cloned()
                    .collect::<crate::notice::Notices>())
                .unwrap_or(serde_json::Value::Null),
        "init_provenance":
            serde_json::to_value(&init_provenance).unwrap_or(serde_json::Value::Null),
    });

    // Each grid point's partition: the focal parameters fixed at the point's
    // values. Recorded in that point's leaves.
    let cell_partitions: Vec<serde_json::Value> = grid_points.iter().map(|gp| {
        let point: Vec<(String, f64)> = focal_grids.iter().zip(gp.iter())
            .map(|(fg, &(_, v))| (fg.name.clone(), v))
            .collect();
        let field = match &profile_roles {
            Ok(p) => match p.cell(&point) {
                Ok(cell) => crate::roles::PartitionField::Resolved(cell.record(&resolved)),
                Err(reason) => crate::roles::PartitionField::Unresolved { reason },
            },
            Err(reason) => crate::roles::PartitionField::Unresolved { reason: reason.clone() },
        };
        serde_json::to_value(&field).unwrap_or(serde_json::Value::Null)
    }).collect();

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
    let base_config_hash = crate::fit::cas::canonical_config_hash(&ProfileBaseLevel {
        base_params: &base_params_hash,
        fixed: &fixed_blob,
        obs_family: &obs_family_key,
        fit_toml: &fit_toml_hash,
        priors: &priors_blob,
        objective,
    }, &[]).unwrap_or_else(|e| {
        eprintln!("error: profile base identity: {e}");
        std::process::exit(1);
    });
    // Gate the RAW floats before `json!` sees them: the macro collapses
    // NaN/Inf to `Null`, so `resolve_profile_point`'s gate on the built blob
    // could never fire and NaN vs Inf would hash alike (2026-08-23 audit).
    // No hand-enumerated finiteness tuple here: `canonical_config_hash` gates
    // the WHOLE struct before serializing, so a float field added to either
    // level is covered automatically. The tuple this replaced listed
    // (cooling, dt, rw_sd) by hand — exclude-by-default applied to
    // finiteness, and the same forget-me shape as the literals themselves.
    // 2026-08-23 audit: three per-cell knobs reached the computation but not
    // the key, so a rerun that changed any of them was served the previous
    // run's landscape, cell for cell, under a "cached — resuming" line.
    //
    //  - rw_sd: the RESOLVED per-parameter IF2 perturbation magnitudes (CLI
    //    over fit-toml), sorted by name. `--rw-sd` is a REQUIRED flag whose
    //    values drive the sampler, and only the estimated *set* was keyed
    //    (incidentally, through the priors blob) — never the magnitudes.
    //    `auto` rides as its own discriminator since it derives the values
    //    from the model rather than the flag.
    //  - init: which starting points seed each start (the gh#514 class, still
    //    open on this command). Its companion PATHS are part of the resolved
    //    InitMethod, so they ride along.
    //  - pf_max_substeps: a degeneracy budget whose trip is swallowed here as
    //    `-inf` (unlike pfilter, which aborts), so a tripped budget could bake
    //    an -inf loglik into a cached cell that a later default-budget run was
    //    then served.
    let mut rw_sd_blob: Vec<(&str, Option<f64>)> =
        specs.iter().map(|s| (s.name.as_str(), s.rw_sd)).collect();
    rw_sd_blob.sort_by(|a, b| a.0.cmp(b.0));
    let method_config_hash = crate::fit::cas::canonical_config_hash(&ProfileMethodLevel {
        algorithm: &algorithm,
        if2: ProfileIf2Knobs {
            particles: n_particles, iterations: n_iterations,
            cooling, dt, starts: n_starts,
        },
        rw_sd: &rw_sd_blob,
        rw_sd_auto,
        simplex_groups: &simplex_names,
        init: &init_method,
        pf_max_substeps: a.inference.pf_max_substeps,
    }, &[]).unwrap_or_else(|e| {
        eprintln!("error: profile method identity: {e}");
        std::process::exit(1);
    });
    // The base fit's `starts_from` lineage would fold into the base as
    // a dep (guardrail 3-base). `camdl profile` does not currently
    // thread a base-fit lineage, so the dep list is empty; when it
    // does, push the resolved `FitStage` ArtifactRef here.
    // 2026-08-23 audit: fold the chain-start file's content into the identity,
    // the same treatment `fit run` got in gh#541. The resolved `init` in the
    // method blob names which source; the dep digests what is in it, so
    // rewriting a draws.tsv or params.toml in place, or re-running the fit a
    // `from_mle` names, re-keys instead of serving the previous landscape.
    let profile_deps: Vec<runid::inputs::ArtifactRef> = resolved_starts
        .source
        .as_ref()
        .map(|src| src.dep.clone())
        .into_iter()
        .collect();
    let store = runid::FsCasStore::new(&root);
    let ir_version_str = ir::IR_VERSION.trim().to_string();
    let stem_label = stem.clone().unwrap_or_else(|| "profile".to_string());

    let total_jobs = grid_points.len() * n_starts * seeds.len();
    let dim_str = focal_grids.iter()
        .map(|fg| format!("{}={}", fg.name, fg.values.len()))
        .collect::<Vec<_>>().join(" × ");
    // Banner names the actual per-cell optimizer + its real budget — a
    // deterministic nl-* cell has no particles/iterations, so labelling every
    // run "IF2 (N particles × M iter)" was misleading.
    let algo_label = match profile_algo {
        ProfileAlgo::If2 => "IF2",
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Sbplx) => "nl-sbplx",
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Bobyqa) => "nl-bobyqa",
        ProfileAlgo::Nlopt(sim::inference::deterministic::NloptAlgorithm::Lbfgs) => "nl-lbfgs",
    };
    eprintln!("profile: {} grid ({}) × {} starts × {} seeds = {} {} runs",
        grid_points.len(), dim_str, n_starts, seeds.len(), total_jobs, algo_label);
    match profile_algo {
        ProfileAlgo::If2 => eprintln!(
            "profile: IF2 per cell = {} particles × {} iterations", n_particles, n_iterations),
        ProfileAlgo::Nlopt(_) => eprintln!(
            "profile: deterministic ODE MLE per cell (no particle filter)"),
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

    // The full sweep grid (each focal axis name and its values), shared by
    // every point. Folded into each point's base identity so a distinct grid
    // (wider range, more points, shifted bounds) is a distinct run rather than
    // a silent merge onto the previous grid's cells.
    let grid_spec: Vec<(String, Vec<f64>)> = focal_grids.iter()
        .map(|fg| (fg.name.clone(), fg.values.clone()))
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
            base_config: base_config_hash,
            method_config: method_config_hash,
            focal: &focal,
            grid: &grid_spec,
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
        let roles = crate::roles::RoleRecord::new(
            profile_roles.as_ref().map(|p| p.record(&resolved)).map_err(String::clone),
            || (resolved_priors_kv.iter().map(|(n, _)| n.clone()).collect(), Default::default()),
        );
        let sidecar = crate::run_meta::FitSidecar {
            label: label_arg.clone(),
            model_path: ir_path.clone(),
            model_identity: model_identity.clone(),
            fit_toml_path: a.fit.as_ref()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
            fit_toml_hash: fit_toml_hash.clone().unwrap_or_default(),
            data_hashes: data_hashes.iter().cloned().collect(),
            estimated: roles.estimated,
            fixed: roles.fixed,
            partition: Some(roles.partition),
            resolved_priors: resolved_priors_kv.iter()
                .map(|(n, s)| crate::run_meta::ResolvedPriorEntry {
                    param: n.clone(), source: s.clone(),
                    overridden_model_prior: prior_overrides.iter()
                        .find(|o| &o.param == n).map(|o| o.model.clone()) })
                .collect(),
            parameters_provenance: parameters_provenance.clone(),
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

    // ── Run remaining jobs: greedy priority work-queue ──────────────
    // Per focal axis, its grid's value range — for normalizing cell
    // coordinates so the nearest-to-best ordering is scale-fair.
    let axis_ranges: Vec<(f64, f64)> = focal_grids.iter().map(|fg| {
        let mn = fg.values.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = fg.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mn, mx)
    }).collect();
    let cell_norm: Vec<Vec<f64>> = (0..grid_points.len())
        .map(|gi| profile_cell_norm(&grid_points, &axis_ranges, gi))
        .collect();
    // The best (finite) profile-loglik seen and the grid cell that produced
    // it. Workers drill toward this cell; the bar surfaces the loglik. This is
    // a read-only side-channel — each job writes its own identity-keyed leaf,
    // so it never changes any output, only the order cells are visited in.
    let best: std::sync::Mutex<(f64, Option<usize>)> =
        std::sync::Mutex::new((f64::NEG_INFINITY, None));

    let eval_job = |ji: usize| {
        let (seed_idx, grid_idx, start_idx, ref resolved_pt_id, ref cas_path) = resolved_jobs[ji];
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

        let job_seed = seed ^ (grid_idx as u64 * 1000 + start_idx as u64);
        let job_t0 = std::time::Instant::now();

        // gh#42: when --starts lhs/uniform, each start uses its own draw
        // across the non-focal estimated params; otherwise (Single, or
        // --starts 1) every start shares `if2_params`. The focal params
        // are already pinned in `params` above, so the LHS draw on
        // non-focal indices doesn't perturb the grid point.
        let per_start_specs: &[sim::inference::if2::EstimatedParam] =
            per_start_params.as_ref()
                .map(|psp| psp[start_idx].as_slice())
                .unwrap_or(if2_params.as_slice());

        // Dispatch on algorithm. IF2 keeps the historical PF-driven
        // per-cell MLE; NLopt branches into the
        // deterministic-MLE path (gh#47) — same `compute_ode_loglik`
        // closure that fit.toml stages call, with the focal grid value
        // pinned in `params` and optimization confined to the non-focal
        // estimated indices.
        //
        // gh#74 Option B: each branch also populates a
        // `PerStartDiagnostics` carrying the convergence info this
        // start produced (cooling-end state for IF2, completion flag
        // for everyone). The rollup phase
        // aggregates these across K starts → per-cell columns.
        let mut diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        let (final_loglik, mle_params): (f64, Vec<f64>) = match profile_algo {
            ProfileAlgo::If2 => {
                let config = IF2Config {
                    n_particles, n_iterations,
                    cooling_fraction: cooling, cooling_target_iters: n_iterations, dt,
                    t_start: process.compiled.model.simulation.t_start,
                    simplex_groups: simplex_groups.clone(),
                    skip_first_obs_from_loglik: objective.skips_first_observation(),
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
                            skip_first_obs_from_loglik: objective.skips_first_observation(),
                            record_ancestry: false,
                            record_prequential: false,
                            record_predictions: false, // gh#520: only `.log_likelihood` is read below
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
                        (true_ll, r.mle)
                    }
                    Err(_) => {
                        // diag.completed stays false; iterations_used /
                        // cooling_final / loglik_trace stay at defaults
                        // — the aggregator records NaN for this start's
                        // contribution.
                        (f64::NEG_INFINITY, params.clone())
                    }
                }
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
                    // A profile's progress unit is a grid cell, not an
                    // evaluation; `profile` reports its own bar (gh#900).
                    None,
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
                        (r.loglik, mle)
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: nlopt optimize_cell failed at grid {}/start {}: {}",
                            grid_idx, start_idx, e
                        );
                        (f64::NEG_INFINITY, params.clone())
                    }
                }
            }
        };
        let elapsed = job_t0.elapsed().as_secs_f64();

        // Report this cell's loglik to the shared best: the drill target for
        // the work-queue and the bar's researcher metric. Never gates the leaf
        // write below (order-independent output).
        if final_loglik.is_finite() {
            if let Ok(mut b) = best.lock() {
                if final_loglik > b.0 {
                    *b = (final_loglik, Some(grid_idx));
                    bar.set(crate::progress::best_ll(final_loglik));
                }
            }
        }

        // gh#147 (M3.3): claim the CAS leaf, write the cell's mle.toml there,
        // finalize. The profile-base is never written (path segment only);
        // each (point, stage, seed, start) is its own leaf. The display
        // payload (method/algorithm/loglik) is recorded in `inputs`, not
        // hashed.
        let algorithm_payload = match profile_algo {
            ProfileAlgo::If2 => serde_json::json!({
                "particles": n_particles, "iterations": n_iterations,
                "cooling": cooling, "dt": dt }),
            ProfileAlgo::Nlopt(_) => serde_json::json!({
                "tolerance": 1e-4, "max_evals": 1500,
                "dt": compiled.model.simulation.dt.unwrap_or(dt) }),
        };
        // A non-finite cell loglik serialises as JSON `null` (JSON has no
        // ±inf/NaN), and every consumer that reads `best_loglik` — the curve,
        // the summary, the rollup — then has nothing to plot for this grid
        // point. The leaf still exists and the run still reports "N cells
        // written", so without this the point just *disappears* from the
        // profile with no indication that anything went wrong. Say so on
        // stderr: a dropped grid point is a result the user must see, not a
        // gap to infer. The usual cause is a particle filter too small to
        // score a θ far from the optimum (the swarm dies and the estimate is
        // −inf); `--particles` is the lever.
        if !final_loglik.is_finite() {
            eprintln!(
                "warning: profile grid point {} (start {}) produced a non-finite \
                 log-likelihood ({}); its `best_loglik` is recorded as null and the \
                 point will be MISSING from the profile curve. This is usually a \
                 particle filter too small to score a θ this far from the optimum — \
                 re-run with more `--particles`.",
                grid_idx, start_idx, final_loglik,
            );
        }
        // The run's provenance, with this cell's partition.
        let mut provenance = run_provenance_json.clone();
        provenance["partition"] = cell_partitions[grid_idx].clone();
        let inputs_json = serde_json::json!({
            "method": profile_algo.method_kind().as_str(),
            "seed": job_seed,
            "n_chains": 1,
            "algorithm": algorithm_payload,
            "grid_point": grid_idx,
            "start": start_idx,
            "best_loglik": if final_loglik.is_finite() { Some(final_loglik) } else { None },
            // The class of the profiled loglik follows the profile method
            // (gh#280): if2 / marginal / ode_marginal — a marginal in every
            // case, never PGAS's joint.
            "loglik_type": crate::fit::loglik::LoglikType::from(profile_algo.method_kind()).tag(),
            // Which likelihood `best_loglik` is: `full` scores every
            // observation, `conditional_on_first_observation` (`ic_free`)
            // leaves y₁'s term out. Compare a cell only with a fit of the same
            // objective.
            "likelihood_objective": objective,
            "wall_time_seconds": elapsed,
            "provenance": provenance,
        });
        // Streaming write through the one resolved-writer seam (gh#241 PR D).
        // The running record carries Null inputs (the cell's loglik summary is
        // a post-run result); the final inputs are supplied to `finalize`.
        let resolved_artifact = crate::resolve::ResolvedArtifact {
            kind: runid::ArtifactKind::ProfilePoint,
            levels: resolved_pt_id.levels.clone(),
            run_id: resolved_pt_id.run_id,
            display_inputs: serde_json::Value::Null,
        };
        let meta = crate::resolve::RecordMeta::new(&ir_version_str, &ir_path, None)
            .with_deps(profile_deps.clone());
        let write = match crate::resolve::begin_resolved_write(
            &store, &root, &resolved_artifact, &meta,
            crate::resolve::WriteMode::Streaming,
        ) {
            Ok(crate::resolve::ResolvedWrite::Streaming(c)) => c,
            Ok(crate::resolve::ResolvedWrite::Committed(_)) => {
                unreachable!("Streaming write mode never returns a committed path")
            }
            // The identical point is already stored — a cache report, not a
            // failure (this start recomputed it for nothing).
            Err(runid::CasError::AlreadyCompleted { .. }) => return,
            // Another process holds this point right now; its result stands.
            Err(e @ runid::CasError::FitInProgress { .. }) => {
                eprintln!("warning: not storing profile point {} — {}",
                    cas_path.display(), e);
                return;
            }
            // Anything else means this point was NOT recorded: fail loudly
            // rather than leaving a gap in the landscape the user never sees.
            Err(e) => {
                eprintln!("error: claim profile point {}: {}", cas_path.display(), e);
                std::process::exit(1);
            }
        };

        let mle_toml = render_mle_toml(&if2_params, &focal_values,
            &focal_grids.iter().map(|fg| fg.name.as_str()).collect::<Vec<_>>(),
            &mle_params, final_loglik, &diag);
        // Through the claim (fsync'd), error fatal: a discarded write error
        // could finalize a Completed point with no mle.toml in it.
        if let Err(e) = write.write("mle.toml", mle_toml.as_bytes()) {
            eprintln!("error: writing mle.toml into {}: {}", cas_path.display(), e);
            std::process::exit(1);
        }

        if let Err(e) = write.finalize(inputs_json) {
            eprintln!("warning: finalize profile point {}: {}", cas_path.display(), e);
        }

        // Passive progress tick. `Task` handles Pretty (redraw) / Plain
        // (throttled `profile pos/len` log line) / None (no-op) internally.
        bar.inc(1);
    };

    // N workers pull the pending job whose cell is nearest the current best
    // cell, keeping every core busy while evaluation drills toward the
    // optimum. Order-independent (each job writes its own identity-keyed
    // leaf), so no barrier and no determinism requirement — the pick just
    // reorders which cell a freed core visits next.
    let pending: std::sync::Mutex<Vec<usize>> =
        std::sync::Mutex::new(remaining.clone());
    let worker = || {
        loop {
            // Snapshot the drill target (release the lock before touching
            // `pending`, so the two locks never nest).
            let target = best.lock().map(|b| b.1).unwrap_or(None);
            let ji = {
                let mut p = match pending.lock() {
                    Ok(p) => p,
                    Err(_) => break,
                };
                if p.is_empty() { break; }
                let pick = match target {
                    // Nearest unevaluated cell to the best so far.
                    Some(bc) => {
                        let mut best_i = 0usize;
                        let mut best_d = f64::INFINITY;
                        for (i, &cand) in p.iter().enumerate() {
                            let d = profile_cell_dist2(
                                &cell_norm[resolved_jobs[cand].1], &cell_norm[bc]);
                            if d < best_d { best_d = d; best_i = i; }
                        }
                        best_i
                    }
                    // No finite best yet — take the next pending job.
                    None => 0,
                };
                p.swap_remove(pick)
            };
            eval_job(ji);
        }
    };
    // One worker loop per pool thread, placed by `broadcast` in that thread's
    // private queue (gh#821, `fit::chain_fanout`). Spawned as ordinary scope
    // jobs, a not-yet-started loop could be stolen by a worker blocked inside
    // an IF2 run's particle `par_iter`, which would then drain the whole queue
    // on top of the run it had not finished.
    let run_sweep = || {
        rayon::broadcast(|_| worker());
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

/// The values a single-point starts rule supplies, and where they came from:
/// `--starts from_params=<toml>` reads the file, `--starts from_mle=<fit>` the
/// stored fit's point estimate (`fit_state.toml` `start_values`). They enter
/// the resolver at tier 1.5 ([`crate::params_resolver::StartsValues`]),
/// above the model default and below every other tier, so the file's value
/// wins over the DSL default and anything the user passes explicitly
/// (`--fit` `[fixed]`, `--fixed-file`, the scenario, `--fixed`) wins over the
/// file, with the replacement reported.
///
/// Per-chain-varying rules (`from_prior`, `from_posterior`) and the spread
/// rules (single, uniform, lhs) have no single point and return `None`; users
/// of those supply a base value another way (`--fit` with `[estimate].start`,
/// or `--fixed-file`).
///
/// Sorted by name, so the resolver applies them in a reproducible order.
fn starts_values_from_init_method(
    init_method: &crate::fit::starts::ChainStarts,
) -> Result<Option<(crate::params_resolver::StartsOrigin, Vec<(String, f64)>)>, String> {
    use crate::fit::starts::{ChainStarts, Point};
    use crate::params_resolver::StartsOrigin;
    let (origin, values): (StartsOrigin, HashMap<String, f64>) = match init_method {
        ChainStarts::Point(Point::FromParams { path }) => {
            let values = crate::util::load_params_toml(&path.to_string_lossy())
                .map_err(|e| format!(
                    "loading the --starts from_params file: {}", e))?;
            (StartsOrigin::Params { path: path.clone() }, values)
        }
        ChainStarts::Point(Point::FromMle { source }) => {
            // The stored fit's point estimate, from the leaf the handle names.
            let leaf = crate::fit::chain_starts::resolve_fit_leaf(&source.0)
                .map_err(|e| format!("--starts from_mle={}: {}", source, e))?;
            let state = crate::fit::state::FitState::load(&leaf.to_string_lossy())
                .map_err(|e| format!("--starts from_mle={}: {}", source, e))?;
            crate::fit::chain_starts::refuse_sampler_as_mle_source(&state.method, &source.0)?;
            (StartsOrigin::Mle { fit: source.0.clone() },
             state.start_values.into_iter().collect())
        }
        _ => return Ok(None),
    };
    let mut values: Vec<(String, f64)> = values.into_iter().collect();
    values.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Some((origin, values)))
}

/// Render a per-start MLE TOML file. Human-readable; also the format
/// `rewrite_rollup` reads back to reconstruct the rollup.
///
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
    diag: &crate::profile_diagnostics::PerStartDiagnostics,
) -> String {
    let mut body = String::new();
    body.push_str("# Per-start MLE for one profile grid point.\n\n");
    body.push_str(&format!("final_loglik = {}\n", final_loglik));
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


/// Today's `profile` selection: the estimated set `--rw-sd` chose (its names,
/// or with `auto` every parameter not swept, held or a constant) and the
/// declarations that held the rest.
struct LegacySelection<'a> {
    estimated: &'a [String],
    focal: &'a [String],
    fit_estimate: &'a indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
    /// `Err` when the fit config's `[fixed]` cannot be read.
    fit_fixed: Result<Vec<(String, crate::roles::FitFixedVia)>, String>,
    /// `--fixed` names.
    cli_fixed: &'a [String],
    fixed_files: &'a [std::path::PathBuf],
    /// The model as declared, against which a fit artifact's legacy
    /// constant entries are set aside ([`crate::params_resolver::load_fixed_file`]).
    model: &'a ir::Model,
}

/// [`LegacySelection`] as role declarations: the owned half of a
/// [`crate::roles::RoleInputs`].
struct LegacyRoleDecls {
    fit_estimate: indexmap::IndexMap<String, crate::fit::config_v2::EstimateSpecV2>,
    fit_fixed: Vec<(String, crate::roles::FitFixedVia)>,
    cli_estimate: Vec<crate::roles::EstimateDecl>,
    cli_fixed: Vec<String>,
    files: Vec<crate::roles::FixedFileDecl>,
    design: Vec<(String, crate::roles::DesignOrigin)>,
    slice: bool,
}

impl LegacyRoleDecls {
    fn inputs<'a>(
        &'a self,
        model: &'a ir::Model,
        simplex_groups: &'a [crate::fit::config_v2::SimplexGroup],
    ) -> crate::roles::RoleInputs<'a> {
        crate::roles::RoleInputs {
            model,
            fit_estimate: &self.fit_estimate,
            fit_fixed: &self.fit_fixed,
            simplex_groups,
            cli_estimate: &self.cli_estimate,
            cli_fixed: &self.cli_fixed,
            cli_fixed_files: &self.files,
            design: &self.design,
            slice: self.slice,
            // A scenario is a value source in today's `profile`, its values
            // the starting points of the estimated parameters (proposal
            // §3.2 item 11); it is checked against the roles from Stage 3.
            scenarios: &[],
        }
    }
}

/// The one adapter between `profile`'s legacy selection and the partition
/// (proposal `2026-10-26-parameter-roles.md` §11, Stage 1): `roles` never
/// sees `--rw-sd`, and no rule in `resolve_roles` depends on it. Stage 3
/// deletes this and builds the declarations from `--estimate` and
/// `[estimate]` directly.
///
/// The translation states today's outcome as declarations:
///
/// - an estimated name is its `[estimate]` entry, else an `--estimate`
///   declaration with no bounds of its own;
/// - a focal name keeps its `[estimate]` entry, which makes it the focal
///   parameter of the profile;
/// - an `[estimate]` name neither estimated nor focal is held: it is left
///   undeclared, fixed by default, or held by the `--fixed` that named it;
/// - a plain `--fixed-file`'s entries for a focal name are dropped (the grid
///   writes over them today); a fit artifact's are kept, and the partition
///   lists them as unused;
/// - nothing estimated is a slice.
fn legacy_role_inputs(sel: LegacySelection<'_>) -> Result<LegacyRoleDecls, String> {
    let kept = |name: &String| sel.estimated.contains(name) || sel.focal.contains(name);
    // In the estimated set's order, which is the order of the method's
    // parameters, then the focal entries.
    let fit_estimate = sel.estimated.iter().chain(sel.focal.iter())
        .filter(|n| kept(n))
        .filter_map(|n| sel.fit_estimate.get(n).map(|s| (n.clone(), s.clone())))
        .collect();
    let cli_estimate = sel.estimated.iter()
        .filter(|n| !sel.fit_estimate.contains_key(*n))
        .map(|n| crate::roles::EstimateDecl { name: n.clone(), bounds: None })
        .collect();
    let fit_fixed = sel.fit_fixed?;
    let cli_fixed = sel.cli_fixed.to_vec();
    let mut files = Vec::new();
    for path in sel.fixed_files {
        let fit_artifact = crate::params_resolver::carries_fit_provenance(path);
        let mut names: Vec<String> = crate::params_resolver::load_fixed_file(sel.model, path)
            .map_err(|e| e.to_string())?.0
            .into_keys()
            .filter(|n| fit_artifact || !sel.focal.contains(n))
            .collect();
        names.sort();
        files.push(crate::roles::FixedFileDecl { path: path.clone(), names, fit_artifact });
    }
    Ok(LegacyRoleDecls {
        fit_estimate,
        fit_fixed,
        cli_estimate,
        cli_fixed,
        files,
        design: sel.focal.iter()
            .map(|n| (n.clone(), crate::roles::DesignOrigin::Sweep))
            .collect(),
        slice: sel.estimated.is_empty(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ProfileBaseLevel` / `ProfileMethodLevel` serialize exactly as these
    /// `json!` literals: the hashed shape of each identity level, stated
    /// once, so a field added to or dropped from either struct is a
    /// deliberate, visible re-key rather than a silent one.
    ///
    /// `digest_value` canonicalizes (sorts keys recursively), so equality of
    /// the serialized values is what matters, not field order.
    #[test]
    fn identity_levels_serialize_as_the_stated_literals() {
        let fixed: Vec<String> = vec!["N0".into()];
        let priors: Vec<(String, String)> = vec![("beta".into(), "log_normal".into())];
        let fit_toml: Option<String> = Some("abc123".into());

        let base_struct = serde_json::to_value(ProfileBaseLevel {
            base_params: "bp-hash",
            fixed: &fixed,
            obs_family: "poisson",
            fit_toml: &fit_toml,
            priors: &priors,
            objective: crate::fit::config_v2::LikelihoodObjective::Full,
        }).unwrap();
        let base_literal = serde_json::json!({
            "base_params": "bp-hash",
            "fixed":       fixed,
            "obs_family":  "poisson",
            "fit_toml":    fit_toml,
            "priors":      priors,
        });
        assert_eq!(base_struct, base_literal,
            "ProfileBaseLevel must reproduce the literal");

        // The full objective is omitted, so a profile without `ic_free` keeps
        // its key; the conditional one (`ic_free = true`) enters the base and
        // re-keys it.
        let conditional = serde_json::to_value(ProfileBaseLevel {
            base_params: "bp-hash",
            fixed: &fixed,
            obs_family: "poisson",
            fit_toml: &fit_toml,
            priors: &priors,
            objective: crate::fit::config_v2::LikelihoodObjective::ConditionalOnFirstObservation,
        }).unwrap();
        let mut conditional_literal = base_literal.clone();
        conditional_literal["objective"] =
            serde_json::json!("conditional_on_first_observation");
        assert_eq!(conditional, conditional_literal,
            "a conditional objective must enter the base level");

        let rw_sd: Vec<(&str, Option<f64>)> = vec![("N0", Some(5.0))];
        let init = crate::fit::starts::ChainStarts::Spread(crate::fit::starts::Spread::Lhs);
        let method_struct = serde_json::to_value(ProfileMethodLevel {
            algorithm: "if2",
            if2: ProfileIf2Knobs {
                particles: 30, iterations: 5, cooling: 0.7, dt: 1.0, starts: 1,
            },
            rw_sd: &rw_sd,
            rw_sd_auto: false,
            simplex_groups: &[],
            init: &init,
            pf_max_substeps: None,
        }).unwrap();
        let method_literal = serde_json::json!({
            "algorithm": "if2",
            "if2": { "particles": 30usize, "iterations": 5usize,
                     "cooling": 0.7, "dt": 1.0, "starts": 1usize },
            "rw_sd": rw_sd,
            "rw_sd_auto": false,
            "init": init,
            "pf_max_substeps": None::<u64>,
        });
        assert_eq!(method_struct, method_literal,
            "ProfileMethodLevel must reproduce the literal");

        // Simplex groups enter the method level only when there are some, so a
        // profile without groups keeps the key above.
        let groups = vec![vec!["p1".to_string(), "p2".to_string()]];
        let with_groups = serde_json::to_value(ProfileMethodLevel {
            algorithm: "if2",
            if2: ProfileIf2Knobs {
                particles: 30, iterations: 5, cooling: 0.7, dt: 1.0, starts: 1,
            },
            rw_sd: &rw_sd,
            rw_sd_auto: false,
            simplex_groups: &groups,
            init: &init,
            pf_max_substeps: None,
        }).unwrap();
        let mut literal_with_groups = method_literal.clone();
        literal_with_groups["simplex_groups"] = serde_json::json!(groups);
        assert_eq!(with_groups, literal_with_groups);
    }

    // ── starts_values_from_init_method: the starts rule's point ───────
    //
    // A single-point starts rule (`from_params=FILE`, `from_mle=FIT`) loads
    // one parameter point, which enters the resolver as its own tier. These
    // pin what is loaded and how it is attributed; the tier's precedence is
    // pinned by the resolver's own tests
    // (`params_resolver::tests::starts_values_*`).

    fn write_toml(dir: &std::path::Path, name: &str, body: &str) -> std::path::PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn starts_from_params_loads_the_file_named_by_its_path() {
        let tmp = tempfile::tempdir().unwrap();
        let toml_path = write_toml(tmp.path(), "start.toml",
            "gamma = 0.1\nbeta = 0.5\n");
        let init = crate::fit::starts::ChainStarts::Point(
            crate::fit::starts::Point::FromParams { path: toml_path.clone() });
        let (origin, values) = starts_values_from_init_method(&init).unwrap().unwrap();
        assert_eq!(origin, crate::params_resolver::StartsOrigin::Params { path: toml_path });
        assert_eq!(values, vec![("beta".to_string(), 0.5), ("gamma".to_string(), 0.1)],
            "every value in the file, sorted by name");
    }

    #[test]
    fn starts_from_mle_reads_the_leaf_point_estimate() {
        // `from_mle=<leaf dir>` reads the stored fit's `fit_state.toml`
        // `start_values` — the same file `fit run`'s `from_mle` reads — and is
        // attributed to the fit handle as given.
        let tmp = tempfile::tempdir().unwrap();
        let leaf = tmp.path().join("if2-abcdef12").join("seed_1-01234567");
        std::fs::create_dir_all(&leaf).unwrap();
        write_toml(&leaf, "fit_state.toml",
            "method = \"if2\"\nseed = 1\ntimestamp = \"2026-01-01T00:00:00Z\"\n\
             best_loglik = -311.13\ninitial_loglik = -400.0\nbest_chain = 1\nn_chains = 2\n\n\
             [start_values]\nbeta = 0.5\n\n[rw_sd]\n");
        let handle = leaf.to_string_lossy().into_owned();
        let init = crate::fit::starts::ChainStarts::Point(crate::fit::starts::Point::FromMle {
            source: crate::fit::starts::Handle(handle.clone()),
        });
        let (origin, values) = starts_values_from_init_method(&init).unwrap().unwrap();
        assert_eq!(origin, crate::params_resolver::StartsOrigin::Mle { fit: handle });
        assert_eq!(values, vec![("beta".to_string(), 0.5)]);
    }

    #[test]
    fn starts_without_a_single_point_supply_no_values() {
        // from_prior / lhs have per-chain-varying values — there is no
        // single point to supply to the resolver.
        let init_prior = crate::fit::starts::ChainStarts::from_prior();
        assert!(starts_values_from_init_method(&init_prior).unwrap().is_none(),
            "from_prior must supply no base values — they are per-chain");
        let init_lhs = crate::fit::starts::ChainStarts::Spread(crate::fit::starts::Spread::Lhs);
        assert!(starts_values_from_init_method(&init_lhs).unwrap().is_none(),
            "lhs must supply no base values");
    }

    // ── render_mle_toml: the per-start leaf body ──────────────────────

    fn estimated_param(name: &str, index: usize) -> sim::inference::if2::EstimatedParam {
        sim::inference::if2::EstimatedParam {
            name: name.into(), index,
            initial: 0.5, rw_sd: 0.1,
            transform: sim::inference::types::Transform::None,
            lower: 0.0, upper: 1.0,
            perturb_only_at_t0: false, rw_sd_auto: false,
        }
    }

    #[test]
    fn render_mle_toml_writes_loglik_focal_mle_and_diagnostics() {
        // The whole body of one IF2 / nl-* leaf, pinned byte for byte: the
        // leaf is what the deferred rollup (gh#154) reads back.
        let if2 = vec![estimated_param("beta", 0)];
        let diag = crate::profile_diagnostics::PerStartDiagnostics::default();
        let body = render_mle_toml(&if2, &[25.0], &["R0"], &[0.42], -11.0, &diag);
        assert_eq!(
            body,
            "# Per-start MLE for one profile grid point.\n\n\
             final_loglik = -11\n\n\
             [focal]\nR0 = 25\n\n\
             [mle]\nbeta = 0.42\n\n\
             [diagnostics]\ncompleted = false\n",
        );
    }

}

