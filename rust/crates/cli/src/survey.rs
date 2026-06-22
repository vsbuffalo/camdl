//! `camdl survey` — likelihood-landscape diagnostic.
//!
//! Draws N Latin-hypercube points across declared parameter bounds,
//! evaluates the marginal log-likelihood at each point via a particle
//! filter (default) or a single deterministic trajectory (opt-in), and
//! writes a TSV ready for visualization. Optional `--render` produces
//! a self-contained interactive HTML pair-plot.
//!
//! This is a **diagnostic tool**, not a fitting routine. It does not
//! produce an MLE. See
//! `docs/dev/proposals/2026-05-03-survey-subcommand.md`.
//!
//! ## CAS layout
//!
//! ```text
//! <root>/surveys/<stem>-<hash[:8]>/
//!   run.json            # RunRecord (kind = survey)
//!   landscape.tsv       # primary artifact (always)
//!   summary.json        # SE distribution, top-K stats, dimensionality info
//!   landscape.html      # interactive pair-plot (only when --render)
//! ```
//!
//! Reuse paths:
//! - LHS sampling via `fit::init::build_chain_starts` (scale-aware)
//! - Bounds resolution via `fit::runner::build_if2_params_from_specs`
//! - PF eval via `sim::inference::particle_filter::bootstrap_filter`

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use indexmap::{IndexMap, IndexSet};
use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        particle_filter::{bootstrap_filter, Observation, PFilterResult},
        BoundObs, ChainBinomialProcess, MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
        traits::{ObservationModel, SMCConfig},
        types::{log_sum_exp, EstimatedParam, ParticleState},
    },
};

use crate::run_meta::SurveyEvalMethod;


// ─── Resolved input payload ──────────────────────────────────────────────────
//
// The fit-aware (`--fit`) and inline (`--estimate` / `--data`) input
// modes converge here: a single `ResolvedSurveyInputs` carrying the
// loaded model, the EstimatedParam specs (with bounds resolved
// fit.toml > model), the per-stream observation data and the scenario
// / fixed context. Everything past this point is mode-agnostic.

struct ResolvedSurveyInputs {
    /// Compiled model (Arc to share across rayon threads).
    compiled: Arc<CompiledModel>,
    /// Raw compiled IR envelope JSON. Source for the `runid` model identity
    /// recorded in the survey's `run.json` `inputs` (the `init = survey_top_k`
    /// cross-check, gh#51) — hashed raw (not the seeded in-memory model) so a
    /// `[estimate].start` edit doesn't spuriously break the cross-check.
    model_ir_json: String,
    /// Default parameter vector (post-fixed/scenario apply).
    base_params: Vec<f64>,
    /// EstimatedParam vector with resolved bounds — drives LHS.
    estimated: Vec<EstimatedParam>,
    /// Resolved IR observation models, in declaration order. The
    /// survey scores against ALL of them simultaneously (matches the
    /// fit-side multi-stream loglik convention).
    obs_models: Vec<ir::observation::ObservationModel>,
    /// Per-stream observations, aligned to `obs_models`.
    per_stream_obs: Vec<Vec<Observation>>,
    /// Per-stream data file content hashes, keyed by stream name.
    data_hashes: HashMap<String, String>,
    /// Resolved fixed params (name → value).
    fixed: HashMap<String, f64>,
    /// Named scenario applied (`None` = baseline).
    scenario: Option<String>,
    /// Per-parameter `[estimate].start =` values from fit.toml when in
    /// fit-aware mode; `None` in inline mode (no fit.toml). Used by
    /// the start-rank diagnostic to flag when the user's seeded
    /// best-guess falls in a low-loglik region of the LHS landscape.
    estimate_starts: Option<HashMap<String, f64>>,
}

// ─── cmd_survey entry point ──────────────────────────────────────────────────

pub fn cmd_survey(a: &crate::args::SurveyArgs) {
    // Validate input mode mutual exclusion at the boundary.
    if a.fit.is_none() {
        if a.data.is_none() {
            eprintln!(
                "error: camdl survey requires either --fit FIT.toml \
                 (fit-aware mode) or --data DATA.tsv with --estimate \
                 NAME=LO:HI flags (inline mode).\n\
                 Got neither.");
            std::process::exit(1);
        }
        if a.estimate.is_empty() {
            eprintln!(
                "error: --data {} given without any --estimate flags. \
                 Pass --estimate NAME=LO:HI for each parameter to vary \
                 across the LHS box (repeat for multiple parameters).",
                a.data.as_ref().unwrap().display());
            std::process::exit(1);
        }
    }

    if a.eval_replicates == 0 {
        eprintln!("error: --eval-replicates must be >= 1 (got 0).");
        std::process::exit(1);
    }
    if a.eval_particles == 0 && matches!(a.eval, SurveyEvalMethod::Pfilter | SurveyEvalMethod::Auto) {
        eprintln!("error: --eval-particles must be >= 1 \
                   (in case --eval auto resolves to pfilter).");
        std::process::exit(1);
    }
    // n_points = 0 is the auto-scale sentinel; resolved against d
    // after model loading. Negative values can't happen — usize.

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

    // gh#audit-H13: --parallel / CAMDL_PARALLEL throttles the rayon thread
    // budget. Build a SCOPED local pool and run the parallel sweep inside
    // `pool.install(...)`. The earlier fix used `build_global`, which is
    // order-dependent and one-shot (AlreadyInitialized is swallowed) — a
    // scoped pool always throttles. parallel == 0 means "use rayon's default"
    // (all logical cores): leave the pool unset and run on the global pool.
    let survey_pool: Option<rayon::ThreadPool> = if a.parallel > 0 {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(a.parallel)
                .build()
                .unwrap_or_else(|e| {
                    eprintln!("error: failed to build thread pool (--parallel {}): {}", a.parallel, e);
                    std::process::exit(1);
                }),
        )
    } else {
        None
    };

    let resolved = match resolve_survey_inputs(a) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    // Resolve `--eval auto` against the compiled model. `Auto` picks
    // Pfilter when the model has stochastic process noise
    // (`Capabilities::OVERDISPERSION` required), Simulate otherwise.
    // Resolved before any persistent state is written — `SurveyMeta`
    // stores the resolved method, not `Auto`.
    let eval_method: SurveyEvalMethod = match a.eval {
        SurveyEvalMethod::Auto => {
            let needs_pf = resolved.compiled
                .required_capabilities()
                .contains(sim::Capabilities::OVERDISPERSION);
            let resolved_eval = if needs_pf {
                SurveyEvalMethod::Pfilter
            } else {
                SurveyEvalMethod::Simulate
            };
            eprintln!(
                "survey: --eval auto resolved to '{}' (model {} \
                 stochastic process noise)",
                resolved_eval,
                if needs_pf { "has" } else { "does not have" });
            resolved_eval
        }
        explicit => explicit,
    };

    // Phase 1 of the ODE-inference proposal rewired `--eval simulate`
    // through `compute_ode_loglik` (the deterministic skeleton). For
    // overdispersed models that's a silent semantic mismatch — the
    // skeleton ignores σ². Auto-resolution above already steers
    // overdispersed models to Pfilter; a *user-explicit*
    // `--eval simulate` on an overdispersed model bypassed the gate
    // entirely (the same class as the profile / fit-run nl-sbplx silent-
    // fail). Fail fast with the same actionable message
    // `methods::check_model_capabilities` produces elsewhere.
    if eval_method == SurveyEvalMethod::Simulate {
        if let Err(msg) = crate::fit::methods::check_model_capabilities(
            crate::run_meta::InferenceBackend::Ode, &resolved.compiled,
        ) {
            eprintln!("error: {}", msg);
            eprintln!(
                "  (drop `--eval simulate` to use `--eval auto`, which \
                 routes overdispersed models through the particle filter \
                 automatically.)"
            );
            std::process::exit(1);
        }
    }

    // Curse-of-dim warnings + n_points auto-resolution
    // (proposal §"Runtime warnings"; gh43 follow-up).
    //
    // n_points = 0 is the auto-scale sentinel; resolve to
    // max(1000, 50 * d^2) so the n/d^2 >= 50 pair-plot coverage rule
    // is satisfied by default. d = 4 gives 1000 (unchanged from v1);
    // d = 8 gives 3200; d = 12 gives 7200. User can override with
    // --n-points N for fast iteration or sparse-basin models.
    let d = resolved.estimated.len();
    let n_points: usize = if a.n_points == 0 {
        let auto = (1000usize).max(50 * d.saturating_mul(d));
        eprintln!("survey: --n-points auto-scaled to {} for d={}", auto, d);
        auto
    } else {
        a.n_points
    };

    if d > 10 {
        eprintln!(
            "warning: surveying {} parameters; pair-plots become \
             hard to interpret past d ~= 8. Consider `camdl profile` \
             for higher-dimensional identifiability questions, or \
             restricting [estimate] to a focal subset.", d);
    } else if d > 6 {
        eprintln!(
            "note: surveying {} parameters; pair-plot 2D marginals \
             project a {}-D joint distribution. Concentrations in \
             one panel may reflect tight conditioning on parameters \
             not visible in that view.", d, d);
    }
    if d > 0 {
        let coverage_floor = 50.0 * (d as f64) * (d as f64);
        if (n_points as f64) < coverage_floor {
            eprintln!(
                "note: --n-points {} is below the rule-of-thumb \
                 coverage floor of n_points/d^2 >= 50 (d={}, \
                 recommended >= {}). Consider --n-points {} for \
                 adequate pair-plot resolution.",
                n_points, d, coverage_floor as usize, coverage_floor as usize);
        }
    }
    if eval_method == SurveyEvalMethod::Simulate && a.eval == SurveyEvalMethod::Simulate {
        // Only warn on explicit --eval simulate. `Auto`-resolved
        // Simulate already eprintln'd that the model has no process
        // noise; doubling the warning would confuse the user.
        eprintln!(
            "warning: --eval simulate uses a single deterministic \
             trajectory per LHS point. This is a 1-sample MC estimator \
             of p(y|theta) — biased toward 'lucky outliers' when \
             process noise is non-trivial (Andrieu & Roberts 2009; \
             Doucet et al. 2015). Use --eval pfilter unless the \
             model is known-deterministic.");
    }

    // Build typed CAS inputs.
    let bounds_map: HashMap<String, (f64, f64)> = resolved.estimated.iter()
        .map(|ep| (ep.name.clone(), (ep.lower, ep.upper)))
        .collect();
    let estimated_names: Vec<String> = resolved.estimated.iter()
        .map(|ep| ep.name.clone()).collect();
    let stem = crate::hashing::path_stem_slug(&a.model.to_string_lossy());
    let bounds_vec: Vec<(String, f64, f64)> = resolved.estimated.iter()
        .map(|ep| (ep.name.clone(), ep.lower, ep.upper)).collect();
    let fixed_vec: Vec<(String, f64)> = resolved.fixed.iter()
        .map(|(n, v)| (n.clone(), *v)).collect();
    let data_hashes_vec: Vec<(String, String)> = resolved.data_hashes.iter()
        .map(|(n, h)| (n.clone(), h.clone())).collect();
    let ir_version_str = ir::IR_VERSION.trim().to_string();
    let model_path_str = a.model.to_string_lossy().into_owned();

    // gh#147 (M3.3): resolve the content-addressed survey identity. The eval
    // count knobs (method/particles/replicates) fold into the `config` level —
    // they change the stored landscape, so they are in the key.
    let survey_ctx = crate::survey_cas::SurveyCtx {
        model:           &resolved.compiled.model,
        ir_version:      &ir_version_str,
        engine_version:  crate::version::VERSION_SHORT,
        stem:            stem.as_deref().unwrap_or("model"),
        data:            &data_hashes_vec,
        eval_method:     eval_method.as_str(),
        eval_particles:  a.eval_particles as u32,
        eval_replicates: a.eval_replicates as u32,
        bounds:          &bounds_vec,
        fixed:           &fixed_vec,
        scenario:        resolved.scenario.as_deref(),
        n_points:        n_points as u32,
        seed:            a.seed,
    };
    let resolved_id = match crate::survey_cas::resolve_survey(&survey_ctx) {
        Ok(r) => r,
        Err(e) => { eprintln!("error: survey CAS identity: {}", e); std::process::exit(1); }
    };

    let output_root = crate::run_paths::output_root(
        a.output.as_ref().map(|p| p.to_string_lossy().into_owned()).as_deref(),
        None,
    );
    let run_dir = runid::store_path(
        &output_root, runid::ArtifactKind::Survey, &resolved_id.levels);
    let run_id_hex = resolved_id.run_id.to_hex();

    // The render-side projection (decoupled from the survey inputs).
    let landscape_meta = crate::landscape_html::LandscapeMeta {
        stem:        stem.clone(),
        hash_short:  resolved_id.run_id.short8().to_string(),
        estimated:   estimated_names.clone(),
        bounds:      bounds_map.clone(),
        n_points,
        eval_method,
    };

    // Cache hit: a finalized landscape.tsv at the content-addressed path means
    // the survey is done (same path ⟺ same inputs). Re-render HTML if asked.
    {
        let landscape_path = run_dir.join("landscape.tsv");
        let html_path = run_dir.join("landscape.html");
        if !a.force && landscape_path.exists() {
            crate::status::step("cached", run_dir.display());
            if a.render && !html_path.exists() {
                eprintln!("  rendering --render HTML from cached landscape.tsv …");
                if let Err(e) = render_landscape_html(
                    &landscape_path, &html_path, &landscape_meta)
                {
                    eprintln!("warning: HTML render failed: {}", e);
                }
            }
            return;
        }
    }

    // Claim a streaming leaf through the one resolved-writer seam (gh#241
    // PR D): artifacts go into the staging dir, finalized atomically at the
    // end (a crash leaves no finalized landscape.tsv). The running record
    // carries Null inputs (the landscape summary is a post-run result); the
    // final inputs are supplied to `finalize`.
    let store = runid::FsCasStore::new(&output_root);
    let resolved_artifact = crate::resolve::ResolvedArtifact {
        kind: runid::ArtifactKind::Survey,
        levels: resolved_id.levels.clone(),
        run_id: resolved_id.run_id,
        display_inputs: serde_json::Value::Null,
    };
    let meta = crate::resolve::RecordMeta::new(
        &ir_version_str, &model_path_str, label_arg.clone());
    let write = match crate::resolve::begin_resolved_write(
        &store, &output_root, &resolved_artifact, &meta,
        crate::resolve::WriteMode::Streaming,
    ) {
        Ok(crate::resolve::ResolvedWrite::Streaming(c)) => c,
        Ok(crate::resolve::ResolvedWrite::Committed(_)) => {
            unreachable!("Streaming write mode never returns a committed path")
        }
        Err(e) => {
            eprintln!("error: claim survey leaf {}: {}", run_dir.display(), e);
            std::process::exit(1);
        }
    };
    let staging = write.dir().to_path_buf();
    let landscape_path = staging.join("landscape.tsv");
    let summary_path = staging.join("summary.json");
    let html_path = staging.join("landscape.html");

    crate::status::done("stored",
        format!("{} \u{b7} {} points (eval={})", run_dir.display(), n_points, eval_method));
    crate::status::hint("camdl list --kind survey");
    if eval_method == SurveyEvalMethod::Simulate {
        eprintln!(
            "  --eval simulate now computes p(y|θ, ODE_skeleton) via the ODE \
             backend (Phase 1 of the ODE-inference proposal). Pre-2026-05-04 \
             this flag ran a 1-particle bootstrap PF on chain_binomial; \
             cached landscape.tsv files from the older schema have been \
             invalidated. The two estimators agree to sub-nat at typhoid-\
             class N but diverge in low-count regimes; prefer --eval pfilter \
             when process noise is non-trivial.");
    }

    let t0 = std::time::Instant::now();

    // ── LHS sampling ────────────────────────────────────────────────
    //
    // gh#42's `build_chain_starts` is the scale-aware sampler. LHS
    // requires n >= 2; reject n_points = 1 upstream so the call here
    // doesn't degenerate to "just use base_params".
    let lhs_starts = crate::fit::init::build_chain_starts(
        crate::fit::init::InitMethod::Lhs,
        &resolved.estimated,
        n_points,
        a.seed,
    ).unwrap_or_else(|| {
        // n_points < 2 returns None from build_chain_starts.
        eprintln!("internal error: LHS sampler returned None at n_points={}", n_points);
        std::process::exit(1);
    });

    // ── Parallel evaluation loop ────────────────────────────────────
    // Survey hardcodes dt=1.0 for the SMCConfig (it doesn't expose a
    // user-tunable dt knob). gh#53: process must be built at the same
    // dt so its internal fire_steps resolves correctly.
    let smc_dt = 1.0_f64;
    let process = Arc::new(ChainBinomialProcess::new(resolved.compiled.clone()));
    let t_start = resolved.compiled.model.simulation.t_start;

    // Concrete `Arc<MultiStreamObsModel>`: trait-typed obs models could
    // also satisfy `eval_point_pfilter`, but `eval_point_simulate` (Phase 1
    // — ODE-skeleton eval through `compute_ode_loglik`) needs the concrete
    // type for `log_likelihood_from_flows_and_counts`. `&*obs_model`
    // auto-coerces to `&dyn ObservationModel<ParticleState>` for the
    // pfilter path.
    let obs_times: Vec<f64> = resolved.per_stream_obs.first()
        .map(|v| v.iter().map(|o| o.time).collect())
        .unwrap_or_default();
    let obs_model: Arc<MultiStreamObsModel> = {
        let mut stream_specs = Vec::with_capacity(resolved.obs_models.len());
        for (obs, stream_obs) in resolved.obs_models.iter().zip(resolved.per_stream_obs.iter()) {
            let projection = sim::inference::multi_stream_obs::StreamProjection::from_ir(
                &obs.projection, &resolved.compiled, &obs.name,
            ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
            // survey is dense + aux-free in v1 (survey denominators bind in the
            // fit / pfilter loaders; survey rejects NA upstream).
            stream_specs.push(StreamSpec::dense(
                projection,
                obs.clone(),
                sim::inference::dense_cells(
                    stream_obs.iter().map(|o| o.value).collect()),
                obs_times.clone(),
            ));
        }
        let (bound, _report) = BoundObs::bind(stream_specs).unwrap_or_else(|report| {
            eprintln!("error: observation data invalid:\n{}", report.render());
            std::process::exit(1);
        });
        Arc::new(MultiStreamObsModel::new(bound, resolved.compiled.clone())
            .unwrap_or_else(|e| {
                eprintln!("error: observation model construction failed: {:?}", e);
                std::process::exit(1);
            }))
    };

    // Progress: one overall bar over the LHS points, ticked from the parallel
    // sweep (`Task` is `Send + Sync`), with the best loglik found so far as the
    // researcher metric. Honors `--progress none/plain`.
    let bar = crate::progress::Reporter::new().task(n_points as u64, "survey", "pts");
    let best = std::sync::Mutex::new(f64::NEG_INFINITY);
    let sweep = || lhs_starts.par_iter().enumerate()
        .map(|(point_id, draw)| {
            // Build the full parameter vector: base_params overwritten
            // at each estimated index. Fixed params are already baked
            // into base_params (resolve_survey_inputs).
            let mut params = resolved.base_params.clone();
            for spec in draw {
                params[spec.index] = spec.initial;
            }
            let row = match eval_method {
                SurveyEvalMethod::Pfilter => eval_point_pfilter(
                    &process, obs_model.as_ref(),
                    &params, &resolved.estimated, draw,
                    a.eval_particles, a.eval_replicates,
                    smc_dt, t_start, a.seed, point_id,
                ),
                SurveyEvalMethod::Auto => unreachable!(
                    "Auto resolved before parallel eval loop"),
                SurveyEvalMethod::Simulate => eval_point_simulate(
                    &resolved.compiled, &obs_model, &obs_times,
                    &params, &resolved.estimated, draw,
                    smc_dt, point_id,
                ),
            };
            bar.inc(1);
            if row.loglik.is_finite() {
                if let Ok(mut b) = best.lock() {
                    if row.loglik > *b {
                        *b = row.loglik;
                        bar.set(crate::progress::best_ll(*b));
                    }
                }
            }
            row
        })
        .collect();
    let rows: Vec<LandscapeRow> = match &survey_pool {
        Some(pool) => pool.install(sweep),
        None => sweep(),
    };
    bar.finish();

    // ── TSV writer (sorted by loglik desc) ──────────────────────────
    let mut sorted = rows;
    sorted.sort_by(|a, b| {
        // -inf goes to the bottom; NaN treated as -inf for sort stability.
        let av = if a.loglik.is_nan() { f64::NEG_INFINITY } else { a.loglik };
        let bv = if b.loglik.is_nan() { f64::NEG_INFINITY } else { b.loglik };
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Err(e) = write_landscape_tsv(
        &landscape_path, &sorted, &resolved.estimated, eval_method,
        &run_id_hex) {
        eprintln!("error writing landscape.tsv: {}", e);
        std::process::exit(1);
    }
    eprintln!("survey: wrote {} ({} rows)", landscape_path.display(), sorted.len());

    // ── SE-distribution warning (proposal §"Runtime warnings") ──────
    if eval_method == SurveyEvalMethod::Pfilter {
        emit_se_warning(&sorted);
    }

    // ── Silent-miss mitigation: bound clustering + start-rank check ─
    //
    // After ranking, scan the top-10% of survey points for parameters
    // whose values pin near a bound. A pinned top-K means either:
    //   (a) the basin extends outside the user's declared bounds —
    //       the survey is ranking points along the bound itself; or
    //   (b) the bounds reflect informed priors and the basin really
    //       is at the edge.
    // Either way, surfacing it is worth ~30 LOC. Real reproducer:
    // typhoid SIRC β_veryhigh pinned at lower bound, ξ_a15plus at
    // upper, both invisible from loglik alone.
    //
    // Threshold: 5% of bound width (linear) or 5% of log-bound range
    // (Log-typed). Fires when the median of top-K values falls within
    // that fraction of either bound.
    emit_bound_clustering_warning(&sorted, &resolved.estimated, 0.10);

    // Start-rank check (fit-aware mode): for each [estimate].start =
    // value present in the fit.toml, find where it falls in the LHS
    // landscape. Quick "is the user's prior best-guess even in the
    // bounds box?" sanity check.
    if let Some(starts) = &resolved.estimate_starts {
        emit_start_rank_report(&sorted, &resolved.estimated, starts);
    }

    // ── summary.json ────────────────────────────────────────────────
    if let Err(e) = write_summary_json(&summary_path, &sorted, a, eval_method, d) {
        eprintln!("warning: could not write summary.json: {}", e);
    }

    // ── Render HTML if requested ────────────────────────────────────
    if a.render {
        if let Err(e) = render_landscape_html(&landscape_path, &html_path, &landscape_meta) {
            eprintln!("warning: HTML render failed: {}", e);
        } else {
            eprintln!("survey: wrote {}", html_path.display());
        }
    }

    // ── Finalize the CAS leaf ───────────────────────────────────────
    let elapsed = t0.elapsed().as_secs_f64();
    let best_loglik = sorted.iter().map(|r| r.loglik).find(|l| l.is_finite());
    // Cross-check provenance for `init = survey_top_k` (gh#51): the
    // `runid` model identity + per-stream data digests + the resolved
    // `[fixed]` block. `build_chain_starts_from_survey` validates these
    // against the fit before seeding chains. Recorded-not-hashed (the
    // identity is the `levels`); these are the human/consumer-readable
    // mirror the cross-check reads back.
    let model_identity = crate::resolve::model_identity_from_ir(&resolved.model_ir_json);
    let inputs_json = serde_json::json!({
        "eval_method":     eval_method.as_str(),
        "eval_particles":  a.eval_particles,
        "eval_replicates": a.eval_replicates,
        "n_points":        n_points,
        "estimated":       estimated_names,
        "best_loglik":     best_loglik,
        // A survey grid evaluates a marginal `log p(y | θ)` at each point
        // (PF or clean-eval), so the whole landscape is marginal-class (gh#280).
        "loglik_type":     crate::fit::loglik::LoglikType::Marginal.tag(),
        "wall_time_seconds": elapsed,
        "model_identity":  model_identity,
        "data_hashes":     resolved.data_hashes,
        "fixed":           resolved.fixed,
    });
    if let Err(e) = write.finalize(inputs_json) {
        eprintln!("warning: finalize survey leaf {}: {}", run_dir.display(), e);
    }
}

// ─── Resolution ──────────────────────────────────────────────────────────────

fn resolve_survey_inputs(a: &crate::args::SurveyArgs)
    -> Result<ResolvedSurveyInputs, String>
{
    use crate::fit::config_v2::FitConfigV2;
    use crate::fit::runner::{build_if2_params_from_specs, ParamSpec};

    let model_path = a.model.to_string_lossy().into_owned();

    if let Some(fit_path) = a.fit.as_ref() {
        // Fit-aware mode: load fit.toml; pull bounds from [estimate],
        // data from [data], fixed from [fixed], scenario from top.
        let fit_path_str = fit_path.to_string_lossy().into_owned();
        let config = FitConfigV2::load(&fit_path_str)?;
        // Make scenario+enable+disable mutual exclusion explicit
        // (matches fit::runner::FitRunConfig::build).
        if config.scenario.is_some() && (!config.enable.is_empty() || !config.disable.is_empty()) {
            return Err(
                "fit.toml: `scenario` is mutually exclusive with `enable`/`disable`. \
                 Use one approach.".into());
        }

        // Load model from fit.toml's `model.camdl` (already path-
        // resolved by FitConfigV2::load). `mut` because the gh#92
        // [estimate].start fall-back below seeds values into
        // `model_pre.parameters[i].value` before the resolver call.
        let (mut model_pre, model_ir_json) = crate::util::load_model(&config.model.camdl)?;

        // gh#92: [estimate].start fall-back for survey, matching the
        // gh#34 pattern applied to `camdl profile` (commit 796bfbe)
        // and `camdl fit run` (commit 19ae908). Survey's whole job is
        // to LHS-sample parameters listed in [estimate], so requiring
        // them to have a concrete value before LHS would be
        // self-defeating. We seed `spec.start` into model_pre.value
        // so the unified resolver's `UnsetRequired` check passes; LHS
        // sampling then overrides per-point downstream.
        //
        // Without this, the canonical workflow `camdl survey --fit
        // X.toml && camdl fit run X.toml --survey-path ...` errors at
        // survey even though `fit run` accepts the same toml.
        for (name, spec) in &config.estimate {
            if let Some(p) = model_pre.parameters.iter_mut().find(|p| p.name == *name) {
                if p.value.resolved_value().is_none() {
                    if let Some(start) = spec.start {
                        p.value = p.value.with_value(start);
                    }
                }
            }
        }

        // Resolve fit.toml [fixed] block via the existing config-side
        // pre-processor (handles `from_file`, `from_scenario`, and
        // inline-`values` shapes). The output is the IndexMap fed into
        // the unified resolver as `fit_toml_fixed`.
        //
        // `expand_from_scenario` + `resolve_with_model` are kept as the
        // fit-toml-side pre-processor (per 2026-05-25 CLI UX rev 2
        // proposal §"Step 4 — Migrate survey"); they no longer act as
        // a value writer on `model.parameters[*].value`. The unified
        // resolver writes those values.
        let mut fixed_cfg = config.fixed.clone();
        fixed_cfg.expand_from_scenario(&model_pre, &config.estimate)?;
        let fixed_resolved_indexmap: IndexMap<String, f64> =
            fixed_cfg.resolve_with_model(&model_pre)?;

        // Build the inputs for the unified resolver. Scenario
        // (or enable/disable) routing matches the previous behaviour.
        let scenario_opt: Option<String> = config.scenario.clone();
        let adhoc_enable: Vec<String> = config.enable.clone();
        let adhoc_disable: Vec<String> = config.disable.clone();
        // No `--fixed` on the survey-fit path; the inline `--fixed`
        // flag is mutually exclusive with `--fit` per SurveyArgs.
        let fixed_cli: Vec<(String, f64)> = Vec::new();
        let fixed_files: Vec<std::path::PathBuf> = Vec::new();
        let fit_toml_estimate: IndexSet<String> = config.estimate.keys().cloned().collect();
        let table_files: HashMap<String, std::path::PathBuf> = HashMap::new();

        let resolved = crate::params_resolver::resolve_parameters(
            crate::params_resolver::ParameterInputs {
                model: &model_pre,
                scenario: scenario_opt.as_deref(),
                adhoc_enable: &adhoc_enable,
                adhoc_disable: &adhoc_disable,
                fixed_cli: &fixed_cli,
                fixed_files: &fixed_files,
                fit_toml_fixed: &fixed_resolved_indexmap,
                fit_toml_estimate: &fit_toml_estimate,
                table_files: &table_files,
            },
        ).map_err(|e| e.to_string())?;
        crate::params_resolver::print_warnings(&resolved);
        let mut model = resolved.model.clone();

        // Apply [estimate].start as a starting-point hint for params
        // that survived the resolver without an inferred value (this is
        // a survey-specific helper for the LHS draw, not a value
        // override — see proposal §"Step 4").
        //
        // For params that don't have a resolved value yet (i.e. the
        // resolver returned them in the estimate set without a model
        // default), fall through to fit.toml [estimate].start, then to
        // a bounds-based draw.
        for (name, spec) in &config.estimate {
            if let Some(p) = model.parameters.iter_mut().find(|p| p.name == *name) {
                if p.value.resolved_value().is_none() {
                    let resolved_bounds = spec.bounds.or(p.bounds());
                    let v = spec.start.or_else(|| resolved_bounds.map(|(lo, hi)| {
                        let transform = crate::fit::runner::derive_transform_with_bounds(
                            p,
                            spec.transform.as_ref().map(|t| t.as_str()),
                            (lo, hi),
                        );
                        let log_scale = matches!(
                            transform, sim::inference::types::Transform::Log { .. }
                        );
                        crate::fit::init::draw_start_in_bounds(lo, hi, log_scale, a.seed, name)
                    }));
                    if let Some(value) = v {
                        p.value = p.value.with_value(value);
                    }
                }
            }
        }
        // After the [estimate].start fill-in, re-validate; the resolver
        // already validated tier-resolved values, but the [estimate]
        // fall-back can introduce new ones for params that were
        // previously valueless. Without this, downstream compile
        // failures lose the parameter-bounds-validation diagnostic
        // shape.
        crate::util::validate_parameter_values(&model)?;

        // Re-export the same IndexMap-shape map that the old code path
        // produced, for ResolvedSurveyInputs.fixed.
        let fixed_resolved: IndexMap<String, f64> = fixed_resolved_indexmap.clone();

        let compiled = Arc::new(CompiledModel::new(model.clone())
            .map_err(|e| format!("compile error: {:?}", e))?);
        let mut base_params = compiled.default_params.clone();
        for (name, spec) in &config.estimate {
            if let Some(start) = spec.start {
                if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                    base_params[idx] = start;
                }
            }
        }
        for (name, &v) in &fixed_resolved {
            if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                base_params[idx] = v;
            }
        }

        // Build EstimatedParam vector from [estimate]'s bounds, in
        // declaration order — the fit-toml-bounds-within-model-bounds
        // check is in `build_if2_params_from_specs`.
        let specs: Vec<ParamSpec> = config.estimate.iter()
            .map(|(name, spec)| ParamSpec {
                name: name.clone(),
                rw_sd: spec.rw_sd,
                transform: spec.transform.as_ref().map(|t| t.as_str().to_string()),
                ivp: spec.ivp,
                // Pass through Option as-is; build_if2_params_from_specs
                // resolves fit.toml > model > unbounded.
                bounds: spec.bounds,
            })
            .collect();
        let estimated = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)?;

        // Load observations from [data] and hash bytes per stream.
        let data_spec = config.data_spec()?;
        let model_obs_names: Vec<String> = model.observations.iter()
            .map(|o| o.name.clone()).collect();
        let effective = data_spec.effective_observations(&model_obs_names)?;
        if effective.is_empty() {
            return Err("fit.toml [data] resolves to zero observation streams.".into());
        }
        // Sort by name so order is canonical.
        let mut entries: Vec<(&String, &String)> = effective.iter().collect();
        entries.sort_by_key(|(k, _)| k.as_str());

        let mut obs_models = Vec::new();
        let mut per_stream_obs = Vec::new();
        let mut data_hashes: HashMap<String, String> = HashMap::new();
        let mut canonical_times: Option<Vec<f64>> = None;
        // Survey hardcodes dt=1.0 (see cmd_survey); reuse it for the dated-
        // loader's substep/grid checks.
        let time_opts = crate::caltime_load::TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt: 1.0,
            t_start: compiled.model.simulation.t_start,
            format: crate::caltime_load::TimeFormat::Auto,
        };
        for (stream_name, data_path) in &entries {
            let obs_model_ir = model.observations.iter()
                .find(|o| o.name == **stream_name).cloned()
                .ok_or_else(|| format!(
                    "no observation block named '{}' in model", stream_name))?;
            let observations = load_observations_from_tsv(data_path, &obs_model_ir, &time_opts)?;
            let times: Vec<f64> = observations.iter().map(|o| o.time).collect();
            match &canonical_times {
                None => canonical_times = Some(times),
                Some(ct) => {
                    if ct.len() != times.len()
                        || ct.iter().zip(&times).any(|(a, b)| (a - b).abs() > 1e-9) {
                        return Err(format!(
                            "observation times for stream '{}' differ from the first \
                             stream; all streams must share identical schedules.",
                            stream_name));
                    }
                }
            }
            let bytes = std::fs::read(data_path)
                .map_err(|e| format!("cannot read data file '{}': {}", data_path, e))?;
            data_hashes.insert((*stream_name).clone(), crate::hashing::sha256_hex(&bytes));
            obs_models.push(obs_model_ir);
            per_stream_obs.push(observations);
        }

        // Capture [estimate].start for the start-rank diagnostic.
        let estimate_starts: HashMap<String, f64> = config.estimate.iter()
            .filter_map(|(name, spec)| spec.start.map(|v| (name.clone(), v)))
            .collect();
        Ok(ResolvedSurveyInputs {
            compiled,
            model_ir_json,
            base_params,
            estimated,
            obs_models,
            per_stream_obs,
            data_hashes,
            fixed: fixed_resolved.into_iter().collect(),
            scenario: config.scenario,
            estimate_starts: if estimate_starts.is_empty() {
                None
            } else {
                Some(estimate_starts)
            },
        })
    } else {
        // Inline mode: --estimate flags + --data (already validated).
        let data_path = a.data.as_ref().unwrap().to_string_lossy().into_owned();
        let (mut model_pre, model_ir_json) = crate::util::load_model(&model_path)?;

        // gh#92 inline-mode counterpart of the fit-aware fall-back
        // above: parameters named in `--estimate NAME=LO:HI` flags
        // are about to be LHS-sampled, so requiring a value before
        // the resolver is self-defeating. `EstimateBoundsSpec` has
        // no `start` field (inline only carries bounds), so the
        // fall-back is the bounds midpoint — a defensible "neutral"
        // value that downstream LHS overwrites per-point.
        for est in &a.estimate {
            if let Some(p) = model_pre.parameters.iter_mut().find(|p| p.name == est.name) {
                if p.value.resolved_value().is_none() {
                    p.value = p.value.with_value(0.5 * (est.lo + est.hi));
                }
            }
        }

        // Inline mode: drive the unified resolver. Inline `--fixed`
        // entries become `fixed_cli`; the scenario flag participates in
        // tier-4 resolution. There is no fit.toml [fixed] / [estimate]
        // in inline mode.
        let fixed_cli_vec: Vec<(String, f64)> = a.fixed.iter()
            .map(|p| (p.name.clone(), p.value)).collect();
        let fixed_files: Vec<std::path::PathBuf> = Vec::new();
        let ftf: IndexMap<String, f64> = IndexMap::new();
        let fte: IndexSet<String> = IndexSet::new();
        let table_files: HashMap<String, std::path::PathBuf> = HashMap::new();
        let resolved = crate::params_resolver::resolve_parameters(
            crate::params_resolver::ParameterInputs {
                model: &model_pre,
                scenario: a.scenario.as_deref(),
                adhoc_enable: &[],
                adhoc_disable: &[],
                fixed_cli: &fixed_cli_vec,
                fixed_files: &fixed_files,
                fit_toml_fixed: &ftf,
                fit_toml_estimate: &fte,
                table_files: &table_files,
            },
        ).map_err(|e| e.to_string())?;
        crate::params_resolver::print_warnings(&resolved);
        let model = resolved.model.clone();
        let fixed_map: HashMap<String, f64> = fixed_cli_vec.iter().cloned().collect();

        let compiled = Arc::new(CompiledModel::new(model.clone())
            .map_err(|e| format!("compile error: {:?}", e))?);
        let mut base_params = compiled.default_params.clone();
        for (name, &v) in &fixed_map {
            if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                base_params[idx] = v;
            }
        }

        // Build EstimatedParam vector from --estimate flags.
        let specs: Vec<ParamSpec> = a.estimate.iter().map(|e| ParamSpec {
            name: e.name.clone(),
            rw_sd: None,
            transform: None,
            ivp: false,
            bounds: Some((e.lo, e.hi)),
        }).collect();
        let estimated = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)?;

        // Inline mode: data is a single file. If the model has one
        // observation, score against it; otherwise treat the file as
        // a wide TSV with one column per declared stream.
        if model.observations.is_empty() {
            return Err("model declares no observations; survey requires \
                an observation block to score against.".into());
        }
        let mut obs_models = Vec::new();
        let mut per_stream_obs = Vec::new();
        let mut data_hashes: HashMap<String, String> = HashMap::new();
        let bytes = std::fs::read(&data_path)
            .map_err(|e| format!("cannot read --data file '{}': {}", data_path, e))?;
        let data_hash = crate::hashing::sha256_hex(&bytes);

        // Sort observation names for canonical ordering.
        let mut sorted_obs: Vec<&ir::observation::ObservationModel> =
            model.observations.iter().collect();
        sorted_obs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut canonical_times: Option<Vec<f64>> = None;
        let time_opts = crate::caltime_load::TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt: 1.0,
            t_start: compiled.model.simulation.t_start,
            format: crate::caltime_load::TimeFormat::Auto,
        };
        for obs in sorted_obs {
            let observations = load_observations_from_tsv(&data_path, obs, &time_opts)?;
            let times: Vec<f64> = observations.iter().map(|o| o.time).collect();
            match &canonical_times {
                None => canonical_times = Some(times),
                Some(ct) => {
                    if ct.len() != times.len()
                        || ct.iter().zip(&times).any(|(x, y)| (x - y).abs() > 1e-9) {
                        return Err(format!(
                            "observation times for stream '{}' differ from the first \
                             stream; all streams must share identical schedules.",
                            obs.name));
                    }
                }
            }
            data_hashes.insert(obs.name.clone(), data_hash.clone());
            obs_models.push(obs.clone());
            per_stream_obs.push(observations);
        }

        Ok(ResolvedSurveyInputs {
            compiled,
            model_ir_json,
            base_params,
            estimated,
            obs_models,
            per_stream_obs,
            data_hashes,
            fixed: fixed_map,
            scenario: a.scenario.clone(),
            // Inline mode has no fit.toml [estimate].start values.
            estimate_starts: None,
        })
    }
}

/// Load (time, value) pairs from a TSV by NAME — the declared `time` column is
/// the axis (by-name-time flip), and `scored` is the value column. Both must
/// match the file header exactly. There is no positional fallback (G1): a
/// typo'd/wrong-cased header is a located error, not a silent bind.
fn load_observations_from_tsv(
    path: &str,
    obs: &ir::observation::ObservationModel,
    opts: &crate::caltime_load::TimeOpts,
) -> Result<Vec<Observation>, String> {
    let time_col = crate::pfilter::obs_time_column(obs)?;
    let raw = crate::pfilter::load_data_tsv_column(path, time_col, &obs.scored, opts)?;
    Ok(raw.into_iter().map(|o| Observation { time: o.time, value: o.value }).collect())
}

// ─── Per-point evaluation ────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct LandscapeRow {
    point_id: usize,
    /// Parameter values at the natural scale, in `estimated` order.
    param_values: Vec<f64>,
    loglik: f64,
    loglik_se: f64,
    /// Mean ESS across observation times. NaN when --eval simulate.
    mean_ess: f64,
    n_replicates: usize,
}

#[allow(clippy::too_many_arguments)]
fn eval_point_pfilter(
    process: &ChainBinomialProcess,
    obs_model: &(dyn ObservationModel<ParticleState> + Sync),
    params: &[f64],
    estimated: &[EstimatedParam],
    draw: &[EstimatedParam],
    n_particles: usize,
    n_replicates: usize,
    dt: f64,
    t_start: f64,
    seed_base: u64,
    point_id: usize,
) -> LandscapeRow {
    // Per-point per-replicate seed, derived from (seed_base, point_id, rep).
    let mut log_liks: Vec<f64> = Vec::with_capacity(n_replicates);
    let mut ess_values: Vec<f64> = Vec::new();
    for rep in 0..n_replicates {
        let seed = derive_point_seed(seed_base, point_id, rep);
        let cfg = SMCConfig {
            n_particles,
            dt,
            t_start,
            skip_first_obs_from_loglik: false,
            record_ancestry: false,
            record_prequential: false,
            // gh#241: deterministic compute budget (engine default). No
            // wall-clock watchdog — the content-addressed `surveys/` landscape
            // is reproducible across machines.
            max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        };
        match bootstrap_filter(process, obs_model, params, &cfg, seed) {
            Ok(PFilterResult { log_likelihood, ess_trace, .. }) => {
                log_liks.push(log_likelihood);
                if !ess_trace.is_empty() {
                    let mean = ess_trace.iter().sum::<f64>() / ess_trace.len() as f64;
                    ess_values.push(mean);
                }
            }
            Err(_) => {
                log_liks.push(f64::NEG_INFINITY);
            }
        }
    }
    let n = log_liks.len() as f64;
    let log_n = n.ln();
    let logmeanexp = log_sum_exp(&log_liks) - log_n;
    // Replicate SE on the natural log-likelihood scale.
    let se = if log_liks.iter().any(|x| !x.is_finite()) || log_liks.len() < 2 {
        0.0
    } else {
        let mean = log_liks.iter().sum::<f64>() / n;
        let var = log_liks.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        (var / n).sqrt()
    };
    let mean_ess = if ess_values.is_empty() {
        f64::NAN
    } else {
        ess_values.iter().sum::<f64>() / ess_values.len() as f64
    };
    LandscapeRow {
        point_id,
        param_values: estimated.iter()
            .map(|spec| draw.iter().find(|d| d.index == spec.index)
                .map(|d| d.initial)
                .unwrap_or(params[spec.index]))
            .collect(),
        loglik: logmeanexp,
        loglik_se: se,
        mean_ess,
        n_replicates,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_point_simulate(
    compiled: &sim::CompiledModel,
    obs_model: &MultiStreamObsModel,
    obs_times: &[f64],
    params: &[f64],
    estimated: &[EstimatedParam],
    draw: &[EstimatedParam],
    dt: f64,
    point_id: usize,
) -> LandscapeRow {
    // Phase 1 of the ODE-inference proposal reroutes `--eval simulate`
    // through `compute_ode_loglik`: pre-Phase 1 the path was a 1-particle
    // bootstrap filter on `ChainBinomialProcess`, which returned a
    // 1-sample MC estimate of `p(y|θ)` under the stochastic process
    // kernel. The new path returns the deterministic-skeleton
    // `p(y|θ, ODE_skeleton)` — a different statistical object that
    // matches the flag's name. For typhoid-class N the two converge to
    // sub-nat; for small populations the discrete-event vs continuous-
    // trajectory difference is larger and the user should prefer
    // `--eval pfilter`. SE remains undefined (single deterministic
    // trajectory; no replicates) → reported as 0.0.
    let loglik = crate::fit::runner::compute_ode_loglik(
        compiled, obs_model, obs_times, dt, params,
    )
    .unwrap_or(f64::NEG_INFINITY);
    LandscapeRow {
        point_id,
        param_values: estimated.iter()
            .map(|spec| draw.iter().find(|d| d.index == spec.index)
                .map(|d| d.initial)
                .unwrap_or(params[spec.index]))
            .collect(),
        loglik,
        loglik_se: 0.0,
        mean_ess: f64::NAN,
        n_replicates: 1,
    }
}

/// Per-(point, rep) seed mixer — the canonical cell-seed mix shared with
/// `engine::run_job` (see [`crate::util::mix_cell_seed`]).
fn derive_point_seed(base: u64, point_id: usize, rep: usize) -> u64 {
    crate::util::mix_cell_seed(base, point_id as u64, rep as u64)
}

// ─── TSV writer ──────────────────────────────────────────────────────────────

fn write_landscape_tsv(
    path: &Path,
    rows: &[LandscapeRow],
    estimated: &[EstimatedParam],
    eval: SurveyEvalMethod,
    run_hash: &str,
) -> std::io::Result<()> {
    use std::io::Write as _;
    let tmp = path.with_extension("tsv.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        // Comment header (TSV consumers tolerant of leading `#` lines).
        writeln!(f, "# camdl survey landscape; run_hash={}; version={}",
            run_hash, crate::version::VERSION_SHORT)?;
        writeln!(f, "# eval={}; n_points={}", eval.as_str(), rows.len())?;
        // Header row: param columns, then loglik / loglik_se /
        // (mean_ess if pfilter) / n_replicates / point_id / loglik_type.
        // `loglik_type` is appended last (a grid is a single marginal class,
        // so the column is constant) — appended, never inserted, so a
        // positional reader of the existing columns is unaffected (gh#280).
        let loglik_type = crate::fit::loglik::LoglikType::Marginal.tag();
        let mut cols: Vec<String> = estimated.iter().map(|ep| ep.name.clone()).collect();
        cols.push("loglik".into());
        cols.push("loglik_se".into());
        if eval == SurveyEvalMethod::Pfilter {
            cols.push("mean_ess".into());
        }
        cols.push("n_replicates".into());
        cols.push("point_id".into());
        cols.push("loglik_type".into());
        writeln!(f, "{}", cols.join("\t"))?;
        for r in rows {
            let mut fields: Vec<String> = r.param_values.iter()
                .map(|v| format_float(*v)).collect();
            fields.push(format_float(r.loglik));
            fields.push(format_float(r.loglik_se));
            if eval == SurveyEvalMethod::Pfilter {
                fields.push(format_float(r.mean_ess));
            }
            fields.push(r.n_replicates.to_string());
            fields.push(r.point_id.to_string());
            fields.push(loglik_type.to_string());
            writeln!(f, "{}", fields.join("\t"))?;
        }
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn format_float(v: f64) -> String {
    if v.is_nan() { "NaN".into() }
    else if v == f64::INFINITY  { "Inf".into() }
    else if v == f64::NEG_INFINITY { "-Inf".into() }
    else { format!("{}", v) }
}

// ─── SE warning ──────────────────────────────────────────────────────────────

fn emit_se_warning(rows: &[LandscapeRow]) {
    // Doucet et al. 2015 (Biometrika): per-point loglik SE > ~1.7 nats
    // makes pseudo-marginal MCMC ranks unreliable. Survey isn't doing
    // PMMH but the same bar applies to ranking N points.
    const DOUCET: f64 = 1.7;
    let finite_se: Vec<f64> = rows.iter()
        .map(|r| r.loglik_se)
        .filter(|s| s.is_finite()).collect();
    if finite_se.is_empty() { return; }
    let n = finite_se.len();
    let above = finite_se.iter().filter(|&&s| s > DOUCET).count();
    let pct = 100.0 * (above as f64) / (n as f64);
    if pct > 25.0 {
        eprintln!(
            "warning: {:.0}% of survey points have loglik_se > {} nats — \
             ranks for those points are unreliable. Consider:\n  \
             --eval-replicates 5  (3x compute, ~sqrt(5/3) variance reduction)\n  \
             --eval-particles 500 (2.5x compute, lower per-replicate variance)",
            pct, DOUCET);
    }
}

// ─── Silent-miss diagnostics: bound clustering & start rank ──────────────────
//
// Survey's documented "silent miss case" (the bounds box excludes the
// true basin and the user has no signal that this happened) gets two
// concrete mitigations at run end:
//
//  1. Bound clustering: scan the top-K rows for parameters whose
//     median value pins near a bound (within 5% of the bound's range).
//     Real reproducer: typhoid SIRC β_veryhigh pinned at lower bound,
//     ξ_a15plus at upper — both invisible from loglik alone.
//  2. Start rank: in fit-aware mode, locate where the user's
//     [estimate].start = values fall in the LHS landscape. A start
//     in the bottom 90% is a "your prior best-guess looks worse
//     than 90% of random draws" sanity flag.
//
// Both are advisory — neither aborts the run. Both surface data the
// user can act on without re-running compute.

const BOUND_PIN_FRACTION: f64 = 0.05;

fn emit_bound_clustering_warning(
    rows: &[LandscapeRow],
    estimated: &[EstimatedParam],
    top_pct: f64,
) {
    let finite: Vec<&LandscapeRow> = rows.iter()
        .filter(|r| r.loglik.is_finite()).collect();
    if finite.is_empty() { return; }
    let k = ((finite.len() as f64) * top_pct).ceil() as usize;
    let k = k.max(1).min(finite.len());
    // rows are already sorted by loglik desc; first k are the top.
    let top: &[&LandscapeRow] = &finite[..k];

    let mut warnings: Vec<String> = Vec::new();
    for (i, spec) in estimated.iter().enumerate() {
        if !spec.lower.is_finite() || !spec.upper.is_finite() { continue; }
        let mut vals: Vec<f64> = top.iter().map(|r| r.param_values[i]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = vals[vals.len() / 2];

        // For Log-typed bounds (both > 0) measure the pin fraction in
        // log space — otherwise a Log-bounds [1e-5, 1.0] median of
        // 1e-4 looks "near the lower bound" on linear scale even
        // though it's the geometric midpoint.
        let (frac_to_lo, frac_to_hi) = if matches!(spec.transform,
            sim::inference::types::Transform::Log { .. })
            && spec.lower > 0.0 && spec.upper > 0.0
        {
            let log_lo = spec.lower.ln();
            let log_hi = spec.upper.ln();
            let log_med = median.max(spec.lower).ln();
            let span = log_hi - log_lo;
            ((log_med - log_lo) / span, (log_hi - log_med) / span)
        } else {
            let span = spec.upper - spec.lower;
            ((median - spec.lower) / span, (spec.upper - median) / span)
        };

        if frac_to_lo < BOUND_PIN_FRACTION {
            warnings.push(format!(
                "  '{}' top-{:.0}% median = {} pinned near LOWER bound {} \
                 (within {:.0}% of bound width)",
                spec.name, top_pct * 100.0,
                format_param_value_short(median), format_param_value_short(spec.lower),
                frac_to_lo * 100.0));
        } else if frac_to_hi < BOUND_PIN_FRACTION {
            warnings.push(format!(
                "  '{}' top-{:.0}% median = {} pinned near UPPER bound {} \
                 (within {:.0}% of bound width)",
                spec.name, top_pct * 100.0,
                format_param_value_short(median), format_param_value_short(spec.upper),
                frac_to_hi * 100.0));
        }
    }

    if !warnings.is_empty() {
        eprintln!(
            "warning: top-{:.0}% of survey points cluster near declared \
             bounds for these parameters:",
            top_pct * 100.0);
        for w in &warnings { eprintln!("{}", w); }
        eprintln!(
            "  this can mean (a) the true basin extends outside your \
             bounds box, or (b) the bounds reflect informed priors and \
             the basin really is at the edge. Consider widening the \
             relevant bounds and re-running to disambiguate.");
    }
}

fn emit_start_rank_report(
    rows: &[LandscapeRow],
    estimated: &[EstimatedParam],
    starts: &HashMap<String, f64>,
) {
    let finite: Vec<&LandscapeRow> = rows.iter()
        .filter(|r| r.loglik.is_finite()).collect();
    if finite.is_empty() || starts.is_empty() { return; }
    let n = finite.len();

    // Find the row whose param vector is closest to the start vector
    // (Euclidean distance on the natural-scale, normalised by bound
    // width per dim — same scale-invariance trick the LHS sampler
    // uses). Report its rank.
    let target: Vec<Option<f64>> = estimated.iter()
        .map(|spec| starts.get(&spec.name).copied()).collect();
    if target.iter().all(|x| x.is_none()) { return; }

    let mut best_rank = 0usize;
    let mut best_dist = f64::INFINITY;
    for (rank, row) in finite.iter().enumerate() {
        let mut d2 = 0.0_f64;
        for (i, spec) in estimated.iter().enumerate() {
            let Some(s) = target[i] else { continue };
            let span = (spec.upper - spec.lower).max(1e-30);
            let dx = (row.param_values[i] - s) / span;
            d2 += dx * dx;
        }
        if d2 < best_dist {
            best_dist = d2;
            best_rank = rank;
        }
    }

    let pct = 100.0 * (best_rank as f64) / (n as f64);
    if pct > 90.0 {
        eprintln!(
            "warning: closest LHS draw to your [estimate].start values \
             ranks {} of {} ({:.0}th percentile from the top) — your \
             seeded best-guess falls in the bottom {}% of survey points. \
             Likely causes: bounds exclude the basin, prior best-guess \
             is in a low-loglik region, or the model is misspecified.",
            best_rank + 1, n, 100.0 - pct, (100.0 - pct).round() as i64);
    } else if pct > 50.0 {
        eprintln!(
            "note: closest LHS draw to your [estimate].start values \
             ranks {} of {} (top {:.0}%) — outside the top half of \
             the survey. The basin LHS found may be a better starting \
             point for refine; consider passing scout's MLE via \
             starts_from instead of a hardcoded start.",
            best_rank + 1, n, pct);
    }
}

/// Compact float formatter for warning messages. Keeps log-scale
/// values readable (1e-5 not 0.00001) and clamps precision.
fn format_param_value_short(v: f64) -> String {
    if v.abs() >= 1e-3 && v.abs() < 1e6 {
        format!("{:.4}", v)
    } else {
        format!("{:.3e}", v)
    }
}

// ─── summary.json ────────────────────────────────────────────────────────────

fn write_summary_json(
    path: &Path,
    rows: &[LandscapeRow],
    a: &crate::args::SurveyArgs,
    eval_method: SurveyEvalMethod,
    d: usize,
) -> std::io::Result<()> {
    let finite_lls: Vec<f64> = rows.iter()
        .map(|r| r.loglik)
        .filter(|x| x.is_finite()).collect();
    let top_loglik = finite_lls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let top_loglik = if top_loglik == f64::NEG_INFINITY { None } else { Some(top_loglik) };

    let finite_ses: Vec<f64> = rows.iter()
        .map(|r| r.loglik_se)
        .filter(|x| x.is_finite()).collect();
    let se_q = quartiles(&finite_ses);

    // Top-K (default 5) param-value ranges, just for the summary
    // (visualization is via the HTML).
    let top_k = 5;
    let top_rows: Vec<&LandscapeRow> = rows.iter().take(top_k).collect();

    let summary = serde_json::json!({
        "n_points": rows.len(),
        "dimensions": d,
        "eval_method": eval_method.as_str(),
        "eval_particles": a.eval_particles,
        "eval_replicates": a.eval_replicates,
        "seed": a.seed,
        "top_loglik": top_loglik,
        "loglik_se_quartiles": se_q,
        "top_k_count": top_rows.len(),
        "n_finite_loglik": finite_lls.len(),
    });
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&summary)
        .map_err(std::io::Error::other)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn quartiles(values: &[f64]) -> serde_json::Value {
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let pick = |q: f64| -> f64 {
        let idx = ((n as f64 - 1.0) * q).round() as usize;
        v[idx.min(n - 1)]
    };
    serde_json::json!({
        "min":    v[0],
        "q25":    pick(0.25),
        "median": pick(0.50),
        "q75":    pick(0.75),
        "max":    v[n - 1],
        "n":      n,
    })
}

// ─── HTML rendering (stub — fleshed out in landscape_html commit) ───────────

fn render_landscape_html(
    _landscape_path: &Path,
    html_path: &Path,
    meta: &crate::landscape_html::LandscapeMeta,
) -> Result<(), String> {
    crate::landscape_html::render(html_path, meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landscape_tsv_header_includes_estimated_and_diagnostic_columns() {
        // Column order: estimated names (in declaration order), then
        // loglik, loglik_se, mean_ess (pfilter only), n_replicates, point_id.
        use sim::inference::types::{Transform, EstimatedParam};
        let estimated = vec![
            EstimatedParam {
                name: "beta".into(), index: 0, initial: 0.5, rw_sd: 0.1,
                transform: Transform::None, lower: 0.0, upper: 1.0,
                rw_sd_auto: false, ivp: false,
            },
            EstimatedParam {
                name: "gamma".into(), index: 1, initial: 0.2, rw_sd: 0.1,
                transform: Transform::None, lower: 0.01, upper: 0.5,
                rw_sd_auto: false, ivp: false,
            },
        ];
        let rows = vec![
            LandscapeRow {
                point_id: 0,
                param_values: vec![0.3, 0.15],
                loglik: -123.4,
                loglik_se: 0.5,
                mean_ess: 180.0,
                n_replicates: 3,
            },
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("landscape.tsv");
        write_landscape_tsv(&path, &rows, &estimated, SurveyEvalMethod::Pfilter, "deadbeef")
            .unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        // First two lines are comments.
        assert!(lines[0].starts_with("# camdl survey"));
        assert!(lines[1].starts_with("# eval="));
        // Header: params, loglik, loglik_se, mean_ess, n_replicates, point_id,
        // then the appended loglik_type column (gh#280).
        let header: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(header,
            vec!["beta", "gamma", "loglik", "loglik_se", "mean_ess",
                 "n_replicates", "point_id", "loglik_type"]);
        // Data row: the existing columns are unmoved (point_id still at 6),
        // and the appended column carries the marginal class.
        let row: Vec<&str> = lines[3].split('\t').collect();
        assert_eq!(row.len(), 8);
        assert_eq!(row[6], "0");
        assert_eq!(row[7], "marginal");
    }

    #[test]
    fn landscape_tsv_simulate_omits_mean_ess() {
        use sim::inference::types::{Transform, EstimatedParam};
        let estimated = vec![EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.5, rw_sd: 0.1,
            transform: Transform::None, lower: 0.0, upper: 1.0,
            rw_sd_auto: false, ivp: false,
        }];
        let rows = vec![
            LandscapeRow {
                point_id: 0,
                param_values: vec![0.3],
                loglik: -123.4,
                loglik_se: 0.0,
                mean_ess: f64::NAN,
                n_replicates: 1,
            },
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("landscape.tsv");
        write_landscape_tsv(&path, &rows, &estimated, SurveyEvalMethod::Simulate, "h")
            .unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        let header: Vec<&str> = lines[2].split('\t').collect();
        assert_eq!(header,
            vec!["beta", "loglik", "loglik_se", "n_replicates", "point_id", "loglik_type"]);
    }

    #[test]
    fn quartiles_handles_small_inputs() {
        // Empty → null.
        assert!(quartiles(&[]).is_null());
        // Single value.
        let q = quartiles(&[1.0]);
        assert_eq!(q.get("min").and_then(|v| v.as_f64()), Some(1.0));
        assert_eq!(q.get("max").and_then(|v| v.as_f64()), Some(1.0));
        // Standard 5-number summary on a known sequence.
        let q = quartiles(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(q.get("median").and_then(|v| v.as_f64()), Some(3.0));
    }

    #[test]
    fn point_seed_distinguishes_points_and_reps() {
        // (point, rep) → distinct seeds. Identical (point, rep) →
        // identical seeds (deterministic).
        let s_a = derive_point_seed(42, 0, 0);
        let s_b = derive_point_seed(42, 0, 0);
        assert_eq!(s_a, s_b);
        assert_ne!(derive_point_seed(42, 0, 0), derive_point_seed(42, 0, 1));
        assert_ne!(derive_point_seed(42, 0, 0), derive_point_seed(42, 1, 0));
        assert_ne!(derive_point_seed(42, 0, 0), derive_point_seed(43, 0, 0));
    }
}
