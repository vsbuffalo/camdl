//! Shared chain-running logic for all fit stages.
//!
//! Handles: model loading, EstimatedParam construction from fit.toml,
//! obs_loglik construction from IR observation model, chain execution,
//! chain-agreement (Â) computation, and MAD-based auto rw_sd calibration.

use crate::fit::loglik_eval;
use crate::fit::state::FitState;
use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        if2::{run_if2_with_progress, IF2Config, EstimatedParam, IF2Result, Observation, Transform},
        pmmh::Prior,
        prior::{Density, TransformReq},
        diagnostic::{DiagnosticCollector, DiagnosticKind},
    },
    rng::StatefulRng,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Everything needed to run IF2 chains, built from fit.toml + optional prior state.
/// One observation data stream with its projection and likelihood.
pub struct ObsStream {
    pub name: String,
    /// Resolved projection (incidence / prevalence / snapshot expression)
    /// built from the IR observation block.
    pub projection: sim::inference::multi_stream_obs::StreamProjection,
    pub obs_model_ir: ir::observation::ObservationModel,
    /// Dense placeholder view (a hole shows as value 0) used only by the
    /// startup diagnostics and the canonical-times path. NOT load-bearing for
    /// scoring — the authoritative per-grid-time cells (with holes) are in
    /// `cells`. Times are always correct here regardless of holes.
    pub data: Vec<Observation>,
    /// Authoritative per-grid-time observation cells, parallel to `data` and to
    /// `obs_times`. `None` = a hole (the `NA` token): its time stays in the
    /// grid (so the incidence accumulator still resets at its index) but it
    /// carries no value (no likelihood term). `Some(ObsCell::Scalar(v))` =
    /// observed value `v`. Threaded into the obs model so the already
    /// hole-correct scoring seam (`MultiStreamObsModel`) handles missing values.
    pub cells: Vec<Option<sim::inference::ObsCell>>,
    /// Per-observation auxiliary data (binomial `n = tested`, person-time
    /// offset), parallel to `cells`; each row a name→value list the likelihood
    /// reads by `Expr::ObsColumnRef`. Empty inner vec when the likelihood
    /// references no aux column or the cell is a hole. (§3, §6.1.)
    pub aux: Vec<Vec<(String, f64)>>,
}

pub struct FitRunConfig {
    pub compiled: Arc<CompiledModel>,
    pub model: ir::Model,
    /// Pre-filter snapshot — every intervention and event declared in the
    /// model file, whether or not the active scenario enabled it. Used by
    /// `print_scheduled_actions_summary` to show a "N active of M declared"
    /// block on startup.
    pub model_declared: ir::Model,
    pub model_ir_json: String,
    pub base_params: Vec<f64>,
    /// Names of all parameters, parallel to `base_params`. Built from
    /// `model.parameters` at setup time. Used by the PMMH / PGAS
    /// hierarchical-prior env to resolve `Expr::Param(name)` references
    /// against current values. Wave 2 / #3 Gate 3a.
    pub param_names: Vec<String>,
    pub estimated_params: Vec<EstimatedParam>,
    /// Canonical observation times (shared across all streams).
    pub observations: Vec<Observation>,
    /// Per-stream data. For single-stream models, len() == 1.
    pub streams: Vec<ObsStream>,
    pub if2_config: IF2Config,
    pub n_chains: usize,
    pub seed: u64,
    /// IC-free inference flag. When true, the PF/IF2/PGAS log-likelihood
    /// accumulation skips the first observation (y₁ is still used to
    /// weight and resample — that's how the initial state gets pinned).
    /// Mirrors `FitConfigV2::ic_free`.
    /// Flows into `SMCConfig.skip_first_obs_from_loglik`. See
    /// docs/dev/proposals/2026-04-18-ic-free-inference.md.
    pub ic_free: bool,
    /// Clean-evaluation re-scoring config (Step 4 plumbing for §Proposal 1).
    /// Set per stage at the `camdl fit run` dispatch site (CLI overrides
    /// over stage TOML); legacy `camdl fit scout`/`fit refine` use the
    /// `Default` (4000 × 8, logmeanexp). Consumed by Step 5.
    pub loglik_eval: super::config_v2::LoglikEvalConfig,
    /// Compound scout-convergence gate config (Step 4 plumbing for
    /// §Proposal 3). Same per-stage override semantics as `loglik_eval`.
    /// Consumed by Step 8.
    pub gate: super::config_v2::GateConfig,
}

/// Result of running multiple IF2 chains.
///
/// `best_chain` / `best_loglik` are the clean-eval winner — each
/// chain's IF2 final-iteration mean re-scored with M high-particle
/// PF replicates and combined via logmeanexp on the likelihood scale
/// (matches pomp's `coef(mif2_out)` + `pfilter` workflow; Ionides et
/// al. 2015 PNAS). They no longer reflect in-run noisy
/// `IF2Result::final_loglik` argmax — that selection was upward-biased
/// from argmaxing over noisy in-run PF estimates. The full per-chain
/// table lives in `loglik_eval.per_chain`; consumers needing the
/// winner's θ̂ / SE read from
/// `loglik_eval.per_chain[overall_winner_idx]`.
pub struct ChainResults {
    pub results: Vec<(usize, IF2Result)>,
    pub best_chain: usize,
    pub best_loglik: f64,
    pub chain_agreement: HashMap<String, f64>,
    pub loglik_eval: super::loglik_eval::LoglikEvalOutcome,
}

impl FitRunConfig {
    /// Build from a v2 fit.toml, optionally overriding from a prior fit_state.
    ///
    /// `cooling_target_iters` is IF2-specific — for non-IF2 stages
    /// (PGAS / PMMH / PFilter), passing `n_iterations` matches the
    /// pre-2026-04-30 behavior. The IF2 dispatch site reads it from
    /// `Stage::IF2.cooling_target_iters` (default 50).
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        fit: &super::config_v2::FitConfigV2,
        prior_state: Option<&FitState>,
        n_chains: usize,
        n_particles: usize,
        n_iterations: usize,
        cooling: f64,
        cooling_target_iters: usize,
        seed: u64,
        random_starts: bool,
    ) -> Result<Self, String> {
        // Load model. Prefer the pre-compiled IR threaded in by
        // `cmd_fit_run_v2` (compiled once up front) so a multi-stage / swept
        // fit doesn't re-invoke camdlc per (cell × sweep point × stage); fall
        // back to `model.camdl` (recompiles) when no compiled IR was provided
        // — the identity path is always `model.camdl`, not this.
        let model_path = fit.compiled_ir.as_deref().unwrap_or(&fit.model.camdl);
        let (mut model_pre, model_ir_json) = crate::util::load_model(model_path)?;
        // gh#616: resolve the model's observation anchors from THIS fit's bound
        // data, before anything compiles. Not optional: `ode_grad.rs` takes the
        // integration window from `simulation.t_end`, so an unresolved horizon
        // would integrate nothing.
        //
        // `run_fit` normally resolves once up front and repoints `compiled_ir`
        // at the substituted IR, so this is a no-op there. It stays because a
        // `FitRunConfig` can also be built by entry points that do not go
        // through `run_fit` — same window, same function, so resolving again is
        // idempotent rather than a second opinion.
        if crate::obs_anchor::model_is_anchored(&model_pre) {
            let dt0 = model_pre.simulation.dt.unwrap_or(1.0);
            let (first, last) = crate::obs_anchors_from_config(&model_pre, fit, dt0)
                .map_err(|e| {
                    format!("resolving this model's observation anchors from [data]: {e}")
                })?;
            let moved = crate::obs_anchor::substitute(
                &mut model_pre,
                ir::anchor::ObsAnchorTimes { first, last },
            )?;
            crate::obs_anchor::report(&moved, &model_pre);
        }
        // Keep a copy of the unfiltered model so the startup diagnostic
        // can show what was declared vs what's active. Cheap clone — the
        // intervention list is small.
        let model_declared = model_pre.clone();

        // Per spec §14.4, toggleable interventions default OFF; events
        // (always_active) stay on unless explicitly disabled. If neither
        // scenario nor enable/disable are set in fit.toml, interventions
        // are cleared (spec default). The unified resolver handles this
        // contract — see 2026-05-25 CLI UX rev 2 proposal.
        if fit.scenario.is_some()
            && (!fit.enable.is_empty() || !fit.disable.is_empty())
        {
            return Err("fit.toml: `scenario` is mutually exclusive \
                with `enable`/`disable`. Use one approach.".into());
        }

        // Resolve fixed up-front (file load + inline overlay, or
        // scenario lookup via gh#33's `from_scenario`). The
        // pre-processor produces the IndexMap fed into the unified
        // resolver as `fit_toml_fixed`.
        let mut fixed_cfg = fit.fixed.clone();
        fixed_cfg.expand_from_scenario(&model_pre, &fit.estimate)?;
        let fixed_resolved: indexmap::IndexMap<String, f64> =
            fixed_cfg.resolve_with_model(&model_pre)?;

        // Apply the [estimate].start fall-back BEFORE the resolver so
        // that params with no model default but an [estimate] block
        // can carry an inferred starting value past the resolver's
        // `UnsetRequired` check (gh#34). This mirrors the legacy code
        // path's ordering — start fall-back, then validate — under the
        // new resolver scaffolding. We seed values into a temporary
        // copy of the IR; the resolver still owns the final
        // tier-merge.
        //
        // Priority: fit_state start_values > estimate.start > model default
        //
        // gh#34: when [estimate] entry has no explicit `start =` AND
        // the model param has no value yet (no scenario default, no
        // model-declared `value`), fall back to a Transform-aware
        // uniform draw within bounds.
        for (name, spec) in &fit.estimate {
            if let Some(p) = model_pre.parameters.iter_mut().find(|p| p.name == *name) {
                if p.value.resolved_value().is_none() {
                    let resolved_bounds = spec.bounds.or(p.bounds());
                    let value = spec.start.or_else(|| resolved_bounds.map(|(lo, hi)| {
                        let transform = derive_transform_with_bounds(
                            p,
                            spec.transform.as_ref().map(|t| t.as_str()),
                            (lo, hi),
                        );
                        let log_scale = matches!(transform, Transform::Log { .. });
                        super::init::draw_start_in_bounds(lo, hi, log_scale, seed, name)
                    }));
                    if let Some(v) = value {
                        // Fill the optimiser start without collapsing to a Fixed
                        // constant — keeps the parameter estimated (bounds/prior
                        // intact) so the resolver passes (gh#34) and `default_params`
                        // gets the start.
                        p.value = p.value.with_value(v);
                    }
                }
            }
        }

        // Route value resolution through the unified resolver. fit.toml
        // has no CLI-level --fixed; the [fixed] block becomes
        // `fit_toml_fixed` and the [estimate] keys become
        // `fit_toml_estimate`. The resolver handles tier ordering,
        // intervention filter, bounds, and finite-value checks.
        let fit_toml_estimate: indexmap::IndexSet<String> =
            fit.estimate.keys().cloned().collect();
        let table_files_resolver: std::collections::HashMap<String, std::path::PathBuf> =
            std::collections::HashMap::new();
        // gh#561: a fit's window is the observation times, so a scenario's own
        // `simulate { to }` cannot be honoured here — refuse rather than drop
        // it silently. "Honour it instead" would be actively wrong on this
        // path: the ODE gradient config reads `simulation.t_end`
        // (`sim/src/inference/ode_grad.rs`), so a shortened scenario horizon
        // would truncate integration before the last scored observation.
        crate::util::refuse_scenario_horizon(
            &model_pre, fit.scenario.as_deref(), "fit run",
            "a fit scores at the observation times, so its window comes from \
             the data, not the model horizon",
        )?;
        let resolved = crate::params_resolver::resolve_parameters(
            crate::params_resolver::ParameterInputs {
                model: &model_pre,
                scenario: fit.scenario.as_deref(),
                adhoc_enable: &fit.enable,
                adhoc_disable: &fit.disable,
                scenario_inline_name: None,
                scenario_inline_set: &[],
                scenario_inline_scale: &[],
                point_overrides: &[],
                fixed_cli: &[],
                fixed_files: &[],
                fit_toml_fixed: &fixed_resolved,
                fit_toml_estimate: &fit_toml_estimate,
                table_files: &table_files_resolver,
            },
        ).map_err(|e| e.to_string())?;
        crate::params_resolver::print_warnings(&resolved);
        let model = resolved.model.clone();

        let compiled = CompiledModel::new(model.clone())
            .map_err(|e| format!("compile error: {:?}", e))?;
        let mut base_params = compiled.default_params.clone();

        // Priority: prior_state > estimate.start > fixed > model default.
        // `base_params` is the single source of truth for IF2's starting
        // point: run_if2_with_progress initialises its particle cloud
        // from `base_params`, not from `EstimatedParam::initial`. If
        // prior_state is applied before est.start (as was the case
        // before 2026-04-18), the est.start write silently overwrites
        // the scout-best values, and `init_mle = "scout"` becomes a
        // no-op for refine's iter-0 parameters. See
        // docs/dev/incidents/2026-04-18-starts-from-scout-ignored.md.

        // 1. Apply estimate start values to base_params (override model defaults).
        for (name, spec) in &fit.estimate {
            if let Some(start) = spec.start {
                if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                    base_params[idx] = start;
                }
            }
        }
        // 2. Apply fixed numeric values (override model defaults).
        for (name, &v) in &fixed_resolved {
            if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                base_params[idx] = v;
            }
        }
        // 3. Apply prior_state last so it wins over config start/fixed.
        //    This is what makes `init_mle = "scout"` actually seed
        //    the IF2 search from scout's best MLE.
        if let Some(state) = prior_state {
            for (name, &value) in &state.start_values {
                if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                    base_params[idx] = value;
                }
            }
        }

        // Build EstimatedParam specs
        let if2_params = build_if2_params(
            &fit.estimate, prior_state, &model, &compiled, &base_params, random_starts, seed,
        )?;

        // Load data — one or more observation streams (real-data only;
        // synthetic-data fits route through a generator before this path).
        // Resolve any single-file shorthand (`[data] file = "..."`) into
        // the canonical per-stream map by mapping every model-declared
        // stream name to that file. From here on the loop is the same.
        let dt = fit.config.dt;
        let data_spec = fit.data_spec()?;
        // `--data`/`[data.observations]` keys by the `from <label>` SOURCE
        // (defaults to the stream name; §2.4). Resolve against the distinct
        // source labels so several streams can share one wide file.
        let mut source_labels: Vec<String> = model.observations.iter()
            .map(|o| o.source.clone()).collect();
        source_labels.sort();
        source_labels.dedup();
        let effective = data_spec.effective_observations(&source_labels)?;
        if effective.is_empty() {
            return Err(
                "fit.toml [data] resolves to zero observation streams. Either \
                 set `[data] file = \"<path>\"` (one wide TSV) or fill \
                 [data.observations] (per-source paths).".into());
        }

        let time_opts = crate::caltime_load::TimeOpts {
            origin: model.origin.as_deref(),
            time_unit: &model.time_unit,
            dt,
            t_start: compiled.model.simulation.t_start,
            format: crate::caltime_load::TimeFormat::Auto,
        };

        // Resolve the bound observation streams (by source) and load each one's
        // per-observation values + aux, via the single shared seam that pfilter
        // and profile also route through (multi-cadence: each stream keeps its
        // OWN times + cells; `bind` merges to the union below).
        let mut streams =
            resolve_and_load_obs_streams(&model, &compiled, &effective, dt, &time_opts)?;

        // Canonical observations: the sorted-unique UNION of every stream's
        // observation times (multi-cadence, proposal 2026-06-10 §3.3). This is
        // what feeds the filter's substep grid, `n_observations`, the W329
        // first-window guard, the obs-alignment gate, and the single-stream
        // output labels — so it MUST be the union, not stream 0's schedule
        // (else heterogeneous streams silently collapse onto stream 0's dates).
        // The per-stream scored VALUES live in each `ObsStream.cells`; the
        // canonical's `value` is a never-scored placeholder (0.0). `bind`
        // re-derives this same union from each stream's own times below.
        let mut observations: Vec<Observation> = {
            let mut times: Vec<f64> = streams.iter()
                .flat_map(|s| s.data.iter().map(|o| o.time))
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).expect("observation times are finite"));
            times.dedup();
            times.into_iter().map(|time| Observation { time, value: 0.0 }).collect()
        };

        // gh#134 / multi-cadence Phase 3: PER-STREAM conditioning + the W329
        // wide-first-window enforcer, both keyed on each stream's
        // observation-block label (its IR `source`). The conditioning window is
        // EXPLICIT — there is no automatic / inferred boundary; a late-starting
        // incidence stream that resolves to no `condition_from` HARD-ERRORS,
        // naming the fix. The boundary, when given, is resolved per stream and
        // prepended as a LEADING reset-only HOLE to THAT stream's data/cells/aux
        // and added to the canonical union grid.
        {
            let union_inserts = apply_conditioning_windows(
                &mut streams,
                fit.condition_from.as_ref(),
                &model,
                compiled.model.simulation.t_start,
                dt,
            )?;

            // Fold the per-stream boundaries into the canonical union grid (the
            // times every algorithm's substep walk reads). Sorted-unique so a
            // boundary shared by streams on the same source appears once.
            if !union_inserts.is_empty() {
                let mut times: Vec<f64> = observations.iter().map(|o| o.time).collect();
                times.extend(union_inserts);
                times.sort_by(|a, b| a.partial_cmp(b).expect("times are finite"));
                times.dedup();
                observations = times.into_iter()
                    .map(|time| Observation { time, value: 0.0 })
                    .collect();
            }
        }

        // (algorithm × obs-alignment) support gate — the fit-dispatch seam.
        // Converts today's SILENT fallbacks into clean errors: `exact` + PGAS
        // would silently snap to a uniform grid; `exact` + off-grid correlated
        // PMMH would silently fall back to fresh RNG, decorrelating the CPM
        // estimator (#17). For valid runs the default resolves to today's
        // behaviour ("exact where supported"), so nothing changes; threading the
        // resolved policy into the filters is Stage 3. (Fires per build; cheap.)
        {
            use crate::run_meta::FitAlgorithm;
            let t_start = compiled.model.simulation.t_start;
            let obs_on_grid = observations.iter().all(|o| {
                let k = ((o.time - t_start) / dt).round();
                ((t_start + k * dt) - o.time).abs() < 1e-9
            });
            for stage in fit.stages.values() {
                if matches!(
                    stage.method_kind(),
                    FitAlgorithm::If2 | FitAlgorithm::Pgas | FitAlgorithm::Pmmh | FitAlgorithm::Pfilter
                ) {
                    let correlated = matches!(
                        stage,
                        crate::fit::config_v2::Stage::PMMH { rho: Some(_), .. }
                    );
                    crate::fit::methods::resolve_obs_alignment(
                        stage.method_kind(),
                        correlated,
                        fit.config.obs_alignment,
                        obs_on_grid,
                    )
                    .map_err(|e| format!("{} stage: {e}", stage.method_name()))?;
                }
            }
        }

        if streams.len() > 1 {
            eprintln!("  {} observation streams: {}",
                streams.len(),
                streams.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", "));
        }

        let ic_free = fit.ic_free.unwrap_or(false);

        // Resolve top-level fit.simplex_groups into sim::SimplexGroup
        // (param names → indices, rw_sds from EstimateSpecV2.rw_sd).
        // Only IF2 currently honours these; PGAS/PMMH/PFilter consume
        // the same FitRunConfig but ignore simplex_groups (validation
        // emits a warning when a non-IF2 stage runs against a fit
        // with simplex groups declared).
        let resolved_simplex_groups = resolve_simplex_groups(
            &fit.simplex_groups, &fit.estimate, &compiled.param_index, &if2_params)?;

        let config = IF2Config {
            n_particles,
            n_iterations,
            cooling_fraction: cooling,
            cooling_target_iters,
            simplex_groups: resolved_simplex_groups,
            dt,
            t_start: compiled.model.simulation.t_start,
            skip_first_obs_from_loglik: ic_free,
            // gh#241: deterministic compute budget (engine default). Fits are
            // content-addressed; with no wall-clock watchdog, theta-hat is a
            // pure function of inputs (reproducible across machines).
            max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        };
        // IC-free precondition (data): y₁ must actually be observed. ic_free
        // conditions the initial state on the first observation (it still
        // reweights/resamples at obs_idx 0, dropping only that term from the
        // accumulated loglik). If y₁ is missing — a hole (`NA`) in every stream
        // at the first observation index — there is nothing to condition on, so
        // ic_free would silently degenerate to *no* initial-state conditioning
        // (a weaker estimand than requested). Checked before the
        // perturb_only_at_t0 precondition: a missing y₁ makes ic_free
        // impossible regardless of the parameter flags.
        if ic_free
            && !streams.is_empty()
            && streams.iter().all(|s| s.cells.first().is_some_and(|c| c.is_none()))
        {
            return Err(
                "ic_free = true conditions the initial state on the first \
                 observation y₁, but y₁ is missing (`NA` / a hole) in every data \
                 stream — there is nothing to condition on. Provide the first \
                 observation, or disable ic_free.".into());
        }
        // IC-free precondition (config): at least one estimated param must be
        // marked perturb_only_at_t0. Under IF2 — the only algorithm ic_free is
        // admitted for (`methods::validate_ic_free`) — the t=0 perturbation
        // moves each particle's θ and each particle then draws its own x₀ from
        // that θ (gh#364), which is what gives the first reweight something to
        // discriminate between. Without such a parameter the first reweight is
        // a no-op and ic-free degenerates to silently dropping y₁. Error at
        // config build so the mistake surfaces before any PF time is spent.
        if ic_free && !if2_params.iter().any(|p| p.perturb_only_at_t0) {
            return Err(
                "ic_free = true requires at least one [estimate.*] entry with \
                 perturb_only_at_t0 = true. Without per-particle variation at \
                 t=0, the first observation cannot discriminate between \
                 particles and ic_free degenerates to dropping the first data \
                 point.\n\n\
                 Example: mark your initial-state parameter perturb_only_at_t0:\
                 \n\n    [estimate]\n    \
                 I0 = { bounds = [1, 500], perturb_only_at_t0 = true }".into());
        }

        let param_names: Vec<String> =
            model.parameters.iter().map(|p| p.name.clone()).collect();
        Ok(FitRunConfig {
            compiled: Arc::new(compiled),
            model,
            model_declared,
            model_ir_json,
            base_params,
            param_names,
            estimated_params: if2_params,
            observations,
            streams,
            if2_config: config,
            n_chains,
            seed,
            ic_free,
            loglik_eval: super::config_v2::LoglikEvalConfig::default(),
            gate: super::config_v2::GateConfig::default(),
        })
    }

    /// Build the inference process. The integrator `dt` is supplied per call to
    /// `step`/`density` (and per rung by the gh#52 Richardson dt-check via the
    /// SMCConfig), so the process itself is dt-agnostic — there is no stored
    /// `fire_steps` to resolve at a particular dt any more (effect firing is
    /// cursor-keyed from the timeline; see `effects::split_due_batch`).
    pub fn build_process(&self) -> sim::inference::ChainBinomialProcess {
        sim::inference::ChainBinomialProcess::new(self.compiled.clone())
    }
    pub fn build_obs_model(&self) -> sim::inference::MultiStreamObsModel {
        let specs = stream_specs_from_obs_streams(&self.streams);
        let (bound, _report) = sim::inference::BoundObs::bind(specs).unwrap_or_else(|report| {
            eprintln!("error: observation data invalid:\n{}", report.render());
            std::process::exit(1);
        });
        sim::inference::MultiStreamObsModel::new(bound, self.compiled.clone())
            .unwrap_or_else(|e| {
                eprintln!("error: observation model construction failed: {:?}", e);
                std::process::exit(1);
            })
    }
    pub fn smc_config(&self) -> sim::inference::traits::SMCConfig {
        sim::inference::traits::SMCConfig {
            n_particles: self.if2_config.n_particles,
            dt: self.if2_config.dt,
            t_start: self.compiled.model.simulation.t_start,
            skip_first_obs_from_loglik: self.ic_free,
            record_ancestry: false,
            record_prequential: false,
            // gh#241: deterministic compute budget (engine default); no
            // wall-clock watchdog. See the IF2Config above.
            max_substeps: sim::inference::degeneracy::ITER_BUDGET,
        }
    }
}

/// ODE step size for `compute_ode_loglik`: prefer `model.simulation.dt`,
/// fall back to the IF2-config's dt (the CLI default of 1.0 unless
/// otherwise set). Either way RK4 substeps are aligned to user-declared
/// output times by the obs-time matching loop above.
pub fn ode_step_dt(config: &FitRunConfig) -> f64 {
    config
        .compiled
        .model
        .simulation
        .dt
        .unwrap_or(config.if2_config.dt)
}

// load_model is now in util.rs

/// Resolve top-level fit.toml `[[simplex_groups]]` entries (param
/// names) into runtime `sim::inference::if2::SimplexGroup` (indices
/// into the model param vector + rw_sds on the log-ratio scale).
///
/// Validation enforced here:
/// - Each member must appear in `[estimate]` (validated upstream by
///   `FitConfigV2::validate`, but defended again here).
/// - Each member must resolve to a model param index.
/// - rw_sd is read from the corresponding `EstimatedParam.rw_sd` (which
///   already encodes auto-derivation when `EstimateSpecV2.rw_sd` is None).
///
/// rw_sd semantics: the IF2 simplex transform perturbs members on the
/// log-ratio scale. The user's `EstimateSpecV2.rw_sd` for a simplex
/// member is taken as-is on that scale (matches pomp's
/// `parameter_trans(barycentric = ...)` + `rw.sd` semantics).
fn resolve_simplex_groups(
    cli_groups: &[super::config_v2::SimplexGroup],
    estimate: &indexmap::IndexMap<String, super::config_v2::EstimateSpecV2>,
    param_index: &HashMap<String, usize>,
    if2_params: &[EstimatedParam],
) -> Result<Vec<sim::inference::if2::SimplexGroup>, String> {
    let mut out = Vec::with_capacity(cli_groups.len());
    for (group_idx, group) in cli_groups.iter().enumerate() {
        let mut indices = Vec::with_capacity(group.params.len());
        let mut rw_sds = Vec::with_capacity(group.params.len());
        for name in &group.params {
            if !estimate.contains_key(name) {
                return Err(format!(
                    "simplex_groups[{}]: member '{}' not in [estimate]. \
                     Members must be free parameters.",
                    group_idx, name));
            }
            let &model_idx = param_index.get(name).ok_or_else(|| format!(
                "simplex_groups[{}]: member '{}' has no model param index \
                 (model load + estimate parity drift?)",
                group_idx, name))?;
            let if2_rw_sd = if2_params.iter()
                .find(|p| p.name == *name)
                .ok_or_else(|| format!(
                    "simplex_groups[{}]: member '{}' missing from \
                     resolved EstimatedParam list (build_if2_params drift?)",
                    group_idx, name))?
                .rw_sd;
            indices.push(model_idx);
            rw_sds.push(if2_rw_sd);
        }
        out.push(sim::inference::if2::SimplexGroup { indices, rw_sds });
    }
    Ok(out)
}

/// Build EstimatedParam specs from v2 [estimate] + optional prior state overrides.
/// Uses the shared build_if2_params_from_specs for core logic, then applies
/// fit-specific overrides (prior state rw_sd, start values, random starts).
fn build_if2_params(
    estimate: &indexmap::IndexMap<String, super::config_v2::EstimateSpecV2>,
    prior_state: Option<&FitState>,
    model: &ir::Model,
    compiled: &CompiledModel,
    base_params: &[f64],
    random_starts: bool,
    seed: u64,
) -> Result<Vec<EstimatedParam>, String> {
    // Build ParamSpecs from v2 [estimate]
    let specs: Vec<ParamSpec> = estimate.iter().map(|(name, est)| {
        // rw_sd priority: prior state > fit.toml explicit > None (auto)
        let rw_sd = prior_state
            .and_then(|s| s.rw_sd.get(name))
            .copied()
            .or(est.rw_sd);
        ParamSpec {
            name: name.clone(),
            rw_sd,
            transform: est.transform.as_ref().map(|t| t.as_str().to_string()),
            perturb_only_at_t0: est.perturb_only_at_t0,
            // Bounds plumbing (gh#42-followup + bounds-optional fix):
            // pass through the Option as-is. `build_if2_params_from_specs`
            // resolves fit.toml > model > unbounded fallback. None now
            // means "no bounds in fit.toml — use model's"; previously it
            // could only mean "non-fit caller (profile/pfilter)".
            bounds: est.bounds,
        }
    }).collect();

    let mut params = build_if2_params_from_specs(model, compiled, base_params, &specs)?;

    // Sort by name for deterministic ordering. IndexMap preserves
    // insertion order, but to keep param order stable across configs
    // that list the same params in different orders (and so resume's
    // z-value mapping survives) we sort by name.
    params.sort_by(|a, b| a.name.cmp(&b.name));

    // Fit-specific: apply start values and random starts
    let mut rng = StatefulRng::new(seed ^ 0xdeadbeef_u64);
    for p in &mut params {
        if random_starts {
            if p.lower.is_finite() && p.upper.is_finite() {
                p.initial = p.lower + rng.uniform() * (p.upper - p.lower);
            } else {
                p.initial *= 1.0 + 0.2 * (rng.uniform() - 0.5);
            }
        } else if let Some(state) = prior_state {
            if let Some(&v) = state.start_values.get(&p.name) {
                p.initial = v;
            }
        } else if let Some(est) = estimate.get(&p.name) {
            if let Some(start) = est.start {
                p.initial = start;
            }
        }
    }

    Ok(params)
}

/// gh#224. Map a raw particle-filter eval `Result` to the inference
/// convention: a **structural** error (`SimError::is_structural` — the
/// model or its configuration cannot run) surfaces unchanged; every
/// other failure is a per-θ excursion or a degenerate / over-budget
/// filter that the MH step rejects, so it collapses to `-∞` (with
/// `FilterStats::failed()`). This is the single classification seam the
/// PMMH / IF2 / PF parameter evaluators share — a structural error must
/// never be silently mistaken for a ruled-out θ (which would yield a
/// degenerate posterior with a successful exit status).
pub fn ruled_out_or_surface(
    r: Result<(f64, super::loglik_eval::FilterStats), sim::error::SimError>,
) -> Result<(f64, super::loglik_eval::FilterStats), sim::error::SimError> {
    match r {
        Ok(v) => Ok(v),
        Err(e) if e.is_structural() => Err(e),
        Err(_) => Ok((f64::NEG_INFINITY, super::loglik_eval::FilterStats::failed())),
    }
}

/// Run a quick pfilter at given params and return the loglik under the
/// inference convention (`Ok(-∞)` = θ ruled out, `Err` = structural —
/// see `ruled_out_or_surface`). Used by scout for the initial_loglik
/// baseline and by the PMMH / IF2 parameter evaluators.
pub fn run_quick_pfilter(
    config: &FitRunConfig, params: &[f64], n_particles: usize, seed: u64,
) -> Result<f64, sim::error::SimError> {
    ruled_out_or_surface(run_quick_pfilter_full(config, params, n_particles, seed))
        .map(|(ll, _)| ll)
}

/// Variant of `run_quick_pfilter` that also returns filter-health
/// statistics (mean / min ESS, the observation step where ESS is
/// worst, and a count of −∞ log-likelihood increments). Cheap to
/// compute since these are already in `PFilterResult.{ess_trace,
/// ll_increments}` — phase 2 of the fit-summary proposal just plumbs
/// them out instead of throwing them away.
///
/// Returns the **raw** filter outcome: `Err` carries the underlying
/// `SimError` so the caller can distinguish structural from recoverable
/// (apply `ruled_out_or_surface` for the inference convention).
pub fn run_quick_pfilter_full(
    config: &FitRunConfig,
    params: &[f64],
    n_particles: usize,
    seed: u64,
) -> Result<(f64, super::loglik_eval::FilterStats), sim::error::SimError> {
    run_quick_pfilter_with_dt(config, params, n_particles, None, seed)
}

/// As `run_quick_pfilter_full` but lets the caller override the
/// integrator step `dt`. `dt_override = None` keeps the fit's
/// `if2_config.dt`. Used by the gh#52 Richardson dt-convergence
/// check at θ̂ to evaluate `loglik(θ̂; dt)` on a halving ladder
/// without rebuilding the run config. Returns the **raw** filter
/// outcome (see `run_quick_pfilter_full`); the gh#110 init-eval guard
/// matches on the raw `Err` to distinguish a `PFDegenerate` bail
/// (→ `BadInit`, skip the chain) from a structural error (→ fatal).
pub fn run_quick_pfilter_with_dt(
    config: &FitRunConfig,
    params: &[f64],
    n_particles: usize,
    dt_override: Option<f64>,
    seed: u64,
) -> Result<(f64, super::loglik_eval::FilterStats), sim::error::SimError> {
    let dt = dt_override.unwrap_or(config.if2_config.dt);
    // gh#53: Process must be built with the same dt the SMCConfig
    // will use, so its internal fire_steps resolves correctly for
    // dt-override calls (gh#52 Richardson ladder).
    let process = config.build_process();
    let obs_model = config.build_obs_model();
    let smc_config = sim::inference::traits::SMCConfig {
        n_particles,
        dt,
        ..config.smc_config()
    };

    let result = sim::inference::bootstrap_filter(&process, &obs_model, params, &smc_config, seed)?;
    let stats = super::loglik_eval::FilterStats::from_pfilter_result(
        &result.ess_trace, &result.ll_increments);
    Ok((result.log_likelihood, stats))
}

/// Which `EstimatedParam` set the preflight table should report, and whether
/// the chains start at different points.
///
/// gh#513: the table used to read `config.estimated_params` unconditionally
/// while the chains ran from `per_chain_params`, so it could disagree with the
/// run in BOTH directions — showing a random draw when the chains used a
/// declared start, and a declared start when the chains used draws. This table
/// is the only thing most users look at to confirm a stage begins where they
/// asked, so a value that is merely *plausible* is worse than none.
///
/// Reports chain 1. With more than one chain the table cannot show them all
/// without becoming unreadable, so it names which chain it is showing and
/// points at `chain_starts.tsv`, which carries the full per-chain truth.
///
/// Split out from the printing so it is testable without capturing stderr.
fn preflight_specs<'a>(
    base: &'a [EstimatedParam],
    per_chain: Option<&'a [Vec<EstimatedParam>]>,
) -> (&'a [EstimatedParam], bool) {
    let Some(chains) = per_chain.filter(|c| !c.is_empty()) else {
        // No per-chain override: every chain runs from `base`, which is then
        // exactly what the table should show.
        return (base, false);
    };
    let shown = &chains[0][..];
    // "Differ" means some chain starts somewhere else — compare against chain
    // 1 rather than against `base`, since `base` is not what any chain ran.
    let differ = chains[1..].iter().any(|c| {
        c.len() != shown.len()
            || c.iter().zip(shown).any(|(a, b)| a.initial != b.initial)
    });
    (shown, differ)
}

/// Print preflight transform report to stderr, pushing diagnostics to collector.
///
/// `per_chain_params` is the per-chain start override the chains will actually
/// run from (`None` when every chain uses `config.estimated_params`). It is
/// reported rather than the config, per gh#513.
pub fn print_preflight(
    config: &FitRunConfig,
    per_chain_params: Option<&[Vec<EstimatedParam>]>,
    collector: &DiagnosticCollector,
) {
    let (specs, starts_differ) =
        preflight_specs(&config.estimated_params, per_chain_params);
    let n_auto = specs.iter()
        .filter(|s| s.rw_sd_auto)
        .count();

    if starts_differ {
        eprintln!("\ntransforms \x1b[2m(chain 1 of {}; chains start at different \
                   points — see chain_starts.tsv)\x1b[0m:", config.n_chains);
    } else {
        eprintln!("\ntransforms:");
    }
    for spec in specs {
        let (tname, pos) = match &spec.transform {
            Transform::Log { lo, hi } => {
                let z = spec.initial.max(1e-300).ln();
                (format!("log     [{}, {}]", lo, hi), format!("log({:.4}) = {:.2}", spec.initial, z))
            }
            Transform::Logit { lo, hi } => {
                let p = ((spec.initial - lo) / (hi - lo)).clamp(1e-10, 1.0 - 1e-10);
                let z = (p / (1.0 - p)).ln();
                let compressed = z.abs() > 2.0;
                if compressed {
                    collector.push(DiagnosticKind::CompressedLogitPosition {
                        param: spec.name.clone(), z,
                    });
                }
                let mark = if compressed { " \x1b[33m⚠ compressed\x1b[0m" } else { "" };
                (format!("logit   [{}, {}]", lo, hi), format!("logit = {:.2}{}", z, mark))
            }
            Transform::None => {
                ("none".into(), format!("{:.4}", spec.initial))
            }
        };
        let source = if spec.rw_sd_auto { "\x1b[33mauto\x1b[0m" } else { "explicit" };
        let transformed_sd = spec.transformed_sd(spec.rw_sd, spec.initial);
        eprintln!("  {:12} {}  {}  rw_sd={:.4} ({:.3}/step, {})",
            spec.name, tname, pos, spec.rw_sd, transformed_sd, source);

        // Push auto rw_sd info diagnostic
        if spec.rw_sd_auto {
            collector.push(DiagnosticKind::AutoRwSd {
                param: spec.name.clone(), rw_sd: spec.rw_sd,
            });
        }
    }

    if n_auto > 0 {
        eprintln!("\n  \x1b[33m⚠ {}/{} parameters using auto rw_sd. Check traces and set explicit values.\x1b[0m",
            n_auto, specs.len());
    }

    // Cooling schedule preview — uses the SAME per-iteration multiplier the run
    // applies (`cooling_multiplier_at_iter`), so preview and actual can't drift.
    let frac = config.if2_config.cooling_fraction;
    let iters = config.if2_config.n_iterations;
    let target_iters = config.if2_config.cooling_target_iters;
    let n_obs = config.observations.len();

    let rw_at =
        |iter: usize| sim::inference::if2::cooling_multiplier_at_iter(frac, target_iters, n_obs, iter);

    eprintln!(
        "\ncooling: cf50={:.2}, reached at iter {} (target), over a {}-iteration run × {} observations",
        frac, target_iters, iters, n_obs
    );
    eprintln!("  iter {:3}: rw_sd at {:.1}%", 1, rw_at(1) * 100.0);
    eprintln!("  iter {:3}: rw_sd at {:.1}% (cf50 reached)", target_iters, rw_at(target_iters) * 100.0);
    eprintln!("  iter {:3}: rw_sd at {:.1}% (run end)", iters, rw_at(iters) * 100.0);

    // Warn if cooling exhausts well before the run ends
    let two_thirds = (iters * 2 / 3).max(1);
    let rw_at_two_thirds = rw_at(two_thirds);
    if rw_at_two_thirds < 0.01 {
        collector.push(DiagnosticKind::CoolingExhausted {
            exhausted_at_iter: two_thirds,
            total_iters: iters,
            rw_fraction_at_exhaustion: rw_at_two_thirds,
        });
    }
    eprintln!();
}

/// Derive the transform for a parameter from its IR metadata.
///
/// Priority: explicit override > param_kind > bounds fallback.
///
/// The param_kind field (populated by the OCaml compiler from the DSL type)
/// is the primary signal: probability → Logit, rate/positive/count → Log.
/// The bounds fallback (lo >= 0 → Log) exists for IR files predating
/// the param_kind field. The hi <= 1.0 probability-detector heuristic
/// was deliberately removed — it caused R0 on [1, 100] to get logit
/// instead of log, which is wrong.
pub fn derive_transform(
    ir_param: &ir::parameter::Parameter,
    transform_override: Option<&str>,
) -> Transform {
    let bounds = ir_param.bounds().unwrap_or((0.0, f64::INFINITY));
    derive_transform_with_bounds(ir_param, transform_override, bounds)
}

/// Like `derive_transform`, but the caller supplies explicit `(lo, hi)`
/// bounds — typically the fit.toml `[estimate].bounds` after validation
/// against the model's declared range. The resulting `Transform::Log`
/// or `Transform::Logit` clamps to *these* bounds, which is what IF2
/// uses to keep particles in the search box. Without this, a fit that
/// tightens bounds would still see IF2 walk particles out to the model
/// bounds, defeating the tightening.
pub fn derive_transform_with_bounds(
    ir_param: &ir::parameter::Parameter,
    transform_override: Option<&str>,
    (lower, upper): (f64, f64),
) -> Transform {
    if let Some(t) = transform_override {
        return match t {
            "log" => Transform::Log { lo: lower, hi: upper },
            "logit" => Transform::Logit { lo: lower, hi: upper },
            _ => Transform::None,
        };
    }
    // For unconstrained-scale kinds (`instant`, `real`, unknown), a
    // finite search box still has to be enforced or IF2 walks the
    // particle out of bounds (gh#66: a bounded `instant` seed-time
    // escaped to τ = −968 / −inf-likelihood). When both bounds are
    // finite, use the scaled-logit (it maps `[lo, hi]` regardless of
    // sign, so a negative lower bound is fine); otherwise there is no
    // box to clamp to, so leave it unconstrained.
    let bounded_or_none = |lo: f64, hi: f64| {
        if lo.is_finite() && hi.is_finite() {
            Transform::Logit { lo, hi }
        } else {
            Transform::None
        }
    };
    if let Some(kind) = ir_param.param_kind {
        // Exhaustive over ParamKind (no `_`): a new kind is a compile error
        // here, forcing an explicit transform decision (the gh#191 payoff).
        use ir::parameter::ParamKind::*;
        match kind {
            Probability => Transform::Logit { lo: lower, hi: upper },
            // `duration` is a positive span → log scale, like rate/count.
            Rate | Positive | Count | Duration => Transform::Log { lo: lower, hi: upper },
            // `instant` is an origin-relative point that may be negative
            // (a seed before the anchor): unconstrained scale unless the
            // fit declares a finite search box, in which case clamp to it.
            Instant => bounded_or_none(lower, upper),
            // `real` is unconstrained, but a finite search box must still
            // be honoured.
            Real => bounded_or_none(lower, upper),
        }
    } else {
        if lower >= 0.0 { Transform::Log { lo: lower, hi: upper } } else { Transform::None }
    }
}

// ── Shared IF2 parameter construction ────────────────────────────────────────

/// What the caller wants to estimate for one parameter.
///
/// Each CLI (if2, profile, fit) builds a Vec<ParamSpec> from its own
/// flags or config. The shared `build_if2_params_from_specs` turns
/// these into Vec<EstimatedParam> — the format the IF2 engine consumes.
///
/// Design: the caller decides WHAT to estimate (the partition).
/// The shared function decides HOW (transform, rw_sd, bounds).
/// This separation eliminates the DRY violations that caused
/// three bugs in one session (profile --rw-sd auto, profile missing
/// --fixed, transform derivation divergence).
pub struct ParamSpec {
    pub name: String,
    /// None = auto from bounds. Some(v) = explicit natural-scale rw_sd.
    pub rw_sd: Option<f64>,
    /// None = auto from param_kind. Some("log") = override.
    pub transform: Option<String>,
    /// Perturb only at t=0 (the IF2 schedule for an initial-state
    /// parameter). See `EstimatedParam::perturb_only_at_t0`.
    pub perturb_only_at_t0: bool,
    /// Caller-supplied bounds override (typically from fit.toml's
    /// `[estimate].bounds`). When `Some`, replaces the model-declared
    /// `ir_param.bounds` for both the `EstimatedParam.{lower, upper}`
    /// fields AND the `Transform::{Log, Logit}.{lo, hi}` clamp ranges,
    /// so IF2's bound enforcement and the init samplers (LHS, uniform)
    /// honour the user's tightening. Must lie within `ir_param.bounds`
    /// when those are present (a fit cannot loosen physical bounds).
    /// `None` = use model bounds verbatim (the profile / pfilter
    /// pattern, where no fit.toml is involved).
    pub bounds: Option<(f64, f64)>,
}

/// Build EstimatedParam specs from caller-provided ParamSpecs.
/// Pure mechanical work: look up indices, derive transforms, compute auto rw_sd.
pub fn build_if2_params_from_specs(
    model: &ir::Model,
    compiled: &CompiledModel,
    base_params: &[f64],
    specs: &[ParamSpec],
) -> Result<Vec<EstimatedParam>, String> {
    let mut params = Vec::with_capacity(specs.len());

    for spec in specs {
        let ir_param = model.parameters.iter()
            .find(|p| p.name == spec.name)
            .ok_or_else(|| format!("parameter '{}' not in model", spec.name))?;
        let idx = *compiled.param_index.get(spec.name.as_str())
            .ok_or_else(|| format!("parameter '{}' not in compiled model", spec.name))?;

        // Resolve bounds: caller-supplied (fit.toml [estimate].bounds)
        // wins over model-declared ir_param.bounds. Reject configurations
        // that try to loosen physical bounds — a fit's bounds must lie
        // within whatever the model declared. This propagates through
        // `EstimatedParam.{lower, upper}` (used by LHS / uniform-random
        // init samplers) AND through the `Transform::{Log, Logit}` clamp
        // bounds (used by IF2 to keep particles in the search box).
        // Without this propagation, fit.toml's bounds are advisory only —
        // IF2 happily walks particles out to model bounds even when the
        // user tightened.
        let (lo, hi) = match (spec.bounds, ir_param.bounds()) {
            (Some((flo, fhi)), Some((mlo, mhi))) => {
                if flo < mlo || fhi > mhi {
                    return Err(format!(
                        "estimate.{}: fit.toml bounds [{}, {}] lie outside \
                         model bounds [{}, {}]. A fit can tighten bounds but \
                         not loosen them — model bounds may reflect physical \
                         constraints. Either widen the model bounds or \
                         tighten the fit bounds.",
                        spec.name, flo, fhi, mlo, mhi));
                }
                (flo, fhi)
            }
            (Some(b), None)  => b,
            (None, Some(b))  => b,
            (None, None)     => (0.0, f64::INFINITY),
        };

        // Transform: spec override > param_kind > fallback. Built with
        // the resolved (lo, hi) so the clamp bounds match the search box.
        let transform = derive_transform_with_bounds(
            ir_param, spec.transform.as_deref(), (lo, hi));

        // rw_sd: spec explicit > auto from resolved bounds
        let rw_sd = spec.rw_sd
            .unwrap_or_else(|| auto_rw_sd_from_value(base_params[idx], lo, hi, &transform));

        params.push(EstimatedParam {
            name: spec.name.clone(),
            index: idx,
            initial: base_params[idx],
            rw_sd,
            transform,
            lower: lo,
            upper: hi,
            perturb_only_at_t0: spec.perturb_only_at_t0,
            rw_sd_auto: spec.rw_sd.is_none(),
        });
    }

    Ok(params)
}

/// Auto-compute rw_sd from bounds on the transformed scale.
///
/// Returns a natural-scale rw_sd value. At each IF2 perturbation step,
/// `EstimatedParam::transformed_sd(natural_sd, current_value)` re-converts
/// this to the transformed scale using the delta method at the CURRENT
/// parameter value. So the midpoint used here is just a reference point
/// for expressing the natural-scale number — the actual perturbation
/// adapts to the current position through transformed_sd. Any reference
/// point (midpoint, lower bound, current value) would produce the same
/// perturbation on the transformed scale.
///
/// Log: log_range / 20 on transformed scale, converted to natural at geometric midpoint.
///   For sigma_se in [0.001, 5.0]: log_range = 8.5, log_sd = 0.43, meaning ~±50% per step.
/// Logit: range / 6 on natural scale. Logit range is ~12 (-6 to 6), /6 gives ~2.0 on logit.
/// Identity: (hi - lo) / 6.
///
/// The /20 vs /6 asymmetry: log is unbounded (perturbations accumulate) while logit
/// saturates at bounds. Log needs more conservative defaults.
///
/// This is a starting heuristic, not a solution. Scout's MAD-based
/// calibration replaces it for refine. The modeler can override with
/// explicit rw_sd in fit.toml or --rw-sd on the CLI.
pub fn auto_rw_sd_from_value(_current_value: f64, lower: f64, upper: f64, transform: &Transform) -> f64 {
    match transform {
        Transform::Log { lo, hi } => {
            let lo = lo.max(1e-300);
            let hi_val = if hi.is_finite() { *hi } else { lo * 1000.0 };
            let log_range = (hi_val / lo).ln();
            let log_sd = log_range / 20.0;
            // Convert to natural scale at geometric midpoint
            let midpoint = (lo * hi_val).sqrt();
            midpoint * log_sd
        }
        Transform::Logit { lo, hi } => {
            (hi - lo) / 6.0
        }
        Transform::None => {
            let lo = if lower.is_finite() { lower } else { -1e6 };
            let hi = if upper.is_finite() { upper } else { 1e6 };
            (hi - lo) / 6.0
        }
    }
}

/// Load observations from TSV, validating time alignment with dt.
///
/// Sparse/holes: the value column may contain the missing-value token `NA`,
/// loaded as a HOLE — its time stays in the grid (so the incidence accumulator
/// still resets at its index) but it carries no value. Returns both the
/// authoritative per-grid-time cells (`None` = hole) and a dense placeholder
/// view of `Observation`s (a hole shows as value 0) for the diagnostics and
/// time-axis callers, where a hole's value is not load-bearing. The cells are
/// threaded into the obs model so the already hole-correct scoring seam handles
/// missing values; the placeholder view is never scored.
pub(crate) fn load_observations(
    path: &str,
    obs_model: &ir::observation::ObservationModel,
    siblings: &[&ir::observation::ObservationModel],
    dt: f64,
    opts: &crate::caltime_load::TimeOpts,
) -> Result<
    (Vec<Observation>, Vec<Option<sim::inference::ObsCell>>, Vec<Vec<(String, f64)>>),
    String,
> {
    // DISPATCH: a stratified (long-form) stream — its `columns { }` declares at
    // least one `: dim` column — loads via the long-form router (routes file
    // rows to the matching stratum leaf BY NAME, builds the partial-coverage
    // union axis). An unstratified stream keeps the existing wide/by-name path.
    let (times, cells, mut aux) = if crate::pfilter::is_long_form_stream(obs_model) {
        crate::pfilter::load_long_form_stream(path, obs_model, siblings, opts)?
    } else {
        // Bind the file columns BY NAME: the declared `Time`-role column is the
        // time axis (the by-name-time flip — no positional "column 0 is time"),
        // and `scored` is the value column.
        let time_col = crate::pfilter::obs_time_column(obs_model)?;
        let value_col = &obs_model.scored;
        let (times, cells) =
            crate::pfilter::load_data_tsv_column_cells(path, time_col, value_col, opts)?;
        // Per-observation auxiliary data (binomial `n = tested`, person-time
        // offset; §3, §6.1). A row where the scored value OR any referenced aux
        // is `NA` is a hole (present-together-or-hole).
        let aux_cols = crate::pfilter::stream_aux_columns(obs_model);
        let (aux, force_hole) =
            crate::pfilter::load_stream_aux(path, &aux_cols, cells.len())?;
        let mut cells = cells;
        for r in 0..cells.len() {
            if force_hole[r] {
                cells[r] = None;
            }
        }
        (times, cells, aux)
    };
    for r in 0..cells.len() {
        if cells[r].is_none() {
            aux[r].clear();
        }
    }
    // Validate time alignment (holes keep their time, so this is unaffected by
    // missing values).
    for &time in &times {
        let remainder = time % dt;
        let aligned = remainder.abs() < 1e-9 || (dt - remainder.abs()).abs() < 1e-9;
        if !aligned {
            return Err(format!(
                "observation at t={} is not a multiple of dt={}.\n\
                 The chain-binomial state only exists at step boundaries.\n\
                 Adjust observation times or dt to align.",
                time, dt
            ));
        }
    }
    // Dense placeholder view (holes → 0.0) for diagnostics/time.
    let observations: Vec<Observation> = times.iter().zip(cells.iter())
        .map(|(&time, cell)| Observation {
            time,
            value: match cell {
                Some(sim::inference::ObsCell::Scalar(v)) => *v,
                None => 0.0,
            },
        })
        .collect();
    Ok((observations, cells, aux))
}

/// Convert a resolved data-binding list (`resolve_data_specs` for the CLI
/// `--data NAME=PATH` form, or `load_data_observations_from_fit_toml` for the
/// fit-toml `[data.observations]` form) into the by-SOURCE `effective` map that
/// [`resolve_and_load_obs_streams`] consumes.
///
/// This is the boundary adapter between the CLI/toml key spaces and the seam.
/// Each binding key is resolved to a stream `source`: a key equal to a stream's
/// `source` (the fit-toml family-root / `[data.observations]` form) is used
/// directly; a key equal to a leaf NAME (the CLI form, where `resolve_data_specs`
/// has already expanded a family root to its leaf names) resolves to that leaf's
/// source. Because a stratified family's leaves share one source, several leaf
/// bindings to the same file dedup to one `(source → path)` entry, so the seam
/// binds every leaf of the source from that one long-form file.
///
/// Errors:
/// - a key matching neither a source nor a leaf name (a typo, not a silent
///   no-op);
/// - the same source bound to two DIFFERENT files (the seam loads one file per
///   source — a stratified family shares one long-form file; split files across
///   one source cannot be honoured).
pub(crate) fn data_bindings_to_effective(
    model: &ir::Model,
    bindings: &[(String, std::path::PathBuf)],
) -> Result<indexmap::IndexMap<String, String>, String> {
    let mut effective: indexmap::IndexMap<String, String> = indexmap::IndexMap::new();
    for (key, path) in bindings {
        let path_str = path.to_string_lossy().into_owned();
        // Match by SOURCE first, then by leaf NAME — the same precedence
        // `profile`'s prior fan-out used (`k == o.source || k == o.name`). Both
        // realistic forms (a source family root, a CLI-expanded leaf name)
        // resolve to the same source regardless of precedence.
        let source = model.observations.iter()
            .find(|o| &o.source == key)
            .or_else(|| model.observations.iter().find(|o| &o.name == key))
            .map(|o| o.source.clone());
        let source = match source {
            Some(s) => s,
            None => {
                let mut avail: Vec<&str> = model.observations.iter()
                    .map(|o| o.source.as_str()).collect();
                avail.sort_unstable();
                avail.dedup();
                return Err(format!(
                    "data binding '{}' matches no observation stream (by source \
                     or name). Available sources: {}", key, avail.join(", ")));
            }
        };
        if let Some(existing) = effective.get(&source) {
            if existing != &path_str {
                return Err(format!(
                    "observation source '{}' is bound to two different files \
                     ('{}' and '{}'). A stratified family shares one long-form \
                     file; split files across one source are not supported.",
                    source, existing, path_str));
            }
        }
        effective.insert(source, path_str);
    }
    Ok(effective)
}

/// Reject a data-binding key that names no observation stream `source`.
///
/// `[data.observations]`, `[data.holdout]` and `--data NAME=PATH` all key by
/// observation SOURCE (the `from <label>`, defaulting to the stream name;
/// §2.4). A key matching none of them binds a file to nothing — at best a
/// mistyped stream name, at worst a *top-level* fit.toml setting that TOML
/// scoping captured into the table by accident, which then silently does not
/// apply (`condition_from` written below the `[data.observations]` header is
/// the motivating case: it reverts conditioning to none while looking set).
///
/// `origin` names the table (or flag) the keys were typed in, so the message
/// points at what the user wrote.
///
/// The fit driver calls this BEFORE the identity digests read any bytes, so an
/// unbound key reports as a binding error naming the real sources rather than
/// as an unreadable data file.
pub(crate) fn check_bound_sources(
    model: &ir::Model,
    origin: &str,
    bound: &indexmap::IndexMap<String, String>,
) -> Result<(), String> {
    let mut sources: Vec<&str> =
        model.observations.iter().map(|o| o.source.as_str()).collect();
    sources.sort_unstable();
    sources.dedup();
    for (key, path) in bound {
        if sources.contains(&key.as_str()) {
            continue;
        }
        // A key whose value is not even a readable file is far likelier to be a
        // top-level setting swallowed by the preceding `[table]` header than a
        // mistyped stream name — and the two have different fixes, so say which
        // one the evidence points at rather than offering both.
        let hint = if std::path::Path::new(path).is_file() {
            String::new()
        } else {
            format!(
                " Its value (\"{path}\") is not a readable file either — if \
                 `{key}` was meant as a TOP-LEVEL fit.toml key, note that TOML \
                 binds every key after a `[table]` header to that table, so it \
                 must sit ABOVE the first `[table]`."
            )
        };
        return Err(format!(
            "{origin}: '{key}' is not an observation source. Keys here bind a \
             data file to an observation stream declared in the model's \
             `observations {{ }}` block.{hint} Available sources: {}",
            sources.join(", ")));
    }
    Ok(())
}

/// Resolve the conditioning spec for a standalone fixed-θ command (gh#621):
/// the CLI `--condition-from` flags win; otherwise a `--fit` toml's
/// `condition_from` key applies; otherwise none. The toml is re-parsed here
/// (cheap, and the data-binding fallback already parses it independently);
/// an unreadable toml is only an error when it is actually consulted.
pub(crate) fn condition_spec_from_cli_or_toml(
    cli_specs: &[String],
    fit_toml: Option<&std::path::Path>,
) -> Result<Option<crate::fit::config_v2::ConditionFrom>, String> {
    if let Some(spec) = crate::fit::config_v2::ConditionFrom::from_cli_specs(cli_specs)? {
        if fit_toml.is_some() {
            eprintln!("--condition-from on CLI overrides --fit toml condition_from");
        }
        return Ok(Some(spec));
    }
    let Some(path) = fit_toml else { return Ok(None) };
    let path_str = path.to_string_lossy().into_owned();
    let fit_cfg = crate::fit::config_v2::FitConfigV2::load(&path_str)
        .map_err(|e| format!("failed to load --fit toml '{path_str}': {e}"))?;
    Ok(fit_cfg.condition_from)
}

/// Where ONE stream's first scored bin opens, per the fit's `condition_from`.
///
/// The three cases are genuinely different and the callers act differently on
/// each, so they are three variants rather than an `Option<f64>` that conflates
/// the first two: "no spec at all" is what the W329 wide-first-window enforcer
/// judges, while "a spec that resolved to the origin" is the user's explicit
/// opt-in to scoring the whole leading window and is only ever announced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum StreamWindow {
    /// No `condition_from` spec applies to this stream. Its first bin opens at
    /// `t_start`; whether that is acceptable is W329's call.
    Unspecified,
    /// A spec applies and resolved to `t_start` — the documented opt-in to
    /// scoring the full leading window, no warm-up discarded.
    AtOrigin,
    /// A spec applies: `[t_start, t)` is simulated as warm-up but not scored,
    /// and the first scored bin is `(t, first_obs]`.
    ConditionedFrom(f64),
}

impl StreamWindow {
    /// The time the stream's first bin opens at, when that is NOT the model
    /// origin. `None` for both [`StreamWindow::Unspecified`] and
    /// [`StreamWindow::AtOrigin`], which open at `t_start` — the unseeded
    /// default every accumulator already has.
    pub(crate) fn boundary(self) -> Option<f64> {
        match self {
            StreamWindow::ConditionedFrom(t) => Some(t),
            StreamWindow::Unspecified | StreamWindow::AtOrigin => None,
        }
    }
}

/// Resolve ONE stream's conditioning window: its per-stream shadow, else the
/// all-streams default, else none — and then the spec string against that
/// stream's own first observation.
///
/// This is the single answer to "where does this stream's first bin open".
/// `fit run` / `pfilter` / `profile` reach it through
/// [`apply_conditioning_windows`], which additionally prepends the reset-only
/// hole and runs the W329 enforcer. `fit predict` calls it directly (gh#702):
/// it needs the BOUNDARY without a hole, because its observation times are also
/// the emitted axis and the `value_at` anchor axis, and a synthetic row must
/// shift neither. Deciding the boundary twice, in two places, is exactly the
/// fork gh#702 was filed on — the likelihood reset its incidence accumulator at
/// the boundary and the predictive did not.
///
/// No label validation here: that is a property of the whole spec against the
/// whole bound stream set, and belongs where the spec is first applied
/// ([`apply_conditioning_windows`]), not at a per-stream read.
pub(crate) fn stream_condition_window(
    condition_from: Option<&crate::fit::config_v2::ConditionFrom>,
    label: &str,
    stream_name: &str,
    first_obs_s: f64,
    model: &ir::Model,
    t_start: f64,
    dt: f64,
) -> Result<StreamWindow, String> {
    let Some(raw) = condition_from.and_then(|c| c.resolve_for(label)) else {
        return Ok(StreamWindow::Unspecified);
    };
    let resolved = resolve_condition_from(
        raw,
        first_obs_s,
        t_start,
        model.origin.as_deref(),
        &model.time_unit,
        dt,
    )
    .map_err(|e| format!("stream '{stream_name}': {e}"))?;
    Ok(match resolved {
        Some(cond_from) => StreamWindow::ConditionedFrom(cond_from),
        None => StreamWindow::AtOrigin,
    })
}

/// Apply the per-stream conditioning windows (gh#134 multi-cadence Phase 3,
/// gh#621) to already-loaded observation streams: validate the spec's shadow
/// labels, resolve each stream's `condition_from` boundary, prepend the
/// leading reset-only HOLE to that stream's data/cells/aux, and run the W329
/// wide-first-window enforcer where a stream resolves to NO conditioning.
/// Returns the inserted boundaries for the caller's canonical union grid
/// (callers whose union is built FROM `streams` afterwards can ignore them —
/// the hole is already in each stream's own schedule).
///
/// Shared by `fit run` (`FitRunConfig::build`), `pfilter`, and `profile`
/// (gh#621): the fixed-θ scorers must score the SAME window the fit scores,
/// or their logliks are incomparable and a −inf is ambiguous (a bad θ vs. an
/// unconstrainable leading window).
pub(crate) fn apply_conditioning_windows(
    streams: &mut [ObsStream],
    condition_from: Option<&crate::fit::config_v2::ConditionFrom>,
    model: &ir::Model,
    t_start: f64,
    dt: f64,
) -> Result<Vec<f64>, String> {
    use ir::observation::TemporalKind;

    // Validate `[condition_from]` shadow labels (typo-safety + the
    // reserved-`default` collision) against the bound streams' labels.
    if let Some(spec) = condition_from {
        let valid_labels: Vec<String> = {
            let mut v: Vec<String> =
                streams.iter().map(|s| s.obs_model_ir.source.clone()).collect();
            v.sort();
            v.dedup();
            v
        };
        spec.validate_labels(&valid_labels)?;
    }

    // Walk streams in a deterministic order, resolving each one's
    // conditioning spec and applying the leading hole (or running the
    // W329 enforcer when it resolves to NONE). Collect each inserted
    // boundary so the canonical union is updated once afterwards.
    let mut union_inserts: Vec<f64> = Vec::new();
    for s in streams.iter_mut() {
        let label = s.obs_model_ir.source.as_str();
        let kind = s.projection.temporal_kind();
        let first_obs_s = s.data.iter()
            .map(|o| o.time)
            .fold(f64::INFINITY, f64::min);

        // Where THIS stream's first bin opens: its shadow, else the
        // all-streams default, else nothing — resolved through the single
        // shared resolver `fit predict` also reads (gh#702).
        let window = stream_condition_window(
            condition_from, label, &s.name, first_obs_s, model, t_start, dt,
        )?;

        match window {
            StreamWindow::ConditionedFrom(cond_from) => {
                eprintln!(
                    "  \x1b[36mconditioning window:\x1b[0m stream \
                     '{}': warm-up [{t_start}, {cond_from}) simulated \
                     but not scored; first scored bin is \
                     ({cond_from}, {first_obs_s}]",
                    s.name
                );
                // Prepend the per-stream leading reset-only hole.
                // The `cells` are authoritative for scoring; the
                // `data` row's value (0.0) is a never-read
                // placeholder.
                s.data.insert(0, Observation { time: cond_from, value: 0.0 });
                s.cells.insert(0, None);
                s.aux.insert(0, Vec::new());
                union_inserts.push(cond_from);
            }
            StreamWindow::AtOrigin => {
                // cond_from == t_start: the user explicitly set
                // conditioning to the model origin — the documented
                // "score the whole leading window" opt-in. No
                // warm-up is discarded; the first bin is the full
                // (t_start, first_obs_s]. This is the deliberate
                // escape hatch out of W329, NOT a no-op to hide: on
                // a WIDE incidence window (the gh#134 shape) say so
                // loudly so the choice is visible, not silent.
                if kind == TemporalKind::Interval {
                    let obs_times: Vec<f64> =
                        s.data.iter().map(|o| o.time).collect();
                    if let Some(anomaly) =
                        crate::util::check_first_interval_window(t_start, &obs_times)
                    {
                        eprintln!(
                            "  \x1b[36mconditioning window:\x1b[0m \
                             incidence stream '{name}': condition_from \
                             resolves to the model origin (t_start = \
                             {t_start}) — scoring the FULL \
                             {window}-{unit} leading window against the \
                             first datum, no warm-up discarded (the \
                             gh#134 wide window, opted into explicitly).",
                            name = s.name,
                            window = fmt_span(anomaly.first_window),
                            unit = cadence_word(&model.time_unit),
                        );
                    }
                }
            }
            StreamWindow::Unspecified => {
                // No conditioning for this stream. The W329 detector
                // decides whether that is fine (window ≈ one cadence) or
                // the gh#134 wrong-number (anomalously wide window on an
                // incidence stream → hard error). Run against THIS
                // stream's own times (per-stream modal gap). A prevalence
                // stream is exempt from the hard error but still
                // soft-warns (free-running drift the first datum
                // corrects).
                let obs_times: Vec<f64> = s.data.iter().map(|o| o.time).collect();
                if let Some(anomaly) =
                    crate::util::check_first_interval_window(t_start, &obs_times)
                {
                    match kind {
                        TemporalKind::Interval => {
                            // The first incidence bin would accumulate the
                            // whole leading span and score it against one
                            // datum. Name the per-stream fix EXACTLY.
                            return Err(format!(
                                "incidence stream '{name}' has a \
                                 {window}-{unit} first window against a \
                                 ~{cadence}-{unit} cadence; the first \
                                 datum cannot constrain that whole span. \
                                 State the conditioning window, e.g. \
                                 `condition_from.{label} = \"first_obs - 1 week\"` \
                                 (or a longer warm-up to discard).",
                                name = s.name,
                                window = fmt_span(anomaly.first_window),
                                cadence = fmt_span(anomaly.modal_gap),
                                unit = cadence_word(&model.time_unit),
                            ));
                        }
                        TemporalKind::Instant => {
                            eprintln!("{}", anomaly.warn_message());
                        }
                    }
                }
            }
        }
    }

    Ok(union_inserts)
}

/// Resolve the DATA-bound observation streams (BY SOURCE) and load each one's
/// per-observation values + aux, returning one [`ObsStream`] per bound leaf.
///
/// `effective` maps observation SOURCE → data-file path — the `[data]`
/// resolution the caller has already done (`DataSpec::effective_observations`
/// for the fit-toml form, or `data_bindings_to_effective` for the CLI `--data`
/// form). This function:
///
/// - filters `model.observations` to the blocks whose `source` is bound (sorted
///   by name for deterministic ordering — two fits with the same observations
///   but different toml ordering hash identically downstream),
/// - hard-errors on a bound source that names no real stream (a typo, not a
///   silent no-op),
/// - for each bound leaf, dispatches [`load_observations`] (long-form vs wide,
///   holes + aux), resolves its projection, and runs the per-stream
///   origin / first-window guards.
///
/// This is the single seam that fit run, pfilter, and profile route through, so
/// the resolve + slice + aux behaviour cannot be live in one command and
/// silently absent in another. Conditioning-window (`condition_from` / W329)
/// handling is NOT here — it is fit-specific and stays in `FitRunConfig::build`.
pub(crate) fn resolve_and_load_obs_streams(
    model: &ir::Model,
    compiled: &CompiledModel,
    effective: &indexmap::IndexMap<String, String>,
    dt: f64,
    time_opts: &crate::caltime_load::TimeOpts,
) -> Result<Vec<ObsStream>, String> {
    let mut obs_blocks: Vec<&ir::observation::ObservationModel> =
        model.observations.iter()
            .filter(|o| effective.contains_key(&o.source))
            .collect();
    obs_blocks.sort_by(|a, b| a.name.cmp(&b.name));

    check_bound_sources(model, "[data.observations] / --data", effective)?;
    if obs_blocks.is_empty() {
        return Err(
            "no observation stream is bound to a data file — check that \
             [data.observations] / --data names match the model's stream \
             sources.".into());
    }

    let mut streams = Vec::new();
    for obs_model in &obs_blocks {
        let stream_name = obs_model.name.clone();
        let data_path = effective.get(&obs_model.source)
            .expect("filtered to bound sources above");
        let siblings: Vec<&ir::observation::ObservationModel> = model.observations.iter()
            .filter(|o| o.source == obs_model.source)
            .collect();
        let (obs, cells, aux) =
            load_observations(data_path, obs_model, &siblings, dt, time_opts)?;
        let obs_model: ir::observation::ObservationModel = (*obs_model).clone();
        let projection = sim::inference::multi_stream_obs::StreamProjection::from_ir(
            &obs_model.projection, compiled, &stream_name,
        )?;

        // F4: reject an observation strictly before the model origin. The
        // integrator never propagates a particle to a time it has already
        // passed, so the window yields zero substeps yet the obs is still
        // scored — a silent wrong answer. Hard error at load. gh#174: reject a
        // positive incidence observation at the model origin (zero-width first
        // window → -Inf masquerading as filter degeneracy).
        {
            let obs_times: Vec<f64> = obs.iter().map(|o| o.time).collect();
            crate::util::check_obs_before_origin(
                &stream_name,
                compiled.model.simulation.t_start,
                &obs_times,
            )?;
            // The degenerate-origin-window check fires only when the FIRST
            // observation sits exactly on the origin AND carries a positive
            // incidence value. A leading HOLE scores no value at the origin, so
            // pass a non-positive sentinel (the check is a no-op then). We must
            // NOT substitute a later present value nor a fictitious 0 that
            // scores.
            let first_value = match cells.first() {
                Some(Some(sim::inference::ObsCell::Scalar(v))) => *v,
                _ => 0.0,
            };
            crate::util::check_incidence_origin_window(
                &stream_name,
                &obs_model.projection,
                compiled.model.simulation.t_start,
                &obs_times,
                first_value,
            )?;
        }

        streams.push(ObsStream {
            name: stream_name,
            projection,
            obs_model_ir: obs_model,
            data: obs,
            cells,
            aux,
        });
    }
    Ok(streams)
}

/// Map loaded [`ObsStream`]s to the [`StreamSpec`]s that
/// [`sim::inference::BoundObs::bind`] consumes. Single source of the
/// `ObsStream -> StreamSpec` mapping for every consumer (fit run's
/// `build_obs_model`, and — once routed — pfilter/profile).
///
/// Multi-cadence (§3.3): each stream is fed its OWN schedule (`s.data`'s times),
/// NOT the union; `bind` re-merges to the union and records per-stream
/// `at_union` membership. `s.cells` is already this stream's own cells
/// (`cells.len() == this stream's obs_times.len()`).
pub(crate) fn stream_specs_from_obs_streams(
    streams: &[ObsStream],
) -> Vec<sim::inference::multi_stream_obs::StreamSpec> {
    streams.iter()
        .map(|s| sim::inference::multi_stream_obs::StreamSpec {
            projection: s.projection.clone(),
            ir_model: s.obs_model_ir.clone(),
            // Authoritative per-grid-time cells (holes = `None`). A hole
            // contributes no likelihood term but its obs time stays in the grid,
            // so the per-obs-index incidence reset still fires at it.
            observations: s.cells.clone(),
            obs_times: s.data.iter().map(|o| o.time).collect(),
            aux: s.aux.clone(),
        })
        .collect()
}


/// Resolve a single conditioning spec string to a concrete `cond_from` in model
/// time, then validate it against this stream's conditioning window
/// `[t_start, first_obs_s)`. The per-stream selection (which spec applies to
/// which stream) happens at the call site via
/// [`crate::fit::config_v2::ConditionFrom::resolve_for`]; this resolves the one
/// spec it is handed.
///
/// Returns:
/// - `Ok(None)`  — no conditioning (the value resolved to `t_start`, the
///   no-op case). The caller inserts NO leading hole and the stream is
///   bit-identical to an unconditioned one.
/// - `Ok(Some(c))` — insert a leading reset-only hole at model time `c`,
///   with `t_start < c < first_obs_s`.
/// - `Err(_)`    — a located error: `c < t_start`, `c >= first_obs_s`, an
///   unparseable form, a date with no model origin, or an off-grid `c`.
///
/// Accepted forms:
/// - a bare model-time number (`"14"`) — used verbatim;
/// - `"date(\"YYYY-MM-DD\")"` / `"YYYY-MM-DD"` — absolute calendar date,
///   resolved via `origin` + `time_unit` (`date_to_internal`);
/// - `"first_obs - <N> <unit>"` — `first_obs_time − N·unit`.
///
/// `dt`-grid alignment is checked here so a mis-specified boundary fails at
/// build time rather than tripping the chain-binomial step-boundary invariant
/// downstream.
pub fn resolve_condition_from(
    spec: &str,
    first_obs_time: f64,
    t_start: f64,
    origin: Option<&str>,
    time_unit: &str,
    dt: f64,
) -> Result<Option<f64>, String> {
    let cond_from = parse_condition_spec(spec, first_obs_time, origin, time_unit)?;

    // No-op case: cond_from == t_start ⇒ no conditioning, no hole. Treat a
    // float-noise-equal value as exactly t_start so the bit-identical
    // guarantee survives a date that rounds onto the origin.
    if (cond_from - t_start).abs() < 1e-9 {
        return Ok(None);
    }

    if cond_from < t_start {
        return Err(format!(
            "condition_from resolves to t = {cond_from}, which is before the \
             model start t_start = {t_start}. The conditioning window must lie \
             within [t_start, first_obs); pick a boundary at or after t_start."
        ));
    }
    if cond_from >= first_obs_time - 1e-9 {
        return Err(format!(
            "condition_from resolves to t = {cond_from}, which is at or after \
             the first observation (t = {first_obs_time}) — nothing to \
             condition on. The conditioning window must lie strictly before \
             the first observation: t_start ≤ condition_from < first_obs."
        ));
    }

    // Must land on the dt grid (the chain-binomial state only exists at step
    // boundaries; the inserted hole becomes an obs-grid time scored/reset
    // there). Same alignment rule the real observations are held to.
    let remainder = (cond_from - t_start).rem_euclid(dt);
    let aligned = remainder.abs() < 1e-9 || (dt - remainder).abs() < 1e-9;
    if !aligned {
        return Err(format!(
            "condition_from resolves to t = {cond_from}, which is not on the \
             dt = {dt} grid relative to t_start = {t_start}. The conditioning \
             boundary must align to a step boundary; adjust condition_from or dt."
        ));
    }

    Ok(Some(cond_from))
}

/// An observation anchor in a time spec (gh#626): `first_obs` / `last_obs`.
/// Which observation(s) it folds over is the CALLER's semantics — per-stream
/// for `condition_from`, global (all bound streams) for `simulate --to`.
///
/// This is the IR's anchor type, not a CLI-local copy (gh#616): the DSL spells
/// the same two anchors in `simulate { to }` / `breakpoints` / `value_at`, and
/// a second enum would be a fork the two sides drift across.
pub(crate) use ir::anchor::ObsAnchor;

/// A parsed, data-free time spec (gh#626): either an absolute model time
/// (number or date, resolved at parse) or an observation anchor plus a signed
/// offset already converted to model time units. Parsing needs no data, so a
/// caller can decide whether to load observations before resolving.
///
/// The anchored arm carries the IR's [`ir::anchor::AnchoredTime`], so a CLI
/// `--to "last_obs + 8 weeks"` and a model's `to = last_obs + 8 'weeks` are the
/// SAME value once parsed, and resolve through the same `AnchoredTime::resolve`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum TimeSpec {
    Absolute(f64),
    Anchored(ir::anchor::AnchoredTime),
}

/// Parse the shared obs-anchored time grammar (gh#626):
///
/// ```text
/// SPEC   := NUMBER | date("YYYY-MM-DD") | YYYY-MM-DD
///         | ANCHOR | ANCHOR (+|-) N UNIT
/// ANCHOR := first_obs | last_obs
/// UNIT   := day(s)|d | week(s)|w | month(s)|mo | year(s)|yr|y
/// ```
///
/// `what` names the surface in every message (`"--to"` / `"condition_from"`).
/// Months and years are fixed spans (`days_per_unit`), not calendar
/// arithmetic. Deliberate rejections, each with a hint: the commuted
/// `8 weeks + last_obs` order and the DSL tick spelling (`8 'weeks`) — the
/// canonical form is `ANCHOR ± N UNIT` with a plain unit word.
pub(crate) fn parse_time_spec(
    what: &str,
    raw: &str,
    origin: Option<&str>,
    time_unit: &str,
) -> Result<TimeSpec, String> {
    let s = raw.trim();

    // Bare model-time number.
    if let Ok(v) = s.parse::<f64>() {
        return Ok(TimeSpec::Absolute(v));
    }

    // Anchored form: ANCHOR, or ANCHOR (+|-) N UNIT. The accepted spellings come
    // from the IR's anchor type, so the CLI and the DSL cannot diverge.
    let head = s.split(['+', '-']).next().unwrap_or("").trim();
    if let Ok(anchor) = head.parse::<ObsAnchor>() {
        let rest = s[head.len()..].trim_start();
        if rest.is_empty() {
            return Ok(TimeSpec::Anchored(ir::anchor::AnchoredTime::bare(anchor)));
        }
        let (sign, after) = match rest.split_at(1) {
            ("+", a) => (1.0, a.trim()),
            ("-", a) => (-1.0, a.trim()),
            _ => {
                return Err(format!(
                    "{what} = \"{raw}\": expected \"{head} + <N> <unit>\" or \
                     \"{head} - <N> <unit>\"."));
            }
        };
        let mut it = after.split_whitespace();
        let n_tok = it.next().ok_or_else(|| format!(
            "{what} = \"{raw}\": expected \"{head} {} <N> <unit>\", e.g. \
             \"last_obs + 8 weeks\".",
            if sign > 0.0 { "+" } else { "-" }))?;
        let unit_tok = it.next().ok_or_else(|| format!(
            "{what} = \"{raw}\": missing unit; expected \
             \"{head} {} <N> <unit>\", e.g. \"last_obs + 8 weeks\".",
            if sign > 0.0 { "+" } else { "-" }))?;
        if it.next().is_some() {
            return Err(format!(
                "{what} = \"{raw}\": trailing tokens after the unit; \
                 expected exactly \"{head} ± <N> <unit>\"."));
        }
        let n: f64 = n_tok.parse().map_err(|_| format!(
            "{what} = \"{raw}\": '{n_tok}' is not a number."))?;
        if n < 0.0 {
            return Err(format!(
                "{what} = \"{raw}\": N must be non-negative (the sign is \
                 the ± before it); got {n}."));
        }
        if let Some(stripped) = unit_tok.strip_prefix('\'') {
            return Err(format!(
                "{what} = \"{raw}\": the DSL tick spelling ('{stripped}) is \
                 not accepted here — write the unit as a plain word: \
                 \"{head} {} {n_tok} {stripped}\".",
                if sign > 0.0 { "+" } else { "-" }));
        }
        let unit = canonical_duration_unit(unit_tok);
        let span_days = n * ir::caltime::days_per_unit(&unit)
            .map_err(|_| format!(
                "{what} = \"{raw}\": unknown unit '{unit_tok}'. \
                 Use days / weeks / months / years."))?;
        let model_unit_days = ir::caltime::days_per_unit(time_unit)
            .map_err(|e| format!("{what}: model time_unit: {e}"))?;
        return Ok(TimeSpec::Anchored(ir::anchor::AnchoredTime {
            anchor,
            offset: sign * span_days / model_unit_days,
        }));
    }

    // Commuted anchored form ("8 weeks + last_obs"): reject with the
    // canonical spelling rather than silently not-a-date.
    if s.contains("first_obs") || s.contains("last_obs") {
        let a = if s.contains("last_obs") { "last_obs" } else { "first_obs" };
        return Err(format!(
            "{what} = \"{raw}\": write the anchor first — e.g. \
             \"{a} + 8 weeks\"."));
    }

    // Absolute date form: date("YYYY-MM-DD") or bare YYYY-MM-DD.
    let date_str = if let Some(inner) = s.strip_prefix("date(") {
        inner.strip_suffix(')')
            .map(|x| x.trim().trim_matches('"').to_string())
            .ok_or_else(|| format!(
                "{what} = \"{raw}\": malformed date(...) — expected \
                 date(\"YYYY-MM-DD\")."))?
    } else {
        s.to_string()
    };
    let origin = origin.ok_or_else(|| format!(
        "{what} = \"{raw}\" is a calendar date, but the model declares no \
         `origin = date(\"…\")`. Either add an origin to the model or use a \
         numeric model-time value for {what}."))?;
    ir::caltime::date_to_internal(origin, &date_str, time_unit)
        .map(TimeSpec::Absolute)
        .map_err(|e| format!(
            "{what} = \"{raw}\": cannot resolve date '{date_str}' against \
             origin '{origin}' (time_unit = {time_unit}): {e:?}"))
}

/// Parse a string-form [`ConditionFrom::Spec`] to model time.
fn parse_condition_spec(
    raw: &str,
    first_obs_time: f64,
    origin: Option<&str>,
    time_unit: &str,
) -> Result<f64, String> {
    // Thin wrapper over the shared obs-anchored grammar (gh#626): the
    // ACCEPTANCE set is unchanged — `first_obs` only, subtraction only —
    // enforced by post-restriction so conditioning and `--to` can never
    // drift apart. Two rejection messages improved with the shared parser
    // (`last_obs …` and `first_obs + …` used to fall through to a confusing
    // calendar-date error).
    match parse_time_spec("condition_from", raw, origin, time_unit)? {
        TimeSpec::Absolute(v) => Ok(v),
        TimeSpec::Anchored(a) if a.anchor == ObsAnchor::Last => Err(format!(
            "condition_from = \"{raw}\": the conditioning window precedes the \
             data, so it anchors to first_obs; last_obs is not meaningful \
             here."
        )),
        TimeSpec::Anchored(a) if a.offset > 0.0 => {
            Err(format!(
                "condition_from = \"{raw}\": the relative form must subtract \
                 from first_obs, e.g. \"first_obs - 1 week\" (a boundary after \
                 the first observation would condition on nothing)."
            ))
        }
        TimeSpec::Anchored(a) => Ok(a.resolve(first_obs_time)),
    }
}

/// Normalize a user-written duration unit token to the canonical
/// [`ir::caltime::days_per_unit`] spelling, accepting common singular/plural
/// abbreviations so `"1 week"` and `"7 days"` both work.
fn canonical_duration_unit(tok: &str) -> String {
    match tok.trim().to_lowercase().as_str() {
        "day" | "days" | "d" => "days",
        "week" | "weeks" | "w" => "weeks",
        "month" | "months" | "mo" => "months",
        "year" | "years" | "yr" | "y" => "years",
        other => other,
    }
    .to_string()
}

/// Format a model-time span for a diagnostic: drop the trailing `.0` on a whole
/// number (`351.0` → `"351"`, `13.5` → `"13.5"`) so the W329 per-stream message
/// reads cleanly.
fn fmt_span(x: f64) -> String {
    if (x - x.round()).abs() < 1e-9 {
        format!("{}", x.round() as i64)
    } else {
        format!("{x}")
    }
}

/// Singularize the model `time_unit` for the noun in the W329 per-stream
/// message (`"days"` → `"day"`). A unit that does not end in `s` is returned
/// unchanged.
fn cadence_word(time_unit: &str) -> &str {
    time_unit.strip_suffix('s').unwrap_or(time_unit)
}

/// Run one IF2 chain (called from thread::scope).
fn run_one_chain(
    chain_id: usize,
    config: &FitRunConfig,
    per_chain_params: Option<&[EstimatedParam]>,
    task: Option<&crate::progress::Task>,
    stage_dir: Option<&str>,
) -> Result<IF2Result, sim::error::SimError> {
    let chain_seed = crate::util::derive_chain_seed(config.seed, chain_id);
    let if2_params = per_chain_params.unwrap_or(&config.estimated_params);

    let process = config.build_process();
    let obs_model = config.build_obs_model();

    let n_iter = config.if2_config.n_iterations;

    // Streaming trace writer (opt-in via stage_dir). Each chain writes
    // its own `chain_N/parameter_traces.tsv` from inside the IF2
    // progress callback, one row per iteration. Users can `tail -f`
    // during a long scout run to watch parameters move in real time.
    // The post-hoc `write_chain_outputs` overwrites this file with the
    // same column schema after the clean-PF re-eval populates
    // `IF2IterResult.loglik`; until then the `loglik` column is `NA`
    // (the in-run perturbed loglik lives in `if2_perturbed_loglik`).
    //
    // Per-chain, single-writer-per-thread: no Mutex needed. RefCell
    // lets the `Fn` closure mutate the BufWriter.
    let trace_writer: Option<std::cell::RefCell<std::io::BufWriter<std::fs::File>>>
        = stage_dir.and_then(|dir| {
            let chain_dir = format!("{}/chain_{}", dir, chain_id + 1);
            if std::fs::create_dir_all(&chain_dir).is_err() { return None; }
            let path = format!("{}/parameter_traces.tsv", chain_dir);
            std::fs::File::create(&path).ok().map(|f| {
                use std::io::Write;
                let mut w = std::io::BufWriter::new(f);
                let _ = writeln!(w, "# {}", crate::version::VERSION);
                let _ = write!(w, "iteration\tloglik\tif2_perturbed_loglik");
                for spec in if2_params {
                    let _ = write!(w, "\t{}", spec.name);
                }
                let _ = writeln!(w);
                std::cell::RefCell::new(w)
            })
        });

    let progress_cb = |iter: usize, loglik: f64, param_means: &[f64]| {
        // Passive bar tick. The callback fires once per iteration in order,
        // so `inc(1)` tracks position = iter+1 exactly. `Task` handles
        // Pretty (redraw) / Plain (throttled log line) / None (no-op) — no
        // mode branching here.
        if let Some(t) = task {
            t.set(crate::progress::ll(loglik));
            t.inc(1);
        }
        // Stream one row per iteration. `loglik` column is `NA` until
        // the post-hoc clean-PF re-eval; `if2_perturbed_loglik` is the
        // in-run value the callback already has. Flush every 10 rows
        // so `tail -f` sees output without waiting on the BufWriter.
        if let Some(cell) = &trace_writer {
            use std::io::Write;
            if let Ok(mut w) = cell.try_borrow_mut() {
                let _ = write!(w, "{}\tNA\t{:.2}", iter, loglik);
                for spec in if2_params {
                    let v = param_means.get(spec.index).copied().unwrap_or(f64::NAN);
                    let _ = write!(w, "\t{}", v); // round-trippable; gh#266
                }
                let _ = writeln!(w);
                if iter % 10 == 0 || iter + 1 == n_iter { let _ = w.flush(); }
            }
        }
    };

    // The chain runner (`run_chains_with_per_chain_params`) decides what a
    // failure means: a PFDegenerate is skip-and-continue (one bad chain
    // shouldn't kill a multi-chain fit), other errors are fatal. So we
    // propagate the error rather than `process::exit`-ing here. Flush the
    // streaming trace first so a bailed chain doesn't leave a truncated
    // partial file.
    let result = match run_if2_with_progress(
        &process, &obs_model, &config.base_params, if2_params,
        &config.if2_config, chain_seed,
        Some(&progress_cb),
    ) {
        Ok(r) => r,
        Err(e) => {
            if let Some(cell) = &trace_writer {
                use std::io::Write;
                if let Ok(mut w) = cell.try_borrow_mut() { let _ = w.flush(); }
            }
            return Err(e);
        }
    };

    // Final flush so partial buffers don't leave the file truncated
    // if the post-hoc rewrite is delayed.
    if let Some(cell) = &trace_writer {
        use std::io::Write;
        if let Ok(mut w) = cell.try_borrow_mut() { let _ = w.flush(); }
    }

    // Final metric on the bar; the driver clears it after the par_iter
    // (`Task::finish` consumes, so it can't be called on the borrowed Task
    // here). The post-loop `eprintln!("best chain: …")` carries the summary.
    if let Some(t) = task {
        t.set(crate::progress::ll(result.final_loglik));
    }

    Ok(result)
}

/// Run N chains with optional per-chain EstimatedParam overrides (for scout random starts).
pub fn run_chains_with_per_chain_params(
    config: &FitRunConfig,
    per_chain_params: Option<&[Vec<EstimatedParam>]>,
    collector: &DiagnosticCollector,
    stage_dir: Option<&str>,
) -> Result<ChainResults, String> {
    eprintln!("running {} chains × {} particles × {} iterations, cooling={}, dt={}",
        config.n_chains, config.if2_config.n_particles, config.if2_config.n_iterations,
        config.if2_config.cooling_fraction, config.if2_config.dt);

    // GH #14: one `Reporter` hands out a per-chain `Task` rendered as a
    // coordinated stack. The Reporter honors --progress (Pretty=bars,
    // Plain=throttled `chain N pos/len ll=…` log lines, None=silent), so the
    // callback no longer branches on mode.
    let reporter = crate::progress::Reporter::new();
    let bars: Vec<crate::progress::Task> = (0..config.n_chains)
        .map(|chain_id| reporter.task(
            config.if2_config.n_iterations as u64,
            format!("chain {}", chain_id + 1), "it"))
        .collect();

    // Preflight transform report
    print_preflight(config, per_chain_params, collector);

    let results: Vec<(usize, IF2Result)> = (0..config.n_chains)
        .into_par_iter()
        .filter_map(|chain_id| {
            let per_chain = per_chain_params.map(|pcp| &pcp[chain_id][..]);
            match run_one_chain(chain_id, config, per_chain, Some(&bars[chain_id]), stage_dir) {
                Ok(result) => Some((chain_id, result)),
                // gh#110 (IF2 follow-up): a chain whose IF2 search wandered
                // into the PF-degenerate region is skipped with a BadInit
                // diagnostic and omitted from downstream R̂/agreement/winner
                // aggregation; the surviving chains continue. Mirrors PMMH's
                // skip-and-continue (`pmmh.rs`) so one bad chain can't kill an
                // otherwise-healthy multi-chain fit. The loud diagnostic
                // (collector + stderr) keeps the skip visible — never silent.
                Err(e @ sim::error::SimError::PFDegenerate { .. }) => {
                    // A statistically-degenerate chain (ESS collapse / all
                    // particles dead) is skipped with a BadInit diagnostic; the
                    // surviving chains continue. (A deterministic compute-budget
                    // bail, PFIterationBudget, is fatal — it falls through to the
                    // structural `Err(other)` arm below, since it trips
                    // identically for every chain.)
                    let (reason, label) = match &e {
                        sim::error::SimError::PFDegenerate { kind, obs_window, elapsed_s } =>
                            (format!("{:?} at obs_window={} after {:.2}s", kind, obs_window, elapsed_s),
                             "PF degenerate"),
                        _ => unreachable!(),
                    };
                    // gh#513: report the start THIS chain ran from, not the
                    // configured one. `per_chain` is the same slice
                    // `run_one_chain` resolved with above, and `.initial` is
                    // what `if2.rs` seeds each estimated slot from — so this is
                    // the value that actually produced the degeneracy.
                    //
                    // Reading `config.base_params[spec.index]` here made the
                    // diagnostic contradict its own advice: under any dispersing
                    // `init` mode it printed the declared start, then told the
                    // reader to consult `chain_starts.tsv`, which disagreed. The
                    // one place a user looks to find out why a chain died is the
                    // worst place to name a value no chain used.
                    let init_specs = per_chain.unwrap_or(&config.estimated_params);
                    let params: std::collections::BTreeMap<String, f64> =
                        init_specs.iter()
                            .map(|spec| (spec.name.clone(), spec.initial))
                            .collect();
                    collector.push(DiagnosticKind::BadInit {
                        chain_id, params, reason: reason.clone(),
                    });
                    eprintln!("  chain {}: \x1b[31m✗ skipped\x1b[0m — {} ({})",
                        chain_id + 1, label, reason);
                    // The bar is cleared in the post-loop finish; the skip
                    // is already loud on stderr above.
                    None
                }
                // Any non-degeneracy error is structural (config bug,
                // unknown compartment, …) — every chain would hit it, so
                // there is no survivor to fall back to. Fail loudly.
                Err(other) => {
                    eprintln!("chain {} error: {:?}", chain_id + 1, other);
                    std::process::exit(1);
                }
            }
        })
        .collect();

    // Clear all chain bars now that the parallel phase is done (`Task::finish`
    // consumes, so it can't run on the per-chain borrow inside the loop). The
    // per-chain summary is the `best chain: …` report below.
    for t in bars { t.finish(); }

    // gh#110 (IF2 follow-up): if EVERY chain degenerated there is no
    // inference to report. Fail with an actionable message rather than
    // letting a later stage trip over an empty result set.
    if results.is_empty() {
        eprintln!(
            "error: all {} IF2 chain(s) bailed via the PF degeneracy watchdog — no usable \
             chain. This is sustained ESS collapse (R0 at its bound, σ too large, or too few \
             particles): raise --particles or tighten parameter bounds (see the per-chain \
             errors above).",
            config.n_chains);
        std::process::exit(1);
    }

    // Evaluate true (unperturbed) loglik at selected iterations for ALL chains.
    // Every 10 iterations, run a clean PF at the filter mean params.
    let eval_interval = 10;
    let mut results = results;
    {
        let n_eval_particles = config.if2_config.n_particles.min(500); // cap at 500 for speed
        eprintln!("\nevaluating loglik (every {} iterations, all {} chains)...",
            eval_interval, results.len());

        for (chain_id, result) in results.iter_mut() {
            for it in &mut result.iterations {
                if it.iteration % eval_interval == 0 || it.iteration == config.if2_config.n_iterations - 1 {
                    // gh#224: a ruled-out θ scores −∞; a structural error
                    // (model/config can't run) aborts the clean-eval rather
                    // than silently reporting a degenerate true-loglik.
                    it.loglik = run_quick_pfilter(
                        config, &it.param_means,
                        n_eval_particles,
                        config.seed + *chain_id as u64 * 1000 + it.iteration as u64,
                    ).map_err(|e| format!(
                        "if2 clean-eval: structural error at chain {} iteration {}: {}",
                        *chain_id + 1, it.iteration, e))?;
                }
            }
            // Overwrite final_loglik with the true loglik
            let true_ll = result.iterations.last()
                .map(|it| it.loglik).unwrap_or(f64::NEG_INFINITY);
            result.final_loglik = true_ll;
            eprint!("\r  chain {}: ll={:.1}    ", *chain_id + 1, true_ll);
        }
        eprintln!();
    }

    // Compute chain agreement (Â) on the per-iteration param-mean
    // trajectories — independent of clean-eval scoring.
    let chain_agreement = compute_chain_agreement(&results, &config.estimated_params, config.if2_config.n_iterations);

    // Step 6 (proposal §Proposal 1): clean-eval re-scoring at high
    // particle count and M replicates. The winner is the argmax over
    // logmeanexp-combined logliks across chains' IF2 final-iteration
    // means, matching pomp's coef(mif2_out) + pfilter convention.
    // Replaces the prior `argmax over result.final_loglik`, which was
    // driven by 500-particle in-run PF noise and exhibited a ~40-nat
    // extraction bias on production runs. The in-run trace above is
    // preserved for diagnostics (Unit B territory).
    eprintln!("\nloglik-eval: re-scoring final-iter θ̂ ({} chains × {} replicates @ {} particles)...",
        results.len(), config.loglik_eval.n_replicates, config.loglik_eval.n_particles);
    let loglik_eval_outcome = loglik_eval::run_loglik_eval(
        config, &results, &config.loglik_eval, config.seed,
    ).unwrap_or_else(|e| {
        eprintln!("error: loglik-eval failed: {}", e);
        std::process::exit(1);
    });

    let (best_chain, best_loglik, best_se) =
        select_winner_summary(&loglik_eval_outcome);

    // gh#226. Whole-fit backstop: the clean-eval winner across every
    // surviving chain reached no finite log-likelihood. IF2 has no MH
    // acceptance, so `best_loglik` non-finite is the whole signal — the
    // reachable surface is uniformly `-inf` at every evaluated θ. Without
    // this, the run selects a `-inf` "winner", writes a degenerate
    // fit_state, and exits 0. Fires ONLY when NOT ONE chain is finite (a
    // single finite chain makes the winner finite → no fire); the
    // is_empty guard above already handles the all-PFDegenerate case.
    if sim::inference::no_finite_anchor(best_loglik) {
        collector.push(DiagnosticKind::InitialLoglikInfinite);
        collector.render_to_stderr();
        if let Some(dir) = stage_dir {
            let _ = collector.write_json(&format!("{}/diagnostics.json", dir));
        }
        return Err(format!(
            "if2: all {} surviving chain(s) reached no finite log-likelihood \
             (best = {}). The likelihood surface is `-inf` at every evaluated \
             θ. Most often the starting values sit in an impossible region — \
             check those first (try `--init lhs` or a different start); less \
             often the data are impossible under this model, or a recoverable \
             error fires at every θ (gh#226). Also check the observation model \
             and parameter bounds.",
            config.n_chains, best_loglik));
    }

    // Report. `best_se` is derived locally; we log it here but don't
    // store it on `ChainResults` — readers that need it go to
    // `loglik_eval.per_chain[overall_winner_idx]`.
    eprintln!("\nbest chain: {} (loglik={:.2} ± {:.2})",
        best_chain + 1, best_loglik, best_se);
    if config.n_chains > 1 {
        let logliks: Vec<f64> = loglik_eval_outcome.per_chain.iter()
            .map(|s| s.loglik).collect();
        eprintln!("chain clean logliks: [{}]",
            logliks.iter().map(|l| format!("{:.1}", l)).collect::<Vec<_>>().join(", "));
    }

    // Report Â (chain agreement) with diagnostic warnings.
    //
    // gh#45: filter NaN entries — these come from degenerate-W param
    // chains (typically refine under cold cooling, where the
    // perturbation has cooled out and per-iteration filter means stop
    // moving). For those params the G-R formula has no diagnostic
    // meaning; the compound gate's Δ_dB leg carries the verdict.
    if config.n_chains > 1 {
        let finite_agreements: Vec<(String, f64)> = chain_agreement.iter()
            .filter(|(_, &r)| r.is_finite())
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        let n_total = chain_agreement.len();
        let n_finite = finite_agreements.len();
        let logliks: Vec<f64> = results.iter().map(|(_, r)| r.final_loglik).collect();
        let ll_spread = logliks.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - logliks.iter().cloned().fold(f64::INFINITY, f64::min);

        // Â = chain agreement (renamed from Rhat: this is not a
        // posterior mixing statistic; it measures IF2 optimizer chain
        // agreement). Suppress the leg entirely when every param's W
        // collapsed; print per-param "n/a (W ≈ 0)" alongside finite
        // entries when only some collapsed.
        if n_finite == 0 {
            eprintln!("\nÂ: suppressed — within-chain variance below numerical \
                       threshold for all estimated params (typical for refine \
                       under cold cooling). Rely on Δ_dB leg of the compound \
                       gate above for the convergence verdict.");
        } else {
            // Name the statistic and the threshold. Â is IF2 chain agreement,
            // NOT a posterior R̂, and the band comes from the gate that will
            // judge this stage — a literal here diverged from `a_thresh` the
            // moment the default moved to 1.01, so the same number printed ✓
            // at the end of the stage and ✗ in `fit summary`'s gate block.
            eprintln!("\nÂ (IF2 chain agreement), threshold {:.2}:",
                config.gate.a_thresh);
            for spec in &config.estimated_params {
                match chain_agreement.get(&spec.name) {
                    Some(&r) if r.is_finite() => {
                        let status = match config.gate.a_band(r) {
                            super::gating::AgreementBand::Pass => "\x1b[32m✓\x1b[0m",
                            super::gating::AgreementBand::SoftWarn => "\x1b[33m~\x1b[0m",
                            super::gating::AgreementBand::Fail => "\x1b[31m✗\x1b[0m",
                            // Unreachable: `r.is_finite()` is the match guard.
                            super::gating::AgreementBand::NotAssessed => "n/a",
                        };
                        eprintln!("  {:12} Â={:.3} {}", spec.name, r, status);
                    }
                    Some(_) => {
                        eprintln!("  {:12} Â=n/a (W ≈ 0; rely on Δ_dB)", spec.name);
                    }
                    None => {}
                }
            }
        }

        // Diagnostics fire only when at least one finite Â value
        // exists AND it crosses the threshold. NaN entries can't
        // contribute (the compound gate's Δ_dB leg already covers
        // the all-suppressed case).
        if n_finite > 0 {
            let max_chain_agreement = finite_agreements.iter()
                .map(|(_, r)| *r)
                .fold(0.0_f64, f64::max);
            if max_chain_agreement > 1.5 && ll_spread > 50.0 {
                collector.push(DiagnosticKind::MultimodalLikelihood {
                    ll_spread, max_chain_agreement,
                });
            } else if max_chain_agreement > 1.1 {
                let n_unconverged = finite_agreements.iter()
                    .filter(|(_, r)| *r > 1.1).count();
                collector.push(DiagnosticKind::ConvergenceIncomplete {
                    max_chain_agreement, n_unconverged, n_total,
                });
            }
        }
    }

    // gh#audit-H4. LowESSAtMLE — wired from the clean-eval
    // FilterStats already aggregated per chain. Threshold: ess_min < 5%
    // of n_particles at the MLE θ̂. The MLE is exactly the regime
    // where ESS *should* be highest; if it isn't, the filter is
    // struggling at the point estimate and the loglik estimate has
    // wide variance. ParamNearBound: any chain's clean-eval θ̂ within
    // 1% of an estimated parameter's bounds (when bounds exist).
    let n_eval_particles = config.loglik_eval.n_particles;
    let ess_threshold = 0.05 * n_eval_particles as f64;
    for chain in &loglik_eval_outcome.per_chain {
        let fs = &chain.filter_stats;
        if fs.ess_min.is_finite() && fs.ess_min < ess_threshold {
            collector.push(DiagnosticKind::LowESSAtMLE {
                ess_mean:    fs.ess_mean,
                ess_min:     fs.ess_min,
                n_particles: n_eval_particles,
            });
        }
        for spec in config.estimated_params.iter() {
            let v = chain.theta[spec.index];
            let (lo, hi) = (spec.lower, spec.upper);
            let span = hi - lo;
            // Skip when bounds are not informative (zero span or
            // unbounded sentinels). Both finite + span > 0 is the
            // meaningful case for "near a bound."
            if span > 0.0 && lo.is_finite() && hi.is_finite() {
                let near_lo = (v - lo) / span;
                let near_hi = (hi - v) / span;
                if near_lo < 0.01 {
                    collector.push(DiagnosticKind::ParamNearBound {
                        param: spec.name.clone(), value: v, bound: lo,
                        bound_type: "lower".to_string(),
                    });
                } else if near_hi < 0.01 {
                    collector.push(DiagnosticKind::ParamNearBound {
                        param: spec.name.clone(), value: v, bound: hi,
                        bound_type: "upper".to_string(),
                    });
                }
            }
        }
    }

    Ok(ChainResults {
        results,
        best_chain,
        best_loglik,
        chain_agreement,
        loglik_eval: loglik_eval_outcome,
    })
}

impl ChainResults {
    /// Per-chain clean-eval log-likelihoods in chain-id order. Used by
    /// scout/refine/validate to populate `FitState.chain_eval_logliks`
    /// for the compound scout-convergence gate.
    pub fn chain_eval_logliks(&self) -> Vec<f64> {
        let mut v: Vec<(usize, f64)> = self.loglik_eval.per_chain.iter()
            .map(|s| (s.chain_id, s.loglik)).collect();
        v.sort_by_key(|(id, _)| *id);
        v.into_iter().map(|(_, ll)| ll).collect()
    }

    /// Per-chain clean-eval standard errors in chain-id order, parallel
    /// to `chain_eval_logliks`.
    pub fn chain_eval_ses(&self) -> Vec<f64> {
        let mut v: Vec<(usize, f64)> = self.loglik_eval.per_chain.iter()
            .map(|s| (s.chain_id, s.se)).collect();
        v.sort_by_key(|(id, _)| *id);
        v.into_iter().map(|(_, se)| se).collect()
    }

    /// Estimated-param θ̂ of the overall clean-eval winner. Indexed by
    /// `EstimatedParam::index`, parallel to `IF2Result.mle`.
    ///
    /// **Use this — not `IF2Result.mle` of the winning chain — anywhere
    /// the user-facing "MLE" parameters are needed** (e.g. building
    /// `start_values` for a downstream stage, writing
    /// `mle_params.toml`, status / summary tables). `IF2Result.mle` is
    /// the IF2 chain's argmax over its own noisy `if2_perturbed_loglik`
    /// — a separate, biased estimator. The clean-eval θ̂ is the
    /// chain's IF2 final-iteration mean (Ionides et al. 2015's
    /// theoretical estimator), unchanged by clean re-evaluation; what
    /// the clean re-eval changes is the *loglik* attached to that θ̂,
    /// and the cross-chain selection of which chain's θ̂ to report.
    pub fn winner_theta(&self) -> &[f64] {
        &self.loglik_eval.per_chain[self.loglik_eval.overall_winner_idx].theta
    }

}

/// Pure helper: extract the (chain_id, ll, se) summary from a
/// `LoglikEvalOutcome`. Factored out so the wiring change in
/// `run_chains_with_per_chain_params` is unit-testable without paying
/// for a real IF2 + PF run. Tested in `tests::winner_summary_*`.
fn select_winner_summary(
    outcome: &loglik_eval::LoglikEvalOutcome,
) -> (usize, f64, f64) {
    let s = &outcome.per_chain[outcome.overall_winner_idx];
    (s.chain_id, s.loglik, s.se)
}

/// Compute chain agreement (Â) across IF2 chains (last half of
/// iterations). The underlying formula is Gelman-Rubin 1992 R̂; the
/// renamed output label reflects that this is applied to IF2
/// optimizer chains, not posterior samples. See
/// docs/dev/proposals/2026-04-24-if2-scout-findings-remediation.md.
///
/// See [`gelman_rubin_1992`] for the split-chain / rank-norm
/// caveat.
pub fn compute_chain_agreement(
    results: &[(usize, IF2Result)],
    if2_params: &[EstimatedParam],
    n_iterations: usize,
) -> HashMap<String, f64> {
    let n_chains = results.len();
    if n_chains < 2 { return HashMap::new(); }

    // Im25 in 2026-04-19 inference review batch 3: use each chain's
    // own last-half rather than `n_iterations` uniformly. Resumed
    // chains have `iterations.len() > n_iterations`; the old formula
    // was `skip(n_iterations − n_tail)` for all chains — so a
    // resumed chain's "last half" started at an absolute iteration
    // index that didn't correspond to the physical last half of its
    // trace. Now each chain defines its own last-half window.
    let mut agreement_map = HashMap::new();

    let chain_tail = |r: &IF2Result, spec: &EstimatedParam| -> Vec<f64> {
        let len = r.iterations.len().max(n_iterations);
        let n_tail = (len / 2).max(1);
        r.iterations.iter()
            .skip(r.iterations.len().saturating_sub(n_tail))
            .map(|it| it.param_means[spec.index])
            .collect()
    };

    for spec in if2_params {
        let chain_means: Vec<f64> = results.iter().map(|(_, r)| {
            let tail = chain_tail(r, spec);
            tail.iter().sum::<f64>() / tail.len() as f64
        }).collect();

        let chain_vars: Vec<f64> = results.iter().map(|(_, r)| {
            let tail = chain_tail(r, spec);
            let m = tail.iter().sum::<f64>() / tail.len() as f64;
            tail.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (tail.len() - 1).max(1) as f64
        }).collect();

        // For the G-R between/within formula, use the tail length of
        // the shortest chain — the formula uses a single N per chain
        // and conservatism argues for the min when lengths differ.
        let min_tail = results.iter()
            .map(|(_, r)| chain_tail(r, spec).len())
            .min().unwrap_or(0).max(1) as f64;
        let grand_mean = chain_means.iter().sum::<f64>() / n_chains as f64;
        let between = chain_means.iter().map(|&m| (m - grand_mean).powi(2)).sum::<f64>()
            * min_tail / (n_chains - 1).max(1) as f64;
        let within = chain_vars.iter().sum::<f64>() / n_chains as f64;

        // gh#45: under cold cooling (refine default cf=0.05) the
        // within-chain variance W collapses to ~0 by mid-tail —
        // perturbations are cooled out and per-iteration filter
        // means stop moving. The G-R formula Â = √(V̂/W) then blows
        // up regardless of actual between-chain agreement, emitting
        // misleading ✗ verdicts on fits the compound gate's Δ_dB
        // leg correctly identifies as converged.
        //
        // Detect numerically-degenerate W relative to the parameter
        // scale and return NaN; the printing/diagnostic layer
        // recognises NaN and suppresses the chain_agreement leg
        // for that param (or the whole leg if all params suppress).
        // Threshold: within-chain SD < 1e-6 of grand_mean (i.e.
        // chain has flatlined to within parts-per-million of its
        // tail mean). Parameters near zero use an absolute floor
        // to avoid division-by-zero pathology.
        let scale = grand_mean.abs().max(1e-15);
        let degenerate_w_threshold = (1e-6 * scale).powi(2);
        let agreement = if within > degenerate_w_threshold {
            (((min_tail - 1.0) / min_tail * within + between / min_tail) / within).sqrt()
        } else {
            // W is numerically zero relative to the parameter scale;
            // Â would diverge with no diagnostic meaning.
            f64::NAN
        };

        agreement_map.insert(spec.name.clone(), agreement);
    }

    agreement_map
}

/// R-hat and ESS diagnostics for a single parameter: everything the chains
/// yielded, or the named precondition that refused them all.
///
/// The rank-normalized statistics of Vehtari, Gelman, Simpson, Carpenter &
/// Bürkner (2021), _Bayesian Analysis_ 16(2):667-718, are carried **whole**, as
/// the [`RankConvergence`] `rank_convergence` returned. Re-listing a subset of
/// its fields here is what dropped `rhat_bulk` and `rhat_folded` — the two
/// halves whose split is the answer to "*why* is R̂ high" — before they reached
/// any surface a user reads.
///
/// `rhat_classic` is the classic Gelman & Rubin (1992) statistic, kept because
/// the healthy band published in `docs/workflow.md` was calibrated against it,
/// because a reader comparing an old fit to a new one needs both numbers to be
/// legible, and because the rank-normalized statistic is BOUNDED (ceiling ~1.85
/// for two chains, ~4.5 for eight) and so cannot express severity while this one
/// can — see `docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md`.
///
/// `ess_bulk` needs no R-hat gate: it uses the *between-chain* variance, so it
/// does not overstate the effective N when chains sit in different modes. That
/// is what retired the previous R-hat-gated pooled ESS, which reported nothing
/// exactly when a fit most needed a number (gh#299).
///
/// `ess_per_chain` holds the Geyer initial-positive-sequence ESS for each chain
/// — interpretable as that chain's effective N for whatever distribution it is
/// sampling, *regardless* of cross-chain agreement. It answers a different
/// question from `ess_bulk`: whether chains stuck in different modes are each
/// mixing well within their own mode (large per-chain ESS) or are both stuck and
/// non-stationary.
#[derive(Debug, Clone, PartialEq)]
pub enum RhatEss {
    /// The estimator ran. Individual statistics may still be `NaN` or `±inf`;
    /// each says so for itself.
    Scored {
        /// Every rank-normalized statistic, as computed.
        rank: RankConvergence,
        /// Gelman & Rubin (1992) R-hat: unsplit chains, raw scale.
        rhat_classic: f64,
        /// Per-chain Geyer ESS, one entry per chain.
        ess_per_chain: Vec<f64>,
    },
    /// `rank_convergence` refused the input, naming which precondition failed
    /// and with what numbers. Rendered by name rather than left as a bare
    /// `NaN`, which reads as a numerical failure and hides which precondition
    /// was missed (gh#84).
    NotScored(ConvergenceError),
}

impl RhatEss {
    /// The rank-normalized statistics, or `None` when the input was refused.
    /// The one accessor everything else goes through — there is no second copy
    /// of any of its fields here to drift from it.
    pub fn rank(&self) -> Option<&RankConvergence> {
        match self {
            Self::Scored { rank, .. } => Some(rank),
            Self::NotScored(_) => None,
        }
    }

    /// Why there are no rank-normalized statistics, when there are none.
    pub fn refusal(&self) -> Option<&ConvergenceError> {
        match self {
            Self::Scored { .. } => None,
            Self::NotScored(e) => Some(e),
        }
    }

    /// Classic Gelman & Rubin (1992) R-hat; `NaN` when the input was refused.
    pub fn rhat_classic(&self) -> f64 {
        match self {
            Self::Scored { rhat_classic, .. } => *rhat_classic,
            Self::NotScored(_) => f64::NAN,
        }
    }

    /// Per-chain Geyer ESS; empty when the input was refused.
    pub fn ess_per_chain(&self) -> &[f64] {
        match self {
            Self::Scored { ess_per_chain, .. } => ess_per_chain,
            Self::NotScored(_) => &[],
        }
    }
}

/// Compute R-hat and ESS diagnostics from per-chain parameter traces.
/// `chains[chain_id]` is a Vec of param values (one per sample).
///
/// Structural preconditions — ≥ 2 chains with ≥ 4 samples each, all chains
/// equal length — are enforced by
/// [`rank_convergence`](sim::inference::convergence::rank_convergence), which
/// names the one that failed. Below them the result is [`RhatEss::NotScored`]
/// carrying that name and its numbers.
pub fn compute_rhat_ess(chains: &[Vec<f64>]) -> RhatEss {
    use sim::inference::convergence::rank_convergence;
    use sim::inference::pmmh::mcmc_ess;

    let rank = match rank_convergence(chains) {
        Ok(r) => r,
        Err(e) => return RhatEss::NotScored(e),
    };

    // Classic Gelman & Rubin (1992): between-chain variance of the chain
    // MEANS over unsplit chains on the raw scale. Reported alongside the
    // rank-normalized statistic, never as the headline — on the ebola
    // 8-chain PGAS fit of gh#84 it read 1.03 where the rank-normalized
    // statistic read 1.13, i.e. inside the published healthy band on a
    // parameter whose chains drift within their own runs.
    let n_chains = chains.len();
    let n_samples = chains[0].len() as f64;
    let chain_means: Vec<f64> = chains.iter()
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let chain_vars: Vec<f64> = chains.iter().zip(&chain_means)
        .map(|(c, &m)| c.iter().map(|&x| (x - m).powi(2)).sum::<f64>()
            / (c.len() - 1).max(1) as f64)
        .collect();
    let grand_mean = chain_means.iter().sum::<f64>() / n_chains as f64;
    let between = chain_means.iter().map(|&m| (m - grand_mean).powi(2)).sum::<f64>()
        * n_samples / (n_chains - 1).max(1) as f64;
    let within = chain_vars.iter().sum::<f64>() / n_chains as f64;
    let rhat_classic = if within > 0.0 {
        (((n_samples - 1.0) / n_samples * within + between / n_samples) / within).sqrt()
    } else { f64::NAN };

    RhatEss::Scored {
        rank,
        rhat_classic,
        ess_per_chain: chains.iter().map(|c| mcmc_ess(c)).collect(),
    }
}

/// The R̂ above which a parameter draws a `RhatHigh` diagnostic and the
/// end-of-stage report marks it.
///
/// Unchanged in value from what every Bayesian stage has always applied; what
/// changed in gh#84 is the STATISTIC — `RhatEss::rhat` is now the
/// rank-normalized split statistic of Vehtari et al. (2021) rather than the
/// classic Gelman & Rubin (1992) one, so the same 1.1 is a stricter bar.
/// Vehtari et al. recommend 1.01 for the rank-normalized statistic; adopting
/// that is a policy decision about what camdl certifies, tracked on gh#84, and
/// is deliberately NOT taken here.
pub const RHAT_REPORT_THRESHOLD: f64 = 1.1;

/// Per-parameter convergence diagnostics for one Bayesian stage — the single
/// shape every sampler fills and every renderer and serializer reads.
///
/// PGAS, PMMH and nuts previously each carried their own parallel maps and
/// their own copy of the report loop, which is how a new statistic ends up
/// live in one sampler and absent in another. They now differ only in how they
/// extract a parameter's per-chain trace from their own result type.
pub struct StageConvergence(Vec<(String, RhatEss)>);

use std::collections::BTreeMap;
use sim::inference::convergence::{ConvergenceError, RankConvergence, RhatRefusal};

impl StageConvergence {
    /// Score each `(param name, chains[chain][draw])` pair. Order is preserved
    /// — callers pass their estimated params in declaration order and the
    /// report follows it.
    pub fn compute(entries: impl IntoIterator<Item = (String, Vec<Vec<f64>>)>) -> Self {
        StageConvergence(
            entries.into_iter()
                .map(|(name, chains)| {
                    let d = compute_rhat_ess(&chains);
                    (name, d)
                })
                .collect(),
        )
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &RhatEss)> {
        self.0.iter().map(|(n, d)| (n.as_str(), d))
    }

    /// Rank-normalized R̂ per param — **finite entries only**, so a `max` over
    /// the map is the max over the params that were assessable at all.
    pub fn rhat(&self) -> BTreeMap<String, f64> {
        self.finite_rank(|r| r.rhat)
    }

    /// The **location** half of the headline: rank-normalized split-R̂ without
    /// the fold. Finite entries only, same convention as [`Self::rhat`].
    pub fn rhat_bulk(&self) -> BTreeMap<String, f64> {
        self.finite_rank(|r| r.rhat_bulk)
    }

    /// The **spread** half: the same statistic on `|x − median(x)|`. Its gap
    /// from [`Self::rhat_bulk`] is what says whether chains disagree about
    /// where the posterior sits or about how wide it is.
    pub fn rhat_folded(&self) -> BTreeMap<String, f64> {
        self.finite_rank(|r| r.rhat_folded)
    }

    /// Why each param ABSENT from [`Self::rhat`] is absent.
    ///
    /// Without this the two halves of `rhat` — "assessed and fine" and "could
    /// not be assessed" — are indistinguishable downstream, which is how a fit
    /// with no computable R̂ came to report `converged: true`. Two sources:
    /// a refusal `rank_convergence` named, and an R̂ that evaluated non-finite
    /// (every chain internally constant at its own value — the 0%-acceptance
    /// deadlock — which is not an `Err` but is just as fatal).
    pub fn rhat_not_reported(&self) -> BTreeMap<String, RhatRefusal> {
        self.0.iter()
            .filter_map(|(n, d)| match d {
                RhatEss::NotScored(e) => Some((n.clone(), e.refusal())),
                RhatEss::Scored { rank, .. } if !rank.rhat.is_finite() => {
                    Some((n.clone(), RhatRefusal::NonFiniteRhat))
                }
                RhatEss::Scored { .. } => None,
            })
            .collect()
    }

    /// The refusal behind each entry of [`Self::rhat_not_reported`], **with its
    /// numbers** — "R̂ needs at least 2 chains; got 1" rather than "fewer than 2
    /// chains". Only params `rank_convergence` refused appear: a non-finite R̂
    /// is not a refusal of the input, so it has no detail to carry.
    pub fn rhat_refusal_detail(&self) -> BTreeMap<String, ConvergenceError> {
        self.0.iter()
            .filter_map(|(n, d)| d.refusal().map(|e| (n.clone(), e.clone())))
            .collect()
    }

    /// Classic Gelman & Rubin (1992) R̂ per param, finite entries only.
    pub fn rhat_classic(&self) -> BTreeMap<String, f64> {
        self.finite(|d| d.rhat_classic())
    }

    /// Bulk-ESS per param. Every param gets an entry, non-finite included: a
    /// NaN serializes to JSON `null` and the loader reads it as absent, which
    /// is the encoding `PosteriorDiagnostics::min_ess_status` expects.
    pub fn ess_bulk(&self) -> BTreeMap<String, f64> {
        self.per_param_rank(|r| r.ess_bulk)
    }

    /// Tail-ESS per param, same key set as [`Self::ess_bulk`].
    pub fn ess_tail(&self) -> BTreeMap<String, f64> {
        self.per_param_rank(|r| r.ess_tail)
    }

    /// Per-chain Geyer ESS per param, omitting params with no per-chain values.
    pub fn ess_per_chain(&self) -> BTreeMap<String, Vec<f64>> {
        self.0.iter()
            .filter(|(_, d)| !d.ess_per_chain().is_empty())
            .map(|(n, d)| (n.clone(), d.ess_per_chain().to_vec()))
            .collect()
    }

    /// The convergence half of a `*_summary.json`, as one object.
    ///
    /// Every Bayesian sampler merges this into its own stage-specific fields,
    /// so a statistic cannot be live in one sampler's summary and silently
    /// absent from another's — the divergence [`StageConvergence`] exists to
    /// prevent, and which three hand-written `json!` blocks reintroduced.
    /// [`crate::fit::method_result::ConvergenceMaps`] is the matching reader.
    pub fn summary_fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut m = serde_json::Map::new();
        let mut put = |k: &str, v: serde_json::Value| {
            m.insert(k.to_string(), v);
        };
        put("rhat", serde_json::json!(self.rhat()));
        put("rhat_bulk", serde_json::json!(self.rhat_bulk()));
        put("rhat_folded", serde_json::json!(self.rhat_folded()));
        put("rhat_classic", serde_json::json!(self.rhat_classic()));
        put("rhat_not_reported", serde_json::json!(self.rhat_not_reported()));
        put("rhat_refusal_detail", serde_json::json!(self.rhat_refusal_detail()));
        put("ess", serde_json::json!(self.ess_bulk()));
        put("ess_tail", serde_json::json!(self.ess_tail()));
        put("ess_per_chain", serde_json::json!(self.ess_per_chain()));
        m
    }

    fn finite(&self, f: impl Fn(&RhatEss) -> f64) -> BTreeMap<String, f64> {
        self.0.iter()
            .filter_map(|(n, d)| {
                let v = f(d);
                v.is_finite().then(|| (n.clone(), v))
            })
            .collect()
    }

    /// One rank-normalized statistic per param, finite entries only.
    fn finite_rank(&self, f: impl Fn(&RankConvergence) -> f64) -> BTreeMap<String, f64> {
        self.finite(|d| d.rank().map_or(f64::NAN, &f))
    }

    /// One rank-normalized statistic per param, EVERY param present — the
    /// encoding the ESS maps need (a NaN becomes JSON `null`, read back as
    /// absent).
    fn per_param_rank(&self, f: impl Fn(&RankConvergence) -> f64) -> BTreeMap<String, f64> {
        self.0.iter()
            .map(|(n, d)| (n.clone(), d.rank().map_or(f64::NAN, &f)))
            .collect()
    }

    /// The end-of-stage convergence block, and the `RhatHigh` diagnostics that
    /// go with it. Returns the text so the caller decides where it lands.
    ///
    /// Every line carries bulk-ESS **and** ESS/N. The ratio is not decoration:
    /// Geyer's truncation destabilizes as the integrated autocorrelation time
    /// approaches the run length, so bulk-ESS 11 out of 11200 draws is the
    /// estimator telling you it summed autocorrelations out to nearly the whole
    /// run (gh#84). A parameter that could not be scored at all names the
    /// reason instead of printing a dash.
    ///
    /// A parameter ABOVE the band gets a second line decomposing its R̂ into
    /// `max(rhat_bulk, rhat_folded)` and naming which half is larger. The
    /// headline says a fit did not converge; the decomposition says what to do
    /// about it — a location disagreement is a warm-up/drift problem, a spread
    /// disagreement points at per-chain effective diversity
    /// (`docs/dev/proposals/2026-08-22-reporting-two-rhat-estimators.md`).
    pub fn report(
        &self,
        collector: &sim::inference::diagnostic::DiagnosticCollector,
        rhat_threshold: f64,
    ) -> String {
        use crate::fit::method_result::{RhatBand, RHAT_CONVERGED_THRESHOLD};
        let mut out = format!(
            "\nR̂ (rank-normalized split, Vehtari et al. 2021) / ESS — \
             R̂ threshold {RHAT_CONVERGED_THRESHOLD}:\n"
        );
        for (name, d) in self.iter() {
            let Some(r) = d.rank() else {
                // `rank()` is `None` exactly when `refusal()` is `Some`.
                let why = d.refusal().expect("an unscored param carries its refusal");
                out.push_str(&format!("  {:12} not reported — {}\n", name, why));
                continue;
            };
            // The glyph is the CERTIFICATION band — the same one `fit summary`
            // uses — so one R̂ cannot print ✓ when the fit finishes and ✗ when
            // the user runs `fit summary` on the same directory. `rhat_threshold`
            // is a different bar: it draws the FINDING. `glyph_with_finding`
            // keeps the glyph from ever being greener than the finding allows.
            let drew_finding = r.rhat > rhat_threshold;
            let band = RhatBand::of(r.rhat);
            let status = match band.glyph_with_finding(drew_finding) {
                "✓" => "\x1b[32m✓\x1b[0m",
                "~" => "\x1b[33m~\x1b[0m",
                "✗" => "\x1b[31m✗\x1b[0m",
                other => other,
            };
            let tail = if r.ess_tail.is_finite() {
                format!("{:.0}", r.ess_tail)
            } else {
                "—".to_string()
            };
            out.push_str(&format!(
                "  {:12} Rhat={:.3} {} ESS bulk={:.0} tail={} ({:.1}% of {} draws)\n",
                name, r.rhat, status, r.ess_bulk, tail,
                100.0 * r.ess_bulk_ratio(), r.n_draws_total,
            ));
            // The decomposition follows the GLYPH, not the finding: it is the
            // "so what do I do" line for anything not certified converged.
            if band != RhatBand::Converged || drew_finding {
                if let Some(driver) = r.rhat_driver() {
                    // The whole ladder on one line. `rhat_classic` is here
                    // because the rank statistic is BOUNDED (ceiling ≈1.85 at
                    // two chains) and cannot express severity, while the
                    // classic one can — it is the only number that separates
                    // "the sampler is dead" from "the sampler mixes badly".
                    out.push_str(&format!(
                        "  {:12}   R̂ = max(bulk {:.3}, folded {:.3}), classic {}; \
                         the {} half is larger — {}\n",
                        "", r.rhat_bulk, r.rhat_folded,
                        crate::fit::method_result::Stat::from_f64(d.rhat_classic())
                            .cell(3, "—"),
                        driver.half(), driver.describe(),
                    ));
                }
                // And whether each chain is mixing well inside its OWN mode,
                // which no cross-chain statistic can say. Similar large values
                // point at multimodality; one small value points at a stuck
                // chain. Different fixes.
                let per_chain = d.ess_per_chain();
                if per_chain.len() >= 2 {
                    let cells: Vec<String> = per_chain
                        .iter()
                        .map(|e| if e.is_finite() { format!("{e:.0}") } else { "—".into() })
                        .collect();
                    out.push_str(&format!(
                        "  {:12}   per-chain ESS [{}]\n", "", cells.join(", ")));
                }
            }
            if drew_finding {
                collector.push(sim::inference::diagnostic::DiagnosticKind::RhatHigh {
                    param: name.to_string(), rhat: r.rhat, threshold: rhat_threshold,
                });
            }
        }
        out
    }
}

/// MAD-based auto rw_sd calibration from chain best-loglik parameters.
///
/// Returns (rw_sd map, n_good_chains) or error if no consensus.
pub fn auto_rw_sd(
    results: &[(usize, IF2Result)],
    if2_params: &[EstimatedParam],
) -> Result<(HashMap<String, f64>, usize), String> {
    let n_chains = results.len();
    if n_chains < 3 {
        return Err("auto rw_sd requires at least 3 chains".into());
    }

    // Collect each chain's best-loglik parameter set
    let chain_params: Vec<Vec<f64>> = results.iter().map(|(_, r)| {
        r.mle.clone()
    }).collect();

    // Per-parameter: compute median and MAD
    let mut medians: Vec<f64> = Vec::new();
    let mut mads: Vec<f64> = Vec::new();

    for spec in if2_params {
        // Filter non-finite values: chains with extreme parameter perturbations
        // can produce NaN (from -inf loglik propagation) or inf. These are dead
        // chains — they contributed nothing to inference. Including them in the
        // MAD would either panic (NaN in sort) or corrupt the scale estimate
        // (inf inflating the deviation).
        let mut values: Vec<f64> = chain_params.iter()
            .map(|p| p[spec.index])
            .filter(|v| v.is_finite())
            .collect();
        if values.len() < 2 {
            medians.push(0.0);
            mads.push(0.0);
            continue;
        }

        let med = median(&mut values);
        let m = mad(&values, med);

        medians.push(med);
        mads.push(m);
    }

    // Classify chains as "good" (all params within 3×MAD of median)
    let good_chains: Vec<bool> = (0..n_chains).map(|c| {
        if2_params.iter().enumerate().all(|(pi, spec)| {
            let v = chain_params[c][spec.index];
            let mad = mads[pi];
            if mad < 1e-15 {
                // All chains agree perfectly on this parameter
                true
            } else {
                (v - medians[pi]).abs() <= 3.0 * mad
            }
        })
    }).collect();

    let n_good = good_chains.iter().filter(|&&g| g).count();

    if n_good < n_chains / 2 {
        // Report which chains diverged and their parameters
        let diverged: Vec<usize> = good_chains.iter().enumerate()
            .filter(|(_, &g)| !g).map(|(i, _)| i + 1).collect();
        return Err(format!(
            "No consensus across chains ({}/{} good). Divergent chains: {:?}\n\
             The likelihood surface may be multimodal or scout iterations are too few.\n\
             Re-run with more iterations or check model specification.",
            n_good, n_chains, diverged
        ));
    }

    if n_good < n_chains {
        let _diverged: Vec<usize> = good_chains.iter().enumerate()
            .filter(|(_, &g)| !g).map(|(i, _)| i + 1).collect();
        eprintln!("warning: {}/{} chains diverged ({:?}), excluded from rw_sd calibration",
            n_chains - n_good, n_chains, _diverged);
    }

    // rw_sd = 0.5 × MAD of good chains
    let mut rw_sd_map = HashMap::new();
    for (pi, spec) in if2_params.iter().enumerate() {
        let good_values: Vec<f64> = (0..n_chains)
            .filter(|&c| good_chains[c])
            .map(|c| chain_params[c][spec.index])
            .collect();

        let good_mad = mad(&good_values, medians[pi]);

        let rw = 0.5 * good_mad;
        // Floor: don't let rw_sd go below 1% of the median (prevents convergence stall)
        let floor = medians[pi].abs() * 0.01;
        rw_sd_map.insert(spec.name.clone(), rw.max(floor));
    }

    Ok((rw_sd_map, n_good))
}

/// Median of a mutable slice (sorts in place).
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n == 0 { return 0.0; }
    if n.is_multiple_of(2) {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

/// Median absolute deviation from a given center.
fn mad(v: &[f64], center: f64) -> f64 {
    let mut abs_devs: Vec<f64> = v.iter().map(|&x| (x - center).abs()).collect();
    median(&mut abs_devs)
}

/// Write per-chain output files: `parameter_traces.tsv` and
/// `final_params.toml` under `<dir>/chain_<N>/`.
///
/// When `loglik_eval` is `Some`, each chain's `final_params.toml` also
/// records the clean-eval winning candidate label and SE for that chain
/// (Step 7, proposal §Proposal 1). PMMH and other consumers that don't
/// run clean-eval pass `None`. The winning θ̂ written into the TOML is
/// also taken from the clean-eval per-chain winner when present (it can
/// be the tail mean or best-in-run iter, not just `result.mle`); this
/// is what makes scout→refine handoff consume the de-biased estimate.
pub fn write_chain_outputs(
    dir: &str,
    results: &[(usize, IF2Result)],
    if2_params: &[EstimatedParam],
    all_param_names: &[String],
    base_params: &[f64],
    compiled: &CompiledModel,
    loglik_eval: Option<&loglik_eval::LoglikEvalOutcome>,
) -> Result<(), String> {
    use std::io::Write;

    for (chain_id, result) in results {
        let chain_dir = format!("{}/chain_{}", dir, chain_id + 1);
        std::fs::create_dir_all(&chain_dir)
            .map_err(|e| format!("cannot create {}: {}", chain_dir, e))?;

        // Parameter traces: post-hoc write is a *fallback* now. The
        // chain-side streaming writer in `run_one_chain` writes this
        // file incrementally during IF2 (so users can `tail -f` long
        // scout runs). When it ran, the file exists and is the source
        // of truth — we skip the post-hoc write to avoid clobbering
        // it. When it didn't run (e.g. an embedded test or non-CLI
        // consumer that called `run_chains_with_per_chain_params`
        // with `stage_dir = None`), fall back to the post-hoc write.
        //
        // Trade-off: the streaming version writes `NA` in the
        // `loglik` column (clean-PF re-eval is post-hoc). The
        // post-hoc fallback populates `loglik` for every 10th
        // iteration from `IF2IterResult.loglik`. Consumers that need
        // the clean-PF values per-iteration can join with
        // `clean_eval.tsv` / `fit_state.toml` on iteration index.
        let trace_path = format!("{}/parameter_traces.tsv", chain_dir);
        if !std::path::Path::new(&trace_path).exists() {
            let mut f = std::fs::File::create(&trace_path)
                .map_err(|e| format!("cannot write {}: {}", trace_path, e))?;
            writeln!(f, "# {}", crate::version::VERSION).unwrap();
            write!(f, "iteration\tloglik\tif2_perturbed_loglik").unwrap();
            for spec in if2_params { write!(f, "\t{}", spec.name).unwrap(); }
            writeln!(f).unwrap();
            for it in &result.iterations {
                let loglik_str = if it.loglik.is_finite() { format!("{:.2}", it.loglik) } else { "NA".into() };
                write!(f, "{}\t{}\t{:.2}", it.iteration, loglik_str, it.if2_perturbed_loglik).unwrap();
                for spec in if2_params { write!(f, "\t{}", it.param_means[spec.index]).unwrap(); } // gh#266
                writeln!(f).unwrap();
            }
        }

        // Resolve this chain's clean-eval score (if any). Falls back to
        // `result.mle` when no loglik_eval was run (PMMH path). Note: the
        // chain's θ̂ is the IF2 final-iteration mean either way; what
        // clean-eval changes is the *loglik* attached to that θ̂.
        let chain_score = loglik_eval.and_then(|ce|
            ce.per_chain.iter().find(|s| s.chain_id == *chain_id));

        // Final params TOML (all params, not just estimated).
        let toml_path = format!("{}/final_params.toml", chain_dir);
        let mut f = std::fs::File::create(&toml_path)
            .map_err(|e| format!("cannot write {}: {}", toml_path, e))?;
        writeln!(f, "# {}", crate::version::VERSION).unwrap();
        writeln!(f, "# Chain {} final parameters", chain_id + 1).unwrap();
        let header_ll = chain_score.map(|s| s.loglik).unwrap_or(result.final_loglik);
        if let Some(s) = chain_score {
            writeln!(f, "# loglik = {:.2} ± {:.2} (clean-eval re-score of IF2 final-iter mean)",
                header_ll, s.se).unwrap();
        } else {
            writeln!(f, "# loglik = {:.2}", header_ll).unwrap();
        }
        writeln!(f).unwrap();
        // Param key/value pairs at the top level so the file is loadable
        // via the standard params loader (`camdl pfilter --params …`,
        // `simulate --params`). Clean-eval metadata lives in a
        // `[provenance]` table at the bottom — keeping it within the
        // file but out of the flat-key namespace (the params loader
        // rejects non-numeric top-level keys; see GH #17).
        for name in all_param_names {
            let value = if let Some(spec) = if2_params.iter().find(|p| p.name == *name) {
                // Prefer clean-eval score's θ for estimated params.
                // (Equal to result.mle's per-spec entry under FinalIter
                // semantics, but kept routed through clean_eval for
                // consistency with the run-root final_params.toml.)
                chain_score
                    .map(|s| s.theta[spec.index])
                    .unwrap_or_else(|| result.mle[spec.index])
            } else if let Some(&idx) = compiled.param_index.get(name.as_str()) {
                base_params[idx]
            } else {
                0.0
            };
            writeln!(f, "{} = {}", name, format_param_value(value)).unwrap();
        }
        if let Some(s) = chain_score {
            writeln!(f).unwrap();
            writeln!(f, "[provenance]").unwrap();
            writeln!(f, "loglik = {:.6}", s.loglik).unwrap();
            writeln!(f, "se = {:.6}", s.se).unwrap();
            writeln!(f, "chain = {}", chain_id + 1).unwrap();
        }
    }
    Ok(())
}

/// Write `<dir>/chain_evaluations.tsv` — the per-chain clean-eval
/// score table. Schema:
/// `chain\tloglik\tse\tess_mean\tess_min\tess_min_step\tn_neg_inf_incr\t<param_1>\t<param_2>\t…`
/// with one header line + N data rows (one per chain) in chain-id
/// order. Each row reports the chain's IF2 final-iteration θ̂
/// re-scored with M high-particle PF replicates and combined via the
/// configured `combine` mode (logmeanexp by default).
///
/// `ess_min_step` is `-1` when no observations were scored (filter
/// failed); `n_neg_inf_incr` counts steps where the filter completely
/// lost the data.
///
/// Used downstream by `camdl fit summary`, the gate's per-chain SE
/// consumption, and book vignettes that report ESS-at-θ̂ diagnostics.
pub fn write_clean_eval_tsv(
    dir: &str,
    outcome: &loglik_eval::LoglikEvalOutcome,
    if2_params: &[EstimatedParam],
) -> Result<(), String> {
    use std::io::Write;
    let path = format!("{}/chain_evaluations.tsv", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;
    writeln!(f, "# {}", crate::version::VERSION).unwrap();
    write!(f, "chain\tloglik\tse\tess_mean\tess_min\tess_min_step\tn_neg_inf_incr").unwrap();
    for spec in if2_params { write!(f, "\t{}", spec.name).unwrap(); }
    writeln!(f).unwrap();
    for s in &outcome.per_chain {
        let ess_min_step_str = s.filter_stats.ess_min_step
            .map(|i| i.to_string()).unwrap_or_else(|| "-1".into());
        write!(f, "{}\t{:.6}\t{:.6}\t{:.2}\t{:.2}\t{}\t{}",
            s.chain_id + 1, s.loglik, s.se,
            s.filter_stats.ess_mean, s.filter_stats.ess_min,
            ess_min_step_str, s.filter_stats.n_neg_inf_increments).unwrap();
        for spec in if2_params {
            write!(f, "\t{}", s.theta[spec.index]).unwrap(); // round-trippable; gh#266
        }
        writeln!(f).unwrap();
    }
    Ok(())
}

/// Write `<dir>/final_params.toml` at the stage root, capturing the
/// overall clean-eval winner across all chains. Mirrors the per-chain
/// TOML schema but identifies which chain produced it.
pub fn write_run_root_final_params(
    dir: &str,
    outcome: &loglik_eval::LoglikEvalOutcome,
    if2_params: &[EstimatedParam],
    all_param_names: &[String],
    base_params: &[f64],
    compiled: &CompiledModel,
) -> Result<(), String> {
    use std::io::Write;
    let s = &outcome.per_chain[outcome.overall_winner_idx];
    let path = format!("{}/final_params.toml", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;
    writeln!(f, "# {}", crate::version::VERSION).unwrap();
    writeln!(f, "# winner: chain={}", s.chain_id + 1).unwrap();
    writeln!(f, "# loglik = {:.2} ± {:.2} (clean-eval re-score of IF2 final-iter mean)",
        s.loglik, s.se).unwrap();
    writeln!(f).unwrap();
    // Top-level keys are parameters only — keeps the file loadable via
    // the standard params loader. Clean-eval metadata lives in the
    // `[provenance]` table at the bottom. See GH #17.
    for name in all_param_names {
        let value = if let Some(spec) = if2_params.iter().find(|p| p.name == *name) {
            s.theta[spec.index]
        } else if let Some(&idx) = compiled.param_index.get(name.as_str()) {
            base_params[idx]
        } else {
            0.0
        };
        writeln!(f, "{} = {}", name, format_param_value(value)).unwrap();
    }
    writeln!(f).unwrap();
    writeln!(f, "[provenance]").unwrap();
    writeln!(f, "loglik = {:.6}", s.loglik).unwrap();
    writeln!(f, "se = {:.6}", s.se).unwrap();
    writeln!(f, "chain = {}", s.chain_id + 1).unwrap();
    Ok(())
}

/// Write `chain_starts.tsv` at the stage root — one row per chain
/// with the pre-filter starting values of every estimated parameter.
///
/// Diagnostic use: "did the random starts span the declared bounds?"
/// and "did all chains collapse to the same basin in one filter
/// pass?" — both questions that `parameter_traces.tsv` can't answer
/// because iteration 0 there is post-first-filter (already perturbed).
/// See the header in `chain_{N}/parameter_traces.tsv`.
///
/// `per_chain_params` is the same slice that `run_one_chain` receives:
/// `Some(&[Vec<EstimatedParam>])` when scout supplies per-chain random
/// starts, `None` when every chain starts from `config.estimated_params`.
pub fn write_chain_starts(
    dir: &str,
    per_chain_params: Option<&[Vec<EstimatedParam>]>,
    fallback: &[EstimatedParam],
    n_chains: usize,
) -> Result<(), String> {
    use std::io::Write;
    let path = format!("{}/chain_starts.tsv", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;
    writeln!(f, "# {}", crate::version::VERSION).unwrap();
    writeln!(f, "# pre-filter starting values per chain (before any IF2 perturbation).").unwrap();
    writeln!(f, "# pairs row-by-row with chain_{{chain}}/parameter_traces.tsv iter-0 rows").unwrap();
    writeln!(f, "# to visualise how far chains moved on the first filter pass.").unwrap();
    write!(f, "chain").unwrap();
    for spec in fallback { write!(f, "\t{}", spec.name).unwrap(); }
    writeln!(f).unwrap();

    for chain_id in 0..n_chains {
        let specs: &[EstimatedParam] = match per_chain_params {
            Some(pcp) => &pcp[chain_id],
            None      => fallback,
        };
        write!(f, "{}", chain_id + 1).unwrap();
        for spec in specs {
            write!(f, "\t{}", format_param_value(spec.initial)).unwrap();
        }
        writeln!(f).unwrap();
    }
    Ok(())
}

/// Format a parameter value with appropriate precision.
/// Shared by chain output and provenance output.
pub fn format_param_value(v: f64) -> String {
    if v.abs() < 1e-6 && v != 0.0 {
        format!("{:.8e}", v)
    } else if v == v.floor() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        let s = format!("{:.10}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Write diagnostics.tsv: per-iteration loglik for all chains.
pub fn write_diagnostics(dir: &str, results: &[(usize, IF2Result)]) -> Result<(), String> {
    use std::io::Write;
    let path = format!("{}/diagnostics.tsv", dir);
    let mut f = std::fs::File::create(&path)
        .map_err(|e| format!("cannot write {}: {}", path, e))?;
    writeln!(f, "# {}", crate::version::VERSION).unwrap();
    writeln!(f, "chain\titeration\tloglik\tif2_perturbed_loglik").unwrap();
    for (chain_id, result) in results {
        for it in &result.iterations {
            let loglik_str = if it.loglik.is_finite() { format!("{:.2}", it.loglik) } else { "NA".into() };
            writeln!(f, "{}\t{}\t{}\t{:.2}", chain_id + 1, it.iteration, loglik_str, it.if2_perturbed_loglik).unwrap();
        }
    }
    Ok(())
}

/// Collect ALL parameter values (estimated + fixed) for MLE output.
pub fn collect_all_params(
    mle: &[f64],
    if2_params: &[EstimatedParam],
    model: &ir::Model,
    base_params: &[f64],
    compiled: &CompiledModel,
) -> std::collections::BTreeMap<String, f64> {
    // gh#519: BTreeMap, not HashMap — this map is serialized into
    // `fit_state.toml` and the params TOMLs, whose byte layout must be a
    // function of their contents alone.
    let mut params = std::collections::BTreeMap::new();
    for p in &model.parameters {
        let idx = compiled.param_index.get(p.name.as_str()).copied().unwrap();
        let value = if let Some(spec) = if2_params.iter().find(|s| s.name == p.name) {
            mle[spec.index]
        } else {
            base_params[idx]
        };
        params.insert(p.name.clone(), value);
    }
    params
}

/// Resolve the prior for a parameter using the precedence chain:
///
///   1. fit.toml `[estimate.<name>.prior]` (typed `ir::PriorDist`;
///      override for sensitivity analysis)
///   2. model IR parameter.prior (from `~` syntax in .camdl)
///   3. Prior::Flat (improper uniform, default for inference)
///
/// Returns the prior and a string describing the source (for logging).
pub fn resolve_prior(
    name: &str,
    estimate: &indexmap::IndexMap<String, super::config_v2::EstimateSpecV2>,
    model: &ir::Model,
) -> (Prior, &'static str) {
    use super::config_v2::EstimatePriorSpec;
    // 1. fit.toml override — recognised in two shapes (gh#75):
    //    (a) regular distribution → `Prior::from_ir(pd)`, source = "fit.toml"
    //    (b) explicit flat opt-in → `Prior::Flat`, source = "flat (explicit)"
    //    The two flat shapes (b vs the tier-3 silent fallback below) are
    //    distinguished in provenance so reviewers can audit the user's
    //    intent — explicit-flat is accountable, silent-flat is a warning.
    if let Some(est) = estimate.get(name) {
        if let Some(ref spec) = est.prior {
            match spec {
                EstimatePriorSpec::Dist(pd) => {
                    return (Prior::from_ir(pd), "fit.toml");
                }
                EstimatePriorSpec::UniformOverBounds { .. } => {
                    // Uniform over the parameter's bounds (fit.toml `bounds`,
                    // else the model's `in [lo, hi]`). If neither supplies
                    // bounds, fall through to the default — the fit-path
                    // validator (validate_prior_transform_compat) reports the
                    // missing bounds precisely.
                    let resolved = est.bounds.or_else(|| model.parameters.iter()
                        .find(|p| p.name == name).and_then(|p| p.bounds()));
                    if let Some((lower, upper)) = resolved {
                        return (Prior::Fixed(Density::Uniform { lower, upper }), "fit.toml");
                    }
                }
                EstimatePriorSpec::Flat { .. } => {
                    return (Prior::Fixed(Density::Flat), "flat (explicit)");
                }
            }
        }
    }
    // 2. model IR
    if let Some(ir_param) = model.parameters.iter().find(|p| p.name == name) {
        if let Some(pd) = ir_param.prior_dist() {
            return (Prior::from_ir(pd), "model");
        }
        // Hierarchical priors carry expression-valued args; wrap them
        // verbatim — evaluation at each MCMC step resolves references
        // against current hyperparameter values. Wave 2 / #3 Gate 3a.
        if let Some(hp) = ir_param.hierarchical() {
            return (Prior::from_hierarchical_ir(hp), "model (hierarchical)");
        }
    }
    // 3. fallback. Reached only on the profile path; `camdl fit run`'s
    //    `validate_priors_present` rejects before we get here.
    (Prior::Fixed(Density::Flat), "flat (default)")
}

/// IC4 in the 2026-04-19 inference review batch 3: validate that
/// each estimated parameter's resolved prior is compatible with its
/// transform. Wrong combinations silently produce a different prior
/// than the user wrote (log_normal on Transform::None collapses to
/// Normal-on-natural; log_normal on Transform::Logit becomes
/// logit-normal; etc.).
///
/// Compatibility matrix:
///   Prior::TransformedNormal (log_normal) — Transform::Log
///   Prior::Beta                           — Transform::Logit
///   Prior::HalfNormal, Gamma, Exponential — Transform::Log
///   Prior::Uniform, Normal, Flat          — any transform
///
/// Call from every fit-stage entry point *before* building IF2
/// params so the user sees a clean error, not a miscalibrated
/// posterior.
pub fn validate_prior_transform_compat(
    estimate: &indexmap::IndexMap<String, super::config_v2::EstimateSpecV2>,
    model: &ir::Model,
) -> Result<(), String> {
    for name in estimate.keys() {
        // Build the same Transform the engine will use.
        let ir_param = match model.parameters.iter().find(|p| p.name == *name) {
            Some(p) => p,
            None => continue, // validate_partition catches unknown params.
        };
        // `prior = { uniform = {} }` is uniform over the parameter's bounds;
        // it needs bounds from *some* source. Catch the missing-bounds case
        // here (resolve_prior can only fall back to flat — it has no Result).
        if matches!(estimate.get(name).and_then(|e| e.prior.as_ref()),
                    Some(super::config_v2::EstimatePriorSpec::UniformOverBounds { .. }))
            && estimate.get(name).and_then(|e| e.bounds).is_none()
            && ir_param.bounds().is_none()
        {
            return Err(format!(
                "parameter '{}': prior = {{ uniform = {{}} }} is uniform over the \
                 parameter's bounds, but none are declared — add `in [lo, hi]` in the \
                 model or `bounds = [lo, hi]` to `[estimate.{}]`.", name, name));
        }

        let transform_override = estimate.get(name)
            .and_then(|e| e.transform.as_ref())
            .map(|t| t.as_str());
        let transform = derive_transform(ir_param, transform_override);
        let (prior, source) = resolve_prior(name, estimate, model);

        let is_log   = matches!(transform, Transform::Log { .. });
        let is_logit = matches!(transform, Transform::Logit { .. });

        let prior_name = prior.kind_str();
        let transform_name = match &transform {
            Transform::Log { .. }   => "Log",
            Transform::Logit { .. } => "Logit",
            Transform::None         => "None",
        };
        let support_desc = match prior.kind_str() {
            "log_normal" => "log_normal",
            "beta"       => "beta",
            _            => "positive-support",
        };
        let err = |needs: &str| Err(format!(
            "parameter '{}': prior {} is incompatible with transform {}; \
             {} priors require a {} transform. Either fix the param_kind \
             in the model (or the `transform` override in fit.toml), or \
             pick a different prior.\n  (prior source: {})",
            name, prior_name, transform_name, support_desc, needs, source,
        ));

        // Transform compatibility is a per-family rule, identical for fixed and
        // hierarchical priors — it lives once on `Density::transform_req`.
        match prior.transform_req() {
            // log_normal / half_normal / gamma / exponential / log_uniform.
            TransformReq::Log => {
                if !is_log { return err("Log"); }
            }
            TransformReq::Logit => {
                if !is_logit { return err("Logit"); }
                // Beta is on [0, 1]; require logit bounds span that.
                if let Transform::Logit { lo, hi } = transform {
                    if lo != 0.0 || hi != 1.0 {
                        return Err(format!(
                            "parameter '{}': beta prior requires bounds [0, 1], \
                             got [{}, {}].", name, lo, hi));
                    }
                }
            }
            // flat / uniform / normal / truncated_normal — any transform.
            TransformReq::Any => {}
        }
        // truncated_normal is a natural-scale Gaussian truncated to the
        // parameter's bounds. INVARIANT: the truncation support MUST equal the
        // parameter's resolved inference bounds, else the prior truncates to one
        // interval while the search box / transform clamp uses another (a silent
        // disagreement). Only fixed priors can be truncated_normal
        // (HierarchicalKind has no truncated_normal variant). The DSL bakes the
        // bounds equal from `in [lo,hi]`; this guards the fit.toml path (explicit
        // 4-field) and the case where fit.toml overrides `bounds` away from a
        // model-declared truncated_normal's support.
        if let Prior::Fixed(Density::TruncatedNormal { lower, upper, .. }) = &prior {
            let resolved = estimate.get(name)
                .and_then(|e| e.bounds)
                .or_else(|| ir_param.bounds());
            match resolved {
                None => return Err(format!(
                    "parameter '{}': truncated_normal prior requires bounds, but none \
                     are declared (model `in [lo, hi]` or fit.toml `bounds`).", name)),
                Some((lo, hi)) => {
                    if (lower - lo).abs() > 1e-9 || (upper - hi).abs() > 1e-9 {
                        return Err(format!(
                            "parameter '{}': truncated_normal truncation [{}, {}] must \
                             equal the parameter's inference bounds [{}, {}] — the \
                             prior's support and the search box must be the same \
                             interval. The model `~ truncated_normal(mean, sd)` form \
                             reads the bounds from `in [lo, hi]` automatically; in \
                             fit.toml set the prior's lower/upper to match `bounds`.",
                            name, lower, upper, lo, hi));
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── gh#513: the preflight table must report the realised starts ─────

    fn spec_at(name: &str, initial: f64) -> EstimatedParam {
        EstimatedParam {
            name: name.to_string(), index: 0, initial, rw_sd: 0.1,
            transform: sim::inference::types::Transform::None,
            lower: 0.0, upper: 1.0, rw_sd_auto: false, perturb_only_at_t0: false,
        }
    }

    /// With no per-chain override every chain runs from the config, so that is
    /// what the table should show — and nothing "differs".
    #[test]
    fn preflight_reports_the_config_when_there_is_no_per_chain_override() {
        let base = vec![spec_at("beta", 0.3)];
        let (shown, differ) = preflight_specs(&base, None);
        assert_eq!(shown[0].initial, 0.3);
        assert!(!differ);
        // An empty per-chain vector is the same case, not a panic.
        let empty: Vec<Vec<EstimatedParam>> = Vec::new();
        let (shown, differ) = preflight_specs(&base, Some(&empty));
        assert_eq!(shown[0].initial, 0.3);
        assert!(!differ);
    }

    /// The bug: the config said one thing and the chains did another. The
    /// table must follow the chains.
    #[test]
    fn preflight_reports_the_chain_start_not_the_config() {
        let base = vec![spec_at("beta", 0.3)];          // never run
        let chains = vec![vec![spec_at("beta", 0.9)]];  // what chain 1 uses
        let (shown, differ) = preflight_specs(&base, Some(&chains));
        assert_eq!(shown[0].initial, 0.9,
            "the table must report the realised chain start, not config \
             .estimated_params — reporting 0.3 here is gh#513");
        assert!(!differ, "one chain cannot differ from itself");
    }

    /// Multi-chain: report chain 1, and say so when the others start elsewhere
    /// (the caller renders that as a header note pointing at chain_starts.tsv).
    #[test]
    fn preflight_flags_when_chains_start_at_different_points() {
        let base = vec![spec_at("beta", 0.3)];
        let spread = vec![
            vec![spec_at("beta", 0.3)],
            vec![spec_at("beta", 0.7)],
        ];
        let (shown, differ) = preflight_specs(&base, Some(&spread));
        assert_eq!(shown[0].initial, 0.3, "chain 1 is the one reported");
        assert!(differ, "chain 2 starts elsewhere — the header must say so");

        // All chains identical (init = single with an override present):
        // no note, because there is nothing the single row hides.
        let same = vec![
            vec![spec_at("beta", 0.3)],
            vec![spec_at("beta", 0.3)],
        ];
        let (_, differ) = preflight_specs(&base, Some(&same));
        assert!(!differ);
    }

    /// Step 6 wiring regression: when in-run `IF2Result::final_loglik`
    /// disagrees with the clean-eval winner, `select_winner_summary`
    /// must follow clean-eval. The handoff calls this out as the
    /// canonical Step 6 test ("synthetic 2-chain run picks the
    /// higher-clean-ll chain even when the other has higher in-run
    /// final_loglik"). Since `run_chains_with_per_chain_params`
    /// requires a real PF, we test the post-IF2 selection helper
    /// (`select_winner_summary`) on a `LoglikEvalOutcome` constructed
    /// via `run_loglik_eval_with_scorer`. The synthetic IF2Results
    /// carry deliberately misleading `final_loglik` values; the
    /// helper must ignore them.
    #[test]
    fn winner_summary_follows_clean_eval_not_in_run_loglik() {
        use crate::fit::loglik_eval::run_loglik_eval_with_scorer;
        use crate::fit::config_v2::{LoglikEvalConfig, CombineMode};
        use sim::inference::if2::{IF2IterResult, IF2Result};

        // Two chains. Chain 0 has *higher* in-run final_loglik (the
        // misleading number); chain 1 has thetas the deterministic
        // scorer prefers. Clean-eval should pick chain 1.
        let mk_chain = |theta: f64, in_run_ll: f64| IF2Result {
            iterations: vec![IF2IterResult {
                iteration: 0,
                loglik: in_run_ll,
                if2_perturbed_loglik: in_run_ll,
                param_means: vec![theta],
                param_diag: vec![],
            }],
            mle: vec![theta],
            final_loglik: in_run_ll,
            last_loglik: in_run_ll,
        };
        let results = vec![
            (0usize, mk_chain(0.5,  -10.0)), // misleading: best in-run
            (1usize, mk_chain(50.0, -200.0)),
        ];

        let scorer = |theta: &[f64], _: usize, _: u64| {
            // Clean PF prefers theta around 50.
            let ll = if theta[0] < 10.0 { -100.0 } else { -50.0 };
            Ok((ll, crate::fit::loglik_eval::FilterStats::failed()))
        };
        let cfg = LoglikEvalConfig {
            n_particles: 1, n_replicates: 4, combine: CombineMode::LogMeanExp,
        };
        let outcome = run_loglik_eval_with_scorer(&results, &cfg, 0, scorer).unwrap();

        let (best_chain, best_ll, best_se) = select_winner_summary(&outcome);
        assert_eq!(best_chain, 1,
            "clean-eval must pick chain 1 despite chain 0's higher in-run loglik");
        assert!((best_ll - (-50.0)).abs() < 1e-12);
        assert!(best_se.abs() < 1e-12, "deterministic scorer → SE = 0");
    }

    /// Verify that write_chain_outputs writes correct values for BOTH
    /// estimated and fixed parameters. Regression test for bug where
    /// fixed params all got base_params[0] instead of their actual value.
    #[test]
    fn chain_output_fixed_params_correct() {
        use ir::{
            expr::{BinOpExpr, BinOpWrap, BinOp, Expr, ParamExpr, PopExpr},
            model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
            parameter::Parameter,
            transition::{Transition, StoichiometryEntry},
            Model,
        };

        // SIR model: beta (estimated), gamma (fixed), N0 (fixed)
        let model = Model {
            ic_grad: Default::default(),
            name: "test".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
                Compartment { name: "I".into(), kind: CompartmentKind::Integer },
                Compartment { name: "R".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![
                Transition {
                    rate_state_grad: Default::default(),
                    name: "infection".into(),
                    stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("I".into(), 1)],
                    rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                    }}),
                    metadata: None, draw_method: ir::transition::DrawMethod::Poisson,
                    rate_grad: Default::default(), lineage: None,
                },
                Transition {
                    rate_state_grad: Default::default(),
                    name: "recovery".into(),
                    stoichiometry: vec![StoichiometryEntry("I".into(), -1), StoichiometryEntry("R".into(), 1)],
                    rate: Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
                        op: BinOp::Mul,
                        left: Box::new(Expr::Param(ParamExpr { param: "gamma".into() })),
                        right: Box::new(Expr::Pop(PopExpr { pop: "I".into() })),
                    }}),
                    metadata: None, draw_method: ir::transition::DrawMethod::Poisson,
                    rate_grad: Default::default(), lineage: None,
                },
            ],
            ode_equations: vec![], time_functions: vec![], tables: vec![],
            interventions: vec![], observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![
                Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Estimated { init: Some(0.3), bounds: Some((0.01, 2.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None },
                Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Estimated { init: Some(0.1), bounds: Some((0.01, 1.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None },
                Parameter { name: "N0".into(), value: ir::parameter::ParamValue::Estimated { init: Some(1000.0), bounds: Some((100.0, 100000.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None },
            ],
            initial_conditions: InitialConditions::Explicit({
                let mut m = HashMap::new();
                m.insert("S".into(), 990.0);
                m.insert("I".into(), 10.0);
                m
            }),
            output: OutputConfig { times: OutputSchedule::AtTimes(vec![0.0, 80.0]), format: "tsv".into(), trajectory: true, observations: false },
            simulation: SimulationConfig { t_start: 0.0, t_end: 80.0, time_semantics: "continuous".into(), dt: Some(1.0), rng_seed: Some(42), integrator: Default::default() , t_end_anchor: None },
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };

        let compiled = CompiledModel::new(model).unwrap();
        let base_params = compiled.default_params.clone();

        // beta is estimated, gamma and N0 are fixed
        let if2_params = vec![EstimatedParam {
            name: "beta".into(),
            index: compiled.param_index["beta"],
            initial: 0.3,
            rw_sd: 0.05,
            transform: Transform::Log { lo: 0.01, hi: 2.0 },
            lower: 0.01,
            upper: 2.0,
            perturb_only_at_t0: false, rw_sd_auto: false,
        }];

        // Fake chain result: MLE has beta=0.5
        let mut mle = base_params.clone();
        mle[compiled.param_index["beta"]] = 0.5;

        let results = vec![(0_usize, IF2Result {
            iterations: vec![],
            mle,
            final_loglik: -100.0,
            last_loglik: -100.0,
        })];

        let dir = std::env::temp_dir().join("camdl_test_chain_output");
        let _ = std::fs::remove_dir_all(&dir);

        let param_names: Vec<String> = vec!["beta".into(), "gamma".into(), "N0".into()];
        write_chain_outputs(
            dir.to_str().unwrap(), &results, &if2_params,
            &param_names, &base_params, &compiled, None,
        ).unwrap();

        // Read back and verify
        let content = std::fs::read_to_string(dir.join("chain_1/final_params.toml")).unwrap();
        let parsed: HashMap<String, f64> = content.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut parts = l.splitn(2, '=');
                let k = parts.next()?.trim().to_string();
                let v: f64 = parts.next()?.trim().parse().ok()?;
                Some((k, v))
            })
            .collect();

        assert_eq!(parsed["beta"], 0.5, "estimated param should be MLE value");
        assert_eq!(parsed["gamma"], 0.1, "fixed param gamma should be 0.1, not base_params[0]");
        assert_eq!(parsed["N0"], 1000.0, "fixed param N0 should be 1000.0, not base_params[0]");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Clean-eval TSV emission. Schema is
    /// `chain\tloglik\tse\tess_mean\tess_min\tess_min_step\tn_neg_inf_incr\t<param...>`
    /// with one header line + N data rows (one per chain), in chain-id
    /// order. Verified for N=2. ESS columns reflect synthetic
    /// `FilterStats::failed()` (NaN ess_mean/min, -1 step).
    #[test]
    fn clean_eval_tsv_schema_and_rows() {
        use crate::fit::loglik_eval::{ChainScore, LoglikEvalOutcome, FilterStats};

        let outcome = LoglikEvalOutcome {
            per_chain: vec![
                ChainScore {
                    chain_id: 0,
                    theta: vec![0.10, 0.20],
                    loglik: -100.0, se: 0.5,
                    filter_stats: FilterStats::failed(),
                },
                ChainScore {
                    chain_id: 1,
                    theta: vec![0.30, 0.40],
                    loglik: -50.0, se: 0.4,
                    filter_stats: FilterStats::failed(),
                },
            ],
            overall_winner_idx: 1,
        };

        let mk_param = |name: &str, idx: usize| EstimatedParam {
            name: name.into(), index: idx, initial: 0.0,
            lower: 0.0, upper: 10.0, rw_sd: 0.1, rw_sd_auto: false,
            transform: Transform::None,
            perturb_only_at_t0: false,
        };
        let if2_params = vec![mk_param("beta", 0), mk_param("gamma", 1)];

        let dir = std::env::temp_dir().join("camdl_test_clean_eval_tsv");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        write_clean_eval_tsv(dir.to_str().unwrap(), &outcome, &if2_params).unwrap();

        let content = std::fs::read_to_string(dir.join("chain_evaluations.tsv")).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.starts_with('#')).collect();
        assert_eq!(lines.len(), 1 + 2, "1 header + 2 chain rows");
        assert_eq!(lines[0],
            "chain\tloglik\tse\tess_mean\tess_min\tess_min_step\tn_neg_inf_incr\tbeta\tgamma");
        assert!(lines[1].starts_with("1\t-100.000000\t0.500000"),
            "chain 1 prefix: {}", lines[1]);
        // Params use shortest round-trippable Display (gh#266), not fixed
        // `{:.6}` — so a small-magnitude param can't be truncated to 0.
        assert!(lines[1].ends_with("\t0.1\t0.2"),
            "chain 1 param suffix: {}", lines[1]);
        assert!(lines[2].starts_with("2\t-50.000000\t0.400000"),
            "chain 2 prefix: {}", lines[2]);
        assert!(lines[2].ends_with("\t0.3\t0.4"),
            "chain 2 param suffix: {}", lines[2]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Step 7: run-root `final_params.toml` carries the overall winner
    /// chain + candidate label and writes the winner's θ̂ for estimated
    /// params (here: chain 1's TailMean theta, NOT chain 0's MLE).
    #[test]
    fn run_root_final_params_uses_overall_winner() {
        use crate::fit::loglik_eval::{ChainScore, LoglikEvalOutcome, FilterStats};

        let outcome = LoglikEvalOutcome {
            per_chain: vec![
                ChainScore { chain_id: 0, theta: vec![0.10], loglik: -100.0, se: 0.3,
                    filter_stats: FilterStats::failed() },
                ChainScore { chain_id: 1, theta: vec![0.42], loglik: -50.0, se: 0.2,
                    filter_stats: FilterStats::failed() },
            ],
            overall_winner_idx: 1,
        };

        use ir::{
            model::{Compartment, CompartmentKind, InitialConditions, OutputConfig,
                    OutputSchedule, SimulationConfig},
            parameter::Parameter,
        };

        let if2_params = vec![EstimatedParam {
            name: "beta".into(), index: 0, initial: 0.0,
            lower: 0.0, upper: 10.0, rw_sd: 0.1, rw_sd_auto: false,
            transform: Transform::None, perturb_only_at_t0: false,
        }];

        // Minimal compiled stand-in. The writer only reads
        // `compiled.param_index` for *fixed* params; here every name in
        // `param_names` is in `if2_params`, so the lookup never fires.
        // Compartments are required because CompiledModel::new validates
        // them, but the simulation isn't run.
        let model = ir::Model {
            ic_grad: Default::default(),
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Estimated { init: Some(0.0), bounds: Some((0.0, 10.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None }],
            initial_conditions: InitialConditions::Explicit({
                let mut m = HashMap::new(); m.insert("S".into(), 100.0); m
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(42),
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        let compiled = CompiledModel::new(model).unwrap();

        let dir = std::env::temp_dir().join("camdl_test_run_root_final");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let param_names = vec!["beta".to_string()];
        write_run_root_final_params(
            dir.to_str().unwrap(), &outcome, &if2_params,
            &param_names, &[0.0], &compiled,
        ).unwrap();

        let content = std::fs::read_to_string(dir.join("final_params.toml")).unwrap();
        // Header records overall winner chain.
        assert!(content.contains("# winner: chain=2"),
            "header missing or wrong: {}", content);
        // Provenance moved under [provenance] table — top-level keys
        // are parameters only so the file is loadable via the standard
        // params loader (GH #17). The metadata is still present, just
        // under the right scope.
        assert!(content.contains("[provenance]"),
            "expected [provenance] table; got: {}", content);
        assert!(content.contains("chain = 2"));
        // The estimated-param value is the overall winner's θ (0.42),
        // NOT chain 0's 0.10.
        assert!(content.contains("beta = 0.42"),
            "expected beta = 0.42 (winner's θ); got: {}", content);

        // Schema invariant: top-level keys are parameters (numeric)
        // only — provenance metadata lives under [provenance] so the
        // standard params loader doesn't reject the file (GH #17).
        let parsed: toml::Value = toml::from_str(&content)
            .expect("final_params.toml must parse as TOML");
        let top = parsed.as_table().unwrap();
        for (k, v) in top {
            if k == "provenance" { continue; }
            assert!(v.as_float().is_some() || v.as_integer().is_some(),
                "top-level key `{}` is `{:?}`, must be numeric (param) — \
                 metadata belongs under [provenance]", k, v);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression for GH #17: `final_params.toml` (run-root) must be
    /// loadable by the standard params loader. The bug pre-fix emitted
    /// a string-typed metadata key at the top level, which
    /// `load_params_toml` rejected with `expected a number or table
    /// section, got String("…")`. The post-fix writer keeps all
    /// metadata under a `[provenance]` table, which the loader skips.
    /// (The original key, `winning_candidate_label`, was itself
    /// dropped in commit `20d48fe`'s clean-eval strip — the
    /// loadability invariant is what this test now guards.) This
    /// asserts loadability + correct parameter values, both of which
    /// are required for "rerun pfilter at the reported MLE"
    /// workflows to function.
    #[test]
    fn final_params_toml_is_loadable_by_params_loader() {
        use crate::fit::loglik_eval::{ChainScore, LoglikEvalOutcome, FilterStats};
        use ir::{
            model::{Compartment, CompartmentKind, InitialConditions, OutputConfig,
                    OutputSchedule, SimulationConfig},
            parameter::Parameter,
        };

        let outcome = LoglikEvalOutcome {
            per_chain: vec![
                ChainScore { chain_id: 5, theta: vec![87.668938],
                    loglik: -6235.11, se: 2.19,
                    filter_stats: FilterStats::failed() },
            ],
            overall_winner_idx: 0,
        };
        let if2_params = vec![EstimatedParam {
            name: "R0".into(), index: 0, initial: 0.0,
            lower: 1.0, upper: 200.0, rw_sd: 1.0, rw_sd_auto: false,
            transform: Transform::None, perturb_only_at_t0: false,
        }];
        let model = ir::Model {
            ic_grad: Default::default(),
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![
                Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter { name: "R0".into(), value: ir::parameter::ParamValue::Estimated { init: Some(0.0), bounds: Some((1.0, 200.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None }],
            initial_conditions: InitialConditions::Explicit({
                let mut m = HashMap::new(); m.insert("S".into(), 100.0); m
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(42),
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        let compiled = CompiledModel::new(model).unwrap();

        let dir = std::env::temp_dir().join("camdl_test_final_params_loadable");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("final_params.toml");

        write_run_root_final_params(
            dir.to_str().unwrap(), &outcome, &if2_params,
            &["R0".to_string()], &[0.0], &compiled,
        ).unwrap();

        // The actual contract: load_params_toml must return Ok and
        // surface R0 as the clean-eval winner's value.
        let loaded = crate::util::load_params_toml(path.to_str().unwrap())
            .expect("final_params.toml must be loadable via load_params_toml \
                     (GH #17). If this errored with `expected a number or \
                     table section, got String(...)`, a top-level string \
                     metadata key has leaked back into the writer.");
        let r0 = loaded.get("R0").copied()
            .expect("R0 must be present after load");
        assert!((r0 - 87.668938).abs() < 1e-6,
            "loaded R0 must equal clean-eval winner θ̂; got {}", r0);

        // Provenance keys are intentionally NOT in the parameter map
        // (the loader skips the [provenance] section).
        assert!(!loaded.contains_key("winning_candidate_label"));
        assert!(!loaded.contains_key("loglik"));
        assert!(!loaded.contains_key("se"));
        assert!(!loaded.contains_key("chain"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression for GH #16 (silent-wrong-answer): `winner_theta`
    /// returns the clean-eval θ̂ (= IF2 final-iter param_means under
    /// FinalIter-only semantics), NOT `IF2Result.mle` (= argmax over
    /// the IF2 chain's noisy `if2_perturbed_loglik`). These are
    /// distinct selection mechanisms even under FinalIter-only clean-
    /// eval: `IF2Result.mle` picks the iteration whose perturbed
    /// loglik happened to be highest, while clean-eval reports
    /// `iterations.last().param_means`. They generally agree when
    /// IF2 has converged but can diverge mid-cooling, and historically
    /// produced silent disagreement between `mle_params.toml` and
    /// `final_params.toml` (GH #16).
    #[test]
    fn winner_theta_picks_clean_eval_winner_not_if2_argmax() {
        use crate::fit::loglik_eval::{ChainScore, LoglikEvalOutcome, FilterStats};

        // IF2 results: .mle represents what pre-fix code would have
        // selected (the chain's IF2 argmax over perturbed loglik). The
        // clean-eval θ̂ for each chain is the chain's final-iter mean,
        // distinct from .mle when IF2's perturbed-loglik argmax landed
        // on a different iteration.
        let if2_chain0 = sim::inference::if2::IF2Result {
            mle: vec![0.10, 0.20],   // chain 0's IF2 argmax
            final_loglik: -100.0,
            last_loglik: -100.0,
            iterations: vec![],
        };
        let if2_chain1 = sim::inference::if2::IF2Result {
            mle: vec![0.30, 0.40],   // chain 1's IF2 argmax (pre-fix bug returned this)
            final_loglik: -50.0,
            last_loglik: -50.0,
            iterations: vec![],
        };

        // Clean-eval reports each chain's final-iter mean as θ̂. Chain
        // 1's clean-eval θ̂ ([0.31, 0.41]) differs from its IF2 .mle
        // ([0.30, 0.40]) — that divergence is what the test discriminates.
        let loglik_eval = LoglikEvalOutcome {
            per_chain: vec![
                ChainScore { chain_id: 0, theta: vec![0.10, 0.20],
                    loglik: -110.0, se: 0.5, filter_stats: FilterStats::failed() },
                ChainScore { chain_id: 1, theta: vec![0.31, 0.41],
                    loglik: -49.0,  se: 0.4, filter_stats: FilterStats::failed() },
            ],
            overall_winner_idx: 1,
        };

        let mut chain_agreement = HashMap::new();
        chain_agreement.insert("beta".to_string(),  1.05);
        chain_agreement.insert("gamma".to_string(), 1.06);
        let cr = ChainResults {
            results: vec![(0, if2_chain0), (1, if2_chain1)],
            best_chain: 1,
            best_loglik: -49.0,
            chain_agreement,
            loglik_eval,
        };

        let theta = cr.winner_theta();
        assert_eq!(theta, &[0.31, 0.41],
            "winner_theta must return clean-eval winner θ̂ \
             (= chain 1's final-iter mean [0.31, 0.41]), NOT chain 1's \
             IF2Result.mle ([0.30, 0.40]). If this fails, \
             mle_params.toml will diverge from final_params.toml — \
             the GH #16 silent-wrong-answer is back.");

        // Pre-fix path for reference: what `&best.mle` of best_chain returns.
        let best = &cr.results.iter().find(|(id, _)| *id == cr.best_chain).unwrap().1;
        assert_eq!(&best.mle, &vec![0.30, 0.40],
            "sanity: chain 1's IF2 mle is [0.30, 0.40] (different \
             from clean-eval winner [0.31, 0.41]) — this is what \
             makes the test discriminate.");
        assert_ne!(theta, best.mle.as_slice(),
            "winner_theta and best.mle must differ in this fixture, \
             else the test isn't catching the bug class");
    }

    /// resolve_prior precedence chain: fit.toml override → model IR → Flat.
    #[test]
    fn resolve_prior_precedence_chain() {
        use ir::parameter::{Parameter, PriorDist, LogNormalPrior, NormalPrior};
        use crate::fit::config_v2::EstimateSpecV2;
        use indexmap::IndexMap;

        let beta_with_ir_prior = Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Estimated { init: None, bounds: Some((0.01, 2.0)), prior: ir::parameter::PriorSpec::Dist(PriorDist::LogNormal(LogNormalPrior { mu: -1.0, sigma: 0.5 })), transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None };
        let gamma_no_prior = Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Estimated { init: None, bounds: Some((0.05, 1.0)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: None, param_dim: None };
        let model = ir::Model {
            ic_grad: Default::default(),
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![], transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], interventions: vec![], observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![beta_with_ir_prior, gamma_no_prior],
            initial_conditions: ir::model::InitialConditions::Explicit(HashMap::new()),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: None, rng_seed: None,
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };

        let est_with_normal = |name: &str, mean: f64, sd: f64| {
            let mut m: IndexMap<String, EstimateSpecV2> = IndexMap::new();
            m.insert(name.to_string(), EstimateSpecV2 {
                bounds: Some((0.01, 2.0)), transform: None,
                prior: Some(crate::fit::config_v2::EstimatePriorSpec::Dist(
                    PriorDist::Normal(NormalPrior { mean, sd }))),
                perturb_only_at_t0: false, rw_sd: None, start: None,
            });
            m
        };

        // (1) fit.toml override beats IR prior
        let estimate_override = est_with_normal("beta", 0.3, 0.1);
        let (p, src) = resolve_prior("beta", &estimate_override, &model);
        assert_eq!(src, "fit.toml", "fit.toml override should take precedence");
        match p {
            Prior::Fixed(Density::Normal { mean, sd }) => {
                assert!((mean - 0.3).abs() < 1e-9);
                assert!((sd - 0.1).abs() < 1e-9);
            }
            other => panic!("expected Normal from fit.toml, got {:?}", other),
        }

        // (2) IR prior used when fit.toml has no override
        let estimate_empty: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        let (p, src) = resolve_prior("beta", &estimate_empty, &model);
        assert_eq!(src, "model", "model IR prior should apply when fit.toml is silent");
        match p {
            Prior::Fixed(Density::TransformedNormal { mean, sd }) => {
                // LogNormal(mu=-1.0, sigma=0.5) in IR → TransformedNormal on log scale
                assert!((mean - (-1.0)).abs() < 1e-9);
                assert!((sd - 0.5).abs() < 1e-9);
            }
            other => panic!("expected TransformedNormal from IR LogNormal, got {:?}", other),
        }

        // (3) Flat fallback when neither fit.toml nor IR provide a prior
        let (p, src) = resolve_prior("gamma", &estimate_empty, &model);
        assert_eq!(src, "flat (default)");
        assert!(matches!(p, Prior::Fixed(Density::Flat)));
    }

    /// Minimal single-parameter model for transform-compat tests.
    #[cfg(test)]
    fn model_with_param(p: ir::parameter::Parameter) -> ir::Model {
        ir::Model {
            ic_grad: Default::default(),
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![], transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], interventions: vec![], observations: vec![],
            bindings: vec![], per_eval_bindings: vec![], parameters: vec![p],
            initial_conditions: ir::model::InitialConditions::Explicit(HashMap::new()),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: None, rng_seed: None, integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![], model_structure: None, balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        }
    }

    /// gh#155: `prior = { uniform = {} }` resolves to a concrete
    /// Uniform over the parameter's bounds (fit.toml `bounds`, else model
    /// `in [lo,hi]`), tagged as a fit.toml source; with no bounds anywhere
    /// the compat validator reports it precisely.
    #[test]
    fn uniform_over_bounds_resolves_to_uniform() {
        use ir::parameter::{Parameter, ParamValue, PriorSpec, ParamKind, Transform as IrTransform};
        use crate::fit::config_v2::{EstimateSpecV2, EstimatePriorSpec, UniformOverBoundsMarker};
        use indexmap::IndexMap;

        // Param with no model bounds — bounds come from fit.toml.
        let p = Parameter {
            name: "beta".into(),
            value: ParamValue::Estimated {
                init: None, bounds: None, prior: PriorSpec::Flat,
                transform: IrTransform::Identity },
            param_kind: Some(ParamKind::Rate), param_dim: None,
        };
        let model = model_with_param(p);
        let mut est: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        est.insert("beta".into(), EstimateSpecV2 {
            bounds: Some((0.05, 1.0)), transform: None,
            prior: Some(EstimatePriorSpec::UniformOverBounds {
                uniform: UniformOverBoundsMarker {} }),
            perturb_only_at_t0: false, rw_sd: None, start: None });

        let (prior, src) = resolve_prior("beta", &est, &model);
        match prior {
            Prior::Fixed(Density::Uniform { lower, upper }) => {
                assert_eq!(lower, 0.05);
                assert_eq!(upper, 1.0);
            }
            other => panic!("uniform = {{}} should resolve to Uniform over bounds, got {:?}", other),
        }
        assert_eq!(src, "fit.toml", "uniform-over-bounds is a fit.toml-sourced prior");
        // With bounds present the compat check passes.
        assert!(validate_prior_transform_compat(&est, &model).is_ok());

        // Strip the bounds → the validator must reject (nothing to be uniform over).
        est.get_mut("beta").unwrap().bounds = None;
        let err = validate_prior_transform_compat(&est, &model).unwrap_err();
        assert!(err.contains("uniform") && err.contains("bounds"),
            "missing-bounds uniform = {{}} must be rejected with a bounds hint: {}", err);
    }

    /// log_uniform is uniform on the log scale → it requires the Log transform,
    /// like log_normal. A non-Log transform would silently realize a different
    /// prior, so the compat validator must reject it. (gh#155)
    #[test]
    fn log_uniform_requires_log_transform() {
        use ir::parameter::{Parameter, ParamValue, PriorSpec, PriorDist, LogUniformPrior,
            ParamKind, Transform as IrTransform};
        use crate::fit::config_v2::EstimateSpecV2;
        use indexmap::IndexMap;

        let p = Parameter {
            name: "kappa".into(),
            value: ParamValue::Estimated {
                init: None, bounds: Some((1e-5, 1e-2)),
                prior: PriorSpec::Dist(PriorDist::LogUniform(
                    LogUniformPrior { lower: 1e-5, upper: 1e-2 })),
                transform: IrTransform::Identity,
            },
            param_kind: Some(ParamKind::Rate), param_dim: None,
        };
        let model = model_with_param(p);
        let mut est: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        est.insert("kappa".into(), EstimateSpecV2 {
            bounds: Some((1e-5, 1e-2)), transform: None, prior: None,
            perturb_only_at_t0: false, rw_sd: None, start: None });

        // rate param_kind → Log transform → ok.
        assert!(validate_prior_transform_compat(&est, &model).is_ok(),
            "log_uniform on a rate param (→Log) must validate");

        // Force a non-Log transform → must be rejected, naming the fix.
        // (EstimateSpecV2.transform is the fit-config Transform, distinct from
        // the IR's Transform on the parameter value.)
        est.get_mut("kappa").unwrap().transform =
            Some(crate::fit::config_v2::Transform::Identity);
        let err = validate_prior_transform_compat(&est, &model).unwrap_err();
        assert!(err.contains("log_uniform") && err.contains("Log"),
            "error must name log_uniform + the required Log transform: {}", err);
    }

    /// truncated_normal's truncation support must equal the parameter's
    /// inference bounds (the prior support and the search box are the same
    /// interval). The fit.toml 4-field form can disagree; the guard catches it.
    #[test]
    fn truncated_normal_bounds_must_match_inference_bounds() {
        use ir::parameter::{Parameter, ParamValue, PriorSpec, ParamKind, Transform as IrTransform};
        use crate::fit::config_v2::{EstimateSpecV2, EstimatePriorSpec};
        use ir::parameter::{PriorDist, TruncatedNormalPrior};
        use indexmap::IndexMap;

        let p = Parameter {
            name: "take".into(),
            value: ParamValue::Estimated {
                init: None, bounds: Some((0.3, 1.0)),
                prior: PriorSpec::Flat, transform: IrTransform::Identity },
            param_kind: Some(ParamKind::Probability), param_dim: None,
        };
        let model = model_with_param(p);
        let est_tn = |lo: f64, hi: f64, blo: f64, bhi: f64| {
            let mut m: IndexMap<String, EstimateSpecV2> = IndexMap::new();
            m.insert("take".into(), EstimateSpecV2 {
                bounds: Some((blo, bhi)), transform: None,
                prior: Some(EstimatePriorSpec::Dist(PriorDist::TruncatedNormal(
                    TruncatedNormalPrior { mean: 0.7, sd: 0.2, lower: lo, upper: hi }))),
                perturb_only_at_t0: false, rw_sd: None, start: None });
            m
        };

        // Truncation == bounds → ok.
        assert!(validate_prior_transform_compat(&est_tn(0.3, 1.0, 0.3, 1.0), &model).is_ok(),
            "matching truncation and bounds must validate");
        // Truncation ≠ bounds → rejected.
        let err = validate_prior_transform_compat(&est_tn(0.3, 0.9, 0.3, 1.0), &model).unwrap_err();
        assert!(err.contains("truncated_normal") && err.contains("must equal"),
            "error must explain the truncation/bounds disagreement: {}", err);
    }

    /// Cover every distribution supported in fit.toml `prior = ...` strings.
    /// Regression guard for the asymmetry bug where fit.toml could only override
    /// 4 of the 7 IR distributions.
    /// End-to-end: priors declared in a .camdl file survive compilation to
    /// Regression for the `init_mle = "scout"` bug: when a FitState
    /// (scout's output) is supplied to `FitRunConfig::build`, the
    /// resulting `base_params` must reflect the scout-best values —
    /// NOT the fit.toml `[estimate].*.start` values. The fix for this
    /// was reversing the application order in build. See
    /// docs/dev/incidents/2026-04-18-starts-from-scout-ignored.md.
    ///
    /// IF2 uses `config.base_params` as its starting point for the
    /// particle cloud (if2.rs:338, `current_params = base_params`).
    /// If the priority inversion lets est.start overwrite scout's
    /// best, refine starts from scratch instead of from scout's MLE.
    #[test]
    fn fit_state_overrides_config_start_in_base_params() {
        use crate::fit::state::FitState;
        use crate::fit::config_v2::FitConfigV2;

        // Tiny v2 fit.toml referencing the seir golden. We set
        // beta's `start = 0.1`; prior_state will supply 0.4. The
        // bug has `start` winning; the fix has `prior_state` winning.
        // Both values must sit within seir's declared beta bounds
        // [0.001, 0.5] so the post-resolution validator (gh#31) lets
        // the build succeed; the precedence test only needs the two
        // values to be distinguishable.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/seir_observations.ir.json", manifest);
        let data_dir = std::env::temp_dir().join(format!(
            "camdl_starts_from_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&data_dir).unwrap();
        let data_path = data_dir.join("obs.tsv");
        std::fs::write(&data_path,
            "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();

        // The v2 fit.toml. We use `start = 0.1` on beta and a [stages.scout]
        // section so the config validates; build() doesn't actually consume
        // the stage block (chains/particles come from its own args).
        let fit_toml_path = data_dir.join("fit.toml");
        let toml = format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[data.observations]
weekly_cases = "{}"

[estimate.beta]
bounds = [0.01, 0.5]
start  = 0.1

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[stages.scout]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 1
particles  = 100
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, data_dir.display(), ir_path, data_path.display());
        std::fs::write(&fit_toml_path, &toml).unwrap();
        let fit = FitConfigV2::load(&fit_toml_path.to_string_lossy())
            .expect("v2 fit.toml parse");

        // Scout produced a very different "best" — a clearly
        // distinguishable value so a win/loss is unambiguous.
        // Within [0.001, 0.5] but visibly far from est.start=0.1.
        let mut start_values = std::collections::BTreeMap::new();
        start_values.insert("beta".to_string(), 0.4);
        let prior_state = FitState {
            stage: "scout".into(), seed: 1,
            timestamp: "2026-04-18T00:00:00Z".into(),
            input_hash: None, camdl_version: None,
            best_loglik: -100.0, initial_loglik: f64::NEG_INFINITY,
            best_chain: 0, n_chains: 1, n_good_chains: Some(1),
            start_values,
            rw_sd: std::collections::BTreeMap::new(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            tail_chain_agreement: std::collections::BTreeMap::new(),
            perturb_only_at_t0_params: Vec::new(),
            chain_logliks: Vec::new(),
            chain_eval_logliks: Vec::new(),
            chain_eval_ses: Vec::new(),
            resolved_gate: None,
            resolved_loglik_eval: None,
            chain_init_source: None,
            dt_check: None,
        };

        let config = FitRunConfig::build(
            &fit, Some(&prior_state),
            1, 100, 1, 0.5, 50, 1, false,
        ).expect("build must succeed");

        let beta_idx = config.compiled.param_index.get("beta").copied()
            .expect("beta present");
        assert!((config.base_params[beta_idx] - 0.4).abs() < 1e-9,
            "prior_state must win over est.start — got {}, expected 0.4 \
             (scout's best). 0.1 means est.start overwrote scout — the \
             pre-fix bug is back.",
            config.base_params[beta_idx]);

        std::fs::remove_dir_all(&data_dir).ok();
    }

    // ── IC-free inference: config validation ────────────────────────────

    fn ic_free_fixture(dir: &std::path::Path, ic_free: bool, perturb_t0: bool)
        -> super::super::config_v2::FitConfigV2
    {
        // Minimal v2 fit.toml against the seir_observations golden IR.
        // Toggles ic_free and whether I0 is perturb_only_at_t0-flagged independently
        // so all four combinations can be built.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!(
            "{}/../../../ocaml/golden/seir_observations.ir.json", manifest);
        let data_path = dir.join("obs.tsv");
        std::fs::write(&data_path,
            "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
        let perturb_t0_line =
            if perturb_t0 { "perturb_only_at_t0 = true\n" } else { "" };
        let fit_toml_path = dir.join("fit.toml");
        let toml_src = format!(r#"
output_dir = "{}"
ic_free = {}

[model]
camdl = "{}"

[data.observations]
weekly_cases = "{}"

[estimate.I0]
bounds = [1, 1000]
start  = 5
{}
[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = 0.1

[stages.scout]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 1
particles  = 100
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, dir.display(), ic_free, ir_path, data_path.display(), perturb_t0_line);
        std::fs::write(&fit_toml_path, toml_src).unwrap();
        super::super::config_v2::FitConfigV2::load(
            &fit_toml_path.to_string_lossy())
            .expect("v2 fit.toml parse")
    }

    fn ic_free_test_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "camdl_icfree_{}_{}_{}", tag, std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// ic_free=true WITHOUT any perturb_only_at_t0 estimate → build errors with a
    /// helpful message naming the fix.
    #[test]
    fn ic_free_true_requires_perturb_only_at_t0() {
        let dir = ic_free_test_dir("requires_ivp");
        let fit = ic_free_fixture(&dir, true, false);
        let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
            Ok(_)  => panic!("ic_free=true + no perturb_only_at_t0 must error"),
            Err(e) => e,
        };
        assert!(err.contains("ic_free") && err.contains("perturb_only_at_t0"),
            "error must name both ic_free and perturb_only_at_t0: {}", err);
        assert!(err.contains("I0 = {") || err.contains("perturb_only_at_t0 = true"),
            "error should include a copy-pasteable example: {}", err);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ic_free=true WITH a perturb_only_at_t0 estimate → build succeeds and
    /// config.ic_free is propagated.
    #[test]
    fn ic_free_true_with_perturb_only_at_t0_succeeds() {
        let dir = ic_free_test_dir("with_ivp");
        let fit = ic_free_fixture(&dir, true, true);
        let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
            .expect("ic_free=true + perturb_only_at_t0 must build");
        assert!(config.ic_free, "FitRunConfig.ic_free must be true");
        // The SMCConfig view also carries the flag — that's what reaches
        // the PF / IF2 loop.
        assert!(config.smc_config().skip_first_obs_from_loglik,
            "smc_config() must thread ic_free into skip_first_obs_from_loglik");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ic_free=true WITH perturb_only_at_t0 but the FIRST observation is a hole (`NA`) →
    /// build errors: there is no y₁ to condition on, so ic_free would silently
    /// degenerate to no initial-state conditioning. The perturb_only_at_t0 precondition is
    /// satisfied here, so only the missing-y₁ guard can fire — isolating it.
    #[test]
    fn ic_free_with_missing_first_obs_is_rejected() {
        let dir = ic_free_test_dir("first_na");
        let fit = ic_free_fixture(&dir, true, true); // passes the perturb_only_at_t0 check
        // Overwrite the data so the FIRST observation (t=7) is a hole. build()
        // loads the data fresh, so it sees this holed series.
        std::fs::write(dir.join("obs.tsv"),
            "time\tweekly_cases\n7\tNA\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();
        let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
            Ok(_) => panic!("ic_free=true + a missing first observation must error"),
            Err(e) => e,
        };
        assert!(err.contains("nothing to condition on"),
            "error must name the missing-y₁ cause, not the perturb_only_at_t0 precondition: {err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// ic_free absent (default false) → build succeeds regardless of
    /// perturb_only_at_t0 presence, and the SMCConfig view reports ic_free=false.
    /// Regression guard: the new flag must default to OFF so no
    /// existing fit.toml silently changes behaviour.
    #[test]
    fn ic_free_default_off_does_not_require_perturb_only_at_t0() {
        let dir = ic_free_test_dir("default_off");
        let fit = ic_free_fixture(&dir, false, false);
        let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
            .expect("ic_free=false + no perturb_only_at_t0 must build");
        assert!(!config.ic_free);
        assert!(!config.smc_config().skip_first_obs_from_loglik);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The `perturb_only_at_t0` TOML key must reach `EstimatedParam` — the
    /// field IF2 reads to decide which parameters it skips at every
    /// observation (`if2.rs`, the inner perturbation loop). If the key stopped
    /// deserializing, or stopped being copied through `ParamSpec`, IF2 would
    /// silently perturb an initial-state parameter at every observation and
    /// the flag would be inert with no diagnostic.
    #[test]
    fn perturb_only_at_t0_key_reaches_the_estimated_param() {
        let dir = ic_free_test_dir("flag_reaches_param");
        let fit = ic_free_fixture(&dir, false, true);
        let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
            .expect("fixture must build");
        let i0 = config.estimated_params.iter()
            .find(|p| p.name == "I0")
            .expect("I0 is the fixture's only estimated parameter");
        assert!(i0.perturb_only_at_t0,
            "`perturb_only_at_t0 = true` in [estimate.I0] must set the flag on \
             the EstimatedParam IF2 reads");

        // Negative control on the same builder: without the key the flag is
        // false, so the assertion above cannot pass by the field defaulting to
        // true or by every parameter being flagged.
        let dir2 = ic_free_test_dir("flag_absent");
        let fit2 = ic_free_fixture(&dir2, false, false);
        let config2 = FitRunConfig::build(&fit2, None, 1, 100, 1, 0.5, 50, 1, false)
            .expect("fixture must build");
        assert!(!config2.estimated_params.iter().any(|p| p.perturb_only_at_t0),
            "no [estimate] entry declares the key, so no param may carry it");
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&dir2).ok();
    }

    /// End-to-end pin for the scout-convergence exemption: a parameter
    /// declared `perturb_only_at_t0` in fit.toml must land in
    /// `FitState::perturb_only_at_t0_params` and be exempted from the
    /// refine-stage Â check, even at an Â far above `a_thresh`.
    ///
    /// The unit test in `gating.rs` starts from a hand-written FitState, so it
    /// cannot see a break in the TOML → `EstimatedParam` → FitState chain.
    /// This one spans that chain: it derives the name list with the same
    /// expression `fit/mod.rs` uses when it writes the scout's fit_state.
    #[test]
    fn perturb_only_at_t0_param_is_exempt_from_the_scout_a_check() {
        use crate::fit::gating::{check_scout_convergence, ScoutGateVerdict};
        use crate::fit::state::FitState;

        let dir = ic_free_test_dir("scout_exempt");
        let fit = ic_free_fixture(&dir, false, true);
        let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
            .expect("fixture must build");

        let names: Vec<String> = config.estimated_params.iter()
            .filter(|p| p.perturb_only_at_t0)
            .map(|p| p.name.clone())
            .collect();
        assert_eq!(names, vec!["I0".to_string()],
            "the scout's fit_state must record I0 as perturb_only_at_t0");

        let mk_state = |t0_params: Vec<String>| FitState {
            stage: "scout".into(),
            seed: 1,
            timestamp: "2026-08-23T00:00:00Z".into(),
            input_hash: None,
            camdl_version: None,
            best_loglik: -60.2,
            initial_loglik: f64::NEG_INFINITY,
            best_chain: 0,
            n_chains: 2,
            n_good_chains: None,
            start_values: Default::default(),
            rw_sd: Default::default(),
            loglik_type: Some(crate::fit::loglik::LoglikType::If2),
            acceptance_rate: None,
            // I0's Â is wildly above any threshold; beta's is fine.
            tail_chain_agreement: [("beta".to_string(), 1.00),
                                   ("I0".to_string(), 16.5)]
                .into_iter().collect(),
            perturb_only_at_t0_params: t0_params,
            chain_logliks: vec![-60.2, -60.5],
            chain_eval_logliks: vec![],
            chain_eval_ses: vec![],
            resolved_gate: None,
            resolved_loglik_eval: None,
            chain_init_source: None,
            dt_check: None,
        };
        let gate = super::super::config_v2::GateConfig::default();
        match check_scout_convergence(&mk_state(names.clone()), &gate) {
            ScoutGateVerdict::Ok => {}
            other => panic!(
                "I0 is perturb_only_at_t0 and must be exempt from the Â check; \
                 got {other:?}"),
        }

        // Negative control: the SAME Â on a parameter that is NOT declared
        // perturb_only_at_t0 must fail, so the pass above is the exemption
        // doing work rather than the check being inert.
        assert!(matches!(check_scout_convergence(&mk_state(vec![]), &gate),
                         ScoutGateVerdict::Hard { .. }),
            "Â = 16.5 on a structural param must fail the scout check");

        std::fs::remove_dir_all(&dir).ok();
    }
    /// pipeline that pgas.rs / pmmh.rs use to build the Prior vector.
    ///
    /// This is the integration counterpart to resolve_prior_precedence_chain
    /// (which uses a hand-constructed ir::Model). Regression guard for any
    /// serde field rename or IR<->compiler drift.
    #[test]
    fn resolve_prior_end_to_end_from_golden_ir() {
        // sir_priors golden has: beta~LogNormal, gamma~HalfNormal,
        // rho~Beta, N0~LogNormal, I0~Exponential.
        use crate::fit::config_v2::EstimateSpecV2;
        use indexmap::IndexMap;
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = crate::util::load_model(&ir_path).expect("load golden");

        let empty: IndexMap<String, EstimateSpecV2> = IndexMap::new();

        // beta: LogNormal in IR → TransformedNormal at the Prior layer.
        let (p, src) = resolve_prior("beta", &empty, &model);
        assert_eq!(src, "model", "beta's IR prior should be picked up");
        match p {
            Prior::Fixed(Density::TransformedNormal { mean, sd }) => {
                assert!((mean - (-1.0)).abs() < 1e-9, "mean {}", mean);
                assert!((sd - 0.5).abs() < 1e-9, "sd {}", sd);
            }
            other => panic!("beta expected TransformedNormal, got {:?}", other),
        }

        // gamma: HalfNormal round-trip
        let (p, src) = resolve_prior("gamma", &empty, &model);
        assert_eq!(src, "model");
        assert!(matches!(p, Prior::Fixed(Density::HalfNormal { .. })), "gamma: {:?}", p);

        // rho: Beta round-trip
        let (p, src) = resolve_prior("rho", &empty, &model);
        assert_eq!(src, "model");
        match p {
            Prior::Fixed(Density::Beta { alpha, beta }) => {
                assert!((alpha - 2.0).abs() < 1e-9);
                assert!((beta - 5.0).abs() < 1e-9);
            }
            other => panic!("rho expected Beta, got {:?}", other),
        }

        // I0: Exponential round-trip
        let (p, src) = resolve_prior("I0", &empty, &model);
        assert_eq!(src, "model");
        assert!(matches!(p, Prior::Fixed(Density::Exponential { .. })), "I0: {:?}", p);
    }

    /// End-to-end: fit.toml [estimate] prior overrides the model IR prior.
    /// Same golden model, but fit.toml specifies a different distribution
    /// for beta — the override must win over what's in the .camdl.
    #[test]
    fn fit_toml_override_beats_golden_ir_prior() {
        use crate::fit::config_v2::EstimateSpecV2;
        use ir::parameter::{PriorDist, NormalPrior};
        use indexmap::IndexMap;
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = crate::util::load_model(&ir_path).expect("load golden");

        // Override beta with a much narrower normal prior; leave gamma alone.
        let mut estimate: IndexMap<String, EstimateSpecV2> = IndexMap::new();
        estimate.insert("beta".to_string(), EstimateSpecV2 {
            bounds: Some((0.01, 5.0)), transform: None,
            prior: Some(crate::fit::config_v2::EstimatePriorSpec::Dist(
                PriorDist::Normal(NormalPrior { mean: 0.25, sd: 0.05 }))),
            perturb_only_at_t0: false, rw_sd: None, start: None,
        });

        let (p, src) = resolve_prior("beta", &estimate, &model);
        assert_eq!(src, "fit.toml", "override should take precedence");
        match p {
            Prior::Fixed(Density::Normal { mean, sd }) => {
                assert_eq!(mean, 0.25); assert_eq!(sd, 0.05);
            }
            other => panic!("override should be Normal(0.25, 0.05), got {:?}", other),
        }

        // gamma is not overridden → still uses the IR's HalfNormal.
        let (p, src) = resolve_prior("gamma", &estimate, &model);
        assert_eq!(src, "model");
        assert!(matches!(p, Prior::Fixed(Density::HalfNormal { .. })));
    }

    /// Replaces the v1-era `parse_prior_covers_all_distributions` +
    /// `parse_prior_rejects_invalid_input` tests. fit.toml carries `prior`
    /// as `ir::PriorDist`; each variant must map onto the correct runtime
    /// `Prior` via `Prior::from_ir`.
    #[test]
    fn prior_dist_to_prior_maps_each_variant() {
        use ir::parameter::{
            PriorDist, LogNormalPrior, NormalPrior, BetaPrior, UniformPrior,
            HalfNormalPrior, GammaPrior, ExponentialPrior,
        };
        match Prior::from_ir(&PriorDist::LogNormal(LogNormalPrior { mu: 1.5, sigma: 0.4 })) {
            Prior::Fixed(Density::TransformedNormal { mean, sd }) => {
                assert_eq!(mean, 1.5); assert_eq!(sd, 0.4);
            }
            other => panic!("LogNormal: {:?}", other),
        }
        match Prior::from_ir(&PriorDist::Normal(NormalPrior { mean: 0.3, sd: 0.1 })) {
            Prior::Fixed(Density::Normal { mean, sd }) => {
                assert_eq!(mean, 0.3); assert_eq!(sd, 0.1);
            }
            other => panic!("Normal: {:?}", other),
        }
        match Prior::from_ir(&PriorDist::Beta(BetaPrior { alpha: 2.0, beta: 5.0 })) {
            Prior::Fixed(Density::Beta { alpha, beta }) => {
                assert_eq!(alpha, 2.0); assert_eq!(beta, 5.0);
            }
            other => panic!("Beta: {:?}", other),
        }
        // Uniform now carries explicit bounds (no silent reduction to Flat
        // on missing fields — that v2 behaviour is intentionally removed).
        match Prior::from_ir(&PriorDist::Uniform(UniformPrior { lower: -1.0, upper: 2.0 })) {
            Prior::Fixed(Density::Uniform { lower, upper }) => {
                assert_eq!(lower, -1.0); assert_eq!(upper, 2.0);
            }
            other => panic!("Uniform: {:?}", other),
        }
        match Prior::from_ir(&PriorDist::HalfNormal(HalfNormalPrior { sigma: 0.3 })) {
            Prior::Fixed(Density::HalfNormal { sigma }) => assert_eq!(sigma, 0.3),
            other => panic!("HalfNormal: {:?}", other),
        }
        match Prior::from_ir(&PriorDist::Gamma(GammaPrior { shape: 3.0, rate: 0.5 })) {
            Prior::Fixed(Density::Gamma { shape, rate }) => {
                assert_eq!(shape, 3.0); assert_eq!(rate, 0.5);
            }
            other => panic!("Gamma: {:?}", other),
        }
        match Prior::from_ir(&PriorDist::Exponential(ExponentialPrior { rate: 2.5 })) {
            Prior::Fixed(Density::Exponential { rate }) => assert_eq!(rate, 2.5),
            other => panic!("Exponential: {:?}", other),
        }
    }

    /// gh#34: when [estimate] entry omits `start =`, the run-config
    /// builder fills in a value automatically. The current rule is a
    /// Transform-aware uniform draw within bounds (log-uniform for
    /// Log-typed positive bounds, linear otherwise), seeded by
    /// (run-seed, param-name). Verifies (i) build succeeds with no
    /// explicit start and (ii) the assigned value is inside
    /// `[estimate].bounds`. No more "parameter 'foo' has no value"
    /// failure for forgetful users.
    #[test]
    fn estimate_without_start_falls_back_within_bounds() {
        use crate::fit::config_v2::FitConfigV2;

        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/seir_observations.ir.json", manifest);
        let data_dir = std::env::temp_dir().join(format!(
            "camdl_gh34_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&data_dir).unwrap();
        let data_path = data_dir.join("obs.tsv");
        std::fs::write(&data_path,
            "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n").unwrap();

        // beta has bounds [0.01, 0.5] and NO `start =`. The fallback
        // is a log-uniform draw within (0.01, 0.5) — beta is param_kind
        // = "rate" in the seir model, so derive_transform yields
        // Transform::Log and the helper picks the log branch. Pre-gh#34
        // this would fail with "parameter 'beta' has no value".
        let fit_toml_path = data_dir.join("fit.toml");
        let toml = format!(r#"
output_dir = "{}"

[model]
camdl = "{}"

[data.observations]
weekly_cases = "{}"

[estimate.beta]
bounds = [0.01, 0.5]

[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
I0       = 1

[stages.scout]
algorithm     = "if2"
backend     = "chain_binomial"
chains     = 1
particles  = 100
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, data_dir.display(), ir_path, data_path.display());
        std::fs::write(&fit_toml_path, &toml).unwrap();
        let fit = FitConfigV2::load(&fit_toml_path.to_string_lossy())
            .expect("v2 fit.toml parse");

        let config = FitRunConfig::build(
            &fit, None,
            1, 100, 1, 0.5, 50, 1, false,
        ).expect("build must succeed without explicit start (gh#34)");

        let beta_idx = config.compiled.param_index.get("beta").copied()
            .expect("beta present");
        let beta = config.base_params[beta_idx];
        assert!(beta > 0.01 && beta < 0.5,
            "missing-start fallback should draw within bounds (0.01, 0.5) \
             — got {}", beta);

        // Determinism: rebuilding at the same seed must give the same
        // base_params[beta] (Transform-aware uniform draw is hashed
        // from (seed, param_name) — re-runs are reproducible).
        let config2 = FitRunConfig::build(
            &fit, None,
            1, 100, 1, 0.5, 50, 1, false,
        ).expect("rebuild");
        assert_eq!(config2.base_params[beta_idx], beta,
            "fallback draw must be deterministic per (seed, name)");

        std::fs::remove_dir_all(&data_dir).ok();
    }

    // ── Fit-bounds plumbing (gh#42 follow-up) ────────────────────────
    //
    // `[estimate].bounds` in fit.toml must propagate to:
    //   - `EstimatedParam.{lower, upper}` — read by LHS / uniform init
    //   - `Transform::{Log, Logit}.{lo, hi}` — read by IF2 to clamp
    //     particles to the search box
    //
    // Without this, fit.toml bounds were advisory only: the search
    // proceeded over the model-declared bounds even when the user
    // tightened. LHS made the bug visible (init draws spanning model
    // bounds, not fit bounds).

    fn make_one_param_model(name: &str, lo: f64, hi: f64, kind: Option<ir::parameter::ParamKind>)
        -> (ir::Model, sim::CompiledModel)
    {
        use ir::{
            model::{Compartment, CompartmentKind, InitialConditions, OutputConfig,
                    OutputSchedule, SimulationConfig},
            parameter::Parameter,
        };
        let model = ir::Model {
            ic_grad: Default::default(),
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
            transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], interventions: vec![],
            observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![Parameter { name: name.into(), value: ir::parameter::ParamValue::Estimated { init: Some((lo + hi) * 0.5), bounds: Some((lo, hi)), prior: ir::parameter::PriorSpec::Flat, transform: ir::parameter::Transform::Identity }, param_kind: kind, param_dim: None }],
            initial_conditions: InitialConditions::Explicit({
                let mut m = HashMap::new(); m.insert("S".into(), 1.0); m
            }),
            output: OutputConfig {
                times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(42),
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        let compiled = sim::CompiledModel::new(model.clone()).expect("compile");
        (model, compiled)
    }

    #[test]
    fn fit_bounds_tighten_estimated_param_lower_upper() {
        let (model, compiled) = make_one_param_model("beta", 0.0, 1.0, Some(ir::parameter::ParamKind::Rate));
        let base_params = compiled.default_params.clone();
        let specs = vec![ParamSpec {
            name: "beta".into(),
            rw_sd: None,
            transform: None,
            perturb_only_at_t0: false,
            bounds: Some((0.1, 0.5)),  // tightened
        }];
        let result = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)
            .expect("tightened bounds within model bounds is valid");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].lower, 0.1, "EstimatedParam.lower must reflect fit bounds");
        assert_eq!(result[0].upper, 0.5, "EstimatedParam.upper must reflect fit bounds");
    }

    #[test]
    fn fit_bounds_propagate_to_log_transform_clamp() {
        // Transform clamp ranges drive IF2 particle clamping. If they
        // don't track fit bounds, the inference walks particles out to
        // model bounds even when the user tightened.
        let (model, compiled) = make_one_param_model("beta", 1e-5, 1.0, Some(ir::parameter::ParamKind::Rate));
        let base_params = compiled.default_params.clone();
        let specs = vec![ParamSpec {
            name: "beta".into(),
            rw_sd: None,
            transform: None,
            perturb_only_at_t0: false,
            bounds: Some((1e-3, 0.5)),
        }];
        let result = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)
            .expect("ok");
        match &result[0].transform {
            Transform::Log { lo, hi } => {
                assert_eq!(*lo, 1e-3, "Transform::Log.lo must reflect fit bounds");
                assert_eq!(*hi, 0.5,  "Transform::Log.hi must reflect fit bounds");
            }
            other => panic!("expected Log transform on rate-typed param, got {:?}", other),
        }
    }

    #[test]
    fn fit_bounds_outside_model_bounds_rejected() {
        // A fit must not loosen physical bounds the model declared.
        let (model, compiled) = make_one_param_model("beta", 0.0, 1.0, Some(ir::parameter::ParamKind::Rate));
        let base_params = compiled.default_params.clone();
        let specs = vec![ParamSpec {
            name: "beta".into(),
            rw_sd: None,
            transform: None,
            perturb_only_at_t0: false,
            bounds: Some((-0.5, 2.0)),  // wider than model — must reject
        }];
        let err = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)
            .expect_err("fit bounds outside model bounds must error");
        assert!(err.contains("outside model bounds"),
            "error must mention the bounds violation, got: {}", err);
    }

    #[test]
    fn fit_bounds_none_falls_back_to_model_bounds() {
        // Profile / pfilter pass bounds: None — they should keep using
        // the model-declared bounds verbatim.
        let (model, compiled) = make_one_param_model("beta", 0.01, 2.0, Some(ir::parameter::ParamKind::Rate));
        let base_params = compiled.default_params.clone();
        let specs = vec![ParamSpec {
            name: "beta".into(),
            rw_sd: None,
            transform: None,
            perturb_only_at_t0: false,
            bounds: None,
        }];
        let result = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)
            .expect("ok");
        assert_eq!(result[0].lower, 0.01);
        assert_eq!(result[0].upper, 2.0);
    }

    #[test]
    fn fit_bounds_propagate_to_rw_sd_auto_log_scale() {
        // auto_rw_sd_from_value works in transformed-scale units, then
        // converts to natural scale at the geometric midpoint. The
        // load-bearing thing is that the *log-scale* perturbation
        // magnitude shrinks with tighter bounds — IF2 perturbs in
        // transformed space, so that's what governs how many steps it
        // takes to traverse the search box. (Natural-scale rw_sd isn't
        // directly comparable across bound widths because the midpoint
        // changes with the bounds.)
        let (model, compiled) = make_one_param_model("beta", 1e-6, 1.0, Some(ir::parameter::ParamKind::Rate));
        let base_params = compiled.default_params.clone();
        let wide = vec![ParamSpec {
            name: "beta".into(), rw_sd: None, transform: None, perturb_only_at_t0: false,
            bounds: None,  // model bounds [1e-6, 1.0]
        }];
        let tight = vec![ParamSpec {
            name: "beta".into(), rw_sd: None, transform: None, perturb_only_at_t0: false,
            bounds: Some((0.1, 0.5)),
        }];
        let r_wide  = build_if2_params_from_specs(&model, &compiled, &base_params, &wide).unwrap();
        let r_tight = build_if2_params_from_specs(&model, &compiled, &base_params, &tight).unwrap();
        // Convert each rw_sd back to log-scale by dividing by midpoint.
        let mid_wide  = (1e-6_f64 * 1.0).sqrt();
        let mid_tight = (0.1_f64 * 0.5).sqrt();
        let log_sd_wide  = r_wide[0].rw_sd  / mid_wide;
        let log_sd_tight = r_tight[0].rw_sd / mid_tight;
        assert!(log_sd_tight < log_sd_wide,
            "tighter bounds must yield smaller log-scale rw_sd \
             (wide_log_sd={}, tight_log_sd={})", log_sd_wide, log_sd_tight);
    }

    // ── Bounded real/instant params honour fit bounds (gh#66) ────────
    //
    // `derive_transform_with_bounds` mapped `real` (the `_` fallback)
    // and `instant` to `Transform::None`, which applies NO clamping.
    // A bounded real/instant param could then random-walk arbitrarily
    // far outside its declared search bounds during IF2 (the issue
    // observed a seed-time param escaping to τ = −968). When finite
    // bounds are present, the scaled-logit (`Transform::Logit`) must
    // be used so IF2 perturbations stay in `[lo, hi]` — matching the
    // other bounded kinds and inference-spec §3.2 rule #2. With no
    // finite bounds, `Transform::None` is still correct.

    #[test]
    fn real_param_with_bounds_gets_bounded_transform_not_none() {
        // A `real` param WITH finite fit bounds must NOT get
        // Transform::None (which lets IF2 escape the box). It must get
        // the scaled-logit that clamps to the search bounds.
        let (model, compiled) = make_one_param_model("tau", 0.0, 55.0, Some(ir::parameter::ParamKind::Real));
        let base_params = compiled.default_params.clone();
        let specs = vec![ParamSpec {
            name: "tau".into(),
            rw_sd: None,
            transform: None,
            perturb_only_at_t0: false,
            bounds: Some((0.0, 55.0)),
        }];
        let result = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)
            .expect("ok");
        match &result[0].transform {
            Transform::Logit { lo, hi } => {
                assert_eq!(*lo, 0.0, "scaled-logit lo must track fit bounds");
                assert_eq!(*hi, 55.0, "scaled-logit hi must track fit bounds");
            }
            other => panic!(
                "bounded `real` must get a bounded (scaled-logit) transform, \
                 not {:?} — Transform::None lets IF2 escape the search box (gh#66)",
                other),
        }
    }

    #[test]
    fn instant_param_with_negative_bounds_gets_bounded_transform_not_none() {
        // `instant` is origin-relative and may be negative (a seed
        // before the anchor). The scaled-logit handles a negative lower
        // bound fine — it maps the whole [lo, hi] interval regardless of
        // sign — so a bounded `instant` must clamp, not escape.
        let (model, compiled) = make_one_param_model("tau", -30.0, 30.0, Some(ir::parameter::ParamKind::Instant));
        let base_params = compiled.default_params.clone();
        let specs = vec![ParamSpec {
            name: "tau".into(),
            rw_sd: None,
            transform: None,
            perturb_only_at_t0: false,
            bounds: Some((-30.0, 30.0)),
        }];
        let result = build_if2_params_from_specs(&model, &compiled, &base_params, &specs)
            .expect("ok");
        match &result[0].transform {
            Transform::Logit { lo, hi } => {
                assert_eq!(*lo, -30.0, "scaled-logit lo must track fit bounds (negative ok)");
                assert_eq!(*hi, 30.0, "scaled-logit hi must track fit bounds");
            }
            other => panic!(
                "bounded `instant` must get a bounded transform, not {:?} (gh#66)",
                other),
        }
    }

    #[test]
    fn unbounded_real_param_stays_none() {
        // No finite bounds → there is no box to clamp to, so an
        // unconstrained real remains Transform::None (no regression for
        // genuinely-unbounded params).
        use ir::parameter::Parameter;
        let real_param = Parameter { name: "tau".into(), value: ir::parameter::ParamValue::Fixed { value: 0.0 }, param_kind: Some(ir::parameter::ParamKind::Real), param_dim: None };
        let instant_param = Parameter {
            param_kind: Some(ir::parameter::ParamKind::Instant), ..real_param.clone()
        };
        // (0.0, INFINITY) is the resolved (lo, hi) when no bounds exist
        // (see build_if2_params_from_specs `(None, None) => (0.0, INF)`).
        let t_real = derive_transform_with_bounds(&real_param, None, (0.0, f64::INFINITY));
        let t_instant = derive_transform_with_bounds(&instant_param, None, (0.0, f64::INFINITY));
        assert!(matches!(t_real, Transform::None),
            "unbounded real must stay None, got {:?}", t_real);
        assert!(matches!(t_instant, Transform::None),
            "unbounded instant must stay None, got {:?}", t_instant);
    }

    // ── Cold-cooling Â suppression (gh#45) ───────────────────────────

    /// Build a synthetic IF2Result with `n_iters` iterations, where
    /// each iteration's `param_means[idx]` is `start + drift_per_iter ×
    /// iter` (deterministic chain trajectory). Used to construct
    /// degenerate-W (drift_per_iter ≈ 0) and non-degenerate-W
    /// (drift_per_iter > 0) test fixtures for `compute_chain_agreement`.
    fn synthetic_if2_result(
        n_params: usize,
        n_iters: usize,
        starts: &[f64],
        drifts: &[f64],
    ) -> sim::inference::if2::IF2Result {
        use sim::inference::if2::{IF2IterResult, IF2Result};
        assert_eq!(starts.len(), n_params);
        assert_eq!(drifts.len(), n_params);
        let iterations = (0..n_iters).map(|it| IF2IterResult {
            iteration: it,
            loglik: 0.0,
            if2_perturbed_loglik: 0.0,
            param_means: (0..n_params).map(|p|
                starts[p] + drifts[p] * (it as f64)).collect(),
            param_diag: vec![],
        }).collect();
        IF2Result {
            iterations,
            mle: starts.to_vec(),
            final_loglik: 0.0,
            last_loglik: 0.0,
        }
    }

    fn ep_simple(name: &str, idx: usize) -> EstimatedParam {
        EstimatedParam {
            name: name.into(), index: idx, initial: 1.0,
            rw_sd: 0.1, transform: Transform::None,
            lower: 0.0, upper: 10.0,
            perturb_only_at_t0: false, rw_sd_auto: false,
        }
    }

    #[test]
    fn chain_agreement_returns_finite_under_normal_within_chain_variance() {
        // Two chains with non-trivial drift across iterations → W is
        // meaningful → Â computed normally.
        let if2_params = vec![ep_simple("a", 0)];
        let chain_a = synthetic_if2_result(1, 50, &[1.00], &[0.01]);
        let chain_b = synthetic_if2_result(1, 50, &[1.05], &[0.01]);
        let results = vec![(0, chain_a), (1, chain_b)];
        let agreement = compute_chain_agreement(&results, &if2_params, 50);
        let r = agreement.get("a").copied().expect("entry present");
        assert!(r.is_finite(),
            "Â must be finite when within-chain variance is non-trivial; got {}", r);
    }

    #[test]
    fn chain_agreement_suppressed_under_cold_cooling_degenerate_w() {
        // Two chains that flatlined to constant tail values (zero
        // drift) → within-chain variance is exactly 0 → Â would
        // blow up → must return NaN.
        let if2_params = vec![ep_simple("a", 0)];
        let chain_a = synthetic_if2_result(1, 50, &[1.00], &[0.0]);
        let chain_b = synthetic_if2_result(1, 50, &[1.05], &[0.0]);
        let results = vec![(0, chain_a), (1, chain_b)];
        let agreement = compute_chain_agreement(&results, &if2_params, 50);
        let r = agreement.get("a").copied().expect("entry present");
        assert!(r.is_nan(),
            "Â must be NaN (suppressed) when within-chain variance ≈ 0; got {}", r);
    }

    #[test]
    fn chain_agreement_suppressed_below_relative_scale_threshold() {
        // Within-chain SD = 1e-9 of grand_mean — well below the
        // 1e-6 relative threshold → suppress. This is the He-measles-
        // refine regime where cooling has shrunk perturbations to
        // essentially numerical zero relative to parameter scale.
        let if2_params = vec![ep_simple("a", 0)];
        // grand_mean ≈ 1.025; within-chain SD ≈ 1e-9 (drift × √(n/12) ≈
        // 2e-11 × √(50/12) ≈ 4e-11 well below threshold).
        let chain_a = synthetic_if2_result(1, 50, &[1.000], &[2e-11]);
        let chain_b = synthetic_if2_result(1, 50, &[1.050], &[2e-11]);
        let results = vec![(0, chain_a), (1, chain_b)];
        let agreement = compute_chain_agreement(&results, &if2_params, 50);
        let r = agreement.get("a").copied().expect("entry present");
        assert!(r.is_nan(),
            "Â must be NaN when within-chain variance is below the 1e-6 \
             relative-scale threshold; got {}", r);
    }

    #[test]
    fn chain_agreement_per_param_independence_some_degenerate_some_not() {
        // Real-world case: some params flatlined under cooling, others
        // didn't. The non-degenerate ones still get a finite Â; the
        // degenerate ones return NaN.
        let if2_params = vec![ep_simple("flat", 0), ep_simple("active", 1)];
        // chain_a/b: param 0 flatlined (zero drift), param 1 still moving
        let chain_a = synthetic_if2_result(2, 50, &[1.0, 1.0], &[0.0, 0.01]);
        let chain_b = synthetic_if2_result(2, 50, &[1.05, 1.1], &[0.0, 0.01]);
        let results = vec![(0, chain_a), (1, chain_b)];
        let agreement = compute_chain_agreement(&results, &if2_params, 50);
        let r_flat   = agreement.get("flat").copied().expect("entry present");
        let r_active = agreement.get("active").copied().expect("entry present");
        assert!(r_flat.is_nan(),
            "flatlined-W param must yield NaN Â; got {}", r_flat);
        assert!(r_active.is_finite(),
            "moving-param Â must be finite; got {}", r_active);
    }

    /// gh#299 item 2: a parameter whose chains have NOT mixed must still
    /// report a finite ESS. `ESS = NaN` is uninformative for triage — a small
    /// finite number tells you *how* bad and sorts against the other
    /// parameters. Bulk-ESS is finite by construction here because it uses the
    /// between-chain variance rather than summing per-chain estimates, so
    /// there is nothing to suppress and no R-hat gate to fail.
    #[test]
    fn ess_bulk_is_finite_for_a_parameter_whose_chains_disagree() {
        // Two chains targeting very different modes (means 1.0 vs 5.0)
        // with small within-chain wobble. Between-chain separation
        // dwarfs within-chain variance → R-hat huge.
        let chain_a: Vec<f64> = (0..200)
            .map(|i| 1.0 + 0.05 * ((i as f64 * 0.7).sin()))
            .collect();
        let chain_b: Vec<f64> = (0..200)
            .map(|i| 5.0 + 0.05 * ((i as f64 * 0.7).sin()))
            .collect();
        let d = compute_rhat_ess(&[chain_a, chain_b]);
        let r = d.rank().expect("scored");
        assert!(r.rhat.is_finite() && r.rhat > 1.1,
            "fixture should produce R-hat > 1.1; got {}", r.rhat);
        assert!(r.ess_bulk.is_finite() && r.ess_bulk > 0.0,
            "a non-converged parameter must still report a finite bulk ESS; got {}",
            r.ess_bulk);
        assert!(r.ess_bulk < 20.0,
            "two chains stuck in separate modes are worth very few effective \
             draws; got {}", r.ess_bulk);
        assert!(r.ess_bulk_ratio() < 0.05,
            "ESS/N must expose how little of the run is usable; got {}",
            r.ess_bulk_ratio());
        assert_eq!(d.ess_per_chain().len(), 2,
            "ess_per_chain must be populated for each chain regardless of R-hat");
        assert!(d.ess_per_chain().iter().all(|&e| e.is_finite() && e > 0.0),
            "per-chain ESS values must be finite; got {:?}", d.ess_per_chain());
    }

    /// Well-mixed regression: chains drawn from the same distribution
    /// should have R-hat ~ 1 AND finite bulk / tail ESS AND finite
    /// per-chain values.
    #[test]
    fn rhat_ess_all_finite_for_well_mixed_chains() {
        // Linear congruential pseudo-random in [0,1) — deterministic
        // and uncorrelated enough that R-hat lands near 1.
        let lcg = |seed: u64, n: usize| -> Vec<f64> {
            let mut s = seed;
            (0..n).map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f64) / (u32::MAX as f64)
            }).collect()
        };
        let chain_a = lcg(1, 500);
        let chain_b = lcg(2, 500);
        let d = compute_rhat_ess(&[chain_a, chain_b]);
        let r = d.rank().expect("scored");
        assert!(r.rhat.is_finite() && r.rhat < 1.1,
            "well-mixed pseudo-iid chains should give R-hat < 1.1; got {}", r.rhat);
        assert!(r.ess_bulk.is_finite() && r.ess_bulk > 0.0,
            "bulk ESS must be finite for well-mixed chains; got {}", r.ess_bulk);
        assert!(r.ess_tail.is_finite() && r.ess_tail > 0.0,
            "tail ESS must be finite for well-mixed chains; got {}", r.ess_tail);
        assert!(r.ess_bulk_ratio() > 0.5,
            "near-independent draws should be worth most of their count; got {}",
            r.ess_bulk_ratio());
        assert!(d.refusal().is_none());
        assert_eq!(d.ess_per_chain().len(), 2);
        assert!(d.ess_per_chain().iter().all(|&e| e.is_finite() && e > 0.0));
    }

    /// Structural preconditions: >= 2 chains, equal length, >= 4 samples
    /// each. Below this there are NO statistics at all — the result is
    /// `NotScored` carrying the precondition that failed, so a caller cannot
    /// read a NaN as a measurement and the reader does not have to guess which
    /// precondition it was.
    #[test]
    fn rhat_ess_refuses_by_name_when_too_few_samples() {
        use sim::inference::convergence::ConvergenceError;
        let chain_a: Vec<f64> = vec![1.0, 1.1, 1.05];
        let chain_b: Vec<f64> = vec![1.0, 1.2, 1.10];
        let d = compute_rhat_ess(&[chain_a, chain_b]);
        assert!(d.rank().is_none(), "< 4 samples per chain → no statistics at all");
        assert!(d.ess_per_chain().is_empty(),
            "< 4 samples per chain → ess_per_chain empty; got {:?}", d.ess_per_chain());
        assert_eq!(d.refusal(), Some(&ConvergenceError::TooFewDraws { n_draws: 3 }));
    }

    #[test]
    fn rhat_ess_refuses_by_name_for_a_single_chain() {
        use sim::inference::convergence::ConvergenceError;
        // R-hat needs >= 2 chains.
        let chain: Vec<f64> = (0..200).map(|i| 1.0 + 0.01 * i as f64).collect();
        let d = compute_rhat_ess(&[chain]);
        assert!(d.rank().is_none() && d.ess_per_chain().is_empty(),
            "single chain → nothing scored");
        assert!(d.rhat_classic().is_nan(),
            "and the classic statistic is refused by the same input; got {}",
            d.rhat_classic());
        assert_eq!(d.refusal(), Some(&ConvergenceError::TooFewChains { n_chains: 1 }));
    }

    /// The ✓/~/✗ glyph in the end-of-stage block and the `RhatHigh` finding in
    /// `diagnostics.json` must agree about where the band is. They came from
    /// two places — the finding from the caller's `rhat_threshold`, the glyph
    /// from a literal `1.1` — which agreed only because the two numbers
    /// happened to be equal. Adopting Vehtari et al.'s 1.01 (the open decision
    /// on gh#84) would have made a parameter at 1.05 print green while drawing
    /// an error.
    #[test]
    fn the_report_glyph_and_the_finding_use_the_same_threshold() {
        use sim::inference::diagnostic::{DiagnosticCollector, DiagnosticKind};
        let chains = load_convergence_chains();
        // R̂ ≈ 1.027 — under the 1.1 bar, over a 1.01 one.
        let conv = StageConvergence::compute([("ar1_mixed".to_string(),
            chains["ar1_mixed"].clone())]);

        let collector = DiagnosticCollector::new("test");
        let lenient = conv.report(&collector, 1.1);
        assert!(lenient.contains("✓"), "1.027 is inside a 1.1 band: {lenient}");
        assert!(!collector.drain().iter().any(|d|
            matches!(d.kind, DiagnosticKind::RhatHigh { .. })),
            "and draws no finding there");

        let collector = DiagnosticCollector::new("test");
        let strict = conv.report(&collector, 1.01);
        assert!(!strict.contains("✓"),
            "the same R̂ must NOT print green under a 1.01 band: {strict}");
        assert!(collector.drain().iter().any(|d|
            matches!(d.kind, DiagnosticKind::RhatHigh { .. })),
            "and must draw the finding that goes with the glyph");
    }

    /// Four pseudo-iid chains, two of them offset by `offset`. R̂ rises
    /// smoothly with the offset; 0.09 lands at 1.0675, inside the disputed
    /// `[RHAT_CONVERGED_THRESHOLD, RHAT_REPORT_THRESHOLD)` gap.
    fn offset_chains(offset: f64) -> Vec<Vec<f64>> {
        let lcg = |seed: u64, n: usize, off: f64| -> Vec<f64> {
            let mut st = seed;
            (0..n)
                .map(|_| {
                    st = st
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    off + ((st >> 33) as f64) / (u32::MAX as f64)
                })
                .collect()
        };
        vec![
            lcg(1, 200, 0.0),
            lcg(2, 200, offset),
            lcg(3, 200, 0.0),
            lcg(4, 200, offset),
        ]
    }

    /// R̂ = `+∞` — every chain frozen at its own value, the 0%-acceptance
    /// deadlock — is a real answer and the worst one camdl can report. It must
    /// glyph as loudly as any other failure. A band that treated every
    /// non-finite R̂ as "no band" would render the most broken fit with the
    /// same neutral dash it uses for "not applicable", which is the softening
    /// direction and therefore the dangerous one.
    ///
    /// Three chains, each constant at its own value: `posterior` 1.7.0 returns
    /// `Inf` for this shape (the folded half is only constant at two chains).
    #[test]
    fn a_frozen_parameter_glyphs_as_the_worst_band_not_as_a_blank() {
        use sim::inference::diagnostic::DiagnosticCollector;
        use crate::fit::method_result::RhatBand;

        let frozen: Vec<Vec<f64>> = vec![
            vec![0.239349270; 30],
            vec![0.438170322; 30],
            vec![0.717000001; 30],
        ];
        let d = compute_rhat_ess(&frozen);
        let r = d.rank().expect("frozen chains are still scored");
        assert!(
            r.all_chains_frozen && r.rhat.is_infinite(),
            "fixture premise: R̂ must be +inf with the cause recorded; got {} (frozen={})",
            r.rhat, r.all_chains_frozen
        );
        assert_eq!(
            RhatBand::of(r.rhat),
            RhatBand::Severe,
            "an infinite R̂ is the worst band, not an absent one"
        );

        let conv = StageConvergence::compute([("beta".to_string(), frozen)]);
        let collector = DiagnosticCollector::new("test");
        let out = conv.report(&collector, RHAT_REPORT_THRESHOLD);
        assert!(
            out.contains('✗'),
            "and the stage block must say so loudly:\n{out}"
        );
    }

    /// One R̂, two commands, two glyphs.
    ///
    /// The end-of-stage block glyphed against the caller's `rhat_threshold`
    /// (`RHAT_REPORT_THRESHOLD` = 1.1, the bar that draws a `RhatHigh`
    /// FINDING); `fit summary` glyphs against `RHAT_CONVERGED_THRESHOLD`
    /// (1.05, the bar `converged_at` and the machine-readable `converged`
    /// column key on). A parameter in `[1.05, 1.1)` therefore printed ✓ when
    /// the fit finished and ✗ when the user ran `fit summary` on the same
    /// directory. Certification is one question; both surfaces must give it
    /// one answer.
    #[test]
    fn the_stage_report_glyphs_rhat_against_the_bar_camdl_certifies_against() {
        use sim::inference::diagnostic::DiagnosticCollector;
        use crate::fit::method_result::{RhatBand, RHAT_CONVERGED_THRESHOLD};

        let chains = offset_chains(0.09);
        let d = compute_rhat_ess(&chains);
        let rhat = d.rank().expect("scored").rhat;
        assert!(
            rhat > RHAT_CONVERGED_THRESHOLD && rhat < RHAT_REPORT_THRESHOLD,
            "fixture premise: R̂ must land in the disputed gap; got {rhat}"
        );

        let conv = StageConvergence::compute([("beta".to_string(), chains)]);
        let collector = DiagnosticCollector::new("test");
        let out = conv.report(&collector, RHAT_REPORT_THRESHOLD);

        assert!(
            !out.contains('✓'),
            "R̂ = {rhat:.4} is above the bar `fit summary` certifies against, \
             so the stage block must not print it green:\n{out}"
        );
        assert_eq!(
            RhatBand::of(rhat),
            RhatBand::NotConverged,
            "and the shared band agrees"
        );

        // The finding is a SEPARATE band and must not have moved: nothing is
        // drawn below `RHAT_REPORT_THRESHOLD`.
        assert!(
            !collector.drain().iter().any(|f| matches!(
                f.kind,
                sim::inference::diagnostic::DiagnosticKind::RhatHigh { .. }
            )),
            "the finding bar is 1.1 and this R̂ is below it — no finding"
        );

        // And the line must NAME the statistic and the threshold it applied,
        // so a reader is never left to infer which bar a glyph used.
        assert!(
            out.contains("1.05"),
            "the block must name the threshold it glyphed against:\n{out}"
        );
        assert!(
            out.contains("rank-normalized"),
            "and which statistic that is:\n{out}"
        );
    }

    /// Two numbers camdl computes on every fit, writes to `*_summary.json`,
    /// and never showed anyone.
    ///
    /// The classic Gelman & Rubin R̂ is the only one of the two estimators that
    /// carries SCALE: the rank-normalized one is bounded — ceiling ≈1.85 at two
    /// chains — and reads between 1.81 and 1.90 across thirteen orders of
    /// magnitude of within-chain movement, from chains frozen at
    /// floating-point resolution to chains that genuinely explore. It cannot
    /// distinguish "the sampler is dead" from "the sampler mixes badly"; the
    /// classic one separates those by fourteen orders of magnitude.
    ///
    /// The per-chain Geyer ESS answers the follow-up question no cross-chain
    /// statistic can: are the chains each mixing well *inside their own mode*,
    /// or is one stuck? Those have different fixes.
    ///
    /// `within_chain_drift` makes the first point on its own — `posterior`
    /// 1.7.0 gives classic 1.0008 where the headline is 1.4280, so the classic
    /// number reads inside every published healthy band on a fit that has not
    /// converged. Printing both is the whole reason to keep both.
    #[test]
    fn a_non_converged_parameter_shows_the_classic_rhat_and_the_per_chain_ess() {
        use sim::inference::diagnostic::DiagnosticCollector;
        let chains = load_convergence_chains();
        let conv = StageConvergence::compute([(
            "within_chain_drift".to_string(),
            chains["within_chain_drift"].clone(),
        )]);
        let collector = DiagnosticCollector::new("test");
        let out = conv.report(&collector, 1.05);

        assert!(
            out.contains("classic 1.001"),
            "the estimator that carries scale must be shown beside the bounded \
             one, not only written to disk:\n{out}"
        );
        assert!(
            out.contains("per-chain ESS ["),
            "and whether each chain mixes inside its own mode:\n{out}"
        );
        // Four chains in this fixture, so four cells.
        let line = out.lines().find(|l| l.contains("per-chain ESS")).expect("the line");
        assert_eq!(
            line.matches(',').count(),
            3,
            "one cell per chain, four chains: {line}"
        );
    }

    /// A high R̂ has two structurally different causes and the end-of-stage
    /// report must say WHICH: chains disagreeing about **location**
    /// (`rhat_bulk`) or about **spread** (`rhat_folded`). Both halves are
    /// computed by `rank_convergence`; carrying only their `max` outward left
    /// the reader a number with no action attached to it.
    ///
    /// `scale_disagree` is the spread case — four chains centred on the same
    /// value with different widths. `posterior` 1.7.0 on this fixture:
    /// `rhat` 1.3130, `rhat_split` 0.9984 (the raw-scale unfolded rung), so
    /// the folded half is what the headline is made of.

    #[test]
    fn the_report_names_which_half_drove_a_high_rhat() {
        use sim::inference::diagnostic::DiagnosticCollector;
        let chains = load_convergence_chains();
        let conv = StageConvergence::compute([(
            "scale_disagree".to_string(),
            chains["scale_disagree"].clone(),
        )]);
        let collector = DiagnosticCollector::new("test");
        let out = conv.report(&collector, 1.05);

        assert!(
            out.contains("folded"),
            "the report must name the folded half — it is the whole reason the \
             headline is above the band here:\n{out}"
        );
        assert!(
            out.contains("bulk 0.99") || out.contains("bulk 1.00"),
            "and must print the location half beside it, so the reader can see \
             the two are far apart:\n{out}"
        );
        assert!(
            out.contains("spread"),
            "and must say what that split MEANS, not only that it exists:\n{out}"
        );
    }

    /// The mirror case: `within_chain_drift` is driven by the LOCATION half
    /// (each chain drifts across its own run), so the same line must name the
    /// other cause. Without this a report that hard-coded "spread" would pass
    /// the test above.
    #[test]
    fn the_report_names_the_location_half_when_that_is_the_driver() {
        use sim::inference::diagnostic::DiagnosticCollector;
        let chains = load_convergence_chains();
        let conv = StageConvergence::compute([(
            "within_chain_drift".to_string(),
            chains["within_chain_drift"].clone(),
        )]);
        let collector = DiagnosticCollector::new("test");
        let out = conv.report(&collector, 1.05);
        assert!(
            out.contains("where the posterior sits"),
            "drifting chains disagree about WHERE the posterior sits:\n{out}"
        );
        assert!(
            out.contains("bulk half is larger"),
            "and it is the unfolded half that says so:\n{out}"
        );
        assert!(
            !out.contains("disagree on spread"),
            "and that is not a spread disagreement:\n{out}"
        );
    }

    /// The classic Gelman & Rubin statistic is still computed and still
    /// available — it is just no longer the headline. On the drifting-chain
    /// fixture the two differ by more than a third, in the direction that
    /// certifies a bad fit as good.
    #[test]
    fn classic_rhat_is_reported_alongside_but_is_not_the_headline() {
        let chains = load_convergence_chains();
        let reference = load_convergence_reference();
        let d = compute_rhat_ess(&chains["within_chain_drift"]);
        let want_classic = reference[&("within_chain_drift".into(), "rhat_classic".into())]
            .expect("fixture carries a classic R-hat");
        let classic = d.rhat_classic();
        assert!((classic - want_classic).abs() / want_classic < 1e-9,
            "classic R-hat = {}, posterior 1.7.0 rhat_basic(split = FALSE) = {}",
            classic, want_classic);
        assert!(classic < 1.05,
            "fixture premise: classic R-hat reads healthy here ({})", classic);
        assert!(d.rank().expect("scored").rhat > 1.4,
            "the headline must be the rank-normalized statistic ({})",
            d.rank().expect("scored").rhat);
    }

    // ── The external oracle for the rank-normalized statistics (gh#84) ──────
    //
    // Reference values come from the R package `posterior` 1.7.0 — the
    // implementation maintained by the authors of Vehtari, Gelman, Simpson,
    // Carpenter & Bürkner (2021), _Bayesian Analysis_ 16(2):667-718, and the
    // same algorithm Stan reports. Both the draws and the statistics are
    // committed under `rust/crates/sim/tests/fixtures/`, regenerated by
    // `scripts/gen_convergence_posterior_fixture.R`.
    //
    // The oracle has to be external. A camdl-side reimplementation compared
    // against camdl would agree with itself about any convention it got wrong
    // — the rank offset is `(r − 3/8) / (S − 2·3/8 + 1)`, and the `+ 1` is the
    // sort of detail that a same-author second implementation reproduces
    // faithfully and wrongly.

    /// Both fixture halves live beside the sim crate's other oracle TSVs.
    fn convergence_fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sim/tests/fixtures")
            .join(name)
    }

    /// `case → chains[chain][draw]`, from the committed draw fixture.
    fn load_convergence_chains() -> std::collections::BTreeMap<String, Vec<Vec<f64>>> {
        let path = convergence_fixture("convergence_chains.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
        let mut out: std::collections::BTreeMap<String, Vec<Vec<f64>>> = Default::default();
        for line in text.lines() {
            if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 4, "expected 4 columns, got {:?}", f);
            let chain: usize = f[1].parse().unwrap();
            let draw: usize = f[2].parse().unwrap();
            let value: f64 = f[3].parse().unwrap();
            let case = out.entry(f[0].to_string()).or_default();
            if case.len() <= chain {
                case.resize(chain + 1, Vec::new());
            }
            assert_eq!(case[chain].len(), draw, "draws must arrive in order");
            case[chain].push(value);
        }
        out
    }

    /// `(case, statistic) → value`, `None` where posterior reports `NA`.
    fn load_convergence_reference()
        -> std::collections::BTreeMap<(String, String), Option<f64>> {
        let path = convergence_fixture("convergence_posterior_ref.tsv");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
        let mut out = std::collections::BTreeMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 3, "expected 3 columns, got {:?}", f);
            let v = if f[2] == "NA" { None } else { Some(f[2].parse::<f64>().unwrap()) };
            out.insert((f[0].to_string(), f[1].to_string()), v);
        }
        out
    }

    /// The headline `rhat` this fit reports must be the rank-normalized
    /// split-R̂ of Vehtari et al. (2021), taken as
    /// `max(rank-normalized split-R̂, folded rank-normalized split-R̂)` — not
    /// the classic Gelman-Rubin (1992) statistic.
    ///
    /// The two disagree in the direction that matters. On the
    /// `within_chain_drift` case — chains whose MEANS agree while each drifts
    /// across its own run, the pattern measured on the ebola 8-chain PGAS fit
    /// in gh#84 — classic R̂ reads 1.0008 (inside any published healthy band)
    /// and the rank-normalized statistic reads 1.4280. Certifying that fit as
    /// converged is the silent-wrong-answer surface this test closes.
    #[test]
    fn rhat_matches_posterior_rank_normalized_reference() {
        let chains = load_convergence_chains();
        let reference = load_convergence_reference();
        assert!(!chains.is_empty(), "fixture must carry cases");
        let mut checked = 0;
        for (case, cs) in &chains {
            let want = reference
                .get(&(case.clone(), "rhat".to_string()))
                .unwrap_or_else(|| panic!("no reference rhat for case {case}"));
            let got = compute_rhat_ess(cs)
                .rank()
                .map_or(f64::NAN, |r| r.rhat);
            match want {
                // posterior declines to report (a constant draw set); camdl
                // must not invent a number either.
                None => assert!(!got.is_finite(),
                    "{case}: posterior reports NA, camdl must not report {got}"),
                Some(w) => {
                    assert!(got.is_finite(), "{case}: expected finite R̂ ≈ {w}, got {got}");
                    let rel = (got - w).abs() / w.abs();
                    assert!(rel < 1e-6,
                        "{case}: R̂ = {got}, posterior 1.7.0 = {w} (rel {rel:.3e})");
                    checked += 1;
                }
            }
        }
        assert!(checked >= 10, "expected ≥10 comparable cases, checked {checked}");
    }

    // ── ODE incidence across a HOLE: the bin reset must still fire ──────────
    //
    // `compute_ode_loglik` walks ODE snapshots, accumulates `cum_flows`, and
    // resets at each obs time. A HOLE keeps its time in the grid, so the reset
    // SHOULD fire across it (fixed-bin / pomp `accumvars` semantics): a missing
    // week still closes its weekly incidence bin — it must NOT merge two weeks
    // of incidence into the next observed bin. This mirrors the stochastic
    // `sparse_holes_reset.rs` test for the ODE-MLE loglik path.
    //
    // Probe: a DETERMINISTIC inflow `--> R @ K`, observed as `incidence` with a
    // Normal likelihood `mean = projected`, on a weekly grid 7/14/21/28 with a
    // HOLE at t=14. With K=10/day the per-week tally is 70. We probe the t=21
    // bin (the week AFTER the hole) by sweeping its datum and locating the
    // likelihood peak (the data value the model's projection most prefers):
    //   * reset fired at the hole  → projected@21 = 70  (one week)  → peak at 70
    //   * reset SKIPPED (merge)     → projected@21 = 140 (two weeks) → peak at 140
    // The Normal observation likelihood is the He-et-al-2010 *discretized* PMF
    // (a ±0.5 continuity correction, NOT the continuous PDF), so we assert the
    // peak LOCATION (70, not 140) rather than a closed-form gap. ll(70) beating
    // both ll(71) and ll(140) pins projected@21 = one week ⇒ the reset fired.
    #[test]
    fn ode_incidence_reset_fires_across_hole() {
        use ir::{
            expr::{ConstExpr, Expr, ProjectedExpr},
            model::{
                Compartment, CompartmentKind, InitialConditions, OutputConfig,
                OutputSchedule, RegularOutputSchedule, SimulationConfig,
            },
            observation::{
                Likelihood, ObservationModel as IrObs, ObservationSchedule,
                NormalLikelihood, Projection,
            },
            parameter::{ParamValue, Parameter},
            transition::{DrawMethod, StoichiometryEntry, Transition},
            Model,
        };
        use sim::inference::{
            BoundObs, MultiStreamObsModel, ObsCell,
            multi_stream_obs::{StreamProjection, StreamSpec},
        };

        let k = 10.0; // deterministic inflow per day → 70/week at the weekly grid
        let weekly = 7.0 * k; // 70
        let sd = 5.0; // tight: a 70-unit residual (the merge error) is 14 sd → huge

        // Daily output schedule so a snapshot lands on every obs time
        // (compute_ode_loglik requires a snapshot at each obs time).
        let m = Model {
            ic_grad: Default::default(),
            name: "ode_hole_reset".into(),
            version: "0.3".into(),
            time_unit: "days".into(),
            description: None,
            origin: None, origin_rata_die: None,
            compartments: vec![
                Compartment { name: "R".into(), kind: CompartmentKind::Integer },
            ],
            transitions: vec![
                Transition {
                    rate_state_grad: Default::default(),
                    name: "inflow".into(),
                    stoichiometry: vec![StoichiometryEntry("R".into(), 1)],
                    rate: Expr::Const(ConstExpr { value: k }),
                    metadata: None,
                    draw_method: DrawMethod::Deterministic,
                    rate_grad: Default::default(),
                    lineage: None,
                },
            ],
            ode_equations: vec![],
            time_functions: vec![],
            tables: vec![],
            interventions: vec![],
            observations: vec![
                IrObs {
                    name: "cases".into(),
                    source: "cases".into(),
                    columns: vec![
                        ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                        ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
                    ],
                    scored: "cases".into(),
                    emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
                    stratum: vec![],
                    projection: Projection::CumulativeFlow("inflow".into()),
                    projection_state_grad: Default::default(),
                    likelihood: Likelihood::Normal(NormalLikelihood {
                        mean: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
                        sd: ir::Diffable::new(Expr::Const(ConstExpr { value: sd })),
                    }),
                },
            ],
            bindings: vec![],
            per_eval_bindings: vec![],
            parameters: vec![
                Parameter { name: "dummy".into(), value: ParamValue::Fixed { value: 0.0 }, param_kind: None, param_dim: None },
            ],
            initial_conditions: InitialConditions::Explicit({
                let mut h = HashMap::new();
                h.insert("R".into(), 0.0); h
            }),
            output: OutputConfig {
                // Daily snapshots 0..=28 so 7/14/21/28 each get one.
                times: OutputSchedule::Regular(RegularOutputSchedule {
                    start: 0.0, step: 1.0,
                }),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: SimulationConfig {
                t_start: 0.0, t_end: 28.0, time_semantics: "continuous".into(),
                dt: Some(1.0), rng_seed: Some(1),
                integrator: Default::default(),
                t_end_anchor: None,
            },
            presets: vec![],
            model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
        };
        let compiled = Arc::new(CompiledModel::new(m).unwrap());
        let params = compiled.default_params.clone();
        let dt = 1.0;

        let times = vec![7.0, 14.0, 21.0, 28.0];
        let inflow_idx = compiled.model.transitions.iter()
            .position(|t| t.name == "inflow").unwrap();

        // Build an obs model with a hole at t=14 and the t=21 datum set to
        // `data21`. Returns the ODE loglik.
        let score = |data21: f64| -> f64 {
            let cells = vec![
                Some(ObsCell::Scalar(weekly)), // t=7
                None,                          // t=14 HOLE
                Some(ObsCell::Scalar(data21)), // t=21 (the probe)
                Some(ObsCell::Scalar(weekly)), // t=28
            ];
            let spec = StreamSpec::dense(
                StreamProjection::FlowSum(vec![inflow_idx]),
                compiled.model.observations[0].clone(),
                cells,
                times.clone(),
            );
            let obs_model = MultiStreamObsModel::new(
                BoundObs::bind(vec![spec]).unwrap().0, compiled.clone()).unwrap();
            sim::inference::compute_ode_loglik(&compiled, &obs_model, &times, dt, &params, dt).unwrap()
        };

        let ll_one_week  = score(weekly);        // data@21 = 70  (reset fired)
        let ll_near_one  = score(weekly + 1.0);  // data@21 = 71  (just off peak)
        let ll_two_weeks = score(2.0 * weekly);  // data@21 = 140 (merge)

        assert!(ll_one_week.is_finite() && ll_two_weeks.is_finite()
                && ll_near_one.is_finite(),
            "ODE loglik across a hole must be finite: one_week={ll_one_week}, \
             near={ll_near_one}, two_weeks={ll_two_weeks}");

        // Peak LOCATION: the t=21 datum the projection most prefers is the
        // projected value itself. If the reset fired at the hole, projected = 70
        // (one week) — so ll(70) is the maximum and beats BOTH a neighbour (71)
        // and the merge value (140). If the reset had been (wrongly) skipped,
        // projected = 140 and ll(140) would be the maximum instead.
        assert!(ll_one_week > ll_near_one,
            "ll(70) must beat ll(71): the t=21 likelihood peak sits at the \
             projected value; projected = one week = {weekly}. Got ll(70)={ll_one_week} \
             vs ll(71)={ll_near_one}.");
        assert!(ll_one_week > ll_two_weeks,
            "post-hole bin must tally ONE week of incidence ({weekly}), not two \
             ({}). ll(70) must beat ll(140) — got ll(70)={ll_one_week} vs \
             ll(140)={ll_two_weeks}. ll(140) ≥ ll(70) would mean the hole \
             suppressed the incidence-bin reset (merged two weeks into one bin).",
            2.0 * weekly);
        // The merge value is far in the tail of the one-week-centred density:
        // the gap must be large (not a numerical wobble). With sd = {sd} the
        // residual at 140 is 70 = 14·sd — many nats below the peak.
        assert!(ll_one_week - ll_two_weeks > 10.0,
            "the merge datum (140) must score MUCH worse than one week (70) — a \
             large gap is only possible if projected ≈ 70. Got gap = {}",
            ll_one_week - ll_two_weeks);
    }

    // ── gh#134: burn-in / conditioning window (`condition_from`) ─────────
    //
    // Two layers of tests:
    //   (1) `resolve_condition_from` — pure resolution + validation of the
    //       surface forms (absolute number, date, relative offset) and the
    //       window-bounds errors. No model load.
    //   (2) `FitRunConfig::build` end-to-end — the leading reset-only hole is
    //       prepended to the shared obs grid + every stream's cells (unset is
    //       bit-identical), and `condition_from` + `ic_free` errors loudly.

    mod condition_from_resolve {
        use crate::fit::runner::resolve_condition_from;

        // first_obs = 7, t_start = 0, unit = days, dt = 1. `resolve_condition_from`
        // now takes a single spec string (the per-stream selection happens at the
        // call site).

        #[test]
        fn absolute_number_interior_resolves_verbatim() {
            // c = 3 ∈ (0, 7) → Some(3.0).
            let c = resolve_condition_from("3", 7.0, 0.0, None, "days", 1.0).unwrap();
            assert_eq!(c, Some(3.0));
            // A non-integer numeric string parses too.
            let c = resolve_condition_from("3.0", 7.0, 0.0, None, "days", 1.0).unwrap();
            assert_eq!(c, Some(3.0));
        }

        #[test]
        fn relative_first_obs_minus_one_week() {
            // first_obs - 1 week = 7 - 7 = 0 = t_start → NO conditioning (None).
            let c = resolve_condition_from(
                "first_obs - 1 week", 7.0, 0.0, None, "days", 1.0).unwrap();
            assert_eq!(c, None, "first_obs - 1 week == t_start ⇒ no-op (None)");

            // first_obs - 4 days = 7 - 4 = 3 ∈ (0,7) → Some(3.0).
            let c = resolve_condition_from(
                "first_obs - 4 days", 7.0, 0.0, None, "days", 1.0).unwrap();
            assert_eq!(c, Some(3.0));
        }

        #[test]
        fn relative_unit_conversion_into_model_units() {
            // Model time_unit = weeks; first_obs = 5 (weeks); "first_obs - 7 days"
            // = 5 weeks − (7 days / 7 days-per-week) = 5 − 1 = 4 weeks.
            let c = resolve_condition_from(
                "first_obs - 7 days", 5.0, 0.0, None, "weeks", 1.0).unwrap();
            assert_eq!(c, Some(4.0));
        }

        #[test]
        fn absolute_date_resolves_via_origin() {
            // origin 2020-01-01, unit days. date("2020-01-04") → t = 3.
            let c = resolve_condition_from(
                "date(\"2020-01-04\")", 7.0, 0.0, Some("2020-01-01"), "days", 1.0).unwrap();
            assert_eq!(c, Some(3.0));

            // Bare ISO date is also accepted.
            let c2 = resolve_condition_from(
                "2020-01-04", 7.0, 0.0, Some("2020-01-01"), "days", 1.0).unwrap();
            assert_eq!(c2, Some(3.0));
        }

        #[test]
        fn date_without_origin_errors() {
            let err = resolve_condition_from(
                "2020-01-04", 7.0, 0.0, None, "days", 1.0).unwrap_err();
            assert!(err.contains("origin"), "must name the missing origin: {err}");
        }

        #[test]
        fn equal_to_t_start_is_noop() {
            let c = resolve_condition_from("0", 7.0, 0.0, None, "days", 1.0).unwrap();
            assert_eq!(c, None, "cond_from == t_start ⇒ no conditioning (None)");
        }

        #[test]
        fn before_t_start_errors() {
            let err = resolve_condition_from("-2", 7.0, 0.0, None, "days", 1.0).unwrap_err();
            assert!(err.contains("before the model start") || err.contains("t_start"),
                "must flag cond_from < t_start: {err}");
        }

        #[test]
        fn at_or_after_first_obs_errors() {
            // Exactly at first_obs.
            let err = resolve_condition_from("7", 7.0, 0.0, None, "days", 1.0).unwrap_err();
            assert!(err.contains("nothing to condition on") || err.contains("first observation"),
                "cond_from == first_obs must error: {err}");
            // After first_obs.
            let err2 = resolve_condition_from("9", 7.0, 0.0, None, "days", 1.0).unwrap_err();
            assert!(err2.contains("nothing to condition on") || err2.contains("first observation"),
                "cond_from > first_obs must error: {err2}");
        }

        #[test]
        fn off_grid_errors() {
            // 3.5 is not a multiple of dt = 1 relative to t_start = 0.
            let err = resolve_condition_from("3.5", 7.0, 0.0, None, "days", 1.0).unwrap_err();
            assert!(err.contains("grid"), "off-grid cond_from must error: {err}");
        }
    }

    /// `ConditionFrom` surface parsing + per-stream resolution (multi-cadence
    /// Phase 3): the `All("...")` form, the `[condition_from]` table with
    /// `default` + shadows, an unknown-label shadow (error), and a `default`-named
    /// stream (error).
    mod condition_from_parsing {
        use crate::fit::config_v2::ConditionFrom;

        /// Parse a top-level `condition_from` value out of a tiny TOML doc.
        fn parse(toml_src: &str) -> ConditionFrom {
            #[derive(serde::Deserialize)]
            struct Doc {
                condition_from: ConditionFrom,
            }
            let d: Doc = toml::from_str(toml_src).expect("must parse");
            d.condition_from
        }

        #[test]
        fn all_form_is_the_default_for_every_stream() {
            // A bare string deserializes to `All` and applies to every label.
            let c = parse("condition_from = \"first_obs - 1 week\"\n");
            assert_eq!(c, ConditionFrom::All("first_obs - 1 week".into()));
            assert_eq!(c.resolve_for("es"), Some("first_obs - 1 week"));
            assert_eq!(c.resolve_for("afp"), Some("first_obs - 1 week"));
            assert_eq!(c.resolve_for("anything"), Some("first_obs - 1 week"));
        }

        #[test]
        fn per_stream_default_and_shadows() {
            // A `[condition_from]` table deserializes to `PerStream`. `default` is
            // the all-streams default; other keys shadow individual streams.
            let c = parse(
                "[condition_from]\n\
                 default = \"first_obs - 1 week\"\n\
                 es      = \"first_obs - 2 weeks\"\n",
            );
            // `es` gets its shadow; an unshadowed stream falls to `default`.
            assert_eq!(c.resolve_for("es"), Some("first_obs - 2 weeks"));
            assert_eq!(c.resolve_for("afp"), Some("first_obs - 1 week"));
        }

        #[test]
        fn per_stream_without_default_resolves_to_none_for_unshadowed() {
            // No `default` → an unshadowed stream resolves to NO conditioning.
            let c = parse(
                "[condition_from]\n\
                 es = \"first_obs - 2 weeks\"\n",
            );
            assert_eq!(c.resolve_for("es"), Some("first_obs - 2 weeks"));
            assert_eq!(c.resolve_for("afp"), None,
                "no shadow + no default ⇒ no conditioning for that stream");
        }

        #[test]
        fn unknown_shadow_label_is_rejected() {
            // `ees` is a typo: not one of the valid labels {afp, es}.
            let c = parse(
                "[condition_from]\n\
                 ees = \"first_obs - 2 weeks\"\n",
            );
            let valid = vec!["afp".to_string(), "es".to_string()];
            let err = c.validate_labels(&valid).unwrap_err();
            assert!(err.contains("'ees'"), "must name the bad label: {err}");
            assert!(err.contains("afp") && err.contains("es"),
                "must list the valid labels: {err}");
        }

        #[test]
        fn stream_named_default_collides_with_reserved_key() {
            // A real stream labelled `default` is indistinguishable from the
            // reserved all-streams-default key → hard error.
            let c = parse("[condition_from]\nes = \"first_obs - 2 weeks\"\n");
            let valid = vec!["default".to_string(), "es".to_string()];
            let err = c.validate_labels(&valid).unwrap_err();
            assert!(err.contains("default") && err.contains("collides"),
                "must flag the reserved-key collision: {err}");
        }

        #[test]
        fn all_form_has_no_labels_to_validate() {
            // `All(_)` carries no per-stream keys, so validation always passes.
            let c = ConditionFrom::All("first_obs - 1 week".into());
            assert!(c.validate_labels(&["afp".to_string()]).is_ok());
            // And `default` as a stream label is fine under `All` (no table).
            assert!(c.validate_labels(&["default".to_string()]).is_ok());
        }
    }

    mod condition_from_build {
        use crate::fit::config_v2::FitConfigV2;
        use crate::fit::runner::FitRunConfig;

        /// Minimal v2 fit.toml against the seir_observations golden IR, with an
        /// optional top-level `condition_from` line and toggleable `ic_free`.
        /// seir: time_unit=days, t_start=0, weekly_cases at t=7,14,…; dt=1.
        fn fixture(
            dir: &std::path::Path,
            condition_from: Option<&str>,
            ic_free: bool,
            perturb_t0: bool,
        ) -> FitConfigV2 {
            // Default: weekly obs from t=7 — first window is one cadence, so the
            // W329 first-window guard never fires.
            fixture_with_obs(dir, condition_from, ic_free, perturb_t0,
                "time\tweekly_cases\n7\t1\n14\t2\n21\t3\n28\t4\n35\t5\n")
        }

        /// Like [`fixture`] but with caller-supplied observation TSV — lets a
        /// test set a wide leading gap (first_obs ≫ t_start) to exercise the
        /// W329 first-window guard (§6.8).
        fn fixture_with_obs(
            dir: &std::path::Path,
            condition_from: Option<&str>,
            ic_free: bool,
            perturb_t0: bool,
            obs_tsv: &str,
        ) -> FitConfigV2 {
            let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let ir_path = format!(
                "{}/../../../ocaml/golden/seir_observations.ir.json", manifest);
            let data_path = dir.join("obs.tsv");
            std::fs::write(&data_path, obs_tsv).unwrap();
            let cond_line = condition_from
                .map(|c| format!("condition_from = {c}\n"))
                .unwrap_or_default();
            let perturb_t0_line =
                if perturb_t0 { "perturb_only_at_t0 = true\n" } else { "" };
            let fit_toml_path = dir.join("fit.toml");
            let toml_src = format!(r#"
output_dir = "{}"
ic_free = {ic_free}
{cond_line}
[model]
camdl = "{ir_path}"

[data.observations]
weekly_cases = "{}"

[estimate.I0]
bounds = [1, 1000]
start  = 5
{perturb_t0_line}
[fixed]
sigma    = 0.25
gamma    = 0.3
rho      = 0.5
k        = 10.0
p_detect = 0.5
N0       = 1000
beta     = 0.1

[stages.scout]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 100
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#, dir.display(), data_path.display());
            std::fs::write(&fit_toml_path, toml_src).unwrap();
            FitConfigV2::load(&fit_toml_path.to_string_lossy()).expect("fit.toml parse")
        }

        fn test_dir(tag: &str) -> std::path::PathBuf {
            let d = std::env::temp_dir().join(format!(
                "camdl_condfrom_{}_{}_{}", tag, std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
            std::fs::create_dir_all(&d).unwrap();
            d
        }

        /// (a) A `condition_from` interior to (t_start, first_obs) prepends a
        /// LEADING reset-only hole to the shared obs grid: observations[0] is
        /// cond_from with a `None` cell in every stream, and the first REAL obs
        /// (t=7) shifts to index 1 — so the first scored bin is (cond_from, 7].
        #[test]
        fn interior_condition_from_inserts_leading_hole() {
            let dir = test_dir("insert");
            let fit = fixture(&dir, Some("\"3.0\""), false, false);
            let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("interior condition_from must build");

            // Canonical times: leading hole at 3.0, then the original grid.
            assert_eq!(config.observations[0].time, 3.0,
                "leading hole must be prepended at cond_from = 3.0");
            assert_eq!(config.observations[1].time, 7.0,
                "the first REAL observation must shift to index 1");
            assert_eq!(config.observations.len(), 6, "5 real obs + 1 leading hole");

            // Every stream: a None cell at index 0; the original first value at 1.
            for s in &config.streams {
                assert!(s.cells[0].is_none(),
                    "stream '{}' cell 0 must be a hole (None) at cond_from", s.name);
                assert!(s.cells[1].is_some(),
                    "stream '{}' cell 1 must be the first real observation", s.name);
                assert_eq!(s.data[0].time, 3.0, "stream data time 0 = cond_from");
            }
            std::fs::remove_dir_all(&dir).ok();
        }

        /// (b) `condition_from` UNSET: no grid change — observations start at the
        /// real first obs (t=7), no leading hole, no `None` at index 0. This is
        /// the bit-identical default.
        #[test]
        fn unset_condition_from_is_unchanged() {
            let dir = test_dir("unset");
            let fit = fixture(&dir, None, false, false);
            assert!(fit.condition_from.is_none(), "fixture without the key parses to None");
            let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("unset condition_from must build");
            assert_eq!(config.observations[0].time, 7.0,
                "unset condition_from must NOT insert a leading hole");
            assert_eq!(config.observations.len(), 5, "5 real obs, no hole");
            for s in &config.streams {
                assert!(s.cells[0].is_some(),
                    "unset: stream '{}' cell 0 must be the real first obs, not a hole", s.name);
            }
            std::fs::remove_dir_all(&dir).ok();
        }

        /// (b, cont.) The no-op boundary `condition_from == t_start` resolves to
        /// None and inserts nothing — bit-identical to unset.
        #[test]
        fn condition_from_at_t_start_inserts_nothing() {
            let dir = test_dir("at_tstart");
            let fit = fixture(&dir, Some("\"0.0\""), false, false);
            let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("condition_from == t_start must build (no-op)");
            assert_eq!(config.observations[0].time, 7.0,
                "condition_from == t_start must insert no hole");
            assert_eq!(config.observations.len(), 5);
            std::fs::remove_dir_all(&dir).ok();
        }

        /// (c) Relative form `"first_obs - 4 days"` resolves to cond_from = 3.0
        /// and inserts the same leading hole as the absolute form.
        #[test]
        fn relative_form_builds_and_inserts() {
            let dir = test_dir("relative");
            let fit = fixture(&dir, Some("\"first_obs - 4 days\""), false, false);
            let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("relative condition_from must build");
            assert_eq!(config.observations[0].time, 3.0,
                "first_obs(7) - 4 days = 3.0 leading hole");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// (d) Validation: cond_from before t_start errors at build.
        #[test]
        fn before_t_start_errors_at_build() {
            let dir = test_dir("before");
            let fit = fixture(&dir, Some("\"-2.0\""), false, false);
            let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
                Ok(_) => panic!("condition_from < t_start must error"),
                Err(e) => e,
            };
            assert!(err.contains("before the model start") || err.contains("t_start"),
                "must flag cond_from < t_start: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// (d) Validation: cond_from at/after the first obs errors at build.
        #[test]
        fn at_or_after_first_obs_errors_at_build() {
            let dir = test_dir("after");
            let fit = fixture(&dir, Some("\"7.0\""), false, false); // == first obs
            let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
                Ok(_) => panic!("condition_from >= first_obs must error"),
                Err(e) => e,
            };
            assert!(err.contains("nothing to condition on") || err.contains("first observation"),
                "must flag cond_from >= first_obs: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// (e) `condition_from` + `ic_free` together error loudly. The leading
        /// hole at obs-index 0 means y₁ is a hole in every stream, tripping the
        /// existing "nothing to condition on" ic_free guard — the desired
        /// orthogonal-mechanisms behaviour (no silent-wrong, no ic_free no-op).
        #[test]
        fn condition_from_with_ic_free_errors_loudly() {
            let dir = test_dir("with_icfree");
            // ic_free + perturb_only_at_t0 (so that precondition passes and only the
            // missing-y₁ guard can fire) + an interior condition_from.
            let fit = fixture(&dir, Some("\"3.0\""), true, true);
            let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
                Ok(_) => panic!("condition_from + ic_free must error"),
                Err(e) => e,
            };
            assert!(err.contains("nothing to condition on"),
                "condition_from + ic_free must trip the missing-y₁ guard: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// W329 escalation (§6.8): a wide leading gap before the first datum on
        /// an INCIDENCE stream (`weekly_cases` = `cumulative_flow infection`)
        /// with no `condition_from` is the gh#134 wrong-number — the first bin
        /// would accumulate the whole gap. The fit must be REJECTED, naming the
        /// fix.
        #[test]
        fn wide_incidence_gap_without_condition_from_is_rejected() {
            let dir = test_dir("widegap_reject");
            // first obs at t=70 vs t_start=0 → 70-day first window = 10× the
            // 7-day weekly cadence (> K=5). All obs ≤ t_end (365).
            let wide = "time\tweekly_cases\n70\t1\n77\t2\n84\t3\n91\t4\n98\t5\n";
            let fit = fixture_with_obs(&dir, None, false, false, wide);
            let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
                Ok(_) => panic!("wide incidence gap + no condition_from must error"),
                Err(e) => e,
            };
            assert!(err.contains("condition_from"),
                "the error must name condition_from as the fix: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// The same wide gap with `condition_from` set suppresses the guard — the
        /// modeler engaged with the boundary, and the leading hole makes the
        /// first scored bin one cadence. The fit builds.
        #[test]
        fn wide_incidence_gap_with_condition_from_builds() {
            let dir = test_dir("widegap_ok");
            let wide = "time\tweekly_cases\n70\t1\n77\t2\n84\t3\n91\t4\n98\t5\n";
            // 63 = 70 − 7 (one cadence before first_obs), interior to (0, 70).
            let fit = fixture_with_obs(&dir, Some("\"63.0\""), false, false, wide);
            FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("wide gap + condition_from must build");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// Test 7 (the headline): PER-STREAM conditioning on a TWO-stream model
    /// (multi-cadence Phase 3). The `surveillance_likelihoods` golden declares
    /// two incidence streams on distinct sources — `cases` and `deaths`. We bind
    /// `cases` as a LATE STARTER (first obs at t=308, vs t_start=0, a ~308-day
    /// first window against a 7-day weekly cadence) and `deaths` on the normal
    /// weekly cadence from t=7.
    ///
    /// The per-stream split is the point: `deaths` (window ≈ one cadence) needs
    /// NO conditioning and never errors, while `cases` (anomalously wide window)
    /// resolves to no conditioning by default and HARD-ERRORS, naming
    /// `condition_from.cases`. With `condition_from.cases` set, `cases`'s first
    /// scored bin shifts from the whole (0, 308] span to one cadence
    /// (308 − 7 = 301, 308].
    mod condition_from_per_stream {
        use crate::fit::config_v2::FitConfigV2;
        use crate::fit::runner::FitRunConfig;

        fn test_dir(tag: &str) -> std::path::PathBuf {
            let d = std::env::temp_dir().join(format!(
                "camdl_condfrom_ps_{}_{}_{}", tag, std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                    .unwrap().as_nanos()));
            std::fs::create_dir_all(&d).unwrap();
            d
        }

        /// Build a two-stream fit.toml against the `surveillance_likelihoods`
        /// golden IR, binding only `cases` (late-starting) + `deaths` (normal
        /// cadence) and leaving the `[condition_from]` body to the caller.
        /// `cond_block` is spliced verbatim (e.g. an empty string for "no
        /// conditioning", or `"[condition_from]\ncases = \"first_obs - 1 week\"\n"`).
        fn fixture(dir: &std::path::Path, cond_block: &str) -> FitConfigV2 {
            let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let ir_path = format!(
                "{}/../../../ocaml/golden/surveillance_likelihoods.ir.json", manifest);

            // `cases` LATE STARTER: 5 weekly obs from t=308 (< t_end ≈ 365). The
            // 308-day first window is ~44× the 7-day weekly cadence (> K=5) → the
            // W329 detector flags it for the incidence (cumulative_flow) `cases`
            // stream.
            let cases_path = dir.join("cases.tsv");
            std::fs::write(&cases_path,
                "time\tcases\n308\t1200\n315\t1300\n322\t1250\n329\t1100\n336\t1000\n").unwrap();
            // `deaths` NORMAL cadence: 5 weekly obs from t=7 (one cadence) → no
            // anomaly → never errors.
            let deaths_path = dir.join("deaths.tsv");
            std::fs::write(&deaths_path,
                "time\tdeaths\n7\t2\n14\t3\n21\t4\n28\t3\n35\t2\n").unwrap();

            let fit_toml_path = dir.join("fit.toml");
            let toml_src = format!(r#"
output_dir = "{out}"
{cond_block}
[model]
camdl = "{ir_path}"

[data.observations]
cases  = "{cases}"
deaths = "{deaths}"

[estimate.I0]
bounds = [1, 1000]
start  = 100

[fixed]
beta      = 0.45
sigma     = 0.2
gamma     = 0.1
mu_d      = 0.002
rho       = 0.2
sigma_rel = 0.3
kappa     = 40.0
phi       = 60.0
n_sero    = 1000
N0        = 1000000

[stages.scout]
algorithm  = "if2"
backend    = "chain_binomial"
chains     = 1
particles  = 100
iterations = 1
cooling    = 0.5

[config]
dt = 1.0
"#,
                out = dir.display(),
                ir_path = ir_path,
                cases = cases_path.display(),
                deaths = deaths_path.display());
            std::fs::write(&fit_toml_path, toml_src).unwrap();
            FitConfigV2::load(&fit_toml_path.to_string_lossy()).expect("fit.toml parse")
        }

        /// Pull a stream's resolved cells/times out of a built config by label
        /// (IR `source`).
        fn stream<'a>(config: &'a FitRunConfig, label: &str) -> &'a crate::fit::runner::ObsStream {
            config.streams.iter()
                .find(|s| s.obs_model_ir.source == label)
                .unwrap_or_else(|| panic!("stream '{label}' not bound"))
        }

        /// Test 7(a) — RED-side: the late-starting incidence stream `cases`,
        /// with NO `condition_from`, HARD-ERRORS naming `condition_from.cases`.
        /// The sibling `deaths` (normal cadence) does NOT need conditioning and
        /// is not the cause — proving the check is per-stream.
        #[test]
        fn late_starter_without_conditioning_hard_errors_naming_the_stream() {
            let dir = test_dir("err");
            let fit = fixture(&dir, ""); // no [condition_from]
            let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
                Ok(_) => panic!("late-starting incidence stream + no condition_from \
                                 must hard-error"),
                Err(e) => e,
            };
            // The W329 per-stream message: names the stream, the window vs
            // cadence, and the per-stream fix `condition_from.<label>`.
            assert!(err.contains("incidence stream 'cases'"),
                "must name the offending stream: {err}");
            assert!(err.contains("308-day first window"),
                "must state the wide first window: {err}");
            assert!(err.contains("~7-day cadence"),
                "must state the modal cadence: {err}");
            assert!(err.contains("condition_from.cases"),
                "must name the per-stream fix `condition_from.<label>`: {err}");
            // It must NOT blame `deaths` (the well-behaved sibling).
            assert!(!err.contains("'deaths'"),
                "the normal-cadence sibling must not be implicated: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// Test 7(b) — GREEN-side: with `condition_from.cases = "first_obs - 1
        /// week"` the fit BUILDS, and `cases`'s first scored bin is moved from
        /// the whole (0, 308] span to ONE cadence (301, 308]. We assert the
        /// mechanism directly: a leading reset-only HOLE (a `None` cell) is
        /// prepended to `cases` at t=301, the first REAL `cases` obs shifts to
        /// index 1 (t=308), the canonical union now carries t=301, and `deaths`
        /// is UNTOUCHED (no leading hole — it needed no conditioning).
        #[test]
        fn late_starter_with_conditioning_builds_and_first_bin_is_one_cadence() {
            let dir = test_dir("ok");
            let fit = fixture(&dir,
                "[condition_from]\ncases = \"first_obs - 1 week\"\n");
            let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("late starter WITH per-stream condition_from must build");

            // `cases`: leading reset-only hole at t = 308 − 7 = 301, then the
            // first real obs at 308. The first scored bin is (301, 308] — one
            // cadence — not the whole (0, 308] span.
            let cases = stream(&config, "cases");
            assert_eq!(cases.data[0].time, 301.0,
                "cases: leading hole must sit at first_obs(308) − 1 week = 301");
            assert!(cases.cells[0].is_none(),
                "cases: cell 0 must be a hole (None) — reset, no likelihood term");
            assert_eq!(cases.data[1].time, 308.0,
                "cases: the first REAL obs must shift to index 1");
            assert!(cases.cells[1].is_some(),
                "cases: cell 1 must be the first scored observation");

            // `deaths`: no conditioning resolved → NO leading hole; cell 0 stays
            // the real first obs at t=7. (Per-stream: the spec only named `cases`,
            // and there is no `default`.)
            let deaths = stream(&config, "deaths");
            assert_eq!(deaths.data[0].time, 7.0,
                "deaths: must be untouched (no leading hole)");
            assert!(deaths.cells[0].is_some(),
                "deaths: cell 0 must be the real first obs, not a hole");

            // The canonical union grid carries the inserted boundary t=301.
            assert!(config.observations.iter().any(|o| (o.time - 301.0).abs() < 1e-9),
                "the union grid must include the per-stream conditioning boundary 301");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// An unknown shadow label is rejected at build (typo-safety), naming the
        /// valid labels. `cases` is real; `caes` is a typo.
        #[test]
        fn unknown_shadow_label_rejected_at_build() {
            let dir = test_dir("typo");
            let fit = fixture(&dir,
                "[condition_from]\ncaes = \"first_obs - 1 week\"\n");
            let err = match FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false) {
                Ok(_) => panic!("an unknown shadow label must error at build"),
                Err(e) => e,
            };
            assert!(err.contains("'caes'") && err.contains("not an observation stream"),
                "must flag the typo'd label: {err}");
            assert!(err.contains("cases") && err.contains("deaths"),
                "must list the valid labels: {err}");
            std::fs::remove_dir_all(&dir).ok();
        }

        /// A `default` spec covers `cases` (the only stream that needs it), so a
        /// bare `[condition_from] default = ...` (no per-stream key) also clears
        /// the wide-window error — `default` is the all-streams default.
        #[test]
        fn default_key_conditions_the_late_starter() {
            let dir = test_dir("default");
            let fit = fixture(&dir,
                "[condition_from]\ndefault = \"first_obs - 1 week\"\n");
            let config = FitRunConfig::build(&fit, None, 1, 100, 1, 0.5, 50, 1, false)
                .expect("a `default` condition_from must cover the late starter");
            // `cases` gets the leading hole via `default`; `deaths`'s window is
            // one cadence so its resolved boundary (first_obs − 1 week = 0 =
            // t_start) is a no-op (no hole).
            let cases = stream(&config, "cases");
            assert_eq!(cases.data[0].time, 301.0, "cases conditioned via `default`");
            let deaths = stream(&config, "deaths");
            assert_eq!(deaths.data[0].time, 7.0,
                "deaths: first_obs(7) − 1 week = 0 = t_start ⇒ no-op, no hole");
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    // ── parse_time_spec / parse_condition_spec (gh#626) ─────────────────────

    #[test]
    fn time_spec_absolute_forms() {
        assert_eq!(parse_time_spec("--to", "120", None, "days").unwrap(),
                   TimeSpec::Absolute(120.0));
        // Dates need an origin; resolved via caltime.
        let got = parse_time_spec("--to", "date(\"2020-01-11\")",
                                  Some("2020-01-01"), "days").unwrap();
        assert_eq!(got, TimeSpec::Absolute(10.0));
        let bare = parse_time_spec("--to", "2020-01-11",
                                   Some("2020-01-01"), "days").unwrap();
        assert_eq!(bare, TimeSpec::Absolute(10.0));
        let err = parse_time_spec("--to", "2020-01-11", None, "days").unwrap_err();
        assert!(err.contains("origin"), "dates without origin must say so: {err}");
    }

    #[test]
    fn time_spec_anchored_forms() {
        assert_eq!(parse_time_spec("--to", "last_obs", None, "days").unwrap(),
                   TimeSpec::Anchored(ir::anchor::AnchoredTime { anchor: ObsAnchor::Last, offset: 0.0 }));
        assert_eq!(parse_time_spec("--to", "last_obs + 8 weeks", None, "days").unwrap(),
                   TimeSpec::Anchored(ir::anchor::AnchoredTime { anchor: ObsAnchor::Last, offset: 56.0 }));
        assert_eq!(parse_time_spec("--to", "first_obs - 1 week", None, "days").unwrap(),
                   TimeSpec::Anchored(ir::anchor::AnchoredTime { anchor: ObsAnchor::First, offset: -7.0 }));
        // Unit conversion into model units: weeks over a weekly model.
        assert_eq!(parse_time_spec("--to", "last_obs + 2 weeks", None, "weeks").unwrap(),
                   TimeSpec::Anchored(ir::anchor::AnchoredTime { anchor: ObsAnchor::Last, offset: 2.0 }));
    }

    #[test]
    fn time_spec_rejections_carry_hints() {
        // Commuted order: rejected, hint names the canonical spelling.
        let err = parse_time_spec("--to", "8 weeks + last_obs", None, "days").unwrap_err();
        assert!(err.contains("anchor first") && err.contains("last_obs + 8 weeks"),
            "commuted form must hint the canonical order: {err}");
        // DSL tick spelling: rejected, hint strips the tick.
        let err = parse_time_spec("--to", "last_obs + 8 'weeks", None, "days").unwrap_err();
        assert!(err.contains("tick") && err.contains("plain word"),
            "tick unit must hint the plain spelling: {err}");
        // Unknown unit, trailing tokens, negative N.
        assert!(parse_time_spec("--to", "last_obs + 8 fortnights", None, "days").is_err());
        assert!(parse_time_spec("--to", "last_obs + 8 weeks extra", None, "days").is_err());
        assert!(parse_time_spec("--to", "last_obs + -8 weeks", None, "days").is_err());
    }

    #[test]
    fn condition_spec_acceptance_set_unchanged() {
        // The wrapper keeps condition_from's restrictions: first_obs-only,
        // subtraction-only — with clearer messages than the old date
        // fall-through (gh#626).
        assert_eq!(
            parse_condition_spec("first_obs - 1 week", 42.0, None, "days").unwrap(),
            35.0);
        assert_eq!(parse_condition_spec("14", 42.0, None, "days").unwrap(), 14.0);
        let err = parse_condition_spec("last_obs - 1 week", 42.0, None, "days").unwrap_err();
        assert!(err.contains("first_obs") && err.contains("last_obs"),
            "last_obs in condition_from must be the anchored rejection, not a \
             date error: {err}");
        let err = parse_condition_spec("first_obs + 1 week", 42.0, None, "days").unwrap_err();
        assert!(err.contains("subtract"),
            "addition in condition_from must be rejected: {err}");
    }
}
