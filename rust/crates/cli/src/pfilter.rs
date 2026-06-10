//! `camdl pfilter` — bootstrap particle filter for log-likelihood estimation.
//!
//! Usage:
//!   camdl pfilter MODEL --params P.toml --data cases.tsv \
//!       --particles 5000 --dt 1.0 --seed 1
//!
//! Output: log-likelihood estimate to stdout.
//! With --trace: per-observation TSV (time, ll_increment, ESS).

use rayon::prelude::*;
use sim::{
    compiled_model::CompiledModel,
    inference::{
        bootstrap_filter,
        particle_filter::Observation,
        traits::SMCConfig,
        ChainBinomialProcess,
        MultiStreamObsModel,
        multi_stream_obs::StreamSpec,
    },
};
use std::collections::HashMap;

pub fn cmd_pfilter(a: &crate::args::PfilterArgs) {
    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();  // gh#audit-H5
    sim::eval_stats::set_allow_degenerate_rates(a.inference.allow_degenerate_rates);  // gh#audit-C6
    crate::args::apply_pf_wallclock_env(&a.inference);  // gh#133
    let ir_path = a.model.to_string_lossy().into_owned();
    let n_particles = a.inference.particles;
    let dt = a.inference.dt;
    let seed = a.inference.seed;
    let n_replicates = a.replicates;
    let trace_path: Option<String> = a.trace.clone();
    let output_path: Option<String> = a.output.as_ref().map(|p| p.to_string_lossy().into_owned());
    let save_final_state: Option<String> = a.save_final_state.as_ref().map(|p| p.to_string_lossy().into_owned());
    let save_filtering: Option<String> = a.save_filtering.as_ref().map(|p| p.to_string_lossy().into_owned());
    let save_paths: Option<(usize, String)> = a.save_paths.as_ref()
        .map(|p| (a.n_paths, p.to_string_lossy().into_owned()));
    let save_prequential: Option<String> = a.save_prequential.clone();
    let save_samples: bool = !a.no_save_samples;
    let scenario_name = a.scenario.scenario.clone();
    let adhoc_enable = a.scenario.enable.clone();
    let adhoc_disable = a.scenario.disable.clone();
    let obs_name = a.flow.obs.clone();
    let flow_name = a.flow.flow.clone();

    // Load model (supports .camdl via camdlc)
    let (model_in, _model_json) = crate::util::load_model(&ir_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // Unified resolver (2026-05-25 CLI UX rev 2). Maps the legacy
    // surface (`--params FILE` (repeatable) + `--param NAME=VALUE` +
    // scenario/enable/disable) onto the resolver inputs. pfilter is
    // non-inference for the value-resolution perspective; the
    // [estimate] set is empty and no kick-out warnings can fire.
    use crate::params_resolver::{ParameterInputs, resolve_parameters, print_warnings};
    use indexmap::{IndexMap, IndexSet};

    let fixed_cli: Vec<(String, f64)> = a.model_overrides.param.iter()
        .map(|p| (p.name.clone(), p.value)).collect();
    let fixed_files: Vec<std::path::PathBuf> = a.model_overrides.params.clone();
    let table_files: HashMap<String, std::path::PathBuf> = a.model_overrides.table.iter()
        .map(|t| (t.name.clone(), t.path.clone())).collect();
    let ftf: IndexMap<String, f64> = IndexMap::new();
    let fte: IndexSet<String> = IndexSet::new();

    let resolved = resolve_parameters(ParameterInputs {
        model: &model_in,
        scenario: scenario_name.as_deref(),
        adhoc_enable: &adhoc_enable,
        adhoc_disable: &adhoc_disable,
        fixed_cli: &fixed_cli,
        fixed_files: &fixed_files,
        fit_toml_fixed: &ftf,
        fit_toml_estimate: &fte,
        table_files: &table_files,
    }).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    print_warnings(&resolved);
    let model = resolved.model.clone();

    let compiled = CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { eprintln!("compile error: {:?}", e); std::process::exit(1); });
    let params = compiled.default_params.clone();

    // ── Resolve --data flags (gh#90) ─────────────────────────────────
    //
    // Polymorphic `--data` mirrors the existing `--table NAME=FILE`
    // pattern. Two forms: `--data PATH` (single-stream) and
    // `--data NAME=PATH` (repeatable, multi-stream). The resolver
    // returns one (stream_name, path) pair per bound stream.
    //
    // fit-toml fallback: when no --data flags supplied AND --fit was,
    // pull the multi-stream binding from the toml's
    // `[data.observations]` block. CLI flags always win; we emit an
    // info line if both forms were supplied.
    let model_obs_names: Vec<String> = model.observations.iter()
        .map(|o| o.name.clone()).collect();
    let cli_data_specs: Vec<crate::args::types::DataSpec> = a.data.clone();
    let bound_streams: Vec<(String, std::path::PathBuf)> = if cli_data_specs.is_empty() {
        // No CLI --data. Try the fit-toml fallback.
        if let Some(fit_path) = a.fit.as_ref() {
            load_data_observations_from_fit_toml(fit_path, &model_obs_names)
                .unwrap_or_else(|e| {
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
            eprintln!("pfilter: --data on CLI overrides --fit toml [data.observations]");
        }
        crate::util::resolve_data_specs(&cli_data_specs, &model_obs_names, obs_name.as_deref())
            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
    };

    if bound_streams.is_empty() {
        eprintln!("error: zero streams resolved from --data / --fit toml. \
                   pfilter requires at least one observation stream.");
        std::process::exit(1);
    }

    // Load data — route the time column through the calendar-time boundary
    // translator (numeric or dated, per --time-format + the model origin).
    let time_opts = TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: compiled.model.simulation.t_start,
        format: a.inference.time_format,
    };

    // Build per-stream Observation vectors. Every stream binds its value
    // column by NAME — the data column header must match the model's
    // `observe` name exactly. There is no positional fallback: a typo'd
    // or wrong-cased header is a located error, not a silent bind to the
    // positionally-first value column (G1). A single-stream `time\tcases`
    // file still loads because `cases` matches the header by name.
    let n_streams = bound_streams.len();
    let mut per_stream_obs: Vec<Vec<Observation>> = Vec::with_capacity(n_streams);
    for (sname, spath) in &bound_streams {
        let path_str = spath.to_string_lossy().into_owned();
        let result = load_data_tsv_column(&path_str, sname, &time_opts);
        match result {
            Ok(obs) => per_stream_obs.push(obs),
            Err(e) => {
                eprintln!("error: cannot load data column '{}' from {}: {}",
                    sname, path_str, e);
                std::process::exit(1);
            }
        }
    }

    // Validate every stream shares the same obs_times schedule.
    {
        let first_times: Vec<f64> = per_stream_obs[0].iter().map(|o| o.time).collect();
        for (i, obs) in per_stream_obs.iter().enumerate().skip(1) {
            let times: Vec<f64> = obs.iter().map(|o| o.time).collect();
            if times.len() != first_times.len()
                || times.iter().zip(&first_times).any(|(a, b)| (a - b).abs() > 1e-9)
            {
                eprintln!(
                    "error: observation times for stream '{}' differ from \
                     stream '{}'. All streams must share identical observation \
                     times.",
                    bound_streams[i].0, bound_streams[0].0);
                std::process::exit(1);
            }
        }
    }

    // Canonical observations: first stream's vector. Downstream
    // single-stream code paths (trace, prequential, save_filtering,
    // save_paths, save_final_state) consume `obs.time` and
    // `obs.value`; for multi-stream the `.value` is the first
    // stream's, which is fine for trace timestamps and the
    // single-loglik scalar output (which sums across streams).
    let observations = per_stream_obs[0].clone();

    eprintln!("pfilter: {} observations × {} streams, {} particles, dt={}, seed={}",
        observations.len(), n_streams, n_particles, dt, seed);
    if n_streams > 1 {
        eprintln!("  streams: {}", bound_streams.iter()
            .map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));
    }

    // ── Resolve IR observation models for each bound stream ──────────
    //
    // Each bound stream resolves to one IR observation block by exact
    // name match. Family-root expansion already happened in the
    // resolver. For the single-stream + --flow override path we
    // additionally rewrite the projection to a flow-sum below.
    let mut bound_ir: Vec<ir::observation::ObservationModel> =
        Vec::with_capacity(n_streams);
    for (sname, _) in &bound_streams {
        match model.observations.iter().find(|o| &o.name == sname) {
            Some(o) => bound_ir.push(o.clone()),
            None => {
                eprintln!("error: bound stream '{}' has no matching IR \
                    observation block (the resolver should have caught \
                    this; this is a bug). Available: {}", sname,
                    model.observations.iter().map(|o| o.name.as_str())
                        .collect::<Vec<_>>().join(", "));
                std::process::exit(1);
            }
        }
    }

    // --flow override: only valid for single-stream pfilter. Building
    // a flow override for a multi-stream invocation would silently
    // overwrite each stream's per-block projection with the same flow
    // sum, which is rarely what the user wants.
    if n_streams > 1 && flow_name.is_some() {
        eprintln!(
            "error: --flow <NAME> is incompatible with multi-stream --data \
             ({} streams bound). --flow rewrites a single stream's \
             projection; for multi-stream pfilter, each stream uses its \
             own IR projection.", n_streams);
        std::process::exit(1);
    }

    let projections: Vec<sim::inference::multi_stream_obs::StreamProjection> =
        if let Some(ref name) = flow_name {
            // Single-stream path with --flow override.
            let indices: Vec<usize> = model.transitions.iter().enumerate()
                .filter(|(_, tr)| tr.name == *name || tr.name.starts_with(&format!("{}_", name)))
                .map(|(i, _)| i)
                .collect();
            if indices.is_empty() {
                eprintln!("error: no transition named '{}' found. Available: {}",
                    name, model.transitions.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "));
                std::process::exit(1);
            }
            eprintln!("pfilter: --flow override → incidence({}) ({} transitions)", name, indices.len());
            vec![sim::inference::multi_stream_obs::StreamProjection::FlowSum(indices)]
        } else {
            bound_ir.iter().map(|o| {
                sim::inference::multi_stream_obs::StreamProjection::from_ir(
                    &o.projection, &compiled, &o.name,
                ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
            }).collect()
        };

    eprintln!("pfilter: bound streams: {}", bound_ir.iter()
        .map(|o| format!("{}({})", o.name, match &o.likelihood {
            ir::observation::Likelihood::NegBinomial(_)  => "neg_binomial",
            ir::observation::Likelihood::Normal(_)       => "normal",
            ir::observation::Likelihood::Poisson(_)      => "poisson",
            ir::observation::Likelihood::Binomial(_)     => "binomial",
            ir::observation::Likelihood::BetaBinomial(_) => "beta_binomial",
            ir::observation::Likelihood::Bernoulli(_)    => "bernoulli",
        })).collect::<Vec<_>>().join(", "));

    // gh#90: emit unbound-streams warning if a multi-block model
    // is only partially covered by the resolved bindings.
    {
        let all_names: Vec<String> = model.observations.iter()
            .map(|o| o.name.clone()).collect();
        let bound_names: Vec<String> = bound_streams.iter()
            .map(|(n, _)| n.clone()).collect();
        if let Some(w) = crate::util::format_unbound_streams_warning(
            "pfilter", &all_names, &bound_names,
        ) {
            eprint!("{}", w);
        }
    }

    // gh#174: reject a positive incidence observation at the model origin
    // (zero-width first window → -Inf masquerading as filter degeneracy).
    // Checked per stream before the filter runs. The --flow override rewrites
    // the single stream's projection to an incidence flow-sum regardless of
    // the IR block, so treat that case as incidence explicitly.
    let t_start = compiled.model.simulation.t_start;
    for (stream_obs, ir_obs) in per_stream_obs.iter().zip(bound_ir.iter()) {
        let times: Vec<f64> = stream_obs.iter().map(|o| o.time).collect();
        // F4: an observation strictly before the model origin can never be
        // propagated to — its window yields zero substeps yet it is still
        // scored (a silent wrong answer). Reject loudly before the filter runs.
        if let Err(e) = crate::util::check_obs_before_origin(&ir_obs.name, t_start, &times) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
        let first_value = stream_obs.first().map(|o| o.value).unwrap_or(0.0);
        let incidence_override = ir::observation::Projection::CumulativeFlow(String::new());
        let effective_projection = if flow_name.is_some() {
            &incidence_override
        } else {
            &ir_obs.projection
        };
        if let Err(e) = crate::util::check_incidence_origin_window(
            &ir_obs.name, effective_projection, t_start, &times, first_value,
        ) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }

    // Build process + observation model via traits
    let compiled = std::sync::Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone(), dt);

    let obs_times: Vec<f64> = observations.iter().map(|o| o.time).collect();
    let stream_specs: Vec<StreamSpec> = bound_ir.iter().zip(projections.into_iter())
        .zip(per_stream_obs.iter()).map(|((o, projection), stream_obs)| StreamSpec {
            projection,
            ir_model: o.clone(),
            observations: stream_obs.iter().map(|x| x.value).collect(),
            obs_times: obs_times.clone(),
        }).collect();
    let obs_model = MultiStreamObsModel::new(stream_specs, compiled.clone())
        .unwrap_or_else(|e| {
            eprintln!("error: observation model construction failed: {:?}", e);
            std::process::exit(1);
        });

    // Record ancestry when either --save-paths or --save-filtering is
    // active. Both flags consume the same per-step snapshot data; the
    // difference is only in what we write to disk at the end.
    let need_ancestry = save_paths.is_some() || save_filtering.is_some();
    let smc_config = SMCConfig {
        n_particles,
        dt,
        t_start: compiled.model.simulation.t_start,
        skip_first_obs_from_loglik: false,
        record_ancestry: need_ancestry,
        record_prequential: save_prequential.is_some(),
        // Non-CAS legacy path keeps the env-resolved wall-clock budget.
        pf_wallclock_disabled: false,
    };

    // --save-filtering caveat log. Fires unconditionally (not quietable)
    // because the failure mode — plotting filtering marginals as if
    // they were smoothing paths — is silent. See
    // docs/dev/proposals/2026-04-19-pf-latent-trajectories.md.
    if save_filtering.is_some() {
        eprintln!("[info] --save-filtering emits filtering marginals \
                   p(x_t | y_{{1..t}}), not smoothing paths. Joining \
                   particles across time by index does NOT yield \
                   trajectory samples from the posterior. For coherent \
                   sample paths use --save-paths N PATH.");
    }

    // ── Default-CAS: record this eval as a content-addressed `Pfilter` leaf
    // (gh#147 M3.3). A pfilter eval is a single leaf — replicates are averaged
    // in-leaf, not an axis. Written after the loglik is computed, from both
    // the replicate and single-run paths; the stdout `loglik = …` line and the
    // explicit `--save-*` paths are unchanged. Idempotent: same inputs → same
    // run_id → same path.
    let write_cas_leaf = |mean_ll: f64, sd_ll: f64, n_reps: usize| {
        let data_hashes: Vec<(String, String)> = bound_streams.iter()
            .filter_map(|(name, path)| std::fs::read(path).ok()
                .map(|bytes| (name.clone(), runid::ContentHash::digest_bytes(&bytes).to_hex())))
            .collect();
        let scored: Vec<(String, f64)> = resolved.params.iter()
            .map(|p| (p.name.clone(), p.value)).collect();
        // `--flow` override transition indices (empty for the default
        // per-stream projection); same selection logic as the projection.
        let flow_indices: Vec<u32> = flow_name.as_ref().map(|name|
            model.transitions.iter().enumerate()
                .filter(|(_, tr)| tr.name == *name || tr.name.starts_with(&format!("{}_", name)))
                .map(|(i, _)| i as u32).collect()
        ).unwrap_or_default();
        let stem = crate::hashing::path_stem_slug(&ir_path).unwrap_or_else(|| "model".to_string());
        let ir_version_str = ir::IR_VERSION.trim().to_string();
        let ctx = crate::pfilter_cas::PfilterCtx {
            model: &model,
            ir_version: &ir_version_str,
            engine_version: crate::version::VERSION_SHORT,
            stem: &stem,
            data: &data_hashes,
            params: &scored,
            particles: n_particles as u32,
            replicates: n_reps as u32,
            dt,
            obs_block: flow_name.as_deref().unwrap_or(""),
            flow_indices: &flow_indices,
            seed,
        };
        let resolved_id = match crate::pfilter_cas::resolve_pfilter(&ctx) {
            Ok(r) => r,
            Err(e) => { eprintln!("warning: pfilter CAS identity: {}", e); return; }
        };
        let root = crate::run_paths::output_root(None, None);
        let cas_path = runid::store_path(&root, runid::ArtifactKind::Pfilter, &resolved_id.levels);
        let store = runid::FsCasStore::new(&root);
        let sd_field = if n_reps > 1 { serde_json::json!(sd_ll) } else { serde_json::Value::Null };
        let inputs_json = serde_json::json!({
            "loglik": mean_ll,
            "loglik_sd": sd_field,
            "n_replicates": n_reps,
            "n_particles": n_particles,
            "params": scored.iter().map(|(n, v)| (n.as_str(), *v)).collect::<Vec<_>>(),
            "data_hashes": data_hashes.iter().map(|(n, h)| (n.as_str(), h.as_str())).collect::<Vec<_>>(),
        });
        let running = crate::pfilter_cas::build_pfilter_record(
            &resolved_id, &ir_version_str, runid::RunStatus::Running,
            serde_json::Value::Null, &ir_path);
        let claim = match store.claim_streaming(&cas_path, running) {
            Ok(c) => c,
            Err(e) => { eprintln!("warning: claim pfilter leaf {}: {}", cas_path.display(), e); return; }
        };
        let mut body = format!("loglik = {}\nn_replicates = {}\nn_particles = {}\n",
            mean_ll, n_reps, n_particles);
        if n_reps > 1 { body.push_str(&format!("loglik_sd = {}\n", sd_ll)); }
        let _ = std::fs::write(claim.dir().join("loglik.toml"), body);
        let completed = crate::pfilter_cas::build_pfilter_record(
            &resolved_id, &ir_version_str, runid::RunStatus::Completed,
            inputs_json, &ir_path);
        if let Err(e) = claim.finalize(completed) {
            eprintln!("warning: finalize pfilter leaf {}: {}", cas_path.display(), e);
        } else {
            crate::status::done("stored", cas_path.display());
            crate::status::hint(format!("camdl cat {}", resolved_id.run_id.to_hex()));
        }
    };

    // ── Replicates mode: run N independent pfilters, output loglik summary ──
    if n_replicates > 1 {
        // gh#audit-H13: --parallel / CAMDL_PARALLEL was previously declared
        // on InferenceCore (args/mod.rs) but never read by cmd_pfilter, so
        // `camdl pfilter --parallel 16 --replicates 100` ran single-
        // threaded. Build a rayon pool from a.inference.parallel before
        // the replicate loop, mirroring profile.rs:849-853 / if2.rs:369-374.
        let parallel = a.inference.parallel;
        if parallel > 0 {
            // build_global is idempotent across processes; ignore the
            // "already initialised" Err so re-entry from tests is safe.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build_global();
        }
        eprintln!("pfilter: {} replicates × {} particles{}",
            n_replicates, n_particles,
            if parallel > 0 { format!(" (parallel = {})", parallel) }
            else { String::new() });
        // Im20 in 2026-04-19 inference review batch 3: replicate
        // seeding was `seed + rep`, which gives highly correlated
        // ChaCha8 initial states across replicates. Use the
        // golden-ratio multiplier to decorrelate low bits.
        const SEED_STRIDE: u64 = 0x9e3779b97f4a7c15;
        // Per-replicate progress bar; the metric is the running-mean loglik
        // (replicates are noisy estimates of ONE loglik at fixed params, so
        // the mean is the live estimate — not a search "best"). Ticked from
        // the parallel loop (`Task` is `Send + Sync`); honors `--progress
        // none/plain`. Finer per-obs-window progress would need a callback
        // into `sim::bootstrap_filter` — deferred (an inference-crate change).
        let bar = crate::progress::Reporter::new().task(n_replicates as u64, "pfilter", "reps");
        let acc = std::sync::Mutex::new((0.0_f64, 0usize)); // (Σ loglik, count)
        let logliks: Vec<f64> = (0..n_replicates).into_par_iter().map(|rep| {
            let rep_seed = seed.wrapping_add((rep as u64).wrapping_mul(SEED_STRIDE));
            let result = bootstrap_filter(
                &process, &obs_model, &params, &smc_config, rep_seed,
            ).unwrap_or_else(|e| {
                eprintln!("pfilter replicate {} error: {:?}", rep + 1, e);
                std::process::exit(1);
            });
            bar.inc(1);
            if result.log_likelihood.is_finite() {
                if let Ok(mut g) = acc.lock() {
                    g.0 += result.log_likelihood;
                    g.1 += 1;
                    bar.set(format!("ll={:.1}", g.0 / g.1 as f64));
                }
            }
            result.log_likelihood
        }).collect();
        bar.finish();

        let mean_ll = logliks.iter().sum::<f64>() / n_replicates as f64;
        let var_ll = logliks.iter().map(|&l| (l - mean_ll).powi(2)).sum::<f64>() / (n_replicates - 1) as f64;
        let sd_ll = var_ll.sqrt();

        eprintln!("loglik = {:.1} ± {:.1} ({} replicates, N={})", mean_ll, sd_ll, n_replicates, n_particles);

        // Output: TSV of seed + loglik, or summary to --output
        match &output_path {
            Some(path) => {
                let mut f = std::fs::File::create(path)
                    .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
                writeln!(f, "seed\tloglik").unwrap();
                for (rep, ll) in logliks.iter().enumerate() {
                    writeln!(f, "{}\t{:.4}", seed.wrapping_add((rep as u64).wrapping_mul(SEED_STRIDE)), ll).unwrap();
                }
                eprintln!("replicate logliks written to {}", path);
            }
            None => {
                println!("seed\tloglik");
                for (rep, ll) in logliks.iter().enumerate() {
                    println!("{}\t{:.4}", seed.wrapping_add((rep as u64).wrapping_mul(SEED_STRIDE)), ll);
                }
            }
        }
        write_cas_leaf(mean_ll, sd_ll, n_replicates);
        return;
    }

    // ── Single pfilter run ─────────────────────────────────────────────────
    let result = bootstrap_filter(
        &process, &obs_model, &params, &smc_config, seed,
    ).unwrap_or_else(|e| {
        eprintln!("pfilter error: {:?}", e);
        std::process::exit(1);
    });

    // Write trace diagnostics
    let trace_to_stdout = trace_path.as_deref() == Some("-");
    if let Some(ref path) = trace_path {
        let mut out: Box<dyn Write> = if path == "-" {
            Box::new(std::io::BufWriter::new(std::io::stdout().lock()))
        } else {
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
            Box::new(std::io::BufWriter::new(f))
        };
        if let Some(ref preds) = result.predictions {
            writeln!(out, "time\tll_increment\tESS\tobs_mean\tobs_q05\tobs_q50\tobs_q95\tstate_mean\tstate_q05\tstate_q50\tstate_q95\tobserved").unwrap();
            for (i, obs) in observations.iter().enumerate() {
                let p = &preds[i];
                writeln!(out, "{}\t{:.4}\t{:.1}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.1}\t{:.0}\t{:.0}\t{:.0}\t{:.0}",
                    obs.time, result.ll_increments[i], result.ess_trace[i],
                    p.obs_mean, p.obs_q05, p.obs_q50, p.obs_q95,
                    p.state_mean, p.state_q05, p.state_q50, p.state_q95,
                    obs.value).unwrap();
            }
        } else {
            writeln!(out, "time\tll_increment\tESS\tobserved").unwrap();
            for (i, obs) in observations.iter().enumerate() {
                writeln!(out, "{}\t{:.4}\t{:.1}\t{:.0}",
                    obs.time, result.ll_increments[i], result.ess_trace[i],
                    obs.value).unwrap();
            }
        }
        drop(out);
        if path != "-" {
            eprintln!("trace written to {}", path);
        }
    }

    // Save final particle states
    if let Some(ref path) = save_final_state {
        if let Some(ref states) = result.final_states {
            write_final_states(path, states, &model).unwrap_or_else(|e| {
                eprintln!("error writing final states: {}", e);
                std::process::exit(1);
            });
            eprintln!("final particle states ({} particles) written to {}", states.len(), path);
        }
    }

    // Save smoothing paths (--save-paths N PATH): ancestor-trace
    // N trajectory samples from the smoothing distribution.
    if let Some((n_paths, ref path)) = save_paths {
        let trace = result.ancestry.as_ref().expect(
            "record_ancestry must be true when save_paths is set");
        let paths = sim::inference::ancestor_trace::sample_paths(
            trace, n_paths, seed);
        write_paths_tsv(path, &paths, &model).unwrap_or_else(|e| {
            eprintln!("error writing paths: {}", e);
            std::process::exit(1);
        });
        eprintln!("{} sample paths written to {}", n_paths, path);
    }

    // Save prequential trace (--save-prequential STEM): writes
    // {STEM}.tsv (per-step scalar scores) + {STEM}.json (full trace,
    // incl. predictive samples unless --no-save-samples).
    if let Some(ref stem) = save_prequential {
        let recorded = result.prequential.as_ref().expect(
            "record_prequential must be true when save_prequential is set");
        let y_obs: Vec<f64> = observations.iter().map(|o| o.value).collect();
        let mut trace = sim::inference::prequential::build_trace(
            recorded, &y_obs, &result.ess_trace, 0);
        if !save_samples {
            for step in &mut trace.steps { step.y_pred_samples.clear(); }
            trace.warnings.push(
                sim::inference::prequential::PrequentialWarning::SamplesNotSaved);
        }
        write_prequential_outputs(stem, &trace).unwrap_or_else(|e| {
            eprintln!("error writing prequential: {}", e);
            std::process::exit(1);
        });
        eprintln!(
            "prequential trace written: elpd={:.2}, mean_crps={:.3}, PIT 90% cov={:.2}",
            trace.elpd(), trace.mean_crps(), trace.pit_coverage(0.90));
    }

    // Save filtering marginals (--save-filtering PATH): per-step
    // pre-resample particle states + log-weights. Caveat log fired
    // earlier at SMCConfig construction.
    if let Some(ref path) = save_filtering {
        let trace = result.ancestry.as_ref().expect(
            "record_ancestry must be true when save_filtering is set");
        write_filtering_tsv(path, trace, &model).unwrap_or_else(|e| {
            eprintln!("error writing filtering: {}", e);
            std::process::exit(1);
        });
        eprintln!("filtering marginals written to {}", path);
    }

    // Write loglik
    match &output_path {
        Some(path) => {
            std::fs::write(path, format!("{:.4}\n", result.log_likelihood))
                .unwrap_or_else(|e| { eprintln!("cannot write {}: {}", path, e); std::process::exit(1); });
            eprintln!("loglik written to {}", path);
        }
        None => {
            if trace_to_stdout {
                eprintln!("{:.4}", result.log_likelihood);
            } else {
                println!("{:.4}", result.log_likelihood);
            }
        }
    }

    // Single run → one replicate, no spread.
    write_cas_leaf(result.log_likelihood, 0.0, 1);
}

use crate::caltime_load::{check_substeps_and_grid, convert_time_column, TimeFormat, TimeOpts};

/// gh#90: fit-toml fallback for `camdl pfilter --fit fit.toml` (no CLI
/// `--data` flags). Reads `[data]` from the toml and returns a list of
/// (stream_name, path) bindings — same shape `resolve_data_specs`
/// produces, so downstream code is uniform.
pub fn load_data_observations_from_fit_toml(
    fit_path: &std::path::Path,
    model_obs_names: &[String],
) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    let path_str = fit_path.to_string_lossy().into_owned();
    let fit_cfg = crate::fit::config_v2::FitConfigV2::load(&path_str)
        .map_err(|e| format!("failed to load --fit toml '{}': {}", path_str, e))?;
    let data_spec = fit_cfg.data_spec()?;
    let effective = data_spec.effective_observations(model_obs_names)?;
    if effective.is_empty() {
        return Err(format!(
            "--fit toml '{}' [data] resolves to zero observation streams.",
            path_str));
    }
    // Sort by name for deterministic ordering (matches fit/runner.rs).
    let mut entries: Vec<(String, std::path::PathBuf)> = effective.iter()
        .map(|(k, v)| (k.clone(), std::path::PathBuf::from(v)))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// Load observations from a specific column in a TSV file.
/// The column name must match a header field. First column is always time.
pub fn load_data_tsv_column(
    path: &str,
    column: &str,
    opts: &TimeOpts,
) -> Result<Vec<Observation>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let mut lines = content.lines();
    let header = lines.next().ok_or("empty data file")?;
    let cols: Vec<&str> = header.split('\t').collect();

    // Find the column index for the requested stream name. Binding is
    // strict by name — there is NO positional fallback. A typo'd,
    // wrong-cased, or renamed header is a located error, not a silent
    // bind to whatever column happens to be positionally first (G1: a
    // wrong-answer-with-exit-0). The data column header must match the
    // model's `observe` name exactly.
    let col_idx = cols.iter().position(|&c| c == column)
        .ok_or_else(|| {
            let available = if cols.len() > 1 {
                cols[1..].join(", ")
            } else {
                "(no value columns — only a time column)".to_string()
            };
            format!(
                "observation stream '{column}' not found in data file '{path}'. \
                 Value column headers present: [{available}]. \
                 Fix: rename the data column to '{column}' (it must match the \
                 model's `observe` name exactly, case-sensitive), or rename the \
                 model's observation block to match the data header."
            )
        })?;

    // Two-pass: collect raw time cells + values, then convert the whole
    // time column at once (whole-column detection — proposal §6.3).
    let mut time_cells: Vec<&str> = Vec::new();
    let mut rows: Vec<usize> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for (line_num, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= col_idx {
            return Err(format!("line {}: expected {}+ columns, got {}",
                line_num + 2, col_idx + 1, fields.len()));
        }
        let value: f64 = fields[col_idx].trim().parse()
            .map_err(|_| format!("line {}: cannot parse value '{}' in column '{}'",
                line_num + 2, fields[col_idx], column))?;
        if !value.is_finite() {
            return Err(format!(
                "line {} (t='{}'): non-finite observation value '{}' in column '{}' \
                 — NaN and infinities are not valid observations. Fix or remove the row.",
                line_num + 2, fields[0].trim(), fields[col_idx].trim(), column));
        }
        time_cells.push(fields[0]);
        rows.push(line_num + 2);
        values.push(value);
    }

    finalize_observations(time_cells, values, rows, opts)
}

/// Shared back-half: convert the raw time column, run the distinct-substep +
/// off-grid checks, validate chronological order, and zip into `Observation`s.
fn finalize_observations(
    time_cells: Vec<&str>,
    values: Vec<f64>,
    rows: Vec<usize>,
    opts: &TimeOpts,
) -> Result<Vec<Observation>, String> {
    let row_offset = rows.first().copied().unwrap_or(2);
    let times = convert_time_column(&time_cells, opts, row_offset)?;
    let was_dated = opts.format != TimeFormat::Numeric
        && time_cells.iter().any(|c| {
            !c.trim().is_empty() && ir::caltime::parse_iso_date(c.trim()).is_ok()
                && c.trim().parse::<f64>().is_err()
        });
    check_substeps_and_grid(&times, &rows, opts, was_dated)?;

    let observations: Vec<Observation> = times
        .iter()
        .zip(values.iter())
        .map(|(&time, &value)| Observation { time, value })
        .collect();

    // Validate chronological ordering (equal times OK — multi-stream observations)
    for i in 1..observations.len() {
        if observations[i].time < observations[i - 1].time {
            return Err(format!(
                "observations not in chronological order: t={} at row {} follows t={} at row {}",
                observations[i].time, rows[i], observations[i - 1].time, rows[i - 1]
            ));
        }
    }

    Ok(observations)
}

use std::io::Write;

/// Write a `PrequentialTrace` to `{stem}.tsv` + `{stem}.json`.
/// Per-step scalar scores (t, y_obs, log_score, crps, pit, ess) go to
/// TSV; full typed trace (incl. predictive samples when retained) to
/// JSON. Downstream tools join on `stem` to avoid re-running the PF.
fn write_prequential_outputs(
    stem: &str,
    trace: &sim::inference::prequential::PrequentialTrace,
) -> std::io::Result<()> {
    use std::io::Write;
    let tsv_path = format!("{}.tsv", stem);
    let json_path = format!("{}.json", stem);
    let mut tsv = std::io::BufWriter::new(std::fs::File::create(&tsv_path)?);
    writeln!(tsv, "t\ty_obs\tlog_score\tcrps\tpit\tess")?;
    for s in &trace.steps {
        writeln!(tsv, "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.2}",
            s.t, s.y_obs, s.log_score, s.crps, s.pit, s.ess)?;
    }
    drop(tsv);
    let json = serde_json::to_string_pretty(trace)
        .map_err(std::io::Error::other)?;
    std::fs::write(&json_path, json)?;
    Ok(())
}

/// Write final particle states to a TSV file.
/// Columns: particle_id, then one column per compartment, then flow_<transition>.
fn write_final_states(
    path: &str,
    states: &[sim::inference::ParticleState],
    model: &ir::Model,
) -> Result<(), String> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {}", path, e))?;

    // Header
    write!(f, "particle").unwrap();
    for c in &model.compartments {
        if c.kind == ir::model::CompartmentKind::Integer {
            write!(f, "\t{}", c.name).unwrap();
        }
    }
    for tr in &model.transitions {
        write!(f, "\tflow_{}", tr.name).unwrap();
    }
    writeln!(f).unwrap();

    // Rows
    for (i, state) in states.iter().enumerate() {
        write!(f, "{}", i).unwrap();
        for &c in &state.counts {
            write!(f, "\t{}", c).unwrap();
        }
        for &fl in &state.flow_accumulators {
            write!(f, "\t{}", fl).unwrap();
        }
        writeln!(f).unwrap();
    }

    Ok(())
}

/// Write ancestor-traced smoothing paths as a long-format TSV.
/// Schema matches `camdl simulate --replicates N` for pipeline reuse:
/// columns `path`, `time`, and one column per integer compartment.
/// Each `path ∈ 1..=N` is an equally-weighted sample from the
/// smoothing distribution; no log_weight column needed.
fn write_paths_tsv(
    path: &str,
    paths: &[sim::inference::ancestor_trace::SampledPath],
    model: &ir::Model,
) -> Result<(), String> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {}", path, e))?;

    write!(f, "path\ttime").unwrap();
    let comp_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str())
        .collect();
    for name in &comp_names {
        write!(f, "\t{}", name).unwrap();
    }
    // gh#48: emit one column per observation stream alongside the
    // compartment columns. Carries the model's declared projection
    // (incidence/prevalence/arithmetic) so downstream tooling doesn't
    // have to reconstruct it by finite-differencing compartment
    // counts — which silently breaks under event + balance
    // interactions (the constrained compartment becomes an
    // unfaithful tracker of its own dynamics, e.g. ΔR records fake
    // flows on cohort-event days).
    //
    // Column source: `SampledPath.projections[t][stream]` (= `mean()`
    // from the obs model at record time, walked along the ancestor
    // chain in lockstep with states). Skipped entirely when the obs
    // model's `mean()` returns empty (the trait default).
    let stream_names: &[String] = paths.first()
        .map(|p| p.stream_names.as_slice())
        .unwrap_or(&[]);
    let has_projections = !stream_names.is_empty()
        && paths.iter().all(|p| !p.projections.is_empty());
    if has_projections {
        for name in stream_names {
            write!(f, "\t{}", name).unwrap();
        }
    }
    writeln!(f).unwrap();

    for (i, p) in paths.iter().enumerate() {
        for (t_idx, &obs_t) in p.obs_times.iter().enumerate() {
            write!(f, "{}\t{}", i + 1, obs_t).unwrap();
            // Only the first n_comp_names columns of the state are
            // integer compartments; the PF records all state counts,
            // but we present only the public compartments.
            for k in 0..comp_names.len() {
                write!(f, "\t{}", p.states[t_idx][k]).unwrap();
            }
            if has_projections {
                let row_projs = &p.projections[t_idx];
                for s in 0..stream_names.len() {
                    if let Some(&v) = row_projs.get(s) {
                        write!(f, "\t{}", v).unwrap();
                    } else {
                        write!(f, "\tNaN").unwrap();
                    }
                }
            }
            writeln!(f).unwrap();
        }
    }
    Ok(())
}

/// Write filtering marginals as a long-format TSV. Schema:
/// `time`, `particle`, one column per integer compartment, and
/// `log_weight`. `particle` is an in-step index only — it is NOT
/// stable across `time`, and joining particles across `time` by
/// index is NOT a sample path.
fn write_filtering_tsv(
    path: &str,
    trace: &sim::inference::ancestor_trace::AncestorTrace,
    model: &ir::Model,
) -> Result<(), String> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {}", path, e))?;

    write!(f, "time\tparticle").unwrap();
    let comp_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str())
        .collect();
    for name in &comp_names {
        write!(f, "\t{}", name).unwrap();
    }
    writeln!(f, "\tlog_weight").unwrap();

    for (t_idx, &obs_t) in trace.obs_times.iter().enumerate() {
        for (i, state) in trace.states[t_idx].iter().enumerate() {
            write!(f, "{}\t{}", obs_t, i + 1).unwrap();
            for k in 0..comp_names.len() {
                write!(f, "\t{}", state[k]).unwrap();
            }
            writeln!(f, "\t{:.6}", trace.log_weights[t_idx][i]).unwrap();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_temp_tsv(name: &str, content: &str) -> String {
        let path = std::env::temp_dir().join(format!("camdl_test_{}.tsv", name));
        std::fs::write(&path, content).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn numeric_opts() -> TimeOpts<'static> {
        TimeOpts {
            origin: None,
            time_unit: "days",
            dt: 1.0,
            t_start: 0.0,
            format: TimeFormat::Auto,
        }
    }

    #[test]
    fn load_data_rejects_out_of_order() {
        let path = write_temp_tsv("out_of_order", "time\tcases\n7\t10\n14\t20\n10\t15\n21\t30\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_err(), "should reject out-of-order times");
        let err = result.err().unwrap();
        assert!(err.contains("not in chronological order"), "error message: {}", err);
        assert!(err.contains("t=10"), "should mention the offending time: {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_accepts_equal_times() {
        // Equal times are valid (multi-stream observations at same time point)
        let path = write_temp_tsv("equal_times", "time\tcases\n7\t10\n7\t5\n14\t20\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_ok(), "equal times should be accepted: {:?}", result.err());
        let obs = result.unwrap();
        assert_eq!(obs.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_accepts_sorted() {
        let path = write_temp_tsv("sorted", "time\tcases\n7\t10\n14\t20\n21\t30\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
        std::fs::remove_file(&path).ok();
    }

    // ── G1: by-name binding is strict — no positional fallback ──────────
    //
    // A stream requested by name against a file whose column does not match
    // (typo / wrong case / renamed header) must ERROR with a located
    // message, never silently bind a value column by position. Pre-fix, the
    // inner 2-column fallback (`if cols.len() == 2 { Some(1) }`) made a
    // mis-cased single-value file load against column 1 — a wrong answer
    // with exit 0.

    #[test]
    fn load_data_tsv_column_rejects_miscased_name_in_2col_file() {
        // Header column is `Cases` (capital C); model asks for `cases`.
        // Pre-fix: 2-column fallback binds column 1 and loads. Post-fix:
        // located error naming the requested column + available headers.
        let path = write_temp_tsv("miscased_2col", "time\tCases\n7\t10\n14\t20\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_err(),
            "a mis-cased column name must NOT silently bind by position; got Ok({:?})",
            result.as_ref().ok());
        let err = result.err().unwrap();
        assert!(err.contains("cases"),
            "error must name the requested stream/column 'cases': {}", err);
        assert!(err.contains("Cases"),
            "error must list the headers actually present (incl. 'Cases'): {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_tsv_column_rejects_renamed_name_in_wide_file() {
        // Multi-column file, requested name absent. (This path already
        // errored pre-fix, but pin the located-message quality.)
        let path = write_temp_tsv("renamed_wide",
            "time\tcase_count\tdeaths\n7\t10\t1\n14\t20\t2\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_err(), "absent column name must error");
        let err = result.err().unwrap();
        assert!(err.contains("cases"), "error must name requested column: {}", err);
        assert!(err.contains("case_count") && err.contains("deaths"),
            "error must list available headers: {}", err);
        std::fs::remove_file(&path).ok();
    }

    // ── G1/NaN: non-finite observation values are rejected at load ──────
    //
    // `"NaN".parse::<f64>()` returns `Ok(NaN)` and `"inf"`/`"Infinity"`
    // return `Ok(±inf)`. Pre-fix these flowed straight into the likelihood.
    // Post-fix: located error (file path implied by caller, column, row).

    #[test]
    fn load_data_tsv_column_rejects_nan_value() {
        let path = write_temp_tsv("nan_value", "time\tcases\n7\t10\n14\tNaN\n21\t30\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_err(),
            "a NaN observation value must be rejected at load; got Ok({:?})",
            result.as_ref().ok());
        let err = result.err().unwrap();
        assert!(err.contains("cases"), "error must name the column: {}", err);
        assert!(err.contains('3') || err.contains("14"),
            "error must locate the offending row/time (line 3, t=14): {}", err);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_tsv_column_rejects_inf_value() {
        let path = write_temp_tsv("inf_value", "time\tcases\n7\t10\n14\tinf\n21\t30\n");
        let result = load_data_tsv_column(&path, "cases", &numeric_opts());
        assert!(result.is_err(),
            "an infinite observation value must be rejected at load; got Ok({:?})",
            result.as_ref().ok());
        let err = result.err().unwrap();
        assert!(err.contains("cases"), "error must name the column: {}", err);
        std::fs::remove_file(&path).ok();
    }

    // ── Negative control: well-formed input is unchanged ────────────────
    //
    // Guards against a vacuous test: a matching column name loads the
    // SAME values as the 2-column loader on the equivalent file, and the
    // by-name path binds the requested column (not column 1) in a wide
    // file. This is the happy path that must remain byte-identical.

    #[test]
    fn load_data_tsv_column_happy_path_loads_named_column() {
        // Wide file: `cases` is column 2 (not column 1). By-name binding
        // must pick column 2's values, not deaths in column 1.
        let path = write_temp_tsv("happy_wide",
            "time\tdeaths\tcases\n7\t1\t10\n14\t2\t20\n21\t3\t30\n");
        let obs = load_data_tsv_column(&path, "cases", &numeric_opts())
            .expect("well-formed named column must load");
        assert_eq!(obs.len(), 3);
        assert_eq!(obs[0].value, 10.0);
        assert_eq!(obs[1].value, 20.0);
        assert_eq!(obs[2].value, 30.0);
        assert_eq!(obs[0].time, 7.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_tsv_column_happy_path_2col_named_file() {
        // The happy single-stream case: a `time\tcases` file binds the
        // `cases` column *because it is named* `cases` (which happens to
        // be column 1), not by position. Values must round-trip exactly —
        // this is the legacy single-stream schema, unchanged by the
        // fallback deletion.
        let path = write_temp_tsv("happy_2col", "time\tcases\n7\t10\n14\t20\n21\t30\n");
        let obs = load_data_tsv_column(&path, "cases", &numeric_opts())
            .expect("named single-stream load");
        assert_eq!(obs.len(), 3);
        assert_eq!(obs[0], Observation { time: 7.0, value: 10.0 });
        assert_eq!(obs[1], Observation { time: 14.0, value: 20.0 });
        assert_eq!(obs[2], Observation { time: 21.0, value: 30.0 });
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_dated_matches_numeric() {
        // §9.4 byte-identity: dated cells against an origin produce the same
        // observation vector as the equivalent numeric day-numbers.
        let dated = write_temp_tsv(
            "dated",
            "time\tcases\n2020-03-01\t10\n2020-03-08\t20\n2020-03-15\t30\n",
        );
        let numeric = write_temp_tsv("dated_num", "time\tcases\n0\t10\n7\t20\n14\t30\n");
        let mut o = numeric_opts();
        o.origin = Some("2020-03-01");
        let from_dates = load_data_tsv_column(&dated, "cases", &o).unwrap();
        let from_nums = load_data_tsv_column(&numeric, "cases", &numeric_opts()).unwrap();
        assert_eq!(from_dates, from_nums);
        std::fs::remove_file(&dated).ok();
        std::fs::remove_file(&numeric).ok();
    }
}
