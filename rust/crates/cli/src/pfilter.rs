//! `camdl pfilter` — bootstrap particle filter for log-likelihood estimation.
//!
//! Usage:
//!   camdl pfilter MODEL --params P.toml --data cases.tsv \
//!       --particles 5000 --dt 1.0 --seed 1
//!
//! Output: log-likelihood estimate to stdout.
//! With --trace: per-observation TSV (time, ll_increment, ESS).

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

pub fn cmd_pfilter(args: &[String]) {
    let mut ir_path: Option<String> = None;
    let mut params_files: Vec<String> = Vec::new();
    let mut data_path: Option<String> = None;
    let mut n_particles = 1000_usize;
    let mut dt = 1.0_f64;
    let mut seed = 1_u64;
    let mut trace_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut overrides: HashMap<String, f64> = HashMap::new();
    let mut scenario_name: Option<String> = None;
    let mut adhoc_enable: Vec<String> = Vec::new();
    let mut adhoc_disable: Vec<String> = Vec::new();
    let mut flow_name: Option<String> = None; // --flow recovery → project that transition
    let mut obs_name: Option<String> = None; // --obs NAME → select observation block
    let mut save_final_state: Option<String> = None;
    let mut n_replicates = 1_usize;
    let mut save_paths: Option<(usize, String)> = None;
    let mut save_filtering: Option<String> = None;
    let mut save_prequential: Option<String> = None;
    let mut save_samples: bool = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--params"    => { i += 1; params_files.push(args[i].clone()); }
            "--data"      => { i += 1; data_path = Some(args[i].clone()); }
            "--replicates" => { i += 1; n_replicates = args[i].parse().unwrap_or_else(|_| { eprintln!("error: --replicates needs integer"); std::process::exit(1); }); }
            "--particles" => { i += 1; n_particles = args[i].parse().unwrap_or_else(|_| { eprintln!("error: --particles needs an integer"); std::process::exit(1); }); }
            "--dt"        => { i += 1; dt = args[i].parse().unwrap_or_else(|_| { eprintln!("error: --dt needs a number"); std::process::exit(1); }); }
            "--seed"      => { i += 1; seed = args[i].parse().unwrap_or_else(|_| { eprintln!("error: --seed needs an integer"); std::process::exit(1); }); }
            "--trace"     => {
                // Accept both: --trace FILE and bare --trace (stdout)
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    i += 1;
                    trace_path = Some(args[i].clone());
                } else {
                    trace_path = Some("-".to_string()); // sentinel for stdout
                }
            }
            "--output" | "-o" => { i += 1; output_path = Some(args[i].clone()); }
            "--scenario"  => { i += 1; scenario_name = Some(args[i].clone()); }
            "--enable"    => { i += 1; adhoc_enable.push(args[i].clone()); }
            "--disable"   => { i += 1; adhoc_disable.push(args[i].clone()); }
            "--obs"       => { i += 1; obs_name = Some(args[i].clone()); }
            "--flow"      => { i += 1; flow_name = Some(args[i].clone()); }
            "--save-final-state" => { i += 1; save_final_state = Some(args[i].clone()); }
            "--save-paths" => {
                // Two-argument form: --save-paths N PATH.TSV. N is
                // the number of trajectory samples from the smoothing
                // distribution; PATH is where to write them.
                i += 1;
                let n: usize = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("error: --save-paths needs an integer count (got '{}')", args[i]);
                    std::process::exit(1);
                });
                i += 1;
                let path = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("error: --save-paths needs both N and a PATH");
                    std::process::exit(1);
                });
                save_paths = Some((n, path));
            }
            "--save-filtering" => { i += 1; save_filtering = Some(args[i].clone()); }
            "--save-prequential" => { i += 1; save_prequential = Some(args[i].clone()); }
            "--no-save-samples"  => { save_samples = false; }
            "--param"     => {
                i += 1;
                let kv = &args[i];
                let mut parts = kv.splitn(2, '=');
                let k = parts.next().unwrap().to_string();
                let v: f64 = parts.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| { eprintln!("error: --param needs NAME=VALUE"); std::process::exit(1); });
                overrides.insert(k, v);
            }
            s if s.starts_with("--") => {
                eprintln!("unknown flag: {}", s);
                eprintln!();
                eprintln!("usage: camdl pfilter MODEL --params P.toml --data cases.tsv \\");
                eprintln!("         --particles 5000 --dt 1.0 --seed 1");
                eprintln!();
                eprintln!("Latent-trajectory outputs (see docs/dev/proposals/2026-04-19-pf-latent-trajectories.md):");
                eprintln!("  --save-paths N PATH      Draw N trajectory samples from the smoothing");
                eprintln!("                           distribution (ancestor tracing). For model-vs-data");
                eprintln!("                           plots, this is what you want.");
                eprintln!("  --save-prequential STEM  Write {{STEM}}.tsv + {{STEM}}.json: per-step log");
                eprintln!("                           score, CRPS, PIT, ESS (one-step-ahead plug-in");
                eprintln!("                           predictive). See 2026-04-20-prequential-evaluation.md.");
                eprintln!("  --no-save-samples        With --save-prequential, drop per-particle");
                eprintln!("                           predictive samples from {{STEM}}.json (keeps");
                eprintln!("                           scalar scores; shrinks file).");
                eprintln!("  --save-filtering PATH    Dump per-step particle states + weights. For PF");
                eprintln!("                           diagnostics (particle degeneracy, obs sanity,");
                eprintln!("                           implementation debugging). NOT a substitute for");
                eprintln!("                           --save-paths when plotting against data.");
                std::process::exit(1);
            }
            path => { ir_path = Some(path.to_string()); }
        }
        i += 1;
    }

    let ir_path = ir_path.unwrap_or_else(|| {
        eprintln!("usage: camdl pfilter MODEL --params P.toml --data cases.tsv --particles 5000");
        std::process::exit(1);
    });
    let data_path = data_path.unwrap_or_else(|| {
        eprintln!("error: --data required");
        std::process::exit(1);
    });

    // Load model (supports .camdl via camdlc)
    let (mut model, _model_json) = crate::util::load_model(&ir_path)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // Apply params
    for pf in &params_files {
        crate::util::apply_params_file(&mut model, pf)
            .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
    }

    // Resolve scenario → enable/disable + preset params; fall through
    // to ad-hoc lists otherwise. Mutually exclusive per spec §18.
    let (enable_list, disable_list) = if let Some(ref name) = scenario_name {
        let preset = model.presets.iter().find(|p| p.name == *name).cloned()
            .unwrap_or_else(|| {
                eprintln!("error: scenario '{}' not found", name);
                std::process::exit(1);
            });
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) { p.value = Some(v); }
        }
        (preset.enable, preset.disable)
    } else {
        (adhoc_enable, adhoc_disable)
    };

    // Single shared filter: events stay on unless explicitly disabled;
    // toggleable interventions stay off unless enabled (matches §14.4).
    crate::util::apply_scenario_filter(&mut model, &enable_list, &disable_list)
        .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });

    // Apply overrides
    for p in &mut model.parameters {
        if let Some(&v) = overrides.get(&p.name) { p.value = Some(v); }
    }

    let compiled = CompiledModel::new(model.clone())
        .unwrap_or_else(|e| { eprintln!("compile error: {:?}", e); std::process::exit(1); });
    let params = compiled.default_params.clone();

    // Load data
    let observations = load_data_tsv(&data_path)
        .unwrap_or_else(|e| { eprintln!("error loading data: {}", e); std::process::exit(1); });

    eprintln!("pfilter: {} observations, {} particles, dt={}, seed={}",
        observations.len(), n_particles, dt, seed);

    // Find observation model from the IR.
    // Im22 in 2026-04-19 inference review batch 3: pfilter is single-
    // stream only — the runtime's MultiStreamObsModel supports
    // joint observation across multiple streams, but this CLI
    // driver scores exactly one `[observations.NAME]` block per
    // invocation. Use `camdl fit run` (PGAS/PMMH stages) for
    // multi-stream joint inference.
    let obs_model_ir = if let Some(ref name) = obs_name {
        model.observations.iter().find(|o| o.name == *name)
            .cloned()
            .unwrap_or_else(|| {
                eprintln!("error: no observation block '{}'. Available: {}",
                    name, model.observations.iter().map(|o| o.name.as_str()).collect::<Vec<_>>().join(", "));
                std::process::exit(1);
            })
    } else if model.observations.len() == 1 {
        model.observations[0].clone()
    } else if !model.observations.is_empty() {
        eprintln!("error: model has {} observation blocks. Use --obs NAME to select one:", model.observations.len());
        for o in &model.observations { eprintln!("  {}", o.name); }
        std::process::exit(1);
    } else {
        eprintln!("error: model has no observations block. Cannot run pfilter without an observation model.");
        std::process::exit(1);
    };

    // Build the projection. An explicit `--flow NAME` overrides the obs
    // model's projection (forces incidence over the named transition);
    // otherwise the projection comes from the obs model's `projection:`
    // field — incidence, prevalence, or a DerivedExpr snapshot.
    let projection: sim::inference::multi_stream_obs::StreamProjection =
        if let Some(ref name) = flow_name {
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
            sim::inference::multi_stream_obs::StreamProjection::FlowSum(indices)
        } else {
            sim::inference::multi_stream_obs::StreamProjection::from_ir(
                &obs_model_ir.projection, &compiled, &obs_model_ir.name,
            ).unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); })
        };

    eprintln!("pfilter: obs_model={}, likelihood={}", obs_model_ir.name,
        match &obs_model_ir.likelihood {
            ir::observation::Likelihood::NegBinomial(_) => "neg_binomial",
            ir::observation::Likelihood::Normal(_) => "normal",
            ir::observation::Likelihood::Poisson(_) => "poisson",
            ir::observation::Likelihood::Binomial(_) => "binomial",
            ir::observation::Likelihood::BetaBinomial(_) => "beta_binomial",
            ir::observation::Likelihood::Bernoulli(_) => "bernoulli",
        });

    // Build process + observation model via traits
    let compiled = std::sync::Arc::new(compiled);
    let process = ChainBinomialProcess::new(compiled.clone());

    let obs_times: Vec<f64> = observations.iter().map(|o| o.time).collect();
    let obs_values: Vec<f64> = observations.iter().map(|o| o.value).collect();
    let obs_model = MultiStreamObsModel::new(
        vec![StreamSpec {
            projection,
            ir_model: obs_model_ir.clone(),
            observations: obs_values,
            obs_times,
        }],
        compiled.clone(),
    ).unwrap_or_else(|e| {
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

    // ── Replicates mode: run N independent pfilters, output loglik summary ──
    if n_replicates > 1 {
        eprintln!("pfilter: {} replicates × {} particles", n_replicates, n_particles);
        // Im20 in 2026-04-19 inference review batch 3: replicate
        // seeding was `seed + rep`, which gives highly correlated
        // ChaCha8 initial states across replicates. Use the
        // golden-ratio multiplier to decorrelate low bits.
        const SEED_STRIDE: u64 = 0x9e3779b97f4a7c15;
        let mut logliks = Vec::with_capacity(n_replicates);
        for rep in 0..n_replicates {
            let rep_seed = seed.wrapping_add((rep as u64).wrapping_mul(SEED_STRIDE));
            let result = bootstrap_filter(
                &process, &obs_model, &params, &smc_config, rep_seed,
            ).unwrap_or_else(|e| {
                eprintln!("pfilter replicate {} error: {:?}", rep + 1, e);
                std::process::exit(1);
            });
            logliks.push(result.log_likelihood);
            if (rep + 1) % 10 == 0 || rep + 1 == n_replicates {
                eprint!("\r  {}/{} replicates", rep + 1, n_replicates);
            }
        }
        eprintln!();

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

    // Save prequential trace (--save-prequential PATH): writes
    // {PATH}.tsv (per-step scalar scores) + {PATH}.json (full trace,
    // incl. predictive samples unless --no-save-samples was given).
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
}

/// Write a `PrequentialTrace` to `{stem}.tsv` + `{stem}.json`.
/// The `.tsv` is a human-readable per-step table of scalar scores;
/// `.json` carries the full typed trace including predictive samples
/// (so downstream tools needn't re-run the filter).
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
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&json_path, json)?;
    Ok(())
}

/// Load observation data from a TSV file.
/// Expected columns: time, then one or more value columns.
/// Uses the first value column.
pub fn load_data_tsv_pub(path: &str) -> Result<Vec<Observation>, String> {
    load_data_tsv(path)
}

/// Load observations from a specific column in a TSV file.
/// The column name must match a header field. First column is always time.
pub fn load_data_tsv_column(path: &str, column: &str) -> Result<Vec<Observation>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let mut lines = content.lines();
    let header = lines.next().ok_or("empty data file")?;
    let cols: Vec<&str> = header.split('\t').collect();

    // Find the column index for the requested stream name
    let col_idx = cols.iter().position(|&c| c == column)
        .or_else(|| {
            // Fallback: if only 2 columns (time + value), use column 1
            if cols.len() == 2 { Some(1) } else { None }
        })
        .ok_or_else(|| format!(
            "column '{}' not found in data file '{}'. Available columns: {:?}",
            column, path, &cols[1..]))?;

    let mut observations = Vec::new();
    for (line_num, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() <= col_idx {
            return Err(format!("line {}: expected {}+ columns, got {}",
                line_num + 2, col_idx + 1, fields.len()));
        }
        let time: f64 = fields[0].trim().parse()
            .map_err(|_| format!("line {}: cannot parse time '{}'", line_num + 2, fields[0]))?;
        let value: f64 = fields[col_idx].trim().parse()
            .map_err(|_| format!("line {}: cannot parse value '{}' in column '{}'",
                line_num + 2, fields[col_idx], column))?;
        observations.push(Observation { time, value });
    }

    for i in 1..observations.len() {
        if observations[i].time < observations[i - 1].time {
            return Err(format!(
                "observations not in chronological order: t={} at row {} follows t={} at row {}",
                observations[i].time, i + 2, observations[i - 1].time, i + 1));
        }
    }

    Ok(observations)
}

fn load_data_tsv(path: &str) -> Result<Vec<Observation>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let mut lines = content.lines();
    let header = lines.next().ok_or("empty data file")?;
    let cols: Vec<&str> = header.split('\t').collect();
    if cols.len() < 2 {
        return Err(format!("data file needs at least 2 columns (time, value), got {}", cols.len()));
    }

    let mut observations = Vec::new();
    for (line_num, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            return Err(format!("line {}: expected 2+ columns, got {}", line_num + 2, fields.len()));
        }
        let time: f64 = fields[0].trim().parse()
            .map_err(|_| format!("line {}: cannot parse time '{}'", line_num + 2, fields[0]))?;
        let value: f64 = fields[1].trim().parse()
            .map_err(|_| format!("line {}: cannot parse value '{}'", line_num + 2, fields[1]))?;
        observations.push(Observation { time, value });
    }

    // Validate chronological ordering (equal times OK — multi-stream observations)
    for i in 1..observations.len() {
        if observations[i].time < observations[i - 1].time {
            return Err(format!(
                "observations not in chronological order: t={} at row {} follows t={} at row {}",
                observations[i].time, i + 2, observations[i - 1].time, i + 1
            ));
        }
    }

    Ok(observations)
}

use std::io::Write;

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

    #[test]
    fn load_data_rejects_out_of_order() {
        let path = write_temp_tsv("out_of_order", "time\tcases\n7\t10\n14\t20\n10\t15\n21\t30\n");
        let result = load_data_tsv(&path);
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
        let result = load_data_tsv(&path);
        assert!(result.is_ok(), "equal times should be accepted: {:?}", result.err());
        let obs = result.unwrap();
        assert_eq!(obs.len(), 3);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_data_accepts_sorted() {
        let path = write_temp_tsv("sorted", "time\tcases\n7\t10\n14\t20\n21\t30\n");
        let result = load_data_tsv(&path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
        std::fs::remove_file(&path).ok();
    }
}
