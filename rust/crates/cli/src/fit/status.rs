//! `camdl fit status` — colored summary of fit progress and convergence.
//!
//! This module's `run_status` is the pre-2026-04-19 v1 status printer.
//! `cmd_fit_status` now routes both v1 and v2 through
//! `run_status_v2_dir` in `mod.rs`, which walks the unified output
//! tree. `run_status` is kept for possible future v1-specific
//! reporting but has no live callers today. Module allows dead code
//! to reflect that.

#![allow(dead_code)]

use crate::fit::config::FitToml;
use crate::fit::state::FitState;
use crate::fit::provenance;
use crate::version;

pub fn run_status(fit: &FitToml) -> Result<(), String> {
    let dir = &fit.fit.output_dir;
    println!("{}/ — {}", dir, fit.fit.model);
    println!();

    // Check each stage
    let scout_state = check_stage(dir, "scout");
    let refine_state = check_stage(dir, "refine");
    let validate_state = check_stage(dir, "validate");
    let pmmh_state = check_stage(dir, "pmmh");

    // Scout
    match &scout_state {
        Some(state) => {
            let n_good = state.n_good_chains.unwrap_or(state.n_chains);
            println!("  scout:     \x1b[32m✓\x1b[0m complete ({} chains, best loglik {:.1}, {}/{} good)",
                state.n_chains, state.best_loglik, n_good, state.n_chains);
            print_stale_warning(state, "scout");
        }
        None => println!("  scout:     \x1b[90m○ not started\x1b[0m"),
    }

    // Refine
    match &refine_state {
        Some(state) => {
            let summary = load_summary(dir, "refine");
            let converged = summary.as_ref()
                .and_then(|s| s.get("converged"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let symbol = if converged { "\x1b[32m✓\x1b[0m" } else { "\x1b[33m~\x1b[0m" };
            let rhat_str = if converged { "Rhat < 1.05" } else { "Rhat > 1.1" };
            println!("  refine:    {} complete ({} chains, {}, loglik {:.1})",
                symbol, state.n_chains, rhat_str, state.best_loglik);
            print_stale_warning(state, "refine");
        }
        None => {
            if scout_state.is_some() {
                println!("  refine:    \x1b[90m○ not started\x1b[0m");
            } else {
                println!("  refine:    \x1b[90m— waiting for scout\x1b[0m");
            }
        }
    }

    // Validate
    match &validate_state {
        Some(state) => {
            let summary = load_summary(dir, "validate");
            let loglik_sd = summary.as_ref()
                .and_then(|s| s.get("loglik_sd"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let ess = summary.as_ref()
                .and_then(|s| s.get("ess_at_mle"))
                .and_then(|v| v.get("min"))
                .and_then(|v| v.as_f64());
            let ess_str = ess.map_or("".into(), |e| format!(", ESS min={:.0}", e));
            println!("  validate:  \x1b[32m✓\x1b[0m complete (loglik {:.1} ± {:.1}{})",
                state.best_loglik, loglik_sd, ess_str);
            print_stale_warning(state, "validate");
        }
        None => {
            if refine_state.is_some() {
                println!("  validate:  \x1b[90m○ not started\x1b[0m");
            } else {
                println!("  validate:  \x1b[90m— waiting for refine\x1b[0m");
            }
        }
    }

    // PMMH
    match &pmmh_state {
        Some(state) => {
            let summary = load_summary(dir, "pmmh");
            let acc_rate = summary.as_ref()
                .and_then(|s| s.get("acceptance_rate"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    let rates: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                    rates.iter().sum::<f64>() / rates.len() as f64
                });
            let acc_str = acc_rate.map_or("".into(), |r| format!(", accept={:.0}%", r * 100.0));
            println!("  pmmh:      \x1b[32m✓\x1b[0m complete ({} chains, loglik {:.1}{})",
                state.n_chains, state.best_loglik, acc_str);
            print_stale_warning(state, "pmmh");
        }
        None => {
            if validate_state.is_some() {
                println!("  pmmh:      \x1b[90m○ not started (optional — posterior sampling)\x1b[0m");
            }
        }
    }

    println!();

    // ESS at MLE (if validate is done)
    if let Some(ref summary) = load_summary(dir, "validate") {
        if let Some(ess) = summary.get("ess_at_mle") {
            let mean = ess.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let min = ess.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
            // Thresholds relative to particle count (from validate summary or default 10000)
            let n_particles = summary.get("n_particles")
                .and_then(|v| v.as_f64()).unwrap_or(10000.0);
            let status = if min > n_particles / 4.0 { "\x1b[32m✓ filter is healthy\x1b[0m" }
                else if min > n_particles / 10.0 { "\x1b[33m~ filter is marginal\x1b[0m" }
                else { "\x1b[31m✗ filter is degenerate\x1b[0m" };
            println!("  ESS at MLE: mean={:.0}, min={:.0}  {}", mean, min, status);
            println!();
        }
    }

    // Estimated parameters
    let best_state = validate_state.as_ref()
        .or(refine_state.as_ref())
        .or(scout_state.as_ref());

    if let Some(state) = best_state {
        // Load profile data for CIs if available
        let profiles = load_profiles(dir);
        let summary = load_summary(dir, &state.stage);

        println!("  Estimated ({} parameters):", fit.estimate.len());
        for (name, spec) in &fit.estimate {
            let value = state.start_values.get(name).copied().unwrap_or(0.0);
            let rw = state.rw_sd.get(name).copied().or(spec.rw_sd).unwrap_or(0.0);

            // Get Rhat from summary
            let rhat = summary.as_ref()
                .and_then(|s| s.get("parameters"))
                .and_then(|p| p.as_array())
                .and_then(|arr| arr.iter().find(|p| p.get("name").and_then(|n| n.as_str()) == Some(name)))
                .and_then(|p| p.get("rhat"))
                .and_then(|v| v.as_f64());

            // Get CI from profiles
            let ci = profiles.get(name);

            let mut line = format!("    {:12} = {:<12.6} rw_sd={:<8.4}", name, value, rw);

            if let Some(&r) = rhat.as_ref() {
                if r < 1.1 {
                    line.push_str(&format!(" \x1b[32m✓\x1b[0m identified"));
                } else {
                    line.push_str(&format!(" \x1b[33m~ Rhat={:.2}\x1b[0m", r));
                }
            }

            if let Some((lo, hi)) = ci {
                line.push_str(&format!("  CI: [{:.4}, {:.4}]", lo, hi));
            }

            if spec.ivp {
                line.push_str("  (ivp)");
            }

            // Boundary proximity warning
            if let Some(bounds) = spec.bounds {
                let range = bounds.1 - bounds.0;
                if (value - bounds.0).abs() < range * 0.01 {
                    line.push_str("  \x1b[33m⚠ AT LOWER BOUND\x1b[0m");
                } else if (value - bounds.1).abs() < range * 0.01 {
                    line.push_str("  \x1b[33m⚠ AT UPPER BOUND\x1b[0m");
                }
            }

            println!("{}", line);
        }

        println!();
        println!("  Fixed ({} parameters):", fit.fixed.len());
        for name in fit.fixed.keys() {
            if let Some(&value) = state.start_values.get(name) {
                println!("    {:12} = {}", name, value);
            } else {
                println!("    {}", name);
            }
        }
    }

    // Provenance check — verify primary output files against fit_record.json manifest
    let record_path = format!("{}/validate/fit_record.json", dir);
    let mle_path = format!("{}/validate/mle_params.toml", dir);
    if std::path::Path::new(&record_path).exists() {
        println!();
        println!("  Provenance:");

        // Read output manifest from fit_record.json
        if let Ok(record_str) = std::fs::read_to_string(&record_path) {
            if let Ok(record) = serde_json::from_str::<serde_json::Value>(&record_str) {
                if let Some(ref input_hash) = record.get("provenance").and_then(|p| p.get("input_hash")).and_then(|v| v.as_str()) {
                    println!("    input hash: {}", input_hash);
                }

                if let Some(outputs) = record.get("outputs").and_then(|o| o.as_object()) {
                    let validate_dir = format!("{}/validate", dir);
                    // Verify primary outputs only (not chain-level)
                    let primary = ["mle_params.toml", "pfilter_trace.tsv", "ess_at_mle.tsv", "fit_state.toml"];
                    for name in &primary {
                        if let Some(expected) = outputs.get(*name).and_then(|v| v.as_str()) {
                            let file_path = format!("{}/{}", validate_dir, name);
                            match crate::hashing::file_hash(&file_path) {
                                Some(actual) if actual == expected => {
                                    println!("    {}: \x1b[32m✓\x1b[0m {}", name, expected);
                                }
                                Some(actual) => {
                                    println!("    {}: \x1b[33m⚠ MODIFIED\x1b[0m (expected {}, got {})", name, expected, actual);
                                }
                                None => {
                                    println!("    {}: \x1b[31m✗ DELETED\x1b[0m", name);
                                }
                            }
                        }
                    }

                    // Profile files
                    for (name, hash_val) in outputs {
                        if name.starts_with("profiles/") {
                            if let Some(expected) = hash_val.as_str() {
                                let file_path = format!("{}/{}", validate_dir, name);
                                match crate::hashing::file_hash(&file_path) {
                                    Some(actual) if actual == expected => {
                                        println!("    {}: \x1b[32m✓\x1b[0m", name);
                                    }
                                    Some(_) => {
                                        println!("    {}: \x1b[33m⚠ MODIFIED\x1b[0m", name);
                                    }
                                    None => {
                                        println!("    {}: \x1b[31m✗ DELETED\x1b[0m", name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if std::path::Path::new(&mle_path).exists() {
        // Fallback: standalone mle_params.toml content hash check
        println!();
        println!("  Provenance:");
        match provenance::verify_content_hash(&mle_path) {
            Ok(provenance::ContentVerification::Valid) => {
                println!("    mle_params.toml: \x1b[32m✓ content hash matches\x1b[0m");
            }
            Ok(provenance::ContentVerification::Modified { declared, computed }) => {
                println!("    mle_params.toml: \x1b[33m⚠ MODIFIED\x1b[0m (expected {}, got {})", declared, computed);
            }
            Ok(provenance::ContentVerification::NoHash) => {
                println!("    mle_params.toml: \x1b[90mno provenance hash\x1b[0m");
            }
            Err(e) => {
                println!("    mle_params.toml: error: {}", e);
            }
        }
    }

    // Next step suggestion
    println!();
    if validate_state.is_some() {
        println!("  Next: camdl simulate batch sweep.toml \\");
        println!("          --params {}/validate/mle_params.toml", dir);
    } else if refine_state.is_some() {
        println!("  Next: camdl fit validate fit.toml --starts-from {}/refine/", dir);
    } else if scout_state.is_some() {
        println!("  Next: camdl fit refine fit.toml --starts-from {}/scout/", dir);
    } else {
        println!("  Next: camdl fit scout fit.toml");
    }

    Ok(())
}

fn check_stage(base_dir: &str, stage: &str) -> Option<FitState> {
    FitState::load(&format!("{}/{}", base_dir, stage)).ok()
}

fn load_summary(base_dir: &str, stage: &str) -> Option<serde_json::Value> {
    let path = format!("{}/{}/{}_summary.json", base_dir, stage, stage);
    std::fs::read_to_string(&path).ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

fn print_stale_warning(state: &FitState, stage: &str) {
    match &state.camdl_version {
        Some(v) if v != version::VERSION_SHORT => {
            println!("             \x1b[33m⚠ stale — produced by {}, current is {}\x1b[0m",
                v, version::VERSION_SHORT);
        }
        None => {
            println!("             \x1b[90m(no version recorded for {})\x1b[0m", stage);
        }
        _ => {} // matches current version
    }
}

fn load_profiles(base_dir: &str) -> std::collections::HashMap<String, (f64, f64)> {
    let mut profiles = std::collections::HashMap::new();
    if let Some(summary) = load_summary(base_dir, "validate") {
        if let Some(params) = summary.get("parameters").and_then(|p| p.as_array()) {
            for p in params {
                if let (Some(name), Some(ci)) = (
                    p.get("name").and_then(|n| n.as_str()),
                    p.get("ci_95").and_then(|c| c.as_array()),
                ) {
                    if ci.len() == 2 {
                        if let (Some(lo), Some(hi)) = (ci[0].as_f64(), ci[1].as_f64()) {
                            profiles.insert(name.to_string(), (lo, hi));
                        }
                    }
                }
            }
        }
    }
    profiles
}
