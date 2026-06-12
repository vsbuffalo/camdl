//! Reproduction test: does log_transition_density_substep return -inf
//! on a trajectory simulated by step_one at the SAME params?
//!
//! If this fails, the density function doesn't mirror step_one.

use std::sync::Arc;
use sim::compiled_model::CompiledModel;
use sim::inference::pgas::{simulate_reference, complete_data_loglik, log_transition_density_substep, build_obs_at_substep};
use sim::inference::MultiStreamObsModel;
use sim::inference::pgas::IVPMapping;
use sim::inference::particle_filter::Observation;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json)  // gh#audit-C8
        .unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

/// Test: complete_data_loglik on the SIR basic model (single-patch, 2 transitions)
/// should be finite at its own params.
#[test]
fn test_density_matches_step_one_sir() {
    let model = load_model("../../../ocaml/golden/sir_basic.ir.json");
    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.4, "gamma" => 0.1, "mu" => 0.01, _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let mut rng = StatefulRng::new(42);
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let observations: Vec<Observation> = vec![];

    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();
    let ll = complete_data_loglik(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings, &oas,
    ).unwrap().total;

    eprintln!("  SIR basic: complete-data LL = {:.4} ({} substeps, {} transitions, {} groups)",
        ll, trajectory.substeps.len(), compiled.model.transitions.len(), compiled.source_groups.len());
    assert!(ll.is_finite(), "SIR basic: LL should be finite at own params, got {}", ll);
}

/// Test: complete_data_loglik on the SIR demography model (3 transitions per S group)
/// should be finite at its own params.
#[test]
fn test_density_matches_step_one_sir_demography() {
    // gh#audit-C6 / S1: model contains rate expressions with stratum-
    // empty divisors (legacy silent-zero); test asserts density-vs-
    // step_one parity, not numerical-collapse semantics — keep
    // legacy mode here. expr_eval.rs::test_*_errors_by_default
    // covers the strict-mode contract.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let model = load_model("../../../ocaml/golden/sir_demography.ir.json");
    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "beta" => 0.4, "gamma" => 0.1, "mu" => 0.02, _ => 0.5,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    let mut rng = StatefulRng::new(42);
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let observations: Vec<Observation> = vec![];

    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();
    let ll = complete_data_loglik(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings, &oas,
    ).unwrap().total;

    eprintln!("  SIR demography: complete-data LL = {:.4} ({} substeps, {} transitions, {} groups)",
        ll, trajectory.substeps.len(), compiled.model.transitions.len(), compiled.source_groups.len());
    assert!(ll.is_finite(), "SIR demography: LL should be finite at own params, got {}", ll);
}

/// Test: complete_data_loglik on a 2-patch model.
#[test]
fn test_density_matches_step_one_two_patch() {
    let path = "../../../ocaml/golden/sir_two_patch.ir.json";
    let model = match std::fs::read_to_string(path) {
        Ok(json) => match ir::from_str(&json) {
            Ok(m) => m, Err(e) => { eprintln!("  skip: {}", e); return; }
        },
        Err(_) => { eprintln!("  skip: not found"); return; }
    };
    let mut model = model;
    for p in &mut model.parameters { if p.value.resolved_value().is_none() { p.value = p.value.with_value(0.1); } }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() { params[compiled.param_index[p.name.as_str()]] = v; }
    }
    let mut rng = StatefulRng::new(42);
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    let empty_obs: Vec<Observation> = vec![];
    let oas = build_obs_at_substep(&empty_obs, compiled.model.simulation.t_start, dt).unwrap();
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let ll = complete_data_loglik(
        &compiled, &trajectory, &params, &empty_obs, dt, &obs_model, &[], &oas,
    ).unwrap().total;
    eprintln!("  two_patch: LL={:.4} ({} substeps, {} tr, {} groups)",
        ll, trajectory.substeps.len(), compiled.model.transitions.len(), compiled.source_groups.len());
    assert!(ll.is_finite(), "two_patch LL should be finite, got {}", ll);
}

/// Test: complete_data_loglik on polio_spatial_5 (5 patches, 5 transitions per S group)
/// This is the exact pattern that caused -inf when testing camdl-vignettes.
#[test]
fn test_density_matches_step_one_polio_spatial_5() {
    // gh#audit-C6 / S1: spatial model with empty patches; legacy
    // silent-zero needed for parity assertion. See companion comment
    // on test_density_matches_step_one_sir_demography.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let path = "../../../ocaml/golden/polio_spatial_5.ir.json";
    let model = match std::fs::read_to_string(path) {
        Ok(json) => match ir::from_str(&json) {
            Ok(m) => m,
            Err(e) => { eprintln!("  skipping: cannot parse {}: {}", path, e); return; }
        },
        Err(_) => { eprintln!("  skipping: {} not found", path); return; }
    };

    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(0.1); // default for any missing param
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    eprintln!("  spatial model: {} transitions, {} source groups",
        compiled.model.transitions.len(), compiled.source_groups.len());
    for (i, (src, group)) in compiled.source_groups.iter().enumerate() {
        eprintln!("    group {}: src={}, {} transitions", i, src, group.len());
    }

    let mut rng = StatefulRng::new(42);
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let ivp_mappings: Vec<IVPMapping> = vec![];
    let observations: Vec<Observation> = vec![];

    let oas = build_obs_at_substep(&observations, compiled.model.simulation.t_start, dt).unwrap();
    let ll = complete_data_loglik(
        &compiled, &trajectory, &params, &observations, dt,
        &obs_model, &ivp_mappings, &oas,
    ).unwrap().total;

    eprintln!("  spatial: complete-data LL = {:.4} ({} substeps)",
        ll, trajectory.substeps.len());
    assert!(ll.is_finite(),
        "spatial: LL should be finite at own params, got {}. \
         Run with CAMDL_TRACE_STEPS=1 for diagnostics.", ll);
}

/// Test: SEIR spatial 5-patch model (reproduced when testing
/// camdl-vignettes). The model has waning immunity (R→S),
/// seasonal forcing, and used to produce -inf in PGAS runs. If
/// this test fails, the underlying bug has returned.
#[test]
fn test_density_seir_spatial_5_vignette_regression() {
    // Uses the golden model with iota, overdispersion, seasonal forcing,
    // and high R0 — the exact combination that triggers spatial density bugs.
    // See docs/dev/incidents/2026-04-07-spatial-pgas-neg-inf.md.
    let path = "../../../ocaml/golden/seir_spatial_5_inference.ir.json";
    let model = load_model(path);

    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "R0" => 20.0, "sigma" => 0.125, "gamma" => 0.2,
                "amplitude" => 0.3, "kappa" => 0.05, "iota" => 1e-6,
                "rho" => 0.4, "sigma_se" => 0.05, "k" => 10.0,
                "N0_p1" => 100000.0, "N0_p2" => 80000.0,
                "N0_p3" => 60000.0, "N0_p4" => 50000.0,
                "N0_p5" => 150000.0,
                _ => 1.0,
            });
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }

    eprintln!("  downstream SEIR spatial 5: {} transitions, {} source groups",
        compiled.model.transitions.len(), compiled.source_groups.len());
    for (i, (src, group)) in compiled.source_groups.iter().enumerate() {
        let names: Vec<&str> = group.iter()
            .map(|&j| compiled.model.transitions[j].name.as_str()).collect();
        if group.len() > 1 {
            eprintln!("    group {}: src={}, {} tr: {:?}", i, src, group.len(), names);
        }
    }

    let mut rng = StatefulRng::new(42);
    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();
    eprintln!("  {} substeps", trajectory.substeps.len());

    // Check EACH substep individually to find the first -inf
    let t_start = compiled.model.simulation.t_start;
    for s in 0..trajectory.substeps.len() {
        let t = t_start + s as f64 * dt;
        let rec = &trajectory.substeps[s];
        let counts_before = &rec.counts_before;

        let td = log_transition_density_substep(
            &compiled, counts_before, &rec.flows, &rec.gammas, &params, t, dt,
        ).unwrap();

        if !td.is_finite() {
            eprintln!("\n  FIRST -inf at substep {} (t={:.1}):", s, t);
            eprintln!("  counts_before ({} compartments): {:?}", counts_before.len(), counts_before);
            eprintln!("  flows ({} transitions): {:?}", rec.flows.len(), &rec.flows);
            eprintln!("  gammas: {:?}", &rec.gammas);

            // Evaluate propensities to find the mismatch
            let mut propensities = vec![0.0; compiled.model.transitions.len()];
            let int_s = sim::state::IntState { counts: counts_before.to_vec() };
            let real_s = sim::state::RealState::new(compiled.real_local_to_global.len());
            sim::propensity::eval_propensities(
                &compiled, &int_s, &real_s, &params, t, 1.0, &mut propensities
            ).unwrap();

            for &(src_local, ref group) in &compiled.source_groups {
                for &tr_idx in group {
                    let rate = propensities[tr_idx];
                    let flow = rec.flows[tr_idx];
                    if (rate <= 0.0 && flow > 0) || (flow > 0) {
                        eprintln!("    {} (idx={}): rate={:.6e}, flow={}, src_count={}",
                            compiled.model.transitions[tr_idx].name, tr_idx,
                            rate, flow, counts_before[src_local]);
                    }
                }
            }

            panic!("density -inf at substep {} — see diagnostics above", s);
        }
    }

    let empty_obs: Vec<Observation> = vec![];
    let oas = build_obs_at_substep(&empty_obs, compiled.model.simulation.t_start, dt).unwrap();
    let obs_model = MultiStreamObsModel::empty(compiled.clone());
    let ll = complete_data_loglik(
        &compiled, &trajectory, &params, &empty_obs, dt, &obs_model, &[], &oas,
    ).unwrap().total;
    eprintln!("  complete-data LL = {:.4}", ll);
    assert!(ll.is_finite(), "LL should be finite, got {}", ll);
}

/// Test: multi-seed round-trip on downstream SEIR spatial model.
/// Runs 100 different seeds to catch rare stochastic edge cases.
#[test]
fn test_density_downstream_multi_seed() {
    let path = "../../../ocaml/golden/seir_spatial_5_inference.ir.json";
    let model = load_model(path);
    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "R0" => 20.0, "sigma" => 0.125, "gamma" => 0.2,
                "amplitude" => 0.3, "kappa" => 0.05, "iota" => 1e-6,
                "rho" => 0.4, "sigma_se" => 0.05, "k" => 10.0,
                "N0_p1" => 100000.0, "N0_p2" => 80000.0,
                "N0_p3" => 60000.0, "N0_p4" => 50000.0,
                "N0_p5" => 150000.0,
                _ => 1.0,
            });
        }
    }
    let compiled = CompiledModel::new(model).unwrap();
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() { params[compiled.param_index[p.name.as_str()]] = v; }
    }

    let dt = compiled.model.simulation.dt.unwrap_or(1.0);
    let t_end = compiled.model.simulation.t_end;
    let t_start = compiled.model.simulation.t_start;
    let mut n_inf = 0;

    for seed in 0..100 {
        let mut rng = StatefulRng::new(seed);
        let trajectory = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

        // Check per-substep
        let mut _this_inf = false;
        for s in 0..trajectory.substeps.len() {
            let t = t_start + s as f64 * dt;
            let rec = &trajectory.substeps[s];
            let counts_before = &rec.counts_before;
            let td = log_transition_density_substep(
                &compiled, counts_before, &rec.flows, &rec.gammas, &params, t, dt,
            ).unwrap();
            if !td.is_finite() {
                if n_inf == 0 {
                    // Print diagnostic for FIRST failure
                    eprintln!("\n  FIRST -inf at seed={}, substep {} (t={:.1}):", seed, s, t);

                    let mut propensities = vec![0.0; compiled.model.transitions.len()];
                    let int_s = sim::state::IntState { counts: counts_before.to_vec() };
                    let real_s = sim::state::RealState::new(compiled.real_local_to_global.len());
                    sim::propensity::eval_propensities(
                        &compiled, &int_s, &real_s, &params, t, 1.0, &mut propensities
                    ).unwrap();

                    for &(src_local, ref group) in &compiled.source_groups {
                        for &tr_idx in group {
                            if rec.flows[tr_idx] > 0 || propensities[tr_idx] <= 0.0 {
                                eprintln!("    {} (idx={}): rate={:.6e}, flow={}, src_count={}",
                                    compiled.model.transitions[tr_idx].name, tr_idx,
                                    propensities[tr_idx], rec.flows[tr_idx],
                                    counts_before[src_local]);
                            }
                        }
                    }
                }
                n_inf += 1;
                _this_inf = true;
                break;
            }
        }
    }

    eprintln!("  multi-seed: {}/100 seeds produced -inf", n_inf);
    assert_eq!(n_inf, 0, "{}/100 seeds produced -inf density at own params", n_inf);
}

/// Targeted test: simulate ONE substep where I_p5=0, check that
/// infection_p5 flow is 0 (not misattributed from importation).
#[test]
fn test_step_one_zero_infection_flow() {
    use sim::chain_binomial::{StepScratch, step_one};

    let path = "tests/fixtures/seir_spatial_5.ir.json";
    let model = match std::fs::read_to_string(path) {
        Ok(json) => ir::from_str(&json).unwrap(),
        Err(_) => { eprintln!("skip"); return; }
    };
    let mut model = model;
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            p.value = p.value.with_value(match p.name.as_str() {
                "R0" => 38.0, "sigma" => 0.055, "gamma" => 0.2,
                "amplitude" => 0.467, "s0" => 0.053, "kappa" => 0.038,
                "rho" => 0.4, "sigma_se" => 0.05, "k" => 10.0,
                "N0_p1" => 100000.0, "N0_p2" => 80000.0,
                "N0_p3" => 60000.0, "N0_p4" => 50000.0,
                "N0_p5" => 150000.0,
                _ => 1.0,
            });
        }
    }
    let compiled = CompiledModel::new(model).unwrap();
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() { params[compiled.param_index[p.name.as_str()]] = v; }
    }

    // Find infection_p5 and I_p5 indices
    let inf_p5_idx = compiled.model.transitions.iter().position(|t| t.name == "infection_p5").unwrap();
    let i_p5_idx = compiled.model.compartments.iter().position(|c| c.name == "I_p5").unwrap();

    eprintln!("  infection_p5 transition idx = {}", inf_p5_idx);
    eprintln!("  I_p5 compartment idx = {}", i_p5_idx);

    // Set up initial state with I_p5 = 0
    let (init_int, _) = compiled.initial_state(&params).unwrap();
    let mut counts = init_int.counts.clone();
    // Force I_p5 = 0
    counts[i_p5_idx] = 0;
    // Give I_p1 some infections so importation can fire
    let i_p1_idx = compiled.model.compartments.iter().position(|c| c.name == "I_p1").unwrap();
    counts[i_p1_idx] = 100;

    eprintln!("  I_p5 = {}, I_p1 = {}", counts[i_p5_idx], counts[i_p1_idx]);

    let n_tr = compiled.model.transitions.len();
    let mut scratch = StepScratch::new(&compiled);

    // Run 100 substeps, check that infection_p5 NEVER fires
    let mut rng = StatefulRng::new(42);
    let fire_steps = compiled.resolve_fire_steps(1.0, &[]);
    let mut real = sim::state::RealState::new(compiled.real_local_to_global.len());
    for step in 0..100 {
        let mut flows = vec![0u64; n_tr];
        scratch.gamma_used.clear();
        sim::effects::due_effects(&compiled, &fire_steps, step as f64 + 1.0, 1.0, &mut scratch.effect_batch);
        step_one(&compiled, &mut counts, &mut flows, &mut real, &params, step as f64, 1.0, &mut rng, &mut scratch).unwrap();

        if flows[inf_p5_idx] > 0 {
            eprintln!("  STEP {}: infection_p5 has {} flows but I_p5 was {}",
                step, flows[inf_p5_idx], counts[i_p5_idx]);
            // Check what I_p5 was BEFORE step_one (it's now after)
            panic!("infection_p5 fired with I_p5={} at step {}", counts[i_p5_idx], step);
        }

        // After importation fires, I_p5 may become > 0. Subsequent steps
        // should be allowed to have infection_p5 fire. Only check when I_p5=0.
        if counts[i_p5_idx] > 0 {
            // I_p5 is now positive (from importation or progression), stop checking
            eprintln!("  I_p5 became {} at step {} — stopping zero-check", counts[i_p5_idx], step);
            break;
        }
    }
    eprintln!("  test passed: infection_p5 never fires when I_p5=0");
}
