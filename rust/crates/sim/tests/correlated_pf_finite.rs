//! Catch-test for gh#193: the correlated PF (CPM) must not return -inf where
//! the plain bootstrap PF is finite.
//!
//! Both filters are unbiased estimators of the SAME marginal likelihood. On a
//! model + data where the plain bootstrap PF gives a finite log-likelihood at
//! θ, the correlated PF — even with FRESH (uncorrelated) noise, which is what
//! the first PMMH evaluation uses — must also be finite and within
//! Monte-Carlo distance. A -inf from the correlated path at a θ the plain PF
//! scores finitely is a bug in the CPM draw machinery, not a property of the
//! likelihood.
//!
//! The CPM tests in tests/pmmh.rs only exercise the observation-grid handling
//! on a gentle pure-death + Poisson model; they never compare the CPM loglik
//! against the plain PF on a sharp, high-count likelihood — which is the regime
//! that trips gh#193.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, Expr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    chain_binomial::run_chain_binomial,
    config::ChainBinomialConfig,
    compiled_model::CompiledModel,
    inference::{
        correlated_pf::{bootstrap_filter_correlated, cpm_steps_per_obs, validate_cpm_obs_grid, PFRandomState},
        obs_loglik::negbin_logpmf,
        particle_filter::bootstrap_filter,
        traits::{ObservationModel, SMCConfig},
        ChainBinomialProcess, ParticleState,
    },
    rng::StatefulRng,
};

/// SEIR with frequency-dependent transmission: infection S→E at β·S·I/N,
/// progression E→I at σ·E, recovery I→R at γ·I. N = S+E+I+R.
fn seir_model() -> (CompiledModel, Vec<f64>) {
    let n_expr = Expr::pop_sum(vec!["S".into(), "E".into(), "I".into(), "R".into()]);
    let beta_s_i = Expr::bin_op(
        BinOp::Mul,
        Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
        Expr::pop("I"),
    );
    let infection_rate = Expr::bin_op(BinOp::Div, beta_s_i, n_expr);
    let progression_rate = Expr::bin_op(BinOp::Mul, Expr::param("sigma"), Expr::pop("E"));
    let recovery_rate = Expr::bin_op(BinOp::Mul, Expr::param("gamma"), Expr::pop("I"));

    let tr = |name: &str, from: &str, to: &str, rate: Expr| Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![
            StoichiometryEntry(from.into(), -1),
            StoichiometryEntry(to.into(), 1),
        ],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    };

    let model = Model {
        ic_grad: Default::default(),
        name: "seir_cpm".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "E".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![
            tr("infection", "S", "E", infection_rate),
            tr("progression", "E", "I", progression_rate),
            tr("recovery", "I", "R", recovery_rate),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Fixed { value: 0.3 }, param_kind: None, param_dim: None },
            Parameter { name: "sigma".into(), value: ir::parameter::ParamValue::Fixed { value: 0.2 }, param_kind: None, param_dim: None },
            Parameter { name: "gamma".into(), value: ir::parameter::ParamValue::Fixed { value: 0.1 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::constants({
            let mut m = HashMap::new();
            m.insert("S".into(), 99_990.0);
            m.insert("E".into(), 0.0);
            m.insert("I".into(), 10.0);
            m.insert("R".into(), 0.0);
            m
        }),
        output: OutputConfig {
            // Weekly output boundaries → each snapshot's `flows` is that week's
            // incidence (cumulative flow since the previous boundary).
            times: OutputSchedule::AtTimes((0..=52).map(|w| (w * 7) as f64).collect()),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 364.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// NegBinomial incidence observation on the `infection` transition (index 0):
/// y_t ~ NegBinomial(mean = ρ · weekly_incidence, dispersion = k).
struct NegBinIncidenceObs {
    observations: Vec<f64>,
    obs_times: Vec<f64>,
    infection_tr_idx: usize,
    rho: f64,
    k: f64,
}

impl ObservationModel<ParticleState> for NegBinIncidenceObs {
    fn log_likelihood(&self, state: &ParticleState, obs_idx: usize, _params: &[f64]) -> f64 {
        let incidence = state.flow_accumulators[self.infection_tr_idx] as f64;
        let mean = self.rho * incidence;
        negbin_logpmf(self.observations[obs_idx], mean, self.k)
    }
    fn n_observations(&self) -> usize { self.observations.len() }
    fn obs_time(&self, obs_idx: usize) -> f64 { self.obs_times[obs_idx] }
    fn n_streams(&self) -> usize { 1 }
    fn sample(&self, _s: &ParticleState, _i: usize, _p: &[f64], _rng: &mut StatefulRng) -> Vec<f64> { vec![] }
    fn mean(&self, _s: &ParticleState, _i: usize, _p: &[f64]) -> Vec<f64> { vec![] }
}

/// Forward-simulate the SEIR once at the truth, draw weekly observations
/// y_t ~ NegBinomial(mean = ρ · weekly_incidence, dispersion = k) for weeks
/// 1..=52 (obs times 7,14,…,364) — matching how the golden data is generated.
fn make_data(compiled: &CompiledModel, params: &[f64], rho: f64, k: f64) -> (Vec<f64>, Vec<f64>) {
    make_data_idx(compiled, params, rho, k, 0)
}

fn make_data_idx(compiled: &CompiledModel, params: &[f64], rho: f64, k: f64, inf_idx: usize) -> (Vec<f64>, Vec<f64>) {
    let cfg = ChainBinomialConfig { t_start: 0.0, t_end: 364.0, dt: 1.0 };
    let traj = run_chain_binomial(compiled, params, 42, &cfg).unwrap();
    let mut obs_rng = StatefulRng::new(42);
    let mut times = Vec::new();
    let mut obs = Vec::new();
    for snap in &traj.snapshots {
        if snap.t == 0.0 { continue; } // first boundary has no incidence window
        let incidence = snap.flows.as_int()[inf_idx] as f64;
        let mu = rho * incidence;
        // NB2(mu, k) = Poisson(mu · G), G ~ Gamma(k, 1/k) (mean 1, var 1/k).
        let g = obs_rng.gamma_multiplier(1.0 / k, 1.0);
        let y = obs_rng.poisson(mu * g) as f64;
        times.push(snap.t);
        obs.push(y);
    }
    (times, obs)
}

/// Load the golden SEIR-observations IR, apply the `baseline` preset, compile.
fn golden_seir() -> (CompiledModel, Vec<f64>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../ocaml/golden/seir_observations.ir.json");
    let contents = std::fs::read_to_string(path).expect("read golden seir_observations");
    let mut model: ir::Model = ir::from_str(&contents).unwrap();
    let preset = model.presets.first().expect("baseline preset").clone();
    for p in &mut model.parameters {
        if let Some(&v) = preset.params.get(&p.name) {
            p.value = p.value.with_value(v);
        }
    }
    // Force weekly output boundaries so each snapshot's flow is that week's
    // incidence — matches the `weekly_cases` obs cadence used in the CLI repro
    // (the golden's own output.times is daily, a different steps_per_obs regime).
    model.output.times = OutputSchedule::AtTimes((0..=52).map(|w| (w * 7) as f64).collect());
    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

/// Index of the `infection` transition in the compiled model.
fn infection_idx(compiled: &CompiledModel) -> usize {
    compiled.model.transitions.iter().position(|t| t.name == "infection").expect("infection transition")
}

/// Shared body: given a compiled process and the index of the incidence
/// transition, generate self-consistent NB data at the truth, then compare the
/// plain bootstrap PF to the fresh-noise correlated PF. Returns (plain, corr).
fn compare_filters(compiled: CompiledModel, params: &[f64], inf_idx: usize, n_particles: usize) -> (f64, f64) {
    let rho = 0.5;
    let k = 5.0;
    let (obs_times, observations) = make_data_idx(&compiled, params, rho, k, inf_idx);
    let peak = observations.iter().cloned().fold(0.0_f64, f64::max);
    eprintln!("  data: {} weekly obs, peak cases = {peak}", observations.len());

    let compiled = Arc::new(compiled);
    let dt = 1.0;
    let process = ChainBinomialProcess::new(compiled.clone());
    let config = SMCConfig {
        n_particles, dt, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false, max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let obs_model = NegBinIncidenceObs {
        observations: observations.clone(), obs_times: obs_times.clone(),
        infection_tr_idx: inf_idx, rho, k,
    };

    let plain = bootstrap_filter(&process, &obs_model, params, &config, 7).unwrap();
    let steps_per_obs = cpm_steps_per_obs(&obs_times, config.t_start, dt);
    let n_source_groups = compiled.source_groups.len();
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(n_particles, &steps_per_obs, n_source_groups, &mut rng);
    let corr = bootstrap_filter_correlated(&process, &obs_model, params, &config, &randoms, 7).unwrap();

    eprintln!("  plain PF loglik = {}   corr PF loglik = {}", plain.log_likelihood, corr.log_likelihood);
    for (i, (&lp, &lc)) in plain.ll_increments.iter().zip(corr.ll_increments.iter()).enumerate() {
        if lc.is_infinite() && lp.is_finite() {
            eprintln!("  FIRST DIVERGENT WINDOW {i} (t={}): plain {lp:.3} vs corr {lc:.3}  (y={})",
                obs_times[i], observations[i]);
            break;
        }
    }
    (plain.log_likelihood, corr.log_likelihood)
}

#[test]
fn golden_seir_correlated_pf_finite_where_plain_pf_is() {
    let (compiled, params) = golden_seir();
    let inf_idx = infection_idx(&compiled);
    eprintln!("golden SEIR (infection idx {inf_idx}):");
    let (plain, corr) = compare_filters(compiled, &params, inf_idx, 100);
    assert!(plain.is_finite(), "plain PF must be finite at truth on golden");
    assert!(corr.is_finite(),
        "correlated PF (fresh noise) must be finite where plain PF is (gh#193); \
         got corr {corr} vs plain {plain}");
}

/// gh#193 catch-test: the real `weekly_cases` schedule is `regular start=0
/// step=7`, so the obs grid starts at t=0 and the FIRST CPM window
/// [t_start=0, obs(0)=0] is zero-substep. The plain PF scores the t=0 obs at
/// the initial state with no trouble; the correlated PF must do the same — a
/// zero-width LEADING window consumes no noise, so the pre-drawn-noise indexing
/// is unaffected. CPM on this grid must therefore be finite and within MC
/// distance of the plain PF.
///
/// (Before the gh#193 fix the obs-grid gate rejected the t=0 first window with
/// a `SimError::Validation`, which profile/fit then swallowed into -inf — a
/// silent all-(-inf) profile on every standard `start=0` model.)
#[test]
fn correlated_pf_finite_on_t0_starting_grid() {
    let (compiled, params) = golden_seir();
    let inf_idx = infection_idx(&compiled);
    let rho = 0.5;
    let k = 5.0;
    // Data on the WEEKLY grid PREPENDED with the t=0 observation, exactly like
    // the golden `regular start=0` schedule.
    let (mut times, mut obs) = make_data_idx(&compiled, &params, rho, k, inf_idx);
    times.insert(0, 0.0);
    obs.insert(0, 0.0);

    let compiled = Arc::new(compiled);
    let dt = 1.0;
    let process = ChainBinomialProcess::new(compiled.clone());
    let config = SMCConfig {
        n_particles: 200, dt, t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false, max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let obs_model = NegBinIncidenceObs {
        observations: obs.clone(), obs_times: times.clone(),
        infection_tr_idx: inf_idx, rho, k,
    };

    // Plain PF: scores the t=0 obs at the initial state → finite.
    let plain = bootstrap_filter(&process, &obs_model, &params, &config, 7).unwrap();
    eprintln!("plain PF on t=0-starting grid: loglik = {}", plain.log_likelihood);
    assert!(plain.log_likelihood.is_finite(),
        "plain PF must handle an obs at t=0 (scores it at the initial state)");

    // Correlated PF with FRESH noise (== the first PMMH eval). Must be finite.
    // The block sizes come from the same function run_pmmh and the filter use;
    // the leading t=0 window is empty and gets a block of zero.
    let steps_per_obs = cpm_steps_per_obs(&times, config.t_start, dt);
    assert_eq!(steps_per_obs[0], 0, "the leading t=0 window consumes no noise");
    let n_source_groups = compiled.source_groups.len();
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(200, &steps_per_obs, n_source_groups, &mut rng);
    let corr = bootstrap_filter_correlated(&process, &obs_model, &params, &config, &randoms, 7)
        .expect("CPM must accept a leading t=0 (zero-width) window (gh#193)");
    eprintln!("corr  PF on t=0-starting grid: loglik = {}", corr.log_likelihood);
    assert!(corr.log_likelihood.is_finite(),
        "correlated PF (fresh noise) must be finite on a t=0-starting grid \
         where the plain PF is finite (gh#193); got {} vs plain {}",
        corr.log_likelihood, plain.log_likelihood);
    // Both estimate the same marginal likelihood — agree within MC slack.
    assert!((corr.log_likelihood - plain.log_likelihood).abs() < 20.0,
        "CPM ({}) and plain PF ({}) should agree within MC distance",
        corr.log_likelihood, plain.log_likelihood);
}

#[test]
fn correlated_pf_finite_where_plain_pf_is() {
    let (compiled, params) = seir_model();
    let rho = 0.5;
    let k = 5.0;
    let (obs_times, observations) = make_data(&compiled, &params, rho, k);
    let peak = observations.iter().cloned().fold(0.0_f64, f64::max);
    eprintln!(
        "data: {} weekly obs, peak cases = {peak}",
        observations.len()
    );
    assert!(peak > 1000.0, "expected a high-count epidemic, got peak {peak}");

    let compiled = Arc::new(compiled);
    let n_particles = 100;
    let dt = 1.0;
    let process = ChainBinomialProcess::new(compiled.clone());
    let config = SMCConfig {
        n_particles,
        dt,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };

    let obs_model = NegBinIncidenceObs {
        observations: observations.clone(),
        obs_times: obs_times.clone(),
        infection_tr_idx: 0,
        rho,
        k,
    };

    // ── Plain bootstrap PF at the truth ──
    let plain = bootstrap_filter(&process, &obs_model, &params, &config, 7).unwrap();
    eprintln!("plain PF  loglik = {}", plain.log_likelihood);

    // ── Correlated PF with FRESH noise (== the first PMMH eval) ──
    let steps_per_obs = cpm_steps_per_obs(&obs_times, config.t_start, dt);
    let n_source_groups = compiled.source_groups.len();
    let mut rng = StatefulRng::new(7);
    let randoms = PFRandomState::draw_fresh(n_particles, &steps_per_obs, n_source_groups, &mut rng);
    let corr = bootstrap_filter_correlated(&process, &obs_model, &params, &config, &randoms, 7).unwrap();
    eprintln!("corr  PF  loglik = {}", corr.log_likelihood);

    // Per-window localisation: print the first window where the correlated
    // increment is -inf while the plain one is finite.
    for (i, (&lp, &lc)) in plain.ll_increments.iter().zip(corr.ll_increments.iter()).enumerate() {
        if lc.is_infinite() && lp.is_finite() {
            eprintln!(
                "FIRST DIVERGENT WINDOW {i} (t={}): plain {lp:.3} vs corr {lc:.3}  (y={})",
                obs_times[i], observations[i]
            );
            break;
        }
    }

    assert!(plain.log_likelihood.is_finite(), "plain PF must be finite at truth");
    assert!(
        corr.log_likelihood.is_finite(),
        "correlated PF (fresh noise) must be finite where the plain PF is \
         finite (gh#193); got {} vs plain {}",
        corr.log_likelihood, plain.log_likelihood
    );
}

// ── Observation-grid unit coverage (the seam shared by the filter gate and
//    the profile/fit preflight, gh#193) ──────────────────────────────────────
//
// The pre-drawn noise is one block per observation window, each sized at that
// window's own substep count, so an irregular grid indexes correctly: a daily
// reporting series that skips a day, or a series that starts mid-period, is
// accepted and each window reads its own block. What `validate_cpm_obs_grid`
// still rejects is a grid that does not describe a forward walk from t_start.

/// The block sizes must be what the schedule's substep walk actually yields:
/// the sizing runs before any `Schedule` exists (the noise is drawn in
/// `run_pmmh`), so a divergence between the two would be a wrong stride, which
/// reads a valid float from the wrong slot rather than failing.
#[test]
fn cpm_block_sizes_match_the_schedule_walk() {
    use sim::boundary_times::{EffectTimes, ObsTimes};
    use sim::intervention::TimelineEffects;
    use sim::schedule::{Cursor, Schedule};

    let cases: Vec<(Vec<f64>, f64, f64)> = vec![
        // (obs times, t_start, dt)
        (vec![0.0, 7.0, 14.0, 21.0], 0.0, 1.0),   // leading t0 window, weekly
        (vec![7.0, 14.0, 21.0], 0.0, 1.0),        // no t0 obs
        (vec![5.0, 12.0, 19.0], 0.0, 1.0),        // mid-period start
        (vec![0.0, 7.0, 14.0, 18.0, 25.0], 0.0, 1.0), // short interior window
        (vec![1.0, 2.0, 4.0, 5.0], 0.0, 1.0),     // daily with day 3 absent
        (vec![3.5, 7.0], 0.0, 1.0),               // off-grid obs (gh#216)
        (vec![0.3, 7.3, 14.3], 0.0, 1.0),         // sub-dt leading offset
        (vec![1.0, 2.0], 0.0, 0.25),              // several substeps per window
        (vec![10.0], 0.0, 1.0),                   // single observation
    ];

    for (times, t_start, dt) in cases {
        let sized = cpm_steps_per_obs(&times, t_start, dt);
        let schedule = Schedule::exact_inference(
            dt,
            *times.last().unwrap(),
            EffectTimes::from_timeline(&TimelineEffects::default()).unwrap(),
            ObsTimes::new(times.clone()).unwrap(),
        );
        let mut t = t_start;
        for obs_idx in 0..times.len() {
            let cur = Cursor { obs_idx, effect_idx: 0, ..Default::default() };
            let mut walked = 0usize;
            for (t0, step_dt, _) in schedule.substeps(cur, t) {
                walked += 1;
                t = t0 + step_dt;
            }
            assert_eq!(
                sized[obs_idx], walked,
                "block size disagrees with the walk for window {obs_idx} of \
                 {times:?} at t_start={t_start}, dt={dt}: sized {}, walked {walked}",
                sized[obs_idx],
            );
        }
    }
}

#[test]
fn cpm_grid_accepts_leading_t0_window() {
    // The universal `regular start=0` case: obs at [0,7,14,21], t_start=0. The
    // leading [0,0] window is empty (scored at init); windows 1.. are 7 each.
    let grid = vec![0.0, 7.0, 14.0, 21.0];
    assert!(validate_cpm_obs_grid(&grid, 0.0, 1.0).is_ok(),
        "a leading window coinciding with t_start must be allowed");
    assert_eq!(cpm_steps_per_obs(&grid, 0.0, 1.0), vec![0, 7, 7, 7]);
}

#[test]
fn cpm_grid_accepts_first_obs_at_obs_dt() {
    // No t=0 obs: first obs at exactly one window from t_start.
    assert!(validate_cpm_obs_grid(&[7.0, 14.0, 21.0], 0.0, 1.0).is_ok());
    assert_eq!(cpm_steps_per_obs(&[7.0, 14.0, 21.0], 0.0, 1.0), vec![7, 7, 7]);
}

#[test]
fn cpm_grid_accepts_mid_period_start() {
    // Data starting mid-period: obs at [5,12,19], t_start=0. The first window
    // [0,5] is 5 substeps where the rest are 7 — its noise block is sized at 5,
    // so the indexing is sound and the run is accepted.
    assert!(validate_cpm_obs_grid(&[5.0, 12.0, 19.0], 0.0, 1.0).is_ok(),
        "a short first window is now sized for, not rejected");
    assert_eq!(cpm_steps_per_obs(&[5.0, 12.0, 19.0], 0.0, 1.0), vec![5, 7, 7]);
}

#[test]
fn cpm_grid_accepts_nonuniform_interior() {
    // Uniform start but a short interior window: [0,7,14,18,25]. Window
    // [14,18] is 4 substeps; it gets a 4-substep block.
    let grid = [0.0, 7.0, 14.0, 18.0, 25.0];
    assert!(validate_cpm_obs_grid(&grid, 0.0, 1.0).is_ok(),
        "an interior window of its own length is now sized for, not rejected");
    assert_eq!(cpm_steps_per_obs(&grid, 0.0, 1.0), vec![0, 7, 7, 4, 7]);
}

#[test]
fn cpm_grid_accepts_a_daily_series_missing_one_day() {
    // The reporting case this supports: daily observations from day 1, with no
    // situation report on day 3. Every window is one substep except the one
    // spanning the absent day, which is two.
    let grid: Vec<f64> = vec![1.0, 2.0, 4.0, 5.0, 6.0];
    assert!(validate_cpm_obs_grid(&grid, 0.0, 1.0).is_ok());
    assert_eq!(cpm_steps_per_obs(&grid, 0.0, 1.0), vec![1, 1, 2, 1, 1]);
}

#[test]
fn cpm_grid_accepts_single_obs() {
    // Degenerate single-observation cases: at t_start and away from it.
    assert!(validate_cpm_obs_grid(&[0.0], 0.0, 1.0).is_ok(), "single obs at t_start");
    assert!(validate_cpm_obs_grid(&[10.0], 0.0, 1.0).is_ok(), "single obs away from t_start");
    assert_eq!(cpm_steps_per_obs(&[0.0], 0.0, 1.0), vec![0]);
    assert_eq!(cpm_steps_per_obs(&[10.0], 0.0, 1.0), vec![10],
        "the whole run is one window — the old scalar sizing called this 1");
}

#[test]
fn cpm_grid_accepts_subdt_leading_offset() {
    // A sub-dt leading offset (obs(0)=0.3, dt=1): the Exact walk takes one
    // clipped 0.3-substep, and the block is sized at 1 to match. Rounding
    // `(obs(0) - t_start)/dt` would have sized it at 0 and overrun.
    let grid = [0.3, 7.3, 14.3];
    assert!(validate_cpm_obs_grid(&grid, 0.0, 1.0).is_ok());
    assert_eq!(cpm_steps_per_obs(&grid, 0.0, 1.0), vec![1, 7, 7]);
}

#[test]
fn cpm_grid_rejects_an_observation_before_t_start() {
    // obs(0) < t_start: the window is backwards, so no substep is due and the
    // observation would be scored at the initial state with no warning.
    let err = validate_cpm_obs_grid(&[-2.0, 5.0, 12.0], 0.0, 1.0)
        .expect_err("an observation before t_start must be rejected");
    let msg = format!("{err}");
    assert!(msg.contains("obs(0)") && msg.contains("t_start"),
        "rejection must name the first observation and t_start; got {msg}");
}

#[test]
fn cpm_grid_rejects_a_backwards_interior_step() {
    let err = validate_cpm_obs_grid(&[1.0, 5.0, 3.0], 0.0, 1.0)
        .expect_err("a grid that goes backwards must be rejected");
    assert!(format!("{err}").contains("obs(2)"),
        "rejection must name the offending observation; got {err}");
}

#[test]
fn cpm_grid_rejects_a_non_finite_observation_time() {
    let err = validate_cpm_obs_grid(&[1.0, f64::NAN, 3.0], 0.0, 1.0)
        .expect_err("a non-finite observation time must be rejected");
    assert!(format!("{err}").contains("not finite"), "got {err}");
}
