//! Thread-invariance A/B gate for the parallel PGAS CSMC loop (gh#209).
//!
//! `csmc_as` propagates, ancestor-weights, and obs-weights its particles with
//! `par_iter` (pgas.rs). Each particle writes only its own slot and draws from
//! its own RNG stream, and every cross-particle reduction (systematic resample,
//! categorical ancestor draw) runs *after* the `par_iter` barrier — so the
//! result must be byte-identical regardless of how many worker threads execute
//! the loop. This gate proves that directly: the SAME deterministic PGAS run,
//! executed in a 1-thread rayon pool vs a 4-thread pool, must produce
//! bit-identical posterior draws and complete-data log-likelihoods.
//!
//! A failure here means the parallelisation introduced an order dependence
//! (a data race, a non-deterministic reduction, or RNG-by-position) — exactly
//! the regression the conversion must never cause.

use std::sync::Arc;

use rayon::ThreadPoolBuilder;
use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig, PGASResult, PGASSweep};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

const SEED: u64 = 20260613;
const DT: f64 = 1.0;
const N_PARTICLES: usize = 64;

fn host_model() -> ir::Model {
    let json = std::fs::read_to_string("../../../ocaml/golden/sir_overdispersion.ir.json")
        .expect("read sir_overdispersion golden");
    let mut model = ir::from_str(&json).expect("parse sir_overdispersion");
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            let v = match p.name.as_str() {
                "beta" => 0.3,
                "gamma" => 0.1,
                "sigma_se" => 0.1,
                "N0" => 1000.0,
                "I0" => 10.0,
                _ => 0.5,
            };
            p.value = p.value.with_value(v);
        }
    }
    model
}

/// A plain Poisson obs stream over the infection flow — keeps the fit well-posed
/// and avoids the BetaBinomial k ≤ n constraint; the obs model is irrelevant to
/// the invariance question (the particle loop runs the same way regardless).
fn poisson_obs_block() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    let rate = Expr::Projected(ProjectedExpr { projected: () });
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: "weekly_cases".into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("infection".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

fn params_from_compiled(compiled: &CompiledModel) -> Vec<f64> {
    let mut params = vec![0.0; compiled.param_index.len()];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }
    params
}

/// One fully-specified, deterministic PGAS run (config + data fixed by `SEED`).
fn run_once() -> PGASResult {
    let mut model = host_model();
    model.observations = vec![poisson_obs_block()];
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = params_from_compiled(&compiled);

    // Synthetic weekly observations from a fixed-seed reference trajectory.
    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(SEED);
    let truth = simulate_reference(&compiled, &params, t_end, DT, &mut rng).unwrap();
    let mut cum: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth.substeps.iter().enumerate() {
        cum += rec.flows[0];
        let t = ((s + 1) as f64) * DT;
        if (t.round() as i64) % 7 == 0 {
            obs.push(Observation { time: t, value: cum as f64 });
            cum = 0;
        }
    }

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec::dense(
            StreamProjection::FlowSum(vec![0]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();

    let pgas_params = vec![EstimatedParam {
        name: "beta".into(),
        index: compiled.param_index["beta"],
        initial: 0.3,
        rw_sd: 0.02,
        transform: Transform::Log { lo: 0.01, hi: 2.0 },
        lower: 0.01,
        upper: 2.0,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        ancestor_sampling: true,
        n_particles: N_PARTICLES,
        n_sweeps: 4,
        burn_in: 1,
        thin: 1,
        dt: DT,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 4,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    run_pgas(
        &compiled, &pgas_params, &priors, &params, &config, &obs, &obs_model,
        SEED, None, None, "thread_invariance".into(),
    )
    .expect("run_pgas must succeed")
}

fn assert_sweeps_bit_identical(a: &[PGASSweep], b: &[PGASSweep]) {
    assert_eq!(a.len(), b.len(), "same number of sweeps (1-thread vs 4-thread)");
    assert!(!a.is_empty(), "non-vacuous: at least one posterior sweep");
    for (i, (sa, sb)) in a.iter().zip(b).enumerate() {
        assert_eq!(sa.params.len(), sb.params.len(), "sweep {i}: same param count");
        for (k, (&pa, &pb)) in sa.params.iter().zip(&sb.params).enumerate() {
            assert_eq!(pa.to_bits(), pb.to_bits(),
                "sweep {i} param {k} differs across thread counts: {pa} vs {pb} \
                 — the parallel CSMC introduced an order dependence");
        }
        assert_eq!(sa.log_complete_data_ll.to_bits(), sb.log_complete_data_ll.to_bits(),
            "sweep {i}: complete-data log-likelihood differs across thread counts");
        assert_eq!(sa.transition_ll.to_bits(), sb.transition_ll.to_bits(),
            "sweep {i}: transition log-likelihood differs across thread counts");
        assert_eq!(sa.obs_ll.to_bits(), sb.obs_ll.to_bits(),
            "sweep {i}: observation log-likelihood differs across thread counts");
        // gh#742: the per-stream decomposition is a reported number too, so it
        // carries the same invariance obligation as the scalar it refines.
        assert_eq!(sa.obs_ll_per_stream.len(), sb.obs_ll_per_stream.len(),
            "sweep {i}: per-stream obs-loglik width differs across thread counts");
        assert!(!sa.obs_ll_per_stream.is_empty(),
            "sweep {i}: non-vacuous — the fixture must declare at least one stream");
        for (k, (&va, &vb)) in sa.obs_ll_per_stream.iter()
            .zip(&sb.obs_ll_per_stream).enumerate() {
            assert_eq!(va.to_bits(), vb.to_bits(),
                "sweep {i} stream {k}: per-stream observation log-likelihood \
                 differs across thread counts: {va} vs {vb}");
        }
    }
}

#[test]
fn pgas_csmc_is_thread_invariant() {
    // Same deterministic run, executed under a 1-worker pool vs a 4-worker pool.
    // `install` makes `csmc_as`'s `par_iter` use that pool, so the two runs
    // differ ONLY in how many threads execute the particle loop.
    let pool1 = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool4 = ThreadPoolBuilder::new().num_threads(4).build().unwrap();

    let serial = pool1.install(run_once);
    let parallel = pool4.install(run_once);

    assert_sweeps_bit_identical(&serial.sweeps, &parallel.sweeps);

    // Acceptance diagnostics are downstream of the same draws → also identical.
    assert_eq!(serial.acceptance_rates.len(), parallel.acceptance_rates.len());
    for (i, (&ra, &rb)) in serial.acceptance_rates.iter()
        .zip(&parallel.acceptance_rates).enumerate() {
        assert_eq!(ra.to_bits(), rb.to_bits(),
            "acceptance rate {i} differs across thread counts: {ra} vs {rb}");
    }
}
