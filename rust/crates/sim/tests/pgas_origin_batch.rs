//! PGAS and the effects at t_start (D1, proposal 2026-10-04 §4 "PGAS", §6.3).
//!
//! `initial_counts` stays x₀, the state the `init {}` law scores; the plan's
//! origin batch is a deterministic map from x₀ to `substeps[0].counts_before`,
//! applied by the reference walk, the CSMC sweep and the initialization filter.
//! Missing any one of those sites is silent-wrong — the sweep mixes x₀ and its
//! image as the substep-0 pre-state and the density scores transitions from a
//! state the producer never used — so each test here would catch one.
//!
//! The model is `sir_basic` (S, I, R; infection `beta·S·I/N`, recovery
//! `gamma·I`) with an initial-state law `I ~ binomial(n = N0, p = 0.4)`,
//! `S = N0 − I`, and at t_start an event `add(I, 1)` and an intervention
//! `absolute_transfer(S → R, 1)`. With N0 = 6 every quantity is hand-checkable.

use std::sync::Arc;

use serde_json::{json, Value};
use sim::compiled_model::CompiledModel;
use sim::inference::multi_stream_obs::MultiStreamObsModel;
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    complete_data_loglik, csmc_as, pgas_effect_plan, simulate_reference, ObsAtSubstep,
};
use sim::rng::StatefulRng;

const N0: i64 = 6;
const P_INIT: f64 = 0.4;
const BETA: f64 = 1.2;
const GAMMA: f64 = 0.5;
const DT: f64 = 1.0;
const T_END: f64 = 5.0;

/// `sir_basic` with the initial-state law and the given effects (`(name, at,
/// action JSON, is_event)`).
fn model(effects: &[(&str, f64, Value, bool)], beta: f64, gamma: f64) -> Arc<CompiledModel> {
    model_with_init(effects, beta, gamma, true)
}

/// [`model`], with the init law (`law = true`) or `sir_basic`'s deterministic
/// `S = N0 - I0`, `I = I0` (I0 = 2).
fn model_with_init(effects: &[(&str, f64, Value, bool)], beta: f64, gamma: f64, law: bool) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let mut doc: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let m = &mut doc["model"];
    if law { m["initial_conditions"] = json!({
        "S": {"deterministic": {"bin_op": {"op": "sub", "left": {"param": "N0"}, "right": {"pop": "I"}}}},
        "I": {"count": {"binomial": {"n": {"param": "N0"}, "p": {"expr": {"const": P_INIT}}}}},
    }); }
    m["interventions"] = Value::Array(
        effects
            .iter()
            .map(|(name, at, action, event)| {
                let mut v = json!({"name": name, "fire": {"scheduled": {"at_times": [at]}}, "actions": [action]});
                if *event {
                    v["kind"] = json!("event");
                }
                v
            })
            .collect(),
    );
    let mut model = ir::from_str(&serde_json::to_string(&doc).unwrap()).expect("parse");
    for p in &mut model.parameters {
        let v = match p.name.as_str() {
            "beta" => beta,
            "gamma" => gamma,
            "N0" => N0 as f64,
            "I0" => 2.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(model).expect("compile"))
}

fn origin_model() -> Arc<CompiledModel> {
    model(
        &[
            ("cull", 0.0, json!({"absolute_transfer": {"src": "S", "dst": "R", "count": {"const": 1.0}}}), false),
            ("seed", 0.0, json!({"add": {"compartment": "I", "count": {"const": 1.0}}}), true),
        ],
        BETA,
        GAMMA,
    )
}

/// The origin map by hand: the event first (`I + 1`), then the intervention
/// (`min(1, S)` from S to R) — §13.10.
fn origin_by_hand(x0: &[i64]) -> Vec<i64> {
    let (mut s, mut i, mut r) = (x0[0], x0[1], x0[2]);
    i += 1;
    let x = 1.min(s);
    s -= x;
    r += x;
    vec![s, i, r]
}

fn ln_binom(n: i64, k: i64, p: f64) -> f64 {
    assert!(0 <= k && k <= n, "k = {k} outside 0..={n}");
    let c = libm::lgamma(n as f64 + 1.0) - libm::lgamma(k as f64 + 1.0) - libm::lgamma((n - k) as f64 + 1.0);
    let lp = if k == 0 { 0.0 } else { k as f64 * p.ln() };
    let lq = if k == n { 0.0 } else { (n - k) as f64 * (1.0 - p).ln() };
    c + lp + lq
}

fn reference(model: &CompiledModel, seed: u64) -> sim::inference::pgas::PGASTrajectory {
    let mut rng = StatefulRng::new(seed);
    simulate_reference(model, &model.default_params, T_END, DT, &mut rng).expect("reference walk")
}

/// The reference walk records x₀ as `initial_counts` (a draw of the law: S + I
/// = N0, R = 0) and its origin image as `substeps[0].counts_before`.
#[test]
fn the_reference_walk_maps_x0_to_its_post_origin_image() {
    let m = origin_model();
    for seed in 1..20 {
        let traj = reference(&m, seed);
        let x0 = &traj.initial_counts;
        assert_eq!(x0[0] + x0[1], N0, "seed {seed}: x0 is a draw of the init law");
        assert_eq!(x0[2], 0);
        assert_eq!(traj.substeps[0].counts_before, origin_by_hand(x0), "seed {seed}");
        assert_ne!(&traj.substeps[0].counts_before, x0, "the origin batch changed the state");
    }
}

/// `complete_data_loglik` scores `log p(x₀)` at `initial_counts` and every
/// transition from the state the origin map and the recorded flows give, both
/// computed here by hand: x₀ and the post-effect state are not conflated.
#[test]
fn complete_data_loglik_scores_x0_and_steps_from_its_image() {
    let m = origin_model();
    let obs_model = MultiStreamObsModel::empty(m.clone());
    let no_obs: Vec<Observation> = vec![];
    for seed in 1..10 {
        let traj = reference(&m, seed);
        let x0 = &traj.initial_counts;
        let mut want = ln_binom(N0, x0[1], P_INIT);
        let mut x = origin_by_hand(x0);
        for rec in &traj.substeps {
            let (s, i, r) = (x[0], x[1], x[2]);
            let n = (s + i + r) as f64;
            let p_inf = 1.0 - (-BETA * i as f64 / n * rec.dt_substep).exp();
            let p_rec = 1.0 - (-GAMMA * rec.dt_substep).exp();
            let (k_inf, k_rec) = (rec.flows[0] as i64, rec.flows[1] as i64);
            want += if p_inf > 0.0 { ln_binom(s, k_inf, p_inf) } else { assert_eq!(k_inf, 0); 0.0 };
            want += ln_binom(i, k_rec, p_rec);
            x = vec![s - k_inf, i + k_inf - k_rec, r + k_rec];
        }
        let got = complete_data_loglik(&m, &traj, &m.default_params, &no_obs, DT, &obs_model, &ObsAtSubstep::new())
            .expect("complete-data density");
        assert!((got.total - want).abs() < 1e-9, "seed {seed}: {} vs hand {want}", got.total);
        assert!((got.initial_state - ln_binom(N0, x0[1], P_INIT)).abs() < 1e-12);
    }
}

/// With one particle — and so no free particle to graft onto — a CSMC sweep
/// returns the reference trajectory unchanged, with ancestor sampling on and
/// off. A sweep that started the reference slot from x₀ instead of its recorded
/// substep-0 pre-state, or applied the origin batch to it a second time, would
/// return a different substep-0 state here.
#[test]
fn a_one_particle_sweep_returns_the_reference_unchanged() {
    let m = origin_model();
    let obs_model = MultiStreamObsModel::empty(m.clone());
    let traj = reference(&m, 3);
    let grid: Vec<(f64, f64)> = traj.substeps.iter().map(|r| (r.t0, r.dt_substep)).collect();
    let plan = pgas_effect_plan(&m, &m.default_params, DT, &grid).unwrap();
    for ancestor_sampling in [true, false] {
        let (out, _) = csmc_as(
            &m, &m.default_params, &[], &traj, 1, DT, &obs_model, 11, &ObsAtSubstep::new(), &plan,
            sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling,
        )
        .expect("csmc sweep");
        assert_eq!(out.initial_counts, traj.initial_counts, "AS={ancestor_sampling}: x0");
        for (s, (a, b)) in out.substeps.iter().zip(&traj.substeps).enumerate() {
            assert_eq!(a.counts_before, b.counts_before, "AS={ancestor_sampling}: substep {s} pre-state");
            assert_eq!(a.counts_after, b.counts_after, "AS={ancestor_sampling}: substep {s} post-state");
            assert_eq!(a.flows, b.flows, "AS={ancestor_sampling}: substep {s} flows");
        }
    }
}

/// A sweep with many particles keeps every returned path consistent: its
/// substep-0 pre-state is the origin image of its `initial_counts`, whichever
/// particle's lineage it is (free particles draw their own x₀).
#[test]
fn every_swept_path_starts_from_the_image_of_its_own_x0() {
    let m = origin_model();
    let obs_model = MultiStreamObsModel::empty(m.clone());
    let traj = reference(&m, 5);
    let grid: Vec<(f64, f64)> = traj.substeps.iter().map(|r| (r.t0, r.dt_substep)).collect();
    let plan = pgas_effect_plan(&m, &m.default_params, DT, &grid).unwrap();
    let mut saw_free = false;
    for seed in 0..20 {
        let (out, _) = csmc_as(
            &m, &m.default_params, &[], &traj, 16, DT, &obs_model, seed, &ObsAtSubstep::new(), &plan,
            sim::rng::BinomialAlgorithm::Btpe, true,
        )
        .expect("csmc sweep");
        assert_eq!(out.substeps[0].counts_before, origin_by_hand(&out.initial_counts), "seed {seed}");
        saw_free |= out.initial_counts != traj.initial_counts;
    }
    assert!(saw_free, "non-vacuity: some sweep must return a free particle's lineage");
}

/// An off-grid event is snapped in the PGAS producer: `add(I, 1)` at 2.4 with
/// dt = 1 acts at the end of the substep ending at 2 (k = ⌊2.4 + ½⌋ = 2). Zero
/// rates, so only the event changes the state.
#[test]
fn an_off_grid_event_is_snapped_in_the_producer() {
    let m = model(&[("late", 2.4, json!({"add": {"compartment": "I", "count": {"const": 1.0}}}), true)], 0.0, 0.0);
    let traj = reference(&m, 1);
    let i: Vec<i64> = traj.substeps.iter().map(|r| r.counts_after[1] - r.counts_before[1]).collect();
    assert_eq!(i, vec![0, 1, 0, 0, 0], "the event acts on the substep [1, 2]");
}

/// The initialization filter (gh#784) maps each particle's x₀ through the same
/// origin batch: the lineage it returns starts from the image of its own x₀.
#[test]
fn the_initialization_filter_starts_from_the_image_of_x0() {
    use sim::inference::pgas_init::{unconditional_smc_pass, UnconditionalPass};
    let m = origin_model();
    let obs_model = MultiStreamObsModel::empty(m.clone());
    let grid: Vec<(f64, f64)> = (0..5).map(|s| (s as f64 * DT, DT)).collect();
    let plan = pgas_effect_plan(&m, &m.default_params, DT, &grid).unwrap();
    for seed in 0..10 {
        let pass = unconditional_smc_pass(
            &m, &m.default_params, &grid, 8, DT, &obs_model, seed, &ObsAtSubstep::new(), &plan,
            sim::rng::BinomialAlgorithm::Btpe,
        )
        .expect("init pass");
        let UnconditionalPass::Path(traj) = pass else { panic!("seed {seed}: no path") };
        assert_eq!(traj.substeps[0].counts_before, origin_by_hand(&traj.initial_counts), "seed {seed}");
    }
}

// ─── An observation at t_start (scored after the origin batch) ──────────────

/// A Poisson reading of `I` (`rate = I + 0.1`) at `times` with values `y`.
fn i_obs(m: &Arc<CompiledModel>, times: &[f64], y: &[f64]) -> MultiStreamObsModel {
    use sim::inference::multi_stream_obs::{dense_cells, BoundObs, StreamProjection, StreamSpec, StreamTimes};
    let spec = StreamSpec {
        projection: StreamProjection::IntCompSum(vec![1]),
        ir_model: ir::observation::ObservationModel {
            name: "i_obs".into(),
            source: "i_obs".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "i_obs".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "i_obs".into(),
            emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
            projection: ir::observation::Projection::CurrentPop("I".into()),
            projection_state_grad: Default::default(),
            likelihood: ir::observation::Likelihood::Poisson(ir::observation::PoissonLikelihood {
                rate: ir::Diffable::new(ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
                    bin_op: ir::expr::BinOpExpr {
                        op: ir::expr::BinOp::Add,
                        left: Box::new(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
                        right: Box::new(ir::expr::Expr::Const(ir::expr::ConstExpr { value: 0.1 })),
                    },
                })),
            }),
            stratum: vec![],
            covers: None,
        },
        observations: dense_cells(y.to_vec()),
        times: StreamTimes::Instants(times.to_vec()),
        aux: vec![],
    };
    MultiStreamObsModel::new(BoundObs::bind(0.0, vec![spec]).unwrap().0, m.clone()).unwrap()
}

fn observations(times: &[f64], y: &[f64]) -> Vec<Observation> {
    times.iter().zip(y).map(|(&time, &value)| Observation { time, value }).collect()
}

/// PGAS scores an observation at t_start on the post-origin state, before any
/// substep, exactly as the bootstrap filter does: on one fixed path, the
/// complete-data log-likelihood with the t_start row exceeds the one without it
/// by `log Poisson(7; I + 0.1)` at the post-origin I = 2 + 1, and that is the
/// filter's first increment (every particle starts from the same state). The
/// observation map counts the row. (Before, `build_obs_at_substep` kept a row
/// only at a substep `s > 0`, and the t_start row changed nothing.)
#[test]
fn an_observation_at_t_start_is_scored_after_the_origin_batch() {
    use sim::inference::obs_loglik::poisson_logpmf;
    use sim::inference::pgas::build_obs_at_substep;
    let m = model_with_init(
        &[("seed", 0.0, json!({"add": {"compartment": "I", "count": {"const": 1.0}}}), true)],
        BETA,
        GAMMA,
        false,
    );
    let (t_with, y_with) = ([0.0, 1.0, 2.0], [7.0, 3.0, 3.0]);
    let (t_no, y_no) = ([1.0, 2.0], [3.0, 3.0]);
    let traj = reference(&m, 4);
    assert_eq!(traj.substeps[0].counts_before[1], 3, "post-origin I = 2 + 1");
    let map_with = build_obs_at_substep(&observations(&t_with, &y_with), 0.0, DT).unwrap();
    let map_no = build_obs_at_substep(&observations(&t_no, &y_no), 0.0, DT).unwrap();
    assert_eq!(map_with.len(), 3, "the t_start row is counted");
    assert_eq!(map_with.at_origin(), Some(0));
    let ll = |times: &[f64], y: &[f64], map: &ObsAtSubstep| {
        complete_data_loglik(&m, &traj, &m.default_params, &observations(times, y), DT, &i_obs(&m, times, y), map)
            .unwrap()
            .total
    };
    let contribution = ll(&t_with, &y_with, &map_with) - ll(&t_no, &y_no, &map_no);
    let want = poisson_logpmf(7.0, 3.1);
    assert!((contribution - want).abs() < 1e-9, "PGAS t_start term {contribution} vs {want}");

    let config = sim::inference::SMCConfig {
        n_particles: 16, dt: DT, t_start: 0.0, skip_first_obs_from_loglik: false,
        record_ancestry: false, record_prequential: false, record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    let pf = sim::inference::particle_filter::bootstrap_filter(
        &sim::inference::ChainBinomialProcess::new(m.clone()),
        &i_obs(&m, &t_with, &y_with), &m.default_params, &config, 9,
    )
    .unwrap();
    assert!((pf.ll_increments[0] - want).abs() < 1e-9, "PF t_start term {} vs {want}", pf.ll_increments[0]);
}

/// Under exact obs-alignment the substep walk scores an observation at t_start
/// at the origin and goes on to the later observations; it used to stop at the
/// empty first window and drop every observation after it.
#[test]
fn the_exact_grid_keeps_the_observations_after_one_at_t_start() {
    use sim::inference::pgas::build_substep_grid;
    use sim::schedule::StepPolicy;
    let g = build_substep_grid(0.0, DT, &observations(&[0.0, 1.5, 3.0], &[1.0, 1.0, 1.0]), StepPolicy::Exact)
        .unwrap();
    assert_eq!(g.obs_at_substep.at_origin(), Some(0));
    assert_eq!(g.obs_at_substep.len(), 3, "all three observations are scored");
    let t0s: Vec<f64> = g.steps.iter().map(|&(t0, _)| t0).collect();
    assert_eq!(t0s, vec![0.0, 1.0, 1.5, 2.5], "substeps [0,1], [1,1.5], [1.5,2.5], [2.5,3]");
}

/// The NUTS gradient path scores the t_start observation too: its log-density
/// equals the complete-data log-likelihood with the row.
#[test]
fn the_gradient_path_scores_the_t_start_observation() {
    use sim::inference::pgas::build_obs_at_substep;
    use sim::inference::pgas_grad::{complete_data_loglik_grad, resolve_rate_grad_for_run};
    let m = origin_model();
    let (times, y) = ([0.0, 2.0, 4.0], [9.0, 3.0, 2.0]);
    let obs = observations(&times, &y);
    let om = i_obs(&m, &times, &y);
    let map = build_obs_at_substep(&obs, 0.0, DT).unwrap();
    let traj = reference(&m, 6);
    let value = complete_data_loglik(&m, &traj, &m.default_params, &obs, DT, &om, &map).unwrap().total;
    let n = m.default_params.len();
    let to_est: Vec<Option<usize>> = (0..n).map(Some).collect();
    let rate_grads = resolve_rate_grad_for_run(&m.resolved.rate_grads_indexed, &to_est);
    let (ll, _) = complete_data_loglik_grad(
        &m, &traj, &m.default_params, &obs, DT, &om, n, &rate_grads, &map, &(0..n).collect::<Vec<_>>(),
    )
    .unwrap();
    assert!((ll - value).abs() < 1e-9, "gradient path {ll} vs value {value}");
}

/// The CSMC sweep and the initialization filter weight their particles by the
/// t_start observation before substep 0. A reading of 40 at t_start makes a
/// large initial I overwhelmingly the likeliest (each extra case multiplies the
/// weight by about (I + 1.1)/(I + 0.1))^40), so the swept and the initial
/// paths start from x₀'s I ≥ 4; unweighted, x₀'s I is a Binomial(6, 0.4) draw
/// and is ≥ 4 about 18% of the time.
#[test]
fn the_sweep_and_the_init_filter_weight_by_the_t_start_observation() {
    use sim::inference::pgas::build_obs_at_substep;
    use sim::inference::pgas_init::{unconditional_smc_pass, UnconditionalPass};
    let m = origin_model();
    let (times, y) = ([0.0, 3.0], [40.0, 7.0]);
    let om = i_obs(&m, &times, &y);
    let map = build_obs_at_substep(&observations(&times, &y), 0.0, DT).unwrap();
    let grid: Vec<(f64, f64)> = (0..3).map(|s| (s as f64 * DT, DT)).collect();
    let plan = pgas_effect_plan(&m, &m.default_params, DT, &grid).unwrap();
    let traj = reference(&m, 2);
    let (mut swept, mut init) = (0, 0);
    for seed in 0..10 {
        let (out, _) = csmc_as(
            &m, &m.default_params, &observations(&times, &y), &traj, 64, DT, &om, seed, &map, &plan,
            sim::rng::BinomialAlgorithm::Btpe, true,
        )
        .unwrap();
        swept += usize::from(out.initial_counts[1] >= 4);
        if let UnconditionalPass::Path(p) = unconditional_smc_pass(
            &m, &m.default_params, &grid, 64, DT, &om, seed, &map, &plan,
            sim::rng::BinomialAlgorithm::Btpe,
        )
        .unwrap()
        {
            init += usize::from(p.initial_counts[1] >= 4);
        }
    }
    assert!(swept >= 8, "the sweep must weight by the t_start reading: {swept}/10 start at I >= 4");
    assert!(init >= 8, "the init filter must weight by it: {init}/10 start at I >= 4");
}
