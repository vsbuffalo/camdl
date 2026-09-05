//! Byte-identical A/B gate for the unconditional initialization pass
//! (`pgas_init::unconditional_smc_pass`).
//!
//! The pass weights the swarm at each observation by folding the interval's
//! per-transition flows into the per-stream bins, scoring, and then zeroing both
//! the per-transition tally and the bins that are due. Splitting that single
//! closure into separate score / check / reset passes — so the collapse check
//! can still read the accumulator it is diagnosing — moves no arithmetic, but it
//! reorders WHEN the reset runs relative to the check. A reset that fired for
//! the wrong particles, or on the wrong schedule, would change the weights at
//! every later observation and therefore the lineage the pass returns.
//!
//! That is what this gate pins, at a fixed seed: the returned trajectory
//! (integer counts, integer flows, overdispersion multipliers, grid) and the
//! complete-data log-density of that trajectory, both bit-for-bit.
//!
//! The fixture binds an INCIDENCE stream deliberately. A prevalence-only model
//! owns no `acc` slot (`n_interval_streams() == 0`), so `fold_into_acc` and
//! `reset_due_acc` are both no-ops there and a gate built on one would pass
//! whatever the reset did.
//!
//! Baselines are captured on the dev machine and are a ratchet, like the sibling
//! `gate_pgas_density_baseline` / `gate_trajectory_baseline` files. Re-capture:
//!   CAMDL_CAPTURE_BASELINE=1 cargo test -p sim --test gate_unconditional_pass_ab -- --nocapture

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{
    build_obs_at_substep, complete_data_loglik, simulate_reference_on_grid, EffectFiring,
    ObsAtSubstep, PGASTrajectory,
};
use sim::inference::pgas_init::{unconditional_smc_pass, UnconditionalPass};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const N_SUBSTEPS: usize = 20;
const N_PARTICLES: usize = 200;
/// The seed the pass runs at. Its swarm keeps support through every
/// observation — asserted by `the_pass_returns_a_path_at_this_seed`.
const PASS_SEED: u64 = 3;
/// The seed whose forward draw supplies the observed series, so the data are
/// reachable at these parameters by construction.
const DATA_SEED: u64 = 0;

/// Local int index of `I` in `sir_basic` (`S`, `I`, `R`).
const I_IDX: usize = 1;
/// Substeps at which the incidence stream is observed; substep `s` spans
/// `(s·dt, (s+1)·dt]`, so its observation sits at `(s+1)·dt`.
const INCIDENCE_SUBSTEPS: [usize; 5] = [3, 7, 11, 15, 19];
/// The prevalence stream's own, coarser cadence — a strict subset, so the union
/// axis has union indices where only the incidence stream is scheduled.
const PREVALENCE_SUBSTEPS: [usize; 2] = [7, 15];

// ── fixture ────────────────────────────────────────────────────────────────

fn konst(v: f64) -> ir::expr::Expr {
    ir::expr::Expr::Const(ir::expr::ConstExpr { value: v })
}

fn projected() -> ir::expr::Expr {
    ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })
}

/// `projected + v`. The offset keeps a zero-projection particle from scoring
/// `-inf` against a positive count, so the gate measures the reset rather than
/// the swarm's survival.
fn projected_plus(v: f64) -> ir::expr::Expr {
    ir::expr::Expr::BinOp(ir::expr::BinOpWrap {
        bin_op: ir::expr::BinOpExpr {
            op: ir::expr::BinOp::Add,
            left: Box::new(projected()),
            right: Box::new(konst(v)),
        },
    })
}

fn obs_block(name: &str, projection: ir::observation::Projection) -> ir::observation::ObservationModel {
    use ir::observation::*;
    ir::observation::ObservationModel {
        name: name.into(),
        source: name.into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: name.into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: name.into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection,
        projection_state_grad: Default::default(),
        likelihood: Likelihood::NegBinomial(NegBinomialLikelihood {
            mean: ir::Diffable::new(projected_plus(0.5)),
            dispersion: ir::Diffable::new(konst(10.0)),
        }),
    }
}

fn model() -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![
        obs_block("incidence", ir::observation::Projection::CumulativeFlow("infection".into())),
        obs_block("prevalence", ir::observation::Projection::CurrentPop("I".into())),
    ];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = N_SUBSTEPS as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.35,
            "gamma" => 0.12,
            "N0" => 400.0,
            "I0" => 5.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

struct Fixture {
    compiled: Arc<CompiledModel>,
    params: Vec<f64>,
    union: Vec<Observation>,
    obs_model: MultiStreamObsModel,
    obs_at_substep: ObsAtSubstep,
    grid: Vec<(f64, f64)>,
}

fn fixture() -> Fixture {
    let compiled = model();
    let params = compiled.default_params.clone();
    let grid: Vec<(f64, f64)> = (0..N_SUBSTEPS).map(|s| (s as f64 * DT, DT)).collect();

    // The observed series is one forward realization at these same parameters,
    // so the data are reachable and the pass is measuring the reset rather than
    // an impossible fixture.
    let mut rng = StatefulRng::new(DATA_SEED);
    let truth = simulate_reference_on_grid(
        &compiled, &params, DT, &grid, EffectFiring::default(), &mut rng,
    )
    .expect("data-generating draw");

    let infection = compiled
        .model
        .transitions
        .iter()
        .position(|t| t.name == "infection")
        .expect("sir_basic has an `infection` transition");

    // Incidence: the infection flow summed over each observation interval.
    let mut incidence_values = Vec::new();
    let mut prev_end = 0usize;
    for &s in &INCIDENCE_SUBSTEPS {
        let bin: u64 = (prev_end..=s).map(|k| truth.substeps[k].flows[infection]).sum();
        incidence_values.push(bin as f64);
        prev_end = s + 1;
    }
    let prevalence_values: Vec<f64> = PREVALENCE_SUBSTEPS
        .iter()
        .map(|&s| truth.substeps[s].counts_after[I_IDX] as f64)
        .collect();

    let incidence_times: Vec<f64> =
        INCIDENCE_SUBSTEPS.iter().map(|&s| (s + 1) as f64 * DT).collect();
    let prevalence_times: Vec<f64> =
        PREVALENCE_SUBSTEPS.iter().map(|&s| (s + 1) as f64 * DT).collect();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![
            StreamSpec::dense(
                StreamProjection::FlowSum(vec![infection]),
                compiled.model.observations[0].clone(),
                dense_cells(incidence_values),
                incidence_times.clone(),
            ),
            StreamSpec::dense(
                StreamProjection::IntCompSum(vec![I_IDX]),
                compiled.model.observations[1].clone(),
                dense_cells(prevalence_values),
                prevalence_times,
            ),
        ])
        .expect("bind streams")
        .0,
        compiled.clone(),
    )
    .expect("build obs model");

    // The union axis is the incidence cadence here: the prevalence cadence is a
    // strict subset of it by construction (`PREVALENCE_SUBSTEPS ⊂
    // INCIDENCE_SUBSTEPS`), asserted below so a fixture edit cannot silently
    // break the union this map is built from.
    assert!(
        PREVALENCE_SUBSTEPS.iter().all(|s| INCIDENCE_SUBSTEPS.contains(s)),
        "the prevalence cadence must be a subset of the incidence cadence"
    );
    let union: Vec<Observation> = incidence_times
        .iter()
        .map(|&t| Observation { time: t, value: 0.0 })
        .collect();
    let obs_at_substep = build_obs_at_substep(&union, compiled.model.simulation.t_start, DT)
        .expect("obs_at_substep");

    Fixture { compiled, params, union, obs_model, obs_at_substep, grid }
}

impl Fixture {
    fn pass(&self, seed: u64) -> UnconditionalPass {
        unconditional_smc_pass(
            &self.compiled,
            &self.params,
            &self.grid,
            N_PARTICLES,
            DT,
            &self.obs_model,
            seed,
            &self.obs_at_substep,
            EffectFiring::default(),
            sim::rng::BinomialAlgorithm::Btpe,
        )
        .expect("unconditional pass")
    }

    fn path(&self, seed: u64) -> PGASTrajectory {
        match self.pass(seed) {
            UnconditionalPass::Path(t) => t,
            UnconditionalPass::NoSupport(r) => {
                panic!("the gate fixture must keep support at seed {seed}: {r}")
            }
        }
    }
}

// ── digest ─────────────────────────────────────────────────────────────────

/// FNV-1a over every byte of the trajectory. A hash rather than a per-substep
/// baseline table because "the lineage moved" is one fact, and the fields it
/// covers (counts before/after, integer flows, gamma multipliers, and the grid
/// each substep sits on) are exactly the ones a reset bug would move.
fn digest(traj: &PGASTrajectory) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let byte = |b: u8, h: &mut u64| {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x1000_0000_01b3);
    };
    let feed = |bytes: &[u8], h: &mut u64| {
        for &b in bytes {
            byte(b, h);
        }
    };
    for &c in &traj.initial_counts {
        feed(&c.to_le_bytes(), &mut h);
    }
    for rec in &traj.substeps {
        for &c in &rec.counts_before {
            feed(&c.to_le_bytes(), &mut h);
        }
        for &c in &rec.counts_after {
            feed(&c.to_le_bytes(), &mut h);
        }
        for &f in &rec.flows {
            feed(&f.to_le_bytes(), &mut h);
        }
        for &g in &rec.gammas {
            feed(&g.to_bits().to_le_bytes(), &mut h);
        }
        feed(&rec.t0.to_bits().to_le_bytes(), &mut h);
        feed(&rec.dt_substep.to_bits().to_le_bytes(), &mut h);
    }
    h
}

// ── the gate ───────────────────────────────────────────────────────────────

/// The fixture's premise: the swarm keeps support, so there IS a lineage and a
/// finite density to pin. A collapsed pass would make the baselines below
/// vacuous.
#[test]
fn the_pass_returns_a_path_at_this_seed() {
    let f = fixture();
    let traj = f.path(PASS_SEED);
    assert_eq!(traj.substeps.len(), N_SUBSTEPS, "the path must span the whole grid");
    assert!(
        f.obs_model.n_interval_streams() > 0,
        "the fixture must bind an INCIDENCE stream, else fold/reset are no-ops \
         and this gate measures nothing"
    );
}

/// Precondition for a byte-identical baseline: same seed → same everything.
#[test]
fn the_pass_is_deterministic_at_a_fixed_seed() {
    let f = fixture();
    let a = f.path(PASS_SEED);
    let b = f.path(PASS_SEED);
    assert_eq!(digest(&a), digest(&b), "the pass is not deterministic at a fixed seed");
}

/// Captured on the dev machine before the score/check/reset split.
const TRAJECTORY_DIGEST: u64 = 0xded6_2575_aaaa_6186;
const LL_TOTAL: f64 = -1.13420103811022386e2;
const LL_TRANSITION: f64 = -8.73213213905123666e1;
const LL_OBSERVATION: f64 = -2.60987824205100267e1;

#[test]
fn the_split_leaves_the_trajectory_and_the_log_likelihood_byte_identical() {
    let f = fixture();
    let traj = f.path(PASS_SEED);
    let ll = complete_data_loglik(
        &f.compiled,
        &traj,
        &f.params,
        &f.union,
        DT,
        &f.obs_model,
        &f.obs_at_substep,
    )
    .expect("complete_data_loglik");
    assert!(ll.total.is_finite(), "the pinned density must be finite, got {}", ll.total);

    let d = digest(&traj);
    if std::env::var("CAMDL_CAPTURE_BASELINE").is_ok() {
        eprintln!("\n// <<CAPTURED-BASELINES>>");
        eprintln!("const TRAJECTORY_DIGEST: u64 = {d:#018x};");
        eprintln!("const LL_TOTAL: f64 = {:.17e};", ll.total);
        eprintln!("const LL_TRANSITION: f64 = {:.17e};", ll.transition);
        eprintln!("const LL_OBSERVATION: f64 = {:.17e};", ll.observation);
        return;
    }

    assert_eq!(
        d, TRAJECTORY_DIGEST,
        "the unconditional pass returned a DIFFERENT lineage at seed {PASS_SEED} \
         ({d:#018x} vs {TRAJECTORY_DIGEST:#018x}) — the fold/score/reset order moved"
    );
    assert_eq!(
        ll.total.to_bits(),
        LL_TOTAL.to_bits(),
        "complete-data log-density moved: got {:.17e}, expected {LL_TOTAL:.17e}",
        ll.total
    );
    assert_eq!(
        ll.transition.to_bits(),
        LL_TRANSITION.to_bits(),
        "transition term moved: got {:.17e}, expected {LL_TRANSITION:.17e}",
        ll.transition
    );
    assert_eq!(
        ll.observation.to_bits(),
        LL_OBSERVATION.to_bits(),
        "observation term moved: got {:.17e}, expected {LL_OBSERVATION:.17e}",
        ll.observation
    );
}
