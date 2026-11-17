//! D77: PGAS's conditional sweep skips the resample exactly where the weights
//! are exactly equal, by the same test the unconditional filters use
//! (`resampling::log_weights_all_equal`), and nowhere else.
//!
//! Before D77 the sweep skipped wherever every unnormalised log weight was
//! within 1e-10 of the first one. That band held weights that differ: here a
//! Poisson reading of prevalence with observed count 0 and rate `1e-13 · I`
//! gives particle `j` the log weight `-1e-13 · I_j`, so particles with
//! different prevalence differ by multiples of 1e-13, well inside the old band
//! and many orders above the resolution of those doubles. Those weights are
//! not uniform, so the sweep must resample. A rate that does not read the
//! state gives every particle the same weight, and the sweep must not.
//!
//! The one-ulp edge of the shared test is pinned on the test itself, in
//! `resampling.rs` (`weights_that_differ_beyond_underflow_are_not_equal`).

use std::sync::Arc;

use sim::compiled_model::CompiledModel;
use sim::effect_plan::EffectPlan;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{build_obs_at_substep, csmc_as, simulate_reference, CSMCDiagnostics};
use sim::rng::StatefulRng;

const DT: f64 = 1.0;
const SEED: u64 = 20261118;
const I_IDX: usize = 1;
const N_SUBSTEPS: usize = 12;
/// Interior observations: each one's weights drive the following substep's
/// resample when they are not all equal.
const OBS_SUBSTEPS: [usize; 4] = [2, 4, 6, 8];

/// A Poisson reading of `I` whose rate is `rate`.
fn obs_block(rate: ir::expr::Expr) -> ir::observation::ObservationModel {
    use ir::observation::*;
    ObservationModel {
        name: "prevalence".into(),
        source: "prevalence".into(),
        columns: vec![
            ObsColumn { name: "time".into(), role: ColumnRole::Time },
            ObsColumn {
                name: "prevalence".into(),
                role: ColumnRole::Value(ir::parameter::ParamKind::Count),
            },
        ],
        scored: "prevalence".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        covers: None,
        projection: Projection::CurrentPop("I".into()),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::Poisson(PoissonLikelihood { rate: ir::Diffable::new(rate) }),
    }
}

/// `sir_basic` with the observation block replaced; `N0 = 400`, `I0 = 12`.
fn model(rate: ir::expr::Expr) -> Arc<CompiledModel> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../ocaml/golden/sir_basic.ir.json");
    let json = std::fs::read_to_string(&path).expect("read sir_basic golden");
    let mut m = ir::from_str(&json).expect("parse sir_basic");
    m.observations = vec![obs_block(rate)];
    m.simulation.t_start = 0.0;
    m.simulation.t_end = N_SUBSTEPS as f64 * DT;
    for p in &mut m.parameters {
        let v = match p.name.as_str() {
            "beta" => 0.35,
            "gamma" => 0.12,
            "N0" => 400.0,
            "I0" => 12.0,
            other => panic!("unexpected parameter {other}"),
        };
        p.value = ir::parameter::ParamValue::Fixed { value: v };
    }
    Arc::new(CompiledModel::new(m).expect("compile sir_basic"))
}

/// One conditional sweep against observed counts of 0 at `OBS_SUBSTEPS`.
fn sweep(rate: ir::expr::Expr, seed: u64) -> CSMCDiagnostics {
    let compiled = model(rate);
    let params = compiled.default_params.clone();
    let mut rng = StatefulRng::new(SEED);
    let reference = simulate_reference(&compiled, &params, N_SUBSTEPS as f64 * DT, DT, &mut rng)
        .expect("reference");
    let obs: Vec<Observation> = OBS_SUBSTEPS
        .iter()
        .map(|&s| Observation { time: (s + 1) as f64 * DT, value: 0.0 })
        .collect();
    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(0.0, vec![StreamSpec::dense(
            0.0,
            StreamProjection::IntCompSum(vec![I_IDX]),
            compiled.model.observations[0].clone(),
            dense_cells(obs.iter().map(|o| o.value).collect()),
            obs.iter().map(|o| o.time).collect(),
        )])
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap();
    let obs_at_substep = build_obs_at_substep(&obs, 0.0, DT).expect("obs_at_substep");
    let (_, d) = csmc_as(
        &compiled, &params, &obs, &reference, 16, DT, &obs_model, seed, &obs_at_substep,
        &EffectPlan::default(), sim::rng::BinomialAlgorithm::Btpe, true,
    )
    .expect("csmc_as");
    d
}

fn konst(value: f64) -> ir::expr::Expr {
    ir::expr::Expr::Const(ir::expr::ConstExpr { value })
}

/// Rate `1e-13 · I`: weights that differ by less than 1e-10, so they are not
/// equal and every interior observation drives a resample.
#[test]
fn weights_that_differ_below_the_old_tolerance_resample() {
    use ir::expr::*;
    let rate = Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
        op: BinOp::Mul,
        left: Box::new(konst(1e-13)),
        right: Box::new(Expr::Projected(ProjectedExpr { projected: () })),
    } });
    for seed in 0..4u64 {
        let d = sweep(rate.clone(), SEED + seed);
        assert_eq!(d.n_resampled, OBS_SUBSTEPS.len(),
            "seed {seed}: weights that differ (by about 1e-13 per infected) must be resampled \
             at every substep following an observation");
        assert_eq!(d.n_resampled + d.n_as_skipped_no_resample, d.n_substeps);
    }
}

/// A constant rate: every particle has exactly the same weight, so no
/// substep resamples, and ancestor sampling is suppressed everywhere.
#[test]
fn exactly_equal_weights_do_not_resample() {
    for seed in 0..4u64 {
        let d = sweep(konst(1e-13 * 12.0), SEED + seed);
        assert_eq!(d.n_resampled, 0, "seed {seed}: equal weights must not resample");
        assert_eq!(d.n_as_skipped_no_resample, d.n_substeps);
        assert_eq!(d.n_as_proposed, 0);
    }
}
