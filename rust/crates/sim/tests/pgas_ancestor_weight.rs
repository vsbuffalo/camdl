//! Regression test for the CSMC-AS ancestor-sampling weight.
//!
//! Lindsten, Jordan & Schön (2014), "Particle Gibbs with Ancestor Sampling",
//! JMLR 15:2145–2184, Eq. (17): the reference's ancestor is drawn with weight
//!
//!   log w̃_j = log w_{s-1}^j + log f_θ(x'_s | x_{s-1}^j)
//!
//! i.e. the previous-substep importance weight `log_weights[j]` PLUS the
//! transition density `td`. A bug dropped `log_weights[j]`, scoring the ancestor
//! draw on transition density alone — biasing it at every substep whose incoming
//! weights are non-uniform (the substep after each observation) and forfeiting
//! the Theorem-1 invariance of the PGAS kernel (the default Bayesian method).
//!
//! This test pins the exact Eq-(17) weight so the dropped term cannot silently
//! return: with a non-uniform `log_weights`, `td` alone ≠ `log_weights[j] + td`.

use std::path::PathBuf;
use std::sync::Arc;

use sim::{
    compiled_model::CompiledModel,
    inference::pgas::{
        fill_ancestor_log_weights, log_transition_density_substep, simulate_reference,
    },
    rng::StatefulRng,
};

fn load_model() -> Arc<CompiledModel> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/corner_cases/ir/seasonal_drift.ir.json");
    let contents =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    let model = ir::from_str(&contents).expect("parse seasonal_drift IR");
    Arc::new(CompiledModel::new(model).expect("compile seasonal_drift"))
}

#[test]
fn ancestor_weight_includes_importance_weight() {
    let compiled = load_model();
    let params = compiled.default_params.clone();
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let dt = 1.0;

    // A deterministic reference gives us valid (finite-density) substep records.
    let mut rng = StatefulRng::new(7);
    let traj = simulate_reference(&compiled, &params, t_end, dt, sim::rng::BinomialAlgorithm::default(), &mut rng)
        .expect("simulate_reference on seasonal_drift");
    assert!(traj.substeps.len() >= 2, "need a couple of substeps");

    let s = 3.min(traj.substeps.len() - 1);
    let rec = &traj.substeps[s];
    let t = t_start + s as f64 * dt;

    // Two candidate ancestor states over N = 2 particles (slot 1 = reference).
    // Reference slot uses its own counts_before. The free slot uses a perturbed
    // state (add to the largest compartment so the reference's flows stay
    // feasible → finite density, but the rate — and hence td — differs).
    let ref_counts_before: Vec<i64> = rec.counts_before.clone();
    let mut free_state = ref_counts_before.clone();
    let big = free_state
        .iter()
        .enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .expect("nonempty state");
    free_state[big] += 50;

    let n = 2usize;
    // Index 0 = the free particle's pre-resample state; index 1 = the reference
    // slot's, which on a sweep with no accepted splice is its recorded
    // `counts_before`. Every slot is scored at its own entry.
    let prev_counts_for_ancestor = vec![free_state.clone(), ref_counts_before.clone()];

    // Sharply NON-UNIFORM incoming weights, as after an observation. If these
    // were uniform the dropped term would be a harmless constant.
    let log_weights = vec![-0.3_f64, -1.7_f64];
    assert!(
        (log_weights[0] - log_weights[1]).abs() > 1e-6,
        "weights must be non-uniform or the test is vacuous"
    );

    // Independently compute the two transition densities via the public fn.
    let td_free = log_transition_density_substep(
        &compiled, &free_state, &rec.flows, &rec.gammas, &params, t, dt, None,
    )
    .expect("finite free-slot density");
    let td_ref = log_transition_density_substep(
        &compiled,
        &ref_counts_before,
        &rec.flows,
        &rec.gammas,
        &params,
        t,
        dt,
        None,
    )
    .expect("finite reference-slot density");
    assert!(
        td_free.is_finite() && td_ref.is_finite(),
        "both densities must be finite (td_free={td_free}, td_ref={td_ref})"
    );

    let mut anc = vec![0.0_f64; n];
    fill_ancestor_log_weights(
        &mut anc,
        &compiled,
        &prev_counts_for_ancestor,
        &rec.flows,
        &rec.gammas,
        &log_weights,
        &params,
        t,
        dt,
        None,
    )
    .expect("fill_ancestor_log_weights");

    // Eq (17): each slot's weight is log_weights[j] + td_j. The bug (td alone)
    // omits the log_weights[j] offset, so these asserts fail against it.
    let expected_free = log_weights[0] + td_free;
    let expected_ref = log_weights[1] + td_ref;
    assert!(
        (anc[0] - expected_free).abs() < 1e-9,
        "free-slot weight = {} but Eq(17) requires log_w + td = {} \
         (the bug drops the log_w = {} importance-weight term)",
        anc[0],
        expected_free,
        log_weights[0]
    );
    assert!(
        (anc[1] - expected_ref).abs() < 1e-9,
        "reference-slot weight = {} but Eq(17) requires log_w + td = {}",
        anc[1],
        expected_ref
    );

    // Guard against a vacuous pass: the correct weight must actually differ from
    // td-alone here (i.e. the importance-weight term is non-zero).
    assert!(
        (expected_free - td_free).abs() > 1e-6 && (expected_ref - td_ref).abs() > 1e-6,
        "importance-weight term must be non-trivial for this test to bite"
    );
}
