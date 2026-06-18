//! gh#204 PR2: integration coverage for the forward reactive observation
//! evaluator (`sim::reactive::ReactiveObs`) on a real compiled model — the
//! interval projection (slice 2) feeding the realized draw on a dedicated obs
//! RNG (slice 3). The committed reactive golden is the model under test.

use sim::reactive::ReactiveObs;
use sim::rng::StatefulRng;
use sim::state::{IntState, RealState};
use sim::CompiledModel;
use std::path::PathBuf;

fn compiled_reactive_sir() -> CompiledModel {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let mut model: ir::Model = ir::from_str(&s).unwrap_or_else(|e| panic!("deser fixture: {e:?}"));
    // The golden leaves params estimated; fill in-bounds placeholders (rho=0.5).
    for prm in &mut model.parameters {
        if prm.value.resolved_value().is_none() {
            prm.value = prm.value.with_value(0.5);
        }
    }
    CompiledModel::new(model).expect("compile reactive golden")
}

#[test]
fn reactive_obs_draw_is_reproducible_and_interval_fed() {
    let c = compiled_reactive_sir();
    let params = c.default_params.clone();
    let int_s = IntState::from_vec(vec![0; c.model.compartments.len()]);
    let real_s = RealState::new(0); // the fixture has no real compartments

    let mut ro =
        ReactiveObs::from_model(&c, &["weekly_cases".to_string()]).expect("build ReactiveObs");

    // `infection` is transition 0; accumulate 50 incidence over the interval.
    ro.accumulate(&[50, 0]);

    // Same obs-RNG seed → identical draw (reproducible; the obs stream is
    // deterministic given its seed, independent of the dynamics RNG).
    let mut rng_a = StatefulRng::new(7);
    let mut rng_b = StatefulRng::new(7);
    let ya = ro.draw(0, &int_s, &real_s, &params, &c, 7.0, &mut rng_a);
    let yb = ro.draw(0, &int_s, &real_s, &params, &c, 7.0, &mut rng_b);
    assert_eq!(ya, yb, "draw must be reproducible for a fixed obs-RNG seed");
    assert!(ya >= 0.0 && ya.is_finite(), "a Poisson report draw is a finite count");

    // After resetting the interval, projected incidence is 0, so the report
    // draw from poisson(rho * 0) is 0 — proving the draw is fed by the
    // (reset) interval accumulator, not stale or output-tied flow.
    ro.reset(0);
    let mut rng_c = StatefulRng::new(7);
    let yc = ro.draw(0, &int_s, &real_s, &params, &c, 14.0, &mut rng_c);
    assert_eq!(yc, 0.0, "no accumulated incidence ⇒ the report draw is 0");
}
