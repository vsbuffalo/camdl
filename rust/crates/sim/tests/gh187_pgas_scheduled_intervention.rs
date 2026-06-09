//! gh#187 REGRESSION — the PGAS producer/CSMC path MUST apply scheduled
//! (non-`always_active`) interventions, not only `always_active` events.
//! No other test pins this: `cross_backend_lifecycle_agreement` excludes PGAS
//! and `pgas_event_density` covers only always-active events — the gap that let
//! gh#187 go silent.
//!
//! gh#187 (labeled `blocker`) claimed the PGAS producer advances state only via
//! `chain_binomial::step_one`, whose sole intervention mechanism was
//! `inject_event_deltas` (always_active events only), so `apply_interventions_at`
//! was never called in the PGAS path → a scheduled SIA transfer never fires in
//! the PGAS-produced latent trajectory.
//!
//! Since that issue, the effects path was unified (`due_effects` →
//! `apply_post_advance`): `step_one` now routes ALL due interventions through
//! `effect_batch`, splitting `always_active` → `event_idx` and scheduled →
//! `intervention_idx`, and applies BOTH. `simulate_reference` (the PGAS producer)
//! resolves `fire_steps` over ALL interventions and calls the same `step_one`.
//!
//! This test settles the question with actual counts, not code reading.
//!
//! Fixture: `event_intervention_agree.ir.json` (same one
//! `cross_backend_lifecycle_agreement` uses). At t=5:
//!   start                                A=50,  B=0
//!   always_active event add(A,100):      A=150, B=0
//!   SCHEDULED intervention transfer
//!     floor(A*keep)=floor(150*0.5)=75:   A=75,  B=75   <- if scheduled fires
//!
//! Signatures the producer trajectory can land on AFTER t=5:
//!   A=75,  B=75   => scheduled transfer FIRED in the PGAS latent trajectory.
//!   A=150, B=0    => scheduled intervention SKIPPED (the gh#187 bug reproduces:
//!                    only the always_active event applied).
//!   A=125, B=25   => pre-M1 inverted lifecycle order (transfer-before-event).
//!
//! `k=0` ⇒ the `drain : A --> B @ k*A` transition has rate ≡ 0, so there is NO
//! stochastic flow on any substep; the only state change is at t=5 and the
//! counts are integer-exact regardless of RNG seed.

use std::path::PathBuf;

use sim::compiled_model::CompiledModel;
use sim::inference::pgas::simulate_reference;
use sim::rng::StatefulRng;

fn fixture_ir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/corner_cases/ir/event_intervention_agree.ir.json")
}

fn load() -> CompiledModel {
    let path = fixture_ir();
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    let model: ir::Model = ir::from_str(&contents)
        .unwrap_or_else(|e| panic!("parse event_intervention_agree: {}", e));
    CompiledModel::new(model).unwrap_or_else(|e| panic!("compile: {:?}", e))
}

#[test]
fn gh187_pgas_applies_scheduled_intervention() {
    let compiled = load();

    // Fixture preconditions: exactly one scheduled (non-active) intervention and
    // one always_active event, both at t=5.
    let scheduled: Vec<_> = compiled.model.interventions.iter()
        .filter(|iv| !iv.always_active).collect();
    let active: Vec<_> = compiled.model.interventions.iter()
        .filter(|iv| iv.always_active).collect();
    assert_eq!(scheduled.len(), 1, "fixture should have one scheduled intervention");
    assert_eq!(active.len(), 1, "fixture should have one always_active event");
    eprintln!("[gh#187] scheduled intervention: {} (always_active=false)", scheduled[0].name);
    eprintln!("[gh#187] always_active event:     {} (always_active=true)", active[0].name);

    let ia = compiled.global_to_int[compiled.comp_index["A"]].expect("A integer");
    let ib = compiled.global_to_int[compiled.comp_index["B"]].expect("B integer");

    let params = compiled.default_params.clone();
    let dt = 1.0;
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let mut rng = StatefulRng::new(42);

    // Drive the PGAS producer directly.
    let traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Dump A,B across the t=5 boundary from the producer's own latent trajectory.
    eprintln!("[gh#187] PGAS producer (simulate_reference) latent trajectory:");
    eprintln!("  t       A     B");
    eprintln!("  init  {:4}  {:4}", traj.initial_counts[ia], traj.initial_counts[ib]);
    let mut last_a = traj.initial_counts[ia];
    let mut last_b = traj.initial_counts[ib];
    for (s, rec) in traj.substeps.iter().enumerate() {
        let t_after = t_start + (s as f64 + 1.0) * dt;
        let a = rec.counts_after[ia];
        let b = rec.counts_after[ib];
        eprintln!("  {:4.0}  {:4}  {:4}", t_after, a, b);
        last_a = a;
        last_b = b;
    }

    // Probe the post-boundary state: the substep that advances to t=5 carries the
    // event + scheduled-intervention application (fire keys on round(t/dt)=5).
    // counts_after at the substep ending at t=5 is the canonical post-lifecycle
    // state; it holds for all t >= 5 (zero rate, no further events).
    let s5 = ((5.0 - t_start) / dt - 1.0).round() as usize; // substep ending at t=5
    let a5 = traj.substeps[s5].counts_after[ia];
    let b5 = traj.substeps[s5].counts_after[ib];

    eprintln!("[gh#187] post-t=5 producer counts: A={a5}, B={b5}  (terminal A={last_a}, B={last_b})");

    // Negative control: the gh#187-bug signature is the scheduled intervention
    // being skipped (only the always_active event applied) → A=150, B=0.
    assert!(
        !(a5 == 150 && b5 == 0),
        "gh#187 REPRODUCES: scheduled intervention SKIPPED in the PGAS producer — \
         got A=150, B=0 (only the always_active add(A,100) applied; the scheduled \
         transfer never fired)."
    );
    // The other wrong-but-not-gh187 outcome: inverted lifecycle order.
    assert!(
        !(a5 == 125 && b5 == 25),
        "pre-M1 inverted lifecycle order (transfer-before-event): A=125, B=25."
    );
    // Correct canonical lifecycle: scheduled transfer fired on the post-event state.
    assert!(
        a5 == 75 && b5 == 75,
        "expected canonical A=75, B=75 (scheduled transfer fired on post-event A=150), \
         got A={a5}, B={b5}"
    );

    // Terminal state must persist (zero rate after t=5).
    assert_eq!((last_a, last_b), (75, 75), "post-t=5 state must persist to t_end");
}
