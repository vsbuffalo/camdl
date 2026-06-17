//! Cross-backend within-substep LIFECYCLE AGREEMENT (M1 canonicalization).
//!
//! The three forward backends (chain_binomial, ode, gillespie) must apply the
//! within-substep effects in the SAME canonical order:
//!
//!     transitions → always_active events (from the start-of-step snapshot)
//!                 → interventions (on the post-event state) → balance
//!
//! Before M1, chain_binomial used this order but ode/gillespie ran the
//! INVERTED order (interventions first, then events reading the post-
//! intervention state). The divergence is only observable when an event and an
//! intervention are coincident AND the intervention reads a compartment the
//! event modified — every existing golden lacks such a model, so the hash gates
//! could not catch the divergence. This test is the missing agreement invariant.
//!
//! Fixture: `tests/fixtures/corner_cases/event_intervention_agree.camdl`
//! (IR baked with k=0, keep=0.5). The single transition `drain : A --> B @ k*A`
//! has rate ≡ 0, so NO stochastic flow occurs on any backend and the only state
//! change at t=5 is the coincident event + intervention. Counts are therefore
//! integer-exact and identical across the deterministic ODE and the three
//! stochastic backends.
//!
//! Hand-computed canonical lifecycle at the t=5 boundary:
//!     start of step:                       A = 50,  B = 0
//!     event  add(A, 100):                  A = 150, B = 0
//!     intervention transfer floor(A*0.5):  delta = floor(150 * 0.5) = 75
//!                                          A = 75,  B = 75   (reads post-event A)
//!     => A = 75, B = 75 for all t >= 5
//!
//! The pre-M1 inverted order would give transfer-first (floor(50*0.5)=25 →
//! A=25, B=25) then add → A=125, B=25. That value (A=125, B=25) is the negative
//! control: a backend stuck on the old order fails this test loudly.

use std::path::PathBuf;

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

const SEED: u64 = 42;

/// Canonical post-substep counts, hand-computed (see module header).
const EXPECTED_A: i64 = 75;
const EXPECTED_B: i64 = 75;
/// The pre-M1 inverted order would produce these — used only to make the
/// negative control explicit in the failure message.
const INVERTED_A: i64 = 125;
const INVERTED_B: i64 = 25;

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

fn local_idx(compiled: &CompiledModel, name: &str) -> usize {
    let g = *compiled.comp_index.get(name).expect("compartment");
    compiled.global_to_int[g].expect("integer compartment")
}

/// Counts of A and B at the FINAL snapshot — the canonical post-lifecycle state,
/// identical on every backend. Final-state agreement is a necessary but not
/// sufficient cross-backend invariant; the full-trajectory check below
/// (`full_trajectory_no_pre_event_leak_or_time_reversal`) is the load-bearing one.
fn final_a_b(compiled: &CompiledModel, sim: &dyn Simulate, cfg: &SimConfig) -> (i64, i64) {
    let params = compiled.default_params.clone();
    let traj = sim
        .run(compiled, &params, SEED, cfg)
        .expect("forward sim must succeed (zero-rate model)");
    let last = traj.snapshots.last().expect("at least one snapshot");
    let ia = local_idx(compiled, "A");
    let ib = local_idx(compiled, "B");
    (last.int_state.counts[ia], last.int_state.counts[ib])
}

#[test]
fn all_backends_agree_on_coincident_event_intervention() {
    let compiled = load();
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;

    let backends: &[(&str, &dyn Simulate, SimConfig)] = &[
        (
            "chain_binomial",
            &ChainBinomialSim,
            SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 }),
        ),
        (
            "ode",
            &OdeSim,
            SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 }),
        ),
        (
            "gillespie",
            &GillespieSim,
            SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        ),
    ];

    for (name, sim, cfg) in backends {
        let (a, b) = final_a_b(&compiled, *sim, cfg);
        assert!(
            a == EXPECTED_A && b == EXPECTED_B,
            "{name}: within-substep lifecycle order DIVERGED — got A={a}, B={b}, \
             expected the canonical A={EXPECTED_A}, B={EXPECTED_B} (event from \
             snapshot BEFORE intervention). A={INVERTED_A}, B={INVERTED_B} would \
             mean this backend still runs the pre-M1 inverted order \
             (intervention before event)."
        );
    }
}

/// The pre-event compartment state: before the t=5 boundary nothing has fired,
/// so A is unchanged at its init value and B is empty on every backend.
const PRE_EVENT_A: i64 = 50;
const PRE_EVENT_B: i64 = 0;
/// The coincident event + intervention both fire at t=5 (see fixture header).
const EVENT_T: f64 = 5.0;

/// gh#70 regression: full-trajectory cross-backend invariant, not just the final
/// snapshot. Two properties every backend must satisfy:
///
///   1. **Time never runs backward** — snapshot `t` is strictly non-decreasing.
///   2. **No pre-event state leak** — for every snapshot at `t < 5`, the state is
///      the untouched init (`A == 50`, `B == 0`); the `add(A, 100)` event and the
///      transfer intervention fire at `t = 5`, so nothing may appear earlier.
///
/// Before the fix, gillespie's absorbing-state branch flushes outputs only up to
/// `next_special` and then jumps `t` to the event time, stranding the output
/// cursor; the next boundary clip pulls `t` *backward* and records the post-event
/// state (`A = 150`) at an earlier output row. This test fails on gillespie row
/// `t = 2` today. chain_binomial and ode are correct. `final_a_b` (above) only
/// probed `snapshots.last()` — the one row neither defect corrupts — which is why
/// this divergence slipped through. (Process post-mortem:
/// docs/dev/incidents/2026-06-16-gillespie-silent-wrong-test-sidestep.md.)
#[test]
fn full_trajectory_no_pre_event_leak_or_time_reversal() {
    let compiled = load();
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;

    let backends: &[(&str, &dyn Simulate, SimConfig)] = &[
        (
            "chain_binomial",
            &ChainBinomialSim,
            SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 }),
        ),
        (
            "ode",
            &OdeSim,
            SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 }),
        ),
        (
            "gillespie",
            &GillespieSim,
            SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        ),
    ];

    let params = compiled.default_params.clone();
    let ia = local_idx(&compiled, "A");
    let ib = local_idx(&compiled, "B");

    for (name, sim, cfg) in backends {
        let traj = sim
            .run(&compiled, &params, SEED, cfg)
            .expect("forward sim must succeed (zero-rate model)");

        for w in traj.snapshots.windows(2) {
            assert!(
                w[1].t >= w[0].t,
                "{name}: trajectory time ran backward — snapshot {} followed by {} \
                 (gh#70: the absorbing-state boundary clip jumped t into the past).",
                w[0].t, w[1].t
            );
        }

        for snap in &traj.snapshots {
            if snap.t < EVENT_T - 1e-9 {
                let a = snap.int_state.counts[ia];
                let b = snap.int_state.counts[ib];
                assert!(
                    a == PRE_EVENT_A && b == PRE_EVENT_B,
                    "{name}: pre-event state leaked at t={} — got A={a}, B={b}, \
                     expected the untouched init A={PRE_EVENT_A}, B={PRE_EVENT_B}. \
                     The add(A,100) event + transfer fire at t={EVENT_T}; recording \
                     post-event state earlier is gh#70 (gillespie back-fills the \
                     event into pre-event output rows).",
                    snap.t
                );
            }
        }
    }
}

// The chain-vs-tau differential oracle that pinned the within-substep event
// read-source (start-of-step snapshot vs post-drain) lived here; with tau-leap
// dropped (scheduling-spine-v2 §D) the property is covered by pgas_event_density
// + the lifecycle audit, which exercise the same chain-side fusion.

// ─────────────────────────────────────────────────────────────────────────────
// LAYER 3 (gh#233): the standing cross-backend battery — the net that must hold
// BEFORE the boundary-dispatch rewiring, so any future backend that bypasses the
// spine and re-derives boundaries by hand is caught. Two tiers:
//
//   Tier A (every fixture × 3 backends × multi-seed): per-backend trajectory
//   integrity — time strictly non-decreasing (the gh#70 catch: time ran
//   backward), no snapshot past t_end, non-negative counts, non-empty. These
//   hold regardless of stochastic flow, so the whole battery runs here.
//
//   Tier B (integer-exact fixtures only): the three backends must agree on the
//   final compartment counts. Zero-rate models have no stochastic flow, so the
//   post-lifecycle state is hand-checkable and identical across backends. (The
//   gh#70-class weakness of final-state-ONLY comparison is covered by Tier A's
//   per-backend full-trajectory monotonicity; Tier B complements it.)
//
// Cases map to the gh#233 task-6 list: coincident event+intervention + output-at-
// t_end (event_intervention_agree), absorbing-then-importation (gh70), multi-
// effect same time (multi_effect_same_time), off-grid effect (off_grid_
// intervention), fractional t_end (fractional_output_end).

/// `≥ 8` seeds: stochastic fixtures must satisfy Tier A on every one (gh#208 was
/// seed-dependent; a single-seed gate can land on a seed that doesn't bite).
const SEEDS: &[u64] = &[1, 2, 3, 5, 8, 13, 21, 42];

/// `(ir stem, integer_exact)`. `integer_exact` ⇒ rate ≡ 0, so no stochastic flow
/// and Tier B (cross-backend final-state agreement) applies.
const BATTERY: &[(&str, bool)] = &[
    ("event_intervention_agree", true),
    ("gh70_absorbing_importation", true),
    ("multi_effect_same_time", true),
    ("off_grid_intervention", false),
    ("fractional_output_end", false),
];

fn load_named(stem: &str) -> CompiledModel {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../../tests/fixtures/corner_cases/ir/{stem}.ir.json"));
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    let model: ir::Model =
        ir::from_str(&contents).unwrap_or_else(|e| panic!("parse {stem}: {}", e));
    CompiledModel::new(model).unwrap_or_else(|e| panic!("compile {stem}: {:?}", e))
}

/// The three forward backends with their per-backend config, at the model's window.
fn battery_backends(t_start: f64, t_end: f64) -> Vec<(&'static str, Box<dyn Simulate>, SimConfig)> {
    vec![
        (
            "chain_binomial",
            Box::new(ChainBinomialSim),
            SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: 1.0 }),
        ),
        ("ode", Box::new(OdeSim), SimConfig::Ode(OdeConfig { t_start, t_end, dt: 1.0 })),
        (
            "gillespie",
            Box::new(GillespieSim),
            SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        ),
    ]
}

#[test]
fn battery_per_backend_trajectory_invariants() {
    for &(stem, _) in BATTERY {
        let compiled = load_named(stem);
        let t_start = compiled.model.simulation.t_start;
        let t_end = compiled.model.simulation.t_end;
        let params = compiled.default_params.clone();

        for (name, sim, cfg) in battery_backends(t_start, t_end) {
            for &seed in SEEDS {
                let traj = sim.run(&compiled, &params, seed, &cfg).unwrap_or_else(|e| {
                    panic!("{stem}/{name} seed={seed}: forward sim must succeed: {e:?}")
                });
                assert!(!traj.snapshots.is_empty(), "{stem}/{name} seed={seed}: empty trajectory");

                // (1) time strictly non-decreasing — the gh#70 backward-jump catch.
                for w in traj.snapshots.windows(2) {
                    assert!(
                        w[1].t >= w[0].t,
                        "{stem}/{name} seed={seed}: trajectory time ran backward — \
                         snapshot t={} followed by t={} (gh#70: absorbing-state boundary \
                         clip jumped t into the past).",
                        w[0].t, w[1].t
                    );
                }

                for snap in &traj.snapshots {
                    // (2) no snapshot recorded past the run end.
                    assert!(
                        snap.t <= t_end + 1e-9,
                        "{stem}/{name} seed={seed}: snapshot at t={} is past t_end={t_end}",
                        snap.t
                    );
                    // (3) non-negative integer counts.
                    for (i, &c) in snap.int_state.counts.iter().enumerate() {
                        assert!(
                            c >= 0,
                            "{stem}/{name} seed={seed}: negative count {c} (compartment local \
                             idx {i}) at t={}",
                            snap.t
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn battery_integer_exact_cross_backend_final_agreement() {
    for &(stem, integer_exact) in BATTERY {
        if !integer_exact {
            continue;
        }
        let compiled = load_named(stem);
        let t_start = compiled.model.simulation.t_start;
        let t_end = compiled.model.simulation.t_end;
        let params = compiled.default_params.clone();

        // Final compartment-count vector per backend; all three must match.
        let mut reference: Option<(&str, Vec<i64>)> = None;
        for (name, sim, cfg) in battery_backends(t_start, t_end) {
            let traj = sim
                .run(&compiled, &params, SEEDS[0], &cfg)
                .unwrap_or_else(|e| panic!("{stem}/{name}: forward sim must succeed: {e:?}"));
            // An integer-exact fixture lands its final snapshot exactly on t_end
            // (the End+Output coincidence; integer t_end on the dt=1 grid).
            let last = traj.snapshots.last().expect("at least one snapshot");
            assert!(
                (last.t - t_end).abs() < 1e-9,
                "{stem}/{name}: final snapshot at t={}, expected t_end={t_end} \
                 (output-at-end coincidence)",
                last.t
            );
            let counts = last.int_state.counts.clone();
            match &reference {
                None => reference = Some((name, counts)),
                Some((ref_name, ref_counts)) => assert!(
                    &counts == ref_counts,
                    "{stem}: cross-backend final-state DIVERGENCE — {ref_name}={ref_counts:?} \
                     vs {name}={counts:?} (zero-rate model: every backend must agree).",
                ),
            }
        }
    }
}
