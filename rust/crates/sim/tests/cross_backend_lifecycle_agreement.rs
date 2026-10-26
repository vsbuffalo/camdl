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

use ir::{
    expr::Expr,
    intervention::{
        AbsoluteTransfer, Action, AddAction, FireSource, FractionTransfer, Intervention,
        InterventionKind, InterventionSchedule, SetAction,
    },
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use std::collections::HashMap;

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
    ("origin_effects", true),
    ("recurring_default_from", true),
    ("offset_origin", true),
    ("inflow_drain_origin", true),
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

// ─────────────────────────────────────────────────────────────────────────────
// gh#198, under the per-backend rule (D2): two DISTINCT fire times within one
// `dt` (`at: [2.3, 2.4]` at `dt = 1`). Chain-binomial is discrete-time: both
// times key to the step ending at 2 (k = ⌊t + ½⌋ = 2), and one intervention
// fires once per step, so it fires ONCE. The ODE and Gillespie are
// continuous-time: each declared time is its own instant, so the intervention
// fires TWICE, at 2.3 and at 2.4. Two interventions at 2.3 and 2.4 each fire
// once on every backend. (Before the plan, the exact backends merged the two
// times onto one rounded step to agree with chain-binomial, which fired an
// effect at 2.4 early, at 2.3 — and at an earlier landing at 2.0.)

/// S=100, V=0 integer model with a single zero-rate transition `S --> V @ 0`
/// (rate ≡ 0 ⇒ no stochastic flow ⇒ integer-exact, identical on every backend),
/// carrying `interventions`. Output at every integer t in [0, 5], dt=1, so the
/// final snapshot is the post-intervention state.
fn collision_model(interventions: Vec<Intervention>) -> CompiledModel {
    zero_rate_model("gh198_within_dt_collision", ("S", 100.0), ("V", 0.0), interventions)
}

/// Two integer compartments `src`, `dst` with their initial counts and a single
/// zero-rate transition `src --> dst @ 0`, so the only state changes are the
/// effects in `interventions`. Output at every integer t in [0, 5], dt=1.
fn zero_rate_model(
    name: &str,
    src: (&str, f64),
    dst: (&str, f64),
    interventions: Vec<Intervention>,
) -> CompiledModel {
    let m = Model {
        ic_grad: Default::default(),
        name: name.into(),
        version: "0.1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: src.0.into(), kind: CompartmentKind::Integer },
            Compartment { name: dst.0.into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "noop".into(),
            stoichiometry: vec![
                StoichiometryEntry(src.0.into(), -1),
                StoichiometryEntry(dst.0.into(), 1),
            ],
            rate: Expr::const_(0.0),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions,
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "p".into(),
            value: ParamValue::Fixed { value: 1.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert(src.0.into(), src.1);
            h.insert(dst.0.into(), dst.1);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 5.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(1),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    };
    CompiledModel::new(m).unwrap_or_else(|e| panic!("compile {name}: {e:?}"))
}

/// A scheduled (`!always_active`) intervention firing `action` at `at`.
fn scheduled(name: &str, at: Vec<f64>, action: Action) -> Intervention {
    Intervention {
        name: name.into(),
        base_name: None,
        fire: FireSource::Scheduled(InterventionSchedule::AtTimes(at)),
        actions: vec![action],
        kind: InterventionKind::Scenario,
    }
}

/// Final (S, V) counts on `sim` for `compiled` at dt=1 / t_end=5.
fn collision_final_s_v(
    compiled: &CompiledModel,
    sim: &dyn Simulate,
    cfg: &SimConfig,
) -> (i64, i64) {
    let params = compiled.default_params.clone();
    let traj = sim
        .run(compiled, &params, SEED, cfg)
        .expect("forward sim (zero-rate) must succeed");
    let last = traj.snapshots.last().expect("at least one snapshot");
    let is = local_idx(compiled, "S");
    let iv = local_idx(compiled, "V");
    (last.int_state.counts[is], last.int_state.counts[iv])
}

/// Whether `backend` runs on a Grid clock (fires on the step) or an Exact one
/// (fires at the declared instant).
fn is_grid(backend: &str) -> bool {
    backend == "chain_binomial"
}

/// gh#198 (i), Add: ONE intervention firing `add(V, 10)` at both 2.3 and 2.4
/// (dt=1). Chain-binomial fires it once, on the step ending at 2 (V=10); the ODE
/// and Gillespie fire it once per declared time (V=20).
#[test]
fn within_dt_pair_fires_once_per_step_on_grid_and_once_per_time_on_exact() {
    let compiled = collision_model(vec![scheduled(
        "campaign",
        vec![2.3, 2.4],
        Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(10.0) }),
    )]);
    for (name, sim, cfg) in battery_backends(0.0, 5.0) {
        let (s, v) = collision_final_s_v(&compiled, &*sim, &cfg);
        let expected = if is_grid(name) { (100, 10) } else { (100, 20) };
        assert_eq!(
            (s, v),
            expected,
            "{name}: add(V,10) at [2.3,2.4] (dt=1) fires once per step on the Grid \
             clock (V=10) and once per declared instant on the Exact clock (V=20)."
        );
    }
}

/// gh#198 (i), Transfer: ONE intervention moving `transfer(S -> V, 30)` at both
/// 2.3 and 2.4. Chain-binomial moves 30 once (S=70, V=30); the ODE and Gillespie
/// move 30 at each instant (S=40, V=60).
#[test]
fn within_dt_pair_transfer_fires_once_per_step_on_grid_and_once_per_time_on_exact() {
    let compiled = collision_model(vec![scheduled(
        "vaccinate",
        vec![2.3, 2.4],
        Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "S".into(),
            dst: "V".into(),
            count: Expr::const_(30.0),
        }),
    )]);
    for (name, sim, cfg) in battery_backends(0.0, 5.0) {
        let (s, v) = collision_final_s_v(&compiled, &*sim, &cfg);
        let expected = if is_grid(name) { (70, 30) } else { (40, 60) };
        assert_eq!(
            (s, v),
            expected,
            "{name}: transfer S->V of 30 at [2.3,2.4] (dt=1): once per step on the Grid \
             clock (S=70, V=30), once per instant on the Exact clock (S=40, V=60)."
        );
    }
}

/// gh#198 (ii), the cousin: TWO separate interventions, at `[2.3]` (add 10) and
/// `[2.4]` (add 100). Each fires exactly once on every backend: V=110.
#[test]
fn within_dt_collision_two_interventions_each_fire_once() {
    let compiled = collision_model(vec![
        scheduled(
            "a",
            vec![2.3],
            Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(10.0) }),
        ),
        scheduled(
            "b",
            vec![2.4],
            Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(100.0) }),
        ),
    ]);
    for (name, sim, cfg) in battery_backends(0.0, 5.0) {
        let (_s, v) = collision_final_s_v(&compiled, &*sim, &cfg);
        assert_eq!(
            v, 110,
            "{name}: interventions at 2.3 (add 10) and 2.4 (add 100), dt=1 → each must \
             fire ONCE (V=110)."
        );
    }
}

/// `V` at every output row (t = 0, 1, ..., 5) on `sim`.
fn v_rows(compiled: &CompiledModel, sim: &dyn Simulate, cfg: &SimConfig) -> Vec<(f64, i64)> {
    let params = compiled.default_params.clone();
    let traj = sim.run(compiled, &params, SEED, cfg).expect("forward sim (zero-rate)");
    let iv = local_idx(compiled, "V");
    traj.snapshots.iter().map(|s| (s.t, s.int_state.counts[iv])).collect()
}

/// An effect at 2.4 does not fire at an earlier landing at 2.0 on the exact
/// backends: with `a` at 2.0 (add 10) and `b` at 2.4 (add 100), the t = 2 row
/// carries `a` only and `b` first appears at t = 3. On chain-binomial both key
/// to the step ending at 2 (2.4 + ½ floors to 2), so the t = 2 row carries both.
#[test]
fn an_effect_does_not_fire_at_an_earlier_landing_in_its_step() {
    let compiled = collision_model(vec![
        scheduled(
            "a",
            vec![2.0],
            Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(10.0) }),
        ),
        scheduled(
            "b",
            vec![2.4],
            Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(100.0) }),
        ),
    ]);
    for (name, sim, cfg) in battery_backends(0.0, 5.0) {
        let rows = v_rows(&compiled, &*sim, &cfg);
        let expected: Vec<(f64, i64)> = if is_grid(name) {
            vec![(0.0, 0), (1.0, 0), (2.0, 110), (3.0, 110), (4.0, 110), (5.0, 110)]
        } else {
            vec![(0.0, 0), (1.0, 0), (2.0, 10), (3.0, 110), (4.0, 110), (5.0, 110)]
        };
        assert_eq!(rows, expected, "{name}: V rows");
    }
}

/// An off-grid effect fires at `t_start + fire_step(t)·dt` under chain-binomial
/// and at `t` under the ODE and Gillespie: `add(V, 10)` at 2.4 is in the t = 2
/// row on chain-binomial (k = ⌊2.4 + ½⌋ = 2) and first in the t = 3 row on the
/// exact backends. At 2.6 it keys to the step ending at 3 on every backend.
#[test]
fn an_off_grid_effect_fires_per_backend_clock() {
    for (at, grid_row) in [(2.4, 2.0), (2.6, 3.0)] {
        let compiled = collision_model(vec![scheduled(
            "pulse",
            vec![at],
            Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(10.0) }),
        )]);
        for (name, sim, cfg) in battery_backends(0.0, 5.0) {
            let first = if is_grid(name) { grid_row } else { at.ceil() };
            let rows = v_rows(&compiled, &*sim, &cfg);
            for (t, v) in rows {
                let want = if t >= first - 1e-9 { 10 } else { 0 };
                assert_eq!(v, want, "{name}: effect at {at}: V at t={t}");
            }
        }
    }
}

/// Non-regression: an ON-GRID fire time (exact dt multiple) still fires exactly
/// once. `add(V, 10)` at t=3.0 (dt=1, step 3) → V=10 on all three backends. Pins
/// that the step-keyed cursor advance does not over- or under-fire the common case.
#[test]
fn on_grid_intervention_still_fires_once() {
    let compiled = collision_model(vec![scheduled(
        "campaign",
        vec![3.0],
        Action::Add(AddAction { compartment: "V".into(), count: Expr::const_(10.0) }),
    )]);
    for (name, sim, cfg) in battery_backends(0.0, 5.0) {
        let (s, v) = collision_final_s_v(&compiled, &*sim, &cfg);
        assert_eq!(
            (s, v),
            (100, 10),
            "{name}: add(V,10) at on-grid t=3.0 must fire exactly once (S=100, V=10)."
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// gh#985 / D4: coincident EVENTS at one instant. Language spec §13.10: inflow
// events (`add`) resolve against the start-of-step snapshot; draining
// `transfer` events and `set` resolve against the state after the inflows (and
// after the transition draw, where there is one). The same rule holds at every
// effect boundary on every backend, so declaration order never matters, a drain
// cannot overdraw, and `set` means set. Each case runs both declaration orders
// on all three backends; the zero-rate model makes the result hand-checkable.

/// An always-active event (`events {}`) firing `action` at `at`.
fn event(name: &str, at: Vec<f64>, action: Action) -> Intervention {
    Intervention {
        name: name.into(),
        base_name: None,
        fire: FireSource::Scheduled(InterventionSchedule::AtTimes(at)),
        actions: vec![action],
        kind: InterventionKind::Event,
    }
}

/// Final counts of the model's two compartments, `(first, second)`, on every
/// backend, for the events declared in `forward` order and in reverse order.
fn coincident_finals(
    name: &str,
    src: (&str, f64),
    dst: (&str, f64),
    forward: Vec<Intervention>,
) -> Vec<(String, (i64, i64))> {
    let mut reversed = forward.clone();
    reversed.reverse();
    let mut out = Vec::new();
    for (order, ivs) in [("declared", forward), ("reversed", reversed)] {
        let compiled = zero_rate_model(name, src, dst, ivs);
        let a = local_idx(&compiled, src.0);
        let b = local_idx(&compiled, dst.0);
        for (backend, sim, cfg) in battery_backends(0.0, 5.0) {
            let params = compiled.default_params.clone();
            let traj = sim
                .run(&compiled, &params, SEED, &cfg)
                .unwrap_or_else(|e| panic!("{name}/{backend}/{order}: forward sim: {e:?}"));
            let last = traj.snapshots.last().expect("at least one snapshot");
            out.push((
                format!("{backend} ({order} order)"),
                (last.int_state.counts[a], last.int_state.counts[b]),
            ));
        }
    }
    out
}

/// Assert every `(backend, order)` cell gave `expected`, listing every cell's
/// result on failure so the whole backend × order table is visible at once.
fn assert_every_backend(finals: &[(String, (i64, i64))], expected: (i64, i64), rule: &str) {
    let wrong: Vec<_> = finals.iter().filter(|(_, got)| *got != expected).collect();
    assert!(
        wrong.is_empty(),
        "{rule}\nexpected {expected:?} on every backend and order; got:\n{}",
        finals
            .iter()
            .map(|(label, got)| format!(
                "  {label}: {got:?}{}",
                if *got == expected { "" } else { "  <- wrong" }
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// `add(X, 10)` and `set X = 5` at t=3 give X = 5 on every backend, in either
/// declaration order. Hand computation: X = 0 → inflow X = 10 → `set` reads
/// the post-inflow state and overwrites it, X = 5. Summing both deltas against
/// the pre-event state gives 0 + 10 + (5 − 0) = 15 (Gillespie before gh#985);
/// applying them one at a time gives 15 when `set` is declared first (the ODE
/// before gh#985).
#[test]
fn coincident_add_and_set_event_gives_the_set_value_on_every_backend() {
    let finals = coincident_finals(
        "gh985_add_set",
        ("X", 0.0),
        ("Y", 0.0),
        vec![
            event(
                "arrivals",
                vec![3.0],
                Action::Add(AddAction { compartment: "X".into(), count: Expr::const_(10.0) }),
            ),
            event(
                "reset",
                vec![3.0],
                Action::Set(SetAction { compartment: "X".into(), value: Expr::const_(5.0) }),
            ),
        ],
    );
    assert_every_backend(&finals, (5, 0),
        "add(X, 10) and set X = 5 at one instant must give X = 5 \
             (§13.10: set reads the post-inflow state); X = 15 means the set was \
             resolved against the pre-inflow state (gh#985).");
}

/// D4's example, mid-run: `seed : add(I, 10)` and
/// `cull : transfer(fraction = 0.5, from = I, to = R)` at t=3 from I = R = 0
/// give I = 5, R = 5 on every backend, in either order. Hand computation:
/// inflow I = 10 → the drain reads the post-inflow I, floor(10 · 0.5) = 5.
/// Reading the pre-inflow I = 0 moves nothing (I = 10, R = 0).
#[test]
fn coincident_add_and_fraction_drain_event_reads_post_inflow_state() {
    let finals = coincident_finals(
        "gh985_add_fraction_drain",
        ("I", 0.0),
        ("R", 0.0),
        vec![
            event(
                "seed",
                vec![3.0],
                Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(10.0) }),
            ),
            event(
                "cull",
                vec![3.0],
                Action::FractionTransfer(FractionTransfer {
                    src: "I".into(),
                    dst: "R".into(),
                    fraction: Expr::const_(0.5),
                }),
            ),
        ],
    );
    assert_every_backend(&finals, (5, 5),
        "add(I, 10) and transfer(fraction = 0.5, from = I, to = R) at \
             one instant must give I = 5, R = 5 (§13.10: the drain reads the \
             post-inflow state); I = 10, R = 0 means it read the pre-inflow \
             state (gh#985).");
}

/// The count form of the drain: `add(I, 10)` and
/// `transfer(count = 4, from = I, to = R)` at t=3 from I = R = 0 give I = 6,
/// R = 4 on every backend, in either order. The count is clamped to the
/// source it reads, so reading the pre-inflow I = 0 moves nothing.
#[test]
fn coincident_add_and_count_drain_event_reads_post_inflow_state() {
    let finals = coincident_finals(
        "gh985_add_count_drain",
        ("I", 0.0),
        ("R", 0.0),
        vec![
            event(
                "seed",
                vec![3.0],
                Action::Add(AddAction { compartment: "I".into(), count: Expr::const_(10.0) }),
            ),
            event(
                "cull",
                vec![3.0],
                Action::AbsoluteTransfer(AbsoluteTransfer {
                    src: "I".into(),
                    dst: "R".into(),
                    count: Expr::const_(4.0),
                }),
            ),
        ],
    );
    assert_every_backend(&finals, (6, 4),
        "add(I, 10) and transfer(count = 4, from = I, to = R) at one \
             instant must give I = 6, R = 4 (§13.10: the drain reads the \
             post-inflow state); I = 10, R = 0 means it read the pre-inflow \
             state (gh#985).");
}

/// `set X = 5` and `transfer(fraction = 0.5, from = X, to = Y)` at t=3 from
/// X = 10, Y = 0 give X = 0, Y = 5 on every backend, in either order. Both are
/// drain-and-`set` actions, so both resolve against the same post-inflow state
/// and their deltas sum: the `set` contributes 5 − 10 = −5, the drain
/// −floor(10 · 0.5) = −5 to X and +5 to Y. Applying them one at a time instead
/// gives X = 5 when the drain comes first and X = 0 when the `set` does (the ODE
/// before this case was pinned).
#[test]
fn coincident_set_and_drain_on_one_compartment_sum_their_deltas() {
    let finals = coincident_finals(
        "gh985_set_drain_same_compartment",
        ("X", 10.0),
        ("Y", 0.0),
        vec![
            event(
                "reset",
                vec![3.0],
                Action::Set(SetAction { compartment: "X".into(), value: Expr::const_(5.0) }),
            ),
            event(
                "cull",
                vec![3.0],
                Action::FractionTransfer(FractionTransfer {
                    src: "X".into(),
                    dst: "Y".into(),
                    fraction: Expr::const_(0.5),
                }),
            ),
        ],
    );
    assert_every_backend(&finals, (0, 5),
        "set X = 5 and transfer(fraction = 0.5, from = X, to = Y) at one instant from X = 10 must give X = 0, Y = 5 (§13.10: both resolve against the same state and their deltas sum); X = 5 means the actions were applied one at a time in declaration order.");
}

/// Run `forward` and its reverse on every backend and return each run's result
/// (the final two counts, or the error), labelled by backend and order.
fn coincident_results(
    name: &str,
    src: (&str, f64),
    dst: (&str, f64),
    forward: Vec<Intervention>,
) -> Vec<(String, Result<(i64, i64), sim::error::SimError>)> {
    let mut reversed = forward.clone();
    reversed.reverse();
    let mut out = Vec::new();
    for (order, ivs) in [("declared", forward), ("reversed", reversed)] {
        let compiled = zero_rate_model(name, src, dst, ivs);
        let a = local_idx(&compiled, src.0);
        let b = local_idx(&compiled, dst.0);
        for (backend, sim, cfg) in battery_backends(0.0, 5.0) {
            let params = compiled.default_params.clone();
            let res = sim.run(&compiled, &params, SEED, &cfg).map(|traj| {
                let last = traj.snapshots.last().expect("at least one snapshot");
                (last.int_state.counts[a], last.int_state.counts[b])
            });
            out.push((format!("{backend} ({order} order)"), res));
        }
    }
    out
}

fn set_event(name: &str, comp: &str, v: f64) -> Intervention {
    event(
        name,
        vec![3.0],
        Action::Set(SetAction { compartment: comp.into(), value: Expr::const_(v) }),
    )
}

fn drain_event(name: &str, frac: f64) -> Intervention {
    event(
        name,
        vec![3.0],
        Action::FractionTransfer(FractionTransfer {
            src: "X".into(),
            dst: "Y".into(),
            fraction: Expr::const_(frac),
        }),
    )
}

/// Two `set` events on one compartment at one instant are refused on every
/// backend, with an error naming the compartment, both events and the time.
/// Summing them gives `v1 + v2 − x`, which means nothing: from X = 5 two sets
/// to 7 would silently give 9, and from X = 10 two sets to 2 give −6, which
/// the ODE used to clamp to 0 in its output.
#[test]
fn two_sets_on_one_compartment_at_one_instant_are_refused() {
    let mut failures = Vec::new();
    for (x0, v) in [(5.0, 7.0), (10.0, 2.0)] {
        let results = coincident_results(
            "gh985_two_sets",
            ("X", x0),
            ("Y", 0.0),
            vec![set_event("first", "X", v), set_event("second", "X", v)],
        );
        let wrong: Vec<String> = results
            .iter()
            .filter_map(|(label, res)| match res {
                Err(e) => {
                    let msg = e.to_string();
                    let names_all = msg.contains("'X'")
                        && msg.contains("'first'")
                        && msg.contains("'second'")
                        && msg.contains("t=3")
                        && msg.contains("more than one `set`");
                    (!names_all).then(|| format!("  {label}: wrong error: {msg}"))
                }
                Ok(got) => Some(format!("  {label}: ran and gave {got:?}")),
            })
            .collect();
        if !wrong.is_empty() {
            failures.push(format!("X = {x0}, two `X = {v}` events at t=3:\n{}", wrong.join("\n")));
        }
    }
    assert!(
        failures.is_empty(),
        "every backend must refuse two sets on one compartment with an error naming \
         'X', 'first', 'second' and t=3:\n{}",
        failures.join("\n")
    );
}

/// A drain-and-`set` phase that leaves a compartment negative is an error on
/// every backend, never a clamped 0. Cases from X = 10: `set X = 2` with a full
/// drain (10 + (2 − 10) − 10 = −8), and two drains of 0.6 (10 − 6 − 6 = −2).
/// Gillespie and the ODE report `InterventionNegative`; chain-binomial reports
/// its post-draw scan's `BinomialOvershoot` (a known mislabel, left as is).
#[test]
fn drain_and_set_phase_going_negative_is_an_error_on_every_backend() {
    use sim::error::{NegativeCountCause, SimError};
    let cases = [
        ("set_plus_full_drain", vec![set_event("reset", "X", 2.0), drain_event("cull", 1.0)]),
        ("two_drains_overdraw", vec![drain_event("cull_a", 0.6), drain_event("cull_b", 0.6)]),
    ];
    let mut failures = Vec::new();
    for (case, events) in cases {
        let results = coincident_results(case, ("X", 10.0), ("Y", 0.0), events);
        let wrong: Vec<String> = results
            .iter()
            .filter_map(|(label, res)| {
                let ok = match res {
                    Err(SimError::NegativeCount { compartment, cause, .. }) => {
                        compartment == "X"
                            && (label.starts_with("chain_binomial")
                                || *cause == NegativeCountCause::InterventionNegative)
                    }
                    _ => false,
                };
                (!ok).then(|| format!("  {label}: {res:?}"))
            })
            .collect();
        if !wrong.is_empty() {
            failures.push(format!("{case}:\n{}", wrong.join("\n")));
        }
    }
    assert!(
        failures.is_empty(),
        "X must be refused as negative on every backend, not clamped:\n{}",
        failures.join("\n")
    );
}

fn count_drain_event(name: &str, count: f64) -> Intervention {
    event(
        name,
        vec![3.0],
        Action::AbsoluteTransfer(AbsoluteTransfer {
            src: "X".into(),
            dst: "Y".into(),
            count: Expr::const_(count),
        }),
    )
}

/// An overdraw by half a person at X = 1e9 is still an overdraw. Two count
/// transfers of 6e8 and 4e8 + 0.5 from X = 1e9 take 1000000000.5 (the integer
/// path rounds the second to 400000001 and goes to −1). Every backend must
/// refuse; a slack proportional to |X| alone would let the ODE print X = 0,
/// Y = 1000000001 and grow the population by one.
#[test]
fn overdraw_by_half_a_person_at_1e9_is_refused_on_every_backend() {
    use sim::error::{NegativeCountCause, SimError};
    let results = coincident_results(
        "gh985_overdraw_1e9",
        ("X", 1e9),
        ("Y", 0.0),
        vec![count_drain_event("cull_a", 6e8), count_drain_event("cull_b", 4e8 + 0.5)],
    );
    let wrong: Vec<String> = results
        .iter()
        .filter_map(|(label, res)| {
            let ok = match res {
                Err(SimError::NegativeCount { compartment, cause, .. }) => {
                    compartment == "X"
                        && (label.starts_with("chain_binomial")
                            || *cause == NegativeCountCause::InterventionNegative)
                }
                _ => false,
            };
            (!ok).then(|| format!("  {label}: {res:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "X must be refused as negative on every backend:\n{}", wrong.join("\n"));
}

/// Drains that remove exactly the snapshot are not overdraws, though the f64
/// sum can land a few ulps below zero: a full-fraction drain from 1000003,
/// fractions 0.3 + 0.7 + 0.0 from 1000003, and ten drains of 0.1 from
/// 999999937. Every backend runs them.
#[test]
fn drains_summing_to_the_whole_compartment_are_not_refused() {
    let cases: Vec<(&str, f64, Vec<Intervention>)> = vec![
        ("full_fraction", 1000003.0, vec![drain_event("all", 1.0)]),
        (
            "point3_point7_zero",
            1000003.0,
            vec![drain_event("a", 0.3), drain_event("b", 0.7), drain_event("c", 0.0)],
        ),
        (
            "ten_tenths",
            999999937.0,
            (0..10).map(|k| drain_event(&format!("d{k}"), 0.1)).collect(),
        ),
    ];
    let mut wrong = Vec::new();
    for (case, x0, events) in cases {
        for (label, res) in coincident_results(case, ("X", x0), ("Y", 0.0), events) {
            if let Err(e) = res {
                wrong.push(format!("  {case} / {label}: {e}"));
            }
        }
    }
    assert!(wrong.is_empty(), "exact drains were refused:\n{}", wrong.join("\n"));
}

// ─────────────────────────────────────────────────────────────────────────────
// D1 / D4 / the run-relative grid key, row by row. Each fixture's header carries
// the hand computation; with a zero rate the three backends must reproduce it on
// EVERY output row, the t_start row included — the row an effect at t_start
// used to be missing from (gh#958: chain-binomial never fired it, the ODE and
// Gillespie fired it after recording the row).

/// Every `(t, counts of `comps`)` row of `stem` on every backend must equal
/// `expected`.
fn assert_rows_on_every_backend(stem: &str, comps: [&str; 2], expected: &[(f64, [i64; 2])]) {
    let compiled = load_named(stem);
    let t_start = compiled.model.simulation.t_start;
    let t_end = compiled.model.simulation.t_end;
    let params = compiled.default_params.clone();
    let idx = comps.map(|c| local_idx(&compiled, c));
    for (name, sim, cfg) in battery_backends(t_start, t_end) {
        let traj = sim
            .run(&compiled, &params, SEED, &cfg)
            .unwrap_or_else(|e| panic!("{stem}/{name}: forward sim: {e:?}"));
        let rows: Vec<(f64, [i64; 2])> = traj
            .snapshots
            .iter()
            .map(|s| (s.t, idx.map(|i| s.int_state.counts[i])))
            .collect();
        assert_eq!(rows, expected, "{stem}/{name}: rows of {comps:?} (hand computation in the fixture header)");
    }
}

/// D1: an event and an intervention at t_start apply to the starting state,
/// events first, before the t_start row; on-grid effects later are unchanged.
#[test]
fn effects_at_t_start_are_in_the_t_start_row_on_every_backend() {
    assert_rows_on_every_backend(
        "origin_effects",
        ["A", "B"],
        &[
            (0.0, [30, 30]),
            (1.0, [30, 30]),
            (2.0, [34, 30]),
            (3.0, [17, 47]),
            (4.0, [17, 47]),
            (5.0, [17, 47]),
        ],
    );
}

/// D1: a recurring schedule with the default `from` fires at t_start too, so
/// `every = 2 'days` over [0, 6] fires four times on every backend.
#[test]
fn a_recurring_schedule_with_default_from_fires_at_t_start_on_every_backend() {
    assert_rows_on_every_backend(
        "recurring_default_from",
        ["A", "B"],
        &[
            (0.0, [1, 0]),
            (1.0, [1, 0]),
            (2.0, [2, 0]),
            (3.0, [2, 0]),
            (4.0, [3, 0]),
            (5.0, [3, 0]),
            (6.0, [4, 0]),
        ],
    );
}

/// The grid key is relative to t_start: with `from = 0.5`, `dt = 1`, the event
/// at 1 fires on the step ending at 1.5 and the intervention at 2.4 on the step
/// ending at 2.5 — every row agrees with the exact backends, which fire at 1
/// and 2.4. The absolute key dropped the event at 1 and fired 2.4 at 1.5.
#[test]
fn an_offset_origin_fires_on_the_runs_own_grid() {
    assert_rows_on_every_backend(
        "offset_origin",
        ["A", "B"],
        &[(0.5, [0, 0]), (1.5, [1, 0]), (2.5, [11, 0]), (3.5, [11, 0]), (4.5, [11, 0])],
    );
}

/// D4 at the origin: the drain reads the post-inflow state, so I = 5, R = 5 in
/// the t_start row on every backend, whatever the declaration order.
#[test]
fn an_inflow_and_a_drain_at_t_start_give_the_post_inflow_split_on_every_backend() {
    assert_rows_on_every_backend(
        "inflow_drain_origin",
        ["I", "R"],
        &[(0.0, [5, 5]), (1.0, [5, 5]), (2.0, [5, 5]), (3.0, [5, 5])],
    );
}
