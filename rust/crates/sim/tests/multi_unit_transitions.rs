//! Transitions that consume more than one unit of a compartment per firing
//! (`S + S --> D`, net stoichiometry `S: -2`).
//!
//! - **Gillespie** applies the feasibility rule: a transition may fire only when
//!   every compartment it consumes holds at least the units one firing removes.
//!   Its effective propensity is the declared rate where feasible and 0
//!   otherwise; the declared rate is never rewritten. No firing can take a
//!   compartment below zero.
//!
//! - **Chain-binomial** refuses such a transition before any step: its
//!   Euler-multinomial draws how many individuals leave a compartment, one at a
//!   time, and has no multi-unit meaning.
//!
//! The models are built as `ir::Model` structs (as in
//! `multi_source_transition.rs`) so the stoichiometry is exact.

use std::collections::HashMap;
use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        RegularOutputSchedule, SimulationConfig,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{GillespieConfig, SimConfig},
    simulate::Simulate,
    GillespieSim,
};

fn param(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn pop(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op, left: Box::new(l), right: Box::new(r) } })
}
fn mul(l: Expr, r: Expr) -> Expr {
    bin(BinOp::Mul, l, r)
}
fn fixed(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}
fn transition(name: &str, stoich: &[(&str, i64)], rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: stoich.iter().map(|(c, d)| StoichiometryEntry((*c).into(), *d)).collect(),
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    }
}

fn build(
    comps: &[&str],
    transitions: Vec<Transition>,
    parameters: Vec<Parameter>,
    init: &[(&str, f64)],
    t_end: f64,
) -> Model {
    Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "multi_unit".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: comps
            .iter()
            .map(|c| Compartment { name: (*c).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions,
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters,
        initial_conditions: InitialConditions::constants(
            init.iter().map(|(k, v)| ((*k).into(), *v)).collect::<HashMap<String, f64>>(),
        ),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 0.25 }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(7),
            integrator: Default::default(),
            t_end_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![],
        contrasts: vec![],
    }
}

/// The reported reproduction: `pair : S + S --> D @ k*S*S/(S+D+R)` and
/// `rec : S --> R @ g*S`, `S = 5`, `k = 5`, `g = 1`. The `pair` rate stays
/// positive at `S = 1`, where a firing would leave `S = -1`.
fn pair_model(stoich_pair: &[(&str, i64)]) -> CompiledModel {
    let n = bin(BinOp::Add, bin(BinOp::Add, pop("S"), pop("D")), pop("R"));
    let pair_rate = bin(BinOp::Div, mul(mul(param("k"), pop("S")), pop("S")), n);
    let model = build(
        &["S", "D", "R"],
        vec![
            transition("pair", stoich_pair, pair_rate),
            transition("rec", &[("S", -1), ("R", 1)], mul(param("g"), pop("S"))),
        ],
        vec![fixed("k", 5.0), fixed("g", 1.0)],
        &[("S", 5.0), ("D", 0.0), ("R", 0.0)],
        3.0,
    );
    CompiledModel::new(model).expect("compiles")
}

fn gillespie(t_end: f64) -> SimConfig {
    SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end, output_dt: None })
}

/// Gillespie never takes `S` below zero on the reproduction, across 200 seeds,
/// and every snapshot conserves `S + 2·D + R = 5` (each `pair` firing turns two
/// S into one D). Also asserts `pair` fires on some seed, so the test cannot
/// pass by the transition never firing.
#[test]
fn gillespie_multi_unit_never_goes_negative_and_conserves() {
    for stoich in [
        // camdlc's collapsed form, and a hand-authored IR repeating S.
        &[("S", -2), ("D", 1)][..],
        &[("S", -1), ("S", -1), ("D", 1)][..],
    ] {
        let compiled = pair_model(stoich);
        let mut pair_fired = 0u64;
        for seed in 1..=200u64 {
            let traj = GillespieSim
                .run(&compiled, &compiled.default_params, seed, &gillespie(3.0))
                .unwrap_or_else(|e| panic!("seed {seed} ({stoich:?}): gillespie errored: {e}"));
            for snap in &traj.snapshots {
                let c = &snap.int_state.counts;
                assert!(c.iter().all(|&x| x >= 0), "seed {seed}: negative count {c:?} at t={}", snap.t);
                assert_eq!(c[0] + 2 * c[1] + c[2], 5, "seed {seed}: S + 2D + R not conserved: {c:?}");
            }
            let last = traj.snapshots.last().unwrap();
            assert!(last.int_state.counts[0] >= 0, "seed {seed}: S ends negative");
            pair_fired += traj.transition_diagnostics[0].total_firings;
        }
        assert!(pair_fired > 0, "pair never fired over 200 seeds: the test would be vacuous");
    }
}

/// A single-unit exit whose rate does not read its source (`S --> R @ k`)
/// stops at `S = 0` instead of firing into `S = -1`. Its rate never changes,
/// so only the feasibility re-check on the consumed compartment (not the
/// rate-dependency graph) can switch it off on Gillespie's sparse path.
#[test]
fn gillespie_constant_rate_exit_stops_at_empty_source() {
    let model = build(
        &["S", "R"],
        vec![transition("leave", &[("S", -1), ("R", 1)], param("k"))],
        vec![fixed("k", 50.0)],
        &[("S", 3.0), ("R", 0.0)],
        10.0,
    );
    let compiled = CompiledModel::new(model).unwrap();
    for seed in 1..=50u64 {
        let traj = GillespieSim
            .run(&compiled, &compiled.default_params, seed, &gillespie(10.0))
            .unwrap_or_else(|e| panic!("seed {seed}: gillespie errored: {e}"));
        let last = &traj.snapshots.last().unwrap().int_state.counts;
        assert_eq!(last, &vec![0, 3], "seed {seed}: expected S drained to exactly 0");
    }
}

/// Chain-binomial refuses a transition consuming two units of its source,
/// before any step: the run returns the up-front `Validation` refusal (naming
/// the transition, the units and the compartment), not a `NegativeCount` from
/// the kernel. Both stoichiometry spellings get the same verdict. A
/// single-unit model is not refused.
#[test]
fn chain_binomial_refuses_multi_unit_before_any_step() {
    use sim::{config::ChainBinomialConfig, error::SimError, ChainBinomialSim};
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 3.0, dt: 1.0 });
    for stoich in [&[("S", -2), ("D", 1)][..], &[("S", -1), ("S", -1), ("D", 1)][..]] {
        let compiled = pair_model(stoich);
        let msg = compiled
            .validate_unit_source_consumption()
            .expect_err("a two-unit source must be refused")
            .to_string();
        assert!(
            msg.contains("transition 'pair' consumes 2 units of S per firing"),
            "names transition, units, compartment: {msg}"
        );
        assert!(msg.contains("help: use --backend gillespie"), "points to gillespie: {msg}");
        match ChainBinomialSim.run(&compiled, &compiled.default_params, 1, &cfg) {
            Err(SimError::Validation(m)) => assert_eq!(m, match compiled.validate_unit_source_consumption() {
                Err(SimError::Validation(v)) => v,
                other => panic!("unexpected {other:?}"),
            }),
            other => panic!("chain_binomial must refuse up front, got {other:?}"),
        }
    }
    let single = build(
        &["S", "R"],
        vec![transition("rec", &[("S", -1), ("R", 1)], mul(param("g"), pop("S")))],
        vec![fixed("g", 1.0)],
        &[("S", 5.0), ("R", 0.0)],
        3.0,
    );
    let single = CompiledModel::new(single).unwrap();
    assert!(single.validate_unit_source_consumption().is_ok());
    // dt matches the 0.25 output step (chain-binomial snaps outputs to its grid).
    let fine = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 3.0, dt: 0.25 });
    let run = ChainBinomialSim.run(&single, &single.default_params, 1, &fine);
    assert!(run.is_ok(), "a single-unit model runs on chain_binomial: {:?}", run.err());
}

/// An overdispersed exit whose gamma rate multiplier is drawn as exactly 0
/// (likely when σ² ≫ dt: the Gamma shape dt/σ² is tiny) has zero hazard for
/// that step, so no member leaves. Every exit of the source group drew 0, the
/// group's total rate is 0 — and the group must draw nothing. It must not fall
/// through to the source-less draw, which is unbounded by the source and took
/// `S` from 5 to −3305.
#[test]
fn chain_binomial_zero_gamma_multiplier_draws_no_exits() {
    use sim::{config::ChainBinomialConfig, ChainBinomialSim};
    let mut leave = transition("leave", &[("S", -1), ("I", 1)], mul(param("beta"), pop("S")));
    leave.draw_method = DrawMethod::Overdispersed {
        sigma_sq: param("sig"),
        sigma_sq_grad: HashMap::new(),
    };
    let mut model = build(
        &["S", "I"],
        vec![leave],
        vec![fixed("beta", 50.0), fixed("sig", 500.0)],
        &[("S", 5.0), ("I", 0.0)],
        10.0,
    );
    model.output.times = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
    let compiled = CompiledModel::new(model).unwrap();
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 10.0, dt: 1.0 });
    for seed in 1..=50u64 {
        let traj = ChainBinomialSim
            .run(&compiled, &compiled.default_params, seed, &cfg)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        for snap in &traj.snapshots {
            let c = &snap.int_state.counts;
            assert!(c[0] >= 0 && c[1] >= 0 && c[0] + c[1] == 5, "seed {seed}: {c:?} at t={}", snap.t);
        }
    }
}

/// The transition-stage negative scan is an internal invariant. Driving the
/// chain-binomial kernel directly (`step_one`, which every entry point guards
/// with `validate_unit_source_consumption` and which is bypassed here on
/// purpose) on the two-unit model takes S below zero in the transition stage;
/// the kernel reports it as `SimulatorInvariant`, which stops the run and is
/// never a per-particle −∞ weight.
#[test]
fn a_transition_stage_negative_is_a_simulator_invariant_not_a_dead_particle() {
    use sim::{chain_binomial::{step_one, StepScratch}, error::SimError, rng::StatefulRng, state::RealState};
    let compiled = pair_model(&[("S", -2), ("D", 1)]);
    let mut hit = None;
    for seed in 1..=20u64 {
        let mut counts = vec![5_i64, 0, 0];
        let mut flows = vec![0_u64; 2];
        let mut real = RealState::new(0);
        let mut rng = StatefulRng::new(seed);
        let mut scratch = StepScratch::new(&compiled);
        if let Err(e) = step_one(
            &compiled, &mut counts, &mut flows, &mut real, &compiled.default_params,
            0.0, 1.0, None, &mut rng, &mut scratch,
        ) {
            hit = Some(e);
            break;
        }
    }
    let err = hit.expect("the unguarded kernel must overdraw S on some seed");
    assert!(matches!(err, SimError::SimulatorInvariant { .. }), "got {err:?}");
    assert!(err.to_string().contains("bug in camdl"), "names it a simulator bug: {err}");
    assert!(err.is_structural(), "stops the run");
    assert!(!err.is_per_particle_recoverable(), "never a −∞ particle");
}

/// A negative count that came from the starting state is reported as
/// `NegativeCount{InitialState}` on both stochastic backends, not as a
/// simulator bug: `R = 2 - I0` with `I0 = 5` starts R at −3. R is in no rate,
/// so nothing else catches it first.
#[test]
fn a_negative_initial_state_is_labelled_initial_state() {
    use sim::{config::ChainBinomialConfig, error::{NegativeCountCause, SimError}, ChainBinomialSim};
    use ir::model::InitSpec;
    let mut model = build(
        &["S", "I", "R"],
        vec![
            transition("inf", &[("S", -1), ("I", 1)], mul(param("beta"), pop("S"))),
            transition("rec", &[("I", -1), ("R", 1)], mul(param("gamma"), pop("I"))),
        ],
        vec![fixed("beta", 1.0), fixed("gamma", 1.0)],
        &[("S", 50.0), ("I", 5.0), ("R", 0.0)],
        3.0,
    );
    model.output.times = OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 });
    *model.initial_conditions.0.get_mut("R").unwrap() =
        InitSpec::Deterministic(bin(BinOp::Sub, Expr::const_(2.0), Expr::const_(5.0)));
    let compiled = CompiledModel::new(model).unwrap();
    let cb = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 3.0, dt: 1.0 });
    for (label, res) in [
        ("chain_binomial", ChainBinomialSim.run(&compiled, &compiled.default_params, 1, &cb)),
        ("gillespie", GillespieSim.run(&compiled, &compiled.default_params, 1, &gillespie(3.0))),
    ] {
        match res {
            Err(ref e @ SimError::NegativeCount { cause: NegativeCountCause::InitialState, ref compartment, .. }) => {
                assert_eq!(compartment, "R", "{label}");
                assert!(e.is_per_particle_recoverable() && !e.is_structural(), "{label}");
            }
            other => panic!("{label}: expected InitialState on R, got {:?}", other.map(|t| t.snapshots.len())),
        }
    }
}
