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
