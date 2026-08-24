//! gh#122 — a `deterministic(rate)` transition that is the SOLE exit from its
//! source compartment must FIRE `round(rate*dt)` (capped by the source count),
//! not be silently frozen; and a source that MIXES a deterministic exit with
//! any other exit must be rejected loudly on the stochastic backends (forward
//! chain_binomial + every stochastic inference producer), while still running
//! on the ODE backend (which treats every transition as a deterministic flow).
//!
//! Before the fix, the source-group loop in `chain_binomial::step_one` marked a
//! sourced deterministic transition `handled` and `continue`d BEFORE pushing it
//! to the competing-risk draw, so it never fired: deterministic aging, waning,
//! and recovery silently froze. The PGAS density / gradient mirrors carried the
//! same skip, so a sole-exit deterministic trajectory scored `-inf`.
//!
//! These tests build `ir::Model` structs directly (mirroring
//! `lifecycle_agreement_under_flow.rs`) so the oracle is an independent integer
//! recursion, not a second copy of the engine.

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
    config::{ChainBinomialConfig, OdeConfig, SimConfig},
    inference::pgas::{log_transition_density_substep, simulate_reference},
    inference::pgas_grad::{log_transition_density_grad, resolve_rate_grad_for_run},
    rng::StatefulRng,
    simulate::Simulate,
    state::Trajectory,
    ChainBinomialSim, OdeSim,
};

// ── tiny expression builders ────────────────────────────────────────────────
fn param(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn pop(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn mul(l: Expr, r: Expr) -> Expr {
    Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(l), right: Box::new(r) },
    })
}
fn fixed(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}
fn determ(name: &str, src: &str, dst: &str, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![StoichiometryEntry(src.into(), -1), StoichiometryEntry(dst.into(), 1)],
        rate,
        metadata: None,
        draw_method: DrawMethod::Deterministic,
        rate_grad: Default::default(),
        lineage: None,
    }
}
fn poisson(name: &str, src: &str, dst: &str, rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![StoichiometryEntry(src.into(), -1), StoichiometryEntry(dst.into(), 1)],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    }
}

fn build(
    name: &str,
    comps: &[&str],
    transitions: Vec<Transition>,
    parameters: Vec<Parameter>,
    init: &[(&str, f64)],
    t_end: f64,
) -> Model {
    Model {
        ic_grad: Default::default(),
        name: name.into(),
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
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
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

fn count_at(compiled: &CompiledModel, traj: &Trajectory, name: &str, t: f64) -> i64 {
    let g = compiled.comp_index[name];
    let local = compiled.global_to_int[g].expect("integer compartment");
    let snap = traj
        .snapshots
        .iter()
        .find(|s| (s.t - t).abs() < 1e-6)
        .unwrap_or_else(|| panic!("no snapshot at t={t}"));
    snap.int_state.counts[local]
}

const DT: f64 = 1.0;

/// `I --> R @ deterministic(gamma*I)`, single source, integer I. The whole
/// chain-binomial trajectory must equal the exact integer recursion
/// `I_{n+1} = I_n − min(round(γ·I_n·dt), I_n)` — deterministic ⇒ seed-independent.
#[test]
fn deterministic_only_decay_matches_integer_recursion() {
    const GAMMA: f64 = 0.1;
    const I0: i64 = 1000;
    const T_END: f64 = 10.0;

    let model = build(
        "determ_decay",
        &["I", "R"],
        vec![determ("decay", "I", "R", mul(param("gamma"), pop("I")))],
        vec![fixed("gamma", GAMMA)],
        &[("I", I0 as f64), ("R", 0.0)],
        T_END,
    );
    let compiled = CompiledModel::new(model).expect("compile determ_decay");

    // Independent oracle: the exact integer recursion.
    let mut i = I0;
    let mut expected = vec![i];
    let n_steps = T_END as usize;
    for _ in 0..n_steps {
        let flow = (GAMMA * i as f64 * DT).round() as i64;
        let flow = flow.clamp(0, i);
        i -= flow;
        expected.push(i);
    }

    // Deterministic ⇒ identical for any seed.
    for seed in [1u64, 42, 12345] {
        let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: DT });
        let traj = ChainBinomialSim.run(&compiled, &compiled.default_params, seed, &cfg).unwrap();
        for (k, &exp_i) in expected.iter().enumerate() {
            let got = count_at(&compiled, &traj, "I", k as f64);
            assert_eq!(
                got, exp_i,
                "seed {seed}: I at t={k} must equal the integer recursion {exp_i}, got {got} \
                 (before the gh#122 fix the deterministic exit never fires and I stays {I0})"
            );
            // Conservation: I + R == I0 at every output.
            let r = count_at(&compiled, &traj, "R", k as f64);
            assert_eq!(got + r, I0, "seed {seed}: I+R conserved at t={k}");
        }
    }
}

/// A chain of sourced deterministic aging flows PLUS a stochastic exit from a
/// DIFFERENT compartment. Total population is invariant and no compartment goes
/// negative, over many seeds. (Every transition is a transfer, so Σ is conserved
/// regardless of the draws.)
#[test]
fn conservation_under_deterministic_aging_and_stochastic_exit() {
    const T_END: f64 = 20.0;
    // A → B → C deterministic aging; I → R stochastic (Poisson).
    let model = build(
        "aging_plus_stochastic",
        &["A", "B", "C", "I", "R"],
        vec![
            determ("age_ab", "A", "B", mul(param("k_ab"), pop("A"))),
            determ("age_bc", "B", "C", mul(param("k_bc"), pop("B"))),
            poisson("recover", "I", "R", mul(param("mu"), pop("I"))),
        ],
        vec![fixed("k_ab", 0.33), fixed("k_bc", 0.2), fixed("mu", 0.05)],
        &[("A", 500.0), ("B", 0.0), ("C", 0.0), ("I", 300.0), ("R", 0.0)],
        T_END,
    );
    let compiled = CompiledModel::new(model).expect("compile aging_plus_stochastic");
    const TOTAL: i64 = 500 + 300;

    for seed in 0..20u64 {
        let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: DT });
        let traj = ChainBinomialSim.run(&compiled, &compiled.default_params, seed, &cfg).unwrap();
        for snap in &traj.snapshots {
            let counts = &snap.int_state.counts;
            let sum: i64 = counts.iter().sum();
            assert_eq!(sum, TOTAL, "seed {seed}: population conserved at t={}", snap.t);
            for (li, &c) in counts.iter().enumerate() {
                assert!(c >= 0, "seed {seed}: compartment local {li} negative ({c}) at t={}", snap.t);
            }
        }
        // Non-vacuity: A actually aged out (deterministic flow fired).
        let a_end = count_at(&compiled, &traj, "A", T_END);
        assert!(a_end < 500, "seed {seed}: A must have aged down from 500 (deterministic), got {a_end}");
    }
}

/// sim ↔ density ↔ gradient parity for a sole-exit deterministic model. The
/// realized (counts_before, flows) recorded by the PGAS producer (which runs
/// `step_one`) must score to a FINITE transition density (a point mass, log 1 =
/// 0), and the gradient path must agree (finite log_p, and NO gradient from the
/// deterministic transition). Before the fix this scored `-inf`.
#[test]
fn sim_density_and_gradient_agree_for_sole_exit_deterministic() {
    const GAMMA: f64 = 0.1;
    const T_END: f64 = 8.0;
    let model = build(
        "determ_decay_density",
        &["I", "R"],
        vec![determ("decay", "I", "R", mul(param("gamma"), pop("I")))],
        vec![fixed("gamma", GAMMA)],
        &[("I", 1000.0), ("R", 0.0)],
        T_END,
    );
    let compiled = CompiledModel::new(model).expect("compile");
    let params = compiled.default_params.clone();
    let t_start = compiled.model.simulation.t_start;

    let mut rng = StatefulRng::new(7);
    let reference = simulate_reference(&compiled, &params, T_END, DT, &mut rng).expect("produce reference");

    // gamma is the only param (index 0); estimate it so the gradient path is
    // exercised. The hand-built model carries no rate_grad, so the resolved run
    // grads are empty — which is exactly right: a deterministic point mass
    // contributes NO gradient regardless of what is estimated.
    let d = 1;
    let model_to_estimated: Vec<Option<usize>> = vec![Some(0)];
    let rate_grads_for_run =
        resolve_rate_grad_for_run(&compiled.resolved.rate_grads_indexed, &model_to_estimated);

    let mut scored_any = false;
    for (s, rec) in reference.substeps.iter().enumerate() {
        let t = t_start + s as f64 * DT;

        let dens = log_transition_density_substep(
            &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, DT, None,
        )
        .expect("density call ok");
        assert!(
            dens.is_finite(),
            "substep {s}: sole-exit deterministic density must be FINITE (point mass), got {dens} \
             (before the gh#122 fix the producer froze the flow and the density scored -inf)"
        );

        let (glp, grad) = log_transition_density_grad(
            &compiled, &rec.counts_before, &rec.flows, &rec.gammas, &params, t, DT, None, d,
            &rate_grads_for_run,
        )
        .expect("grad call ok");
        assert!(glp.is_finite(), "substep {s}: gradient-path log_p must be finite, got {glp}");
        assert_eq!(grad.len(), d);
        assert_eq!(
            grad[0], 0.0,
            "substep {s}: a deterministic point mass contributes NO gradient, got {}",
            grad[0]
        );
        scored_any = true;
    }
    assert!(scored_any, "reference must have at least one substep");

    // Non-vacuity: the producer actually recorded a nonzero deterministic flow
    // at least once (I depleted), so the finite-density assertion is informative.
    let total_flow: u64 = reference.substeps.iter().map(|r| r.flows.iter().sum::<u64>()).sum();
    assert!(total_flow > 0, "producer must record nonzero deterministic flow (I should deplete)");
}

/// A source that MIXES a deterministic exit with another exit is rejected on the
/// stochastic paths (forward chain_binomial + the shared validation the
/// inference dispatch gate calls) with a located message, while still RUNNING on
/// the ODE backend.
#[test]
fn mixed_deterministic_source_rejected_on_stochastic_but_runs_on_ode() {
    const T_END: f64 = 5.0;
    // Source `I` has TWO exits: a deterministic recovery and a Poisson death.
    let model = build(
        "mixed_source",
        &["I", "R", "D"],
        vec![
            determ("recover", "I", "R", mul(param("gamma"), pop("I"))),
            poisson("die", "I", "D", mul(param("mu"), pop("I"))),
        ],
        vec![fixed("gamma", 0.1), fixed("mu", 0.02)],
        &[("I", 1000.0), ("R", 0.0), ("D", 0.0)],
        T_END,
    );
    let compiled = CompiledModel::new(model).expect("a mixed model still COMPILES (rejected at dispatch)");

    // The shared structural validation (the inference dispatch gate delegates to
    // this) errors with a located, gh#122-tagged message.
    let err = compiled
        .validate_deterministic_source_exits()
        .expect_err("mixed source must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("gh#122"), "message must cite the issue: {msg}");
    assert!(msg.contains('I'), "message must name the source compartment I: {msg}");
    assert!(msg.contains("recover"), "message must name the deterministic exit: {msg}");

    // Forward chain_binomial hard-errors (does not silently over-draw the source).
    let cb = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: DT });
    let cb_res = ChainBinomialSim.run(&compiled, &compiled.default_params, 1, &cb);
    assert!(cb_res.is_err(), "forward chain_binomial must reject a mixed deterministic source");
    assert!(cb_res.unwrap_err().to_string().contains("gh#122"));

    // ODE runs it: it treats every transition as a deterministic flow, so the
    // mix is well-defined there (no independent over-draw).
    let ode = SimConfig::Ode(OdeConfig { t_start: 0.0, t_end: T_END, dt: DT });
    let ode_res = OdeSim.run(&compiled, &compiled.default_params, 1, &ode);
    assert!(ode_res.is_ok(), "ODE must still run a mixed deterministic source: {:?}", ode_res.err());
}

/// A sole-exit deterministic source is NOT rejected (the fix supports it).
#[test]
fn sole_exit_deterministic_source_is_accepted() {
    let model = build(
        "sole_exit",
        &["I", "R"],
        vec![determ("decay", "I", "R", mul(param("gamma"), pop("I")))],
        vec![fixed("gamma", 0.1)],
        &[("I", 100.0), ("R", 0.0)],
        5.0,
    );
    let compiled = CompiledModel::new(model).unwrap();
    assert!(
        compiled.validate_deterministic_source_exits().is_ok(),
        "a deterministic transition that is the SOLE exit from its source is supported"
    );
}

/// CRN / byte-identity guard rail: a model with NO sourced-deterministic
/// transition never enters the new code path, so its chain_binomial trajectory
/// is unchanged. Here we pin determinism (same seed → identical trajectory); the
/// authoritative byte-for-byte non-regression is `gate_trajectory_baseline` and
/// `gate_pgas_density_baseline` under `make test`, which must show ZERO drift
/// after this change (the deterministic branch consumes no RNG and is only
/// reached for a deterministic source member).
#[test]
fn deterministic_free_model_is_unperturbed() {
    // Plain SIR-ish: S --> I (Poisson), I --> R (Poisson). No deterministic exit.
    let model = build(
        "sir_poisson",
        &["S", "I", "R"],
        vec![
            poisson("infect", "S", "I", mul(param("beta"), pop("I"))),
            poisson("recover", "I", "R", mul(param("gamma"), pop("I"))),
        ],
        vec![fixed("beta", 0.0003), fixed("gamma", 0.1)],
        &[("S", 990.0), ("I", 10.0), ("R", 0.0)],
        20.0,
    );
    let compiled = CompiledModel::new(model).unwrap();
    // No sourced-deterministic ⇒ validation is a no-op.
    assert!(compiled.validate_deterministic_source_exits().is_ok());

    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 20.0, dt: DT });
    let a = ChainBinomialSim.run(&compiled, &compiled.default_params, 99, &cfg).unwrap();
    let b = ChainBinomialSim.run(&compiled, &compiled.default_params, 99, &cfg).unwrap();
    assert_eq!(a.snapshots.len(), b.snapshots.len());
    for (sa, sb) in a.snapshots.iter().zip(&b.snapshots) {
        assert_eq!(sa.int_state.counts, sb.int_state.counts, "same seed ⇒ identical trajectory");
    }
}
