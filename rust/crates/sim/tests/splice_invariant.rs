//! The splice invariant — the correctness oracle for the start-from-state engine
//! seam (gh#322).
//!
//! A continuous chain-binomial run can be SPLICED at any output-grid time T*:
//! injecting the continuous run's EXACT compartment state and RNG state at T*
//! and resuming must reproduce the continuous tail byte-for-byte.
//!
//!     run(t0 → t_end, seed)
//!         ≡  run(t0 → T*, seed)  ++  resume(state@T*, rng@T*, T* → t_end)
//!
//! This is the strongest available oracle: get the cursor / clock / flows / RNG
//! re-seat wrong and the spliced tail diverges from the continuous tail on the
//! first substep. The headline test pins it byte-identically for integer dt.
//! The remaining tests pin the gates (off-grid T*, reactive) and the no-op
//! identity of `Resume::default()`.

use std::collections::HashMap;
use std::path::PathBuf;

use ir::{
    expr::{BinOp, ConstExpr, Expr},
    intervention::{
        Action, FireSource, FractionTransfer, Intervention, InterventionKind, InterventionSchedule,
    },
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    chain_binomial::{run_chain_binomial, run_chain_binomial_with_observer, Resume, StartState},
    compiled_model::CompiledModel,
    config::ChainBinomialConfig,
    rng::StatefulRng,
};

const SEED: u64 = 7;
const T_STAR: f64 = 10.0;
const T_END: f64 = 30.0;

/// An SIR + vaccination model (S, I, R, V) with stochastic Poisson dynamics and
/// two fractional-transfer interventions S→V, one BEFORE T* (t=5) and one AFTER
/// T* (t=15). The pre-T* fire is baked into the injected state@T* and must NOT
/// re-fire in the resumed run (fire steps are absolute — hazard #4); the post-T*
/// fire must fire correctly in the resumed tail. `output_times` and `t_end` are
/// parameters so each run's output schedule matches its own window (the final
/// flush drains all remaining output times regardless of `t_end`).
fn sir_model(output_times: Vec<f64>, t_end: f64) -> (CompiledModel, Vec<f64>) {
    let n_expr = Expr::pop_sum(vec!["S".into(), "I".into(), "R".into(), "V".into()]);
    let beta_s_i = Expr::bin_op(
        BinOp::Mul,
        Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
        Expr::pop("I"),
    );
    let infection_rate = Expr::bin_op(BinOp::Div, beta_s_i, n_expr);
    let recovery_rate = Expr::bin_op(BinOp::Mul, Expr::param("gamma"), Expr::pop("I"));

    let tr = |name: &str, from: &str, to: &str, rate: Expr| Transition {
        name: name.into(),
        stoichiometry: vec![
            StoichiometryEntry(from.into(), -1),
            StoichiometryEntry(to.into(), 1),
        ],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    };

    let iv = |name: &str, at: f64, frac: f64| Intervention {
        name: name.into(),
        base_name: None,
        fire: FireSource::Scheduled(InterventionSchedule::AtTimes(vec![at])),
        kind: InterventionKind::Scenario,
        actions: vec![Action::FractionTransfer(FractionTransfer {
            src: "S".into(),
            dst: "V".into(),
            fraction: Expr::Const(ConstExpr { value: frac }),
        })],
    };

    let int_comp = |name: &str| Compartment { name: name.into(), kind: CompartmentKind::Integer };

    let model = Model {
        name: "sir_splice".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![int_comp("S"), int_comp("I"), int_comp("R"), int_comp("V")],
        transitions: vec![
            tr("infection", "S", "I", infection_rate),
            tr("recovery", "I", "R", recovery_rate),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![iv("sia_pre", 5.0, 0.3), iv("sia_post", 15.0, 0.5)],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: ParamValue::Fixed { value: 0.4 }, param_kind: None, param_dim: None },
            Parameter { name: "gamma".into(), value: ParamValue::Fixed { value: 0.2 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::Explicit({
            let mut m = HashMap::new();
            m.insert("S".into(), 990.0);
            m.insert("I".into(), 10.0);
            m.insert("R".into(), 0.0);
            m.insert("V".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(output_times),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![],
        quantities: vec![], contrasts: vec![],
    };

    let compiled = CompiledModel::new(model).unwrap();
    let params = compiled.default_params.clone();
    (compiled, params)
}

fn ints(end: u64) -> Vec<f64> {
    (0..=end).map(|t| t as f64).collect()
}

/// 1. **Splice byte-identity (integer dt)** — the headline. A continuous run
///    spliced at T* reassembles byte-for-byte.
#[test]
fn splice_is_byte_identical_to_the_continuous_tail() {
    // ── Head: run [0, T*], capturing the final RNG state. ──
    let (head_model, head_params) = sir_model(ints(T_STAR as u64), T_STAR);
    let mut captured = StatefulRng::new(0); // overwritten by the capture
    let head_cfg = ChainBinomialConfig { t_start: 0.0, t_end: T_STAR, dt: 1.0 };
    let traj_head = run_chain_binomial_with_observer(
        &head_model,
        &head_params,
        SEED,
        &head_cfg,
        None,
        None,
        Resume { start: None, capture_final_rng: Some(&mut captured) },
    )
    .unwrap();

    let state_at_tstar = traj_head.snapshots.last().expect("head trajectory is non-empty");
    assert!(
        (state_at_tstar.t - T_STAR).abs() < 1e-9,
        "head's last snapshot must be at T* = {T_STAR}, got {}",
        state_at_tstar.t
    );

    // ── Resume: inject state@T* + the captured RNG, run [T*, t_end]. ──
    let (tail_model, tail_params) = sir_model(ints(T_END as u64), T_END);
    let start = StartState {
        int_s: state_at_tstar.int_state.clone(),
        real_s: state_at_tstar.real_state.clone(),
        rng: Some(captured),
    };
    let tail_cfg = ChainBinomialConfig { t_start: T_STAR, t_end: T_END, dt: 1.0 };
    let traj_tail = run_chain_binomial_with_observer(
        &tail_model,
        &tail_params,
        SEED,
        &tail_cfg,
        None,
        None,
        Resume { start: Some(&start), capture_final_rng: None },
    )
    .unwrap();

    // ── Full: the continuous run [0, t_end]. ──
    let (full_model, full_params) = sir_model(ints(T_END as u64), T_END);
    let full_cfg = ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 };
    let traj_full = run_chain_binomial(&full_model, &full_params, SEED, &full_cfg).unwrap();

    // Index the tail by time for lookup.
    let tail_at = |t: f64| {
        traj_tail
            .snapshots
            .iter()
            .find(|s| (s.t - t).abs() < 1e-9)
            .unwrap_or_else(|| panic!("tail has no snapshot at t={t}"))
    };

    // The T* boundary row: STATE (int/real) must match between the two runs —
    // this is the join point. Its FLOWS differ by the initial-row convention
    // (the resumed run emits T* with ZEROED flows; the continuous run carries the
    // [T*-dt, T*] interval's incidence there), so flows are NOT compared at T*.
    let full_tstar = traj_full
        .snapshots
        .iter()
        .find(|s| (s.t - T_STAR).abs() < 1e-9)
        .expect("full run has a T* row");
    let tail_tstar = tail_at(T_STAR);
    assert_eq!(
        full_tstar.int_state.counts, tail_tstar.int_state.counts,
        "T* int state must match across the splice"
    );
    assert_eq!(
        full_tstar.real_state.values, tail_tstar.real_state.values,
        "T* real state must match across the splice"
    );

    // STRICTLY after T*: every snapshot must be byte-identical in state AND flows.
    let mut compared = 0usize;
    for fs in &traj_full.snapshots {
        if fs.t <= T_STAR + 1e-9 {
            continue;
        }
        let ts = tail_at(fs.t);
        assert_eq!(
            fs.int_state.counts, ts.int_state.counts,
            "int state diverges at t={} (the splice re-seat is wrong)",
            fs.t
        );
        assert_eq!(
            fs.real_state.values, ts.real_state.values,
            "real state diverges at t={}",
            fs.t
        );
        assert_eq!(
            fs.flows.as_int(),
            ts.flows.as_int(),
            "flows diverge at t={} (cursor/flow-accumulator re-seat is wrong)",
            fs.t
        );
        compared += 1;
    }
    assert!(compared >= (T_END - T_STAR) as usize - 1, "must compare the whole tail, got {compared}");

    // Guard against a vacuous pass: the tail must actually have non-trivial
    // dynamics (the post-T* intervention fired and incidence flowed), or
    // "identical" would be trivially true.
    let v_idx = full_model.model.compartments.iter().position(|c| c.name == "V").unwrap();
    let v_end = traj_full.snapshots.last().unwrap().int_state.counts[v_idx];
    let v_tstar = full_tstar.int_state.counts[v_idx];
    assert!(
        v_end > v_tstar,
        "the post-T* SIA must move mass into V (v@T*={v_tstar}, v@end={v_end}) — \
         otherwise the splice test is vacuous"
    );
}

/// 2. **No-op identity** — a run with `Resume::default()` is byte-identical to
///    the existing `run_chain_binomial` wrapper.
#[test]
fn resume_default_is_byte_identical_to_the_plain_wrapper() {
    let (model, params) = sir_model(ints(T_END as u64), T_END);
    let cfg = ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 };

    let traj_wrapper = run_chain_binomial(&model, &params, SEED, &cfg).unwrap();
    let traj_default =
        run_chain_binomial_with_observer(&model, &params, SEED, &cfg, None, None, Resume::default())
            .unwrap();

    assert_eq!(traj_wrapper.snapshots.len(), traj_default.snapshots.len());
    for (a, b) in traj_wrapper.snapshots.iter().zip(&traj_default.snapshots) {
        assert_eq!(a.t, b.t, "snapshot times align");
        assert_eq!(a.int_state.counts, b.int_state.counts, "counts identical at t={}", a.t);
        assert_eq!(a.real_state.values, b.real_state.values, "real identical at t={}", a.t);
        assert_eq!(a.flows.as_int(), b.flows.as_int(), "flows identical at t={}", a.t);
    }
}

/// 3. **Off-grid T\* rejection** — a T* between output emits is a located error,
///    never a silent snap to a neighbour.
#[test]
fn off_grid_resume_time_is_rejected() {
    let (model, params) = sir_model(ints(T_END as u64), T_END);
    let (int_s, real_s) = model.initial_state(&params).unwrap();
    let start = StartState { int_s, real_s, rng: None };

    // T* = 10.5 lands BETWEEN the integer output emits.
    let cfg = ChainBinomialConfig { t_start: 10.5, t_end: T_END, dt: 1.0 };
    let err = run_chain_binomial_with_observer(
        &model,
        &params,
        SEED,
        &cfg,
        None,
        None,
        Resume { start: Some(&start), capture_final_rng: None },
    )
    .expect_err("an off-grid resume time must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("output-emit time") || msg.contains("T*"),
        "off-grid rejection must name the output-emit constraint / T*; got: {msg}"
    );
}

/// 4. **Reactive rejection** — a model with a reactive intervention entering the
///    seam produces the located capability error, not a silent fork.
#[test]
fn reactive_model_resume_is_rejected() {
    // Load the reactive golden (a SIR with an observed-threshold trigger). Its
    // output schedule is `regular start=0 step=1`, so T* = 0.0 is on-grid and the
    // off-grid gate passes — isolating the reactive gate.
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let mut model: Model = ir::from_str(&s).unwrap_or_else(|e| panic!("deser reactive golden: {e:?}"));
    // Fill any estimated parameter with a concrete placeholder so it compiles.
    for prm in &mut model.parameters {
        if prm.value.resolved_value().is_none() {
            prm.value = prm.value.with_value(0.5);
        }
    }
    let compiled = CompiledModel::new(model).expect("compile reactive golden");
    let params = compiled.default_params.clone();
    let (int_s, real_s) = compiled.initial_state(&params).unwrap();
    let start = StartState { int_s, real_s, rng: None };

    let cfg = ChainBinomialConfig { t_start: 0.0, t_end: 365.0, dt: 1.0 };
    let err = run_chain_binomial_with_observer(
        &compiled,
        &params,
        SEED,
        &cfg,
        None,
        None,
        Resume { start: Some(&start), capture_final_rng: None },
    )
    .expect_err("a reactive model must be rejected at the start-from-state seam");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("reactive"),
        "reactive rejection must name the limitation; got: {msg}"
    );
}
