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
    sir_model_at(0.0, [5.0, 15.0], output_times, t_end)
}

/// [`sir_model`] on a window starting at `t_start`, with the two interventions
/// at `iv_at` (pre-T*, post-T*). gh#964 needs the same model placed at a large
/// time coordinate.
fn sir_model_at(
    t_start: f64,
    iv_at: [f64; 2],
    output_times: Vec<f64>,
    t_end: f64,
) -> (CompiledModel, Vec<f64>) {
    let n_expr = Expr::pop_sum(vec!["S".into(), "I".into(), "R".into(), "V".into()]);
    let beta_s_i = Expr::bin_op(
        BinOp::Mul,
        Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
        Expr::pop("I"),
    );
    let infection_rate = Expr::bin_op(BinOp::Div, beta_s_i, n_expr);
    let recovery_rate = Expr::bin_op(BinOp::Mul, Expr::param("gamma"), Expr::pop("I"));

    let tr = |name: &str, from: &str, to: &str, rate: Expr| Transition {
        rate_state_grad: Default::default(),
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
        constants: vec![],
        ic_grad: Default::default(),
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
        interventions: vec![iv("sia_pre", iv_at[0], 0.3), iv("sia_post", iv_at[1], 0.5)],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter { name: "beta".into(), value: ParamValue::Fixed { value: 0.4 }, param_kind: None, param_dim: None },
            Parameter { name: "gamma".into(), value: ParamValue::Fixed { value: 0.2 }, param_kind: None, param_dim: None },
        ],
        initial_conditions: InitialConditions::constants({
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
            t_start,
            t_end,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
            integrator: Default::default(),
            t_end_anchor: None,
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
    let (int_s, real_s) = model.initial_state_mean(&params).unwrap();
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
    let (int_s, real_s) = compiled.initial_state_mean(&params).unwrap();
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

/// 5. **Splice at a large time coordinate (gh#964)** — the headline invariant
///    with the window placed above 2^13 = 8192, on a weekly emit grid at a
///    fractional `dt`. The resume time T* = 8196 is on the grid (8000 + 28·7).
///    An absolute output tolerance of 1e-12 is below one ulp of t there
///    (2^-39 ≈ 1.8e-12), so the re-seat drain consumed the T* output and the
///    run was refused as off-grid; the tolerance must scale with |t|.
#[test]
fn splice_above_8192_is_byte_identical_to_the_continuous_tail() {
    const T0: f64 = 8000.0;
    const T_STAR_BIG: f64 = 8196.0; // 8000 + 28·7
    const T_END_BIG: f64 = 8231.0; // 8000 + 33·7
    const DT: f64 = 0.25;
    let weekly = |end: f64| -> Vec<f64> {
        (0..).map(|k| T0 + 7.0 * k as f64).take_while(|&t| t <= end).collect()
    };
    let ivs = [8100.0, 8210.0];

    let (head_model, head_params) = sir_model_at(T0, ivs, weekly(T_STAR_BIG), T_STAR_BIG);
    let mut captured = StatefulRng::new(0);
    let head_cfg = ChainBinomialConfig { t_start: T0, t_end: T_STAR_BIG, dt: DT };
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
    let at_tstar = traj_head.snapshots.last().expect("head trajectory is non-empty");
    assert_eq!(at_tstar.t, T_STAR_BIG, "head's last snapshot is at T*");

    let (tail_model, tail_params) = sir_model_at(T0, ivs, weekly(T_END_BIG), T_END_BIG);
    let start = StartState {
        int_s: at_tstar.int_state.clone(),
        real_s: at_tstar.real_state.clone(),
        rng: Some(captured),
    };
    let tail_cfg = ChainBinomialConfig { t_start: T_STAR_BIG, t_end: T_END_BIG, dt: DT };
    let traj_tail = run_chain_binomial_with_observer(
        &tail_model,
        &tail_params,
        SEED,
        &tail_cfg,
        None,
        None,
        Resume { start: Some(&start), capture_final_rng: None },
    )
    .expect("an on-grid T* above 8192 must resume");
    assert_eq!(traj_tail.snapshots[0].t, T_STAR_BIG, "the resumed run's first row is T*");

    let full_cfg = ChainBinomialConfig { t_start: T0, t_end: T_END_BIG, dt: DT };
    let traj_full = run_chain_binomial(&tail_model, &tail_params, SEED, &full_cfg).unwrap();
    let full_tail: Vec<_> = traj_full.snapshots.iter().filter(|s| s.t >= T_STAR_BIG).collect();
    assert_eq!(full_tail.len(), traj_tail.snapshots.len(), "same tail rows");
    for (i, (f, t)) in full_tail.iter().zip(&traj_tail.snapshots).enumerate() {
        assert_eq!(f.t, t.t, "row times align");
        assert_eq!(f.int_state.counts, t.int_state.counts, "state at t={}", f.t);
        // The T* row carries zeroed flows on the resumed side (initial-row
        // convention); every later row must match in flows too.
        if i > 0 {
            assert_eq!(f.flows.as_int(), t.flows.as_int(), "flows at t={}", f.t);
        }
    }
    assert_eq!(full_tail.len(), 6, "T* plus the five weekly rows to 8231");

    // Not vacuous: the post-T* intervention at 8210 moved mass into V.
    let v = tail_model.model.compartments.iter().position(|c| c.name == "V").unwrap();
    assert!(
        traj_tail.snapshots.last().unwrap().int_state.counts[v]
            > traj_tail.snapshots[0].int_state.counts[v],
        "the post-T* SIA must move mass into V"
    );
}

/// 6. **Off-grid T\* above 8192 is still refused (gh#964)** — the scaled
///    tolerance must not swallow a genuine misalignment. T* = 8197 is one day
///    off the weekly grid; the message names T* and the neighbouring emits.
#[test]
fn off_grid_resume_above_8192_is_rejected_naming_the_neighbours() {
    let out: Vec<f64> = (0..=33).map(|k| 8000.0 + 7.0 * k as f64).collect();
    let (model, params) = sir_model_at(8000.0, [8100.0, 8210.0], out, 8231.0);
    let (int_s, real_s) = model.initial_state_mean(&params).unwrap();
    let start = StartState { int_s, real_s, rng: None };
    let cfg = ChainBinomialConfig { t_start: 8197.0, t_end: 8231.0, dt: 0.25 };
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
    assert!(msg.contains("8197"), "names T*; got: {msg}");
    assert!(msg.contains("8196") && msg.contains("8203"), "names the neighbouring emits; got: {msg}");
}

/// 6. **A fork at an effect instant does not fire it twice** (proposal
///    2026-10-04, D1 and `Origin::Resumed`). With an intervention AT T*, the head
///    run [0, T*] fires it at the end of its last step, so the saved state@T* is
///    already post-effect. The resumed run starts at T* with that effect in its
///    origin batch's position — and must not apply it again. The splice is then
///    byte-identical to the continuous run, which fires it once.
#[test]
fn a_fork_at_an_effect_instant_does_not_fire_it_twice() {
    let model_at = |outs: Vec<f64>, t_end: f64| sir_model_at(0.0, [T_STAR, 15.0], outs, t_end);
    let (head_model, head_params) = model_at(ints(T_STAR as u64), T_STAR);
    let mut captured = StatefulRng::new(0);
    let traj_head = run_chain_binomial_with_observer(
        &head_model,
        &head_params,
        SEED,
        &ChainBinomialConfig { t_start: 0.0, t_end: T_STAR, dt: 1.0 },
        None,
        None,
        Resume { start: None, capture_final_rng: Some(&mut captured) },
    )
    .unwrap();
    let at_tstar = traj_head.snapshots.last().unwrap();
    let (tail_model, tail_params) = model_at(ints(T_END as u64), T_END);
    let start = StartState {
        int_s: at_tstar.int_state.clone(),
        real_s: at_tstar.real_state.clone(),
        rng: Some(captured),
    };
    let traj_tail = run_chain_binomial_with_observer(
        &tail_model,
        &tail_params,
        SEED,
        &ChainBinomialConfig { t_start: T_STAR, t_end: T_END, dt: 1.0 },
        None,
        None,
        Resume { start: Some(&start), capture_final_rng: None },
    )
    .unwrap();
    let (full_model, full_params) = model_at(ints(T_END as u64), T_END);
    let traj_full = run_chain_binomial(
        &full_model,
        &full_params,
        SEED,
        &ChainBinomialConfig { t_start: 0.0, t_end: T_END, dt: 1.0 },
    )
    .unwrap();
    for fs in traj_full.snapshots.iter().filter(|s| s.t >= T_STAR - 1e-9) {
        let ts = traj_tail.snapshots.iter().find(|s| (s.t - fs.t).abs() < 1e-9).unwrap();
        assert_eq!(
            fs.int_state.counts, ts.int_state.counts,
            "state diverges at t={}: the resumed run re-applied the effect at T*",
            fs.t
        );
    }
    // Non-vacuity: the effect at T* moved mass into V in the head run.
    let v = full_model.model.compartments.iter().position(|c| c.name == "V").unwrap();
    let before = traj_head.snapshots.iter().find(|s| (s.t - (T_STAR - 1.0)).abs() < 1e-9).unwrap();
    assert!(at_tstar.int_state.counts[v] > before.int_state.counts[v], "the T* effect must fire in the head");
}
