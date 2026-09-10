//! Cross-writer flow/Δstate reconciliation invariant (gh#270).
//!
//! Every writer of a `(state, flows)` trajectory must honour the initial-row
//! convention documented on [`sim::state::Trajectory`]: the first row is the
//! initial-condition snapshot at `t_start` with **zeroed** flows, so the
//! aggregate identity
//!
//!     Σ flow_<transition> == −Δcompartment   (over the whole path)
//!
//! holds for a compartment whose only dynamics are transitions (here `S`,
//! which leaves a basic SIR only via `infection`). gh#270 was a writer that
//! dropped the `t_start` row: every consecutive step still reconciled, but the
//! aggregate was off by exactly the first interval's flow — a defect a per-step
//! check is structurally blind to (it exempts the first row). This test asserts
//! the property directly, across every forward backend.
//!
//! The PGAS smoother — the writer the bug was actually filed against — is
//! guarded separately, where its `SubstepRecord → Snapshot` adapter lives:
//!   * unit: `sim::inference::pgas::grid_tests::gh270_seed_stratum_flow_reconciles_with_s_depletion`
//!   * end-to-end: `cli/tests/pgas_trajectory_coherence.rs` (aggregate check).
//!
//! Together these cover all four flow-emitting writers.

use std::path::Path;

use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, OdeConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim, GillespieSim, OdeSim,
};

fn golden_path(name: &str) -> String {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest)
        .join("../../../ocaml/golden")
        .join(format!("{}.ir.json", name))
        .to_string_lossy()
        .to_string()
}

/// Load `sir_basic` with its baseline preset applied (a seeded SIR: `I₀ > 0`,
/// `S` leaves only via `infection`).
fn load_seeded_sir() -> CompiledModel {
    let contents = std::fs::read_to_string(golden_path("sir_basic"))
        .unwrap_or_else(|e| panic!("read sir_basic: {e}"));
    let mut model: ir::Model =
        ir::from_str(&contents).unwrap_or_else(|e| panic!("parse sir_basic: {e}"));
    if let Some(preset) = model.presets.first().cloned() {
        for p in &mut model.parameters {
            if let Some(&v) = preset.params.get(&p.name) {
                p.value = p.value.with_value(v);
            }
        }
    }
    CompiledModel::new(model).unwrap_or_else(|e| panic!("compile sir_basic: {e:?}"))
}

fn s_local_idx(c: &CompiledModel) -> usize {
    let g = *c.comp_index.get("S").expect("S compartment");
    c.global_to_int[g].expect("S is an integer compartment")
}

fn infection_tr_idx(c: &CompiledModel) -> usize {
    c.model
        .transitions
        .iter()
        .position(|t| t.name == "infection")
        .expect("infection transition")
}

/// Assert the initial-row convention + aggregate reconciliation on one path.
/// `tol` is the absolute slack on the aggregate (0 for the integer backends;
/// a small value for the real-valued ODE flows).
fn assert_reconciles(
    label: &str,
    traj: &sim::state::Trajectory,
    t_start: f64,
    s_idx: usize,
    inf_idx: usize,
    tol: f64,
) {
    let snaps = &traj.snapshots;
    assert!(!snaps.is_empty(), "{label}: empty trajectory");

    // (1) Structural: the first row is the t_start initial-condition row with
    // zeroed flows. This is the direct guard against dropping the row — exact
    // and deterministic regardless of whether the first interval had any flow.
    assert!(
        (snaps[0].t - t_start).abs() < 1e-9,
        "{label}: first row must be at t_start={t_start}, got {}",
        snaps[0].t
    );
    for i in 0..snaps[0].flows.len() {
        assert_eq!(
            snaps[0].flows.value(i),
            0.0,
            "{label}: the t_start row must carry zeroed flows (transition {i})"
        );
    }

    // (2) Aggregate: Σ flow_infection == S₀ − S_final over the WHOLE path.
    let s0 = snaps[0].int_state.counts[s_idx] as f64;
    let s_final = snaps.last().unwrap().int_state.counts[s_idx] as f64;
    let sum_inf: f64 = snaps.iter().map(|sn| sn.flows.value(inf_idx)).sum();
    assert!(
        (sum_inf - (s0 - s_final)).abs() <= tol,
        "{label}: Σ flow_infection ({sum_inf}) must equal S₀−S_final ({}) within {tol} \
         — initial-row convention violated (gh#270)",
        s0 - s_final
    );
}

#[test]
fn chain_binomial_path_reconciles_flow_with_s_depletion() {
    let c = load_seeded_sir();
    let (s_idx, inf_idx) = (s_local_idx(&c), infection_tr_idx(&c));
    let t_start = c.model.simulation.t_start;
    let params = c.default_params.clone();
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig {
        t_start,
        t_end: c.model.simulation.t_end.min(60.0),
        dt: 1.0,
    });
    // Many seeds: at least some have flow_infection in the first interval, so a
    // dropped t_start row fails the aggregate too (not only the structural check).
    for seed in 0..20u64 {
        let traj = ChainBinomialSim.run(&c, &params, seed, &cfg).unwrap();
        assert_reconciles(
            &format!("chain_binomial seed={seed}"),
            &traj,
            t_start,
            s_idx,
            inf_idx,
            0.0,
        );
    }
}

#[test]
fn gillespie_path_reconciles_flow_with_s_depletion() {
    let c = load_seeded_sir();
    let (s_idx, inf_idx) = (s_local_idx(&c), infection_tr_idx(&c));
    let t_start = c.model.simulation.t_start;
    let params = c.default_params.clone();
    let cfg = SimConfig::Gillespie(GillespieConfig {
        t_start,
        t_end: c.model.simulation.t_end.min(60.0),
        output_dt: Some(1.0),
    });
    for seed in 0..20u64 {
        let traj = GillespieSim.run(&c, &params, seed, &cfg).unwrap();
        assert_reconciles(
            &format!("gillespie seed={seed}"),
            &traj,
            t_start,
            s_idx,
            inf_idx,
            0.0,
        );
    }
}

#[test]
fn ode_path_reconciles_flow_with_s_depletion() {
    let c = load_seeded_sir();
    let (s_idx, inf_idx) = (s_local_idx(&c), infection_tr_idx(&c));
    let t_start = c.model.simulation.t_start;
    let params = c.default_params.clone();
    let cfg = SimConfig::Ode(OdeConfig {
        t_start,
        t_end: c.model.simulation.t_end.min(60.0),
        dt: 1.0,
    });
    // ODE is deterministic; flows are real-valued (augmented-flow integration),
    // so the integer S column and the real flow sum reconcile within a small
    // tolerance (rounding of the reported counts + integration error).
    let traj = OdeSim.run(&c, &params, 0, &cfg).unwrap();
    assert_reconciles("ode", &traj, t_start, s_idx, inf_idx, 1.0);
}
