//! The sibling of the forcing-coefficient freeze: an inline-table *value* that
//! references a parameter must be evaluated live at lookup, not baked at
//! construction. Same root cause (`eval_table_expr` against `default_params`),
//! same fix (store the value as a live `ResolvedExpr`). Proposal
//! `2026-06-09-const-parametric-forcing.md` §3/§5.
//!
//! A minimal decay model `S --> D @ table_lookup(k_tbl, 0) * S` whose only
//! transmission coefficient is the inline-table value `k`. Build the model
//! once, vary `k` in the live slice, and assert the trajectory responds.
//! Against the frozen (pre-fix) code the table value is baked to its
//! default-param value and this FAILS (byte-identical trajectory).

use std::collections::HashMap;
use ir::{
    expr::{Expr, TableLookupExpr, TableLookupWrap},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::{ParamValue, Parameter},
    table::{OobPolicy, Table, TableSource},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    state::Trajectory,
    ChainBinomialSim,
};

fn fixed(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}

fn table_lookup(table: &str, index: f64) -> Expr {
    Expr::TableLookup(TableLookupWrap {
        table_lookup: TableLookupExpr { table: table.into(), indices: vec![Expr::const_(index)] },
    })
}

/// `S --> D` with rate `table_lookup(k_tbl, 0) * S`; `k_tbl = [k]`.
fn decay_model() -> Model {
    Model {
        name: "table_decay".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None, origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "D".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            name: "decay".into(),
            stoichiometry: vec![StoichiometryEntry("S".into(), -1), StoichiometryEntry("D".into(), 1)],
            rate: Expr::bin_op(ir::expr::BinOp::Mul, table_lookup("k_tbl", 0.0), Expr::pop("S")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad: HashMap::new(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![Table {
            name: "k_tbl".into(),
            source: TableSource::Inline { values: vec![Expr::param("k")] },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        }],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![fixed("k", 0.1), fixed("S0", 1000.0)],
        initial_conditions: InitialConditions::Parameterized(
            HashMap::from([("S".to_string(), Expr::param("S0"))]),
        ),
        output: OutputConfig {
            // Capture the decay over time, not just t=0 — else both runs share
            // the identical initial snapshot and the assertion is vacuous.
            times: OutputSchedule::AtTimes(vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0, t_end: 30.0,
            time_semantics: "continuous".into(),
            dt: None, rng_seed: Some(42),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![],
    }
}

fn all_counts(traj: &Trajectory) -> Vec<i64> {
    traj.snapshots.iter()
        .flat_map(|s| s.int_state.counts.iter().copied())
        .collect()
}

fn run(compiled: &CompiledModel, params: &[f64]) -> Vec<i64> {
    let cfg = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 30.0, dt: 1.0 });
    all_counts(&ChainBinomialSim.run(compiled, params, 7, &cfg).expect("run failed"))
}

#[test]
fn inline_table_value_param_is_live() {
    let compiled = CompiledModel::new(decay_model()).unwrap();

    // Build once; vary the table-coefficient param `k` only in the live slice.
    let p_lo = compiled.default_params.clone();
    let mut p_hi = p_lo.clone();
    p_hi[compiled.param_index["k"]] = 0.3;

    let lo = run(&compiled, &p_lo);
    let hi = run(&compiled, &p_hi);

    assert_ne!(lo, hi,
        "varying inline-table value param `k` (0.1 → 0.3) in the live slice must \
         change the decay trajectory; identical means the table value is frozen \
         at construction");
}
