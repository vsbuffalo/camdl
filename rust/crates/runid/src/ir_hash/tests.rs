//! Tests for the hand-written `ir`-tree `ContentAddressed` impls:
//! a golden-hash regression pin over a representative `Model`, the ±0.0
//! structural-float distinctness the IR treats as real, and a check that
//! the sorted-map rule flows through the hand impls (insertion order of a
//! `rate_grad` map does not change the hash).

use std::collections::HashMap;

use ir::expr::{BinOp, Expr, ProjectedExpr};
use ir::intervention::{Action, Intervention, InterventionSchedule, RecurringSchedule, SetAction};
use ir::model::{
    BalanceSpec, Binding, Compartment, CompartmentKind, Dimension, InitialConditions, Model,
    ModelStructure, OutputConfig, OutputSchedule, Preset, RegularOutputSchedule, SimulationConfig,
};
use ir::observation::{
    Likelihood, ObservationModel, ObservationSchedule, PoissonLikelihood, Projection, RegularSchedule,
};
use ir::ode_equation::OdeEquation;
use ir::parameter::{Parameter, PriorDist, Transform, UniformPrior};
use ir::table::{OobPolicy, Table, TableSource};
use ir::time_func::{Sinusoidal, TimeFuncKind, TimeFunction};
use ir::transition::{DrawMethod, StoichiometryEntry, Transition, TransitionMetadata};

use crate::hash::ContentAddressed;

/// A broad, representative SIR-with-seasonality model that exercises a wide
/// slice of the IR tree: integer + real compartments, a transition with a
/// rate expr + rate_grad + metadata, an ODE equation, a sinusoidal forcing,
/// an inline table, a recurring intervention, a Poisson observation, a
/// parameter with bounds/prior/transform/dim, a binding, explicit initial
/// conditions, a regular output schedule, a preset, model structure, and a
/// balance constraint.
fn representative_model() -> Model {
    let rate = Expr::bin_op(
        BinOp::Mul,
        Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
        Expr::pop("I"),
    );
    let mut rate_grad: HashMap<String, Expr> = HashMap::new();
    rate_grad.insert(
        "beta".into(),
        Expr::bin_op(BinOp::Mul, Expr::pop("S"), Expr::pop("I")),
    );

    let mut ic: HashMap<String, f64> = HashMap::new();
    ic.insert("S".into(), 999.0);
    ic.insert("I".into(), 1.0);
    ic.insert("R".into(), 0.0);

    let mut preset_params: HashMap<String, f64> = HashMap::new();
    preset_params.insert("beta".into(), 1.5);

    let mut compartment_dims: HashMap<String, Vec<String>> = HashMap::new();
    compartment_dims.insert("S".into(), vec!["age".into()]);

    Model {
        name: "sir_seasonal".into(),
        version: "1.0".into(),
        time_unit: "days".into(),
        description: Some("test model".into()),
        origin: Some("2020-01-01".into()),
        origin_rata_die: Some(737_425),
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Real },
        ],
        transitions: vec![Transition {
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate,
            metadata: Some(TransitionMetadata {
                origin_kind: Some("transmission".into()),
                source_compartment: Some("S".into()),
                dest_compartment: Some("I".into()),
            }),
            draw_method: DrawMethod::Poisson,
            rate_grad,
            lineage: None,
        }],
        ode_equations: vec![OdeEquation {
            compartment: "R".into(),
            derivative: Expr::bin_op(BinOp::Mul, Expr::param("gamma"), Expr::pop("I")),
        }],
        time_functions: vec![TimeFunction {
            name: "seasonal".into(),
            kind: TimeFuncKind::Sinusoidal(Sinusoidal {
                amplitude: Expr::const_(0.1),
                period: Expr::const_(365.0),
                phase: Expr::const_(0.0),
                baseline: Expr::const_(1.0),
            }),
            dim: (0, 0),
        }],
        tables: vec![Table {
            name: "contact".into(),
            source: TableSource::Inline { values: vec![Expr::const_(1.0), Expr::const_(0.5)] },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        }],
        interventions: vec![Intervention {
            name: "pulse_vax".into(),
            base_name: None,
            schedule: InterventionSchedule::Recurring(RecurringSchedule {
                start: 100.0,
                period: 365.0,
                end: 1000.0,
                at_day: Some(50.0),
            }),
            actions: vec![Action::Set(SetAction {
                compartment: "I".into(),
                value: Expr::const_(0.0),
            })],
            kind: ir::intervention::InterventionKind::Scenario,
        }],
        observations: vec![ObservationModel {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn { name: "cases".into(), role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count) },
            ],
            scored: "cases".into(),
            emit_schedule: Some(ObservationSchedule::Regular(RegularSchedule {
                start: 0.0,
                step: 7.0,
                end: 364.0,
            })),
            projection: Projection::CumulativeFlow("infection".into()),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: Expr::bin_op(
                    BinOp::Mul,
                    Expr::param("rho"),
                    Expr::Projected(ProjectedExpr { projected: () }),
                ),
            }),
        }],
        parameters: vec![Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Estimated { init: Some(0.5), bounds: Some((0.0, 2.0)), prior: ir::parameter::PriorSpec::Dist(PriorDist::Uniform(UniformPrior { lower: 0.0, upper: 2.0 })), transform: Transform::Log }, param_kind: Some(ir::parameter::ParamKind::Rate), param_dim: Some((0, -1)) }],
        bindings: vec![Binding {
            name: "N".into(),
            expr: Expr::pop_sum(vec!["S".into(), "I".into(), "R".into()]),
        }],
        initial_conditions: InitialConditions::Explicit(ic),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule {
                start: 0.0,
                step: 1.0,
                end: 365.0,
            }),
            format: "tsv".into(),
            trajectory: true,
            observations: true,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 365.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(42),
        },
        presets: vec![Preset {
            name: "high_beta".into(),
            label: "High beta".into(),
            params: preset_params,
            enable: vec![],
            disable: vec![],
            scale: HashMap::new(),
            compose: vec![],
            t_end: None,
        }],
        model_structure: Some(ModelStructure {
            dimensions: vec![Dimension {
                name: "age".into(),
                values: vec!["young".into(), "old".into()],
            }],
            compartment_dims,
            base_compartments: vec!["S".into(), "I".into(), "R".into()],
            transmission_transitions: vec!["infection".into()],
            infectious_compartments: vec!["I".into()],
        }),
        balance: Some(BalanceSpec { target: "R".into(), expr: Expr::pop("R") }),
        identity_tracked_compartments: vec![],
    }
}

/// Golden-hash regression. A fixed `Model` → a committed 64-hex digest.
/// Only a `HASH_VERSION` bump or an intentional encoding change may move
/// this value; an unintended change to any hand impl trips it.
#[test]
fn model_golden_hash() {
    // Updated for IR 0.12 (observation data-entry, 2026-06-10): the
    // `ObservationModel` gained `source`/`columns`/`scored` and renamed
    // `schedule` -> optional `emit_schedule`. All hash into the content
    // address, so every run_id moves; the version handshake (ir/VERSION ->
    // 0.12) signposts it. (Earlier moves: gh#191 ParamValue ADT at 0.11;
    // param_kind/kind enum-ification at 0.10; table OOB Clamp/Wrap -> Error.)
    const GOLDEN: &str = "4387f80bbbbd67410ae95abf13da602ff77030d04505600c65656bf915959baf";
    let got = representative_model().content_hash().to_hex();
    assert_eq!(got, GOLDEN, "ir Model golden hash changed (got {got})");
}

/// `Const(0.0)` vs `Const(-0.0)` must produce *distinct* hashes — the
/// structural-IR-float policy keeps signed zero observable, matching
/// `ConstExpr::PartialEq` (which compares `to_bits()`).
#[test]
fn ir_const_signed_zero_is_distinct() {
    let pos = Expr::const_(0.0).content_hash();
    let neg = Expr::const_(-0.0).content_hash();
    assert_ne!(pos, neg, "IR Const(±0.0) must hash distinctly");

    // And inside a full Model: flipping one init-condition value to -0.0
    // changes the model hash.
    let mut m_pos = representative_model();
    let mut m_neg = representative_model();
    if let InitialConditions::Explicit(ref mut map) = m_pos.initial_conditions {
        map.insert("R".into(), 0.0);
    }
    if let InitialConditions::Explicit(ref mut map) = m_neg.initial_conditions {
        map.insert("R".into(), -0.0);
    }
    assert_ne!(
        m_pos.content_hash(),
        m_neg.content_hash(),
        "a -0.0 init condition must change the model hash"
    );
}

/// The sorted-map rule must flow through the hand impls: building the same
/// model twice with different `rate_grad` insertion order yields the same
/// hash. (We add a second gradient key in two orders.)
#[test]
fn rate_grad_map_order_invariant() {
    let build = |order_ab: bool| -> Model {
        let mut m = representative_model();
        let mut rg: HashMap<String, Expr> = HashMap::new();
        let a = ("beta".to_string(), Expr::pop("S"));
        let b = ("gamma".to_string(), Expr::pop("I"));
        if order_ab {
            rg.insert(a.0.clone(), a.1.clone());
            rg.insert(b.0.clone(), b.1.clone());
        } else {
            rg.insert(b.0.clone(), b.1.clone());
            rg.insert(a.0.clone(), a.1.clone());
        }
        m.transitions[0].rate_grad = rg;
        m
    };
    assert_eq!(
        build(true).content_hash(),
        build(false).content_hash(),
        "rate_grad insertion order must not change the model hash"
    );
}

/// Negative control: a structurally different model must hash differently
/// (guards against a hand impl that drops a field entirely).
#[test]
fn changing_a_field_changes_the_hash() {
    let base = representative_model().content_hash();

    let mut m_name = representative_model();
    m_name.name = "different".into();
    assert_ne!(base, m_name.content_hash(), "name change must matter");

    let mut m_tend = representative_model();
    m_tend.simulation.t_end = 730.0;
    assert_ne!(base, m_tend.content_hash(), "t_end change must matter");

    let mut m_out = representative_model();
    m_out.output.times = OutputSchedule::AtTimes(vec![1.0, 2.0, 3.0]);
    assert_ne!(base, m_out.content_hash(), "output schedule change must matter");
}

/// Run-id stability: every `Projection` variant has a PERMANENT hash index.
/// This pins each variant's content hash. The `CumulativeFlowSum` addition
/// (gh#160) originally renumbered `CurrentPop`/`CurrentPopSum`/`DerivedExpr`,
/// silently churning the run_id of every prevalence/derived-projection model;
/// nothing caught it because `representative_model()` only exercises the
/// index-0 variant. A future insert-and-renumber (or a reorder of the
/// `ir_hash` arms) now trips this. The four pre-existing variants keep their
/// pre-gh#160 indices (0-3), so their pins are unchanged and existing run_ids
/// are preserved; `CumulativeFlowSum` is appended at 4.
#[test]
fn projection_variant_hashes_are_pinned() {
    let hex = |p: Projection| p.content_hash().to_hex();
    // Indices 0-3 are byte-identical to the pre-gh#160 `ir_hash` arms, so
    // these four pins equal existing run_ids (no churn); 4 is the appended
    // `CumulativeFlowSum`.
    assert_eq!(
        hex(Projection::CumulativeFlow("x".into())),
        "bae333c82bc85a194d4899c1c76fd8f50120f506f4bf07ccbf4e6f1681a6c38e"
    );
    assert_eq!(
        hex(Projection::CurrentPop("x".into())),
        "bce336b063891b80dd2c16513f0e4938597ee2b1d600425e56ee0f7b88e5f30f"
    );
    assert_eq!(
        hex(Projection::CurrentPopSum(vec!["x".into()])),
        "2b4a65d0eb4d300730fee44b465903c6b615464a67ed503f4421a6ccaac9e8d6"
    );
    assert_eq!(
        hex(Projection::DerivedExpr(Expr::const_(1.0))),
        "eea65976e3ecc65043310f5f9ba5df473ade14554ee2f2f05b761142e73257cf"
    );
    assert_eq!(
        hex(Projection::CumulativeFlowSum(vec!["x".into()])),
        "439f6c25062e7e8e22487f37d5d2536725ae6487a902fd1b2e4cc9e51e54b0a1"
    );
}
