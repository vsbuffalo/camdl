//! Tests for the hand-written `ir`-tree `ContentAddressed` impls:
//! a golden-hash regression pin over a representative `Model`, the ±0.0
//! structural-float distinctness the IR treats as real, and a check that
//! the sorted-map rule flows through the hand impls (insertion order of a
//! `rate_grad` map does not change the hash).

use std::collections::HashMap;

use ir::expr::{BinOp, Expr, ProjectedExpr};
use ir::intervention::{Action, Intervention, InterventionSchedule, RecurringSchedule, SetAction};
use ir::model::{
    BalanceSpec, Binding, Compartment, CompartmentKind, Dimension, InitialConditions, Integrator,
    Model, ModelStructure, OutputConfig, OutputSchedule, Preset, RegularOutputSchedule,
    SimulationConfig,
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
    let mut rate_grad: HashMap<String, ir::deriv::DerivEntry> = HashMap::new();
    rate_grad.insert(
        "beta".into(),
        ir::deriv::DerivEntry::Grad(Expr::bin_op(BinOp::Mul, Expr::pop("S"), Expr::pop("I"))),
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
        ic_grad: Default::default(),
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
            rate_state_grad: Default::default(),
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
            lag: None,
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
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::Recurring(RecurringSchedule {
                start: 100.0,
                period: 365.0,
                end: 1000.0,
                at_day: Some(50.0),
            })),
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
            stratum: vec![],
            projection: Projection::CumulativeFlow("infection".into()),
            likelihood: Likelihood::Poisson(PoissonLikelihood {
                rate: ir::Diffable::new(Expr::bin_op(
                    BinOp::Mul,
                    Expr::param("rho"),
                    Expr::Projected(ProjectedExpr { projected: () }),
                )),
            }),
        }],
        parameters: vec![Parameter { name: "beta".into(), value: ir::parameter::ParamValue::Estimated { init: Some(0.5), bounds: Some((0.0, 2.0)), prior: ir::parameter::PriorSpec::Dist(PriorDist::Uniform(UniformPrior { lower: 0.0, upper: 2.0 })), transform: Transform::Log }, param_kind: Some(ir::parameter::ParamKind::Rate), param_dim: Some((0, -1)) }],
        bindings: vec![Binding {
            name: "N".into(),
            expr: Expr::pop_sum(vec!["S".into(), "I".into(), "R".into()]),
        }],
        per_eval_bindings: vec![],
        initial_conditions: InitialConditions::Explicit(ic),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule {
                start: 0.0,
                step: 1.0,
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
            integrator: Default::default(),
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
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

/// Golden-hash regression. A fixed `Model` → a committed 64-hex digest.
/// Only a `HASH_VERSION` bump or an intentional encoding change may move
/// this value; an unintended change to any hand impl trips it.
#[test]
fn model_golden_hash() {
    // Updated for IR 0.17 (gh#204 reactive interventions, 2026-06-18): the
    // intervention `schedule: InterventionSchedule` field became
    // `fire: FireSource`, so the `Intervention` content address now hashes the
    // `FireSource` enum layer (variant tag + inner schedule) instead of the
    // bare schedule. All hash into the content address, so every run_id moves;
    // the version handshake (ir/VERSION -> 0.17) signposts it. (Earlier moves:
    // observation data-entry at 0.12; gh#191 ParamValue ADT at 0.11;
    // param_kind/kind enum-ification at 0.10; table OOB Clamp/Wrap -> Error.)
    // gh#272 LICM (ir/VERSION -> 0.19): `Model::hash_into` now folds
    // `per_eval_bindings`. The empty Vec's length prefix shifts every model hash,
    // so the GOLDEN moves once at the schema bump even with the pass default-off
    // (a deliberate, version-bumped re-key — see the proposal's "Run identity").
    // gh#314 (ir/VERSION -> 0.20): `TimeFunction::hash_into` now folds the
    // optional `lag`. The representative model carries a forcing, so its
    // `Option<Expr>` presence tag (`None` -> one byte) shifts the model hash —
    // another deliberate, version-bumped re-key (forcing lag is run identity).
    // gh#143 (ir/VERSION -> 0.23): `RegularOutputSchedule` dropped `end` (the
    // output horizon collapsed onto `simulation.t_end`), so `hash_into` folds one
    // fewer f64 and the model hash shifts once — a deliberate, version-bumped
    // re-key (the horizon stays in identity via `simulation.t_end`).
    // gh#180 (ir/VERSION -> 0.24): unified obs-gradient autodiff. `Likelihood`
    // now folds each arg's `*_grad` map (and `DrawMethod::Overdispersed` its
    // `sigma_sq_grad`) right after the arg — the obs/σ² analogue of the
    // transition `rate_grad`, hashed into run identity. The representative model
    // carries a Poisson observation, so its empty `rate_grad` length-0 prefix
    // shifts the model hash once — a deliberate, version-bumped re-key (obs
    // gradients are run identity, mirroring `rate_grad`).
    // gh#342 (ir/VERSION -> 0.25): each differentiable obs arg is now a `Diffable`
    // (expr + classified grad), and `Likelihood::hash_into` folds them via the
    // derived `diffables()` traversal — so each arg gains a `Diffable` type-tag
    // and hashes as (expr, grad) together instead of expr-then-adjacent-grad. The
    // representative model's Poisson `rate` shifts the model hash once — a
    // deliberate, version-bumped re-key (the obs wire moved to the nested shape).
    // gh#342 P3 (ir/VERSION -> 0.26): the transition `rate_grad` value type moved
    // from bare `Expr` to the classified `DerivEntry`, so its `write_str_map` now
    // hashes each entry through `DerivEntry::hash_into` (type tag + `Grad` variant
    // + expr) instead of the bare expr. The representative model's `rate_grad`
    // (∂/∂beta) shifts the model hash once more — the same deliberate,
    // version-bumped re-key on the rate side.
    // gh#275 (ir/VERSION -> 0.27): the ODE gradient spine adds two compiler-emitted
    // derivative fields — `Transition::rate_state_grad` (∂rate/∂compartment, `J_x`)
    // and `Model::ic_grad` (∂init/∂θ, the sensitivity seed). Both are hashed into
    // run identity like `rate_grad` (a state gradient changes the fit, so it must
    // re-key). They are empty until the WrtPop/WrtParam passes emit them, but the
    // two new length-0 prefixes shift the model hash once at the bump — a
    // deliberate, version-bumped re-key (state/IC gradients are run identity).
    const GOLDEN: &str = "d62ed1b118c1724d50aa65f4bd0b5d373488b020af94fb0b155f65b23f37879f";
    let got = representative_model().content_hash().to_hex();
    assert_eq!(got, GOLDEN, "ir Model golden hash changed (got {got})");
}

/// gh#272: a non-empty `per_eval_bindings` must change the model hash. The
/// blocker the proposal flagged was that adding the field to the struct (and the
/// `hash_into` line) is invisible to `model_golden_hash` on its own — an empty Vec
/// hashes the same whether or not the field is folded. This pins that the field is
/// actually read by the hash, so flipping the LICM pass on re-keys `run_id`.
#[test]
fn ir_per_eval_bindings_changes_hash() {
    let base = representative_model().content_hash();
    let mut m = representative_model();
    m.per_eval_bindings.push(Binding {
        name: "__licm_0".into(),
        expr: Expr::param("beta"),
    });
    assert_ne!(
        base,
        m.content_hash(),
        "a non-empty per_eval_bindings must change the model hash (else flipping LICM \
         on would collide run_id with the off-form)"
    );
}

/// gh#275: a non-empty `rate_state_grad` (∂rate/∂compartment, `J_x`) must change
/// the model hash. Same blocker as `per_eval_bindings`: adding the struct field +
/// the `hash_into` line is invisible to `model_golden_hash` on its own (an empty
/// map hashes the same folded or not). This pins that the field is genuinely read,
/// so a model that computes a state gradient cannot collide `run_id` with one that
/// does not — a state gradient changes the fit and must re-key.
#[test]
fn ir_rate_state_grad_changes_hash() {
    use ir::deriv::{CompGradMap, DerivEntry};
    let base = representative_model().content_hash();
    let mut m = representative_model();
    let mut g = std::collections::HashMap::new();
    g.insert("S".to_string(), DerivEntry::Grad(Expr::param("beta")));
    m.transitions[0].rate_state_grad = CompGradMap(g);
    assert_ne!(
        base,
        m.content_hash(),
        "a non-empty rate_state_grad must change the model hash (state gradients are run identity)"
    );
}

/// gh#275: a non-empty `ic_grad` (∂init/∂θ, the forward-sensitivity seed) must
/// change the model hash — the IC/state sensitivity axis is run identity for the
/// same reason as `rate_grad`.
#[test]
fn ir_ic_grad_changes_hash() {
    use ir::deriv::DerivEntry;
    let base = representative_model().content_hash();
    let mut m = representative_model();
    let mut inner = std::collections::HashMap::new();
    inner.insert("beta".to_string(), DerivEntry::Grad(Expr::param("beta")));
    m.ic_grad.insert("I".to_string(), inner);
    assert_ne!(
        base,
        m.content_hash(),
        "a non-empty ic_grad must change the model hash (the IC sensitivity seed is run identity)"
    );
}

/// gh#275: the OUTER `ic_grad` map is hand-sorted in `Model::hash_into` (unlike
/// the inner param-maps, which ride `write_str_map`'s sort — pinned by
/// `rate_grad_map_order_invariant`). Build the same two-compartment `ic_grad` in
/// two insertion orders and assert an identical hash — pins that outer sort so a
/// future refactor that drops it cannot make `run_id` depend on `HashMap`
/// iteration order (the gh#160 non-determinism class).
#[test]
fn ic_grad_map_order_invariant() {
    let grad_for = |p: &str| -> HashMap<String, ir::deriv::DerivEntry> {
        let mut h = HashMap::new();
        h.insert(p.to_string(), ir::deriv::DerivEntry::Grad(Expr::param("beta")));
        h
    };
    let build = |order_ab: bool| -> Model {
        let mut m = representative_model();
        let a = ("I".to_string(), grad_for("beta"));
        let b = ("R".to_string(), grad_for("gamma"));
        if order_ab {
            m.ic_grad.insert(a.0.clone(), a.1.clone());
            m.ic_grad.insert(b.0.clone(), b.1.clone());
        } else {
            m.ic_grad.insert(b.0.clone(), b.1.clone());
            m.ic_grad.insert(a.0.clone(), a.1.clone());
        }
        m
    };
    assert_eq!(
        build(true).content_hash(),
        build(false).content_hash(),
        "ic_grad compartment insertion order must not change the model hash (the outer sort)"
    );
}

/// gh#275: injectivity — two `ic_grad`s that differ (here by compartment key)
/// must hash differently, not merely differ from empty. Pins that the outer
/// compartment key is folded into the digest (the encoding is length-prefixed
/// and thus injective; this locks that property).
#[test]
fn ic_grad_distinct_values_hash_differently() {
    let mk = |comp: &str| {
        let mut m = representative_model();
        let mut inner = HashMap::new();
        inner.insert("beta".to_string(), ir::deriv::DerivEntry::Grad(Expr::param("beta")));
        m.ic_grad.insert(comp.to_string(), inner);
        m.content_hash()
    };
    assert_ne!(
        mk("I"),
        mk("R"),
        "ic_grad keyed by different compartments must hash differently"
    );
}

/// Inverse polarity of `ir_per_eval_bindings_changes_hash`: a non-empty
/// `quantities` must NOT change the model hash. Quantities (proposal 2026-06-25)
/// are derived reports, deliberately excluded from `Model::hash_into` — the one
/// Model field outside the run-id walk — so adding a `quantities {}` block must
/// never re-key a model's sim/fit. This pins that the field is genuinely absent
/// from the hash; a future refactor that re-adds the walk line (e.g. a
/// derive-based `ContentAddressed`) would trip this.
#[test]
fn ir_quantities_excluded_from_hash() {
    use ir::quantity::{Quantity, QuantityBody, QuantitySource, TemporalReduce, ValueReduce};
    let base = representative_model().content_hash();
    let mut m = representative_model();
    m.quantities.push(Quantity {
        name: "peak".into(),
        stratum: vec![],
        body: QuantityBody::Reduced {
            source: QuantitySource::State(Expr::pop("I")),
            reduce: Some(TemporalReduce::Value(ValueReduce::Max)),
        },
        dimension: None,
    });
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty quantities must NOT change the model hash (quantities are \
         non-identity derived reports, excluded from Model::hash_into)"
    );
}

/// Symmetric with `ir_quantities_excluded_from_hash`: a non-empty `contrasts`
/// must NOT change the model hash. Contrasts (proposal 2026-06-25) are derived
/// counterfactual reports, like quantities deliberately excluded from
/// `Model::hash_into` — adding a `contrasts {}` block must never re-key a model's
/// sim/fit. This pins that the field is genuinely absent from the hash; a future
/// refactor that re-adds the walk line would trip it.
#[test]
fn ir_contrasts_excluded_from_hash() {
    use ir::contrast::{Contrast, ContrastExpr, RunNamespace};
    let base = representative_model().content_hash();
    let mut m = representative_model();
    m.contrasts.push(Contrast {
        name: "averted".into(),
        body: ContrastExpr::BinOp {
            op: BinOp::Sub,
            left: Box::new(ContrastExpr::RunMember {
                run: "no_sia".into(),
                ns: RunNamespace::Quantities,
                member: "total".into(),
            }),
            right: Box::new(ContrastExpr::RunMember {
                run: "with_sia".into(),
                ns: RunNamespace::Quantities,
                member: "total".into(),
            }),
        },
    });
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty contrasts must NOT change the model hash (contrasts are \
         non-identity derived reports, excluded from Model::hash_into)"
    );
}

/// gh#314: a forcing `lag` is run identity. Two models that differ only by a
/// forcing's evaluation-time shift produce different trajectories, so they must
/// re-key. This pins that `TimeFunction::lag` is actually folded into the hash
/// (an absent vs present lag, and two distinct lag values, all hash distinctly).
#[test]
fn ir_forcing_lag_changes_hash() {
    let base = representative_model().content_hash();

    let mut m_lag = representative_model();
    m_lag.time_functions[0].lag = Some(Expr::const_(10.0));
    assert_ne!(
        base,
        m_lag.content_hash(),
        "adding a forcing lag must change the model hash (lag is run identity)"
    );

    let mut m_lag2 = representative_model();
    m_lag2.time_functions[0].lag = Some(Expr::const_(20.0));
    assert_ne!(
        m_lag.content_hash(),
        m_lag2.content_hash(),
        "two distinct lag values must hash distinctly"
    );
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
        let mut rg: HashMap<String, ir::deriv::DerivEntry> = HashMap::new();
        let a = ("beta".to_string(), ir::deriv::DerivEntry::Grad(Expr::pop("S")));
        let b = ("gamma".to_string(), ir::deriv::DerivEntry::Grad(Expr::pop("I")));
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

/// gh#166: the chosen integrator is part of the model's content identity (it
/// changes the numerics), so it must flow into the run-id — but the `Rk4`
/// default must be hash-invisible so pre-gh#166 run-ids are unchanged (the
/// GOLDEN pin above is the proof of that, since `representative_model()` is
/// `Rk4`).
#[test]
fn integrator_choice_changes_run_id() {
    let with = |i: Integrator| {
        let mut m = representative_model();
        m.simulation.integrator = i;
        m.content_hash()
    };
    let rk4 = with(Integrator::Rk4);
    let rk45_default = with(Integrator::Rk45 { atol: None, rtol: None });
    let rk45_a = with(Integrator::Rk45 { atol: Some(1e-8), rtol: Some(1e-6) });
    let rk45_b = with(Integrator::Rk45 { atol: Some(1e-10), rtol: Some(1e-6) });

    // Explicit Rk4 == the default model (Rk4 is hash-invisible / omitted).
    assert_eq!(rk4, representative_model().content_hash(), "default rk4 must not move the run-id");
    // rk4 vs rk45, and tolerance variations, all hash distinctly.
    assert_ne!(rk4, rk45_default, "rk4 vs rk45 must hash distinctly");
    assert_ne!(rk45_default, rk45_a, "rk45 default-tols vs explicit tols must differ");
    assert_ne!(rk45_a, rk45_b, "a different atol must change the run-id");
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

// `#'` documentation is structurally outside `run_id`: it lives on the IR
// envelope's `docs` dictionary (crate `ir`), not on `Model`, so `content_hash`
// (a fold over `Model`) cannot see it. There is therefore no per-model doc
// field to pin here — the exclusion is by construction.

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
