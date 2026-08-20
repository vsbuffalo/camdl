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
            data_source: None,
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
            projection_state_grad: Default::default(),
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
            t_end_anchor: None,
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
            t_end_anchor: None,
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
    // gh#275 (∂arg/∂projected): `Diffable` gains `proj_grad` — the compiler-emitted
    // observation FACTOR-2 derivative (∂arg/∂projected, via the WrtProjected pass).
    // It is hashed like `grad` (an obs gradient change re-keys the fit), so the
    // representative model's Poisson `rate` (a bare `Projected` argument whose
    // proj_grad = 1) shifts the model hash once — a deliberate re-key on the obs
    // side, mirroring the `rate_grad`/`rate_state_grad` re-keys above.
    // gh#275 (∂projection/∂compartment, §1h): `ObservationModel` gains
    // `projection_state_grad` — the WrtPop gradient of a `DerivedExpr` (nonlinear)
    // projection, the factor-2 ingredient. Hashed like `rate_state_grad` (a
    // projection-gradient change re-keys a gradient fit); empty on the
    // representative model, so its length-0 prefix shifts the hash once at the
    // 0.29 bump — a deliberate, version-bumped re-key.
    // SV=2 (2026-07-16, proposal 2026-07-16-gradient-maps-out-of-run-identity.md):
    // model identity is now gradient-independent. All the compiler-derived gradient
    // maps folded above (`rate_grad`, `rate_state_grad`, `sigma_sq_grad`,
    // `projection_state_grad`, `ic_grad`, obs `grad`/`proj_grad`) are dropped from
    // `hash_into`, and the SV header bumps 1→2. The representative model carries a
    // `rate_grad` (∂/∂beta), so removing it AND the SV bump move the hash once — a
    // deliberate, version-bumped re-key (gradients are derived, not identity).
    const GOLDEN: &str = "33c669da9a280e8519592fe1ad7a4d1764b19b908d2fff6e52bbf61424948356";
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

// ── gh#442: the presentation strip, and why moving it did not re-key sim/fit ──

/// gh#442 moved the presentation strip from the caller
/// (`cli::resolve::model_digest`, which did `from_model(&normalize(m))`) into
/// `ModelDigest::from_model` itself (which now does `from_model(m)` =
/// `content_hash(normalize(m))`). The two agree **iff** the normalizer is
/// idempotent — and it is, because it assigns a constant rather than
/// transforming.
///
/// That is the whole no-collateral-re-key argument, made executable: the RHS
/// below (`from_model(&normalize(m))`) is *literally the pre-gh#442 sim/fit code
/// path*, so equality proves those two kinds' bytes did not move. If someone
/// later makes normalization non-idempotent (e.g. appends instead of assigns),
/// the double-normalize the sim/fit callers used to perform would silently
/// re-key them — this test is the tripwire.
#[test]
fn normalization_is_idempotent_and_sim_fit_bytes_unchanged() {
    use crate::inputs::{normalize_for_hash, EngineVersion, ModelDigest};

    // The fixture carries non-default presentation fields (`format = "tsv"`,
    // `time_semantics = "continuous"`), so this is not vacuous.
    let m = representative_model();
    assert!(!m.output.format.is_empty() && !m.simulation.time_semantics.is_empty());

    let digest = |model: &Model| {
        ModelDigest::from_model(model, "0.7".into(), EngineVersion("0.3.0".into())).content_hash()
    };
    assert_eq!(
        digest(&normalize_for_hash(&m)),
        digest(&m),
        "pre-gh#442 (caller normalizes, then from_model) and post-gh#442 (from_model \
         normalizes) must produce IDENTICAL bytes — else gh#442 collaterally re-keyed \
         sim and fit, which it is NOT sanctioned to do"
    );

    // The property that makes the above true, asserted directly.
    assert_eq!(
        normalize_for_hash(&normalize_for_hash(&m)).content_hash(),
        normalize_for_hash(&m).content_hash(),
        "normalize_for_hash must be idempotent"
    );

    // Non-vacuous: the strip actually does something on this fixture.
    assert_ne!(
        normalize_for_hash(&m).content_hash(),
        m.content_hash(),
        "the fixture must carry presentation fields the strip removes, else the \
         idempotence assertions above prove nothing"
    );
}

/// gh#442: the two presentation fields are inert in the model digest, and each
/// one on its own — a half-fix (one field stripped, the other not) must fail.
#[test]
fn presentation_fields_are_inert_in_the_model_digest() {
    use crate::inputs::{EngineVersion, ModelDigest};
    let digest = |model: &Model| {
        ModelDigest::from_model(model, "0.7".into(), EngineVersion("0.3.0".into())).content_hash()
    };
    let base = digest(&representative_model());

    let mut fmt = representative_model();
    fmt.output.format = "parquet".into();
    assert_eq!(base, digest(&fmt), "output.format must be inert in the model digest");

    let mut ts = representative_model();
    ts.simulation.time_semantics = "calendar".into();
    assert_eq!(base, digest(&ts), "simulation.time_semantics must be inert");

    // Negative control: a structural edit is NOT inert.
    let mut renamed = representative_model();
    renamed.name = "different".into();
    assert_ne!(base, digest(&renamed), "a model rename must move the model digest");
}

/// Model identity is gradient-independent (proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md): `rate_state_grad`
/// (∂rate/∂compartment, `J_x`, gh#275) is compiler-derived autodiff of `rate`, so
/// it is NOT hashed. A model that carries a state gradient must therefore produce
/// the SAME model hash as one that does not — the lean-vs-full guarantee that lets
/// `camdlc --no-state-grad` (gh#439) dispatch by method without re-keying.
/// (Previously asserted the opposite; flipped by the proposal.)
#[test]
fn ir_rate_state_grad_is_inert() {
    use ir::deriv::{CompGradMap, DerivEntry};
    let base = representative_model().content_hash();
    let mut m = representative_model();
    let mut g = std::collections::HashMap::new();
    g.insert("S".to_string(), DerivEntry::Grad(Expr::param("beta")));
    m.transitions[0].rate_state_grad = CompGradMap(g);
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty rate_state_grad must NOT change the model hash (gradients are \
         derived, not identity)"
    );
}

/// Model identity is gradient-independent (proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md): `ic_grad` (∂init/∂θ, the
/// forward-sensitivity seed, gh#275) is compiler-derived, so it is NOT hashed —
/// a model with an IC gradient hashes the same as one without.
/// (Previously asserted the opposite; flipped by the proposal.)
#[test]
fn ir_ic_grad_is_inert() {
    use ir::deriv::DerivEntry;
    let base = representative_model().content_hash();
    let mut m = representative_model();
    let mut inner = std::collections::HashMap::new();
    inner.insert("beta".to_string(), DerivEntry::Grad(Expr::param("beta")));
    m.ic_grad.insert("I".to_string(), inner);
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty ic_grad must NOT change the model hash (gradients are derived, \
         not identity)"
    );
}

/// Model identity is gradient-independent (proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md): a likelihood `Diffable`'s
/// `proj_grad` (∂arg/∂projected, the obs FACTOR-2 derivative, gh#275) is
/// compiler-derived, so it is NOT hashed — only the argument `expr` is identity.
/// (Previously asserted the opposite; flipped by the proposal.)
#[test]
fn ir_proj_grad_is_inert() {
    use ir::deriv::DerivEntry;
    use ir::observation::Likelihood;
    let base = representative_model().content_hash();
    let mut m = representative_model();
    match &mut m.observations[0].likelihood {
        Likelihood::Poisson(p) => {
            p.rate.proj_grad = Some(DerivEntry::Grad(Expr::const_(1.0)));
        }
        other => panic!("representative model's first obs must be Poisson, got {other:?}"),
    }
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty proj_grad must NOT change the model hash (gradients are derived, \
         not identity)"
    );
}

/// Model identity is gradient-independent (proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md): `projection_state_grad`
/// (∂projection/∂compartment, the DerivedExpr factor-2 ingredient, gh#275 §1h) is
/// compiler-derived, so it is NOT hashed — a model with it hashes the same as one
/// without. (Previously asserted the opposite; flipped by the proposal.)
#[test]
fn ir_projection_state_grad_is_inert() {
    use ir::deriv::DerivEntry;
    let base = representative_model().content_hash();
    let mut m = representative_model();
    m.observations[0]
        .projection_state_grad
        .0
        .insert("I".to_string(), DerivEntry::Grad(Expr::param("beta")));
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty projection_state_grad must NOT change the model hash (gradients \
         are derived, not identity)"
    );
}

/// Model identity is gradient-independent (proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md): the transition `rate_grad`
/// (∂rate/∂θ) is compiler-derived, so it is NOT hashed. Populating it — even with
/// distinct compartment/param derivatives — must NOT change the model hash. This
/// is the rate-side lean-vs-full guarantee (the flipped form of the removed
/// `rate_grad_map_order_invariant`, which pinned the now-deleted rate_grad sort).
#[test]
fn ir_rate_grad_is_inert() {
    use ir::deriv::DerivEntry;
    let base = representative_model().content_hash();
    let mut m = representative_model();
    let mut rg: HashMap<String, DerivEntry> = HashMap::new();
    rg.insert("beta".to_string(), DerivEntry::Grad(Expr::pop("S")));
    rg.insert("gamma".to_string(), DerivEntry::Grad(Expr::pop("I")));
    m.transitions[0].rate_grad = rg;
    assert_eq!(
        base,
        m.content_hash(),
        "a non-empty rate_grad must NOT change the model hash (gradients are derived, \
         not identity)"
    );
}

/// Model identity is gradient-independent: the overdispersion `sigma_sq_grad`
/// (∂σ²/∂θ) is compiler-derived, so it is NOT hashed. The semantic σ² expression
/// still is (a negative control below). Proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md.
#[test]
fn ir_sigma_sq_grad_is_inert() {
    use ir::deriv::DerivEntry;
    use ir::transition::DrawMethod;
    let mk = |grad: bool| {
        let mut m = representative_model();
        let mut sg: HashMap<String, DerivEntry> = HashMap::new();
        if grad {
            sg.insert("k".to_string(), DerivEntry::Grad(Expr::const_(1.0)));
        }
        m.transitions[0].draw_method = DrawMethod::Overdispersed {
            sigma_sq: Expr::param("sigma_se"),
            sigma_sq_grad: sg,
        };
        m.content_hash()
    };
    assert_eq!(
        mk(true),
        mk(false),
        "sigma_sq_grad must NOT change the model hash (gradients are derived, not identity)"
    );

    // Negative control: the semantic σ² expression IS identity — a different σ²
    // (still gradient-free) must re-key.
    let mut m_other = representative_model();
    m_other.transitions[0].draw_method = DrawMethod::Overdispersed {
        sigma_sq: Expr::param("sigma_other"),
        sigma_sq_grad: HashMap::new(),
    };
    assert_ne!(
        mk(false),
        m_other.content_hash(),
        "the σ² expression itself is identity (only its gradient is stripped)"
    );
}

/// Model identity is gradient-independent: a likelihood `Diffable`'s `grad` map
/// (∂arg/∂θ, the obs autodiff sibling of `proj_grad`) is compiler-derived, so it
/// is NOT hashed — only the argument `expr` is identity. The negative control
/// pins that the argument expression itself still re-keys. Proposal
/// 2026-07-16-gradient-maps-out-of-run-identity.md.
#[test]
fn ir_likelihood_grad_is_inert() {
    use ir::deriv::DerivEntry;
    use ir::observation::Likelihood;
    let base = representative_model().content_hash();
    let mut m = representative_model();
    match &mut m.observations[0].likelihood {
        Likelihood::Poisson(p) => {
            p.rate.grad.insert("rho".to_string(), DerivEntry::Grad(Expr::pop("S")));
        }
        other => panic!("representative model's first obs must be Poisson, got {other:?}"),
    }
    assert_eq!(
        base,
        m.content_hash(),
        "a likelihood Diffable's grad must NOT change the model hash (gradients are \
         derived, not identity)"
    );

    // Negative control: the argument `expr` IS identity.
    let mut m_expr = representative_model();
    match &mut m_expr.observations[0].likelihood {
        Likelihood::Poisson(p) => p.rate.expr = Expr::param("rho_changed"),
        other => panic!("expected Poisson, got {other:?}"),
    }
    assert_ne!(
        base,
        m_expr.content_hash(),
        "the likelihood argument expr is identity (only its gradient is stripped)"
    );
}

/// The `write_str_map` sort (gh#160 non-determinism guard) must still make
/// `HashMap` iteration order irrelevant for the maps that remain in the hash.
/// `compartment_dims` is a still-hashed `HashMap<String, Vec<String>>`
/// (`ModelStructure::hash_into`); building it in two insertion orders must yield
/// the same model hash. (Replaces the removed `rate_grad_map_order_invariant`,
/// whose map is no longer hashed.)
#[test]
fn compartment_dims_map_order_invariant() {
    let build = |order_ab: bool| -> Model {
        let mut m = representative_model();
        let ms = m.model_structure.as_mut().expect("representative model has structure");
        ms.compartment_dims.clear();
        let a = ("S".to_string(), vec!["age".to_string()]);
        let b = ("I".to_string(), vec!["risk".to_string()]);
        if order_ab {
            ms.compartment_dims.insert(a.0.clone(), a.1.clone());
            ms.compartment_dims.insert(b.0.clone(), b.1.clone());
        } else {
            ms.compartment_dims.insert(b.0.clone(), b.1.clone());
            ms.compartment_dims.insert(a.0.clone(), a.1.clone());
        }
        m
    };
    assert_eq!(
        build(true).content_hash(),
        build(false).content_hash(),
        "compartment_dims insertion order must not change the model hash (write_str_map sort)"
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

/// ir/VERSION 0.33: a forcing's `data_source` is provenance, NOT run identity.
/// Symmetric with `ir_quantities_excluded_from_hash` — it pins that the field
/// is genuinely absent from `TimeFunction::hash_into`, which a bare struct
/// field is otherwise invisible about (an absent-vs-present `Option` hashes
/// the same whether or not the field is folded, so only a test can tell).
///
/// Three claims, and the second and third are the ones that would bite:
///
/// - recording provenance at all must not re-key a model that already
///   compiles today;
/// - **the same file read from a second path is the same model.** A copy, or
///   a checkout at a different relative prefix, must reuse the cached fit;
/// - **a byte change that moves no compiled value must not re-key.** A
///   comment line, a trailing newline, CRLF, a reordered column, rows for a
///   stratum this model does not read — all change the file's SHA-256 while
///   leaving the inlined knots identical, i.e. leaving the model identical.
///
/// What still re-keys on a changed file is pinned next door by
/// `ir_forcing_knots_change_hash`.
#[test]
fn ir_forcing_data_source_excluded_from_hash() {
    use ir::time_func::DataSource;
    let with_source = |path: &str, sha: &str| {
        let mut m = representative_model();
        m.time_functions[0].data_source = Some(DataSource {
            path: path.into(),
            sha256: sha.into(),
        });
        m.content_hash()
    };
    let none = representative_model().content_hash();
    let a = with_source("data/forcing.tsv", "a".repeat(64).as_str());
    let b = with_source("copies/forcing.tsv", "a".repeat(64).as_str());
    let c = with_source("data/forcing.tsv", "b".repeat(64).as_str());

    assert_eq!(
        none, a,
        "recording a forcing's data_source must NOT change the model hash \
         (it is compile-time provenance, not identity)"
    );
    assert_eq!(
        a, b,
        "the SAME file bytes read from a DIFFERENT path are the same model — \
         a copy or a different checkout prefix must not re-key the fit"
    );
    assert_eq!(
        a, c,
        "a file byte-change that moves no compiled value must not re-key — \
         the knots are what identity is built from, and they are unchanged"
    );
}

/// The other half of the `data_source` identity argument: a changed forcing
/// FILE still re-keys, because its knots are inlined into `TimeFuncKind` and
/// those *are* hashed. This is why folding the content hash in as well would
/// be redundant — and it is asserted rather than assumed, because the whole
/// case for excluding `data_source` rests on it.
#[test]
fn ir_forcing_knots_change_hash() {
    let knots = |v: f64| {
        let mut m = representative_model();
        m.time_functions[0].kind = TimeFuncKind::Interpolated(ir::time_func::Interpolated {
            times: vec![Expr::const_(0.0), Expr::const_(30.0)],
            values: vec![Expr::const_(1.4), Expr::const_(v)],
            method: ir::time_func::InterpMethod::Linear,
        });
        m.content_hash()
    };
    assert_ne!(
        knots(1.3),
        knots(1.31),
        "a forcing value read out of the data file is run identity — editing \
         the file so a knot moves MUST re-key the fit"
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

// (`rate_grad_map_order_invariant` removed: `rate_grad` is no longer hashed
// (SV = 2; proposal 2026-07-16-gradient-maps-out-of-run-identity.md). The
// `write_str_map` sort it exercised is now pinned by
// `compartment_dims_map_order_invariant` above, over a still-hashed map.)

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

/// gh#616: an anchored horizon is model identity — two models whose horizons
/// anchor differently produce different trajectories from the same data, so they
/// must not share a content address. `None` (the whole pre-gh#616 corpus, and
/// every model after the resolver substitutes) contributes NOTHING, which is
/// what `model_golden_hash` above proves: the pin did not move at the 0.32 bump.
#[test]
fn t_end_anchor_changes_run_id() {
    use ir::anchor::{AnchoredTime, ObsAnchor};
    let with = |a: Option<AnchoredTime>| {
        let mut m = representative_model();
        m.simulation.t_end_anchor = a;
        m.content_hash()
    };
    let none = with(None);
    let bare_last = with(Some(AnchoredTime::bare(ObsAnchor::Last)));
    let bare_first = with(Some(AnchoredTime::bare(ObsAnchor::First)));
    let last_plus_28 = with(Some(AnchoredTime { anchor: ObsAnchor::Last, offset: 28.0 }));
    let last_plus_56 = with(Some(AnchoredTime { anchor: ObsAnchor::Last, offset: 56.0 }));

    assert_eq!(none, representative_model().content_hash(),
        "an unanchored model must keep its pre-gh#616 run-id");
    assert_ne!(none, bare_last, "an anchored horizon must re-key");
    assert_ne!(bare_last, bare_first, "first_obs vs last_obs must hash distinctly");
    assert_ne!(bare_last, last_plus_28, "an offset must re-key");
    assert_ne!(last_plus_28, last_plus_56, "a different offset must re-key");
}

/// A preset's anchored horizon re-keys on the same terms.
#[test]
fn preset_t_end_anchor_changes_run_id() {
    use ir::anchor::{AnchoredTime, ObsAnchor};
    let with = |a: Option<AnchoredTime>| {
        let mut m = representative_model();
        m.presets[0].t_end_anchor = a;
        m.content_hash()
    };
    let none = with(None);
    assert_eq!(none, representative_model().content_hash(),
        "an unanchored preset must keep its pre-gh#616 run-id");
    assert_ne!(none, with(Some(AnchoredTime::bare(ObsAnchor::Last))),
        "an anchored preset horizon must re-key");
    assert_ne!(with(Some(AnchoredTime::bare(ObsAnchor::Last))),
               with(Some(AnchoredTime { anchor: ObsAnchor::Last, offset: 56.0 })),
        "a preset offset must re-key");
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
    // The variant *indices* (0-4) are permanent — the property this pins. The
    // absolute hexes moved once at SV=2 (2026-07-16, gradients out of identity:
    // the per-type `header` folds the SV, so bumping it re-keys every ir hash),
    // a deliberate, version-bumped re-key; the *relative* ordering/injectivity of
    // the variants is unchanged.
    assert_eq!(
        hex(Projection::CumulativeFlow("x".into())),
        "e2d143527737965155c5b9ab7d36dc38ec6af37d2ce4279fcbbc2b84ca9c46ca"
    );
    assert_eq!(
        hex(Projection::CurrentPop("x".into())),
        "f0d6fc200544318ad4724b41335177b069b91ff2765caae38d61a38479b12a02"
    );
    assert_eq!(
        hex(Projection::CurrentPopSum(vec!["x".into()])),
        "bb7ede75805e3a90522a4511f66ff9bfeb106934f57e46639764643986852206"
    );
    assert_eq!(
        hex(Projection::DerivedExpr(Expr::const_(1.0))),
        "e726f60975126a21179c87f7a7bd77853a76fedce0541374fb2a80334347c06c"
    );
    assert_eq!(
        hex(Projection::CumulativeFlowSum(vec!["x".into()])),
        "df05acb4d865d837590a535138e7be26546f019da197fb68d7d92603c020f09a"
    );
}
