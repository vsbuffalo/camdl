//! The model fixture the crate's unit tests share, and the inertness of the
//! gradient maps in the model hash, which [`super::model_identity_from_ir`]
//! records.

use ir::model_hash::ContentHash;
use ir::model::{
    InitialConditions, Model, OutputConfig, OutputSchedule, RegularOutputSchedule, SimulationConfig,
};

pub(crate) fn tiny_model() -> Model {
    Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "sir".into(),
        version: "1".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        parameters: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        initial_conditions: InitialConditions::default(),
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 100.0,
            dt: Some(1.0),
            rng_seed: None,
            integrator: Default::default(),
            t_end_anchor: None, t_start_anchor: None,
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
    }
}

/// The model hash, the unit the gradient inertness tests compare.
fn model_digest_hash(m: &Model) -> ContentHash {
    ir::model_hash::model_ir_hash(m)
}

#[test]
fn gradient_maps_are_inert() {
    // `model_ir_hash` leaves the compiler-derived gradient maps out, so the
    // same model compiled full (with gradients) and lean (`camdlc
    // --no-state-grad`) has one model identity string. A stored result keys on
    // the stored compile instead, gradient maps included, so this test says
    // nothing about reuse.
    use ir::deriv::{CompGradMap, DerivEntry};
    use ir::expr::{BinOp, Expr};
    use ir::transition::{DrawMethod, StoichiometryEntry, Transition};

    let tx = |with_grad: bool| -> Transition {
        let mut rate_grad = std::collections::HashMap::new();
        let mut rate_state_grad = std::collections::HashMap::new();
        if with_grad {
            rate_grad.insert("beta".to_string(), DerivEntry::Grad(Expr::pop("I")));
            rate_state_grad.insert("I".to_string(), DerivEntry::Grad(Expr::param("beta")));
        }
        Transition {
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate: Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("I")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad,
            rate_state_grad: CompGradMap(rate_state_grad),
            lineage: None,
        }
    };

    let mut lean = tiny_model();
    lean.transitions = vec![tx(false)];
    let mut full = tiny_model();
    full.transitions = vec![tx(true)];
    assert_eq!(
        model_digest_hash(&lean),
        model_digest_hash(&full),
        "gradient maps must not affect the model digest — lean and full compiles of \
         the same model share one identity (proposal 2026-07-16)"
    );

    // Negative control: the rate EXPRESSION itself is identity — changing it
    // (still gradient-free) must re-key.
    let mut changed = tiny_model();
    let mut t = tx(false);
    t.rate = Expr::param("beta");
    changed.transitions = vec![t];
    assert_ne!(
        model_digest_hash(&lean),
        model_digest_hash(&changed),
        "the rate expression itself is identity (only its gradient is stripped)"
    );
}
