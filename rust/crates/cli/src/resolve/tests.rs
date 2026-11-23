//! The model fixture the crate's unit tests share, and the inertness of the
//! gradient maps in `runid`'s model digest, which
//! [`super::model_identity_from_ir`] records.

use runid::{ContentAddressed, ContentHash};
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

/// The `model`-level digest hash under fixed versions, the unit the gradient
/// inertness tests compare. Goes through the real `ModelDigest::from_model`.
fn model_digest_hash(m: &Model) -> ContentHash {
    runid::inputs::ModelDigest::from_model(m, "0.7".into(), runid::inputs::EngineIdentity::of_build("0.3.0")).content_hash()
}

#[test]
fn gradient_maps_are_inert() {
    // Model identity is gradient-independent (proposal
    // 2026-07-16-gradient-maps-out-of-run-identity.md): the compiler-derived
    // gradient maps are not folded into the model hash, so the SAME model
    // compiled full (with gradients, for nuts/ode) and lean (`camdlc
    // --no-state-grad`, for simulate/mh) shares ONE model digest — the re-key-free
    // guarantee that unblocks gh#439 A2. Proven end-to-end through
    // `ModelDigest::from_model`, the identity used for every `model` level.
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
