//! gh#284: the LICM per-eval keystone invariant — a `per_eval_binding` body must
//! be loop-invariant within a θ-stable span (parameters / tables / constants
//! only) — is enforced at `CompiledModel::new`, not merely assumed by comments.
//!
//! The staged prologue (`stage_per_eval`) evaluates each body ONCE at `t_start`
//! against a zero scratch and reads it every substep. A body that references
//! compartment state panics index-OOB on that zero scratch (`IntState::new(0)`);
//! one that references time / dt / a forcing is staged once and read STALE every
//! later substep (silent-wrong) — the failure a bare `references_state` guard
//! would miss. The OCaml LICM pass never emits such a body (`licm.ml
//! is_invariant`), but a hand-edited or future-emitted IR could, so the Rust
//! constructor rejects it with a located error. Defense-in-depth counterpart to
//! the OCaml predicate, mirroring the overdispersion σ² and intervention-schedule
//! `references_state` guards in `compiled_model.rs`.

use std::collections::HashMap;

use ir::{
    expr::{BinOp, Expr},
    model::{
        Binding, Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::{ParamValue, Parameter},
    Model,
};
use sim::compiled_model::CompiledModel;

/// Minimal one-compartment model carrying a single per-eval binding `body`.
fn model_with_per_eval(body: Expr) -> Model {
    model_with_per_eval_bindings(vec![Binding { name: "__licm_0".into(), expr: body }])
}

/// Minimal one-compartment model carrying an arbitrary list of per-eval bindings.
fn model_with_per_eval_bindings(per_eval_bindings: Vec<Binding>) -> Model {
    Model {
        name: "per_eval".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment { name: "S".into(), kind: CompartmentKind::Integer }],
        transitions: vec![],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings,
        parameters: vec![Parameter {
            name: "beta".into(),
            value: ParamValue::Fixed { value: 1.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::Parameterized(HashMap::new()),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 1.0,
            time_semantics: "continuous".into(),
            dt: None,
            rng_seed: Some(1),
            integrator: Default::default(),
        },
        presets: vec![],
        model_structure: None,
        balance: None,
        identity_tracked_compartments: vec![], quantities: vec![],
    }
}

fn assert_per_eval_rejection(body: Expr, what: &str) {
    let err = CompiledModel::new(model_with_per_eval(body))
        .err()
        .unwrap_or_else(|| panic!("{what}: expected a per-eval invariance rejection, model built"));
    let msg = format!("{err}");
    assert!(
        msg.contains("per-eval binding") && msg.contains("__licm_0"),
        "{what}: error should name the offending per-eval binding; got: {msg}"
    );
}

#[test]
fn per_eval_body_referencing_state_is_rejected() {
    // The panic case: `Pop` resolves to `IntPop(local)`, staged against the zero
    // scratch `IntState::new(0)` → index-OOB. Must be a located error instead.
    assert_per_eval_rejection(Expr::pop("S"), "compartment state (Pop)");
}

#[test]
fn per_eval_body_referencing_pop_sum_is_rejected() {
    assert_per_eval_rejection(Expr::pop_sum(vec!["S".into()]), "compartment state (PopSum)");
}

#[test]
fn per_eval_body_referencing_time_is_rejected() {
    // The silent-wrong case a bare `references_state` guard would MISS: `Time`
    // is staged once at `t_start` and read stale every later substep. Caught
    // only because the per-eval guard rejects time-varying nodes too.
    assert_per_eval_rejection(Expr::time(), "simulation time");
}

#[test]
fn per_eval_body_referencing_dt_is_rejected() {
    assert_per_eval_rejection(Expr::dt(), "integrator step (dt)");
}

#[test]
fn per_eval_forward_reference_is_rejected() {
    // A per-eval body must reference only EARLIER slots — `stage_per_eval` lends
    // body `i` only `&scratch[..i]`, so a forward ref reads out of bounds. Here
    // slot 0 references `__licm_1` (slot 1): accepted by the pre-fix constructor,
    // then panics index-OOB at staging. The guard must reject it instead.
    let bindings = vec![
        Binding { name: "__licm_0".into(), expr: Expr::per_eval_ref("__licm_1") },
        Binding { name: "__licm_1".into(), expr: Expr::const_(1.0) },
    ];
    let err = CompiledModel::new(model_with_per_eval_bindings(bindings))
        .err()
        .expect("forward per-eval reference must be rejected, not built");
    let msg = format!("{err}");
    assert!(
        msg.contains("per-eval binding") && msg.contains("__licm_0"),
        "error should name the offending binding; got: {msg}"
    );
}

#[test]
fn per_eval_self_reference_is_rejected() {
    // A self-reference (slot 0 → slot 0) is the cyclic case: the on-demand
    // fallback would infinitely recurse, the staged path reads its own unfilled
    // slot. Reject at construction.
    let bindings = vec![
        Binding { name: "__licm_0".into(), expr: Expr::per_eval_ref("__licm_0") },
    ];
    let err = CompiledModel::new(model_with_per_eval_bindings(bindings))
        .err()
        .expect("self-referential per-eval binding must be rejected, not built");
    assert!(format!("{err}").contains("__licm_0"));
}

#[test]
fn per_eval_backward_reference_builds() {
    // The legitimate topological case: slot 1 references the EARLIER slot 0.
    // This is what the LICM pass emits when one hoisted body reuses another;
    // it must build.
    let bindings = vec![
        Binding { name: "__licm_0".into(), expr: Expr::param("beta") },
        Binding {
            name: "__licm_1".into(),
            expr: Expr::bin_op(BinOp::Mul, Expr::per_eval_ref("__licm_0"), Expr::const_(2.0)),
        },
    ];
    assert!(
        CompiledModel::new(model_with_per_eval_bindings(bindings)).is_ok(),
        "a backward (earlier-slot) per-eval reference must build"
    );
}

#[test]
fn per_eval_param_const_body_builds() {
    // Sanity: a genuinely invariant body (param × const) is accepted — the guard
    // rejects non-invariance, not per-eval bindings per se.
    let body = Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::const_(2.0));
    assert!(
        CompiledModel::new(model_with_per_eval(body)).is_ok(),
        "a param/const-only per-eval body must build"
    );
}
