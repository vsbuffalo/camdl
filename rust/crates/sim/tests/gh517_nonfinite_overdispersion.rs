//! gh#517: a non-finite overdispersion must not silently switch off the noise
//! model.
//!
//! The RATE is already guarded — `propensity::eval_propensities` checks
//! `!p.is_finite()` per transition, attributes a table-OOB if that was the
//! cause, coerces to 0 under `--allow-degenerate-rates` and hard-errors
//! otherwise. The DRAW METHOD's parameter was not: `chain_binomial::step_one`
//! resolves `sigma_sq` with a bare `eval_resolved`, which has no error channel.
//!
//! The failure is silent and points the wrong way. Both consumers in `rng.rs`
//! treat `sigma_sq <= 0.0` as a legitimate, *counted* "no overdispersion"
//! (`inc_neg_binomial_pois`) — but `NaN <= 0.0` is false, so a NaN slips past
//! that arm, reaches `Gamma::new(NaN, NaN)`, and lands in the uncounted
//! `Err(_) => 1.0` fallback. The run then continues with the noise model
//! switched off and reports a posterior for a model the user never specified.
//! An overdispersion parameter driven to NaN by a sampler is exactly how this
//! happens in a real fit.

use std::collections::HashMap;
use ir::{
    expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, ParamExpr, PopExpr},
    model::{Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule, SimulationConfig},
    parameter::Parameter,
    transition::{StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, SimConfig},
    simulate::Simulate,
    ChainBinomialSim,
};

/// S --> I at rate `beta * S`, drawn with overdispersion `sigma_expr`.
///
/// The overdispersion is an EXPRESSION, not a bare parameter, because a bare
/// parameter is already covered: `eval_propensities`'s gh#81 guard walks
/// `param_index` before any rate eval and refuses a non-finite value by name
/// (`NonFiniteParameter { name: "sigma_sq", .. }`) — a better error than
/// anything downstream could produce. The uncovered path is an expression that
/// *evaluates* to a non-finite value, because `resolved_expr::eval_resolved`
/// returns a bare `f64` and has none of the div-by-zero / pow / domain guards
/// the fallible rate evaluator in `propensity.rs` applies.
///
/// `sigma_sq = 1 / I` is the shape a modeller actually writes — overdispersion
/// inversely proportional to prevalence — and it is non-finite at exactly the
/// moment the model is most fragile: `I = 0`.
fn overdispersed_model(sigma_expr: Expr) -> Model {
    Model {
        ic_grad: Default::default(),
        name: "gh517".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(Expr::Param(ParamExpr { param: "beta".into() })),
                    right: Box::new(Expr::Pop(PopExpr { pop: "S".into() })),
                },
            }),
            metadata: None,
            draw_method: ir::transition::DrawMethod::Overdispersed {
                sigma_sq: sigma_expr,
                sigma_sq_grad: Default::default(),
            },
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![
            Parameter {
                name: "beta".into(),
                value: ir::parameter::ParamValue::Fixed { value: 0.01 },
                param_kind: None,
                param_dim: None,
            },
            // Finite, so the gh#81 non-finite-parameter guard passes it — and
            // zero, so any expression dividing by it is not.
            Parameter {
                name: "phi".into(),
                value: ir::parameter::ParamValue::Fixed { value: 0.0 },
                param_kind: None,
                param_dim: None,
            },
        ],
        initial_conditions: InitialConditions::constants({
            let mut m = HashMap::new();
            m.insert("S".into(), 10000.0);
            m.insert("I".into(), 0.0);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 1.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 1.0,
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
        quantities: vec![],
        contrasts: vec![],
    }
}

/// `1 / phi` — every PARAMETER is finite, but the expression is not when the
/// sampler puts `phi` on zero. This is the whole reachable surface: σ² is
/// restricted to parameters, time and constants (`CompiledModel::new` rejects
/// state-dependent σ²), and a non-finite parameter is already caught by name,
/// so a division that collapses is what is left.
fn one_over_phi() -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr {
        op: BinOp::Div,
        left: Box::new(Expr::Const(ConstExpr { value: 1.0 })),
        right: Box::new(Expr::Param(ParamExpr { param: "phi".into() })),
    }})
}

fn constant(v: f64) -> Expr { Expr::Const(ConstExpr { value: v }) }

fn run(sigma_expr: Expr, seed: u64) -> Result<u64, sim::error::SimError> {
    let compiled = CompiledModel::new(overdispersed_model(sigma_expr)).unwrap();
    let params = compiled.default_params.clone();
    let config = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 1.0, dt: 1.0 });
    let traj = ChainBinomialSim.run(&compiled, &params, seed, &config)?;
    Ok(traj.snapshots.last().unwrap().flows.as_int()[0])
}

/// All three phases live in ONE test on purpose. `allow_degenerate_rates` is a
/// process-global (`AtomicBool`), and `cargo test` runs the tests inside one
/// binary on parallel threads — a separate `#[test]` that flipped the flag
/// would race the strict-mode assertion and turn this file into a flake.
/// Sequential phases in a single test make the flag's scope explicit.
#[test]
fn a_nonfinite_overdispersion_is_refused_by_default_and_coerced_under_the_flag() {
    sim::eval_stats::set_allow_degenerate_rates(false);

    // ── Negative control: a FINITE sigma^2 is untouched by this change. ──
    // Without this, the test below passes just as well against a guard that
    // rejects every draw, which is the failure mode that matters here: the
    // fix must be invisible to every model that was already correct.
    let finite = run(constant(0.5), 42).expect("a finite sigma^2 must still simulate");
    assert!(finite > 0, "control produced no infections; fixture is wrong, not the guard");

    // ── Strict mode (the default): a NaN sigma^2 is an error, not a draw. ──
    let err = run(one_over_phi(), 42)
        .expect_err("a NaN overdispersion must be refused, not silently drawn as Poisson");
    assert!(
        matches!(err, sim::error::SimError::NumericalCollapse { .. }),
        "expected NumericalCollapse, got {err:?}"
    );

    // +inf reaches the same guard — `Gamma::new(0.0, inf)` is the other way
    // the noise model disappears without a word.
    assert!(
        run(one_over_phi(), 7).is_err(),
        "an infinite overdispersion must be refused too"
    );

    // ── Under --allow-degenerate-rates: coerced, and coerced to the value
    // that lands on the DOCUMENTED "no overdispersion" path rather than an
    // invented one. sigma^2 = 0 is what `neg_binomial` already counts via
    // `inc_neg_binomial_pois`, so the run stays legible in the eval stats.
    sim::eval_stats::set_allow_degenerate_rates(true);
    let coerced = run(one_over_phi(), 42).expect("under the flag a NaN sigma^2 must coerce, not error");
    let no_overdispersion = run(constant(0.0), 42).expect("sigma^2 = 0 is a legal no-overdispersion model");
    assert_eq!(
        coerced, no_overdispersion,
        "coerced NaN must be byte-identical to an explicit sigma^2 = 0, from the same seed \
         — otherwise the coercion invented a distribution of its own"
    );

    // Leave the global as we found it, so a later test in this binary (or a
    // future one added to this file) sees the default.
    sim::eval_stats::set_allow_degenerate_rates(false);
}
