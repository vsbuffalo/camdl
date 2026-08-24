//! gh#719 / gh#723 — PGAS decides whether a parameter is an initial-value
//! parameter by a rounding-gated finite difference on the *chain's own start*,
//! so two chains of one fit can disagree, and a count-valued parameter that
//! wins the coin flip is then used directly as a Binomial probability.
//!
//! The detector (`detect_ivp_mappings`) nudges each estimated parameter by
//! `(upper - lower).min(1.0) * PROBE_STEP` and asks whether any non-balance
//! compartment's *rounded* initial count moved. `initial_state_mean`
//! rounds integer compartments to `i64`, so for a parameter whose range is
//! wider than 1.0 the probe is a flat 0.01 in the parameter's own units:
//!
//!   * a FRACTION parameter driving a large population has slope `N0`, so
//!     0.01 moves the count by thousands and the probe always fires;
//!   * a COUNT parameter has slope 1, so the probe fires only for starts
//!     within 0.01 of a half-integer — about 1% of them.
//!
//! Both tests below are on the same model and differ only in the starting
//! value of `I0`, which is what `run_pgas` receives per chain via
//! `chain_starts[chain_id]`.

use ir::{
    expr::{BinOp, ConstExpr, Expr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        obs_loglik::binom_logpmf,
        pgas::{detect_ivp_mappings, PROBE_STEP},
        types::{EstimatedParam, Transform, PROB_FRACTION_EPS},
    },
};

/// The ebola national model's population, so the arithmetic below is the
/// arithmetic gh#719 reports rather than a scaled-down analogue.
const N_POP: f64 = 18_334_302.0;

/// `I0`'s declared bounds in `fit_national_delay_od_lab_direct_sum_8k.toml`.
/// Note the range is 4097.5, far wider than the `.min(1.0)` cap.
const I0_LOWER: f64 = 2.5;
const I0_UPPER: f64 = 4100.0;

/// A two-parameter SIR whose `I0` is a COUNT: `I = I0`, `S = N0 - I0`.
/// Compartment order puts `I` first so the detector reports `I` rather than
/// `S`, matching the downstream model where `S` is the balance compartment.
fn count_ivp_model() -> CompiledModel {
    let infection = Expr::bin_op(
        BinOp::Div,
        Expr::bin_op(
            BinOp::Mul,
            Expr::bin_op(BinOp::Mul, Expr::param("beta"), Expr::pop("S")),
            Expr::pop("I"),
        ),
        Expr::const_(N_POP),
    );

    let model = Model {
        ic_grad: Default::default(),
        name: "gh719_count_ivp".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "I".into(), kind: CompartmentKind::Integer },
            Compartment { name: "S".into(), kind: CompartmentKind::Integer },
            Compartment { name: "R".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "infection".into(),
            stoichiometry: vec![
                StoichiometryEntry("S".into(), -1),
                StoichiometryEntry("I".into(), 1),
            ],
            rate: infection,
            metadata: None,
            draw_method: DrawMethod::Poisson,
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
                value: ir::parameter::ParamValue::Estimated {
                    init: Some(0.4),
                    bounds: Some((0.01, 2.0)),
                    prior: ir::parameter::PriorSpec::Flat,
                    transform: ir::parameter::Transform::Identity,
                },
                param_kind: None,
                param_dim: None,
            },
            Parameter {
                name: "I0".into(),
                value: ir::parameter::ParamValue::Estimated {
                    init: Some(550.0),
                    bounds: Some((I0_LOWER, I0_UPPER)),
                    prior: ir::parameter::PriorSpec::Flat,
                    transform: ir::parameter::Transform::Identity,
                },
                // As the downstream model declares it: `I0 : count`.
                param_kind: Some(ir::parameter::ParamKind::Count),
                param_dim: None,
            },
        ],
        initial_conditions: InitialConditions::exprs([
            ("I".to_string(), Expr::param("I0")),
            (
                "S".to_string(),
                Expr::bin_op(BinOp::Sub, Expr::const_(N_POP), Expr::param("I0")),
            ),
            ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
        ]),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 40.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 40.0,
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
    };
    CompiledModel::new(model).unwrap()
}

fn i0_spec(compiled: &CompiledModel) -> Vec<EstimatedParam> {
    let idx = compiled.param_index["I0"];
    vec![EstimatedParam {
        name: "I0".into(),
        index: idx,
        initial: 550.0,
        rw_sd: 0.02,
        transform: Transform::None,
        lower: I0_LOWER,
        upper: I0_UPPER,
        rw_sd_auto: false,
        // NOT declared `perturb_only_at_t0` — the downstream fit configs
        // carry no such key for `I0`. The detector enters the IVP path
        // regardless.
        perturb_only_at_t0: false,
    }]
}

fn params_with_i0(compiled: &CompiledModel, i0: f64) -> Vec<f64> {
    let mut p = compiled.default_params.clone();
    p[compiled.param_index["I0"]] = i0;
    p
}

/// The fix: a `count`-kinded parameter never enters the IVP path, from ANY
/// start — including the one that used to fire.
///
/// Before the kind guard, this was decided per chain by a rounding-gated probe.
/// `run_pgas` calls `detect_ivp_mappings` once per chain from
/// `chain_starts[chain_id]`, nudging each parameter by
/// `(upper - lower).min(1.0) * PROBE_STEP` and asking whether any compartment's
/// ROUNDED initial count moved. For `I0` with a range wider than 1.0 that step
/// is a flat 0.01 individuals, so with `I = I0` it fired only for starts within
/// 0.01 of a half-integer — about 1% of them. Two chains of one fit therefore
/// carried different targets: one with a ~4.2e8 offset, one with no IVP term.
///
/// The guard removes the whole class rather than the nondeterminism alone: the
/// Binomial term reads the parameter as a probability, and a count is not one.
#[test]
fn a_count_parameter_never_enters_the_ivp_path() {
    let compiled = count_ivp_model();
    let specs = i0_spec(&compiled);

    let step = (I0_UPPER - I0_LOWER).min(1.0) * PROBE_STEP;
    assert!(
        (step - 0.01).abs() < 1e-12,
        "probe step should be 0.01 individuals for a count parameter, got {step}"
    );

    // `614.4998` is the start that DOES cross a rounding boundary:
    // round(614.4998) = 614, round(614.5098) = 615. It is the case that used to
    // register, so it is the one that proves the guard rather than the probe.
    let crossing = params_with_i0(&compiled, 614.4998);
    let (base, _) = compiled.initial_state_mean(&crossing).unwrap();
    let mut nudged = crossing.clone();
    nudged[compiled.param_index["I0"]] += step;
    let (pert, _) = compiled.initial_state_mean(&nudged).unwrap();
    assert_ne!(
        base.counts, pert.counts,
        "this start must still move a rounded initial count, else the guard is          not what is being tested — the probe simply missed"
    );

    for start in [614.998_f64, 614.4998, 3.0, 2999.5] {
        let m = detect_ivp_mappings(&compiled, &specs, &params_with_i0(&compiled, start))
            .expect("detection must not error");
        assert!(
            m.is_empty(),
            "a `count` parameter must never register an IVP mapping; it did at              start {start}: {m:?}"
        );
    }

    // Negative control, and the other half of the contract: a `probability`
    // parameter still registers, and now does so from EVERY start, because its
    // slope in the initial count is the population rather than 1.
    let frac = fraction_ivp_model();
    let frac_specs = fraction_spec(&frac);
    for start in [0.001_f64, 0.00131, 0.0491, 0.02] {
        let m = detect_ivp_mappings(&frac, &frac_specs, &params_with_frac(&frac, start))
            .expect("detection must not error");
        assert_eq!(
            m.len(), 1,
            "a probability IVP must register from every start; missed at {start}"
        );
    }
}

/// The consequence: once a COUNT parameter has registered, `complete_data_loglik`
/// uses it directly as a Binomial probability, where the clamp turns it into
/// `1 - 1e-10` and the term becomes an ~4.2e8 constant offset on the chain's
/// log-posterior — finite, so `NonFiniteChainStart` walks straight past it.
#[test]
fn a_count_used_as_a_binomial_probability_is_finite_and_astronomically_wrong() {
    // The value gh#719 reports from the frozen ebola chain.
    let i0 = 614.998_f64;
    let drawn_initial_count = 468_u64; // the CSMC-drawn I(0) on that chain
    let patch_pop = N_POP as u64;

    let frac = i0.clamp(PROB_FRACTION_EPS, 1.0 - PROB_FRACTION_EPS);
    assert_eq!(
        frac,
        1.0 - PROB_FRACTION_EPS,
        "any count above 1 clamps to the upper edge, losing the value entirely"
    );

    let term = binom_logpmf(drawn_initial_count, patch_pop, frac);

    assert!(
        term.is_finite(),
        "the term must be FINITE — that is why the non-finite guard misses it"
    );
    assert!(
        (-4.3e8..-4.1e8).contains(&term),
        "expected the ~-4.2e8 offset gh#719 reports, got {term}"
    );

    // It is dominated by (patch_pop - k) * ln(PROB_FRACTION_EPS): every
    // individual NOT in the seeded compartment scores log(1e-10).
    let predicted = (patch_pop - drawn_initial_count) as f64 * PROB_FRACTION_EPS.ln();
    assert!(
        (term - predicted).abs() / term.abs() < 1e-3,
        "term {term} should be within 0.1% of the clamp-driven {predicted}"
    );

    // Negative control: the same call with a genuine fraction is ordinary.
    let honest = binom_logpmf(drawn_initial_count, patch_pop, 468.0 / N_POP);
    assert!(
        honest.is_finite() && honest > -20.0,
        "a real fraction gives an ordinary density, got {honest}"
    );
}

// ── fraction-parameterised twin, for the negative controls ──────────────────

fn fraction_ivp_model() -> CompiledModel {
    let base = count_ivp_model();
    let mut model = (*base.model).clone();
    model.name = "gh719_fraction_ivp".into();
    model.parameters[1] = Parameter {
        name: "I0".into(),
        value: ir::parameter::ParamValue::Estimated {
            init: Some(0.00003),
            bounds: Some((0.0005, 0.05)),
            prior: ir::parameter::PriorSpec::Flat,
            transform: ir::parameter::Transform::Identity,
        },
        param_kind: Some(ir::parameter::ParamKind::Probability),
        param_dim: None,
    };
    model.initial_conditions = InitialConditions::exprs([
        (
            "I".to_string(),
            Expr::bin_op(BinOp::Mul, Expr::param("I0"), Expr::const_(N_POP)),
        ),
        (
            "S".to_string(),
            Expr::bin_op(
                BinOp::Sub,
                Expr::const_(N_POP),
                Expr::bin_op(BinOp::Mul, Expr::param("I0"), Expr::const_(N_POP)),
            ),
        ),
        ("R".to_string(), Expr::Const(ConstExpr { value: 0.0 })),
    ]);
    CompiledModel::new(model).unwrap()
}

fn fraction_spec(compiled: &CompiledModel) -> Vec<EstimatedParam> {
    let idx = compiled.param_index["I0"];
    vec![EstimatedParam {
        name: "I0".into(),
        index: idx,
        initial: 0.001,
        rw_sd: 0.02,
        transform: Transform::None,
        lower: 0.0005,
        upper: 0.05,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }]
}

fn params_with_frac(compiled: &CompiledModel, v: f64) -> Vec<f64> {
    let mut p = compiled.default_params.clone();
    p[compiled.param_index["I0"]] = v;
    p
}
