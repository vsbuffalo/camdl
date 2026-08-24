//! gh#719 / gh#723 — an initial-state density is DECLARED or it does not
//! exist.
//!
//! PGAS used to decide whether a parameter was an initial-value parameter by a
//! rounding-gated finite difference on the *chain's own start*
//! (`detect_ivp_mappings`, `PROBE_STEP`), then attached a Binomial density to
//! whatever compartment that parameter moved — reading the parameter as a
//! probability. Two chains of one fit could disagree about whether the term was
//! there at all, and a count-valued parameter that won the coin flip clamped to
//! `1 - 1e-10` and charged `log(1e-10)` for every individual outside the
//! compartment: a finite ~-4.2e8 offset the non-finite chain-start guard walks
//! straight past.
//!
//! Both the detector and the density it attached are gone. The initial-state
//! term now comes from `init { I ~ poisson(rate = I0) }` through the shared
//! seam (`CompiledModel::initial_state_logpdf`), so a model that declares no
//! law has no term — from ANY start, on EVERY chain.
//!
//! The model below is the one gh#719 reports: `I0 : count`, `init { I = I0 }`,
//! at the ebola national population. The starts include `614.4998`, which is
//! exactly the rounding-boundary crossing the old probe fired on.

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

#[allow(dead_code)]
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

/// A model that declares no `init { }` law carries no initial-state density —
/// from every start, including the one the deleted probe used to fire on.
///
/// The non-vacuity guard comes first: this model's initial state genuinely
/// MOVES with `I0`, and at `614.4998` a 0.01 nudge genuinely crosses a rounding
/// boundary. So a zero density here is "nothing was declared", not "the
/// parameter is inert" and not "the probe happened to miss".
#[test]
fn an_undeclared_initial_state_carries_no_density() {
    let compiled = count_ivp_model();

    // The probe's own step, spelled out rather than imported: `PROBE_STEP` is
    // deleted, and the number is what makes the crossing start meaningful.
    let step = (I0_UPPER - I0_LOWER).min(1.0) * 0.01;
    assert!((step - 0.01).abs() < 1e-12, "expected a 0.01-individual step, got {step}");

    let crossing = params_with_i0(&compiled, 614.4998);
    let (base, _) = compiled.initial_state_mean(&crossing).unwrap();
    let mut nudged = crossing.clone();
    nudged[compiled.param_index["I0"]] += step;
    let (pert, _) = compiled.initial_state_mean(&nudged).unwrap();
    assert_ne!(
        base.counts, pert.counts,
        "this start must still move a rounded initial count, else a zero density \
         below would prove nothing"
    );

    assert!(!compiled.has_init_law, "this fixture declares no `init {{ }}` law");

    for start in [614.998_f64, 614.4998, 3.0, 2999.5] {
        let params = params_with_i0(&compiled, start);
        let (int_s, real_s) = compiled.initial_state_mean(&params).unwrap();
        let lp = compiled
            .initial_state_logpdf(&int_s.counts, &real_s.values, &params)
            .expect("logpdf");
        assert_eq!(
            lp, 0.0,
            "no law is declared, so there is no initial-state density; got {lp} at \
             start {start}"
        );
        let g = compiled
            .initial_state_logpdf_grad(&int_s.counts, &real_s.values, &params)
            .expect("logpdf_grad");
        assert!(
            g.iter().all(|&v| v == 0.0),
            "the density and its gradient move together or not at all; got {g:?} at \
             start {start}"
        );
    }

    // The other half of the old contract, and the sharper half: a
    // `probability`-kinded parameter driving the same initial state used to
    // register from EVERY start, so this model carried the Binomial term
    // whether or not its author asked for one. It no longer does.
    let frac = fraction_ivp_model();
    assert!(!frac.has_init_law, "the fraction twin declares no law either");
    for start in [0.001_f64, 0.00131, 0.0491, 0.02] {
        let params = params_with_frac(&frac, start);
        let (int_s, real_s) = frac.initial_state_mean(&params).unwrap();
        let lp = frac
            .initial_state_logpdf(&int_s.counts, &real_s.values, &params)
            .expect("logpdf");
        assert_eq!(
            lp, 0.0,
            "a probability-kinded parameter no longer attracts an undeclared \
             Binomial; got {lp} at start {start}"
        );
    }
}

/// What the deleted class cost, kept as arithmetic rather than as prose: a
/// count used directly as a Binomial probability clamps to `1 - 1e-10` and
/// becomes an ~-4.2e8 constant offset on the chain's log-posterior — FINITE, so
/// `NonFiniteChainStart` walks straight past it.
///
/// This is why the surface is `binomial(n = ..., p = ...)` with an explicit,
/// kind-checked `p`: the author writes the denominator and the probability, and
/// a `count`-kinded parameter in the `p` position is a compile error (E344)
/// rather than a number this large.
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

#[allow(dead_code)]
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
