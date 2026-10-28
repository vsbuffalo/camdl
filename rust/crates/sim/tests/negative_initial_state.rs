//! A negative starting count is classified by whether the `init {}` entry that
//! produced it depends on a random initial law.
//!
//! - **θ-only** (`R = 5 - I0` with `I0 = 10`): every particle starts at the
//!   same negative value, so the filter fails at every particle and the θ is
//!   rejected — `NegativeCount{InitialState}`, per-particle recoverable and not
//!   structural, so the PMMH / IF2 evaluator maps the filter's failure to −∞
//!   (`cli::fit::runner::ruled_out_or_surface`) and the fit continues.
//! - **Drawn** (`I ~ poisson(rate = 5)`, `R = 5 - I`): some particles start
//!   negative and others do not. Killing only those would condition the
//!   posterior on "the draw was feasible", which the model never declared, so
//!   it is `NegativeCount{InitialStateDrawn}`: structural, never a dead
//!   particle, and the run stops with a message naming the fix.
//!
//! `R` is in no rate, so nothing but the negative-count checks can catch it.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    deriv::Diffable,
    expr::{BinOp, BinOpExpr, BinOpWrap, Expr, ParamExpr, PopExpr},
    model::{
        Compartment, CompartmentKind, InitCountLaw, InitSpec, InitialConditions, OutputConfig,
        OutputSchedule, RegularOutputSchedule, SimulationConfig,
    },
    observation::{BinomialLikelihood, PoissonLikelihood},
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Model,
};
use sim::{
    compiled_model::CompiledModel,
    config::{ChainBinomialConfig, GillespieConfig, SimConfig},
    error::{NegativeCountCause, PFDegenerateKind, SimError},
    inference::{
        particle_filter::bootstrap_filter,
        traits::{ObservationModel, SMCConfig},
        ChainBinomialProcess, ParticleState,
    },
    simulate::Simulate,
    ChainBinomialSim, GillespieSim,
};

fn param(name: &str) -> Expr {
    Expr::Param(ParamExpr { param: name.into() })
}
fn pop(name: &str) -> Expr {
    Expr::Pop(PopExpr { pop: name.into() })
}
fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
    Expr::BinOp(BinOpWrap { bin_op: BinOpExpr { op, left: Box::new(l), right: Box::new(r) } })
}
fn fixed(name: &str, value: f64) -> Parameter {
    Parameter { name: name.into(), value: ParamValue::Fixed { value }, param_kind: None, param_dim: None }
}
fn transition(name: &str, stoich: &[(&str, i64)], rate: Expr) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: stoich.iter().map(|(c, d)| StoichiometryEntry((*c).into(), *d)).collect(),
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        lineage: None,
    }
}

/// An SI(R) model whose `init {}` is `S = 50`, `I = <init_i>`, `R = 5 - I`,
/// plus `D = <init_d>` for a compartment in no transition. `R` and `D` are
/// read by no rate.
fn model(init_i: InitSpec, init_d: f64) -> Arc<CompiledModel> {
    model_with_rec_rate(init_i, init_d, bin(BinOp::Mul, param("gamma"), pop("I")))
}

/// [`model`] with the `rec` rate supplied, so a test can make a rate read `R`.
fn model_with_rec_rate(init_i: InitSpec, init_d: f64, rec_rate: Expr) -> Arc<CompiledModel> {
    model_full(init_i, init_d, rec_rate, &[("I", -1)])
}

/// [`model`] with the `rec` rate and stoichiometry supplied.
fn model_full(
    init_i: InitSpec,
    init_d: f64,
    rec_rate: Expr,
    rec_stoich: &[(&str, i64)],
) -> Arc<CompiledModel> {
    let mut ic = InitialConditions::constants(
        [("S", 50.0), ("I", 0.0), ("R", 0.0), ("D", init_d)]
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect::<HashMap<String, f64>>(),
    );
    *ic.0.get_mut("I").unwrap() = init_i;
    *ic.0.get_mut("R").unwrap() =
        InitSpec::Deterministic(bin(BinOp::Sub, Expr::const_(5.0), pop("I")));
    let model = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "negative_initial_state".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: ["S", "I", "R", "D"]
            .iter()
            .map(|c| Compartment { name: (*c).into(), kind: CompartmentKind::Integer })
            .collect(),
        transitions: vec![
            transition(
                "inf",
                &[("S", -1), ("I", 1)],
                bin(BinOp::Mul, bin(BinOp::Mul, param("beta"), pop("S")), pop("I")),
            ),
            transition("rec", rec_stoich, rec_rate),
        ],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![fixed("beta", 0.01), fixed("gamma", 0.5)],
        initial_conditions: ic,
        output: OutputConfig {
            times: OutputSchedule::Regular(RegularOutputSchedule { start: 0.0, step: 1.0 }),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 3.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(7),
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
    Arc::new(CompiledModel::new(model).expect("compiles"))
}

/// `I = 10`: `R = 5 - 10 = -5` at every θ-free start.
fn deterministic() -> Arc<CompiledModel> {
    model(InitSpec::Deterministic(Expr::const_(10.0)), 0.0)
}

/// `I ~ poisson(rate = 5)`: `R = 5 - I` is negative whenever `I > 5`,
/// probability ≈ 0.38 per draw.
fn drawn(init_d: f64) -> Arc<CompiledModel> {
    model(
        InitSpec::Count(InitCountLaw::Poisson(PoissonLikelihood {
            rate: Diffable::new(Expr::const_(5.0)),
        })),
        init_d,
    )
}

fn forward(m: &CompiledModel, seed: u64) -> [(&'static str, Result<(), SimError>); 2] {
    let cb = SimConfig::ChainBinomial(ChainBinomialConfig { t_start: 0.0, t_end: 3.0, dt: 1.0 });
    let gi = SimConfig::Gillespie(GillespieConfig { t_start: 0.0, t_end: 3.0, output_dt: None });
    [
        ("chain_binomial", ChainBinomialSim.run(m, &m.default_params, seed, &cb).map(|_| ())),
        ("gillespie", GillespieSim.run(m, &m.default_params, seed, &gi).map(|_| ())),
    ]
}

/// A state-independent observation model: two observations, so the filter
/// propagates and every particle is checked.
struct FlatObs;
impl ObservationModel<ParticleState> for FlatObs {
    fn log_likelihood(&self, _s: &ParticleState, _i: usize, _p: &[f64]) -> f64 { 0.0 }
    fn n_observations(&self) -> usize { 2 }
    fn obs_time(&self, i: usize) -> f64 { (i + 1) as f64 }
    fn n_streams(&self) -> usize { 0 }
}

fn filter(m: &Arc<CompiledModel>) -> Result<f64, SimError> {
    let process = ChainBinomialProcess::new(Arc::clone(m));
    let config = SMCConfig {
        n_particles: 200,
        dt: 1.0,
        t_start: 0.0,
        skip_first_obs_from_loglik: false,
        record_ancestry: false,
        record_prequential: false,
        record_predictions: false,
        max_substeps: sim::inference::degeneracy::ITER_BUDGET,
    };
    bootstrap_filter(&process, &FlatObs, &m.default_params, &config, 11).map(|r| r.log_likelihood)
}

fn local(m: &CompiledModel, name: &str) -> usize {
    m.global_to_int[m.comp_index[name]].expect("integer compartment")
}

#[test]
fn init_dependence_on_a_law_is_read_off_the_init_block() {
    let det = deterministic();
    assert!(det.init_depends_on_law.iter().all(|&d| !d), "no law, nothing drawn");

    let law = drawn(0.0);
    assert!(law.init_depends_on_law[local(&law, "I")], "I is drawn");
    assert!(law.init_depends_on_law[local(&law, "R")], "R = 5 - I reads the draw");
    assert!(!law.init_depends_on_law[local(&law, "S")], "S = 50 is a constant");
    assert!(!law.init_depends_on_law[local(&law, "D")], "D is a constant");
}

/// Case (a): a θ-only negative start rejects the θ and nothing more.
#[test]
fn a_parameter_only_negative_start_rejects_the_parameters() {
    let m = deterministic();
    for (label, res) in forward(&m, 1) {
        match res {
            Err(ref e @ SimError::NegativeCount {
                cause: NegativeCountCause::InitialState, ref compartment, attempted_value, ..
            }) => {
                assert_eq!(compartment, "R", "{label}");
                assert_eq!(attempted_value, -5, "{label}: reports the starting value");
                assert!(e.is_per_particle_recoverable() && !e.is_structural(), "{label}");
                let msg = e.to_string();
                assert!(msg.contains("'R' starts at -5"), "{label}: {msg}");
                assert!(msg.contains("function of the parameters alone"), "{label}: {msg}");
            }
            other => panic!("{label}: expected InitialState on R, got {other:?}"),
        }
    }

    // Every particle dies, the filter reports the θ as degenerate, and that is
    // not structural: the PMMH / IF2 evaluator maps it to −∞ and continues.
    match filter(&m) {
        Err(ref e @ SimError::PFDegenerate { kind: PFDegenerateKind::AllParticlesDead, .. }) => {
            assert!(!e.is_structural(), "a θ-only negative start must reject θ, not stop the fit");
        }
        other => panic!("expected PFDegenerate(AllParticlesDead), got {other:?}"),
    }
}

/// Case (b): a negative start drawn from a random initial law is a structural
/// error on both forward backends and under the particle filter.
#[test]
fn a_drawn_negative_start_is_a_structural_error() {
    let m = drawn(0.0);
    let (mut failed, mut ran) = (0, 0);
    for seed in 0..40 {
        for (label, res) in forward(&m, seed) {
            match res {
                Ok(()) => ran += 1,
                Err(ref e @ SimError::NegativeCount {
                    cause: NegativeCountCause::InitialStateDrawn, ref compartment, attempted_value, ..
                }) => {
                    failed += 1;
                    assert_eq!(compartment, "R", "{label} seed {seed}");
                    assert!(attempted_value < 0, "{label} seed {seed}");
                    assert!(e.is_structural() && !e.is_per_particle_recoverable(), "{label}");
                    let msg = e.to_string();
                    assert!(msg.contains(&format!("'R' starts at {attempted_value}")), "{msg}");
                    assert!(msg.contains("random initial law"), "{msg}");
                    assert!(msg.contains("which the model does not define"), "{msg}");
                    assert!(msg.contains("Fix: constrain the init expression or the law"), "{msg}");
                }
                other => panic!("{label} seed {seed}: expected Ok or InitialStateDrawn, got {other:?}"),
            }
        }
    }
    // P(I > 5) ≈ 0.38 per draw: both outcomes must occur, or the loop
    // asserted nothing about one of them.
    assert!(failed > 0 && ran > 0, "failed {failed}, ran {ran}");

    match filter(&m) {
        Err(ref e @ SimError::NegativeCount { cause: NegativeCountCause::InitialStateDrawn, .. }) => {
            assert!(e.is_structural(), "must stop the fit, not reject θ");
        }
        other => panic!("expected NegativeCount(InitialStateDrawn) from the filter, got {other:?}"),
    }
}

/// The classification is per compartment: in a model that declares a law, a
/// compartment whose entry does not read it (`D = -2`) is still θ-only, and
/// still rejects θ rather than stopping the fit. The law here is
/// `I ~ binomial(n = 5, p = 0.5)`, so `R = 5 - I` never goes negative and `D`
/// is the only negative start. A draw of `I = 0` leaves no transition able to
/// fire, so Gillespie must check the start itself, not only after a firing.
#[test]
fn a_parameter_only_negative_in_a_law_bearing_model_still_rejects_the_parameters() {
    let m = model(
        InitSpec::Count(InitCountLaw::Binomial(BinomialLikelihood {
            n: Expr::const_(5.0),
            p: Diffable::new(Expr::const_(0.5)),
        })),
        -2.0,
    );
    assert!(m.has_init_law);
    let mut caught = HashMap::new();
    for seed in 0..20 {
        for (label, res) in forward(&m, seed) {
            match res {
                Err(ref e @ SimError::NegativeCount {
                    cause: NegativeCountCause::InitialState, ref compartment, ..
                }) => {
                    assert_eq!(compartment, "D", "{label}");
                    assert!(!e.is_structural(), "{label}");
                    *caught.entry(label).or_insert(0) += 1;
                }
                other => panic!("{label} seed {seed}: expected InitialState on D, got {other:?}"),
            }
        }
    }
    assert_eq!(caught.get("chain_binomial"), Some(&20), "{caught:?}");
    assert_eq!(caught.get("gillespie"), Some(&20), "{caught:?}");
    match filter(&m) {
        Err(ref e @ SimError::PFDegenerate { kind: PFDegenerateKind::AllParticlesDead, .. }) => {
            assert!(!e.is_structural());
        }
        other => panic!("expected PFDegenerate(AllParticlesDead), got {other:?}"),
    }
}

/// When a rate reads the compartment a draw made negative (`rec` at
/// `gamma * I * R`), the draw is still reported as `InitialStateDrawn`, not as
/// the negative propensity the rate would produce: the check runs where the
/// initial state is drawn, before any rate is evaluated. A negative
/// propensity is θ-dependent and not structural, so reporting it would turn
/// the modelling error back into a rejected θ.
#[test]
fn a_drawn_negative_start_read_by_a_rate_is_reported_as_drawn() {
    let m = model_with_rec_rate(
        InitSpec::Count(InitCountLaw::Poisson(PoissonLikelihood {
            rate: Diffable::new(Expr::const_(5.0)),
        })),
        0.0,
        bin(BinOp::Mul, bin(BinOp::Mul, param("gamma"), pop("I")), pop("R")),
    );
    let mut failed = 0;
    for seed in 0..40 {
        for (label, res) in forward(&m, seed) {
            match res {
                Ok(()) => {}
                Err(SimError::NegativeCount { cause: NegativeCountCause::InitialStateDrawn, .. }) => {
                    failed += 1;
                }
                other => panic!("{label} seed {seed}: expected Ok or InitialStateDrawn, got {other:?}"),
            }
        }
    }
    assert!(failed > 0, "no seed drew I > 5");
    assert!(matches!(
        filter(&m),
        Err(SimError::NegativeCount { cause: NegativeCountCause::InitialStateDrawn, .. })
    ));
}

/// A negative start is reported even when the first step's inflows would lift
/// it back to zero or above. `I = 6` gives `R = 5 - 6 = -1`, and `rec : I --> R`
/// at `gamma = 5` moves most of `I` into `R` in the first step (chain-binomial)
/// or as the first firing (Gillespie, probability ≈ 0.9). A check that looked
/// only at the state after the transitions would let the run proceed from
/// `R = -1`, and under inference score those parameters as feasible.
#[test]
fn a_negative_start_is_reported_even_when_the_first_step_lifts_it() {
    let m = model_full(
        InitSpec::Deterministic(Expr::const_(6.0)),
        0.0,
        bin(BinOp::Mul, Expr::const_(5.0), pop("I")),
        &[("I", -1), ("R", 1)],
    );
    for seed in 0..20 {
        for (label, res) in forward(&m, seed) {
            match res {
                Err(SimError::NegativeCount {
                    cause: NegativeCountCause::InitialState, ref compartment, attempted_value, ..
                }) => {
                    assert_eq!((compartment.as_str(), attempted_value), ("R", -1), "{label} seed {seed}");
                }
                other => panic!("{label} seed {seed}: expected InitialState on R, got {other:?}"),
            }
        }
    }
    assert!(matches!(
        filter(&m),
        Err(SimError::PFDegenerate { kind: PFDegenerateKind::AllParticlesDead, .. })
    ));
}
