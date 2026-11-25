use std::path::{Path, PathBuf};

use ir::expr::{BinOp, Expr};
use ir::intervention::{
    Action, AddAction, CmpOp, FireSource, Intervention, InterventionSchedule, ObsReducer, ReactiveTrigger,
    TriggerExpr, TriggerQuantity, TriggerThreshold,
};
use ir::model::{BalanceSpec, Binding, InitialConditions};
use ir::transition::{DrawMethod, Transition, TransitionLineage};

use super::*;

fn mul(a: Expr, b: Expr) -> Expr {
    Expr::bin_op(BinOp::Mul, a, b)
}

fn transition(name: &str, rate: Expr) -> Transition {
    Transition {
        name: name.into(),
        stoichiometry: vec![],
        rate,
        metadata: None,
        draw_method: DrawMethod::Poisson,
        rate_grad: Default::default(),
        rate_state_grad: Default::default(),
        lineage: None,
    }
}

/// A model with `init { I = I0 }` and whatever `edit` adds.
fn model(edit: impl FnOnce(&mut ir::Model)) -> ir::Model {
    let mut m = crate::test_support::tiny_model();
    m.initial_conditions = InitialConditions::exprs([("I".to_string(), Expr::param("I0"))]);
    edit(&mut m);
    m
}

fn resolved(m: &ir::Model) -> ParamSurfaces {
    match param_surfaces(m) {
        Classification::Resolved(s) => s,
        Classification::Uncertain { reason, surface } => panic!("uncertain: {reason}, in {surface}"),
    }
}

fn reactive(name: &str, threshold: &str) -> Intervention {
    Intervention {
        name: name.into(),
        base_name: None,
        fire: FireSource::Reactive(ReactiveTrigger {
            when_: TriggerExpr::Not(Box::new(TriggerExpr::Cmp {
                lhs: TriggerQuantity::Observed { stream: "cases".into(), window: None, reducer: ObsReducer::Latest },
                op: CmpOp::Ge,
                rhs: TriggerThreshold::Param(threshold.into()),
            })),
            after: 0.0,
            once: true,
            cooldown: None,
        }),
        actions: vec![],
        kind: Default::default(),
    }
}

fn add_at(name: &str, at: Vec<Expr>, count: Expr) -> Intervention {
    Intervention {
        name: name.into(),
        base_name: None,
        fire: FireSource::Scheduled(InterventionSchedule::AtTimesExpr(at)),
        actions: vec![Action::Add(AddAction { compartment: "I".into(), count, count_grad: Default::default() })],
        kind: Default::default(),
    }
}

// ─── One test per surface, on hand-built IR ─────────────────────────────────

#[test]
fn an_init_expression_is_the_initial_state() {
    let s = resolved(&model(|_| {}));
    assert!(s.initial_state_only("I0"), "{s:?}");
    assert_eq!(s.initial_state["I0"], Surface::InitialCondition { compartment: "I".into() });
}

#[test]
fn a_transition_rate_is_the_dynamics() {
    let s = resolved(&model(|m| m.transitions.push(transition("infection", mul(Expr::param("beta"), Expr::pop("S"))))));
    assert_eq!(s.after_t0_use("beta"), Some(&Surface::TransitionRate { transition: "infection".into() }));
    assert!(s.initial_state_only("I0"), "a compartment the rate reads does not carry I0 into the dynamics");
}

#[test]
fn an_overdispersion_variance_is_the_dynamics() {
    let s = resolved(&model(|m| {
        let mut t = transition("infection", Expr::const_(1.0));
        t.draw_method = DrawMethod::Overdispersed { sigma_sq: Expr::param("sigma_se"), sigma_sq_grad: Default::default() };
        m.transitions.push(t);
    }));
    assert_eq!(s.dynamics["sigma_se"], Surface::Overdispersion { transition: "infection".into() });
}

#[test]
fn a_lineage_weight_is_the_dynamics() {
    let s = resolved(&model(|m| {
        let mut t = transition("spillover", Expr::const_(1.0));
        t.lineage = Some(TransitionLineage { is_lineage_event: true, parent_pool_weights: vec![("I".into(), Expr::param("w"))] });
        m.transitions.push(t);
    }));
    assert_eq!(s.dynamics["w"], Surface::LineageWeight { transition: "spillover".into() });
}

#[test]
fn an_ode_equation_is_the_dynamics() {
    let s = resolved(&model(|m| m.ode_equations.push(ir::ode_equation::OdeEquation {
        compartment: "W".into(),
        derivative: mul(Expr::param("decay"), Expr::pop("W")),
    })));
    assert_eq!(s.dynamics["decay"], Surface::OdeEquation { compartment: "W".into() });
}

#[test]
fn an_intervention_action_and_its_times_are_the_dynamics() {
    let s = resolved(&model(|m| m.interventions.push(add_at("seed", vec![Expr::param("t_seed")], Expr::param("n_seed")))));
    assert_eq!(s.dynamics["t_seed"], Surface::InterventionTimes { intervention: "seed".into() });
    assert_eq!(s.dynamics["n_seed"], Surface::InterventionAction { intervention: "seed".into() });
}

#[test]
fn a_reactive_trigger_threshold_is_the_dynamics() {
    let s = resolved(&model(|m| m.interventions.push(reactive("sia", "threshold"))));
    assert_eq!(s.dynamics["threshold"], Surface::TriggerThreshold { intervention: "sia".into() });
}

#[test]
fn the_balance_expression_is_the_dynamics() {
    let s = resolved(&model(|m| m.balance = Some(BalanceSpec { target: "R".into(), expr: Expr::param("pop") })));
    assert_eq!(s.dynamics["pop"], Surface::Balance { target: "R".into() });
}

#[test]
fn a_likelihood_and_a_derived_projection_are_the_observations() {
    use ir::observation::{Likelihood, NegBinomialLikelihood, ObservationModel, Projection};
    use ir::deriv::Diffable;
    let s = resolved(&model(|m| m.observations.push(ObservationModel {
        name: "cases".into(),
        source: "cases".into(),
        columns: vec![],
        scored: "cases".into(),
        emit_schedule: None,
        stratum: vec![],
        covers: None,
        projection: Projection::DerivedExpr(mul(Expr::param("scale"), Expr::pop("I"))),
        projection_state_grad: Default::default(),
        likelihood: Likelihood::NegBinomial(NegBinomialLikelihood {
            mean: Diffable::new(mul(Expr::param("rho"), Expr::param("I0"))),
            dispersion: Diffable::new(Expr::param("k")),
        }),
    })));
    assert_eq!(s.observations["scale"], Surface::ObservationProjection { stream: "cases".into() });
    assert_eq!(s.observations["rho"], Surface::ObservationLikelihood { stream: "cases".into() });
    assert_eq!(s.observations["k"], Surface::ObservationLikelihood { stream: "cases".into() });
    assert!(!s.initial_state_only("I0"), "I0 in a likelihood is read after t = 0");
    assert_eq!(s.after_t0_use("I0"), Some(&Surface::ObservationLikelihood { stream: "cases".into() }));
}

// ─── Indirect reach ─────────────────────────────────────────────────────────

#[test]
fn a_binding_of_a_binding_carries_its_parameter() {
    let s = resolved(&model(|m| {
        m.bindings = vec![
            Binding { name: "a".into(), expr: Expr::param("beta") },
            Binding { name: "b".into(), expr: mul(Expr::binding_ref("a"), Expr::const_(2.0)) },
        ];
        m.per_eval_bindings = vec![Binding { name: "pe".into(), expr: Expr::param("gamma") }];
        m.transitions.push(transition("infection", Expr::binding_ref("b")));
        m.transitions.push(transition("recovery", Expr::per_eval_ref("pe")));
    }));
    assert_eq!(s.dynamics["beta"], Surface::TransitionRate { transition: "infection".into() });
    assert_eq!(s.dynamics["gamma"], Surface::TransitionRate { transition: "recovery".into() });
}

#[test]
fn a_binding_read_only_by_init_is_the_initial_state() {
    let s = resolved(&model(|m| {
        m.bindings = vec![Binding { name: "n0".into(), expr: Expr::param("N0") }];
        m.initial_conditions = InitialConditions::exprs([("S".to_string(), Expr::binding_ref("n0"))]);
    }));
    assert!(s.initial_state_only("N0"), "{s:?}");
}

#[test]
fn a_time_function_coefficient_and_its_lag_reach_the_rate() {
    use ir::time_func::{Sinusoidal, TimeFuncKind, TimeFunction};
    let s = resolved(&model(|m| {
        m.time_functions.push(TimeFunction {
            name: "season".into(),
            kind: TimeFuncKind::Sinusoidal(Sinusoidal {
                amplitude: Expr::param("amp"),
                period: Expr::const_(365.0),
                phase: Expr::const_(0.0),
                baseline: Expr::const_(1.0),
            }),
            dim: (0, 0),
            lag: Some(Expr::param("lag")),
            data_source: None,
        });
        m.transitions.push(transition("infection", Expr::TimeFunc(ir::expr::TimeFuncWrap {
            time_func: ir::expr::TimeFuncRef { name: "season".into() },
        })));
    }));
    assert_eq!(s.dynamics["amp"], Surface::TransitionRate { transition: "infection".into() });
    assert_eq!(s.dynamics["lag"], Surface::TransitionRate { transition: "infection".into() });
}

#[test]
fn a_table_cell_and_its_index_reach_the_rate() {
    use ir::table::{OobPolicy, Table, TableSource};
    let s = resolved(&model(|m| {
        m.tables.push(Table {
            name: "stage".into(),
            source: TableSource::Inline { values: vec![Expr::param("sigma_1"), Expr::param("sigma_2")] },
            out_of_bounds: OobPolicy::Error,
            cell_kind: None,
        });
        m.transitions.push(transition("progression", Expr::TableLookup(ir::expr::TableLookupWrap {
            table_lookup: ir::expr::TableLookupExpr { table: "stage".into(), indices: vec![Expr::param("which")] },
        })));
    }));
    for p in ["sigma_1", "sigma_2", "which"] {
        assert_eq!(s.dynamics[p], Surface::TransitionRate { transition: "progression".into() }, "{p}");
    }
}

#[test]
fn a_binding_cycle_terminates_with_every_member_s_parameters() {
    let s = resolved(&model(|m| {
        m.bindings = vec![
            Binding { name: "a".into(), expr: mul(Expr::param("p"), Expr::binding_ref("b")) },
            Binding { name: "b".into(), expr: mul(Expr::param("q"), Expr::binding_ref("a")) },
        ];
        m.transitions.push(transition("t1", Expr::binding_ref("b")));
        m.transitions.push(transition("t2", Expr::binding_ref("a")));
    }));
    assert_eq!(s.dynamics["p"], Surface::TransitionRate { transition: "t1".into() });
    assert_eq!(s.dynamics["q"], Surface::TransitionRate { transition: "t1".into() });
}

#[test]
fn an_unresolved_reference_is_uncertain() {
    let m = model(|m| m.transitions.push(transition("infection", Expr::binding_ref("missing"))));
    match param_surfaces(&m) {
        Classification::Uncertain { reason, surface } => {
            assert!(reason.contains("`missing`"), "{reason}");
            assert_eq!(surface, Surface::TransitionRate { transition: "infection".into() });
        }
        other => panic!("expected Uncertain, got {other:?}"),
    }
}

#[test]
fn a_hierarchical_prior_argument_is_no_surface() {
    use ir::parameter::{HierarchicalKind, HierarchicalPrior, ParamValue, Parameter, PriorSpec};
    let s = resolved(&model(|m| {
        m.parameters.push(Parameter {
            name: "gamma".into(),
            value: ParamValue::Estimated {
                init: None,
                bounds: None,
                prior: PriorSpec::Hierarchical(HierarchicalPrior {
                    kind: HierarchicalKind::LogNormal,
                    args: [("mu".to_string(), Expr::param("I0")), ("sigma".to_string(), Expr::const_(0.1))].into(),
                    pool_over: String::new(),
                }),
            },
            param_kind: None,
            param_dim: None,
        });
    }));
    assert!(s.initial_state_only("I0"), "{s:?}");
}

// ─── Compiled models ────────────────────────────────────────────────────────

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn compiled(path: &Path) -> ir::Model {
    let camdlc = repo().join("ocaml/_build/default/bin/camdlc.exe");
    assert!(camdlc.exists(), "camdlc missing at {}: run `make build-ocaml`", camdlc.display());
    let out = std::process::Command::new(&camdlc).arg(path).output().unwrap();
    assert!(out.status.success(), "camdlc {}: {}", path.display(), String::from_utf8_lossy(&out.stderr));
    let dir = tempfile::tempdir().unwrap();
    let ir_path = dir.path().join("model.ir.json");
    std::fs::write(&ir_path, &out.stdout).unwrap();
    crate::util::load_ir(ir_path.to_str().unwrap()).unwrap().0
}

fn compiled_source(src: &str) -> ir::Model {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.camdl");
    std::fs::write(&path, src).unwrap();
    compiled(&path)
}

#[test]
fn sir_i0_and_n0_are_initial_state_only_and_the_rates_are_not() {
    let m = compiled(&repo().join("tests/recovery/cases/sir/model.camdl"));
    let s = resolved(&m);
    for p in ["I0", "N0"] {
        assert!(s.initial_state_only(p), "{p}: {s:?}");
    }
    assert_eq!(s.dynamics["beta"], Surface::TransitionRate { transition: "infection".into() });
    assert_eq!(s.dynamics["gamma"], Surface::TransitionRate { transition: "recovery".into() });
    for p in ["rho", "k"] {
        assert!(matches!(s.observations[p], Surface::ObservationLikelihood { .. }), "{p}: {s:?}");
    }
}

#[test]
fn init_law_arguments_are_initial_state_only() {
    let m = compiled(&repo().join("ocaml/golden/init_laws.camdl"));
    let s = resolved(&m);
    for p in ["I0", "N0", "exposed_per_infectious", "exposed_k", "frac_immune", "W0", "W0_sd"] {
        assert!(s.initial_state_only(p), "{p}: {s:?}");
    }
    for p in ["beta", "sigma", "gamma", "shed", "decay"] {
        assert!(s.after_t0_use(p).is_some(), "{p}: {s:?}");
    }
}

#[test]
fn a_parametric_fire_time_is_the_dynamics() {
    let m = compiled_source(r#"
time_unit = 'days
compartments { S, I, R }
parameters {
  beta   : rate in [0.05, 1.0]
  t_seed : real in [1, 50]
  I0     : count in [1, 100]
}
transitions {
  infection : S --> I @ beta * S * I / (S + I + R)
}
interventions {
  vacc : transfer(fraction = 0.5, from = S, to = R) at [t_seed]
}
init { S = 1000 - I0  I = I0 }
simulate { from = 0 'days  to = 60 'days }
"#);
    let s = resolved(&m);
    assert_eq!(s.dynamics["t_seed"], Surface::InterventionTimes { intervention: "vacc".into() });
    assert!(s.initial_state_only("I0"));
}

#[test]
fn a_parameter_trigger_threshold_is_the_dynamics() {
    let m = compiled(&repo().join("tests/fixtures/reactive/reactive_sir_observed_threshold.camdl"));
    let s = resolved(&m);
    assert_eq!(s.dynamics["trigger_threshold"], Surface::TriggerThreshold { intervention: "sia".into() });
    assert!(s.initial_state_only("I0") && s.initial_state_only("N0"), "{s:?}");
}

#[test]
fn forcing_and_table_coefficients_are_the_dynamics() {
    let forcing = crate::util::load_ir(repo().join("ocaml/golden/periodic_spline_param_coefs.ir.json").to_str().unwrap())
        .unwrap().0;
    let s = resolved(&forcing);
    for p in ["s1", "s6"] {
        assert!(matches!(s.dynamics[p], Surface::TransitionRate { .. }), "{p}: {s:?}");
    }
    let table = crate::util::load_ir(repo().join("ocaml/golden/seir_erlang_staged.ir.json").to_str().unwrap())
        .unwrap().0;
    let s = resolved(&table);
    for p in ["sigma_e1", "sigma_e3"] {
        assert!(matches!(s.dynamics[p], Surface::TransitionRate { .. }), "{p}: {s:?}");
    }
}

// ─── IF2's timing ───────────────────────────────────────────────────────────

fn if2_steps(at: PerturbAt) -> ResolvedSteps {
    use crate::fit::steps::{ResolvedPerturbation, StepFrom};
    use sim::inference::types::{Perturbation, ZSd};
    let step = If2Step::Single(Perturbation { sd: ZSd::new(0.05).unwrap(), at });
    ResolvedSteps::If2 {
        params: [("I0".to_string(), ResolvedPerturbation { step, from: StepFrom::Default })].into_iter().collect(),
        simplex: vec![],
    }
}

fn if2() -> Algorithm {
    toml::from_str("algorithm = \"if2\"\nbackend = \"chain_binomial\"\nchains = 1\nparticles = 10\niterations = 1\ncooling = 0.5")
        .unwrap()
}

fn timing_check(m: &ir::Model, at: PerturbAt) -> Result<Vec<String>, String> {
    let c = param_surfaces(m);
    let classes = [("I0".to_string(), UseClass::Likelihood)].into_iter().collect();
    refuse_for_method(&if2(), &if2_steps(at), &c, &classes, &[], m)
}

#[test]
fn an_uncertain_classification_never_licenses_initial_state() {
    let m = model(|m| m.transitions.push(transition("infection", Expr::binding_ref("missing"))));
    let err = timing_check(&m, PerturbAt::TimeZeroOnly).unwrap_err();
    assert!(err.contains("`I0` has `initial_state = true`") && err.contains("binding `missing`")
        && err.contains("the rate of transition `infection`"), "{err}");
    assert_eq!(timing_check(&m, PerturbAt::EveryObservation), Ok(vec![]),
        "an uncertain classification does not ask for initial_state");
}

#[test]
fn a_resolved_classification_checks_both_ways() {
    let m = model(|_| {});
    assert_eq!(timing_check(&m, PerturbAt::TimeZeroOnly), Ok(vec![]));
    let err = timing_check(&m, PerturbAt::EveryObservation).unwrap_err();
    assert!(err.contains("`I0` enters only the initial state (the initial condition of `I`)"), "{err}");
}

// ─── Use classes ────────────────────────────────────────────────────────────

/// An SIR with `extra_params` declared, `gamma` among them, and the
/// parameters of `estimated` estimated under their model priors.
fn classes_of(extra_params: &str, estimated: &[&str]) -> IndexMap<String, UseClass> {
    let m = compiled_source(&format!(r#"
time_unit = 'days
compartments {{ S, I, R }}
parameters {{
  beta  : rate in [0.05, 1.0]
{extra_params}
}}
transitions {{
  infection : S --> I @ beta * S * I / (S + I + R)
  recovery  : I --> R @ gamma * I
}}
init {{ S = 990  I = 10 }}
simulate {{ from = 0 'days  to = 6 'days }}
"#));
    let names: Vec<String> = estimated.iter().map(|s| s.to_string()).collect();
    let priors = crate::fit::priors_precedence::resolve_priors_with_precedence(&names, &IndexMap::new(), &m);
    use_classes(&param_surfaces(&m), &priors, &m)
}

#[test]
fn a_hyperparameter_of_a_rate_parameter_is_hierarchical() {
    let c = classes_of(
        "  mu_g  : real ~ normal(mu = 0.0, sigma = 1.0)\n  gamma : rate ~ log_normal(mu = mu_g, sigma = 0.1)",
        &["mu_g", "gamma"],
    );
    assert_eq!(c["gamma"], UseClass::Likelihood);
    assert_eq!(c["mu_g"], UseClass::Hierarchical);
}

#[test]
fn a_hyperparameter_two_edges_up_is_hierarchical() {
    let c = classes_of(
        "  s_mu  : positive ~ half_normal(sigma = 1.0)\n  mu_g  : real ~ normal(mu = 0.0, sigma = s_mu)\n  \
         gamma : rate ~ log_normal(mu = mu_g, sigma = 0.1)",
        &["s_mu", "mu_g", "gamma"],
    );
    assert_eq!(c["s_mu"], UseClass::Hierarchical);
    assert_eq!(c["mu_g"], UseClass::Hierarchical);
}

#[test]
fn a_hyperparameter_over_a_held_leaf_is_not_hierarchical() {
    let c = classes_of(
        "  mu_g  : real ~ normal(mu = 0.0, sigma = 1.0)\n  gamma : rate ~ log_normal(mu = mu_g, sigma = 0.1)",
        &["mu_g"],
    );
    assert_eq!(c["mu_g"], UseClass::PriorOnly, "a child counts only while it is estimated");
}

#[test]
fn a_parameter_in_nothing_but_its_own_prior_is_prior_only() {
    let c = classes_of("  gamma : rate in [0.01, 0.5]\n  tau : positive ~ half_normal(sigma = 1.0)", &["gamma", "tau"]);
    assert_eq!(c["gamma"], UseClass::Likelihood);
    assert_eq!(c["tau"], UseClass::PriorOnly);
}

#[test]
fn a_hyperparameter_of_a_prior_only_parameter_is_prior_only() {
    let c = classes_of(
        "  gamma : rate in [0.01, 0.5]\n  mu_t : real ~ normal(mu = 0.0, sigma = 1.0)\n  \
         tau : positive ~ log_normal(mu = mu_t, sigma = 1.0)",
        &["gamma", "mu_t", "tau"],
    );
    assert_eq!(c["mu_t"], UseClass::PriorOnly);
    assert_eq!(c["tau"], UseClass::PriorOnly);
}

#[test]
fn a_parameter_in_nothing_with_a_flat_prior_is_unused() {
    let c = classes_of("  gamma : rate in [0.01, 0.5]\n  tau : positive in [0.1, 10.0]", &["gamma", "tau"]);
    assert_eq!(c["tau"], UseClass::Unused);
}

#[test]
fn an_uncertain_classification_puts_every_parameter_in_the_likelihood() {
    let m = model(|m| m.transitions.push(transition("infection", Expr::binding_ref("missing"))));
    let priors =
        crate::fit::priors_precedence::resolve_priors_with_precedence(&["tau".to_string()], &IndexMap::new(), &m);
    assert_eq!(use_classes(&param_surfaces(&m), &priors, &m)["tau"], UseClass::Likelihood);
}
