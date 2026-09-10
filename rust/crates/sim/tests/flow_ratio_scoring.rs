//! Proposal 2026-09-09: a `FlowRatio` stream is scored as the ratio of its two
//! accumulated bins over the row's window, and a window with no denominator
//! events projects NaN, which every paired likelihood family scores under the
//! existing NaN contract: a row with `n = 0` scores exactly 0 (gh#812), and a
//! row that recorded events the trajectory made none of is refused with the
//! argument named.
//!
//! Deterministic flows (`--> Dc @ deterministic(Kc)`, `--> Df @
//! deterministic(Kf)`) so the per-window bins are exact and the projected
//! ratio is a known fraction. The tests drive the fold / score / reset seam
//! every filter routes through, as `per_stream_reset.rs` does, rather than a
//! filter, because the property under test is the seam's.
//!
//! Two mutation checks are made executable here rather than left as a
//! reviewer's exercise: the fraction is asymmetric (3 : 8, not 1 : 2), so a
//! reader that swapped the numerator and denominator bins fails the value
//! assertion; and a reader that returned 0 for a zero denominator instead of
//! NaN fails `is_nan` and the named-cause assertions.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{BinOp, Expr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::{
        BernoulliLikelihood, BetaBinomialLikelihood, BetaLikelihood, BinomialLikelihood,
        ColumnRole, Covers, Likelihood, ObsColumn, ObservationModel as IrObs,
        ObservationSchedule, Projection,
    },
    parameter::{ParamKind, ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Diffable, Model,
};
use sim::{
    compiled_model::CompiledModel,
    inference::{
        dense_cells,
        multi_stream_obs::{BoundObs, MultiStreamObsModel, StreamProjection, StreamSpec, StreamTimes},
        obs_attempt::NegInfCause,
        obs_loglik::binom_logpmf,
    },
};

fn projected() -> Expr {
    Expr::Projected(ProjectedExpr { projected: () })
}

/// `--> target @ deterministic(k)`: exactly `k` events per unit time.
fn deterministic(name: &str, target: &str, k: f64) -> Transition {
    Transition {
        rate_state_grad: Default::default(),
        name: name.into(),
        stoichiometry: vec![StoichiometryEntry(target.into(), 1)],
        rate: Expr::const_(k),
        metadata: None,
        draw_method: DrawMethod::Deterministic,
        rate_grad: Default::default(),
        lineage: None,
    }
}

fn col(name: &str, kind: ParamKind) -> ObsColumn {
    ObsColumn { name: name.into(), role: ColumnRole::Value(kind) }
}

/// The stream `comm / (comm + fac)` over weekly windows, scored by
/// `likelihood`; `extra` are the data columns the likelihood reads.
fn ratio_stream(name: &str, scored_kind: ParamKind, likelihood: Likelihood, extra: &[&str]) -> IrObs {
    let mut columns = vec![
        ObsColumn { name: "time".into(), role: ColumnRole::Time },
        col(name, scored_kind),
    ];
    columns.extend(extra.iter().map(|c| col(c, ParamKind::Count)));
    IrObs {
        name: name.into(),
        source: name.into(),
        columns,
        scored: name.into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        covers: Some(Covers::Until { offset: 0.0, span: 7.0 }),
        projection: Projection::FlowRatio {
            numerator: vec!["comm".into()],
            denominator: vec!["comm".into(), "fac".into()],
        },
        projection_state_grad: Default::default(),
        likelihood,
    }
}

/// Two absorbing death compartments fed by two deterministic flows.
fn model(kc: f64, kf: f64, observations: Vec<IrObs>) -> Arc<CompiledModel> {
    let m = Model {
        ic_grad: Default::default(),
        name: "flow_ratio".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "Dc".into(), kind: CompartmentKind::Integer },
            Compartment { name: "Df".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![deterministic("comm", "Dc", kc), deterministic("fac", "Df", kf)],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations,
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![Parameter {
            name: "dummy".into(),
            value: ParamValue::Fixed { value: 0.0 },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut h = HashMap::new();
            h.insert("Dc".into(), 0.0);
            h.insert("Df".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 21.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 21.0,
            time_semantics: "continuous".into(),
            dt: Some(1.0),
            rng_seed: Some(1),
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
    Arc::new(CompiledModel::new(m).unwrap())
}

/// Bind the model's one stream at `times` with the given cells and per-row
/// aux, through the same `bind` the fit loader uses.
fn bind_ratio(
    compiled: Arc<CompiledModel>,
    times: Vec<f64>,
    cells: Vec<f64>,
    aux: Vec<Vec<(String, f64)>>,
) -> MultiStreamObsModel {
    let om = compiled.model.observations[0].clone();
    let projection = StreamProjection::from_ir(&om.projection, &compiled, &om.name).unwrap();
    let spec = StreamSpec {
        times: StreamTimes::contiguous_for(&projection, 0.0, times).unwrap(),
        projection,
        ir_model: om,
        observations: dense_cells(cells),
        aux,
    };
    MultiStreamObsModel::new(BoundObs::bind(0.0, vec![spec]).unwrap().0, compiled).unwrap()
}

/// Fold one closed window's per-transition flow into the bins, read what the
/// stream projects and scores there, then reset — the lifecycle every filter
/// runs at a union index. Returns the bins as read, so a test can see that a
/// window was scored on its own tally: a ratio of cumulative bins would give
/// the same quotient, so the projected value alone cannot prove the reset.
fn score_window(
    obs: &MultiStreamObsModel,
    acc: &mut [u64],
    flows: &[u64],
    ui: usize,
    params: &[f64],
) -> (Vec<u64>, f64, f64) {
    let counts = vec![0i64; 2];
    obs.fold_into_acc(flows, acc);
    let bins = acc.to_vec();
    let p = obs.project_stream(0, acc, &counts, params, ui);
    let ll = obs.log_likelihood_from_flows_and_counts(acc, &counts, ui, params);
    obs.reset_due_acc(ui, acc);
    (bins, p, ll)
}

fn binomial_on(column: &str) -> Likelihood {
    Likelihood::Binomial(BinomialLikelihood {
        n: Expr::obs_column_ref(column),
        p: Diffable::new(projected()),
    })
}

/// The projected value is `Σ comm / (Σ comm + Σ fac)` over each window, and
/// the row scores `binomial(k, n, that)` — with `n` the data column, not the
/// model's denominator, which enters only through `p` (the likelihood is the
/// conditional `p(k | n, x)`).
#[test]
fn a_ratio_stream_scores_the_windows_flow_ratio() {
    let (kc, kf) = (3.0, 5.0);
    let compiled = model(kc, kf, vec![ratio_stream(
        "comm_frac", ParamKind::Count, binomial_on("n_deaths"), &["n_deaths"],
    )]);
    let times = vec![7.0, 14.0, 21.0];
    let k = [8.0, 21.0, 0.0];
    let n = [20.0, 56.0, 10.0];
    let aux = n.iter().map(|&v| vec![("n_deaths".to_string(), v)]).collect();
    let obs = bind_ratio(compiled.clone(), times.clone(), k.to_vec(), aux);
    let params = compiled.default_params.clone();

    // One ratio stream owns two bins, numerator first, and names them so a
    // posterior trajectory can carry both counts.
    assert_eq!(obs.n_interval_streams(), 2, "a ratio stream owns two acc bins");
    let (comm, fac) = (0usize, 1usize); // transition order in `model`
    assert_eq!(
        obs.incidence_streams(),
        vec![
            ("comm_frac_numerator".to_string(), vec![comm]),
            ("comm_frac_denominator".to_string(), vec![comm, fac]),
        ]
    );

    let mut acc = vec![0u64; obs.n_interval_streams()];
    let week = [(7.0 * kc) as u64, (7.0 * kf) as u64]; // 21 and 35 events
    let want_p = 21.0 / 56.0;
    for ui in 0..times.len() {
        let (bins, p, ll) = score_window(&obs, &mut acc, &week, ui, &params);
        // Each window is scored on its own week's tally: the two bins were
        // reset at the window's start (gh#833's schedule), so the third
        // window reads 21 and 56, not 63 and 168.
        assert_eq!(bins, vec![21, 56], "window {ui}: numerator then denominator bins");
        assert_eq!(
            p, want_p,
            "window {ui}: Σcomm / (Σcomm + Σfac) = 21/56; the swapped bins would give {}",
            56.0 / 21.0
        );
        let want = binom_logpmf(k[ui] as u64, n[ui] as u64, want_p);
        assert_eq!(
            ll, want,
            "window {ui}: the row scores binomial(k = {}, n = {}, p = 21/56)",
            k[ui], n[ui]
        );
    }
}

/// A window in which no denominator event occurred projects NaN, and every
/// family the ratio pairs with scores that under the existing NaN contract.
#[test]
fn a_window_with_no_denominator_events_projects_nan_under_every_family() {
    let beta_binomial = || {
        Likelihood::BetaBinomial(BetaBinomialLikelihood {
            n: Expr::obs_column_ref("n_deaths"),
            alpha: Diffable::new(Expr::bin_op(BinOp::Mul, projected(), Expr::const_(50.0))),
            beta: Diffable::new(Expr::bin_op(
                BinOp::Mul,
                Expr::bin_op(BinOp::Sub, Expr::const_(1.0), projected()),
                Expr::const_(50.0),
            )),
        })
    };
    let beta = || {
        Likelihood::Beta(BetaLikelihood {
            mean: Diffable::new(projected()),
            concentration: Diffable::new(Expr::const_(50.0)),
        })
    };
    let bernoulli = || Likelihood::Bernoulli(BernoulliLikelihood { p: Diffable::new(projected()) });
    let counts = vec![0i64; 2];
    let no_events = [0u64, 0u64];

    // k of n: an `n = 0` row scores exactly 0 whatever the model's fraction
    // (gh#812); an `n > 0` row is refused and the refusal names the argument
    // the value function reads first.
    for (label, lik, arg) in [
        ("binomial", binomial_on("n_deaths"), "p"),
        ("beta_binomial", beta_binomial(), "alpha"),
    ] {
        let compiled = model(0.0, 0.0, vec![ratio_stream(
            "comm_frac", ParamKind::Count, lik, &["n_deaths"],
        )]);
        let aux = vec![
            vec![("n_deaths".to_string(), 0.0)],
            vec![("n_deaths".to_string(), 5.0)],
        ];
        let obs = bind_ratio(compiled.clone(), vec![7.0, 14.0], vec![0.0, 2.0], aux);
        let params = compiled.default_params.clone();
        let mut acc = vec![0u64; 2];

        let (_, p0, ll0) = score_window(&obs, &mut acc, &no_events, 0, &params);
        assert!(p0.is_nan(), "{label}: a fraction of no events is not a number, got {p0}");
        assert_eq!(ll0, 0.0, "{label}: n = 0 has one possible outcome and scores exactly 0");

        let (_, p1, ll1) = score_window(&obs, &mut acc, &no_events, 1, &params);
        assert!(p1.is_nan());
        assert_eq!(
            ll1,
            f64::NEG_INFINITY,
            "{label}: the data recorded 5 classified events in a window this trajectory \
             produced none in"
        );
        let live: Vec<(&[u64], &[i64])> = vec![(&[0u64, 0][..], counts.as_slice())];
        let a = &obs.stream_attempts(1, &live, 0, &params)[0];
        assert_eq!(
            a.neg_inf_causes,
            vec![(NegInfCause::ArgumentNaN { arg: arg.into() }, 1)],
            "{label}: the refusal names the argument: {a}"
        );
    }

    // A fraction reported directly, and a binary outcome: no `n` to be zero,
    // so the row is refused outright, and named (bernoulli's NaN guard is
    // 736e4f2e; before it the row scored a finite floor).
    for (label, lik, kind, observed, arg) in [
        ("beta", beta(), ParamKind::Probability, 0.4, "mean"),
        ("bernoulli", bernoulli(), ParamKind::Count, 1.0, "p"),
    ] {
        let compiled = model(0.0, 0.0, vec![ratio_stream("comm_frac", kind, lik, &[])]);
        let obs = bind_ratio(compiled.clone(), vec![7.0], vec![observed], vec![vec![]]);
        let params = compiled.default_params.clone();
        let mut acc = vec![0u64; 2];
        let (_, p, ll) = score_window(&obs, &mut acc, &no_events, 0, &params);
        assert!(p.is_nan(), "{label}: got {p}");
        assert_eq!(ll, f64::NEG_INFINITY, "{label}: a NaN projection is -inf under every family");
        let live: Vec<(&[u64], &[i64])> = vec![(&[0u64, 0][..], counts.as_slice())];
        let a = &obs.stream_attempts(0, &live, 0, &params)[0];
        assert_eq!(
            a.neg_inf_causes,
            vec![(NegInfCause::ArgumentNaN { arg: arg.into() }, 1)],
            "{label}: the refusal names the argument: {a}"
        );
    }
}
