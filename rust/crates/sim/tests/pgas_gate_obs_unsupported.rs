//! The `DerivEntry::Unsupported` preflight (proposal §4.4) at the `run_pgas`
//! boundary — the "green ≠ correct" gate for observation/σ² gradients.
//!
//! A NUTS fit runs only if every estimated parameter reaching an observation
//! (through a projection or any likelihood argument, after projection inlining)
//! is covered by a `DerivEntry::Grad`. These tests pin both verdicts:
//!
//!   * an estimated param carrying a real obs `Grad` is ADMITTED
//!     (`preflight_admits_obs_forcing_param_with_grad`);
//!   * an estimated param whose obs/σ² grad map carries an `Unsupported{code}` —
//!     or that reaches a parametric Binomial/BetaBinomial `n` — is REFUSED with
//!     the compiler's own reason, at the boundary, before any gradient is
//!     evaluated. Reaching `eval_emitted_grad` with an `Unsupported` entry would
//!     trip its `debug_assert!(false)`; a clean `Validation` error here proves
//!     the preflight fired first (`preflight_fires_before_eval_emitted_grad`).
//!
//! The obs grad maps are hand-built to carry exactly the entry the OCaml autodiff
//! would emit for the modelled position (a `Grad`, or an `Unsupported{code}` for a
//! tier-2b `lag`/Periodic coefficient). The preflight consumes these entries — it
//! never re-derives coverage.

use std::collections::HashMap;
use std::sync::Arc;

use ir::deriv::{DerivEntry, UnsupportedReason};
use sim::compiled_model::CompiledModel;
use sim::error::SimError;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::BoundObs;
use sim::inference::dense_cells;
use sim::inference::multi_stream_obs::{MultiStreamObsModel, StreamSpec, StreamProjection};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, simulate_reference, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::rng::StatefulRng;

fn load_model(path: &str) -> ir::Model {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    ir::from_str(&json).unwrap_or_else(|e| panic!("cannot parse {}: {}", path, e))
}

fn set_param_defaults(model: &mut ir::Model, defaults: &[(&str, f64)]) {
    for p in &mut model.parameters {
        if p.value.resolved_value().is_none() {
            if let Some(&(_, v)) = defaults.iter().find(|(n, _)| *n == p.name) {
                p.value = p.value.with_value(v);
            } else {
                p.value = p.value.with_value(0.5);
            }
        }
    }
}

/// SIR-overdispersion host with sane defaults. Its transitions carry `rate_grad`
/// (from the overdispersion σ²), so `has_gradients` is true and — absent the
/// preflight — the NUTS θ|X step WOULD reach `eval_emitted_grad`.
fn host_model() -> ir::Model {
    let mut model = load_model("../../../ocaml/golden/sir_overdispersion.ir.json");
    set_param_defaults(&mut model, &[
        ("beta", 0.3), ("gamma", 0.1), ("sigma_se", 0.1),
        ("N0", 1000.0), ("I0", 10.0),
    ]);
    model
}

fn params_from_compiled(compiled: &CompiledModel) -> Vec<f64> {
    let n_params = compiled.param_index.len();
    let mut params = vec![0.0; n_params];
    for p in &compiled.model.parameters {
        if let Some(v) = p.value.resolved_value() {
            params[compiled.param_index[p.name.as_str()]] = v;
        }
    }
    params
}

fn estimated_param(name: &str, init: f64) -> ir::parameter::Parameter {
    ir::parameter::Parameter {
        name: name.into(),
        value: ir::parameter::ParamValue::Estimated {
            init: Some(init),
            bounds: Some((0.01, 100.0)),
            prior: ir::parameter::PriorSpec::Flat,
            transform: ir::parameter::Transform::Identity,
        },
        param_kind: Some(ir::parameter::ParamKind::Positive),
        param_dim: None,
    }
}

fn obs_columns() -> Vec<ir::observation::ObsColumn> {
    vec![
        ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
        ir::observation::ObsColumn {
            name: "weekly_cases".into(),
            role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count),
        },
    ]
}

/// Attempt a small NUTS fit that estimates `param` (init `param_init`) against
/// `obs_block`, returning the `run_pgas` result. The preflight runs at the top of
/// `run_pgas`, so a refusal returns before the sweep loop.
fn attempt_nuts_fit(
    obs_block: ir::observation::ObservationModel,
    param: &str,
    param_init: f64,
) -> Result<sim::inference::pgas::PGASResult, SimError> {
    attempt_nuts_fit_with(param, param_init, |m| m.observations = vec![obs_block])
}

/// As `attempt_nuts_fit`, but `setup` gets the model (with `param` already pushed
/// as estimated) to configure — set the observation and/or inject a transition
/// `rate_grad` entry. Used by the rate-domain gate tests (gh#342 P4).
fn attempt_nuts_fit_with(
    param: &str,
    param_init: f64,
    setup: impl FnOnce(&mut ir::Model),
) -> Result<sim::inference::pgas::PGASResult, SimError> {
    let mut model = host_model();
    model.parameters.push(estimated_param(param, param_init));
    setup(&mut model);

    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = params_from_compiled(&compiled);

    let dt = 1.0;
    let mut rng = StatefulRng::new(42);
    let t_end = compiled.model.simulation.t_end;
    let truth_traj = simulate_reference(&compiled, &params, t_end, dt, &mut rng).unwrap();

    // Synthetic weekly obs by summing the infection flow into windows.
    let mut cum: u64 = 0;
    let mut obs: Vec<Observation> = Vec::new();
    for (s, rec) in truth_traj.substeps.iter().enumerate() {
        cum += rec.flows[0];
        let t = ((s + 1) as f64) * dt;
        if (t.round() as i64) % 7 == 0 {
            // Halve so a Binomial/BetaBinomial `k <= n` constraint holds where relevant.
            obs.push(Observation { time: t, value: (cum as f64 * 0.5).round() });
            cum = 0;
        }
    }

    let stream_proj = StreamProjection::from_ir(
        &compiled.model.observations[0].projection, &compiled, "weekly_cases",
    ).unwrap();

    let obs_model = MultiStreamObsModel::new(
        BoundObs::bind(vec![StreamSpec {
            projection: stream_proj,
            ir_model: compiled.model.observations[0].clone(),
            observations: dense_cells(obs.iter().map(|o| o.value).collect()),
            obs_times: obs.iter().map(|o| o.time).collect(),
            aux: vec![],
        }]).unwrap().0,
        compiled.clone(),
    ).unwrap();

    let if2_params = vec![EstimatedParam {
        name: param.into(),
        index: compiled.param_index[param],
        initial: param_init,
        rw_sd: 0.02,
        transform: Transform::Log { lo: 0.01, hi: 100.0 },
        lower: 0.01,
        upper: 100.0,
        rw_sd_auto: false,
        ivp: false,
    }];
    let priors = vec![Prior::Fixed(sim::inference::prior::Density::Flat)];

    let config = PGASConfig {
        n_particles: 50,
        n_sweeps: 3,
        burn_in: 1,
        thin: 1,
        dt,
        use_nuts: true,
        dense_mass: false,
        max_tree_depth: 4,
        tempering: vec![1.0],
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };

    run_pgas(
        &compiled, &if2_params, &priors, &params,
        &config, &obs, &obs_model, 12345, None, None, "gate_obs_unsupported".into(),
    )
}

/// Poisson obs over the infection flow whose `rate_grad` carries exactly `entry`
/// for `param` — the position the compiler differentiated.
fn poisson_obs_with_grad(param: &str, entry: DerivEntry) -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: obs_columns(),
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("infection".into()),
        likelihood: Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable { expr: Expr::Projected(ProjectedExpr { projected: () }), grad: HashMap::from([(param.to_string(), entry)]), proj_grad: None },
        }),
    }
}

/// A benign Poisson obs with an EMPTY grad — no estimated param is refused via
/// the observation, so a rate-domain refusal can be isolated.
fn benign_obs() -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: obs_columns(),
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection: Projection::CumulativeFlow("infection".into()),
        likelihood: Likelihood::Poisson(PoissonLikelihood {
            rate: ir::Diffable::new(Expr::Projected(ProjectedExpr { projected: () })),
        }),
    }
}

/// A model-setup closure that sets a benign obs and injects `entry` as `param`'s
/// rate_grad on the `infection` transition — the RATE analogue of an obs grad.
fn with_rate_grad(param: &str, entry: DerivEntry) -> impl FnOnce(&mut ir::Model) {
    let param = param.to_string();
    move |m: &mut ir::Model| {
        m.observations = vec![benign_obs()];
        for t in &mut m.transitions {
            if t.name == "infection" {
                t.rate_grad.insert(param.clone(), entry.clone());
            }
        }
    }
}

// ── Rate-domain gate (gh#342 P4): the preflight now scans transition rate_grad ──

#[test]
fn preflight_refuses_periodic_coeff_in_rate() {
    // A Periodic step-value coefficient in a RATE serialises Unsupported{Periodic}
    // (gh#342 P3); the preflight refuses it at the run_pgas boundary — the rate
    // analogue of the obs case, subsuming coeff_guard's periodic set.
    let r = attempt_nuts_fit_with("wpeak", 1.0, with_rate_grad(
        "wpeak",
        DerivEntry::Unsupported { node: "time_func:weekly".into(), code: UnsupportedReason::PeriodicCoeff },
    ));
    match r {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("wpeak"), "must name the refused rate param; got: {}", msg);
            assert!(msg.contains("Periodic"), "must carry the Periodic reason; got: {}", msg);
        }
        Ok(_) => panic!("expected a rate Periodic-coeff refusal, but the fit was admitted"),
        Err(e) => panic!("expected a Validation refusal, got: {:?}", e),
    }
}

#[test]
fn preflight_refuses_lag_in_rate() {
    // A forcing `lag` param reaching a RATE: Unsupported{Lag}; refused.
    let r = attempt_nuts_fit_with("tau", 1.0, with_rate_grad(
        "tau",
        DerivEntry::Unsupported { node: "time_func:seasonal".into(), code: UnsupportedReason::Lag },
    ));
    match r {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("tau"), "must name the refused rate param; got: {}", msg);
            assert!(msg.contains("lag"), "must carry the lag reason; got: {}", msg);
        }
        Ok(_) => panic!("expected a rate lag refusal, but the fit was admitted"),
        Err(e) => panic!("expected a Validation refusal, got: {:?}", e),
    }
}

#[test]
fn preflight_admits_rate_forcing_param_with_grad() {
    // A Sinusoidal amplitude in a RATE carries a real Grad → admitted (the rate
    // analogue of the obs acceptance test; the tier-1 case coeff_guard's has_grad
    // escape allowed, now admitted by construction).
    let r = attempt_nuts_fit_with("amp", 0.3, with_rate_grad(
        "amp",
        DerivEntry::Grad(ir::expr::Expr::pop("S")),
    ));
    match r {
        Err(SimError::Validation(msg)) if msg.contains("could not emit") =>
            panic!("preflight must ADMIT a rate param with a real Grad; got refusal: {}", msg),
        _ => {} // Ok, or an unrelated numerical error, is acceptable here
    }
}

#[test]
fn preflight_refuses_non_const_table_value_in_rate() {
    // An inline-table VALUE reached by a non-constant index (tier 2b) serialises
    // Unsupported{NonConstTableIndex}; the preflight refuses it, completing the
    // rate-domain tier-2b triad (Periodic / lag / non-const table). The scan is
    // code-agnostic — it refuses any `Unsupported` — so this exercises the same
    // path as the Periodic/lag tests with the remaining reason code.
    let r = attempt_nuts_fit_with("kcell", 1.0, with_rate_grad(
        "kcell",
        DerivEntry::Unsupported {
            node: "table `k_tbl`".into(),
            code: UnsupportedReason::NonConstTableIndex,
        },
    ));
    match r {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("kcell"), "must name the refused rate param; got: {}", msg);
            assert!(msg.contains("rate"), "must attribute the refusal to a rate term; got: {}", msg);
        }
        Ok(_) => panic!("expected a rate non-const-table-value refusal, but the fit was admitted"),
        Err(e) => panic!("expected a Validation refusal, got: {:?}", e),
    }
}

// ── Acceptance (headline): a real obs Grad is ADMITTED ────────────────────────

#[test]
fn preflight_admits_obs_forcing_param_with_grad() {
    // A parameter carrying a real emitted observation gradient (as a Sinusoidal
    // amplitude used only in an observation does — the compiler emits
    // `∂rate/∂amp` as a `DerivEntry::Grad`) is ADMITTED and NUTS-estimated. This
    // is the false-positive-refusal the ledger prevents: before P5 the coeff
    // guard fenced such a param (its `has_grad` was `rate_grad`-only). The
    // CLI-side coeff_guard partition is pinned in
    // `cli::fit::coeff_guard::tests`; here we pin that the `run_pgas` preflight
    // admits a `Grad` entry and the fit runs.
    let obs = poisson_obs_with_grad(
        "amp",
        DerivEntry::Grad(ir::expr::Expr::Projected(ir::expr::ProjectedExpr { projected: () })),
    );
    let result = attempt_nuts_fit(obs, "amp", 0.3);
    match result {
        Ok(_) => {}
        Err(SimError::Validation(msg)) if msg.contains("could not emit") =>
            panic!("preflight must ADMIT a param with a real obs Grad; got refusal: {}", msg),
        Err(e) => panic!("run_pgas failed for an unrelated reason: {:?}", e),
    }
}

// ── Refusals: an Unsupported obs/σ² entry, carried with its reason ────────────

#[test]
fn preflight_refuses_lag_in_observation() {
    // A parameter driving a forcing's `lag` and reaching an observation: the
    // compiler classifies `∂obs/∂tau` as `Omitted{Lag}` → serialized as
    // `DerivEntry::Unsupported{Lag}`. The preflight refuses with the lag reason.
    // (No lag-in-observation coverage existed before P5 — this is net-new, and
    // pins the §4.2 lag guard.)
    let obs = poisson_obs_with_grad(
        "tau",
        DerivEntry::Unsupported { node: "time_func:seasonal".into(), code: UnsupportedReason::Lag },
    );
    let result = attempt_nuts_fit(obs, "tau", 1.0);
    match result {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("tau"), "must name the refused param; got: {}", msg);
            assert!(msg.contains("lag"), "must carry the lag reason; got: {}", msg);
        }
        Ok(_) => panic!("expected a lag refusal, but the fit was admitted"),
        Err(e) => panic!("expected a lag Validation refusal, got: {:?}", e),
    }
}

#[test]
fn preflight_refuses_periodic_coeff_in_observation() {
    // tier-2b: a Periodic forcing coefficient in an observation → the compiler
    // emits `Unsupported{PeriodicCoeff}`; the preflight refuses, surfacing the
    // Periodic reason (the masquerade-as-genuine-zero fix, §4.2).
    let obs = poisson_obs_with_grad(
        "wpeak",
        DerivEntry::Unsupported { node: "time_func:weekly".into(), code: UnsupportedReason::PeriodicCoeff },
    );
    let result = attempt_nuts_fit(obs, "wpeak", 1.0);
    match result {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("wpeak"), "must name the refused param; got: {}", msg);
            assert!(msg.contains("Periodic"), "must carry the Periodic reason; got: {}", msg);
        }
        Ok(_) => panic!("expected a Periodic refusal, but the fit was admitted"),
        Err(e) => panic!("expected a Periodic Validation refusal, got: {:?}", e),
    }
}

#[test]
fn preflight_fires_before_eval_emitted_grad() {
    // The invariant that keeps `eval_emitted_grad`'s `debug_assert!(false)`
    // unreachable: an estimated param with an `Unsupported` obs grad is refused
    // at the boundary, before the NUTS θ|X step evaluates any gradient. The host
    // has `rate_grad` (overdispersion) so `has_gradients` is true — WITHOUT the
    // preflight this fit would reach `eval_emitted_grad` and panic on the
    // debug_assert in this (debug) build. A clean `Validation` return proves the
    // preflight fired first.
    let obs = poisson_obs_with_grad(
        "kstruct",
        DerivEntry::Unsupported {
            node: "time_func:spline".into(),
            code: UnsupportedReason::StructuralForcing,
        },
    );
    let result = attempt_nuts_fit(obs, "kstruct", 0.5);
    match result {
        Err(SimError::Validation(msg)) =>
            assert!(msg.contains("kstruct"), "must name the refused param; got: {}", msg),
        Ok(_) => panic!(
            "preflight must refuse a StructuralForcing obs param before eval_emitted_grad; \
             the fit was admitted (had the preflight not fired, eval_emitted_grad would \
             have tripped its debug_assert in this build)"
        ),
        Err(e) => panic!("expected a Validation refusal, got: {:?}", e),
    }
}

// ── D-n: a θ-dependent Binomial/BetaBinomial `n` is REFUSED ───────────────────

/// Binomial obs with an explicit `n` expression and `p = 0.5` (constant).
fn binomial_obs_with_n(n: ir::expr::Expr, projection: ir::observation::Projection)
    -> ir::observation::ObservationModel {
    use ir::expr::*;
    use ir::observation::*;
    ObservationModel {
        name: "weekly_cases".into(),
        source: "weekly_cases".into(),
        columns: obs_columns(),
        scored: "weekly_cases".into(),
        emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
        stratum: vec![],
        projection,
        likelihood: Likelihood::Binomial(BinomialLikelihood {
            n,
            p: ir::Diffable::new(Expr::Const(ConstExpr { value: 0.5 })),
        }),
    }
}

#[test]
fn preflight_refuses_parametric_binomial_n_direct() {
    // `n = qgam` directly: the D-n scan collects `qgam` and refuses with the
    // ParametricN reason — `n` is rounded to an integer and carries no gradient,
    // so a param there would be silently unconstrained under NUTS.
    use ir::expr::*;
    let obs = binomial_obs_with_n(
        Expr::Param(ParamExpr { param: "qgam".into() }),
        ir::observation::Projection::CumulativeFlow("infection".into()),
    );
    let result = attempt_nuts_fit(obs, "qgam", 100.0);
    match result {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("qgam"), "must name the refused param; got: {}", msg);
            assert!(msg.contains("θ-independent"), "must give the n message; got: {}", msg);
        }
        Ok(_) => panic!("expected a parametric-n refusal, but the fit was admitted"),
        Err(e) => panic!("expected a parametric-n Validation refusal, got: {:?}", e),
    }
}

#[test]
fn preflight_refuses_parametric_binomial_n_via_projection() {
    // `n = projected` where the projection is a parametric `DerivedExpr`
    // (`qgam * I`): inlining the projection into `n` (matching the autodiff)
    // reveals `qgam`, so the D-n scan refuses it. Without the inlining this
    // silent hole would pass.
    use ir::expr::*;
    let projection_expr = Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Mul,
            left: Box::new(Expr::Param(ParamExpr { param: "qgam".into() })),
            right: Box::new(Expr::pop("I")),
        },
    });
    let obs = binomial_obs_with_n(
        Expr::Projected(ProjectedExpr { projected: () }),
        ir::observation::Projection::DerivedExpr(projection_expr),
    );
    let result = attempt_nuts_fit(obs, "qgam", 100.0);
    match result {
        Err(SimError::Validation(msg)) => {
            assert!(msg.contains("qgam"), "must name the refused param; got: {}", msg);
            assert!(msg.contains("θ-independent"), "must give the n message; got: {}", msg);
        }
        Ok(_) => panic!("expected a parametric-n (via projection) refusal, but the fit was admitted"),
        Err(e) => panic!("expected a parametric-n Validation refusal, got: {:?}", e),
    }
}
