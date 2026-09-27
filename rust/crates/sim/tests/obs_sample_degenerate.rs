//! Regression, gh#619: the observation sampler must be total over degenerate
//! likelihood arguments.
//!
//! A prior draw whose epidemic dies can evaluate a likelihood argument to a
//! tiny positive value, `0/0 = NaN`, or `x/0 = inf` (a collapsed-compartment
//! denominator). Pre-fix, the neg_binomial arms called
//! `Gamma::new(k, m / k).unwrap()` behind a `m <= 0.0 || k <= 0.0` guard that
//! misses both hazards: a positive mean whose `m / k` underflows to exactly
//! `0.0`, and a NaN (`NaN <= 0.0` is false). One such draw aborted the whole
//! `simulate --obs` run with `rand_distr`'s `ScaleTooSmall`, leaving a partial
//! observation file. The Poisson arm handed a NaN rate to `rng.poisson`, whose
//! `NaN.min(1e15)` cap produced a silent draw of ~1e15; the Binomial /
//! Beta-family arms turned a NaN argument into in-range garbage draws.
//!
//! Contract pinned here: a NaN likelihood argument draws 0 and increments the
//! `obs_sample_nan` counter (surfaced by the end-of-run eval-stats summary); a
//! positive-but-underflowing NB mean falls back to the exact `k → ∞` limit,
//! `Poisson(mean)`, instead of panicking.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    expr::{ConstExpr, Expr, ProjectedExpr},
    model::{
        Compartment, CompartmentKind, InitialConditions, OutputConfig,
        OutputSchedule, SimulationConfig,
    },
    observation::{
        BernoulliLikelihood, BetaBinomialLikelihood, BetaLikelihood,
        BinomialLikelihood, ColumnRole, Likelihood, NegBinomialLikelihood,
        NormalLikelihood, ObsColumn, ObservationModel as IrObs,
        ObservationSchedule, PoissonLikelihood, Projection,
        ZeroInflatedNegBinomialLikelihood,
    },
    parameter::{ParamValue, Parameter},
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Diffable, Model,
};
use sim::{compiled_model::CompiledModel, rng::StatefulRng};

fn const_expr(v: f64) -> Expr {
    Expr::Const(ConstExpr { value: v })
}

fn projected() -> Expr {
    Expr::Projected(ProjectedExpr { projected: () })
}

/// A one-compartment inflow model with the given likelihood. The likelihood
/// arguments that matter are wired to `projected`, so each test drives the
/// degenerate value in as the sampler's `projected` call argument.
fn model_with_likelihood(likelihood: Likelihood) -> Arc<CompiledModel> {
    let m = Model {
        ic_grad: Default::default(),
        name: "degenerate_obs".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment {
            name: "R".into(),
            kind: CompartmentKind::Integer,
        }],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "inflow".into(),
            stoichiometry: vec![StoichiometryEntry("R".into(), 1)],
            rate: const_expr(1.0),
            metadata: None,
            draw_method: DrawMethod::Deterministic,
            rate_grad: Default::default(),
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![IrObs {
            name: "cases".into(),
            source: "cases".into(),
            columns: vec![
                ObsColumn { name: "time".into(), role: ColumnRole::Time },
                ObsColumn {
                    name: "count".into(),
                    role: ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
            ],
            scored: "count".into(),
            emit_schedule: Some(ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            covers: None,
            projection: Projection::CumulativeFlow("inflow".into()),
            projection_state_grad: Default::default(),
            likelihood,
        }],
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
            h.insert("R".into(), 0.0);
            h
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, 28.0]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: 28.0,
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

/// One draw through the real `compile_obs_sample_pf` closure.
fn draw_one(likelihood: Likelihood, projected_val: f64, seed: u64) -> f64 {
    let compiled = model_with_likelihood(likelihood);
    let obs = &compiled.model.observations[0];
    let params = compiled.default_params.clone();
    let sampler =
        sim::inference::obs_model::compile_obs_sample_pf(obs, compiled.clone(), &params)
            .expect("the fixture resolves");
    let counts = vec![0_i64];
    let mut rng = StatefulRng::new(seed);
    sampler(projected_val, 7.0, &counts, &[], &mut rng)
}

fn nan_counter() -> u64 {
    sim::eval_stats::EvalStats::snapshot().obs_sample_nan
}

fn neg_binomial(mean: Expr, dispersion: Expr) -> Likelihood {
    Likelihood::NegBinomial(NegBinomialLikelihood {
        mean: Diffable::new(mean),
        dispersion: Diffable::new(dispersion),
    })
}

// ── The reported crash: scale underflow ─────────────────────────────────────

#[test]
fn nb_mean_underflow_draws_zero_not_panic() {
    // m = 5e-324 (the smallest subnormal), k = 500 → m/k underflows to
    // exactly 0.0 → pre-fix: Gamma::new(500, 0.0).unwrap() aborts with
    // ScaleTooSmall. Post-fix: the Poisson(m) limit, which draws 0.
    let y = draw_one(neg_binomial(projected(), const_expr(500.0)), 5e-324, 42);
    assert_eq!(y, 0.0, "NB with underflowing mean must draw 0, got {y}");
}

#[test]
fn zinb_mean_underflow_draws_zero_not_panic() {
    // pi = 0 routes every draw into the NB base — same underflow, same abort.
    let lik = Likelihood::ZeroInflatedNegBinomial(ZeroInflatedNegBinomialLikelihood {
        mean: Diffable::new(projected()),
        dispersion: Diffable::new(const_expr(500.0)),
        pi: Diffable::new(const_expr(0.0)),
    });
    let y = draw_one(lik, 5e-324, 42);
    assert_eq!(y, 0.0, "ZINB with underflowing mean must draw 0, got {y}");
}

#[test]
fn nb_infinite_dispersion_is_poisson_limit() {
    // NB(m, k → ∞) → Poisson(m): scale = m/k underflows for any finite mean.
    // The fallback must keep the mean, not just avoid the panic.
    let lik = neg_binomial(const_expr(4.0), const_expr(f64::INFINITY));
    let compiled = model_with_likelihood(lik);
    let obs = &compiled.model.observations[0];
    let params = compiled.default_params.clone();
    let sampler =
        sim::inference::obs_model::compile_obs_sample_pf(obs, compiled.clone(), &params)
            .expect("the fixture resolves");
    let counts = vec![0_i64];
    let mut rng = StatefulRng::new(7);
    let n = 4000;
    let mean: f64 =
        (0..n).map(|_| sampler(0.0, 7.0, &counts, &[], &mut rng)).sum::<f64>() / n as f64;
    assert!(
        (mean - 4.0).abs() < 0.3,
        "Poisson(4) limit: sample mean ≈ 4, got {mean}"
    );
}

// ── NaN arguments: every arm draws 0 and counts ─────────────────────────────

/// Assert `draw = 0` and the `obs_sample_nan` counter advanced. The counter is
/// process-global and tests run in parallel, so assert on the per-call diff
/// being at least 1, never on absolute values.
fn assert_nan_draws_zero(likelihood: Likelihood, label: &str) {
    let before = nan_counter();
    let y = draw_one(likelihood, f64::NAN, 42);
    assert_eq!(y, 0.0, "{label}: NaN argument must draw 0, got {y}");
    assert!(
        nan_counter() > before,
        "{label}: NaN argument must increment obs_sample_nan"
    );
}

#[test]
fn nb_nan_mean_draws_zero_and_counts() {
    // 0/0 from a collapsed-compartment denominator. Pre-fix: NaN <= 0.0 is
    // false, so it reached Gamma::new(k, NaN).unwrap() → abort.
    assert_nan_draws_zero(neg_binomial(projected(), const_expr(500.0)), "neg_binomial");
}

#[test]
fn nb_nan_dispersion_draws_zero_and_counts() {
    assert_nan_draws_zero(neg_binomial(const_expr(5.0), projected()), "neg_binomial k");
}

#[test]
fn zinb_nan_pi_draws_zero_and_counts() {
    let lik = Likelihood::ZeroInflatedNegBinomial(ZeroInflatedNegBinomialLikelihood {
        mean: Diffable::new(const_expr(5.0)),
        dispersion: Diffable::new(const_expr(500.0)),
        pi: Diffable::new(projected()),
    });
    // Pre-fix: uniform() < NaN is false → silently "never zero-inflated",
    // and the NB base draws as if pi were 0.
    assert_nan_draws_zero(lik, "zero_inflated pi");
}

#[test]
fn poisson_nan_rate_draws_zero_and_counts() {
    // Pre-fix: rng.poisson(NaN) → NaN.min(1e15) = 1e15 → a silent draw of
    // ~1e15 in the observation file.
    assert_nan_draws_zero(
        Likelihood::Poisson(PoissonLikelihood { rate: Diffable::new(projected()) }),
        "poisson",
    );
}

#[test]
fn binomial_nan_p_draws_zero_and_counts() {
    // Pre-fix: NaN.clamp(0,1) = NaN walks past rng.binomial's p-guards.
    assert_nan_draws_zero(
        Likelihood::Binomial(BinomialLikelihood {
            n: const_expr(100.0),
            p: Diffable::new(projected()),
        }),
        "binomial",
    );
}

#[test]
fn beta_binomial_nan_alpha_draws_zero_and_counts() {
    // Pre-fix: NaN.max(LOG_PROB_FLOOR) = LOG_PROB_FLOOR → Beta(ε, β) → p ≈ 0
    // or 1 → an in-range garbage draw.
    assert_nan_draws_zero(
        Likelihood::BetaBinomial(BetaBinomialLikelihood {
            n: const_expr(100.0),
            alpha: Diffable::new(projected()),
            beta: Diffable::new(const_expr(2.0)),
        }),
        "beta_binomial",
    );
}

#[test]
fn beta_nan_mean_draws_zero_and_counts() {
    assert_nan_draws_zero(
        Likelihood::Beta(BetaLikelihood {
            mean: Diffable::new(projected()),
            concentration: Diffable::new(const_expr(10.0)),
        }),
        "beta",
    );
}

#[test]
fn normal_nan_mean_draws_zero_and_counts() {
    assert_nan_draws_zero(
        Likelihood::Normal(NormalLikelihood {
            mean: Diffable::new(projected()),
            sd: Diffable::new(const_expr(1.0)),
        }),
        "normal",
    );
}

#[test]
fn bernoulli_nan_p_draws_zero_and_counts() {
    assert_nan_draws_zero(
        Likelihood::Bernoulli(BernoulliLikelihood { p: Diffable::new(projected()) }),
        "bernoulli",
    );
}

// ── Negative control: valid arguments still draw from the real NB ───────────

#[test]
fn nb_valid_arguments_unchanged() {
    // NB(mean 5, k 10): draws must be non-degenerate (mean ≈ 5, overdispersed
    // enough to vary) — pins that the guards did not reroute the healthy path.
    let compiled = model_with_likelihood(neg_binomial(projected(), const_expr(10.0)));
    let obs = &compiled.model.observations[0];
    let params = compiled.default_params.clone();
    let sampler =
        sim::inference::obs_model::compile_obs_sample_pf(obs, compiled.clone(), &params)
            .expect("the fixture resolves");
    let counts = vec![0_i64];
    let mut rng = StatefulRng::new(11);
    let n = 4000;
    let draws: Vec<f64> = (0..n).map(|_| sampler(5.0, 7.0, &counts, &[], &mut rng)).collect();
    let mean = draws.iter().sum::<f64>() / n as f64;
    assert!((mean - 5.0).abs() < 0.4, "NB(5, 10) sample mean ≈ 5, got {mean}");
    assert!(draws.iter().any(|&y| y > 0.0), "healthy NB must draw non-zero values");
}

/// gh#651: a non-positive Normal `sd` is out of the likelihood's domain — the
/// value path scores it `-inf` — so it has no defined draw either. Pre-fix the
/// sampler drew `Normal(m, s.max(1e-10))`, emitting the rounded mean (10 here)
/// for parameters the scorer calls impossible. Same contract as a NaN
/// argument: draw 0 and count it.
#[test]
fn normal_non_positive_sd_draws_zero_and_counts() {
    let lik = || Likelihood::Normal(NormalLikelihood {
        mean: Diffable::new(const_expr(10.0)),
        sd: Diffable::new(projected()),
    });
    // Non-vacuity control: a positive sd draws near the mean.
    let y = draw_one(lik(), 1.0, 42);
    assert!((y - 10.0).abs() <= 5.0, "sd = 1 must draw near 10, got {y}");

    for sd in [-3.0, 0.0] {
        let before = nan_counter();
        let y = draw_one(lik(), sd, 42);
        assert_eq!(y, 0.0, "sd = {sd} must draw 0, got {y}");
        assert!(nan_counter() > before, "sd = {sd} must increment obs_sample_nan");
    }
}

/// gh#925: a Bernoulli `p` outside `[0, 1]` is out of the likelihood's domain
/// — the value path scores it `-inf` — so it has no defined draw either.
/// Pre-fix the sampler clamped it, so `p = 1.3` always drew 1 and `p = -0.3`
/// always drew 0, silently. Same contract as a NaN argument: draw 0 and count
/// it.
#[test]
fn bernoulli_out_of_range_p_draws_zero_and_counts() {
    let lik = || Likelihood::Bernoulli(BernoulliLikelihood { p: Diffable::new(projected()) });
    // Non-vacuity control: p = 1 is in the domain and always draws 1.
    assert_eq!(draw_one(lik(), 1.0, 42), 1.0, "p = 1 must draw 1");

    for p in [1.3, 1.0 + f64::EPSILON, -0.3] {
        let before = nan_counter();
        let y = draw_one(lik(), p, 42);
        assert_eq!(y, 0.0, "p = {p} must draw 0, got {y}");
        assert!(nan_counter() > before, "p = {p} must increment obs_sample_nan");
    }
}

/// gh#925: the zero-inflation probability `pi` outside `[0, 1]` likewise has
/// no defined draw. Pre-fix `pi = 1.3` was clamped to "always a structural
/// zero" and `pi = -0.3` to "never zero-inflated" (the NegBinomial base drew
/// as if `pi` were 0), neither counted.
#[test]
fn zinb_out_of_range_pi_draws_zero_and_counts() {
    let lik = || Likelihood::ZeroInflatedNegBinomial(ZeroInflatedNegBinomialLikelihood {
        mean: Diffable::new(const_expr(50.0)),
        dispersion: Diffable::new(const_expr(500.0)),
        pi: Diffable::new(projected()),
    });
    // Non-vacuity control: pi = 0 is the plain NegBinomial(50, 500), which
    // draws near its mean, not the zero the refusal draws.
    let y = draw_one(lik(), 0.0, 42);
    assert!(y > 0.0, "pi = 0 must draw from NegBinomial(50, 500), got {y}");

    for pi in [1.3, 1.0 + f64::EPSILON, -0.3] {
        let before = nan_counter();
        let y = draw_one(lik(), pi, 42);
        assert_eq!(y, 0.0, "pi = {pi} must draw 0, got {y}");
        assert!(nan_counter() > before, "pi = {pi} must increment obs_sample_nan");
    }
}

/// gh#925 sibling: a Binomial `p` outside `[0, 1]` has no defined draw — the
/// value path scores it `-inf`. Pre-fix the sampler clamped it, so `p = 1.3`
/// drew all `n` and `p = -0.3` drew 0, neither counted.
#[test]
fn binomial_out_of_range_p_draws_zero_and_counts() {
    let lik = || Likelihood::Binomial(BinomialLikelihood {
        n: const_expr(100.0),
        p: Diffable::new(projected()),
    });
    // Non-vacuity control: p = 1 is in the domain and draws all 100.
    assert_eq!(draw_one(lik(), 1.0, 42), 100.0, "p = 1 must draw n");

    for p in [1.3, 1.0 + f64::EPSILON, -0.3] {
        let before = nan_counter();
        let y = draw_one(lik(), p, 42);
        assert_eq!(y, 0.0, "p = {p} must draw 0, got {y}");
        assert!(nan_counter() > before, "p = {p} must increment obs_sample_nan");
    }
}

/// gh#925 sibling: a non-positive BetaBinomial shape is outside the domain —
/// `beta_binomial_logpmf` returns `-inf` for `alpha <= 0` or `beta <= 0` —
/// so it has no defined draw. Pre-fix the sampler floored each shape at
/// `LOG_PROB_FLOOR` and drew from `Beta(1e-300, ·)`: a point mass at 0 or at
/// `n`, uncounted.
#[test]
fn beta_binomial_non_positive_shape_draws_zero_and_counts() {
    let lik = |alpha: Expr, beta: Expr| Likelihood::BetaBinomial(BetaBinomialLikelihood {
        n: const_expr(100.0),
        alpha: Diffable::new(alpha),
        beta: Diffable::new(beta),
    });
    // Non-vacuity control: Beta(2, 2) mixing draws near n/2, not 0.
    let y = draw_one(lik(projected(), const_expr(2.0)), 2.0, 42);
    assert!(y > 0.0, "alpha = beta = 2 must draw from the BetaBinomial, got {y}");

    for shape in [0.0, -1.0] {
        for (which, l) in [
            ("beta", lik(const_expr(2.0), projected())),
            ("alpha", lik(projected(), const_expr(2.0))),
        ] {
            let before = nan_counter();
            let y = draw_one(l, shape, 42);
            assert_eq!(y, 0.0, "{which} = {shape} must draw 0, got {y}");
            assert!(nan_counter() > before, "{which} = {shape} must increment obs_sample_nan");
        }
    }
}

/// gh#925 sibling: `beta_logpdf` scores a mean outside `(0, 1)` or a
/// non-positive concentration as `-inf` (either Beta shape `mean·φ`,
/// `(1 − mean)·φ` is then non-positive), so the sampler has no defined draw.
/// Pre-fix it floored the shapes at `LOG_PROB_FLOOR` and drew a proportion
/// pinned at 0 or 1, uncounted.
#[test]
fn beta_invalid_shape_draws_zero_and_counts() {
    let lik = |mean: Expr, conc: Expr| Likelihood::Beta(BetaLikelihood {
        mean: Diffable::new(mean),
        concentration: Diffable::new(conc),
    });
    // Non-vacuity control: Beta(mean 0.5, φ 10) draws strictly inside (0, 1).
    let y = draw_one(lik(projected(), const_expr(10.0)), 0.5, 42);
    assert!(y > 0.0 && y < 1.0, "mean 0.5 must draw inside (0, 1), got {y}");

    let cases = [
        ("mean", 1.3), ("mean", 1.0), ("mean", 0.0), ("mean", -0.3),
        ("concentration", 0.0), ("concentration", -1.0),
    ];
    for (which, v) in cases {
        let l = if which == "mean" {
            lik(projected(), const_expr(10.0))
        } else {
            lik(const_expr(0.5), projected())
        };
        let before = nan_counter();
        let y = draw_one(l, v, 42);
        assert_eq!(y, 0.0, "{which} = {v} must draw 0, got {y}");
        assert!(nan_counter() > before, "{which} = {v} must increment obs_sample_nan");
    }
}

/// gh#925 sibling: `negbin_logpmf` scores a non-positive dispersion as `-inf`
/// (for a positive mean), so the sampler has no defined draw. Pre-fix
/// `draw_neg_binomial` returned 0 for it silently, indistinguishable from a
/// genuine zero count. It must be counted like any undefined argument. The
/// zero-inflated family draws its base through the same helper.
#[test]
fn neg_binomial_non_positive_dispersion_draws_zero_and_counts() {
    let zinb = || Likelihood::ZeroInflatedNegBinomial(ZeroInflatedNegBinomialLikelihood {
        mean: Diffable::new(const_expr(50.0)),
        dispersion: Diffable::new(projected()),
        pi: Diffable::new(const_expr(0.0)),
    });
    // Non-vacuity control: NB(50, k = 10) draws near 50, not 0.
    let y = draw_one(neg_binomial(const_expr(50.0), projected()), 10.0, 42);
    assert!(y > 0.0, "NB(50, 10) must draw a positive count, got {y}");

    for k in [0.0, -1.0] {
        for (label, lik) in [
            ("neg_binomial", neg_binomial(const_expr(50.0), projected())),
            ("zero_inflated", zinb()),
        ] {
            let before = nan_counter();
            let y = draw_one(lik, k, 42);
            assert_eq!(y, 0.0, "{label}: dispersion = {k} must draw 0, got {y}");
            assert!(nan_counter() > before,
                "{label}: dispersion = {k} must increment obs_sample_nan");
        }
    }
}
