//! Is NUTS, as PGAS drives it, a valid sampler of the target the fit parameter
//! infrastructure hands it?
//!
//! The target NUTS sees for one estimated parameter is assembled from three
//! pieces of shared infrastructure: the `Transform` of the `EstimatedParam`
//! (θ = f(z) and its derivatives), the `Prior` density on the natural scale,
//! and the log-Jacobian log|dθ/dz|. This file checks each layer against an
//! independent answer, bottom up:
//!
//! 1. *Transforms, deterministically.* The maps a fit reaches
//!    (`LowerBounded { 0 }`, `LogInterval`, `Interval`, `Identity`): the
//!    round trip z → θ → z and θ → z → θ, and `transform_deriv`,
//!    `log_jacobian` and `jacobian_grad` against central finite differences
//!    of `from_transformed`. And that no map has a wall: the log-Jacobian is
//!    finite at every finite z.
//! 2. *The NUTS kernel alone* (`nuts::nuts_step` with dual-averaging
//!    adaptation, no PGAS, no simulator) on targets built from that
//!    infrastructure, with known answers: an independent Gaussian, and
//!    log τ for τ ~ exponential(λ) with no box.
//!    The sharper stationarity check of the kernel at fixed step (200,000
//!    draws, which resolves an asymmetric leapfrog kick these runs do not)
//!    is `nuts_stationarity.rs`; these add adaptation and divergences.
//! 3. *PGAS + NUTS with no likelihood* (`run_pgas`): a model whose estimated
//!    parameters enter neither the dynamics nor the observation model, so the
//!    posterior is the prior exactly. τ's draws must match exponential(λ)
//!    (Kolmogorov–Smirnov) and z's must match N(0, 1) in mean and variance.
//! 4. *A boxed parameter has no wall*
//!    (`boxed_parameter_diverges_no_more_than_the_unboxed_one`): log τ boxed to
//!    [1e-6, 2.5] under `LogInterval` diverges no more than the unboxed
//!    control's bound, and its draws are the truncated exponential.
//! 5. *Prior-only runs recover the truncated prior* under PGAS+NUTS and under
//!    PMMH (whose kernel ODE-MH shares): τ ~ exponential(5.99) on [1e-6, 2.5],
//!    ρ ~ beta(2, 5) on [0.1, 0.9] (both `LogInterval`) and κ ~ half_normal(1)
//!    on [0, ∞) (`LowerBounded { 0 }`), proposal §12.3.
//!
//! The per-parameter target *value and gradient* the PGAS closure assembles
//! from these pieces (`Prior::log_density_and_grad_z`) are
//! checked against analytic densities in `src/inference/nuts_target_tests.rs`.
//!
//! Every stochastic test runs at a fixed seed, so it is deterministic; each
//! threshold's comment states the measured value at that seed and why the
//! margin is safe.

use std::sync::Arc;

use ir::{
    deriv::DerivEntry,
    expr::Expr,
    model::{
        Compartment, CompartmentKind, InitSpec, InitialConditions, OutputConfig, OutputSchedule,
        SimulationConfig,
    },
    observation::PoissonLikelihood,
    parameter::Parameter,
    transition::{DrawMethod, StoichiometryEntry, Transition},
    Diffable, Model,
};
use sim::compiled_model::CompiledModel;
use sim::inference::convergence::rank_convergence;
use sim::inference::if2::{EstimatedParam, Transform};
use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
use sim::inference::nuts::{nuts_step, DualAveraging, MassMatrix, NUTSConfig};
use sim::inference::particle_filter::Observation;
use sim::inference::pgas::{run_pgas, PGASConfig};
use sim::inference::pmmh::Prior;
use sim::inference::prior::Density;
use sim::inference::{BoundObs, MultiStreamObsModel};
use sim::rng::StatefulRng;


/// The exponential prior rate on τ used throughout (the rate of the B-spline
/// smoothing-scale prior these diagnostics were first run for).
const LAMBDA: f64 = 5.99;
/// Euler–Mascheroni γ: E[log E] = −γ for E ~ exponential(1), so
/// E[log τ] = −γ − ln λ for τ ~ exponential(λ).
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
/// Var[log τ] = π²/6 for τ ~ exponential(λ), whatever λ.
const VAR_LOG_EXP: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
/// |z| at which a Monte Carlo moment check fails. At 5 MCSE a correct sampler
/// fails with probability below 1e-6 per check, so a red here is a bias, not
/// bad luck; with these chain lengths it still resolves a bias of a few
/// percent of a standard deviation. Over ten disjoint seed sets of every
/// moment check below, the largest |z| seen was 2.9 (2.1 at the committed
/// seeds), consistent with a calibrated MCSE.
const Z_FAIL: f64 = 5.0;

fn ep(transform: Transform) -> EstimatedParam {
    EstimatedParam {
        name: "p".into(),
        index: 0,
        initial: 0.0,
        rw_sd: 0.1,
        transform,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }
}

// ── 1. Transforms, deterministically ────────────────────────────────────────

/// Step of the central finite differences. The functions differentiated are
/// smooth on the scale of 1e-5 everywhere they are sampled below (the nearest
/// point to a bound is 1e-3 inside it), so the truncation error is ~h² ≈ 1e-10
/// and rounding ~ε/h ≈ 1e-11, both far below the tolerances.
const FD_H: f64 = 1e-5;
/// Tolerance of a finite-difference comparison, relative to max(1, |value|).
const FD_TOL: f64 = 1e-7;

fn fd(f: &dyn Fn(f64) -> f64, z: f64) -> f64 {
    (f(z + FD_H) - f(z - FD_H)) / (2.0 * FD_H)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

/// One transform case: the parameter, and the z-points at which to check it
/// (interior points and points 1e-3 inside each finite bound).
fn transform_cases() -> Vec<(&'static str, EstimatedParam, Vec<f64>)> {
    let (lo, hi) = (1e-6_f64, 2.5_f64);
    vec![
        ("lower-bounded, [0, ∞)", ep(Transform::LowerBounded { lo: 0.0 }),
         vec![-30.0, -5.0, -1.0, 0.0, 1.5, 6.0]),
        ("log-interval, [1e-6, 2.5]", ep(Transform::for_domain(lo, hi, true).expect("domain")),
         vec![-8.0, -3.0, -0.5, 0.0, 0.5, 3.0, 8.0]),
        ("interval, [0, 1]", ep(Transform::Interval { lo: 0.0, hi: 1.0 }),
         vec![-8.0, -3.0, -0.4, 0.0, 0.7, 3.0, 8.0]),
        ("interval, [-3, 5]", ep(Transform::Interval { lo: -3.0, hi: 5.0 }),
         vec![-8.0, -2.0, 0.0, 1.3, 8.0]),
        ("identity", ep(Transform::Identity), vec![-40.0, -1.0, 0.0, 2.5, 40.0]),
    ]
}

#[test]
fn every_transform_round_trips() {
    for (label, p, zs) in transform_cases() {
        for &z in &zs {
            let x = p.from_transformed(z);
            let (lower, upper) = p.bounds();
            assert!(x >= lower && x <= upper, "{label}: θ({z}) = {x} left its bounds");
            let back = p.to_transformed(x);
            // The logit inverse loses ~ε/(p(1−p)) of z near its bounds: at
            // |z| = 12 that is ~1e-11, under the 1e-9 tolerance.
            assert!(close(back, z, 1e-9), "{label}: z → θ → z gave {z} → {x} → {back}");
            let again = p.from_transformed(back);
            assert!(close(again, x, 1e-12), "{label}: θ → z → θ gave {x} → {back} → {again}");
        }
    }
}

#[test]
fn every_transform_derivative_matches_finite_differences() {
    for (label, p, zs) in transform_cases() {
        for &z in &zs {
            // dθ/dz against the finite difference of the inverse map itself.
            let dtheta = fd(&|z| p.from_transformed(z), z);
            assert!(close(p.transform_deriv(z), dtheta, FD_TOL),
                "{label} @ z={z}: transform_deriv {} vs finite difference {dtheta}",
                p.transform_deriv(z));
            // log|dθ/dz|, against the log of that finite difference.
            assert!(close(p.log_jacobian(z), dtheta.abs().ln(), 1e-6),
                "{label} @ z={z}: log_jacobian {} vs ln|FD dθ/dz| {}",
                p.log_jacobian(z), dtheta.abs().ln());
            // d/dz log|dθ/dz|, against the finite difference of log_jacobian.
            let djac = fd(&|z| p.log_jacobian(z), z);
            assert!(close(p.jacobian_grad(z), djac, 1e-6),
                "{label} @ z={z}: jacobian_grad {} vs finite difference {djac}",
                p.jacobian_grad(z));
        }
    }
}

/// No map has a wall: the log-Jacobian and its gradient are finite at every
/// finite z, and θ stays in the closed domain, out to z = ±700 (θ = lo + eᶻ
/// overflows past z ≈ 709.8, which no posterior reaches). This replaces the
/// test that a boxed `Log` was −∞ outside its box.
#[test]
fn every_log_jacobian_is_finite_everywhere() {
    for (label, p, _) in transform_cases() {
        let (lower, upper) = p.bounds();
        for z in [-700.0, -40.0, -23.0, -1.0, 0.0, 1.0, 23.0, 40.0, 700.0] {
            assert!(p.log_jacobian(z).is_finite(), "{label}: log_jacobian({z}) = {}", p.log_jacobian(z));
            assert!(p.jacobian_grad(z).is_finite(), "{label}: jacobian_grad({z})");
            let x = p.from_transformed(z);
            assert!(x >= lower && x <= upper, "{label}: θ({z}) = {x} outside [{lower}, {upper}]");
        }
    }
}

// ── 2. The NUTS kernel alone ────────────────────────────────────────────────

/// A z-space target: (log density, gradient).
type Target<'a> = dyn Fn(&[f64]) -> (f64, Vec<f64>) + Sync + 'a;

/// The one-parameter z-scale target the PGAS NUTS closure assembles for an
/// exponential(λ) prior with no likelihood: prior + log-Jacobian, with the
/// gradient chained through `transform_deriv` (the `Exponential` arm of
/// `Prior::log_density_and_grad_z`, d log p/dθ = −λ).
fn exponential_target(p: EstimatedParam) -> impl Fn(&[f64]) -> (f64, Vec<f64>) + Sync {
    let prior = Prior::Fixed(Density::Exponential { rate: LAMBDA });
    move |z: &[f64]| {
        let value = prior.log_density(&p.prior_point(z[0])) + p.log_jacobian(z[0]);
        let grad = -LAMBDA * p.transform_deriv(z[0]) + p.jacobian_grad(z[0]);
        (value, vec![grad])
    }
}

struct KernelRun {
    /// Kept draws per chain, `chains[c][i][j]` = coordinate j of draw i.
    chains: Vec<Vec<Vec<f64>>>,
    /// Divergent transitions after adaptation, per chain.
    divergent: Vec<usize>,
}

const KERNEL_CHAINS: u64 = 4;
const KERNEL_WARMUP: usize = 1_000;
const KERNEL_KEPT: usize = 4_000;

/// The NUTS kernel PGAS uses, alone: `KERNEL_CHAINS` chains, seeds 1.., dual
/// averaging to acceptance 0.8 over `KERNEL_WARMUP` iterations, then
/// `KERNEL_KEPT` kept at the adapted step, diagonal mass `var`.
fn run_kernel(lp: &Target, start: &[f64], var: &[f64]) -> KernelRun {
    let mut chains = Vec::new();
    let mut divergent = Vec::new();
    for seed in 1..=KERNEL_CHAINS {
        let mut rng = StatefulRng::new(seed);
        let mut cfg = NUTSConfig {
            max_tree_depth: 10,
            step_size: 0.1,
            mass_matrix: MassMatrix::diagonal(var.to_vec()),
        };
        let mut da = DualAveraging::new(cfg.step_size, 0.8);
        let mut u = start.to_vec();
        let (mut l, mut g) = lp(&u);
        let (mut kept, mut div) = (Vec::with_capacity(KERNEL_KEPT), 0);
        for it in 0..KERNEL_WARMUP + KERNEL_KEPT {
            let r = nuts_step(&u, l, &g, &cfg, lp, &mut rng);
            assert!(r.log_posterior.is_finite(), "NUTS committed a non-finite log density");
            if it < KERNEL_WARMUP {
                cfg.step_size = if it + 1 == KERNEL_WARMUP {
                    da.final_step_size()
                } else {
                    da.update(r.mean_accept_prob)
                };
            } else {
                div += r.divergent as usize;
            }
            u = r.params;
            (l, g) = lp(&u);
            if it >= KERNEL_WARMUP {
                kept.push(u.clone());
            }
        }
        chains.push(kept);
        divergent.push(div);
    }
    KernelRun { chains, divergent }
}

/// Coordinate `j` of every chain, as `rank_convergence` takes it.
fn coord(chains: &[Vec<Vec<f64>>], j: usize, f: &dyn Fn(f64) -> f64) -> Vec<Vec<f64>> {
    chains.iter().map(|c| c.iter().map(|v| f(v[j])).collect()).collect()
}

/// z-scores of the pooled mean and variance of coordinate `j` against the
/// exact `mean` and `var`. Each MCSE is sd/√ESS of the statistic's summand,
/// with ESS the rank-normalized bulk ESS (Vehtari et al. 2021) of that summand
/// — x for the mean, (x − mean)² for the variance — so autocorrelation is
/// accounted for.
fn moment_z(chains: &[Vec<Vec<f64>>], j: usize, mean: f64, var: f64) -> (f64, f64) {
    let x: Vec<f64> = chains.iter().flat_map(|c| c.iter().map(|v| v[j])).collect();
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let ess_x = rank_convergence(&coord(chains, j, &|v| v)).unwrap().ess_bulk;
    let z_mean = (m - mean) / (var / ess_x).sqrt();
    let sq: Vec<f64> = x.iter().map(|v| (v - mean).powi(2)).collect();
    let v = sq.iter().sum::<f64>() / n;
    let sd_sq = (sq.iter().map(|s| (s - v).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let ess_sq = rank_convergence(&coord(chains, j, &|u| (u - mean).powi(2))).unwrap().ess_bulk;
    let z_var = (v - var) / (sd_sq / ess_sq.sqrt());
    (z_mean, z_var)
}

/// An independent Gaussian with scales from 0.01 to 1 (the level, slope and
/// modes of a smoothing spline's prior), sampled with the true variances as
/// the mass matrix: the easiest target NUTS has. A divergence here is a
/// kernel defect, not a geometry problem.
#[test]
fn nuts_on_an_independent_gaussian_has_no_divergences_and_the_right_moments() {
    let mean = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let sd = [0.6, 0.01, 1.0, 1.0, 1.0, 1.0];
    let var: Vec<f64> = sd.iter().map(|s| s * s).collect();
    let gauss = |u: &[f64]| -> (f64, Vec<f64>) {
        let l = u.iter().zip(mean.iter().zip(&sd)).map(|(x, (m, s))| -0.5 * ((x - m) / s).powi(2)).sum();
        (l, u.iter().zip(mean.iter().zip(&sd)).map(|(x, (m, s))| -(x - m) / (s * s)).collect())
    };
    let run = run_kernel(&gauss, &mean, &var);
    // Measured: 0 in every chain, adapted and at fixed steps 0.5, 1.0 and
    // 1.5. With the mass equal to the covariance the energy error at the
    // adapted step stays below ~3, far from the divergence threshold of
    // 1000, so any divergence is a defect. Zero is the threshold.
    assert_eq!(run.divergent, vec![0; KERNEL_CHAINS as usize], "divergences on a Gaussian");
    for j in 0..mean.len() {
        let (zm, zv) = moment_z(&run.chains, j, mean[j], var[j]);
        eprintln!("gaussian coord {j}: z(mean) {zm:+.2}, z(var) {zv:+.2}");
        assert!(zm.abs() < Z_FAIL && zv.abs() < Z_FAIL,
            "coord {j}: mean z {zm:.2}, variance z {zv:.2} (|z| must be < {Z_FAIL})");
    }
}

/// log τ for τ ~ exponential(λ), no box: the target u − λeᵘ is smooth but
/// stiff in its right tail (the curvature λeᵘ grows without bound), so the
/// odd trajectory overshoots there. Divergences must stay rare, and the
/// moments of log τ must be exact.
#[test]
fn nuts_on_unboxed_log_tau_rarely_diverges_and_has_the_right_moments() {
    let lp = exponential_target(ep(Transform::LowerBounded { lo: 0.0 }));
    let run = run_kernel(&lp, &[-2.0], &[VAR_LOG_EXP]);
    let total: usize = run.divergent.iter().sum();
    eprintln!("unboxed log τ: divergences per chain {:?}", run.divergent);
    // Bound: at most MAX_UNBOXED_DIVERGENT of the 16,000 kept transitions.
    // See the constant for the measured value and margin.
    assert!(total <= MAX_UNBOXED_DIVERGENT,
        "unboxed log τ: {total} divergences (per chain {:?}), bound {MAX_UNBOXED_DIVERGENT}",
        run.divergent);
    let (zm, zv) = moment_z(&run.chains, 0, -EULER_GAMMA - LAMBDA.ln(), VAR_LOG_EXP);
    eprintln!("unboxed log τ: z(mean) {zm:+.2}, z(var) {zv:+.2}");
    assert!(zm.abs() < Z_FAIL && zv.abs() < Z_FAIL,
        "log τ: mean z {zm:.2}, variance z {zv:.2} (|z| must be < {Z_FAIL})");
}

/// Upper bound on the divergent transitions, over 4 chains × 4,000 kept, of
/// the NUTS kernel on unboxed log τ. Measured at seeds 1–4: 2 (per chain
/// 0, 2, 0, 0). Over ten disjoint seed sets (1–4, 11–14, …, 91–94) the total
/// ranged 2–13, so the bound sits 3× above the worst seen, and still 5×
/// below the fewest the walled boxed `Log` map produced before
/// support-keyed transforms (208), which keeps "rare overshoot in a stiff
/// tail" apart from "a wall".
const MAX_UNBOXED_DIVERGENT: usize = 40;

// ── 4. A boxed parameter has no wall ─────────────────────────────────────

/// log τ under exponential(λ) boxed to [1e-6, 2.5]. Under the walled `Log`
/// map this diverged routinely (271 over 4 × 4,000 at seeds 1–4, 208–288
/// over ten seed sets): the stiff right tail carried trajectories into the
/// −∞ log-Jacobian past ln 2.5. Under `LogInterval` the end is z = +∞ and the
/// tail is the log-Jacobian's, so the boxed run must diverge no more than the
/// unboxed control's bound, and its draws must be the exponential truncated
/// to the box.
#[test]
fn boxed_parameter_diverges_no_more_than_the_unboxed_one() {
    let (lo, hi) = (1e-6_f64, 2.5_f64);
    let p = ep(Transform::for_domain(lo, hi, true).expect("domain"));
    assert!(matches!(p.transform, Transform::LogInterval { .. }));
    let lp = exponential_target(p.clone());
    // A unit mass: dual averaging tunes the step, and the target's
    // z-variance is within an order of magnitude of 1.
    let run = run_kernel(&lp, &[0.0], &[1.0]);
    let total: usize = run.divergent.iter().sum();
    eprintln!("boxed log τ (LogInterval): divergences per chain {:?}", run.divergent);
    assert!(total <= MAX_UNBOXED_DIVERGENT,
        "boxed τ: {total} divergences (per chain {:?}), above the unboxed bound \
         {MAX_UNBOXED_DIVERGENT}", run.divergent);
    let theta: Vec<Vec<Vec<f64>>> = run.chains.iter()
        .map(|c| c.iter().map(|v| vec![p.from_transformed(v[0])]).collect()).collect();
    let prior = Prior::Fixed(Density::Exponential { rate: LAMBDA });
    let law = Truncated::new(&prior, lo, hi);
    law.check("kernel, boxed τ", &theta, 0);
}

// ── 3. PGAS + NUTS with no likelihood ───────────────────────────────────────

const TAU_IDX: usize = 0;
const Z_IDX: usize = 1;
const KAPPA_IDX: usize = 2;
const PRIOR_T_END: f64 = 10.0;

/// X → D at rate κ·X, X₀ = 0, with the estimated τ and z appearing nowhere —
/// not in the rate, not in the observation model. The fixed κ carries a
/// `rate_grad` only so `pgas::nuts_active` routes the θ-step through NUTS.
/// The complete-data likelihood is then constant in (τ, z), and PGAS+NUTS
/// must draw the prior exactly.
fn prior_only_model() -> Arc<CompiledModel> {
    let mut rate_grad = ir::deriv::ParamGradMap::new();
    rate_grad.insert("kappa".into(), DerivEntry::Grad(Expr::pop("X")));
    let fixed = |name: &str, value: f64| Parameter {
        name: name.into(),
        value: ir::parameter::ParamValue::Fixed { value },
        param_kind: None,
        param_dim: None,
    };
    let model = Model {
        constants: vec![],
        ic_grad: Default::default(),
        name: "nuts_validity_prior_only".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![
            Compartment { name: "X".into(), kind: CompartmentKind::Integer },
            Compartment { name: "D".into(), kind: CompartmentKind::Integer },
        ],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "decay".into(),
            stoichiometry: vec![StoichiometryEntry("X".into(), -1), StoichiometryEntry("D".into(), 1)],
            rate: Expr::bin_op(ir::expr::BinOp::Mul, Expr::param("kappa"), Expr::pop("X")),
            metadata: None,
            draw_method: DrawMethod::Poisson,
            rate_grad,
            lineage: None,
        }],
        ode_equations: vec![],
        time_functions: vec![],
        tables: vec![],
        interventions: vec![],
        observations: vec![],
        bindings: vec![],
        per_eval_bindings: vec![],
        parameters: vec![fixed("tau", 0.2), fixed("z", 0.0), fixed("kappa", 0.1)],
        initial_conditions: InitialConditions(
            [("X".to_string(), InitSpec::Deterministic(Expr::const_(0.0))),
             ("D".to_string(), InitSpec::Deterministic(Expr::const_(0.0)))].into_iter().collect(),
        ),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![PRIOR_T_END]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: PRIOR_T_END,
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
    let m = Arc::new(CompiledModel::new(model).expect("prior-only model must compile"));
    assert_eq!((m.param_index["tau"], m.param_index["z"], m.param_index["kappa"]),
        (TAU_IDX, Z_IDX, KAPPA_IDX));
    m
}

/// One Poisson stream on the decay incidence with every value missing.
fn all_missing_obs(model: &Arc<CompiledModel>, times: &[f64]) -> MultiStreamObsModel {
    let projection = StreamProjection::FlowSum(vec![0]);
    let spec = StreamSpec {
        times: StreamTimes::contiguous_for(&projection, 0.0, times.to_vec()).unwrap(),
        projection,
        ir_model: ir::observation::ObservationModel {
            name: "removals".into(),
            source: "removals".into(),
            columns: vec![
                ir::observation::ObsColumn { name: "time".into(), role: ir::observation::ColumnRole::Time },
                ir::observation::ObsColumn {
                    name: "removals".into(),
                    role: ir::observation::ColumnRole::Value(ir::parameter::ParamKind::Count),
                },
            ],
            scored: "removals".into(),
            emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
            stratum: vec![],
            covers: None,
            projection: ir::observation::Projection::CumulativeFlow("decay".into()),
            projection_state_grad: Default::default(),
            likelihood: ir::observation::Likelihood::Poisson(PoissonLikelihood {
                rate: Diffable::new(Expr::bin_op(
                    ir::expr::BinOp::Add,
                    Expr::Projected(ir::expr::ProjectedExpr { projected: () }),
                    Expr::const_(0.1),
                )),
            }),
        },
        observations: vec![None; times.len()],
        aux: vec![],
    };
    MultiStreamObsModel::new(BoundObs::bind(0.0, vec![spec]).unwrap().0, model.clone()).unwrap()
}

const PGAS_CHAINS: u64 = 4;
const PGAS_BURN: usize = 500;
const PGAS_KEPT: usize = 2_000;

/// Kolmogorov–Smirnov distance between the pooled draws `x` and the
/// exponential(λ) CDF 1 − e^{−λx}.
fn ks_exponential(mut x: Vec<f64>, lambda: f64) -> f64 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = x.len() as f64;
    x.iter().enumerate().fold(0.0_f64, |d, (i, &v)| {
        let f = 1.0 - (-lambda * v).exp();
        d.max((f - i as f64 / n).abs()).max(((i + 1) as f64 / n - f).abs())
    })
}

/// PGAS + NUTS on a model with no likelihood for its parameters draws the
/// prior: τ ~ exponential(5.99) through the unbounded `Log` transform, and
/// z ~ N(0, 1) through `None`. This is the whole θ-step as a fit runs it —
/// transform, prior, Jacobian, gradient, NUTS, dense-mass and step-size
/// adaptation — composed with the CSMC x-step.
#[test]
fn pgas_nuts_without_a_likelihood_draws_the_prior() {
    let model = prior_only_model();
    let inf = f64::INFINITY;
    let mut tau = ep(Transform::for_domain(0.0, inf, true).expect("domain"));
    tau.name = "tau".into();
    tau.index = TAU_IDX;
    tau.initial = 0.2;
    let mut z = ep(Transform::Identity);
    z.name = "z".into();
    z.index = Z_IDX;
    let est = vec![tau, z];
    let priors = vec![
        Prior::Fixed(Density::Exponential { rate: LAMBDA }),
        Prior::Fixed(Density::Normal { mean: 0.0, sd: 1.0 }),
    ];
    let times: Vec<f64> = (1..=PRIOR_T_END as usize).map(|t| t as f64).collect();
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let obs = all_missing_obs(&model, &times);
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 4,
        n_sweeps: PGAS_BURN + PGAS_KEPT,
        burn_in: PGAS_BURN,
        thin: 1,
        dt: 1.0,
        use_nuts: true,
        dense_mass: true,
        max_tree_depth: 8,
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };
    let mut chains: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut divergent = Vec::new();
    for seed in 1..=PGAS_CHAINS {
        let mut base = model.default_params.clone();
        for p in &est {
            base[p.index] = p.initial;
        }
        let r = run_pgas(&model, &est, &priors, &base, &config, &observations, &obs, seed, None, None,
            "nuts-validity-prior-only".into()).expect("PGAS run");
        assert!(r.sweeps.iter().any(|s| s.nuts.n_leapfrog > 0), "the θ-step did not run NUTS");
        chains.push(r.sweeps.iter().map(|s| vec![s.params[TAU_IDX], s.params[Z_IDX]]).collect());
        divergent.push(r.n_divergent_post_burn);
    }
    assert!(chains.iter().all(|c| c.len() == PGAS_KEPT), "expected {PGAS_KEPT} kept sweeps per chain");
    eprintln!("prior-only PGAS: divergences per chain {divergent:?}");
    let total: usize = divergent.iter().sum();
    assert!(total <= MAX_PRIOR_ONLY_DIVERGENT,
        "prior-only PGAS: {total} divergences (per chain {divergent:?}), bound {MAX_PRIOR_ONLY_DIVERGENT}");

    // τ against exponential(λ). The KS distance of n_eff independent draws
    // exceeds 1.63/√n_eff with probability 0.01 (the asymptotic Kolmogorov
    // quantile); n_eff is the bulk ESS of τ, the usual correction for
    // autocorrelated draws. The bound is 2.0/√n_eff (p ≈ 6e-4). Measured at
    // seeds 1–4: D = 0.0095 against a bound of 0.0354 (bulk ESS 3192); over
    // ten disjoint seed sets D never exceeded half its bound.
    let tau_draws: Vec<f64> = chains.iter().flat_map(|c| c.iter().map(|v| v[0])).collect();
    let ess_tau = rank_convergence(&coord(&chains, 0, &|v| v)).unwrap().ess_bulk;
    let d = ks_exponential(tau_draws, LAMBDA);
    let d_crit = KS_C / ess_tau.sqrt();
    eprintln!("prior-only PGAS: τ KS distance {d:.4}, bound {d_crit:.4} (bulk ESS {ess_tau:.0})");
    assert!(d < d_crit, "τ draws are not exponential({LAMBDA}): KS {d:.4} ≥ {d_crit:.4}");
    // And log τ's moments, as for the kernel alone.
    let log_chains: Vec<Vec<Vec<f64>>> =
        chains.iter().map(|c| c.iter().map(|v| vec![v[0].ln()]).collect()).collect();
    let (zm, zv) = moment_z(&log_chains, 0, -EULER_GAMMA - LAMBDA.ln(), VAR_LOG_EXP);
    eprintln!("prior-only PGAS: log τ z(mean) {zm:+.2}, z(var) {zv:+.2}");
    assert!(zm.abs() < Z_FAIL && zv.abs() < Z_FAIL, "log τ: mean z {zm:.2}, variance z {zv:.2}");

    // z against N(0, 1).
    let (zm, zv) = moment_z(&chains, 1, 0.0, 1.0);
    eprintln!("prior-only PGAS: z z(mean) {zm:+.2}, z(var) {zv:+.2}");
    assert!(zm.abs() < Z_FAIL && zv.abs() < Z_FAIL, "z: mean z {zm:.2}, variance z {zv:.2}");
}

/// The Kolmogorov–Smirnov constant: P(√n·D > 2.0) ≈ 6e-4 for n i.i.d. draws.
const KS_C: f64 = 2.0;

/// Upper bound on the post-adaptation divergences of the prior-only PGAS
/// run, over 4 chains × 2,000 kept. Measured at seeds 1–4: 0; over ten
/// disjoint seed sets, at most 1. Without a box neither coordinate has a
/// wall, so the bound only has to separate that from the boxed regime the
/// branch measured through PGAS (21–51 per chain at 4,000 kept).
const MAX_PRIOR_ONLY_DIVERGENT: usize = 10;

// ── 5. Prior-only runs recover the truncated prior (proposal §12.3) ─────────

/// |z| of a moment against the truncated law, the §12.3 pass bar (4 MCSE).
const Z_TRUNC: f64 = 4.0;
/// Kolmogorov critical value √n·D at p = 0.001 (asymptotic): 1.95.
const KS_P001: f64 = 1.95;

/// A prior truncated to [lo, hi], by quadrature on 200,000 midpoints of
/// [lo, min(hi, lo + 40)]: its mean, variance and CDF.
struct Truncated {
    grid: Vec<f64>,
    cdf: Vec<f64>,
    mean: f64,
    var: f64,
}

impl Truncated {
    fn new(prior: &Prior, lo: f64, hi: f64) -> Self {
        let top = hi.min(lo + 40.0);
        let n = 200_000;
        let h = (top - lo) / n as f64;
        let grid: Vec<f64> = (0..n).map(|i| lo + (i as f64 + 0.5) * h).collect();
        let dens: Vec<f64> = grid.iter()
            .map(|&x| prior.log_density(&sim::inference::prior::PriorPoint::natural(x)).exp())
            .collect();
        let z: f64 = dens.iter().sum::<f64>() * h;
        let mean = grid.iter().zip(&dens).map(|(x, d)| x * d).sum::<f64>() * h / z;
        let var = grid.iter().zip(&dens).map(|(x, d)| (x - mean).powi(2) * d).sum::<f64>() * h / z;
        let mut acc = 0.0;
        let cdf = dens.iter().map(|d| { acc += d * h / z; acc }).collect();
        Truncated { grid, cdf, mean, var }
    }

    fn cdf_at(&self, x: f64) -> f64 {
        match self.grid.partition_point(|g| *g <= x) {
            0 => 0.0,
            i => self.cdf[i - 1],
        }
    }

    /// The §12.3 pass: mean and variance of coordinate `j` within `Z_TRUNC`
    /// MCSE of the truncated law's, and a KS test on the pooled draws thinned
    /// to about the bulk ESS with p > 0.001.
    fn check(&self, label: &str, chains: &[Vec<Vec<f64>>], j: usize) {
        let (zm, zv) = moment_z(chains, j, self.mean, self.var);
        let pooled: Vec<f64> = chains.iter().flat_map(|c| c.iter().map(|v| v[j])).collect();
        let ess = rank_convergence(&coord(chains, j, &|v| v)).unwrap().ess_bulk;
        let stride = ((pooled.len() as f64 / ess).ceil() as usize).max(1);
        let mut thin: Vec<f64> = pooled.iter().step_by(stride).copied().collect();
        thin.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = thin.len() as f64;
        let d = thin.iter().enumerate().fold(0.0_f64, |d, (i, &v)| {
            let f = self.cdf_at(v);
            d.max((f - i as f64 / n).abs()).max(((i + 1) as f64 / n - f).abs())
        });
        eprintln!("{label}: z(mean) {zm:+.2}, z(var) {zv:+.2}, KS √n·D {:.3} (n = {n}, bulk ESS {ess:.0})",
            n.sqrt() * d);
        assert!(zm.abs() < Z_TRUNC && zv.abs() < Z_TRUNC,
            "{label}: mean z {zm:.2}, variance z {zv:.2} against the truncated law (|z| < {Z_TRUNC})");
        assert!(n.sqrt() * d < KS_P001, "{label}: KS √n·D = {:.3} ≥ {KS_P001}", n.sqrt() * d);
    }
}

/// The three §12.3 parameters as `(name, index, transform, prior, lo, hi)`,
/// on the prior-only model's slots τ, z and κ.
fn truncated_cases() -> Vec<(&'static str, usize, Transform, Prior, f64, f64)> {
    use ir::parameter::{BetaPrior, HalfNormalPrior, PriorDist};
    let inf = f64::INFINITY;
    vec![
        ("tau", TAU_IDX, Transform::for_domain(1e-6, 2.5, true).unwrap(),
         Prior::Fixed(Density::Exponential { rate: LAMBDA }), 1e-6, 2.5),
        ("z", Z_IDX, Transform::for_domain(0.1, 0.9, true).unwrap(),
         Prior::from_ir(&PriorDist::Beta(BetaPrior { alpha: 2.0, beta: 5.0 })), 0.1, 0.9),
        ("kappa", KAPPA_IDX, Transform::for_domain(0.0, inf, true).unwrap(),
         Prior::from_ir(&PriorDist::HalfNormal(HalfNormalPrior { sigma: 1.0 })), 0.0, inf),
    ]
}

fn truncated_params() -> (Vec<EstimatedParam>, Vec<Prior>, Vec<(f64, f64)>) {
    let cases = truncated_cases();
    assert!(matches!(cases[0].2, Transform::LogInterval { .. }));
    assert!(matches!(cases[1].2, Transform::LogInterval { .. }));
    assert!(matches!(cases[2].2, Transform::LowerBounded { lo } if lo == 0.0));
    let est = cases.iter().map(|(name, idx, t, _, lo, hi)| {
        let mut p = ep(*t);
        p.name = (*name).into();
        p.index = *idx;
        p.initial = if hi.is_finite() { (lo * hi).sqrt() } else { 0.5 };
        p
    }).collect();
    let priors = cases.iter().map(|c| c.3.clone()).collect();
    let boxes = cases.iter().map(|c| (c.4, c.5)).collect();
    (est, priors, boxes)
}

/// PGAS+NUTS, 4 chains × 4,000 kept, on the prior-only model: each of τ, ρ
/// and κ must be its prior truncated to its box, and the boxed parameters
/// must not diverge more than the unboxed kernel control's bound.
#[test]
fn pgas_nuts_without_a_likelihood_draws_the_truncated_priors() {
    let model = prior_only_model();
    let (est, priors, boxes) = truncated_params();
    let times: Vec<f64> = (1..=PRIOR_T_END as usize).map(|t| t as f64).collect();
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let obs = all_missing_obs(&model, &times);
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 4,
        n_sweeps: PGAS_BURN + 4_000,
        burn_in: PGAS_BURN,
        thin: 1,
        dt: 1.0,
        use_nuts: true,
        dense_mass: true,
        max_tree_depth: 8,
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };
    let mut chains: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut divergent = Vec::new();
    for seed in 1..=PGAS_CHAINS {
        let mut base = model.default_params.clone();
        for p in &est {
            base[p.index] = p.initial;
        }
        let r = run_pgas(&model, &est, &priors, &base, &config, &observations, &obs, seed, None, None,
            "nuts-validity-truncated".into()).expect("PGAS run");
        assert!(r.sweeps.iter().any(|s| s.nuts.n_leapfrog > 0), "the θ-step did not run NUTS");
        chains.push(r.sweeps.iter().map(|s| est.iter().map(|p| s.params[p.index]).collect()).collect());
        divergent.push(r.n_divergent_post_burn);
    }
    let total: usize = divergent.iter().sum();
    eprintln!("truncated PGAS+NUTS: divergences per chain {divergent:?}");
    assert!(total <= MAX_UNBOXED_DIVERGENT,
        "truncated PGAS+NUTS: {total} divergences (per chain {divergent:?}), bound {MAX_UNBOXED_DIVERGENT}");
    for (j, (p, (lo, hi))) in est.iter().zip(&boxes).enumerate() {
        Truncated::new(&priors[j], *lo, *hi).check(&format!("PGAS+NUTS {}", p.name), &chains, j);
    }
}

/// The same three parameters under PMMH with a zero log-likelihood, 4 chains
/// × 4,000 kept. ODE-MH runs this kernel (`run_pmmh`) with a deterministic
/// likelihood, so with none the two are the same test. It exercises the MH
/// ratio's Jacobian, which the NUTS run does not: acceptance must lie in
/// [0.1, 0.6] and the draws must be the truncated priors.
#[test]
fn pmmh_without_a_likelihood_draws_the_truncated_priors() {
    use sim::inference::pmmh::{run_pmmh, PMMHConfig};
    let model = prior_only_model();
    let (est, priors, boxes) = truncated_params();
    let config = PMMHConfig {
        n_steps: 1_000 + 4_000,
        n_particles: 1,
        dt: 1.0,
        proposal_sd: vec![0.5; est.len()],
        adapt: true,
        adapt_start: 200,
        adapt_stop: 1_000,
        thin: 1,
        burn_in: 1_000,
    };
    let zero = |_: &[f64], _: u64| -> Result<f64, sim::error::SimError> { Ok(0.0) };
    let mut chains: Vec<Vec<Vec<f64>>> = Vec::new();
    for seed in 1..=PGAS_CHAINS {
        let mut base = model.default_params.clone();
        for p in &est {
            base[p.index] = p.initial;
        }
        let r = run_pmmh(&est, &priors, &base, &[], &config, &zero, seed, None, None,
            "nuts-validity-pmmh".into()).expect("PMMH run");
        let acc = r.steps.iter().filter(|s| s.accepted).count() as f64 / r.steps.len() as f64;
        eprintln!("truncated PMMH chain {seed}: acceptance {acc:.3}");
        assert!((0.1..=0.6).contains(&acc), "chain {seed}: acceptance {acc:.3} outside [0.1, 0.6]");
        chains.push(r.steps.iter().map(|s| est.iter().map(|p| s.params[p.index]).collect()).collect());
    }
    for (j, (p, (lo, hi))) in est.iter().zip(&boxes).enumerate() {
        Truncated::new(&priors[j], *lo, *hi).check(&format!("PMMH {}", p.name), &chains, j);
    }
}
