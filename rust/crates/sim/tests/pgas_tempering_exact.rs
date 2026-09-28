//! gh#551: does tempered PGAS sample the right distribution?
//!
//! The incident's §7.3 test
//! (`docs/dev/incidents/2026-08-10-pgas-tempering-swap-sign.md`). The other
//! tempering tests assert that the machinery runs — swap rates in `[0, 1]`,
//! finite log-likelihoods. A sampler can do all of that while sampling the
//! wrong distribution. These assert agreement with a distribution computed
//! exactly, so they test the composed kernel (θ-move ∘ x-move ∘ swap) rather
//! than any one factor.
//!
//! # The target
//!
//! Rung `k` of the ladder targets observation-only tempering:
//!
//! ```text
//!   π_k(θ, x) ∝ p(θ) · p(x | θ) · p(y | x, θ)^{β_k}
//! ```
//!
//! prior and process density at full weight, observation density raised to
//! `β_k`. Its θ-marginal is `π_k(θ) ∝ p(θ) · Σ_x p(x | θ) · p(y | x, θ)^{β_k}`,
//! and at `β = 1` that is the ordinary posterior. The replica-exchange swap is
//! a Metropolis move on the product `Π_k π_k`, so the cold rung's marginal
//! must be the `β = 1` posterior **whatever the ladder** — tempering may change
//! how fast the chain mixes, never what it converges to.
//!
//! # Why it is exactly computable
//!
//! A pure-death chain-binomial, `N → ∅` at per-capita rate `μ`, over
//! [`N_SUBSTEPS`] unit substeps from `N₀ =` [`N0`]. Its only noise is the
//! per-substep death count `d_t ~ Binomial(n_{t−1}, 1 − e^{−μ})`, so a latent
//! path is its death sequence `(d_1, …, d_T)` with `Σ d_t ≤ N₀` — 1001 paths,
//! enumerated outright. Each substep's deaths are observed as
//! `y_t ~ Poisson(μ·d_t + OBS_FLOOR)`.
//!
//! The observation rate depends on `μ` deliberately. If it did not, the
//! θ-move's conditional `p(θ | x, y)` would not involve the observation term
//! at all, and a θ-move that tempered it wrongly would be indistinguishable
//! from one that tempered it right. With `μ` in both the process and the
//! observation density, each of the three moves (θ, x, swap) weights the two
//! differently under a wrong target.
//!
//! `μ` has a flat prior on its declared box `[MU_LO, MU_HI]` (the `Log`
//! transform's support), so `π_k(μ)` is the enumerated sum restricted to the
//! box. Its functionals are one-dimensional integrals, done by composite
//! Simpson on [`QUAD_POINTS`] nodes. The oracle codes the binomial and Poisson
//! pmfs from their definitions rather than calling camdl's densities, so
//! agreement is evidence about the sampler, not two calls to one function.
//!
//! # Tolerance
//!
//! Each assertion compares a posterior functional estimated from one chain
//! against its exact value, in units of that chain's Monte Carlo standard
//! error, estimated by non-overlapping batch means over [`N_BATCHES`] batches.
//! A batch spans `(N_SWEEPS − BURN_IN) / N_BATCHES` = 300 retained sweeps.
//! The chain's integrated autocorrelation time for `E[μ]`, measured as
//! (batch-means variance / i.i.d. variance) and printed by every case, is 2–5
//! sweeps, so a batch is 60–150 autocorrelation times long, the batch means
//! are close to independent, and the SE estimate is honest. Under the null the
//! standardized error is then a `t` with 39 degrees of freedom, and the bar is
//! [`Z_BAR`] = 4.5: a two-sided per-assertion false-alarm probability of
//! `6e-5` under `t_39`. The four sampler cases make eight such comparisons, so
//! a correct sampler fails the file with probability `< 5e-4` for a fresh
//! seed. The chains are seeded, so each build is deterministic; the bar says
//! how unlikely a correct sampler is to have landed outside it. It is not a
//! knob: a correct build that fails it needs more sweeps, never a wider bar.
//!
//! # What the ladder cases can and cannot see
//!
//! `run_pgas` reports only the cold rung, so these cases see a hot rung's
//! kernel only through its effect on the cold chain via swaps. They are
//! strongly sensitive to a wrong swap ratio and to a wrong θ-move target on
//! the hot rungs, both of which move the cold marginal by many standard
//! errors. They are weakly sensitive to a hot rung whose trajectory move
//! ignores β — that error shifts the cold marginal by about three standard
//! errors at these sample sizes, under the bar — so the x-move is pinned
//! directly and exactly instead, by
//! `csmc_exact_invariance::one_sweep_leaves_the_tempered_smoothing_target_invariant`.
//! The NUTS energy and gradient are pinned directly by
//! `gradient_check_obs::gh551_tempered_energy_and_gradient_match_fd`.

use std::collections::HashMap;
use std::sync::Arc;

use ir::{
    deriv::{DerivEntry, ParamGradMap},
    expr::{BinOp, BinOpExpr, BinOpWrap, ConstExpr, Expr, ParamExpr, PopExpr, ProjectedExpr},
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
        dense_cells,
        if2::{EstimatedParam, Transform},
        multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes},
        particle_filter::Observation,
        pgas::{run_pgas, PGASConfig},
        pmmh::Prior,
        BoundObs, MultiStreamObsModel,
    },
};

const N0: u64 = 10;
const N_SUBSTEPS: usize = 4;
/// One observation per unit substep, of that substep's deaths. A large count
/// next to zeros puts the data in tension with the binomial process, so the
/// process and observation densities pull `μ` in different directions and a
/// target that weights them wrongly lands visibly elsewhere.
const Y: [f64; N_SUBSTEPS] = [5.0, 0.0, 3.0, 0.0];
/// `y_t ~ Poisson(μ·d_t + OBS_FLOOR)`: keeps a zero-death substep scoreable.
const OBS_FLOOR: f64 = 0.1;
const MU_LO: f64 = 0.02;
const MU_HI: f64 = 3.0;
const MU_START: f64 = 0.3;
/// The fixed cut for the tail-probability functional `P(μ < MU_CUT)`.
const MU_CUT: f64 = 0.5;

/// Composite-Simpson nodes over `[MU_LO, MU_HI]` (odd). The integrand is
/// smooth on the box, so the quadrature error on a functional is `O(h⁴)`,
/// many orders below the Monte Carlo SE; the step discontinuity of the
/// tail-probability integrand at `MU_CUT` costs at most one node weight,
/// `O(h) ≈ 1e-3` of the local density — still two orders below its SE.
const QUAD_POINTS: usize = 3001;
const N_SWEEPS: usize = 14_000;
const BURN_IN: usize = 2_000;
const N_BATCHES: usize = 40;
const Z_BAR: f64 = 4.5;

// ── model ────────────────────────────────────────────────────────────

fn mu() -> Expr {
    Expr::Param(ParamExpr { param: "mu".into() })
}

fn pure_death_model() -> (Arc<CompiledModel>, Vec<f64>) {
    // `rate_grad` is carried so the NUTS case has its gradient: d(μN)/dμ = N.
    let mut rate_grad = ParamGradMap::new();
    rate_grad.insert("mu".into(), DerivEntry::Grad(Expr::Pop(PopExpr { pop: "N".into() })));
    let model = Model {
        ic_grad: Default::default(),
        name: "pure_death_tempering_exact".into(),
        version: "0.3".into(),
        time_unit: "days".into(),
        description: None,
        origin: None,
        origin_rata_die: None,
        compartments: vec![Compartment { name: "N".into(), kind: CompartmentKind::Integer }],
        transitions: vec![Transition {
            rate_state_grad: Default::default(),
            name: "death".into(),
            stoichiometry: vec![StoichiometryEntry("N".into(), -1)],
            rate: Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr {
                    op: BinOp::Mul,
                    left: Box::new(mu()),
                    right: Box::new(Expr::Pop(PopExpr { pop: "N".into() })),
                },
            }),
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
        parameters: vec![Parameter {
            name: "mu".into(),
            value: ir::parameter::ParamValue::Fixed { value: MU_START },
            param_kind: None,
            param_dim: None,
        }],
        initial_conditions: InitialConditions::constants({
            let mut m = HashMap::new();
            m.insert("N".into(), N0 as f64);
            m
        }),
        output: OutputConfig {
            times: OutputSchedule::AtTimes(vec![0.0, N_SUBSTEPS as f64]),
            format: "tsv".into(),
            trajectory: true,
            observations: false,
        },
        simulation: SimulationConfig {
            t_start: 0.0,
            t_end: N_SUBSTEPS as f64,
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
    let compiled = Arc::new(CompiledModel::new(model).unwrap());
    let params = compiled.default_params.clone();
    (compiled, params)
}

fn observations() -> Vec<Observation> {
    Y.iter()
        .enumerate()
        .map(|(k, &v)| Observation { time: (k + 1) as f64, value: v })
        .collect()
}

/// `y_t ~ Poisson(μ · deaths_t + OBS_FLOOR)`, with `∂rate/∂μ = deaths_t`.
fn obs_model(compiled: &Arc<CompiledModel>) -> MultiStreamObsModel {
    let obs = observations();
    let projection = StreamProjection::FlowSum(vec![0]);
    let projected = || Expr::Projected(ProjectedExpr { projected: () });
    let mut rate = ir::Diffable::new(Expr::BinOp(BinOpWrap {
        bin_op: BinOpExpr {
            op: BinOp::Add,
            left: Box::new(Expr::BinOp(BinOpWrap {
                bin_op: BinOpExpr { op: BinOp::Mul, left: Box::new(mu()), right: Box::new(projected()) },
            })),
            right: Box::new(Expr::Const(ConstExpr { value: OBS_FLOOR })),
        },
    }));
    rate.grad.insert("mu".into(), DerivEntry::Grad(projected()));
    MultiStreamObsModel::new(
        BoundObs::bind(
            0.0,
            vec![StreamSpec {
                projection: projection.clone(),
                ir_model: ir::observation::ObservationModel {
                    name: "deaths".into(),
                    source: "deaths".into(),
                    columns: vec![
                        ir::observation::ObsColumn {
                            name: "time".into(),
                            role: ir::observation::ColumnRole::Time,
                        },
                        ir::observation::ObsColumn {
                            name: "deaths".into(),
                            role: ir::observation::ColumnRole::Value(
                                ir::parameter::ParamKind::Count,
                            ),
                        },
                    ],
                    scored: "deaths".into(),
                    emit_schedule: Some(ir::observation::ObservationSchedule::AtTimes(vec![])),
                    stratum: vec![],
                    covers: None,
                    projection: ir::observation::Projection::CumulativeFlow("death".into()),
                    projection_state_grad: Default::default(),
                    likelihood: ir::observation::Likelihood::Poisson(
                        ir::observation::PoissonLikelihood { rate },
                    ),
                },
                observations: dense_cells(obs.iter().map(|o| o.value).collect()),
                times: StreamTimes::contiguous_for(
                    &projection,
                    0.0,
                    obs.iter().map(|o| o.time).collect(),
                )
                .unwrap(),
                aux: vec![],
            }],
        )
        .unwrap()
        .0,
        compiled.clone(),
    )
    .unwrap()
}

fn mu_param() -> EstimatedParam {
    EstimatedParam {
        name: "mu".into(),
        index: 0,
        initial: MU_START,
        rw_sd: 0.1,
        transform: Transform::Log { lo: MU_LO, hi: MU_HI },
        lower: MU_LO,
        upper: MU_HI,
        rw_sd_auto: false,
        perturb_only_at_t0: false,
    }
}

// ── exact oracle ─────────────────────────────────────────────────────

fn ln_factorial(n: u64) -> f64 {
    // Table-backed: the oracle calls this ~10⁸ times in a debug build.
    const TABLE: usize = N0 as usize + 1;
    static LN_FACT: std::sync::OnceLock<[f64; TABLE]> = std::sync::OnceLock::new();
    let table = LN_FACT.get_or_init(|| {
        let mut t = [0.0; TABLE];
        for k in 1..TABLE {
            t[k] = t[k - 1] + (k as f64).ln();
        }
        t
    });
    match table.get(n as usize) {
        Some(&v) => v,
        None => (1..=n).map(|k| (k as f64).ln()).sum(),
    }
}

fn binom_logpmf(k: u64, n: u64, p: f64) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
        + k as f64 * p.ln()
        + (n - k) as f64 * (1.0 - p).ln()
}

fn poisson_logpmf(y: f64, rate: f64) -> f64 {
    y * rate.ln() - rate - ln_factorial(y as u64)
}

/// Every death sequence `(d_1, …, d_T)` with `Σ d_t ≤ N₀`.
fn all_paths() -> Vec<[u64; N_SUBSTEPS]> {
    fn rec(t: usize, left: u64, cur: &mut [u64; N_SUBSTEPS], out: &mut Vec<[u64; N_SUBSTEPS]>) {
        if t == N_SUBSTEPS {
            out.push(*cur);
            return;
        }
        for d in 0..=left {
            cur[t] = d;
            rec(t + 1, left - d, cur, out);
        }
    }
    let mut out = Vec::new();
    rec(0, N0, &mut [0; N_SUBSTEPS], &mut out);
    out
}

/// How a rung's target weights the two densities: `π ∝ p(x|μ)^a · p(y|x,μ)^b`.
#[derive(Clone, Copy)]
struct Weights {
    process: f64,
    observation: f64,
}

/// Observation-only tempering at `β` — the target every rung must have.
fn obs_tempered(beta: f64) -> Weights {
    Weights { process: 1.0, observation: beta }
}

/// `log Σ_x p(x | μ)^a · p(y | x, μ)^b` by enumeration.
fn log_marginal(mu: f64, w: Weights, paths: &[[u64; N_SUBSTEPS]]) -> f64 {
    let p = 1.0 - (-mu).exp();
    let mut terms = Vec::with_capacity(paths.len());
    for path in paths {
        let mut n = N0;
        let (mut lp, mut lo) = (0.0, 0.0);
        for (t, &d) in path.iter().enumerate() {
            lp += binom_logpmf(d, n, p);
            lo += poisson_logpmf(Y[t], mu * d as f64 + OBS_FLOOR);
            n -= d;
        }
        terms.push(w.process * lp + w.observation * lo);
    }
    let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + terms.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

/// The functionals every assertion compares: the mean of `μ`, and the
/// probability that `μ` lies below [`MU_CUT`].
#[derive(Debug)]
struct Functionals {
    mean: f64,
    below_cut: f64,
}

/// Exact functionals of the θ-marginal of `π ∝ p(μ) · Σ_x p(x|μ)^a p(y|x,μ)^b`
/// under the flat prior on the box, by composite Simpson.
fn exact_functionals(w: Weights) -> Functionals {
    let paths = all_paths();
    let h = (MU_HI - MU_LO) / (QUAD_POINTS - 1) as f64;
    let nodes: Vec<f64> = (0..QUAD_POINTS).map(|i| MU_LO + i as f64 * h).collect();
    let logd: Vec<f64> = nodes.iter().map(|&mu| log_marginal(mu, w, &paths)).collect();
    let m = logd.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut wt: Vec<f64> = logd
        .iter()
        .enumerate()
        .map(|(i, &l)| {
            let simpson = if i == 0 || i == QUAD_POINTS - 1 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            simpson * (l - m).exp()
        })
        .collect();
    let z: f64 = wt.iter().sum();
    for x in &mut wt {
        *x /= z;
    }
    Functionals {
        mean: nodes.iter().zip(&wt).map(|(x, q)| x * q).sum(),
        below_cut: nodes.iter().zip(&wt).filter(|(x, _)| **x < MU_CUT).map(|(_, q)| q).sum(),
    }
}

// ── the sampler under test ───────────────────────────────────────────

fn run_cold_chain(ladder: &[f64], use_nuts: bool, seed: u64) -> Vec<f64> {
    let (compiled, base_params) = pure_death_model();
    let obs = observations();
    let obs_m = obs_model(&compiled);
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        ancestor_sampling: true,
        n_particles: 16,
        n_sweeps: N_SWEEPS,
        burn_in: BURN_IN,
        thin: 1,
        dt: 1.0,
        use_nuts,
        dense_mass: false,
        max_tree_depth: 6,
        tempering: ladder.to_vec(),
        trajectory_warmup: 0,
        csmc_sweeps_per_nuts: 1,
        step_policy: sim::schedule::StepPolicy::Snap,
    };
    let result = run_pgas(
        &compiled,
        &[mu_param()],
        &[Prior::Fixed(sim::inference::prior::Density::Flat)],
        &base_params,
        &config,
        &obs,
        &obs_m,
        seed,
        None,
        None,
        "hash".into(),
    )
    .unwrap();
    if ladder.len() > 1 {
        let swaps = &result.swap_acceptance_rates;
        assert!(
            swaps.iter().all(|&r| r > 0.05),
            "the ladder must actually exchange states for the case to test the swap \
             (swap rates {swaps:?})"
        );
    }
    result.sweeps.iter().map(|s| s.params[0]).collect()
}

/// `(estimate, batch-means SE)` of `E[f(μ)]` from one chain.
fn batch_mean_se(draws: &[f64], f: impl Fn(f64) -> f64) -> (f64, f64) {
    let b = draws.len() / N_BATCHES;
    let means: Vec<f64> = (0..N_BATCHES)
        .map(|k| draws[k * b..(k + 1) * b].iter().map(|&x| f(x)).sum::<f64>() / b as f64)
        .collect();
    let grand = means.iter().sum::<f64>() / N_BATCHES as f64;
    let var = means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / (N_BATCHES - 1) as f64;
    (grand, (var / N_BATCHES as f64).sqrt())
}

fn assert_cold_marginal_is_the_posterior(label: &str, ladder: &[f64], use_nuts: bool, seed: u64) {
    let exact = exact_functionals(obs_tempered(1.0));
    let draws = run_cold_chain(ladder, use_nuts, seed);
    let (m, m_se) = batch_mean_se(&draws, |x| x);
    let (p, p_se) = batch_mean_se(&draws, |x| if x < MU_CUT { 1.0 } else { 0.0 });
    let zm = (m - exact.mean) / m_se;
    let zp = (p - exact.below_cut) / p_se;
    let n = draws.len() as f64;
    let sd = (draws.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    // (batch-means SE / i.i.d. SE)² — the integrated autocorrelation time.
    let tau = (m_se / (sd / n.sqrt())).powi(2);
    eprintln!(
        "{label} {ladder:?}: E[mu] {m:.4} (exact {:.4}, se {m_se:.4}, z {zm:+.2}); \
         P(mu<{MU_CUT}) {p:.4} (exact {:.4}, se {p_se:.4}, z {zp:+.2}); tau(E[mu]) {tau:.1}",
        exact.mean, exact.below_cut
    );
    assert!(
        zm.abs() < Z_BAR && zp.abs() < Z_BAR,
        "{label} {ladder:?}: the cold-rung marginal of mu is not the exact posterior \
         (E[mu] z = {zm:+.2}, P(mu<{MU_CUT}) z = {zp:+.2}; bar {Z_BAR}). Tempering may \
         change mixing, never the answer."
    );
}

/// Non-vacuity. The ladder cases below can only detect a wrong hot-rung
/// target if this fixture's tempered laws are genuinely different — from the
/// posterior, and from complete-data tempering `[p(x|μ) p(y|x,μ)]^β` (the other
/// reading of "temper the model"). Checked on the exact values, so it costs no
/// sampling.
///
/// "Far apart" is [`SEPARATION`]: five times the ~0.005 Monte Carlo SE of
/// `E[μ]` the ladder cases achieve, so the difference is well resolved at the
/// sample sizes used. Checked for the hottest rung of each ladder.
#[test]
fn the_tempered_targets_are_far_apart_on_this_fixture() {
    const SEPARATION: f64 = 0.025;
    let cold = exact_functionals(obs_tempered(1.0));
    eprintln!("beta 1: {cold:?}");
    for beta in [0.25, 0.1] {
        let hot = exact_functionals(obs_tempered(beta));
        let complete = exact_functionals(Weights { process: beta, observation: beta });
        eprintln!("beta {beta}: obs-tempered {hot:?}, complete-data-tempered {complete:?}");
        assert!(
            (hot.mean - cold.mean).abs() > SEPARATION,
            "β = {beta}: π_β(μ) mean {} is too close to the posterior's {}",
            hot.mean, cold.mean
        );
        assert!(
            (hot.mean - complete.mean).abs() > SEPARATION,
            "β = {beta}: observation-only and complete-data tempering are too close \
             ({} vs {}) for this fixture to tell the designs apart",
            hot.mean, complete.mean
        );
    }
}

/// Control: the untempered chain. Validates the fixture and the oracle
/// before any ladder is involved.
#[test]
fn single_rung_matches_the_exact_posterior() {
    assert_cold_marginal_is_the_posterior("MH", &[1.0], false, 11);
}

#[test]
fn cold_rung_matches_the_exact_posterior_under_a_mild_ladder() {
    assert_cold_marginal_is_the_posterior("MH", &[1.0, 0.5, 0.25], false, 12);
}

#[test]
fn cold_rung_matches_the_exact_posterior_under_a_steep_ladder() {
    assert_cold_marginal_is_the_posterior("MH", &[1.0, 0.3, 0.1], false, 13);
}

/// The NUTS θ-move reaches the same target through a different path — the
/// tempered energy and gradient of `complete_data_loglik_grad` — so it gets
/// its own ladder case.
#[test]
fn cold_rung_matches_the_exact_posterior_under_a_ladder_with_nuts() {
    assert_cold_marginal_is_the_posterior("NUTS", &[1.0, 0.3, 0.1], true, 14);
}
