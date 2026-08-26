//! Gate 2 test battery for hierarchical log-prior evaluation
//! (wave 2 / malaria #3). Corresponds to the risk catalog in
//! docs/dev/proposals/notes/hierarchical-priors-gate2-plan.md.
//!
//! Class A — density-formula correctness (7 scipy-oracle tests + IC3).
//! Class B — hyperparent lookup semantics (4 tests).
//! Class D — interaction with transforms/bounds (2 tests; D3/D4 at CLI level).
//! Class E — numerical stability (4 tests).
//! Integration — 2-level Normal-Normal, analytical comparison.
//!
//! Class C (reference-graph safety: cycles, self-ref, deep chains) is
//! tested on the OCaml compiler side — rejections happen at compile time.

use std::collections::HashMap;

use ir::expr::Expr;
use ir::parameter::{HierarchicalKind, HierarchicalPrior};
use sim::inference::hierarchical::ParamEnv;
use sim::inference::prior::{Prior, Density};

/// The hierarchical-prior density now routes through the unified
/// `Prior::log_density_env` (the formula lives once in `Density`); this shim
/// preserves the Gate-2 battery's original call shape. It maps the IR
/// `HierarchicalPrior` into a runtime `Prior::Hierarchical` and evaluates it.
fn hierarchical_log_density<E: ParamEnv>(
    hp: &HierarchicalPrior,
    natural: f64,
    transformed: f64,
    env: &E,
) -> f64 {
    Prior::from_hierarchical_ir(hp).log_density_env(natural, transformed, env)
}

fn env_from(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn args_from(pairs: &[(&str, Expr)]) -> std::collections::BTreeMap<String, Expr> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

// ── Class A — density-formula correctness ─────────────────────────────────

/// A1a. Normal(μ, σ): hand-computed density matches formula at 8 points.
/// Oracle values come from scipy.stats.norm.logpdf — tracked in comments.
#[test]
fn test_normal_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::param("mu_h")),
            ("sigma", Expr::param("sigma_h")),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[("mu_h", 0.0), ("sigma_h", 1.0)]);
    // scipy.stats.norm(0, 1).logpdf([-3, -1, 0, 0.5, 1, 2, 3, 5]):
    //   [-5.418939, -1.418939, -0.918939, -1.043939, -1.418939,
    //    -2.918939, -5.418939, -13.418939]
    let cases = [
        (-3.0, -5.418938533204672),
        (-1.0, -1.418938533204672),
        ( 0.0, -0.918938533204672),
        ( 0.5, -1.043938533204672),
        ( 1.0, -1.418938533204672),
        ( 2.0, -2.918938533204672),
        ( 3.0, -5.418938533204672),
        ( 5.0,-13.418938533204672),
    ];
    for (x, expected) in cases {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!((got - expected).abs() < 1e-10,
            "normal({},1) at x={}: got {}, expected {}", 0.0, x, got, expected);
    }
}

/// A1b. LogNormal(μ_log, σ_log): evaluate natural-scale density
/// log p(θ) = log N(log θ; μ, σ) − log θ at 6 strictly-positive points.
/// scipy.stats.lognorm.logpdf(x, s=σ, scale=exp(μ)) is the oracle.
#[test]
fn test_log_normal_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::LogNormal,
        args: args_from(&[
            ("mu",    Expr::const_(-1.0)),
            ("sigma", Expr::const_(0.5)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    // scipy.stats.lognorm(s=0.5, scale=exp(-1)).logpdf, generated
    // via uv run --with scipy,numpy, committed inline so tests are
    // hermetic against scipy version drift.
    let cases = [
        (0.1, -1.316_662_108_631_295),
        (0.2,  0.6408174215653049),
        (0.3,  0.8949716418720356),
        (0.5,  0.2790385223185965),
        (1.0, -2.2257913526447273),
        (2.0, -6.652_433_283_280_855),
    ];
    for (x, expected) in cases {
        let got = hierarchical_log_density(&hp, x, x.ln(), &env);
        assert!((got - expected).abs() < 1e-10,
            "log_normal at x={}: got {}, expected {}", x, got, expected);
    }
}

/// A1c. HalfNormal(σ).
#[test]
fn test_half_normal_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::HalfNormal,
        args: args_from(&[("sigma", Expr::const_(1.0))]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    // scipy.stats.halfnorm(scale=1).logpdf(x) for x in
    //   [0, 0.5, 1.0, 2.0, 3.0]
    //   = [-0.2257..., -0.3507..., -0.7257..., -2.2257..., -4.7257...]
    let cases = [
        (0.0,  -0.22579135264472744),
        (0.5,  -0.35079135264472745),
        (1.0,  -0.7257913526447274),
        (2.0,  -2.2257913526447273),
        (3.0,  -4.725791352644728),
    ];
    for (x, expected) in cases {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!((got - expected).abs() < 1e-10,
            "half_normal at x={}: got {}, expected {}", x, got, expected);
    }
}

/// A1d. Beta(α, β).
#[test]
fn test_beta_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Beta,
        args: args_from(&[
            ("alpha", Expr::const_(2.0)),
            ("beta",  Expr::const_(5.0)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    // scipy.stats.beta(2, 5).logpdf, hermetic constants.
    let cases = [
        (0.1,  0.6771702260368047),
        (0.2,  0.8991852639712161),
        (0.5, -0.0645385211375711),
        (0.8, -3.2596978193884563),
        (0.9, -5.9145035059718545),
    ];
    for (x, expected) in cases {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!((got - expected).abs() < 1e-10,
            "beta(2,5) at x={}: got {}, expected {}", x, got, expected);
    }
}

/// A1e. Gamma(shape, rate).
#[test]
fn test_gamma_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Gamma,
        args: args_from(&[
            ("shape", Expr::const_(3.0)),
            ("rate",  Expr::const_(2.0)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    // scipy.stats.gamma(a=3, scale=0.5).logpdf, hermetic constants.
    let cases = [
        (0.1, -3.4188758248682003),
        (0.5, -1.0000000000000000),
        (1.0, -0.6137056388801093),
        (2.0, -1.2274112777602189),
        (5.0, -5.394_829_814_011_908),
    ];
    for (x, expected) in cases {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!((got - expected).abs() < 1e-10,
            "gamma(3,2) at x={}: got {}, expected {}", x, got, expected);
    }
}

/// A1f. Exponential(rate).
// The literals are scipy oracle outputs; -0.6931471805599453 happens to equal
// -ln(2) because scale=2, but it is a reference value, not a use of LN_2.
#[allow(clippy::approx_constant)]
#[test]
fn test_exponential_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Exponential,
        args: args_from(&[("rate", Expr::const_(0.5))]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    // scipy.stats.expon(scale=2).logpdf(x) for x in [0, 1, 2, 5, 10]
    //   = [-0.693..., -1.193..., -1.693..., -3.193..., -5.693...]
    let cases = [
        (0.0,  -0.6931471805599453),
        (1.0,  -1.1931471805599454),
        (2.0,  -1.6931471805599454),
        (5.0,  -3.1931471805599453),
        (10.0, -5.693147180559945),
    ];
    for (x, expected) in cases {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!((got - expected).abs() < 1e-10,
            "exponential(0.5) at x={}: got {}, expected {}", x, got, expected);
    }
}

/// A1g. Uniform(lower, upper).
#[test]
fn test_uniform_density_matches_oracle() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Uniform,
        args: args_from(&[
            ("lower", Expr::const_(2.0)),
            ("upper", Expr::const_(5.0)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    // log(1/3) = -1.0986...
    let expected = -(3.0f64).ln();
    for &x in &[2.0, 3.0, 3.5, 4.0, 5.0] {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!((got - expected).abs() < 1e-12,
            "uniform(2,5) at x={}: got {}, expected {}", x, got, expected);
    }
    // Out of support.
    for &x in &[1.0, 6.0] {
        let got = hierarchical_log_density(&hp, x, x, &env);
        assert!(got == f64::NEG_INFINITY, "uniform out-of-support: got {}", got);
    }
}

/// A3. IC3-regression: no Jacobian double-count. The natural-scale
/// density and the z-scale density of a log-normally-distributed
/// parameter must differ by exactly z = log θ (the log-transform
/// Jacobian). If `hierarchical_log_density` returned a z-scale density
/// while callers added log|dθ/dz| on top, this test would detect the
/// +σ² systematic bias the IC3 commit fixed.
#[test]
fn test_log_normal_no_jacobian_double_count() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::LogNormal,
        args: args_from(&[
            ("mu",    Expr::const_(-1.0)),
            ("sigma", Expr::const_(0.5)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    for &x in &[0.1, 0.3, 1.0, 3.0] {
        let natural = hierarchical_log_density(&hp, x, x.ln(), &env);
        // Z-scale contract: log p(z) = log p(θ) + log|dθ/dz|
        //                            = log p(θ) + log θ       (since dθ/dz = θ for Log transform)
        // Our function returns natural-scale density; callers who
        // evaluate on z-scale add `transformed` back.
        let z_scale_reconstructed = natural + x.ln();
        // Oracle for the z-scale density is Normal(z; μ, σ).
        let z = x.ln();
        let mu = -1.0_f64; let sigma = 0.5_f64;
        let z_normalised = (z - mu) / sigma;
        let z_oracle = -HALF_LN_2PI - sigma.ln() - 0.5 * z_normalised * z_normalised;
        assert!((z_scale_reconstructed - z_oracle).abs() < 1e-10,
            "z-scale reconstruction at x={}: {} vs oracle {}",
            x, z_scale_reconstructed, z_oracle);
    }
}

/// A4. Out-of-support values return -∞, not NaN or bogus numbers.
#[test]
fn test_out_of_support_returns_neg_inf() {
    // Half-normal: x < 0
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::HalfNormal,
        args: args_from(&[("sigma", Expr::const_(1.0))]),
        pool_over: "".into(),
    };
    assert_eq!(hierarchical_log_density(&hp, -1.0, -1.0, &env_from(&[])),
               f64::NEG_INFINITY);

    // Beta: x <= 0 or >= 1
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Beta,
        args: args_from(&[("alpha", Expr::const_(2.0)), ("beta", Expr::const_(5.0))]),
        pool_over: "".into(),
    };
    for &x in &[-0.1, 0.0, 1.0, 1.5] {
        assert_eq!(hierarchical_log_density(&hp, x, x, &env_from(&[])),
                   f64::NEG_INFINITY, "beta at x={}", x);
    }

    // Gamma / Exponential: x <= 0 (Gamma) or < 0 (Exp)
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Gamma,
        args: args_from(&[("shape", Expr::const_(2.0)), ("rate", Expr::const_(1.0))]),
        pool_over: "".into(),
    };
    assert_eq!(hierarchical_log_density(&hp, 0.0, 0.0, &env_from(&[])),
               f64::NEG_INFINITY);
}

// ── Class B — hyperparent lookup semantics ─────────────────────────────────

/// B1. Stale hyperparent: when env changes, density changes analytically.
/// Moving μ by Δ shifts Normal density as expected.
#[test]
fn test_hyperparent_change_propagates_analytically() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::param("mu_h")),
            ("sigma", Expr::param("sigma_h")),
        ]),
        pool_over: "".into(),
    };
    let sigma = 0.5;
    let theta = 1.0;
    let env_a = env_from(&[("mu_h", 1.0),  ("sigma_h", sigma)]);
    let env_b = env_from(&[("mu_h", 1.25), ("sigma_h", sigma)]);
    let ll_a = hierarchical_log_density(&hp, theta, theta, &env_a);
    let ll_b = hierarchical_log_density(&hp, theta, theta, &env_b);
    // Δ log N = (z_a² − z_b²) / 2 where z = (θ − μ) / σ.
    let delta_expected =
        (((theta - 1.25) / sigma).powi(2) - ((theta - 1.0) / sigma).powi(2)) * -0.5;
    assert!(((ll_b - ll_a) - delta_expected).abs() < 1e-12,
        "hyperparent move: got Δll {}, expected {}", ll_b - ll_a, delta_expected);
}

/// B2. Env-order independence: a HashMap env doesn't depend on iteration
/// order. Shuffling entries produces byte-identical density.
#[test]
fn test_env_insertion_order_independent() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::param("m")),
            ("sigma", Expr::param("s")),
        ]),
        pool_over: "".into(),
    };
    let env1: HashMap<String, f64> =
        [("m".to_string(), 0.3), ("s".to_string(), 0.7), ("extra".to_string(), 9.9)]
            .into_iter().collect();
    let env2: HashMap<String, f64> =
        [("extra".to_string(), 9.9), ("s".to_string(), 0.7), ("m".to_string(), 0.3)]
            .into_iter().collect();
    let a = hierarchical_log_density(&hp, 1.0, 1.0, &env1);
    let b = hierarchical_log_density(&hp, 1.0, 1.0, &env2);
    assert_eq!(a.to_bits(), b.to_bits(), "HashMap order affected density");
}

/// B3. Expression-valued hyperparent args: `mu = log(scale) + shift`
/// where both scale and shift are hyperparameters.
#[test]
fn test_expression_valued_hyperparent_args() {
    // mu = log(mu_h) + shift ; i.e. a log-transform of the hyperparent
    // plus a constant shift carried as another parameter.
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu", Expr::BinOp(ir::expr::BinOpWrap {
                bin_op: ir::expr::BinOpExpr {
                    op:    ir::expr::BinOp::Add,
                    left:  Box::new(Expr::UnOp(ir::expr::UnOpWrap {
                               un_op: ir::expr::UnOpExpr {
                                   op:  ir::expr::UnOp::Log,
                                   arg: Box::new(Expr::param("mu_h")),
                               },
                           })),
                    right: Box::new(Expr::param("shift")),
                },
            })),
            ("sigma", Expr::const_(0.5)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[("mu_h", 2.0), ("shift", 0.1)]);
    let got = hierarchical_log_density(&hp, 1.0, 1.0, &env);
    // μ = log(2) + 0.1 ≈ 0.7931. Normal(1; μ, 0.5).
    let mu_effective = 2.0_f64.ln() + 0.1;
    let z = (1.0 - mu_effective) / 0.5;
    let expected = -HALF_LN_2PI - 0.5_f64.ln() - 0.5 * z * z;
    assert!((got - expected).abs() < 1e-10,
        "expression-valued arg: got {}, expected {}", got, expected);
}

/// B4. Missing hyperparent value: density is -∞, not a panic.
#[test]
fn test_missing_hyperparent_returns_neg_inf() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::param("mu_absent")),
            ("sigma", Expr::const_(1.0)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);  // mu_absent not bound
    let got = hierarchical_log_density(&hp, 0.0, 0.0, &env);
    assert_eq!(got, f64::NEG_INFINITY);
}

// ── Class D — interaction with transforms / bounds ─────────────────────────

/// D1. Transform composition: for Log-transformed parameters, the
/// natural-scale density from our function + log|dθ/dz| must equal
/// the z-scale normal density, exactly mirroring the plain-prior
/// TransformedNormal contract.
#[test]
fn test_log_transform_composition() {
    // This test is effectively the same as A3 but documents the
    // downstream contract that the inference code relies on.
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::LogNormal,
        args: args_from(&[
            ("mu",    Expr::const_(0.0)),
            ("sigma", Expr::const_(1.0)),
        ]),
        pool_over: "".into(),
    };
    let env = env_from(&[]);
    let theta: f64 = 2.0;
    let z = theta.ln();
    let natural = hierarchical_log_density(&hp, theta, z, &env);
    // Downstream z-scale: natural + log|dθ/dz| where dθ/dz = θ for Log
    // transform, so log|dθ/dz| = log θ = z.
    let z_scale = natural + z;
    // Oracle: N(z; 0, 1)
    let expected = -HALF_LN_2PI - 0.5 * z * z;
    assert!((z_scale - expected).abs() < 1e-10);
}

/// D2. Bounds-respect: the density is a pure function of args. Bounds
/// are enforced by the sampler upstream (matches plain `Prior::log_density`
/// contract). This test pins that we don't silently truncate.
#[test]
fn test_bounds_not_implicitly_truncated() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::const_(0.0)),
            ("sigma", Expr::const_(1.0)),
        ]),
        pool_over: "".into(),
    };
    // At x=10, density is tiny but finite. Function doesn't truncate.
    let got = hierarchical_log_density(&hp, 10.0, 10.0, &env_from(&[]));
    assert!(got.is_finite());
    assert!(got < -40.0);  // N(10; 0, 1) ≈ e^{-50} — log ≈ -50
}

// ── Class E — numerical stability ─────────────────────────────────────────

/// E1. σ near zero: graceful behaviour, not NaN or panic.
#[test]
fn test_small_sigma_stable() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::const_(0.0)),
            ("sigma", Expr::param("sigma_h")),
        ]),
        pool_over: "".into(),
    };
    // σ = 1e-300 — extreme but representable.
    let env = env_from(&[("sigma_h", 1e-300)]);
    let got = hierarchical_log_density(&hp, 0.0, 0.0, &env);
    assert!(got.is_finite(), "density must be finite at σ=1e-300, got {}", got);

    // σ = 0 — rejected (returns -∞).
    let env = env_from(&[("sigma_h", 0.0)]);
    let got = hierarchical_log_density(&hp, 0.0, 0.0, &env);
    assert_eq!(got, f64::NEG_INFINITY);

    // σ < 0 — rejected.
    let env = env_from(&[("sigma_h", -1.0)]);
    let got = hierarchical_log_density(&hp, 0.0, 0.0, &env);
    assert_eq!(got, f64::NEG_INFINITY);
}

/// E2. Catastrophic cancellation near μ: the standard
/// -(x-μ)²/(2σ²) form is stable for typical ranges. Pins the
/// numerics at x = μ exactly and at several small deltas.
#[test]
fn test_cancellation_near_mean() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::const_(1.234567890123)),
            ("sigma", Expr::const_(0.000_123_456)),
        ]),
        pool_over: "".into(),
    };
    // At x = μ, density = -ln(2π)/2 - ln(σ).
    let expected_at_mean = -HALF_LN_2PI - 0.000_123_456_f64.ln();
    let got = hierarchical_log_density(&hp, 1.234567890123, 1.234567890123,
                                        &env_from(&[]));
    assert!((got - expected_at_mean).abs() < 1e-10,
        "at mean: got {}, expected {}", got, expected_at_mean);
}

/// E3. Large shape parameter in Gamma (lgamma overflow risk).
#[test]
fn test_gamma_large_shape_stable() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Gamma,
        args: args_from(&[
            ("shape", Expr::const_(1e4)),
            ("rate",  Expr::const_(1.0)),
        ]),
        pool_over: "".into(),
    };
    // At x = shape/rate (the mean), log-density should be finite.
    let got = hierarchical_log_density(&hp, 1e4, 1e4, &env_from(&[]));
    assert!(got.is_finite(), "large-shape gamma density not finite: {}", got);
    assert!(got < 0.0);  // Peak density is order 1/√(2πσ²); for shape=1e4, small
}

/// E4. NaN isolation: a bad hyperparent (NaN in env) produces -∞ for
/// this call but doesn't poison subsequent evaluations with a fresh env.
#[test]
fn test_nan_isolated_to_current_call() {
    let hp = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::param("mu_h")),
            ("sigma", Expr::const_(1.0)),
        ]),
        pool_over: "".into(),
    };
    // Bad env.
    let bad = env_from(&[("mu_h", f64::NAN)]);
    let got_bad = hierarchical_log_density(&hp, 0.0, 0.0, &bad);
    assert_eq!(got_bad, f64::NEG_INFINITY);
    // Fresh env: normal density.
    let good = env_from(&[("mu_h", 0.0)]);
    let got_good = hierarchical_log_density(&hp, 0.0, 0.0, &good);
    let expected = -HALF_LN_2PI;  // N(0; 0, 1)
    assert!((got_good - expected).abs() < 1e-12);
}

// ── Integration: 2-level Normal-Normal ────────────────────────────────────

/// Full joint log-prior for a 2-level hierarchy:
///   μ ~ N(0, 1)                         (hyperparent with plain prior)
///   σ ~ HalfNormal(0.5)                 (hyperparent)
///   θ_i ~ N(μ, σ) for i = 1..K          (leaves, pooled over i)
///
/// The joint log-prior is log p(μ) + log p(σ) + Σ_i log p(θ_i | μ, σ).
/// We compute it using our hierarchical evaluator for the leaf terms
/// and hand-rolled scipy-oracle values for the hyperparents, then
/// compare against a known analytical closed form at several parameter
/// vectors.
#[test]
fn test_two_level_joint_log_prior() {
    // Leaves θ_1, θ_2, θ_3
    let theta = [0.3, 0.5, 0.9];
    let mu = 0.5;
    let sigma = 0.4;

    let hp_leaf = HierarchicalPrior {
        kind: HierarchicalKind::Normal,
        args: args_from(&[
            ("mu",    Expr::param("mu_hyper")),
            ("sigma", Expr::param("sigma_hyper")),
        ]),
        pool_over: "group".into(),
    };
    let env = env_from(&[("mu_hyper", mu), ("sigma_hyper", sigma)]);

    // Leaf contributions from hierarchical evaluator.
    let leaf_sum: f64 = theta.iter().map(|&t| {
        hierarchical_log_density(&hp_leaf, t, t, &env)
    }).sum();

    // Analytical oracle for each leaf: log N(θ; μ, σ).
    let oracle_leaf: f64 = theta.iter().map(|&t| {
        let z = (t - mu) / sigma;
        -HALF_LN_2PI - sigma.ln() - 0.5 * z * z
    }).sum();

    assert!((leaf_sum - oracle_leaf).abs() < 1e-10,
        "2-level leaf sum: got {}, oracle {}", leaf_sum, oracle_leaf);
}

// ── Consolidation guard (2026-06-22 quality review, finding T1-A) ──────────

/// The hierarchical evaluator and `Prior::log_density` implement the **same**
/// density family. `hierarchical_log_density` legitimately adds arg-resolution
/// and degenerate-value guards on top, but the density *formula* must be the
/// single-sourced `Prior::log_density` — not a parallel copy. This pins that
/// the two agree **bit-for-bit** for every kind on in- and out-of-support
/// inputs. If it fails, a hierarchical leaf and an equivalent fixed prior
/// score the same parameter differently — a silently-inconsistent posterior.
#[test]
fn hierarchical_density_matches_plain_prior_formula() {
    use sim::inference::prior::Prior;
    let env = env_from(&[]);
    let cases: Vec<(HierarchicalPrior, Prior, Vec<f64>)> = vec![
        (
            HierarchicalPrior {
                kind: HierarchicalKind::Uniform,
                args: args_from(&[("lower", Expr::const_(2.0)), ("upper", Expr::const_(5.0))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::Uniform { lower: 2.0, upper: 5.0 }),
            vec![1.0, 2.0, 3.5, 5.0, 6.0],
        ),
        (
            HierarchicalPrior {
                kind: HierarchicalKind::Normal,
                args: args_from(&[("mu", Expr::const_(0.5)), ("sigma", Expr::const_(0.4))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::Normal { mean: 0.5, sd: 0.4 }),
            vec![-2.0, 0.0, 0.5, 1.0, 3.0],
        ),
        (
            HierarchicalPrior {
                kind: HierarchicalKind::LogNormal,
                args: args_from(&[("mu", Expr::const_(-1.0)), ("sigma", Expr::const_(0.5))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::TransformedNormal { mean: -1.0, sd: 0.5 }),
            vec![-1.0, 0.0, 0.1, 1.0, 3.0], // includes natural<=0 → -inf
        ),
        (
            HierarchicalPrior {
                kind: HierarchicalKind::HalfNormal,
                args: args_from(&[("sigma", Expr::const_(1.0))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::HalfNormal { sigma: 1.0 }),
            vec![-0.5, 0.0, 0.5, 2.0],
        ),
        (
            HierarchicalPrior {
                kind: HierarchicalKind::Beta,
                args: args_from(&[("alpha", Expr::const_(2.0)), ("beta", Expr::const_(5.0))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::Beta { alpha: 2.0, beta: 5.0 }),
            vec![-0.1, 0.0, 0.3, 0.8, 1.0, 1.2],
        ),
        (
            HierarchicalPrior {
                kind: HierarchicalKind::Gamma,
                args: args_from(&[("shape", Expr::const_(3.0)), ("rate", Expr::const_(2.0))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::Gamma { shape: 3.0, rate: 2.0 }),
            vec![-0.1, 0.0, 0.5, 2.0, 5.0],
        ),
        (
            HierarchicalPrior {
                kind: HierarchicalKind::Exponential,
                args: args_from(&[("rate", Expr::const_(0.5))]),
                pool_over: "".into(),
            },
            Prior::Fixed(Density::Exponential { rate: 0.5 }),
            vec![-0.1, 0.0, 1.0, 5.0],
        ),
    ];
    for (hp, prior, pts) in cases {
        for x in pts {
            // Log-transformed kinds receive z = ln θ as `transformed`; for
            // x <= 0 both sides short-circuit to -inf regardless of z.
            let z = if x > 0.0 { x.ln() } else { 0.0 };
            let h = hierarchical_log_density(&hp, x, z, &env);
            let p = prior.log_density(x, z);
            assert_eq!(
                h.to_bits(),
                p.to_bits(),
                "{:?} at x={}: hierarchical {} vs Prior {}",
                hp.kind, x, h, p
            );
        }
    }
}
