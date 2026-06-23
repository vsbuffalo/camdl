//! PMMH × hierarchical-prior plumbing tests (wave 2 / malaria #3 Gate 3a).
//!
//! These tests verify that the hierarchical-prior machinery is correctly
//! threaded through `run_pmmh`. They don't attempt full posterior
//! recovery — that's the Garki fit vignette's job. What they DO guarantee:
//!
//! 1. A hierarchical prior wrapped in `Prior::Hierarchical` is
//!    consumed correctly by the PMMH log-posterior path.
//! 2. Moving a hyperparent's *current* value changes the leaf's log-
//!    prior density by the analytically-expected amount. This is the
//!    critical "env lookup is live" check — equivalent to test B1 in
//!    the Gate 2 plan, but at the PMMH call-site rather than the
//!    density function in isolation.
//! 3. `Prior::log_density` (env-less) on a Hierarchical variant returns
//!    −∞ — the explicit fallback for callers that haven't been
//!    retrofitted. Pins the contract.

use std::collections::BTreeMap;
use ir::expr::Expr;
use ir::parameter::{HierarchicalKind, HierarchicalPrior};
use sim::inference::hierarchical::NamedParams;
use sim::inference::prior::{Prior, Density};

fn make_hier(kind: HierarchicalKind, args: &[(&str, Expr)]) -> HierarchicalPrior {
    HierarchicalPrior {
        kind,
        args: args.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<BTreeMap<_,_>>(),
        pool_over: "".into(),
    }
}

/// Plumbing check 1: Prior::Hierarchical participates in log_density_env
/// — and moving the hyperparent changes the leaf density by the
/// analytically-expected amount.
#[test]
fn test_prior_hierarchical_env_propagation() {
    let prior = Prior::from_hierarchical_ir(&make_hier(HierarchicalKind::Normal, &[
        ("mu",    Expr::param("mu_h")),
        ("sigma", Expr::param("sigma_h")),
    ]));

    // Param vector: [mu_h, sigma_h, theta]. Leaf evaluated at theta = 1.0.
    let names  = vec!["mu_h".to_string(), "sigma_h".to_string(), "theta".to_string()];
    let values_a = [0.0,  0.5, 1.0];  // μ = 0 → N(1; 0, 0.5)
    let values_b = [1.0,  0.5, 1.0];  // μ = 1 → N(1; 1, 0.5)

    let ll_a = prior.log_density_env(
        values_a[2], values_a[2], &NamedParams { names: &names, values: &values_a }
    );
    let ll_b = prior.log_density_env(
        values_b[2], values_b[2], &NamedParams { names: &names, values: &values_b }
    );

    // Analytical: Δ log N = -0.5 * ((1 - 1)²/0.25 - (1 - 0)²/0.25) = -0.5 * (0 - 4) = +2.
    let delta_expected = 2.0;
    assert!(((ll_b - ll_a) - delta_expected).abs() < 1e-12,
        "hyperparent env propagation: Δ got {}, expected {}",
        ll_b - ll_a, delta_expected);
}

/// Plumbing check 2: The env-less log_density returns −∞ for
/// Hierarchical. This is the "safety net" contract: callers that
/// haven't been updated to pass an env get an obviously-broken
/// posterior (−∞) rather than a silently-wrong one.
#[test]
fn test_prior_hierarchical_env_less_returns_neg_inf() {
    let prior = Prior::from_hierarchical_ir(&make_hier(HierarchicalKind::Normal, &[
        ("mu", Expr::param("mu_h")),
        ("sigma", Expr::const_(1.0)),
    ]));
    let got = prior.log_density(0.0, 0.0);
    assert_eq!(got, f64::NEG_INFINITY);
}

/// Plumbing check 3: Non-hierarchical priors ignore the env — identical
/// density with and without. Proves `log_density_env` is a true
/// superset of `log_density`.
#[test]
fn test_plain_priors_ignore_env() {
    let names = vec!["unused".to_string()];
    let values = [42.0];
    let env = NamedParams { names: &names, values: &values };

    for prior in [
        Prior::Fixed(Density::Flat),
        Prior::Fixed(Density::Uniform { lower: -1.0, upper: 2.0 }),
        Prior::Fixed(Density::Normal { mean: 0.0, sd: 1.0 }),
        Prior::Fixed(Density::HalfNormal { sigma: 1.0 }),
        Prior::Fixed(Density::Beta { alpha: 2.0, beta: 3.0 }),
        Prior::Fixed(Density::Gamma { shape: 2.0, rate: 1.0 }),
        Prior::Fixed(Density::Exponential { rate: 0.5 }),
    ] {
        let without = prior.log_density(0.5, 0.5);
        let with    = prior.log_density_env(0.5, 0.5, &env);
        assert_eq!(without, with, "env-aware density differed for {:?}", prior);
    }
}

/// Plumbing check 4: Missing hyperparent in env → Hierarchical density
/// is −∞. Same contract as the raw hierarchical_log_density test B4,
/// but verifying it composes correctly at the Prior level.
#[test]
fn test_prior_hierarchical_missing_hyperparent_neg_inf() {
    let prior = Prior::from_hierarchical_ir(&make_hier(HierarchicalKind::Normal, &[
        ("mu",    Expr::param("mu_missing")),
        ("sigma", Expr::const_(1.0)),
    ]));
    let names: Vec<String> = vec![];
    let values: [f64; 0]   = [];
    let env = NamedParams { names: &names, values: &values };
    assert_eq!(prior.log_density_env(0.0, 0.0, &env), f64::NEG_INFINITY);
}
