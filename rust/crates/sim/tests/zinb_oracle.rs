//! Cross-validate camdl's zero-inflated negative-binomial (ZINB) observation
//! likelihood and its gradient against an external reference.
//!
//! The fixture `zinb_gradient_ref.tsv` is committed, so this test is offline
//! and CI never needs R. Regenerate with
//! `Rscript scripts/gen_zinb_gradient_fixture.R`.
//!
//! Why the oracle is external. A finite difference taken over camdl's own
//! `zi_negbin_logpmf` would only establish that camdl's derivative matches
//! camdl's density — if the density itself were wrong the two would agree and
//! both be wrong. The reference therefore supplies the density independently:
//! base R's `dnbinom` for the NB2 component, and `numDeriv`'s Richardson
//! extrapolation for the derivative, neither of which knows the closed form
//! evaluated here.
//!
//! The one construction the generator and camdl would otherwise share is the
//! mixture itself. The generator does not get to assert it: before writing the
//! fixture it checks the mixture against normalization (summing to the NB
//! quantile carrying all but 1e-15 of the mass, with the remaining tail taken
//! exactly from `pnbinom`) and against the ZINB moment identities
//! `E[Y] = (1-pi)*mu`, `Var[Y] = (1-pi)*mu*(1 + mu/k + pi*mu)`. Those follow
//! from the mixture's definition rather than from its pmf algebra, so a
//! mis-stated pmf fails to produce a fixture at all.
//!
//! Rows where `pi` sits on a boundary carry `d_pi_onesided = 1`: a two-sided
//! step would leave `[0, 1]`, so the reference `d_pi` there is a one-sided
//! difference good to ~1e-4 rather than ~1e-10. Those rows are held to the
//! looser bound; `d_mu` and `d_k` stay two-sided on every row.

use sim::inference::obs_loglik::{zi_negbin_logpmf, zi_negbin_logpmf_grad};
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

struct Row {
    case: String,
    y: f64,
    mu: f64,
    k: f64,
    pi: f64,
    logpmf: f64,
    d_mu: f64,
    d_k: f64,
    d_pi: f64,
    d_pi_onesided: bool,
}

fn load() -> Vec<Row> {
    let path = fixture("zinb_gradient_ref.tsv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("case\t") || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 10, "malformed fixture row: {line}");
        rows.push(Row {
            case: f[0].to_string(),
            y: f[1].parse().unwrap(),
            mu: f[2].parse().unwrap(),
            k: f[3].parse().unwrap(),
            pi: f[4].parse().unwrap(),
            logpmf: f[5].parse().unwrap(),
            d_mu: f[6].parse().unwrap(),
            d_k: f[7].parse().unwrap(),
            d_pi: f[8].parse().unwrap(),
            d_pi_onesided: f[9].trim() == "1",
        });
    }
    assert!(!rows.is_empty(), "fixture had no rows");
    rows
}

fn assert_close(got: f64, want: f64, rel: f64, what: &str, case: &str) {
    let scale = want.abs().max(1.0);
    let err = (got - want).abs() / scale;
    assert!(
        err <= rel,
        "{case}: {what} = {got:.17e}, reference {want:.17e} \
         (relative error {err:.3e} exceeds {rel:.1e})"
    );
}

#[test]
fn zinb_logpmf_matches_external_reference() {
    for r in load() {
        let got = zi_negbin_logpmf(r.y, r.mu, r.k, r.pi);
        assert_close(got, r.logpmf, 1e-12, "logpmf", &r.case);
    }
}

#[test]
fn zinb_gradient_matches_external_reference() {
    for r in load() {
        let (d_mu, d_k, d_pi) = zi_negbin_logpmf_grad(r.y, r.mu, r.k, r.pi);
        // The two-sided Richardson reference is good to ~1e-10 relative; 1e-8
        // leaves headroom for the extrapolation's own error without admitting
        // a wrong closed form.
        assert_close(d_mu, r.d_mu, 1e-8, "d/d(mu)", &r.case);
        assert_close(d_k, r.d_k, 1e-8, "d/d(k)", &r.case);
        let tol_pi = if r.d_pi_onesided { 1e-4 } else { 1e-8 };
        assert_close(d_pi, r.d_pi, tol_pi, "d/d(pi)", &r.case);
    }
}

/// When both `pi` and `f_NB(0)` underflow in linear space, `S = pi + (1-pi)·f0`
/// underflows to 0.0 while `log S` stays finite, and a linear-space
/// `(1 - f0)/S` overflows to `+inf` — which the chain-rule accumulator then
/// turns into NaN via `inf * 0.0` for every estimated parameter absent from
/// `pi`'s gradient map, pairing a *finite* log-density with a NaN gradient.
/// The reference implementations bound the score analytically by
/// differentiating on the logit scale (pscl's `gradNegBin`; statsmodels'
/// `GenericZeroInflated.score_obs`); camdl's kernel is transform-agnostic, so
/// it must stay finite on its own.
#[test]
fn zinb_gradient_is_finite_at_the_s_underflow_corner() {
    // log f0 = k·ln(k/(k+mu)) ≈ -804.7, below ln(5e-324): f0 underflows to 0.
    let (mu, k) = (2000.0, 500.0);
    for &pi in &[0.0, 5e-324, 1e-310] {
        let v = zi_negbin_logpmf(0.0, mu, k, pi);
        let (d_mu, d_k, d_pi) = zi_negbin_logpmf_grad(0.0, mu, k, pi);
        assert!(v.is_finite(), "value must be finite (pi = {pi:e}): {v}");
        assert!(
            d_mu.is_finite() && d_k.is_finite(),
            "(mu, k) partials must be finite (pi = {pi:e}): ({d_mu}, {d_k})"
        );
        assert!(
            d_pi.is_finite() && d_pi > 0.0,
            "d/d(pi) must be finite and positive at the corner (pi = {pi:e}): {d_pi}"
        );
    }
}

/// At `pi = 0` the mixture is the plain NegBinomial, so the ZINB value and its
/// (mu, k) gradient must reduce to the NB ones exactly. This is an internal
/// consistency property rather than external evidence, and it is here because
/// it localizes a sign or factor error to the zero-inflation wrapper rather
/// than the NB base underneath it.
#[test]
fn zinb_reduces_to_negbin_at_pi_zero() {
    use sim::inference::obs_loglik::{negbin_logpmf, negbin_logpmf_grad};
    for &(y, mu, k) in &[
        (0.0, 5.0, 2.0),
        (7.0, 5.0, 2.0),
        (0.0, 200.0, 3.0),
        (12.0, 5.0, 0.05),
        (4.0, 5.0, 500.0),
    ] {
        let (zi_mu, zi_k, _) = zi_negbin_logpmf_grad(y, mu, k, 0.0);
        let (nb_mu, nb_k) = negbin_logpmf_grad(y, mu, k);
        assert_eq!(
            zi_negbin_logpmf(y, mu, k, 0.0),
            negbin_logpmf(y, mu, k),
            "value at pi=0 must equal the NB value (y={y}, mu={mu}, k={k})"
        );
        assert_eq!(zi_mu, nb_mu, "d/d(mu) at pi=0 (y={y}, mu={mu}, k={k})");
        assert_eq!(zi_k, nb_k, "d/d(k) at pi=0 (y={y}, mu={mu}, k={k})");
    }
}
