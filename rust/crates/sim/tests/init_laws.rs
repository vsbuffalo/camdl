//! Declared initial-state laws: the sampler, the density, and the gradient.
//!
//! `init { I ~ poisson(rate = I0) }` makes an initial condition a random
//! variable (proposal
//! `docs/dev/proposals/2026-08-23-initial-state-parameters.md`, staging steps 2
//! and 4). A law is three things at once — a **sampler**
//! (`initial_state_draw`), a **density** (`initial_state_logpdf`) and a
//! **gradient** (`initial_state_logpdf_grad`) — and wiring one without the
//! others is the silent-bias class: a NUTS gradient identically zero on a
//! coordinate the energy does depend on. Each of the three is asserted here
//! separately, against a number rather than against "is non-zero".
//!
//! The fixture is `ocaml/golden/init_laws.camdl`, which seeds `I` from a
//! Poisson, `E` from a NegBinomial, `R` from a Binomial and the real reservoir
//! `W` from a Normal, then computes `S = N0 - I - E - R`.
//!
//! Its hand-checkable property, and the reason the dependency-ordered
//! evaluation exists: **`S + E + I + R == N0` on every draw.** `S` reads what
//! the three laws DREW, not what they were expected to be. Take each law at its
//! mean while drawing the others (the pre-gh#733 behaviour, where an init RHS
//! read a compartment as zero) and the budget breaks by exactly the sampling
//! error of the three draws.

use std::sync::Arc;

use sim::{compiled_model::CompiledModel, rng::StatefulRng};

const SEED: u64 = 20260824;

fn load(rel: &str) -> ir::Model {
    let path = format!("{}/../../../{}", env!("CARGO_MANIFEST_DIR"), rel);
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    ir::from_str(&json).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// The golden fixture under its first scenario's parameter values.
fn fixture() -> (Arc<CompiledModel>, Vec<f64>) {
    let mut model = load("ocaml/golden/init_laws.ir.json");
    let preset = model.presets.first().cloned().expect("the fixture declares a scenario");
    for p in &mut model.parameters {
        if let Some(&v) = preset.params.get(&p.name) {
            p.value = p.value.with_value(v);
        }
    }
    let compiled = Arc::new(CompiledModel::new(model).expect("fixture must compile"));
    let params = compiled.default_params.clone();
    (compiled, params)
}

fn idx(c: &CompiledModel, name: &str) -> usize {
    let global = c.comp_index[name];
    c.global_to_int[global].expect("an integer compartment")
}

// ── 1. The sampler ───────────────────────────────────────────────────────────

/// The draw is genuinely stochastic: repeated draws from one advancing stream
/// differ, and they differ in EVERY law-seeded compartment.
///
/// Zero variance is what the same call produced before the laws were wired
/// (`initial_state_draw` returned `initial_state_mean` and consumed nothing),
/// so the numbers below are the red→green evidence for the sampler half. The
/// deterministic entry `S` is asserted to vary too — its variance is inherited
/// from the three draws it reads, which is the dependency order working.
#[test]
fn the_draw_is_stochastic_in_every_law_seeded_compartment() {
    let (compiled, params) = fixture();
    assert!(compiled.has_init_law, "the fixture must declare at least one law");

    let mut rng = StatefulRng::new(SEED);
    let n = 400;
    let mut draws: Vec<(f64, f64, f64, f64, f64)> = Vec::with_capacity(n);
    for _ in 0..n {
        let (int_s, real_s) = compiled.initial_state_draw(&params, &mut rng).expect("draw");
        draws.push((
            int_s.counts[idx(&compiled, "I")] as f64,
            int_s.counts[idx(&compiled, "E")] as f64,
            int_s.counts[idx(&compiled, "R")] as f64,
            int_s.counts[idx(&compiled, "S")] as f64,
            real_s.values[compiled.global_to_real[compiled.comp_index["W"]].unwrap()],
        ));
    }

    let var = |f: fn(&(f64, f64, f64, f64, f64)) -> f64| {
        let xs: Vec<f64> = draws.iter().map(f).collect();
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64, m)
    };

    // Poisson(50): Var = mean = 50. Wide tolerance — this is a sampler check,
    // not a distributional one; the point is that it is not zero.
    let (v_i, m_i) = var(|d| d.0);
    assert!(v_i > 5.0, "I has no initial-state variance (var = {v_i}, mean = {m_i})");
    assert!((m_i - 50.0).abs() < 10.0, "I's mean should be near rate = 50, got {m_i}");

    // NegBinomial(mean = 1.5 * I, k = 5): overdispersed, so var >> mean.
    let (v_e, m_e) = var(|d| d.1);
    assert!(v_e > m_e, "E must be overdispersed (var {v_e} <= mean {m_e})");

    // Binomial(100000, 0.2): Var = n p (1-p) = 16000.
    let (v_r, m_r) = var(|d| d.2);
    assert!(v_r > 1000.0, "R has no initial-state variance (var = {v_r})");
    assert!((m_r - 20000.0).abs() < 500.0, "R's mean should be near n*p = 20000, got {m_r}");

    // Normal(10, 2) on the real compartment.
    let (v_w, m_w) = var(|d| d.4);
    assert!(v_w > 0.5, "W has no initial-state variance (var = {v_w})");
    assert!((m_w - 10.0).abs() < 1.0, "W's mean should be near 10, got {m_w}");

    // The deterministic entry inherits the spread, because it reads the DRAWN
    // values. A zero here with non-zero variance above is precisely the gh#733
    // bug (the RHS reading its dependencies as constants).
    let (v_s, _) = var(|d| d.3);
    assert!(v_s > 0.0, "S must inherit the draws' spread; got zero variance");

    // Not one repeated state: distinct draws, not one value 400 times.
    let distinct: std::collections::HashSet<i64> =
        draws.iter().map(|d| d.0 as i64).collect();
    assert!(distinct.len() > 10, "only {} distinct I0 draws in {n}", distinct.len());
}

/// **The property.** `S + E + I + R == N0` on every single draw, with no
/// `balance { }` block — the invariant the dependency-ordered evaluation
/// exists to guarantee, and the one that was broken when an init RHS read a
/// compartment as zero.
///
/// This is what a captured trajectory baseline cannot tell you: a baseline
/// freezes whatever the code produced, so it can only ratchet against future
/// change. This asserts something that must be true of ANY correct
/// implementation.
#[test]
fn the_population_budget_holds_on_every_draw() {
    let (compiled, params) = fixture();
    let n0 = params[compiled.param_index["N0"]].round() as i64;
    assert_eq!(n0, 100_000, "the baseline scenario's N0");

    let mut rng = StatefulRng::new(SEED);
    let mut saw_nonmean = false;
    for k in 0..500 {
        let (int_s, _) = compiled.initial_state_draw(&params, &mut rng).expect("draw");
        let (s, e, i, r) = (
            int_s.counts[idx(&compiled, "S")],
            int_s.counts[idx(&compiled, "E")],
            int_s.counts[idx(&compiled, "I")],
            int_s.counts[idx(&compiled, "R")],
        );
        assert_eq!(
            s + e + i + r,
            n0,
            "draw {k}: S {s} + E {e} + I {i} + R {r} = {} != N0 {n0}",
            s + e + i + r
        );
        assert!(s >= 0 && e >= 0 && i >= 0 && r >= 0, "draw {k}: negative compartment");
        // Non-vacuity: if every draw landed on the mean, the budget would hold
        // for a reason that has nothing to do with reading the drawn values.
        if i != 50 || r != 20_000 {
            saw_nonmean = true;
        }
    }
    assert!(saw_nonmean, "every draw was the mean — the budget check proves nothing");
}

// ── 2. The density ───────────────────────────────────────────────────────────

/// `initial_state_logpdf` scores the DECLARED laws and nothing else, and it
/// equals the hand-computed sum of the four log-densities at the same `x0`.
///
/// Before the laws were wired this returned `0.0` unconditionally; the number
/// below is the red→green evidence for the density half.
#[test]
fn the_density_is_the_sum_of_the_declared_laws() {
    use sim::inference::obs_loglik::{binom_logpmf, negbin_logpmf, normal_logpdf, poisson_logpmf};

    let (compiled, params) = fixture();
    let mut rng = StatefulRng::new(SEED);
    let (int_s, real_s) = compiled.initial_state_draw(&params, &mut rng).expect("draw");

    let lp = compiled
        .initial_state_logpdf(&int_s.counts, &real_s.values, &params)
        .expect("logpdf");

    let p = |name: &str| params[compiled.param_index[name]];
    let i = int_s.counts[idx(&compiled, "I")] as f64;
    let e = int_s.counts[idx(&compiled, "E")] as f64;
    let r = int_s.counts[idx(&compiled, "R")];
    let w = real_s.values[compiled.global_to_real[compiled.comp_index["W"]].unwrap()];

    let expected = poisson_logpmf(i, p("I0"))
        + negbin_logpmf(e, p("exposed_per_infectious") * i, p("exposed_k"))
        + binom_logpmf(r as u64, p("N0").round() as u64, p("frac_immune"))
        + normal_logpdf(w, p("W0"), p("W0_sd"));

    assert!(lp.is_finite(), "the density must be finite at its own draw, got {lp}");
    assert!(
        (lp - expected).abs() < 1e-9,
        "logpdf {lp} != the sum of the four declared laws {expected}"
    );
    // Non-vacuity: `0.0` was the pre-wiring answer, so a test that would pass
    // against it proves nothing.
    assert!(lp < -5.0, "the density must be a real number, not the old 0.0 stub (got {lp})");

    // The NegBinomial's mean reads the DRAWN `I`, not `I0`. Scoring it against
    // the mean-based `I0` instead gives a different number, which is what makes
    // "evaluate the arguments against x0" a load-bearing claim.
    let against_mean = poisson_logpmf(i, p("I0"))
        + negbin_logpmf(e, p("exposed_per_infectious") * p("I0"), p("exposed_k"))
        + binom_logpmf(r as u64, p("N0").round() as u64, p("frac_immune"))
        + normal_logpdf(w, p("W0"), p("W0_sd"));
    assert!(
        (lp - against_mean).abs() > 1e-9,
        "this draw happens to sit at the mean, so the x0-vs-mean distinction is \
         untested here — reseed the fixture"
    );
}

// ── 3. The gradient ──────────────────────────────────────────────────────────

/// `initial_state_logpdf_grad` against central finite differences of
/// `initial_state_logpdf` at FIXED `x0` — the honest test for a gradient, and
/// the one that catches a term wired into the density but not the derivative
/// (or vice versa).
///
/// Before the laws were wired this returned all zeros; every parameter below
/// has a finite-difference derivative that is not zero, so the comparison is
/// red→green rather than a tautology.
#[test]
fn the_gradient_matches_finite_differences_of_the_density() {
    let (compiled, params) = fixture();
    let mut rng = StatefulRng::new(SEED);
    let (int_s, real_s) = compiled.initial_state_draw(&params, &mut rng).expect("draw");

    let grad = compiled
        .initial_state_logpdf_grad(&int_s.counts, &real_s.values, &params)
        .expect("logpdf_grad");
    assert_eq!(grad.len(), params.len(), "the gradient is in the MODEL parameter basis");

    // Every parameter that enters a law's arguments. `N0` is deliberately
    // absent: it is the Binomial's `n`, which is theta-independent by the
    // `#[differentiate(skip)]` seal, so its emitted derivative is zero BY
    // DESIGN and an FD check on it would be comparing against a discontinuity.
    let checked = ["I0", "exposed_per_infectious", "exposed_k", "frac_immune", "W0", "W0_sd"];
    let mut n_nonzero = 0;
    for name in checked {
        let j = compiled.param_index[name];
        let h = (1e-6 * params[j].abs()).max(1e-9);
        let mut plus = params.clone();
        let mut minus = params.clone();
        plus[j] += h;
        minus[j] -= h;
        let lp_plus = compiled
            .initial_state_logpdf(&int_s.counts, &real_s.values, &plus)
            .expect("logpdf+");
        let lp_minus = compiled
            .initial_state_logpdf(&int_s.counts, &real_s.values, &minus)
            .expect("logpdf-");
        let fd = (lp_plus - lp_minus) / (2.0 * h);

        let tol = 1e-4 * fd.abs().max(1.0);
        assert!(
            (grad[j] - fd).abs() < tol,
            "d(logpdf)/d({name}): analytic {} vs finite difference {fd}",
            grad[j]
        );
        // Non-vacuity per parameter: an FD of ~0 would make the comparison
        // pass against the old all-zeros gradient.
        assert!(
            fd.abs() > 1e-6,
            "{name}'s finite difference is ~0 ({fd}), so this comparison would \
             pass against a gradient that was never wired"
        );
        if grad[j] != 0.0 {
            n_nonzero += 1;
        }
    }
    assert_eq!(n_nonzero, checked.len(), "every checked parameter must carry a gradient");

    // The Binomial's `n` carries no gradient, and that is the seal working, not
    // an omission: `BinomialLikelihood::n` is `#[differentiate(skip)]`.
    // `N0` also reaches `S`'s deterministic entry, which contributes no density,
    // so its total initial-state gradient is exactly zero.
    assert_eq!(
        grad[compiled.param_index["N0"]], 0.0,
        "the Binomial's `n` is theta-independent and must carry no gradient"
    );
}

// ── 4. A law-free model is untouched ─────────────────────────────────────────

/// The three answers for a model whose `init {}` declares no law: the draw
/// consumes nothing, the density is exactly zero, the gradient is exactly zero.
/// This is the claim that keeps 108 golden trajectories byte-identical.
#[test]
fn a_law_free_model_is_unchanged_in_all_three() {
    let mut model = load("ocaml/golden/init_dependency_order.ir.json");
    let preset = model.presets.first().cloned().expect("a scenario");
    for p in &mut model.parameters {
        if let Some(&v) = preset.params.get(&p.name) {
            p.value = p.value.with_value(v);
        }
    }
    let compiled = CompiledModel::new(model).expect("compile");
    let params = compiled.default_params.clone();
    assert!(!compiled.has_init_law);

    let mut rng = StatefulRng::new(SEED);
    let (int_s, real_s) = compiled.initial_state_draw(&params, &mut rng).expect("draw");
    let mut untouched = StatefulRng::new(SEED);
    let a: Vec<f64> = (0..8).map(|_| rng.uniform()).collect();
    let b: Vec<f64> = (0..8).map(|_| untouched.uniform()).collect();
    assert_eq!(a, b, "a law-free draw must consume nothing from the stream");

    assert_eq!(
        compiled
            .initial_state_logpdf(&int_s.counts, &real_s.values, &params)
            .expect("logpdf"),
        0.0
    );
    assert!(compiled
        .initial_state_logpdf_grad(&int_s.counts, &real_s.values, &params)
        .expect("grad")
        .iter()
        .all(|&g| g == 0.0));
}

