//! The B-spline validation harness (proposal 2026-10-27-estimable-bsplines
//! §8.1): R4 regression checks, and the R1–R4 checks of the `smooth`
//! coefficient form (the curvature penalty Ω, its whitening, and its prior).
//!
//! The rest of the harness lives beside the code it checks, and `make
//! validate-bspline` runs all of it, the heavy runs included:
//!
//! - R1 basis values against scipy and R `splines` (clamped uniform, uniform at
//!   absolute time, irregular; degrees 0–5; ends, every breakpoint, between,
//!   outside) and periodic degrees 0–5: `bspline_reference_fixtures.rs`;
//!   against the replaced recursion: `bspline_recursion_oracle.rs`; partition
//!   of unity, non-negativity, end interpolation, constants and the line
//!   reproduced, the Bernstein anchor: `sim::bspline::tests`. Ω against
//!   scipy: `bspline_reference_fixtures.rs`; against 10-point quadrature, its
//!   null space and whitening: here.
//! - R2 gradients against finite differences, forcing level and the
//!   complete-data likelihood: `bspline_gradient_check.rs`.
//! - R3 PGAS+NUTS integration and recovery: `bspline_inference.rs`; the
//!   prior-only run of `smooth`: here; the prior's calibration:
//!   `smooth_prior_calibration.rs`.
//! - R4 the `link` no-collision hash: `runid` `ir_spline_link_changes_hash`;
//!   round-trip and reproducibility: here.
//! - R5 (simulation-based calibration) awaits the maintainer and is not here.

use std::path::PathBuf;

use sim::compiled_model::CompiledModel;
use sim::resolved_expr::{eval_resolved, ResolvedDerivEntry, ResolvedExpr};

const SPLINE_GOLDENS: [&str; 6] = [
    "bspline_uniform",
    "bspline_explicit_dated",
    "bspline_loglink",
    "bspline_param_coefs",
    "periodic_spline_param_coefs",
    "bspline_smooth",
];

fn golden(name: &str) -> String {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "../../../ocaml/golden", &format!("{name}.ir.json")].iter().collect();
    std::fs::read_to_string(p).unwrap()
}

/// R4: the OCaml compiler's JSON for every spline golden parses in Rust, and
/// Rust writes back the same JSON value for every part a spline reaches (the
/// forcing's kind, link, knots and coefficients; each rate and its gradients,
/// with their `forcing_basis` nodes; the observations); and a second Rust
/// round trip changes nothing.
#[test]
fn spline_goldens_round_trip_ocaml_json_rust_json() {
    for name in SPLINE_GOLDENS {
        let ocaml_text = golden(name);
        let ocaml: serde_json::Value = serde_json::from_str(&ocaml_text).unwrap();
        let model = ir::from_str(&ocaml_text).unwrap_or_else(|e| panic!("{name}: Rust cannot parse it: {e}"));
        let rust = serde_json::to_value(&model).unwrap();
        // The two serializers differ elsewhere only in null-versus-absent
        // defaults unrelated to splines (`param_dim`, scenario defaults), so the
        // comparison is over every part a spline reaches.
        for key in ["time_functions", "observations"] {
            assert_eq!(ocaml["model"][key], rust[key], "{name}: `{key}` differs between OCaml and Rust JSON");
        }
        for (o, r) in ocaml["model"]["transitions"].as_array().unwrap().iter()
            .zip(rust["transitions"].as_array().unwrap())
        {
            for key in ["rate", "rate_grad", "rate_state_grad"] {
                assert_eq!(o[key], r[key], "{name}: transition `{}` `{key}` differs", o["name"]);
            }
        }
        // Rust's own round trip. Compared as JSON values, not bytes: the
        // gradient maps are `HashMap`s, so their key order in the text is not
        // stable across instances (general to the IR, not to splines).
        let back: ir::Model = serde_json::from_str(&serde_json::to_string(&model).unwrap()).unwrap();
        assert_eq!(rust, serde_json::to_value(&back).unwrap(), "{name}: Rust JSON changed across a round trip");
        assert_eq!(model, back, "{name}: the model changed across a Rust round trip");
    }
}

/// R4: building a spline model twice gives bitwise-identical forcing values
/// and coefficient gradients (the knots, and everything derived from them, are
/// a deterministic function of the IR).
#[test]
fn spline_models_build_reproducibly() {
    for name in SPLINE_GOLDENS {
        let build = || {
            let mut m = ir::from_str(&golden(name)).unwrap();
            if let Some(preset) = m.presets.first().cloned() {
                for p in &mut m.parameters {
                    if let Some(&v) = preset.params.get(&p.name) {
                        p.value = p.value.with_value(v);
                    }
                }
            }
            CompiledModel::new(m).unwrap()
        };
        let (a, b) = (build(), build());
        let bits = |model: &CompiledModel| -> Vec<u64> {
            let (int_s, real_s) = model.initial_state_mean(&model.default_params).unwrap();
            let mut out = Vec::new();
            for i in 0..=400 {
                let t = -20.0 + 800.0 * i as f64 / 400.0;
                let ctx = sim::propensity::EvalCtx {
                    model, int_s: &int_s, real_s: &real_s, params: &model.default_params,
                    t, dt: 1.0, projected: None, aux: None, int_float_override: None, per_eval: None,
                };
                for tf in 0..model.time_func_cache.len() {
                    out.push(eval_resolved(&ResolvedExpr::TimeFunc(tf), &ctx).to_bits());
                }
                for grads in &model.resolved.rate_grads_indexed {
                    for (_, g) in grads {
                        if let ResolvedDerivEntry::Grad(e) = g {
                            out.push(eval_resolved(e, &ctx).to_bits());
                        }
                    }
                }
            }
            out
        };
        assert_eq!(bits(&a), bits(&b), "{name}: two builds disagree");
    }
}

// ── The smooth coefficient form: Ω, its whitening, and its prior ───────────

use sim::bspline::Knots;
use sim::compiled_model::CompiledTimeFuncKind;
use sim::spline_penalty::{greville, kappa, penalty, penalty_with, Whitening};

type M = Vec<Vec<f64>>;

fn layouts() -> Vec<(&'static str, Knots)> {
    vec![
        ("uniform K=12", Knots::uniform(0.0, 168.0, 9).unwrap()),
        ("irregular K=12", Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap()),
        ("uniform 17 intervals at t~1e3", Knots::uniform(1000.0, 1168.0, 17).unwrap()),
    ]
}

fn dense(m: &sim::spline_penalty::Mat) -> M {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m.at(i, j)).collect()).collect()
}
fn mul(x: &M, y: &M) -> M {
    x.iter().map(|r| (0..y[0].len()).map(|j| r.iter().zip(y).map(|(a, yr)| a * yr[j]).sum()).collect()).collect()
}
fn tr(x: &M) -> M {
    (0..x[0].len()).map(|j| x.iter().map(|r| r[j]).collect()).collect()
}
fn max_abs(x: &M) -> f64 {
    x.iter().flatten().fold(0.0_f64, |a, v| a.max(v.abs()))
}
fn max_diff(x: &M, y: &M) -> f64 {
    x.iter().flatten().zip(y.iter().flatten()).fold(0.0_f64, |a, (u, v)| a.max((u - v).abs()))
}
fn quad(m: &M, x: &[f64], y: &[f64]) -> f64 {
    m.iter().zip(x).map(|(r, xi)| xi * r.iter().zip(y).map(|(a, b)| a * b).sum::<f64>()).sum()
}
/// ‖x‖₂, the largest singular value, by power iteration on xᵀx.
fn spectral(x: &M) -> f64 {
    let g = mul(&tr(x), x);
    let mut v = vec![1.0; g.len()];
    let mut lam = 0.0;
    for _ in 0..500 {
        let w: Vec<f64> = g.iter().map(|r| r.iter().zip(&v).map(|(a, b)| a * b).sum()).collect();
        let n = w.iter().map(|a| a * a).sum::<f64>().sqrt();
        lam = n;
        v = w.iter().map(|a| a / n).collect();
    }
    lam.sqrt()
}

/// R1: Ω by the (p − 1)-point Gauss–Legendre rule against the 10-point rule,
/// ≤ 1e-12·max|Ω|; symmetric; banded with bandwidth p; rank K − 2 (Ω
/// annihilates the null basis N, and Q₊ᵀΩQ₊ is positive definite). The
/// independent reference is scipy (second-derivative splines integrated by
/// adaptive quadrature), `bspline_reference_fixtures.rs`
/// `curvature_penalty_matches_scipy`, in place of R fda, which is not
/// installed (the proposal allows a substitute).
#[test]
fn r1_curvature_penalty_against_references() {
    for (name, k) in layouts() {
        for p in 2..=5 {
            let om = dense(&penalty(&k, p));
            let hi = dense(&penalty_with(&k, p, 10));
            let scale = max_abs(&om);
            let diff = max_diff(&om, &hi);
            assert!(diff <= 1e-12 * scale, "{name} p={p}: exact vs 10-point {diff:e} of {scale:e}");
            for i in 0..om.len() {
                for j in 0..om.len() {
                    assert_eq!(om[i][j].to_bits(), om[j][i].to_bits(), "{name} p={p}: not symmetric");
                    if i.abs_diff(j) > p {
                        assert_eq!(om[i][j], 0.0, "{name} p={p}: ({i},{j}) outside the band");
                    }
                }
            }
            let w = Whitening::new(&k, p).expect("Q₊ᵀΩQ₊ positive definite: rank ≥ K − 2");
            let on = mul(&om, &dense(&w.null));
            let (a, b) = k.window();
            assert!(max_abs(&on) <= 1e-12 * scale * (b - a), "{name} p={p}: ΩN = {:e}: rank > K − 2", max_abs(&on));
        }
    }
}

/// R1: a line has zero curvature penalty: 1ᵀΩ1 and ξᵀΩξ ≤
/// 1e-12·max|Ω|·(b − a)², uniform and irregular knots, degrees 2–5, and a
/// bent vector is penalized (the check is not vacuous).
#[test]
fn r1_a_line_has_zero_curvature_penalty() {
    for (name, k) in layouts() {
        let (a, b) = k.window();
        for p in 2..=5 {
            let om = dense(&penalty(&k, p));
            let scale = max_abs(&om) * (b - a) * (b - a);
            let ones = vec![1.0; om.len()];
            let xi = greville(&k, p);
            let xi_c: Vec<f64> = xi.iter().map(|x| x - 0.5 * (a + b)).collect();
            for (label, v) in [("1", &ones), ("ξ", &xi), ("ξ − t_m", &xi_c)] {
                let q = quad(&om, v, v).abs();
                assert!(q <= 1e-12 * scale, "{name} p={p}: {label}ᵀΩ{label} = {q:e} (scale {scale:e})");
            }
            let bend: Vec<f64> = xi.iter().map(|x| ((x - a) / (b - a)).powi(2)).collect();
            assert!(quad(&om, &bend, &bend) > 1e-6 * max_abs(&om), "{name} p={p}: a bend has no penalty");
        }
    }
}

/// MᵀΩM = I to `tol_mom`; ‖NᵀM‖_max ≤ 1e-12·‖N‖₂‖M‖₂; MMᵀ = Ω⁺ by the four
/// Penrose conditions to 1e-10 relative.
fn check_whitening(degrees: std::ops::RangeInclusive<usize>, tol_mom: f64) {
    for (name, k) in layouts() {
        for p in degrees.clone() {
            let w = Whitening::new(&k, p).unwrap();
            let (om, m, n) = (dense(&w.omega), dense(&w.m), dense(&w.null));
            let d = w.n_modes();
            let eye: M = (0..d).map(|i| (0..d).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
            let e = max_diff(&mul(&mul(&tr(&m), &om), &m), &eye);
            assert!(e <= tol_mom, "{name} p={p}: |MᵀΩM − I| = {e:e}");
            let ntm = max_abs(&mul(&tr(&n), &m));
            assert!(ntm <= 1e-12 * spectral(&n) * spectral(&m), "{name} p={p}: NᵀM = {ntm:e}");
            let x = mul(&m, &tr(&m));
            let rel = |got: &M, want: &M| max_diff(got, want) / max_abs(want);
            let (ax, xa) = (mul(&om, &x), mul(&x, &om));
            assert!(rel(&mul(&ax, &om), &om) <= 1e-10, "{name} p={p}: ΩXΩ ≠ Ω");
            assert!(rel(&mul(&xa, &x), &x) <= 1e-10, "{name} p={p}: XΩX ≠ X");
            assert!(rel(&ax, &tr(&ax)) <= 1e-10, "{name} p={p}: ΩX not symmetric");
            assert!(rel(&xa, &tr(&xa)) <= 1e-10, "{name} p={p}: XΩ not symmetric");
        }
    }
}

/// The induced density of §6.3: for w = a·1 + b(ξ − t_m) + (τ/κ)Mz, the
/// density through (a, b, z) — p(a)p(b)Πφ(zᵢ)·τ^{−(K−2)}, the Jacobian
/// being constant apart from τ — equals the improper Gaussian computed from w
/// alone: p(a(w))p(b(w))·τ^{−(K−2)}·exp(−κ²wᵀΩw/2τ²), with (a(w), b(w)) the
/// orthogonal projection of w onto null(Ω). Compared at 100 random points;
/// the log-densities agree to 1e-10 after removing their constant offset.
fn check_induced_density(name: &str, k: &Knots) {
    use rand::SeedableRng;
    use rand_distr::{Distribution, Exp, Normal};
    let w = Whitening::new(k, 3).unwrap();
    let (kk, d) = (w.n_basis(), w.n_modes());
    let (om, m, n) = (dense(&w.omega), dense(&w.m), dense(&w.null));
    let kap = kappa(kk, 7.0);
    let ln_n = |x: f64, mu: f64, sd: f64| -0.5 * ((x - mu) / sd).powi(2) - sd.ln();
    let (a_mu, a_sd, b_sd) = (-1.9, 0.6, 0.005);
    // (NᵀN)⁻¹Nᵀ, 2 × K.
    let g = mul(&tr(&n), &n);
    let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    let ginv = vec![vec![g[1][1] / det, -g[0][1] / det], vec![-g[1][0] / det, g[0][0] / det]];
    let proj = mul(&ginv, &tr(&n));
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(61);
    let mut offsets = Vec::new();
    for _ in 0..100 {
        let a = Normal::new(a_mu, a_sd).unwrap().sample(&mut rng);
        let b = Normal::new(0.0, b_sd).unwrap().sample(&mut rng);
        let tau = Exp::new(5.99).unwrap().sample(&mut rng);
        let z: Vec<f64> = (0..d).map(|_| Normal::new(0.0, 1.0).unwrap().sample(&mut rng)).collect();
        let wv: Vec<f64> = (0..kk).map(|i| a + b * (w.xi[i] - w.t_mid) + tau / kap * (0..d).map(|r| m[i][r] * z[r]).sum::<f64>()).collect();
        let via_z = ln_n(a, a_mu, a_sd) + ln_n(b, 0.0, b_sd) - 0.5 * z.iter().map(|v| v * v).sum::<f64>()
            - (d as f64) * tau.ln();
        let ab: Vec<f64> = proj.iter().map(|r| r.iter().zip(&wv).map(|(x, y)| x * y).sum()).collect();
        // wᵀΩw on the range part w − N(a(w), b(w)), which equals it exactly
        // (ΩN = 0); the full w carries the line, of order 2, whose
        // cancellation in wᵀΩw is amplified by κ²/τ² for small τ.
        let perp: Vec<f64> = (0..kk).map(|i| wv[i] - n[i][0] * ab[0] - n[i][1] * ab[1]).collect();
        let via_w = ln_n(ab[0], a_mu, a_sd) + ln_n(ab[1], 0.0, b_sd) - (d as f64) * tau.ln()
            - kap * kap * quad(&om, &perp, &perp) / (2.0 * tau * tau);
        offsets.push(via_w - via_z);
    }
    let c = offsets[0];
    let spread = offsets.iter().fold(0.0_f64, |s, o| s.max((o - c).abs()));
    assert!(spread <= 1e-10, "{name}: log-densities differ by a non-constant {spread:e}");
}

/// R1, the whitening: degrees 2–4 at the proposal's 1e-12 for MᵀΩM = I;
/// degree 5, the largest allowed and worst-conditioned, reaches 1.23e-12 with
/// 22 basis functions at t ≈ 10³, so it is pinned at 2e-12 (the proposal
/// states no separate bound). Then the induced density at 100 points, on
/// uniform and irregular knots.
#[test]
fn r1_whitening_identities() {
    check_whitening(2..=4, 1e-12);
    check_whitening(5..=5, 2e-12);
    for (name, k) in layouts() {
        check_induced_density(name, &k);
    }
}

/// R2 for `smooth`: the penalty ½wᵀΩw has gradient Ωw (central differences,
/// step 1e-5·max(|w|, 1)); the map w(a, b, τ, z) has Jacobian
/// [1, ξ − t_m, Mz/κ, (τ/κ)M] (same step). The forcing value and the
/// complete-data likelihood through `smooth` are checked in
/// `bspline_gradient_check.rs` (`smooth_gradients_*`).
#[test]
fn r2_smooth_gradients() {
    let k = Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap();
    let wh = Whitening::new(&k, 3).unwrap();
    let om = dense(&wh.omega);
    let w: Vec<f64> = (0..om.len()).map(|i| ((i * 5 + 2) % 7) as f64 / 3.0 - 1.0).collect();
    let half = |v: &[f64]| 0.5 * quad(&om, v, v);
    let tol = |an: f64, fd: f64, scale: f64| (an - fd).abs() <= 1e-4 * an.abs().max(1e-3 * scale);
    for i in 0..om.len() {
        let h = 1e-5 * w[i].abs().max(1.0);
        let (mut up, mut dn) = (w.clone(), w.clone());
        up[i] += h;
        dn[i] -= h;
        let fd = (half(&up) - half(&dn)) / (2.0 * h);
        let an: f64 = om[i].iter().zip(&w).map(|(a, b)| a * b).sum();
        assert!(tol(an, fd, max_abs(&om)), "penalty component {i}: {an} vs {fd}");
    }
    let (kk, d) = (wh.n_basis(), wh.n_modes());
    let kap = kappa(kk, 7.0);
    let map = |th: &[f64]| -> Vec<f64> {
        (0..kk).map(|i| th[0] + th[1] * (wh.xi[i] - wh.t_mid)
            + th[2] / kap * (0..d).map(|r| wh.m.at(i, r) * th[3 + r]).sum::<f64>()).collect()
    };
    let mut th = vec![-1.9, 0.004, 0.3];
    th.extend((0..d).map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0));
    for c in 0..th.len() {
        let h = 1e-5 * th[c].abs().max(1.0);
        let (mut up, mut dn) = (th.clone(), th.clone());
        up[c] += h;
        dn[c] -= h;
        let (wu, wd) = (map(&up), map(&dn));
        for i in 0..kk {
            let fd = (wu[i] - wd[i]) / (2.0 * h);
            let an = match c {
                0 => 1.0,
                1 => wh.xi[i] - wh.t_mid,
                2 => (0..d).map(|r| wh.m.at(i, r) * th[3 + r]).sum::<f64>() / kap,
                _ => th[2] / kap * wh.m.at(i, c - 3),
            };
            assert!(tol(an, fd, 1.0 / kap), "∂w{i}/∂θ{c}: {an} vs {fd}");
        }
    }
}

/// Batch-means MCSE of the mean of `x`, 200 batches (the
/// `nuts_stationarity.rs` rule).
fn mcse(x: &[f64]) -> f64 {
    let nb = 200usize;
    let b = x.len() / nb;
    let means: Vec<f64> = (0..nb).map(|k| x[k * b..(k + 1) * b].iter().sum::<f64>() / b as f64).collect();
    let m = means.iter().sum::<f64>() / nb as f64;
    (means.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (nb - 1) as f64 / nb as f64).sqrt()
}

/// Kolmogorov–Smirnov p-value of `x` against the CDF `f` (asymptotic
/// Kolmogorov distribution with Stephens' small-sample correction).
fn ks_p(x: &mut [f64], f: impl Fn(f64) -> f64) -> f64 {
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = x.len() as f64;
    let dn = x.iter().enumerate().fold(0.0_f64, |m, (i, &v)| {
        let c = f(v);
        m.max((c - i as f64 / n).abs()).max(((i + 1) as f64 / n - c).abs())
    });
    let l = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * dn;
    (1..=100).map(|j| 2.0 * (-1.0_f64).powi(j - 1) * (-2.0 * (j as f64 * l).powi(2)).exp()).sum::<f64>().clamp(0.0, 1.0)
}

/// ∫ g″(t)² dt over the window, g = ln f from the runtime's own value path
/// at parameters `params`: on each knot interval, two-point Gauss–Legendre
/// (exact, g″ being linear for a cubic) of the central second difference
/// with step `h` (exact for a cubic up to rounding).
fn runtime_curvature(model: &CompiledModel, params: &[f64], knots: &[f64]) -> f64 {
    let (int_s, real_s) = model.initial_state_mean(params).unwrap();
    let g = |t: f64| {
        let ctx = sim::propensity::EvalCtx {
            model, int_s: &int_s, real_s: &real_s, params, t, dt: 1.0,
            projected: None, aux: None, int_float_override: None, per_eval: None,
        };
        eval_resolved(&ResolvedExpr::TimeFunc(0), &ctx).ln()
    };
    let h = 0.25;
    let node = 1.0 / 3.0_f64.sqrt();
    knots.windows(2).map(|w| {
        let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
        [mid - half * node, mid + half * node].iter()
            .map(|&t| ((g(t + h) - 2.0 * g(t) + g(t - h)) / (h * h)).powi(2) * half)
            .sum::<f64>()
    }).sum()
}

/// Post-adaptation divergences allowed over the 4 × 4,000 prior-only
/// transitions: one per chain, against 0 measured.
const MAX_PRIOR_DIVERGENCES: usize = 4;

/// R3, required for merge (the maintainer's D7): PGAS+NUTS on the prior
/// alone. `prior_only_smooth` is `X --> D @ f·X` with X0 = 0 and one
/// observation stream whose every value is missing, so every flow is zero,
/// the complete-data likelihood is constant, and the sampler must draw the
/// stated prior. The level, slope and modes are unbounded and sampled on the
/// identity scale, τ on the log scale, as `fit run` does. 4 chains × 4,000
/// kept sweeps after 1,000 adaptation sweeps. Checked:
///
/// - τ against exponential(λ = 5.99) by KS on draws thinned by 10;
/// - each standardized innovation zᵢ: mean 0 and variance 1 (z-scores on the
///   batch-means MCSE, |z| < 5, the `nuts_stationarity.rs` rule) and N(0, 1)
///   by KS on thinned draws; a and b likewise against their normals;
/// - the implied covariance of every pair of coefficients,
///   cov(w) = Var(a)·11ᵀ + Var(b)·(ξ − t_m)(ξ − t_m)ᵀ + (E[τ²]/κ²)·Ω⁺ with
///   E[τ²] = 2/λ², and the mean E[w] = E[a]·1, as z-scores;
/// - the integrated curvature of the curve the runtime evaluates: given τ,
///   S = (K − 2)·T_ref³·∫g″²/τ² ~ χ²_{K−2}, by KS on thinned draws, and per
///   draw S = ‖z‖² to 1e-6 (the identity that pins κ and M in the value
///   path the prior acts on);
/// - at most [`MAX_PRIOR_DIVERGENCES`] divergences after adaptation.
#[test]
#[ignore = "about a minute in release; run by `make validate-bspline`"]
fn r3_prior_only_run_draws_the_smooth_prior() {
    use sim::inference::if2::{EstimatedParam, Transform};
    use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
    use sim::inference::particle_filter::Observation;
    use sim::inference::pgas::{run_pgas, PGASConfig};
    use sim::inference::pmmh::Prior;
    use sim::inference::prior::Density;
    use sim::inference::{normal_cdf, BoundObs, MultiStreamObsModel};
    use std::sync::Arc;

    let path = format!("{}/../../../tests/fixtures/bspline_inference/ir/prior_only_smooth.ir.json", env!("CARGO_MANIFEST_DIR"));
    let mut ir_m = ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for p in &mut ir_m.parameters {
        p.value = p.value.with_value(if p.name == "tau" { 0.2 } else if p.name == "a" { -1.0 } else { 0.0 });
    }
    let model = Arc::new(CompiledModel::new(ir_m).unwrap());
    let (lambda, t_ref, a_mu, a_sd, b_sd) = (5.99, 7.0, -1.0, 0.6, 0.01);
    let (wh, kap) = match &model.time_func_cache[0].kind {
        CompiledTimeFuncKind::BSpline(b) => (b.whitening().unwrap().clone(), kappa(b.n_basis(), t_ref)),
        _ => panic!("the fixture's forcing is a bspline"),
    };
    let (kk, d) = (wh.n_basis(), wh.n_modes());
    let breakpoints = [0.0, 20.0, 40.0, 60.0]; // window [0, 60], K = 6, cubic

    let est_p = |name: &str, transform: Transform, lo: f64, hi: f64, init: f64| EstimatedParam {
        name: name.into(), index: model.param_index[name], initial: init, rw_sd: 0.05, transform,
        lower: lo, upper: hi, rw_sd_auto: false, perturb_only_at_t0: false,
    };
    let inf = f64::INFINITY;
    let mut est = vec![
        est_p("a", Transform::None, -inf, inf, a_mu),
        est_p("b", Transform::None, -inf, inf, 0.0),
        est_p("tau", Transform::Log { lo: 0.0, hi: f64::INFINITY }, 0.0, f64::INFINITY, 0.2),
    ];
    let mut priors = vec![
        Prior::Fixed(Density::Normal { mean: a_mu, sd: a_sd }),
        Prior::Fixed(Density::Normal { mean: 0.0, sd: b_sd }),
        Prior::Fixed(Density::Exponential { rate: lambda }),
    ];
    for i in 1..=d {
        est.push(est_p(&format!("z_m{i}"), Transform::None, -inf, inf, 0.0));
        priors.push(Prior::Fixed(Density::Normal { mean: 0.0, sd: 1.0 }));
    }
    let times: Vec<f64> = (1..=60).map(|t| t as f64).collect();
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    // The one stream, every value missing (`None`): no likelihood term.
    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, &model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.clone()).unwrap(),
            projection,
            ir_model: om.clone(),
            observations: vec![None; times.len()],
            aux: vec![],
        }
    }).collect();
    let obs_model = MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap();
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling: true,
        n_particles: 4, n_sweeps: 5000, burn_in: 1000, thin: 1, dt: 1.0,
        use_nuts: true, dense_mass: true, max_tree_depth: 8,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };
    // Per chain, the full parameter vector of every kept sweep.
    let runs: Vec<(Vec<Vec<f64>>, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (1..=4u64).map(|seed| {
            let (model, est, priors, observations, obs_model, config) = (&model, &est, &priors, &observations, &obs_model, &config);
            sc.spawn(move || {
                let mut base = model.default_params.clone();
                for p in est {
                    base[p.index] = p.initial;
                }
                let r = run_pgas(model, est, priors, &base, config, observations, obs_model, seed, None, None,
                    "bspline-r3-prior".into()).expect("prior-only PGAS-NUTS run");
                eprintln!("chain {seed}: {} divergent ({} after adaptation)", r.n_divergent_total, r.n_divergent_post_burn);
                (r.sweeps.iter().map(|s| s.params.clone()).collect::<Vec<_>>(), r.n_divergent_post_burn)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // A Gaussian-and-log-exponential target with no box on τ: the kernel
    // diverges essentially never (0 per chain measured; a box [1e-6, 2.5] on
    // τ gave 10–31, its walls counted as divergences).
    let divergent: usize = runs.iter().map(|r| r.1).sum();
    assert!(divergent <= MAX_PRIOR_DIVERGENCES, "{divergent} divergences after adaptation on the prior alone");
    let runs: Vec<Vec<Vec<f64>>> = runs.into_iter().map(|r| r.0).collect();
    assert!(runs.iter().all(|c| c.len() == 4000), "4000 kept sweeps per chain");
    let ix = |n: &str| model.param_index[n];
    let zi: Vec<usize> = (1..=d).map(|i| ix(&format!("z_m{i}"))).collect();
    let pooled: Vec<&Vec<f64>> = runs.iter().flatten().collect();
    let thinned: Vec<&Vec<f64>> = runs.iter().flat_map(|c| c.iter().step_by(10)).collect();
    let col = |f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> { pooled.iter().map(|v| f(v)).collect() };
    let mut worst: f64 = 0.0;
    let mut score = |label: &str, x: Vec<f64>, exact: f64| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let z = (m - exact) / mcse(&x);
        worst = worst.max(z.abs());
        assert!(z.abs() < 5.0, "{label}: mean {m:.5} vs {exact:.5}, z = {z:.2}");
    };
    let mut ks_worst: f64 = 1.0;
    let mut ks = |label: &str, f: &dyn Fn(&[f64]) -> f64, cdf: &dyn Fn(f64) -> f64| {
        let mut x: Vec<f64> = thinned.iter().map(|v| f(v)).collect();
        let p = ks_p(&mut x, cdf);
        ks_worst = ks_worst.min(p);
        assert!(p > 1e-3, "{label}: KS p = {p:e}");
    };

    // τ, a, b and each standardized innovation.
    ks("τ ~ exponential", &|v| v[ix("tau")], &|x| 1.0 - (-lambda * x).exp());
    score("E[τ]", col(&|v| v[ix("tau")]), 1.0 / lambda);
    score("E[a]", col(&|v| v[ix("a")]), a_mu);
    score("Var[a]", col(&|v| (v[ix("a")] - a_mu).powi(2)), a_sd * a_sd);
    score("E[b]", col(&|v| v[ix("b")]), 0.0);
    score("Var[b]", col(&|v| v[ix("b")].powi(2)), b_sd * b_sd);
    for (i, &k) in zi.iter().enumerate() {
        score(&format!("E[z{i}]"), col(&|v| v[k]), 0.0);
        score(&format!("Var[z{i}]"), col(&|v| v[k].powi(2)), 1.0);
        ks(&format!("z{i} ~ N(0, 1)"), &|v| v[k], &|x| normal_cdf(x));
    }

    // The implied coefficients and their full covariance.
    let w_of = |v: &[f64], j: usize| v[ix("a")] + v[ix("b")] * (wh.xi[j] - wh.t_mid)
        + v[ix("tau")] / kap * (0..d).map(|r| wh.m.at(j, r) * v[zi[r]]).sum::<f64>();
    let m = dense(&wh.m);
    let omega_plus = mul(&m, &tr(&m));
    let e_tau2 = 2.0 / (lambda * lambda);
    for j in 0..kk {
        score(&format!("E[w{j}]"), col(&|v| w_of(v, j)), a_mu);
        for l in j..kk {
            let (dj, dl) = (wh.xi[j] - wh.t_mid, wh.xi[l] - wh.t_mid);
            let exact = a_sd * a_sd + b_sd * b_sd * dj * dl + e_tau2 / (kap * kap) * omega_plus[j][l];
            score(&format!("cov(w{j}, w{l})"), col(&|v| (w_of(v, j) - a_mu) * (w_of(v, l) - a_mu)), exact);
        }
    }

    // The integrated curvature of the runtime's curve, against its law.
    let s_of = |v: &[f64]| (kk - 2) as f64 * t_ref.powi(3) * runtime_curvature(&model, v, &breakpoints) / v[ix("tau")].powi(2);
    let mut worst_identity: f64 = 0.0;
    for v in thinned.iter().take(400) {
        let zz: f64 = zi.iter().map(|&k| v[k] * v[k]).sum();
        let rel = (s_of(v) - zz).abs() / zz;
        worst_identity = worst_identity.max(rel);
    }
    assert!(worst_identity <= 1e-6, "(K − 2)·T_ref³·∫g″²/τ² differs from ‖z‖² by {worst_identity:e} relative");
    let dof = (kk - 2) as f64;
    assert_eq!(dof, 4.0, "the χ² CDF below is for 4 degrees of freedom");
    ks("(K − 2)T_ref³∫g″²/τ² ~ χ²₄", &s_of, &|x| 1.0 - (-x / 2.0).exp() * (1.0 + x / 2.0));
    eprintln!("prior-only: worst |z| = {worst:.2}, smallest KS p = {ks_worst:.3}, \
               worst curvature identity {worst_identity:.1e}");
}

/// R4: the whitening M is bitwise identical across two builds of one model,
/// and across compiler builds: an FNV-1a fingerprint of M for the irregular
/// cubic layout is pinned, which fixes the construction itself (column order,
/// reflector signs, the Cholesky and the solves), not only its identities,
/// which another valid construction would also satisfy. Only + − × ÷ and sqrt
/// are used, all correctly rounded under IEEE 754, so the value is portable.
#[test]
fn r4_whitening_is_reproducible() {
    const PINNED_M: u64 = 0xb9f928f84765b44c;
    let fingerprint = |w: &Whitening| {
        w.m.data.iter().fold(0xcbf29ce484222325u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x100000001b3))
    };
    let k = Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap();
    let fp = fingerprint(&Whitening::new(&k, 3).unwrap());
    assert_eq!(fp, PINNED_M, "M changed: {fp:#018x}");

    let path = format!("{}/../../../tests/fixtures/gradient/ir/bspline_smooth_grad.ir.json", env!("CARGO_MANIFEST_DIR"));
    let build = || {
        let mut m = ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for p in &mut m.parameters {
            p.value = p.value.with_value(0.1);
        }
        CompiledModel::new(m).unwrap()
    };
    let ms = |m: &CompiledModel| -> Vec<u64> {
        m.time_func_cache.iter().filter_map(|tf| match &tf.kind {
            CompiledTimeFuncKind::BSpline(b) => b.whitening().map(fingerprint),
            _ => None,
        }).collect()
    };
    let (a, b) = (ms(&build()), ms(&build()));
    assert_eq!(a.len(), 2, "both forcings of the fixture are smooth");
    assert_eq!(a, b, "two builds of one model give different M");
}

/// R3 for explicit coefficients, required for merge of the explicit form on
/// its own: PGAS+NUTS on `prior_only_explicit` (X --> D @ f·X, X0 = 0, one
/// all-missing stream, six log-link coefficients with independent
/// normal(−1.5, 1) priors, unbounded on the identity scale as `fit run`
/// samples them). 4 chains × 4,000 kept sweeps after 1,000 adaptation.
/// Checked: each coefficient's mean, variance and N(−1.5, 1) law (z-scores
/// on batch-means MCSE |z| < 5; KS on draws thinned by 10, p > 1e-3); every
/// pairwise covariance against 0; log f(t) from the runtime's value path at
/// five times against its law, mean −1.5 and variance Σₖ Bₖ(t)²; and at
/// most [`MAX_PRIOR_DIVERGENCES`] divergences after adaptation.
#[test]
#[ignore = "about a minute in release; run by `make validate-bspline`"]
fn r3_prior_only_run_draws_the_explicit_prior() {
    use sim::inference::if2::{EstimatedParam, Transform};
    use sim::inference::multi_stream_obs::{StreamProjection, StreamSpec, StreamTimes};
    use sim::inference::particle_filter::Observation;
    use sim::inference::pgas::{run_pgas, PGASConfig};
    use sim::inference::pmmh::Prior;
    use sim::inference::prior::Density;
    use sim::inference::{normal_cdf, BoundObs, MultiStreamObsModel};
    use std::sync::Arc;

    let path = format!("{}/../../../tests/fixtures/bspline_inference/ir/prior_only_explicit.ir.json", env!("CARGO_MANIFEST_DIR"));
    let mut ir_m = ir::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    for p in &mut ir_m.parameters {
        p.value = p.value.with_value(-1.5);
    }
    let model = Arc::new(CompiledModel::new(ir_m).unwrap());
    let names: Vec<String> = (1..=6).map(|i| format!("b{i}")).collect();
    let inf = f64::INFINITY;
    let est: Vec<EstimatedParam> = names.iter().map(|n| EstimatedParam {
        name: n.clone(), index: model.param_index[n.as_str()], initial: -1.5, rw_sd: 0.05,
        transform: Transform::None, lower: -inf, upper: inf, rw_sd_auto: false, perturb_only_at_t0: false,
    }).collect();
    let priors = vec![Prior::Fixed(Density::Normal { mean: -1.5, sd: 1.0 }); 6];
    let times: Vec<f64> = (1..=60).map(|t| t as f64).collect();
    let observations: Vec<Observation> = times.iter().map(|&t| Observation { time: t, value: 0.0 }).collect();
    let specs: Vec<StreamSpec> = model.model.observations.iter().map(|om| {
        let projection = StreamProjection::from_ir(&om.projection, &model, &om.name).unwrap();
        StreamSpec {
            times: StreamTimes::contiguous_for(&projection, 0.0, times.clone()).unwrap(),
            projection, ir_model: om.clone(), observations: vec![None; times.len()], aux: vec![],
        }
    }).collect();
    let obs_model = MultiStreamObsModel::new(BoundObs::bind(0.0, specs).unwrap().0, model.clone()).unwrap();
    let config = PGASConfig {
        binomial: sim::rng::BinomialAlgorithm::Btpe, ancestor_sampling: true,
        n_particles: 4, n_sweeps: 5000, burn_in: 1000, thin: 1, dt: 1.0,
        use_nuts: true, dense_mass: true, max_tree_depth: 8,
        trajectory_warmup: 0, csmc_sweeps_per_nuts: 1, step_policy: sim::schedule::StepPolicy::Snap,
    };
    let runs: Vec<(Vec<Vec<f64>>, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (1..=4u64).map(|seed| {
            let (model, est, priors, observations, obs_model, config) = (&model, &est, &priors, &observations, &obs_model, &config);
            sc.spawn(move || {
                let mut base = model.default_params.clone();
                for p in est {
                    base[p.index] = p.initial;
                }
                let r = run_pgas(model, est, priors, &base, config, observations, obs_model, seed, None, None,
                    "bspline-r3-prior-explicit".into()).expect("prior-only PGAS-NUTS run");
                eprintln!("chain {seed}: {} divergent ({} after adaptation)", r.n_divergent_total, r.n_divergent_post_burn);
                (r.sweeps.iter().map(|s| s.params.clone()).collect::<Vec<_>>(), r.n_divergent_post_burn)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let divergent: usize = runs.iter().map(|r| r.1).sum();
    assert!(divergent <= MAX_PRIOR_DIVERGENCES, "{divergent} divergences after adaptation on the prior alone");
    let runs: Vec<Vec<Vec<f64>>> = runs.into_iter().map(|r| r.0).collect();
    assert!(runs.iter().all(|c| c.len() == 4000), "4000 kept sweeps per chain");
    let ix: Vec<usize> = names.iter().map(|n| model.param_index[n.as_str()]).collect();
    let pooled: Vec<&Vec<f64>> = runs.iter().flatten().collect();
    let thinned: Vec<&Vec<f64>> = runs.iter().flat_map(|c| c.iter().step_by(10)).collect();
    let col = |f: &dyn Fn(&[f64]) -> f64| -> Vec<f64> { pooled.iter().map(|v| f(v)).collect() };
    let mut worst: f64 = 0.0;
    let mut score = |label: &str, x: Vec<f64>, exact: f64| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let z = (m - exact) / mcse(&x);
        worst = worst.max(z.abs());
        assert!(z.abs() < 5.0, "{label}: mean {m:.5} vs {exact:.5}, z = {z:.2}");
    };
    let mut ks_worst: f64 = 1.0;
    for (j, &k) in ix.iter().enumerate() {
        score(&format!("E[b{}]", j + 1), col(&|v| v[k]), -1.5);
        score(&format!("Var[b{}]", j + 1), col(&|v| (v[k] + 1.5).powi(2)), 1.0);
        for &l in &ix[j + 1..] {
            score(&format!("cov(b{}, ·)", j + 1), col(&|v| (v[k] + 1.5) * (v[l] + 1.5)), 0.0);
        }
        let mut x: Vec<f64> = thinned.iter().map(|v| v[k]).collect();
        let p = ks_p(&mut x, |y| normal_cdf(y + 1.5));
        ks_worst = ks_worst.min(p);
        assert!(p > 1e-3, "b{} ~ N(-1.5, 1): KS p = {p:e}", j + 1);
    }
    // log f(t) through the runtime: mean −1.5, variance Σₖ Bₖ(t)².
    let knots = Knots::uniform(0.0, 60.0, 3).unwrap();
    for t in [0.0, 17.0, 30.0, 45.0, 60.0] {
        let lf = |v: &[f64]| -> f64 {
            let (int_s, real_s) = model.initial_state_mean(v).unwrap();
            let ctx = sim::propensity::EvalCtx {
                model: &model, int_s: &int_s, real_s: &real_s, params: v, t, dt: 1.0,
                projected: None, aux: None, int_float_override: None, per_eval: None,
            };
            eval_resolved(&ResolvedExpr::TimeFunc(0), &ctx).ln()
        };
        let var: f64 = (0..6).map(|k| sim::bspline::basis(&knots, 3, t, k).powi(2)).sum();
        score(&format!("E[log f({t})]"), col(&lf), -1.5);
        score(&format!("Var[log f({t})]"), col(&|v| (lf(v) + 1.5).powi(2)), var);
    }
    eprintln!("explicit prior-only: worst |z| = {worst:.2}, smallest KS p = {ks_worst:.3}, {divergent} divergences");
}
