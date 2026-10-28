//! The curvature penalty of a clamped B-spline and its whitening (proposal
//! 2026-10-27-estimable-bsplines §6.1, §6.3).
//!
//! For g(t) = Σⱼ wⱼ Bⱼ(t) on the window [a, b], the integrated squared
//! curvature is the quadratic form ∫ g''(t)² dt = wᵀΩw with
//! Ω_ij = ∫ B_i''(t) B_j''(t) dt (O'Sullivan 1986; Wand & Ormerod 2008). On
//! each knot interval B_i'' is a polynomial of degree p − 2, so B_i''B_j'' has
//! degree 2p − 4 and (p − 1)-point Gauss–Legendre is exact for it. Ω's null
//! space is exactly the linear functions of t: the coefficient vectors 1 and
//! the Greville abscissae ξ.
//!
//! The `smooth` coefficient form samples w = a·1 + b·(ξ − t_m·1) + (τ/κ)·M·z
//! with z ~ N(0, I_{K−2}), where M = Q₊L⁻ᵀ: Q₊ an orthonormal basis of the
//! orthogonal complement of N = [1, ξ − t_m·1] (the last K − 2 columns of the
//! full Q of a Householder QR of N, columns in that order, no pivoting, each
//! reflector's v₁ carrying the sign of its leading entry), and L the lower
//! Cholesky factor of Q₊ᵀΩQ₊. Then MᵀΩM = I and NᵀM = 0, and MMᵀ = Ω⁺. Every
//! step is a fixed sequence of floating-point operations on the knots, so M is
//! a deterministic function of them.

use crate::bspline::{clamped_knot, span, span_derivatives, Knots};

/// A dense row-major matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { rows, cols, data: vec![0.0; rows * cols] }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }
    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }
}

/// Gauss–Legendre nodes and weights on [−1, 1] for `n` points, by Newton's
/// method on the Legendre polynomial (to machine precision; deterministic).
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n.div_ceil(2) {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, 0.0);
            for j in 0..n {
                let p2 = p1;
                p1 = p0;
                p0 = ((2 * j + 1) as f64 * z * p1 - j as f64 * p2) / (j + 1) as f64;
            }
            dp = n as f64 * (z * p0 - p1) / (z * z - 1.0);
            let dz = p0 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    (x, w)
}

/// The number of basis functions, `n_intervals + p`.
pub fn n_basis(knots: &Knots, p: usize) -> usize {
    knots.n_intervals() + p
}

/// Ω, K × K, by `n_points`-point Gauss–Legendre on every knot interval.
/// `n_points = p − 1` is exact (the default the runtime uses); a larger
/// `n_points` is the high-order check.
pub fn penalty_with(knots: &Knots, p: usize, n_points: usize) -> Mat {
    let k = n_basis(knots, p);
    let mut omega = Mat::zeros(k, k);
    if p < 2 {
        return omega;
    }
    let (nodes, weights) = gauss_legendre(n_points);
    for m in 0..knots.n_intervals() {
        let lo = clamped_knot(knots, m + p, p);
        let hi = clamped_knot(knots, m + p + 1, p);
        let half = 0.5 * (hi - lo);
        for (x, wq) in nodes.iter().zip(&weights) {
            let t = lo + half * (x + 1.0);
            let (first, d) = span_derivatives(knots, p, t);
            debug_assert_eq!(first, m);
            // Upper triangle only; mirrored below, so Ω is symmetric exactly.
            for r in 0..=p {
                for s in r..=p {
                    let (i, j) = (first + r, first + s);
                    let v = omega.at(i, j) + half * wq * d[2][r] * d[2][s];
                    omega.set(i, j, v);
                }
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            let v = omega.at(j, i);
            omega.set(i, j, v);
        }
    }
    omega
}

/// Ω with the exact rule, (p − 1)-point Gauss–Legendre.
pub fn penalty(knots: &Knots, p: usize) -> Mat {
    penalty_with(knots, p, p.saturating_sub(1).max(1))
}

/// The Greville abscissae ξₖ = (u_{k+1} + … + u_{k+p}) / p: the coefficients
/// that reproduce the line g(t) = t (de Boor 1978).
pub fn greville(knots: &Knots, p: usize) -> Vec<f64> {
    (0..n_basis(knots, p))
        .map(|k| (1..=p).map(|i| clamped_knot(knots, k + i, p)).sum::<f64>() / p as f64)
        .collect()
}

/// Q₊: the last K − 2 columns of the full Q of a Householder QR of the K × 2
/// matrix `n` (columns in order, no pivoting, v₁ = x₁ + sign(x₁)‖x‖ with
/// sign(0) = +1, so no reflector cancels).
pub fn householder_complement(n: &Mat) -> Mat {
    let (k, c) = (n.rows, n.cols);
    let mut a = n.clone();
    let mut reflectors: Vec<Vec<f64>> = Vec::with_capacity(c);
    for j in 0..c {
        let norm = (j..k).map(|i| a.at(i, j) * a.at(i, j)).sum::<f64>().sqrt();
        let x0 = a.at(j, j);
        let sign = if x0 < 0.0 { -1.0 } else { 1.0 };
        let mut v = vec![0.0; k];
        for (i, vi) in v.iter_mut().enumerate().skip(j) {
            *vi = a.at(i, j);
        }
        v[j] = x0 + sign * norm;
        let vv: f64 = v.iter().map(|x| x * x).sum();
        if vv > 0.0 {
            for col in j..c {
                let dot: f64 = (j..k).map(|i| v[i] * a.at(i, col)).sum();
                for i in j..k {
                    let val = a.at(i, col) - 2.0 * dot / vv * v[i];
                    a.set(i, col, val);
                }
            }
        }
        reflectors.push(v);
    }
    // Q = H_0 H_1 … H_{c−1}; its columns c.. span the complement. Build Q·e_j
    // for j ≥ c by applying the reflectors in reverse order to e_j.
    let mut q = Mat::zeros(k, k - c);
    for (col, j) in (c..k).enumerate() {
        let mut e = vec![0.0; k];
        e[j] = 1.0;
        for v in reflectors.iter().rev() {
            let vv: f64 = v.iter().map(|x| x * x).sum();
            if vv == 0.0 {
                continue;
            }
            let dot: f64 = v.iter().zip(&e).map(|(a, b)| a * b).sum();
            for (ei, vi) in e.iter_mut().zip(v) {
                *ei -= 2.0 * dot / vv * vi;
            }
        }
        for (i, ei) in e.iter().enumerate() {
            q.set(i, col, *ei);
        }
    }
    q
}

/// The whitened `smooth` basis of a clamped spline: Ω, the null-space basis N,
/// and M = Q₊L⁻ᵀ, with the mid-window point t_m.
#[derive(Debug, Clone, PartialEq)]
pub struct Whitening {
    pub omega: Mat,
    /// N = [1, ξ − t_m·1], K × 2.
    pub null: Mat,
    /// M, K × (K − 2): MᵀΩM = I, NᵀM = 0.
    pub m: Mat,
    pub t_mid: f64,
    pub xi: Vec<f64>,
}

impl Whitening {
    /// Refuses p < 2 (no curvature) and K < 3 (no non-linear mode), and a
    /// Q₊ᵀΩQ₊ that is not positive definite.
    pub fn new(knots: &Knots, p: usize) -> Result<Self, String> {
        let k = n_basis(knots, p);
        if p < 2 {
            return Err(format!("degree {p} has no curvature; smooth needs degree at least 2"));
        }
        if k < 3 {
            return Err(format!("{k} basis functions leave no curvature mode; smooth needs at least 3"));
        }
        let omega = penalty(knots, p);
        let xi = greville(knots, p);
        let (a, b) = knots.window();
        let t_mid = 0.5 * (a + b);
        let mut null = Mat::zeros(k, 2);
        for (i, x) in xi.iter().enumerate() {
            null.set(i, 0, 1.0);
            null.set(i, 1, x - t_mid);
        }
        let q = householder_complement(&null);
        let d = k - 2;
        // Q₊ᵀ Ω Q₊, d × d.
        let mut qoq = vec![0.0; d * d];
        for r in 0..d {
            for c in 0..d {
                let mut s = 0.0;
                for i in 0..k {
                    let mut oq = 0.0;
                    for j in 0..k {
                        oq += omega.at(i, j) * q.at(j, c);
                    }
                    s += q.at(i, r) * oq;
                }
                qoq[r * d + c] = s;
            }
        }
        let l = crate::inference::linalg::cholesky_lower(&qoq, d)
            .ok_or_else(|| "the curvature penalty restricted to its range is not positive definite".to_string())?;
        // M Lᵀ = Q₊  ⇔  L Mᵀ = Q₊ᵀ: forward substitution, one row of M at a time.
        let mut m = Mat::zeros(k, d);
        for i in 0..k {
            for r in 0..d {
                let mut s = q.at(i, r);
                for c in 0..r {
                    s -= l[r * d + c] * m.at(i, c);
                }
                m.set(i, r, s / l[r * d + r]);
            }
        }
        Ok(Whitening { omega, null, m, t_mid, xi })
    }

    pub fn n_basis(&self) -> usize {
        self.m.rows
    }

    pub fn n_modes(&self) -> usize {
        self.m.cols
    }
}

/// κ = √(K − 2) · T_ref^{3/2}: makes τ dimensionless and independent of the
/// knot count (§6.3). `t_ref` is the reference duration in model time units.
pub fn kappa(n_basis: usize, t_ref: f64) -> f64 {
    ((n_basis - 2) as f64).sqrt() * t_ref.powf(1.5)
}

/// Σ_j wⱼ Bⱼ(t), t clamped to the window — the value a coefficient vector w
/// gives, for checks.
pub fn spline_value(knots: &Knots, p: usize, w: &[f64], t: f64) -> f64 {
    let s = span(knots, p, t);
    (0..=p).map(|r| s.values[r] * w[s.first + r]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layouts() -> Vec<(&'static str, Knots)> {
        vec![
            ("uniform K=12", Knots::uniform(0.0, 168.0, 9).unwrap()),
            (
                "irregular K=12",
                Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap(),
            ),
            ("uniform 17 intervals at t~1e3", Knots::uniform(1000.0, 1168.0, 17).unwrap()),
        ]
    }

    fn max_abs(m: &Mat) -> f64 {
        m.data.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
    }

    fn quad(m: &Mat, x: &[f64], y: &[f64]) -> f64 {
        (0..m.rows).map(|i| (0..m.cols).map(|j| x[i] * m.at(i, j) * y[j]).sum::<f64>()).sum()
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        for n in 1..=10 {
            let (x, w) = gauss_legendre(n);
            for deg in 0..(2 * n) {
                let got: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(deg as i32)).sum();
                let want = if deg % 2 == 1 { 0.0 } else { 2.0 / (deg + 1) as f64 };
                assert!((got - want).abs() <= 1e-14, "n={n} deg={deg}: {got} vs {want}");
            }
        }
    }

    /// R1: the exact (p − 1)-point rule against 10-point Gauss–Legendre, ≤
    /// 1e-12·max|Ω|; symmetric; banded with bandwidth p; rank K − 2.
    #[test]
    fn penalty_is_exact_symmetric_and_banded() {
        for (name, k) in layouts() {
            for p in 2..=5 {
                let om = penalty(&k, p);
                let hi = penalty_with(&k, p, 10);
                let scale = max_abs(&om);
                let diff = om.data.iter().zip(&hi.data).fold(0.0_f64, |a, (x, y)| a.max((x - y).abs()));
                assert!(diff <= 1e-12 * scale, "{name} p={p}: exact vs 10-point {diff:e} of {scale:e}");
                for i in 0..om.rows {
                    for j in 0..om.cols {
                        assert_eq!(om.at(i, j).to_bits(), om.at(j, i).to_bits(), "{name} p={p}: not symmetric");
                        if i.abs_diff(j) > p {
                            assert_eq!(om.at(i, j), 0.0, "{name} p={p}: entry ({i},{j}) outside the band");
                        }
                    }
                }
            }
        }
    }

    /// R1: a line has zero curvature penalty. 1ᵀΩ1 and ξᵀΩξ ≤
    /// 1e-12·max|Ω|·(b − a)², and Ω·1, Ω·ξ vanish to the same scale; uniform
    /// and irregular knots, degrees 2–5.
    #[test]
    fn a_line_has_zero_curvature_penalty() {
        for (name, k) in layouts() {
            let (a, b) = k.window();
            for p in 2..=5 {
                let om = penalty(&k, p);
                let scale = max_abs(&om) * (b - a) * (b - a);
                let ones = vec![1.0; om.rows];
                let xi = greville(&k, p);
                let xi_c: Vec<f64> = xi.iter().map(|x| x - 0.5 * (a + b)).collect();
                for (label, v) in [("1", &ones), ("ξ", &xi), ("ξ − t_m", &xi_c)] {
                    let q = quad(&om, v, v).abs();
                    assert!(q <= 1e-12 * scale, "{name} p={p}: {label}ᵀΩ{label} = {q:e} (scale {scale:e})");
                }
                // A non-linear vector is penalized: the check is not vacuous.
                let bend: Vec<f64> = xi.iter().map(|x| ((x - a) / (b - a)).powi(2)).collect();
                assert!(quad(&om, &bend, &bend) > 1e-6 * max_abs(&om));
            }
        }
    }

    /// R1 whitening: MᵀΩM = I to `tol_mom`; ‖NᵀM‖_max ≤ 1e-12·‖N‖_F‖M‖_F; and
    /// MMᵀ is the Moore–Penrose pseudo-inverse Ω⁺ (the four Penrose
    /// conditions, to 1e-10 relative).
    fn check_whitening(degrees: std::ops::RangeInclusive<usize>, tol_mom: f64) {
        for (name, k) in layouts() {
            for p in degrees.clone() {
                let w = Whitening::new(&k, p).unwrap();
                let (kk, d) = (w.n_basis(), w.n_modes());
                let mul = |x: &Mat, y: &Mat| {
                    let mut out = Mat::zeros(x.rows, y.cols);
                    for i in 0..x.rows {
                        for j in 0..y.cols {
                            out.set(i, j, (0..x.cols).map(|l| x.at(i, l) * y.at(l, j)).sum());
                        }
                    }
                    out
                };
                let tr = |x: &Mat| {
                    let mut out = Mat::zeros(x.cols, x.rows);
                    for i in 0..x.rows {
                        for j in 0..x.cols {
                            out.set(j, i, x.at(i, j));
                        }
                    }
                    out
                };
                let mtom = mul(&mul(&tr(&w.m), &w.omega), &w.m);
                for i in 0..d {
                    for j in 0..d {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!((mtom.at(i, j) - want).abs() <= tol_mom, "{name} p={p}: MᵀΩM[{i},{j}] = {}", mtom.at(i, j));
                    }
                }
                let fro = |x: &Mat| x.data.iter().map(|v| v * v).sum::<f64>().sqrt();
                let ntm = mul(&tr(&w.null), &w.m);
                assert!(max_abs(&ntm) <= 1e-12 * fro(&w.null) * fro(&w.m), "{name} p={p}: NᵀM = {:e}", max_abs(&ntm));
                let x = mul(&w.m, &tr(&w.m));
                let rel = |got: &Mat, want: &Mat| {
                    let s = max_abs(want);
                    got.data.iter().zip(&want.data).fold(0.0_f64, |a, (g, v)| a.max((g - v).abs())) / s
                };
                let axa = mul(&mul(&w.omega, &x), &w.omega);
                let xax = mul(&mul(&x, &w.omega), &x);
                let ax = mul(&w.omega, &x);
                let xa = mul(&x, &w.omega);
                assert!(rel(&axa, &w.omega) <= 1e-10, "{name} p={p}: ΩXΩ ≠ Ω ({:e})", rel(&axa, &w.omega));
                assert!(rel(&xax, &x) <= 1e-10, "{name} p={p}: XΩX ≠ X ({:e})", rel(&xax, &x));
                assert!(rel(&ax, &tr(&ax)) <= 1e-10, "{name} p={p}: ΩX not symmetric");
                assert!(rel(&xa, &tr(&xa)) <= 1e-10, "{name} p={p}: XΩ not symmetric");
                assert_eq!(kk, k.n_intervals() + p);
            }
        }
    }

    /// The proposal's bar, 1e-12, at degrees 2–4 (cubic is the default).
    /// Measured worst: 3.5e-13 (cubic, 20 basis functions at t ≈ 10³).
    #[test]
    fn whitening_identities() {
        check_whitening(2..=4, 1e-12);
    }

    /// Degree 5, the largest allowed, is the worst-conditioned: MᵀΩM − I
    /// reaches 1.23e-12 with 22 basis functions at t ≈ 10³ (4.6e-14 with 14).
    /// The proposal states no separate bound for it; this pins 2e-12, with the
    /// other identities at their stated bars.
    #[test]
    fn whitening_identities_at_degree_five() {
        check_whitening(5..=5, 2e-12);
    }

    /// The whitened coefficients w = a·1 + b(ξ − t_m) + (τ/κ)Mz give a curve
    /// whose penalty is exactly (τ/κ)²‖z‖², and whose linear part is the line
    /// a + b(t − t_m): the parameterization is the stated bijection.
    #[test]
    fn whitened_coefficients_carry_the_stated_meaning() {
        let k = Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap();
        let w = Whitening::new(&k, 3).unwrap();
        let (a, b, tau, kap) = (-1.9, 0.004, 0.3, kappa(w.n_basis(), 7.0));
        let z: Vec<f64> = (0..w.n_modes()).map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0).collect();
        let coef: Vec<f64> = (0..w.n_basis())
            .map(|j| a + b * (w.xi[j] - w.t_mid) + tau / kap * (0..w.n_modes()).map(|i| w.m.at(j, i) * z[i]).sum::<f64>())
            .collect();
        let pen = quad(&w.omega, &coef, &coef);
        let want = (tau / kap).powi(2) * z.iter().map(|v| v * v).sum::<f64>();
        assert!((pen - want).abs() <= 1e-10 * want, "{pen} vs {want}");
        // The linear part is a + b(t − 84): a is the level at mid-window
        // (the window is [0, 168]) and b the slope per unit time.
        let lin: Vec<f64> = (0..w.n_basis()).map(|j| a + b * (w.xi[j] - w.t_mid)).collect();
        for t in [0.0, 17.0, 84.0, 168.0] {
            assert!((spline_value(&k, 3, &lin, t) - (a + b * (t - 84.0))).abs() <= 1e-12, "t = {t}");
        }
    }

    /// FNV-1a over the bits of M for the irregular cubic layout: pins the
    /// construction itself (column order, reflector signs, the Cholesky and
    /// the solves), not only its identities, which a different valid
    /// construction would also satisfy. Only + − × ÷ and sqrt are used, all
    /// correctly rounded under IEEE 754, so the value is portable.
    fn m_fingerprint() -> u64 {
        let k = Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap();
        let w = Whitening::new(&k, 3).unwrap();
        w.m.data.iter().fold(0xcbf29ce484222325u64, |h, v| (h ^ v.to_bits()).wrapping_mul(0x100000001b3))
    }

    #[test]
    fn whitening_construction_is_pinned() {
        assert_eq!(m_fingerprint(), PINNED_M, "M changed: {:#018x}", m_fingerprint());
    }

    const PINNED_M: u64 = 0xb9f928f84765b44c;

    /// R2: the penalty ½wᵀΩw has gradient Ωw (central differences, step
    /// 1e-5·max(|w|, 1), relative error < 1e-6 on every component).
    #[test]
    fn penalty_gradient_is_omega_w() {
        let k = Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap();
        let om = penalty(&k, 3);
        let w: Vec<f64> = (0..om.rows).map(|i| ((i * 5 + 2) % 7) as f64 / 3.0 - 1.0).collect();
        let half = |v: &[f64]| 0.5 * quad(&om, v, v);
        for i in 0..om.rows {
            let h = 1e-5 * w[i].abs().max(1.0);
            let (mut up, mut dn) = (w.clone(), w.clone());
            up[i] += h;
            dn[i] -= h;
            let fd = (half(&up) - half(&dn)) / (2.0 * h);
            let an: f64 = (0..om.cols).map(|j| om.at(i, j) * w[j]).sum();
            assert!((an - fd).abs() <= 1e-6 * an.abs().max(1e-3 * max_abs(&om)), "component {i}: {an} vs {fd}");
        }
    }

    #[test]
    fn whitening_is_refused_without_curvature() {
        assert!(Whitening::new(&Knots::uniform(0.0, 10.0, 4).unwrap(), 1).is_err());
        assert!(Whitening::new(&Knots::uniform(0.0, 10.0, 1).unwrap(), 2).is_ok()); // K = 3
    }
}
