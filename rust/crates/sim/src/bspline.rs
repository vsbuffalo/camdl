//! The arithmetic every spline forcing shares: find the knot interval holding
//! `t`, then compute the `p + 1` basis functions that can be nonzero there.
//!
//! A B-spline of degree `p` is a weighted sum `g(t) = Σₖ cₖ Bₖ(t)` of basis
//! functions, each a piecewise polynomial nonzero on at most `p + 1` adjacent
//! knot intervals. On the interval `[u_μ, u_μ₊₁)` only `B_{μ−p} … B_μ` are
//! nonzero, so one evaluation needs `p + 1` basis values and `p + 1`
//! coefficients, never all `K`.
//!
//! The values come from the Cox–de Boor triangle (Cox 1972; de Boor 1972) in
//! the stable form of Piegl & Tiller, *The NURBS Book*, 2nd ed. (1997),
//! Algorithm A2.2. Every step combines nonnegative quantities with nonnegative
//! weights, so there is no subtractive cancellation (de Boor 1978, ch. X) even
//! when `t` is an absolute model time of order 10³.
//!
//! This module knows nothing of expressions or parameters; the forcing kinds
//! supply coefficients. The workspaces are fixed-size arrays on the stack, so
//! no call allocates.
//!
//! Design: `docs/dev/proposals/2026-10-27-estimable-bsplines.md` §4.1.

/// The largest degree a spline forcing may declare. The workspaces below are
/// sized by it; a larger degree is refused where the forcing is compiled.
pub const MAX_DEGREE: usize = 5;

/// Width of the fixed workspaces: the `p + 1` nonzero basis values at most.
const WIDTH: usize = MAX_DEGREE + 1;

/// The smallest gap between adjacent breakpoints, as a fraction of the window
/// length `b − a`. A closer pair would make a triangle denominator (a
/// difference of knots) small enough to lose precision, so it is refused.
pub const MIN_BREAKPOINT_GAP_REL: f64 = 1e-9;

/// A clamped knot vector over the window `[a, b]`, stored by its distinct
/// breakpoints `a = x₀ < x₁ < … < x_n = b`. The full knot vector repeats `a`
/// and `b` each `p + 1` times, so for degree `p` it has `n + 2p + 1` entries
/// and `K = n + p` basis functions.
#[derive(Debug, Clone, PartialEq)]
pub enum Knots {
    /// `n` equal intervals of width `h`: `x_i = a + i·h`. Span lookup is O(1).
    Uniform { a: f64, h: f64, n: usize },
    /// Strictly increasing breakpoints. Span lookup is a binary search.
    Explicit(Vec<f64>),
}

/// The `p + 1` basis values that can be nonzero at one time:
/// `values[r] = B_{first + r}(t)` for `r = 0..=p`; entries past `p` are zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// Index of the first nonzero basis function.
    pub first: usize,
    pub values: [f64; WIDTH],
}

impl Knots {
    /// `n` equal intervals over `[a, b]`. Refuses a non-finite or empty window
    /// and `n = 0`.
    pub fn uniform(a: f64, b: f64, n: usize) -> Result<Self, String> {
        check_window(a, b)?;
        if n == 0 {
            return Err("a uniform knot vector needs at least one interval".into());
        }
        let h = (b - a) / n as f64;
        if h < MIN_BREAKPOINT_GAP_REL * (b - a) {
            return Err(format!(
                "{n} intervals over [{a}, {b}] are narrower than {MIN_BREAKPOINT_GAP_REL}·(b − a)"
            ));
        }
        Ok(Knots::Uniform { a, h, n })
    }

    /// Explicit breakpoints, first and last being the window. Refuses fewer
    /// than two, non-finite values, and a pair not increasing by at least
    /// `MIN_BREAKPOINT_GAP_REL · (b − a)`.
    pub fn explicit(breakpoints: Vec<f64>) -> Result<Self, String> {
        let (Some(&a), Some(&b)) = (breakpoints.first(), breakpoints.last()) else {
            return Err("a knot vector needs at least two breakpoints".into());
        };
        if breakpoints.len() < 2 {
            return Err("a knot vector needs at least two breakpoints".into());
        }
        check_window(a, b)?;
        let min_gap = MIN_BREAKPOINT_GAP_REL * (b - a);
        for w in breakpoints.windows(2) {
            if w[0].is_nan() || w[1].is_nan() || w[1] - w[0] < min_gap {
                return Err(format!(
                    "breakpoints must increase by at least {MIN_BREAKPOINT_GAP_REL}·(b − a), \
                     but {} is followed by {}",
                    w[0], w[1]
                ));
            }
        }
        Ok(Knots::Explicit(breakpoints))
    }

    /// The window `[a, b]`.
    pub fn window(&self) -> (f64, f64) {
        (self.breakpoint(0), self.breakpoint(self.n_intervals()))
    }

    /// The number of knot intervals `n`.
    pub fn n_intervals(&self) -> usize {
        match self {
            Knots::Uniform { n, .. } => *n,
            Knots::Explicit(xs) => xs.len() - 1,
        }
    }

    /// Breakpoint `x_i`, `i ∈ [0, n]`. The uniform form computes every
    /// breakpoint, the last included, by the same expression, so `interval`
    /// and `knot` cannot disagree about where a knot lies.
    fn breakpoint(&self, i: usize) -> f64 {
        match self {
            Knots::Uniform { a, h, .. } => a + i as f64 * h,
            Knots::Explicit(xs) => xs[i],
        }
    }

    /// The interval index `m ∈ [0, n)` with `x_m ≤ t < x_{m+1}`, for `t`
    /// already clamped to the window; `t = b` belongs to the last interval.
    fn interval(&self, t: f64) -> usize {
        let n = self.n_intervals();
        match self {
            Knots::Uniform { a, h, .. } => {
                let guess = ((t - a) / h).floor();
                // `as usize` saturates: a NaN or negative guess becomes 0.
                let mut m = (guess as usize).min(n - 1);
                // The division can round across a knot; settle against the
                // breakpoints themselves so that `t − u_μ ≥ 0` and
                // `u_{μ+1} − t ≥ 0` hold in the triangle.
                if m + 1 < n && t >= self.breakpoint(m + 1) {
                    m += 1;
                } else if m > 0 && t < self.breakpoint(m) {
                    m -= 1;
                }
                m
            }
            // The number of interior breakpoints at or below t.
            Knots::Explicit(xs) => xs[1..n].partition_point(|&x| x <= t),
        }
    }

    /// Knot `u_i` of the clamped vector for degree `p` (length `n + 2p + 1`):
    /// `a` for `i ≤ p`, `b` for `i ≥ n + p`, the breakpoint `x_{i−p}` between.
    fn knot(&self, i: usize, p: usize) -> f64 {
        let n = self.n_intervals();
        self.breakpoint(i.saturating_sub(p).min(n))
    }
}

fn check_window(a: f64, b: f64) -> Result<(), String> {
    if !(a.is_finite() && b.is_finite()) {
        return Err(format!("the window [{a}, {b}] must be finite"));
    }
    // Both are finite here, so this comparison has no NaN case.
    if a >= b {
        return Err(format!("the window [{a}, {b}] must have a < b"));
    }
    Ok(())
}

/// The Cox–de Boor triangle of Piegl & Tiller (1997) Algorithm A2.2 at `t` on
/// the span `μ`, reading knots through `u`. Returns `N[r] = B_{μ−p+r}(t)`.
///
/// Requires `u(μ) ≤ t ≤ u(μ+1)` and `p ≤ MAX_DEGREE`. The denominators
/// `right[r+1] + left[j−r] = u(μ+r+1) − u(μ+1−j+r)` are differences of
/// distinct knots, so they are positive whenever the breakpoints are strictly
/// increasing.
#[inline]
fn triangle(p: usize, t: f64, mu: usize, u: impl Fn(usize) -> f64) -> [f64; WIDTH] {
    assert!(p <= MAX_DEGREE, "spline degree {p} exceeds MAX_DEGREE = {MAX_DEGREE}");
    let mut n = [0.0; WIDTH];
    let mut left = [0.0; WIDTH];
    let mut right = [0.0; WIDTH];
    n[0] = 1.0;
    for j in 1..=p {
        left[j] = t - u(mu + 1 - j);
        right[j] = u(mu + j) - t;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// The nonzero basis functions of degree `p` at `t`. `t` is clamped to the
/// window first, so outside it the basis holds its value at the nearer end;
/// every spline forcing therefore holds outside its window by construction.
pub fn span(knots: &Knots, p: usize, t: f64) -> Span {
    let (a, b) = knots.window();
    let t = t.clamp(a, b);
    let m = knots.interval(t);
    Span { first: m, values: triangle(p, t, m + p, |i| knots.knot(i, p)) }
}

/// One basis function, `Bₖ(t)`, with `t` clamped as in [`span`]. Zero without
/// running the triangle when `k` is outside the `p + 1` nonzero at `t`; NaN
/// when `k` is not a basis function at all (`k ≥ n + p`), so a bad index can
/// never read as a genuine zero.
pub fn basis(knots: &Knots, p: usize, t: f64, k: usize) -> f64 {
    if k >= knots.n_intervals() + p {
        return f64::NAN;
    }
    let (a, b) = knots.window();
    let t = t.clamp(a, b);
    let m = knots.interval(t);
    if k < m || k > m + p {
        return 0.0;
    }
    triangle(p, t, m + p, |i| knots.knot(i, p))[k - m]
}

/// The `p + 1` nonzero basis functions at `t` and their first and second
/// derivatives in t: `ders[k][r]` is the k-th derivative of `B_{first + r}`.
/// Piegl & Tiller (1997) Algorithm A2.3 (derivatives of the nonzero basis
/// functions), on the same knots and the same clamped `t` as [`span`];
/// verified against finite differences of [`span`] and against scipy's
/// `BSpline.derivative`. For `p < 2` the second derivatives are zero.
pub fn span_derivatives(knots: &Knots, p: usize, t: f64) -> (usize, [[f64; WIDTH]; 3]) {
    assert!(p <= MAX_DEGREE, "spline degree {p} exceeds MAX_DEGREE = {MAX_DEGREE}");
    let (a, b) = knots.window();
    let t = t.clamp(a, b);
    let m = knots.interval(t);
    let mu = m + p;
    let u = |i: usize| knots.knot(i, p);
    // ndu: basis values (upper triangle) and knot differences (lower).
    let mut ndu = [[0.0; WIDTH]; WIDTH];
    let mut left = [0.0; WIDTH];
    let mut right = [0.0; WIDTH];
    ndu[0][0] = 1.0;
    for j in 1..=p {
        left[j] = t - u(mu + 1 - j);
        right[j] = u(mu + j) - t;
        let mut saved = 0.0;
        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    let mut ders = [[0.0; WIDTH]; 3];
    for (j, row) in ndu.iter().enumerate().take(p + 1) {
        ders[0][j] = row[p];
    }
    let n = p.min(2) as isize;
    let pi = p as isize;
    for r in 0..=pi {
        let (mut s1, mut s2) = (0usize, 1usize);
        let mut arr = [[0.0; WIDTH]; 2];
        arr[0][0] = 1.0;
        for k in 1..=n {
            let mut d = 0.0;
            let (rk, pk) = (r - k, pi - k);
            if r >= k {
                arr[s2][0] = arr[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                d = arr[s2][0] * ndu[rk as usize][pk as usize];
            }
            let j1 = if rk >= -1 { 1 } else { -rk };
            let j2 = if r - 1 <= pk { k - 1 } else { pi - r };
            for j in j1..=j2 {
                let (ju, rkj) = (j as usize, (rk + j) as usize);
                arr[s2][ju] = (arr[s1][ju] - arr[s1][ju - 1]) / ndu[(pk + 1) as usize][rkj];
                d += arr[s2][ju] * ndu[rkj][pk as usize];
            }
            if r <= pk {
                let ku = k as usize;
                arr[s2][ku] = -arr[s1][ku - 1] / ndu[(pk + 1) as usize][r as usize];
                d += arr[s2][ku] * ndu[r as usize][pk as usize];
            }
            ders[k as usize][r as usize] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }
    let mut f = p as f64;
    for k in 1..=(n as usize) {
        for v in ders[k].iter_mut().take(p + 1) {
            *v *= f;
        }
        f *= (p - k) as f64;
    }
    (m, ders)
}

/// Knot `u_i` of the clamped vector for degree `p` (see [`Knots`]).
pub fn clamped_knot(knots: &Knots, i: usize, p: usize) -> f64 {
    knots.knot(i, p)
}

/// The `p + 1` nonzero uniform (cardinal) B-splines of degree `p` on the
/// unit interval: knots at the integers, `u ∈ [0, 1]` the position within the
/// interval `[0, 1)`. `values[r]` is the basis function whose support is
/// `[r − p, r + 1]`. The same triangle as [`span`], on unclamped knots; the
/// periodic spline's wrap-fold is built on it.
pub fn cardinal_span(p: usize, u: f64) -> [f64; WIDTH] {
    // μ = p puts u_μ = 0 and u_{μ+1} = 1; knot i sits at i − p.
    triangle(p, u, p, |i| i as f64 - p as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The knot layouts the property checks run over: uniform from 0, uniform
    /// at absolute model time ~10³ (an anchored window), and irregular.
    fn layouts() -> Vec<(&'static str, Knots)> {
        vec![
            ("uniform", Knots::uniform(0.0, 168.0, 9).unwrap()),
            ("uniform_offset", Knots::uniform(1000.0, 1168.0, 9).unwrap()),
            (
                "irregular",
                Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0])
                    .unwrap(),
            ),
        ]
    }

    /// 10⁴ points spanning the window, its ends included.
    fn points(k: &Knots) -> impl Iterator<Item = f64> + '_ {
        let (a, b) = k.window();
        (0..10_000).map(move |i| if i == 9_999 { b } else { a + (b - a) * i as f64 / 9_999.0 })
    }

    /// The full clamped knot vector, for the Greville abscissae.
    fn full_knots(k: &Knots, p: usize) -> Vec<f64> {
        (0..k.n_intervals() + 2 * p + 1).map(|i| k.knot(i, p)).collect()
    }

    /// Σ cₖ Bₖ(t) for coefficient vector `c`, through `span`.
    fn eval(k: &Knots, p: usize, t: f64, c: &[f64]) -> f64 {
        let s = span(k, p, t);
        (0..=p).map(|r| s.values[r] * c[s.first + r]).sum()
    }

    #[test]
    fn partition_of_unity() {
        for (name, k) in layouts() {
            for p in 0..=MAX_DEGREE {
                let mut worst: f64 = 0.0;
                for t in points(&k) {
                    let s = span(&k, p, t);
                    worst = worst.max((s.values.iter().sum::<f64>() - 1.0).abs());
                }
                assert!(worst <= 1e-14, "{name} p={p}: |Σ B − 1| = {worst:e} > 1e-14");
            }
        }
    }

    #[test]
    fn every_basis_value_is_nonnegative() {
        for (name, k) in layouts() {
            for p in 0..=MAX_DEGREE {
                for t in points(&k) {
                    let s = span(&k, p, t);
                    assert!(s.values.iter().all(|&v| v >= 0.0), "{name} p={p} t={t}: {:?}", s.values);
                    assert!(s.first + p < k.n_intervals() + p, "{name} p={p}: first {} out of range", s.first);
                }
            }
        }
    }

    /// Clamped ends interpolate: f(a) = c₀ and f(b) = c_{K−1}, to 10⁻¹⁴ of
    /// the coefficient scale.
    #[test]
    fn clamped_ends_interpolate_the_end_coefficients() {
        for (name, k) in layouts() {
            for p in 0..=MAX_DEGREE {
                let n_basis = k.n_intervals() + p;
                let c: Vec<f64> = (0..n_basis).map(|i| 1.0 + 0.37 * i as f64 - 0.05 * (i * i) as f64).collect();
                let scale = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                let (a, b) = k.window();
                let fa = eval(&k, p, a, &c);
                let fb = eval(&k, p, b, &c);
                assert!((fa - c[0]).abs() <= 1e-14 * scale, "{name} p={p}: f(a) = {fa}, c₀ = {}", c[0]);
                assert!(
                    (fb - c[n_basis - 1]).abs() <= 1e-14 * scale,
                    "{name} p={p}: f(b) = {fb}, c_K = {}",
                    c[n_basis - 1]
                );
            }
        }
    }

    /// Coefficients 1 reproduce the constant 1 and the Greville abscissae
    /// ξₖ = (u_{k+1} + … + u_{k+p})/p reproduce t, to 10⁻¹³·(b − a). Degree 0
    /// has no linear precision, so the line is checked from degree 1.
    #[test]
    fn constants_and_the_line_are_reproduced() {
        for (name, k) in layouts() {
            let (a, b) = k.window();
            for p in 1..=MAX_DEGREE {
                let u = full_knots(&k, p);
                let n_basis = k.n_intervals() + p;
                let xi: Vec<f64> = (0..n_basis).map(|j| u[j + 1..=j + p].iter().sum::<f64>() / p as f64).collect();
                let ones = vec![1.0; n_basis];
                for t in points(&k) {
                    let line = eval(&k, p, t, &xi);
                    assert!((line - t).abs() <= 1e-13 * (b - a), "{name} p={p} t={t}: Σ ξB = {line}");
                    let c = eval(&k, p, t, &ones);
                    assert!((c - 1.0).abs() <= 1e-13, "{name} p={p} t={t}: Σ B = {c}");
                }
            }
        }
    }

    /// Analytic anchor: with one interval (K = p + 1), the clamped basis is
    /// the Bernstein basis C(p, k) sᵏ (1 − s)^{p−k}, s = (t − a)/(b − a).
    #[test]
    fn one_interval_is_the_bernstein_basis() {
        let binom = |n: usize, k: usize| -> f64 { (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64) };
        for (a, b) in [(0.0, 1.0), (3.0, 17.5), (1000.0, 1168.0)] {
            let knots = Knots::uniform(a, b, 1).unwrap();
            for p in 0..=MAX_DEGREE {
                for i in 0..=200 {
                    let t = a + (b - a) * i as f64 / 200.0;
                    let s = (t - a) / (b - a);
                    let sp = span(&knots, p, t);
                    assert_eq!(sp.first, 0);
                    for k in 0..=p {
                        let want = binom(p, k) * s.powi(k as i32) * (1.0 - s).powi((p - k) as i32);
                        let got = sp.values[k];
                        assert!((got - want).abs() <= 1e-12, "[{a}, {b}] p={p} t={t} k={k}: {got} vs {want}");
                    }
                }
            }
        }
    }

    /// Outside the window the basis holds its end value, exactly.
    #[test]
    fn outside_the_window_holds_the_end_value() {
        for (name, k) in layouts() {
            let (a, b) = k.window();
            for p in 0..=MAX_DEGREE {
                assert_eq!(span(&k, p, a - 50.0), span(&k, p, a), "{name} p={p} before a");
                assert_eq!(span(&k, p, b + 50.0), span(&k, p, b), "{name} p={p} after b");
            }
        }
    }

    /// `basis` agrees with `span` bit for bit, and is zero off the span.
    #[test]
    fn basis_agrees_with_span() {
        for (name, k) in layouts() {
            for p in 0..=MAX_DEGREE {
                let n_basis = k.n_intervals() + p;
                for t in points(&k).step_by(97) {
                    let s = span(&k, p, t);
                    for j in 0..n_basis {
                        let want = if (s.first..=s.first + p).contains(&j) { s.values[j - s.first] } else { 0.0 };
                        assert_eq!(basis(&k, p, t, j).to_bits(), want.to_bits(), "{name} p={p} t={t} j={j}");
                    }
                }
            }
        }
    }

    /// The cardinal span is the uniform basis on integer knots: it matches the
    /// clamped span on an interior interval far from either end.
    #[test]
    fn cardinal_span_matches_an_interior_clamped_span() {
        for p in 0..=MAX_DEGREE {
            // 2p + 1 unit intervals: interval p is p knots from either end.
            let k = Knots::uniform(0.0, (2 * p + 1) as f64, 2 * p + 1).unwrap();
            for i in 0..100 {
                let u = i as f64 / 100.0;
                let s = span(&k, p, p as f64 + u);
                assert_eq!(s.first, p);
                let c = cardinal_span(p, u);
                for r in 0..=p {
                    assert!((s.values[r] - c[r]).abs() <= 1e-14, "p={p} u={u} r={r}");
                }
            }
        }
    }

    /// Every breakpoint belongs to the interval it opens (`x_m ≤ t < x_{m+1}`,
    /// the last interval closed), which is what decides a degree-0 spline's
    /// value at a knot. For uniform knots `(x_i − a)/h` rounds below `i` at
    /// about a quarter of these breakpoints, so the lookup must settle against
    /// the breakpoints rather than trust the division.
    #[test]
    fn each_breakpoint_opens_its_own_interval() {
        let mut layouts = Vec::new();
        for a in [0.0, 3.0, 1000.0, 20000.5] {
            for len in [1.0, 7.0, 168.0, 365.25] {
                for n in 1..60 {
                    layouts.push(Knots::uniform(a, a + len, n).unwrap());
                }
            }
        }
        layouts.push(
            Knots::explicit(vec![0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0]).unwrap(),
        );
        for k in &layouts {
            let n = k.n_intervals();
            for i in 0..=n {
                let s = span(k, 0, k.breakpoint(i));
                assert_eq!(s.first, i.min(n - 1), "{k:?}: breakpoint {i} in the wrong interval");
                assert_eq!(s.values[0], 1.0);
            }
        }
    }

    /// A2.3 against central finite differences of the span values, at
    /// interior points of every layout and degree (h = 1e-5·(b − a)).
    #[test]
    fn span_derivatives_match_finite_differences() {
        for (name, k) in layouts() {
            let (a, b) = k.window();
            let h = 1e-5 * (b - a);
            for p in 0..=MAX_DEGREE {
                for i in 1..200 {
                    let t = a + (b - a) * (i as f64 + 0.37) / 200.0;
                    let (m, d) = span_derivatives(&k, p, t);
                    let s = span(&k, p, t);
                    assert_eq!(m, s.first);
                    for r in 0..=p {
                        assert!((d[0][r] - s.values[r]).abs() <= 1e-15, "{name} p={p}: values differ");
                    }
                    let (sp, sm) = (span(&k, p, t + h), span(&k, p, t - h));
                    if sp.first != m || sm.first != m {
                        continue; // straddles a knot: one-sided pieces
                    }
                    let scale1 = 1.0 / (b - a) * k.n_intervals() as f64 * 10.0;
                    for r in 0..=p {
                        let fd1 = (sp.values[r] - sm.values[r]) / (2.0 * h);
                        let fd2 = (sp.values[r] - 2.0 * s.values[r] + sm.values[r]) / (h * h);
                        assert!((d[1][r] - fd1).abs() <= 1e-6 * scale1, "{name} p={p} t={t} r={r}: B' {} vs {fd1}", d[1][r]);
                        if p >= 2 {
                            let scale2 = scale1 * scale1;
                            assert!((d[2][r] - fd2).abs() <= 1e-3 * scale2, "{name} p={p} t={t} r={r}: B'' {} vs {fd2}", d[2][r]);
                        } else {
                            assert_eq!(d[2][r], 0.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn malformed_knots_are_refused() {
        assert!(Knots::uniform(1.0, 1.0, 3).is_err());
        assert!(Knots::uniform(0.0, f64::INFINITY, 3).is_err());
        assert!(Knots::uniform(0.0, 1.0, 0).is_err());
        assert!(Knots::explicit(vec![0.0]).is_err());
        assert!(Knots::explicit(vec![0.0, 2.0, 1.0, 3.0]).is_err());
        assert!(Knots::explicit(vec![0.0, 1.0, 1.0 + 1e-12, 3.0]).is_err());
        assert!(Knots::explicit(vec![0.0, f64::NAN, 3.0]).is_err());
    }
}
