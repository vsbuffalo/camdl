#!/usr/bin/env python3
"""Reference basis values for camdl's B-spline evaluator, from scipy.

Writes three TSVs, each an independent oracle for `rust/crates/sim/src/bspline.rs`
and the periodic spline built on it (proposal
`docs/dev/proposals/2026-10-27-estimable-bsplines.md` §8.1, R1):

- `bspline_basis_scipy.tsv`: every clamped basis function B_k(t), k = 0..K-1,
  over three knot layouts and degrees 0-5, at the window ends, every
  breakpoint, points between, and points outside the window. camdl holds a
  spline at its end value outside the window, so the reference value at an
  outside t is scipy's value at t clamped to [a, b].
- `bspline_smooth_values_numpy.tsv`: the two `smooth` forcings of
  `tests/fixtures/gradient/bspline_smooth_grad.camdl` at fixed (a, b, tau, z),
  from w = a + b(xi - t_m) + (tau/kappa) M z with M = Q+ L^-T built here from
  the proposal's stated construction (Householder reflectors with
  v1 = x0 + sign(x0)||x||, columns [1, xi - t_m], no pivoting; numpy's
  Cholesky) and kappa = sqrt(K - 2) per^1.5, then scipy's BSpline, the lag and
  the unit scale. Pins kappa and M in camdl's runtime value path.
- `periodic_bspline_degrees_scipy.tsv`: the periodic spline f(t) for degrees
  0-5, in camdl's convention (the coefficient vector rolled by the centring
  shift max(p - 1, 0) // 2), at points inside one period and outside it.

The layouts must match `layout()` in
`rust/crates/sim/tests/bspline_reference_fixtures.rs`. Uniform breakpoints are
computed as a + i*h with h = (b - a)/n, the same expression the Rust evaluator
uses, so a point at a breakpoint is the same double on both sides.

Run: make fixtures-bspline (pins the scipy and numpy versions).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline

MAX_DEGREE = 5


def uniform(a: float, b: float, n: int) -> list[float]:
    h = (b - a) / n
    return [a + i * h for i in range(n + 1)]


LAYOUTS: dict[str, list[float]] = {
    "uniform": uniform(0.0, 168.0, 9),
    "uniform_offset": uniform(1000.0, 1168.0, 9),
    "irregular": [0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0],
}


def points(xs: list[float]) -> list[float]:
    """Window ends, every breakpoint, quarter points of every interval, and
    two points outside the window."""
    a, b = xs[0], xs[-1]
    pts = set(xs) | {a - 5.0, b + 7.0}
    for lo, hi in zip(xs[:-1], xs[1:]):
        for f in (0.25, 0.5, 0.75):
            pts.add(lo + f * (hi - lo))
    return sorted(pts)


def clamped_knots(xs: list[float], p: int) -> np.ndarray:
    return np.array([xs[0]] * p + xs + [xs[-1]] * p, dtype=np.float64)


def clamped_rows() -> list[str]:
    rows = []
    for name, xs in LAYOUTS.items():
        a, b = xs[0], xs[-1]
        for p in range(MAX_DEGREE + 1):
            u = clamped_knots(xs, p)
            n_basis = len(xs) - 1 + p
            eye = np.eye(n_basis)
            for t in points(xs):
                tc = min(max(t, a), b)
                vals = [float(BSpline(u, eye[k], p, extrapolate=False)(tc)) for k in range(n_basis)]
                total = sum(vals)
                assert abs(total - 1.0) < 1e-13, (name, p, t, total)
                for k, v in enumerate(vals):
                    rows.append(f"{name}\t{p}\t{t!r}\t{k}\t{v!r}")
    return rows


def penalty_rows() -> list[str]:
    """Omega_ij = integral of B_i''(t) B_j''(t) over the window, each entry by
    adaptive quadrature (scipy.integrate.quad) on every knot interval of
    scipy's own second-derivative splines: an implementation independent of
    camdl's Gauss-Legendre rule and derivative recurrence."""
    from scipy.integrate import quad
    rows = []
    for name in ("uniform", "irregular"):
        xs = LAYOUTS[name]
        for p in range(2, MAX_DEGREE + 1):
            u = clamped_knots(xs, p)
            n_basis = len(xs) - 1 + p
            eye = np.eye(n_basis)
            d2 = [BSpline(u, eye[k], p, extrapolate=False).derivative(2) for k in range(n_basis)]
            for i in range(n_basis):
                for j in range(i, min(n_basis, i + p + 1)):
                    total = 0.0
                    for lo, hi in zip(xs[:-1], xs[1:]):
                        mid = 0.5 * (lo + hi)
                        f = lambda t: float(d2[i](t if t < hi else np.nextafter(hi, lo))) * float(d2[j](t if t < hi else np.nextafter(hi, lo)))
                        val, _ = quad(f, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200, points=[mid])
                        total += val
                    rows.append(f"{name}\t{p}\t{i}\t{j}\t{total!r}")
    return rows


PERIOD = 4.0
N_BASIS = 7
COEFS = [0.7, 1.2, 0.9, 0.5, 1.1, 0.8, 1.3]


def periodic_rows() -> list[str]:
    dx = PERIOD / N_BASIS
    ts = [i * PERIOD / 200 for i in range(200)] + [-9.95, -1.3, 4.0, 4.7, 12.1]
    rows = []
    for p in range(MAX_DEGREE + 1):
        knots = np.arange(-p, N_BASIS + p + 1) * dx
        shift = max(p - 1, 0) // 2
        c = np.roll(np.asarray(COEFS, dtype=np.float64), shift)
        spline = BSpline(knots, np.concatenate([c, c[:p]]), p, extrapolate=False)
        for t in ts:
            rows.append(f"{p}\t{t!r}\t{float(spline(t % PERIOD))!r}")
    return rows


# The fixed parameter values of bspline_smooth_grad (must match
# `smooth_values_match_numpy` in rust/crates/sim/tests/bspline_reference_fixtures.rs).
SMOOTH_VALUES = {
    "a": -1.5, "b": 0.01, "tau": 0.4, "z": [0.8, -0.5, 1.1, 0.2, -0.9, 0.4],
    "c": 0.3, "d": 0.02, "s": 0.7, "y": [0.6, -1.2, 0.3, 0.9],
}


def householder_complement(n: np.ndarray) -> np.ndarray:
    k, c = n.shape
    a = n.copy()
    vs = []
    for j in range(c):
        x = a[j:, j]
        sign = -1.0 if x[0] < 0 else 1.0
        v = np.zeros(k)
        v[j:] = x
        v[j] = x[0] + sign * np.linalg.norm(x)
        a -= 2.0 * np.outer(v, v @ a) / (v @ v)
        vs.append(v)
    q = np.eye(k)[:, c:]
    for v in reversed(vs):
        q -= 2.0 * np.outer(v, v @ q) / (v @ v)
    return q


def smooth_rows() -> list[str]:
    v = SMOOTH_VALUES
    forcings = [
        # name, breakpoints, link_log, unit scale, lag, per, level, slope, scale, modes
        ("f_log", uniform(0.0, 60.0, 5), True, 1.0 / 7.0, 3.0, 7.0, v["a"], v["b"], v["tau"], v["z"]),
        ("f_id", [0.0, 10.0, 25.0, 60.0], False, 1.0, 0.0, 5.0, v["c"], v["d"], v["s"], v["y"]),
    ]
    p = 3
    rows = []
    nodes, weights = np.polynomial.legendre.leggauss(10)
    for name, xs, log, unit, lag, per, lev, slo, sca, z in forcings:
        u = clamped_knots(xs, p)
        k = len(xs) - 1 + p
        eye = np.eye(k)
        d2 = [BSpline(u, eye[j], p).derivative(2) for j in range(k)]
        om = np.zeros((k, k))
        for lo, hi in zip(xs[:-1], xs[1:]):
            t = lo + (hi - lo) * (nodes + 1) / 2
            vals = np.array([d(t) for d in d2])
            om += (vals * (weights * (hi - lo) / 2)) @ vals.T
        xi = np.array([u[j + 1:j + p + 1].mean() for j in range(k)])
        t_mid = 0.5 * (xs[0] + xs[-1])
        qp = householder_complement(np.column_stack([np.ones(k), xi - t_mid]))
        lo_ = np.linalg.cholesky(qp.T @ om @ qp)
        m = np.linalg.solve(lo_, qp.T).T
        kappa = np.sqrt(k - 2) * per ** 1.5
        w = lev + slo * (xi - t_mid) + sca / kappa * (m @ np.array(z))
        spline = BSpline(u, w, p, extrapolate=False)
        for t in [-5.0, 0.0, 2.0, 3.0, 7.5, 10.0, 13.0, 25.0, 31.7, 44.0, 59.0, 60.0, 63.0, 70.0, 80.0]:
            g = float(spline(np.clip(t - lag, xs[0], xs[-1])))
            f = float(unit * np.exp(g) if log else unit * g)
            rows.append(f"{name}\t{t!r}\t{f!r}")
    return rows


def write(path: Path, header: list[str], columns: str, rows: list[str]) -> None:
    path.write_text("".join(f"# {h}\n" for h in header) + columns + "\n" + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    versions = f"scipy {scipy.__version__}, numpy {np.__version__}"
    layouts = [f"  {name}: breakpoints {xs!r}" for name, xs in LAYOUTS.items()]
    write(
        args.out_dir / "bspline_basis_scipy.tsv",
        [
            "Clamped B-spline basis values B_k(t) from scipy.interpolate.BSpline",
            f"({versions}). Written by scripts/gen_bspline_scipy_fixtures.py;",
            "regenerate with `make fixtures-bspline`. An oracle, not an input.",
            "Outside the window the value is scipy's at t clamped to [a, b]",
            "(camdl holds a spline at its end value). Layouts:",
            *layouts,
        ],
        "layout\tdegree\tt\tk\tvalue",
        clamped_rows(),
    )
    write(
        args.out_dir / "bspline_penalty_scipy.tsv",
        [
            "Curvature penalty Omega_ij = int B_i'' B_j'' dt (upper band) from scipy",
            f"BSpline.derivative(2) and scipy.integrate.quad per interval ({versions}).",
            "Written by scripts/gen_bspline_scipy_fixtures.py; regenerate with",
            "`make fixtures-bspline`. An oracle, not an input. Layouts as in",
            "bspline_basis_scipy.tsv.",
        ],
        "layout\tdegree\ti\tj\tvalue",
        penalty_rows(),
    )
    write(
        args.out_dir / "bspline_smooth_values_numpy.tsv",
        [
            "The smooth forcings f_log and f_id of tests/fixtures/gradient/",
            "bspline_smooth_grad.camdl at fixed parameter values, from an independent",
            f"numpy construction of M and kappa and scipy's BSpline ({versions}).",
            "Written by scripts/gen_bspline_scipy_fixtures.py; regenerate with",
            f"`make fixtures-bspline`. An oracle, not an input. Values: {SMOOTH_VALUES}",
        ],
        "forcing\tt\tvalue",
        smooth_rows(),
    )
    write(
        args.out_dir / "periodic_bspline_degrees_scipy.tsv",
        [
            "Periodic B-spline f(t) from scipy.interpolate.BSpline",
            f"({versions}). Written by scripts/gen_bspline_scipy_fixtures.py;",
            "regenerate with `make fixtures-bspline`. An oracle, not an input.",
            f"period={PERIOD} n_basis={N_BASIS} coefs={COEFS}",
            "camdl's convention: coefficients rolled by max(degree - 1, 0) // 2.",
        ],
        "degree\tt\ty",
        periodic_rows(),
    )


if __name__ == "__main__":
    main()
