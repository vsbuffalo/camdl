#!/usr/bin/env python3
"""Prior-predictive calibration of the `smooth` scale prior (proposal
2026-10-27-estimable-bsplines §6.5).

tau ~ exponential(lambda), lambda = -ln(alpha)/U, is the PC prior for the
standard deviation of the curvature modes, its base model a straight line in
log beta. A tail statement bounds curvature, not the curve, so U is calibrated
against a band on R(t) = beta(t)/gamma: at every t of the run (window [0, 168],
held to 182) the 95% prior band of R must contain [0.5, 3] and lie inside
[0.1, 20].

Independent of camdl's Rust code: scipy's B-splines and numpy's QR and
Cholesky (numpy's QR may differ from camdl's by column signs, which changes M
by an orthogonal factor and leaves the prior's law unchanged).

Writes tests/recovery/cases/spline_seir/smooth_prior_calibration.tsv. Run:
make -f tests/recovery/Makefile calibrate-smooth-prior.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline

SEED = 20261027
DRAWS = 4000
ALPHA = 0.05
U_SCAN = [0.3, 0.4, 0.5, 0.6, 0.8]
GAMMA = 0.125
T_REF = 7.0
A_MEAN, A_SD = math.log(1.2 * GAMMA), 0.6
B_SD = 0.005
REPORT_T = [0.0, 42.0, 84.0, 126.0, 168.0, 182.0]
GRID_T = np.arange(0.0, 183.0)
DEGREE = 3


def uniform(a: float, b: float, n: int) -> list[float]:
    h = (b - a) / n
    return [a + i * h for i in range(n + 1)]


KNOT_SETS = {
    "K8_uniform": uniform(0.0, 168.0, 5),
    "K12_uniform": uniform(0.0, 168.0, 9),
    "K20_uniform": uniform(0.0, 168.0, 17),
    "K12_irregular": [0.0, 10.0, 25.0, 30.0, 60.0, 90.0, 100.0, 130.0, 150.0, 168.0],
}


def design(xs: list[float], ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basis values at ts (clamped to the window: held outside), the penalty
    Omega by 10-point Gauss-Legendre, and the knot vector."""
    p = DEGREE
    u = np.array([xs[0]] * p + xs + [xs[-1]] * p)
    k = len(xs) - 1 + p
    eye = np.eye(k)
    tc = np.clip(ts, xs[0], xs[-1])
    basis = np.column_stack([BSpline(u, eye[j], p, extrapolate=False)(tc) for j in range(k)])
    nodes, weights = np.polynomial.legendre.leggauss(10)
    d2 = [BSpline(u, eye[j], p).derivative(2) for j in range(k)]
    omega = np.zeros((k, k))
    for lo, hi in zip(xs[:-1], xs[1:]):
        t = lo + (hi - lo) * (nodes + 1) / 2
        w = weights * (hi - lo) / 2
        vals = np.array([d(t) for d in d2])
        omega += (vals * w) @ vals.T
    return basis, omega, u


def whitening(xs: list[float], omega: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    p = DEGREE
    k = omega.shape[0]
    xi = np.array([u[j + 1:j + p + 1].mean() for j in range(k)])
    t_mid = 0.5 * (xs[0] + xs[-1])
    n = np.column_stack([np.ones(k), xi - t_mid])
    q, _ = np.linalg.qr(n, mode="complete")
    qp = q[:, 2:]
    l = np.linalg.cholesky(qp.T @ omega @ qp)
    m = np.linalg.solve(l, qp.T).T  # M = Q+ L^-T
    return m, xi, t_mid


def bands(xs: list[float], lam: float, rng: np.random.Generator) -> tuple[dict, float, float]:
    ts = np.array(sorted(set(REPORT_T) | set(GRID_T.tolist())))
    basis, omega, u = design(xs, ts)
    m, xi, t_mid = whitening(xs, omega, u)
    k = omega.shape[0]
    kappa = math.sqrt(k - 2) * T_REF ** 1.5
    a = rng.normal(A_MEAN, A_SD, DRAWS)
    b = rng.normal(0.0, B_SD, DRAWS)
    tau = rng.exponential(1.0 / lam, DRAWS)
    z = rng.normal(0.0, 1.0, (DRAWS, k - 2))
    w = a[:, None] + b[:, None] * (xi - t_mid)[None, :] + (tau / kappa)[:, None] * (z @ m.T)
    log_r = w @ basis.T - math.log(GAMMA)
    lo = np.quantile(log_r, 0.025, axis=0)
    hi = np.quantile(log_r, 0.975, axis=0)
    per_t = {t: (math.exp(lo[i]), math.exp(hi[i])) for i, t in enumerate(ts)}
    return per_t, float(np.exp(lo.min())), float(np.exp(hi.max()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for name, xs in KNOT_SETS.items():
        for u_tail in U_SCAN:
            lam = -math.log(ALPHA) / u_tail
            per_t, lo_all, hi_all = bands(xs, lam, rng)
            meets = all(lo <= 0.5 and hi >= 3.0 and lo >= 0.1 and hi <= 20.0 for lo, hi in per_t.values())
            for t in REPORT_T:
                lo, hi = per_t[t]
                rows.append(f"{name}\t{u_tail!r}\t{lam!r}\t{t!r}\t{lo!r}\t{hi!r}\t{int(meets)}")
            rows.append(f"{name}\t{u_tail!r}\t{lam!r}\tall\t{lo_all!r}\t{hi_all!r}\t{int(meets)}")
    header = [
        "Prior-predictive calibration of the smooth scale prior tau ~ exponential(lambda),",
        "lambda = -ln(0.05)/U (proposal 2026-10-27-estimable-bsplines §6.5). Columns: knot set,",
        "U, lambda, t (or `all` for the range over t in [0, 182]), the 95% band of",
        "R(t) = beta(t)/gamma, and whether the target holds at every t: the band",
        "contains [0.5, 3] and lies inside [0.1, 20].",
        f"Cubic clamped, window [0, 168] held to 182, gamma = {GAMMA}, T_ref = {T_REF} days,",
        f"a ~ N(ln(1.2 gamma), {A_SD}^2), b ~ N(0, {B_SD}^2), {DRAWS} draws, seed {SEED}.",
        f"scipy {scipy.__version__}, numpy {np.__version__}. Written by scripts/calibrate_smooth_prior.py;",
        "regenerate with `make -f tests/recovery/Makefile calibrate-smooth-prior`.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(f"# {h}\n" for h in header)
                        + "knots\tU\tlambda\tt\tR_lo\tR_hi\tmeets_target\n" + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
