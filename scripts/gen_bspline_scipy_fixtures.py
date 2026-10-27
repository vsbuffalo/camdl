#!/usr/bin/env python3
"""Reference basis values for camdl's B-spline evaluator, from scipy.

Writes two TSVs, both independent oracles for `rust/crates/sim/src/bspline.rs`
and the periodic spline built on it (proposal
`docs/dev/proposals/2026-10-27-estimable-bsplines.md` §8.1, R1):

- `bspline_basis_scipy.tsv`: every clamped basis function B_k(t), k = 0..K-1,
  over three knot layouts and degrees 0-5, at the window ends, every
  breakpoint, points between, and points outside the window. camdl holds a
  spline at its end value outside the window, so the reference value at an
  outside t is scipy's value at t clamped to [a, b].
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
