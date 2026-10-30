#!/usr/bin/env python3
"""Generate a reference TSV of the periodic B-spline forcing computed
via scipy.interpolate.BSpline.

This is an independent oracle: camdl's Rust evaluator must match the
output to 1e-12 relative tolerance. If scipy ever changes their
basis-indexing convention, this script will surface the change loudly
when regenerated; CI loads the *committed* TSV and stays offline.

scipy's periodic B-spline uses the standard de Boor recurrence with
no centering shift. camdl applies a `(degree-1)/2` centering shift
(Wand & Ormerod 2008 §3 convention, matching pomp), so we ROLL the
coefficient vector before feeding it to scipy to compensate.

Run: make fixtures-bspline (pins the scipy and numpy versions)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline


def gen(period: float, n_basis: int, degree: int, coefs: list[float], n_points: int) -> np.ndarray:
    """Returns an (n_points, 2) array of (t, y) where y = sum_i coefs[i] B_i(t)
    with camdl's basis-indexing convention applied."""
    dx = period / n_basis
    # Knot vector spanning [-degree·dx, (n_basis + degree)·dx].
    knots = np.arange(-degree, n_basis + degree + 1) * dx

    # camdl applies a (degree-1)//2 centering shift; scipy does not.
    # To use scipy as the oracle, roll the coefficient vector by the
    # same shift in the OPPOSITE direction so the curves agree.
    shift = (degree - 1) // 2

    # scipy BSpline requires a coefficient vector of length >= len(knots) - degree - 1.
    # We need len = n_basis + degree (=9 for cubic n_basis=6). Extend
    # the periodic coef vector cyclically to that length.
    n_coef = len(knots) - degree - 1
    assert n_coef == n_basis + degree, f"expected {n_basis + degree}, got {n_coef}"

    # Apply shift first (so c[k] is now in the centered convention),
    # then tile cyclically to length n_coef.
    coefs_arr = np.asarray(coefs, dtype=np.float64)
    coefs_shifted = np.roll(coefs_arr, shift)
    coefs_ext = np.concatenate([coefs_shifted, coefs_shifted[:degree]])

    spline = BSpline(knots, coefs_ext, degree, extrapolate=False)
    ts = np.linspace(0.0, period, n_points, endpoint=False)
    ys = spline(ts)
    return np.column_stack([ts, ys])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rust/crates/sim/tests/fixtures/periodic_bspline_scipy.tsv"),
        help="Output TSV path",
    )
    args = parser.parse_args()

    # Fixed test case: period=4, n_basis=6, degree=3 (cubic),
    # asymmetric coefs to make the shift convention visible.
    out = gen(
        period=4.0,
        n_basis=6,
        degree=3,
        coefs=[0.7, 1.2, 0.9, 0.5, 1.1, 0.8],
        n_points=200,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# gh#59 v2 oracle: periodic B-spline via scipy.interpolate.BSpline\n"
        "# fixture parameters:\n"
        "#   period=4.0  n_basis=6  degree=3\n"
        "#   coefs=[0.7, 1.2, 0.9, 0.5, 1.1, 0.8]\n"
        "# camdl applies a (degree-1)//2 centering shift; scipy doesn't.\n"
        "# This fixture rolls the coef vector by +shift before passing to\n"
        "# scipy, so the output represents camdl's convention.\n"
        "# Regenerate with: scripts/gen_periodic_bspline_scipy_fixture.py\n"
        "t\ty"
    )
    np.savetxt(args.out, out, fmt="%.18e", delimiter="\t",
               header=header, comments="")
    print(f"wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
