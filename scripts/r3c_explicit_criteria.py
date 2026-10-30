#!/usr/bin/env python3
"""R3's recovery pass criteria for the explicit form, on the `fit run` draws.

Proposal 2026-10-27-estimable-bsplines §8.1, R3: every true coefficient
inside its central 99% posterior interval, and the posterior median of
log β(t) within 0.2 of the truth at every t in the window's middle 80%. The
truth is the R3 recovery truth [-0.7, -1.0, -1.6, -1.4, -1.9]. Evaluated, per
data seed (23, 31, 47, 59), on the four chains pooled of

- `from_prior`: the default-tuning `smooth-diagnosis` runs (starts drawn from
  the prior, 1,000 adaptation + 3,000 kept);
- `if2_starts`: the `smooth-multistart` runs (four single-chain fits started
  around the distinct good IF2 solutions, 1,000 + 3,000).

Writes r3c_explicit.tsv under --diag-dir.
Run: make -f tests/recovery/Makefile r3c-explicit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.interpolate import BSpline

import diagnose_smooth_sampling as d

TRUTH = np.array([-0.7, -1.0, -1.6, -1.4, -1.9])
NAMES = [f"b{i}" for i in range(1, 6)]


def kept(trace: Path, burn: int) -> np.ndarray:
    t = pl.read_csv(trace, separator="\t", null_values=["NA"], infer_schema_length=None)
    t = t.filter(pl.col("sweep") >= burn).sort("sweep")
    return np.stack([t[n].to_numpy() for n in NAMES], -1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag-dir", type=Path, required=True)
    args = ap.parse_args()
    u = d.knots()
    basis = np.stack([BSpline(u, np.eye(d.K)[j], d.DEGREE)(d.GRID) for j in range(d.K)])
    truth_lb = TRUTH @ basis
    rows = []
    for seed in d.SEEDS:
        sources = {
            "from_prior": [c / "trace.tsv" for c in sorted(p for p in d.leaf(args.diag_dir / f"explicit_default_{seed}").glob("chain_*") if p.is_dir())],
            "if2_starts": [next((args.diag_dir / "multistart" / f"explicit_{seed}" / f"pgas_{c}" / "out" / "fits").glob("fit-*/pgas-*/seed_*/chain_*/trace.tsv")) for c in range(4)],
        }
        for label, traces in sources.items():
            w = np.concatenate([kept(t, 1000) for t in traces])
            lo, hi = np.quantile(w, 0.005, axis=0), np.quantile(w, 0.995, axis=0)
            inside = (lo <= TRUTH) & (TRUTH <= hi)
            curve = np.abs(np.median(w @ basis, axis=0) - truth_lb)
            rows.append({"seed": seed, "starts": label, "draws": len(w),
                         "coefs_inside_99": int(inside.sum()),
                         "outside": ",".join(n for n, ok in zip(NAMES, inside) if not ok) or "-",
                         "worst_curve_error": round(float(curve.max()), 3),
                         "passes": bool(inside.all() and curve.max() <= 0.2)})
            print("\t".join(str(v) for v in rows[-1].values()), flush=True)
    pl.DataFrame(rows).write_csv(args.diag_dir / "r3c_explicit.tsv", separator="\t")


if __name__ == "__main__":
    main()
