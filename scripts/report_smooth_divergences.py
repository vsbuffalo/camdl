#!/usr/bin/env python3
"""Where and why the smooth R3 fits diverge, from the `smooth-diagnosis` runs.

Proposal 2026-10-27-estimable-bsplines §8.1, R3. For each run directory
`<form>_<tuning>_<seed>` under the diagnosis output that exists, reads every
chain's `trace.tsv` (post-adaptation sweeps) and writes one row per chain of
`divergence_report.tsv`:

- divergent transitions, leapfrog evaluations, and divergences per 1,000
  leapfrog evaluations;
- the adapted step size and the mean acceptance statistic;
- the distribution of tree depths (counts at each depth) and the transitions
  that hit the run's maximum depth;
- the median log τ and median integrated curvature ∫g''² = wᵀΩw at the start
  of divergent transitions (the previous sweep's state) and over all kept
  sweeps (smooth form only).

Run: make -f tests/recovery/Makefile smooth-divergence-report
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

import diagnose_smooth_sampling as d

DEPTH = {"single_d8": 8, "rung4c": 8, "single_d12": 12}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag-dir", type=Path, required=True)
    args = ap.parse_args()
    om, m, xc, kappa = d.whitening()
    rows = []
    for run in sorted(args.diag_dir.glob("*_*_*")):
        mt = re.fullmatch(r"(explicit|smooth)_(.+)_(\d+)", run.name)
        if not mt or not list(run.glob("out/fits/fit-*/pgas-*/seed_*")):
            continue
        form, tuning, seed = mt.group(1), mt.group(2), int(mt.group(3))
        burn = d.TUNINGS[tuning][0]
        max_depth = DEPTH.get(tuning, 10)
        for c in sorted(p for p in d.leaf(run).glob("chain_*") if p.is_dir()):
            t = pl.read_csv(c / "trace.tsv", separator="\t", null_values=["NA"], infer_schema_length=None).sort("sweep")
            post = t.filter(pl.col("sweep") >= burn)
            div = post["n_divergent"].to_numpy() > 0
            lf = int(post["n_leapfrog"].sum())
            depths = post["tree_depth"].to_numpy()
            row = {"form": form, "tuning": tuning, "seed": seed, "chain": c.name,
                   "divergent": int(div.sum()), "leapfrog": lf,
                   "div_per_1000_leapfrog": round(1000 * div.sum() / lf, 3),
                   "step_size": round(float(post["step_size"][-1]), 4),
                   "accept_mean": round(float(post["accept_stat"].mean()), 3),
                   "depth_counts": ",".join(f"{k}:{int((depths == k).sum())}" for k in range(0, max_depth + 1) if (depths == k).any()),
                   "max_depth": max_depth, "hits_max_depth": int((depths >= max_depth).sum()),
                   "logtau_div_median": None, "logtau_all_median": None,
                   "curvature_div_median": None, "curvature_all_median": None}
            if form == "smooth":
                # A divergent transition starts from the previous sweep's state.
                prev = t.filter(pl.col("sweep") >= burn - 1).head(len(post))
                z = np.stack([prev[f"z_m{i}"].to_numpy() for i in range(1, 4)], -1)
                w = (prev["a"].to_numpy()[:, None] + prev["b"].to_numpy()[:, None] * xc
                     + (prev["tau"].to_numpy() / kappa)[:, None] * (z @ m.T))
                curv = np.einsum("sk,kl,sl->s", w, om, w)
                lt = np.log(prev["tau"].to_numpy())
                row |= {"logtau_div_median": round(float(np.median(lt[div])), 3) if div.any() else None,
                        "logtau_all_median": round(float(np.median(lt)), 3),
                        "curvature_div_median": float(np.median(curv[div])) if div.any() else None,
                        "curvature_all_median": float(np.median(curv))}
            rows.append(row)
    out = args.diag_dir / "divergence_report.tsv"
    pl.DataFrame(rows).write_csv(out, separator="\t")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
