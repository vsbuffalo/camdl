#!/usr/bin/env python3
"""Does longer adaptation remove the smooth fits' divergences? (R3, H1 vs H2)

Proposal 2026-10-27-estimable-bsplines §8.1, R3. The multistart probe
(`smooth-multistart`) leaves the smooth form diverging from IF2-good starts
with τ unboxed. Two explanations: (H1) warm-up never visits the large-τ region,
so the dense metric PGAS estimates from it (one window, sweeps 0 to
0.7 × burn_in, pgas.rs) is mismatched there; (H2) the (τ, z) geometry is
intrinsically hard and more adaptation will not help.

For smooth seeds 31 and 59 it reruns the four multistart PGAS chains from the
same `start.toml`s with the same fit seed and every setting identical except
the adaptation length: burn_in 3,000 instead of 1,000 (3,000 kept in both;
max tree depth 10, the default). For both arms it writes, per seed, to
warmup_length.tsv:

- post-adaptation divergences (four chains);
- the 5–95% range of τ over warm-up sweeps and over kept sweeps;
- worst rank-normalized R̂, bulk-ESS and tail-ESS of log β(t) over the 41
  window times, chains pooled;
- the adapted metric's log τ standard deviation (printed by PGAS when it
  estimates the dense mass matrix), mean over chains, against the standard
  deviation of log τ over the kept sweeps;
- the median log τ at the start of divergent transitions against all kept
  sweeps, and the fraction of divergent starts above the kept sweeps' 90%
  quantile of τ.

Run: make -f tests/recovery/Makefile smooth-warmup-length
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
from scipy.interpolate import BSpline

import diagnose_smooth_sampling as d

SEEDS = [31, 59]
BURN = {"baseline": 1000, "long_warmup": 3000}
KEPT = 3000


def chain_dir(run: Path) -> Path:
    (leaf,) = list((run / "out" / "fits").glob("fit-*/pgas-*/seed_*"))
    (c,) = [p for p in leaf.glob("chain_*") if p.is_dir()]
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--multistart-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camdl", type=Path, required=True)
    args = ap.parse_args()
    camdl = str(args.camdl.resolve())
    om, m, xc, kappa = d.whitening()
    u = d.knots()
    basis = np.stack([BSpline(u, np.eye(d.K)[j], d.DEGREE)(d.GRID) for j in range(d.K)])
    rows = []
    for seed in SEEDS:
        for arm, burn in BURN.items():
            traces, metric_sd = [], []
            for c in range(4):
                src = (args.multistart_dir / f"smooth_{seed}" / f"pgas_{c}").resolve()
                if arm == "baseline":
                    run = src
                else:
                    # Same depth below the diagnosis directory as the multistart
                    # run, so the fit.toml's relative paths resolve unchanged.
                    run = (args.out_dir / f"smooth_{seed}" / f"pgas_{c}").resolve()
                    run.mkdir(parents=True, exist_ok=True)
                    (run / "start.toml").write_text((src / "start.toml").read_text())
                    toml = (src / "fit.toml").read_text()
                    toml = re.sub(r"(?m)^sweeps = \d+$", f"sweeps = {burn + KEPT}", toml)
                    toml = re.sub(r"(?m)^burn_in = \d+$", f"burn_in = {burn}", toml)
                    (run / "fit.toml").write_text(toml)
                    if not list((run / "out").glob("fits/fit-*/pgas-*/seed_*/chain_*/trace.tsv")):
                        with open(run / "run.log", "w") as log:
                            subprocess.run([camdl, "fit", "run", "fit.toml"], cwd=run, stdout=log,
                                           stderr=subprocess.STDOUT, check=True)
                t = pl.read_csv(chain_dir(run) / "trace.tsv", separator="\t", null_values=["NA"],
                                infer_schema_length=None).sort("sweep")
                traces.append(t)
                log = (run / "run.log").read_text()
                block = log.split("dense mass matrix estimated")[1]
                metric_sd.append(float(re.search(r"tau\s+sd=([0-9.eE+-]+)", block).group(1)))
            warm = pl.concat([t.filter(pl.col("sweep") < burn) for t in traces])
            kept = [t.filter(pl.col("sweep") >= burn).head(KEPT) for t in traces]
            prev = [t.filter(pl.col("sweep") >= burn - 1).head(KEPT) for t in traces]
            pooled = pl.concat(kept)
            div = np.concatenate([k["n_divergent"].to_numpy() > 0 for k in kept])
            ltau_prev = np.log(np.concatenate([p["tau"].to_numpy() for p in prev]))
            tau_kept = pooled["tau"].to_numpy()

            def w_of(k: pl.DataFrame) -> np.ndarray:
                z = np.stack([k[f"z_m{i}"].to_numpy() for i in range(1, 4)], -1)
                return (k["a"].to_numpy()[:, None] + k["b"].to_numpy()[:, None] * xc
                        + (k["tau"].to_numpy() / kappa)[:, None] * (z @ m.T))

            lb = np.stack([w_of(k) @ basis for k in kept])
            q = lambda x, p: float(np.quantile(x, p))
            rows.append({
                "seed": seed, "arm": arm, "burn_in": burn,
                "divergent": int(div.sum()),
                "tau_warmup_q05_q95": f"{q(warm['tau'].to_numpy(), 0.05):.3f}-{q(warm['tau'].to_numpy(), 0.95):.3f}",
                "tau_kept_q05_q95": f"{q(tau_kept, 0.05):.3f}-{q(tau_kept, 0.95):.3f}",
                "rhat_beta": round(max(float(az.rhat(lb[..., i], method="rank")) for i in range(lb.shape[2])), 3),
                "bulk_ess_beta": round(min(float(az.ess(lb[..., i], method="bulk")) for i in range(lb.shape[2]))),
                "tail_ess_beta": round(min(float(az.ess(lb[..., i], method="tail", prob=(0.05, 0.95)))
                                           for i in range(lb.shape[2]))),
                "metric_logtau_sd": round(float(np.mean(metric_sd)), 3),
                "kept_logtau_sd": round(float(np.std(np.log(tau_kept), ddof=1)), 3),
                "logtau_div_median": round(float(np.median(ltau_prev[div])), 3) if div.any() else None,
                "logtau_all_median": round(float(np.median(ltau_prev)), 3),
                "div_above_tau_q90": round(float(np.mean(np.exp(ltau_prev[div]) > q(tau_kept, 0.9))), 3) if div.any() else None,
            })
            print("\t".join(str(v) for v in rows[-1].values()), flush=True)
    pl.DataFrame(rows).write_csv(args.out_dir / "warmup_length.tsv", separator="\t")


if __name__ == "__main__":
    main()
