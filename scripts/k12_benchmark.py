#!/usr/bin/env python3
"""The §8.2 benchmark of the explicit B-spline form (proposal
2026-10-27-estimable-bsplines §8.2, §8.4).

Data: `seir_k12_truth.camdl` simulated at tests/recovery/cases/spline_k12/
truth.toml, chain-binomial, seed 8 (written by the `k12-data` target to the
case's ignored results/data/onsets.tsv). Fit: `seir_k12_explicit.camdl`, twelve
explicit log-link coefficients over [0, 168] with normal(-1.9, 1) priors.

1. PGAS through `camdl fit run`: 4 chains, 64 particles, dense mass,
   1,000 adaptation + 2,000 kept sweeps, the default `from_prior` starts.
   Per sampled parameter: rank-normalized R̂, bulk-ESS, tail-ESS (arviz) and
   bulk-ESS per second; the worst of each over log β(t) at 41 times in the
   window's middle 80%; post-adaptation divergences.
2. A plot of the truth β(t) against the posterior median and central 90%
   band, the window end (day 168) and the held region marked.
3. IF2 from up to 8 prior draws at which `camdl pfilter` (200 particles) is
   finite (one chain, 500 particles, 50 iterations, cooling 0.7, rw_sd 0.05 on
   the box [-8, 2]), each end point scored by `camdl pfilter` (2,000
   particles, 3 seeds), against the same score at the truth's projection onto
   the basis (least squares of log β_true on B₁..B₁₂ over the window's
   integer days). Not asserted.

Writes, under --out-dir: benchmark_parameters.tsv, benchmark_summary.tsv,
if2_pfilter.tsv and beta_posterior.png.
Run: make -f tests/recovery/Makefile k12-benchmark
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import polars as pl
from scipy.interpolate import BSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K, DEGREE, WINDOW, HORIZON = 12, 3, (0.0, 168.0), 182.0
NAMES = [f"b{i}" for i in range(1, K + 1)]
FIXED = {"sigma": 0.1, "gamma": 0.125, "rho": 0.5, "k": 10.0, "N0": 1_000_000, "E0": 10, "I0": 10}
GRID = [WINDOW[0] + (0.1 + 0.8 * i / 40) * (WINDOW[1] - WINDOW[0]) for i in range(41)]
SEED = 20261029


def basis_at(ts) -> np.ndarray:
    n_int = K - DEGREE
    xs = [WINDOW[0] + i * (WINDOW[1] - WINDOW[0]) / n_int for i in range(n_int + 1)]
    u = np.array([xs[0]] * DEGREE + xs + [xs[-1]] * DEGREE)
    tc = np.clip(np.asarray(ts, dtype=float), WINDOW[0], WINDOW[1])
    return np.stack([BSpline(u, np.eye(K)[j], DEGREE, extrapolate=False)(tc) for j in range(K)])


def beta_true(ts) -> np.ndarray:
    t = np.asarray(ts, dtype=float)
    return FIXED["gamma"] * (0.7 + 1.5 / (1.0 + np.exp((t - 90.0) / 10.0)))


def write_params(path: Path, th: dict) -> None:
    path.write_text("".join(f"{k} = {v!r}\n" for k, v in {**FIXED, **th}.items()))


def pfilter(camdl: str, model: Path, wd: Path, params: str, data: Path, particles: int, seeds) -> float:
    lls = []
    for s in seeds:
        o = subprocess.run([camdl, "pfilter", str(model), "--params", params, "--data", f"onsets={data}",
                            "--particles", str(particles), "--seed", str(s)], cwd=wd, capture_output=True, text=True)
        try:
            lls.append(float(o.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return -math.inf
    return statistics.mean(lls) if all(math.isfinite(x) for x in lls) else -math.inf


def fit(camdl: str, wd: Path, model: Path, data: Path, est: str, method: str) -> tuple[Path, float]:
    rel = lambda p: os.path.relpath(p, wd)
    (wd / "fit.toml").write_text(
        f'output_dir = "out"\n[model]\ncamdl = "{rel(model)}"\n[data.observations]\nonsets = "{rel(data)}"\n'
        f"[config]\ndt = 1.0\n[estimate]\n{est}[fixed]\n" + "".join(f"{k} = {v!r}\n" for k, v in FIXED.items())
        + method)
    if not list((wd / "out").glob("fits/fit-*/*/seed_*/chain_*")):
        start = time.monotonic()
        with open(wd / "run.log", "w") as log:
            subprocess.run([camdl, "fit", "run", "fit.toml"], cwd=wd, stdout=log, stderr=subprocess.STDOUT, check=True)
        (wd / "wall_s").write_text(f"{time.monotonic() - start}\n")
    (leaf,) = list((wd / "out").glob("fits/fit-*/*/seed_*"))
    return leaf, float((wd / "wall_s").read_text()) if (wd / "wall_s").exists() else math.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camdl", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    camdl, repo = str(args.camdl.resolve()), args.repo.resolve()
    model = repo / "tests/fixtures/bspline_inference/seir_k12_explicit.camdl"
    data = args.data.resolve()
    work = args.work_dir.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. PGAS benchmark.
    wd = work / "pgas"
    wd.mkdir(parents=True, exist_ok=True)
    leaf, wall = fit(camdl, wd, model, data, "".join(f"{n} = {{}}\n" for n in NAMES),
                     '[method]\nalgorithm = "pgas"\nbackend = "chain_binomial"\nchains = 4\nparticles = 64\n'
                     "sweeps = 3000\nburn_in = 1000\nthin = 1\n")
    traces = []
    for c in sorted(p for p in leaf.glob("chain_*") if p.is_dir()):
        t = pl.read_csv(c / "trace.tsv", separator="\t", null_values=["NA"], infer_schema_length=None)
        traces.append(t.filter(pl.col("sweep") >= 1000).sort("sweep"))
    n = min(len(t) for t in traces)
    traces = [t.head(n) for t in traces]
    draws = np.stack([np.stack([t[nm].to_numpy() for nm in NAMES], -1) for t in traces])  # c × s × K
    rows = []
    for j, nm in enumerate(NAMES):
        x = draws[..., j]
        e = float(az.ess(x, method="bulk"))
        rows.append({"parameter": nm, "rhat": round(float(az.rhat(x, method="rank")), 3), "bulk_ess": round(e),
                     "tail_ess": round(float(az.ess(x, method="tail", prob=(0.05, 0.95)))),
                     "bulk_ess_per_s": round(e / wall, 3), "median": round(float(np.median(x)), 3)})
    pl.DataFrame(rows).write_csv(out / "benchmark_parameters.tsv", separator="\t")
    lb = np.einsum("csk,kt->cst", draws, basis_at(GRID))
    rb = [float(az.rhat(lb[..., i], method="rank")) for i in range(lb.shape[2])]
    eb = [float(az.ess(lb[..., i], method="bulk")) for i in range(lb.shape[2])]
    et = [float(az.ess(lb[..., i], method="tail", prob=(0.05, 0.95))) for i in range(lb.shape[2])]
    summary = {"chains": len(traces), "kept_per_chain": n, "wall_s": round(wall, 1),
               "divergent": int(sum(t["n_divergent"].sum() for t in traces)),
               "leapfrog_mean": round(float(np.mean([t["n_leapfrog"].mean() for t in traces])), 1),
               "renewal_mean": round(float(np.mean([t["trajectory_renewal"].mean() for t in traces])), 3),
               "worst_rhat_logbeta": round(max(rb), 3), "min_bulk_ess_logbeta": round(min(eb)),
               "min_tail_ess_logbeta": round(min(et)), "min_bulk_ess_per_s_logbeta": round(min(eb) / wall, 3),
               "chain_mean_cdll": ",".join(f"{t['log_complete_data_ll'].mean():.1f}" for t in traces)}

    # 2. The posterior of β(t) against the truth.
    ts = np.arange(0.0, HORIZON + 1)
    beta_draws = np.exp(draws.reshape(-1, K) @ basis_at(ts))
    lo, med, hi = (np.quantile(beta_draws, q, axis=0) for q in (0.05, 0.5, 0.95))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axvspan(WINDOW[1], HORIZON, color="0.92", label="held past the window")
    ax.fill_between(ts, lo, hi, color="#7aa6d6", alpha=0.5, label="posterior 90% band")
    ax.plot(ts, med, color="#1f5aa6", lw=1.6, label="posterior median")
    ax.plot(ts, beta_true(ts), color="black", lw=1.2, ls="--", label="truth")
    ax.axvline(WINDOW[1], color="0.4", lw=0.8)
    ax.set_xlabel("day")
    ax.set_ylabel("β(t) per day")
    ax.set_title("Explicit B-spline, K = 12: known against inferred β(t)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "beta_posterior.png", dpi=150, metadata={"Software": None})
    summary["max_abs_median_log_error_middle80"] = round(float(np.max(np.abs(
        np.log(np.quantile(np.exp(lb.reshape(-1, len(GRID))), 0.5, axis=0)) - np.log(beta_true(GRID))))), 3)

    # 3. IF2 from feasible prior draws, scored by the particle filter.
    rng = np.random.default_rng(SEED)
    if2_rows = []
    kept = 0
    for i in range(16):
        start = {nm: float(rng.normal(-1.9, 1.0)) for nm in NAMES}
        wi = work / f"if2_{i:02d}"
        wi.mkdir(parents=True, exist_ok=True)
        write_params(wi / "start.toml", start)
        if not math.isfinite(pfilter(camdl, model, wi, "start.toml", data, 200, [1])):
            continue
        kept += 1
        try:
            lf, _ = fit(camdl, wi, model, data, "".join(f"{nm} = {{ bounds = [-8.0, 2.0], rw_sd = 0.05 }}\n" for nm in NAMES),
                        '[method]\nalgorithm = "if2"\nbackend = "chain_binomial"\nchains = 1\nparticles = 500\n'
                        'iterations = 50\ncooling = 0.7\nstarts = { from_params = "start.toml" }\n')
        except subprocess.CalledProcessError:
            if2_rows.append({"fit": f"if2_{i:02d}", "pfilter_ll": None})
            continue
        tr = pl.read_csv(lf / "chain_0" / "parameter_traces.tsv", separator="\t", comment_prefix="#", null_values=["NA"]).tail(1)
        th = {nm: float(tr[nm][0]) for nm in NAMES}
        write_params(wi / "end.toml", th)
        if2_rows.append({"fit": f"if2_{i:02d}", "pfilter_ll": round(pfilter(camdl, model, wi, "end.toml", data, 2000, [1, 2, 3]), 2)})
        if kept == 8:
            break
    days = np.arange(0.0, WINDOW[1] + 1)
    proj, *_ = np.linalg.lstsq(basis_at(days).T, np.log(beta_true(days)), rcond=None)
    wt = work / "truth_projection"
    wt.mkdir(parents=True, exist_ok=True)
    write_params(wt / "projection.toml", {nm: float(v) for nm, v in zip(NAMES, proj)})
    if2_rows.append({"fit": "truth_projection", "pfilter_ll": round(pfilter(camdl, model, wt, "projection.toml", data, 2000, [1, 2, 3]), 2)})
    pl.DataFrame(if2_rows).write_csv(out / "if2_pfilter.tsv", separator="\t")
    summary["projection_max_abs_log_error"] = round(float(np.max(np.abs(proj @ basis_at(days) - np.log(beta_true(days))))), 4)
    pl.DataFrame([summary]).write_csv(out / "benchmark_summary.tsv", separator="\t")
    print(summary)


if __name__ == "__main__":
    main()
