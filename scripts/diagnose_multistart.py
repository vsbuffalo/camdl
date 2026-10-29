#!/usr/bin/env python3
"""Experimental, opt-in PGAS starts from distinct pre-fit solutions (R3).

Proposal 2026-10-27-estimable-bsplines §8.1, R3. Changes no default. For each
recovery dataset (data seeds 23, 31, 47, 59) and form (explicit, smooth):

1. Draws 16 starting points from the prior (numpy, fixed seed) and keeps the
   first 8 at which `camdl pfilter` (200 particles) gives a finite
   log-likelihood: under the N(-1.5, 1) level prior many draws give
   β(t) < γ, an epidemic that dies out, and an impossible likelihood.
2. Fits IF2 from each kept start (one chain, 500 particles, 50 iterations,
   cooling 0.7). The explicit coefficients get the box [-8, 2]; for `smooth`
   IF2 refuses an estimated scale, so τ is fixed at its prior median
   ln 2 / 5.99 and a, b, z are boxed to [-8, 2], [-0.2, 0.2], [-8, 8]: an
   unpenalised maximum-likelihood start. A penalised (MAP) start would need an
   optimiser that reads the prior; camdl has none, so it is not run.
3. Scores each IF2 end point with `camdl pfilter` (2,000 particles, 3 seeds),
   keeps the good ones (within 5 nats of the best) and, of those, the
   distinct ones (log β(t) differing by more than 0.1 somewhere in the
   window's middle 80%).
4. Runs 4 independent single-chain PGAS fits, chain k started at solution
   k mod n with dispersion (N(0, 0.05) on coefficients, level and modes,
   N(0, 0.002) on the slope, τ = median × exp(N(0, 0.2))), each with its own
   fresh latent path; defaults otherwise (1,000 adaptation, 3,000 kept).
5. Pools the 4 chains: rank-normalized R̂, bulk-ESS and tail-ESS (arviz) for
   log β(t) and R_eff(t) = β(t)S(t)/(γN), worst over 41 window times;
   post-adaptation divergences; and between-region movement: each kept
   sweep's curve is assigned to the nearest distinct solution (largest
   |Δ log β| over the grid), and the script counts switches per chain.

Writes multistart_summary.tsv and multistart_solutions.tsv under --out-dir.
Run: make -f tests/recovery/Makefile smooth-multistart
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
from scipy.interpolate import BSpline

import diagnose_smooth_sampling as d

TAU_MEDIAN = math.log(2) / 5.99
SEED = 20261029


def model_path(repo: Path, form: str) -> Path:
    return repo / "tests/fixtures/bspline_inference" / (
        "seir_bspline_recovery.camdl" if form == "explicit" else "seir_smooth_recovery.camdl")


def names(form: str) -> list[str]:
    return [f"b{i}" for i in range(1, 6)] if form == "explicit" else ["a", "b", "z_m1", "z_m2", "z_m3"]


def prior_draw(form: str, rng: np.random.Generator) -> dict[str, float]:
    if form == "explicit":
        return {n: float(rng.normal(-1.5, 1.0)) for n in names(form)}
    return {"a": float(rng.normal(-1.5, 1.0)), "b": float(rng.normal(0.0, 0.02)),
            **{f"z_m{i}": float(rng.normal()) for i in range(1, 4)}}


def coefficients(form: str, th: dict[str, float], wh) -> np.ndarray:
    om, m, xc, kappa = wh
    if form == "explicit":
        return np.array([th[f"b{i}"] for i in range(1, 6)])
    z = np.array([th[f"z_m{i}"] for i in range(1, 4)])
    return th["a"] + th["b"] * xc + th.get("tau", TAU_MEDIAN) / kappa * (m @ z)


BASIS = None


def log_beta(w: np.ndarray) -> np.ndarray:
    global BASIS
    if BASIS is None:
        u = d.knots()
        BASIS = np.stack([BSpline(u, np.eye(d.K)[j], d.DEGREE)(d.GRID) for j in range(d.K)])
    return w @ BASIS


def write_params(path: Path, th: dict[str, float], form: str) -> None:
    full = {**d.FIXED, **th}
    if form == "smooth":
        full.setdefault("tau", TAU_MEDIAN)
    path.write_text("".join(f"{k} = {v!r}\n" for k, v in full.items()))


def pfilter(camdl: str, model: Path, wd: Path, params: str, data: Path, particles: int, seeds: range) -> float:
    lls = []
    for s in seeds:
        o = subprocess.run([camdl, "pfilter", str(model), "--params", params, "--data", f"onsets={data}",
                            "--particles", str(particles), "--seed", str(s)], cwd=wd, capture_output=True, text=True)
        try:
            lls.append(float(o.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            lls.append(-math.inf)
    return statistics.mean(lls) if all(math.isfinite(x) for x in lls) else -math.inf


def fit_toml(wd: Path, model: Path, data: Path, est: str, fixed: dict, method: str) -> str:
    rel = lambda p: os.path.relpath(p, wd)
    return (f'output_dir = "out"\n[model]\ncamdl = "{rel(model)}"\n[data.observations]\nonsets = "{rel(data)}"\n'
            f"[config]\ndt = 1.0\n[estimate]\n{est}[fixed]\n" + "".join(f"{k} = {v!r}\n" for k, v in fixed.items())
            + method)


def run(camdl: str, wd: Path, toml: str) -> Path:
    (wd / "fit.toml").write_text(toml)
    if not list((wd / "out").glob("fits/fit-*/*/seed_*/chain_*")):
        with open(wd / "run.log", "w") as log:
            subprocess.run([camdl, "fit", "run", "fit.toml"], cwd=wd, stdout=log, stderr=subprocess.STDOUT, check=True)
    (leaf,) = list((wd / "out").glob("fits/fit-*/*/seed_*"))
    return leaf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camdl", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    camdl, repo = str(args.camdl.resolve()), args.repo.resolve()
    wh = d.whitening()
    summary, solutions = [], []
    for form in ["explicit", "smooth"]:
        for seed in d.SEEDS:
            model = model_path(repo, form)
            data = (args.data_dir / f"onsets_{seed}.tsv").resolve()
            root = (args.out_dir / f"{form}_{seed}").resolve()
            root.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(SEED + seed)
            # 1-2. feasible prior starts, IF2 from each.
            ends = []
            kept = 0
            for i in range(16):
                start = prior_draw(form, rng)
                wd = root / f"if2_{i:02d}"
                wd.mkdir(exist_ok=True)
                write_params(wd / "start.toml", start, form)
                if not math.isfinite(pfilter(camdl, model, wd, "start.toml", data, 200, range(1, 2))):
                    continue
                kept += 1
                if form == "explicit":
                    est = "".join(f"{n} = {{ bounds = [-8.0, 2.0] }}\n" for n in names(form))
                    fixed = d.FIXED
                else:
                    box = {"a": "[-8.0, 2.0]", "b": "[-0.2, 0.2]", "z_m1": "[-8.0, 8.0]", "z_m2": "[-8.0, 8.0]", "z_m3": "[-8.0, 8.0]"}
                    est = "".join(f"{n} = {{ bounds = {box[n]} }}\n" for n in names(form))
                    fixed = {**d.FIXED, "tau": TAU_MEDIAN}
                leaf = run(camdl, wd, fit_toml(wd, model, data, est, fixed,
                    '[method]\nalgorithm = "if2"\nbackend = "chain_binomial"\nchains = 1\nparticles = 500\n'
                    'iterations = 50\ncooling = 0.7\nstarts = { from_params = "start.toml" }\n'))
                tr = pl.read_csv(leaf / "chain_0" / "parameter_traces.tsv", separator="\t", comment_prefix="#",
                                 null_values=["NA"]).tail(1)
                th = {n: float(tr[n][0]) for n in names(form)}
                if form == "smooth":
                    th["tau"] = TAU_MEDIAN
                write_params(wd / "end.toml", th, form)
                ll = pfilter(camdl, model, wd, "end.toml", data, 2000, range(1, 4))
                ends.append((ll, th))
                if kept == 8:
                    break
            # 3. good and distinct solutions.
            ends.sort(key=lambda e: -e[0])
            best = ends[0][0]
            distinct: list[tuple[float, dict]] = []
            for ll, th in ends:
                if ll < best - 5.0:
                    continue
                lb = log_beta(coefficients(form, th, wh))
                if all(np.max(np.abs(lb - log_beta(coefficients(form, t2, wh)))) > 0.1 for _, t2 in distinct):
                    distinct.append((ll, th))
            for k, (ll, th) in enumerate(distinct):
                solutions.append({"form": form, "seed": seed, "solution": k, "marginal_ll": round(ll, 2),
                                  **{n: round(v, 4) for n, v in th.items()}})
            # 4. dispersed single-chain PGAS fits around the solutions.
            chains = []
            divergent = 0
            for c in range(4):
                ll, th = distinct[c % len(distinct)]
                start = {}
                for n, v in th.items():
                    sd = 0.002 if n == "b" else 0.05
                    start[n] = v * math.exp(rng.normal(0, 0.2)) if n == "tau" else v + rng.normal(0, sd)
                wd = root / f"pgas_{c}"
                wd.mkdir(exist_ok=True)
                write_params(wd / "start.toml", start, form)
                if form == "explicit":
                    est = "".join(f"{n} = {{ prior = {{ normal = {{ mean = -1.5, sd = 1.0 }} }} }}\n" for n in names(form))
                else:
                    est = "".join(f"{n} = {{}}\n" for n in names(form) + ["tau"])
                leaf = run(camdl, wd, fit_toml(wd, model, data, est, d.FIXED,
                    '[method]\nalgorithm = "pgas"\nbackend = "chain_binomial"\nchains = 1\nparticles = 64\n'
                    'sweeps = 4000\nburn_in = 1000\nthin = 1\nn_trajectories = 1000\n'
                    'starts = { from_params = "start.toml" }\n'))
                (cd,) = [p for p in leaf.glob("chain_*") if p.is_dir()]
                t = pl.read_csv(cd / "trace.tsv", separator="\t", null_values=["NA"], infer_schema_length=None)
                post = t.filter(pl.col("sweep") >= 1000).sort("sweep")
                divergent += int(post["n_divergent"].sum())
                paths = pl.read_csv(cd / "trajectories.tsv", separator="\t", comment_prefix="#")
                chains.append((post, paths))
            # 5. pooled diagnostics.
            n = min(len(p) for p, _ in chains)
            ws = np.stack([np.stack([coefficients(form, {k: float(r[k]) for k in names(form) + (["tau"] if form == "smooth" else [])}, wh)
                                     for r in p.head(n).iter_rows(named=True)]) for p, _ in chains])
            lb = np.einsum("csk,kt->cst", ws, BASIS)
            rb = max(float(az.rhat(lb[..., i], method="rank")) for i in range(lb.shape[2]))
            eb = min(float(az.ess(lb[..., i], method="bulk")) for i in range(lb.shape[2]))
            et = min(float(az.ess(lb[..., i], method="tail")) for i in range(lb.shape[2]))
            # R_eff on the sweeps with saved paths common to all chains.
            grid_t = [round(g) for g in d.GRID]
            common = sorted(set.intersection(*[set(pa["draw"].to_list()) for _, pa in chains]))
            reff = []
            u = d.knots()
            for tg in grid_t:
                bt = np.array([BSpline(u, np.eye(d.K)[j], d.DEGREE)(tg) for j in range(d.K)])
                col = []
                for ci, (p, pa) in enumerate(chains):
                    sel = pa.filter((pl.col("time") == tg) & pl.col("draw").is_in(common)).sort("draw")
                    idx = {s: i for i, s in enumerate(p.head(n)["sweep"].to_list())}
                    rows = [idx[x] for x in sel["draw"].to_list() if x in idx]
                    sel = sel.filter(pl.col("draw").is_in([x for x in sel["draw"].to_list() if x in idx]))
                    s_t = sel["S"].to_numpy().astype(float)
                    n_t = (sel["S"] + sel["E"] + sel["I"] + sel["R"]).to_numpy().astype(float)
                    col.append(np.exp(ws[ci, rows, :] @ bt) * s_t / (d.GAMMA * n_t))
                m = min(len(x) for x in col)
                reff.append(np.stack([x[:m] for x in col]))
            rr = max(float(az.rhat(x, method="rank")) for x in reff)
            er = min(float(az.ess(x, method="bulk")) for x in reff)
            etr = min(float(az.ess(x, method="tail")) for x in reff)
            sol_curves = [log_beta(coefficients(form, th, wh)) for _, th in distinct]
            switches = []
            for c in range(4):
                lab = [int(np.argmin([np.max(np.abs(lb[c, s] - sc)) for sc in sol_curves])) for s in range(n)]
                switches.append(sum(1 for a, b in zip(lab, lab[1:]) if a != b))
            row = {"form": form, "seed": seed, "feasible_starts": kept, "if2_fits": len(ends),
                   "good_distinct_solutions": len(distinct), "rhat_beta": round(rb, 3), "bulk_ess_beta": round(eb),
                   "tail_ess_beta": round(et), "rhat_Reff": round(rr, 3), "bulk_ess_Reff": round(er),
                   "tail_ess_Reff": round(etr), "divergent": divergent, "region_switches": ",".join(map(str, switches))}
            summary.append(row)
            print("\t".join(str(v) for v in row.values()), flush=True)
            pl.DataFrame(summary).write_csv(args.out_dir / "multistart_summary.tsv", separator="\t")
            pl.from_dicts(solutions, infer_schema_length=None).write_csv(args.out_dir / "multistart_solutions.tsv", separator="\t")


if __name__ == "__main__":
    main()
