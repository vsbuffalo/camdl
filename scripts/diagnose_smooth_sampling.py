#!/usr/bin/env python3
"""Sampling diagnosis of PGAS+NUTS on the R3 recovery SEIR, through `camdl fit run`.

Proposal 2026-10-27-estimable-bsplines §8.1, R3. For each recovery dataset
(data seeds 23, 31, 47, 59; written by the `write_recovery_datasets` test, the
same draws the in-process R3 tests use), fits the explicit-coefficient model
(`seir_bspline_recovery`, five unbounded coefficients, fit.toml
normal(-1.5, 1) priors) and the smooth one (`seir_smooth_recovery`, level,
slope, scale and modes unbounded) with `camdl fit run`, under each tuning:

- `default`: 1,000 adaptation sweeps, 3,000 kept;
- `long`: 3,000 adaptation sweeps, 3,000 kept;
- `rung4c`: the settings of `nuts_rungs.rs` rung 4c (direct `run_pgas`):
  1,000 adaptation, 4,000 kept, max tree depth 8, and every chain started
  at one point (`starts = "single"`: a = -1.5, b = 0, τ = 0.2, z = 0, or
  every coefficient -1.5), instead of `from_prior`;
- `single`: `default` with only the starts changed to that one point;
- `single_d8`, `single_d12`: `single` with only the maximum tree depth
  changed, to 8 or 12 (the default is 10).

4 chains, 64 particles, dense mass, thin 1. Per run it reads every chain's
`trace.tsv` (parameters and per-sweep NUTS and CSMC diagnostics) and
`trajectories.tsv` (S(t) of 1,000 kept sweeps per chain) and writes one row of
`summary.tsv`:

- wall-clock seconds; post-adaptation divergences; mean leapfrog steps per
  sweep; mean path renewal (fraction of time points whose reference-path
  state was replaced per sweep, CSMC `trajectory_renewal`);
- per group, the largest rank-normalized split-R̂ and smallest bulk-ESS
  (Vehtari et al. 2021, via arviz), and that ESS per second: z (the modes),
  tau, a, b, w (the five coefficients), beta (log β(t) at 41 times over the
  middle 80% of the window, worst over them), Reff (β(t)S(t)/(γN) at those
  times, worst over them, from the saved paths);

and to `divergences.tsv`, per run, the mean log τ and mean integrated
curvature ∫g''² = wᵀΩw of the divergent and of the other kept sweeps.

Run: make -f tests/recovery/Makefile smooth-diagnosis
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import time
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
from scipy.interpolate import BSpline

SEEDS = [23, 31, 47, 59]
TUNINGS = {"default": (1000, 3000), "long": (3000, 3000), "rung4c": (1000, 4000), "single": (1000, 3000),
           "single_d8": (1000, 3000), "single_d12": (1000, 3000)}
WINDOW = (0.0, 86.0)
GAMMA = 0.2
DEGREE = 3
K = 5
T_REF = 7.0
GRID = [WINDOW[0] + (0.1 + 0.8 * i / 40) * (WINDOW[1] - WINDOW[0]) for i in range(41)]
MIN_REFF_DRAWS = 100  # saved paths per chain needed to score R_eff(t)
FIXED = {"sigma": 0.2, "gamma": GAMMA, "rho": 0.5, "k": 10.0, "N0": 1_000_000, "E0": 100, "I0": 100}


def knots() -> np.ndarray:
    xs = [WINDOW[0], 0.5 * (WINDOW[0] + WINDOW[1]), WINDOW[1]]
    return np.array([xs[0]] * DEGREE + xs + [xs[-1]] * DEGREE)


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


def whitening() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Ω, M, ξ − t_m and κ for the recovery window (the stated construction)."""
    u = knots()
    eye = np.eye(K)
    d2 = [BSpline(u, eye[j], DEGREE).derivative(2) for j in range(K)]
    nodes, weights = np.polynomial.legendre.leggauss(10)
    om = np.zeros((K, K))
    for lo, hi in [(0.0, 43.0), (43.0, 86.0)]:
        t = lo + (hi - lo) * (nodes + 1) / 2
        vals = np.array([d(t) for d in d2])
        om += (vals * (weights * (hi - lo) / 2)) @ vals.T
    xi = np.array([u[j + 1:j + DEGREE + 1].mean() for j in range(K)])
    xc = xi - 0.5 * (WINDOW[0] + WINDOW[1])
    qp = householder_complement(np.column_stack([np.ones(K), xc]))
    m = np.linalg.solve(np.linalg.cholesky(qp.T @ om @ qp), qp.T).T
    return om, m, xc, math.sqrt(K - 2) * T_REF ** 1.5


def fit_toml(form: str, seed: int, tuning: str, data_dir: Path, run_dir: Path, repo: Path) -> str:
    burn, kept = TUNINGS[tuning]
    model = repo / "tests/fixtures/bspline_inference" / (
        "seir_bspline_recovery.camdl" if form == "explicit" else "seir_smooth_recovery.camdl")
    point = tuning in ("rung4c", "single", "single_d8", "single_d12")
    if form == "explicit":
        start = ", start = -1.5" if point else ""
        est = "".join(f"b{i} = {{ prior = {{ normal = {{ mean = -1.5, sd = 1.0 }} }}{start} }}\n" for i in range(1, 6))
    else:
        starts = {"a": -1.5, "b": 0.0, "tau": 0.2, "z_m1": 0.0, "z_m2": 0.0, "z_m3": 0.0}
        est = "".join(f"{n} = {{ start = {v} }}\n" if point else f"{n} = {{}}\n" for n, v in starts.items())
    extra = {"rung4c": 'starts = "single"\nmax_tree_depth = 8\n', "single": 'starts = "single"\n',
             "single_d8": 'starts = "single"\nmax_tree_depth = 8\n',
             "single_d12": 'starts = "single"\nmax_tree_depth = 12\n'}.get(tuning, "")
    fixed = "".join(f"{k} = {v}\n" for k, v in FIXED.items())
    rel = lambda p: os.path.relpath(p, run_dir)
    return (f'output_dir = "out"\n[model]\ncamdl = "{rel(model)}"\n'
            f'[data.observations]\nonsets = "{rel(data_dir / f"onsets_{seed}.tsv")}"\n'
            f"[config]\ndt = 1.0\n[estimate]\n{est}[fixed]\n{fixed}"
            f'[method]\nalgorithm = "pgas"\nbackend = "chain_binomial"\nchains = 4\nparticles = 64\n'
            f"sweeps = {burn + kept}\nburn_in = {burn}\nthin = 1\nn_trajectories = 1000\n{extra}")


def leaf(run_dir: Path) -> Path:
    (found,) = list((run_dir / "out" / "fits").glob("fit-*/pgas-*/seed_*"))
    return found


def worst(arrays: list[np.ndarray]) -> tuple[float, float]:
    """Largest R̂ and smallest bulk-ESS over a list of (chains, draws) arrays."""
    r = max(float(az.rhat(x, method="rank")) for x in arrays)
    e = min(float(az.ess(x, method="bulk")) for x in arrays)
    return r, e


def analyse(form: str, seed: int, tuning: str, run_dir: Path, wall: float, wh) -> tuple[dict, dict]:
    om, m, xc, kappa = wh
    burn, kept = TUNINGS[tuning]
    traces = []
    paths = []
    for c in sorted(p for p in leaf(run_dir).glob("chain_*") if p.is_dir()):
        t = pl.read_csv(c / "trace.tsv", separator="\t", null_values=["NA"], infer_schema_length=None)
        traces.append(t.filter(pl.col("sweep") >= burn).sort("sweep"))
        p = pl.read_csv(c / "trajectories.tsv", separator="\t", comment_prefix="#")
        paths.append(p)
    n = min(len(t) for t in traces)
    traces = [t.head(n) for t in traces]

    def arr(col: str) -> np.ndarray:
        return np.stack([t[col].to_numpy() for t in traces])

    if form == "explicit":
        w = np.stack([arr(f"b{i}") for i in range(1, 6)], axis=-1)
    else:
        z = np.stack([arr(f"z_m{i}") for i in range(1, 4)], axis=-1)
        w = (arr("a")[..., None] + arr("b")[..., None] * xc
             + (arr("tau") / kappa)[..., None] * np.einsum("kr,csr->csk", m, z))
    u = knots()
    basis = np.stack([BSpline(u, np.eye(K)[j], DEGREE)(GRID) for j in range(K)])  # K × T
    log_beta = np.einsum("csk,kt->cst", w, basis)
    groups: dict[str, list[np.ndarray]] = {
        "w": [w[..., j] for j in range(K)],
        "beta": [log_beta[..., i] for i in range(len(GRID))],
    }
    if form == "smooth":
        groups |= {"z": [arr(f"z_m{i}") for i in range(1, 4)], "tau": [arr("tau")],
                   "a": [arr("a")], "b": [arr("b")]}
    # R_eff(t) = β(t)S(t)/(γN) on the sweeps whose paths were saved.
    sweeps0 = traces[0]["sweep"].to_numpy()
    reff = []
    for ci, (t, p) in enumerate(zip(traces, paths)):
        saved = p.filter(pl.col("time").is_in([round(g) for g in GRID]))
        draws = sorted(set(saved["draw"].to_list()))
        reff.append((draws, saved))
    common = sorted(set.intersection(*[set(d) for d, _ in reff]))
    reff_groups = []
    if len(common) >= MIN_REFF_DRAWS:
        idx = {s: i for i, s in enumerate(sweeps0)}
        for g in GRID:
            tg = round(g)
            col = []
            for ci, (_, saved) in enumerate(reff):
                sel = saved.filter(pl.col("time") == tg).filter(pl.col("draw").is_in(common)).sort("draw")
                s_t = sel["S"].to_numpy().astype(float)
                n_t = (sel["S"] + sel["E"] + sel["I"] + sel["R"]).to_numpy().astype(float)
                rows = [idx[d] for d in sel["draw"].to_list()]
                lb = np.einsum("sk,k->s", w[ci, rows, :], np.array([BSpline(u, np.eye(K)[j], DEGREE)(tg) for j in range(K)]))
                col.append(np.exp(lb) * s_t / (GAMMA * n_t))
            reff_groups.append(np.stack(col))
        groups["Reff"] = reff_groups
    row: dict = {"seed": seed, "form": form, "tuning": tuning, "wall_s": round(wall, 1),
                 "kept_per_chain": n,
                 "divergent": int(sum(t["n_divergent"].sum() for t in traces)),
                 "leapfrog_mean": round(float(np.mean(arr("n_leapfrog"))), 1),
                 "renewal_mean": round(float(np.mean(arr("trajectory_renewal"))), 3)}
    for g in ["z", "tau", "a", "b", "w", "beta", "Reff"]:
        if g in groups:
            r, e = worst(groups[g])
            row |= {f"rhat_{g}": round(r, 3), f"ess_{g}": round(e), f"ess_per_s_{g}": round(e / wall, 2)}
        else:
            row |= {f"rhat_{g}": None, f"ess_{g}": None, f"ess_per_s_{g}": None}
    div = arr("n_divergent").ravel() > 0
    curv = np.einsum("csk,kl,csl->cs", w, om, w).ravel()
    ltau = np.log(arr("tau")).ravel() if form == "smooth" else np.full(div.shape, np.nan)
    drow = {"seed": seed, "form": form, "tuning": tuning, "n_divergent": int(div.sum()),
            "logtau_divergent": round(float(np.mean(ltau[div])), 3) if div.any() else None,
            "logtau_other": round(float(np.mean(ltau[~div])), 3),
            "curvature_divergent": float(np.mean(curv[div])) if div.any() else None,
            "curvature_other": float(np.mean(curv[~div]))}
    return row, drow


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camdl", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma list of form:tuning to run")
    args = ap.parse_args()
    wh = whitening()
    only = set(filter(None, args.only.split(",")))
    rows, drows = [], []
    for tuning in TUNINGS:
        for form in ["explicit", "smooth"]:
            if only and f"{form}:{tuning}" not in only:
                continue
            for seed in SEEDS:
                run_dir = (args.out_dir / f"{form}_{tuning}_{seed}").resolve()
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "fit.toml").write_text(
                    fit_toml(form, seed, tuning, args.data_dir.resolve(), run_dir, args.repo.resolve()))
                if not list((run_dir / "out").glob("fits/fit-*/pgas-*/seed_*/chain_*/trace.tsv")):
                    start = time.monotonic()
                    with open(run_dir / "run.log", "w") as log:
                        subprocess.run([str(args.camdl.resolve()), "fit", "run", "fit.toml"], cwd=run_dir,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    (run_dir / "wall_s").write_text(f"{time.monotonic() - start}\n")
                wall = float((run_dir / "wall_s").read_text())
                row, drow = analyse(form, seed, tuning, run_dir, wall, wh)
                rows.append(row)
                drows.append(drow)
                print("\t".join(str(v) for v in row.values()), flush=True)
    pl.DataFrame(rows).write_csv(args.out_dir / "summary.tsv", separator="\t")
    pl.DataFrame(drows).write_csv(args.out_dir / "divergences.tsv", separator="\t")


if __name__ == "__main__":
    main()
