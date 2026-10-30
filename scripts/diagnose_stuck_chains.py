#!/usr/bin/env python3
"""Are the R3 recovery chains that disagree stuck by initialisation or by PGAS?

Proposal 2026-10-27-estimable-bsplines §8.1, R3. The `fit run` diagnosis
(`smooth-diagnosis`, default tuning) has chains whose complete-data
log-likelihood, log p(x, y | θ) on the chain's own reference path, sits far
below the others (smooth seed 31 chain 2 near −3,442 against −1,737; explicit
seed 47 chain 3 near −2,332 against −1,705). That score is not comparable
across chains as evidence about θ: it depends on the path. For each chain of
those two runs this script

1. takes the chain's last kept θ and estimates the marginal log-likelihood
   log p(y | θ) with `camdl pfilter` (2,000 particles, five seeds; mean and
   spread), which is comparable across chains;
2. restarts PGAS at that θ alone (`starts = { from_params = ... }`, one chain,
   a fresh unconditional particle pass for the latent path, 500 adaptation +
   1,000 kept sweeps) and records the complete-data log-likelihood at the
   first and last kept sweeps and θ's distance moved.

A chain whose θ has a marginal log-likelihood like the good chains', and
which recovers once its path is re-initialised, was stuck in its path; one
whose θ is itself poor and does not move was stuck in θ.

Writes stuck_chains.tsv under the diagnosis directory.
Run: make -f tests/recovery/Makefile smooth-stuck-chains
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
from pathlib import Path

import polars as pl

import diagnose_smooth_sampling as d

RUNS = [("smooth", 31), ("explicit", 47)]


def theta(form: str, trace: pl.DataFrame) -> dict[str, float]:
    names = [f"b{i}" for i in range(1, 6)] if form == "explicit" else ["a", "b", "tau", "z_m1", "z_m2", "z_m3"]
    last = trace.sort("sweep").tail(1)
    return {n: float(last[n][0]) for n in names}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diag-dir", type=Path, required=True)
    ap.add_argument("--camdl", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    args = ap.parse_args()
    camdl = str(args.camdl.resolve())
    repo = args.repo.resolve()
    out_root = (args.diag_dir / "stuck").resolve()
    rows = []
    for form, seed in RUNS:
        run = (args.diag_dir / f"{form}_default_{seed}").resolve()
        model = repo / "tests/fixtures/bspline_inference" / (
            "seir_bspline_recovery.camdl" if form == "explicit" else "seir_smooth_recovery.camdl")
        data = (args.diag_dir / "data" / f"onsets_{seed}.tsv").resolve()
        for c in sorted(p for p in d.leaf(run).glob("chain_*") if p.is_dir()):
            t = pl.read_csv(c / "trace.tsv", separator="\t", null_values=["NA"], infer_schema_length=None)
            post = t.filter(pl.col("sweep") >= 1000)
            th = theta(form, post)
            wd = out_root / f"{form}_{seed}_{c.name}"
            wd.mkdir(parents=True, exist_ok=True)
            params = {**d.FIXED, **th}
            (wd / "theta.toml").write_text("".join(f"{k} = {v}\n" for k, v in params.items()))
            lls = []
            for s in range(1, 6):
                o = subprocess.run([camdl, "pfilter", str(model), "--params", "theta.toml",
                                    "--data", f"onsets={data}", "--particles", "2000", "--seed", str(s)],
                                   cwd=wd, capture_output=True, text=True, check=True)
                lls.append(float(o.stdout.strip().splitlines()[-1]))
            # Restart PGAS at θ alone, with a fresh latent path.
            est = "".join(
                (f"{k} = {{ prior = {{ normal = {{ mean = -1.5, sd = 1.0 }} }} }}\n" if form == "explicit" else f"{k} = {{}}\n")
                for k in th)
            rel = lambda p: os.path.relpath(p, wd)
            (wd / "restart.toml").write_text(
                f'output_dir = "out"\n[model]\ncamdl = "{rel(model)}"\n[data.observations]\nonsets = "{rel(data)}"\n'
                f"[config]\ndt = 1.0\n[estimate]\n{est}[fixed]\n" + "".join(f"{k} = {v}\n" for k, v in d.FIXED.items())
                + '[method]\nalgorithm = "pgas"\nbackend = "chain_binomial"\nchains = 1\nparticles = 64\n'
                + 'sweeps = 1500\nburn_in = 500\nthin = 1\nstarts = { from_params = "theta.toml" }\n')
            if not list((wd / "out").glob("fits/restart-*/pgas-*/seed_*/chain_*/trace.tsv")):
                with open(wd / "restart.log", "w") as log:
                    subprocess.run([camdl, "fit", "run", "restart.toml"], cwd=wd, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            (leaf,) = list((wd / "out" / "fits").glob("restart-*/pgas-*/seed_*"))
            (rc,) = [p for p in leaf.glob("chain_*") if p.is_dir()]
            rt = pl.read_csv(rc / "trace.tsv", separator="\t", null_values=["NA"], infer_schema_length=None).sort("sweep")
            kept = rt.filter(pl.col("sweep") >= 500)
            moved = max(abs(float(kept[k].tail(1)[0]) - v) for k, v in th.items())
            rows.append({"form": form, "seed": seed, "chain": c.name,
                         "cdll_original_mean": round(float(post["log_complete_data_ll"].mean()), 1),
                         "marginal_ll_mean": round(statistics.mean(lls), 1),
                         "marginal_ll_sd": round(statistics.stdev(lls), 2),
                         "restart_cdll_first": round(float(rt["log_complete_data_ll"][0]), 1),
                         "restart_cdll_kept_first": round(float(kept["log_complete_data_ll"][0]), 1),
                         "restart_cdll_kept_last": round(float(kept["log_complete_data_ll"].tail(1)[0]), 1),
                         "restart_cdll_kept_mean": round(float(kept["log_complete_data_ll"].mean()), 1),
                         "theta_max_abs_move": round(moved, 3)})
            print("\t".join(str(v) for v in rows[-1].values()), flush=True)
    pl.DataFrame(rows).write_csv(args.diag_dir / "stuck_chains.tsv", separator="\t")


if __name__ == "__main__":
    main()
