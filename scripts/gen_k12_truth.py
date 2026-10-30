#!/usr/bin/env python3
"""The §8.4 truth for the explicit-spline benchmark (proposal
2026-10-27-estimable-bsplines §8.2, §8.4): β(t) = γ·R(t) with γ = 0.125 per day
and R(t) = 0.7 + 1.5 / (1 + exp((t − 90)/10)), on the daily grid 0..182.

Writes tests/fixtures/bspline_inference/beta_true.tsv, the `interpolated`
forcing of seir_k12_truth.camdl. Run: make -f tests/recovery/Makefile k12-truth
"""
import argparse
import math
from pathlib import Path

GAMMA = 0.125


def r_true(t: float) -> float:
    return 0.7 + 1.5 / (1.0 + math.exp((t - 90.0) / 10.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = "".join(f"{t}\t{GAMMA * r_true(t)!r}\n" for t in range(0, 183))
    args.out.write_text("t\tbeta\n" + rows)


if __name__ == "__main__":
    main()
