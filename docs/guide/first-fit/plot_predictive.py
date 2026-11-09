# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0", "matplotlib>=3.8"]
# ///
"""Plot posterior predictive checks from `camdl fit predict` output.

One row per fit, one column per horizon: `free_forward` (simulated from day 0
without the data) and `one_step` (each day predicted from the counts up to the
day before). Each panel shows the 50% and 90% bands, the median, and the
observed counts. Writes an SVG.

Usage: uv run plot_predictive.py OUT.svg LABEL=PREDICT_DIR [LABEL=PREDICT_DIR...]

PREDICT_DIR is a `results/fits/<fit>/predict/<id>/` directory holding
`predictive/in_bed.tsv` and `observed/in_bed.tsv`.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import polars as pl

INK = "#010203"  # becomes currentColor, as in plot_prior.py
BAND = "#2f6f8f"
HORIZONS = {"free_forward": "free-forward, from day 0", "one_step": "one day ahead"}


def main(out_path: str, fits: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "text.color": INK,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": out_path,
        }
    )
    fig, axes = plt.subplots(
        len(fits), 2, figsize=(9, 3.4 * len(fits)), layout="constrained",
        sharex=True, sharey=True, squeeze=False,
    )
    for row, spec in zip(axes, fits):
        label, directory = spec.split("=", 1)
        pred = pl.read_csv(Path(directory) / "predictive" / "in_bed.tsv", separator="\t")
        obs = pl.read_csv(Path(directory) / "observed" / "in_bed.tsv", separator="\t")
        for ax, (horizon, title) in zip(row, HORIZONS.items()):
            band = pred.filter(pl.col("horizon") == horizon).sort("time")
            joined = band.join(obs, on="time")
            inside = ((joined["value"] >= joined["q05"]) & (joined["value"] <= joined["q95"])).sum()
            ax.fill_between(band["time"], band["q05"], band["q95"], color=BAND, alpha=0.18, lw=0)
            ax.fill_between(band["time"], band["q25"], band["q75"], color=BAND, alpha=0.32, lw=0)
            ax.plot(band["time"], band["q50"], color=BAND, lw=1.5)
            ax.plot(obs["time"], obs["value"], "o-", color=INK, ms=4, lw=1.2)
            ax.set_title(f"{label}: {title}", loc="left", fontsize=13)
            ax.text(0.97, 0.95, f"{inside} of {len(joined)} days\ninside the 90% band",
                    transform=ax.transAxes, ha="right", va="top")
    for ax in axes[-1]:
        ax.set_xlabel("day")
    for ax in axes[:, 0]:
        ax.set_ylabel("boys in bed")

    fig.savefig(out_path, transparent=True)
    with open(out_path) as f:
        svg = f.read()
    with open(out_path, "w") as f:
        f.write(svg.replace(INK, "currentColor"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
