# /// script
# requires-python = ">=3.11"
# dependencies = ["polars>=1.0", "matplotlib>=3.8"]
# ///
"""Plot a prior predictive check.

Reads the per-draw parameters (`--draws-out`) and trajectories (`-o`) written
by `camdl simulate --draws prior`, and the observed counts, and draws two
panels: the prior on R0 (a column of the draws, or beta / gamma), and the
epidemics the prior implies against the data. Writes an SVG.

Usage: uv run plot_prior.py DRAWS.tsv TRAJECTORIES.tsv DATA.tsv OUT.svg
"""

import sys

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import polars as pl

# Drawn in a placeholder colour that becomes `currentColor` in the SVG, so the
# figure takes the text colour of the page it is placed in.
INK = "#010203"
DRAWS = "#2f6f8f"


def main(draws_path: str, traj_path: str, data_path: str, out_path: str) -> None:
    draws = pl.read_csv(draws_path, separator="\t")
    if "R0" not in draws.columns:
        draws = draws.with_columns(R0=pl.col("beta") / pl.col("gamma"))
    traj = pl.read_csv(traj_path, separator="\t", comment_prefix="#")
    data = pl.read_csv(data_path, separator="\t")

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
            "svg.hashsalt": out_path,  # distinct ids when a page inlines several
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")

    r0 = draws["R0"].to_numpy()
    left.hist(r0, bins=[10 ** (k / 10) for k in range(-6, 15)], color=DRAWS, alpha=0.75)
    left.set_xscale("log")
    left.set_xticks([0.3, 1, 3, 10, 30], labels=["0.3", "1", "3", "10", "30"])
    left.minorticks_off()
    left.axvline(1.0, color=INK, lw=1.2, ls="--")
    left.text(
        0.97, 0.95, f"{(r0 < 1).mean():.0%} of draws\nhave R0 < 1",
        transform=left.transAxes, ha="right", va="top",
    )
    left.set_xlabel("R0 (log scale)")
    left.set_ylabel("prior draws")
    left.yaxis.get_major_locator().set_params(integer=True)

    for _, g in traj.group_by("draw"):
        right.plot(g["t"], g["I"], color=DRAWS, alpha=0.12, lw=0.8)
    right.plot(data["time"], data["in_bed"], "o-", color=INK, ms=4, lw=1.2, label="observed")
    right.set_xlabel("day")
    right.set_ylabel("number ill")
    right.legend(frameon=False, loc="upper right")

    fig.savefig(out_path, transparent=True)
    with open(out_path) as f:
        svg = f.read()
    with open(out_path, "w") as f:
        f.write(svg.replace(INK, "currentColor"))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
