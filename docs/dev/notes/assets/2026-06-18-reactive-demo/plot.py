# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "matplotlib"]
# ///
"""Render the reactive-interventions demo figure (gh#204).

Usage:  uv run plot.py <data_dir> <out_png>

<data_dir> holds the four `traj_<name>.tsv` trajectories and the three
`rx_after<lag>.tsv` reactive logs produced by run.sh. Same seed across runs +
obs draws on a separate RNG ⇒ the trajectories are byte-identical until each
campaign fires; earlier response ⇒ more S protected ⇒ smaller final size.
"""
import sys
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
N = 1000  # N0, matching run.sh

def traj(name):
    return pl.read_csv(f"{D}/traj_{name}.tsv", separator="\t", comment_prefix="#")

def fire_time(after):
    return float(pl.read_csv(f"{D}/rx_after{after}.tsv", separator="\t")["fire_time"][0])

runs = [
    ("baseline", "OFF (no campaign)", "#444444", "--", None),
    ("after0",   "ON · wait 0 d",     "#1f77b4", "-",  0),
    ("after21",  "ON · wait 21 d",    "#ff7f0e", "-",  21),
    ("after42",  "ON · wait 42 d",    "#2ca02c", "-",  42),
]

fig, (ax_i, ax_r) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)

for name, label, color, ls, after in runs:
    df = traj(name)
    t = df["t"].to_list()
    ax_i.plot(t, df["I"].to_list(), color=color, lw=2.4, ls=ls, label=label, zorder=3)
    ax_r.plot(t, (df["R"] / N * 100).to_list(), color=color, lw=2.4, ls=ls, zorder=3)
    if after is not None:
        ft = fire_time(after)
        for ax in (ax_i, ax_r):
            ax.axvline(ft, color=color, ls=":", lw=1.4, alpha=0.85, zorder=1)
        ax_i.annotate(f"fire t={int(ft)}", xy=(ft, 300), xytext=(ft + 0.5, 300 - 1.2 * after),
                      color=color, fontsize=8.5, fontweight="bold", rotation=90, va="top")
    final_ar = df["R"][-1] / N * 100
    ax_r.annotate(f"{final_ar:.0f}%", xy=(t[-1], final_ar), xytext=(122, final_ar),
                  color=color, fontsize=10, fontweight="bold", va="center")

for ax in (ax_i, ax_r):
    ax.axvline(7, color="black", ls="-", lw=0.8, alpha=0.5, zorder=1)
ax_i.annotate("trigger crosses (t=7,\nreported ≥ 2)", xy=(7, 250), xytext=(13, 285),
              fontsize=8.5, arrowprops=dict(arrowstyle="->", color="black", alpha=0.6))

ax_i.set_ylabel("Infectious  $I(t)$")
ax_r.set_ylabel("Attack rate  $R(t)/N$  (%)")
ax_r.set_xlabel("day")
ax_i.set_title(
    "camdl reactive SIA (gh#204) — scenario off vs on, varying response lag\n"
    "vaccinate S→V when reported cases cross threshold, `after` days later · "
    "earlier response → smaller epidemic",
    fontsize=11)
ax_i.legend(loc="upper right", fontsize=9, framealpha=0.95, title="reactive policy")
for ax in (ax_i, ax_r):
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 130)
ax_r.set_ylim(0, 105)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print("saved", OUT)
