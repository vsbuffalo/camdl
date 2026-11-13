#!/usr/bin/env bash
# Regenerate the boarding-school fit figure at the top of README.md.
#
# GitHub shows the README as a static page, so its figure is a committed file
# made here, not render output. This runs the fit the docs home page shows
# (docs/index.qmd: same files, same seed) in a temporary copy of
# docs/guide/first-fit/ and draws it with plot_predictive.py on a white
# background, so it reads on GitHub's light and dark themes alike. Re-run after
# a change to that fit, the model or the plot.
#
# Needs: this checkout's camdl first on PATH (`make readme-figure` arranges it)
# and uv.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
out=$root/docs/assets/boarding-school-fit.svg
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

cp -R "$root/docs/guide/first-fit/." "$work/"
cd "$work"
camdl fit run fit-r0.toml --label r0 --seed 4 >/dev/null
camdl fit predict @r0 >/dev/null
PLOT_OPAQUE=1 uv run --quiet plot_predictive.py "$out" SIR=results/fits/fit_r0-*/predict/*/
echo "wrote ${out#"$root"/}"
