#!/usr/bin/env bash
# Regenerate the camdl-scope screenshots on docs/guide/first-fit.qmd.
#
# The page runs its commands at render time, but its screenshots need a
# running viewer, so they are committed images made here instead. This runs
# the page's fits (same files, same seeds) in a temporary copy of
# docs/guide/first-fit/, serves the result with camdl-watch, and saves the
# tabs the page shows to docs/guide/first-fit/img/. The posterior predictive
# plots are drawn at render time instead (plot_predictive.py): camdl-scope
# reads `fit predict` output from its pre-0.2 location (camdl-scope#9).
#
# Needs: this checkout's camdl first on PATH (`make docs-first-fit-screenshots`
# arranges it), camdl-watch (docs/agents.md, "camdl 'scope"), uv, and Google
# Chrome.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
src=$root/docs/guide/first-fit
port=${SCOPE_PORT:-8899}

work=$(mktemp -d)
watch_pid=""
cleanup() {
  [ -n "$watch_pid" ] && kill "$watch_pid" 2>/dev/null || true
  rm -rf "$work"
}
trap cleanup EXIT

cp -R "$src/." "$work/"
cd "$work"

camdl simulate sir-r0.camdl --params truth.toml --seed 7 --obs-only synthetic.tsv
camdl fit run fit-beta.toml --label beta --seed 4
camdl fit run fit-synthetic.toml --label synthetic --seed 1
camdl fit run fit-r0.toml --label r0 --seed 4

camdl-watch --store results/fits --port "$port" >watch.log 2>&1 &
watch_pid=$!
up=""
for _ in $(seq 120); do
  if curl -sf "http://127.0.0.1:$port/api/health" >/dev/null; then up=1; break; fi
  sleep 0.5
done
if [ -z "$up" ]; then
  echo "camdl-watch did not answer on port $port:" >&2
  cat watch.log >&2
  exit 1
fi

run_id() { basename "$(ls -d results/fits/"$1"-*)"; }

uv run "$root/scripts/scope_screenshots.py" "http://127.0.0.1:$port" "$src/img" \
  "beta=$(run_id fit_beta):traces" \
  "synthetic=$(run_id fit_synthetic):posterior" \
  "r0=$(run_id fit_r0):traces,pair,diagnostics"
