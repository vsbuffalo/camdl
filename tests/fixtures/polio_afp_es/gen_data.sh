#!/usr/bin/env bash
# Deterministic synthetic-data generator for the spatial polio AFP+ES
# multi-cadence fixture.
#
# Two stages, kept separate per the project data-step convention:
#   1. `camdl simulate ... --obs-only-dir` writes ONE wide TSV per stratum leaf
#      (afp_urban.tsv, afp_rural.tsv, es_urban.tsv, es_rural.tsv), each on its
#      OWN cadence (AFP monthly = every 30 d; ES biweekly = every 14 d).
#   2. A deterministic pivot folds the per-leaf wide files into the LONG-FORM
#      per-SOURCE files the multi-cadence fit loader consumes
#      (`afp.tsv` = time,patch,cases ; `es.tsv` = time,patch,conc), routing each
#      leaf's rows to its patch level. The stratified observation header
#      `afp[p in patch]` makes all patch leaves share one `source` (afp/es), so
#      the fit binds ONE long-form file per source (2026-06-10 §4.2).
#
# Why the pivot: `simulate --obs-only-dir` emits per-leaf wide today, but a
# stratified (`: dim` column) stream loads via the long-form router. This step
# bridges the two until simulate gains a native long-form emit.
#
# Fixed seed (1) → byte-reproducible data. Run from anywhere.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
MODEL="$REPO/tests/fixtures/polio_afp_es_2patch.camdl"
PARAMS="$REPO/tests/fixtures/polio_afp_es_2patch.params.toml"
CAMDL="${CAMDL:-$REPO/rust/target/release/camdl}"
DATA="$HERE/data"
SEED="${SEED:-1}"

export CAMDL_SKIP_VERSION_CHECK=1
export CAMDLC="${CAMDLC:-$REPO/ocaml/_build/default/bin/camdlc.exe}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "[1/2] simulate (seed=$SEED) → per-leaf wide TSVs"
"$CAMDL" simulate "$MODEL" --params "$PARAMS" \
  --backend chain_binomial --dt 1 --seed "$SEED" \
  --obs-only-dir "$tmp/obs" --output-dir "$tmp/results" >/dev/null

mkdir -p "$DATA"

# pivot <source> <scored-col> <level...>: fold {source}_{level}.tsv into
# long-form {source}.tsv with columns: time, patch, <scored-col>.
pivot() {
  local source="$1" scored="$2"; shift 2
  local out="$DATA/$source.tsv"
  printf 'time\tpatch\t%s\n' "$scored" > "$out"
  for level in "$@"; do
    # skip header; emit (time, level, value) per row
    tail -n +2 "$tmp/obs/${source}_${level}.tsv" \
      | awk -v lvl="$level" -F'\t' '{printf "%s\t%s\t%s\n", $1, lvl, $2}'
  done | sort -t$'\t' -k1,1n -k2,2 >> "$out"
}

echo "[2/2] pivot per-leaf wide → long-form per-source"
pivot afp cases urban rural
pivot es  conc  urban rural

echo "wrote:"
for f in afp es; do
  echo "  $DATA/$f.tsv  ($(($(wc -l < "$DATA/$f.tsv") - 1)) rows)"
done
