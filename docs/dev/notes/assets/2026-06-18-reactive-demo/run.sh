#!/usr/bin/env bash
# Regenerate the reactive-interventions demo figure (gh#204).
#
# Drives off the committed golden IR + a release `camdl` build, runs the SIR
# reactive SIA as scenario OFF (baseline) vs ON at three response lags, and
# renders the committed PNG next to this script. Intermediates live in a temp
# dir and are discarded.
#
#   make build-rust        # once: build the release camdl
#   bash docs/dev/notes/assets/2026-06-18-reactive-demo/run.sh
#
# The `after` lag is a model field, not a CLI param, so the three ON runs use
# IR variants generated from the one committed golden by swapping that field.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
BIN="$REPO/rust/target/release/camdl"
GOLD="$REPO/tests/fixtures/reactive/ir/reactive_sir_observed_threshold.ir.json"

[ -x "$BIN" ] || { echo "build camdl first:  (cd $REPO && make build-rust)" >&2; exit 1; }
export CAMDL_SKIP_VERSION_CHECK=1   # running an IR directly; no camdlc handshake

D="$(mktemp -d)"; trap 'rm -rf "$D"' EXIT
P=(--param beta=0.3 --param gamma=0.1 --param rho=0.2 --param trigger_threshold=2
   --param sia_cov=0.7 --param N0=1000 --param I0=10)

# IR variants: identical model, different reactive `after` lag (0 / 21 / 42 d).
python3 - "$GOLD" "$D" <<'PY'
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
out = pathlib.Path(sys.argv[2])
assert src.count('"after":21.0') == 1, "expected exactly one reactive `after` field"
for a in (0, 21, 42):
    (out / f"ir_a{a}.json").write_text(src.replace('"after":21.0', f'"after":{float(a)}'))
PY

run() {  # name  ir  [extra args...]
  local name=$1 ir=$2; shift 2
  "$BIN" simulate "$D/$ir" "${P[@]}" --seed 1 --backend chain_binomial --dt 1.0 \
    -o "$D/traj_$name.tsv" "$@" --output-dir "$D/cas_$name" >/dev/null
}

run baseline ir_a21.json                                              # reactive OFF
run after0   ir_a0.json  --enable sia --reactive-log "$D/rx_after0.tsv"
run after21  ir_a21.json --enable sia --reactive-log "$D/rx_after21.tsv"
run after42  ir_a42.json --enable sia --reactive-log "$D/rx_after42.tsv"

uv run "$HERE/plot.py" "$D" "$HERE/reactive_demo.png"
