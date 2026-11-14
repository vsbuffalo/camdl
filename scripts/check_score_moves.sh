#!/usr/bin/env bash
# check_score_moves.sh — record what the scoring commands print, so that a
# commit that only moves code can be shown to have changed no output.
#
# What it is for
#   Phase 1 of docs/dev/proposals/2026-11-14-scoring-report-library.md moves
#   the scoring code out of `cli` with no change to any function body. Run
#   this with the binary built at the commit before a move, then again with
#   the binary built after it; the two directories must be byte-identical:
#
#     make build-rust && scripts/check_score_moves.sh /tmp/before
#     # ... apply the move, rebuild ...
#     make build-rust && scripts/check_score_moves.sh /tmp/after
#     diff -r /tmp/before /tmp/after
#
#   It uses the binary in this checkout (rust/target/release/camdl) and
#   reads the golden IR, so no camdlc is needed. It runs in well under a
#   minute.
#
# What it runs (fixtures: tests/fixtures/score_moves/, model
# ocaml/golden/seir_observations.ir.json, two streams: weekly_cases and
# detection)
#   1. `fit run` of a.toml and b.toml: PGAS, 2 chains, seed 1. The two
#      configs differ only in the fixed reporting fraction rho. Their logs
#      are kept in OUT_DIR/work only.
#   2. `fit summary` of each fit in every format: text, json, md, latex.
#   3. `pfilter --save-prequential` of the model at fixed parameters, seed
#      1: its stdout, its stderr (the scored summary), and the .json, .tsv
#      and .streams.tsv it writes.
#   4. `compare @a @b` in table, md and json:
#        (i)   --draws 1 --replicates 1 (plug-in at the posterior mean);
#        (ii)  --draws 2 --replicates 2, so both trace combiners run (the
#              posterior mixture over draws and the replicate combination);
#        (iii) --streams weekly_cases at the settings of (ii).
#      stdout and stderr are kept separately.
#
# Layout of OUT_DIR
#   OUT_DIR/<name>           one file per output, normalized (below).
#   OUT_DIR/work/            private output root, IR cache and raw outputs;
#                            removed on exit unless
#                            CHECK_SCORE_MOVES_KEEP_WORK=1, so that
#                            `diff -r` compares outputs only.
#   Nothing is written outside OUT_DIR: CAMDL_OUTPUT_DIR and
#   CAMDL_IR_CACHE_DIR point into OUT_DIR/work.
#
# What is normalized, and why (every substitution is in normalize() below)
#   - OUT_DIR's absolute path -> <OUT_DIR>, and the repository root ->
#     <REPO>: the same run in two directories prints different paths.
#   - The build's version suffix `+<git hash>` -> `+<GIT>`: it differs at
#     every commit, which is exactly what the check compares across.
#   - Wall-clock fields of `fit summary`: the timings (`wall_time_secs`,
#     `sampling_seconds`, `total_seconds`, `age_seconds`, `ess_per_sec` and
#     the text format's "ESS/sec ... s wall" line) and the run's timestamps
#     (`timestamp`, `created_at`). They vary between two runs of the same
#     binary.
#   - The fit-level identity: the `fits/<stem>-<h8>` directory, `fit_id`,
#     `fit_hash`, `baseline_hash`, the leaf's `run_id`, and the `<stem>_<h8>`
#     row names on compare's stderr. The fit-level digest includes the
#     engine version, git hash and all (fit/cas.rs, `fit_level_digest`),
#     so every commit re-keys every fit by design and these differ between
#     any two commits. The method and seed directory hashes do not include
#     it and are left as they are.
#   Nothing else is rewritten. The `fit run` logs are not compared at all
#   (see step 1).
#
# What it does not cover (the unit tests in rust/crates/cli/src/compare.rs
# pin these as committed text instead)
#   - waived comparisons (the --allow-* flags),
#   - non-finite scores,
#   - prequential traces older than the current schema.
#
# Usage: scripts/check_score_moves.sh OUT_DIR   (OUT_DIR must not exist)

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUT_DIR" >&2
    exit 2
fi
if [[ -e "$1" ]]; then
    echo "error: $1 exists; give a new directory" >&2
    exit 2
fi

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
CAMDL="$REPO/rust/target/release/camdl"
FIX="$REPO/tests/fixtures/score_moves"
MODEL="$REPO/ocaml/golden/seir_observations.ir.json"
if [[ ! -x "$CAMDL" ]]; then
    echo "error: $CAMDL not built; run \`make build-rust\`" >&2
    exit 2
fi

mkdir -p "$1"
OUT=$(cd "$1" && pwd -P)
WORK="$OUT/work"
RAW="$WORK/raw"
mkdir -p "$RAW"
if [[ "${CHECK_SCORE_MOVES_KEEP_WORK:-0}" != 1 ]]; then
    trap 'rm -rf "$WORK"' EXIT
fi

export CAMDL_OUTPUT_DIR="$WORK/store"
export CAMDL_IR_CACHE_DIR="$WORK/ir-cache"
# The binary reads the golden IR, not a .camdl, so the camdlc handshake has
# nothing to check.
export CAMDL_SKIP_VERSION_CHECK=1
export NO_COLOR=1
unset CLICOLOR_FORCE

# run_into DIR NAME ARGS...: run camdl with stdout to DIR/NAME and stderr
# to DIR/NAME.stderr; stop on a non-zero exit.
run_into() {
    local dir=$1 name=$2; shift 2
    if ! "$CAMDL" "$@" > "$dir/$name" 2> "$dir/$name.stderr"; then
        echo "error: camdl $* failed; stderr in $dir/$name.stderr" >&2
        cat "$dir/$name.stderr" >&2
        exit 1
    fi
}
run() { run_into "$RAW" "$@"; }

# 1. The two fits, from the fixture directory so their relative paths
#    resolve as committed. Their logs stay in WORK and are not compared:
#    the two chains run in parallel and their progress lines interleave in
#    a different order on every run. What the fits computed is compared
#    through `fit summary`.
cd "$FIX"
run_into "$WORK" fit_run_a fit run a.toml --label a --seed 1
run_into "$WORK" fit_run_b fit run b.toml --label b --seed 1

# 2. fit summary, every format.
for fit in a b; do
    for fmt in text json md latex; do
        run "fit_summary_${fit}.${fmt}" fit summary "@$fit" --format "$fmt" --no-color
    done
done

# 3. pfilter --save-prequential at fixed parameters.
run pfilter.stdout pfilter "$MODEL" \
    --data weekly_cases=weekly_cases.tsv --data detection=detection.tsv \
    --param beta=0.33 --param sigma=0.25 --param gamma=0.2 --param rho=0.5 \
    --param k=10 --param p_detect=0.5 --param N0=10000 --param I0=10 \
    --particles 100 --dt 1 --seed 1 --no-store \
    --save-prequential "$RAW/prequential"

# 4. compare, three settings × three formats.
for fmt in table md json; do
    run "compare_draws1.${fmt}" compare @a @b --particles 100 --seed 1 \
        --draws 1 --replicates 1 --format "$fmt"
    run "compare_draws2.${fmt}" compare @a @b --particles 100 --seed 1 \
        --draws 2 --replicates 2 --format "$fmt"
    run "compare_streams.${fmt}" compare @a @b --particles 100 --seed 1 \
        --draws 2 --replicates 2 --streams weekly_cases --format "$fmt"
done

# normalize SRC DST: the substitutions listed at the top of this file, and
# nothing else.
normalize() {
    # The LaTeX summary escapes `_` in the paths it prints.
    local out_tex=${OUT//_/\\\\_} repo_tex=${REPO//_/\\\\_}
    sed -E \
        -e "s#${OUT}#<OUT_DIR>#g" -e "s#${out_tex}#<OUT_DIR>#g" \
        -e "s#${REPO}#<REPO>#g" -e "s#${repo_tex}#<REPO>#g" \
        -e 's#([0-9]+\.[0-9]+\.[0-9]+)\+[0-9a-f]{7,40}#\1+<GIT>#g' \
        -e 's#"(wall_time_secs|sampling_seconds|total_seconds|age_seconds|ess_per_sec)": [0-9.eE+-]+#"\1": <SECS>#g' \
        -e 's#"(timestamp|created_at)": "[^"]*"#"\1": "<TIME>"#g' \
        -e 's#ESS/sec  = [0-9.]+  \(min-param ESS ([0-9]+) / [0-9.]+s wall\)#ESS/sec  = <RATE>  (min-param ESS \1 / <SECS>s wall)#' \
        -e 's#fits/([a-z]+)-[0-9a-f]{8}#fits/\1-<FIT>#g' \
        -e 's#"(run_id|fit_hash|baseline_hash)": "[0-9a-f]{64}"#"\1": "<HASH>"#g' \
        -e 's#"fit_id": "[0-9a-f]{8}"#"fit_id": "<FIT>"#g' \
        -e 's#^compare: ([a-z]+)_[0-9a-f]{8}:#compare: \1_<FIT>:#' \
        "$1" > "$2"
}

for f in "$RAW"/*; do
    normalize "$f" "$OUT/$(basename "$f")"
done
echo "wrote $(find "$OUT" -maxdepth 1 -type f | wc -l | tr -d ' ') outputs to $OUT"
