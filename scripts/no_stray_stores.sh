#!/usr/bin/env bash
# Runs a test command and fails if it leaves a run store (`camdl-store/`) or an
# old results root (`results/`) in the repository root or in the cli crate's
# directory, the working directory of the cli tests. A test that runs a
# command on the store must point it at a temporary store of its own
# (`CAMDL_STORE` or `--store`); one that does not writes into the checkout,
# and the next test that lists a store from there reads its runs.
#
# A directory that was already there before the command is reported but does
# not fail the run, so a leftover from an earlier run is named rather than
# blamed on this one.
#
# Usage: scripts/no_stray_stores.sh -- COMMAND [ARGS...]
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
paths=("$root/camdl-store" "$root/results" "$root/rust/crates/cli/camdl-store" "$root/rust/crates/cli/results")

[ "${1:-}" = "--" ] && shift
before=()
for p in "${paths[@]}"; do
    if [ -e "$p" ]; then
        before+=("$p")
        echo "no_stray_stores: $p was there before the tests; remove it" >&2
    fi
done

"$@"
status=$?

for p in "${paths[@]}"; do
    [ -e "$p" ] || continue
    pre=0
    for b in "${before[@]+"${before[@]}"}"; do [ "$b" = "$p" ] && pre=1; done
    if [ "$pre" = 0 ]; then
        echo "no_stray_stores: the tests left $p; a test runs a command on the store without a store of its own" >&2
        status=1
    fi
done
exit $status
