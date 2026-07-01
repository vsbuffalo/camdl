#!/usr/bin/env bash
set -euo pipefail

CAMDLC=${CAMDLC:-camdlc}
CAMDL=${CAMDL:-camdl}
GOLDEN=ocaml/golden
PASS=0
FAIL=0

# Single temp file reused across iterations; cleaned up on any exit.
tmpir=$(mktemp /tmp/camdl_XXXXXX)
trap 'rm -f "$tmpir"' EXIT

for camdl in "$GOLDEN"/*.camdl; do
    name=$(basename "$camdl" .camdl)

    if ! "$CAMDLC" "$camdl" > "$tmpir"; then
        echo "FAIL [compile] $name"
        FAIL=$((FAIL + 1))
        continue
    fi

    # Prefer an explicit .params.toml; otherwise use the first scenario in the IR.
    if [ -f "$GOLDEN/$name.params.toml" ]; then
        params_flag="--params $GOLDEN/$name.params.toml"
    else
        first_scenario=$(python3 -c "
import json, sys
# gh#audit-C8: IR is now wrapped in { ir_version, validated_by, model }.
# Descend into envelope.model when present; fall back to top-level
# for any future bare-Model JSON.
env = json.load(open('$tmpir'))
m = env.get('model', env)
s = m.get('scenarios', [])
print(s[0]['name'] if s else '')
" 2>/dev/null || echo "")
        if [ -n "$first_scenario" ]; then
            params_flag="--scenario $first_scenario"
        else
            params_flag=""
        fi
    fi

    ok=1
    for backend in gillespie chain_binomial; do
        tmperr=$(mktemp /tmp/camdl_err_XXXXXX)
        # gh#audit-C6: --allow-degenerate-rates restores the legacy
        # silent-zero on rate-eval collapse. The integration test
        # asserts backend dispatch + IR round-trip work for every
        # golden — not that every golden has explicit Cond guards
        # against empty-stratum dividers (sir_five_age in particular
        # legitimately produces 0/0 in the inner age sum at t=0).
        # Keep legacy mode here; the strict-mode contract is asserted
        # by sim/tests/expr_eval.rs::test_*_errors_by_default.
        # shellcheck disable=SC2086
        if ! "$CAMDL" simulate "$tmpir" $params_flag --backend "$backend" \
                --seed 42 --allow-degenerate-rates > /dev/null 2>"$tmperr"; then
            # Expected: model needs features this backend doesn't support, so
            # the backend cleanly REFUSES it. Two refusal classes:
            #  - capability gate (e.g. overdispersed() → OVERDISPERSION, which
            #    Gillespie/ODE reject): "does not support required capabilities"
            #    (dispatch guard in rust/crates/sim/src/lib.rs);
            #  - structural validation (gh#121): a multi-source transition
            #    (`A + B --> C`, e.g. the `bimolecular` golden) is bounded by
            #    only the first source on chain_binomial, so it is rejected with
            #    "... not supported on chain_binomial ..." — gillespie/ode run it.
            # Keep these in sync if either message is reworded.
            if grep -qE "does not support required capabilities|not supported on chain_binomial" "$tmperr"; then
                rm -f "$tmperr"
                continue
            fi
            echo "FAIL [$backend] $name"
            ok=0
            FAIL=$((FAIL + 1))
        fi
        rm -f "$tmperr"
    done

    if [ $ok -eq 1 ]; then
        echo "PASS $name"
        PASS=$((PASS + 1))
    fi
done

# ── Batch pipeline tests ─────────────────────────────────────────────────────
#
# Previously named `experiment` — the subcommand was renamed to
# `batch run` on 2026-04-17 (commit 4d1291b). The `summarize`
# sub-subcommand was removed at the same time (see
# docs/dev/proposals/2026-04-16-cas-simulate.md); trajectory aggregation
# now lives in `camdl list` / `camdl cat`.

run_batch_test() {
    local fixture="$1"        # e.g. tests/fixtures/exp_sir_basic.toml
    local expected_runs="$2"  # e.g. 50
    local name
    name=$(basename "$fixture" .toml)

    local outdir
    outdir=$(mktemp -d /tmp/camdl_batch_XXXXXX)
    trap "rm -rf '$outdir'" RETURN

    # run
    # gh#audit-C6: --allow-degenerate-rates for the same reason as the
    # per-backend simulate loop above (sir_five_age has empty-stratum
    # divisors that the new strict-mode would reject).
    if ! "$CAMDL" batch run "$fixture" --output-dir "$outdir" --parallel 2 \
            --allow-degenerate-rates > /dev/null; then
        echo "FAIL [batch run] $name"; FAIL=$((FAIL+1)); return
    fi

    # check completed run count. manifest.json was retired with the CAS
    # store (gh#147); count the per-cell sim leaves the batch wrote via the
    # browse surface instead. `list --format json` emits one JSON object per
    # run as NDJSON (plus a trailing `[]`), so count the object lines; `--all`
    # lifts the default 50-row limit (some fixtures run 60).
    local completed
    completed=$("$CAMDL" list --root "$outdir" --kind sim --all --format json 2>/dev/null \
        | python3 -c "import sys, json; print(sum(1 for l in sys.stdin if l.strip() and isinstance(json.loads(l), dict)))")
    if [ "$completed" -ne "$expected_runs" ]; then
        echo "FAIL [run count] $name: expected $expected_runs runs, got $completed"
        FAIL=$((FAIL+1)); return
    fi

    # resume is a no-op (re-run without --force, check it succeeds)
    if ! "$CAMDL" batch run "$fixture" --output-dir "$outdir" --parallel 2 \
            --allow-degenerate-rates > /dev/null; then
        echo "FAIL [resume] $name"; FAIL=$((FAIL+1)); return
    fi

    echo "PASS [batch] $name"
    PASS=$((PASS+1))
}

run_batch_test tests/fixtures/exp_malaria.toml               60
run_batch_test tests/fixtures/exp_sir_basic.toml             50
run_batch_test tests/fixtures/exp_seir_erlang.toml           40
run_batch_test tests/fixtures/exp_sir_five_age.toml          40
run_batch_test tests/fixtures/exp_sir_patches_5.toml         40
run_batch_test tests/fixtures/exp_seir_vaccine.toml          30
run_batch_test tests/fixtures/exp_seir_vaccine_seasonal.toml 30
run_batch_test tests/fixtures/exp_polio_spatial_5.toml       45

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
