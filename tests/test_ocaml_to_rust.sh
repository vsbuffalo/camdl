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

# ── Store pipeline tests ─────────────────────────────────────────────────────
#
# The one end-to-end multi-model store run in the gate: eight golden models,
# each run as a `simulate --sweep … --replicates` grid into a fresh store, the
# cell count checked through the browse surface, then the same grid run again
# to check that resume reads every cell back from the store and simulates none
# (proposal docs/dev/proposals/2026-10-24-batch-to-simulate.md §6).

run_store_test() {
    local name="$1"            # golden model stem, e.g. sir_basic
    local expected_runs="$2"   # cells = sweep points × scenarios × replicates
    shift 2                    # the rest: this model's simulate flags

    local model="$GOLDEN/$name.camdl"
    local params="$GOLDEN/$name.params.toml"
    local outdir tmperr
    outdir=$(mktemp -d /tmp/camdl_store_XXXXXX)
    tmperr=$(mktemp /tmp/camdl_store_err_XXXXXX)
    trap "rm -rf '$outdir' '$tmperr'" RETURN

    # run. gh#audit-C6: --allow-degenerate-rates for the same reason as the
    # per-backend simulate loop above (sir_five_age has empty-stratum
    # divisors that strict mode would reject). --parallel 2 as the grid ran
    # before this test moved onto `simulate`; `simulate` plans its cells
    # sequentially whatever the flag says (run-spec §4.4), so the flag is
    # carried for the day it is honoured, not exercised today.
    if ! "$CAMDL" simulate "$model" --params "$params" "$@" \
            --output-dir "$outdir" --parallel 2 --allow-degenerate-rates \
            > /dev/null 2>"$tmperr"; then
        echo "FAIL [simulate] $name"; cat "$tmperr"; FAIL=$((FAIL+1)); return
    fi

    # check the cell count: one `sim` leaf per cell, counted through the browse
    # surface. `list --format json` emits one JSON object per run as NDJSON
    # (plus a trailing `[]`), so count the object lines; `--all` lifts the
    # default 50-row limit (some grids run 60).
    local completed
    completed=$("$CAMDL" list --root "$outdir" --kind sim --all --format json 2>/dev/null \
        | python3 -c "import sys, json; print(sum(1 for l in sys.stdin if l.strip() and isinstance(json.loads(l), dict)))")
    if [ "$completed" -ne "$expected_runs" ]; then
        echo "FAIL [run count] $name: expected $expected_runs runs, got $completed"
        FAIL=$((FAIL+1)); return
    fi

    # resume: the same grid again, without --force, must succeed and read
    # every cell back from the store — "N of N cells read from the store;
    # simulated 0" on stderr.
    if ! "$CAMDL" simulate "$model" --params "$params" "$@" \
            --output-dir "$outdir" --parallel 2 --allow-degenerate-rates \
            > /dev/null 2>"$tmperr"; then
        echo "FAIL [resume] $name"; cat "$tmperr"; FAIL=$((FAIL+1)); return
    fi
    if ! grep -qE "read from the store; simulated 0([^0-9]|$)" "$tmperr"; then
        echo "FAIL [resume] $name: the second run did not read every cell back"
        cat "$tmperr"; FAIL=$((FAIL+1)); return
    fi

    echo "PASS [store] $name"
    PASS=$((PASS+1))
}

# Each grid keeps the cell count the retired manifest ran. A preset that
# `set`s a parameter refuses a sweep over it (gh#572), so the models whose
# presets pin every parameter sweep under the implicit arm, with the base
# values from the params file; `with_sia` only enables interventions, so it
# composes with a sweep.
run_store_test malaria_two_species   60 --backend gillespie \
    --sweep beta_BH=0.15,0.3 --sweep beta_AH=0.1,0.2 --replicates 15
run_store_test sir_basic             50 --backend gillespie \
    --sweep beta=0.2,0.5 --replicates 25
run_store_test seir_erlang           40 --backend gillespie \
    --sweep beta=0.3,0.45 --sweep sigma=0.2,0.6 --replicates 10
run_store_test sir_five_age          40 --backend chain_binomial --dt 1.0 \
    --sweep beta=0.2,0.4 --replicates 20
run_store_test sir_patches_5         40 --backend chain_binomial --dt 0.5 \
    --sweep gamma=0.07,0.2 --replicates 20
run_store_test seir_vaccine          30 --backend chain_binomial --dt 1.0 \
    --scenario with_sia --sweep vacc_frac=0.0,0.8 --replicates 15
run_store_test seir_vaccine_seasonal 30 --backend chain_binomial --dt 1.0 \
    --scenario with_sia --sweep vacc_frac=0.0,0.8 --replicates 15
run_store_test polio_spatial_5       45 --backend chain_binomial --dt 1.0 \
    --scenario with_sia --sweep kappa=0.0,0.01,0.05 --replicates 15

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
