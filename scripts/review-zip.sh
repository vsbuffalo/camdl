#!/usr/bin/env bash
# review-zip.sh — canonical code-review zip generator.
#
# This is the single entry point for producing review zips. See
# scripts/README.md for the design rationale (why four subsystems,
# why the plumbing layer is shared, why git-archive rather than
# copy).
#
# Usage:
#   ./scripts/review-zip.sh <subsystem>    # one subsystem zip
#   ./scripts/review-zip.sh all            # every subsystem
#   ./scripts/review-zip.sh full           # whole repo (no slicing)
#   ./scripts/review-zip.sh list           # subsystems + token estimates
#   ./scripts/review-zip.sh clean          # rm review-zips/*.zip
#
# Subsystems:
#   inference   — fit algorithms (IF2/PGAS/NUTS/PMMH/PF) + fit CLI
#                 + shared plumbing. Anchor for most inference work.
#   engine      — simulation backends (Gillespie/tau-leap/ODE/CB) +
#                 propensity + shared plumbing. Anchor for simulate
#                 + observation work.
#   compiler    — OCaml DSL → IR.
#   docs        — specs, proposals, dev notes.
#
# Output: review-zips/review-<subsystem>-<YYYYMMDD>.zip
# Environment:
#   REVIEW_OUTDIR overrides the output directory (default: review-zips).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUTDIR="${REVIEW_OUTDIR:-review-zips}"
DATE=$(date +%Y%m%d)

# Global exclusions — paths stripped from every zip, applied via
# git-archive pathspec magic (':!path'). These are internal-only
# artifacts not appropriate to send to an external reviewer:
#   - docs/dev/proposals/ — in-flight RFCs
#   - docs/dev/incidents/ — internal post-mortems
# Subsystem include lists may still mention these paths; the
# exclusion wins.
REVIEW_EXCLUDES=(
    ':!docs/dev/proposals/'
    ':!docs/dev/incidents/'
)

# ─── Subsystem file lists ─────────────────────────────────────────────
#
# Each subsystem declares an array of paths (files or dirs) that
# git-archive will pull from HEAD. Keep these lists explicit — no
# wildcards beyond what git-archive naturally supports — so the scope
# of each review is auditable by reading this file.

# Shared CLI plumbing: cache/metadata/path/browse infrastructure used
# by both inference (fit) and engine (simulate) code paths. Duplicated
# into both zips rather than given its own subsystem — reviewers of
# either inference or engine work need this context to trace data
# flow end-to-end (§"Shared plumbing" in README.md).
CLI_PLUMBING=(
    rust/crates/cli/src/main.rs
    rust/crates/cli/src/browse.rs
    rust/crates/cli/src/cas/
    rust/crates/cli/src/run_meta.rs
    rust/crates/cli/src/run_paths.rs
    rust/crates/cli/src/hashing.rs
    rust/crates/cli/src/batch.rs
    rust/crates/cli/src/util.rs
    rust/crates/cli/src/version.rs
)

# Inference-relevant tests only — exercises PF / IF2 / PMMH / PGAS / NUTS,
# observation models, priors, profile/fit CLI, and the seam between them.
# Engine smoke tests and lineage/intervention/forcing oracles live in
# ENGINE_TESTS so the inference reviewer isn't paying ~100K tokens for
# tests outside their scope.
INFERENCE_TESTS=(
    rust/crates/sim/tests/if2.rs
    rust/crates/sim/tests/pmmh.rs
    rust/crates/sim/tests/pmmh_hierarchical.rs
    rust/crates/sim/tests/pgas_resume.rs
    rust/crates/sim/tests/pgas_tempering.rs
    rust/crates/sim/tests/particle_filter.rs
    rust/crates/sim/tests/obs_level_params.rs
    rust/crates/sim/tests/obs_time_dependence.rs
    rust/crates/sim/tests/hierarchical_log_density.rs
    rust/crates/sim/tests/gradient_check.rs
    rust/crates/sim/tests/multi_stream_obs.rs
    rust/crates/cli/tests/fit_experiment_management.rs
    rust/crates/cli/tests/fit_priors.rs
    rust/crates/cli/tests/synthetic_fit_grid.rs
    rust/crates/cli/tests/profile_priors.rs
    rust/crates/cli/tests/profile_diagnostics.rs
    rust/crates/cli/tests/profile_pmmh.rs
    rust/crates/cli/tests/profile_multi_stream.rs
    rust/crates/cli/tests/survey_top_k_pmmh.rs
    rust/crates/cli/tests/survey_top_k_pgas.rs
    rust/crates/cli/tests/pgas_resume.rs
    rust/crates/cli/tests/calendar_fit_summary.rs
    rust/crates/cli/tests/pfilter_trajectories.rs
)

# Engine-relevant tests: forward simulation, lineage runtime, interventions,
# forcings, conservation invariants, golden agreement. Inference-side
# observation tests cross-cut and live in INFERENCE_TESTS; that overlap
# is small enough not to be worth bundling.
ENGINE_TESTS=(
    rust/crates/sim/tests/golden_simulate.rs
    rust/crates/sim/tests/smoke_all_golden.rs
    rust/crates/sim/tests/expr_eval.rs
    rust/crates/sim/tests/resolved_expr.rs
    rust/crates/sim/tests/gillespie_determinism.rs
    rust/crates/sim/tests/gillespie_invariants.rs
    rust/crates/sim/tests/chain_binomial_invariants.rs
    rust/crates/sim/tests/bimolecular_conservation.rs
    rust/crates/sim/tests/branching_destinations.rs
    rust/crates/sim/tests/ode.rs
    rust/crates/sim/tests/erlang_distribution.rs
    rust/crates/sim/tests/statistical_distribution.rs
    rust/crates/sim/tests/sparse_propensity.rs
    rust/crates/sim/tests/scenario_application.rs
    rust/crates/sim/tests/interventions.rs
    rust/crates/sim/tests/intervention_dt_invariance.rs
    rust/crates/sim/tests/periodic_forcing.rs
    rust/crates/sim/tests/periodic_bspline_oracle.rs
    rust/crates/sim/tests/fourier_oracle.rs
    rust/crates/sim/tests/cubic_spline.rs
    rust/crates/sim/tests/interpolation.rs
    rust/crates/sim/tests/simplex.rs
    rust/crates/sim/tests/spatial_density.rs
    rust/crates/sim/tests/snapshot_projections.rs
    rust/crates/sim/tests/rng_extreme_inputs.rs
    rust/crates/sim/tests/lineage_runtime.rs
    rust/crates/sim/tests/lineage_stratified.rs
    rust/crates/sim/tests/lineage_coalescent.rs
    rust/crates/sim/tests/lineage_batch.rs
    rust/crates/sim/tests/lineage_streaming.rs
    rust/crates/sim/tests/lineage_offspring.rs
    rust/crates/sim/tests/lineage_oracle_tier5.rs
    rust/crates/sim/tests/fixtures/
    rust/crates/cli/tests/cas_integration.rs
    rust/crates/cli/tests/backend_provenance.rs
    rust/crates/cli/tests/parameter_bounds_validation.rs
    rust/crates/cli/tests/dated_data_loader.rs
    rust/crates/cli/tests/seed_timing_e2e.rs
    rust/crates/cli/tests/lineage_e2e.rs
    rust/crates/cli/tests/lineage_migration_e2e.rs
    rust/crates/cli/tests/events_backend_parity.rs
    rust/crates/cli/tests/intervention_event_defaults.rs
    rust/crates/cli/tests/scenario_runtime_application.rs
    rust/crates/cli/tests/compile_output_flag.rs
)

INFERENCE=(
    rust/crates/sim/src/inference/
    rust/crates/sim/src/compiled_model.rs
    rust/crates/sim/src/propensity.rs
    rust/crates/sim/src/resolved_expr.rs
    rust/crates/sim/src/rng.rs
    rust/crates/sim/src/error.rs
    rust/crates/sim/src/state.rs
    rust/crates/sim/src/lib.rs
    rust/crates/cli/src/fit/
    rust/crates/cli/src/pfilter.rs
    rust/crates/cli/src/if2.rs
    rust/crates/cli/src/profile.rs
    rust/crates/cli/src/profile_diagnostics.rs
    rust/crates/cli/src/sampling.rs
    rust/crates/cli/src/survey.rs
    rust/crates/ir/src/
    "${CLI_PLUMBING[@]}"
    "${INFERENCE_TESTS[@]}"
    docs/camdl-inference-spec.md
    docs/camdl-run-spec.md
    docs/inference.md
    CLAUDE.md
)

ENGINE=(
    rust/crates/sim/src/
    rust/crates/sim/Cargo.toml
    rust/crates/cli/src/eval.rs
    rust/crates/cli/src/data.rs
    rust/crates/ir/src/
    "${CLI_PLUMBING[@]}"
    "${ENGINE_TESTS[@]}"
    ocaml/golden/
    docs/runtimes.md
    docs/compartmental-ir-spec.md
    docs/camdl-run-spec.md
    CLAUDE.md
)

COMPILER=(
    ocaml/lib/
    ocaml/bin/
    ocaml/test/
    ocaml/golden/
    rust/crates/ir/src/
    docs/camdl-language-spec.md
    docs/compartmental-ir-spec.md
    CLAUDE.md
)

DOCS=(
    docs/
    CLAUDE.md
    README.md
    ocaml/golden/
)

# ─── Helpers ──────────────────────────────────────────────────────────

estimate_tokens() {
    # Approximation: 1 token ≈ 4 bytes of source text. Good enough for
    # deciding which zips to hand a reviewer in what order.
    git archive HEAD -- "$@" "${REVIEW_EXCLUDES[@]}" 2>/dev/null \
        | tar -xf - -O 2>/dev/null \
        | wc -c \
        | awk '{printf "%.0fK", $1/4/1000}'
}

make_zip() {
    local name=$1; shift
    local out="$OUTDIR/review-$name-$DATE.zip"
    mkdir -p "$OUTDIR"
    git archive HEAD --prefix="camdl/" -o "$out" -- "$@" "${REVIEW_EXCLUDES[@]}"
    local tokens
    tokens=$(estimate_tokens "$@")
    local bytes
    bytes=$(ls -l "$out" | awk '{print $5}')
    printf "  %-10s → %s (~%s tokens, %sB)\n" "$name" "$out" "$tokens" "$bytes"
}

# ─── Dispatch ─────────────────────────────────────────────────────────

cmd=${1:-help}
case "$cmd" in
    inference) make_zip inference "${INFERENCE[@]}" ;;
    engine)    make_zip engine    "${ENGINE[@]}"    ;;
    compiler)  make_zip compiler  "${COMPILER[@]}"  ;;
    docs)      make_zip docs      "${DOCS[@]}"      ;;

    all)
        echo "Generating all subsystem zips in $OUTDIR/:"
        make_zip inference "${INFERENCE[@]}"
        make_zip engine    "${ENGINE[@]}"
        make_zip compiler  "${COMPILER[@]}"
        make_zip docs      "${DOCS[@]}"
        ;;

    full)
        # Whole-repo snapshot. Useful when a reviewer needs the entire
        # tree in one blob rather than a scoped subsystem (new
        # contributor onboarding, bisection across subsystems).
        out="$OUTDIR/review-full-$DATE.zip"
        mkdir -p "$OUTDIR"
        git archive HEAD --prefix="camdl/" -o "$out" -- ":/" "${REVIEW_EXCLUDES[@]}"
        tokens=$(estimate_tokens ":/")
        bytes=$(ls -l "$out" | awk '{print $5}')
        printf "  %-10s → %s (~%s tokens, %sB)\n" "full" "$out" "$tokens" "$bytes"
        ;;

    list)
        echo "Available subsystems:"
        echo
        for sub in inference engine compiler docs; do
            case "$sub" in
                inference) tokens=$(estimate_tokens "${INFERENCE[@]}") ;;
                engine)    tokens=$(estimate_tokens "${ENGINE[@]}")    ;;
                compiler)  tokens=$(estimate_tokens "${COMPILER[@]}")  ;;
                docs)      tokens=$(estimate_tokens "${DOCS[@]}")      ;;
            esac
            printf "  %-10s ~%s tokens\n" "$sub" "$tokens"
        done
        echo
        echo "Plus:"
        echo "  all        generates every subsystem"
        echo "  full       whole-repo snapshot"
        echo "  clean      rm $OUTDIR/*.zip"
        ;;

    clean)
        if [ -d "$OUTDIR" ]; then
            rm -f "$OUTDIR"/*.zip
            echo "cleaned $OUTDIR/*.zip"
        else
            echo "no $OUTDIR/ to clean"
        fi
        ;;

    help|--help|-h)
        sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        ;;

    *)
        echo "error: unknown subcommand '$cmd'" >&2
        echo "run '$0 help' for usage" >&2
        exit 1
        ;;
esac
