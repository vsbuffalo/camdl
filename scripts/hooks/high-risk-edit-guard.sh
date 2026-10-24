#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Edit|Write|MultiEdit). When CAMDL_BATCH=1
# is set, blocks edits to the seven high-risk inference files that batch work
# may not touch (CLAUDE.md "High-risk surfaces" plus chain_binomial.rs, per the
# burn-down skill). The target is taken relative to the input JSON's `cwd`,
# which follows a worktree; ${CLAUDE_PROJECT_DIR} does not. Exit 2 with a
# message on stderr blocks the call. Policy: docs/dev/agentic-burn-down.md.
set -euo pipefail

[ "${CAMDL_BATCH:-}" = "1" ] || exit 0

input=$(cat)
cwd=$(jq -r '.cwd // empty' <<<"$input")
path=$(jq -r '.tool_input.file_path // empty' <<<"$input")
[ -z "$path" ] && exit 0

rel=${path#"$cwd"/}

case "$rel" in
  *rust/crates/sim/src/inference/pgas.rs | \
    *rust/crates/sim/src/inference/pgas_grad.rs | \
    *rust/crates/sim/src/inference/obs_loglik.rs | \
    *rust/crates/sim/src/inference/obs_model.rs | \
    *rust/crates/sim/src/inference/if2.rs | \
    *rust/crates/sim/src/inference/particle_filter.rs | \
    *rust/crates/sim/src/chain_binomial.rs)
    echo "blocked by scripts/hooks/high-risk-edit-guard.sh: $rel is a high-risk inference file and CAMDL_BATCH=1 is set. Batch work may not edit it; stop and report \"needs solo session\"." >&2
    exit 2
    ;;
esac
exit 0
