#!/usr/bin/env bash
# Table test for the PreToolUse hooks in this directory: feeds each script the
# hook-input JSON Claude Code would send and checks the exit code (2 = blocked,
# 0 = allowed). Covers the forbidden forms and the near-misses that must pass.
# Run from the repo root: scripts/hooks/test-hooks.sh
set -uo pipefail

here=$(dirname "$0")
fail=0

check() { # expected-rc actual-rc label
  if [ "$1" = "$2" ]; then r=ok; else r=FAIL; fail=1; fi
  printf '%-4s expect=%s got=%s  %s\n' "$r" "$1" "$2" "$3"
}

bash_case() { # expected-rc command
  printf '%s' "$2" | jq -Rs '{tool_name: "Bash", cwd: "/x", tool_input: {command: .}}' |
    "$here/bash-guard.sh" >/dev/null 2>&1
  check "$1" $? "$2"
}

edit_case() { # expected-rc CAMDL_BATCH cwd file_path
  jq -n --arg c "$3" --arg p "$4" '{tool_name: "Edit", cwd: $c, tool_input: {file_path: $p}}' |
    env CAMDL_BATCH="$2" "$here/high-risk-edit-guard.sh" >/dev/null 2>&1
  check "$1" $? "CAMDL_BATCH=$2 $4"
}

echo "bash-guard.sh"
bash_case 2 'git commit --no-verify -m x'
bash_case 2 'git add -A'
bash_case 2 'git add --all'
bash_case 2 'git add .'
bash_case 2 'cd rust && git add . && git commit -m x'
bash_case 2 'git commit -a -m x'
bash_case 2 'git commit -am x'
bash_case 2 'git commit --all'
bash_case 2 'git push --force'
bash_case 2 'git push -f origin x'
bash_case 2 'git push --force-with-lease origin x'
bash_case 2 'git push origin +main'
bash_case 2 'git -C ../wt push -f'
bash_case 2 'git -c core.hooksPath=/dev/null commit -m x'
bash_case 0 'git add ./scripts/hooks/a.sh docs/x.md'
bash_case 0 'git add rust/crates/cli/src/fit/fit_summary.rs'
bash_case 0 'git commit -F msg.txt'
bash_case 0 'git commit --amend -F msg.txt'
bash_case 0 'git push origin HEAD:refs/heads/fix/gh972-chain-ids'
bash_case 0 'git push -u origin fix/x'
bash_case 0 'git status --porcelain'
bash_case 0 'cargo test -p cli --lib'

echo "high-risk-edit-guard.sh"
w=/repo/.claude/worktrees/agent-x
edit_case 2 1 "$w" "$w/rust/crates/sim/src/inference/pgas.rs"
edit_case 2 1 "$w" "$w/rust/crates/sim/src/inference/pgas_grad.rs"
edit_case 2 1 "$w" "$w/rust/crates/sim/src/inference/obs_loglik.rs"
edit_case 2 1 "$w" "$w/rust/crates/sim/src/inference/obs_model.rs"
edit_case 2 1 "$w" "$w/rust/crates/sim/src/inference/if2.rs"
edit_case 2 1 "$w" "$w/rust/crates/sim/src/inference/particle_filter.rs"
edit_case 2 1 "$w" "$w/rust/crates/sim/src/chain_binomial.rs"
edit_case 2 1 "$w" "rust/crates/sim/src/inference/pgas.rs"
edit_case 0 0 "$w" "$w/rust/crates/sim/src/inference/pgas.rs"
edit_case 0 1 "$w" "$w/rust/crates/cli/src/fit/fit_summary.rs"
edit_case 0 1 "$w" "$w/rust/crates/sim/tests/if2.rs"

exit $fail
