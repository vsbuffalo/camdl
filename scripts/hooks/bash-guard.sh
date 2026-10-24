#!/usr/bin/env bash
# Claude Code PreToolUse hook (matcher: Bash). Blocks the git invocations that
# CLAUDE.md forbids: --no-verify (and core.hooksPath, its config-level twin),
# git add -A/--all/., git commit -a/--all, and any force-push (--force,
# --force-with-lease, -f, +refspec). Reads the hook
# input JSON on stdin; exit 2 with a message on stderr blocks the call.
# Matching is on the raw command string, so a forbidden form inside a quoted
# argument is also blocked: fail closed. Policy: docs/dev/agentic-burn-down.md.
set -euo pipefail

cmd=$(jq -r '.tool_input.command // empty')
[ -z "$cmd" ] && exit 0

# A git invocation, allowing `git -C <dir>` / `git -c k=v` before the verb.
git_pre='(^|[;&|(`[:space:]])git([[:space:]]+-[Cc][[:space:]]+[^[:space:]]+)*[[:space:]]+'
args='([^;&|]*[[:space:]])?'
end='([[:space:];&|)]|$)'

block() {
  echo "blocked by scripts/hooks/bash-guard.sh: $1 (CLAUDE.md, \"Never lower the bar\" / docs/dev/agentic-burn-down.md). Stage files by name; never skip hooks or force-push." >&2
  exit 2
}

if grep -Eq -- "(^|[[:space:]])--no-verify([[:space:]=]|$)" <<<"$cmd"; then
  block "--no-verify"
fi
if grep -Eq -- "core\.hooksPath" <<<"$cmd"; then
  block "core.hooksPath (skips hooks like --no-verify)"
fi
if grep -Eq -- "${git_pre}add[[:space:]]+${args}(-A|--all|\.)${end}" <<<"$cmd"; then
  block "git add -A / --all / ."
fi
if grep -Eq -- "${git_pre}commit[[:space:]]+${args}(-[[:alpha:]]*a[[:alpha:]]*|--all)${end}" <<<"$cmd"; then
  block "git commit -a / --all"
fi
if grep -Eq -- "${git_pre}push[[:space:]]+${args}(--force(-with-lease)?(=[^[:space:]]*)?|-[[:alpha:]]*f[[:alpha:]]*|\+[^[:space:]]+)${end}" <<<"$cmd"; then
  block "force-push"
fi
exit 0
