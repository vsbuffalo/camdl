#!/usr/bin/env bash
# gate-stamp.sh — record which source trees have passed `make test`, so the
# pre-push hook need not run the same gate a second time on the same content.
#
#   gate-stamp.sh tree           print HEAD's tree hash if the checkout is clean
#                                (see below); print nothing otherwise
#   gate-stamp.sh write TREE     record TREE as passed, if TREE is non-empty and
#                                the checkout is still clean at TREE
#   gate-stamp.sh check SHA...   exit 0 if the tree of every commit SHA is
#                                recorded as passed, or passed on camdl-cloud
#                                (below), 1 otherwise
#
# A stamp names a git tree hash: the hash of every tracked file's content and
# path. Two commits with the same tree compile and test identically, whatever
# their messages or parents, so a stamp written in one worktree also covers the
# same content pushed from another. Stamps live in
# <git common dir>/camdl-gate-passed/, shared by every worktree of the
# repository, and are removed after 14 days.
#
# "Clean" means the gate cannot have read anything the tree hash does not
# cover: no staged or unstaged change to a tracked file, and no untracked,
# unignored file under a directory the gate reads (cargo compiles every file in
# a crate's tests/ directory whether or not git tracks it). Untracked files at
# the repository root, such as a gate's own log, do not count.
#
# What a stamp does not cover: state outside the repository that the gate also
# reads, such as the installed opam packages. CI runs the full gate on every
# push and pull request, so a stamp only ever saves the local re-run.
#
# A passed camdl-cloud run counts as a stamp. `camdl-cloud test --rev SHA`
# runs `make test` on a machine of its own, in a fresh clone of the commit
# that it refuses to run unless the tree is clean, and records the outcome in
# its runs bucket. `check` asks `camdl-cloud runs --json` for runs whose target
# is the whole gate (`test`) and whose outcome is `passed`, and accepts a
# commit whose tree is the tree of one of them. Without camdl-cloud on PATH,
# or with CAMDL_GATE_CLOUD=0, only local stamps count.

set -euo pipefail

GATED_PATHS=(rust ocaml tests ir scripts docs .githooks Makefile)

stamp_dir() {
  local common
  common="$(git rev-parse --git-common-dir)"
  printf '%s/camdl-gate-passed' "$(cd "$common" && pwd -P)"
}

clean_tree() {
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    return 0
  fi
  if [ -n "$(git ls-files --others --exclude-standard -- "${GATED_PATHS[@]}")" ]; then
    return 0
  fi
  git rev-parse 'HEAD^{tree}'
}

# Print "<tree> <run name>" for every passed full-gate camdl-cloud run whose
# commit this repository has. Prints nothing when camdl-cloud is unavailable
# or its listing cannot be read.
cloud_passed() {
  [ "${CAMDL_GATE_CLOUD:-1}" != 0 ] || return 0
  command -v camdl-cloud > /dev/null 2>&1 || return 0
  command -v python3 > /dev/null 2>&1 || return 0
  local listing
  listing="$(camdl-cloud runs --json 2> /dev/null)" || return 0
  printf '%s' "$listing" | python3 -c '
import json, sys
try:
    runs = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for run in runs if isinstance(runs, list) else []:
    t = run.get("test") or {}
    if t.get("target") == "test" and t.get("outcome") == "passed" and t.get("commit"):
        print(t["commit"], run.get("spec", {}).get("name", "?"))
' | while read -r commit name; do
    tree="$(git rev-parse -q --verify "$commit^{tree}" 2> /dev/null)" || continue
    printf '%s %s\n' "$tree" "$name"
  done
}

case "${1:-}" in
  tree)
    clean_tree
    ;;
  write)
    tree="${2:-}"
    if [ -z "$tree" ]; then
      echo "gate-stamp: checkout was not clean when the gate started; no stamp written" >&2
      exit 0
    fi
    now="$(clean_tree)"
    if [ "$now" != "$tree" ]; then
      echo "gate-stamp: checkout changed during the gate; no stamp written" >&2
      exit 0
    fi
    dir="$(stamp_dir)"
    mkdir -p "$dir"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$dir/$tree"
    find "$dir" -type f -mtime +14 -delete
    echo "gate-stamp: tree ${tree:0:12} passed make test"
    ;;
  check)
    shift
    [ $# -gt 0 ] || exit 1
    dir="$(stamp_dir)"
    cloud=""
    cloud_read=0
    for sha in "$@"; do
      tree="$(git rev-parse "$sha^{tree}" 2>/dev/null)" || exit 1
      [ -f "$dir/$tree" ] && continue
      if [ "$cloud_read" = 0 ]; then
        cloud="$(cloud_passed)"
        cloud_read=1
      fi
      run="$(printf '%s\n' "$cloud" | awk -v t="$tree" '$1 == t { print $2; exit }')"
      [ -n "$run" ] || exit 1
      echo "gate-stamp: ${sha:0:12} passed make test on camdl-cloud (run $run)" >&2
    done
    ;;
  *)
    echo "usage: gate-stamp.sh tree | write TREE | check SHA..." >&2
    exit 64
    ;;
esac
