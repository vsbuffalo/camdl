#!/usr/bin/env bash
# gate-stamp.sh — record which source trees have passed `make test`, so the
# pre-push hook need not run the same gate a second time on the same content.
#
#   gate-stamp.sh tree           print HEAD's tree hash if the checkout is clean
#                                (see below); print nothing otherwise
#   gate-stamp.sh write TREE     record TREE as passed, if TREE is non-empty and
#                                the checkout is still clean at TREE
#   gate-stamp.sh check SHA...   exit 0 if the tree of every commit SHA is
#                                recorded as passed, 1 otherwise
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
    for sha in "$@"; do
      tree="$(git rev-parse "$sha^{tree}" 2>/dev/null)" || exit 1
      [ -f "$dir/$tree" ] || exit 1
    done
    ;;
  *)
    echo "usage: gate-stamp.sh tree | write TREE | check SHA..." >&2
    exit 64
    ;;
esac
