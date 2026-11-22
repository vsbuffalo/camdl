#!/usr/bin/env bash
# Tests for scripts/gate-stamp.sh, which records the source trees that passed
# `make test` so the pre-push hook can skip re-running the gate on them.
#
# Covers the properties a stamp must hold to be trusted:
#   - a clean checkout yields HEAD's tree, and a stamp for it is found;
#   - a tracked change, or an untracked file under a gated directory, yields
#     no tree, so no stamp can be written;
#   - an untracked file outside the gated directories (a gate log) does not;
#   - a checkout that changed during the gate gets no stamp;
#   - a commit with a different tree is not covered, one with the same tree is;
#   - a stamp written in one worktree is found from another;
#   - a passed camdl-cloud run of the whole gate covers its commit's tree, and
#     a run of part of the gate, a failed run, a run of an unknown commit or
#     an unreadable listing does not.
#
# Every case runs in a private scratch repository.
# Run: bash tests/gate_stamp_test.sh

set -uo pipefail
# Only the cases that put a stand-in camdl-cloud on PATH consult the cloud.
export CAMDL_GATE_CLOUD=0

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad() { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
check() { local d="$1"; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }

repo="$WORK/repo"
mkdir -p "$repo/scripts" "$repo/rust/crates/x/tests"
cp "$ROOT/scripts/gate-stamp.sh" "$repo/scripts/"
echo 'fn main() {}' > "$repo/rust/crates/x/tests/a.rs"
cd "$repo" || exit 1
git init -q
git -c user.name=t -c user.email=t@t commit -q --allow-empty -m root
git add -A
git -c user.name=t -c user.email=t@t commit -q -m one

stamp() { bash scripts/gate-stamp.sh "$@"; }
head_tree="$(git rev-parse 'HEAD^{tree}')"

echo "clean checkout"
check "tree prints HEAD's tree" test "$(stamp tree)" = "$head_tree"
check "an unstamped commit is not covered" bash -c '! bash scripts/gate-stamp.sh check HEAD'
stamp write "$(stamp tree)" > /dev/null
check "a written stamp covers HEAD" stamp check HEAD
check "it does not cover a commit with another tree" bash -c '! bash scripts/gate-stamp.sh check HEAD~1'
check "check with no commits fails" bash -c '! bash scripts/gate-stamp.sh check'

echo "dirty checkout"
echo '// edit' >> rust/crates/x/tests/a.rs
check "a tracked change yields no tree" test -z "$(stamp tree)"
git checkout -q -- rust/crates/x/tests/a.rs
echo 'fn main() {}' > rust/crates/x/tests/b.rs
check "an untracked test file yields no tree" test -z "$(stamp tree)"
rm rust/crates/x/tests/b.rs
echo log > gate.log
check "an untracked file at the root is ignored" test "$(stamp tree)" = "$head_tree"
rm gate.log

echo "checkout changed during the gate"
rm -rf .git/camdl-gate-passed
git -c user.name=t -c user.email=t@t commit -q --allow-empty -m same-tree
started="$(stamp tree)"
echo 'fn f() {}' > rust/crates/x/tests/c.rs
git add rust/crates/x/tests/c.rs
git -c user.name=t -c user.email=t@t commit -q -m two
stamp write "$started" > /dev/null 2>&1
check "no stamp for the tree the gate started on" bash -c '! bash scripts/gate-stamp.sh check HEAD~1'
check "no stamp for the tree it ended on" bash -c '! bash scripts/gate-stamp.sh check HEAD'
git checkout -q HEAD~1
stamp write "$(stamp tree)" > /dev/null
check "an empty commit on a stamped tree is covered" stamp check HEAD
check "so is the commit before it, with the same tree" stamp check HEAD~1
git checkout -q -

echo "worktrees share stamps"
git worktree add -q "$WORK/wt" HEAD~1
check "a stamp from the main checkout is found from a worktree" \
  bash -c "cd '$WORK/wt' && bash scripts/gate-stamp.sh check HEAD"

echo "a passed camdl-cloud run counts as a stamp"
rm -rf .git/camdl-gate-passed
git -c user.name=t -c user.email=t@t commit -q --allow-empty -m cloud
cloud_head="$(git rev-parse HEAD)"
fake="$WORK/bin"; mkdir -p "$fake"
# A stand-in for `camdl-cloud runs --json`: prints the file CLOUD_RUNS names.
printf '#!/usr/bin/env bash\n[ "$1 $2" = "runs --json" ] && cat "$CLOUD_RUNS"\n' > "$fake/camdl-cloud"
chmod +x "$fake/camdl-cloud"
run_json() { # target outcome commit
  printf '[{"spec":{"name":"r1"},"test":{"target":"%s","outcome":"%s","commit":"%s"}}]\n' "$1" "$2" "$3"
}
fails() { ! "$@"; }
cloud() { CAMDL_GATE_CLOUD=1 PATH="$fake:$PATH" CLOUD_RUNS="$WORK/runs.json" bash scripts/gate-stamp.sh check "$@" 2> /dev/null; }
run_json test passed "$cloud_head" > "$WORK/runs.json"
check "a passed full-gate run covers its commit" cloud HEAD
check "and a commit with the same tree" cloud HEAD~1
check "but not a commit with another tree" fails cloud HEAD~2
run_json test-rust passed "$cloud_head" > "$WORK/runs.json"
check "a run of part of the gate does not count" fails cloud HEAD
run_json test failed "$cloud_head" > "$WORK/runs.json"
check "a failed run does not count" fails cloud HEAD
run_json test passed 0123456789abcdef0123456789abcdef01234567 > "$WORK/runs.json"
check "a run of a commit this repository lacks does not count" fails cloud HEAD
echo 'not json' > "$WORK/runs.json"
check "an unreadable listing does not count" fails cloud HEAD
run_json test passed "$cloud_head" > "$WORK/runs.json"
check "CAMDL_GATE_CLOUD=0 ignores the cloud" bash -c \
  "! CAMDL_GATE_CLOUD=0 PATH='$fake:$PATH' CLOUD_RUNS='$WORK/runs.json' bash scripts/gate-stamp.sh check HEAD"

echo
echo "gate_stamp_test: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
