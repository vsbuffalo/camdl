#!/usr/bin/env bash
# test-slot.sh N CMD... — run CMD while holding one of N machine-wide slots.
#
# A counting semaphore for camdl's heavy test targets, so that many sessions
# and agents on one machine queue instead of all running the full suite at
# once (seven concurrent `make test` runs reached a load of ~145 on a 16-core
# machine). The Makefile wraps `test`, `test-rust`, `test-inference` and
# `test-integration` in it; see docs/dev/testing.md, "Tiered gate".
#
# Slots are the files slot.0 .. slot.(N-1) in one shared directory:
#   $CAMDL_TEST_SLOT_DIR if set; else /tmp/claude-<uid>/camdl-test-slots when
#   /tmp/claude-<uid> exists (the temp directory every Claude Code session of
#   this user shares); else ${TMPDIR:-/tmp}/camdl-test-slots.
# A slot is held by an exclusive flock(2) on its file, taken without waiting
# (`lockf -t 0` on macOS, `flock -n` on Linux) and polled every two seconds
# while all N are taken. The lock belongs to this script's open file, so it
# is released when this script exits for any reason, SIGKILL included. CMD
# runs with the lock descriptor closed: a daemon that CMD starts and leaves
# behind (an sccache server, say) cannot inherit and pin the slot.
#
# N = 0 disables the queue (CI runs one job per machine), and so does
# CAMDL_TEST_SLOT_HELD being set: it is exported to CMD, so a nested make
# under a held slot runs straight through instead of taking a second one.
#
# Exit status is CMD's.

set -uo pipefail

usage() { echo "usage: test-slot.sh N CMD [ARGS...]" >&2; exit 64; }
[ $# -ge 2 ] || usage
n="$1"; shift
case "$n" in ''|*[!0-9]*) echo "test-slot.sh: N must be a non-negative integer, got '$n'" >&2; exit 64 ;; esac

if [ "$n" -eq 0 ] || [ -n "${CAMDL_TEST_SLOT_HELD:-}" ]; then
  exec "$@"
fi

if [ -n "${CAMDL_TEST_SLOT_DIR:-}" ]; then
  dir="$CAMDL_TEST_SLOT_DIR"
elif [ -d "/tmp/claude-$(id -u)" ]; then
  dir="/tmp/claude-$(id -u)/camdl-test-slots"
else
  dir="${TMPDIR:-/tmp}"; dir="${dir%/}/camdl-test-slots"
fi
mkdir -p "$dir" || { echo "test-slot.sh: cannot create slot directory $dir" >&2; exit 73; }

# try_lock: take an exclusive lock on fd 9 without waiting. Returns 0 when
# taken, 75 (EX_TEMPFAIL) when another process holds it, anything else on error.
if command -v lockf >/dev/null 2>&1; then
  try_lock() { lockf -s -t 0 9; }
elif command -v flock >/dev/null 2>&1; then
  try_lock() { flock -n -E 75 9; }
else
  echo "test-slot.sh: neither lockf (macOS) nor flock (Linux) found; set CAMDL_TEST_SLOTS=0 to run unqueued" >&2
  exit 69
fi

said_waiting=
while :; do
  i=0
  while [ "$i" -lt "$n" ]; do
    exec 9>>"$dir/slot.$i" || { echo "test-slot.sh: cannot open $dir/slot.$i" >&2; exit 73; }
    try_lock; rc=$?
    if [ "$rc" -eq 0 ]; then
      break 2
    fi
    exec 9>&-
    if [ "$rc" -ne 75 ]; then
      echo "test-slot.sh: locking $dir/slot.$i failed (exit $rc)" >&2
      exit 70
    fi
    i=$((i + 1))
  done
  if [ -z "$said_waiting" ]; then
    echo "waiting for a test slot ($n in use; slots in $dir)" >&2
    said_waiting=1
  fi
  sleep 2
done

export CAMDL_TEST_SLOT_HELD=1
"$@" 9>&-
exit $?
