#!/usr/bin/env bash
# Tests for scripts/test-slot.sh, the machine-wide queue for heavy test targets.
#
# Covers the properties the queue exists to hold:
#   - with N slots and N+1 jobs, at most N run at once, and N do run at once
#     (the queue limits concurrency without serialising it);
#   - a holder killed with SIGKILL releases its slot;
#   - a daemon left behind by CMD does not pin the slot;
#   - CMD's exit status is the script's, queued or not;
#   - N = 0 and CAMDL_TEST_SLOT_HELD bypass the queue (CI; nested make).
#
# Every run uses a private slot directory, so it never touches the shared one.
# Run: bash tests/test_slot_test.sh

# shellcheck disable=SC2016 # single-quoted job bodies expand in the child shell
set -uo pipefail
# Run under `make test`, this inherits the held slot's marker; drop it so the
# script under test queues.
unset CAMDL_TEST_SLOT_HELD

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLOT="$ROOT/scripts/test-slot.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad() { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
check() { local d="$1"; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }

# A fresh slot directory per test.
fresh() { local d; d="$(mktemp -d "$WORK/slots.XXXX")"; echo "$d"; }

# wait_for SECONDS CMD... — poll CMD every 0.1 s; succeed once it does.
wait_for() {
  local limit=$(( $1 * 10 )); shift
  local t=0
  while [ "$t" -lt "$limit" ]; do
    if "$@"; then return 0; fi
    sleep 0.1; t=$((t + 1))
  done
  return 1
}

# ------------------------------------------------------------- concurrency
check_concurrency() {
  echo "concurrency (2 slots, 3 jobs):"
  local d run log
  d="$(fresh)"; run="$WORK/running"; log="$WORK/conc.log"
  mkdir -p "$run"; : > "$log"
  # Each job marks itself running, records how many are running, holds for
  # 1.5 s, then unmarks. The count only includes jobs inside CMD.
  local job='touch "$1/$2"; ls "$1" | wc -l | tr -d " " >> "$3"; sleep 1.5; rm -f "$1/$2"'
  for k in 1 2 3; do
    CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 2 bash -c "$job" _ "$run" "j$k" "$log" \
      2>> "$WORK/conc.err" &
  done
  wait
  local max ran
  max="$(sort -n "$log" | tail -n 1)"
  ran="$(wc -l < "$log" | tr -d ' ')"
  check "all three jobs ran"                 [ "$ran" = 3 ]
  check "at most two ran at once (max $max)" [ "$max" -le 2 ]
  check "two did run at once (not serial)"   [ "$max" -eq 2 ]
  check "the third printed the waiting line" grep -q "waiting for a test slot (2 in use" "$WORK/conc.err"
}

# ------------------------------------------------------------ killed holder
check_killed_holder() {
  echo "killed holder:"
  local d pidf holder
  d="$(fresh)"; pidf="$WORK/sleep.pid"
  CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 1 bash -c 'echo $$ > "$1"; exec sleep 30' _ "$pidf" &
  holder=$!
  wait_for 5 test -s "$pidf"
  # The slot is held: a second job cannot start yet.
  CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 1 touch "$WORK/second" 2>/dev/null &
  local second=$!
  sleep 1
  check "a second job waits while the slot is held" [ ! -e "$WORK/second" ]
  kill -9 "$holder"; wait "$holder" 2>/dev/null
  check "after SIGKILL of the holder the second job runs" wait_for 5 test -e "$WORK/second"
  wait "$second" 2>/dev/null
  kill "$(cat "$pidf")" 2>/dev/null
}

# ------------------------------------------------------- left-behind daemon
check_daemon_does_not_pin() {
  echo "left-behind daemon:"
  local d
  d="$(fresh)"
  # CMD starts a background sleep and exits; the sleep keeps running.
  CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 1 bash -c 'sleep 20 >/dev/null 2>&1 & echo $! > "$1"' _ "$WORK/daemon.pid"
  CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 1 touch "$WORK/after-daemon" 2>/dev/null &
  local next=$!
  check "the next job runs while the daemon lives" wait_for 2 test -e "$WORK/after-daemon"
  kill "$next" 2>/dev/null; wait "$next" 2>/dev/null
  kill "$(cat "$WORK/daemon.pid")" 2>/dev/null
}

# -------------------------------------------------------------- exit status
status_of() { local d="$1"; shift; CAMDL_TEST_SLOT_DIR="$d" "$SLOT" "$@"; echo $?; }
check_exit_status() {
  echo "exit status:"
  local d; d="$(fresh)"
  check "0 propagates"                [ "$(status_of "$d" 2 true)" = 0 ]
  check "1 propagates"                [ "$(status_of "$d" 2 false)" = 1 ]
  check "7 propagates"                [ "$(status_of "$d" 2 sh -c 'exit 7')" = 7 ]
  check "75 propagates (lockf's busy code, from CMD)" [ "$(status_of "$d" 2 sh -c 'exit 75')" = 75 ]
  check "7 propagates with N = 0"     [ "$(status_of "$d" 0 sh -c 'exit 7')" = 7 ]
  check "a non-integer N is a usage error (64)" [ "$(status_of "$d" x true 2>/dev/null)" = 64 ]
  check "CMD sees CAMDL_TEST_SLOT_HELD" \
    [ "$(CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 1 sh -c 'echo "$CAMDL_TEST_SLOT_HELD"')" = 1 ]
}

# ------------------------------------------------------------------ bypass
check_bypass() {
  echo "bypass (N = 0, nested):"
  local d holder pidf; d="$(fresh)"; pidf="$WORK/bypass.pid"
  CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 1 bash -c 'echo $$ > "$1"; exec sleep 30' _ "$pidf" &
  holder=$!
  wait_for 5 test -s "$pidf"
  CAMDL_TEST_SLOT_DIR="$d" "$SLOT" 0 touch "$WORK/bypass-zero" &
  CAMDL_TEST_SLOT_DIR="$d" CAMDL_TEST_SLOT_HELD=1 "$SLOT" 1 touch "$WORK/bypass-held" &
  check "N = 0 runs while every slot is held" \
    wait_for 2 test -e "$WORK/bypass-zero"
  check "CAMDL_TEST_SLOT_HELD runs while every slot is held" \
    wait_for 2 test -e "$WORK/bypass-held"
  kill "$holder" "$(cat "$pidf")" 2>/dev/null; wait 2>/dev/null
}

check_concurrency
check_killed_holder
check_daemon_does_not_pin
check_exit_status
check_bypass

echo
echo "test-slot: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
