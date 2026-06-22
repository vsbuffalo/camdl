#!/usr/bin/env bash
# Fast, offline unit tests for install.sh.
#
# Covers the parts that hide bugs and the property this script exists to hold:
#   - version_ge   : the cmake >= 3.13 gate (Ubuntu 18.04 ships 3.10)
#   - cmake_plat   : OS/arch -> Kitware portable-build platform string
#   - the no-sudo contract: with the toolchain present every ensure_* step
#     early-returns and NEVER invokes sudo (a shim aborts if it tries); with a
#     base tool missing, ensure_base_tools fails fast with install instructions.
#
# Network and a real build are NOT exercised here — that is the container test
# (tests/install/Dockerfile.ubuntu1804), which reproduces gh#205's exact box.
#
# Run: bash tests/install_sh_test.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=install.sh
source "$ROOT/install.sh"      # guarded main() does not run when sourced
set +e +u; set +o pipefail     # install.sh enables -euo; tests assert on failures

PASS=0; FAIL=0
ok()  { PASS=$((PASS+1)); printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad() { FAIL=$((FAIL+1)); printf '  \033[31mFAIL\033[0m %s\n' "$*"; }

# pass_if DESC CMD... — record ok if CMD succeeds, bad if it fails.
# fail_if DESC CMD... — the negative assertion (ok iff CMD fails).
pass_if() { local d="$1"; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }
fail_if() { local d="$1"; shift; if "$@"; then bad "$d"; else ok "$d"; fi; }
eq()      { [ "$1" = "$2" ]; }

# ---------------------------------------------------------------- version_ge
check_version_ge() {
  echo "version_ge:"
  # The gate is always version_ge <cmake-reported> "$CMAKE_MIN": cmake always
  # reports three components (3.10.2, 3.30.5) and CMAKE_MIN is "3.13", so these
  # are the only shapes that occur.
  pass_if "3.30.5 >= 3.13"               version_ge 3.30.5 3.13
  pass_if "3.13.4 >= 3.13 (above floor)" version_ge 3.13.4 3.13
  pass_if "3.13.0 >= 3.13 (at floor)"    version_ge 3.13.0 3.13
  fail_if "3.10.2 < 3.13 (Ubuntu 18.04 cmake rejected)" version_ge 3.10.2 3.13
  fail_if "3.9 < 3.13"                   version_ge 3.9 3.13
}

# ---------------------------------------------------------------- cmake_plat
check_cmake_plat() {
  echo "cmake_plat:"
  local got
  OS=linux ARCH=x86_64  ; got=$(cmake_plat); pass_if "linux/x86_64 -> linux-x86_64"     eq "$got" linux-x86_64
  OS=linux ARCH=aarch64 ; got=$(cmake_plat); pass_if "linux/aarch64 -> linux-aarch64"   eq "$got" linux-aarch64
  OS=macos ARCH=arm64   ; got=$(cmake_plat); pass_if "macos/arm64 -> macos-universal"   eq "$got" macos-universal
  OS=macos ARCH=x86_64  ; got=$(cmake_plat); pass_if "macos/x86_64 -> macos-universal"  eq "$got" macos-universal
  OS=linux ARCH=riscv64 ; fail_if "unknown arch rejected"                               cmake_plat
}

# ----------------------------------------------------- no-sudo, all present
# Shim dir: believable tool versions + a `sudo` that records any invocation.
check_no_sudo_when_present() {
  echo "no-sudo contract (toolchain present):"
  local d; d="$(mktemp -d)"
  local sudolog="$d/sudo.log"; : > "$sudolog"
  mkdir -p "$d/bin"
  printf '#!/bin/sh\necho "SUDO: $*" >> "%s"\nexit 97\n' "$sudolog" > "$d/bin/sudo"
  printf '#!/bin/sh\necho "cmake version 3.30.5"\n' > "$d/bin/cmake"
  printf '#!/bin/sh\necho 2.5.1\n'                  > "$d/bin/opam"
  local t
  for t in make git curl tar cargo rustc; do printf '#!/bin/sh\nexit 0\n' > "$d/bin/$t"; done
  chmod +x "$d"/bin/*

  OS=linux ARCH=x86_64
  ( PATH="$d/bin:$PATH"
    ensure_base_tools && ensure_cmake && ensure_opam && ensure_rust ) >/dev/null 2>&1
  local rc=$?

  if [ -s "$sudolog" ]; then bad "sudo was invoked: $(cat "$sudolog")"
  elif [ "$rc" -ne 0 ]; then bad "ensure_* failed (rc=$rc) with all tools present"
  else                       ok "all ensure_* early-returned, sudo never called"
  fi
  rm -rf "$d"
}

# ------------------------------------------------- fail-fast, base tool gone
check_fail_fast_when_missing() {
  echo "fail-fast (base tool missing):"
  OS=linux ARCH=x86_64
  local out rc
  # Simulate a missing `make` by shadowing the `have` probe ensure_base_tools
  # uses; run in a subshell so its err-exit doesn't take down the harness.
  out=$(
    have() { [ "$1" = make ] && return 1; command -v "$1" >/dev/null 2>&1; }
    ensure_base_tools 2>&1
  ); rc=$?

  if [ "$rc" -eq 0 ]; then bad "ensure_base_tools should fail when make is missing"
  elif ! grep -q 'make' <<<"$out"; then bad "message should name the missing tool: $out"
  elif ! grep -q 'apt-get install' <<<"$out"; then bad "message should give an install hint: $out"
  else ok "exits nonzero, names 'make', prints the one-time install hint"
  fi
}

echo "== install.sh unit tests =="
check_version_ge
check_cmake_plat
check_no_sudo_when_present
check_fail_fast_when_missing
echo
echo "== $PASS passed, $FAIL failed =="
[ "$FAIL" -eq 0 ]
