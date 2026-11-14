#!/usr/bin/env bash
# check_cli_docs.sh — CLI command-surface drift gate for camdl docs.
#
# Extracts every logical `camdl …` invocation from fenced ```bash blocks in
# the given docs and asks the real CLI parser — not a stderr string-matcher —
# whether each command's SURFACE exists:
#
#     camdl __check-args -- <argv...>
#         exit 0  surface parses (subcommand + flags + arg shape are real)
#         exit 2  surface drift  (unknown subcommand / unrecognized flag /
#                                 unexpected positional / bad arg count / …)
#
# `__check-args` runs ONLY clap parsing against the same typed command tree the
# binary uses — no file I/O, no compilation, no simulation. So a missing file,
# an unreadable model, or a not-yet-created directory NEVER trips the gate;
# only a command/flag the CLI does not expose does. That makes EXPECTED
# (input-layer) failures free: we don't run the command, we only parse it.
#
# Known scope limit: `compile` / `check` / `inspect` forward their tail
# verbatim to camdlc (clap `trailing_var_arg`), so any flags after them parse
# as OK here — they are camdlc's surface, not camdl's, and out of scope for
# camdl-surface drift. (camdlc-side drift is covered by `make test-docs`.)
#
# Usage:
#   check_cli_docs.sh [DOC.md ...]     gate the given docs (default: workflow.md)
#   check_cli_docs.sh --selftest       non-vacuous proof the gate catches drift

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMDL="$REPO/rust/target/release/camdl"
CAMDLC_DIR="$REPO/ocaml/_build/default/bin"

if [[ ! -x "$CAMDL" ]]; then
  echo "FATAL: $CAMDL not built. Run: make build-rust" >&2
  exit 1
fi

# Prepend the freshly-built worktree binaries so `camdl` (and its camdlc
# delegation) resolve to THIS build, shadowing any stale ~/.local/bin copy —
# avoids a false "version mismatch" from a drifted installed binary.
WORKBIN="$(mktemp -d)" || { echo "FATAL: cannot create a temporary directory" >&2; exit 1; }
ln -sf "$CAMDL" "$WORKBIN/camdl"
[[ -x "$CAMDLC_DIR/camdlc.exe" ]] && ln -sf "$CAMDLC_DIR/camdlc.exe" "$WORKBIN/camdlc"
export PATH="$WORKBIN:$PATH"

# ── Surface check ────────────────────────────────────────────────────────────
# Strip the leading `camdl` token and ask the parser. Returns the child's exit
# code: 0 = surface OK, 2 = DRIFT.
check_surface() {
  local cmd="$1"
  # Drop the leading `camdl` (and any leading whitespace already trimmed).
  local rest="${cmd#camdl}"
  # Neutralize angle-bracket PLACEHOLDERS (`<fit-dir>`, `<model>`) BEFORE eval.
  # Docs use them as "substitute your value here"; they are not shell syntax,
  # but `<`/`>` are redirection operators that would otherwise crash `eval`.
  # Rewrite each to a literal token so the *surface* still parses (the
  # positional is satisfied) — drift is about the command shape, not the value.
  rest="$(printf '%s' "$rest" | sed -E 's/<[A-Za-z0-9_.:-]+>/PLACEHOLDER/g')"
  # `eval` to honor the doc's own quoting/word-splitting (e.g. quoted
  # `--sweep "tau=lin(-60,5,12)"`). The `--` guards args that look like flags
  # to __check-args itself.
  eval "\"$CAMDL\" __check-args -- $rest" >/dev/null 2>&1
  echo $?
}

# ── Self-test (NON-VACUOUS) ──────────────────────────────────────────────────
# Proves two things at once:
#   1. the gate CATCHES drift  — a bogus flag / unknown subcommand → DRIFT
#   2. the gate does NOT over-flag — a valid-but-input-missing command is OK
# If (1) ever silently passed, the whole gate would be vacuous; this fails loud.
if [[ "${1:-}" == "--selftest" ]]; then
  echo "===== SELF-TEST (non-vacuous drift gate) ====="
  fail=0
  expect() { # $1 = expected verdict (DRIFT|OK), $2 = full command
    local got code
    code="$(check_surface "$2")"
    if [[ "$code" == 2 ]]; then got=DRIFT; else got=OK; fi
    if [[ "$got" == "$1" ]]; then
      printf '  ok    %-5s  %s\n' "$got" "$2"
    else
      printf '  FAIL  want=%-5s got=%-5s  %s\n' "$1" "$got" "$2"; fail=1
    fi
  }
  # (1) drift MUST be caught — the negative test. If the gate can't catch these
  # it is worthless, so a regression here fails the build.
  expect DRIFT 'camdl simulate --no-such-flag'
  expect DRIFT 'camdl frobnicate foo'
  expect DRIFT 'camdl fit bogus fit.toml'
  expect DRIFT 'camdl simulate model.camdl --backend not_a_backend'
  expect DRIFT 'camdl simulate model.camdl --particles 10'   # wrong subcommand's flag
  # (2) valid surface with missing/placeholder inputs MUST NOT be flagged.
  expect OK 'camdl simulate /no/such/model.camdl --seed 1'
  expect OK 'camdl fit run /no/such/fit.toml'
  expect OK 'camdl fit summary /no/such/dir'
  expect OK 'camdl compare a b'
  expect OK 'camdl check model.camdl'                        # camdlc passthrough
  expect OK 'camdl fit summary <fit-dir>'                    # angle-bracket placeholder
  expect OK 'camdl simulate <model> --seed 1 --obs <out>'   # multiple placeholders
  if [[ $fail -eq 0 ]]; then echo "self-test PASS"; else echo "self-test FAIL"; fi
  exit $fail
fi

# ── Extraction + gate ────────────────────────────────────────────────────────
DOCS=("$@")
[[ ${#DOCS[@]} -eq 0 ]] && DOCS=("$REPO/docs/workflow.md")

n_drift=0 n_ok=0 n_skip=0
LINES="$(mktemp)" || { echo "FATAL: cannot create a temporary file" >&2; exit 1; }
trap 'rm -f "$LINES"; rm -rf "$WORKBIN"' EXIT
declare -a DRIFT_REPORT=()

gate_one() {
  local file="$1" line="$2" cmd="$3" mode="$4"
  # Skip any command carrying a bare `...` prose-ellipsis token — docs use it
  # as "and so on", not a real argv token; it is a known false-positive source.
  if [[ "$cmd" == *" ... "* || "$cmd" == *" ..." || "$cmd" == "... "* || "$cmd" == *"..." ]]; then
    ((n_skip++)); printf '  %-5s [%s:%s] %s\n' "SKIP" "$file" "$line" "$cmd"; return
  fi
  # Skip a usage synopsis (`camdl cat <hash> [--stream NAME]`, `[OPTIONS]`):
  # its square brackets mark optional parts, so it is not a runnable command.
  if [[ "$cmd" =~ \[(-|[A-Z]) ]]; then
    ((n_skip++)); printf '  %-5s [%s:%s] %s\n' "SKIP" "$file" "$line" "$cmd"; return
  fi
  local code; code="$(check_surface "$cmd")"
  # A ```bash cli-check=refused fence shows a command camdl rejects; it must
  # still be rejected, or the example no longer shows what its prose says.
  if [[ "$mode" == refused ]]; then
    if [[ "$code" == 2 ]]; then
      ((n_ok++)); printf '  %-5s [%s:%s] %s\n' "OK" "$file" "$line" "(refused, as documented) $cmd"
    else
      ((n_drift++))
      DRIFT_REPORT+=("$file:$line: documented as refused, but accepted: $cmd")
      printf '  %-5s [%s:%s] %s\n' "DRIFT" "$file" "$line" "(documented as refused) $cmd"
    fi
    return
  fi
  if [[ "$code" == 2 ]]; then
    ((n_drift++))
    DRIFT_REPORT+=("$file:$line: $cmd")
    printf '  %-5s [%s:%s] %s\n' "DRIFT" "$file" "$line" "$cmd"
  else
    ((n_ok++))
    printf '  %-5s [%s:%s] %s\n' "OK" "$file" "$line" "$cmd"
  fi
}

for doc in "${DOCS[@]}"; do
  echo "===== $doc ====="
  base="$(basename "$doc")"
  # Pull lines inside ```bash fences, tagged with their 1-based line number.
  awk '
    /^```bash[[:space:]]*$/ {inblk=1; mode="ok"; next}
    /^```bash[[:space:]]+cli-check=refused[[:space:]]*$/ {inblk=1; mode="refused"; next}
    /^```/ {inblk=0; next}
    inblk {print NR"\t"mode"\t"$0}
  ' "$doc" > "$LINES" || { echo "FATAL: cannot read $doc" >&2; exit 1; }

  # Reassemble `\`-continuations into one logical command, then gate only
  # lines that start a `camdl ` invocation (skip comments + non-camdl lines).
  acc=""; accln=""
  while IFS=$'\t' read -r ln mode text; do
    case "$text" in \#*) continue;; esac          # comment-only line
    if [[ -n "$acc" ]]; then
      acc="$acc ${text%\\}"                        # continuation: append, drop trailing backslash
    else
      accln="$ln"; acc="${text%\\}"
    fi
    [[ "$text" == *\\ ]] && continue               # still continuing
    trimmed="$(echo "$acc" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    case "$trimmed" in
      camdl\ *|camdl) gate_one "$base" "$accln" "$trimmed" "$mode";;
      *) : ;;                                       # non-camdl shell line — skip
    esac
    acc=""; accln=""
  done < "$LINES"
done

echo
echo "===== SUMMARY ====="
echo "OK:    $n_ok"
echo "SKIP:  $n_skip   (prose ellipsis or usage synopsis)"
echo "DRIFT: $n_drift"
if [[ $n_drift -gt 0 ]]; then
  echo
  echo "CLI doc drift — these documented commands reference a subcommand/flag"
  echo "the binary does not expose:"
  for d in "${DRIFT_REPORT[@]}"; do echo "  $d"; done
  exit 1
fi
exit 0
