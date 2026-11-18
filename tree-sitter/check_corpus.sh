#!/bin/sh
# Parse every committed .camdl file with the tree-sitter grammar and fail on
# any error node, except the fixtures that are deliberately invalid (the real
# parser, ocaml/lib/compiler/parser.mly, rejects them too).
#
# Run from anywhere:  sh tree-sitter/check_corpus.sh
# Uses `tree-sitter` on PATH; the generated src/parser.c must be current
# (regenerate with the pinned tree-sitter-cli 0.22.6).

root=$(git rev-parse --show-toplevel) || exit 2
cd "$root/tree-sitter" || exit 2

expected_invalid="
ocaml/golden/diagnostics/semantic/e115_bare_duration_word.camdl
ocaml/test/errors/e001_lex_error.camdl
ocaml/test/errors/e001_parse_error.camdl
"

status=0
for f in $(git ls-files '*.camdl'); do
  case "$expected_invalid" in *"$f"*) continue ;; esac
  if tree-sitter parse "../$f" 2>/dev/null | grep -q 'ERROR\|MISSING'; then
    echo "does not parse: $f"
    status=1
  fi
done
exit $status
