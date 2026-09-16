#!/usr/bin/env bash
# What the install-e2e container actually runs. Kept as a file rather than an
# inline CMD so it is greppable, shellcheck-able, and runnable by hand against
# the same image.
#
# Two assertions, in order:
#   1. install.sh completes as an unprivileged user with no sudo present.
#   2. the camdl it produced runs, reached exactly the way install.sh's own
#      closing notes tell the user to reach it.
#
# (2) is not a formality: the install can finish while leaving a `camdl` that
# no shell can find, and a test that stops at (1) would call that a pass.
set -euo pipefail

./install.sh

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    . "$HOME/.cargo/env"
fi
eval "$(opam env)"

camdl --version
