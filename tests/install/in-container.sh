#!/usr/bin/env bash
# Runs INSIDE the Ubuntu 18.04 image (tests/install/Dockerfile.ubuntu1804) as
# the non-root `builder` user, with no sudo on PATH. Kept out of the image
# BUILD and run via `docker run` instead: `opam init` sandboxes with bubblewrap,
# which needs unprivileged user namespaces, and `docker build` has no way to
# grant them (buildkit gates that behind an `insecure` entitlement). `docker
# run` does — see tests/install/run.sh.
#
# A green run == install.sh produced a working `camdl` from a clean box with
# zero privilege, WITH opam's sandbox active (never NO_SANDBOX=1: that would
# make the job pass by deleting the thing it verifies).
set -euo pipefail

./install.sh

# Mirrors the rc lines install.sh prints, then proves the binary actually runs.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
eval "$(opam env)"
camdl --version
