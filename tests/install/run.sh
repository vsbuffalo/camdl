#!/usr/bin/env bash
# Build the gh#205 reproduction image and run the no-sudo install inside it.
# Used by .github/workflows/install-e2e.yml and runnable by hand:
#
#   ./tests/install/run.sh
#
# SLOW: an amd64 image and a full from-source build of both toolchains. Run it
# on an amd64 host or CI runner, not in the inner loop (under emulation on
# Apple Silicon it takes hours).
set -euo pipefail

IMAGE="${IMAGE:-camdl-install-test}"
cd "$(dirname "$0")/../.."

docker build -f tests/install/Dockerfile.ubuntu1804 -t "$IMAGE" .

# The install runs HERE, not in the image build. `opam init` sandboxes package
# builds with bubblewrap, which creates an unprivileged user namespace; Docker's
# default seccomp and AppArmor profiles both deny that, and `docker build` has
# no flag to permit it (buildkit gates it behind an `insecure` entitlement that
# needs a custom builder). `docker run` takes the flags directly.
#
# This relaxes the CONTAINER's confinement, not the install's. The image still
# has no sudo and still runs as the non-root `builder` user, so the claim the
# job makes — install.sh needs zero privilege — is unchanged and still enforced.
# What it buys is that opam's sandbox is genuinely ACTIVE during the run, which
# is the configuration real users get; NO_SANDBOX=1 would go green by deleting
# exactly what this exercises.
docker run --rm \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  "$IMAGE"
