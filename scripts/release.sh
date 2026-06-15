#!/usr/bin/env bash
#
# Guided release cut for camdl. The policy is VERSIONING.md; the long-form
# runbook is RELEASING.md. This script is the short path you actually run.
#
#   make release-suggest                 # what changed + the suggested bump
#   make release-prep VERSION=0.2.0      # bump manifests + changelog (reviewable)
#   ... review CHANGELOG.md, draft RELEASE_NOTES-0.2.0.md (/release-notes skill) ...
#   make release-publish VERSION=0.2.0   # commit + tag + push + gh release (IRREVERSIBLE)
#
# VERSION is bare SemVer, no leading 'v' (e.g. 0.2.0, or 0.2.0-rc.1).
set -euo pipefail
cd "$(dirname "$0")/.."

die() { echo "release: $*" >&2; exit 1; }

# Every manifest carries its own `version` (inter-crate deps are path-only, so a
# bump needs no dependent edits). Bump them all + the opam file in one pass.
bump_manifests() {
  local v="$1"
  sed -i.bak -E "s/^version = \"[^\"]*\"/version = \"$v\"/" \
    rust/Cargo.toml rust/crates/*/Cargo.toml
  sed -i.bak -E "s/^version: \"[^\"]*\"/version: \"$v\"/" ocaml/camdl.opam
  rm -f rust/Cargo.toml.bak rust/crates/*/Cargo.toml.bak ocaml/camdl.opam.bak
}

check_manifests_at() {
  local v="$1"
  grep -q "^version = \"$v\"$" rust/Cargo.toml \
    || die "rust/Cargo.toml is not at $v — run 'make release-prep VERSION=$v' first"
}

cmd="${1:-}"; ver="${2:-}"

case "$cmd" in
  suggest)
    git fetch -q origin 2>/dev/null || true
    last="$(git describe --tags --abbrev=0 2>/dev/null || true)"
    echo "last release tag: ${last:-<none>}"
    echo "commits since:"
    if [ -n "$last" ]; then git log --oneline "$last"..HEAD; else git log --oneline | head -50; fi
    if command -v git-cliff >/dev/null; then
      echo "git-cliff --bumped-version: $(git-cliff --bumped-version 2>/dev/null || echo '?')"
    fi
    echo
    echo "NOTE: the current tag v0.1.0-alpha has no dotted counter, so the auto-bump"
    echo "      continues it to v0.1.0-alpha.1. For the first real minor cut v0.2.0"
    echo "      explicitly: make release-prep VERSION=0.2.0"
    ;;

  prep)
    [ -n "$ver" ] || die "usage: make release-prep VERSION=<x.y.z>"
    command -v git-cliff >/dev/null || die "git-cliff not found (brew install git-cliff)"
    echo "==> bumping all manifests to $ver"
    bump_manifests "$ver"
    echo "==> regenerating CHANGELOG.md (make changelog)"
    make changelog >/dev/null
    cat <<EOF

prep complete for $ver. Manifests + CHANGELOG.md updated (not committed).
Next:
  1. Review:  git diff CHANGELOG.md rust/Cargo.toml ocaml/camdl.opam
  2. Notes:   draft RELEASE_NOTES-$ver.md  (run the /release-notes $ver skill, then edit)
  3. Gate:    make test   (and confirm CI is green; goldens clean)
  4. If crossing a maturity rung, update the README status badge.
  5. Publish: make release-publish VERSION=$ver
EOF
    ;;

  publish)
    [ -n "$ver" ] || die "usage: make release-publish VERSION=<x.y.z>"
    check_manifests_at "$ver"
    [ -f "RELEASE_NOTES-$ver.md" ] \
      || die "RELEASE_NOTES-$ver.md missing — draft it (/release-notes $ver) before publishing"
    branch="$(git rev-parse --abbrev-ref HEAD)"
    prerelease=""; case "$ver" in *-*) prerelease="--prerelease";; esac
    echo "About to, on branch '$branch':"
    echo "  • commit  chore(release): v$ver   (manifests + CHANGELOG.md)"
    echo "  • tag     v$ver  (annotated)"
    echo "  • push    HEAD + tag to origin"
    echo "  • gh release create v$ver ${prerelease:+(prerelease) }from RELEASE_NOTES-$ver.md"
    echo "This is IRREVERSIBLE (tags are immutable once pushed)."
    read -r -p "proceed? [y/N] " ok
    [ "$ok" = "y" ] || [ "$ok" = "Y" ] || die "aborted"

    git add rust/Cargo.toml rust/crates/*/Cargo.toml ocaml/camdl.opam CHANGELOG.md
    [ -f rust/Cargo.lock ] && git add rust/Cargo.lock || true
    git commit -m "chore(release): v$ver"
    git tag -a "v$ver" -m "camdl v$ver"
    git push origin "$branch" --follow-tags
    gh release create "v$ver" --title "camdl v$ver" --notes-file "RELEASE_NOTES-$ver.md" $prerelease
    echo "published v$ver"
    echo "NOTE: if 'main' is a protected branch, the chore(release) commit must land"
    echo "      via PR — in that case re-run publish from the merge commit, or push the"
    echo "      tag separately once the release commit is on main."
    ;;

  *)
    die "usage: $0 {suggest|prep|publish} [VERSION]"
    ;;
esac
