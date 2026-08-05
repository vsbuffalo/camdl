---
name: releasing
description: The camdl release runbook — version policy, the suggest/prep/notes/green/publish sequence, and what an agent may and may not do. Use when cutting a release, bumping a version, or asked what a version number promises.
---

# Releasing camdl

Policy lives in [`VERSIONING.md`](../../../VERSIONING.md) (what a version
promises a user); the runbook is [`RELEASING.md`](../../../RELEASING.md). This
skill is the load-bearing summary.

## Publishing is maintainer-gated

An agent may run suggest/prep and draft notes. **Tagging and publishing are the
maintainer's call, never an agent's.** This is also stated in `CLAUDE.md`; it is
repeated here only because this skill is the place someone lands when they
actually try to cut a release.

## Version policy

- Tags are `vMAJOR.MINOR.PATCH`, **always annotated** (`git tag -a`).
- Pre-releases are dot-numbered: `v0.2.0-rc.1`, never a bare `-alpha`.
- Pre-1.0: MINOR (`0.x.0`) may break the DSL/CLI/output-format surface; PATCH
  (`0.x.y`) is fixes-only.
- The version covers DSL + CLI + output/file formats. The IR schema
  (`ir/VERSION`) and `fit.toml` are versioned **separately** and _reported_ in
  the notes, not folded into the release number.

## Never hand-tag or hand-`gh release`

Cut through the tooling. The `make` targets are thin wrappers over
`scripts/release.sh`:

1. `make release-suggest` — commits since the last tag + the suggested bump.
2. `make release-prep VERSION=x.y.z` — bumps every manifest, regenerates
   `CHANGELOG.md`.
3. Draft `RELEASE_NOTES-x.y.z.md` with the `release-notes` skill; edit it.
4. `make test` green + CI green.
5. `make release-publish VERSION=x.y.z` — the only irreversible step; prompts
   before it commits, tags, pushes, and publishes.

## First release

The first published release is `v0.2.0` (the `v0.1.0-alpha` tag was never
published). Cut it **explicitly** — the auto-bump continues the bare alpha tag
to `v0.1.0-alpha.1`, which is wrong.

## Related

- Drafting the narrative notes: the `release-notes` skill.
- Commit conventions that feed the generated changelog:
  [`docs/dev/commit-style.md`](../../../docs/dev/commit-style.md).
