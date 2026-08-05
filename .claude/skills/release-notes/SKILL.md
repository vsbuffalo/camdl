---
name: release-notes
description: Draft user-facing release notes for a camdl version from the Conventional-Commit history + dev notes. Use when cutting a release, tagging a version, or asked to "write release notes" / "what changed since <tag>". Pairs a deterministic changelog (git-cliff) with an agent-written narrative.
---

# Release notes for camdl

Produce **user-facing** release notes for a version — not a commit dump. The
commit history is the input; the output reads like something a user tracking the
project wants to read. Follow `VERSIONING.md` for what the version covers and
the SemVer policy.

## Inputs

- A target version (e.g. `0.4.0`) and/or a commit range. If not given, default
  the range to `git describe --tags --abbrev=0`..`HEAD` (last tag → HEAD); if
  there is no tag yet, use the full history on the branch.
- `$ARGUMENTS` may carry the version and/or an explicit range (`v0.3.0..HEAD`).

## Procedure

1. **Resolve the range.** `git log --oneline <range>` to see the commits.
   Confirm the range with the user if ambiguous.

2. **Deterministic spine.** If `git-cliff` is installed
   (`command -v git-cliff`), run `git-cliff <range>` to get the grouped
   Conventional-Commit changelog — this is the completeness guarantee (nothing
   dropped). If it is not installed, derive the same grouping yourself from
   `git log <range>` by parsing the `type(scope): subject` prefixes. Do **not**
   skip a commit just because it is `chore`/`ci`/`refactor`; decide per item
   whether it is user-visible.

3. **Compute the SemVer bump** from the commit types over the range (per
   `VERSIONING.md`): any `!`/`BREAKING CHANGE:` → MINOR while 0.x (MAJOR at
   ≥1.0); any `feat` → MINOR; otherwise PATCH. State the recommended bump and
   why.

4. **Read for context, not just titles.** Scan `docs/dev/notes/`,
   `docs/dev/incidents/`, and `docs/dev/proposals/` whose dates fall in the
   range — they carry the _why it matters_ that commit subjects omit (e.g. a
   dimcheck fix's user impact, a found-bug's blast radius). Also check whether
   `ir/VERSION` changed in the range — an IR-schema bump is a compatibility
   event users must know about.

5. **Translate commit-speak → user-speak.** "fix(dimcheck): projected carries
   its projection's dimension" → "Fixed: prevalence-as-proportion observation
   models now type-check (previously rejected with a spurious dimension error)."
   Lead with the effect on the user, not the implementation.

## Output shape

Write to `RELEASE_NOTES-<version>.md` (or stdout if asked), in this order:

```
# camdl <version> — <date>

## Highlights
- <the 3–5 things a user actually cares about, one line each>

## Breaking changes
- <surface that changed + the exact migration step. Omit the section if none.>

## Language (DSL)
## CLI
## Inference & engine
## Formats & compatibility      <- IR schema (ir/VERSION) bumps, fit.toml, output changes
## Fixed
## Internal / docs / CI         <- terse; group the rest, don't enumerate every commit
```

Rules:

- Group by **user-relevant area**, not by commit type.
- Every breaking change gets a concrete migration line ("rename
  `--focal X --grid …` to `--sweep "X=lin(…)"`").
- Cite the surface, not the SHA, in the body; a trailing "Full changelog:
  `<range>`" line is enough provenance.
- Keep "Internal/docs/CI" short — one bullet per cluster, not per commit.
- Hedge honestly; do not claim stability the version doesn't have (it is 0.x).

## After drafting

Show the draft and the recommended version bump. The maintainer is the editor —
do not tag or publish. If asked to finalize, the tag + `CHANGELOG.md` update
follow the flow in `VERSIONING.md`.
