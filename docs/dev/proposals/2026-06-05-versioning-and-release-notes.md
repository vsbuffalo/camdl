---
date: 2026-06-05
status: proposal
establishes: ../../VERSIONING.md, ../../.claude/skills/release-notes/SKILL.md
---

# Versioning and release notes for the run-up to beta

## Problem

camdl is `0.x`, unreleased, moving fast across several surfaces (DSL, CLI,
inference engine, IR schema). There is no version policy, no changelog, and no
way for a user to track "what changed and does it affect me." As we approach
beta this becomes a real cost: users need a stable surface they can build on, a
signal when it moves, and migration guidance when it breaks. We also generate a
large volume of work (this is being written after a session that produced a
dozen commits), and hand-writing release notes for that volume does not scale.

We already have the raw material: commits are written as Conventional Commits
(`feat(dimcheck):`, `fix(cli):`, …), and `docs/dev/notes/` carries the _why_
behind the changes. The task is to turn that into a versioning discipline and an
(agent-assisted) release-notes pipeline.

## Decisions (the load-bearing ones)

1. **Define the versioned surface before adopting SemVer.** SemVer is
   meaningless without saying what it covers. The policy (`VERSIONING.md`): the
   release version governs DSL grammar + CLI + output/file formats; the IR
   schema stays separately versioned (`ir/VERSION`) as an internal contract
   whose bumps are _reported_ in release notes; `fit.toml` counts as CLI surface
   for breaking purposes.

2. **`0.x` semantics through beta.** MINOR may break, PATCH is fixes-only; `1.0`
   is reserved for a stable DSL+CLI surface (not "feature complete"). Beta = a
   frozen-ish `0.x` with a deprecation policy (warn → remove no sooner than next
   MINOR).

3. **Two-layer release notes: deterministic spine + agent narrative.** The spine
   guarantees completeness; the agent makes it readable. Neither alone is enough
   — a raw changelog is commit-shaped and unreadable to users; an agent without
   the spine drops things.

## Tooling

### Spine — git-cliff (recommended)

[git-cliff](https://git-cliff.org/) is a single static binary (Rust) that reads
`git log`, parses Conventional Commits, and renders a `CHANGELOG.md` grouped by
type/scope via a TOML template (`cliff.toml`). It is language-agnostic — ideal
for this OCaml+Rust monorepo, where a Rust- or Node-specific release tool would
only see half the repo. It also computes the next SemVer tag from the commit
types. Zero runtime deps, runs in CI or locally.

_Alternative considered:_ `release-please` (GitHub-native; opens a standing
"Release PR" that maintains the changelog and version bump as commits land).
More automation, but GitHub-coupled and opinionated about the flow.
Recommendation: start with git-cliff (a file + a command, fully under our
control); adopt release-please later if we want the release-PR automation. _Not_
`cargo-release` / `semantic-release` — both assume a single-language (Rust/Node)
project.

### Narrative — the `/release-notes` skill

`git-cliff` output is commit-shaped
(`fix(dimcheck): projected carries its
projection's dimension`). Users need
effect-shaped notes. The skill (`.claude/skills/release-notes/SKILL.md`) takes
the commit range + the git-cliff spine + the dated `docs/dev/notes/` for
context, and drafts user-facing notes: a **Highlights** section, changes grouped
by _user_ area (Language / CLI / Inference / Formats), a **Breaking changes**
section with a migration line each, and a terse internal/docs tail. It
translates implementation into impact ("prevalence-as- proportion observation
models now type-check") and surfaces `ir/VERSION` bumps as compatibility events.
The maintainer is the editor; the skill never tags or publishes.

This is where coding agents earn their place: the deterministic tool can group
and sort, but only an agent that has read the dev notes can say _why a change
matters_ and write the migration step.

## Where the skill lives

Project skills live at `.claude/skills/<name>/SKILL.md`. `.claude/` was fully
gitignored (local settings + worktrees), so this proposal also un-ignores
`.claude/skills/` and `.claude/commands/` — shared, version-controlled agent
tooling belongs in the repo and on GitHub, while local settings and worktrees
stay ignored. The skill is then available to anyone who clones the repo and is
itself versioned alongside the code it documents.

## Rollout

1. Land `VERSIONING.md` + this RFC + the `/release-notes` skill (this change).
2. Add `cliff.toml` and a `make changelog` / `make release-notes` target (a
   small follow-up; the skill works without it by parsing `git log` directly).
3. Cut the first tagged `0.x` with notes drafted by the skill — exercises the
   whole pipeline end to end.
4. At beta, turn on the deprecation policy and consider release-please for
   release-PR automation.

## Out of scope

Automated publishing (GitHub Releases API), binary release artifacts, and a
homebrew/opam distribution story — all post-beta concerns; this proposal is the
versioning policy + the notes pipeline only.
