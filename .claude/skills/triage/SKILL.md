---
name: triage
description: Triage the camdl GitHub backlog — rebase on main, classify unclassified issues, audit for stale/fixed ones (close candidates for approval), apply effort tiers, and produce a ranked external-weighted priority list, then refresh the dashboard. Use when asked to "triage the issues", "audit the backlog", "what's been fixed / can we close", "what should I work on next", or to prioritize the open issues.
---

# Issue triage for camdl

Sweep the open GitHub backlog into an actionable, prioritized state. **External
reporters** (anyone other than the maintainer `vsbuffalo`) are weighted up —
they're real users waiting on a response. Closing is **always** a numbered list
of candidates for the maintainer's approval; never auto-close.

Reference surfaces:

- `docs/dev/issue-labels.md` — the label scheme (kind × area × effort × status +
  `blocker`/`upstream-audit`). One `kind/`, ≥1 `area/` per issue.
- `docs/dev/issue-triage-tiers.md` — the S/M/L + dup/stale/s-class reduction
  discipline.
- `docs/dev/dashboard/build.py` — the live dashboard (kind×area heat map,
  blocker path, external cohort, momentum trend).

## Procedure

1. **Rebase on current main — fetch FIRST.** The maintainer pushes often, so a
   local `origin/main` ref goes stale fast; always `git fetch origin`
   immediately before rebasing, then rebase the working branch (verify
   ff-ready). Mine the new-commit delta since last triage for closures:
   `git log <prev>..origin/main`, grep `gh#NN`.

2. **Classify the unclassified.** Find open issues missing a `kind/` or `area/`
   label; read each (never label from the title alone); apply exactly one
   `kind/`
   - at least one `area/` (`gh issue edit N --add-label …`).

3. **Audit for stale/fixed — conservatively, against CODE.** For issues
   plausibly addressed by landed work, verify against current main with a
   concrete mechanism (commit SHA / file:line / passing test) — NEVER trust a
   commit message. Subagents are good here (one per issue or small cluster,
   read-only). Default is KEEP; CLOSE only when the full ask is met with
   evidence; a **blocker** CLOSEs only with a fix _and_ a pinning test.
   Distinguish "the forward thing changed" from "the inference thing changed" —
   different claims (e.g. real-flow ODE forward sim ≠ real-compartment
   inference).

4. **Present close candidates — do NOT close.** Print a NUMBERED list, each with
   a 2-3 sentence draft close comment citing the evidence. On the maintainer's
   nod:
   `gh issue comment N --body-file f.md && gh issue close N --reason completed`.
   External-reporter closes especially warrant their eyes.

5. **Apply effort tiers** (`effort/S|M|L`) where a code read supports it — never
   bulk-stamp from titles. `kind/design` is `effort/L` by construction; an issue
   that needs a `docs/dev/proposals/` RFC is `effort/L`.

6. **Produce a ranked priority list, weighting external.** Order: P0 correctness
   blockers (author-agnostic) → external high-ROI → external medium → external
   low/FYI → internal correctness cluster → internal long tail. One line per
   issue: `#NN · short summary · heaviness (S/M/L + why)`. Flag anything riding
   current momentum (a related subsystem actively landing on main).

7. **Refresh + redeploy the dashboard.**
   `python3 docs/dev/dashboard/build.py --snapshot`, then rsync `index.html` to
   `vincebuffalo:apps/vincebuffalo/camdl-dashboard/` for the external view. Note
   day-over-day movement.

## Notes

- `$ARGUMENTS` may scope the pass: `external` = only the external reporters'
  issues; a `#NN` = re-audit just that issue.
- Full backlog pull:
  `gh issue list --state open --limit 500 --json number,title,labels,author`.
- Merge gate for any code fix this triage produces: full `make test` (includes
  the cross-language integration phase, which catches CAS-path regressions
  scoped `cargo test` misses), then ff-merge.
- This is read-mostly: it labels and reports. The only writes are label edits
  and maintainer-approved closes — it changes no code on its own.
