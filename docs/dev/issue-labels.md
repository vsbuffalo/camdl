# Issue label taxonomy

A multi-axis label scheme for the GitHub backlog. Every issue is classified
along several **orthogonal** axes, each carried by its own namespaced label
(`kind/`, `area/`, `effort/`, `status/`) plus two flat flags (`blocker`,
`upstream-audit`). The axes are independent: an issue picks one value from each
axis that applies, and the combination — not any single label — is the
classification.

This is the label-level companion to
[`issue-triage-tiers.md`](issue-triage-tiers.md), which defines the _working
vocabulary_ used during a triage pass (the S/M/L tiers and the
dup/stale/s-class/tricky buckets). The vocabulary lives in the labels:
`effort/*` is the tier, `status/*` is the bucket. Read that doc for the
parallel-knockdown discipline; read this one for what to put on an issue.

## The axes

| Axis           | Prefix           | Cardinality | Answers                              |
| -------------- | ---------------- | ----------- | ------------------------------------ |
| **Kind**       | `kind/`          | exactly one | What _is_ this issue?                |
| **Area**       | `area/`          | one or more | Which surface(s) does it touch?      |
| **Effort**     | `effort/`        | zero or one | How big is the fix?                  |
| **Status**     | `status/`        | zero or one | What's its current triage state?     |
| **Priority**   | `blocker`        | flag        | Does it block correct use / release? |
| **Provenance** | `upstream-audit` | flag        | Where did it come from?              |

`kind/` and `area/` are the **stable** axes — set them once, on triage, and they
rarely change. `effort/`, `status/`, and `blocker` are **live** — they reflect
the current state of a triage/knockdown pass and move as work proceeds.

### Kind — what it is (exactly one)

| Label           | Meaning                                                                                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kind/bug`      | Defective behavior: wrong output, crash, or a **silent-wrong** result.                                                                                    |
| `kind/feature`  | A new capability the system doesn't have yet (incl. perf/ergonomics improvements).                                                                        |
| `kind/refactor` | Internal-quality work with no intended user-facing behavior change — hardening, dead-code removal, type-tightening, **test-coverage debt**.               |
| `kind/design`   | The deliverable is a _decision_, not code yet: an RFC / proposal-stage question. Promote to `kind/feature` once the design lands and it's ready to build. |
| `kind/docs`     | Documentation only.                                                                                                                                       |
| `kind/question` | A question or discussion with no committed change.                                                                                                        |

A bug whose defining symptom is a _wrong number the system reports as correct_
is `kind/bug` **and** a `blocker` candidate — silent-wrong is the
highest-severity class here (this software informs public-health decisions).

### Area — which surface (one or more)

| Label            | Surface                                                                                                                                                                                         |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `area/compiler`  | OCaml DSL → IR: lexer, parser, expander, dimcheck, autodiff, lints, language surface.                                                                                                           |
| `area/engine`    | Rust runtime: simulation backends (Gillespie/ODE/chain-binomial/tau-leap), propensity eval, intervention/event execution, lineage.                                                              |
| `area/inference` | Rust inference: PGAS, IF2, NUTS, PMMH, particle filter, gradients, priors, fit diagnostics. **Owner-coordinated** — primary edits land in another active owner's files; coordinate, don't fork. |
| `area/cli`       | CLI args/flags, output/progress UX, CAS run identity, run orchestration, IR compile caching.                                                                                                    |
| `area/obs-model` | Observation model: projection, likelihood, data streams, surveillance geometry.                                                                                                                 |
| `area/ir-schema` | The IR schema itself — the OCaml↔Rust cross-language contract.                                                                                                                                  |
| `area/testing`   | Test infrastructure and coverage: meta-tests, oracles, regression-test debt.                                                                                                                    |

Use as many `area/*` as genuinely apply — a cross-cutting bug like "param used
in a `time_function` is silently frozen" is legitimately `area/compiler` +
`area/engine` + `area/inference`. Don't pad: an area label means the fix
_touches that surface_, not that the issue is vaguely related to it.

### Effort — how big (zero or one)

The S/M/L tiers from `issue-triage-tiers.md`:

| Label      | Tier                                                                                                             |
| ---------- | ---------------------------------------------------------------------------------------------------------------- |
| `effort/S` | Small: localized, ~single-area, < ~1h, no design decision; fix shape obvious from the issue + a quick code read. |
| `effort/M` | Medium: ~half-day; a few files or a non-obvious but bounded change; no proposal.                                 |
| `effort/L` | Large: multi-day, cross-cutting, or needs a `docs/dev/proposals/` RFC first.                                     |

Effort is **not bulk-stamped from titles** — the S definition requires a code
read, and an honest tier is the product of an actual triage pass. Leave it unset
until someone has looked. A `kind/design` issue is `effort/L` by construction.

### Status — current triage state (zero or one)

The triage buckets, for issues actively moving through a knockdown pass:

| Label            | State                                                                                                                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `status/s-class` | `reliably_landable`: S-effort **and** collision-free **and** no-proposal **and** a clean red→green test exists. The strict bar for automated worktree batches — when unsure, it's not s-class. |
| `status/tricky`  | Genuine engineering + scientific-judgment call; do deliberately.                                                                                                                               |
| `status/blocked` | Can't proceed — waiting on a dependency (another issue, a decision, an owner). Distinct from `blocker` (see below).                                                                            |

`dup` and `stale` are terminal verdicts, not states to carry — verify by
_mechanism_ (commit ancestor / file:line / test), then **close** with that
evidence and GitHub's built-in `duplicate` label where apt. They don't get a
`status/*`.

### blocker — priority flag

A single, scarce, high-signal flag: this issue **blocks correct use or a
release**. Reserve it for the critical path — chiefly silent-wrong results on
the inference/simulation path, and the fixes that directly unblock them.
Scarcity is the point; if everything is a blocker, nothing is.

`blocker` (priority — "this blocks _us_") is **not** `status/blocked` (state —
"this is blocked _by_ something"). They can co-occur or stand alone.

### upstream-audit — provenance flag

Findings originating from the 2026-05-26 upstream OCaml-compiler / Rust-engine
review. Kept as a provenance tag so an audit cohort stays queryable.

## Classification rules

1. Every open issue gets **exactly one `kind/`** and **at least one `area/`**.
2. `effort/`, `status/`, and `blocker` are optional and set during a triage
   pass.
3. Prefer the most specific true labels; never pad an axis to look thorough.
4. `dup` / `stale` → verify the mechanism, close with evidence; don't leave them
   labeled-but-open.

## Querying

The axes compose into useful slices:

```bash
# Every collision-free small fix ready for a worktree batch
gh issue list --label "status/s-class" --label "effort/S"

# What needs a proposal before any code
gh issue list --label "kind/design"

# The critical path
gh issue list --label "blocker"

# Open inference bugs (the high-risk, owner-coordinated surface)
gh issue list --label "kind/bug" --label "area/inference"

# Audit cohort still open
gh issue list --label "upstream-audit"
```

## Bootstrap / maintenance

The label set is the source of truth below. To (re)create it on a fresh clone or
mirror, run these once (`gh label create --force` is idempotent — it updates
color/description if the label already exists). Renames preserve existing issue
assignments, so migrating the old flat labels (`bug`, `enhancement`, `compiler`,
`engine`) into the namespaced scheme via `gh label edit --name` reclassifies
every already-labeled issue for free.

```bash
# --- Kind ---
gh label create "kind/bug"      -c d73a4a -d "Defective behavior (incl. silent-wrong)" --force
gh label create "kind/feature"  -c a2eeef -d "New capability (incl. perf/ergonomics)"  --force
gh label create "kind/refactor" -c fbca04 -d "Internal-quality work, no behavior change (incl. test debt)" --force
gh label create "kind/design"   -c 6f42c1 -d "RFC / proposal-stage decision"           --force
gh label create "kind/docs"     -c 0075ca -d "Documentation only"                      --force
gh label create "kind/question" -c d876e3 -d "Question or discussion"                  --force

# --- Area (one shared color: the namespace reads as one axis) ---
gh label create "area/compiler"  -c 1d76db -d "OCaml DSL → IR compiler surface"        --force
gh label create "area/engine"    -c 1d76db -d "Rust runtime engine — backends, eval, interventions, lineage" --force
gh label create "area/inference" -c 1d76db -d "Rust inference (PGAS/IF2/NUTS/PMMH/PF/priors) — owner-coordinated" --force
gh label create "area/cli"       -c 1d76db -d "CLI, output/progress UX, CAS run identity, orchestration" --force
gh label create "area/obs-model" -c 1d76db -d "Observation model — projection, likelihood, data streams" --force
gh label create "area/ir-schema" -c 1d76db -d "IR schema — the OCaml↔Rust contract"   --force
gh label create "area/testing"   -c 1d76db -d "Test infrastructure and coverage"      --force

# --- Effort (S→L heat gradient) ---
gh label create "effort/S" -c c2e0c6 -d "Small: < ~1h, localized, no design decision" --force
gh label create "effort/M" -c fef2c0 -d "Medium: ~half-day, bounded, no proposal"     --force
gh label create "effort/L" -c f9d0c4 -d "Large: multi-day, cross-cutting, or needs an RFC" --force

# --- Status ---
gh label create "status/s-class" -c d4c5f9 -d "reliably_landable: S + collision-free + clean red→green" --force
gh label create "status/tricky"  -c d4c5f9 -d "Genuine engineering/scientific-judgment call"          --force
gh label create "status/blocked" -c d4c5f9 -d "Blocked by a dependency (≠ the blocker flag)"          --force

# --- Flags ---
gh label create "blocker"        -c b60205 -d "Blocks correct use / a release (scarce, high-signal)"  --force
gh label create "upstream-audit" -c 5319e7 -d "Findings from the 2026-05-26 upstream review"          --force
```

GitHub's built-in `duplicate`, `invalid`, `wontfix`, `good first issue`, and
`help wanted` are kept as-is; they're orthogonal to these axes.
