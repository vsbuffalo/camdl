---
name: burn-down
description: Work the camdl issue backlog down safely and fast — cluster independent issues into one branch, apply the pre-flight and hand-off checks that catch sibling sites and unexercised claims, and escalate only the calls that are genuinely the maintainer's. Use when asked to "knock down the issues", "work the blockers", "fix these bugs", "burn down the backlog", or when handed a list of issue numbers to fix.
---

# Issue burn-down for camdl

Throughput is not the constraint — a red→green fix is quick. **Regressions and
half-fixes are the constraint**, and the specific ways they get in are known and
mechanical. This is that checklist, plus the batching rules that make a run of
fixes cheaper than the same fixes one at a time.

Triage (which issues, in what order) is the `triage` skill. This is the doing.

## The four checks that actually catch things

Every one of these came from a real miss. Run them; they are cheap.

1. **Sibling sweep — before claiming done.** The fix is for a _site_, the issue
   is usually about a _class_. Grep for the rest of it:
   - other callers of the function you changed (`rg 'fn_name\('`);
   - the **readers** of anything you moved or re-anchored — changing a writer
     without its readers is how `gh#507` broke `show`/`cat`/`list` (gh#526);
   - the rest of the flag/variant/branch family — `gh#514` fixed the `--init`
     flags and left ~13 siblings (gh#540);
   - the same guard on the other path — `gh#496` guarded `batch.rs` and left
     `fit/mod.rs` (gh#536).

   If siblings exist and you are not fixing them, **say so in the PR body and
   file the follow-up in the same breath.** Silence reads as "handled".

2. **Every claim is executed.** A PR body or doc comment that says a surface
   works must have a pasted command showing it. `gh#508` shipped "`@symbol` …
   works as it does elsewhere" for a code path that never read it (gh#527). If
   you did not run it, write "not exercised" or do not write it.

3. **Concurrency check — before starting.** Others work this repo.
   `gh pr list --state open --search 'gh#NN'` and `git branch -r --list '*NN*'`.
   Thirty seconds; a duplicated fix costs an hour and gets closed unmerged.

4. **Class-or-instance call.** If the sweep in (1) turns up a _family_, stop and
   ask. "Fix these fourteen flags, or make the category impossible?" is a design
   decision and it is the maintainer's, not yours. Fixing instance 1 of 14
   quietly is the worst of the three options.

## Batching

One PR per issue is the default and is right for anything with semantics. Batch
only when it is genuinely free:

- **Batch** doc/comment corrections, dead-code deletions, and stale-claim sweeps
  that touch no behaviour — one branch, **one commit per issue** so any can be
  dropped in review, one CI run.
- **Never batch** two issues whose fixes touch the same function's semantics, or
  anything on the high-risk surfaces — CLAUDE.md's six (`pgas.rs`,
  `pgas_grad.rs`, `obs_loglik.rs`, `obs_model.rs`, `if2.rs`,
  `particle_filter.rs`) plus `chain_binomial.rs`, which is not on that list but
  earned a place: gh#517 was a silent noise-model drop three lines from a draw,
  in a function whose rate path was already guarded. One at a time, each with
  its own red→green.
- A batch's commit subjects still each name their `gh#NN`, so `git log --grep`
  and the changelog stay honest.

Cluster by **file and subsystem**, not by effort tier — the saving is the build
and the loaded context, not the typing.

## Parallel agents

Worth it when ≥3 issues are independent AND touch disjoint files. Give each a
worktree (`isolation: worktree`), a single issue, and the disjoint file set.
Then **review every diff yourself before it becomes a PR** — an agent's fix
inherits none of the four checks above unless you hand them over.

Not worth it for: anything on the high-risk list, anything where the fix shape
is still an open question, or fewer than three issues.

## The gate

The working loop is the narrowest suite that covers the change — one
`cargo test -p <crate> --lib`, one `dune exec` test binary, one `--test` target.
`make test` is the authoritative gate and is slow (tens of minutes on a warm
tree, longer from cold); CLAUDE.md accepts **either** a full local `make test`
**or** CI as authoritative, so push early and let CI run while you start the
next issue. Do not skip both.

Two things CI will not tell you, so check them locally before pushing:

- `make update-golden` moves **zero** goldens (any compiler/IR change);
- `cargo clippy --all-targets` introduces no new warning **in the files you
  touched** — the repo has pre-existing lint drift, so diff against the file
  list, not the total count.

## Byte-identity work

When a change is supposed to be value-preserving (a refactor, a routing fix, a
consolidation), prove it rather than asserting it:

- a **differential test** against the old implementation kept verbatim as an
  oracle, over a grid wide enough to include the divergent region — and assert
  the grid _reached_ that region, or the test proves nothing;
- an **end-to-end A/B**: same seed before and after, diff the artifacts. Expect
  timestamps and mtimes to differ; everything else must not.

## Escalate, and otherwise proceed

Interrupt for: a numerics or semantics choice with no objectively right answer;
anything that rejects previously-valid models, moves user output, or touches run
identity / `ir/VERSION` / goldens; a defect _class_ (check 4); and evidence that
contradicts the issue's own premise, since that changes severity.

Do not interrupt for: confirmed-live verifications, mechanical red→green fixes,
label and tier decisions, or progress. Put the decision in one sentence at the
top with a recommendation; the log goes below, or nowhere.
