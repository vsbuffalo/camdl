# Issue triage tiers + the parallel-knockdown discipline

Working vocabulary used when triaging the GitHub backlog for batch reduction.
Not formal process — a shared shorthand so "S-class" / "tricky-leave" mean the
same thing across passes. The vocabulary is carried on GitHub by the label
taxonomy in [`issue-labels.md`](issue-labels.md): the effort tiers below are
`effort/*`, the triage buckets are `status/*`.

## Effort tiers (rough wall-clock for one engineer/agent)

- **S** — small: a localized, ~single-area fix, < ~1h, no design decision. The
  fix shape is obvious from the issue + a quick code read.
- **M** — medium: ~half-day; touches a few files or needs a non-obvious but
  bounded change; no proposal, but not a one-liner.
- **L** — large: multi-day, cross-cutting, or **needs a `docs/dev/proposals/`
  RFC first** (schema/IR change, new inference method, an architectural seam).

## Triage buckets (what to _do_ with an issue)

- **dup** — same defect/ask as another open issue → close it, comment pointing
  to the canonical (richer / lower-number) one. Verify by _mechanism_, not
  title.
- **stale** — already fixed on current `main`, or obsolete (feature never
  merged). Verify the load-bearing artifact (commit ancestor / file:line / test)
  exists, then close with that evidence.
- **s-class / `reliably_landable`** — the parallel-batch tier: **S-effort AND
  collision-free AND no-proposal AND a clean red→green test exists**. Only these
  go into automated worktree batches. Strict bar: when unsure, it's not s-class.
- **m-class** — real but bounded; do deliberately (some are S in disguise once
  scoped tighter).
- **l-or-proposal** — needs design / an RFC.
- **inference-owned** — primary edits land in another active owner's files
  (`sim/src/inference/*`, `effects.rs`, `lifecycle.rs`) → coordinate, don't
  fork.
- **tricky-leave** — genuine engineering + scientific-judgment calls; the
  residue left after dedup + stale + s-class.

## Count-reduction order (fastest → slowest, by risk)

1. **dedup** (no code) → close dups.
2. **stale** (verify-only) → close already-fixed. _These two are the biggest
   cheap wins — backlogs accumulate issues whose fixes landed but were never
   closed._
3. **s-class** → parallel worktree batches (~4 at a time), each clean-verified.
4. leaves the **tricky-leave / l-or-proposal / inference-owned** tier for
   deliberate work.

## The parallel-knockdown discipline (hard-won)

Each s-class fix runs as a **worktree-isolated worker → independent adversarial
reviewer**, then:

- **Hardened isolation**: edit only via worktree-relative paths; never write a
  shared-checkout path. (A prior batch leaked edits into the main checkout via
  absolute paths.)
- **Branch from current `origin/main`**, not a stale base. (Inherited-commit
  bases masked two real bugs that passed worker + reviewer.)
- **worktree-green ≠ mergeable.** Always **re-verify on a clean integration**:
  apply to a clean `main`, build, run the issue's test in isolation. The clean
  re-verify is the gate that caught the `#191` gate regression and the `#147`
  `model_hash` drift after both were "approved."
- For OCaml/camdlc builds in a worktree, generate the gitignored
  `ocaml/lib/ir/ir_version_generated.ml` from `ir/VERSION` first.
- Land via `gh#NN`-named branches / patches; **don't merge or close until the
  clean re-verify is green**; rebase onto `origin/main` before push (it moves).
- **Generate the patch against the merge-base, not the moving `origin/main`.**
  Use `git diff $(git merge-base origin/main HEAD)..HEAD`, not
  `git diff origin/main..HEAD`. If `origin/main` advances after the worktree is
  created, the latter form shows everything added upstream meanwhile as a
  phantom _deletion_ by the branch (a Tier-A worker's diff appeared to delete a
  doc section that had just landed on `origin/main` — the actual patch was
  clean). When **reviewing**, review the captured patch, not
  `git diff origin/main..branch` (same artifact).
- **Split bundled issues before batching.** Some tracker issues bundle 2+
  independent findings (e.g. #127 = OOB-panic + Gillespie-clamp; #134 =
  Invalid_argument-crash + calendar-symmetry). One worker per bundled issue
  yields a partial fix the reviewer correctly flags on scope. Either split into
  focused issues first, or brief the worker to fix the named part and flag the
  rest — and never auto-close the bundle on a partial fix.
