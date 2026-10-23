# Agentic issue burn-down

How s-class issues are fixed by agents in this repository: who may do what, the
stages a fix passes through, and the mechanical checks that hold the line. The
checklist itself is the `burn-down` skill; which issues qualify is
[`issue-triage-tiers.md`](issue-triage-tiers.md) ("s-class"). This doc does not
restate either.

## Who may do what

| role                                              | may                                                                                              | may not                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| **fixer** (`.claude/agents/camdl-fixer.md`)       | reproduce, write the red test, fix, push a `fix/ghNN-*` branch, open a PR, file follow-up issues | merge, close an issue, approve, edit a high-risk file, move a golden |
| **reviewer** (`.claude/agents/camdl-reviewer.md`) | read, run tests, post findings as a PR comment                                                   | edit files, approve or request changes, merge                        |
| **maintainer**                                    | merge, close, and every design call (class-or-instance, seams, anything in Design calls)         | —                                                                    |

The reviewer is given the PR number and nothing else. A review that has seen the
fixer's reasoning is not independent of it.

## Stages

Each stage's command and output is pasted into the PR body or the review
comment; a stage without pasted output did not happen.

1. **Fixer.** Concurrency check → reproduce on current main (or stop: _stale
   candidate_) → red test shown failing → fix → narrow test run, clippy diff on
   the touched files, `make update-golden` moves nothing → sibling sweep → one
   `fix(<area>): … (gh#NN)` commit → PR with sections Summary, Reproduction, Red
   test, Gate, Sibling sweep, Design radar, Not exercised.
2. **Reviewer.** Patch taken against the merge-base → red test fails on clean
   main and passes on the branch → independent sibling sweep compared with the
   PR's → every claim in the body backed by a pasted command → Design radar
   evidenced → findings posted in `review-report` format.
3. **Maintainer.** Reads the review, settles the Design calls, re-verifies on a
   clean integration (`issue-triage-tiers.md`: worktree-green is not mergeable),
   waits for CI or a full `make test`, merges, closes.

A fixer stops and reports, instead of fixing, on: an open PR or remote branch
for the issue; a reproduction that does not reproduce; a fix that needs a
high-risk file (_needs solo session_); a sweep that turns up a class (burn-down
check 4).

## Concurrency

At most **four** agent worktrees at once, fixers and reviewers counted together.
Each fixer gets one issue; two fixers never share a file.

## Mechanical checks

Two PreToolUse hooks in `.claude/settings.json` (scripts in `scripts/hooks/`,
table test `scripts/hooks/test-hooks.sh`). Project hooks also fire inside
subagents.

- `bash-guard.sh` blocks `--no-verify`, `core.hooksPath`, `git add -A/--all/.`,
  `git commit -a/--all`, and every force-push form.
- `high-risk-edit-guard.sh`, active when `CAMDL_BATCH=1`, blocks Edit/Write on
  the six CLAUDE.md high-risk files plus `sim/src/chain_binomial.rs`. Start a
  batch session with `CAMDL_BATCH=1 claude`. It does not see Bash writes
  (`sed -i`, redirection), so the fixer's own "never" list still applies.

## `/batch`

Claude Code's bundled `/batch` decomposes a change into independent units and
runs one background worktree agent per unit, each of which implements, tests and
publishes its change without a review stage in between.

- **Allowed** for `area/cli` and `area/docs` clusters with no semantics: doc and
  comment corrections, message wording, dead-code deletion.
- **Not allowed** for anything touching the high-risk files or numerics. Its
  workers apply edits and publish with no independent review, and the bundled
  `/simplify` — "review the changed code … then apply the fixes" — rewrites code
  for style, which is exactly the change "correct before clean" forbids on those
  surfaces.

## Invoking

From a session in this repository:

```text
Use the camdl-fixer agent on gh#972.
Use the camdl-reviewer agent on PR #NNNN.
```

or `@agent-camdl-fixer gh#972`. Both run in their own worktree, cut from the
remote default branch.
