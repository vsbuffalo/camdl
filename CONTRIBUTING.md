# Contributing to camdl

Thanks for your interest. camdl is alpha software that informs public-health
modelling, so the bar for correctness is high and the contribution process
reflects that. This document is the human counterpart to `CLAUDE.md` (which
guides AI agents working in the repo); both encode the same philosophy.

## Philosophy

camdl is built on the premise that the most consequential bugs in scientific
software don't crash — they produce plausible, wrong numbers that reach
decisions. See
[_On AI-assisted scientific software_](https://vincebuffalo.com/blog/introducing-camdl/#on-ai-assisted-scientific-software)
for the full argument. In practice, for contributors:

- **The compiler and tests are ground truth.** A change that's wrong should
  surface as a compile error or a failing test, never as a silent change that
  merely looks plausible.
- **Never lower the bar to make something pass.** No `--no-verify`, no weakened
  assertions, no skipped gates, no tolerance-widening to get green. If something
  fails, find the cause.
- **Surface uncertainty.** If a change touches inference math or numerics and
  you are not certain it is correct, say so in the PR and propose the test that
  would settle it. "Plausible" is not "verified."
- A found bug or a flagged dubious design is more valuable than a fast green
  diff. Scrutiny is welcome.

## Where work gets tracked

- **Small, well-scoped change** (bug fix, doc fix, small feature) → open a
  [GitHub issue](https://github.com/vsbuffalo/camdl/issues) first
  (`gh issue create`), then reference it as `gh#NN` in your commit subject. No
  design doc needed.
- **Bigger lift** (IR/schema change, new inference method, anything
  cross-cutting) → write a proposal in `docs/dev/proposals/` and get alignment
  _before_ implementing. Implement against the proposal; document any deviation
  inline.

The `docs/dev/` layout: `notes/` (design sketches, investigation logs),
`incidents/` (serious-bug writeups), `reviews/` (audits), `proposals/` (RFCs).
Stable normative docs live at `docs/dev/` root.

## Development setup

```bash
./install.sh          # OCaml (opam) + Rust (rustup) toolchains, deps, build
# or follow the manual steps in README.md "Prerequisites"

brew install sccache         # optional compile cache — used only if on PATH

make build            # build OCaml + Rust
make test-fast        # fast inner loop: Rust workspace (cargo test)
make test             # authoritative gate (every phase; mirrors CI) — green before a PR
```

The gate is tiered: `make test-fast` while iterating, full `make test` (or CI)
before a change lands. CI mirrors every phase, so what the fast tier skips is
still caught before merge — see `docs/engineering/testing.md`, "How `make test`
runs".

If you change the IR schema, follow the atomic-update procedure in `CLAUDE.md`
("Changing the IR schema") — schema + both language implementations + golden
files in one commit.

## Commit & PR style

Commit messages follow **`docs/dev/commit-style.md`** — read it. The
load-bearing rules:

- Subject: `type(scope): summary` or `gh#NN: summary`, imperative mood, ≤72
  chars. Types: feat/fix/docs/refactor/test/chore/perf/ cleanup/proposal.
- A body explaining _why_ and the mechanism is the default; the diff shows
  _what_.
- **No AI or `Co-authored-by` trailers.** This is not the project's practice (0
  of ~1000 commits have one).
- One concern per commit; don't batch unrelated changes.

Conventional Commits are load-bearing twice over: they drive the changelog
(`cliff.toml` / `make changelog`) and the SemVer bump. What a version number
promises is in [`VERSIONING.md`](VERSIONING.md); how a release is cut is in
[`RELEASING.md`](RELEASING.md).

## High-risk files

Changes touching inference math are high-risk regardless of how mechanical they
look: `pgas.rs`, `pgas_grad.rs`, `obs_loglik.rs`, `obs_model.rs`, `if2.rs`,
`particle_filter.rs`. Read the full function before editing any part of it, and
include a validation argument (a test, a reference comparison, a derivation) in
the PR.

## Before you open a PR

- [ ] Full `make test` is green locally (or you've let CI run it) —
      `make
      test-fast` alone is the inner loop, not the gate.
- [ ] Commits follow `docs/dev/commit-style.md` (no AI trailers).
- [ ] One concern per commit; subject is `type(scope):` or `gh#NN:`.
- [ ] If it touches inference math, the PR explains why it's correct.
- [ ] Linked to a `gh#NN` issue (small change) or a `docs/dev/proposals/` doc
      (bigger lift).
- [ ] Schema changes update both languages + golden files atomically.

## After you open a PR

CI (`make test` on Linux) gates the merge — `main` is branch-protected and will
not accept a merge until the `test` check is green.

**First-time contributors:** GitHub requires a maintainer to approve the first
workflow run on your PR, so checks may show as empty until that happens. That's
expected, not a failure — it's a one-time security gate, not something you need
to fix.

Squash-merge is the norm; the maintainer rewrites the squash message to the
commit-style guide before merging.
