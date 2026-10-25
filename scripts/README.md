# `scripts/` — review zip generator

The single script in this directory is `review-zip.sh`. It produces source-tree
snapshots scoped to a subsystem, for handing to an external code reviewer
(another LLM agent, a collaborator, whoever). Everything else about review
packaging has been deliberately consolidated here.

Paste-ready reviewer prompt + staging order lives in
[`docs/dev/review-request.md`](../docs/dev/review-request.md) — copy from there
when handing a zip to an agent.

## Usage

```
./scripts/review-zip.sh inference    # one subsystem
./scripts/review-zip.sh all          # every subsystem
./scripts/review-zip.sh full         # whole repo
./scripts/review-zip.sh list         # subsystems + token estimates
./scripts/review-zip.sh clean        # rm review-zips/*.zip
```

Output lands in `review-zips/` (overridable via `REVIEW_OUTDIR=…`). Each
subsystem produces `review-<subsystem>-<YYYYMMDD>.zip` with `camdl/` as the
top-level prefix so the reviewer unzips into a recognisable layout.

## Design

### Why subsystem zips instead of one whole-repo blob

A reviewer with the whole repo sees everything, which sounds good but is
counterproductive: the LLM's context window gets eaten by code the reviewer
doesn't need to audit, leaving less room for the actual reasoning about the code
that matters. Subsystem slicing trades completeness for focus. The reviewer gets
enough code to trace data flow end-to-end _within the subsystem_ but not so much
that they lose their place.

Whole-repo snapshots are still available (`full` subcommand) for cases where the
scope is genuinely repo-wide: new-contributor onboarding, a cross-cutting bug
hunt, a bisection across subsystems. That's the exception, not the default.

### The four subsystems

The taxonomy carves the codebase along how reviews actually happen in practice,
not along directory boundaries:

- **`inference`** — fit algorithms (IF2, PGAS, NUTS, PMMH, particle filter) +
  the fit CLI that drives them (`cli/src/fit/`) + the IR types the algorithms
  consume + shared plumbing (see below). Most reviews anchored in fit accuracy /
  convergence / output shape land here. This zip is the most common request.

- **`engine`** — simulation backends (Gillespie, ODE, chain-binomial),
  propensity evaluator, intervention processing, obs sampling + shared plumbing.
  Reviews of simulate, scenarios, sweeps, or any "does the forward simulation
  compute the right thing" question land here.

- **`compiler`** — OCaml DSL frontend (`ocaml/lib/`, `ocaml/bin/`),
  stratification expansion, IR emission, OCaml tests, golden DSL fixtures, and
  the IR type definitions (`rust/crates/ir/src/`) that serve as the contract
  between compiler and runtime.

- **`docs`** — every spec and proposal in `docs/`, the root `README.md`,
  `CLAUDE.md`, and the golden DSL fixtures as reference material. Reviews of
  wording, consistency, or proposal design.

### Shared plumbing

A handful of CLI files are duplicated into both `inference` and `engine`:
`cas.rs`, `run_meta.rs`, `run_paths.rs`, `hashing.rs`, `browse.rs`, `main.rs`,
`serve.rs`, `util.rs`, `version.rs`, and `cli/tests/`. These form a "plumbing"
layer — cache invariants, directory layout, run.json schema, CLI dispatch — used
by both fit and simulate codepaths.

We considered making this its own `cli` subsystem. Rejected because
cross-cutting refactors of the plumbing (2026-04-19 unified output tree, for
example) always involve changes in fit _or_ simulate at the same time; a
reviewer of either needs the plumbing in the same zip to follow the data flow.
The per-zip duplication cost is ~90K tokens; the context-switch cost of forcing
reviewers to open two zips is worse.

A standalone plumbing-only review (adding a new subcommand, reworking cache
detection) is the only case where this split costs us — and those can use the
`full` zip or just hand the reviewer specific files directly.

### Why `git archive` rather than filesystem copy

`git archive HEAD` pulls exactly the tracked state at `HEAD`. No working-tree
cruft, no stale scratch files, no half-staged edits. What the reviewer sees is
what git sees. It's also fast (no filesystem walk), deterministic (same HEAD →
same archive bytes), and respects `.gitattributes` (e.g. `export-ignore`).

Consequence: uncommitted changes don't make it into the zip. If you want them
reviewed, commit them first. This is a feature, not a limitation — reviewing
uncommitted local state is an anti-pattern because it can't be bisected,
reverted, or pointed at with a sha.

### Token estimates

Each subcommand prints a token estimate (`~380K tokens`). The estimate is crude:
byte count / 4. It's good enough to decide whether a subsystem fits in one
context window and in what order to hand multiple zips to a reviewer. Sub-10%
accuracy isn't the goal; "does this fit" is.

### Deliberate non-features

- **No diff-scoped zips.** Review only the files touched by a specific commit
  range. Rejected because diff review misses bugs in surrounding code that the
  refactor perturbed but didn't originate. Always ship the reviewer the
  surrounding code too.

- **No auto-exclusion of build artifacts.** Not needed — `git archive` doesn't
  ship anything untracked, so build output isn't a concern. Eliminating this
  manual exclude list (which prior scripts maintained) is one of the wins of the
  `git archive` approach.

- **No per-reviewer zip variants.** We don't produce different scopes for
  different reviewers. If a reviewer needs something other than the four
  subsystems, they need a `full` zip and we trust them to scope it themselves.

## Extending

Adding a subsystem:

1. Define a new `NEW_SUBSYSTEM=( paths… )` array at the top of `review-zip.sh`.
2. Add a `make_zip <name> "${NEW_SUBSYSTEM[@]}" ;;` case in the dispatch `case`
   block and in `all`.
3. Update the `list` subcommand's hardcoded subsystem list.
4. Update this README's "four subsystems" section — the count and the taxonomy
   paragraph both need to move.

Adding a file to an existing subsystem: edit the relevant array. The
explicit-list style is deliberate — no wildcards except what git-archive
naturally supports — so that what's in scope is auditable by reading the script.
