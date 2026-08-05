# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Implementation standard

This software is used to inform major public health decisions. Errors in
inference, simulation, or data handling are not just bugs — they can mislead
policy. Every implementation must be:

- **Correct before clean**: verify logic against the mathematical derivation or
  spec before refactoring for style.
- **Tested at every step**: run `cargo test` before and after each change; do
  not batch multiple semantic changes into one commit without an intermediate
  green test run.
- **Reviewed against the proposal**: when implementing from a proposal in
  `docs/dev/proposals/`, follow it exactly unless a concrete reason to deviate
  is documented inline. Do not improvise design changes mid-implementation.
- **Conservatively scoped**: if a change touches inference math (`pgas.rs`,
  `pgas_grad.rs`, `obs_loglik.rs`, `obs_model.rs`, `if2.rs`,
  `particle_filter.rs`), treat it as high-risk regardless of how mechanical it
  looks. Read the full function before editing any part of it.

## Working on this codebase

AI is leverage; the standards belong to the maintainer. You are the careful
counterpart, not the arbiter of scientific judgment.

- **The compiler and tests are ground truth.** When unsure what a construct
  means, check the compiler, don't guess. A wrong guess must surface as a
  compile error or failing test — never as a silent change that looks plausible.
- **Verify against code, not docs — and paste the verification inline.** Doc
  text describes intent that may have drifted from the implementation. Before
  writing an incident report, a fix section, or any normative claim about how
  the system behaves _today_, run the command that verifies it (grep the file,
  read the function, run the test) and _paste the command and its output into
  the artifact alongside the claim_. Not "expander.ml uses Julian `365.25/12`"
  but "`rg 365 ocaml/lib/compiler/expander.ml` → no matches in the expander;
  OCaml does not use 365.25." The pattern self-corrects: you can't write a
  load-bearing claim without first running the command, and the command either
  confirms or refutes. If the output is too long, paste the command alone with a
  one-line summary of what it confirmed.
- **Mark inference vs verified.** "The spec says X" and "the code does X" are
  different claims. If you've only read the doc, write "the spec says X (not yet
  confirmed against the implementation)" — one clause surfaces the gap. The
  failure mode the previous rule prevents is the silent promotion of "the doc
  implies" to "the code does."
- **Fix bugs via TDD: red → green → refactor.** When fixing a reported bug,
  write a test that _asserts the correct behaviour_ first, run it and confirm it
  FAILS against the current code, then apply the fix and confirm the test now
  PASSES. The failure is the diagnostic — a test that doesn't fail on the buggy
  code isn't actually exercising the bug, and a "fix" that passes a
  never-failing test isn't proof of anything. After green: re-run the existing
  suite to confirm no regressions. This applies even when the fix looks obvious
  — "I'll write the test after" routinely produces tests that pass for the wrong
  reason (assert the symptom, not the cause; assert a related fact that was
  already true; or get the baseline wrong and silently pass). Concretely: paste
  the red-then-green test output in the commit message as the proof the fix
  landed where intended.
- **Incident reports require a reproduction.** A concrete input → wrong output,
  with the command that produced it. "Would be off by ~0.4 days" is a
  hypothesis, not an incident. If you can't produce a reproduction, the artifact
  is a _question_ filed under `docs/dev/notes/`, not a `docs/dev/incidents/`
  entry. The reproduction bar is what keeps phantoms out of the incident
  archive.
- **Classify discrepancies before proposing fixes.** Three classes, three
  different fixes:
  - _doc-vs-doc_ — edit a doc.
  - _doc-vs-code_ — verify which side is right, then sync the loser.
  - _code-vs-code_ — fix the code and add a test pinning the agreement. State
    the class explicitly at the top of any incident or proposal that depends on
    the answer. Misclassifying inflates a typo into an engineering project (or,
    the other direction, hides a real bug behind a doc edit).
- **Ship the fix; don't document the broken interim.** When a bug fix is
  straightforward and the fixed state is the right state, apply the fix and
  update the user-facing doc to describe the _fixed_ reality. Long descriptions
  of the broken interim state belong in incident reports, not in
  spec/cheatsheet/user-features. Doc-around-the-bug is noise that delays
  shipping and confuses the next reader.

### Self-check tells that you're describing rather than verifying

When you catch any of these in your own draft, stop and run the verification
before continuing:

- Hedged tense (_would_, _could_, _might_) where _is_ belongs to describe
  current behaviour.
- A detection story that doesn't name the file you read or the command you ran
  to confirm the finding.
- Corroborating detail — specific line numbers, conversion tables, three-decimal
  constants — too complete for a claim that was trivially checkable.
- Process-moralising disproportionate to what was actually verified (three
  "lessons learned" about a bug whose existence was never demonstrated).
- Self-narrated diligence as a load-bearing claim — "a careful read would have
  caught this" is itself an unverified claim about your own conduct.
- **Never lower the bar to make something pass.** No `--no-verify`, no weakening
  an assertion, no skipping a gate, no widening a tolerance to get green. If
  something fails, find the cause.
- **Surface uncertainty.** If a change touches inference math or numerics and
  you are not certain it is correct, say so explicitly and propose the test that
  would settle it. "Plausible" is not "verified" — this software informs
  public-health decisions.
- The maintainer welcomes scrutiny over speed: a found bug or a flagged dubious
  design is more valuable than a fast green diff.

### Reporting a review: the issue list goes last

Any time you report the results of a review — a proposal reviewed against the
code, a PR review, an audit, a subagent fan-out you are relaying — **the last
thing on screen must be a numbered list of concrete issues.** Not a narrative,
not a recommendation, not a summary of what you did. The list is what gets acted
on, so it must be the thing in front of the reader when you stop talking.

Each entry carries, in this order:

1. **A one-line claim** — what is wrong, stated as a fact.
2. **Severity** — blocker / high / medium / low. A blocker is something that
   makes the reviewed artifact unimplementable or unsafe as written.
3. **Where** — `file:line` or the doc section.
4. **Evidence** — the command and its output, or the input → wrong output. A
   finding without a reproduction is a _question_, and must be labelled as one.
5. **Independent or entangled** — does fixing this depend on any other item, or
   can it land alone? Say which, explicitly. This is the field that determines
   what can be parallelized or knocked out immediately, and it is the one most
   often omitted.
6. **Disposition** — fixed already (with the commit), needs a decision from the
   maintainer (with the specific question), or filed as `gh#NN`.

Rules that make the list usable:

- **Order by severity, not by discovery order or by file.**
- **Separate what is verified from what is inferred.** A reviewer's claim you
  did not re-check is not a finding — either verify it or mark it unverified.
- **Independent bugs found adjacent to the reviewed work get their own entry and
  their own issue**, never a footnote inside the reviewed artifact's findings. A
  bug in a different code path with a different trigger is a different bug even
  if you found it while looking at this one; say so in as many words, because
  the reader cannot tell from the narrative.
- **State plainly what is blocked on the maintainer and what is not.** If some
  findings are actionable now and others need a design call, split the list so
  the actionable ones can proceed while the decision is pending.
- Prose framing before the list is fine and often useful. It does not replace
  the list.

### Say "gate" only when you mean the merge bar

"Gate" is reserved for **the thing that stops a change from landing**:
`make
test`, CI, the merge criteria. In that sense it has one referent and is
worth keeping.

Do not use it for anything else. It has been stretched to cover compile-time
checks, diagnostics, rejection rules, capability dispatch, and proposals'
proposed behaviour all in the same paragraph, at which point the reader has to
re-derive which one is meant every time. Say the specific thing instead:

| instead of               | write                                              |
| ------------------------ | -------------------------------------------------- |
| "the E280 gate"          | "the E280 check", "E280", "the aggregation rule"   |
| "gate the projection"    | "reject the projection", "diagnose the projection" |
| "the gate must not fire" | "E280 must not fire"                               |
| "gated by capabilities"  | "rejected at dispatch by the capability check"     |
| "the gate as drafted"    | "the rule as drafted", "the check as drafted"      |

The general habit this is an instance of: when a word is doing several jobs in
one document, name each job. This applies to prose, commit messages, proposals,
and code comments alike.

### Design calls: say what you need, and how sure you are

When work is blocked on the maintainer's judgment, end with a **Design calls**
block — separate from the issue list, because these are questions, not defects.
Each entry:

1. **The question**, in one sentence, answerable without re-reading the thread.
2. **The options**, with the concrete consequence of each — not a survey, two or
   three real candidates.
3. **My recommendation** — always give one, even when uncertain. "You decide" is
   not an answer; it pushes the work back without adding anything.
4. **Confidence**, as one of exactly three labels:

- **Solid** — the evidence is decisive and I would act on it if you did not
  reply. Say so: "proceeding with X unless you object." Do not manufacture a
  decision point for something the code already answers.
- **Leaning** — I have a real preference and a real reason, but the tradeoff is
  genuine and one sentence from you settles it. State what would flip me.
- **Need you** — outside my judgment: a scientific call, a product call, a
  tolerance for breakage, a question about what modellers actually do. Do
  **not** guess these, and do not bury them in a recommendation dressed up as
  confident. Say plainly what I cannot determine and why.

Be honest about which label applies. Marking a genuine unknown "solid" to seem
decisive is worse than the delay it saves; marking a clear call "need you"
wastes a decision the evidence already made.

### Required reading before structural proposals

Before drafting a `docs/dev/proposals/` document or making non-trivial changes
to load-bearing surfaces, read the normative docs for that area first. Working
from a mental model of the language rather than from the spec has, in practice,
produced proposals that reinvent existing surface badly — once is bad luck,
twice is a pattern, and the pattern is fixed by reading first, not by trying
harder to remember.

Per area:

- **DSL changes** (lexer, parser, expander, dimcheck, new unit literals, new
  functions in DSL constant positions): read
  [`docs/camdl-language-spec.md`](docs/camdl-language-spec.md) end-to-end
  (especially §2 on units and dimensions, §4 on parameter kinds, §6 on tables,
  §7 on forcings), [`docs/user-features.md`](docs/user-features.md) for example
  patterns, and [`docs/dsl-cheatsheet.md`](docs/dsl-cheatsheet.md) for a fast
  orientation. For the actual grammar: `ocaml/lib/compiler/lexer.mll` (unit
  literals + tokens), `ocaml/lib/compiler/parser.mly` (the rule for whatever
  you're changing), `ocaml/lib/compiler/dimcheck.ml` (dimensional behaviour).
- **IR / schema changes**: read `ir/schema.json` (the OCaml↔Rust contract) and
  `ir/VERSION`. The atomic update procedure is at "Changing the IR schema"
  below. Cross-language constants follow the pattern of
  `rust/crates/ir/src/caltime.rs::rata_die` — single source of truth, mirror
  only with an equivalence test.
- **Calendar / time / date changes**: [`docs/dates.md`](docs/dates.md) is the
  policy document; `docs/camdl-language-spec.md` §2.1 has the unit table;
  `rust/crates/ir/src/caltime.rs` is the conversion code;
  `docs/dev/proposals/2026-05-22-calendar-time.md` and
  `docs/dev/proposals/2026-05-22-typed-time-and-dsl-ergonomics.md` are in-flight
  design.
- **Inference math**: the proposal that introduced the feature (under
  `docs/dev/proposals/`), the relevant module in
  `rust/crates/sim/src/inference/`, and any related incident reports in
  `docs/dev/incidents/`.
- **CAS / run-identity changes** (anything that feeds a `run_id`: a new
  `SimConfig` / `FitConfigV2` field, a new identity level, an output-affecting
  CLI flag): read the `runid` crate doc (`rust/crates/runid/src/lib.rs` — the
  two hashing paths and the version stack), `rust/crates/cli/src/resolve.rs`
  (`normalize_for_hash` + the factored model/config/params/scenario/seed
  levels), and `rust/crates/cli/src/fit/cas.rs` (the fit canonical-JSON hash).
  The rule: a field that changes stored bytes is identity (it must re-key); a
  re-encoding of the same values is presentation (strip it). Re-keys are
  deliberate and version-bumped, never collateral.

When a proposal is the _first_ thing you'd read about a topic, that proposal
needs to either be self-contained (cites all the existing surface relevant to
its claims) or explicitly state what background the reader is assumed to bring.
The "read the spec first" rule is for the author, not just the reviewer.

**A shipped proposal has no open questions.** A proposal in
`docs/dev/proposals/` is the spec an implementer follows — by the time it ships
(is committed as the decision record and implemented against), every design
question it raises must be **resolved**: make the call and record it inline. An
`## Open questions` section with undecided items is the tell that the proposal
is still a draft, not a spec — it punts the design onto the implementer, who
then either guesses (a silent-wrong risk) or stalls. So decide each open
question before shipping, or convert any that genuinely can't be settled yet
into a named follow-up (a `gh#NN` issue or a separate RFC) with the reason —
never leave a bare list of undecided questions in a proposal you are treating as
done. Drafting with open questions is fine; _shipping_ with them is not.

## docs/dev layout and where work gets tracked

- `docs/dev/notes/` — dated design sketches, investigation logs.
- `docs/dev/incidents/` — serious bugs/outages: cause, fix, what it changes.
- `docs/dev/reviews/` — audits and PR write-ups. Audit-fix commits cite these
  via an `Audit ref:` footer.
- `docs/dev/proposals/` — RFCs for non-trivial changes. Implementation commits
  cite via a `Proposal:` footer; follow the proposal exactly unless a deviation
  is documented inline.
- Stable normative docs live at `docs/dev/` root (e.g. `commit-style.md`,
  `testing.md`, `warning-catalog.md`).

Now that camdl is alpha:

- **Small, well-scoped work → a GitHub issue** (`gh issue create`), referenced
  as `gh#NN` in the commit subject. No proposal needed.
- **Bigger lifts** (schema/IR changes, new inference methods, anything
  cross-cutting) → a `docs/dev/proposals/` doc first, then implement against it.
- Commit/PR conventions: `docs/dev/commit-style.md`. Contributor onboarding:
  `CONTRIBUTING.md`.
- **No AI/Claude trailers in commit messages — ever.** A commit message (and a
  squash-merge body, which becomes one) must never contain `Claude-Session:`,
  `Co-Authored-By: Claude…`, `Generated with Claude Code`, a `🤖` line, a
  `claude.ai/code/session_…` URL, or any equivalent assistant attribution. This
  applies to commits authored by agents too — strip the trailer before
  committing, and when merging a PR whose commit/body carries one, rewrite the
  message clean rather than landing it. The `commit-msg` hook
  (`scripts/check_commit_trailers.sh`) rejects these; do not `--no-verify` past
  it. Provenance belongs in git history and the PR thread, not stamped into the
  permanent commit log.
- **Format Markdown with `mdfmt` (`dprint fmt`) before committing any `.md`.**
  Run it as the last step on any Markdown you touch, so formatting never rides
  in on the next substantive commit (or forces an amend after a push — `main` is
  a protected branch, so a post-push reformat can't be force-pushed; it has to
  be a fresh follow-up commit).
  - **EXCEPTION: never run `mdfmt`/`dprint fmt` on
    `docs/camdl-language-spec.md`.** The spec embeds ~30
    `<!-- camdl-doctest-preamble -->` markers and fenced `camdl` snippets that
    the doc-test harness parses; `dprint`'s reflow disturbs that structure and
    turns passing doc-tests into failures/ICEs. Edit the spec by hand and leave
    its formatting alone. (Any other `.md` with embedded doctest preambles is
    the same hazard — check before formatting.)

## Releasing

Releases follow [`VERSIONING.md`](VERSIONING.md) (policy — what a version
promises a user) and [`RELEASING.md`](RELEASING.md) (the runbook). The
load-bearing rules:

- **Tags are `vMAJOR.MINOR.PATCH`, always annotated** (`git tag -a`);
  pre-releases are dot-numbered (`v0.2.0-rc.1`, never bare `-alpha`). Pre-1.0:
  MINOR (`0.x.0`) may break the DSL/CLI/output-format surface, PATCH (`0.x.y`)
  is fixes-only. The version covers DSL + CLI + output/file formats; the IR
  schema (`ir/VERSION`) and `fit.toml` are versioned separately and _reported_
  in the notes, not folded into the release number.
- **Never hand-tag or hand-`gh release`.** Cut through the tooling (the `make`
  targets are thin wrappers over `scripts/release.sh`):
  1. `make release-suggest` — commits since the last tag + the suggested bump.
  2. `make release-prep VERSION=x.y.z` — bumps every manifest, regenerates
     `CHANGELOG.md`.
  3. Draft `RELEASE_NOTES-x.y.z.md` with the `/release-notes` skill; edit it.
  4. `make test` green + CI green.
  5. `make release-publish VERSION=x.y.z` — the only irreversible step; prompts
     before it commits, tags, pushes, and publishes.
- **Publishing is maintainer-gated.** An agent may run suggest/prep and draft
  notes; tagging and publishing are the maintainer's call, never an agent's.
- The first published release is `v0.2.0` (the `v0.1.0-alpha` tag was never
  published); cut it **explicitly** — the auto-bump continues the bare alpha tag
  to `v0.1.0-alpha.1`, which is wrong.

## Project Overview

`camdl` is a monorepo for stochastic compartmental epidemic modelling. It has
two independent subsystems connected by a shared JSON IR (Intermediate
Representation):

- **OCaml frontend** (`ocaml/`): DSL → stratification expansion → IR
  serialization
- **Rust backend** (`rust/`): IR deserialization → simulation →
  trajectory/observation output

The IR schema (`ir/schema.json`) is the contract between them. Changes to the
schema must be reflected in both language implementations atomically.

## Build Commands

```bash
make build           # build both OCaml and Rust
make build-ocaml     # cd ocaml && dune build
make build-rust      # cd rust && cargo build --release
```

## Test Commands

The gate is **tiered** (see `docs/dev/testing.md` "Tiered gate"). Run the fast
tier while iterating; the full `make test` (or CI) is the authoritative gate
before a change lands. CI mirrors every phase, so what the fast tier skips is
still caught before merge.

```bash
make test-fast       # inner loop: whole Rust workspace via cargo test
                     # (skips OCaml/integration/doc phases — NOT authoritative)
make test            # authoritative, slow: every phase; mirrors CI
make test-ocaml      # OCaml compiler + dimcheck + IR round-trip
make test-rust       # Rust workspace except sim (cargo test)
make test-inference  # the sim crate (engine + inference stack)
make test-integration # cross-language CLI shell-out (slow)

# A single Rust test file
cd rust && cargo test --test expr_eval

# Setup: optionally `brew install sccache` (compile cache; the Makefile uses it
# only when on PATH). cargo-nextest is NOT used — its parallel launch burst
# wedges macOS syspolicyd/code-signing; cargo test is sequential and safe.
```

### camdlc↔camdl version guard ("camdlc version mismatch")

`camdl` (the Rust runtime) refuses to run a `camdlc` (the OCaml compiler) whose
git hash differs from its own — an end-user safeguard against a drifted
runtime/compiler pair emitting/reading incompatible IR (the check is in
`rust/crates/cli/src/util.rs`: `find_camdlc` / `check_camdlc_version_once`).

**If you see `error: camdlc version mismatch` during a test run, it is almost
always environmental, NOT your change.** The `camdlc` on PATH (usually the
shared `~/.local/bin/camdlc`) goes stale whenever a parallel checkout runs
`make install` at a different commit. The message prints both hashes — check
them before suspecting your diff.

What to do:

- **Gate with `make test`.** It handles the guard itself — `test-rust` prepends
  the freshly-built `camdlc` to PATH and skips the handshake, so `camdl` uses
  the compiler under test regardless of `~/.local/bin`. A plain `make test` is
  the gate; you do **not** need to install or sync camdlc first.
- **Ad-hoc `camdl <model>.camdl` runs:** set `CAMDL_SKIP_VERSION_CHECK=1`, or
  `CAMDLC=ocaml/_build/default/bin/camdlc.exe`.

Do **not**, to "fix" a mismatch red:

- `make install` / `make install-camdlc` — clobbers the shared
  `~/.local/bin/camdlc` that parallel checkouts rely on, causing false reds in
  _other_ worktrees.
- Pin via the `CAMDLC` env var in the test harness — it overrides the
  PATH-injected camdlc shims that `compile_once`/`ir_cache` install
  (`find_camdlc` priority 2 > 3) and breaks those tests; the harness uses
  PATH-prepend (priority 3) so a test's own shim still wins.

Full runbook: `docs/dev/testing.md` ("Gotcha: camdlc version check"). The deeper
fix (key the guard on IR schema version, not git hash, so no skip is needed) is
in `docs/dev/proposals/2026-06-04-camdlc-version-guard.md`.

## Golden File Management

Golden IR files are committed, fully-expanded IR JSON that both languages must
parse and agree on. `make update-golden` regenerates them from their DSL
fixtures — it fans out to `update-ocaml-golden` (→ `ocaml/golden/*.ir.json`)
plus the per-fixture sets under `tests/fixtures/*/ir/` (corner_cases,
regression, reactive, quantities, contrasts, gradient):

```bash
make update-golden    # recompile every DSL fixture → its committed *.ir.json
```

There is no `make update-expected` / `ir/expected/`. Forward-trajectory
baselines for the corner-case / regression goldens are captured into the gate
tests (e.g. `gate_corner_case_baseline.rs`) by re-running with
`CAMDL_CAPTURE_BASELINE=1` — not a separate expected-TSV directory.

`ir/golden/*.ir.json` is a **separate, frozen committed set** — the
cross-language serde + forward-sim smoke surface (`rust/tests/golden_deser.rs`,
`sim/tests/golden_simulate.rs`, and ~two dozen other integration tests read it).
It is **not** regenerated by `update-golden`; folding it into the regenerable
`ocaml/golden` set is tracked in gh#384.

When adding a new model: write the DSL under the appropriate `tests/fixtures/…`
(or `ocaml/golden/`) directory, run `make update-golden`, review the emitted
JSON, re-capture any gate baseline it feeds (`CAMDL_CAPTURE_BASELINE=1`), and
commit the fixture + golden together.

### Goldens are an explicit, reviewed, human-loop change — never collateral

A golden or `ir/VERSION` change is a deliberate act, not a side effect. The
serialized format and content of `ir/golden/`, `ocaml/golden/`, and the
`tests/fixtures/*/ir/` sets are load-bearing (the `ir.json` format is
`bf5d13b`'s compact serialization — one element per line — chosen for a 4.6×/5×
compile+size win on national-scale models; see
`docs/dev/proposals/archive/post-alpha/2026-05-30-compact-ir-serialization.md`).

- **Stage goldens explicitly.** Never `git add -A` / `git commit -a` when golden
  or doc files are dirty — a formatter/editor that reformats `*.ir.json` or
  reflows markdown must not ride along in an unrelated commit. Review
  `git status` / `git diff --stat` before every commit; if goldens changed, that
  is the commit's subject, not a footnote.
- **A golden diff is reviewed by a human.** If `make update-golden` changes a
  golden, say what changed and why in the commit, and surface it — do not bundle
  it silently into a feature/docs/proposal commit. (Incident:
  `docs/dev/incidents/2026-06-09-golden-format-reverted-by-autoformat.md` — a
  docs-proposal commit silently re-pretty-printed 48 goldens; the CI gate that
  would have caught it was masked for 4 days.)
- **An `ir/VERSION` bump or an edit to `ocaml/lib/ir/` or
  `rust/crates/ir/src/`** breaks every golden and requires the atomic
  OCaml+Rust+golden update in "Changing the IR schema" below — flag it and
  confirm before proceeding.

## Quick Simulation

```bash
make sim MODEL=ir/golden/sir_basic.ir.json
# or directly:
rust/target/release/camdl simulate <model.ir.json> --traj /tmp/traj.tsv --obs /tmp/obs.tsv
```

## Debugging a diverging simulation

When a simulation's dynamics don't match a reference implementation (pomp, Stan,
a paper's published trajectory), the first tool is the per-substep tracer built
into the chain-binomial backend:

```bash
CAMDL_TRACE_STEPS=1 camdl simulate model.camdl --params p.toml \
    --backend chain_binomial --dt 1 --seed 1 --obs-only /tmp/obs.tsv \
    2> /tmp/trace.tsv 1>/dev/null
```

The trace dumps one TSV row per substep to **stderr** with columns: `t`, all
compartment counts, all `flow_<name>` (counts per substep), all `rate_<name>`
(total per-source rates evaluated this substep), and `total_pop`. Redirect
stderr to a file — stdout carries the normal TSV simulation output, so keep them
separate.

Workflow: pick a few diagnostic times (t=1, after seasonal onset, at peak,
post-epidemic trough) and compare the rate/flow columns against hand-computed
values from the reference implementation's rate expressions. A mismatch at t=1
localizes to init or rate construction; a mismatch that grows over time
localizes to dynamics (noise, forcing interaction, event ordering).

Other logging channels worth knowing about:

- `log::debug!` in `pgas.rs`, `particle_filter.rs`, `if2.rs`: inference
  diagnostics (-inf logliks, skipped observations, density mismatches). Enable
  with `RUST_LOG=camdl_sim=debug` or similar.
- `CAMDL_TRACE_STEPS=1` also activates in `intervention.rs` — logs intervention
  firings alongside the substep trace.

Before inventing new logging, check the existing paths above. They already cover
most per-step/per-iteration diagnostics.

## Architecture

### The IR as contract

The IR is a **fully-expanded** declarative model — no stratification shorthand
survives serialization. The OCaml compiler performs stratification expansion;
what reaches Rust is a flat list of compartments, transitions (with
stoichiometry + rate expression), observation models, parameters, and initial
conditions.

The expression language (`expr`) is a pure, total, first-order AST over
`Const | Param | Pop | PopSum | Time | Dt | BinOp | UnOp | Cond | TimeFunc | TableLookup | Projected | UncheckedDim | Reduce | BindingRef`.
No recursion, no lexical binding — propensities evaluate in bounded time. `Cond`
guards against division-by-zero in Gillespie. `TableLookup` keeps stratified
models compact (contact matrices, age-specific rates). `Dt` exposes the runtime
integrator step (gh#54); `Projected` is the observation-projection value (in
likelihoods); `UncheckedDim` is the dimensional escape; `Reduce` is an n-ary sum
(left-fold, matching the OCaml Add-chain order); `BindingRef` references a
hoisted model-level binding (a shared subexpression resolved by slot).

### Rust crate dependency order

```
cli → io → observe → sim → ir
```

- `ir`: pure types + serde, no simulation logic
- `sim`: simulation backends (Gillespie, ODE, chain-binomial) + propensity
  evaluator; defines the `Model` trait
- `observe`: projection + likelihood sampling/scoring; depends on `sim` for
  `Trajectory`
- `io`: TSV read/write glue
- `cli`: arg parsing + orchestration

### OCaml library order

```
expand → dsl → ir
```

- `ir`: OCaml types mirroring the schema + Yojson serialization/deserialization
- `dsl`: embedded DSL builder combinators; produces pre-expansion IR
- `expand`: base model × stratification spec → flat expanded IR (the core
  compiler logic)

### RNG and paired-seed coupling

The runtime uses a plain ChaCha8 `StatefulRng`. Paired scenarios with the same
seed produce identical trajectories only while the RNG is consumed in the same
order on both sides: pre-intervention trajectories are byte-identical for
`enable`/`disable` scenarios, and correlated-but-not-identical for `set`/`scale`
scenarios that modify propensities from t=0. Any structural change that reorders
draws also breaks the coupling — this is paired-seed CRN, NOT event-keyed RNG.

### Implementation phases

| Phase | Status      | Scope                                                                                |
| ----- | ----------- | ------------------------------------------------------------------------------------ |
| v0.1  | Complete    | Forward simulation + synthetic data generation                                       |
| v0.2  | Complete    | Inference: IF2 (MLE), PGAS+NUTS (Bayesian), particle filter, priors, real data input |
| v0.3  | In progress | Hierarchical priors, reporting pipelines, spatial coupling                           |

Public **alpha** as of 2026-05 (blog announcement): usable for real fits, public
surface documented, breaking changes still expected.

### Inference algorithms

The inference stack lives in `rust/crates/sim/src/inference/`:

- `if2.rs` — Iterated filtering for maximum likelihood estimation
- `pgas.rs` — Particle Gibbs with Ancestor Sampling (default Bayesian method)
- `pgas_grad.rs` — Gradient evaluation for PGAS (uses compiler-emitted
  `rate_grad`)
- `nuts.rs` — No-U-Turn Sampler for gradient-based parameter proposals within
  PGAS
- `pmmh.rs` — Particle Marginal Metropolis-Hastings (production; prefer PGAS for
  long observation series)
- `particle_filter.rs` — Bootstrap particle filter
- `dmeasure.rs` — Observation likelihood compilation
- `obs_loglik.rs` — Distribution log-PMFs + analytical gradients (incl. digamma)

The OCaml compiler (`ocaml/lib/ir/autodiff.ml`) performs source-to-source
symbolic differentiation of rate expressions, emitting `rate_grad` fields in the
IR. The Rust backend evaluates these derivative expressions via `eval_expr` — no
runtime autodiff, no finite differences.

### DSL features for inference

- `events {}` — Scheduled discrete state modifications (cohort entry,
  importation). Sister construct to `interventions {}` but fires every substep.
  Uses `add()`, `transfer()`, `set()` actions.
- `balance {}` — Population conservation constraint. Applied last in each
  substep after transitions and events.
- `ivp: true` — Parameter type for initial value parameters (s0, e0). PGAS draws
  stochastic initial states via Binomial(N, param).

### Backend capabilities

The `Capabilities` bitflags (`rust/crates/sim/src/lib.rs`) are **one of three**
compatibility axes — model-feature × backend.
`CompiledModel::required_capabilities()` derives a model's needs from the IR (a
DSL primitive: `overdispersed(...)`, `balance {}`, a real compartment, `dt` in a
rate); each backend declares what it provides; mismatch → hard error at
dispatch.

- `OVERDISPERSION`: `overdispersed(rate, σ²)` transitions require chain-binomial
  (NegBinomial draws). Gillespie and ODE reject these models with a hard error.
- `REAL_COMPARTMENTS`: real-valued compartments with ODE equations.

Subtlety: the "what a backend provides" side **forks by execution mode** —
`Simulate::capabilities()` (simulate path) vs a separate hardcoded table in
`fit/methods.rs::check_model_capabilities` (inference path), which deliberately
withholds `REAL_COMPARTMENTS` from chain-binomial inference (gh#191). The other
two axes — algorithm × backend (the `METHODS` registry) and model-feature ×
algorithm (scattered ad-hoc checks) — plus the known gaps are mapped in
[`docs/dev/capabilities-system.md`](docs/dev/capabilities-system.md); read it
before touching any backend/algorithm/capability gate.

### Scheduled interventions and simulation backends

Interventions are deterministic state modifications (not stochastic events).
Each backend handles them differently and the interaction is non-trivial — see
§2.3.1 of `compartmental-ir-spec.md` for the Gillespie/ODE/discrete-time
specifics. The key constraint: after a Gillespie intervention, propensities must
be fully recomputed from the modified state; do not resume with remaining
exponential time.

### Changing the IR schema

1. Update `ir/schema.json` + bump `ir/VERSION`
2. Update OCaml types in `ocaml/lib/ir/` (ir.ml, serialize.ml, deserialize.ml)
3. Update Rust types in `rust/crates/ir/src/`
4. `make test-fast` — fix type errors (then full `make test` before the commit)
5. `make update-golden` — regenerate all golden files, then re-capture any gate
   baseline the changed fixtures feed with `CAMDL_CAPTURE_BASELINE=1`
6. Commit schema + both language changes + updated golden files in one atomic
   commit

## Design Principles

### No loose semantics

Never silently accept invalid input. If a construct looks like it means
something, it must either mean exactly that or produce a clear error. Examples:
`_args` patterns that discard function arguments, optional fields that default
to "works but wrong." If the compiler accepts it, the behavior must be fully
specified and intentional.

### Every backend × inference method is a supported cell — no silent gaps

The product of forward backends (chain_binomial, gillespie, ode) and inference
methods (particle filter, IF2, PGAS, PMMH) is a dense matrix. Every cell must
either work and be tested, or fail loudly through the capability system — there
is no third option. A combination that is silently untested, silently skipped,
or excluded from a cross-cutting test behind a "covered elsewhere" hand-wave is
a latent silent-wrong-answer bug. (This is how gh#187 hid: the PGAS path
silently dropped scheduled interventions, behind a cross-backend lifecycle test
that excluded PGAS and a comment claiming another test covered it — it did not.)

- **Consolidate to the shared substrate before the matrix can drift.** Push the
  bug-prone, genuinely-shared mechanism into one path every cell routes through
  (e.g. every backend and the PGAS producer step with
  `chain_binomial::step_one`, which owns intervention/event/balance application
  via the `effects` seam) so a feature cannot be live in one cell and silently
  absent in another. Unify the shared substrate, keep the distinct algorithms
  distinct — the "natural seam" rule. Reimplementing shared behaviour per-cell
  is how cells diverge.
- **A genuine capability gap is expressed in code, not omitted.** If a
  combination truly cannot be supported, route it through the `Capabilities`
  bitflags (`rust/crates/sim/src/lib.rs`:
  `CompiledModel::required_capabilities()` vs each backend's
  `Simulate::capabilities()`), which hard-errors at dispatch with a message
  naming the limitation — and the error tells the user. Never drop the
  combination from a test or skip it silently. The capability system exists (see
  "Backend capabilities" above); if you think it does not, look harder or flag
  it to the maintainer rather than inventing a silent exclusion.
- **Tests follow the matrix.** A property that must hold across cells is tested
  for each cell it applies to. A "covered by test X" claim must name X, and X
  must actually exercise that property for that cell — verify it, don't assert
  it.

### Error messages are a feature, not polish

Error quality is a first-class design goal. A bad error message is a bug — it
means the compiler detected a problem but failed to help the user fix it.

Every diagnostic should:

- Show what went wrong (the mismatch, the constraint violation)
- Show where (source location, transition name, parameter name)
- Show why (the expected vs actual value, with domain-specific names)
- Suggest a fix when possible (hint text, corrected code)

When two possible error codes could fire for the same root cause, prefer the one
that points closest to the actual mistake. E.g., a parameter used inconsistently
across transitions should produce E303 ("conflicting dimensions in transition A
vs B") not E302 ("dimension mismatch in addition") — even though E302 is
technically correct, E303 gives the user the cross-transition context they need.

Never use `failwith` or `assert false` for user-facing errors. These produce
stack traces instead of diagnostics. Use the Diagnostics module with error
codes, source locations, and hint text.

### Design the DSL for humans first; agents follow

A meaningful fraction of `.camdl` files now come from coding agents, and that
share will grow. The temptation is to optimize the surface for agents directly —
explicit verbosity, machine-friendly tags, lots of "obvious" guardrails. Resist
it. The DSL's value to agents comes from the _same_ property that makes it value
to humans: that a sharp non-software-engineer epidemiologist (a health-ministry
modeler in an under-resourced setting, the recurring target user) can read a
model and have a chance of being right about what it does. Agents do well on
this DSL because it is human-readable, not in spite of it. When a syntax choice
is in tension between "what an agent would tolerate" and "what a model author
would understand at a glance," the model author's gut is the tiebreaker — that
is the choice that serves both audiences, because it is the one that doesn't ask
either of them to carry hidden calendar arithmetic, ambiguous units, or implicit
conventions in their head. Concretely: prefer explicitly named functions over
polymorphic operators where the semantics differ (`add_calendar_months(d, 1)`
beats `d + 1.month` when the operation is non-affine), prefer hard errors with
hint text over warnings (warnings are noise an agent will suppress and a
non-specialist will skim), and keep the surface small enough that the entire
grammar fits in a head.

### Backwards compatibility is a non-goal

camdl is alpha: the public surface is documented but breaking changes are still
expected. Do not add backwards-compatibility shims, `alias` attributes, fallback
deserialization paths, or deprecated field names. When a field is renamed,
rename it everywhere atomically. When a format changes, update all golden files.
Clean design beats legacy support — at alpha a clean break with updated golden
files is preferred over a compatibility shim.

### Breaking language changes must signpost the migration

Backwards compatibility is a non-goal, but a _silent_ break is a bug. When you
change the DSL surface in a breaking way — rename or remove a keyword, require
new syntax, tighten a semantic rule — the compiler must reject the old form with
a diagnostic that **names the replacement (old → new)**, not a bare `E001`
syntax error. A model written against last month's grammar should fail with a
migration, not a mystery. This is the error-quality bar from "Error messages are
a feature" applied to language evolution: the diagnostic is the migration tool.

And every breaking language change gets an entry — newest first, with the old →
new migration — in `docs/language-changes.md`, which is embedded into
`camdl docs language-changes` so an agent on any binary can see what changed.
The diagnostic should point there (`… see \`camdl docs language-changes\``)
until the targeted hint exists. Backfilling old changes into that log is
welcome; not adding new ones is a regression.

### Delete dead code on sight

Same principle, enforcement mechanism. Unused functions, unused modules, "v1"
paths kept around after a "v2" rewrite, prototype code kept around "in case we
need it" — all delete-on-sight. There is no consumer to placate, no migration to
stage, no contract to honour. Code that comes back can come back from
`git log -S '<symbol>'`.

- **`#[allow(dead_code)]` is a smell, not a fix.** At a definition site it tells
  a future reader "I know this is dead but didn't delete it." At a module level
  (`#![allow(dead_code)]` or `#[allow(dead_code)] mod foo;`) it hides _which
  specific items_ are dead, blocking the compiler from reporting individual rot.
  Either prove the item is reachable from a live entry point, or delete it.
- **"v1" alongside "v2" is dead code.** When a rewrite lands, the old path is
  deleted in the same commit. Carrying both is the number-one source of context
  tax.
- **Comments saying "kept in case X" are dead code with extra steps.** If X
  happens, `git log` recovers the file in seconds. Carrying it in the working
  tree forever costs every reader.
- **Ruthlessness is collegial.** Smaller surface = humans review faster, agents
  edit faster and read less context. The reader you're helping most is the one
  six months from now (often you, often an agent acting on your behalf) who has
  to load this code into a head.

When you encounter dead code while doing other work, delete it in a separate
commit before the substantive change — review is easier when each commit is one
thing.

### Reach for the existing seam before adding a parallel one

Before adding a primitive, helper, method, or constant, search for one that
already answers the question and extend it. A second function that answers the
same question a hair differently is not a convenience — it is a fork the two
sides drift across, the exact mechanism behind the silent-wrong matrix bugs
above. The boundary loop is the cautionary tale: "where does the integrator stop
next" is now answered **four** incompatible ways — `Schedule::substep`,
`Schedule::clip`, `Schedule::next_boundary`, and the unused
`Schedule::next_stop` — because successive changes each reached for a fresh
accessor instead of the one-carrying-the-reasons that already existed (gh#233).
Before you add `foo_v2` / `next_thing` / a sibling accessor: `rg` the type for
its existing methods, read them, and either call one or extend one. If you
genuinely need a new one, the commit must say _why the existing seam could not
serve_ — that one sentence is the review gate, and its absence is the smell.

### A shared primitive ships wired into a consumer, or as a named step of a committed arc

"Delete dead code on sight" has a mirror image: do not _create_ dead code by
landing a primitive that nothing calls **and that nothing is committed to
call**. The failure mode is the _speculative_ primitive — landed "in case we
need it," with no consumer and no plan, often advertised as if the consolidation
it promises were already done. `Schedule::next_stop` shipped exactly this way —
advertised as the "single boundary authority," unit-tested, never called, with
no tracked plan to finish the centralization — so the consolidation stayed
half-done, which is the soil the gh#70 / gh#208 silent-wrong bugs grew from
(gh#233). A unit-tested function with zero callers is unexercised on every path
that matters, and its tests are thin evidence (written to the same mental model
the code was — they confirm the author's intent, not the system's behaviour).

Two honest ways to land a primitive:

1. **Wired now** — the change routes at least one real consumer through it, in
   the same PR. The default.
2. **A named step of a committed arc** — a foundation landed _ahead_ of its
   consumer is sound engineering, not a stub, when it is an explicit
   prerequisite of a feature we are committed to shipping: a tracked
   proposal/issue names the consumer that will wire it, the commit says
   `foundation for <arc>; wired by
   <next step>`, and the primitive is
   exercised by its own tests meanwhile. Building good foundations on the way to
   a feature is the point — what makes it legitimate is that the consumer is
   _committed and named_, not _hypothetical_.

What stays prohibited is the orphan: a primitive with no committed consumer, or
one dressed up as if the work it enables were already complete. If you cannot
name the consumer or the arc, you do not have one yet — do not land it.

### Name tolerances and magic numbers once; never inline a bare epsilon

A bare numeric literal in control flow (`if dt <= 1e-15`,
`(iv - t).abs() < 1e-10`) is unreadable and un-greppable: the next reader cannot
tell a step floor from a due-tolerance from a rate floor, and the same concept
silently drifts in value across call sites. Define each threshold **once**, as a
named `const` at the module that owns the concept (time tolerances belong in
`schedule.rs`), with a doc comment saying what the check _means_, and reference
it everywhere. Distinct concepts that share a value keep **distinct names** — a
time `MIN_STEP_EPS` and a `RATE_EPSILON` are not the same thing even at `1e-15`.
The cost of inlining is concrete: the "effectively-zero step" threshold was
spelled `1e-15` at four sites while PGAS's equivalent floor `GRID_STEP_EPS`
silently used `1e-12` — a three-orders-of-magnitude disagreement that surfaced
only when someone tried to give it a name (gh#233).

### Parse at the boundary; don't pass raw and validate

We want **illegal states unrepresentable** — ideally a wrong wiring won't
compile, rather than being caught by a comment, a `debug_assert!`, or a test.
Hold this in a **careful, pragmatic balance**: the aim is to delete a class of
silent-wrong bug, not to turn the code into a type exercise. Add structure where
it removes a real, plausible mistake; stop before it becomes ceremony.

The high-leverage move: at a trust boundary — where raw/loosely-typed data
enters the typed core (`Vec<f64>`, `String`, `&CompiledModel`, CLI args, JSON) —
_parse_ it once into a type whose constructor is the only way to make it and
whose existence proves the invariant ("parse, don't validate", Alexis King 2019;
the operational form of "make illegal states unrepresentable" from the global
Design Philosophy). Downstream receives the parsed type and never re-checks.
Prefer a fallible smart constructor that folds _produce + validate + role-tag_
into one seam: `OutputTimes::from_model(model)?` is the producer, the
sort/finite check, and the "this is the output axis, not the effect axis" tag,
all in one place.

Tells you're validating instead of parsing — each is a cue to promote to a
parsed type:

- a `debug_assert!` of an invariant on a _public_ constructor (checked only in
  debug; the type still permits the illegal value — e.g. `Schedule::new`'s
  `debug_assert!(sorted)`);
- a comment carrying an invariant ("must be sorted", "caller guarantees finite")
  instead of a type;
- the same check repeated at several call sites;
- a primitive-heavy signature where the primitives have distinct semantics and
  the same underlying type (`fn(…, Vec<f64>, Vec<f64>)` — adjacent, swappable, a
  swap compiles → silent-wrong).

**The pragmatic line (this is where the balance lives).** Wrap a value when its
instances are genuinely different _and_ swappable into the same slot — different
semantics, same underlying type, so a swap type-checks and silently corrupts. Do
**not** wrap values that are usually the same number or already validated
elsewhere — that is the over-engineering the global "don't over-engineer" warns
against, and it is a real cost (noisier signatures and tests, tiny types the
maintainer must mentally unwrap). gh#233 shows both sides: `OutputTimes` /
`EffectTimes` / `ObsTimes` over a checked `SortedFiniteTimes` earn their keep
(three `Vec<f64>` axes with distinct meaning — record / fire / score+reset — so
a swap is silent-wrong), while `NominalStep` / `SnapGrid` scalar newtypes were
dropped (`dt == grid` at six of seven sites — ceremony). Keep wrappers at the
construction boundary and unwrap to the primitive for the hot path so nothing
threads through the inner loop.
