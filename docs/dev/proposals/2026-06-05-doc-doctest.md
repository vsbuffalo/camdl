---
date: 2026-06-05
status: accepted (v1 + CI shipped)
related: ../../ocaml/lib/compiler/doctest.ml, ../../ocaml/bin/camdlc.ml, ../../docs/camdl-language-spec.md
forward-compatible-with: 2026-06-05-compiler-diagnostic-surface.md (gh#181; consumes collect_diagnostics, which that refactor preserves)
implemented: camdlc doctest + make test-docs + .github/workflows/docs.yml + test_doctest self-test. Spec gates green (28 pass, 87 skip, 0 fail). A fragment can borrow a hidden preamble and inline data carried in the doc as invisible HTML comments — `<!-- camdl-doctest-preamble: LABEL … -->` (prepended source, referenced by ```camdl preamble=LABEL) and `<!-- camdl-doctest-data: PATH … -->` (materialised to a temp dir so read() resolves) — so converting a bucket-B fragment or bucket-C data block is self-contained, with nothing to drift. The per-block conversion of the spec's ~50 fragments / 11 data blocks remains incremental follow-up. Closing `-->` must be on its own line (CAMDL transitions contain `-->`).
---

# Doctest the CAMDL specs: compile the spec's code blocks against the real compiler

## Problem

The language spec documents CAMDL by example — 115 `` ```camdl `` blocks in
`docs/camdl-language-spec.md` (179 across all user-facing docs). Each is
verified by hand at authoring time and silently rots thereafter: the grammar
churns (`functions {}`→`forcing {}`, `~`-priors, multi-source transitions), and
a documented model that used the old surface still _looks_ authoritative but no
longer compiles. Nothing signals the drift until a reader copies it.

Scope note: this addresses **`camdl` source blocks** only. `docs/workflow.md`'s
"every command here is verified against the current CLI" is about **bash**
commands — a separate concern for the deferred run-gate, not this tool.

## What already exists (verified against current main, 40618d8)

`Compiler.collect_diagnostics : ?name -> ?filename -> string ->
Diagnostics.diagnostic list`
(`compiler.ml:401`) runs the full pipeline
(lex→parse→expand→validate→dimcheck→lint→autodiff) and returns every diagnostic
as a structured value (`{severity; code; loc; …}`) **without raising or
exiting**. That is the oracle — no subprocess, no temp files, no new compile
entry point. (gh#181 will wrap this as `compile : … -> Ir.model outcome`; the
proposal there names `collect_detail`/`collect_diagnostics` as "the right shape
to keep" and deletes only the string-typed `result` paths, so consuming it now
is forward-compatible — nothing blocks this on that refactor.)

## The corpus reality (measured: compile every block today)

Compiling all 115 language-spec blocks standalone with `camdlc`:

```
27 pass   /   88 fail
fails: 54 E001 (syntax — legends & bare fragments)
       17 E100 (undeclared name — transitions{} whose params live in prose)
       12 read() → E200 (external data file)
        few E220/E267/E268 (scenario/origin context defined elsewhere)
~6 of 115 are self-contained whole models;  ~2 blocks carry an in-fence error code.
```

The decisive finding: **the 88 failures are fragments failing as fragments, not
models that rotted.** No whole model is broken by grammar churn today. So the
job is _classification_ (tell a broken model from an intentional fragment), not
repair, and not mass-tagging.

## v1 design: compile-and-classify

A command that compiles every `` ```camdl `` block and classifies the outcome.
No directive vocabulary required to land it; intent is inferred from the
compiler's own verdict plus block shape.

**Command (maintainer-runnable):** `camdlc doctest [--gate] FILE.md …`

A `camdlc` subcommand rather than a hidden `dune runtest`-only test, because the
maintainer wants to _run the audit and read the report_. (Tradeoff noted: it
adds one line to `camdlc --help`. Accepted for ergonomics; the same code path
backs the CI gate.)

**Per-block classification** (via `collect_diagnostics`):

1. No `Error`-severity diagnostic → **PASS**.
2. Body calls `read(` or emits E200 → **SKIP:data** (needs an external file).
3. All error codes are `E001` → **SKIP:parse** (legend / bare-expression
   fragment).
4. No top-level `compartments {` (not a self-contained model attempt) →
   **SKIP:fragment** (e.g. a bare `transitions {}` whose names are declared in
   surrounding prose; typically E100).
5. Otherwise (a complete-model-shaped block with semantic errors) → **FAIL**.
6. An explicit `` ```camdl ignore `` fence forces SKIP (escape hatch for a
   complete-model-shaped block that _intentionally_ omits context for brevity —
   expected to be a small, reviewable handful, ~2–4 blocks).

**Report (default):** per-file counts
(`N pass, M skip[parse/data/fragment], K
fail`) and, for each FAIL, `file:line`,
the diagnostic codes, and the first message. Auditing prints the full picture;
`--gate` exits nonzero iff any FAIL.

This catches the failure mode that bites — _a documented complete model stops
compiling for a semantic reason_ — at near-zero migration cost: fragments are
skipped by the compiler's own verdict + the `compartments {` shape test, not by
hand-tagging ~100 blocks. The only authoring surface is the ~2–4 `ignore` tags
on complete-shaped-but-deliberately-incomplete blocks, which the first `--gate`
run enumerates.

## Recommended companion: transclude golden-mirrored whole models

`docs/book/` is a real mdbook and `docs/book/src/language/spec.md` is a
_symlink_ to `docs/camdl-language-spec.md`, so the doctest validates exactly
what ships. For the canonical _whole_ models, several are retyped copies of
`ocaml/golden/*.camdl` fixtures that `test_diagnostics.ml` already compiles. For
those, mdbook `{{#include ../../ocaml/golden/<m>.camdl}}` (verified to work in
this toolchain) makes drift _structurally impossible_ — the page embeds the file
the suite already gates. Lead with transclusion for golden-mirrored models;
doctest covers the residue (doc-only fragments and any negatives). Decide
per-block; the two mechanisms need not be uniform.

## Deferred to a later roadmap (not v1)

Earned only when a block demonstrates the need:

- **`expect=CODE` negatives.** Only ~2 in-fence negatives exist today, and
  they're inline-mixed (a correct line + a wrong line + prose in one fence; the
  wrong line is a bare transition that yields E001, not the documented E300).
  Gating them needs splitting each into a discrete must-fail _whole model_ —
  defer until the corpus warrants it. (If/when added, reuse
  `test_diagnostics.ml`'s `# expect:` parser — but it must first be extracted
  into a shared `test/expect_parse.{ml,mli}` library; today it is private to an
  isolated test executable and reports via `Alcotest.failf`. Note also the
  semantic choice: that harness asserts exact (code,severity) set-equality; a
  doc oracle likely wants membership.)
- **`needs-data` + a fixture data dir / path-base** so the 12 `read()` blocks
  can be gated rather than skipped.
- **`flags=` directive** for blocks that only compile under `--no-dim-check` or
  need `--set`.
- **`expect=W103 Warning`** severity assertions (warning-demo blocks currently
  pass without asserting the warning fires).
- **Machine-readable output** (JSON/SARIF, stable ordering) for PR annotations.
- **Multi-file scope.** 64 `camdl` blocks live outside the language spec
  (`user-features.md` 17, `intro.md` 16, `camdl-data-spec.md` 16,
  `dsl-cheatsheet.md` 7, `dates.md` 6, `run-spec`/`lineages` 1 each).
  `concepts.md`/`workflow.md`/ `fit-toml.md` have zero. Extend once v1 is proven
  on the language spec.

## CI wiring

`ci.yml` is one monolithic job that ignores `docs/**` and `**/*.md` on both
triggers, so a gate folded into it would never fire on doc PRs, and dropping the
ignore would run the full Rust+clippy+integration suite on every typo. The fix
is a **dedicated doc-triggered job** that builds only `camdlc` (~1.3 s
OCaml-only build + existing opam setup) and runs `camdlc doctest --gate` on the
spec set.

## Testing strategy

Self-test fixtures under `ocaml/test/`: a clean whole model (PASS); a bare
transition fragment (SKIP:fragment); a legend (SKIP:parse); a `read()` block
(SKIP:data); a complete model with a real dimensional error (FAIL); an `ignore`
block (SKIP). Negative control: `--gate` must return nonzero when a
complete-model block is broken, and zero on the curated-green spec. Idempotence:
two runs, identical report.
