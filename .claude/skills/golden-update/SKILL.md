---
name: golden-update
description: Regenerate camdl's committed golden IR files after adding or changing a DSL fixture — the make update-golden fan-out, which golden sets are and aren't regenerable, baseline re-capture with CAMDL_CAPTURE_BASELINE=1, and how to stage the result. Use when adding a fixture, changing the expander, or when make update-golden moves a golden.
---

# Updating golden files

Golden IR files are committed, fully-expanded IR JSON that both languages must
parse and agree on.

**The human-loop rule lives in `CLAUDE.md` and is not repeated here.** This
skill is the mechanics.

## The regenerable sets

`make update-golden` recompiles every DSL fixture into its committed
`*.ir.json`. It fans out to `update-ocaml-golden` (→ `ocaml/golden/*.ir.json`)
plus the per-fixture sets under `tests/fixtures/*/ir/`:

```
update-golden: update-ocaml-golden update-corner-golden update-regression-golden \
               update-reactive-golden update-quantities-golden \
               update-contrasts-golden update-gradient-golden
```

```bash
make update-golden    # recompile every DSL fixture → its committed *.ir.json
```

## The frozen set that is NOT regenerated

`CLAUDE.md` states the rule (`ir/golden/` is separate and frozen). What it does
not say is what reads it: `rust/tests/golden_deser.rs`,
`sim/tests/golden_simulate.rs`, and ~two dozen other integration tests — the
cross-language serde + forward-sim smoke surface. That is why regenerating it is
a deliberate act, not a refresh.

## There is no `make update-expected`

There is no `make update-expected` / `ir/expected/` directory.
Forward-trajectory baselines for the corner-case / regression goldens are
captured into the gate tests (e.g. `gate_corner_case_baseline.rs`) by re-running
with `CAMDL_CAPTURE_BASELINE=1` — not a separate expected-TSV directory.

## Adding a new model

1. Write the DSL under the appropriate `tests/fixtures/…` (or `ocaml/golden/`)
   directory.
2. `make update-golden`.
3. **Review the emitted JSON** — read it, don't just stage it.
4. Re-capture any gate baseline it feeds: `CAMDL_CAPTURE_BASELINE=1`.
5. Commit the fixture + golden together.

## Format note

The `ir.json` format is `bf5d13b`'s compact serialization — one element per line
— chosen for a 4.6×/5× compile+size win on national-scale models; see
`docs/dev/proposals/archive/post-alpha/2026-05-30-compact-ir-serialization.md`.
An editor or formatter that re-pretty-prints these files is a regression, not a
cleanup.

## Related

- The human-loop staging rule and the `ir/VERSION` confirmation requirement:
  `CLAUDE.md`.
- The atomic OCaml+Rust+golden procedure for a schema change:
  `.claude/rules/ir-schema.md`.
- Incident that motivated the staging rule:
  `docs/dev/incidents/2026-06-09-golden-format-reverted-by-autoformat.md`.
