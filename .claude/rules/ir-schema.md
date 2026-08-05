---
paths:
  - "ir/**"
  - "ocaml/lib/ir/**"
  - "rust/crates/ir/src/**"
description: The IR contract between OCaml and Rust — required reading, the atomic schema-change procedure, cross-language constants
---

# IR schema — the OCaml↔Rust contract

## Required reading

`ir/schema.json` (the contract) and `ir/VERSION`.

## The IR as contract

The IR is a **fully-expanded** declarative model — no stratification shorthand
survives serialization. The OCaml compiler performs stratification expansion;
what reaches Rust is a flat list of compartments, transitions (each carrying
stoichiometry and a rate expression), observation models, parameters, and
initial conditions.

The expression language (`expr`) is a pure, total, first-order AST over
`Const | Param | Pop | PopSum | Time | Dt | BinOp | UnOp | Cond | TimeFunc | TableLookup | Projected | UncheckedDim | Reduce | BindingRef`.
No recursion, no lexical binding — propensities evaluate in bounded time.

- `Cond` guards against division-by-zero in Gillespie.
- `TableLookup` keeps stratified models compact (contact matrices, age-specific
  rates).
- `Dt` exposes the runtime integrator step (gh#54).
- `Projected` is the observation-projection value (in likelihoods).
- `UncheckedDim` is the dimensional escape.
- `Reduce` is an n-ary sum (left-fold, matching the OCaml Add-chain order).
- `BindingRef` references a hoisted model-level binding (a shared subexpression
  resolved by slot).

The same properties that make dimension-checking tractable make source-to-source
autodiff a compact pattern match in `ocaml/lib/ir/autodiff.ml`.

## Changing the IR schema

Both language implementations must change atomically.

1. Update `ir/schema.json` + bump `ir/VERSION`
2. Update OCaml types in `ocaml/lib/ir/` (ir.ml, serialize.ml, deserialize.ml)
3. Update Rust types in `rust/crates/ir/src/`
4. `make test-fast` — fix type errors (then full `make test` before the commit)
5. `make update-golden` — regenerate all golden files, then re-capture any gate
   baseline the changed fixtures feed with `CAMDL_CAPTURE_BASELINE=1`. There is
   no `make update-expected` target — do not invoke one.
6. Commit schema + both language changes + updated golden files in one atomic
   commit

An `ir/VERSION` bump or an edit to `ocaml/lib/ir/` or `rust/crates/ir/src/`
breaks every golden. Flag it and confirm before proceeding — see the human-loop
rule in `CLAUDE.md` and the `golden-update` skill.

## Cross-language constants

Follow the pattern of `rust/crates/ir/src/caltime.rs::rata_die` — single source
of truth, mirror only with an equivalence test.
