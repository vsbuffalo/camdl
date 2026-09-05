---
date: 2026-06-05
status: goal met (2026-06-07); step-4 polish deliberately deferred
implemented: ea842d7 (compile_outcome) · ad20a78 (pure passes) · e7f484b (compile non-raising — the fix)
related: ../../ocaml/lib/compiler/compiler.ml, ../../ocaml/lib/compiler/diagnostics.ml
issue: gh#181
supersedes-partially: the gh#170 front-end-unification (collect_detail), which this generalizes
---

# Compiler diagnostic/result surface: accumulate, don't throw

## Status

The core goal is met: **the library no longer throws.** `compile` returns
`Error` on every failure (it raised `Compile_error` on late-phase errors before
— a real bug: `camdlc <late-error>` exited 2 with a `Fatal error` trace instead
of a clean 1), `report_and_exit`/`Compile_error` are deleted, the post-expansion
passes return `diagnostic list`, and `compile_outcome` is the structured
non-raising surface. The reproduction and fix are in `e7f484b`.

Deliberately **not** done, because the reproduction showed they are aesthetic
once `compile` is non-raising (consolidate to the seam, not past it): the
render-relocation to the CLI (C3), the `compile → Ir.model outcome` flip with
its ~65-caller migration, and `compiler.mli` (C5). `compile` keeps its
`(Ir.model,
string) result` type; the structured path is `compile_outcome`. If
revisited, `compiler.mli` (private `outcome` + smart constructor) is the
highest-value leftover. The Constraints (C1–C6) and Migration below are kept as
the record of what the full step-4 _would_ entail.

## Problem

The compiler expresses one notion — "did this source compile, and what is wrong
with it" — four incompatible ways, so no caller can handle errors uniformly:

| entry point                 | type                              | how a failure escapes                                                             |
| --------------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| `compile_detail_result`     | `(compile_detail, string) result` | front-end **only**; `Error` = a _rendered string_                                 |
| `run_validate`              | `compile_detail -> bool`          | diagnostics side-effected into a `mutable ctx`                                    |
| `run_dimcheck` / `run_lint` | `compile_detail -> unit`          | pure side-effect into `ctx`                                                       |
| `compile`                   | `(Ir.model, string) result`       | **raises** `Compile_error of string` on validate/dimcheck (via `report_and_exit`) |

Consequences, each verified in `compiler.ml` / `diagnostics.ml`:

1. **The `result` type lies for late-phase errors.** `compile`'s signature
   promises errors as `Error` values, but validate/dimcheck failures _raise_
   `Compile_error` (`report_and_exit`, diagnostics.ml:242, raises — it does not
   `exit`). A caller writing `match compile src with Ok … | Error …` hits an
   uncaught exception on, e.g., E507.
2. **Structure is flattened at every boundary.** Both the exception and the
   `result` error carry a pre-rendered _string_, discarding the structured
   `diagnostic list` (severity/code/loc/hint). A library caller cannot filter by
   code or re-render.
3. **Passes report output by mutation, not by type.** `run_dimcheck : … -> unit`
   _produces_ diagnostics but its type doesn't say so — output is smuggled
   through `ctx.diags` (a `mutable diagnostic list`, cons-prepended, hence the
   recurring "reverse to source order" step).
4. **Two entry points, same type, different pipelines.** `compile_detail_result`
   (front-end only) and `compile` (full) both return `result`; the type can't
   tell you one validated and one didn't, so a caller picks the short one and
   silently skips validation. This is the gh#170 root (`check` used the short
   path) and the gh#160 symptom (`check` returned a model `simulate` rejected).
5. **Location is discarded at 85% of emit sites.** 111 of 131
   `Diagnostics.{error,warning,info}` calls pass `Diagnostics.no_loc`
   (`grep -rc '~loc:Diagnostics.no_loc' ocaml/lib` over a total of 131 emit
   sites). Some of that is honest — post-expansion structural errors (E5xx,
   "duplicate compartment after stratification") have no single source span
   because stratification synthesized the clash from two origins. Much is not:
   the front-end date-literal parser `failwith`s into an E001 at `no_loc`, and
   `run_dimcheck` / `run_validate` / `run_lint` re-emit every downstream
   diagnostic at `no_loc` even where the AST carries a span. The `loc` type is
   rich; the plumbing throws it away. The surface refactor is the moment to
   thread real spans through the pass-return values, so this should land with it
   rather than as a separate sweep.

These are the same class CLAUDE.md names: stringly/flag-riddled data where an
ADT belongs, and illegal states (an unvalidated model used as if valid) left
representable.

## Sound types to keep

- `severity = Error | Warning | Info` — a clean ADT.
- `diagnostic = { severity; code; loc; message; detail; hint; related }` — the
  real currency.
- `collect_detail` (gh#170) already runs the full pipeline and returns
  diagnostics as values without raising — this is the right shape; the work is
  to make it _the_ surface, not a parallel one.

## Target design

1. **Passes return their diagnostics.**
   `run_validate / run_dimcheck /
   run_lint : compile_detail -> diagnostic list`.
   No mutation, no `bool`/`unit`. The pipeline becomes a fold that accumulates
   and short-circuits on the first `Error`-severity result.

2. **One structured outcome type** for every caller:
   ```ocaml
   type 'a outcome = {
     value       : 'a option;          (* Some iff no Error-severity diag *)
     diagnostics : diagnostic list;    (* errors + warnings + infos, source order *)
     source      : Source_cache.t;     (* for rendering *)
   }
   val compile : ?name:string -> ?filename:string -> string -> Ir.model outcome
   ```
   Errors are _values_; nothing in the library raises. `value = None` exactly
   when `diagnostics` contains an `Error`. (This is `collect_detail`
   generalized + a clean projection.)

3. **`report_and_exit` leaves the library.** Rendering-and-exiting is a CLI
   concern. The CLI top-level (and only it) does
   `match compile src with { value = Some m; _ } -> … | { diagnostics; source; _ } -> render diagnostics source; exit 1`.
   If an exception is kept anywhere, it carries `diagnostic list`, not a string.

4. **Make `Diagnostics.t` immutable (or local).** Each pass returns a list; the
   fold concatenates. Removes the `mutable` + cons-reverse dance.

## Design note: accumulate (applicative) vs sequence (monad)

`outcome` is not a `Result` and not a short-circuiting error monad. Structurally
it is `(value : 'a option, diagnostics : diagnostic list)` — i.e.
`MaybeT (Writer (diagnostic list))`: a Writer effect that accumulates the
diagnostic log monoidally, over a Maybe effect that carries the value. That
combination _is_ a lawful monad — unlike `Validation` / `Either`-with-
accumulation, which is applicative-only (its `bind` needs the success value to
choose the next step, so it cannot run a failed step's successor to collect more
errors; accumulation is inherently the applicative `<*>`, per McBride &
Paterson, _Applicative Programming with Effects_, JFP 2008). What buys the monad
back is that errors accumulate in a _separate channel_ (the Writer log) from
success/failure (the Maybe) — and the same split is what lets `outcome`
represent "compiled successfully **with** warnings," which an `Either` cannot.

Two combinators, two jobs:

- **Sequential, dependent phases → monadic `let*` (bind).** expand → dimcheck →
  autodiff: if expand structurally fails there is no model to dimcheck, so
  short-circuit the _value_ while retaining the log. This is the pipeline fold.
- **Independent sibling checks → applicative `let+ … and+ …` / traverse.**
  Within dimcheck, N transitions each produce their own diagnostics; run all,
  concat the lists. Do not `bind` siblings — bind short-circuits at the first
  bad one and hides the rest.

In OCaml (4.08+ binding operators) that is a ~15-line module:

```ocaml
module Outcome : sig
  type 'a t = { value : 'a option; diags : Diagnostics.diagnostic list }
  val return  : 'a -> 'a t
  val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t   (* sequence: short-circuit value, keep log *)
  val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t
  val ( and+ ) : 'a t -> 'b t -> ('a * 'b) t    (* accumulate: concat both logs *)
end
```

`( and+ )` (the applicative product) is where accumulation lives; `( let* )` is
where short-circuit lives. A pass is `traverse` over its siblings with `and+`;
the pipeline is `let*` over its phases.

Peer compilers split the same way by different means. Stan's compiler (stanc3 —
OCaml + Menhir, this project's stack) reports the _first_ semantic error via an
internal exception (`exception TypecheckerException of Semantic_error.t`, caught
at the boundary and turned into a `Result.t`) while _accumulating warnings_ in a
`Warnings.t list ref` (`src/frontend/Typechecker.ml`) — almost exactly camdl's
current `Compile_error` + `mutable diags`. rustc and GHC instead accumulate and
recover: rustc threads a side-effecting diagnostics context (`DiagCtxt`) and
uses `ErrorGuaranteed` as a type-level witness that an error was reported (the
same idea as the phantom-typed `Validated.t` below); GHC's typechecker monad
(`TcRn`) accumulates into an error bag and recovers. The `outcome` type puts
camdl in the accumulate camp **without** a global mutable sink — cleaner than
the stanc3 baseline, not a remediation of something uniquely broken.

## Migration

The original "every step independently green" plan does not survive contact with
the code: steps 3 and 4 are coupled and land as one commit (see **C6**). The
corrected sequence:

1. **✓ landed (`ea842d7`).** `outcome` type + `compile_outcome` over
   `collect_detail`. Pure addition, nothing raises, no caller changed.

2. **Repoint the value-typed consumers** to `compile_outcome`: `run_inspect`
   (`inspect.ml:1050`, currently on `compile_detail_result`) and the
   diagnostic-list tests. `run_check` (`inspect.ml:1099`) already routes through
   `collect_detail`. Independent and safe — the surface exists. The CLI cannot
   _fully_ migrate here because its render/exit contract moves in 3+4; the CLI
   being half-migrated after step 2 is an acceptable intermediate.

3. **+4. One atomic commit.** Change
   `run_validate / run_dimcheck / run_lint / differentiate_transitions` to
   `compile_detail -> diagnostic list`, rewrite `compile` as a fold over the
   post-expansion passes, **and** move render+exit out of the library to the two
   CLI sites (`camdlc.ml`, `inspect.ml` `run_inspect`) — all together. They
   cannot be separated: once the passes return lists instead of mutating
   `ctx.diags`, `compile`'s four inline `report_and_exit (d.ctx.diags)` reads
   (`compiler.ml:347–369`) no longer see the late-phase errors, so the fold
   rewrite strands the render sites. This commit must migrate, in lockstep, the
   three surfaces that pin the _old_ render-and-raise contract:
   - `test_json_errors.ml:123,142,174` — assert `compile` writes exactly one
     JSON array / ANSI box to stderr. Repoint to drive the CLI, or an explicit
     `render outcome.diagnostics outcome.source`.
   - `test_diagnostics.ml:500–503` — the check↔compile parity helper matches
     **both** `Error _` and `exception Compile_error _`. When `compile` stops
     raising, the `exception` arm goes dead and the parity test passes
     _vacuously_ (the trap CLAUDE.md warns against). Rewrite onto `outcome`.
   - `test_compiler.ml:5772–5788` — the step-1 test positively asserts `compile`
     RAISES on a late error (the contrast that proved step 1). Rewrite to assert
     via the new surface.
   - Also `test_dimcheck.ml:665` — a
     `with exn -> Error (Printexc.to_string exn)` catch-all that currently masks
     a raised `Compile_error` as a skipped test; fix it onto the non-raising
     path here.

4. **Delete the old entry points.** Remove `compile_detail_result` (10 sites:
   `inspect.ml:1050` + 9 tests, plus the `compile_with_diags` helper at
   `test_compiler.ml:4869`) and the string-typed `compile`. Add a `compiler.mli`
   that exposes `outcome` abstractly with a smart constructor (**C5**) and lists
   only the post-migration surface.

Gate: steps 2 and 3+4 each on a clean `make test` (OCaml unit + golden +
integration). The Rust CLI shells `camdlc` and keys only on **exit codes**, not
on parsing `--json-errors` (`rust/crates/cli/src/util.rs`), so exit codes must
stay byte-identical across the relocation; the JSON shape is not a Rust-side
constraint.

## Constraints surfaced by code review (must hold)

- **C1 — render stays out of the fold.** The one non-blocking render fires
  exactly once, as a single projection on the final `outcome`, never as a pass
  effect (compiler.ml:334–347 documents the invariant). A fold that renders
  per-pass double-emits — two JSON arrays under `--json-errors` — and fails
  `test_json_errors`, which calls `compile` directly (so it breaks before the
  CLI is touched). The fold is _pure accumulation_ into `diagnostics`.
- **C3 — the CLI replaces the string-shape sniff, doesn't relocate it.**
  `camdlc.ml:157–163` and `inspect.ml:1050–1055` branch on the payload string
  (`= "compilation failed"` / `e.[0] = '['`) to suppress a redundant error line.
  That sniff is dead once the library stops rendering; both sites must instead
  render `outcome.diagnostics outcome.source` once (honoring `json_errors_mode`
  exactly as `Diagnostics.render`) then `exit 1`, with **no**
  `Printf.eprintf "Error: …"` fallback. `run_check` (inspect.ml:1099) is the
  template.
- **C4 — immutability is post-expansion only.** The expander emits via the
  mutable `ctx.diags` at 117 sites, and `front_end_collect` drains two global
  refs (`Lexer.pending_warnings`, `Parser_errors.pending_errors`,
  compiler.ml:117–130) into it. Those remain. The fold's _seed_ is the
  already-reversed front-end diagnostic list; only the post-expansion passes
  become pure list-returning functions. Do not promise expander immutability —
  Target-design §4's "make `Diagnostics.t` immutable" applies to the
  post-expansion segment, not the whole pipeline.
- **C5 — enforce the `outcome` invariant in the type, not by convention.**
  `value = Some` with an Error-severity diagnostic present is constructible
  today (the invariant holds only at the `compile_outcome` construction site).
  Expose `outcome` abstractly via `compiler.mli` with a smart constructor that
  forces `value = None` whenever the log carries an Error, and make the
  applicative `and+` recompute `value` from the _merged_ log — otherwise
  "value=Some from one branch + an Error in the other branch's log" leaks the
  very illegal state this proposal exists to remove.
- **C6 — steps 3 and 4 are one commit** (see Migration). Steps 1–2 are
  independent; 3-as-separate-from-4 cannot land green.

## Aspirational (separate, larger): phantom-typed validated model

Distinguish `Ir.model` (unvalidated, straight from the expander) from a
`Validated.t` that is _only_ constructible by passing validation, and have
`simulate`/`fit` require `Validated.t`. Then an unvalidated model cannot reach
the runtime — the gh#160 class becomes a compile error by construction, not a
runtime E507. Bigger change (touches the OCaml↔Rust boundary and every runtime
entry); call it out, don't bundle it.

## Out of scope

The sibling type-design issues — #107 (`bool always_active` / `ParamKind`
enums), #101 (lineage ID newtypes), #98 (typed-time unification) — are the same
"ADTs over flags/strings" family but independent; this proposal is the
diagnostic/result surface only.
