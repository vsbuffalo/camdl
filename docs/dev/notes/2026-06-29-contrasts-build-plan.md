# Contrasts `{}` build plan — stages A–C

Date: 2026-06-29 Project: camdl Tags: contrasts, dsl, ir-schema, rust-reducer,
gh#322

The executable implementation plan for the counterfactual `contrasts {}`
feature. The **spec** is
`docs/dev/proposals/2026-06-25-counterfactual-contrasts.md` (the what/why); this
note is the **how** — exact AST, tokens, productions, match sites, IR fields,
and the gated landing order. The infrastructure prerequisites (#1 keyed `(θ, X)`
output, #2 start-from-state engine seam, #4 `LatentPath` classifier) are already
built and gated; #3 (this surface + reducer) and #5 (stored quantity dimension)
remain.

## Why this lands as gated units, not small slices

Adding `ERunMember` to `expr` (`ast.ml:70`) and `DContrasts` to `declaration`
(`ast.ml:404`) breaks every exhaustive `match` on those types (expander,
dimchecker, helpers) — so the first state that even compiles is essentially the
whole OCaml frontend. And the expander's contrast lowering targets a **new IR
node**, which is an `ir/schema.json` change (golden regen). So:

- **Stage A + B land together as one gated commit** (frontend + IR node + the #5
  stored dimension + golden regen). There is no meaningful frontend-only commit.
- **Stage C (the Rust reducer)** lands as a second gated commit on top.

Both require `make update-golden` + `make test`, so both need a free machine.

## Stage A — OCaml frontend (`ocaml/lib/compiler/`)

### AST (`ast.ml`)

```ocaml
(* in `expr` (line ~70), mirroring EObsAccess (line 86): *)
| ERunMember of { run : string; ns : run_namespace; member : string; loc : loc }

(* new, near the expr type: *)
and run_namespace = NsQuantities | NsObservations

(* new decl type, mirroring quantity_decl (line 323): *)
type contrast_decl = {
  cd_name   : string;
  cd_body   : expr;            (* arithmetic over ERunMember; reuses EBinOp *)
  cd_window : expr * expr;     (* over [from_instant, to_instant] *)
  cd_doc    : doc option;
  cd_loc    : loc;
}

(* in `declaration` (line 404), after DQuantities (line 427): *)
| DContrasts of contrast_decl list
```

Reusing the shared `expr` means contrast arithmetic (`-`, `+`, scalar `*`) comes
free via `EBinOp` — no new expression grammar. `ERunMember` is parseable
anywhere an expr is (like `EObsAccess`) and rejected outside a `contrasts {}`
body by the contextual check in the expander (below).

### Lexer (`lexer.mll`)

Add to the keyword table (near line 70–81): `"contrasts", CONTRASTS;` and
`"over", OVER;`. `DOT`/`QUANTITIES`/`OBSERVATIONS` already exist (217/71/70). No
float-lexing hazard: `IDENT DOT QUANTITIES` is `.`-then-letter, so the number
rule never matches (`no_sia.5` would lex `IDENT FLOAT`, a parse error).

### Parser (`parser.mly`)

```
%token CONTRASTS
%token OVER

(* top-level decl, after the QUANTITIES rule (line 201): *)
| CONTRASTS LBRACE cs = contrast_list RBRACE   { DContrasts cs }

(* the three new primary-expr productions, mirroring OBSERVATIONS DOT IDENT
   (line ~1156). The middle token is a keyword, so these are unambiguous and
   do NOT collide with the existing observations.<stream> form: *)
| run = IDENT DOT QUANTITIES   DOT member = IDENT  { ERunMember { run; ns = NsQuantities;   member; loc = … } }
| run = IDENT DOT OBSERVATIONS DOT member = IDENT  { ERunMember { run; ns = NsObservations; member; loc = … } }

contrast_list:  list of contrast (mirror quantity_list, parser.mly:904)
contrast:       doc_opt name = IDENT EQ body = expr OVER LBRACKET a = expr COMMA b = expr RBRACKET
                  { { cd_name = name; cd_body = body; cd_window = (a, b); cd_doc; cd_loc = … } }
```

`over` precedence: the `contrast` rule consumes the whole `body = expr` _then_
`OVER`, so `a - b over [..]` parses as `(a - b) over [..]` structurally — no
precedence declaration needed (the production boundary does it). Verify no
menhir conflict against the bracketed `[..]` used elsewhere; if one appears,
make `OVER` a `%nonassoc` token below the additive operators.

**Endpoint type-check:** the window endpoints `a`, `b` must be _instants_
(`origin + 20 'weeks`, `date(...)`), not bare durations. Reuse the typed-time
instant check the `at [...]` schedule parsing uses; reject a bare duration with
a located error naming the fix (this is the same `at`-loophole the spec calls
out).

### Expander (`expander.ml`)

- Collect: add `ctx.contrast_decls` alongside
  `ctx.quantity_decls`/`scenario_decls`; the `DContrasts cs` arm appends like
  `DQuantities`.
- `ERunMember` handling in the expr walk:
  - **Inside a contrast body:** resolve `run` (a declared scenario, or the
    reserved `fitted` — see below), `member` against that namespace
    (`NsQuantities` → a `quantities {}` entry; `NsObservations` → an
    observations stream). Two-sided name resolution; located error naming the
    undeclared side.
  - **Outside a contrast body:** located error ("`scenario.quantities.x` is a
    contrast operand; …"). Mirrors how `EObsAccess` is "valid only inside a
    `quantities {}` body".
- `fitted` is the reserved no-overlay run name (E291 already reserves it for
  presets, `expander.ml`); accept it as a valid `run` in a contrast and resolve
  its members against the fitted (no-overlay) run.
- Lower each `contrast_decl` to the new IR `Contrast` node (Stage B).

### Dimchecker (the `dimcheck` pass — verify path; CLAUDE.md cites

`ocaml/lib/compiler/dimcheck.ml`)

- Give `ERunMember` a dimension = the referenced quantity/stream's dimension
  (this is what the **#5 stored dimension** is for — read it off the resolved
  member).
- The contrast binop (`a - b`) already flows through the existing dimensional
  rules (equal dims required) → reuse; E303-style error naming both members and
  their dims on mismatch.

## Stage B — IR schema (`ir/schema.json`, `ir/VERSION` 0.20 → 0.21; atomic

OCaml + Rust + golden, per "Changing the IR schema" in CLAUDE.md)

1. New IR node `Contrast`:
   ```
   Contrast { name: String, body: ContrastExpr, window: (Instant, Instant) }
   ContrastExpr = RunMember { run, ns, member } | BinOp(op, ContrastExpr, ContrastExpr)
   ```
   On `Model`: add `contrasts: Vec<Contrast>` (rust `model.rs:219` neighbour) /
   `contrasts: contrast list` (ocaml `ir.ml:750`). Default `[]`,
   `skip_serializing_if` empty so existing goldens are byte-identical.
2. **#5 stored quantity dimension:** add a resolved-dimension field to the
   `Quantity` IR node (`rust/crates/ir/src/quantity.rs`, ocaml mirror) carrying
   the dimension `dimcheck` already computes, so the Rust reducer can check
   contrast operand-dimension agreement without re-deriving.
   `skip_serializing_if` to keep no-quantity goldens unmoved (but
   quantity-bearing goldens WILL gain the field — a reviewed golden change).
3. Update `ocaml/lib/ir/{ir.ml,serialize.ml,deserialize.ml}` and
   `rust/crates/ir/src/` together; `make test-fast` to fix types; then
   `make update-golden && make update-expected`; review the golden diff (the
   contrasts field on contrast-bearing fixtures + the dimension field on
   quantity fixtures) and commit schema + both languages + goldens atomically.

## Stage C — Rust reducer (`rust/crates/`, second gated commit)

The `contrasts {}` evaluator, **auto-emitted on `fit predict`** (no new
verb/flag: when the model declares a `contrasts {}` block, `fit predict` also
writes `contrasts/<name>.tsv` under the predict output dir, alongside the
predictive output). The `fitted`/scenario arms read the fit's `(θ, X)` output:

1. Resolve the forkable subset via `fit::joint::classify_joint` (built); reject
   a point-estimate fit (`LatentPath` gate). Surface the forkable count.
2. Per forkable draw `i`, per contrast: for each `run` referenced (a scenario or
   `fitted`), resolve its param vector through the existing 5-tier resolver
   (`params_resolver.rs`: fitted draw tier-3.5, scenario `set`/`scale` tier-4),
   fork from `X_i(T*)` via the engine seam (`Resume{ start: Some(StartState) }`
   for chain_binomial `Sampled`; recompute from θ for ODE `Deterministic`), and
   evaluate the operand quantities/observation-reductions on the arm's
   trajectory (reuse the `sim::quantity` evaluator — the `DrawProducts` seam).
3. Difference the operand values **elementwise, preserving shape** (scalar /
   series / stratified / time × strata — inherited from the quantity shape).
   Shape-agreement check (located error on mismatched axes) + the dimension-
   agreement check (reads the #5 stored dimension).
4. Band per draw over the forkable subset → tidy/long `contrasts/<name>.tsv`
   keyed by `(stratum, time)` as applicable, `q05…q95 / mean / n_forkable`
   columns (reuse the quantities series/stratified emitter shape — confirm it is
   reusable, don't assume).
5. `over [window]` clips the time axis of a series operand / scopes the
   reduction window of a reduced one.

## Test plan

- **Parser/expander (OCaml):** a fixture model with a `contrasts {}` block →
  golden IR shape; the two cross-context diagnostics (`observations.x` as a
  contrast operand; `scenario.quantities.x` in a quantity recipe); the
  bare-duration endpoint rejection; the dimension-mismatch error.
- **IR round-trip:** the new golden compiles + round-trips both languages.
- **Reducer (Rust):** a small fit (PGAS, chain_binomial) → a scalar contrast
  (`fitted.quantities.x - scen.quantities.x`) with a known sign (the SIA averts
  cases); a series contrast → an averted curve (per-time output); a
  point-estimate fit (IF2) → the `LatentPath` rejection; a shape-mismatch
  (`series - scalar`) → the located error.

## Open at execution time (verify, don't assume)

- The exact dimchecker file/function (CLAUDE.md cites `dimcheck.ml`; confirm).
- Whether the quantities series/stratified TSV emitter is directly reusable for
  the contrast band writer, or needs a thin adapter.
- The menhir conflict check on `OVER` + the bracketed window.

## Pre-PR reshape + review fixes (v1.1, NOT yet done) — the runbook

Stages A–C landed (a762d370 / 8e075278) on the **`over [from, to]`** surface. A
four-agent pre-PR review + a design pass converged on a cleaner surface and
found fixes. Execute all of this as one focused pass, then PR. The branch is
pre-PR and gate-green, so this is unhurried.

### Surface change — REMOVE the window entirely (decided)

A contrast is just `name = runA.member - runB.member` (+ arithmetic). No `over`,
no fork instant, no horizon. Rationale: the fork is **derivable** (just before
the toggled intervention) and the result is **naturally shaped** (time × strata,
inherited from the operand quantity) over `[fork, run-end]`, so the horizon is
the run's own extent, not a concept. This makes the P0 (fork-at-intervention)
and the inverted-window bug **unrepresentable**, not merely guarded.

- **DELETE:** the `OVER` token + window grammar (`parser.mly`); `eval_instant`,
  **E296** (bare-duration endpoint), the endpoint type-check, the
  inverted-window guard (`expander.ml`); the `ContrastWindow` field on the IR
  `Contrast` node (`contrast.rs` / `ir.ml` / `schema.json`); `clip_trajectory` +
  its inlined `1e-9` (`contrasts.rs`). Regenerate the one contrasts golden
  (contained — the node is new in the unmerged 0.21, no version re-bump).
- **ADD (reducer, the only new logic):** derive the fork. From the contrast
  body's referenced runs, diff the scenario arms' `enable`/`disable` sets → the
  toggled intervention → its fire time (per-draw if parametric) → **fork = the
  last saved trajectory snapshot strictly before that fire time**; the arm sim
  runs `[fork, run-end]` (run-end = the sim/predict horizon). Edge cases, all
  located errors: no toggled intervention / param-only scenario → defer to
  **gh#327** (the time-scheduled-param-intervention unification); multiple
  toggled interventions → earliest, or error. Keep the shape-preserving
  difference / band / tidy-long emit unchanged.
- Update the proposal Surface section + the showcase fixture to the no-window
  form; "averted by week N" = read the time-indexed output / set the run horizon
  (a decoupled `by <instant>` clause is a later refinement, NOT the conflated
  window).

### Review findings to fix (4 agents; full detail in the session)

- **[P0] Duplicate contrast names** silently overwrite (`contrasts/<name>.tsv`
  keyed on name only). Dedup in `expand_contrasts` (cd_loc in scope), mirror
  quantities' E289.
- **[P1] E293 hint suggests invalid syntax** — for `NsQuantities` it says "write
  `quantities.foo`", which doesn't parse; correct is bare `foo`. Branch the hint
  on `ns` (`expander.ml` ~5840).
- **[P1] E295 (malformed body) has no source location** — thread `cd_loc`
  (`expander.ml:6035`). (Survives the reshape; E296/endpoint do not.)
- **[P1] Stale `#[allow(dead_code)]` + comment on `JointDraw`**
  (`joint.rs:44-60`) — the fork now consumes `d.latent`/`d.params`; delete the
  allows, fix the comment.
- **[P1] Missing tests:** the diagnostics (E292/E293/E294/E295/E297 — E296
  gone); the gh#325 (ODE → note+no-file) and gh#326 (obs-sourced → skip+note)
  deferrals; a stratified contrast (+ a deliberate stratum-mismatch error); and
  an assertion that the arms fork from the **smoothed `X(T*)`** (today's e2e
  would pass even if it forked from `init{}` at t=0).
- **[P2]:** series time-axes compared by length-not-value (`contrasts.rs:562`,
  assert `l.times==r.times`); `n_forkable` column is per-cell finite count
  (rename `n_used`/document); schema `contrast_expr.bin_op.op` unconstrained
  string (enumerate like `:183`); add `ir_contrasts_excluded_from_hash` guard
  (symmetric with quantities); OCaml contrast deserializer round-trip test;
  `resolve_joint` bypasses the #273 fixed-param backfill (`joint.rs:109` —
  backfill or assert full coverage); ODE skip note cites gh#322, should be
  **gh#325** (`contrasts.rs:139`).

### Clean (verified by review, no action)

OCaml↔Rust↔schema parity exact; run-id correctly excludes `contrasts` +
`Quantity.dimension`; 99 goldens version-only; CRN-zero + positive-median e2e
jointly non-vacuous; fork mechanics + param-resolver tiers + loud-deferral
discipline all correct.
