# Stratum provenance: telling a subpopulation from an integration stage

Date: 2026-07-27 Status: draft — the gate design (§2) is unresolved; see "The
gate as specified does not hold" Fixes: gh#478 (part 1), gh#487 Related: gh#459,
gh#333

The dispatch repairs (§3) have landed; everything else here is still draft.
Landed: `prevalence()` lowers every argument shape, multi-argument
`prevalence()` sums rather than dropping, E280's hint prints forms that compile,
the explicit-aggregation sum peels nested sums and honours its `where` guard.

## The problem

`incidence()` and `prevalence()` are siblings: one projects a cumulative flow,
the other a state snapshot. On a stratified family, `incidence` refuses to guess
what a bare name means and `prevalence` guesses silently.

Same model, same un-indexed stream, four strata:

```text
projected = incidence(infection)
  → error[E280]: observation 'cases' is un-indexed, but `incidence(infection)`
    would silently sum all 4 strata of 'infection' and apply reporting uniformly

projected = prevalence(I)
  → compiles, runs, emits numbers
    IR: {"current_pop_sum": ["I_child_north", "I_child_south",
                             "I_adult_north", "I_adult_south"]}
```

E280 exists for a reason the spec states at
`docs/camdl-language-spec.md:2616-2617`: the bare form is rejected "precisely so
this aggregation decision is never made silently." `rho * sum(...)` and
`sum(rho[p] * ...)` are different models, and pooling picks the first. A
modeller who writes `prevalence(I)` on an age-stratified model fits pooled
prevalence and is never told.

The other half of gh#478 — a partial index compiling to a dangling `current_pop`
— is already fixed (PR#480, now `E503` at compile time).

### The hole is bigger than the un-indexed case

Adding `[a in age]` to the stream header — the first thing a user will try when
the new error appears — lands somewhere worse. Measured:

```text
obs[a in age] { … projected = prevalence(E) … }
  → compiles, no diagnostic
    obs_child  {"current_pop_sum": [E_child_s1 … E_adult_s3]}    # all 6 cells
    obs_adult  {"current_pop_sum": [E_child_s1 … E_adult_s3]}    # all 6 cells
```

Every stratum row is scored against the **pooled total**, with any per-stratum
`rho[a]` absorbing the mismatch. That is a silent-wrong fit, not merely a silent
aggregation, and `incidence` has the identical hole on an indexed stream. So the
gate must not be scoped to un-indexed streams.

## Why the obvious fix is wrong

Extend E280 to `prevalence` and staged-residence models break.

`via erlang(stages = 3, …)` splits a compartment into `E_s1`, `E_s2`, `E_s3` to
give it a realistic dwell time. Those stages are a **numerical device**, not
subpopulations — nobody reports "prevalence in latent stage 2." Pooling them is
the only sensible reading, it is specified in
`docs/dev/proposals/2026-04-17-state-snapshot-projections.md`, and two tests pin
it. Verified:

```text
onset : E --> I via erlang(stages = 3, mean = 4 'days)
projected = prevalence(E)   → {"current_pop_sum": ["E_s1","E_s2","E_s3"]}   # must keep working
```

And nothing distinguishes the two cases, because `via` lowering builds the same
record the parser builds for a user declaration:

```ocaml
(* parser.mly:1305 — a user's stratify(...) *)
{ sdim = !dim; sonly = !only }

(* expander.ml:1458 — synthesized by via lowering *)
{ sdim = dim_name; sonly = Some [ src ] }
```

`stratify_decl` is `{ sdim; sonly }` (`ast.ml:280`). Those are the only two
construction sites in the tree.

## The gate as specified does not hold

The rule in §2 is enforced in three AST arms (§5). Every other projection shape
reaches `| ProjDerived e -> Ir.DerivedExpr (resolve_expr ctx env e)`, where the
bare-name-sums rule of spec §5.1 applies with no gate at all. Measured on an
age-stratified SEIR with an un-indexed stream:

```text
projected = prevalence(I)               → current_pop_sum [I_child, I_adult]     GATED
projected = rho * I                     → derived_expr {mul, pop_sum[I_child, I_adult]}   NOT GATED
projected = rho * sum(a in age, I[a])   → byte-identical to the line above
```

The third line is the migration this proposal recommends. It produces the same
IR as the second, which is one character away from the form the gate rejects and
is itself ungated. A modeller who hits E280, reads "write the sum over cells
directly," and writes `rho * I` gets the identical pooled model — now with the
compiler appearing to have checked it.

Only the per-stratum form is IR-distinguishable:

```text
projected = sum(a in age, rho[a] * I[a]) → derived_expr {reduce: […]}
```

So the gate as drafted rejects a spelling and accepts its synonym. Two coherent
designs; this proposal cannot ship until one is chosen.

**A. Lint at the resolved IR.** Accept that the question is "did you mean to
pool?", and ask it of `Ir.expr` rather than the surface AST: walk the resolved
projection for a `PopSum` spanning more than one reportable cell of one family.
That covers `rho * I`, `I / N`, let-bound names, and every future spelling for
free, because it looks at the value rather than the syntax. Since no spelling
silences it honestly, it must be a `W1xx`, not an error.

**B. Semantic gate with a distinguishable escape.** Reject any projection whose
resolved value pools more than one reportable cell — including
`rho * sum(a in age, I[a])` — and accept only forms the IR can tell apart, i.e.
`sum(a in age, rho * I[a])` (a `Reduce`). The hint then names a form that
genuinely states something the bare name did not.

B is the stronger guarantee and matches the proposal's stated thesis; A is
cheaper, breaks nothing, and is honest about being advisory. B changes what
`rho * I` means for existing models and needs its own breakage sweep. Either way
the gate does **not** belong in the three AST arms of §5.

## Proposal

### 1. Give the record its provenance

```ocaml
type stratum_origin =
  | UserStratum       (* a `stratify(...)` declaration in the model *)
  | ResidenceStages   (* synthesized by `via erlang(...)` lowering *)

type stratify_decl = { sdim : string; sonly : string list option;
                       sorigin : stratum_origin }
```

Compiler context only — **not** IR. No schema change, no `ir/VERSION` bump, no
golden churn. Two predicates, both needed:

```ocaml
(** Is this dimension NAME a synthesized residence axis? For consumers that
    hold dimension names and no compartment (the gh#460 E237 check). *)
val is_residence_stage : ctx -> string -> bool

(** The dimensions of compartment [c] that denote real subpopulations. *)
val reportable_dims : ctx -> string -> string list
```

### 2. Gate on cells pooled, not on stream indexing

**A projection is rejected (E280) when it would pool more than one _reportable_
cell.** Residence stages never count. This single rule covers every case:

| projection                              | reportable cells | outcome                             |
| --------------------------------------- | ---------------- | ----------------------------------- |
| `prevalence(I)`, `I` unstratified       | 1                | ok                                  |
| `prevalence(E)`, stages only            | 1                | ok — stages pool                    |
| `prevalence(I)`, user-stratified        | n > 1            | **E280**                            |
| `prevalence(E)`, user dims + stages     | n > 1            | **E280**, naming only the user dims |
| `prevalence(E[a])` in an indexed stream | 1                | ok — stages pool                    |
| `prevalence(I)` in an indexed stream    | n > 1            | **E280** (closes the hole above)    |

This supersedes "un-indexed streams only." It also means the **`incidence` gate
must move to the same rule**, since it has the same indexed-stream hole. That is
a wider break than gh#478 alone; see Migration.

A single-level dimension pools nothing (`{"current_pop": "I_main"}`), so it
passes — counting cells rather than dimensions gets this right for free.

### 3. Fix the `'?'` fall-through, and reuse E287 — LANDED

The dispatch matched on the argument's _shape_, and **three** distinct shapes
reached the catch-all, not one:

| shape                    | produced by                                |
| ------------------------ | ------------------------------------------ |
| `ESum` over a stage axis | `via erlang` (`sum_staged_refs`)           |
| `EBinOp` Add-chain       | `via hyper_erlang` (`sum_hyper_refs`)      |
| arbitrary arithmetic     | a user writing `prevalence(Y1[a] + Y2[a])` |

Matching the `ESum` shape alone — which is what the text below prescribed —
fixes `via erlang` and leaves gh#487 open, because hyper_erlang's lowering
builds an Add-chain, not a sum. The third shape had a live instance in-tree:
`docs/dev/proposals/fixtures/garki_post_proposal.camdl` did not compile.

The landed fix is shape-agnostic: anything that is not a single bare or fully
indexed compartment reference is resolved as an ordinary state expression.
Multi-argument `prevalence(X1, X2)` sums its arguments as
`docs/camdl-run-spec.md` §14.1 documents, instead of silently dropping every
argument after the first.

The E287 half is **not** landed and is still open. One hazard the text below
does not address: E287's message and hint are built from the _raw_ dimension
vector, so on a mixed compartment it would print `[age, __onset_stage]` and
suggest `sum(s in __onset_stage, …)` — naming a synthesized identifier the user
never wrote, which is exactly what the neighbouring gh#460 code comments say
must not happen. E287's catalog row also scopes it to "a rate read" and would
need rewording. Note `test_compiler.ml`'s
`test_prevalence_partial_index_is_rejected_at_compile` asserts E503 today and
breaks under this change — a third breaking test the Migration section does not
list.

Original text follows.

On a compartment that is both user-stratified and staged:

```text
projected = prevalence(E[child])
  → error[E503]: unknown compartment referenced: '?'
```

The cause is **not** partial indexing. `sum_staged_refs` (`expander.ml:1243`)
recurses into `EFuncCall` arguments and is applied to `od.oprojection`
(`expander.ml:1541-1545`), rewriting the argument into
`sum(s in __onset_stage, E[child, s])`. It is then an `ESum`, so
`prevalence_projection`'s dispatch falls to `| _ -> Ir.CurrentPop "?"`.

`via hyper_erlang` reaches the _same_ catch-all by the _same_ mechanism
(`sum_hyper_refs`, `expander.ml:1283`), which is why gh#487 folds in here rather
than standing alone — it is one fix, not two.

Two changes: teach the dispatch to consume the rewritten `ESum` shape, and for a
genuinely partial projection index emit **E287** — the diagnostic the rate path
already gives for the identical mistake — instead of E503 on `'?'`:

```text
error[E287]: compartment 'I' has dimensions [age, patch] but only 1 of 2
             were indexed; a partial index has no defined cell
```

### 4. Second consumer, already written

gh#460 ships a staged-endpoint check that sniffs the dimension name
(`expander.ml:6331-6333`):

```ocaml
let staged ep =
  List.exists (fun d -> String.length d > 2 && String.sub d 0 2 = "__") ep.ep_dims
```

That is the stringly-typed shortcut that produced a silent-wrong in gh#460. It
becomes `List.exists (is_residence_stage ctx)`; `ctx` is in scope. So the field
lands wired into two consumers.

### 5. Live match arms — the obvious one is dead

`ProjPrevalence` (`ast.ml:302`) is **never constructed**; the parser emits only
`ProjDerived`. Its arm at `expander.ml:7006` is unreachable. (The occurrence at
`8034` is an or-pattern shared with `ProjIncidence`, which _is_ constructed at
`expander.ml:6819`, and with `None`; it cannot simply be deleted.)

The arm _names_ below are right and the line numbers are not — as cited,
`7009-7014` is the **incidence** arm, so an implementer following the numbers
puts the prevalence gate in the wrong one. Identify the arms by pattern, never
by line:

- `ProjDerived (EFuncCall ("prevalence", …))`
- `ProjDerived (EIdent …)` — the `projected = I` form
- `ProjDerived (EIndex …)`

Putting it in the arm named after the feature yields a gate that never fires,
with green tests. But see "The gate as specified does not hold" above: gating
these three arms is the wrong mechanism regardless of which arms they are,
because every other shape flows through the `ProjDerived e` catch-all ungated.

### 6. Readers that must keep the RAW dimension vector

`reportable_dims` is for the gate and the gh#460 predicate. Everything else
keeps `comp_dims`. Two are load-bearing:

| site                                                 | why raw                                                                                                                                             |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `expander.ml:6247` `resolve_action_endpoint.ep_dims` | filtering stages would let `transfer(to = E)` on a staged `E` pair with an unstaged endpoint — the gh#459-class silent-wrong E237 exists to prevent |
| `expander.ml:8825` `build_model_structure`           | the IR carries `"compartment_dims": {"E": ["age","__onset_stage"]}`; stripping it churns every staged golden and removes dims Rust needs            |

Others that stay raw: `expand_compartment_name` (2064),
`check_declaration_names` (2132), the E287 guard (3372), via lowering's `n_pre`
(1417-1419), the hyper_erlang E248 guard (1578-1580), E214 (1995), W103 (7960),
and `inspect.ml`.

**Invariant this relies on:** stage axes are always a _suffix_ of a
compartment's dimension vector, because `lower_via_transitions` runs after
`collect_declarations` and appends (`expander.ml:1458`). True today,
load-bearing for cell enumeration order, previously unstated.

## Migration

`prevalence()` is **head-position sugar**, not an expression function —
verified:

```text
projected = rho * sum(a in age, prevalence(I[a]))    → error[E100]: undeclared function 'prevalence'
projected = sum(a in age, rho[a] * prevalence(I[a])) → error[E100]
```

So E280's existing hint text, which prints `sum(p in dim, incidence(tr[p]))`
forms, is **non-compiling** for prevalence. The migration is to drop the
wrapper, which works everywhere including the mixed case:

```text
projected = rho * sum(a in age, I[a])   → derived_expr {mul, pop_sum:[I_child,I_adult]}
projected = sum(a in age, rho[a] * I[a]) → derived_expr {reduce:[…]}
projected = sum(a in age, E[a])          → derived_expr {pop_sum:[E_child_s1 … E_adult_s3]}
```

That last line settles a question the first draft got wrong: **the arity change
is not needed.** `sum(a in age, E[a])` already pools stages implicitly on a
mixed compartment, so `reportable_dims` does not have to become the arity
authority for projections, and spec §12.1's "an index never marginalizes" stands
unamended. The proposal is smaller for it.

This keeps the spec's own position (§12.1, line 2601): "write the sum directly;
`prevalence(x)` is kept as sugar only for the single-compartment case."

**The migrated projection is not IR-identical.** `CurrentPopSum` becomes
`DerivedExpr`, which gains a `projection_state_grad` entry per cell. Same
`TemporalKind::Instant`, no inference capability lost, but affected models
**re-key their `run_id`** and cached fits invalidate. Announce it.

### Measured breakage

- **camdl repo: zero.** All 119 committed `.camdl` compiled; no observation
  stream emits `current_pop_sum`.
- **camdl-garki: zero.** All 62 models inspected at source (national-scale ones
  not compiled). Every stream is indexed and every projection is an explicit
  indexed expression. Their discipline is exactly what the gate would enforce.
- **camdl-book: zero.** Eight models use bare `prevalence(I)`; all unstratified.
- **Two OCaml tests break**, both the hand-rolled staged-residence shape:
  `test_compiler.ml:5679` `test_prevalence_on_stratified_compartment` and
  `test_compiler.ml:5726` `test_projected_bare_stratified_compartment`.

Hand-rolled staged residence (`stratify(by = …_stage, only = [X])`) exists in 8
files plus those 2 tests — `ocaml/golden/seir_erlang*.camdl`, the book's
teaching model, and Garki's 5 `gstage` mosquito models — but **none observes
prevalence on the staged compartment**, so none breaks. A `kind = stages` opt-in
keyword would buy zero real models today.

The incidence half of §2 has now been swept, closing decision 8's open item:
**zero** affected models. Across `camdl`, `camdl-book`, `camdl-garki`,
`camdl-nigeria-polio`, `camdl-overfit`, `camdl-vignettes` and
`playpen-camdl-measles`, no committed model has an indexed observation header
followed by a bare `incidence(...)` / `prevalence(...)` / bare-identifier
projection. Every indexed stream found indexes its projection too. The only
bare-`incidence` hits are un-indexed streams (already E280) or pre-alpha models
that no longer parse.

Two corrections to the numbers above: the camdl repo has 119 `.camdl` files but
93 compile — 25 of the rest are deliberate error/lint fixtures and one,
`docs/dev/proposals/fixtures/garki_post_proposal.camdl`, was a live instance of
the `'?'` bug (now fixed). The operative claim, that no observation stream emits
`current_pop_sum`, holds: zero hits across all that compile, and zero in any
committed `*.ir.json`. Hand-rolled staged residence exists in 9 files, not 8 —
`camdl-garki/models/ctl_bb_erlang.camdl` uses `stratify(by = stage)` and is
missed by a `gstage` grep. Its projection is a `DerivedExpr` over lets, so the
conclusion (none breaks) is unaffected.

## Docs to update

- **spec §25.4 (~line 4979)** documents the removed behaviour verbatim
  (`prevalence(R)` → `CurrentPopSum(["R_child","R_adult"])`) and is _already_
  stale for incidence in the same block. The doctest harness skips it, so CI
  will not catch either.
- **spec §12.1 (2612-2617)** — the "state the aggregation explicitly" bullets
  are incidence-only; add the compiling prevalence forms.
- **spec §9.4.1 (line 2016)** — "a bare `E` or `prevalence(E)` still sums every
  stage" stays true; add _why_ (stages are not reportable).
- **`docs/dev/warning-catalog.md:79`** — E280's row is incidence-worded.
- **`docs/language-changes.md`** — entry required (this is agent-visible via
  `camdl docs language-changes`).
- **`camdl-book/guide/getting-started.qmd:705-740`** teaches
  `projected = prevalence(I) / N0`, which E100s today. Pre-existing bug, worth
  fixing alongside. The book's spec chapter is auto-synced and needs no edit.
- No change needed in `user-features.md`, `dsl-cheatsheet.md`, `intro.md`,
  `agents.md`, `inference.md` — every hit is a `quantities { }` block or prose.

## Diagnostic

Keep **E280** (same rule, same fix; E2xx is nearly exhausted). Reword for
prevalence and name the actual dimensions instead of the literal `<dim>`
placeholder the current message prints:

```text
error[E280]: observation 'cases' is un-indexed, but `prevalence(I)` would pool
             all 4 cells of 'I' — across age, patch — into one number

  = hint: pooling across age, patch is a modelling decision, so state it.
          `prevalence(...)` is the bare single-compartment form and is not
          available inside an expression; write the sum over cells directly:
            • pooled, one reporting rate:
                projected = rho * sum(a in age, sum(p in patch, I[a,p]))
            • per-stratum reporting:
                projected = sum(a in age, sum(p in patch, rho[a] * I[a,p]))
          To report each stratum on its own row, index the stream:
            cases[a in age, p in patch] { … projected = prevalence(I[a,p]) … }
```

On a mixed compartment, say which axis is which:

```text
(the 3 `onset` residence stages pool correctly — it is `age`
 that is a modelling decision)
```

## Tests

- Bare `prevalence` on a user-stratified family → E280 naming the user dims.
- Bare `prevalence` on a `via erlang` compartment → still pools. **Positive
  control, must pass before and after.**
- Mixed → E280 naming only the user dim.
- Single-level dimension → **no** error (pools nothing).
- Indexed stream + bare `prevalence(I)` → E280 (the hole).
- Indexed stream + `prevalence(I[a])` → ok.
- `prevalence(E[child])` on mixed → `CurrentPopSum` over that stratum's stages,
  asserted on the IR.
- Partial projection index → E287, not E503 `'?'`.
- `prevalence(E)` on `hyper_erlang` → pools branch stages (gh#487).
- gh#460's E237 staged branch still fires after dropping the `"__"` sniff.
- The `sum(a in age, I[a])` migration compiles and yields the same numbers.

## Decisions taken

1. **Field on `stratify_decl`, not the IR** — needed during expansion only.
   Construction sites are `parser.mly:1305` and `expander.ml:1458`; verified
   exhaustively, those are the only two in the tree. Note `via hyper_erlang`
   creates **no** `stratify_decl` at all — its branch stages are flat
   compartments (`I__fatal__1`, …) — so `stratum_origin` structurally cannot
   answer "is this a residence axis?" for hyper-staged compartments, and
   `reportable_dims` returns `[]` for them. §2's cell-counting rule needs to say
   what it does there; it currently does not.
2. **Gate on reportable cells pooled (> 1), not on dimensions and not on stream
   indexing.** Fixes the single-level over-fire and the indexed-stream hole in
   one rule. Counting cells rather than dimensions is right, and it handles
   `sonly`-restricted stratification correctly for free. **What is unresolved is
   where the rule is enforced** — see "The gate as specified does not hold".
3. **`prevalence()` stays head-position sugar.** Migration drops the wrapper.
   The arity change from the first draft is dropped as unnecessary.
4. **Mixed compartments gate on the user dimension**; stages pool.
5. **Hand-rolled staged residence breaks**; no `kind = stages` keyword. Zero
   real models affected; the two tests migrate to `via erlang`.
6. **gh#487 folds in** — same dispatch, same catch-all, one fix.
7. **Keep E280**, reworded; partial projection index becomes E287.
8. **The `incidence` gate moves to the same rule**, closing its indexed-stream
   hole. The sweep is done: zero affected models across seven repos (see
   Migration).

## Still open

Each of these must be closed before this ships as a spec.

1. **Where the gate is enforced** — lint at the resolved IR, or a semantic gate
   with an IR-distinguishable escape. The blocking decision.
2. **What `reportable_dims` means for a `via hyper_erlang` compartment**, which
   has no stage `stratify_decl` to filter (Decision 1).
3. **What E287 prints on a mixed compartment**, given its message and hint are
   built from the raw dimension vector and would name `__onset_stage` (§3).
4. **Whether `stratum_origin` is the right concept or just the available one.**
   Reportability is a property of the _axis_ — could a data column carry it? —
   not of who constructed it. Under Decision 5 a hand-rolled residence chain and
   a `via erlang` one produce, by the repo's own assertion in
   `ocaml/golden/seir_age_erlang_via.camdl`, IR equal modulo stage names, yet
   get different diagnostics. "Zero real models affected" is a sound reason to
   ship the mechanism; it is not a reason to state "no `kind = stages` keyword"
   as settled. Either name it a deferral with a tracked follow-up, or decide it.

Two follow-ups this proposal should file rather than fold in: per-stratum
reporting into a _single pooled column_ has no compiling form (only per-stratum
_rows_ do), and `quantities {}` pools stratified compartments silently (`max(I)`
on an age-stratified `I` reduces over the pooled total, which is not the peak of
any stratum).
