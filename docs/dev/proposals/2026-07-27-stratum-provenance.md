# Stratum provenance: telling a subpopulation from an integration stage

Date: 2026-07-27 Status: draft — awaiting decision Fixes: gh#478 (part 1),
gh#487 Related: gh#459, gh#333

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

### 3. Fix the `'?'` fall-through, and reuse E287

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
`ProjDerived`. Its match arms at `expander.ml:7006` and `8034` are dead. The
gate goes in the three live arms:

- `expander.ml:7009-7014` — `ProjDerived (EFuncCall ("prevalence", …))`
- `expander.ml:7015-7027` — `ProjDerived (EIdent …)`, the `projected = I` form
- `expander.ml:7028-7047` — `ProjDerived (EIndex …)`

Putting it in the arm named after the feature yields a gate that never fires,
with green tests.

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

Note the incidence half of §2 needs its own sweep for indexed streams with bare
`incidence(...)` before landing; that number is not yet measured.

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
   Construction sites are `parser.mly:1305` and `expander.ml:1458`.
2. **Gate on reportable cells pooled (> 1), not on dimensions and not on stream
   indexing.** Fixes the single-level over-fire and the indexed-stream hole in
   one rule.
3. **`prevalence()` stays head-position sugar.** Migration drops the wrapper.
   The arity change from the first draft is dropped as unnecessary.
4. **Mixed compartments gate on the user dimension**; stages pool.
5. **Hand-rolled staged residence breaks**; no `kind = stages` keyword. Zero
   real models affected; the two tests migrate to `via erlang`.
6. **gh#487 folds in** — same dispatch, same catch-all, one fix.
7. **Keep E280**, reworded; partial projection index becomes E287.
8. **The `incidence` gate moves to the same rule**, closing its indexed-stream
   hole. Requires a sweep for affected models before landing.
