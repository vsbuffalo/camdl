# Multi-index parameters: `mu[village, season]`

Date: 2026-07-12 Status: Draft Area: compiler (parser, expander), DSL surface
Motivates: garki vector cell model (village × season × species design matrix);
panel/design-matrix models generally

## Summary

Extend indexed parameter _declarations_ from one dimension to N:

```camdl
parameters {
  mu[village, season]          : rate                # → mu_kwaru_wet, mu_kwaru_dry, …
  m[village, season, species]  : positive 'ratio     # → m_kwaru_wet_gam, …
}
let C[v in village, s in season] = m[v,s] * a * fcap[v,s]
```

**Compiler-frontend-only, backward-compatible.** The IR sees only expanded flat
scalar params (`mu_kwaru_wet`), so **no `ir/VERSION` bump, no golden format
change, no Rust change**. A 1-D declaration is the 1-element case, so every
existing model and golden is untouched.

The 1-D restriction lives entirely in the parameter _declaration_. The _use_
side (resolving `mu[v,s]` to `Param("mu_kwaru_wet")`, arity checking, obs
multi-column matching, scenario/init refs, `--param-vec`, collision detection,
CAS, dimcheck) is already N-dim — verified against code (see "What already
works").

## What already works (verified — do not touch)

- **Use-site resolution** — `expander.ml:3260–3276`. Arity-checks against the
  full `pdims` list (`indexed_param_dims`, `:3103`) and mangles multi-item
  indices via `String.concat "_"`. `mu[v,s]` already resolves.
- **N-dim name expansion** — `expand_indexed_decl_names` (`:2020`) already
  produces the cartesian product `<base>_<l1>_<l2>_…`.
- **`pdims` is already `string list`** — `ast.ml:140`.
- **Collision detection** — `check_declaration_names` (`:2054`, emits **E278**)
  already calls `expand_indexed_decl_names` over the full `pdims` list,
  **independent of** the lookup table. Multi-index collisions (e.g.
  `m[village,
  species]` shadowing a compartment `m_kwaru_gam`) are caught the
  instant the parser emits multi-dim `pdims` — **no collision-specific code
  change needed**, only a test.
- **Scenario `set/scale`** — the scenario parser already mangles N-dim keys
  (`parser.mly:1448`, `String.concat "_"`); `expand_scenarios` (`:8390`)
  validates the mangled key against a name set that includes
  `expanded_param_tbl` (site 3 below), which runs first (`build_lookup_tables`
  at `:8983` precedes `expand_scenarios` at `:9050`).
- **init blocks, `--param-vec`, hierarchical priors, run_id/CAS** — all
  dim-agnostic (route through use-site resolution, opaque-suffix matching, or
  hash by expanded name). Confirmed no single-dim assumption.
- **Multi-column observations** — proven in production: the garki ladder model's
  `prevalence[v in village, a in age]` matches data on `village` _and_ `age`.
- **Dimcheck** — each expanded cell is a scalar of the declared kind/dim.

## The change

### Site 1 — Grammar (`parser.mly:344–357`, 4 `param_decl` variants)

```
- name = IDENT LBRACKET dim = IDENT RBRACKET …  { PIndexed { pdims = [dim]; … } }
+ name = IDENT LBRACKET dims = separated_nonempty_list(COMMA, IDENT) RBRACKET …
+                                                { PIndexed { pdims = dims; … } }
```

`separated_nonempty_list(COMMA, IDENT)` is used elsewhere in a bare-IDENT
bracket context (`parser.mly:704`, a guard-atom table lookup), so it composes
here — **but adding a comma-list to the param-decl position can introduce a
menhir shift/reduce conflict**, so a clean `dune build` (menhir reports **no new
conflicts**) is a hard gate for this site.

### Site 2 — Declaration expansion (`expander.ml:4795 + 4814–4833`)

Replace the single-dim arm and the (now-dead, parser-unreachable) E274 stub with
one N-dim arm. Bounds/prior/kind/dim resolution is unchanged (per-declaration,
shared across cells); only the name list generalizes, with two new guards:

```ocaml
| PIndexed { pname; pdims; pbounds; pkind; pdim = pdim_ann; punit; pprior; ploc; _ } ->
    (* guard 1 (E331): duplicate index dimension — no dedup in the cartesian
       product, so `mu[village, village]` would silently emit mu_kwaru_kwaru … *)
    if List.length pdims <> List.length (List.sort_uniq compare pdims) then
      Diagnostics.error … ~code:"E331"
        ~message:(Printf.sprintf "indexed parameter '%s' repeats a dimension" pname)
        ~hint:"each index axis must be a distinct dimension" …;
    (* guard 2 (E330): unknown or empty index dimension — dim_values / the
       cartesian both silently return [] on these, producing zero cells *)
    List.iter (fun d ->
      match List.assoc_opt d ctx.dim_registry with
      | None    -> Diagnostics.error … ~code:"E330"
          ~message:(Printf.sprintf "indexed parameter '%s': unknown dimension '%s'" pname d) …
      | Some [] -> Diagnostics.error … ~code:"E330"
          ~message:(Printf.sprintf "indexed parameter '%s': dimension '%s' has no levels" pname d) …
      | Some _  -> ()) pdims;
    let bounds = resolve_bounds ctx pbounds in
    let pk = Some (ir_param_kind_of_ast pkind) in
    let resolved_dim = resolve_param_dim ctx ~loc ~pname pkind pdim_ann punit in
    let (prior, hierarchical) = (* … unchanged … *) in
    List.map (fun nm ->
      { Ir.name = nm;
        Ir.value = mk_estimated_or_required ~bounds ~prior ~hierarchical;
        Ir.param_kind = pk; Ir.param_dim = resolved_dim })
      (expand_indexed_decl_names ctx pname pdims)
```

Two **fresh** error codes (E2xx is saturated; E330/E331 verified unused):

- **E330** — index dimension unknown or empty. This also fixes a _latent 1-D
  bug_: today `mu[nonesuch]` (unknown dim) silently produces a dangling
  `Param("mu_…")` with no diagnostic (`dim_values` returns `[]` mutely). The
  guard closes both 1-D and N-D.
- **E331** — duplicate index dimension.

E274 is **not** reused (it already carries three unrelated meanings —
`parser.mly:846`, `expander.ml:6309`, `:6406`, tested at
`test_compiler.ml:8692/
8718`); the dead E274 param stub is simply deleted with
the merged arm.

### Site 3 — Lookup table (`expander.ml:2156`)

```ocaml
- | PIndexed { pname; pdims = [dim]; _ } ->
-     let vals = … in List.iter (fun v -> Hashtbl.replace ept (pname ^ "_" ^ v) ()) vals
+ | PIndexed { pname; pdims; _ } ->
+     List.iter (fun nm -> Hashtbl.replace ept nm ()) (expand_indexed_decl_names ctx pname pdims)
```

### Site 4 — Stale comment (`expander.ml:8318–8322`)

The comment there states the expanded-name set is "populated only for single-dim
indexed params today." After site 3 that is false. Update/delete it
(comment-only; the scenario `parameter_names` table auto-benefits).

## Decisions (resolved)

- **Naming:** `<base>_<l1>_<l2>_…` in declaration-dim order, via the existing
  mangling. Consistent with the 1-D rule in spec §4.3.
- **Index order is positional and must match the declaration.** `mu[s,v]` when
  declared `mu[village, season]` mangles to `mu_wet_kwaru`, undeclared → the
  existing **E100** undeclared-name error fires (verified). A dedicated "did you
  mean `mu[village, season]`?" hint is a follow-up, not a blocker.
- **Bounds and priors are shared across all cells** (declared once, replicated —
  as 1-D does). Per-cell priors are out of scope.
- **Each index dim must be a declared `stratify` dimension** (unchanged from
  1-D). For a compartment-free regression model this means
  `stratify(by = season)` with no compartments — verified to compile clean
  (empty `compartments { }` + two `stratify` + a multi-index `let` → 0 errors).
  Decoupling param-indexing from `stratify` is a possible future cleanup, not in
  this change.

## Test plan (red → green; each asserted to FAIL on current code first)

OCaml compiler tests (`ocaml/test/test_compiler.ml`):

1. `mu[village, season] : rate ~ …` → 4 cells, correct mangled names + declared
   kind/bounds/prior on each.
2. 3-dim `m[village, season, species]` → 8 cells.
3. Use in a `let`: `let C[v in village, s in season] = m[v,s] * 2.0` resolves
   each `m[v,s]` to the right `Param`.
4. Multi-column obs over the same axes compiles + matches (regression guard on
   the already-working path).
5. dimcheck: a dimensional error in a multi-index-param expression is caught
   with the cell's dimension.
6. **E331** duplicate dims: `mu[village, village]` errors (this case is
   _unsatisfiable_ without guard 1 — it currently would emit `mu_kwaru_kwaru`
   silently; the guard is what makes the test greenable).
7. **E330** unknown dim: `mu[village, nonesuch]` errors. And empty-levels: a
   registered dim with zero levels errors (not a silent zero-cell no-op).
8. Use-arity mismatch: `mu[v]` when declared `mu[village, season]` errors naming
   the declared arity (existing `check_index_arity`).
9. Collision: a multi-index declaration colliding with a stratified compartment
   errors (**E278**, existing machinery — test only).
10. **Regression pins:** the reactive **E274** tests (`8692`, `8718`) still pass
    (proves E330/E331 don't cross the E274 meanings); existing goldens
    byte-identical (1-D params emit identical IR).

**Gate:** a clean `dune build` with no new menhir conflicts (site 1).

Golden fixture: `ocaml/golden/multi_index_beta.camdl` — a region × age SIR whose
transmission is a `beta[region, age]` design matrix (4 cells) — pins the feature
end-to-end. It's discovered by `smoke_all_golden` (all backends, invariants) and
by `gate_trajectory_baseline` (byte-identical trajectory + ODE-state hashes,
captured via `CAMDL_CAPTURE_BASELINE=1`). Regenerating all goldens leaves every
existing `ir.json` byte-identical (frontend-only backward-compat check).

## Docs

- **Spec §4.3:** "declared with a single dimension index" → N dimensions; add
  the `mu[village, season]` example + the ordering/mangling rule.
- **`--param-vec` (§21 / §4.3):** note the key column for a multi-dim param
  holds the pre-joined suffix (`kwaru_wet`), not separate dim columns.
- **`docs/language-changes.md`:** newest-first entry — indexed params now accept
  multiple dimensions (widening; no migration needed).
- **`docs/dev/warning-catalog.md`:** add E330, E331.

## Out of scope (already done)

- **Count observations in compartment-free models** (garki REQUEST 2): **already
  fixed** by `817705ec` (indexed-let-in-`projected`). Verified on `ce77349d`: a
  `compartments { }` model with an indexed `neg_binomial` obs compiles clean.
  The projection resolver is family-independent; count-as-incidence
  (`projected = incidence(...)`) is a different, untouched arm. garki should
  re-test on `ce77349d`.

## Rollout

Frontend-only, no IR bump. `make test` (OCaml + integration) is the gate; the
golden diff is additive (new fixture only). Pure surface widening; no feature
flag.
