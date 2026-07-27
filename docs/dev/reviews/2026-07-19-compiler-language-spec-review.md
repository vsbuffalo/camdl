---
status: triaged; all 6 fixed
date: 2026-07-19
kind: compiler/spec review
scope: OCaml compiler against `docs/camdl-language-spec.md`
reviewer: Codex
methodology: static audit of parser/AST/expander/validator plus targeted `camdlc check` repros; Rust runtime reviewed only where needed to compare validator coverage
counts: 4 High / 2 Medium + 4 maintainability notes
triage: all six re-reproduced against 84c3341e; filed gh#459-465; 3/4/6 in PR#477, 5 in PR#479, 2 in PR#485; only 1 (gh#459) remains
---

## Triage (2026-07-24, against `84c3341e`)

Every finding was re-run as a `camdlc check` repro. **All six reproduced; none
had been fixed.** Findings 3, 4 and 6 were fixed in PR#477, finding 5 in PR#479,
and finding 2 in PR#485. **Only finding 1 (gh#459) remains open**, now labelled
`blocker` after a silent-misbind repro was added to it.

| # | Issue                                                   | Title                                      | State                |
| - | ------------------------------------------------------- | ------------------------------------------ | -------------------- |
| 1 | [gh#459](https://github.com/vsbuffalo/camdl/issues/459) | Named indexing resolved positionally       | open — see caveat §1 |
| 2 | [gh#460](https://github.com/vsbuffalo/camdl/issues/460) | Bare stratified `transfer` rejected (E264) | **fixed** (PR#485)   |
| 3 | [gh#461](https://github.com/vsbuffalo/camdl/issues/461) | `set`/`add` action targets unvalidated     | **fixed** (PR#477)   |
| 4 | [gh#462](https://github.com/vsbuffalo/camdl/issues/462) | Event/reactive `where` guards skip E217    | **fixed** (PR#477)   |
| 5 | [gh#463](https://github.com/vsbuffalo/camdl/issues/463) | `hyper_erlang` rewrite misses action exprs | **fixed** (PR#479)   |
| 6 | [gh#464](https://github.com/vsbuffalo/camdl/issues/464) | `time_unit` accepts non-time units         | **fixed** (PR#477)   |

### One sub-claim did not survive verification

**Finding 1, "Additional drift" — NOT A DEFECT.** The claim that
`S[patch = north]` in a rate expression should pin the named dimension and sum
over the rest is a misread of the spec. `docs/camdl-language-spec.md:815-820`
specifies E287 for a partial compartment index in so many words: "a partial
index has no defined cell; the bare name S sums, but `S[child]` neither sums nor
picks a cell". The compiler's E287 is correct. Marked inline at the finding.

Everything else in the review verified. In particular both `Fix` sections that
prescribe two-layer work were **already complete and correct** as written —
finding 3 names `ocaml/lib/ir/validate.ml` alongside the expander change, and
finding 5 names `ATransfer` `fraction`/`count` alongside `ASet`/`AAdd`. Only
finding 5's _repro_ is narrower than its defect: it shows `add`, but the
`count =` kwarg fails identically.

### What the misread exposed

Chasing that sub-claim surfaced a genuine doc-vs-doc bug the review did not
name: §12.1 line 2517 describes a named index as pinning "and summing over the
rest (order-independent)", while line 2578 of the same section says "A partial
index does not marginalize". Both cannot hold. The code sides with 2578. Filed
as [gh#465](https://github.com/vsbuffalo/camdl/issues/465) — worth fixing before
gh#459, whose resolver has to implement whichever rule the spec settles on.

### Maintainability notes

A–D all still hold: `expander.ml` is 9158 lines, `index_item_to_str` has 18 call
sites, `eval_guard` still returns a bare `bool` (`expander.ml:3076`). Note D
(source locations on action errors) is partly addressed — the E265 and guard
diagnostics added in PR#477 are located, and `check_guards` now supplies a
`~loc` on the intervention arm, which it previously omitted.

# OCaml compiler review against language spec - 2026-07-19

This review checks the current compiler against `docs/camdl-language-spec.md` as
the contract. It focuses on observable spec drift, correctness bugs, and a few
places where the OCaml would be clearer or less error-prone with a more typed
implementation shape.

I ran targeted `dune exec bin/camdlc.exe -- check` repros for the main issues. I
did not run the full test suite.

---

## High findings

### 1. Named indexing is accepted, then resolved as positional suffixes

**Location** - `ocaml/lib/compiler/parser.mly:695-697`,
`ocaml/lib/compiler/ast.ml:59-62`, `ocaml/lib/compiler/expander.ml:2350-2355`,
`ocaml/lib/compiler/expander.ml:3189-3194`,
`ocaml/lib/compiler/expander.ml:3212-3223`,
`ocaml/lib/compiler/expander.ml:3259-3278`,
`ocaml/lib/compiler/expander.ml:3314-3316`,
`ocaml/lib/compiler/expander.ml:6725-6727`

**Spec** - `docs/camdl-language-spec.md:816-827`, `2468-2475`, `2536-2545`

**Defect** - The parser and AST preserve named index labels, but most lowering
paths erase them immediately:

```ocaml
| INamed (_, EIdent (s, _)) -> ...
```

The downstream code then concatenates index values in source order. This means
`S[patch = north, age = child]` is lowered as if the user wrote
`S[north, child]`, producing `S_north_child` instead of resolving by declared
dimensions to `S_child_north`.

The same shortcut is used for table lookups, indexed lets, forcings, parameters,
compartments, and observation projections. The observation path is especially
visible because the spec recommends forms such as
`incidence(infection[patch = p])`.

**Observed repro** - With dimensions declared as `[age, patch]`:

```camdl
transitions {
  inf : S[child, north] --> I[child, north]
    @ beta * S[patch = north, age = child]
}
```

fails with:

```text
error[E100]: undeclared name 'S_north_child'
```

The control case `S[age = child, patch = north]` compiles, proving the syntax is
recognized but treated positionally.

**Additional drift** - The spec says partial named selectors pin the named
dimension and sum over the rest:

```camdl
S[patch = north]
incidence(infection[patch = p])
```

The current expression resolver rejects this with E287 for compartments because
it treats it as a partial positional index, not a named projection/marginal.

> **[TRIAGE 2026-07-24: NOT A DEFECT — this sub-claim is withdrawn.]** The spec
> does not say this for compartments. `docs/camdl-language-spec.md:815-820`
> specifies E287 for a partial index explicitly: "a partial index has no defined
> cell; the bare name S sums, but `S[child]` neither sums nor picks a cell." The
> compiler is correct to reject `S[patch = north]`. The premise came from §12.1
> line 2517, which describes a named index as summing over the rest — and which
> line 2578 of the same section contradicts. That contradiction is a real bug in
> the spec, filed as gh#465, but it is not a compiler defect. **The rest of
> finding 1 (order-dependence) stands and is confirmed** — see the verification
> note below.

**Why it matters** - Named indexing is the safe form the spec tells users to
prefer for multidimensional models. The compiler currently rejects valid models
and can misbind any case where a reversed named selector happens to form an
existing concrete suffix.

**[TRIAGE 2026-07-24: CONFIRMED, and it reaches further than the repro shows.]**
The order-dependence also breaks the **observation** path, which is precisely
where the spec pushes named indexing hardest (line 2590: prefer named because it
"survives a later reordering of the dimension declarations"). On a
`[age, patch]` family:

```text
incidence(infection[patch = north, age = child])
  → error[E507]: unknown transition referenced in observation: 'infection_north_child'
incidence(infection[age = child, patch = north])
  → no errors
```

The spec promises order-independence in three separate places — lines 827, 2517
and 2590 — and the compiler delivers the opposite at every one. Tracked as
gh#459.

**Fix** - Replace `index_item_to_str` for semantic resolution with a typed
selector resolver:

```ocaml
type resolved_selector = {
  dim : string;
  level : string;
}
```

That resolver should know the referenced object's declared dimension vector, map
named indices by dimension, map positional indices by position, validate level
membership in the selected dimension, and lower omitted dimensions according to
the context: exact cell for fully indexed references, sums for
projection/marginal contexts, arity errors where the spec requires full
coordinates.

### 2. Bare stratified `transfer(from = S, to = V)` is rejected instead of expanded

> **[TRIAGE 2026-07-24: CONFIRMED — FIXED in PR#485, gh#460.]** Reproduced
> verbatim. The fix expands per stratum, pairing by _declared dimensions_;
> review found and closed three silent-wrong shapes it initially introduced
> (cross-dimension pairing on shared level names, single-cell families escaping
> the check, and per-instance fan-out inside an indexed family — E239).

**Location** - `ocaml/lib/compiler/expander.ml:5310-5340`,
`ocaml/lib/compiler/expander.ml:6110-6120`

**Spec** - `docs/camdl-language-spec.md:2738-2745`,
`docs/camdl-language-spec.md:4840-4858`, `docs/camdl-language-spec.md:4970-4990`

**Defect** - The spec requires `transfer(from = S, to = V)` on stratified
compartments to expand over all matching strata, emitting one atomic transfer
per stratum. The compiler routes `from` and `to` through `resolve_comp_name`,
which accepts only a single `Ir.Pop`. A bare stratified compartment resolves to
`Ir.PopSum`, so the valid spec form is rejected with E264.

**Observed repro**:

```camdl
dimensions { age = [child, adult] }
compartments { S, V }
stratify(by = age)
parameters { cov : probability }
transitions {}
interventions {
  vacc : transfer(fraction = cov, from = S, to = V) at [1]
}
init { S[a in age] = 100 }
simulate { from = 0 'days to = 10 'days }
```

fails with:

```text
error[E264]: expected a bare compartment name, got a sum of populations (PopSum)
```

**Why it matters** - This is a central intervention idiom in the spec and in
model examples. Users must write one explicit indexed intervention per stratum
today, or manually index the intervention family, despite the spec promising the
bare form.

**Fix** - Lower action endpoint references with an action-specific resolver
instead of `resolve_comp_name`. It should return either a single concrete
compartment or a list of matched concrete compartments. For `transfer`, both
endpoints must expand to the same stratum shape, then emit paired transfers over
that shape.

### 3. Action target validation is incomplete for `set` and absent for `add`

> **[TRIAGE 2026-07-24: CONFIRMED — FIXED in PR#477, gh#461.]** Both repros
> reproduced (`I = 0` on a stratified family and `add(Z, 1)` both exited 0). The
> `Fix` below was complete as written: the OCaml validator did have zero
> intervention coverage while the Rust one checked all four action targets, and
> both layers were implemented.

**Location** - `ocaml/lib/compiler/expander.ml:6122-6141`,
`ocaml/lib/ir/validate.ml:108-212`

**Spec** - `docs/camdl-language-spec.md:2740-2745`,
`docs/camdl-language-spec.md:2893-2900`, `docs/camdl-language-spec.md:5015`

**Defect** - `set` is supposed to target a single expanded compartment name, and
the compiler is supposed to reject unknown targets with E265. The current check
accepts either an expanded compartment or the base surface compartment:

```ocaml
Hashtbl.mem ctx.expanded_comp_tbl concrete || Hashtbl.mem ctx.comp_tbl comp
```

That lets a bare stratified `I = 0` pass even though `I` is not a single
expanded compartment. `add` is worse: `AAdd` constructs an `Ir.AddAction`
without any target validation at all.

**Observed repros**:

```camdl
dimensions { age = [child, adult] }
compartments { S, I }
stratify(by = age)
parameters { x : rate }
transitions {}
interventions {
  zap : { I = 0 at = [1] }
}
init { S[a in age] = 100 }
simulate { from = 0 'days to = 10 'days }
```

compiled with only unused-compartment warnings.

```camdl
time_unit = 'days
compartments { S }
parameters { x : rate }
transitions {}
interventions {
  seed : add(Z, 1) at [1]
}
init { S = 100 }
simulate { from = 0 'days to = 10 'days }
```

compiled with no errors.

**Why it matters** - `camdlc check` can approve DSL that emits IR with dangling
action targets. The Rust IR validator has action-target checks, but the OCaml
validator does not, and the compiler CLI should reject invalid DSL before
runtime.

**Fix** - Validate every `Set`/`AddAction`/`Transfer` target in
`ocaml/lib/ir/validate.ml`, matching the Rust validator. In the expander, make
`set` require `expanded_comp_tbl concrete` exactly; do not accept a stratified
base name as a concrete target.

### 4. Event and reactive `where` guards skip compile-time guard validation

> **[TRIAGE 2026-07-24: CONFIRMED — FIXED in PR#477, gh#462.]** Reproduced; the
> event-block model compiled clean while the identical intervention-block guard
> errored E217. `check_guards` now walks all four guard-carrying surfaces.

**Location** - `ocaml/lib/compiler/parser.mly:902-935`,
`ocaml/lib/compiler/parser.mly:948-970`, `ocaml/lib/compiler/ast.ml:404-413`,
`ocaml/lib/compiler/expander.ml:3055-3063`,
`ocaml/lib/compiler/expander.ml:4005-4058`,
`ocaml/lib/compiler/expander.ml:6167-6170`,
`ocaml/lib/compiler/expander.ml:6408-6410`

**Spec** - `docs/camdl-language-spec.md:1735-1765`,
`docs/camdl-language-spec.md:2890-2891`

**Defect** - Events share the intervention grammar and support indexed `where`
guards. Reactive policies also carry `rxguard` and evaluate it during expansion.
But `check_guards` only walks transitions and `interv_decls`; it does not check
`event_decls` or `reactive_decls`.

That matters because `eval_guard` substitutes loop variables from the expansion
environment, then treats every other identifier as a literal string. The E217
pass is the only thing preventing a parameter or compartment name in a
compile-time guard from silently changing which instances are emitted.

**Observed repro**:

```camdl
dimensions { patch = [north, south] }
compartments { S, I }
stratify(by = patch)
parameters { beta : rate  keep : probability }
transitions {}
events {
  seed[p in patch] : transfer(count = 1, from = S[p], to = I[p])
    at [1] where p != keep
}
init { S[p in patch] = 100 }
simulate { from = 0 'days to = 10 'days }
```

compiled with no errors. The analogous transition and intervention versions
correctly emit:

```text
error[E217]: ... where guard references 'keep', which is a parameter
```

Reactive policies show the same gap when `where p != keep` is added after the
policy body.

**Why it matters** - A model can compile while expanding the wrong support set.
For example, `p != keep` is always true unless a dimension level is literally
named `keep`.

**Fix** - Extend `check_guards` to walk `ctx.event_decls` and
`ctx.reactive_decls`, using `ivloc`/`rxloc` and the corresponding index binders.
Also consider making `eval_guard` return an error-aware result instead of a bare
`bool`, so callers cannot bypass validation accidentally.

---

## Medium findings

### 5. `hyper_erlang` bare-source rewrite misses action expressions

> **[TRIAGE 2026-07-24: CONFIRMED — open, gh#463.]** Reproduced
> (`error[E100]: undeclared name 'I'`). The `Fix` below is correctly scoped;
> only the repro is narrower than the defect — `transfer(count = I)` fails
> identically to the `add(S, I)` shown.

**Location** - `ocaml/lib/compiler/expander.ml:1236-1256`,
`ocaml/lib/compiler/expander.ml:1752-1794`,
`ocaml/lib/compiler/expander.ml:6074-6141`

**Spec** - `docs/camdl-language-spec.md:1878-1886`,
`docs/camdl-language-spec.md:1950-1956`, `docs/camdl-language-spec.md:2893-2900`

**Defect** - `hyper_erlang` removes the base source compartment and creates flat
branch-stage compartments such as `I__a__1`. The expander has `sum_hyper_refs`
to rewrite bare `I` references into an explicit sum over all branch stages. The
pass applies that rewrite to transition rates, lets, init, balance, and
observations, but not to scheduled event/intervention or reactive action
expressions.

**Observed repro**:

```camdl
compartments { S, I, R }
parameters {
  beta : rate
  p : probability
  tau_a : duration
  tau_b : duration
}
transitions {
  infection : S --> I @ beta * S
  clearance : I --> R via hyper_erlang(
    branch(label = a, weight = p, stages = 1, mean = tau_a),
    branch(label = b, stages = 1, mean = tau_b)
  )
}
interventions {
  pulse : add(S, I) at [1]
}
init { S = 100  I = 10 }
simulate { from = 0 'days to = 10 'days }
```

fails with:

```text
error[E100]: undeclared name 'I'
```

**Why it matters** - The spec calls out force-of-infection examples, but action
operands are ordinary expressions. After `hyper_erlang` lowering, a valid
pre-macro expression can become invalid because the rewrite is not applied to
the whole surface AST.

**Fix** - Apply `sum_hyper_refs` to `ASet` and `AAdd` expressions and to
`ATransfer` `fraction`/`count` expressions. If action endpoint expansion is
implemented for finding 2, it should share the same lowered-source awareness.

### 6. `time_unit` accepts non-time units

> **[TRIAGE 2026-07-24: CONFIRMED — FIXED in PR#477, gh#464.]** Both `'count`
> and `'per_day` compiled clean, and forcing a conversion produced the predicted
> bare `E001 Invalid_argument`. Now a located **E228** at the declaration.

**Location** - `ocaml/lib/compiler/parser.mly:231-233`,
`ocaml/lib/compiler/parser.mly:291-302`,
`ocaml/lib/compiler/expander.ml:1937-1946`,
`ocaml/lib/compiler/expander.ml:7875-7890`

**Spec** - `docs/camdl-language-spec.md:91-99`,
`docs/camdl-language-spec.md:115-119`

**Defect** - The parser accepts any `unit_lit` for `time_unit`, including
`'per_day`, `'count`, and `'ratio`. Later validation only rejects
`'months`/`'years` in anchored mode. As a result:

```camdl
time_unit = 'count
compartments { S }
parameters { x : rate }
transitions {}
init { S = 100 }
simulate { from = 0 to = 10 }
```

compiles with no errors. If a duration literal later forces conversion under
`time_unit = 'count`, the compiler degrades to a generic E001:

```text
error[E001]: Invalid_argument("days_per: non-time unit has no time scale")
```

`time_unit = 'per_day` also compiles with plain numeric simulate times, even
though a rate unit is not a duration unit.

**Why it matters** - The IR can carry a nonsensical canonical time unit until a
later path happens to call `days_per`. The comment in `parse_date_to_float`
still says non-time units are unreachable because `time_unit` is validated, but
the current parser/expander do not enforce that.

**Fix** - Validate top-level `time_unit` immediately after declaration
collection. Accept only duration units (`'days`, `'weeks`, `'months`, `'years`)
or, if anchored dates remain restricted, perhaps only exact units in anchored
mode. Reject rate, count, and ratio units with a located E2xx/E3xx diagnostic
instead of allowing a later `Invalid_argument`.

---

## Maintainability notes

### A. Centralize index resolution

Many surfaces reimplement the same pattern:

```ocaml
List.map (index_item_to_str env) items
String.concat "_" (base :: idx_vals)
```

This is why named-index labels disappear in so many places. A single
dimension-aware resolver would remove duplication and make the intended
semantics explicit at the type level.

### B. Split `expander.ml` by semantic domain

`ocaml/lib/compiler/expander.ml` is over 9k lines and currently owns declaration
collection, stratification, `via` lowering, expression resolution, parameters,
tables, forcings, interventions/events/reactive policies, observations,
generated quantities, scenarios, time typing, and lints. The file is navigable
but fragile: cross-cutting rewrites such as `hyper_erlang` can easily miss one
AST surface.

Good split candidates:

- `Index_resolver`
- `Action_lowering`
- `Observation_lowering`
- `Via_lowering`
- `Scenario_validation`
- `Surface_time_checks`

### C. Make guard evaluation impossible before validation

`eval_guard : context -> env -> guard -> bool` is convenient but unsafe: it
silently interprets unknown identifiers as literal strings. Returning a
diagnostic-aware result, or requiring a prevalidated guard token, would prevent
future event/reactive-style omissions.

### D. Improve source locations on action errors

Several action diagnostics still use `Diagnostics.no_loc`, notably transfer
kwarg and endpoint errors. The enclosing intervention/event/reactive declaration
has a loc, and the parser already carries enough structure to improve this over
time. Better locations matter because action declarations are often dense and
multi-action blocks can fail in more than one place.

---

## Suggested implementation order

1. Add OCaml IR validation for all intervention/action targets. This is small
   and prevents invalid IR from passing `camdlc check`.
2. Extend `check_guards` to events and reactive policies.
3. Introduce a dimension-aware index resolver and migrate the named-index
   call-sites incrementally.
4. Implement stratified action endpoint expansion for `transfer`.
5. Apply `hyper_erlang` rewrites to action operands.
6. Add top-level `time_unit` validation.

Each item should get a minimal regression test in `ocaml/test/test_compiler.ml`.
The named-index resolver deserves multidimensional tests for reversed named
order and partial named projection, not just single-dimension controls.
