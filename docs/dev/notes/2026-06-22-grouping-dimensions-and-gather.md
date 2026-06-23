---
date: 2026-06-22
status: DRAFT — design sketch; open questions unresolved, NOT a shippable spec
  (per CLAUDE.md "a shipped proposal has no open questions"). Parked in notes/;
  graduates to proposals/ once §"Open questions" is decided.
related:
  - ../proposals/2026-06-04-hierarchical-priors-gate3-pgas-nuts.md # group-level pooling this unblocks for stratified models
  - hierarchical-priors-gate2-plan.md # the Gate 1/2/3 framing for hierarchical priors
area: DSL / dimensions / parameter indexing / expander
---

# Grouping dimensions and the gather: group-level parameters without the product

## Problem

A spatial model stratified by district wants to attach a **province-level**
estimated parameter to each district — e.g. a province-specific turnover time
`T`, four estimated scalars `T_E, T_N, T_S, T_W`, broadcast to the ~14 districts
by each district's province membership. There is no native way to write that
broadcast. The working idiom today is a one-hot **design matrix**: one 0/1 table
per province, plus a dot product.

```camdl
dimensions { patch = read("zones.tsv", column = "patch") }
stratify(by = patch)

tables {                                   # one indicator column per province
  isE : patch = read("isE.tsv")
  isN : patch = read("isN.tsv")
  isS : patch = read("isS.tsv")
  isW : patch = read("isW.tsv")
}

parameters { T_E : real …  T_N : real …  T_S : real …  T_W : real … }

let Tr[p in patch] = T_E*isE[p] + T_N*isN[p] + T_S*isS[p] + T_W*isW[p]
```

This is correct and zero-overhead — it lowers to the same flat-scalar arithmetic
any hand-written formula would — but it is the design-matrix encoding of a
**gather** the language will not let you write directly:
`Tr[p] = T[province(p)]`, where `province : patch → {E,N,S,W}` is a partition
(each district in exactly one province). The boilerplate grows with the number
of groups, the membership is spread across N indicator files, and — the
load-bearing cost — there is no way to put a **shared prior** over the four
`T`s, because they are four unrelated scalar names rather than one group-indexed
parameter.

## Why the obvious alternatives don't work (verified)

Two natural-looking forms are both blocked, by design, in the current grammar:

1. **A `province` stratify dimension** (`stratify(by = patch, province)`) forces
   the Cartesian product: every compartment gains a `[patch, province]` index
   (spec §5), so a 14-district × 4-province model has 56 cells, 42 of them
   structurally empty (a district is in one province). `province` here is a
   **quotient** of `patch`, not an orthogonal axis; product stratification
   models the wrong thing. There is no partition/grouping concept to express it
   otherwise — grepping the compiler and IR for `group` / `partition` / `subdim`
   / `parent_dim` returns nothing.

2. **A computed gather index** `T[province[p]]` is rejected by the indexer. A
   parameter index "must refer to a declared `stratify` dimension"
   (`docs/camdl-language-spec.md` §4.3, line 662), and inside `[...]` the
   compiler resolves **only** bound loop variables and literal stratum values —
   "let bindings and other parameters are never checked in index position"
   (§4.3, lines 669–677). The expander confirms it: `index_item_to_str`
   (`ocaml/lib/compiler/expander.ml`) maps a bare identifier (loop var, via the
   substitution env) or a literal to a dimension value and returns `"?"` for any
   other expression, and a table lookup lowers to a **compile-time-constant**
   linear index (`Ir.TableLookup (base, [Ir.Const linear])`). A table value
   (`province[p]`) is neither a loop var nor a literal, so it can never appear
   in index position.

Tables _can_ hold estimated-parameter values
(`T_tbl : province = [T_E, T_N, T_S, T_W]`, spec §6.6), but since you still
cannot index `T_tbl` by `province[p]`, that does not recover the gather. The
one-hot design matrix is therefore the cleanest _current_ idiom — not a hack to
apologise for, but the correct encoding given the indexer's constraints.

## Proposed surface

Add a **grouping dimension** — a coarser dimension declared as a partition of a
finer one via a membership column, that does _not_ enter stratification — and a
**gather** access that resolves a group-indexed parameter at a fine index.

```camdl
dimensions {
  patch    = read("zones.tsv", column = "patch")
  province = group(patch, by = read("zones.tsv", column = "province"))
}
stratify(by = patch)                    # province does NOT stratify compartments

parameters {
  # 4 estimated scalars over the grouping dim; pooling optional via `| province`
  T[province] : real in [0, 350] ~ normal(mu = T_mu, sigma = T_sd) | province
  T_mu : real …
  T_sd : positive …
}

let Tr[p in patch] = T[province @ p]    # gather: each district picks its province's T
```

- `group(patch, by = <column>)` declares `province` as a dimension whose
  elements are the distinct values of the per-`patch` membership column, **and**
  records the `patch → province` map. It is a partition: every `patch` maps to
  exactly one `province`. It is marked a grouping, so `stratify` ignores it (no
  product).
- `T[province]` is a parameter indexed by the grouping dimension — four
  estimated scalars `T_E … T_W`, the existing §4.3 expansion, with no
  compartment expansion.
- `T[province @ p]` is the gather: in a `patch`-indexed expression, select the
  `province`-element for `p`'s group. `@` reads "at"; it is deliberately
  distinct from ordinary value indexing so the membership application is
  explicit, not magic.

### Before / after

The one-hot block above collapses to:

```camdl
dimensions {
  patch    = read("zones.tsv", column = "patch")
  province = group(patch, by = read("zones.tsv", column = "province"))
}
parameters { T[province] : real in [0, 350] }
let Tr[p in patch] = T[province @ p]
```

Four indicator files and a four-term dot product become one membership column
and one gather — and `T` is now a single named, group-indexed parameter that a
prior can pool over.

## Semantics and lowering — front-end sugar only

The decisive property: **this changes nothing below the OCaml front end.** The
gather is resolved at expansion, when `p` is already unrolled to a concrete
district:

```
T[province @ p]   with p = "kailahun", province(kailahun) = "E"
        ⇒  Param("T_E")
```

That is _identical_ to the IR you get by writing the scalar directly, and to
what the one-hot dot product evaluates to — the same flat-scalar mangling §4.3
already performs (`N[patch] → N_urban, N_rural, …`). Consequences:

- **No new IR node.** The gather lowers to an existing `Param` reference; the
  grouping dimension lowers to nothing (it never reaches the IR — it is consumed
  to expand `T[province]` into scalars and to resolve gathers).
- **No Rust, runtime, or inference change.** The simulation and inference stacks
  see the same flat `Param("T_E")` references they see today.
- **autodiff/`rate_grad` unaffected** — it differentiates the same `Param`
  leaves.

The new machinery is entirely in the compiler: the lexer/parser (the `group(…)`
declaration and the `@` gather), the expander (register the grouping dimension +
its membership map; resolve `T[g @ i]` to `Param("T_<group_of_i>")`), and
dimcheck (below). This is a tractable, well-scoped front-end addition rather
than a cross-language schema change.

## dimcheck, validation, error messages

- **Membership is total and single-valued.** Every `patch` element must appear
  in the membership column with exactly one `province`. A missing or duplicated
  district is an error naming the offending row — the membership file is the
  partition's source of truth.
- **The group's element set** is the sorted distinct values of the column; the
  IR-mangled scalar names are `T_<value>` (same rule as §4.3's `N_urban`).
- **Gather type rule.** `T[g @ i]` requires that `g` is a grouping _of_ the
  dimension `i` ranges over. `T[province @ p]` is well-typed because `province`
  groups `patch` and `p : patch`; `T[province @ a]` with `a : age` is an error
  ("`province` is a grouping of `patch`, not `age`").
- **Grouping dims reject `stratify`.** `stratify(by = province)` on a grouping
  dimension is an error pointing at the grouping declaration ("`province` is a
  grouping of `patch`; stratify by `patch`, index parameters/tables by
  `province`"). This is the type-correct refusal of the product the one-hot was
  working around.

Per the project's error-quality bar, each of these names the construct and the
fix, old → new where a migration applies.

## Relationship to hierarchical priors

This is the missing half of the hierarchical-priors work. The pooling grammar
`~ dist(…) | dim` is shipped (Gate 1/2; `parser.mly` `prior_clause` →
`ps_pool_over`), and gradient support for PGAS+NUTS is in flight
([`2026-06-04-hierarchical-priors-gate3-pgas-nuts.md`](2026-06-04-hierarchical-priors-gate3-pgas-nuts.md)).
Today you can partial-pool a parameter over the **stratify** dimension
(`R0[patch] ~ … | patch` — vary `R0` across districts with a shared prior). You
**cannot** pool over a coarser _grouping_ of it (`T[province] ~ … | province`),
because `T[province]` requires `province` to be a stratify dimension — and that
is the product again. The grouping dimension unblocks exactly that:
province-level varying effects with a shared prior, attached to
district-stratified dynamics, no product. The four `T`s in the motivating model
become the natural home for a pooled prior, whether the author keeps them
independent (informative per-province normals) or pools them (`| province`).

## Alternatives considered

- **Status quo (one-hot design matrix).** Correct and zero-overhead, but verbose
  (one file per group), the membership is scattered, and — the real limitation —
  the per-group scalars are unrelated names, so no prior can pool them. Keep it
  working; this RFC is the ergonomic on top.
- **Product stratification by `province`.** Rejected: wrong semantics
  (mostly-empty cells), and it inflates every compartment, transition, and the
  state vector by the group count.
- **General computed indices** (`T_tbl[province[p]]`, lifting §4.3's
  index-namespace rule to allow arbitrary integer-valued index expressions).
  Broader and riskier than needed — it would admit runtime-dependent indices and
  re-open dimensional-soundness questions the current rule closes. The
  membership here is **compile-time-known**, so a scoped compile-time gather is
  strictly smaller and keeps "indices resolve at compile time" intact.
- **A runtime `lookup(map, key)` expression.** Overkill: needs an IR node, a
  runtime evaluator, and inference plumbing, to express a map that is fixed at
  compile time. The front-end gather avoids all of it.
- **One integer `prov_id` column + comparison selection**
  (`T_E*(prov[p]==0) + …`; comparison operators exist — `BinOp::{Eq,Lt,Gt}` in
  `rust/crates/ir/src/expr.rs`, `==`/`!=`/`<`/`>` in the lexer). Collapses four
  indicator files to one but keeps the dot-product boilerplate and the
  unrelated-scalars limitation, and equality-on-floats reads worse than
  self-documenting 0/1 columns. A marginal tidy, not a fix.

## Open questions

1. **Declaration surface.** `province = group(patch, by = read(…))` vs. a
   dedicated block vs. a refinement operator (`province <: patch = …`). The
   named-function form is recommended (human-first; explicit over a cryptic
   operator), but the keyword (`group` / `grouping` / `partition`) is open.
   Avoid `region` — it reads as a domain term, not a structural one.
2. **Gather spelling.** `T[province @ p]` vs. `T[province(p)]` (map-application)
   vs. `T[p in province]`. `@` is the working recommendation for being explicit
   and unambiguous against value-indexing; `province(p)` reads most naturally
   but collides with call syntax.
3. **Nesting.** District → province → region (multi-level groupings). v1 should
   handle one level but not preclude `group(group(...))`; the membership-map
   resolution composes, so this is mostly a surface question.
4. **Grouping-indexed tables.** Should `tables {}` also accept a grouping index
   (a per-group data column gathered the same way)? Same lowering; likely yes,
   but out of scope for v1.
5. **Membership provenance.** The map is read from a column of the same file
   that declares the fine dimension (recommended: one `zones.tsv` with `patch`
   and `province` columns, so the partition is inspectable in one place). Worth
   a spec note tying it to the reference-data conventions.

## Scope and phasing

Front-end only; no IR/VERSION, no golden, no Rust change.

1. **Grammar + AST.** `group(fine, by = <column>)` in `dimensions {}`; the `@`
   gather in expression index position. Spec §2 (dimensions) and §4 (parameter
   indexing) get the new forms.
2. **Expander.** Register the grouping dimension and its `patch → province` map;
   allow a grouping dim as a parameter/table index (expand to flat scalars);
   resolve `T[g @ i]` to the group-element `Param`. Validate
   totality/single-value and the gather type rule.
3. **dimcheck + diagnostics.** The type rule, the `stratify`-on-grouping
   refusal, and the membership errors, each with a fix hint.
4. **Pooling.** Allow a grouping dim as a `| dim` pool target, closing the gap
   with the hierarchical-priors line. (Density/gradient already exist; this is
   surface + classification.)

A doc-test fixture mirroring the motivating model (district dynamics, province
turnover) is the acceptance artifact: it must produce byte-identical IR to the
hand-written one-hot version, proving the sugar is exactly that.

## Non-goals

- Runtime/dynamic group membership (membership is fixed compile-time data).
- Overlapping groups or fuzzy membership (a partition is total and disjoint).
- Any change to the IR, the inference stack, or cross-language schema.
