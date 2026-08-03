# Observation and aggregation primitives

Date: 2026-07-31 Status: Increments A–D ready to implement; Increment E ready
once A–D land; Increment F is a named follow-up RFC Supersedes:
`2026-07-27-stratum-provenance.md` Fixes: gh#478, gh#488 Related: gh#459,
gh#333, gh#487

## 1. Summary

camdl spells "collapse a stratified family" five incompatible ways, and one of
those spellings is the absence of syntax. Separately, its two projection heads —
`incidence(...)` and `prevalence(...)` — are legal only as the entire right-hand
side of `projected =`, which makes a common surveillance model inexpressible and
has concentrated four silent-wrong bugs in one dispatch function.

This proposal derives a small set of primitives from the operations real disease
models actually need, then specifies them:

| primitive                                   | job                                  |
| ------------------------------------------- | ------------------------------------ |
| a bare compartment reference                | a stock, as an absolute count        |
| `prevalence(of = …, among = …, across = …)` | a stock, as a checked proportion     |
| `incidence(tr)`                             | a flow accumulated over the interval |
| `sum(…)`                                    | all collapsing, in three forms       |

**A bare family name continues to mean the total across all strata**, everywhere
in model dynamics. That is an intentional design decision, it is what makes
`let N = S + E + I + R` readable, and this proposal preserves it.

## 2. Why this design shape

The primitives below are not derived from tidiness. They are derived from
enumerating what disease modellers observe, across the diseases camdl is used
for, and collapsing that list into distinct shapes.

### 2.1 Malaria

| what is observed               | what the model must compute                                                      |
| ------------------------------ | -------------------------------------------------------------------------------- |
| Parasite prevalence, under-5   | infected children ÷ children — a **proportion over a restricted stratum**        |
| Parasite prevalence, all ages  | infected ÷ population — a **proportion pooled over age**                         |
| Slide positivity               | patent parasitaemia across **two compartments** (`Y1 + Y2`), with test sens/spec |
| RDT vs microscopy              | detection probability varies by **parasitaemia density class** — a weighted sum  |
| Clinical incidence by age      | a **flow** into clinical disease, per age band                                   |
| Entomological inoculation rate | a **flow** from the mosquito submodel to humans                                  |

Under-5 parasite prevalence is _the_ standard malaria indicator, so restriction
to a subset of strata is a first-class need, not an edge case. And the
denominator is rarely the global population — it is the population of the same
stratum.

### 2.2 Polio

| what is observed           | what the model must compute                                                       |
| -------------------------- | --------------------------------------------------------------------------------- |
| AFP cases                  | a **flow** into infection × a small paralysis fraction (≈1/200)                   |
| Environmental surveillance | total shedding across a catchment — a **weighted sum over infected compartments** |
| Seroprevalence by age      | immune ÷ population, per age band                                                 |
| Wild vs vaccine-derived    | the same quantities, **stratified by lineage**                                    |

Environmental surveillance is the clearest case where the projection is a
weighted sum that is _not_ a proportion and _not_ a plain compartment read.

### 2.3 Influenza

| what is observed           | what the model must compute                                           |
| -------------------------- | --------------------------------------------------------------------- |
| ILI consultations          | a **flow** × a reporting fraction                                     |
| Virological positivity     | positives ÷ specimens tested — proportion with a **data** denominator |
| Hospital admissions by age | a **flow** into hospitalization, per age band                         |
| HAI serology               | seropositive ÷ population — a proportion                              |
| Multiple subtypes          | all of the above, **stratified by strain**                            |

Virological positivity is the case that separates the two kinds of denominator:
the _model_ denominator makes the projection a proportion, while the _data_
denominator (`n` in a binomial) is the number of specimens.

### 2.4 COVID-19

| what is observed       | what the model must compute                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| Case notifications     | a **flow**, with **age-varying ascertainment** into one national column |
| Hospital bed occupancy | a **stock**, as an **absolute count** — not a proportion                |
| ICU occupancy          | likewise                                                                |
| Deaths                 | a **flow**                                                              |
| Seroprevalence surveys | (recovered + vaccinated) ÷ population                                   |
| Wastewater             | a **weighted sum** of shedding across infected compartments             |
| Test positivity        | proportion with a data denominator                                      |

Bed occupancy is the shape that shows a stock is not always a proportion. Age-
varying ascertainment into a single national column is the shape that is
currently inexpressible (§3.3).

### 2.5 Tuberculosis

| what is observed   | what the model must compute                       |
| ------------------ | ------------------------------------------------- |
| Prevalence survey  | bacteriologically confirmed ÷ population surveyed |
| Case notifications | a **flow** into treatment                         |
| Mortality          | a **flow** into death                             |

### 2.6 The distinct shapes

Everything above reduces to four:

1. **A stock, as an absolute count.** Hospital beds, ICU occupancy.
2. **A stock, as a proportion.** Every prevalence and serology survey. Requires
   a denominator, and the denominator is usually the matching stratum.
3. **A flow accumulated since the last observation.** Cases, deaths, admissions,
   AFP.
4. **A weighted sum over strata or compartments.** Wastewater, detection by
   density class, age-varying ascertainment, environmental surveillance.

Two observations follow, and they determine the whole design.

**Stock-versus-flow and count-versus-proportion are independent axes.** Today's
`prevalence` marks the first and says nothing about the second, even though the
second is where the errors are. Shape 1 and shape 2 are both stocks and need
different treatment; shape 3 can also be a proportion (test positivity).

**Shape 4 is arithmetic, and it applies to all the others.** A weighted sum over
strata is not a new kind of projection — it is multiplication and `sum` around
whatever the underlying quantity is. It only needs the underlying quantities to
be ordinary expressions.

### 2.7 What that implies

A stock as an absolute count needs **no operator** — a compartment reference is
already that.

A stock as a proportion needs an operator that carries the **denominator**,
because that is where the domain errors live. Two of them are silent today, and
both are invisible to dimensional analysis since every quantity involved is a
count:

```camdl
projected = I[child] / N_local[child]   # correct
projected = I[child] / N_all            # wrong by ~the stratum count — COMPILES CLEAN
```

```camdl
# numerator pooled over age; denominator pooled over age AND patch — COMPILES CLEAN
projected = sum(a in age, Y1[a,p] + Y2[a,p])
          / sum(a in age, sum(q in patch, N[a,q]))
```

A flow needs a **genuine coercion**, because a transition is not a compartment
and "accumulate this counter since the last observation, then reset it" cannot
be expressed any other way.

And shape 4 requires all of the above to be **ordinary expressions**, so that
arithmetic and `sum` compose with them.

This proposal is those four conclusions.

## 3. What is broken today

Each behaviour below was compiled against `ocaml/_build/default/bin/camdlc.exe`.

### 3.1 `prevalence(X)` is exactly `X`

```text
projected = I                     → {"current_pop_sum": ["I_child", "I_adult"]}
projected = prevalence(I)         → identical
projected = I[child]              → {"current_pop": "I_child"}
projected = prevalence(I[child])  → identical
```

The operator has no semantic content. It also has the wrong name: it returns a
**count**, while prevalence in epidemiology is a **proportion** — the actual
prevalence is the `/ N` that appears on the next line.

### 3.2 A denominator mismatch is silent

Both forms in §2.7 compile with no diagnostic.

### 3.3 `incidence` is restricted by position, not scope

`incidence(tr)` is legal only as the entire right-hand side of `projected =`.

```camdl
projected = sum(a in age, rho_a[a] * incidence(infection[a]))
  → error[E100]: undeclared function 'incidence'
```

The scoping restriction is legitimate — the counter is tied to a stream's
observation interval, so it only means something inside an observation block.
The _position_ restriction is an accident, and it makes COVID-style age-varying
ascertainment into a single national column impossible to write. The only
workaround is to index the stream, which changes the shape of the data file the
user must supply.

The same accident is why that dispatch accumulated four independent
silent-wrongs — dropped arguments, an ignored `where` guard, only one level of
nested sum understood, and a hint printing non-compiling code. All four are now
fixed (`4dcfe673`, `a387838c`), in a dispatch that should not exist.

### 3.4 Aggregation defects

```camdl
I[]                              # compiles; IR byte-identical to bare `I`
sum(a in age, I)                 # → each cell twice: the binder is never used
sum(b in aeg, I[b])              # → 0.0, silently: `aeg` is a typo for `age`
sum(a in age, p in patch, X)     # → error[E001]: flat form unsupported
```

The typo case is the most dangerous, and its silence is shape-dependent:
dividing by a literal is silent, dividing by a population trips `E300` by
accident. Neither names the undeclared dimension.

The cause is one function:

```ocaml
let dim_values ctx dim =
  match List.assoc_opt dim ctx.dim_registry with
  | Some vs -> vs
  | None    -> []          (* "no such dimension" becomes an empty list *)
```

`resolve_expr`'s `ESum` arm then maps an empty domain to `Const 0.0`, which is
correct for an empty sum and catastrophic for a typo. `table_dims`
(`expander.ml:2291`) has the identical shape.

## 4. Source-level structure vs lowered representation

**Source-level axes** are dimensions the modeller declared. **Lowered
representation** is compiler-generated: `via erlang` residence stages,
`via hyper_erlang` mixture branches.

```camdl
onset : E --> I via erlang(stages = 3, mean = 4 'days)
```

`E` becomes `E_s1`, `E_s2`, `E_s3`, hung on an invented dimension named
`__onset_stage`. Those cells are not subpopulations — no measurement
distinguishes two people in stage 2, and no data column could carry the axis.
The justification is representational, not epidemiological: `via` creates
several cells for **one declared compartment**.

The distinction is already load-bearing and undocumented. In
`ocaml/golden/seir_age_erlang_via.camdl:45`, `I` has axes `[age,
__onset_stage]`
and the contact-matrix FOI writes `I[b]` — naming one axis of two. That is a
partial index, which is normally `E287`, and it compiles:

```text
I[b]  →  pop_sum ["I_child_s1", "I_child_s2", "I_child_s3"]
```

The staging pass rewrites it before the arity check runs. So the rule the
compiler already follows, written down here for the first time, is: **you may
omit lowered axes; you may not omit source-level ones.**

Consequences:

- Any rule of the form "name every axis" counts source-level axes only.
- `via hyper_erlang` creates **no dimension at all** — it erases its source
  compartment and emits flat cells (`I__fatal__1`, …), so nothing keyed on a
  dimension can describe it.
- **Hand-rolled staging** — `dimensions { latent_stage = … }` plus
  `stratify(by = latent_stage, only = [E])` — is source-level. Public model
  structure is public.

The compiler currently detects lowered axes by testing whether the name starts
with `__`, at one site (`expander.ml:6332`). That misfires: a user dimension
named `__risk` produces an `E237` telling the modeller their compartment "has a
staged residence" and to write `S_s1`, for a compartment with no stages. It also
leaks: `E287` on a staged compartment suggests `sum(s in __onset_stage, …)`, an
identifier the user never wrote.

## 5. The primitives

### 5.1 A stock, as an absolute count — no operator

```camdl
projected = I_hosp[a]          # ICU beds occupied in age band a
projected = I_hosp             # ...across all bands
```

A compartment reference is already an instantaneous count. Nothing to add.

### 5.2 `prevalence(of = …, among = …, across = …)` — a checked proportion

```camdl
# fully indexed
projected = prevalence(of = R[a] + V[a], among = N_local[a])

# pooled over one axis of two; `patch` pinned by the stream index
prev[p in patch] {
  projected = prevalence(of = Y1[p] + Y2[p], among = N[p], across = age)
}

# pooled over everything
projected = prevalence(of = Y1 + Y2, among = N, across = [age, patch])
```

**Every axis must be accounted for** — indexed in the arguments, or named in
`across`. Nothing is inferred from context, which keeps the language's
no-auto-localization guarantee intact.

```text
error[E2xx]: 'Y1' has axes [age, patch]; 'patch' is indexed but 'age' is not
  = hint: collapse it with `across = age`, or index it
```

Because `across` accounts for the omitted axis, `Y1[p]` in that position is a
**complete** reference, not a partial index — arity is total, and the
partial-index footgun cannot occur here.

Two checks that plain division cannot carry:

1. **Subset.** The cells of `of` must lie within the cells of `among`. Rejects
   `I[child] / N_all`.
2. **Matched collapse.** `across` applies to numerator and denominator alike, so
   they cannot pool different axes. Rejects the second form in §2.7.

Both failure modes are invisible to dimensional analysis, because every quantity
involved is a count.

`prevalence` does **not** gain any other collapsing behaviour. Weighted sums,
subsets by predicate, and every other reduction stay with `sum` — otherwise the
five-spellings problem reappears inside one operator.

### 5.3 `incidence(tr)` — a flow over the interval

Reads the accumulated per-transition counter since this stream's last
observation, then resets it. Becomes an **ordinary expression**, legal anywhere
inside an observation block:

```camdl
projected = incidence(infection[a])                              # per stratum
projected = sum(a in age, incidence(infection[a]))               # pooled
projected = sum(a in age, rho_a[a] * incidence(infection[a]))    # per-stratum ascertainment
projected = paralysis_frac * incidence(infection)                # polio AFP
```

The restriction is by **scope** — inside an observation block — not by position.

### 5.4 `sum` — all collapsing, three forms

```camdl
sum(I)                                    # every source-level axis
sum(a in age, I[a])                       # one named axis
sum(a in age, p in patch, I[a, p])        # several, flat instead of nested
sum(a in age where a == under5, I[a])     # restricted
sum(b in age, C_age[a,b] * I[b])          # weighted — the binder form is required here
```

`sum(name)` takes a **family reference**, not an arbitrary expression, over four
declaration classes: compartments, indexed parameters, numeric tables, indexed
`let` bindings. Anything else is a located error:

```text
error[E2xx]: `sum(...)` takes the name of a stratified family, not an expression
  = hint: reduce each family — `sum(S) + sum(I)` — or reduce over an axis
          explicitly with `sum(a in age, S[a] + I[a])`
```

The restriction exists because without family-valued expressions (§10) there is
no principled meaning for `sum(S + I)`. A production accepting arbitrary `expr`
would parse more than the semantics can define.

Flat multi-binder is sugar for nested and takes **per-binder guards**, since the
nested form admits one at each level. Gated by a byte-identity test against the
nested form whose suite includes guarded cases.

### 5.5 Bare names are unchanged

```camdl
let N = S + E + I + R                     # the total population — unchanged
@ beta * S[a] * I / N                     # global force of infection — unchanged
```

A bare family name means the total across all strata, in every model-dynamics
position. This is deliberate: it is what makes the most-written line in
compartmental modelling readable, and the corpus shows zero live models using a
bare name over a genuine subpopulation axis in a rate.

## 6. Increment A — safety fixes

Measured over 322 `.camdl` across camdl and six sibling repos. Corpus impact is
**zero for every item**. These remain breaking language changes: a user with a
model we have not seen experiences them as such.

**A1. Dimension lookup returns an option.**

```ocaml
val dim_values : ctx -> string -> string list option
val table_dims : ctx -> string -> string list option
```

`option`, not `result`: the failure has one shape, the caller already knows the
name, and a string payload would tempt callers to surface it directly instead of
through the Diagnostics module with a code and a source span. This matches the
file's own idiom — every sibling lookup uses `_opt`.

There are 11 call sites. Changing the return type makes the type checker
enumerate them; each must decide what an unknown dimension means for it. **No
`Option.get` and no `_exn` variant**, including at sites where an upstream check
makes failure currently unreachable — those sites are safe by accident (E263
rejects a table over an unknown dimension at declaration), not by design, and
three of them use the result as an array stride.

**A2. Unknown dimension in a reduction → hard error.** gh#488. Resolution fails
before enumeration, so "unknown collection" and "known collection, zero
survivors" never share a representation.

**A3. Statically empty restricted reduction → aggregated warning.** A guard that
selects no levels stays `Const 0.0` and warns. It must not be an error: an
isolated patch legitimately contributes nothing, and emptiness is per-outer-
index. The compiler already warns rather than errors for the same situation in
transition guards (`W200`, `expander.ml:4387-4400`).

The warning aggregates **per source site**, never per unrolled instantiation:

```text
warning[W2xx]: reduction guard selected no levels for 37 of 400 instantiations
  = note: first affected binding: p = island_north
```

**A4. Reject empty index lists.** `I[]` on the compartment read path only —
`beta[]` is already `E299`, `C_age[]` is `E202`, `S[]` in stoichiometry is
`E272`.

**A5. Unused reduction binder → hard error, distinct names only.** `E283`
already owns shadowing, across transitions, lets, init, observations,
interventions, events and forcing args — a shadowed binder cannot reach this
check. Ship a regression test pinning that `sum(a in age, sum(a in age, I[a]))`
continues to give `E283`.

**A6. Bare reference to an indexed `let` diagnosed at the use site.** Today it
emits `undeclared name 'I_a'` and `'S_a'` — identifiers the user never wrote,
located inside the `let` body. Emit one located error naming the binding and its
arity.

**A7. Spec corrections, using syntax that compiles today.**
`docs/camdl-language-spec.md:69-70` claims the compiler tracks which dimension
each index variable belongs to. It does not (§8). Delete or qualify it. Fix
§25.4 and §23, whose bare-`incidence` examples contradict §12, and unskip those
doctest blocks. Replacement examples must compile against the current grammar.

## 7. Increment B — `incidence` as an expression

Ships before any tightening of observation rules, because it is the only thing
standing between a modeller and an error whose correct fix does not compile.

B1. `incidence(tr)` becomes an ordinary node in the projection-expression AST,
scoped to observation blocks.

B2. Acceptance matrix — specify all six columns for each form:

```camdl
incidence(infection[a])
rho[a] * incidence(infection[a])
sum(a in age, incidence(infection[a]))
sum(a in age, rho[a] * incidence(infection[a]))
paralysis_frac * incidence(infection)
if season then incidence(a) else incidence(b)
```

Columns: where the stream binder is in scope; stock or interval flow; where
reporting applies; whether `where` binds inside or outside; whether transition
indexing resolves before or after flow projection; the required IR node. Plus:
flat and nested reductions stay byte-identical.

## 8. Increment C — `sum` forms and dimension identity

C1. `sum(family)` over the four declaration classes (§5.4). Verified: adding a
`sum` production over a single identifier produces an **identical menhir
conflict set** to baseline.

Resolve the `quantities {}` reservation first — `expander.ml:7714-7718` rejects
`EFuncCall (("total"|"sum"), _)`. That arm is currently unreachable because
`sum` is a lexer keyword; desugaring makes it reachable, and it would fire with
"summing a stock over snapshots is cadence-dependent," the wrong message.

C2. Flat multi-binder with per-binder guards. Its menhir conflict set is **not
yet verified**; verifying it is a gate item. Corpus: 32 nested sites in 11
files, maximum depth 3, all camdl-garki.

C3. **Named indexing resolves by name, not position.** Today `INamed` parses
(`parser.mly:697`) and is discarded (`expander.ml:2415`):

```camdl
I[age = a, patch = p]     # ok
I[patch = p, age = a]     # error[E100]: undeclared name 'I_north_adult'
```

Correct dimension names in the wrong order produce an error naming an identifier
the user never wrote. This is a live defect independent of everything else here.

Cross-dimension level-name collisions stay legal — `I[low, high]` with
`age = [low, high]` and `risk = [low, high]` resolves unambiguously by declared
axis order.

C4. **Lowering metadata describes lowering, not axes**, at family granularity —
O(number of `via` declarations), not O(cells):

```ocaml
type lowering =
  | Erlang      of { source_compartment : string; transition : string; stages : int }
  | HyperErlang of { source_compartment : string; transition : string;
                     branches : branch_spec list }
```

Declared metadata is authoritative; generated names derive from it, never the
reverse. This retires the `__` sniff and with it the `E237` misdiagnosis.
Serialized as an **inert, skip-if-default** IR field — inert meaning excluded
from the run-identity hash, following `projection_state_grad`'s precedent. That
holds only while the metadata cannot affect runtime or fitting behaviour; if a
Rust consumer ever reads it semantically, the hash policy must be revisited.

There is no user-facing annotation. Only the compiler mints lowering metadata,
from `via` lowering, where it is true by construction.

## 9. Increment D — `prevalence` as a checked proportion

D1. New form: `prevalence(of = <expr>, among = <expr>, across = <dims>)`, with
`across` optional.

D2. Axis-completeness check: every source-level axis of every family referenced
in `of` and `among` is either indexed or named in `across`. Lowered axes are
never counted and never named.

D3. Subset check: the cells of `of` lie within the cells of `among`.

D4. Matched-collapse: `across` applies identically to both.

D5. The old single-argument form `prevalence(X)` is removed. It is exactly `X`
(§3.1), so migration is mechanical, and the diagnostic names the replacement:

```text
error[E2xx]: `prevalence(X)` is the same value as `X`
  = hint: for an absolute count write `Y1[a] + Y2[a]`
          for a proportion write `prevalence(of = Y1[a] + Y2[a], among = N_local[a])`
```

## 10. Increment E — the observation-boundary rule

Ships after A–D. Where a value is scored against data, pooling a
**source-level** axis must be stated — the projection names the axis, rather
than relying on a bare name to collapse it.

```camdl
# rejected: pools age silently, into a column scored against data
projected = incidence(infection)

# accepted: one reporting rate, stated
projected = sum(a in age, incidence(infection[a]))

# accepted: per-stratum reporting — a different model
projected = sum(a in age, rho_a[a] * incidence(infection[a]))
```

Model dynamics are untouched. The asymmetry is principled: in a force of
infection a bare `I` _is_ the definition — transmission is driven by everyone
infectious and there is no second reading. In an observation,
`rho * (I_child +
I_adult)` asserts one reporting rate across age groups, and
the alternative is a different model with different posteriors.

**Mechanism.** `sum(I)` and bare `I` produce identical IR, so the check cannot
inspect the resolved value. It requires a one-bit tag recording whether a
collapse arose from a bare name or an explicit reduction. Because `let` bindings
are hoisted and their bodies resolved exactly once (`register_hoisted_binding`,
`expander.ml:3060-3066`), **the tag lives on the binding, not on the
expression** — otherwise which context resolved first determines the diagnostic,
and record-field evaluation order is unspecified in OCaml.

**Interim warning, shipping with Increment A.** Non-breaking. Warn when all
three hold:

```text
E-warn(stream S):
  B := binders(S)                          # [(var, dim)] from `S[v in d, ...]`
  require B != []                                                     # (a)
  P := inline_lets(projection_expr(S))
  for (v, d) in B where d is source-level:
    for each family reference R = (f, idx) in P
        with d in axes(f) and cells(f) > 1:                           # (b)
      p    := position of d in axes(f)     # by name if INamed
      item := (idx == [] ? BARE : idx[p])
      if   item is BARE                     -> WARN
      elif item is identifier w:
             w == v                         -> selects the cell; no warning  # (c) false
             w bound by an enclosing sum(w in d, …) in P -> WARN             # (c) true
```

Conjunct (c) is evaluated at the index position of the stream's axis inside the
projection — never over the stream body. The weak reading goes silent on the
motivating case, because the binder is used in the likelihood:

```camdl
prev[a in age] {
  projected = prevalence(of = I, among = N)       # pools age silently
  prev ~ poisson(rate = rho_a[a] * projected)     # binder IS used, here
}
```

Ship that model as a positive-control test. Corpus with the correct predicate:
**0 hits across 89 indexed streams.**

## 11. Increment F — deferred

Family-valued expressions: `sum(a in age, I[a])` on an `[age, patch]` family
yielding a family over patch. Requires axis rules for all 13 `Ast.expr`
constructors, broadcasting, and a collapse-or-error decision at ~56
scalar-required call sites of `resolve_expr` — which is the same question as
whether bare names remain legal. One RFC. Blocked on C3.

Not a prerequisite for anything above. `resolve_expr` is scalar in, scalar out
(`expander.ml:3210`); reduction is compile-time unrolling; the IR has 17 scalar
constructors and no family.

## 12. Migration

Measured with a source-level detector built from the compiler's own lexer and
grammar, then confirmed by rewriting each hit into an explicit cell enumeration
and checking for byte-identical IR — 13/13.

**`prevalence(X)` → `X`**, or → the new proportion form where a `/ N` follows on
the next line. Mechanical; the diagnostic names both.

**Bare-name pooling into a data column (Increment E): 2 hits**, both
`camdl-book/vignettes/garki/garki.camdl:175`, in a file that does not compile
today for unrelated stale-syntax reasons (`E266`, `E270`, `E272`, `E273`).

The other 11 bare references are in model dynamics and are **not** affected: 5
are `via`-created axes, 6 are hand-rolled staging chains in rate expressions.

Increments A–D are additive or zero-hit. `ir/golden/` is frozen and out of scope
for regeneration (gh#384); note it is also not in the canonical compact
serialization — 5744 pretty-printed lines against 96 compact ones for the same
model — so it cannot absorb a required IR field.

## 13. Decisions taken

1. Primitives are derived from observed disease-modelling operations (§2), not
   from symmetry.
2. A bare family name means the total, in all model-dynamics positions.
   Unchanged.
3. A stock as an absolute count needs no operator.
4. `prevalence` is a **proportion** with an explicit denominator, carrying a
   subset check and a matched-collapse check. Its old single-argument form is
   removed as redundant.
5. `across` is the collapse keyword; every axis must be indexed or named, so
   arity is total and nothing is inferred from context.
6. `prevalence` gains no other collapsing behaviour — that stays with `sum`.
7. `incidence(tr)` is an ordinary expression scoped to observation blocks, not a
   head-position form.
8. `sum` is the single collapsing verb, in three forms; its whole-family form
   takes a family reference over four declaration classes.
9. Flat multi-binder takes per-binder guards and is gated by byte-identity
   including guarded cases.
10. Dimension lookup returns `option`; no `_exn` escape at any of the 11 sites.
11. Unknown dimension is an error; an empty guard is a warned zero, aggregated
    per source site.
12. Named indexing resolves by name; cross-dimension level collisions stay
    legal.
13. Lowering metadata describes lowering at family granularity, is inert in the
    run-identity hash, and has no user-facing annotation.
14. Composability (`incidence`) precedes the observation-boundary rule.

## 14. Tests

- Every §3 behaviour, as a red test before its fix.
- `prevalence` subset violation → error; matched-collapse violation → error;
  axis-incompleteness → error naming the missing axis.
- `prevalence(of = …, among = …)` fully indexed, and with `across`, produce the
  ratio the equivalent division produces.
- Each `incidence` form in §7's acceptance matrix.
- `sum(I)` equals the explicit enumeration; `sum(I) == I` when unstratified.
- Flat multi-binder byte-identical to nested, **including guarded cases**.
- `sum(S + I)` and the other rejected arguments → located errors.
- `E283` regression: same-name shadowing is not rerouted to A5.
- Positive controls that must keep passing: bare `I` in a rate;
  `let N = S + E +
  I + R`; the contact-matrix FOI
  (`sum(b in age, C_age[a,b] * I[b] / N[b])`); `E` on a `via erlang` compartment
  pooling its stages; `I[b]` on an age×stage compartment pooling stages only.
- The §10 indexed-stream positive control.

## 15. Known limitations

**Explicit cell enumeration is not dimension-safe.**
`let I_total = I[child] +
I[adult]` silently drops `elderly` if the dimension
gains a level. Only naming and reducing an axis tracks membership changes.

**Multiplicity is preserved.** `I[child] + I[child]` counts twice; no
aggregation path may collect cells into a set.

**Restricted stratification is silent about arity.** With
`stratify(by = risk,
only = [S, I])`, `sum(S)` and `sum(R)` collapse different
numbers of axes and nothing in either spelling says so.

**`via` opacity persists.** A reader of `sum(a in age, E[a])` cannot see that a
stage axis was also collapsed.

## 16. Follow-ups

- **Increment F**, above.
- **`__` namespace reservation.** The lexer permits `_` in identifiers
  (`lexer.mll:135`), so reservation needs enforcement plus a migration
  diagnostic. Corpus: 0 identifiers affected. C4 already removes the compiler's
  own dependence on the convention.
- **The cumulative-flow primitive.** Do not reserve a name now. The existing
  `total`/`sum` reservation protects a feature its own hint calls
  cadence-dependent; `integral(...)` already accumulates a stock over continuous
  time. What is missing is cumulative _flow_, to be named when designed.
- **Vector-borne and household-structured shapes** were not enumerated in §2 and
  may stress the primitives differently.
