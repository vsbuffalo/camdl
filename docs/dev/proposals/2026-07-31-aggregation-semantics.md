# Observation and aggregation primitives

Date: 2026-07-31 Status: Increment A shipped, and C2/C3 with it; B, the rest of
C, D ready to implement; Increment E ready once B–D land; Increment F is a named
follow-up RFC Supersedes: `2026-07-27-stratum-provenance.md` Fixes: gh#488
Partially fixes: gh#478 (closed by Increment E) Related: gh#111, gh#333, gh#487

## 1. Summary

camdl spells "collapse a stratified family" five incompatible ways, and one of
those spellings is the absence of syntax. Separately, `incidence(...)` and
`prevalence(...)` are legal only as the entire right-hand side of `projected =`,
which makes a common surveillance model inexpressible and has concentrated four
silent-wrong bugs in one dispatch function.

This proposal derives a small set of primitives from the operations real disease
models need, then specifies them:

| primitive                       | job                                  |
| ------------------------------- | ------------------------------------ |
| a bare compartment reference    | a stock, as an absolute count        |
| `prevalence(of = …, among = …)` | a stock, as a checked proportion     |
| `incidence(tr)`                 | a flow accumulated over the interval |
| `sum(…)`                        | all collapsing                       |

**A bare family name continues to mean the total across all strata**, everywhere
in model dynamics. That is an intentional design decision, it is what makes
`let N = S + E + I + R` readable, and this proposal preserves it.

## 2. Why this design shape

The primitives are derived from enumerating what disease modellers observe,
across the diseases camdl is used for, and collapsing that list into distinct
shapes.

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
to a subset of strata is a first-class need. And the denominator is rarely the
global population — it is the population of the same stratum.

### 2.2 Polio

| what is observed           | what the model must compute                                                       |
| -------------------------- | --------------------------------------------------------------------------------- |
| AFP cases                  | a **flow** into infection × a small paralysis fraction (≈1/200)                   |
| Environmental surveillance | total shedding across a catchment — a **weighted sum over infected compartments** |
| Seroprevalence by age      | immune ÷ population, per age band                                                 |
| Wild vs vaccine-derived    | the same quantities, **stratified by lineage**                                    |

### 2.3 Influenza

| what is observed           | what the model must compute                                           |
| -------------------------- | --------------------------------------------------------------------- |
| ILI consultations          | a **flow** × a reporting fraction                                     |
| Virological positivity     | positives ÷ specimens tested — proportion with a **data** denominator |
| Hospital admissions by age | a **flow** into hospitalization, per age band                         |
| HAI serology               | seropositive ÷ population — a proportion                              |
| Multiple subtypes          | all of the above, **stratified by strain**                            |

Virological positivity separates the two kinds of denominator: the _model_
denominator makes the projection a proportion, while the _data_ denominator (`n`
in a binomial) is the number of specimens.

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

Bed occupancy shows a stock is not always a proportion. Age-varying
ascertainment into a single national column is the shape that is currently
inexpressible (§4.3).

### 2.5 Tuberculosis

| what is observed   | what the model must compute                       |
| ------------------ | ------------------------------------------------- |
| Prevalence survey  | bacteriologically confirmed ÷ population surveyed |
| Case notifications | a **flow** into treatment                         |
| Mortality          | a **flow** into death                             |

### 2.6 The distinct shapes

1. **A stock, as an absolute count.** Hospital beds, ICU occupancy.
2. **A stock, as a proportion.** Every prevalence and serology survey. Requires
   a denominator, usually the matching stratum.
3. **A flow accumulated since the last observation.** Cases, deaths, admissions,
   AFP.
4. **A weighted sum over strata or compartments.** Wastewater, detection by
   density class, age-varying ascertainment.

Two facts follow, and they determine the design.

**Stock-versus-flow and count-versus-proportion are independent axes.** Today's
`prevalence` marks the first and says nothing about the second, even though the
second is where the errors are.

**Shape 4 is arithmetic.** It is multiplication and `sum` around whatever the
underlying quantity is — which requires those quantities to be ordinary
expressions.

### 2.7 What that implies

A stock as an absolute count needs **no operator**.

A stock as a proportion needs an operator carrying the **denominator**, because
that is where the domain errors live. Two are silent today, both invisible to
dimensional analysis since every quantity is a count:

```camdl
projected = I[child] / N_local[child]   # correct
projected = I[child] / N_all            # wrong by ~the stratum count — COMPILES CLEAN
```

```camdl
# numerator pooled over age; denominator pooled over age AND patch — COMPILES CLEAN
projected = sum(a in age, Y1[a,p] + Y2[a,p])
          / sum(a in age, sum(q in patch, N[a,q]))
```

A flow needs a **genuine coercion** — "accumulate this counter since the last
observation, then reset it" cannot be said any other way.

Shape 4 requires composability, but only where a workaround does not exist. §5.2
records why that lands differently for `incidence` than for `prevalence`.

## 3. Axis typology

Two kinds of axis exist, they behave differently, and the language spec has
never defined either. Both terms below are normative and belong in
`docs/camdl-language-spec.md`.

**Population strata** are declared in `dimensions {}` — age, patch, risk group,
vaccination status. They partition individuals.

**Residence structure** is created by the compiler from `via erlang` (dwell-time
stages) or `via hyper_erlang` (exit pathways). It describes _how long an
individual stays in a compartment and by which exit they leave_, not who they
are.

|                               | **population stratum**      | **residence structure**                                |
| ----------------------------- | --------------------------- | ------------------------------------------------------ |
| how it arises                 | declared in `dimensions {}` | generated by `via` lowering                            |
| what it denotes               | a partition of individuals  | position in a dwell-time law, or exit pathway          |
| indexable?                    | yes — `I[child]`            | no — never named by the modeller                       |
| bare name collapses it?       | yes                         | yes                                                    |
| data can be stratified by it? | yes — an `age : dim` column | no — no measurement distinguishes stage 2 from stage 3 |
| trajectory columns?           | yes — `I_child`             | yes — `I_child_s1`                                     |
| counts toward index arity?    | yes                         | **no**                                                 |

The behavioural rule that falls out, stated here for the first time:

> **You may omit residence structure from an index; you may not omit a
> population stratum.**

The compiler already follows it. In `ocaml/golden/seir_age_erlang_via.camdl:45`
the contact-matrix force of infection writes `I[b]` where `I` has axes
`[age, __recovery_stage]` — a partial index, normally `E287`, and it compiles:

```text
I[b]  →  pop_sum ["I_child_s1", "I_child_s2", "I_child_s3"]
```

The staging pass rewrites it before the arity check runs. Today that is an
accident of pass ordering; this proposal makes it a stated rule.

The justification is representational, not epidemiological: `via` creates
several cells for **one declared compartment**. A modeller who writes
`onset : E --> I via erlang(stages = 3, …)` declared one `E`.

Two consequences that matter later:

- **Hand-rolled staging is a population stratum.**
  `dimensions { latent_stage =
  … }` plus
  `stratify(by = latent_stage, only = [E])` is declared structure, so it is
  indexable and counts toward arity. Public model structure is public.
- **`via hyper_erlang` creates no dimension at all.** It erases its source
  compartment and emits flat cells (`I__fatal__1`, …), so nothing keyed on a
  dimension can describe it. Its generated _compartment_ names are intentionally
  visible — they are trajectory columns and scenario-referenceable transition
  names.

## 4. What is broken today

Each behaviour was compiled against `ocaml/_build/default/bin/camdlc.exe`.

### 4.1 `prevalence(X)` is exactly `X`

```text
projected = I                     → {"current_pop_sum": ["I_child", "I_adult"]}
projected = prevalence(I)         → identical
projected = I[child]              → {"current_pop": "I_child"}
projected = prevalence(I[child])  → identical
```

No semantic content, and the wrong name: it returns a **count**, while
prevalence in epidemiology is a **proportion**.

The multi-positional form is live and documented (`docs/camdl-run-spec.md`
§14.1):

```text
projected = prevalence(I, R) → {"derived_expr": {"pop_sum": ["I_child","I_adult","R_child","R_adult"]}}
```

### 4.2 Denominator mismatches are silent

Both forms in §2.7 compile with no diagnostic.

### 4.3 `incidence` is restricted by position, not scope

```camdl
projected = sum(a in age, rho_a[a] * incidence(infection[a]))
  → error[E100]: undeclared function 'incidence'
```

The scoping restriction is legitimate — the counter is tied to a stream's
observation interval. The _position_ restriction is an accident, and it makes
COVID-style age-varying ascertainment into a single national column impossible.
The only workaround indexes the stream, changing the data file the user must
supply.

Two of the four forms §5.3 motivates are already expressible with the
coefficient in the likelihood — verified: `projected = incidence(infection)`
plus `afp ~ poisson(rate = paralysis_frac * projected)` compiles. **Only the
per-stratum weight inside a pooled sum is genuinely inexpressible**, and that is
what justifies the change.

The same accident is why that dispatch accumulated four independent
silent-wrongs — dropped arguments, an ignored `where` guard, only one level of
nested sum understood, and a hint printing non-compiling code. All four are now
fixed (`4dcfe673`, `a387838c`), in a dispatch that should not exist.

### 4.4 Aggregation defects

```camdl
I[]                              # compiles; IR byte-identical to bare `I`
sum(a in age, I)                 # → each cell twice: the binder is never used
sum(b in aeg, I[b])              # → 0.0, silently: `aeg` is a typo for `age`
sum(a in age, p in patch, X)     # → error[E001]: flat form unsupported
```

The typo case is most dangerous, and its silence is shape-dependent: dividing by
a literal is silent, dividing by a population trips `E300` by accident. Neither
names the undeclared dimension.

The cause is one function:

```ocaml
let dim_values ctx dim =
  match List.assoc_opt dim ctx.dim_registry with
  | Some vs -> vs
  | None    -> []          (* "no such dimension" becomes an empty list *)
```

`table_dims` (`expander.ml:2291`) has the identical shape.

## 5. The primitives

### 5.1 A stock, as an absolute count — no operator

```camdl
projected = I_hosp[a]          # ICU beds occupied in age band a
projected = I_hosp             # ...across all bands
```

### 5.2 `prevalence(of = …, among = …)` — safe sugar for a proportion

`prevalence(of = X, among = Y)` **is** `X / Y` — same value, same IR — plus two
checks that plain division cannot carry. It adds no arithmetic of its own and
does no collapsing of its own. Division stays legal everywhere; you simply do
not get the checks.

```camdl
# fully indexed
projected = prevalence(of = R[a] + V[a], among = N_local[a])

# pooled over age — `sum` does the collapsing, on each side
projected = prevalence(of    = sum(a in age, Y1[a] + Y2[a]),
                       among = sum(a in age, N[a]))

# pooled over two axes
projected = prevalence(of    = sum(a in age, p in patch, Y1[a,p] + Y2[a,p]),
                       among = sum(a in age, p in patch, N[a,p]))

# density-weighted detection (§2.1) — weights go in the numerator
projected = prevalence(of = 0.4 * Y1[a] + 0.9 * Y2[a], among = N_local[a])
```

**There is no collapsing argument.** An earlier draft carried an
`across = <dims>` keyword that wrapped each side in `sum(v in d, …)`. That
construction multiplies by `|d|` every additive term not carrying `d` — a scalar
`let`, a parameter, a constant, or an already-pinned reference. Measured:

```text
sum(v in age, N_h)       →  reduce [N_h, N_h]                 # denominator doubled
sum(v in age, I[child])  →  pop_sum ["I_child", "I_child"]    # numerator doubled
```

On a garki-shaped model with a scalar denominator that reports exactly half the
true prevalence — the error class §2.7 says the operator exists to prevent.
`sum` expresses everything `across` did, so the keyword is dropped rather than
repaired.

**Two checks.**

1. **Subset.** `cells(e)` is the set of `Ir.Pop` names reachable in
   `resolve_expr`'s output for `e`, with `BindingRef` resolved through the
   hoisted-binding table (`register_hoisted_binding`, `expander.ml:3060`), and
   with a `visiting` set — the compiler has no `let`-cycle guard, so a
   self-referential `let` reached from a use site hangs it (§16). Requires
   `cells(of) ⊆ cells(among)`.

   **The limits, stated.** This rejects a numerator outside its denominator
   (`of = I[child]`, `among = S[child] + R[child]`) and a pinned-level mismatch
   (`of = I[child]`, `among = N_local[adult]`). It is **vacuous** when `among`
   has no cells — a parameter, a constant, or a table lookup, all of which are
   legitimate denominators (§2.5's "population surveyed" is often a table). It
   is **defeated** when the denominator is written as an explicit cell
   enumeration rather than an indexed family, because that spelling erases the
   axis (§13); `docs/dev/proposals/fixtures/garki_post_proposal.camdl:46,78` is
   exactly that case and passes while being wrong.

   The check is set-based, so it does not establish that the result lies in
   `[0,1]`: `of = I[child] + I[child]` passes and yields a ratio up to 2. The
   name is "safe sugar", not "proven proportion".

2. **Matched collapse.** Compare the multiset of dimensions each side collapses
   — the binder dimensions of every `sum` on the spine of `of` against those of
   `among`. A mismatch is an error. This catches §2.7's second motivating case:

   ```text
   error[E2xx]: numerator collapses [age]; denominator collapses [age, patch]
     = hint: both sides of a proportion must pool the same dimensions
   ```

   This is a check rather than a construction, so it can be evaded by writing
   the division directly — which is the honest position, since the construction
   that would have made it unavoidable was itself wrong.

**`prevalence` stays head-position for now.** Unlike `incidence` (§5.3), the
nested form has a working escape hatch — plain division, which stays legal and
unchecked:

```camdl
# age-standardised prevalence — compiles today, and continues to
projected = sum(a in age, w[a] * (I[a] / N_local[a]))

# the same thing with prevalence nested — remains E100 in this increment
projected = sum(a in age, w[a] * prevalence(of = I[a], among = N_local[a]))
```

Composable `prevalence` would buy _checking on a form that already works_, where
composable `incidence` buys _a model that is otherwise impossible_. Different
strength of case, so different answer. Revisit when a concrete model needs it;
tracked in §16.

**A flow may not appear in `of` or `among`.** Once `incidence` is an ordinary
expression (§7), `prevalence(of = incidence(tr), among = N)` parses, and it has
no IR representation: B1's whole design keeps flow reads out of `Expr`, and the
enclosing `DerivedExpr` would be classified `Instant` — the wrong temporal kind
for a quantity that must accumulate and reset. Reject with a named diagnostic
pointing at the pooled-column form. Incidence-per-capita is a conventional
reporting quantity (§2.1, §2.3), so this will be written on day one.

`prevalence` gains no collapsing behaviour — weighted sums, subsets by predicate
and every other reduction stay with `sum`.

### 5.3 `incidence(tr)` — a flow over the interval

Reads the accumulated per-transition counter since this stream's last
observation, then resets it. Becomes an **ordinary expression** scoped to
observation blocks:

```camdl
projected = incidence(infection[a])                              # per stratum
projected = sum(a in age, incidence(infection[a]))               # pooled
projected = sum(a in age, rho_a[a] * incidence(infection[a]))    # ← the load-bearing case
projected = paralysis_frac * incidence(infection)                # expressible today via the likelihood
```

### 5.4 `sum` — all collapsing, three forms

```camdl
sum(I)                                    # every population stratum
sum(a in age, I[a])                       # one named axis
sum(a in age, p in patch, I[a, p])        # several, flat instead of nested
sum(a in age where a == under5, I[a])     # restricted
sum(b in age, C_age[a,b] * I[b])          # weighted — the binder form is required here
```

**The no-binder form needs no separate production and no new semantics.** Under
the uniform rule (C2) `sum(I)` is a one-element list whose single element is the
body, so it evaluates to the body — and a bare compartment name already means
the total across all strata (§5.5). The right answer falls out:

```camdl
sum(I)        # → the total of I, exactly as bare `I` means today
sum(S + I)    # → total of S plus total of I; redundant, not wrong
sum(rho_a)    # → E100, as a bare indexed parameter is today
```

So `sum(family)` is documentation for compartments — an explicit way to say "I
mean the total here" — and for indexed parameters, tables and shaped `let`s the
binder form is the spelling: `sum(a in age, rho_a[a])`.

An earlier draft made `sum(family)` a dedicated production over four declaration
classes and required `sum(S + I)` to be an error. Both were wrong. The dedicated
production conflicts with the uniform rule (measured: two arbitrarily-resolved
shift/reduce conflicts at `SUM LPAREN IDENT . RPAREN`), and `sum(S + I)` has a
perfectly good meaning under bare-name semantics.

### 5.5 Bare names are unchanged

```camdl
let N = S + E + I + R                     # the total population — unchanged
@ beta * S[a] * I / N                     # global force of infection — unchanged
```

## 6. Increment A — safety fixes

**Shipped.** A1 + A3 in `b9e6e04a` (PR#489), A2 in `bbf94fb1` (PR#498, gh#488),
A6 in `1d03947d` (PR#500), A4 + A5 + A7 in `9643c246` (PR#558). The user-facing
migrations are in `docs/language-changes.md`. Retained below as the decision
record; open residue is gh#559, gh#560 and gh#504.

Measured over 322 `.camdl` across camdl and six sibling repos. **Corpus impact
is zero for every item. These remain breaking language changes** — a user with a
model we have not seen experiences them as such.

**A1. Dimension lookup returns an option.**

```ocaml
val dim_values : ctx -> string -> string list option
val table_dims : ctx -> string -> string list option
```

`option`, not `result`: the failure has one shape, the caller already knows the
name, and a string payload would tempt callers to surface it directly rather
than through Diagnostics with a code and a span. This matches the file's idiom —
every sibling lookup uses `_opt`.

**There are 15 typed call sites**, not 11: 11 `dim_values` in `expander.ml`, 2
in `inspect.ml:891,894`, and 2 `table_dims` (`expander.ml:3248,5671`).

**The type checker does not enumerate the hazard.** Ten further sites
hand-inline the same lookup with a silent default and will compile unchanged:

```
expander.ml:470, 2089, 3111, 8049, 8913
inspect.ml:21, 176, 292, 588, 1170
```

`expander.ml:2089` even carries the comment "A dim with no registered levels
contributes nothing (the dim error is reported elsewhere)" — the
safe-by-accident assertion A1 exists to eliminate. **All 25 sites must route
through the new accessor**; a signature change alone leaves 10 identical hazards
behind.

A 26th lookup at `expander.ml:4911` already diagnoses correctly (`E330`) and is
**deliberately excluded** — noted so an implementer who greps 26 hits knows
which one to leave.

**Disposition is per-site, and one is not a `[]` default.** `expander.ml:8913`
is a `List.filter_map` whose `None` arm _drops_ the dimension from
`model_structure.dimensions` — the field `deme.rs` reads (§8.1). It must become
an error, not an empty list. Every other site emits a located diagnostic at the
point of use. Where A2's new error would fire alongside an existing E263
(`:2454`, `:3275`), A2 **replaces** it rather than adding a second diagnostic
for one mistake.

**No `Option.get`, no `_exn` variant**, including where an upstream check makes
failure currently unreachable. Three sites use the result as an array stride
(`expander.ml:2454` `shape_index`, `:3275` table lookup, `:5711`
`table_backed_knots`). The first two are protected by E263 fired from the _same
expression_; the third is protected by nothing for an inline table (see §16).

**A2. Unknown dimension in a reduction → hard error.** gh#488. Resolution fails
before enumeration, so "unknown collection" and "known collection, zero
survivors" never share a representation.

**A3. Statically empty restricted reduction → aggregated warning.** A guard
selecting no levels stays `Const 0.0` and warns. Not an error: an isolated patch
legitimately contributes nothing, and emptiness is per-outer-index.

**This requires two pieces of scaffolding the proposal must fund.** `ESum` has
no source location (`ast.ml:77`, unlike its siblings at `:73-74`), and
`Diagnostics.t` (`diagnostics.ml:35`) is an append-only list with no tally. So
A3 needs (i) a `loc` added to `ESum` and threaded through its 36 sites across 7
files — which A2 wants anyway, so do it once — and (ii) a per-site counter in
`ctx` flushed at phase end. W200 (`expander.ml:4387-4400`) is **not** a
precedent: it is one post-hoc check per transition, aggregating nothing.

```text
warning[W2xx]: reduction guard selected no levels for 37 of 400 instantiations
  = note: first affected binding: p = island_north
```

**A4. Reject empty index lists.** `I[]` on the compartment read path only —
`beta[]` is already `E299`, `C_age[]` is `E202`, `S[]` in stoichiometry is
`E272`. Verified: all four.

**A5. Unused reduction binder → hard error, distinct names only.** `E283` owns
shadowing, and its scope is verified across all four binding sources — a nested
same-name binder, a transition binder, a stream index, and an enclosing shaped
`let` index. A shadowed binder cannot reach A5.

`E283` does **not** cover a binder shadowing a _global_ declared name; that is a
separate defect filed in §16, and A5's premise is scoped to bindings
accordingly. Ship a regression test pinning that
`sum(a in age, sum(a in age, I[a]))` continues to give `E283`.

**A6. Bare reference to an indexed `let` diagnosed at the use site.** Today it
emits `undeclared name 'I_a'` and `'S_a'` — identifiers the user never wrote,
located inside the `let` body.

**A7. Spec corrections, using syntax that compiles today.**
`docs/camdl-language-spec.md:69-70` claims the compiler tracks which dimension
each index variable belongs to. It does not (§8). Delete or qualify it. Fix
§25.4 and §23, whose bare-`incidence` examples contradict §12, and unskip those
doctest blocks. Add the §3 typology. Replacement examples must compile against
the current grammar — do not document Increment B's composable syntax as
available.

## 7. Increment B — `incidence` as an expression

Ships before any tightening of observation rules: it is the only thing standing
between a modeller and an error whose correct fix does not compile.

**B1. Lowering.** `Projection::WeightedFlowSum(Vec<(Expr, String)>)` — the sum
of weight × flow — appended at hash index 5, with the accompanying `ir/VERSION`
bump. **Do not pin the target number here**: it has gone 0.30 → 0.31 → 0.32 →
0.33 while this proposal sat, so a literal is stale on arrival. Read
`ir/VERSION` and take the next one.

This is deliberately **not** a new `Expr` constructor. A flow-read node in
`Expr` would make `temporal_kind()` (`rust/crates/ir/src/observation.rs:47`)
stop being a total function of the variant, would require a conditional
`resets_after_observation`, a `WrtFlow` autodiff pass, a `projection_flow_grad`
IR field, and — decisively — would make a flow read representable inside a rate
expression. The variant form keeps `temporal_kind` total (`WeightedFlowSum` is
`Interval`) and gets `∂proj/∂flow_i = w_i` structurally free.

**Unit-weight forms keep their existing lowering** to `CumulativeFlow` /
`CumulativeFlowSum`, so all 18 tracked goldens containing `cumulative_flow` stay
byte-identical. That is a one-line implementation rule with an 18-golden blast
radius if violated.

**B2. The accumulator becomes per-reference.** `acc` is a per-stream scalar
today (`types.rs:320`), and an Interval stream never evaluates a projection
expression at all — `project_stream_from_acc` short-circuits
(`multi_stream_obs.rs:1087`). So `IntervalSlot` gains `(offset, len)` and
`fold_into_acc` / `reset_due_acc` index by offset, one bin per weighted term.

**Reading `flow_accumulators` at scoring time is prohibited.** It is
blanket-zeroed at every union observation index, so it is correct only on
homogeneous cadence and silently wrong otherwise — pinned by
`sim/tests/per_stream_reset.rs:266`, a mutation guard asserting 20 where the
correct 30-day AFP bin is 300. `tests/fixtures/polio_afp_es_2patch.camdl` is a
live multi-cadence model of exactly that shape.

The reset stays keyed on the **stream**, never the reference: a reference
appearing twice contributes twice and resets once.

**B3. Weight restrictions.** Each weight must be **constant over the observation
interval**. Concretely it must be free of flow reads, of `t`/`TimeFunc`/`Cond`,
**and of `Pop`** — a state read is the case a syntactic "time-independent" rule
misses, and it breaks the same identity: the projection is evaluated once at the
observation instant, so `w(x(t_obs))·ΣΔN ≠ ∫ w(x(s)) dN(s)` exactly as
`w(t)·ΣΔN ≠ ∫w dN`. A weight reached through a hoisted `let` must be resolved
through `ctx.hoisted_tbl` before the walk, or `BindingRef` hides the violation.

Without the `Pop` clause the B3/B4 boundary is syntactic and one algebraic step
crosses it: B4 defers `Σ rho_a·inc_a / N`, but `Σ (rho_a/N)·inc_a` is the same
quantity with the division moved inside the weight, and a flow-free, `t`-free
rule admits it. The `Pop` clause also preserves B1's decisive argument —
`∂proj/∂flow_i = w_i` is structurally free only when the weight is state-free.

**B4. Deferred with named diagnostics, not silently rejected**: `incidence`
under `Cond`, under a nonlinear function, and mixed with instant state in one
expression (`Σ rho_a·inc_a / N`). The mixed case matters — it would otherwise
pass `gradient_capability.rs:442-456`, which gates on
`matches!(om.projection, Projection::DerivedExpr(_))` and therefore does not
examine a new `WeightedFlowSum` variant at all. **Extend that gate to the new
variant** as part of B, or a state-dependent weight silently drops its term from
the ODE-NUTS gradient.

**B5. Delete `explicit_incidence_sum`** (`expander.ml:7800`, dispatched at
`:7978`) rather than extend it. It is the syntactic walker that accumulated the
four silent-wrongs in §4.3; "reach for the existing seam" cuts the other way
when the seam is the defect.

**B6. Acceptance matrix.**

| form                                              | stream binder | kind     | reporting  | `where`                             | indexing                                  | IR node                           |
| ------------------------------------------------- | ------------- | -------- | ---------- | ----------------------------------- | ----------------------------------------- | --------------------------------- |
| `incidence(infection[a])`                         | stream header | Interval | likelihood | n/a                                 | before                                    | `cumulative_flow` — unchanged     |
| `rho[a] * incidence(infection[a])`                | stream header | Interval | projection | n/a                                 | before                                    | `weighted_flow_sum` (1 term)      |
| `sum(a in age, incidence(infection[a]))`          | sum binder    | Interval | likelihood | outside; prunes compile-time domain | before                                    | `cumulative_flow_sum` — unchanged |
| `sum(a in age, rho[a] * incidence(infection[a]))` | sum binder    | Interval | projection | outside                             | before                                    | `weighted_flow_sum` (n terms)     |
| `paralysis_frac * incidence(infection)`           | none          | Interval | projection | n/a                                 | before; still E280 on a stratified family | `weighted_flow_sum` (1 term)      |
| `if season then incidence(a) else incidence(b)`   | —             | —        | —          | —                                   | —                                         | **deferred (B4)**                 |

**B7. The `ir/VERSION` bump re-keys every cached run, for every model.**
`ir_version` is a hashed field of `ModelDigest` (`runid/src/inputs.rs:174-182`),
so the bump invalidates every stored fit and simulate — not only models using
`incidence`. Unavoidable if B ships; it belongs in the release notes, not only
here.

**One bump carries both changes.** The typed dimension name (§8.1) also re-keys,
so it lands in this bump rather than paying a second invalidation. Land it as a
stacked sequence — §8.1 first, goldens regenerated, green; then B1/B2; then
B3/B4; then B5 — so no commit batches two semantic changes without an
intermediate green run.

**B8. `inc_<stream>` stays the raw flow sum.** `incidence_streams()`
(`multi_stream_obs.rs:1026`) builds it as the unweighted `Σ flows[i]`. Under a
weighted projection it diverges from `projected`, which is correct — a modeller
wants true incidence in the trajectory and reported counts in the predictive —
but its doc comment currently claims it is "the model's declared `FlowSum`
projection" and must be restated.

**B9. Scope becomes a rule, because B5 deletes the accident that enforced it.**
Today `incidence` is contained to observation blocks only because the projection
dispatch is the sole thing that resolves it. Verified at HEAD: a `let` holding
`incidence` compiles as long as nothing references it, because an unused hoisted
binding is never walked, and errors the moment it is used:

```camdl
let recent = incidence(infection)                     # compiles — never resolved
transitions { waning : R --> S @ 0.001 * recent }     # error[E100]: undeclared function 'incidence'
quantities  { cum = final(incidence(infection)) }     # error[E100]: undeclared function 'incidence'
```

`E100` is the wrong diagnostic — `incidence` is a documented construct, not an
unknown name — and after B5 there is no dispatch left to produce even that. B
must therefore add an explicit check with a located diagnostic, modelled on the
`E278` arm that already does this for `observed`/`sum_observed`
(`expander.ml:4280`):

```text
error[E2xx]: `incidence(infection)` is only meaningful inside an observation stream
  = note: it reads the flow accumulated since THIS stream's last observation and
          then resets it, so it needs an observation interval to be defined
  = hint: for a policy trigger on recent cases, use
              sum_observed(<stream>, window = 28 'days)
          in `reactive_interventions { … }`
  = note: reporting cumulative incidence as a quantity is gh#565
```

Why the restriction is semantic and not an implementation limit: the counter is
per-stream and resets on that stream's observation times. A rate reading it
would (i) have no defined window on a model with two cadences, and (ii) make the
process itself depend on the reporting schedule, so the same model fitted to
weekly and to monthly data would be two different processes. B2's reset
invariant — keyed on the stream, a reference contributing twice but resetting
once — also has no owner outside a stream.

Build the check on the scope table from gh#566 if that audit has landed; a
fourth ad-hoc arm otherwise, with the table as the stated follow-up. Ship a red
test for the `let` route specifically — it is the one an implementer misses,
since the naive test (`incidence` directly in a rate) already fails today for
the wrong reason.

## 8. Increment C — `sum` forms and dimension identity

**C2 and C3 are shipped**; C4/C4a remain; C5 and C6 are superseded by the typed
dimension name (gh#568, §8.1 below), which lands inside Increment B's
`ir/VERSION` bump rather than here.

**C1 is subsumed by C2.** There is no separate `sum(family)` production; the
no-binder form falls out of the uniform rule (§5.4). Nothing is added to the
grammar beyond C2, no new AST node is needed, and the `quantities {}`
reservation at `expander.ml:7714-7718` stays unreachable because `sum` remains a
lexer keyword rather than becoming an `EFuncCall`.

**C2. Flat multi-binder — replace the existing productions, do not add alongside
them.** _Shipped in `b9e6e04a` (PR#489)._

```camdl
sum(a in age, m in imm, k in compound, N[v,a,m,k])
```

The naive formulation — adding a multi-binder production while keeping the
single-binder one — introduces two shift/reduce conflicts, because at the comma
after `IDENT IN IDENT` both the old rule (shift, body follows) and the new list
rule (reduce, list continues) apply. **Measured**:

```text
baseline                        s/r=1  r/r=1
add multi-binder alongside      s/r=3  r/r=1     ← two new conflicts
replace with one uniform rule   s/r=1  r/r=1     ← identical to baseline
```

The replacement expresses every arity through a single production over a
`separated_nonempty_list` whose last element is the body; the one-binder case is
just the shortest list. The conflict report is byte-identical to baseline modulo
state numbering, same token (`LBRACE`, the pre-existing transitions conflict).
No bracketed binder list is needed.

Per-binder guards are retained, since the nested form admits one at each level.
Gated by a **byte-identity** test against the nested form whose suite **includes
guarded cases** — only 2 of 300 corpus sums carry a guard, so an unguarded-only
suite would pass while the sugar claim was false.

Corpus: 32 nested sites in 11 files, maximum depth 3, all camdl-garki.

**C3. Named indexing resolves by name, not position.** _Shipped as gh#459
(`6a3330a2`), outside this proposal's PRs._ `INamed` used to parse
(`parser.mly:697`) and be discarded, so correct dimension names in the wrong
order produced an error naming an identifier the user never wrote. Both
spellings now lower identically:

```camdl
I[age = a, patch = p]     # both resolve to I_child_north
I[patch = p, age = a]     # byte-identical IR
```

Cross-dimension level-name collisions stay legal — `I[low, high]` with
`age = [low, high]` and `risk = [low, high]` resolves unambiguously by declared
axis order.

This closed the user-facing symptom of gh#111 (indexed references lowered by
string concat) while leaving the underlying resolver unbuilt — see §8.1, which
takes a further slice of it.

**C4 folds into §8.1.** Measured after C4a landed: the `__` sniff has exactly
**one** consumer — `expander.ml`'s staged-endpoint diagnostic, which tests
`String.sub d 0 2 = "__"` to decide whether to name a synthesized axis the user
never wrote. §8.1's `Generated` constructor answers that question directly
(`staged ep = List.exists is_generated ep.ep_dims`), so building C4's standalone
`lowering` record first and the typed name second would build the same authority
twice. C4 ships as part of §8.1, inside B's bump; the record below is retained
as the shape of the payload `Generated` carries.

**C4 (retained, as §8.1's payload). Lowering metadata describes lowering, not
axes**, at family granularity — O(number of `via` declarations), not O(cells):

```ocaml
type lowering =
  | Erlang      of { source_compartment : string; transition : string; stages : int }
  | HyperErlang of { source_compartment : string; transition : string;
                     branches : branch_spec list }
```

Declared metadata is authoritative; generated names derive from it, never the
reverse. This retires the `__` sniff (`expander.ml:6332`) and with it the `E237`
misdiagnosis of a user dimension named `__risk`.

**`lowering` is compiler-internal.** It does not reach the IR; C6's
per-dimension flag is the IR-visible part, and it is derived from `lowering`. So
C4 alone needs no schema change, no `ir/VERSION` bump and no golden
regeneration.

**C4a. Apply the `via` rewrite to every container that reads state.**
`sum_staged_refs` / `sum_hyper_refs` are applied to five containers today —
`ctx.transitions`, `ctx.let_bindings`, `ctx.init_entries`, `ctx.balance_decl`,
`ctx.obs_decls` (`expander.ml:1526-1561`, `:1809-1822`). `ctx.quantity_decls`,
`ctx.interv_decls`, `ctx.event_decls` and `ctx.reactive_decls` are absent, so
§3's rule is false there. Measured — identical syntax, opposite outcome by
container:

```camdl
# I is [age, __recov_stage]
observations { … projected = I[child] … }   → pop_sum ["I_child_s1","I_child_s2","I_child_s3"]
quantities   { peak = max(I[child]) }       → error[E287]: … only 1 of 2 were indexed
interventions { isolate : transfer(from = I[child], …) }  → error[E287]
```

and on a `hyper_erlang` compartment, `quantities { peak = max(I) }` gives
`error[E100]: undeclared name 'I'` for a compartment declared in
`compartments { S, I, R }`.

This is a prerequisite for §8.1, not an independent nicety: the only
dimension-safe workaround today is `sum(s in __recov_stage, I[child, s])`, which
compiles now and which §8.1 removes from the surface. Without C4a the sole
surviving spelling is the explicit enumeration
`I_child_s1 + I_child_s2 + I_child_s3`, which §13 names as the form that
silently drops a stage when `stages` changes. No committed model reports a
quantity on one stratum of a staged compartment, so corpus measurement does not
establish safety here.

### 8.1 Generated dimension names are a type, not a prefix

Supersedes C5 (reserve `__` in the lexer) and C6 (an inert `generated : bool` on
`Dimension`). gh#568. **Lands inside Increment B's `ir/VERSION` bump**, not here
— it re-keys, and B re-keys anyway (§7, B7).

A `via erlang(...)` declaration mints its staging axis by string concatenation —
`let dim_name = Printf.sprintf "__%s_stage" tr.trname` (`expander.ml:1529`) —
and registers it in the user's own `dim_registry` (`expander.ml:1453-1458`). The
prefix is then carrying three jobs: classification (the sniff at
`expander.ml:6332`), collision avoidance, and display. C4 takes the first
(`lowering` becomes authoritative). What remains is a name that must not collide
and must not be shown:

```camdl
dimensions { __onset_stage = [x1, x2] }
onset : E --> I via erlang(stages = 3, mean = 4 'days)
  → error[E212]: dimension '__onset_stage' is declared more than once in dimensions {}
```

The modeller declared it once.

**The fix is to make the discriminant a type:**

```ocaml
type dim_name =
  | User      of string
  | Generated of { kind : lowering_kind; source : string }
```

A user-supplied string cannot construct a `Generated`, so the collision is
structurally impossible, no lexer reservation is needed, and "did someone
remember to check the flag?" stops being a question. Display and diagnostics
match on the constructor.

**This does not split the registry.** A separate namespace for generated axes
was considered and rejected on measurement: most readers of `ctx.stratifies`
want the **union** — `comp_dims` enumerates cells and needs every axis, as do
`n_pre` (`:1419`), `src_is_stratified` (`:1580`) and the E214 check (`:1995`).
Splitting would make the common case worse to serve the rare one. The typed
variant keeps one list and moves the discriminant into the element, so every
union consumer is unaffected. The earlier rejection is of a different design.

**What it costs, stated.** `Dimension.name` is a `String` in the IR and
`ModelStructure.compartment_dims` is hashed as `HashMap<String, Vec<String>>`
via `h.write_str_map(...)` (`runid/ir_hash.rs:1092-1094`), so the representation
change forces `hash_into` to change and re-keys every stored fit and simulate.
That is accepted rather than designed around: pre-1.0, invalidating cached runs
is cheap and a wrong representation is not. Riding B's bump makes it one
invalidation instead of two.

**Names must still reach the IR, not be omitted.**
`rust/crates/sim/src/lineage/deme.rs:85-99` reconstructs expanded cell names
from `compartment_dims` + `dimensions`; a missing dimension hits its defensive
`None => break` and leaves every `I_*` cell at `DemeId(0)`, silently
mis-attributed in the `#[lineage]` line list. The serialized form therefore
carries the discriminant alongside the name rather than dropping generated axes.

**The ADT is the in-memory type; the wire keeps `name` plus an optional
discriminant.** `"name"` stays a plain string and the discriminant is a sibling
field, absent for `User`. Two reasons, one of them binding:

- `ir/golden/` is a **frozen** set, deliberately not regenerated (gh#384), and
  its files are loaded directly as IR by Rust tests (`backend_provenance.rs:27`,
  `gh90_multi_stream_data.rs:44`, `output_view_e2e.rs:23`, …). A required new
  field or a retagged `name` would fail to deserialize there, and §12 already
  records that the frozen set cannot absorb a required IR field. An
  absent-means-`User` discriminant loads every frozen file unchanged. Verified:
  **no file under `ir/golden/` contains a `__` identifier**, so none carries a
  generated axis and none is mislabelled by the default.
- Illegal states become unrepresentable where the mistakes happen — in the
  expander, which mints and consumes these names — without making the wire
  format carry an OCaml sum type.

So the parse direction is total (`name` + optional tag → `dim_name`) and the
serialize direction is a match. `hash_into` writes the discriminant, which is
what re-keys a `via` model; a model with no generated axis is hash-identical
under this change alone and re-keys only from B's `ir_version` bump.

**Generated compartment names (`I__fatal__1`) stay visible** — they are
trajectory columns and scenario-referenceable transition names. The invariant is
scoped to generated _dimension_ names.

**Scope.** This is one slice of gh#111 (indexed references lowered by string
concat), taken because it is small and B's bump pays for it. It is not the
`resolve_indexed_ref` resolver. If the design starts to demand that resolver,
stop and scope gh#111 rather than growing this section.

## 9. Increment D — `prevalence` as safe sugar

D1. New form `prevalence(of = <expr>, among = <expr>)`. Both are ordinary
keyword arguments, so **no new keywords and no new grammar production** are
needed — `prevalence` stays an `EFuncCall` handled where it already is
(`expander.ml:7097`), and `kw_arg_name` (`parser.mly:1406`) already accepts any
identifier. This matters: keyword-ising `prevalence` would break
`tests/fixtures/quantities/quantities_showcase.camdl:53`, which uses it as a
quantity name.

D2. Subset check (§5.2 check 1), with a `visiting` set on the `let` walk.

D3. Matched-collapse check (§5.2 check 2): compare the multiset of binder
dimensions on the spine of each side.

D4. Reject a flow read in `of` or `among` with a named diagnostic (§5.2).

D5. Diagnostics aggregate per source site, not per unrolled stream cell (one
`projected =` line currently produces one error per stream leaf). Share the
mechanism with A3 rather than growing a second one — same key, same flush point.

D6. **Removals.** The single-argument form `prevalence(X)` and the
multi-positional form `prevalence(X1, X2)` both go. The `→ X` migration is
IR-identical (verified on `all_lifecycle.camdl` and `ross_macdonald.camdl`, and
re-verified at HEAD: `prevalence(I)` and bare `I` both lower to
`{"current_pop_sum": ["I_child","I_adult"]}`), so no golden regenerates.

**The diagnostic carries the whole migration inline.** Someone hitting this —
often an agent updating a model it did not write — must not have to open a doc
to fix a one-line change; a hint that only names a doc gets guessed at instead.
Both branches, spelled with the user's own operand:

```text
error[E2xx]: `prevalence(I)` was removed — it returned a count, not a proportion
  17│      projected     = prevalence(I)
    │                      ^^^^^^^^^^^^^
  = note: `prevalence(X)` produced exactly the same IR as `X`; the name promised
          a proportion and returned a stock
  = hint: for the same value, drop the call:
              projected = I
          for an actual proportion, name the denominator:
              projected = prevalence(of = I, among = N_local)
  = see: `camdl docs language-changes` (2026-XX-XX)
```

The multi-positional form gets the same shape with `projected = I + R` and
`prevalence(of = I + R, among = N_local)`.

The trailing `see:` is keyed to the **date**, which is how
`docs/language-changes.md` headings are keyed, so a reader can jump to the
entry. It is deliberately not a version: every alpha build reports
`0.1.0+<hash>`, so a version in the message would be wrong for anyone on a dev
build and would duplicate a fact that belongs in one place.

D6a. **The `docs/language-changes.md` entry is a deliverable of this increment,
not a follow-up.** It is the surface an agent reaches when someone else's model
stops compiling, and it ships offline and version-matched via
`camdl docs language-changes`. Format is the file's existing one — what changed,
migration old → new, the diagnostic. D7's dimension change gets its own
paragraph there rather than a line in a hint, because it is the part that cannot
be mechanically applied.

D7. **The proportion form changes the projection's dimension**, so every
downstream likelihood kwarg must be rechecked — a count into `poisson(rate = …)`
becomes a proportion, which is `E304`. Migration must move such streams to a
probability slot or keep the count form. Unlike D6's removals this is a
modelling decision, not a rewrite: which likelihood the stream should have is a
judgment about the surveillance system, so it cannot be codemodded and must be
called out as manual in the language-changes entry.

## 10. Increment E — the observation-boundary rule

Ships after A–D. Closes gh#478. Where a value is scored against data, pooling a
**population stratum** must be stated.

```camdl
projected = incidence(infection)                              # rejected
projected = sum(a in age, incidence(infection[a]))            # one reporting rate, stated
projected = sum(a in age, rho_a[a] * incidence(infection[a])) # per-stratum — a different model
```

Model dynamics are untouched. The asymmetry is principled: in a force of
infection a bare `I` _is_ the definition — transmission is driven by everyone
infectious, and no second reading exists. In an observation,
`rho * (I_child + I_adult)` asserts one reporting rate across age groups, and
the alternative is a different model with different posteriors.

**Mechanism.** `sum(I)` and bare `I` produce identical IR, so the check cannot
inspect the resolved value. It requires a one-bit tag recording whether a
collapse arose from a bare name or an explicit reduction. Because `let` bindings
are hoisted and their bodies resolved exactly once (`register_hoisted_binding`,
`expander.ml:3060-3066`), **the tag lives on the binding, not the expression** —
otherwise which context resolves first determines the diagnostic, and
record-field evaluation order is unspecified in OCaml.

**Interim warning — decided: fold into Increment E, do not ship separately.** It
was specified to ship with Increment A; A shipped (§6) without it, and on review
that is the right outcome rather than an omission to correct. Its stated purpose
is to give users warning-time before E breaks them, and it cannot serve that
purpose: conjunct (a) requires an **indexed** stream, while both of §12's
Increment-E hits are in un-indexed ones, so the predicate is structurally blind
to the models E will actually break — a point §12 already makes below. Corpus is
0 hits. So it would add a per-site tally mechanism to warn about a case nobody
has, while staying silent on the case everybody has.

Ship the predicate as part of E instead, where the same walk backs a hard error,
and keep the positive-control model below as an E test. Retained here because
the predicate itself is E's, and the two rejected variants are load-bearing
design history.

Non-breaking. Warn when all three hold:

```text
E-warn(stream S):
  B := binders(S)                          # [(var, dim)] from `S[v in d, ...]`
  require B != []                                                     # (a)
  P := inline_lets(projection_expr(S))
  for (v, d) in B where d is a population stratum:
    for each family reference R = (f, idx) in P
        with d in axes(f) and cells(f) > 1:                           # (b)
      p    := position of d in axes(f)     # by name if INamed
      item := (idx == [] ? BARE : idx[p])
      if   item is BARE                     -> WARN                        # (c)
      else                                  -> no warning
```

Conjunct (c) is **only** the bare case, and it is evaluated at the index
position of the stream's axis inside the projection — never over the stream
body. An earlier draft also warned when the item was a _different_ binder bound
by an enclosing `sum`; that fired on
`sum(b in age, I[b]) / sum(b in age, N_local[b])` — the very form Increment E
blesses — and could not be silenced, because reusing the stream's own binder
name is `E283`. The only quiet spelling left was §13's dimension-unsafe
enumeration, so the warning pointed the wrong way.

Evaluating (c) over the stream body rather than the index position goes silent
on the motivating case, because the binder is used in the likelihood:

```camdl
prev[a in age] {
  columns   { time : time, age : dim, prev : count }
  projected = prevalence(of = I, among = N)       # pools age silently
  prev ~ binomial(n = tested, p = rho_a[a] * projected)
}
```

Ship that model as a positive-control test. Corpus with the correct predicate:
**0 hits across 89 indexed streams**, cross-checked by an independent IR-only
method.

**The warning does not preview the migration.** Conjunct (a) requires an indexed
stream, and both of §12's Increment-E hits
(`camdl-book/vignettes/garki/garki.camdl:175`) are in an **un-indexed** one. So
the 0-hit figure says nothing about the models Increment E will break; the
un-indexed case is E's own job, and E280 already covers it for `incidence`.

## 11. Increment F — deferred

Family-valued expressions: `sum(a in age, I[a])` on an `[age, patch]` family
yielding a family over patch. Requires axis rules for all 13 `Ast.expr`
constructors, broadcasting, and a collapse-or-error decision at ~56
scalar-required call sites of `resolve_expr` — the same question as whether bare
names remain legal. One RFC. Blocked on C3.

Not a prerequisite for anything above. `resolve_expr` is scalar in, scalar out
(`expander.ml:3210`); reduction is compile-time unrolling; the IR has 17 scalar
constructors and no family.

## 12. Migration

Measured with a source-level detector built from the compiler's own lexer and
grammar, then confirmed by rewriting each hit into an explicit cell enumeration
and checking for byte-identical IR — 13/13.

**`prevalence(X)` → `X`, or the proportion form**: 20 committed `.camdl` sites
across camdl and siblings, plus two gated spec doctest blocks
(`camdl-language-spec.md:2510`, `:2702`), two parse-only fragments,
`camdl-run-spec.md` §14.1, `camdl-inference-spec.md:1299`, and ~10 test-source
sites. Three `ocaml/golden/errors/e304_*` fixtures exist _because_ `prevalence`
returns a count and must keep the bare form.

**Bare-name pooling into a data column (Increment E): 2 hits**, both
`camdl-book/vignettes/garki/garki.camdl:175`, in a file that does not compile
today for unrelated stale-syntax reasons (`E266`, `E270`, `E272`, `E273`).

The other 11 bare references are in model dynamics and are **not** affected: 5
are `via`-created, 6 are hand-rolled staging chains in rate expressions.

Increments A–D are additive or zero-hit at the corpus level. `ir/golden/` is
frozen and out of scope for regeneration (gh#384); it is also not in the
canonical compact serialization — 5744 pretty-printed lines against 96 compact
ones for the same model — so it cannot absorb a required IR field.

## 13. Known limitations

**Explicit cell enumeration is not dimension-safe.**
`let I_total = I[child] +
I[adult]` silently drops `elderly` if the dimension
gains a level. Only naming and reducing an axis tracks membership changes. This
is also what defeats the subset check (§5.2).

**Multiplicity is preserved.** `I[child] + I[child]` counts twice; no
aggregation path may collect cells into a set.

**Restricted stratification is silent about arity.** With
`stratify(by = risk, only = [S, I])`, `sum(S)` and `sum(R)` collapse different
numbers of axes and nothing says so.

**Residence structure is opaque at the point of use.** A reader of
`sum(a in age, E[a])` cannot see that a stage axis was also collapsed.

## 14. Decisions taken

1. Primitives are derived from observed disease-modelling operations (§2).
2. A bare family name means the total, in all model-dynamics positions.
3. **Population strata** and **residence structure** are the two axis kinds
   (§3); the rule is that residence structure may be omitted from an index and a
   population stratum may not.
4. A stock as an absolute count needs no operator.
5. `prevalence` is **safe sugar for a division** — `of / among`, same value and
   same IR, plus a subset check with stated limits and a matched-collapse check.
   Its single- and multi-argument forms are removed as redundant.
6. **There is no collapsing argument.** An `across = <dims>` keyword was
   specified and dropped: wrapping each side in `sum(v in d, …)` multiplies by
   `|d|` every additive term not carrying `d`, which is the error class the
   operator exists to prevent. `sum` expresses everything it did.
7. **`prevalence` stays head-position** — plain division is a working escape
   hatch, so composability would buy checking on an expressible form rather than
   expressiveness.
8. **`incidence` becomes composable** — the per-stratum-weight-in-a-pooled-sum
   case is otherwise inexpressible.
9. `incidence` lowers to a new `Projection` variant, not a new `Expr`
   constructor; the accumulator becomes per-reference.
10. `sum` is the single collapsing verb. The no-binder form needs no separate
    production: `sum(I)` is a one-element list whose element is the body, and a
    bare compartment name already means the total.
11. Flat multi-binder **replaces** the existing productions rather than being
    added alongside; verified conflict-neutral. Adding a dedicated `sum(family)`
    production on top of it is not — measured, two arbitrarily-resolved
    conflicts.
12. Dimension lookup returns `option`; all 25 sites route through it; no `_exn`
    escape.
13. Unknown dimension is an error; an empty guard is a warned zero, aggregated
    per source site, which requires a `loc` on `ESum`.
14. Named indexing resolves by name; cross-dimension level collisions stay
    legal.
15. Lowering metadata describes lowering at family granularity, is inert in the
    run-identity hash, and has no user-facing annotation.
16. Generated dimension names are labelled, never omitted; generated compartment
    names stay visible.
17. **A generated dimension name is a type, not a `__` prefix** (§8.1, gh#568).
    Supersedes the earlier decisions to reserve `__` at the lexer and to carry
    an inert `generated : bool`. The registry is not split — the discriminant
    moves into the element, so every union consumer is unaffected. It re-keys,
    and rides Increment B's bump rather than paying a second invalidation.
18. **A re-key is not a design constraint.** Where representation quality and
    identity stability conflict, take the representation and schedule the bump
    (`CLAUDE.md`, `4833858f`).
19. Composability (`incidence`) precedes the observation-boundary rule.
20. Reduction weights must be constant over the interval — free of flows, of
    time, **and of state**.
21. The `via` rewrite extends to `quantities`, `interventions`, `events` and
    `reactive_interventions` (C4a), and §8.1 ships after it.
22. **`incidence` stays scoped to observation streams, by an explicit rule**
    (§7, B9) — B5 deletes the dispatch that enforced it by accident. Reporting a
    flow outside an observation is gh#565, not a relaxation of this.
23. The interim warning fires only on a **bare** reference, does not preview the
    un-indexed migration, and therefore **folds into Increment E** rather than
    shipping separately (§10).

## 15. Tests

- Every §4 behaviour, as a red test before its fix.
- `prevalence`: subset violation; matched-collapse violation (`of` collapsing
  `[age]` against `among` collapsing `[age, patch]`); a flow in `of` rejected;
  `prevalence(of = X, among = Y)` byte-identical to `X / Y`.
- `sum(v in d, e)` where `e` does not carry `d` — the dropped-`across` hazard —
  as a documented non-goal, not a silent multiply.
- The same expression compiles identically in `observations`, `quantities`,
  `interventions` and `events` on a `via`-staged compartment (C4a).
- A state-dependent reduction weight is rejected (B3).
- Each `incidence` form in §7 B6, plus the B3 weight restrictions and the B4
  deferrals as located errors.
- Unit-weight `incidence` forms lower unchanged — all 18 `cumulative_flow`
  goldens byte-identical.
- Multi-cadence per-reference accumulation, against
  `tests/fixtures/polio_afp_es_2patch.camdl`.
- `sum(I)` equals the explicit enumeration; `sum(I) == I` when unstratified.
- Flat multi-binder byte-identical to nested, **including guarded cases**.
- `sum(S + I)` and the other rejected arguments → located errors.
- `E283` regression: same-name shadowing is not rerouted to A5.
- `deme.rs` reconstruction still works after §8.1 — cell names round-trip
  through the typed dimension name, with the `#[lineage]` line list unchanged on
  a `via` model. `run_id` **does** move (§8.1 re-keys); the assertion is that it
  moves once, with B's bump, and that goldens regenerate atomically.
- A user identifier spelled `__foo` compiles and cannot collide with a generated
  axis (§8.1) — the case C5's lexer reservation would have rejected outright.
- `incidence` outside an observation stream is a located error naming the
  construct, **including via an indirection**: `let r = incidence(tr)` used in a
  rate or a quantity (B9). The direct-in-a-rate case is not sufficient — it
  already fails today for the wrong reason.
- The §10 indexed-stream positive control.
- Positive controls that must keep passing: bare `I` in a rate;
  `let N = S + E +
  I + R`; the contact-matrix FOI; `E` on a `via erlang`
  compartment pooling its stages; `I[b]` on an age×stage compartment pooling
  stages only; `sum(a in age, w[a] * (I[a] / N_local[a]))`.

## 16. Follow-ups

Named, tracked, not folded in.

- **Increment F**, above.
- **A flow source for `quantities {}` — gh#565.** The concrete instance of the
  cumulative-flow primitive named below. `QuantitySource` is
  `State(Expr) | Observation { stream }`, so cumulative incidence has no correct
  spelling; the workaround `final(N0 - S)` is right only when `S` never refills.
  Measured on a 730-day SIRS with 100-day immunity: true cumulative infections
  52,694 against the workaround's 7,109, a factor of 7.41, and 7,109 reads as a
  plausible 71% attack rate. It compounds through `contrasts {}`, whose operands
  are quantity members. **Increment B does not close this** — B scopes
  `incidence` to observation streams (§7, B9).
- **Block-scoped constructs have no declaration — gh#566.** Three different
  mechanisms enforce scope today: a targeted `E278` arm for
  `observed`/`sum_observed`, a hardcoded name list plus `E290` for the temporal
  reductions, and nothing at all for `incidence`/`prevalence` (which fall
  through to `E100: undeclared function`). B9 needs a fourth check; the audit
  and one scope table are the fix.
- **`observed(s, window =, reduce =)` — gh#567.** `sum_observed` encodes the
  reducer in the function name, and the IR's `Mean`/`Max` variants have no
  surface syntax at all. Cosmetic; no IR change.
- **Composable `prevalence`.** Revisit when a model needs a checked ratio nested
  inside an expression; the escape hatch is plain division.
- **Deme identity should exclude residence structure.** `global_stratum_index`
  (`lineage/deme.rs:152-157`) mixed-radix encodes over every dimension, so on an
  age × 3-stage model `I_child_s1` and `I_child_s2` land in demes 0 and 1 — six
  demes for two populations. A residence stage is not a population, so
  transmission is attributed across categories that do not correspond to groups.
  Scope is narrow: `DemeId` is real only in `lineage/`, both backends pass
  `DemeId(0)`, and lineage is opt-in per transition via `#[lineage]`. The right
  shape is a model-level declaration of which axes define a deme — defaulting to
  population strata, and allowing a model to opt an axis in (immunity status is
  the plausible case). Changing the default alters line-list output for existing
  runs, so it is a separate, versioned change.
- **The cumulative-flow primitive.** Do not reserve a name now. The existing
  `total`/`sum` reservation protects a feature its own hint calls
  cadence-dependent; `integral(...)` already accumulates a stock over continuous
  time. What is missing is cumulative _flow_, to be named when designed.
- **Vector-borne and household-structured shapes** were not enumerated in §2 and
  may stress the primitives differently.

### Independent defects found during review

Each is a different code path with a different trigger, and each gets its own
issue rather than riding along here.

1. **An inline table over an undeclared dimension, used as a forcing `time_dim`,
   compiles clean and emits a forcing with zero knots**
   (`expander.ml:5711,5733`). The Rust side rejects it
   (`compiled_model.rs:2111-2137`), so a compile-time-detectable error is
   deferred to an unlocated runtime one.
2. **`E283` does not catch a sum binder shadowing a global declared name.**
   `sum(gamma in age, gamma * I[gamma])` with `gamma : rate` declared gives
   `error[E100]: undeclared name 'adult'` — a level name surfaced as an
   undeclared identifier.
3. **`E287`'s hint is wrong independently of the `__` leak**
   (`expander.ml:3381-3392`): the "index all dimensions" example puts dimension
   names where level names belong, and when more than one axis is dropped it
   suggests marginalizing an axis the user _did_ index.
4. **A self- or mutually-recursive `let` hangs the compiler** with no output
   (`let a = a + 1.0` → SIGXCPU, zero bytes). Relevant because D2's axis walk
   descends `let` bodies and must carry its own `visiting` set.
5. **A partial index in projection head position gives `E503` naming a
   compartment the user never wrote**, where the rate path gives `E287`
   (`expander.ml:7150-7166` vs `:3378`). The bad diagnostic is currently pinned
   by `ocaml/test/test_compiler.ml:5921`.
