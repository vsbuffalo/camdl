# Observation and aggregation primitives

Date: 2026-07-31 Status: Increments A–D ready to implement; Increment E ready
once A–D land; Increment F is a named follow-up RFC Supersedes:
`2026-07-27-stratum-provenance.md` Fixes: gh#488 Partially fixes: gh#478 (closed
by Increment E) Related: gh#459, gh#333, gh#487

## 1. Summary

camdl spells "collapse a stratified family" five incompatible ways, and one of
those spellings is the absence of syntax. Separately, `incidence(...)` and
`prevalence(...)` are legal only as the entire right-hand side of `projected =`,
which makes a common surveillance model inexpressible and has concentrated four
silent-wrong bugs in one dispatch function.

This proposal derives a small set of primitives from the operations real disease
models need, then specifies them:

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

# density-weighted detection (§2.1) — weights go in the numerator
projected = prevalence(of = 0.4 * Y1[a] + 0.9 * Y2[a], among = N_local[a])
```

**What `across` means.** It is a reduction, not an annotation. For each named
dimension the compiler allocates a fresh binder, inserts it at that dimension's
declared position in every family reference in `of` and `among` that carries the
dimension and does not already index it, and wraps each side in
`sum(v in d, …)`. Consequently the index items a reference _does_ carry bind to
the axes **not** named in `across`, in declaration order — `Y1[p]` means `patch`
under `across = age` and `age` without it. Prefer named indexing
(`Y1[patch = p]`, C3) on families with more than one candidate axis. A family in
`of` or `among` that lacks a named dimension entirely is an error, not a
broadcast.

The insertion generalizes `sum_staged_refs` (`expander.ml:1243`) from one named
source appended at the end to any family inserted at position.

**Three checks.**

1. **Axis completeness.** Every population stratum of every family referenced in
   `of` and `among` is either indexed or named in `across`. Residence structure
   is never counted and never nameable (§3).

   ```text
   error[E2xx]: 'Y1' has population strata [age, patch]; 'patch' is indexed but 'age' is not
     = hint: collapse it with `across = age`, or index it
   ```

   Because `across` accounts for the omitted axis, `Y1[p]` in that position is a
   **complete** reference, so the partial-index footgun cannot occur here.

2. **Subset.** `cells(e)` is the set of `Ir.Pop` names reachable in
   `resolve_expr`'s output for `e`, with `BindingRef` resolved through the
   hoisted-binding table (`register_hoisted_binding`, `expander.ml:3060`).
   Requires `cells(of) ⊆ cells(among)`.

   **State the limits honestly.** This rejects a numerator outside its
   denominator (`of = I[child]`, `among = S[child] + R[child]`) and a
   pinned-level mismatch (`of = I[child]`, `among = N_local[adult]`). It is
   **vacuous** when `among` is a parameter or a constant — those have no cells —
   and it is **defeated** when the denominator is written as an explicit cell
   enumeration rather than an indexed family, because that spelling erases the
   axis (§13). `docs/dev/proposals/fixtures/garki_post_proposal.camdl:46,78` is
   exactly that case: a per-age numerator over a hand-enumerated all-ages
   denominator, which passes all three checks while being wrong. Writing the
   denominator as an indexed family (`let N[a in age] = …`) is what makes the
   axis visible to check 1.

3. **Matched collapse.** `across` applies to both sides by construction, so a
   mismatch is **unrepresentable** rather than detected. The equivalent explicit
   form is _checkable_ by comparing binder-dim multisets, but a check can be
   evaded and a construction cannot; that, not checkability, is why `across`
   exists.

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

**Unknown dimension in `across` is a hard error** (A2 is scoped to reductions;
`across` is a new dimension-consuming construct and `dim_values` returns `[]` on
a miss).

`prevalence` gains no other collapsing behaviour — weighted sums, subsets by
predicate and every other reduction stay with `sum`.

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

`sum(name)` takes a **family reference**, not an arbitrary expression, over four
declaration classes: compartments, indexed parameters, numeric tables, indexed
`let` bindings. Anything else is a located error:

```text
error[E2xx]: `sum(...)` takes the name of a stratified family, not an expression
  = hint: reduce each family — `sum(S) + sum(I)` — or reduce over an axis
          explicitly with `sum(a in age, S[a] + I[a])`
```

Without family-valued expressions (§15) there is no principled meaning for
`sum(S + I)`; a production accepting arbitrary `expr` would parse more than the
semantics can define.

### 5.5 Bare names are unchanged

```camdl
let N = S + E + I + R                     # the total population — unchanged
@ beta * S[a] * I / N                     # global force of infection — unchanged
```

## 6. Increment A — safety fixes

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
of weight × flow — appended at hash index 5. `ir/VERSION` 0.30 → 0.31.

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
`sim/tests/per_stream_reset.rs:240`, which asserts 20 where the correct 30-day
AFP bin is 300. `tests/fixtures/polio_afp_es_2patch.camdl` is a live
multi-cadence model of exactly that shape.

The reset stays keyed on the **stream**, never the reference: a reference
appearing twice contributes twice and resets once.

**B3. Weight restrictions.** Each weight must be **flow-free** and
**time-independent**. A flow read inside a weight, or a `Cond`/`TimeFunc`/`t`,
is a located error — `w(t)·ΣΔN ≠ ∫w dN`, and the projection is evaluated once at
the observation instant.

**B4. Deferred with named diagnostics, not silently rejected**: `incidence`
under `Cond`, under a nonlinear function, and mixed with instant state in one
expression (`Σ rho_a·inc_a / N`). The mixed case matters — it would otherwise
pass `gradient_capability.rs:442-456` (which refuses only on an empty
`projection_state_grad`) and silently drop the flow term from the ODE-NUTS
gradient.

**B5. Delete `explicit_incidence_sum`** (`expander.ml:7058-7080`) rather than
extend it. It is the syntactic walker that accumulated the four silent-wrongs in
§4.3; "reach for the existing seam" cuts the other way when the seam is the
defect.

**B6. Acceptance matrix.**

| form                                              | stream binder | kind     | reporting  | `where`                             | indexing                                  | IR node                           |
| ------------------------------------------------- | ------------- | -------- | ---------- | ----------------------------------- | ----------------------------------------- | --------------------------------- |
| `incidence(infection[a])`                         | stream header | Interval | likelihood | n/a                                 | before                                    | `cumulative_flow` — unchanged     |
| `rho[a] * incidence(infection[a])`                | stream header | Interval | projection | n/a                                 | before                                    | `weighted_flow_sum` (1 term)      |
| `sum(a in age, incidence(infection[a]))`          | sum binder    | Interval | likelihood | outside; prunes compile-time domain | before                                    | `cumulative_flow_sum` — unchanged |
| `sum(a in age, rho[a] * incidence(infection[a]))` | sum binder    | Interval | projection | outside                             | before                                    | `weighted_flow_sum` (n terms)     |
| `paralysis_frac * incidence(infection)`           | none          | Interval | projection | n/a                                 | before; still E280 on a stratified family | `weighted_flow_sum` (1 term)      |
| `if season then incidence(a) else incidence(b)`   | —             | —        | —          | —                                   | —                                         | **deferred (B4)**                 |

**B7. `inc_<stream>` stays the raw flow sum.** `incidence_streams()`
(`multi_stream_obs.rs:1026`) builds it as the unweighted `Σ flows[i]`. Under a
weighted projection it diverges from `projected`, which is correct — a modeller
wants true incidence in the trajectory and reported counts in the predictive —
but its doc comment currently claims it is "the model's declared `FlowSum`
projection" and must be restated.

## 8. Increment C — `sum` forms and dimension identity

**C1. `sum(family)`** over the four declaration classes (§5.4). Verified: adding
`SUM LPAREN IDENT RPAREN` produces a conflict set identical to baseline.

Indexed `let`s need a dimension accessor built — compartments have `comp_dims`
(`expander.ml:2057`), tables have `table_dims` (`:2291`), indexed parameters
have `indexed_param_dims` (`:3185`), indexed `let`s have none and must be
reconstructed from `lb.lindices` / `lb.lshape`. The `IConsec` (adjacent-pair)
and `IComp` (compartment-indexed) forms have no dimension and are rejected.

Resolve the `quantities {}` reservation first: `expander.ml:7714-7718` rejects
`EFuncCall (("total"|"sum"), _)`. That arm is unreachable today because `sum` is
a lexer keyword; desugaring makes it reachable and it would fire with "summing a
stock over snapshots is cadence-dependent," the wrong message.

**C2. Flat multi-binder — replace the existing productions, do not add alongside
them.**

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

**C3. Named indexing resolves by name, not position.** Today `INamed` parses
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

**C4. Lowering metadata describes lowering, not axes**, at family granularity —
O(number of `via` declarations), not O(cells):

```ocaml
type lowering =
  | Erlang      of { source_compartment : string; transition : string; stages : int }
  | HyperErlang of { source_compartment : string; transition : string;
                     branches : branch_spec list }
```

Declared metadata is authoritative; generated names derive from it, never the
reverse. This retires the `__` sniff (`expander.ml:6332`) and with it the `E237`
misdiagnosis of a user dimension named `__risk`.

**C5. Reserve the `__` prefix in the lexer.** The C4 tag answers "is this
generated?" but does not prevent a _collision_, because generated axes are
registered in the same `dim_registry` (`expander.ml:1453-1458`). Today:

```camdl
dimensions { __onset_stage = [x1, x2] }
onset : E --> I via erlang(stages = 3, mean = 4 'days)
  → error[E212]: dimension '__onset_stage' is declared more than once in dimensions {}
```

The modeller declared it once. Rejecting `__`-prefixed identifiers at
`lexer.mll:131` makes the collision structurally impossible. Corpus: **zero**
`__`-prefixed identifiers across 6507 `.camdl` files.

**Scope the guarantee honestly.** It covers identifiers passing through the
lexer. It does **not** cover dimension levels read from a data file —
`dimensions { patch = read("patches.tsv", column = "patch") }` with a `__north`
row compiles today and produces `S___north`. Either reject those at
`resolve_dimensions` or document the asymmetry; this proposal takes the former.
It also makes an observation column named `__cases` a lex error — accepted.

A separate namespace for generated axes was considered and rejected on
measurement: most readers of `ctx.stratifies` want the **union** — `comp_dims`
enumerates cells and needs every axis, as do `n_pre` (`:1419`),
`src_is_stratified` (`:1580`) and the E214 check (`:1995`). Splitting would make
the common case worse to serve the rare one.

**C6. Generated dimension names are labelled, never omitted.** They currently
leak: `model_structure.dimensions[].name` and
`model_structure.compartment_dims.<comp>[]` both carry `__recovery_stage`, as do
two `camdlc inspect` lines. `camdlc render` is clean.

**Omitting them is not an option.** `rust/crates/sim/src/lineage/deme.rs:85-99`
reconstructs expanded cell names from `compartment_dims` + `dimensions`; a
missing dimension hits its defensive `None => break` and leaves every `I_*` cell
at `DemeId(0)`, silently mis-attributed in the `#[lineage]` line list. Both
fields are also hashed into run identity (`runid/ir_hash.rs:1092-1094`), so
omission re-keys every `via` model's cached fits.

So: add an inert `generated : bool` to `Dimension` and to the `compartment_dims`
payload, leave `hash_into` untouched so the change is hash-neutral, and filter
at the presentation layer (`inspect`, diagnostics).

Generated **compartment** names (`I__fatal__1`) are intentionally visible — they
are trajectory columns and scenario-referenceable transition names. The
invariant is scoped to generated **dimension** names.

## 9. Increment D — `prevalence` as a checked proportion

D1. New form `prevalence(of = <expr>, among = <expr>, across = <dims>)`, with
`across` optional and taking a dimension name or a bracketed list.

**`across` needs a dedicated grammar production**, as `sum(v in d, …)` has
(`parser.mly:1354`), not a generic keyword-argument slot: dimension names are
`E100` in expression position, `E278` does not cover dimension-vs-compartment
collisions (`dimensions { I = … }` alongside `compartments { S, I, R }` compiles
clean today), and generic expression walkers including `sum_hyper_refs`
(`:1287`) traverse kwarg values.

D2. Axis-completeness check (§5.2), counting **population strata only**.
**Depends on C4 and C5** — without them the implementation is the `__` sniff and
inherits the `E237` misdiagnosis.

D3. Subset check, with the limits stated in §5.2.

D4. Matched collapse via `across`.

D5. Unknown dimension in `across` → hard error.

D6. Diagnostics aggregate per source site, not per unrolled stream cell (one
`projected =` line currently produces one error per stream leaf).

D7. **Removals.** The single-argument form `prevalence(X)` and the
multi-positional form `prevalence(X1, X2)` both go. The `→ X` migration is
IR-identical (verified on `all_lifecycle.camdl` and `ross_macdonald.camdl`), so
no golden regenerates:

```text
error[E2xx]: `prevalence(X)` is the same value as `X`
  = hint: for an absolute count write `Y1[a] + Y2[a]`
          for a proportion write `prevalence(of = Y1[a] + Y2[a], among = N_local[a])`
```

D8. **The proportion form changes the projection's dimension**, so every
downstream likelihood kwarg must be rechecked — a count into `poisson(rate = …)`
becomes a proportion, which is `E304`. Migration must move such streams to a
probability slot or keep the count form.

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

**Interim warning, shipping with Increment A.** Non-breaking. Warn when all
three hold:

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
      if   item is BARE                     -> WARN
      elif item is identifier w:
             w == v                         -> selects the cell; no warning  # (c) false
             w bound by an enclosing sum(w in d, …) in P -> WARN             # (c) true
```

Conjunct (c) is evaluated at the index position of the stream's axis inside the
projection — **never over the stream body**. The weak reading goes silent on the
motivating case, because the binder is used in the likelihood:

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
5. `prevalence` is a **proportion** with an explicit denominator, carrying an
   axis-completeness check, a subset check with stated limits, and a
   matched-collapse construction. Its single- and multi-argument forms are
   removed as redundant.
6. `across` is a reduction that inserts binders by position; every population
   stratum must be indexed or named.
7. **`prevalence` stays head-position** — plain division is a working escape
   hatch, so composability would buy checking on an expressible form rather than
   expressiveness.
8. **`incidence` becomes composable** — the per-stratum-weight-in-a-pooled-sum
   case is otherwise inexpressible.
9. `incidence` lowers to a new `Projection` variant, not a new `Expr`
   constructor; the accumulator becomes per-reference.
10. `sum` is the single collapsing verb, in three forms; the whole-family form
    takes a family reference over four declaration classes.
11. Flat multi-binder **replaces** the existing productions rather than being
    added alongside; verified conflict-neutral.
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
17. `__` is reserved at the lexer, with the file-sourced-level asymmetry closed.
18. Composability (`incidence`) precedes the observation-boundary rule.

## 15. Tests

- Every §4 behaviour, as a red test before its fix.
- `prevalence`: subset violation, matched-collapse violation, axis
  incompleteness naming the missing stratum; the fully-indexed and `across`
  forms produce the ratio the equivalent division produces.
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
- `deme.rs` reconstruction still works after C6, and `run_id` is unchanged for a
  `via` model.
- The §10 indexed-stream positive control.
- Positive controls that must keep passing: bare `I` in a rate;
  `let N = S + E +
  I + R`; the contact-matrix FOI; `E` on a `via erlang`
  compartment pooling its stages; `I[b]` on an age×stage compartment pooling
  stages only; `sum(a in age, w[a] * (I[a] / N_local[a]))`.

## 16. Follow-ups

Named, tracked, not folded in.

- **Increment F**, above.
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
   by `ocaml/test/test_compiler.ml:5920`.
