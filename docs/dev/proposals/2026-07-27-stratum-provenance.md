# No undischarged implicit marginalization

Date: 2026-07-27 Status: **superseded** by
[`2026-07-31-aggregation-semantics.md`](2026-07-31-aggregation-semantics.md)
Fixes: gh#478 Related: gh#459, gh#333, gh#488

Superseded on two counts. Its discharge condition — that naming the cells
satisfies the rule — is unsound: `let I_total = I[child] + I[adult]` on a
three-level `age` passes while silently dropping `elderly`, so the rule accepts
an incorrect explicit enumeration and rejects the correct implicit form. And its
scope is narrower than the defect: the same silent pooling reaches data through
`incidence`, `quantities` and initial-condition right-hand sides, and the
underlying cause is that camdl spells one aggregation operation five
incompatible ways. The successor consolidates those.

## The problem

Stratification turns one declared compartment into several cells. `I` under
`stratify(by = age)` is `I_child` and `I_adult`. Every read of the name `I` must
then resolve to something, and camdl's answer today depends on which side of the
model you are on.

On the **write** side it refuses to guess. A bare stratified name is a hard
error as an initial condition (`E277`), a transition endpoint (`E272`), or an
intervention target (`E265`) — name a cell.

On the **read** side it silently sums. That is defensible in a rate expression,
where a bare `I` in a force of infection means the whole infectious population
by construction and no parameter can absorb a mismatch. It is not defensible in
an observation projection, where the sum is scored against data:

```camdl
observations {
  cases {
    columns   { time : time, cases : count }
    projected = prevalence(I)                    # → I_child + I_adult, silently
    cases ~ poisson(rate = rho * projected)
  }
}
```

That model asserts a single reporting fraction `rho` across age groups. The
alternative, `rho_child * I_child + rho_adult * I_adult`, is a different model
with different estimates and different forecasts. Choosing between them is a
scientific judgment about the surveillance system, and it is currently made by
the compiler, invisibly, always in favour of the first.

Indexing the stream — the first thing a modeller tries — lands somewhere worse.
Measured:

```camdl
prev[a in age] {
  columns   { time : time, age : dim, prev : count }
  projected = prevalence(I)                      # note: still bare
  prev ~ poisson(rate = rho_a[a] * projected)
}
```

```text
✓ no errors, 0 warnings
prev_child → {"current_pop_sum": ["I_child", "I_adult"]}
prev_adult → {"current_pop_sum": ["I_child", "I_adult"]}
```

Every stratum row is scored against the pooled total, with the per-stratum
`rho_a[a]` absorbing the mismatch. So the rule cannot be scoped to un-indexed
streams.

### Why the obvious rule is wrong

**Rejecting the spelling does not work.** These three are the same model:

```text
projected = prevalence(I)               → current_pop_sum [I_child, I_adult]
projected = rho * I                     → derived_expr {mul, pop_sum[I_child, I_adult]}
projected = rho * sum(a in age, I[a])   → byte-identical to the line above
```

A rule that rejects the first and accepts the others rejects a spelling and
accepts its synonym.

**Counting cells in the resolved expression does not work either.** It
over-fires on legitimate models. From a fixture in this repo:

```camdl
slide_positivity[a in age] {
  projected = prevalence(Y1[a] + Y2[a])   → pop_sum ["Y1_child", "Y2_child"]
}
```

Two cells, but two _different families_ in one fully-indexed stratum — "total
parasitaemic," not an aggregation across age. And from a committed golden:

```camdl
let I_total = I[child] + I[adult]
projected  = I_total                      → pop_sum ["I_child", "I_adult"]
```

Two cells of one family across two strata — but the modeller named both. Both
resolve to a bare `PopSum` indistinguishable from `prevalence(I)`.

**And a `let` erases the distinction entirely.** Measured:

```text
let I_tot = I                     ; projected = rho * I_tot
  → {"bin_op": {"mul", "left": {"param":"rho"}, "right": {"binding_ref": "I_tot"}}}

let I_sum = sum(a in age, I[a])   ; projected = rho * I_sum
  → {"bin_op": {"mul", "left": {"param":"rho"}, "right": {"binding_ref": "I_sum"}}}
```

Identical projection IR. The first must be rejected and the second accepted, and
neither the surface syntax of the projection nor its resolved expression can
tell them apart.

The missing information is not _what the expression sums_. It is **whether the
modeller identified the cells or the axis being aggregated.**

## The rule

User-facing:

> An observation projection may not implicitly expand a compartment family
> across a user-declared dimension. Index the family, or reduce over the
> dimension explicitly.

Compiler-facing, and this is the load-bearing form:

> **No undischarged implicit marginalization** reaches an observation boundary.

An _implicit marginalization_ is created at exactly one kind of site: a bare
family name resolving to more than one cell. It is **not** created by an
explicit index (`I[a]`, `I[child]`) or by an explicit reduction
(`sum(a in age, I[a])`). It propagates through arithmetic, conditionals,
wrappers, and `let` expansion, carrying the family, the axes, and the source
span where it arose. At the observation boundary any effect still outstanding
over an `Explicit`-policy axis is `E280`.

Resolution therefore returns the expression _and_ its effect set:

```text
ResolvedExpr {
  expr,
  implicit_marginalizations: [ { family: "I", axes: ["age"], span } ]
}
```

Behaviour, every row measured against the current compiler:

| expression                                    | today                                  | under this rule                |
| --------------------------------------------- | -------------------------------------- | ------------------------------ |
| `projected = I`                               | `pop_sum[I_child, I_adult]`            | **E280**                       |
| `projected = prevalence(I)`                   | `current_pop_sum[…]`                   | **E280**                       |
| `projected = rho * I`                         | `derived_expr{mul, pop_sum[…]}`        | **E280**                       |
| `projected = I / N`                           | pools inside the quotient              | **E280** if either side pools  |
| `let x = I ; projected = x`                   | `binding_ref` — indistinguishable      | **E280**, span at the `let`    |
| `let x = sum(a in age, I[a]) ; projected = x` | `binding_ref` — indistinguishable      | ok                             |
| `projected = sum(a in age, I[a])`             | `pop_sum[I_child, I_adult]`            | ok                             |
| `projected = I[child] + I[adult]`             | `pop_sum[I_child, I_adult]`            | ok                             |
| `projected = sum(a in age, rho_a[a] * I[a])`  | `reduce[…]`                            | ok                             |
| `projected = prevalence(Y1[a] + Y2[a])`       | `pop_sum[Y1_child, Y2_child]`          | ok — two families, one stratum |
| `projected = prevalence(E)`, stages only      | `current_pop_sum[E_s1, E_s2, E_s3]`    | ok — stages are transparent    |
| `projected = sum(a in age, I)`                | `pop_sum[I_c, I_a, I_c, I_a]` ← **2×** | **E280**                       |

That last row is a live silent-wrong the effect formulation catches and a
syntactic "did you name the axis?" check would not. The binder `a` is never
used, so the bare `I` expands inside every iteration and the sum double-counts:

```text
projected = sum(a in age, I)
  → {"derived_expr": {"pop_sum": ["I_child", "I_adult", "I_child", "I_adult"]}}
```

It looks like an explicit reduction and silently returns twice the intended
quantity. Under the effect rule the inner bare `I` creates a marginalization
that the enclosing `sum` does not discharge — the loop variable never indexed
the family — so it is rejected.

Duplicates are otherwise legitimate and must survive: `I[child] + I[child]`
resolves to `pop_sum ["I_child", "I_child"]` and means what it says.

## Axis provenance and marginalization policy

`via erlang(stages = 3, …)` splits a compartment into `E_s1 … E_s3` to give it a
gamma-shaped dwell time. Those cells are not subpopulations, and pooling them is
the only reading the expression can have. Two separate pieces of metadata, both
compiler-internal:

```ocaml
type axis_provenance =
  | UserDeclared      of string   (* dimension name from `dimensions {}` *)
  | ViaResidenceStage of string   (* transition that lowered it *)
  | ViaMixtureBranch  of string   (* `via hyper_erlang` branch axis *)

type marginalization_policy =
  | Explicit                     (* pooling must be stated *)
  | RepresentationTransparent    (* pooling is the meaning *)
```

Policy is derived from provenance — `UserDeclared → Explicit`, both `Via…` forms
→ `RepresentationTransparent` — and provenance is **kept** after deriving it, so
diagnostics can say which axis is which and a future lowering pass cannot
silently reclassify an axis.

The justification is a language fact, not an epidemiological one. `via` lowering
creates several _representation cells_ for **one declared compartment**. A
modeller who writes `onset : E --> I via erlang(...)` declared one `E`, not
three public strata, so a bare `E` means the occupancy of the logical `E` — the
sum of its representation cells. The same argument covers `hyper_erlang`: its
branches are part of the residence law of one declared source compartment. A
model that needs branch-specific observable state should declare explicit
compartments rather than reach into `via`'s internals.

Stating it this way avoids resting the design on a claim about what is
observable in practice, which is contextual: a risk group absent from one
dataset is still a real partition, and retrospective data can classify states
that were not observable prospectively.

**Hand-rolled staging is `Explicit`, deliberately.** A modeller who writes

```camdl
dimensions { latent_stage = [e1, e2, e3] }
stratify(by = latent_stage, only = [E])
```

declared public model structure, so a bare `E` in a projection requires
`sum(s in latent_stage, E[s])`. `via` staging is an abstraction whose lowering
is meant to stay hidden; a hand-written dimension is not.

**Note on where provenance lives.** `via erlang` synthesizes a `stratify_decl`
(`expander.ml:1458`), the same record the parser builds at `parser.mly:1305`;
those are the only two construction sites in the tree. `via hyper_erlang`
synthesizes **no** `stratify_decl` at all — its branch stages are flat
compartments (`I__fatal__1`, …). So provenance is recorded per axis at each
lowering site, not as a field bolted onto `stratify_decl`, or `hyper_erlang`
cells get no policy and default to `Explicit`, breaking a construct that must
keep pooling.

### No user-facing annotation

There is deliberately no `kind = stages` keyword and no `@role(latent)`
dimension attribute. This is protective, not economical. Every other part of
this proposal _adds_ checking; a user-writable "this axis needs no explicit
aggregation" marker is the one piece that _removes_ it, and a mislabelled axis
would be permanently silent about a real modelling decision — the exact failure
the rule exists to catch, carrying the compiler's endorsement. Today only the
compiler can mint `RepresentationTransparent`, from `via` lowering, where it is
true by construction. Defaulting every user-declared dimension to `Explicit` is
the safe direction. Revisit only when a concrete model needs it; tracked as a
follow-up, not a gap.

## Scope: one principle, not a table of exceptions

The rule reads differently in different places, and that difference has to be
principled or it is just arbitrary. The principle:

> **Inside the model's own equations, a bare family name means the whole
> population, by construction. Anywhere a state value crosses out of the
> dynamics — to data, to a report, or into an initial state — you must name what
> you are pooling.**

A bare `I` in a force of infection _is_ the definition of the force of
infection: transmission is driven by every infectious person, no parameter sits
between that sum and anything else, and no alternative reading exists. Nothing
to decide, so nothing to diagnose. The moment the same sum is scored against
data, printed in a report, or used to seed state, a second reading exists and a
parameter or a reader can absorb the difference.

Every context that resolves a state expression, enumerated from the callers of
`resolve_expr` in `expander.ml`, and where it lands:

| context                                         | site                                            | bare user-stratified family                   | omitted `via` axes           |
| ----------------------------------------------- | ----------------------------------------------- | --------------------------------------------- | ---------------------------- |
| transition rates                                | `expand_transitions_counted`                    | global sum — **unchanged**                    | summed                       |
| ODE equations                                   | `expand_ode_equations`                          | global sum — **unchanged**                    | summed                       |
| `let` bindings feeding the above                | `resolve_ident_name`                            | global sum — **unchanged**                    | summed                       |
| observation projection                          | `expand_observations`                           | **E280**                                      | summed                       |
| observation likelihood body                     | `expand_observations`                           | **E280**                                      | summed                       |
| `quantities {}` body                            | `classify_quantity_body`                        | **E280**                                      | summed                       |
| initial-condition RHS                           | `expand_init`                                   | **E280**                                      | summed                       |
| write target (init LHS, endpoint, intervention) | `resolve_action_endpoint`, …                    | already an error — E277 / E272 / E265         | existing staged-target rules |
| reactive trigger                                | `lower_threshold`                               | cannot read raw state — inherits observations | n/a                          |
| time functions / forcing / table literals       | `expand_time_function_one`, `flatten_expr_list` | no compartment reads                          | n/a                          |

Three of those rows deserve a note.

**`quantities {}` is a hard error, not a warning.** A quantity is a number a
human reads and acts on; `max(I)` on an age-stratified `I` is the peak of the
pooled total, which is the peak of no stratum. That it does not enter the
likelihood makes the mistake _visible in output_ rather than _silently biasing a
fit_, which is a real difference — but it is not a reason for a second rule.
Writing `sum(a in age, I[a])` is available and blocks nothing. One rule is worth
more than a severity ladder nobody can remember. Measured breakage: **one**
committed model, `tests/fixtures/quantities/quantities_showcase.camdl` (two
`prevalence` quantities over 3 cells each), which migrates with the fixture.

**Initial-condition right-hand sides read state too**, which is easy to miss
because they usually read only parameters. Today this compiles:

```text
init { I[child] = I }
  → {"I_child": {"pop_sum": ["I_child", "I_adult"]}}
```

a self-referential initial condition defining `I_child` in terms of itself.
Measured breakage across all committed models: **zero** — no init RHS contains a
multi-cell `PopSum`.

**Reactive triggers need no new rule.** `lower_threshold` already requires the
threshold side to be a constant or a parameter (E272) and the other side to be a
`sum_observed(...)` call, so a trigger reads observation streams rather than raw
compartments and inherits the observation semantics automatically.

The write side was already living by this principle — E277, E272 and E265 all
refuse a bare stratified name — so this proposal is not introducing a new kind
of rule. It is making reads behave the way writes already do, everywhere the
value leaves the dynamics.

## Diagnostics

`E280` is retained — same user-facing question, widened domain — and reworded.
It names the family, the axes, and the expansion, and it prints forms that
compile:

```text
error[E280]: observation 'cases' implicitly pools compartment family 'I'
             over model dimension 'age'

  `I` expands here to `I_child + I_adult`.

  = hint: to select this stream's stratum, index the family:
              projected = I[a]
          to pool explicitly, reduce over the dimension:
              projected = sum(a in age, I[a])
          for a per-stratum reporting rate:
              projected = sum(a in age, rho[a] * I[a])

  = note: residence-stage cells created by `via` are pooled automatically and
          are not part of this decision
```

When the effect originated in a `let`, the primary span points at the bare
occurrence in the `let` body, with a secondary note naming the observation that
made it illegal.

The hint suggests the **indexed-variable and `sum` forms only**. It must not
generate named-index fix-its (`I[age = child]`) until gh#459 lands: named labels
are currently lowered by source order and can bind the wrong cell when level
names overlap across dimensions.

On a compartment carrying both policies, the message names only the `Explicit`
axes. A `RepresentationTransparent` axis is never printed as something the user
could index — that would show them `__onset_stage`, an identifier they never
wrote, and the suggested fix would be impossible to follow.

### A second, separate diagnostic for indexed streams

Explicit pooling inside an indexed stream stays legal under the rule above:

```camdl
prev[a in age] {
  projected = sum(b in age, I[b])          # explicit — allowed
  prev ~ poisson(rate = rho_a[a] * projected)
}
```

Every row uses the same pooled latent quantity. That is legitimate when several
reporting channels observe one pooled process with channel-specific
ascertainment, and a mistake more often than not. It gets a warning, not an
error:

```text
warning[W1xx]: observation stream 'prev' is indexed by 'age', but its projection
               does not depend on index variable 'a'
```

### Partial projection index

A genuinely partial index — naming some but not all `Explicit` axes — is `E287`,
the diagnostic the rate path already gives for the same mistake, rather than
`E503` on a mangled name. `E287`'s message and hint are built from the raw
dimension vector today and must be changed to enumerate `Explicit` axes only,
for the reason above. Its catalog row currently scopes it to "a rate read" and
needs rewording.

## What must not break

Positive controls, each pinned by a test:

- `prevalence(E)` on a `via erlang` compartment → pools its stages.
- `prevalence(I)` on a `via hyper_erlang` compartment → pools its branch stages.
- `prevalence(E[a])` on a compartment both user-stratified and staged → pools
  that stratum's stages only.
- A single-level dimension pools nothing (`{"current_pop": "I_main"}`) and
  passes — the effect is created only when a family expands to more than one
  cell.
- `prevalence(Y1[a] + Y2[a])` — two families, one stratum.
- Bare `I` in a rate expression — unchanged.

## Migration and measured breakage

Committed models, all compiled with `ocaml/_build/default/bin/camdlc.exe`:

- **camdl: one stream.** `ocaml/golden/seir_age_let_projection.camdl` pools
  `I_child + I_adult` — and passes, because its `let` names both cells. Of 119
  committed `.camdl`, 94 compile (the rest are error/lint fixtures); exactly
  three observation streams pool more than one cell, and all three are explicit.
- **`quantities {}`: one model.**
  `tests/fixtures/quantities/quantities_showcase.camdl` has two `prevalence`
  quantities pooling 3 cells each; both migrate to an explicit `sum(...)` with
  the fixture.
- **Initial-condition RHS: zero.** No committed model's init RHS contains a
  multi-cell `PopSum`.
- **camdl-book: zero.** Eight models use bare `prevalence(I)`; all unstratified.
- **camdl-garki, camdl-nigeria-polio, camdl-overfit, camdl-vignettes,
  playpen-camdl-measles: zero.** No committed model has an indexed observation
  header followed by a bare projection; every indexed stream indexes its
  projection.
- **Hand-rolled staged residence** exists in 9 files, none of which observes
  prevalence on the staged compartment, so none breaks.
- **Two OCaml tests migrate**: `test_compiler.ml:5679`
  `test_prevalence_on_stratified_compartment` and `:5726`
  `test_projected_bare_stratified_compartment`, both hand-rolled staging, to
  `via erlang`. A third, `test_prevalence_partial_index_is_rejected_at_compile`
  (`:5897`), asserts `E503` and moves to `E287`.

## Tests

- Every row of the behaviour table above, asserted on the IR or the error code.
- `let`-aliased pooling → E280 with the span at the `let`.
- `let`-aliased explicit reduction → compiles.
- `sum(a in age, I)` → E280, not a silently doubled `PopSum`.
- `I[child] + I[child]` → `pop_sum` with the duplicate preserved.
- Each positive control in "What must not break", as a control that passes
  before and after.
- Indexed stream whose projection ignores its index → the new warning, and the
  fit still runs.
- `quantities {}` pooling → E280; `quantities_showcase` migrated to an explicit
  sum still compiles and yields the same numbers.
- `init { I[child] = I }` on a stratified model → E280, not a self-referential
  initial condition.
- A bare `I` in a rate expression and in an ODE equation → **no** diagnostic,
  same IR as today. The scope boundary is itself a test.
- E287 on a mixed compartment names no `__`-prefixed axis.

## Decisions taken

1. **The invariant is "no undischarged implicit marginalization," not a check on
   spelling.** A syntactic check misses `let`-aliasing; a check on the flattened
   IR cannot distinguish `let x = I` from `let x = sum(a in age, I[a])`.
   Resolution carries the effect set until the observation boundary.
2. **One principle generates the scope table**: inside the model's own equations
   a bare family is the whole population by construction; anywhere a value
   crosses out of the dynamics — projection, likelihood, quantity, initial-state
   RHS — it must name what it pools. Hard error in all four, no change to rate
   or ODE semantics, no severity ladder. `quantities {}` is an error rather than
   a warning because a second rule costs more than the one fixture it breaks.
3. **Provenance and policy are separate, both compiler-internal**;
   `RepresentationTransparent` is derived from `via` lowering and provenance is
   retained for diagnostics.
4. **Provenance is recorded per axis at each lowering site**, not as a field on
   `stratify_decl`, because `via hyper_erlang` synthesizes no `stratify_decl`.
5. **The justification for transparency is representational, not
   epidemiological** — `via` creates cells for one declared compartment.
6. **No user-facing annotation.** A user-writable silencer defeats the rule;
   revisit only for a concrete model.
7. **Hand-rolled staging stays `Explicit`.** Public model structure is public.
8. **`prevalence(…)` is a projection-expression wrapper**, accepting any state
   expression, marking "instantaneous state" against `incidence`'s "flow over an
   interval". Multi-argument forms sum.
9. **Keep E280, reworded**; add a separate warning for an indexed stream whose
   projection ignores its index; partial projection index becomes E287.
10. **Named-index fix-its are withheld** from the hint until gh#459.

## Follow-ups

Named, tracked, and deliberately not folded in:

- **gh#488 — `sum()` over an undeclared dimension silently yields `0.0`,** which
  zeroes a force of infection with no diagnostic. Independent of this proposal
  (rate path, not projections) and more dangerous. `dim_values` returns `[]` for
  an unknown dimension, which `resolve_expr`'s `ESum` arm cannot distinguish
  from a guard that excluded every level. The fix is structural: resolving an
  unknown dimension must fail before enumeration, so "unknown collection" and
  "known collection, zero survivors" never share a representation. **Fix this
  first.**
- **Make `incidence(...)` a proper atom in the projection AST.** It is
  head-position sugar today, so `sum(a in age, rho[a] * incidence(tr[a]))` is
  `E100` and per-stratum reporting into a single pooled column is inexpressible
  — the exact alternative model this proposal exists to make visible. The
  bespoke head-position handling is also where four separate silent-wrongs lived
  (discarded arguments, single-level sum peeling, ignored `where`, non-compiling
  hint). A unified projection-expression AST removes the special case.
  Immediately behind this work, not blocking it.
- **Projection evaluation fast path.** A projection that is structurally a sum
  of state cells should not copy the whole state vector per particle per
  observation time (`multi_stream_obs.rs:50-60`, called at `:333`). Build a
  runtime plan at setup — `StreamProjection` is runtime-only with no serde, so
  the model hash is untouched and no run re-keys. Preserve term order and
  duplicates (`I + I` counts twice; never collect into a set). Start by checking
  whether the copy exists only because the evaluator takes ownership — making it
  borrow may speed every derived projection, with sum-planning as an additional
  win. Benchmark allocations and wall time on a national-scale model plus a
  controlled sweep over compartment count, particles and streams. Semantically
  independent; land separately.
- **Spec semantic cleanup.** The language spec currently documents both sides of
  this rule and must carry the context table above. Verified contradictions:
  §25.4 (line ~4993) documents bare `incidence(infection)` on an age-stratified
  model expanding to `CumulativeFlowSum(["infection_child","infection_adult"])`,
  which §12 rejects as E280; and the spatial age-structured example at line 4822
  uses `projected = incidence(infection)` on a family declared
  `infection[a in age, p in patch]`, invalid under §12's own rule. The doctest
  harness skips both blocks, so CI does not catch either.
