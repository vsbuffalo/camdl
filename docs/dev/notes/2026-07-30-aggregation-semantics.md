# Aggregation semantics: one operation, five spellings

Date: 2026-07-30 Project: camdl Tags: dsl, stratification, aggregation,
observations

Working note for circulation. Everything marked "measured" was compiled against
`ocaml/_build/default/bin/camdlc.exe` at `a387838c` and the output pasted;
anything I did not verify is marked as such.

## The through-line

A stratified compartment is a family of cells. `I` under `stratify(by = age)` is
`I_child` and `I_adult`. Almost every problem in this note is a consequence of
one design fact:

> **Collapsing a family to fewer axes is the only arity-reducing operation in
> camdl that can be spelled by the _absence_ of syntax.**

`sum`, `max`, `mean`, `integral`, `time_of_max`, `sum_observed` are all named
functions. Writing a bare family name is not — and yet it silently produces a
scalar. Every comparable modelling language spells this differently: in Stan,
JAGS, NIMBLE and odin a bare container name denotes the **container**, and
reduction is always a named function. camdl is the outlier, and the outlying
choice is where the silent-wrong bugs live.

There is a second fact that makes the first one harder:

> **`via erlang(...)` and `via hyper_erlang(...)` create axes the modeller never
> declared and cannot name.**

So "name every axis" is not a rule the user can always follow, and any design
that requires naming axes has to say which axes are nameable.

## Part I — The problems

### A. The aggregation surface

**A1. Five spellings for one operation.** All of these collapse a family, and
they do not compose with each other:

```camdl
I                                  # bare name — silent, scalar
sum(a in age, I[a])                # binder form
prevalence(I)                      # head-position sugar, observations only
incidence(infection)               # head-position sugar, transitions
sum_observed(cases, window = 28 'days)   # reactive triggers only
I[]                                # accidental — see A3
```

**A2. Partial marginalization already works, but only nests.** This is measured
and it changes the shape of the problem — I had assumed the operation was
missing. It is not:

```camdl
let N[p in patch] = sum(a in age, S[a, p] + I[a, p])   # compiles: sums age, keeps patch
```

The axes you do not bind stay bound by the enclosing context. What does _not_
work is doing several at once:

```camdl
sum(a in age, m in imm, k in compound, N[v, a, m, k])
  → error[E001]: syntax error
```

so real models write this, from `camdl-garki/models/ajura_compound_re.camdl:61`:

```camdl
let Nvil[v in village] = sum(a in age, sum(m in imm, sum(k in compound, N[v,a,m,k])))
```

Three nested `sum(`s and three closing parens for one idea. That is an
ergonomics defect with a one-line grammar fix, not a missing concept.

**A3. `I[]` is silently accepted.** Measured — byte-identical IR to the bare
name:

```camdl
@ beta * S[a] * I   / N_local[a]    # → pop_sum ["I_child", "I_adult"]
@ beta * S[a] * I[] / N_local[a]    # → identical; transitions compare equal
```

It appears in no spec, no doc, no test, and no `.camdl` in any repo. The parser
accepts it because the index list is a `separated_list`, which admits zero
items, and the arity guard is skipped when the count is zero. Under "no loose
semantics" this is a defect independent of everything else here.

**A4. An unused `sum` binder silently doubles the answer.** Two shapes, both
measured, both compiling clean with zero diagnostics:

```camdl
projected = sum(a in age, I)
  → pop_sum ["I_child", "I_adult", "I_child", "I_adult"]      # 2x

projected = sum(a in age, sum(b in age, I[b]))
  → pop_sum ["I_child", "I_adult", "I_child", "I_adult"]      # 2x
```

In the first the binder `a` never indexes anything, so the bare `I` expands
inside every iteration. In the second every read _is_ indexed — there is no bare
name anywhere — and it still doubles, because the outer binder is unused. The
defect class is **a `sum` binder that never appears in its body**, and it fires
in rates as well as projections. In a force of infection this multiplies
transmission N-fold.

Note the scope precisely: the _same-name_ case is already caught. Measured:

```camdl
projected = sum(a in age, sum(a in age, I[a, north]))
  → error[E283]: sum variable 'a' shadows an enclosing binding of 'a'.
    First-match-wins resolution would silently rebind it (turning a
    per-stratum term into a global sum). Rename the sum variable.
```

So a new unused-binder check needs to cover only the **distinct-name** case;
`E283` already owns shadowing, and with a better message than a generic
unused-binder error would give.

**A5. Dropping _all_ axes is fine; dropping _some_ is an error.** Measured:

```camdl
I           # I is [age, patch] → pop_sum over all four cells, silent
I[child]    # → error[E287]: compartment 'I' has dimensions [age, patch]
            #   but only 1 of 2 were indexed; a partial index has no defined cell
```

The rule the book teaches — "Omitting a dimension always means 'sum over it'" —
is therefore false as stated. It holds for omitting everything and fails for
omitting anything less.

### B. Observations and projections

**B1. The two projection heads disagree on the same model.** Measured, one file,
age-stratified:

```camdl
projected = incidence(infection)
  → error[E280]: … would silently sum all 2 strata … and apply reporting uniformly

projected = prevalence(I)
  → compiles clean → {"current_pop_sum": ["I_child", "I_adult"]}
```

The expander's own comment calls the transition case "symmetric to bare
`prevalence` over a stratified compartment", then checks only the transition
side.

**B2. Indexing the stream does not help, and nothing says so.** Measured:

```camdl
prev[a in age] {
  columns   { time : time, age : dim, prev : count }
  projected = prevalence(I)                 # still bare
  prev ~ poisson(rate = rho_a[a] * projected)
}
  → ✓ no errors, 0 warnings
    prev_child → {"current_pop_sum": ["I_child", "I_adult"]}
    prev_adult → {"current_pop_sum": ["I_child", "I_adult"]}
```

Every stratum row is scored against the pooled total, and the per-stratum
`rho_a[a]` absorbs the mismatch. `incidence` has the identical hole on an
indexed stream — E280 is scoped to un-indexed streams only.

**B3. The heads do not compose, so one real model is inexpressible.** Measured:

```camdl
projected = sum(a in age, incidence(infection[a]))          # ok
projected = rho * sum(a in age, incidence(infection[a]))    # error[E100]: undeclared function 'incidence'
projected = sum(a in age, rho[a] * incidence(infection[a])) # error[E100]
```

`incidence(...)` is legal only in head position. So per-stratum reporting into a
single pooled column — which is exactly the alternative model the whole
aggregation debate exists to surface — cannot be written. Only per-stratum
_rows_, via an indexed stream, which changes the data file too.

**B4–B7 are fixed** (`4dcfe673`, `a387838c`, full `make test` green): the `'?'`
sentinel on three argument shapes, multi-argument `prevalence` dropping
arguments, the ignored `where` guard, and E280's hint printing non-compiling
forms plus the single-level sum peel.

### C. Axis kinds

**C1. Nothing distinguishes a subpopulation from an integration stage.**
`via erlang(stages = 3, …)` splits `E` into `E_s1 … E_s3` for a gamma-shaped
dwell time. Those are not subpopulations — nobody reports "prevalence in latent
stage 2", and no measurement distinguishes two people in stage 2. Pooling them
is the only reading the expression has. But the record `via` lowering builds is
the same record the parser builds for a user's `stratify(...)`, so no consumer
can tell them apart.

**C2. The current workaround is a string sniff.** `expander.ml:6331` tests
whether a dimension name starts with `__`. A user dimension named `__risk` is
misdiagnosed — I reproduced an E237 telling the user that `S` "has a staged
residence" and to "name the explicit stage instead (e.g. `S_s1`)", where `S` has
no stages and no `S_s1`.

**C3. E287 leaks the synthesized name.** On a compartment that is both
user-stratified and staged, the diagnostic prints
`[age, patch,
__progression_stage]` and suggests summing over
`__progression_stage` — an identifier the user never wrote, derived from the
transition name, documented nowhere, and which changes if the transition is
renamed.

**C4. `via hyper_erlang` creates no stratification record at all.** Its branch
stages are flat compartments (`I__fatal__1`, …) with no dimension. Any design
that hangs axis metadata on the stratification record leaves hyper_erlang
unclassified.

### D. Namespace

**D1. `total` and `sum` are both reserved for something that may be the wrong
feature.** `expander.ml:7715` rejects them in `quantities {}` with:

> "summing a stock over snapshots is cadence-dependent; cumulative sums arrive
> with the flow source in a later increment"

Read closely, that is not "we deferred a time-sum" — it is "summing a _stock_
over snapshots is meaningless, because the answer changes with observation
cadence." The real deferred feature is cumulative **flow**. `integral(...)`
already handles continuous accumulation of a stock. So the reservation may be
protecting a feature that should never exist in that form, and the names may be
free.

**D2. The `"sum"` arm of that check appears unreachable.** `sum` is a lexer
keyword, so it can never surface as a function call. Unverified beyond reading
the lexer and grammar; worth a red test either way.

### E. Adjacent defects

**E1. An unknown dimension in a `sum` silently yields `0.0`** — filed as gh#488,
independent of everything else here, and the most dangerous item in this note:

```camdl
@ beta * S[a] * sum(b in aeg, I[b]) / N     # 'aeg' is a typo for 'age'
  → exit 0, zero diagnostics
  → beta * S_child * {"const": 0.0} / N     # transmission is dead
```

`dim_values` returns `[]` for an unknown dimension, which `resolve_expr` cannot
distinguish from a `where` guard that excluded every level. It is also live in
projection position, not just rates.

**The silence is expression-shape-dependent, which is worse than uniform
silence.** The model above divides by a literal, so the folded `0` leaves the
rate dimensionally valid and nothing fires. Divide by a population instead and
dimensional analysis catches the typo _by accident_:

```camdl
@ beta * S[a] * sum(b in aeg, I[b]) / 100.0        → exit 0, 0 diagnostics
@ beta * S[a] * sum(b in aeg, I[b]) / N_local[a]   → error[E300]: rate has wrong dimension
```

Neither diagnostic names the undeclared dimension. So a modeller cannot rely on
the accidental catch, and the shapes that slip through — a dead term divided by
a constant, or sitting among dimensionally-identical addends — are exactly the
ones where the zeroed term is invisible.

**E2. A bare reference to an indexed `let` produces synthetic names.** Measured:

```camdl
let N_local[a in age] = S[a] + I[a]
@ beta * S[a] * I[a] / N_local              # bare, no index
  → error[E100]: undeclared name 'I_a'
  → error[E100]: undeclared name 'S_a'
    hint: check spelling, or add a declaration in compartments/parameters/let/tables
```

Two errors naming identifiers the user never wrote, located inside the `let`
body rather than at the use site, with a hint pointing at correct code. This is
the same failure mode E287 exists to prevent for compartments.

**E3. An explicit cell enumeration goes stale silently.** Measured, `age` with
three levels:

```camdl
let I_total = I[child] + I[adult]           # names two of three
projected  = I_total
  → ✓ no errors, 0 warnings → pop_sum ["I_child", "I_adult"]    # elderly dropped
```

This matters because "the modeller named the cells" is tempting as a safety
criterion, and it is not one — naming cells does not track the dimension. Only
naming the _axis_ does.

### F. Documentation inconsistencies

Verified: the spec documents both sides of the aggregation rule. §25.4 (line
~4993) shows bare `incidence(infection)` on an age-stratified model expanding to
`CumulativeFlowSum(["infection_child","infection_adult"])`, while §12 rejects
that same expression as E280. §23's spatial example (line 4822) uses bare
`incidence(infection)` on a family declared `infection[a in age, p in patch]`,
invalid under §12's own rule. The doctest harness skips both blocks, so CI
catches neither. §8.1 states "Bare names are always global … No
auto-localization, ever" as an invariant. `camdl-run-spec.md` §14.1 defines the
prevalence projection as bare-name → `CurrentPopSum`.

## Part II — What the language could look like

### The core proposal: one verb, three forms, no new punctuation

Everything below is **additive and golden-neutral** — it adds spellings, and
changes no existing model's IR. The question of where the _bare_ form remains
legal is deliberately deferred (Part V).

**Form 1 — collapse every axis.** New. Overloads the existing keyword with a
third grammar production:

```camdl
sum(I)              # total occupancy of I, however stratified or staged
```

**Form 2 — collapse named axes, keep the rest.** Already works today; nothing
changes:

```camdl
let N[p in patch] = sum(a in age, S[a, p] + I[a, p])
```

**Form 3 — collapse several axes at once.** New, and purely a grammar
relaxation. This is the piece that removes the nesting pain:

```camdl
# today
let Nvil[v in village] = sum(a in age, sum(m in imm, sum(k in compound, N[v,a,m,k])))

# proposed
let Nvil[v in village] = sum(a in age, m in imm, k in compound, N[v,a,m,k])
```

The binder stays available throughout, so weighted reductions are unchanged and
still expressible — this is the case no bracket or brace notation can express,
and the reason the binder form must survive:

```camdl
@ beta * S[a] * sum(b in age, C_age[a, b] * I[b] / N_local[b])
```

That is the whole aggregation surface. `sum` is the verb; you either name the
axes or you don't.

**Why not brackets or braces.** I explored `I[+, p]`, `I{age}`, and `I[]` and I
do not think any earns its place. `I[]` already compiles by accident and is an
inverted false cognate — in JAGS, `B[]` denotes the whole _array_, and on a
two-dimensional array it is a compile error, so camdl would mean the opposite
thing with the same characters. `I[+, p]` makes arity total, which is a real
virtue, but `+` is punctuation carrying load-bearing meaning that is easy to
miss when scanning `I[+, p]` against `I[a, p]`. `I{age}` adds a second bracket
type to teach. None of them can express a weighted reduction, so none of them
removes the binder form — they would all be a _fourth_ spelling alongside it,
which is the disease, not the cure.

### Axis kinds

Compiler-internal, recorded per axis at each lowering site — not as a field on
the stratification record, because `via hyper_erlang` creates no such record
(C4):

```
AxisProvenance  = UserDeclared of dimension
                | ViaResidenceStage of transition
                | ViaMixtureBranch of transition

MarginalizationPolicy = MustBeNamed        (* from UserDeclared *)
                      | InternalStage      (* from either Via… *)
```

The justification is representational, not epidemiological: `via` creates
several cells for **one declared compartment**. A modeller who writes
`onset : E --> I via erlang(stages = 3, …)` declared one `E`, so a bare `E` is
the occupancy of that one `E`. Stating it this way avoids resting the design on
a claim about what surveillance systems can observe, which is contextual.

Consequences:

- `InternalStage` axes always collapse, and never count toward arity. So `E[a]`
  on an age×stage `E` is a _complete_ index, not a partial one, and E287 must
  not mention the stage axis (fixes C3).
- Hand-rolled staging (`dimensions { latent_stage = … }` +
  `stratify(by = latent_stage, only = [E])`) is `MustBeNamed`. Public model
  structure is public.
- `sum(I)` collapses both kinds — it is total occupancy.
- gh#460's `"__"` string sniff becomes a predicate over provenance (fixes C2).
- **No user-facing annotation.** Everything else here adds checking; a
  user-writable "this axis needs no explicit aggregation" marker is the one
  thing that removes it, and a mislabelled axis would be permanently silent
  about a real modelling decision.

### Bugs this subsumes

| defect                              | fix                                                                    |
| ----------------------------------- | ---------------------------------------------------------------------- |
| A3 `I[]` accepted                   | reject empty index lists                                               |
| A4 unused `sum` binder doubles      | error when a binder never appears in its body — every context          |
| A5 partial-index asymmetry          | arity counts `MustBeNamed` axes only; the rule becomes statable        |
| C2 `"__"` sniff                     | provenance predicate                                                   |
| C3 E287 leaks `__onset_stage`       | enumerate `MustBeNamed` axes only                                      |
| D1/D2 `total`/`sum` reservation     | re-decide; the deferred feature is cumulative _flow_, not a stock-sum  |
| E2 bare indexed-`let` error cascade | diagnose at the use site, naming `N_local`, not synthesized cell names |

### The flow primitive

If `sum` becomes the axis-aggregation verb everywhere, the deferred temporal
feature needs a name that says _time_. `cumulative(...)` does not — it reads as
"running total" without saying over what. Candidates, none obviously right:

```camdl
cumulative_incidence(infection)     # says what it is, but only fits flows
sum_over_time(I)                    # explicit; matches sum_observed house style
total_flow(infection, window = ...) # flow-specific and windowed
```

I lean toward not naming it yet, because D1 suggests the feature as originally
conceived (summing a _stock_ over snapshots) is cadence-dependent and probably
should not exist. `integral(...)` already accumulates a stock over continuous
time. What is genuinely missing is cumulative _flow_, and that should be named
when it is designed, not reserved speculatively now.

## Part III — Semantic consistency

Walking the surface for cases where two spellings must agree, or where a rule
would produce a surprise.

**Mixed axes.** `E` is `age × __onset_stage`. Then `sum(E)` is total occupancy
across both. `sum(a in age, E[a])` is the same number — the stage axis collapses
automatically inside `E[a]`, because it is `InternalStage`. Two spellings, one
value. Consistent, but note the reader of `sum(a in age, E[a])` cannot see that
a second axis was collapsed. That opacity is inherent to `via` and no notation
here fixes it.

**Restricted stratification.** `stratify(by = risk, only = [S, I])` means `R`
has fewer axes than `S`. `sum(S)` and `sum(R)` then collapse different numbers
of axes. Measured today with bare names and it is unsurprising in practice, but
nothing in either spelling tells the reader the arities differ.

**Multi-axis composition.** `sum(a in age, p in patch, I[a, p])` and
`sum(a in age, sum(p in patch, I[a, p]))` must be identical — the flat form is
pure sugar. Worth a byte-identity test, not just an equality assertion.

**Unstratified compartments.** `sum(I)` where `I` has no axes should be `I`, not
an error — otherwise adding a `stratify(...)` later silently changes which
expressions are legal, and removing one breaks working models.

**Non-compartment operands.** `sum(beta)` where `beta[age]` is declared: today a
bare `beta` is `E100`. Does `sum` generalize to indexed parameters and tables?
If not, it reads as a general reduction with a hidden domain restriction. I
think the answer should be yes for parameters and tables both, but that is scope
creep worth flagging rather than assuming.

**Order and duplicates.** `I[child] + I[child]` resolves to
`pop_sum ["I_child", "I_child"]` today and must keep counting twice. Any
normalization must not collect cells into a set.

## Part IV — Problems I see with this proposal

Listed because they are the parts most likely to be wrong.

**1. `sum(I)` and `sum(a in age, I[a])` are different operations under one
word.** The first is a whole-family reduction with no binder; the second
introduces a variable, ranges over one named dimension, and admits a `where`
guard. A modeller told "`sum(I)` is everything, `sum(a in age, I[a])` is one
axis" will generalize to "so on a two-axis family `sum(a in age, I[a])` sums age
and keeps patch" — which is E287 today. That false bridge exists already, but
naming both `sum` advertises it.

**2. Overloading may be blocked by the existing keyword.** `sum` is a lexer
keyword with exactly two productions, both requiring `IDENT IN IDENT`. Adding
`SUM LPAREN expr RPAREN` looks LR(1)-separable (the lookahead after `IDENT` is
`IN` or not), and one reviewer reports adding it produces zero new menhir
conflicts. **I have not run menhir myself and would not write this into a
proposal without doing so.**

**3. Making the unused-binder case an error may have false positives.** I cannot
think of a legitimate `sum(v in dim, body)` where `v` never appears — the value
is just `|dim|` copies of the body — but I have not proven it, and the check
fires in rate expressions where breakage is most expensive.

**4. Axis kinds are compiler-internal, so Rust cannot see them.** Fine today. If
the viewer, `render`, or a fit diagnostic ever needs "is this axis reportable",
it must re-derive from the `__` prefix — the exact string sniff being removed.

**5. This note deliberately does not decide where bare names stay legal.** That
is the larger question and it should land after, once there is a good thing to
migrate to. But it means the proposal fixes no silent-wrong on its own — it
builds the vocabulary that makes fixing them possible.

**6. `sum` may not be the right verb for total occupancy.** "Sum of I" describes
the arithmetic; "the total in I" describes the epidemiology. A modeller reading
`sum(E)` on a staged compartment may reasonably ask "sum of how many terms?" and
the answer is 6 when they declared 2 levels. A name from the stock vocabulary
would not raise that question. The counterweight is that `sum` costs no new
vocabulary and matches Stan, JAGS, NIMBLE and odin.

## Part V — What I would like outside input on

1. Is `sum` the right verb for both forms, or does whole-family total deserve
   its own word? (Part IV.1 and IV.6 are the tension.)
2. Should the unused-`sum`-binder case be an error, a warning, or left alone?
3. Should `sum` generalize to indexed parameters and tables, or stay
   compartments-only?
4. Is the representational justification for auto-collapsing `via` axes
   airtight, or does `hyper_erlang` — whose branches carry user-written labels
   like `branch(label = fatal, …)` and user-chosen endpoints — count as declared
   structure?
5. Sequencing: does the pooling _policy_ (where bare names stop being legal)
   belong in this note, or after it?
