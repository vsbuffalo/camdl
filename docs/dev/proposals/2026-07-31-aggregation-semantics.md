# Aggregation semantics: source-level structure and lowered representation

Date: 2026-07-31 Status: Increment A ready to implement; Increments B–C ready;
Increment D architecturally approved, implementation-ready once its acceptance
matrix is specified; Increment E is a named follow-up RFC Supersedes:
`2026-07-27-stratum-provenance.md` Fixes: gh#478, gh#488 Related: gh#459,
gh#333, gh#487

Background and measured problem catalogue:
[`docs/dev/notes/2026-07-30-aggregation-semantics.md`](../notes/2026-07-30-aggregation-semantics.md).

## 1. The problem

Collapsing a stratified family to fewer axes is spelled five incompatible ways —
a bare name, `sum(v in dim, …)`, the `prevalence(…)`/`incidence(…)` heads,
`sum_observed(…)`, and the accidental `I[]` — and one of those spellings is the
**absence of syntax**. Every other arity-reducing operation in camdl (`sum`,
`max`, `mean`, `integral`, `time_of_max`) is a named function. In Stan, JAGS,
NIMBLE and odin a bare container name denotes the container and reduction is
always named; camdl is the outlier, and the outlying choice is where the
silent-wrong bugs live.

Four operations are currently conflated: reduction over source-level axes,
observation of a current stock, accumulation of a transition flow, and
aggregation over observation time. Failing to separate them permits models that
compile while silently doubling a rate, pooling observation strata, dropping
cells, or replacing an unknown dimension with zero.

## 2. Source-level structure vs lowered representation

**Source-level axes** are dimensions the modeller declared. They determine the
shape of compartment, parameter, table and transition families.

**Lowered representation** is compiler-generated: `via erlang` residence stages,
`via hyper_erlang` mixture branches. `E[a]` is _one source-level cell_ even when
represented at runtime by `E_child_s1 … E_child_s3`. Combining those cells
realizes the source-level cell; it is not a user-requested marginalization.

The justification is representational, not epidemiological: `via` creates
several cells for **one declared compartment**. Stating it this way avoids
resting the design on a claim about what surveillance systems can observe, which
is contextual.

Two consequences that are load-bearing later:

- Lowered structure is never nameable by the user, so any rule of the form "name
  every axis" must count **source-level axes only**.
- `via hyper_erlang` creates **no dimension at all**. It erases its source
  compartment from `comp_decls` and emits flat cells (`I__fatal__1`, …). Nothing
  keyed on a dimension or a stratification record can describe it — see §8.

**Hand-rolled staging** — `dimensions { latent_stage = … }` plus
`stratify(by = latent_stage, only = [E])` — is source-level. Public model
structure is public. §9 shows this decision accounts for **all** of Increment
E's live migration cost, and §9 states the migration it implies.

## 3. What ships now, and what does not

The single most important structural decision in this proposal:

> **Name-directed sugar ships now. Anything that requires an expression to have
> a shape is one project, and that project begins with dimension identity, not
> with `sum`.**

Verified: no expression in camdl has a shape. `resolve_expr` is `… -> Ir.expr` —
scalar in, scalar out (`expander.ml:3210`). Reduction is compile-time unrolling,
not a value: the `ESum` arm maps `resolve_expr` over the levels and left-folds
an `Add` chain (`expander.ml:3410-3439`). Bare names collapse at the leaf
(`expander.ml:3837-3841`). The IR has 17 scalar constructors and no family
(`rust/crates/ir/src/expr.rs`). There is no AST-level type pass anywhere in the
pipeline.

Therefore a family-valued intermediate is a **new kind of value**, and deciding
what a scalar-required context does with one _is_ the bare-name policy — there
are ~56 consumer call sites of `resolve_expr` across 23 functions. That decision
cannot be smuggled in ahead of the RFC that owns it.

### What this means for two tempting rules

**`sum(a in age, I[a])` on a multi-axis family is a hard error today**, and this
proposal does **not** change that:

```text
projected = sum(a in age, I[a])        # I : [age, patch]
  → error[E287]: compartment 'I' has dimensions [age, patch] but only 1 of 2
    were indexed; a partial index has no defined cell
```

Making it mean "a family over patch" requires shapes. It belongs to Increment E.
Partial marginalization is already expressible where the remaining axes are
bound by context:

```camdl
let N[p in patch] = sum(a in age, S[a, p] + I[a, p])   # compiles today
```

**"The reduction body must retain no unresolved quantified axis"** likewise
requires `axes(e)` for arbitrary `e`, which is the broadcasting rule set of
Increment E. It is not statable now. Note that its motivating example compiles
cleanly today and means something well-defined:

```camdl
let W[b in age] = sum(a in age, C_age[a, b] * I)   # (Σ_a C[a,b]) · I_total
```

## 4. Increment A — safety fixes, no new surface

Independent of every design question below. Measured over 322 `.camdl` across
camdl and six sibling repos (185 compile clean; the rest are stale-syntax or
intentional error fixtures). **These counts are corpus breakage. Each item is
still a breaking language change** — an external user with a model we have never
seen experiences it as one, regardless of the corpus number.

**A1. Unknown dimension in a reduction → hard error.** gh#488, the most
dangerous item here. `dim_values` returns `[]` for an unknown dimension
(`expander.ml:2048-2051`), which `resolve_expr`'s `ESum` arm cannot distinguish
from a guard that excluded every level, so both become `Const 0.0`
(`expander.ml:3419`). Resolving a dimension must fail **before** enumeration, so
"unknown collection" and "known collection, zero survivors" never share a
representation.

The silence is expression-shape-dependent, which is worse than uniform silence:

```camdl
@ beta * S[a] * sum(b in aeg, I[b]) / 100.0        → exit 0, 0 diagnostics
@ beta * S[a] * sum(b in aeg, I[b]) / N_local[a]   → error[E300]: rate has wrong dimension
```

Neither names the undeclared dimension. Dimensional analysis catches it by
accident in the second shape only; a dead term divided by a constant, or sitting
among dimensionally-identical addends, is invisible. **Corpus: 0 of 300 `sum`
sites rely on the current behaviour.**

**A2. Statically empty restricted reduction → aggregated warning, not error.** A
guard that selects no levels must stay `Const 0.0` and gain a `W2xx` warning. It
must **not** be an error. Measured, two guarded sums exist
(`ocaml/golden/sir_spatial_where.camdl:40`,
`camdl-nigeria-polio/models/nigeria_states_pois.camdl:45`), and both can
legitimately select zero levels: a patch with no in-radius neighbour folds its
coupling term away, which is the correct epidemiology.

The compiler already has a policy for this exact situation in the sibling
construct, and it is a warning (`expander.ml:4387-4400`):

```text
warning[W200]: 'where' guard in transition 'coupling' produced 0 transitions
```

**The warning aggregates per source site, never per unrolled instantiation.**
Emptiness is per-outer-index — on a 400-patch model a correct configuration
could emit one warning per isolated patch, which teaches people to ignore
`W2xx`. Required form:

```text
warning[W2xx]: reduction guard selected no levels for 37 of 400 instantiations
  = note: first affected binding: p = island_north
```

Together with A1 this is the whole fix: unknown → error, empty-by-guard → warned
zero.

**A3. Reject empty index lists.** `I[]` compiles today with IR byte-identical to
bare `I`; it is in no spec, doc, test, or model in any repo. Only the
**compartment read path** needs fixing — `beta[]` is already E299, `C_age[]`
E202, `N_local[]` E299, and `S[]` in stoichiometry E272. **Corpus: 0
occurrences.**

**A4. Unused reduction binder → hard error.** A `sum(v in dim, body)` where `v`
occurs in neither `body` nor its guard silently multiplies the result by
`|dim|`. Four demonstrated shapes:

```camdl
sum(a in age, I)                        # → pop_sum with each cell twice
sum(b in age, I[a])                     # a is the transition binder
sum(a in age, sum(b in age, I[b]))      # outer binder unused
sum(a in age, I[child])
```

`E283` already owns **shadowing**, and its scope is broader than nested sums —
it also catches a reduction binder shadowing a transition binder, a stream
index, or an enclosing `let` index (`expander.ml:8066-8078`, applied across
transitions, lets, init, observations, interventions, events and forcing args):

```text
error[E283]: transition 'infection': sum variable 'a' shadows an enclosing
             binding of 'a'. First-match-wins resolution would silently rebind it
```

Because a shadowed binder cannot reach A4 at all, the new check needs only the
**distinct-name** case. Ship a regression test pinning that
`sum(a in age, sum(a in age, I[a]))` continues to give E283 rather than being
rerouted to A4 or accepted after a resolver refactor. **Corpus: 0 of 300 `sum`
sites.** Substrate exists (`mentions`/`guard_mentions`,
`expander.ml:8166-8186`).

**A5. Diagnose a bare reference to an indexed `let` at the use site.** Today:

```camdl
let N_local[a in age] = S[a] + I[a]
@ beta * S[a] * I[a] / N_local
  → error[E100]: undeclared name 'I_a'
  → error[E100]: undeclared name 'S_a'
```

Two errors naming identifiers the user never wrote, located inside the `let`
body rather than at the use site, with a hint pointing at correct code. Emit one
located error naming `N_local` and its required arity.

**A6. Spec correction, using syntax that compiles today.**
`docs/camdl-language-spec.md:69-70` claims "The compiler tracks which dimension
each index variable belongs to and rejects mismatches at compile time, not
simulation time." It does not (§5). This is almost certainly why prior design
work assumed dimension identity exists. Delete or qualify it.

In the same pass fix §25.4 and §23, whose bare-`incidence` examples contradict
§12 and are skipped by the doctest harness, and unskip those blocks. **The
replacement examples must compile against today's grammar** — an indexed stream,
or an explicit per-stratum projection, or an unstratified model. Do not document
Increment D's composable syntax as if it were available.

## 5. Increment B — dimension identity

This is the prerequisite for Increment E, and it is **not** a `sum` feature.
Today camdl resolves indices **positionally** and has no notion of which
dimension an index belongs to.

**What is not a problem.** Cross-dimension level-name collisions are legal and
unambiguous, and this proposal does not change that. Measured:

```camdl
dimensions { age = [low, high]  risk = [low, high] }
I[low, high]     →  I_low_high      # resolves by declared axis order
```

Full positional indexing resolves by the target's axis order; named indexing
resolves by name; a binder carries its dimension. Ambiguity would arise only for
a _partial_ literal index, which is already E287. Forbidding collisions would
pay a permanent expressiveness cost to simplify a case that resolves locally.

**What is a problem.** The named-index form already parses and its names are
discarded — `INamed` is defined (`ast.ml:62`) and parsed (`parser.mly:697`),
then thrown away at `index_item_to_str` (`expander.ml:2415`) and validated
nowhere:

```camdl
I[age = a, patch = p]     # ok
I[patch = p, age = a]     # error[E100]: undeclared name 'I_north_adult'
```

Correct dimension names in the wrong order produce an error naming an identifier
the user never wrote — the precise failure E287 exists to prevent. **This is a
live defect independent of this proposal and should be filed as such.**

B1. Make `INamed` resolve **by name, not position**, validated against the
target's declared axes. Order-independent; a wrong dimension name gets a located
error naming the axis, not a mangled cell.

B2. Replace positional arity checks with dimension-set checks wherever names are
given. E287 enumerates axes by name and — per §2 — counts **source-level axes
only**, so a compartment that is both age-stratified and `via`-staged reports
`[age]`, never `__onset_stage`.

## 6. Increment C — name-directed aggregation

Additive. Ships after A; independent of B.

### C1. `sum(family)` — collapse every source-level axis of a named family

**The grammar production must not be wider than the semantics.** A production
accepting arbitrary `expr` would parse `sum(I + J)`, `sum(beta * I)`,
`sum(prevalence(I))` and `sum(I[a])`, none of which a name-directed desugaring
can give a meaning — which is exactly the class of defect this document exists
to eliminate. The argument is restricted to a **family reference**: a bare
identifier naming a declared family.

```camdl
sum(I)              # compartment family
sum(rho_a)          # indexed parameter
sum(C_age)          # numeric table
sum(N_local)        # indexed let
sum(I) == I         # when I has no source-level axes
```

Supported declaration classes, exhaustively: **compartments** (`comp_dims`),
**indexed parameters** (`pdims`), **numeric tables** (their declared axes), and
**indexed `let` bindings** (their declared index dimensions). Indexed lets are
included deliberately — `sum(N)` over a shaped `let` is the obvious thing to
write, and it is what the camdl-garki files with the deepest nested sums would
reach for. Omitting it would reproduce A5's failure mode on new syntax.

Anything else — an arithmetic expression, an indexed reference, a projection
head — is a **located error naming the argument** and pointing at either the
binder form or Increment E:

```text
error[E2xx]: `sum(...)` takes the name of a stratified family, not an expression
  = note: `S + I` is an expression over two families
  = hint: reduce each family — `sum(S) + sum(I)` — or reduce over an axis
          explicitly with `sum(a in age, S[a] + I[a])`
```

Pin every rejected form with a test. This needs **no shapes**: `sum(name)`
desugars name-directed over the target's declared dimensions.

Verified: adding a `sum` production over a single identifier produces an
**identical menhir conflict set** to baseline — the two pre-existing conflicts
stay two, the `.conflicts` file differs only in state renumbering.

Resolve the `quantities {}` reservation first: `expander.ml:7714-7718` rejects
`EFuncCall (("total"|"sum"), _)`. The `"sum"` arm is currently unreachable
(`sum` is a lexer keyword), but desugaring makes it reachable and it would fire
with "summing a stock over snapshots is cadence-dependent" — the wrong message.
Add a red test pinning the intended behaviour before the desugaring lands.

### C2. Flat multi-binder reduction, with per-binder guards

Sugar for nested. **Guards attach per binder**, because the nested form admits a
`where` at each level and a single trailing guard would not be sugar for it:

```camdl
# nested, today
sum(a in age, sum(p in patch where dist[p,q] < 50, I[a, p]))

# flat, proposed — exactly equivalent
sum(a in age, p in patch where dist[p,q] < 50, I[a, p])
```

Gate with a **byte-identity** test against the nested form, not an equality
assertion — the `ESum` arm's `normalize_expr` fold is order-sensitive and a
reassociation would move trajectories. **The byte-identity suite must include
guarded cases.** Only 2 of 300 corpus sums carry a guard, so an unguarded-only
suite would pass while the sugar claim was false.

The grammar change is larger than C1's and its menhir conflict set has **not**
been verified. Verifying it is a gate item before C2 lands, on the same footing
C1's was.

```camdl
# from camdl-garki/models/ajura_compound_re.camdl:61
let Nvil[v in village] = sum(a in age, sum(m in imm, sum(k in compound, N[v,a,m,k])))
let Nvil[v in village] = sum(a in age, m in imm, k in compound, N[v,a,m,k])
```

**Corpus: 32 nested sites in 11 files, max depth 3**, all camdl-garki.

### C3. Aggregation applies to all numeric families

Already true for the binder form — `sum(a in age, rho_a[a])` and
`sum(a in age, w_age[a])` both compile. C3 reduces to "C1 accepts the four
declaration classes listed above."

### C4. Lowering metadata describes lowering, not axes

`via hyper_erlang` creates no dimension (§2), so an "axis provenance" enum is
the wrong shape — it would be awkward for erlang and simply incorrect for
hyper_erlang. Describe the **lowering** instead, at family granularity:

```ocaml
type lowering =
  | Erlang of {
      source_compartment : string;
      transition         : string;
      stages             : int;
    }
  | HyperErlang of {
      source_compartment : string;
      transition         : string;
      branches           : branch_spec list;
    }
```

This is O(number of `via` declarations), not O(number of cells) — the reason §8
rejects a per-cell manifest. Consumers reconstruct source-to-lowered membership
without parsing generated identifiers.

**The generated naming convention must not be the authoritative
representation.** That would recreate the string-sniff problem at a more formal
boundary. Declared lowering metadata is authoritative; names are derived from
it.

This retires the `"__"` sniff, of which there is exactly **one** site
(`expander.ml:6332`) — so retiring it also fixes the `E237` misdiagnosis that
today tells a user with a dimension named `__risk` that their compartment "has a
staged residence" and to "name the explicit stage instead (e.g. `S_s1`)" for a
compartment with no stages. No separate follow-up is needed.

**No user-facing annotation.** Everything else here adds checking; a
user-writable "this axis needs no explicit aggregation" marker is the one thing
that removes it, and a mislabelled axis would be permanently silent about a real
modelling decision. Only the compiler mints lowering metadata, from `via`
lowering, where it is true by construction.

## 7. Increment D — composable projection expressions

**Status: architecturally approved. Not implementation-ready until the
acceptance matrix below is filled in.** The direction is settled; the surface is
too consequential to implement from two bullets, and two implementers reading
this section today would make different choices.

`incidence(...)` and `prevalence(...)` are legal only in head position, so this
does not compile:

```camdl
projected = sum(a in age, rho[a] * incidence(infection[a]))
  → error[E100]: undeclared function 'incidence'
```

Per-stratum reporting into a single pooled column — precisely the alternative
model the whole aggregation debate exists to surface — is therefore
inexpressible. Only per-stratum _rows_, via an indexed stream, which also
changes the data file.

**Composability must precede any prohibition on implicit pooling.** Making the
bare form an error while the correct rewrite does not compile strands models.

### D1. Acceptance matrix — required before implementation

Specify, for each of these forms, all six columns:

```camdl
incidence(infection[a])
rho[a] * incidence(infection[a])
sum(a in age, incidence(infection[a]))
sum(a in age, rho[a] * incidence(infection[a]))
prevalence(I[a])
rho[a] * prevalence(I[a])
sum(a in age, prevalence(I[a]))
integral(I[a])
```

Columns: where the stream binder is in scope; whether the primitive yields a
stock or an interval flow; where reporting is applied; whether `where` binds
inside or outside the primitive; whether transition indexing resolves before or
after flow projection; the required projection IR node. Plus: flat and nested
reductions stay byte-identical.

Making these ordinary AST nodes also retires the bespoke head-position dispatch
that produced four separate silent-wrongs (discarded arguments, single-level sum
peeling, an ignored `where` guard, a non-compiling hint) — all now fixed, but in
a dispatch that should not exist.

### D2. Interim warning — the predicate, precisely

Non-breaking; ships with Increment A. **Conjunct (c) is evaluated at the index
position of the stream's axis inside the projection expression — never over the
stream body.** That scoping is the whole check:

```text
D2(stream S):
  B := binders(S)                          # [(var, dim)] from `S[v in d, ...]`
  require B != []                                                     # (a)
  P := inline_lets(projection_expr(S))     # as resolve_expr inlines them
  for (v, d) in B where d is source-level:
    for each family reference R = (f, idx) in P
        with d in axes(f) and cells(f) > 1:                           # (b)
      p    := position of d in axes(f)     # by name if INamed
      item := (idx == [] ? BARE : idx[p])
      if   item is BARE                     -> WARN (implicit pool)
      elif item is identifier w:
             w == v                         -> selects the cell; no warning  # (c) false
             w bound by an enclosing sum(w in d, …) in P -> WARN             # (c) true
```

The weak reading — "the stream binder is unused anywhere in the stream" — goes
**silent on this proposal's own motivating bug**, because `rho_a[a] * projected`
uses the binder:

```camdl
prev[a in age] {
  columns   { time : time, age : dim, prev : count }
  projected = prevalence(I)                       # bare — pools all strata
  prev ~ poisson(rate = rho_a[a] * projected)     # binder IS used, here
}
```

**Ship that model as a positive control test.** A "zero hits" figure produced by
a predicate that does not fire on it measures predicate weakness, not corpus
cleanliness.

**Corpus, with the corrected predicate: 0 hits across 89 indexed streams**,
cross-checked by an independent IR-only method (comparing projections between
leaves of the same stream family that differ only on one axis). The predicate
also fires on bare `incidence(infection)` in an indexed stream — a live hole,
since E280 is scoped to un-indexed streams.

Two camdl-garki streams (`vector_cell_fun.camdl:61`, `vector_cell_gam.camdl:57`)
index by `season` without using it and correctly do **not** warn: both models
declare `compartments { }`, so no family carries `season` and conjunct (b)
fails.

## 8. Lowering metadata serialization

The proposal this supersedes called for an unconditional per-cell manifest in
the IR. Measured, that is the wrong shape.

**Size.** On `tb-household-probe/scale/m_global_6400` (38,400 compartments, 52.1
MB IR) an unconditional manifest is **3.11 MB / 5.97%** empty and 5.07 MB /
9.73% staged — **2.18× the size of the compartments array it annotates**. The
lean IR is the default path for simulate/batch/predict
(`rust/crates/cli/src/util.rs:394` passes `--no-state-grad`), so that is the
operative figure, not the 0.013% seen on gradient-heavy IRs.

**The frozen golden set cannot absorb a required field.** `ir/golden/` is not
regenerated by `make update-golden`, and it is not even in the canonical
serialization — same model, two sets:

```
ir/golden/seir_spatial_5_inference.ir.json      187364 B   5744 lines   (pretty-printed)
ocaml/golden/seir_spatial_5_inference.ir.json   110241 B     96 lines   (canonical compact)
```

So "one interpretation path for downstream tools" is not achieved by adding a
required field; it is defeated by the frozen set on day one.

**Most of it is reconstructible.** The `via erlang` mapping is derivable from
`compartment_dims` plus the C4 descriptor — verified reconstruction equals the
actual mapping for `seir_age_erlang_via`. The genuine gap is `via hyper_erlang`,
whose source compartment is erased from `comp_decls`.

**Decision.**

1. Repair `build_model_structure` so a `hyper_erlang` family's source
   compartment and branch labels survive. This closes the only real gap.
2. Serialize the C4 `lowering` descriptor — family-granular, O(via declarations)
   — as an **inert, skip-if-default** IR field. Inert means excluded from the
   run-identity hash, following the established pattern (`projection_state_grad`
   is explicitly inert, pinned by `rust/crates/runid/src/ir_hash/tests.rs`).
   Skip-if-default means a model with no `via` declaration pays nothing. **The
   inert decision holds only while the metadata cannot affect runtime or fitting
   behaviour; if a Rust consumer ever reads it semantically, the hash policy
   must be revisited.** Record that condition next to the field.
3. Reconstruct per-cell membership on demand rather than storing it.
4. Expose to tooling through a `camdlc render --format` variant **only when a
   consumer is named** — the repo already owns that seam (`model.render.json`),
   and a second required artifact would create a two-file consistency problem.

An `ir/VERSION` bump and the atomic OCaml+Rust+golden update are required for
(2); that is a deliberate, reviewed golden change, flagged as its own commit.

## 9. Migration

Measured with a source-level detector built from the compiler's own lexer and
grammar, then confirmed **by the compiler**: rewriting each bare reference into
the parenthesised enumeration of its family's cells yields byte-identical IR,
13/13. A reverse sweep found 0 false negatives. Counting IR `pop_sum` nodes
instead would have returned 38 — the distinction is load-bearing, and an earlier
measurement in this project was wrong for exactly that reason.

**Increments A–D: zero corpus breakage.** Every Increment A change has 0 corpus
hits; C and D are additive. This is a statement about models we can see, not a
compatibility guarantee.

**Increment E (bare-name migration): 13 hits in 8 files.**

| axis kind                          | hits | live | edits |
| ---------------------------------- | ---- | ---- | ----- |
| `via`-created (auto-collapse)      | 5    | 5    | 0     |
| hand-rolled staging (source-level) | 6    | 6    | 6     |
| genuine subpopulation              | 2    | 0    | 0     |
| **total**                          | 13   | 11   | **6** |

The arithmetic: `total = 5 + 6 + 2 = 13`; `dead = 2`; `live = 11`;
`edits = live − via = 6`.

**Both genuine-subpopulation hits are the dead ones** —
`camdl-book/vignettes/garki/garki.camdl:175`, which fails on four stale-syntax
migrations (E266, E270, E272, E273) unrelated to aggregation. So the §2 decision
to treat hand-rolled staging as source-level accounts for **6 of 6 — all** of
Increment E's live migration cost.

Composition of the 6: two camdl goldens, one book teaching file, three in a
single **archived** garki research model. **Zero live production research models
require an edit.**

**The migration hint is `sum(E)`, not "rewrite with `via erlang`."** Rewriting
the declaration changes source-level names, transition identities, outputs, and
`inspect`/`render` structure — too much for a diagnostic to recommend as the
primary fix. Secondary note only, and only where the compiler recognizes the
canonical Erlang chain with high confidence:

```text
error[E2xx]: bare 'E' pools 3 cells of dimension 'latent_stage'
  = hint: state the reduction — `sum(E)`
  = note: if 'latent_stage' only encodes residence stages, `via erlang(...)`
          expresses the same chain and pools automatically
```

**Increment E's scope includes prose, not only model edits.** Both
genuine-subpopulation hits are in camdl-book, and the book plus spec §8.1 ("Bare
names are always global … No auto-localization, ever") actively teach the idiom
E overturns. Budget the documentation rewrite as part of E.

One measure the "bare reference" definition excludes: an _indexed_ reference to
a `via`-staged compartment (`I[a]`) indexes the source-level axes but still lets
the lowered stage collapse implicitly — 2 such references in
`ocaml/golden/seir_age_erlang_via.camdl:25,45`. So the "5 via-created" row
measures only the fully-bare subset of that surface.

`ir/golden/` is frozen and out of scope for regeneration; see gh#384.

## 10. Decisions taken

1. Source-level structure and lowered representation are semantically distinct;
   the justification is representational, not epidemiological.
2. Name-directed sugar ships now; anything requiring expression shapes is one
   project, gated on dimension identity.
3. `sum` is the structural family-reduction verb. Its whole-family form takes a
   **family reference, not an expression**, over four declaration classes
   including indexed `let`s; every other argument is a located error.
4. Flat multi-binder replaces nesting, with **per-binder guards**, gated by a
   byte-identity test whose suite includes guarded cases.
5. A binder introduces an index variable and nothing else; an unused binder is
   an error, scoped to distinct names because `E283` owns shadowing across every
   binding source.
6. An unknown dimension is a hard error; a guard that selects no levels is a
   warned zero, aggregated per source site, symmetric with `W200`.
7. Empty index lists are rejected, on the compartment path only.
8. Cross-dimension level-name collisions stay legal — positional, named and
   binder indexing each resolve unambiguously.
9. Lowering metadata describes **lowering at family granularity**, not axis
   kinds; declared metadata is authoritative and generated names are derived
   from it, never the reverse.
10. Lowering metadata is an inert skip-if-default IR field plus on-demand
    reconstruction, not an unconditional per-cell manifest — with the inertness
    condition recorded.
11. Projection composability precedes any prohibition on implicit pooling, and
    D2's predicate is scoped to the projection's index positions.
12. Hand-rolled staging is source-level; its migration hint is `sum(E)`, with
    `via erlang` as a secondary note only.

## 11. Known limitations

**Explicit cell enumeration is not dimension-safe.** Measured, `age` with three
levels:

```camdl
let I_total = I[child] + I[adult]     # names two of three
  → ✓ no errors → pop_sum ["I_child", "I_adult"]   # elderly silently dropped
```

Shape checking cannot detect this because every operand is scalar. Naming cells
is not a safety property; only naming and reducing an axis tracks future changes
to a dimension's membership. A lint for apparent-exhaustive enumeration may be
considered later.

**Multiplicity is preserved.** `I[child] + I[child]` counts the cell twice and
must continue to. No aggregation path may collect cells into a set.

**Restricted stratification is silent about arity.** With
`stratify(by = risk, only = [S, I])`, `sum(S)` and `sum(R)` collapse different
numbers of axes and nothing in either spelling says so.

**`via` opacity is not fixed here.** A reader of `sum(a in age, E[a])` cannot
see that a stage axis was also collapsed. That is inherent to `via` and no
notation in this proposal changes it.

## 12. Named follow-ups

- **Increment E — the shape RFC.** Family-valued expressions; axis rules for all
  13 `Ast.expr` constructors and the named `EFuncCall` heads; broadcasting; the
  collapse-or-error policy at ~56 scalar-required consumer sites; the bare-name
  migration and its documentation rewrite. `sum(a in age, I[a])` on a multi-axis
  family, the "no unresolved quantified axis" body rule, and the final bare-name
  semantics are all _this_ RFC — one decision, not three. Blocked on Increment
  B.
- **Named-index resolution defect** (§5) — file independently; it is live today.
- **`__` namespace reservation** — the lexer permits `_` in identifiers
  (`lexer.mll:135`), so reservation requires enforcement plus a migration
  diagnostic. Corpus: 0 identifiers affected. Unbundled from this proposal; note
  that C4 already removes the compiler's own dependence on the convention.
- **The cumulative-flow primitive.** Do not reserve a name now. The existing
  `total`/`sum` reservation protects a feature ("summing a stock over
  snapshots") its own hint calls cadence-dependent; `integral(...)` already
  accumulates a stock over continuous time. What is missing is cumulative
  _flow_, to be named when designed.
