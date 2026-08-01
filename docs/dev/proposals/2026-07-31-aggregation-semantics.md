# Aggregation semantics: source-level structure and lowered representation

Date: 2026-07-31 Status: ready to implement (Increments A–D; Increment E is a
named follow-up RFC) Supersedes: `2026-07-27-stratum-provenance.md` Fixes:
gh#478, gh#488 Related: gh#459, gh#333, gh#487

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

- Lowered axes are never nameable by the user, so any rule of the form "name
  every axis" must count **source-level axes only**.
- `via hyper_erlang` erases its source compartment from `comp_decls` and emits
  flat cells (`I__fatal__1`, …) with no dimension at all, so it cannot be
  handled by anything keyed on the stratification record.

**Hand-rolled staging** — `dimensions { latent_stage = … }` plus
`stratify(by = latent_stage, only = [E])` — is source-level. Public model
structure is public. See §9 for the migration cost this choice buys, which is
larger than it first appears and is accepted deliberately.

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

Independent of every design question below. Each is measured over 326 `.camdl`
across camdl and seven sibling repos (293 parse; 164 compile clean; the rest are
stale-syntax or intentional error fixtures).

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

**A2. Statically empty restricted reduction → warning, not error.** A guard that
selects no levels must stay `Const 0.0` and gain a `W2xx` warning naming the
guard. It must **not** be an error. Measured, two guarded sums exist
(`ocaml/golden/sir_spatial_where.camdl:40`,
`camdl-nigeria-polio/models/nigeria_states_pois.camdl:45`), and both can
legitimately select zero levels: a patch with no in-radius neighbour folds its
coupling term away, which is the correct epidemiology, and the emptiness is
per-outer-index — empty for one patch, non-empty for the others in the same
model. A single-level `patch` does the same, and nigeria-polio already ships
single-state slices.

The compiler already has a policy for this exact situation in the sibling
construct, and it is a warning:

```text
warning[W200]: 'where' guard in transition 'coupling' produced 0 transitions
  = note: The guard `where p != q` filtered all 1 combinations.
```

(`expander.ml:4387-4400`.) The new warning is symmetric to W200. Together with
A1 this is the whole fix: unknown → error, empty-by-guard → warned zero.

**A3. Reject empty index lists.** `I[]` compiles today with IR byte-identical to
bare `I`; it is in no spec, doc, test, or model in any repo. Only the
**compartment read path** needs fixing — `beta[]` is already E299, `C_age[]`
E202, `N_local[]` E299, and `S[]` in stoichiometry E272. **Corpus: 0
occurrences.**

**A4. Unused reduction binder → hard error, distinct names only.** A
`sum(v in dim, body)` where `v` occurs in neither `body` nor its guard silently
multiplies the result by `|dim|`. Four demonstrated shapes:

```camdl
sum(a in age, I)                        # → pop_sum with each cell twice
sum(b in age, I[a])                     # a is the transition binder
sum(a in age, sum(b in age, I[b]))      # outer binder unused
sum(a in age, I[child])
```

Scope it to **distinct names**: the same-name case is already `E283`
(`expander.ml:8066-8078`) with a better message than a generic unused-binder
error would give. **Corpus: 0 of 300 `sum` sites.** Substrate for the check
already exists (`mentions`/`guard_mentions`, `expander.ml:8166-8186`).

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

**A6. Spec correction.** `docs/camdl-language-spec.md:69-70` claims "The
compiler tracks which dimension each index variable belongs to and rejects
mismatches at compile time, not simulation time." It does not (§5). This is
almost certainly why prior design work assumed dimension identity exists. Delete
or qualify it. In the same pass, fix §25.4's and §23's bare-`incidence` examples
(both contradict §12 and both are skipped by the doctest harness) and unskip
those blocks.

## 5. Increment B — dimension identity

This is the prerequisite for Increment E, and it is **not** a `sum` feature.
Today camdl resolves indices **positionally** and has no notion of which
dimension an index belongs to. Three verified consequences:

**Two dimensions may share level names, silently.** `age = [low, high]` and
`risk = [low, high]` compiles clean, so a literal index `I[low]` is not
attributable to a dimension at all.

**Binder order errors surface as synthetic names.**

```camdl
sum(p in patch, sum(a in age, I[p, a]))
  → error[E100]: undeclared name 'I_north_adult'
```

**The named-index form already parses and its names are discarded.** `INamed` is
defined (`ast.ml:62`) and parsed (`parser.mly:697`), then thrown away at
`index_item_to_str` (`expander.ml:2415`) and validated nowhere:

```camdl
I[age = a, patch = p]     # ok
I[patch = p, age = a]     # error[E100]: undeclared name 'I_north_adult'
```

Correct dimension names in the wrong order produce an error naming an identifier
the user never wrote — the precise failure E287 exists to prevent. **This is a
live defect independent of this proposal and should be filed as such.**

B1. Decide the level namespace: either forbid cross-dimension level collisions
with a located error and a migration, or require the named form for
disambiguation. **Decision: forbid collisions.** Positional indexing is the
common spelling and must stay unambiguous; corpus impact is zero (no model
declares colliding levels).

B2. Make `INamed` resolve **by name, not position**, validated against the
target's declared axes. Order-independent; a wrong dimension name gets a located
error naming the axis, not a mangled cell.

B3. Replace positional arity checks with dimension-set checks wherever names are
given. E287 enumerates axes by name and — per §2 — counts **source-level axes
only**, so a compartment that is both age-stratified and `via`-staged reports
`[age]`, never `__onset_stage`.

## 6. Increment C — name-directed aggregation

Additive. Ships after A; independent of B.

**C1. `sum(expr)` — collapse every source-level axis.**

```camdl
sum(I)              # total occupancy, including all lowered cells
sum(I) == I         # when I has no source-level axes
```

This needs **no shapes**: `sum(name)` desugars name-directed over the target's
declared dimensions (`comp_dims` for compartments, `pdims` for indexed
parameters, the table's axes). Verified: adding `SUM LPAREN e = expr RPAREN` to
the grammar produces an **identical menhir conflict set** to baseline — the two
pre-existing conflicts stay two, the `.conflicts` file differs only in state
renumbering.

Resolve the `quantities {}` reservation first: `expander.ml:7714-7718` rejects
`EFuncCall (("total"|"sum"), _)`. The `"sum"` arm is currently unreachable
(`sum` is a lexer keyword), but desugaring `sum(e)` to an `EFuncCall` makes it
reachable and it would fire with "summing a stock over snapshots is
cadence-dependent" — the wrong message. Add a red test pinning the intended
behaviour before the desugaring lands.

**C2. Flat multi-binder reduction.** Pure sugar for nested; gated by a
**byte-identity** test against the nested form, not an equality assertion — the
`ESum` arm's `normalize_expr` fold is order-sensitive and a reassociation would
move trajectories.

```camdl
# from camdl-garki/models/ajura_compound_re.camdl:61
let Nvil[v in village] = sum(a in age, sum(m in imm, sum(k in compound, N[v,a,m,k])))
let Nvil[v in village] = sum(a in age, m in imm, k in compound, N[v,a,m,k])
```

**Corpus: 32 nested sites in 11 files, max depth 3**, all camdl-garki.

**C3. Aggregation applies to all numeric families.** Already true for the binder
form — `sum(a in age, rho_a[a])` and `sum(a in age, w_age[a])` both compile. So
C3 reduces to "C1 accepts parameters and tables too."

**C4. Axis kinds.** Recorded per axis at each lowering site — _not_ as a field
on the stratification record, because `via hyper_erlang` creates none (§2):

```ocaml
type axis_provenance =
  | UserDeclared      of string   (* dimension name *)
  | ViaResidenceStage of string   (* transition *)
  | ViaMixtureBranch  of string   (* transition *)

type marginalization_policy = MustBeNamed | InternalStage
```

Policy derives from provenance; provenance is retained for diagnostics. This
retires the `"__"` string sniff at `expander.ml:6331`, which today misdiagnoses
a user dimension named `__risk` as a staged residence and tells the user to
"name the explicit stage instead (e.g. `S_s1`)" for a compartment that has no
stages.

**No user-facing annotation.** Everything else here adds checking; a
user-writable "this axis needs no explicit aggregation" marker is the one thing
that removes it, and a mislabelled axis would be permanently silent about a real
modelling decision. Only the compiler mints `InternalStage`, from `via`
lowering, where it is true by construction.

## 7. Increment D — composable projection expressions

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

D1. Make stock and flow projections ordinary nodes in the projection-expression
AST. This also retires the bespoke head-position dispatch that produced four
separate silent-wrongs (discarded arguments, single-level sum peeling, an
ignored `where` guard, a non-compiling hint) — all now fixed, but in a dispatch
that should not exist.

D2. Interim warning, non-breaking, shipping with Increment A: warn when an
observation stream is indexed by a source-level dimension **and** its projection
collapses a family carrying that dimension **and** the stream binder is not used
to select the corresponding cell. All three conjuncts are required — measured,
two legitimate camdl-garki regression streams (`vector_cell_fun.camdl:60`,
`vector_cell_gam.camdl:56`) index by `season` without using it, and have no
family for that axis to collapse. **Corpus: 0 true hits** across 87 indexed
streams.

## 8. Lowering metadata

The proposal this supersedes called for an unconditional per-cell manifest in
the IR. Measured, that is the wrong shape.

**Size.** On `tb-household-probe/scale/m_global_6400` (38,400 compartments, 52.1
MB IR) an unconditional manifest is **3.11 MB / 5.97%** empty and 5.07 MB /
9.73% staged — **2.18× the size of the compartments array it annotates**. The
lean IR is the default path for simulate/batch/predict
(`rust/crates/cli/src/util.rs:394` passes `--no-state-grad`), so that is the
operative figure, not the 0.013% seen on gradient-heavy IRs. Per-record cost is
69–131 B empty, +51 B per lowering entry.

**The frozen golden set cannot absorb a required field.** `ir/golden/` is not
regenerated by `make update-golden`, and it is not even in the canonical
serialization — same model, two sets:

```
ir/golden/seir_spatial_5_inference.ir.json      187364 B   5744 lines   (pretty-printed)
ocaml/golden/seir_spatial_5_inference.ir.json   110241 B     96 lines   (canonical compact)
```

So "one interpretation path for downstream tools" is not achieved by adding a
required field; it is defeated by the frozen set on day one.

**Most of it is reconstructible.** The `via erlang` half is derivable from
`compartment_dims` plus the naming convention — verified reconstruction equals
the actual mapping for `seir_age_erlang_via`. The genuine gap is
`via hyper_erlang`, whose source compartment is erased from `comp_decls`.

**Decision.**

1. Repair `build_model_structure` so a `hyper_erlang` family's source
   compartment and branch labels survive. This closes the only real gap.
2. Add an **inert, skip-if-default** provenance tag on the IR's `dimensions`
   entries. Inert means excluded from the run-identity hash, following the
   established pattern (`projection_state_grad` is explicitly inert, pinned by
   `rust/crates/runid/src/ir_hash/tests.rs`). Skip-if-default means a model with
   no lowered axes pays nothing.
3. Reconstruct the erlang mapping on demand rather than storing it.
4. Expose to tooling through a `camdlc render --format` variant **only when a
   consumer is named** — the repo already owns that seam (`model.render.json`),
   and a second required artifact would create a two-file consistency problem.

An `ir/VERSION` bump and the atomic OCaml+Rust+golden update are still required
for (2); that is a deliberate, reviewed golden change, flagged as its own
commit.

## 9. Migration

Measured with an analyzer built against the compiler's own lexer and parser,
cross-checked against `camdlc inspect` (the set of multi-cell families agrees on
all 164 compiling files).

**Increments A–D: zero breakage.** Every Increment A change has 0 corpus hits; C
and D are additive.

**Increment E (bare-name migration): 13 hits in 8 files, 11 of them live.**

| axis kind                           | hits | note                                         |
| ----------------------------------- | ---- | -------------------------------------------- |
| `via`-created (auto-collapse)       | 5    | no edit required                             |
| hand-rolled staging (`MustBeNamed`) | 6    | 3 files ×`latent_stage`, 1 file ×`inc_stage` |
| genuine subpopulation pooling       | 2    | `camdl-book/vignettes/garki/garki.camdl:175` |

Only **2 of 13** are a bare reference over a genuine subpopulation axis. Six are
hand-rolled staging, semantically identical to `via erlang` and spelled by hand
— one of them says so in a trailing comment. The §2 decision to treat
hand-rolled staging as source-level is buying 6 of the 11 edits. **Decision:
keep it** — public model structure is public, and the alternative is a rule
whose behaviour depends on which of two equivalent spellings the author chose —
but ship an E272-style migration hint naming the `via erlang` rewrite, since
that is what those models mean.

`ir/golden/` is frozen and out of scope for regeneration; see gh#384.

## 10. Decisions taken

1. Source-level structure and lowered representation are semantically distinct;
   the justification is representational, not epidemiological.
2. Name-directed sugar ships now; anything requiring expression shapes is one
   project, gated on dimension identity.
3. `sum` is the structural family-reduction verb, in both whole-family and
   binder forms. Verified: zero new menhir conflicts.
4. Flat multi-binder replaces nesting, gated by a byte-identity test.
5. A binder introduces an index variable and nothing else; an unused binder is
   an error, scoped to distinct names because `E283` owns shadowing.
6. An unknown dimension is a hard error; a guard that selects no levels is a
   warned zero, symmetric with `W200`.
7. Empty index lists are rejected, on the compartment path only.
8. Axis provenance is recorded per axis at each lowering site, is
   compiler-internal, and has no user-facing annotation.
9. Hand-rolled staging is source-level, with a migration hint pointing at
   `via erlang`.
10. Lowering metadata is an inert skip-if-default tag plus on-demand
    reconstruction, not an unconditional per-cell manifest.
11. Projection composability precedes any prohibition on implicit pooling.
12. Cross-dimension level collisions become an error (Increment B1).

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
  migration. `sum(a in age, I[a])` on a multi-axis family, the "no unresolved
  quantified axis" body rule, and the final bare-name semantics are all _this_
  RFC — they are one decision, not three. Blocked on Increment B.
- **Named-index resolution defect** (§5) — file independently; it is live today.
- **`__` namespace reservation** — the lexer permits `_` in identifiers
  (`lexer.mll:135`), so reservation requires enforcement plus a migration
  diagnostic. Corpus: 0 identifiers affected. Unbundle from this proposal.
- **E287 hint repair** — beyond the `__` leak, the hint is wrong in other
  respects; and `E237` misdiagnoses a user dimension named `__risk`. Small,
  independent.
- **The cumulative-flow primitive.** Do not reserve a name now. The existing
  `total`/`sum` reservation protects a feature ("summing a stock over
  snapshots") its own hint calls cadence-dependent; `integral(...)` already
  accumulates a stock over continuous time. What is missing is cumulative
  _flow_, to be named when designed.
