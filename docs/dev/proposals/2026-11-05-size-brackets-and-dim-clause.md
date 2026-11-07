---
date: 2026-11-05
status: stages 1 and 2 implemented (ac278266, 8fb21df8, 952e9cf2) except §5; §4's solver is replaced and §5 is decided under 2026-11-06-dimension-checking-as-one-constraint-system.md (§4.4, §9 decision 8); stage 3 is families Stage A
area: DSL parameter declarations, dimensional checking, parameter families
related:
  - 2026-10-30-parameter-families-and-spline-coefficients.md (accepted; this proposal amends its section 2.1 and its E417 size rule)
  - gh#1006 (a unit literal on a parameter silently drops its scale)
---

# Brackets mean size; dimensions move to a `dim` clause

Background the reader is assumed to bring: `docs/camdl-language-spec.md`
sections 2.2 and 2.4 (the dimensional type system and its three tiers), 4.1.1
(bracket dimension annotations) and 4.1.2 (unit literals on `positive` and
`real`), and the accepted parameter-families proposal listed above.

## 1. Problem

The accepted parameter-families proposal declares a sized family by writing the
size in brackets on the type, as Stan does with `vector[6] b;`:

```camdl
parameters {
  b             : real[6] ~ normal(mu = -1.5, sigma = 1.0)   # six cells, b[0] .. b[5]
  b[province]   : real[6]                                    # six cells per province
}
```

That position is already taken. Brackets directly after a parameter kind are
today a dimension annotation (spec section 4.1.1):

```camdl
parameters {
  amplitude : real [1]          # dimensionless
  b_b       : real [1/T]        # a rate-scale spline coefficient, may be negative
  coupling  : positive [P*T^-1] # population per time
}
```

So `real[1]` would mean two different things: "a dimensionless real" under
today's grammar, "a family of one real" under the families grammar. The two
readings parse identically (`[1]` is an integer in both), so no rule based on
the bracket's content can tell them apart. A model written for one reading would
compile under the other and mean something else. That is the worst outcome
available for a modelling language whose output informs public-health decisions,
and it is exactly the case in use: every dimension annotation in the Ebola
province models is `[1]` (section 6).

The clash is not hypothetical friction: the Stage A implementation of the
families proposal stopped on it.

### 1.1 Why families need this now

Families exist to make hierarchical and spline-coefficient priors writable
without hand-naming every cell. The Ebola work needs a shared B-spline
transmission curve with 5 to 6 coefficients, and next a per-province curve
`b[province] : real[6]` whose cells are pooled toward a shared mean,
`coefs = mu_b + sd_b * z[p]`. Spline coefficients that enter a rate additively
carry dimension T⁻¹ and may be negative, which is why the B-spline fixtures
declare them `real [1/T]`. A family of them therefore needs a size and a
dimension on one declaration, and the two cannot share one bracket.

## 2. What exists today

### 2.1 How a parameter gets its dimension

Verified by reading `ocaml/lib/ir/dimcheck.ml` (`param_dim_of_kind`,
`init_params`, and the I300 emission at lines 1494-1507) and
`ocaml/lib/compiler/parser.mly` (`dim_annotation_opt`, `param_unit_opt`, lines
563-575):

1. A kind keyword fixes the dimension: `rate` is T⁻¹, `probability` is 1,
   `count` is P, `instant` and `duration` are T.
2. `positive` and `real` get a fresh dimension variable, solved from how the
   parameter is used (section 4).
3. An explicit annotation overrides both: a bracket (`[1]`, `[P]`, `[T]`,
   `[1/T]`, `[T^-1]`, `[P/T]`, `[P*T^-1]`) or a unit literal (`'ratio`,
   `'count`, `'per_day`, ...). A unit literal on a parameter is read for its
   dimension only (spec 4.1.2); writing both is error E282.
4. If neither the kind, an annotation nor inference determines the dimension,
   the compiler emits I300 ("dimension of parameter 'x' could not be
   determined"). I300 has severity Info: it never blocks compilation, and that
   parameter is simply not dimension-checked.

So annotations are optional today and are never required.

### 2.2 The scale hole in unit literals (gh#1006)

In a model with `time_unit = 'days`, `gamma : positive 'per_year in [0.1, 10.0]`
compiles with no diagnostic, and its IR is byte-identical to the same model
written with `'per_day`. The bounds, prior and any supplied value are read per
day. The spec contradicts itself here: section 4.1.2 documents the form as exact
sugar, and section 2.4 says "you can't write `beta : positive 'per_day`" because
a prior samples in model time units. Section 2.4 has the right reason: a
parameter's value has no scale of its own, so a scale-carrying notation on a
parameter is a place for a silently dropped option.

## 3. Design

Three notations, one job each:

| Notation        | Job                                     | Where                              |
| --------------- | --------------------------------------- | ---------------------------------- |
| `kind[n]`       | size of a parameter family              | parameter declarations             |
| `dim <dim>`     | dimension, no scale                     | `positive` and `real` declarations |
| `'unit` literal | dimension and scale of a concrete value | data, constants, `time_unit`       |

### 3.1 Brackets on the type mean size, and only size

Exactly as in the accepted families proposal: `b : real[6]`,
`b[province] : real[6]`, and every numeric kind accepts the suffix (`rate[4]`,
`probability[3]`). The bracket dimension annotation is removed from the grammar.

### 3.2 The `dim` clause

A dimension is written with the keyword `dim` followed by a dimension expression
over the base dimensions `P` (population) and `T` (time), the same set the
bracket accepted:

```camdl
parameters {
  b       : real[6] dim T^-1 in [-1, 1] ~ normal(mu = 0.0, sigma = 0.01)
  z_r[province] : real dim 1 in [-4, 4] ~ normal(mu = 0.0, sigma = 1.0)
  mu      : positive dim P/T
  lag     : positive dim T
  alpha   : positive dim 1 in [0.5, 1.0]
}
```

The order of a declaration is: name and named axes, kind, size, `dim`, bounds,
prior. `dim` is accepted only on `positive` and `real`, because every other kind
already fixes its dimension; `rate dim T^-1` is refused as redundant, and
`rate dim 1` as contradictory (both under the existing E281 family of messages,
reworded). Accepted spellings: `1`, `P`, `T`, `1/T`, `T^-1`, `P/T`, `P*T^-1`,
and integer powers of `P` and `T` combined with `*` and `/`. The IR is
unchanged: `dim` lowers to the existing `param_dim` field.

`dim` is optional. Section 5 makes it required exactly when inference leaves a
parameter's dimension undetermined.

### 3.3 Unit literals leave parameter declarations

A unit literal on a parameter declaration is refused, with a fix-it naming the
`dim` form: `gamma : positive 'per_year` gives "a parameter has no scale of its
own; it is always in the model's time unit. Write `positive dim T^-1`." This
closes gh#1006 by removing the notation, not by adding a scale-match rule, and
it makes spec sections 2.4 and 4.1.2 agree (4.1.2 is deleted). Unit literals
stay where a scale is real: data, constants, `let` values, `time_unit`.

### 3.4 Every old form is refused, none reinterpreted

The migration must never let an old annotation compile with a new meaning.

- **A size below 2 is refused.** `real[1]` is "a parameter family needs at least
  two cells; a single cell is a scalar, written `real`. If this was a dimension
  annotation, write `real dim 1`." This amends the families proposal's E417,
  which made a size of 1 legal so a model could later be resized; that
  convenience is given up because it is the one size that collides with every
  old `[1]`. The refusal is permanent, not a migration window: a window that
  later re-legalised `real[1]` would silently reinterpret any file not yet
  migrated.
- **A non-integer bracket is refused.** `[1/T]`, `[P]`, `[T^-1]`, `[P*T^-1]`
  after a kind give "dimension brackets were replaced by the `dim` clause; write
  `real dim 1/T`".
- **Fix-its are machine-applicable.** Under `--json-errors` each refusal carries
  the replacement span and text, so an agent or editor can apply it without
  parsing prose.

### 3.5 Two-dimensional families

Hierarchies over two named dimensions (`z[province, age] : real`) are untouched:
named axes stay after the name. A named family with one sized axis
(`b[province] : real[6]`) is already two-dimensional and is the case the Ebola
work needs. A family with two sized axes (`real[6, 3]`, a tensor-product spline
surface) is not part of this proposal or of families Stage A. The grammar reads
a comma-separated size list, so adding it later changes no existing syntax;
until then a second size is refused with "a parameter family has at most one
sized axis".

## 4. Is dimension inference complete?

No, and section 5 depends on fixing that. Verified by reading `dimcheck.ml`
(`unify`, `infer_binop`, `propagate_product`, lines 219-605).

The checker is a two-pass solver over dimension variables with union-find:

1. Bottom-up, `infer` assigns dimensions from leaves and unifies the operands of
   `+`, `-`, `min`, `max` and comparisons.
2. Top-down, `propagate` pushes the expected dimension of a transition rate
   (P·T⁻¹) into the expression, and `propagate_product` flattens a product and
   solves the one unknown factor if exactly one remains.

This is sound for the mismatches it sees, but it is incomplete in a specific
way. When both operands of `*` or `/` are unknown, `infer_binop` returns a fresh
variable that is not linked to either operand, so the constraint "result = left
× right" is discarded. `propagate_product` recovers the case of one unknown
factor and explicitly gives up on two or more (its comment says so). The checker
repeats the transition passes a fixed three times (line 1078, "two rounds
suffices for most models"), so an unknown that is pinned only through a chain of
such products longer than the rounds stays unresolved. Whether any of the I300s
measured in section 5 are such gaps rather than genuinely free dimensions is not
yet known; Stage 1 settles it.

The standard method is Kennedy's dimension inference: dimensions form a free
abelian group, so unification becomes solving a system of linear equations over
the integers, one unknown exponent vector per variable (A. Kennedy, "Dimension
types", ESOP 1994, LNCS 788, pp. 348-362; and _Programming Languages and
Dimensions_, PhD thesis, University of Cambridge, 1996, technical report
UCAM-CL-TR-391, chapter 3). Each product contributes one equation
(`d_result = d_left + d_right` on exponent vectors); Gaussian elimination over
the rationals, followed by an integrality check, solves the whole model at once
and reports, for each parameter, whether its dimension is determined, free, or
contradictory. With two base dimensions the system is small.

The fix is to replace the bottom-up fresh variable for `*`, `/` and integer
powers with an equation, collect every equation, and solve once after both
passes. The existing error codes (E300 to E304) keep their meaning; what changes
is that "undetermined" becomes a fact about the model rather than about the
order of traversal.

## 5. When a dimension is required

**Decided (2026-11-06): I300 stays an Info, with no strict mode.** Under the
complete solver of `2026-11-06-dimension-checking-as-one-constraint-system.md`
five repo models of about 150 keep an I300, and none of them is a mistake: the
model fixes only a pair (`R0 / N` in `sir_two_patch`, `tau · z` in
`prior_only_all`, `kappa · W` in `seir_spatial_5_inference`), or the count-noise
rule leaves the P exponent free by design (`sigma_rel` in
`surveillance_likelihoods`). An undetermined dimension satisfies the model's
equations whatever it is, so an error, or an opt-in mode forcing a `dim` clause,
would demand an arbitrary choice of representative for a pair and an unchecked
declaration for count noise, and would catch no error in practice. Parameters
used only in priors, or not at all, get no I300. A user who wants a dimension
stated writes a `dim` clause, which is then checked. The text below is the
original plan.

`dim` is optional, and the compiler requires it exactly when the complete solver
of section 4 leaves a parameter's dimension free. I300 becomes an error
("dimension of parameter 'lead_days' is not determined by its use; add
`dim ...`") listing the expressions it appears in.

This waits on section 4 deliberately. Turning I300 into an error under the
current solver would demand annotations on parameters whose dimension is in fact
determined, only because the heuristic gave up. That would teach users to
annotate everything, which defeats inference.

Measured on current main (455a2c96), compiling every model with the built
`camdlc` and counting I300 lines:

| Models                                                            | Compile  | With I300 | I300 total |
| ----------------------------------------------------------------- | -------- | --------- | ---------- |
| camdl repo, tracked `.camdl` files (152, error fixtures excluded) | 140      | 21        | 44         |
| ../ebola-bdbv-camdl, all `.camdl` files                           | 8 of 111 | 8         | 30         |

The undetermined Ebola parameters are `phi_split_*`, `phi_lab`, `lead_days`,
`sd_iso` and `s_width`. 103 of the 111 Ebola models do not compile on current
main (error E350: an observation that accumulates a flow does not say what
period each row covers), so their I300 count is not measured. Stage 1 below
re-measures both sets under the complete solver and classifies each remaining
I300 as either genuinely free or a solver gap.

## 6. What this breaks

Dimension annotations in use, counted with `git grep` on main and `grep -r` in
the Ebola repository:

| Where               | Files     | Annotations | Forms                                                                       |
| ------------------- | --------- | ----------- | --------------------------------------------------------------------------- |
| camdl repo          | 14        | 23          | `positive [1]` ×11, `real [1/T]` ×10, `real [1]` ×1, `positive [P*T^-1]` ×1 |
| ../ebola-bdbv-camdl | 18 of 111 | 54          | `real [1]` ×36, `positive [1]` ×18                                          |

Unit literals on parameter declarations (refused by section 3.3) are counted in
Stage 2; the families proposal's own examples use `'ratio` and are amended to
`dim 1` or to no annotation.

Every one of these is caught by section 3.4, none compiles with a new meaning,
and the rewrite is mechanical:

```bash
# [1] -> dim 1, [1/T] -> dim 1/T, [P*T^-1] -> dim P*T^-1; leaves real[6] alone
perl -pi -e 's/\b(real|positive) *\[([^\]]*[A-Za-z][^\]]*|1)\]/$1 dim $2/' model.camdl
```

Many of the Ebola `[1]` annotations can simply be deleted. In
`models/bvd_province_hier3.camdl`, for example, `mu_r`, `tau_r` and `z_r` are
used only in `exp(mu_r + tau_r * z_r[p])`, and `exp` already forces its argument
dimensionless. Whether that holds for every annotated parameter is checked per
file in Stage 2, not assumed.

The IR is unchanged by sections 3.2 to 3.4 (`dim` lowers to `param_dim`).
Section 4 can resolve dimensions that are undetermined today, and resolved
dimensions are written to the IR, so Stage 1 can move goldens; it is landed in
the same IR version bump as families Stage A (IR 0.43) so stored runs re-key
once.

## 7. Stages

1. **Complete dimension inference** (section 4). Golden-moving only where a
   dimension newly resolves; each moved golden is reviewed. Re-measure the I300
   table of section 5.
2. **Surface change** (section 3): the `dim` clause; bracket annotations and
   unit literals on parameters refused with fix-its; I300 an error (section 5);
   spec sections 2.4, 4.1.1 and 4.1.2 rewritten; the 14 repo files migrated.
   Closes gh#1006.
3. **Families Stage A** with `kind[n]` as in the families proposal, size ≥ 2.

Stages 1 and 3 share one IR version bump.

## 8. Alternatives considered

- **Size as an exponent, `real^6`.** No clash and nothing breaks, but it is not
  how Stan users write a size, and it leaves brackets meaning dimension in a
  language where brackets mean index everywhere else.
- **Size before the kind, `[6] real`.** Brackets would mean size or dimension by
  position only.
- **Disambiguate by bracket content** (an integer is a size, `P`/`T` is a
  dimension). Fails on `[1]`, the one form the Ebola models use.
- **Unit literals as the only dimension notation.** No new keyword, but every
  use carries a scale that means nothing on a parameter, so it needs a
  scale-match rule forever and `'ratio` reads oddly for "dimensionless".
- **`of T^-1`.** Reads awkwardly beside bounds (`real of T^-1 in [-4, 4]`) and
  suggests element types, a meaning a future array kind may want.
- **Angle brackets, `real<T^-1>`.** Stan uses `<lower=0>` for bounds, so Stan
  users would misread it.
- **Make I300 an error now.** Rejected until inference is complete (section 5).
