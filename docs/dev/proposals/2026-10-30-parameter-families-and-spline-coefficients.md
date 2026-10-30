---
date: 2026-10-30
status: proposed; every design question is decided
area: DSL parameters, indexed expressions, family priors, bspline coefficients, IR family metadata, fit-config and CLI parameter names
related:
  - 2026-10-27-estimable-bsplines.md (D5, section 6.4 smooth, E413 and E414)
  - 2026-10-26-parameter-roles.md (Stage 2; ResolvedProblem)
  - 2026-10-29-method-scoped-settings.md (resolved fit identity)
  - 2026-07-17-ordered-dimensions.md (proposed, not landed; on branch docs/date-interval-semantics)
---

# Parameter families and spline coefficients

Background the reader is assumed to bring: `docs/camdl-language-spec.md` section
4.3 (indexed parameters), section 7 (forcings: `bspline`, `periodic_spline`,
`coefs`, `smooth`), and the proposals above. A _family_ below means a set of
scalar parameters that share one declaration, one source name and one prior
expression; the cells of the family are the individual scalars. An _axis_ is one
index position of a family: either a named dimension (`province`) or the sized
axis introduced here (`real[6]`). A _binder_ is a name that stands for the
current level of an axis, as `p` does in a per-stratum forcing
`beta_t[p in province]`.

## 1. Problem

A cubic B-spline transmission curve with 12 basis functions needs 12 estimated
coefficients. Today the author writes 12 separately named parameters and a
12-element `coefs` list, or declares a family over a named dimension that exists
only to count (`dimensions { mode = [m1, ..., m10] }`). Either way three numbers
must agree: the basis count, the list or dimension length, and the number of
declared parameters. The compiler checks only some of the agreements, and the
reader of the model cannot see from the `parameters` block that twelve cells are
one curve.

The motivating model is the Ebola province analysis: one log-transmission curve
per province, each with six B-spline coefficients, modelled as a _shared
national curve plus province deviations_. The national curve has its own six
coefficients `mu_b[0..5]`; each province's coefficient at position `k` is the
national coefficient at the same position plus a scaled standard-normal
deviation, `mu_b[k] + sd_b * z[p, k]` with `z[p, k] ~ normal(0, 1)`, and the
deviation scale `sd_b` estimated. Provinces with little data are pulled toward
the national trajectory, and provinces with much data follow their own. This is
the _non-centred_ form of the hierarchical model
`b[p, k] ~ normal(mu_b[k],
sd_b)`: the two define the same distribution for the
coefficients, but the non-centred one uses only fixed-argument priors. That
matters here because the Ebola analysis is fitted with PGAS, the default
Bayesian method, and PGAS refuses every hierarchical prior (section 2.5).

The closest thing today's surface can express is:

```camdl
dimensions {
  province = [kinshasa, nord_kivu, ituri]
  knot     = [k0, k1, k2, k3, k4, k5]       # exists only to count
}
parameters {
  mu_b : real     'ratio ~ normal(mu = -1.5, sigma = 1.0)
  sd_b : positive 'ratio ~ half_normal(sigma = 1.0)
  b[province, knot] : real 'ratio ~ normal(mu = mu_b, sigma = sd_b)
}
forcing {
  beta_t[p in province] : bspline 'per_day {
    window = [0 'days, 168 'days]
    n_basis = 6
    link = log
    coefs = [b_kinshasa_k0, b_kinshasa_k1, b_kinshasa_k2,
             b_kinshasa_k3, b_kinshasa_k4, b_kinshasa_k5]
  }
}
```

It has three defects.

1. **It shares no temporal shape.** The scalar `mu_b` is one number for all 18
   cells, so every coefficient of every province shrinks toward the same value.
   The B-spline basis sums to one at every time (language spec section 7, "link
   = log"), so a curve whose coefficients all equal `mu_b` is the constant
   `mu_b`: this prior pools every province toward a _flat_ curve and carries no
   national trajectory. Writing per-position hyperparameters is not possible on
   a family today: `expand_parameters` resolves the prior once per declaration
   and copies the same resolved prior to every cell
   (`ocaml/lib/compiler/expander.ml`, the `PIndexed` branch around lines
   5868-5879; the arguments are resolved with an empty binding environment,
   `resolve_expr ctx [] e`, line 5690), so no cell's prior can depend on which
   cell it is. The only workaround is 18 hand-declared scalars. (The prior shown
   is also hierarchical, so PGAS refuses it.)
2. **Every province receives Kinshasa's coefficients.** Verified by compiling
   this shape with the current compiler: the family is expanded to
   `b_kinshasa_k0 ...` with the prior on every cell, but the forcing is expanded
   once per province from one `coefs` list, so `beta_t_nord_kivu` and
   `beta_t_ituri` silently receive Kinshasa's coefficients. There is no way to
   say "the coefficients of this province".
3. **A counting dimension** stands in for a size, and its length must be kept
   equal to `n_basis` by hand.

The after form (section 3) removes all three.

## 2. Design

### 2.1 Size belongs to the type; brackets after a name mean indexing

All estimated quantities are declared in `parameters`, in one block, so a reader
sees everything the data inform. Spline coefficients are parameters there and
are never declared inside a forcing.

A size is written on the type:

```camdl
parameters {
  b : real[6] 'ratio in [-6, 1] ~ normal(mu = -1.5, sigma = 1.0)
}
```

declares a family of six cells, `b[0]` through `b[5]`, each `real 'ratio` with
that range and prior. Indexing counts from 0 (gh#990, closed: every user-facing
index counts from 0; the language spec states no other base).

Existing named-dimension families keep their syntax,
`z[mode] : real 'ratio ~ ...`, and the two compose: `b[province] : real[6]`
declares six cells per province. In a declaration, bracketed names after the
identifier are the dimensions the family is indexed by; the bracketed integer
after the type is the number of anonymous cells (the sized axis). A family has
at most one sized axis and it always comes last in the index order.

A sized axis is not a dimension: it cannot appear in `stratify`, as a `let`
binder or as a table axis. It exists to count and to be indexed.

The `[N]` suffix is a feature of _parameter declarations only_. It is not a
general array type: `let`, compartments, forcings, tables and observation
declarations do not accept it, and there is no array value at runtime (section
4, "a first-class vector type"). The other numeric kinds accept the suffix
(`positive[N]`, `rate[N]`, and so on) so a size is available to every parameter
kind, not only `real`.

### 2.2 Why the size is on the type, not on the name

The alternative `b[6] : real` was rejected, for three reasons.

1. The same brackets would mean a size when declaring and an index when using.
   With 0-based indexing, `b[6]` in an expression is out of range for a family
   declared `b[6]`, which is the off-by-one trap C programmers know: the
   declaration `int a[6]` names a 6-cell array whose last valid index is `a[5]`.
2. A named-dimension declaration `z[mode]` already uses name-position brackets
   for _which axes_ the family has. Putting an integer there too would make one
   position carry two kinds of content.
3. Stan separates the two. Its current syntax declares sizes on the type, as in
   `array[6] real b;` and `vector[6] b;`, and uses brackets after the name only
   for indexing (Stan Reference Manual, v2.40, chapter "Data Types and
   Declarations", sections "Array types" and "Vector and matrix data types";
   fetched and read 2026-10-30, the quoted examples are verbatim). Stan indexes
   from 1, which is why it does not share the trap; camdl indexes from 0, which
   makes the separation matter more.

A literal per-cell declaration such as `b[1] : real ...` is not valid today:
`param_decl` in `ocaml/lib/compiler/parser.mly` accepts only
`IDENT LBRACKET separated_nonempty_list(COMMA, IDENT) RBRACKET` after the name
(verified, lines 530-544), so a number in that position is a syntax error. This
proposal leaves it one.

### 2.3 Cell names and the diagnostics around sizes

The compiler lowers a family to scalar cells. A named-dimension family already
does so through one producer, `expand_indexed_decl_names` in
`ocaml/lib/compiler/expander.ml`, giving `<base>_<level>`; the lookup tables,
the name-collision check (E278) and `expand_parameters` all use it. A sized axis
adds index suffixes in the same producer: `b : real[6]` yields `b_0 ... b_5`,
and `b[province] : real[6]` yields `b_kinshasa_0 ... b_ituri_5`. These lowered
names are what the scalar IR and the output files contain (section 2.6 makes
membership explicit rather than parsed; section 2.14 states which names the user
may write).

New and reused diagnostics (codes E415 to E420 verified free: the catalog and
`ocaml/lib` stop at E414):

- **E417**, invalid size. The size must be a positive integer literal.
  `real[0]`, `real[-3]`, `real[2.5]` and a size given by an expression are
  refused with "the size of a parameter family must be a positive integer
  literal". A size of 1 is legal (a family with one cell, indexed `b[0]`); it is
  not collapsed to a scalar, so a model can be written uniformly and later
  resized.
- **E278** (existing), name collision. `check_declaration_names` already
  registers every expanded name of every declaration, not only base names
  (`expander.ml`, the doc comment above `check_declaration_names`, around line
  2240), so it covers both collision shapes without a new check:
  family-versus-scalar (`b : real[6]` beside a hand-declared scalar `b_3`, both
  producing `b_3`) and family-versus-family (`b_k : real[3]` and
  `b[g] : real[3]` where `g` has a level `k`, both producing `b_k_0`). Both are
  tested (section 6).

### 2.4 Indexed access and partial indexing

A family is read with brackets, one index per axis, in declaration order. Each
index is one of:

- for a named axis, a _binder_ over that dimension (`p` in a context that binds
  `p in province`) or a _literal level_ (`kinshasa`);
- for the sized axis, an integer literal (`3`) or a sized-axis binder (`k`,
  section 2.5).

A dimension name is never an index. `b[province, 3]` does not say which province
and is refused with the hint "name the level (`b[kinshasa, 3]`) or use a binder
(`b[p, 3]`)".

**Partial indexing.** Indices may be omitted only from the right. The omitted
axes remain in the result, in their declared order. For the family
`b[province, age] : real[6]` (three axes: `province`, `age`, sized 6):

| Reference    | Axes fixed        | Result                                            |
| ------------ | ----------------- | ------------------------------------------------- |
| `b[p, a, 3]` | all three         | a scalar, the cell `b_<p>_<a>_3`                  |
| `b[p, a]`    | `province`, `age` | a 1-D slice: the 6 cells of province `p`, age `a` |
| `b[p]`       | `province`        | a 2-D slice: age by 6 cells for province `p`      |
| `b`          | none              | the whole family, 3-D                             |

Literal levels may stand anywhere a binder does: `b[kinshasa, a, 3]`,
`b[kinshasa]`. Fixing a later axis while leaving an earlier one free (`b[_, a]`)
is not provided.

How each form resolves:

- A full index with constant entries resolves at compile time to the scalar cell
  (`b[kinshasa, adult, 3]` is the parameter `b_kinshasa_adult_3`), so the scalar
  IR is exactly what hand-written scalar names would produce.
- A full index containing a binder is resolved when the enclosing construct is
  expanded per stratum: the expander substitutes the binder's level (or, for a
  sized binder, the integer), then resolves as above. This is the existing rule
  for named axes in transitions and forcings, extended to the sized axis.
- A slice is list-valued. It is accepted only where a list of cells is expected,
  which in this proposal means `coefs` (section 2.8): as the whole `coefs`
  value, or as an operand of element-wise arithmetic inside it, and in both
  cases only when it is exactly one-dimensional. `b` itself is a 1-D slice when
  the family has one axis.

Diagnostics:

- **E418**, index out of range. A constant sized index past the end, or
  negative, is refused with the range stated: "family `b` has 6 cells on its
  sized axis, indexed 0 through 5; `b[kinshasa, adult, 6]` is out of range". A
  sized binder whose range differs from the indexed family's sized axis (`k`
  over 6 cells indexing a 4-cell family) is E418 with both lengths named. A
  named index that is not a level of its dimension keeps its existing
  diagnostic.
- **E419**, the family has no such axis. More indices than the family has axes,
  a sized index on a family with named axes only, a named index in the position
  of the sized axis (or the reverse), or a binder over `age` in the `province`
  position: "family `b` is indexed `[province, age, <6 cells>]`; position 2 is
  the sized axis and takes an integer or a sized binder". Indexing a scalar
  parameter is E419 with "`mu` is a scalar, not a family".
- **E420**, the reference has the wrong shape for its use. A slice used as a
  number ("`b[p]` is a 2-D family slice, not a number; index every axis, as in
  `b[p, a, 3]`"), or a `coefs` slice that is not exactly 1-D ("coefs needs a
  one-dimensional slice; `b[p]` leaves the axes `age` and the 6 cells free;
  write `b[p, a]`"), or element-wise operands of different lengths ("`mu_b` has
  6 cells but `z[p]` has 4; element-wise arithmetic needs equal lengths").

### 2.5 Axis binders in a family declaration

A family's prior may depend on which cell it is drawn for. The axes of the
declaration may therefore carry binders, written with the same binder form
`name in set` that per-stratum forcings already use (`p in province`), where a
size `N` denotes the index set `0 .. N-1`:

```camdl
parameters {
  mu_b : real[6] 'ratio ~ normal(mu = -1.5, sigma = 1.0)
  sd_b : positive 'ratio ~ exponential(rate = 2)
  b[p in province] : real[k in 6] 'ratio ~ normal(mu = mu_b[k], sigma = sd_b)
}
```

Binders are optional and are needed only when the prior refers to them:
`b[province] : real[6]` remains the declaration form when the prior is the same
for every cell. A binder's scope is that declaration's prior arguments, and
nothing else. Binders may be mixed with plain dimension names
(`b[p in province, age] : real[6]`). A binder name that collides with a declared
identifier is refused, using the namespace list of E278, so `mu_b[k]` can never
mean two things.

Inside the prior, references follow section 2.4: `mu_b[k]` is a full index of a
one-axis family and is a scalar; a family over provinces `mu_p[province] : real`
is read `mu_p[p]`; a cell-specific scale `sd_b[k]` is equally legal. A slice in
a prior argument is E420, because a prior argument is a number.

**Why this binder form.** Three forms were considered. An explicit binder on the
type, `real[k in 6]`, reuses the one binder syntax camdl already has, so a
reader who knows `beta_t[p in province]` can read the declaration, and the
tree-sitter grammar and the parser gain one production rather than a keyword
(`in` is already a token; `as` is not). An alternative keyword form,
`real[6 as k]`, reads naturally but adds a keyword used nowhere else. An
implicit positional binder (a reserved name such as `_k` meaning "my position on
the sized axis") needs no syntax but is invisible at the declaration, can
collide with a user's parameter, and has no natural counterpart for the named
axes, so `mu_p[p]` would need a second mechanism. `real[k in 6]` is chosen. It
is consistent with partial indexing because a declaration binder is exactly a
binder in the sense of section 2.4: it stands where a literal level or integer
would, and is substituted before resolution.

**How the expander substitutes it.** Today `classify_and_resolve_prior_spec`
resolves the prior once per declaration with an empty environment, and the
`PIndexed` branch of `expand_parameters` copies the result to every cell
(sections 1 and the line references there). What must be added:

1. The parser records the optional binder on each declaration axis (named and
   sized) in `PIndexed`/`PScalar`.
2. `expand_parameters` iterates over cells together with their axis coordinates,
   and resolves the prior _per cell_ with an environment binding each declared
   binder to that cell's level or integer. When the prior mentions no binder,
   the per-cell results are identical and the IR is unchanged (asserted by test
   1, section 6).
3. `resolve_expr` resolves an indexed reference whose indices are all literals
   after substitution to the lowered cell name (`mu_b[2]` becomes the IR
   reference to the parameter `mu_b_2`), using the same producer as section 2.3.
4. The unknown-parameter check E230 inside `classify_and_resolve_prior_spec`
   (line 5695) builds its list from declaration base names only (`pname`), so a
   reference to a family cell would be reported as unknown. It must check
   against the lowered cell names. (This also makes a hand-written cell
   reference such as `mu = mu_b_kinshasa` legal, which today is E230; verified
   by reading the check.)
5. The cycle check E236 (`check_hierarchical_cycles`, line 10220) works on
   declaration base names; it must map an indexed reference `mu_b[k]` to its
   family `mu_b`. Detection stays at family granularity, which is conservative:
   a family whose prior references another cell of itself is refused, which is
   correct for the scope here (an autoregressive prior over cells is out of
   scope, section 4).

The IR representation is unchanged: each cell already carries its own
`Hierarchical { hkind; hargs; hpool_over }` (`ocaml/lib/ir/ir.ml`, around line
745), and the per-cell `hargs` now differ between cells. The runtime evaluates a
hierarchical prior from its argument expressions, so nothing downstream of the
IR changes. `hpool_over` stays empty for these priors; a pooling clause `| axis`
over a sized axis is not added.

**Which inference methods accept this.** The position-indexed prior above is a
_hierarchical_ prior (its arguments reference parameters), and hierarchical
priors are supported by PMMH and MH but refused by PGAS:
`rust/crates/sim/src/inference/pgas.rs` (around line 3766, gh#175) stops with
"PGAS does not support hierarchical priors". The reason in the code is that the
NUTS gradient for a hierarchical prior is not implemented (it is stubbed to
negative infinity), so the log-posterior would be negative infinity everywhere
and the chain would freeze at its start while looking well mixed; refusing is
the honest alternative. Implementing that gradient is the subject of
`2026-06-04-hierarchical-priors-gate3-pgas-nuts.md` and is out of scope here.
Even once it exists, the centred form is a poor target for NUTS: when the data
say little about a province, the posterior of `(sd_b, b[p, k])` has the "funnel"
shape (the spread of `b` collapses as `sd_b` goes to zero), which a fixed step
size cannot traverse. The non-centred form below removes the funnel and is the
form recommended for PGAS.

**The non-centred form.** camdl has no built-in reparameterisation of
hierarchical priors (verified: no non-centred transform in `ocaml/lib` or
`rust/crates`; the language spec lists a reparameterisation file only as future
work). The non-centred model is written directly, with fixed-argument priors and
the arithmetic in `coefs`, using the element-wise list arithmetic of section
2.8:

```camdl
parameters {
  mu_b : real[6] 'ratio ~ normal(mu = -1.5, sigma = 1.0)
  sd_b : positive 'ratio ~ exponential(rate = 2)
  z[province] : real[6] 'ratio ~ normal(mu = 0.0, sigma = 1.0)
}
forcing {
  beta_t[p in province] : bspline 'per_day {
    window = [0 'days, 168 'days]
    link   = log
    coefs  = mu_b + sd_b * z[p]
  }
}
```

Every prior here is plain, so PGAS accepts the model. The coefficient of
province `p` at position `k` is `mu_b_k + sd_b * z_p_k`, which has the
distribution `normal(mu_b[k], sd_b)` given the hyperparameters, exactly as in
the centred form. What differs is what is estimated: the cells of `z` are
standardised deviations, and the province coefficients are derived quantities.

### 2.6 Family metadata in the IR

The compiler records each family in the IR so that nothing downstream
reconstructs membership by parsing names. A name like `b_2` is ambiguous between
cell 2 of a family `b` and a scalar the author called `b_2`; a resolver that
guesses will eventually be wrong, silently. The IR gains one list:

```text
parameter_families: [
  { name: "b",
    cells: ["b_kinshasa_0", ..., "b_ituri_5"],   // in index order
    axes:  [ Named "province" (levels in declared order), Sized 6 ] },
  ...
]
```

`name` is the source name; `cells` is the ordered list of scalar parameter names
(each of which is an ordinary entry of `parameters`, unchanged), in row-major
index order with the last axis fastest; `axes` is the ordered list of axes, each
either named (dimension and its levels) or sized (length). Scalar parameters are
not families. Binder names are source syntax and are not recorded.

Everything computational is unchanged: the evaluator, the particle filter, PGAS,
and the gradient code read scalar parameters exactly as before. The family list
is metadata for the fit config, the CLI, summaries, traces and the viewer. It
adds an IR field, so `ir/VERSION` bumps (0.42 to 0.43) and every golden and the
Rust deserializer change atomically with it (CLAUDE.md, "Goldens and
`ir/VERSION` are a human-loop change"). The bump re-keys cached runs; pre-1.0
that is the cost of the better representation and is accepted.

### 2.7 Computational identity and provenance identity are different things

Two identities are in play and the proposal treats them separately.

**Computational identity** is the content hash used to key and deduplicate
cached runs (`impl ContentAddressed for Model` in
`rust/crates/runid/src/ir_hash.rs`, around line 1486). It answers "would these
two runs compute the same numbers?". It is defined by the scalar lowering: if a
model written with `b : real[6]` lowers to the same scalar parameters, priors
and forcing coefficients as the hand-written six-parameter model, the two
evaluate identically and must share computational identity. The hash therefore
excludes `parameter_families`; because that impl destructures `Model`
exhaustively, the new field is bound there and skipped with a comment stating
why, rather than silently omitted. The equivalence test in section 6 asserts
equal hashes.

The fit configuration enters computational identity only through its resolved
scalar partition (which cells are estimated, with what bounds, priors and
transforms), which is already part of the resolved problem of parameter-roles
(`ResolvedProblem`, section 6.3) and of the resolved fit identity of
method-scoped-settings. This proposal adds nothing to that identity; it requires
that what is keyed is the _expanded scalar partition_, never the unexpanded
text, so that `b = {}`, six `"b[k]"` entries and six `b_k` entries share one
identity.

**Provenance identity** answers "which model source produced this run, and how
was it written?". It is the model source the fit records, not the hash. Two
spellings with one computational identity have different provenance, as any
textual edit does.

The consequence is a rule for every reader of family structure: viewers,
summaries, trace grouping and the fit-config resolver take the family structure
from the _current compiled model_, never from a cached numerical artifact. A
cached run may have been produced by the other spelling, and its stored metadata
(if any) describes that spelling, not the one the user is looking at. Grouping
the six columns `b_kinshasa_0 ... b_kinshasa_5` under `b[kinshasa]` in a summary
is done by compiling the model in hand and reading its `parameter_families`.

### 2.8 `coefs` takes a 1-D slice

```camdl
forcing {
  beta_t[p in province] : bspline 'per_day {
    window = [0 'days, 168 'days]
    link   = log
    coefs  = b[p]
  }
}
```

`coefs = <slice>` takes the cells of a one-dimensional slice in index order: for
the sized axis the integer order, for a named axis the dimension's declared
level order (section 2.10 restricts that case). A whole one-axis family is such
a slice (`coefs = b` for `b : real[6]`). Any other shape is E420. A list
`coefs = [..]` remains valid and unchanged.

**Element-wise arithmetic inside `coefs`.** The `coefs` value may also be an
arithmetic expression (`+`, `-`, `*`, `/`, unary `-`) whose operands are 1-D
slices, scalar expressions and parenthesised sub-expressions of the same kind,
as in `coefs = mu_b + sd_b * z[p]`. The rules:

- every slice operand must be 1-D (otherwise E420), and all slice operands of
  one expression must have the same length (otherwise E420, naming both
  lengths); slices from different families may be combined;
- a scalar operand (a scalar parameter, a fully indexed cell such as `mu_b[0]`,
  a constant, or arithmetic over these) is broadcast: it combines with every
  element;
- the result is a 1-D list of that length, element `k` being the expression with
  each slice replaced by its cell `k`; an expression with no slice operand is a
  scalar and is not a valid `coefs` value (E420);
- the expansion happens in the compiler: `mu_b + sd_b * z[p]` for province
  `kinshasa` lowers to the same IR as the hand-written list
  `[mu_b_0 + sd_b * z_kinshasa_0, ..., mu_b_5 + sd_b * z_kinshasa_5]`, which
  `coefs` already accepts (language spec section 7, "Coefficients may be
  parameters, or arithmetic over them"), so evaluation and the compiler-emitted
  coefficient gradients are unchanged;
- the length of the result is the slice length for the purpose of `n_basis`
  (section 2.9), and the ordering rule E416 (section 2.10) applies to every
  slice operand.

This arithmetic exists only inside a list-valued context, which in this proposal
is `coefs` alone. A slice is still not a number in a rate, a `let`, a prior
argument or an observation (E420), and there is no list-valued `let`. It is not
a vector type (section 4): nothing list-valued reaches the IR.

Per-stratum forcings (`name[p in dim] : bspline ...`, gh#345) already accept
`bspline` and expand once per stratum. Inside such a forcing the binder indexes
the family: `b[p]` is the slice for the current province. This is the piece the
current surface lacks (section 1, defect 2): today the single `coefs` list is
not indexed by the binder, so every stratum shares it. The proposal adds binder
substitution inside `coefs` slices and inside indexed references in `coefs` list
elements, using the same mechanism that substitutes binders in rates.

One family used by two forcings is allowed, and it means the two forcings share
coefficients (one estimated curve used twice, for example a transmission curve
used for two pathogens). It is stated in the diagnostics catalog as intentional,
not warned.

### 2.9 `n_basis` from the family

| Form                              | Basis count                                 | Cells required |
| --------------------------------- | ------------------------------------------- | -------------- |
| `window` and `coefs = <slice>`    | `n_basis` if written, else the slice length | `n_basis`      |
| `window` and `coefs = smooth(..)` | `n_basis` if written, else `modes size + 2` | `n_basis - 2`  |
| `breakpoints` (either coefs form) | from the knots: `(count - 1) + degree`      | as above       |

`n_basis` may be omitted whenever a slice fixes it. If it is written and
disagrees with the slice, or the slice disagrees with the count the breakpoints
give, the compiler refuses with **E415**: "the family `b` has 5 cells but this
spline has 6 basis functions", with the hint naming both numbers and the
declaration. `n_basis` written beside `breakpoints` keeps its existing refusal
(E408: the knots already fix it). E415 also replaces the `modes` cell-count
message presently reported as E413, so one code says "a family's length
disagrees with the spline" for both uses.

The basis count is part of the model, not a tuning detail. A family prior of
independent cells `b_j ~ normal(0, sigma^2)` induces on the spline
`f(t) = sum_j b_j B_j(t)` the Gaussian-process-like prior with covariance
`Cov(f(s), f(t)) = sigma^2 sum_j B_j(s) B_j(t)`, where the `B_j` are the basis
functions. That covariance depends on the knots and the degree, so changing
`n_basis` changes the prior on the curve and therefore the model, not merely its
resolution. The authored size is thus load-bearing and is written down, not
inferred from data.

### 2.10 Ordering safety for a named axis used as `coefs`

A coefficient list is ordered: cell `j` multiplies basis function `B_j`. A slice
whose free axis is a named dimension is positional only if that dimension's
order is fixed by the model source, not by the data.

Decision: _refuse, with a hint naming the sized form_. A `coefs` slice whose
free axis is a named dimension is **E416** unless the dimension is _explicitly
ordered_, with the message "dimension `knot` has no declared order; coefficients
are positional" and the hint "declare the order (`knot = [k0, k1, ...]` or
`order_by`), or use a sized family: `b : real[6]`".

"Explicitly ordered" is the predicate of the ordered-dimensions proposal
(`2026-07-17-ordered-dimensions.md`, section "Ordered/nominal status"): a
dimension declared by an inline list is ordered; one read from a file is ordered
only if the `read()` carries `order_by` (levels ascending by that numeric
column, ties broken by level name); otherwise it is nominal. The predicate is a
property of the declaration's _form_, decided without reading the file, so E416
never depends on CSV row order: shuffling the rows of a file cannot turn a
refusal into an acceptance or change which coefficient multiplies which basis
function. Until ordered-dimensions lands, `order_by` does not exist, so every
file-derived dimension is refused as `coefs` and only inline lists are accepted.
When it lands, nominal dimensions are sorted canonically by level name, which is
still not a meaningful coefficient order, so the refusal stands.

A warning would be wrong for this: the failure is silent and changes the result,
and the project's rule is that warnings are skimmed (`dsl-surface.md`). E416 is
the spline instance of the ordered-dimensions rule that forbids positional
interpretation of an unordered dimension (there for `consecutive` and inline
tables).

### 2.11 Statistical semantics of a family prior

**No hyperparameters.** `b : real[6] ~ normal(mu = -1.5, sigma = 1.0)` means six
independent, identically distributed cells. There is no pooling and no
correlation between neighbouring cells.

**Scalar hyperparameters pool toward a flat curve.**
`b[province] : real[6] ~ normal(mu = mu_b, sigma = sd_b)` with scalar `mu_b` and
`sd_b` makes the cells independent _given_ `mu_b` and `sd_b`. After integrating
the hyperparameters out, the cells are exchangeable (their joint distribution is
unchanged by permuting them) and dependent. What is pooled is a single level:
every coefficient, at every position and in every province, shrinks toward
`mu_b`. Since the basis sums to one, the shared component of every province's
curve is the constant `mu_b`; the prior carries no shared temporal shape.

**Position-indexed hyperparameters pool toward a shared trajectory.**
`b[p in province] : real[k in 6] ~ normal(mu = mu_b[k], sigma = sd_b)` with
`mu_b : real[6]` writes each province curve as `f_p(t) = g(t) + d_p(t)`, where
`g(t) = sum_k mu_b[k] B_k(t)` is the shared national curve and `d_p(t)` is a
province deviation whose prior covariance, given `sd_b`, is
`sd_b^2 sum_k B_k(s) B_k(t)` (section 2.9). Provinces shrink toward `g`, not
toward a constant. The non-centred form of section 2.5,
`coefs = mu_b + sd_b * z[p]` with `z ~ normal(0, 1)`, gives each province
coefficient the same conditional distribution `normal(mu_b[k], sd_b)`, so
everything in this section holds for it with `b[p, k]` read as
`mu_b[k] + sd_b * z[p, k]`.

**Covariance and dependence, precisely.** A shared _mean_ produces positive
covariance: for two cells whose prior mean is the same hyperparameter `m`,
`Cov(b_i, b_j) = Var(m)`, by the law of total covariance (the conditional
covariance is zero and the conditional means are both `m`). With scalar `mu_b`
this holds for every pair of cells. With position-indexed `mu_b[k]`, it holds
for two provinces at the same position, `Cov(b[p, k], b[q, k]) = Var(mu_b[k])`,
while cells at different positions have covariance zero when the `mu_b[k]` are a
priori independent. A shared _scale_ alone produces dependence without linear
covariance: if `b_i ~ normal(c, s)` with `c` constant and `s` shared, then
`Cov(b_i, b_j) = 0`, yet the cells are not independent, because a large
`|b_i - c|` makes a large `s` more probable and so predicts a large `|b_j - c|`.
Cells at different positions in the position-indexed model are of this kind:
uncorrelated, but dependent through `sd_b`.

**What works today and what is added.** Verified by compiling: a family over two
named dimensions whose prior arguments reference scalar hyperparameters (the
prior lowers to a hierarchical prior on every cell, `pool_over` empty); a
per-stratum `bspline` forcing. Added here: the sized axis (2.1), indexed and
partial access (2.4), declaration binders and per-cell prior resolution (2.5),
binder substitution and element-wise slice arithmetic in `coefs` (2.8). Not
added: PGAS support for hierarchical priors (a separate proposal, section 2.5);
a pooling clause `| axis` over a sized axis, or any prior with correlation
between cells of one family (section 4).

### 2.12 `periodic_spline`

`periodic_spline` also takes `coefs`, and accepts a 1-D slice as sugar for the
list, but its coefficient-count rule is its own. The rule, from language spec
section 7 and `expander.ml` (the `"periodic_spline"` branch, lines 7300-7343):
the knots are uniform over one `period`, and there is no endpoint clamping, so
the number of coefficients equals `n_basis` exactly, with `n_basis` required
(E408 if missing) and greater than `degree` (default 3, at most 5; E408
otherwise). There is no `+ degree` term, unlike the open `bspline` with
`breakpoints`, because the periodic basis wraps around instead of adding
boundary functions. With a family, `n_basis` may be omitted and is then the
slice length; if written, it must equal the slice length (E415, replacing the
present E408 length message for this case). The rule `n_basis > degree` applies
to the inferred count too. The periodic path does not call the `bspline` count
helper of section 2.9, so a change to one cannot move the other.

### 2.13 Smooth modes

`smooth` is not released and stays out of scope except for compatibility. The
modes family keeps its meaning and its refusal:

```camdl
parameters {
  z : real[10] 'ratio ~ normal(mu = 0.0, sigma = 1.0)
}
...
coefs = smooth(level = a_b, slope = b_b, scale = tau_b, modes = z, per = 1 'weeks)
```

`n_basis` is the modes size plus 2 (12 here). The existing E414
(`ocaml/lib/compiler/expander.ml`, in the `smooth` handling: modes must be a
`real`-kind family with no range and the visible prior `normal(0, 1)`) is kept
unchanged and applies to a sized family as to a named one. The prior stays
visible in `parameters`: the curvature law of the estimable-bsplines proposal
depends on it, a prior implied by where a parameter is used would make the same
declared `z` mean different things in different places, and visibility keeps one
reading of "what is estimated and with what prior".

### 2.14 Names in the fit config, scenarios and the CLI

Every place a user names a parameter accepts the indexed form, resolved through
the compiled model's `parameter_families` (section 2.7), and also the lowered
name:

| Where                        | Indexed form                                                    | Lowered form      |
| ---------------------------- | --------------------------------------------------------------- | ----------------- |
| fit config `[estimate]` etc. | `"b[3]" = {...}`, `"b[kinshasa, 2]" = {...}` (quoted TOML keys) | `b_3 = {...}`     |
| scenario `set`               | `b[3] = 0.3`                                                    | `b_3 = 0.3`       |
| `--fixed`, `--param`         | `--fixed 'b[3]=0.3'`                                            | `--fixed b_3=0.3` |

Indices in a name are literal levels and integers (there is no binder outside a
model). A whole family is named by its base name (`b = {}`, section 2.15). A
partial index (a slice such as `"b[kinshasa]"`) is refused in the first version,
for the same reason mixing is (section 2.15): it is a cell-group override. An
indexed name is checked like an expression index (E418, E419 wording, reported
as a fit-config or CLI error). Naming one cell twice, once in each form, is
refused as a duplicate.

Output files (summaries, traces, posterior tables) keep the lowered names as
their canonical column names. This is deliberate. The lowered name is what the
IR contains, so it is the one identifier every consumer already agrees on; it is
a plain column name for R, polars and DuckDB, where `b[kinshasa, 2]` needs
quoting; and it needs no shell quoting (`b[3]` is a glob pattern in zsh, so
`--fixed b[3]=0.3` unquoted fails with "no matches found"). Accepting both forms
on input means a user can write what the model says (`b[3]`) or copy a column
name straight out of an output file (`b_3`), and both reach the same cell.
Making the indexed form canonical in outputs would have every downstream script
quote column names; making only the lowered form valid on input would force the
author to translate between the model's notation and the config's.

### 2.15 Fit configuration

`[estimate] b = {}` expands to every cell of the family with shared settings;
`b = { bounds = [..], start = .. }` likewise. This is D5 of the
estimable-bsplines proposal and belongs to parameter-roles Stage 2 (Rust, the
CLI's roles resolution). The resolver reads membership and order from the IR
family list (2.6), not from names. The partition it produces is scalar, one
entry per cell, named by the lowered scalar names, so roles section 4.5 (scalar
IR names) is unchanged.

Mixing is refused in the first version: if `b = {}` and any per-cell entry
(`b_k = {..}` or `"b[k]" = {..}`) appear, the fit config is refused with "`b` is
configured both as a family and per cell; choose one", with no silent
precedence. A cell-level override of a family-level setting is a later, separate
design.

The compiler half (2.1-2.13, the IR field) and the fit-config and CLI half
(2.14, 2.15) ship in the same release, because the IR field has no reader
without the resolver and the resolver has no source for membership without the
IR field.

## 3. Before and after

Explicit coefficients (`ocaml/golden/bspline_param_coefs.camdl`):

```camdl
# before                                    # after
parameters {                                parameters {
  b1 : real 'ratio in [-6, 1]                 b : real[6] 'ratio in [-6, 1]
  ...six lines...                           }
}                                           forcing {
forcing {                                     beta_t : bspline 'per_day {
  beta_t : bspline 'per_day {                   window = [0 'days, 100 'days]
    window  = [0 'days, 100 'days]              link   = log
    n_basis = 6                                 coefs  = b
    link    = log                             }
    coefs   = [b1, b2, b3, b4, b5, b6]      }
  }
}
```

The scenario lines `set = { b1 = -1.0 ... }` become `b[0] = -1.0 ...` (or,
equivalently, `b_0 = -1.0 ...`).

Ebola, a shared national curve plus province deviations, in the non-centred
form, which PGAS accepts:

```camdl
dimensions { province = [kinshasa, nord_kivu, ituri] }
parameters {
  mu_b : real[6]     'ratio ~ normal(mu = -1.5, sigma = 1.0)  # national curve
  sd_b : positive    'ratio ~ exponential(rate = 2)           # deviation scale
  z[province] : real[6] 'ratio ~ normal(mu = 0.0, sigma = 1.0) # standardised deviations
}
forcing {
  beta_t[p in province] : bspline 'per_day {
    window = [0 'days, 168 'days]
    link   = log
    coefs  = mu_b + sd_b * z[p]
  }
}
```

with the fit config `[estimate] mu_b = {}`, `sd_b = {}`, `z = {}`. The IR holds
25 scalar parameters (`mu_b_0 ... mu_b_5`, `sd_b`, and the 18 cells
`z_kinshasa_0 ... z_ituri_5`), all with plain priors, and `beta_t_nord_kivu` has
the coefficient list
`[mu_b_0 + sd_b * z_nord_kivu_0, ...,
mu_b_5 + sd_b * z_nord_kivu_5]`. The
national curve is `exp(sum_k mu_b[k] B_k(t))`, read from the posterior of `mu_b`
directly.

The centred form of the same model, for PMMH or MH:

```camdl
parameters {
  mu_b : real[6]  'ratio ~ normal(mu = -1.5, sigma = 1.0)
  sd_b : positive 'ratio ~ exponential(rate = 2)
  b[p in province] : real[k in 6] 'ratio ~ normal(mu = mu_b[k], sigma = sd_b)
}
...
    coefs  = b[p]
```

Here the IR holds `b_kinshasa_0 ... b_ituri_5` with, for example, the prior of
`b_nord_kivu_2` equal to `normal(mu = mu_b_2, sigma = sd_b)`. PGAS refuses this
form (section 2.5).

## 4. Alternatives considered

- **Size on the name, `b[6] : real`.** Rejected: one bracket position would mean
  size in a declaration and index in an expression; with 0-based indexing `b[6]`
  in an expression is out of range (section 2.2).
- **Sized binder as a keyword form, `real[6 as k]`, or an implicit positional
  binder.** Rejected in favour of `real[k in 6]` (section 2.5): the keyword form
  adds a keyword used nowhere else; the implicit binder is invisible, can
  collide with a user name, and has no counterpart for named axes.
- **Partial indexing that may fix any subset of axes** (`b[_, a]`). Not
  provided: trailing omission covers `coefs` and keeps the shape of a reference
  readable from its length; a wildcard can be added later without breaking
  anything.
- **Spline coefficients declared inside the forcing**
  (`coefs = estimate(6, ...)`). Rejected: it splits "what is estimated" across
  two blocks, defeats the parameters-block-as-inventory rule, and gives the
  priors no home that sampling, roles and `fit predict` already read.
- **A first-class vector type with arithmetic.** Rejected: the runtime and
  inference are scalar; a vector type would need vector values in the IR,
  autodiff through them and dimension analysis with no consumer beyond `coefs`.
  The element-wise arithmetic of section 2.8 gives the notation the non-centred
  form needs (`coefs = mu_b + sd_b * z[p]`) by expanding to scalar expressions
  in the compiler, so nothing list-valued reaches the IR, and Stan-style vector
  arithmetic remains addable later.
- **Lift PGAS's refusal of hierarchical priors here.** Out of scope: it needs
  the hierarchical-prior gradient for NUTS
  (`2026-06-04-hierarchical-priors-gate3-pgas-nuts.md`, gh#175), a change to a
  high-risk inference file that this proposal otherwise does not touch. The
  non-centred form serves PGAS meanwhile.
- **No IR metadata; recover membership from names.** Rejected: `b_2` is
  ambiguous between a family cell and a user scalar, so the resolver would guess
  (section 2.6).
- **Hash the family list into computational identity.** Rejected: identical
  scalar lowerings must evaluate identically, so splitting their identity would
  defeat deduplication for no numerical reason; the scalar partition is what is
  keyed, and the spelling is provenance (section 2.7).
- **Indexed names canonical in output files, or lowered names only on input.**
  Rejected (section 2.14).
- **Warn when a named dimension without declared order is used as `coefs`.**
  Rejected for E416: the failure silently permutes coefficients.
- **A smoothness (random-walk) or autoregressive prior over adjacent cells of a
  family.** Out of scope: it needs a cell-adjacency notion the family does not
  carry, and the estimable-bsplines proposal covers smoothness through `smooth`.
- **1-based cell names `b_1 ... b_n`.** Rejected: gh#990 sets 0 as the base
  wherever a user sees an index, and the cell suffix is an index.

## 5. What changes where, and how big

| Piece                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Estimate                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Parser (`parser.mly`): `real[N]` and `real[k in N]` kind suffix on the scalar and indexed `param_decl` shapes (factor the bounds/prior tail first); optional `p in dim` binders on declaration axes; full and partial index forms in expressions; E417                                                                                                                                                                                                                  | medium, 1.5 days                                                               |
| AST and expander: sized axis and binders in `PIndexed`/`PScalar`; name producer; per-cell prior resolution with a binder environment; indexed-reference resolution in `resolve_expr`; E230 against lowered names; E236 mapping indexed references to families; binder substitution in `coefs`; element-wise slice arithmetic in `coefs`, expanded to per-element scalar expressions; E415, E416, E418, E419, E420; `n_basis` table; the separate `periodic_spline` path | large, 4 days; the compiler lists every `param_decl` and index match to extend |
| IR: `parameter_families` in `ir.ml` and `serde.ml`; `ir/VERSION` 0.43; Rust `ir` crate; bound and skipped in `ir_hash.rs`; every golden regenerated (human reviewed)                                                                                                                                                                                                                                                                                                    | medium, 1.5 days, plus golden review                                           |
| Tree-sitter: `grammar.js` (`real[N]`, `real[k in N]`, declaration binders, index forms), regenerate the committed `src/parser.c`, `grammar.json`, `node-types.json` with the pinned `npx -y tree-sitter-cli@0.22.6 generate`; corpus tests                                                                                                                                                                                                                              | small, 0.5 day, plus review of a large generated diff                          |
| Language spec: sections 4.3, 7 (bspline, smooth, periodic_spline), the new error codes; edited by hand, never `dprint fmt` on `docs/camdl-language-spec.md`; also the warning catalog, cheatsheet, user-features, and a `language-changes.md` entry for the E413-to-E415 renumbering                                                                                                                                                                                    | small, 1 day                                                                   |
| Rust fit config and CLI: `[estimate]` family expansion through the IR metadata; indexed and lowered names in fit-config keys, scenario `set`, `--fixed`, `--param`; refusal of mixed settings, slice keys and duplicates; scalar partition keyed in the resolved problem (parameter-roles Stage 2); summaries group by the current model's families                                                                                                                     | medium to large, 3 days                                                        |

The compiler half touches none of the high-risk inference files (`pgas.rs`,
`pgas_grad.rs`, `obs_loglik.rs`, `obs_model.rs`, `if2.rs`,
`particle_filter.rs`). The Rust half reads the new field in roles resolution,
name resolution and summaries only.

## 6. Tests

Each test is mutation-checked (revert the source, keep the test, confirm red).

1. **IR equivalence.** `b : real[6]` with `coefs = b` compiles to scalar
   parameters, priors, bounds and forcing coefficients identical to the
   hand-written six-parameter model, and the computational hash of the two IRs
   is equal (the family list is excluded); likewise `z : real[10]` against a
   ten-level named family, and a binder-free family prior against the same
   declaration with unused binders. Both Ebola models of section 3 against their
   hand-written scalar versions (the non-centred one with hand-written
   coefficient lists `[mu_b_0 + sd_b * z_kinshasa_0, ...]`, the centred one with
   18 hand-written per-cell priors): equal IR content and hash.
2. **Density and gradient equivalence.** The same pairs: equal log prior, equal
   log likelihood and equal gradients, deterministically evaluated at a set of
   fixed parameter vectors. Fixed-seed sampler output (posterior draws) is
   compared for equality only for pairs whose execution path is identical (the
   same scalar IR in the same parameter order, test 1); where the order differs,
   only the deterministic evaluations are compared, because a different
   parameter order legitimately changes the random-number stream.
3. **Prior sampling.** Draws from a family prior have the declared marginals;
   cells are independent given fixed hyperparameters; with a shared scalar mean,
   the empirical pairwise covariance matches `Var(mu_b)`; with position-indexed
   means, same-position cells across provinces covary by `Var(mu_b[k])` and
   different-position cells are uncorrelated; with a shared scale only, cells
   are uncorrelated but their absolute deviations are positively correlated.
4. **Per-cell prior resolution and backend coverage.** In the centred Ebola
   model, the prior of each `b_<province>_<k>` references `mu_b_<k>` and `sd_b`;
   a reference `mu_b[k]` with `mu_b : real[4]` and `k in 6` is E418; a binder
   colliding with a parameter name is refused; a family whose prior references
   itself is E236; a hand-written `mu = mu_b_kinshasa` is accepted. The centred
   model runs under PMMH and is refused by PGAS with the existing gh#175
   message; the non-centred model runs under PGAS (a short run, asserting it
   starts and moves, not a convergence claim).
5. **Partial indexing.** For `b[province, age] : real[6]`: `b[p, a, 3]` is a
   scalar, `b[p, a]` is accepted as `coefs`, `b[p]` and `b` as `coefs` are E420
   naming the free axes, `b[p]` as a number is E420, `b[province, a, 3]` (a
   dimension name as index) is refused with the binder hint, four indices is
   E419, a binder over `age` in the `province` position is E419. Element-wise
   arithmetic: `mu_b + sd_b * z[p]` expands per element as stated;
   `mu_b - z[p]`, `-z[p]` and `z[p] / 2` likewise; operands of lengths 6 and 4
   are E420 naming both; a 2-D operand is E420; `mu_b[0] + sd_b` (no slice) as
   `coefs` is E420; `mu_b + 1` used in a rate is E420.
6. **Round-trip names.** Names, order and membership of a family are identical
   through compile, fit config expansion, outputs (summaries, traces) and
   reload, for `b : real[6]` and `b[province] : real[6]`. Summaries group a
   cached run's columns by the families of the model compiled now, asserted by
   reusing a run produced by the hand-written spelling.
7. **Boundaries.** Size 1; sizes 0, negative, fractional and non-literal (E417);
   family-versus-scalar collision (`b : real[6]` beside `b_3`, E278);
   family-versus-family collision (`b_k : real[3]` and `b[g] : real[3]` with a
   level `k`, E278); constant index 6 on a 6-cell family (E418, with "indexed 0
   through 5"); a sized index on a named-only family (E419); one family used as
   `coefs` by two forcings (accepted, shared coefficients, asserted in the IR);
   `n_basis` disagreeing with the slice (E415); `coefs = b[p]` yielding
   different coefficients per province in the expanded forcings.
8. **Ordering.** A file-derived dimension as the free axis of `coefs` is E416
   with the original and with row-shuffled files alike (the decision does not
   read the rows); an inline-list dimension is accepted, in declared order.
9. **periodic_spline.** A slice of length `n_basis` is accepted; `n_basis`
   omitted infers the slice length; a disagreeing `n_basis` is E415; an inferred
   count not greater than `degree` is E408.
10. **Fit config and CLI names.** `b = {}` expands to all cells; `"b[3]"`,
    `"b[kinshasa, 2]"`, `b_3`, scenario `b[3] = 0.3` and `--fixed 'b[3]=0.3'`
    each reach the intended cell; the same cell named in both forms is a
    duplicate error; a slice key is refused; mixed family and per-cell settings
    are refused; `b = {}`, six `"b[k]"` entries and six `b_k` entries have the
    same resolved scalar partition and the same fit identity; output columns use
    lowered names.
11. **Smooth.** `z : real[10]` with a non-standard prior as `modes` is E414;
    with `normal(0, 1)` accepted and `n_basis` 12.
12. **Tree-sitter.** Corpus entries for `real[N]`, `real[k in N]`, declaration
    binders and full and partial indexing; regeneration with the pinned CLI is
    reproducible.
13. **Diagnostics.** Each new code (E415 to E420 and the fit-config refusals)
    has a test asserting the exact message, location and hint.
