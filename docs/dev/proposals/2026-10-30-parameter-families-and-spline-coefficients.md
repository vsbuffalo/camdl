---
date: 2026-10-30
status: proposed; every design question is decided
area: DSL parameters, indexed expressions, bspline coefficients, IR family metadata, fit-config roles
related:
  - 2026-10-27-estimable-bsplines.md (D5, section 6.4 smooth, E413 and E414)
  - 2026-10-26-parameter-roles.md (Stage 2; ResolvedProblem)
  - 2026-10-29-method-scoped-settings.md (resolved fit identity)
  - 2026-07-17-ordered-dimensions.md (proposed, not landed)
---

# Parameter families and spline coefficients

Background the reader is assumed to bring: `docs/camdl-language-spec.md` section
4.3 (indexed parameters), section 7 (forcings: `bspline`, `coefs`, `smooth`),
and the proposals above. A _family_ below means a set of scalar parameters that
share one declaration, one source name and one prior; the cells of the family
are the individual scalars.

## 1. Problem

A cubic B-spline transmission curve with 12 basis functions needs 12 estimated
coefficients. Today the author writes 12 separately named parameters and a
12-element `coefs` list, or declares a family over a named dimension that exists
only to count (`dimensions { mode = [m1, ..., m10] }`). Either way three numbers
must agree: the basis count, the list or dimension length, and the number of
declared parameters. The compiler checks only some of the agreements, and the
reader of the model cannot see from the `parameters` block that twelve cells are
one curve.

The motivating model is the Ebola province analysis: one transmission curve per
province, each with six coefficients, with the coefficients partially pooled
across provinces (each province's coefficient shrinks toward a shared mean, with
the amount of shrinkage itself estimated). Written with today's surface:

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

Two defects, both verified by compiling this shape with the current compiler:
the two-axis family and its hyperparameter-referencing prior compile and are
expanded to `b_kinshasa_k0 ...` with the `normal(mu = mu_b, sigma = sd_b)` prior
on every cell; but the forcing is expanded once per province from one `coefs`
list, so `beta_t_nord_kivu` and `beta_t_ituri` silently receive Kinshasa's
coefficients. There is no way to say "the coefficients of this province". The
after form of section 2 removes both the counting dimension and that defect.

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
`z[mode] : real 'ratio ~ ...`, and the two compose:
`b[province] : real[6]
'ratio ~ ...` declares six cells per province,
`b[province, 0]` through `b[province, 5]`. In a declaration, bracketed names
after the identifier are dimensions the family is indexed by; the bracketed
integer after the type is the number of anonymous cells (the sized axis) and
always comes last in the index order.

A sized axis is not a dimension: it cannot appear in `stratify`, as a `let`
binder or as a table axis. It exists to count and to be indexed.

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

The new surface is the type `real[N]`; the other numeric kinds accept the same
suffix (`positive[N]`, `rate[N]`, and so on) so a size is available to every
kind, not only `real`.

### 2.3 Cell names and the diagnostics around sizes

The compiler lowers a family to scalar cells. A named-dimension family already
does so through one producer, `expand_indexed_decl_names` in
`ocaml/lib/compiler/expander.ml`, giving `<base>_<level>`; the lookup tables,
the name-collision check (E278) and `expand_parameters` all use it. A sized axis
adds index suffixes in the same producer: `b : real[6]` yields `b_0 ... b_5`,
and `b[province] : real[6]` yields `b_kinshasa_0 ... b_ituri_5`. These mangled
names are what the scalar IR, the fit config and the output files contain
(section 2.5 makes that membership explicit rather than parsed).

New diagnostics (all codes verified free: the catalog and `ocaml/lib` stop at
E414):

- **E417**, invalid size. The size must be a positive integer literal.
  `real[0]`, `real[-3]`, `real[2.5]` and a size given by an expression are
  refused with "the size of a parameter family must be a positive integer
  literal". A size of 1 is legal (a family with one cell, indexed `b[0]`); it is
  not collapsed to a scalar, so a model can be written uniformly and later
  resized.
- **E278** (existing), name collision: a family `b : real[6]` beside a hand
  declared scalar `b_3` collides, because both produce the scalar name `b_3`. No
  new check is needed.

### 2.4 Indexed access in expressions

A family is read with brackets: `b[3]`; for two axes `b[province, 3]` with the
named index first, matching the declaration order; and a _slice_ that fixes the
named axes and leaves the sized one free, `b[province]`, meaning the six
coefficients of one province.

- A constant index resolves at compile time to the scalar cell (`b[3]` is the
  parameter `b_3`), so the scalar IR is exactly what hand-written scalar names
  would produce.
- An index that is a stratum binder (`b[province, 3]` inside a transition or
  forcing indexed over `province`) follows the existing rule for named axes: the
  expander substitutes the binder's level.
- A slice is a list-valued expression. It is accepted only where a list of cells
  is expected, which in this proposal means `coefs` (section 2.6). Used as a
  number, it is refused with "`b[province]` is a family slice, not a number".
- **E418**, index out of range. A constant index past the sized axis is refused
  with the range stated: "family b has 6 cells, indexed 0 through 5; `b[6]` is
  out of range". The same code covers a negative constant index, an index into a
  family that has no sized axis, and a sized index on a family with named axes
  only. A named index that is not a level of the dimension keeps its existing
  diagnostic.

The old mangled spelling `b_3` keeps working as a scalar name, because it is
what the IR contains, but the model author is expected to write `b[3]`.

### 2.5 Family metadata in the IR

The compiler records each family in the IR so that nothing downstream
reconstructs membership by parsing names. A name like `b_2` is ambiguous between
cell 2 of a family `b` and a scalar the author called `b_2`; a resolver that
guesses will eventually be wrong, silently. The IR gains one list:

```text
parameter_families: [
  { name: "b",
    cells: ["b_kinshasa_0", ..., "b_ituri_5"],   // in declaration order
    axes:  [ Named "province" (levels in declared order), Sized 6 ] },
  ...
]
```

`name` is the source name; `cells` is the ordered list of scalar parameter names
(each of which is an ordinary entry of `parameters`, unchanged); `axes` is the
ordered list of axes, each either named (dimension and its levels) or sized
(length). Scalar parameters are not families.

Everything computational is unchanged: the evaluator, the particle filter, PGAS,
and the gradient code read scalar parameters exactly as before. The family list
is metadata for the fit config, summaries, traces and the viewer. It adds an IR
field, so `ir/VERSION` bumps (0.42 to 0.43) and every golden and the Rust
deserializer change atomically with it (CLAUDE.md, "Goldens and `ir/VERSION` are
a human-loop change"). The bump re-keys cached runs; pre-1.0 that is the cost of
the better representation and is accepted.

Effect on run identity. Numerical identity is defined by what the likelihood and
prior evaluate, which is the scalar lowering. If a model written with
`b :
real[6]` lowers to the same scalar parameters, priors and forcing
coefficients as the hand-written six-parameter model, the two must evaluate
identically, and the family list must not by itself split the identity of
otherwise identical fits. The hashing of the compiled IR therefore excludes
`parameter_families` (the field is non-semantic for evaluation), and the
equivalence test in section 5 asserts this. What the family structure _does_
influence is the fit configuration: `[estimate] b = {}` expands, through the
family list, into a per-cell partition, and that expansion can differ between
two runs with the same IR. A resolved fit's scalar partition (which cells are
estimated, with what bounds, priors and transforms) is already part of the
resolved problem of parameter-roles (`ResolvedProblem`, section 6.3) and of the
fit identity of method-scoped-settings (the resolved fit identity). This
proposal adds nothing to that identity; it requires that the partition recorded
and keyed is the _expanded scalar partition_, never the unexpanded `b = {}`
text, so that `b = {}` and the same six per-cell entries share one identity.

### 2.6 `coefs` takes a family or a slice

```camdl
forcing {
  beta_t[p in province] : bspline 'per_day {
    window = [0 'days, 168 'days]
    link   = log
    coefs  = b[p]
  }
}
```

`coefs = <family>` and `coefs = <slice>` take the cells in order: for a sized
axis the index order, for a named dimension the dimension's declared level
order. A list `coefs = [..]` remains valid and unchanged.

Per-stratum forcings (`name[p in dim] : bspline ...`, gh#345) already accept
`bspline` and expand once per stratum. Inside such a forcing the binder indexes
the family: `b[p]` is the slice for the current province. This is the piece the
current surface lacks (section 1): today the single `coefs` list is not indexed
by the binder, so every stratum shares it. The proposal adds binder substitution
inside `coefs` slices, using the same mechanism that substitutes binders in
rates. The slice is always taken with a binder or a constant level:
`b[province]` written with the dimension name alone does not say which province,
so inside a forcing it is refused (E418, "name the province, or use the
forcing's binder").

One family used by two forcings is allowed, and it means the two forcings share
coefficients (one estimated curve used twice, for example a transmission curve
used for two pathogens). It is stated in the diagnostics catalog as intentional,
not warned.

### 2.7 `n_basis` from the family

| Form                              | Basis count                                 | Cells required |
| --------------------------------- | ------------------------------------------- | -------------- |
| `window` and `coefs = <family>`   | `n_basis` if written, else the family size  | `n_basis`      |
| `window` and `coefs = smooth(..)` | `n_basis` if written, else `modes size + 2` | `n_basis - 2`  |
| `breakpoints` (either coefs form) | from the knots: `(count - 1) + degree`      | as above       |

`n_basis` may be omitted whenever a family fixes it. If it is written and
disagrees with the family, or the family disagrees with the count the
breakpoints give, the compiler refuses with **E415**: "the family `b` has 5
cells but this spline has 6 basis functions", with the hint naming both numbers
and the declaration. `n_basis` written beside `breakpoints` keeps its existing
refusal (E408: the knots already fix it). E415 also replaces the `modes`
cell-count message presently reported as E413, so one code says "a family's
length disagrees with the spline" for both uses.

The basis count is part of the model, not a tuning detail. A family prior of
independent cells `b_j ~ normal(0, sigma^2)` induces on the spline
`f(t) = sum_j b_j B_j(t)` the Gaussian-process-like prior with covariance
`Cov(f(s), f(t)) = sigma^2 sum_j B_j(s) B_j(t)`, where the `B_j` are the basis
functions. That covariance depends on the knots and the degree, so changing
`n_basis` changes the prior on the curve and therefore the model, not merely its
resolution. The authored size is thus load-bearing and is written down, not
inferred from data.

### 2.8 Ordering safety for a named dimension used as `coefs`

A coefficient list is ordered: cell `j` multiplies basis function `B_j`. A named
dimension has a declared order only when the author wrote it (an inline list or
an `order_by` column). A data-derived dimension without one has no guaranteed
order, and a row-order change in a CSV would silently permute the coefficients,
which is a silent wrong answer.

Decision: _refuse, with a hint naming the sized form_. Using a named-dimension
family (or a slice whose free axis is named) as `coefs` is **E416** unless the
dimension is explicitly ordered, with the message "dimension `knot` has no
declared order; coefficients are positional" and the hint "declare the order
(`knot = [k0, k1, ...]` or `order_by`), or use a sized family: `b : real[6]`". A
warning would be wrong for this: the failure is silent and changes the result,
and the project's rule is that warnings are skimmed (`dsl-surface.md`). The
proposed ordered-dimensions work (`2026-07-17-ordered-dimensions.md`, not yet
landed) separately forbids positional interpretation of an unordered dimension
for `consecutive` and inline tables for the same reason; E416 is the spline
instance of that rule and uses the same "explicitly ordered" predicate when it
lands (until then, an inline dimension list counts as ordered and a data-derived
one as unordered).

### 2.9 Statistical semantics of a family prior

- `b : real[6] ~ normal(mu = m, sigma = s)` means six _independent, identically
  distributed_ cells given `m` and `s`. There is no pooling, and no correlation
  between neighbouring cells, unless hyperparameters appear in the arguments.
- Partial pooling across provinces is written with hyperparameters:
  `b[province] : real[6] ~ normal(mu = mu_b, sigma = sd_b)` with `mu_b` and
  `sd_b` declared (and given priors) as scalars. Given them, all `provinces x 6`
  cells are independent; marginally they are exchangeable and correlated through
  the shared `mu_b` and `sd_b`, which is the standard hierarchical model.
- What works today, verified by compiling: a family over two named dimensions
  with arguments that reference scalar hyperparameters (the prior lowers to a
  hierarchical prior on every cell, `pool_over` empty); a per-stratum `bspline`
  forcing. What this proposal must add: the sized axis in the declaration (2.1),
  indexed access (2.4), binder substitution in `coefs` (2.6), and recognising
  hyperparameter arguments on the sized form. What it does not add:
  hyperparameters that vary by axis (`mu = mu_b[province]`), a pooling clause
  over a sized axis (`| axis`), or any prior with correlation between cells (see
  section 4).

### 2.10 `periodic_spline`

`periodic_spline` also takes `coefs`, but its coefficient count and its endpoint
(wrap-around) condition differ from the open `bspline`, whose count follows from
the knots and degree. Those rules are specified in the `periodic_spline` part of
the language spec and implemented in its own code path; there it accepts
`coefs = <family>` as sugar for the list and checks the family size against its
own count with E415. It does not call the `bspline` count helper of section 2.7,
so a change to one cannot move the other. (The exact periodic count is stated in
that spec section when it is edited; it is deliberately not derived here.)

### 2.11 Smooth modes

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

### 2.12 Fit configuration

`[estimate] b = {}` expands to every cell of the family with shared settings;
`b = { bounds = [..], start = .. }` likewise. This is D5 of the
estimable-bsplines proposal and belongs to parameter-roles Stage 2 (Rust, the
CLI's roles resolution). The resolver reads membership and order from the IR
family list (2.5), not from names. The partition it produces is scalar, one
entry per cell, named by the mangled scalar names, so roles section 4.5 (scalar
IR names) is unchanged.

Mixing is refused in the first version: if `b = {}` and any `b_k = {..}` (or
`b[k]`) appear, or if a scalar named like the family exists beside it, the fit
config is refused with "`b` is configured both as a family and per cell; choose
one", with no silent precedence. A cell-level override of a family-level setting
is a later, separate design.

The compiler half (2.1-2.11, the IR field) and the fit-config half ship in the
same release, because the IR field has no reader without the resolver and the
resolver has no source for membership without the IR field.

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

The scenario lines `set = { b1 = -1.0 ... }` become `b_0 = -1.0 ...` (scenario
`set` addresses scalar cells by their lowered names; section 2.4's `b[0] = ..`
form in `set` is not added, so scenario files, fit configs and outputs all speak
one scalar vocabulary).

Ebola, per-province curves with partial pooling:

```camdl
dimensions { province = [kinshasa, nord_kivu, ituri] }
parameters {
  mu_b : real     'ratio ~ normal(mu = -1.5, sigma = 1.0)
  sd_b : positive 'ratio ~ half_normal(sigma = 1.0)
  b[province] : real[6] 'ratio ~ normal(mu = mu_b, sigma = sd_b)
}
forcing {
  beta_t[p in province] : bspline 'per_day {
    window = [0 'days, 168 'days]
    link   = log
    coefs  = b[p]
  }
}
```

with the fit config `[estimate] mu_b = {}`, `sd_b = {}`, `b = {}`.

## 4. Alternatives considered

- **Size on the name, `b[6] : real`.** Rejected: one bracket position would mean
  size in a declaration and index in an expression; with 0-based indexing `b[6]`
  in an expression is out of range (section 2.2).
- **Spline coefficients declared inside the forcing**
  (`coefs = estimate(6, ...)`). Rejected: it splits "what is estimated" across
  two blocks, defeats the parameters-block-as-inventory rule, and gives the
  priors no home that sampling, roles and `fit predict` already read.
- **A first-class vector type with arithmetic.** Rejected: the runtime and
  inference are scalar; a vector type would need element-wise operations,
  autodiff through them and dimension analysis with no consumer beyond `coefs`.
  Families lowering to scalars give the same notation at a fraction of the cost,
  and Stan-style vector arithmetic remains addable later.
- **No IR metadata; recover membership from names.** Rejected: `b_2` is
  ambiguous between a family cell and a user scalar, so the resolver would guess
  (section 2.5).
- **Hash the family list into identity.** Rejected: identical scalar lowerings
  must evaluate identically, so splitting their identity would defeat
  deduplication for no numerical reason; the scalar partition is what is keyed.
- **Warn when a named dimension without declared order is used as `coefs`.**
  Rejected for E416: the failure silently permutes coefficients.
- **A smoothness (random-walk) prior over adjacent cells of a family.** Out of
  scope: it needs a cell-adjacency notion the family does not carry, and the
  estimable-bsplines proposal covers smoothness through `smooth`.
- **1-based cell names `b_1 ... b_n`.** Rejected: gh#990 sets 0 as the base
  wherever a user sees an index, and the cell suffix is an index.

## 5. What changes where, and how big

| Piece                                                                                                                                                                                                                                                                                | Estimate                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Parser (`parser.mly`): `real[N]` kind suffix on all four indexed and scalar `param_decl` shapes (factor the bounds/prior tail first); `b[i, j]` and slice forms in expressions; E417                                                                                                 | medium, 1 day                                                                            |
| AST and expander: sized axis in `PIndexed`/`PScalar`, name producer, shared bounds/prior helper, constant-index resolution, binder substitution in `coefs`, E415, E416, E418; `n_basis` table; a separate `periodic_spline` path                                                     | medium to large, 3 days; the compiler lists every `param_decl` and index match to extend |
| IR: `parameter_families` in `ir.ml` and `serde.ml`, excluded from the content hash; `ir/VERSION` 0.43; Rust `ir` crate; every golden regenerated (human reviewed)                                                                                                                    | medium, 1.5 days, plus golden review                                                     |
| Tree-sitter: `grammar.js` (`real[N]`, sized index forms), regenerate the committed `src/parser.c`, `grammar.json`, `node-types.json` with the pinned `npx -y tree-sitter-cli@0.22.6 generate`; corpus tests                                                                          | small, 0.5 day, plus review of a large generated diff                                    |
| Language spec: sections 4.3, 7 (bspline, smooth, periodic_spline), the new error codes; edited by hand, never `dprint fmt` on `docs/camdl-language-spec.md`; also the warning catalog, cheatsheet, user-features, and a `language-changes.md` entry for the E413-to-E415 renumbering | small, 1 day                                                                             |
| Rust fit config: `[estimate]` family expansion through the IR metadata, refusal of mixed settings, scalar partition keyed in the resolved problem (parameter-roles Stage 2)                                                                                                          | medium to large, 2-3 days                                                                |

The compiler half touches none of the high-risk inference files (`pgas.rs`,
`pgas_grad.rs`, `obs_loglik.rs`, `obs_model.rs`, `if2.rs`,
`particle_filter.rs`). The Rust half reads the new field in roles resolution
only.

## 6. Tests

Each test is mutation-checked (revert the source, keep the test, confirm red).

1. **IR equivalence.** `b : real[6]` with `coefs = b` compiles to scalar
   parameters, priors, bounds and forcing coefficients identical to the
   hand-written six-parameter model; the content hash of the two IRs is equal
   (the family list is excluded); likewise `z : real[10]` against a ten-level
   named family.
2. **Statistical equivalence, end to end.** The same two models, same seeds:
   equal log prior, equal log likelihood and equal gradients at fixed parameter
   vectors, and equal fitted posteriors in a short fixed-seed run.
3. **Prior sampling.** Draws from a family prior have the declared marginals,
   independence across cells given fixed hyperparameters, and, with
   hyperparameters, the exchangeable correlation they imply.
4. **Round-trip identity.** Names, order and membership of a family are
   identical through compile, fit config expansion, outputs (summaries, traces)
   and reload; the same for a mixed family `b[province] : real[6]`.
5. **Boundaries.** Size 1; sizes 0, negative, fractional and non-literal (E417);
   collision with a hand-declared `b_3` (E278); constant index 6 on a 6-cell
   family (E418, with "indexed 0 through 5"); one family used as `coefs` by two
   forcings (accepted, shared coefficients, asserted in the IR); `n_basis`
   disagreeing with the family (E415); a named dimension without declared order
   as `coefs` (E416); `coefs = b[p]` yielding different coefficients per
   province in the expanded forcings.
6. **Fit config.** `b = {}` expands to all cells; mixed family and per-cell
   settings are refused; `b = {}` and six per-cell entries have the same
   resolved scalar partition and the same fit identity.
7. **Smooth.** `z : real[10]` with a non-standard prior as `modes` is E414; with
   `normal(0, 1)` accepted and `n_basis` 12.
8. **Tree-sitter.** Corpus entries for `real[N]` and indexed access;
   regeneration with the pinned CLI is reproducible.
9. **Diagnostics.** Each new code (E415 to E418 and the fit-config refusal) has
   a test asserting the exact message, location and hint.
