---
date: 2026-10-30
status: proposed — one item pending the maintainer (§8); everything else decided
area: DSL parameters / forcings (bspline) / fit-config roles
related:
  - 2026-10-27-estimable-bsplines.md (D5, §6.4 smooth, E413/E414)
  - 2026-10-26-parameter-roles.md (§4.5 scalar IR names; Stage 2)
---

# Sized parameter families, for B-spline coefficients and modes

Background assumed: `docs/camdl-language-spec.md` §4.3 (indexed parameters) and
§7 (`bspline`, `coefs`, `smooth`), and the two proposals above.

## 1. Problem

A B-spline with `n_basis = 12` needs twelve coefficients. Today they are either
twelve hand-named parameters and a twelve-element list, or (for `smooth`) a
family indexed by a named dimension that exists only to count:
`dimensions { mode = [m1, …, m10] }`. Both make the user keep three numbers in
agreement (`n_basis`, the list or dimension length, the parameter count) and the
compiler checks only some of them. The Ebola models will have several splines;
the bookkeeping should be one number, written once.

## 2. Design

### 2.1 A sized family

```camdl
parameters {
  b[6] : real 'ratio ~ normal(mu = -1.5, sigma = 1.0)
}
```

declares six scalar parameters, each `real 'ratio` with that prior (and bounds,
if `in [lo, hi]` is written). It is the single-axis case of §4.3 with an
anonymous axis of length 6; it is not a dimension (it cannot be used in
`stratify`, a `let` binder or a table axis).

**Cell names.** A named-dimension family `z[mode]` expands through one producer,
`expand_indexed_decl_names` (`expander.ml:2218`), to `<base>_<level>` over the
dimension's declared levels in order (`z_m1 …`); the same producer feeds the
lookup tables, the E278 collision check and `expand_parameters`. A sized family
adds a second producer beside it with the same shape: `b[6]` becomes
`b_0, b_1, …, b_5`.

The base is 0, not 1. gh#990 (closed; the maintainer's 2026-10-24 decision in
`2026-10-24-batch-to-simulate.md` §3, §12) standardises on one index base, 0,
wherever a user sees an index, and the language spec states no separate base for
the DSL (verified: it has no "1-based" or "0-based" statement). A cell suffix is
an index the user sees, so it counts from 0. This differs from the request's
`b_1 … b_n`; the spelling is one line in the new producer, so reversing it is
cheap if the maintainer prefers 1.

`b[0]` and `b[-1]` are refused (E331's neighbour; reuse E330 with a size
message). `b[6, region]` (a sized axis mixed with a named one) is not accepted:
one anonymous axis, alone.

Cells are referred to by their mangled names (`b_3`). The indexed expression
form `b[3]` is not added by this change; today's unknown-name diagnostic must
carry a hint naming `b_3` (checked at implementation).

### 2.2 `coefs = <family>`

`coefs = b` takes the family's cells in order: for a sized family `b_0 … b_5`,
for a named-dimension family the dimension's declared level order (the order
`expand_indexed_decl_names` returns). It is exactly sugar for the list
`[b_0, …, b_5]` and lowers to the same `Coefs_explicit` list. A bare name that
is not a family, or a scalar, is refused as it is today (`get_kw_list` shape
error); a list stays accepted.

### 2.3 `n_basis` from the family

| Form                                   | Basis count                                | Family length required                |
| -------------------------------------- | ------------------------------------------ | ------------------------------------- |
| `window` + explicit `coefs = b`        | `n_basis` if written, else `length(b)`     | `n_basis`                             |
| `window` + `coefs = smooth(modes = z)` | `n_basis` if written, else `length(z) + 2` | `n_basis − 2`                         |
| `breakpoints` (either coefs form)      | from the knots: `(count − 1) + degree`     | `n_basis`, or `n_basis − 2` for modes |

`n_basis` may now be omitted whenever a family fixes it. If it is written and
disagrees with the family, or the family disagrees with the count the
breakpoints give, the compiler refuses with a new error, **E415**: "the family
`b` has 5 cells but this spline has 6 basis functions", with the hint naming
both and the declaration. It never overrides silently. `n_basis` written with
`breakpoints` stays refused as today (E408; the knots already fix it). E415 also
replaces the `modes` cell-count refusal currently reported as E413
(`expander.ml:6776`), so one code says "a family's length disagrees with the
spline" for both uses; and replaces the E408 "coefs has length …" message when
`coefs` is a bare family. A list of the wrong length remains E408. (E415 is the
next free code: the catalog and `ocaml/lib` stop at E414, verified by grep.)

### 2.4 The smooth modes prior

The modes must be standard normal; the curvature law of the estimable-bsplines
proposal §6.3 depends on it. The compiler already refuses any other prior with
E414 when the family is used as `modes` (`expander.ml:6785-6798`), and requires
`real` kind with no range. This change keeps E414 unchanged and applies it to
sized families: `z[10] : real 'ratio ~ normal(mu = 0.0, sigma = 1.0)` is written
by the user in full. See §8 for the one open choice.

### 2.5 Fit-config side: bare family names in `[estimate]`

`b = {}` (and `z = { bounds = …, start = … }`) expands to every cell with shared
settings; the partition records each cell separately because roles §4.5 keeps
scalar IR names. This is D5 of the estimable-bsplines proposal, planned in
parameter-roles Stage 2, and it is Rust (the CLI's roles resolution), not the
compiler. It pairs with this feature as follows: the compiler change makes `b` a
_family name_ that exists in the IR only as `b_0 … b_5`, so the roles resolver
needs to know which scalar names share a base. The IR carries no family metadata
(verified: `ir.ml` has none), and §3 below keeps it that way, so the resolver
recovers membership from the names. Because a name pattern `b_<digits>` is
ambiguous with a user-written scalar `b_2`, the resolver must not guess: the
fit-config expansion therefore lands with a way to ask the compiler for family
membership, either a field in the compiled model's non-hashed metadata or a
`camdlc --families` listing. Choosing it is part of the roles Stage 2 change,
and is the reason this proposal does not land the fit-config half.

Landing order: the compiler half (§2.1–2.4) first and alone, since explicit
`[estimate] b_0 = {…}` per cell works for any family today; the bare-name
expansion with roles Stage 2, which also owns D5's two refusals (a scalar of the
same name exists; per-cell entries for the same family also appear).

## 3. What changes, from the code

**IR: no change.** Cells are ordinary `Ir.parameter` records (`name`, `value`,
`param_kind`, `param_dim`), built in `expand_parameters`; the existing named
form already emits no family record. A sized `b[6]` lowers to six records
byte-identical to six hand-declared ones (§6, test 1), so there is no
`ir/VERSION` bump and no golden churn beyond the new fixtures. Anything that
would need a family record in the IR is deliberately not in this change.

**OCaml parser (menhir, `ocaml/lib/compiler/parser.mly`).** `param_decl` has
four indexed productions (lines 530-542: bounds x prior), each
`IDENT LBRACKET
separated_nonempty_list(COMMA, IDENT) RBRACKET`. Add four sized
productions with `LBRACKET n = INT RBRACKET` (the `INT` token exists, line 314);
no new lexer token. Because the existing productions are in a 2x2 matrix, factor
the bounds/prior tail into a shared nonterminal in the same edit rather than
adding four more copies.

**AST and expander.** `Ast.param_decl` gains
`PSized of { pname; psize : int;
pkind; pdim; punit; pbounds; pprior; pdoc; ploc }`
beside `PIndexed` (`ast.ml:149`); the compiler's exhaustiveness checking finds
every consumer. `expand_parameters` (`expander.ml:5801`) gets a `PSized` case
that shares the bounds, kind and prior resolution with the `PIndexed` case
(extract it to one helper: the bug-prone part is the shared part) and a size
guard. The name producer, the lookup tables and `check_declaration_names` (E278)
take the new names from the one new function, so a sized family colliding with a
scalar (`b_3` declared by hand) is E278 with no new check. The bspline forcing
code (`expander.ml:7300-7440`) gains the family reader for `coefs` and the §2.3
table; `spline_smooth` (`:6745`) takes its cell list from the same reader.

**Tree-sitter.** `tree-sitter/grammar.js:136`: `param_index` is
`seq("[", field("dim", $.identifier), "]")`; add
`choice($.identifier,
$.number)`. Regenerate the committed `src/parser.c`,
`grammar.json`, `node-types.json` with the pinned CLI,
`npx -y tree-sitter-cli@0.22.6 generate` (0.25.x produces different output); add
a highlight-query case if the queries colour `dim`.

**Language spec (edit by hand; never `dprint fmt` this file, it carries doctest
preambles).** §4.3 (sized form and its names, the `b[0]` refusal), §7's
`bspline` section (`coefs = <family>`, the table of §2.3, E415) and the `smooth`
paragraph (modes may be a sized family; the preamble example's
`dimensions {
mode = … }` line goes). Also `docs/dev/warning-catalog.md` (E415,
and the changed site of the cell-count message), `docs/dsl-cheatsheet.md` and
`docs/user-features.md`. No `docs/language-changes.md` entry: nothing that
compiled before is refused or changes meaning (the E413 cell-count message
changes code to E415; mention it there as a diagnostic renumbering).

**Rust.** None for §2.1-2.4: the runtime reads the IR. Only §2.5 is Rust.

## 4. Before and after

Explicit coefficients, `ocaml/golden/bspline_param_coefs.camdl`:

```camdl
# before
parameters {
  b1 : real 'ratio in [-6, 1]
  b2 : real 'ratio in [-6, 1]
  b3 : real 'ratio in [-6, 1]
  b4 : real 'ratio in [-6, 1]
  b5 : real 'ratio in [-6, 1]
  b6 : real 'ratio in [-6, 1]
}
forcing {
  beta_t : bspline 'per_day {
    window  = [0 'days, 100 'days]
    n_basis = 6
    link    = log
    coefs   = [b1, b2, b3, b4, b5, b6]
  }
}

# after: six cells b_0 … b_5; n_basis is the family length
parameters {
  b[6] : real 'ratio in [-6, 1]
}
forcing {
  beta_t : bspline 'per_day {
    window = [0 'days, 100 'days]
    link   = log
    coefs  = b
  }
}
```

The `set = { b1 = -1.0 … }` scenario lines in that fixture become
`b_0 = -1.0 …`.

Smooth form, spec §7 (`coefs = smooth`):

```camdl
# before: a dimension that exists only to count the modes
dimensions { mode = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] }
parameters {
  z[mode] : real 'ratio ~ normal(mu = 0.0, sigma = 1.0)
}
forcing {
  beta_t : bspline 'per_day {
    window  = [0 'days, 168 'days]
    n_basis = 12
    link    = log
    coefs   = smooth(level = a_b, slope = b_b, scale = tau_b, modes = z, per = 1 'weeks)
  }
}

# after: n_basis = 10 + 2 = 12, from the family
parameters {
  z[10] : real 'ratio ~ normal(mu = 0.0, sigma = 1.0)
}
forcing {
  beta_t : bspline 'per_day {
    window = [0 'days, 168 'days]
    link   = log
    coefs  = smooth(level = a_b, slope = b_b, scale = tau_b, modes = z, per = 1 'weeks)
  }
}
```

## 5. Size, honestly

| Piece                                                                       | Estimate                                                                                              |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Parser: sized productions, shared tail factored                             | small, half a day                                                                                     |
| AST + `expand_parameters` + name producer + shared resolution helper        | small to medium, half a day to a day; the compiler lists every `param_decl` match to extend           |
| `coefs = <family>`, `n_basis` derivation, E415 incl. breakpoints and smooth | medium, a day: the §2.3 table has six cells and each needs a test                                     |
| Tree-sitter grammar and regeneration                                        | small, an hour, plus review of a large generated `parser.c` diff                                      |
| Spec, catalog, cheatsheet, user-features                                    | small, half a day, by hand                                                                            |
| Fixtures, goldens, tests (§6)                                               | medium, a day; folded into the pieces above                                                           |
| Fit-config bare names (§2.5), Rust                                          | medium to large, 1-2 days including the family-membership channel; lands with roles Stage 2, not here |

The compiler half is about three days in one PR. It touches none of the
high-risk inference files.

## 6. Test plan

Each test is mutation-checked (revert the source, keep the test, confirm red).

1. **Expansion equivalence.** `bspline_param_coefs` rewritten with `b[6]` and
   `b_0 … b_5` scenario names compiles to IR equal to the hand-written model
   with the same names (parameter order, bounds, priors, coefficient list).
   Likewise `z[10]` against `z[mode]` with ten levels named `0 … 9`, so the cell
   names coincide. Goldens: new `ocaml/golden/bspline_sized_coefs.camdl` and
   `bspline_sized_smooth.camdl`, picked up by the `OCAML_GOLDENS` wildcard and
   generated by `make update-ocaml-golden` (reviewed by hand as new files; no
   existing golden moves).
2. **Cell names and collisions.** `b[3]` yields `b_0, b_1, b_2`; a hand-declared
   `b_1` beside it is E278; `b[0]` is refused; `b[2, region]` is refused; a
   named family `z[mode]` as `coefs` takes the declared level order, not
   alphabetical.
3. **E415, n_basis mismatch.** `n_basis = 7` with `b[6]`: E415, naming 6 and 7;
   `n_basis` omitted: accepted and equal to 6; smooth with `z[9]` and
   `n_basis = 12`: E415 (needs 10).
4. **E415, breakpoints.** Explicit breakpoints giving 8 basis functions with
   `b[6]`: E415; with `b[8]`: accepted; `n_basis` written beside `breakpoints`:
   still E408. Smooth with breakpoints giving 8 and `z[5]`: E415; `z[6]`:
   accepted.
5. **Modes prior.** `z[10] : real 'ratio ~ normal(mu = 0.0, sigma = 2.0)` used
   as `modes`: E414; `z[10] : positive …`: E414; a range on `z`: E414; the same
   non-standard family used only as `coefs`: accepted.
6. **Shape.** `coefs = gamma` (a scalar), `coefs = nosuchname`, a list of the
   wrong length (still E408).
7. **Tree-sitter.** Highlighting corpus entry for `b[6] : real …`; regenerate
   and check `parser.c` is reproducible with the pinned CLI.
8. **Rust (with roles Stage 2).** `b = {}` expands to all cells with shared
   settings; refused beside a scalar `b` and beside `b_2 = …`.

## 7. What it does not do

It adds no indexed-expression syntax `b[3]`, no multi-axis sized families, no
family record in the IR, and no `periodic_spline` smoothness prior.
`coefs =
<family>` applies to `periodic_spline` as well as `bspline`, since both
read `get_kw_list "coefs"`; the length rule there is the `n_basis` it already
requires.

## 8. Pending the maintainer

**Where the standard-normal modes prior is written.** The question: for
`coefs = smooth(…, modes = z)`, does the user spell the prior of `z`, or does
`smooth` imply it? Under the visible form,
`z[10] : real 'ratio ~ normal(mu =
0.0, sigma = 1.0)` appears in `parameters`,
and E414 refuses any other prior on a family used as `modes`, as it does today.
Under the implied form, `z[10] : real
'ratio` carries no prior and `smooth`
attaches `normal(0, 1)`. What it changes: the visible form keeps every prior in
one place, readable by a scientist who never opens the forcing block and by
tools (the roles resolver and `prior` sampling read declared priors); it costs
one line the user could forget, which E414 turns into a clear refusal. The
implied form is shorter but makes a parameter's prior depend on where it is
used, so the same `z` declared elsewhere would silently differ, and the prior
reaching the IR would not match any declaration. Recommendation: the visible
form, which is the coordinator's decision and what is already implemented.
Confidence: solid. Needed from the maintainer only to confirm it.
