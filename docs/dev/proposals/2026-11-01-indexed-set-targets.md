---
date: 2026-11-01
status: accepted 2026-11-01
area: DSL effects (`set`, `add`), effect families, expander
related:
  - gh#459 (named indices on action targets; the expander comment anticipating this change)
  - gh#461 (E265: a `set`/`add` target must be one expanded cell)
  - gh#1003 (found while porting the same model)
---

# Indexed `set` and `add` targets in effect families

Background the reader is assumed to bring: `docs/camdl-language-spec.md` §5
(stratification and expanded cell names such as `I_a1`), §9.5 (the
`c in compartments` binder), §13.1 (the three actions), §13.3 (indexed
interventions: families with `[a in age]` and `where`), §13.5 (events) and
§13.10 (within-instant effect ordering).

## 1. Problem

An effect family can index everything except the left-hand side of a `set` and
the target of an `add`. This works today:

```camdl
events {
  shift[c in compartments, (a, a_next) in consecutive(age)] :
    transfer(fraction = age_out[a], from = c[a], to = c[a_next]) at [1]
}
```

This does not parse (E001):

```camdl
events {
  rescale[c in compartments, a in age] : { c[a] = c[a] * pop[a] / N[a]  at = [1] }
}
```

A `set` names one expanded cell by its mangled name (`I_a1 = 0`); `add` likewise
(`add(I_a1, 5)`). A model that needs a per-stratum discrete update must
therefore write one line per cell. The motivating model, tbmod's annual
demographic map (Tanvir et al. 2026, a TB vaccine model being replicated in
camdl), multiplies each of 738 cells (9 TB states × 82 ages) once a year by an
age-specific census factor. Its camdl port was a Python generator emitting
38,000 lines, about 2,200 of them per-cell `set`s that cannot be avoided today.
The same pattern arises whenever a discrete-time update acts on a whole
stratified slice:

- **Rescaling to a census:** each age's total set to a data value, in proportion
  across states.
- **Discrete cohort maps:** annual ageing in school-cohort or Leslie-style
  models.
- **Resetting a stratum:** zeroing or reseeding every cell of one patch or age.

`transfer` cannot express these, because it only drains (a factor below 1, into
some destination); `add` cannot, because it takes one cell and refuses negative
values. `set` writes any value, which is what these updates need. Only its
target lacks the family form that transitions and `transfer` already have.

## 2. Existing surface this builds on

Most of the machinery exists; the gap is the parser and target resolution.

| Piece                                                      | Where                                                                                                                            | State                                                                                            |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Effect families with binders and `where` on the block form | `parser.mly` `intervention_decl` (`name = IDENT ibs = index_bindings_opt COLON LBRACE iv_kvs … RBRACE guard = where_clause_opt`) | exists                                                                                           |
| `set` entry inside the block                               | `parser.mly` `iv_kv`: `IDENT EQ e` → `ASet ($1, [], e)`                                                                          | bare name only                                                                                   |
| `ASet` / `AAdd` carry an index list                        | `ast.ml` `action_decl`: `ASet of string * index_item list * expr`                                                                | exists, always `[]` from source                                                                  |
| Family expansion and guard filtering                       | `expander.ml` `expand_scheduled_actions` (`cartesian_product` over `ivindices`, `ivguard`)                                       | exists                                                                                           |
| `c in compartments` binder                                 | `expander.ml` `IComp v` → one combo per integer compartment                                                                      | exists                                                                                           |
| Target name built from indices                             | `expander.ml` `concrete_action_target`                                                                                           | exists; its gh#459 comment says an indexed target "needs `resolve_index_order` with `comp_dims`" |
| One-cell check                                             | `expander.ml` `check_action_target` (E265)                                                                                       | exists                                                                                           |
| Simultaneous application                                   | runtime, §13.10: every `set` and draining `transfer` due at one instant resolves against the same post-inflow state              | exists, all backends                                                                             |
| Recurring `set` schedule                                   | `iv_kv`: `every = … from = … to = …` (all three required)                                                                        | exists                                                                                           |

No IR type, runtime path or golden changes: an indexed family expands to the
same per-cell `Ir.Set` and `Ir.AddAction` values the runtime already executes.

## 3. Design

### 3.1 Grammar

A `set` entry accepts an indexed target:

```
iv_kv  ::= … | IDENT EQ expr                     (* today *)
             | IDENT index_items_opt EQ expr     (* new: replaces the line above *)
```

`add` accepts the same indexed target in its first argument, in each of its four
productions in `parser.mly` (`at [...]`, recurring block, `every … at_day`, and
the reactive `when` action):

```
ADD LPAREN comp = IDENT idxs = index_items_opt COMMA count = expr RPAREN …
```

`index_items_opt` is the existing rule in `parser.mly` for the bracketed index
list of a stoichiometric reference (empty, or a bracketed comma-separated list
of `index_item`); an `index_item` is a positional expression or a named
`dim = expr`. So positional (`c[a, v]`) and named (`c[age = a, vx = v]`) forms
come for free, with E326 refusing a mix.

### 3.2 Target resolution

For each family instance (one binder environment that passes the guard):

1. **Base compartment.** If the target identifier is a compartment binder bound
   in this environment (`c` in `[c in compartments, …]`), the base is the bound
   compartment name. Otherwise the identifier itself is the base (`S[a]`). When
   a compartment binder and a compartment share a name, the binder wins.
2. **Indices.** Each index item resolves through the environment (a binder gives
   its level; a literal level stands for itself). Named items are placed with
   `resolve_index_order ~dims:(comp_dims base)`.
3. **Full indexing is required.** The number of resolved indices must equal the
   number of the base compartment's declared dimensions. A target with a
   dimension left unindexed is E265, with a hint naming the missing dimensions.
   Unlike a transition (§9.5), a `set` does not fan out over omitted dimensions:
   the right-hand side would be ambiguous, since `c[a]` on it could mean the
   cell being set or the stratum total. Write the dimension's binder explicitly.
4. **Concrete name and check.** The concrete name is built as
   `expand_compartment_name` builds cell names (declared dimension order), then
   passed to the existing `check_action_target` (E265 for a non-cell or unknown
   name).

`concrete_action_target` and its gh#459 comment are replaced by this resolution,
and the comment goes with them.

### 3.3 Semantics

Unchanged and inherited.

- **Each family instance is one intervention or event** with `base_name` set to
  the family name, as for `transfer` families (§13.3). Scenario
  `enable`/`disable` by family name works as it does today.
- **All instances due at one instant read one snapshot (§13.10),** so a family
  whose right-hand side reads other cells (an age shift, a rescale reading
  stratum totals) is a simultaneous map, with no dependence on declaration or
  expansion order.
- **Existing same-instant conflicts still apply.** Two `set`s on one cell, or a
  `set` and a `transfer` touching the same cell at one instant, are errors on
  every backend. A family that overlaps another effect at one instant is refused
  exactly as two hand-written effects would be.
- **A negative `add` remains an error.** Indexed `add` changes only how the
  target is named.

### 3.4 Diagnostics

- **E265** gains the under-indexed case. The message names the target as written
  and the dimensions left unindexed (for `S[a]` with `S` stratified by
  `[age, vx]`: "sets 'S[a]', which leaves dimension 'vx' unindexed; a set names
  one cell"), and the hint gives the full form (`S[a, v]`, with `v` bound in the
  family). E265 has no entry in `docs/dev/warning-catalog.md` today, although it
  already has two emit sites (gh#461); add one entry covering all three
  messages.
- Every other failure reuses an existing code: E326 (mixed named and
  positional), E332/E333 (bad index label), E265 (non-cell), E296 (empty action
  list).

This change is additive: every model that parses today parses to the same IR. No
`docs/language-changes.md` entry is needed. Spec §13.1 loses the sentence
"(Index-binder forms like `I[child, p1]` on the left of a `set` are not part of
the grammar.)" and gains the indexed form, with the example in §4.

### 3.5 Decided out of scope

- **Compartment names in `where` guards (E217).** `where c != S` is refused
  today because a guard may not mention a compartment. Allowing compile-time
  name comparisons with a compartment binder is a change to the guard rules for
  every family kind, not just effects. Where a family must skip one compartment,
  write that compartment's cells in a separate single block (the newborn block
  in §4).
- **Demography sugar** (declarative ageing, census conformance). It belongs in
  its own proposal and can lower to the families defined here.
- **Ordered phases within one instant** ("transfer, then rescale the result").
  Not needed: a map that composes steps is written as one expression per cell
  (§4). A future need would be a runtime and IR change with its own proposal.

## 4. Worked example: tbmod's annual demographic map

Within each year the model runs TB transitions only. At each year boundary tbmod
shifts every age group up, puts births into age 0, and multiplies every state in
age `a` by `pop[a] / N_shifted[a]`, the UN population divided by the post-ageing
model population.

```camdl
tables {
  age_out, adult : age = read("data/ages.tsv")       # share moving up per year: 1, 0.1 at 80–89, 0 at 90+
  younger : age × age = read("data/younger.tsv", default = 0.0)  # 1 where band b ages into band a
}
forcing {
  pop[a in age] : interpolated 'count { data = "data/pop.tsv"  key_col = "age"
                                        time_col = "year"  value_col = "pop"  method = constant }
}

# each age's population after the shift, from the pre-event state
let N_shift[a in age] = (1 - age_out[a]) * N[a] + sum(b in age, younger[a, b] * age_out[b] * N[b])

events {
  annual[c in compartments, a in age] : {
    c[a] = ((1 - age_out[a]) * c[a] + sum(b in age, younger[a, b] * age_out[b] * c[b]))
           * pop[a] / N_shift[a]
    every = 1 'years  from = 1901 'years  to = 2052 'years
  } where a != a0

  # age 0 after the shift holds only births; tbmod rescales it to the census, all susceptible
  newborns : {
    S_a0 = pop[a0]   C_a0 = 0   I_a0 = 0   Dn_a0 = 0   Ds_a0 = 0
    Dc_a0 = 0        T_a0 = 0   Rt_a0 = 0  R_a0 = 0
    every = 1 'years  from = 1901 'years  to = 2052 'years
  }
}
```

`annual` expands to 9 × 81 `set` actions. For `Ds` at age 30 the right-hand side
folds to `Ds_a29 * pop_a30 / N_shift_a30`: the same expression the generator
wrote by hand, with the denominator now shared through a `let`. Vaccination
stays an intervention, which runs after events at the same instant (§13.10),
matching tbmod's order (rescale, then vaccinate).

## 5. Tests

Red first; each must fail before the change.

1. **Expansion (OCaml).** The family `r[c in compartments, a in age]` with the
   entry `c[a] = 2 * c[a]`, over compartments `{S, I}` and ages `[a0, a1, a2]`,
   yields six events whose `Set` targets are `S_a0` to `I_a2` and whose values
   read the same cell.
2. **Simultaneity on every backend (Rust, cli).** A `set` family implementing an
   age shift (`age_out = [0.9, 0.5, 0]`, `S = [100, 20, 3]`) gives
   `[10, 100, 13]` on ODE, chain-binomial and Gillespie with transitions off.
   That is the same result as the equivalent `transfer` family, which today
   gives `[10, 100, 13]` on all three; a cascading map would give a different
   answer at `a1`.
3. **Under-indexing is E265** with the missing dimension named (a compartment
   stratified by `[age, vx]`, set as `S[a]`).
4. **Named indices place correctly:** `c[vx = v, age = a]` sets the cell
   `c_<a>_<v>`. A positional/named mix is E326.
5. **The guard filters instances:** `where a != a0` emits no `*_a0` targets.
6. **A literal compartment in a family:** `S[a]` in `[a in age]` sets `S_a*`
   only.
7. **Indexed `add`:** `seed[a in age] : add(I[a], 5) at [1]` adds to each
   `I_a*`; a negative count is still refused.
8. **Same-instant conflict:** a `set` event family and a `transfer` event family
   touching `S_a1` at one instant raise the existing runtime conflict error
   (`effects.rs`, `SimError::Validation`) when simulated. Interventions apply
   one after another and cannot conflict, so this test uses `events`.
9. **Goldens:** `make update-golden` produces no diff. The IR of every existing
   fixture is unchanged.

## 6. Acceptance beyond the gate

The camdl port of the tbmod model rewritten with §4 matches tbmod's state dump
to the same precision as the generated port (four significant figures in every
disease state at mid-year times, parameter set 1). That measurement belongs in
the replication's notes, not in this repository's gate.
