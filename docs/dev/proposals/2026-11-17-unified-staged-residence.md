# Proposal: a unified model for staged residences

**Status:** Implemented **Date:** 2026-11-17 **Motivation:** A compartment given
an Erlang dwell time with `via erlang(...)` should behave like any other
compartment when people are moved into it, out of it, or between its strata,
without the modeller writing stage bookkeeping.

Related: `docs/dev/proposals/2026-06-26-transition-delay-laws.md` (the `via`
clause; its §7 reserved competing exits), gh#331 (competing exits), gh#568
(typed generated dimension names).

## Background

`via erlang(stages = k, rate = r)` on a transition gives its source compartment
an Erlang-distributed dwell time by splitting it into `k` sub-compartments, the
_stages_, chained at rate `k * r`.

```camdl
latency[a in age] : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
```

`E[pre]` becomes `E_pre_s1`, `E_pre_s2`, `E_pre_s3`. This proposal calls a
compartment with a `via` law a _staged compartment_, its set of stages its
_residence_, and one stage a _residence location_. The dimensions the modeller
declared (`age`) are _population dimensions_.

What works today on a staged compartment:

- A rate, `let`, observation or `balance` entry may write `E[a]` (or bare `E`)
  and gets the sum over stages.
- `init { E[pre] = 10 }` and every transition arriving in `E` land in stage 1.

What does not, verified on `33f03defa`:

- `transfer(fraction = 0.2, from = E[pre], to = E[sch])` is refused (E287, "only
  1 of 2 dimensions indexed"), as is `transfer(from = S[pre], to = E[pre])`.
- `add(E[pre], 5)` is refused (E265).
- `death_E[a in age] : E[a] --> @ mu * E[a]` beside the staged `latency` is
  refused (E246, single exit).
- `aging_E : E[pre] --> E[sch] @ ar * E[pre]` is refused with a misleading E503
  ("unknown compartment 'E_pre'").
- The forms that do compile name the generated stage dimension
  (`[st in __latency_stage]`, `E[pre, s2]`) and stop compiling at `stages = 1`,
  where no stage dimension exists.

Separately, a fraction transfer on integer compartments moves
`floor(count * fraction)` per cell (`rust/crates/sim/src/effects.rs`,
`resolve_action`). Applied stage by stage, the number moved depends on the stage
count: eight stages of 10 people at fraction 0.2/3 move nobody, where one
unstaged compartment of 80 moves 5.

## Principle

The modeller says which population an event affects and at what total rate or
amount. The compiler decides which residence locations the affected people come
from and go to, by fixed rules. A modeller who wants a specific stage says so
with `stage(...)`.

Two properties follow and are tested as invariants:

- **Stage-count invariance of amounts.** An aggregate transfer or exit moves the
  same total, given the same compartment total, whatever `stages` is.
- **Same text for every stage count.** A model that compiles at `stages = 3`
  compiles at `stages = 1` with only that number changed.

## Residence mapping rules

These apply identically to ordinary `@` transitions, scheduled interventions,
events and reactive interventions. A _residence reference_ is a reference to a
staged compartment with all its population dimensions indexed and no stage
(`E[pre]`; bare `E` when `E` has no population dimensions).

| Movement                                          | Residence mapping                             |
| ------------------------------------------------- | --------------------------------------------- |
| Same compartment, different population stratum    | Keep the stage (`E[pre, s]` to `E[sch, s]`)   |
| Unstaged source to staged destination             | Enter stage 1                                 |
| Staged source to unstaged destination             | Draw from all stages                          |
| Staged source to a _different_ staged compartment | Draw from all stages, enter stage 1 (warning) |
| `add` to a staged compartment                     | Enter stage 1                                 |
| Source or destination written with `stage(...)`   | That location only                            |

- **Different staged compartment.** The destination dwell restarts at stage 1,
  as every existing inflow does. Because the modeller may have meant the
  progression to continue (an isolation compartment mirroring the infectious
  period), an ordinary `@` transition or a `transfer` from one staged
  compartment into a different staged one emits a warning: "this movement
  restarts '<dst>' at stage 1; use explicit `stage(...)` endpoints if
  continuation is intended". The `via` transition's own exit does not warn.
  Writing either endpoint with `stage(...)` silences it.
- **`set`.** `set` on a residence reference with more than one stage stays
  refused: a total has no single meaning across stages. `set(stage(E, 2), x)` is
  allowed.
- **A stage into the same compartment's residence.** A `transfer` whose source
  names one stage and whose destination is a residence reference of the same
  compartment (`from = stage(E[pre], 2), to = E[sch]`) is refused (E367): the
  table gives both "keep the stage" and "that location only" for it. The
  destination stage is written explicitly.
- **Population dimensions are never pooled.** `transfer(from = E, to = R)` with
  both stratified by `age` pairs cell for cell over `age`, as today, and groups
  only the stages within each age cell.

## Competing exits

An ordinary `@` transition whose single source is a residence reference, written
beside the compartment's `via` transition, is a _competing exit_: a hazard that
acts throughout the dwell and does not depend on the stage.

```camdl
latency[a in age] : E[a] --> I[a] via erlang(stages = 3, rate = sigma)
death_E[a in age] : E[a] --> @ mu * E[a]
treat[a in age]   : E[a] --> T[a] @ cmax * E[a] / (K + E[a])
```

The written rate `R` is the total propensity for the whole compartment cell, as
everywhere in camdl. It is allocated to stage `s` in proportion to occupancy:

```
R_s = R * E_s / E     if E > 0
R_s = 0               if E = 0
```

where `E` is the sum over the stages of that cell. Every stage then faces the
same per-person hazard `R / E`, the stage rates sum to `R`, and the `via`
progression runs unchanged alongside.

- **Lowering.** One copy of the transition per stage, source `E[a, s]`, as an
  indexed family over the stage. `R` is evaluated with residence references
  summed over stages (the existing rewrite). The destination follows the
  residence mapping rules.
- **Linear rates simplify, and must.** When the rate is a product in which the
  source residence reference appears exactly once as a factor, and the source
  compartment appears nowhere else in the expression, the stage rate is that
  product with the factor replaced by `E[a, s]` (`mu * E[a, s]`). A quotient
  whose divisor does not mention the source compartment counts as a product
  (`mu * E[a] / N_ref`). A sum or difference of such products, every term one,
  lowers term by term (`cmax * E[a] + mu * E[a]` gives
  `cmax * E[a, s] + mu * E[a, s]`); a term that does not read the source
  (`cmax * E[a] + mu * K`) makes the rate nonlinear. This is the same value and
  is byte-identical to the hand-written per-stage model. All other rates lower
  to the guarded quotient above.
- **Families over compartments.** A transition family bound over the compartment
  set (`bg[a in age, c in compartments] : c[a] --> @ mu * c[a]`) is a competing
  exit for each staged member and an ordinary exit for the rest.
- **Numerical contract.** For linear rates the compiled transitions equal the
  hand-written per-stage model exactly. For other rates the sum of the stage
  rates equals `R` to floating-point tolerance.
- **Meaning of `mean` and `rate`.** With a competing exit, the `via` law's
  `mean` is the dwell absent the competing exit. With a constant per-person
  hazard `mu`, the fraction completing an Erlang-`k` dwell with stage rate
  `k * sigma` is `(k*sigma / (k*sigma + mu))^k`.
- **Observations and flows.** A flow reference to the transition as written
  (`incidence(death_E[a])`) sums its stage copies. The stage axis is exempt from
  the refusal to pool a flow over an unbound dimension (E280). The per-stage
  flow columns are named like any indexed family, with the stage level last
  (`death_E_pre_s1`).
- **Lineage.** A `#[lineage]` competing exit takes every stage cell of its
  source cell as the source (`E_pre_s1` to `E_pre_s3` for each copy of
  `treat_pre`), as the unstaged transition takes `E_pre`, so the stages are
  never parents of one another and the E601 check gives the same answer at every
  stage count. A conditional whose predicate reads no parent is linear for
  lineage when its branches are, with weight `if p then w_a else w_b`, so the
  guarded share keeps the written rate's parents: copy `s` of
  `mu * E[a] * I[a] / (K + E[a])` has the parent `I[a]` with weight
  `if E > 0 then mu * E / (K + E) * E_s / E else 0`, and the copies' weights sum
  to the unstaged weight.
- **`stages = 1`.** The source is unstaged and the transition is an ordinary
  second exit; nothing is rewritten. The refusals below therefore do not apply
  at `stages = 1`: an `overdispersed(...)` second exit compiles there and is
  refused at any larger stage count. This is the one exception to "same text for
  every stage count", and the diagnostic says so.

Refused, each with its own diagnostic naming the reason:

- A competing exit with `overdispersed(...)` noise. The noise multiplier is
  drawn per transition, so `k` copies would make the variance of the total
  depend on the stage count.
- A competing exit that consumes more than one individual of the staged source
  per event (source stoichiometry above 1).
- A transition with two or more staged sources, or with a staged source beside
  any other source (`E + S --> ...`): the allocation rule is defined for a
  single source only.
- A second `via` on the same source (unchanged, E246, with its message narrowed
  to this case).
- Any competing exit or residence-reference endpoint on a `hyper_erlang`
  compartment. `hyper_erlang` lowers to flat per-branch compartments, not a
  stage dimension; extending these rules to it is a follow-up on gh#333.

## Transfers and `add`

A `transfer` endpoint may be a residence reference. The lowering depends on the
source:

- **Unstaged source, staged destination.** One ordinary transfer into stage 1.
- **Staged source (more than one stage).** One _grouped transfer_ per paired
  population cell, listing a (source, destination) cell pair per stage. The
  destination of pair `s` is the same stage of the destination (same
  compartment), the single destination cell (unstaged), or the destination's
  stage 1 (different staged compartment).
- `add` to a residence reference is an ordinary `add` to stage 1.

### The grouped transfer action

A new IR action, beside `fraction_transfer` and `absolute_transfer`:

```ocaml
type grouped_amount =
  | GroupedFraction of { fraction : expr; fraction_grad : (string * expr) list }
  | GroupedCount    of { count : expr }

type grouped_transfer = {
  pairs  : (string * string) list;   (* (src cell, dst cell), stage order *)
  amount : grouped_amount;
}
```

Serialized as
`{"grouped_transfer": {"pairs": [["E_pre_s1","E_sch_s1"], ...],
"fraction": ..., "fraction_grad": ...}}`
or with `"count"` in place of the two fraction fields. Validation: at least two
pairs, source cells distinct, every cell declared, all cells in one arena (all
integer or all real).

Semantics, with `n_i` the snapshot value of source `i` and `N` their sum:

- **Integer compartments, fraction `f`** (clamped to [0, 1]):
  `M = floor(N * f)`.
- **Integer compartments, count `c`:** `M = min(round(c), N)`. A negative `c` is
  an error on every path (integer, real, ODE): there is no allocation of a
  negative `M`, and a count out of a staged compartment is not read as a count
  into it.
- **Allocation of `M`.** Stage `i` gets `floor(M * n_i / N)`, and the remaining
  `M - sum` individuals go one each to the stages with the largest remainders
  `M * n_i mod N`, ties broken toward the earlier stage. Computed in integer
  arithmetic. This never takes more than `n_i` from stage `i`. With `N = 0`
  nothing moves. The tie-break favours earlier stages by at most one individual
  per transfer; a randomised allocation (multivariate hypergeometric) is a
  possible later option and is not part of this proposal.
- **Real compartments, fraction:** `x_i = n_i * f`, exactly as separate fraction
  transfers.
- **Real compartments, count:** `x_i = min(c, N) * n_i / N`, zero when `N = 0`.
- **Evaluation.** The amount expression is evaluated once per firing. The action
  reads the same snapshot, runs in the same phase and obeys the same ordering as
  a fraction or absolute transfer at that position; conflict analysis
  (overdrawing drains, coincident `set`) treats it as a drain of every source
  cell.
- **Two grouped transfers at one instant.** Two grouped transfers that fire at
  the same instant as events and share a source cell are refused on every
  backend, whatever their fractions and whatever the cell holds. Each divides
  its own `M` and gives the individuals left over to the largest remainders, so
  both can take the same individual: from stages (1, 1), two transfers of one
  half each take stage 1 and would leave it at -1. The refusal is raised where
  overdrawing drains are, names both events and the shared cell, and does not
  depend on the state. Scheduled interventions apply sequentially and are not
  affected.
- **A grouped transfer beside a plain one.** A grouped fraction transfer and
  plain fraction transfers from one of its source cells are allowed when the
  fractions sum to at most 1, by the existing check, and cannot overdraw: from a
  cell holding `n` the grouped transfer takes at most `ceil(n * f)` (its share
  of `M <= N * f` is at most `ceil(M * n / N)`) and each plain one
  `floor(n * f')`; when `n * f` is not whole, the floors sum to a whole number
  below `n - n * f`, so to at most `n - ceil(n * f)`.
- **Gradients.** The fraction form has the per-pair sensitivity rule of
  `fraction_transfer`. The count form is nonsmooth and is treated as
  `absolute_transfer` is.

`ir/VERSION` goes 0.45 to 0.46. No existing compiled model changes: every form
that emits a grouped transfer is refused today.

## Explicit stages

Two constructs replace every user-written reference to a generated stage
dimension.

```camdl
stage(E[pre], 2)          # one residence location: stage 2 of E[pre]
stages(E)                 # the residence locations of E, as a binder domain

isolate_late : transfer(fraction = 0.8, from = stage(I[pre], 3), to = Q[pre]) at [d]
move[st in stages(E)] : transfer(fraction = 0.2, from = stage(E[pre], st), to = stage(E[sch], st)) at [d]
late_death : stage(E[pre], 3) --> @ mu_late * stage(E[pre], 3)
init { stage(E[pre], 2) = 10 }
```

- `stage(ref, loc)` takes a residence reference and a residence location. For
  `erlang` a location is an integer literal from 1 to `stages`, or a variable
  bound by `in stages(...)` of the same compartment. It is valid as a state
  expression and as a transition, transfer, `add`, `set` and `init` endpoint.
  For `hyper_erlang` a location is a branch and a phase; that form is reserved
  and refused with a diagnostic until gh#333.
- `stages(C)` is valid only as a binder domain (`[st in stages(E)]`), on a
  declaration and in `sum(st in stages(E), ...)`. A variable bound over
  `stages(E)` cannot index `stage(I[...], st)` (E371), nor a population
  dimension, a parameter, a table or a flow, and is not a value (E372). The one
  other thing it indexes is a `let` family declared over `stages(E)`.
- At `stages = 1`, `stages(E)` has one element and `stage(E[pre], 1)` is
  `E[pre]`. The same holds for a compartment with no `via` law. A declaration
  bound over a one-element `stages(E)` is the single declaration it would be
  without the binder: `move[st in stages(E)]` is `move`, not `move_s1`, as a
  competing exit is `death_E_pre` there and not `death_E_pre_s1`.
- An out-of-range literal, a location for the wrong compartment, or `stage` on
  an unstaged compartment with a location other than 1, is an error naming the
  compartment's stage count (E370). A literal location above 1 therefore does
  not compile at `stages = 1`: "same text for every stage count" holds for a
  model written with location 1 and stage variables.
- **The reference.** The first argument of `stage` is a residence reference
  written with the compartment's own name. A compartment bound by
  `c in compartments`, or a reference with a population dimension left
  unindexed, is refused (E369).
- **Families over stages.** The stage binder of a transition goes last in its
  expanded names, wherever it is written (`shift_pre_s2`), as the stage level
  does for a competing exit, and a flow reference to the transition as written
  reads its stage members by the same rule.
- **`init`.** `stage(E, 2) = 10` is an `init` entry. A binder over `stages(E)`
  in the bracket form (`E[st in stages(E)] = ...`) is refused (E374): the
  bracket binds population indices.
- **Observation streams.** A binder over `stages(E)` on a stream header is
  refused (E374): a stream family is matched to a data column per index, and
  data has no column for a stage. A stream reads one location with
  `projected = stage(E, 1)`.
- **The declared model.** The declared-model section records `stage(E[a], 2)` as
  `E` indexed by `a` and the call `stage(2)`, and `st in stages(E)` as a binder
  whose domain is the text `stages(E)`. Its schema (version 1) has no
  constructor for either; one would be a schema change of its own.
- There is no stage-count accessor. A transfer fraction or rate must not depend
  on the number of stages; the grouped transfer and the allocation rule remove
  the legitimate reasons to compute one.

A stage is a stochastic progression state, not an elapsed-time interval:
`stage(E, 3)` selects people who have completed two stage transitions, not
people infected between two dates.

### Existing spellings

- `[st in __latency_stage]`, and any user-written reference to a generated stage
  dimension by name: refused (E375), with a diagnostic naming
  `stages(<compartment>)`. This holds at `stages = 1` too, where the dimension
  is not generated. A parameter or table declared over the dimension has no
  replacement: nothing is declared over stages.
- A stage level name as an index (`E[pre, s2]`): refused (E376), with a
  diagnostic naming `stage(E[pre], 2)`.
- The refusal is made on the declarations as parsed, before the `via` lowering
  writes its own references into the stage dimension, so it cannot reach them.
- Expanded cell names (`E_pre_s2`): unchanged. They are how every stratified
  cell is named in trajectory columns and may still be written where a cell name
  is accepted; they do not exist at `stages = 1`.

Each is recorded in `docs/language-changes.md`.

## Not in this proposal

- A spelling that spreads an initial value across stages. The stage-1 default
  stays and is documented beside the explicit `stage(...)` form.
- Phase preservation between different staged compartments as a declared
  property. It is written with explicit `stage(...)` endpoints until a model
  needs more.
- `hyper_erlang` (gh#333), other phase-type laws (gh#329), multiple residence
  clocks on one compartment.

## Implementation order

Each step is its own set of commits and passes the full gate before the next.

1. **Competing exits.** Compiler only. Record a residence (compartment,
   population index count, stage levels, law, `via` transition) where the `via`
   pass creates the stage dimension; lower competing exits and plain
   same-compartment transitions per the rules; the new refusals; flow summing.
2. **Transfers and `add`, with the grouped transfer.** The IR action on both
   sides (`ocaml/lib/ir`, `rust/crates/ir`), `ir/VERSION` 0.46, the runtime
   rules in `effects.rs` (discrete resolver, real-valued path, sensitivity rule,
   conflict analysis), the endpoint lowering, the restart warning.
3. **`stage()` and `stages()`**, and the refusal of the old spellings.

## Tests

- **Manual equivalence.** A model with a linear competing exit compiles to the
  same transitions as the hand-written per-stage model.
- **Propensity preservation.** For a nonlinear competing exit, the stage rates
  sum to the written rate at several states, including an empty compartment.
- **Survival fraction.** Simulated fraction completing the dwell against
  `(k*sigma / (k*sigma + mu))^k`.
- **Stage-count invariance.** The same transfer text at `stages = 1` and
  `stages = 3`, given equal compartment totals, moves the same total, on integer
  and real compartments, for `fraction` and `count`.
- **Allocation.** Conservation, never more than a stage holds, the tie-break,
  `N = 0`, and the worked case of 5 people in stages (3, 1, 1) at fraction 0.4
  moving 2 as (1, 1, 0).
- **Ordering.** A grouped transfer beside other actions at the same time, as an
  intervention and as an event, gives the totals the phase rules prescribe; a
  grouped drain beside a plain drain is refused as two fraction transfers would
  be when the fractions sum above 1, and never overdraws a stage when they do
  not (a property check over random counts and fractions); two grouped drains
  sharing a source cell are refused whatever the state.
- **Backends.** ODE, Gillespie and chain-binomial agree on the real-valued
  semantics where both apply; inference runs with a grouped transfer are
  reproducible under a fixed seed.
- **No repeated effect.** An indexed intervention family over population
  dimensions fires one grouped transfer per population cell.
- **Flows.** `incidence(death_E[a])` equals the sum of the per-stage flow
  columns.
- **Diagnostics.** Each refusal and the restart warning, with its hint compiling
  where it suggests code.
