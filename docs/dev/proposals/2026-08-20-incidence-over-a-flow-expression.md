# `incidence()` should take a flow expression, parsed into a type that cannot hold anything else

Date: 2026-08-20 Status: SUPERSEDED — do not implement Issue: gh#669

> **Superseded before implementation.** Three adversarial reviews established
> that this question was already decided, in the opposite direction, by
> [`2026-07-31-aggregation-semantics.md`](2026-07-31-aggregation-semantics.md)
> **Decision 8 — "`incidence` becomes composable"** — whose Increment B lowers
> `incidence(a) + incidence(b)` to a new `WeightedFlowSum` variant while
> **unit-weight forms keep their existing `CumulativeFlowSum` lowering**, i.e.
> exactly the node this document targets, from exactly the surface the
> downstream reached for first.
>
> B1 also rebuts this document's central objection in advance: it is
> deliberately _not_ a new `Expr` constructor, which is what keeps
> `temporal_kind()` total and stops a flow read being representable inside a
> rate expression. That is a better answer than forbidding composition. Decision
> 5 additionally _removes_ prevalence's multi-argument form — the precedent this
> document's justification rests on.
>
> **The right move is to fold the flow-union case into Increment B as an
> unblocking slice, or to reopen Decision 8 head-on with the comparison
> argued.** The two cannot coexist: they leave `2 * incidence(a)` legal and
> `incidence(2 * a)` an error, for reasons no modeller can derive.
>
> Kept for the evidence below, which survives whichever surface wins — and for
> the record of a proposal that failed review for not reading the repo it was
> proposing against.

### Corrections to this document, established by review

- **The acceptance test named in "Verification" cannot pass.** A junction model
  and a direct-sum model do _not_ produce the same log-likelihood: measured
  −73.37 vs −72.18 at identical parameters, 4000 particles, far outside the
  ±0.07 Monte-Carlo spread — because of the drain dwell listed as cost #4. The
  exact equivalence available is `incidence(tr[child] + tr[adult])` against
  `sum(a in age, incidence(tr[a]))`.
- **"The runtime path is already exercised" is false.** One model in the repo
  emits a ≥2-name `CumulativeFlowSum`, and its only Rust consumer excludes
  observations from its hash — three models differing only in their observations
  block share one baseline hash. Every `StreamProjection::FlowSum` in the Rust
  test suite has a single index.
- **The gradient is fine**, contrary to the suspicion that prompted the review:
  a two-flow model matches an ODE-identical one-flow model under `nuts` to
  8.8e-12 across 40 draws.
- **`parse_flow_expr : expr -> result` cannot perform the overlap check assigned
  to it.** At least four spellings collapse to one post-expansion name — the
  mangled form (`infection_child`, legal today and used in the committed test
  suite), branch suffixes (`outcome_R`), named-index reordering, and a per-cell
  diagonal collision that is not a property of the expression at all. The check
  belongs after expansion _and_ in both IR validators, since `camdl simulate`
  accepts raw IR.
- **Cost #1 overstates by one compartment**: the junction adds `P`, not `P` and
  `M`.
- **E280's behaviour for a stratified family inside a union was left
  undecided**, which `.claude/rules/proposals.md` forbids in a shipped proposal.

Bugs found while reviewing this document, none of which depend on it: gh#677
(indexed stream scored against the pooled total), gh#678 (duplicate and empty
flow lists), gh#679 (E304's missing hint), gh#680 (`nuts` × `ode` multi-cadence
mis-binning).

## The problem, as a modeller hits it

An observed stream is often the sum of two flows. A confirmed-deaths series
counts deaths occurring in the community _and_ deaths occurring in care; an
admissions series may count arrivals through more than one route. The quantity
observed is one thing — a count of events over the reporting window — that the
model happens to produce through several transitions.

CAMDL cannot say that. `projected` names exactly one transition, so a modeller
who needs the sum must change the _model_ to suit the _observation_: route both
flows into a junction compartment, add a fast transition to drain it, and
observe the drain.

```camdl
die_community : I --> P @ gamma * q_comm * I
die_facility  : C --> P @ f_cfr * delta * C
register      : P --> M @ 5.0 'per_day * P      # plumbing, not epidemiology
...
projected = incidence(register)
```

Costs, in the order they matter:

1. **Two extra compartments in the latent state.** `P` and `M` are sampled on
   every PGAS sweep, for models whose inference cost is already the binding
   constraint. This is the largest cost and the least visible one.
2. **The rate constant is invisible.** `5.0 'per_day` is an inline literal in a
   transition rate — not a parameter, not `[fixed]`, absent from every parameter
   table, from `fit summary`, and from run provenance. It is findable only by
   reading the transitions.
3. **The compartment diagram stops describing epidemiology.** `P` is not a state
   of anybody; `M` exists to give it somewhere to go.
4. **An unchosen dwell.** At `dt` = 1 d, `5.0 'per_day` registers 99.33%
   same-day, so 0.67% of deaths slip a bin. Small, and it shrinks as `dt`
   coarsens — but nobody chose it.

It scales with the model rather than the problem: one downstream project carries
this construction in nine model files.

## Why the obvious workarounds fail, and which one fails silently

Measured with `camdlc` at `70247af6`:

```
incidence(a) + incidence(b)        error[E100]: undeclared function 'incidence'
incidence(a, b)                    error[E203]  (the arity fix)
incidence(a + b)                   error[E507]: unknown transition referenced: '?'
incidence(a + I)                   error[E507]: unknown transition referenced: '?'
let total = a + b; incidence(total) error[E507]: unknown transition referenced: 'total'
projected = <the rates longhand>   COMPILES — and is a different quantity
```

Two things worth reading off this.

**The head-position dispatch does not fall through to the expression resolver.**
`ProjDerived (EFuncCall ("incidence", args))` matches before the generic path,
and every non-name shape lands on a `| _ -> CumulativeFlow "?"` sentinel. So
`incidence(<anything>)` can never become a `DerivedExpr`. What is broken is the
_diagnostic_: `'?'` names nothing the modeller wrote.

**`let` does not help, and the error misleads.** `let` binds in the expression
namespace over state; `incidence` looks its argument up in the transition
namespace. The binding is invisible to the head, and the message asserts that
`total` is not a transition to a reader who believes they just defined it.

**The longhand form is the dangerous one.** Measured downstream with `pfilter`
at fixed parameters, 1,000 particles, one seed: `incidence(register)` scored
−1036.74, the rate expression −1042.22. Five log units on 232 observations —
indistinguishable from a refactor.

## The temporal kinds, and why the restriction is real

`Projection::temporal_kind()` (`rust/crates/ir/src/observation.rs:47`) is total
over the five projection variants:

| variant                                      | kind       | runtime meaning                                                               |
| -------------------------------------------- | ---------- | ----------------------------------------------------------------------------- |
| `CumulativeFlow`, `CumulativeFlowSum`        | `Interval` | a per-stream accumulator sums flows across substeps; resets when a bin closes |
| `CurrentPop`, `CurrentPopSum`, `DerivedExpr` | `Instant`  | state read once at the observation time                                       |

`prevalence(R + D)` works because prevalence **steps aside**: compartments are
variables in the expression language, so it lowers to a plain `DerivedExpr` —
verified against emitted IR —

```
prevalence(I)       ->  {"current_pop": "I"}
prevalence(R, D)    ->  {"derived_expr": {"pop_sum": ["R", "D"]}}
prevalence(R + D)   ->  {"derived_expr": {"pop_sum": ["R", "D"]}}
incidence(recovery) ->  {"cumulative_flow": "recovery"}
```

— and `DerivedExpr` is `Instant`, which is what prevalence already is. The
degradation costs it nothing.

Incidence cannot take that escape. It is `Interval`, and there is no
interval-valued expression form: a flow is not a variable in the expression
language, it is an accumulator the integrator maintains with window-scoped reset
semantics a state read does not have. An incidence that degraded to
`DerivedExpr` would be evaluated pointwise — which is the −5.5 log-unit error
above, exactly.

So keeping `incidence(...)` in head position is load-bearing, not an accident.
What is missing is the ability to name more than one flow inside it.

## The design

```camdl
projected = incidence(die_community + die_facility)
```

`+` is the right notation and means what it says. Transitions are disjoint — an
event passes through exactly one — so the count of the union is the sum of the
counts. This is addition of counts, not a borrowed symbol.

### The type is the guard

The surface is the smaller half. The mechanism is to stop matching syntactically
and start parsing:

```ocaml
type flow_ref  = { tr : string; idx : index_item list }   (* a transition, maybe a cell *)
type flow_expr = flow_ref list                            (* non-empty; a union of flows *)

val parse_flow_expr : expr -> (flow_expr, diagnostic) result
```

`parse_flow_expr` accepts a transition name, an indexed cell, and `+` between
them. Everything else returns a located diagnostic naming what the modeller
actually wrote.

The property this buys: **no constructor of `flow_expr` can produce a
`DerivedExpr`.** The `Instant` path becomes unreachable from the incidence head
by construction rather than by match-arm ordering, and the `"?"` sentinel
disappears because failure returns a diagnostic instead of a structurally valid
`Projection` carrying a bogus name.

That ordering-based guard is what failed twice already: gh#669 (the arm matched,
read one argument with `List.assoc_opt ""`, discarded the rest) and the sentinel
itself. This is `.claude/rules/rust-conventions.md`'s "parse at the boundary;
don't pass raw and validate", at the one boundary where it earns its keep.

### What must be rejected, and one hazard that is new

- `incidence(2 * a)`, `incidence(a - b)` — not expressible: `CumulativeFlowSum`
  is a list of names with no coefficients. A reporting fraction belongs in the
  likelihood (`mean = rho * projected`), where it is conventional and stays
  dimensionless.
- `incidence(a + I)` — a flow plus a compartment. Rejected naming `I` as not a
  transition, rather than `'?'`.
- `incidence(total)` after a `let` — rejected, and the message should say the
  argument must be a transition and that `let` binds state expressions, not
  flows.
- **`incidence(a + a)`, and `incidence(family + family[cell])` — overlap.**
  `validate.ml:164` checks each `CumulativeFlowSum` name exists but **does not
  check for duplicates**, so these would silently double-count. Unreachable
  today (multi-flow is E203; the `sum(a in dim, …)` form generates distinct
  strata by construction) but **this proposal makes it reachable**, so
  `parse_flow_expr` must reject overlap, including a family against its own
  cell.

### It requires no new representation

- `Ir.CumulativeFlowSum of string list` (`ocaml/lib/ir/ir.ml:425`) already
  exists, with serialization (`serde.ml:875,885`), validation
  (`validate.ml:164`), dimension checking (`dimcheck.ml:816,1084`), constant
  folding (`constant_fold.ml:131`) and lint (`lint.ml:140`).
- `StreamProjection::FlowSum(Vec<usize>)` on the Rust side, classified
  `Interval`, with accumulate-and-reset already implemented.
- `expander.ml` already emits `CumulativeFlowSum` on two live paths: a bare
  `incidence(T)` over a stratified family (~8099), and the explicit
  `sum(a in dim, incidence(tr[a]))` form (~8304).

The IR does not move, `ir/VERSION` does not move, no golden moves.

### After

```camdl
compartments { S, E, I, C, G, R }          # P and M are gone

die_community : I --> D @ gamma * q_comm * I
die_facility  : C --> D @ f_cfr * delta * C

projected = incidence(die_community + die_facility)
```

Two compartments out of the latent state, one transition and one invisible
constant deleted, per model file.

## Alternatives considered

**`incidence(a, b)`** — multi-positional, mirroring `prevalence(R, D)`.
Rejected: `f(a, b)` reads as "two arguments" in every language a modeller has
used, and "several positionals desugar to their sum" is a convention you have to
be told. That `prevalence` already carries the obscurity is precedent, not
justification. Nothing stops both spellings later; `+` should be the one the
documentation teaches.

**A new head, `incidence_sum(...)`** — rejected. It is vocabulary for a special
case: `incidence` already means "count of these events over the window", and
naming more than one flow does not change what is counted. It would want a
`prevalence_sum` sibling for symmetry. Most importantly it guards nothing —
`incidence_sum(a + I)` has exactly the problem `incidence(a + I)` has. The name
does not constrain; `flow_expr` does.

**Make `incidence()` a first-class expression term** — what the downstream
originally asked for. Rejected: it admits `incidence(a) + prevalence(I)`, a
windowed count added to an instantaneous state, which the temporal-kind
classification cannot represent and which has no defensible semantics.

**Do nothing; document the junction.** Rejected on the four costs above, and
because until `70247af6` our own diagnostic _recommended_ the junction — a
construction that distorts the model, hides a constant from provenance, adds two
compartments to the sampled latent state, and imposes an unchosen delay.

## If this passes review, it is a language-surface change

It adds an accepted form to the DSL, so per `.claude/rules/dsl-surface.md` it
does not land without:

- **`docs/camdl-language-spec.md`** — the observation/projection section gains
  `incidence(a + b)` beside `prevalence(R + D)`, and, more importantly, states
  the rule that distinguishes them: `prevalence`'s argument is an expression
  over _state_, `incidence`'s is a union of _flows_, and the two are not
  interchangeable because a flow is accumulated over the reporting window while
  state is read at an instant. That paragraph is the one a modeller needs in
  order not to write the longhand rate expression. **Hand-edit only** — the file
  embeds `<!-- camdl-doctest-preamble -->` markers and fenced `camdl` snippets
  that `dprint`'s reflow breaks.
- **`docs/language-changes.md`** — a dated entry. This one _widens_ what
  compiles rather than tightening it, so no existing model breaks; the entry is
  a "you no longer need the junction" note with the before/after.
- **`docs/dsl-cheatsheet.md`** and **`docs/user-features.md`** — the projection
  row, wherever `prevalence(R, D)` is already shown.
- **`docs/dev/warning-catalog.md`** — a row per new emit site in
  `parse_flow_expr` (non-flow argument, overlap, unsupported arithmetic).

## Verification

- Red first: `incidence(a + b)` must lower to `CumulativeFlowSum ["a"; "b"]`,
  asserted against emitted IR, not against compile-success.
- **The equivalence test that matters:** a junction model and the direct-sum
  model must produce the _same_ log-likelihood at fixed parameters. That is what
  proves the new form means what the workaround meant.
- Rejection tests, each asserting the diagnostic names what the modeller wrote
  and never `'?'`: `2 * a`, `a - b`, `a + I`, a `let`-bound name, `a + a`, and a
  family against its own cell.
- Negative controls that must not move: `incidence(T)`; `incidence(T[i])`; a
  bare `incidence(T)` over a stratified family still lowering to
  `CumulativeFlowSum`; `sum(a in dim, incidence(tr[a]))`; `prevalence(R, D)`;
  `prevalence(R + D)`.
- No golden may move.
