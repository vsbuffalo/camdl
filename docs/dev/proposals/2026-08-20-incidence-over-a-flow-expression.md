# `incidence()` should accept several flows, the way `prevalence()` accepts several compartments

Date: 2026-08-20 Status: proposed Supersedes: the first draft's
`incidence(a + b)` form, dropped — see "a list, not an expression" Issue: gh#669

## The problem, stated as a modeller hits it

An observed stream is often the sum of two flows. A confirmed-deaths series
counts deaths occurring in the community _and_ deaths occurring in care; a
hospitalisation series may count admissions arriving through more than one
route. The quantity observed is one thing — a count of events over the reporting
window — that the model happens to produce through several transitions.

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

This is not a stylistic complaint. The workaround has four concrete costs:

1. **`P` is not a state of anybody.** A reader must decode a compartment that
   exists only to merge two arrows, and `M` exists only to give it somewhere to
   go. The compartment diagram no longer describes the epidemiology.
2. **The rate constant is un-provenanced.** `5.0 'per_day` is an inline literal
   in a transition rate — not a parameter, not `[fixed]`, absent from every
   parameter table, from `fit summary`, and from run provenance. It is findable
   only by reading the transitions.
3. **The junction is not free.** It imposes a real dwell. At `dt` = 1 day, 0.67%
   of deaths register a day late — small, but a bias nobody chose, whose size
   changes with `dt`.
4. **It scales with the model, not the problem.** One downstream project carries
   this construction in nine model files.

## Why the natural workarounds fail, and one of them fails silently

`incidence(a) + incidence(b)` is `error[E100]: undeclared function 'incidence'`.
`incidence(a, b)` used to compile and silently observe only `a`; as of gh#669 it
is a located error. Neither is a route.

The dangerous one is writing the flows out longhand:

```camdl
projected = gamma * q_comm * I + f_cfr * delta * C     # compiles, WRONG
```

Measured downstream with `pfilter` at fixed parameters, 1,000 particles, one
seed: `incidence(register)` scored −1036.74, the rate expression −1042.22. Five
log units on 232 observations — indistinguishable from a refactor.

The cause is not carelessness. It is a **temporal-kind** error the language does
not currently name. A projection has exactly one temporal kind
(`rust/crates/sim/src/inference/multi_stream_obs.rs:201-206`):

- `FlowSum` → `Interval` — accumulated over the reporting window, reset each bin
- `IntCompSum`, `Expr` → `Instant` — evaluated at the observation time

`incidence(...)` produces `FlowSum`; **any** other expression produces `Expr`,
hence `Instant`. So the longhand rate expression is not an approximation of the
incidence — it is a different kind of quantity, evaluated pointwise. At `dt` = 1
day a per-day rate and a one-day accumulated count coincide numerically, which
is what makes the error survive review; at weekly bins it is wrong by roughly
the window length.

## The design

**Keep `incidence(...)` in head position. Enrich its argument.**

Head position is not an accident to be corrected — it is what guarantees a
projection has one coherent temporal kind. Making `incidence()` a general
expression function would permit `incidence(a) + prevalence(I)`, a windowed
count added to an instantaneous state, which has no meaning and which the
`TemporalKind` classification could not represent. The restriction is load-
bearing and stays.

What is missing is not expressiveness _outside_ `incidence()` but the ability to
name more than one flow _inside_ it.

### Why `prevalence` has an expression form and `incidence` cannot

This is worth stating precisely, because the obvious symmetry argument is wrong.

`prevalence(R + D)` is not prevalence doing something clever. It is prevalence
**stepping aside**. Compartments are variables in the expression language, so
`R + D` is an ordinary expression, and the projection lowers to a plain
`DerivedExpr` — verified against emitted IR:

```
prevalence(I)      ->  {"current_pop": "I"}
prevalence(R, D)   ->  {"derived_expr": {"pop_sum": ["R", "D"]}}
prevalence(R + D)  ->  {"derived_expr": {"pop_sum": ["R", "D"]}}
incidence(recovery)->  {"cumulative_flow": "recovery"}
```

Only the single-compartment case keeps a projection node at all. Everything else
is handed to the expression evaluator.

That escape is available to prevalence for one reason: **`DerivedExpr` is
classified `Instant`, and prevalence is `Instant`.** Degrading to an ordinary
expression costs prevalence nothing, because evaluating a state expression at
the observation time is exactly what prevalence means.

Incidence cannot take that escape. It is `Interval` — accumulated over the
reporting window and reset each bin — and there is no interval-valued expression
form. A flow is not a variable in the expression language: it is an accumulator
the integrator maintains, with window-scoped reset semantics that a state read
does not have. An incidence that degraded to `DerivedExpr` would land in the
`Instant` branch and be evaluated pointwise.

**That is not a hypothetical: it is the −5.5 log-unit error above.** Writing the
flows longhand _is_ the degradation, and it silently succeeds.

So prevalence's expression form and incidence's lack of one are the same fact
seen twice, not an inconsistency to be corrected.

### Therefore: a list, not an expression

The first draft of this proposal offered both `incidence(a, b)` and
`incidence(a + b)`. The second should be dropped.

If `a` and `b` are not expression values, then `+` between them is not the
expression language's `+`. Spelling a list with plus signs invites exactly the
arithmetic the representation cannot hold — `incidence(2 * a)`,
`incidence(a - b)` — which we would then have to reject with an error explaining
that the `+` the modeller just used was not really addition. That is a worse
surface than not offering it.

The comma is honest. It is a list separator, it carries no algebraic promise,
and it is already what `prevalence(R, D)` uses:

```camdl
projected = incidence(die_community, die_facility)
```

|              | one thing       | several things                   | an expression                                                          |
| ------------ | --------------- | -------------------------------- | ---------------------------------------------------------------------- |
| `prevalence` | `prevalence(I)` | `prevalence(R, D)`               | `prevalence(Y1 + Y2)` — via `DerivedExpr`, `Instant`                   |
| `incidence`  | `incidence(T)`  | **`incidence(A, B)`** — proposed | not available, and should not be: no `Interval` expression form exists |

### It requires no new representation

Every layer beneath the surface already carries this:

- `Ir.CumulativeFlowSum of string list` (`ocaml/lib/ir/ir.ml:425`) — an
  arbitrary list of transition names — with serialization (`serde.ml:875,885`),
  validation (`validate.ml:164`), dimension checking (`dimcheck.ml:816,1084`),
  constant folding (`constant_fold.ml:131`) and lint (`lint.ml:140`).
- `StreamProjection::FlowSum(Vec<usize>)` on the Rust side, classified
  `Interval`, with the accumulate-and-reset semantics already implemented.
- `expander.ml` already emits `CumulativeFlowSum` on two live paths: a bare
  `incidence(T)` over a stratified family (~8099), and the explicit
  `sum(a in dim, incidence(tr[a]))` form (~8304).

So the change is one lowering case. The IR does not move, `ir/VERSION` does not
move, and no golden moves.

### Scope: addition of named flows, and nothing else

`CumulativeFlowSum` is a list of names with no coefficients, so that is exactly
the expressible set:

- `incidence(a + b + c)` — accepted.
- `incidence(a, b, c)` — accepted, identical meaning.
- `incidence(2 * a)`, `incidence(a - b)` — rejected with a located error stating
  that flows combine by addition only, and that a reporting fraction belongs in
  the likelihood (`mean = rho * projected`), where it is already conventional
  and where it stays dimensionless.

Rejecting rather than silently supporting a wider grammar keeps the surface
honest about what the IR can hold. If coefficients are ever wanted, that is an
IR change and a separate proposal.

Indexed cells compose as they do today: `incidence(tr[a], other[a])` is the sum
of two named cells; the existing cross-strata aggregation gate (E280) is
untouched, since it fires on a _bare stratified family_, which is a different
question from a sum of distinct flows.

### What the modeller writes afterwards

```camdl
compartments { S, E, I, C, G, R }          # P and M are gone

die_community : I --> D @ gamma * q_comm * I
die_facility  : C --> D @ f_cfr * delta * C

projected = incidence(die_community, die_facility)
```

One compartment, one transition and one magic constant removed per model file,
the compartment diagram back to describing epidemiology, and the observation
saying what it observes.

## Alternatives considered

**Make `incidence()` a first-class expression term.** What the downstream
originally asked for, and the most general. Rejected: it admits mixed temporal
kinds (`incidence(a) + prevalence(I)`), which the projection classification
cannot represent and which has no defensible semantics. The narrower change gets
the modelling benefit without the incoherence.

**Introduce a named "flow union" construct.** A new IR node and new syntax for
something `CumulativeFlowSum` already represents and `prevalence` already
expresses conventionally. More surface for no additional capability.

**Do nothing; document the junction pattern.** Rejected on the four costs above,
and because the junction is currently what our own diagnostic _suggests_ —
gh#669's hint recommends it. Recommending a construction that distorts the
model, hides a constant from provenance and adds an unchosen delay is not a
resolution.

## Verification

- Red first: `incidence(a, b)` must lower to `CumulativeFlowSum ["a"; "b"]`,
  asserted against emitted IR rather than against a compile-success.
- A likelihood equivalence test: a junction model and the direct-sum model must
  produce the _same_ log-likelihood at fixed parameters. This is the assertion
  that matters — it is what proves the new form means what the workaround meant.
- Negative controls that must not move: single `incidence(T)`;
  `incidence(T[i])`; a bare `incidence(T)` over a stratified family still
  lowering to `CumulativeFlowSum`; `sum(a in dim, incidence(tr[a]))`;
  `prevalence(R, D)`.
- Rejection tests for `incidence(a + b)`, `incidence(2 * a)`,
  `incidence(a - b)`.
- No golden may move.
