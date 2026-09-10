# Proportion of flows over a window as a projection

- **Status:** Proposal
- **Date:** 2026-09-09
- **Motivation:** A stream that measures "of the events in this reporting
  window, what fraction were of kind A" — community versus facility deaths,
  confirmed versus suspected cases, positives versus tests, deaths versus cases
  — is a ratio of two flows accumulated over the same window. camdl cannot
  express it. The only spelling that compiles is a ratio of _rates_ read at an
  instant, which is a different estimand and fits without complaint. This
  proposal adds the window ratio as a projection and makes the instant spelling
  loud.

## The gap

An observation stream in camdl has a `projected` expression — the model's
version of the quantity the data column measures — and a likelihood that scores
the column against it. Two families of projection exist. An _incidence_
projection (`incidence(tr)`) accumulates the events on a transition over the
period a row covers; a _prevalence_ or derived projection reads the state at one
instant. The ratio of two accumulated flows is neither today.

```
$ camdlc e341.camdl        # projected = incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))
error[E341]: observation 'comm_frac': `incidence(...)` here is not a sum of flows. A projection
  may add incidence terms (`incidence(a) + incidence(b)`) or sum a family
  (`sum(p in patch, incidence(tr[p]))`), but it may not weight them, subtract them, or mix
  them with a state read
error[E348]: observation 'comm_frac': `covers` states the period each row covers, but this
  stream's `projected` reads state at an instant rather than accumulating a flow over an interval
exit=1
```

(The E348 is a misdiagnosis that follows the E341 — see "Found while reading".)
The modeller who then writes the proportion as a ratio of the two transitions'
_rates_ — `mu_c * I / (mu_c * I + mu_f * H)` — gets a projection that compiles,
is classified as an instant reading, refuses `covers`, and is scored against the
state at a single time. It is a different quantity from the window fraction the
data measures; the two agree only when both flows are constant across the
window. Measured below: 7–27 % relative error across a single simulated
epidemic.

This is a correctness hazard, not an ergonomic one, and it is not one model's
problem — proportion confirmed, test positivity and case fatality within a
window all have this shape.

## What exists

### E341: what a projection may do with `incidence`

The lowering site is the `ProjDerived e` arm of observation expansion in
`ocaml/lib/compiler/expander.ml`:

```
$ rg -n 'code:"E341"' ocaml/lib/compiler/expander.ml
8547:      Diagnostics.error ctx.diags ~code:"E341" ~loc:od_loc
```

The arm (`expander.ml:8817-8824`) tries `explicit_incidence_sum e` — a syntactic
walk that accepts `incidence(<transition or cell>)`, `+` between such terms, and
`sum(v in dim [where …], …)` over them, and returns the concatenated list of
concrete transition names. `Some [single]` lowers to `CumulativeFlow`,
`Some many` to `CumulativeFlowSum` after the E342 disjointness check, `Some []`
(a `where` that prunes every level) to a literal `DerivedExpr (Const 0.0)`.
Anything else that mentions `incidence` is E341 and lowers to the same
`Const 0.0` sentinel; anything that does not mention `incidence` becomes
`DerivedExpr (resolve_expr …)`. So the admitted grammar over flows is exactly: a
unit-weighted union. Division is refused, and the E341 hint offers no route to
the window fraction — it points to a reporting coefficient in the likelihood,
which is the answer to a different question.

### How a sum of flows is accumulated and read

Both places that produce a number for an incidence row compute the same
quantity: the flow over the row's window `[start, stop)`.

_Scoring._ `MultiStreamObsModel`
(`rust/crates/sim/src/inference/multi_stream_obs.rs`) gives every interval
stream one accumulator bin (`IntervalSlot`, one per `StreamProjection::FlowSum`,
built at `:1469-1484`). Once per union step, after substeps and before scoring,
`fold_into_acc` adds the step's per-transition flows over the slot's
`flow_indices` into `acc[k]` (`:1547`); `reset_due_acc` zeroes the bin on the
bin's own reset schedule — the window's start (`:1588`). A stream is scored by
reading its bin directly (`project_stream_from_acc`, `:1618-1632`:
`Some(k) => acc[k] as f64`). So at a stop boundary `acc[k]` is the flow over
`[start, stop)`.

_Emission._ `project_coverages` (`rust/crates/cli/src/main.rs:3411`) walks a
recorded trajectory once, building the running cumulative flow at each snapshot,
and reads each row as the difference at its two boundaries:

```
$ sed -n 3456,3457p rust/crates/cli/src/main.rs
        rows.iter().map(|&(label, cov)| match cov {
            Coverage::Interval { start, stop } => Ok(cum_at(stop)? - cum_at(start)?),
```

Every emitter goes through it — `simulate --obs` and `--obs-dir`
(`main.rs:2925`), the wide `[synthetic]` file (`fit/synthetic.rs:223`),
`fit predict` (`fit/predict.rs:1130`), obs-sourced quantities (`main.rs:2263`)
and batch (`batch.rs:1925,1954`) — so emission and scoring agree on what an
incidence row is by construction. That seam is the one the ratio reuses.

### What a `DerivedExpr` projection evaluates to at fit time

`Projection::temporal_kind` (`rust/crates/ir/src/observation.rs:47-59`) is total
over the five variants: `CumulativeFlow` and `CumulativeFlowSum` are `Interval`;
`CurrentPop`, `CurrentPopSum` and `DerivedExpr` are `Instant`. The runtime
`StreamProjection` mirrors it (`multi_stream_obs.rs:203-208`).

An instant stream owns no bin. Its rows are `StreamTimes::Instants(labels)`
(`:1224-1225`), so the observation time on the union axis _is_ the row's time
label, and scoring evaluates the expression against the integer compartment
state at that boundary (`score_streams`, `:1763-1800`:
`t = self.obs_times[obs_idx]`, `project(si, t)` →
`eval_stream_projection(Expr, …, counts, …)`). On the emission side the same
evaluator is called on the snapshot at the label (`main.rs:3520-3527`:
`snap_at(traj, obs_t)`). So a ratio of rates written as `projected` is read _at
whatever time the row's `time` column says_, with no window at all; the data's
window is not representable on the stream.

Verified against the emitted IR for the illustration model below:

```
$ camdlc illus.camdl | jq -c '.model.observations[] | {name, projection: (.projection | to_entries[0].key), covers}'
{"name":"deaths_comm","projection":"cumulative_flow","covers":{"kind":"until","offset":0.0,"span":7.0}}
{"name":"deaths_fac","projection":"cumulative_flow","covers":{"kind":"until","offset":0.0,"span":7.0}}
{"name":"comm_frac_instant","projection":"derived_expr","covers":null}
```

### How far the two quantities diverge

A six-compartment model, deterministic (`--backend ode`), daily output. Deaths
occur from the infectious compartment in the community and from the hospitalised
compartment in a facility, with a hospitalisation transition so facility deaths
lag community deaths:

```camdl
hospitalise : I --> H  @ eta * I
die_comm    : I --> Dc @ mu_c * I
die_fac     : H --> Df @ mu_f * H
```

Three streams: `incidence(die_comm)`, `incidence(die_fac)` (weekly,
`closing_at(time, 7 'days)`), and the instant ratio
`mu_c * I / (mu_c * I + mu_f * H)` emitted through
`beta(mean = projected, concentration = 1e9)` so the draw is the projected value
to four decimals.

Per seven-day window, from the run's own `flow_die_comm`/`flow_die_fac` columns
(window = `Σ comm / (Σ comm + Σ fac)` over the window; instant =
`mu_c I / (mu_c I + mu_f H)` at the window's close):

```
 t   Σ comm    Σ fac    window   instant@close   instant − window
 7     4.31     3.02    0.5877     0.5455        −0.042   (−7 %)
14    27.80    25.08    0.5257     0.5145        −0.011   (−2 %)
21   117.65   129.20    0.4766     0.4387        −0.038   (−8 %)
28   184.62   320.61    0.3654     0.2953        −0.070  (−19 %)
35   108.23   348.28    0.2371     0.1808        −0.056  (−24 %)
42    39.23   225.28    0.1483     0.1156        −0.033  (−22 %)
49    12.24   113.07    0.0976     0.0783        −0.019  (−20 %)
56     3.67    50.00    0.0683     0.0546        −0.014  (−20 %)
63     1.09    20.60    0.0502     0.0426        −0.008  (−15 %)
70     0.32     8.13    0.0382     0.0278        −0.010  (−27 %)
```

camdl's own emitted `comm_frac_instant` column is
`0.545463, 0.514512,
0.438688, 0.295274, 0.180808, …` at
`t = 7, 14, 21, 28, 35, …` — the instant-at-close column above, confirming that
the derived expression is read at the label. The instant ratio sits below the
window ratio throughout, because within each week `H` is still rising relative
to `I`: the state at the close is not representative of the week. The error is
largest where the two flows move fastest relative to each other, which is the
peak — the part of the series that carries most of the information about the
fraction. A modeller fitting `q_comm` (the community share) against this stream
with the instant spelling would be pulled roughly a fifth low, with nothing in
the fit output to say so.

At `t = 77` and `84` the emitted instant value is `0` while the flows give
`0.030` and `0.024`. That is a second defect, not this one: on the ODE backend
the instant read is taken off the integer-rounded state (`I` prints as `0`
there), which is gh#693's rounding on the emission path. It is worth noting
because a `FlowRatio` reads real-valued bins on that backend and does not
inherit it.

## The record

Searched issues by mechanism (`incidence projection`, `E341`, `proportion`,
`covers window`, `ratio`, `positivity`, `denominator`, `WeightedFlowSum`) and
`docs/dev/proposals/` for `E341`, `WeightedFlowSum`, `interval-valued`,
`ratio of flows`, `incidence(...) /`.

- **`2026-07-31-aggregation-semantics.md`, Decision 8–9 and §7 Increment B** is
  the governing prior decision. `incidence` becomes composable by adding a
  `Projection` variant, never an `Expr` constructor, because a flow read inside
  `Expr` would make `temporal_kind` stop being total and would let a flow be
  read from a rate expression (B1). Increment B's variant is
  `WeightedFlowSum(Vec<(Expr, String)>)` — a weighted _sum_; B3 requires weights
  constant over the window; **B4 defers `incidence` under a nonlinear function
  and `incidence` mixed with state** "with named diagnostics, not silently
  rejected". B1a (unit-weighted addition, `CumulativeFlowSum`) shipped as PR#684
  and is what E341 fences today. This proposal is the one slice of B4 that has
  an unambiguous estimand and a closed-form gradient: division of one
  unit-weighted flow sum by another. It follows Decision 9 (a variant, not an
  `Expr` node) and B2's accumulator generalisation (a stream owns more than one
  bin), which it lands first so Increment B inherits it.
- **`2026-08-20-incidence-over-a-flow-expression.md`** (superseded by the above)
  records the measured cost of the longhand-rates spelling on a death stream —
  5.5 log units on 232 observations, "indistinguishable from a refactor" — the
  same silent-wrong class, for a count rather than a fraction.
- **gh#182** (open): make `incidence()`/`prevalence()` composable expression
  operators. The addition half is done (B1a); this proposal closes the ratio
  half without making `incidence` an expression operator, for B1's reason.
- **gh#332** (open): reporting-delay convolution over an incidence history. Also
  an interval-valued projection over flows, also a `Projection` variant in the
  sketch; orthogonal to this one (a delay kernel, not a quotient). The
  accumulator generalisation here is the substrate a convolution would want too.
- **gh#812** (closed): `binomial`/`beta_binomial` accept `n = 0` and score
  exactly `0`. Load-bearing for the zero-denominator rule below.
- **gh#829** (open): `simulate --obs` writes `0` for every row of a stream whose
  likelihood reads a data column. It bites the ratio-plus-binomial pairing
  directly; see "Emission".
- **gh#693** (open): the ODE value path scores instant streams off the
  integer-rounded state. Corroborated on the emission path above.
- **`quantities {}`** (`2026-06-25-generated-quantities.md`,
  `2026-08-17-value-at-quantity-reducer.md`): reductions over a trajectory, not
  observation projections; a proportion of flows there is a derived quantity
  over two series and is not scored. No overlap.
- **`2026-08-28-observation-lag.md`**: `lag(...)` on a flow expression; relies
  on the same "accepted shapes over flows" fence (E341) and would compose with a
  ratio by lagging each side. Not addressed here.

No issue or proposal proposes the window ratio itself.

## Design

### The type

A projection variant holding the two sides as unit-weighted flow lists — the
same object `CumulativeFlowSum` carries — read as their quotient at the row's
close.

Rust (`rust/crates/ir/src/observation.rs`):

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    CumulativeFlow(String),
    CurrentPop(String),
    CurrentPopSum(Vec<String>),
    DerivedExpr(Expr),
    CumulativeFlowSum(Vec<String>),
    /// The ratio of two unit-weighted flow sums accumulated over the same row
    /// window: `Σ numerator / Σ denominator`, read when the window closes.
    /// Interval-valued; dimensionless. `NaN` when the denominator flow is 0.
    FlowRatio { numerator: Vec<String>, denominator: Vec<String> },
}
```

with `temporal_kind` gaining `Projection::FlowRatio { .. } => Interval`. On the
wire:
`{"flow_ratio": {"numerator": ["die_comm"], "denominator":
["die_comm", "die_fac"]}}`;
the `ir/schema.json` `projection` `oneOf` gains that entry, both arrays
`minItems: 1`.

OCaml (`ocaml/lib/ir/ir.ml`), appended last as the comment there requires:

```ocaml
type projection =
  | CumulativeFlow    of string
  | CurrentPop        of string
  | CurrentPopSum     of string list
  | DerivedExpr       of expr
  | CumulativeFlowSum of string list
  | FlowRatio         of { numerator : string list; denominator : string list }
```

Run-id hash (`rust/crates/runid/src/ir_hash.rs:525-556`): variants are tagged by
a permanent positional index; `FlowRatio` takes the next unused one (5 at the
time of writing; if `WeightedFlowSum` lands first it takes 6 — read the file, do
not pin), writing numerator then denominator.

Why a dedicated variant and not a general interval-valued expression over flow
bins: Increment B's B1 argument, unchanged. A `FlowExpr(expr-over-bins)` would
need its own autodiff pass for `∂proj/∂bin`, would admit `Cond` and non-smooth
forms whose gradient the ODE path cannot carry, and would invite exactly the
mixed forms B4 defers. The ratio has one gradient (the quotient rule) and one
undefined point (a zero denominator), both stated below. If a second nonlinear
shape ever earns its place, it is a second variant with its own rule, not a
widening of this one.

Why the sides are unit-weighted lists and not weighted terms: a differential
ascertainment inside the ratio (`rho_c·inc(c) / (rho_c·inc(c) +
rho_f·inc(f))`)
is a real model, and it is Increment B's `WeightedFlowSum` applied to each side.
It is not carved here: the sides are the object `CumulativeFlowSum` already is,
so the ratio lands on the existing lowering and validation, and when weighted
sums land the sides generalise in the same bump. The unit-weighted ratio is what
the four motivating streams need.

### Surface syntax

```camdl
comm_frac {
  columns     { time : time, comm_deaths : count, n_deaths : count }
  covers      = closing_at(time, 7 'days)
  projected   = incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))
  comm_deaths ~ binomial(n = n_deaths, p = projected)
}
```

The admitted grammar over flows becomes: a unit-weighted flow sum (as today), or
**one** such sum divided by another. Each side is anything
`explicit_incidence_sum`'s walker accepts — a single `incidence(tr)`, an indexed
cell, `+` between terms, `sum(v in dim [where …], incidence(tr[v]))` — so a
stratified model writes
`sum(p in patch, incidence(die_comm[p])) / (sum(p in patch, incidence(die_comm[p])) + sum(p in patch, incidence(die_fac[p])))`.

The operator is `/`, not a named form (`proportion(num, den)`), for the reason
`+` already is: the temporal kind of a projection is decided by its leaves, not
by its operator. All-flow leaves are interval-valued; all-state leaves are
instant; mixing is E341. Division of two accumulated counts is division — the
same arithmetic the modeller writes for a prevalence proportion
`(I_m + I_s) / (S + I_m + I_s + R)` — so a second name would be teaching
vocabulary for the same operation. `.claude/rules/dsl-surface.md` asks for a
named function where an operator's _semantics_ differ; here they do not.

What stays refused, each by name in the E341 message rather than falling
through: a weight on either side (`rho * incidence(a) / …`, `incidence(a) / 7` —
B1's `WeightedFlowSum`), a state read on either side (`incidence(a) / N` — B4),
subtraction inside a side, a second division
(`incidence(a) / incidence(b) / incidence(c)`), a constant or state numerator
(`1 / incidence(a)`), and a side that does not mention `incidence` at all. A
flow named twice _within_ a side is E342 as today; the same flow on both sides
is the expected shape (`a / (a + b)`) and is not a collision. There is no
requirement that the numerator be a subset of the denominator: case fatality
within a window (`incidence(die) / incidence(onset)`) is a ratio of different
transitions and may exceed 1 for a particle whose deaths outrun its cases — the
binomial then scores that particle `−∞` when `k ≤ n` in the data, which is the
truthful answer.

### Lowering (OCaml)

In the `ProjDerived e` arm, before `explicit_incidence_sum`:

```ocaml
(* `num / den` where both sides are unit-weighted flow sums — the window
   proportion. Each side goes through the same walker a projection's
   `+`/`sum` already does, so indexing, `where` pruning, the stream binder and
   E280 behave identically inside a ratio. *)
let flow_ratio = function
  | EBinOp (Div, l, r) ->
    (match go l env, go r env with
     | Some ls, Some rs -> Some (ls, rs)
     | _ -> None)
  | _ -> None
```

(`go` is `explicit_incidence_sum`'s inner walker, which already accepts a bare
`incidence(tr)` — only the `top` dispatch restricts to `ESum | Add`.) On
`Some (num, den)`: an empty side is a new error (E351: "the
numerator/denominator of the ratio names no flow after `where` pruning" — a
`Some []` here is a literal `NaN` projection and must not lower silently the way
an empty sum lowers to `Const 0.0`); otherwise `check_flows_disjoint` runs on
each side and the result is `Ir.FlowRatio { numerator; denominator }`. On `None`
for a `Div` where either side mentions `incidence`, E341 fires with the message
rewritten to name the two admitted shapes and the refused ones. A `Div`
mentioning no `incidence` continues to `DerivedExpr` as today, and then meets
the rate-ratio check under "Migration".

The catalog row for E341 changes in the same commit; E351 gets a row.

If Increment B's B5 replaces the syntactic walker with a typed
`parse_flow_expr`, the ratio's two sides are two `flow_expr`s and nothing here
changes shape. This proposal takes no position on B5.

### Runtime: a stream owns a range of bins

`multi_stream_obs.rs`:

```rust
pub enum StreamProjection {
    FlowSum(Vec<usize>),
    /// Two flow bins over the same window; the value is their quotient.
    FlowRatio { numerator: Vec<usize>, denominator: Vec<usize> },
    IntCompSum(Vec<usize>),
    Expr(ResolvedExpr),
}
```

`temporal_kind`: `FlowSum | FlowRatio => Interval`, so
`resets_after_observation` and `StreamTimes::Intervals` follow without a second
decision.

The bin layout generalises from "one slot per interval stream" to "a contiguous
range of slots per interval stream":

```rust
/// Where a stream's bins live in the dense `acc` vector. `None` for an
/// instant stream.
struct BinRange { first: usize, len: usize }
stream_bins: Vec<Option<BinRange>>,   // replaces stream_to_slot: Vec<Option<usize>>
```

`IntervalSlot` is unchanged — one bin, its `flow_indices`, its owning stream's
`reset_at_union` — so `fold_into_acc`, `fold_into_acc_real_blocks`,
`reset_due_acc` and the block-strided siblings are untouched: a `FlowRatio`
stream registers two consecutive slots (numerator first) with the same reset
schedule, and both fold and reset with the rest. `n_interval_streams()` is
renamed `n_acc_bins()` because the two counts diverge as soon as one stream owns
two bins; its 40-odd call sites (`types.rs` sizes `ParticleState.acc` with it;
`pgas.rs`, `particle_filter.rs`, `if2.rs`, `correlated_pf.rs`, `pgas_init.rs`,
`pgas_grad.rs`, `ode_loglik.rs`, `ode_grad.rs`, `traits.rs`,
`chain_binomial_process.rs`, and three CLI files) are a mechanical rename.
Increment B's B2 asks for exactly this generalisation; it lands here.

Scoring reads the bins:

```rust
fn project_stream_from_acc(&self, si, acc: &[u64], counts, params, t) -> f64 {
    match (&self.streams[si].projection, self.stream_bins[si]) {
        (StreamProjection::FlowSum(_), Some(b)) => acc[b.first] as f64,
        (StreamProjection::FlowRatio { .. }, Some(b)) =>
            flow_ratio(acc[b.first] as f64, acc[b.first + 1] as f64),
        (_, None) => eval_stream_projection(/* as today */),
        _ => unreachable!("an interval projection owns a bin range by construction"),
    }
}

/// `num / den` over one window. A zero denominator is not a number: the
/// stream measures a fraction of events that did not occur.
fn flow_ratio(num: f64, den: f64) -> f64 { if den == 0.0 { f64::NAN } else { num / den } }
```

`project_stream_from_acc_real` (the ODE path) is the same on `f64` bins.
`eval_stream_projection` — the per-transition-counter evaluator the CLI and
tests use — gains the mirror arm (sum each side over `flows`, divide) so the
three call shapes stay uniform. `incidence_streams()`, which names the
`inc_<stream>` output columns of a posterior trajectory, reports a ratio
stream's two bins as `<stream>_numerator` and `<stream>_denominator`: the two
counts are what a reader wants to see, and the ratio is one division away.
`util.rs:2352`'s kind label for `check` renders
`incidence(a + b) / incidence(c + d)`.

### `covers`, temporal kind, dimension

The stream is `Interval`. `projection_accumulates` (`expander.ml:7984`) returns
`true` for `FlowRatio`, so `covers = …` or the window columns are _required_
(E350 without them) and E348 does not fire. The row's period is what the ratio
is taken over; the label is only a label. The `MultiStreamObsModel`
defence-in-depth check (`:715`) that a stream's `times` variant agrees with its
projection's kind holds without change.

A ratio of two counts is dimensionless. `dimcheck.ml` has two sites that assign
a projection's dimension (`obs_projection_dim`, `:811-817`, and the
`projected_dim` set at `:1084-1087`); both gain
`FlowRatio _ -> Known dimensionless`. That is what makes the pairing below
type-check: `binomial`'s `p`, `beta`'s `mean`, `bernoulli`'s `p` and the
`beta_binomial` mean/concentration sugar all `require_dimensionless`
(`dimcheck.ml:1142,1169,1180`), and `poisson`'s `rate` / `neg_binomial`'s `mean`
`require_count`, so a ratio fed to a count family is E304 with the existing
message, and a count fed to a proportion family stays the "missing `/N`" E304.

### Gradient

On the ODE forward-sensitivity path the per-bin `∂bin/∂θ` is already folded
beside the value (`fold_into_acc_real_blocks`, `acc_sens[slot·d ..]`). The
gradient reader (`log_likelihood_grad_from_flows_and_counts`, `:2198-2212`)
gains a `FlowRatio` arm: with `a = acc[first]`, `b = acc[first+1]`, `p = a/b`
and `∂p/∂θ_k = (b·∂a_k − a·∂b_k) / b²` for `b > 0`; for `b = 0` the value is
`NaN` and the gradient contribution is `0`, the convention every likelihood
kernel in `obs_loglik.rs` already follows for its `−∞` floor.
`gradient_capability.rs:442` admits the variant: it is not a `DerivedExpr`, so
it carries no `projection_state_grad` and needs none — its θ-dependence is
through the bins, as `FlowSum`'s is. `autodiff.ml:841,891` treat it as
`CumulativeFlowSum` (no state gradient; `Projected` stays opaque in the
likelihood argument). On the particle paths (PGAS, pfilter, IF2, PMMH) a fixed
latent path fixes the flows, so `∂projected/∂θ = 0` exactly as for `FlowSum`;
nothing new.

### A zero denominator

When no event of the denominator kind occurred in the window, the fraction is
undefined. The projection is `NaN`, and `NaN` is already a defined input to
every likelihood family (gh#645's rule), so no new likelihood-layer behaviour is
introduced. What that means per pairing, verified in `obs_loglik.rs`:

- `binomial(n = <data>, p = projected)`: `binom_logpmf` checks `n == 0` _before_
  `p` (`:492`) — a row with no classified events scores exactly `0` whatever the
  model's fraction, which is the gh#812 contract; a row with `n > 0` meets
  `p.is_nan()` (`:497`) and scores `−∞`, and `explain_likelihood_neg_inf` names
  it (`ArgumentNaN { p }`, `obs_model.rs:287`). That is the right answer: the
  data recorded `n` classified events in a window where this particle produced
  none, so the particle is inconsistent with the row. It is not a hole — a hole
  is a statement about the data, and skipping here would let a particle with no
  deaths pass a row that has some.
- `beta_binomial(n = <data>, mean = projected, concentration = φ)`: same order
  (`beta_binomial_logpmf:540,547`).
- `beta(mean = projected, concentration = φ)`: `NaN` mean is `−∞`
  (`beta_logpdf:600`). A proportion with no denominator gives the model no way
  to say "no events", so `−∞` is the only honest score.
- `bernoulli(p = projected)`: the arm at `obs_model.rs:137-147` clamps `p`, and
  `NaN.clamp(0, 1)` is `NaN`; `NaN.max(LOG_PROB_FLOOR)` is the floor (`f64::max`
  returns the non-NaN operand), so the row scores `ln(1e-300) ≈ −691` — a finite
  penalty, not `−∞`, and the one family that disagrees with the other three. Add
  the `is_nan` guard the other kernels carry (gh#645's rule) so a `NaN`
  projection is `−∞` under every pairing, and assert it.

On the ODE backend flows are real and positive after seeding, so the case is
reached only on the stochastic backends with small counts, where killing the
particle is what the filter is for.

## The likelihood pairing

Two data shapes, both supported:

```camdl
comm_deaths ~ binomial(n = n_deaths, p = projected)                     # k of n
comm_deaths ~ beta_binomial(n = n_deaths, mean = projected, concentration = phi)
comm_frac   ~ beta(mean = projected, concentration = phi)               # a fraction, no n
```

The `n` of a binomial stays what `BinomialLikelihood` says it is
(`observation.rs:93-100`): θ-independent, data-supplied or constant, no
gradient, rounded to an integer; `collect_n_param_refs`
(`gradient_capability.rs:78`) refuses an estimated parameter reaching it and §1h
refuses `projected` inside it. The model's own denominator does **not** go in
`n` — `n = incidence(die_comm) + incidence(die_fac)` is not a likelihood
argument and stays refused. It enters through `p`, inside the ratio. The
likelihood is then the conditional `p(k | n, x, θ)`: given that `n` deaths were
classified, each was a community death with probability equal to the model's
community share of that window's deaths. `n` is conditioned on, not scored. If
the modeller also wants the death count itself to inform the fit, that is a
second stream — `deaths ~ neg_binomial(mean = rho * projected, r = k)` over
`incidence(die_comm) + incidence(die_fac)` — and the joint factorises as
`p(n | x)·p(k | n, x)`: two streams, two pieces of evidence, no double count.
The L404 lint (two streams reading one latent quantity) keys projections by
identity (`lint.ml:264-276`); `FlowRatio` gets its own key
(`KRatio of string list * string list`, each side canonicalised) so a ratio
stream and the count stream over its denominator are never reported as one
measurement.

## Emission

`project_coverages` gains a `FlowRatio` arm: `incidence_over(numerator)` and
`incidence_over(denominator)` — the same closure the two flow variants use,
which already refuses an `Instant` row and an unrecorded boundary — zipped
through `flow_ratio`. Because every emitter reaches rows through that one
function, `simulate --obs`, `--obs-dir`, the `[synthetic]` file, `fit
predict`
and obs-sourced quantities all produce the ratio the scoring path scores:
numerator and denominator are the existing incidence quantity on both paths, and
the only new arithmetic is one division in each.

What the sampler then does with it is per pairing. `fit predict` forwards the
_observed_ `n` at each row (`predict.rs:1138-1143`), so the predictive draws
`k_rep ~ binomial(n_obs, p̂)` and its mean companion reports `n_obs · p̂` — the
correct conditional-on-`n` predictive. `beta` draws the fraction directly.
`simulate --obs` has no data, passes `&[]`, and the binomial's `n` evaluates to
`0`, so the synthetic column is identically `0` — gh#829, reported from exactly
this stream shape (a community-death fraction with a data-column denominator).
This proposal does not change that; it records, under "Decisions", the
resolution it recommends for a ratio stream: unlike `tested`, the denominator
here is a quantity the model generates, so the emitter can write the `n` column
from the stream's own denominator bin (rounded on the ODE backend) and draw
`k ~ binomial(n, ratio)` — a synthetic file that round-trips.

## The IR change is a human-loop change

A new `Projection` variant is an `ir/schema.json` change and an `ir/VERSION`
bump (0.39 → 0.40 at the time of writing), with the atomic OCaml + Rust + golden
update the `golden-update` skill describes. What it invalidates: every committed
golden moves by its `ir_version` line; every cached compiled IR re-keys; every
stored run whose identity includes the model IR (all of them) gets a fresh
run_id, so `camdl list` shows the old runs as a separate lineage and nothing is
reused across the bump. Existing variant indices are untouched, so a model that
does not use `FlowRatio` hashes identically _within_ a version. None of this is
an argument for a smaller design: the maintainer's rule is to flag the bump and
confirm the sequencing, then do the better representation. Two other re-keying
changes are pending (Increment B's `WeightedFlowSum`; gh#568's typed
generated-dimension name); whether this rides alone or shares a bump is a
decision below.

## Migration for the reporting model

Before — compiles, instant, no window:

```camdl
comm_frac {
  columns     { time : time, comm_deaths : count, n_deaths : count }
  projected   = mu_c * I / (mu_c * I + mu_f * H)
  comm_deaths ~ binomial(n = n_deaths, p = projected)
}
```

After — interval, the window declared:

```camdl
comm_frac {
  columns     { time : time, comm_deaths : count, n_deaths : count }
  covers      = ending_on(time, 7 'days)     # whichever form the file's labels follow
  projected   = incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))
  comm_deaths ~ binomial(n = n_deaths, p = projected)
}
```

The likelihood line does not move. The `covers` line is new and required, and it
is the one place the modeller has to know something about their file: which day
a weekly label names (`§12.1.1`).

**How to tell which quantity a stream has today.** Structurally: if a proportion
stream's `projected` compiled before this proposal, it is an instant read,
because no interval-valued division existed. The emitted IR says so —
`camdlc model.camdl | jq '.model.observations[] | {name, projection:
(.projection | keys[0]), covers}'`
shows `derived_expr` and `covers: null` for the instant form, `cumulative_flow*`
with a `covers` object for an interval one. Numerically, on one's own fit:
`simulate --draws posterior --fit <dir>
--backend ode --output-every 1 -o traj.tsv`
and compute, per declared window, `Σ flow_a / (Σ flow_a + Σ flow_b)` against the
rate expression evaluated at the row label (the awk in the illustration above is
the whole computation). The relative gap per window is the bias the instant
spelling carried into the likelihood; it is largest where the flows changed
fastest inside a window.

**Making the instant spelling loud.** The hazard is that the rate ratio is
silent. After this proposal a projection that is a `Div` whose numerator and
denominator terms are each structurally equal to a lowered transition's rate
expression is an error:

```
error[E352]: observation 'comm_frac': `projected` divides transition rates — `mu_c * I` is
  the rate of `die_comm` and `mu_f * H` the rate of `die_fac` — so it is read at the row's
  instant, not accumulated over its window. If the column is the fraction of the window's
  events, write the flows: `incidence(die_comm) / (incidence(die_comm) + incidence(die_fac))`
  (and declare `covers`). If you mean the instantaneous ratio of hazards, say so:
  `prevalence(mu_c * I / (mu_c * I + mu_f * H))`.
```

`prevalence(<expr>)` already lowers any state expression to `DerivedExpr`
(verified: `prev.camdl` with that spelling compiles with exit `0`, projection
`derived_expr`), so the explicit instant form exists and costs nothing. The
comparison is structural equality of `Ir.expr` between the projection's additive
terms and the lowered transitions' rates, made in the expander before constant
folding and LICM run (both are later passes, `compiler.ml:686-693,838`), so both
sides are un-hoisted plain expressions. It has no false positives — a match
means the term _is_ a transition's rate — and has false negatives when a rate is
written differently in the two places, which the diagnostic's own text cannot
close; that residue is documented in the spec paragraph. Blast radius in-repo:
two projections divide anything
(`ocaml/golden/surveillance_likelihoods.camdl:69,81`, `R / N` and `I / N`),
neither a transition rate, so no golden trips.

## Spec and documentation

- `docs/camdl-language-spec.md` §12.1: the projection grammar over flows — a
  unit-weighted sum, or one such sum divided by another; the rule that the
  temporal kind is decided by the leaves; the sentence that a ratio of rates is
  an instant reading and is E352 unless written `prevalence(…)`. §12.1.1: the
  list of accumulating shapes gains "a ratio of two such sums". §12.2: the
  count-argument paragraph gains the dimensionless pairing. §25.4: a `FlowRatio`
  lowering row. Hand-edit; no `dprint`.
- `docs/language-changes.md`: one entry. It _widens_
  (`incidence(a) /
  (incidence(a) + incidence(b))` now compiles) and it
  _tightens_ (a projection dividing transition rates is E352, migration to
  `incidence(…)/(…)` or `prevalence(…)`), so both halves are stated with old →
  new.
- `docs/dev/warning-catalog.md`: E341 row rewritten; E351, E352 rows.
- `docs/dsl-cheatsheet.md`, `docs/user-features.md`: the projection row; the
  `beta` bullet's "reach for it when the observation is a fraction with no
  reported denominator" gains the ratio as the projection to pair it with.
- `docs/agents.md`: the "proportion stream needs a non-zero denominator even on
  unscored rows" trap is about a data-column denominator and stays; add the
  window-fraction pattern beside it.

## Verification

Red before green, each asserting on the emitted IR or a number, never on exit
status.

- Lowering: the ratio compiles to `flow_ratio` with the expected two lists; the
  stratified `sum(…)/(sum(…)+sum(…))` form too; `covers` accepted, E350 without
  it, E348 never.
- Rejections, each naming what was written: a weighted side, a state side, a
  constant side, subtraction in a side, a second division, a flow twice in one
  side (E342), an empty side after `where` (E351), a rate ratio (E352), and
  `prevalence(<rate ratio>)` accepted.
- Scoring: a two-flow chain-binomial model; the projected value at each window
  equals `Σ num / Σ den` recomputed from `CAMDL_TRACE_STEPS=1` flows; the
  pfilter log-likelihood under `binomial(n, projected)` equals the hand-computed
  sum of `binom_logpmf`.
- Zero denominator: a window with no denominator events — `n = 0` row scores
  `0`, `n > 0` row scores `−∞` with `NegInfCause::ArgumentNaN { p }`.
- Emission equals scoring: `simulate --obs` on the ODE backend at θ* and the
  in-sim `project_stream` on the same trajectory agree to floating-point on
  every window.
- Gradient: the ODE-NUTS gradient of a ratio stream against central finite
  differences, the pattern of `ode_grad.rs`'s multi-index `FlowSum` test.
- L404 does not fire between a ratio stream and a count stream over its
  denominator.
- Goldens: `make update-golden`; a new golden fixture carrying the ratio with
  first-scenario parameters and a trajectory baseline; every existing golden
  differs only in `ir_version`. Full `make test` before the commit.

## Decisions

1. **Surface: `/` between two flow sums, not a named `proportion(num, den)`.**
   The temporal kind follows the leaves, as it already does for `+`; a name
   would be vocabulary for plain division and would still need the same fence on
   its arguments. Recommendation: `/`. Confidence: leaning-solid.
2. **Representation: a dedicated `FlowRatio` variant, not an interval-valued
   expression over flow bins.** Follows aggregation-semantics Decision 9 and B1;
   the ratio has one gradient and one undefined point. Recommendation: the
   variant. Confidence: solid.
3. **Sides are unit-weighted flow lists.** Weighted sides are Increment B's
   `WeightedFlowSum` applied per side and generalise in that bump.
   Recommendation: unit-weighted now. Confidence: solid.
4. **No numerator-subset-of-denominator rule.** Case fatality within a window
   needs the general ratio; a ratio above 1 scores `−∞` under a binomial with
   `k ≤ n`, which is correct. Recommendation: no rule. Confidence: leaning.
5. **Zero denominator → `NaN` projection, the existing `NaN` contract.** `n = 0`
   rows score `0`, `n > 0` rows `−∞` and are explained; no hole, no refusal, no
   new likelihood-layer rule. Recommendation: as stated. Confidence: solid.
6. **E352 (a projection dividing transition rates) is an error with
   `prevalence(…)` as the explicit instant spelling, not a warning.** The
   dsl-surface rule prefers a hard error with a hint; the escape costs one word
   and states the intent. The alternative — a warning — leaves the silent form
   silent for anyone who skims. Recommendation: error. Confidence: leaning.
7. **`simulate --obs` for a ratio stream with a data-column `n`: write the `n`
   column from the stream's denominator bin and draw
   `k ~ binomial(n,
   ratio)`.** The denominator is model-generated, unlike
   surveillance effort, so this is the one shape gh#829 can resolve by emitting
   rather than refusing. _Ruled 2026-09-09._ Emit `n` from the stream's
   denominator bin, and "emit what the model can generate, refuse what it
   cannot" is the general resolution of gh#829. Lands in this arc's emission
   commit, on the design-preserving emitter.
8. **Bin-range generalisation lands here.** `BinRange` and `n_acc_bins()`
   replace the one-slot-per-stream layout; Increment B's B2 inherits it.
   Recommendation: as stated. Confidence: solid.
9. **Sequencing of the `ir/VERSION` bump.** Alone, or folded with
   `WeightedFlowSum` and gh#568 into one invalidation. Neither of the other two
   is implemented; waiting on them holds a correctness fix for an ergonomic one.
   _Ruled 2026-09-09._ Bump alone, now; the next re-keying change pays its own.

## Found while reading

- **E348 fires after E341 on the same stream (pile-on).** `incidence_misuse`
  returns `DerivedExpr (Const 0.0)`; `lower_covers` (`expander.ml:7996`, called
  at `:9011`) then sees a non-accumulating projection and reports the `covers`
  line as the mistake. The expander already suppresses dependent checks after a
  primary error — `has_real_measurement` at `:8183`, "its E273 already fired, so
  we don't pile on". A flag set in `incidence_misuse` and read by `lower_covers`
  (skip E348 and E350 when the projection itself errored) is the same
  convention. Small, independent; worth its own issue and commit ahead of this
  proposal.
- **gh#693 has a second site.** `project_coverages`'s `DerivedExpr` arm
  evaluates on `snap.int_state.counts` (`main.rs:3522-3526`), so
  `simulate --obs` on the ODE backend emits an instant projection off the
  rounded state: at `t = 77` in the illustration `I` prints as `0` and the
  emitted ratio is `0` where the real-valued flows give `0.030`. Add to gh#693
  so the two paths are resolved together, as that issue asks.
- **The E341 hint has no route to the window fraction.** It offers a likelihood
  coefficient and the stratified forms; for a ratio the only compiling move it
  leaves is the longhand rate, which E203's hint (the multi-argument `incidence`
  case) explicitly warns is a different quantity, "evaluated at an instant
  rather than accumulated over the window". The two hints disagree about whether
  to mention the hazard. Resolved by the E341 rewrite above; noted because until
  then E341 steers toward the silent form.
- **gh#829 is reached by this stream shape today**, and the fix it wants differs
  by shape (Decision 7).
- **`eval_stream_projection`'s `t` is documented as unused** and threaded "for
  forward compatibility with time-dependent projections"
  (`multi_stream_obs.rs:318-319`). Still true after this proposal; nothing to
  do.
