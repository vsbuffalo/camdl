# A reporting lag on observation projections

Date: 2026-08-28 Status: Active Issue: gh#332

An administrative delay between an event happening in the model and the same
event appearing in a data file is not a movement of population. It changes only
_when a flow is counted_. camdl cannot express it, and the only workaround that
compiles distorts the model to say something it does not mean.

This document specifies a per-term delay on incidence projections: the surface,
the runtime mechanism on each backend, what a delay costs where the accumulator
already lives, and what may and may not be estimated from data.

## 1. The problem, as the modeller states it

An Ebola outbreak team measured the interval between symptom onset and
publication of a case record, and split it into two limbs. The first —
onset-to-sample — is already a dwell time in a compartment, because a person who
has not yet been sampled is in a different epidemiological state from one who
has. The second — sample-to-publication — is 1.5 days at the median with almost
no spread. Its mechanism is a clerk: the person is sampled and isolated at the
`confirm` transition, and the record is published a day and a half later. Nobody
moves.

Their national case series sums three administrative routes:

```camdl
cases_national:
  projected = incidence(confirm_fast) + incidence(confirm_slow)
            + incidence(die_community_fast) + incidence(die_community_slow)
```

`confirm_*` is a lab-confirmed live case; `die_community_*` is a post-mortem
swab taken in the community; a third route, `die_facility`, records a death
where the patient already was. These are three different paper trails, and there
is no reason to expect one delay to describe all of them. So the delay is not a
property of the _stream_ — it is a property of each _term_ in the stream's
projection.

That last sentence is the whole design constraint. Everything below follows from
it.

## 2. What exists today

Established by reading the compiler and runtime at `a7885c1d`, not from memory.

**A `lag` exists, but on forcings.** A forcing declaration accepts
`lag = 10 'days` alongside its own arguments, for every forcing kind
(`expander.ml:6841`, the `known = "lag" :: keys` list). It resolves through the
ordinary expression resolver so a unit-annotated literal is rescaled into the
model's time unit, and lowers to a field on the intermediate representation's
`time_function` record (`Ir.lag`, `expander.ml:7221`). The dimension checker
requires it to carry the time dimension `T` (`dimcheck.ml:967`), naming a
`: duration` parameter as the alternative to a literal. The runtime shifts the
lookup: `t_eff = ctx.t − eval_resolved(lag, ctx)` (`propensity.rs:233`).

**Why it does not transfer.** A forcing is a closed-form function of time — a
sinusoid, an interpolation table, a Fourier series — so evaluating it at `t − d`
is a change of argument and nothing else. gh#314 says so explicitly: the lag is
"Markovian-safe" because the object being shifted carries no state.
`incidence(...)` is the opposite. It is an accumulator the integrator maintains
over the reporting window, with reset semantics a state read does not have. The
mechanism that lags a forcing has no counterpart here.

**Projections are a separate type from expressions.** `Ir.projection`
(`ir.ml:411`) has five variants — `CumulativeFlow`, `CumulativeFlowSum`,
`CurrentPop`, `CurrentPopSum`, `DerivedExpr` — and `Projection::temporal_kind()`
(`ir/src/observation.rs:47`) is total over them: the first two are `Interval` (a
flow accumulated over the window) and the last three are `Instant` (state read
at the observation moment). Nothing in the expression language can produce an
`Interval` value.

**Unit-weighted flow unions already compile.** `incidence(a) + incidence(b)`
lowers to a single `CumulativeFlowSum` with the flow names concatenated
(`expander.ml:8286`, the `EBinOp (Add, l, r)` arm of `explicit_incidence_sum`).
A weighted term (`rho * incidence(a)`), a subtraction, and a flow mixed with an
instant state read are each refused by name with `E341`; a flow appearing twice
is `E342`. So the modelling team's four-term `projected` compiles today and
produces one `CumulativeFlowSum` over four transition names.

`docs/camdl-language-spec.md:2829-2836` still says `incidence(a) + incidence(b)`
does not sum and directs the reader to a junction transition. That paragraph is
stale and must be corrected whether or not this proposal lands.

**The accumulator has two levels.** Per particle (`inference/types.rs:299`):
`flow_accumulators`, a per-transition tally written each substep and
blanket-zeroed at every observation index; and `acc`, one bin per
interval-valued stream, folded from `flow_accumulators` once per interval
(`fold_into_acc`) and zeroed only for the streams scheduled at that index
(`reset_due_acc`). Reading `flow_accumulators` at scoring time is prohibited: it
is correct only when every stream shares a cadence, and
`sim/tests/per_stream_reset.rs:266` is a mutation guard asserting 20 where the
correct 30-day bin is 300.

**A weighted-flow variant is already designed and unlanded.**
`2026-07-31-aggregation-semantics.md` §7 specifies
`Projection::WeightedFlowSum(Vec<(Expr, String)>)` at hash index 5, with the
accumulator becoming per-reference (`IntervalSlot` gains an offset and length,
one bin per term). `AGGREGATION-ARC.md` item 4b tracks it. Its Increment B1a
shipped (PR#684); the rest has not.

**The integrator can already land on an arbitrary time.** Under
`StepPolicy::Exact`, `Schedule::next_boundary` (`schedule.rs:317`) is the
minimum over `(t_end, next_output, next_effect, next_obs)`, and the walk clips
to it. Under `StepPolicy::Snap`, observation times are rounded onto the `dt`
grid by `build_obs_at_substep` (`pgas.rs:612`), which hard-errors rather than
silently drop an observation when two round to the same substep.

**Warm-up already has a mechanism.** `condition_from` runs the dynamics without
scoring so the first scored bin is properly filled
(`2026-06-09-burnin-conditioning-window.md`, shipped at `588a40e`), and
`first_window_guard` (`cli/src/util.rs`) is gated on temporal kind — a hard
error for interval streams, a soft warning for instant ones, suppressed when
`condition_from` is set.

**Backends.** For fitting, the supported pairs are chain-binomial × {IF2, PGAS,
PMMH, particle filter} and ODE × {NLopt Sbplx, and the gradient methods}
(`fit/methods.rs:68`). Gillespie is not an inference backend
(`fit/methods.rs:779`); it appears only on the forward simulation path.

## 3. The mechanism

### 3.1 Semantics

`lag(<flow union>, d)` shifts the accumulation window. For observation times
`t_{j-1} < t_j` on a stream's own schedule, the projected value at `t_j` is

```
P_j = N(t_j − d) − N(t_{j−1} − d)
```

where `N` is the cumulative count of the union's flows. Every event is reported
exactly `d` later — a point (Dirac) delay. The distributed case (gh#332's
`Convolved`) is `Σ_k w_k · [N(t_j − d_k) − N(t_{j−1} − d_k)]`, which is the same
object with several atoms; §6 uses that to show the point case does not
foreclose it.

### 3.2 The buffer is small, because the boundary moves instead of the history

The obvious reading of "evaluate incidence at `t − d`" is that the runtime must
retain flows over a window that has already passed. That framing overstates the
cost. The retained object is not a flow history — it is a small number of
already-closed bins.

Today a bin closes at `t_j` and is scored immediately. With a lag it closes at
`t_j − d` and must survive until `t_j`. So:

- add `{ t_j − d }` to the set of times the integrator lands on, the same axis
  as `obs_times` and `effect_times`;
- at a lag boundary, fold the term's flows into a bin and push it onto a
  per-term queue;
- at the observation, score the oldest queued bin instead of the running one.

The queue depth is the number of lag boundaries open at once, `⌈d / Δ_min⌉ + 1`,
where `Δ_min` is the stream's smallest inter-observation gap. For the requesting
team — `d = 1.5` days, weekly reporting — the depth is one. One extra `u64` per
lagged term.

This must extend the existing boundary authority, not add a parallel one.
`.claude/rules/rust-conventions.md` records that "where does the integrator stop
next" already has four incompatible answers (gh#233); a `next_lag_stop` accessor
would be the fifth. The lag times join `Schedule`'s boundary set and
`next_boundary`'s minimum.

### 3.3 Cost per backend

**Chain-binomial (IF2, PMMH, particle filter, and the conditional sequential
Monte Carlo sweep inside PGAS).** The queue lives in `ParticleState`, which
today is `{ counts, flow_accumulators, acc }` and is cloned on every resampling
step. The addition is one `Vec<u64>` of total length `Σ_terms depth_term`. At
4000 particles with one lagged term of depth one, that is 32 KB copied per
resample, against the `counts` and `flow_accumulators` vectors already copied.
This is a depth extension of the field the multi-cadence work already added, not
a new mechanism.

**ODE.** The per-transition flow is already augmented state integrated by the
same Runge–Kutta step as the compartments: `d_flow[i] = propensity_i`
(`ode.rs:70-76`, `:121`), so `c_r(t)` is the exact cumulative integral. The
lagged bin is `c_r(t_j − d) − c_r(t_{j−1} − d)`, exact at the boundaries the
integrator lands on. The forward-sensitivity blocks ride the same fold
(`fold_into_acc_real_blocks`), so the sensitivity bin closes on the same
schedule as the value bin it differentiates — the invariant that machinery
exists to protect.

**Gillespie.** Not an inference backend. On the forward path, synthetic
observations are emitted from a materialized trajectory
(`main.rs::project_all_obs_times`), where a lag is a re-binning of data already
in hand and needs no per-particle state.

### 3.4 When `d` is not a multiple of the step

The two alignment policies give different answers, and only one of them is
acceptable silently.

Under `Exact`, `d` needs no special handling: the lag boundary is one more term
in `next_boundary`'s minimum, the walk clips to land on `t_j − d`, and the bin
closes there. No interpolation, no fractional share of a substep's flow.

Under `Snap`, a lag boundary rounds onto the `dt` grid like everything else, so
the realized delay is `round(d/dt)·dt`. At `dt = 1` day, the team's 1.5-day
delay becomes 1 or 2 days — a third of the quantity being modelled, silently.

**Decision D6 (recommended): refuse.** Under `Snap`, a lag that is not a
multiple of `dt` is a hard error naming both the declared and the realized value
and pointing at `--obs-alignment exact`. Rounding a delay is exactly the kind of
"works but wrong" default `.claude/rules/dsl-surface.md` forbids, and
`build_obs_at_substep` already sets the precedent by erroring rather than
dropping a colliding observation.

Fractional-share interpolation within a substep is rejected outright. On a
stochastic backend the flow in a substep is an integer draw, and apportioning
part of it to a window is a quantity the model cannot produce.

### 3.5 The start of the series

A lagged stream's first scored bin covers `(first_obs − Δ − d, first_obs − d]`.
If that reaches before `t_start`, the bin is short by however much of the window
falls outside the simulation, and the first observation is scored against a
systematically undercounted projection — the same defect, one window further
back, that `condition_from` exists to fix.

**Decision (recommended): extend `first_window_guard`.** Its interval arm adds
the stream's maximum lag to the required lead-in, keeping the existing hard
error and `condition_from` suppression. No new mechanism and no new knob: a
lagged stream simply needs `t_start ≤ first_obs − Δ − d`.

### 3.6 The interaction that will bite

`reset_due_acc` is keyed on the _stream_. Under a per-term design, terms in one
stream can carry different lags, so their bins close at different times while
all of them reset when the stream is scored.

Close and reset are two different events on the same bin, and conflating them is
the silent-wrong here. Closing on the stream's schedule rather than the term's
would apply the lag to the reset and not to the count, producing a projection
that is neither lagged nor unlagged and that looks entirely plausible. The rule:
**the close is term-keyed; the reset stays stream-keyed** — a term appearing
twice contributes twice and resets once, which is the aggregation proposal's B2
rule unchanged.

## 4. Options, with their measured costs

### Option A — a block key on the observations block

`lag = 1.5 'days` alongside `columns` and `projected`.

The observations block accepts exactly five item kinds today (`build_obs_decl`,
`parser.mly:117`): `columns`, `emit_schedule`, `projected`, the `~` measurement,
and the legacy `likelihood`. A sixth means a grammar rule, an abstract-syntax
field, an `observation_model` field, and a version bump.

It fails on the requirement in §1: a block key is per-stream, and the stream in
question sums three administrative routes. The requesting team withdrew it
themselves.

### Option B — a variant on the expression type

Blast radius, counted rather than estimated.

_OCaml_ — 13 source files enumerate all 18 `expr` variants exhaustively:
`ir/{dimcheck, autodiff, constant_fold, licm, lint, lineage, expr_analysis,
init_order, serde, validate}.ml`
and `compiler/{expander, inspect,
pp_expr}.ml` — plus `ir/ir.ml` where the type
lives. **14 files.**

_Rust_ — 12 source files enumerate all 18 `Expr` variants exhaustively:
`ir/src/{expr, validate, init_order}.rs`, `runid/src/ir_hash.rs`,
`sim/src/{resolved_expr, propensity, compiled_model}.rs`,
`sim/src/inference/{multi_stream_obs, hierarchical, gradient_capability}.rs`,
`cli/src/obs_anchor.rs`, `cli/src/fit/coeff_guard.rs`. `ir/src/expr.rs` alone
needs three edits — the enum, the `Serialize` wrapper struct, and an arm in the
hand-written `Deserialize`, whose own maintenance comment says so. **12 files.**

_Two mirror representations_ — `ResolvedExpr` (19 variants; pre-resolved indices
for the hot path) is enumerated exhaustively in `sim/src/resolved_expr.rs` and
`sim/src/flat_eval.rs`, and the flat-bytecode `Op` in `flat_eval.rs` is a third
encoding of the same language.

_Schema and goldens_ — `ir/schema.json`, `ir/VERSION` (currently 0.37), the
regenerable goldens via `make update-golden`, and a hand-edit of the
`ir_version` line in the 17 frozen `ir/golden/*.ir.json` — a set
`make update-golden` does not cover (gh#384). The last comparable bump touched
125 files.

**Total: 28 files carrying exhaustive matches across 4 enums, plus the schema,
the version, and the goldens.**

The count is not the decisive objection. That is semantic, and it is already on
record. `2026-07-31-aggregation-semantics.md` B1 refuses a flow-read expression
node because it would stop `temporal_kind()` being a total function of the
variant, force `resets_after_observation` to become conditional, require a
differentiation pass with respect to flows and a corresponding gradient field on
the intermediate representation, and — decisively — make a flow read
representable inside a transition rate, which has no defensible meaning. A `lag`
node inherits every one of those, because the only thing worth lagging is a
flow.

### Option C — a projection variant

`Projection` is a separate sum type. Its blast radius, counted the same way: 8
OCaml source files (`compiler/expander.ml`,
`ir/{autodiff, constant_fold,
dimcheck, ir, lint, serde, validate}.ml`) and 7
Rust source files (`cli/src/{main, util}.rs`,
`ir/src/{observation, validate}.rs`, `runid/src/ir_hash.rs`,
`sim/src/inference/{multi_stream_obs, ode_grad}.rs`), plus tests. **15 files** —
and none of them is in the hot evaluator path: `resolved_expr.rs`,
`flat_eval.rs`, `propensity.rs`, `compiled_model.rs`, `gradient_capability.rs`,
`hierarchical.rs`, `obs_anchor.rs`, `coeff_guard.rs` and `init_order.rs` are all
untouched.

Ten goldens contain `cumulative_flow` today (7 regenerable, 3 frozen). Keeping
the unit-weight, unlagged lowering unchanged keeps every one byte-identical — a
one-line implementation rule with a ten-golden consequence if violated.

### Option D — the junction compartment, which is the status quo

Route both flows into a holding compartment and observe a fast drain. Costs
measured in `2026-08-20-incidence-over-a-flow-expression.md`: two extra
compartments in the sampled latent state, a rate constant invisible to every
parameter table and to run provenance, a compartment diagram that stops
describing epidemiology, and a dwell nobody chose.

For this case it is also simply the wrong object. gh#332 states the distinction:
staging moves real individuals between compartments; a reporting delay only
shifts when a flow is counted, and conflating them is a modelling error. And
quantitatively it cannot represent the measurement: a junction with rate `r`
gives an exponential delay with mean `1/r` and coefficient of variation 1, while
the measured administrative delay is 1.5 days with almost no spread. A
deterministic conveyor (`via fixed(τ)`, gh#330) would give the right shape but
still moves population through a compartment that nobody occupies.

## 5. Recommendation

**Add the lag as a per-term field on the weighted-flow projection variant that
Increment B already specifies, and expose it as a positional `lag(expr, d)` in
head position.**

```
Projection::WeightedFlowSum(Vec<FlowTerm>)

FlowTerm = { weight : Expr, flow : String, lag : Option<Expr> }
```

One variant at hash index 5, one `ir/VERSION` bump, two independently gated
surfaces. The per-reference accumulator that Increment B's B2 already requires
is the same refactor a per-term lag needs; doing it twice is where the
divergence in §3.6 would come from.

Unit-weight, unlagged forms keep their existing lowering to `CumulativeFlow` and
`CumulativeFlowSum`, so no golden containing them moves.

```camdl
cases_national:
  projected = lag(incidence(confirm_fast) + incidence(confirm_slow), 1.5 'days)
            + lag(incidence(die_community_fast) + incidence(die_community_slow),
                  3 'days)
```

## 6. Staging

**L1 — a literal point lag on a flow union, chain-binomial and ODE.** The
intermediate-representation variant, the per-term bin with a term-keyed close
and a stream-keyed reset, the lag axis on `Schedule`, the `Snap` refusal, the
`first_window_guard` extension, and the diagnostics in §9. Gradient-free methods
work unchanged. PGAS with NUTS works because a literal lag has no derivative
with respect to a parameter — but a rate coefficient _inside_ a lagged window
does, and that is where §7's incident recurs.

**L2 — an estimable lag, ODE backend only.** See §7.

**L3 — a distributed kernel** (gh#332).
`Σ_k w_k · [N(t_j − d_k) −
N(t_{j−1} − d_k)]`: the same boundary machinery with
`k` boundaries per term instead of one, and the same queue with depth
`⌈max_k d_k / Δ_min⌉ + 1`. Every mechanism L1 builds is the kernel's mechanism
at `k = 1`, which is why the point case does not foreclose it.

### Is the instant case separable, and should it go first?

It is separable, and it should go **last**, not first. The premise that the
accumulator is where the difficulty concentrates is inverted.

A lagged prevalence needs `x(t_j − d)` — the entire compartment vector at a past
time. The particle swarm carries only the current state, so a snapshot costs
`n_compartments` values per pending boundary per particle, against a lagged
incidence's _one_ value per lagged term per particle. On a stratified model with
compartments in the hundreds that is two to three orders of magnitude more
copying on every resample.

It is also less well defined. The lagged incidence bin is exact on both backends
because the flow is an integral over the window, and the integrator lands on the
window edges. A lagged prevalence read at an off-grid time is a snapshot of a
jump process; any implementation that interpolates between grid points is
interpolating counts, which is not a state the model can occupy.

And nobody has asked for it. gh#332 scopes the feature to a convolution over an
incidence history. A delayed serosurvey readout is a coherent object, but paying
the harder cost for the case with no user is the wrong order.

**Decision D4 (recommended): `lag(...)` accepts a flow union and refuses a state
argument by name**, with a diagnostic saying a lagged state read is not
supported and naming the follow-up issue. This is the `E341` pattern, which
already refuses mixing a flow with an instant state read in one projection.

## 7. Estimability

**What exists.** A parameter driving a forcing's `lag` yields
`DerivEntry::Unsupported { code: UnsupportedReason::Lag }`
(`ir/src/deriv.rs:38`, emitted by `autodiff.ml:250-254` via the `lag_mentions`
guard). The code is serialized and hashed into run identity; the human-readable
message is derived, so a copy-edit cannot re-key a run.

Inside `projected` the exposure is worse in kind, not only in degree, because
`projected` feeds the observation likelihood's `proj_grad` chain
(`Ir.diffable.proj_grad`, the derivative of a likelihood argument with respect
to the projection output), which both PGAS with NUTS and the ODE gradient path
consume.

**What an estimable lag actually requires.** With
`P_j(τ) = N(t_j − τ) −
N(t_{j−1} − τ)`:

_On the ODE backend_ the flow is the cumulative integral of the propensity, so

```
∂P_j/∂τ  =  −propensity_r(t_j − τ)  +  propensity_r(t_{j−1} − τ)
```

— a difference of instantaneous rates at the two window edges, both of which the
integrator already evaluates because those are the boundaries it lands on. The
derivative is available in closed form for free. An estimable lag is sound here.

_On chain-binomial_ `N` is an integer counting process, so `N(t_j − τ)` is a
step function of `τ`. The derivative is zero almost everywhere and undefined at
the jumps. This is not a derivative the compiler declines to emit; it is one
that does not exist. The gradient-free methods need no derivative, but they
inherit the other half: the likelihood is piecewise constant in `τ`, so a
Metropolis proposal smaller than the gap between jumps moves on a flat surface,
and mixing in `τ` is governed by the substep width rather than by the data — a
failure the sampler does not report.

**Decision D5 (recommended): an estimable lag is a backend-conditional
capability.** It goes through the capability system
(`CompiledModel::required_capabilities()` against each backend's declared set,
and `fit/methods.rs::check_model_capabilities`), refusing at dispatch with a
message naming the limitation. It is not omitted from a test and not silently
accepted — `.claude/rules/sim-and-inference.md` is explicit that every backend ×
method cell either works or fails loudly.

**A literal-only first release does not foreclose the estimable case.** The
field is `Option<Expr>` either way. Literal-only restricts what the expander
accepts there — after `resolve_expr` folds the unit literal, a `Const` — and
emits the existing `Unsupported { code: Lag }` for any parameter that reaches
it, exactly as the forcing path does. Nothing in the runtime mechanism changes
when the restriction lifts: the boundary set becomes dependent on the parameter
vector, which under `Exact` means recomputing `Schedule`'s boundary list per
evaluation. That is a per-iteration cost, not a change of representation.

The one thing that _would_ foreclose it: baking the lag into the schedule as a
constant at model-construction time. Keep it an `Expr`, evaluated when the
schedule is built for a given parameter vector.

**The precedent that sets the verification bar.**
`docs/dev/incidents/2026-07-05-lagged-forcing-autodiff-wrong-gradient.md`: the
forcing lag shipped with the forward path tested and the differentiated path
untested. The emitted rate gradient was built over bare `Time` while the value
was evaluated at `t − lag`, so the gradient consumed by the default Bayesian
method was wrong and sign-flipped over the period — correct `∂R/∂α = −0.6995` at
`t = 45` against an emitted `+0.6995`. The lesson transfers verbatim one layer
out: a lag feature lands with a finite-difference gradient gate against the
runtime forward evaluation, not only a value test.

## 8. Syntax

**Verified convention.** Built-in expression operators are positional with fixed
arity: `mod(a, b)` takes exactly two positional arguments and lowers to
`BinOp { op = Mod }` (`expander.ml:4344`, `E101` otherwise); `min(a, b)` and
`max(a, b)` likewise; the unary set `exp`, `log`, `sqrt`, `abs`, `floor`,
`ceil`, `sin`, `cos`, `tanh` takes exactly one. The grammar's argument rule
permits `k = e`, but every built-in arm matches the empty keyword. Declarations
take keywords: forcing arguments, likelihood arguments, priors, `transfer(...)`.
The `lag` that exists today is a forcing declaration argument, which is the
declaration convention correctly applied.

`incidence(...)` and `prevalence(...)` are head-position sugar on the right-hand
side of `projected`, and are positional. `lag(...)` sits in the same position.

**Decision D2 (recommended): positional, `lag(expr, 1.5 'days)`.**

What that trades away, plainly: a reader who has never seen the form does not
learn from the syntax which argument is the delay, where `length = 1.5 'days`
would tell them. Three things make the trade the right one. The unit literal is
self-documenting — `1.5 'days` in the second position of a call spelled `lag` is
not ambiguous. The argument order matches every other two-argument built-in, so
there is one rule to hold rather than two. And a keyword here would be the only
keyword argument to a head-position projection function, so `incidence(x)` and
`lag(x, length = …)` would disagree about a convention that is currently
uniform. There is also no good keyword available: `length` reads as a window
width, which is precisely what it is not; `by` is vague; `delay` repeats the
head.

**Decision D3 (recommended): no block-key sugar in this change.** As a primitive
it fails §1. As sugar it would be a second spelling whose scope — all of
`projected` — is invisible at the term carrying it, and Increment B's B5 records
that a syntactic convenience layered over a projection walk is where
silent-wrongs accumulate. If single-lag streams turn out to be the common case,
add it then; do not ship two spellings at once.

## 9. Diagnostics

`E300` through `E346` are taken; new codes start at `E347`. Each gets a row in
`docs/dev/warning-catalog.md`, and each names what the modeller wrote:

- a `lag` argument that is not a duration (dimension `T`), naming the inferred
  dimension and the `: duration` parameter kind — the wording the forcing path
  already uses;
- a negative lag;
- a `lag` over a state expression rather than a flow union, naming the deferral;
- a parameter-valued lag on a backend that cannot differentiate it, naming the
  backend and the method;
- a lag that is not a multiple of `dt` under `Snap`, naming both the declared
  and the realized delay and pointing at `--obs-alignment exact`;
- a lagged interval stream whose first window reaches before `t_start`, naming
  the required `condition_from`.

## 10. Verification

- **Red first, on the representation.** `lag(incidence(a), 3 'days)` lowers to
  the variant with the term's lag folded into model time units, asserted against
  emitted intermediate representation rather than against compile success.
- **The shift identity.** On the ODE backend with `d` an exact multiple of the
  observation cadence, a lagged stream's projected series equals the unlagged
  series shifted by `d` bins. No interpolation is involved, so it cannot pass
  vacuously.
- **Exactness off-grid.** On the ODE backend under `Exact` with `d` not a
  multiple of `dt`, the lagged bin equals `c_r(t_j − d) − c_r(t_{j−1} − d)` from
  an independent fine-grid integration, within integrator tolerance.
- **The gradient gate the 2026-07-05 incident demands.** Finite-difference
  against the analytic gradient for a parameter driving a rate _inside_ a lagged
  window — the case where the value uses the shifted window and a naive gradient
  would use the unshifted one. This is the forcing-lag bug one layer out, and it
  is the single most likely way this feature ships silently wrong.
- **Multi-cadence.** A two-stream model, one lagged and one not, on different
  cadences, with the lagged stream's bin asserted against a hand-computed
  window. The `per_stream_reset.rs:266` mutation-guard pattern, which is what
  catches a close keyed on the wrong event.
- **Close against reset.** A stream carrying the same flow twice with different
  lags contributes twice and resets once. A term-keyed reset silently drops one
  contribution and must fail this test.
- **Refusals**, each asserting the diagnostic names what the modeller wrote and
  never a sentinel: every case in §9.
- **Negative controls.** Every unlagged projection lowers unchanged, and the ten
  goldens containing `cumulative_flow` stay byte-identical.

## 11. Decisions for the maintainer

Each carries a recommendation; none is left open.

| #      | Decision                                                                    | Recommendation                                                                                                                        |
| ------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **D1** | One projection variant carrying both weight and lag, or two variants?       | **One.** The per-reference accumulator is shared; a second variant claims hash index 6 for a distinction that is not real.            |
| **D2** | Positional `lag(expr, d)` or keyword `length =`?                            | **Positional**, matching `mod`/`min`/`max`. §8 states the cost.                                                                       |
| **D3** | Block key `lag =` as sugar?                                                 | **No**, not in this change. It fails the per-term requirement as a primitive and duplicates the primitive as sugar.                   |
| **D4** | Interval and instant together, or interval first?                           | **Interval first**; refuse a state argument by name. §6 shows the instant case is the harder one and has no user.                     |
| **D5** | Literal-only first, or estimable immediately?                               | **Literal-only (L1)**, then estimable on ODE only (L2), capability-gated. The field stays `Option<Expr>`, so nothing is foreclosed.   |
| **D6** | Non-`dt`-multiple lag under `Snap`: round or refuse?                        | **Refuse**, naming the realized value. Rounding turns a 1.5-day delay into 1 or 2 days silently.                                      |
| **D7** | Land inside the aggregation arc's item 4b, or as its own `ir/VERSION` bump? | **Inside.** The accumulator refactor is shared, one bump serves both, and CLAUDE.md prefers landing pending re-keying changes as one. |

D7 carries a schedule risk worth naming: it couples this feature to Increment B.
The fallback keeps them decoupled without a second bump — land the variant with
both fields, and gate the two _surfaces_ independently, so a weighted term stays
refused by `E341` while `lag(...)` is accepted.

## 12. Surface obligations

This adds an accepted form to the language, so per
`.claude/rules/dsl-surface.md` it does not land without:

- **`docs/camdl-language-spec.md`** — the projection section gains `lag(...)`,
  and the stale paragraph at `:2829-2836` (which still says
  `incidence(a) + incidence(b)` does not sum) is corrected. Hand-edit only: the
  file embeds doctest preamble markers that `dprint`'s reflow breaks.
- **`docs/language-changes.md`** — a dated entry. This widens what compiles, so
  no existing model breaks.
- **`docs/dsl-cheatsheet.md`** and **`docs/user-features.md`** — the projection
  row.
- **`docs/dev/warning-catalog.md`** — one row per emit site in §9.
- **Release notes** — the `ir/VERSION` bump re-keys every stored fit and
  simulate for every model, not only models using a lag, because `ir_version` is
  a hashed field of the model digest.
