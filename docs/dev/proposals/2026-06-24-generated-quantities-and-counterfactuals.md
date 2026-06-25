# Generated quantities and counterfactual contrasts

Status: **Superseded** — split into two proposals after a third review round
showed the counterfactual half is a real fit-output + engine build, not
buildable from today's artifacts, while the quantities half is independent and
shippable now:

- Quantities →
  [`2026-06-25-generated-quantities.md`](2026-06-25-generated-quantities.md)
  (ships first).
- Counterfactual contrasts →
  [`2026-06-25-counterfactual-contrasts.md`](2026-06-25-counterfactual-contrasts.md)
  (deferred behind its prerequisites).

Retained as the design-history record of how the two converged.

## Summary

A camdl user can fit a model and recover a posterior, but cannot yet ask the
model to _report a derived quantity_ — cumulative incidence, attack rate, peak
prevalence, time to peak, or the headline policy number, **cases averted**.
Today the only way to compute a function of latent state is to smuggle it in as
a _scored_ observation stream (the expander requires a likelihood, `E266`),
which forces the author to pretend a reported quantity is data.

This proposal adds two blocks — `quantities {}` (the Stan `generated quantities`
analog) and `compare {}` (counterfactual contrasts across scenarios) — built
from machinery camdl already has. Quantities and contrasts are evaluated
**post-hoc over the posterior** by `camdl fit predict`; nothing here touches the
inference kernels, and the shared `Expr` type stays closed.

## Design at a glance

- A quantity is the **non-scored twin of an observation**: it reuses the flat
  `Projection` enum (`observation.rs:9`) + an optional temporal reduction, minus
  the likelihood. **No new `Expr` leaf.**
- **Most cumulative quantities are an absorbing stock** read directly
  (`final(D)`, `N0 - S`). The lifetime _flow_ accumulator (`cumulative(flow)`)
  is deferred to a follow-up, which dissolves the only "reset/differencing"
  hazard.
- A counterfactual is **forward simulation, not fitting**: it reads parameters
  (and, for the conditioned form, the fit's latent state at the fork) and runs
  two scenario arms forward, differencing a quantity per draw and banding over
  the posterior.
- The **conditioned fork** reads the fit's own conditioned latent state `X(T*)`
  and branches both arms from it under CRN. **Free-forward** (start from `t0`,
  parameters only) is the no-conditioning special case. Validity per inference
  method is enforced by a **`ForkableFit` type witness**.
- Scenario/quantity references use a **dot** (`no_sia.deaths`) — one general
  namespace operator (also the future home of grouped params, `immunity.gamma`),
  a non-breaking one-line lexer addition.
- Window endpoints are **instants** in the existing typed-time system
  (`origin + 20 'weeks`, `date("…")`), not durations.

## The seam: define vs serialize

camdl already separates _defining_ a computed-from-state quantity from
_serializing_ it. An `ObservationModel` is computed from state and written to a
file, yet it is **defined** in `observations {}` (`model.rs:180`) while
`output {}` only carries the serialization switch
(`output { observations = true }`). A generated quantity is the **non-scored
twin of an `ObservationModel`** — same `Projection`, no likelihood — so it gets
its own definition block beside `observations {}`, not a sub-block of
`output {}`. Three independent reviews converged on this seam.

Three blocks, one concern each: `scenarios {}` _defines_ baseline/intervention
(exists); `quantities {}` _defines_ derived functions of state (new);
`compare {}` _contrasts_ a named quantity across a scenario pair (new).

## Stocks first

A compartment count is a lifetime running stock (`CurrentPop`, never reset), so
the common cumulative quantities are read directly off state —
`total_deaths = final(D)` (`D` absorbing), `N0 - S` (cumulative incidence when
`S` is monotone). The lifetime _flow_ accumulator is needed only for flows
captured by no stock (SIRS waning, reinfections); it is a **named follow-up**,
not v1. This sidesteps the observation `CumulativeFlow` projection entirely —
that projection accumulates over a reporting _interval_ and **resets on the
cadence** (`observation.rs:32`), the opposite of a lifetime total, and reusing
it by "ambient context" would be an illegal second meaning of one IR value
(`temporal_kind()` is documented as derived-single-source). So v1 reads stocks
and never touches it.

## Types

### Quantity

```rust
/// The non-scored twin of an ObservationModel: a derived function of latent
/// state, no likelihood. Evaluated post-hoc over posterior draws.
pub struct Quantity {
    pub name:       String,
    pub strata:     Vec<StratumKey>,        // empty = whole-pop; reuses obs strata
    pub projection: Projection,             // reuses the observation enum
    pub reduce:     Option<TemporalReduce>, // None = series; Some = scalar
    pub window:     Option<Window>,         // None = whole horizon
}
```

`projection` reuses the existing enum: `prevalence = I/N` is `DerivedExpr(I/N)`;
`total_deaths = final(D)` is `CurrentPop(D)` + a reduction. v1 admits only the
non-flow projections (`CurrentPop`, `CurrentPopSum`, `DerivedExpr`) — see
"Stocks first".

### Temporal reductions — typed so the result kind is not underdetermined

A flat reduction enum would let `argmax` (returns a _time_) and `max` (returns a
_value_) share one shape, leaving the output dimension a function of the runtime
variant. Split them so the result kind is in the type:

```rust
pub enum TemporalReduce {
    /// value-returning: result has dim(series).  final, max, min, mean
    Value(ValueReduce),
    /// time-returning: result has dim T.          time_of_max, time_of_min
    Time(TimeReduce),
    /// dim(series)·T.                              integral (area under curve)
    Integral,
}
pub enum ValueReduce { Final, Max, Min, Mean }
pub enum TimeReduce  { ArgMax, ArgMin }
```

This is the third reduction axis (over _time_) and is intentionally distinct
from the existing `ObsReducer { Latest, Sum, Mean, Max }` (`intervention.rs:90`,
folds a trigger window) and the n-ary `Reduce` (`expr.rs`, sums over _strata_ at
one instant) — neither is temporal. Reductions fold over the trajectory
`fit predict` holds; **v1 folds over the output-cadence snapshots**, with the
resolution-honest substep fold a named fast-follow (see "Staging") — a `peak`
between output times can be missed at a coarse `every`, so a fine default
cadence is documented for peak-sensitive quantities. (Reading off the _endpoint_
— `final`, stocks, totals, **cases averted** — is cadence-invariant and exact
regardless.)

### Window — instants, reusing typed-time

```rust
pub struct Window { pub from: Instant, pub to: Instant }
```

Endpoints are **instants** (`[T]`, absolute), not durations. camdl already types
these (`lexer.mll:99-119`, `time_typing.ml`): an instant is `date(...)`,
`origin`, or origin-relative; a duration (`20 'weeks`) is a span. So an endpoint
is `origin + 20 'weeks` (instant = origin + duration; the affine algebra already
exists) or `date("2021-05-20")` for anchored models — the time checker _rejects_
a bare duration here. `last_obs` / `first_obs` are added as named instants
resolved at predict time (typed `instant`), not `Expr` leaves.

### Contrast

```rust
/// Forward-sim contrast: scalar arithmetic over a quantity under named scenarios.
/// A small AST (not Expr) so Expr stays closed; the referenced quantities carry
/// dimensions, so its dimcheck is a binop-agreement check.
pub struct Contrast { pub name: String, pub expr: ContrastExpr, pub window: Option<Window> }

pub enum ContrastExpr {
    Ref { scenario: String, quantity: String },   // surface: no_sia.deaths
    BinOp(Box<ContrastExpr>, ArithOp, Box<ContrastExpr>),
    Const(f64),
}
```

### `ForkableFit` — the witness that makes the conditioned fork valid by method

The conditioned ("state-continued") counterfactual is valid only for inference
methods that _produced_ a conditionable latent state, and that differs by
method. The type encodes it — `from_fit` is the only constructor, mirroring
`FilterableFit` (`predict.rs:189`):

```rust
/// Proof a fit supports the conditioned counterfactual. Constructible only from a
/// method that yields a forkable latent state; the gate is the type, not a runtime
/// check.
pub struct ForkableFit { draws: PosteriorDraws, continuation: Continuation }

pub enum Continuation {
    /// Stochastic backend with an aligned posterior latent path per draw
    /// (PGAS smoother; PMMH/PF saved paths). Both arms branch from X(T*),
    /// CRN-coupled. Variance-reducing pairing only means something here.
    StochasticCrn { latent_paths: AlignedLatentPaths },
    /// Deterministic backend (ODE): X(T*) is θ-determined; the fork continues the
    /// ODE solve. No CRN; conditioned ≡ free-forward.
    DeterministicOde,
}

impl ForkableFit {
    pub fn from_fit(f: &FitArtifacts) -> Result<Self, NotForkable> {
        match (f.method, f.backend, f.saved_aligned_latent_paths()) {
            (Pgas | Pmmh, ChainBinomial, true ) => Ok(stochastic_crn(f)),
            (_,           Ode,           _    ) => Ok(deterministic_ode(f)),
            (If2,         _,             _    ) => Err(NotForkable::PointEstimate),  // MLE: no posterior path
            (_,           ChainBinomial, false) => Err(NotForkable::PathsNotSaved),  // re-run saving paths
            (_,           Gillespie,     _    ) => Err(NotForkable::NoSmoother),
        }
    }
}
```

The conditioned-`compare` API **takes a `ForkableFit`**, so a conditioned
contrast on an IF2 fit doesn't type-check — the user gets
`NotForkable::PointEstimate` with a directive ("IF2 is an MLE — no posterior
latent trajectory to condition on; re-fit with PGAS, or run free-forward").
Free-forward needs **no** witness (parameters only), so it works for any fit
including IF2 and ODE.

### Model

```rust
pub struct Model {
    // …existing…
    pub quantities: Vec<Quantity>,  // new
    pub compare:    Vec<Contrast>,  // new
}
```

**Run-identity (owned):** the IR hash is a hand-written `ContentAddressed` walk,
not serde, so an empty `Vec` still writes a length prefix — adding these two
fields **re-keys every model**. This is a deliberate `ir/VERSION` bump + golden
regeneration (the same posture as the LICM field at 0.19, `ir_hash.rs:1071`),
done as one reviewed atomic golden update. v1 is _not_ run-id-neutral, and that
is intended.

## The dot namespace operator

`no_sia.deaths` is one dotted member-access operator (scenario.quantity now,
grouped params later). It is **non-breaking**: the lexer disambiguates by what
_follows_ the dot — `.5` needs a digit (`lexer.mll:174`, stays a float),
`no_sia.deaths` has a letter, so it falls to a new `DOT` token (the current
bare-`.` error at `:208`). An identifier never starts with a digit, so
`ident.ident` is unambiguous; the only non-construct that mis-lexes is `1.foo`
(number-dot-name), which no model writes. v1 restricts the grammar to
`IDENT DOT IDENT` (no `a.b[p]` postfix chains yet). (`@` rejected — it is the
rate operator `S --> E @ rate`; `->` rejected — one stroke from the `-->` flow
arrow.)

## The counterfactual runtime

`compare` is **forward simulation that reads a fit's output** — it never invokes
the inference kernels. Per posterior draw `i`:

```
read θ_i  (and, for the conditioned fork, the fit's latent state X_i(T*))
arm A (factual):        forward-sim from the start, no intervention,  seed s_i
arm B (counterfactual): forward-sim from the start, with intervention, seed s_i   ← CRN
averted_i = quantity(arm A) − quantity(arm B)
band averted over draws
```

- **Conditioned (default, retrospective):** both arms branch from the **fit's
  own conditioned latent state** `X_i(T*)` at the fork `T*` = the window's
  `from` instant. Zero pre-fork variance (one shared state); CRN cancels the
  common forward noise, so the band reflects the **joint (θ, X) posterior**.
  Requires `ForkableFit`. The fit already did the conditioning — compare _reads_
  `X_i(T*)`, it does not re-filter. The one fit-side need: PGAS already produces
  a conditioned latent path per sweep; **save it aligned to the draw** (today
  subsampled by `traj_stride`) so `saved_aligned_latent_paths()` holds.
- **Free-forward (prospective):** both arms start at `t0` from initial
  conditions, parameters only, CRN-identical until the intervention fires. No
  witness; any fit (or a point θ).

**Every contrast is banded over the posterior, and the band carries latent
uncertainty:** a stochastic-fit draw is the joint `(θ_i, X_i)`, so the band over
`averted_i` propagates parameter _and_ latent uncertainty. ODE's `X` is
θ-determined, so its band is over θ alone (no separate latent uncertainty); IF2
has no posterior, hence no band.

`enable`/`disable` of a timed intervention keeps the two arms byte-identical
through the firing substep (the vaccination `transfer` is RNG-free; draws happen
against pre-intervention state) and correlated after — the clean paired
contrast. `set`/`scale` from `t=0` are correlated-only (the backend consumes a
rate-dependent number of RNG words), so the recommended surface is the
intervention-toggle form. `fit predict` has no multi-scenario plumbing today
(`predict.rs:860` builds one inline baseline), so the two-arm replay is a real
(moderate) build, composed from the existing `simulate --draws` forward path + a
start-from-state seam.

## DSL surface — worked examples (against the live grammar)

### Generated quantities (single scenario)

```
compartments { S, E, I, R, D }

quantities {
  prevalence       = I / N                  # series   (DerivedExpr, no reduce)
  attack_rate      = final((N0 - S) / N0)   # scalar   (stock-derived)
  total_deaths     = final(D)               # scalar   (absorbing stock)
  peak_prevalence  = max(I / N)             # scalar   (value reduction)
  time_to_peak     = time_of_max(I)         # scalar   (time reduction → an instant)
  person_days_inf  = integral(I)            # scalar   (dim P·T)
}
```

Stratified, reusing the observation stratum form:

```
quantities {
  prevalence[p in patch]  = I[p] / N[p]
  attack_rate[p in patch] = final((N0[p] - S[p]) / N0[p])
}
```

### Counterfactual: cases averted from an SIA

```
interventions {
  sia : transfer(fraction = 0.6, from = S, to = V) at [origin + 20 'weeks]
}
scenarios {
  no_sia    { disable = [sia] }
  with_sia  { enable  = [sia] }
}
quantities { deaths = final(D) }

compare {
  averted     = no_sia.deaths - with_sia.deaths   over [origin + 20 'weeks, origin + 52 'weeks]
  rel_averted = 1 - with_sia.deaths / no_sia.deaths over [origin + 20 'weeks, origin + 52 'weeks]
}
```

`compare {}` is a block (consistent with `quantities {}`/`scenarios {}`); each
entry is `name = <contrast_expr> over [<instant>, <instant>]`, where `over` is a
new keyword binding looser than arithmetic (the whole contrast expression, then
the window). The window's `from` instant is the fork/conditioning point. For an
anchored model the same lines read
`over [date("2021-05-20"), date("2021-08-15")]`.

Free-forward (prospective), no witness needed:

```
compare { averted = no_sia.deaths - with_sia.deaths  free_forward to [origin + 52 'weeks] }
```

## Scoping: reductions are quantity-only, by a named check

`final`, `max`, `time_of_max`, `integral`, etc. live on `Quantity`, not in
`Expr`, so a temporal reduction cannot appear in a transition rate _in the IR_.
At the surface they are funcall syntax (`EFuncCall`), so a reduction name in a
rate parses and is rejected at expansion — with a **dedicated diagnostic**
(`E2xx`: "temporal reduction `max` is only valid in `quantities {}`; a rate is
evaluated per substep") naming the block, not the generic `E100`. Reduction
names are reserved against collision with `forcing {}` functions.

## Dimensional checking

Quantities and contrasts run through `dimcheck`; a quantity's dimension is
computed and **stored** (not the transient `projected_dim` scratch the obs path
discards) so a contrast can read it. Rules: `CurrentPop`/`DerivedExpr` per the
existing inference; `Value` reductions preserve `dim(series)`; `Time` reductions
(`time_of_max`) yield `T`; `Integral` yields `dim(series)·T`. A `ContrastExpr`
`+`/`-` requires equal operand dimension (`deaths - deaths` ✓;
`peak_prevalence - total_deaths` rejected), `/` yields the quotient
(`rel_averted` dimensionless).

## Output format

TSV, **one file per quantity / contrast** (`quantities/<name>.tsv`,
`compare/<name>.tsv`), **long/tidy keyed by stratum level**, bands reusing
`fit predict`'s `q05 q25 q50 q75 q95` over draws. One-per is required: a
_series_ quantity has a `time` column, a _scalar_ does not.

```
# series quantity, stratified     quantities/prevalence.tsv
time   patch    n_draws  q05 q25 q50 q75 q95

# scalar quantity                 quantities/total_deaths.tsv     (no time axis)
patch  n_draws  q05 q25 q50 q75 q95

# contrast                        compare/averted.tsv
patch  n_draws  q05 q25 q50 q75 q95
```

A `quantities.json` / `compare.json` manifest lists each entry's name, shape
(series|scalar), strata, reduction, window, **unit**, and (for contrasts) the
`Continuation` used and fork instant — so a consumer renders generically and the
provenance of a conditioned vs free-forward band is explicit. Resolved spec
points: `time_of_max` returns the **first** argmax on ties; an empty
`quantities {}` writes no files (not an error); quantities band over **all**
posterior draws; a `compare` ref to an undefined scenario, or with no
`scenarios {}` block, is a located error naming the missing scenario; a quantity
name sharing the compartment/obs namespace is a duplicate-name error; on ODE,
stocks/`final` are real-valued (expected incidence) vs integer counts on
chain-binomial — the manifest records the backend.

## Staging

- **v1 — quantities + the counterfactual.** `quantities {}`
  (stock/`DerivedExpr` + reductions, output-cadence fold), `compare {}` with the
  **conditioned fork via the fit's aligned latent path + CRN** (`ForkableFit`:
  PGAS/PMMH = `StochasticCrn`, ODE = `DeterministicOde`) and **free-forward**;
  the dot operator; instant windows; the multi-scenario two-arm replay in
  `fit predict`; the `ir/VERSION` re-key. Delivers attack rate, total
  deaths/cases, prevalence, and **cases averted** (the honest retrospective
  object).
- **Fast-follow — substep recorder.** A streaming reducer in the forward step
  loop → resolution-honest `peak` / `time_of_max` / `integral`. Orthogonal;
  nothing about cases-averted needs it.
- **Follow-up — flow `cumulative`.** A lifetime running-total accessor for flows
  not captured by a stock, with flow arithmetic (`cfr`); and
  `at_time(series, t)`.

## Decisions recorded

- Quantities reuse the flat `Projection` enum (no new `Expr` leaf); `Expr`
  (autodiff/dimcheck/flat-eval/run-id) stays closed.
- Stocks for cumulatives in v1; flow-`cumulative` deferred — dissolves the
  one-projection-two-meanings smell.
- Reduction result kind is typed (`Value`/`Time`/`Integral`), not a flat enum.
- Window endpoints are typed **instants** (`origin + duration`, `date`), reusing
  typed-time; durations rejected.
- Counterfactual = forward sim; conditioned fork _reads_ the fit's latent
  `X(T*)` (no re-filter), validity gated by the `ForkableFit` witness per
  method; CRN couples the arms; the band is over the joint (θ, X) posterior.
- Dot namespace operator, non-breaking; reductions quantity-scoped with a
  dedicated diagnostic; quantities/contrasts dimchecked.
- Own the `ir/VERSION` re-key.
- Separate `quantities {}` / `compare {}` blocks (the define/serialize seam).

## Follow-ups (named, not blocking)

- Flow `cumulative` + flow arithmetic (`cfr`); `at_time(series, t)`;
  mid-expression reductions; `a.b[p]` postfix dot chains; decoupling the
  conditioning instant from the accumulation window; geometry-aware scalar
  quantities once gh#306 lands.
