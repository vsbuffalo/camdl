# Simulation Backends

`camdl` ships three simulation backends for the same IR model: Gillespie (exact
SSA), chain-binomial, and ODE (fixed-step `rk4` or adaptive `rk45`). They all
take the same compiled model and parameter vector and return the same
`Trajectory` type. The choice is a tradeoff between fidelity, speed, and what
the downstream analysis requires.

```
camdl simulate model.ir.json --params base.toml --backend gillespie
camdl simulate model.ir.json --params base.toml --backend chain_binomial --dt 1
camdl simulate model.ir.json --params base.toml --backend ode          --dt 0.1
```

---

## Shared concepts

### Compartment types

The IR distinguishes two compartment kinds:

- **Integer** (`CompartmentKind::Integer`) — the standard epidemic compartment.
  Head-counted; must be a non-negative integer. The stochastic/discrete backends
  maintain these as `i64`. The ODE backend promotes them to `f64`.

- **Real** (`CompartmentKind::Real`) — a continuous-valued auxiliary variable,
  typically a pathogen concentration, waning immunity index, or environmental
  reservoir. These are governed by user-written explicit ODE equations (`ode {}`
  block) and advanced with RK4 in every backend.

### Rate expressions and propensity

Each transition has a rate expression. For stochastic backends, the rate is the
_propensity_ of the Poisson process (events per unit time). For the ODE backend
it is the _flux_ (individuals per unit time) that enters the derivative.

Rate expressions are evaluated by `eval_expr` (see `propensity.rs`), a pure
expression interpreter that reads compartment counts, parameters, tables, and
time functions. No allocation occurs in steady state.

### Interventions

Interventions are deterministic state modifications at scheduled times (e.g.,
vaccination campaigns). All backends handle them by:

1. Truncating the current step at the intervention time.
2. Calling `apply_interventions_at`, which applies `transfer` / `set` actions.
3. Resuming the simulation from the new state.

After a Gillespie intervention, propensities are fully recomputed from the
modified state — the remaining exponential waiting time is discarded (the
process is memoryless, so this is exact).

### Common Random Numbers (CRN)

Scenarios run with the same seed produce correlated trajectories. The baseline
and an intervention scenario share the same `StatefulRng` stream initialized
from the same seed. Before the intervention fires, states and propensities are
identical — so the sequential RNG draws are identical — meaning trajectories are
byte-identical up to the intervention time. After the intervention, trajectories
diverge naturally while remaining positively correlated. This coupling reduces
variance in scenario comparisons without any special handling.

### Output recording

All backends record snapshots at the output times declared in the model's
`output {}` block. Between output times, the simulation advances internally
using its own time stepping. Cumulative flows (transition fire counts, or
continuous-flow integrals in ODE mode) are accumulated between output boundaries
and reset at each snapshot.

---

## Gillespie (exact SSA)

**When to use:** small-to-medium populations where stochastic extinction,
outbreak probability, or the exact distribution of event times matters. The gold
standard for correctness.

### Algorithm

The Gillespie Stochastic Simulation Algorithm (Gillespie 1977) is exact for
continuous-time Markov chains. Given state **X** and propensities λ₁, …, λ_M:

1. Compute the total propensity: Λ = Σᵢ λᵢ
2. Draw the waiting time to the next event: τ ~ Exp(Λ), i.e. τ = −ln(U₁)/Λ where
   U₁ ~ Uniform(0,1)
3. Select which event fires: choose transition j with probability λⱼ/Λ
4. Apply the stoichiometry: **X** ← **X** + νⱼ
5. Update propensities and goto 1

This is _exact_ in the sense that the joint distribution of (τ, j) is exactly
the correct first-passage distribution of the CTMC.

### Implementation details (`gillespie.rs`)

**Incremental propensity updates.** After transition j fires and changes
compartments according to νⱼ, only transitions whose rate expressions depend on
those compartments need recomputation. The `comp_to_transitions` map (built at
model compile time) gives the set of affected transitions for each compartment.
This makes each event O(k) where k is the mean number of transitions sharing a
compartment, rather than O(M).

**Full recompute every 10 000 events.** Incremental floating-point additions to
`lambda_total` accumulate roundoff error. A periodic full recompute (sum of all
propensities from scratch) prevents drift. The interval is tunable via the
`FULL_RECOMPUTE_INTERVAL` constant.

**Time-dependent transitions.** Transitions whose rate expression contains a
`TimeFunc` (seasonal forcing, piecewise schedules) cannot be maintained by the
compartment-change dependency graph alone. They are tracked separately in
`time_dep_transitions` and re-evaluated whenever simulation time advances to a
boundary (output time, intervention time), even if no integer state changed.

**Absorbing states.** When Λ = 0 (no events possible), the simulation fast-
forwards to the next intervention or output time. After applying an
intervention, propensities are recomputed — the system may become active again
(e.g., a vaccination campaign seeds a new outbreak).

**Real compartments (PDMP).** If the model contains real compartments with ODE
equations, Gillespie operates as a Piecewise-Deterministic Markov Process
(PDMP). Between discrete events, the real state advances continuously by RK4.
This is an approximation for v0.1 — a correct PDMP implementation would use
thinning to account for propensity changes driven by the real state during the
interevent interval. The v0.1 implementation advances real state in one RK4 step
at each event, which is accurate when real-state dynamics are slow relative to
the event rate.

### Complexity and scaling

- **Time per event:** O(k) amortized (incremental update) where k ≪ M for sparse
  dependency graphs (typical epidemic models).
- **Total events:** proportional to Λ × T. For a 10 000-person SIR at peak, Λ ≈
  3000 events/day; a 100-day simulation ≈ 300 000 events per run.
- **Scales poorly** with N: events ∝ N, so 1M-person runs are often impractical.
  Use chain-binomial or ODE for large populations.

---

## Chain-binomial

**When to use:** discrete-time models where the generation interval is the
natural time step (e.g., daily surveillance data, weekly incidence). Respects
integer constraints by construction (multinomial competing risks), so no
clamping is needed.

### Algorithm

Reed-Frost chain-binomial model (Abbey 1952): in a discrete time step Δt, each
susceptible escapes infection independently with probability exp(−λ · Δt)
(survival probability under a constant-rate Poisson process). Therefore the
number who become infected is:

```
Δn_infection ~ Binomial(S, 1 − exp(−λ · Δt))
```

More generally, for any transition with rate λ and source population n_src:

```
p = 1 − exp(−λ · Δt)
Δnⱼ ~ Binomial(n_src, p)
```

This is exact if λ is truly constant over the interval and events are
independent (no competition between transitions for the same source).

### Implementation details (`chain_binomial.rs`)

**Multinomial competing risks.** When multiple transitions draw from the same
source compartment (e.g., infection and death both depleting S), the
chain-binomial uses a multinomial draw — not independent binomials. This matches
pomp's `reulermultinom` semantics: the total number leaving a compartment is
bounded by the compartment size, and the split between competing outflows is
proportional to their rates.

Transitions are precomputed into **source groups** at model compilation time
(`CompiledModel::source_groups`). For a source compartment holding `n_src`
individuals with k competing outflows, each step is pomp's `reulermultinom`:

1. Compute the per-capita rate of each outflow: `r_i = propensity_i / n_src`
   (times the Gamma multiplier for an `overdispersed` outflow, below).
2. Draw the total number leaving the compartment from one exact binomial on the
   summed rate: `n_exit ~ Binom(n_src, 1 − exp(−(r_1 + … + r_k) · Δt))`.
3. Split `n_exit` among the outflows in proportion to their rates, by
   conditional binomials:
   - `count_1 ~ Binom(n_exit, r_1 / (r_1 + … + r_k))`
   - `count_2 ~ Binom(n_exit − count_1, r_2 / (r_2 + … + r_k))`
   - and so on; the last outflow takes the remainder.
4. Guarantee: `Σ count_i = n_exit ≤ n_src` (no overdraw).

This is not the same as drawing each outflow with its own probability
`p_i = 1 − exp(−r_i · Δt)`: because `1 − exp(−x)` is subadditive, those
probabilities sum to more than the probability of leaving at all, and the total
would be over-counted. For a source with a single outflow the step is one
binomial, `Binom(n_src, 1 − exp(−r · Δt))`.

**Exact binomial draws.** Every binomial above is an exact draw, not a Poisson
approximation: BTPE (Kachitvichyanukul & Schmeiser 1988) by default, or BTRS
(Hörmann 1993) when a PGAS fit selects it. A draw never exceeds its `n`.

**Inflows.** Transitions with no source compartment (births, importation) are
not part of any source group. They draw from the total propensity directly via
`Poisson(rate · dt)`.

**Per-capita rate conversion.** The IR stores total propensities (e.g.,
`mu × S`). The chain-binomial divides by `n_src` to recover the per-capita rate
before converting to probability. This is critical: using the total propensity
directly would give `p = 1 − exp(−mu·S·dt) ≈ 1.0` for large compartments,
killing the entire population in one step.

**Overdispersion.** When a transition has `overdispersed(rate, σ²)`, a Gamma
multiplier is applied to its per-capita rate before the source's exit draw:
`effective_rate = per_capita × G` where
`G ~ Gamma(shape = Δt/σ², scale =
σ²/Δt)`, drawn once per transition per step.

**Real compartments.** Advanced with RK4 _before_ the multinomial draws (using
the start-of-step integer state): for chain-binomial, the continuous dynamics
represent processes that run in parallel with (rather than after) the discrete
transitions.

### Bounded competing risks

The multinomial draw makes chain-binomial bounded by construction:

1. **Multinomial, not independent.** Chain-binomial draws competing transitions
   from a shared source as a single multinomial, so total outflow cannot exceed
   the source population.

2. **No clamping needed.** The multinomial guarantees `Σ count_i ≤ n_src` by
   construction — there is no post-step clamping to zero.

3. **Matches Euler-multinomial.** The chain-binomial is equivalent to pomp's
   `reulermultinom` when using the same per-capita probabilities — making it the
   appropriate backend for validating against pomp implementations.

---

## ODE

**When to use:** large populations where stochasticity is negligible, or for
fast deterministic exploration of parameter space before running stochastic
ensembles. Agrees with Gillespie/chain-binomial in expectation (same rate
expressions drive both).

### Algorithm

Fourth-order Runge-Kutta integration of the system:

```
dXᵢ/dt = Σⱼ νᵢⱼ · λⱼ(X, θ, t)
```

where νᵢⱼ is the stoichiometry (±1 or 0) of compartment i in transition j, and
λⱼ is the transition rate evaluated at state X, parameters θ, and time t.

For each RK4 step from t to t + h:

```
k₁ = f(X(t),       t)
k₂ = f(X + h/2·k₁, t + h/2)
k₃ = f(X + h/2·k₂, t + h/2)
k₄ = f(X + h·k₃,   t + h)

X(t + h) = X(t) + h/6 · (k₁ + 2k₂ + 2k₃ + k₄)
```

Global truncation error is O(h⁴).

### Adaptive stepping (`rk45`)

`rk4` is the default, fixed-step integrator. Select the adaptive Dormand–Prince
RK4(5) integrator when the dynamics are stiff or you would rather control error
than hand-tune `dt`; it sizes its internal substeps to meet a tolerance:

```bash
camdl simulate model.camdl --backend ode --integrator rk45
```

Tolerances are a **model property** — declared in the `simulate {}` block so a
fit and its forward simulations always share them — and default to
`atol = 1e-8`, `rtol = 1e-6`:

```
simulate {
  from = 0 'days
  to   = 365 'days
  integrator = rk45 { atol = 1e-9 rtol = 1e-7 }
}
```

(`rk4` takes no tolerances — giving it `atol`/`rtol` is a located **E106**.) An
`rk45` simulation refuses a model whose rate reads `dt`: with adaptive stepping
there is no fixed step for `dt` to denote.

### Implementation details (`ode.rs`)

**Unified float state.** Integer compartments are promoted to `f64` at the start
of the ODE run and remain continuous throughout. The ODE system has dimension =
n_int + n_real (all compartments).

**Derivative sources.**

- _From transitions (auto-derived):_ for each transition j, the rate λⱼ is
  evaluated at the current state; the contribution to compartment i's derivative
  is `νᵢⱼ · λⱼ`.
- _From explicit ODE equations:_ compartments declared `real` have their
  derivative specified directly in the `ode {}` block and evaluated verbatim.

**Propensity evaluation during RK4 substeps.** At each k₁–k₄ evaluation, integer
compartments are read at their full `f64` value: the evaluator takes the float
state as an override (`int_float_override`), so a rate that reads an integer
compartment sees `12.37`, not `12`. Nothing is rounded to `i64` inside a step,
and the augmented flow derivative reuses the same propensities. A very small
compartment therefore decays smoothly rather than snapping to zero; the
deterministic approximation still cannot represent extinction of a finite
population. Rounding to `i64` happens only where an integer state is needed: in
the output snapshot, and for the one exception below.

The exception is a model whose rate depends on the step size (`dt` in a rate
expression). Its cumulative flow is accumulated by a left-rectangle rule at the
realized substep, with the propensities evaluated at the start-of-step state
rounded to `i64`; the RK4 stages themselves still read the unrounded state. The
CLI warns once when it loads such a model.

**Non-negativity.** A fixed-step `rk4` step that drives a compartment below zero
is refused, and the run stops with an error that names the compartment, the time
and the step size. Setting the compartment back to zero instead would keep the
flow the step had already passed downstream, so the step would add people: a
closed cohort of 10⁹ in an 8-stage `via erlang` chain at rate 0.2 per day grows
to 3.2 × 10⁹ in eight days at `dt = 1`. Only a floating-point undershoot,
smaller than 10⁻⁹ of the compartment's value before the step (or 10⁻⁹ absolute
below one person), is set to zero silently.

The usual cause is a step that is too long for the fastest rate. A linear chain
of `k` stages at rate `r` per stage, taken on its own, stays non-negative on
every step when `dt · k · r ≤ 1`; above that, a step can overshoot while an
upstream stage holds much more than a downstream one, as at the start of an
outbreak or after an importation. The fix is a smaller `dt`. If a smaller `dt`
does not remove the error, a rate in the model does not fall to zero as its
source compartment empties (a constant outflow, for example), and the exact
solution itself goes negative.

Inside a fit the refusal depends on the parameters, so it rules out that
parameter set rather than stopping the fit: `mh` and `nl-sbplx`/`nl-bobyqa`
score it as a log-likelihood of −∞ and move on. `nl-lbfgs` stops, because it has
no gradient to follow at such a point. The adaptive `rk45` integrator still sets
a negative accepted result to zero.

**Interventions.** At each intervention time the action is applied exactly in
`f64`, with no rounding, so `transfer(fraction = 0.5, ...)` from a compartment
holding 5 moves 2.5. The changes an event phase makes combine against that
phase's snapshot, as on the discrete backends. An integer compartment the phase
leaves negative beyond floating-point rounding is refused, as the discrete
backends refuse it.

**Flows.** Cumulative flows are accumulated as `f64` and recorded without
rounding, so a flow below one individual over an output interval survives into
the output and the likelihood. The stochastic backends record integer flows.

**Seed parameter.** The `--seed` CLI argument is accepted but ignored; ODE runs
are fully deterministic.

### When ODEs and SSA disagree

The ODE is the mean-field limit (N → ∞) of the CTMC. They disagree in three
important ways:

1. **Extinction.** The ODE never reaches zero — it asymptotically approaches
   zero. The CTMC can and does hit the absorbing state I=0. If outbreak
   probability or time-to-extinction is the question, use Gillespie.

2. **Stochastic amplification.** Near critical points (R₀ ≈ 1), random
   fluctuations can drive large outbreaks that the ODE's deterministic
   trajectory misses. The ODE underestimates outbreak probability in this
   regime.

3. **Jensen's inequality effects.** Nonlinear rate expressions (e.g., βSI/N)
   evaluated at the mean state differ from the mean of the expression evaluated
   at random states. For compartmental models this is typically a small
   correction, but it can matter for highly variable forcing functions.

### The ODE backend as an inference backend

The ODE skeleton is not only a forward simulator — it is a fitting backend.
Because integrating the mean-field system yields one trajectory per parameter
point, the latent-trajectory integral collapses and the likelihood
`p(y | θ, ODE skeleton)` can be evaluated directly, with no particle filter.
`camdl fit` fits this deterministic likelihood two ways:

- **`nl-sbplx` / `nl-bobyqa` / `nl-lbfgs`** — maximum likelihood (NLopt
  deterministic optimizers). `nl-sbplx` and `nl-bobyqa` are gradient-free;
  `nl-lbfgs` is driven by symbolic forward sensitivities (`∂x/∂θ` carried
  through fixed-step `rk4`) and requires a differentiable model.
- **`mh`** — Bayesian posteriors, by gradient-free adaptive Metropolis-Hastings.

This targets a different statistical object than the stochastic backends
(`chain_binomial → p(y | θ)` under process noise; `ode → p(y | θ, ODE skeleton)`
— the Jensen's-inequality difference above), appropriate when demographic
stochasticity is negligible. See `camdl docs inference` (the ODE-backend fitting
section) for when to pick which, and `camdl fit methods` for the live
`(algorithm, backend)` matrix. Adaptive `rk45` is refused by `nl-lbfgs` (no
fixed step for the sensitivities); use `rk4`.

---

## Comparison table

| Property                | Gillespie     | Chain-binomial | ODE                |
| ----------------------- | ------------- | -------------- | ------------------ |
| Time type               | continuous    | discrete       | continuous         |
| Stochastic              | yes           | yes            | no                 |
| Exact                   | yes           | approx         | approx             |
| Extinction behavior     | correct       | correct        | never              |
| Step size               | event-driven  | user (--dt)    | user (--dt)        |
| Speed (large N)         | slow          | fast           | fast               |
| Speed (small N)         | fast          | overhead       | fast               |
| Real compartments       | PDMP (approx) | hybrid         | native             |
| CRN coupling            | yes           | yes            | n/a                |
| Overdispersion          | incompatible  | supported      | incompatible       |
| Int rounding during RK4 | n/a           | n/a            | yes (O(1/N) error) |

**Rule of thumb:** use Gillespie for N < 10 000 and when extinction matters;
chain-binomial for N > 10 000 in production stochastic runs (especially when the
generation interval aligns with your Δt); ODE for fast parameter sweeps or very
large spatial models. If the model uses `overdispersed()` transitions, Gillespie
and ODE are rejected at runtime — use chain-binomial.

---

## Extra-demographic stochasticity (overdispersion)

Transitions wrapped in `overdispersed(rate, σ²)` receive Gamma-distributed
multiplicative noise on their rate (He et al. 2010): over a step of length Δt
the rate is multiplied by a Gamma variable of mean 1 and variance σ²/Δt, so the
step's event count keeps its mean and gains variance.

**Chain-binomial implementation.** For each overdispersed transition per step:

1. Evaluate the propensity λ and the intensity σ² at the start-of-step state
2. Draw a rate multiplier G ~ Gamma(shape = Δt/σ², scale = σ²/Δt), which has
   mean 1 and variance σ²/Δt
3. For a transition out of a compartment, multiply its per-capita rate by G and
   draw the source's exits exactly as above (one binomial for the total, then
   the proportional split), so the count stays bounded by the source. For an
   inflow (no source compartment), draw `Poisson(G · λ · Δt)`.

For an inflow this is a Poisson–Gamma mixture, a negative-binomial count with
mean λΔt and size Δt/σ² (variance λΔt + (λΔt)²σ²/Δt). For an exit it is a
binomial whose hazard is Gamma-perturbed, as in He et al. (2010) and its pomp
implementation (`rgammawn` noise on the rate passed to `reulermultinom`). When
σ² → 0 the Gamma concentrates at 1 and the draw converges to the plain binomial
or Poisson draw.

**Backend capability system.** Each model declares required capabilities
(derived from the IR at compile time). Each backend declares what it supports.
Mismatch produces a hard error before simulation starts — no silent wrong
answers.

```
$ camdl simulate model.ir.json --backend gillespie --seed 42
error: model requires capabilities not supported by backend 'gillespie':
  - OVERDISPERSION: transitions with overdispersion require --backend chain_binomial
```
