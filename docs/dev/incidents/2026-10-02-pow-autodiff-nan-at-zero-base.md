# Power-rule autodiff emits NaN derivatives where the base is zero

Date: 2026-10-02

Status: **fixed** (gh#952). The autodiff rule is fixed in `6311034c`; the ODE
NUTS startup refusal and the corrected `nl-lbfgs` message are in `466f5ab6`.
Every new test was confirmed red before its fix and green after.

Class: **code-vs-code**. The OCaml autodiff emits a derivative expression that
the Rust runtime cannot evaluate at a point where the true derivative is finite.
The fix belongs on the OCaml side. Tests pin the emitted derivative's value at
the domain edge, and the ODE NUTS startup probe refuses a non-finite gradient.

## What happened

The compiler differentiates `f^g` with the general power rule:

```
d(f^g) = f^g · (g′·ln f + g·f′/f)
```

This rule is correct for `f > 0`. At `f = 0` both of its terms contain a
singular factor: `ln 0` in the first, `1/f` in the second. In the true
derivative those factors are multiplied by a vanishing `f^g`, and the product's
limit is finite. The simplifier removes a term whose coefficient is a literal
zero. It never cancels `f^g · (1/f)` into `f^(g−1)`, so the singular factor
survives into the emitted expression.

By default the runtime evaluates `log(0)` and `x/0` to NaN
(`rust/crates/sim/src/resolved_expr.rs:594`, `:630`), and then `0 · NaN = NaN`.
The emitted derivative is therefore NaN wherever the base is exactly zero:

- `∂(I^α)/∂α = I^α · ln I`. The true value at `I = 0` is 0 for `α > 0`; the
  emitted value is NaN.
- `∂(I^2)/∂I = 2I`. The true value at `I = 0` is 0; the emitted value is NaN.

Compartment counts are exactly zero all the time: an empty compartment at the
start of a run, a patch with no infections, a stochastic trajectory that dies
out. A zero base is ordinary, not a corner case.

Forward simulation and the gradient-free methods (IF2, bootstrap particle
filter, PMMH) never evaluate derivative expressions and are unaffected.

## Detection

An external code review of `autodiff.ml` reported the domain problem in the
symbolic rule but did not run anything. This report confirms it by compiling a
model and running the ODE forward sensitivities. No existing test evaluates a
compiled derivative at a zero base.

## Reproduction

### 1. The emitted derivative contains the singular factor

Model (`probe.camdl`):

```
time_unit = 'days
compartments { S, I, R }
let N = S + I + R
parameters {
  beta  : rate  in [0.01, 5.0]
  alpha : positive [1] in [0.5, 1.5]
  gamma : rate  in [0.01, 1.0]
}
transitions {
  inf_alpha : S --> I  @ beta * S * unchecked_dim(I ^ alpha, dim = population, reason = "probe") / N
  rec_sq    : I --> R  @ gamma * R * unchecked_dim(I ^ 2, dim = population, reason = "probe") / N
}
init { S = 990  I = 10  R = 0 }
simulate { from = 0 'days  to = 50 'days }
```

```
$ ocaml/_build/default/bin/camdlc.exe probe.camdl > probe.ir.json
```

`inf_alpha`'s `rate_grad["alpha"]` contains `I^α · log(I)`:

```
{"bin_op":{"op":"mul","left":{"bin_op":{"op":"pow","left":{"pop":"I"},"right":{"param":"alpha"}}},"right":{"un_op":{"op":"log","arg":{"pop":"I"}}}}}
```

`rec_sq`'s `rate_state_grad["I"]` contains `I^2 · (2 · (1/I))`. The fragment
below is the `2 · (1/I)` factor:

```
{"bin_op":{"op":"mul","left":{"const":2.0},"right":{"bin_op":{"op":"div","left":{"const":1.0},"right":{"pop":"I"}}}}}
```

### 2. The NaN propagates through the whole ODE sensitivity solve

Model (`probe2.camdl`): an SIR model with a waning flow whose rate contains a
power of a compartment that starts at zero.

```
transitions {
  inf  : S --> I  @ beta * S * I / N
  rec  : I --> R  @ gamma * I
  wane : R --> S  @ w * R * unchecked_dim(R ^ alpha, dim = population, reason = "probe") / N
}
init { S = 990  I = 10  R = 0 }
```

The parameters are `beta = 0.5`, `gamma = 0.1`, `w = 0.01` and `alpha = 0.97`.
All four are estimated. The model is integrated through
`sim::ode::integrate_obs_sensitivity` with `dt = 1` over `t ∈ [0, 30]`. The true
derivatives of the `wane` rate at `R = 0` are all zero:
`∂/∂R = w·(1+α)·R^α/N − w·R^(1+α)/N²` and `∂/∂α = w·R^(1+α)·ln R/N`, both of
which vanish as `R → 0`.

| `R(0)` | forward state            | `state_sens`, `inc_sens` (all 4 params, t = 1…30) |
| ------ | ------------------------ | ------------------------------------------------- |
| 0      | finite (`R(30) = 770.5`) | **every entry NaN**                               |
| 1      | finite (`R(30) = 771.1`) | every entry finite                                |

The NaN reaches `beta` and `gamma` as well, though neither appears in the `wane`
rate. The forward sensitivity equation multiplies the state Jacobian
`J_x = ∂rate/∂x` by the sensitivity matrix. One NaN entry in `J_x` therefore
contaminates every parameter's column at the first step. It is never washed out,
because `NaN · 0 = NaN`.

## Root cause

`ocaml/lib/ir/autodiff.ml`, the `Pow` arm of `differentiate` (`:465–474`):

- It uses the general rule for every exponent, including a constant one or a
  parameter, which is constant with respect to the differentiation variable.
- It expresses the `g·f′/f` term with an explicit division by the base, instead
  of the equivalent `g·f^(g−1)·f′`.
- It applies `ln f` with no guard for `f = 0`, where `f^g · ln f → 0` for
  `g > 0`.

`simplify` (`:677–757`) cannot repair this. It folds constants and removes
identities, but has no rule to combine `f^g · (1/f)`.

## Blast radius

What happens downstream depends on which gradient method consumes the NaN.

- **`nl-lbfgs`** (gradient MLE on the ODE). It refuses a non-finite gradient and
  fails the fit with an error (`rust/crates/cli/src/fit/nlopt_stage.rs:615`).
  The error message misdirects for this cause: it advises narrowing bounds or
  starting elsewhere. The zero comes from a fixed initial condition, so every θ
  fails. Loud, but wrongly explained.
- **ODE NUTS.** The startup probe requires only a finite `log_p`
  (`ode_nuts.rs:266`), so it accepts the NaN gradient. Every trajectory's first
  leapfrog step starts from that gradient and produces a NaN energy, so every
  proposal is rejected and the chain stays at its initial values. The run
  returns normally with every transition divergent. Measured with a
  hand-installed NaN state derivative
  (`ode_nuts::tests::non_finite_initial_gradient_is_refused`, before its fix): 5
  samples, 5 divergent.
- **PGAS (particle Gibbs with ancestor sampling) with NUTS for θ.** A transition
  whose rate is at most `RATE_EPSILON` is skipped before its derivative is
  evaluated (`pgas_grad.rs:139`). That hides the common case: `β·S·I^α/N` is
  itself zero when `I = 0`. The NaN is reached only when a zero-base power sits
  inside a _positive_ rate, e.g. `β·S·(I₁^α + I₂^α)/N` with one patch empty.
  - The NUTS target checks the complete-data log-likelihood for errors but does
    not check that the gradient is finite (`pgas.rs:4495`). The leapfrog
    integrator therefore produces a NaN energy, the step is marked divergent,
    and θ does not move on that sweep.
  - This slows mixing but does not bias the stationary distribution. Whether a
    sweep hits the NaN depends only on the current trajectory `x`. On those
    sweeps the θ-update is the identity kernel, which leaves `p(θ | x)`
    invariant.

In-repo exposure: the only models containing `^` are the He et al. (2010)
measles cases (`tests/external/cases/he2010_*`) and the
`phenom_mixing_unchecked` golden. Both write `(I + iota)^alpha`. In the He 2010
cases `iota` has the lower bound `0.01`, so their base never reaches zero.
Models outside this repository have not been audited.

Results from earlier gradient-based fits are unlikely to be biased. On the ODE
path, the only way a continuous state is exactly zero is a fixed initial
condition, which does not depend on θ: the fit either fails at every θ or is
unaffected. A step that clamps a state to zero is refused separately. On the
PGAS path the defect slows mixing without biasing the draws, and the stalled
sweeps are counted as divergences.

## A genuinely singular derivative is a different case

For a constant exponent `0 < c < 1`, which includes He et al.'s mixing exponent
`α ≈ 0.97`, the state derivative `c·f^(c−1)` is genuinely infinite at `f = 0`.
No differentiation rule makes this finite. The fixed rule still produces a
non-finite value there, by design. A correct gradient method cannot use a model
with an infinite derivative, and a finite value at that point would be wrong.

`--allow-degenerate-rates` maps every NaN in expression evaluation to 0,
derivatives included. With it, the `x^2` case happens to be correct. The
`0 < c < 1` case becomes a finite wrong value: 0 where the truth is `+∞`. The
flag is opt-in and outside this fix's scope.

## Remediation

1. **Autodiff** (`autodiff.ml`). Differentiate powers as

   ```
   d(f^g) = g·f^(g−1)·f′ + g′·Cond(f > 0, f^g·ln f, 0)
   ```

   - It is exact for `f > 0`.
   - At `f = 0` with `g > 0` it gives the one-sided limit.
   - When `g′` simplifies to `0` it reduces to the constant-exponent rule
     `g·f^(g−1)·f′`.
   - The `Cond` evaluates only the branch it takes, so the `ln f` branch is
     never evaluated at `f ≤ 0`.
   - For `f < 0` with a differentiated exponent the guard returns 0. The forward
     value `f^g` is itself NaN there unless `g` is an integer, so that region is
     already outside the model's domain.
2. **ODE NUTS startup.** The startup probe refuses a finite posterior with a
   non-finite gradient, naming the parameter and the usual cause, instead of
   starting a chain that cannot move. Later evaluations already map a non-finite
   gradient to `−∞` (`target_or_neg_inf`), so the starting point was the only
   unguarded entry. This is the gh#811 class again: the value and its gradient
   disagree about where the target is defined. After the autodiff fix it is
   still reachable through a genuinely singular derivative (previous section).
3. **PGAS: no target change.** `nuts_step` already marks a NaN-energy leaf
   divergent and refuses a non-finite proposal (gh#81). A sweep whose current
   trajectory makes the gradient non-finite leaves θ in place, which preserves
   `p(θ | x)`. Returning `−∞` from the target would behave identically. What
   remains open is reporting: these sweeps are counted as ordinary divergences,
   indistinguishable from a step size that is too large.
4. **The `nl-lbfgs` message.** It now names a singular derivative at a zero
   compartment as the likely cause when every start fails, before suggesting
   bounds or a different start.

## What this suggests

No gate compares compiled derivatives with finite differences at the edges of
their domain. The existing finite-difference tests evaluate in the interior,
where every rule is correct. That is the second recurring gap here:

- Derivative rules should be tested against finite differences at their boundary
  points, such as a zero base or a zero compartment, not only at interior
  points.
- Every gradient entering NUTS should pass through one finiteness check, rather
  than each target having its own.
