---
date: 2026-10-27
status: decided — D1–D4, D6, D7 decided by the maintainer on 2026-10-27; D5 pending confirmation (§11); implementation follows §9 on `feat/bsplines`, which stays off main until §8.1's required criteria are met
area: forcings / IR / compiler autodiff / PGAS+NUTS / fit config
related:
  - archive/pre-alpha/2026-05-12-periodic-bspline-algorithm.md (the periodic spline's algorithm and oracles)
  - 2026-10-26-parameter-roles.md (accepted; §7 here states what it needs from it)
  - 2026-10-24-remove-pgas-tempering-ode-nuts-and-correlated-pmmh.md (why a new inference capability lands with a measurement note)
---

# Spline forcings whose coefficients the samplers can move

All code references are against `origin/main` at `e6867779`. A claim marked
_verified_ was read at the cited line. _Measured_ means the command shown was
run with the `camdlc` built from that commit (`make build-ocaml` in this
worktree). _Inferred_ means reasoned from the code and not executed.

## 1. Learning a rate that varies through time needs three pieces camdl lacks

An outbreak fit often needs a transmission rate that changes through time:
behaviour, control measures and depletion of high-contact groups all move it,
and the change is smooth on the time scale of weeks rather than a step on a
known date. The standard flexible model for such a curve is a **B-spline**: a
weighted sum of K fixed, smooth, local bump functions (the **basis**), whose
weights (the **coefficients**) are estimated. On the log scale, β(t) = exp(Σₖ cₖ
Bₖ(t)) is positive by construction.

camdl has a periodic B-spline forcing, but a fit cannot estimate its
coefficients, and there is no non-periodic B-spline at all. Three pieces are
missing, and this proposal designs each:

1. **Live coefficients.** A coefficient that names a parameter must be evaluated
   at the parameter's current value at every step, as the Fourier forcing's
   coefficients already are (§3, §4).
2. **Gradients.** PGAS with NUTS parameter updates needs ∂β(t)/∂cₖ. For a
   B-spline that derivative is the basis function itself, Bₖ(t); the compiler
   must emit it, and the runtime must evaluate it with the same code that
   evaluates the forcing (§5).
3. **A smoothness prior.** With 8–20 coefficients and an outbreak's worth of
   data, independent coefficients overfit and mix slowly. The standard remedy is
   a penalized spline: a Gaussian prior that makes a curve unlikely in
   proportion to how much it bends, with scale τ (§6).

The design adds one forcing kind (`bspline`, cubic and clamped by default,
uniform or non-uniform knots, held outside its window), one IR expression node
(`forcing_basis`, the derivative), a `link = log` option, and a coefficient
form, `smooth(...)`, that places the integrated-squared-curvature prior on the
coefficients and samples it in a non-centred, whitened parameterization built
from ordinary per-parameter priors. No high-risk inference file changes (§5.3,
§6.6).

## 2. Terms

- **Basis function** Bₖ(t): a piecewise polynomial of **degree** p (cubic, p =
  3, by default), nonzero on at most p + 1 adjacent knot intervals.
- **Knots**: the times where the polynomial pieces join. A **knot vector** lists
  them with multiplicity. Between distinct knots the spline is a polynomial; at
  a simple knot it has p − 1 continuous derivatives (C² for cubic).
- **Window** [a, b]: the time interval the basis covers.
- **Clamped** knot vector: a and b each repeated p + 1 times, interior knots
  simple. The spline then equals its first coefficient at a and its last at b.
- **Span**: the knot interval [u_μ, u_μ₊₁) containing t; only the p + 1 basis
  functions B_{μ−p}, …, B_μ are nonzero there.
- **Partition of unity**: Σₖ Bₖ(t) = 1 for every t in the window. Consequence: a
  constant added to every coefficient adds that constant to the spline.
- **Link**: the map from the linear combination g(t) = Σ cₖBₖ(t) to the forcing
  value — identity (f = g) or log (f = exp g).
- **Curvature penalty** Ω: the K × K matrix with Ω_ij = ∫ B_i''B_j'' dt, so that
  wᵀΩw = ∫ g''(t)² dt for coefficients w. The smoothness prior is Gaussian with
  precision Ω/τ²; **τ** is its scale.
- **Greville abscissae** ξₖ: the knot averages (u_{k+1} + … + u_{k+p})/p.
  Coefficients ξ reproduce the line g(t) = t exactly.
- **Centred** parameterization: the sampler moves the coefficients w and τ.
  **Non-centred**: the sampler moves standardized modes zᵢ ~ N(0, 1) and τ, and
  w is computed from them.
- **PGAS**: particle Gibbs with ancestor sampling, camdl's default Bayesian
  method (Lindsten, Jordan & Schön 2014). It alternates a θ-update given the
  latent trajectory X with a refresh of X given θ. The θ-update uses NUTS
  (Hoffman & Gelman 2014) when the compiler has emitted gradients.
- **IF2**: iterated filtering, camdl's maximum-likelihood method (Ionides et al.
  2015). It reads no prior.
- **ESS**: effective sample size; **R̂**: the rank-normalized split R-hat of
  Vehtari et al. (2021), a measure of agreement between chains.

## 3. Today a spline coefficient cannot be a parameter, and the workarounds are worse

**The IR already carries spline coefficients as expressions.**
`PeriodicSpline { period: Expr, n_basis: u32, degree: u32, coefs: Vec<Expr> }`
(verified, `rust/crates/ir/src/time_func.rs:57-62`). Nothing in the IR stops a
coefficient from naming a parameter.

**The runtime freezes them, and the compiler refuses them.**
`CompiledModel::new` evaluates every periodic-spline coefficient to `f64`
through `eval_structural`, which refuses any expression that names a parameter
(verified, `rust/crates/sim/src/compiled_model.rs:1872-1898`, `:725-745`). The
compiler refuses first, with E600. Measured, on a model with
`c[knot] : real 'ratio` over `dimensions { knot = [k1, …, k6] }` and
`coefs = [c[k1], …, c[k6]]`:

```
$ camdlc a.camdl -o a.ir.json
error[E600]: transition 'infection': forcing `logb`: parameter 'c_k1' drives the
periodic_spline forcing coefficient, which is structural data — …
```

The refusal comes from the autodiff pass, which classifies a parameter in a
`Piecewise`, `Interpolated` or `PeriodicSpline` coefficient as `Unsupported`
with code `URStructuralForcing` (verified, `ocaml/lib/ir/autodiff.ml:357-372`);
`differentiate_rate` turns that into an error (`:800-815`). The same probe with
constant coefficients compiles, and an indexed parameter over a dimension that
no compartment is stratified by expands to scalar cells `c_k1 … c_k6` (measured:
`grep -o '"c_k1"' b.ir.json` matches).

**The Fourier forcing is the pattern to copy.** Its coefficients compile to live
`ResolvedExpr` (verified, `compiled_model.rs:1853-1870`) evaluated per call in
`eval_forcing` (`rust/crates/sim/src/propensity.rs:486-496`), and its gradient
is emitted by differentiating a closed form (`autodiff.ml:208-231`, `:343`).

**The derivative IR has no node for a basis function.** Every other forcing
derivative is expressed by rebuilding the forcing as ordinary arithmetic over
`Time` (`sinusoidal_closed`, `fourier_closed`). A B-spline's basis has no such
compact closed form: it is a different polynomial on each knot interval.

**The existing periodic evaluator allocates and recurses on every call.**
`eval_periodic_bspline` builds a knot `Vec` of n_basis + 3p + 2 entries and a
basis `Vec` of n_basis + 2p + 1 entries per call, and fills the latter with a
naive recursive Cox–de Boor evaluation of every basis function, including the
ones that are zero at t (verified, `rust/crates/sim/src/periodic_bspline.rs:95`,
`:112-115`, recursion `:41-66`). For n_basis = 6, p = 3 that is 13 basis
functions × 15 recursive calls each (inferred). The module's doc claims
O(n_basis · degree²); the recursion is O(n_basis · 2^degree).

**A positive rate cannot be written as exp of a forcing without a multiplier.**
Measured, same probe: `S --> I @ exp(logb) * S * I / N` with a `'ratio` forcing
is `E300` (the rate lacks a 1/time dimension); with a `'per_day` forcing it is
both `E300` and `E301` (the argument of `exp` must be dimensionless). The rate
must therefore be `beta0 * exp(logb)`, and an estimated `beta0` is exactly
non-identified against a common shift of the coefficients (§4.4).

**An unbounded `real` coefficient gets the wrong search box today, and that is
being fixed separately on main.** With no `in [lo, hi]` and no fit-config
bounds, the box falls back to `[0, ∞)` for every kind but `probability`
(verified, `rust/crates/cli/src/fit/runner.rs:1265-1267`; `kind_support` returns
`None` for `real`, `params_resolver.rs:118-126`), and a box with an infinite end
gets no transform (`runner.rs:1167-1173`). A log-scale coefficient is usually
negative. Its auto `rw_sd` for IF2 is then derived from an infinite box. The
correction to (−∞, ∞) is in progress on main; this design does not depend on the
fallback either way, because the validation declares explicit finite bounds on
every coefficient family (§7.2).

**PGAS refuses hierarchical priors outright** (verified,
`rust/crates/sim/src/inference/pgas.rs:3775-3784`): the NUTS gradient of a
hierarchical leaf is stubbed to −∞ (`prior_log_density_and_grad_z`,
`:3557-3570`). A random-walk prior written as a chain of hierarchical leaves,
`c2 ~ normal(mu = c1, sigma = tau)`, therefore cannot run under PGAS.

`ir/schema.json`, which `.claude/rules/ir-schema.md` calls the contract, omits
`fourier` and `periodic_spline` and is read by no test (verified,
`ir/schema.json:374-437`); that is being fixed separately on main, and Stage 2
adds the new kind and node to whatever form it takes there. The periodic-spline
scipy and pomp fixtures have generator scripts
(`scripts/gen_periodic_bspline_*_fixture.*`) but no make target (verified:
`grep -n bspline Makefile` has no match); Stage 2 adds one (§9).

## 4. One evaluator serves both spline kinds, and the new kind is clamped and held

### 4.1 The evaluation core computes only the p + 1 nonzero basis functions

A new pure module, `rust/crates/sim/src/bspline.rs`, holds the arithmetic both
spline kinds share. It has no knowledge of expressions or parameters.

```rust
// rust/crates/sim/src/bspline.rs (new)

/// The largest degree a spline forcing may declare. Fixed-size workspaces
/// below; the compiler refuses a larger degree.
pub const MAX_DEGREE: usize = 5;

/// A clamped knot vector over the window [a, b], stored by its distinct knots.
#[derive(Debug, Clone, PartialEq)]
pub enum Knots {
    /// n equal intervals of width h = (b − a)/n. Span lookup is O(1).
    Uniform { a: f64, h: f64, n: usize },
    /// Strictly increasing breakpoints a = x₀ < x₁ < … < x_n = b.
    /// Span lookup is a binary search, O(log n).
    Explicit(Vec<f64>),
}

/// The p + 1 basis values that can be nonzero at one time.
#[derive(Debug, Clone, Copy)]
pub struct Span {
    /// Index of the first nonzero basis function: B_first … B_{first+p}.
    pub first: usize,
    pub values: [f64; MAX_DEGREE + 1],
}

impl Knots {
    pub fn window(&self) -> (f64, f64);
    pub fn n_intervals(&self) -> usize;
    /// The interval index m ∈ [0, n) holding t, for t already clamped to the
    /// window; t = b belongs to the last interval.
    fn interval(&self, t: f64) -> usize;
    /// Knot u_i of the clamped vector (length n + 2p + 1), computed on demand:
    /// a for i ≤ p, b for i ≥ n + p, the interior breakpoints between.
    fn knot(&self, i: usize, p: usize) -> f64;
}

/// Nonzero basis functions at t (Piegl & Tiller 1997, Algorithm A2.2).
pub fn span(knots: &Knots, p: usize, t: f64) -> Span;
/// One basis function, Bₖ(t): zero without further work when k is outside
/// [first, first + p].
pub fn basis(knots: &Knots, p: usize, t: f64, k: usize) -> f64;
```

**The algorithm.** For t in the window, find the interval m, set the span index
μ = m + p in the clamped vector, and run the Cox–de Boor triangle (Cox 1972; de
Boor 1972) in the stable form of Piegl & Tiller (1997, Algorithm A2.2):

```
N[0] = 1
for j in 1..=p:
    left[j]  = t − u[μ+1−j]
    right[j] = u[μ+j] − t
    saved = 0
    for r in 0..j:
        temp  = N[r] / (right[r+1] + left[j−r])
        N[r]  = saved + right[r+1] · temp
        saved = left[j−r] · temp
    N[j] = saved
# N[r] = B_{μ−p+r}(t), r = 0..p
```

The forcing value is then g(t) = Σ_{r=0}^{p} c_{μ−p+r} N[r], so a call evaluates
exactly p + 1 coefficient expressions, not K.

**Cost per call** (inferred, counted from the pseudocode): span lookup O(1) for
uniform knots, ⌈log₂ n⌉ comparisons for explicit; the triangle p(p+1)/2 inner
iterations of one division and four multiply-adds — 6 iterations for a cubic;
p + 1 coefficient evaluations and multiply-adds. No allocation: `left`, `right`
and `N` are `[f64; MAX_DEGREE + 1]` on the stack. For a cubic that is roughly 40
flops plus four coefficient reads, against about 200 recursive calls and two
heap allocations in today's periodic evaluator. An independent check, not rerun
here, measured A2.2 with the periodic fold map at 13.0 ns per call against 279.9
ns for today's evaluator, agreeing to 6.7·10⁻¹⁶; Stage 1 commits a dedicated
before/after microbenchmark and pastes its result (§9).

**Numerical stability.** Every step of the triangle combines nonnegative
quantities with nonnegative weights — `left` and `right` are nonnegative within
the span, and the N[r] stay nonnegative and sum to one at every level — so there
is no subtractive cancellation (de Boor 1978, ch. X). This is the reason to
prefer it over evaluating each interval's polynomial in the power basis, whose
coefficients in absolute model time (t ≈ 10³ days in an anchored model) would
cancel badly. Denominators `right[r+1] + left[j−r]` are differences of distinct
knots, positive because breakpoints are strictly increasing; the compiler and
the runtime both refuse breakpoints closer than 10⁻⁹·(b − a) (§4.3). The test
`partition_of_unity` asserts |Σ N[r] − 1| ≤ 10⁻¹⁴ over 10⁴ points per knot
layout.

### 4.2 Periodic-spline coefficients become live without an IR change

The compiled form of `periodic_spline` keeps its period and integers as data and
turns its coefficients into live expressions:

```rust
// compiled_model.rs: before (verified, :50)
PeriodicSpline { period: f64, n_basis: u32, degree: u32, coefs: Vec<f64> },

// after
PeriodicSpline(CompiledPeriodicSpline),

pub struct CompiledPeriodicSpline {
    pub period: f64,               // structural: a parameter here stays refused
    pub n_basis: usize,
    pub degree: usize,
    pub coefs: Vec<ResolvedExpr>,  // live, as Fourier's harmonics
    pub link: SplineLink,          // §4.4; Identity until Stage 2 adds the field
}
```

The evaluation reuses `bspline::span` on the uniform cardinal knots of the
existing construction. With x = t mod period, dx = period/n_basis and j =
⌊x/dx⌋, the nonzero unfolded basis functions are indices j … j + p; folding maps
index i to i mod n_basis, and the existing centring shift s = (p − 1) div 2 maps
folded index i to coefficient (i − s) mod n_basis (verified against the current
construction, `periodic_bspline.rs:96-131`). So

f(t) = Σ_{r=0}^{p} N_r(x/dx − j) · c_{(j + r − s) mod n_basis},

and p < n_basis makes the p + 1 indices distinct. The scipy and pomp fixtures,
unchanged, are the acceptance test at 10⁻¹² (verified tolerance,
`rust/crates/sim/tests/periodic_bspline_oracle.rs`).

`first_param_read`, which the PGAS and IF2 refusals use to ask whether an
expression moves with an estimated parameter, returns `None` for every periodic
spline today (verified, `compiled_model.rs:1442-1446`). With live coefficients
it must read them, exactly as its Fourier arm does; otherwise an intervention
amount that reads a spline-coefficient forcing would escape the gh#986 refusal
in `refuse_unscored_parameters` (`pgas.rs:3683`). The arm is in
`compiled_model.rs`, not in a high-risk file.

Floating-point rounding of `periodic_spline` values changes in the last bits,
because the summation order differs. No stored run is served stale because of
it: the engine version (package version plus git hash) is folded into the model
level of every run identity (verified,
`rust/crates/runid/src/inputs.rs:163-168`, `:292`; it reaches the digest through
`cli/src/resolve.rs:206` and `fit/cas.rs:514`, reproduced by an independent
check), so every commit re-keys stored runs and a recomputation is a cache miss.

### 4.3 The `bspline` kind: a clamped basis over a window, held constant outside it

**DSL.** Two knot forms, mutually exclusive:

```camdl
forcing {
  # Uniform knots: n_basis − degree equal intervals over the window.
  log_beta : bspline 'per_day {
    window  = [date("2025-08-04"), last_obs - 2 'weeks]   # end before the reporting delay
    n_basis = 8
    degree  = 3                       # optional, default 3
    link    = log                     # optional: identity (default) | log
    coefs   = [c[k1], c[k2], c[k3], c[k4], c[k5], c[k6], c[k7], c[k8]]
  }

  # Explicit breakpoints (first and last are the window); n_basis is implied:
  # len(breakpoints) − 1 + degree.
  reporting_drift : bspline 'ratio {
    breakpoints = [date("2025-08-04"), date("2025-09-15"), last_obs - 2 'weeks, last_obs]
    coefs       = [d1, d2, d3, d4, d5, d6]
  }
}
```

Window ends and breakpoints are classified by the same reader as `piecewise`
`breakpoints` (`get_breakpoints_list` and `classify_anchor`, verified,
`ocaml/lib/compiler/expander.ml:6674-6693`, `:5342`): a `date(...)` under an
`origin`, a duration literal, `first_obs`/`last_obs` ± a constant duration, or a
constant. `lag` applies as to every kind. Unknown keys are E409 through the
existing per-kind key list (`expander.ml:6883-6902`). Shape errors —
`n_basis ≤ degree`, a coefficient count that disagrees with the knots,
`degree > 5`, a non-integer `n_basis`, both or neither knot forms — are E408
with the messages `periodic_spline` uses (`expander.ml:7172-7224`).

**IR.** One new kind; `link` also joins `periodic_spline` (§4.4).

```ocaml
(* ocaml/lib/ir/ir.ml *)
type spline_link = Link_identity | Link_log
type bspline_knots =
  | Uniform of { lower : expr; upper : expr; n_intervals : int }
  | Explicit of expr list            (* strictly increasing, at least 2 *)
type bspline = {
  knots  : bspline_knots;
  degree : int;  coefs  : spline_coefs;             (* Explicit of expr list | Smooth …, §6.4;
                                        length n_intervals + degree *)
  link   : spline_link;
}
(* time_func_kind gains:  | BSpline of bspline
   periodic_spline gains: link : spline_link *)
```

```rust
// rust/crates/ir/src/time_func.rs
#[serde(rename_all = "snake_case")]
pub enum SplineLink { Identity, Log }

#[serde(rename_all = "snake_case")]
pub enum BSplineKnots {
    Uniform { lower: Expr, upper: Expr, n_intervals: u32 },
    Explicit { breakpoints: Vec<Expr> },
}

pub struct BSpline {
    pub knots: BSplineKnots,
    pub degree: u32,    pub coefs: SplineCoefs,            // §6.4
    #[serde(default, skip_serializing_if = "SplineLink::is_identity")]
    pub link: SplineLink,
}
// TimeFuncKind gains BSpline(BSpline); PeriodicSpline gains the same `link` field.
```

The uniform form is a variant rather than a list of computed breakpoints because
its window end may be an observation anchor: the compiler cannot compute a +
i·(b − a)/n when b is `last_obs`, and the runtime substitutes anchors only into
expressions (`rust/crates/cli/src/obs_anchor.rs:172-233`, verified; its
`substitute_expr` recurses through arithmetic, `:239-278`). Carrying
`Uniform { lower, upper, n_intervals }` keeps the anchor a single node and gives
the evaluator its O(1) span lookup. Window ends and breakpoints are structural:
a parameter there is `Unsupported` (E600), as for `piecewise`.

**Compiled form and runtime checks.** `CompiledModel::new` evaluates the knot
expressions with `eval_structural` and builds `bspline::Knots`; coefficients
resolve live. It refuses, naming the forcing: a < b not holding, non-finite
knots, breakpoints not strictly increasing or closer than 10⁻⁹·(b − a), and a
coefficient count that disagrees with the knots. `obs_anchor`'s exhaustive
`forcing_exprs`/`forcing_exprs_mut` gain the kind, and
`validate_after_substitution` gains the same ordering check it applies to
`piecewise` (`obs_anchor.rs:313-355`), because an anchored window can invert
under one data vintage and not another.

**Ends: clamped** (D3). The knot vector repeats a and b p + 1 times. Every basis
function lies inside the window, f(a) = c₁ and f(b) = c_K exactly, and explicit
non-uniform breakpoints need no extension beyond the data. The smoothness prior
penalizes the curvature of the curve itself (§6.1), so the uneven Greville
spacing that clamping creates at the ends needs no special treatment.

**Outside the window: hold** (D4). For t < a the forcing takes its value at a;
for t > b, its value at b. The evaluator clamps t to [a, b] before the span
lookup, and `forcing_basis` (§5.1) clamps identically, so ∂f/∂cₖ beyond b is
Bₖ(b) and the value and gradient cannot disagree. Three reasons:

- A cubic extrapolated past b diverges, and exp of it diverges faster: a
  forecast would inherit an explosion no data support.
- Holding is the statement "transmission stays where the data last put it",
  which is also what a `piecewise` forcing does past its last breakpoint and a
  linear `interpolated` one past its last knot (verified,
  `propensity.rs:364-373` and the doc at `:375-376`), and it composes with the
  `last_obs` scenario forks the Ebola models already use (`ramp_control` in
  `ebola-bdbv-camdl/models/bvd_province_reff.camdl:148`). A modeller who wants a
  trend after b states it as a scenario.
- The forcing stays continuous at a and b; only its derivative in t jumps, which
  no rate consumes.

**What a run past the window computes** (D4). For a `bspline` forcing with
window [a, b], lag ℓ (0 when absent) and link h, the value at simulation time t
is f(t) = h(g(clamp(t − ℓ, a, b))). For t − ℓ > b this is the constant h(g(b)):
the same in every substep, particle and replicate of a forecast, carrying only
the posterior uncertainty of g(b). A scenario fork multiplies or replaces it as
it does any forcing. The same holds before a with g(a). A window that does not
cover the run is allowed.

**The notice.** Every command that runs the model, after resolving anchors
(where `obs_anchor::report` already prints resolved knots, `obs_anchor.rs:422`),
compares each `bspline` forcing's resolved window end b with the run's resolved
horizon t_end. If t_end − ℓ > b it adds one info notice through the CLI notice
system (`rust/crates/cli/src/notice.rs`; notices are recorded with the run and
never hashed), code `spline-held-past-window`:

```
info: forcing `log_beta` is held at its value at t = 168 from t = 168 to the horizon t = 220; its window ends at last_obs - 2 'weeks
```

(with the calendar date appended under an `origin`, as resolved anchors are). It
compares two resolved numbers in the IR, so no seam from the simulator to the
CLI is needed, and `Notices` emits it once per job rather than once per cell.
With the recommended window it fires on fits as well as forecasts, which is
correct: the last two weeks of data are fitted under the held value. The
language spec's `bspline` section, added in Stage 2 by hand, documents the
definition and the notice.

**End the window before the reporting delay.** The last basis function
integrates to a quarter of an interior one (cubic, clamped), and infections near
b are mostly reported after the data end: with incubation near 10 days and h =
20 days, c_K is largely the prior's extrapolation, yet it is the value held for
every forecast. The language spec section Stage 2 adds therefore recommends
`window = [..., last_obs - d]`, d the infection-to-report delay (two weeks for
Ebola), so the held value is a coefficient the data inform; the synthetic study
does this (§8.4).

### 4.4 `link = log` gives a positive rate with the right dimension and no intercept

With `link = log` the forcing is f(t) = s · exp(g(t)), g = Σ cₖBₖ, where s is
the scale of the declared unit in model time (`'per_week` under
`time_unit = 'days` gives s = 1/7). The coefficients are dimensionless
log-values; the forcing carries the declared dimension. That resolves the
measured E300/E301 of §3: the rate is written `log_beta * S * I / N`, with no
multiplier.

The unit scale is applied through the coefficients, as the expander does today
for `periodic_spline` (`expander.ml:7260-7263`), but additively: each
coefficient becomes cₖ + ln s. By the partition of unity, Σ (cₖ + ln s) Bₖ = g +
ln s, so the value is s·exp(g) exactly, inside and outside the window (where the
held value inherits the same identity). The identity link keeps today's
multiplicative scaling. Dimension checking requires dimensionless coefficients
under `link = log` (E301, the existing code for a dimensioned `exp` argument).

A user may still write `exp(logb)` with an identity-link `'ratio` forcing and a
unit-carrying typed constant (`let per_day : rate = 1.0`). The gradient is
correct either way (§5.2), and both forms are tested.

**Identifiability with an intercept.** Because Σ Bₖ = 1, the rate
`beta0 * log_beta` with `link = log` is unchanged by (beta0, c) → (beta0·e^δ, c
− δ): an estimated `beta0` is exactly non-identified. The recommendation is no
separate intercept: the coefficients carry the level. A modeller who needs an
interpretable baseline (an R₀ with a prior, say) fixes the redundancy with a
corner constraint written in the coefficient list,
`coefs = [0.0, c[k2], …, c[kK]]`, so that `beta0` is the rate at a and the
coefficients are log-ratios to it; or with a sum-to-zero constraint
`coefs = [c[k1], …, c[kK−1], -(c[k1] + … + c[kK−1])]`. Both are plain
coefficient expressions and need no new surface. The smoothness constructor of
§6 carries its own level parameter and is not combined with an intercept.

## 5. The gradient is a basis-function node, emitted by the compiler and evaluated by the forcing's own code

### 5.1 `forcing_basis(f, k)` is the derivative node

∂g(t)/∂cₖ = Bₖ(t). The IR gains one expression node meaning "basis function k of
spline forcing f, at the forcing's evaluation time":

```ocaml
(* ocaml/lib/ir/ir.ml, expr *)
| ForcingBasis of { forcing : string; index : int }   (* 0-based *)
```

```rust
// rust/crates/ir/src/expr.rs
pub struct ForcingBasisRef { pub name: String, pub index: u32 }   // gains `space`, §6.4
pub struct ForcingBasisWrap { pub forcing_basis: ForcingBasisRef }
// Expr gains ForcingBasis(ForcingBasisWrap):
//   {"forcing_basis": {"name": "log_beta", "index": 2}}

// rust/crates/sim/src/resolved_expr.rs, ResolvedExpr gains
ForcingBasis { tf: usize, index: u32 },
```

A distinct variant, rather than an optional `index` on the existing `time_func`
node, so every exhaustive match in both languages must classify it: an optional
field would let each existing `TimeFunc` arm silently treat a basis as the
forcing's value.

**Evaluation** applies the forcing's `lag` exactly as the `TimeFunc` arm does
(`resolved_expr.rs:653-664`, verified), clamps t to the window (B-spline) or
wraps it (periodic), and calls `bspline::basis`. It is one code path with the
value: the forcing value and its derivative read the same knots through the same
span function, so they cannot diverge — the defect class of gh#197/gh#200, a
value and a gradient computed by two implementations. `resolve_expr` refuses a
`forcing_basis` whose target is not a spline kind or whose index is out of
range, so the evaluator's remaining arm is unreachable; it returns NaN, the
strict-mode sentinel the propensity guard already catches, rather than
panicking.

**Where it may appear.** Only the compiler writes it, only inside a
`DerivEntry::Grad` (a `rate_grad`, an observation `*_grad`, a `sigma_sq_grad`).
`ir::validate` refuses it anywhere else, and refuses an unknown forcing or
index. The DSL has no syntax for it.

### 5.2 The autodiff rule sums coefficient derivatives against the basis

In `differentiate`'s `TimeFunc` arm, after the existing `lag` guard (which stays
first, `autodiff.ml:332-342`), the two spline kinds get a real derivative:

```ocaml
| Some ({ kind = BSpline s; _ }) ->
    spline_deriv fname s.coefs s.link (bspline_structural_exprs s)
| Some ({ kind = PeriodicSpline ps; _ }) ->
    spline_deriv fname ps.coefs ps.link [ ps.period ]

(* ∂f/∂θ = link′ · Σ_j (∂c_j/∂θ) · B_j *)
let spline_deriv fname coefs link structural =
  if List.exists (mentions param) structural then
    Unsupported { code = URStructuralForcing; … }          (* knots, window, period *)
  else
    let term j c = map1 (d c) (fun dc ->
      BinOp { op = Mul; left = dc;
              right = ForcingBasis { forcing = fname; index = j } }) in
    let sum = List.mapi term coefs |> fold map2 Add (Known (Const 0.0)) in
    match link with
    | Link_identity -> sum
    | Link_log -> map1 sum (fun s -> BinOp { op = Mul; left = TimeFunc fname; right = s })
```

The rule differentiates each coefficient expression, so a coefficient that is
arithmetic over parameters — a corner constraint, a sum-to-zero, or the explicit
expressions over parameters — gets the chain rule with no special case. The
`smooth` form has its own rule (§6.4). `simplify_fixpoint` drops the terms whose
∂cⱼ/∂θ folds to zero, so for cⱼ = `Param θ` the emitted gradient is the single
node `forcing_basis(f, j)`.

**Through `exp`.** Under `link = log`, ∂f/∂cⱼ = f(t)·Bⱼ(t), emitted as
`time_func(f) * forcing_basis(f, j)`. When the user writes `exp(logb)` with an
identity link, the existing `Exp` rule (`autodiff.ml:575-576`, verified)
produces exp(g)·Bⱼ. Both are finite-difference tested (§8.1, R2).

`forcing_coeff_exprs` (`autodiff.ml:162-170`) splits into the coefficient
expressions and the structural ones for the two spline kinds, so that
`forcing_mentions` no longer conflates them. With respect to state or the
projection a `forcing_basis` is zero, like every forcing. It is never itself
differentiated with respect to a parameter, because gradients are not
differentiated again; the rule is nonetheless total — zero, or `Omitted URLag`
when the parameter drives the lag.

E600 for spline coefficients is lifted by this rule; E600 for a parameter in a
knot, window end or period remains.

### 5.3 Every exhaustive match gains an arm, and none is in a high-risk file

The new expression variant must be classified wherever `TimeFunc` is (verified
by `grep -rn "Expr::TimeFunc"` and `"ResolvedExpr::TimeFunc"` outside tests): in
Rust, `ir` (`expr.rs`, `validate.rs`, `init_order.rs`), `runid/ir_hash.rs` (a
new tag appended, as for `UnsupportedReason`, so existing hashes keep their
positions), `cli` (`eval.rs`, `fit/coeff_guard.rs`, `obs_anchor.rs`), `sim`
(`compiled_model.rs`, `flat_eval.rs`, `propensity.rs`, `resolved_expr.rs` ×4,
`inference/gradient_capability.rs`, `inference/hierarchical.rs`,
`inference/multi_stream_obs.rs`) and `benches/eval_ab.rs`; in OCaml,
`ir/{ir,serde,validate,lint,lineage,expr_analysis,licm,dimcheck,autodiff,init_order,constant_fold}.ml`
and `compiler/{expander,inspect,pp_expr}.ml`. The loop-invariant hoisting pass
(LICM, gh#272) treats it as time-varying, like `TimeFunc`;
`per_eval_staging_violation` refuses it in a per-eval body.

**No high-risk file changes in Stages 1–5.** The six high-risk files contain no
match on `Expr` or `ResolvedExpr` variants (measured:
`grep -c 'Expr::\|TimeFunc\|CompiledTimeFuncKind'` returns 0 for `pgas.rs`,
`pgas_grad.rs`, `obs_loglik.rs`, `if2.rs` and `particle_filter.rs`; the 28 hits
in `obs_model.rs` are likelihood variants and `ResolvedExpr::Const` in tests).
They consume gradients only through `eval_resolved` and `eval_emitted_grad`
(`pgas_grad.rs:16`, `obs_model.rs:12`, `:460`), which gain the arm in
`resolved_expr.rs`. The Rust compiler is the check: if any high-risk file held
an exhaustive match, adding the variant would fail to build it.

### 5.4 The NUTS preflight admits the coefficients without change

`nuts_active` requires a non-empty `rate_grad` (verified, `pgas.rs:3662-3664`).
The `run_pgas` preflight refuses an estimated parameter only if some gradient
map holds `Unsupported` for it (`pgas.rs:4150-4185`, via
`gradient_capability::scan_unsupported_gradients`). With §5.2 a coefficient
parameter carries `Grad`, so it is admitted and differentiated through the
existing `complete_data_loglik_grad`. `coeff_guard`'s initial-condition-only
domain (`coeff_guard.rs:324`) needs the new kind in its exhaustive
`collect_forcing`; a spline reached only through `init {}` still has no gradient
and is still refused, correctly.

### 5.5 Gradient cost is O(K) cheap zeros plus p + 1 basis evaluations per parameter

The NUTS θ-update evaluates every emitted `rate_grad` entry at every substep of
the reference trajectory. With K coefficient parameters a rate carries K
`forcing_basis` nodes; for each, `bspline::basis` finds the span in O(1)
(uniform) and returns zero immediately for the K − p − 1 basis functions not
supported there, so the per-substep cost is O(K + (p + 1)·p²) basis work per
referencing transition (inferred). A per-(forcing, t) cache of the span is the
obvious optimization and is not built until a profile asks for it.

### 5.6 The IR bump, its goldens, and the `link` hash

Three schema additions — the `bspline` kind with `link` (Stage 2),
`forcing_basis` (Stage 3), and `SplineCoefs::Smooth` with the basis spaces
(Stage 4) — share one pending `ir/VERSION` bump on the branch, made in Stage 2
(0.40 → 0.41 at today's main; renumbered at rebase, §10). The envelope embeds
the version in every serialized IR file (verified:
`ocaml/golden/sir_basic.ir.json` begins `{"ir_version":"0.40"`), and a bump
moves about 118 of them (counted by an independent check): 55 in
`ocaml/golden/`, 17 in the frozen `ir/golden/` set (not regenerated by
`make update-golden`; hand-edited at the last bump, `25609219`), 19 under
`rust/crates/`, 23 under `tests/fixtures/`, 3 under `tests/external/` and 1
under `docs/dev/`. Stage 2's atomic commit moves them all, with no change
besides the version string for a model without a spline (`link` is omitted when
identity). New fixtures add goldens: `bspline_uniform.camdl`,
`bspline_explicit_anchored.camdl`, `bspline_loglink.camdl`, and in Stage 3
`bspline_param_coefs.camdl` and `periodic_spline_param_coefs.camdl`, whose
`rate_grad` carries `forcing_basis` nodes. Each new golden carries
first-scenario parameters and a trajectory baseline.

**`link` must be hashed, and a missing field must not compile.** The
run-identity hash reads `PeriodicSpline` by field access, not by destructuring
(verified, `rust/crates/runid/src/ir_hash.rs:840-846`), so a new `link` field
would compile without being hashed. Under a `'ratio` unit the scale is 1 and ln
s = 0, so the coefficient rewrite of §4.4 leaves both links' coefficients
identical: two models that differ only in `link` would share a `run_id`, and one
would be served the other's results. Stage 2 hashes `link` in both arms; the
`PeriodicSpline` arm and the new `BSpline` arm destructure exhaustively
(`let PeriodicSpline { period, n_basis, degree, coefs, link } = s;`), so a
future field is a compile error there; and a red→green test asserts that two
models differing only in `link` hash differently.

## 6. The smoothness prior penalizes integrated squared curvature, and NUTS samples it whitened

### 6.1 The penalty is ∫ f''(t)² dt, computed exactly from the knots

For a spline g(t) = Σⱼ wⱼ Bⱼ(t) with coefficient vector w, the integrated
squared curvature over the window is a quadratic form,

∫ₐᵇ g''(t)² dt = wᵀΩw, Ω_ij = ∫ₐᵇ B_i''(t) B_j''(t) dt.

This is the penalty of O'Sullivan penalized splines (O'Sullivan 1986; Wand &
Ormerod 2008, who give its computation and its relation to the difference
penalty of Eilers & Marx 1996). Its prior form is the Gaussian with precision
Ω/τ²: a curve is a priori unlikely in proportion to how much it bends, measured
in the units of the curve and of time, whatever the knots.

**Ω is exact with a fixed quadrature rule.** On each knot interval B_i'' is a
polynomial of degree p − 2, so B_i''B_j'' has degree 2p − 4, and (p − 1)-point
Gauss–Legendre quadrature on that interval (exact to degree 2p − 3) integrates
it exactly. For a cubic that is 2 points per interval. Ω is assembled interval
by interval from the p + 1 nonzero second derivatives, obtained from the same
span routine as the values (Piegl & Tiller 1997, Algorithm A2.3; the algorithm
number is cited from memory, and Stage 1 checks it against the book).
Properties, each checked numerically with a scratch scipy script (cubic,
clamped, window [0, 168], K = 12, uniform and the irregular breakpoints 0, 10,
25, 30, 60, 90, 100, 130, 150, 168):

- 2-point and 10-point Gauss–Legendre agree to 2·10⁻¹⁵ of max|Ω|;
- Ω is banded with bandwidth p = 3;
- Ω has rank K − 2 (10), and Ω·1 and Ω·ξ vanish to 2·10⁻¹⁶ of max|Ω| (scaled by
  the window length for ξ), for both knot sets.

**Its null space is exactly the linear functions of t, for any knot spacing.**
g'' ≡ 0 on [a, b] if and only if g is linear in t. B-splines are linearly
independent, the constant function has coefficients 1 (partition of unity), and
t has coefficients ξ, the **Greville abscissae** ξₖ = (u_{k+1} + … + u_{k+p})/p
(the knot averages; Σ ξⱼBⱼ(t) = t, the linear-precision property, de Boor 1978).
So null(Ω) = span{1, ξ} exactly — no end weights, no assumption of equal
spacing. Measured on both knot sets above: Σ ξⱼBⱼ(t) = t to 6·10⁻¹⁴ and Σ Bⱼ(t)
= 1 to 3·10⁻¹⁶ over 1,001 points.

### 6.2 Against the two alternatives, the curvature penalty is the only one exact for irregular knots

|                   | Curvature penalty Ω (chosen)                                                                                                       | Greville divided differences                             | Unclamped Eilers–Marx, index differences                                        |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Null space        | exactly the linear functions of t, any knots                                                                                       | linear in t, any knots                                   | linear in t only for equal spacing                                              |
| End behaviour     | no special case: the penalty is on g itself                                                                                        | end increments need spacing weights that are a heuristic | p coefficients per end supported mostly outside the data; f(a), f(b) are blends |
| Non-uniform knots | yes                                                                                                                                | yes, with the heuristic weights                          | no (refused)                                                                    |
| Meaning of τ      | dimensionless, normalized by a declared reference duration and by √(K − 2): the same at every K, time unit and data release (§6.3) | depends on the spacing h; rescaled per K                 | depends on h                                                                    |
| Precomputation    | Ω by quadrature, O(K·p²); a Householder QR and one (K − 2)-dimensional Cholesky factorization, O(K³)                               | none                                                     | none                                                                            |
| Per evaluation    | O(pK) (§6.4)                                                                                                                       | O(pK)                                                    | O(pK)                                                                           |

The precomputation runs once per compiled model, after anchors are resolved; for
K ≤ 50 it is a few hundred thousand flops (inferred). The divided-difference
construction approximates the curvature penalty in the interior; it is not
adopted because its end weights are a heuristic with no derivation, and the
curvature penalty removes the need for them at a one-off O(K³) cost.

### 6.3 The prior, its completion, and the non-centred whitened parameterization

**Units and scaling.** ∫ g''² dt has units of time⁻³, and its prior expectation
grows with the number of modes. Two fixed factors make τ a dimensionless number
with one meaning: a declared reference duration T_ref (`per = 1 'weeks` in
`smooth`, dimension-checked as a duration, converted to model time units), and
the number of non-linear modes K − 2. The prior is

p(w | τ) ∝ τ^{−(K−2)} exp(−κ² · wᵀΩw / (2τ²)), κ = √(K − 2) · T_ref^{3/2},

improper along null(Ω), completed by proper priors on the level and slope of the
linear part. Under it, E[T_ref³ ∫ g''² dt | τ] = τ²: τ² is the expected total
curvature measured in reference durations, independent of the time unit, of the
knot count and of the data release. Normalizing by the window length instead
would make the implied prior move whenever an anchored window moves: the
statistics re-review measured the equivalent tail at 21.8 for a 168-day window
against 16.6 for a 140-day one. The √(K − 2) factor is a recommendation (D3
detail, §11): without it E[T_ref³ ∫ g''²] = τ²(K − 2), so adding knots would
silently loosen the smoothness prior; with it, knot count and smoothness are
separate choices.

**Whitening by Cholesky.** At model build, with t_m = (a + b)/2:

1. N = [1, ξ − t_m·1] spans null(Ω). Q₊, an orthonormal basis (K × (K − 2)) of
   its orthogonal complement — which is range(Ω), Ω being symmetric — is the
   last K − 2 columns of the full Q of a Householder QR of N, computed with the
   columns in that fixed order, no pivoting, and the standard reflector sign
   convention (each reflector chosen to avoid cancellation, v₁ carrying the sign
   of the leading entry). Every step is a fixed sequence of floating-point
   operations on fixed inputs, so Q₊ is a deterministic function of the knots.
2. L = chol(Q₊ᵀΩQ₊), the lower Cholesky factor of a (K − 2) × (K − 2)
   positive-definite matrix, computed by the existing
   `inference/linalg.rs::cholesky_lower` (verified, `:7`).
3. M = Q₊ L⁻ᵀ (triangular solves), with MᵀΩM = I and NᵀM = 0.

The coefficients are

w = a·1 + b·(ξ − t_m·1) + (τ/κ) M z, z ~ N(0, I_{K−2}),

so the linear part of the curve is a + b(t − t_m): a is the level of log β at
mid-window and b its slope per unit time. An eigendecomposition would give the
same MMᵀ, and so the same prior — the statistics re-review measured them equal
to 9·10⁻¹³, and the scratch check here to 9·10⁻¹³ at K = 20 — but on uniform
knots Ω's eigenvalues are nearly degenerate (smallest relative gap 10⁻⁶ at K =
12, 6·10⁻¹³ at K = 20), so individual eigenvectors, and with them the meaning of
each zᵢ, would depend on rounding and on the last bits of an anchored window.
The Cholesky construction is unique, non-iterative and needs no sign rule. The
zᵢ carry no individual interpretation; only their joint law matters.

**Exact equivalence.** The map (a, b, y) → w = a·1 + b(ξ − t_m) + Q₊ y is a
linear bijection with constant Jacobian, and with y = (τ/κ) L⁻ᵀ z, κ² wᵀΩw = κ²
yᵀ(Q₊ᵀΩQ₊)y = τ² zᵀz. The density z ~ N(0, I) therefore induces on w

p(w | τ) = p_{a,b}(a(w), b(w)) · c · τ^{−(K−2)} exp(−κ² wᵀΩw/(2τ²)),

where (a(w), b(w)) are the coordinates of the orthogonal projection of w onto
null(Ω) and c is a constant: exactly the improper Gaussian with precision κ²Ω/τ²
over range(Ω), with the proper prior p_{a,b} on the null space, including the
τ^{−(K−2)} normalizer (the rank of Ω, not K). Checked numerically with the
Cholesky construction (scratch script, four knot sets): MᵀΩM = I to 2.7·10⁻¹³ at
worst; ‖NᵀM‖_max/(‖N‖₂‖M‖₂) ≤ 5.4·10⁻¹⁷; and, with the earlier eigenvector
construction, the Monte Carlo covariance of Mz over 2·10⁵ draws matched Ω⁺ to
4·10⁻³, its Monte Carlo error.

**Gradient.** In the sampler's coordinates the map is linear: ∂w/∂z = (τ/κ) M,
∂w/∂a = 1, ∂w/∂b = ξ − t_m, ∂w/∂τ = M z / κ. Through the spline, with t_c =
clamp(t − lag, a, b):

∂g/∂a = 1, ∂g/∂b = t_c − t_m, ∂g/∂zᵢ = (τ/κ) Wᵢ(t_c), ∂g/∂τ = Σᵢ zᵢ Wᵢ(t_c)/κ,

where Wᵢ(t) = Σⱼ M_jᵢ Bⱼ(t) is the i-th **whitened basis function**. The first
two use partition of unity and linear precision, so they need no basis
evaluation. Under `link = log` each is multiplied by f(t) (§5.2). Because the
engine version is part of every run identity (§4.2), any change to this
construction re-keys stored runs rather than reinterpreting their draws.

### 6.4 `smooth(...)` is a coefficient form, and the priors stay visible

```camdl
dimensions { mode = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] }  # n_basis − 2

parameters {
  # log β at mid-window (R ≈ 1.2)
  a_b     : real 'ratio ~ normal(mu = -1.9, sigma = 0.6)
  # slope of log β, per day
  b_b     : real [1/T] ~ normal(mu = 0.0, sigma = 0.005)
  # PC prior on an SD, calibrated (§6.5): P(tau_b > 0.5) = 0.05
  tau_b   : positive 'ratio ~ exponential(rate = 5.99)
  z[mode] : real 'ratio ~ normal(mu = 0.0, sigma = 1.0)
}

forcing {
  log_beta : bspline 'per_day {
    window  = [date("2025-08-04"), last_obs - 2 'weeks]
    n_basis = 12
    link    = log
    coefs   = smooth(level = a_b, slope = b_b, scale = tau_b, modes = z, per = 1 'weeks)
  }
}
```

The level, slope and modes are declared **without a range**: `fit run` gives
every parameter with a finite search box the scaled-logit transform
(`fit/runner.rs` `derive_transform_with_bounds`), so a boxed mode would be
sampled as logit(z) rather than z, losing the whitened geometry and truncating
its N(0, 1) law. Unbounded, they are searched over (−∞, ∞) on the identity scale
and their chains start from prior draws. Under the samplers (`pgas`, `pmmh`,
`mh`) `fit run` refuses a finite box on any of them, from the model's range or
`[estimate]` bounds; forcing the identity on a boxed parameter would instead
leave its box unenforced (gh#66). τ is declared without a range too, so its log
transform runs over (0, ∞): a finite box on a log-transformed parameter is a
hard wall where `log_jacobian` is −∞, and the prior-only runs counted 10–31
divergences per chain with τ boxed to [1e-6, 2.5] and none unboxed. The refusal
covers τ as well.

`smooth` names a penalized curve; there is no walk, so the form is not called a
random walk. It is accepted only as the `coefs` of a `bspline` (not
`periodic_spline`, whose wrap-around penalty is a different matrix) of degree at
least 2, so curvature exists. Compile-time checks, new codes in the E4xx forcing
range: `modes` is an indexed family with exactly `n_basis − 2` cells, each
`real` with prior exactly `normal(mu = 0, sigma = 1)`, so the name stays honest;
`level` and `slope` are scalars. The dimension check requires a dimensionless
level, a slope of 1/time, a dimensionless scale, and a duration `per`; a
parameter in `per` is refused as structural (E600).

**IR.** Because M depends on knots that may be observation anchors, it cannot be
computed by the compiler. The coefficients therefore stay symbolic in the IR,
and the runtime builds M after anchor substitution:

```rust
// rust/crates/ir/src/time_func.rs
pub enum SplineCoefs {
    Explicit { coefs: Vec<Expr> },
    Smooth { level: Expr, slope: Expr, scale: Expr, modes: Vec<Expr>, per: Expr },
}
// BSpline.coefs: SplineCoefs. ForcingBasis gains `space: BasisSpace`:
#[serde(rename_all = "snake_case")]
pub enum BasisSpace { Spline, Whitened, Linear }   // B_k(t), W_k(t), t_c − t_m
```

All join the branch's single pending IR bump. `CompiledModel::new` computes Ω,
Q₊, L and M for each `Smooth` forcing — O(K·p²) for Ω, O(K³) for the QR, the
Cholesky factor and the solves, once per compiled model. Evaluation computes the
span, the p + 1 values Bⱼ(t_c), then g = a + b(t_c − t_m) + (τ/κ) Σᵢ zᵢ Σⱼ M_jᵢ
Bⱼ — O(p·K) multiply-adds and K − 2 parameter reads per call, about 40 + 4·10
for K = 12 (inferred). The derivative nodes are `forcing_basis(f, i, whitened)`,
Wᵢ(t), O(p) each.

**Gradient cost** is O(pK) per referencing rate per substep for the K − 2 mode
gradients, plus O(1) for a and b and O(pK) for τ — trivial for K ≤ 50. The
autodiff rule (§5.2) emits, for `Smooth`: ∂g/∂a → 1, ∂g/∂b →
`forcing_basis(f, 0, linear)` (which evaluates t_c − t_m, so the clamp and lag
cannot be re-derived differently), ∂g/∂zᵢ → (τ/κ)·Wᵢ, ∂g/∂τ → Σᵢ zᵢWᵢ/κ, each
multiplied by `time_func(f)` under `link = log`. κ is a constant the runtime
knows and the compiler does not when the window is anchored, so the Whitened
space returns Wᵢ/κ and the rule emits τ·`forcing_basis(f, i, whitened)`.

**Under IF2 and the optimizers the scale must be fixed.** They read no prior,
and the likelihood depends on (τ, z) only through w, so an estimated τ beside
estimated z is a ridge with no maximum. `fit run` refuses an estimated `smooth`
scale under `if2` and the NLopt methods, naming the scale and the remedy: fix τ
(the fit is then unpenalized maximum likelihood over an affine
reparameterization of w), or estimate explicit coefficients. The refusal sits
beside `validate_hierarchical_leaf_boxes` (`fit/config_v2.rs:2604`) and moves
into `resolve_roles` with roles Stage 2.

### 6.5 τ has a PC prior whose tail is calibrated once

τ = 0 is the base model, log β linear in t, and the penalized-complexity prior
of Simpson et al. (2017) for a Gaussian standard deviation measured from that
base is exponential: τ ~ exponential(λ), with λ = −ln α/U for the tail statement
that τ exceeds U with probability α.

A tail statement bounds curvature, not the curve, so U is calibrated by
prior-predictive simulation against a target band. Target: at every t in the run
(window [0, 168], held to 182), the 95% prior band of R(t) = β(t)/γ contains
[0.5, 3] (every plausible value is admitted) and lies inside [0.1, 20] (absurd
ones are excluded). With a ~ N(ln(1.2γ), 0.6²) (median R = 1.2 mid-window), b ~
N(0, 0.005²) per day (one SD changes β by a factor e^{0.42} ≈ 1.5 over half the
window) and T_ref = 7 days, a scan of U ∈ {0.3, 0.4, 0.5, 0.6, 0.8} meets the
target for U ≤ 0.5 on every knot set. At U = 0.6 it fails on K = 8 (lower band
0.096 < 0.1 in the committed calibration table) and holds on the others; at 0.8
it fails on three of the four. U = 0.5 is chosen. Measured with a scratch script
(scipy `BSpline`, cubic, clamped, γ = 0.125, 4,000 draws, seed 20261027; α =
0.05, U = 0.5, λ = 5.99):

| Knots            | R(t) band at t = 0 | t = 42       | t = 84       | t = 126      | t = 168 and held to 182 | range over t  |
| ---------------- | ------------------ | ------------ | ------------ | ------------ | ----------------------- | ------------- |
| K = 8 uniform    | [0.16, 10.94]      | [0.24, 5.96] | [0.14, 9.34] | [0.25, 5.80] | [0.14, 9.69]            | [0.14, 10.94] |
| K = 12 uniform   | [0.17, 8.50]       | [0.28, 5.26] | [0.21, 7.72] | [0.28, 5.16] | [0.17, 8.06]            | [0.17, 8.50]  |
| K = 20 uniform   | [0.20, 7.84]       | [0.31, 4.56] | [0.27, 4.89] | [0.31, 4.56] | [0.20, 7.78]            | [0.20, 7.84]  |
| K = 12 irregular | [0.19, 7.76]       | [0.26, 5.76] | [0.19, 8.36] | [0.27, 5.38] | [0.18, 7.82]            | [0.18, 8.54]  |

With the √(K − 2) normalization the bands now narrow slightly as K grows, the
opposite of the unnormalized form, where they widened (K = 20 reached 11.9). The
K = 8 band is widest because a coarse basis concentrates the same total
curvature in fewer, larger bends; the target holds at every K. τ is not boxed:
its log transform runs over (0, ∞) (§6.4).

**Converting the calibration.** A tail U₀ stated for the unnormalized penalty in
model days (τ₀ multiplying M z directly) corresponds to U = U₀ · T_ref^{3/2} ·
√(K − 2) here; the earlier day-scale tail U₀ = 0.01 maps to 0.59 at K = 12,
consistent with the 0.5 chosen. A model whose `time_unit` is weeks uses the same
U, because T_ref is converted to model time.

Stage 4 commits this calibration as part of the default prior's definition:
`scripts/calibrate_smooth_prior.py`, run by
`make -f tests/recovery/Makefile calibrate-smooth-prior`, writes
`tests/recovery/cases/spline_seir/smooth_prior_calibration.tsv` (U, λ and the
band at every t, per knot set) from a fixed seed; the shipped example configs
read their τ prior from it, and a test asserts the committed table reproduces.

### 6.6 The non-centred form now; the realization may become a setting later

D1 is decided: the sampler moves (a, b, τ, z), the non-centred whitened form.
Every prior is a scalar `Prior::Fixed` density, summed by the existing
per-coordinate code, so no high-risk file changes and the PGAS hierarchical
refusal (`pgas.rs:3775-3784`) is not reached. The centred form (sampling w under
the Gaussian prior with precision Ω/τ²) mixes better where the latent path is
very informative about β(t) (Papaspiliopoulos, Roberts & Sköld 2007), and
interweaving the two in each sweep (ASIS; Yu & Meng 2011) is robust across both
regimes. Either may later become a user setting — the goal is whatever mixes
best for Ebola-style fits — but no setting is built now. Both would need a
non-separable prior term at the four prior-summing sites of `pgas.rs`
(`:4064-4069`, `:4287-4303`, `:4444-4446`, `:4544-4548`) and in PMMH; that
design is recorded here so it is not re-derived: one function
`log_prior_and_grad_z(priors, joint, specs, θ, z)` that all four sites call.

## 7. Under parameter roles the coefficients are an estimated indexed family

### 7.1 Declaring the family

An indexed parameter compiles to one scalar per cell (`z_m1 … z_m10`, measured
§3), and the roles partition is keyed by scalar names (roles §4.5). D5 proposes
a family declaration. With it, a `smooth` forcing needs four `[estimate]` lines
and no per-member lines:

```toml
[estimate]
a_b = {}
b_b = {}
tau_b = {}
z = {} # expands to z_m1 … z_m10 with shared settings
```

The bounds and priors come from the model, as for any parameter. A constructor
that generated its own parameters with default priors was considered and not
proposed: the priors would be invisible in the model file, and the generated
names would join the parameter namespace without a declaration.

### 7.2 Bounds, boxes and transforms

Every family declares finite model bounds, so the design does not depend on the
`real` fallback being fixed on main (§3). A finite box gets the scaled-logit
transform (`runner.rs:1167-1198`, verified). For the modes, `in [-8, 8]` cuts
2·Φ(−8) ≈ 10⁻¹⁵ of the N(0, 1) mass; they are not hierarchical leaves, so
`validate_hierarchical_leaf_boxes` does not apply. Once the `real` support is
(−∞, ∞) on main, the modes should drop their box and take the identity
transform.

### 7.3 Starts

Starts come from `--starts` or `[estimate].start` (roles §5.3). `from_prior`
draws inside the box (`e6867779`), which suits a, b, τ and z; for explicit
coefficients with flat priors, box-wide draws are wild curves, and a shared
scalar start (a constant curve) or `--starts from_mle=` is better.

## 8. Merge criteria come in three tiers

Each required item is a committed test or make target; every artifact it writes
is produced by a rule. Tolerances are scale-aware: an absolute tolerance is
stated relative to the magnitude of the quantity compared (max|Ω|, the window
length, max|g|), never as a bare number.

### 8.1 Required for merge

**R1 — Numerical correctness.**

- Basis values against three independent references: scipy's `BSpline`, R's
  `splines::splineDesign`, and analytic anchors (at K = p + 1, a clamped spline
  on one interval is the Bernstein basis C(p, k) uᵏ(1 − u)^{p−k}). Uniform,
  explicit non-uniform, anchored-resolved and periodic knots; degrees 0–5;
  points at boundaries, interior knots, in between, and outside the window. Max
  abs error ≤ 10⁻¹² (values are O(1)).
- Properties over 10⁴ points per layout: partition of unity ≤ 10⁻¹⁴;
  non-negativity of every Bⱼ; end interpolation f(a) = w₁, f(b) = w_K to 10⁻¹⁴;
  exact reproduction of constants (coefficients 1) and of t (coefficients ξ) to
  10⁻¹³·(b − a), uniform and irregular.
- The penalty: Ω (2-point Gauss–Legendre) against 10-point Gauss–Legendre and
  against `bsplinepen` from the R package fda (Ramsay, Graves & Hooker), which
  computes ∫ D²B_i D²B_j — its independence from this implementation is
  unverified, because fda is not installed here; Stage 4 confirms it computes
  the penalty by its own quadrature rather than the same closed rule, or
  substitutes another reference, ≤ 10⁻¹²·max|Ω|; bandwidth p; rank K − 2; 1ᵀΩ1
  and ξᵀΩξ ≤ 10⁻¹²·max|Ω|·(b − a)² for uniform and irregular knots.
- The whitening: MᵀΩM = I to 10⁻¹²; ‖NᵀM‖_max ≤ 10⁻¹²·‖N‖₂‖M‖₂ (N's second
  column is of order the window half-length, about 84, so an unscaled bound
  would fail on correct code); τ²MMᵀ = τ²Ω⁺ to 10⁻¹⁰ relative; the induced
  density of §6.3 equals the improper Gaussian density with the null-space
  prior, compared pointwise at 100 random w (log-densities agree to 10⁻¹⁰ after
  removing the constant).
- Fixtures regenerated by `make fixtures-bspline` (scipy, R `splines`, R `fda`,
  and the existing pomp periodic fixture), committed; CI runs offline.

**R2 — Gradient correctness.** Central finite differences with step ε =
10⁻⁵·max(|θ|, 1) — not the harness's `max(10⁻⁵|θ|, 10⁻⁸)`
(`gradient_check.rs:653`), which is far too small near 0. Pass: relative error <
10⁻⁴ for every component with |g_fd| ≥ 10⁻³·max_k|g_fd,k|; absolute error ≤
10⁻⁷·max_k|g_fd,k| for the rest. Covered: the forcing value against each
coefficient (explicit and `smooth`, both links, user-written `exp`, constant
lag, inside, at a knot, at a and b, past b); the penalty ½wᵀΩw (gradient Ωw) and
the `smooth` map against a, b, τ, z; and the complete-data log-likelihood
through `complete_data_loglik_grad` for a spline in a rate and in an observation
mean, in the `gradient_check.rs` harness.

**R3 — Inference integration.**

- A small stochastic epidemic: chain-binomial SEIR, N₀ = 10⁴, daily steps, 60
  days, a K = 6 log-link `smooth` spline on β, PGAS+NUTS (dense mass matrix,
  stated in the config), 2 chains × 200 sweeps. It runs, every recorded
  log-likelihood and gradient is finite, and two runs at the same seeds produce
  bitwise-identical draws.
- A prior-only run under PGAS+NUTS, the cheap check of what the sampler actually
  draws: the toy model `X --> D @ f(t) * X` (f a log-link K = 6 `smooth` spline)
  with X₀ = 0, so every flow is zero and the complete-data likelihood is
  constant (inferred; Stage 5 confirms PGAS runs it, or uses an observation
  stream with every value missing). Over 4 chains × 4,000 sweeps, τ's draws must
  match exponential(λ) (Kolmogorov–Smirnov p > 10⁻³ on thinned draws), a and b
  their normals, and the sample covariance of w − a·1 − b(ξ − t_m) must match
  E[τ²]/κ²·Ω⁺ = (2/λ²κ²)·Ω⁺, each moment scored as a z-score on its batch-means
  Monte Carlo standard error with |z| < 5 (the `nuts_stationarity.rs` rule).
  This exercises the box transforms, the prior sums and τ together.
- One synthetic recovery with informative data and known parameters: the SEIR of
  §8.4 with N₀ = 10⁶ and a truth that lies in the spline space (a K = 6 `smooth`
  curve with stated a, b, τ, z), E₀, I₀ and ρ fixed, 4 chains. Pass: every true
  coefficient wⱼ lies in its central 99% posterior interval, and the posterior
  median of log β(t) is within 0.2 of the truth at every t in the window's
  middle 80%.

**R5 — Small simulation-based calibration (proposed addition to D7; pending the
maintainer's confirmation).** The single-dataset recovery above has little
power: six coefficients inside their 99% intervals passes most wrong posteriors.
Proposed: SBC (Talts et al. 2018) on the toy model `X --> D @ f(t) * X`, X₀ =
10⁵, f a log-link K = 6 `smooth` spline, daily `poisson` observations of the
flow; 100 prior draws, PGAS+NUTS, 99 thinned posterior draws each (100 possible
ranks, 20 bins); test quantities a, b, τ and f(tⱼ) at 5 times; rank histograms
pass a χ² test at α = 0.01/d (Bonferroni). Not decided here.

**R4 — Regression.** `make test` green, so every non-spline path passes.
Coefficient ordering, IR serialization (round-trip OCaml → JSON → Rust → JSON
byte-identical), knot definitions and M (bitwise, for a fixed model across two
builds) are reproducible; the `link` no-collision hash test (§5.6) passes.

### 8.2 Documented, not blocking

In a dated `docs/dev/notes/` note the merge commit cites:

- a representative benchmark on the §8.4 SEIR at K = 12: R̂, bulk-ESS, ESS per
  second and divergences for every sampled parameter (4 chains, 1,000 warm-up +
  2,000 sweeps, dense mass);
- one plot of the known against the inferred β(t), with the window end and the
  held region marked;
- known limitations: the level trades off against an estimated intercept or
  against I₀ and ρ (§4.4); the end of the window is the least informed part of
  the curve and the held value inherits that uncertainty (§4.3); the forecast
  past the window is a held value, not an extrapolated trend (§4.3);
- the τ calibration table of §6.5 as reproduced by its make target;
- IF2 on chain-binomial with τ fixed, scored by the particle filter against the
  truth's projection onto the basis (not asserted).

### 8.3 Deferred research

Not part of this branch: simulation-based calibration at scale (Talts et al.
2018); coverage studies over many datasets; efficiency against step-function
models; latent random walks and other parameterizations (centred, ASIS); joint
inference of transmission, initial conditions and ascertainment under weak
identification.

### 8.4 The synthetic outbreak used by R3 and §8.2

A chain-binomial SEIR, dt = 1 day, N₀ = 10⁶, E₀ = I₀ = 10, σ = 0.1 and γ = 0.125
per day, daily onsets observed as `neg_binomial(mean = 0.5 * projected, r = 10)`
over 182 days, window [0, 168] (two weeks before the last observation, §4.3).
For §8.2 the truth is R(t) = 0.7 + 1.5/(1 + exp((t − 90)/10)): deterministically
about 107 onsets/day at the peak near day 101 and about 8,750 in total (measured
with a scratch Euler integration; Stage 5 commits the generator). It is
generated by the same model with the spline replaced by an
`interpolated 'per_day` forcing from `beta_true.tsv`, written by a make target;
a test asserts the two IRs differ only in that forcing.

## 9. Five stages, each green alone, none touching a high-risk file

Each stage ends on a green `make test`; each named test is mutation-checked.
Every commit re-keys stored runs through the engine version (§4.2); the last
column says what changes beyond that.

| Stage | Content                                                                                                                                                                                                                         | IR                                               | High-risk files | Run identity beyond the engine version                      |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------- | ----------------------------------------------------------- |
| 1     | `bspline.rs` core; periodic evaluator on it; live periodic coefficients; `first_param_read` arm; before/after microbenchmark                                                                                                    | none                                             | none            | none                                                        |
| 2     | `bspline` kind with explicit coefficients, `link`, `ir_hash` arms with the no-collision test, the past-window notice (§4.3), `make fixtures-bspline`                                                                            | the pending bump (about 118 IR files move, §5.6) | none            | every model re-keys (version string); `link` hashed         |
| 3     | `forcing_basis` (space `Spline`); autodiff rule for both spline kinds; E600 lifted for coefficients; stale refusal messages rewritten                                                                                           | same pending bump                                | none            | models with coefficient parameters gain `rate_grad` entries |
| 4     | `smooth`: Ω, Householder QR and Cholesky whitening in `sim/src/spline_penalty.rs`; `SplineCoefs::Smooth`; spaces `Whitened` and `Linear`; autodiff rule; dimension checks; IF2/NLopt scale refusal; the calibration make target | same pending bump                                | none            | models using `smooth`                                       |
| 5     | R1–R4 completed (the stochastic-epidemic and recovery tests), the §8.2 note                                                                                                                                                     | none                                             | none            | none                                                        |

No stage touches `pgas.rs`, `pgas_grad.rs`, `obs_loglik.rs`, `obs_model.rs`,
`if2.rs` or `particle_filter.rs` (§5.3, §6.6).

**Stage 1.** Compiler unchanged; the runtime half is tested with hand-built IR.
The scipy and pomp oracles unchanged at 10⁻¹²; `partition_of_unity`;
`basis_matches_cox_de_boor_recursion` (the old recursion kept as a test oracle
only); `periodic_spline_coef_param_is_rejected` becomes `…_builds_live`; a PGAS
refusal test of an intervention amount that reads a spline-coefficient forcing.
`eval_ab` compares two evaluators inside one build and cannot measure this
change, so `benches/spline_eval.rs` times the old and new evaluators side by
side; its result is pasted in the commit.

**Stage 2.** The R1 basis items; each new golden compiles and simulates; an
anchored window resolves and its inversion is refused; `link = log` with
`'per_week` equals `'per_day` scaled by 1/7 to 10⁻¹²; parameter coefficients
still E600; the red→green `link` no-collision test; the notice fires for a
horizon past the window and not otherwise, and its code and message are pinned.

**Stage 3.** R2 for explicit coefficients; `forcing_basis` outside a gradient
map, with a bad index, or with a space its forcing lacks, is refused by
`ir::validate`; a parameter in a window end is still E600, and the
`URStructuralForcing` text (`autodiff.ml:360-370`), the
`UnsupportedReason::StructuralForcing` message (`deriv.rs:73-75`) and the
`eval_structural` message (`compiled_model.rs:736-741`) no longer name spline
coefficients.

**Stage 4.** The R1 penalty and whitening items; R2 for `smooth`; the
calibration target reproduces the committed table; `smooth` on a degree-0 or
degree-1 spline, on a `periodic_spline`, or with a wrong mode count or mode
prior is refused with its code.

**Stage 5.** R3, R4, and the §8.2 note.

## 10. Sequencing with the typed-constants change and the roles stages

The planned IR change that moves typed `let` constants into a `constants`
section lands on main first; this branch neither waits for it nor joins it (D6).
At a rebase onto a main that has bumped `ir/VERSION`, the branch's single
pending bump is renumbered and its IR files regenerated — about 101 generated
files plus the 17 hand-edited ones in `ir/golden/` (§5.6) — so the branch
rebases at stage boundaries and once before merge, not continuously. The two
changes are orthogonal in content: a coefficient is an expression, wherever a
constant it names lives.

Roles Stage 1 does not interact. Roles Stage 2 brings `resolve_roles`, into
which the `smooth` scale refusal moves, and is the natural place for D5's family
declaration.

## 11. Decisions

**D1 — Decided: the non-centred prior now.** The sampler moves the level, slope,
scale and whitened modes (§6.3). The realization may later become a user
setting, centred or interweaving (§6.6), chosen by what mixes best for
Ebola-style fits; no setting is built now.

**D2 — Decided: offer `link = log`** (§4.4).

**D3 — Decided: cubic clamped B-splines, uniform or non-uniform knots, under the
integrated squared curvature penalty** ∫ g''² dt = wᵀΩw, with Ω computed exactly
(§6.1), compared with the two alternatives (§6.2), and sampled in the
null-space-separated whitened form (§6.3), whitened by Cholesky rather than
eigendecomposition. No heuristic end weights. One detail is recommended within
it: normalize the penalty by √(K − 2) as well as by the declared reference
duration, so τ means the same at every knot count and knot count and smoothness
are separate choices; without it, adding knots silently loosens the prior
(§6.3). Confidence: leaning.

**D4 — Decided: hold outside the window, and say so.** §4.3 defines what a run
past the window computes, and a static info notice reports it.

**D5 — Pending the maintainer's confirmation: a bare family name in
`[estimate]`.** `z = { bounds = [lo, hi], start = s }` expands to every member
of the indexed family `z` with those shared settings, and the partition records
each member separately (roles §4.5 keeps scalar keys). It is refused if a scalar
named `z` exists, and refused if per-member entries (`z_m1 = …`) for the same
family also appear, because each parameter has one declaration (roles §4.2). Two
options were considered. A quoted key `"z[*]"` is explicit about expansion but
introduces a pattern syntax, and invites partial patterns that would need their
own rules. A family header with per-member overrides (`[estimate.z]` plus
`z_m3 = { start = … }`) is more flexible but reintroduces two declarations for
one parameter, which roles §4.2 removed. The bare name needs no new syntax, and
its two refusals make it unambiguous. A `smooth` forcing is then four lines,
`a_b`, `b_b`, `tau_b` and `z` (§7.1). Recommendation: the bare name, in roles
Stage 2. Confidence: leaning.

**D6 — Decided: (a), one IR bump kept on the branch** (§10).

**D7 — Decided: the merge criteria are the three tiers of §8.** One change is
proposed for confirmation: add R5, a small SBC on the toy model, to the required
tier, because R3's single-dataset recovery has little power. Recommendation: add
it. Confidence: leaning.

## 12. Out of scope, each with its reason

- **A latent random walk on log β** (stochastic transmission, β(t) as a state of
  the process rather than a smooth function of θ). It is a different model
  class, planned separately.
- **A centred or interweaving realization, or a setting to choose one** (D1,
  §6.6).
- **Penalized maximum likelihood under IF2.** IF2 reads no prior; adding a
  penalty to its likelihood is an IF2 change, in a high-risk file.
- **A smoothness prior on `periodic_spline`.** Its wrap-around penalty is a
  different matrix; add it when a periodic fit needs it.
- **Estimated knot positions or period.** Their derivatives move the basis; they
  stay structural (E600).
- **Degree above 5, coincident interior knots.** Not needed for smooth rates,
  and both complicate §4.1's stability argument.
- **A span cache per (forcing, t).** An optimization after a profile (§5.5).
- **The `[0, ∞)` box for an unbounded `real`, and the stale, untested
  `ir/schema.json`** (§3). Both are being fixed separately on main; nothing here
  depends on either staying as it is.
- **Splines over a stratification dimension** (one curve per province sharing a
  τ). Expressible as one forcing per stratum today; a stratified kind is
  separate design.

## References

Load-bearing for the design:

- Cox, M. G. (1972). The numerical evaluation of B-splines. _IMA Journal of
  Applied Mathematics_ 10(2):134–149. The stable recurrence (§4.1).
- de Boor, C. (1972). On calculating with B-splines. _Journal of Approximation
  Theory_ 6(1):50–62. The same recurrence (§4.1).
- de Boor, C. (1978). _A Practical Guide to Splines_. Springer, Applied
  Mathematical Sciences 27. Ch. X, stable evaluation of B-splines; the
  linear-reproduction property behind the Greville abscissae; the periodic
  construction already used.
- Piegl, L. & Tiller, W. (1997). _The NURBS Book_, 2nd ed. Springer. §2.5,
  Algorithms A2.1 (span), A2.2 (nonzero basis functions) and A2.3 (their
  derivatives) (§4.1, §6.1). A2.3 is cited from memory.
- Ramsay, J. O., Graves, S. & Hooker, G. _fda: Functional Data Analysis_. R
  package, CRAN; the version is pinned in the R1 fixture's provenance. Its
  `bsplinepen` is a reference for Ω (§8.1); its independence is unverified.
- O'Sullivan, F. (1986). A statistical perspective on ill-posed inverse
  problems. _Statistical Science_ 1(4):502–518. The curvature penalty (§6.1).
- Wand, M. P. & Ormerod, J. T. (2008). On semiparametric regression with
  O'Sullivan penalized splines. _Australian & New Zealand Journal of Statistics_
  50(2):179–198. Computing Ω, and its relation to the difference penalty (§6.1);
  already cited by `periodic_bspline.rs`.
- Eilers, P. H. C. & Marx, B. D. (1996). Flexible smoothing with B-splines and
  penalties. _Statistical Science_ 11(2):89–121. The difference penalty (§6.1).
- Papaspiliopoulos, O., Roberts, G. O. & Sköld, M. (2007). A general framework
  for the parametrization of hierarchical models. _Statistical Science_
  22(1):59–73. Centred versus non-centred (§6.2).
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018).
  Validating Bayesian inference algorithms with simulation-based calibration.
  arXiv:1804.06788 (§8.3).
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: an improved R̂ for assessing
  convergence of MCMC. _Bayesian Analysis_ 16(2):667–718 (§8.2).

- Yu, Y. & Meng, X.-L. (2011). To center or not to center: that is not the
  question — an ancillarity–sufficiency interweaving strategy (ASIS) for
  boosting MCMC efficiency. _Journal of Computational and Graphical Statistics_
  20(3):531–570. Interweaving (D1).

Context:

- Simpson, D., Rue, H., Riebler, A., Martins, T. G. & Sørbye, S. H. (2017).
  Penalising model component complexity: a principled, practical approach to
  constructing priors. _Statistical Science_ 32(1):1–28. The prior on τ (§6.1).
- Lindsten, F., Jordan, M. I. & Schön, T. B. (2014). Particle Gibbs with
  ancestor sampling. _Journal of Machine Learning Research_ 15:2145–2184.
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler. _Journal of Machine
  Learning Research_ 15:1593–1623.
- Ionides, E. L., Nguyen, D., Atchadé, Y., Stoev, S. & King, A. A. (2015).
  Inference for dynamic and latent variable models via iterated, perturbed Bayes
  maps. _PNAS_ 112(3):719–724.
