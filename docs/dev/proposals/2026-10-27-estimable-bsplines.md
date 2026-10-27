---
date: 2026-10-27
status: proposed — decisions D1–D7 (§11) await the maintainer; implementation follows §9 on `feat/bsplines`, which stays off main until §8's thresholds are met
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
3. **A smoothness prior.** With 8–12 coefficients and an outbreak's worth of
   data, independent coefficients overfit and mix slowly. The Bayesian P-spline
   prior — a random walk on the coefficients with scale τ — is the standard
   remedy (§6).

The recommended design adds one forcing kind (`bspline`), one IR expression node
(`forcing_basis`, the derivative), a `link = log` option, and a coefficient
constructor (`random_walk`) that realizes the smoothness prior in its
non-centred form out of ordinary per-parameter priors. That last choice means no
high-risk inference file changes in the recommended path (§5.6, §6.4); a centred
realization that does touch `pgas.rs` is specified as a separate, conditional
stage (§6.6, D1).

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
- **Random walk of order r** (RW1, RW2) on c₁…c_K: the r-th differences (Δcₖ =
  cₖ − cₖ₋₁; Δ²cₖ = cₖ − 2cₖ₋₁ + cₖ₋₂) are independent N(0, τ²). **τ** is its
  scale. It is a Gaussian Markov random field (GMRF): a Gaussian whose precision
  matrix is sparse (banded).
- **Centred** parameterization: the sampler moves c and τ. **Non-centred**: the
  sampler moves standardized innovations zₖ ~ N(0, 1) and τ, and c is computed
  from them.
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
  degree : int;
  coefs  : expr list;                (* length = n_intervals + degree *)
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
    pub degree: u32,
    pub coefs: Vec<Expr>,
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

**Ends: clamped, with the prior's differences taken over the Greville
abscissae** (D3). The knot vector repeats a and b p + 1 times. Every basis
function lies inside the window, f(a) = c₁ and f(b) = c_K exactly, and explicit
breakpoints need no extension beyond the data. Clamping makes the knot averages
uneven at the ends, so the random-walk prior of §6 takes its differences over
those averages rather than over the coefficient index; otherwise it would
penalize a log-linear curve at both ends (§6.1). The alternative is the
unclamped construction of Eilers & Marx (1996): equally spaced knots extended p
intervals past each end, plain index differences, and the forcing held at f(a),
f(b). D3 sets out both.

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

A window that does not cover the simulation is allowed; `camdl simulate` prints,
once per run on stderr, the window and the held values, as it prints resolved
anchors.

**End the window before the reporting delay.** The last basis function
integrates to a quarter of an interior one (cubic, clamped), and infections near
b are mostly reported after the data end: with incubation near 10 days and h =
20 days, c_K is largely the prior's extrapolation, yet it is the value held for
every forecast. The language spec section Stage 2 adds therefore recommends
`window = [..., last_obs - d]`, d the infection-to-report delay (two weeks for
Ebola), so the held value is a coefficient the data inform; the validation does
this (§8.1) and checks coverage in the held region.

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
pub struct ForcingBasisRef { pub name: String, pub index: u32 }
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
arithmetic over parameters — a corner constraint, a sum-to-zero, or the
`random_walk` lowering of §6 — gets the chain rule with no special case.
`simplify_fixpoint` drops the terms whose ∂cⱼ/∂θ folds to zero, so for cⱼ =
`Param θ` the emitted gradient is the single node `forcing_basis(f, j)`.

**Through `exp`.** Under `link = log`, ∂f/∂cⱼ = f(t)·Bⱼ(t), emitted as
`time_func(f) * forcing_basis(f, j)`. When the user writes `exp(logb)` with an
identity link, the existing `Exp` rule (`autodiff.ml:575-576`, verified)
produces exp(g)·Bⱼ. Both are finite-difference tested (§8, V2).

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

Two schema additions — the `bspline` kind with `link` (Stage 2), and
`forcing_basis` (Stage 3) — share one pending `ir/VERSION` bump on the branch,
made in Stage 2 (0.40 → 0.41 at today's main; renumbered at rebase, §10). The
envelope embeds the version in every serialized IR file (verified:
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

## 6. The smoothness prior is a random walk on the coefficients, realized non-centred

### 6.1 What the prior is, and why its differences are taken over the Greville abscissae

The Bayesian **P-spline** prior — a B-spline with a random-walk prior on its
coefficients (Lang & Brezger 2004), the Bayesian form of the difference penalty
of Eilers & Marx (1996) — is a random walk of order r on the coefficient vector.
In its textbook form the r-th index differences are independent, Δʳcₖ ~ N(0,
τ²), k = r + 1, …, K.

**With clamped knots the index differences have the wrong null space.** A spline
equals a straight line exactly when its coefficients are linear in the
**Greville abscissae** ξₖ = (u_{k+1} + … + u_{k+p})/p, the knot averages, not in
the index k (de Boor 1978). Interior ξ are evenly spaced by h; at a clamped end
they crowd together: for K = 8 over [0, 182], ξ = 0, 12.1, 36.4, …, 145.6,
169.9, 182. Index-linear coefficients therefore give a curve that bends near
both ends — 0.57 on a log range of 7 (measured by the statistics review) — so
the RW2 prior penalizes a truly exponential growth or decline at exactly the
ends, and the hold rule (§4.3) takes the forecast value from the end. The fix is
to take the differences over ξ (D3 gives the alternative, unclamped knots):

- first divided differences sₖ = (cₖ − cₖ₋₁)/(ξₖ − ξₖ₋₁), the local slope of log
  β per unit time;
- for RW2, slope increments sₖ − sₖ₋₁ ~ N(0, (τ/h)² · ω̄ₖ/h), with ω̄ₖ = (ξₖ −
  ξₖ₋₂)/2 the mean spacing around ξₖ₋₁; for RW1, cₖ − cₖ₋₁ ~ N(0, τ² · (ξₖ −
  ξₖ₋₁)/h).

In the interior ω̄ₖ = h and the RW2 rule reduces to Δ²cₖ ~ N(0, τ²) exactly. The
end weights are a heuristic, stated as one: they scale each increment's variance
by its local spacing, as a Brownian (RW1) or integrated-Brownian (RW2) increment
would, and they touch only the p − 1 increments at each end, where clamping
makes the spacing uneven. No irregular-grid RW2 from the literature is adopted
here, because none was checked against this exact construction; V6 and V7 check
what the heuristic does. The null space at τ = 0 is now constant slope in time,
i.e. log β linear in t, which the spline reproduces exactly at every t in the
window.

As a GMRF (Rue & Held 2005, ch. 3) the density is

p(c | τ) ∝ τ^{−(K−r)} exp(−‖W_r c‖² / (2τ²)),

where W_r is the (K − r) × K matrix of the weighted divided differences above
(W_r = D_r, the plain difference matrix, when every spacing is h). The precision
Q = W_rᵀW_r/τ² has rank K − r; its null space is constants (RW1) or functions
linear in ξ (RW2). Two consequences are load-bearing. The prior is improper
along the null space, so it must be completed there; and the normalizing power
is τ^{−(K−r)}, not τ^{−K}, which matters as soon as τ is estimated (using K
biases τ downward).

**Completion.** Give the null-space coordinates proper priors, anchored at the
middle of the window rather than at an end, so the prior band does not fan out
from one side: with m the index whose Greville abscissa ξ_m is nearest (a +
b)/2, the level c_m for RW1, and for RW2 the level c_m and the slope s_{m+1} =
(c_{m+1} − c_m)/(ξ_{m+1} − ξ_m). The completed density is p(c_m) [p(s_{m+1})] Πₖ
N(·; 0, ·) as above: proper, with log-density log p(c_m) [+ log p(s_{m+1})] − (K
− r) log τ − ‖W_r c‖²/(2τ²) + const and gradients ∂/∂c = (null-space terms) −
W_rᵀW_r c/τ² and ∂/∂τ = −(K − r)/τ + ‖W_r c‖²/τ³. W_rᵀW_r is banded with
bandwidth r, so both are O(K).

**τ is estimated, under a penalized-complexity prior on a standard deviation.**
τ = 0 is the base model — a constant log-rate (RW1) or a log-linear one (RW2) —
and the PC prior of Simpson et al. (2017) for a Gaussian standard deviation
measured from that base is exponential: τ ~ exponential(λ), with λ set by a tail
statement P(τ > U) = α, λ = −ln α / U. The scale of τ depends on the knot
spacing: second differences of an integrated Wiener process at spacing h have
standard deviation ∝ h^{3/2} (first differences ∝ h^{1/2} for RW1) (inferred
from the integrated-Wiener-process limit; Sørbye & Rue 2014 make the same point
for intrinsic GMRFs). The tail is therefore stated per spacing, U_h = U₁₄ · (h /
14 days)^{3/2} with α = 0.05, 14 days being near the Ebola serial interval.
Spacing h is chosen at or below the generation interval, so the basis can follow
changes on the time scale transmission can change.

**U₁₄ is calibrated by prior-predictive simulation, not chosen by intuition.** A
tail statement on τ bounds a local second difference, not the curve: the random
walk integrates K − 2 innovations twice, so the prior spread of log β(t) grows
with distance from the anchor and with K, on top of the slope's own
contribution. A first choice of U₁₄ = 1.0 with b₁ ~ N(0, 0.05/day) gave a 95%
prior band of R(t) reaching about 10¹² at t = 168 (found by the statistics
re-review): the slope alone has a log-scale SD of 8.4 there. The calibration
therefore fixes a target band and searches U₁₄. Target: at every t in the
window, the 95% prior band of R(t) contains [0.5, 3] (the prior admits every
plausible value) and lies inside [0.1, 20] (it excludes absurd ones). With the
level log β(ξ_m) ~ N(ln(1.2γ), 0.6²) (median R = 1.2 mid-window) and b₁ ~ N(0,
0.005²) per day — one SD changes β by a factor e^{0.42} ≈ 1.5 over half the
window, the scale of change control measures produce, whereas 0.05/day would be
an e-fold every 20 days sustained for months — a scan over U₁₄ ∈ {0.1, 0.2, 0.3,
0.5} meets the target at U₁₄ = 0.2 and fails it at 0.3 for K = 8 and at 0.5 for
every K. Measured with a scratch prior-predictive script (scipy `BSpline`,
cubic, clamped, window [0, 168], γ = 0.125, 4,000 draws, seed 20261027):

| K  | h (days) | U_h   | λ = ln 20 / U_h | R(t) band at t = 0 | t = 42       | t = 84       | t = 126      | t = 168 (and held) | range over t  |
| -- | -------- | ----- | --------------- | ------------------ | ------------ | ------------ | ------------ | ------------------ | ------------- |
| 8  | 33.6     | 0.744 | 4.03            | [0.27, 5.37]       | [0.36, 3.97] | [0.36, 3.81] | [0.29, 4.76] | [0.14, 9.29]       | [0.14, 9.29]  |
| 12 | 18.7     | 0.308 | 9.73            | [0.22, 6.88]       | [0.34, 4.08] | [0.36, 3.99] | [0.28, 4.68] | [0.14, 10.28]      | [0.14, 10.28] |
| 20 | 9.9      | 0.119 | 25.26           | [0.17, 8.06]       | [0.34, 4.56] | [0.37, 3.86] | [0.30, 4.55] | [0.15, 10.39]      | [0.15, 10.39] |

Stage 5 commits this calibration as a reproducible step:
`scripts/calibrate_rw_prior.py`, run by
`make -f tests/recovery/Makefile calibrate-rw-prior`, writes
`tests/recovery/cases/spline_seir/rw_prior_calibration.tsv` (U₁₄, U_h, λ and the
band at every t, per K) from a fixed seed, and the fit configs read their τ
priors from it.

### 6.2 Three realizations, and why the non-centred one comes first

**(A) A centred GMRF prior type.** A new joint prior on an indexed family,
`c ~ random_walk(order = 2, scale = tau)`, whose log-density and gradient are
those above, added to the prior sum. The sampler moves c and τ. The prior sum in
camdl is per scalar: `prior_log_density_and_grad_z(prior, param, θ, z)`
(verified, `pgas.rs:3557`), summed coordinate by coordinate in the NUTS target
(`:4287-4303`), the MH-within-Gibbs move (`:4444-4446`), the chain-start check
(`:4064-4069`) and the probation check (`:4544-4548`), and likewise in PMMH
(inferred from the acceptance ratio the roles proposal verified,
`pmmh.rs:707-714`). A non-separable term needs a new seam at every one of those
sites, in a high-risk file. Centring also creates the funnel when τ is estimated
and the data are weak: as τ → 0 the conditional of c narrows, and a fixed-step
integrator cannot follow both the neck and the mouth (Betancourt & Girolami
2015).

**(B) Lifting PGAS's hierarchical refusal.** The random walk is already
expressible as a chain of hierarchical leaves,
`c3 ~ normal(mu = 2*c2 - c1,
sigma = tau)`, one declaration per cell, because
hierarchical prior arguments are expressions (`ir/src/parameter.rs:98-108`).
PGAS would need the env-aware density and its gradient with respect to the leaf
and every parameter its arguments read — the same non-separable seam as (A), in
general form — and the roles proposal's hierarchical refusals (held leaf, boxed
leaf, `9ccb2899`) then apply to every coefficient. It is centred, with (A)'s
funnel, and verbose.

**(C) Non-centred by construction, from ordinary priors** (recommended, D1). The
sampler moves standardized innovations zₖ ~ N(0, 1), the mid-window level b₀
(and for RW2 the mid-window slope b₁), and τ, each with an ordinary scalar prior
the samplers already handle. The coefficients are deterministic functions of
them:

RW1: c_m = b₀, cₖ = cₖ₋₁ + τ √((ξₖ − ξₖ₋₁)/h) zₖ for k > m and symmetrically for
k < m; RW2: c_m = b₀, s_{m+1} = b₁, sₖ = sₖ₋₁ + (τ/h) √(ω̄ₖ/h) zₖ, cₖ = cₖ₋₁ +
(ξₖ − ξₖ₋₁) sₖ for k > m + 1, and the mirror-image recursion toward k = 1,

with m the mid-window index of §6.1 and b₁ the mid-window slope of log β per
unit time; in closed form for RW2 and k > m, cₖ = b₀ + b₁ (ξₖ − ξ_m) + (τ/h)
Σ_{i=m+2}^{k} √(ω̄ᵢ/h) (ξₖ − ξᵢ₋₁) zᵢ, with the mirror image for k < m. There are
still K − 2 innovations.

The map (b₀, b₁, z) → c, for fixed τ, is linear with Jacobian proportional to
τ^{K−r} — τ^{K−2} for RW2, confirmed by the statistics review — times spacing
factors that do not involve τ, which is exactly the GMRF normalizer: (C) induces
on (c, τ) the same completed prior as (A). The posterior over the curve is
identical; only the sampler's coordinates differ. Nothing in the samplers
changes, because every prior in (C) is a scalar prior; the coefficients are
coefficient expressions, which §5.2 differentiates by the chain rule (for RW2,
∂cⱼ/∂zᵢ = (τ/h) √(ω̄ᵢ/h) (ξⱼ − ξᵢ₋₁) for j ≥ i, on the right of m, and ∂cⱼ/∂τ =
(cⱼ − b₀ − b₁(ξⱼ − ξ_m))/τ).

**The trade-off is information-dependent, and PGAS sits on both sides of it.**
Papaspiliopoulos, Roberts & Sköld (2007) show the centred form mixes better when
the data are informative about c and the non-centred form when they are not. The
PGAS θ-update conditions on the whole latent path, which is very informative
about β(t) where many infections occur and barely informative in a long tail of
few cases. In the informative regime (C) has its own pathology: with c pinned, z
and τ lie on the ridge z ∝ 1/τ. The dense mass matrix PGAS adapts by default
(`default_dense_mass` = true, verified `fit/config_v2.rs:1725`) captures a
linear correlation but not that curvature; spline fits require it, and the V4
configs state it explicitly rather than relying on the default. Interweaving the
two parameterizations within each sweep (the ancillarity–sufficiency
interweaving strategy, ASIS, of Yu & Meng 2011) is robust across both regimes,
but it needs the centred seam as well; it is an extension of Stage 6, not an
alternative to it. Which regime dominates on an Ebola-scale outbreak is a
measurement, made in Stage 5 (§8, V4). (C) comes first because it is correct
with no high-risk change; (A) is built only if (C) fails the efficiency
threshold (§6.6).

### 6.3 `random_walk(...)` is a coefficient constructor, and the priors stay visible

```camdl
dimensions { innov = [i3, i4, i5, i6, i7, i8, i9, i10, i11, i12] }   # n_basis − order levels

parameters {
  b0        : real 'ratio in [-8, 2]  ~ normal(mu = -1.9, sigma = 0.6)       # log β mid-window (R ≈ 1.2)
  b1        : real [1/T] in [-0.05, 0.05] ~ normal(mu = 0.0, sigma = 0.005)  # mid-window slope of log β, per day
  # PC prior on an SD, calibrated (§6.1): P(tau_b > 0.308) = 0.05 at h = 18.7 days
  tau_b     : positive 'ratio in [1e-6, 1.54] ~ exponential(rate = 9.73)
  z[innov]  : real 'ratio in [-8, 8]  ~ normal(mu = 0.0, sigma = 1.0)
}

forcing {
  log_beta : bspline 'per_day {
    window  = [date("2025-08-04"), last_obs - 2 'weeks]
    n_basis = 12
    link    = log
    coefs   = random_walk(order = 2, start = [b0, b1], scale = tau_b, innovations = z)
  }
}
```

The expander computes the Greville abscissae from the knots and lowers
`random_walk(...)` to the K closed-form coefficient expressions above (`Reduce`
sums over `Param` nodes with constant weights). When a window end is an
observation anchor the abscissae are not known at compile time; the weights are
then emitted as expressions over the window ends, which anchor substitution
resolves like any knot (§4.3), so the IR contains only existing expression forms
and **no IR change** is needed for the prior. The priors are declared by the
user, with today's syntax, where a reader sees them. The constructor checks, at
compile time (new codes in the E4xx forcing range): `order` is 1 or 2; the
spline's degree is at least 1 (the Greville abscissae are undefined at degree 0,
so `random_walk` on a degree-0 spline is refused); `start` has `order` entries,
the mid-window level and, for order 2, slope; `innovations` is an indexed family
with exactly `n_basis − order` cells; and each innovation is `real`, with prior
exactly `normal(mu = 0, sigma = 1)`. That last check keeps the name honest: a
different innovation prior is a different model, which the user can still write
as explicit coefficient expressions. The `quantities {}` block can report the
curve, `beta_t = log_beta` (a series, language spec §16.4).

**Under IF2 and the optimizers the scale must be fixed.** They read no prior
(`docs/camdl-inference-spec.md`; roles §6.4), and the likelihood depends on (τ,
z) only through c, so an estimated τ beside estimated z is a ridge with no
maximum. `fit run` refuses an estimated `random_walk` scale under `if2` and the
NLopt methods, naming the scale and the remedy: fix τ (the fit is then an
unpenalized maximum likelihood over an affine reparameterization of c), or
estimate the plain coefficients. The refusal lives in the fit-config validation
next to `validate_hierarchical_leaf_boxes` (`fit/config_v2.rs:2604`), and moves
into `resolve_roles` with roles Stage 2 as a new
`RoleError::RandomWalkScaleUnderLikelihood` (§7).

### 6.4 The recommended prior touches no high-risk file

Every prior in §6.3 is a `Prior::Fixed` scalar density, summed by the existing
per-coordinate code. The PGAS hierarchical refusal (`pgas.rs:3775-3784`) is not
reached and stays. Stage 5 therefore changes the expander, the dimension checker
(innovations and start are dimensionless, scale positive), and the fit config
validation.

### 6.5 The improper null space never reaches the sampler

In (C) the null-space coordinates are the mid-window parameters b₀ and b₁, each
with a proper prior, so the joint prior over everything the sampler moves is
proper by construction. A user who omits a prior on b₀ gets the model's ordinary
treatment of a prior-less parameter — a flat density over its declared box —
which is proper because the box is finite (§7.2).

### 6.6 The centred realization is a separate, conditional stage that changes `pgas.rs`

If V4 (§8) shows (C) below threshold in the informative regime, Stage 6 builds
(A). Its seam is one function the four PGAS sites call, replacing their inline
loops — the shared substrate, so the sites cannot drift:

```rust
// sim/src/inference/prior.rs
pub trait JointPrior: Send + Sync {
    /// Positions in the estimated-parameter vector this term reads.
    fn members(&self) -> &[usize];
    fn log_density(&self, theta: &[f64]) -> f64;
    /// ∂ log p / ∂θ_m for each member, natural scale, accumulated into `out`.
    fn grad(&self, theta: &[f64], out: &mut [f64]);
}
pub struct RandomWalkPrior { members: Vec<usize>, order: u8, scale: ScaleRef,
                             start: Vec<Prior> }

// pgas.rs: the one prior authority
pub(crate) fn log_prior_and_grad_z(priors: &[Prior], joint: &[Box<dyn JointPrior>],
    specs: &[EstimatedParam], theta: &[f64], z: &[f64]) -> (f64, Vec<f64>);
```

The exact sites in `pgas.rs`: the NUTS target loop (`:4287-4303`) adds the joint
term and its z-gradient (chain rule through `transform_deriv`); the
MH-within-Gibbs move (`:4444-4446`) adds the joint term's change when coordinate
i moves (O(r) banded terms); the chain-start (`:4064-4069`) and probation
(`:4544-4548`) sums use the new function. PMMH's prior sum gains the same term.
The IR carries the joint prior as a model-level list; that is a second
`ir/VERSION` bump. Stage 6 lands with its own note comparing it to Stage 5 on
the same fixtures.

## 7. Under parameter roles the coefficients are an estimated indexed family

### 7.1 Declaring the family

An indexed parameter compiles to one scalar per cell (`c_k1 … c_k8`, measured
§3), and the roles partition is keyed by scalar names; a declaration naming the
family `c` is refused as an unknown name (roles §4.5). Listing K entries in
`[estimate]` works today and under roles, unchanged. D5 asks whether to amend
roles §4.5 so that `[estimate] c = { … }` declares every cell of an indexed
family that has no scalar of that name, with shared settings, expanded in the
partition and recorded per cell.

### 7.2 Bounds, boxes and transforms

A coefficient family declares finite model bounds, so the design does not depend
on the `real` fallback being fixed on main (§3):
`c[knot] : real 'ratio in [-8, 2]` for a log-rate per day (β between 3·10⁻⁴ and
7.4). A finite box gets the scaled-logit transform (`runner.rs:1167-1198`,
verified), which is near-linear in the interior and keeps the sampler in the
box. For the innovations, `in [-8, 8]` cuts 2·Φ(−8) ≈ 10⁻¹⁵ of the N(0, 1) mass
— negligible, and not a hierarchical leaf, so `validate_hierarchical_leaf_boxes`
does not apply. Once the `real` support is (−∞, ∞) on main, the innovations
should drop their box and take the identity transform; V4 is repeated then
(§10).

### 7.3 Starts

Starts come from `--starts` or `[estimate].start` (roles §5.3). A shared scalar
start for a coefficient family is a constant curve, a sensible start. The
`from_prior` rule draws inside the box (`e6867779`): good for innovations, poor
for flat-prior coefficients, whose box-wide draws are wild curves. The
validation uses `--starts from_mle=` an IF2 fit for the plain-coefficient arm
and `from_prior` for the random-walk arm.

### 7.4 Hierarchical boxes

The boxed-hierarchical-leaf refusal (`9ccb2899`) does not arise in the
recommended design, which has no hierarchical prior. Realization (B) would make
every coefficient a hierarchical leaf, refused under PMMH whenever its box cuts
the normal's support — every finite box — which is one more reason against it.

## 8. Validation decides "proven"; the branch lands only when every row passes

Each harness is a committed, named target; every artifact it writes is produced
by a rule. The synthetic study lives in `tests/recovery/cases/spline_seir/` with
its Makefile targets; figures and tables go into a dated note under
`docs/dev/notes/`, which the merge commit cites (agent verification conventions,
"A new inference method merges with the note that measures it").

Two terms used below. **Bulk-ESS** is the effective sample size of the
rank-normalized draws (Vehtari et al. 2021): how many independent draws the
chains are worth for estimating the centre of the posterior. **Coverage** is the
fraction of datasets whose 90% central posterior interval contains the truth.

### 8.1 The synthetic outbreak

A chain-binomial SEIR with daily steps (dt = 1 day), N₀ = 10⁶, σ = 0.1 and γ =
0.125 per day (fixed), daily onsets observed as
`neg_binomial(mean = rho * projected, r = 10)` over 182 days, generated with E₀
= I₀ = 10 and ρ = 0.5. The true reproduction number is a smooth decline,

R(t) = 0.7 + 1.5 / (1 + exp((t − 90)/10)), β(t) = γ R(t),

from 2.2 to 0.7. Deterministically it gives a peak of about 107 onsets/day near
day 101 and about 8,750 onsets in total, about 28/day at day 182 (measured: a
15-line Euler integration in a scratch script; Stage 4 replaces it with the
committed generator). The truth is a logistic curve outside every spline space,
so recovery includes approximation error.

The generating model is the fitted model with the spline replaced by an
`interpolated 'per_day` forcing read from `beta_true.tsv`, written by a make
target; a test asserts the two models' IR differ only in that forcing. Datasets
are 100 seeds of `camdl simulate --obs-dir`.

**The window ends two weeks before the data** (§4.3): [0, 168], so the last
coefficient is informed by infections whose onsets fall inside the data
(incubation about 10 days). Past 168 the forcing is held. K ∈ {8, 12, 20}
coefficients means 5, 9 and 17 uniform intervals (h = 33.6, 18.7, 9.9 days). K =
12 and K = 20 put h at or below the generation interval and carry the
assertions; K = 8 is reported as a coarse reference.

**Arms.** Each K is fitted (i) with plain coefficients, flat priors in
`[-8, 2]`, and (ii) with `random_walk(order = 2)` and the calibrated priors of
§6.1: mid-window level log β ~ N(ln(1.2γ), 0.6²), mid-window slope b₁ ~ N(0,
0.005²) per day, and τ ~ exponential(λ_h) with λ = 4.03, 9.73, 25.26 for K = 8,
12, 20 (U_h = 0.744, 0.308, 0.119), τ boxed to [10⁻⁶, 5U_h], which leaves 3·10⁻⁷
of the prior mass outside. Both arms fix E₀, I₀ and ρ. A third arm, (iii), is
(ii) at K = 12 with E₀, I₀ and ρ estimated — `E ~ poisson(rate = E0)`,
`I ~ poisson(rate = I0)` (the law form PGAS can score; a plain `init` assignment
is refused, `pgas.rs:3683`), E₀, I₀ ~ log_normal(ln 10, 1), ρ ~ beta(5, 5) —
because in a real Ebola fit ρ·I₀ is a near-ridge and the first coefficient
trades off against I₀; a spline proven only with them fixed is not proven for
that use.

### 8.2 The checks and their thresholds

| Check                 | What it establishes                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Pass                                                                                                                                                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V1 basis values       | `bspline::span`/`basis` and both forcings agree with three independent references: scipy's `BSpline`, R's `splines::splineDesign`, and analytic anchors — at K = p + 1 a clamped spline on one interval is the Bernstein basis C(p, k) uᵏ(1 − u)^{p−k}, which shares no knot construction with any library. Uniform, explicit, anchored-resolved and periodic knots, degrees 0–5, 2,000 points each including knots, window ends and points outside                             | max abs error ≤ 10⁻¹²; partition of unity ≤ 10⁻¹⁴; fixtures (scipy, R, and the existing pomp periodic one) regenerated by `make fixtures-bspline`, committed, CI offline                                                                                                                          |
| V2 gradients          | central finite differences of `complete_data_loglik_grad` in the `gradient_check.rs` harness with step ε = 10⁻⁵·max(\|θ\|, 1) — not the harness's `max(10⁻⁵\|θ\|, 10⁻⁸)` (`gradient_check.rs:653`), which is far too small for log-scale coefficients near 0 — for: identity link, `link = log`, user-written `exp`, periodic spline, a constant `lag`, a spline inside an observation mean, `random_walk` (z, b₀, b₁, τ), at times inside, at a knot, at a and b, and beyond b | relative error < 10⁻⁴ for every parameter                                                                                                                                                                                                                                                         |
| V3 maximum likelihood | (i) the exact maximum: the same model on the ODE backend with NB observations, maximized by `nl-bobyqa` and independently by `nl-sbplx`, 8 starts each. (ii) IF2 on chain-binomial (IF2 on the ODE backend is refused as an inference method), its estimate scored by 10 independent particle-filter runs of 2,000 particles, against the same scoring at θ_proj, the least-squares projection of the true log β onto the basis, and at the ODE maximum's coefficients          | (i) the two optimizers agree within 0.05 nats for each K; (ii) mean ℓ̂_PF(θ̂_IF2) ≥ max(mean ℓ̂_PF(θ_proj), mean ℓ̂_PF(θ_ODE)) − 2 standard errors                                                                                                                                                    |
| V4 mixing             | PGAS+NUTS, 4 chains, 1,000 warm-up + 2,000 sweeps, dense mass-matrix adaptation (recorded in the config), on dataset 1, every arm; bulk-ESS per second of wall time for the worst coefficient or innovation, and for τ                                                                                                                                                                                                                                                          | R̂ ≤ 1.01 and bulk-ESS ≥ 400 for every sampled parameter in the K = 12 and K = 20 random-walk arms and arm (iii); worst-coordinate ESS/s at K = 12 random walk ≥ 0.2 × that of the two-rate step model (`if t < 90 then b1 else b2`, the workaround available today) on the same data and settings |
| V5 coverage and width | 100 datasets; 90% central intervals for log β(t) at the 26 times t = 7, 14, …, 182, computed from the draws by a script that evaluates the spline with scipy (not camdl's evaluator); arms (ii) at K = 12 and K = 20, and arm (iii)                                                                                                                                                                                                                                             | each asserted arm: pooled coverage over the interior times 35–168 and all datasets in [0.85, 0.95]; the early times t ≤ 28 (where I₀ confounds β) and the held times 175, 182 each reported separately and each ≥ 0.80; mean interval width at 35–154 ≤ 0.5 × the prior-predictive width there    |
| V6 prior lowering     | sample (b₀, b₁, z, τ) from their priors, evaluate the compiled coefficient expressions, recompute the weighted second divided differences                                                                                                                                                                                                                                                                                                                                       | equal z to 10⁻¹²: the lowering is the random walk it names; and coefficients linear in ξ give a curve linear in t to 10⁻¹²                                                                                                                                                                        |
| V7 prior predictive   | 4,000 draws of the random-walk prior per K from the calibration step (§6.1), R(t) over [0, 182]                                                                                                                                                                                                                                                                                                                                                                                 | two-sided: at every t the 95% band of R(t) contains [0.5, 3] and lies inside [0.1, 20]; the committed `rw_prior_calibration.tsv` reproduces the table of §6.1                                                                                                                                     |
| V8 calibration        | simulation-based calibration (Talts et al. 2018) on a small model where it is affordable: `X --> D @ f(t) * X` with X₀ = 10⁵, `f` a log-link K = 12 random-walk spline, daily `poisson` observations of the flow; 200 prior draws, PGAS+NUTS, 99 thinned posterior draws each (so the 100 possible ranks fall evenly into 20 bins); test quantities b₀, b₁, τ, each zₖ, and the curve values f(tⱼ) at 10 times                                                                  | rank histograms in 20 bins pass a χ² test at α = 0.01/d, d the number of test quantities (Bonferroni)                                                                                                                                                                                             |

**Power of V5.** Treat the 100 datasets as the independent units and, in the
worst case, the 26 times within a dataset as perfectly correlated, so the pooled
coverage has the binomial standard deviation of n = 100 (inferred; a real
correlation below one only tightens it). That worst case is the valid one to
plan for. The band [0.85, 0.95] then passes, by exact binomial probabilities
(computed by the statistics re-review), a true coverage of 0.90 with probability
0.936, of 0.85 with 0.568, of 0.80 with 0.129 and of 0.75 with 0.011; and it
fails over-coverage, passing a true 0.97 with probability 0.182. The width row
stops a prior-dominated, over-wide posterior from passing on coverage alone.
Cost: 3 asserted arms × 100 datasets × 4 chains, an offline run, not CI.

SBC on the SEIR itself is out of scope for cost; V5 is calibration at one truth,
which `[synthetic]` already supports (`docs/camdl-run-spec.md` §11.5). The
thresholds are fixed before any fit runs; D7 asks the maintainer to confirm
them.

**Proven** means V1, V2, V6 green in `make test`, and V3–V5, V7, V8 meeting
their rows in the note, at the branch head that merges. If V4's efficiency row
fails while every correctness row passes, Stage 6 is built and V4 rerun on it.

## 9. Six stages, each green alone

Each stage ends on a green `make test`; each named test is mutation-checked
(revert the source change, keep the test, confirm it fails). Every commit
re-keys stored runs through the engine version (§4.2), so the run-identity
column says what changes beyond that.

| Stage | Content                                                                                                                                                                                | IR bump                                               | High-risk files             | Run identity beyond the engine version                                     |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------- |
| 1     | `bspline.rs` core; periodic evaluator on it; live periodic coefficients; `first_param_read` arm                                                                                        | no                                                    | none                        | none: IR and hash unchanged                                                |
| 2     | `bspline` kind and `link` (OCaml types, serde, expander, dimcheck, inspect, lint; Rust IR, compiled form, `obs_anchor`, `ir_hash` arms); `make fixtures-bspline` for scipy, R and pomp | yes (one pending bump; about 118 IR files move, §5.6) | none                        | every model re-keys (version string); `link` hashed                        |
| 3     | `forcing_basis`; autodiff rule for both spline kinds; E600 lifted for coefficients; stale refusal messages updated; every exhaustive arm                                               | same pending bump                                     | none (§5.3)                 | models with spline-coefficient parameters gain `rate_grad` entries, hashed |
| 4     | the validation harness: recovery case, generators, scripts, V3–V5 on the plain-coefficient arms, the note                                                                              | no                                                    | none                        | none                                                                       |
| 5     | `random_walk` constructor, its checks, the IF2/NLopt scale refusal; V4–V8 on the random-walk arms                                                                                      | no                                                    | none                        | none: lowers to existing expressions                                       |
| 6     | conditional on V4 (D1): centred `JointPrior` seam                                                                                                                                      | second bump                                           | `pgas.rs` (§6.6), `pmmh.rs` | every model re-keys again                                                  |

**Stage 1.** Compiler unchanged, so a parameter in a periodic coefficient is
still E600 at `camdlc`; the runtime half is tested with hand-built IR. Tests:
the scipy and pomp oracles unchanged at 10⁻¹²; `partition_of_unity` and
`basis_matches_cox_de_boor_recursion` (the old recursion kept as a test oracle
only);
`structural_coefficient_rejection.rs::periodic_spline_coef_param_is_rejected`
becomes `…_builds_live`, asserting the value moves with the parameter; the
`first_param_read` arm by a refusal test of an intervention amount reading a
spline-coefficient forcing under PGAS. `eval_ab` compares two evaluators inside
one build and cannot measure this change, so a dedicated microbenchmark,
`benches/spline_eval.rs`, times the old and new periodic evaluators side by
side; its before/after is pasted in the commit.

**Stage 2.** Tests: V1; compile and simulate each new golden; the anchored
window resolves and its inversion is refused; a window not covering the run is
held, with the stderr line; `link = log` with `'per_week` equals `'per_day`
scaled by 1/7 to 10⁻¹²; parameter coefficients still E600; and the red→green
no-collision test of §5.6 — two models differing only in `link` must have
different IR hashes, red with the `link` hash removed.

**Stage 3.** Tests: V2; a `forcing_basis` outside a gradient map, or with a bad
index, is refused by `ir::validate`; a PGAS+NUTS smoke fit with spline
coefficients runs past the preflight; a parameter in a window end is still E600,
and its message no longer lists the spline basis among things a parameter cannot
drive. The `URStructuralForcing` text (`autodiff.ml:360-370`), the
`UnsupportedReason::StructuralForcing` message (`deriv.rs:73-75`) and the
runtime `eval_structural` message (`compiled_model.rs:736-741`) all name
`periodic_spline` coefficients as frozen; each is rewritten to name only knots,
windows and periods.

**Stages 4 and 5** are measurement stages; their code is the harness and the
constructor. **Stage 6** lands only if D1's condition fires.

## 10. Sequencing with the typed-constants change and the roles stages

The planned IR change that moves typed `let` constants into a `constants`
section precedes roles Stage 2 and lands on main first. This branch does not
wait for it and does not join it: its schema changes stay on the branch until §8
passes. At each rebase onto a main that has bumped `ir/VERSION`, the branch's
single pending bump is renumbered to main's version plus one step and its golden
commit regenerated (the `golden-update` skill); the branch never carries two
pending bumps for Stages 2–3. The two changes are orthogonal in content: a
coefficient is an expression, and whether a constant it names lives in the
parameter list or the `constants` section does not change the spline code. Stage
6, if built, is a second bump after the first has landed.

Roles Stage 1 (types, no behaviour change) does not interact. Roles Stage 2
brings `resolve_roles`, into which the random-walk scale refusal moves. The
`real` support fix is landing separately on main; once it is there, V4 is rerun
with unboxed innovations and the note records both.

**The cost of renumbering.** A rebase across a main bump regenerates about 101
generated IR files and hand-edits the 17 in `ir/golden/` (§5.6). The branch
therefore rebases at stage boundaries and once before merge, not continuously
(D6).

## 11. Decisions

**D1 — How the smoothness prior is realized.** The question is which coordinates
the sampler moves. (a) Non-centred constructor now (§6.2 C): no high-risk
change, no IR change, the same posterior over the curve, standard normal
innovations that suit weakly informative stretches; the risk is a curved z–τ
ridge where the latent path is informative, which shows as low ESS/s in V4. (b)
Centred GMRF prior type now (§6.2 A): the sampled coordinates are the
coefficients themselves, efficient where data are informative, but it needs a
new non-separable prior seam at four sites in `pgas.rs` and one in PMMH, a
second IR bump, and it has the funnel where data are weak. (c) Interweave both
in each sweep (ASIS, Yu & Meng 2011): robust across the two regimes, but it
needs (b)'s seam plus a second θ-move per sweep in `pgas.rs`. (d) Lift the
hierarchical refusal (§6.2 B): the general mechanism, but centred, verbose (one
declaration per coefficient), and every coefficient becomes a hierarchical leaf
subject to the box refusals. User-facing difference: under (a) the fit summary
lists innovations, level, slope and scale, and the curve is read through
`quantities {}`; under (b) it lists coefficients. Recommendation: (a), with
Stage 6 building (b) — and (c) on top of it if (b) alone does not meet V4 — only
if V4's efficiency row fails. Confidence: leaning; the efficiency comparison is
exactly what V4 measures.

**D2 — A `link = log` option on spline forcings.** Without it, a positive rate
is `unit * exp(spline)` with a typed constant carrying the unit (measured: bare
`exp` of a forcing is E300, plus E301 for a dimensioned forcing), and unit
scaling such as `'per_week` must be applied by hand. With it, the forcing
carries its unit and the rate reads `log_beta * S * I / N`; the cost is one enum
field on two kinds, hashed, and the additive-scaling rule of §4.4.
Recommendation: offer it. Confidence: solid.

**D3 — Knots at the window ends, and the difference the prior penalizes.** The
RW2 prior must have log-linear curves as its null space, or it penalizes a real
exponential growth or decline at the ends, where the forecast value is read. (a)
Clamped knots with divided differences over the Greville abscissae (§6.1): f(a)
= c₁ and f(b) = c_K exactly; one rule serves uniform and explicit breakpoints;
interior penalty identical to the textbook RW2; the end weights are a modelling
choice (Brownian scaling) that V6 checks. (b) Unclamped, equally spaced knots
extended p intervals past each end (Eilers & Marx 1996), plain index
differences, and the forcing held at f(a), f(b): the textbook P-spline, a
homogeneous penalty with the correct null space and no weights, but p
coefficients at each end supported mostly outside the data; f(b) is a blend of
the last p + 1 coefficients; knot positions outside the data; and `random_walk`
must be refused on explicit breakpoints, whose spacing is uneven. Either choice
fixes every coefficient's meaning, so changing it later re-keys and
re-interprets every fit. Recommendation: (a). Confidence: leaning.

**D4 — The forcing outside its window.** Hold (recommended): continuous,
conservative for forecasts, matches `piecewise` and linear `interpolated`, and
composes with `last_obs` forks; the window should end at `last_obs` minus the
infection-to-report delay, so the held value is a coefficient the data inform
(§4.3). Refuse: forces every window to cover the simulation, which breaks
forecasting runs past `last_obs`. Polynomial or linear extrapolation: an
unsupported trend in log β, exponential in β. A user-visible consequence of
hold: a forecast past the window uses the last estimated level unless a scenario
says otherwise. Confidence: solid.

**D5 — Declaring an indexed family in `[estimate]`.** Today and under roles as
accepted, a K = 20 family is 20 `[estimate]` lines (18 innovations under
`random_walk`). Amending roles §4.5 to accept the family name, when no scalar of
that name exists, expands to the cells with shared bounds, transform and start,
recorded per cell in the partition, and still refuses a family name that
collides with a scalar. It touches `resolve_roles`, not the samplers.
Recommendation: amend, in roles Stage 2. Confidence: leaning; it is a roles
decision this proposal motivates rather than needs.

**D6 — The IR bump's timing.** (a) One pending bump on this branch, landing when
proven (recommended). Each rebase onto a main that has bumped `ir/VERSION`
renumbers it, which regenerates about 101 generated IR files and hand-edits the
17 frozen ones in `ir/golden/` (§5.6); so the branch rebases at stage boundaries
and once before merge, not continuously. (b) Bundle the schema additions with
the typed-constants bump to save a churn — which would put unproven surface on
main and contradict the branch's purpose. Confidence: solid.

**D7 — The acceptance thresholds of §8.2.** The numbers that decide "proven":
three independent basis references at 10⁻¹²; gradient relative error < 10⁻⁴ at
step 10⁻⁵·max(|θ|, 1); two optimizers agreeing within 0.05 nats and IF2 within
two standard errors of the better of the projected truth and the ODE maximum; R̂
≤ 1.01 and bulk-ESS ≥ 400, including the arm with E₀, I₀ and ρ estimated; ESS/s
≥ 0.2× the two-rate step model; pooled interior coverage in [0.85, 0.95] over
100 datasets for arms (ii) at K = 12, 20 and (iii), with the early and held
times each ≥ 0.80 and interior width ≤ half the prior's; a prior-predictive band
of R(t) containing [0.5, 3] and inside [0.1, 20] at every t, met by the
calibrated priors (§6.1 table); SBC χ² at α = 0.01/d. Loosening any of them
later is lowering the bar, so they are fixed now. Recommendation: accept as
written. Confidence: leaning; the ESS/s ratio is the least grounded, set so the
spline costs at most five times the effective samples per second of a model with
a sixth of the parameters.

## 12. Out of scope, each with its reason

- **Penalized maximum likelihood (a MAP estimate under IF2).** IF2 reads no
  prior; adding a penalty to its likelihood is an IF2 change (high-risk) for a
  use V3 does not need.
- **Estimated knot positions or period.** Their derivatives move the basis
  itself; they stay structural (E600).
- **Degree above 5, coincident interior knots.** Neither is needed for smooth
  rates, and both complicate the stability argument of §4.1.
- **A span cache per (forcing, t).** An optimization after a profile (§5.5).
- **SBC on the SEIR study.** Cost (§8.2); V8 covers the prior and gradient
  machinery on a small model.
- **A lint for an estimated intercept beside a free-level log spline** (§4.4).
  The non-identifiability is documented in the language spec section Stage 2
  adds; a syntactic lint belongs with the model-lints work.
- **The `[0, ∞)` box for an unbounded `real` and the stale, untested
  `ir/schema.json`** (§3). Both are being fixed separately on main; nothing here
  depends on either staying as it is.
- **Splines over a stratification dimension** (one curve per province sharing a
  τ). Expressible as one forcing per stratum today; a stratified forcing kind is
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
  Algorithms A2.1 (span) and A2.2 (nonzero basis functions) (§4.1).
- Eilers, P. H. C. & Marx, B. D. (1996). Flexible smoothing with B-splines and
  penalties. _Statistical Science_ 11(2):89–121. The difference penalty (§6.1).
- Lang, S. & Brezger, A. (2004). Bayesian P-splines. _Journal of Computational
  and Graphical Statistics_ 13(1):183–212. The random-walk prior on coefficients
  (§6.1).
- Rue, H. & Held, L. (2005). _Gaussian Markov Random Fields: Theory and
  Applications_. Chapman & Hall/CRC. Ch. 3, intrinsic GMRFs: rank, null space,
  normalizer (§6.1).
- Papaspiliopoulos, O., Roberts, G. O. & Sköld, M. (2007). A general framework
  for the parametrization of hierarchical models. _Statistical Science_
  22(1):59–73. Centred versus non-centred (§6.2).
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A. & Gelman, A. (2018).
  Validating Bayesian inference algorithms with simulation-based calibration.
  arXiv:1804.06788 (V8).
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. & Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: an improved R̂ for assessing
  convergence of MCMC. _Bayesian Analysis_ 16(2):667–718 (V4).

- Yu, Y. & Meng, X.-L. (2011). To center or not to center: that is not the
  question — an ancillarity–sufficiency interweaving strategy (ASIS) for
  boosting MCMC efficiency. _Journal of Computational and Graphical Statistics_
  20(3):531–570. Interweaving (D1).

Context:

- Betancourt, M. & Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical
  models. In _Current Trends in Bayesian Methodology with Applications_, CRC
  Press; arXiv:1312.0906. The funnel (§6.2).
- Simpson, D., Rue, H., Riebler, A., Martins, T. G. & Sørbye, S. H. (2017).
  Penalising model component complexity: a principled, practical approach to
  constructing priors. _Statistical Science_ 32(1):1–28. The prior on τ (§6.1).
- Sørbye, S. H. & Rue, H. (2014). Scaling intrinsic Gaussian Markov random field
  priors in spatial modelling. _Spatial Statistics_ 8:39–51. Scale of τ (§6.1).
- Lindsten, F., Jordan, M. I. & Schön, T. B. (2014). Particle Gibbs with
  ancestor sampling. _Journal of Machine Learning Research_ 15:2145–2184.
- Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler. _Journal of Machine
  Learning Research_ 15:1593–1623.
- Ionides, E. L., Nguyen, D., Atchadé, Y., Stoev, S. & King, A. A. (2015).
  Inference for dynamic and latent variable models via iterated, perturbed Bayes
  maps. _PNAS_ 112(3):719–724.
