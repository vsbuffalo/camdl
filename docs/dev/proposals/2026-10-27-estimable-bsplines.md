---
date: 2026-10-27
status: proposed — decisions D1–D8 (§11) await the maintainer; implementation follows §9 on `feat/bsplines`, which stays off main until §8's thresholds are met
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
`E301` (the argument of `exp` must be dimensionless). The rate must therefore be
`beta0 * exp(logb)`, and an estimated `beta0` is exactly non-identified against
a common shift of the coefficients (§4.4).

**An unbounded `real` coefficient gets the wrong search box today.** With no
`in [lo, hi]` and no fit-config bounds, the box falls back to `[0, ∞)` for every
kind but `probability` (verified, `rust/crates/cli/src/fit/runner.rs:1265-1267`;
`kind_support` returns `None` for `real`, `params_resolver.rs:118-126`), and a
box with an infinite end gets no transform (`runner.rs:1167-1173`). A log-scale
coefficient is usually negative. The parameter-roles proposal corrects the
fallback to (−∞, ∞) for `real` in its Stage 2 (roles §6.4); until then every
coefficient family must declare finite bounds (§7.2).

**PGAS refuses hierarchical priors outright** (verified,
`rust/crates/sim/src/inference/pgas.rs:3775-3784`): the NUTS gradient of a
hierarchical leaf is stubbed to −∞ (`prior_log_density_and_grad_z`,
`:3557-3570`). A random-walk prior written as a chain of hierarchical leaves,
`c2 ~ normal(mu = c1, sigma = tau)`, therefore cannot run under PGAS.

Two drifted artifacts surfaced while checking the above. `ir/schema.json`, which
`.claude/rules/ir-schema.md` calls the contract, lists four forcing kinds and
omits `fourier` and `periodic_spline` (verified, `ir/schema.json:374-437`); no
test reads it. The periodic-spline scipy and pomp fixtures have generator
scripts (`scripts/gen_periodic_bspline_*_fixture.*`) but no make target
(verified: `grep -n bspline Makefile` has no match). Stage 2 fixes both (§9).

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
heap allocations in today's periodic evaluator. Stage 1 measures the ratio with
the existing `eval_ab` bench and pastes it in its commit.

**Numerical stability.** Every step of the triangle combines nonnegative
quantities with nonnegative weights — `left` and `right` are nonnegative within
the span, and the N[r] stay nonnegative and sum to one at every level — so there
is no subtractive cancellation (de Boor 1978, ch. IX). This is the reason to
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
amount that reads a spline-coefficient forcing would escape the D3 refusal in
`refuse_unscored_parameters` (`pgas.rs:3683`). The arm is in
`compiled_model.rs`, not in a high-risk file.

Floating-point rounding of `periodic_spline` values changes in the last bits,
because the summation order differs. Run identity does not hash the engine
(`engine_version` is recorded, not keyed: inferred from
`rust/crates/runid/src/record.rs:162-175`), so a stored run keeps its identity
and is served as before; D8 decides whether that is acceptable.

### 4.3 The `bspline` kind: a clamped basis over a window, held constant outside it

**DSL.** Two knot forms, mutually exclusive:

```camdl
forcing {
  # Uniform knots: n_basis − degree equal intervals over the window.
  log_beta : bspline 'per_day {
    window  = [date("2025-08-04"), last_obs]
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

**Ends: clamped** (D3). The knot vector repeats a and b p + 1 times. Every basis
function lies inside the window, f(a) = c₁ and f(b) = c_K exactly, and explicit
breakpoints need no extension beyond the data. The unclamped alternative of
Eilers & Marx (1996), equally spaced knots extended p intervals past each end,
gives all basis functions the same shape — so a difference penalty treats every
coefficient alike — but its outermost coefficients are informed only by the few
days their bump overlaps the window, and its knot positions outside the data
have no meaning to a modeller.

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

### 5.6 The IR bump and its goldens

Two schema additions — the `bspline` kind with `link` (Stage 2), and
`forcing_basis` (Stage 3) — share one pending `ir/VERSION` bump on the branch,
made in Stage 2 (0.40 → 0.41 at today's main; renumbered at rebase, §10). The
envelope embeds the version in every golden (verified:
`ocaml/golden/
sir_basic.ir.json` begins `{"ir_version":"0.40"`), so all 55
`*.ir.json` goldens under `ocaml/golden/` move in Stage 2's atomic commit, with
no change besides the version string for a model without a spline (`link` is
omitted when identity). `ir/golden/` is the separate frozen set and is not
regenerated. New fixtures add goldens: `bspline_uniform.camdl`,
`bspline_explicit_anchored.camdl`, `bspline_loglink.camdl`, and in Stage 3
`bspline_param_coefs.camdl` and `periodic_spline_param_coefs.camdl`, whose
`rate_grad` carries `forcing_basis` nodes. Each new golden carries
first-scenario parameters and a trajectory baseline. `ir/schema.json` gains
`fourier`, `periodic_spline`, `bspline`, `forcing_basis` and the `constant`
interpolation method it is missing.

## 6. The smoothness prior is a random walk on the coefficients, realized non-centred

### 6.1 What the prior is

The Bayesian P-spline prior (Lang & Brezger 2004, building on the difference
penalty of Eilers & Marx 1996) is a random walk of order r on the coefficient
vector:

Δʳcₖ ~ N(0, τ²) independently, k = r + 1, …, K.

As a GMRF (Rue & Held 2005, ch. 3) its density is

p(c | τ) ∝ τ^{−(K−r)} exp(−‖D_r c‖² / (2τ²)),

where D_r is the (K − r) × K r-th difference matrix and Q = D_rᵀD_r/τ² the
precision. Q has rank K − r: its null space is the polynomials of degree below r
in the index — constants for RW1, constants and linear trends for RW2. Two
consequences are load-bearing. The prior is improper along the null space, so it
must be completed there; and the normalizing power is τ^{−(K−r)}, not τ^{−K},
which matters as soon as τ is estimated (using K biases τ downward).

**Completion.** Give the null-space coordinates proper priors: for RW1 a prior
on the level c₁; for RW2 on the level c₁ and the first slope c₂ − c₁. The
completed density is

p(c | τ) = p(c₁) [p(c₂ − c₁)] Πₖ N(Δʳcₖ; 0, τ²),

proper, with log-density and gradient

log p = log p(c₁) [+ log p(c₂ − c₁)] − (K − r) log τ − ‖D_r c‖²/(2τ²) + const,
∂/∂c = (null-space terms) − D_rᵀD_r c / τ², ∂/∂τ = −(K − r)/τ + ‖D_r c‖²/τ³.

D_rᵀD_r is banded with bandwidth r, so both are O(K).

**τ is estimated, with a proper prior that shrinks toward smoothness.** τ = 0 is
the polynomial of degree r − 1 (a constant log-rate for RW1, a log-linear trend
for RW2), so a prior with mode at zero — an exponential on τ, the
penalized-complexity prior of Simpson et al. (2017), or a half-normal — states
"no more wiggle than the data ask for". The scale of τ depends on the knot
spacing h: for a curve whose second derivative is white noise, second
differences at spacing h have standard deviation ∝ h^{3/2} (∝ h^{1/2} for RW1,
first differences) (inferred from the integrated-Wiener-process limit; Sørbye &
Rue 2014 make the same point for IGMRFs and scale by the generalized variance).
A prior on τ is therefore stated per knot spacing, and the validation configs
scale it by (h/h₀)^{r−1/2} so the prior on β(t) is comparable across K (§8).

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
sampler moves standardized innovations zₖ ~ N(0, 1), the level b₀ (and for RW2
the slope b₁), and τ, each with an ordinary scalar prior the samplers already
handle. The coefficients are deterministic functions of them:

RW1: c₁ = b₀, cₖ = cₖ₋₁ + τ zₖ; RW2: c₁ = b₀, c₂ = b₀ + b₁, cₖ = 2cₖ₋₁ − cₖ₋₂ +
τ zₖ,

in closed form cₖ = b₀ + τ Σ_{i≤k} zᵢ (RW1) and cₖ = b₀ + (k − 1) b₁ + τ
Σ_{i=3}^{k} (k − i + 1) zᵢ (RW2).

The map (b₀, b₁, z) → c, for fixed τ, is linear with Jacobian τ^{K−r}, which is
exactly the GMRF normalizer: (C) induces on (c, τ) the same completed prior as
(A). The posterior over the curve is identical; only the sampler's coordinates
differ. Nothing in the samplers changes, because every prior in (C) is a scalar
prior; the coefficients are coefficient expressions, which §5.2 differentiates
by the chain rule (∂cⱼ/∂zᵢ = τ·(j − i + 1) for RW2, ∂cⱼ/∂τ = Σ (j − i + 1) zᵢ).

**The trade-off is information-dependent, and PGAS sits on both sides of it.**
Papaspiliopoulos, Roberts & Sköld (2007) show the centred form mixes better when
the data are informative about c and the non-centred form when they are not. The
PGAS θ-update conditions on the whole latent path, which is very informative
about β(t) where many infections occur and barely informative in a long tail of
few cases. In the informative regime (C) has its own pathology: with c pinned, z
and τ lie on the ridge z ∝ 1/τ. The dense mass matrix PGAS adapts by default
(`default_dense_mass` = true, verified `fit/config_v2.rs:1725`) captures a
linear correlation but not that curvature. Which regime dominates on an
Ebola-scale outbreak is a measurement, made in Stage 5 (§8, V4). (C) comes first
because it is correct with no high-risk change; (A) is built only if (C) fails
the efficiency threshold (§6.6).

### 6.3 `random_walk(...)` is a coefficient constructor, and the priors stay visible

```camdl
dimensions { innov = [i3, i4, i5, i6, i7, i8] }        # n_basis − order levels

parameters {
  b0        : real 'ratio in [-8, 2]  ~ normal(mu = -1.4, sigma = 1.0)   # log β at a
  b1        : real 'ratio in [-3, 3]  ~ normal(mu = 0.0, sigma = 0.5)    # first slope
  tau_b     : positive 'ratio in [1e-4, 5] ~ exponential(rate = 2.0)
  z[innov]  : real 'ratio in [-8, 8]  ~ normal(mu = 0.0, sigma = 1.0)
}

forcing {
  log_beta : bspline 'per_day {
    window  = [date("2025-08-04"), last_obs]
    n_basis = 8
    link    = log
    coefs   = random_walk(order = 2, start = [b0, b1], scale = tau_b, innovations = z)
  }
}
```

The expander lowers `random_walk(...)` to the K closed-form coefficient
expressions above (`Reduce` sums over `Param` nodes), so the IR contains only
existing expression forms and **no IR change** is needed for the prior. The
priors are declared by the user, with today's syntax, where a reader sees them.
The constructor checks, at compile time (new codes in the E4xx forcing range):
`order` is 1 or 2; `start` has `order` entries; `innovations` is an indexed
family with exactly `n_basis − order` cells; and each innovation is `real`, with
prior exactly `normal(mu = 0, sigma = 1)`. That last check keeps the name
honest: a different innovation prior is a different model, which the user can
still write as explicit coefficient expressions. The `quantities {}` block can
report the curve, `beta_t = log_beta` (a series, language spec §16.4).

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

In (C) the null-space coordinates are the parameters b₀ and b₁, each with a
proper prior, so the joint prior over everything the sampler moves is proper by
construction. A user who omits a prior on b₀ gets the model's ordinary treatment
of a prior-less parameter — a flat density over its declared box — which is
proper because the box is finite (§7.2).

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

Until roles Stage 2 corrects the `real` support, a coefficient family declares
finite model bounds: `c[knot] : real 'ratio in [-8, 2]` for a log-rate per day
(β between 3·10⁻⁴ and 7.4). A finite box gets the scaled-logit transform
(`runner.rs:1167-1198`, verified), which is near-linear in the interior and
keeps the sampler in the box. For the innovations, `in [-8, 8]` cuts 2·Φ(−8) ≈
10⁻¹⁵ of the N(0, 1) mass — negligible, and not a hierarchical leaf, so
`validate_hierarchical_leaf_boxes` does not apply. After roles Stage 2 the
innovations should drop their box and take the identity transform; the
validation is repeated then (§10).

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

### 8.1 The synthetic outbreak

A chain-binomial SEIR with daily steps (dt = 1 day), N₀ = 10⁶, E₀ = I₀ = 10, σ =
0.1 and γ = 0.125 per day (fixed), daily onsets observed as
`neg_binomial(mean = 0.5 * projected, r = 10)` with ρ = 0.5 and r fixed, over
182 days. The true reproduction number is a smooth decline,

R(t) = 0.7 + 1.5 / (1 + exp((t − 90)/10)), β(t) = γ R(t),

from 2.2 to 0.7. Deterministically it gives a peak of about 107 onsets/day near
day 101 and about 8,750 onsets in total, with about 28/day at day 182 (measured:
a 15-line Euler integration in a scratch script; Stage 4 replaces it with the
committed generator). The truth is a logistic curve, deliberately outside every
spline space, so recovery includes approximation error.

The generating model is the fitted model with the spline replaced by an
`interpolated 'per_day` forcing read from `beta_true.tsv`, written by a make
target; a test asserts the two models' IR differ only in that forcing. Datasets
are 40 seeds of `camdl simulate --obs-dir`. K ∈ {4, 8, 12} coefficients means 1,
5 and 9 uniform intervals over [0, 182] (h = 182, 36.4, 20.2 days). Each K is
fitted two ways: plain coefficients with flat priors in `[-8, 2]`, and
`random_walk(order = 2)` with b₀ ~ N(ln 0.25, 1), b₁ ~ N(0, 0.5) and τ ~
exponential(rate = 2·(h/20)^{−3/2}) (prior mean 0.5 at h = 20 days).

### 8.2 The checks and their thresholds

| Check                 | What it establishes                                                                                                                                                                                                                                                                                                                                                                  | Pass                                                                                                                                                                                                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V1 basis values       | `bspline::span`/`basis` and both forcings agree with scipy's `BSpline` (an independent implementation) on uniform, explicit, anchored-resolved and periodic knots, degrees 0–5, 2,000 points each including knots, window ends and points outside                                                                                                                                    | max abs error ≤ 10⁻¹²; partition of unity ≤ 10⁻¹⁴; fixtures regenerated by `make fixtures-bspline` (`uv run --with scipy`), committed, CI offline                                                                                                                                 |
| V2 gradients          | finite-difference check of `complete_data_loglik_grad` in the `gradient_check.rs` harness (central differences, ε = 10⁻⁵·\|θ\|, as the seasonal test at `:576-690`) for: identity link, `link = log`, user-written `exp`, periodic spline, a constant `lag`, a spline inside an observation mean, `random_walk` (z, b₀, b₁, τ), at times inside, at a knot, at a and b, and beyond b | relative error < 10⁻⁴ for every parameter (the harness's existing tolerance)                                                                                                                                                                                                      |
| V3 maximum likelihood | (i) the exact maximum: the same model on the ODE backend, NB observations, `nl-bobyqa` from 8 starts, is the reference; IF2 on the ODE backend must reach it. (ii) on chain-binomial, IF2's estimate scored by 10 independent particle-filter runs (2,000 particles)                                                                                                                 | (i) ℓ(IF2) ≥ ℓ(bobyqa) − 0.5 nats for K = 4, 8, 12; (ii) mean ℓ̂_PF(θ̂_IF2) ≥ mean ℓ̂_PF(θ_proj) − 2 standard errors, θ_proj the least-squares projection of the true log β onto the basis                                                                                           |
| V4 mixing             | PGAS+NUTS, 4 chains, 1,000 warm-up + 2,000 sweeps, on dataset 1, all six arms; bulk-ESS per second of wall time for the worst coefficient (or innovation) and for τ                                                                                                                                                                                                                  | R̂ ≤ 1.01 and bulk-ESS ≥ 400 for every sampled parameter in the K = 8 and K = 12 random-walk arms; worst-coordinate ESS/s at K = 8 random walk ≥ 0.2 × that of the two-rate step model (`if t < 90 then b1 else b2`, the workaround available today) on the same data and settings |
| V5 coverage           | 40 datasets; 90% central posterior intervals for log β(t) at t = 30, 60, 90, 120, 150, computed from the draws by a script that evaluates the spline with scipy (not camdl's evaluator)                                                                                                                                                                                              | K = 8 and K = 12 random-walk arms: at each of the five times, between 31 and 40 of 40 intervals contain the truth (36 expected; 31 is 2.7 binomial SDs below); plain-coefficient arms and K = 4 reported, not asserted                                                            |
| V6 prior lowering     | sample (b₀, b₁, z, τ) from their priors, evaluate the compiled coefficient expressions, recompute Δ²c/τ                                                                                                                                                                                                                                                                              | equals z to 10⁻¹²: the lowering is the random walk it names                                                                                                                                                                                                                       |
| V7 prior predictive   | 1,000 draws of the random-walk prior at each K, β(t) curves over [0, 182]                                                                                                                                                                                                                                                                                                            | reported as a figure; the 95% band of R(t) contains [0.5, 3] at every t (the prior admits the plausible range)                                                                                                                                                                    |
| V8 calibration        | simulation-based calibration (Talts et al. 2018) on a small model where it is affordable: `X --> D @ f(t) * X` with X₀ = 10⁵, `f` a log-link K = 8 random-walk spline, daily `poisson` observations of the flow; 200 prior draws, PGAS+NUTS, 100 thinned draws each                                                                                                                  | rank histograms in 20 bins pass a χ² test at α = 0.01/d (Bonferroni over the d sampled parameters)                                                                                                                                                                                |

SBC on the SEIR itself (200 PGAS fits of 4 chains) is out of scope for cost; V5
is calibration at one truth, which `[synthetic]` already supports
(`docs/camdl-run-spec.md` §11.5). The thresholds are fixed before any fit runs;
D7 asks the maintainer to confirm them.

**Proven** means V1, V2, V6 green in `make test`, and V3–V5, V7, V8 meeting
their rows in the note, at the branch head that merges. If V4's efficiency row
fails while every correctness row passes, Stage 6 is built and V4 rerun on it.

## 9. Six stages, each green alone

Each stage ends on a green `make test`; each named test is mutation-checked
(revert the source change, keep the test, confirm it fails).

| Stage | Content                                                                                                                                                                  | IR bump                                  | High-risk files             |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | --------------------------- |
| 1     | `bspline.rs` core; periodic evaluator on it; live periodic coefficients; `first_param_read` arm                                                                          | no                                       | none                        |
| 2     | `bspline` kind and `link` (OCaml types, serde, expander, dimcheck, inspect, lint; Rust IR, compiled form, `obs_anchor`); `schema.json` repaired; `make fixtures-bspline` | yes (one pending bump; all goldens move) | none                        |
| 3     | `forcing_basis`; autodiff rule for both spline kinds; E600 lifted for coefficients; every exhaustive arm                                                                 | same pending bump                        | none (§5.3)                 |
| 4     | the validation harness: recovery case, generators, scripts, V3–V5 on the plain-coefficient arms, the note                                                                | no                                       | none                        |
| 5     | `random_walk` constructor, its checks, the IF2/NLopt scale refusal; V4–V8 on the random-walk arms                                                                        | no                                       | none                        |
| 6     | conditional on V4 (D1): centred `JointPrior` seam                                                                                                                        | second bump                              | `pgas.rs` (§6.6), `pmmh.rs` |

**Stage 1.** Compiler unchanged, so a parameter in a periodic coefficient is
still E600 at `camdlc`; the runtime half is tested with hand-built IR. Tests:
the scipy and pomp oracles unchanged at 10⁻¹²; `partition_of_unity` and
`basis_matches_cox_de_boor_recursion` (the old recursion kept as a test oracle
only);
`structural_coefficient_rejection.rs::periodic_spline_coef_param_is_rejected`
becomes `…_builds_live`, asserting the value moves with the parameter; the
`first_param_read` arm by a refusal test of an intervention amount reading a
spline-coefficient forcing under PGAS. The `eval_ab` bench before and after is
pasted in the commit.

**Stage 2.** Tests: V1; compile and simulate each new golden; the anchored
window resolves and its inversion is refused; a window not covering the run is
held, with the stderr line; `link = log` with `'per_week` equals `'per_day`
scaled by 1/7 to 10⁻¹²; parameter coefficients still E600 (gradients come in
Stage 3).

**Stage 3.** Tests: V2; a `forcing_basis` outside a gradient map, or with a bad
index, is refused by `ir::validate`; a PGAS+NUTS smoke fit with spline
coefficients runs past the preflight; a parameter in a window end is still E600.

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
brings the `real` support fix (§7.2) and `resolve_roles`, into which the
random-walk scale refusal moves; if it lands before this branch merges, V4 is
rerun with unboxed innovations and the note records both.

## 11. Decisions

**D1 — How the smoothness prior is realized.** The question is which coordinates
the sampler moves. (a) Non-centred constructor now (§6.2 C): no high-risk
change, no IR change, the same posterior over the curve, standard normal
innovations that suit weakly informative stretches; the risk is a curved z–τ
ridge where the latent path is informative, which shows as low ESS/s in V4. (b)
Centred GMRF prior type now (§6.2 A): interpretable sampled coordinates (the
coefficients themselves), efficient where data are informative, but a new
non-separable prior seam at four sites in `pgas.rs` and one in PMMH, a second IR
bump, and the funnel where data are weak. (c) Lift the hierarchical refusal (B):
the general mechanism, but centred, verbose (one declaration per coefficient),
and every coefficient becomes a hierarchical leaf subject to the box refusals.
User-facing difference: under (a) the fit summary lists innovations, level,
slope and scale, and the curve is read through `quantities {}`; under (b) it
lists coefficients. Recommendation: (a), and build (b) as Stage 6 only if V4's
efficiency row fails. Confidence: leaning — the efficiency comparison is exactly
what V4 measures.

**D2 — A `link = log` option on spline forcings.** Without it, a positive rate
is `unit * exp(spline)` with a typed constant carrying the unit (measured: bare
`exp` of a forcing is E300 or E301), and the unit scaling of `'per_week` would
have to be applied by hand. With it, the forcing carries its unit and the rate
reads `log_beta * S * I / N`; the cost is one enum field on two kinds and the
additive-scaling rule of §4.4. Recommendation: offer it. Confidence: solid.

**D3 — Clamped or unclamped ends.** Clamped: all basis functions inside the
window, f(a) = c₁, f(b) = c_K, explicit breakpoints need nothing outside the
data; the difference penalty treats the p boundary coefficients slightly
differently, because their bumps are narrower. Unclamped (Eilers & Marx
extension): uniform basis shapes, a homogeneous penalty, but p weakly identified
coefficients at each end and knot positions outside the data. Changing this
later changes every fitted coefficient's meaning. Recommendation: clamped.
Confidence: leaning.

**D4 — The forcing outside its window.** Hold (recommended): continuous,
conservative for forecasts, matches `piecewise`, composes with `last_obs` forks.
Refuse: forces every model's window to cover the simulation, which breaks
forecasting runs whose horizon passes `last_obs`. Polynomial or linear
extrapolation: an unsupported trend in log β, exponential in β. A user-visible
consequence of hold: a forecast past `last_obs` uses the last estimated level
unless a scenario says otherwise. Confidence: solid.

**D5 — Declaring an indexed family in `[estimate]`.** Today and under roles as
accepted, a K = 12 family is 12 `[estimate]` lines. Amending roles §4.5 to
accept the family name (when no scalar of that name exists) expands to the cells
with shared bounds, transform and start, recorded per cell in the partition, and
still refuses a family name that collides with a scalar. It touches
`resolve_roles`, not the samplers. Recommendation: amend, in roles Stage 2.
Confidence: leaning — it is a roles decision this proposal motivates rather than
needs.

**D6 — The IR bump's timing.** (a) One pending bump on this branch, renumbered
at each rebase, landing when proven (recommended); (b) bundle the schema
additions with the typed-constants bump to save a golden churn — which would put
unproven surface on main and contradict the branch's purpose. Confidence: solid.

**D7 — The acceptance thresholds of §8.2.** The numbers that decide "proven":
gradient relative error < 10⁻⁴; IF2 within 0.5 nats of the exact maximum; R̂ ≤
1.01 and bulk-ESS ≥ 400; ESS/s ≥ 0.2× the two-rate step model; coverage 31–40 of
40 at five times; SBC χ² at α = 0.01/d. Loosening any of them later is lowering
the bar, so they are fixed now. Recommendation: accept as written. Confidence:
leaning — the ESS/s ratio is the least grounded; it is set so the spline costs
at most five times the effective samples per second of a model with a fifth of
the parameters.

**D8 — Rounding changes in stored `periodic_spline` runs.** Stage 1 changes the
last bits of periodic-spline values; stored runs keep their identity because the
engine is not hashed. (a) Accept, and list it under _Run identity_ in the
release notes (recommended: the change is rounding, bounded by V1's 10⁻¹²
against two external oracles); (b) re-key every periodic-spline run by a
hash-version bump, which recomputes results that are equal to within rounding.
Confidence: leaning.

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
  Mathematical Sciences 27. Stability of the recurrence; the periodic
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
