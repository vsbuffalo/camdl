---
date: 2026-10-29
status: proposed — the direction is approved by the maintainer and by an external statistician; D1–D10 (§14) carry recommendations awaiting the maintainer's decision
area: inference setup / sim inference transforms / priors
issues: closes gh#470; carries gh#1001 (commit series ii); gh#1002 is the optimiser follow-up (D5)
related:
  - 2026-10-26-parameter-roles.md (accepted; §4.4 states what this amends in its §6.2 and §6.4)
---

# A parameter's transform is a function of its domain alone, and every method shares it

All code references are against `origin/main` at `c603ea5f` unless another
commit or branch is named. A claim marked _verified_ was read at the cited line
or produced by the command shown; _measured_ names the commit whose test
produced the number; _inferred_ is reasoned from the code and not executed.

## 1. A transform today carries a box it enforces by clamping; the fix is four maps chosen by which ends are finite

An estimated parameter θ is searched or sampled through an unconstrained
coordinate z, with θ = g(z). Today the map g is chosen from the parameter's kind
and a search box, and one of its variants, `Log { lo, hi }`, is two things at
once: the map θ = eᶻ, whose image is (0, ∞), and a box [lo, hi] that the inverse
enforces by clamping. Outside the box the map is flat, so the target density
there is zero. Every finite bound under the log map is therefore a hard wall,
and NUTS (the No-U-Turn Sampler, the gradient-based parameter update inside
PGAS) diverges on it.

This proposal replaces the transform with Stan's four constraint transforms —
identity, lower-bounded, upper-bounded and interval — chosen by which ends of
the parameter's _effective domain_ are finite. No variant carries a bound it
does not keep, so nothing is ever clamped and no target has a wall. The
statistical target does not change: under PMMH, MH and PGAS it is the posterior
truncated to the domain, as now; under IF2 it is the likelihood maximized over
the domain, as now. What changes is the geometry the samplers move through, and
three latent defects the change exposes (§3.3): a log-normal prior that reads z
as ln θ, half-bounded boxes on `real` and `instant` parameters that are recorded
but not enforced, and a logit log-Jacobian that underflows to −∞ in its upper
tail.

## 2. Terms

- **θ, z, g.** θ is a parameter on its natural scale; z is the unconstrained
  coordinate a method moves in; g is the map θ = g(z). The _forward map_ is g⁻¹
  (θ to z), the _inverse map_ is g.
- **Jacobian term.** A density for θ becomes a density for z by multiplying by
  |g′(z)|, so a sampler moving in z adds ln|g′(z)| to the log target. This is
  the _log-Jacobian_. Its z-derivative enters the NUTS gradient.
- **Support.** The values the model admits: its declared `in [lo, hi]`, else its
  kind's range (`probability` [0, 1]; `rate`, `positive`, `count`, `duration`
  [0, ∞); `real`, `instant` (−∞, ∞)). Verified,
  `rust/crates/cli/src/roles.rs:1018-1032`.
- **Search box.** Where the method looks: the fit config's `[estimate].bounds`
  or `--estimate NAME=LO:HI`, else the support. Verified,
  `fit/runner.rs:1267-1306` (`resolve_search_box`).
- **Effective domain.** The intersection of support and search box: the set the
  method ranges over. Its _shape_ is which of its two ends are finite.
- **Wall.** A point in z where the target density drops to zero discontinuously.
  A leapfrog step that crosses one has infinite energy error and NUTS records a
  _divergence_.
- **σ, softplus.** σ(z) = 1/(1 + e⁻ᶻ) is the logistic function; softplus(x) =
  ln(1 + eˣ), evaluated as max(x, 0) + log1p(e^(−|x|)), which neither overflows
  nor loses precision.
- **MCSE, ESS.** Monte Carlo standard error, and effective sample size, of a
  posterior summary.

## 3. What the code does today

### 3.1 Selection happens once, in the CLI, from the kind and the box

Verified by reading `rust/crates/cli/src/fit/runner.rs`:

- `default_search_box` (`:1158-1165`) gives the box when nothing is declared:
  `(−∞, ∞)` for `real` and `instant`, `[0, ∞)` for every other kind and for a
  parameter whose IR carries no kind; `probability` gets `[0, 1]` through
  `resolved_bounds` (`params_resolver.rs:141-143`).
- `derive_transform_with_bounds` (`:1174-1229`) maps kind, box and an optional
  override to
  `sim::inference::types::Transform { None, Log { lo, hi }, Logit { lo, hi } }`:
  `probability` → `Logit`; `rate`, `positive`, `count`, `duration` → `Log` when
  lo ≥ 0, else `Logit` when the box is finite, else `None` (`:1205-1207`);
  `real`, `instant` → `Logit` when finite at both ends, else `None`
  (`:1194-1200`, `:1221-1224`).
- The override `[estimate].transform` is a typed enum (`fit/config_v2.rs:588`,
  `:630-634`) threaded as `Option<&str>` (`config_v2.rs:641-647`); the only
  override refused is `identity` on a box finite at both ends
  (`check_transform_keeps_box`, `runner.rs:1243-1256`).
- The IR also carries a per-parameter `transform` field. The compiler always
  emits `Identity` (`ocaml/lib/compiler/expander.ml:5758`) and no runtime code
  reads it: verified, `grep -rn "\.transform()" rust/crates --include='*.rs'`
  finds no caller of the accessor at `ir/src/parameter.rs:366-368`; the field is
  only hashed (`runid/src/ir_hash.rs:478-488`). §15 lists its removal.

### 3.2 Every sampling method calls the same six methods, but three files also match on the variant

`EstimatedParam` (`sim/src/inference/types.rs:49-74`) exposes `to_transformed`
(`:106`), `from_transformed` (`:118`), `log_jacobian` (`:147`), `jacobian_grad`
(`:165`), `transform_deriv` (`:183`) and `transformed_sd` (`:197`). Verified by
`grep -n "to_transformed\|from_transformed\|log_jacobian\|jacobian_grad\|transform_deriv\|transformed_sd\|Transform::" rust/crates/sim/src/inference/*.rs`:

- **IF2** perturbs each particle in z and maps back, with no Jacobian
  (`if2.rs:428-430`, `:533-535`). It also matches on `Transform::Log` to count
  clamp activations (`if2.rs:433-437`, `:537-541`) into `clamp_fraction`
  (`:226`, `:723`), which nothing reads: `grep -rn clamp_fraction rust/crates`
  finds only those two lines.
- **PMMH and MH** propose in z and add the log-Jacobian to the acceptance ratio
  (`pmmh.rs:622-624`, `:686`, `:707-714`). The prior is evaluated as
  `prior.log_density_env(θ, z, env)` (`pmmh.rs:601`, `:698`).
- **PGAS** does the same under MH-within-Gibbs (`pgas.rs:4428-4449`) and, under
  NUTS (the default, `config_v2.rs:1759`), adds the Jacobian and its gradient to
  the target (`pgas.rs:4284-4300`). Its prior gradient,
  `prior_log_density_and_grad_z` (`pgas.rs:3557-3627`), lives in `pgas.rs`. The
  MH-within-Gibbs initial proposal scale evaluates the forward map at the box's
  ends, `p.to_transformed(p.lower.max(1e-10))` (`pgas.rs:3957-3965`).
- **NLopt** (`nl-sbplx`, `nl-bobyqa`, `nl-lbfgs`) uses no transform: it
  maximizes on the natural scale with NLopt's native box
  (`sim/src/inference/deterministic.rs:177-243`).
- **Profile and survey** build their parameters through
  `build_if2_params_from_specs` (`runner.rs:1336-1384`).

### 3.3 The log map's box is a wall, and the change exposes three more defects

**The wall.** `from_transformed` computes `z.exp().clamp(lo, hi)`
(`types.rs:123`); outside [ln lo, ln hi] `log_jacobian` is −∞ and
`transform_deriv` and `jacobian_grad` are 0 (`log_saturated`,
`types.rs:100-102`, and its three uses). So both ends of a two-sided box, and a
lower end above zero, are walls. Measured at `c8b6ee52` (branch
`feat/bsplines-s2`, `rust/crates/sim/tests/nuts_rungs.rs`, 4 chains × 4,000 kept
draws): NUTS on ln τ under an exponential prior gave 0–4 divergences per chain
unboxed and 53–88 with τ ∈ [1e-6, 2.5], with infinite energy error. gh#470 asked
for this smooth parameterisation after gh#374 made the clamped map's derivatives
consistent.

**A log-normal prior reads z as ln θ.** `Density::TransformedNormal` (the
`log_normal` family) computes its density from the `transformed` argument,
assuming z = ln θ (`prior.rs:203-218`), and its NUTS gradient arm returns −(z −
μ)/σ² − 1 on the same assumption (`pgas.rs:3581-3590`). Under any map other than
θ = eᶻ that is a different prior. Today the assumption is enforced by
`validate_prior_transform_compat` (`runner.rs:3572-3690`), which requires `Log`
for `log_normal`, `half_normal`, `gamma`, `exponential` and `log_uniform`, and
`Logit` on exactly [0, 1] for `beta` (`:3626-3642`). That table is why a naive
switch breaks: `beta : rate in [0.05, 1.0] ~ log_normal(...)`, the recovery
fixture `tests/recovery/cases/sir/model.camdl:14`, would be refused once its box
takes the interval map. It is also why
`rho : probability in [0.1, 0.9] ~ beta(...)` (the same file, line 18) is
refused today under every method, with "beta prior requires bounds [0, 1]"
(`runner.rs:3637`), although a beta prior truncated to [0.1, 0.9] is well
defined.

**Half-bounded boxes on `real` and `instant` are not enforced.** A box with one
infinite end on these kinds gets `Transform::None` (`runner.rs:1194-1200`),
whose inverse is z itself (`types.rs:130`), so a box such as `[0, ∞)` on a
`real` is recorded and searched outside. The same holds for an explicit
`transform = "identity"` on a rate's `[0, ∞)`, which `check_transform_keeps_box`
accepts because one end is infinite. Inferred from the code; not executed.

**The logit log-Jacobian underflows.** It is computed as ln((hi − lo)·p·(1 − p))
with p = σ(z) (`types.rs:153`). In double precision 1 − σ(z) is exactly 0 for z
≥ 37 (verified: `python3 -c "import math; print(1-1/(1+math.exp(-37)))"` →
`0.0`), so the log-Jacobian is −∞ there: an accidental wall far in the upper
tail.

**A false doc comment.** `transformed_sd` says it "matches pomp's convention:
the user specifies `rw.sd` on the natural scale" (`types.rs:196`). pomp's `mif2`
documentation says the opposite: "If parameter transformations have been
supplied, then the perturbations are applied on the transformed (estimation)
scale" (pomp `man/mif2.Rd`; the sentence is present at pomp 6.5.0.0, read from
the package's GitHub master on 2026-10-28; gh#1001 cites 6.4.0.6). The
`EstimatedParam.rw_sd` field doc (`types.rs:56-57`) already says "on the
_transformed_ scale", contradicting the method. Commit series (ii) resolves both
(§11).

## 4. Support, search box and effective domain stay separate; the transform is the effective domain

### 4.1 The types

The sim crate's transform becomes the representation of the effective domain.
Each variant carries exactly the finite ends it keeps:

```rust
// rust/crates/sim/src/inference/types.rs

/// The effective domain of an estimated parameter, as the map onto it.
/// Stan's constraint transforms (Stan Reference Manual, "Constraint
/// Transforms": "Lower bounded scalar", "Upper bounded scalar", "Lower and
/// upper bounded scalar"). Built only by `Transform::for_domain`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    Identity,                       // (−∞, ∞): θ = z
    LowerBounded { lo: f64 },       // [lo, ∞): θ = lo + eᶻ
    UpperBounded { hi: f64 },       // (−∞, hi]: θ = hi − eᶻ
    Interval { lo: f64, hi: f64 },  // [lo, hi]: θ = lo + (hi − lo)·σ(z)
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    NotAnInterval { lo: f64, hi: f64 },   // NaN, lo ≥ hi, lo = +∞ or hi = −∞
}

impl Transform {
    /// The map for the domain with these ends; an infinite end is open.
    pub fn for_domain(lo: f64, hi: f64) -> Result<Self, DomainError>;
    /// The domain's ends, (−∞, ∞) for `Identity`.
    pub fn bounds(&self) -> (f64, f64);
}

pub struct EstimatedParam {
    pub name: String,
    pub index: usize,
    pub initial: f64,
    pub rw_sd: f64,
    pub transform: Transform,   // the domain; `lower` and `upper` are removed
    pub rw_sd_auto: bool,
    pub perturb_only_at_t0: bool,
}

impl EstimatedParam {
    pub fn bounds(&self) -> (f64, f64) { self.transform.bounds() }
    // The six existing methods keep their names and signatures (§5).
    /// ln θ, ln(θ − lo), ln(hi − θ) and ln(1 − θ), computed from z without
    /// cancellation, for the prior (§6). NaN where undefined.
    pub fn prior_point(&self, z: f64) -> PriorPoint;
}
```

Today `EstimatedParam` carries `lower` and `upper` beside `transform`
(`types.rs:61-64`), and nothing ties them together: test fixtures build
`Transform::None` with `lower: 0.0, upper: 1.0` (for example
`cli/src/fit/runner.rs:3688-3692`), a box nothing keeps. Deriving the bounds
from the transform makes that state unrepresentable (D1). The fields are read at
about 37 sites in `cli/src` and at `pgas.rs:3959-3960`; each becomes `bounds()`.

In the CLI the three concepts stay three fields. `roles.rs` already has
`Support` (`:66-72`) and `SearchBox` (`:81-102`); `Estimated` keeps both and its
`transform` becomes the effective domain:

```rust
// rust/crates/cli/src/roles.rs
pub struct Estimated {
    pub support: Support,        // from the model: declared range, else kind
    pub search: SearchBox,       // from the user, within the support
    /// The effective domain, support ∩ search, as the map onto it.
    pub transform: Transform,
    pub prior: ResolvedPrior,
    pub prior_text: String,
    pub start: Option<f64>,
}

/// support ∩ search. A search box that leaves the support is refused, so
/// the intersection is the search box whenever this returns `Ok`.
pub fn effective_domain(support: &Support, search: &SearchBox)
    -> Result<Transform, RoleError>;
```

`TransformFrom` (`roles.rs:121-126`) and `Estimated.transform_from` are deleted
with the override (D2). `BoxFrom::Fallback` (`roles.rs:97-101`) is deleted with
the no-kind fallback (D6). The stored record keeps `support`, `search` and a
`transform` tag (`ParamRecord::Estimated`, `roles.rs:550-566`), whose values
become `identity`, `lower_bounded`, `upper_bounded` and `interval`; the record
drops `transform_from`. The tag is derivable from `search`, but it is kept so a
record states the map that ran even if the selection rule later changes.

The selection rule is the shape of the domain alone:

| lo     | hi     | transform             |
| ------ | ------ | --------------------- |
| −∞     | ∞      | `Identity`            |
| finite | ∞      | `LowerBounded { lo }` |
| −∞     | finite | `UpperBounded { hi }` |
| finite | finite | `Interval { lo, hi }` |

The kind enters only through the support. Today's choices are reproduced
wherever they were sound: `rate` on [0, ∞) is `LowerBounded { lo: 0 }`, which is
θ = eᶻ exactly; `probability` on [0, 1] is `Interval { 0, 1 }`, which is the
logit; a finite box on `real`, `instant` or a signed `duration` is the scaled
logit it is today. Two things change: a positive kind with a finite upper bound
takes `Interval` instead of a walled log, and a lower bound above zero becomes a
shifted log instead of a wall. In addition, a half-bounded box on `real` or
`instant` is now kept (§3.3).

### 4.2 A finite bound truncates the Bayesian target and restricts the IF2 search

Under PMMH, MH and PGAS the target in z is π̃(z) = p(y | g(z))·p(g(z))·|g′(z)|.
Because g maps onto the domain D, ∫ π̃(z) dz = ∫_D p(y | θ) p(θ) dθ, so the
θ-marginal of the chain is p(θ | y, θ ∈ D): the posterior truncated to D. For a
prior with constant parameters the truncation needs no renormalisation, because
the prior's mass in D does not depend on θ. This is the target the code states
today for the walled log (`types.rs:28-33`) and for the logit; this proposal
changes how it is reached, not what it is.

For a hierarchical leaf the prior's mass in D is Z(h) = ∫_D p(leaf | h) d(leaf),
which depends on the hyperparameter h, so the truncated target is wrong for h
unless −ln Z(h) is added. That case is already refused by
`validate_hierarchical_leaf_boxes` (`fit/config_v2.rs:2637-2672`, main
`9ccb2899`). A transform alone cannot fix it, since the defect is a missing
normalising term, not a geometry. The check reads "the box the transform
enforces" and skips `Transform::None` (`config_v2.rs:2657-2659`); after this
change every domain is enforced, so it reads `bounds()` for every leaf, and a
half-bounded `real` leaf is checked for the first time.

Under IF2 the domain restricts the search: the estimate is the maximizer of the
likelihood over D. The map is a search device for the perturbation kernel and
carries no Jacobian, since no density in θ is integrated (§9.1).

### 4.3 The effective domain is the search box because a box outside the support is refused

A box that leaves a _declared_ range is refused today (`runner.rs:1278-1290`). A
box that leaves a _kind's_ range is not: `resolve_search_box` accepts any fit
box when the model declares none (`runner.rs:1291`). This proposal refuses that
too (D7), with the remedy "declare `in [lo, hi]` in the model". A signed phase
is then written as the language already allows,
`phi : duration in [-180.0, 180.0]`, whose declared range is its support
(`roles.rs:1023-1025`). With both refusals the intersection is always the search
box, and `effective_domain` is a check plus a constructor.

### 4.4 Amendments to the parameter-roles proposal

`2026-10-26-parameter-roles.md` is accepted and partly implemented. This
proposal amends it in three places, and supersedes nothing else in it:

- §6.2: `Estimated.transform`'s comment "clamps to `search`" becomes "the
  effective domain, mapped onto"; `transform_from` and `TransformFrom` go.
- §6.4: "The transform comes from `[estimate].transform`, else the kind, and
  clamps to the search box" becomes "the transform is the effective domain's map
  (this proposal §4.1)". The sentence "an identity-transform parameter's box is
  enforced by refusing a proposal outside it (a −∞ log-prior)" is replaced: no
  domain needs a rejection rule, because every finite end is in the map.
- §6.4's statement that the box is enforced "only by the transform — the `Log`
  inverse clamps to [lo, hi] and its log-Jacobian is −∞ outside it" describes
  the code before this change.

## 5. The four maps, with forms that stay finite

Write w = hi − lo. Every quantity below is a closed form in z; none is computed
by differencing θ.

| map            | θ = g(z)    | z = g⁻¹(θ)              | g′(z)        | ln\|g′(z)\|                       | d/dz ln\|g′(z)\| |
| -------------- | ----------- | ----------------------- | ------------ | --------------------------------- | ---------------- |
| `Identity`     | z           | θ                       | 1            | 0                                 | 0                |
| `LowerBounded` | lo + eᶻ     | ln(θ − lo)              | eᶻ           | z                                 | 1                |
| `UpperBounded` | hi − eᶻ     | ln(hi − θ)              | **−eᶻ**      | z                                 | 1                |
| `Interval`     | lo + w·σ(z) | ln(θ − lo) − ln(hi − θ) | w·σ(z)·σ(−z) | ln w − softplus(−z) − softplus(z) | −tanh(z/2)       |

These are the maps of the Stan Reference Manual's "Lower bounded scalar", "Upper
bounded scalar" and "Lower and upper bounded scalar" sections (v2.40, chapter
"Constraint Transforms", mc-stan.org, read 2026-10-28), whose absolute
derivatives are eᶻ, eᶻ and (b − a)·logit⁻¹(y)·(1 − logit⁻¹(y)). The softplus
form of the interval's log-Jacobian is ours: ln σ(z) = −softplus(−z) and ln(1 −
σ(z)) = −softplus(z), and its derivative is σ(−z) − σ(z) = −tanh(z/2).

**Stable evaluation.**

- σ(z) is evaluated as eᶻ/(1 + eᶻ) for z ≤ 0 and 1/(1 + e⁻ᶻ) for z > 0, so
  neither branch overflows.
- The interval's inverse map is evaluated from the nearer end: lo + w·σ(z) for z
  ≤ 0, hi − w·σ(−z) for z > 0. The distance to the nearer end then keeps full
  relative precision.
- The interval's forward map is ln(θ − lo) − ln(hi − θ), not ln(u/(1 − u)) with
  u = (θ − lo)/w, which loses the distance to hi to cancellation in 1 − u.
- The interval's log-Jacobian is finite for every finite z, close to ln w − |z|
  in both tails, which removes the z ≥ 37 underflow of §3.3. Its gradient
  −tanh(z/2) is bounded by 1.
- g′(z) for the interval is computed as w·exp(−softplus(−z) − softplus(z)). It
  underflows to 0 only for |z| > 745, where the log-Jacobian term, which is
  computed from z, still carries the gradient. A zero there only drops the
  likelihood's contribution to the gradient, which is negligible at that
  distance from the data.

**The sign of the upper-bounded derivative.** `transform_deriv` returns the
signed g′(z), −eᶻ for `UpperBounded`. Every caller multiplies by it in the chain
rule (`pgas.rs:4286-4289`, and the prior arms at `pgas.rs:3578-3624`), which is
correct for a negative derivative. `log_jacobian` uses |g′|, which is eᶻ.
`transformed_sd` divides by |g′(z(θ))| (until series ii deletes it). No caller
takes the logarithm of `transform_deriv`: verified by the grep of §3.2.

**Values that round to an endpoint.** θ = lo + eᶻ equals lo in floating point
once eᶻ is below half an ulp of lo (for lo = 1, at z ≤ −37: verified,
`python3 -c "import math; print(1+math.exp(-37)==1)"` → `True`); the interval
does the same at either end. The rounded θ is still in the closed domain, so it
is a legitimate value at which to evaluate the model. What must not round is
anything that becomes singular there: the log-Jacobian is computed from z, and
the prior's logarithms of θ and of its distances to the ends come from z as well
(§6). The Stan Reference Manual states the same limitation for its own
transforms in "Limitations due to finite accuracy presentation": "if the closest
floating-point number for the inverse transformed value is the boundary, then
the value is rounded to the boundary."

**Overflow.** θ = lo + eᶻ is +∞ for z > ln(f64::MAX) ≈ 709.78, and θ = hi − eᶻ
is −∞ there. The likelihood or prior at an infinite θ is non-finite, which every
consumer already treats as a rejected proposal or a divergence
(`types.rs:141-146` documents `mh_accept` and the NUTS handling). This is the
behaviour of today's unboxed `Log { 0, ∞ }`, at a z no posterior reaches.
`Interval` cannot overflow. `from_transformed(±∞)` returns the corresponding end
for the half- and two-sided maps, so it is total.

**A start exactly on a bound** has no finite preimage: z = ln 0 = −∞. The
forward map is made total by moving such a value inward by δ = ε·s, with ε =
10⁻¹⁰ (the existing `PROB_FRACTION_EPS`, `types.rs:228`) and s = w for
`Interval`, or s = max(1, |end|) for a half-bounded end. For the interval this
gives z ≈ ±23.03, which is today's logit behaviour exactly (`types.rs:109-112`).
For `LowerBounded { lo: 0 }` it gives z ≈ −23.03, where today's `Log` gives
ln(10⁻³⁰⁰) ≈ −690.8 (`LOG_PROB_FLOOR`, `types.rs:222`). The start resolver
reports each nudged start in one notice, which names the parameter, the bound
and the value it moved to (D3). A value outside the domain never reaches the
forward map: it is refused when values are resolved (`roles.rs`,
`RoleError::BoxOutsideSupport` and the value checks of parameter-roles §5.4).
The forward map's clamp of such a value stays only as a `debug_assert!`
backstop.

`pgas.rs:3957-3965` evaluates the forward map at `lower.max(1e-10)` and
`upper.min(1e10)` to set the MH-within-Gibbs initial scale. With the total
forward map it keeps its meaning — one tenth of the z-width of the box, with an
end nudged as above — and its only edit is the field access `p.bounds()`. The
value changes for a positive kind with a two-sided box: ln(hi/lo)/10 today (0.30
for [0.05, 1]), and 46.05/10 = 4.6 after, the value every probability and every
unbounded rate already starts from. The Robbins–Monro adaptation
(`pgas.rs:3941-3950`) tunes it during burn-in. NUTS, the default, does not use
it.

## 6. Priors read stable logarithms of θ from the map, not z

The prior's API changes from `(natural, transformed)` to a point that carries θ
and the logarithms a density needs:

```rust
// rust/crates/sim/src/inference/prior.rs
pub struct PriorPoint {
    pub theta: f64,
    pub ln_theta: f64,        // ln θ; NaN unless the domain is within [0, ∞)
    pub ln_one_minus: f64,    // ln(1 − θ); NaN unless the domain is within (−∞, 1]
    /// d/dz of the two logarithms, for the NUTS gradient.
    pub d_ln_theta_dz: f64,
    pub d_ln_one_minus_dz: f64,
}
impl Prior {
    pub fn log_density_env<E: ParamEnv>(&self, at: &PriorPoint, env: &E) -> f64;
    /// Value and z-gradient; moves here from `pgas.rs:3557-3627`.
    pub fn log_density_and_grad_z(&self, at: &PriorPoint, dtheta_dz: f64) -> (f64, f64);
}
```

`EstimatedParam::prior_point(z)` computes the logarithms from the distances to
the ends, which are exact in z: ln(θ − lo) is z for the half-bounded maps and ln
w − softplus(−z) for the interval, and ln(hi − θ) is z or ln w − softplus(z).
Then ln θ = logaddexp(ln lo, ln(θ − lo)) when lo ≥ 0 (with ln 0 = −∞, so ln θ =
ln(θ − lo) at lo = 0), and ln(1 − θ) = logaddexp(ln(1 − hi), ln(hi − θ)) when hi
≤ 1. The derivatives are d ln θ/dz = g′(z)/θ = exp(ln|g′(z)| − ln θ) and d ln(1
− θ)/dz = −g′(z)/(1 − θ), each evaluated through the logarithms.

Each density that takes a logarithm reads it from the point: `log_normal` (ln
θ), `log_uniform` (ln θ), `gamma` (ln θ), `beta` (ln θ and ln(1 − θ)). For
`LowerBounded { lo: 0 }`, ln θ = z exactly, so the `log_normal` density and
gradient are bit-for-bit today's (`prior.rs:203-218`, `pgas.rs:3581-3590`). For
every other map they become correct where today they would be a different prior.
Densities that take no logarithm (`normal`, `half_normal`, `exponential`,
`uniform`, `truncated_normal`, `flat`) read θ, as now.

`validate_prior_transform_compat` (`runner.rs:3572-3690`) becomes a support
rule: **the effective domain must lie within the prior family's support.**
`log_normal`, `gamma`, `exponential`, `half_normal` need D ⊆ [0, ∞); `beta`
needs D ⊆ [0, 1]; `log_uniform(L, U)` and `uniform(L, U)` need D ⊆ [L, U] (D ⊆
[lo, hi] is checked only when the ends are constants); `normal` and `flat`
accept any D. `truncated_normal` keeps its rule that its truncation equals the
box (`runner.rs:3647-3686`). A prior whose support is narrower than D would put
zero density inside the domain, which is a wall by another route, so it is
refused. Under the new rule `beta` on `probability in [0.1, 0.9]` and
`log_normal` on `rate in [0.05, 1.0]` are accepted, and the prior is the family
truncated to the box. `Density::transform_req`, `Prior::transform_req` and
`TransformReq` (`prior.rs:82`, `:142-155`, `:348`) are deleted (D4).

## 7. An explicit transform serves no purpose, so the field is deleted

An override preserves the effective domain only if it names the map the domain
already selects: `log` on [0, ∞), `logit` on a finite box, `identity` on (−∞,
∞). Every other override changes the domain — `log` on a finite box reinstates
the wall, `identity` on a finite end leaves the end unenforced, and `logit` on a
half-bounded domain has no map. So an accepted override is always redundant, and
a refused one is always the only kind that would have mattered.
`[estimate].transform` (`config_v2.rs:588`), `config_v2::Transform`
(`:628-648`), `ParamSpec.transform` (`runner.rs:1320`),
`check_transform_keeps_box` and the threaded `Option<&str>` are deleted (D2).
Because `EstimateSpecV2` has `deny_unknown_fields` (`config_v2.rs:573`), a
config that still carries the key is refused at load, naming the key; under the
alpha posture there is no compatibility path. §13 gives the migration.

## 8. No parameter is discrete; `count` is continuous and rounded where it is used

`ParamKind` has seven variants and no integer one
(`ir/src/parameter.rs:132-140`); `integer` is a compartment type keyword, not a
parameter kind (language spec §14.2, "Reserved Identifiers",
`docs/camdl-language-spec.md:3995-3998`; the AST's `param_type` has
`PRate | PProbability | PPositive | PCount | PReal | PInstant | PDuration`,
`ocaml/lib/compiler/ast.ml:129`). Every parameter value is an `f64`. A `count`
parameter is a continuous quantity of dimension P (language spec §2.2.1, `:347`)
and is estimated as such. Where an expression over it initialises an integer
compartment, the value is rounded at that point
(`sim/src/compiled_model.rs:2420-2421`). Verified by reading.

So a `count` such as `I0` has a likelihood that is a step function of its value
when it reaches the model only through an integer initial state, and NUTS sees a
zero gradient in that direction. That is a property of rounding at use, not of
the transform, and it is out of scope here (§15). No genuinely discrete
parameter exists to treat separately.

## 9. Each method uses the shared map as it does now

### 9.1 IF2 cannot reach a bound, and reports when the estimate approaches one

IF2 perturbs in z, maps back, and uses no Jacobian; that does not change. Under
the walled log a perturbation past an end was clamped onto it, so the swarm
could hold mass exactly at the end. Under the smooth maps an end is z = ±∞: a
maximum on a bound is approached as the swarm drifts outward, and the reported
estimate (the particle mean at the best iteration) lies inside the domain at a
distance that shrinks with cooling. The likelihood at that estimate is within
Monte Carlo error of the boundary value whenever the likelihood is continuous at
the end.

The existing near-bound diagnostic, `ParamNearBound` (`runner.rs:2514-2547`),
fires when a chain's estimate is within 1% of the box's width from an end, and
only when both ends are finite. It is extended to a finite end of a half-bounded
domain, where there is no width: it fires when the distance to the end is below
1% of the start's distance to it, (θ̂ − lo) < 0.01·(θ₀ − lo) (D10). The message
gains one sentence: IF2 cannot place an estimate exactly on a bound; to test
whether the maximum is on it, profile the parameter or refit with `nl-bobyqa`,
which can (§9.3).

`clamp_fraction` measures an event that no longer exists. It is computed and
never read (§3.2), so it is deleted in its own commit before the refactor (§11).

The step size is unchanged in commit series (i) (§9.4 gives the auto rule);
series (ii) changes what `rw_sd` means.

### 9.2 PMMH, MH and PGAS add the Jacobian, as now

The acceptance ratio and the NUTS target are unchanged in form: likelihood plus
log-prior plus `log_jacobian`, with `jacobian_grad` and `transform_deriv` in the
gradient (`pmmh.rs:707-714`, `pgas.rs:4284-4300`, `pgas.rs:4444-4449`). Only the
prior's argument changes (§6). Because no map has a wall, a NUTS trajectory near
a bound sees the log-Jacobian's tail, ln w − |z|, instead of a cliff.

### 9.3 NLopt stays on the natural scale; moving it is a measured follow-up

**Option A: optimise in z.** The objective becomes ℓ(g(z)); L-BFGS's gradient is
∇θℓ ⊙ g′(z), formed in `fit/nlopt_stage.rs` from `det_grad`'s ∇θ. Since every
end is at z = ±∞, a maximum on a bound is reachable only if NLopt's z-box is
clipped at the nudged ends of §5 (z ≈ ±23.03 for an interval), which places it
within 10⁻¹⁰ of the width from the end. The gain is conditioning: a rate that
spans orders of magnitude is better scaled in ln θ, and Sbplx's simplex and
Bobyqa's trust region are set in coordinate units. The cost is the stopping
rule: `xtol_rel` is a _relative_ tolerance, "stop when an optimization step …
causes a relative change the parameters by less than `tol`" (NLopt Reference,
"Stopping criteria", nlopt.readthedocs.io, read 2026-10-28), and an interval
coordinate near z = 0 has no meaningful relative change, so option A must also
move the stopping rule to `xtol_abs`.

**Option B: keep the natural scale.** NLopt enforces the box natively — "NLopt
guarantees that your objective function and any nonlinear constraints will
_never_ be evaluated outside of these bounds" (NLopt Reference, "Bound
constraints") — and a maximum exactly on a bound is reachable. This is the one
place where an optimiser and a smooth map genuinely differ, and the optimiser's
behaviour is the one wanted for the MLE.

The MLE is invariant under reparameterisation in exact arithmetic, so B changes
no estimand. Coordinate-dependent quantities are a density's mode (a MAP
estimate, whose density carries the Jacobian), and curvature-based standard
errors (Hessian, Wald, Laplace). camdl computes none of the second kind
(verified:
`grep -rn -i "hessian\|wald\|laplace" rust/crates/cli/src rust/crates/sim/src`
finds no such computation). Its "MAP" is the draw with the highest natural-scale
log-likelihood plus log-prior, with no Jacobian
(`sim/src/inference/pmmh.rs:747-749`; the PGAS `log_posterior` column likewise
excludes it, `cli/src/fit/pgas.rs:839`), so it does not depend on the sampling
coordinate. Posterior means and quantiles of θ are expectations under the target
measure, which the coordinate does not change; only their Monte Carlo error
does. The draws are stored on the natural scale.

Recommendation: B in this proposal; A under gh#1002, which already specifies its
benchmark (iterations to convergence and final log-likelihood, `nl-bobyqa` on
the natural scale against the log scale), and which must also switch to
`xtol_abs` (D5). gh#1002 also audits existing comparisons of optimiser and
sampler outputs across the sibling repositories; one of its items, whether the
PMMH "MAP" carries the Jacobian, is answered above: it does not.

### 9.4 Starts, Latin-hypercube draws and the auto step key on the kind, not the map

Three consumers decide "log-spaced" by matching `Transform::Log`:
`lhs_map_to_natural` (`fit/init.rs:221-224`), `survey`'s start draw
(`survey.rs:900-909`) and its pin-fraction diagnostic (`survey.rs:1693-1696`).
Under the new selection a `rate in [0.05, 1.0]` is `Interval`, so matching on
the map would silently switch its starts to linear spacing. One function decides
spacing for all three:
`fn spacing(kind: Option<ParamKind>, domain: &Transform) -> Spacing { Linear, Log }`,
log-spaced exactly when the kind is `rate`, `positive`, `count` or `duration`
and 0 < lo < hi < ∞. Without overrides that reproduces today's choice for every
case. `kept_boxes` (`chain_starts.rs:822-836`) becomes `bounds()` for every
parameter, since every domain is now kept.

The auto `rw_sd` (`runner.rs:1429-1451`) also forks on the map: log range/20 at
the geometric midpoint for `Log`, (hi − lo)/6 for `Logit`. In series (i) it
forks on the same `spacing` instead (D8). Because IF2 converts the natural-scale
step through the delta method at the current value, the natural-scale step is
then the same as today wherever the parameter is away from its ends. For the SIR
fixture's `beta` on [0.05, 1.0] that keeps 0.0335. Keying it on the map instead
would make it (1.0 − 0.05)/6 = 0.158, 4.7 times larger. A half-bounded or
unbounded domain has no auto step, as now
(`refuse_auto_rw_sd_without_finite_box`, `runner.rs:1393-1405`).

### 9.5 Profile and survey inherit the change

Both build through `build_if2_params_from_specs`, so they need no change beyond
the deleted override (`profile.rs:889`, `runner.rs:714` pass it today). Simplex
groups use their own log-ratio map (`if2.rs:58-67`) and are untouched.

## 10. The high-risk files need small edits, listed here; the implementation needs two reviews

`pgas.rs`, `pmmh.rs` and `if2.rs` do not call only `EstimatedParam` methods.
They depart from that in three ways, each verified by the grep of §3.2: `if2.rs`
matches `Transform::Log`; `pgas.rs` reads the `lower` and `upper` fields and
owns the prior gradient; and both `pmmh.rs` and `pgas.rs` pass z to the prior.
The complete list of edits:

- **`if2.rs`** — series (i): delete `clamp_fraction` (`:226`), `clamp_counts`
  (`:405`), the two `if let Transform::Log` blocks (`:433-437`, `:537-541`) and
  its fill (`:723`). Series (ii): `:428`, `:533` and `:639` use `spec.rw_sd`
  directly instead of `spec.transformed_sd(spec.rw_sd, current)`. The
  perturbation itself (`:429-430`, `:534-535`) is unchanged.
- **`pmmh.rs`** — series (i): `:601` and `:698` pass `p.prior_point(z)` instead
  of `z`. The proposal, the Jacobian and the acceptance (`:622-714`) are
  unchanged.
- **`pgas.rs`** — series (i): `prior_log_density_and_grad_z` (`:3550-3627`)
  moves to `prior.rs` as `Prior::log_density_and_grad_z`, its `log_normal` arm
  rewritten through `PriorPoint` (§6), and the NUTS target's call (`:4292-4294`)
  follows it; `:3959-3960` use `p.bounds()`; the prior calls at `:4067`,
  `:4444-4446` and `:4547` pass `prior_point`. The NUTS target's Jacobian lines
  (`:4286-4300`) and the MH-within-Gibbs ratio (`:4428-4449`) are unchanged.
- `particle_filter.rs`, `obs_loglik.rs`, `obs_model.rs`, `pgas_grad.rs` and
  `nuts.rs`: no edit (verified: none of the grep's patterns occur in them).

Every one of these files changes behaviour even where its text does not, because
the map it calls changes. The implementation therefore needs two independent
reviews before it lands, each re-running the validation of §12 rather than
reading it.

## 11. Two separately reviewable commit series, bisectable

**Series (i), the transform refactor.** Each commit passes `make test`.

1. Delete IF2's `clamp_fraction`: dead code, its own commit.
2. Introduce `PriorPoint` and move the prior gradient to `prior.rs`, under the
   existing transforms. For `Log`, ln θ from z equals z, so this commit is
   byte-identical on every fixture; its check is a before/after diff of a PMMH
   and a PGAS+NUTS run of `tests/recovery/cases/seir_age` at a fixed seed.
3. Replace `Transform`, derive `EstimatedParam`'s bounds, implement §5's
   numerics, add §12.1's tests. The CLI selects by domain shape; the override is
   deleted; the prior rule becomes the support rule; spacing and auto `rw_sd`
   key on `spacing`; a box outside the kind's range is refused; the near-bound
   diagnostic is extended. This is one commit because the enum change breaks
   every match on it; it changes results and is the commit the benchmark
   measures.
4. Documentation: `docs/camdl-inference-spec.md` §3.2 rule 2 (its table,
   `:152-187`), `docs/fit-toml.md` and `docs/agents.md` (the override), and the
   parameter-roles amendments of §4.4.

**Series (ii), the `rw_sd` scale (gh#1001).** `rw_sd` is specified on the
transformed scale, as pomp's `mif2` does. The auto step becomes
`transformed_sd(auto_natural, θ₀)`, computed once at the start, so a default
run's first step is unchanged and only the later steps stop moving with the
parameter. `transformed_sd` and its false doc comment are deleted; the field doc
becomes true. PMMH's initial proposal becomes `5 × rw_sd`
(`cli/src/fit/pmmh.rs:1049-1052`). Documentation: inference spec `:86-91`,
`docs/fit-toml.md`. One commit for the change, one for documentation (D9).

The two series touch overlapping lines only in `if2.rs:428`, `:533` and `:639`
and in `types.rs`. Series (i) lands first; (ii) is rebased on it. Either can be
reverted alone.

## 12. Validation, with pass criteria

### 12.1 Deterministic tests of the maps

In `sim/src/inference/types.rs`, for each map and each of the domains [0, ∞),
[10⁻⁶, ∞), (−∞, 30], [0, 1], [10⁻⁶, 2.5], [−120, 120] and [1, 10⁸]:

- round trip g(g⁻¹(θ)) = θ for interior θ, within 4 ulp of the distance to the
  nearer end; g⁻¹(g(z)) = z for |z| ≤ 30, within 10⁻¹²;
- `transform_deriv`, `jacobian_grad` and `prior_point`'s derivatives against
  central finite differences, relative error below 10⁻⁶ on |z| ≤ 30;
- `log_jacobian` finite for every |z| ≤ 700, and g(z) inside the closed domain
  for z ∈ {±40, ±800, ±∞} (red on main for the interval at z = 40, where today's
  logit gives −∞);
- the sign of `UpperBounded`'s `transform_deriv` is negative;
- `prior_point(z).ln_theta == z` exactly for `LowerBounded { lo: 0 }` at z =
  −800, where θ underflows to 0;
- a start on each end maps to the finite z of §5 and produces one notice;
- `log_normal` on `LowerBounded { lo: 0 }`: the new
  `Prior::log_density_and_grad_z` equals today's `prior_log_density_and_grad_z`
  bit for bit.

Each test is mutation-checked: reverting its fix turns it red.

### 12.2 A prior-only PGAS+NUTS run recovers the truncated prior

A likelihood-free PGAS+NUTS run (4 chains × 4,000 kept draws) with three
parameters: τ ~ exponential(5.99) on [10⁻⁶, 2.5] (`Interval`, the case measured
at `c8b6ee52`); ρ ~ beta(2, 5) on [0.1, 0.9]; κ ~ half_normal(1) on [0.5, ∞)
(`LowerBounded { lo: 0.5 }`). Pass: each mean and standard deviation within 4
MCSE of the truncated law's; a Kolmogorov–Smirnov test on draws thinned to the
bulk ESS with p > 0.001; divergences after adaptation no more than the unboxed
exponential control's bound.

The branch `nuts-divergence-tests` (not on main at `c603ea5f`) adds permanent
NUTS validity tests in `rust/crates/sim/tests/`, among them
`boxed_log_scale_diverges_no_more_than_the_scaled_logit`, which is `#[ignore]`d
"until doubly bounded params use the scaled logit under NUTS". It builds
`Transform::Log { lo, hi }` by hand, so after series (i) it is rewritten to the
map the CLI gives a doubly bounded rate, `Interval`, and un-ignored; it is
expected to pass. Its sibling tests that construct `Log` or `Logit` move to the
new variants in the same commit.

### 12.3 A boxed posterior is unchanged under PGAS+NUTS

`tests/recovery/cases/seir_age` with its `pgas.toml` (three rates boxed to
[0.001, 0.5] and [0.01, 1.0], log-normal priors), on synthetic data from its
`truth.toml`, at main and after series (i), 4 chains each. Pass: for each
parameter, the posterior mean and the 5%, 50% and 95% quantiles agree within
4·√(MCSE_before² + MCSE_after²); R̂ ≤ 1.01 in both; divergences after ≤ before.
The target is the same truncated posterior in both runs (§4.2), so a
disagreement is a defect or a sampling failure of the walled run; if one
appears, a run four times longer at main decides which.

### 12.4 Benchmark on three boxed fixtures

The fixtures are `seir_age` (its `if2.toml`, `pmmh.toml`, `pgas.toml`), `sir`
(its `if2.toml`, plus a PMMH config with the same `[estimate]`), and `sir` with
`gamma`'s box tightened to [0.01, 0.16] around its truth of 0.15
(`tests/recovery/cases/sir/if2.toml:1-4`), so that the posterior has mass near
an upper bound — the case the change is for. Each is run at main and after
series (i), then again after series (ii), at the default scales and with `rw_sd`
scaled by 0.5 and 2.

Reported: PMMH acceptance and bulk ESS per wall-second (minimum over
parameters); IF2 final log-likelihood (the clean particle-filter evaluation,
mean and standard deviation over chains), iterations until within one
log-likelihood unit of the final value, and chain agreement.

Pass: IF2's final log-likelihood is no worse than main's by more than twice the
particle-filter standard deviation on any fixture, at the best of the three
scales; PMMH ESS per second is at least 0.8 of main's on the two interior
fixtures and higher than main's on the near-bound fixture; PMMH acceptance is
between 0.1 and 0.5. The measurements go in a `docs/dev/notes/` note, which the
series (i) merge cites.

## 13. What changes for users

- **Run identity.** Every fit re-keys: the fit level folds the engine version,
  `VERSION_SHORT`, which includes the git hash (`cli/src/fit/cas.rs:501-521`,
  `cli/src/version.rs:12-15`). No schema bump is needed.
- **Configs that set `transform`** are refused at load, naming the key. The
  migration is to delete the line. Counted (verified,
  `grep -rh --include='*.toml' -oE 'transform\s*=\s*"[a-z]+"'` over this
  repository and the sibling repositories `camdl-book`, `camdl-vignettes`,
  `camdl-garki`, `camdl-india-polio`, `camdl-nigeria-polio`, `ebola-camdl`,
  `ebola-bdbv-camdl` and `camdl-paper`): 167 `log` and 74 `logit` overrides, in
  about 100 files; in this repository, 14 lines in code, tests and docs. A
  `transform = "log"` on a two-sided box ran with walls and now runs on the
  interval map; a `transform = "identity"` on a half-bounded box was not
  enforced and now is.
- **Posteriors** are the same truncated posteriors. Sampling near a bound
  changes: fewer NUTS divergences, different PMMH acceptance. The PGAS
  MH-within-Gibbs initial scale changes for positive kinds with two-sided boxes
  (§5).
- **IF2 and PMMH dynamics** change only near a bound in series (i), because the
  auto step keys on spacing (D8). In series (ii), an explicit `rw_sd` changes
  meaning. A value tuned on the natural scale converts as rw_sd_z =
  rw_sd_natural / |g′(z(θ₀))|; for a `LowerBounded { lo: 0 }` rate, that is
  rw_sd_natural / θ₀. A natural `rw_sd = 0.02` on a rate started at 0.4 was a
  log-scale step of 0.05; after series (ii), 0.02 in a config is a log-scale
  step of 0.02.
- **Priors newly accepted:** `beta` on a narrowed `probability`, and
  `log_normal`, `gamma` and the other positive families on a positive kind's
  two-sided box, each truncated to the box. **Newly refused:** a prior whose
  support is narrower than the domain, such as `log_normal` on a `real`; a fit
  box outside the kind's range on a parameter that declares no range (D7); a
  half-bounded `real` hierarchical leaf whose box cuts its prior (§4.2).
- **Diagnostics and records:** the near-bound warning covers half-bounded
  domains; the launch transforms table names the four maps; the partition
  record's `transform` tag takes the new values and loses `transform_from`.

## 14. Decisions

**D1 — Does the transform carry the domain, with `EstimatedParam`'s bounds
derived from it? Recommended: yes. Confidence: solid.** Today `lower`, `upper`
and `transform` are independent fields, and a box can be recorded that nothing
enforces (§4.1). Option (a), one enum that is the domain, makes an unenforced
box unrepresentable; it rewrites about 95 construction sites (40 in `src`, 55 in
tests), mechanically, and every reader of `lower`/`upper` calls `bounds()`.
Option (b), a separate `Domain` type from which a `Transform` is derived, has
two isomorphic enums and the same sites to touch. Option (c), keeping the
fields, preserves the drift this proposal exists to remove. No user-facing
behaviour differs among them.

**D2 — Delete `[estimate].transform`? Recommended: delete. Confidence: solid.**
§7 shows that an override is either redundant or would change the domain. Option
(a), deleting the field, makes the refusal hold by construction and removes
`TransformFrom`, `check_transform_keeps_box` and the string threading; every
config that sets it fails at load until the line is removed (about 100 files
across the sibling repositories, §13). Option (b), keeping the field and
accepting only the redundant value, keeps a knob with no effect and a validation
path to maintain. Option (c), keeping the field and refusing every value, is (a)
with an extra error message. Under the alpha posture (a) is the form.

**D3 — A start exactly on a bound: nudge inward with a notice, or refuse?
Recommended: nudge. Confidence: leaning.** Starts arrive on bounds legitimately:
an NLopt optimum on a bound fed to PGAS through `--starts`, a user's `start = 0`
for an amplitude. Option (a) moves the start 10⁻¹⁰ of the scale inward (§5) and
prints one notice. This is today's logit behaviour exactly, and it replaces
today's log behaviour (z = −690.8) with z = −23.03, which a chain can climb back
from. Option (b) refuses, which is more explicit but breaks NLopt-to-sampler
chaining whenever the optimum is on a bound. The cost of (a) is a start deep in
a tail, which burn-in must absorb; the notice says so.

**D4 — Do priors read stable logarithms of θ, with the compatibility table
replaced by a support rule? Recommended: yes. Confidence: solid.** This is
required for correctness, not style. Without it, a `log_normal` prior under any
map but θ = eᶻ is silently a different prior (§3.3). With it, the prior's API
changes at five call sites in `pmmh.rs` and `pgas.rs` (§10), the `log_normal`
case under `LowerBounded { lo: 0 }` stays bit-identical (§12.1), and users gain
truncated `beta` and positive-family priors on boxes. The alternative, keeping
`TransformReq` and forcing `Log` for log-family priors, would bring back the
walled log for every boxed rate with a `log_normal` prior — the common case in
the recovery fixtures.

**D5 — NLopt coordinates. Recommended: keep the natural scale here; measure
z-coordinates under gh#1002. Confidence: leaning.** §9.3 sets out both options.
Moving now would change the optimiser's conditioning, its stopping rule and its
behaviour at bounds, with no measurement in the repository to say it helps.
`docs/dev/agent-verification-conventions.md` ("A new inference method merges
with the note that measures it") asks for that measurement first. Keeping B
means the optimisers and samplers differ in coordinates, which changes no
reported estimand (§9.3). The trade-off is that a user comparing an `nl-bobyqa`
fit with an IF2 fit near a bound sees the optimiser on the bound and IF2 just
inside it; the extended near-bound message (D10) points to this.

**D6 — The domain of a parameter whose IR carries no kind. Recommended: (−∞, ∞),
deleting `BoxFrom::Fallback`. Confidence: leaning.** The compiler always emits a
kind (`ast.ml:150-151` makes it mandatory; `expander.ml:5806`, `:5866`), so only
hand-built test IR and IR from before the field existed reach this case.
`roles.rs` already gives such a parameter the support (−∞, ∞) (`:1029`), while
the runner searches [0, ∞) (`runner.rs:1163`), so the record's support and
search differ for no modelled reason. Option (a), (−∞, ∞), makes them agree.
Unbounded parameters of this kind become `Identity`, which changes some unit
tests that build kind-less unbounded parameters. Option (b) keeps [0, ∞) and the
fallback variant. Making `param_kind` non-optional in the IR is the complete fix
and is an `ir/VERSION` change (§15).

**D7 — A fit box outside the kind's range, on a parameter that declares no
range: refuse or intersect? Recommended: refuse. Confidence: leaning.** Today it
is accepted (`runner.rs:1291`). That is how a `duration` with a fit box [−120,
120] runs as a signed phase (gh#967). Option (a), refusing with "declare
`in [lo, hi]` in the model", makes the support the single source of what a
parameter may be. The fit config then only narrows, which is today's rule for
declared ranges. Configs that widen an undeclared kind range fail until the
model declares it. Option (b), intersecting, would silently turn [−120, 120]
into [0, 120] for that phase: a changed estimand with no message. Option (c),
accepting as now, keeps a search box that is not inside the support, so the
three concepts of §4 would no longer nest.

**D8 — What the auto `rw_sd` keys on in series (i). Recommended: the kind-based
spacing rule, not the map. Confidence: solid.** Keying on the map would multiply
the natural-scale IF2 step of every positive kind with a two-sided box by about
4.7 (§9.4), changing default IF2 dynamics in a commit whose purpose is the map.
Keying on spacing keeps the step wherever the parameter is away from its ends,
so series (i)'s benchmark measures the map alone. No user-facing change beyond
§13.

**D9 — The `rw_sd` scale (gh#1001). Recommended: transformed scale, with the
auto step computed once at the start. Confidence: leaning.** Option (a) matches
pomp's `mif2` and makes the step a fixed z-distance, removing the
position-dependent step that gh#1001 hypothesises biases the swarm. Computing
the auto step once at θ₀ keeps a default run's first step identical. Every
explicit `rw_sd` changes meaning (§13 gives the conversion), and PMMH's initial
proposal becomes 5 × `rw_sd`. Option (b), the natural scale converted at each
step, is today's convention and needs only the false comment fixed; it keeps a
step that changes as the parameter moves. Option (c), the natural scale
converted once at the start, gives the same dynamics as (a) under a unit the
user cannot read off the coordinate. Series (ii) is reviewable and revertable
alone (§11).

**D10 — The near-bound report for a half-bounded domain. Recommended: fire when
the estimate's distance to the end is below 1% of the start's. Confidence:
leaning.** A half-bounded domain has no width, so today's 1%-of-width rule
(`runner.rs:2530-2546`) cannot apply and the diagnostic is silent. Option (a),
the ratio to the start's distance, is scale-free and says "the fit moved two
orders of magnitude toward this end". Option (b), an absolute threshold, needs a
unit the parameter does not carry. Option (c), silence, leaves a boundary
optimum under IF2 unreported. The cost of (a) is a false alarm when the start
was far from the end and the estimate is legitimately small; the message says
"may be on the bound" and names the test (profile, or `nl-bobyqa`).

## 15. Out of scope, each with its reason

- **The IR's per-parameter `transform` field** is always `Identity` and read by
  nothing (§3.1). Removing it is an `ir/VERSION` change with a golden update,
  and is batched with the next bump. With D6, `param_kind` could become
  non-optional in the same bump.
- **The boxed hierarchical leaf's normalising constant**, −ln Z(h), which would
  lift the refusal of `9ccb2899`; it is a density term, not a map
  (parameter-roles §13).
- **NLopt in z-coordinates**, gh#1002 (D5).
- **Rounding at use for `count` initial states** (§8). A likelihood that is a
  step function of a parameter is a modelling property that no transform
  changes.
- **The PGAS MH-within-Gibbs initial scale for an unbounded `Identity`
  parameter**, which today evaluates the forward map at 10⁻¹⁰ and 10¹⁰ and
  starts at a scale of about 10⁹ (`pgas.rs:3957-3965`). This proposal keeps that
  behaviour unchanged.
