---
date: 2026-10-29
status: accepted — D1–D11 decided by the maintainer on 2026-10-29 (§14); D8's IF2 constants are measured during implementation by the §12.6 benchmark
area: inference setup / sim inference transforms / priors / IF2 and MH step sizes
issues: closes gh#470; carries gh#1001 (commit series ii); gh#1002 is the optimiser follow-up (D5)
related:
  - 2026-10-26-parameter-roles.md (accepted; §4.4 states what this amends in its §6.2 and §6.4)
---

# A parameter's transform is chosen from its effective domain, and every method shares it

Code references are against `origin/main` at `87a426a9`, whose `src` files other
than `sim/src/inference/{mod.rs,nuts_target_tests.rs}` are identical to
`c603ea5f`. A claim marked _verified_ was read at the cited line or produced by
the command shown. _Reproduced by an independent check_ means another agent ran
it and it was not rerun here. _Inferred_ means reasoned from the code and not
executed.

## 1. A transform today carries a box it enforces by clamping; the fix is a map chosen by which ends are finite

An estimated parameter θ is searched or sampled through an unconstrained
coordinate z, with θ = g(z). Today g is chosen from the parameter's kind and a
search box, and one of its variants, `Log { lo, hi }`, is two things at once:
the map θ = eᶻ, whose image is (0, ∞), and a box [lo, hi] that the inverse
enforces by clamping. Outside the box the map is flat and the target density is
zero. Every finite bound under the log map is therefore a hard wall, and NUTS
(the No-U-Turn Sampler, the gradient-based parameter update inside PGAS)
diverges on it.

This proposal replaces the transform with Stan's constraint transforms
(identity, lower-bounded, upper-bounded, interval) and adds one composite, a
log-interval map for positive kinds boxed with 0 < lo (D11). The map is chosen
by the ends of the parameter's _effective domain_, and, for a box with 0 < lo,
by whether the kind is a positive one. No map carries a bound it does not keep,
so nothing is clamped and no target has a wall.

The statistical targets do not change. Under PMMH, MH and PGAS the target is the
posterior truncated to the domain, as now; under IF2 the estimate maximises the
likelihood over the domain, as now. What changes is the geometry the samplers
move through and the IF2 and MH step sizes. IF2 and MH take a fixed step in z,
derived from the domain (D8); today the step is converted to z at each
particle's current value, which blows up near the ends of an interval (§9.1). A
start exactly on a bound has no finite z and is refused (D3).

## 2. Terms

- **θ, z, g.** θ is a parameter on its natural scale; z is the unconstrained
  coordinate a method moves in; g is the map θ = g(z). The _forward map_ is g⁻¹
  (θ to z), the _inverse map_ is g.
- **Jacobian term.** A density for θ becomes a density for z when multiplied by
  |g′(z)|, so a sampler moving in z adds ln|g′(z)| to the log target: the
  _log-Jacobian_. Its z-derivative enters the NUTS gradient.
- **Support.** The values the model admits: its declared `in [lo, hi]`, else its
  kind's range. That range is `probability` [0, 1]; `rate`, `positive`, `count`,
  `duration` [0, ∞); `real`, `instant` (−∞, ∞). Verified,
  `rust/crates/cli/src/roles.rs:1018-1032`.
- **Search box.** Where the method looks: the fit config's `[estimate].bounds`
  or `--estimate NAME=LO:HI`, else the support (`fit/runner.rs:1267-1306`,
  `resolve_search_box`).
- **Effective domain.** Support ∩ search box: the set the method ranges over.
  Its _shape_ is which of its two ends are finite.
- **Wall.** A point in z where the target density drops to zero discontinuously.
  A leapfrog step across one has infinite energy error, and NUTS records a
  _divergence_.
- **Step.** The standard deviation of IF2's per-observation perturbation, or of
  an MH proposal, in z. In the fit config it is `rw_sd`.
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
  `sim::inference::types::Transform { None, Log { lo, hi }, Logit { lo, hi } }`.
  `probability` gets `Logit`. `rate`, `positive`, `count` and `duration` get
  `Log` when lo ≥ 0, else `Logit` when the box is finite, else `None`
  (`:1205-1207`). `real` and `instant` get `Logit` when the box is finite at
  both ends, else `None` (`:1194-1200`, `:1221-1224`).
- The override `[estimate].transform` is a typed enum (`fit/config_v2.rs:588`,
  `:630-634`) threaded as `Option<&str>` (`config_v2.rs:641-647`).
- The IR also carries a per-parameter `transform` field. The compiler always
  emits `Identity` (`ocaml/lib/compiler/expander.ml:5758`), and no runtime code
  reads it: `grep -rn "\.transform()" rust/crates --include='*.rs'` finds no
  caller of the accessor at `ir/src/parameter.rs:366-368`. The field is only
  hashed (`runid/src/ir_hash.rs:478-488`). §15 lists its removal.

### 3.2 The transform's use sites, listed mechanically

The sim crate's `EstimatedParam` (`sim/src/inference/types.rs:49-74`) carries
`lower`, `upper` and `transform`, and exposes `to_transformed`,
`from_transformed`, `log_jacobian`, `jacobian_grad`, `transform_deriv` and
`transformed_sd` (`:106-207`). The prior is evaluated as
`Prior::log_density(θ, z)` or `log_density_env(θ, z, env)` (`prior.rs:318-335`).

Every use site was found by the compiler. In a scratch copy of `87a426a9`, the
`Transform` variants, `EstimatedParam::{lower, upper}`,
`EstimatedParam::transformed_sd`, both prior density methods and
`pgas::prior_log_density_and_grad_z` were marked `#[deprecated]`, and this was
run:

```
RUSTFLAGS="--cap-lints warn" cargo check --workspace --all-targets --message-format=short
```

It reported 381 distinct source lines in 46 files. Of these, 279 are in tests:
the `sim/tests/` files, `nuts_target_tests.rs`, and the in-file `#[cfg(test)]`
modules. The other 102 lines are in 16 non-test files:

| file                          | non-test lines using the old API                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `sim/src/inference/types.rs`  | 21: the methods themselves, `:108-206`                                                                                                                                                                 |
| `sim/src/inference/prior.rs`  | 4: `:322-323`, `:332-333` (the `Prior` wrappers)                                                                                                                                                       |
| `sim/src/inference/if2.rs`    | 5: `transformed_sd` at `:428`, `:533`, `:639`; `Transform::Log` at `:433`, `:537`                                                                                                                      |
| `sim/src/inference/pmmh.rs`   | 2: prior called with z at `:601`, `:698`                                                                                                                                                               |
| `sim/src/inference/pgas.rs`   | 8: prior with z at `:3561`, `:4068`, `:4445-4446`, `:4548`; `lower`/`upper` at `:3960-3961`; `prior_log_density_and_grad_z` at `:4293`                                                                 |
| `cli/src/fit/runner.rs`       | 28: log-spacing match `:256`; `lower`/`upper` `:737-738`, `:1376-1377`, `:1397`, `:1403`, `:2529`; launch table `:1059-1080`; selection `:1181-1213`; auto step `:1431-1443`; prior check `:3602-3634` |
| `cli/src/survey.rs`           | 16: `lower`/`upper` `:321`, `:327`, `:682-683`, `:1684-1719`, `:1762`; `Transform::Log` `:907`, `:1694`                                                                                                |
| `cli/src/fit/init.rs`         | 6: `:61-62`, `:214-227` (LHS mapping)                                                                                                                                                                  |
| `cli/src/fit/chain_starts.rs` | 4: `:659-660`; `kept_boxes` `:830-833`                                                                                                                                                                 |
| `cli/src/roles.rs`            | 3: `transform_tag` `:428-430`                                                                                                                                                                          |
| `cli/src/fit/config_v2.rs`    | 2: `validate_hierarchical_leaf_boxes` `:2658-2659`                                                                                                                                                     |
| `cli/src/fit/pmmh.rs`         | 2: prior with z `:669`; `transformed_sd` `:1051`                                                                                                                                                       |
| `cli/src/fit/pgas.rs`         | 1: prior with z `:836` (the `log_posterior` trace column)                                                                                                                                              |
| `cli/src/fit/nlopt_stage.rs`  | 1: `lower`/`upper` `:107`                                                                                                                                                                              |
| `cli/src/profile.rs`          | 1: `lower`/`upper` `:1675`                                                                                                                                                                             |

The test lines include three that assert today's wall, which commit 5 inverts
(§11):

- `sim/tests/nuts_validity.rs:366`,
  `boxed_log_wall_produces_divergences_under_nuts`, which asserts at least 100
  divergences;
- `nuts_validity.rs:173`, which asserts the log-Jacobian is −∞ outside the box;
- `sim/src/inference/nuts_target_tests.rs:101`, which asserts −∞ outside the
  box.

The tests also include the hand-built transforms in `pgas.rs`'s test module at
`:5097`, `:5105` and `:5179`. `nuts_target_tests.rs:32` imports
`pgas::prior_log_density_and_grad_z`, so the commit that moves that function
must also move this import.

How each method uses the transform today. Each claim was verified at the lines
cited in the table above:

- **IF2** perturbs each particle in z and maps back, with no Jacobian
  (`if2.rs:428-430`, `:533-535`). The step is `transformed_sd(rw_sd, θᵢ)`: the
  natural-scale `rw_sd` divided by |g′| at that particle's current value. IF2
  also counts clamp activations into `clamp_fraction` (`:226`, `:405`, `:723`),
  which nothing reads (`grep -rn clamp_fraction rust/crates` finds only those
  lines).
- **PMMH and MH** propose in z and add the log-Jacobian (`pmmh.rs:622-624`,
  `:686`, `:707-714`). Their initial proposal is `5 × transformed_sd(rw_sd, θ₀)`
  (`cli/src/fit/pmmh.rs:1049-1052`).
- **PGAS** does the same under MH-within-Gibbs (`pgas.rs:4428-4449`). Under
  NUTS, the default (`config_v2.rs:1759`), it adds the Jacobian and its gradient
  (`pgas.rs:4284-4300`). The MH-within-Gibbs initial scale is one tenth of
  `to_transformed(upper.min(1e10)) − to_transformed(lower.max(1e-10))`
  (`pgas.rs:3957-3965`).
- **NLopt** (`nl-sbplx`, `nl-bobyqa`, `nl-lbfgs`) uses no transform. It
  maximises on the natural scale with NLopt's native box
  (`sim/src/inference/deterministic.rs:177-243`), passed from
  `nlopt_stage.rs:107`.
- **Profile and survey** build their parameters through
  `build_if2_params_from_specs` (`runner.rs:1336-1384`).

### 3.3 The log map's box is a wall, and four more defects sit beside it

**The wall.** `from_transformed` computes `z.exp().clamp(lo, hi)`
(`types.rs:123`). Outside [ln lo, ln hi], `log_jacobian` is −∞, and
`transform_deriv` and `jacobian_grad` are 0 (`log_saturated`,
`types.rs:100-102`). So both ends of a two-sided box, and a lower end above
zero, are walls.

Measured on main (`sim/tests/nuts_validity.rs:383-389`): the NUTS kernel on ln τ
under an exponential(5.99) prior, boxed to [10⁻⁶, 2.5], gives 271 divergences
over 4 chains × 4,000 kept draws (per chain 83, 81, 63, 44, seeds 1–4). Over ten
disjoint seed sets the total ranged 208–288. The unboxed target diverges at most
0–4 per chain (`c8b6ee52`, `feat/bsplines-s2`). gh#470 asked for the smooth
parameterisation after gh#374 made the clamped map's derivatives consistent.

**A `log_normal` prior reads z as ln θ, and today's check does not prevent it.**
`Density::TransformedNormal` (the `log_normal` family) computes its density from
the `transformed` argument on the assumption that z = ln θ (`prior.rs:203-218`).
Its NUTS gradient arm returns −(z − μ)/σ² − 1 on the same assumption
(`pgas.rs:3581-3590`). `validate_prior_transform_compat` (`runner.rs:3572-3690`)
is meant to require `Log` for this family. It derives the transform with
`derive_transform(ir_param, override)` (`runner.rs:3599`), which uses
`default_search_box`, the model's box. The engine uses the fit box. So a `rate`
with no declared range, a fit box [−1, 1] and `~ log_normal(-1, 0.5)` passes the
check (the model's box is [0, ∞), so the derived transform is `Log`), while the
engine runs `Logit { −1, 1 }` and the prior reads its logit coordinate as ln θ.
Reproduced by an independent check under MH: posterior mean 0.080 (sd 0.071)
against the correct 0.398 (sd 0.183).

**Auto `rw_sd` freezes a positive kind boxed to [0, hi] under IF2.**
`auto_rw_sd_from_value` floors lo at 10⁻³⁰⁰ for `Log` and returns
√(lo·hi)·ln(hi/lo)/20 (`runner.rs:1431-1438`). For [0, 2] that is
√(2·10⁻³⁰⁰)·ln(2·10³⁰⁰)/20 ≈ 4.9·10⁻¹⁴⁹ (verified by evaluating that
expression). The step at θ is that over θ, so a parameter with an auto step does
not move. Inferred from the code and the arithmetic; not executed. 309
positive-kind declarations with lower bound 0 and a finite upper bound exist in
this repository's goldens and tests and the sibling model repositories
(verified,
`grep -rhoE '^\s*\w+\s*:\s*(rate|positive|count|duration)\s+in\s*\[\s*0(\.0*)?\s*,' --include='*.camdl'`).
§12.1 adds a red test.

**The logit log-Jacobian underflows.** It is computed as ln((hi − lo)·p·(1 − p))
with p = σ(z) (`types.rs:153`). In double precision 1 − σ(z) is exactly 0 for z
≥ 37 (verified, `python3 -c "import math; print(1-1/(1+math.exp(-37)))"` →
`0.0`), so the log-Jacobian is −∞ there: an accidental wall far in the upper
tail.

**A false doc comment.** `transformed_sd` says it "matches pomp's convention:
the user specifies `rw.sd` on the natural scale" (`types.rs:196`). pomp's `mif2`
documentation says the opposite: "If parameter transformations have been
supplied, then the perturbations are applied on the transformed (estimation)
scale". The source is pomp's `man/mif2.Rd`; the sentence is present at pomp
6.5.0.0, read from GitHub master on 2026-10-28, and gh#1001 cites 6.4.0.6. The
`EstimatedParam.rw_sd` field doc (`types.rs:56-57`) already says "on the
_transformed_ scale". Series (ii) resolves both (§11).

**Half-bounded boxes are reachable only through an override.** The DSL has no
infinity literal, `1e400` makes `camdlc` emit `Infinity`, which `camdl` refuses,
and an `inf` fit-config bound is refused when the fit is hashed. All three were
reproduced by an independent check. So an effective domain with one finite end
arises only as a kind's [0, ∞) or from hand-built code. The one reachable
mis-enforcement is an explicit `transform = "identity"` on a rate's [0, ∞),
which `check_transform_keeps_box` accepts because one end is infinite
(`runner.rs:1243-1256`); it is refused by the interim fix (§3.4).

### 3.4 An interim fix is landing on main now

A separate change, not on main at `87a426a9`, is landing four refusals and
corrections. This proposal builds on them and does not re-propose them:

- a fit box outside the kind's support is refused;
- `validate_prior_transform_compat` checks the transform the engine uses, taken
  from the fit box, which closes the `log_normal` reproduction of §3.3;
- a `beta` prior on a sub-box of [0, 1] is handled consistently;
- any `transform` override that does not keep the support is refused.

With the first refusal, a search box always lies within the support, so the
effective domain is the search box. The interim fix owns the tests it changes,
among them `cli/tests/fit_bounds_transforms.rs:182` and `:239` (the gh#967
`duration` box [−1, 1] on a parameter with no declared range).

## 4. Support, search box and effective domain stay separate; the transform is the effective domain

### 4.1 The types

The sim crate's transform becomes the representation of the effective domain.
Each variant carries exactly the finite ends it keeps:

```rust
// rust/crates/sim/src/inference/types.rs

/// The effective domain of an estimated parameter, as the map onto it.
/// Built only by `Transform::for_domain`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    Identity,                          // (−∞, ∞): θ = z
    LowerBounded { lo: f64 },          // [lo, ∞): θ = lo + eᶻ
    UpperBounded { hi: f64 },          // (−∞, hi]: θ = hi − eᶻ
    Interval { lo: f64, hi: f64 },     // [lo, hi], otherwise: θ = lo + (hi − lo)·σ(z)
    LogInterval { lo: f64, hi: f64 },  // [lo, hi], 0 < lo, positive kind: ln θ = ln lo + ln(hi/lo)·σ(z)
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    NotAnInterval { lo: f64, hi: f64 },   // NaN, lo ≥ hi, lo = +∞ or hi = −∞
}

impl Transform {
    /// The map for the domain with these ends; an infinite end is open.
    /// `positive_kind` is true for `rate`, `positive`, `count`, `duration`.
    pub fn for_domain(lo: f64, hi: f64, positive_kind: bool) -> Result<Self, DomainError>;
    /// The domain's ends, (−∞, ∞) for `Identity`.
    pub fn bounds(&self) -> (f64, f64);
    /// Whether start draws and Latin-hypercube points are log-spaced.
    pub fn log_spaced(&self) -> bool { matches!(self, Transform::LogInterval { .. }) }
    /// IF2's default step in z (D8), one constant per map family. `None`
    /// for `Identity`, which has no scale.
    pub fn default_if2_step(&self) -> Option<f64>;
    /// PMMH's and MH's default initial proposal SD in z (D8). Adaptation
    /// tunes it; `None` for `Identity`.
    pub fn default_mh_step(&self) -> Option<f64>;
}

pub struct EstimatedParam {
    pub name: String,
    pub index: usize,
    pub initial: f64,
    /// The step in z: IF2's perturbation SD before cooling; MH's initial
    /// proposal is five times it.
    pub rw_sd: f64,
    pub transform: Transform,   // the domain; `lower` and `upper` are removed
    pub rw_sd_auto: bool,
    pub perturb_only_at_t0: bool,
}

impl EstimatedParam {
    pub fn bounds(&self) -> (f64, f64) { self.transform.bounds() }
    // `to_transformed`, `from_transformed`, `log_jacobian`, `jacobian_grad`
    // and `transform_deriv` keep their names and signatures (§5).
    /// θ and the logarithms a prior needs, computed from z (§6).
    pub fn prior_point(&self, z: f64) -> PriorPoint;
}
```

Today `EstimatedParam` carries `lower` and `upper` beside `transform`
(`types.rs:61-64`), and nothing ties them together. Test fixtures build
`Transform::None` with `lower: 0.0, upper: 1.0` (for example
`runner.rs:3688-3692`), a box nothing keeps. Deriving the bounds from the
transform makes that state unrepresentable (D1). The non-test readers of the
fields are listed in §3.2; each becomes `bounds()`.

`log_spaced` answers what `init.rs:221`, `survey.rs:907`, `survey.rs:1694` and
`runner.rs:256` each answer today by matching `Transform::Log`, so no consumer
needs the parameter's kind, which `EstimatedParam` does not carry. If D11 is
decided against `LogInterval`, log spacing can no longer be read off the map.
`EstimatedParam` then needs a `kind: Option<ParamKind>` field, set in
`build_if2_params_from_specs`, and `log_spaced` takes it.

In the CLI the three concepts stay three fields. `roles.rs` already has
`Support` (`:66-72`) and `SearchBox` (`:81-102`); `Estimated` keeps both, and
its `transform` becomes the effective domain:

```rust
// rust/crates/cli/src/roles.rs
pub struct Estimated {
    pub support: Support,        // from the model: declared range, else kind
    pub search: SearchBox,       // from the user, within the support (§3.4)
    /// The effective domain, support ∩ search, as the map onto it.
    pub transform: Transform,
    pub prior: ResolvedPrior,
    pub prior_text: String,
    pub start: Option<f64>,
}

/// support ∩ search. The interim fix refuses a search box that leaves the
/// support, so this is the search box whenever resolution got this far.
pub fn effective_domain(support: &Support, search: &SearchBox) -> Result<Transform, RoleError>;
```

`TransformFrom` (`roles.rs:121-126`) and `Estimated.transform_from` are deleted
with the override (D2). `BoxFrom::Fallback` (`roles.rs:97-101`) is deleted,
since a parameter with no kind is refused (D6). The stored record keeps
`support`, `search` and a `transform` tag (`ParamRecord::Estimated`,
`roles.rs:550-566`). The tag takes the values `identity`, `lower_bounded`,
`upper_bounded`, `interval` and `log_interval`, and the record drops
`transform_from`. The tag is derivable from `search`, but it is kept so that a
record states which map ran even if the selection rule later changes.

The selection rule reads the domain's ends, and the kind only to split a finite
box with 0 < lo:

| lo     | hi     | kind                                    | transform                |
| ------ | ------ | --------------------------------------- | ------------------------ |
| −∞     | ∞      | any                                     | `Identity`               |
| finite | ∞      | any                                     | `LowerBounded { lo }`    |
| −∞     | finite | any                                     | `UpperBounded { hi }`    |
| > 0    | finite | `rate`, `positive`, `count`, `duration` | `LogInterval { lo, hi }` |
| finite | finite | any other case                          | `Interval { lo, hi }`    |

Otherwise the kind enters only through the support. Splitting a box with 0 < lo
by kind is a geometry choice, not a consequence of the support: on a positive
quantity the log-interval map interpolates geometrically between the ends, the
interval map arithmetically, and both keep the same domain (D11). Four rows are
reachable from a model and fit config: `Identity` (an undeclared `real` or
`instant`), `LowerBounded { 0 }` (an undeclared positive kind), `Interval`
(`probability`, a positive kind on [0, hi], a `real` or `instant` box, a signed
box) and `LogInterval` (a positive kind with 0 < lo, the common case for rates).
`LowerBounded { lo > 0 }` and `UpperBounded` are reachable only from hand-built
code (§3.3). They are kept so that every domain shape has a map and `for_domain`
is total.

Where today's choices were sound they are reproduced. A `rate` on [0, ∞) is θ =
eᶻ exactly; `probability` on [0, 1] is the logit; a signed box is the scaled
logit it is today. A rate on [lo, hi] with lo > 0 keeps today's ln θ geometry in
the interior of its box, and the walls go. A positive kind on [0, hi] moves from
a walled log to the interval map.

### 4.2 A finite bound truncates the Bayesian target and restricts the IF2 search

Under PMMH, MH and PGAS the target in z is π̃(z) = p(y | g(z))·p(g(z))·|g′(z)|.
Because g maps onto the domain D, ∫ π̃(z) dz = ∫_D p(y | θ) p(θ) dθ, so the
θ-marginal of the chain is p(θ | y, θ ∈ D): the posterior truncated to D. For a
prior with constant parameters the truncation needs no renormalisation, because
the prior's mass in D does not depend on θ. This is the target the code states
today (`types.rs:27-32`); this proposal changes how it is reached, not what it
is.

For a hierarchical leaf the prior's mass in D is Z(h) = ∫_D p(leaf | h) d(leaf),
which depends on the hyperparameter h, so the truncated target is wrong for h
unless −ln Z(h) is added. That case is already refused by
`validate_hierarchical_leaf_boxes` (`fit/config_v2.rs:2637-2672`, main
`9ccb2899`). A transform alone cannot fix it, because the missing piece is a
normalising term, not a change of coordinates. The check reads the box the
transform enforces and skips `Transform::None` (`config_v2.rs:2657-2659`); it
becomes a read of `bounds()`, which gives the same answer for every reachable
domain.

Under IF2 the domain restricts the search: the estimate maximises the likelihood
over D. The map serves the perturbation kernel and carries no Jacobian, since no
density in θ is integrated (§9.1).

### 4.3 Amendments to the parameter-roles proposal

`2026-10-26-parameter-roles.md` is accepted and partly implemented. This
proposal amends it in three places and supersedes nothing else:

- §6.2: `Estimated.transform`'s comment "clamps to `search`" becomes "the
  effective domain, mapped onto"; `transform_from` and `TransformFrom` go.
- §6.4: "The transform comes from `[estimate].transform`, else the kind, and
  clamps to the search box" becomes "the transform is the effective domain's
  map". The sentence "an identity-transform parameter's box is enforced by
  refusing a proposal outside it (a −∞ log-prior)" goes: no domain needs a
  rejection rule, because every finite end is in the map.
- §6.4's statement that the box is enforced "only by the transform — the `Log`
  inverse clamps to [lo, hi] and its log-Jacobian is −∞ outside it" describes
  the code before this change.

## 5. The five maps, with forms that stay finite

Write w = hi − lo and, for `LogInterval`, L = ln(hi/lo). Every quantity below is
a closed form in z; none is computed by differencing θ.

| map            | θ = g(z)       | z = g⁻¹(θ)                          | g′(z)          | ln\|g′(z)\|                              | d/dz ln\|g′(z)\|         |
| -------------- | -------------- | ----------------------------------- | -------------- | ---------------------------------------- | ------------------------ |
| `Identity`     | z              | θ                                   | 1              | 0                                        | 0                        |
| `LowerBounded` | lo + eᶻ        | ln(θ − lo)                          | eᶻ             | z                                        | 1                        |
| `UpperBounded` | hi − eᶻ        | ln(hi − θ)                          | **−eᶻ**        | z                                        | 1                        |
| `Interval`     | lo + w·σ(z)    | ln(θ − lo) − ln(hi − θ)             | w·σ(z)·σ(−z)   | ln w − softplus(−z) − softplus(z)        | −tanh(z/2)               |
| `LogInterval`  | lo·exp(L·σ(z)) | ln(ln θ − ln lo) − ln(ln hi − ln θ) | θ·L·σ(z)·σ(−z) | ln θ + ln L − softplus(−z) − softplus(z) | L·σ(z)·σ(−z) − tanh(z/2) |

The first four are the maps of the Stan Reference Manual's "Lower bounded
scalar", "Upper bounded scalar" and "Lower and upper bounded scalar" sections
(v2.40, chapter "Constraint Transforms", mc-stan.org, read 2026-10-28). That
chapter gives their absolute derivatives as eᶻ, eᶻ and (b − a)·logit⁻¹(y)·(1 −
logit⁻¹(y)). `LogInterval` is the interval map applied to ln θ on [ln lo, ln
hi]; it is not one of Stan's built-in transforms. The softplus forms are ours:
ln σ(z) = −softplus(−z), ln(1 − σ(z)) = −softplus(z), and the derivative of
their sum is σ(−z) − σ(z) = −tanh(z/2). For `LogInterval`, ln θ = ln lo + L·σ(z)
exactly, and d ln θ/dz = L·σ(z)·σ(−z).

**Stable evaluation.**

- σ(z) is eᶻ/(1 + eᶻ) for z ≤ 0 and 1/(1 + e⁻ᶻ) for z > 0; neither branch
  overflows.
- The two interval maps are evaluated from the nearer end. `Interval`: lo +
  w·σ(z) for z ≤ 0, hi − w·σ(−z) for z > 0. `LogInterval`: lo·exp(L·σ(z)) for z
  ≤ 0, hi·exp(−L·σ(−z)) for z > 0. The distance to the nearer end then keeps
  full relative precision: θ − lo = lo·expm1(L·σ(z)) and hi − θ =
  −hi·expm1(−L·σ(−z)).
- The forward maps difference distances to the ends, not a normalised position
  u, so 1 − u never cancels.
- Both interval log-Jacobians are finite for every finite z. `Interval`'s is
  close to ln w − |z| in both tails, which removes the z ≥ 37 underflow of §3.3;
  its gradient −tanh(z/2) is bounded by 1.
- g′(z) for the interval maps is computed as the exponential of the
  log-Jacobian. It underflows to 0 only for |z| > 745. There the log-Jacobian,
  computed from z, still carries the gradient, and a zero only drops the
  likelihood's contribution, which is negligible that far from the data.

**The sign of the upper-bounded derivative.** `transform_deriv` returns the
signed g′(z), −eᶻ for `UpperBounded`. Every caller multiplies by it in the chain
rule (`pgas.rs:4286-4289`, and the prior arms that move to `prior.rs`, §6),
which is correct for a negative derivative. `log_jacobian` uses |g′| = eᶻ. No
caller takes the logarithm of `transform_deriv` (the call sites are those of
§3.2).

**Values that round to an endpoint.** θ = lo + eᶻ equals lo in floating point
once eᶻ is below half an ulp of lo; for lo = 1 that happens at z ≤ −37
(verified, `python3 -c "import math; print(1+math.exp(-37)==1)"` → `True`). The
interval maps do the same at either end. The rounded θ is still in the closed
domain, so evaluating the model there is legitimate. What must not round is
anything singular there: the log-Jacobian and the prior's logarithms of θ and of
its distances to the ends are computed from z (§6). The Stan Reference Manual
states the same limitation for its own transforms in "Limitations due to finite
accuracy presentation": "if the closest floating-point number for the inverse
transformed value is the boundary, then the value is rounded to the boundary."

**Overflow.** θ = lo + eᶻ is +∞ for z > ln(f64::MAX) ≈ 709.78. The likelihood or
prior at an infinite θ is non-finite, which every consumer already treats as a
rejected proposal or a divergence (`types.rs:141-146`). This matches today's
unboxed `Log { 0, ∞ }`, at a z no posterior reaches. The interval maps cannot
overflow. `from_transformed(±∞)` returns the corresponding end, so the map is
total.

**A start exactly on a bound** has no finite preimage: z = ln 0 = −∞. Such a
start is refused before any method runs (D3). A start given by the user — a
config `start`, a `--starts` file or a sourced rule (`from_mle`, `from_params`)
— is refused with an error that names the parameter and the bound, gives an
interior value to start from instead (1% of the width inside the end, or
0.01·max(1, |bound|) inside a half-bounded domain's end), and says that if the
optimum on the bound is the scientific point, `nl-sbplx` (ODE backend) or a
profile can place an estimate on it. A start the fit draws (`uniform`, `lhs`,
`uniform_unconstrained`, `from_prior`) that lands within rounding of a bound is
redrawn, as a start the filter cannot score is.

The forward map is still total: a value on an end maps to z = ±ln(1/ε − 1) ≈
±23.03 on the interval maps and to z = ln ε on the half-bounded ones, with ε =
10⁻¹⁰, the existing `PROB_FRACTION_EPS` (`types.rs:228`). No start reaches it
there; the values that can are a resumed state, which carries its own z, and a
debug assertion's backstop. A value outside the domain never reaches the forward
map, because it is refused when values are resolved (parameter-roles §5.4). The
forward map's clamp of such a value stays only as a `debug_assert!` backstop,
and no caller evaluates the forward map at a point outside the domain (§9.2
replaces the one that does).

## 6. Priors read logarithms of θ computed from z

The prior's API changes from `(natural, transformed)` to a point that carries θ
and the logarithms a density needs:

```rust
// rust/crates/sim/src/inference/prior.rs
pub struct PriorPoint {
    pub theta: f64,
    pub ln_theta: f64,          // ln θ; NaN unless the domain is within [0, ∞)
    pub ln_one_minus: f64,      // ln(1 − θ); NaN unless the domain is within (−∞, 1]
    pub d_ln_theta_dz: f64,     // for the NUTS gradient
    pub d_ln_one_minus_dz: f64,
}
impl Prior {
    pub fn log_density_env<E: ParamEnv>(&self, at: &PriorPoint, env: &E) -> f64;
    /// Value and z-gradient. Moves here from `pgas.rs:3550-3627`.
    pub fn log_density_and_grad_z(&self, at: &PriorPoint, dtheta_dz: f64) -> (f64, f64);
}
```

`EstimatedParam::prior_point(z)` computes the logarithms from the distances to
the ends, which are exact in z. ln(θ − lo) is z for `LowerBounded` and ln w −
softplus(−z) for `Interval`; ln(hi − θ) is z for `UpperBounded` and ln w −
softplus(z) for `Interval`. Then ln θ = logaddexp(ln lo, ln(θ − lo)) when lo ≥
0, which is ln(θ − lo) itself at lo = 0. For `LogInterval`, ln θ = ln lo +
L·σ(z) directly. Similarly ln(1 − θ) = logaddexp(ln(1 − hi), ln(hi − θ)) when hi
≤ 1. The derivatives are d ln θ/dz = exp(ln|g′(z)| − ln θ) and d ln(1 − θ)/dz =
−g′(z)/(1 − θ), each evaluated through the logarithms.

Each density that takes a logarithm reads it from the point: `log_normal` (ln
θ), `log_uniform` (ln θ), `gamma` (ln θ), `beta` (ln θ and ln(1 − θ)). Densities
that take no logarithm (`normal`, `half_normal`, `exponential`, `uniform`,
`truncated_normal`, `flat`) read θ.

The guards on θ go with the change. The densities return −∞ when θ is at or past
an end of their support: `log_normal` at `natural <= 0.0` (`prior.rs:213`),
`half_normal` (`:224`), `beta` (`:236`), `gamma` (`:249`) and `exponential`
(`:260`). The gradient arms return a zero gradient early on the same tests
(`pgas.rs:3574`, `:3592`, `:3597`, `:3602`, `:3607`, `:3612`, `:3620`). Under
the new maps θ is never outside the domain, and θ rounding to an end is not a
reason to drop a density whose logarithms are finite. Each guard becomes a test
of whether the needed logarithm is finite.

For `LowerBounded { 0 }`, ln θ = z exactly, so the `log_normal` value and
gradient equal today's for z ≥ −745. Below that, θ = eᶻ underflows to 0 and
today's guard returns −∞; the new value is finite and correct. The equality
therefore holds on z ≥ −745, not everywhere.

`validate_prior_transform_compat`, as corrected by the interim fix (§3.4),
becomes a support rule: **the effective domain must lie within the prior
family's support.**

- `log_normal`, `gamma`, `exponential` and `half_normal` need D ⊆ [0, ∞).
- `beta` needs D ⊆ [0, 1].
- `log_uniform(L, U)` and `uniform(L, U)` need D ⊆ [L, U], checked when the ends
  are constants.
- `normal` and `flat` accept any D.
- `truncated_normal` keeps its rule that its truncation equals the box
  (`runner.rs:3647-3686`).

A prior whose support is narrower than D would put zero density inside the
domain, which is a wall by another route, so it is refused. This is a language
choice, not a mathematical necessity: the truncated target would still be
proper, and a sampler could treat the zero-density region as rejected moves.
camdl refuses it because a box that reaches outside its prior's support is
almost always a mistake in the box or in the prior, and the refusal says which.
Every family is then truncated to its box. `Density::transform_req`,
`Prior::transform_req` and `TransformReq` (`prior.rs:82`, `:142-155`, `:348`)
are deleted (D4).

## 7. An explicit transform serves no purpose, so the field is deleted

The interim fix (§3.4) refuses an override that does not keep the support. Under
the new selection an override keeps the effective domain only if it names the
map the domain already selects, so every accepted override is redundant.
`[estimate].transform` (`config_v2.rs:588`), `config_v2::Transform`
(`:628-648`), `ParamSpec.transform` (`runner.rs:1320`),
`check_transform_keeps_box` and the threaded `Option<&str>` are deleted (D2).
`EstimateSpecV2` has `deny_unknown_fields` (`config_v2.rs:573`), so a config
that still carries the key is refused at load, naming it. Under the alpha
posture there is no compatibility path. §13 gives the migration.

## 8. No parameter is discrete; `count` is continuous and rounded where it is used

`ParamKind` has seven variants and no integer one
(`ir/src/parameter.rs:132-140`). `integer` is a compartment type keyword, not a
parameter kind (language spec §14.2, "Reserved Identifiers",
`docs/camdl-language-spec.md:3995-3998`). The AST's `param_type` is
`PRate | PProbability | PPositive | PCount | PReal | PInstant | PDuration`
(`ocaml/lib/compiler/ast.ml:129`). Every parameter value is an `f64`.

A `count` parameter is a continuous quantity of dimension P (language spec
§2.2.1, `:347`) and is estimated as such. Where an expression over it
initialises an integer compartment, the value is rounded at that point
(`sim/src/compiled_model.rs:2420-2421`). Verified by reading.

So a `count` such as `I0` that reaches the model only through an integer initial
state has a likelihood that is a step function of its value, and NUTS sees a
zero gradient in that direction. That comes from rounding at use, not from the
transform, and is out of scope (§15). No genuinely discrete parameter exists to
treat separately.

## 9. Each method uses the shared map

### 9.1 IF2 takes a fixed step in z, and cannot place an estimate on a bound

IF2 perturbs in z, maps back, and uses no Jacobian. Two things change.

**The step is fixed in z.** Today's step at particle i is rw_sd/|g′(z(θᵢ))|
(`if2.rs:428`, `:533`). Under the interval map |g′| goes to 0 at both ends, so
the step grows without bound for particles near an end and throws them further
out. Reproduced by an independent check, with a flat likelihood and `beta` on
[0.05, 1] over 50 uncooled steps:

| step rule                           | particles within 10⁻³ of an end |
| ----------------------------------- | ------------------------------- |
| today (walled log, per particle)    | 2.0%                            |
| interval map, per-particle division | 25.3%                           |
| interval map, fixed step in z       | 12.1%                           |

Under a flat likelihood a random walk in z spreads without limit, so some mass
near the ends is expected under any rule. The per-particle rule doubles it.

Converting once at the start instead makes each chain's step depend on where it
starts. Reproduced by the same check: 0.21 at θ₀ = 0.25, 3.4 at θ₀ = 0.06, and
about 3·10⁸ at a start on an end.

The step is therefore a property of the domain. `EstimatedParam.rw_sd` holds the
step in z, set once when the parameter is built and the same for every chain and
particle. It is the kernel Ionides et al. (2015) analyse: a perturbation of
fixed shape on the estimation scale, shrunk by geometric cooling (Ionides,
Nguyen, Atchadé, Stoev and King, "Inference for dynamic and latent variable
models via iterated, perturbed Bayes maps", _PNAS_ 112(3):719–724, Algorithm 1).
It is also the kernel pomp's `mif2` applies (§3.3). The default is
`Transform::default_if2_step`, one constant per map family — interval,
log-interval, and the half-bounded maps — measured by §12.6 (D8). Every map with
a finite end has one, so a `rate` on [0, ∞) runs without an explicit `rw_sd`;
`Identity` has no scale and still needs one. An explicit `rw_sd` is converted to
z once, in series (i), and read in z directly from series (ii) on (§11, D9).
Where no single constant serves a family's fixtures, the launch table prints
each parameter's default step together with the natural-scale step it implies at
the box's midpoint, so the user sees the scale and can override it.

**A maximum on a bound is approached, not reached.** Under the walled log a
perturbation past an end was clamped onto it, so the swarm could hold mass
exactly at the end. Under a smooth map an end is z = ±∞. With geometric cooling
the perturbation variances form a convergent series, so the swarm's total
displacement is finite and its distance to a bound levels off rather than going
to zero. How close it gets depends on the likelihood's slope toward the bound in
z. A steep rise draws the swarm far out during the early, large-step iterations,
and the final estimate's likelihood is within Monte Carlo error of the boundary
value. A shallow rise stops the swarm at a visible distance. A maximum on a
bound is a legitimate result, and IF2 alone cannot say whether the maximum is on
the bound.

The near-bound diagnostic, `ParamNearBound` (`runner.rs:2514-2547`), fires when
a chain's estimate is within 1% of the box's width from either end, and only
when both ends are finite. A domain with an infinite end gets no report: it has
no width, and telling a maximum at the bound from a small interior one needs the
likelihood, which is a separate design (§15) (D10). The message gains one
sentence: IF2 cannot place an estimate exactly on a bound; to test whether the
maximum is on it, profile the parameter or refit with `nl-sbplx` (ODE backend
only), which can (§9.3).

`clamp_fraction` measures an event that no longer exists. It is computed and
never read (§3.2), so it is deleted in its own commit first (§11).

### 9.2 PMMH, MH and PGAS add the Jacobian, and start MH from a default step

The acceptance ratio and the NUTS target are unchanged in form: likelihood plus
log-prior plus `log_jacobian`, with `jacobian_grad` and `transform_deriv` in the
gradient (`pmmh.rs:707-714`, `pgas.rs:4284-4300`, `pgas.rs:4444-4449`). Only the
prior's argument changes (§6). Because no map has a wall, a NUTS trajectory near
a bound sees the log-Jacobian's tail instead of a cliff.

PMMH and MH start their proposal at `Transform::default_mh_step`, a constant in
z, or at five times an explicit `rw_sd` converted to z, in place of the
conversion at θ₀ (`cli/src/fit/pmmh.rs:1049-1052`). Their own default is
separate from IF2's because the two steps do different jobs: IF2's sets the
perturbation kernel the whole run uses, while the MH proposal is adapted from
the first warm-up step, so its starting value matters little. PGAS's
MH-within-Gibbs initial scale (`pgas.rs:3957-3965`) follows the same rule.

The formula it replaces is not one tenth of the domain's z-width. It evaluates
the forward map at `lower.max(1e-10)` and `upper.min(1e10)`. For a box lying
below 10⁻¹⁰, such as an `instant` on [−120, −60], both points are outside the
domain, the clamped logit maps both to the same z, and the scale is the floor
0.01. Under the new forward map the same evaluation would trip the
`debug_assert!` of §5. A parameter with no default (an unbounded `Identity`) and
no explicit `rw_sd` starts MH-within-Gibbs at 1.0 in z. Robbins–Monro adaptation
(`pgas.rs:3941-3950`) tunes the scale during burn-in in every case. NUTS, the
default, uses none of this.

IF2, PMMH and MH refuse an `Identity` parameter with no explicit `rw_sd`
(`runner.rs:369-378`, `refuse_degenerate_auto_rw_sd`); every other map has a
default.

### 9.3 NLopt stays on the natural scale; moving it is a measured follow-up

**Option A: optimise in z.** The objective becomes ℓ(g(z)). L-BFGS's gradient is
∇θℓ ⊙ g′(z), formed in `fit/nlopt_stage.rs` from `det_grad`'s ∇θ. Every end is
at z = ±∞, so a maximum on a bound is reachable only if NLopt's z-box is clipped
at the finite ends the forward map of §5 gives a value on a bound (z ≈ ±23.03
for the interval maps). That places it within 10⁻¹⁰ of the end in the map's own
scale.

The gain is conditioning: a rate that spans orders of magnitude is better scaled
in ln θ, and Sbplx's simplex and Bobyqa's trust region are set in coordinate
units. The cost is the stopping rule. `xtol_rel` is a _relative_ tolerance:
"stop when an optimization step … causes a relative change the parameters by
less than `tol`" (NLopt Reference, "Stopping criteria", nlopt.readthedocs.io,
read 2026-10-28). An interval coordinate near z = 0 has no meaningful relative
change, so option A must also move the stopping rule to `xtol_abs`.

**Option B: keep the natural scale.** NLopt enforces the box natively, and its
reference says so: "NLopt guarantees that your objective function and any
nonlinear constraints will _never_ be evaluated outside of these bounds" (NLopt
Reference, "Bound constraints"). A maximum exactly on a bound is reachable.

The MLE is invariant under reparameterisation in exact arithmetic, so B changes
no estimand. Two kinds of quantity depend on coordinates: a density's mode (a
MAP estimate, whose density carries the Jacobian), and curvature-based standard
errors (Hessian, Wald, Laplace). camdl computes no curvature-based errors
(verified:
`grep -rn -i "hessian\|wald\|laplace" rust/crates/cli/src rust/crates/sim/src`
finds no such computation). Its "MAP" is the draw with the highest natural-scale
log-likelihood plus log-prior, with no Jacobian
(`sim/src/inference/pmmh.rs:747-749`). The PGAS `log_posterior` column likewise
excludes it (`cli/src/fit/pgas.rs:836-839`). So the MAP draw does not depend on
the sampling coordinate.

Posterior means and quantiles of θ are expectations under the target measure,
which the coordinate does not change; only their Monte Carlo error changes. The
draws are stored on the natural scale.

Decided (D5): B in this proposal, and A under gh#1002. gh#1002 already specifies
the benchmark: iterations to convergence and final log-likelihood, with
`nl-bobyqa` on the natural scale against the log scale. It must also switch to
`xtol_abs`. gh#1002 also audits existing optimiser-against-sampler comparisons.
Its question whether the PMMH "MAP" carries the Jacobian is answered above: it
does not.

### 9.4 Starts and Latin-hypercube points follow `log_spaced`

`lhs_map_to_natural` (`fit/init.rs:214-227`), `survey`'s start draw
(`survey.rs:900-909`), its pin-fraction diagnostic (`survey.rs:1693-1699`) and
the seeded start of gh#34 (`runner.rs:248-257`) each decide "log-spaced" today
by matching `Transform::Log`. They call `Transform::log_spaced` instead. Today a
positive kind on 0 < lo < hi < ∞ is log-spaced, which `LogInterval` keeps.

No case changes. A positive kind on [0, hi] is linear-spaced today, because
`init.rs:221` also requires lo > 0, and it stays linear-spaced; `probability`,
`real` and `instant` boxes take `Interval` and stay linear-spaced. `kept_boxes`
(`chain_starts.rs:822-836`) becomes `bounds()` for every parameter, since every
domain is kept.

### 9.5 Profile and survey inherit the change

Both build through `build_if2_params_from_specs`, so they need no change beyond
the deleted override (`profile.rs:889` and `runner.rs:714` pass it today).
Simplex groups use their own log-ratio map (`if2.rs:58-67`) and are untouched.

## 10. The high-risk files need these edits; the implementation needs two reviews

The edits to the high-risk files, taken from the compiler listing of §3.2:

- **`if2.rs`**:
  - commit 1: delete `clamp_fraction` (`:226`), `clamp_counts` (`:405`), the two
    `if let Transform::Log` blocks (`:433-437`, `:537-541`) and the fill at
    `:723`;
  - commit 6: `:428`, `:533` and `:639` read `spec.rw_sd`, which now holds the
    step in z, instead of `spec.transformed_sd(spec.rw_sd, current)`.

  The perturbation itself (`:429-430`, `:534-535`) is unchanged.
- **`pmmh.rs`**, commit 4: `:601` and `:698` pass `p.prior_point(z)`. The
  proposal, the Jacobian and the acceptance (`:622-714`) are unchanged.
- **`pgas.rs`**:
  - commit 4: `prior_log_density_and_grad_z` (`:3550-3627`) moves to `prior.rs`
    as `Prior::log_density_and_grad_z`. Its call at `:4293` follows it, and the
    prior calls at `:3561`, `:4068`, `:4445-4446` and `:4548` pass
    `prior_point`;
  - commit 6: the MH-within-Gibbs initial scale (`:3957-3965`, reading
    `lower`/`upper` at `:3960-3961`) becomes 5 × `rw_sd`, else 1.0 (§9.2), and
    the test module's hand-built transforms (`:5097`, `:5105`, `:5179`) move to
    the new variants.

  The NUTS target's Jacobian lines (`:4286-4300`) and the MH-within-Gibbs ratio
  (`:4428-4449`) are unchanged.
- `particle_filter.rs`, `obs_loglik.rs`, `obs_model.rs`, `pgas_grad.rs` and
  `nuts.rs`: no use site in the listing.

Every one of these files changes behaviour even where its text does not, because
the map it calls changes. The implementation therefore needs two independent
reviews before it lands. Each re-runs the validation of §12 rather than reading
it, and re-runs the deprecation listing of §3.2 against its own base, so that no
site is missed.

## 11. Two commit series; each commit is green and bisectable

Both series ship in one release. Series (ii) follows (i), and either can be
reverted alone. Each commit passes `make test`. The interim fix (§3.4) lands
before both.

**Series (i), the transform refactor.**

1. Delete IF2's `clamp_fraction` (dead code, its own commit).
2. Delete `[estimate].transform` (D2). After the interim fix every accepted
   override is redundant, so results do not change. The 14 lines in this
   repository's code, tests and docs that set one are removed in the same
   commit.
3. Refuse a parameter whose IR carries no kind (D6), and give the hand-built
   test IR that has none a kind.
4. Introduce `PriorPoint` and move the prior gradient to `prior.rs`, under the
   existing transforms, with `nuts_target_tests.rs:32`'s import updated. In this
   commit `log_normal` reads `ln_theta`, which under `Log` is z, as today. The
   other families keep their θ-based formulas and guards until commit 5. So the
   commit is byte-identical; its check is a before-and-after diff of a PMMH and
   a PGAS+NUTS run of `tests/recovery/cases/seir_age` at a fixed seed.
5. Replace `Transform` and derive `EstimatedParam`'s bounds. Implement §5's
   numerics, the selection of §4.1, the support rule and the logarithm-based
   densities of §6 with their guards removed, and `log_spaced`. Extend the
   near-bound diagnostic. Add §12.1's tests. Invert the three wall tests of
   §3.2:
   - `boxed_log_wall_produces_divergences_under_nuts` becomes a test that the
     boxed parameter diverges no more than the unboxed one;
   - `nuts_validity.rs:173` becomes a test that the log-Jacobian is finite
     everywhere;
   - `nuts_target_tests.rs:101` becomes the analytic target under `LogInterval`.

   This commit is unavoidably large, because the enum change breaks every match
   on it.
6. Refuse a start on a bound (D3, §5), each in its own commit: a user-given
   start is refused with the interior value to use; a drawn start within
   rounding of a bound is redrawn.
7. Fix the step in z (§9.1, §9.2). `build_if2_params_from_specs` sets `rw_sd` to
   `default_if2_step()`, or converts an explicit natural-scale `rw_sd` once at
   the domain's reference point. That point is z = 0 for the interval maps (the
   box's midpoint, geometric for `LogInterval`) and θ₀ for the others. IF2 reads
   `rw_sd` directly. PMMH, MH and PGAS's MH-within-Gibbs start at
   `default_mh_step()`, or at 5 × an explicit `rw_sd`. The launch table
   (`runner.rs:1059-1080`) prints the map, the z-step and the natural-scale step
   it implies at the midpoint. Its `CompressedLogitPosition` warning is computed
   on the map's own z. This commit changes IF2 and MH dynamics and is the one
   the benchmark measures.
8. The near-bound report for boxes finite at both ends only (D10, §9.1).
9. The IF2 constants from §12.6.
10. Documentation:

- `docs/camdl-inference-spec.md` §3.2 rule 2 (the transform table at
  `:152-187`);
- `docs/fit-toml.md` and `docs/agents.md` (the override);
- the parameter-roles amendments of §4.3.

**Series (ii), the `rw_sd` scale (gh#1001).**

11. An explicit `rw_sd` is read in z directly, as pomp's `mif2` reads `rw.sd`.
    The conversion of commit 7 and `transformed_sd` are deleted, along with its
    remaining callers: the conversion in `build_if2_params_from_specs`,
    `runner.rs:1080` and `cli/src/fit/pmmh.rs:1051`. So is the false doc
    comment. The field doc becomes true. Default runs do not change.
12. Documentation: inference spec `:86-91` and `docs/fit-toml.md`.

Between commits 7 and 11, an explicit natural-scale `rw_sd` on
`LowerBounded { 0 }` is converted at θ₀, which still makes it start-dependent.
This intermediate state is never released.

## 12. Validation, with pass criteria

### 12.1 Deterministic tests of the maps and the priors

These tests live in `sim/src/inference/types.rs` and `prior.rs`. They run for
each map and each of the domains [0, ∞), [0, 1], [0, 2], [10⁻⁶, 2.5], [1, 100],
[−120, 120] and [−120, −60], plus `LowerBounded { 10⁻⁶ }` and
`UpperBounded { 30 }` as hand-built cases:

- round trip g(g⁻¹(θ)) = θ for interior θ, within 4 ulp of the distance to the
  nearer end; g⁻¹(g(z)) = z for |z| ≤ 30, within 10⁻¹²;
- `transform_deriv`, `jacobian_grad` and `prior_point`'s derivatives against
  central finite differences, relative error below 10⁻⁶ on |z| ≤ 30;
- `log_jacobian` finite for every |z| ≤ 700, and g(z) inside the closed domain
  for z ∈ {±40, ±800, ±∞}. This is red on main for the logit at z = 40;
- `UpperBounded`'s `transform_deriv` is negative;
- prior **densities**, not only `ln_theta`, against direct evaluation where θ is
  representable: `log_normal`, `gamma`, `beta` and `log_uniform` on each domain
  in their support. Under `LowerBounded { 0 }` at z = −800, the `log_normal`
  density must be finite and equal its closed form in z;
- `log_normal` on `LowerBounded { 0 }` equals today's value and gradient bit for
  bit on z ∈ [−745, 30];
- a value on each end maps to the finite z of §5; a user-given start on an end
  is refused naming an interior value, and a drawn one is redrawn;
- `default_if2_step` and `default_mh_step` for each map; auto `rw_sd` for a rate
  on [0, 2] gives a step in z above 0.1 (red on main, §3.3), and a rate on [0,
  ∞) gets one too.

Each test is mutation-checked: reverting its fix turns it red.

### 12.2 A red test for the prior read in the wrong coordinate

The reproduction of §3.3, as a `cli` integration test, is the interim fix's
regression test: a `rate` with no declared range, `~ log_normal(-1, 0.5)`, fit
box [−1, 1], under MH. It is red on `87a426a9` (the posterior mean is 0.080
against 0.398) and green once the interim fix refuses the box. This proposal
adds the companion that must run: the same prior with fit box [0.05, 1] under
MH. That parameter takes `LogInterval` from commit 5. The pass is a posterior
mean and standard deviation within 4 MCSE of the log-normal truncated to [0.05,
1], computed by quadrature.

### 12.3 Prior-only runs recover the truncated prior, under NUTS and under MH

These are likelihood-free runs, with 4 chains × 4,000 kept draws, of three
parameters:

- τ ~ exponential(5.99) on [10⁻⁶, 2.5] (`LogInterval`, the case of
  `nuts_validity.rs:366`);
- ρ ~ beta(2, 5) on [0.1, 0.9] (`LogInterval`);
- κ ~ half_normal(1) on [0, ∞) (`LowerBounded { 0 }`).

Each is run under PGAS+NUTS, extending `nuts_validity.rs:532`
(`pgas_nuts_without_a_likelihood_draws_the_prior`), and under PMMH and MH. The
NUTS run alone does not exercise the MH acceptance ratio, which uses the
Jacobian separately.

Pass:

- each mean and standard deviation within 4 MCSE of the truncated law's;
- a Kolmogorov–Smirnov test on draws thinned to the bulk ESS with p > 0.001;
- under NUTS, divergences after adaptation no more than the unboxed exponential
  control's bound (`nuts_validity.rs:323`);
- under MH, acceptance between 0.1 and 0.6.

### 12.4 A hierarchical prior under PMMH is unchanged

`sim/tests/pmmh_hierarchical.rs` and `pgas_gate_hierarchical.rs` move to the new
variants unchanged in meaning. Two pass conditions apply:

- the refusal of `9ccb2899` still fires for a leaf boxed to [1, 2] under a
  normal hyperprior;
- a hierarchical model whose leaves are unboxed `real`s (`Identity`) recovers
  its known posterior under PMMH. Its hyperparameter's posterior mean and
  standard deviation must be within 4 MCSE of the analytic normal–normal answer.

### 12.5 A boxed posterior is unchanged under PGAS+NUTS

The run is `tests/recovery/cases/seir_age` with its `pgas.toml` (three rates
boxed to [0.001, 0.5] and [0.01, 1.0], with log-normal priors), on synthetic
data from its `truth.toml`, at main and after commit 6, with 4 chains each.

Pass:

- for each parameter, the posterior mean and the 5%, 50% and 95% quantiles agree
  within 4·√(MCSE_before² + MCSE_after²);
- R̂ ≤ 1.01 in both runs;
- divergences after ≤ divergences before.

The target is the same truncated posterior in both runs (§4.2), so a
disagreement means either a defect or a sampling failure of the walled run. If
one appears, a run four times longer at main decides which.

### 12.6 Benchmark: IF2's default step per map family

IF2's default step decides how far the swarm moves in the early iterations and
therefore whether a short or a wide-box fit finds the maximum. PMMH and MH adapt
their proposal, so their default is set once (`default_mh_step`) and is not
benchmarked. The candidates for IF2's step in z are 2/3, 1/3, 1/5, 1/10, 1/20
and 1/50, run for each family on the fixtures that exercise it:

| fixture                                                                                          | family exercised                   |
| ------------------------------------------------------------------------------------------------ | ---------------------------------- |
| `sir` with `gamma : rate in [0.0, 0.5]`                                                          | `Interval` on a positive kind      |
| `seir_age`, its `if2.toml`                                                                       | `LogInterval`                      |
| `polio_afp_es`, its `fit.toml`                                                                   | `LogInterval`, `Interval` (`rho`)  |
| the gh912 short-budget cell on `polio_afp_es` (2 chains × 3 iterations, 50 particles, seeds 1–8) | the same, at a short budget        |
| `sir` with `lb : real in [−8, 2]` entering as `exp(lb)`                                          | `Interval` on a wide log-scale box |
| the Ebola-like recovery fixture closest to `spline_seir`                                         | the families its parameters take   |
| `sir` with `gamma : rate` undeclared                                                             | `LowerBounded { 0 }`               |

Each fixture also runs at main, and the positive-kind fixtures run with
`Interval` in place of `LogInterval`, so D11's geometry choice is measured
beside the constant.

Reported for each fixture and candidate: IF2's final log-likelihood (the clean
particle-filter evaluation, best and mean over chains, with the particle-filter
standard deviation), the number of chains whose final estimate is −∞, and
iterations until within one log-likelihood unit of the final value.

A constant serves a fixture when IF2's best final log-likelihood is within twice
the particle-filter standard deviation of the best over all candidates (and, on
the gh912 cell, when no more seeds end with every chain at −∞ than at main). The
family's constant is the largest that serves every fixture of the family; the
largest is preferred because a larger step explores faster under the same
cooling. If no constant serves every fixture of a family, none is chosen by
rule: the table returns to the maintainer, and the fallback is the launch-table
notice of §9.1. A maximum on a bound is a legitimate result, and where the swarm
sits relative to the ends is not a criterion. The measurements go in a
`docs/dev/notes/` note, which the series (i) merge cites.

## 13. What changes for users

- **Run identity.** Every fit re-keys, because the fit level folds the engine
  version, `VERSION_SHORT`, which includes the git hash
  (`cli/src/fit/cas.rs:501-521`, `cli/src/version.rs:12-15`; the resume guard at
  `cas.rs:603` folds it too). No schema bump and no IR or golden change is
  needed.
- **Configs that set `transform`** are refused at load, naming the key. The
  migration is to delete the line. The count is 167 `log` and 74 `logit`
  overrides in about 100 files (verified,
  `grep -rh --include='*.toml' -oE 'transform\s*=\s*"[a-z]+"'` over this
  repository and `camdl-book`, `camdl-vignettes`, `camdl-garki`,
  `camdl-india-polio`, `camdl-nigeria-polio`, `ebola-camdl`, `ebola-bdbv-camdl`
  and `camdl-paper`). After the interim fix every override that still loads is
  redundant, so deleting it changes nothing.
- **A model parameter without a kind** is refused (D6). The compiler always
  emits one, so only hand-built IR is affected.
- **Posteriors** are the same truncated posteriors. Sampling near a bound
  changes: fewer NUTS divergences, different PMMH and MH acceptance. The PGAS
  MH-within-Gibbs initial scale changes for every parameter (§9.2).
- **A start exactly on a bound** is refused, naming an interior value to start
  from; a drawn start on a bound is redrawn (D3).
- **IF2 and MH steps** are fixed in z (commit 7), with IF2's default chosen per
  map family by §12.6 and PMMH's and MH's set separately (D8). Near an end the
  step no longer grows, a positive kind on [0, hi] moves at all for the first
  time (§3.3), and a `rate` on [0, ∞) no longer needs an explicit `rw_sd`. With
  an explicit `rw_sd`, series (i) keeps the natural-scale unit and converts it
  once at the midpoint. Series (ii) reads it in z:
  - a value tuned on the natural scale converts as rw_sd_z =
    rw_sd_natural/|g′(0)|;
  - for `LogInterval` that is rw_sd_natural/(θ_mid·L/4), where θ_mid = √(lo·hi);
  - for `Interval` it is rw_sd_natural/(w/4);
  - for a rate on [0.05, 1] tuned at `rw_sd = 0.02`, the new value is
    0.02/(0.224 × 3.0/4) ≈ 0.12.
- **Priors**: `beta` on a narrowed `probability`, and the positive families on a
  positive kind's two-sided box, are accepted under the interim fix's consistent
  handling and evaluated exactly here. Each is truncated to the box. A prior
  whose support is narrower than the domain is refused.
- **Diagnostics and records**:
  - the near-bound warning is for boxes finite at both ends, and names
    `nl-sbplx` (ODE backend only) or a profile;
  - the launch table names the map and the z-step;
  - the partition record's `transform` tag takes the new values and loses
    `transform_from`;

## 14. Decisions

D7 is not used: refusing a fit box outside the kind's range lands with the
interim fix (§3.4).

**D1 — Does the transform carry the domain, with `EstimatedParam`'s bounds
derived from it? Decided (2026-10-29): yes.** Today `lower`, `upper` and
`transform` are independent, and a box can be recorded that nothing enforces
(§4.1). Option (a), one enum that is the domain, makes that state
unrepresentable. It touches the 102 non-test lines and 279 test lines of §3.2,
mechanically. Option (b), a separate `Domain` type from which a `Transform` is
derived, has two isomorphic enums and the same lines to touch. Option (c),
keeping the fields, keeps the drift. No user-facing behaviour differs among
them.

**D2 — Delete `[estimate].transform`? Decided (2026-10-29): delete.** After the
interim fix every accepted override is redundant (§7). Option (a), deletion,
removes `TransformFrom`, `check_transform_keeps_box` and the string threading.
Every config that sets the key fails at load until the line is removed, about
100 files across the sibling repositories (§13). Option (b), keeping the field
and accepting only the redundant value, keeps a knob with no effect. Under the
alpha posture (a) is the form.

**D3 — A start exactly on a bound: move it inward, or refuse? Decided
(2026-10-29): refuse a given start; redraw a drawn one.** A start on a bound has
no finite z. Moving it inward puts the chain at z ≈ ±23, far out in a tail: a
natural-scale step converted there is ~10¹⁰ in z, and even a fixed step in z
leaves the chain walking back through the tail for longer than burn-in (PMMH on
`sir`, `gamma` started on 0.26 of [0.01, 0.26]: bulk ESS 14–17 against 444 at
main). A given start on a bound is therefore refused, with the interior value to
use and, where the optimum on the bound is the point, `nl-sbplx` (ODE backend)
or a profile, which can place an estimate on a bound. Starts a fit draws are
redrawn when they land within rounding of a bound. The cost is that chaining an
optimiser's on-bound optimum into a sampler through `--starts` needs the user to
move it inward by hand, which the error spells out.

**D4 — Do priors read logarithms of θ computed from z, with the compatibility
table replaced by a support rule? Decided (2026-10-29): yes.** This is required
for correctness. Without it, `log_normal` under any map but θ = eᶻ is a
different prior, as the reproduction of §3.3 shows. With it, the prior API
changes at the eight non-test call sites of §3.2. The guards on θ in the priors
go. `log_normal` under `LowerBounded { 0 }` stays bit-identical on z ≥ −745 and
becomes correct below. The alternative, keeping `TransformReq` and forcing a log
map for log-family priors, would reinstate the walled log for every boxed rate
with a `log_normal` prior.

D4's support rule is a language choice: a prior narrower than the box gives a
proper truncated target, and refusing it catches a mistaken box or prior rather
than a mathematical impossibility (§6).

**D5 — NLopt coordinates. Decided (2026-10-29): keep the natural scale here;
measure z-coordinates under gh#1002.** §9.3 sets out both options. Moving now
would change the optimiser's conditioning, its stopping rule and its behaviour
at bounds, with no measurement in the repository to show it helps;
`docs/dev/agent-verification-conventions.md` asks for the measurement first.
Keeping B means the optimisers and samplers use different coordinates, which
changes no reported estimand. The trade-off is that, near a bound, an NLopt fit
can sit on the bound while IF2 stops just inside it; the near-bound message
(D10) names `nl-sbplx` or a profile as the test.

**D6 — A parameter whose IR carries no kind. Decided (2026-10-29): refuse.** The
compiler always emits a kind (`ast.ml:150-151` makes it mandatory;
`expander.ml:5806`, `:5866`), so only hand-built IR and IR from before the field
existed reach this case. Today the runner searches it over [0, ∞)
(`runner.rs:1163`) while `roles.rs` records its support as (−∞, ∞) (`:1029`), so
the record contradicts the search. Option (a), refusing at resolution with the
parameter named, removes the case and `BoxFrom::Fallback`. The test IR that
builds kind-less parameters gains a kind (commit 3). Option (b), (−∞, ∞), makes
a guess in the other direction. Making `param_kind` non-optional in the IR is
the complete fix and is an `ir/VERSION` change (§15).

**D8 — The default IF2 and MH step. Decided (2026-10-29): a fixed step in z;
IF2's default chosen per map family by the §12.6 benchmark; PMMH's and MH's set
separately.** The step stops moving with the parameter, which removes the
end-of-box blow-up of the per-particle conversion (b, below) and the
start-dependence of a conversion at θ₀ (c). The alternatives each fail §9.1's
measurement:

- (b), today's per-particle conversion, throws particles at the ends (25.3%);
- (c), conversion once at θ₀, gives each chain a different step, up to 3·10⁸ at
  a start on an end.

IF2's default is one constant for each family: interval, log-interval, and the
half-bounded maps. Reproducing today's midpoint steps (2/3 in z for the logit's
(hi − lo)/6, 1/5 for the log's ln(hi/lo)/20) is a starting point, not the
answer: on a wide `real` box over a log-scale quantity, `lb in [−8, 2]` entering
as `exp(lb)`, every step from 1/5 to 2/3 left IF2 1.5–38 nats short of what a
step of 0.02 reaches. §12.6 runs the candidates on narrow and wide boxes and
short and long budgets together. If one constant per family serves every
fixture, it is the default; if not, the maintainer chooses, and the launch table
prints the natural-scale step each default implies. The half-bounded maps get a
default too, so a `rate` on [0, ∞) runs without an explicit `rw_sd`.

PMMH's and MH's default is a separate constant, `default_mh_step`, because their
proposal is adapted from the first warm-up step; its starting value matters
little and is not benchmarked.

**D9 — The unit of an explicit `rw_sd` (gh#1001). Decided (2026-10-29): z, in
series (ii); series (i) converts a natural-scale value once at the domain's
midpoint.** Option (a) matches pomp's `mif2`. It makes the user's number the
kernel IF2 runs, and lets `transformed_sd` and its false comment go. Every
explicit `rw_sd` changes meaning; §13 gives the conversion. Option (b), the
natural scale converted once at the midpoint for good, keeps hand-tuned values
meaning roughly what they meant, under a unit the user cannot read off the
coordinate. Option (c), today's per-particle conversion, is refused by §9.1.
Default runs are identical under (a) and (b), since they use `default_if2_step`.

**D10 — The near-bound report. Decided (2026-10-29): boxes finite at both ends
only, at a fixed fraction (1%) of the width from either end; no report on a
half-infinite domain.** A ratio to the start (θ̂ < 0.01·θ₀) was considered for
[0, ∞) and dropped: it fires on a legitimately small estimate from a large start
and is silent for a start on the bound. Telling a maximum on the bound from a
small interior one needs the likelihood near the bound, which is its own design
(§15). The message says the estimate may be on the bound and names the test:
profile the parameter, or refit with `nl-sbplx` (ODE backend only).

**D11 — The map for a positive kind (`rate`, `positive`, `count`, `duration`)
boxed with 0 < lo. Decided (2026-10-29): `LogInterval`, for positive kinds only,
as a geometry choice, measured beside `Interval` in §12.6.** The support does
not decide it: both maps keep [lo, hi]. `LogInterval` interpolates geometrically
between the ends, `Interval` arithmetically. These boxes are the common case:
every rate in the recovery fixtures, and the B-spline scale τ on [10⁻⁶, 2.5].
Option (a), `LogInterval`, keeps today's ln θ geometry in the box's interior, so
default steps and log-spaced starts carry over.

The trade-off is that the transform is no longer a function of the domain alone:
the kind splits a box with 0 < lo. Choosing by lo > 0 alone would keep that
principle at two costs. A `probability` on [0.1, 0.9] would move from the logit
to `LogInterval`, and its default natural-scale step would shrink 4.05× (0.133
to 0.033), against D8's aim of keeping today's step. An `instant` or `real` on
[10, 50] would get log geometry while one on [−10, 50] would not, so the
geometry would depend on where the time origin sits. Log geometry is a property
of positive quantities, which is what the positive kinds declare. ln θ is an
exact function of z, so log-family priors are exact. Its default step means the
same thing at every scale of box. It is a composite, not one of Stan's built-in
maps. Option (b), `Interval` everywhere, is Stan's map and one variant fewer.
But it spaces a rate on [10⁻⁶, 2.5] linearly, so almost all of z's interior lies
above 10⁻² and the sampler resolves the low end poorly. It also needs the kind
threaded into `EstimatedParam` for log-spaced starts (§4.1); under (a) the kind
is read once, at construction, and the map carries the answer. The benchmark
runs both. If (b) matches (a) on every criterion, (b) wins on simplicity.

## 15. Out of scope, each with its reason

- **The IR's per-parameter `transform` field** is always `Identity` and read by
  nothing (§3.1). Removing it is an `ir/VERSION` change with a golden update,
  batched with the next bump. With D6, `param_kind` can become non-optional in
  the same bump.
- **The boxed hierarchical leaf's normalising constant**, −ln Z(h), which would
  lift the refusal of `9ccb2899`. It is a density term, not a map
  (parameter-roles §13).
- **NLopt in z-coordinates**, gh#1002 (D5).
- **A boundary-optimum diagnostic for half-infinite domains** (D10). Whether
  IF2's small estimate on [0, ∞) is a maximum at 0 needs the likelihood near the
  bound (a profile or a likelihood-ratio check), which is its own design.
- **Rounding at use for `count` initial states** (§8). A likelihood that is a
  step function of a parameter is a modelling property no map changes.
