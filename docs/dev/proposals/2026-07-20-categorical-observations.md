# Ordinal observations for parasite-density ladders

Status: Draft. Scope: score ordered categorical (parasite-density-class) counts
via the continuation-ratio binomial factorization on camdl's **existing** scalar
likelihoods; cumulative proportional-odds as the default link, non-proportional
continuation-ratio as the relaxation. A native vector `multinomial` likelihood
and an `ordered_logistic` sugar are scoped follow-ups, not v1. Target: Garki
`ctl_prev_density_ladder`.

## Problem

Garki records parasite density in ordered monograph classes, derived from a
continuous count (`pfa`/`exam`). The current `ctl_prev_density_ladder` collapses
this to one binary cut (`HI_CUT`, class ≥ 4) scored as a `beta_binomial` on the
high-density fraction — one bit of a multi-class gradient. We want the **full
ordered class distribution**, which carries the intensive-margin immunity signal
(density falls fast with immunity rung, before prevalence does).

## Two things not to conflate: the _scoring substrate_ and the _link_

"Continuation-ratio" names two separate things, and separating them is the whole
design:

- **Scoring substrate (an algebraic identity).** For _any_ class-probability
  vector `π` and aggregate counts `x` with `n = Σ x_k`:
  ```
  Multinomial(x ; n, π) = ∏_{k=1}^{K-1} Binomial(x_k ; n_k, r_k),
      n_k = Σ_{j≥k} x_j          (tail count — θ-independent → a data column)
      r_k = π_k / Σ_{j≥k} π_j     (a SCALAR ratio of the π)
  ```
  Proof: sequential conditioning of the multinomial (chain rule) —
  `x_1|n ~
  Binomial(n, π_1)`; the remainder given `n − x_1` is a renormalized
  multinomial, so `x_2|(n−x_1) ~ Binomial(n_2, r_2)`; recurse. The identity is
  elementary and model-free (no citation needed): it holds **regardless of how
  `π` was built**, so it is an _implementation_ layer, not a modelling
  commitment. Every `r_k` is a scalar, so it maps directly onto camdl's existing
  scalar `beta_binomial`.

- **Link (the science).** How the immunity latent enters `π` — cumulative logit,
  continuation-ratio logit, etc. This is a real modelling choice, made _above_
  the scoring substrate.

## Scientific context: the ordinal families and the tradeoffs

K ordered classes; latent/covariate `x` (for us the immunity-driven shift);
`π_k = P(Y=k|x)`, cumulative `F_k = P(Y≤k)`. The families (Agresti 2013, ch. 8):

- **Cumulative / proportional-odds** (McCullagh 1980; Stan `ordered_logistic`):
  `logit F_k = κ_k − βx`, ordered cutpoints `κ_1<…<κ_{K-1}`. Latent-variable
  form: `z = βx + ε`, `ε∼Logistic`, `Y=k` iff `κ_{k-1}<z≤κ_k` — a continuous
  latent binned by fixed thresholds. One slope, K−1 ordered intercepts.
- **Continuation-ratio / sequential** (Fahrmeir & Tutz):
  `logit P(Y=k|Y≥k) = α_k − β_k x`.
- **Adjacent-category**: `log(π_{k+1}/π_k) = α_k − βx`.
- **Baseline-category** (unordered; Stan `multinomial_logit`):
  `log(π_k/π_K) = α_k − β_k x`.

Four tradeoffs decide the choice for density:

1. **Non-proportional safety (the sharpest).** If the immunity effect differs by
   class (`β_k` free — e.g. premunition compresses the _high_ tail more than the
   low): the cumulative model's lines `logit F_k = κ_k − β_k x` can **cross**,
   giving `F_k > F_{k+1} ⇒ π_{k+1} < 0` — invalid (Peterson & Harrell 1990).
   Continuation- ratio and baseline-category with free `β_k` are **always
   valid** (each conditional logit is independently in `[0,1]`). So
   category-specific immunity slopes are safe only in the continuation-ratio
   link.
2. **Invariance.** Cumulative is invariant to reversing category order and to
   **collapsing adjacent classes** (proportional-odds preserved under grouping;
   Agresti §8.2) — conclusions don't hinge on binning granularity.
   Continuation-ratio is directional (forward ≠ backward) and binning-sensitive;
   for a _directed_ continuum (density low→high) the direction is meaningful but
   the binning sensitivity is a real cost.
3. **Mechanistic honesty.** Parasite density _is_ a continuous quantity binned
   by microscopy, so the cumulative (latent-threshold) model is the
   data-generating- honest one: `κ_k` are the class boundaries on the
   log-density scale, `β` is how immunity shifts log-density. Continuation-ratio
   is a staged stop/continue process — honest for genuine progressions, a
   reparameterization for a binned continuum.
4. **Factorization.** With free `β_k`, the continuation-ratio _link_ fits as K−1
   independent binomial GLMs (Agresti §8.3); the others need the joint
   multinomial for a _native_ fit but can still be **scored** via the
   (link-agnostic) identity above. Note the _links_ are genuinely distinct model
   families under the logit link we use — cumulative and continuation-ratio
   coincide only under the complementary-log-log link (Läärä & Matthews 1985,
   _Biometrika_ 72:206–207), which is exactly why the link choice above is a
   real modelling commitment, not just a reparameterization.

## The Garki wrinkle: the cell is a rung-mixture

A `(village, age)` cell's detected-positives are a **mixture over immunity
rungs** (the model stratifies compartments by `imm`), so the cell
class-probabilities are

```
p̄[v,a,k] = Σ_{m∈imm} w[v,a,m] · P(class = k | rung m),   w[v,a,m] = detpos[v,a,m] / detpos_tot[v,a]
```

A finite mixture of ordinal distributions is **not** itself single-latent
ordinal: for K≥3 the mixture ranges over the (K−1)-simplex while a single-`η`
curve is 1-dimensional and cannot cover it (dimension count). So no scalar
`η_cell` reproduces the cell, and no monolithic `ordered_logistic(η_cell, κ)`
fits — you must build `p̄` explicitly (per-rung link, then mix), then score. The
mixture induces **no extra dispersion**: `n` iid draws from the cell-averaged
`p̄` are exactly `Multinomial(n, p̄)` (the mixing is absorbed into `p̄`);
`beta_binomial` overdispersion is a separate _empirical_ choice (matching the
current `phi_hd`), not a mixture fix. Note the exact factorization holds only in
the binomial limit (`concentration→∞`, which the equivalence test pins); with
finite `phi_hd` the model is a **product of beta-binomials** — a legitimate
sequential-overdispersion model, but _not_ the Dirichlet-multinomial (the
canonical overdispersed multinomial). A single shared `phi_hd` across the K−1
steps is a parsimony choice matching today's model; per-step `phi_k` is more
flexible and cheap — worth a v1 sensitivity check.

## Design decision

- **Scoring substrate: the continuation-ratio binomial factorization.** `p̄`
  (however built) is scored as K−1 scalar `beta_binomial` streams on the tail
  counts. This reuses existing likelihoods and — because `beta_binomial` is
  `Diffable` and the `r_k` are scalar `DerivedExpr` projections — **gradients
  work today** (a NUTS-on-ODE refit needs no new machinery). Verified by
  compiling the probe to IR: the two load-bearing pieces both emit cleanly — the
  factor-2 state chain `projection_state_grad` (gh#275 §1h, ∂r_k/∂compartment;
  96 `Grad` entries, zero `Unsupported`, so the ODE-NUTS gate passes) and the
  factor-1 param gradients for `hd_b`/`κ`/`phi` (the OCaml autodiff inlines the
  projection into the `beta_binomial` `alpha`/`beta`). `beta_binomial` also
  already has both dmeasure _and_ rmeasure, so `simulate --obs` / `fit predict`
  / SBC work.
- **Default link: cumulative proportional-odds** — the mechanistically-honest
  model for a binned continuum, and exactly Stan's `ordered_logistic`. One
  immunity slope, ordered cutpoints. K=2 recovers today's high-density
  _probability_ (as a binomial; the `phi_hd` overdispersion returns via
  `beta_binomial`).
- **Relaxation: non-proportional continuation-ratio** — free per-cut immunity
  slopes, for premunition that caps the high tail differentially. The _same_
  binomial scoring supports it; no cutpoint/ordering machinery at all.
- **Reserved (not v1): a native vector `multinomial` likelihood** for genuinely
  _unordered_ categorical outcomes, where there is no natural continuation-ratio
  to factor through.

## Engineering: what v1 needs (and, notably, does not)

For Garki's K (the monograph classes, K≈4–6) the model is **expressible today
with existing primitives** — no new IR, schema, grammar, or `ir/VERSION` bump.
The primitives: `beta_binomial(n, mean, concentration)` (scalar, `Diffable`);
scalar and indexed `let`s (`let x[m in imm] = …`, already used by this model for
`q_hidens`); `sum(m in imm, …)` for the mixture; `exp` + arithmetic for
`σ(x)=1/(1+exp(−x))`; scalar parameters. The cost is **verbosity** — K per-class
probability `let`s, K mixture sums, K−1 continuation ratios, K−1 streams — not
new language surface. The `ordered_logistic` sugar (Follow-up A) exists to
remove that verbosity later.

Concrete v1 model (cumulative-PO, K=4 classes `c1_2,c3,c4,c5plus` ⇒ 3 cutpoints)
— **every construct below parses under the current grammar**:

```
parameters {
  # cutpoints on the latent log-density scale; ordered BY CONSTRUCTION (base +
  # positive spacings) so no `ordered[]` param kind is needed for v1.
  kap1       : real
  kstep2     : positive
  kstep3     : positive
  hd_b       : real          # < 0: density falls with immunity rung
  phi_hd_inv : positive
  # …plus the existing g, r1, a1, delta, nu, Cscale
}

# IDENTIFIABILITY: the latent's location is absorbed into the cutpoints — a free
# intercept `hd_a` would be confounded with the overall κ level (adding c to hd_a
# and to every κ_k leaves σ(η−κ) unchanged). So there is NO hd_a; the cutpoints
# carry the absolute location. (This is also the honest K=2→K generalization:
# today's `hd_a` IS the single K=2 cutpoint.) The `m0` rung is the reference
# (eta[m0]=0), and κ_k are the class boundaries relative to it.
let kap2   = kap1 + kstep2
let kap3   = kap2 + kstep3
let phi_hd = 1.0 / phi_hd_inv
let eta[m in imm] = hd_b * rung_idx[m]                # latent propensity per rung

# per-rung class probabilities via cumulative logit (σ hand-written)
let p1[m in imm] = 1.0 - 1.0/(1.0 + exp(-(eta[m] - kap1)))
let p2[m in imm] = 1.0/(1.0 + exp(-(eta[m] - kap1))) - 1.0/(1.0 + exp(-(eta[m] - kap2)))
let p3[m in imm] = 1.0/(1.0 + exp(-(eta[m] - kap2))) - 1.0/(1.0 + exp(-(eta[m] - kap3)))
let p4[m in imm] = 1.0/(1.0 + exp(-(eta[m] - kap3)))

let w[v in village, a in age, m in imm] = detpos[v,a,m] / detpos_tot[v,a]

# cell (mixture) class probabilities
let pbar1[v in village, a in age] = sum(m in imm, w[v,a,m] * p1[m])
let pbar2[v in village, a in age] = sum(m in imm, w[v,a,m] * p2[m])
let pbar3[v in village, a in age] = sum(m in imm, w[v,a,m] * p3[m])
let pbar4[v in village, a in age] = sum(m in imm, w[v,a,m] * p4[m])

# continuation ratios r_k = pbar_k / Σ_{j≥k} pbar_j (scalar projections)
let r1[v in village, a in age] = pbar1[v,a] / (pbar1[v,a] + pbar2[v,a] + pbar3[v,a] + pbar4[v,a])
let r2[v in village, a in age] = pbar2[v,a] / (pbar2[v,a] + pbar3[v,a] + pbar4[v,a])
let r3[v in village, a in age] = pbar3[v,a] / (pbar3[v,a] + pbar4[v,a])

observations {
  # one stream per continuation step; data columns are the tail counts (make-target).
  dens_step1[v in village, a in age] {
    columns   { time, village:dim, age:dim, x1:count, n1:count }
    projected = r1[v,a]
    x1 ~ beta_binomial(n = n1, mean = projected, concentration = phi_hd)
  }
  dens_step2[v in village, a in age] {
    columns   { time, village:dim, age:dim, x2:count, n2:count }
    projected = r2[v,a]
    x2 ~ beta_binomial(n = n2, mean = projected, concentration = phi_hd)
  }
  dens_step3[v in village, a in age] {
    columns   { time, village:dim, age:dim, x3:count, n3:count }
    projected = r3[v,a]
    x3 ~ beta_binomial(n = n3, mean = projected, concentration = phi_hd)
  }
}
```

Non-proportional relaxation: replace the cumulative `p_k[m]` block with free
per-cut continuation ratios `r_{m,k} = σ(a_k + b_k·rung_idx[m])` and build
`p_k[m]` from them (`p1[m]=r_{m,1}`, `p2[m]=r_{m,2}·(1−r_{m,1})`, …) — same
downstream mixture + streams; no cutpoints, no ordering.

**Identifiability — the one to watch.** The location/scale confound is handled
(cutpoints carry the location, `rung_idx` fixes the scale). The _remaining_
concern is a **ridge between `hd_b` and the immunity-ladder dynamics
`{g, r1}`**: `hd_b` is the density shift _within_ a rung, while `{g, r1}` set
the rung _mix_ `w` through the ODE — both push cell density down with age, so a
faster rung climb with smaller `|hd_b|` mimics a slower climb with larger
`|hd_b|`. Two facts sharpen it: `hd_b` vanishes at `m0`-dominated cells
(`eta[m0]=0`), so it is informed _only_ by cells with rung spread (older ages /
later times), and the mix is dynamics-driven (the prevalence stream + fixed
anchored ladder `{q_det, r_mult}` constrain it, mitigating but not removing the
`hd_b`↔`{g,r1}` posterior correlation). This is consistent with the known garki
`mh/ode` frozen-chains/multimodality behaviour. So the honest claim is
"`{κ, hd_b, phi_hd}` identified given age-stratified rung spread, with an
expected `hd_b`–climb correlation" — not "cleanly separated." A (near-)empty top
class also only weakly identifies the top cutpoint (small `n_3`); the usual
sparse-ordinal caveat.

## Follow-ups (scoped; not gating Garki)

**A — `ordered_logistic` / `ordinal` DSL sugar.** A frontend expansion
(precedent: `diagnostic_test`) that lowers one declaration into the K−1
continuation-ratio `beta_binomial` streams, removing the verbosity above.
Gradient-transparent (it emits ordinary differentiable streams). Cost the
reviewers mapped: it presupposes a category axis and either a category-indexed
construction or a fixed-K unroll; the single-latent form is a clean expansion,
the mixture form is inherently model- specific and only partly sugar-able. Ships
after v1 proves the pattern.

**B — native vector `multinomial` likelihood** (for _unordered_ categorical).
This is the large lift the architecture review mapped precisely; recording it so
a future pickup is accurate:

- `Likelihood::Multinomial { probs: Expr }` appended at **run-id index 8** (the
  hash is the hand-written positional match in `runid/src/ir_hash.rs`, decoupled
  from enum declaration order — a bare `probs` must be hashed _explicitly_
  there, as Binomial/BetaBinomial `n` is, or two models differing only in
  `probs` collide). Cross-language contract is the **name** `"multinomial"`, not
  declaration order. Bump `ir/VERSION` (now 0.30).
- Scoring-only, if it ships that way: bare `Expr` arguments, and a refusal in
  `gradient_capability.rs`. There is no longer a family to copy this from —
  `ZeroInflatedNegBinomial` was the last scoring-only likelihood and is now
  differentiable, so the `unreachable!()` grad arms it used are gone.
- **A vector obs cell + a category-aware loader.** `ObsCell` is
  `Scalar(f64)`-only (its doc comment already names `Vector` as the planned
  extension); ~12 match sites. The long-form loader currently rejects duplicate
  `(time, stratum)` and builds the per-leaf time axis one-row-per-cell — a
  category axis needs a _new_ grouping pass (parse/validate/order the K levels,
  assemble one `Vector` cell), not a branch.
- **A vector projection** and a new `multinomial_logpmf`; **an rmeasure
  sampler** (`sample_obs_resolved`) or `simulate --obs`/`fit predict`/SBC break
  on the stream.
- A `category` column role + `categories {}` block (additive; `category(<set>)`
  needs a new parser arm). dimcheck can enforce `probs` dimensionless but
  **not** sum-to-1 (a value constraint, not a dimensional one) — that is a
  runtime/normalization check.

## Data

A make target bins `garki-data/…/parademo.parquet` (`pfa`/`exam`; 41,199
positive rows; density index 0.003–1.635) into the K monograph classes and emits
the **tail-count** long-form the streams read: per `(time, village, age, step)`,
`x_step` and `n_step = Σ_{j≥step} x_j`. Offline, deterministic, committed with
provenance, separate from any network fetch.

## Sequencing

1. Data make-target (tail-count long-form + provenance).
2. Hand-write the cumulative-PO CR model into `ctl_prev_density_ladder` (or a
   `_dclass` variant); validate the fit recovers the cutpoints + immunity slope
   and that `g` moves toward the entomology anchor as the serology/binary
   streams did.
3. Residual check on the top class; if premunition caps the high tail, switch
   that block to the non-proportional relaxation (same streams).
4. Follow-up A (`ordered_logistic` sugar) once the pattern is proven; Follow-up
   B (native `multinomial`) only if unordered categorical is needed for its own
   sake.

## Tests

- Model compiles and forward-simulates; a synthetic-data fit under `mh+ode`
  recovers the cutpoints + immunity slope **with the dynamics params `{g, r1}`
  free** (not fixed at truth), and a profile of `hd_b` against `r1`/`g` shows
  the ridge is not a flat non-identifiability; a `nuts`-on-`ode` refit runs
  (gradients present via `beta_binomial`). _(Compile + gradient-structure
  already verified on the probe model; `projection_state_grad` all-`Grad`,
  ODE-NUTS gate passes.)_
- Data target reproduces byte-identically; `n_step1` per cell equals the cell's
  examined-positive total; `x_step` sums to the cell total.
- Equivalence check: the K−1 `beta_binomial(concentration→∞)` (binomial)
  log-likelihood equals `multinomial_lpmf(x | p̄)` on fixed vectors (pins the
  factorization).
