# Complete dimension inference: what it resolves, and what is left

Date: 2026-11-05

Proposal: `docs/dev/proposals/2026-11-05-size-brackets-and-dim-clause.md` (§4,
§5, Stage 1)

## Question

The proposal makes a `dim` clause required exactly when inference leaves a
parameter's dimension undetermined (I300 becomes an error, §5), and defers that
until inference is complete (§4). This note measures what the complete solver
resolves and classifies every I300 that remains, so the §5 decision can be made
on the evidence.

## What changed in the checker

Two commits, on main:

1. `367d1739` — every product, quotient, integer power and square root with an
   unresolved operand records a linear equation over exponent vectors; each
   transition rate, ODE derivative, balance expression and overdispersion term
   is pinned to its required dimension; the equations are solved together by
   Gauss–Jordan elimination over the rationals and iterated with the existing
   rounds to a fixpoint (Kennedy, "Dimension types", ESOP 1994). A variable is
   bound only to an integer exponent vector. Strict likelihood arguments
   (Poisson `rate`, Binomial `p`, dispersions, beta-binomial shapes) record
   equations; permissive ones (Normal `mean`/`sd`) do not.
2. `33f24bf6` — two positions with a defined dimension are now visited: forcing
   arguments (E312; definitions from `rust/crates/sim/src/propensity.rs`) and
   initial conditions (E313; literals there take the dimension their context
   needs).

## Measurements

Every tracked `.camdl` in the repo and every `.camdl` in `../ebola-bdbv-camdl`,
compiled with each binary; counts of I300 lines in stderr. "Compile" counts exit
code 0 (the repo set includes error fixtures, which are meant to fail).

| Checker                   | Repo: compile | Repo: models with I300 | Repo: I300 | Ebola: compile | Ebola: I300 |
| ------------------------- | ------------- | ---------------------- | ---------- | -------------- | ----------- |
| main (`6bcffa45`)         | 145 of 174    | 21                     | 44         | 8 of 111       | 30          |
| + linear solve            | 145           | 15                     | 35         | 8              | 8           |
| + forcing and init checks | 145           | 10                     | 25         | 8              | 3           |

Exit codes and IR bytes were compared model by model against main. The linear
solve changed neither for any model. The forcing check rejected exactly one,
`tests/fixtures/gradient/bspline_forcing_grad.camdl`, whose `'per_day`
identity-link splines took `'ratio` coefficients also used by log-link splines —
a genuine inconsistency, fixed in the same commit. The solver reported no
inconsistent system and no fractional solution in any of the 164 models that
reach dimcheck.

103 of the 111 Ebola models stop before dimcheck (E350 on current main), so the
Ebola column covers 8 models.

Examples the old checker could not resolve and the new one does, each checked by
hand against the model:

- `licm_ab.camdl`: `theta` appears only in `exp(theta * log(dens[l]))`, so
  `theta · 1 = 1` and `theta` is dimensionless. `kappa` multiplies a ratio whose
  unknown table factors `N0` cancel and is added to `I / N`, so it is
  dimensionless.
- Ebola province models: `phi_split[p]` is a beta-binomial concentration, which
  the sugar lowers to `alpha = mean · phi`; with `alpha` dimensionless and
  `mean` a proportion, `phi` is dimensionless.
- `init_laws.camdl`: `E = exposed_per_infectious * I` makes the factor
  dimensionless, and `E ~ neg_binomial(..., r = exposed_k)` makes `exposed_k`
  dimensionless.
- `lead_days` in the Ebola province models reaches only an initial condition,
  through `I0 = I_obs / exp(r_growth * lead_days)`: `exp` makes the product
  dimensionless and `r_growth` is T⁻¹, so `lead_days` is T.

## Observations: the 28 that remain

**Free because unused** (declared, set by a scenario or not at all, never in a
checked expression): `sir_two_patch` `N[·]`, `R0[·]` (4); `quantities_showcase`
`N_a`, `N_b`; `data_spec/ituri` `k`.

**Free by structure** (the model pins only a combination): `prior_only_all` and
`prior_only_all_unboxed`, where `tau · (z1 + z2 + z3 + z4)` sits inside `exp`,
so only `tau + z = 0` is known (10); `seir_spatial_5_inference` `kappa`, which
multiplies an all-literal table `W`, so only `kappa · W = T⁻¹` is known.

**Exempted by the user**: `phenom_mixing_unchecked` `alpha`, inside
`unchecked_dim(...)`.

**In a position with no dimensional rule yet**:

- Prior hyperparameters — `garki_post_proposal` `sigma_alpha`, `sigma_r1`
  (`log_normal(..., sigma = σ)` on a rate) and `alpha_psymp`, `beta_psymp`
  (`beta(alpha = a, beta = b)`). The prior's arguments are not dimension-checked
  at all.
- Normal-likelihood arguments — `sd_iso` (3 Ebola national models,
  `normal(mean = projected, sd = sd_iso)`), `surveillance_likelihoods`
  `sigma_rel`, `known_profiles/k1` `mu`. Normal `mean`/`sd` are permissive by
  design (the He et al. 2010 variance formula is not homogeneous), and the
  observed column's dimension is not declared, so nothing fixes them.

## Interpretation

The solver itself is no longer the reason any parameter is undetermined: every
remaining case is either free in the model or sits in a position the checker
gives no rule. Turning I300 into an error now would therefore demand `dim` on
(a) unused parameters, whose dimension matters to nothing; (b) structurally free
pairs, where a `dim` clause is the right ask; (c) prior hyperparameters, whose
dimension the prior's family determines and the checker does not yet read; (d)
Normal arguments, where the data's dimension is genuinely unstated.

## Next

The §5 decision, with these classes in view: whether unused parameters are
exempt (a lint, not a dimension error), whether prior hyperparameters get
per-family rules first, and whether Normal `sd` is tied to `mean`.
