# camdl and the alternatives

Where camdl sits among the tools that solve nearby problems, when you should
reach for one of them instead, and — the question that comes up most — whether a
model you have already written in camdl could be written and fitted somewhere
else.

This is the honest version, not the marketing version. camdl is alpha software
with one implementation and a small user base; several of the tools below are a
decade older, more general, and better documented. What camdl offers is a narrow
bet: that for *stochastic compartmental models*, a compiled DSL with real
dimensional types plus a purpose-built inference stack beats assembling the same
thing out of general-purpose parts.

Related reading: [`user-features.md`](user-features.md) has a line-by-line
pomp-vs-camdl comparison of the same model; [`runtimes.md`](runtimes.md) covers
the three simulation backends; [`inference.md`](inference.md) explains what
statistical object each fitting method targets.

---

## The two questions that decide everything

**1. Does the discreteness and stochasticity of the latent process matter for
your inference?**

A compartmental model's latent state is a vector of integer counts evolving by
random transitions. If your populations are large, your epidemic is well
established, and you are content to fit the mean-field skeleton with observation
noise on top, then the latent process is effectively deterministic and *almost
every general-purpose Bayesian tool can fit your model*. If small counts,
fadeout and re-introduction, or extra-demographic noise on transmission are
load-bearing — if the difference between "the epidemic went extinct" and "the
epidemic went to 0.3 infected" changes your answer — then you need the
stochastic-process likelihood, and the field of tools narrows sharply.

**2. Do you need a compiler, or is a hand-written model fine?**

A DSL buys you dimensional checking (`beta * S * I` without the `/ N` is a
compile error, not a wrong answer), stratification expansion, automatic
parameter transforms derived from types, symbolic gradients, and run provenance.
It costs you generality: you can only say what the language can say. If your
model is one SEIR fitted once, a hand-written likelihood in a mature ecosystem
is a perfectly good answer and you get the whole R or Python toolchain with it.
The DSL pays off when the model is stratified, refitted, compared across
variants, and maintained by more than one person (or by an AI agent, which is
very good at writing plausible-looking wrong rate expressions and rather bad at
noticing).

---

## The landscape

| Tool                          | How the model is specified        | Latent process                             | Inference                                                    | Where it beats camdl                                             |
| ----------------------------- | --------------------------------- | ------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| **camdl**                     | DSL, compiled (OCaml → IR → Rust) | discrete-state stochastic, or ODE          | IF2, PGAS+NUTS, PMMH (chain-binomial); MH, NUTS, NLopt (ODE) | —                                                                |
| **Stan**                      | DSL, compiled (Stan → C++)        | continuous, sampled jointly with θ         | NUTS/HMC, ADVI, optimize                                     | generality, hierarchies, ecosystem, maturity                     |
| **pomp** (R)                  | R + C snippets                    | any simulator (plug-and-play)              | IF2, PMMH, PF, ABC, nlf, spectrum                            | maturity, arbitrary non-compartmental processes, literature      |
| **odin / dust / monty** (R)   | DSL, compiled to C++              | discrete-time stochastic, or ODE           | particle filter, PMCMC, SMC²                                 | proven at national scale, GPU/multicore runtime, R integration   |
| **LibBi**                     | DSL, compiled (C++/CUDA)          | state-space models generally               | SMC, PMCMC, SMC², bridge PF                                  | GPU SMC; largely dormant now                                     |
| **NIMBLE** (R)                | BUGS dialect, compiled to C++     | anything expressible in BUGS               | MCMC (customizable), PF, PMCMC                               | generality, custom sampler assignment                            |
| **Turing.jl** (Julia)         | Julia program                     | anything, incl. discrete latents           | HMC/NUTS, particle Gibbs, **PGAS**, SMC                      | the only general PPL with camdl's PGAS natively; full Julia      |
| **Catalyst.jl** (Julia)       | reaction-network DSL              | ODE / SDE / jump (Gillespie)               | via Turing / DiffEqBayes                                     | chemistry-general, one model → many backends, SciML stack        |
| **PyMC / NumPyro**            | Python program                    | continuous (small discrete by enumeration) | NUTS, SVI                                                    | Python ecosystem, ODE via sunode/diffrax                         |
| **EpiNow2 / epidemia** (R)    | fixed semi-mechanistic structure  | renewal equation                           | Stan NUTS                                                    | ready-made Rt estimation, no model to write                      |
| **Starsim / EMOD / EpiModel** | agent-based frameworks            | individuals, networks                      | calibration, not likelihood-based                            | individual heterogeneity, contact networks, individual histories |

---

## Stan

Stan is the closest match in **philosophy** and the furthest in **capability
overlap**, which is a confusing combination. camdl's README says it is inspired
by Stan, and the debt is real: declare the model, let the compiler handle
constraint transforms, gradients, and diagnostics; a fixed vocabulary of
distributions; convergence diagnostics you can't skip. The parameter-type system
(`rate` → log transform, `probability` → logit transform) is Stan's
`<lower=,upper=>` idea with the dimension carried along too.

But the object each language describes is different. **Stan describes a
differentiable log-density over continuous parameters.** camdl describes a
stochastic process plus an observation model, and the log-density — where one
exists at all — is derived.

|              | Stan                                          | camdl                                                                                                                                     |
| ------------ | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| You write    | `target += ...` — a log-density               | compartments, `transitions { }` with `@` rate, `observations { }`                                                                         |
| Latent state | continuous parameters, sampled jointly with θ | integer compartment counts: integrated out (PF, PMMH), Gibbs-sampled (PGAS), or collapsed (ODE)                                           |
| Inference    | NUTS/HMC, ADVI, L-BFGS                        | IF2, PGAS+NUTS, PMMH, particle filter; MH/NUTS/NLopt on the ODE backend                                                                   |
| Types        | `real`, `int`, constrained containers         | dimensional: `rate` is `T⁻¹`, `count` is `P`; `beta * S * I` is `E300`                                                                    |
| Scope        | anything with a density                       | compartmental models only — but with stratification, interventions, calendar time, multi-stream observations, and run provenance built in |

### Can a camdl model be written in Stan? It depends entirely on the backend

**On the ODE backend — yes, essentially exactly.** A fit with
`backend = "ode"` targets the deterministic marginal likelihood

```
p(y | θ, ODE) = ∏_t p(y_t | π_t(x(θ)), θ)
```

— integrate the mean-field system, score the observation model against the
projected incidence or prevalence. That is precisely what a Stan ODE-SIR does
with `ode_rk45` plus `y ~ neg_binomial_2(...)`. The translation is mechanical:

| camdl                                                 | Stan                                                  |
| ----------------------------------------------------- | ----------------------------------------------------- |
| `compartments { S, E, I, R }`                         | the state vector of the derivative function           |
| `infection : S --> I @ beta * S * I / N`              | a `-`/`+` pair of terms in `dz_dt`                    |
| `beta : rate in [0.001, 2.0]`                         | `real<lower=0.001, upper=2.0> beta;`                  |
| `~ log_normal(mu = -1, sigma = 0.5)`                  | `beta ~ lognormal(-1, 0.5);`                          |
| `cases ~ neg_binomial(mean = rho * projected, r = k)` | `cases ~ neg_binomial_2(rho * incidence, k);`         |
| `stratify(...)` over an age dimension                 | hand-written index loops and a flattened state vector |

What camdl gives you here is convenience and safety, not capability: dimensional
checking, stratification expansion, symbolic forward sensitivities instead of
autodiff through the solver, content-addressed run storage. What Stan gives you
is everything outside the compartmental box. In one corner Stan is strictly
*more* capable: camdl's `nuts`-on-`ode` refuses scheduled interventions and
events, adaptive `rk45`, and drawn or parameterized initial conditions (see "The
differentiability requirement" in [`inference.md`](inference.md)), whereas in
Stan you can split the integration at intervention times by hand and keep the
gradient.

**On the stochastic backends (`chain_binomial`, `gillespie`) — no, not the same
model.** This is the real divide, and it is structural rather than a matter of
effort:

1. **The transition density has no closed form.** You can simulate from the
   Euler-multinomial or Gillespie kernel, but the transition density
   `p(x_t | x_{t-1}, θ)` has no pointwise form in general. HMC needs a
   log-density it can evaluate *and* differentiate.
2. **The latent state is discrete.** Stan cannot sample discrete parameters at
   all. Its supported workaround — marginalizing them out by explicit
   enumeration — is fine for a two-state HMM and useless for compartment counts
   in the tens of thousands.
3. **A particle-filter estimate does not rescue it.** Pseudo-marginal MCMC
   (which is exactly what PMMH is) is valid inside a Metropolis–Hastings
   acceptance ratio, but *not* inside HMC: HMC's dynamics need the gradient of
   the true log-density, and a noisy unbiased estimate of the density is not a
   substitute. So there is no way to smuggle camdl's particle filter into Stan
   and keep a valid sampler.

What people actually do in Stan is write a **different, approximating model**: a
continuous state-space form — Euler–Maruyama diffusion, the linear noise
approximation, or a log-normal transition density — with the latent trajectory
declared as parameters and sampled jointly with θ. This is a legitimate and
widely used approach, but be clear about what you have bought:

- it is an approximation that degrades exactly where you wanted the
  stochasticity, at small counts. Extinction and re-introduction are the classic
  case: a diffusion approximation has no absorbing state at zero, so it cannot
  represent fadeout, which is the phenomenon that made you choose a stochastic
  model.
- the latent path adds `T × n_compartments` parameters. The posterior develops
  funnel geometry, and divergent transitions follow.
- `overdispersed(rate, σ²)` (Gamma noise on the rate, giving Negative-Binomial
  event counts, He et al. 2010) has the same discreteness problem.

**If you are leaving camdl for Stan anyway**, the things you gain are worth
naming: arbitrary hierarchical and partial-pooling structure, arbitrary
likelihood terms, regression/spline/GP components alongside the mechanism, a
`generated quantities` block, and the `posterior`/`loo`/`bayesplot` ecosystem.
camdl has none of that and does not intend to.

---

## pomp

**The closest tool in problem domain**, and the reference implementation camdl
checks itself against. pomp (King, Ionides, and collaborators) is the R
framework for partially-observed Markov process inference, and camdl's
stochastic path is deliberately the same statistical machinery: the
chain-binomial backend matches pomp's `reulermultinom` semantics exactly, IF2
follows the pomp-canonical estimator including the `cooling.fraction.50`
convention, and the particle-filter ESS default matches pomp's. A model fitted
in both should agree; that is a test, not a coincidence
(`tests/external/cases/boarding_school_sir` fits pomp's canonical `bsflu`
tutorial).

**The honest statement of overlap: any camdl chain-binomial model can be written
in pomp.** pomp is strictly more general in what process it can represent — you
supply `rprocess` as a C snippet and it can do anything, compartmental or not.
The reverse is *not* true: a pomp model whose state update is not a set of
compartment-to-compartment flows has no camdl form at all.

What camdl adds is the compiler layer. [`user-features.md`](user-features.md)
shows the same He et al. (2010) London measles model side by side: 20 lines of C
for the school-term forcing becomes 4 ranges; the manual
`rgammawn`/`reulermultinom`/index arithmetic becomes one `overdispersed(...)`
transition; the hand-maintained `parameter_trans(log = ..., logit = ...)` list
becomes a consequence of the parameter types. Beyond ergonomics camdl adds
dimensional checking, symbolic gradients (which is what makes PGAS+NUTS possible
— pomp has no gradient path), and content-addressed run storage.

What pomp keeps: two decades of maturity, the R data and plotting ecosystem, a
large published literature that reviewers recognize, and estimators camdl does
not have (ABC, nonlinear forecasting, spectrum matching, `probe` synthetic
likelihood).

**Rule of thumb:** if you are writing a one-off POMP model and you are fluent in
R and C, pomp is a completely reasonable choice. If the model is stratified,
will be refitted many times, or needs a Bayesian posterior with good mixing on a
long series, camdl's compiler and PGAS are the reason to switch.

---

## odin / dust / monty (the Imperial stack)

**The closest tool in architecture.** odin is an R DSL for compartmental models
that compiles to C++; `dust` is the fast parallel (and GPU-capable) runtime for
the discrete-time stochastic form; `mcstate`, and its successor stack
`odin2`/`dust2`/`monty`, provide particle-filter-based inference including
PMCMC. That is the same three-layer shape as camdl: DSL → compiler → fast
runtime + SMC inference. It is the stack behind `sircovid`, which ran England's
COVID-19 response modelling — a scale of production use camdl has not remotely
approached.

The differences are in the DSL's level of abstraction. odin is
**equation-oriented**: you write the update equations or derivatives directly,
including the random draws (`n_SI <- rbinom(S, p_SI)`) and the bookkeeping that
moves counts between states. camdl is **transition-oriented**: you declare
`infection : S --> I @ rate` and the compiler derives the stoichiometry, the
competing-risks draw, and the gradient. odin has array dimensions but not
physical dimensions, so a dropped `/ N` is not a compile error there. On
inference, the Imperial stack centres on particle filters and PMCMC; camdl adds
IF2 and PGAS+NUTS.

If you work in R and want a battle-tested stochastic compartmental stack today,
this is the serious alternative to camdl, and the one whose feature set overlaps
most. (Their tooling moves quickly — check the current `odin2`/`monty`
documentation rather than trusting this paragraph's details.)

---

## LibBi

A DSL for state-space models compiled to C++/CUDA, with SMC, PMCMC, SMC², and
bridge particle filters — historically the closest thing to camdl's ambition,
and GPU-accelerated well before it was common. It is not compartment-oriented:
you write the transition and observation *densities* of a general state-space
model, so there is no stoichiometry derivation and no dimensional checking.
Development has been largely dormant for years; mentioned here because if you
find LibBi papers while searching for prior art, this is where it fits.

---

## NIMBLE

A BUGS-dialect DSL with its own compiler to C++, plus `nimbleSMC` for bootstrap
and auxiliary particle filters, Liu–West, and particle MCMC. It can represent a
discrete-state stochastic epidemic model — BUGS has always been able to write
`n_SI ~ dbin(p, S)` — and can fit it with particle MCMC, so a camdl
chain-binomial model does have a NIMBLE form. What you write by hand is
everything camdl's compiler derives: the flows, the competing-risks structure,
the transforms, the stratification loops. NIMBLE's counterpart advantage is real
generality plus the ability to assign custom samplers to individual nodes, which
is a level of control camdl deliberately does not expose.

---

## Turing.jl and AdvancedPS

Turing is the general-purpose PPL whose inference menu overlaps camdl's most:
alongside HMC/NUTS it supports SMC, particle Gibbs, and **particle Gibbs with
ancestor sampling** — the same PGAS algorithm that is camdl's default Bayesian
path — via `AdvancedPS`. It also handles discrete latent variables natively, so
unlike Stan there is no structural wall. Combine it with `Catalyst.jl` or
`JumpProcesses.jl` for the process and you can assemble, in Julia, something
with camdl's capability profile.

The cost is that *you* assemble it: no dimensional checking, no stratification
expansion, no observation-block/data-alignment layer, no fit workflow or
provenance, and performance that depends on how carefully you wrote the model.
For a methodologist who is fluent in Julia and wants to modify the algorithm,
this is more attractive than camdl. For a modeller who wants to fit a stratified
measles model to weekly case data by Thursday, it is not.

---

## Catalyst.jl and the SciML stack

Catalyst's reaction-network DSL is the closest thing anywhere to camdl's
"transitions read as math" surface — `S + I --> 2I` is a chemical reaction and a
transmission event in the same notation — and one Catalyst model lowers to ODE,
SDE (chemical Langevin), or jump (exact Gillespie and tau-leaping) form,
mirroring camdl's three backends. If you want reaction-network modelling
generally, it is excellent and far more general than camdl.

What it is not is an epidemiology tool: there is no observation block, no
mapping from a data file's dated columns to a latent projection, no incidence
accumulation over a reporting interval, no `interventions {}`, and inference is
whatever you bolt on through Turing or DiffEqBayes. Roughly, Catalyst is camdl's
`transitions` block done better and more generally, with none of the rest of
camdl attached.

---

## PyMC and NumPyro

Same structural story as Stan. ODE models are fine and well supported (`sunode`
for PyMC, `diffrax` for NumPyro). Discrete latent state hits the same wall —
NumPyro can enumerate small discrete latent variables, which does not extend to
compartment counts. If you are in Python and your model is
deterministic-plus-observation-noise, these are strong choices; if it is
genuinely stochastic, they are not.

---

## Renewal-equation tools: EpiNow2, epidemia, EpiEstim

These answer a different question. They estimate a time-varying reproduction
number from case data using a renewal process and a delay structure, without a
mechanistic compartment model underneath. If what you want is "what is Rt right
now, with a decent uncertainty interval, this afternoon", use one of these —
there is no model to write. camdl is for when you need mechanism: explicit
compartments you can intervene on, counterfactual scenarios, structural
comparison between model variants, or a latent state with meaning beyond
incidence.

---

## Agent-based frameworks: Starsim, EMOD, EpiModel

A different modelling resolution, not a different tool for the same job. When
individual heterogeneity, contact networks, individual-level histories, or
within-host state matter, compartments are the wrong abstraction and no amount
of stratification fixes it — you are approximating a network by a mixing matrix.
These frameworks are simulation-first: calibration is typically optimizer-driven
or ABC-flavoured rather than likelihood-based, and none of them offers a
particle filter over the joint agent state (the state space makes that
infeasible).

Choose by the question, not the tool: compartments if the population can be
partitioned into a modest number of homogeneous groups, agents if the answer
depends on who is connected to whom.

---

## Feature-by-feature: what survives a port out of camdl

| camdl feature                            | Stan                        | pomp             | odin/monty        | Turing.jl           |
| ---------------------------------------- | --------------------------- | ---------------- | ----------------- | ------------------- |
| ODE backend fit                          | yes, mechanical             | yes              | yes               | yes                 |
| `chain_binomial` process likelihood      | no (approximation only)     | yes, native      | yes, native       | yes                 |
| Gillespie / extinction dynamics          | no                          | yes              | via other tooling | yes (JumpProcesses) |
| `overdispersed(rate, σ²)`                | no                          | yes (`rgammawn`) | by hand           | by hand             |
| IF2                                      | no                          | yes, canonical   | no                | no                  |
| PGAS                                     | no                          | no               | no                | yes (AdvancedPS)    |
| PMMH                                     | no (invalid in HMC)         | yes              | yes (PMCMC)       | yes                 |
| Dimensional checking of rates            | no                          | no               | no                | no                  |
| `stratify(...)` expansion                | by hand                     | by hand          | array dimensions  | by hand             |
| `interventions {}` / `events {}`         | by hand (split integration) | by hand          | by hand           | callbacks           |
| Multi-stream, multi-cadence observations | by hand                     | by hand          | by hand           | by hand             |
| Hierarchical / partially-pooled priors   | yes, best in class          | yes              | yes               | yes                 |
| Arbitrary likelihood terms               | yes                         | yes              | yes               | yes                 |
| Content-addressed run provenance         | no                          | no               | no                | no                  |

The pattern: the compiler-derived conveniences are camdl-specific, the
statistical capability is not unique, and the general-purpose tools beat camdl
decisively the moment you step outside a compartmental model.

---

## What camdl deliberately doesn't do

Worth knowing before you pick it:

- **It is not a general probabilistic programming language.** There is no
  `target +=`, no arbitrary likelihood, no regression, spline, or GP component
  beside the mechanism. Observation models come from a fixed menu — `poisson`,
  `neg_binomial`, `normal` (a *discretized count* likelihood, pomp/He et al.
  convention, not a continuous density), `binomial`, `beta_binomial`, `beta`,
  `bernoulli`, plus `diagnostic_test` sugar. If your data needs something else,
  camdl cannot express it.
- **Compartmental models only.** No individual heterogeneity, no networks, no
  within-host state.
- **Hierarchical priors are limited.** Declared hierarchical priors are gated
  under PGAS (gh#175); the working route today is the non-centered form written
  with a `let` (see the funnel section in
  [`diagnosing-fits.md`](diagnosing-fits.md)). Any general PPL is far ahead of
  camdl here.
- **Only two inference backends.** `chain_binomial` and `ode`. Gillespie
  simulates; it does not fit.
- **Alpha software.** One implementation, breaking changes expected pre-1.0
  ([`VERSIONING.md`](../VERSIONING.md)), a small user base, and very little
  camdl in any LLM's pretraining data — which is why an agent should analogize
  from pomp or Stan and then verify against the spec rather than trusting
  recall.

---

## Choosing, in one paragraph

If the mean-field skeleton is good enough and you want maximum flexibility, use
**Stan**, **PyMC**, or **NumPyro** — camdl's ODE backend competes on
convenience, not capability. If you need the stochastic-process likelihood and
you live in R, the real choice is **pomp**, the **odin/monty** stack, or camdl,
and it turns on whether you want the compiler layer (dimensional types,
stratification expansion, symbolic gradients feeding PGAS) enough to accept
alpha software. If you live in Julia and want to modify the algorithm rather
than the model, **Turing.jl + Catalyst.jl** gets closest. If your question is Rt
from case counts, use **EpiNow2** and skip all of this. If the answer depends on
who is connected to whom, you want an **agent-based** model and compartments are
the wrong abstraction. camdl is the right call when the model is a stochastic
compartmental model that is stratified, refitted often, compared across
variants, and needs to be *correct* under maintenance by people — or agents —
who did not write it.
