---
date: 2026-10-29
status: proposed — the maintainer decisions of 2026-10-29 (rename `rw_sd` rather than change its unit; move algorithm-specific parameter settings into `[method]`; PMMH and MH get their own initial proposal SD defaulting to `default_mh_step`; PGAS MH-within-Gibbs reads no per-parameter step) are taken as given; D1–D13 (§13) record the calls made under them; D12, an amendment to the accepted parameter-roles types (§5.4), is pending the maintainer's confirmation
area: fit config / run identity / IF2, PMMH and MH step sizes / profile CLI
issues: gh#1001
supersedes: 2026-10-29-support-keyed-transforms.md D9 and its commit series (ii) (§11 commits 11–12)
amends:
  - 2026-10-29-support-keyed-transforms.md §2 ("Step"), §9.1, §9.2, §11, §13 (§12.1 here)
  - 2026-10-26-parameter-roles.md §3.2 item 8, §4.1, §4.2, §6.3, §6.4, §6.6, §7.2, §10.3, §11.2 Stages 2 and 3, D5, §13 (§12.3 here; §6.3/§6.4/§6.6/Stage 2 pending confirmation, D12)
  - 2026-09-08-workflow-first-fit-config.md §5 "Kept as they are" and decision 12 (§3.7 here)
---

# A step size belongs to the method that takes it: `[estimate]` states the parameter, `[method]` states how each algorithm moves it

Code references are against `origin/main` at `a9a89178` unless marked _branch_,
which means `origin/feat/support-transforms` at `63dcafd6`, the implementation
of the support-keyed transforms proposal. That proposal is cited from the
branch, whose text is newer than the revision on main; the section numbers are
the same in both. A claim marked _verified_ was read at the cited line or
produced by the command shown. _Inferred_ means reasoned from the code and not
executed. Nothing in this document was measured by running a fit.

## 1. One key, `rw_sd`, sets two different algorithms' steps in a unit that is about to change; it is replaced by one name per algorithm

An estimated parameter θ is moved by a method in an unconstrained coordinate z,
with θ = g(z) (the support-keyed transforms proposal, §2). Two methods take a
per-parameter step there. IF2 (iterated filtering, Ionides et al. 2015) adds a
random-walk **perturbation** to each particle's θ before each observation. PMMH
(particle marginal Metropolis–Hastings) and MH (Metropolis–Hastings on the ODE
likelihood) draw a **proposal** around the current θ. These are different
objects: the perturbation is the kernel of a cooled random walk that defines
what IF2 maximizes, and the proposal is the starting shape of an adapted
Metropolis kernel whose stationary distribution does not depend on it.

Today both are set by one fit-config key, `[estimate].rw_sd`, a step on the
natural scale of θ. IF2 converts it to z at each particle's current value; PMMH
and MH take five times it, converted at the start (§3.1). The support-keyed
transforms proposal fixes IF2's step in z and gives each map family a default
(its D8), and its D9 made an explicit `rw_sd` mean z. D9 is on hold, because
reading the same key in a new unit would silently change every hand-tuned
config.

This proposal replaces D9. `[estimate]` keeps only what is true of a parameter
whatever method runs: its bounds, its start and its prior (and its role, per the
parameter-roles proposal). The steps move into the `[method]` table under a name
per algorithm, in z, typed:

```toml
# IF2                                  # PMMH or MH
[method.perturbation]                  [method.proposal]
beta = 0.107                           beta  = 0.537
I0   = { sd = 0.174, initial_value = true }
```

The defect that makes this more than a rename: because both keys sit in
`[estimate]`, which the problem identity hashes, and because every method but
IF2 refuses `perturb_only_at_t0`, an IF2 file and a PGAS file for one problem
with an initial-value parameter cannot share a fit segment (§3.7).

`rw_sd` and `perturb_only_at_t0` in `[estimate]` are refused at load, with the
new spelling and the converted value printed where it can be computed (§6). A
name left out takes the D8 default. Because the steps leave `[estimate]`, they
leave the fit-level digest, which is the identity of the statistical problem,
and enter the method-level digest, which is the identity of one way of running
it (§5). `profile`'s `--rw-sd`, which both chose the estimated set and gave
steps, is replaced by `--perturbation` and `--initial-value`, with the estimated
set declared by `--estimate` as the parameter-roles proposal already decides
(§7).

## 2. Terms

- **θ, z, g.** θ is a parameter on its natural scale; z is the unconstrained
  coordinate a method moves in; g is the map θ = g(z), chosen from the
  parameter's effective domain (support-keyed transforms, §4.1). |dθ/dz| = |g′|.
- **Simplex group.** A set of estimated parameters constrained to sum to 1
  (`[[simplex_groups]]`), which IF2 perturbs jointly in log-ratio coordinates
  (`sim/src/inference/if2.rs:58-104`). For a member, "z" in this document means
  its log-ratio coordinate.
- **Perturbation SD.** The standard deviation, in z, of IF2's random-walk
  increment for one parameter before cooling. pomp's `mif2` calls it `rw.sd`.
- **Initial-value parameter (IVP).** A parameter that IF2 perturbs at t = 0
  only, not before every observation: an initial state such as `I0`, whose
  effect on the trajectory is spent before the first step. pomp marks one with
  `ivp()`. Today's key is `perturb_only_at_t0`.
- **Proposal SD.** The standard deviation, in z, of the diagonal Gaussian
  proposal PMMH and MH draw from before adaptation reshapes it.
- **D8 default.** The step the support-keyed transforms proposal gives each map
  family when none is written: for IF2 `Transform::default_if2_step` (2/3 on
  `Interval`, 0.2 on `LogInterval` and the half-bounded maps, _branch_
  `types.rs:66-74`), for PMMH and MH `Transform::default_mh_step` (1.0,
  `types.rs:79`). The identity map has neither.
- **Fit level, method level.** The first two of the three levels a fit leaf's
  `run_id` is hashed from (`cli/src/fit/cas.rs`, `fit_stage_levels`). The fit
  level hashes the model, the data and the `Problem` (every fit-config table but
  `[method]` and `fit_seeds`); its hash is the eight-hex suffix of
  `fits/{stem}-{h8}/`. The method level hashes `[method]`. Here the fit level is
  called the _problem identity_ and the method level the _run identity_ of a
  method.

## 3. What the code does today

### 3.1 `rw_sd` is read by IF2 and by PMMH and MH, in natural units, and by nothing else

| reader                  | what it does with `rw_sd`                                                                                                                                                                                    | where (verified)                                                                        |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| IF2                     | perturbation SD at particle i is `transformed_sd(rw_sd, θᵢ)`, the natural step over \|g′\| at that particle's value; the same at the per-parameter diagnostic                                                | `sim/src/inference/if2.rs:428`, `:533`, `:639`                                          |
| IF2, simplex member     | taken as-is on the log-ratio scale when written; an automatic value is computed in natural units and used on the log-ratio scale                                                                             | `cli/src/fit/runner.rs:647-692`                                                         |
| PMMH, MH                | initial proposal SD is `5 × transformed_sd(rw_sd, θ₀)`, converted at the start; the sampler multiplies a standard normal by it until `adapt_start` (default 300), and for the whole run when `adapt = false` | `cli/src/fit/pmmh.rs:183`, `:1045-1053`; `sim/src/inference/pmmh.rs:654-667`            |
| PGAS (MH-within-Gibbs)  | not read: the initial scale is a tenth of the z-range of the box                                                                                                                                             | `sim/src/inference/pgas.rs:3955-3965`                                                   |
| `nl-sbplx/bobyqa/lbfgs` | not read; copied into `fit_state.toml`'s `[rw_sd]`                                                                                                                                                           | `cli/src/fit/nlopt_stage.rs:262-299`                                                    |
| `pfilter`               | not read                                                                                                                                                                                                     | `grep -n "rw_sd\|perturb" rust/crates/cli/src/pfilter.rs` → no match                    |
| `survey`                | passed into `EstimatedParam` and never used: survey evaluates a landscape and runs no IF2                                                                                                                    | `cli/src/survey.rs:950-956`; `grep -n run_if2 rust/crates/cli/src/survey.rs` → no match |
| `profile`               | `--rw-sd` is required; its keys are the estimated set (`auto` estimates every non-focal, non-fixed parameter); a value overrides `[estimate].rw_sd` from `--fit`                                             | `cli/src/profile.rs:294-302`, `:847-857`, `:870-891`                                    |

On the branch the PGAS row is the same: commit `6bd627d1` ("PGAS MH-within-Gibbs
does not read rw_sd") starts it at `default_mh_step`, 1.0 in z (_branch_
`pgas.rs:3872-3886`), after an earlier branch commit had made it read `rw_sd`. A
description of the branch that says PGAS's MH-within-Gibbs reads `rw_sd`
describes that earlier commit. On the branch, PMMH and MH start at
`default_mh_step` when `rw_sd` is omitted and at five times the z-converted
`rw_sd` otherwise (_branch_ `types.rs:437-443`, `mh_initial_step`).

The automatic value (no `rw_sd` written) is in natural units on main
(`runner.rs:1411-1413`, `auto_rw_sd_from_value`) and is replaced by the D8
default in z on the branch (_branch_ `runner.rs:1458-1466`).

### 3.2 `perturb_only_at_t0` is IF2's schedule, and four places read it

IF2 skips a flagged parameter in the per-observation perturbation
(`if2.rs:533`). Every other algorithm refuses the flag at load
(`config_v2.rs:2437-2457`). Three further readers depend on it: the `ic_free`
precondition, which accepts an IVP as the source of per-particle variation at t
= 0 under IF2 (`runner.rs:511-556`); the IF2 convergence check, which exempts
IVPs from the chain-agreement threshold (`gating.rs:190`); and the stored record
`fit_state.toml`, which lists them as `perturb_only_at_t0_params`
(`state.rs:62-68`) so that `fit summary` applies the same exemption later
(`fit_summary.rs:1597`, `:3071`).

### 3.3 Both keys are part of the problem identity

The fit level hashes the whole `Problem` minus `output_dir` and `provenance`
(`cas.rs:420-442`, `fit_config_blob_hash`, which calls
`canonical_config_hash(problem, FIT_PROVENANCE_KEYS)`). `Problem.estimate` is an
`IndexMap<String, EstimateSpecV2>` (`config_v2.rs:111`), and `EstimateSpecV2`
serializes `rw_sd` and `perturb_only_at_t0` with no skip
(`config_v2.rs:573-626`, `#[serde(default)]` only). So two fit configs that
differ only in an IF2 step have different fit-level hashes: the store files them
as two statistical problems, in two fit directories, although they pose one. The
same holds for a PMMH config and an IF2 config of one problem, since the PMMH
config may not carry `perturb_only_at_t0` (§3.2) and the IF2 one does.

`profile --fit` hashes the fit config's raw bytes into its base level, the level
that is meant to identify the profiled problem (`profile.rs:359-366`, the
`fit_toml` field of `ProfileBaseLevel`, `:134-147`). A change to any byte,
`[method]` and comments included, re-keys every profile cell of that config.

### 3.4 The parameter-roles proposal's finding 8 is false as written, and needs re-checking there

`2026-10-26-parameter-roles.md` §3.2 item 8 says that `perturb_only_at_t0` and
`rw_sd` in `[estimate]` "change nothing under PMMH", reproduced by an
independent check. For `rw_sd` that is false. The PMMH stage builds its proposal
from it (`cli/src/fit/pmmh.rs:183` calls `build_proposal_sd`, `:1049-1052`,
which returns `p.transformed_sd(p.rw_sd, p.initial) * 5.0`), passes the vector
as `PMMHConfig.proposal_sd` (`:575`), and the sampler draws its increment as
`rng.normal() * config.proposal_sd[i]` before `adapt_start` and through the
learned shape after it (`sim/src/inference/pmmh.rs:661-667`). Changing `rw_sd`
therefore changes PMMH's draws; it also moves the fit-level hash (§3.3). This is
verified by reading, not by running a PMMH fit. The same proposal's §6.4 says
"PMMH reads `rw_sd` for its initial proposal", so the document contradicts
itself. For `perturb_only_at_t0` the finding is also stale: PMMH refuses it at
load (`config_v2.rs:2437-2457`). The comment at `config_v2.rs:2441-2442` gives
the correct reading. Item 8 should be re-checked and corrected in that document;
§12.3 lists the edit.

### 3.5 `fit_state.toml`'s `[rw_sd]` is written and never read

IF2 writes a MAD-calibrated suggestion (`fit/mod.rs:1434-1439`,
`runner::auto_rw_sd`), the NLopt stages copy the input steps
(`nlopt_stage.rs:262-273`), and PMMH and PGAS write an empty table. Nothing
reads `FitState.rw_sd`: `grep -rn "\.rw_sd" rust/crates/cli/src` returns only
`EstimatedParam` and `EstimateSpecV2` field accesses and the writers. Its values
are in natural units and would be in the wrong unit after this change.
`FitState` has no `deny_unknown_fields` (`state.rs:19-20`), so dropping the
field leaves old files loadable.

### 3.6 pomp keeps the two knobs apart, and Stan has no per-parameter step at all

pomp's IF2 takes `rw.sd`, built with `rw_sd()`: "specification of the magnitude
of the random-walk perturbations that will be applied to some or all model
parameters … If parameter transformations have been supplied, then the
perturbations are applied on the transformed (estimation) scale" (`man/mif2.Rd`,
the `rw.sd` argument, pomp 6.5.0.0 on GitHub master, read 2026-10-29).
Initial-value parameters are marked inside that specification: "`ivp(s)` can be
used in this context as shorthand for `ifelse(time==time[1],s,0)`" (same entry).
pomp's PMCMC takes a separate argument, `proposal`, a function drawing θ′ given
θ (`R/pmcmc.R`, roxygen `@param proposal`; the loop calls `proposal(theta, …)`
at line 371). The usual constructor is `mvn_diag_rw(rw.sd)`
(`man/proposals.Rd`), so the vector of SDs is again called `rw.sd`, but it is an
argument of the proposal constructor, not of `pmcmc`. `R/pmcmc.R` applies no
parameter transformation to the proposal (`grep -n partrans R/pmcmc.R` → no
match): pomp's PMCMC proposes on the natural scale, unlike its IF2.

The software paper makes the same split: Algorithm 3 is
`mif(P, start = θ0, Nmif = M, Np = J, rw.sd = σ1:p, ic.lag = L, …)` with "I ⊂
{1, …, p}, designating IVPs", and Algorithm 4 is
`pmcmc(P, start = θ0, Nmcmc = M, Np = J, proposal = q)` (King, Nguyen and
Ionides, "Statistical Inference for Partially Observed Markov Processes via the
R Package pomp", _J. Stat. Softw._ 69(12), 2016, doi:10.18637/jss.v069.i12,
§3.2). Its worked examples pass `rw.sd = c(r = 0.02, …)` to `mif` with
`transform = TRUE` (§4.3) and `proposal = mvn.diag.rw(c(r = 0.01, …))` to
`pmcmc` (§4.4).

Stan transforms every constrained parameter "to an unconstrained variable behind
the scenes" for sampling (Stan Reference Manual 2.40, "Constraint Transforms"),
and its HMC tuning — "discretization time ε (aka step size), metric M (aka mass
matrix), and number of steps taken L" ("MCMC Sampling", "HMC algorithm
parameters") — is global and adapted, not per-parameter. Stan has no user-facing
per-parameter step to name; what it does settle is that the sampler's tuning
lives on the unconstrained scale, which camdl's z is.

### 3.7 The motivating defect: an IF2 file and a PGAS file for one problem cannot share a fit segment

The workflow-first fit-config proposal kept `perturb_only_at_t0` and `rw_sd` in
`[estimate]` on the claim that "under one method per file they are inert for a
non-IF2 method and the loader says nothing, so that a problem's IF2 comparator
file and its PGAS file differ only in `[method]` and share a fit-level hash"
(`2026-09-08-workflow-first-fit-config.md:934-942`, and its decision 12). The
code does the opposite for `perturb_only_at_t0`: any method but IF2 refuses it
at load (`config_v2.rs:2438-2457`, verified). So a model with an initial-value
parameter has an IF2 file that carries the flag and a PGAS file that must not.
Their `[estimate]` tables differ, `fit_config_blob_hash` covers `[estimate]`
(§3.3), and the two files land in different fit segments although they pose one
problem. `rw_sd` is not inert either: PMMH and MH read it (§3.1, §3.4).

The red test this series starts from, in `cli/src/fit/cas.rs`'s test module:

```rust
#[test]
fn an_if2_file_and_a_pgas_file_of_one_problem_share_the_fit_level() {
    // The §9 problem. The IF2 file declares I0 an initial-value parameter;
    // the PGAS file is the same problem written as PGAS must be written.
    let if2  = FitConfig::parse(SIR_WITH_I0_IF2).unwrap();   // today: I0 = { …, perturb_only_at_t0 = true }
    let pgas = FitConfig::parse(SIR_WITH_I0_PGAS).unwrap();  // today: I0 = { … }, no flag
    assert_eq!(fit_level_hash_of(&if2.problem), fit_level_hash_of(&pgas.problem));
}
```

Today it is red: the two `Problem`s serialize different `[estimate]` entries,
and the fit level hashes them (inferred from `cas.rs:420-442` and
`config_v2.rs:573-626`; the test has not been run). After commit 3 of §10 the
IF2 file carries the flag as
`[method.perturbation] I0 = { initial_value = true }`, both `[estimate]` tables
are identical, and the test is green. The mutation check is to put
`perturbation` back into `EstimateSpecV2`'s serialization, which turns it red.

## 4. The types: `[estimate]` loses the steps, each algorithm variant gains its own table

### 4.1 The unit is a type

```rust
// rust/crates/sim/src/inference/types.rs

/// A standard deviation in z, the coordinate a method moves in (θ = g(z),
/// `Transform`; a simplex member's log-ratio coordinate). Finite and > 0 by
/// construction, so no consumer re-checks it.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ZSd(f64);

#[derive(Debug, Clone, PartialEq)]
pub struct NotAZSd(pub f64);           // NaN, ±∞ or ≤ 0

impl ZSd {
    pub fn new(sd: f64) -> Result<Self, NotAZSd>;
    pub fn get(self) -> f64;
}
// Deserialize goes through `new`; the error names the value and says the
// unit is z.

/// When IF2 perturbs a parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerturbAt {
    /// At t = 0 and before every observation.
    EveryObservation,
    /// At t = 0 only: an initial-value parameter.
    InitialValue,
}

/// IF2's random-walk perturbation of one estimated parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Perturbation {
    pub sd: ZSd,
    pub at: PerturbAt,
}
```

### 4.2 The sim crate separates the parameter from the setting

`EstimatedParam` keeps what every method needs and loses the three IF2 and MH
fields, so an IF2 setting cannot reach PMMH through it:

```rust
// before (branch types.rs:360-386)          // after
pub struct EstimatedParam {                  pub struct EstimatedParam {
    pub name: String,                            pub name: String,
    pub index: usize,                            pub index: usize,
    pub initial: f64,                            pub initial: f64,
    pub rw_sd: f64,                              pub transform: Transform,
    pub transform: Transform,                }
    pub rw_sd_auto: bool,
    pub perturb_only_at_t0: bool,
}
// `natural_step` and `mh_initial_step` (branch types.rs:427-443) are deleted;
// their callers read the resolved steps (§4.4).
```

IF2 takes its perturbations beside the parameters, as PMMH already takes
`PMMHConfig.proposal_sd: Vec<f64>` (`sim/src/inference/pmmh.rs:33`):

```rust
// rust/crates/sim/src/inference/if2.rs
pub struct IF2Config {
    // …existing fields…
    /// One per entry of `if2_params`, in the same order. `run_if2` returns
    /// `SimError::Config` on a length mismatch.
    pub perturbations: Vec<Perturbation>,
}

pub struct SimplexGroup {
    pub indices: Vec<usize>,
    pub sds: Vec<ZSd>,          // was `rw_sds: Vec<f64>`, on the log-ratio scale
}
```

The three read sites become `config.perturbations[pi].sd.get() * cooling_now`
(`if2.rs:428`, `:533`, and the diagnostic at `:639`), and the per-observation
skip reads `config.perturbations[pi].at == InitialValue` in place of
`spec.perturb_only_at_t0` (`:533`). The perturbation draw itself (`:429-430`,
`:534-535`) is unchanged. `PMMHConfig.proposal_sd` stays `Vec<f64>` so
`sim/src/inference/pmmh.rs` is not edited; the CLI unwraps `ZSd` at the
boundary.

### 4.3 The fit config

```rust
// rust/crates/cli/src/fit/config_v2.rs

/// A parameter fact that holds under every method.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EstimateSpecV2 {
    pub bounds: Option<(f64, f64)>,
    pub prior: Option<EstimatePriorSpec>,
    pub start: Option<f64>,
}
// `transform` is deleted by the transforms series (its D2); `rw_sd` and
// `perturb_only_at_t0` by this one. The legacy detector (§6) runs before
// the typed parse, so neither key reaches `deny_unknown_fields`.

/// One `[method.perturbation]` entry. On the wire, a bare number is
/// `{ sd = n }`; a table takes `sd` and `initial_value`, and nothing else.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerturbationSpec {
    /// `None`: the map's D8 default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sd: Option<ZSd>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub initial_value: bool,
}

pub enum Algorithm {
    IF2 {
        // …existing fields…
        /// Per estimated parameter. A name left out takes the D8 default and
        /// is perturbed before every observation.
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        perturbation: IndexMap<String, PerturbationSpec>,
    },
    PMMH {
        // …existing fields…
        /// Initial proposal SD per estimated parameter; a name left out takes
        /// the D8 default.
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        proposal: IndexMap<String, ZSd>,
    },
    Mh {
        // …existing fields…
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        proposal: IndexMap<String, ZSd>,
    },
    // PGAS, PFilter, NlSbplx, NlBobyqa, NlLbfgs: unchanged, no table (D6).
}
```

Because each table is a field of its own variant, `[method.proposal]` under
`algorithm = "if2"` is an unknown key of that variant, and
`validate_method_keys` (`config_v2.rs:2181-2272`) refuses it. Two entries are
added to its retired-key table so that the message names the right table rather
than only "unknown key":

- `proposal` under IF2, PGAS, `pfilter` or `nl-*`: "`[method.proposal]` is the
  initial proposal SD of `pmmh` and `mh`. IF2's per-parameter setting is
  `[method.perturbation]`; `pgas`, `pfilter` and the `nl-*` optimizers have
  none."
- `perturbation` under any method but IF2: the same message with the roles
  swapped.

`validate_method_keys` builds its allowed set from the serialized method, and
the two tables skip serialization when empty, so both names are added to `known`
for their variants, as `binomial` and `ancestor_sampling` are today
(`config_v2.rs:2207-2211`).

### 4.4 Resolution fills every estimated parameter, and records where each value came from

```rust
// rust/crates/cli/src/fit/steps.rs (new)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepFrom { Written, Default }

pub struct ResolvedPerturbation { pub perturbation: Perturbation, pub from: StepFrom }
pub struct ResolvedProposal { pub sd: ZSd, pub from: StepFrom }

/// Every estimated parameter's per-method setting, in `[estimate]` order.
pub enum ResolvedSteps {
    If2(IndexMap<String, ResolvedPerturbation>),
    Mh(IndexMap<String, ResolvedProposal>),       // pmmh and mh
    None,                                          // pgas, pfilter, nl-*
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepError {
    /// A table key that is not an estimated parameter.
    NotEstimated { table: &'static str, name: String },
    /// The identity map has no default, and nothing was written.
    NoDefault { table: &'static str, name: String, kind: &'static str },
}

pub fn resolve_steps(
    algorithm: &Algorithm,
    estimated: &[EstimatedParam],          // the transforms, in [estimate] order
    simplex_groups: &[config_v2::SimplexGroup],
) -> Result<ResolvedSteps, StepError>;

impl ResolvedSteps {
    /// What the method level hashes (§5): name → value, `from` left out, so
    /// a written default and an omitted name key the same.
    pub fn identity(&self) -> serde_json::Value;
    /// The sim-side vectors, aligned with `estimated`.
    pub fn if2_perturbations(&self) -> Option<Vec<Perturbation>>;
    pub fn mh_proposal_sd(&self) -> Option<Vec<f64>>;
}
```

The rules, in order:

1. A key not in `[estimate]` is `NotEstimated`. This is the parameter-roles
   proposal's `RoleError::SettingForNonEstimated` (its §6.3) for these tables,
   and `resolve_steps` runs after `resolve_roles` so the estimated set is known.
2. A written `sd` is used as written.
3. An omitted name, or an entry with no `sd`, takes `default_if2_step()` under
   IF2 and `default_mh_step()` under PMMH and MH. A simplex member takes the
   branch's `simplex_member_default_step` (_branch_ `runner.rs:686-689`).
4. The identity map has no default: `NoDefault`, naming the table and the kind
   (an undeclared `real` or `instant`), and saying that on this map z is θ
   itself, so the value to write is a natural-scale SD. This replaces
   `refuse_degenerate_auto_rw_sd` (`runner.rs:1445-1475`; _branch_
   `:1481-1505`).
5. `initial_value = true` sets `PerturbAt::InitialValue`. On a simplex member it
   is accepted and changes nothing, because IF2 perturbs simplex members at t =
   0 only (`if2.rs:518-521`); the launch table says so for every member.

`resolve_steps` is the only producer of the sim-side vectors. The CLI's
`build_proposal_sd` (`cli/src/fit/pmmh.rs:1049`) becomes
`steps.mh_proposal_sd()`, and the IF2 runner fills `IF2Config.perturbations`
from `steps.if2_perturbations()`. The `ic_free` precondition
(`runner.rs:540-543`) asks whether any resolved perturbation is `InitialValue`,
and the IF2 convergence check (`gating.rs`) reads the same list.
`ParamSpec.rw_sd` and `ParamSpec.perturb_only_at_t0` (`runner.rs:1353-1376`) are
deleted, so `build_if2_params_from_specs` builds parameters only.

The launch table (`runner.rs:1059-1093`; _branch_ `:1135-1182`) prints, per
parameter, the map, the SD in z, `written` or `default`, the natural-scale SD it
implies at the domain's reference point (the branch's `natural_from_step`), and
`t0 only` for an IVP. The warning "N parameters using auto rw_sd" becomes "N
parameters use the default step".

### 4.5 The names

**`perturbation`** names what IF2 does with the value. Ionides et al. (2015,
_PNAS_ 112(3):719–724) and pomp's `mif2` documentation both call the IF2 kernel
a perturbation ("the magnitude of the random-walk perturbations", §3.6), and the
value is the perturbation's SD. **`proposal`** is pomp's name for the PMCMC
argument (§3.6) and the standard Metropolis–Hastings term. Under `adapt = false`
it is the proposal for the whole run; under `adapt = true` it is the proposal's
shape until `adapt_start`, while a global scale adapts from the first step
(`sim/src/inference/pmmh.rs:654-667`).

**`initial_value`** follows pomp's IVP ("initial-value parameter", JSS 2016
Algorithm 3), spelled out because `ivp` is jargon to anyone who has not used
pomp. It replaces `perturb_only_at_t0`, which describes the schedule instead of
the parameter, and which would go stale if a lag (pomp's `ivp(s, lag)`) were
ever added.

Rejected names:

- **`rw_sd` under `[method]`.** Same spelling, new unit: a config copied from an
  older document would load and run with every step reinterpreted, which is the
  silent change the maintainer's first decision rules out.
- **One name for both, such as `[method.step]`.** That is today's coupling under
  a new name: the reader cannot tell from the key which kernel it sets, and a
  value tuned for IF2 would still flow into a PMMH run of the same file.
- **`[method.rw]` or `random_walk`.** Both kernels are random walks, so the name
  does not separate them.

## 5. Identity: the problem digest stops seeing steps, the method digest sees them resolved

### 5.1 Fit run

The fit level needs no code change: once `EstimateSpecV2` has no step fields,
`fit_config_blob_hash(problem)` no longer covers them. Two configs that differ
only in `[method]`, steps included, have the same fit-level hash. That was
already true of every other `[method]` key (the docstring at `cas.rs:425-428`).

The method level hashes the resolved steps, not the written tables, so a config
that spells a default and one that omits it key the same. This follows the
precedent of `starts`, whose default is also a function of the problem and is
resolved before identity (`Method::resolve_starts`, `config_v2.rs:1281-1287`,
and the refusal in `stage_config_hash`, `cas.rs:470-484`):

```rust
// rust/crates/cli/src/fit/config_v2.rs, Method::identity_payload
// before
Algorithm::PMMH { .. } | Algorithm::Mh { .. } => self.payload_minus(&["iterations"]),
Algorithm::IF2 { .. } | … => self.payload_minus(&[]),
// after: the written tables are subtracted; their resolved form is added by
// the caller, which has the domains.
Algorithm::PMMH { .. } | Algorithm::Mh { .. } => self.payload_minus(&["iterations", "proposal"]),
Algorithm::IF2 { .. } => self.payload_minus(&["perturbation"]),
Algorithm::PFilter { .. } | Algorithm::NlSbplx(_) | … => self.payload_minus(&[]),
```

```rust
// rust/crates/cli/src/fit/cas.rs
pub struct FitStageCtx<'a> {
    // …existing fields…
    pub steps: &'a ResolvedSteps,
}

// before
fn stage_config_hash(method: &Method) -> Result<ContentHash, String> {
    method.starts()?;
    ensure_finite(method)?;
    let v = serde_json::json!({
        "identity": method.identity_payload(),
        "n_trajectories": method.algorithm.cas_n_trajectories(),
    });
    Ok(digest_value(&v))
}

// after
fn stage_config_hash(method: &Method, steps: &ResolvedSteps) -> Result<ContentHash, String> {
    method.starts()?;
    ensure_finite(method)?;
    let v = serde_json::json!({
        "identity": method.identity_payload(),
        "n_trajectories": method.algorithm.cas_n_trajectories(),
        // Every estimated parameter's step, written or default (D7).
        "steps": steps.identity(),
    });
    Ok(digest_value(&v))
}
```

`fit_stage_levels` passes `ctx.steps`, so `resume_guard`, which hashes the same
levels with the extension dimension zeroed (`cas.rs:584-599`), refuses to resume
a PMMH or MH chain under a different proposal. That is today's behaviour, since
today the step moves the fit level. `config_identity_hash` (`cas.rs:461-465`),
the handle lookup, already covers the whole file and is unchanged. The
`[provenance]` subtraction and `output_dir` are unchanged.

`ensure_finite` cannot fail on a step, since `ZSd` is finite by construction; it
stays as the check on the rest of the method.

### 5.2 Profile

`profile` runs IF2 in every cell, so it reads `[method.perturbation]` from a
`--fit` config whose `[method]` is IF2 (D9). It loads the file with
`FitConfig::load` in place of `Problem::load` (`profile.rs:337`), uses
`.problem` as today, and reads only that one table from `.inference`. Its
identity changes in two places:

- **Base level.** `ProfileBaseLevel.fit_toml`, the SHA-256 of the file's bytes
  (`profile.rs:359-366`), is replaced by `fit_problem`, the hex of
  `fit_config_blob_hash(&config.problem)`. The base level then identifies the
  problem, as its docstring says it does (`profile.rs:126-133`): a change to
  `[method]`, or to a comment, no longer re-keys it, and a change to the problem
  still does.
- **Method level.** `ProfileMethodLevel.rw_sd` and `.rw_sd_auto`
  (`profile.rs:148-163`, filled at `:1293-1310`) are replaced by
  `perturbation: serde_json::Value`, the `identity()` of the profile's resolved
  steps. A step written on the command line, written in the fit config, or taken
  by default keys by its value alone.

`survey` reads neither table and runs no IF2 (§3.1); its identity does not
change beyond the version re-key.

### 5.3 Tests the identity change lands with

- Two fit configs identical but for `[method.perturbation] beta`: equal
  `fit_level_hash`, different method-level hash.
- An IF2 config and a PMMH config of one problem (the §9 pair): equal
  `fit_level_hash`.
- `[method.perturbation] beta = 0.2` on a `LogInterval` parameter and the same
  config without the line: equal method-level hash.
- `beta = 0.2` against `beta = { sd = 0.2, initial_value = true }`: different
  method-level hash.
- Two `profile --fit` runs whose configs differ only in `[method] particles`:
  equal base-level hash.
- The §3.7 test: an IF2 file declaring `I0` an initial-value parameter and a
  PGAS file of the same problem have equal `fit_level_hash`. It is the red test
  of commit 3.

Each is checked red against a mutation: dropping `"steps"` from
`stage_config_hash` turns the first test red, and hashing the written table in
place of the resolved one turns the third red.

### 5.4 The resolved problem is shared by every method; the execution plan is not (amends parameter-roles §6.3 and §6.4; pending the maintainer's confirmation)

The parameter-roles proposal's `ResolvedProblem` (its §6.3) bundles the
partition and fixed values with `starts: DrawnStarts` and
`algorithm: AlgorithmSettings`. That is the problem/run conflation this proposal
removes from the fit config, reproduced in the resolver's output type: two
methods, or two seeds, of one fit segment cannot share a `ResolvedProblem`,
because its starts and its algorithm differ between them. Its
`AlgorithmSettings::Other` also puts PMMH, PGAS, NLopt and `pfilter` in one arm
and fills PMMH's step from IF2's automatic value (its §6.4), which §4.4 here
replaces. The amendment splits the type along the same line as the identity
levels:

```rust
// rust/crates/cli/src/fit/plan.rs (new)

/// What every method and every seed of one fit segment shares. Its digest is
/// the fit level (§5.1); nothing in it depends on `[method]`, `fit_seeds` or
/// the chain starts.
pub struct ResolvedProblem {
    pub model: Arc<CompiledModel>,
    pub partition: Partition,                   // roles §6.2
    pub fixed: ResolvedParameters,              // fixed parameters only, with provenance
    pub interventions: InterventionSet,
    pub streams: ObservationStreams,            // the loaded [data] streams
    pub objective: LikelihoodObjective,         // ic_free
    pub window: TrainingWindow,                 // the scored window and any holdout
}

/// One run: a problem, a method on it, and where its chains begin.
pub struct ExecutionPlan {
    pub problem: Arc<ResolvedProblem>,
    pub method: MethodPlan,
    pub starts: fit::chain_starts::DrawnStarts,
    pub n_chains: usize,
    pub seed: u64,
    pub identity: ResolvedFitStage,             // the three levels and the run_id (cas.rs)
}

/// The per-method settings, typed per method. Not `InferenceMethod`, which
/// names the (algorithm, backend) registry entry (`fit/methods.rs:50`).
pub enum MethodPlan {
    If2(If2Plan),        // perturbations (§4.4), cooling, iterations, particles, simplex groups
    Pgas(PgasPlan),      // sweeps, NUTS or MH-within-Gibbs, …; no per-parameter step (D6)
    Pmmh(MhPlan),        // proposal (§4.4), adaptation, particles
    Mh(MhPlan),          // proposal, adaptation, burnin_dt
    Pfilter(PfilterPlan),
    Nlopt(NloptPlan),    // nl-sbplx, nl-bobyqa, nl-lbfgs
}

pub struct If2Plan { pub perturbations: IndexMap<String, ResolvedPerturbation>, /* … */ }
pub struct MhPlan  { pub proposal: IndexMap<String, ResolvedProposal>, /* … */ }
```

`ResolvedSteps` (§4.4) is the step-bearing part of `MethodPlan`: its `If2` arm
is `If2Plan.perturbations`, its `Mh` arm `MhPlan.proposal`, and its `None` arm
the other four variants. In this series it travels alone, in `FitStageCtx.steps`
(§5.1); when the plan lands, `MethodPlan::steps_identity()` replaces it and the
method-level hash reads the plan. One `Arc<ResolvedProblem>` is built per fit
segment and shared by every `ExecutionPlan` in it.

Sections of the parameter-roles proposal this changes: §6.3 (`ResolvedProblem`
loses `starts` and `algorithm` and gains `model`, `streams`, `objective`,
`window`; `resolve_problem` loses its `starts`, `algorithm`, `n_chains` and
`seed` arguments; `If2ParamSettings`, `RwSdFrom` and `AlgorithmSettings` are
deleted; `ExecutionPlan` and `MethodPlan` are added), §6.4 (the "IF2 settings"
paragraph), §6.6 (each command builds a `ResolvedProblem` once and an
`ExecutionPlan` per method and seed), and §11.2 Stage 2 (which introduces both
types). This is pending the maintainer's confirmation (D12): it amends an
accepted proposal's types. Until it is confirmed, §4.4's `ResolvedSteps` is
self-contained and does not depend on it.

### 5.5 A later stage: the particle-filter settings leave `IF2Config`

Every method's runner is built through `FitRunConfig::build`, whose result
carries an `IF2Config` (`runner.rs:106`, `:571`). The other methods read the
time step from it (`cli/src/fit/pmmh.rs:142`, `:308`; `pgas.rs:312`;
`nlopt_stage.rs`; `dt_check.rs`; verified by
`grep -rnoE "if2_config\.\w+" rust/crates/cli/src`), and they pass IF2's cooling
arguments as placeholders: PMMH passes `1, 1.0, 1` with the comment
"IF2-specific and never read by PMMH — pass harmless values"
(`cli/src/fit/pmmh.rs:132-140`), and PGAS the same (`pgas.rs:304-310`). The
architecture review reports that `n_particles`, `t_start`,
`skip_first_obs_from_loglik` and `max_substeps` are also read by every
particle-filter method; this document verified the `dt` reads only.

The split:

```rust
// rust/crates/sim/src/inference/filter.rs (new)
/// What any particle-filter likelihood needs, under every method.
pub struct FilterSettings {
    pub n_particles: usize,
    pub dt: f64,
    pub t_start: f64,
    pub skip_first_obs_from_loglik: bool,
    pub max_substeps: u64,
}

// rust/crates/sim/src/inference/if2.rs
pub struct IF2Config {
    pub filter: FilterSettings,
    pub n_iterations: usize,
    pub cooling_fraction: f64,
    pub cooling_target_iters: usize,
    pub simplex_groups: Vec<SimplexGroup>,
    pub perturbations: Vec<Perturbation>,       // §4.2
}
```

`FitRunConfig::build` loses its `n_iterations`, `cooling` and
`cooling_target_iters` arguments and carries `FilterSettings`; IF2's runner
builds its `IF2Config` from `FilterSettings` and its `If2Plan`. This is a
separate stage after the support-transforms branch has merged, because it edits
`if2.rs`, a high-risk file the branch also edits, and it changes no result: it
lands with a byte-identity check of an IF2, a PMMH and a PGAS fit at a fixed
seed, as commit 2 of §10 does.

## 6. A config with the old keys is refused, with the new spelling and the converted value

### 6.1 Where it is refused

`FitConfig::parse` already runs migration detectors before the typed parse
(`config_v2.rs:2274-2280`). A new detector reads the raw `[estimate]` table and
returns every entry that carries `rw_sd` or `perturb_only_at_t0`:

```rust
pub struct LegacyStepKey {
    pub param: String,
    pub rw_sd: Option<f64>,            // natural scale, as written
    pub perturb_only_at_t0: bool,
}
pub struct LegacyStepKeys {
    pub algorithm: Option<String>,     // `[method] algorithm`, if the file has one
    pub entries: Vec<LegacyStepKey>,
}

/// None when the file carries neither key.
pub fn detect_legacy_step_keys(raw: &toml::Value) -> Option<LegacyStepKeys>;

/// θ₀ and |dθ/dz| at θ₀ for one parameter, when they can be found (§6.2).
pub enum Conversion { At { theta0: f64, from: Theta0From, dtheta_dz: f64 }, Unavailable(&'static str) }
pub enum Theta0From { Start, ModelValue, BoxMidpoint }

impl LegacyStepKeys {
    /// The refusal with each value converted.
    pub fn message(&self, conversions: &IndexMap<String, Conversion>) -> String;
}
```

`parse` has no model, so it refuses with every conversion `Unavailable`: the
message names each key, its new place and the formula. Every command that loads
a fit config and has a model to compile calls one function,
`config_v2::load_with_model(path)`, which on a refusal of this kind compiles the
model named by `[model] camdl`, builds each listed parameter's map with the
transforms series' `Transform::for_domain`, and prints the converted values. The
commands are `fit run`, `fit recovery`, `profile --fit`, `survey --fit`,
`simulate --fit` and `pfilter --fit`. A reader with no model, such as a handle
lookup in `fit table` or `compare`, gives the formula-only message. A model that
does not compile gives the formula-only message and the compiler's error.

Every reader refuses. None accepts the old keys, because a reader that ignored
them (`simulate --fit` never takes a step) would leave the file in circulation
until a fit read it.

### 6.2 How the value is converted, and when it cannot be

The value printed is the z-SD that gives the old natural-scale step at the
start:

sd_z = rw_sd / |dθ/dz| at θ₀, with |dθ/dz| = |g′(g⁻¹(θ₀))| for the new map g.

At θ₀ this reproduces the step the old config took there. On main, PMMH's
initial proposal was computed at exactly this point (`p.initial`,
`cli/src/fit/pmmh.rs:1051`), so a PMMH value converted this way gives the same
natural-scale proposal at the start, and the message prints five times it under
`[method.proposal]` (D5). Under IF2 main converted at every particle's current
value (`if2.rs:428`), so no single z-SD reproduces it away from θ₀; the message
says the value matches at the start only.

θ₀ is, in order: the entry's `start`; the model's declared value for the
parameter; for the two interval maps, the domain's reference point z = 0 (the
box's arithmetic midpoint for `Interval`, geometric for `LogInterval`), with the
message saying which was used. The conversion is unavailable in four cases, each
named in the message:

- the half-bounded maps (`LowerBounded`, `UpperBounded`) with no start and no
  model value, where |dθ/dz| is the distance to the bound and has no reference
  point; the message gives the formula with θ₀ to be filled in;
- a start on a bound, which the transforms series refuses anyway (its D3);
- no model available (§6.1);
- a parameter the model does not declare, which is the unknown-parameter refusal
  and is reported as that.

The identity map is always convertible: |dθ/dz| = 1, so the value is unchanged.
A simplex member is always convertible and unchanged: an explicit `rw_sd` was
already read on its log-ratio scale (`runner.rs:656-660`).

### 6.3 What the message says

What the refusal says depends on the file's `[method]`:

| `[method] algorithm`       | `rw_sd` becomes                                       | `perturb_only_at_t0 = true` becomes                   |
| -------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `if2`                      | `[method.perturbation] NAME = <sd_z>`                 | `NAME = { sd = <sd_z>, initial_value = true }`        |
| `pmmh`, `mh`               | `[method.proposal] NAME = <5 × sd_z>`                 | already refused under these methods (§3.2); delete it |
| `pgas`, `pfilter`, `nl-*`  | was never read (§3.1); delete it                      | already refused; delete it                            |
| none (a problem-only file) | both rows above, plus `profile --perturbation NAME=…` | the IF2 row, plus `profile --initial-value NAME`      |

For the §9 IF2 config the message is:

```text
error: [estimate] sets IF2 and MH step sizes, which are settings of the method:
  `rw_sd` (a natural-scale step) and `perturb_only_at_t0` moved to [method],
  in z, the coordinate each method moves in. This file's method is `if2`.
  Delete both keys from [estimate] and add:

    [method.perturbation]
    beta  = 0.107                                 # rw_sd 0.02 at start 0.25, |dθ/dz| = 0.186
    gamma = 0.106                                 # rw_sd 0.01 at start 0.1,  |dθ/dz| = 0.0947
    I0    = { sd = 0.174, initial_value = true }  # rw_sd 2 at start 10,      |dθ/dz| = 11.5

  These reproduce the old step at the start only: IF2 now takes a fixed step
  in z. To use the default step instead (0.2 for these three parameters'
  log-interval maps), leave a parameter out, or write `I0 = { initial_value = true }`.
  See `camdl docs fit-toml`, "Step sizes".
```

The three values were computed from the `LogInterval` map, ln θ = ln lo +
ln(hi/lo)·σ(z), for which dθ/dz = θ·ln(hi/lo)·s(1 − s) with s =
ln(θ/lo)/ln(hi/lo): β on [0.05, 1] at 0.25 gives 0.18619, so 0.02/0.18619 =
0.1074; γ on [0.01, 0.5] at 0.10 gives 0.09473, so 0.1056; `I0` on [1, 100] at
10 gives 11.513, so 0.1737 (verified by evaluating the formula; not produced by
camdl).

### 6.4 The stored `fit_state.toml`

`[rw_sd]` is deleted from `FitState` (§3.5); old files still load.
`perturb_only_at_t0_params` is renamed `initial_value_params`. A file that still
carries the old key is refused by `FitState::load`, naming the key and saying
the fit predates the rename and must be run again; it would be re-run
regardless, because every fit re-keys at this release (§10). The refusal has a
precedent in the same function: `chain_index::refuse_one_based_layout`
(`state.rs:190`) refuses a pre-gh#990 layout. The alternative, loading the old
file with the new field empty, would make `fit summary` apply the
chain-agreement threshold to an old fit's IVPs and report a failure that the run
did not have (D10). The callers that walk the store already handle a load error
per fit (`fit_summary.rs:618`, `:2857` match on it).

## 7. `profile` declares its estimated set with `--estimate` and its steps with `--perturbation`

The parameter-roles proposal already decides that `profile`'s estimated set
comes from `--estimate NAME,…` or `[estimate]` in `--fit` (its §4.4, Stage 3),
and that `--rw-sd auto` is refused (its D5). It keeps `--rw-sd` as an optional,
IF2-only step flag. This proposal replaces that flag:

- `--perturbation NAME=SD[,NAME=SD…]`, repeatable: IF2 perturbation SD in z for
  an estimated name. It overrides `[method.perturbation]` from an IF2 `--fit`
  config per name, which is today's precedence for `--rw-sd` over
  `[estimate].rw_sd` (`profile.rs:870-891`).
- `--initial-value NAME[,NAME…]`, repeatable: perturb at t = 0 only. It adds to
  the IVPs an IF2 `--fit` config declares; there is no flag to remove one,
  because a profile that perturbs an initial state at every observation is a
  different estimand, made by editing the config.
- `--rw-sd` is a removed-flag trap, like `_removed_pmmh_rho`
  (`args/mod.rs:1817-1820`). After the model loads, it prints the `--estimate`
  list the old keys selected and the `--perturbation` values converted by §6.2.
  For `--rw-sd auto` it prints the parameter-roles D5 text.
- A `--perturbation` or `--initial-value` name that is not estimated is refused
  (`SettingForNonEstimated`); a name the model does not declare is the
  unknown-parameter refusal.

`survey` needs no flag. It declares its estimated set with `--estimate` or
`[estimate]` already (`args/mod.rs:1984-1992`) and takes no step (§3.1).

Until Stage 3 lands, `--rw-sd` keeps today's two jobs: it selects the estimated
set and gives natural-scale steps, converted once at the domain's reference
point by the branch's `step_from_natural` (_branch_ `types.rs:166-169`). Its
unit does not change and its name says which unit it is, so the decision not to
reinterpret a name holds in the interval. In that interval profile reads both
`--rw-sd` (natural) and `[method.perturbation]` (z), and the launch table prints
each parameter's source and its z value.

## 8. What breaks, and the migration text

### 8.1 Inputs refused

| what                                              | where, and how many                                                                                                                                                                                                                                                                                      |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rw_sd` / `perturb_only_at_t0` in a fit config    | this repository: no committed `.toml` config sets either (`git ls-files '*.toml' \| xargs grep -l rw_sd` → one comment, `tests/fixtures/polio_afp_es/fit.toml:42`); 28 occurrences in inline test configs across 14 Rust files (`git grep -c -E 'rw_sd *= *[0-9]\|perturb_only_at_t0 *= *true' -- rust`) |
| the same, in sibling repositories (tracked files) | `camdl-book`: 17 (`rw_sd =`/`rw_sd=`) in `guide/fitting/fits/poisson_free_ic.toml` and five `.qmd` pages; `camdl-vignettes`: 77 across 18 files outside `results/`                                                                                                                                       |
| `profile --rw-sd`                                 | 35 occurrences in 20 test files (`git grep -c -- '"--rw-sd"' -- rust`); `camdl-book` 14, `camdl-vignettes` 14, `camdl-garki` 3. Lands with roles Stage 3 (§7)                                                                                                                                            |
| `fit_state.toml` with `perturb_only_at_t0_params` | refused at load (§6.4); none committed in `camdl-vignettes` (`git ls-files \| grep fit_state.toml \| xargs grep -l perturb_only_at_t0_params` → 0)                                                                                                                                                       |

The sibling counts are from
`git ls-files | xargs grep -h -o -E 'rw_sd *=|--rw-sd|perturb_only_at_t0'` in
each repository, excluding `_freeze/` and `results/`.

### 8.2 Code and documents that change with it

- The sim API: `EstimatedParam` loses three fields; 76 struct literals in 36
  files on main construct one (`git grep -c 'rw_sd_auto:' -- rust`), 80 on the
  branch. Each drops the fields; tests that set a step move it into
  `IF2Config.perturbations` or the PMMH proposal vector.
- `fit_state.toml`: `[rw_sd]` disappears; `initial_value_params`. The 16
  committed `camdl-vignettes` states that carry `[rw_sd]` still load.
- User documents
  (`git grep -c -E 'rw_sd|rw-sd|perturb_only_at_t0' -- docs ':!docs/dev'`):
  `camdl-inference-spec.md` (35), `inference.md` (16), `camdl-run-spec.md` (15),
  `camdl-language-spec.md` (6), `methods/cooling.md` (6), `fit-toml.md` (4), and
  `docs/agents.md`. `inference.md:567` ("rw_sd is on the natural scale") is
  replaced, not amended. `camdl-language-spec.md` is edited by hand and never
  run through `dprint`, which breaks its doctests (CLAUDE.md, "Markdown").
- The `camdl profile` help text and examples (`args/mod.rs:1669-1689`).

### 8.3 Migration text, for `docs/fit-toml.md` and the release notes

> **Step sizes moved from `[estimate]` to `[method]`, in z.** `rw_sd` set two
> different things — IF2's perturbation and one fifth of PMMH's and MH's initial
> proposal — on the natural scale of the parameter. Each is now a table of the
> method that uses it, in z, the unconstrained coordinate the method moves in:
>
> ```toml
> [method.perturbation] # algorithm = "if2"
> beta = 0.1
> I0 = { sd = 0.2, initial_value = true } # was perturb_only_at_t0 = true
>
> [method.proposal] # algorithm = "pmmh" or "mh"
> beta = 0.5
> ```
>
> A parameter you leave out takes a default step chosen from its domain, so most
> configs can delete their `rw_sd` lines. A config that still carries `rw_sd` or
> `perturb_only_at_t0` is refused, and the message prints the value in z that
> reproduces your old step at the start. `pgas` and the `nl-*` optimizers never
> read `rw_sd`: delete it. `profile --rw-sd` is replaced by `--estimate` (which
> parameters) and `--perturbation NAME=SD` (their steps, in z). Every fit
> re-keys at this release; stored fits whose `fit_state.toml` lists
> `perturb_only_at_t0_params` are refused and must be run again.

## 9. Before and after, one problem fitted by IF2 and by PMMH

The SIR recovery case (`tests/recovery/cases/sir/`), estimating β, γ and the
initial infected count `I0`. `beta : rate in [0.05, 1.0]` and
`gamma : rate in [0.01, 0.5]` take `LogInterval`; `I0 : count` boxed to [1, 100]
in `[estimate]` takes `LogInterval` too.

Before, IF2:

```toml
[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "data/weekly_cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], start = 0.25, rw_sd = 0.02 }
gamma = { bounds = [0.01, 0.5], start = 0.10, rw_sd = 0.01 }
I0 = { bounds = [1, 100], start = 10, rw_sd = 2.0, perturb_only_at_t0 = true }

[fixed]
N0 = 10000
rho = 0.6
k = 10.0

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 150
cooling = 0.7
```

Before, PMMH: the same file with `perturb_only_at_t0 = true` deleted (PMMH
refuses it, §3.2) and

```toml
[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 5000
```

The two `[estimate]` tables differ, so today the two files have different
fit-level hashes: the store records two problems.

After, both files share these tables verbatim:

```toml
[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "data/weekly_cases.tsv"

[estimate]
beta = { bounds = [0.05, 1.0], start = 0.25 }
gamma = { bounds = [0.01, 0.5], start = 0.10 }
I0 = { bounds = [1, 100], start = 10 }

[fixed]
N0 = 10000
rho = 0.6
k = 10.0
```

and differ only in `[method]`. IF2:

```toml
[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 150
cooling = 0.7

[method.perturbation]
beta = 0.107
gamma = 0.106
I0 = { sd = 0.174, initial_value = true }
```

PMMH, carrying the old natural-scale proposal at the start (5 × the IF2 values
of §6.3):

```toml
[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 5000

[method.proposal]
beta = 0.537
gamma = 0.528
I0 = 0.869
```

Their fit-level hashes are equal; their method-level hashes differ. With the
default steps, the IF2 file's table shrinks to `I0 = { initial_value = true }`
and the PMMH file needs no `[method.proposal]` at all.

## 10. Sequencing: in place of the transforms series (ii), before that branch merges

The support-keyed transforms series (i) is implemented on
`feat/support-transforms` (_branch_ commits `a1292b18` through `63dcafd6`,
including the §12.6 benchmark note
`docs/dev/notes/2026-10-29-support-keyed-transforms-benchmark.md`). Its series
(ii), which would read an explicit `rw_sd` in z (its §11 commits 11–12), has not
been written. This series takes its place on the same branch, so the release
carries one change of meaning instead of two. Every fit re-keys at that release
already, because the fit level folds the engine version, which includes the git
hash (transforms §13, verified there at `cas.rs:501-521` and
`version.rs:12-15`); this series adds no re-key of its own to schedule. No
`ir/VERSION` bump and no golden change is involved: the change is in the fit
config and the CLI, not the IR.

Commits, each green under `make test`:

1. Delete `FitState.rw_sd`, `runner::auto_rw_sd` and the NLopt copy (§3.5):
   written, never read. Dead code, its own commit.
2. Add `ZSd`, `PerturbAt`, `Perturbation` and `IF2Config.perturbations`; remove
   the three fields from `EstimatedParam`; the IF2 read sites move (§4.2). The
   CLI fills the vectors from today's `rw_sd` values, converted by the branch's
   `step_from_natural`, so results are byte-identical: an IF2 fit and a PMMH fit
   of `tests/recovery/cases/sir` at a fixed seed produce identical
   `fit_state.toml` and `draws.tsv` before and after (identical `run_id`s are
   not expected, since the engine hash moves). `if2.rs` is a high-risk file;
   this commit edits only its read sites.
3. The `[method]` tables, `resolve_steps`, the identity change (§5.1), the
   removal of `rw_sd` and `perturb_only_at_t0` from `EstimateSpecV2`, the
   formula-only refusal (§6.1), and the `ic_free`, convergence-check and
   `fit_state.toml` readers (§6.4). Inline test configs move to the new tables
   in this commit. The §5.3 tests.
4. `load_with_model` and the converted values in the refusal (§6.2, §6.3), with
   a test per row of §6.3 and per unavailable case.
5. `profile` reads `[method.perturbation]` from an IF2 `--fit`, and its base and
   method levels change (§5.2).
6. Documents (§8.2), the migration text (§8.3), and the two amended proposals'
   status lines (§12).

The `profile` flags of §7 land in parameter-roles Stage 3, which already
replaces `--rw-sd` as a selector. The `FilterSettings` split (§5.5) is a later
stage, after the support-transforms branch merges.

**With parameter-roles Stage 2.** Stage 2 is not implemented on main (its Stage
1 is: `2e80a07c`, `beec9c41`). Stage 2 is where `ResolvedProblem` is introduced,
so the §5.4 split lands there: Stage 2 builds `ResolvedProblem` without starts
or method settings, and `ExecutionPlan` with a `MethodPlan` whose `If2` and
`Pmmh`/`Mh` arms hold the resolved steps of this series. If this series lands
first, as recommended, Stage 2 wraps `resolve_steps`; if Stage 2 lands first, it
builds `AlgorithmSettings` nowhere and this series fills the two plan arms in
commit 3. The `SettingForNonEstimated` refusal is Stage 2's and is used here
unchanged. If the maintainer does not confirm D12, Stage 2 follows the roles
proposal as accepted, and `ResolvedSteps` takes the place of its
`AlgorithmSettings` field only.

## 11. Validation

- `ZSd::new` refuses NaN, ±∞, 0 and negatives; deserialization names the value.
- `PerturbationSpec` reads `beta = 0.1`, `beta = { sd = 0.1 }` and
  `beta = { initial_value = true }`, and refuses `beta = { sd = 0.1, lag = 2 }`
  and `beta = "auto"`, naming both accepted forms.
- `resolve_steps`, per rule of §4.4: a key not estimated; a written value; an
  omitted name on each map family; the identity map with nothing written; an
  IVP; a simplex member.
- `[method.proposal]` under `if2` and `[method.perturbation]` under `pmmh`,
  `pgas` and `nl-sbplx` are refused with the §4.3 messages.
- The legacy refusal: one test per row of §6.3; the three values of §6.3 for the
  §9 config to three significant figures; each unavailable case of §6.2; and a
  check that `simulate --fit` refuses too.
- `FitState::load` refuses `perturb_only_at_t0_params` and loads a file with
  `[rw_sd]`.
- The identity tests of §5.3, mutation-checked as stated there.
- The byte-identity check of commit 2.

## 12. Amendments to other proposals

### 12.1 Support-keyed transforms (`2026-10-29-support-keyed-transforms.md`)

- **D9 is superseded.** An explicit step is not read in a new unit under its old
  name; it moves to the tables of this proposal. Series (ii) (§11 commits 11–12)
  is replaced by §10 here.
- §2 "Step": "In the fit config it is `rw_sd`" becomes "In the fit config it is
  `[method.perturbation]` (IF2) or `[method.proposal]` (PMMH, MH)".
- §9.1 and §9.2: "an explicit `rw_sd` is converted to z once … and read in z
  directly from series (ii) on", and "five times an explicit `rw_sd`", become
  the written table value, read in z. The factor of five goes (D5 here). `rw_sd`
  "stays IF2's, PMMH's and MH's knob" becomes "IF2 and PMMH/MH each have their
  own".
- §13: the series (ii) conversion bullets are replaced by §6.2 here.

### 12.2 Workflow-first fit config (`2026-09-08-workflow-first-fit-config.md`)

- §5 "Kept as they are" and decision 12 say `perturb_only_at_t0` and `rw_sd` are
  inert for a non-IF2 method so that an IF2 file and a PGAS file share a
  fit-level hash. Both claims are false (§3.7 here); the relocation that
  proposal deferred as a follow-up is this proposal.

### 12.3 Parameter roles (`2026-10-26-parameter-roles.md`)

- §3.2 item 8 is false for `rw_sd` and stale for `perturb_only_at_t0` (§3.4
  here). It is re-checked and rewritten in that document: PMMH reads `rw_sd` as
  its initial proposal; PMMH refuses `perturb_only_at_t0` at load.
- §4.1: "`--rw-sd`, `[estimate].rw_sd`, `perturb_only_at_t0` … do not declare a
  role" becomes "`--perturbation`, `--initial-value`, `[method.perturbation]`
  and `[method.proposal]` do not declare a role".
- §4.2's last table row becomes "`--perturbation`, `--initial-value`, or a
  `[method.perturbation]`/`[method.proposal]` key, for a non-estimated name |
  refused".
- §6.3: `If2ParamSettings`, `RwSdFrom` and `AlgorithmSettings` are deleted;
  `ResolvedProblem` loses `starts` and `algorithm`, and `ExecutionPlan` and
  `MethodPlan` are added (§5.4 here; pending, D12). The refusals of
  `AlgorithmSettings::resolve` become `StepError` and `SettingForNonEstimated`.
- §6.6: each command builds one `ResolvedProblem` per fit segment and one
  `ExecutionPlan` per method and seed (pending, D12).
- §11.2 Stage 2 introduces `ResolvedProblem` and `ExecutionPlan` in the split
  form (pending, D12).
- §6.4 "IF2 settings": the paragraph is replaced by §4.2 and §4.4 here;
  `EstimatedParam::from_parts` takes no settings argument.
- §7.2: the record's `per_param` entry becomes
  `"perturbation": {"gamma": {"sd": 0.2, "from": "default", "initial_value": false}}`
  under IF2 and `"proposal": {…}` under PMMH and MH.
- §10.3 `profile`: the method level hashes the resolved perturbations, and the
  base level hashes the problem digest in place of the file's bytes (§5.2 here).
- §11.2 Stage 3: "`--rw-sd` optional and IF2-only" becomes "`--rw-sd` removed;
  `--perturbation` and `--initial-value`" (§7 here); its test "`--sweep k` +
  `--rw-sd beta=0.02`" becomes "`--sweep k` + `--perturbation beta=0.1`".
- D5: `--rw-sd` is refused in every form, not only `auto`.
- §13: "The `rw_sd` scale convention" and "PMMH's own proposal tuning (replacing
  `rw_sd × 5`)" are resolved by this proposal and leave the list.

## 13. Decisions

**D1 — A new name per algorithm, not a new unit for `rw_sd`.** Maintainer
decision (2026-10-29), recorded here. Reading `rw_sd` in z would run every
hand-tuned config with a different step and no message.

**D2 — The tables are fields of their algorithm's variant.** Maintainer decision
on location (2026-10-29); the variant-field form is chosen here. A top-level
`[method.perturbation]` checked against the algorithm at validation would also
work, but makes "an IF2 setting on a PMMH run" a state the type admits and a
rule must catch. As a variant field it is an unknown key.

**D3 — The unit is z, typed `ZSd`; for a simplex member, its log-ratio
coordinate.** z is the coordinate both kernels act in, the unit pomp's `mif2`
uses (§3.6), and the unit of the D8 defaults, so a written value and a default
are comparable. The type makes a non-positive or non-finite value
unrepresentable after load. The launch table prints the natural-scale equivalent
at the reference point, which is what a user tuning by eye needs.

**D4 — The IVP flag is `initial_value = true` inside the perturbation entry.** A
separate list (`[method] initial_value = ["I0"]`) was considered: it matches the
`--initial-value` flag's shape. It was not chosen because the flag is a property
of one parameter's perturbation, as pomp's `ivp()` is part of that parameter's
`rw.sd` entry, and one entry per parameter keeps a parameter's IF2 settings in
one place. `I0 = { initial_value = true }` takes the default SD.

**D5 — `[method.proposal]` is the proposal SD itself, with no factor of five,
and defaults to `default_mh_step`.** That PMMH and MH get their own initial
proposal SD defaulting to `default_mh_step`, and that `rw_sd`'s successor is
IF2's alone, is the maintainer's decision (2026-10-29); the absence of the
factor is decided here. The factor existed because one number served both
kernels, and an MH proposal should move further than one IF2 perturbation
(`cli/src/fit/pmmh.rs:1045-1048`). With separate names it hides a multiplier in
the unit of a value the user writes. A converted legacy value prints the
multiplied number (§6.3).

**D6 — PGAS gets no per-parameter table.** Maintainer decision (2026-10-29):
PGAS's MH-within-Gibbs reads no per-parameter step. Its default update is NUTS,
which has no per-parameter proposal, and its MH-within-Gibbs fallback starts at
1.0 in z and adapts by Robbins–Monro (_branch_ `pgas.rs:3872-3886`). No
measurement shows that a per-parameter start would help, and a new sampler
option lands only with one (CLAUDE.md, "A new inference feature lands with its
measurement"). `[method.proposal]` under `pgas` is refused with the §4.3
message.

**D7 — The method level hashes the resolved steps; a written default keys the
same as an omitted name.** The alternative, hashing the written table, would
give two keys to one run, contrary to the `starts` precedent (§5.1). The
resolved value depends on the map, which depends on the model and the box; those
are in the fit level, so the method level's hash never stands in for them.

`start` stays in `[estimate]` and in the fit level. A start does not change the
statistical problem, so in principle it is run identity. It is kept where it is
because the parameter-roles proposal owns starts (its §5.3, `--starts` or
`[estimate].start`) and moving its hash is a change to that design, not to step
sizes.

**D8 — Every reader refuses the old keys; values are converted at the start,
then the model value, then the box midpoint.** The conversion at the start is
the maintainer's specification and reproduces main's PMMH initial proposal
exactly (§6.2). Refusing in `simulate --fit` and `pfilter --fit`, which take no
step, keeps one rule: a file with the old keys is not a valid config anywhere.

**D9 — `profile` reads `[method.perturbation]` from an IF2 `--fit` config, and
nothing else from `[method]`.** Profile runs IF2, and a config's IF2
perturbations are settings of the same problem's IF2 run; today profile already
reads them from `[estimate]` (`profile.rs:878-891`), so not reading them would
silently perturb an IVP at every observation. Profile's particles, iterations
and cooling stay its own flags, as today. A PMMH config's `[method.proposal]` is
not read, because profile does not run PMMH.

**D10 — `fit_state.toml` drops `[rw_sd]` and renames `perturb_only_at_t0_params`
to `initial_value_params`; a file with the old name is refused.** Keeping the
old stored name would leave the retired vocabulary in every new record. Reading
old files with the field empty would misreport their convergence (§6.4). Refusal
matches the `refuse_one_based_layout` precedent and the release's re-key.

**D11 — This series replaces transforms series (ii) on its branch; the profile
flags land in roles Stage 3.** One release changes the meaning of steps once.
`profile --rw-sd` cannot be removed before `profile --estimate` exists, which is
Stage 3's; until then it keeps its natural unit under its own name (§7).

**D12 — The resolved problem and the execution plan are separate types. Pending
the maintainer's confirmation.** `ResolvedProblem` holds only what every method
and seed of a fit segment shares, and `ExecutionPlan` holds the method, the
starts, the chain count and the seed (§5.4). The roles proposal's single
`ResolvedProblem` with `starts` and `algorithm` inside cannot be shared across
the methods of one problem, which is the sharing this proposal's identity change
exists to allow. The cost is one more type and an `Arc` per segment. The
alternative, keeping the accepted type and adding `ResolvedSteps` as its
`algorithm` field, works for this series and leaves the conflation in the
resolver; it is the fallback if D12 is not confirmed (§10). New names avoid the
existing `fit::methods::InferenceMethod` (the algorithm/backend registry,
`fit/methods.rs:50`) and the two `InitSource` enums
(`cli/src/fit/chain_starts.rs:45`, `sim/src/error.rs:352`).

**D13 — `FilterSettings` is split out of `IF2Config` in a later stage.** The
split removes the placeholder cooling arguments every non-IF2 method passes
(§5.5). It is not part of this series because it edits `if2.rs` while the
support-transforms branch also does, and because it changes no result; it lands
after that branch merges, with its own byte-identity check.

## 14. Out of scope, each with its reason

- **Command-line step overrides for `fit run`.** `fit run` has no override for
  any per-algorithm setting today (`args/mod.rs:668-750`: `--starts`,
  `--decibans-thresh` and run controls only). Adding `--perturbation` there is a
  general decision about `[method]` overrides, not part of moving the steps.
- **Moving `start`'s hash to the method level** (D7): a change to the
  parameter-roles design.
- **A proposal on a correlated (dense) initial covariance for PMMH.** The table
  is diagonal, as the sampler's initial proposal is today
  (`sim/src/inference/pmmh.rs:664`); Haario adaptation learns the correlation.
- **IVP lags (pomp's `ivp(s, lag)`).** No model in this repository or the
  sibling repositories needs one; `initial_value` leaves room for a `lag` key.
