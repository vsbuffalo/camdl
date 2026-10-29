---
date: 2026-10-29
status: proposed — built on the maintainer decisions of 2026-10-29 listed in §1; D1–D22 (§14) are decided, nothing is pending; the IF2 default constants are pending the extended benchmark (maintainer decision T1: extend before choosing; §4.4)
  area: fit config / run identity / resolved problem and execution plan / IF2, PMMH and MH step sizes / profile CLI
  issues: gh#1001
  supersedes: 2026-10-29-support-keyed-transforms.md D9 and its commit series (ii) (its §11 commits 11–12)
  amends:
    - 2026-10-29-support-keyed-transforms.md §2, §4.1, §9.1, §9.2, §9.5, §10, §11, §13 (§13.1 here)
    - 2026-10-26-parameter-roles.md §3.2 item 8, §4.1, §4.2, §5.3, §6.3, §6.4, §6.6, §7.1, §7.2, §7.3, §10.1, §10.2, §10.3, §11.2 Stages 2 and 3, D5, §13 (§13.3 here)
    - 2026-09-08-workflow-first-fit-config.md §5 and decision 12; its §9 follow-up is done (§13.2 here)
---

# A step size belongs to the method that takes it: `[estimate]` states the problem, `[method]` states how each algorithm moves through it

Code references are against `origin/main` at `a9a89178` unless marked _branch_,
which means `origin/feat/support-transforms` at `193525d9`, the implementation
of the support-keyed transforms proposal. That proposal is cited from the
branch; its section numbers are the same as in the revision on main. A claim
marked _verified_ was read at the cited line or produced by the command shown.
_Inferred_ means reasoned from the code and not executed. No fit was run for
this document.

## 1. The decisions this builds on

The maintainer decided, on 2026-10-29:

1. `rw_sd` is renamed rather than given a new unit. A config that uses it is
   refused, naming the new key.
2. Algorithm-specific parameter settings move out of `[estimate]` into
   `[method]`, with separate names for IF2's perturbations and PMMH's proposal
   scales; the statistical problem's identity is kept apart from the identity of
   one run of it.
3. PMMH and MH get their own initial proposal SD in `[method]`, defaulting to
   the support-keyed transforms' `default_mh_step`. `rw_sd`'s successor is IF2's
   alone. PGAS's MH-within-Gibbs reads no per-parameter step.
4. The parameter-roles `ResolvedProblem` is amended before its Stage 2: it is
   the fully resolved statistical problem, independent of algorithm, starts and
   seed, and the algorithm-specific configuration lives in a separate
   `ExecutionPlan` (§6).
5. Consequently `[estimate].start` is execution, hashed at the method level, not
   the fit level (§5.2).
6. An unsupported old `fit_state.toml` is refused as an input to a new
   computation and read with one notice by read-only commands (§7.5).
7. The release waits for parameter-roles Stage 3, so `profile` has one z-scale
   step interface and never two units at once (§8, §11).
8. `[method.proposal]` values are exactly the proposal SD the user writes; no
   factor of five or any other hidden multiplier remains (§3.8, §7.4).
9. The flag is `initial_state` (`--initial-state` on `profile`). The model
   property that a parameter enters only the initial state is kept distinct from
   IF2's perturbation timing (§4.6).
10. A natural-scale step is converted at the resolved start value, everywhere:
    migration errors, launch summaries and `profile` (§7.3).
11. An incorrect initial-state schedule is refused when the model reliably
    identifies initial-state-only parameters (§4.6).
12. The resolved perturbation and proposal scales are part of the
    `ExecutionPlan` and of saved run provenance (§5.6).

## 2. One key, `rw_sd`, sets two algorithms' steps in a unit about to change, and keeps two fit files of one problem apart

An estimated parameter θ is moved by a method in an unconstrained coordinate z,
with θ = g(z) (support-keyed transforms, §2). Two methods take a per-parameter
step there. IF2 (iterated filtering, Ionides et al. 2015) adds a random-walk
**perturbation** to each particle's θ before each observation. PMMH (particle
marginal Metropolis–Hastings) and MH (Metropolis–Hastings on the ODE likelihood)
draw a **proposal** around the current θ. The perturbation is the kernel of a
cooled random walk that defines what IF2 maximizes. The proposal is the starting
shape of an adapted Metropolis kernel whose stationary distribution does not
depend on it.

Today both are set by one key, `[estimate].rw_sd`, a step on the natural scale
of θ (§3.1). The support-keyed transforms fix IF2's step in z with a default per
map family (its D8); its D9 made an explicit `rw_sd` mean z. D9 is on hold,
because reading the same key in a new unit would silently change every
hand-tuned config. This proposal replaces it.

Because `rw_sd` and `perturb_only_at_t0` sit in `[estimate]`, which the
problem's identity hashes, and because every method but IF2 refuses
`perturb_only_at_t0`, an IF2 file and a PGAS file for one problem with an
initial-state parameter cannot share a fit segment (§3.7). That is the defect
that makes this more than a rename.

After the change, `[estimate]` holds only what is true of a parameter under any
method (bounds, start, prior), and each algorithm's step is a table of its
`[method]`, in z, typed:

```toml
# IF2                                       # PMMH or MH
[method.perturbation]                       [method.proposal]
beta = 0.107                                beta = 0.537
I0   = { sd = 0.174, initial_state = true }
```

A name left out takes the D8 default. The old keys are refused with the new
spelling and the converted value (§7). The steps and the start leave the
fit-level digest and enter the method-level digest (§5). `profile`'s `--rw-sd`,
which both chose the estimated set and gave steps, is replaced by
`--perturbation` and `--initial-state`, with the estimated set declared by
`--estimate` (§8).

## 3. What the code does today

### 3.1 `rw_sd` is read by IF2 and by PMMH and MH, in natural units, and by nothing else

| reader                 | what it does with `rw_sd`                                                                                                                                                                                           | where (verified)                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| IF2                    | perturbation SD at particle i is `transformed_sd(rw_sd, θᵢ)`, the natural step over \|g′\| at that particle's value; the per-parameter diagnostic and the per-iteration `effective_rw_sd` use the same value        | `sim/src/inference/if2.rs:428`, `:533`, `:639`, `:722`                                  |
| `profile` cell summary | `cooling_final` is the mean over parameters of the last iteration's `effective_rw_sd`                                                                                                                               | `cli/src/profile.rs:1633-1650`                                                          |
| IF2, simplex member    | taken as-is on the log-ratio scale when written; the automatic value is computed in natural units and used on the log-ratio scale                                                                                   | `cli/src/fit/runner.rs:647-692`                                                         |
| PMMH, MH               | initial proposal SD is `5 × transformed_sd(rw_sd, θ₀)`; the sampler uses it as the proposal's shape until `adapt_start` (default 300), and as the whole proposal when `adapt = false`                               | `cli/src/fit/pmmh.rs:183`, `:1045-1053`, `:575`; `sim/src/inference/pmmh.rs:654-667`    |
| PGAS (MH-within-Gibbs) | not read: the initial scale is a tenth of the z-range of the box                                                                                                                                                    | `sim/src/inference/pgas.rs:3955-3965`                                                   |
| `nl-*`, fit run        | not read; copied into `fit_state.toml`'s `[rw_sd]`                                                                                                                                                                  | `cli/src/fit/nlopt_stage.rs:262-299`                                                    |
| `pfilter`              | not read                                                                                                                                                                                                            | `grep -n "rw_sd\|perturb" rust/crates/cli/src/pfilter.rs` → no match                    |
| `survey`               | passed into `EstimatedParam` and never used: survey evaluates a landscape and runs no IF2                                                                                                                           | `cli/src/survey.rs:950-956`; `grep -n run_if2 rust/crates/cli/src/survey.rs` → no match |
| `profile`              | `--rw-sd` is required; its keys are the estimated set (`auto` estimates every non-focal, non-fixed parameter); a value overrides `[estimate].rw_sd` from `--fit`; hashed under every `--algorithm`, `nl-*` included | `cli/src/profile.rs:294-302`, `:847-857`, `:870-891`, `:1293-1310`                      |

On the branch:

- IF2 reads `spec.rw_sd`, the step already in z (_branch_ `if2.rs:424`, `:523`,
  `:624`, `:705`).
- An explicit natural `rw_sd` is converted once at the domain's reference point
  by `Transform::step_from_natural` (_branch_ `types.rs:172`, called at
  `runner.rs:1463`). The reference point is the box midpoint for the interval
  maps and θ₀ otherwise (`types.rs:163`, `reference_z`). An omitted one takes
  `default_if2_step` (`runner.rs:1464`).
- PMMH and MH start at `default_mh_step`, `MH_DEFAULT_STEP` = 0.25 in z
  (_branch_ `types.rs:83`, commit `90d593a0`), when `rw_sd` is omitted, and at
  five times the converted `rw_sd` otherwise (`types.rs:441-447`,
  `mh_initial_step`; `pmmh.rs:1049-1052`).
- PGAS's MH-within-Gibbs reads no `rw_sd` and starts at `default_mh_step`, else
  1.0 for the identity map (_branch_ `pgas.rs:3872-3886`, commit `6bd627d1`). An
  earlier branch commit had made it read `rw_sd`; a description saying it does
  describes that commit.

### 3.2 `perturb_only_at_t0` is IF2's schedule, and seven places read it

| reader                                                                           | where (verified)                                                                                         |
| -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| IF2's per-observation perturbation skips a flagged parameter                     | `if2.rs:531`                                                                                             |
| every other algorithm refuses the flag at load                                   | `config_v2.rs:2438-2457`                                                                                 |
| a simplex member may not carry it                                                | `config_v2.rs:3248-3256`                                                                                 |
| the `ic_free` precondition accepts an initial-state parameter as t = 0 spread    | `runner.rs:511-556` (the test at `:540-543`)                                                             |
| the `ic_free` startup line names the initial-state parameters as a spread source | `fit/mod.rs:809-833`                                                                                     |
| IF2's convergence check exempts initial-state parameters from chain agreement    | `gating.rs:190`                                                                                          |
| `fit_state.toml` lists them; `fit summary` applies the exemption                 | `state.rs:62-68`; `fit_summary.rs:1597`, `:3071`; written by `fit/mod.rs:1489`, `nlopt_stage.rs:270-299` |

`profile` reads the flag only from a `--fit` config's `[estimate]`
(`profile.rs:890-891`). A profile with no `--fit` cannot mark an initial-state
parameter at all, so it perturbs it before every observation. The `camdl-book`
identifiability chapter does this: it profiles with `I0` and `R_init` estimated
through `--rw-sd` and no `--fit` (`guide/fitting/identifiability.qmd:402`,
`:485` at `camdl-book` `5e689ac`). That is a silent case today;
`--initial-state` (§8) is the fix.

### 3.3 The steps and the start are part of the problem identity

The fit level hashes the whole `Problem` minus `output_dir` and `provenance`
(`cas.rs:420-441`, `fit_config_blob_hash`). `Problem.estimate`
(`config_v2.rs:115`) serializes `rw_sd`, `perturb_only_at_t0` and `start` with
no skip (`config_v2.rs:573-626`). So two fit configs that differ only in an IF2
step, or only in a start, have different fit-level hashes: the store files them
as two statistical problems, in two fit segments.

`profile --fit` hashes the fit config's raw bytes into its base level, the level
meant to identify the profiled problem (`profile.rs:359-366`, the `fit_toml`
field of `ProfileBaseLevel`, `:134-147`). A change to any byte, `[method]` and
comments included, re-keys every profile cell of that config.

### 3.4 The parameter-roles finding 8 is false as written

`2026-10-26-parameter-roles.md` §3.2 item 8 says `perturb_only_at_t0` and
`rw_sd` in `[estimate]` "change nothing under PMMH". For `rw_sd` that is false:
PMMH builds its proposal from it (`cli/src/fit/pmmh.rs:183` → `:1049-1052` →
`PMMHConfig.proposal_sd` at `:575` → `rng.normal() * config.proposal_sd[i]` at
`sim/src/inference/pmmh.rs:661-667`), so changing it changes PMMH's draws, and
it moves the fit-level hash (§3.3). Verified by reading; no PMMH fit was run.
The same proposal's §6.4 says "PMMH reads `rw_sd` for its initial proposal", so
the document contradicts itself. For `perturb_only_at_t0` the finding is out of
date: PMMH refuses it at load (`config_v2.rs:2438-2457`). §13.3 lists the
correction.

### 3.5 `fit_state.toml`'s `[rw_sd]` is written and never read

IF2 writes a MAD-calibrated suggestion (`fit/mod.rs:1434-1439`,
`runner::auto_rw_sd`), the NLopt stages copy the input steps
(`nlopt_stage.rs:262-273`), and PMMH and PGAS write an empty table. Nothing
reads `FitState.rw_sd` (`grep -rn "\.rw_sd" rust/crates/cli/src` returns only
`EstimatedParam` and `EstimateSpecV2` accesses and the writers). Its values are
in natural units. `FitState` has no `deny_unknown_fields` (`state.rs:19-20`), so
dropping the field leaves old files loadable.

### 3.6 pomp keeps the two knobs apart; Stan has no per-parameter step

pomp's IF2 takes `rw.sd`, built with `rw_sd()`: "specification of the magnitude
of the random-walk perturbations that will be applied to some or all model
parameters … If parameter transformations have been supplied, then the
perturbations are applied on the transformed (estimation) scale" (`man/mif2.Rd`,
the `rw.sd` argument, pomp 6.5.0.0, GitHub master read 2026-10-29).
Initial-value parameters are marked inside it: "`ivp(s)` can be used in this
context as shorthand for `ifelse(time==time[1],s,0)`" (same entry). pomp's PMCMC
takes a separate argument, `proposal`, a function drawing θ′ given θ
(`R/pmcmc.R`, `@param proposal`; the loop calls `proposal(theta, …)` at line
371). Its usual constructor is `mvn_diag_rw(rw.sd)` (`man/proposals.Rd`): the SD
vector is again called `rw.sd`, but as an argument of the proposal constructor.
`R/pmcmc.R` applies no parameter transformation (`grep -n partrans R/pmcmc.R` →
no match), so pomp's PMCMC proposes on the natural scale, unlike its IF2.

The software paper makes the same split: Algorithm 3 is
`mif(P, start = θ0, Nmif = M, Np = J, rw.sd = σ1:p, ic.lag = L, …)` with "I ⊂
{1, …, p}, designating IVPs", and Algorithm 4 is
`pmcmc(P, start = θ0, Nmcmc = M, Np = J, proposal = q)` (King, Nguyen and
Ionides, "Statistical Inference for Partially Observed Markov Processes via the
R Package pomp", _J. Stat. Softw._ 69(12), 2016, doi:10.18637/jss.v069.i12,
§3.2). Its examples pass `rw.sd = c(r = 0.02, …)` to `mif` with
`transform = TRUE` (§4.3) and `proposal = mvn.diag.rw(c(r = 0.01, …))` to
`pmcmc` (§4.4).

Stan transforms every constrained parameter "to an unconstrained variable behind
the scenes" for sampling (Stan Reference Manual 2.40, "Constraint Transforms"),
and its HMC tuning — "discretization time ε (aka step size), metric M (aka mass
matrix), and number of steps taken L" ("MCMC Sampling", "HMC algorithm
parameters") — is global and adapted. Stan has no per-parameter step to name;
what it settles is that a sampler's tuning lives on the unconstrained scale,
which camdl's z is.

### 3.7 The motivating defect: an IF2 file and a PGAS file for one problem cannot share a fit segment

The workflow-first fit-config proposal kept `perturb_only_at_t0` and `rw_sd` in
`[estimate]` on the claim that "under one method per file they are inert for a
non-IF2 method and the loader says nothing, so that a problem's IF2 comparator
file and its PGAS file differ only in `[method]` and share a fit-level hash"
(`2026-09-08-workflow-first-fit-config.md:934-942`, and its decision 12). Any
method but IF2 refuses `perturb_only_at_t0` at load (`config_v2.rs:2438-2457`,
verified). So a model with an initial-state parameter has an IF2 file that
carries the flag and a PGAS file that must not; their `[estimate]` tables
differ, and they land in different fit segments although they pose one problem.

The red test the series starts from, in `cli/src/fit/cas.rs`'s test module, on
the §10 problem. Both files declare a prior for every estimated parameter
(`beta` and `gamma` from the model's `~`, `I0` in `[estimate]`), so PGAS's
`requires_priors` check (`config_v2.rs:1495`, enforced by
`validate_priors_present`, `:2790`) accepts the PGAS file:

```rust
#[test]
fn an_if2_file_and_a_pgas_file_of_one_problem_share_the_fit_level() {
    // Today: SIR_I0_IF2 has `I0 = { …, perturb_only_at_t0 = true }`;
    // SIR_I0_PGAS has `I0 = { … }`, because PGAS refuses the flag.
    // After: both have the same [estimate]; the IF2 file carries
    // `[method.perturbation] I0 = { initial_state = true }`.
    let if2 = FitConfig::from_toml_str(SIR_I0_IF2).unwrap();
    let pgas = FitConfig::from_toml_str(SIR_I0_PGAS).unwrap();
    assert_eq!(fit_level_hash_of(&if2.problem), fit_level_hash_of(&pgas.problem));
}
```

It is red today: the two `Problem`s serialize different `[estimate]` entries,
which the fit level hashes (inferred from `cas.rs:433-435` and
`config_v2.rs:573-626`; not run). It turns green in commit 5 of §11. Its
mutation check adds the resolved IF2 steps to the fit level (`FitDigest`), which
must turn it red.

### 3.8 Every place a backend multiplies, divides or replaces a user's scale

After this change, every remaining factor is a documented adaptation or a
recorded backend default.

| what                                                                                                                                          | `origin/main`                                                              | branch `193525d9`                                                                                                       | after this proposal                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| PMMH/MH initial proposal = 5 × the IF2 step                                                                                                   | `cli/src/fit/pmmh.rs:1049-1052` (5 × `transformed_sd(rw_sd, θ₀)`)          | `pmmh.rs:1049-1052` → `types.rs:441-447` (5 × the z step, or the default)                                               | **removed**: `[method.proposal]` is the proposal SD (D5)                                                                                            |
| IF2 step divided by \|g′\| at each particle's current value                                                                                   | `sim/src/inference/if2.rs:428`, `:533`, `:639`                             | removed (fixed step in z, `if2.rs:424`, `:523`)                                                                         | removed; the written z value is the kernel                                                                                                          |
| IF2 automatic step from the box: ln(hi/lo)/20 at the geometric midpoint, (hi − lo)/6, ±10⁶ for an infinite end                                | `fit/runner.rs:1499-1520` (`auto_rw_sd_from_value`)                        | replaced by `default_if2_step` (`types.rs:141-150`)                                                                     | the D8 default, printed with its natural equivalent at the start                                                                                    |
| a simplex member's natural-scale automatic value read as a log-ratio SD                                                                       | `runner.rs:681-689` (the member's `EstimatedParam.rw_sd`)                  | `runner.rs:686-689`, `:1516-1535` (`simplex_member_default_step`: the old natural formula, read on the log-ratio scale) | **removed**: one log-ratio constant, `SIMPLEX_LOG_RATIO_STEP` = 1/6, today's value for a member on [0, 1]; a written value is used as written (D20) |
| PGAS MH-within-Gibbs initial scale                                                                                                            | `sim/src/inference/pgas.rs:3955-3965` (a tenth of the z-range, floor 0.01) | `pgas.rs:3872-3886` (`default_mh_step`, else 1.0 for `Identity`)                                                        | no user scale; **exposed**: the resolved initial scale per parameter is in `PgasPlan` and the run record (§5.6)                                     |
| PGAS MH-within-Gibbs Robbins–Monro clamp on the log scale, [−20, 5]                                                                           | `pgas.rs:4486`                                                             | `pgas.rs:4405`                                                                                                          | an adaptation bound; documented in `fit-toml.md`                                                                                                    |
| PGAS NUTS initial step size 0.1 and identity mass, reset to 0.1 after the mass estimate                                                       | `pgas.rs:4097`, `:4415`                                                    | `pgas.rs:4019`, `:4336`                                                                                                 | not per-parameter; dual averaging adapts it; **exposed** as `PgasPlan.nuts_initial_step` and recorded                                               |
| PMMH/MH Haario shape (2.38²/d)·Cov after `adapt_start`; global scale λ adapted from step one                                                  | `sim/src/inference/pmmh.rs:402-406`, `:654-667`                            | the same lines                                                                                                          | adaptation, not reinterpretation; `fit-toml.md` says the written SD is the shape until `adapt_start` and the whole proposal when `adapt = false`    |
| IF2 cooling factor on every step                                                                                                              | `if2.rs` (`per_step_cooling`), `IF2Config.cooling_fraction`                | the same                                                                                                                | a declared schedule; unchanged                                                                                                                      |
| `fit_state.toml` `[rw_sd]` suggestion: MAD calibration, else 0.5 × the step                                                                   | `fit/mod.rs:1434-1439`                                                     | `fit/mod.rs:1432-1437`                                                                                                  | **removed** (§3.5)                                                                                                                                  |
| `profile`'s `cooling_final`: mean over parameters of each last-iteration SD (natural units on main, z on the branch, mixed across parameters) | `profile.rs:1633-1650`                                                     | `profile.rs` (same computation over `effective_rw_sd`)                                                                  | **replaced** by the last iteration's cooling factor, which is unit-free and the same for every parameter (D21)                                      |
| `profile` hashes `rw_sd` under `nl-*`, where no step is taken                                                                                 | `profile.rs:1292-1302`                                                     | `profile.rs:1292-1301`                                                                                                  | **removed** (§5.4)                                                                                                                                  |
| launch-table natural equivalent                                                                                                               | `runner.rs:1080` (at the start)                                            | `runner.rs:1169-1171` (at the box midpoint, `reference_z`)                                                              | at the resolved start everywhere (D19)                                                                                                              |
| PMMH preflight's one-step spread                                                                                                              | `cli/src/fit/pf_noise.rs:275-283` reads `proposal_sd` as given             | the same                                                                                                                | no factor; reads the resolved proposal                                                                                                              |

The commands that found these:

```sh
grep -rn "transformed_sd\|\* 5\.0\|5\.0 \*\|/ 20\.0\|/ 6\.0\|nuts_step_size = 0\.1\|2\.38\|clamp(-20" \
  rust/crates/cli/src rust/crates/sim/src
grep -rnoE "if2_config\.\w+" rust/crates/cli/src
```

Each was run on both refs, and every hit was read.

## 4. The types: `[estimate]` loses the steps, each algorithm variant gains its own table

### 4.1 The unit is a type

```rust
// rust/crates/sim/src/inference/types.rs

/// A standard deviation in z, the coordinate a method moves in (θ = g(z),
/// `Transform`; a simplex member's log-ratio coordinate). Finite and > 0 by
/// construction.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ZSd(f64);

#[derive(Debug, Clone, PartialEq)]
pub struct NotAZSd(pub f64);           // NaN, ±∞ or ≤ 0

impl ZSd {
    pub fn new(sd: f64) -> Result<Self, NotAZSd>;
    pub fn get(self) -> f64;
}
// Deserialize goes through `new`; the error names the value and the unit.

/// When IF2 perturbs a parameter: IF2's timing, distinct from the model
/// property `initial_state` that selects it (§4.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PerturbAt {
    /// At t = 0 and before every observation.
    EveryObservation,
    /// At t = 0 only: the timing for a parameter declared `initial_state`.
    TimeZeroOnly,
}

/// IF2's random-walk perturbation of one estimated parameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Perturbation {
    pub sd: ZSd,
    pub at: PerturbAt,
}

/// IF2's step for one entry of `if2_params`. A simplex member has none of its
/// own: its SD lives in its `SimplexGroup`, which perturbs it jointly at t = 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum If2Step {
    Single(Perturbation),
    SimplexMember,
}
```

### 4.2 The sim crate separates the parameter from the setting

`EstimatedParam` keeps what every method needs and loses the three IF2 and MH
fields, so an IF2 setting cannot reach PMMH through it:

```rust
// before (branch types.rs:364-392)          // after
pub struct EstimatedParam {                  pub struct EstimatedParam {
    pub name: String,                            pub name: String,
    pub index: usize,                            pub index: usize,
    pub initial: f64,                            pub initial: f64,
    pub rw_sd: f64,                              pub transform: Transform,
    pub transform: Transform,                }
    pub rw_sd_auto: bool,
    pub perturb_only_at_t0: bool,
}
// `natural_step` and `mh_initial_step` (branch types.rs:433-447) are deleted.
// `rw_sd_auto` (main types.rs:66) is provenance inside an engine type; it is
// deleted, and its job (written or default) is `StepFrom`, held by the CLI
// and recorded in run.json, never seen by an engine (I2).
```

IF2 takes its steps beside the parameters, as PMMH already takes
`PMMHConfig.proposal_sd: Vec<f64>` (`sim/src/inference/pmmh.rs:33`):

```rust
// rust/crates/sim/src/inference/if2.rs
pub struct IF2Config {
    // …existing fields…
    /// One per entry of `if2_params`, in the same order. `run_if2` returns
    /// `SimError::Config` on a length mismatch, or on `SimplexMember` for a
    /// parameter no group contains.
    pub steps: Vec<If2Step>,
}

pub struct SimplexGroup {
    pub indices: Vec<usize>,
    pub sds: Vec<ZSd>,           // was `rw_sds: Vec<f64>`, on the log-ratio scale
}

pub struct ParamIterDiag {
    // …
    pub effective_sd: f64,       // was `effective_rw_sd`; in z
}
```

Every reader of the removed fields, and what it reads instead:

| reader (main; branch line where it differs)                                             | after                                                                                 |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `if2.rs:428`, `:533` (branch `:424`, `:523`): the perturbation SD                       | `config.steps[pi]`'s `sd`, times `cooling_now`                                        |
| `if2.rs:531` (branch `:521`): the per-observation skip                                  | `config.steps[pi]` is `Single` with `at == TimeZeroOnly`, or `SimplexMember`          |
| `if2.rs:639`, `:722` (branch `:624`, `:705`): diagnostics                               | the same `sd`; `effective_rw_sd` becomes `effective_sd`                               |
| `profile.rs:1633-1650`, `cooling_final`                                                 | `effective_sd`; the stored profile diagnostic is in z from this release               |
| `cli/src/fit/pmmh.rs:1049-1052`, `build_proposal_sd`                                    | `ExecutionPlan`'s resolved proposal (§4.4, §6)                                        |
| `runner.rs:540-543`, the `ic_free` precondition; `fit/mod.rs:809-833`, its startup line | any resolved IF2 step is `TimeZeroOnly`                                               |
| `gating.rs:190`; `fit_summary.rs:1597`, `:3071`                                         | `fit_state.toml`'s initial-state list (§7.5)                                          |
| `fit/mod.rs:1434-1439`, `:1489`; `nlopt_stage.rs:262-299`                               | the initial-state list from the resolved steps; `[rw_sd]` is deleted (§3.5)           |
| `runner.rs:1059-1093` (branch `:1135-1182`), the launch table                           | the resolved steps, with `written`/`default` shown                                    |
| `runner.rs:369-378`, `:1445-1475` (branch `:1489`), `refuse_degenerate_auto_rw_sd`      | `StepError::NoDefault` (§4.4)                                                         |
| `runner.rs:647-692` (branch `:659-693`), `resolve_simplex_groups`                       | the resolved simplex SDs                                                              |
| `survey.rs:950-956`, `:1146`                                                            | nothing: survey builds parameters only                                                |
| the 76 test literals on main (80 on the branch) that set the three fields               | dropped; a test that sets a step moves it to `IF2Config.steps` or the proposal vector |

The perturbation draw itself (`if2.rs:429-430`, `:534-535`) is unchanged.
`PMMHConfig.proposal_sd` stays `Vec<f64>`, so `sim/src/inference/pmmh.rs` is not
edited; the CLI unwraps `ZSd` at the boundary.

### 4.3 The fit config

```rust
// rust/crates/cli/src/fit/config_v2.rs

/// A parameter fact that holds under every method.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EstimateSpecV2 {
    pub bounds: Option<(f64, f64)>,
    pub prior: Option<EstimatePriorSpec>,
    /// Where chains begin under `starts = "single"`, and chain 1 under
    /// `"uniform"`. Execution: hashed at the method level (§5.2).
    pub start: Option<f64>,
}
// `transform` is deleted by the transforms series (its D2); `rw_sd` and
// `perturb_only_at_t0` by this one. The legacy detector (§7) runs before
// the typed parse, so neither key reaches `deny_unknown_fields`.

/// One `[method.perturbation]` entry. On the wire a bare number is `{ sd = n }`;
/// a table takes `sd` and `initial_state` and nothing else; an empty table
/// is refused, because it says nothing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerturbationSpec {
    /// `None`: the map's D8 default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sd: Option<ZSd>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub initial_state: bool,
}

pub enum Algorithm {
    IF2 {
        // …existing fields…
        /// Per estimated parameter; a name left out takes the D8 default and
        /// is perturbed before every observation.
        #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
        perturbation: IndexMap<String, PerturbationSpec>,
    },
    PMMH {
        // …existing fields…
        /// Initial proposal SD per estimated parameter; a name left out takes
        /// `default_mh_step`.
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

serde cannot deny unknown fields on the internally tagged `Algorithm`, so a
table in the wrong variant is caught by the existing post-parse rule,
`validate_method_keys` (`config_v2.rs:2181-2272`). Each table is a field of its
own variant, so that rule sees it as unknown under any other algorithm. Two
entries are added to its retired-key list so the message names the right table:

- `proposal` under `if2`, `pgas`, `pfilter` or `nl-*`: "`[method.proposal]` is
  the initial proposal SD of `pmmh` and `mh`. IF2's per-parameter setting is
  `[method.perturbation]`; `pgas`, `pfilter` and the `nl-*` optimizers have
  none."
- `perturbation` under any method but `if2`: the same message, roles swapped.

The rule builds its allowed set from the serialized method, and the tables skip
serialization when empty, so both names are added to `known` for their variants,
as `binomial` and `ancestor_sampling` are (`config_v2.rs:2207-2211`).

### 4.4 Resolution works from the domains, once, beside the starts

```rust
// rust/crates/cli/src/fit/steps.rs (new)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepFrom { Written, Default }       // launch table and run.json inputs; never hashed

pub struct ResolvedPerturbation { pub step: If2Step, pub from: StepFrom }
pub struct ResolvedProposal { pub sd: ZSd, pub from: StepFrom }

/// Every estimated parameter's per-method step, in `[estimate]` order.
pub enum ResolvedSteps {
    If2 {
        params: IndexMap<String, ResolvedPerturbation>,
        /// One per `[[simplex_groups]]` entry, members in declaration order.
        simplex: Vec<IndexMap<String, ResolvedProposal>>,
    },
    Proposal(IndexMap<String, ResolvedProposal>),   // pmmh and mh
    None,                                            // pgas, pfilter, nl-*
}

#[derive(Debug, Clone, PartialEq)]
pub enum StepError {
    NotEstimated { table: &'static str, name: String },
    NoDefault { table: &'static str, name: String, kind: &'static str },
    InitialStateOnSimplexMember { name: String, group: usize },
    EmptyEntry { name: String },
}

/// The domains are the transforms series' effective domains (model + box),
/// computed before any parameter value or start is drawn.
pub fn resolve_steps(
    algorithm: &Algorithm,
    domains: &IndexMap<String, Transform>,
    simplex_groups: &[config_v2::SimplexGroup],
) -> Result<ResolvedSteps, StepError>;

impl ResolvedSteps {
    /// What the method level hashes (§5.3).
    pub fn identity(&self) -> serde_json::Value;
    pub fn if2_steps(&self, order: &[EstimatedParam]) -> Option<Vec<If2Step>>;
    pub fn simplex_sds(&self) -> Option<Vec<Vec<ZSd>>>;
    pub fn proposal_sd(&self, order: &[EstimatedParam]) -> Option<Vec<f64>>;
}
```

`fit run` calls `resolve_steps` once, next to `Method::resolve_starts`
(`fit/mod.rs:526`), after `resolve_roles`. The one value is passed both to the
identity (`FitStageCtx.steps`, §5.3) and to the runner, which today assembles
its parameters in `FitRunConfig::build` (`fit/mod.rs:1349`, `:1755`); no
consumer resolves the steps a second time. A `--sweep` over held parameters
(roles §4.2 case 2) does not change any domain, so the value is shared across
its points.

The rules, in order:

1. A key not in `[estimate]` is `NotEstimated`: the parameter-roles
   `RoleError::SettingForNonEstimated` for these tables.
2. An empty entry, `beta = {}`, is `EmptyEntry`.
3. A written `sd` is used as written.
4. An omitted name, or an entry with no `sd`, takes `default_if2_step()` under
   IF2 and `default_mh_step()` under PMMH and MH (_branch_ `types.rs:141-160`).
   The MH constant is 0.25 in z, for PMMH, MH and PGAS's MH-within-Gibbs
   (`types.rs:83`). The IF2 constants in the code are 2/3 on `Interval` and 0.2
   on `LogInterval` and the half-bounded maps (`types.rs:66-74`). They are
   pending the extended benchmark: the maintainer decided (T1) to extend the
   grid before choosing. The transforms benchmark
   (`docs/dev/notes/2026-10-29-support-keyed-transforms-benchmark.md`,
   "Interpretation") found smaller steps better on every family and, under its
   rule, points to 1/20 on `Interval`, 1/50 on the half-bounded maps, and 1/20
   or 1/50 on `LogInterval`, and its smallest candidates sit at the edge of the
   grid. Every default quoted in this document is the current constant and moves
   with the chosen one.
5. The identity map has no default: `NoDefault`, naming the table and the kind
   (an undeclared `real` or `instant`), and saying that on this map z is θ
   itself, so the value to write is a natural-scale SD.
6. `initial_state = true` sets `PerturbAt::TimeZeroOnly`. Under IF2 the flag is
   checked against the model (§4.6): it is refused on a parameter that enters
   the dynamics, and an estimated parameter that enters only the initial state
   must carry it.
7. **A simplex member under IF2** is `If2Step::SimplexMember`. A written `sd` is
   its SD on the log-ratio coordinate; an omitted one takes the branch's
   `simplex_member_default_step` (_branch_ `runner.rs:1516`). Both go into
   `simplex`, not `params`. `initial_state = true` on a member is
   `InitialStateOnSimplexMember`, today's refusal (`config_v2.rs:3248-3256`)
   moved here: the group already perturbs its members at t = 0 only
   (`if2.rs:518-521`).
8. **A simplex member under PMMH or MH** is an ordinary parameter. Both treat
   members as independent coordinates on their own maps (the `simplex_groups`
   field doc, `config_v2.rs:96-99`), so a member takes `default_mh_step` on its
   own map or its `[method.proposal]` entry.

The launch table prints, per parameter, the map, the SD in z, `written` or
`default`, the natural-scale SD it implies at the resolved start (§7.3), and
`t0 only` for an `initial_state` parameter. "N parameters using auto rw_sd"
becomes "N parameters use the default step".

### 4.5 The names

**`perturbation`** names what IF2 does with the value: Ionides et al. (2015,
_PNAS_ 112(3):719–724) and pomp's `mif2` both call the kernel a perturbation
(§3.6). **`proposal`** is pomp's PMCMC argument and the standard
Metropolis–Hastings term. Under `adapt = false` it is the proposal for the whole
run; under `adapt = true` it is the proposal's shape until `adapt_start`, while
a global scale adapts from the first step (`sim/src/inference/pmmh.rs:654-667`).
**`initial_state`** names the model property, not the IF2 timing: "this
parameter sets only the initial state". It is preferred to pomp's `ivp`, which
is opaque to anyone who has not used pomp, and to `initial_value`, which names
pomp's expansion rather than the model fact. IF2's timing is
`PerturbAt::TimeZeroOnly`, derived from the flag. If perturbation scheduling is
ever extended (pomp's `ivp(s, lag)`, or a perturbation that stops after some
observation), a schedule key such as `timing = "initial"` beside `sd` may state
the timing more clearly than a property flag does; that representation is not
designed here.

Rejected:

- **`rw_sd` under `[method]`**: same spelling, new unit, so a config copied from
  an older document would load and run with every step reinterpreted.
- **One name for both, such as `[method.step]`**: today's coupling under a new
  name; a value tuned for IF2 would still flow into a PMMH run of the file.
- **`[method.rw]`, `random_walk`**: both kernels are random walks.

### 4.6 `initial_state` is a model property the IR determines; IF2's timing is checked against it

Two things are kept distinct: the **model property** that a parameter enters
only the initial state, and IF2's **perturbation timing**
(`PerturbAt::TimeZeroOnly`). The flag `initial_state = true` declares the first.
IF2 derives the second from it.

**What the compiler and IR offer today (verified by reading):**

- No analysis classifies parameters by where they enter the model. The OCaml
  lint lists an unused-parameter check as future work
  (`ocaml/lib/ir/lint.ml:14`).
- The nearest template is `referenced_compartments`
  (`ocaml/lib/ir/lint.ml:94-194`). It walks every surface for compartment
  references:
  - transitions: stoichiometry, rate, overdispersion σ², lineage weights;
  - ODE equations;
  - intervention actions;
  - observation projections and likelihoods;
  - bindings and initial conditions, including law arguments;
  - the balance constraint;
  - forcing coefficients.
- `fit/coeff_guard.rs:327-370` sorts forcings and tables into the
  initial-condition, rate and observation surfaces. It skips interventions, ODE
  equations, σ² and balance, because its job is gradient coverage, not a full
  classification.
- `ir::Model.ic_grad` (`rust/crates/ir/src/model.rs:447-457`) records, per
  deterministic initial condition, the parameters with a non-zero derivative.
  The compiler emits it only on the autodiff path
  (`ocaml/lib/compiler/compiler.ml:805-843`), and it omits initial-condition
  laws. It is therefore partial evidence that a parameter enters the initial
  state, and never evidence that it enters _only_ the initial state.

**The surfaces a parameter can reach in the IR.** All are read from the compiled
`ir::Model` (`rust/crates/ir/src/model.rs:405-480`).

- **Dynamics or likelihood:**
  - transition rates, the σ² of `DrawOverdispersed`, and lineage parent-pool
    weights;
  - ODE equations;
  - intervention actions and fire sources: a parametric `at` schedule or a
    reactive trigger (`rust/crates/ir/src/intervention.rs:169-196`);
  - observation likelihoods and `DerivedExpr` projections;
  - the balance expression.
- **Indirect:** bindings and `per_eval_bindings` (read through `BindingRef`),
  and time functions and tables (read through `TimeFunc` and `TableLookup`) pass
  a parameter on to whichever surface reads them.
- **Initial state:** `initial_conditions`, every expression of each spec
  (`InitSpec::exprs`, `model.rs:150`), law arguments included.
- **Neither, for this purpose:**
  - hierarchical-prior arguments (`parameter.rs:104-108`), which IF2 ignores;
  - `quantities`, which are outputs;
  - presets, scenarios and constants, which supply values rather than reference
    parameters.

The property can therefore be derived reliably from the compiled IR, with no
compiler or IR change, provided the walk is exhaustive:

```rust
// rust/crates/cli/src/fit/param_surfaces.rs (new)
pub struct ParamSurfaces {
    pub initial_state: BTreeSet<String>,   // reached from initial_conditions
    pub dynamics: BTreeSet<String>,        // reached from any dynamics or likelihood surface
}
pub fn param_surfaces(model: &ir::Model) -> ParamSurfaces;
impl ParamSurfaces {
    pub fn initial_state_only(&self, name: &str) -> bool;   // in the first, not the second
    /// The first dynamics surface naming `name`, for the refusal message.
    pub fn first_dynamics_use(&self, model: &ir::Model, name: &str) -> Option<String>;
}
```

**Exhaustive matching is necessary, not sufficient.** The function destructures
`ir::Model { … }` with every field named and no `..`, and matches every
`ir::Expr` variant with no wildcard arm. Adding an IR field or an expression
variant therefore does not compile until this function classifies it. That
proves only that every field and variant is _visited_, not that each is
classified _correctly_. Three further rules give the semantic guarantee.

1. **Transitive dependencies, with cycles handled.** A parameter reaches a
   surface when that surface reads:
   - a binding (`BindingRef`, in `bindings` and `per_eval_bindings`),
   - a time function (`TimeFunc`, through its coefficients),
   - or a table (`TableLookup`, through its values)

   that itself reaches the parameter, at any depth. The walk keeps a visited set
   per surface, keyed by binding, time-function and table name, so it terminates
   on any IR. Classification is a union over everything reachable, so a
   revisited node adds nothing.

   The compiler already refuses definition cycles: a `let` cycle is E201
   (`check_let_cycles`, `ocaml/lib/compiler/expander.ml:83-85`, gh#492), and an
   initial-condition cycle is `InitDependencyCycle`
   (`ocaml/lib/compiler/compiler.ml:497`). So the visited set guards only
   hand-built IR.
2. **References outside expressions count.** Some parameter references are
   names, not `Expr` nodes. A reactive trigger's threshold is one:
   `TriggerThreshold::Param` (`rust/crates/ir/src/intervention.rs:304`). Each
   such field is classified explicitly. The destructuring rule makes a new one
   impossible to miss.
3. **Uncertainty fails conservatively.** A reference the walk cannot resolve
   makes the classification `Uncertain`, naming the reference and the surface.
   Examples: a `BindingRef`, `TimeFunc` or `TableLookup` whose name is absent
   from the model, or a `PerEvalRef` without its binding. Under `Uncertain`,
   every parameter is treated as entering the dynamics. So `initial_state` is
   refused for all of them, and the refusal names the unresolved reference. The
   "must carry `initial_state`" refusal is not raised. An uncertain
   classification can therefore never license `initial_state`.

```rust
pub enum Classification {
    Resolved(ParamSurfaces),
    Uncertain { reason: String, surface: &'static str },
}
pub fn param_surfaces(model: &ir::Model) -> Classification;
```

**Parameters that reach neither the initial state nor the dynamics** are a
defined class: `unused_by_likelihood`. Its behaviour depends on whether the
parameter appears anywhere at all.

- **At compile time,** a parameter referenced nowhere, not even in a prior, is
  the planned unused-parameter lint (`ocaml/lib/ir/lint.ml:14`). That lint is a
  warning, because such a model is still well-formed to simulate.
- **At fit time, estimating such a parameter is a request with no defined
  effect** (invariant I3), so it is refused:
  - under every method, when the parameter appears nowhere;
  - under IF2 and `nl-*`, also when it appears only in a hierarchical-prior
    argument, because those methods ignore priors and nothing selects it.
- **A hyperparameter under PMMH, MH or PGAS** appears only in prior arguments
  and is accepted. It enters the posterior through the prior, which is its
  purpose.

The refusal names the parameter and says it enters neither the initial state nor
the likelihood.

**Tests run on compiled models**, real `.camdl` files through `camdlc`, not only
hand-built IR:

- the SIR recovery case (`tests/recovery/cases/sir/model.camdl`):
  - `init { S = N0 - I0  I = I0 }`, with `N` a `let` over compartments, so `N0`
    and `I0` are initial-state-only;
  - `beta`, `gamma`, `rho` and `k` are dynamics;
- initial-condition laws (`ocaml/golden/init_laws.camdl`): each law's parameter
  is initial-state-only;
- a parametric `at [...]` (`ocaml/golden/sia_anchored_dates.camdl:52`): the
  timing parameter is dynamics;
- a reactive trigger with a parameter threshold
  (`tests/fixtures/reactive/reactive_sir_observed_threshold.camdl:37`): the
  threshold is dynamics;
- a model with forcing (`ocaml/golden/flu_data_forcing.camdl`) and one with
  inline tables (`rust/crates/sim/tests/fixtures/licm_ab.camdl`): a coefficient
  parameter is dynamics, through the forcing or table;
- a hand-built IR for each path that `camdlc` cannot produce:
  - a binding cycle, which must terminate;
  - an unresolved `BindingRef`, which must give `Uncertain`;
  - a parameter in an `init` and a hierarchical prior's argument, which must be
    initial-state-only.

**Refusals under IF2** (`fit run` with `algorithm = "if2"`, and every `profile`
IF2 cell):

- An estimated, initial-state-only parameter without `initial_state` is refused.
  The message: "`I0` enters only the initial state (`init { I = I0 }`). IF2
  perturbing it before every observation moves it with nothing to select it
  after t = 0 (`if2.rs:443-455`). Write
  `[method.perturbation] I0 = { initial_state = true }`, or
  `profile --initial-state I0`."
- These refusals are about the IF2 schedule, not the statistical target.
  Perturbing an initial-state-only parameter before every observation is an
  incorrect IF2 schedule for the same estimand: the random walk after t = 0
  moves the parameter with nothing to select it, so the estimate degrades. The
  target is unchanged.
- `initial_state = true` on a parameter that enters the dynamics is refused,
  naming the first dynamics use. Perturbing it only at t = 0 would leave it
  almost unexplored.
- Simplex members are outside both rules. A member may not carry the flag (§4.4
  rule 7), and the group perturbs it at t = 0 regardless.

**Where it is not raised.** PGAS, PMMH, MH, `pfilter` and `nl-*` have no
perturbation timing, so they never warn. A PGAS or PMMH source config therefore
says nothing on its own. `profile --fit` with such a config is refused only when
profile's own IF2 cells would perturb an initial-state-only parameter unmarked,
and the message names that parameter and `--initial-state`. This closes the
`camdl-book` case of §3.2.

The flag stays explicit although the property is derivable: a schedule is
declared, not defaulted, and the check makes a wrong declaration impossible to
run. Deriving the timing automatically is possible later on top of
`param_surfaces`. Configs that today estimate an initial-state-only parameter
under IF2 without `perturb_only_at_t0` are newly refused, with the one-line fix
(§9.1).

## 5. Identity: the problem digest stops seeing steps and starts; the method digest sees them resolved

### 5.1 The fit level

The fit level needs one change. With no step fields in `EstimateSpecV2`,
`fit_config_blob_hash(problem)` no longer covers steps. `start` stays in the
file's `[estimate]` but leaves the fit level (§5.2). Two configs that differ
only in `[method]`, steps included, or only in starts, then have the same
fit-level hash; that was already true of every other `[method]` key
(`cas.rs:428-430`).

### 5.2 `start` is hashed at the method level

`FIT_PROVENANCE_KEYS` subtracts top-level keys only (`serialize_minus`,
`cas.rs:231-255`), and `start` is a field of each `[estimate]` entry. The
problem's hashed view therefore drops it explicitly. The method level hashes the
written `[estimate].start` values (`starts_base`) beside the resolved starts
rule (`starts`, already in the method payload):

```rust
// rust/crates/cli/src/fit/cas.rs
// before
fn fit_config_blob_hash(problem: &Problem) -> Result<ContentHash, String> {
    Ok(canonical_config_hash(problem, FIT_PROVENANCE_KEYS)?.into_inner())
}
// after
fn fit_config_blob_hash(problem: &Problem) -> Result<ContentHash, String> {
    Ok(canonical_config_hash(&problem.identity_view(), FIT_PROVENANCE_KEYS)?.into_inner())
}

// rust/crates/cli/src/fit/config_v2.rs
impl Problem {
    /// The problem as its identity sees it: every field, with each
    /// `[estimate]` entry's `start` removed (execution, §5.2).
    pub fn identity_view(&self) -> ProblemIdentityView<'_>;
    /// The `[estimate].start` values, by name; what the method level hashes.
    pub fn start_values(&self) -> BTreeMap<&str, f64>;
}
```

The parameter-roles proposal owns starts (its §5.3: starts come only from
`--starts` or `[estimate].start`), so this is a coordinated amendment to it
(§13.3). The test: two configs identical but for `[estimate] beta.start` have
equal `fit_level_hash` and different method-level hashes. Its mutation check
leaves `start` in `identity_view`, which must turn the first assertion red.

### 5.3 The method level

```rust
// rust/crates/cli/src/fit/config_v2.rs, Method::identity_payload
// before
Algorithm::PMMH { .. } | Algorithm::Mh { .. } => self.payload_minus(&["iterations"]),
Algorithm::IF2 { .. } | … => self.payload_minus(&[]),
// after: the written tables are subtracted; their resolved form is added below.
Algorithm::PMMH { .. } | Algorithm::Mh { .. } => self.payload_minus(&["iterations", "proposal"]),
Algorithm::IF2 { .. } => self.payload_minus(&["perturbation"]),
Algorithm::PFilter { .. } | Algorithm::NlSbplx(_) | … => self.payload_minus(&[]),
```

```rust
// rust/crates/cli/src/fit/cas.rs
pub struct FitStageCtx<'a> {
    // …existing fields…
    pub steps: &'a ResolvedSteps,
    pub starts_base: &'a BTreeMap<&'a str, f64>,    // Problem::start_values
}

// before (cas.rs:482-493)
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
fn stage_config_hash(
    method: &Method,
    steps: &ResolvedSteps,
    starts_base: &BTreeMap<&str, f64>,
) -> Result<ContentHash, String> {
    method.starts()?;
    ensure_finite(method)?;
    ensure_finite(starts_base)?;
    let v = serde_json::json!({
        "identity": method.identity_payload(),
        "n_trajectories": method.algorithm.cas_n_trajectories(),
        "steps": steps.identity(),
        "starts_base": starts_base,
    });
    Ok(digest_value(&v))
}
```

`ResolvedSteps::identity()` is a name-keyed object (canonical JSON sorts keys,
`cas.rs:155-169`, so order in the file does not matter). `from` is never in it,
so a written default and an omitted name key the same, following the `starts`
precedent of resolving before identity (`config_v2.rs:1321`,
`Method::resolve_starts`):

```json
{ "kind": "if2",
  "perturbation": { "beta": { "sd": 0.107, "at": "every_observation" },
                    "I0":   { "sd": 0.174, "at": "time_zero_only" } },
  "simplex": [ { "p_a": 0.2, "p_b": 0.2, "p_c": 0.2 } ] }

{ "kind": "proposal",
  "proposal": { "beta": 0.537, "gamma": 0.528, "I0": 0.869 } }

{ "kind": "none" }
```

`StepFrom` is never hashed. It appears in the launch table, and in the run
record's unhashed `inputs` (§5.6), so a reader can tell whether a step was
written or defaulted.

`resume_guard` (`cas.rs:603`) hashes the same levels with the extension
dimension zeroed. A PMMH or MH resume under a different proposal is therefore
still refused: the step now moves the method level, which the resume key hashes,
where today it moves the fit level. `config_identity_hash` (`cas.rs:467`), the
handle lookup, subtracts `[provenance]` and, from result-records Stage 2,
`[assessment]` (that proposal's §4.3); this proposal does not change it, and the
step tables stay inside it.

### 5.4 Profile

`profile --algorithm` selects IF2 or one of the `nl-*` optimizers per cell
(`profile.rs:209-222`). For IF2 it reads `[method.perturbation]` from a `--fit`
config whose `[method]` is IF2 (D9). Under `nl-*` there is no step: the step
flags are refused, a `[method.perturbation]` read through `--fit` is ignored
with one notice, and the method level hashes `{"kind": "none"}`. This is
consistent with invariant I3 ("Every accepted user option has a defined,
observable effect on the computation, or is refused",
`docs/dev/inference-architecture.md`): the step flags are options given to this
command and are refused, while the table is content of a referenced config,
written for that config's IF2 method, which this command does not run; the
notice makes its non-use visible. Today profile hashes the `rw_sd` blob under
every algorithm, `nl-*` included (`profile.rs:1293-1310`), a setting that does
not reach the computation keying the result: the same class of defect as §3.3.

Profile loads the file with `FitConfig::load` in place of `Problem::load`
(`profile.rs:337`), uses `.problem` as today, and reads only that one table from
`.inference`. Its identity changes in two places:

- **Base level.** `ProfileBaseLevel.fit_toml`, the SHA-256 of the file's bytes
  (`profile.rs:359-366`), becomes `fit_problem`, the hex of
  `fit_config_blob_hash(&config.problem)`. The base level then identifies the
  problem, as its docstring says (`profile.rs:126-133`): a change to `[method]`,
  a start or a comment no longer re-keys it.
- **Method level.** `ProfileMethodLevel.rw_sd` and `.rw_sd_auto`
  (`profile.rs:148-163`, filled at `:1293-1310`) become `steps`, the
  `identity()` of the profile's resolved steps.

### 5.5 Tests the identity change lands with

Each is mutation-checked: the named mutation must turn it red.

| test                                                                                                   | mutation                                              |
| ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| configs identical but for `[method.perturbation] beta`: equal fit level, different method level        | drop `"steps"` from `stage_config_hash`               |
| the §3.7 IF2 and PGAS files: equal fit level                                                           | add the resolved steps to `FitDigest`                 |
| `beta = 0.2` on a `LogInterval` parameter against the line omitted: equal method level                 | hash the written table in place of `steps.identity()` |
| `beta = 0.2` against `beta = { sd = 0.2, initial_state = true }`: different method level               | drop `at` from `identity()`                           |
| configs identical but for `beta.start`: equal fit level, different method level                        | keep `start` in `identity_view`                       |
| two `profile --fit` runs whose configs differ only in `[method] particles`: equal base level           | hash the file's bytes again                           |
| `profile --algorithm nl-sbplx` with and without `[method.perturbation]` in `--fit`: equal method level | hash the perturbation table under `nl-*`              |

### 5.6 The resolved scales are part of the plan and of the saved run record

Each method's `ExecutionPlan` carries every estimated parameter's resolved step
in z, with its natural-scale equivalent at the resolved start (§6.1):

- `If2Plan.steps`, `If2Plan.simplex`;
- `PmmhPlan.proposal.sd`, `MhOdePlan.proposal.sd`;
- `PgasPlan.mwg_initial_scale` and `PgasPlan.nuts_initial_step`.

```rust
pub struct StepRecord {
    pub sd_z: f64,
    pub at: Option<PerturbAt>,         // IF2 only
    pub start: f64,                    // the resolved start θ₀ it was converted at (§7.3)
    pub natural_at_start: f64,         // sd_z × |dθ/dz| at θ₀
    pub from: StepFrom,                // written or default; recorded, never hashed
}
```

The saved provenance is `run.json`'s `inputs`, the field the run record keeps
for "Resolved-input summary for display/audit — provenance, not hashed"
(`rust/crates/runid/src/record.rs:196-198`). `fit run` fills it in `finalize`
(`fit/mod.rs:1225-1240`, where it starts `Null`) with:

```json
"steps": { "kind": "if2", "map": { "beta": "log_interval" },
           "perturbation": { "beta": { "sd_z": 0.107, "at": "every_observation",
                                       "start": 0.25, "natural_at_start": 0.02,
                                       "from": "written" } } }
```

`profile` writes the same block into each cell's record, beside its existing
`parameters_provenance` (`profile.rs:1213-1224`).

This squares with run identity because nothing recorded is new information:

- `sd_z` and `at` are already hashed at the method level (§5.3).
- `from` (`written` or `default`) is a per-execution fact outside the hash. A
  written default and an omitted name run the same numbers and share one
  execution identity. Which one the user wrote is provenance.
- `start` is hashed at the method level (§5.2).
- The map is fixed by the model and the box, both at the fit level.
- `natural_at_start` is a function of those three.

The record therefore re-encodes identity inputs, which the run-identity rule
calls presentation (`.claude/rules/run-identity.md`). Recording `from` there
follows an existing precedent: the run record already keeps per-execution facts
outside the hash. Its `provenance` holds `argv`, `host`, `created_at` and
`thread_count` (`rust/crates/runid/src/record.rs:144-160`), none of which keys
the run. Two runs that share an identity may have different `from` values, as
they may have different `argv`. `fit_state.toml` keeps only
`initial_state_params`, which the convergence check reads (§7.5).

## 6. The resolved problem is shared by every method; the execution plan is per run (amends parameter-roles §6.3, §6.4, §6.6)

### 6.1 The split

The parameter-roles `ResolvedProblem` (its §6.3) bundles the partition and fixed
values with `starts: DrawnStarts` and `algorithm: AlgorithmSettings`, so two
methods or two seeds of one problem cannot share one. Its
`AlgorithmSettings::Other` also puts PMMH, PGAS, NLopt and `pfilter` in one arm
and fills PMMH's step from IF2's automatic value (its §6.4). The amendment,
approved by the maintainer, splits the type along the identity levels:

```rust
// rust/crates/cli/src/fit/plan.rs (new)

/// The fully resolved statistical problem: every input that changes the
/// statistical target, and nothing that depends on the algorithm, the starts
/// or the seed. An IF2 run and a PGAS+NUTS run of one problem share one.
/// Keyed by the fit level (§5.1): a `fit run --sweep` point, and a profile
/// cell with its focal values held, are each a problem of their own, sharing
/// the compiled model.
pub struct ResolvedProblem {
    pub model: Arc<CompiledModel>,
    /// Roles, and for each estimated parameter its support, search box,
    /// transform (the effective domain: the bounds that truncate the posterior)
    /// and prior, in `Estimated` (roles §6.2, amended: no `start`). The one
    /// holder of prior, box and transform.
    pub partition: Partition,
    pub fixed: ResolvedParameters,             // fixed values, with provenance
    pub interventions: InterventionSet,
    pub streams: ObservationStreams,           // the loaded [data] streams
    pub simplex_groups: Vec<config_v2::SimplexGroup>,
    pub objective: LikelihoodObjective,        // ic_free: which initial states are inferred
    pub window: TrainingWindow,                // the scored window and any holdout
}

/// One run's execution configuration: a problem, a method on it, and where
/// its chains begin. Validated when built, at the CLI boundary.
pub struct ExecutionPlan {
    pub problem: Arc<ResolvedProblem>,
    pub method: MethodPlan,
    pub starts: fit::chain_starts::DrawnStarts,
    pub seed: u64,
}

impl ExecutionPlan {
    /// Validates that `method` can run on `problem`, and refuses if it cannot.
    /// Takes the problem by shared `Arc`: no `&mut`, no owned copy. A method
    /// may refuse a problem; it can never reinterpret its priors, roles or
    /// constraints.
    pub fn new(
        problem: Arc<ResolvedProblem>,
        method: &config_v2::Method,
        steps: ResolvedSteps,
        starts: fit::chain_starts::DrawnStarts,
        seed: u64,
    ) -> Result<ExecutionPlan, PlanError>;
    pub fn n_chains(&self) -> usize;              // from `method`
    pub fn identity(&self, ctx: &IdentityInputs) -> Result<ResolvedFitStage, String>;  // the three levels and run_id
}

// `ExecutionPlan::new` runs every check that depends on the method, so that
// `ResolvedProblem` is method-free by construction: the roles D8 refusals
// (`UnboundedForMethod`, `HyperOverHeldLeaf`, `HierarchicalLeafBoxed`, each
// raised only for the methods they concern), the `ic_free` precondition
// (which reads the IF2 steps), and the initial-state checks of §4.6.
//
// **The rule: a method may refuse a problem, never reinterpret it.**
// `ExecutionPlan::new` receives `Arc<ResolvedProblem>`, and `ResolvedProblem`
// has no interior mutability and no `&mut self` methods, so the type system
// stops a plan from changing a prior, a role, a box or a transform. Today a
// method can reinterpret a problem; for example, PMMH filled its step from
// IF2's automatic value (roles §6.4). What a method needs that the problem
// does not state is method configuration, and goes in `MethodPlan`.
//
// **Plans are fully resolved** (invariant I2, `docs/dev/inference-architecture.md`:
// "no `Option` that lets the engine pick a default later"). Today PGAS's and
// PMMH's `burn_in` and `thin` are `Option` in the method
// (`config_v2.rs:1066-1069`) and are filled at dispatch: PGAS 2000 and 5
// (`cli/src/fit/pgas.rs:217-218`, `:237-238`), PMMH and MH 5000 and 10
// (`cli/src/fit/pmmh.rs:37-38`, `:54-55`, `:72-73`). This has two
// consequences:
//
// - a file that writes the default and one that omits it key differently;
// - changing a default changes execution without changing the method payload.
//
// The plan series resolves them when the plan is built. `PgasPlan`,
// `PmmhPlan` and `MhOdePlan` hold `burn_in: usize` and `thin: usize`.
// `Method::resolve_defaults` fills them before the identity is taken, as
// `resolve_starts` does, so the method level hashes the resolved numbers and
// the engines receive concrete values. The omitted `rw_sd`, today resolved
// inside `FitRunConfig::build` after the identity is taken
// (`runner.rs:1411-1413`), is the same defect; `resolve_steps` (§4.4) fixes it.

/// Per-method settings, typed per method. Not `InferenceMethod`, the
/// (algorithm, backend) registry entry (`fit/methods.rs:50`).
pub enum MethodPlan {
    If2(If2Plan),
    Pgas(PgasPlan),
    Pmmh(PmmhPlan),
    MhOde(MhOdePlan),
    Pfilter(PfilterPlan),
    Nlopt(NloptPlan),          // nl-sbplx, nl-bobyqa, nl-lbfgs
}

pub struct If2Plan {
    pub steps: IndexMap<String, ResolvedPerturbation>,   // §4.4
    pub simplex: Vec<IndexMap<String, ResolvedProposal>>,
    pub dt_check: DtCheckMeasurement,
    /* chains, particles, iterations, cooling, cooling_target_iters, loglik_eval */
}

/// What a post-fit dt-check measures; its tolerance is assessment policy.
pub struct DtCheckMeasurement {
    pub enabled: bool,
    pub n_halvings: usize,
    pub n_particles: usize,
    pub n_replicates: usize,
}

/// The proposal PMMH and MH share.
pub struct MhProposal {
    pub sd: IndexMap<String, ResolvedProposal>,          // §4.4
    pub adapt: bool,
    pub adapt_start: usize,
}
pub struct PmmhPlan  { pub proposal: MhProposal, /* chains, particles, iterations, burn_in, thin */ }
pub struct MhOdePlan {
    pub proposal: MhProposal,
    pub dt_check: DtCheckMeasurement,
    /* chains, iterations, burn_in, thin, burnin_dt */
}
/// nl-sbplx, nl-bobyqa, nl-lbfgs. NLopt runs the dt-check too (`nlopt_stage.rs:243`).
pub struct NloptPlan {
    pub algorithm: NloptAlgorithm,
    pub chains: usize,
    pub tolerance: f64,          // xtol_rel: a stopping control, execution
    pub max_evals: usize,
    pub dt_check: DtCheckMeasurement,
}
pub struct PgasPlan {
    /// MH-within-Gibbs initial scale per parameter, in z (backend default, §3.8).
    pub mwg_initial_scale: IndexMap<String, f64>,
    /// NUTS initial step size before dual averaging (backend default, §3.8).
    pub nuts_initial_step: f64,
    /* chains, particles, sweeps, use_nuts, dense_mass, … */
}
```

A plan holds execution settings only. Post-run thresholds are assessment policy
under the result-records proposal's decided D3 (its §5), and live in its new
`[assessment]` table, not in any plan:

- IF2's chain-agreement threshold (`gate.a_thresh`);
- the decibans threshold (`gate.decibans_thresh`, IF2 and NLopt);
- the dt-check tolerance (`threshold_nats`, `--dt-check-strict`).

What a dt-check or clean evaluation _measures_ (particles, replicates, halvings)
stays execution and stays in the plan.

`ResolvedSteps` (§4.4) is the step-bearing part of `MethodPlan`: its `If2` arm
is `If2Plan.steps` and `.simplex`, its `Proposal` arm `MhProposal.sd`, and its
`None` arm the other variants. `MethodPlan::steps_identity()` produces the JSON
of §5.3. The chain starts' base points (`[estimate].start`) are part of
`ExecutionPlan.starts`, not of the problem.

### 6.2 Four things stay distinct

- **Execution configuration**: `ExecutionPlan`, what a run is told to do.
- **What was executed**: the run record (`run.json` and the stored artifacts),
  with immutable identities and provenance, written by the run.
- **Method-specific results**: the existing per-method result types
  (`IF2Result`, `PMMHResult`, the PGAS chains, NLopt outcomes).
- **Assessment**: an interpretation of a result under a stated policy
  (convergence verdicts), the `Assessment` type of the result-records proposal
  (`2026-10-29-result-records-and-assessment.md` §4.4), computed from a
  `ResultRecord` (its §4.1, the method-specific result) under a
  `DiagnosticPolicy` (its §4.3). It is not designed here.

`ExecutionPlan` does not replace `FitRunConfig`: during the migration
`FitRunConfig` stays the runner-side assembly, built from the plan, until each
backend consumes the plan directly. It carries no results, no record fields and
no verdicts.

### 6.3 Sequencing of the split

1. `ExecutionPlan` is introduced as a fully validated object at the CLI
   boundary: `fit run` builds one `Arc<ResolvedProblem>` per fit segment and one
   `ExecutionPlan` per method and seed, and every runner still receives what it
   receives today, derived from the plan.
2. Backends move to consume the plan one at a time (IF2, PMMH, MH, PGAS,
   `pfilter`, `nl-*`), each in its own commit.
3. Resolving `burn_in` and `thin` in the plan (I2) is its own commit. It keeps
   output bit-identical, since the resolved numbers equal the dispatch defaults,
   and it re-keys only through the payload.
4. Each structural commit keeps fixed-seed numerical behaviour, proved by the
   fixed-seed pin of §11.

This lands before parameter-roles Stage 2, which then builds on it (§11).

### 6.4 A later stage: the particle-filter settings leave `IF2Config`

Every method's runner is built through `FitRunConfig::build`, whose result
carries an `IF2Config` (`runner.rs:106`, `:571`). The other methods read the
time step from it (`cli/src/fit/pmmh.rs:142`, `:308`; `pgas.rs:312`;
`nlopt_stage.rs`; `dt_check.rs`; verified by
`grep -rnoE "if2_config\.\w+" rust/crates/cli/src`), and they pass IF2's cooling
arguments as placeholders: PMMH passes `1, 1.0, 1`, commented "IF2-specific and
never read by PMMH — pass harmless values" (`cli/src/fit/pmmh.rs:132-140`), and
PGAS the same (`pgas.rs:304-310`). The split:

```rust
// rust/crates/sim/src/inference/filter.rs (new)
/// What a particle-filter likelihood needs, under every method.
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
    pub steps: Vec<If2Step>,
}
```

`FitRunConfig::build` loses its `n_iterations`, `cooling` and
`cooling_target_iters` arguments and carries `FilterSettings`. This stage edits
`if2.rs`, a high-risk file the support-transforms branch also edits, and changes
no result, so it lands after that branch merges, with the bit-identity check of
§6.3.

## 7. A config with the old keys is refused where it would run, and set aside where it is only read

### 7.1 Live configs are refused

`FitConfig::from_toml_str_named` runs migration detectors before the typed parse
(`config_v2.rs:2288-2294`). A new one reads the raw `[estimate]` table and
returns every entry carrying `rw_sd` or `perturb_only_at_t0`:

```rust
pub struct LegacyStepKey {
    pub param: String,
    pub rw_sd: Option<f64>,            // natural scale, as written
    pub perturb_only_at_t0: bool,
}
pub struct LegacyStepKeys {
    pub algorithm: Option<String>,     // `[method] algorithm`, if any
    pub entries: Vec<LegacyStepKey>,
}

pub fn detect_legacy_step_keys(raw: &toml::Value) -> Option<LegacyStepKeys>;

pub enum Conversion {
    At { theta0: f64, from: Theta0From, dtheta_dz: f64, default_sd: Option<ZSd> },
    Unavailable(&'static str),
}
pub enum Theta0From { Start, ModelValue, BoxMidpoint }

impl LegacyStepKeys {
    pub fn message(&self, conversions: &IndexMap<String, Conversion>) -> String;
}
```

A live config is one a command is asked to run or resolve: `fit run`,
`fit recovery`, `profile --fit`, `survey --fit`, `simulate --fit`,
`pfilter --fit`, and a lookup by a config the user passes. Every such command
calls `config_v2::load_with_model(path)`. On this refusal it compiles the model
named by `[model] camdl`, builds each listed parameter's map with the transforms
series' `Transform::for_domain`, and prints the converted values. If the model
does not compile, it prints the formula-only message and the compiler's error. A
live config is refused even by `simulate --fit`, which takes no step, so the
file does not stay in circulation until a fit reads it.

### 7.2 Stored copies are set aside with one notice

Two readers reload a stored fit's archived config, `fit.toml.original`, through
`FitConfig::load`: `fit table` (`fit/fit_table.rs:297-309`,
`load_archived_fit_toml`) and the `@handle` lookup (`fit/handle.rs:174-199`,
`load_config_for_segment`, through `FitConfig::load_anchored_at`). A fit stored
before this release has the old keys in that copy. Refusing there would make
every such fit unlistable and unaddressable.

The precedent is `ef81ade7`, which sets aside a verified pre-0.41 fit artifact's
typed-constant entries with one info notice, and refuses anything that is not
such an artifact. Here: a loader for archived copies,
`FitConfig::load_archived(path)`, removes `rw_sd` and `perturb_only_at_t0` from
`[estimate]` and returns the config with one notice listing the keys removed. It
is called only by read-only consumers (`fit table`, `fit summary`, `compare`, a
handle used as a starts source or a `fit predict` posterior). Anything that
would run a fit from the stored config — `fit run --resume` against a stored
leaf, or re-running an archived config — goes through `load_with_model` and
refuses. The stored fit's `run_id` is not recomputed, so setting the keys aside
does not misattribute it.

Tests: `fit table` over a store holding one pre-release fit lists it and prints
the notice once; `@handle` resolution of that fit succeeds with the notice;
`fit run --resume @handle` on it is refused naming the keys. Each is red when
`load_archived` refuses, or when `--resume` goes through `load_archived`.

### 7.3 How the value is converted, and when it cannot be

A natural-scale step is converted at the **resolved start value** θ₀ (D19). The
same rule holds everywhere a natural step and a z step meet:

- the migration errors;
- the launch summaries and the run record's `natural_at_start` (§5.6);
- `profile`'s `--rw-sd` refusal;
- `profile --rw-sd` on the development branch before Stage 3.

The branch's `Transform::reference_z`, which uses the box midpoint for the
interval maps (_branch_ `types.rs:163-176`), is replaced by the resolved start.

sd_z = rw_sd / |dθ/dz| at θ₀, for the new map g.

**The conversion is a local approximation.** It matches the old step at θ₀ only.
IF2 now takes a fixed step in z, so the natural-scale step it implies changes as
θ moves away from θ₀, and the more so the more curved the map is there.

- **PMMH:** main computed its initial proposal at the same point (`p.initial`,
  `cli/src/fit/pmmh.rs:1051`). The printed `[method.proposal]` value therefore
  gives the old proposal SD at the start to first order. Away from θ₀ the two
  proposals differ, because main's map and the new one differ.
- **IF2:** main converted at every particle's current value (`if2.rs:428`), so
  no single z-SD reproduces it away from θ₀.

θ₀ is, in order (D19):

1. the entry's explicit `start`;
2. the resolved default start: the base point the chain-starts resolution gives
   when no start is written (today the model's declared value, the
   `EstimatedParam.initial` of `runner.rs:1418`);
3. the domain midpoint: arithmetic for `Interval`, geometric for `LogInterval`.

Under a spread starts rule each chain starts elsewhere; the conversion still
uses this one base point, and the launch summary says so.

**Near a bound the slope goes to 0.** On every map with a finite end, |dθ/dz| →
0 as θ₀ approaches that end, so the converted sd_z grows without limit. A start
exactly on a bound is already refused by the transforms series (its D3). For a
start near one:

- **Converted values above 3 in z are not printed as a recommendation.** A z-SD
  of 3 already spans most of an interval map's range. For such a parameter the
  migration message prints the number with its reason: "`rho` starts at 0.1002,
  2·10⁻⁴ above its bound 0.1, where |dθ/dz| = 2.0·10⁻⁴. A natural step of 0.01
  there is 50 in z, which crosses the whole box in one step." It then recommends
  the default step, with the default's natural equivalent at the start.
- **The launch summary prints the natural equivalent as computed** (small near a
  bound) and adds "start near bound".

The conversion is unavailable, and the message gives the formula with θ₀ to fill
in, when:

- a half-bounded map has no start and no default start, so |dθ/dz| (the distance
  to the bound) has no point to be taken at;
- no model is available, or it does not compile;
- the parameter is not in the model (the unknown-parameter refusal applies).

The identity map is always convertible (|dθ/dz| = 1). A simplex member is always
convertible and unchanged, because an explicit `rw_sd` was already read on its
log-ratio scale (`runner.rs:656-660`).

Per map, with θ₀ the conversion point:

| kind and box                                                      | map            | \|dθ/dz\| at θ₀                    | sd_z                                       |
| ----------------------------------------------------------------- | -------------- | ---------------------------------- | ------------------------------------------ |
| `real`, `instant`, no range                                       | `Identity`     | 1                                  | rw_sd                                      |
| `rate`, `positive`, `count`, `duration`, no range: [0, ∞)         | `LowerBounded` | θ₀ − lo                            | rw_sd / (θ₀ − lo)                          |
| `probability`; a positive kind on [0, hi]; a `real`/`instant` box | `Interval`     | (θ₀ − lo)(hi − θ₀)/(hi − lo)       | rw_sd·(hi − lo) / ((θ₀ − lo)(hi − θ₀))     |
| a positive kind on [lo, hi], 0 < lo                               | `LogInterval`  | θ₀·ln(θ₀/lo)·ln(hi/θ₀) / ln(hi/lo) | rw_sd·ln(hi/lo) / (θ₀·ln(θ₀/lo)·ln(hi/θ₀)) |

The `LogInterval` form follows from ln θ = ln lo + ln(hi/lo)·σ(z), whose
derivative is θ·ln(hi/lo)·σ(1 − σ) with σ = ln(θ/lo)/ln(hi/lo).

### 7.4 What the message says

| `[method] algorithm`       | `rw_sd` becomes                                                       | `perturb_only_at_t0 = true` becomes             |
| -------------------------- | --------------------------------------------------------------------- | ----------------------------------------------- |
| `if2`                      | `[method.perturbation] NAME = <sd_z>`                                 | `NAME = { sd = <sd_z>, initial_state = true }`  |
| `pmmh`, `mh`               | `[method.proposal] NAME = <5 × sd_z>`, the old initial proposal scale | already refused there; delete it                |
| `pgas`, `pfilter`, `nl-*`  | never read (§3.1); delete it                                          | already refused; delete it                      |
| none (a problem-only file) | both rows above, and `profile --perturbation NAME=…`                  | the IF2 row, and `profile --initial-state NAME` |

Deleting a line is not neutral: it replaces the old step with the default, which
is a different step. The message says so and prints, beside each converted
value, the default's natural-scale equivalent at the same θ₀, so the user can
choose. For the §10 IF2 config:

```text
error: [estimate] sets step sizes, which are settings of the method:
  `rw_sd` (a natural-scale step) and `perturb_only_at_t0` moved to [method],
  in z, the coordinate each method moves in. This file's method is `if2`.
  Delete both keys from [estimate] and add:

    [method.perturbation]
    beta  = 0.107                                 # rw_sd 0.02 at start 0.25
    gamma = 0.106                                 # rw_sd 0.01 at start 0.1
    I0    = { sd = 0.174, initial_state = true }  # rw_sd 2 at start 10

  These give your old step at the start; IF2 now takes a fixed step in z,
  so away from the start the natural-scale step differs.

  Leaving a parameter out uses the default step instead, which is a
  different step. The default, 0.2 in z for these log-interval maps, gives
  at the start:  beta 0.037 (was 0.02)   gamma 0.019 (was 0.01)   I0 2.30 (was 2)
  For I0 with the default step write `I0 = { initial_state = true }`.
  See `camdl docs fit-toml`, "Step sizes".
```

The values come from the `LogInterval` row of §7.3: β on [0.05, 1] at 0.25 has
|dθ/dz| = 0.18619, so 0.02/0.18619 = 0.1074 and 0.2 × 0.18619 = 0.0372; γ on
[0.01, 0.5] at 0.10 has 0.09473, giving 0.1056 and 0.0189; `I0` on [1, 100] at
10 has 11.513, giving 0.1737 and 2.30 (verified by evaluating the formula; not
produced by camdl).

### 7.5 An old `fit_state.toml` is refused as an input and readable for inspection

`[rw_sd]` is deleted from `FitState` (§3.5); old files that carry it still load.
The initial-state list is renamed with the flag, `initial_state_params`. A
`fit_state.toml` whose schema this release does not support, which includes one
that carries `perturb_only_at_t0_params`, is handled by its use (D16, maintainer
decision):

- **As an input to a new computation it is refused.** The input paths are the
  same as in the result-records proposal
  (`2026-10-29-result-records-and-assessment.md` §6, on
  `docs/inference-invariants-and-assessment`):
  - a `from_mle` starts source: `draw_from_mle` in `fit run`
    (`fit/chain_starts.rs:1031`) and `profile`'s `--starts from_mle`
    (`profile.rs:1953`);
  - the warm-start source check, which every `from_mle` and `from_posterior`
    source passes through (`check_source_converged`, `fit/gating.rs:441`). A
    `from_posterior` source reads draws, not the state, so this is its one read
    of `fit_state.toml`;
  - the runner reading back the record it has just written (`fit/mod.rs:2101`).

  Both reads in the source check must propagate a load error rather than treat
  it as "no verdict". Today `sampler_source_verdict` swallows them (`.ok()?`,
  `fit/gating.rs:497`). A fix is in progress on `fix/settings-silent-noops`, and
  `load_input` keeps it.

  The error names the producer version (`FitState.camdl_version`,
  `state.rs:30-32`) and the schema version it was written under, and says the
  fit must be run again.

  Two paths that look like inputs are not on the list:
  - `fit run --resume` does not read `fit_state.toml`. `copy_resume_carryover`
    (`fit/mod.rs:2131-2149`) copies only each chain's `resume_state.bin` and
    `parameter_traces.tsv`. Its compatibility is the `resume_guard` key (§5.3),
    not this split.
  - `profile --fit` and `profile --fixed-file` read the fit config and
    `mle_params.toml`, not `fit_state.toml`. Stored configs are §7.2, and fit
    artifacts follow `ef81ade7`.
- **Read-only commands inspect it with one notice**: `fit summary` and
  `fit table` read it. So does `fit predict`, whose only read of the state is
  `method_result::point_start` (`fit/method_result.rs:935`); that check decides
  whether to demote the diagnostics of a point-started fit. Its posterior comes
  from the draws, not from `fit_state.toml`. Today that read drops any load
  error silently (`.ok()?`). Under the split it calls `load_for_inspection`: a
  legacy record gives the notice, and a corrupt one gives an error. In each of
  these commands the notice names the version and the unsupported keys. An old
  file's initial-state list is read under its old key for this purpose, so
  `fit summary` applies the chain-agreement exemption to the parameters the run
  exempted.
- **Listing commands never abort**: `fit table` reports an unreadable state as
  an error in that fit's row and lists the rest (it already matches on the load
  result, `fit_summary.rs:618`, `:2857`).

`FitState` gains a schema tag in the same commit (§11 commit 4):
`schema = "camdl.fit-state/1"`, written on every new state. A file without it
predates this release. `FitState::load` then takes the use:

- `FitState::load_input(dir)` refuses a missing or unknown tag, naming
  `camdl_version` and the tag, or "no schema tag, before camdl.fit-state/1";
- `FitState::load_for_inspection(dir)` returns the state with its notice.

The tag is what makes the refusal general. A refusal keyed on
`perturb_only_at_t0_params` would catch only old fits that had an initial-state
parameter, since that field is skipped when empty
(`#[serde(skip_serializing_if = "Vec::is_empty")]`, `state.rs:67-68`). The
precedent for a refusal inside `load` is `chain_index::refuse_one_based_layout`
(`state.rs:190`).

The split is interim. At result-records Stage 3, `ResultRecord` replaces
`FitState`, and the two functions become `ResultRecord::load_input` and
`ResultRecord::load_for_inspection` with the same semantics:

- the same input paths refuse;
- old records are still read for inspection;
- an old record is still assessed with its old initial-state exemption, read
  under its old key by that proposal's legacy adapter.

Tests: a pre-release IF2 state with `perturb_only_at_t0_params` is refused as a
`from_mle` source, naming the version; `fit summary` on it exempts the listed
initial-state parameters and prints the notice once; `fit table` over a store
with one such fit and one unreadable state lists both, the second as a row
error. Each is red when `load_input` and `load_for_inspection` are swapped.

## 8. `profile` declares its estimated set with `--estimate` and its steps with `--perturbation`

The parameter-roles proposal decides that `profile`'s estimated set comes from
`--estimate NAME,…` or `[estimate]` in `--fit` (its §4.4, Stage 3), and refuses
`--rw-sd auto` (its D5). This proposal replaces `--rw-sd`:

- `--perturbation NAME=SD[,NAME=SD…]`, repeatable: IF2 perturbation SD in z for
  an estimated name. Under IF2 it overrides `[method.perturbation]` from an IF2
  `--fit` config per name.
- `--initial-state NAME[,NAME…]`, repeatable: perturb at t = 0 only. It adds to
  the initial-state parameters an IF2 `--fit` config declares. There is no flag
  to remove one, because perturbing an initial-state-only parameter at every
  observation is an incorrect IF2 schedule (§4.6), which this proposal refuses.
- Under `--algorithm nl-*` both flags are refused: the optimizers take no step.
- `--rw-sd` is a removed-flag trap, like `_removed_pmmh_rho`
  (`args/mod.rs:1815-1819`). After the model loads it prints the `--estimate`
  list the old keys selected and the `--perturbation` values converted by §7.3;
  for `--rw-sd auto`, the parameter-roles D5 text.
- A name that is not estimated is `SettingForNonEstimated`; a name the model
  does not declare is the unknown-parameter refusal.

`survey` needs no flag: it declares its estimated set with `--estimate` or
`[estimate]` (`args/mod.rs:1984-1992`) and takes no step.

There is no interim. The release that carries the step series waits for
parameter-roles Stage 3, so no released `profile` takes `--rw-sd` beside
`[method.perturbation]`: one z-scale interface, never two units at once
(maintainer decision 7). On the development branch, between the step series and
Stage 3, `profile --rw-sd` converts at the resolved start like every other path
(§7.3).

## 9. What breaks, and the migration text

### 9.1 Inputs refused

| what                                                                 | where, and how many                                                                                                                                                                                                                                                                                  |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rw_sd` / `perturb_only_at_t0` in a live fit config                  | this repository: no committed `.toml` config sets either (`git ls-files '*.toml' \| xargs grep -l rw_sd` → one comment, `tests/fixtures/polio_afp_es/fit.toml:42`); 28 occurrences in inline test configs in 14 Rust files (`git grep -c -E 'rw_sd *= *[0-9]\|perturb_only_at_t0 *= *true' -- rust`) |
| the same, in sibling repositories (tracked files)                    | `camdl-book`: 17 `rw_sd =` in 2 files (`guide/fitting/fits/poisson_free_ic.toml`, `reference/inference-spec.qmd`); `camdl-vignettes`: 77 in 12 files; `ebola-bdbv-camdl`, `camdl-garki`: none                                                                                                        |
| an estimated initial-state-only parameter under IF2 without the flag | newly refused (§4.6), with the one-line fix; the `camdl-book` profiles of `I0` and `R_init` (`identifiability.qmd:402`, `:485`) are such cases                                                                                                                                                       |
| `profile --rw-sd`                                                    | 35 occurrences in 20 test files (`git grep -c -- '"--rw-sd"' -- rust`); `camdl-book` in 5 files; `camdl-vignettes` 14 in 6 files, 7 of them `--rw-sd auto`, already refused by roles D5; `camdl-garki` 3 in 1 file                                                                                   |

Sibling counts are from
`git ls-files | xargs grep -h -o -E 'rw_sd *=|rw_sd=|--rw-sd'` (and `-l` for
files) in each repository, excluding `_freeze/` and `results/`. The sibling
TOMLs that carry `rw_sd` use the schema before `[method]` (`[stages.scout]` in
the book, `[scout]` in the vignettes; `grep -l '^\[method\]'` over the
vignettes' rw_sd-carrying TOMLs → none), so they already fail to load on main
(`detect_legacy_stages`, `config_v2.rs:2289`). What this change newly breaks in
the sibling repositories is mainly their `--rw-sd` command lines. Stored fits
are not refused (§7.2).

### 9.2 Code and documents that change

- The sim API (§4.2): 76 struct literals in 36 files on main construct an
  `EstimatedParam` (`git grep -c 'rw_sd_auto:' -- rust`), 80 on the branch.
- `fit_state.toml` loses `[rw_sd]`; the 16 committed `camdl-vignettes` states
  that carry it still load.
- The profile diagnostic `cooling_final` is in z (§4.2).
- User documents
  (`git grep -c -E 'rw_sd|rw-sd|perturb_only_at_t0' -- docs ':!docs/dev'`):
  `camdl-inference-spec.md` (35), `inference.md` (16), `camdl-run-spec.md` (15),
  `camdl-language-spec.md` (6), `methods/cooling.md` (6), `fit-toml.md` (4), and
  `docs/agents.md`. `inference.md:567` ("rw_sd is on the natural scale") is
  replaced. `camdl-language-spec.md` is edited by hand and never run through
  `dprint`, which breaks its doctests (CLAUDE.md, "Markdown").
- The `camdl profile` help text and examples (`args/mod.rs:1669-1689`).

### 9.3 `docs/fit-toml.md`, "Step sizes"

> **Step sizes are settings of the method, in z.** IF2 perturbs each estimated
> parameter by a random walk; PMMH and MH propose around the current value. Each
> step is a table of the method that takes it, in z, the unconstrained
> coordinate the method moves in:
>
> ```toml
> [method.perturbation] # algorithm = "if2"
> beta = 0.1
> I0 = { sd = 0.2, initial_state = true } # perturbed at t = 0 only
>
> [method.proposal] # algorithm = "pmmh" or "mh"
> beta = 0.5
> ```
>
> A parameter you leave out takes a default chosen from its domain. Under IF2
> the defaults are pending an extended benchmark: currently 0.2 in z on a
> log-interval or one-sided map and 2/3 on an interval (logit-type) map. Under
> PMMH, MH and PGAS's MH-within-Gibbs the default is 0.25. An unbounded `real`
> or `instant` has no default. A `[method.proposal]` value is exactly the
> proposal SD: no factor is applied. With `adapt = true` it is the proposal's
> shape until `adapt_start`, after which adaptation reshapes it; with
> `adapt = false` it is the proposal for the whole run.
>
> **What a step of s in z means:**
>
> | map (from the kind and the box)                     | a step of s in z                                                                                  |
> | --------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
> | `Identity` (`real`, `instant`, no range)            | θ moves by s                                                                                      |
> | `LowerBounded` (a positive kind, no range)          | the distance to the bound is multiplied by about e^±s: s = 0.2 is about ±20%                      |
> | `Interval` (`probability`; [0, hi]; a signed box)   | at the midpoint θ moves by s·(hi − lo)/4; near an end the odds (θ − lo)/(hi − θ) multiply by e^±s |
> | `LogInterval` (a positive kind on [lo, hi], 0 < lo) | at the geometric midpoint ln θ moves by s·ln(hi/lo)/4                                             |
>
> On `Interval` and `LogInterval`, the natural-scale step a fixed z value means
> depends on the box: widening the bounds of a `LogInterval` parameter increases
> ln(hi/lo) and so the natural-scale step of the same z value. After changing
> bounds, check the natural-scale step the launch table prints.
>
> **Porting from pomp.** pomp's `mif2` applies `rw.sd` on its transformed
> (estimation) scale. That scale equals camdl's z only where the maps coincide:
> pomp `log` on an unboxed rate (camdl `LowerBounded` at 0), pomp `logit` on an
> unboxed probability (camdl `Interval` on [0, 1]), and no transformation
> (`Identity`). There, copy the number. A boxed parameter has a different z. For
> pomp `log` on a camdl `LogInterval` [lo, hi], divide by L·s(1 − s), with L =
> ln(hi/lo) and s = ln(θ₀/lo)/L at your start θ₀. Example: β in [0.05, 1] at
> 0.25 has L = 3.00 and s = 0.537, so L·s(1 − s) = 0.745, and pomp's
> `rw.sd = 0.02` becomes `beta = 0.027`. In general, convert through the natural
> scale at θ₀: natural step = s_pomp × |dθ/d(pomp coordinate)| (θ₀·s_pomp for
> `log`), then sd_z = natural step / |dθ/dz| from the conversion table (§7.3).
> pomp's `ivp(s)` is `{ sd = s, initial_state = true }`. pomp's `pmcmc` proposes
> on the natural scale, so a `mvn_diag_rw` SD divides by |dθ/dz| directly into
> `[method.proposal]`.

### 9.4 Release notes

> **Step sizes moved from `[estimate]` to `[method]`, in z.** `rw_sd` set IF2's
> perturbation and one fifth of PMMH's and MH's initial proposal, on the natural
> scale. Each is now `[method.perturbation]` (IF2) or `[method.proposal]` (PMMH,
> MH), in z; `perturb_only_at_t0 = true` is `initial_state = true` inside the
> IF2 entry, and IF2 refuses an estimated parameter that enters only the initial
> state unless it is marked so. A config with the old keys is refused, and the
> message prints, per parameter, the z value that gives your old step at the
> start and the natural step the default would give. Deleting a line is not
> neutral: it switches that parameter to the default step. `pgas` and the `nl-*`
> optimizers never read `rw_sd`: delete it. `profile --rw-sd` is replaced by
> `--estimate` (which parameters) and `--perturbation NAME=SD` with
> `--initial-state NAME` (their IF2 steps). `[estimate].start` no longer changes
> a fit's fit-level hash, so fits that differ only in starts or steps share a
> fit segment. Stored fits made before this release are still listed and
> addressable; their old keys are set aside with a notice. Every fit re-keys at
> this release.

## 10. Before and after: one problem fitted by IF2 and by PMMH

The SIR recovery case (`tests/recovery/cases/sir/`), estimating β, γ and the
initial infected count `I0`. `beta : rate in [0.05, 1.0]` and
`gamma : rate in [0.01, 0.5]` take `LogInterval` and carry the model's priors
(`log_normal`, `model.camdl:14-15`); `I0 : count` has no prior in the model, so
`[estimate]` gives one, which PMMH requires (§3.7). Boxed to [1, 100], `I0`
takes `LogInterval` too.

Before, IF2:

```toml
[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "data/weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.25, rw_sd = 0.02 }
gamma = { bounds = [0.01, 0.5], start = 0.10, rw_sd = 0.01 }
I0    = { bounds = [1, 100], start = 10, rw_sd = 2.0, perturb_only_at_t0 = true,
          prior = { log_normal = { mu = 2.3, sigma = 1.0 } } }

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
refuses it) and

```toml
[method]
algorithm = "pmmh"
backend = "chain_binomial"
chains = 4
particles = 500
iterations = 5000
```

The two `[estimate]` tables differ, so today the two files have different
fit-level hashes.

After, both files share these tables verbatim:

```toml
[model]
camdl = "model.camdl"

[data.observations]
weekly_cases = "data/weekly_cases.tsv"

[estimate]
beta  = { bounds = [0.05, 1.0], start = 0.25 }
gamma = { bounds = [0.01, 0.5], start = 0.10 }
I0    = { bounds = [1, 100], start = 10,
          prior = { log_normal = { mu = 2.3, sigma = 1.0 } } }

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
I0 = { sd = 0.174, initial_state = true }
```

PMMH, carrying the old natural-scale proposal at the start (5 × the IF2 values;
§7.4):

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
default steps, the IF2 table shrinks to `I0 = { initial_state = true }` and the
PMMH file needs no `[method.proposal]`; both choices change the steps from the
old config's.

## 11. Sequencing

The support-keyed transforms series (i) is implemented on
`feat/support-transforms` (_branch_ `a1292b18` through `193525d9`, including the
§12.6 benchmark note
`docs/dev/notes/2026-10-29-support-keyed-transforms-benchmark.md`). Its series
(ii), which would read an explicit `rw_sd` in z (its §11 commits 11–12), has not
been written. The step series below takes its place on the same branch, so one
release changes the meaning of steps once. Every fit re-keys at that release
already, because the fit level folds the engine version, which includes the git
hash (transforms §13, verified there at `cas.rs:501-521` and
`version.rs:12-15`). No `ir/VERSION` bump and no golden change is involved.

**Four independently landable series.** Each has its own regression tests and
can land, or be reverted, without the others:

- **A. The step correction** (below): the rename, z units, no factor, the legacy
  refusal, resolved steps and their identity, and profile's reading of
  `[method.perturbation]`.
- **B. The plan series** (§6): `ResolvedProblem` and `ExecutionPlan`, and
  `burn_in`/`thin` resolution.
- **C. `param_surfaces` and the initial-state refusals** (§4.6). Its tests are
  the compiled-model classification tests and the two IF2 refusals.
- **D. The profile flags** (§8), in parameter-roles Stage 3.

A does not need B, C or D. C needs only A's `initial_state` key. D needs A and
parameter-roles Stage 3.

**Every structural commit names one test that proves fixed-seed outputs
unchanged:** `fixed_seed_fit_outputs_are_bit_identical`, in a new
`rust/crates/cli/tests/fit_fixed_seed_pin.rs`, modelled on `determinism_pin.rs`.

- It runs `tests/recovery/cases/sir` at seed 1 under IF2, `nl-sbplx`, PMMH and
  PGAS with default steps.
- It compares `draws.tsv`, every `chain_*` TSV, `chain_results.tsv`,
  `chain_evaluations.tsv` and `mle_params.toml` byte for byte, and every numeric
  field of `fit_state.toml` by value, against pins committed under
  `tests/fixtures/fit_pins/sir_seed1/`.
- It is the result-records proposal's fixed-seed check (its §9), shared rather
  than duplicated.
- A commit that intends a numerical change re-captures the pins, and the pin
  diff is that commit's subject.

Run IDs are not compared, because the engine version moves them. In series A
with default steps, every commit keeps the pins: the defaults and the default
PMMH initial proposal are unchanged.

**Series A, the step correction, on `feat/support-transforms`**, each commit
green under `make test`:

1. Delete `FitState.rw_sd`, `runner::auto_rw_sd` and the NLopt copy (§3.5):
   written, never read. Its own commit.
2. Add `ZSd`, `PerturbAt`, `Perturbation`, `If2Step` and `IF2Config.steps`;
   remove the three fields from `EstimatedParam`; move the readers of §4.2. The
   CLI fills the vectors from today's `rw_sd` values through the branch's
   `step_from_natural`, so results are bit-identical: an IF2 and a PMMH fit of
   `tests/recovery/cases/sir` at a fixed seed produce identical `fit_state.toml`
   and `draws.tsv` before and after. `if2.rs` is high-risk; this commit edits
   only its read sites.
3. `resolve_steps`, called beside `resolve_starts`; the `[method]` tables; the
   removal of `rw_sd` and `perturb_only_at_t0` from `EstimateSpecV2`; the
   formula-only refusal (§7.1) and `load_archived` (§7.2); the moved readers.
   Inline test configs move to the new tables.
4. **The D16 commit.** It adds the `FitState` schema tag, the
   `load_input`/`load_for_inspection` split with its callers (§7.5), and the
   rename of `perturb_only_at_t0_params` to `initial_state_params`, with the
   §7.5 tests. Result-records Stage 3 lands after this commit.
5. The identity change (§5.1–§5.3), with the §5.5 tests; the §3.7 test turns
   green here.
6. `load_with_model` and the converted values (§7.3, §7.4), with a test per row
   of §7.4 and per unavailable case.
7. `profile` reads `[method.perturbation]` from an IF2 `--fit`, ignores it under
   `nl-*`, and its base and method levels change (§5.4).
8. Documents (§9.2–§9.4) and the amended proposals' status lines (§13).

**Series C** follows A on any branch: `param_surfaces` with its compiled-model
tests, then the initial-state refusals and the `unused_by_likelihood` refusal,
each in its own commit with its red test.

**Series B, the plan series (§6)**, before parameter-roles Stage 2:
`ExecutionPlan` at the CLI boundary first, then one backend per commit, each
bit-identical at a fixed seed. Parameter-roles Stage 2 then builds
`ResolvedProblem` without starts or method settings, and its
`SettingForNonEstimated` refusal is used by `resolve_steps` unchanged.

**Landing order:**

1. the step series, on `feat/support-transforms`;
2. the plan series;
3. parameter-roles Stage 2;
4. parameter-roles Stage 3, which carries the profile flags;
5. the release.

Result-records Stages 0, 1 and 2 are independent of this proposal and may land
at any point. Its Stage 3 waits for commit 4 (the D16 commit), and its Stage 4
follows its own Stage 3.

**The profile flags (§8)** land in parameter-roles Stage 3, which already
replaces `--rw-sd` as a selector. The release that ships the step series is not
cut until Stage 3 has landed (D17). **`FilterSettings` (§6.4)** follows the
merge of the support-transforms branch.

## 12. Validation

- `ZSd::new` refuses NaN, ±∞, 0 and negatives; deserialization names the value.
- `PerturbationSpec` reads `beta = 0.1`, `beta = { sd = 0.1 }` and
  `beta = { initial_state = true }`, and refuses `beta = {}`,
  `beta = { sd = 0.1, lag = 2 }` and `beta = "auto"`, naming the accepted forms.
- `resolve_steps`, per rule of §4.4, including a simplex member under IF2 (its
  SD in `simplex`, not `params`), `initial_state` on a member (refused), and a
  member under PMMH (an ordinary proposal entry).
- `[method.proposal]` under `if2` and `[method.perturbation]` under `pmmh`,
  `pgas` and `nl-sbplx` are refused with the §4.3 messages.
- The legacy refusal: a test per row of §7.4; the §7.4 values to three
  significant figures; each unavailable case of §7.3; `simulate --fit` refuses.
- The stored-copy tests of §7.2.
- The identity tests of §5.5.
- `profile --algorithm nl-sbplx --perturbation beta=0.1` is refused.
- The bit-identity checks of §11 commit 2 and §6.3.
- `param_surfaces`: one test per surface of §4.6, and the two IF2 refusals.
- A legacy conversion above 3 in z, for a start near a bound, prints its reason
  and recommends the default (§7.3).
- `run.json`'s `inputs.steps` carries every estimated parameter's `sd_z`,
  `start`, `natural_at_start` and `from`, and two configs differing only in
  writing a default have equal method-level hashes and different `from` (§5.6).
- `profile` cell records carry `cooling_final` as the cooling factor (D21).

## 13. Amendments to other proposals

### 13.1 Support-keyed transforms (`2026-10-29-support-keyed-transforms.md`)

- **D9 is superseded.** An explicit step is not read in a new unit under its old
  name; it moves to the tables of this proposal. Series (ii) (§11 commits 11–12)
  is replaced by the step series of §11 here.
- §2 "Step": "In the fit config it is `rw_sd`" becomes "In the fit config it is
  `[method.perturbation]` (IF2) or `[method.proposal]` (PMMH, MH)".
- §4.1: `EstimatedParam` loses `rw_sd`, `rw_sd_auto` and `perturb_only_at_t0`;
  `natural_step` and `mh_initial_step` go (§4.2 here).
- §9.1, §9.2: an explicit step is the written table value, in z; the factor of
  five is removed (D5 here); "`rw_sd` stays IF2's, PMMH's and MH's knob" becomes
  "IF2 and PMMH/MH each have their own".
- §9.5: profile and survey no longer build steps through
  `build_if2_params_from_specs`; profile resolves its own (§5.4 here).
- §10: the `if2.rs` edits read `config.steps` instead of `spec.rw_sd` (§4.2
  here).
- §13: the series (ii) conversion bullets are replaced by §7.3 here.
- §9.1 and the launch table: `Transform::reference_z` (the box midpoint for the
  interval maps) is replaced by the resolved start as the conversion point
  (D19).
- `simplex_member_default_step` is replaced by the log-ratio constant of D20.

### 13.2 Workflow-first fit config (`2026-09-08-workflow-first-fit-config.md`)

- §5 "Kept as they are" and decision 12 say `perturb_only_at_t0` and `rw_sd` are
  inert for a non-IF2 method so that an IF2 file and a PGAS file share a
  fit-level hash. Both claims are false (§3.7 here).
- §9's follow-up "Relocate `perturb_only_at_t0` and `rw_sd` under `[method]`" is
  done by this proposal.

### 13.3 Parameter roles (`2026-10-26-parameter-roles.md`)

- §3.2 item 8 is false for `rw_sd` and out of date for `perturb_only_at_t0`
  (§3.4 here); it is re-checked and rewritten there.
- §4.1: "`--rw-sd`, `[estimate].rw_sd`, `perturb_only_at_t0` … do not declare a
  role" becomes "`--perturbation`, `--initial-state`, `[method.perturbation]`
  and `[method.proposal]` do not declare a role".
- §4.2's last table row becomes "`--perturbation`, `--initial-state`, or a
  `[method.perturbation]`/`[method.proposal]` key, for a non-estimated name |
  refused".
- §5.3: `[estimate].start` stays a source of starts and is hashed at the method
  level (§5.2 here).
- §6.2: `Estimated` loses `start`. A written `[estimate].start` goes to the
  chain-starts input (the base points of `DrawnStarts`, held by
  `ExecutionPlan.starts`), so no start reaches `ResolvedProblem`. `Estimated`
  stays the one holder of support, search box, transform and prior.
- §6.3: `RoleInputs.method` is deleted, so `resolve_roles` and `ResolvedProblem`
  are method-free by construction. The method-dependent refusals
  (`UnboundedForMethod`, D8's `HyperOverHeldLeaf`, `HierarchicalLeafBoxed`) move
  to `ExecutionPlan::new`, with the `ic_free` precondition and the initial-state
  checks.
- §6.3: `ResolvedProblem` loses `starts` and `algorithm` and gains `model`,
  `streams`, `simplex_groups`, `objective` and `window`; `resolve_problem` loses
  its `starts`, `algorithm`, `n_chains` and `seed` arguments;
  `If2ParamSettings`, `RwSdFrom` and `AlgorithmSettings` are deleted;
  `ExecutionPlan` and `MethodPlan` are added (§6 here).
- §6.4 "IF2 settings": replaced by §4.2 and §4.4 here;
  `EstimatedParam::from_parts` takes no settings argument.
- §6.6: each command builds one `ResolvedProblem` per fit segment (per sweep
  point or profile cell) and one `ExecutionPlan` per method and seed.
- §7.1: the launch block's `rw_sd gamma = 0.0138 (auto)` line becomes
  `perturbation gamma = 0.2 (default, z)`, with the natural-scale equivalent.
- §7.2: the record's `per_param` entry becomes
  `"perturbation": {"gamma": {"sd": 0.2, "at": "every_observation"}}` under IF2
  and `"proposal": {…}` under PMMH and MH, each with its unhashed `from`.
- §7.3: the `profile --sweep k … --rw-sd beta=0.02` example becomes
  `--estimate beta,gamma --perturbation beta=0.1`.
- §10.1: "`--rw-sd` is optional, IF2-only, and sets step sizes only" becomes
  "`--rw-sd` is removed; `--perturbation` and `--initial-state`, refused under
  `nl-*`".
- §10.2: the `profile --rw-sd …` test migrations become `--estimate` plus
  `--perturbation`.
- §10.3, `profile`: the method level hashes the resolved steps, and the base
  level the problem digest (§5.4 here). §10.3, `fit run`: "nothing re-keys
  beyond the engine version" gains "steps and starts move from the fit level to
  the method level".
- §11.2 Stage 2 builds on the plan series (§11 here); Stage 3's "`--rw-sd`
  optional and IF2-only" becomes "`--rw-sd` removed; `--perturbation` and
  `--initial-state`", and its test "`--sweep k` + `--rw-sd beta=0.02`" becomes
  "`--sweep k` + `--perturbation beta=0.1`".
- D5: `--rw-sd` is refused in every form.
- §13: "The `rw_sd` scale convention" and "PMMH's own proposal tuning (replacing
  `rw_sd × 5`)" leave the list, resolved here.

## 14. Decisions

**D1 — A new name per algorithm, not a new unit for `rw_sd`.** Maintainer
decision 1 (§1).

**D2 — Each table is a field of its algorithm's variant.** Location is
maintainer decision 2; the variant-field form is chosen here. serde cannot
reject an unknown field of the tagged enum, so a misplaced table is still
refused by a rule, `validate_method_keys`; what the variant field buys is that
the parsed `Method` has no place to hold an IF2 setting on a PMMH run, so no
downstream code can read one.

**D3 — The unit is z, typed `ZSd`; for a simplex member, its log-ratio
coordinate.** z is the coordinate both kernels act in, pomp's `mif2` unit
(§3.6), and the unit of the D8 defaults. The type makes a non-positive or
non-finite value unrepresentable after load.

**D4 — The initial-state flag is a key inside the perturbation entry.** One
entry per parameter keeps its IF2 settings in one place, as pomp's `ivp()` is
part of the parameter's `rw.sd` entry. A separate list was considered; it
matches the `--initial-state` flag's shape but splits one parameter's settings
across two places. The key is `initial_state` (D18).

**D5 — `[method.proposal]` is exactly the proposal SD the user writes, and
defaults to `default_mh_step`** (maintainer decisions 3 and 8; 0.25 in z,
_branch_ `types.rs:83`). No factor is applied anywhere. The legacy message
prints 5 × the converted value to reproduce the old initial proposal scale.

**D6 — PGAS gets no per-parameter table.** Maintainer decision 3. Its default
update, NUTS, has no per-parameter proposal; its MH-within-Gibbs starts at
`default_mh_step` and adapts by Robbins–Monro (_branch_ `pgas.rs:3872-3886`).
`[method.proposal]` under `pgas` is refused with the §4.3 message.

**D7 — The method level hashes the resolved steps, and the written
`[estimate].start` values (`starts_base`) beside the starts rule; `from` is
recorded in the run record's unhashed `inputs` and never hashed.** A written
default and an omitted name key the same, as `starts` does (§5.3). The run
record still says which the user wrote, as it records `argv` (§5.6). The start
moves to the method level by maintainer decision 5 (§5.2).

**D8 — Live configs with the old keys are refused; stored copies are set aside
with one notice.** Refusing a stored copy would make every pre-release fit
unlistable; accepting a live one would let the file circulate. This follows
`ef81ade7` (§7.2).

**D9 — `profile` reads `[method.perturbation]` from an IF2 `--fit`, and nothing
else from `[method]`; under `nl-*` it reads no step.** Profile already reads
these settings from `[estimate]` (`profile.rs:878-891`), so not reading them
would perturb an initial-state parameter at every observation without saying so.
Profile's particles, iterations and cooling stay its own flags.

**D10 — `fit_state.toml` drops `[rw_sd]`.** Written and never read (§3.5).

**D11 — The step series replaces transforms series (ii) on its branch; the plan
series precedes roles Stage 2; the profile flags land in roles Stage 3.**

**D12 — `ResolvedProblem` and `ExecutionPlan` are separate types** (maintainer
decision 4; §6). New names avoid `fit::methods::InferenceMethod`
(`fit/methods.rs:50`) and the two `InitSource` enums
(`cli/src/fit/chain_starts.rs:45`, `sim/src/error.rs:352`).

**D13 — `FilterSettings` leaves `IF2Config` in a later stage** (§6.4), after the
support-transforms branch merges, because it edits `if2.rs` and changes no
result.

**D14 — Simplex members.** Under IF2 a member's step lives only in its
`SimplexGroup` (`If2Step::SimplexMember`), and `initial_state` on a member stays
refused. Under PMMH and MH a member is an ordinary proposal entry with
`default_mh_step` on its own map (§4.4 rules 7–8).

**D16 — An unsupported old `fit_state.toml` is refused as an input and read with
a notice for inspection** (maintainer decision; §7.5).

**D15 — `beta = {}` is refused.** An empty entry has no SD and no flag, so it
states nothing that omitting the line does not.

**D17 — The release waits for parameter-roles Stage 3** (maintainer decision 7).
`profile` gets `--estimate`, `--perturbation` and `--initial-state` in the same
release that removes `--rw-sd`, so no released `profile` has two step units. No
interim precedence rule exists.

**D18 — The flag is `initial_state`; IF2's timing is `PerturbAt::TimeZeroOnly`**
(maintainer decision 9; §4.5, §4.6). A future `timing = "initial"` schedule key
is noted, not designed.

**D19 — Conversions happen at the resolved start value: the explicit start, then
the resolved default start, then the domain midpoint** (maintainer decision 10;
§7.3). Converted values above 3 in z are reported with their reason and not
recommended.

**D20 — A simplex member's default IF2 step is one log-ratio constant, 1/6.**
That is the value a member on [0, 1] receives today (`(hi − lo)/6`, _branch_
`runner.rs:1529`). Members boxed otherwise get the same constant in place of a
natural-scale formula read on the log-ratio scale (§3.8).

**D21 — `profile`'s `cooling_final` is the cooling factor of the last
iteration.** The mean of per-parameter SDs mixed units across parameters. The
factor is the quantity the summary meant ("how cool the perturbation got"), and
it is the same for every parameter.

**D22 — An incorrect initial-state schedule is refused under IF2** (maintainer
decision 11; §4.6). The IR determines the property reliably by an exhaustive
walk, so the check is a refusal, not a warning. Non-IF2 methods never warn.

## 15. Out of scope, each with its reason

- **Command-line step overrides for `fit run`.** `fit run` overrides no
  per-algorithm setting today (`args/mod.rs:668-750`: `--starts`,
  `--decibans-thresh` and run controls). Adding `--perturbation` there is a
  decision about `[method]` overrides in general.
- **A dense initial PMMH covariance.** The table is diagonal, as the sampler's
  initial proposal is (`sim/src/inference/pmmh.rs:664`); Haario adaptation
  learns the correlation.
- **IVP lags (pomp's `ivp(s, lag)`).** No model in this repository or the
  sibling repositories needs one; the entry leaves room for a `lag` key.
- **`Assessment`, `ResultRecord` and the `[assessment]` table** (§6.2): designed
  in `2026-10-29-result-records-and-assessment.md` §4.
