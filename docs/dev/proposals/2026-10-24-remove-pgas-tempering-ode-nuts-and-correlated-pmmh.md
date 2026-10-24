# Remove PGAS tempering, NUTS on the ODE backend, and correlated PMMH; archive each under a git tag

- Date: 2026-10-24
- Status: accepted by the maintainer on 2026-10-24: all three cuts and the PGAS
  fit re-key confirmed; the `camdl batch` removal is its own proposal,
  `2026-10-24-batch-to-simulate.md`. One consequence surfaced while writing the
  correlated-PMMH section and has not yet been put to the maintainer: removing
  `rho` also re-keys every PMMH fit and every PMMH `camdl profile` run (§5.6).
- Archive tags: `archive/pgas-tempering`, `archive/ode-nuts`,
  `archive/correlated-pmmh` (§7).
- Relates-to: gh#789, gh#569, gh#551, gh#917, gh#390, gh#921, gh#767, gh#102,
  gh#167, gh#958, gh#220, gh#335; `2026-10-24-batch-to-simulate.md` (the batch
  removal); `2026-06-15-ode-gradient-inference.md` and
  `2026-06-26-ode-nuts-gradient-spine.md` (the proposals that introduced the ODE
  gradient stack and NUTS on it);
  `2026-09-27-coordinate-kind-design-vs-sample.md` (accepted; owns scenario
  algebra); `2026-10-04-effect-fire-time-authority.md` (draft; its PGAS step
  touches the sweep this proposal simplifies, and its clock table has a
  correlated-PMMH row); the note
  `2026-10-06-run-grid-workflows-and-composition.md` (the usage survey;
  uncommitted at the time of writing);
  `docs/dev/incidents/2026-08-10-pgas-tempering-swap-sign.md`; the notes
  `2026-06-08-correlated-pf-degenerate-loglik.md` and
  `2026-08-28-cpm-irregular-observation-grid.md`.
- Required reading: `.claude/rules/run-identity.md` (the `tempering` and `rho`
  fields are fit identity; §3.6, §5.6), `.claude/rules/sim-and-inference.md`
  (every backend × method cell works and is tested or fails loudly; §1),
  `VERSIONING.md` (alpha posture: rename or remove atomically, no shims).
- Evidence: a read-only inventory of main `bff4cec4` taken on 2026-10-24 for §1
  and §3; for §4 and §5, greps and reads of the same tree on 2026-10-24. Every
  `path:line` below is from one of those two; no test was run.

## 1. Why: the bugs come from cells, and every cell removed is a copy removed

camdl's surface is a matrix: three forward backends, about ten inference
methods, many DSL features, many verbs. The rule in `sim-and-inference.md` is
that every cell of that matrix either works and is tested or fails loudly. That
rule is right, and it has a price: each surface is a tested cell per backend and
per method, and each shared concern (when an effect fires, how it is applied,
which chain a number names, what `dt` is, what the time tolerance is, which
posterior stage is "the" posterior, which parameter source wins) is implemented
once per cell. The copies drift. Most of the correctness bugs found in October
2026 are of exactly that shape:

| concern                       | copies on main              | issues         |
| ----------------------------- | --------------------------- | -------------- |
| when an effect fires          | 3                           | gh#957, gh#958 |
| how an effect is applied      | 3 readings                  | gh#985         |
| the chain number a user sees  | 81 hand-written `+ 1` sites | gh#980         |
| the step size `dt`            | 3 to 4 structs              | gh#923, gh#966 |
| time tolerance                | 3 epsilons                  | gh#606, gh#964 |
| which posterior stage is read | 2 verbs, 2 answers          | gh#609         |
| parameter-source precedence   | one guard per pair          | gh#571, gh#572 |

Two things follow. The shared concerns have to be extracted into one place each,
which the fire-time and coordinate-kind proposals do for time, effects and
parameter sources. And the matrix has to match the people who use it. With one
maintainer, a surface with no users is not free: it is a cell in every future
seam extraction, a row in every future proposal's table, and a place for the
next silent disagreement.

What people actually run, counted from fit configs in the book and in six
external modelling projects on the maintainer's machine, counted in aggregate
(2026-10-24). The survey includes `camdl-garki`, an ODE project added on
2026-10-24: 45 `mh` and 12 `nl-sbplx` configs, `burnin_dt` under `mh`, nothing
on the cut list.

| method or surface                    | book configs | project configs     | notes                                  |
| ------------------------------------ | ------------ | ------------------- | -------------------------------------- |
| IF2                                  | 38           | 10                  | the teaching and profiling method      |
| PMMH                                 | 11           | 46                  | two projects                           |
| PGAS                                 | 2            | 207                 | one project, touched 2026-10-20        |
| `mh`, `nl-sbplx`, `nl-bobyqa` on ODE | 0            | 87                  | derivative-free; 57 of them from garki |
| `nuts` on ODE                        | 0            | 1                   | a synthetic self-check (§4.2)          |
| `nl-lbfgs`                           | 0            | 1                   | a labelled demonstration; stays (§4.5) |
| PGAS `tempering` ladder              | 0            | 0 live (2 archived) |                                        |
| correlated PMMH (`rho`)              | 0            | 1                   | 2026-09-10, `rho = 0.99` (§5.2)        |

## 2. Summary

| surface                                                                                                          | decision      | size that goes                                                                                                                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PGAS tempering (`tempering` ladder, swaps, `LowSwapRate`, `--tempering`)                                         | **cut** (§3)  | ≈550 lines of sim `pgas.rs`, ≈150 CLI, ≈1,000 tests, plus ≈150 mechanical edits removing the β parameter                                                                                                                             |
| NUTS on the ODE backend (`algorithm = "nuts"`)                                                                   | **cut** (§4)  | ≈1,800 lines: `ode_nuts.rs` (760), `fit/nuts.rs` (647), the warm-up half of `nuts.rs`, one method row, one test file and two test cells; the ODE gradient itself stays                                                               |
| correlated PMMH (`rho`)                                                                                          | **cut** (§5)  | `correlated_pf.rs` (1,150), the pre-drawn-noise arms of the chain-binomial kernel and of the initial-state draw, the correlated branch of the PMMH driver and of the noise preflight, two CLI flags, six test files, two test groups |
| ODE gradient (`det_grad`), `nl-lbfgs`, `burnin_dt`, reactive interventions, Gillespie, IF2, PMMH, PGAS with NUTS | **kept** (§6) | 0                                                                                                                                                                                                                                    |

The removal of `camdl batch` was the first cut in an earlier draft of this
document. It moved to its own proposal, `2026-10-24-batch-to-simulate.md`,
because it transfers a user-facing capability (parameter sweeps) to `simulate`
and is reviewed separately. The three cuts here remove inference surface that
has no replacement to build, which is why they share one document.

## 3. Cut 1: PGAS tempering

### 3.1 What it is

`[method] tempering = [β₁, β₂, …]` under `algorithm = "pgas"` runs one PGAS
chain per inverse temperature β with replica-exchange swaps between neighbouring
rungs; the first rung is snapped to β = 1 (cold). Default `[1.0]`, one rung, no
swaps. With it come the `--tempering` CLI override, the `LowSwapRate`
diagnostic, `swap_acceptance_rates` in the result and the per-chain NUTS
diagnostics, the observation-only tempered target from gh#551 (PR #945), and the
incident `2026-08-10-pgas-tempering-swap-sign.md`. There is no `n_rungs` option;
that name is a local, `config.tempering.len().max(1)` (`pgas.rs:4267`).

### 3.2 Users

No live config sets a ladder. The two that do are archived (both
`[1.0, 0.6, 0.3, 0.1]`, in one project's `archive/` directory). The book
describes it (`inference/algorithms.qmd:104-121` "Parallel tempering",
`inference/likelihood.qmd:456, 469`, `inference/diagnostics.qmd:26, 39`) and
never runs it.

### 3.3 Why cut rather than keep

It was experimental and reached main without a user. Its only open bug (gh#789,
trace rows mixing two rungs) and its only incident exist solely for more than
one rung. It puts a rung loop and a temperature parameter through the
highest-risk file in the repository, and the fire-time implementation has to
step around both in its PGAS stage. The gh#551 observation-only tempered target
goes with it; that is a deliberate loss, named here so it is not rediscovered as
a regression.

### 3.4 Delete

- `rust/crates/sim/src/inference/pgas.rs`: `PGASConfig.tempering` (`:64-76`);
  `InverseTemperature::new`, `parse_ladder`, `COLD_RUNG_TOLERANCE`
  (`:1048-1089`); `PGASResult.swap_acceptance_rates` (`:1210-1216`);
  `swap_log_alpha` and its tests (`:3641-3838`); ladder setup and log line
  (`:4266-4273`); heated-rung resume comment and warning
  (`:4276-4290,
  4327-4333`); `swap_proposed`/`swap_accepted`
  (`:4335-4336, 4343`); the replica-exchange block (`:4842-4898`); swap-rate
  logging (`:4926-4948`); `swap_acceptance_rates` computation
  (`:4995-5003, 5012`); the ladder tests in `rung_lik_tests` (`:5479-`) and the
  β = 0.3 assertions in `theta_proposal_score_tests` (`:5413, 5444`).
- `rust/crates/sim/src/inference/diagnostic.rs`: `DiagnosticKind::LowSwapRate`
  (`:337-344, 439-440, 578-580, 634-637`; serialized tag `low_swap_rate`).
- CLI: `rust/crates/cli/src/args/mod.rs:863-870` (`--tempering`) and
  `parse_f64_list` (`:943-954`, no other user);
  `rust/crates/cli/src/fit/mod.rs:276-286, 293`;
  `rust/crates/cli/src/fit/config_v2.rs` `Algorithm::PGAS.tempering`
  `:1047-1056`, overrides `:1609, 1613`, `CliStageOverrides.tempering` `:1685`,
  `default_pgas_tempering` `:1805`, tests
  `:3374-3375, 3491, 3787-3790, 6826,
  6885-6894`;
  `rust/crates/cli/src/fit/pgas.rs` `PgasStageOpts.tempering` and ladder
  validation `:201, 228-261`, `ChainNutsDiag.swap_acceptance_rates`
  `:570, 1238`, the `PGASConfig` field `:688`, the `LowSwapRate` emission
  `:1499-1508`, tests `:2079-2146`;
  `rust/crates/cli/src/fit/config_diff.rs:719-737`; stale comment
  `fit/cas.rs:486`.
- Tests: `rust/crates/sim/tests/pgas_tempering.rs` (keep
  `test_single_rung_deterministic` as the §10 pin, renamed),
  `pgas_tempering_exact.rs`, `gradient_check_obs.rs:570`
  (`gh551_tempered_energy_and_gradient_match_fd`),
  `csmc_exact_invariance.rs:409`
  (`one_sweep_leaves_the_tempered_smoothing_target_invariant`).
- Docs: `docs/fit-toml.md:336, 413-441`; `docs/workflow.md:235, 238`;
  `docs/agents.md:213`; `docs/camdl-run-spec.md:2657, 3320, 5679, 6272`; the
  book sections in §3.2. The incident report stays as history.

### 3.5 Remove the temperature parameter, not just the ladder

`InverseTemperature` is threaded as a parameter through `splice_log_ratio`
(`pgas.rs:2591`), `csmc_as` (`:2849`), `pgas_grad::complete_data_loglik_grad`
(`obs_weight`, `pgas_grad.rs:466, 566`) and `RungLik::log_target(β)`
(`:2030-2076`, used by the θ accept at `:4703`). Keeping it, always `COLD`,
would leave a parameter with one legal value in the highest-risk file: a flag
that means nothing and that every reader has to prove is inert. Decision: remove
it. `log_target(β)` collapses to `total`; the `rungs: Vec<RungState>` loop
(`:4298-4316, 4463-4791`) flattens to one chain state (`RungState` is the cold
chain's whole state: params, NUTS mass and step, Welford accumulators, MH
standard deviations, so it is renamed, not deleted); `rung_accepted`,
`rung_csmc_diag` (`:4453-4455`) and the `rungs[0]` reads (`:4950-4992`) lose
their index. About twenty sim test files pass `InverseTemperature::COLD`
(`gradient_check.rs`, `gradient_check_overdisp.rs`, `gradient_check_obs.rs`,
`csmc_splice_ratio_oracle.rs`, the eight `csmc_*` files, `gate_licm_ab.rs`,
`pgas_exact_tiling.rs`, `pgas_no_ancestor_sampling.rs`,
`binomial_selection_reaches_draws.rs`, `zz_cond_systematic_probe.rs`) and eleven
build `PGASConfig { tempering: vec![1.0], .. }`; those edits are mechanical and
go in their own commit (§9).

### 3.6 Identity: this re-keys every PGAS fit

`tempering` is part of the PGAS stage identity: the canonical fit payload golden
at `config_v2.rs:7042` contains `"tempering":[1.0]`, and its own message says a
change "invalidate[s] every existing resume_state.bin". Removing the field
therefore re-keys every PGAS fit leaf, so no new PGAS run deduplicates against
an existing one and no in-flight PGAS fit can resume. Stored results stay
readable. Per `run-identity.md` this is a deliberate, versioned re-key: the
implementer bumps the fit-config identity level per `VERSIONING.md`, moves the
identity golden in its own reviewed commit, and lands nothing else in that
commit. A `fit.toml` that still says `tempering = [1.0]` is refused with an
actionable message naming the removed field and the fix (delete the line), in
the shape of the removed-flag traps at `args/mod.rs:115-125`; no alias, no
silent acceptance. The re-key is the price of the cut; the maintainer confirmed
it on 2026-10-24.

### 3.7 Issues

Close gh#789. gh#569 names the `pgas_tempering` and `swap_forms_agree` tests in
its body but its defect is the θ-block accept call site; it stays open and is
relabelled. gh#710 and gh#172 matched the word "rung" only and are unaffected.

### 3.8 Archive tag

`archive/pgas-tempering`, on the last main commit that carries the ladder (§7).

## 4. Cut 2: NUTS on the ODE backend

### 4.1 What it is

`algorithm = "nuts"` with `backend = "ode"` runs the no-U-turn sampler on the
deterministic ODE marginal likelihood, with the gradient supplied by forward
sensitivities: `det_grad` integrates the ODE together with one extra sensitivity
system per parameter and returns the log-likelihood and its gradient in one call
(`rust/crates/sim/src/inference/ode_grad.rs`, 1,366 lines). It arrived as gh#275
Phase 2 (commit `951dc5c9`, 2026-07-07), designed in
`2026-06-15-ode-gradient-inference.md` and
`2026-06-26-ode-nuts-gradient-spine.md`.

It owns `rust/crates/sim/src/inference/ode_nuts.rs` (760 lines) and
`rust/crates/cli/src/fit/nuts.rs` (647), the `Algorithm::Nuts` configuration
(`config_v2.rs:1205-1250`: `chains`, `warmup`, `samples`, `max_tree_depth`,
`target_accept`, `dense_mass`, `burnin_dt`), and its registration in
`rust/crates/cli/src/fit/methods.rs:167, 235, 348` (the last is the
chain-binomial refusal that steers a `nuts` request to PGAS). In the shared
`rust/crates/sim/src/inference/nuts.rs` (877 lines) it owns the warm-up half:
`MassMetric` (`:517`), `warmup_schedule` (`:537`), `WarmupAdapter` (`:582`) and
`mod warmup_tests` (`:746-`). A grep of `rust/crates` finds no use of those
three names outside `ode_nuts.rs` and `cli/src/fit/nuts.rs`. PGAS's NUTS
parameter move uses only `nuts_step` (`:192`), `MassMatrix` (`:23`),
`NUTSConfig` (`:160`), `DualAveraging` (`:476`) and the leapfrog and tree
internals, and adapts its own mass matrix inline with Welford accumulators
(`pgas.rs:4550-4659`; `MassMatrix::dense_from_covariance` at `:4627`,
`MassMatrix::diagonal` at `:4655`). The two halves of `nuts.rs` separate
cleanly.

### 4.2 Users

One config on the maintainer's machine, a synthetic self-check; nothing in the
book. The two ODE projects in the survey fit with `mh` and `nl-sbplx` (87
configs, §1).

### 4.3 Why cut rather than keep

The measured case against it is cost, not correctness. Metropolis-Hastings pays
one ODE solve per step; forward-sensitivity NUTS pays one-plus-parameters solves
per leapfrog step and several leapfrog steps per draw. The DELENDA comparison of
2026-07 put that into effective samples per unit compute and found
forward-sensitivity NUTS dominated by adaptive MH at every dimension tried; only
an adjoint gradient, about twice a solve regardless of parameter count, changes
the picture, and none exists. That comparison lives in project notes outside
this repository, not under `docs/dev/notes/`, which is itself the gap §8 closes.

The cost result applies to the sampler, not to the optimiser. A sampler pays the
gradient a million times on its way to a posterior; an optimiser pays it a few
hundred times on its way to a maximum, where a dominated per-evaluation cost is
still a good trade against a derivative-free method's iteration count. So the
cut is the sampler alone, and the gradient it ran on stays (§4.5).

The second reason is the matrix. Removing an algorithm removes a whole row of
the backend × method matrix that every new feature must work on or refuse: the
fire-time proposal, every observation-model change, every new DSL construct that
touches the state Jacobian. gh#390 (event-jump sensitivities for ODE-NUTS) is an
example of a row-specific feature that would otherwise have to be built.

This is a moderate simplification, about 1,800 lines plus a method row, not a
large one: the gradient machinery, the capability check and the OCaml autodiff
stay because `nl-lbfgs` needs them.

### 4.4 Delete

- `rust/crates/sim/src/inference/ode_nuts.rs`, whole file, and its `pub mod`
  line in `inference/mod.rs`.
- `rust/crates/cli/src/fit/nuts.rs`, whole file.
- `rust/crates/sim/src/inference/nuts.rs`: `MassMetric` (`:517`),
  `warmup_schedule` (`:537`), `WarmupAdapter` (`:582`), `mod warmup_tests`
  (`:746-`). `nuts_step`, `MassMatrix`, `NUTSConfig`, `DualAveraging` and the
  leapfrog and tree code stay for PGAS.
- `rust/crates/cli/src/fit/config_v2.rs`: `Algorithm::Nuts` (`:1205-1250`), its
  defaults, and the `needs_state_grad` arm that names it (`:2380-2388`; the
  `nl-lbfgs` arm stays).
- `rust/crates/cli/src/fit/methods.rs`: the `nuts` registry row (`:167`), the
  parse arm (`:235`), the chain-binomial steer (`:348`), the `nuts` wording in
  the capability messages (`:151, 310`); the test `nuts_on_ode_is_supported`
  (`:1345`) goes; `nuts_on_stochastic_backend_steers_to_pgas` (`:1354`) is
  rewritten so that `nuts` on any backend is an unknown-algorithm error whose
  message says that PGAS's parameter move is NUTS (`use_nuts`) and that `mh` is
  the ODE sampler.
- Comments that name `run_ode_nuts` or `ode_nuts`: `pgas.rs:3566`,
  `nuts.rs:580`, `types.rs:696`.
- Tests: `rust/crates/cli/tests/nuts_ode_recovery.rs`, whole file; the `nuts`
  cell of `coarse_burnin_ode.rs` (its fixture at `:131`; the `mh` cell at `:274`
  stays); the `"nuts"` entry in `gh900_every_method_writes_progress.rs:223` and
  its fixture at `:170`. `state_grad_gating.rs` stays, with its header (`:4`,
  "read only by `fit --method nuts`") reworded to name `nl-lbfgs`.
- Docs: every passage that presents `nuts` as a method on the ODE backend. A
  grep for `nuts` across `docs/*.md` and `README.md` returns fourteen files;
  each hit is sorted by the implementer into ODE-NUTS (goes) or PGAS's NUTS move
  (stays). `docs/camdl-language-spec.md` is edited by hand, never formatted.

### 4.5 What stays, because `det_grad` and `nl-lbfgs` need it

- `rust/crates/sim/src/inference/ode_grad.rs` (1,366 lines; forward
  sensitivities), including its finite-difference oracle tests.
- The `nl-lbfgs` arm of `rust/crates/cli/src/fit/nlopt_stage.rs` (`:611, 996`)
  and `camdl profile --algorithm nl-lbfgs` (`profile.rs:768-780`). `nl-lbfgs` is
  NLopt's quasi-Newton optimiser fed by `det_grad`.
- `preflight_gradient_ode`, the gradient capability check
  (`gradient_capability.rs:233-510`), and the `needs_state_grad` switch
  (`config_v2.rs:2380-2388`, `util.rs:410-482`) that decides whether the
  compiler's state-Jacobian fields are read.
- The OCaml state-Jacobian autodiff (`differentiate_rate_state`
  `autodiff.ml:829`, `differentiate_projection` `:852`, `differentiate_ic`
  `:950`) and the IR fields they emit (`transition.rs:144`,
  `observation.rs:364`, `model.rs:453`). No IR change, so no `ir/VERSION` bump.
- `burnin_dt`, the coarse integration step for the unscored warm-up before the
  first observation: `Algorithm::Mh` carries it too (`config_v2.rs:1173`), the
  garki project uses it under `mh`, and `validate_burnin_dt` (`:1833`) is
  shared.
- Tests: `nl_lbfgs_ode_recovery.rs`, `state_grad_gating.rs`, the `mh` cell of
  `coarse_burnin_ode.rs`.

### 4.6 Identity

Removing a method removes its own fit leaves and changes no other method's
identity payload; `burnin_dt` on `mh` is identity-defining and untouched. No
identity bump.

### 4.7 Issues

Close gh#390 (ODE-NUTS event-jump sensitivity). gh#917 (`preflight_gradient_ode`
names `nuts` and recommends `mh` even when `nl-lbfgs` is the caller) stays open
and gets simpler: with one caller left, the sixteen messages name `nl-lbfgs` and
recommend `nl-sbplx`. gh#921 (rank-1 state Jacobian) stays; its "NUTS and ODE
forward-sensitivity" clause loses the first half, and the consumers it must keep
bit-identical are `det_grad` and the PGAS gradient.

### 4.8 Archive tag

`archive/ode-nuts`, on the last main commit that carries `ode_nuts.rs` (§7).

## 5. Cut 3: correlated PMMH

### 5.1 What it is

`rho` under `algorithm = "pmmh"` (`Algorithm::PMMH { rho: Option<f64> }`,
`config_v2.rs:1133`) runs the correlated pseudo-marginal variant of
particle-marginal Metropolis-Hastings, in
`rust/crates/sim/src/inference/correlated_pf.rs` (1,150 lines; the header cites
Deligiannidis, Doucet & Pitt 2018, JRSSB). Every random number one particle
filter evaluation consumes is pre-drawn as a standard normal (`PFRandomState`:
the gamma, binomial, resampling and initial-state blocks) and transformed at the
point of use; successive evaluations are coupled by the Crank–Nicolson update
`u' = ρu + √(1−ρ²)z`, so that the likelihood _ratio_ in the acceptance step has
lower variance than the two estimates do. Resampling is systematic with one
correlated uniform per observation (`:43-45`). It reached main on 2026-04-05
(commit `009b6f08`).

Around the filter: `--rho` on `fit run` (`args/mod.rs:924-928`,
`fit/mod.rs:286-288, 303`) and `--pmmh-rho` on `camdl profile`
(`args/mod.rs:1947`, `profile.rs:260-273`); the obs-alignment resolution for
`pmmh`, where the correlated rows require on-grid observations
(`cli/src/fit/methods.rs:540-560`) and the flag that feeds them
(`config_v2.rs:2455`); one row of the effect fire-time proposal's clock table;
and the correlated-PMMH passages in `docs/inference.md` and `docs/fit-toml.md`.

### 5.2 Users

One config on the maintainer's machine (2026-09-10, `rho = 0.99`); nothing in
the book. No note anywhere measures a benefit: the two notes on the correlated
filter, `2026-06-08-correlated-pf-degenerate-loglik.md` and
`2026-08-28-cpm-irregular-observation-grid.md`, are about a `-inf` defect and
about sizing the noise block on an irregular grid, and neither compares
acceptance or effective sample size against plain PMMH.

### 5.3 Why cut rather than keep

It is off by default, and the case for turning it on is unmeasured on the models
camdl fits. The method's variance reduction depends on the coupled noise driving
neighbouring trajectories: for a discrete-state model such as chain-binomial,
resampling permutes the particles, and a permuted particle set with the same
resampling uniform destroys most of the induced correlation unless particles are
first sorted so that neighbours in the noise are neighbours in the state.
Deligiannidis et al. (2018, §4) sort along a Hilbert curve through the
multidimensional state. This implementation does sort before resampling, but on
a one-dimensional heuristic key, the sum of each particle's flow accumulators
(`correlated_pf.rs:1072-1081`; the comment there calls it "a heuristic for
correlation preservation"). How much correlation survives that key on a model
with several compartments and strata is not measured, and the one config that
sets `rho = 0.99` has no measurement beside it. The method is sound and may
return later with a multidimensional sort, or a measured justification of the
one-dimensional key, and the note §8 requires.

The second reason is the shared substrate. The pre-drawn-noise design put a
second randomness regime through code every backend and method uses: the
chain-binomial transition kernel has a branch that reads a pre-drawn normal
instead of the RNG (`chain_binomial.rs:708-716`, via `StepScratch`'s
`gamma_override`, `binomial_z_values` and `binomial_z_idx`, `:60-71`), and the
initial-state draw has an `InitDraws::Normals` arm beside its RNG arm
(`compiled_model.rs:874-879` and the arms that follow). Every change to the
transition kernel or to initial-state laws has had to be made twice; gh#772 is
the record of one such doubling. Removing the method collapses both to one arm.

### 5.4 Delete

- `rust/crates/sim/src/inference/correlated_pf.rs`, whole file, and its
  `pub
  mod` line in `inference/mod.rs`. The normal-to-count transforms it
  exports (`normal_to_gamma`, `binomial_from_normal`, `poisson_from_normal`,
  `binomial_quantile`, `poisson_quantile`) have no caller outside the correlated
  arms listed below.
- `rust/crates/sim/src/inference/pmmh.rs`: `PMMHConfig.rho`, `n_source_groups`,
  `init_noise_width` and `t_start` (`:34, 69-80`; the comment at `:34` says only
  correlated PMMH reads `t_start`), `PMMHState.current_randoms` (`:163`),
  `CorrelatedEvalFn` (`:539`), the `eval_loglik_correlated` parameter of
  `run_pmmh` (`:554`) and the correlated branches (`:633-643, 757-805`).
- `rust/crates/sim/src/chain_binomial.rs`: `StepScratch.gamma_override`,
  `binomial_z_values`, `binomial_z_idx` (`:60-71`) and the arms that read them
  in `step_one` (`:691, 708-716`).
- `rust/crates/sim/src/compiled_model.rs`: `InitDraws::Normals` and its arms
  (`:874-` onward), `CompiledModel::init_noise_offsets` and `init_noise_width`
  (`:830, 837`) and their construction (`:2059-2061`).
- `rust/crates/cli/src/fit/pmmh.rs`: `PmmhStageOpts.rho` and its validation
  (`:34-65`), the forced `None` on the ODE-MH path (`:139-143`), the gh#193
  obs-grid preflight (`:195-205`), the correlated evaluator (`:662-684`).
- `rust/crates/cli/src/fit/pf_noise.rs`: the `Some(rho)` arm of the pair
  measurement and `eval_correlated` (`:339-392`). The `None` arm and the
  preflight itself stay: they serve plain PMMH (gh#767).
- `rust/crates/cli/src/profile.rs`: `--pmmh-rho` handling (`:179, 260-273`), the
  obs-grid preflight (`:929-937`), the correlated evaluator (`:1615-1626`), the
  `"rho"` key of the PMMH algorithm payload (`:1824`) and the knob plumbing
  (`:1170, 1582, 2182, 2192`); `args/mod.rs:1947`.
- `rust/crates/cli/src/fit/config_v2.rs`: the `rho` field (`:1133`), its CLI
  override (`:1631-1634`), `CliStageOverrides.rho` (`:1697`), the correlated
  flag in the alignment check (`:2455`), the tests at `:3390-3391` and
  `:7076-7077`, and the PMMH payload golden at `:7057` (moved in the identity
  commit, §5.6).
- `rust/crates/cli/src/fit/methods.rs:540-560`: the correlated rows of the
  `pmmh` alignment match collapse into the plain rows.
- `rust/crates/cli/src/args/mod.rs:924-928` (`--rho`) and
  `rust/crates/cli/src/fit/mod.rs:286-288, 303`.
- Tests, whole files: `rust/crates/sim/tests/correlated_pf_finite.rs`,
  `gh367_correlated_pf_death_mask.rs`, `normal_to_gamma_unbiased.rs`,
  `binomial_quantile_underflow.rs`, `gh772_poisson_quantile.rs`;
  `rust/crates/cli/tests/gh772_cpm_initial_state_laws.rs`. Tests inside mixed
  files: the `cpm_*` group in `sim/tests/pmmh.rs` (`:433-540`); the
  correlated-PF cells of `gh216_cursor_firing.rs` (`:55, 357, 431, 469`; the
  firing-time invariant keeps its particle-filter, IF2 and PGAS cells).
- The comment at `intervention.rs:230`; the correlated-PMMH passages in
  `docs/inference.md` and `docs/fit-toml.md`, and whatever a grep for
  `correlated` finds in the other `docs/*.md` and `README.md`.

### 5.5 What stays

Plain PMMH, unchanged: the bootstrap particle filter, the Haario adaptive
proposal with Robbins–Monro scaling, the likelihood-noise preflight
(`pf_noise.rs`) and its persisted spread. `camdl profile --algorithm pmmh`
stays, without `--pmmh-rho`.

### 5.6 Identity: this re-keys every PMMH fit and every PMMH profile

`rho` is part of the PMMH stage identity: the canonical payload golden at
`config_v2.rs:7057` contains `"rho":null`, and `:7076-7077` asserts that setting
it changes the payload. Removing the field re-keys every PMMH fit leaf, with the
same consequences as §3.6: no new PMMH run deduplicates against an existing one,
no in-flight PMMH fit resumes, stored results stay readable. `camdl profile`
keys its PMMH cells on a payload that carries `"rho"` (`profile.rs:1824`, hashed
whole by `canonical_config_hash`), so every PMMH profile run re-keys as well.

Same mechanics as §3.6: a versioned identity bump per `VERSIONING.md`, the two
goldens moved in their own reviewed commit with nothing else in it, and removal
traps so that `rho = …` under `pmmh` in a `fit.toml`, `--rho` and `--pmmh-rho`
are refused with a message naming the removed field and the fix (delete it), in
the shape of `args/mod.rs:115-125`. If PR A's identity bump has not shipped in a
release when PR C lands, PR C's identity commit joins that bump rather than
adding a second one; pending re-keys land in one bump.

The maintainer confirmed the PGAS re-key on 2026-10-24. This PMMH re-key is a
consequence of the `rho` decision that was not on the table at the time and is
recorded here so it is confirmed before PR C's identity commit lands.

### 5.7 Issues

Found with `gh issue list --search "correlated" --state open` and
`gh issue list --search "rho" --state open` on 2026-10-24. None closes outright;
each keeps a plain-PMMH or firing-time core and loses its correlated clause.

- gh#767 "PMMH targets an acceptance rate its own estimator cannot reach — the
  noise-aware target": its mechanism section describes `s` under `rho`; the
  issue is about plain PMMH and stays open, with that clause edited.
- gh#102 "Profile-PMMH ergonomics: expose burn_in/thin/proposal_scale; reject
  pmmh_steps ≤ burn_in": lists `--pmmh-rho` among the flags profile exposes;
  stays, the flag goes.
- gh#167 "survey: per-point independent seeds — add a common-random-numbers
  (CRN) option for landscape viz": proposes building on `correlated_pf.rs`;
  stays, relabelled to say the substrate it named is gone.
- gh#958 "A scheduled intervention at exactly t_start never fires (particle
  filter and forward chain-binomial alike, silently)": lists the correlated PF
  among the affected paths; one fewer path to fix.
- gh#220 "harden: property-test the full scheduling-invariant matrix +
  consolidate the scattered firing tests": its first item names IF2 and
  correlated-PF proptests; scope shrinks to IF2.
- gh#335 "Support mixed deterministic + stochastic exits from one source
  (chain-binomial reserve-off-the-top)": its plan includes a parity check under
  `rho≠None`; that check is dropped from the plan.

The other issues those two searches return match "correlated" or "rho" in
another sense (a spatial kernel parameter, a model's reporting rate, correlated
posteriors) and are unaffected.

### 5.8 Archive tag

`archive/correlated-pmmh`, on the last main commit that carries
`correlated_pf.rs` (§7).

## 6. Kept, and why

- **The ODE gradient and `nl-lbfgs`.** §4.5. The forward-sensitivity gradient is
  the exact oracle any future adjoint would be checked against, and the
  optimiser's evaluation count makes its cost a fair trade.
- **Reactive interventions.** Kept by the maintainer. They run on chain-binomial
  forward and refuse loudly on every inference path; the fire-time proposal does
  not touch them.
- **Gillespie.** The exact continuous-time reference the cross-backend agreement
  tests check chain-binomial against, and the only exact event log, which the
  lineage tests and the book's transmission-trees chapter depend on.
- **IF2, PMMH, PGAS, PGAS with NUTS.** The three methods people run (§1). PGAS's
  NUTS parameter move is on by default (`use_nuts`, `config_v2.rs:1814`) and is
  what the 207 PGAS configs use; its gradient code (`pgas_grad.rs`), the NUTS
  core it shares (`nuts.rs`: `nuts_step`, `MassMatrix`, `NUTSConfig`,
  `DualAveraging`) and the OCaml autodiff that emits `rate_grad`, `obs_grad`,
  `overdispersion_grad` and `initial_state_logpdf_grad` all stay.

## 7. Archive tags

Every surface cut by a camdl proposal gets an annotated git tag,
`archive/<surface>`, on the last commit on main that still carries it: in
practice the parent of the PR's first deletion commit, pushed before the PR
merges. The tag message names the proposal; the proposal records the tag name.
Here:

| tag                       | last commit carrying                            | placed by |
| ------------------------- | ----------------------------------------------- | --------- |
| `archive/pgas-tempering`  | the `tempering` ladder and `InverseTemperature` | PR A      |
| `archive/ode-nuts`        | `ode_nuts.rs` and `Algorithm::Nuts`             | PR B      |
| `archive/correlated-pmmh` | `correlated_pf.rs` and `rho`                    | PR C      |

```
git tag -a archive/<surface> <sha> -m "Last main commit carrying <surface>.
Removed by docs/dev/proposals/2026-10-24-remove-pgas-tempering-ode-nuts-and-correlated-pmmh.md."
git push origin archive/<surface>
```

A living "in-dev" branch holding the removed code is explicitly not used. It
would stay useful only if rebased every time main moved, nobody will do that,
and a branch that no longer merges is worse than a tag because it looks like it
might. Git history plus a tag is the archive: anyone who wants the code reads it
at the tag, and anyone who wants it back ports it forward against the current
tree, with the measurement note §8 asks for.

Freezing was considered and rejected by the maintainer: a frozen surface keeps
every cost §1 names (a cell in every seam extraction, a row in every table) and
removes only the right to change it.

## 8. The standard this sets

A new inference feature (a new algorithm, a new sampler option, a new
variance-reduction or tempering scheme) arrives on a branch, never directly on
main, and merges only together with a `docs/dev/notes/` note that measures the
claimed improvement against the existing method on at least one committed
fixture. For a sampler the measurement is effective samples per unit compute,
acceptance, or bias against an oracle; for an optimiser it is iterations to the
same maximum. The merge commit cites the note.

Tempering (commit `163c80d7`, 2026-04-09), correlated PMMH (`009b6f08`,
2026-04-05) and NUTS on the ODE backend (`951dc5c9`, 2026-07-07) each reached
main without such a note. The one comparison that was made, DELENDA, was made
after the fact and lives outside the repository. This proposal, and the three
PRs it specifies, are the cost of that. The rule is written into `CLAUDE.md`
("Commits and tracking") and `docs/dev/agent-verification-conventions.md` in the
same commit as this proposal.

## 9. Order of work

1. **PR A, delete tempering.** Three commits: (i) the ladder, swaps, diagnostic,
   CLI, tests and docs (§3.4); (ii) the β parameter and the rung loop (§3.5)
   with the mechanical test edits; (iii) the identity bump, the moved identity
   golden and the removal trap (§3.6), alone. Issues per §3.7. The §10
   single-rung pin runs before and after (i)+(ii).
2. **PR B, delete NUTS on the ODE backend.** §4.4 in one commit, the
   `methods.rs` steer rewrite and the doc edits included; no identity commit
   (§4.6). Issues per §4.7.
3. **PR C, delete correlated PMMH.** Two commits: (i) §5.4, with the plain-PMMH
   pin of §10 run before and after; (ii) the identity bump or its join to PR
   A's, the two moved goldens and the removal traps (§5.6), alone. Issues per
   §5.7.
4. **Then** the fire-time application-seam PR, whose PGAS step is smaller for PR
   A having landed and whose clock table is one row shorter for PR C.

Each PR: the archive tag first (§7), the deletion second; full `make test`;
`mdfmt` on every touched `.md` except `docs/camdl-language-spec.md`; no
attribution trailers. The batch removal has its own order of work in
`2026-10-24-batch-to-simulate.md`.

## 10. Tests that prove the cuts are clean

- No source line under `rust/` or `ocaml/` matches `tempering`,
  `InverseTemperature`, `n_rungs`, `swap_log_alpha`, `low_swap_rate`,
  `ode_nuts`, `WarmupAdapter`, `MassMetric`, `warmup_schedule`,
  `Algorithm::Nuts` (which also catches `FitAlgorithm::Nuts`), `correlated_pf`,
  `PFRandomState`, `binomial_z_values`, `gamma_override` or `init_noise_width`.
  `rho` is a common parameter name in models, so its check is structural: no
  `rho` field remains in `Algorithm::PMMH`, `PMMHConfig`, `PmmhStageOpts`,
  `CliStageOverrides` or `ProfilePmmhKnobs`.
- A `fit.toml` with `tempering = [1.0]` under `pgas`, or `rho = 0.99` under
  `pmmh`, is refused with the removal message; the same file without the line
  validates. `algorithm = "nuts"` is refused as an unknown algorithm with the
  steer of §4.4. `--tempering`, `--rho` and `--pmmh-rho` hit removed-flag traps.
- **The single-rung pin.** `pgas_tempering.rs::test_single_rung_deterministic`
  is kept and renamed; its `draws.tsv` must be byte-identical before and after
  PR A commits (i) and (ii). Removing the ladder and β must not move a single
  cold-chain number; this test is the proof.
- **The plain-PMMH pin.** A plain PMMH fixture (no `rho`) at a fixed seed
  produces a byte-identical `draws.tsv` before and after PR C commit (i). The
  plain path never enters the pre-drawn-noise arms, so removing them must not
  move a number; the pin is the proof that the arms were as dead on that path as
  §5.3 claims.
- **The gradient stayed.** `nl_lbfgs_ode_recovery.rs`, `state_grad_gating.rs`,
  the finite-difference oracle tests in `ode_grad.rs` and the `mh` cell of
  `coarse_burnin_ode.rs` pass unchanged after PR B.
- The three archive tags exist on `origin`, each on a commit that still contains
  the file or field it names.

## 11. Follow-ups

- Book, filed as `vsbuffalo/camdl-docs` issues: #1 (the scenarios chapter runs
  `camdl batch run`; rewrite the sweep on `simulate`), #2 (remove the PGAS
  parallel-tempering sections once camdl removes `tempering`), #3 (the backends
  reference lists four backends and `--backend tau_leap`, which no longer
  exists). PRs B and C each start with a grep of the book for `nuts` on the ODE
  backend and for `rho` under `pmmh`, and file a camdl-docs issue for whatever
  they find; the config survey found no executed use of either.
- The scenario `compose` field has no external users; whether it survives is a
  question for the coordinate-kind rewrite, which owns scenario algebra.
- The first feature to return under §8 should be the one with a measurement in
  hand, whichever that turns out to be.
