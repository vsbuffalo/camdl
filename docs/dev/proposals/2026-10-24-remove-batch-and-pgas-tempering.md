# Remove `camdl batch` and PGAS tempering; freeze the ODE gradient stack

- Date: 2026-10-24
- Status: draft — cuts 1 and 2 (§3, §4) decided by the maintainer on 2026-10-24;
  three calls pending: the ODE gradient stack, freeze or cut (§6.2); correlated
  PMMH, keep or cut (§6.3); and confirmation of the PGAS fit re-key that cut 2
  entails (§4.6). Everything else is settled.
- Relates-to: gh#789, gh#930, gh#286, gh#738, gh#599, gh#923, gh#598, gh#569,
  gh#551, gh#917, gh#390; `2026-09-27-coordinate-kind-design-vs-sample.md`
  (accepted; rebuilds grid composition on `simulate`),
  `2026-10-04-effect-fire-time-authority.md` (draft; its PGAS step touches the
  sweep this proposal simplifies), the note
  `2026-10-06-run-grid-workflows-and-composition.md` (the usage survey),
  `docs/dev/incidents/2026-08-10-pgas-tempering-swap-sign.md`.
- Required reading: `.claude/rules/run-identity.md` (the `tempering` field is
  fit identity; §4.6), `.claude/rules/sim-and-inference.md` (every backend ×
  method cell works and is tested or fails loudly; §1), `VERSIONING.md` (alpha
  posture: rename or remove atomically, no shims).
- Evidence: a read-only inventory of main `bff4cec4` taken on 2026-10-24. Every
  `path:line` below is from that inventory; nothing was executed.

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

What people actually run, counted from fit configs in the book and in the six
modelling projects on the maintainer's machine (2026-10-24):

| method or surface                    | book configs          | project configs                               | notes                                   |
| ------------------------------------ | --------------------- | --------------------------------------------- | --------------------------------------- |
| IF2                                  | 38                    | 10                                            | the teaching and profiling method       |
| PMMH                                 | 11                    | 46                                            | seed-timing, `seed-modeling`            |
| PGAS                                 | 2                     | 207                                           | `ebola-bdbv-camdl`, touched 2026-10-20  |
| `mh`, `nl-sbplx`, `nl-bobyqa` on ODE | 0                     | 30                                            | derivative-free                         |
| `nuts` on ODE, `nl-lbfgs`            | 0                     | 2                                             | both labelled demonstrations            |
| PGAS `tempering` ladder              | 0                     | 0 live (2 archived in `ebola-camdl/archive/`) |                                         |
| correlated PMMH (`rho`)              | 0                     | 2                                             | both in `ebola-bdbv-camdl`              |
| `camdl batch`                        | 2 executed book cells | 0                                             | `ebola-camdl` tried it and rejected it  |
| scenario `compose = [...]` field     | 0                     | 0                                             | used only by camdl's own tests and spec |

## 2. Summary

| surface                                                                                                 | decision                                  | size that goes                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camdl batch run`, `batch status`, batch manifests, `[design.*]` generation                             | **cut** (§3)                              | ≈1,850 lines of `batch.rs`, `sampling.rs` (395), ≈80 of args/dispatch, ≈1,000 of tests, 55 shell lines, 176 fixture lines; ≈1,100 lines of `batch.rs` move, not delete |
| PGAS tempering (`tempering` ladder, swaps, `LowSwapRate`, `--tempering`)                                | **cut** (§4)                              | ≈550 lines of sim `pgas.rs`, ≈150 CLI, ≈1,000 tests, plus ≈150 mechanical edits removing the β parameter                                                               |
| ODE gradient stack (`det_grad`, `nuts` on ODE, `nl-lbfgs`)                                              | **frozen**, pending the maintainer (§6.2) | 0 now                                                                                                                                                                  |
| correlated PMMH (`rho`)                                                                                 | pending the maintainer (§6.3)             | 0 now                                                                                                                                                                  |
| scenario `compose` field, reactive interventions, `--design-from`, Gillespie, IF2, PMMH, PGAS with NUTS | **kept** (§5)                             | 0                                                                                                                                                                      |

## 3. Cut 1: `camdl batch`

### 3.1 What it is

`camdl batch run <manifest.toml>` expands a TOML manifest (`[config]`,
`[config.seeds]`, `[[scenario]]`, `[sweep]` with `list`/`linspace`/`range`,
`[design.*]` Latin-hypercube/Halton/Saltelli designs, `[obs]`, `[output]`) into
a grid of simulate cells and runs them into the content-addressed store;
`batch status` reports on a manifest's cells. It predates `simulate`'s own
multi-cell path (`--scenario`, `--draws`, `--replicates`, `--seeds`), which now
carries every real workload.

### 3.2 Users

None. `ebola-camdl`'s `CAMDL-PROJECTION-GAP.md:36, 56, 119, 166-167` records
that batch was tried and rejected (the documented `[draws]` manifest form fails
with "unknown field `model`") and the project moved to `camdl simulate --draws`
(`scripts/project_scenarios.py:3`). No other project, Makefile or workflow calls
it. The book's scenarios chapter executes it in two cells
(`guide/scenarios.qmd:492, 518`) and mentions it in prose at `:86, 823`,
`language/spec.qmd:2924-2933`, `reference/inference-spec.qmd:22, 882, 1060`,
`_lib/README.md:94`, `guide/getting-started.qmd:599, 951`,
`guide/experiments.qmd:865`; the manifest is `guide/boarding-school/sweep.toml`.

### 3.3 Delete

- CLI: `rust/crates/cli/src/main.rs:149-151` (`Batch(BatchCmd)`), `:345-361`
  (`BatchCmd`), `:571-572` (dispatch), `:27` (`mod batch;`, after §3.4);
  `rust/crates/cli/src/args/mod.rs:590-646` (`BatchArgs`, `BatchStatusArgs`);
  stale comments at `args/mod.rs:3, 183-186`.
- `rust/crates/cli/src/batch.rs`, the batch-only parts: `:37-65`
  (`build_parallel_pool`, `run_pooled`); `:67-370` (`ExperimentToml`,
  `ObsSection`, `DesignBlock`, `DesignParamToml`, `RangeMinMax`, the private
  TOML `SweepSpec` and `expand_sweep`, `ConfigSection`, `SeedsSection`);
  `:376-386` (`ScenarioEntry`); `:450-500` (`resolve_batch_scenarios`);
  `:527-560` (`ManifestModel`, `load_manifest_model`); `:568-648` (`batch_job`,
  `scenario_refs`, `sweep_source`); `:650-995` (`cmd_batch_run`);
  `CasSink::predict_cells` `:1274-1340` and `CellPrediction` `:1342-1357`
  (called only from batch dry-run, status and design dry-run at
  `:1966, 2283,
  2348`); `:1775-2046` (`run_design_experiment`,
  `print_design_dry_run`, `build_priors_txt`); `:2188-2314`
  (`cmd_batch_status`); `:2315-2489` (`print_batch_dry_run`); the batch-only
  unit tests under `:2490` (the `experiment_toml_*`, `sweep_*`, `parse_sweep`
  and range-refusal tests).
- `rust/crates/cli/src/sampling.rs`, whole file (395 lines): its only caller is
  `batch.rs:23`, so the `[design.*]` designs go with it.
- Dead after the above: `ParamSource::Sweep`
  (`rust/crates/cli/src/sim_job.rs:179` and its arms at `sim_job.rs:205, 218`,
  `engine.rs:368`), constructed only at `batch.rs:639, 1841`. `Seeds::Explicit`
  stays (`simulate --seeds`, `main.rs:1746`).
- Shell: `tests/test_ocaml_to_rust.sh:84-143` (`run_batch_test` and its eight
  calls, invoked from `Makefile:209`); the eight manifests
  `tests/fixtures/exp_*.toml`.
- Docs and spec: `docs/commands.md:21, 47, 174-205, 272`; `docs/intro.md:413`;
  `docs/inference.md:1511`; `docs/camdl-inference-spec.md:15, 936, 1134-1136`;
  `docs/camdl-language-spec.md:4180-4213, 4796, 5060-5069` (edit by hand, never
  `dprint`); `docs/camdl-run-spec.md` §5 header `:1956`, `:1998-2001`, §5.5
  `:2162-2361`, §5.6 `:2362-2393`, §3.3 `:1079-1105`, `:1204-1208`,
  `:1275-1276`, §9.7 `:4663-4674`, Appendix A `:6224-6239`, "Batch TOML
  stability" `:6530-6544`, and the inline mentions at
  `:7, 56, 69, 72-74, 85,
  128-136, 148, 192, 766, 820-825, 869, 885, 894, 903-907, 949-963, 1017, 1057,
  1068`;
  `README.md:232, 360, 401`; `scripts/review-zip.sh:66`, `scripts/README.md:68`.

### 3.4 Shared code that moves out of `batch.rs` first

About 1,100 lines of `batch.rs` are the store sink that `simulate` uses, and
must move to a new module (`rust/crates/cli/src/cas_sink.rs`) in a pure-move
commit before anything is deleted: `ResolvedEntry` and `from_preset`
(`:404-448`, used at `main.rs:2603-2636`); `RunEntry` (`:502-524`,
`main.rs:2798`); `CasSink` (`:1004-1066`, `:1068`, `:1072-1265` minus
`predict_cells`, `RunSink for CasSink` `:1417-1715`; used at
`main.rs:2242,
2564, 2647, 2686, 2672`); `params_path_label` with its tests
(`:1359-1415`), `params_level_hash` `:1717`, `read_traj_digest` `:1724`,
`ensure_provenance_label` `:1736`, `obs_subtree_hash` `:2048` (also
`emit_every.rs:18`), `write_obs_into_cas` `:2071-2187`. `simulate` builds the
same sink with `obs_enabled` and `emit_every` (`main.rs:2647-2672`), so the
obs-into-store path stays live and its coverage is ported to `simulate` (§3.5).

Unaffected and staying: `engine.rs` (`run_job`, `plan_grid`,
`build_cell_sim_run`, `check_scenario_coordinate_collision`; used by `simulate`
and `fit predict` at `fit/predict.rs:1091, 2241-2291`); `sim_job.rs`
(`SimulateJob`, `ParamSource::{Point, Draws}`, `Seeds`, `ObsOutput`,
`ScenarioRef`, `resolve_scenario_ref`); `sim_ensemble_cas.rs` (used only by
`simulate`); `args::types::SweepSpec`, the CLI `--sweep` of `fit run`,
`fit predict` and `profile` (`args/mod.rs:750, 1207, 1845`), which is a
different type from batch's TOML `SweepSpec`;
`params_resolver::{resolve_preset_delta,
composed_preset_t_end, UserCoordinate::Sweep}`;
`config_v2::SeedsSpec` (belongs to `[synthetic]`); `fit/contrasts.rs` (no
reference to any of this).

### 3.5 Tests

Delete whole files: `rust/crates/cli/tests/acceptance_batch_named_scenario.rs`,
`acceptance_batch_obs_ensemble.rs`, `acceptance_design_cas.rs`,
`gh589_batch_obs_preflight.rs`, `model_graph_archive.rs` (its only test runs
batch; re-pin the model-graph archive through `simulate`). Delete the batch
tests inside mixed files: `cas_integration.rs:349, 434, 492, 878, 973, 1018`;
`design_preserving_simulate.rs:707`; `scenario_coordinate_collision_gh572.rs`
helper `:116` and tests `:162, 172, 186, 201, 214, 234`;
`scenario_identity_gh573.rs` helper `:126` and `:205, 213, 224`;
`simulate_identity_parity.rs:460`. Where one of these was a
simulate-versus-batch parity test guarding `CasSink`, rewrite it against
`simulate` alone rather than drop it; the sink is what it protects. Re-home the
three `CasSink` identity unit tests (`table_content_folds_into_sim_run_id`,
`every_simrun_field_has_an_identity_route`,
`design_cell_dedupes_with_normal_sim_cell`) with the moved code.

### 3.6 Migration

`simulate` has no `--sweep` flag today (`SimulateArgs` carries `--scenario`,
`--enable`/`--disable`, `--seeds`, `--replicates`, `--draws`, `--n-draws`,
`--draws-out`, `--design-from`; read at `args/mod.rs`). The coordinate-kind
proposal adds `--sweep` and `--sweep-file` to `simulate` in its flag-split
stage. Decision: delete batch now rather than hold it until then. Batch has no
users to migrate, and holding it would couple the deletion to the
coordinate-kind arc, whose rewrite is simpler without batch in its blast radius.
The book's scenarios chapter is rewritten in step with the deletion PR as a loop
over `camdl simulate … --param beta=<v> --scenario <s> --replicates 50`, which
is the pattern every surveyed project uses today, with a sentence that `--sweep`
is coming. The chapter's claim that "the batch system manages the Cartesian
product" goes.

### 3.7 Issues

Close gh#930 (`batch status` ignores two flags) and gh#286 (CSV-row batch
source). Relabel to `simulate`/`CasSink`, since the code is shared and the bugs
survive: gh#738 (obs subtree at a different cadence, `CasSink::should_run`),
gh#599 (committed parent declares a failed obs child, `CasSink::merge_cell`).
Scope shrinks: gh#923 (batch's `ConfigSection` and `SeedsSection` were two of
the three to four `dt`/seed structs), gh#598 (batch was one of three `snap_at`
callers). Unaffected: gh#929, gh#911, gh#577, gh#588, gh#586, gh#704, gh#699.

## 4. Cut 2: PGAS tempering

### 4.1 What it is

`[method] tempering = [β₁, β₂, …]` under `algorithm = "pgas"` runs one PGAS
chain per inverse temperature β with replica-exchange swaps between neighbouring
rungs; the first rung is snapped to β = 1 (cold). Default `[1.0]`, one rung, no
swaps. With it come the `--tempering` CLI override, the `LowSwapRate`
diagnostic, `swap_acceptance_rates` in the result and the per-chain NUTS
diagnostics, the observation-only tempered target from gh#551 (PR #945), and the
incident `2026-08-10-pgas-tempering-swap-sign.md`. There is no `n_rungs` option;
that name is a local, `config.tempering.len().max(1)` (`pgas.rs:4267`).

### 4.2 Users

No live config sets a ladder. The two that do are archived
(`ebola-camdl/archive/models/fit_merged_pgas.toml:136`, `fit_2p.toml:137`, both
`[1.0, 0.6, 0.3, 0.1]`). The book describes it
(`inference/algorithms.qmd:104-121` "Parallel tempering",
`inference/likelihood.qmd:456, 469`, `inference/diagnostics.qmd:26, 39`) and
never runs it.

### 4.3 Why cut rather than keep

It was experimental and reached main without a user. Its only open bug (gh#789,
trace rows mixing two rungs) and its only incident exist solely for more than
one rung. It puts a rung loop and a temperature parameter through the
highest-risk file in the repository, and the fire-time implementation has to
step around both in its PGAS stage. The gh#551 observation-only tempered target
goes with it; that is a deliberate loss, named here so it is not rediscovered as
a regression.

### 4.4 Delete

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
  `test_single_rung_deterministic` as the §8 pin, renamed),
  `pgas_tempering_exact.rs`, `gradient_check_obs.rs:570`
  (`gh551_tempered_energy_and_gradient_match_fd`),
  `csmc_exact_invariance.rs:409`
  (`one_sweep_leaves_the_tempered_smoothing_target_invariant`).
- Docs: `docs/fit-toml.md:336, 413-441`; `docs/workflow.md:235, 238`;
  `docs/agents.md:213`; `docs/camdl-run-spec.md:2657, 3320, 5679, 6272`; the
  book sections in §4.2. The incident report stays as history.

### 4.5 Remove the temperature parameter, not just the ladder

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
go in their own commit (§7).

### 4.6 Identity: this re-keys every PGAS fit

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
silent acceptance. The re-key is the price of the cut and needs the maintainer's
explicit confirmation before PR C lands; it is the third pending call in the
status line.

### 4.7 Issues

Close gh#789. gh#569 names the `pgas_tempering` and `swap_forms_agree` tests in
its body but its defect is the θ-block accept call site; it stays open and is
relabelled. gh#710 and gh#172 matched the word "rung" only and are unaffected.

## 5. Kept, and why

- **The scenario `compose = [SCENARIO, ...]` field.** There is no
  `camdl compose` verb (`Command` at `main.rs:142-246` has no such variant;
  `git log
  -S"Compose("` on `cli/src` is empty). `compose` is a language
  feature: a scenario layers other scenarios' patches, its own applying last
  (spec §17.1, `camdl-language-spec.md:4523-4584`), resolved in
  `params_resolver.rs:482-488, 529-535, 587-593, 642-648, 772` for every
  `--scenario` user. It has no external users, but removing a language construct
  is a DSL decision for the coordinate-kind rewrite, which owns scenario
  algebra. Its collision bug (the note's C9) is filed separately.
- **Reactive interventions.** Kept by the maintainer. They run on chain-binomial
  forward and refuse loudly on every inference path; the fire-time proposal does
  not touch them.
- **`simulate --design-from <fit.toml>`.** Independent of batch (it reads a fit
  configuration, `args/mod.rs:493`, `main.rs:1841`) and used by
  `ebola-bdbv-camdl`'s prior-predictive rules (`workflow/model.smk:1035, 1377`).
- **Gillespie.** The exact continuous-time reference the cross-backend agreement
  tests check chain-binomial against, and the only exact event log, which the
  lineage tests and the book's transmission-trees chapter depend on.
- **IF2, PMMH, PGAS, PGAS with NUTS.** The three methods people run (§1). PGAS's
  NUTS parameter move is on by default (`use_nuts`, `config_v2.rs:1814`) and is
  what the 207 PGAS configs use; its gradient code (`pgas_grad.rs`), the NUTS
  core it shares (`nuts.rs`: `nuts_step`, `MassMatrix`, `NUTSConfig`,
  `DualAveraging`) and the OCaml autodiff that emits `rate_grad`, `obs_grad`,
  `overdispersion_grad` and `initial_state_logpdf_grad` all stay whatever §6.2
  decides.

## 6. Frozen (two calls pending)

### 6.1 What frozen means

**Frozen** means: the code and its tests stay in the gate so it cannot rot
unnoticed; `camdl docs` and `--help` mark the surface experimental and name the
supported alternative; no new features land on it; its open issues get a
`frozen` label and are excluded from the agent burn-down; the decision is
revisited when a model needs it.

### 6.2 The ODE gradient stack

`det_grad` integrates the ODE together with its forward sensitivity equations,
one extra system per parameter, returning the log-likelihood and its gradient in
one call (`sim/src/inference/ode_grad.rs`, 1,366 lines). `nl-lbfgs` is NLopt's
quasi-Newton optimiser fed by that gradient
(`cli/src/fit/nlopt_stage.rs:611,
996`; also
`camdl profile --algorithm nl-lbfgs`, `profile.rs:768-780`). `nuts` on the ODE
backend is the no-U-turn sampler on that gradient
(`sim/src/inference/ode_nuts.rs`, 760; `cli/src/fit/nuts.rs`, 647). Around them:
`preflight_gradient_ode` in `gradient_capability.rs:233-510`, the
`needs_state_grad` switch (`config_v2.rs:2380-2388`, `util.rs:410-482`), the
`WarmupAdapter`/`MassMetric`/`warmup_schedule` half of `nuts.rs` (`:517-`), the
ODE-only autodiff (`differentiate_rate_state` `autodiff.ml:829`,
`differentiate_projection` `:852`, `differentiate_ic` `:950`, and the IR fields
`transition.rs:144`, `observation.rs:364`, `model.rs:453`), and the tests
`nuts_ode_recovery.rs`, `nl_lbfgs_ode_recovery.rs`, `state_grad_gating.rs`, plus
ODE cells in `coarse_burnin_ode.rs` and `gh900_every_method_writes_progress.rs`.

The measured case against it is cost, not correctness. Metropolis-Hastings pays
one solve per step; forward-sensitivity NUTS pays one-plus-parameters solves per
leapfrog step and several leapfrog steps per draw. The DELENDA comparison
(`project` notes, 2026-07) put that into effective samples per unit compute and
found forward-sensitivity NUTS dominated by adaptive MH at every dimension; only
an adjoint gradient, about twice a solve regardless of parameter count, changes
the picture, and none exists. That is why the two ODE projects fit with `mh` and
`nl-sbplx`, and why the only two gradient-stack configs are demonstrations. Open
issues that would close with a cut: gh#917, gh#390.

Recommendation: freeze, not cut, leaning. The forward-sensitivity gradient is
the exact oracle an adjoint would be checked against, and the maintainer has
asked to keep the option. Awaiting the maintainer's call.

### 6.3 Correlated PMMH

`rho` under `algorithm = "pmmh"` runs the correlated pseudo-marginal variant
(`sim/src/inference/correlated_pf.rs`, 1,150 lines). It requires on-grid
observations (`fit/methods.rs:540-560`), carries its own noise-layout refusals,
and is one row of the fire-time plan. Two configs use it, both the maintainer's
(`ebola-bdbv-camdl`). Awaiting the maintainer's call; no recommendation either
way, since the user is the maintainer.

## 7. Order of work

1. **PR A, pure move.** `batch.rs`'s shared sink (§3.4) to `cas_sink.rs`; no
   behaviour change; `simulate` outputs byte-identical on the fixtures. Gate.
2. **PR B, delete batch.** §3.3 and §3.5, the shell test and fixtures, docs,
   spec and README (§3.3), `ParamSource::Sweep`. The book's scenarios chapter is
   rewritten in step (§3.6). Issues per §3.7. Gate.
3. **PR C, delete tempering.** Three commits: (i) the ladder, swaps, diagnostic,
   CLI, tests and docs (§4.4); (ii) the β parameter and the rung loop (§4.5)
   with the mechanical test edits; (iii) the identity bump, the moved identity
   golden and the removal trap (§4.6), alone. Issues per §4.7. Gate after each
   commit; the §8 pin runs before and after (i)+(ii).
4. **Then** the fire-time application-seam PR, whose PGAS step is smaller for PR
   C having landed.

Each PR: full `make test`; `mdfmt` on every touched `.md` except
`docs/camdl-language-spec.md`; no attribution trailers.

## 8. Tests that prove the cuts are clean

- `camdl batch` exits with clap's unknown-subcommand error.
- No source line matches `tempering`, `InverseTemperature`, `n_rungs`,
  `swap_log_alpha`, `low_swap_rate`, `ParamSource::Sweep` or `sampling::`
  outside `docs/dev/` history.
- A `fit.toml` with `tempering = [1.0]` under `pgas` is refused with the removal
  message; the same file without the line validates.
- **The single-rung pin.** `pgas_tempering.rs::test_single_rung_deterministic`
  is kept and renamed; its `draws.tsv` must be byte-identical before and after
  PR C commits (i) and (ii). Removing the ladder and β must not move a single
  cold-chain number; this test is the proof.
- The three `CasSink` identity tests pass from their new home, and the ported
  `simulate` obs-into-store tests cover what `acceptance_batch_obs_ensemble.rs`
  covered.

## 9. Follow-ups

- Book: rewrite the scenarios chapter (§3.6); remove the tempering sections
  (§4.2); re-sync `reference/runtimes.qmd`, which still lists four backends and
  four commands with `--backend tau_leap`, a backend that no longer exists.
- The scenario `compose` field has no external users; whether it survives is a
  question for the coordinate-kind rewrite.
- If §6.2 resolves to "cut", the inventory in that section is the deletion list;
  what stays is listed in §5's last bullet.
