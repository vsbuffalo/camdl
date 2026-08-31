# Example models

Every model shipped in this repository — 93 `.camdl` files, plus 34 deliberately-invalid sources used to pin compiler diagnostics. Together they are the worked-example corpus: whatever you are trying to express, something here probably expresses it already.

**This file is generated.** Run `make examples-doc` after adding or renaming a model; `scripts/gen_examples_doc.py --check` gates it. Edit the generator, not this file.

## Reading the tables

**Structure** is the unstratified compartment list, followed by the dimensions it is expanded over and their sizes — so `S,E,I,R × age[2] × patch[5]` is a 4-compartment model that expands to 40 states. **Features** are flags read out of the compiled IR:

- `obs` — has an `observations` block (can be fitted to data)
- `intervention` — declares at least one intervention
- `scenarios` — declares scenarios beyond the baseline
- `tables` — reads an indexed table (contact matrix, populations, …)
- `forcing` — has a time-varying forcing function
- `ode` — has explicit ODE equations (real-valued compartments)
- `dt` — pins an explicit discretisation step
- `priors` — declares a non-flat prior on at least one parameter
- `calendar` — anchored to real dates via `origin`
- `data` — reads an external file with `read(…)`
- `params` — ships a `.params.toml`, so it runs without you supplying values
- `fit-config` — ships a fit configuration (`fit.toml` / `if2.toml`)

Descriptions are each model's own header comment.

## Start here

If you are looking for a model to copy rather than a model to study, these ten cover most of the language between them:

- [`sir_basic`](../ocaml/golden/sir_basic.camdl) — the smallest complete model — start here
- [`seir_observations`](../ocaml/golden/seir_observations.camdl) — how observations attach to a model
- [`sir_priors`](../ocaml/golden/sir_priors.camdl) — declaring priors for inference
- [`sir_two_patch`](../ocaml/golden/sir_two_patch.camdl) — indexed parameters over a dimension
- [`seir_age`](../ocaml/golden/seir_age.camdl) — stratification and a contact matrix
- [`seir_vaccine`](../ocaml/golden/seir_vaccine.camdl) — an intervention plus a scenario to switch it on
- [`seir_erlang_via`](../ocaml/golden/seir_erlang_via.camdl) — non-exponential dwell times via `via`
- [`sirv_anchored_calendar`](../ocaml/golden/sirv_anchored_calendar.camdl) — calendar time: real dates, seasonal forcing, dated campaigns
- [`ross_macdonald`](../ocaml/golden/ross_macdonald.camdl) — multi-species host-vector transmission
- [`seir_spatial_5_inference`](../ocaml/golden/seir_spatial_5_inference.camdl) — a spatial model set up as an inference stress test

## Example models

The primary example set. Every file here is automatically enrolled in the IR round-trip, Rust smoke, and cross-language simulate tests, so each one is known to compile, survive the OCaml↔Rust IR contract, and simulate on both stochastic backends. — `ocaml/golden/*.camdl` (54 models)

| Model                                                                          | Structure                                                                             | Features                                        | Description                                                                                                                                                       |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`bimolecular`](../ocaml/golden/bimolecular.camdl)                             | A,B,C                                                                                 | —                                               | Bimolecular reaction A + B --> C (wave 1 / malaria #1).                                                                                                           |
| [`branching_si_symp_asym`](../ocaml/golden/branching_si_symp_asym.camdl)       | S,I_symp,I_asym                                                                       | —                                               | Probabilistic branching test: S --> {I_symp : p_symp, I_asym : 1-p_symp} under mass-action infection.                                                             |
| [`ebola_outcome_hyper`](../ocaml/golden/ebola_outcome_hyper.camdl)             | S,E,R,D,I__fatal__1,I__fatal__2,I__fatal__3,I__recover__1,I__recover__2,I__recover__3 | —                                               | Ebola onset-to-outcome as a CFR-split mixture, via `via hyper_erlang`.                                                                                            |
| [`flu_data_forcing`](../ocaml/golden/flu_data_forcing.camdl)                   | S,E,I,R                                                                               | obs, forcing                                    | Seasonal influenza SEIR with a data-driven transmission forcing.                                                                                                  |
| [`init_dependency_order`](../ocaml/golden/init_dependency_order.camdl)         | S,E,I,R,D                                                                             | —                                               | SEIR seeded mid-outbreak, where each initial condition is written in terms of the ones it depends on (gh#733).                                                    |
| [`init_laws`](../ocaml/golden/init_laws.camdl)                                 | S,E,I,R,W                                                                             | ode                                             | An SEIR whose introduction is DRAWN rather than assumed, with a real-valued environmental reservoir whose starting level is also uncertain.                       |
| [`malaria_two_species`](../ocaml/golden/malaria_two_species.camdl)             | L_B,S_B,E_B,I_B,S_R,E_R,I_R,S_H,E_H,I_H,R_H,L_A,S_A,E_A,I_A,S_r,E_r,I_r               | scenarios, params                               | Two-mosquito-species malaria transmission model Three host populations in swim-lane style: Top:    An.                                                            |
| [`multi_index_beta`](../ocaml/golden/multi_index_beta.camdl)                   | S,I,R × region[2] × age[2]                                                            | —                                               | Multi-index parameters: transmission as a region × age design matrix.                                                                                             |
| [`phenom_mixing_unchecked`](../ocaml/golden/phenom_mixing_unchecked.camdl)     | S,E,I,R                                                                               | forcing                                         | SEIRS with phenomenological α-mixing — the `unchecked_dim` escape hatch.                                                                                          |
| [`polio_age`](../ocaml/golden/polio_age.camdl)                                 | S,E,I,R,V × age[2]                                                                    | intervention, scenarios, tables, params         | Layer 3: Age-structured SEIR+V with SIA targeting under-5s Adds 2-group age stratification and age-specific contact matrix to Layer 1.                            |
| [`polio_shedding_bimodal`](../ocaml/golden/polio_shedding_bimodal.camdl)       | S,R,I__typical__1,I__typical__2,I__prolonged__1                                       | —                                               | Polio shedding as a bimodal residence, expressed with `via hyper_erlang`.                                                                                         |
| [`polio_spatial_5`](../ocaml/golden/polio_spatial_5.camdl)                     | S,E,I,R,V × patch[5]                                                                  | intervention, scenarios, tables, params         | Layer 4: 5-patch spatial SEIR+V with gravity coupling Adds spatial stratification over 5 patches to Layer 1.                                                      |
| [`ross_macdonald`](../ocaml/golden/ross_macdonald.camdl)                       | S_h,I_h,S_v,E_v,I_v                                                                   | obs                                             | Ross-Macdonald malaria model (Ross 1911; Macdonald 1957).                                                                                                         |
| [`seir_age`](../ocaml/golden/seir_age.camdl)                                   | S,E,I,R × age[2]                                                                      | scenarios, tables                               | SEIR with age mixing (§23.3, primitive form)                                                                                                                      |
| [`seir_age_erlang_via`](../ocaml/golden/seir_age_erlang_via.camdl)             | S,E,I,R × age[2] × __recovery_stage[3]                                                | partial-strat, tables                           | Age-stratified SEIR with an Erlang-3 INFECTIOUS period, expressed with `via`.                                                                                     |
| [`seir_age_incidence_sum`](../ocaml/golden/seir_age_incidence_sum.camdl)       | S,E,I,R × age[2]                                                                      | obs, tables                                     | SEIR with age mixing + EXPLICIT cross-strata incidence aggregation on a stratified transition.                                                                    |
| [`seir_age_let_projection`](../ocaml/golden/seir_age_let_projection.camdl)     | S,E,I,R × age[2]                                                                      | obs, tables                                     | Stratified SEIR observing a LET-BOUND projection target.                                                                                                          |
| [`seir_age_table_rates`](../ocaml/golden/seir_age_table_rates.camdl)           | S,E,I,R × age[5]                                                                      | tables                                          | SEIR with age-stratified aging rates from a per-bin table (gh#32).                                                                                                |
| [`seir_cross_dim`](../ocaml/golden/seir_cross_dim.camdl)                       | S,E,I,R × patch[4] × age[3]                                                           | tables, data                                    | SEIR stratified over TWO axes at once — spatial patches x age groups.                                                                                             |
| [`seir_defines_adj`](../ocaml/golden/seir_defines_adj.camdl)                   | S,E,I,R × patch[3]                                                                    | tables, data                                    | Spatial SEIR whose `patch` dimension is DEFINED by a column of a data file.                                                                                       |
| [`seir_defines_patch`](../ocaml/golden/seir_defines_patch.camdl)               | S,E,I,R × patch[3]                                                                    | tables, data                                    | Minimal spatial SEIR whose `patch` dimension is DEFINED by a column of a data file.                                                                               |
| [`seir_erlang`](../ocaml/golden/seir_erlang.camdl)                             | S,E,I,R × latent_stage[3]                                                             | partial-strat, scenarios, params                | SEIR with Erlang-3 latent period (§23.7)                                                                                                                          |
| [`seir_erlang_staged`](../ocaml/golden/seir_erlang_staged.camdl)               | S,E,I,R × latent_stage[3]                                                             | partial-strat, tables                           | SEIR with Erlang-3 latent period, independent per-stage progression rates Demonstrates: 1-D table over a stage dimension used in consecutive transitions          |
| [`seir_erlang_via`](../ocaml/golden/seir_erlang_via.camdl)                     | S,E,I,R × __onset_stage[3]                                                            | partial-strat                                   | SEIR with an Erlang-3 latent period, expressed with the `via` clause.                                                                                             |
| [`seir_observations`](../ocaml/golden/seir_observations.camdl)                 | S,E,I,R                                                                               | obs                                             | SEIR with observations block Demonstrates neg_binomial and bernoulli observation models.                                                                          |
| [`seir_pop_balance`](../ocaml/golden/seir_pop_balance.camdl)                   | S,E,I,R                                                                               | forcing                                         | SEIR reconciled to a known (seasonally varying) census population via `balance`.                                                                                  |
| [`seir_seasonal_importation`](../ocaml/golden/seir_seasonal_importation.camdl) | S,E,I,R                                                                               | intervention                                    | Near-elimination SEIR sustained by annual seasonal importation.                                                                                                   |
| [`seir_seasonal_patch`](../ocaml/golden/seir_seasonal_patch.camdl)             | S,E,I,R × patch[2]                                                                    | forcing                                         | SEIR with patch-indexed sinusoidal seasonality Demonstrates indexed time functions: seasonal[p in patch] Each patch gets its own expanded function …              |
| [`seir_spatial_5_inference`](../ocaml/golden/seir_spatial_5_inference.camdl)   | S,E,I,R × patch[5]                                                                    | obs, tables, forcing                            | 5-patch spatial SEIR — inference stress test                                                                                                                      |
| [`seir_vaccine`](../ocaml/golden/seir_vaccine.camdl)                           | S,E,I,R,V                                                                             | intervention, scenarios, params                 | Layer 1: SEIR+V with SIA intervention Baseline polio model: vaccination compartment + supplemental immunisation activity.                                         |
| [`seir_vaccine_seasonal`](../ocaml/golden/seir_vaccine_seasonal.camdl)         | S,E,I,R,V                                                                             | intervention, scenarios, forcing, params        | Seasonal variant of `seir_vaccine`: SEIR+V with sinusoidal forcing and SIA rounds.                                                                                |
| [`sia_anchored_dates`](../ocaml/golden/sia_anchored_dates.camdl)               | S,I,R,V × region[2]                                                                   | intervention, scenarios, tables, calendar, data | SIA (Supplementary Immunization Activity) schedule supplied as anchored calendar DATES read from a table — the common operational input shape for measles/polio … |
| [`sia_instance_enable`](../ocaml/golden/sia_instance_enable.camdl)             | S,I,R,V × region[2]                                                                   | intervention, scenarios                         | gh#130: a scenario may enable/disable a single FULLY-EXPANDED intervention-instance name (e.g.                                                                    |
| [`sir_basic`](../ocaml/golden/sir_basic.camdl)                                 | S,I,R                                                                                 | scenarios, params                               | Bare SIR: 3 compartments, 2 transitions (§23.1)                                                                                                                   |
| [`sir_coupling`](../ocaml/golden/sir_coupling.camdl)                           | S,I,R × age[2]                                                                        | scenarios, tables                               | Age-structured SIR with contact matrix transmission Demonstrates explicit age-stratified coupling via sum(b in age, C[a,b] * I[b] / N[b]) The old coupling[] …    |
| [`sir_demography`](../ocaml/golden/sir_demography.camdl)                       | S,I,R                                                                                 | scenarios                                       | SIR with demography (§23.2)                                                                                                                                       |
| [`sir_dim_annotated`](../ocaml/golden/sir_dim_annotated.camdl)                 | S,I,R                                                                                 | —                                               | SIR model with explicit dimension annotations on parameters.                                                                                                      |
| [`sir_dt`](../ocaml/golden/sir_dt.camdl)                                       | S,I,R                                                                                 | dt                                              | SIR with an explicit discretization step `dt` in the simulate block (gh#161).                                                                                     |
| [`sir_five_age`](../ocaml/golden/sir_five_age.camdl)                           | S,I,R × age[5]                                                                        | scenarios, tables, params                       | SIR with 5 age groups and consecutive aging (§23.6, simplified)                                                                                                   |
| [`sir_guarded_foi`](../ocaml/golden/sir_guarded_foi.camdl)                     | S,I,R × patch[3]                                                                      | scenarios, tables                               | SIR metapopulation with a divide-by-zero-guarded force of infection.                                                                                              |
| [`sir_init_table`](../ocaml/golden/sir_init_table.camdl)                       | S,I,R × patch[3]                                                                      | tables, data                                    | SIR with table-indexed initial conditions Verifies that table lookups work in init blocks: S[p] = N0[p] - I0 where N0 is a read() table keyed by patch.           |
| [`sir_overdispersion`](../ocaml/golden/sir_overdispersion.camdl)               | S,I,R                                                                                 | —                                               | SIR with extra-demographic stochasticity (He et al.                                                                                                               |
| [`sir_patches_5`](../ocaml/golden/sir_patches_5.camdl)                         | S,I,R × patch[5]                                                                      | params                                          | Five-patch SIR: simple spatial model using numeric patch indices (p0..p4).                                                                                        |
| [`sir_priors`](../ocaml/golden/sir_priors.camdl)                               | S,I,R                                                                                 | priors                                          | SIR model with explicit priors declared in the DSL.                                                                                                               |
| [`sir_reservoir`](../ocaml/golden/sir_reservoir.camdl)                         | S,I,R,W                                                                               | ode                                             | SIR with an environmental reservoir compartment (real-valued, ODE).                                                                                               |
| [`sir_reservoir_mixed`](../ocaml/golden/sir_reservoir_mixed.camdl)             | S,I,R,W1,W2,W3,W4,W5                                                                  | ode                                             | Gate fixture: a >=8-term aggregate mixing INTEGER and REAL compartments, so MixedPopSum's int-then-real fold order is actually exercised.                         |
| [`sir_spatial_sum`](../ocaml/golden/sir_spatial_sum.camdl)                     | S,I,R × patch[4]                                                                      | scenarios, tables                               | 4-patch SIR with gravity coupling via sum(q in patch, w[p,q] * I[q] / N[q]).                                                                                      |
| [`sir_spatial_where`](../ocaml/golden/sir_spatial_where.camdl)                 | S,I,R × patch[6]                                                                      | tables, data                                    | SIR on a line of 6 patches with radius-limited spatial coupling (gh#185).                                                                                         |
| [`sir_two_overdispersed`](../ocaml/golden/sir_two_overdispersed.camdl)         | S,I,R,V                                                                               | —                                               | SIR with TWO overdispersed transitions sharing the same source group.                                                                                             |
| [`sir_two_patch`](../ocaml/golden/sir_two_patch.camdl)                         | S,I,R × patch[2]                                                                      | —                                               | Two-patch SIR: demonstrates indexed parameters N[patch] and R0[patch]                                                                                             |
| [`sir_two_patch_long_obs`](../ocaml/golden/sir_two_patch_long_obs.camdl)       | S,I,R × patch[2]                                                                      | obs                                             | Two-patch SIR with a STRATIFIED observation header `cases[p in patch]`.                                                                                           |
| [`sirv_anchored_calendar`](../ocaml/golden/sirv_anchored_calendar.camdl)       | S,I,R,V                                                                               | intervention, scenarios, forcing, calendar      | Anchored SIRV with school-term seasonal forcing + calendar-aligned campaigns.                                                                                     |
| [`surveillance_likelihoods`](../ocaml/golden/surveillance_likelihoods.camdl)   | S,E,I,R                                                                               | obs                                             | SEIR with four surveillance streams — one per under-covered likelihood family.                                                                                    |
| [`zinb_vector_catch`](../ocaml/golden/zinb_vector_catch.camdl)                 | S,I,R                                                                                 | obs                                             | Zero-inflated NegBinomial observation of focal-vector catch counts.                                                                                               |

## Literature replications and analytic references

Models reproducing a published result or a closed-form answer. These are the external-validation surface: camdl's output is checked against the reference, not merely against itself. — `tests/external/cases/*/model.camdl` (5 models)

| Model                                                                                              | Structure | Features | Description                                                                                                            |
| -------------------------------------------------------------------------------------------------- | --------- | -------- | ---------------------------------------------------------------------------------------------------------------------- |
| [`boarding_school_sir`](../tests/external/cases/boarding_school_sir/model.camdl)                   | S,I,R     | params   | Boarding-school flu outbreak — reference for pomp's canonical SIR tutorial (Anderson & May 1991; pomp::bsflu dataset). |
| [`he2010_forward`](../tests/external/cases/he2010_forward/model.camdl)                             | S,E,I,R   | params   | Measles SEIR — exact He et al.                                                                                         |
| [`he2010_pfilter_loglik`](../tests/external/cases/he2010_pfilter_loglik/model.camdl)               | S,E,I,R   | params   | Measles SEIR — exact He et al.                                                                                         |
| [`he2010_pfilter_loglik_sparse`](../tests/external/cases/he2010_pfilter_loglik_sparse/model.camdl) | S,E,I,R   | params   | Measles SEIR — exact He et al.                                                                                         |
| [`sir_analytical`](../tests/external/cases/sir_analytical/model.camdl)                             | S,I,R     | params   | Bare SIR at R0 = 3, used as the harness dogfood case.                                                                  |

## ODE oracles

Deterministic models whose trajectories are checked against an independent ODE integrator (gh#166). — `tests/external/ode_oracle/models/*.camdl` (3 models)

| Model                                                    | Structure   | Features | Description                                                                                                                                                       |
| -------------------------------------------------------- | ----------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`seir`](../tests/external/ode_oracle/models/seir.camdl) | S,E,I,R     | —        | gh#166 Phase B incidence oracle — frequency-dependent SEIR, no vital dynamics.                                                                                    |
| [`sir`](../tests/external/ode_oracle/models/sir.camdl)   | S,I,R       | —        | gh#166 Phase B incidence oracle — frequency-dependent SIR, no vital dynamics.                                                                                     |
| [`tb`](../tests/external/ode_oracle/models/tb.camdl)     | S,Lf,Ls,I,R | —        | gh#166 Phase B incidence oracle — 2-stage-latency TB with timescale separation: fast progression from early latency vs slow (per-decade) reactivation from late … |

## Parameter-recovery cases

Models used to fit synthetic data generated from known truth, to check that the inference stack recovers the parameters it was given. — `tests/recovery/cases/*/model.camdl` (2 models)

| Model                                                      | Structure | Features   | Description                                                                                                                                            |
| ---------------------------------------------------------- | --------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`seir_age`](../tests/recovery/cases/seir_age/model.camdl) | S,E,I,R   | fit-config | Age-structured SEIR — first case in the synthetic-data parameter-recovery harness.                                                                     |
| [`sir`](../tests/recovery/cases/sir/model.camdl)           | S,I,R     | fit-config | Closed SIR — the book's canonical getting-started model (camdl-book/guide/getting-started/sir_priors.camdl), used here as the clean, well-identified … |

## Engine fixtures

Models pinning specific simulation-engine behaviour: coupling semantics, seed timing, lineage tracking, and optimiser A/B gates. — `rust/crates/sim/tests/fixtures/*.camdl` (8 models)

| Model                                                                              | Structure                   | Features        | Description                                                                                                                 |
| ---------------------------------------------------------------------------------- | --------------------------- | --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [`human_migration`](../rust/crates/sim/tests/fixtures/human_migration.camdl)       | S,I,R × patch[2]            | —               | Human migration: transmission is LOCAL within each patch; infectives physically MOVE between patches (m·I[p]).              |
| [`licm_ab`](../rust/crates/sim/tests/fixtures/licm_ab.camdl)                       | S,E,I,R × patch[4]          | tables, data    | A/B gate fixture for the gh#272 loop-invariant code-motion (LICM) pass.                                                     |
| [`licm_grad_fd`](../rust/crates/sim/tests/fixtures/licm_grad_fd.camdl)             | S,E,I,R                     | obs             | Finite-difference gradient gate for the gh#272 LICM pass (gh#284 coverage).                                                 |
| [`pathogen_migration`](../rust/crates/sim/tests/fixtures/pathogen_migration.camdl) | S,I,R × patch[2]            | —               | Pathogen migration: patches couple through the FORCE OF INFECTION only.                                                     |
| [`seed_timing`](../rust/crates/sim/tests/fixtures/seed_timing.camdl)               | S,I,R                       | obs             | Seed-timing model: SIR seeded by a smooth importation pulse (mechanism B of the 2026-05-20 seed-timing-inference proposal). |
| [`seed_timing_dated`](../rust/crates/sim/tests/fixtures/seed_timing_dated.camdl)   | S,I,R                       | obs, calendar   | Seed-timing model, calendar-dated variant (2026-05-22 calendar-time §9.10).                                                 |
| [`sparse_coupling_ab`](../rust/crates/sim/tests/fixtures/sparse_coupling_ab.camdl) | S,E,I,R × patch[8] × age[1] | tables, forcing | A/B gate fixture for the sparse-coupling constant-fold pass.                                                                |
| [`spatial_lineage`](../rust/crates/sim/tests/fixtures/spatial_lineage.camdl)       | S,I,R × patch[2]            | tables          | Two-patch SIR with #[lineage] transmission and an ASYMMETRIC contact / coupling matrix.                                     |

## Corner cases

Models pinning behaviour at awkward boundaries — off-grid observations and interventions, coincident lifecycle events, fractional end times. — `tests/fixtures/corner_cases/*.camdl` (11 models)

| Model                                                                                             | Structure | Features          | Description                                                                      |
| ------------------------------------------------------------------------------------------------- | --------- | ----------------- | -------------------------------------------------------------------------------- |
| [`all_lifecycle`](../tests/fixtures/corner_cases/all_lifecycle.camdl)                             | S,I,R     | obs, intervention | Corner case: ALL LIFECYCLE STAGES at a coincident boundary.                      |
| [`coincident_obs_intervention`](../tests/fixtures/corner_cases/coincident_obs_intervention.camdl) | S,I,R     | obs, intervention | Corner case: COINCIDENT OBSERVATION + INTERVENTION.                              |
| [`dt_rate`](../tests/fixtures/corner_cases/dt_rate.camdl)                                         | S,I,R     | obs               | Corner case: a transition RATE that references `dt` (Ir.Dt, gh#54).              |
| [`event_drain_fusion`](../tests/fixtures/corner_cases/event_drain_fusion.camdl)                   | A,B,C     | intervention      | Corner case: DRAINING-EVENT RESIDUAL READ (gh#217).                              |
| [`event_intervention_agree`](../tests/fixtures/corner_cases/event_intervention_agree.camdl)       | A,B       | intervention      | Corner case: CROSS-BACKEND LIFECYCLE AGREEMENT (M1 canonicalization).            |
| [`fractional_output_end`](../tests/fixtures/corner_cases/fractional_output_end.camdl)             | S,I,R     | —                 | Corner case: FRACTIONAL SIMULATE END.                                            |
| [`gh70_absorbing_importation`](../tests/fixtures/corner_cases/gh70_absorbing_importation.camdl)   | I,R       | intervention      | Corner case: ABSORBING INITIAL STATE + SCHEDULED IMPORTATION (gh#70 regression). |
| [`multi_effect_same_time`](../tests/fixtures/corner_cases/multi_effect_same_time.camdl)           | A,B,C     | intervention      | Corner case: MULTIPLE SCHEDULED EFFECTS AT THE SAME BOUNDARY.                    |
| [`off_grid_intervention`](../tests/fixtures/corner_cases/off_grid_intervention.camdl)             | S,I,R     | intervention      | Corner case: OFF-GRID INTERVENTION.                                              |
| [`off_grid_obs`](../tests/fixtures/corner_cases/off_grid_obs.camdl)                               | S,I,R     | obs               | Corner case: OFF-GRID OBSERVATION CADENCE.                                       |
| [`seasonal_drift`](../tests/fixtures/corner_cases/seasonal_drift.camdl)                           | S,I       | forcing           | Corner case: SUBSTEP-TIME CONVENTION drift (forward accumulation vs PGAS s*dt).  |

## Feature and regression fixtures

Models exercising one feature or reproducing one fixed bug. — `tests/fixtures/*.camdl`, `tests/fixtures/*/*.camdl`, `tests/fixtures/*/*/*.camdl` (8 models)

| Model                                                                                                 | Structure        | Features                          | Description                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`contrasts_showcase`](../tests/fixtures/contrasts/contrasts_showcase.camdl)                          | S,I,R,D,V        | intervention, scenarios, calendar | Counterfactual-contrasts showcase fixture (proposal 2026-06-25).                                                                                           |
| [`gh208_sparse_negative_rate`](../tests/fixtures/regression/gh208_sparse_negative_rate.camdl)         | S,I,R            | —                                 | gh#208 regression: a transition rate that goes NEGATIVE on a sparse Gillespie propensity update must raise SimError::NegativePropensity, not be silently … |
| [`polio_afp_es_2patch`](../tests/fixtures/polio_afp_es_2patch.camdl)                                  | S,E,I,I_shed,R   | params                            | Spatial polio with TWO surveillance streams on DIFFERENT cadences.                                                                                         |
| [`quantities_showcase`](../tests/fixtures/quantities/quantities_showcase.camdl)                       | S,I,R × patch[2] | obs                               | Generated-quantities showcase fixture (proposal 2026-06-25).                                                                                               |
| [`reactive_indexed_patch_sia`](../tests/fixtures/reactive/reactive_indexed_patch_sia.camdl)           | S,I,V × patch[2] | obs, intervention                 | gh#204 PR1 compiler/IR golden: an INDEXED reactive policy.                                                                                                 |
| [`reactive_sir_observed_threshold`](../tests/fixtures/reactive/reactive_sir_observed_threshold.camdl) | S,I,R,V          | obs, intervention                 | gh#204 compiler/IR golden: a minimal SIR with one observation stream and a single reactive policy.                                                         |
| [`seir_seasonal_lagged`](../tests/fixtures/gradient/seir_seasonal_lagged.camdl)                       | S,E,I,R          | forcing                           | Gradient regression fixture for the lagged-forcing autodiff bug (docs/dev/incidents/2026-07-05-lagged-forcing-autodiff-wrong-gradient.md).                 |
| [`sir_patches`](../tests/fixtures/mre/model/sir_patches.camdl)                                        | S,I,R            | data                              | Minimal 2-patch SIR with a read() population table and weekly case observations — a self-contained fixture for `camdl mre fit`.                            |

## Proposal fixtures

Before/after models attached to a design proposal, showing what a language change buys. — `docs/dev/proposals/fixtures/*.camdl` (2 models)

| Model                                                                             | Structure                        | Features | Description                                                                                                                                        |
| --------------------------------------------------------------------------------- | -------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`garki_post_proposal`](../docs/dev/proposals/fixtures/garki_post_proposal.camdl) | X,Y1,Y2,Y3,Sv,Ev,Iv              | —        | Garki post-proposal: uses all five Wave-2 features that have landed.                                                                               |
| [`garki_pre_proposal`](../docs/dev/proposals/fixtures/garki_pre_proposal.camdl)   | X,Y1_symp,Y1_asym,Y2,Y3,Sv,Ev,Iv | —        | Garki pre-proposal: the same malaria model written WITHOUT the Wave-1 and Wave-2 features, for side-by-side comparison with `garki_post_proposal`. |

## Not models

34 further `.camdl` files exist only to be REJECTED — they pin the text and code of a compiler diagnostic, and none of them describe a disease. They are excluded from the tables above.

- `ocaml/golden/errors/` (11) — dimension/type errors the compiler must reject
- `ocaml/test/errors/` (6) — lex/parse/name-resolution errors
- `ocaml/test/lints/` (17) — lint and diagnostic fixtures (clean + expected-warning pairs)

## Running one

```bash
camdl check   ocaml/golden/sir_basic.camdl
camdl simulate ocaml/golden/sir_basic.camdl --params ocaml/golden/sir_basic.params.toml
```

A model without a `params` flag leaves its parameters estimated, so `simulate` needs values supplied with `--params` or `--set`.

Reading this through `camdl docs examples` with no checkout on disk? The sources are a sparse clone away (~5 MB), and the paths above are relative to its root:

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/vsbuffalo/camdl .camdl-source
cd .camdl-source && git sparse-checkout set docs ocaml/golden && cd ..
```

You do not need this repository to build models of your own — see `camdl docs getting-started`.
