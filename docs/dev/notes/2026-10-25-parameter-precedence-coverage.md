# Parameter precedence: test coverage audit

Date: 2026-10-25 Tags: params_resolver, precedence, notices, run identity, tests

## Question

Is the parameter precedence system (docs/camdl-run-spec.md §1.3,
`rust/crates/cli/src/params_resolver.rs`) fully tested? In particular: for each
route by which a user sets a parameter, is there a test that the run uses the
value, that equivalent routes give equal output, and that every replacement of
one tier's value by another's is reported (stderr notice, correct severity) and
recorded (`run.json` `parameters_provenance`)?

The tiers, last wins: 1 model default; 2 fit.toml `[fixed]`; 3 `--params` /
`--fixed-file` files in order; 3.5 a draw row or sweep point; 4 the scenario
(`set` and `scale` folded into one value); 5 `--param` / `--fixed` /
`--param-vec`. A replacement is a warning when the replaced value came from a
params file carrying a fit `[provenance]` block, info otherwise; replacing the
model default, or replacing a value with an equal one, is not reported.

## Findings

**1. `fit run` fits at a value other than the one its notice reports.** A
fit.toml whose top-level `scenario` sets a parameter that `[fixed]` also sets:
the resolver applies the scenario and prints
`info: scenario`fastrec`replaced gamma = 0.2 (from fit.toml [fixed]) with 0.3`,
then `fit/runner.rs` (the "2. Apply fixed numeric values" loop after the
resolver) writes `[fixed]`'s 0.2 back into `base_params`, the vector every chain
starts from. The fit runs at 0.2: its `mle_params.toml` records `gamma = 0.2`
and its β estimate is bit-identical to the same fit with no scenario. The
resolver was called; its result was overwritten. The same loop makes a
`fit run --sweep gamma=…` point beat a fit scenario that sets gamma, while the
notice says the scenario won. Not fixed: deleting the loop makes the red test
pass, but then the scenario collapses a `fit run --sweep` over that parameter to
one value — the case `simulate` refuses (gh#572) — so the fix needs a collision
rule on `fit run` as well, which is a design decision.

**2. `profile --starts from_params=FILE` and `--starts from_mle=FIT` enter as
the model default.** `seed_params_from_init_method` (`profile.rs`) writes every
value in the file into the model before the resolver, so the resolver sees
tier 1. Consequences, both verified: without a scenario, `run.json` records
`"source": "model_default"` for values the model never declared; with
`--scenario base`, the scenario replaces the file's `gamma = 0.2` with 0.1 and
nothing is printed or recorded. For `from_mle` the replaced value is a fit's
estimate, exactly the case the rule makes a warning.

**3. Only `simulate` and `profile` record overrides.** `pfilter`, `survey` and
`fit run` print the notices but record none: `pfilter`'s `run.json` carries only
the scored `params`; `survey`'s carries `fixed`; `fit run` writes
`parameters_provenance: {}` into `fit.meta.json` unconditionally (`fit/mod.rs`,
"populated by the fit-finalization layer", which does not exist). The spec's
"records it in each leaf's `run.json`" holds for two verbs. Contrasts
(`fit/contrasts.rs`) resolve each arm without emitting notices; the same
scenarios over the same draws are reported by the `fit predict` run that
computes them, so nothing is unreported, but the contrast path is silent on its
own.

**4. Smaller observations.** A one-column `--draws FILE` is refused ("needs at
least 2 columns", `main.rs`), so a single-parameter draws file cannot be passed.
`--sweep beta=lin(0.3,0.3,1)` is refused (n ≥ 2); a one-value list
`--sweep beta=0.3` is the one-point sweep. On `fit predict`, a `--sweep` value
is written into each posterior draw row (same tier 3.5), so it replaces the
draw's value without a notice; its `sweep:<param>` column is what shows it.

## Coverage, by route

Before = tests that existed; after = with this change. `routes` =
`tests/parameter_precedence_routes.rs`, `matrix` = the `params_resolver.rs`
precedence-matrix unit tests.

| Route                                 | Tier  | Verbs                                                | Before                                                                                | After                                                |
| ------------------------------------- | ----- | ---------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| model default                         | 1     | all                                                  | resolver unit tests                                                                   | + matrix (no notice from any tier)                   |
| fit.toml `[fixed]`                    | 2     | fit run, profile/survey `--fit`, simulate `--fit`    | resolver units; `generated_draws_varying_only_gh949::fit_fixed_resolves_below_params` | + matrix; finding 1 red test                         |
| `--params` (repeated)                 | 3     | simulate, pfilter, dev eval, mre simulate            | `param_override_notices` (simulate)                                                   | + routes: pfilter, dev eval                          |
| `--fixed-file`                        | 3     | profile                                              | resolver units only                                                                   | + routes: profile                                    |
| `mle_params.toml` as `--params`       | 3     | simulate, pfilter                                    | `param_override_notices::a_fits_estimate_replaced_by_a_scenario_is_a_warning`         | + matrix: warning under every higher tier            |
| `--sweep`, `--sweep-file`             | 3.5   | simulate, fit predict (sweep), fit run (tier 2!)     | `simulate_sweep`, `scenario_coordinate_collision_gh572`, `fit_predict_sweep`          | + routes: one-point sweep = other routes             |
| `--draws FILE`                        | 3.5   | simulate                                             | `scenario_runtime_application`, gh572                                                 | + routes                                             |
| `--draws prior/uniform/posterior`     | 3.5   | simulate; fit predict, contrasts (posterior)         | gh949, `simulate_posterior_cap`, `contrasts_e2e`                                      | + routes: posterior × `--param` = × scenario         |
| scenario `set`                        | 4     | simulate, pfilter, profile, survey, fit run, predict | `scenario_runtime_application`, `scenario_identity_gh573`                             | + routes on simulate, pfilter, predict               |
| scenario `scale`                      | 4     | same                                                 | `scenario_runtime_application::scenario_scale_multiplies_mu_value`                    | + routes; matrix                                     |
| `extends` (compile-time)              | 4     | same                                                 | horizon tests only                                                                    | + routes: inherited set, inherited scale, expression |
| `--param`                             | 5     | simulate, pfilter, dev eval                          | `param_override_notices::param_over_a_scenario_is_reported`                           | + routes; matrix                                     |
| `--fixed`                             | 5     | profile, survey                                      | profile/survey runs (no override check)                                               | + routes: profile, survey notices; profile record    |
| `--param-vec`                         | 5     | simulate                                             | `simulate_identity_parity::param_vec_values_split_the_cas_leaves`                     | unchanged                                            |
| `--enable`/`--disable`                | —     | simulate, pfilter, profile, predict                  | `simulate_arm_label`, `intervention_event_defaults`                                   | unchanged (not a value)                              |
| `starts = { from_mle / from_params }` | 1 (!) | fit run (chain starts), profile (`--starts`)         | gh871, gh904                                                                          | finding 2                                            |
| `--init-state`                        | —     | simulate                                             | gh641, gh697                                                                          | unchanged (state, not a parameter)                   |

## Coverage, by tier pair

Before, the override tests covered: file → later file, file → scenario (info and
fit-estimate warning), scenario → `--param`, file → draw/sweep (multi-cell),
equal value, model default. Value-only tests (no notice check) covered
draw/sweep vs scenario `set`/`scale` and vs `--param`. Missing: every pair with
fit.toml `[fixed]` as the replaced tier; file → `--param`; draw/sweep → scenario
and → `--param` notices; the fit-estimate warning under a later file, a
draw/sweep point and `--param`; the `run.json` record for any pair but file →
scenario.

After: `every_tier_pair_higher_wins_is_reported_and_is_recorded` walks all 23
ordered pairs of the seven tier kinds (fit.toml, file, fit-estimate file,
draw/sweep, scenario set, scenario scale, `--param`), asserting the value, the
source, the one `Override`, the notice's exact text and severity, and the
`ParameterProvenance` record.
`an_equal_value_from_any_higher_tier_is_not_an_override` and
`replacing_the_model_default_from_any_tier_is_not_an_override` cover the two
non-overrides for every pair and tier, and
`all_six_tiers_on_one_parameter_replace_in_spec_order` the full chain.

## Route equivalence and identity

`simulate_every_route_to_one_value_gives_one_trajectory` sets β = 0.3 by eleven
routes under seed 7. Trajectory and observation output are byte-identical across
all of them, once the one-point sweep's `point_id`, `sweep:beta` and `seed`
columns are removed (asserted to read 0, 0.3, 7). Every route has the same
`model`, `config`, `params` and `seed` level hashes: one resolved value is one
`params` level, a single cell runs at the base seed, and a design point never
enters the seed. The no-scenario routes share one `run_id`. A scenario route has
its own `scenario` level, because that level hashes the scenario's effective
delta: `set_b`, an `extends` child inheriting that set, and a child computing
`beta * 2.0` from a parent's 0.15 share one level; a `scale` delta is a
different level even where it lands on the same value. That split is the
identity rule, not a defect.

The same holds for `pfilter` (`--params`, `--param`, file then `--param`, and a
scenario setting both parameters give one log-likelihood and one `run_id`), for
`simulate --draws posterior` (`--param k=12` = `--scenario k_alt`), and for
`fit predict` (`--scenario k_alt` = `--sweep k=12`, band for band).

## Mutation checks

Each class of test was run against a deliberately broken resolver or verb and
went red; the code was then restored.

| Break                                                      | Red                                                       |
| ---------------------------------------------------------- | --------------------------------------------------------- |
| tier 3.5 block moved after tier 4                          | 5 resolver tests (matrix, equal-value, chain, 2 existing) |
| fit-estimate replacement reported as info                  | matrix, chain, 1 existing                                 |
| `--param` replacements not recorded                        | matrix, chain, 2 existing                                 |
| `ParameterProvenance` drops overrides                      | matrix, chain                                             |
| scenario `scale` × (1 + 1e-7) / × 1.1                      | routes: `params` level / trajectory differs               |
| `pfilter` ignores `--param`                                | routes: pfilter                                           |
| `fit predict` does not write the sweep value into the rows | routes: predict bands differ                              |
| engine stops emitting run notices; profile drops overrides | 4 routes tests                                            |

## Next

Findings 1–3 are for the maintainer: 1 needs a decision on `fit run --sweep` ×
scenario before the re-application loop can go; 2 needs a tier for a starts file
(or a decision that a starts file sets only estimated parameters); 3 is wiring
`ParameterProvenance` into the `pfilter`, `survey` and `fit run` records.
