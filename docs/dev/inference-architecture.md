# Inference architecture: the concepts and invariants every proposal respects

Normative. A proposal that touches the layer between a fit config and an
inference engine states which of these invariants it moves toward, and may not
introduce a new violation. Each invariant carries its status on `origin/main` at
`a9a89178`, verified by reading the cited lines; nothing here was measured by
running a fit.

## The five concepts

A fit answers one statistical question, attempts it with one algorithm, records
what it ran, measures what came out, and is then judged. Those are five objects,
and they change for different reasons.

**`ResolvedProblem`** is the statistical target: every scientific input that
defines it, and nothing that only affects how it is attempted. It contains

- which parameters are fixed, and their values;
- which parameters are estimated;
- the priors, including their hyperparameters;
- the effective constraints: each estimated parameter's support and search box;
- the observation model and the data actually used;
- the deterministic model transformations and the interventions;
- `ic_free`;
- the training window.

It never contains a seed or a start. A fixed value belongs to the problem; a
drawn initial value for an estimated parameter belongs to the `ExecutionPlan`.
It is built once per fit segment and shared by every method and seed of that
segment.

The parameter **roles** are those of `proposals/2026-10-26-parameter-roles.md`
§1–§2 and `proposals/2026-09-27-coordinate-kind-design-vs-sample.md` §2:

- **fixed**: one value per scenario arm, part of the problem;
- **estimated**: searched or sampled by the method within its search box; its
  prior is part of the problem;
- **design**: a value on a swept axis the user chose (a sweep value, a grid row,
  a profile's focal point), fixed within each cell. Each design point defines
  its own problem, in which the parameter is fixed at that value; the axis is a
  partition axis, and results are never pooled across it;
- **sampled**: a quantity drawn from a stated probability measure (a prior, a
  posterior, a uniform law over declared bounds). The measure is part of the
  problem; the realised draw belongs to the cell that uses it, and through its
  seed to the execution.

How each estimation method treats them, from parameter-roles §4.4 and §6.4:

| role      | IF2, `nl-sbplx`, `nl-bobyqa`, `nl-lbfgs`                                 | PMMH, MH, PGAS                                                                                                         |
| --------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| fixed     | held at its value                                                        | held at its value                                                                                                      |
| estimated | maximized over the search box; the prior unread                          | sampled from the posterior, the prior truncated to the search box (PGAS inferred to follow PMMH, parameter-roles §6.4) |
| design    | one fit per design point (`fit run`); `profile` runs IF2 per focal point | one fit per design point (`fit run`)                                                                                   |
| sampled   | refused (`fit run` accepts no sampled role)                              | refused                                                                                                                |

A sampled role is accepted by `simulate` (optional) and required by
`fit predict` (the posterior), neither of which estimates.

**`ExecutionPlan`** is how one algorithm attempts that problem: a per-method
plan (IF2's perturbations and cooling, a sampler's proposal and adaptation
settings, an optimizer's tolerances), the drawn starts, the chain count and the
seed. It is fully resolved and validated before launch, and it is the only
configuration an engine receives. A setting belongs to exactly one method's arm,
and a setting one method needs is never carried, as a placeholder, by another.
The split is defined in `proposals/2026-10-29-method-scoped-settings.md` §6.1
(branch `docs/method-scoped-settings`).

**`RunRecord`** (`run.json`) is what was executed: the factored identity levels
and the `run_id` derived from them, the engine and IR versions, and the parents.
It describes what happened and never depends on an assessment policy. Every
stored artifact resolves to exactly one `run_id`. A label is a name for a run
and never stands in for its identity. A parent is recorded by its `run_id`.

**Result records** are typed per method: the measurements, estimates and draws a
method produced, and nothing it did not. An optimizer's record holds its point
estimate and the measurements of whether its chains agree; a sampler's holds its
draws and the measurements of whether its chains mix. Records may share
measurement components (a likelihood-estimate type, a dt-check ladder); what is
excluded is a field with no meaning for the method that wrote it, and a sentinel
standing in for one.

**`Assessment`** is A(recorded measurements, policy): an interpretation of
stored measurements under a stated `DiagnosticPolicy` (the thresholds:
acceptable Â, decibans spread, R̂, ESS, optimizer agreement, dt-check tolerance),
computed on read. Changing the policy changes the assessment; it never changes a
stored measurement, a run record or an identity. Every reported verdict names
the policy it was computed under. When an assessment is persisted (the as-run
verdict at the end of a run, which a later reader compares against) it goes in a
separate **assessment record**: policy-versioned, referencing the `run_id` and
the policy digest, stored beside the leaf but outside `run.json` and outside
every addressed artifact. The design is
`proposals/2026-10-29-result-records-and-assessment.md`.

## The invariants

**I1. The problem does not depend on how it is attempted.** _Violated today._
`[estimate].rw_sd` and `perturb_only_at_t0` (IF2 and MH settings) and
`[estimate].start` live in `EstimateSpecV2` (`cli/src/fit/config_v2.rs:574`;
`perturb_only_at_t0` at `:613`, `rw_sd` at `:619`, `start` at `:625`), which the
fit level hashes whole (`fit_config_blob_hash`, `cas.rs:433-435`). The value
resolver draws starts with the seed (`resolve_fit_parameters(fit, seed)`,
`runner.rs:171-174`, draw at `:245-265`). Branch `docs/method-scoped-settings`
moves both out of the problem: the step keys into `[method]` tables (its §4),
and `start` out of the fit-level digest through `Problem::identity_view()`, with
the method level hashing the written `[estimate].start` values (`starts_base`)
beside the starts rule (its §5.2, a coordinated amendment to parameter-roles).

**I2. An engine receives one resolved, validated plan, with nothing for other
methods.** Resolved means no `Option` that lets the engine pick a default later:
engines execute plans, and never interpret user configuration or supply a
configuration default. Whether a value was written, inherited or defaulted is
provenance, recorded and never read by the engine. Runtime adaptation is a
separate thing: NUTS step-size and mass-matrix adaptation, PMMH's proposal-scale
adaptation and IF2's cooling evolve during the run under rules the plan fixes,
and are not configuration resolution. _Violated today._

- `FitRunConfig::build` takes IF2's `n_iterations`, `cooling` and
  `cooling_target_iters` for every method (`runner.rs:325-335`), and PMMH, PGAS,
  NLopt and pfilter pass placeholders (`pmmh.rs:133-140`, `pgas.rs:304-310`,
  `nlopt_stage.rs:80-92`, `mod.rs:1752-1758`).
  `FitRunConfig.if2_config:
  IF2Config` (`runner.rs:106`) carries the
  particle-filter settings every method reads (`if2.rs:110-147`), and
  `FitRunConfig.gate` (`runner.rs:124`) an IF2 verdict threshold.
- `EstimatedParam` mixes the parameter's identity, start and IF2 tuning
  (`sim/src/inference/types.rs:48-74`), and carries `rw_sd_auto`, whether the
  step was defaulted (`:66`): provenance inside an engine type.
- Defaults are filled after identity is taken. An omitted `rw_sd` is computed
  from the bounds inside `FitRunConfig::build` (`runner.rs:1411-1413`), and the
  method identity sees only the written `None`. `burn_in` and `thin` are
  `Option<usize>` in the method (`config_v2.rs:1067-1069`) and are filled with
  `DEFAULT_BURN_IN` / `DEFAULT_THIN` at dispatch (`pgas.rs:217-218`, `:237-238`;
  `pmmh.rs:37-38`, `:54-55`, `:72-73`), so a file that writes the default and
  one that omits it key differently, and a changed default changes the execution
  without changing the method payload.

**I3. Every accepted user option has a defined, observable effect on the
computation, or is refused.** A redundant option the program recognises as
redundant (a value equal to the default, a setting another one makes moot) may
be accepted if the run says so; a silent no-op never is. _Violated today._
`--cooling-target-iters` with an `nl-sbplx` method is accepted and read by
nothing: the NLopt arm of the override application has no cooling field
(`config_v2.rs:1605-1611`). Branch `fix/settings-silent-noops` adds the refusal
and its test.

**I4. The sim crate takes no CLI, filesystem or store types.** A source run is
an opaque identifier there, never a path or a store handle. _Holds for CLI
types_: `sim/Cargo.toml` does not depend on `cli`. _Violated in one place for
the filesystem_: `DiagnosticCollector::write_json` writes a file from the
inference module (`sim/src/inference/diagnostic.rs:779-785`); every other
`std::fs` use under `sim/src/inference/` is in a test module (verified by
locating each one against its file's first `#[cfg(test)]`).

**I5. Every artifact resolves to exactly one run; labels never substitute for
the identity; parents are recorded by run ID; a run record is never
policy-dependent.** Two executions of one `run_id` produce the same `run.json`
apart from facts about the execution itself (time, host, argv), whatever
assessment policy was in force. _Violated today for labels._ `@label` resolves
to a problem folder, which holds every run of one problem
(`cli/src/fit/handle.rs:133-141`, the label read from the segment's
`fit.meta.json`); branch `fix/gh896-handles` moves the label to the leaf.
Parents hold: a warm-start source that is a stored run is recorded by its
`run_id` and the consumed file's digest (`cas.rs:658-670`). A source directory
with no readable `run.json` is recorded by the consumed file's digest in the
`run_id` slot (`cas.rs:661-665`), which names content rather than a run.

**I6. Each method writes its own result type.** _Violated today._ One flat
`FitState` (`cli/src/fit/state.rs:19-182`) is written by IF2, NLopt, PMMH, MH
and PGAS, each leaving the others' fields empty, and NLopt stores its
per-parameter relative range under the field named for IF2's Â
(`tail_chain_agreement`, `nlopt_stage.rs:297`).

**I7. One assessment rule per measurement type, computed on read under a stated
policy.** An assessment reports a diagnostic it could not compute as
unavailable, never as a silent pass or fail. It keeps numerical-execution
diagnostics (divergences, the dt-check, particle-filter degeneracy) apart from
statistical-estimation diagnostics (R̂, ESS, Â, the decibans spread). _Violated
today._ The IF2 verdict is computed four times with three rules:
`check_scout_convergence` exempts initial-state parameters and has a warning
band (`gating.rs:182-260`); `compute_if2_gate_verdict` exempts nothing
(`method_result.rs:1082-1123`); `fit summary`'s text block exempts nothing
(`fit_summary.rs:1527-1532`) and its structured block returns no overall verdict
when the decibans leg is missing (`fit_summary.rs:3028-3056`). A NaN Â is
skipped by `fold(0.0, f64::max)` and so passes silently (`gating.rs:202`,
`method_result.rs:1087-1091`). The NLopt verdict is the stage's two-leg rule
(`nlopt_stage.rs:785`, `:803`) at run time and a bare `max_rel_range < 0.05` in
`fit table` (`table_row.rs:293-294`). R̂ is judged against an unrecorded constant
(`RHAT_CONVERGED_THRESHOLD = 1.05`, `method_result.rs:585`), so changing it
re-judges every stored fit without saying so.

**I8. Problem identity is the scientific specification; run identity adds the
execution plan and the implementation; assessment policy is in neither.** The
target split is

    Problem ID = H(canonical scientific specification: compiled IR, data, resolved problem)
    Run ID     = H(Problem ID, execution plan, implementation identity)

_Violated today, twice._ First, policy is in run identity: the IF2 and NLopt
gate thresholds are fields of `[method]`, whose identity payload subtracts
nothing for those methods (`config_v2.rs:1396-1400`), and `--decibans-thresh` is
written into the method before its identity is taken (`config_v2.rs:1597`,
`:1609`, documented at `:1652-1655`), so changing a threshold re-keys and
re-runs the fit; the dt-check tolerance is stored in `fit_state.toml` and keyed
the same way (`config_v2.rs:1270-1272`, gh#730). Second, the implementation is
in problem identity: the fit level folds the engine version, which includes the
git hash (`cas.rs:508-520`). Removing the engine version from problem identity
is an explicit future migration, not a tolerated exception; until it lands,
every build re-keys every problem.

## Warm-start sources: the selection is history, a new assessment is not

A run that starts from a stored fit (`from_mle`, `from_posterior`) depends on
two different things, kept apart.

The **selection** is what the run consumed: the source's `run_id` and the draws
or rows it read. That is execution, and it is already identity through the run's
`deps` (the source `run_id` and the consumed file's digest, `cas.rs:658-670`).
Changing a policy can never change it under the same identity.

The **admission** is whether the source was judged good enough to start from. It
is an assessment of the source, under the consuming run's policy, computed when
the run is launched. A policy change can only change whether a new run is
allowed to start. The policy that admitted a run is persisted in an assessment
record for that run (referencing its `run_id`, the source's `run_id` and the
policy digest), never in `run.json`.

## What counts as which

An **execution control** changes what the engine computes or when it stops:
iteration and sweep counts, particle counts, optimizer stopping tolerances and
evaluation budgets, adaptation targets, cooling, the clean-evaluation particle
and replicate counts, the dt-check ladder depth. It is part of the execution
plan and of run identity.

**Runtime adaptation state** is what evolves inside a run under rules the plan
fixes: a NUTS step size and mass matrix, a PMMH proposal scale, IF2's current
perturbation scale. It is neither configuration nor identity; the rule that
drives it and its targets are execution controls.

An **assessment policy** value is read only after the run, to judge stored
measurements: Â and R̂ thresholds, the decibans threshold and its SE-floor
multiplier, ESS minimums, the optimizer agreement range, the dt-check tolerance.
It is in no identity and in no run record. A verdict computed under it is shown
with it, and when persisted goes in an assessment record. That placement is what
lets this rule and the run-identity rule (`.claude/rules/run-identity.md`: a
field that changes stored bytes is identity) both hold.

A value used both ways is split: NLopt's `tolerance` is an execution control,
and the agreement floor the assessment derives from it is policy applied to the
recorded tolerance.
