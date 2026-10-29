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

**`ResolvedProblem`** is the statistical target: the compiled model, the
parameter partition (which parameters are fixed, estimated, design or sampled,
per `proposals/2026-10-26-parameter-roles.md`), the fixed values, the
interventions, the observation streams, the priors, the effective constraints
(support and search box), `ic_free` and the training window. It is built once
per fit segment and is immutable with respect to the method, the run settings,
the chain starts and the seed: two methods and two seeds of one problem share
one value.

**`ExecutionPlan`** is how one algorithm attempts that problem: a typed
per-method plan (IF2's perturbations and cooling, a sampler's proposal and
adaptation, an optimizer's tolerances), the drawn starts, the chain count and
the seed. It is fully validated before launch, and it is the only configuration
an engine receives. It is not a catch-all: a setting belongs to exactly one
method's arm, and a setting one method needs is never carried, as a placeholder,
by another. The split is defined in
`proposals/2026-10-29-method-scoped-settings.md` §5.4 (branch
`docs/method-scoped-settings`).

**`RunRecord`** (`run.json`) is what was executed: the factored identity levels
and the `run_id` derived from them, the engine and IR versions, and the parents.
Every stored artifact resolves to exactly one `run_id`. A label is a name for a
run and never stands in for its identity. A parent is recorded by its `run_id`.

**Result records** are typed per method: the measurements, estimates and draws a
method produced, and nothing it did not. An optimizer's record holds its point
estimate and the measurements of whether its chains agree; a sampler's holds its
draws and the measurements of whether its chains mix. No method writes a field
another method defines, and no field leaves its meaning to a sentinel.

**`Assessment`** is an interpretation of measurements under a stated
`DiagnosticPolicy` (the thresholds: acceptable Â, decibans spread, R̂, ESS,
optimizer agreement). Changing the policy changes the assessment; it never
changes a stored measurement or an identity. Every reported verdict names the
policy it was computed under, so a historical result never silently flips from
accepted to rejected. The design is
`proposals/2026-10-29-result-records-and-assessment.md`.

## The invariants

**I1. The problem does not depend on how it is attempted.** _Violated today._
`[estimate].rw_sd` and `perturb_only_at_t0` (IF2 and MH settings) and
`[estimate].start` live in `EstimateSpecV2`
(`cli/src/fit/config_v2.rs:574-600`), which the fit level hashes whole
(`fit_config_blob_hash`, `cas.rs:433-435`). The value resolver draws starts with
the seed (`resolve_fit_parameters(fit,
seed)`, `runner.rs:171-174`, draw at
`:245-265`). The step keys are moved by method-scoped-settings; `start` is moved
by no proposal yet.

**I2. An engine receives one validated plan, with nothing for other methods.**
_Violated today._ `FitRunConfig::build` takes IF2's `n_iterations`, `cooling`
and `cooling_target_iters` for every method (`runner.rs:325-335`), and PMMH,
PGAS, NLopt and pfilter pass placeholders (`pmmh.rs:133-140`, `pgas.rs:304-310`,
`nlopt_stage.rs:80-92`, `mod.rs:1752-1758`).
`FitRunConfig.if2_config: IF2Config` (`runner.rs:106`) carries the particle
filter settings every method reads (`if2.rs:110-147`), and `FitRunConfig.gate`
(`runner.rs:124`) an IF2 verdict threshold. `EstimatedParam` mixes the
parameter's identity, start and IF2 tuning (`sim/src/inference/types.rs:48-74`).

**I3. Every accepted user option has a defined, observable effect on the
computation, or is refused.** _Violated today._ `--cooling-target-iters` with an
`nl-sbplx` method is accepted and read by nothing: the NLopt arm of the override
application has no cooling field (`config_v2.rs:1605-1611`). Branch
`fix/settings-silent-noops` adds the refusal and its test.

**I4. The sim crate takes no CLI types.** _Holds today._ `sim/Cargo.toml` does
not depend on `cli`. This is a forward rule: a plan type that an engine reads is
defined in `sim` or built from `sim` types at the boundary, never passed in as a
fit-config or `clap` type.

**I5. Every artifact resolves to exactly one run; labels never substitute for
the identity; parents are recorded by run ID.** _Violated today for labels._
`@label` resolves to a problem folder, which holds every run of one problem
(`cli/src/fit/handle.rs:133-141`, the label read from the segment's
`fit.meta.json`); branch `fix/gh896-handles` moves the label to the leaf.
Parents hold: a warm-start source that is a stored run is recorded by its
`run_id` and the consumed file's digest (`cas.rs:658-670`).

**I6. Each method writes its own result type.** _Violated today._ One flat
`FitState` (`cli/src/fit/state.rs:19-182`) is written by IF2, NLopt, PMMH, MH
and PGAS, each leaving the others' fields empty, and NLopt stores its
per-parameter relative range under the field named for IF2's Â
(`tail_chain_agreement`, `nlopt_stage.rs:297`).

**I7. One assessment rule per measurement type, applied under a recorded
policy.** _Violated today._ The IF2 verdict is computed four times with three
rules: `check_scout_convergence` exempts initial-value parameters and has a
warning band (`gating.rs:182-260`); `compute_if2_gate_verdict` exempts nothing
(`method_result.rs:1082-1123`); `fit summary`'s text block exempts nothing
(`fit_summary.rs:1527-1532`) and its structured block returns no overall verdict
when the decibans leg is missing (`fit_summary.rs:3028-3056`). The NLopt verdict
is the stage's two-leg rule (`nlopt_stage.rs:785`, `:803`) at run time and a
bare `max_rel_range < 0.05` in `fit table` (`table_row.rs:293-294`). R̂ is judged
against an unrecorded constant (`RHAT_CONVERGED_THRESHOLD = 1.05`,
`method_result.rs:585`), so changing it re-judges every stored fit without
saying so.

**I8. Problem identity is the `ResolvedProblem`'s inputs; run identity is those
plus the execution controls, the seed and the code; assessment policy is in
neither.** _Violated today for policy._ The IF2 and NLopt gate thresholds are
fields of `[method]`, whose identity payload subtracts nothing for those methods
(`config_v2.rs:1396-1400`), and `--decibans-thresh` is written into the method
before its identity is taken (`config_v2.rs:1597`, `:1609`, documented at
`:1652-1655`), so changing a threshold re-keys and re-runs the fit. The dt-check
threshold is stored in `fit_state.toml` and keyed the same way
(`config_v2.rs:1270-1272`, gh#730). Today the fit level, which is the problem
identity, also folds the engine version (`cas.rs:508-520`); code is therefore in
both levels, which over-keys the problem without breaking the rule for runs.

## What counts as which

An **execution control** changes what the engine computes or when it stops:
iteration and sweep counts, particle counts, optimizer stopping tolerances and
evaluation budgets, adaptation targets, cooling, the clean-evaluation particle
and replicate counts, the dt-check ladder depth. It is in run identity.

An **assessment policy** value is read only after the run, to judge stored
measurements: Â and R̂ thresholds, the decibans threshold and its SE-floor
multiplier, a minimum ESS, the optimizer agreement range, the dt-check
tolerance. It is in no identity, and it is recorded beside each verdict where
recorded-only facts live (the `run.json` provenance, which already varies
between executions of one `run_id`), never in a leaf's artifact set. That
placement is what lets this rule and the run-identity rule
(`.claude/rules/run-identity.md`: a field that changes stored bytes is identity)
both hold.

A value used both ways is split: NLopt's `tolerance` is an execution control,
and the agreement floor the assessment derives from it is policy applied to the
recorded tolerance.
