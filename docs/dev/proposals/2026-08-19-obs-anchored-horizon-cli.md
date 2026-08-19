# Obs-anchored horizons on the simulate CLI: `--to "last_obs + 8 weeks"`

Status: proposed (revised after 3-agent adversarial review) Issue: gh#626 Scope:
Rust CLI only — no DSL, no IR change, no `ir/VERSION` bump.

## Problem

A forecasting run wants its horizon stated relative to the end of observed data
— "eight weeks past the last observation" — but a horizon can only be set
absolutely, in the model file (`simulate { to = ... }`) or a scenario's
`simulate { to }`. Downstream modeling (ebola-bdbv, friction F15) hand-computes
`data_to + 28 d` and `data_to + 56 d` into literal `date("...")` horizons,
pinned by repo-side regex tests that re-fail on every data release.

The relative vocabulary exists in two disconnected corners:
`condition_from = "first_obs - 1 week"` (fit toml + `--condition-from`,
per-stream anchor) and `value_at(expr, last_obs)` (quantities, bare global
anchor). Horizons have neither.

This proposal removes the **horizon** half of the hand-edit cycle. The
forcing-fork half (`breakpoints = [date(...)]` pinned to `data_to`) needs DSL
anchors (symbolic IR, `ir/VERSION` bump) and stays a follow-up — the ebola
release loop still edits that one literal per release until then. The model also
keeps a literal `simulate { to }` as its un-overridden default; `--to` demotes
it from horizon-of-record to default, and the downstream repo should say so in a
comment where today a regex test pins it.

## Design

One new flag on `camdl simulate`:

```
camdl simulate model.camdl --draws posterior --fit runs/fit_national \
    --to "last_obs + 8 weeks" --obs bands.tsv
```

`--to SPEC` overrides the run's horizon (`simulation.t_end`). SPEC grammar:

```
SPEC   := NUMBER                      # model time, e.g. "120"
        | date("YYYY-MM-DD")          # calendar, needs a model origin
        | YYYY-MM-DD                  # bare-date sugar, same
        | ANCHOR                      # bare anchor
        | ANCHOR (+|-) N UNIT         # relative form, e.g. "last_obs + 8 weeks"
ANCHOR := last_obs | first_obs
UNIT   := day(s)|d | week(s)|w | month(s)|mo | year(s)|yr|y
```

Deliberately NOT accepted (review round 1): the commuted `8 weeks + last_obs`
order and the DSL tick spelling (`8 'weeks`) — the tick is a shell-quoting
hazard, and tolerating it in a shared parser would silently widen
`condition_from`'s acceptance set. Both rejections carry a hint naming the
canonical spelling. `months`/`years` are fixed spans (`days_per_unit`: 30.4535 d
/ 365.2425 d), not calendar arithmetic — same rule as `condition_from`, stated
in `--help`.

The shared parser produces a data-free `TimeSpec` (absolute value, or anchor +
signed offset in model units); `parse_condition_spec` becomes a wrapper that
restricts it (anchor `first_obs` only, `-` only) and resolves per stream. Its
ACCEPTANCE set is unchanged; two rejection messages improve (today
`"last_obs - 1 week"` falls through to a confusing calendar-date error, and
those paths gain e2e coverage).

### Anchor semantics

`last_obs` = max observation time over the run's bound streams; `first_obs` =
min. Global — matching `value_at(…, last_obs)`'s resolution in `fit predict`
(predict.rs:779), NOT `condition_from`'s per-stream anchor: scope follows the
construct, and a horizon is one number per run. Two fine points, both matching
predict: hole (`NA`) rows count as observation times (a trailing-NA-padded
stream moves `last_obs`); and the anchors fold over the RAW loaded streams —
`simulate` never applies conditioning windows, so a `condition_from` leading
hole cannot shift `first_obs`.

### Where the data comes from

Plain `simulate` binds no observation data, so an anchored SPEC requires `--fit`
(fit toml, or fit run directory):

- toml: `FitConfigV2::load` → `[data.observations]`, paths relative to the toml
  (`resolve_relative_to_toml`).
- run directory: the existing `fit::handle::load_config_for_segment` recovery —
  the sidecar's original `fit_toml_path` when it still exists unchanged, else
  the archived `fit.toml.original` (relative data paths then resolve against the
  segment; correct only if data is co-located — the same limit
  `fit predict <run-dir>` has today). A CLI-only fit archives no config: a named
  error ("this fit run carries no data spec; pass the fit toml").
- streams load through the same shared seam as pfilter/profile
  (`data_bindings_to_effective` + `resolve_and_load_obs_streams`) against the
  simulate model's observation blocks — dated time columns, long-form families,
  and holes resolve as at fit time; mismatched bindings get the existing errors.
  `apply_conditioning_windows` is NOT called.

An anchored SPEC without `--fit` is a hard error mirroring the
`value_at`-under-simulate refusal. Absolute forms need no data. `--fit`'s clap
`requires = "draws"` is relaxed to a manual check: `--fit` must come with
`--draws` and/or an anchored `--to`.

Adjacent non-goal, named so it isn't mistaken for an oversight: quantities
anchored to `last_obs` still refuse under `simulate`, even on an invocation that
just resolved `last_obs` for `--to` (ebola F23). Threading the anchor + obs
series into the quantities evaluator is a separate follow-up.

### Where the override lands (the seam)

`simulation.t_end` is the sole horizon authority: `apply_scenario_horizon`
(util.rs:2666) mutates only it, inside `resolve_run_model` (util.rs:2804,
callsite :2829); output grid, obs emission, boundary times, and integrator
bounds derive from the resolved cell's `t_end`. Since gh#561 the Regular
obs-emit walk deliberately ignores the compiler-baked schedule end
(main.rs:2360-2372), so `every N` emission extends with the horizon — the
`--obs` forecast-band use case works without touching the IR.

Mechanics:

1. `run_simulate` resolves SPEC → `t_end_override: f64` once, up front (loading
   fit data iff anchored), and validates
   `t_end_override >
   simulation.t_start` itself, with an error naming the
   spec, the resolved value, and `t_start` — NO existing validator checks
   horizon ordering (review: `ir::validate` never reads `t_end`; the failure
   mode today is a silent header-only TSV).
2. New `t_end_override: Option<f64>` on `SimulateJob` and `SimRun` (incl.
   `SimRun::default`), copied by `engine::build_cell_sim_run` — one construction
   seam covers `--replicates`, `--draws prior/posterior`, sweeps, and `[design]`
   batch cells. `batch run`'s own TOML surface gets no `to` key (deliberate; its
   `ResolvedEntry` is built separately and keeps `t_end: None` semantics).
   `fit/synthetic.rs`'s SimRun keeps no override (fit refuses horizons).
3. Applied in `resolve_run_model` immediately after `apply_scenario_horizon`,
   before validate/compile.
4. The gh#561 baked-end guard (`check_baked_recurring_ends`, util.rs:2691,
   callsite :2928) already sees the post-override end and the pre-scenario
   `old_end` — which equals the pre-override end because of the conflict rule
   below. Its refusal message gains an origin label: today it prints "scenario
   '?'" and recommends retyping a literal model horizon — wrong and useless for
   `--to`.

### Guards (all review round 1; the silent-drop class)

- **Reactive policies (blocker):** a reactive intervention's monitoring walk is
  bounded by its compiler-baked `end` (reactive.rs:111-117) and
  `check_baked_recurring_ends` matches only `Scheduled(Recurring)` — under an
  extending `--to`, dynamics run to the new end while the policy silently stops
  reacting at the old one. Fix here: extend the guard to refuse an extending
  `--to` when any reactive policy's monitored emit schedule has
  `end == old_end`. The deeper fix (the reactive walk takes the run `t_end`, the
  same gh#143 move the obs walk already made) is filed as follow-up.
- **`at [...]` obs-emit schedules:** the list never grows, so an extending
  `--to` with `--obs*` would emit zero rows past the listed times, exit 0 —
  refuse when any emitted stream's `AtTimes` max < the new end (the feature's
  whole point is new rows).
- **`at [...]` output schedules:** trajectory-only runs keep the existing warn
  posture (gh#125 fires only when ALL times are beyond the horizon — the partial
  case is a pre-existing gap the scenario path shares); the gh#589 and gh#125
  messages gain wording that names `--to` as a possible horizon source (today
  they blame a scenario).
- **Multi-scenario `--obs` preflight:** main.rs:898-946 computes the shared obs
  axis and schedule-compatibility at `effective_horizon` of the RAW model; under
  `--to` it must validate at the overridden horizon
  (`obs_end = t_end_override.unwrap_or(...)`), or two `at`-list schedules that
  agree at the old horizon can diverge over the extension.
- **Recurring campaigns:** any recurring intervention/event without `until`
  bakes `end = t_end` (expander default), so `--to` extension on such models is
  refused by the existing guard — correct but blunt; the lift that removes it is
  gh#605. Stated as a known limit. (The ebola models carry no interventions
  blocks; the primary use case is not refused.)

### Conflicts — never silently discard a declared horizon (gh#561)

- `--to` + a scenario whose composed `effective_horizon` differs from BOTH the
  model's `t_end` and the resolved `--to`: hard error naming both. A scenario
  horizon EQUAL to the resolved `--to` is allowed — the same no-op precedent as
  `refuse_scenario_horizon`'s "a preset restating the run horizon still works".
- Migration note for downstream: a horizon-only scenario like `forecast_8w`
  becomes a label-only preset plus `--to "last_obs + 8 weeks"` on its own
  invocation. What the model loses is the structural tripwire that a
  horizon-differing scenario cannot be contrasted against the ladder — that
  protection moves from model-declared to workflow discipline, and the
  downstream repo should note it where the scenario comment lives today.

### Run identity (count-in-the-key)

On the simulate path the horizon is keyed at the config level —
`ResolvedEntry.t_end` (main.rs:1856 baseline arm, :1876 preset arm; folded
batch.rs:954) → `runid::inputs::SimConfig.t_end` — NOT via the mutated model
(`TrajectoryCtx.model` is the raw IR). `--to` sets
`ResolvedEntry.t_end =
Some(resolved)` at both arms. Keyed value = the resolved
numeric t_end: a `--to` restating the model horizon hashes identically to no
`--to` (same output, correct collision); moved data under the same spec re-keys.
A no-collision test pins this. Plain `--stdout` runs skip the CAS and need no
key; ensembles compose member run_ids and inherit it.

## Non-goals (explicit)

- `fit predict --to` — free-forward bands exist only at observed times
  (predict.rs:1694); forecast rows in predict are the separate P5 arc.
- DSL anchors (`to = last_obs + 4 'weeks`, anchored forcing `breakpoints`) —
  symbolic IR anchor, `ir/VERSION` bump, OCaml expander; F15's other half.
- `--to` on data-windowed commands (pfilter/profile/survey/fit run) — their
  window comes from the data (gh#561 refusals stay).
- `value_at` offset arithmetic, and `last_obs` quantities under simulate (F23) —
  follow-ups.
- Re-deriving reactive monitoring from the run horizon — follow-up (guard
  refuses meanwhile).

## Testing

- Parser units: every SPEC form; rejections for commuted order, tick units,
  unknown anchor/unit, trailing tokens — each with its hint.
- `condition_from` neutrality: acceptance set unchanged (existing conditioning
  e2e), plus new e2e pinning the two improved rejections.
- e2e (`simulate_to_e2e.rs`): absolute `--to` extends trajectory rows past the
  baked horizon (Regular output); anchored `--to` without `--fit` is the named
  error; tiny 1-iteration IF2 fit, then
  `--draws posterior --fit <run> --to "last_obs + 2 weeks"` runs to exactly
  max-obs-time + 14 model-days with `--obs` rows past `last_obs`; `--to` ≤
  `t_start` is the named error; `--to` + horizon-carrying scenario errors,
  equal-horizon scenario allowed; `at [...]` emit schedule + extending `--to` +
  `--obs` refused.
- Identity: distinct `--to` values → distinct run_ids; `--to` == model horizon →
  identical run_id to no `--to`.

## Alternatives considered

- Fit-toml key as the primary surface: rejected (CLI-first; simulate reads no
  fit toml for config); a toml mirror can follow.
- Rewriting the IR file (`--output-every` rematerialize pattern): re-keys via
  the model level and still needs `ResolvedEntry.t_end` consistency; more moving
  parts than the config-level seam.
- General expression anchors: rejected — leaks data-dependence into compiled
  models and run identity; `value_at` settled that anchors stay positional.
