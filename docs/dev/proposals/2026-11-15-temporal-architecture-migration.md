# Proposal: temporal architecture migration for forecasting and scoring

**Status:** Draft for review **Date:** 2026-11-15 **Spec:**
`docs/dev/specs/temporal-architecture.md` (the spec; section numbers prefixed
"spec") **Motivation:** camdl is about to gain fixed-origin forecasts, forecast
scoring (gh#808) and published score artifacts (`2026-11-14-score-artifact.md`).
Each of these consumes observation times, training cutoffs and forecast targets
that are today carried as bare model-time numbers and re-interpreted at each
consumer under different conventions. This proposal audits that state at `main`
`0ed74876d`, ranks the defects, and stages the smallest change that makes
forecast scores trustworthy.

Code references are to `0ed74876d`. A claim marked _verified (read)_ rests on
reading the cited lines; _verified (run)_ on running the stated command with the
binaries built from that commit (`CAMDL_SKIP_VERSION_CHECK=1`,
`CAMDLC=ocaml/_build/default/bin/camdlc.exe`); _inferred_ on neither. No
production code was changed for this proposal.

## 1. Summary

The compiler's temporal types are mostly sound, and the runtime already has the
right primitives (`Period`, `StreamTimes`, `Covers::period_of`). The defects sit
in three places:

- at the seam where a data row's **label** is compared with a cutoff instead of
  its **support** (the instant or period it measures);
- in **forecast targets**, which are chosen by three different rules in four
  places, one of them a guessed cadence;
- in a handful of **compile-time lowering holes** that silently produce the
  wrong instant.

The proposed fix adds four small runtime types (`ObservedRow.support`,
`TrainingPartition`, `ConditioningRecord`, `ForecastPlan`), factors one existing
one (`CalendarAxis`), and needs no new DSL syntax and no IR change until the
optional Stage 7 calendar periods (`covers = month(time)` and week forms).

## 2. Audit of current representations

### 2.1 Compiler and IR

The time typing pass (`ocaml/lib/compiler/time_typing.ml`) classifies
expressions as `TExact | TCalendar | TInstant | TOther`. It enforces:

- E321: a calendar duration translating an instant (anchored mode);
- E320: `time_unit` months or years with an origin;
- E322: a calendar cadence in a recurring schedule or `emit_schedule every`;
- E327: anchor-only primitives in an unanchored model;
- W324: a bare number in a time position under an origin.

_Verified (read and run)._

The IR erases instant versus duration and calendar versus affine to `f64`, and
keeps the calendar axis (`time_unit`, `origin`, `origin_rata_die`,
`ir.ml:1139-1145`). For each stream it carries `covers` as
`From/Until {offset,
span}` or `WindowColumns`, plus the column roles, the emit
schedule (`regular{start, step, end}` or `at_times`), and forcing knot times. It
carries no forcing extrapolation field. _Verified (read)._ The erasure is
correct (spec §7). What is wrong is a set of lowering holes:

| ID | Defect                                                                                                                                                                                                                                                                                                      | Evidence                                                                                                                      | Status                            |
| -- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| K1 | `date_range(d, count = n, calendar_months = k)` steps from the previous date, not from `d`, so after a month-end clamp every later date keeps the clamped day: `2020-01-31` gives Feb 29, Mar 29, Apr 29, May 29, where `add_calendar_months` gives Mar 31, Apr 30                                          | `expander.ml:3343-3364` (`cur := step_fn !cur k_step`)                                                                        | verified (read and compiled)      |
| K2 | `covers = closing_at(time, 1 'months)` compiles to an affine span of 30.436875 days, while the same model's `emit_schedule = every 1 'months` is E322 and `origin + 1 'months` is E321                                                                                                                      | `expander.ml:9481-9518`, `span_of` has no time-class check                                                                    | verified (compiled)               |
| K3 | Instant algebra is unchecked except `Instant ± Calendar`. These compile silently in `at [...]`: `date(a) + date(b)`, `date(a) * 2`, `7 'days - date(a)`, `date(a) + 5`, and `date(b) - date(a)` used as an instant                                                                                          | `time_typing.ml` `classify` (~~:170) maps `TInstant + _` to `TInstant`; `walk_rule1` (~~:201) fires only on calendar operands | verified (compiled)               |
| K4 | `date()` lowers to a dimensionless constant: `let d = date(...)` then `at [d]` is E315; `if t > date(...)` in a rate is E302. `origin` works only because it lowers to `Const 0.0`                                                                                                                          | `expander.ml:4388`                                                                                                            | verified (compiled); fails closed |
| K5 | A generated emit grid takes its phase from `simulate.from`, not from `origin` as `docs/dates.md:525-531` states                                                                                                                                                                                             | `expander.ml:9706-9712`; `simulate --obs --dates` with `from = 2019-12-01` emits Sundays under a Wednesday origin             | verified (run)                    |
| K6 | Temporal expressions are accepted inconsistently across positions: `origin + 30 'days` is E401 in `emit_schedule at` and `simulate to` but allowed in `at` and `breakpoints`; `date_range` is E270 in `emit_schedule at` and `breakpoints`; piecewise breakpoints may be non-monotone with no compile error | `resolve_float_expr` (`expander.ml:5622`) does not fold `unit_literal`; gh#844 (closed as a doc fix), gh#847                  | verified (compiled)               |
| K7 | `simulate --to` help says a month is 30.4375 days; the code uses 30.436875                                                                                                                                                                                                                                  | `cli/src/args/mod.rs:310` vs `ir/src/caltime.rs:87`                                                                           | verified (read)                   |

A calendar-month emit cadence cannot be expressed in an anchored model at all.
K6 closes both routes to one (E322 on `every 1 'months`, E270 on `date_range` in
`emit_schedule at`).

### 2.2 Observation data and training partitions

Runtime representation (_verified, read_):

- `Period` (`sim/src/inference/multi_stream_obs.rs:1434`) has a private-field
  smart constructor that enforces finite, ordered endpoints.
- `StreamTimes::{Instants, Intervals}` (`:1505`) holds a stream's supports;
  `closes()` (`:1520`) is the single source of each row's scoring time.
- `BoundObs::bind` (`:945`) refuses non-increasing closes and overlapping
  periods.

But the loader converts each row to `Observation { time, value }`
(`particle_filter.rs:22`), where `time` is the label, and every holdout cut
compares that label.

Every site that resolves a cutoff or `last_obs`, and the time it compares
(_verified, read_; the obs-data audit tabulated these):

| Site                                                                                                                             | Compares                                    |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `holdout_after` cut, `within_training` (`fit/runner.rs:2099`)                                                                    | label                                       |
| `holdout_after = "last_obs - N"` (`resolve_holdout_cutoff`, `runner.rs:2063`)                                                    | label, over full files                      |
| `[data.holdout]` `train_end` and "holdout after training" check (`runner.rs:1998`, `:2017`)                                      | label                                       |
| `survey` truncation, a separate copy of the loop (`survey.rs:1001-1024`)                                                         | label                                       |
| `profile` (`profile.rs:1003`)                                                                                                    | label                                       |
| `TrainingWindow.train_end` written to `fit.meta.json` (`run_meta.rs:626`)                                                        | label                                       |
| `compare` chain-binomial boundary (`training_scoring_ends`, `heldout_scoring_boundary`, `compare.rs:2069`, `:2145`; gh#1062 fix) | close                                       |
| `compare` ODE boundary (`derive_ode_trace`, `compare.rs:2799-2802`)                                                              | label                                       |
| `pfilter --score-from` (`first_scored_index`, `runner.rs:2121`) and its `last_obs` anchor (`pfilter.rs:975-990`)                 | close                                       |
| DSL anchors such as `simulate { to = last_obs + … }` (`obs_anchors_from_bindings`, `main.rs:4139-4176`)                          | label, over full files, holdout not applied |
| `fit predict` `last_obs` and `value_at` (`fit/predict.rs:1791-1803`, `:1925`, `:3140`)                                           | label, over full files                      |

Counterexamples (scratch fixtures are reproduced in §8 so they survive):

- **C1. A held-out row whose period overlaps trained time.** _Verified (run)._
  Setup: a weekly stream (`closing_at(time, 7 'days)`, labels 7..70) and a daily
  stream (`day(time)`, labels 0..69), `holdout_after = 24`. `camdl compare`
  reports `Scored steps: 45 (t0=25) … held-out tail (train_end
  = 24)` and
  scores weekly row 28, which covers `[21, 28)`. Daily rows for days 21–24
  trained the fit. No trained row is scored, so this is not a leak under today's
  definition. But the "held-out" score for that week partly measures days the
  parameters saw.
- **C2. Training runs past the cutoff it reports.** _Verified (run)._ A single
  `starting_on(time, 7 'days)` stream, labels 0..63, `holdout_after = 24`.
  - The fit prints `training window [0, 24]`.
  - Row 21 trains, and it covers `[21, 28)`. The parameters saw infections
    through day 28.
  - `fit.meta.json` records `train_end: 24`.
  - `compare` scores closes 35..70 only, so nothing leaks. The record of what
    trained is wrong, and every consumer that trusts it inherits the error.
- **C3. The ODE comparison path still cuts by label (the open half of
  gh#1062).** _Verified (run)._ A `day(time)` stream, labels 1..16,
  `holdout_after = 10`, `mh` on the `ode` backend. `camdl compare` exits 1:
  `stream 'cases' is scored at t = 11, but its last training row … is scored at
  t = 11`.
  The refusal comes from `check_no_training_row_scored` (`compare.rs:2158`), so
  this fails closed. `cli/tests/compare_ode_prequential.rs` covers only
  `closing_at`.

**Row identity.** No row index or key travels from a fit to a score. The fit
records whole-file SHA-256 digests (`data_hashes`) and a label-axis `train_end`,
and `compare` re-derives membership from labels (`compare.rs:2096-2125`). With
`--allow-changed-data`, that membership is re-derived on bytes the fit never
read. _Verified (read)._ The score-artifact proposal's row digest (§"The row
digest") and training-row prefix check close most of this. This proposal makes
them position-based and support-based (spec §5).

### 2.3 Inference, alignment and forecasting

**Alignment.**

- `ObsAlignment::{Exact, Snap}` is resolved per algorithm
  (`fit/methods.rs:499-551`): if2, pfilter and pmmh get `Exact`; pgas gets
  `Snap`. Nothing in `sim` reads the resolved value; it is a validation gate and
  a run-identity input (`runid/src/inputs.rs:468`). _Verified (read, grep)._
- Before the policy is resolved, `load_observations` refuses any time with
  `time % dt ≠ 0` (`runner.rs:1678-1688`). It does so on every path (fit,
  pfilter, compare, predict), measured from 0, not from `t_start`.
- Consequences, _verified (run)_ unless marked:
  - **A1.** Off-grid observations are refused even under `exact`.
    - Model (`dt = 1`): `covers = closing_at(time, 2.5 'days)`, data at 2.5, 5,
      …, 15 (fixture F1, §8).
    - if2, `exact`, pfilter and pgas all fail with
      `observation at t=2.5 is
      not a multiple of dt=1`.
    - This contradicts `docs/dates.md:632` and `docs/camdl-run-spec.md:3126`.
    - It fails closed.
  - **A2.** PGAS moves a support silently.
    - With `from = 0.5`, `dt = 1` and integer observations at 3, 5, 7, 9, the
      guard passes (they are multiples of `dt` from 0).
    - PGAS substeps are 0.5, 1.5, …. `interval_steps` rounds
      (`sim/src/time.rs:125`), so the observation labelled 3 lands on the
      substep ending at 3.5.
    - The scored period becomes `[1.5, 3.5)` instead of `[1, 3)`.
    - `filter_ess.tsv` still labels the row 3. No notice is printed, and no
      artifact records an alignment.
    - The period shift is _inferred_ from `time.rs:125`; per-row windows were
      not dumped.
  - **A3.** Two PGAS errors advise `--obs-alignment exact`, a flag that does not
    exist, for a policy PGAS refuses (`pgas.rs:1263`, `:1276`).

**Forecast targets.** Four places choose targets, by three rules (_verified,
read_):

| Command                                                            | Target rule                                                                                                                                     |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit predict` (`forecast_times`, `predict.rs:3135`)                | last label + k × most-common observed gap (`modal_value`, 1% tolerance, ties to the smaller), walked over the output grid to `simulation.t_end` |
| `simulate --obs` and `simulate --init-state` (`main.rs:4203-4229`) | `emit_schedule`, filtered to after the restart                                                                                                  |
| `simulate --design-from`                                           | the fit's bound rows, no tail                                                                                                                   |

- **T1. Monthly drift.** _Verified (run)._ For window-column rows closing at 31,
  59, 90, 120, 151, 181, the gaps are 28, 31, 30, 31, 30. 30 and 31 tie, and the
  smaller wins. Forecast rows fall at 211, 241, …, 361, while month-ends are
  212, 243, 273, 304, 334, 365. This is gh#861.
- **T2. Forecast periods not the observed shape.** _Verified (read)._
  `leaf_row_coverages` (`predict.rs:3049-3082`) gives a window-column forecast
  row `[previous label, label)`. A series with gaps between observed windows
  therefore gets forecast rows wider than its observed rows. For uniform forms
  the declared span is used even when the guessed cadence differs, so forecast
  rows can overlap or leave gaps.
- **T3. A mid-period restart drops a row silently.** _Verified (run)._ Filter
  through a weekly `closing_at` stream to t = 25, then
  `simulate --init-state …
  --to 49 --obs`. The output rows are 35, 42, 49. The
  28 row, covering `[21, 28)`, is dropped without notice. The control, with
  origin 21, emits 28.
- **T4. No length for the forecast.** _Verified (read)._ `fit predict` has no
  duration, count or date for how far ahead to forecast: the extent is the
  model's `simulation.t_end` (`predict.rs:1915-1931`). Its `--horizon` flag
  selects a mode (`free_forward | one_step`, `args/types.rs:130`).

**Conditioning in `fit predict`.**

- **P1. The one-step band conditions on held-out rows, unmarked.** _Verified
  (run)._ A PMMH fit with `holdout_after = 24` (fixture F3).
  `fit predict
  --horizon one_step` runs the filter over every row of the file,
  held-out rows included (`load_leaf_obs`, `predict.rs:2986`; `one_step_bands`,
  `:3759-3787`, both via `training_by_source()`). `predictive/weekly.tsv` has
  one-step rows at all ten closes. Neither `predictive.json` nor `report.json`
  mentions the training extent. `predict.rs` contains no reference to holdout.
- **P2. The two holdout forms give different `fit predict` output.** _Inferred._
  Under `[data.holdout]` the same code reads only the training files, so the two
  forms of one scientific split produce different `fit predict` output.

**Backends.**

- `fit predict` supports chain-binomial with both modes, and ODE with
  free-forward only. It refuses Gillespie and optimizer fits.
- `--init-state` supports chain-binomial only (`predict.rs:367-373`,
  `args/mod.rs:1226`).
- Free-forward accumulation over a target `[a, b)` is exact: both ends are
  recorded snapshot times (`predict.rs:1990-2105`).

_Verified (read)._

**Forcings.**

- **F1. Silent endpoint hold.** _Verified (read)._ Interpolated (linear and
  constant), cubic and piecewise forcings hold their end value outside their
  knots with no notice (`sim/src/propensity.rs:378`, `:396-425`;
  `compiled_model.rs:335-337`). B-splines hold too, but report it
  (`spline-held-past-window`, `cli/src/obs_anchor.rs:497-560`).
- **F2. Future covariates are readable.** _Inferred_ from structure; not run.
  Forcing values are baked into the IR at compile time, with no notion of a
  forecast origin. A forecast or held-out score therefore reads any covariate
  value the table holds past the forecast origin.

### 2.4 Scoring, artifacts and CLI

**Score and compare representations.** _Verified (read)._

- `PrequentialTrace` (`score/src/trace.rs:227`) is at **schema 6** on `main`.
  Schema 6 added `basis` (`PredictiveBasis::{Filtered, OdeFreeForward}`,
  gh#312).
- The score-artifact proposal also calls its own trace change "schema 6", and
  says the ODE score is not built. Both statements are stale. That proposal's
  trace change becomes schema 7.
- `Conditioning::{InSample, HoldOutTail{train_end, theta_source}}`
  (`trace.rs:109`) has no forecast variant.
- `Coverage::{Unrecorded, Instant, Interval{start, stop}}` records the support
  of each scored value. It is already the right object.
- Of the score-artifact and scoring-report proposals, only the `score` crate and
  the moved trace module exist. `ScoreArtifact`, `TimeAxis`, `ScoredLeadTime`,
  the row digest and `camdl score` do not.

**Calendar metadata in artifacts.** _Verified (run)_ unless marked.

- **M1. The simulation leaf cannot be dated.** Its `run.json` has no `origin`,
  `time_unit` or `calendar`, and the leaf archives no IR, so a stored trajectory
  cannot be dated from the leaf alone. The `-o` file has date columns; the
  stored `traj.tsv` does not. This is gh#838 (the metadata half).
- **`fit predict` JSON sidecars do carry the calendar.** Each has
  `calendar: {time_unit, origin, days_per_unit}` (`fit/predict.rs:2765`,
  `:2806`). _Verified (read)._
- **M2. Sub-day dated output does not load back.**
  `--dt 0.5 --output-every
  0.5 --dates` writes `2026-07-01+0.5d`, which
  `--data` refuses. This is gh#839.
- **Whole-day dated output does load back.** `--obs-dir --dates` output read as
  `--data` gives the bit-identical elpd (−34.534024212843676).
- **M3. Dated diagnostics are mixed.** The prequential TSVs and pfilter's
  worst-scored table print numeric `t` on dated data, while error messages quote
  dates (`quote_time`, `caltime_load.rs:127`).

**Naming.** _Verified (read)._

- "horizon" has four meanings:
  - the run end (`simulate --to` help);
  - the predictive mode (`fit predict --horizon`, and the `horizon` column);
  - the scored-step count (`compare --allow-mismatched-horizon`,
    `compare.rs:1189-1202`);
  - the lead `k` (gh#808).
- "origin" means both the calendar origin and the forecast origin (the
  `--init-state` help, gh#808).
- The cut is called `holdout_after`, `train_end`, `score_from` and "cutoff" in
  different places.
- `simulate --to` accepts dates, while `data split --at-time`, `eval`,
  `--output-every` and `--emit-every` take only numbers.

## 3. Ranked issues

Ranked by scientific severity: a silent wrong number first, then a misleading
record, then a refusal. Architectural weight is given beside each. "Gate" marks
the issues that must be resolved before a forecast score can be trusted.

| Rank | ID         | Issue                                                                                                     | Effect                                                                                         | Arch.                                            | Gate                           |
| ---- | ---------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------ |
| 1    | P1         | `fit predict` one-step band filters through held-out rows, unmarked                                       | a band presented as a forecast has seen the data; looks excellent                              | high: conditioning sets not represented          | yes                            |
| 2    | T1, T2     | forecast targets from a guessed cadence                                                                   | wrong targets for monthly and irregular series; forecast rows not the observed quantity        | high: no shared target plan                      | yes                            |
| 3    | A2         | PGAS snaps an off-grid support silently                                                                   | scores a different period than declared                                                        | medium: alignment enforced at the wrong boundary | yes                            |
| 4    | C2         | training cut by label under period-start forms                                                            | parameters see data past the cutoff they report                                                | high: label vs support                           | yes                            |
| 5    | C1         | held-out row overlapping another leaf's trained time                                                      | held-out score is partly in-sample                                                             | high: same                                       | yes                            |
| 6    | K2         | `covers` accepts a calendar width as an affine span                                                       | a monthly period is 30.44 days, silently                                                       | low: one check                                   | yes, for monthly streams       |
| 7    | K1         | `date_range(calendar_months)` drifts after a clamp                                                        | wrong instants for month-end schedules                                                         | low: one loop                                    | no                             |
| 8    | K3         | instant algebra unchecked                                                                                 | a nonsensical expression yields an instant                                                     | medium: typing rule                              | no                             |
| 9    | F1, F2     | silent forcing hold on every run (F1, Stage 0(f)); future covariates readable by a forecast (F2, Stage 5) | a simulation, fit or forecast quietly assumes the last value; a forecast is quietly a scenario | low: one note (F1); medium: plan notes (F2)      | yes, as a record               |
| 10   | T3         | mid-period restart drops a row silently                                                                   | missing target, no notice                                                                      | covered by the plan                              | yes                            |
| 11   | P2         | two holdout forms disagree in `fit predict`                                                               | inconsistency                                                                                  | covered by the partition                         | yes                            |
| 12   | M1, M2, M3 | calendar metadata missing or unreadable in artifacts                                                      | artifacts cannot be dated or reloaded                                                          | medium: one axis type                            | M1 yes for published forecasts |
| 13   | K5         | emit grid phase from `from`, not `origin`                                                                 | doc vs code; affects only generated grids                                                      | low                                              | no                             |
| 14   | C3         | ODE compare cuts by label                                                                                 | refusal (fails closed)                                                                         | low                                              | yes for ODE                    |
| 15   | A1, A3     | off-grid refused before the policy; wrong advice                                                          | refusal                                                                                        | low                                              | no                             |
| 16   | K4, K6, K7 | `date()` dimensionless; position inconsistency; help text                                                 | refusals, doc                                                                                  | low                                              | no                             |

## 4. Overlapping work

| Item                                                              | Relation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2026-11-14-score-artifact.md` (proposal)                         | Supplies the row digest, `TimeAxis`, `ScoredLeadTime`, the store kind and `camdl score`. This proposal amends it in three ways: `ObservedRow.time` becomes `support`; `train_end` becomes the extent and a per-leaf position prefix; the trace schema becomes 7. The two land in one `HASH_VERSION` bump.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `2026-11-14-scoring-report-library.md` (Phase 1 steps 0–2 landed) | Unaffected; its preflight reads `ConditioningRecord` once that exists.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `2026-11-13-published-comparison-principles.md`                   | Vocabulary amendment: "horizon" gets one meaning (decision D3).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `2026-08-29-honest-predictive-evaluation.md` §3.7.2, gh#808       | Forecast mode is built on `ForecastPlan` (Stage 6).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `2026-09-05-observation-time-as-a-sum-type.md` (accepted, landed) | Supplies `Period`, `StreamTimes` and `covers`; extended, not replaced.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `2026-06-09-time-interval-model.md` (draft)                       | Its forecast and forcing sections are superseded by the spec. Two disagreements are resolved in the spec's favour: the June draft makes the forecast horizon the end of integration (`forecast: Option<f64>`, "horizon ≥ D.end"), where the spec makes it a bound on targets (spec §2.4, §6.2), with integration running to the last target's close; and the June draft refuses a run past a forcing's knots (§7.1 point 5), where the spec records a note (I13, D11). Its `RunWindows` / `cond_to` interval abstraction is not adopted; `ConditioningRecord` covers the conditioning end. Its conditioning-window lineage is moot: the `condition_from` key it generalizes was removed (`fit/config_v2.rs:2256` refuses it), and per-stream periods now open where `covers` says. Status line to add on acceptance: "forecast and forcing-domain sections superseded by `docs/dev/specs/temporal-architecture.md`". |
| `2026-06-09-time-and-observation-overview.md` (overview)          | Calls itself the entry point to time and observations. On acceptance its status line points to the spec as the current account of forecasting, partitions and supports, and marks its description of the interval model as superseded.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `2026-06-06-scheduling-effect-topology.md` (scheduling spine)     | Unaffected: the spec hands engines plain `f64` vectors through `Schedule` (spec §7) and merges no output or intervention schedule generators. `ForecastPlan::required_times` is one more producer of output times through the existing seam.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `2026-09-10-camdlc-version-guard.md`                              | Unrelated.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Open issues                                                       | gh#1062 (C3), gh#861 (T1), gh#838 (M1), gh#839 (M2), gh#624 (forecast TSV lacks supports), gh#808 (forecast mode), gh#847 and gh#844 (K6), gh#1054 (draw and seed checks), gh#277 (held-out ergonomics), gh#454 (`output at` below `t_start`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| No issue found                                                    | K1, K2, K3, K4, K5, A1, A2, A3, C1, C2, P1, T2, T3, F1, F2. To be filed on approval.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## 5. Recommended architecture and rejected alternatives

The spec is the recommended architecture. Its parts, each with its justification
against the alternatives:

**`ObservedRow.support` instead of a label** (spec §4.1).

- One reader computes each row's support once, and the label is never again
  readable as a time.
- _Alternative: newtypes `LabelTime` / `CloseTime` over `f64`._ Rejected. A row
  carries both values, and every consumer would unwrap the right one by
  convention. Removing the label from the struct makes the wrong comparison
  impossible to write, without a type the reader has to unwrap.

**`TrainingPartition` as a per-leaf prefix with a digest** (spec §4.2, §5).

- _Alternative S (the simpler one): extend the gh#1062 fix everywhere._ Compare
  closes with `τ` at every site, keep `train_end: f64` (as a close), add the
  score-artifact prefix digest, and add no partition type.
  - This fixes C2, C3 and P2 with fewer lines.
  - It does not fix C1. A weekly row closing after `τ` but opening before it is
    still scored.
  - It leaves six sites each comparing a float with `τ`, which is the
    duplication that produced gh#1062.
  - The partition type costs one struct, two constructors and a JSON block. It
    removes the duplication by construction.
- _Alternative: a general observation-identity system_ (stable row keys across
  files, vintages). Deferred (spec §10). No current consumer needs a non-prefix
  set.
- **Recommendation:** the partition type. Confidence: leaning. The marginal cost
  over Alternative S is small, and only the partition type fixes C1.

**`ConditioningRecord` with separate `Iθ` and `Ix`** (spec §4.3, §5.5).

- _Alternative: one information set._ Rejected. It cannot describe a forecast
  made after new data and before a refit, and that is the ordinary operational
  case. It also cannot describe one-step-ahead scoring honestly.

**`ForecastPlan` as the single target authority** (spec §4.4, §6).

- _Alternative: fix `forecast_times` to use `emit_schedule`, and leave
  `--init-state` alone._ Rejected. It leaves two target rules, and P1 needs the
  conditioning record anyway.
- _Alternative: put target scheduling in the IR._ Rejected. Targets depend on
  data (held-out rows) and on a forecast origin, which the compiler never sees.

**`CalendarAxis` factored** (spec §4.5). The cheapest way to satisfy I4 and I14
across three artifact kinds.

**What stays as it is.**

- The engines' `f64` times, `Schedule`, `StreamTimes`, `Coverage`, the trace's
  `Provenance` and `basis` axes, `combine`/`finish`, the evidence label, and the
  CLI time grammar.
- Every DSL surface. One exception is optional: calendar-month emit schedules
  (Stage 7, decision D7).

## 6. Staged plan

Each stage lands green under `make test` and is one coherent change. Behaviour
changes are named; everything else must leave existing numbers bit-identical.

**Stage 0: independent small fixes.** No dependencies, and each is its own
commit and issue.

- (a) **C3.** In `derive_ode_trace`, use `heldout_scoring_boundary`, and add a
  `day(time)` case to `compare_ode_prequential.rs`. Behaviour change: refusal →
  score.
- (b) **K1.** Compute each `date_range` entry from the start. Behaviour change:
  month-end `date_range` instants.
- (c) **K2.** E321 on a calendar-classified `covers` width. Behaviour change:
  refusal of a currently compiling form.
- (d) **A3, K7.** Error and help text.
- (e) **K3.** Refuse `Instant + Instant`, `k · Instant`, `Duration − Instant`
  and `Instant + bare number` under an origin (a new error code). Behaviour
  change: refusals.
- (f) **F1.** Every run that evaluates a data-backed forcing (interpolated,
  cubic, piecewise) outside its knot range says so once, as a `RunNote` naming
  the forcing, the knot range and the span held, as B-splines already do
  (`spline-held-past-window`, `cli/src/obs_anchor.rs:497-560`). This applies to
  plain `simulate` and `fit`, not only to forecasts: the hold is live today on
  every path. _Verified (read)_ at `0ed74876d` (§2.3, F1); the June audit
  reproduced it by running (`2026-06-09-time-interval-model.md`, defect F2: a
  forcing whose last knot is `t = 750` held at that value over `t = 751..1100`,
  exit 0, no message). Numbers do not change; the note is presentation. Stage
  5's plan notes reuse it (decision D11).

**Stage 1: one row reader.** Land the score-artifact proposal's `observed_rows`
with `support` in place of `time`, and rebuild `load_observations` and
`stream_times_for` on it.

- Pure refactor: every existing test unchanged, every fit bit-identical.
- Depends on nothing.
- This is also score-artifact step 1, so both proposals share it.

**Stage 2: `TrainingPartition`.**

- Replace the label cuts listed in §2.2 (`within_training`,
  `resolve_holdout_cutoff`, the `[data.holdout]` checks, `survey`'s copy,
  `profile`, `training_scoring_ends`) with `TrainingPartition::from_cutoff` /
  `whole`.
- Write `training_partition` in `fit.meta.json` for every fit. `compare`
  classifies against the extent (spec §5.2).
- Resolve `last_obs` for the holdout anchor and for the DSL anchors over
  supports. Decision D2 settles which rows.
- **Behaviour changes:**
  - Under `day`, `starting_on` and `ending_on` forms, the row straddling `τ`
    stops training (C2).
  - Straddling rows stop being scored (C1).
  - `fit.meta.json` changes shape.
  - `closing_at` and window-column streams train on the same rows as today, and
    their fits stay bit-identical. This is required, and pinned by test.
- Re-keys: the partition rule enters `FitDigest`. This lands in the
  `HASH_VERSION` 3 bump with the score-artifact identity changes (decision D8).
- Depends on Stage 1.

**Stage 3: alignment at the right boundary.**

- Delete the `time % dt` check in `load_observations`.
- Check per algorithm, after the algorithm is known, measuring from `t_start`:
  - if2, pfilter and pmmh step exactly to every off-grid open and close.
  - PGAS refuses any off-grid open or close, naming the row and the nearest grid
    times (decision D6).
- Delete the `[config] obs_alignment` key, `ObsAlignment` and
  `resolve_obs_alignment`. With the refusal above, the policy no longer changes
  any number: on-grid data runs identically under either value, and off-grid
  data is either stepped to exactly or refused, determined by the algorithm.
  Today the key can only equal its algorithm's default or be refused
  (`fit/methods.rs:515-551`), and nothing in `sim` reads the resolved value; it
  is a validation gate and a run-identity input (`runid/src/inputs.rs:468`).
  What a method can do becomes a capability of the method in the `METHODS`
  registry (`docs/dev/capabilities-system.md`), and `run.json` records the
  stepping the run used. Removing the key touches configuration and identity,
  not how PGAS receives observation times; the comment at `fit/pgas.rs:712`
  anticipating a flipped default is updated to name the capability.
- **Behaviour changes:**
  - Off-grid data now fits under if2, pfilter and pmmh (A1).
  - PGAS refuses the A2 case instead of shifting it.
  - A fit.toml with `obs_alignment` is refused with a message saying the key was
    removed.
- Re-keys: dropping the field from the fit identity rides the D8 bump.
- Depends on Stage 1.

**Stage 4: `CalendarAxis` in every artifact; render and parse round-trip.**

- Add `calendar` to the simulation leaf's `run.json`, `fit.meta.json` and the
  predictive sidecars, all through one type.
- Make the loader accept the sub-day form the renderer writes (gh#839), with a
  property test (I14).
- Print dates beside `t` in dated diagnostics (M3).
- Re-key: the simulation leaf's stored bytes change, so `SimConfig`'s schema
  version is bumped.
- Depends on nothing.

**Stage 5: `ForecastPlan` and `ConditioningRecord` in prediction.**

- `fit predict` and `simulate --init-state` build a plan, and drop
  `forecast_times` and `leaf_row_coverages`.
- `fit predict` reads the fit's partition.
  - Under `one_step`, the band is computed over `Iθ` plus held-out rows. Rows
    are marked `trained / held_out / straddles / target`, and the JSON states
    both information sets (P1, P2).
- The plan's notes cover future covariates (F2) and straddled first periods
  (T3), and carry Stage 0(f)'s forcing-domain note (F1) into the predictive
  artifact.
- `fit predict` gains `--horizon DURATION`. The mode flag is renamed (decision
  D3).
- **Behaviour changes:**
  - Forecast target times move for any series whose modal gap differs from its
    declared span or schedule (T1, T2).
  - Window-column series with no `emit_schedule` are refused a forecast tail
    (decision D5).
  - The predictive TSV gains `role`, `t_start` and `t_stop` columns (gh#624).
- Depends on Stages 1, 2 and 4.

**Stage 6: forecast-mode scoring (gh#808).**

- On top of the score artifact, add `ScoredLeadTime::FromOrigin`, a per-target
  `lead`, and per-lead summaries never summed.
  `camdl score FIT
  --horizon "4 weeks"` scores the `FixedOrigin` plan whose
  forecast origin is the training extent.
- `compare` refuses to difference scores with different modes, forecast origins
  or lead sets.
- The free-running filter after the forecast origin uses the existing
  `--init-state` machinery for chain-binomial, and the ODE free-forward basis
  for ODE.
- Depends on Stage 5 and on the score artifact.

**Stage 7: calendar periods (deferred to its own decision, D7).**

After Stage 0(c), a monthly flow stream can be declared only with per-row window
columns, has no declared cadence, and is refused a forecast tail (D5). What a
monthly row spans would then need three declarations: window columns to load it,
an emit schedule to forecast it, and the same schedule to simulate it. One
`covers` form states it once.

- **`covers = month(time)`**, anchored models only. The row's label must be the
  first day of a month; the support is
  `[that day, add_calendar_months(that day,
  1))`. A label on any other day is
  refused at load, which is a phase check the loader cannot make today. One IR
  variant, `Covers::Calendar { period: Month }`, serves all three uses:
  - loading: the support is computed with calendar arithmetic;
  - forecast targets: the stream becomes the spec's case 2 (spec §6.2), tiling
    forward by calendar months from its last close;
  - emission: `simulate` emits at the same calendar periods.
- **Week forms**, `iso_week(time)` (label a Monday) and `mmwr_week(time)` (label
  a Sunday; MMWR is the US CDC epidemiological week), are the same variant with
  `period: IsoWeek | MmwrWeek`. They add the ingestion check that matters most
  for weekly surveillance files: a file of Wednesday-dated rows declared as MMWR
  weeks is refused at load. They land in the same bump if the label convention
  (week-starting versus week-ending labels) is settled by then; if not, they
  follow without a further IR change to the variant's shape.
- **The calendar-month `emit_schedule`** remains, for instant streams only (a
  monthly census): `date_range` allowed in `emit_schedule at`, and a calendar
  cadence in `emit_schedule every` when an origin exists, lowered to calendar
  stepping from a stated start.
- The Rust runtime needs calendar-month stepping, mirroring the compiler's
  `add_calendar_months` with an equivalence test
  (`.claude/rules/calendar-time.md`, "Cross-language constants").
- One `ir/VERSION` bump and its golden update cover the variant and the
  schedule.
- Also in this stage: K4 (`date()` carries `[T]`), K5 (decision D9) and K6 (fold
  unit literals in `resolve_float_expr`).

**Dependency graph.**

```
0 (independent)
1 ──┬── 2 ──┐
    └── 3   ├── 5 ── 6
4 ──────────┘
7 (independent; unblocks monthly and weekly calendar-period forecasts)
```

**What must land before a forecast score can be trusted:** Stages 1, 2, 3 and 5,
Stage 0(a), Stage 0(c) (for monthly streams), Stage 0(f), and Stage 4's calendar
metadata for any forecast that is published. Stage 0(b) and (e) and Stage 7 are
independent improvements.

## 7. Compatibility and identity

- **Alpha posture.** No shims. `training_window` is renamed to
  `training_partition` atomically. Pre-change fit records cannot be scored
  held-out; the refusal says to re-run, which the hash bump requires anyway.
- **Re-keys.** Stage 2 (fit identity: the partition rule version and row-digest
  data), the score-artifact changes, and Stage 0(c) (models that stop compiling)
  are best landed in one `HASH_VERSION` bump (decision D8). Stage 4 bumps
  `SimConfig`. Stage 7 bumps `ir/VERSION` and its goldens, which is a human-loop
  change.
- **Goldens.** Stages 0(b), 0(c) and 0(e) change compiler output only for models
  using those forms. A regenerated golden under them is the commit's subject.

## 8. Tests

### 8.1 Must stay unchanged (bit-identical numbers)

These tests pin results that a redesign must not move:

- `sim/tests/prequential_scores_oracle.rs`
- `sim/tests/prequential_loglik_invariant.rs`
- `sim/tests/ode_prequential.rs`
- `cli/tests/compare_ode_prequential.rs` (its `closing_at` cases)
- `cli/tests/dated_data_loader.rs`, including `dated_loglik_matches_numeric` and
  `numeric_shift_invariance`
- `cli/tests/holdout_applied.rs`, its `closing_at` cases
- `cli/tests/compare_heldout_scoring_axis.rs`, its `closing_at`, files and
  window-column cases
- `fit/predict.rs::forecast_times_land_on_the_snapshot_grid`, converted to
  assert the same times through the plan for a regular `closing_at` stream
- `fit_predict_e2e.rs`, for free-forward bands at observed times
- the compare unit tests with exact values (`compare.rs:5193`, `:5266-5269`)

Stage 1 additionally requires every existing test to pass unmodified.

### 8.2 Must change, deliberately

Each change below is pinned by a new test that fails on `0ed74876d`:

- `compare_heldout_scoring_axis.rs`: the `day`, `starting_on` and `ending_on`
  cases. The trained set loses the row straddling `τ`.
- The two-stream case. The straddling weekly row is no longer scored (C1).
- `holdout_applied.rs::every_method_records_the_training_window`, which becomes
  `…_training_partition`.
- `gh696_predict_horizon_rows.rs`, `predict_forecast_close_past_horizon.rs` and
  `predict_keeps_one_step_when_tail_fails.rs`. These are re-expressed against
  plan targets.

### 8.3 New tests, by required property

| Property                                         | Test (fixture)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Start- vs end-labeled periods                    | the same synthetic series written under `day`, `starting_on`, `ending_on` and `closing_at`, with relabeled files: identical partitions, identical held-out scores (F2)                                                                                                                                                                                                                                                                                                                                                                                             |
| Period straddling the boundary                   | C1 and C2 as tests: the straddling row is neither trained nor scored; `fit.meta.json` extent equals the latest trained close                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Multiple cadences                                | daily plus weekly (C1 fixture); targets per leaf tile from each leaf's own last close                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| Irregular times                                  | window-column rows with gaps: held-out targets are exactly the observed supports; the forecast tail is refused without `emit_schedule`                                                                                                                                                                                                                                                                                                                                                                                                                             |
| Exact vs snapped alignment                       | F1 under if2 `exact` fits; the A2 case under PGAS is refused, naming the row; `run.json` records the policy                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Calendar months and leap years                   | `date_range(2020-01-31, count = 4, calendar_months = 1)` equals `add_calendar_months` from the start; month-end window rows across 2020-02-29                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Deterministic vs stochastic semantics            | one plan, three backends: identical target lists and identical `(open, close)` per target; no trajectory equality required. ODE predictive mean within Monte Carlo error of the chain-binomial mean at large population, as a distributional check                                                                                                                                                                                                                                                                                                                 |
| Incidence at boundaries                          | a target `[a, b)` equals `N(b) − N(a)` from the trajectory's cumulative flow at the recorded snapshots, on each backend                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Continuing the observation schedule              | a `closing_at(time, 7 'days)` stream whose observed gaps are irregular (a missing week): the tail tiles by 7 days, not by the modal gap                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Forcing data ending before the forecast horizon  | an interpolated forcing ending at 100 with a forecast to 120: one note, recorded in the artifact; the numbers are unchanged                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Parameter vs state information sets              | a fit with `holdout_after`; `fit predict --mode one_step` marks held-out rows and records `Iθ` ≠ `Ix`; `[data.holdout]` gives the identical record                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Membership proof                                 | a score on a file with an edited training row refuses as changed data; appended rows keep the fit's identity and are scored                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Round-trip                                       | dated output at sub-day `dt` loads back to the identical model times (I14); simulation leaf `run.json` carries `calendar`                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Forecast origin validation                       | E4, E5 from the spec as tests                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| Intervention on a period boundary with live flow | an intervention at `date("…")` (a day boundary), a `day(time)` incidence stream and transitions firing at non-zero rates, on chain-binomial, Gillespie and ODE: the flow of the day before the boundary is unchanged with and without the intervention, and the day it opens is affected (`docs/dates.md`, "Interventions and events are instants"). _Not verified:_ the claim that the existing cross-backend lifecycle test uses zero-rate models; the new test is needed either way, because it combines a boundary intervention with a scored incidence period |
| Forcing past its knots                           | Stage 0(f): `simulate` and `fit` on a model whose interpolated, cubic and piecewise forcings end before `t_end` each print the note once; numbers bit-identical with and without the change                                                                                                                                                                                                                                                                                                                                                                        |
| Removed `obs_alignment` key                      | a fit.toml carrying the key is refused with the removal message; PGAS on F1's A2 variant is refused                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

### 8.4 Fixtures

The following fixtures were written for this audit in a scratch directory and
are restated here so they survive.

- **F1 (off-grid observations).**

  ```camdl
  time_unit = 'days
  compartments { S, I, R }
  parameters { beta : rate in [0.01, 2.0]  gamma : rate in [0.01, 1.0]  rho : probability in [0.01, 1.0] }
  transitions { infection : S --> I @ beta * S * I / (S + I + R)
                recovery  : I --> R @ gamma * I }
  observations { cases { columns { time : time, cases : count }
    covers = closing_at(time, 2.5 'days)  projected = incidence(infection)
    cases ~ poisson(rate = rho * projected + 0.1) } }
  init { S = 990  I = 10  R = 0 }
  simulate { from = 0 'days  to = 30 'days  dt = 1 'days }
  ```

  Data at `t = 2.5, 5, …, 15`.

  The A2 variant: `from = 0.5 'days`, `covers = closing_at(time, 2 'days)`, data
  at `3, 5, 7, 9`, `algorithm = "pgas"`.

- **F2 (C1).** The same compartments, with two streams:

  ```camdl
  weekly { columns { time : time, cases : count }
    covers = closing_at(time, 7 'days)  projected = incidence(infection)
    cases ~ poisson(rate = rho * projected + 0.1) }
  daily  { columns { time : time, recovered : count }
    covers = day(time)  projected = incidence(recovery)
    recovered ~ poisson(rate = rho * projected + 0.1) }
  ```

  Weekly labels are 7..70 and daily labels 0..69, with `holdout_after = 24`. Two
  fits differing in a fixed parameter value, then
  `camdl compare fitA.toml
  fitB.toml --pointwise pw.tsv`. Today
  `pw.streams.tsv` lists `28:weekly`.

- **F3 (C2, P1).** A single `starting_on(time, 7 'days)` stream, labels 0..63,
  `holdout_after = 24`, PMMH. The fit prints `training window [0, 24]` (C2).
  `camdl fit predict --fit fit.toml --horizon one_step` writes ten one-step rows
  with no held-out marking (P1).

- **F4 (C3).** A `day(time)` stream, labels 1..16, `holdout_after = 10`,
  `algorithm = "mh"`, `backend = "ode"`. Today `camdl compare` exits 1.

- **F5 (T1).** A window-column stream with rows closing at 31, 59, 90, 120, 151,
  181, and `simulate { to = 365 }`. Today the forecast rows fall at 211, 241, ….

- **F6 (T3).** F2's weekly stream with `emit_schedule = every 7 'days`. Filter
  through data at 11, 18, 25, then run `simulate --init-state … --to 49 --obs`.
  Today the output rows are 35, 42, 49.

## 9. Risks to scientific behaviour

- **Stage 2 changes fitted numbers** for every `holdout_after` fit on a stream
  that is not `closing_at` or window-column. The row straddling `τ` leaves
  training. This is the intended correction, but it moves published held-out
  comparisons that used those forms.
  - Mitigation: the hash bump forces recomputation.
  - The release notes name the forms affected.
- **C1's fix shrinks held-out sets.** The straddling row was scored before and
  is not now, so the number of scored steps changes, and comparisons across the
  change are refused by the existing scored-step preflight. That refusal is
  correct.
- **Stage 3 removes a refusal.** Off-grid observations now fit under `Exact`.
  The exact-stepping path for observations has been exercised for interventions
  and output times, and for observations only via `caltime_load` off-grid
  warnings.
  - _Not verified:_ that every if2, pfilter and pmmh path honours an off-grid
    close.
  - The Stage 3 test runs F1 on all three and compares with a fit at `dt = 0.5`
    (on-grid) to Monte Carlo error.
- **Stage 5 moves forecast targets** for series whose modal gap differs from the
  declared span. Any downstream figure keyed to those times moves too.
- **High-risk files.** Stage 3 touches `pgas.rs` (the A3 messages) and how PGAS
  receives closes. Stage 6 touches `particle_filter.rs` (stopping assimilation
  at the forecast origin). Both require reading the full functions first, per
  `CLAUDE.md`.

## 10. Decisions for review

**D1. The partition rule.**

- The question is how a cutoff splits rows whose period crosses it.
- Options:
  - (a) Train iff close ≤ `τ`; score only rows opening at or after the training
    extent; straddlers are neither trained nor scored. This is the spec.
  - (b) Close-only (Alternative S): train iff close ≤ `τ`; score iff close >
    extent. Simpler, but scores straddlers, so C1 remains.
  - (c) Keep labels. Rejected: C2.
- (a) changes fits and held-out sets for non-`closing_at` streams only.
- **Recommendation: (a).** Leaning.

**D2. What `last_obs` means.**

- It is used by `holdout_after = "last_obs - N"`,
  `simulate { to = last_obs +
  … }`, `pfilter --score-from` and `fit predict`.
  Today it is the last label over the full files at some sites, and the last
  close at others.
- Proposal: `last_obs` is the latest close.
  - For `holdout_after` it is computed over the full files, since it defines the
    split.
  - For every other DSL anchor in a fit it is computed over `Iθ`, so that
    appending held-out rows never changes a fit's horizon or identity.
- Effects:
  - For `closing_at` and window-column streams, label equals close, and nothing
    changes.
  - For `day(time)` streams, `last_obs` moves one day later, and every model
    that anchors to it moves its `t_end` by one day.
- **Recommendation: as proposed.** Leaning. The alternative, keeping the last
  label, preserves `t_end` but keeps two meanings.

**D3. Vocabulary for "horizon".**

- The principles forbid the word because it had four meanings, and you asked for
  `--horizon "4 weeks"`.
- Proposal: "forecast horizon" is the maximum lead time, a duration, and nothing
  else. Accordingly:
  - `fit predict --horizon {free_forward, one_step}` becomes
    `--mode
    {free-forward, one-step}`;
  - `compare --allow-mismatched-horizon` becomes
    `--allow-mismatched-scored-steps`;
  - the `simulate --to` help stops calling the run end a horizon;
  - the principles' vocabulary is amended.
- Trade-off: two flag renames, which the alpha allows, against a reserved word
  for the concept every forecasting user expects.
- **Recommendation: as proposed.** Leaning. The alternative is to keep the ban
  and spell the flag `--lead "4 weeks"`.

**D4. The default forecast origin, and whether users may move it.**

- Default `s =` the latest close in `Ix`. With `Ix = Iθ` that is the training
  extent.
- Proposal: allow `--forecast-origin` (date or model time, never earlier than
  `Ix`'s last close) and `--condition-through` (extend `Ix` past `Iθ` with newer
  rows) in Stage 5 on chain-binomial.
  - These two cover "fit last month, forecast today".
  - Leaving them out keeps the CLI smaller, but forces a refit for every new
    week of data.
- **Recommendation:** ship the default in Stage 5; add the two flags in Stage 5
  only if the `--init-state` path makes them cheap, else defer. Leaning.

**D5. Forecast targets for a leaf with no declared cadence.**

- This concerns a window-column series without `emit_schedule`, including every
  monthly series until Stage 7.
- Options:
  - (a) Refuse the forecast tail, saying how to declare a cadence.
  - (b) Keep the modal gap with a warning.
- (a) is honest but blocks monthly forecasts until Stage 7, whose `month(time)`
  form gives a monthly stream a declared cadence (D7); (b) produces T1-type
  wrong targets that a warning does not prevent anyone from using.
- **Recommendation: (a).** Solid.

**D6. PGAS and off-grid supports.**

- Options:
  - (a) Refuse.
  - (b) Snap, and record the per-row numerical period beside the declared one.
- (b) lets more models run under PGAS but scores a period other than the
  declared one; the record only documents that.
- **Recommendation: (a).** Solid. A user can choose `dt` to put the supports on
  the grid.
- **Consequence: retire `obs_alignment` (Stage 3).** Under (a) the setting no
  longer changes any number, so it would remain only as a run-identity input
  that never alters a result. Delete it, record the method's capability instead,
  and ride the D8 bump. Leaning, because a future PGAS exact-stepping path would
  then arrive as a capability change rather than a user setting, which is the
  more honest form for it anyway.

**D7. Calendar periods (Stage 7).**

- The question is how a monthly (or ISO/MMWR weekly) series states its periods
  once Stage 0(c) refuses `closing_at(time, 1 'months)`.
- Options:
  - (a) Calendar `covers` forms, `month(time)` first and the week forms in the
    same IR variant, plus the calendar-month `emit_schedule` for instant streams
    (Stage 7 as written).
  - (b) The calendar-month `emit_schedule` alone. Monthly flow streams keep
    window columns to load and the schedule to forecast: two declarations of one
    fact, and no phase check at ingestion.
- Both are one `ir/VERSION` bump. (a) is a new DSL form, which this proposal
  otherwise avoids; the usability gap it closes is demonstrable (a monthly
  series cannot be forecast at all, and a mis-phased weekly file loads without
  complaint today).
- **Recommendation: (a), after Stage 5, bundled with any other pending IR
  bump.** Leaning. If the week label convention is not settled, ship
  `month(time)` and the schedule first; the week forms follow without changing
  the variant.

**D8. One hash bump.**

- Proposal: land Stage 2's fit-identity change, the score-artifact identity
  changes and the docket's D25 (`HASH_VERSION` 3) together. Every stored result
  then re-keys once.
- The cost is sequencing: Stage 2 waits for the score-artifact identity work, or
  the score-artifact work adopts Stage 2's partition.
- **Recommendation: bundle.** Solid. Adopt the partition into the score-artifact
  implementation, since its `training_rows` is exactly `LeafPrefix`.

**D9. Emit grid phase (K5).**

- The code phases a generated grid from `simulate.from`, while `docs/dates.md`
  says `origin`.
- For forecasting it does not matter: targets tile from observed closes.
- Options:
  - (a) Fix the doc.
  - (b) Fix the code, which moves generated observation times for every model
    whose `from ≠ origin`.
- **Recommendation: (a).** Leaning. `from` is the run's own start, and changing
  the code would move synthetic data in existing fixtures for no forecasting
  benefit.

**D10. Amending the score-artifact proposal.**

- That proposal needs three edits before it is implemented:
  - `ObservedRow.time` becomes `support`;
  - `train_end` becomes extent plus prefix;
  - trace schema 6 becomes 7.
- **Recommendation:** amend it in place when this proposal is accepted, rather
  than carrying a contradiction between two documents. Solid.

**D11. A forcing evaluated past its knots: note or refusal.**

- `2026-06-09-time-interval-model.md` §7.1 (point 5) proposed refusing a run
  that integrates past the end of an interpolated forcing. The spec's I13
  records it instead.
- Options:
  - (a) A note on every run (Stage 0(f)), carried into every predictive and
    score artifact. Endpoint holding is a legitimate scenario assumption ("hold
    temperature at its last observed value"), so a refusal would block
    deliberate use and push modelers to pad their tables with invented values,
    which hides the assumption instead of stating it.
  - (b) A refusal, with an explicit opt-in. Safer against a forgotten table end,
    at the cost of a new opt-in surface before any declaration exists for it.
- **Recommendation: (a), stated as overriding the June draft's point 5.**
  Leaning on note over error; solid that the note must appear on every run, not
  only on forecasts. An explicit per-forcing `extrapolation = hold | error`
  declaration remains deferred (spec §10) and would make (b) available per
  forcing.
