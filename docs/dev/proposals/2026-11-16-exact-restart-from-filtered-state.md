# Proposal: exact restarts from a filtered state

**Status:** Proposal **Date:** 2026-11-16 **Implements:** D64 (gh#1081, gh#1085)

`camdl pfilter --save-final-state` writes the filtered particle swarm at the end
of the data, and `camdl simulate --init-state` restarts a simulation from it.
Maintainer decision D64 keeps the pair as a permanent lower-level forecasting
tool and asks for it to be hardened: a restart in the middle of a reporting
period must be exact, the file must record the model, parameters and scenario it
was filtered under and be checked against them, and an edited file must be
detected. Today a restart drops or truncates the period under way (gh#1081,
gh#1085), trusts an edited header, and cannot tell what it was filtered under.

This proposal makes a restart exact and checked. The periods a restart writes
come from the function that builds the fixed-origin forecast's targets, so a
restart and a forecast cannot disagree about what a stream's next period is. The
saved-state file (format v2) carries each emitted stream's running total for the
period under way, the data periods the period function reads, and a provenance
block sealed by a checksum. `simulate --init-state` completes the straddling
period and refuses a file that does not match the run.

**Background assumed.** Observation periods and `covers` forms as in
`docs/dev/temporal-architecture.md` §2.2 (the spec); where targets come from,
spec §4.3; how a target's value is computed, spec §4.4; the user-facing workflow
in `docs/inference.md`, "Forecasting from the filtered state". The decisions
cited (D50, D64, D72, D77, D82) are recorded in Part 8 of
`docs/dev/proposals/2026-11-15-temporal-architecture-migration.md` (the
runbook). Everything else is cited where it is used.

Code references are to `main` at `6dce561c9`. _Verified (read)_ means read in
the code at that commit; _verified (run)_ means reproduced by running binaries
built from it, recorded in the cited issue or test; _inferred_ means neither.
The commands behind the verified claims are in §12.

## 1. Terms

- **Forecast origin `T`.** The model time the saved states are at: the last stop
  of the particle filter's union axis (the sorted merge of every bound stream's
  period ends and declared period starts). `pfilter` takes it as the time of the
  last union stop (`rust/crates/cli/src/pfilter.rs:1084`). Under D72 that stop
  need not carry an observation: a trailing row every stream holds as `NA` is
  still a stop.
- **Emitted stream.** An observation stream with an `emit_schedule`, so
  `simulate --obs` writes rows for it. A fit-only stream (no `emit_schedule`)
  writes nothing after a restart and needs nothing carried
  (`rust/crates/cli/src/main.rs:2731-2738`).
- **Interval stream.** A stream whose projection accumulates a flow
  (`incidence(...)` or a ratio of flows). An instant stream reads a state at its
  row's time.
- **Period under way at `T`.** For an emitted interval stream, the stream's
  first period ending after `T` (its horizon-0 target in the forecast's terms,
  spec §2.5), when that period starts before `T`. At a boundary of the stream's
  periods there is none.
- **Running total `F_s[a, T)`.** The flow stream `s` projects (the sum of its
  transitions' counts) accumulated by one particle over `[a, T)`. For a ratio of
  flows it is two totals, numerator and denominator.
- **Straddling row.** The emitted row whose period is the period under way: it
  began before the restart and closes after it.

## 2. What happens today

**The file holds counts and an origin, nothing else.** Verified (read):
`write_final_states` (`rust/crates/io/src/final_state.rs:118`) writes the header
`# camdl-final-state v1<TAB>t=<T>` (`FORMAT_TAG`, `:65`), a `particle` column
and one column per integer compartment. `read_final_states` (`:176`) resolves
compartments by name and takes `t` from `parse_origin_t` (`:299`), which checks
only that it is a finite number. `simulate` reads the file at
`rust/crates/cli/src/main.rs:1506-1556` and checks nothing about where it came
from beyond the `--draws` and replicate-count refusals there; the ensemble's
identity is the SHA-256 of the file's bytes (`:1547`). The run's `t_start`
becomes `T` at `rust/crates/cli/src/util.rs:3810`.

**Defect 1, the first forecast period is lost or truncated.** Verified (read),
and reproduced (run) in gh#1081. A restarted run computes its emission times
with `obs_schedule_times` (`main.rs:4275`), which drops every _label_ before `T`
(`:4280-4281`). `plan_emission` (`rust/crates/cli/src/obs_emit.rs:167`) then
assigns periods by a rule of its own, separate from the forecast plan's:

| `covers` form                | Restart at `T = 26`, weekly rows closing 7, 14, 21, 28, …                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `closing_at(time, 7 'days)`  | label 28 survives the label filter, its period `[21, 28)` starts before the run, and `within_run` (`obs_emit.rs:177`, applied at `:214`) drops it silently; output begins at 35 |
| `starting_on(time, 7 'days)` | label 21 (period `[21, 28)`) is dropped by the label filter before planning                                                                                                     |
| `ending_on`, `day`           | as `closing_at` or `starting_on`, by where the label sits in its period                                                                                                         |
| `window_start`/`window_stop` | the first window opens at the run start (`prev = t_start`, `obs_emit.rs:229`), so row 28 is written as the partial `[26, 28)` under the full row's label (gh#1085, inferred)    |

The root cause is the same in every row: the file has no record of what each
particle accumulated over `[a, T)`, so no row can be completed.

**Defect 2, the header is trusted.** Verified (read): changing `t=26` to `t=3`
yields a valid file; nothing ties `t` to the states.

**Defect 3, no provenance.** Verified (read): the file records no model,
parameter vector, scenario or `dt`; restarting under different ones is silent.

**Kept as is.** Interventions that fired before `T` are not replayed on restart.
A `T` off the output grid is refused
(`rust/crates/sim/src/chain_binomial.rs:467-475`). Reactive interventions and
backends other than chain-binomial are refused (`main.rs:1479-1503`).

## 3. Why counts are not enough, and what is

A straddling row's value is the stream's flow over `[a, b)`, which splits at the
origin:

`F_s[a, b) = F_s[a, T) + F_s[T, b)`.

The restarted run produces the second term. The first is a property of each
particle's _path_ before `T`, not of its state at `T`: two particles with
identical counts at `T` can have counted different numbers of infections since
`a`. So the saved row must carry `F_s[a, T)`, and the pair `(x_T, F_s[a, T))`
must come from the same particle. The bootstrap filter provides exactly that: a
quantity accumulated along each particle's path travels with the particle
through resampling (`rust/crates/sim/src/inference/particle_filter.rs:780`), so
after the final assimilation `(x_T, F_s[a, T))` is a draw from the filtering
distribution `p(x_T, F_s[a, T) | y_{1:T}, θ)`. The spec's experiment (§4.4)
checked this for the forecast driver: a running total opened at the right start
reproduced an importance-sampling reference (68.245 ± 0.021 against 68.227 ±
0.034).

The pre-origin part is latent and conditioned on the data through `T`; it is not
known. The stream's own row for `[a, b)` cannot have been assimilated, because
`b > T` and the filter stops at `T`.

## 4. Which periods a restart writes: the forecast plan's

A restart and a fixed-origin forecast answer the same question for every emitted
interval stream: given the data through `T`, which periods come next, and which
of them is under way? The forecast plan already answers it
(`rust/crates/cli/src/fit/forecast_plan.rs`, spec §4.3, D80): a stream with a
uniform `covers` form continues its data's tiling at its declared width
(`TargetSource::DeclaredTiling`, phase from the last data row,
`forecast_plan.rs:486-511`); a stream with window columns and an `emit_schedule`
takes the periods between consecutive schedule times
(`TargetSource::EmitSchedule`, `:524-533`); and where the data's periods and the
`emit_schedule`'s disagree within the reach, the stream is refused with
`StreamRefusal::ScheduleConflict` (D82, `check_schedule_agrees`, `:679-720`).
Verified (read).

A restart therefore uses that rule, not a second one. `plan_emission`'s own
label-to-period rule (`obs_emit.rs:197-245`) stays for runs that do not restart
and is not consulted for a restarted cell's interval streams.

**The seam.** `stream_targets` (`forecast_plan.rs:470`) does two jobs: it
computes a stream's periods after the origin (the match on the stream's temporal
kind and `covers`, and the D82 check, `:482-549`), and it turns them into
`Target`s (held-out rows, design values, horizons, leads, `:550-593`). The first
job moves into its own function, which `stream_targets` calls and the restart
calls:

```rust
/// What the period rule reads of a stream's data: each row's file, position
/// and the period (or instant) it measures. Never its value.
pub(crate) struct DataPeriods<'a> {
    pub training: &'a [ObservedRow],
    pub held_out: &'a [HeldOut],
}

/// A stream's periods (or instants) that end in `(origin, reach]`, and where
/// they come from (§4.3). Refuses the stream when it declares no rhythm, binds
/// no rows for a tiling, or when its data and its emit schedule disagree within
/// the reach (D82). Reads no row's value and no design.
pub(crate) fn stream_periods(
    obs: &ObservationModel,
    data: &DataPeriods,
    origin: f64,
    reach: f64,
) -> Result<(TargetSource, Vec<Measures>), StreamRefusal>;

/// The stream's first period ending after `origin`, when it starts before
/// `origin`: `stream_periods` with the reach set to the smallest one that
/// includes that period (the span for a tiling, the step for a regular
/// schedule, the first listed time after `origin` for an `at [...]` schedule).
pub(crate) fn period_under_way(
    obs: &ObservationModel,
    data: &DataPeriods,
    origin: f64,
) -> Result<Option<Period>, StreamRefusal>;
```

The body of `stream_periods` is today's `:482-549` unchanged; it reads only
`measures`, `position` and the held-out row's file, never `cell`, `aux` or
`design` (verified by reading). The design check at `:478-481` stays in
`stream_targets`, because it concerns predictions, not periods; a restart's
design-reading streams keep `simulate --obs`'s existing refusal
(`check_data_columns_supplied`, `obs_emit.rs:450`, applied at `main.rs:1582`).
`stream_targets`' output is byte-identical before and after the move, which the
existing `forecast_plan.rs` tests pin.

A restart calls `stream_periods` with `held_out` empty and with `training` set
to the rows the filter bound for the stream, so the restart's periods are the
ones a forecast from the same rows and origin would target. Twice:

1. `pfilter --save-final-state` calls `period_under_way` with origin `T` to
   learn where each running total opens (§5), and writes the stream's data
   periods to the file (§6).
2. `simulate --init-state` calls `stream_periods` with origin `T`, the data
   periods read back from the file, and `reach` set to the cell's run end, to
   plan every row it writes (§7.4).

**D82 for restarts.** Where a stream's data-implied periods and its
`emit_schedule` disagree within the reach, the restart is refused with the
forecast's diagnostic: the stream, both periods and the source of each
(`StreamRefusal::ScheduleConflict`, `forecast_plan.rs:191-198`, rendered at
`:276-282`). There is no precedence rule. The variant's message names its
consumer ("is refused a forecast"); it gains the consumer as a parameter, so the
restart prints the same sentence with "is refused a restart from a filtered
state". At `pfilter --save-final-state` a refusal does not stop the save: the
stream gets its data periods and no running total, and `pfilter` prints the
refusal as a note saying that a restart which writes this stream will be
refused. `simulate --init-state --obs` then recomputes it from the same inputs
and refuses. A `simulate --init-state` that writes no observations needs no
periods and is not refused.

**What this changes for the gh#1081 reproduction.** Fixture F6 (runbook Part 6)
filters a weekly `closing_at(time, 7 'days)` stream through data closing 11, 18
and 25, under `emit_schedule = every 7 'days`, whose periods close 7, 14, 21,
28. The data's period `[4, 11)` overlaps the schedule's `[0, 7)` and `[7, 14)`
without being either, so under D82 a restart that writes this stream is refused,
naming `[4, 11)` (training row 0) and `[0, 7)` (the emit schedule). Inferred
from reading `check_schedule_agrees` against F6 as gh#1081 describes it, with
the schedule starting at the model's `t_start = 0`; test 7 in §10 runs it. The
straddling case this proposal completes is the one where data and schedule agree
and `T` falls inside the stream's period because another stream observed later
(spec §2.5, Figure 3): weekly data closing 7, 14, 21 beside a daily stream
observed through day 25, so `T = 26` and the week `[21, 28)` is under way.

**`--emit-every`.** The override replaces a stream's `emit_schedule` cadence at
its consumption sites (`rust/crates/cli/src/emit_every.rs:209`,
`apply_override`). Under a restart it does the same, and `stream_periods` reads
the overridden schedule. For a window-column stream the override changes the
periods themselves; for a uniform stream the periods follow the data's tiling,
and an override that disagrees with it is a D82 refusal. In both cases the
restart's period under way is then checked against the file's (§7.3).

**What needs nothing carried.** An instant stream reads the state at its row's
time, which is after `T`. A stream whose likelihood reads a state at the period
end (a binomial whose `n` is a compartment) has its end `b > T` on the restarted
trajectory.

## 5. Where the filter keeps the running totals

The filter already carries per-stream running totals. Verified (read):
`ParticleState.acc` (`rust/crates/sim/src/inference/types.rs:632`) holds one bin
per incidence stream and two per flow-ratio stream, laid out by
`MultiStreamObsModel`'s `interval_slots` (`multi_stream_obs.rs:1958-1975`). It
is folded once per union interval (`fold_into_acc`, `multi_stream_obs.rs:2061`,
called at `particle_filter.rs:530`), copied with the particle when the filter
resamples (`particle_filter.rs:780`), and reset per bin only where that bin's
reset schedule fires (`reset_due_acc`, `multi_stream_obs.rs:2102`, called at
`particle_filter.rs:826`). The filter returns the swarm after the last stop as
`final_states` (`particle_filter.rs:866`), and that is what the writer receives
(`pfilter.rs:1082-1094`), which discards `acc`.

**A stream's own bin cannot hold the running total.** A bin resets at each
declared period start inside the run (`multi_stream_obs.rs:1434-1462`), and the
only way to declare one is to bind a period, whose end the binder also puts on
the union axis (`:1393-1404`). Declaring the period under way `[a, b)` would
open the bin at `a` but also add the stop `b > T`, and the filter would run past
the origin. And an emitted stream the filter does not bind has no bin at all.

**The design: carry slots.** The observation model gains a second, separate set
of bins, one per emitted interval stream with a period under way at `T` (two for
a ratio of flows), appended after the scoring bins in the same `acc` vector:

```rust
/// A running total kept for a restart, never scored.
pub struct CarrySlot {
    /// The stream leaf this total belongs to (an indexed block's leaf name).
    pub stream: String,
    /// Which side of a ratio of flows, or `Total` for an incidence stream.
    pub part: CarryPart,
    /// Transitions whose counts the total sums (the stream's projection).
    pub flow_indices: Vec<usize>,
    /// The period start `a`, where the total opens (is zeroed).
    pub opens_at: f64,
}
pub enum CarryPart { Total, Numerator, Denominator }

impl BoundObs {
    /// Carry slots for `--save-final-state`. Puts each `opens_at` on the union
    /// axis when it is not already a stop. Built only by `pfilter`.
    pub fn with_carry(self, carry: Vec<CarrySlot>) -> Result<Self, Finding>;
    /// The `acc` index of each carry slot, in `carry` order.
    pub fn carry_bins(&self) -> &[usize];
}
```

`n_interval_streams()` (`multi_stream_obs.rs:2003`) counts scoring and carry
bins together, so every `ParticleState` is sized for both
(`particle_filter.rs:240`, `:271`, `:285`) and the whole-vector copy on
resampling (`:780`) carries them. `fold_into_acc` and `reset_due_acc` iterate
both kinds; a carry slot resets only at the union index of `opens_at`. Scoring
reads bins through `stream_bins` (`project_stream_from_acc`,
`multi_stream_obs.rs:2133`, `:2141`), which never points at a carry bin, so
scoring is unchanged. **`particle_filter.rs` is not edited.** Carry slots are
built only by `pfilter --save-final-state` (on the binder output of
`pfilter.rs:632-637`); every other binder caller passes none, so `acc` and every
inference path are unchanged for them. The carry slots of an emitted stream the
filter does not bind are subject to Decision 1 (§11).

**When `a` is not already a stop.** The fold happens once per union interval, so
a total can open only at a union time. When `a` is already one (another stream's
period end, a declared start, or the last data row's end), the axis does not
change. When it is not, `with_carry` inserts `a`. A stop with no observation is
not an assimilation event (D72). Verified (read) at `particle_filter.rs:561-571`
and `:771`: at such a stop the filter folds the flows into `acc` and resets the
bins due there, sets every live particle's log-weight to 0 rather than scoring
it, adds no likelihood term (`:728-736`), opens no collapse-check window
(`:751-756`) and skips the resampling call, so neither the resampling stream nor
the ancestry moves. The propagation draws are per-particle streams carried
across stops (`:487-512`), so an extra stop on the `dt` grid does not change
them. Verified (run) by `a_value_less_period_start_leaves_the_filter_unchanged`
(`rust/crates/sim/tests/assimilate_only_at_observations.rs:356`), which compares
the latent process, ancestry and likelihood bit for bit, and end to end by
`rust/crates/cli/tests/pfilter_assimilates_only_at_observations.rs`, which also
compares the saved final particle states.

So inserting `a` leaves the log-likelihood, the particles, their ancestry and
the saved counts exactly as they are without it, and the pfilter leaf, which
stores the log-likelihood (`pfilter.rs:835-843`), is unchanged and keeps its
key. Two per-stop outputs do see the stop, as for any value-less stop: `--trace`
and `--save-prequential` gain a row at `a`, and the predictive draws recorded
after it (the trace's predictive quantiles, the prequential CRPS and PIT) come
from a diagnostic stream the extra stop advances
(`pfilter_assimilates_only_at_observations.rs:9-13`). `pfilter` says once which
times it inserted and why.

An `a` off the `dt` grid is refused before filtering:

```
error: --save-final-state: stream '<s>': the period under way at the last
stop t = <T> opens at t = <a>, which is not on the dt = <dt> grid, so the
filter cannot stop there to start its running total.
  Fix: filter with a dt that divides <a>, or declare the stream's periods so
  that they start on the dt grid.
```

## 6. Saved-state format v2

```text
# camdl-final-state v2
# t	26
# dt	1
# model	4be1…<64 hex>
# param	beta	0.31
# param	gamma	0.1
# param	rho	0.4
# scenario	baseline
# intervention	vaccination
# source	pfilter	9a1e…<64 hex>
# rows	weekly	0	0	7
# rows	weekly	1	7	14
# rows	weekly	2	14	21
# open	weekly	21	28
# checksum	sha256	<64 hex>
particle	S	E	I	R	carry:weekly
0	9812	14	31	143	57
1	9805	19	28	148	61
```

Header lines are `# <key><TAB><fields…>`, in this order, each present exactly
once except `param`, `intervention`, `rows` and `open` (zero or more, sorted by
name, and `rows` by stream then position):

- `t`: the forecast origin, Rust's shortest round-trip `f64` form.
- `dt`: the filter's step.
- `model`: `runid::inputs::model_ir_hash` of the compiled IR as loaded by
  `util::load_model`, before scenario or parameter resolution: the function the
  `model` identity level and the PGAS trajectories header both use
  (`rust/crates/runid/src/inputs.rs:316`; `model_identity_from_ir`,
  `rust/crates/cli/src/resolve.rs:104-112`).
- `param`: every parameter value the filter ran at, after scenario `set`/`scale`
  and `--param`/`--params` resolution (`pfilter.rs:377`), round-trip form.
- `scenario`: the scenario label; `intervention`: the name of every intervention
  live in the filtered model under that scenario, one line each. With the model
  hash these rebuild the filtered model exactly (§7.2).
- `source`: the command and the `run_id` of the pfilter leaf
  (`pfilter_cas::resolve_pfilter`), computed even under `--no-store`. Display
  only; nothing checks it.
- `rows`: for each emitted interval stream, one line per row the filter bound:
  the stream's leaf name, the row's position in its data file, and the start and
  end of the period it measures. These are `stream_periods`' inputs (§4), with
  no values. A stream's rows after a `holdout_after` cut are the training rows
  the filter bound (`pfilter.rs:511-560`).
- `open`: one line per stream with a period under way and no refusal: leaf name,
  `a`, `b`.
- `checksum`: SHA-256 (`runid::ContentHash::digest_bytes`) of the file's bytes
  with this one line removed.

Columns: `particle`, every integer compartment in model order (as v1), then one
`carry:<stream>` column per incidence stream with an `open` line, or
`carry:<stream>:numerator` and `carry:<stream>:denominator` for a ratio of
flows, in `open` order. Carry values are non-negative integers. Compartment
names are identifiers, so a `carry:` column cannot collide with one. A stream at
a period boundary has no `open` line and no column.

The reader parses into one type, so nothing downstream re-checks:

```rust
pub struct FinalStates {
    pub origin_t: f64,
    pub provenance: Provenance,          // dt, model hash, params, scenario, interventions, source
    pub rows: Vec<StreamRows>,           // per emitted interval stream: (position, start, end) per row
    pub open: Vec<OpenPeriod>,           // stream, start a, stop b; a < origin_t < b
    pub counts: Vec<Vec<i64>>,           // [particle][integer compartment]
    pub carry: Vec<Vec<u64>>,            // [particle][carry column, `open` order]
}
```

## 7. Checks and output at `simulate --init-state <file>`

### 7.1 Integrity and version, once per file, in `read_final_states`

The first line decides the version. A v1 file is refused:

```
error: --init-state: <path> is a version-1 filtered-state file. Version 2 also
records each stream's running total for the period under way at the origin and
the model, parameters and scenario the state was filtered under; a version-1
file holds neither, so a restart from it cannot write its first forecast period
or be checked against this run.
  Fix: regenerate it with this camdl:
  camdl pfilter <model> --data <data> --params <params> --save-final-state <path>
```

Then the checksum, before any other field is believed:

```
error: --init-state: <path> has been changed since camdl pfilter wrote it: its
contents hash to <got8>…, but its checksum line records <rec8>…. A saved state
ties a forecast origin, a parameter vector and every particle's counts and
running totals together; an edited file cannot be checked, so it is refused.
  Fix: regenerate it with camdl pfilter --save-final-state. To start a run from
  a state of your own, write it in the model's init {} block.
```

A missing checksum line gets the same message with "has no checksum line". The
remaining structural checks are v1's (compartments by name, integer counts) plus
`a < t < b` on every `open` line, a carry column for each, and `rows` periods of
positive width in increasing position order.

This makes defect 2 a refusal: editing `t=26` to `t=3` changes the bytes.

### 7.2 Model, step, parameters and scenario, against the run

Once per run:

```
error: --init-state: <path> was filtered under a different model (<rec8>…) from
the one being simulated (<got8>…). Particle states are meaningful only for the
model whose dynamics produced them.
  Fix: re-run camdl pfilter --save-final-state with <model>.
```

```
error: --init-state: <path> was filtered at dt = <rec>; this run uses
dt = <got>. A chain-binomial model at another step is another process.
  Fix: pass --dt <rec>, or re-run the filter at dt = <got>.
```

Per cell, because a sweep point or scenario resolves its own parameters and
interventions, the cell is compared with the filtered model by the rule a
forecast arm is compared with the fitted model (D50): the cell may differ only
in what first acts at or after `T` (Decision 2, §11). The filtered model is
rebuilt from the file (the model, its `intervention` set, its `param` values).
The comparison is `first_difference` (`fit/forecast_arms.rs:198`), which today
takes its baseline from `ArmInputs`; it is narrowed to take the two compiled
models, their parameter vectors, the origin and `dt`, so the forecast arms and
the restart call one function. It refuses a switched intervention that first
fires before `T` or at a time that is not fixed before the run, and a changed
parameter the model reads outside interventions or inside one that fires before
`T` (`:226-262`, verified by reading). Its reason is printed under a restart's
own sentence:

```
error: --init-state: this run<cell label> changes parameter 'beta', which the
model reads outside any intervention (in its dynamics, initial state, forcings
or observation models) from the start of the run, before the forecast origin
t = <T> (0.31 in <path>, 0.35 here). The saved states are a sample from
p(x_T | y, θ) at the file's θ and interventions; a run that differs before the
origin describes a history the state did not have.
  Fix: run at the file's values (--params) and scenario (<label>), or re-run the
  filter under this run's.
```

A difference that first acts at or after `T` is allowed and said once, naming
what differs and when it first acts.

### 7.3 Periods, against the file

For each emitted interval stream the run writes, `simulate` calls
`stream_periods` (§4) with origin `T`, the file's `rows` for the stream, the
stream's schedule under this run's `--emit-every`, and reach the cell's run end.
A refusal there, D82's included, refuses the restart. The first period ending
after `T` must then match the file's `open` line for the stream when it starts
before `T`, or both must be absent:

```
error: --init-state: stream '<s>' writes the period [<a'>, <b'>) across the
origin t = <T>, but <path> carries its running total from t = <a> (period
[<a>, <b>)). The periods differ from the ones the filter was run for
(--emit-every?), so the part of [<a'>, <T>) before the origin is recorded
nowhere.
  Fix: drop --emit-every for '<s>', or choose a step with a period boundary at
  t = <T>.
```

When the file has an `open` line the run does not need (the run's periods put a
boundary at `T`), the carry is unused and nothing is said.

### 7.4 Writing the rows

Under a restart, each emitted interval stream's rows are the periods
`stream_periods` returned, every one ending after `T`. Each row's label is the
emit time whose period it is: for a uniform form the time `l` with
`Covers::period_of(l)` (`rust/crates/ir/src/observation.rs:304`) equal to the
period, for window columns the period's end. Under D82 agreement every period is
one emit time's, so a period with none is an internal error. Instant streams
keep today's rule: a row at each emit time at or after `T`. Every site that
plans a restarted cell's rows reads this one plan: the `--obs` writers
(`main.rs:2838`, `:3701`, `:3879`) and the observation-sourced `quantities`
series (`materialize_obs_for_quantities`, `main.rs:2702`), so a quantity reduces
over the rows `--obs` writes.

Each interval row is one of:

```rust
enum RowSpan {
    /// Inside the restarted run: projected from the trajectory, as today.
    After { start: f64, stop: f64 },
    /// Began before the origin: `carry` (the particle's `F_s[a, T)`) plus the
    /// trajectory's flow over `[T, stop)`.
    Straddles { start: f64, origin: f64, stop: f64 },
}
```

`project_coverages` (`main.rs:4492`) projects a `Straddles` row over `[T, b)`
and adds the cell's carry: for an incidence stream the sum, for a ratio of flows
`(num_carry + num[T, b)) / (den_carry + den[T, b))`, NaN when the denominator is
zero (as `flow_ratio` does today). The observation model then draws at `b` from
that projection. The emitted row carries the true period `[a, b)` in its
`t_start`/`t_stop` columns. `OriginState` (`rust/crates/cli/src/sim_job.rs:100`)
gains `carry: Vec<u64>`, indexed like the file's carry columns, with the stream
mapping held once on `InitStateSource` (`:134`).

At a restart on a period boundary there is no straddling row. For a stream whose
data and schedule agree, every emitted row is then the one written today (§10,
test 3).

## 8. `--init-state fit` (PGAS posterior draws)

**The saved paths contain enough.** Verified (read): PGAS writes its latent
paths at `Granularity::Substep` (`rust/crates/cli/src/fit/pgas.rs:1209`,
`:1221`), one row per step with each transition's `flow_*` count over
`[t_start, t_stop)` (`rust/crates/io/src/trajectories.rs:229-234`).
`resolve_forecast_ensemble` (`rust/crates/cli/src/fit/joint.rs:215`) reads only
the terminal snapshot of each path (`read_terminal_states`,
`io/src/trajectories.rs:599`, called at `joint.rs:251`). For any `a` on the `dt`
grid, `F_s[a, T)` on draw `i` is the sum of the stream's transitions' `flow_*`
over the path rows with `t_start ≥ a`. It belongs to the same path as that
draw's `X_i(T)`, so `(θ_i, X_i(T), F_s[a, T))` is one joint posterior draw.

**Periods.** The fit's training rows give `stream_periods` its data, read and
digest-checked as `fit predict` reads them for a forecast (the fit record's
`TrainingPartition`), with origin `T` the paths' terminal time. The fit source
has no file, so these are the inputs §7.3 and §7.4 use.

**The design.** `read_terminal_states` takes the distinct period starts the run
needs and returns, per `(chain, draw)`, each transition's flow total since each
start, in the same single pass. `resolve_forecast_ensemble` turns them into
per-stream carries with the streams' `flow_indices`, so the fit source and the
file source hand the engine the same `OriginState`. An `a` that is not a step of
the saved paths (off the `dt` grid) is refused:

```
error: --init-state fit: stream '<s>': the period under way at the origin
t = <T> opens at t = <a>, which is not a step of the saved latent paths
(dt = <dt>), so the flow before the origin cannot be summed.
  Fix: declare the stream's periods so that they start on the dt grid.
```

The fit source's parameters are each draw's own, so the comparison of §7.2 takes
the fit's model under the fit's own intervention selection (`util::FitBase`,
applied at `main.rs:1815`) as the baseline, as `fit predict`'s forecast arms do
(`forecast_arms.rs:95-112`). The model check reads the trajectories header's
model hash, computed by `model_identity_from_ir` (`pgas.rs:653`), and names the
fit directory.

## 9. Run identity

**Today.** Verified (read). The file source keys on the SHA-256 of the file's
bytes (`main.rs:1547`) through `InitStateDigest { ensemble, row }`
(`runid/src/inputs.rs:104`); the fit source on
`InitStateEnsemble { origin_t, rows: Vec<InitStateRow { chain, draw, counts, reals }> }`
(`:117-144`). `HASH_VERSION` is 8 (`runid/src/hash.rs:171`).

**v2.** Both sources hash one structure built from parsed values (Decision 3):

```rust
pub struct InitStateEnsemble {
    pub origin_t: FiniteF64,
    pub periods: Vec<StreamPeriodsId>,    // per emitted interval stream: its data periods
    pub open: Vec<OpenPeriodId>,          // stream, start, stop
    pub rows: Vec<InitStateRow>,
}
pub struct InitStateRow {
    pub key: InitStateKey,                // Particle(i) | Draw { chain, draw }
    pub counts: Vec<i64>,
    pub reals: Vec<FiniteF64>,
    pub carry: Vec<u64>,
}
```

The carry is identity: two ensembles with equal counts and different carries
write different straddling rows, so leaving it out would serve one run's output
for the other. The data periods are identity for the same reason: they fix the
phase of every row the restart writes. The provenance lines are not in this
digest. Each is checked against something the cell already keys on (the model
level, the params level, the scenario level, `dt` in the config level) or is
display (`source`, which `run.json` records unhashed as the restart's source).

**Version.** This changes stored bytes for unchanged identity inputs: a
fit-source restart whose origin is inside a period now writes a row it did not
write before, under the `InitStateEnsemble` today's code would hash. Under D58
the implementing change bumps `HASH_VERSION` from 8 to 9, with a doc-comment
entry naming the changes (the new row, the carry and data periods in the key).
File-source runs cannot alias old ones, since v1 files are refused.

**The pfilter leaf.** Unchanged, as is its key: an inserted stop changes no byte
the leaf stores (§5).

**`ir/VERSION`.** Unchanged: nothing here touches the IR.

## 10. Tests

Fixture F7 is committed under `rust/crates/cli/tests/fixtures/`: F2's weekly
`closing_at(time, 7 'days)` stream and daily `day(time)` stream, with
`emit_schedule = every 7 'days` on the weekly stream (weeks closing 7, 14, …),
weekly data closing 7, 14, 21 and daily data through day 25, so `T = 26` and the
week `[21, 28)` is under way with `a = 21` already a stop. A sparse twin
replaces the daily stream with a `closing_at(time, 5 'days)` stream observed
through day 25, so `T = 25` and `a = 21` is inserted. A window-column twin, a
`starting_on` twin and a ratio-of-flows twin share the dynamics.

1. **Distribution.** `pfilter --save-final-state` on F7 at fixed θ with data
   whose values are all `NA` (the filter is then the prior; the end-to-end D72
   test binds an all-`NA` stream, so the binder accepts one), then
   `simulate --init-state --to 49 --obs`, 20,000 replicates. The week-28 row
   exists, its period is `[21, 28)`, and its mean and variance match a
   continuous `simulate` from `t_start` over 20,000 replicates within 4 Monte
   Carlo standard errors. Repeated with F7's data against the
   importance-sampling reference of spec §4.4 (prior draws weighted by the daily
   likelihood). Red on main: the row is absent.
2. **Splice exactness.** Through the test-only RNG-restore seam
   (`StartState.rng`, `chain_binomial.rs:21-30`): a continuous run to 49, and a
   head run to 26 whose final state, RNG and `F[21, 26)` seed a resumed tail.
   The tail's week-28 row equals the continuous run's byte for byte, for the
   uniform, `starting_on`, window-column and ratio-of-flows twins. Mutation
   check: dropping the carry term turns it red.
3. **Boundary restart is unchanged.** On main, record `simulate --init-state`'s
   trajectory and `--obs` files for a restart at `T = 21`, from F7 with its data
   cut to end at day 21 so that `T` is a boundary of every emitted period, from
   the current v1 file; commit them as expected output. After the change,
   regenerating the state as v2 and restarting writes byte-identical files, and
   the v2 file has no `open` line.
4. **An inserted stop changes no filter output it should not.** On the sparse
   twin, `pfilter` with and without `--save-final-state` prints a bit-identical
   log-likelihood, stores the same pfilter `run_id`, and saves identical
   compartment columns; with it, `--trace` has one more row, at 21, and
   `pfilter` names the inserted time.
5. **Edited file refused.** `t=26` changed to `t=3`; one count changed; one
   carry changed; one `rows` period changed; the checksum line deleted: each
   refused with §7.1's message, asserted verbatim up to the hashes.
6. **Mismatched provenance refused.** `--param beta=<other>` on a parameter the
   dynamics read (names `beta` and both values); a sweep cell with a different
   value (names the cell); another model; another `--dt`; a scenario enabling an
   intervention that fires before `T` (refused), and one that first fires after
   `T` (allowed, and said).
7. **D82.** F6 as gh#1081 describes it (data closing 11, 18, 25 against weeks
   closing 7, 14, 21): `pfilter --save-final-state` prints the note and writes
   no `open` line for the weekly stream; `simulate --init-state --obs` is
   refused naming the stream, `[4, 11)` (training row 0) and `[0, 7)` (the emit
   schedule); `simulate --init-state` without `--obs` runs. A unit test asserts
   that the restart's and the forecast's `ScheduleConflict` text differ only in
   the consumer.
8. **`--emit-every` across the origin.** On the window-column twin, a step that
   moves the period under way is refused with §7.3's message; one that puts a
   boundary at `T` runs. On the uniform stream, a step that disagrees with the
   data's tiling is the D82 refusal.
9. **One period rule.** For F7 and each twin, the restart's planned periods
   equal the `ForecastPlan` targets of the same stream from the same rows and
   origin over the same reach.
10. **v1 refused** with §7.1's message.
11. **Fit source.** A PGAS fit on F7: the restarted week-28 row of draw `i`
    equals the sum of its saved path's flows over `[21, 26)` plus the restarted
    run's flow over `[26, 28)`, before observation noise (read from the
    trajectory, not the obs file). An off-grid start is refused with §8's
    message.
12. **Identity.** `InitStateEnsemble` hashes differ for two ensembles equal but
    for one carry value, and for two equal but for one data period; the
    `resolve` identity matrix (`rust/crates/cli/src/resolve/tests.rs:418`) gains
    an `init_state_carry` row.
13. **Backend matrix.** No cell changes: `--init-state` remains chain-binomial
    only, and the existing refusals for Gillespie, ODE and reactive models
    (`main.rs:1479-1503`) keep their tests.

Existing tests that write v1 files by hand (`gh641_init_state_forecast.rs`,
`gh1084_pfilter_final_state_holdout.rs`, `simulate_input_classes.rs`,
`runid/src/inputs/tests.rs`) produce their files through `pfilter` or the
library writer instead; only the refusal tests in 5 and 10 write a file by hand.

## 11. Decisions for the maintainer

**Decision 1. Emitted streams the filter does not bind.** A model stream with an
`emit_schedule` but no `--data` binding at `pfilter` time is still written by
`simulate --obs`. It has no rows, so a uniform stream has no phase for
`stream_periods` (`StreamRefusal::NoBoundRows`), while `plan_emission` takes its
phase from the emit labels. (a) Give `stream_periods` a fourth source for a
uniform stream with no rows: the periods of its emit labels
(`Covers::period_of`); carry slots then depend only on the model's transitions
and these periods, and forecasts are unaffected because a forecast reads only
bound streams (D76). (b) Refuse a restart that writes such a stream across a
period under way, and write it unchanged at a boundary. Recommendation: (a),
leaning; it costs one integer per stream per particle, and (b) would refuse a
row the filter could have completed.

**Decision 2. Scenario and parameter differences at restart.** The state at `T`
reflects the interventions and parameters in force before `T`. (a) Refuse any
difference. (b) Apply the forecast arms' rule (D50) through the same function:
allow a difference that first acts at or after `T`, refuse the rest (§7.2). (c)
Warn and proceed. Recommendation: (b), solid; it is the rule `fit predict`
already enforces for the same situation, and (a) forbids the main use of a
restart, comparing interventions that start after now.

**Decision 3. What the file source's identity hashes.** (a) Keep the SHA-256 of
the file's bytes: simplest, but a display-only field (`source`) and formatting
enter the key. (b) Hash the parsed ensemble, one structure for both sources
(§9): formatting is presentation, and the two sources cannot drift. Either way
`HASH_VERSION` is bumped once. Recommendation: (b), solid.

## 12. What changes in the runbook and the spec

Not edited in this PR; the implementing PR updates them to point here.

Runbook, `docs/dev/proposals/2026-11-15-temporal-architecture-migration.md`:

- Stage 5, "The fixed-origin driver" bullet, its last sentence (lines 577-578):
  "No change to `particle_filter.rs`, resampling or the saved-state format." The
  saved-state format changes (here); `particle_filter.rs` and resampling still
  do not.
- Stage 5, "Restarts stay conservative." bullet (lines 604-607): the refusal of
  periods that began before the restart (T3, gh#1081; R2, gh#1085) is replaced
  by this proposal's exact restart. The `holdout_after` (R1) and PMMH-message
  (R3) clauses stand.
- Stage 5 tests, "Restarts." bullet (lines 659-663): "across a period under way
  is refused, for a uniform stream and a window-column stream" becomes §10 tests
  1, 2 and 7 here.
- Stage 5, "Intentional changes" (lines 680-682): "restarts across a period
  under way are refused" is removed.
- Part 6, fixture F6 (lines 836-838): its data and schedule disagree under D82;
  it becomes the refusal fixture of §10 test 7, beside F7.
- Part 7, "High-risk files" (lines 857-860): "changes neither the filter nor its
  saved-state format": the format changes here, not in Stage 5.

Spec, `docs/dev/temporal-architecture.md`:

- §4.4, "Restarts." paragraph (lines 672-675): restores counts only, refuses a
  period begun before the restart → restores counts and running totals, and
  completes the period, whose bounds come from §4.3's rule.
- §5.2 invariant table, row "A restart never emits a period that began before
  it" (line 806) → "A restart's straddling row equals the pre-origin running
  total plus the post-origin flow", enforced by `simulate --init-state`, checked
  (§7.3's match and §10 test 2).
- Part 7 table, row "Restarts that continue a period under way" (line 927): no
  longer deferred.

User docs, `docs/inference.md`, "What a restart does and does not carry": the
implementing PR rewrites it for v2.

## 13. Verification record

Commands run at `6dce561c9` (a `git archive` of `origin/main`), output
summarized.

- `grep -n "pub const FORMAT_TAG\|^fn parse_origin_t\|^pub fn read_final_states\|^pub fn write_final_states" rust/crates/io/src/final_state.rs`
  → `:65`, `:118`, `:176`, `:299`; `parse_origin_t` checks only finiteness.
- `sed -n 1,840p rust/crates/cli/src/fit/forecast_plan.rs` → `stream_targets` at
  `:470`, the period match `:482-544`, the D82 call `:547-549`,
  `check_schedule_agrees` `:679-720`, `ScheduleConflict` `:191-198` and its text
  `:276-282`; the period code reads only `measures`, `position` and the held-out
  row's file.
- `sed -n 515,870p rust/crates/sim/src/inference/particle_filter.rs` → fold
  `:530`, `assimilates` `:570`, log-weight 0 at a stop with no observation
  `:574-582`, collapse window only at assimilation `:751-756`, resample only at
  an assimilation with unequal weights `:771`, `acc` copied on resample `:780`,
  reset `:825-826`, `final_states` `:866`.
- `sed -n 1370,1480p rust/crates/sim/src/inference/multi_stream_obs.rs` → union
  axis = closes plus declared starts inside the run; reset schedule from
  declared starts only.
- `sed -n 340,372p rust/crates/sim/tests/assimilate_only_at_observations.rs` and
  `sed -n 1,20p rust/crates/cli/tests/pfilter_assimilates_only_at_observations.rs`
  → the D72 invariance tests and what they compare.
- `sed -n 760,845p rust/crates/cli/src/pfilter.rs` → the pfilter leaf stores
  `loglik.toml` and its inputs record, nothing per stop.
- `sed -n 167,246p rust/crates/cli/src/obs_emit.rs` and
  `sed -n 4255,4300p rust/crates/cli/src/main.rs` → the per-form behavior in
  §2's table.
- `sed -n 1460,1560p rust/crates/cli/src/main.rs` → no provenance checks; the
  file-source digest is `digest_bytes` of the file.
- `sed -n 80,290p rust/crates/cli/src/fit/forecast_arms.rs` → `first_difference`
  and the D50 rule.
- `grep -n "pub const HASH_VERSION" rust/crates/runid/src/hash.rs` → `8`.
- `gh issue view 1081`, `gh issue view 1085` → the reproduction (run) of gh#1081
  and the inferred gh#1085 cited in §2.

Not run for this proposal: every test in §10, the D82 refusal of F6 (inferred
from reading `check_schedule_agrees`), and the gh#1085 window-column truncation,
which its issue records as inferred.
