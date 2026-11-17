# Proposal: exact restarts from a filtered state

**Status:** Proposal **Date:** 2026-11-16 **Motivation:**
`camdl simulate
--init-state` is kept permanently as a lower-level forecasting
tool (maintainer decision D64, option c). Today it drops or truncates the
reporting period under way at the restart (gh#1081, gh#1085), trusts an edited
header, and cannot tell whether it is restarting under the model, parameters and
scenario the state was filtered under. This proposal makes a restart exact and
checked: saved-state format v2 carries each emitted stream's running total for
the period under way and a provenance block sealed by a checksum, and
`simulate --init-state` completes the straddling period and refuses a file that
does not match the run.

**Supersedes:** the restart half of D39 in
`docs/dev/proposals/2026-11-15-temporal-architecture-migration.md` (the
runbook), which planned to refuse restarts across a period under way and keep
the saved-state format unchanged. §10 lists the passages that change.

**Background assumed.** Observation periods and `covers` forms as in
`docs/dev/temporal-architecture.md` §2.2 (the spec); the user-facing workflow in
`docs/inference.md`, "Forecasting from the filtered state". Everything else this
proposal relies on is cited below.

Code references are to `main` at `f8e02d162`. _Verified (read)_ means read in
the code at that commit; _verified (run)_ means reproduced by running binaries
built from it, recorded in the cited issue; _inferred_ means neither. Commands
behind the verified claims are in §12.

## 1. Terms

- **Forecast origin `T`.** The model time the saved states are at: the particle
  filter's last observation time on the union grid (the sorted merge of every
  stream's period ends and declared period starts).
- **Emitted stream.** An observation stream with an `emit_schedule`, so
  `simulate --obs` writes rows for it. A fit-only stream (no `emit_schedule`)
  writes nothing after a restart and needs nothing carried.
- **Period under way at `T`.** For an emitted interval stream (one whose
  projection accumulates a flow, `incidence(...)` or a flow ratio), the emission
  period `[a, b)` with `a < T < b`. It exists only when `T` falls strictly
  inside one of the stream's emission periods; at a period boundary there is
  none.
- **Running total `F_s[a, T)`.** The flow stream `s` projects (the sum of its
  transitions' counts) accumulated by one particle over `[a, T)`. For a flow
  ratio it is two totals, numerator and denominator.
- **Straddling row.** The emitted row whose period is the period under way: it
  began before the restart and closes after it.

## 2. What happens today

**The file holds counts and an origin, nothing else.** Verified (read):
`write_final_states` (`rust/crates/io/src/final_state.rs:118`) writes the header
`# camdl-final-state v1<TAB>t=<T>` (`FORMAT_TAG`, `:65`), a `particle` column
and one column per integer compartment. `read_final_states` (`:176`) resolves
compartments by name and takes `t` from `parse_origin_t` (`:299`), which checks
only that it is a finite number. `simulate` reads the file at
`rust/crates/cli/src/main.rs:1490-1540` and checks nothing about where it came
from except the `--draws` and replicate-count refusals there. The run's
`t_start` becomes `T` at `rust/crates/cli/src/util.rs:3800-3810`.

**Defect 1, the first forecast period is lost.** Verified (read), and reproduced
(run) in gh#1081 and gh#1085. A restarted run computes its emission times with
`obs_schedule_times` (`main.rs:4218`), which drops every _label_ before `T`
(`:4223-4224`). `plan_emission` (`rust/crates/cli/src/obs_emit.rs:167`) then
assigns periods:

| `covers` form                | Today, restart at `T = 26`, weekly rows closing 7, 14, 21, 28, …                                                                                                       |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `closing_at(time, 7 'days)`  | label 28 survives the label filter, its period `[21, 28)` starts before the run, and `within_run` (`obs_emit.rs:214`) drops it silently: output begins at 35 (gh#1081) |
| `starting_on(time, 7 'days)` | label 21 (period `[21, 28)`) is dropped by the label filter before planning                                                                                            |
| `ending_on`, `day`           | as `closing_at` or `starting_on`, by where the label sits in its period                                                                                                |
| `window_start`/`window_stop` | the first window opens at the run start (`prev = t_start`, `obs_emit.rs:229`), so row 28 is written as the partial `[26, 28)` under the full row's label (gh#1085)     |

The cause is the same in every row: the file has no record of what each particle
accumulated over `[a, T)`, so no row can be completed.

**Defect 2, the header is trusted.** Verified (read): changing `t=26` to `t=3`
yields a valid file; nothing ties `t` to the states.

**Defect 3, no provenance.** Verified (read): the file records no model,
parameter vector, scenario or `dt`; restarting under different ones is silent.

**Kept as is.** Interventions that fired before `T` are not replayed on restart,
and a `T` off the output grid is refused
(`rust/crates/sim/src/chain_binomial.rs:463-470`). Reactive interventions and
non-chain-binomial backends are refused (`main.rs:1466-1488`).

## 3. Why counts are not enough, and what is

A straddling row's value is the stream's flow over `[a, b)`, which splits at the
origin:

`F_s[a, b) = F_s[a, T) + F_s[T, b)`.

The second term is produced by the restarted run. The first is a property of
each particle's _path_ before `T`, not of its state at `T`: two particles with
identical counts at `T` can have counted different numbers of infections since
`a`. So the saved row must carry `F_s[a, T)`, and the pair `(x_T, F_s[a, T))`
must come from the same particle. The bootstrap filter gives exactly that: a
quantity accumulated along each particle's path travels with the particle
through resampling, so after the final resample `(x_T, F_s[a, T))` is a draw
from the filtering distribution `p(x_T, F_s[a, T) | y_{1:T}, θ)`. The spec's
experiment (`temporal-architecture.md` §4.4) checked this for the forecast
driver: a running total opened at the right start reproduced an
importance-sampling reference (68.245 ± 0.021 against 68.227 ± 0.034).

The pre-origin part is latent and conditioned on the data through `T`; it is not
known. The stream's own row for `[a, b)` cannot have been assimilated, because
`b > T` and `T` is the last observation.

## 4. Where the filter's totals are, and why the existing ones are not used

The filter already carries a per-stream running total. Verified (read):
`ParticleState.acc` (`rust/crates/sim/src/inference/types.rs:621-632`) holds one
bin per incidence stream and two per flow-ratio stream, laid out by
`MultiStreamObsModel`'s `interval_slots` (`multi_stream_obs.rs:1806-1830`). It
is folded once per union interval (`particle_filter.rs:427-429`, via
`fold_into_acc`, `multi_stream_obs.rs:1907`), copied with the particle on
resampling (`particle_filter.rs:609`), and reset per stream only where that
stream's reset schedule fires (`reset_due_acc`, `multi_stream_obs.rs:1946`;
called at `particle_filter.rs:650`). The filter returns the swarm after the
final reset as `final_states` (`particle_filter.rs:689`), and that is what the
writer receives (`pfilter.rs:1011-1029`). So `acc` reaches the writer today and
is discarded there.

**`acc` cannot be written as it is.** A stream's reset schedule is built from
its _bound data_: the bin resets at each declared period start inside the run
(`multi_stream_obs.rs:1332-1360`). After the stream's last data row, no start is
declared, so the bin keeps accumulating across the boundary where the emission
period under way begins. In the gh#1081 fixture (weekly data closing 11, 18, 25;
emission weeks closing 7, 14, 21, 28) the bin at 25 holds `[18, 25)`, while the
straddling emission row needs `[21, 25)`. The spec measured the same hazard for
the forecast driver: undeclared, the total was 256.951 against the correct 68.2
(§4.4). Moving the scoring bin's reset would change what the likelihood scores
whenever the emission tiling differs from the data's.

**The design: carry slots.** The observation model gains a second, separate set
of bins, one per emitted interval stream with a period under way at `T` (two for
a flow ratio), appended after the scoring bins in the same `acc` vector:

```rust
/// A running total kept for a restart, never scored.
pub struct CarrySlot {
    /// The stream leaf this total belongs to (an indexed block's leaf name).
    pub stream: String,
    /// Which side of a flow ratio, or `Total` for an incidence stream.
    pub part: CarryPart,
    /// Transitions whose counts the total sums (the stream's projection).
    pub flow_indices: Vec<usize>,
    /// Union index at which the total opens (is zeroed): the period start `a`.
    pub opens_at_union: usize,
}
pub enum CarryPart { Total, Numerator, Denominator }

impl BoundObs {
    /// Carry slots for `--save-final-state`. Adds each `a` to the union axis
    /// when it is not already a union time. Built only by `pfilter`.
    pub fn with_carry(self, carry: Vec<CarrySpec>) -> Result<(Self, Vec<f64>), Finding>;
}
```

`n_interval_streams()` counts scoring and carry bins together, so every
`ParticleState` is sized for both and the existing whole-vector copy on
resampling (`particle_filter.rs:609`) carries them. `fold_into_acc` and
`reset_due_acc` iterate both kinds. Scoring reads bins through `stream_bins`
(`project_stream_from_acc`, `multi_stream_obs.rs:1979`), which never points at a
carry bin, so scoring is unchanged. **`particle_filter.rs` is not edited.**
Carry slots are built only by `pfilter --save-final-state`; every other binder
caller passes none, so `acc` and every inference path are unchanged for them
(inferred from the call-site list in §12; test 4 in §9 pins the pfilter case).

**The period start `a` must be a stop of the filter.** The fold happens once per
union interval, so the total can only open at a union time. When `a` is already
one (another stream's period end, a declared start, or `T` itself), nothing
about the filter's stepping changes and its log-likelihood is byte-identical
with and without `--save-final-state`. When it is not, `with_carry` inserts `a`
into the union axis. An inserted stop carries no observation; the binder already
creates such stops for declared period starts and documents their only cost as
one resampling draw (`multi_stream_obs.rs:1276-1289`). The filter's
log-likelihood is then a different realisation of the same estimator, so the
pfilter leaf records the inserted times in its identity (§8) and `pfilter` says
so (Decision 1). An `a` off the `dt` grid is refused before filtering:

```
error: --save-final-state: stream '<s>': the period under way at the last
observation t = <T> opens at t = <a>, which is not on the dt = <dt> grid, so the
filter cannot stop there to start its running total.
  Fix: give the stream an emit_schedule whose periods start on the dt grid, or
  filter with a dt that divides <a>.
```

**Which `a`, per form.** `a` comes from the stream's _emission_ plan, the same
one the restarted run writes, so the two cannot disagree. A new function in
`obs_emit.rs` owns it:

```rust
/// The emission period of `obs` that contains `t` strictly inside it, or `None`
/// when `t` is a period boundary or lies in a gap between periods.
pub(crate) fn period_under_way(
    obs: &ObservationModel, schedule: &ObservationSchedule,
    run_start: f64, t: f64,
) -> Result<Option<Period>, String>;
```

| Form                                            | Periods                                                                      | Period under way at `T`                                                   |
| ----------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `closing_at`, `ending_on`, `day`, `starting_on` | `Covers::period_of(label)` (`ir/src/observation.rs:304`) for each emit label | the one with `start < T < stop`; none at a boundary or in a gap           |
| `window_start`/`window_stop`                    | `[previous emit time, emit time)`, the first opening at `run_start`          | `[last emit time ≤ T, first emit time > T)`, when `T` is not an emit time |
| an instant stream (`measures` a state)          | no period                                                                    | none: the value is read at the row's time, after `T`                      |

A stream whose likelihood reads a state at the period end (a binomial whose `n`
is a compartment, or an instant read) needs nothing carried: the end `b` is
after `T`, so the restarted trajectory has it. Data-column reads after `T` are
already refused by `check_data_columns_supplied` (`main.rs:2683`, and
`main.rs:1566` for `--obs`).

## 5. Saved-state format v2

```text
# camdl-final-state v2
# t	26
# dt	1
# model	4be1…<64 hex>
# param	beta	0.31
# param	gamma	0.1
# param	rho	0.4
# scenario	baseline
# intervention	school_closure	off
# intervention	vaccination	on
# source	pfilter	9a1e…<64 hex>
# open	weekly	21	28
# open	hosp_ratio	0	30
# checksum	sha256	<64 hex>
particle	S	E	I	R	carry:weekly	carry:hosp_ratio:numerator	carry:hosp_ratio:denominator
0	9812	14	31	143	57	12	40
1	9805	19	28	148	61	9	38
```

Header lines are `# <key><TAB><fields…>`, in this order, each present exactly
once except `param`, `intervention` and `open` (zero or more, sorted by name):

- `t`: the forecast origin, Rust's shortest round-trip `f64` form.
- `dt`: the filter's step.
- `model`: `runid::inputs::model_ir_hash` of the compiled IR as loaded by
  `util::load_model`, before scenario or parameter resolution — the same
  function the `model` identity level and the trajectories header use
  (`runid/src/inputs.rs:316`, `resolve.rs:131-139`).
- `param`: every parameter value the filter ran at, after scenario `set`/`scale`
  and `--param`/`--params` resolution (`pfilter.rs:343-360`), round-trip form.
  Written as values, not a hash, so a refusal can name the parameter that
  differs.
- `scenario`: the scenario label; `intervention`: every non-reactive
  intervention with its effective on/off state under that scenario.
- `source`: the command and the `run_id` of the pfilter leaf
  (`pfilter_cas::resolve_pfilter`), computed even under `--no-store`. Display
  only; nothing checks it.
- `open`: one line per stream with a period under way: leaf name, `a`, `b`.
- `checksum`: SHA-256 (`runid::ContentHash::digest_bytes`) of the file's bytes
  with this one line removed.

Columns: `particle`, every integer compartment in model order (as v1), then one
`carry:<stream>` column per incidence stream with an `open` line, or
`carry:<stream>:numerator` and `carry:<stream>:denominator` for a flow ratio, in
`open` order. Carry values are non-negative integers. Compartment names are
identifiers, so a `carry:` column cannot collide with one. A stream at a period
boundary has no `open` line and no column.

The reader parses into one type, so nothing downstream re-checks:

```rust
pub struct FinalStates {
    pub origin_t: f64,
    pub provenance: Provenance,          // dt, model hash, params, scenario, source
    pub open: Vec<OpenPeriod>,           // stream, start a, stop b; a < origin_t < b
    pub counts: Vec<Vec<i64>>,           // [particle][integer compartment]
    pub carry: Vec<Vec<u64>>,            // [particle][carry column, `open` order]
}
```

## 6. Checks at `simulate --init-state <file>`

### 6.1 Integrity and version, once per file, in `read_final_states`

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
`a < t < b` on every `open` line and a carry column for each.

This makes defect 2 a refusal: editing `t=26` to `t=3` changes the bytes.

### 6.2 Provenance, against the run

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

Per cell, because a sweep or scenario arm resolves its own parameters:

```
error: --init-state: <path> was filtered at different parameter values from
this run's<cell label>: beta = 0.31 in the file, 0.35 here; rho is in the file
but not in this run. The saved states are a sample from p(x_T | y, θ) at the
file's θ; restarting them under another θ forecasts a state and parameters that
never went together.
  Fix: run at the file's values (--params), or re-run the filter at the new ones.
```

Equality is exact on the `f64` bits; the file's values are round-trip exact.

Scenario, per cell (Decision 2): an intervention whose on/off state differs is
allowed only when its first fire time is at or after `T`, read from the cell's
effect plan. Otherwise:

```
error: --init-state: this run's scenario differs from the one <path> was
filtered under in intervention(s) that act before the forecast origin
t = <T>: school_closure (first fires at t = 14; off in the file, on here).
The saved state already reflects the file's interventions up to t = <T>;
changing one that acted earlier describes a history the state did not have.
  Fix: run under the file's scenario (<label>), or re-run the filter under
  this one. Interventions that first fire at or after t = <T> may differ.
```

An allowed difference is said once:
`simulate: --init-state: the scenario
differs from the filtered one only after the origin: vaccination (first fires at
t = 30).`

### 6.3 Open periods, against the run's emission plan

For each emitted stream, `simulate` computes `period_under_way` under its own
emission settings (including `--emit-every`, which can change the periods). The
result must match the file's `open` line for that stream, or both must be
absent:

```
error: --init-state: stream '<s>' writes the period [<a'>, <b'>) across the
origin t = <T>, but <path> carries its running total from t = <a> (period
[<a>, <b>)). The emission periods differ from the ones the filter was run for
(--emit-every?), so the part of [<a'>, <T>) before the origin is recorded
nowhere.
  Fix: drop --emit-every for '<s>', or choose a step with a period boundary at
  t = <T>.
```

When the file has an `open` line the run does not need (the run's periods put a
boundary at `T`), the carry is unused and nothing is said.

### 6.4 Writing the straddling row

The restart filter moves from labels to periods. `obs_schedule_times` stops
filtering by `restart_origin`; `plan_emission` takes the origin and keeps a row
when its period ends after `T` (an instant row when its time is at or after `T`,
as today). Each interval row is one of:

```rust
enum RowSpan {
    /// Inside the restarted run: projected from the trajectory, as today.
    After { start: f64, stop: f64 },
    /// Began before the origin: `carry` (the particle's `F_s[a, T)`) plus the
    /// trajectory's flow over `[T, stop)`.
    Straddles { start: f64, origin: f64, stop: f64 },
}
```

`project_coverages` (`main.rs:4431`) projects a `Straddles` row over `[T, b)`
and adds the cell's carry: for an incidence stream the sum, for a flow ratio
`(num_carry + num[T, b)) / (den_carry + den[T, b))`, NaN when the denominator is
zero (as `flow_ratio` does today). The observation model then draws at `b` from
that projection. The emitted row carries the true period `[a, b)` in its
`t_start`/`t_stop` columns. The obs-sourced `quantities` series
(`materialize_obs_for_quantities`, `main.rs:2665`) goes through the same plan
and projection, so it reduces over the same rows. `OriginState`
(`cli/src/sim_job.rs:92`) gains `carry: Vec<u64>`, indexed like the file's carry
columns, with the stream mapping held once on `InitStateSource`.

At a restart on a period boundary there is no straddling row, and every emitted
row is the one written today (§9, test 3).

## 7. `--init-state fit` (PGAS posterior draws)

**The saved paths contain enough.** Verified (read): PGAS writes its latent
paths at `Granularity::Substep` (`cli/src/fit/pgas.rs:1215`), one row per step
with each transition's `flow_*` count over `[t_start, t_stop)`
(`io/src/trajectories.rs:229-240`). `resolve_forecast_ensemble`
(`cli/src/fit/joint.rs:215`) reads only the terminal snapshot of each path
(`read_terminal_states`, `io/src/trajectories.rs:599`). For any `a` on the `dt`
grid, `F_s[a, T)` on draw `i` is the sum of the stream's transitions' `flow_*`
over the path rows with `t_start ≥ a`. It belongs to the same path as that
draw's `X_i(T)`, so `(θ_i, X_i(T), F_s[a, T))` is one joint posterior draw.

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
  Fix: give the stream an emit_schedule whose periods start on the dt grid.
```

The fit source's parameters are each draw's own, so §6.2's parameter check does
not apply. The model check does: the trajectories header's `model=` hash
(`pgas.rs:659`) must equal the simulated model's, with §6.2's model message
naming the fit directory. Scenario differences follow §6.2's rule against the
fit's own scenario, which `simulate` already applies to each arm (the `fit_base`
handling around `main.rs:1807`; inferred that it exposes the effective
intervention set).

## 8. Run identity

**Today.** The file source keys on the SHA-256 of the file's bytes
(`main.rs:1530-1531`) through `InitStateDigest { ensemble, row }`
(`runid/src/inputs.rs:104`); the fit source on
`InitStateEnsemble { origin_t,
rows: Vec<InitStateRow { chain, draw, counts, reals }> }`
(`:117-144`).

**v2.** Both sources hash one structure built from parsed values (Decision 3):

```rust
pub struct InitStateEnsemble {
    pub origin_t: FiniteF64,
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
for the other (the count-in-the-key rule). The provenance lines are not in this
digest. Each is either checked equal to something the cell already keys on (the
model level, the params level, the scenario level, `dt` in the config level) or
is display (`source`, which `run.json` records unhashed as the restart's
source).

**Version.** This changes stored bytes for unchanged identity inputs: a
fit-source restart whose origin is inside a period now writes a row it did not
write before, under the same `InitStateEnsemble` today's code would hash. Under
D58 the implementing change bumps `HASH_VERSION` (`runid/src/hash.rs:118`, now
5) by one, with a doc-comment entry naming both changes (the new row, the carry
in the key). File-source runs cannot alias old ones, since v1 files are refused.

**The pfilter leaf.** `PfilterConfigLevel` (`cli/src/pfilter_cas.rs:79`) gains
`inserted_stops: Vec<f64>`, skipped when empty in the way its
`allow_forcing_extrapolation` field is (`:89-92`). A pfilter whose carry starts
are already union times, and every pfilter without `--save-final-state`, keeps
its key; one with an inserted stop, whose log-likelihood is a different
realisation, re-keys.

**`ir/VERSION`.** Unchanged: nothing here touches the IR.

## 9. Tests

Fixture F6 (runbook Part 6) is committed under
`rust/crates/cli/tests/fixtures/`: F2's weekly `closing_at(time, 7 'days)`
stream and daily `day(time)` stream, weekly `emit_schedule = every 7 'days`
(emission weeks closing 7, 14, …), data at 11, 18, 25. A window-column twin and
a `starting_on` twin share its dynamics.

1. **gh#1081, distribution.** `pfilter --save-final-state` on F6 at fixed θ with
   data whose weekly and daily values are all `NA` (the filter is then the
   prior; inferred that the binder accepts an all-`NA` stream — if it refuses
   one, a single observed row at day 1 with a flat likelihood serves), then
   `simulate --init-state --to 49 --obs`, 20,000 replicates. The week-28 row
   exists, its period is `[21, 28)`, and its mean and variance match a
   continuous `simulate` from `t_start` over 20,000 replicates within 4 Monte
   Carlo standard errors. Repeated with the real F6 data against the
   importance-sampling reference of spec §4.4 (prior draws weighted by the daily
   likelihood). Red on main: the row is absent.
2. **Splice exactness.** Through the test-only RNG-restore seam
   (`StartState.rng`, `chain_binomial.rs:20-29`): a continuous run to 49, and a
   head run to 25 whose final state, RNG and `F[21, 25)` seed a resumed tail.
   The tail's week-28 row equals the continuous run's byte for byte, for the
   uniform, `starting_on` and window-column twins and for a flow-ratio stream.
   Mutation check: dropping the carry term turns it red.
3. **Boundary restart is unchanged.** On main, record `simulate --init-state`'s
   trajectory and `--obs` files for a restart at `T = 21`, with F6's data cut to
   end at day 21 so that `T` is a boundary of every emitted period, from the
   current v1 file; commit them as expected output. After the change,
   regenerating the state as v2 and restarting writes byte-identical files, and
   the v2 file has no `open` line.
4. **The filter is unchanged when no stop is inserted.** `pfilter` on F6 with
   the daily stream bound (21 is a union time) prints a byte-identical
   log-likelihood with and without `--save-final-state`, and the pfilter leaf's
   `run_id` is the same. With the daily stream unbound (21 inserted), the
   `run_id` differs and the inserted-stop line is printed.
5. **Edited file refused.** `t=26` changed to `t=3`; one count changed; one
   carry changed; the checksum line deleted: each refused with §6.1's message,
   asserted verbatim up to the hashes.
6. **Mismatched provenance refused.** `--param beta=<other>` (names `beta` and
   both values); a sweep cell with a different value (names the cell); another
   model; another `--dt`; a scenario enabling an intervention that fires before
   `T` (refused) and one that first fires after `T` (allowed, and said).
7. **`--emit-every` across the origin.** A step that moves the period under way
   is refused with §6.3's message; one that puts a boundary at `T` runs.
8. **v1 refused** with §6.1's message.
9. **Fit source.** A PGAS fit on F6: the restarted week-28 row of draw `i`
   equals the sum of its saved path's flows over `[21, 25)` plus the restarted
   run's flow over `[25, 28)`, before observation noise (read from the
   trajectory, not the obs file). An off-grid start is refused with §7's
   message.
10. **Identity.** `InitStateEnsemble` hashes differ for two ensembles equal but
    for one carry value; the `resolve` identity matrix
    (`cli/src/resolve/tests.rs:420`) gains an `init_state_carry` row.
11. **Backend matrix.** No cell changes: `--init-state` remains chain-binomial
    only, and the existing refusals for Gillespie, ODE and reactive models
    (`main.rs:1466-1488`) keep their tests.

Existing tests that write v1 files by hand (`gh641_init_state_forecast.rs`,
`gh1084_pfilter_final_state_holdout.rs`, `runid/src/inputs/tests.rs`) produce
their files through `pfilter` or the library writer instead; only the refusal
tests in 5 and 8 write a file by hand.

## 10. What changes in the runbook and the spec

Not edited in this PR; the implementing PR updates them to point here.

Runbook, `docs/dev/proposals/2026-11-15-temporal-architecture-migration.md`:

- Stage 5, "The fixed-origin driver" bullet, its last sentence: "No change to
  `particle_filter.rs`, resampling or the saved-state format." The saved-state
  format changes (here); `particle_filter.rs` and resampling still do not.
- Stage 5, "Restarts stay conservative." bullet (lines 603-607): the refusal of
  periods that began before the restart (T3 gh#1081, R2 gh#1085) is replaced by
  this proposal's exact restart. The `holdout_after` (R1) and PMMH-message (R3)
  clauses stand.
- Stage 5 tests, "Restarts." bullet (lines 658-662): "across a period under way
  is refused, for a uniform stream and a window-column stream" becomes §9 tests
  1-2 here.
- Part 7, "High-risk files" (lines 853-856): "changes neither the filter nor its
  saved-state format" — the format changes here, not in Stage 5.
- Part 8, D39 (lines 867-871): "restarts across such a period are refused and
  the saved-state format is unchanged" is superseded by D64.

Spec, `docs/dev/temporal-architecture.md`:

- §4.4, "Restarts." paragraph (lines 658-661): restores counts only, refuses a
  period begun before the restart → restores counts and running totals, and
  completes the period.
- §5.2 invariant table, row "A restart never emits a period that began before
  it" (line 792) → "A restart's straddling row equals the pre-origin running
  total plus the post-origin flow", enforced by `simulate --init-state`, checked
  (the §6.3 match and §9 test 2).
- Part 7 table, row "Restarts that continue a period under way" (line 913): no
  longer deferred.

User docs, `docs/inference.md`, "What a restart does and does not carry": the
implementing PR rewrites it for v2.

## 11. Decisions for the maintainer

**Decision 1. When the period under way starts between the filter's stops.** The
running total can open only at a union time. In the gh#1081 fixture with only
the weekly stream bound, `a = 21` is not one. (a) Insert `a` as a stop without
an observation, as the binder already does for declared period starts: the
filter's log-likelihood becomes another realisation of the same estimator, its
ESS and prequential traces gain a step with no observation, and the pfilter leaf
re-keys through `inserted_stops`; `particle_filter.rs` is not touched. (b) Open
the total at the substep that begins at `a`, inside the propagation loop in
`particle_filter.rs`: every pfilter output stays byte-identical, at the cost of
an edit to a high-risk file's inner loop. (c) Refuse: the canonical case would
be refused. Recommendation: (a), leaning. It adds no new kind of step to the
filter, and the change in outputs is confined to runs that ask to save a state,
said, and keyed.

**Decision 2. Scenario differences at restart.** The state at `T` reflects the
interventions active before `T`. (a) Refuse any scenario difference: safest, but
forbids the main use of a restart, comparing interventions that start after now.
(b) Allow differences only in interventions whose first fire time is at or after
`T`, refuse the rest; parameter `set`/`scale` differences are refused by the
parameter check. This is the runbook's D50 rule for forecast-mode scenario arms.
(c) Warn and proceed. Recommendation: (b), solid.

**Decision 3. What the file source's identity hashes.** (a) Keep the SHA-256 of
the file's bytes: simplest, but a display-only field (`source`) and formatting
enter the key. (b) Hash the parsed ensemble, one structure for both sources
(§8): formatting is presentation, and the two sources cannot drift. Either way
`HASH_VERSION` is bumped once. Recommendation: (b), solid.

**Decision 4. Carry for streams the filter did not bind.** A model stream with
an `emit_schedule` but no `--data` binding at pfilter time is still emitted by
`simulate --obs`. (a) Carry every emitted interval stream, bound or not: carry
slots depend only on the model's transitions and the emission plan. (b) Carry
bound streams only and refuse a straddling row for the others. Recommendation:
(a), solid; it costs one integer per stream per particle, and (b) would refuse a
row the filter could have completed.

## 12. Verification record

Commands run in this worktree at `f8e02d162`, output summarised.

- `rg -n "pub const FORMAT_TAG|^fn parse_origin_t|^pub fn read_final_states|^pub fn write_final_states" rust/crates/io/src/final_state.rs`
  → `:65`, `:118`, `:176`, `:299`; `parse_origin_t` checks only finiteness.
- `rg -n "final_states|reset_flows|fold_into_acc|reset_due_acc" rust/crates/sim/src/inference/particle_filter.rs`
  → fold `:428`, reset `:649-650`, returned swarm `:689`; resampling copies
  `acc` whole at `:609`.
- `sed -n 1270,1375p rust/crates/sim/src/inference/multi_stream_obs.rs` → union
  axis includes declared starts with the "costs a resampling draw" comment;
  `reset_at_union` built from declared starts only.
- `rg -n "n_interval_streams\(\)" rust/crates/sim/src rust/crates/cli/src` →
  callers in `particle_filter.rs`, `pgas.rs`, `pgas_grad.rs`, `pgas_init.rs`,
  `if2.rs`, `ode_loglik.rs`, `pmmh.rs`: each sizes `acc` and folds/resets
  through the model, the basis for "unchanged when no carry slots" (inferred for
  each caller's bit-identity; test 4 pins pfilter's).
- `sed -n 167,246p rust/crates/cli/src/obs_emit.rs` and
  `sed -n 4218,4245p rust/crates/cli/src/main.rs` → the per-form behaviour in
  §2's table.
- `sed -n 1490,1540p rust/crates/cli/src/main.rs` → no provenance checks; the
  file-source digest is `digest_bytes` of the file.
- `rg -n "Granularity::" rust/crates/cli/src` → PGAS paths are
  `Granularity::Substep` (`pgas.rs:1215`).
- `rg -n "pub const HASH_VERSION" rust/crates/runid/src` → `5`.
- `gh issue view 1081`, `gh issue view 1085` → the reproductions (run) cited in
  §2.

Not run for this proposal: the distributional and splice tests of §9, and the
claim that a stop already on the union grid leaves the filter byte-identical (by
reading: no scoring bin's schedule changes and no stop is added; test 4 settles
it).
