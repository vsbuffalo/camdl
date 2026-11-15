# Proposal: the score artifact

**Amended by** `docs/dev/temporal-architecture.md` and
`2026-11-15-temporal-architecture-migration.md` (2026-11-15). Where they differ,
they govern: `ObservedRow` carries what the row measures (`measures`, its
instant or period) instead of `time`; the fit records a `training_partition`
(per-stream row prefix and digest, with the training extent) instead of
`train_end`; held-out means every row that did not train and ends after the
training extent; the trace change here is schema 7, since `main` already writes
schema 6 (gh#312); the lead-time type gains a fixed-origin variant with
per-target horizons (D35, D41).

**Status:** Proposal **Date:** 2026-11-14 **Motivation:** camdl fits are to be
published as immutable bundles and compared on a static site that computes
nothing. The scores a bundle carries must therefore be computed once, stored,
and able to say on their own which fit, which data, which settings and which
dates they describe. Today `camdl compare` computes them in a temporary
directory and keeps nothing.

This is the score half of proposal 2 in the set defined by
`2026-11-13-published-comparison-principles.md` (the principles). It uses that
document's vocabulary: a **bundle** is the published form of one fit; **scored
lead time** is how far ahead of the last conditioning observation a scored
prediction is made; **scored steps** is the number of observations scored. The
word "horizon" is not used. Predictive artifacts (the plotted predictive series)
are a separate proposal; this one defines the lead-time type they will carry.

The typed comparison report that reads these artifacts is
`2026-11-14-scoring-report-library.md` (proposal B). The crate both proposals
put their types in, `score`, is introduced there.

Code references are to `main` at `e8e7c24a0` (2026-11-14). Since then the trace
module has moved from `rust/crates/sim/src/inference/prequential.rs` to
`rust/crates/score/src/trace.rs` and the evidence module from `cli` to
`rust/crates/score/src/evidence.rs` (phase 1 of proposal B), and `compare`'s
chain selection has gained `--chains` (`982eca439`, `770013f76`). Three later
changes on `main` bear on the design and are stated where they apply: fits on
the ODE backend can be compared (gh#312, `derive_ode_trace`); the held-out cut
is made on the scoring axis (gh#1062, pull requests #1064 and #1066); and
`HASH_VERSION` is 3 (gh#1046). File paths and line numbers below are as they
were at `e8e7c24a0`. Every statement under
[What exists today](#what-exists-today) was read in the code at that commit.
Nothing was built or run for this proposal; a statement that rests on something
other than reading the cited lines is marked _not verified_ with the reason.

## Terms

- **Stream, source, leaf.** A model's `observations { }` block declares
  observation streams. A data file is bound to a **source**; each concrete
  stream that reads it is a **leaf** (an indexed block over two patches is two
  leaves of one source). Defined in `2026-11-13-compare-on-selected-streams.md`.
- **Prequential trace.** The per-time-step record of one-step-ahead predictive
  scores a particle filter produces while passing over the data
  (`PrequentialTrace`, `rust/crates/sim/src/inference/prequential.rs:175`). Its
  format is at schema 5.
- **elpd.** Expected log predictive density: the sum over scored steps of the
  log of the predictive density at the observed value.
- **CRPS.** Continuous ranked probability score, an accuracy score in the units
  of the observed value.
- **PIT.** Probability integral transform: where the observed value falls in the
  predictive distribution. Uniform on (0, 1) for a calibrated model.
- **Replicate.** One of several repeat filter runs at different seeds. Their
  spread gives the **filter-noise standard error**, the Monte Carlo error of the
  elpd that comes from the filter and not from the data.
- **Held-out.** Scored on observations the fit never saw: those after the fit's
  **training boundary**. Today the recorded boundary (`train_end`) is the time
  of the last training row for a fit with separate holdout files, and the
  declared cutoff for a fit with `holdout_after`; this proposal makes it the
  time of the latest training row in both cases.
- **Run id.** The content hash that names a stored result
  (`docs/camdl-run-spec.md` §9).

## What exists today

### The trace carries scores and settings, not identity

`PrequentialTrace` (`prequential.rs:175-228`) holds `schema_version`, `t0` (the
index of the first scored observation), `provenance` (plug-in at one parameter
vector, or a mixture over `n_draws` posterior draws), `conditioning` (in-sample,
or held-out with `train_end`), `steps`, `warnings`, `score_from` (the scoring
boundary as a model time), `pit_randomization_seed`, `derivation` and
`scored_streams`.

A step (`PrequentialStep`, `prequential.rs:128-144`) is
`{t, log_score, ess,
per_stream}`. `log_score` is the joint score of every leaf
observed at `t`. Each `per_stream` entry (`StreamScore`,
`prequential.rs:80-113`) holds the leaf name, the observed value `y_obs`, the
predictive draws `y_pred_samples`, that leaf's own `log_score`, `crps`, one
randomized `pit`, a five-quantile `interval`, and `coverage` (the window the
observed value was accumulated over).

The `derivation` block (`Derivation`, `prequential.rs:238-277`) records
`particles`, `replicates`, `dt`, `seed`, where the parameters came from
(`theta`), `excluded_chains`, enabled and disabled interventions,
`replicate_elpd` (each replicate's total), `replicate_elpd_by_stream`,
`mc_se_elpd` (the filter-noise standard error) and `overrides`.

A trace does not record which fit it scored, which data, which engine version,
or any calendar date. `compare` reads an explicit `prequential.json` with its
data marked unknown for that reason
(`rust/crates/cli/src/compare.rs:1912-1918`).

### `compare` derives each trace in a temporary directory

For a fit handle, `derive_prequential` (`compare.rs:2254-2608`) writes parameter
files under the system temporary directory, spawns
`camdl pfilter
--save-prequential … --no-store` once per parameter draw per
replicate (`compare.rs:2428-2439`), reads each trace back, combines them
(`combine_mixture`, `compare.rs:2653`; `combine_scored_traces`,
`compare.rs:2682`), and deletes the files. Nothing is written to the store. The
store has eight artifact kinds and none is a score (`ArtifactKind`,
`rust/crates/runid/src/kind.rs:18-58`).

Default settings are 1000 particles, seed 1, 64 draws and 5 replicates
(`compare.rs:38-51`), which is 320 filter passes for a Bayesian fit.

### Held-out data is inside the fit's identity, in two places

The fit level of a run id is `FitDigest`
(`rust/crates/runid/src/inputs.rs:430-444`): the model digest, `data` (training
digests), `holdout_data`, the configuration blob and the engine version.

- With `[data.holdout]` (separate files of later observations), the files' bytes
  are folded as `holdout_data` (`rust/crates/cli/src/fit/cas.rs:490`), and their
  paths are in the configuration blob, which is the whole problem half of the
  configuration less two top-level keys, `output_dir` and `provenance`
  (`cas.rs:403-411`). The holdout rows do not affect training: the training
  boundary is the last training row, and holdout rows are only checked to lie
  after it (`rust/crates/cli/src/fit/runner.rs:1990-2045`).
- With `holdout_after = <time>` (one file holding the whole series), the fit
  drops later rows at load. `holdout_data` is empty, and the training digest is
  the SHA-256 of the whole file (`StreamBindings::digests`,
  `rust/crates/cli/src/fit/stream_bindings.rs:109-124`), which contains the
  held-out rows.

So in both forms, giving a fit one more week of held-out observations changes
its run id.

### Data digests are of file bytes

`StreamBindings::digests` hashes each bound file's bytes once; the run identity
and the fit record (`fit.meta.json`: `data_hashes`, `holdout_hashes`, keyed by
source) share those digests. A file with a column added, its columns reordered,
or a number respelled digests differently. `compare` checks each file it is
about to score against those byte hashes and refuses a difference as changed
data (`check_training_freshness`, `compare.rs:447`; the holdout files at
`compare.rs:2359-2374`). No dataset name and no date of the data is recorded
anywhere: `grep -rn 'as_of\|dataset_key\|dataset_id'` over the `cli`, `sim`,
`runid` and `io` crates matches only an unrelated loop variable.

### Held-out status rests on the declaration

`compare` stamps a score held-out only when the fit's configuration declares a
holdout and the fit record carries a training window (`detect_sealed_holdout`,
`compare.rs:1997-2025`). A fit with no declaration records no window
(`apply_holdout_declaration` returns `None`, `runner.rs:2048`;
`record_training_window` then writes nothing,
`rust/crates/cli/src/run_meta.rs:902-906`). A `fit.toml` handle finds its fit by
a hash of the whole configuration less `[provenance]` (`config_identity_hash`,
`cas.rs:437-441`), which includes `[data.holdout]`.

### A combined score pools draws across passes

`combine_scored_traces` concatenates the predictive draws of every pass and
computes the CRPS, the PIT and the interval on the pool
(`compare.rs:2726-2740`), drawing one uniform per pooled value for the PIT
(`compare.rs:2711`, `:2735`). Each pass's draws reach it through the trace file
the child `pfilter` process wrote. Log scores combine by log-mean-exp across
draws within a replicate and then across replicates; a step's effective sample
size is the minimum over passes (`compare.rs:2721`); warnings are merged
(`compare.rs:2755-2763`); passes whose steps are not at the same times are
refused (`compare.rs:2697-2711`).

### The training cut is by label, the scoring cut by scoring time

A data row has a **label time**, the value in its time column. A stream that
accumulates over a period is **scored** at the close of that period, which need
not be the label: a stream declared `covers = day(D)` gives a row labelled D the
period [D, D+1) and scores it at D+1 (`Covers::From`,
`rust/crates/ir/src/observation.rs:304-309`).

The fit cuts its training rows by label (`o.time <= tau`,
`rust/crates/cli/src/fit/runner.rs:1972`) and records that label time as
`train_end`. At `e8e7c24a0`, `compare` cut scoring at `train_end` on the scoring
axis, so for a stream scored after its label the last training row was scored as
held-out; gh#1062 reproduced it. Since #1064 and #1066, `compare` takes the
boundary on the scoring axis, as the latest scoring time of any training row
over the bound leaves (`heldout_scoring_boundary`), passes it to both the filter
and the ODE path, and checks per leaf that no scored row is a training row
(`check_no_training_row_scored`). The design below keeps exactly that rule and
adds the proof that the earlier rows are the ones the fit trained on.

### Two ways a trace is derived

A fit on the chain-binomial backend is scored by particle-filter passes
(`camdl pfilter` child processes). A fit on the ODE backend is scored in process
by its own likelihood, with no filter (`derive_ode_trace`,
`rust/crates/cli/src/compare_ode.rs`): given the parameters the trajectory is a
point, and the predictive at each observation is the observation density there.
A trace records which, as its `basis` (`PredictiveBasis::Filtered` or
`OdeFreeForward { samples_per_draw }`, `rust/crates/score/src/trace.rs`), and
`compare` refuses to difference traces on different bases, with no override.
Both kinds are combined across posterior draws by the same function
(`combine_mixture`).

### The score may run a model other than the fit's

`derive_prequential` uses the fit's archived compiled model when present and
otherwise recompiles the source file the configuration names
(`compare.rs:2306-2311`), which may have changed since the fit.

### Time is model time

`t`, `train_end` and `score_from` are `f64` model times. A model may declare a
calendar origin and a time unit, and `ir::caltime::internal_to_date`
(`rust/crates/ir/src/caltime.rs:231`) converts a model time to a date. The trace
records no date.

### Lead time is implicit

Every score in a trace is one observation ahead: the filter assimilates each
observation after scoring it. No field says so. Scoring at longer lead times
(forecast mode, where assimilation stops at an origin) is designed in
`2026-08-29-honest-predictive-evaluation.md` §3.7.2 and tracked as gh#808, which
is open; the `Conditioning` enum has no forecast variant
(`prequential.rs:57-69`). `fit predict` names its two modes with a flag called
`--horizon` (`rust/crates/cli/src/args/types.rs:130`).

### Settings checks cover particles and replicates only

`compare` refuses rows whose `derivation.particles` or `derivation.replicates`
differ (`filter_settings_mismatch`, `compare.rs:2108-2134`). It does not compare
the number of posterior draws: the provenance check compares the enum variant
and ignores `n_draws` (`compare.rs:1115-1116`). It does not compare seeds. Filed
as gh#1054.

### The PIT is one randomized draw

For a count predictive the PIT of an observed value `y` lies anywhere between
P(X < y) and P(X ≤ y). The trace stores one uniform draw on that interval
(`pit_sample_randomized`, `prequential.rs:910`, called at `:808`) and the seed.
gh#1049 (open) measured the 90% coverage figure moving by up to 7.7 percentage
points on heavily tied streams when only the uniform sequence changed.

## The gap

1. A score cannot be found again, because it is not stored, and cannot be
   interpreted alone, because it does not name its fit, data, engine or dates.
2. New held-out data cannot be scored against an existing fit without changing
   the fit's identity.
3. Two fits published independently have no shared settings to be scored under,
   and a draw-count difference is not detected.
4. A published calibration figure depends on a seed.

## The design

### The artifact

A **score artifact** is one JSON document, `score.json`, holding one fit's
prequential scores on one set of observations under one group of settings. Its
Rust type is `score::ScoreArtifact`. Its schema is named `camdl.score` and
versioned on its own, as `{major, minor}`. A reader refuses a major version it
does not know. Within a major version, fields are only added, and a reader
reports an enum value it does not know as unknown (principles, R4 and R5). This
is the commitment the principles call "read forever", and it starts at major 1.

```rust
pub struct ScoreArtifact {
    pub schema: SchemaVersion,        // { name: "camdl.score", major: 1, minor: 0 }
    pub engine: EngineStamp,          // camdl version and git hash that computed it
    pub identity: ScoreIdentity,      // this artifact's run id and its three level hashes
    pub fit: ScoredFit,               // what was scored
    pub settings: ScoreSettings,      // what was asked for
    pub time: TimeAxis,               // origin, unit, window
    pub lead: ScoredLeadTime,
    pub data: Vec<StreamData>,        // one per leaf the filter read
    pub joint: JointSummary,
    pub streams: Vec<StreamSummary>,  // one per scored leaf
    pub trace: PrequentialTrace,      // per-step scores, trace schema 6
}
```

The table says, for each part, what the schema-5 trace already carries and what
is new.

| Part               | In the schema-5 trace                                     | New                                                               |
| ------------------ | --------------------------------------------------------- | ----------------------------------------------------------------- |
| `schema`, `engine` | `schema_version` of the trace only                        | The artifact's own schema version; the engine stamp               |
| `identity`         | nothing                                                   | all of it                                                         |
| `fit`              | `conditioning.theta_source` (a directory name)            | fit run id, model digest, method, convergence record              |
| `settings`         | `derivation` records what ran                             | what was requested, and the profile name                          |
| `time`             | `train_end`, `score_from`, each step's `t`, as model time | origin, unit, and each of those as a date                         |
| `lead`             | implicit                                                  | all of it                                                         |
| `data`             | nothing                                                   | all of it                                                         |
| `joint`, `streams` | computable from `steps` and `derivation`                  | stored summaries, so a reader computes nothing                    |
| `trace`            | all of it                                                 | PIT endpoints; a fixed quantile set in place of draws; step dates |

### The row digest

The row digest answers "are these the same observations", independent of how the
file is laid out. It is computed per leaf with the canonical hasher
(`runid::CanonicalHasher`).

**Where the rows come from.** Today the loader converts each time cell to a
model time and discards the cell (`load_observations`,
`rust/crates/cli/src/fit/runner.rs:1600`), and a per-row window's start is read
in a separate pass (`stream_times_for`,
`rust/crates/cli/src/pfilter.rs:1489-1532`). A new function,
`observed_rows(path, obs, siblings, opts) -> Vec<ObservedRow>` in
`rust/crates/cli/src/fit/observed_rows.rs`, reads one leaf's rows once, through
the column readers the loader already uses (`load_data_tsv_column_cells`,
`load_long_form_stream`, `load_stream_aux`, `read_boundary_column`), which are
extended to return each temporal cell as written and its column's kind beside
the converted time. `load_observations` and `stream_times_for` are rebuilt on
`observed_rows`, and so is the survey command's loader, which reads observations
through a separate dense reader today (`load_data_tsv_column`,
`rust/crates/cli/src/survey.rs:1271-1280`). Then the digest and every likelihood
read the same rows by construction, and there is one reader. The rows are those
of the file, routed to the leaf, before any filtering that depends on the run
(the drop of a hole whose period opens before the run starts,
`runner.rs:1647-1669`, and the alignment check against the integration step).

```rust
pub struct ObservedRow {
    pub time: f64,                       // model time, as the loader resolves it
    pub temporal: Vec<TemporalCell>,     // as written: [window_start,] time
    pub cell: Option<ObsCell>,           // None is a hole
    pub aux: Vec<(String, f64)>,
}
pub enum TemporalCell { Number(f64), Date(String) }  // Date is YYYY-MM-DD
```

**Row digest version 1** hashes, in order:

1. `ROW_DIGEST_VERSION`, a `u16`, equal to 1.
2. The number of temporal columns the stream binds, a `u8`: 2 for a stream
   declared with per-row windows (`window_start`, then the time column, which
   for such a stream is the window's stop), 1 for any other. A uniform window is
   part of the model and is in the model digest.
3. For each temporal column in that order, its kind, a `u8`: 0 when its cells
   are numbers, 1 when they are dates. The kind is per column, because the
   loader detects it per column (`convert_time_column` is called separately for
   `window_start`, `pfilter.rs:1523`), and a numeric start beside a dated stop
   is accepted.
4. The number of rows, a `u64`.
5. Each row, in the order the leaf's rows appear in the file. A row is:
   1. **Each temporal cell**, in the column order of item 2. A number is hashed
      as a finite `f64` with negative zero normalized to zero (`FiniteF64`,
      `rust/crates/runid/src/float.rs:23-40`), by its bits. A date is hashed as
      the text `YYYY-MM-DD`, length-prefixed. The cell as written is hashed, not
      the model time it resolves to, so two models with different origins that
      read one dated file agree on its digest.
   2. **The cell.** A `u8` tag: 0 for a hole (`NA`), 1 for a value, followed for
      a value by a `FiniteF64`. A hole is a row, with its temporal cells: the
      data specification makes `NA` different from a deleted row
      (`docs/camdl-data-spec.md`, "Missing observations").
   3. **Auxiliary values.** The count as a `u64`, then each `(name, value)` pair
      the stream's likelihood reads (a binomial denominator, an offset), sorted
      by name, the name length-prefixed and the value a `FiniteF64`. A hole has
      none: the loader clears them (`runner.rs:1643-1647`).

Time, window start, value and auxiliary values are everything the likelihood
reads from a file for one leaf.

**Order.** A file whose rows for a leaf are not in strictly increasing time is
refused when the stream is bound
(`rust/crates/sim/src/inference/multi_stream_obs.rs:1003-1010`), after the
digest is computed. So every file that any stored result was computed from has
its leaf's rows in one order, and file order is canonical for them. In a
long-form file the rows of different leaves may be interleaved in any way; each
leaf's digest sees only its own rows.

**What is and is not hashed.** Numbers are hashed as values, so `10`, `10.0` and
`1e1` agree. A dated cell is always `YYYY-MM-DD`: the date parser rejects
time-of-day and zone forms (`parse_iso_date`,
`rust/crates/ir/src/caltime.rs:112-119`). A column read under
`--time-format
numeric` is numeric whatever its cells look like
(`cells_are_dated`, `rust/crates/cli/src/caltime_load.rs:108`). Not hashed:
column order, columns no stream binds, the interleaving of other leaves' rows,
whitespace around cells, line endings, the file name, and the leaf's name.
Observation files are tab-separated (`pfilter.rs:1259`, `:1308`), so the
separator does not vary.

**Two limits.** A file with dated cells and the same series written with numeric
times digest differently: the digest hashes what the file states, and which
calendar a number means is the model's business. And a long-form stream with
per-row windows is not a form the loader supports (`stream_times_for` reads one
window start per file row, `pfilter.rs:1517-1524`); the digest is defined for
the forms the loader reads, and `observed_rows` refuses the others as the loader
does.

**Names.** The digest leaves the leaf's name out, so one series has one digest
wherever it is bound. Every identity that folds row digests folds
`(leaf name,
row digest)` pairs, sorted by leaf name, so a series bound to one
leaf and the same series bound to another never share an identity.

The byte digests stay in the fit record (`data_hashes`) as provenance: which
exact bytes were read when the fit ran. Nothing is looked up by them and nothing
is checked against them (see
[What replaces the byte checks](#what-replaces-the-byte-checks)).

### What a fit records, and its identity

**Every fit records its training boundary and its training rows.** Today a fit
records a training window only when its configuration declares a holdout
(`apply_holdout_declaration` returns `None` otherwise, `runner.rs:2048`;
`record_training_window` then writes nothing,
`rust/crates/cli/src/run_meta.rs:902-906`). `fit.meta.json` gains, for every
fit:

- `training_window.train_end`: the label time of the latest row the fit trained
  on, over every bound leaf. This is the one definition. Under
  `holdout_after =
  τ` the recorded value today is `τ` (`runner.rs:1981`); it
  becomes the latest training row at or before `τ`. The two select the same
  later rows from the fit's own file, and the row time does not move when `τ`
  falls between rows.
- `training_rows`: for each bound leaf, the row digest of its training rows
  (those with model time at or before `train_end`, with the tolerance the
  holdout code uses, `TIME_TIE_EPS`), and their count.

`holdout_hashes` is deleted from the fit record: held-out files are not a fact
about a fit.

**The fit's identity.** In `FitDigest` (`rust/crates/runid/src/inputs.rs:430`):

- `holdout_data` is deleted.
- `data` becomes one `(leaf name, row digest)` pair per bound leaf, sorted by
  leaf name, over that leaf's training rows. The two layouts
  `identity_keyed_training` preserves today (by source, or once per leaf under
  the one-file shorthand, `stream_bindings.rs:137-145`) collapse into this one.
- The `[data.holdout]` table is removed from the configuration blob before
  hashing. `serialize_minus` removes top-level keys only (`cas.rs:228-250`), so
  this needs a nested removal, added to that one function. `holdout_after` stays
  in the blob.
- The same removal is made in `config_identity_hash` (`cas.rs:437-441`), the
  hash by which a `fit.toml` handle finds its fit
  (`rust/crates/cli/src/fit/handle.rs:627-635`). Otherwise editing a holdout
  path would make `camdl compare fit.toml` find no fit.

With these, appending held-out rows to a `holdout_after` file, or adding,
changing or removing a `[data.holdout]` table, leaves the fit's run id unchanged
and produces a new score run id. A fit run with no holdout declared and the same
fit with one declared are one fit. A `holdout_after` written relative to the
last observation (`last_obs - 6 weeks`) re-keys the fit when the appended rows
move the boundary past another training row, and not otherwise; in both cases
the recorded `train_end` and `training_rows` describe what the fit trained on.

One exception is correct and stays. A model that declares a time anchor from the
data (`last_obs`) resolves it over the whole training file, without the holdout
cut (`rust/crates/cli/src/main.rs:4106-4160`), and the resolved value is in the
model digest (`rust/crates/cli/src/obs_anchor.rs:10-15`). Appending rows to such
a fit's file changes its model and re-keys it.

**The other identities.** `Pfilter`, `Survey` and `ProfilePoint` fold
observation data too. They switch to `(leaf name, row digest)` pairs over every
row read for each bound leaf, in the same commit. `Pfilter` already folds name
and digest pairs (`rust/crates/cli/src/pfilter_cas.rs:82`, `:104-106`).
`--table` files are not observation data and keep byte digests.

**One re-key.** These changes raise `HASH_VERSION` from 3 to 4
(`rust/crates/runid/src/hash.rs`). The plan to share the bump with the gh#1046
fix is no longer possible: that fix landed on its own and took 3. So every
stored result re-keys once more; `HASH_VERSION` is folded into every root hash
(`rust/crates/runid/src/kind.rs:79-83`). What the bump invalidates: every stored
leaf is recomputed on next use. Old leaves stay on disk and readable by path.

### Held-out is decided by rows, not by a declaration

Today a score is stamped held-out only if the fit's configuration declares a
holdout and the fit record carries a training window (`detect_sealed_holdout`,
`compare.rs:1997-2025`). With the declaration out of the fit's identity it can
no longer carry that, and the cut it implies is on the wrong axis for some
streams (above). The rule becomes a statement about rows.

For each leaf, the rows read split into the **training prefix** (rows whose
label time is at or before the fit's `train_end`) and the **later rows**.

- **The prefix result**, per leaf, is a fact: `Matches` when the training prefix
  digests to the fit's recorded `training_rows` for that leaf, `Differs` when it
  does not, `NotRecorded` when the fit record has none.
- **The scoring boundary** is on the scoring axis: the latest scoring time of
  any training-prefix row, over every leaf. The filter assimilates and does not
  score every step at or before it. A later row of one leaf whose scoring time
  falls at or before the boundary is therefore assimilated and not scored: the
  filter has one boundary for all leaves (a boundary per leaf needs the filter
  to record more; it is out of scope, as
  `2026-11-13-compare-on-selected-streams.md` records). This errs toward not
  scoring a held-out row, never toward scoring a training row.
- After scoring, the derivation checks that no scored row is a training-prefix
  row, per leaf, by row. This replaces the check that scored times exceed
  `train_end`.

The conditioning recorded in the trace is then one of three:

| Conditioning          | When                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `HoldOutTail`         | scoring is past the boundary and every leaf's prefix result is `Matches`                 |
| `HoldOutTailUnproven` | scoring is past the boundary and some leaf's prefix result is `Differs` or `NotRecorded` |
| `InSample`            | `--in-sample` was given: every row read is scored, and no boundary is applied            |

`HoldOutTail` loses its `theta_source` field, which is a directory name and not
a function of the score's key; the fit is named by `ScoredFit.run_id`.

Three refusals, each made before any filter pass runs:

- **Nothing to score.** No later row has a scoring time past the boundary and
  `--in-sample` was not given. This keeps today's refusal
  (`compare.rs:2582-2587`): a missing or wrong holdout file must not turn into
  an in-sample score without being asked for. A fit with no later data is scored
  with `--in-sample`.
- **A different set of leaves.** The leaves read must be exactly the leaves the
  fit recorded `training_rows` for.
- **No recorded training rows.** A fit record written before this change has no
  `training_rows`; it is refused held-out with a message to re-run the fit,
  which the hash-version bump (below) forces in any case, and may be scored with
  `--in-sample`.

Which files are read. Each source is read from one file holding the whole series
for that source, training rows and later rows together:

- from `--data SOURCE=FILE` when given;
- else, for a fit whose configuration has a `[data.holdout]` file for the
  source, the training file followed by the holdout file, joined as today
  (`merge_holdout_file`, `compare.rs:2825`). A `fit.toml` handle reads the
  configuration as it is now; a stored fit reads its archived one;
- else the training file, which under `holdout_after` holds the later rows.

A `--data` file that holds only later rows has a training prefix that differs
from the fit's, and is reported as such.

### What replaces the byte checks, and where a waiver lives

`compare` today compares the bytes of each training and holdout file against the
fit record (`check_training_freshness`, `compare.rs:447`; the holdout files at
`compare.rs:2359-2374`) and refuses a difference as changed data. Under this
proposal a file that grew, or a new holdout file, is the expected case. The
check becomes the prefix result above. Rows after `train_end` have no recorded
expectation and are not checked; they are part of the score's own identity.

The prefix result is stored in the artifact, per leaf (`StreamData.prefix`). It
is a function of the rows read and the fit, so of the score's key. A waiver is
not: it is a fact about one command. So no waiver is stored in the artifact or
its trace. `TraceOverride::ChangedData` and `TraceOverride::InSampleForced` are
deleted from the trace, and what they said is carried by facts:

- "the scored data was not what the fit trained on" is `HoldOutTailUnproven`
  with a `Differs` prefix;
- "scored in-sample although later rows were available" is `InSample` with a
  leaf whose `n_rows_read` exceeds its `n_rows_training`.

`camdl score` refuses to compute an artifact with a `Differs` prefix unless
`--allow-changed-data` is given. `camdl compare` reads the fact from whatever
artifact it is handed, stored or fresh, and applies its changed-data check to it
(proposal B): a `Differs` prefix refuses unless the waiver is given on that
`compare` command, and the override is recorded in that report. An artifact
computed under a waiver therefore cannot pass a later comparison silently.

### The score's identity and the store

`ArtifactKind` gains `Score`, stored under `scores/`, with tag index 8. The tag
indices are an explicit table (`kind.rs:49-58`), so appending one changes no
existing run id.

A score's run id has three levels, in path order:

| Level      | Folds                                                                                                                                                                                                                                                      | Path label                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `fit`      | The run id of the fit-method leaf that supplied the parameters, and the content digest of the parameter file read from it (`draws.tsv`, or the optimizer's winner)                                                                                         | the fit's directory name      |
| `data`     | For each leaf the filter read, sorted by leaf name: the leaf name and the row digest of every row read. Then the scoring boundary: absent for in-sample, `train_end` otherwise                                                                             | `in_sample` or `held_out`     |
| `settings` | Particles, replicates, draws, seed, integration step, the basis with its samples per draw, excluded chains, enabled and disabled interventions; the engine version that computes the score; the trace schema version; the major and minor of `camdl.score` | the profile name, or `custom` |

The fit level uses the existing lineage reference
(`ArtifactRef { run_id, kind,
artifact, digest }`, `docs/camdl-run-spec.md`
§9.2.4, "Lineage").

The engine version is in the key because a store lookup tests the run id and the
hash and format versions and nothing else
(`rust/crates/runid/src/store.rs:273-279`): without it, a later camdl with a
corrected filter, or a different convergence rule, would be served the old
`score.json`. The pfilter identity folds the engine the same way, through its
model digest. So a new camdl computes its own scores; the old leaves remain.

The leaf holds `run.json` (the standard record) and `score.json`. Everything in
`score.json` is a function of the three levels.

### Data identity in the artifact

```rust
pub struct StreamData {
    pub leaf: String,
    pub source: String,
    pub role: StreamRole,
    pub rows_training: RowDigest,     // the fit's training rows for this leaf
    pub rows_read: RowDigest,         // every row the filter read
    pub rows_scored: RowDigest,       // the rows it scored
    pub prefix: PrefixResult,         // Matches, Differs, or NotRecorded
    pub n_rows_training: usize,
    pub n_rows_read: usize,
    pub n_rows_scored: usize,
}
pub enum StreamRole {
    /// The leaf has scored rows.
    Scored,
    /// The filter assimilated the leaf's rows and scored none of them: the leaf
    /// has no row after the training boundary.
    AssimilatedOnly,
}
```

`RowDigest` is the digest as lowercase hexadecimal text; the hashing is in
`runid`, so the `score` crate does not depend on it. `Recorded<T>`, used below,
is `Value(T)` or `NotRecorded`; it serializes as an explicit variant, never as
an absent key or `null`, so a reader renders "not recorded" (R5).

Two artifacts were scored on the same observations when, for each scored leaf,
their `rows_read` digests are equal. That is what proposal B's shared-data check
compares.

### The dataset label and the as-of date

The dataset label (a name by which a series is recognized across repositories)
and the as-of date (the date a file's contents were current) are the author's
statements and cannot be computed. They are written in a file beside the data
file: `cases.tsv` is described by `cases.meta.toml`.

```toml
dataset = "drc-insp-sitrep/nord-kivu/confirmed-weekly"
as_of = "2026-11-10"
```

The file has these two keys and refuses any other. When it is absent both values
read "not recorded". The label is a label: two series are known to be the same
observations only by equal row digests.

Neither value is ever hashed. The store's rule is that anything in a
content-addressed file is a function of its key
(`.claude/rules/run-identity.md`), so the two values are **not** in
`score.json`: there, a statement edited after scoring would be served stale from
the store. They are recorded in three places that are not keyed:

- the fit record (`fit.meta.json`), per source, as read when the fit ran, beside
  the byte hash of the file;
- the score leaf's `run.json`, in its provenance block, per source, with the
  byte hash of each file read. `camdl score` and `camdl compare` rewrite this
  block when a store hit finds the statements changed, as a label on `run.json`
  is rewritten today;
- the bundle, which carries the data files and so carries each `.meta.toml`
  beside its file. The bundle record (proposal 1) lists both values per stream,
  read at publish time, and that is where the published page reads the data date
  (R10).

`docs/camdl-data-spec.md` gains a section for the file.

### Scoring through the store

A new command writes the artifact:

```
camdl score FIT [--data SOURCE=FILE ...] [--profile NAME | --particles N
                 --replicates N --draws N --seed N] [--in-sample]
                 [--chains ... | --exclude-chains ...] [--allow-changed-data]
```

`FIT` is any fit handle `compare` accepts. Chain selection is `compare`'s: the
selector `main` gained in `982eca439` and `770013f76`, in either spelling. The
score's key folds the resolved set of excluded chains, so the two spellings of
one selection are one score.

**The model scored is the fit's.** The derivation uses the fit's archived
compiled model (`model.ir.json`) and checks that its digest equals the model
digest in the fit's run identity. A fit folder without the archive, or whose
archive does not match, is refused. The fallback that recompiles the source file
the configuration names is deleted: the score's key names the fit, so the score
must be of the fit's model.

**Order of work.** Read the rows; compute the row digests, the prefix results
and the scoring boundary; make the three refusals above and the changed-data
refusal; resolve the score's run id; look it up; on a miss, derive. Everything a
refusal depends on is computed before the lookup, so a stored artifact is never
returned for a request that would have been refused.

The derivation is the body of today's `derive_prequential`, moved to a function
both commands call. It still runs one `camdl pfilter` process per parameter draw
per replicate.

`camdl compare`, given a fit handle, does the same and reads the stored or
freshly derived `score.json`. Its `--particles`, `--replicates`, `--draws` and
`--seed` options remain and select a `custom` settings level. `compare` no
longer keeps traces in the temporary directory.

`compare` also accepts a path to a `score.json` or to a score leaf. It continues
to accept a bare trace file, for a score computed outside camdl's derivation;
such a row has no identity and every check that needs one reports it as not
checked.

### One filter pass, and the stored trace

A mixture's CRPS and quantiles cannot be formed from each pass's CRPS and
quantiles, so the predictive draws must survive until the passes are combined,
and are dropped only then. There are therefore two types.

- **`score::trace::PassTrace`** is one scoring pass at one parameter vector and
  one seed, by a particle filter or by the ODE scorer: per step, the time, the
  joint log score and the effective sample size; per step and leaf, the observed
  value, the predictive draws, the leaf's log score and the coverage; and the
  pass's settings and warnings. `build_trace` produces it, for both bases. For a
  filter pass it is the interchange between a `camdl pfilter` child process and
  its parent, written by a new option `pfilter --save-pass FILE` and read back
  by the same binary; an ODE pass is produced in process. It is not a stored or
  published format and carries no schema promise.
- **`PrequentialTrace`, schema 6** is the finished trace, with no draws.

One function produces the second from the first:

```rust
pub fn finish(
    passes: &[Vec<PassTrace>],            // replicates, each a list of draws
    date_of: &dyn Fn(f64) -> Recorded<String>,
) -> Result<PrequentialTrace, CombineError>
```

`passes[r][m]` is replicate `r` at parameter draw `m`. Every replicate has the
same number of draws. `finish` computes, in this order:

1. **Agreement.** Every pass must have the same step times and, at each step,
   the same leaves with the same observed values and coverage. Otherwise it
   returns `CombineError`, with the pass and step named. This is today's
   refusal. Every pass must also have the same `basis`.
2. **Per replicate.** At each step, the joint log score and each leaf's log
   score are the log-mean-exp over that replicate's draws. The replicate's total
   elpd is the sum over steps of its joint log scores, and its per-leaf totals
   likewise. These are `replicate_elpd` and `replicate_elpd_by_stream`, and
   `mc_se_elpd` is `logmeanexp_with_se` of the totals, absent for one replicate.
3. **Across replicates.** The finished step's joint log score and each leaf's
   log score are the log-mean-exp over replicates of the per-replicate values.
   This is the nested combination `main` computes today, and the finished log
   scores are numerically identical to today's.
4. **Effective sample size.** The finished step's is the minimum over all
   passes, as today.
5. **The pool.** At each step and leaf, the draws of every pass are
   concatenated, replicates in order and draws in order within a replicate. From
   the pool, once: the CRPS (`crps_sample_fair`), the PIT endpoints, and the
   quantiles (below).
6. **Warnings.** `EssCollapse` is recomputed on the finished steps against the
   threshold the passes recorded; `StartsAtPrior` is kept if any pass carries
   it. `SamplesNotSaved` no longer exists.
7. **Provenance.** `PlugIn` for one draw per replicate, `Posterior { n_draws }`
   otherwise.
8. **Dates.** Each step's `date` is `date_of(t)`. The caller supplies the
   function, because `score` does not depend on the crate that knows calendars.

The `derivation` block's remaining fields (particles, seed, step, where the
parameters came from, excluded chains, interventions) are filled by the caller,
as today. A single pass is finished by the same function with `passes` holding
one replicate of one draw; `pfilter --save-prequential` writes that.

`finish` replaces `combine_mixture` and `combine_scored_traces`, which are
deleted. It draws no random number, so it lives in `score`.

### Trace schema 6

**PIT endpoints.** `StreamScore.pit` is replaced by

```rust
pub pit_lower: f64,   // P̂(X < y_obs): the share of pooled draws below the observed value
pub pit_upper: f64,   // P̂(X ≤ y_obs): the share at or below it
```

and `pit_randomization_seed` is deleted. Both are shares of the pool's finite
draws, the same draws the quantiles use. Today the randomized value divides by
every draw, finite or not (`prequential.rs:910-915`), while the interval uses
the finite ones (`prequential.rs:950`); the two now agree. When the pool has no
finite draw both endpoints are not a number.

**Expected coverage.** Coverage at level ℓ, with central interval
`[a, b] =
[(1−ℓ)/2, (1+ℓ)/2]`, is the mean over scored values of each value's
share: for `pit_lower < pit_upper`, the length of
`[pit_lower, pit_upper] ∩ [a, b]` divided by `pit_upper − pit_lower`; for
`pit_lower = pit_upper = u`, 1 when `a ≤ u ≤ b` and 0 otherwise, the closed
interval today's statistic uses (`prequential.rs:439-444`). A value with
non-finite endpoints is left out and counted in `n_nonfinite`, as today. This is
the expectation of today's statistic over the uniform draw, and depends on no
seed.

**The histogram** is the non-randomized PIT histogram of Czado, Gneiting and
Held (2009, "Predictive model assessment for count data", _Biometrics_
65(4):1254–1261, doi:10.1111/j.1541-0420.2009.01191.x). Each scored value
contributes a distribution function on [0, 1]: for `pit_lower < pit_upper` it is
0 up to `pit_lower`, rises linearly to 1 at `pit_upper`, and is 1 after; for
`pit_lower = pit_upper = u` it is the step at `u`. These are averaged over
scored values with finite endpoints. With `J = 10` equal bins, the height of bin
`j` is the mass the average puts on `[(j−1)/J, j/J)`, the last bin closed at 1;
a step at `u = 0` falls in the first bin and a step at `u = 1` in the last. The
heights sum to 1. When no scored value has finite endpoints there is no
histogram: `PitHistogram.bins` is `NotRecorded`.

`PrequentialTrace` gains `expected_coverage(stream, level)` and
`pit_histogram(stream, bins)` in the step that introduces schema 6, replacing
`pit_coverage`; the comparison table, the prequential summary `pfilter` prints
and, later, `StreamSummary` all call them. The per-stream TSV
(`rust/crates/cli/src/prequential_out.rs:65-71`) replaces its `pit` column with
`pit_lower` and `pit_upper`.

In the artifact both are stored in `StreamSummary`, because the viewer computes
no statistic (R3). This closes gh#1049.

**Quantiles.** `y_pred_samples` and the five-quantile `interval` are replaced by
`quantiles: Vec<f64>` at the levels listed once in the trace, `quantile_levels`.
The estimator is the one the interval uses today: linear interpolation between
order statistics of the sorted pool (`sample_quantile_sorted`,
`prequential.rs:920`; R's type 7), over the finite draws. The levels are the 23
the COVID-19 Forecast Hub required for its death and hospitalization targets:
0.01, 0.025, 0.05 to 0.95 in steps of 0.05, 0.975 and 0.99, which are the median
and the ends of 11 central prediction intervals (the hub's submission
instructions, `reichlab/covid19-forecast-hub`, `data-processed/README.md`; its
incident-case target asked for 7 quantiles only). The hub scores forecasts from
those intervals with the weighted interval score, which approximates the CRPS
(Bracher, Ray, Gneiting and Reich 2021, "Evaluating epidemic forecasts in an
interval format", _PLOS Computational Biology_ 17(2):e1008618,
doi:10.1371/journal.pcbi.1008618). Here the set is how a predictive distribution
is stored, not how it is scored. The 23 levels include the five the per-stream
TSV writes today (`rust/crates/cli/src/prequential_out.rs:65`). The principles
ask for a dense quantile set so that export to hub formats stays possible.

No code outside `compare.rs`, `pfilter.rs`, `prequential_out.rs` and the trace
module reads the draws (`grep -rl y_pred_samples rust/crates/cli/src`), so
nothing else changes reader. `--no-save-samples` and the `SamplesNotSaved`
warning are deleted. camdl-scope was not read for this proposal.

**Dates.** Each step gains `date: Recorded<String>` beside `t` (below).

**Older traces.** `camdl` reads schema 6 only and refuses an older trace with a
message to re-derive it. This follows the repository's alpha rule
(`VERSIONING.md`): local traces are cheap to recompute, and none has been
published. The read-forever commitment begins with `camdl.score` major 1, which
contains a schema-6 trace.

### The publish profile

`score::profile` defines named groups of settings. There is one:

```rust
pub const PROVISIONAL_1: Profile = Profile {
    name: "provisional-1", particles: 1000, replicates: 5, draws: 64, seed: 1,
};
```

These are `compare`'s present defaults. The integration step is not in the
profile: a score is computed at the step the fit ran at. `camdl score` and
`camdl compare` use this profile unless told otherwise, and `camdl publish` will
score with it.

The profile is named provisional because its values have not been checked
against real fits. They are to be settled later by measurement: the filter-noise
standard error against the paired differences being reported, and whether those
differences shift with the particle count. When settled, a new named profile is
added; artifacts scored under `provisional-1` stay valid and are comparable with
each other.

```rust
pub struct ScoreSettings {
    pub profile: ProfileRef,   // Named("provisional-1") or Custom
    pub particles: usize,
    pub replicates: usize,
    pub draws: usize,          // requested
    pub seed: u64,
}
```

The artifact records the name and the values. `trace.derivation` continues to
record what ran, including the number of draws actually used when a fit retained
fewer than requested (`ThetaSource::PosteriorDraws { n_used, n_available }`).

**Mismatch between artifacts** is decided on values, never on names, so a custom
run with the profile's values is compatible with it. Proposal B's preflight:

- refuses differing particle counts, replicate counts, or draws used, under the
  one existing override (`--allow-mismatched-filter-settings`);
- reports differing seeds as a note and does not refuse. A different seed is a
  different sample of the same estimator;
- reports an artifact scored under no named profile as such.

gh#1054 asks for the draw and seed parts on `main` now, ahead of this proposal.

### Everything the evidence label reads is in the artifact

The evidence label for a row against a baseline is a function of three numbers
(`evidence_label`, `rust/crates/cli/src/evidence.rs:201-209`): the difference in
elpd; its paired standard error; and the pair's combined filter-noise standard
error.

- The difference and its paired standard error are computed from the two rows'
  per-step joint `log_score` values (`paired_delta`, `compare.rs:3048-3057`).
  They are in `trace.steps`.
- The combined filter-noise standard error is `√(a² + b²)` of the two rows'
  `derivation.mc_se_elpd` (`mc_se_delta`, `compare.rs:2096-2101`). Each is in
  its own artifact.
- For a comparison on one stream, the per-stream figures are rebuilt from
  `derivation.replicate_elpd_by_stream` (`project_onto_stream`,
  `prequential.rs:497`). It is in the artifact.

So the schema-5 `derivation` block is sufficient for the label, and nothing is
missing for it. What the block lacks is context, not inputs: the profile name
and the requested draw count, both added in `settings`. `replicate_elpd` is kept
so that a paired filter-noise estimate, which would use the replicates' shared
seeds, can be computed later without a schema change.

`joint` and `streams` store what a reader would otherwise compute:

```rust
pub struct JointSummary {
    pub n_scored_steps: usize,
    pub elpd: Num,                     // proposal B: a float that may be non-finite
    pub mc_se_elpd: Recorded<Num>,     // NotRecorded for one replicate
}
pub struct StreamSummary {
    pub leaf: String,
    pub n_scored_steps: usize,
    pub elpd: Num,                     // sum of this leaf's marginal log scores
    pub mc_se_elpd: Recorded<Num>,
    pub mean_crps: StreamMean,
    pub expected_coverage_50: StreamMean,
    pub expected_coverage_90: StreamMean,
    pub pit_histogram: PitHistogram,   // 10 equal bins, non-randomized
    pub lead_span: Recorded<LeadSpan>,
}
```

There is no pooled CRPS or coverage across streams and no total of the
per-stream elpd values (D14, on `main`).

### Calendar time

```rust
pub struct TimeAxis {
    pub origin: Recorded<String>,      // the model's calendar origin
    pub time_unit: String,
    pub train_end: Recorded<TimePoint>,// NotRecorded for an in-sample score
    pub first_scored: TimePoint,
    pub last_scored: TimePoint,
}
pub struct TimePoint { pub t: f64, pub date: Recorded<String> }
```

A `date` is the calendar day that contains the model time `t`, as `YYYY-MM-DD`,
computed once, at scoring time, by `ir::caltime::internal_to_date`
(`caltime.rs:231`). The higher-resolution form (`internal_to_date_hires`) is not
used: for a time inside a day it returns text with a fractional-day suffix,
which is not a date. `t` is always stored beside it, so nothing is lost. A
step's `t` is the time the observation is scored at. For a stream that
accumulates over a period, that is the period's close, not the label the data
file gives the row: a row labelled day D under `covers = day(D)` is scored at
D+1 (`multi_stream_obs.rs:1507-1523`). The period itself is already recorded per
scored value (`coverage`). The page states dates as "scored at".

For a model with no calendar origin every `date` and `origin` is `NotRecorded`;
the page then shows model time with its unit. The training boundary and the
scoring window are therefore stated as dates wherever the model makes that
possible, and the viewer converts nothing.

### Scored lead time and scored steps

```rust
/// How far ahead of the last conditioning observation each scored prediction
/// was made.
pub enum ScoredLeadTime {
    /// Each scored value was predicted from every observation, of every
    /// assimilated stream, at earlier times; the filter then assimilated it.
    /// The basis is `Filtered`.
    OneObservationAhead,
    /// Nothing is assimilated: each scored value is predicted from the
    /// parameters alone, by the trajectory run forward from the start. The
    /// basis is `OdeFreeForward`. For a held-out score the lead time of a
    /// scored value is its scoring time minus the scoring boundary, and grows
    /// across the window.
    Unassimilated,
}

/// The gap, in model time, between a scored observation and the latest earlier
/// observation the filter had assimilated, over one leaf's scored steps.
pub struct LeadSpan { pub min: f64, pub max: f64, pub time_unit: String }
```

Both live in the `score` crate. `ScoreArtifact.lead` is the artifact's lead
time; each `StreamSummary` records `n_scored_steps` and its `lead_span`, because
"one observation ahead" is one week for a weekly stream and one day for a daily
one, and shorter still when another stream reports in between. A scored step
with no earlier observation (the first step of a score that starts from the
prior, which the trace's `StartsAtPrior` warning already flags) is left out of
the span; a leaf whose only scored step is that one has
`lead_span:
NotRecorded`.

For an `Unassimilated` held-out score, `lead_span` is measured from the scoring
boundary: its minimum is the first scored value's distance from the boundary and
its maximum the last one's. For an `Unassimilated` in-sample score the
parameters were fitted to every scored row and no lead time is defined;
`lead_span` is `NotRecorded`. The artifact's `lead` follows from the trace's
`basis`.

The predictive-artifact proposal will give each predictive series a value of
type `score::LeadTimeStatus`, defined here so the two cannot drift:

```rust
pub enum LeadTimeStatus {
    /// Predictions of this kind were scored, at this lead time.
    Scored(ScoredLeadTime),
    /// Nothing scored predictions at this series' lead time. R8: marked
    /// untested on the figure.
    NotScored,
}
```

**What this proposal assumes about gh#808.** Forecast-mode scoring will add
variants to `ScoredLeadTime` (a lead counted from a forecast origin) and will
store its scores as separate entries per lead time, never summed with one
another or with one-observation-ahead scores, as gh#808's own scope states.
Adding variants is a minor version of `camdl.score`; an older reader shows the
new value as unknown. Nothing in major 1 has to change for it. Until then a
filtered artifact is `OneObservationAhead`, an ODE artifact is `Unassimilated`,
and a free-running projection from a filtered fit is `NotScored`.

`fit predict` spells its modes with a flag named `--horizon`. Renaming it is
left for a later decision and is not part of this proposal.

### Only marginal scores per stream are stored

The artifact keeps the trace's limit: per leaf it stores that leaf's marginal
score, and the joint score of every leaf together. The joint score of a proper
subset of leaves cannot be recovered from it.

Recovering it needs each particle's likelihood for each leaf at each step, for
every replicate and draw: at the provisional profile, 1000 × 5 × 64 values per
leaf per step. That is too large to publish and is not needed by anything in the
first release. Consequently a single combined table over several selected
streams, and a separate training boundary per stream, remain unavailable, as
`2026-11-13-compare-on-selected-streams.md` records. Each needs the filter to
record more and its own proposal. `StreamSummary.elpd` is documented in the
schema as a marginal that must not be summed across leaves.

### Scored observations are published

The artifact contains the observed value at every scored step, for every scored
leaf (`y_obs`). A bundle that carries a score artifact therefore publishes those
observations, and the bundle record states it in those words. There is no
private-data mode in this proposal.

A note for a future private-data proposal: removing `y_obs` alone would not
withhold the observations. The per-step log score is the predictive density at
the observed value, the CRPS and PIT endpoints are functions of it, and the
bundle publishes the predictive distribution through its quantiles and its
model. For a count stream the log score narrows the value to at most two
candidates and the PIT endpoints separate them. This is inferred from the
definitions (`build_trace`, `prequential.rs:795-815`) and was not demonstrated.
A mode that withholds scored observations has to withhold per-step scores too,
and then nobody without the data can form a paired difference against that
member.

### The fit's convergence travels with its score

```rust
pub struct ScoredFit {
    pub run_id: String,                 // the fit-method leaf
    pub model_digest: String,
    pub method: FitAlgorithm,
    pub backend: InferenceBackend,
    pub convergence: ConvergenceRecord, // defined in proposal B
}
```

`ConvergenceRecord` holds the verdict as judged when the score was computed, the
rule it was judged by, and the diagnostics needed to judge it again with the
version of their computation. The diagnostics are those of the chains that were
scored: when chains are excluded, they are recomputed over the retained chains
(proposal B, "The sample judged is the sample scored"). Proposal B defines the
type and says how `compare` uses it. It is here so that a score artifact read
without its fit still says whether the sample behind it converged. Excluded
chains and the engine are in the score's key, so the record is a function of the
key.

`FitAlgorithm` and `InferenceBackend` are the existing enums in
`rust/crates/cli/src/run_meta.rs`, moved to `score` so the field is an enum a
reader handles exhaustively (R4). _Not verified:_ that the two enums depend on
nothing else in `cli`; if either does, the dependent part stays in `cli` as a
function over the moved enum.

## Invariants, as tests

Each is a test that fails if the property is lost.

1. **Self-description.** A `ScoreArtifact` serialized and read back with no
   other file present yields the same evidence label, against a second such
   artifact, as the two computed in one process.
2. **New data, same fit.** Appending rows after the training boundary to a
   `holdout_after` file leaves the fit's run id unchanged and changes the score
   run id. The same for adding, replacing and removing a `[data.holdout]` table,
   and a `fit.toml` handle still finds the fit after each. For a model that
   anchors a time on `last_obs`, appending rows does change the fit's run id.
3. **Training data still keys the fit.** Changing one training value, one
   training row's time, one auxiliary value, or one per-row window start changes
   the fit's run id.
4. **Layout does not key.** A file and a copy with columns reordered, an unbound
   column added, `10` written as `10.0`, CRLF line endings, and (long-form) the
   rows of different strata interleaved differently have equal row digests per
   leaf, and the fit's run id is equal.
5. **Holes are data.** Replacing a value with `NA` changes the row digest;
   deleting the row changes it differently.
6. **Row digest is pinned.** Committed fixture files (a wide stream, a long-form
   stream, a stream with per-row windows and a numeric start beside a dated
   stop, a stream with an auxiliary column) have committed digests per leaf; the
   test fails if a digest changes without `ROW_DIGEST_VERSION` changing.
7. **The digest reads what the likelihood reads.** For every fixture,
   `load_observations`, `stream_times_for` and the survey loader return what
   they returned before they were rebuilt on `observed_rows`. And the rows the
   fit keeps for training (`retain_rows`) are the rows `training_rows` digests.
8. **Names do not alias.** One file bound to leaf `a` and the same file bound to
   leaf `b` give different fit run ids and different pfilter run ids.
9. **Store round trip.** `camdl compare` run twice on the same fits spawns no
   filter pass the second time, and prints byte-identical output.
10. **`score` then `compare`.** `camdl score` followed by `camdl compare` at the
    same settings reads the stored leaf and derives nothing.
11. **Engine keys the score.** The score run id changes when the engine version
    string changes, with fit, data and settings fixed.
12. **Prefix result.** Scoring with a file whose rows up to the training
    boundary differ from the fit's is refused by `camdl score`; under
    `--allow-changed-data` the artifact is stored with `prefix: Differs` and
    conditioning `HoldOutTailUnproven`, and a later `camdl compare` without the
    waiver refuses it from the store. Scoring with a file that only grew is not
    refused and is `HoldOutTail`.
13. **Held-out without a declaration.** A fit run with no holdout declared,
    scored with `--data` naming the whole series extended past its training
    rows, is stamped held-out and scores only the later rows. Scored with no
    later rows and without `--in-sample`, it is refused.
14. **The boundary is a row time.** Under `holdout_after = τ` with `τ` between
    two rows, the recorded `train_end` is the earlier row's time, and appending
    rows after `τ` changes neither it nor the fit's run id.
15. **Pooled, then summarized.** For two passes with known draws, the finished
    trace's CRPS and quantiles equal those computed by hand on the concatenated
    draws, and differ from the mean of the two passes' own.
16. **Coverage is seed-free.** On a fixture of tied counts the stored expected
    coverage equals the value computed by hand, including a value below every
    draw (`pit_lower = pit_upper = 0`) and one above every draw.
17. **The histogram sums to one**, on the same fixture, and
    `pit_lower ≤
    pit_upper` at every scored value, with equality on a
    continuous fixture.
18. **Dates.** For a model with an origin, an instant stream's
    `first_scored.date` equals the date in the data file's first scored row, and
    a `covers = day(D)` stream's is the day after. For a model without an
    origin, every date is `NotRecorded` and the artifact still validates.
19. **Lead span.** For a weekly single-stream fixture, `lead_span.min` and
    `.max` both equal one week in model time; a leaf scored once from the prior
    has `NotRecorded`.
20. **Profile by value.** A custom run with the profile's values is compatible
    with a profile run; changing the draw count is refused; changing the seed is
    reported and not refused.
21. **Major version.** A document with `schema.major = 2` is refused by name. A
    document with an unknown `ScoredLeadTime` value is read, and that value is
    reported as unknown.
22. **Statements are never keyed and never stale.** Adding, changing or removing
    `cases.meta.toml` changes no run id and no byte of `score.json`; after the
    change, `camdl score` on a store hit leaves `run.json` holding the new
    values.
23. **No sum across streams.** No field of the artifact equals the sum of
    `StreamSummary.elpd` on a two-stream fixture, and none is named as one.
24. **No training row is scored.** A fit of a `covers = day(D)` stream with a
    holdout: the last training row, whose scoring time is after `train_end`, is
    not among the scored rows, and the first scored row is the first row the fit
    did not train on. The same fixture on `main` scores the training row.
25. **A leaf that ends early.** With two streams, one having no row after the
    boundary, that leaf is `AssimilatedOnly` and the other is scored.
26. **The fit's model.** A fit folder whose archived model is missing, or does
    not match the fit's model digest, is refused.
27. **One selection, one score.** `--chains` naming the kept chains and
    `--exclude-chains` naming the rest give the same score run id.
28. **Finished scores are today's.** On the phase 1 script's fixture with two
    draws and two replicates, the finished trace's per-step log scores, elpd,
    `replicate_elpd` and `mc_se_elpd` equal the values `main` computes.
29. **In-sample past training.** `--in-sample` on a fit with later rows gives
    `InSample` with `n_rows_read` greater than `n_rows_training`, and the same
    run id whether or not the configuration declares a holdout.
30. **Mutation checks.** For tests 3, 8, 11, 12 and 24, removing the property in
    the source makes the test fail; the commit message records the run.

## What this proposal does not do

- It defines no predictive artifact. Only the lead-time types those will carry.
- It defines no bundle, no `camdl publish`, and no private-data mode.
- It does not build forecast-mode scoring for filtered fits (gh#808).
- It does not store what a joint score over a subset of streams would need.
- It does not settle the publish profile's values.
- It does not rename `fit predict --horizon`.
- It does not change the particle filter (`particle_filter.rs`) or run a filter
  pass in-process. It changes what a pass writes for its parent (`PassTrace`).
- It does not add the report, the convergence refusal or the draw-count check;
  those are proposal B. It does replace the byte-based changed-data check with
  the prefix result, changes how held-out is decided, and deletes the fallback
  that scores a recompiled model, as stated above.
- It does not give each stream its own scoring boundary.

## Steps

Each step is one commit with its tests and a `Proposal:` footer. The order of
these steps among proposal B's is given once, in proposal B's "Steps": B's
phases 1 and 2 come first, then the steps below.

- **A1.** `PassTrace`, `pfilter --save-pass`, `score::combine::finish` (which
  replaces and deletes `combine_mixture` and `combine_scored_traces`), and trace
  schema 6: PIT endpoints, `expected_coverage` and `pit_histogram` on the trace,
  quantiles, step dates, `theta_source` removed. Every reader updated, including
  the per-stream TSV's columns, and the descriptions of the trace in
  `docs/camdl-run-spec.md` and `docs/inference.md`. The printed coverage figures
  change; the commit re-pins them and that is its subject. Tests 15, 16, 17, 18,
  28. Closes gh#1049.
- **A2.** `observed_rows`, with `load_observations`, `stream_times_for` and the
  survey loader rebuilt on it, and `runid::row_digest` with
  `ROW_DIGEST_VERSION`. Tests 4, 5, 6, 7.
- **A3.** The fit record: `train_end` as the latest training row's label time
  and `training_rows` for every fit; `holdout_hashes` deleted. The derivation:
  the prefix result, the scoring boundary on the scoring axis, the three
  conditionings, the three refusals, the row check after scoring, the archived
  model required; the two trace overrides deleted. Tests 12, 13, 14, 24, 25, 26,
  29.
- **A4.** Identity: `FitDigest.data` as `(leaf, row digest)` pairs,
  `holdout_data` deleted, `[data.holdout]` removed from both configuration
  hashes, the other data-folding identities switched, `HASH_VERSION` raised
  to 4. The commit states what the bump invalidates. Tests 2 (fit half), 3, 8.
  This step needs the maintainer's confirmation before it lands, as every
  identity change does. A3 and A4 land in one pull request, in this order: A3
  makes fits recorded earlier unscorable as held-out until they are re-run, and
  A4's bump is what re-runs them.
- **A5.** The `.meta.toml` reader; the two values in the fit record;
  `docs/camdl-data-spec.md`.
- **A6.** `ScoreArtifact` and its parts in `score`; `ScoredLeadTime`,
  `LeadSpan`, `LeadTimeStatus`; the profile and proposal B's `Profile` check.
  Tests 1, 19, 21, 23.
- **A7.** `ArtifactKind::Score`, its resolver (`cli/src/score_cas.rs`), and
  `camdl score`, with the statements in `run.json`. `docs/camdl-run-spec.md`
  §9.2.4, §9.3 and a new §10 entry for `score.json`. Tests 2 (score half), 11,
  22, 27.
- **A8.** `compare` reads and derives through the store, and its shared-data
  check compares `rows_read`. `docs/commands.md` and
  `docs/methods/model-comparison.md`. Tests 9, 10, 20.

## Verification

Commands run on `main` at `e8e7c24a0` for the claims the design rests on.

- `grep -n 'Sim =>\|FitMethod =>\|Pfilter =>\|Survey =>\|ProfilePoint =>\|Obs =>\|Projection =>\|SimEnsemble =>' rust/crates/runid/src/kind.rs`
  → eight lines, indices 0 to 7: no score kind, and indices are explicit.
- `grep -n 'no-store' rust/crates/cli/src/compare.rs` → line 2438, inside the
  `Command::new(&exe)` built at 2428: `compare` spawns `pfilter` and stores
  nothing.
- `sed -n 2726,2729p rust/crates/cli/src/compare.rs` → the combiner extends a
  pool with each pass's `y_pred_samples`.
- `grep -n 'holdout_data' rust/crates/cli/src/fit/cas.rs` → line 490,
  `holdout_data: data.identity_holdout()`.
- `sed -n 437,441p rust/crates/cli/src/fit/cas.rs` → `config_identity_hash`
  removes `provenance` only.
- `grep -n 'digest_bytes' rust/crates/cli/src/fit/stream_bindings.rs` → line
  114: digests are of file bytes.
- `sed -n 902,906p rust/crates/cli/src/run_meta.rs` → `record_training_window`
  returns at once on `None`.
- `grep -n "split('\\t')" rust/crates/cli/src/pfilter.rs` → lines 1259, 1308,
  1744, 1819: observation files are split on tabs.
- `sed -n 270,279p rust/crates/runid/src/store.rs` → lookup tests the run id,
  then `hash_version` and `format_version`.
- `grep -n 'd.particles != \|d.replicates != ' rust/crates/cli/src/compare.rs` →
  lines 2118 and 2121, and no line comparing draws or seeds.
- `grep -rl 'y_pred_samples' rust/crates/cli/src` → `pfilter.rs`, `compare.rs`,
  `prequential_out.rs`.
- `grep -n 'HASH_VERSION: u16' rust/crates/runid/src/hash.rs` → line 77, value
  2.
- `gh issue view 808 --json state` → `OPEN`; `gh issue view 1049 --json state` →
  `OPEN`.

The two literature citations were checked against their sources on 2026-11-14
(Crossref and the authors' technical report for Czado et al., whose journal text
was not reachable; the PLOS article and the hub's README for the quantile set).
The design was reviewed against the code by an independent agent the same day;
its twelve findings are resolved in the text above. A second independent review
of the revised text, against `main` after phase 1 of proposal B, found two
blockers and nine major items; they are resolved above too.
