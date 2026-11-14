# Proposal: the score artifact

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

Code references are to `main` at `e8e7c24a0` (2026-11-14). Every statement under
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
- **Held-out.** Scored on observations after the fit's **training boundary**
  (`train_end`, the model time of the last training observation), which the fit
  never saw.
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
source) share those digests. A file that is re-sorted, re-delimited or has a
column added digests differently. No dataset name and no date of the data is
recorded anywhere: `grep -rn 'as_of\|dataset_key\|dataset_id'` over the `cli`,
`sim`, `runid` and `io` crates matches only an unrelated loop variable.

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

### Identity and the store

`ArtifactKind` gains `Score`, stored under `scores/`, with tag index 8. The tag
indices are an explicit table (`kind.rs:49-58`), so appending one changes no
existing run id.

A score's run id has three levels, in path order:

| Level      | Folds                                                                                                                                                                | Path label                    |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `fit`      | The run id of the fit-method leaf that supplied the parameters, and the content digest of the parameter file read from it (`draws.tsv`, or the optimizer's winner)   | the fit's directory name      |
| `data`     | For each leaf the filter read, sorted by leaf name: the row digest (below) of every row read. Then the scoring boundary: absent for in-sample, `train_end` otherwise | `in_sample` or `held_out`     |
| `settings` | Particles, replicates, draws, seed, integration step, excluded chains, enabled and disabled interventions, the artifact's schema major, and the row digest version   | the profile name, or `custom` |

The fit level uses the existing lineage reference
(`ArtifactRef { run_id, kind,
artifact, digest }`, `docs/camdl-run-spec.md`
§9.2.4, "Lineage"), so a score depends on its fit the way a fit depends on the
scout that supplied its starts.

The leaf holds `run.json` (the standard record) and `score.json`.

**What moves out of the fit's identity.** `FitDigest.holdout_data` is deleted.
The `[data.holdout]` table is removed from the configuration blob before
hashing. `serialize_minus` removes top-level keys only (`cas.rs:228-250`), so
this needs a nested removal, added to that one function. `holdout_after` stays
in the blob: it decides which rows the fit trains on.

**What `FitDigest.data` becomes.** One row digest per bound leaf, sorted by leaf
name, over that leaf's training rows: those at or before the training boundary,
decided with the tolerance the holdout code already uses (`TIME_TIE_EPS`). The
two key layouts `identity_keyed_training` preserves today (by source, or once
per leaf under the one-file shorthand, `stream_bindings.rs:137-145`) collapse
into this one.

With both changes, appending held-out rows to a `holdout_after` file, or
supplying a new `[data.holdout]` file, leaves the fit's run id unchanged and
produces a new score run id. A `holdout_after` written relative to the last
observation (`last_obs - 6 weeks`) moves the training boundary when the file
grows; the training rows then differ and the fit re-keys, correctly.

**One re-key.** These changes raise `HASH_VERSION`
(`rust/crates/runid/src/hash.rs:77`, currently 2). The docket's D25 already
raises it to 3 for an unrelated fix. Both land in the same bump, so every stored
result re-keys once. `HASH_VERSION` is folded into every root hash
(`kind.rs:79-83`), so the bump re-keys every kind regardless; the `Pfilter`,
`Survey` and `ProfilePoint` identities that fold observation data switch to row
digests in the same commit, so one rule holds everywhere. `--table` files are
not observation data and keep byte digests.

What the bump invalidates: every stored leaf reads `Stale(SchemaDrift)` and is
recomputed on next use. Old leaves stay on disk and readable by path.

### The row digest

The row digest answers "are these the same observations", independent of how the
file is written. It is computed per leaf with the canonical hasher
(`runid::CanonicalHasher`), over the rows the data loader produced for that leaf
after reading the file and routing long-form rows to their leaf, and before any
filtering that depends on the run (the drop of a hole whose period opens before
the run starts, `runner.rs:1647-1669`, and the alignment check against the
integration step).

Row digest version 1 hashes, in order:

1. `ROW_DIGEST_VERSION`, a `u16`, equal to 1.
2. The time kind, a `u8`: 0 when the file's time cells are numbers, 1 when they
   are dates.
3. The number of rows, a `u64`.
4. Each row, in ascending time. Times within one leaf are strictly increasing
   (enforced by the loader,
   `rust/crates/sim/src/inference/multi_stream_obs.rs:919` and `:1008`), so the
   order is total and no tie rule is needed. A row is:
   1. **Time.** For numeric cells, the parsed value as a finite `f64` with
      negative zero normalized to zero (`FiniteF64`,
      `rust/crates/runid/src/float.rs:23-40`), hashed by its bits. For dated
      cells, the date as the text `YYYY-MM-DD`, length-prefixed. The file's own
      time cell is hashed, not the model time it resolves to, so two models with
      different origins that read one dated file agree on its digest.
   2. **Window.** For a stream declared with per-row windows, `window_start` and
      `window_stop`, each encoded as the time is. For any other stream, nothing:
      a uniform window is part of the model and is in the model digest.
   3. **Cell.** A `u8` tag: 0 for a hole (`NA`), 1 for a value. A value follows
      as a `FiniteF64`. A hole is a row, with its time and window: the data
      specification makes `NA` different from a deleted row
      (`docs/camdl-data-spec.md`, "Missing observations").
   4. **Auxiliary values.** The count as a `u64`, then each `(name, value)` pair
      the stream's likelihood reads (a binomial denominator, an offset), sorted
      by name, the name length-prefixed and the value a `FiniteF64`. A hole has
      none: the loader clears them (`runner.rs:1643-1647`).

Consequences, each a test below. Numbers are hashed as values, so `10`, `10.0`
and `1e1` agree. Not hashed: the separator, column order, columns no stream
binds, comment lines, whitespace, line endings, the file name, and the leaf's
name. The leaf name is left out so that one series bound under two names in two
repositories has one digest; the pairing of digests with leaves is by position
in a list sorted by leaf name, and the names are in the model digest.

A dated cell is always `YYYY-MM-DD`: the loader's date parser rejects
time-of-day and zone forms (`parse_iso_date`,
`rust/crates/ir/src/caltime.rs:112-119`). A column read under
`--time-format numeric` is numeric whatever its cells look like
(`cells_are_dated`, `rust/crates/cli/src/caltime_load.rs:108`), and is hashed as
numbers.

The byte digests stay in the fit record (`data_hashes`, `holdout_hashes`) as
provenance: which exact bytes were read. Nothing is looked up by them.

### Data identity in the artifact

```rust
pub struct StreamData {
    pub leaf: String,
    pub source: String,
    pub role: StreamRole,             // Scored, or AssimilatedOnly
    pub rows_training: RowDigest,     // the fit's training rows for this leaf
    pub rows_read: RowDigest,         // every row the filter read
    pub rows_scored: RowDigest,       // the rows it scored
    pub n_rows_scored: usize,
    pub file_sha256: Vec<String>,     // the bytes read, as provenance
    pub dataset: Recorded<String>,    // the author's label for the series
    pub as_of: Recorded<String>,      // the date the contents were current, YYYY-MM-DD
}
```

`RowDigest` is the digest as lowercase hexadecimal text; the hashing is in
`runid` (step 1), so the `score` crate does not depend on it. `Recorded<T>` is
`Value(T)` or `NotRecorded`; it serializes as an explicit variant, never as an
absent key or `null`, so a reader renders "not recorded" (R5).

**The dataset label and the as-of date** are the author's statements and cannot
be computed. They are read from a file beside the data file: `cases.tsv` is
described by `cases.meta.toml`.

```toml
dataset = "drc-insp-sitrep/nord-kivu/confirmed-weekly"
as_of = "2026-11-10"
```

The file has these two keys and refuses any other. It is read by every command
that reads the data file, for training and for held-out data alike, so a holdout
file scored a month after the fit carries its own date. When it is absent both
values are `NotRecorded`. Neither value enters any hash. The dataset label is a
label: two artifacts are known to describe the same observations only when their
row digests are equal, and the label says which series a reader should expect to
find compared. The fit record gains the same two values per source, read at fit
time. `docs/camdl-data-spec.md` gains a section for the file.

### Scoring through the store

A new command writes the artifact:

```
camdl score FIT [--data SOURCE=FILE ...] [--profile NAME | --particles N
                 --replicates N --draws N --seed N] [--in-sample]
                 [--exclude-chains ...]
```

`FIT` is any fit handle `compare` accepts. Without `--data` it scores the files
the fit's configuration names, as `compare` does today. With `--data`, the named
source is read from the given file instead, which is how later observations are
scored. In both cases the rows at or before the training boundary must digest to
the fit's recorded `rows_training` for that leaf. A mismatch is the existing
changed-data refusal, with its existing override and its record in the trace
(`TraceOverride::ChangedData`).

The derivation is the body of today's `derive_prequential`, moved to a function
both commands call. It still runs `camdl pfilter` passes and combines them; this
proposal does not change how a pass is run.

`camdl compare`, given a fit handle, resolves the score identity for its
settings and looks it up. On a hit it reads `score.json`. On a miss it derives,
stores the leaf, and reads it. Its `--particles`, `--replicates`, `--draws` and
`--seed` options remain and select a `custom` settings level. `compare` no
longer writes to the temporary directory for traces.

`compare` also accepts a path to a `score.json` or to a score leaf. It continues
to accept a bare trace file, for a score computed outside camdl's derivation (an
ODE fit, gh#312); such a row has no identity and every check that needs one
reports it as not recorded, as today.

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
    pub elpd: f64,
    pub mc_se_elpd: Recorded<f64>,     // NotRecorded for one replicate
}
pub struct StreamSummary {
    pub leaf: String,
    pub n_scored_steps: usize,
    pub elpd: f64,                     // sum of this leaf's marginal log scores
    pub mc_se_elpd: Recorded<f64>,
    pub mean_crps: StreamMean,
    pub expected_coverage_50: StreamMean,
    pub expected_coverage_90: StreamMean,
    pub pit_histogram: PitHistogram,   // 10 equal bins, nonrandomized
    pub lead_span: LeadSpan,
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

Each step of the trace gains `date: Recorded<String>` beside `t`. Dates are
computed once, at scoring time, by `ir::caltime::internal_to_date_hires`
(`caltime.rs:250`), which stays one-to-one with a time that falls inside a day.
For a model with no calendar origin every `date` is `NotRecorded` and `origin`
is `NotRecorded`; the page then shows model time with its unit. The training
boundary and the scoring window are therefore stated as dates wherever the model
makes that possible, and the viewer converts nothing.

### Scored lead time and scored steps

```rust
/// How far ahead of the last conditioning observation each scored prediction
/// was made.
pub enum ScoredLeadTime {
    /// Each scored value was predicted from every observation, of every
    /// assimilated stream, at earlier times; the filter then assimilated it.
    OneObservationAhead,
}

/// The gap, in model time, between a scored observation and the latest earlier
/// observation the filter had assimilated, over one leaf's scored steps.
pub struct LeadSpan { pub min: f64, pub max: f64, pub time_unit: String }
```

Both live in the `score` crate. `ScoreArtifact.lead` is the artifact's lead
time; each `StreamSummary` records `n_scored_steps` and its `lead_span`, because
"one observation ahead" is one week for a weekly stream and one day for a daily
one, and shorter still when another stream reports in between. The first scored
step of an in-sample score that starts from the prior has no earlier
observation; it is left out of the span and is already flagged by the trace's
`StartsAtPrior` warning.

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
new value as unknown. Nothing in major 1 has to change for it. Until then every
artifact is `OneObservationAhead`, and a free-running projection is `NotScored`.

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

### The PIT is stored as its endpoints

Trace schema 6 replaces `StreamScore.pit` with

```rust
pub pit_lower: f64,   // P̂(X < y_obs)
pub pit_upper: f64,   // P̂(X ≤ y_obs)
```

computed from the predictive draws, and deletes `pit_randomization_seed` and the
draw. For a continuous predictive the two are equal up to ties in the sample.

Coverage at level ℓ becomes the expected coverage: for each scored value, the
fraction of `[pit_lower, pit_upper]` lying inside the central interval
`[(1−ℓ)/2, (1+ℓ)/2]`, taken as 1 or 0 when the endpoints are equal, averaged
over scored values. It is the expectation of today's statistic over the uniform
draw and does not depend on a seed. The histogram is the nonrandomized PIT
histogram of Czado, Gneiting and Held (2009, "Predictive model assessment for
count data", _Biometrics_ 65(4):1254–1261,
doi:10.1111/j.1541-0420.2009.01191.x): each scored value contributes a function
that rises linearly from 0 at `pit_lower` to 1 at `pit_upper`, these are
averaged over scored values, and the height of a bin is the difference of that
average across the bin. Both are computed at scoring time and stored in
`StreamSummary`, because the viewer computes no statistic (R3). This closes
gh#1049 for the artifact and for `compare`.

### Quantiles replace draws

Trace schema 6 replaces `y_pred_samples` and the five-quantile `interval` with
`quantiles: Vec<f64>` at the levels listed once in the trace's header,
`quantile_levels`. The levels are the 23 the COVID-19 Forecast Hub required for
its death and hospitalization targets: 0.01, 0.025, 0.05 to 0.95 in steps of
0.05, 0.975 and 0.99, which are the median and the ends of 11 central prediction
intervals (the hub's submission instructions, `reichlab/covid19-forecast-hub`,
`data-processed/README.md`; its incident-case target asked for 7 quantiles
only). The hub scores forecasts from those intervals with the weighted interval
score, which approximates the CRPS (Bracher, Ray, Gneiting and Reich 2021,
"Evaluating epidemic forecasts in an interval format", _PLOS Computational
Biology_ 17(2):e1008618, doi:10.1371/journal.pcbi.1008618). Here the set is how
a predictive distribution is stored, not how it is scored: CRPS and the PIT
endpoints are computed from the draws before they are discarded. The principles
ask for a dense quantile set so that export to hub formats stays possible.
`--no-save-samples` and the `SamplesNotSaved` warning are deleted.

_Not verified:_ the size of the pooled draw array a combined trace carries today
(`combine_mixture` pools across draws); it was not measured. The change is made
for the fixed, publishable size, whatever that number is.

Readers of the draws on `main` (`fit predict`'s one-step band, camdl-scope) read
the quantiles instead. camdl-scope was not read for this proposal.

### Trace schema 6 and older traces

Schema 6 is: PIT endpoints, quantiles, step dates. `camdl` reads schema 6 only
and refuses an older trace with a message to re-derive it. This follows the
repository's alpha rule (`VERSIONING.md`): local traces are cheap to recompute,
and none has been published. The read-forever commitment begins with
`camdl.score` major 1, which contains a schema-6 trace.

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
    pub method: String,
    pub backend: String,
    pub convergence: ConvergenceRecord, // defined in proposal B
}
```

`ConvergenceRecord` holds the verdict as judged when the score was computed, the
rule it was judged by, and the diagnostics needed to judge it again with the
version of their computation. Proposal B defines it and says how `compare` uses
it. It is here so that a score artifact read without its fit still says whether
the sample behind it converged.

## Invariants, as tests

Each is a test that fails if the property is lost.

1. **Self-description.** A `ScoreArtifact` serialized and read back with no
   other file present yields the same evidence label, against a second such
   artifact, as the two computed in one process.
2. **New data, same fit.** Appending rows after the training boundary to a
   `holdout_after` file leaves the fit's run id unchanged and changes the score
   run id. The same for replacing a `[data.holdout]` file.
3. **Training data still keys the fit.** Changing one training value changes the
   fit's run id.
4. **Reformatting.** A file and a copy with columns reordered, tabs for commas,
   an unbound column added, `10` written as `10.0`, and CRLF line endings have
   equal row digests per leaf, and the fit's run id is equal.
5. **Holes are data.** Replacing a value with `NA` changes the row digest;
   deleting the row changes it differently.
6. **Row digest is pinned.** A committed fixture file has a committed digest per
   leaf; the test fails if the digest changes without `ROW_DIGEST_VERSION`
   changing.
7. **Store round trip.** `camdl compare` run twice on the same fits spawns no
   filter pass the second time, and prints byte-identical output.
8. **`score` then `compare`.** `camdl score` followed by `camdl compare` at the
   same settings reads the stored leaf and derives nothing.
9. **Prefix check.** `camdl score --data` with a file whose rows up to the
   training boundary differ from the fit's is refused, and under the override is
   recorded in the trace.
10. **Coverage is seed-free.** Two artifacts from the same predictive draws have
    equal `expected_coverage_90` and equal histograms. On a fixture of tied
    counts the stored value equals the value computed by hand.
11. **Endpoints.** `pit_lower ≤ pit_upper` at every scored value, with equality
    on a continuous fixture.
12. **Dates.** For a model with an origin, `first_scored.date` equals the date
    in the data file's first scored row. For a model without one, every date is
    `NotRecorded` and the artifact still validates.
13. **Lead span.** For a weekly single-stream fixture, `lead_span.min` and
    `.max` both equal one week in model time.
14. **Profile by value.** A custom run with the profile's values is compatible
    with a profile run; changing the draw count is refused; changing the seed is
    reported and not refused.
15. **Major version.** A document with `schema.major = 2` is refused by name. A
    document with an unknown `ScoredLeadTime` value is read, and that value is
    reported as unknown.
16. **No statement is hashed.** Adding, changing or removing `cases.meta.toml`
    changes no run id.
17. **No sum across streams.** No field of the artifact equals the sum of
    `StreamSummary.elpd` on a two-stream fixture, and none is named as one.

## What this proposal does not do

- It defines no predictive artifact. Only the lead-time types those will carry.
- It defines no bundle, no `camdl publish`, and no private-data mode.
- It does not build forecast-mode scoring (gh#808) or the ODE score (gh#312). A
  fit on the ODE backend has no derived score artifact and is not comparable
  through one.
- It does not store what a joint score over a subset of streams would need.
- It does not settle the publish profile's values.
- It does not rename `fit predict --horizon`.
- It does not change how a filter pass is run, or `particle_filter.rs`.
- It does not change the checks `compare` runs, beyond reading the same facts
  from the artifact. The report, the convergence refusal and the draw-count
  check are proposal B.

## Steps

Each step is one commit with its tests and a `Proposal:` footer. Steps 1 to 3 of
proposal B (the `score` crate) come first; the types below are created there.

1. `runid::row_digest`, with `ROW_DIGEST_VERSION` and the canonical form above,
   computed from the loader's per-leaf rows. Tests 4, 5, 6.
2. Fit identity: `FitDigest.data` as training-row digests, `holdout_data`
   deleted, `[data.holdout]` removed from the blob, the other data-folding
   identities switched, `HASH_VERSION` raised together with the D25 change. The
   commit states what the bump invalidates. Tests 2 (fit half), 3. This step
   needs the maintainer's confirmation before it lands, as every identity change
   does.
3. The `.meta.toml` reader; `dataset` and `as_of` in the fit record;
   `docs/camdl-data-spec.md`. Test 16.
4. Trace schema 6: PIT endpoints, expected coverage, the nonrandomized
   histogram, quantiles, step dates. `pit_coverage` and every reader updated.
   Tests 10, 11. Closes gh#1049.
5. `ScoreArtifact` and its parts in `score`; `ScoredLeadTime`, `LeadSpan`,
   `LeadTimeStatus`; the profile. Tests 1, 12, 13, 15, 17.
6. `ArtifactKind::Score`, its resolver (`cli/src/score_cas.rs`), and
   `camdl score`. `docs/camdl-run-spec.md` §9.2.4, §9.3 and a new §10 entry for
   `score.json`. Tests 2 (score half), 9.
7. `compare` reads and derives through the store. Tests 7, 8, 14.
8. `docs/commands.md` and `docs/methods/model-comparison.md`.

## Verification

Commands run on `main` at `e8e7c24a0` for the claims the design rests on.

- `grep -n 'Sim =>\|FitMethod =>\|Pfilter =>\|Survey =>\|ProfilePoint =>\|Obs =>\|Projection =>\|SimEnsemble =>' rust/crates/runid/src/kind.rs`
  → eight lines, indices 0 to 7: no score kind, and indices are explicit.
- `grep -n 'no-store' rust/crates/cli/src/compare.rs` → line 2438, inside the
  `Command::new(&exe)` built at 2428: `compare` spawns `pfilter` and stores
  nothing.
- `grep -n 'holdout_data' rust/crates/cli/src/fit/cas.rs` → line 490,
  `holdout_data: data.identity_holdout()`.
- `grep -n 'digest_bytes' rust/crates/cli/src/fit/stream_bindings.rs` → line
  114: digests are of file bytes.
- `grep -n 'd.particles != \|d.replicates != ' rust/crates/cli/src/compare.rs` →
  lines 2118 and 2121, and no line comparing draws or seeds.
- `grep -n 'HASH_VERSION: u16' rust/crates/runid/src/hash.rs` → line 77,
  value 2.
- `gh issue view 808 --json state` → `OPEN`; `gh issue view 1049 --json state` →
  `OPEN`.
