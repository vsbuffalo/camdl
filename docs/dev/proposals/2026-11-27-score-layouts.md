# Score files: one layout for data rows, one for forecast targets

Status: accepted.

Amends: `2026-11-23-prediction-and-scores.md`, its "Output" section and the
`lead` column described under "Types" and "`compare` reads stored scores".
Everything else in that proposal stands.

## Summary

`camdl score` writes every mode's scores in one layout, `scored.tsv`, with one
header for free-forward, one-step and forecast rows. This proposal splits the
score into two layouts that share a core of columns:

- **data-row scores**, for free-forward and one-step predictions, which predict
  the data's own rows;
- **forecast-target scores**, for forecasts, which predict the periods after a
  forecast origin.

A one-step row gains `assimilated_through`, the end of the newest observation
the filter had assimilated before predicting it. Joint rows, which score all the
streams observed at one time together, move to files of their own. The score's
identity version and its schema tag change, so stored scores are recomputed.

## The problem

A prediction belongs to one of two kinds, and one stored score never holds both.
`camdl predict --mode` takes one value, or defaults to free-forward with
one-step, or to forecast when `--horizon` is given (`args/mod.rs:1638`), and a
score is stored inside the one prediction it scores. At predict time the two
kinds are already separate files, `predictive/<stream>.tsv` and
`forecast/<stream>.tsv`. Only the score puts them under one header.

That header carries columns that have a meaning in one kind and not the other,
and the summary groups by one of them.

1. **`lead` on one-step rows.** A forecast target's lead is its end minus the
   forecast origin. A one-step row has no origin, and the score writes the
   length of the row's own period as its lead (`predict.rs:5972`), or for a row
   that measures an instant, the time since the stream's previous row. With
   streams at different cadences the number is wrong: in a fit with a daily and
   a weekly stream, the week of 9–15 February is predicted after the daily rows
   for 9–14 February are assimilated, and is written with lead 7 days.
2. **The summary splits one stream.** `summary.tsv` groups by (scenario, sweep,
   stream, source, stratum, mode, role, lead) (`SummaryKey`,
   `score_command.rs:721`). A calendar-month stream's one-step rows have periods
   of 28, 29, 30 and 31 days, so their summary falls into up to four rows per
   role, and the summed one-step log score is divided among them. An irregularly
   sampled instant stream gets a summary row per distinct gap.
3. **Two meanings of `lead` in one column.** `compare` reads `lead` only for
   forecast scores and gives every other mode one lead group
   (`compare/stored.rs:955-970`), and `compare --help` calls the column
   "forecast only" (`args/mod.rs:3502`), while the score writes it for one-step
   rows. A reader who stacks a one-step score under a forecast score and groups
   by `lead` pools one-step rows with lead-7 forecast targets, which the
   forecasting page (§4.4) says are not comparable.
4. **Joint rows among per-stream rows.** A row with `stream = joint` holds the
   log score of every stream observed at one time together
   (`score_command.rs:782-799`). Summing `log_score` over `scored.tsv` counts
   each observation twice.
5. **Stacking is not free under one header either.** A column that is empty in
   every row of a file is read as text by polars, so `pl.concat` of a one-step
   score and a forecast score fails with
   `SchemaError: type Int64 is incompatible with expected type String` (checked
   with polars on two two-row files) unless the reader passes
   `how="vertical_relaxed"` or explicit types.

Columns planned for one kind only would add to this. The reweighted one-step
score of held-out rows, which is being built, reports a weight effective sample
size and a Pareto k̂ per held-out row. A forecast target that began before the
origin (`docs/engineering/forecasting.md` §1.6) has a count already observed in
it. Under one header each such column is empty on every row of the other kind.

## Design

### The shared core

Both layouts begin with the same columns, in this order, with the meanings they
have today:

```
scenario [sweep:<p> ...] stream source <dims> t_start t_stop [date_start date_stop]
role parameters quantity mean q05 q25 q50 q75 q95 observed
log_score crps wis wis_dispersion wis_over wis_under pit in_50 in_90
```

Code that reads only these columns (a band plot against the data, a PIT
histogram, a coverage table) reads either layout unchanged.

### Data-row scores

For a free-forward or one-step prediction. Three files.

- `scored_rows.tsv`: one row per scored data row. The core, then:
  ```
  mode assimilated_through [date_assimilated_through]
  ```
  `mode` is `free_forward` or `one_step`. `assimilated_through` is empty on a
  free-forward row (below).
- `summary_rows.tsv`: one row per (scenario, sweep, stream, source, stratum,
  mode, role):
  ```
  scenario [sweep:<p> ...] stream source <dims> mode role n n_unscored
  log_score_sum log_score_mean crps_mean wis_mean coverage_50 coverage_90
  ```
- `joint_rows.tsv`: one row per observation time of a one-step prediction, the
  log score of every stream's row ending at that time, scored together:
  ```
  scenario [sweep:<p> ...] t_start t_stop [date_start date_stop] role streams
  assimilated_through [date_assimilated_through] log_score
  ```
  `streams` lists the streams scored, comma-separated. A free-forward score
  writes no joint rows, as now. Its summary, `joint_summary_rows.tsv`, has one
  row per (scenario, sweep, role) with `n`, `n_unscored`, `log_score_sum` and
  `log_score_mean`. Over held-out rows, `log_score_sum` there is the one-step
  score of all streams together, log p(held-out rows | training rows).

### Forecast-target scores

For a forecast. Three files.

- `scored_targets.tsv`: one row per scored target. The core, then:
  ```
  origin periods_ahead lead
  ```
- `summary_targets.tsv`: one row per (scenario, sweep, stream, source, stratum,
  role, lead), with the same score columns as `summary_rows.tsv`.
- `joint_targets.tsv`: one row per target end time, the log score of every
  stream's target ending then, scored together, with its `lead`; and one row
  with `lead` empty for all the forecast's targets together, which is the joint
  log score of the forecast. These are the joint rows the forecast writes today
  (`forecast_output.rs:755-765`). Its summary, `joint_summary_targets.tsv`, has
  one row per (scenario, sweep, role, lead).

A forecast score has no `mode` column: its file names say what it holds, and
`score.json` records the mode, as now.

### `assimilated_through`

On a one-step row, `assimilated_through` is the end of the latest row, of any
stream, that the filter had assimilated before it predicted this row. The
one-step filter, at each time where some row ends, first predicts every row that
ends then and then assimilates those with a value, so `assimilated_through` is
the latest end among the rows with a value at earlier stops. An `NA` row is not
assimilated and does not count. In a score, held-out rows are assimilated after
they are scored, so they count for the rows after them. A row predicted before
any row was assimilated has it empty.

```text
fit: weekly stream only
stream  t_stop   mode      assimilated_through
weekly  15 Feb   one_step  8 Feb

fit: daily and weekly streams
stream  t_stop   mode      assimilated_through
daily   15 Feb   one_step  14 Feb
weekly  15 Feb   one_step  14 Feb
```

It changes no prediction and no score. It records what each prediction was
conditioned on, so that `t_stop − assimilated_through` says how far ahead it was
made. A free-forward row is conditioned on no row, so its value is empty. The
column is written in model time, and also as a date when the file is dated, as
`t_start` and `t_stop` are.

### What the totals mean

Each row of `scored_rows.tsv` and `scored_targets.tsv` is one stream's row, and
both summaries are per stream. In one-step mode with several streams, each row
of a stream is predicted from every stream's earlier rows, so the per-stream
`log_score_sum` measures that stream given all the data before each of its rows.
It is not the score of the stream on its own, and the per-stream sums do not add
to the score of all the held-out rows, because rows ending at one time are
scored separately although they share one predicted state. That total is the
joint summary's `log_score_sum`. In forecast mode each target is predicted from
the training rows alone, so the per-stream totals have no such qualification.

## Types

The score's rows become a sum type, so a row cannot carry the columns of the
other kind:

```rust
/// The columns every scored row has.
struct CoreLine<'a> {
    scenario: &'a str,
    sweep: &'a [(String, f64)],
    stream: String,
    source: String,
    stratum: &'a [(String, String)],
    start: f64,
    stop: f64,
    role: RowRole,
    observed: Option<f64>,
    log_score: f64,
    rules: Option<RowScore>,
    unscored: bool,
}

/// One stored score's rows: a score holds one kind.
enum ScoredLines<'a> {
    DataRows(Vec<DataRowLine<'a>>),
    Targets(Vec<TargetLine<'a>>),
}

struct DataRowLine<'a> {
    core: CoreLine<'a>,
    mode: DataRowMode,                 // FreeForward | OneStep
    assimilated_through: Option<f64>,  // None on free-forward rows
}

struct TargetLine<'a> {
    core: CoreLine<'a>,
    origin: f64,
    periods_ahead: i64,
    lead: f64,
}
```

Today's `Line` (`score_command.rs:694`) has `mode`, `origin: Option<f64>`,
`periods_ahead: Option<i64>` and `lead: Option<f64>` side by side, and nothing
stops a one-step line from carrying a lead. `JointCell` (`fit/scoring.rs:113`)
splits the same way into a data-row joint and a target joint. The summary key
becomes one per kind: the data-row key has no lead, the target key has no mode.

The score's identity version (`fit_store.rs:480`, now 1) and its schema tag
(`camdl.score/v1`, `score_command.rs:35`) change, because the stored files
change. Stored scores are recomputed at about the cost of the predictions they
rerun.

`predictive/<stream>.tsv` writes a `periods_ahead` column that is always empty
(`predict.rs:777`, `:813`); it is deleted, in a commit of its own.

## `compare`

`compare` reads `scored_rows.tsv` or `scored_targets.tsv`, chosen by the mode
recorded in `score.json`, and the joint file of the same kind. Its table keeps
its columns. Its refusals keep their meaning: two scores of different kinds are
refused as two modes are now, and two forecast scores at different leads are
refused as now.

## Stages

1. Delete the empty `periods_ahead` column of `predictive/<stream>.tsv`.
2. Split the score's rows and summary into the two kinds, move joint rows to
   their files, change the score's version and schema tag, and point `compare`
   and `camdl show`/`cat` at the new files.
3. Add `assimilated_through` to one-step rows and joint rows.
4. Update the documents that name the score's files: `docs/camdl-run-spec.md`,
   `docs/commands.md`, `docs/engineering/forecasting.md` §4, and the amended
   sections of `2026-11-23-prediction-and-scores.md`, which gain a line pointing
   here.

## Tests

- A one-step score of a calendar-month stream has one summary row per role.
  Written first, against the current code, where it fails.
- A one-step score of a fit with a daily and a weekly stream writes no `lead`,
  and each weekly row's `assimilated_through` is the end of the daily row before
  it. A fit with the weekly stream alone writes the end of the previous week.
- An `NA` row does not advance `assimilated_through`; a held-out row does, for
  the rows after it.
- The joint summary's held-out `log_score_sum` equals the sum of
  `joint_rows.tsv`'s held-out `log_score`, and no per-stream file holds a joint
  row.
- A forecast score's files carry `origin`, `periods_ahead` and `lead` and no
  `mode`; a data-row score's files carry none of the three.
- `compare` of two one-step scores and of two forecast scores gives the numbers
  it gives today, and refuses a one-step score against a forecast score.
- A score stored before the change is not read as one of the new version.
