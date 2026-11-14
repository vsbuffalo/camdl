# Proposal: compare fits on chosen observation streams

**Status:** Accepted **Date:** 2026-11-13 **Motivation:** a modeler who wants to
know how much an observation stream contributes to prediction fits the model
with and without it and compares the two fits on a stream they share;
`camdl compare` refuses that comparison today, because its only score is the
joint score over every stream a fit bound.

## What exists today

### Streams, sources and leaves

A model's `observations { }` block declares observation streams. One declared
block can expand into several concrete streams: an indexed block
`cases_by_patch` over patches becomes `cases_by_patch_north`,
`cases_by_patch_south`, and several blocks can read one data source through
`from <label>`. This document calls the label a data file is bound to the
**source**, and each concrete stream a **leaf**. For a plain stream the two
names are the same string.

A fit config binds data files by source (`[data.observations]`, checked by
`check_bound_sources` in `rust/crates/cli/src/fit/runner.rs`), or binds every
source to one file with the shorthand `[data] file = "..."`. A fit may bind only
some of the sources its model declares: `resolve_and_load_obs_streams` keeps the
observation blocks whose source is bound and ignores the rest. The shorthand
always binds every source.

### The trace

`camdl compare` works from one **prequential trace** per fit: the record of
one-step-ahead predictive scores a particle filter produced while passing over
the data (`PrequentialTrace`, `rust/crates/sim/src/inference/prequential.rs`).
Each scored time step (`PrequentialStep`) carries two kinds of score:

- A **joint** score. `log_score` is the log of the particle average of the
  product of the likelihoods of every leaf observed at that step.
- A **per-leaf** breakdown, `per_stream: Vec<StreamScore>`, present since trace
  schema 2. For each leaf observed at the step it holds that leaf's own
  `log_score` (the log of the particle average of that leaf's likelihood alone),
  `crps`, `pit`, `interval`, predictive draws, observed value and `coverage`
  (the window the observed value was accumulated over). `StreamScore.stream` is
  the leaf name.

The comparison table is built from the joint score only. The per-leaf scores
leave `compare` only through `--pointwise`, a TSV of per-step differences.

### The refusals

Because the joint score covers every leaf a fit bound, two fits that bound
different sets score different quantities. gh#570 recorded the consequence: a
model scored on six extra observations was reported as "overwhelming against"
purely for having been scored on them. Since then
`check_shared_observation_axis` (`rust/crates/cli/src/compare.rs`) refuses any
pair whose steps do not carry the same leaf names, with no override, and
`check_shared_data` refuses fits whose recorded data hashes differ, including a
stream bound by one fit only, unless `--allow-data-mismatch` is passed.

Those refusals are correct for the joint score and stay as they are. The cost is
that the stream ablation experiment — one model fitted to `cases` and `deaths`,
and to `cases` alone, compared on `cases` — cannot be run through `compare`.

## Prerequisite: one resolution of stream names

Three places name a stream, and they do not agree. The trace uses leaf names.
`[data.observations]`, `[data.holdout]` and the fit record's `data_hashes` under
that form use sources. Under the one-file shorthand, `fit run` expands by source
(`fit/runner.rs`, the `source_labels` expansion) while the fit record's
`data_hashes` (`resolved_data_hashes`, `fit/mod.rs`) and `compare`'s
`resolve_streams` expand by leaf name.

This proposal depends on a single typed resolution, built from the model and the
fit's `[data]` block, that answers for every bound leaf: its source, and the
data file bound to that source. `fit run`, the fit record, `compare`'s derive
step and the option below all read it, and `data_hashes` is keyed by source in
every config form. That change lands first, as its own commits, and is not
specified further here; this document assumes it under the name
`StreamBindings`.

## The design

`compare` gains one option, `--streams NAME[,NAME...]` (and `streams = [...]` in
`compare.toml`). When it is given, `compare` produces **one table per selected
leaf**, each scoring every row on that leaf alone. Fits that bound different
stream sets may be compared this way, subject to the checks below.

A name may be a leaf or a source. A source selects every leaf it expands to, in
the model's order. A name that is neither, in any row's model, is refused with
the list of available leaves and sources.

### The projection

The feature rests on one pure function,

```rust
/// The trace re-scored on one leaf.
fn project_onto_stream(
    trace: &PrequentialTrace,
    leaf: &str,
) -> Result<PrequentialTrace, String>
```

in `prequential.rs`. For each step of the input trace it finds the `per_stream`
entry for `leaf`. A step with no such entry is dropped: the fit observed only
other leaves at that time. Otherwise the step's `log_score`, `crps`, `pit`,
`interval`, `y_obs` and `y_pred_samples` become that entry's, copied unchanged,
and `per_stream` is reduced to that one entry. No score is recomputed and no
random number is drawn.

The projected score is exact: it is the fit's one-step-ahead predictive density
for that leaf, given everything the fit's filter had assimilated up to that
step.

Fields that describe the filter pass rather than the scored set are handled
explicitly:

- `ess` is carried over. It describes the filter at that step.
- `t0` and `score_from` are carried over, and the table footer's
  `Scored steps: n (t0=…)` drops the `t0` part under `--streams`, because `t0`
  indexes each fit's own time axis and means different things for fits with
  different stream sets.
- `PrequentialWarning::StartsAtPrior` is kept only when the projected trace's
  first step is the input trace's first step.
- `PrequentialWarning::EssCollapse` is recomputed over the surviving steps with
  the threshold the warning recorded, and dropped when none of them is below it.
- A new optional field `scored_streams: Option<Vec<String>>` on
  `PrequentialTrace` (serde default, omitted when absent) is set to `[leaf]`. A
  projected trace is never written by `compare`, but if one is ever serialized
  it cannot be read back as a joint trace. `check_shared_observation_axis`
  refuses a pair whose `scored_streams` differ.

The function refuses, naming the row and the leaf, when the leaf appears in no
step, when the trace has steps but no `per_stream` breakdown (a schema 1 trace),
or when no step survives.

### Where the projection sits

A derived trace may be a mixture over posterior draws and a combination over
replicate filter runs (`combine_mixture`, `combine_scored_traces`). Both combine
each `per_stream` entry's `log_score` by log-mean-exp across components, which
is the leaf's marginal predictive under the mixture. The projection therefore
runs on the fully combined trace.

`derive_prequential` computes each row's filter-noise Monte-Carlo standard error
(`mc_se`) from the replicate traces' total joint `elpd`, before combining them.
Under `--streams` it returns, per selected leaf, the standard error computed
from the replicates' projected totals. The selection is passed to `load_trace`
and on to `derive_prequential` as its own argument, not as a field of
`DeriveSettings`, because it applies to explicit `prequential.json` rows too.
For an explicit row the projection runs in `load_trace` and there is no `mc_se`.

`cmd_compare` then runs its existing pipeline once per selected leaf, on rows
whose traces are the projections for that leaf: the conditioning and provenance
preflights, the T_score check, `check_shared_observation_axis`,
`check_shared_coverage`, the data check, baseline choice, `paired_delta` with
its Newey–West standard error, and the renderers. A refusal for one leaf is a
refusal of the whole command and names the leaf.

### Fits that assimilated different streams must be scored held-out

A row's **assimilated set** is the set of leaves its fit bound: from
`StreamBindings` for a derived row, and unknown for an explicit
`prequential.json`. An unknown set is treated as different from every other.

When the rows' assimilated sets differ, the comparison is the ablation
experiment, and scored in-sample it is biased by construction. The fit trained
on `cases` alone chose its parameters to fit `cases`; the fit trained on `cases`
and `deaths` traded the two off, so on the training data it fits `cases` no
better and usually slightly worse, whatever `deaths` contributes to prediction.
The in-sample difference therefore leans toward the fit trained on fewer
streams. Held-out scoring removes this, because neither fit's parameters were
tuned to the scored rows.

So under `--streams`, when assimilated sets differ and the rows are stamped
`InSample`, `compare` refuses (exit 2) and says to declare the same
`holdout_after` in every fit. `--allow-in-sample-ablation` renders anyway, and
every format then carries a caveat stating the direction of the bias. When the
rows are stamped `HoldOutTail`, or when all rows assimilated the same set, no
flag is needed.

A held-out difference between rows with different assimilated sets still
combines two effects: parameters that were trained on the extra stream, and a
filter that assimilates the extra stream's held-out rows as it goes. That is the
predictive each fit would actually issue, and it is what the table reports. The
methods page says so.

### What each output states

- Each table is headed by the leaf it scores.
- When assimilated sets differ, the table and markdown footers list each
  distinct set with the rows that have it ("`full` assimilated cases, deaths;
  `cases_only` assimilated cases").
- Under `--streams` the JSON document is
  `{ "streams_scored": [...], "tables":
  [...] }`, where each element of
  `tables` is the document `compare` emits today plus `"stream"`, and each row
  gains `streams_assimilated` (a list, or `null` when unknown). Without
  `--streams` the JSON is unchanged.
- `--pointwise` under `--streams` writes rows for the selected leaves only, with
  `scope=stream`, and its header comment says so.

### The data check under `--streams`

For the table of a leaf, `check_shared_data` compares the recorded hash of that
leaf's source, found through `StreamBindings`, across rows. A row that carries
digests but has none for that source is refused, not skipped: an absent key must
not read as "no difference". `--allow-data-mismatch` overrides as today. Other
sources are not compared, and the existing "not checked" note names them. The
same applies to holdout-file hashes.

The result of the data check is computed once per table in `cmd_compare` and
passed to the renderers. Today `warning_lines` and `render_json` each call
`check_shared_data` again on the unrestricted entries, which would print a
mismatch the restricted preflight had cleared.

### What does not change

- Without `--streams`, every check and every number is as today. The stream-set
  refusal's message gains one sentence when the two rows share at least one
  leaf, naming the option and the shared leaves. It does not suggest
  `--allow-mismatched-horizon`, which skips the observation-axis check.
- The conditioning check is untouched: rows must all be in-sample or all
  held-out, with the same training boundary.
- The selected leaf must be scored at the same times, over the same windows, in
  every row. When the first scored step differs between rows that assimilated
  different sets, the refusal adds that a fit which conditions on its first
  observation (`ic_free`) starts scoring one step later on its own time axis.
- Which streams a fit binds is decided by the fit config. This proposal adds
  nothing to `fit run`.

### What cannot be ablated

Binding is by source, so a fit cannot drop one leaf of an indexed block or one
of several blocks sharing a `from` source. A fit using the one-file shorthand
binds every source; the ablated fit must use `[data.observations]`.

### No figure combines streams

Per-leaf scores are marginal: each is the predictive density of one leaf with
the others integrated out. Their sum over leaves is not the joint density of
those leaves, and nothing in a trace can turn one into the other. So under
`--streams`:

- No table, footer or JSON field reports a sum, mean or total across leaves.
  With two or more selected leaves the output is two or more separate tables and
  nothing else.
- The JSON `tables` array has no sibling aggregate.
- The `--pointwise` TSV carries no `scope=joint` rows.

If a combined figure is added later (below), it gets its own column and field
name — never `elpd`, `delta_elpd` or `lr` — and its own label saying it is a sum
of marginal scores.

## Not in this proposal

**One combined table over several leaves.** The trace stores each leaf's
marginal score, not the per-particle likelihoods the true joint over a subset
would need. A combined table built from a sum of marginal log scores would have
no valid likelihood ratio, evidence label or PIT column. A later flag may add
it, with those columns blank, without changing anything specified here.

**The existing joint CRPS and PIT.** For a multi-stream fit they are computed on
the sum of all leaves' counts while the log score is the joint density. This
proposal neither uses nor changes them: every number in a `--streams` table
comes from one leaf's own entry.

## Alternatives considered

**An override flag on the stream-set refusal.** Rejected: it would difference
joint scores over different stream sets, which is the gh#570 defect with a flag
on it.

**Re-running the filter on the shared streams only.** This scores the larger
fit's parameters through a filter that never assimilates the extra stream, which
is not the predictive that fit makes. It would isolate the parameter effect from
the assimilation effect, and may be worth its own option later; it is not the
experiment's primary answer.

**A keyword for "the streams all rows share".** Rejected in favour of explicit
names: the scored set is then visible in the command, and a stream dropping out
of the shared set through a typo in one fit config cannot change what a script
computes. The refusal message supplies the shared leaves.

## Implementation

Each step is one commit, cites this document in a `Proposal:` footer, and lands
with its tests. The `StreamBindings` prerequisite lands before step 1.

1. `project_onto_stream` and the `scored_streams` field in `prequential.rs`.
   Tests: the projection copies the entry exactly; steps without the leaf are
   dropped; each refusal; the two warning rules; projecting a one-leaf trace
   onto its leaf reproduces the trace's `elpd`; a trace without the field still
   deserializes.
2. `--streams` on `CompareArgs` and `streams` on `CompareToml`; name resolution
   through `StreamBindings`; the selection threaded through `load_trace` and
   `derive_prequential`; per-leaf `mc_se`; the per-leaf loop in `cmd_compare`.
   Test: per-leaf `mc_se` equals the value computed by hand from projected
   replicate traces.
3. The data check computed once and passed to the renderers, restricted per
   table, refusing an absent key. Tests: two rows whose `deaths` hashes differ
   and whose `cases` hashes agree are refused without `--streams` and pass with
   `--streams cases`, with no mismatch warning in any format; a row lacking the
   selected source's hash is refused.
4. The assimilated set, the in-sample refusal, `--allow-in-sample-ablation` and
   its caveat. Tests: refusal when sets differ in-sample; no refusal when they
   match; the caveat in every format under the flag; an explicit trace counts as
   a different set.
5. Table headings, footer lines, the JSON shape, the `--pointwise` header and
   the hint in the stream-set refusal. Test: every format states the scored leaf
   and each row's assimilated set.
6. An end-to-end test in `rust/crates/cli/tests/`: one two-stream model fitted
   twice with `[data.observations]` and the same `holdout_after`, once binding
   both sources and once binding one. `compare` refuses the pair and the message
   names `--streams <shared>`; with it the table renders held-out, and each
   row's `elpd` equals the sum of that leaf's scores in the `--pointwise`
   output. The same pair without a holdout is refused, and renders with the
   caveat under `--allow-in-sample-ablation`.
7. `docs/methods/model-comparison.md` (the text behind `camdl compare --explain`
   and `camdl docs model-comparison`): the option, what a per-leaf score is, the
   in-sample bias, and the two effects a held-out ablation difference combines.
