# Proposal: fit labels that move, handles that name runs, one chain selector

**Status:** Proposal **Date:** 2026-11-14 **Motivation:** A modeler iterating on
a fit cannot reuse its label, cannot find one run's files from its label, and
selects chains through two unrelated mechanisms. Each is a contained change to
existing seams.

## Background: what exists today

This section is the surface the proposal changes, so the proposal can be read
without the code open. File references are to `rust/crates/cli/src/` unless
another crate is named.

**Fit labels.** A label is provenance, never identity (`camdl-run-spec.md`
§6.8): it is stored on one run's `run.json` as `provenance.label` and is not
hashed into any `run_id` (`runid/src/record.rs`). For fits, today a label names
exactly one run and never moves (gh#896):

- `fit run --label L` is refused before any sampling when another fit run that
  has not failed holds `L` (`check_label_is_free`, `fit/mod.rs`), when the run
  is already stored under another label (a cache hit on a run labelled `M`), or
  when the invocation writes several runs. A cache hit on an unlabelled run
  writes `L` onto it. The check-and-write runs under `fit_label_lock`
  (`cas_sink.rs`, an exclusive lock on `<root>/.fit-labels.lock`).
- `camdl label <run_id> L` on a fit run refuses a held label, a run already
  carrying a different label, and a run still recorded as running (`fit/mod.rs`,
  `cmd_label`; `cas_sink.rs`, `set_fit_label`).
- A run started from a labelled source (`from_mle`, `from_posterior`) records
  the source run's `run_id` in its `deps` and in `run.json`
  `inputs.starts_source`, and `fit summary` prints it beside the handle
  ("from_mle @scout, run 3a63b6"), so the source is identity whatever happens to
  the label.

**Labels on other kinds** mean something different. Sims, particle filters and
surveys carry `provenance.label` too, but it is a free-text tag: it is not
unique (`simulate --label` with several seeds labels every run), `camdl label`
replaces it freely, and no handle resolves it (`resolve_label` walks fit leaves
only). Profiles keep their label on the base sidecar (`fit.meta.json`).

**Handles.** `fit/handle.rs` (`FitRef`) classifies a typed fit reference by
syntax: `@label`; a `*.toml` config; an existing directory (a run's leaf, or a
problem folder holding several runs); otherwise a hash prefix (a run's `run_id`,
else the fit id of a problem folder). `@label` resolves through `resolve_label`
to the one completed run carrying it; two or more is an ambiguity listed
git-style. Every fit verb resolves through this module.

**Browsing.** `camdl list`, `camdl show` and `camdl cat` live in `browse.rs`.
For fits:

- `list` prints one row per problem folder (model + data + config without the
  method), all its runs' labels joined in one cell; `list --format json` emits
  one `fit_list_row` (schema v2) per folder with a `methods[]` array.
  `list --all` lifts the row limit (it conflicts with `--limit`).
- `show` and `cat` resolve through `browse.rs`'s own `resolve_any`, which maps
  `@L` to the problem folder with that run's id attached. `show` lists the
  folder's `predict/<run id>/`, `observed/`, `quantities/` and `fit.meta.json`,
  and never the run's leaf, so `draws.tsv` is absent. `cat` searches
  `predict/<run id>/` and then the folder, never the leaf, so
  `cat @L --stream draws.tsv` fails. `show` accepts `--format json` and ignores
  it: it has no JSON output.

**Chain selection.** Two types for one concept:

- _Read side_ (`chain_selection.rs`, `ChainSelection`): `--exclude-chains 3,5`
  on `fit summary`, `fit predict` and `fit table`, and per fit
  (`--exclude-chains @a:4`) on `compare`. One filter, `apply_keyed`; a
  non-quietable warning naming the dropped chains; a prediction made with a
  selection is stored under its own name (`predictive-excl3,5`, gh#795).
  Exclusion only. `parse_exclude` collapses a repeated id and refuses an empty
  token.
- _Start side_ (`fit/starts.rs`, `ChainSubset`, gh#1056):
  `starts = { from_posterior = "@base", chains = [0, 1, 2, 4] }` or
  `--starts-chains 0,1,2,4`, judging the source by the selected chains' own R̂.
  Inclusion only. `ChainSubset::new` refuses a repeated id; `parse_list` skips
  an empty token. Its identity value is `{"from_posterior": {"chains": [..]}}`.
- `simulate --draws posterior` and `cat` select no chains.

## Problems, from one worked case

The London measles guide (`docs/guide/measles.qmd`, a Snakemake pipeline in
`docs/guide/measles/`, on branch `docs/measles-guide`, landing alongside this
proposal) fits `m0_mh` (six chains, two of which settle in a mode with
negligible posterior weight), then `m0_mh_mode_a` from the four dominant chains
(`--starts from_posterior=@m0_mh --starts-chains 0,1,2,4`), then draws each
mode's predictions. It surfaced three frictions.

**1. A changed fit cannot keep its label.** A changed prior, model, data or
camdl build gives the fit a new identity, as it must, and
`fit run --label m0_mh` is then refused because the old run holds `m0_mh`. The
pipeline's recovery was deleting the results store. A modeler iterating "fit,
adjust, refit" under one name hits the same refusal.

**2. A label does not lead to its run's files.** The two measles fits share a
problem folder, so `camdl list` shows one row labelled "m0_mh_mode_a, m0_mh";
`camdl show @m0_mh` describes the folder and omits `draws.tsv`; and
`camdl cat @m0_mh --stream draws.tsv` fails. The reliable routes are the leaf
path `fit run` prints when it finishes, or
`fit summary --format json | jq .leaves[0].leaf`. The pipeline read chain traces
from the leaf directly, and a screenshot script addressing runs by folder showed
one fit's pair plot under the other's name.

**3. Chain selection is two types and covers half the cases.** One mode's
predictions or draws need "these chains only"; the read side offers only
exclusion, and `simulate` and `cat` neither.

## Design

### A. A fit label is a moving name with history

Scope: **fit runs only.** Labels on sims, particle filters and surveys stay
free-text tags, unchanged. That one word names two concepts is a known smell;
giving the non-unique one its own name is a follow-up (gh issue to be filed with
this proposal), not part of this change. Profiles keep today's rule.

**One definition.** A new field, `provenance.label_set_at` (an RFC 3339
timestamp with millisecond resolution, recorded and not hashed, like
`created_at`), is stamped every time a fit run is given a label. A label's
**current holder** is, among the fit runs carrying it whose live state is
`Completed`, the one with the latest `label_set_at`; ties are broken by
`run_id`. A run whose `run.json` has a label but no `label_set_at` (written
before this change) counts with its `created_at`.

Every path that gives a fit run a label stamps `label_set_at`, under
`fit_label_lock`, and that is the only way a run becomes a holder:

| Event                                                  | Result                                                                                    |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `fit run --label L`, new identity                      | The new run gets `L`, stamped when it completes.                                          |
| `fit run --label L`, cache hit, stored run unlabelled  | `L` written and stamped: the run becomes `@L`.                                            |
| `fit run --label L`, cache hit, stored run carries `L` | Re-stamped: the run becomes `@L` again (this is how reverting a change points `@L` back). |
| `fit run --label L`, cache hit, stored run carries `M` | The label is replaced by `L` and stamped; prints `label: run 3a63b6 was @M, now @L`.      |
| `camdl label <run_id> L`                               | The same as the cache-hit rows: the named run becomes `@L`, whatever its age.             |

The invariant every script can rely on: **the run you just invoked with
`--label L`, or labelled `L`, is `@L`.** When another run held `L`, the command
prints one line after the lock is released:

```
note: @m0_mh now names run 7c01e2 (previously run 3a63b6, completed 2026-11-14, max R̂ 2.18)
```

**How the record is written.** A label move rewrites exactly one `run.json`: the
run gaining the label. The run losing it is not touched, since the holder is
computed from the stamps. The write goes through `runid::update_record`, which
takes the run's leaf lock, re-reads the record under it, writes `run.json.tmp`,
fsyncs it, renames it over `run.json` and fsyncs the directory, so a crash or a
concurrent reader sees the old record or the new one, never a partial file.
`provenance` is not hashed and `run.json` is never a dependency digest, so no
identity or cache key changes. `update_record` hands its caller the whole
record; every label path goes instead through one narrow function,
`set_fit_label(leaf, label, label_set_at)`, that can change only those two
provenance fields, so a mistaken edit of an identity field cannot compile.

**Refusals that remain:**

- An invocation that writes several runs (several `fit_seeds`, synthetic
  datasets or sweep points) cannot take one label.
- A label whose holder-candidate is _live_ — a fit run carrying `L` whose
  `store::leaf_state` is `Running` or `OwnerUnknown` — refuses
  `fit run --label L` and `camdl label … L`, naming the running run. Two runs in
  flight under one name is a race. A run whose state is `Interrupted` (its
  record says running but its lock is gone) or `Failed` never holds or blocks a
  label; this replaces today's `status != Failed` test, under which a killed run
  held its label forever.
- `camdl label` keeps refusing to label a live running run.
- A run carries at most one label (a new label replaces the old, as above).

**History.** `@L~N` names the Nth earlier holder: the completed runs carrying
`L`, ordered by `label_set_at` descending, `@L~0` being `@L`. Grammar:
`@<label>(~<N>)?`, N a non-negative integer. `~` is outside the label alphabet
(`^[a-zA-Z0-9 ,._-]{1,64}$`), so it cannot collide with a label. `@L~` or `@L~x`
is a parse error naming the form; `@L~N` past the end is an error listing the
history (run id, `label_set_at`, verdict). History is derived from the runs that
still carry `L`, so relabelling one of them, or rerunning one with `--force`
(which quarantines the old leaf; the walker skips quarantine), renumbers the
rest. `camdl list --history` makes the current numbering visible.

**Every resolution says what it resolved to.** A move notice reaches only the
terminal that made the move, so every verb that resolves `@L` or `@L~N` prints
to stderr `@m0_mh → run 7c01e2 (results/fits/…)`, followed by
`(2 earlier holders)` when the history is non-empty. A `fit.toml` that writes
`from_posterior = "@m0~1"` records the handle as typed and the resolved run id
(already true), and the run id is what identity uses.

**Provenance and caching are unaffected.** A label was never identity. A
downstream run records its source's `run_id` in `deps`, so moving `@m0_mh`
cannot change what an existing `m0_mh_mode_a` was started from. Rerunning the
downstream command after the source moved resolves the new source run id, which
is a new dependency, so _camdl_ re-keys and reruns the downstream fit. A
pipeline tool (Make, Snakemake) reruns a rule only when that rule's declared
inputs change; a pipeline that wants the downstream rule to follow a label move
declares an input that changes with it, such as the upstream run's
`show --format json` record (below).

**Design choice recorded: the pointer lives on the runs.** The alternative is a
store-level append-only label log (git's refs and reflog): stable history,
atomic moves, but new mutable state beside the runs that can disagree with them
(a deleted leaf), where the store's principle is that `run.json` is the source
of truth and `index.json` only a cache. `label_set_at` keeps one source of truth
at the cost of history that renumbers on relabel, which is stated and visible.
If stable history is later needed (a `@L~3` cited in a paper), the log is the
upgrade path; `label_set_at` is its natural seed.

**Why the old rule existed, and why this keeps its purpose.** gh#896 made labels
immovable so that "no later label can obscure what a run was started from". That
guarantee is carried by `deps` and `starts_source.run_id`, not by the label.
What moving labels give up is that `@L` typed today and next month may name
different runs; the resolution line on every command, `@L~N`, and the recorded
run ids make that visible, and a script that must pin one run uses its run id.

### B. Every handle that names a run resolves to that run

**One resolver.** `show` and `cat` stop using `browse.rs`'s `resolve_any` for
fits and route through `fit/handle.rs`, which every other fit verb uses, so `@L`
means the same run everywhere. Handles that name a run: `@L`, `@L~N`, a leaf
directory, a `run_id` prefix. Handles that name a problem folder: a fit-id
prefix, the folder's path, a `fit.toml`.

**`camdl list`** prints fits one row per run, grouped by problem folder:

```
$ camdl list
fits
problem 76aecea7 · models/m0_seir_terms.camdl · configs/m0_mh.toml
  LABEL         RUN     METHOD  SEED  STARTS                                          STATUS     VERDICT
  m0_mh         3a63b6  mh      2     from_prior                                      completed  not converged (R̂ 2.18)
  m0_mh_mode_a  9f2c1d  mh      1     from_posterior run 3a63b6 (@m0_mh) chains 0,1,2,4  completed  converged (R̂ 1.006)
```

- `STARTS` prints the recorded source run id, with the handle as typed beside
  it, because the handle may since have moved.
- `VERDICT` is the run's stored verdict, the line `fit summary` prints: from the
  sampler's schema-tagged `<method>_summary.json`, or for IF2 and the optimizers
  from `fit_state.toml` through the gating verdict; a one-chain IF2 has none
  (`—`). A verdict that cannot be read renders as `unreadable: <reason>`; `list`
  never exits non-zero because one row is unreadable.
- Runs whose `leaf_state` is `Interrupted` are listed with that status.
- `--history` adds the superseded holders of each label, as `m0_mh~1`,
  `m0_mh~2`. `--all` keeps its meaning (no row limit), and `--limit` counts
  runs.
- Several seeds of one method in one folder collapse to one row with a `SEEDS`
  range and a combined verdict column, unless `--runs` is given.
- `list --format json` emits one object per run, `fit_list_row` schema **v3**:
  `{schema, problem {fit_id, model, config}, label, label_set_at, history_index,
  run_id, method, seed, starts {rule, handle, run_id, chains}, status, verdict,
  path}`.
  v2 consumers (scripts; camdl-scope reads the store directly and is checked in
  its own repository, follow-up issue) see a version change, not a silent
  reshape. _Implementation note:_ each row also carries `"kind": "fit"`, as the
  v2 row did, because `list --format json` interleaves the rows of every kind
  and a consumer tells them apart by `kind`.

**`camdl show <handle>`** on a run handle describes the run:

```
$ camdl show @m0_mh_mode_a
@m0_mh_mode_a → run 9f2c1d (results/fits/m0_mh-76aecea7/mh-8914c832/seed_1-e6a47b86)
run      9f2c1d  (@m0_mh_mode_a)
problem  76aecea7 · models/m0_seir_terms.camdl · configs/m0_mh.toml
method   mh · 6 chains · 100,000 iterations · seed 1
starts   from_posterior run 3a63b6 (@m0_mh), chains 0,1,2,4
verdict  converged · max R̂ 1.006 · min bulk ESS 1,329
path     results/fits/m0_mh-76aecea7/mh-8914c832/seed_1-e6a47b86
outputs
  draws.tsv                                  posterior draws (chain, draw, parameters)
  chain_<N>/trace.tsv                        every iteration of each chain
  predict/9f2c1d…/predictive/weekly_cases.tsv   posterior predictive bands
```

`outputs` is the run's recorded artifact set (`run.json` `artifacts`) plus its
predictions under the folder's `predict/<run id>/`, never a directory listing,
so a stray file is not reported as an output. On a problem-folder handle, `show`
lists the folder's runs (the rows `list` prints for it) and the handle that
selects each.

**`camdl show --format json`** is new: it emits one `fit_show` (schema v1)
object per run handle:
`{schema, run_id, label, label_set_at, history_index,
problem {fit_id, model, config}, method, seed, chains, starts {rule, handle,
run_id, chains}, status, verdict {kind, max_rhat, min_ess_bulk}, path,
outputs [{path, role}]}`,
with `role` from `run.json` `output_schema`. On a problem-folder handle it emits
`{schema, problem, runs: [<fit_show>…]}`. `show` for the other kinds (sims,
profiles, …) also honours `--format json` (today the flag is silently ignored);
their objects are specified in the same commit and are out of scope here only in
their field lists.

**`camdl cat <handle> --stream <name>`** resolves a run handle and looks for
`<name>` in the run's leaf, then in `predict/<run id>/`, then in the problem
folder (today's lookup is the last two only, so existing `cat` paths keep
working). A problem-folder handle with more than one run is an ambiguity listing
the runs, as `fit summary` does.

_Implementation note:_ a problem-folder handle holding several runs still reads
a stream that is a file of the folder itself (`fit.meta.json`, or a
`predict/<run id>/…` path), since that path names one file whichever run is
meant; any other stream is the ambiguity. Without `--stream`, `cat` on a run
handle prints the run's `fit_state.toml`, as it did for a leaf directory, so
`cat @L` and `cat <leaf>` agree; on a folder it prints `fit.meta.json`, as
before.

### C. One chain selector

`ChainSubset` and `ChainSelection` become one type in `chain_selection.rs`:

```rust
/// A validated set of chain ids of one fit and whether it keeps or drops them.
pub struct ChainSelection { ids: BTreeSet<ChainIndex>, polarity: Polarity }
pub enum Polarity { Keep, Drop }
```

- **Parsing**, once at the boundary, from `0,1,2,4` (CLI) or `[0, 1, 2, 4]`
  (TOML). An empty list, a repeated id, an empty token and a non-integer are
  refused. This tightens two cases: `--exclude-chains 3,3` (collapsed today) and
  `--starts-chains 0,,1` (an empty token skipped today) become errors.
- **Validation** against the fit when applied: an id the fit lacks is refused,
  as is a selection that keeps no chain. _Implementation note:_ a selection that
  drops no chain (a `--chains` list naming every chain of the fit) is refused
  too. Its normal form is the empty dropped set, which would write the whole
  cloud under a chain-subset address (`predictive-excl/`) beside the pooled one;
  refusing it keeps the dropped set non-empty, as `--exclude-chains` already
  guaranteed.
- **Normal form.** Once the fit's chain count is known, every selection is
  reduced to its dropped set, so `--chains 3,5` and `--exclude-chains 0,1,2,4`
  on a six-chain fit are the same selection: the same stored prediction name
  (`predictive-excl0,1,2,4`), the same `chain_selection` manifest
  `{excluded, kept, n_total}`, which carries no polarity.
- **One filter**, the existing `apply_keyed`, and the existing warning, worded
  by polarity ("keeping only chains 3, 5; dropping 0, 1, 2, 4"), always on
  stderr.
- **Start side.** `from_posterior`'s `chains` and `--starts-chains` take a
  `Keep` selection; the identity value stays
  `{"from_posterior": {"chains": [..]}}`, built from the kept ids and never from
  the struct, so gh#1056 runs keep their identity.

Flags:

| Command                                     | Flags                                                                                                                                                                                                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `fit summary`, `fit predict`, `fit table`   | `--chains IDS` (new) or `--exclude-chains IDS` (existing); never both                                                                                                                                                                                              |
| `compare`                                   | per fit, `--chains @a:0,1` (new) or `--exclude-chains @b:4` (existing). The two flags may name different fits in one invocation; naming one fit in both is refused; a bare (cohort-wide) value in either flag is refused when any keyed value is present in either |
| `simulate --draws posterior --fit <handle>` | `--chains` / `--exclude-chains` (new)                                                                                                                                                                                                                              |
| `cat <handle> --stream <name>`              | `--chains` / `--exclude-chains` (new), only for a stream whose recorded `output_schema` has a `chain` column (in practice `draws.tsv`); refused by name for any other stream                                                                                       |
| `fit run`                                   | `starts = { from_posterior = "@base", chains = [..] }` / `--starts-chains` (gh#1056, now this type, `Keep` only)                                                                                                                                                   |

Read-side selection never re-keys anything (unchanged). Start-side selection is
identity, as gh#1056 made it.

## Workflows

**A modeler iterating on one fit.**

```
$ camdl fit run fit.toml --label m0 --seed 2
  ✓ stored @m0  run 3a63b6  results/fits/…
$ camdl fit summary @m0
@m0 → run 3a63b6 (results/fits/…)
  …
  … edit a prior in fit.toml …
$ camdl fit run fit.toml --label m0 --seed 2
  ✓ stored @m0  run 7c01e2  results/fits/…
  note: @m0 now names run 7c01e2 (previously run 3a63b6, completed 2026-11-14, max R̂ 2.18)
$ camdl compare @m0~1 @m0              # the old prior against the new
  … revert the prior …
$ camdl fit run fit.toml --label m0 --seed 2
  cache hit: run 3a63b6
  note: @m0 now names run 3a63b6 (previously run 7c01e2, …)
```

**A multimodal fit.**

```
$ camdl fit summary @m0_mh               # per-chain log-likelihoods: chains 3 and 5 disagree
$ camdl fit predict @m0_mh --chains 3,5  # mode B alone; warning on stderr
$ camdl cat @m0_mh --stream draws.tsv --chains 3,5 > mode_b_draws.tsv
$ camdl fit run fit.toml --label m0_mh_mode_a --seed 1 \
    --starts from_posterior=@m0_mh --starts-chains 0,1,2,4 --allow-nonconverged-source
$ camdl list                             # both runs, one row each, verdicts side by side
```

**An agent.** Resolve each label once with `camdl show @L --format json`, read
`run_id` and `path`, and use the run id for the rest of the session; read the
stderr resolution line when using `@L` directly. `docs/agents.md` says so.

**A pipeline.** Labels are stable names a pipeline can hard-code. A changed
input reruns the fit (a pipeline target) and moves its label; a downstream rule
that should follow declares the upstream run's `show --format json` record as an
input, so it reruns when the upstream run id changes.

## Run identity

No identity changes: `label_set_at`, like `provenance.label`, is recorded and
not hashed; read-side chain selection is a view; start-side selection keeps
gh#1056's identity value. No `ir/VERSION` bump and no golden changes. The
`fit_list_row` JSON goes from v2 to v3 and `fit_show` is new; both are output
schemas, not identity.

## Tests and docs that change

Existing tests that assert the old rule, rewritten to the new one:
`tests/gh896_fit_handles_name_one_run.rs::a_label_names_one_run_and_never_moves`
and `::camdl_label_labels_a_fit_run_once`;
`tests/fit_experiment_management.rs::fit_label_workflow_persists_and_surfaces_in_table`
(step 3); the label assertions in `tests/fit_predict_e2e.rs`; the unit test of
`set_fit_label` in `cas_sink.rs`; the gh#1056 unit tests that name `ChainSubset`
(ported to the merged type); `tests/exclude_chains_e2e.rs` text assertions
(polarity wording).

Documentation, updated in the commit that changes the behaviour:

- `docs/camdl-run-spec.md` §6.8 (the label rule, refusals, `label_set_at`,
  `@L~N`), and its other statements that a label names one run.
- `docs/agents.md`: the `list`/`show`/`cat` section; resolve-once guidance.
- `docs/fit-toml.md`: the handle paragraph under "Where a chain starts" and the
  `chains` paragraph.
- `docs/workflow.md`, `docs/inference.md`: `--chains` beside `--exclude-chains`;
  the refit loop under one label.
- `docs/commands.md`: `list`, `show`, `cat`, `label`, the chain flags.
- CLI help: `--label` (`args/mod.rs`), `camdl label`, `cmd_label`'s docstring
  ("never moves").
- `CHANGELOG.md`: labels move; `list` rows are runs; `show --format json`.
- `docs/guide/measles/Snakefile`: drop the empty-the-store instruction.

## New tests

- Labels: a new-identity `fit run --label L` moves `L` and prints the note; a
  cache hit on a run carrying `L`, `M` or no label makes it `@L` (three cases);
  `camdl label` moves `L` to an older run and `@L` follows; `@L~1` resolves to
  the previous holder, `@L~0` to `@L`; `@L~`, `@L~x` and `~N` past the end are
  refused with the documented messages; a live running holder refuses; an
  interrupted one does not block; a multi-run invocation is refused; a
  downstream `from_posterior=@L` re-keys after `L` moves; a pre-change run with
  no `label_set_at` orders by `created_at`; every resolving verb prints the
  resolution line.
- Browse: one row per run for a two-run folder, with `STARTS` run ids;
  `--history`; seeds collapse and `--runs` expands; v3 JSON; an unreadable
  verdict renders and `list` exits zero; `show @L` and `show --format json` on a
  run and on a folder; `cat @L --stream draws.tsv`; existing
  `cat … --stream predictive/…` paths still work; `cat` on a two-run folder is
  an ambiguity.
- Chains: `--chains` and its complementary `--exclude-chains` give the same
  stored prediction; the two parse tightenings; both flags together refused;
  unknown id and keep-nothing refused; compare's flag combinations;
  `simulate --draws posterior --chains`; `cat --chains` on `draws.tsv` and its
  refusal on another stream; warnings on stderr; gh#1056 identity unchanged.

## Implementation order

Each step lands with its tests and docs, `make test` green, a `Proposal:` footer
citing this document.

1. Chain selector: merge `ChainSubset` into `ChainSelection` with polarity and
   the normal form; the two parse tightenings are the only behaviour change.
2. `--chains` on summary, predict, table and compare; then `simulate` and `cat`.
3. `show` and `cat` through `fit/handle.rs`; run-level `show`;
   `show --format json`; `cat`'s lookup order.
4. Run-level `list`, `--history`, seed collapsing, v3 JSON.
5. Moving labels: `label_set_at`, the holder rule, the cache-hit table, `@L~N`,
   the liveness refusal, the resolution line; spec and docs in the same commit.
6. The measles pipeline drops its empty-the-store instruction.

Follow-up issues filed with this proposal: a distinct name for the non-unique
label on sims, particle filters and surveys; camdl-scope's handling of
`fit_list_row` v3 and moving labels, including a `?label=` deep link.
