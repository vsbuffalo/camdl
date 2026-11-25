# Feature removals around the run store: assessment

Audit for `docs/dev/proposals/2026-11-25-stored-models-and-invocations.md`.

Read against main `658f3fac0`, with camdl-book, camdl-scope and ebola-bdbv-camdl
(the one downstream modeling project on this machine) as the places a user would
show up. Nothing was edited, built or run beyond `rg`, `wc`, `sed`, `git log`
and one `gh issue
view`. Each claim is marked **verified** (with how) or
**inferred**. Line counts are from `wc -l` and are approximate where a file is
only partly deleted.

Terms used below. A **run id** is the hash that names a stored result; it holds
the result's own inputs and the **build**, a fingerprint of the camdl binary. An
**invocation record** is the JSON file one command writes under `.invocations/`,
listing the results it asked for and its `--label`. A **handle** is what names a
result on the command line: `@label`, a run id prefix, or a folder. The **model
presentation** is the stored copy of a model file's text, documentation and
description, kept apart from the compiled model so that a comment edit does not
recompute a fit.

## Summary

All three agreed removals are worth making, and each has a condition the plan as
stated does not meet.

1. The stated replacement for `--quantities FILE` is "edit the model, run
   `fit run` again (reused instantly), then `predict @label`". The reuse holds
   only on the same build. The build fingerprint includes the commit
   (**verified**, `rust/crates/cli/src/engine_fingerprint.rs:10-15`), and every
   run id holds it (`store/src/artifact.rs:105`). After any camdl upgrade,
   `fit run` on the edited file computes the fit again. A researcher who
   corrects a reporting formula after upgrading would have to refit. The removal
   needs one more route: `predict @run --model FILE`, which compiles the file
   the user names and checks it against the stored compile, as
   `profile MODEL --fit @run` does today.
2. `rsync` without `.invocations/` leaves the copied results out of `camdl list`
   entirely, because all three listings start from invocation records
   (**verified**, `fit/fit_tree.rs:80-98`, `browse.rs:478`,
   `profile_store.rs:734`). They remain reachable by run id only, with no label,
   no model documentation and, once quantities come from the recorded model
   file, no quantities.
3. Quantities should stay a stored kind of their own. A prediction's identity
   holds a dependency on the quantities content (**verified**,
   `fit/predict.rs:2284`, `id.quantities`). If quantities become a file inside
   the model source, that dependency names the model source's run id, which
   moves on every comment edit, and every prediction is computed again after
   one.

## Ranked table

Ranked by complexity removed per unit of user cost. "Lines" include tests.

| #  | Item                                                           | User cost                                                                         | What it deletes                                                                                                                           | Recommendation                                       | Confidence                                  |
| -- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| 1  | `--quantities FILE` (agreed 2)                                 | Sharing one reporting block across a family of model files; no working user found | about 1,400 lines of Rust and 300 of OCaml: 2 CLI flags, 2 `camdlc` flags, 2 error codes, the second compilation unit, 3 test files       | Remove, with `predict @run --model FILE` added first | Solid on the removal; need-you on `--model` |
| 2  | Recompiles from the store (agreed 1)                           | None: `profile` already takes `MODEL`                                             | Little code today (about 150 lines); stops `compile_stored`, the read closure and one proposal commit from being written                  | Remove                                               | Solid                                       |
| 3  | `camdl receive` (agreed 3)                                     | Copied results are not listed and carry no label, documentation or quantities     | about 1,900 lines: 1 command, 27 functions, 6 types, 2 error variants                                                                     | Remove, and state the listing gap                    | Leaning                                     |
| 4  | `fit.meta.json`'s `docs` field                                 | None: no reader found                                                             | 1 field, 1 extra read of the IR per fit                                                                                                   | Remove                                               | Solid                                       |
| 5  | Index invocation lines; record `counts` and `progress`         | None: no reader                                                                   | 1 enum variant, 2 writers, 2 fields                                                                                                       | Remove                                               | Solid                                       |
| 6  | Quantities lineage and the per-run `quantities` label          | None once item 1 lands                                                            | 2 lineage writers, `problem_quantities`, `run_quantities`, the silent fallback in `open_fit` (about 110 lines)                            | Remove with item 1                                   | Solid                                       |
| 7  | The problem's copy of the model file                           | camdl-scope reads it in two places                                                | `ModelSourceFile`, `ProblemFile::ModelSource`, `OpenedFit::model_source` (about 90 lines); one of the six open questions gets easier      | Remove after items 1 and 2                           | Leaning                                     |
| 8  | The scratch copy of a stored fit, and its path table           | None (internal)                                                                   | `RunFiles`, the `SHOWN` table, `FitView::model_path`, `load_fit_envelope`, `load_calendar_context`; a full copy of each run on every read | Simplify                                             | Leaning                                     |
| 9  | `--link-root`                                                  | None found; one day old                                                           | 790 lines, 1 flag, 1 environment variable                                                                                                 | Remove                                               | Need-you                                    |
| 10 | `camdl label`                                                  | None: it refuses every handle                                                     | about 70 lines, 1 command                                                                                                                 | Remove, or make it record a label                    | Need-you                                    |
| 11 | Removed-flag traps                                             | An old script gets clap's message, not a hint                                     | about 125 references to `_removed_*` across 10 files                                                                                      | Remove the ones older than one release               | Leaning                                     |
| 12 | The `camdl batch` removal test                                 | None                                                                              | 1 test file                                                                                                                               | Remove                                               | Solid                                       |
| 13 | The state-Jacobian compile as a second stored compile          | Blocks `nl-lbfgs` on the ODE backend, which has a user                            | n/a                                                                                                                                       | Keep                                                 | Solid                                       |
| 14 | `repose`: posing a stored prediction again for `score`         | Blocks `camdl score`                                                              | n/a                                                                                                                                       | Keep; simplify later                                 | Leaning                                     |
| 15 | `@label~n`, the IR cache, `--manifest`, `camdl path` and `cat` | Each has a use the removals above rely on                                         | n/a                                                                                                                                       | Keep                                                 | Solid                                       |

## 1. `--quantities FILE`, the reporting-vocabulary override

**What it does.** `simulate MODEL --quantities FILE --quantities-out DIR` and
`predict @run --quantities FILE` compile FILE's `quantities {}` block in place
of the model's own. `camdlc` parses FILE as a second unit of the same compile
(`splice_quantities`, `ocaml/lib/compiler/compiler.ml:246`).

**Who uses it.**

- _ebola-bdbv-camdl_ asked for it (`agent-channel.md:219`; the proposal
  `docs/dev/proposals/2026-08-19-quantities-as-a-separable-layer.md` names 12
  model files with drifted copies of one 50-line block). Its workflow does not
  call it: `rg -e '--quantities[ =]'` over its `workflow/`, `scripts/` and
  `Makefile` finds nothing (**verified**). Its one recorded attempt is friction
  F38 (`camdl-frictions.md:370`, 2026-08-22): the flag refused a byte-identical
  model on every fit in the project. That report is gh#714, still open
  (**verified**, `gh issue view 714`). I did not check whether the defect
  persists on this build. The project now has 111 model files that each hold
  their own block (**verified**, `rg -l '^quantities \{'`).
- _camdl-book_: no use (**verified**, `rg`, excluding `_site` and `_freeze`).
- _camdl-scope_: no use. `agent-channel.md:161` mentions the idea once.
- _Docs_: `docs/camdl-language-spec.md:5491`, `docs/agents.md:578`,
  `docs/camdl-run-spec.md:3304`, `:6948`, `:8391`, `docs/dsl-cheatsheet.md`,
  `docs/dev/warning-catalog.md`.
- _Tests_: `cli/tests/fit_predict_quantities_vocabulary.rs` (465 lines, 10
  tests), `cli/tests/quantities_vocabulary.rs` (407),
  `ocaml/test/test_quantities.ml:514-662` (149), one case each in
  `test_declared_emit.ml`, `util.rs` and
  `compile_settings_reach_every_compile.rs`.

**What a user loses.** Two uses, with different answers.

- _Reporting an existing fit with corrected quantities._ Replaced, on the same
  build, by editing the model and running `fit run` again. A test pins that
  route: `a_prediction_evaluates_the_quantities_of_the_invocation_it_names`
  (`cli/src/fit_command/tests.rs:1120`, **read, not run**). It fails across
  builds, as the summary says. Whether a build can read a fit another build
  stored is **inferred** from `Store::list`'s doc comment ("perhaps by another
  build", `storage.rs:362`) and `docs/engineering/run-store.md:1579`; if it can,
  then today `predict @old --quantities FILE` works after an upgrade and the
  replacement does not.
- _One block shared by many model files._ Not replaced. camdl has no import, so
  the block is copied into each file again, which is the drift the proposal set
  out to stop. In practice the one project with this need already copies.

**What it deletes** (approximate).

- Rust, about 1,400 lines: most of `cli/src/quantities_file.rs` (182; the two
  directory-name helpers shrink to constants); `quantities_from_vocabulary`
  (`fit/predict.rs:4653`, 110 lines); the `vocabulary` argument of
  `resolve_ir_path` and its component of the IR cache key (`util.rs:437`,
  `:698`, `:1185`); the `vocabulary` fields of `QuantitiesWant` and the sim
  options (`sim_command.rs:62`, `sim_store.rs:821`); the suffix branch in
  `result_files/fit.rs:260-270`; `stamp_provenance`; three fixtures in
  `input_fixtures.rs`; the refusal in `repose` (`predict.rs:2171`); and the two
  test files (872 lines).
- OCaml, about 300 lines: `splice_quantities` (54); the `?quantities` argument
  of seven functions in `compiler.ml` and of `Inspect.run_check`; `--quantities`
  on `camdlc compile` and `camdlc check`; `Source_cache.add_unit` and the
  `extra` field; the `quantities_file` argument of `Declared_emit` (`:804`);
  error codes E339 and E340; 150 lines of tests.
- CLI: `simulate --quantities`, `predict --quantities`, `camdlc --quantities`
  twice.

**How much simpler.**

- A compile has one source file. The diagnostics no longer track which of two
  files a location belongs to.
- A prediction from a stored fit compiles nothing. `--no-licm` and
  `--no-ir-cache` on such a prediction then reach nothing
  (`args/mod.rs:2070-2082` says the vocabulary compile is their only effect
  there) and can be refused.
- One failure mode goes: a prediction made with a vocabulary cannot be scored,
  because its identity keeps the file's digest and not the file
  (`predict.rs:2171`).
- A prediction's quantity tables have one directory name, not `quantities-model`
  or `quantities-<key8>`.
- In the combined proposal: the "vocabulary compile" row of the policy table,
  the `vocabulary` field of `CompileRequest`, the choice "a `--quantities`
  file's own documentation is not shown", and the alternative "no source for the
  vocabulary compile" all go.

**What breaks.**

- Every stored prediction's run id moves, if the `quantities` entry is taken out
  of the prediction's settings (`predict.rs:5258`). **Inferred** from reading;
  predictions are cheap to compute again. Fit run ids do not move.
- Goldens: the declared-model section should be unchanged for a model compiled
  without a vocabulary. **Inferred**; confirm that `make update-golden` shows no
  diff before committing.
- `camdlc` and `camdl` must change in one commit, since `camdl` passes the flag.

**Decision needed: the cross-build route.** The question is how a user applies
corrected quantities to a fit that this build cannot reuse. Option A adds
`predict @run --model FILE`: compile FILE, rewrite it from the problem's data as
`fit run` does, refuse unless its structural hash is the stored compile's, and
take its quantities and documentation. The code exists twice already
(`refuse_another_model`, `profile_store.rs:327`, and the comparison half of
`quantities_from_vocabulary`) and would become one function; it also serves a
copied store that holds no invocation records. Option B keeps the lineage, the
first command's quantities sealed with the fit, which is the first-writer copy
the redesign removes. Option C accepts the refit. I recommend A (leaning): it is
the rule of removal 1 applied to `predict`, it costs about 60 lines net, and
without it removals 2 and 3 together leave a copied fit from another build with
no way to evaluate any quantity.

## 2. No command recompiles a model from the store

**What it does today.** Two readers compile the fit problem's stored copy of the
model file: `quantities_from_vocabulary` and `jacobian_compile`
(`profile_store.rs:283`), which an `nl-lbfgs` profile on the ODE backend calls
(`profile.rs:452`).

**Who uses it.** `nl-lbfgs` has a user: ebola-bdbv-camdl fits with it
(`models/fit_national_v2_mle_lbfgs.toml`, **verified** by `rg`). The method
stays. Only the source of its compile changes.

**User cost: none.** `ProfileArgs::model` is a required positional
(`args/mod.rs:2856`, **verified**), and `pose_problem` already compiles it and
checks it against the stored compile's structural hash (`refuse_another_model`).
`jacobian_compile` can compile that file. The predict half is item 1.

**What it deletes.** Today about 150 lines: the three `opened.model_source()`
call sites in `profile_store.rs` (`:138`, `:293`, `:337`), the byte-equality
shortcut in `refuse_another_model`, and then `OpenedFit::model_source`
(`fit_store.rs:1511`, 20 lines), whose last reader is a refusal's text
(`fit/fitted_model.rs:245`).

**How much simpler.** The value is in what the combined proposal no longer
needs:

- `compile_stored`, and commit 15;
- the read closure: `reads` and `outside` on `ModelSourceIdentity`, the scratch
  folder with the model's relative files, and `--emit-deps` as a store input;
- the rule "a reader that compiles refuses when no source is recorded";
- the "Jacobian compile" row of the policy table, which rests on an inference;
- three choices: "the read closure stays", "reads are stored inside the source",
  and "fits stored before this design have no read closure";
- two defects by construction: E200 for a model that reads a file by a relative
  path, and the compile of the first command's file in place of the labelled
  one.

The stored model source is then display text and nothing computes from it, apart
from the quantities question in the summary.

**What breaks.** Nothing found. `refuse_another_model` always compiles the given
file; `named_presentation` already compiles it on the same path, and the IR
cache serves the second compile.

## 3. `camdl receive`

**What it does.** Copies stored results from another store's folder into this
one. It checks every result folder as `camdl verify` does before placing any,
refuses a result whose bytes differ from one already stored, places each at the
path it had, and writes one invocation record that lists what it placed, with
the labels the source's records gave.

**Who uses it.** Added 2026-11-23 (`daa352fed`). Tests: its own
(`store/src/receive/tests.rs`, 700 lines; `cli/tests/receive.rs`, 342, 7 tests).
Docs: `docs/engineering/run-store.md` (14 mentions), `docs/camdl-run-spec.md`
(17), `docs/commands.md`, `docs/runtimes.md`, and the proposal
`2026-11-23-workflow-bindings.md`. camdl-book: none. camdl-scope: listed in
`HANDOFF.md:70` as a feature "to consider later"; no code calls it.
ebola-bdbv-camdl: none. All **verified** by `rg`.

**What a user loses with `rsync` and `camdl verify`.**

- _The listing._ `camdl list` builds its fit, simulation and profile rows from
  invocation records (**verified**, cited in the summary). A copied result with
  no record is not listed.
- _Labels, documentation, quantities._ All three are per-command facts held in
  invocation records. `fit summary <run id>` works without the `#'` legend.
  `predict <run id>` falls back today to the problem's lineage
  (`fit_store.rs:1700-1705`); once item 6 removes the lineage it has no
  quantities.
- _Recording them again._ Running the same command in the destination reuses the
  copied result and writes a record, but only when the destination's build is
  the source's.
- _The refusal of differing bytes._ `run.json` holds the owner and timestamps,
  so two stores' copies of one result always differ in that file, and a plain
  `rsync` overwrites it in a sealed result (**inferred** from the record's
  fields). The documented command needs `--ignore-existing` and the exclusion of
  `.invocations/`, `.index/` and `.staging/`.

So a user is not blocked from copying, and is blocked from finding the copies by
anything but a run id. With `predict @run --model FILE` (item 1) the copies are
usable.

**What it deletes.** About 1,900 lines: `store/src/receive.rs` (724 lines, 27
functions, the types `Received`, `Found`, `Action`, `Planned`, `Moved`,
`Placing`), its tests (700), `cli/tests/receive.rs` (342), `cmd_receive` and
`ReceiveArgs` (about 100), `Error::ReceivedDiffers` and `Error::ReceiveRefused`,
and the draft-folder names in `fsutil.rs:54-70`.

**How much simpler.**

- No function writes a result folder this process did not compute.
- No invocation record is synthesised from another store's records
  (`copy_labels`, `receive.rs:594`, one of the nine sites that choose an
  invocation).
- In the combined proposal: commit 10 goes, commit 9's red test changes, and two
  of the six open questions go (whether `receive` refuses a copy that lacks a
  referenced result; how a received store answers a run whose quantities differ
  between the copied records). The defect "`receive` changes what a run id
  means" goes with it.

**What breaks.** `docs/camdl-run-spec.md:2222` describes the manifest by way of
a result "placed by `receive`"; the sections in `run-store.md` near lines 2091
and 2137 go.

**Recommendation: remove (leaning).** The command is two days old, has no
caller, and carries two open questions and three review findings. The listing
gap is the reason this is not solid: say in the run spec that a copied result is
named by run id until a command lists it, and see item 10 for a small way to
give one a label.

## 4. `fit.meta.json`'s `docs` field

`FitSidecar::docs` (`run_meta.rs:825`) holds the first command's `#'`
documentation. It is written at `fit/mod.rs:1561`, by loading the IR envelope a
second time. No Rust code reads it: `rg '\.docs\b'` over `cli/src/fit` finds
that writer and one read of an envelope's `docs` (**verified**). camdl-scope's
`ModelDocs.from_meta` has one caller, which passes it `docs.json` from the
model's folder, not the sidecar (`ingest.py:566`, **verified**). Remove (solid).
The sidecar's schema version moves.

## 5. Index invocation lines, and the record's `counts` and `progress`

`IndexLine::Invocation` is written at `store/src/invocation.rs:227` and
`maintenance.rs:71`; `Index::learn` matches `Completed` only (`index.rs:54`,
**verified**). The combined proposal already removes these and the two record
fields. Remove (solid); independent of everything else.

## 6. The quantities lineage and the per-run `quantities` label

`fit run` and `profile` record the first command's quantities as lineage of the
sealed fit (`fit_command.rs:382`, `profile_store.rs:392`), and each run's labels
name the quantities this command compiled. `open_fit` uses the label, and falls
back to the lineage without saying so. With item 1, a prediction's quantities
come from the model file its handle's invocation recorded, or from `--model`.
That removes `problem_quantities`, `run_quantities`, both lineage writers and
the fallback (about 110 lines), and merges two of the proposal's roles
(`RUN_QUANTITIES`, `PROBLEM_QUANTITIES`) into the one reference from a compiled
model to its source. The refusal "invocations compiled this run with different
quantities blocks" stays, as a comparison of the quantities of the recorded
sources. Remove with item 1 (solid). Keep `Quantities` as a stored kind, for the
reason in the summary.

## 7. The problem's copy of the model file

`ProblemFile::ModelSource` stores the first command's model file in the fit
problem. After items 1 and 2 nothing compiles it. camdl-scope shows it as the
fit's source (`api/routes.py:1264`) and parses priors from it (`ingest.py:996`),
so its removal needs the model source reachable through `camdl path` first, as
the combined proposal orders it. The open question "what does `problem.toml`
name as its model" remains, with lower stakes: the field is display text, and
the model's stem is enough. Remove (leaning).

## 8. The scratch copy of a stored fit

**What it does.** `open_fit` and `RunFiles::of` (`fit_store.rs:1431`) copy every
result and provenance file of a stored run into a temporary directory, write a
rebuilt `model.ir.json` beside them, and register the copy in a process-wide
table, `SHOWN`, so that messages print the store's path (`shown`, `show_as`; 23
matching lines across 7 files, **verified** by `rg -c`). The readers
(`fit summary`, `predict`, `compare`, `fit table`) then read a directory by
path. `walk_fits_root` opens every fit this way to list them (`fit_tree.rs:93`).

**Why it costs.** `FitView::model_path`, `load_fit_envelope` and
`load_calendar_context` (`fit_summary.rs:324-345`) exist to find the model in
that copy and parse it again. Each read of a fit copies and hashes its draws. I
did not measure this.

**Recommendation: simplify (leaning).** Readers take the `Stored<FitRun>` and
ask it for a named file; the copy, the table and the three functions go. The
proposal's commit 13 does the model half. The size of the rest is unknown to me:
it depends on how many readers take a directory path, which I did not count.

## 9. `--link-root`

A command with `--link-root DIR` writes symbolic links to its result files under
a browsable directory (`link_tree.rs`, 620 lines, plus 171 of tests; added
2026-11-24). It is the third way to find a result file, after the output
manifest (`--manifest`) and `camdl path`. No user found in the book, the viewer
or the downstream project. It has its own lifecycle rules (claim, record file,
refusal of foreign directories) to keep stale links from looking current.

**Decision.** The workflow-bindings proposal accepted it two days ago for
workflow tools that want a stable path without calling camdl.
`camdl path
@label NAME` and the manifest cover a rule's inputs and outputs. I
would remove it until a workflow asks (need-you: it is your recent call, and I
found no evidence either way beyond the absence of users).

## 10. `camdl label`

The command refuses every handle (`fit/mod.rs:1786-1826`, **verified** by
reading). Its message says a stored result is sealed. A label is held by an
invocation record, not by the result, and `Store::list` (`storage.rs:367`)
already lists a stored result in a new record without storing anything.

**Decision.** Remove the command (70 lines), or make it write an invocation
record that lists the result under the label. The second is about 30 lines and
is the missing step after an `rsync`: it puts a copied result in `camdl list`
and gives it a name. I lean to the second if `receive` goes, and to removal if
`receive` stays (need-you).

## 11. Removed-flag traps

Hidden flags named `_removed_*` (20 on `fit run` and `profile`), the
`FitPredictRemovedArgs` command, and three functions (`refuse_removed_flags`,
`refuse_removed`, `check_removed_flags`) answer a removed flag with the new
spelling: about 125 references in 10 files. `CLAUDE.md` rules out compatibility
shims; these compute nothing, but each is a field every constructor of the
argument struct must fill (`fit_command.rs:202-213`). Remove the traps older
than one release (leaning).

## 12. The `camdl batch` removal test

`cli/tests/camdl_batch_removed.rs` asserts that a removed command is absent. I
found no other `camdl batch` leftover: the `BatchReplicated` mode in
`external-harness` runs a command a case file names. Remove (solid).

## What to keep

- **The state-Jacobian compile as a second stored compile**
  (`ChainIdentity::jacobian_model`, `fit_store.rs:280`). `nl-lbfgs` on the ODE
  backend reads it and has a user. The proposal measured it at 4.1 times the
  ordinary compile, so compiling it for every fit is the wrong trade.
- **`repose`** (`fit/predict.rs:2156`, about 200 lines). `camdl score` and
  `compare` compute a stored prediction again with outcomes bound, and rebuild
  the command's flags from the identity's JSON to do it. That is fragile, and it
  is the only route to a score. Keep; a typed options record would replace the
  JSON walk later.
- **`Quantities` as a stored kind**, for the reason in the summary.
- **`@label~n`**. It is about ten lines in `handle.rs`, and after item 1 it is
  how a user reaches the quantities of an earlier command.
- **The IR cache, `--manifest`, `camdl path` and `camdl cat`.** Item 2 leans on
  the first; items 7 and 9 lean on the others.

## Order of landing

Each line is one commit.

1. Index invocation lines and the record's `counts` and `progress` (item 5).
2. `fit.meta.json`'s `docs` (item 4).
3. `camdl receive` (item 3), with its documentation. It comes before the
   record-format work so that commit 10 of the combined proposal is never
   written.
4. `profile`'s Jacobian compile reads the positional `MODEL` (item 2). Red test:
   a model that reads `mult.tsv` by a relative path, which fails with E200 today
   (`exp_e.sh`).
5. `predict @run --model FILE`, if you take option A of item 1.
6. `--quantities FILE`, in `camdl` and `camdlc` together (item 1).
7. The quantities lineage and the per-run label (item 6).
8. The problem's model copy and `OpenedFit::model_source` (item 7), after the
   note to camdl-scope.

Items 8 to 12 are independent of these.

## What is left of the combined proposal

Of its six open questions, two go with `receive`, and "what a prediction does
when no invocation recorded the fit's quantities" merges into the existing
choice for a result with no recorded source. Three remain: what `problem.toml`
names, whether an unresolved reference changes `verify`'s exit status, and the
word that replaces "labels". Commits 10 and 15 go, commit 14 loses `reads` and
`outside`, and the policy table keeps two rows: display, and quantities for a
prediction.
