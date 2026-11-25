# Stored models and invocation records

Status: accepted. It supersedes
`docs/dev/proposals/2026-11-24-model-presentation.md`. Read against main
`658f3fac0`. Claims are marked **verified** (run, or read in the code, as
stated) or **inferred**. The removal assessment is
`docs/dev/reviews/2026-11-25-run-store-removals.md`. Reproductions named
`exp_*.sh` were run during review and are not in the repository; each becomes a
test in the commit that fixes its defect.

## Summary

The run store shares one sealed result among every command that asks for the
same computation. Several features want facts that are not part of the
computation (which model text, which quantities, a source to compile again) from
that shared result, per command. They are the source of most of the complexity
in this part of the code: first-writer copies, recompiles of a stored copy,
references hidden in labels, and readers that each choose a command by their own
rule.

This proposal does two things, in order:

1. **It removes features that cost more than they give**: the
   `--quantities FILE` override, recompiling a model from the store, the
   quantities bookkeeping those two need, `camdl label`, hints for removed
   flags, the `camdl batch` leftovers, and fields nothing reads. About 2,200
   lines go. It adds one small route, `predict @run --model FILE`, and renames
   `camdl receive` to `camdl import`.
2. **It then reshapes the types of what is left**: the invocation record's
   states as types, one typed reference from a compiled model to the model file
   a command used, one function that decides which command answers for a result,
   and one value every reader gets a stored model from.

## The problem

### Files that differ only in text share one computation

A `.camdl` file holds what the engine runs (compartments, transitions,
parameters, observation models, the `simulate` window, the output schedule,
interventions, initial conditions) and text (comments, whitespace, the `#'`
documentation, the `description`, and `let` bindings the dynamics never use). A
compiled model's identity is a hash of what the engine runs, never of the file's
text, name or location. Every result that runs the model depends on that
compiled model (a fit's identity also holds its data and settings, and every run
id holds the build).

- A change to anything the engine runs is a new compiled model: every
  simulation, fit and prediction that depends on it is computed again.
- A change to the text only keeps the compiled model, so nothing is recomputed.
  The test `only_what_the_engine_runs_keys_the_model_part`
  (`camdl_store/tests.rs`) pins this.

So several files can map to one compiled model and one stored fit when they
differ only in text. Two files that define `let R0 = beta / gamma` and
`let R0 = 2 * beta / gamma`, and use `R0` nowhere in the dynamics, share one
compiled model (verified: both store as `compiled_model d6ac22d65438`).

A `quantities {}` block is in the same position: it changes no dynamics, so it
is stored apart from the compiled model, as its own content keyed by its bytes.

### Per-command facts, and where they are kept

Results are sealed: once finished, a result's files and record are never written
again, and every command that asks for the same thing shares it. In the current
design, what differs between those commands is written in an **invocation
record**, one JSON file per command run, `<store>/.invocations/<seq>.json`. Each
entry in it, one result the command asked for, is a **listing**, with
**labels**: what a person recognises that result by, held by the store as JSON
it never reads. A **handle** names a result on the command line: `@label`, a run
id or a prefix of one, or a result's folder.

Three per-command facts end up somewhere only the first command can write:

- **Which quantities a fit used** is recorded twice: as a field in the fit run's
  labels, and as **lineage** (a provenance field of a sealed record, naming what
  a result was made from, not hashed) of the compiled model, the fit problem and
  the fit run (`fit_command.rs:382`, `profile_store.rs:392`; verified by
  reading). When no record names quantities, `open_fit` silently uses the
  lineage, the first command's (`fit_store.rs:1700-1705`).
- **Which model text a command compiled** is a field, `presentation`, in the
  compiled model's labels.
- **`fit.meta.json`'s `docs`**, the first command's `#'` documentation. Nothing
  reads it.

### Two commands compile a stored copy of the model again

`predict @run --quantities FILE` compiles a second file's quantities into the
model, and an `nl-lbfgs` profile (the gradient-based optimizer) on the ODE
backend needs the compile with the state Jacobian. Both compile the fit
problem's copy of the model file, written to a scratch folder. Both go wrong:

- the copy has none of the files it reads by a relative path beside it, so a
  model with `read("mult.tsv")` fails with E200 (verified, `exp_e.sh`,
  `exp_f.sh`);
- the copy is the first command's file: fit `plain.camdl`, then `with_r0.camdl`,
  and `predict @r0 --quantities q.camdl` with `q.camdl` naming `R0` fails with
  "undeclared name" (verified, `exp_j.sh`).

Making these recompiles work would mean storing every file a source reads and a
compile-from-the-store function. Removing them is smaller.

### References are hidden in labels, and each reader picks a command by its own rule

The store cannot follow a run id kept inside opaque labels. `camdl receive`,
which copies stored results from another store into this one, accepts a copy
that lacks the referenced model text or quantities, and `camdl verify`, which
checks every result's files against their digests, passes it (verified,
`exp_receive.sh`, `exp_quantities.sh`).

Nine sites decide "which invocation answers for this result" by about six rules
(`Store::invocation`, `run_quantities`, `shown_for`, `one_with_by_products`,
`OutputManifest::of_results`, `copy_labels`, `latest_label`, `run_label`,
`note_model_stored_under_another_name`; verified by reading). Each reads every
record, often several times: `fit summary @f` takes 54 ms with 2 records, 121 ms
with 300 and 664 ms with 3000 (measured). Resolving a run id walks every result
folder and rewrites the index, so a read-only command writes to the store
(verified). Only `fit run` checks a label, so `simulate --label 'f~1'` records a
label no handle can reach (verified).

## Part 1: removals

Each removal is one commit, with its gate. The user cost and the size are from
the removal assessment (`docs/dev/reviews/2026-11-25-run-store-removals.md`).

### 1.1 `predict @run --model FILE` (added first)

`predict @run --model FILE` compiles the file named and checks it with
`check_stored_fit` (`fit/fitted_model.rs:210`), the check
`simulate MODEL --fit
@run` already uses: it resolves anchors on both sides,
refuses a file whose model differs, and says so when only the quantities differ
(`ModelAgreement::ReportingDiffers`, `:83-93`). The command then stores `FILE`'s
source and quantities like any compile, records that source on the compiled
model's listing, and evaluates `FILE`'s quantities. It is the route for a
researcher who edits a quantity after a camdl upgrade: the build is part of
every run id (`engine_fingerprint.rs:10-15`, verified), so running `fit run`
again after an upgrade would compute the fit again. On the same build, running
`fit run` again (reused at once) and then `predict @label` also works. That a
fit stored by an earlier build can be opened by a later one is inferred, not
verified; commit 1 checks it once by hand.

### 1.2 Remove `--quantities FILE`

**What it is.** `simulate MODEL --quantities FILE` and
`predict --quantities
FILE` compile FILE's `quantities {}` block in place of the
model's own; `camdlc --quantities` parses FILE as a second unit of the compile.

**Why remove it.** Quantities belong in the model file. Adding or editing one
there changes no compiled model, so no fit is invalidated; only predictions that
evaluate quantities are computed again. No working use was found: the one
downstream project that asked for it never calls it, and its report that the
feature was unusable (gh#714) is open. What is lost is one quantities block
shared by a family of model files.

**What it deletes.** About 1,400 lines of Rust and 300 of OCaml: two CLI flags,
two `camdlc` flags, two error codes (E339, E340), the second compilation unit
(`splice_quantities`), `quantities_file.rs`, three test files. The same commit
corrects `docs/agents.md` (shipped as `camdl docs agents`), the language spec
(edited by hand, never formatted), the run spec, the cheatsheet and the error
catalogue, and closes gh#714.

### 1.3 Remove recompiling from the store

After 1.2, `predict @run --quantities` is gone. The Jacobian compile of an
`nl-lbfgs` profile on ODE compiles the `MODEL` file the user passes `profile`,
which it already takes and checks, in place of the problem's copy. The store
then never compiles a model. About 150 lines go, and the read closure and
compile-from-the-store function the earlier design needed are never written. The
problem keeps its copy of the model file, for reproducibility and for viewers;
it is a model file that compiles to the problem's model.

### 1.4 Remove the quantities lineage and the per-run quantities label

With quantities only in the model file, which quantities a command used follows
from which model file it used. The model source (Part 2) records the quantities
its compile produced. The two lineage writers, `problem_quantities`,
`run_quantities` and the silent fallback in `open_fit` go (about 110 lines).
Quantities stay their own stored kind, keyed by their bytes, because a
prediction's identity depends on them (`fit/predict.rs:2284`, verified): kept
inside the model source, a comment edit would recompute every prediction.

### 1.5 Remove `camdl label`

It refuses every handle it is given (verified by the assessment), so it labels
nothing. `--label` at run time names a command. About 70 lines.

### 1.6 Remove the hints for removed flags

About 125 references to `_removed_*` arguments across 10 files keep a hint for
flags removed in earlier releases. A removed flag then gets clap's own message.
The tests that assert a hint (`gh789_tempering_flag_removed.rs` and others) go
with them, and `PredictArgs::source` loses its
`required_unless_present =
"_removed_fit"` (`args/mod.rs:1645`).

### 1.7 Remove the `camdl batch` leftovers

The test that `camdl batch` is gone, and any remaining references.

### 1.8 Remove fields nothing reads

`fit.meta.json`'s `docs`, with its second read of the IR; and the index's
invocation lines and their two writers. The invocation record's `counts` and
`progress` fields go too, but with the format change (commit 12), because both
are required fields of today's record type and a build from before their removal
would drop a record that lacks them without a message. Live progress is
unaffected: it is the separate file `.invocations/<seq>.progress.json`
(`store/src/invocation.rs:198`), which camdl-scope reads
(`camdl_scope/store.py:461`) and which stays.

### 1.9 Rename `camdl receive` to `camdl import`

`import` does what `rsync` cannot: it writes one local record that lists what it
placed, which is what makes imported results appear in `camdl list` (records are
numbered per store, so copying another store's records would overwrite this
one's); and it refuses a copy that lacks a result another depends on. The rename
is one commit, with `docs/engineering/run-store.md`, the run spec,
`docs/commands.md`, `docs/runtimes.md`, `cli/tests/receive.rs` and the error
variants named for it; its behaviour changes in Part 2.

## Part 2: the types

### Concepts

- **Result.** Sealed, keyed by its identity, shared. Unchanged.
- **Dependency.** A `Dep<A>`: a field of a result's identity naming another
  stored result of kind `A` it was computed from, with the digest of what it
  read. Hashed into the run id and sealed. A fit problem's `Dep<CompiledModel>`
  is why a change to the dynamics re-keys it. Unchanged.
- **Invocation record.** One per command, as today, with its states as types:
  running, and closed (ended, or interrupted and closed by `camdl clean`, which
  closes the records of commands that died; it deletes no result).
- **Reference.** Unlike a dependency, in no identity: a typed edge in an
  invocation record from a result the command listed to another stored result.
  There is one: a compiled model's **source**, the model file the command
  compiled it from. The store reads it, so `import` and `verify` can follow it.
- **Model source.** One stored result per distinct model file: the file, its
  name, its views, and the quantities its compile produced. It is the model
  presentation of today, renamed, with the quantities added.
- **Views.** What the compiler shows of a source: `docs.json` (the `#'`
  documentation and `description`), `declared.json` (the model as declared: each
  parameter's kind, bounds and prior, each `let`'s definition), the rendered
  mathematics and the compartment flow graph. Each is a function of the source
  and the build, and part of the model source's identity.

The code below uses five types from the store library: `Id<A>`, a typed run id;
`Stored<A>`, a completed result read from the store; `Located`, a result found
by run id, of any kind; `Hashed<T>`, a value with the SHA-256 digest of its
bytes; and `RunInfo`, the current command's record while it is written.

### The invocation record

The file, its path, the moments it is written, the lock and the replacement by
rename are as today, and so are its field names and encodings. camdl-scope's
record reader needs one change: a format constant of its own for invocation
records, since its one `RECORD_FORMAT` (`camdl_scope/store.py:46`) covers
`run.json` too.

```rust
// store
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Record {
    Running(Running),
    Ended(Closed),
    Interrupted(Closed),                 // written by `clean` for a dead owner
}
pub(crate) struct Head {                 // fixed when the command starts
    pub format: u16,                     // 2; its own constant, not RECORD_FORMAT
    pub record: RecordKind,              // "invocation", as today
    pub seq: u64, pub owner: Owner, pub started: Timestamp,
    pub label: Option<Label>,
    pub argv: Vec<String>, pub given: serde_json::Value, pub config_files: Vec<ConfigFile>,
}
pub(crate) struct Running { #[serde(flatten)] pub head: Head, pub results: Vec<Listing<Pending>> }
pub(crate) struct Closed  { #[serde(flatten)] pub head: Head, pub ended: Timestamp, pub results: Vec<Listing<Outcome>> }

pub(crate) struct Listing<O> {
    pub kind: String, pub run_id: Key,
    pub attributes: serde_json::Value,                   // names and values only (today `labels`)
    /// Set on a compiled model's listing and absent on every other kind's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Reference>,
    #[serde(flatten)] pub outcome: O,
}
/// `"outcome": "computed" | "reused" | "busy" | "failed"`, with `"failure"`
/// beside it when failed, as today. A running record may also hold `"planned"`.
pub enum Outcome { Computed, Reused, Busy, Failed(Failure) }
pub enum Pending { Planned, Done(Outcome) }
pub struct Reference { pub kind: String, pub run_id: Key }

/// `fit run`'s rule today (letters, digits, space, comma, dot, underscore and
/// hyphen; at most 64 characters; trimmed), plus not `.` or `..`. Parsed by
/// `StoreRequest::new`, so every command applies it.
pub struct Label(String);

impl RunInfo {
    /// In this invocation, the listed result `of` was made from `source`. Refused
    /// unless this invocation listed `of`. The store is generic over kinds; camdl
    /// calls it with a compiled model and its model source.
    pub fn record_source<A: Artifact, B: Artifact>(&self, of: &Id<A>, source: &Id<B>) -> Result<()>;
}
impl Store {
    /// Lists a stored result in this invocation as reused, storing nothing.
    pub fn list<A>(&self, id: &Id<A>, labels: &A::Labels, info: &RunInfo) -> Result<()>;   // exists
    /// A snapshot of the closed records, read once per command.
    pub fn invocations(&self) -> Arc<[Invocation]>;
    /// The running records, read from the directory under the lock each time.
    pub(crate) fn running(&self) -> Vec<Running>;
    pub fn listings(&self, handle: &Handle) -> Result<Listings>;
}
```

One closed listing, before and after:

```json
{"kind": "compiled_model", "run_id": "631b…", "labels": {"name": "model", "presentation": "5e87…"},
 "outcome": "computed", "failure": null}
{"kind": "compiled_model", "run_id": "631b…", "labels": {"name": "model"},
 "source": {"kind": "model_source", "run_id": "5e87…"}, "outcome": "computed"}
```

With one reference kind, the record holds it as one typed field. A general map
of named references can come when a second kind appears.

### How a lookup finds an invocation

`Store::listings(handle)` returns every listing of what a handle names, one
`Listed` per invocation that listed it. It is the only function that chooses.

It works on a snapshot of the records, read once per command, to which the store
adds its own process's record when it writes it. Two reads stay on the
directory, under the lock, because they must see what another process wrote a
moment ago: the label check when a command starts, and the lookup of a label's
previous holder when it ends.

- **A label** (`@f`, `@f~1`) names one invocation, which answers alone.
- **A run id or a folder** names every closed listing of the result whose
  outcome is computed or reused, newest first; the newest answers. When none
  counts, the newest listing in a running record answers, marked unfinished.
- **A full run id** is found through the index and confirmed against its folder;
  **a prefix** walks the result folders. Neither writes the index.

```rust
pub struct Listings { by: By, listed: Vec<Listed> /* newest first */ }
pub enum By { Label { seq: u64 }, Result { run_id: Key } }
/// One invocation's listing of the result.
pub struct Listed { pub seq: u64, pub label: Option<Label>, pub unfinished: bool, labels: serde_json::Value }
impl Listed { pub fn labels<A: Artifact>(&self) -> Option<A::Labels>; }

impl Listings {
    /// The source the answering invocation recorded for `of`. `None` when it
    /// recorded none, even if an earlier one did: no fallback. `Several` when
    /// earlier listings recorded other sources.
    pub fn source<A: Artifact, B: Artifact>(&self, of: &Id<A>) -> Recorded<B>;
    pub fn answering(&self) -> Option<&Listed>;
    pub fn labelled(&self) -> Option<&Listed>;   // the newest listing whose invocation has a --label
    pub fn first(&self) -> Option<&Listed>;      // the oldest, for "first stored under another name"
}
pub enum Recorded<B: Artifact> {
    None,
    One { source: Id<B>, unfinished: bool },
    Several { latest: Id<B>, others: Vec<(u64, Id<B>)> },
}
```

What each reader does with it:

| Reader                                                  | `None`                                                                                        | `Several`                                                                                                                                     |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| display: `fit summary`, `camdl path`, `camdl cat`       | no documentation or views, one notice                                                         | the latest, one notice naming the others and their labels                                                                                     |
| `predict @run` quantities                               | refused: "no invocation recorded the model file; run `fit run` again, or pass `--model FILE`" | refused when their quantities differ, asking for an `@label`; the latest when they agree; refused, naming it, when one of them cannot be read |
| `predict @run --model FILE`, `profile MODEL --fit @run` | the file given                                                                                | the file given                                                                                                                                |

A label always names one invocation, so `predict @label` is never ambiguous.

### The model

```rust
// cli
pub(crate) struct ModelLabels { pub name: String }          // `presentation` moves to `Listing::source`
pub(crate) struct RunLabels { pub method: String, pub seed: u64 }   // `quantities` goes

#[identity(tag = "model_source", version = 1)]
pub(crate) struct ModelSourceIdentity {
    pub file_name: String,                       // `camdlc render` writes its stem
    pub file: Hashed<SourceFile>,
    pub camdl: bool,
    pub quantities: Dep<Quantities>,             // what its compile produced; `import` and `verify` follow it
    pub docs: Hashed<SourceFile>,                // the views
    pub declared: Hashed<SourceFile>,
    pub render: Option<Hashed<SourceFile>>,
    pub graph: Option<Hashed<SourceFile>>,
}

/// The model a stored result was computed from, read once.
pub(crate) struct StoredModel {
    pub dep: Dep<CompiledModel>,
    pub model: ir::Model,                        // what the engine runs
    pub source: Shown,
}
/// The recorded source, read: its views held in memory.
pub(crate) enum Shown {
    Source { source: Stored<ModelSource>, views: Views, others: Vec<(u64, Id<ModelSource>)>, unfinished: bool },
    NotRecorded,
    /// A record names a source whose folder cannot be read.
    Unreadable { run_id: Key, why: String },
}
impl StoredModel {
    pub fn of(store: &Store, handle: &Handle, result: &Located) -> Result<StoredModel, String>;
    pub fn quantities(&self) -> Result<Dep<Quantities>, String>;   // the reader policy above
    pub fn docs(&self) -> &ModelDocs;                              // empty unless `Shown::Source`
    pub fn declared(&self) -> Option<&DeclaredSection>;
    pub fn render(&self) -> Option<&[u8]>;
    pub fn graph(&self) -> Option<&[u8]>;
    pub fn notice(&self) -> Option<Notice>;
}
```

The flow:

```
command given sir.camdl
  compile ─▶ CompiledModel (what the engine runs) ─put─▶ compiled_model/sir-<id>/
             Quantities (the block, by its bytes)  ─put─▶ quantities/sir-<id>/
             ModelSource (file + views + Dep<Quantities>) ─put─▶ model_source/sir-<id>/
  record:    listing compiled_model sir-<id>, source = model_source sir-<id>

reader given @label, a run id or a folder
  StoredModel::of ─▶ the result's Dep<CompiledModel> ─▶ what the engine runs
                  └▶ store.listings(handle).source(compiled model) ─▶ Recorded
  display: docs(), declared(), render(), graph()     predict: quantities()
```

### How each consumer gets what it needs

| Consumer                                              | Needs                                        | After                                                                                                                                                                      |
| ----------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit summary --parameters`                            | the `#'` legend                              | `opened.model.docs()`                                                                                                                                                      |
| `fit summary` dates                                   | `origin`, `time_unit`                        | `opened.model.model`                                                                                                                                                       |
| `predict @run`                                        | the quantities                               | `opened.model.quantities()`                                                                                                                                                |
| `predict @run --model FILE`                           | another file's quantities                    | compile `FILE`, check it with `check_stored_fit`, store its source and quantities, record that source, evaluate its quantities                                             |
| `score <prediction>`                                  | the prediction's quantities                  | `repose` takes them from the prediction's own identity, for a prediction from a fit as for a fixed point (today only the fixed-point arm does, `fit/predict.rs:2251-2291`) |
| `profile --fit @run`, `nl-lbfgs` on ODE               | the Jacobian compile                         | compile the `MODEL` file `profile` is given                                                                                                                                |
| `predict @run`, `score`                               | to record which source they used             | `list` the compiled model, then `record_source` with the source `listings` gave                                                                                            |
| `camdl path`, `camdl cat`                             | a view as a file                             | `model_source:docs`, `:declared`, `:source`, `:render`, `:graph`                                                                                                           |
| `import`                                              | the referenced sources                       | copied with the listings; refused when missing                                                                                                                             |
| `verify`                                              | that each source reference resolves          | an unresolved one is listed apart; the exit status still means "a result is damaged"                                                                                       |
| the output manifest                                   | the link from a compiled model to its source | a `source` field naming another entry                                                                                                                                      |
| camdl-scope: the record                               | status, results, labels, live progress       | a format constant for invocation records; `<seq>.progress.json` unchanged                                                                                                  |
| camdl-scope: parameters, documentation, render, graph | the views                                    | `camdl cat <run> model_source:declared`, `camdl path <run> model_source:docs model_source:render model_source:graph`; the problem's `model.camdl` stays for its Source tab |

### `import`

`import` copies results and the model sources their listings reference, and
refuses a copy that lacks one, as it already refuses a copy that lacks a
dependency. It writes one local record. That record keeps a listing for each
distinct combination of labels and source the copied records gave a result, so
`predict <run id>`, refused as ambiguous in the source store, is refused in this
one too; the refusal says the source store's labels did not travel. The source
store's labels do not become labels here. A copied folder of a kind this build
does not register, such as an old `model_presentation`, is not placed and is
reported.

### Choices this proposal makes

- **Old records are not read, and the store says so once per command.** A
  format-1 record is reported, not silently dropped; results stay reusable,
  because reuse goes by identity. `predict @run` on a fit with no readable
  record refuses, as the table says, so no reader falls back to the first
  command's quantities.
- **Old model presentations.** The `model_presentation` kind is deleted.
  `verify` reports a folder of a kind this build does not register as not
  checked, in one line, without changing its exit status. No command deletes
  those folders.
- **Per-result labels are renamed attributes.** A listing's `labels` field
  becomes `attributes`, and each kind's `…Labels` type becomes `…Attributes`
  (`ModelAttributes { name }`, `RunAttributes { method, seed }`). They are what
  camdl fills in to tell results apart: folder names, `camdl list` columns,
  manifest fields. A `--label` keeps its name and meaning: the name of one
  command's invocation, reached as `@label`. The rename goes in commit 12, with
  the format change.
- **Record states are types; Git's split of refs and reflog into files of their
  own is not taken.** A command claims its `--label` when it starts
  (`LabelInUse`, `invocation.rs:319-332`) and holds it only once it ends with
  every result complete (`:293-296`). One atomic replacement of the record makes
  "ended" and "holds its label" true together; a separate label file would make
  that three steps.

## What changes outside the code

- **Run ids of simulations, filters, fit problems, fit runs and profiles:** none
  move. Labels, references and provenance are not hashed (verified by reading
  `maintenance.rs:174`, `artifact.rs:105`).
- **Run ids of predictions and scores move.** A prediction's identity holds the
  `--quantities` file's digest (`fit/predict.rs:5258`), so removing the option
  re-keys every prediction, and every score that depends on one. They are
  computed again when next asked for.
- **Run ids of model sources:** every one moves (the kind is renamed and its
  identity gains the name and the quantities key).
- **The invocation record:** format 2, with its own version constant;
  `store.json` and every result's record keep format 1.
- **File contracts** (`result_files`, run spec §2.9): `model_presentation`
  becomes `model_source`; `fit.meta.json`'s schema version moves; each content
  kind gets its own contract version.
- **The output manifest** gains `source`.

## Alternatives

- **Make the recompiles work**, by storing every file a source reads and
  compiling from the store. It fixes E200 and the wrong-file case at the cost of
  a read closure, a compile request type and a refusal policy for sources that
  disagree. Rejected for the removal in 1.3.
- **Git's split** of an immutable command record, labels as refs in their own
  files, and a reflog. Rejected for the three-step end above.
- **The documentation in the compiled model's identity.** Every comment edit
  would recompute every fit.

## Order of commits

Each passes the full gate and starts from a test that is red on main, except a
pure deletion, whose gate is the existing tests.

**Removals:**

1. `predict @run --model FILE`. Red (the flag does not exist on main): on one
   build, a model file whose quantities were edited is evaluated by
   `predict
   @run --model FILE`, and no new fit listing is written. Checked
   once by hand: a fit stored by an earlier build opens.
2. `--quantities FILE` removed.
3. The profile's Jacobian compile uses `MODEL`; recompiling from the store
   removed. Red: `exp_f.sh` (E200) passes.
4. `camdl label` removed.
5. Hints for removed flags removed.
6. `camdl batch` leftovers removed.
7. `fit.meta.json`'s `docs` and the index's invocation lines removed.
8. `receive` renamed `import`.

**Store fixes:**

9. The store parses a label. Red: `simulate --label 'f~1'` is refused.
10. One read of the records per command. Red: a counted-read store test.
11. A full run id resolves through the index, a prefix walks, and a read never
    writes the index. Red: `.index/` unchanged after `camdl path <run id>`.

**The record and the model, pushed together so format 2 is written by one
build:**

12. Record states as types, format 2, old records reported, and the record's
    `counts` and `progress` removed. Red: a format-1 record is reported; a
    closed record with a `planned` listing is refused. From here until commit
    15, `open_fit`'s silent fallback to the quantities lineage is live for an
    old record; that is safe only because 12 to 16 are pushed together.
13. `Store::listings`, moving the readers of names: `problem_labels`,
    `problem_model_name`, `predict.rs:2274`, `one_with_by_products`,
    `of_results`, `latest_label`, `run_label` and the first-listing note. Red:
    `None`, `One`, `Several`, a failed listing passed over, a running one marked
    unfinished.
14. The model source and its reference: the rename, its name and
    `Dep<Quantities>`; `Listing::source` replaces the `presentation` label, and
    `shown_for` moves to `Listings::source`. Red: records expect `source`.
15. The quantities lineage, the per-run label, `run_quantities` and the fallback
    go; `predict @run` takes its quantities from the source; `repose` takes the
    prediction's own. Red: `predict <run id>` with differing quantities refuses;
    a prediction made with `--model FILE` scores.
16. `StoredModel` for readers; the scratch-file routes in `fit summary` go. No
    behaviour change; the date and legend tests guard it.

**Following the reference:**

17. `import` copies and checks sources, keeps distinct listings, and reports
    unregistered kinds. Red: `exp_receive.sh` and `exp_quantities.sh` as tests.
18. `verify` reports unresolved sources and unregistered kinds. Red: a removed
    model source folder is named; an old `model_presentation` folder is listed
    as not checked.
19. The manifest carries `source`.
20. The store's engineering document, `docs/engineering/run-store.md`, and run
    spec §2.9 and §4.5. Each removal commit above corrects the documentation and
    tests it changes; the notes for camdl-scope go with commit 12 (the format
    constant) and commit 14 (the views).
