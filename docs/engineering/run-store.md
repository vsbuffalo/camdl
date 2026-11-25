# The run store

Most camdl computations are expensive and are asked for more than once. A
modeler filters the same model at the same parameters again to export a
diagnostic they forgot, reruns a fit script after editing an unrelated line, or
extends a profile grid by two points. Each time, the work already done should be
found and used, and each time, the answer must be the one the new request would
have computed. A stored result that is reused for a request that would have
produced something else is a plausible-looking wrong answer, the worst outcome
camdl can give.

The run store is the part of camdl that keeps results on disk and decides when
one can be reused. It stores each result under a hash of everything that
determines it, finds the result again when the same computation is asked for,
and records what produced it. This page is for a developer who is about to add a
command to the store, add a new artifact kind (a kind of stored result), or work
out why a stored result is what it is. What enters each command's identity, flag
by flag, is [Run identity](run-identity.md).

The store is two crates. `store` (`rust/crates/store`) is a library that holds
no camdl type, and `store-derive` holds its derive macros. camdl's side is in
the `cli` crate: `camdl_store.rs` holds what every command on the store shares,
and `sim_store.rs`, `pfilter_store.rs`, `fit_store.rs` and `profile_store.rs`
define the artifact kinds each command stores.

## Guarantees

The store makes a small number of promises. Each is listed below in plain words
and links to its row in
[What is enforced, and by what](#what-is-enforced-and-by-what), which says what
carries the promise (the compiler, a type, the library or a test) and what it
leaves open.

- **A safe cache.** A fit or a filter run can take hours, and asking for it
  again should not recompute it or overwrite what was stored. The store reuses a
  stored result only when the request would have produced exactly it: the
  result's identity holds [everything that can change it](#enf-hash),
  [each input that can change it does](#enf-identity), and
  [inputs that only affect how it runs do not](#enf-execution).
- **Every input is classified, or the code does not compile.** Every
  command-line argument, settings-file key and environment variable a command
  reads must state what it does to stored results ([its class](#enf-class)), and
  [every declared input reaches the code that resolves it](#enf-resolved).
- **What ran is what was hashed.** [The engine](#enf-ran) takes its model and
  settings from the stored inputs the identity names, and every command checks
  the model it runs against the stored compile.
- **The same identity gives the same files.**
  [A result's output files are a function of its identity](#enf-function), and
  the store compares them whenever it computes a result again.
- **No hidden reads or writes.** [Every upstream read is declared](#enf-reads):
  a computation reads another stored result only through a dependency its
  identity names, and [writes only through the store](#enf-writes).
- **Only whole results are read.**
  [Only a completed, undamaged result is reused or read](#enf-completed), and
  [a record of another format is never misread](#enf-records).
- **Finished results are sealed.** [Nothing writes into a result](#enf-sealed)
  after it is finished.
- **One computation at a time.**
  [One run id is computed by one process at a time](#enf-claim), and
  [a run whose process died is noticed](#enf-dead).

Most of these are carried by the compiler and the types. Two rest on tests
instead, and are only as strong as the tests' reach: that an execution input
leaves results unchanged, and that engine code reads nothing behind the store's
back.

## How camdl reads its command line

Everything a computation reads enters camdl at one boundary: the command-line
arguments, the settings files such as `fit.toml`, the environment variables and
the data and model files they name. This page calls it the input boundary. The
store can promise a safe cache only if every input is classified there, as it
enters: either it can change a result, and so belongs in the result's identity,
or it cannot, and the class says what it does instead. Only camdl can make that
call, because only camdl knows what each input means to the computation; a
workflow tool or a user script sees only files and flags. So the classification
is declared beside each argument's definition, and the compiler refuses an
argument that has none. This section describes how camdl parses its arguments,
the machinery the classes are attached to.

camdl parses arguments with [clap](https://docs.rs/clap/4) (version 4), the
standard Rust argument parser. clap is used through _derive macros_: code the
Rust compiler runs on a type definition, at compile time, to generate an
implementation for that type. A struct marked `#[derive(Args)]` becomes a set of
command-line options, one per field, and each field's `#[arg(..)]` attribute
says how it is spelled and parsed. `#[arg(long)]` on a field `particles` makes
the option `--particles`, parsed as the field's type. The top-level struct `Cli`
is marked `#[derive(Parser)]` (`main.rs:93`), which adds the entry point that
reads `std::env::args`, and its subcommands are the variants of an enum marked
`#[derive(Subcommand)]`, one per command, such as `Pfilter(args::PfilterArgs)`.

A command's options are often grouped into smaller structs that several commands
share. `#[command(flatten)]` on a field whose type is itself an `Args` struct
copies that struct's options into the parent, so they are spelled and parsed as
if declared there. `PfilterArgs` flattens `ModelOverrides` (`--params`,
`--param` and the like), `ScenarioArgs` and `PfilterCore` (`--dt`, `--seed`,
`--parallel` and the other inference settings). The global options, such as
`--no-licm` and `--verbosity`, sit on `GlobalArgs`, which `Cli` flattens with
`global = true` so they can be written after any subcommand.

The store adds a derive of its own to the same structs.
`#[derive(store::Inputs)]` reads the struct beside clap and requires an
`#[input(..)]` attribute on every field, naming the field's class: what the
input does to a stored result ([Input classes](#input-classes)). A field that
clap flattens is marked `#[input(group)]`, and its own fields carry their own
classes. clap ignores the `#[input]` attribute and the store ignores `#[arg]`,
so one struct declares both how an option is parsed and what it means for the
store:

```rust
// rust/crates/cli/src/args/mod.rs:2212 (trimmed)
#[derive(Args, Clone, store::Inputs)]
pub struct PfilterArgs {
    #[input(identity, vary = crate::input_fixtures::other_model)]
    pub model: PathBuf,                 // the positional model file

    #[command(flatten)]                 // clap: PfilterCore's options appear here
    #[input(group)]                     // store: each of its fields has its own class
    pub inference: PfilterCore,

    /// Number of particles
    #[arg(long)]                        // clap: the option --particles
    #[input(identity)]                  // store: it can change the stored result
    pub particles: usize,
    // ...
}
```

A struct can also describe a settings file (`#[derive(FileInputs)]`) or a list
of environment variables (`#[derive(EnvInputs)]`), under the same rules.

## The running example: `pfilter`

This page uses `camdl pfilter` as its running example, and we step through how
it uses the store, section by section.

A particle filter estimates the likelihood of observed data under a stochastic
model at one set of parameter values. It simulates many copies of the epidemic
(the particles) forward between observations, weights each by how well it
explains the next observation, and resamples them by those weights. The product
of the average weights is an estimate of the likelihood.
`camdl pfilter model.camdl --params p.toml --data cases.tsv --particles 500`
runs one such filter at the parameter vector `p.toml` gives and reports the
estimated log-likelihood, which it stores in `loglik.toml`. Optional flags ask
for diagnostics of the same run, such as the per-observation trace (`--trace`)
or sampled latent paths (`--save-paths`), and `--sweep-file` runs one filter per
row of a table of parameter values.

`pfilter` makes a good example because it exercises most of the store in one
command: its identity is small, it is written in place so its progress can be
watched, it has seven by-products, it distinguishes failures that are results
from failures that are not, and a sweep makes it a batch.

## Commands on the store

Five commands store results: `camdl simulate`, `camdl pfilter`, `camdl fit run`,
`camdl predict` and `camdl profile`. Each opens the store at `--store DIR`, else
`$CAMDL_STORE`, else `./camdl-store` (`camdl_store.rs:97-108`), and each has a
`--force` and a `--label`. `camdl pfilter --no-store` computes in a temporary
store that is deleted when the command ends.

`camdl list`, `camdl show` and `camdl cat` read stored results of every artifact
kind (`browse.rs`), as do `camdl fit summary`, `camdl fit table` and every
command that takes a stored fit with `--fit`. A stored fit named by its folder
is read in the store that holds the folder
([`fit_store::store_holding`](rust:cli::fit_store::store_holding),
`fit_store.rs:1645`). `camdl status`, `camdl verify`, `camdl clean` and
`camdl reindex` are the store library's maintenance commands, run against the
same root (`store_commands.rs:15`).

## Stored results

A stored result is one folder holding the files one computation wrote, two
record files the store wrote, and nothing else. Four things describe it: its
artifact kind, which says what sort of result it is; its identity, which
determines its output; its labels, which people recognize it by; and its run id,
which names it. The subsections below take each in turn.

### Artifact kinds

An artifact kind is a sort of stored result: a filter run (`pfilter`), its trace
(`trace`), a compiled model (`compiled_model`). In code, an artifact kind is a
Rust type that implements the trait `Artifact`. A _trait_ is Rust's interface: a
set of methods, constants and _associated types_ (types each implementation
chooses) that a type promises to provide. `Artifact` asks for the artifact
kind's name, which is also the name of its directory in the store; how its files
reach disk; the type of its identity; and the type of its labels.

```rust
// rust/crates/store/src/artifact.rs:15
pub trait Artifact: 'static {
    const KIND: &'static str;          // "pfilter"; also the artifact kind's directory name
    const WRITE: WriteMode;            // Atomic or InPlace
    type Identity: Identity;           // what determines the result's output
    type Attributes: Serialize + DeserializeOwned + Clone + 'static;
}

// :30
pub trait Primary: Artifact {
    fn layout(identity: &Self::Identity, labels: &Self::Attributes) -> Placed;
}

// :37
pub trait ByProduct: Artifact {
    type Of: Primary;
    fn upstream(identity: &Self::Identity)
        -> &<<Self::Of as Artifact>::Identity as Identity>::Record;
    fn label(identity: &Self::Identity, labels: &Self::Attributes) -> String;
}
```

An artifact kind is one of two sorts. A `Primary` kind is the main output of its
own computation, and its own `layout` places it. A `ByProduct` kind is a second
output of another artifact kind's computation, such as the trace of a filter
run, and the library places it inside the upstream result's folder. Only a
`Primary` kind can be begun on its own. A third trait, `ContentFiles`, marks a
primary kind whose identity is its content: the digests of its result files
([Content artifacts and dependencies](#content-artifacts-and-dependencies)).

camdl's filter run is a primary kind written in place:

```rust
// rust/crates/cli/src/pfilter_store.rs:142
pub(crate) struct Pfilter;

impl Artifact for Pfilter {
    const KIND: &'static str = "pfilter";
    const WRITE: WriteMode = WriteMode::InPlace;
    type Identity = PfilterIdentity;
    type Attributes = PfilterAttributes;
}
```

Every artifact kind a project stores is registered once in a `Kinds` value
passed to [`Store::open`](rust:store::storage::Store::open). The commands that
read across artifact kinds use it to parse any record, and registering one
`KIND` twice is refused when the store opens
([`Error::KindRegisteredTwice`](rust:store::error::Error::KindRegisteredTwice)).
camdl's list is `camdl_kinds()` (`camdl_store.rs:79`), nineteen artifact kinds
in all:

| Module             | Primary kinds                                  | By-products                                                                                        |
| ------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `camdl_store.rs`   | `compiled_model`, `quantities`, `data_table`   | none                                                                                               |
| `sim_store.rs`     | `sim`                                          | `obs`, `event_log`, `sim_quantities`                                                               |
| `pfilter_store.rs` | `pfilter`                                      | `trace`, `pf_health`, `filtering`, `filtered_streams`, `final_state`, `saved_paths`, `prequential` |
| `fit_store.rs`     | `fit` (a fit problem), `fit_run`, `prediction` | none                                                                                               |
| `profile_store.rs` | `profile_point`                                | none                                                                                               |

The three artifact kinds of `camdl_store.rs` are content artifacts.

### Identities

An identity is a typed value that determines every byte of a result's output
files. If two requests have equal identities, they would compute the same files,
so the second can reuse the first's. The filter run's identity is the compiled
model it ran, its settings, the parameter values, the interventions kept, the
seed, and the stored fit problem `--fit` named, if any:

```rust
// rust/crates/cli/src/pfilter_store.rs:107
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "pfilter", version = 1)]
pub(crate) struct PfilterIdentity {
    pub model: Dep<CompiledModel>,     // a dependency on a stored compiled model
    pub config: PfilterConfig,         // particles, replicates, dt, data, objective, ...
    pub params: ResolvedParams,        // every parameter's value, BTreeMap<String, Finite>
    pub scenario: ActiveInterventions, // the interventions the scenario selection keeps
    pub seed: ProcessSeed,
    pub fit: Option<Dep<FitProblem>>,  // the stored problem `--fit` named
}
```

The identity holds resolved values, not the arguments as typed: the parameter
values after every source has been applied, not the name of the params file. Two
command lines that resolve to the same values therefore have one identity.

### Keys

The store cannot compare identities directly. It has to find a result again in a
later process, from records on disk, among thousands of others. So it reduces
each identity to a key: a fixed-size fingerprint, the SHA-256 digest of a
canonical byte encoding of the value. SHA-256 is a cryptographic hash, which
means two different encodings in practice never produce the same digest. Two
identities have the same key exactly when they encode to the same bytes, and the
encoding is designed so that this happens exactly when the identities are equal
in every field. A key is 32 bytes, written as 64 hexadecimal digits.

#### Dependencies between results

Many results are computed from other stored results: a filter run reads a stored
compiled model and stored data tables, and a prediction reads a stored fit run's
posterior draws. An identity names such an input with a dependency, `Dep<T>`,
rather than by copying the input's value or by naming its path. A `Dep` names
the upstream result by its run id and the digest of the file it consumes. Two
things follow. Changing the upstream (a model edited, a fit run again under
other settings) changes the `Dep`, and so changes the key of every result
computed from it, transitively down the chain. And a computation can read the
upstream only through its `Dep`, which checks the file against the recorded
digest, so it cannot silently read another copy of the file, an edited file at
the same path, or a different result with the same name. The `Dep` type and the
reads it allows are described in
[Content artifacts and dependencies](#content-artifacts-and-dependencies).

#### The encoding

The key of a value is a `store::Key`, computed by
`ContentAddressed::content_hash` (`hash.rs:177`). It hashes `HASH_VERSION` (9,
as a little-endian `u16`) and then the bytes the value's `hash_into` writes.

`#[derive(Identity)]` writes those bytes for a struct as its tag string, its
version as a `u16`, and then, for each field in name order, the field's name and
the field's own bytes. An enum writes its tag, version and variant name, then
the variant's fields. Every variable-length value is written with its length
first.

An `f64` cannot be hashed at all. A float enters an identity as `Finite`, which
refuses NaN and infinity and maps `-0.0` to `+0.0` (`identity.rs:142`), so that
two equal numbers always have one key.

These rules decide which source changes re-key stored results. Renaming a field,
a variant or a tag changes every key of that type, while moving the type to
another module or reordering its fields does not. A type's `version` is raised
when the same values would now produce different stored bytes.

### The record form

A stored result has to carry its identity on disk, so that a reader can see what
produced it and the store can check its run id. The identity type itself is not
always suitable for that. A field of type `Hashed<T>` holds a whole value, such
as a data table, together with its digest; the result's own files already hold
that value, and writing it again into the record would duplicate it. So
`#[derive(Identity)]` also generates a record form of each identity type, here
`PfilterIdentityRecord`, which is what goes to disk.

In the record form each field is replaced by its own record: a `Hashed<T>`
(content the result itself stores) becomes its digest, a nested identity becomes
its record, and a `Dep` is its own record. The record form hashes to the same
bytes as the identity, so an identity and its record have one key. The record is
what `identity.json` holds and what readers get back from `Stored::identity()`.
A by-product's identity holds its upstream's record, and so hashes the upstream
exactly as the upstream hashes itself.

### The run id

The identity's key is not yet the result's name. The same identity computed by
two different builds of camdl, or under different engine switches, may produce
different bytes, and the same identity value could in principle belong to two
artifact kinds. The run id therefore combines the artifact kind, the identity's
key and an ambient key, and it is computed by one function:

```rust
// rust/crates/store/src/artifact.rs:87
pub(crate) fn run_id(kind: &str, ambient: &Key, identity: &Key) -> Key {
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    h.write_str(kind);        // u64 length, then the bytes
    h.write_key(ambient);     // 32 bytes
    h.write_key(identity);    // 32 bytes
    h.finalize()
}
```

`kind` is `A::KIND`. `identity` is the key of the result's identity. `ambient`
is the key of the ambient identity, a value the project passes to
[`Store::open`](rust:store::storage::Store::open) that every run id of that
store includes. camdl's ambient identity is the build fingerprint and the
environment variables that can change a result:

```rust
// rust/crates/cli/src/camdl_store.rs:31
#[identity(tag = "engine", version = 1)]
pub(crate) struct Engine {
    pub build: String,              // crate::version::ENGINE_ID
    pub env: CamdlEnvIdentity,      // CAMDL_EVAL_UNRESOLVED, CAMDL_NO_BINDING_CACHE,
}                                   // every set CAMDL_EXPERIMENTAL_*
```

`ENGINE_ID` (`version.rs:37`) is 32 hex digits that the build script
(`build.rs`, `engine_fingerprint.rs`) computes from the source, uncommitted
changes included, `Cargo.lock`, the binary's resolved features, the profile,
`rustc -vV`, `RUSTFLAGS` and the target. A rebuilt worktree therefore has a
different ambient key from its clean commit, and reuses none of its results.

`camdl_store::ambient()` (`camdl_store.rs:55`) wraps the value in
`Build::Known`, or in `Build::Incomplete` when the build script could not
determine some component (`ENGINE_REUSE`, `version.rs:59`). Under
`Build::Incomplete` the store reuses nothing: every `begin` behaves as under
`Reuse::Force` (`storage.rs:790`).

`CAMDL_NO_LICM` and `CAMDL_NO_CONSTANT_FOLD` are not in the ambient identity,
because they change the compiled model's bytes, which the run's identity already
holds.

### Attributes

A result's attributes are what a user recognizes it by but what does not
determine it. For a filter run they are the model file's stem, the seed as
typed, and for a point of a `--sweep-file` design, its row and values
(`PfilterAttributes`, `pfilter_store.rs`). Each kind names their type as
`Artifact::Attributes`.

Attributes are passed to `begin` beside the identity. The store never hashes
them and never looks inside them. They are written into the invocation record,
and the first writer's attributes name the result's folder through `layout`. A
second request for the same result under other attributes finds the folder
already named.

A `--label L` on the command line is different: it names the invocation, not a
result ([The invocation record](#the-invocation-record)). `StoreRequest::new`
parses it with `Label::parse` (`label.rs`), so every command applies one rule:
one to 64 characters after trimming, each an ASCII letter or digit, a space, a
comma, a dot, an underscore or a hyphen, and not `.` or `..`. The rule keeps out
`~`, which `@L~n` reads as a previous holder, and anything that is not a
directory name of the link tree.

### The record files

Every result folder holds two files the store writes: `identity.json`, which
records what the run id was computed from, and `run.json`, which records
everything else about the run. Here is `identity.json` from the filter run built
for this page
([Example: the store after two pfilter runs](#example-the-store-after-two-pfilter-runs)),
abbreviated:

```json
// identity.json
{
  "format": 1,
  "kind": "pfilter",
  "run_id": "f395dfde1a35b13ad7542a4a5cb64679db7887256c392ba6b1e01efd3aa5a915",
  "hash_version": 9,
  "ambient": { "build": "ac4131137c1ef405b220f6dc0650294d",
               "env": { "eval_unresolved": false, "experimental": {}, "no_binding_cache": false } },
  "identity": {
    "config": { "particles": 500, "replicates": 1, "dt": 0.1, "objective": "Full",
                "binomial": "Btpe", "training_rows": null,
                "data": { "in_bed": { "kind": "data_table", "run_id": "2e05b331…",
                "file": "table.tsv", "digest": "e118bad6…", "build": "380dcb72…" } }, "…": "…" },
    "fit": null,
    "model": { "kind": "compiled_model", "run_id": "cb8e23e2…", "file": "model.ir.json",
               "digest": "6b60d30e…", "build": "380dcb72…" },
    "params": { "values": { "I0": 3.0, "N0": 763.0, "R0": 3.75, "gamma": 0.4 } },
    "scenario": { "names": [] },
    "seed": { "seed": 1 }
  },
  "parent": null
}
```

`identity.json` holds everything the run id is computed from: the artifact kind,
the ambient identity's record and the identity's record. `parent` is `null` for
a result under its artifact kind's directory, and names the upstream for a
by-product (`"relation": "upstream"`) or the parent for a result placed with
`Layout::inside` (`"relation": "inside"`).

`run.json` holds the status and everything recorded about the run but not hashed
(`RunFile`, `record.rs:145`):

| Field                 | What it holds                                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `status`              | `running`, `completed` or `failed`                                                                          |
| `owner`               | The process that claimed the run: host, process id, process start time                                      |
| `invocation`          | The sequence number of the invocation record that claimed it                                                |
| `started`, `finished` | When the claim was taken and when the run ended                                                             |
| `given`               | Every recorded input as typed, by field name, groups nested                                                 |
| `execution`           | The execution inputs' values                                                                                |
| `files`               | The manifest: each file's role, and for result and provenance files its size, modification time and SHA-256 |
| `deps`                | Each `Dep` the identity holds: artifact kind, run id, file, digest, and the build it was made under         |
| `lineage`             | Inputs recorded and not hashed, such as `{"from": "input", "field": "data.in_bed", "digest": "e118bad6…"}`  |
| `reads`               | Each file of another result the running computation read through the store                                  |
| `recorded`, `failure` | A content artifact's provenance values; for a failed in-place run, `{ "operation", "reason" }`              |

A file's role is set when the run opens it, and decides how the store treats the
file:

- A result file is determined by the identity, is in the manifest and is
  compared whenever the result is computed again. `pfilter`'s `loglik.toml` is
  one.
- A provenance file is in the manifest and never compared: `pfilter`'s
  `provenance.json` says where each scored value came from.
- An attempt file describes one attempt and is outside the manifest:
  `progress.json` and `events.jsonl`, which
  [`Run::progress`](rust:store::run::Run::progress) and
  [`Run::event`](rust:store::run::Run::event) write.

Every record carries `format`. A reader that finds another version refuses the
file by name ([`Error::OldRecord`](rust:store::error::Error::OldRecord),
`record.rs:273`), and a result whose record has another version is treated as
damaged and computed again.

### Statuses

`run.json` states one of three statuses, and readers derive two more.

| Status      | Written by                                                    | Meaning                                                                     |
| ----------- | ------------------------------------------------------------- | --------------------------------------------------------------------------- |
| running     | the claim                                                     | a process holds the run id                                                  |
| completed   | [`Run::finish`](rust:store::run::Run::finish)                 | the manifest is written and the result is whole                             |
| failed      | [`Run::fail`](rust:store::run::Run::fail), or a dropped `Run` | an in-place run stopped; the folder stays with the reason, if one was given |
| interrupted | nobody; derived                                               | the record says running and its owner is dead                               |
| damaged     | nobody; derived                                               | a completed folder whose files or records fail a check                      |

A completed result is damaged when a listed file is missing or has a different
size, when a listed file's modification time differs and its SHA-256 differs
too, when an unlisted file or directory sits in the folder, or when a record
does not read (`check::damage`, `check.rs:39`). A subdirectory is not unlisted
when every result folder below it names this result as its parent (its
by-products, or a result placed inside it).

A file whose size and modification time both match is not hashed by this check.
A copy of a store that keeps modification times is therefore checked cheaply,
and one that does not (`cp -r`, `rsync` without `-t`) is checked by digest.
Every read through the store computes the digest of what it reads, whatever this
check found.

Only a completed, undamaged result is ever reused or read as a result. An atomic
result that fails leaves no folder, so it has no status of its own; its failure
is kept in the invocation record. A reader asks for a result in any status with
[`Store::inspect`](rust:store::storage::Store::inspect), which returns an
`Entry` whose `EntryState` is one of `Running { owner, liveness }`, `Completed`,
`Damaged { what }`, `Failed { operation, reason }` and `Interrupted { owner }`
(`stored.rs:379`).

## Input classes

Every input a command declares has a class, assigned where it crosses the input
boundary. The class states what the input does to stored results: whether it
enters the identity, changes which results a request has, asks for a by-product,
only affects how the computation runs, only copies or renders stored files
somewhere, only addresses the store, is only recorded, or does nothing the store
needs to know about.

The classes matter because the store's first guarantee, that a result is reused
only when the request would have produced it, rests on them. An input that can
change the output but is classed as affecting only execution would be left out
of the identity, and a request that differs in it would reuse a result it would
not have computed. Each class is therefore a claim about the command's
behaviour, and each is checked by running the command
([The class checks](#the-class-checks)).

### The classes

| Class       | Meaning                                                      | Goes to      | Example                                                   |
| ----------- | ------------------------------------------------------------ | ------------ | --------------------------------------------------------- |
| `identity`  | can change a result; reaches the identity through resolution | `Declared`   | `pfilter --particles`, `--seed`, `--data`, `--dt`         |
| `expands`   | decides which results a batch has                            | `Declared`   | `pfilter --sweep-file`, `simulate --seeds`                |
| `adds`      | asks for a by-product                                        | `Declared`   | `pfilter --trace`, `--save-paths`; `simulate --event-log` |
| `execution` | used by the engine; must not change a result                 | `Execution`  | `--parallel`, `pfilter --pf-max-substeps`                 |
| `export`    | a path to write a copy or a rendering of stored files to     | `Exports`    | `-o`                                                      |
| `render`    | how an export is rendered                                    | `Exports`    | `simulate --dates`, `profile --format`                    |
| `store`     | a request to the store                                       | `Request`    | `--label`, `--store`, `--force`, `pfilter --no-store`     |
| `recorded`  | kept on record; nothing reads it                             | `Given` only | none in camdl                                             |
| `skip`      | affects only what is printed; recorded nowhere               | nowhere      | `--verbosity`, `--progress`                               |

The "Goes to" column names the part of the split arguments the input lands in
([How the derive splits arguments](#how-the-derive-splits-arguments)).

A tenth attribute, `group`, is not a class. It marks a field that is itself a
struct deriving `Inputs`, the store's counterpart of clap's
`#[command(flatten)]`
([How camdl reads its command line](#how-camdl-reads-its-command-line)), such as
`ModelOverrides` or `PfilterCore`. The group's fields go wherever their own
classes send them, and are listed under the group's name, as
`args.inference.parallel`.

### One flag, two classes

The same flag can have different classes in different commands, because what it
does to the stored result differs between them. `--pf-max-substeps` shows this.
It caps the particle-substeps one filter evaluation may execute: the number of
particles times the simulation steps of size `dt` between consecutive
observations, `n_particles × Σ_k ceil((t_k − t_{k−1}) / dt)`. The default is
10¹⁰ (`ITER_BUDGET`, `sim/src/inference/degeneracy.rs`), far above any real fit,
so the cap catches a misconfiguration such as a near-zero `dt`. The filter
computes each window's cost from that closed form before moving the particles
(`particle_filter.rs:524`), so whether the cap trips depends on the observation
times, `dt` and the particle count, never on the parameter values or the random
draws, and is the same on every machine.

In `pfilter` the cap is `execution`. A run that hits the cap is refused: the
command exits with status 3, and the in-place folder is left marked failed with
no result file. A run that stays under the cap computes the same log-likelihood
whatever the cap was. So for one request the cap decides only whether a stored
result comes to exist, never what a stored result holds, which is what the
`execution` class claims. The class check runs the base fixture with the
engine's default cap and with a cap of 400, the cost of one pass of that
fixture, and requires both to store the same bytes under the same run id; a
second test, `a_budget_below_one_pass_is_refused_and_stores_nothing`, runs it at
399 and requires a failed run with no result file (`input_fixtures.rs:116`,
`pfilter_store/tests.rs:756`).

In `profile` the cap is `identity`. A profile point is an IF2 fit at one value
of the focal parameter, and an IF2 cell that hits the cap does not end the
command: the point is stored as completed, with a log-likelihood of `-inf`
(`profile.rs:1814`). The cap therefore changes what a stored point holds, and
two requests that differ in it must not share a point. The two commands declare
the flag in separate structs, `PfilterCore` and `ProfileCore`, for this reason
(`args/mod.rs:254`, `:330`).

`pfilter --save-final-state` is `identity` for a similar reason. Under a fit
configuration's `holdout_after`, it binds only the training rows, which changes
the log-likelihood, and the identity records how many rows each stream bound
(`PfilterConfig::training_rows`).

### How the derive splits arguments

Once clap has parsed a command line into a struct, the command needs to send
each input where its class says: identity inputs to the code that builds the
identity, execution inputs to the engine, store inputs to the store. The
`Inputs` derive generates that routing, so that no command sorts its inputs by
hand and no input can be forgotten.

The derive also refuses, at compile time, the declarations that would make a
class check meaningless (`store-derive/src/lib.rs:399-417`):

- a field without an `#[input(..)]` attribute;
- an `execution` field without `exercised_by = <fixture>`, a fixture on which
  the input is used, so that its check is not vacuous (an environment variable
  is excepted);
- an `execution` field of type `InputFile`, because a file that is read is an
  input to the result;
- an `export` or `render` key in a settings file.

For each struct, the derive generates four part structs and one implementation
of the `Inputs` trait:

```rust
// rust/crates/store/src/inputs.rs:72
pub trait Inputs: Sized {
    type Declared;    // identity, expands and adds fields; a group as its own Declared
    type Execution: Serialize;
    type Exports;     // export and render fields
    type Request;     // store fields
    fn fields() -> &'static [FieldInfo];   // every field with its class, groups' prefixed
    fn split(self) -> Split<Self>;
    fn varied(&self, field: &str) -> Self;          // one identity or render field changed
    fn exercised(field: &str) -> (Self, Self);      // an execution field's fixture, two values
}
```

A part is named after the struct with a trailing `Args` removed, so
`PfilterArgs` gives `PfilterDeclared`, `PfilterExecution`, `PfilterExports` and
`PfilterRequest`.

`Declared`, `Exports` and `Request` have private fields and one method,
`into_parts`, which returns every field as a tuple in declaration order. Code
can reach a declared input only by taking the whole part apart, so an input
added to a struct breaks every caller until the caller places it. `Execution`
has public fields and derives `Serialize`, because the engine reads them and
`run.json` records them. A group contributes its own part to each of its
parent's parts, `()` where it has no field of that class.

The split reads no file and cannot fail. Beside the four parts it returns
`Given`, the inputs as given, for the records: every field except `store` and
`skip` ones, by name, a group's fields nested under the group's name, with the
command line.

#### The split in `pfilter`

When a user runs `camdl pfilter …`, clap parses the command line into
`PfilterArgs` (the struct shown in "How camdl reads its command line"), and the
global options such as `--no-licm` into `GlobalArgs`. Before calling the
command, `main.rs` puts the two together in one value, a `PfilterInvocation`
(`args/mod.rs:2434`): `PfilterArgs` held as a group, plus the global options
that reach this command (`--no-licm`, `--no-ir-cache`, the verbosity), so that
those options carry classes too. `PfilterInvocation` derives `Inputs` but not
clap, since it is built in code rather than parsed. That value is what the
`pfilter` command receives, and it splits it once, at the top:

```rust
// rust/crates/cli/src/pfilter.rs:202
let (declared, execution, exports, request, given) = store::Inputs::split(inv).into_parts();
let (declared, no_licm) = declared.into_parts();
let crate::args::PfilterInvocationExecution { args: execution, no_ir_cache } = execution;
// ...
let ((), (), _inference_request, (), no_store, force, label, root) = request.into_parts();
let request = crate::camdl_store::store_request(label, force, root);
// ...
let (config, requests, _forcing_route) =
    crate::pfilter_config::resolve(declared, compile, &out, &crate::camdl_store::root_of(&request));
```

The declared part goes to one resolution function,
[`pfilter_config::resolve`](rust:cli::pfilter_config::resolve), which consumes
it in full and returns a `ResolvedPfilterConfig`: the parameter values after
every source, the data bindings, the settings.

From that resolved configuration, `pfilter.rs` compiles the model and reads the
data. It stores both in the store, and only then builds the
[`PfilterIdentity`](rust:cli::pfilter_store::PfilterIdentity) from the stored
model's and tables' dependencies and the resolved values (`pfilter.rs:1176`).

An input that a command accepts and does not use is consumed with
`store::release(value, "reason")`, so the decision is written in the source
beside the input: `pfilter.rs:567` releases the compiled model's quantities,
which a filter run does not report. Nothing records the reason at run time.

#### From arguments to a folder

The whole path, from the parsed arguments to the folder the result is stored in:

```text
PfilterInvocation ──Inputs::split──┬─ Declared ──pfilter_config::resolve──► ResolvedPfilterConfig
  (clap + global options)          │                                          │ compile; read data;
                                   │                                          │ Store::put model, tables
                                   │                                          ▼
                                   │                                    PfilterIdentity ──content_hash──► identity key ─┐
                                   ├─ Execution ──► the engine; run.json "execution"                                    │
                                   ├─ Exports ────► -o: a rendering of the stored result                                │
                                   ├─ Request ────► StoreRequest: --label, --force, --store; --no-store                 │
                                   └─ Given ──────► run.json "given"; the invocation record                             │
                                                                                                                        │
  camdl_store::ambient() ── Engine { build, env } ──content_hash──► ambient key ────────────────────────────────────────┤
                                                                                                                        ▼
                                           run id = SHA-256( HASH_VERSION ‖ "pfilter" ‖ ambient key ‖ identity key )
                                                                                                                        │
  Pfilter::layout(identity, labels) ─────────────────────────────────────────────────────────────────────────────────────┤
                                                                                                                        ▼
  <root>/<version>-<12 hex ambient key>/pfilter/<model>-<12 hex>/pf-n…-<12 hex>/params-<12 hex>/seed_<seed>-<12 hex run id>/
```

### The class checks

A class is a claim about what an input does to stored results, and
`store::check_input_classes` (`harness.rs:110`) tests that claim for every field
of a command by running the command. It takes the command's `fields()` and a
harness the project writes once per command. The harness says how to build the
base invocation, how to vary one field, and how to run an invocation in a fresh
store and read back what it stored:

```rust
// rust/crates/store/src/harness.rs:22
pub trait InputHarness {
    type Invocation: Clone;
    fn base(&self) -> Self::Invocation;
    fn varied(&self, field: &FieldInfo) -> Self::Invocation;
    fn exercised(&self, field: &FieldInfo) -> (Self::Invocation, Self::Invocation);
    fn shows_use(&self, field: &FieldInfo, seen: &Observed) -> bool;
    fn run(&self, invocation: &Self::Invocation, store: Option<&Path>) -> Result<Observed>;
}
```

`run` returns an `Observed`, which lists each stored result's artifact kind, run
id, upstream and result-file digests; each export's digest; which exports are
byte copies of a stored file, and of which; and `effects`, what the harness read
of the run besides its files, which only `shows_use` looks at. Results that are
content artifacts are marked and left out of the run-id comparisons, so an input
that moves only a stored data table, not the filter run, fails the identity
check. Each class asserts the following.

| Class                       | Runs                                                 | Fails when                                                                                                                             |
| --------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `identity`                  | the base, and the base with the field varied         | every run id is unchanged                                                                                                              |
| `expands`                   | the base, and a larger set containing it             | a member of the smaller set changes run id, or the larger set adds no result                                                           |
| `adds`                      | the base, and the base with the by-product asked for | the upstream's run id changes, or no new by-product is stored                                                                          |
| `execution`                 | the field's `exercised_by` fixture at two values     | any result file, run id or export differs, or `shows_use` returns false for either run                                                 |
| `export`                    | the base with the export set, twice into one store   | no export is written; a copy differs from the stored file it copies; a rendering made again from the same stored files has other bytes |
| `render`                    | the base, and the base with the field varied         | any stored file or run id differs                                                                                                      |
| `store`, `recorded`, `skip` | nothing                                              |                                                                                                                                        |

The second value of a varied field comes from the field's type through the
`Vary` trait (integers plus one, `f64` doubled or 1 for 0, `bool` negated,
`String` suffixed, and `Option`, `Vec` and `BTreeMap` of these), or from the
function the field names with `vary = path::to_fn` (`inputs.rs:455-516`). A
`PathBuf` has no `Vary`, so a file field without `vary` does not compile.
camdl's second values and execution fixtures are in `input_fixtures.rs`:
`other_model()` returns `sir_other.ir.json`, and `pfilter_parallel()` returns
the base fixture at two `--parallel` values.

#### The `pfilter` harness

`pfilter`'s test harness (`pfilter_store/tests.rs:409-568`) runs the release
`camdl` binary in a fresh copy of the fixture files with a cleared environment.
Its `for_field` picks, for each field, a base on which that field is live: the
`--scenario` check starts from a model with a named scenario, and the
`--n-paths` check from a base that already asks for `--save-paths`.

Its `shows_use` compares, for each execution input, the value the run recorded
in `run.json` with the effect the run reports in its debug log: `--parallel N`
filters on N threads, the budget handed to the engine is `--pf-max-substeps` or
the engine's default, the data loader reads the `--time-format` given, and
`--no-ir-cache` leaves the IR cache empty. An execution input with no effect
named there fails its check.

Two tests run the harness. `every_pfilter_input_keeps_its_class` runs the check
over every field, and `every_pfilter_argument_is_declared` checks that every
argument clap accepts for `pfilter` is a declared field and every declared field
is one clap accepts. Each other command on the store has the same pair
(`every_<command>_input_keeps_its_class`, `every_<command>_argument_is_declared`
in `sim_store/tests.rs`, `fit_command/tests.rs`, `predict_command/tests.rs` and
`profile_store/tests.rs`).

## Layout

Every result has a run id, and the store could keep each result in a folder
named by its run id alone, all in one flat directory: thousands of folders named
by 64 hexadecimal digits, side by side. The store itself would need nothing
more, because it finds results by run id. More structure is wanted for other
reasons:

- People browse the store. A folder `seed_1-f395dfde1a35` under the model's and
  the settings' directories says what it holds; a bare hash does not.
- Related results belong together. Every point of one profile should sit under
  one directory, so that the profile can be looked at, copied or deleted as a
  unit.
- Some results belong to another result. A fit run is a run of one fit problem,
  a prediction from a posterior is made from one fit run, and a by-product is a
  second output of its upstream. Each should sit inside its parent's folder.

Each artifact kind's `layout` function states this structure for its results.
For example, a fit problem, the fit runs on it and a prediction from one run's
posterior are nested like this, under the build directory
([The build directory](#the-build-directory)):

```text
fit/sir_r0-<12 hex>/                  a fit problem
  problem.toml  partition.json  inputs/ ...
  runs/
    <algorithm>-<12 hex>/             group: the method's settings
      seed_1-<12 hex>/                a fit run of this problem
        predict/<mode>-<12 hex>/      a prediction from this run's posterior
```

and a profile's points are grouped by the problem they profile, the focal
parameter and the method, so the whole grid of one profile is one directory:

```text
profile_point/<model>-<12 hex of the problem's key>/<focal>-<12 hex>/<method>-<12 hex>/
  seed_1-start_0-<12 hex>/            one point: a focal value, a seed and a start
  seed_1-start_1-<12 hex>/
  ...
```

The path is a view for people browsing the store, and nothing reads meaning out
of it. The seed, the model and the settings of a result come from its identity
record, never from its folder's name, and the store confirms a folder against
the run id in its `identity.json` before using it
([Finding a result's folder](#finding-a-results-folder)).

### Placing a result

A `Primary::layout` function builds its path from a `Layout` and returns a
`Placed`:

```rust
// rust/crates/store/src/layout.rs
impl Layout {
    pub fn in_kind() -> Self;                                    // :77  <build dir>/<KIND>/
    pub fn inside<P: Artifact>(parent: &Id<P>, subdir: &'static str) -> Self;   // :83
    pub fn group<T: ContentAddressed + ?Sized>(self, label: &str, part: &T) -> Self; // :89
    pub fn result(self, label: &str) -> Placed;                  // :95  ends the layout
}
```

`in_kind` starts at the artifact kind's directory. `inside` starts at a
subdirectory of a parent result's folder, wherever that folder is: a `FitRun`
sits inside its `FitProblem` at `runs/`, and a prediction from a posterior
inside its `FitRun` at `predict/`.

Each `group` adds one directory level,
`<label>-<first 12 hex of the part's key>/`. `group` takes a part of the
identity, not a key the caller computed, so a layout cannot group by something
outside the identity. Results whose identities share that part share the
directory.

`result` names the result's own folder, `<label>-<first 12 hex of the run id>/`.
Grouping chooses only where a folder sits; every field of the identity is in the
run id whether or not a group uses it.

camdl's model-running artifact kinds share one grouping: the model, then the
settings, then the parameters. `pfilter`'s layout is that grouping followed by
the seed:

```rust
// rust/crates/cli/src/camdl_store.rs:150
pub(crate) fn camdl_groups<S: ContentAddressed>(
    model_label: &str, model: &Dep<CompiledModel>,
    settings_label: &str, settings: &S,
    params: Option<(&str, &ResolvedParams)>,
) -> Layout

// rust/crates/cli/src/pfilter_store.rs:151
impl Primary for Pfilter {
    fn layout(id: &PfilterIdentity, labels: &PfilterAttributes) -> Placed {
        camdl_groups(&labels.model, &id.model, &id.config.label(), &id.config,
                     Some(("params", &id.params)))
            .result(&format!("seed_{}", labels.seed))
    }
}
```

Every label is made safe for one path component by `path_label`
(`layout.rs:27`): lowercased, each character outside `a-z`, `0-9`, `_`, `-` and
`.` replaced by `_`, a leading `.` replaced by `_`, and a label longer than 64
bytes cut to 46 bytes followed by `..` and 16 hex digits of the label's SHA-256.
`pf-N500-r1-dt0.1` therefore becomes `pf-n500-r1-dt0.1`, and the version
`0.1.0+ca899034b` becomes `0.1.0_ca899034b`.

### The build directory

Results made by different builds of camdl never share a run id, because the
ambient key is in every run id ([The run id](#the-run-id)). The store can also
keep them in separate directories, so that everything one build made can be seen
or removed together.

[`Store::open`](rust:store::storage::Store::open) takes a `TopLevel`, which has
no default. Under `TopLevel::ByBuild { label }` every result placed with
`in_kind` sits under a directory for the ambient key,
`<label>-<12 hex of the ambient key>/`; under `TopLevel::ByKind` the artifact
kinds' directories are at the root. The choice is recorded in `store.json`
(`{"format": 1, "top_level": "by_build"}`), and opening an existing store the
other way is refused with
[`Error::TopLevelDiffers`](rust:store::error::Error::TopLevelDiffers).

camdl uses `ByBuild` with its readable version, `VERSION_SHORT`, as the label
(`camdl_store.rs:74`): `0.1.0+<commit>` for a clean build,
`0.1.0+<commit>.dirty-<hash>` for one with uncommitted changes. So one directory
holds everything one build made under one set of engine switches, and removing
it removes exactly that. The label is for people; the 12 hex digits after it are
what distinguish two builds, since two builds with one label (a rebuilt dirty
worktree, another `RUSTFLAGS`) have different `ENGINE_ID`s.

A `Dep`'s recorded `build` is the 64 hex digits of the ambient key it was made
under, which is how a result made by one build can name a model stored by
another. [`Store::scratch`](rust:store::storage::Store::scratch), which
`pfilter --no-store` uses, lays a temporary root out by artifact kind and
deletes it when the `Store` is dropped.

### Example: the store after two pfilter runs

To see the layout whole, we run `pfilter` twice on the boarding-school model of
the first-fit guide, the second time asking for one more diagnostic. These two
commands, run from a scratch directory holding a copy of
`docs/guide/first-fit/sir-r0.camdl`, `truth.toml` and `data/in_bed.tsv`, with
`CAMDL_SKIP_VERSION_CHECK=1` and a release binary whose version is
`0.1.0+ca899034b`, produced the tree below.

```bash
camdl pfilter sir-r0.camdl --params truth.toml --data in_bed.tsv --particles 500 \
    --seed 1 --save-paths paths.tsv --store store --label first
camdl pfilter sir-r0.camdl --params truth.toml --data in_bed.tsv --particles 500 \
    --seed 1 --save-paths paths.tsv --trace trace.tsv --store store --label again
```

```text
store/
  store.json
  .invocations/1.json  .invocations/2.json      one record per command run
  .index/<host>-<pid>-<start>.log  (two)        one index log per process
  .owners/  .staging/                           beat files, atomic claims: empty once both ended
  0.1.0_ca899034b-380dcb726337/                 the build directory: version, 12 hex of the ambient key
    compiled_model/
      sir_r0-cb8e23e2e2a4/                      a content artifact, labelled by the model's stem
        identity.json  run.json  model.ir.json  source.camdl  docs.json  declared.json
    data_table/
      in_bed-2e05b33118fc/                      a content artifact, labelled by the stream
        identity.json  run.json  table.tsv
    pfilter/
      sir_r0-b26413d1ed10/                      group: the model (key of the Dep)
        pf-n500-r1-dt0.1-450de25ca20d/          group: PfilterConfig
          params-575611dd7794/                  group: ResolvedParams
            seed_1-f395dfde1a35/                the filter run; 12 hex of its run id
              identity.json  run.json  loglik.toml  provenance.json  .staging/
              saved_paths/paths-d453fc256b8f/   by-product, written with the run
                identity.json  run.json  paths.tsv
              trace/trace-083e32f9b862/         by-product, added by the second command
                identity.json  run.json  trace.tsv
```

The second command reused the filter run and its saved paths, and computed only
the trace, by running the filter again in reproducing mode
([By-products](#by-products)). Its invocation record lists five results in call
order: the compiled model, the data table and the filter run as `reused`, the
trace as `computed` and the saved paths as `reused`.

The test `a_run_makes_the_tree_of_the_proposal` (`pfilter_store/tests.rs:617`)
asserts this shape for a run with `--save-paths`, with each 12-hex suffix
replaced by `*`; `sim_store/tests.rs:716` does the same for `simulate`.

### Finding a result's folder

The store often has a run id and needs the folder that holds it: to reuse a
result, to follow a `Dep`, or to answer `camdl show <id prefix>`. It cannot
compute the path from the run id. A folder's name is built from the labels of
the first request that stored the result (the model file's stem, the seed as
typed), and a later request for the same run id may carry other labels. So the
store looks the folder up.

It tries three sources, cheapest first (`storage.rs:649`, `locate`; `:690`,
`find_any`):

1. **The index.** `.index/` holds one append-only log per process; each
   completed result adds one line, such as
   `{"event":"completed","format":1,"kind":"trace","run_id":"083e…","folder":"0.1.0_…/trace/trace-083e32f9b862","invocation":2,"deps":[],"lineage":[],"upstream":"f395…"}`.
   Readers merge the logs. The index is a cache and can be rebuilt from the
   result records (`camdl reindex`).
2. **The layout, when the identity is at hand**, as it is in `begin`. Each group
   directory ends in the first 12 hex digits of the identity part it groups by,
   and the store can compute those from the identity. From the artifact kind's
   directory in this build's directory (or the parent's subdirectory), it
   follows at each group every directory whose name ends in that group's 12 hex
   digits, then takes every folder ending in the run id's 12 hex digits or all
   64.
3. **A walk**, when only a run id is at hand and the index does not name a
   folder that holds it: every result tree under the root, every build included.
   A walk writes nothing, so a command that only reads leaves `.index/` as it
   was; only `camdl reindex` rewrites the index.

A handle that is a full run id goes to the index first and then, if the folder
it names does not confirm it, to a walk. A handle that is a prefix of a run id
can be matched only against every stored id, so it walks.

Whatever the source, each candidate folder is confirmed against the full run id
in its `identity.json`; a folder whose id differs is another result and is
passed over. Two group keys that share 12 hex digits therefore share a directory
without harm.

When writing, a writer creating a group directory uses the first existing one
with the right suffix, else creates one. A result folder whose 12-hex name is
taken by another run id gets the whole 64-hex run id as its suffix
(`storage.rs:1346`, `place_staged`).

### Handles

A command-line argument that names a stored result, such as the target of
`camdl show` or `--fit @F`, parses as a `Handle` (`handle.rs:19`). Which form an
argument takes is decided from its spelling alone:

| Form                                                         | Parsed as              | Resolves to                                                                            |
| ------------------------------------------------------------ | ---------------------- | -------------------------------------------------------------------------------------- |
| `@L`, `@L~n`                                                 | `Label { name, back }` | the latest invocation labelled `L` whose results all completed, or the `n`th before it |
| up to 64 lowercase hex digits that name no file or directory | `IdPrefix`             | the one result whose run id starts with it; several matches are refused, listed        |
| anything else                                                | `Path`                 | the result whose `identity.json` is in that folder                                     |

`Store::resolve::<A>(&RunRef<A>)` returns the completed result and refuses one
of another artifact kind
([`Error::WrongKind`](rust:store::error::Error::WrongKind)). A label resolves
through its invocation: when the invocation lists exactly one distinct result of
artifact kind `A`, that is the result; when it lists several, it is refused with
[`Error::NotOneResult`](rust:store::error::Error::NotOneResult); when it lists
none, with [`Error::WrongKind`](rust:store::error::Error::WrongKind) naming the
artifact kinds it does list. A group directory has no record, so a path to one
is refused with [`Error::NoRecord`](rust:store::error::Error::NoRecord).

`Store::listings(&Handle)` (`listings.rs`) is the one function that decides
which invocation answers for what a handle names. It returns a `Listings`: one
`Listed` per invocation that listed it, newest first, each with its sequence
number, its `--label`, whether it is `unfinished`, and the attributes it gave. A
label names one invocation, which answers alone. A run id, a prefix or a folder
names every closed listing of the result that ended computed or reused, and the
newest answers; a failed or busy listing is passed over. When none counts, the
newest listing in a running record answers, marked `unfinished`; running records
are read from the directory under the lock. `answering()`, `labelled()` (the
newest listing with a `--label`) and `first()` (the oldest) are what camdl's
readers ask: the model's name of a fit problem or a prediction, the label
`camdl list` and `fit table` show, the attributes of a manifest entry, and the
first name a compiled model was stored under.

`Store::invocation(&Handle)` returns the answering invocation itself, which a
rendering needs: `camdl cat @L` on a `--sweep-file` design renders the whole
design table from it. An invocation with a failed or busy member, or one still
running, never holds its label.

## The lifecycle of a run

Every command on the store, whatever it computes, follows the same sequence: it
stores the inputs its result depends on, asks the store whether the result
already exists, computes it only if not, and writes what the user asked for from
the stored copy. The store's part in each step is what keeps the guarantees:
reuse is decided by the identity alone, a result is written by one process, and
nothing the user receives bypasses the stored files.

In `pfilter`, and in the other commands, the steps are:

1. Open the store ([`camdl_store::open`](rust:cli::camdl_store::open), or
   `scratch` under `--no-store`) and build one `RunInfo` from the split inputs.
2. `put` each content artifact the result depends on (the compiled model, its
   quantities, each data table) and take a `Dep` on it.
3. Build the identity and labels, and `begin` it (or `begin_with` its wanted
   by-products, or `declare` a batch and begin its members through it).
4. On `Claimed`, compute under the claim, write files through the `Run`, and end
   with `finish` or `fail`. On `Reused`, take the stored result. On `Busy`,
   report the owner.
5. Write each export from the stored result, never from memory.
6. End the invocation with `RunInfo::finish`.

Every result except a content artifact is written under a claim, which `begin`
takes and `finish` or `fail` ends. The subsections below follow a run through
these steps.

### Beginning a run

To begin a run, a command hands the store the identity, the labels and, if it
wants any, the by-products it is asking for. The store looks up the run id and
answers in one of three ways: the result is already stored and is reused;
another owner holds it; or this process now holds a claim on it and must compute
it. The answer is a `Begin`:

```rust
// rust/crates/store/src/storage.rs:61
pub enum Begin<A: Artifact> {
    Reused(Stored<A>),   // a completed result exists, with every by-product wanted
    Busy(Owner),         // another owner holds it and is, or may be, alive; or this process does
    Claimed(Run<A>),     // this process now owns the run id
}

// :280
pub fn begin_with<A: Primary>(&self, identity: &A::Identity, labels: &A::Attributes,
    wanted: &Wanted<A>, info: &RunInfo, reuse: Reuse) -> Result<Begin<A>>;
```

`begin` is `begin_with` with no by-products, and
[`Store::run`](rust:store::storage::Store::run) wraps the match for a command
that needs nothing more. `declare` registers a batch's members in the invocation
record as planned, in the order the command asks for them, writes the running
record with all of them, and claims none; it returns a
[`Declared`](rust:store::storage::Declared) handle. `Declared::begin` claims one
member, for a command that computes its members one at a time; `begin_all`
claims every member not yet begun, in run-id order so that two overlapping
batches cannot deadlock. A repeated identity is listed once per time it is asked
for and claimed once. `plan` and `plan_with` report what is stored for each
identity without claiming or registering anything.

`Reuse::Force` recomputes a completed result after moving it to `.quarantine/`,
and takes a claim whose owner is on another host and is stale or of unknown
liveness, moving it to quarantine first. It never takes a claim from an owner
known to be alive. camdl passes it under `--force`: `StoreRequest::reuse`
returns `Force` when the request's `force` is set (`invocation.rs:59`), and
[`camdl_store::store_request`](rust:cli::camdl_store::store_request) builds the
request from the command's `--force` (`camdl_store.rs:97`).

#### What `begin` does with each state

`begin_inner` (`storage.rs:779`) reads what the store holds for the run id and
acts on it, according to the artifact kind's write mode
([Claims, owners and liveness](#claims-owners-and-liveness)):

| What the store holds                                  | `Atomic`                                              | `InPlace`                         |
| ----------------------------------------------------- | ----------------------------------------------------- | --------------------------------- |
| nothing                                               | claim                                                 | claim                             |
| a claim by a live owner                               | wait for it, then look again                          | `Busy`                            |
| a claim by this process                               | `Busy`                                                | `Busy`                            |
| a claim whose owner is dead on this host              | remove it, then claim                                 | reclaim the folder                |
| a claim by an owner on another host, stale or unknown | `Busy` (moved to quarantine under `Force`)            | the same                          |
| a failed run                                          | cannot occur: no folder is left                       | reclaim the folder                |
| a completed result                                    | `Reused`, or reproducing mode for missing by-products | the same                          |
| a damaged result                                      | move to quarantine, then claim                        | the same, under the folder's lock |

To reclaim an in-place folder, the store takes its lock, moves every file at its
root except `identity.json` and the lock into `.attempts/<n>/`, keeping
`run.json` there as `attempt.json`, removes its subdirectories, keeps the last
three attempts (`MAX_PRESERVED_ATTEMPTS`), and writes fresh records
(`storage.rs:1200`, `reclaim`; `:1540`, `preserve_attempt`).

### Claims, owners and liveness

A claim is how the store ensures that one run id is computed by one process at a
time. Every claim names its owner, and the store must be able to tell whether
that owner is still running, so that a claim left by a crashed process does not
block the result forever.

An owner is the process holding claims, named by host, process id and the
process's start time (`owner.rs`). At its first call to a store root it starts a
beat: a file under `.owners/` whose counter it raises every two seconds
(`BEAT_INTERVAL`), removed when the process's last `Store` and `RunInfo` are
dropped.

On the owner's own host, an owner is alive when a process with its id exists and
has its start time, so a reused process id does not read as the owner. From
another host, an owner is alive while its beat counter changes; one whose
counter has not changed for six seconds (`STALE_AFTER`) is stale. No timestamp
is compared across hosts. A dead owner's claim on this host is reclaimed by the
next `begin`; a stale owner on another host is reported as `Busy`, because it
may still be running there, unless the request asks for `Force`.

The two write modes take different claims:

- **Atomic.** The claim is `<root>/.staging/<64 hex of the run id>/`. The store
  builds it whole in a temporary folder holding both records, then renames it
  into place under the lock `.staging/<64 hex>.lock`; the rename fails if
  another claimant's folder is there. After claiming it looks once more for a
  completed result that arrived meanwhile. The run writes its files in the
  staging folder, and `finish` renames the folder into its final place.
- **InPlace.** The claim is the final folder itself: the store creates the group
  directories and the folder, creates `.lock` exclusively, and then writes
  `identity.json` and `run.json` with status running. The run writes its files
  in the final folder, where readers can watch them grow.

Atomic suits quick results whose partial state is of no use; in-place suits long
runs whose progress someone wants to see. camdl writes `pfilter`, `fit_run` and
`profile_point` in place, and every other artifact kind atomically.

### Writing files

A claimed run writes its output through the `Run`, which hands out files by name
and never its folder. Because every byte reaches disk through a file the run
opened, the store knows each file's role and digest when the run finishes, and
no computation can leave a file in the folder that the manifest does not list.

```rust
// rust/crates/store/src/run.rs
impl<A: Artifact> Run<A> {
    pub fn expect(&mut self, files: ExpectedFiles);                   // the files finish requires
    pub fn result_file(&self, name: &str) -> Result<FileHandle>;      // :296  in the manifest, compared
    pub fn provenance_file(&self, name: &str) -> Result<FileHandle>;  // :301  in the manifest, not compared
    pub fn attempt_file(&self, name: &str) -> Result<FileHandle>;     // :306  outside the manifest
    pub fn folder(&self) -> Option<&Path>;                            // :314  an in-place run's folder, to show
    pub fn reproducing(&self) -> bool;                                // :321
    pub fn also<B: ByProduct<Of = A>>(&mut self) -> Result<Option<ByProductRun<B>>>;
    pub fn progress(&self, snapshot: &impl Serialize) -> Result<()>;  // progress.json, replaced
    pub fn event(&self, item: &impl Serialize) -> Result<()>;         // events.jsonl, appended
    pub fn open<U: Artifact>(&mut self, dep: &Dep<U>) -> Result<Cursor<Vec<u8>>>;
    pub fn open_in<U: Artifact>(&mut self, dep: &Dep<U>, file: &str) -> Result<Cursor<Vec<u8>>>;
    pub fn load<C: ContentArtifact>(&mut self, dep: &Dep<C>) -> Result<Arc<C::Content>>;
    pub fn finish(self) -> Result<Stored<A>>;
    pub fn fail(self, operation: &str, reason: &str) -> Result<()>;
    pub fn fail_by_products(&mut self, operation: &str, reason: &str);
}
```

A name is a relative path and may contain subdirectories. `check_name`
(`run.rs:144`) refuses an empty or absolute path, a component starting with `.`,
`..`, a name containing `:`, `,` or `\`, the name `*`, the store's own file
names at the folder's root, and a first component equal to a registered `KIND`,
since by-products are placed there. A name is opened once
([`Error::FileOpenedTwice`](rust:store::error::Error::FileOpenedTwice)). A
`FileHandle` can be cloned and sent to other threads, and each clone appends to
the same file while the handle hashes the bytes. `finish` refuses while a
buffered writer from `FileHandle::writer` is alive
([`Error::WriterOpen`](rust:store::error::Error::WriterOpen)), and a write after
`finish` or `fail` is refused
([`Error::WriteAfterEnd`](rust:store::error::Error::WriteAfterEnd)).

A run declares the files it will hold with `Run::expect`, by name and role
(`ExpectedFiles`: its result and its provenance files; attempt files are not in
it), and a by-product with `ByProductRun::expect`. The last declaration holds.
camdl declares each kind's set from that kind's file enum, when it begins the
run (`camdl_store::begin`) and again from the outcome before it finishes
(`result_files::declare`); `Store::put` declares a content artifact's set
itself.

`pfilter` writes its run and every by-product it claimed in one function:

```rust
// rust/crates/cli/src/pfilter.rs:1782
fn write_run(run: &mut store::Run<Pfilter>, f: &RunFiles) -> store::Result<()> {
    declare(run, PfilterFile::of(f.replicates.is_some()));
    create(run, &PfilterFile::Loglik)?.write_all(&f.loglik.to_bytes())?;
    if let Some(b) = &f.replicates {
        create(run, &PfilterFile::Replicates)?.write_all(b)?;
    }
    create(run, &PfilterFile::Provenance)?.write_all(&f.parameters)?;
    // for each by-product B the run was asked for, declared with its files:
    //     if let Some(bp) = camdl_store::also::<B>(run, files)? { create_by_product(&bp, &file)?.write_all(bytes)?; }
    // ...
}
```

#### Finishing a run

`finish` (`run.rs:545`, `finish_inner`) closes every handle and builds the
manifest. It refuses a run, or a by-product of it, that declared no file set
([`Error::NoExpectedFiles`](rust:store::error::Error::NoExpectedFiles)), and one
whose result and provenance files differ from its declaration
([`Error::FileSetMismatch`](rust:store::error::Error::FileSetMismatch), naming
the missing and the undeclared files); a reproduced upstream is compared with
its stored copy first. It checks that each digest the identity holds as `Hashed`
content has a result file with that digest
([`Error::ContentMissing`](rust:store::error::Error::ContentMissing)), and
writes `run.json` completed. For an atomic result it then renames the staging
folder into place; for an in-place one it replaces `run.json` through
`run.json.tmp` and a rename and removes the lock. Each completed result appends
one line to the index. If `finish` itself fails, the run is failed.

An atomic `finish` can find a completed result of its own run id already in
place, put there by a writer whose claim it did not see (one moved aside under
`Force`). It then compares the result files by digest: on a match the stored
copy is the result, and on a difference `finish` returns
[`Error::Divergent`](rust:store::error::Error::Divergent) and keeps its own
files in `.quarantine/` (`run.rs:664`, `met_stored_copy`).

### By-products

A by-product is a second artifact of the same computation, such as the sample
paths a filter run draws. Asking for one must not change the main result's run
id: `pfilter --trace` filters the same way as `pfilter` without it, and should
find the same stored run. So a by-product's identity holds its upstream's
identity record plus its own settings:

```rust
// rust/crates/cli/src/pfilter_store.rs:228
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "saved_paths", version = 1)]
pub(crate) struct SavedPathsIdentity {
    pub filter: PfilterIdentityRecord,
    pub paths: u64,                       // --n-paths
}
```

A command lists the by-products it wants in a `Wanted<A>` and passes it to
`begin_with`; `pfilter` builds one from `--trace`, `--save-paths` and the rest
(`pfilter.rs:1205`). The running computation takes each one's files with
`Run::also::<B>()`, which returns `None` for a by-product that was not wanted or
is already stored.

A by-product has no `finish` of its own: the upstream's `finish` commits the
upstream and then each by-product, so a completed by-product never exists
without its upstream. The library places each one at
`<upstream folder>/<B::KIND>/<B::label>-<12 hex>/`.

#### Adding a by-product to a stored result

If the upstream is already stored and a wanted by-product is not, the by-product
can only be made by running the computation again, as the second command of the
[example](#example-the-store-after-two-pfilter-runs) did for the trace. The
store uses that rerun as a check. `begin_with` returns `Claimed` in reproducing
mode, and `Run::reproducing()` is true. Only the missing by-products are
claimed, under `<upstream folder>/.staging/`; the stored upstream is not locked,
moved or written.

The command computes the upstream again, and every file it writes goes to
`.reproduce/` inside the first by-product's staging folder. At `finish` each
result file written there is compared by digest with the stored upstream's file
of the same name; a stored result file that was not written again, or a result
file the stored upstream lacks, counts as a difference. On a match the
by-products are committed and `finish` returns the stored upstream.

On a difference `finish` returns
[`Error::Divergent`](rust:store::error::Error::Divergent), moves `.reproduce/`
to `.quarantine/`, and fails the by-products. A divergence means the upstream's
result depends on something outside its identity, and the quarantined files are
the evidence.

### Failure

A run can end without a result in several ways, and the store records each so
that the next request knows what to do.

`Run::fail(operation, reason)` ends a run as failed. A `Run` dropped without
`finish` or `fail` (an early return, an unwinding panic) fails with no reason.
For an in-place run the folder stays with `run.json` saying `failed` and the
reason, and the next `begin` reclaims it. For an atomic run the staging folder
is removed and the failure is kept only in the invocation record. A process that
is killed writes nothing; its claim says running, and readers derive interrupted
from its dead owner. Its by-products are failed with it.

#### Which outcomes are failures

Not every error is a failed run. Which outcomes are failures is the command's
decision, and the rule is the same in each command that makes the distinction: a
failure the run's identity determines is a completed result, and any other is a
failed run. A failure the identity determines would happen again on every rerun,
so storing it saves the rerun; any other failure might not recur, so the next
request must try again.

In `pfilter`, every particle's weight reaching zero, or an error the filter
stops with at these parameters, is determined by the model, the parameters, the
data, the particles and the seed. Such a run is stored as completed, with a
`loglik.toml` that has no `loglik` and records `n_failed` and the first
failure's message, and a rerun reuses it.

The compute budget depends on `--pf-max-substeps`, an execution input, so a run
it stops is not a function of its identity. `refuses_the_run`
(`pfilter.rs:1403`) names that one error, and the run is failed:

```rust
// rust/crates/cli/src/pfilter.rs:1032-1037
match run_pooled(&pf_pool, || bootstrap_filter(&process, &obs_model, &params, &smc_config, seed)) {
    Ok(result) => Ok(Filtered::Single(Box::new(result))),
    Err(e) if refuses_the_run(&e) => Err(filter_failed(/* ... */)),   // → run.fail("filtering", …)
    Err(e) => Ok(Filtered::Failed(format!("pfilter error: {}", failure_text(&e)))), // stored
}
```

An I/O error, a store error and another process holding the run are failures
too, and nothing is stored for them.

When the filter stopped with an error, it left no state to write by-products
from, so `pfilter` calls `run.fail_by_products` before `finish`
(`pfilter.rs:1276`): each wanted by-product is recorded as failed in the
invocation record, none is committed, and the run itself is stored.

### Sealed records

After `finish`, a result's files and records are not written again, so a reader
who has checked a result once can rely on it staying as it was. Anything added
later goes elsewhere: a by-product, an attempt file written before `finish`, or
an invocation record. A second request for a stored result is recorded in its
own invocation record, and a name given later is a later invocation's `--label`.

## Reading and rendering

Once a result is stored, everything a user receives from it is made from the
stored files: a file read back by later code, a copy at a path the user gave, or
a table rendered from several results. This section describes those reads and
the record of each command run that ties them together.

### Reading stored files from code

`Stored<A>` (`stored.rs:23`) is a completed, undamaged result, and the only type
that opens result files or makes a `Dep`. `Store::get(&Id<A>)` returns one, as
do `finish`, `Begin::Reused` and `resolve`.

```rust
impl<A: Artifact> Stored<A> {
    pub fn id(&self) -> &Id<A>;
    pub fn identity(&self) -> &<A::Identity as Identity>::Record;
    pub fn folder(&self) -> &Path;                       // for display only
    pub fn result_files(&self) -> Vec<&str>;
    pub fn open(&self, file: &str) -> Result<Cursor<Vec<u8>>>;             // digest checked
    pub fn provenance_files(&self) -> Vec<&str>;
    pub fn open_provenance(&self, file: &str) -> Result<Cursor<Vec<u8>>>;  // digest checked
    pub fn export(&self, file: &str, to: &ExportPath) -> Result<()>;
    pub fn dep(&self, file: &str) -> Result<Dep<A>>;
    pub fn dep_all(&self) -> Dep<A>;
    pub fn dep_files(&self, files: &[&str]) -> Result<Dep<A>>;
    pub fn dep_part(&self, file: &str, part: &str, extract: impl Fn(&[u8]) -> Result<Vec<u8>>) -> Result<Dep<A>>;
    pub fn recorded(&self, name: &str) -> Option<&serde_json::Value>;
    // given(), started(), deps(), lineage(): the matching fields of run.json
}
```

`open` reads the whole file and checks it against the manifest's digest before
returning any byte, so a file changed behind the store is refused rather than
read, with [`Error::Damaged`](rust:store::error::Error::Damaged) naming the
result (`run.rs:444`, `read_checked`).
[`camdl_store::read_stored`](rust:cli::camdl_store::read_stored)
(`camdl_store.rs:139`) is camdl's helper for reading one file this way.

### Exports and renderings

An export is a file written to a path the user gave, such as
`pfilter -o
table.tsv`. It is made from stored files after `finish` or `get`,
never by running the computation again or from the computation's memory, so what
the user receives is always what the store holds. It is either a byte copy of a
stored result file (`Stored::export`) or a rendering: a pure function of an
invocation's stored results, their labels and options.

```rust
// rust/crates/store/src/invocation.rs:564
pub trait Rendering {
    type Of: Artifact;
    type Options;
    fn render(members: &[(Stored<Self::Of>, <Self::Of as Artifact>::Attributes)],
              options: &Self::Options, out: &mut dyn Write) -> Result<()>;
}

// :502
pub trait ByProductRendering {
    type Of: Primary;
    type By: ByProduct<Of = Self::Of>;
    type Options;
    fn render(members: &[MemberWith<Self::Of, Self::By>], options: &Self::Options,
              out: &mut dyn ExportSink) -> Result<()>;
}
```

`Invocation::render` writes a rendering to a path and `render_to` to any writer,
so `-o` and the terminal print the same bytes. Both refuse an invocation with a
member that is not completed
([`Error::MemberNotCompleted`](rust:store::error::Error::MemberNotCompleted)). A
`ByProductRendering` reads each member together with the by-product of one
artifact kind the invocation asked for of it, and writes one file or a directory
of files through an `ExportSink`; `simulate --obs-dir` and `--quantities-out`
are renderings of this kind.

`pfilter` has two renderings: `LoglikTable` (`pfilter_store.rs:412`), the
log-likelihood to four decimals or one row per replicate, and `DesignTable`
(`:452`), one row per design point in file order, which `-o` writes after a
`--sweep-file` run. The by-product flags (`--trace`, `--save-paths` and the
rest) export by reading the by-product the invocation lists through the store
and writing its bytes to the path (`pfilter.rs:1828`, `by_product_bytes`).

### The invocation record

A stored result says what produced it, but not who asked for it or when. That is
the invocation record's job. Every command run that calls the store writes one,
`.invocations/<seq>.json`, whether its results were computed or reused. It is
what a `--label` names, and what a rendering of several results reads.

The record is owned by the `RunInfo` the command builds once and passes to every
`begin`, `begin_with`, `declare` and `put`:

```rust
// rust/crates/cli/src/pfilter.rs:544-554
let info = store::RunInfo::new(&given, &execution_record, &request)?;
info.lineage(store::Lineage::Input { field: format!("data.{name}"), digest: table.digest() });
```

At the command's first call to the store, the store takes the lock
`.invocations/.lock`, refuses the `--label` if an invocation carrying it is
still running and its owner is not known to have stopped
([`Error::LabelInUse`](rust:store::error::Error::LabelInUse)), takes the next
sequence number, and writes the record's first version with status `running`
(`start`). A `declare` rewrites the running record once, so that it lists every
member the batch declared. When the command ends, `RunInfo::finish` (or dropping
the `RunInfo`) replaces it with the last version, status `ended`, and the record
is not written again. A record whose owner died while it was running is closed
by `camdl clean` with status `interrupted`, its unfinished results recorded as
failed (`maintenance.rs:198`).

The record holds the command line, the inputs as given, the digest of each
configuration file read, the `--label`, and every result the command passed to
the store in call order, each with its artifact kind, run id, `attributes` and
outcome: `computed`, `reused`, `failed` with its reason, or `busy`. A result
still `planned` when the command ends is recorded as failed. Its states are
types (`record::Record`): a `Running` record's results may be `planned`, and a
`Closed` one (`ended` or `interrupted`) holds an ended outcome for every result,
so a closed record that lists one as `planned` does not read. A command's live
progress is the separate file `.invocations/<seq>.progress.json`, removed when
the command ends.

The record has its own format, `INVOCATION_FORMAT` (2); `store.json` and every
result's records are `RECORD_FORMAT` (1). A record of another format, or one
that does not read, is left out of every lookup and listed by
`Store::unread_records`; the store calls the reporter the project sets
(`report_unread_records_to`) once, the first time a read meets one, and camdl
prints one warning naming each such record. The results such a record lists stay
reusable, since reuse goes by identity, but its label names nothing.

The record is durable. The index lists completed results only, and is rebuilt
from their records. `Invocation::members::<A>()` returns the entries of one
artifact kind with their attributes, which is how a rendering and `camdl list`
read them.

A store reads `.invocations/` for its lookups once. `Store::invocations()`
returns an `Arc<[Invocation]>` snapshot: the closed records (`ended` or
`interrupted`) read at the store's first lookup, and each record this store
writes for its own invocations, added as it writes it. `Store::listings`,
`Store::invocation`, `Store::label_holders`, `Store::runs_of_configuration` and
every camdl reader of the records answer from it, so `fit summary @f` reads the
directory once however many lookups it makes. A clone of a `Store` is the same
opening and shares the snapshot; a command opens its store once and passes it
down (`camdl list` builds every kind's rows from one opening, and `predict`
stores in the opening its fit was read through). A record another process writes
after the snapshot is not in it, nor is another process's running record. Two
reads must see those, and read the directory each time, under the lock: the
label check when a command starts, and the previous holder of a label when it
ends. `camdl status` and `camdl clean` read the running records from the
directory too.

## Content artifacts and dependencies

Most stored results are computed: their run id is a hash of the inputs that
produce them. Some stored things are not computed at all. A compiled model is
the bytes the compiler emitted, and a data table is the bytes of a file the user
gave. For these, the natural identity is the content itself. A content artifact
is a stored file keyed by its own bytes: its identity is the digest of its one
result file.

Commands share content artifacts. When `simulate`, `pfilter` and `fit run` run
the same model, or `pfilter` runs it at a hundred parameter points, they all
depend on one stored compiled model, stored once. Each result names it with a
`Dep`, so the result records exactly which model bytes it ran, and the model
itself appears in the store once rather than inside every result.

### The content-artifact traits

```rust
// rust/crates/store/src/artifact.rs
pub trait ContentFiles: Primary {
    /// Each result file's name and bytes; their digests are exactly the
    /// identity's held_content().
    fn files(identity: &Self::Identity) -> Vec<(String, Vec<u8>)>;
}

pub trait ContentArtifact: Primary<Identity = Hashed<Self::Content>> {
    type Content: Content + Send + Sync + 'static;
    const FILE: &'static str;           // "model.ir.json", "quantities.json", "table.tsv"
}
// Every ContentArtifact is ContentFiles, with the one file FILE.
```

`Content` is a typed value with a byte form (`to_bytes`, `from_bytes`), and
`Hashed<T>` holds a value with the SHA-256 of its bytes, computed in its only
constructor. A content artifact of one file implements `ContentArtifact`, and
its identity is that file's `Hashed<T>`; a result that depends on it reads the
typed value with `run.load(&dep)`. A content artifact of several files
implements `ContentFiles` directly, with an identity that holds a `Hashed<T>`
for each file. camdl has three content artifacts of one file (`camdl_store.rs`):

- `CompiledModel`, whose file is the part of the compiled model the engine runs;
- `Quantities`, whose file is the model's `quantities` and `contrasts` blocks;
- `DataTable`, whose file is a data file's bytes as read.

### What keys the compiled model

The compiled model's key is the digest of exactly what a result consumes, so
that a change the engine cannot see does not re-key every result.
[`CompiledParts::split`](rust:cli::camdl_store::CompiledParts::split)
(`camdl_store.rs:391`) destructures every field of the compiled envelope and the
model, so a field added to the IR does not compile until it is assigned to a
part.

The model part is every field of [`ir::Model`](rust:ir::model::Model) except its
name, description, quantities and contrasts, written as canonical JSON with
object keys sorted, except that the initial conditions are an array in the
model's declaration order, which the runtime evaluates by. The name becomes a
label, and the quantities and contrasts are the `Quantities` content.

A model's docstring, file name or quantities block therefore changes no
simulation's or filter run's key. The parameters and scenario are applied after
the model is keyed, so runs at other parameter values share one compiled model
(`the_parameters_and_the_scenario_do_not_key_the_compiled_model`). The results
that evaluate quantities (`sim_quantities`, `prediction`) hold a
`Dep<Quantities>` of their own.

### A model's source

The model file, its name, the description, the documentation, the declared model
and, for a `.camdl`, the model's mathematics and flow graph as `camdlc render`
writes them, are the model source
([`camdl_store::model_source`](rust:cli::camdl_store::model_source)): a content
artifact of several files, `model_source`, keyed by their bytes, the file's name
and a `Dep<Quantities>` on the quantities its compile produced. No simulation or
fit depends on it, so a comment edit re-keys none of them. `put_compiled` stores
the quantities, the source and the compiled model, and records the source on the
compiled model's listing with `RunInfo::record_source`: the listing's `source`
is `{"kind": "model_source", "run_id": ...}`, and its attributes hold only the
model's name. Two files that differ only in their comments therefore share one
compiled model, and each invocation records its own source.

A command that stores a result read from a stored one compiles nothing: a
prediction or a score from a stored fit run. It lists the stored compiled model
in its own invocation record, as reused, with the source the invocation it read
from recorded, or the source of the file `--model` gave (`link_source`, through
[`Store::list`](rust:store::storage::Store::list), which registers a stored
result in the current invocation and stores nothing, and `record_source`).

A reader finds the source by one rule,
[`shown_for`](rust:cli::camdl_store::model_source::shown_for): the source the
invocation that answers for its handle (`Store::listings`) recorded for the
result's compiled model, which `model_of` finds among the result's recorded
dependencies (`Listings::source`, a `Recorded`: `None`, `One` or `Several`).
`fit summary` and the re-posed prediction of `camdl score` take it with the
stored fit; `camdl path` and `camdl cat` take it for a name qualified by
`model_source`. No other invocation stands in. The return value, `Shown`, says
which of three things holds:

- `Recorded`: the source is read. For a run id or a folder it also holds the
  different sources earlier invocations that listed the result recorded, which
  the reader names in an info notice with the labels of their invocations
  (`model-source-one-of-several`).
- `NotRecorded`: the answering invocation records none. The reader goes on
  without the documentation and says so in a warning
  (`model-source-not-recorded`).
- `Unreadable`: the invocation names a source that cannot be read. The reader
  goes on without the documentation and says so in a warning
  (`model-source-unreadable`); nothing is computed from it.

### Storing a content artifact

[`Store::put`](rust:store::storage::Store::put) stores a content artifact
without a claim, because two writers of one content write identical bytes. It
refuses a file whose digest the identity does not hold (`Error::ContentUnheld`)
and an identity holding a digest that no file has (`Error::ContentMissing`), so
the run id is the digest of exactly the stored files. It writes the folder whole
in a temporary staging folder, renames it into place if the destination is
absent, and otherwise removes its own copy and returns the stored one; a damaged
copy in place is moved to quarantine first (`storage.rs`, `put_inner`). It
returns a `Stored<C>`, from which the command takes a dependency:

```rust
// rust/crates/cli/src/pfilter.rs:568-571 (error handling trimmed)
let model = store
    .put::<CompiledModel>(model_part, &model_labels, &provenance, &info)?
    .dep(CompiledModel::FILE)?;
```

### Dependencies

A `Dep<A>` (`stored.rs:236`) names one file, several files, or a part of one
file of another result, which may be a content artifact or any other stored
result. It hashes, as a struct with tag `dep` and version 1, three values: the
digest of what is consumed, the file name (`*` for every result file, names
joined by `,`, `<file>:<part>` for a part), and the upstream's run id. It
records the upstream's artifact kind and build, and for a part the whole file's
digest, without hashing them.

A `Stored` result makes each form: `Stored::dep(file)` names one file,
`dep_all()` every result file, `dep_files(names)` several, and
`dep_part(file, part, extract)` the extracted part of one file.

A running computation reads a dependency only through one of three methods, and
each read is listed under `reads` in `run.json`:

- [`Run::open`](rust:store::run::Run::open), one file or a part, checked against
  the digest;
- [`Run::open_in`](rust:store::run::Run::open_in), one file of a several-file
  dependency, after the dependency's combined digest is checked;
- [`Run::load`](rust:store::run::Run::load), the parsed content of a content
  artifact, cached per process by digest.

A value that was used and is not hashed is recorded with `RunInfo::lineage`:
`pfilter` records each data file's digest that way, because the compiled model
was rewritten from those rows.

### Checking the model that runs

Every command that runs a model checks that the model the engine runs is the
stored compiled model its identity names. `simulate` and `pfilter` load the
stored model, apply the identity's parameters, interventions and window, and
compare the model part's digest with that of the model the engine holds:
`simulate` before it runs the cell (`sim_command.rs:528`), `pfilter` after the
filter has run and before it writes any file (`pfilter.rs:1253-1270`).
`fit run`, `predict` and `profile` call
[`camdl_store::require_stored_model`](rust:cli::camdl_store::require_stored_model)
before they compute (`camdl_store.rs:224`; `fit/runner.rs:288`,
`fit/predict.rs:2296`, `profile.rs:464`). A difference is an internal error, and
the run fails rather than store a result under the identity of another model.

## How camdl's artifact kinds use the store

Each command on the store defines its own artifact kinds. This section takes
them one command at a time: what the command computes, what it stores, how its
results are nested, what it can add as by-products, and how it handles failures.

### `simulate`

`camdl simulate` runs a model forward from given parameter values and produces
trajectories: the compartment counts and flows over time. One command can ask
for many, one per seed (`--seeds 1:20`), per scenario, per point of a sweep or
per posterior draw, and it stores each trajectory as its own result, a cell
(`sim`, `sim_store.rs:211`). Storing cells one by one means a request for seeds
1 to 30 after one for seeds 1 to 20 computes only the ten new cells.

A cell's identity is everything one trajectory depends on: the stored compiled
model; `SimConfig` (the backend, `dt`, the cell's window, a restored
`--init-state` row, the two evaluator policies and the column view); the
parameter values; the active interventions; and the process seed. Its folder
holds `traj.tsv`, `reactive_log.tsv` when a reactive policy was active, and the
provenance file `provenance.json`. Cells are atomic, and `simulate` declares its
cells and claims them all with one `begin_all`.

The cells are nested in that order, so that cells sharing a model, settings,
parameters and scenario sit side by side and differ only in seed:

```text
sim/<model>-…/<backend>-dt<dt>-…/<params>-…/<scenario>-…/seed_<seed>[-r<n>]-…/
```

The observations (`obs`), the event log (`event_log`) and the generated
quantities (`sim_quantities`) are by-products of a cell, so `--obs`,
`--event-log` and `--quantities-out` never change a cell's run id.

No combined table is stored. `-o`, `--stdout` and `camdl cat @L` render
`CombinedTable` (`sim_store.rs:548`) from the cells the invocation lists, and
`--obs-dir` and `--quantities-out` render `ObsTable` and `QuantityTable`.

`simulate --fit @F --draws posterior` takes each draw's parameter values from
the stored fit into the cells' `ResolvedParams`. The cell holds no dependency on
the fit: two fits whose draws give the same values produce the same cells.

When a cell's engine stops with an error, the error is a function of the cell's
seeded identity, so the cell is stored as a completed result holding
`failure.toml` in place of the trajectory, and a rerun reuses it. A by-product
refused for its cell likewise stores its own `failure.toml`. An error from the
store itself fails the cell, which, being atomic, leaves no folder.

### `pfilter`

`camdl pfilter` runs a particle filter at one parameter vector and reports the
estimated log-likelihood ([The running example](#the-running-example-pfilter)).
It stores one result per filter evaluation, a filter run (`pfilter`), holding
`loglik.toml` (with `replicates.tsv` under `--replicates N`) and the provenance
file `provenance.json`. A filter run is written in place, so its progress can be
watched.

`--sweep-file` makes one filter run per design point, and `-o` renders
`LoglikTable` or `DesignTable` from the runs the invocation lists. Filter runs
are nested by the model, the settings and the parameters, then the seed
([Placing a result](#placing-a-result)).

A filter run has seven atomic by-products, one per diagnostic flag: `trace`,
`pf_health`, `filtering`, `filtered_streams`, `final_state`, `saved_paths` and
`prequential`. Each sits inside the filter run's folder.

Under `--fit` naming a stored fit, the run's identity holds a `Dep` on its fit
problem (`PfilterIdentity::fit`). Under a fit file, read in process, it holds
none, and what the file contributed is in the config and parameters.

How a filter that stops with an error is handled is described in
[Which outcomes are failures](#which-outcomes-are-failures): an error the
identity determines is a stored result with no `loglik`, a run stopped by the
substep budget is failed, and a stopped filter's by-products are recorded as
failed.

### `fit run`

`camdl fit run` estimates a model's parameters from data with the method a fit
configuration names. It separates two things: the problem being solved (the
model, the data, which parameters are estimated and their priors) and one
attempt at solving it (a method, its settings and a seed). It stores each as its
own artifact kind (`fit_store.rs`), so that several methods or seeds run on one
problem share it.

A fit problem (`fit`, `FitProblem`, `:243`) is atomic and written once at
`fit/<model>-…/`. Its identity is everything that poses the problem and no
method setting: the compiled model rewritten from the data, each bound file's
training lines, each stream's held-out dates without their values, the partition
rule, the `[data.tables]` files, the estimated set with bounds and priors, the
resolved `[fixed]` values, `simplex_groups`, the settings, the scenario request,
`ic_free` and, for a `[synthetic]` problem, what generated its data.

Its result files are `problem.toml` (written in `fit.toml` syntax from the
identity alone), `partition.json`, `inputs/<source>.<ext>`,
`inputs/tables/<name>.<ext>`, `held_out/<stream>.tsv` and, for a synthetic
problem, `synthetic/truth.toml`; the model source is a provenance file. Every
reader of a stored fit reads it through
[`fit_store::read_fit`](rust:cli::fit_store::read_fit) (`:1294`), which takes
each file through the store.

A fit run (`fit_run`, `FitRun`, `:311`) is in place, nested inside its problem
at `runs/<algorithm>-…/seed_<n>-…/`. Its identity is a `ChainIdentity` (the
problem; the method, a `MethodIdentity` (`fit_store/method.rs`) holding the
algorithm's settings without its length and, for IF2, PMMH and MH, every
estimated parameter's resolved step in z; the compile with the state Jacobian
for `nl-lbfgs` on the ODE backend; the process seed; and the chain starts, a
`StartSource` holding the rule, what it reads and the `[estimate]` `start`
values in `written`), the length, and for a resume a `Dep` on the base run. The
steps are resolved by `fit::steps::resolve_steps` before the identity is built,
so a step written at its default and a step left out key alike.

The method writes each file into the run as it produces it, through
[`fit::output::LiveRun`](rust:cli::fit::output::LiveRun): a name that is one of
a fit run's files is opened in that file's role (`Run::result_file`,
`Run::provenance_file`), so the run's folder holds each chain's `trace.tsv` as
it grows, and any other name is opened as an attempt file. A file the method
rewrites whole as it learns more, such as `chain_starts.tsv` once a chain is
refused, is replaced through `FileHandle::replace`. A method that reads back
what it wrote (a chain's trace, for its diagnostics) reads it from the run's
folder. The progress snapshot and the event log go to the run's own `progress`
and `event` records. When the method ends, the run declares the files its
recorded outcome calls for and is finished: `method.toml` and the method's
outputs are result files, and the execution record, the optimizer trace,
`fit.meta.json` and the fit file's text (`fit.toml`) are provenance. `fit run`
has no by-products.

`fit run` stores every problem first, then builds the identity of every run
(each problem with each seed, a `--resume` base resolved for each), declares
them, and claims each run only when it computes it. While the first seed of a
20-seed fit runs, its invocation record lists all 20 runs as planned, and only
the run being computed has a folder.

When a method stops with an error, the run's folder keeps what the method wrote
and is marked failed with the reason; a process killed while the method runs
leaves its folder interrupted. The next `fit run` claims the folder and computes
the run again. `fit run` does not distinguish a failure the identity determines.

### `predict`

`camdl predict` produces predictive bands or a forecast from parameter values:
either a stored fit run's posterior draws or a single point given with a model
file. It stores each as one prediction (`prediction`, `Prediction`,
`fit_store.rs:490`), which is atomic and has no by-products.

A prediction's identity is the parameter source, the data the filter conditions
on, the quantities it evaluates and `PredictOptions`, the resolved options as
canonical JSON, an option added to `camdl predict` included unless it is named
as left out.

Where a prediction is stored depends on its source. A prediction from a stored
fit run's posterior
([`ParameterSource::Posterior`](rust:cli::fit_store::ParameterSource::Posterior),
a `Dep` on the run's files and its problem's) is nested inside the run at
`predict/<mode>-…/`. A prediction from a fixed point given with a model file
([`ParameterSource::FixedPoint`](rust:cli::fit_store::ParameterSource::FixedPoint):
the compiled model, the parameter values, the numerics and the training cutoff)
is placed at `prediction/<model>-…/<mode>-…/`.

`predict` writes its files into a scratch directory and then stores each as a
result file (`fit/predict.rs:1905-1940`). For every mode but a forecast it first
looks the prediction up with `plan` and `get`, and reuses a completed one
without computing it; a forecast's identity holds its targets and design files,
which are known only after it is computed.

When a prediction's computation stops with an error, the error ends the command
before the prediction is claimed, so nothing is stored, and the next `predict`
computes it again.

### `profile`

`camdl profile` computes a profile likelihood: for each value of a focal
parameter on a grid, it maximizes the likelihood over the other parameters, from
several starting points and seeds. A profile is of a fit problem in the store: a
stored fit's, or the one a fit file poses, with the command line's `--fixed`,
`--table`, scenario, policy and `--dt` flags written in, stored as `fit run`
stores a problem
([`profile_store::pose_problem`](rust:cli::profile_store::pose_problem),
`profile_store.rs:99`).

It stores one result per focal value, seed and start, a profile point
(`profile_point`, `ProfilePoint`, `:511`), written in place and holding
`mle.toml` and the provenance file `provenance.json`. A point's identity is the
problem, the focal values, `ProfileMethod`, the start's values and a seed
derived from the user's seed, the focal values and the start's values
(`point_seed`, `:482`). It holds nothing about the grid it was asked for in, so
a larger grid reuses every point a smaller one stored.

Points are nested by the problem, the focal parameter and the method, then the
seed and start:

```text
profile_point/<model>-…/<focal>-…/<method>-…/seed_<n>-start_<i>-…/
```

`profile` declares its points and claims them all with one `begin_all`, and has
no by-products. The profile table is the rendering `ProfileTableRendering`
(`profile_store.rs:549`) of the points the invocation lists.

A point whose log-likelihood is not finite, including an IF2 cell that hit the
substep budget, is a completed result whose `mle.toml` holds
`final_loglik = -inf`. A point that cannot be computed at all, because a forcing
would be evaluated outside its range or the model fails to compile for the
point, is failed with that reason (`profile.rs:1661`, `:1675`).

## Adding an artifact kind or a command

This section walks through what a new command on the store, or a new artifact
kind for an existing command, needs, step by step, with `pfilter` as the worked
example. Each code block is trimmed to the lines that matter.

### 1. Declare the inputs

Add `#[derive(store::Inputs)]` to the command's clap struct and give every field
a class. The compiler refuses a field without one.

```rust
#[derive(Args, Clone, store::Inputs)]
pub struct PfilterArgs {
    #[arg(long)]
    #[input(identity)]                 // changes the result: in the identity
    pub particles: usize,

    #[arg(long, value_name = "FILE")]
    #[input(expands)]                  // one result per design point
    pub sweep_file: Option<PathBuf>,

    #[arg(long)]
    #[input(adds)]                     // asks for the `trace` by-product
    pub trace: Option<String>,

    #[arg(long)]
    #[input(store)]                    // names the invocation, not the result
    pub label: Option<String>,
    // ...
}

#[derive(Args, Clone, store::Inputs)]
pub struct PfilterCore {
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    #[input(execution, exercised_by = crate::input_fixtures::pfilter_parallel)]
    pub parallel: usize,               // must not change the result; checked on a fixture
    // ...
}
```

Wrap the arguments and the global options the command reads in one invocation
struct, as `PfilterInvocation` (`args/mod.rs:2434`). Give each file and
non-trivial identity field a `vary = ...`, and each execution field an
`exercised_by = ...`, in `input_fixtures.rs`.

### 2. Write the identity

Define the identity with `#[derive(store::Identity)]` and a tag and version,
from typed resolved values: `Finite` for floats, `Dep<_>` for stored inputs,
`ResolvedParams` and `ProcessSeed` from `camdl_store.rs`. Nothing that cannot
change a result goes in it.

```rust
#[derive(store::Identity, Clone, Debug)]
#[identity(tag = "pfilter", version = 1)]
pub(crate) struct PfilterIdentity {
    pub model: Dep<CompiledModel>,     // a stored input, by run id and digest
    pub config: PfilterConfig,         // resolved settings; floats as Finite
    pub params: ResolvedParams,
    pub scenario: ActiveInterventions,
    pub seed: ProcessSeed,
    pub fit: Option<Dep<FitProblem>>,
}
```

### 3. Resolve

Split once with `Inputs::split`, take every part apart with `into_parts`, and
consume the declared part in one resolution function, as
[`pfilter_config::resolve`](rust:cli::pfilter_config::resolve). Release what the
command does not use with `store::release`.

```rust
let (declared, execution, exports, request, given) = store::Inputs::split(inv).into_parts();
let (config, requests, _) = crate::pfilter_config::resolve(declared, compile, &out, &root);
store::release(quantities, "a filter run does not report quantities");
```

### 4. Define the artifact kind

Implement `Artifact` and `Primary`, choose `Atomic` or `InPlace`, and write
`layout` with `camdl_groups`. Define each by-product with an identity holding
the upstream's record, as `run_by_product!` does (`pfilter_store.rs:165`). Add
every artifact kind to `camdl_kinds()`.

```rust
impl Artifact for Pfilter {
    const KIND: &'static str = "pfilter";
    const WRITE: WriteMode = WriteMode::InPlace;   // a long run whose progress is watched
    type Identity = PfilterIdentity;
    type Attributes = PfilterAttributes;
}

impl Primary for Pfilter {
    fn layout(id: &PfilterIdentity, labels: &PfilterAttributes) -> Placed {
        camdl_groups(&labels.model, &id.model, &id.config.label(), &id.config,
                     Some(("params", &id.params)))
            .result(&format!("seed_{}", labels.seed))
    }
}
```

### 5. Store the inputs

Open the store with [`camdl_store::open`](rust:cli::camdl_store::open) (or
`scratch` under `--no-store`), build one `RunInfo`, `put` the compiled model and
each data table, and put their `Dep`s in the identity.

```rust
let info = store::RunInfo::new(&given, &execution_record, &request)?;
let model = store
    .put::<CompiledModel>(model_part, &model_labels, &provenance, &info)?
    .dep(CompiledModel::FILE)?;        // goes into PfilterIdentity::model
```

### 6. Run under a claim

`begin_with` the identity, labels and `Wanted`; check that the model about to
run is the stored compile (`require_stored_model`); write files through the
`Run`; read stored inputs with [`Run::load`](rust:store::run::Run::load),
[`Run::open`](rust:store::run::Run::open) or
[`Run::open_in`](rust:store::run::Run::open_in); end with `finish` or `fail`.
Decide which failures are results and which are failed runs, and write the rule
in the code.

```rust
match store.begin_with::<Pfilter>(&identity, &labels, &wanted, &info, request.reuse())? {
    Begin::Reused(stored) => stored,                       // computed before: use it
    Begin::Busy(owner) => return Err(busy_message(owner)), // another process holds it
    Begin::Claimed(mut run) => {
        let filtered = run_filter()?;                      // a refused run is `run.fail(..)`
        let stored_model = run.load(&identity.model)?;     // the stored compile, digest checked
        // ... refuse to store unless the model the filter ran is stored_model
        //     with the identity's parameters and interventions applied ...
        write_run(&mut run, &run_files(&filtered, &stored_model))?;
        run.finish()?
    }
}
```

### 7. Export from the store

Write each `-o` through a `Rendering` and each by-product export from the stored
by-product, never from in-memory state. End with `RunInfo::finish`.

```rust
invocation.render::<LoglikTable>(&store, &(), &store::ExportPath(path.to_path_buf()))?; // from the stored runs
info.finish(&store)?;
```

### 8. Test it

Write the command's `InputHarness` and add
`every_<command>_input_keeps_its_class` and
`every_<command>_argument_is_declared`, following `pfilter_store/tests.rs`, and
a tree test.

```rust
#[test]
fn every_pfilter_argument_is_declared() {
    let cmd = crate::Cli::command();                     // clap's view of the command line
    let sub = cmd.find_subcommand("pfilter").expect("the pfilter subcommand");
    // ... every argument clap accepts, and every field the Inputs derive declares ...
    assert_eq!(accepted, declared);
}
```

Then add a case to the poisoning test (`camdl_store/poisoning/tests.rs`,
`CASES`) that computes the new artifact kind, and a command to its base store
(`BASE`) that stores one result of it. The test fails while a registered
artifact kind is computed by no case.

```rust
const CASES: &[Case] = &[
    // ...
    Case {
        name: "pfilter",
        command: &["pfilter", "model.camdl", "--params", "truth.params.toml",
                   "--data", "weekly_cases=weekly_cases.tsv", "--particles", "40",
                   "--seed", "3", "--trace", "trace.tsv", /* ... */],
        named: None,                   // the label of a stored fit it names, if any
    },
];
```

### Changing an identity later

A new field of an identity type changes the key of every value of that type, so
every stored result of that artifact kind is computed again. Raise the type's
`version` when the same values would now produce different bytes.

## Reading a stored result

A stored result can be found and read from the command line with four commands,
which take a run id prefix, an `@label` or a result's folder
([Handles](#handles)).

`camdl list` prints the stored results as tables, one each for fits, profiles,
filter runs and simulations, newest first. It filters by `--model`,
`--scenario`, `--since` and `--kind`, and prints JSON under `--format json`.

`camdl show <handle>` prints a result's details, such as its folder, labels,
model, settings and seed for a filter run; `--format json` prints them as JSON.

`camdl cat <handle>` prints a result's main file: `traj.tsv` for a simulation
cell, `loglik.toml` for a filter run, `mle.toml` for a profile point,
`fit_state.toml` for a fit run and `problem.toml` for a fit problem. With a
label naming a whole invocation it prints the rendering of that invocation: the
combined table of a `simulate`, the design table of a `pfilter --sweep-file` or
the profile table of a `profile`. `--stream NAME` prints another file of the
result or of one of its by-products, such as `trace.tsv` or `draws.tsv`, and
`--chains` and `--exclude-chains` select chains of a fit run's `draws.tsv`.

`camdl verify` computes each completed result's run id again from its
`identity.json` and each manifest file's SHA-256, and exits 1 when it finds a
damaged result (`maintenance.rs:143`).

A result imported from another store (`camdl import`, under
[Maintaining a store](#maintaining-a-store)) is read like a local one: `list`
and `@label` find it through the invocation record `import` wrote, and a
computation here can depend on it by run id, although its folder is under the
other build's directory. Its `run.json` keeps the `invocation` number it had in
the other store, which names no invocation, or another one, here.

To work out why a result is what it is, look in its folder. `identity.json` is
the complete list of what the run id hashed, so two results that should have
been one and are not differ somewhere in two `identity.json` files, and
comparing them names the field. `run.json` gives the inputs as typed (`given`),
the execution values, the files with their digests, the dependencies and the
reads. The invocation record named by `run.json`'s `invocation` gives the
command line and, for each result the command asked for, whether it was computed
or reused.

A result that is present and not reused is usually damaged:
[`Store::inspect`](rust:store::storage::Store::inspect) reports
`EntryState::Damaged` with the reason, and the next `begin` moves it to
`.quarantine/`, where it keeps its folder name and records. Earlier attempts of
an in-place run are in its `.attempts/<n>/`, each with `attempt.json`. A
divergence leaves the recomputed files in `.quarantine/`.

The store library has `list` and `show` commands of its own; they are not
mounted, and camdl's own `list` and `show` take their place
(`store_commands.rs`).

### Maintaining a store

Four commands look after the store as a whole.

`camdl status` lists what is running, failed or interrupted.

`camdl clean` deletes the quarantine, every `.attempts/`, staging whose owner is
dead on this host and empty directories, and closes interrupted invocations. It
deletes no result.

`camdl reindex` rebuilds the index from the result records. The index is a
cache, so this is safe at any time.

`camdl import <copy>` brings in results computed in another store
([`Store::import`](rust:store::storage::Store::import), `import.rs`). The copy
holds the other store's `store.json`, the result folders wanted at the paths
they had there, with the parent folders a nested result sits in, and the
invocation records that listed them. The copy is moved into
`<root>/.staging/import.<owner>.<n>`, and every result folder in it is checked
as `verify` checks one, hashing every file; each result's dependencies
(`run.json`'s `deps`) must also be in the copy or completed here, so no result
is placed that its readers cannot open. Nothing is placed unless every folder
passes, and a refused copy is moved back where it was given. Each result keeps
its path relative to the root, so a result of another build sits under that
build's directory. A run id held here with the same result files is left in
place; one held with other result files, or a destination already taken, is
refused with
[`Error::ImportedDiffers`](rust:store::error::Error::ImportedDiffers), naming
both folders. Each folder is placed by a rename that never replaces what is at
the destination (`fsutil::rename_noreplace`: `renamex_np` with `RENAME_EXCL` on
macOS, `renameat2` with `RENAME_NOREPLACE` on Linux with glibc, a check before
the rename elsewhere), so a concurrent import of the same result that placed it
first makes it `present`. One local invocation lists every imported result with
outcome `reused` and the labels the copy's records gave it, so `camdl list` and
`@label` find them. An import stopped midway leaves each folder it placed whole,
and `clean` removes its staging once its process has stopped.

## What is enforced, and by what

The compiler and the types carry most of the store's guarantees. Two things they
cannot carry are that an execution input leaves the result unchanged, and that
engine code reads nothing behind the store's back.

| Property                                                                             | Enforced by                                                                                                                                                                                                | What remains open                                                                                                                                                           |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Every argument, settings-file key and environment variable has a class]{#enf-class} | Compiler: the derives reject an unannotated field; an unknown file key fails to parse                                                                                                                      | Code that reads the environment outside the declared struct                                                                                                                 |
| [Every declared input reaches a resolution step]{#enf-resolved}                      | Compiler: generated parts have private fields and one `into_parts` that returns them all; an unused binding is a warning, which `make lint` refuses                                                        | A step that ignores what an input means; the class checks cover it                                                                                                          |
| [The hash covers the whole identity]{#enf-hash}                                      | Compiler: one derive, every field hashed; the record form is generated from the same field list and hashes the same bytes; `camdl verify` recomputes each run id from its record                           | A content type whose byte form leaves something out                                                                                                                         |
| [A content's digest is the SHA-256 of the stored bytes]{#enf-digest}                 | Types: a content artifact's identity is its file's digest; `put` and `finish` check each digest the identity holds against a result file                                                                   | Nothing                                                                                                                                                                     |
| [What ran is what was hashed]{#enf-ran}                                              | Types: the engine takes its settings and content from the run; every command checks the model it runs against the stored compile by digest                                                                 | Engine code that reads a global or opens a file directly                                                                                                                    |
| [An output is a function of its identity]{#enf-function}                             | Library: the divergence check whenever a by-product is produced after its upstream, and when an atomic finish meets a stored copy of its own id                                                            | A dependence that no reproducing run has yet exposed; a recompute under `Force` or an incomplete build, which quarantines first and is not compared                         |
| [Execution inputs do not change results]{#enf-execution}                             | Class checks, each on a named fixture that exercises the input, with `shows_use` requiring the run to show the input was used                                                                              | Only as good as the fixture; not provable                                                                                                                                   |
| [Identity inputs do change the id]{#enf-identity}                                    | Class checks, using each field's `Vary` value                                                                                                                                                              | A second value that happens to be equivalent                                                                                                                                |
| [Only completed results are reused or read as results]{#enf-completed}               | Types: `Stored<A>` exists only for a completed result                                                                                                                                                      | A reader that opens files by a path it built                                                                                                                                |
| [Completed records are sealed]{#enf-sealed}                                          | Types: nothing takes a completed result mutably; labels live in invocation records                                                                                                                         | A hand-built path                                                                                                                                                           |
| [Every write goes through the store]{#enf-writes}                                    | Types: a run hands out files; its folder only as a path to show ([`Run::folder`](rust:store::run::Run::folder), `Stored::folder`)                                                                          | A hand-built path under the root                                                                                                                                            |
| [Every upstream read is declared]{#enf-reads}                                        | Types: [`Run::open`](rust:store::run::Run::open) takes a `Dep`; the lint and the poisoning test below                                                                                                      | A plain file read on a hand-built path that a lint exemption covers; a read, while a command resolves its inputs, of a stored fit it names, which the poisoning test allows |
| [One id is never computed twice at once]{#enf-claim}                                 | Library: a claim in both write modes; a second request for an atomic id waits for the first, and one for an in-place id is refused. A content artifact is not claimed: two writers produce identical bytes | Filesystems without exclusive create                                                                                                                                        |
| [A dead run is noticed]{#enf-dead}                                                   | Library: the owner check, with start time, and the beat counter                                                                                                                                            | A hung process still beats                                                                                                                                                  |
| [Old records are never misread]{#enf-records}                                        | Library: versions in every record; a mismatch is refused by name                                                                                                                                           | Nothing                                                                                                                                                                     |

### Guarding against undeclared reads

The types stop the obvious route, because a stored result gives the engine no
path. What they cannot stop is an ordinary file read on a path someone built by
hand. Two guards cover that.

**The lint.** `rust/clippy.toml` bans `std::fs::read`, `read_to_string`,
`File::open` and `OpenOptions::open`, and `std::env::var`, `var_os` and
`vars_os`, through Clippy's `disallowed-methods`. The store crate, where the
declared reads are implemented, is exempt as a whole
(`rust/crates/store/clippy.toml`). Elsewhere each remaining direct read carries
an `#[expect(clippy::disallowed_methods, reason = ...)]` whose reason says why
it is not an undeclared input: a file the user names whose bytes enter an
identity, a scratch copy of a stored result read out through the store, the
command's own working directory, the IR cache, or a command that stores nothing.

`make lint` runs the lint as an error on library and binary targets, and allows
it in tests (`Makefile:198-206`). An exemption silences it, so the lint is a
strong default and not a proof. camdl reads its environment in one function,
[`cli::env::get`](rust:cli::env::get) (`env.rs:259`), and
`rust/crates/cli/tests/environment_is_read_in_one_module.rs` checks that no
other code reads it. `std::fs::read_dir` is not banned: a listing yields no
bytes of a file.

**The poisoning test.** `every_stored_kind_reads_only_its_declared_dependencies`
(`camdl_store/poisoning/tests.rs:526`) builds a base store holding a result of
every artifact kind, then runs eight commands (`simulate`, `pfilter`, `fit run`;
`pfilter`, `simulate --draws posterior` and `profile` on a stored fit; `predict`
from a fit and from a fixed point), each on a copy of it. It runs each again on
a copy in which every file of every result is corrupted, keeping its size and
modification time, except the files reached from the new results' `deps`,
followed through each dependency's own `deps`. The second run must compute the
same run ids with the same result digests. A read of a corrupted file through
the store fails its digest check, and a plain read of a hand-built path yields
other bytes.

A stored fit the command names (`--fit @fit`, `camdl predict @fit`) is opened
whole while the command resolves its inputs, so its files, and the results its
lineage names, are left uncorrupted: what the command takes from it enters the
identity by value, which the class checks test. Every artifact kind
`camdl_kinds()` registers must be computed by some case, so a new artifact kind
fails the test until a case stores it.

A second test,
`a_fit_run_s_method_reaches_what_is_computed_from_it_only_through_its_id`
(`:583`), runs `pfilter`, `simulate --draws posterior` and `profile` against two
runs of one problem that differ only in `[method]`, and requires each run id the
two share to hold the same bytes.

### The tests behind each guarantee

| Property                                                       | Test                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Each input keeps its class; every argument is declared         | `every_<command>_input_keeps_its_class`, `every_<command>_argument_is_declared` for `simulate`, `pfilter`, `fit_run`, `predict` and `profile`                                                                                                                                                                                                                                        |
| A run makes the layout described here                          | `a_run_makes_the_tree_of_the_proposal` (`cli/src/pfilter_store/tests.rs`, `cli/src/sim_store/tests.rs`)                                                                                                                                                                                                                                                                              |
| A by-product added later reproduces the run and is then reused | `a_by_product_is_added_later_and_then_reused`                                                                                                                                                                                                                                                                                                                                        |
| A label names the invocation, and `cat` finds it               | `a_label_names_the_invocation_and_cat_finds_it`                                                                                                                                                                                                                                                                                                                                      |
| No stored artifact kind reads an undeclared file               | `every_stored_kind_reads_only_its_declared_dependencies`, `a_fit_run_s_method_reaches_what_is_computed_from_it_only_through_its_id` (`cli/src/camdl_store/poisoning/tests.rs`); `corrupting_undeclared_files_changes_nothing_a_run_computes` (`pfilter`'s fixture files)                                                                                                             |
| Only the stored compile is run                                 | `only_the_stored_compile_is_accepted_as_the_model_a_result_runs` (`cli/src/camdl_store/tests.rs`)                                                                                                                                                                                                                                                                                    |
| The budget refuses a run and stores no result file             | `a_budget_below_one_pass_is_refused_and_stores_nothing`                                                                                                                                                                                                                                                                                                                              |
| Compile settings and engine switches reach the run id          | `the_compile_settings_and_the_engine_switches_key_the_run`                                                                                                                                                                                                                                                                                                                           |
| Parameters and scenario do not key the compiled model          | `the_parameters_and_the_scenario_do_not_key_the_compiled_model`                                                                                                                                                                                                                                                                                                                      |
| Only the engine part keys the compiled model, canonically      | `only_what_the_engine_runs_keys_the_model_part`, `the_model_part_is_canonical_over_map_order` (`cli/src/camdl_store/tests.rs`)                                                                                                                                                                                                                                                       |
| `--no-store` leaves no store                                   | `no_store_leaves_nothing_behind`, `a_failed_no_store_run_leaves_no_temporary_store`                                                                                                                                                                                                                                                                                                  |
| `--force` and a claim from another host                        | `a_run_held_from_another_host_is_refused_and_force_computes_it_again`, `a_run_held_from_another_host_with_a_live_beat_is_refused_under_force`                                                                                                                                                                                                                                        |
| The encoding: tags, field names, variants, floats, run ids     | `a_struct_writes_its_tag_version_and_each_field_by_name_in_name_order`, `an_enum_writes_its_variant_by_name`, `a_float_is_finite_and_its_zeros_are_one_key`, `a_run_id_hashes_the_version_the_kind_the_ambient_key_and_the_identity_key` (`store/src/identity/tests.rs`)                                                                                                             |
| The class checks catch a wrong class                           | `an_identity_input_that_moves_only_a_content_artifact_fails`, `an_ignored_expands_input_fails` (`store/src/harness/tests.rs`)                                                                                                                                                                                                                                                        |
| Claims, waits, reclaims and quarantine                         | `an_atomic_claim_held_by_a_live_process_is_waited_for`, `an_in_place_run_whose_owner_died_reads_as_interrupted_and_is_reclaimed`, `a_run_held_from_another_host_is_busy_and_force_moves_it_to_quarantine`, `a_damaged_result_is_not_served_and_the_next_begin_quarantines_it`, `a_reused_process_id_with_another_start_time_is_not_read_as_the_owner` (`store/src/storage/tests.rs`) |
| A copied store and a changed file                              | `a_store_copied_without_modification_times_reuses_every_result`, `a_result_file_with_another_digest_is_refused_naming_its_result` (`store/src/storage/tests.rs`)                                                                                                                                                                                                                     |
| Failure in each write mode                                     | `a_failed_atomic_result_leaves_no_folder_and_its_reason_is_in_the_invocation_record`, `a_failed_in_place_run_is_marked_failed_and_a_retry_keeps_the_attempt`, `retries_keep_at_most_three_attempts`                                                                                                                                                                                  |
| Divergence in reproducing mode                                 | `a_reproduced_upstream_that_differs_is_divergent_and_its_files_are_kept`, `a_reproduced_upstream_that_skips_a_stored_file_is_divergent` (`store/src/run/tests.rs`)                                                                                                                                                                                                                   |
| Held content and file names                                    | `finish_refuses_held_content_with_no_result_file_of_its_digest`, `file_names_are_checked`, `finish_refuses_while_a_buffered_writer_is_open` (`store/src/storage/tests.rs`)                                                                                                                                                                                                           |
| Handles and labels                                             | `an_id_prefix_names_one_result_and_an_ambiguous_one_lists_every_match`, `a_label_names_the_latest_invocation_whose_results_all_completed`, `a_label_is_refused_while_an_invocation_carrying_it_runs` (`store/src/handle/tests.rs`)                                                                                                                                                   |
| `verify`, `clean` and the index                                | `verify_computes_the_run_id_again_from_the_identity_file`, `clean_removes_quarantine_attempts_dead_staging_and_empty_directories_only`, `the_index_is_a_cache_rebuilt_from_the_records` (`store/src/maintenance/tests.rs`)                                                                                                                                                           |

Three properties rest on tests whose reach is limited. The execution class is
checked only on the fixture each field names. A dependence of a result on
something outside its identity is found only when a reproducing run computes
that result again, or when the poisoning test's cases happen to read the file in
question. And a direct read under a lint exemption is checked only by the
poisoning test, for the commands and options its cases run.

## Where to read the code

The library, `rust/crates/store/src`:

- `artifact.rs`: `Artifact`, `Primary`, `ByProduct`, `ContentFiles`,
  `ContentArtifact`, `WriteMode`, `Id`, `Kinds`, `TopLevel`, `Build`, and
  `run_id`.
- `hash.rs`, `identity.rs`: `Key`, `CanonicalHasher`, `ContentAddressed`,
  `HASH_VERSION`; `Identity`, `Finite`, `Hashed`, `Content`.
- `inputs.rs`, `harness.rs`: the input classes, the `Inputs` trait and `Vary`;
  the class checks.
- `storage.rs`: `Store`, `Begin`, `Reuse`, `Wanted`, `begin_inner`, `put`,
  locating and placing folders.
- `run.rs`: `Run`, `FileHandle`, `finish` and reproducing mode.
- `stored.rs`: `Stored`, `Dep`, `Entry`, `EntryState`.
- `record.rs`, `check.rs`: the record files' formats; the damage check.
- `invocation.rs`, `handle.rs`: `RunInfo`, `StoreRequest`, the invocation record
  and renderings; handles and labels.
- `layout.rs`, `owner.rs`, `index.rs`, `maintenance.rs`, `commands.rs`: layouts
  and `path_label`; owners and the beat; the index; `verify`, `clean` and
  `reindex`; `StoreCommand`.

camdl, `rust/crates/cli/src`:

- `camdl_store.rs`: the ambient identity, `camdl_kinds`, the root,
  `camdl_groups`, the shared value types, the three content artifacts and
  `require_stored_model`.
- `sim_store.rs` with `sim_command.rs`; `pfilter_store.rs` with `pfilter.rs`;
  `fit_store.rs` with `fit_command.rs`; `fit_store.rs` (`Prediction`) with
  `predict_command.rs` and `fit/predict.rs`; `profile_store.rs` with
  `profile.rs`: each command's artifact kinds and the command that stores them.
- `browse.rs`, `store_commands.rs`: `list`, `show`, `cat`; `status`, `verify`,
  `clean`, `reindex`.
- `version.rs`: `ENGINE_ID`, `ENGINE_REUSE`, `VERSION_SHORT`.
- `camdl_store/poisoning/tests.rs`: the poisoning test.

The design is `docs/dev/proposals/2026-11-20-run-store-api.md`; the user-facing
behaviour is `docs/camdl-run-spec.md` §2 and §4.4-4.5.
