# Run store API: identity, provenance and layout from argument types

Nov 20, 2026 · @Vince Buffalo

Supersedes: phase 3 of `2026-11-17-honest-input-addressed-store.md` (the input
architecture) and its named follow-up, the fit's typed identity. Phases 1 and 2
of that proposal stand.

## Summary

This is the target design for camdl's run store, in two parts. Part 1 is the
store library, written without reference to camdl's commands. Part 2 is camdl on
that store: worked examples, the features that depend on it, and the path from
today's store.

The core of it: a project declares every input once, in arguments and in the
files they name. Typed steps turn those into one identity per result. An output
is a function of its identity alone. The store does every write and every read
of a stored file.

Most of the library exists in camdl today, scattered: locks naming process and
host, heartbeats, `progress.json`, `.attempts/`, `.quarantine/`, `.staging/`,
profile invocation records and `index.json`. This design unifies them and adds
no mechanism of its own beyond the invocation record.

The main points:

- **Names are labels, never identity.** A scenario's name, a model file's name,
  a replicate number and the user's seed are recorded with the invocation. No
  stored result file contains them and no folder name is computed from an
  identity.
- **The combined table of a multi-cell `simulate` is a rendering, not a stored
  result.** `-o` and `camdl cat` render it from the stored cells and the
  invocation record, so it keeps today's shape.
- **An export is a copy or a rendering of stored files.** `--dates` and a file
  format are rendering options. An export never runs the model again.
- **Every command run writes an invocation record.** It holds the inputs as
  given and the ordered list of results the command asked for, with their
  labels. A batch is an invocation with several results. Looking a result up by
  its configuration file is a query over these records.
- **One compile yields several contents.** A section says which part of a model
  file is identity for which result. Adding a quantity or editing a docstring
  still recomputes nothing.
- **By-products commit with their upstream,** and every particle-filter
  diagnostic is stored. Thinning large ones is a later feature.
- **A fit is a problem and the runs on it.** Method settings, including those
  that sit in `[estimate]`, belong to the run.
- **Existing fits are abandoned at migration.**
- **The compiled model is stored whole,** derivative tables included, under the
  digest of what is stored. The IR does not change.
- **A configuration file never names a run.** Labels, ids and paths do.
- **Some features are left for later.** Archive and prune, search by identity
  field, `diff`, a `watch` command and stored combinations are listed under
  future features and are not built.

Status: accepted design; not yet implemented. The code in this document is a
sketch and has not been compiled. File and line references are to camdl at
`1c79769fe`.

## Part 1: the store library

Everything from here to the enforcement section is the library. It names no
camdl command.

## Guarantees

The store must never return a wrong answer; every other guarantee serves that
one. Recomputing something reusable costs time. Reusing something that should
have been recomputed is a silent scientific error, so doubt resolves towards
recomputing.

1. **Sound reuse.** A stored result is reused only if the requested computation
   would have produced it, by the same build, under a stated meaning of "same".
2. **A stored result file is a function of its identity alone.** Any value
   written into a result that is not in its identity is a bug: a position in a
   grid, the order of a list the identity treats as a set, a label. A recompute
   that differs from the stored bytes makes such a bug visible. What a user
   recognises a result by (a name, a replicate number, the seed they typed) is a
   label: it is recorded with the invocation and appears only in renderings.
3. **Complete identity by construction.** Everything that can change a result is
   part of its identity: arguments, file contents, environment switches, derived
   seeds, the build. Adding an input without deciding its class does not
   compile.
4. **Canonical identity.** Two requests that mean the same thing get the same
   id. Files are hashed by content and their paths only recorded. Defaults are
   resolved first, so a flag at its default and no flag are one identity.
5. **A content key is the digest of what is stored and consumed.** Never of a
   projection of it.
6. **Only completed results are reused.** Every stored folder has a status:
   running, completed, failed or interrupted. A running or stopped run is
   visible and explains itself, and is never served as a result.
7. **Completed records are sealed.** Nothing rewrites a completed result or its
   record. The only change is appending a naming event.
8. **Every write and every read of a stored file goes through the store.**
   Claims, manifests and records cannot be skipped, and every read of another
   result's files is recorded.
9. **Self-explaining results.** The record alone says what produced a result:
   the inputs as given, the build, the environment, and the results it consumed
   with their digests and builds.
10. **Readers ask the record.** Seed, kind, parent and status come from the
    record, never from a folder name or its depth.
11. **Old records are refused loudly, never misread.**
12. **Cheap adoption.** A project declares inputs in one place and writes no
    hashing or locking code. The library knows nothing about the science.

## Concepts

Two ideas carry the design: every input has a class and only identity inputs are
hashed, and every stored folder has a status and only completed ones are reused.

&#91;embedded content: input split · three classes, where each ends up\]

The split separates the arguments by class. Typed resolution steps then turn the
identity inputs into the identity, and the engine receives that identity and the
execution values. Requests to the store, such as a label, are recorded and never
hashed.

&#91;embedded content: run statuses · four states, one reusable\]

Running, completed and failed are written to the record. Interrupted is never
written: a reader concludes it when the record says running and the owner is
gone.

| Term              | Meaning                                                                                                                                                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Input             | Anything that reaches a command: a flag, a key in a settings file, the content of a data file, an environment variable, or an earlier result.                                                                                                                |
| Class             | What an input may do. The main three: identity (can change a result, hashed), execution (used by the engine, must not change the result), store (a request to the store). API 1 lists them all.                                                              |
| Declared inputs   | The struct of identity-class inputs a derive generates from a command's arguments or from a settings file. Its fields are private and it cannot be hashed.                                                                                                   |
| Resolution        | The typed steps that consume declared inputs by value and produce identity values: parsing a file, compiling a model, merging flags over file settings, filling in defaults.                                                                                 |
| Identity          | A project-written typed value that fully determines one result's bytes.                                                                                                                                                                                      |
| Content           | A typed value with a byte form. Its key is the digest of those bytes.                                                                                                                                                                                        |
| Content artifact  | An artifact whose identity is the digest of its one result file, such as a compiled model. Other results depend on it.                                                                                                                                       |
| Identity record   | The form of an identity that `identity.json` holds and readers get. Content the artifact stores itself appears there as a key.                                                                                                                               |
| Key               | A 32-byte hash. A struct's key covers its tag, its version, and each field's name and key.                                                                                                                                                                   |
| Ambient identity  | The build fingerprint and the identity-class environment variables. The store adds it to every run id.                                                                                                                                                       |
| Artifact          | A kind of stored result, such as a simulation, a fit problem, a fit run or a prediction. Each has one identity type and one write mode.                                                                                                                      |
| By-product        | A second artifact produced by the same computation as another, such as observations drawn while a trajectory is simulated. Its identity holds the upstream's identity record plus its own settings.                                                          |
| Run id            | The hash of the artifact's kind, the ambient identity and the identity key.                                                                                                                                                                                  |
| Status            | Running, completed or failed, as written in the record. Interrupted is derived by readers and never written.                                                                                                                                                 |
| Claim             | Exclusive ownership of a run id while it is computed.                                                                                                                                                                                                        |
| Owner and beat    | The process holding claims, named by host, process id and start time. It increments one counter, its beat, every few seconds.                                                                                                                                |
| Invocation record | The store-owned, durable record of one command run: the inputs as given, a digest of each configuration file, the ordered list of results the command asked for with each one's labels, and how many completed and failed. It is written on a cache hit too. |
| Batch             | An invocation that asks for several results.                                                                                                                                                                                                                 |
| Label             | Anything a user recognises a result by that is not in its identity: a scenario's name, a model file's name, a replicate number, the seed the user typed, a `--label`. Labels live in invocation records and naming events.                                   |
| Dependency        | An identity field naming a file of another result: the upstream run id, the file, and the digest of what was consumed. The upstream may come from another build.                                                                                             |
| Naming event      | A `--label` minted on a result or an invocation when a command runs. Naming events are appended and never edited.                                                                                                                                            |
| Record            | `identity.json` (the identity record) and `run.json` (status, manifest, dependencies, inputs as given, naming events). Sealed once completed.                                                                                                                |
| File role         | Result files are in the manifest and determined by the identity. Provenance files are in the manifest but not compared on a recompute. Attempt files describe one attempt and are outside the manifest.                                                      |
| Export            | A file written to a path the user gave: a copy of a stored result file, or a rendering computed from stored files and the invocation record.                                                                                                                 |
| Rendering         | A pure function of stored result files, an invocation record and rendering options such as date columns or a file format. It is not stored.                                                                                                                  |
| Index             | A rebuildable log of results, their labels and their dependency edges in both directions. It can be rebuilt from the records and the invocation records.                                                                                                     |
| Layout            | Where a folder sits under the store root. It is a view; readers never parse it.                                                                                                                                                                              |
| Store             | The only code that writes or reads stored folders. One command uses one store root.                                                                                                                                                                          |

## API 1: inputs and identity

Inputs are declared where they enter: in arguments, in settings files and in the
environment. Typed steps turn them into the identity. Nothing is loaded outside
those steps.

### Declaring arguments

`#[derive(Inputs)]` goes on the clap structs. A field with no `#[input(..)]`
attribute is a compile error.

```rust
#[derive(clap::Args, Inputs)]
pub struct NumericsArgs {
    #[arg(long)] #[input(identity)] pub dt: f64,
    #[arg(long)] #[input(identity)] pub binomial: bool,
}

#[derive(clap::Args, Inputs)]
pub struct ExampleArgs {
    #[command(flatten)] #[input(group)] pub numerics: NumericsArgs,
    #[input(identity)]                  pub model: InputFile<ModelSource>,
    #[arg(long)] #[input(identity)]     pub params: InputFile<ParamsToml>,
    #[arg(long)] #[input(identity)]     pub seed: u64,
    #[arg(long)] #[input(execution)]    pub parallel: usize,
    #[arg(long)] #[input(store)]        pub label: Option<String>,
    #[arg(long)] #[input(skip)]         pub quiet: bool,
}
```

| Attribute   | Use it when                                                                 | Where it goes                                                              | Its generated test                                                                                       |
| ----------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `identity`  | The value can change a result                                               | The declared inputs, and from there the identity                           | Varying it changes the run id                                                                            |
| `expands`   | It decides which results a batch has, such as a list of scenarios or seeds  | The declared inputs; each member's own value enters that member's identity | A smaller set's members keep their run ids inside a larger set                                           |
| `adds`      | It requests a by-product, such as observations                              | The declared inputs; it selects by-products                                | The upstream's run id is unchanged and the by-product exists                                             |
| `execution` | The engine needs it, and it must not change the result                      | The execution struct; recorded                                             | Outputs are identical across two values, on a fixture that exercises it                                  |
| `export`    | It is a path to write a copy or a rendering of stored files to              | The exports                                                                | A copy matches the stored digest; a rendering made twice from the same stored files gives the same bytes |
| `render`    | It chooses how an export is rendered, such as date columns or a file format | The exports                                                                | No stored file differs across two values                                                                 |
| `store`     | It is a request to the store: a label, or which store root to use           | The store request                                                          | None                                                                                                     |
| `recorded`  | People want it on record and nothing reads it                               | `run.json`                                                                 | None                                                                                                     |
| `group`     | The field is a nested struct that itself derives `Inputs`                   | Wherever its own fields go                                                 | Its fields' tests                                                                                        |
| `skip`      | It only affects what is printed                                             | Nowhere                                                                    | None                                                                                                     |

This is close to the set camdl's input classes have today, without the level.

- **An `execution` field may not be an `InputFile`;** the derive refuses it. A
  file that is read is an input.
- **A group shared between commands must mean the same in each.** If a flag is
  an identity input in one command and only a check in another, it is not
  shared: the second command gets its own field. camdl's `--backend` is the case
  in point. In `profile` it only decides whether the run is refused, since each
  algorithm runs on one backend (the comment at `input_class.rs:290-293`); in
  `simulate` it changes the result.

The derive generates one struct per class. The split does no reading and cannot
fail.

```rust
/// Fields are private. The only way in is into_parts, which returns every field.
pub struct ExampleDeclared { /* numerics, model, params, seed */ }

impl ExampleDeclared {
    pub fn into_parts(self) -> (NumericsDeclared, InputFile<ModelSource>, InputFile<ParamsToml>, u64);
}

pub struct ExampleExecution { pub parallel: usize }
pub struct ExampleRequest   { /* label */ }

impl Inputs for ExampleArgs {
    type Declared  = ExampleDeclared;
    type Execution = ExampleExecution;
    type Exports   = ();                      // this command has no export paths
    type Request   = ExampleRequest;

    const FIELDS: &'static [FieldInfo];      // name and class of every field, for tests
    fn split(self) -> Split<Self>;
}

impl<I: Inputs> Split<I> {
    /// The declared inputs, the execution inputs, the export paths, the store request,
    /// and the inputs as given (argv included), for the record.
    pub fn into_parts(self) -> (I::Declared, I::Execution, I::Exports, I::Request, Given);
}
```

Two rules follow from the private fields, and they apply to every generated
struct, groups included.

- **No field access.** `declared.seed` does not compile. A caller gets the
  fields only from `into_parts`, all at once, so adding a field breaks every
  caller until it is handled.
- **A group is taken apart in full or passed on whole.** `NumericsDeclared` is
  either split with its own `into_parts` or handed to one function that takes
  it.

`into_parts` returns a tuple, so two neighbouring fields of the same type could
be swapped by mistake. Distinct types for distinct meanings, such as a seed
type, keep that rare.

`ExampleDeclared` does not implement `Identity`. It cannot be hashed or passed
to the store.

### Declaring the contents of input files

A file argument says what kind of file it is, and a settings file declares a
class for every key.

```rust
/// A file argument. T says what the file holds.
pub struct InputFile<T>(PathBuf);

/// A settings file. An unknown key is a parse error; a key with no class is a compile error.
#[derive(Deserialize, FileInputs)]
#[serde(deny_unknown_fields)]
pub struct ParamsToml {
    #[input(identity)] pub values: BTreeMap<String, f64>,
    #[input(identity)] pub tables: BTreeMap<String, InputFile<Table>>,
}

impl<T: FileInputs> InputFile<T> {
    /// Parse and split, as for arguments.
    pub fn parse(self) -> Result<Parsed<T>>;
}

impl<T: FileInputs> Parsed<T> {
    /// Every part of the file: declared inputs, execution inputs, store request,
    /// and the raw bytes, which are kept as a provenance file.
    pub fn into_parts(self) -> (T::Declared, T::Execution, T::Request, SourceBytes);
}

impl<T: Content> InputFile<T> {
    /// Read a file that is hashed whole. The path is recorded; only the content is hashed.
    pub fn content(self) -> Result<Hashed<T>>;
}
```

| Kind of file | Example                      | What is hashed                   | What is kept                                                              |
| ------------ | ---------------------------- | -------------------------------- | ------------------------------------------------------------------------- |
| Settings     | `fit.toml`, a parameter file | The parsed identity-class values | The raw bytes, as a provenance file                                       |
| Content      | A data table                 | The content                      | The content, as a content artifact                                        |
| Source       | A model file                 | What it compiles to              | The compiled parts, as content artifacts; the source as a provenance file |

Comments and formatting in a settings file never change a run id. Field classes
replace a hand-kept list of excluded keys.

### Declaring environment variables

Environment variables get the same classes. The project lists them once; nothing
else may read the environment.

```rust
#[derive(EnvInputs)]
pub struct CamdlEnv {
    #[env("CAMDL_EVAL_UNRESOLVED")]         #[input(identity)] pub eval_unresolved: bool,
    #[env("CAMDL_NO_BINDING_CACHE")]        #[input(identity)] pub no_binding_cache: bool,
    #[env(prefix = "CAMDL_EXPERIMENTAL_")]  #[input(identity)] pub experimental: BTreeMap<String, String>,
    #[env("CAMDL_OUTPUT_DIR")]              #[input(store)]    pub output_dir: Option<PathBuf>,
}
```

### From declared inputs to identity

Resolution is a function that takes the declared inputs by value and hands each
one to the typed step that uses it.

```rust
#[deny(unused_variables)]
pub fn resolve(declared: ExampleDeclared) -> Result<ExampleIdentity> {
    // Every declared input, by value. A new one changes this tuple and breaks the build here.
    let (numerics, model, params, seed) = declared.into_parts();

    let model  = compile(model.content()?)?;              // Hashed<CompiledModel>
    let params = resolve_params(params.parse()?, &model)?;   // takes the parsed file whole

    Ok(ExampleIdentity {
        model,
        numerics: Numerics::check(numerics)?,             // takes the group whole
        params,
        seed: ProcessSeed::new(seed),
    })
}
```

- **Every step is a typed function** that takes the declared values it uses. No
  function takes "everything that was loaded".
- **A new input does not compile until it is handled.** `into_parts` returns
  every field, an unused binding is an error, and there is no field access to
  skip one with.
- **Every part of a parsed file is handled too:** its declared inputs, its
  execution inputs and its store request.
- **Defaults are resolved here, before the identity.** An identity field holds a
  plain value, not an `Option`, so a flag at its default and no flag give one
  key.
- **A merge is one function.** Where a flag overrides a key in a file, one
  `merge` takes both values and sets the precedence.
- **An input a command does not use is released by name,** with a reason, such
  as held-out tables in a command that only fits.
- **Files are read only through `InputFile`.**
- **What the compiler cannot see** is a step that takes an input and ignores
  what it means. The generated identity tests cover that.

The declared inputs are also written to `run.json` as given. The user's `--seed`
stays on record even when a derived seed is what is hashed.

### Content: one key, one value

A key must be the digest of exactly what is stored and consumed. Today's model
digest hashes a projection of the compiled model: it leaves out quantities,
contrasts, derivative tables and some presentation fields. Two compiles that
differ only in those then share one digest, so storing either under that key
could hand a method a model without the derivatives it needs.

The rule: normalize by splitting the content, never by hashing part of it. The
split is one project-written function that takes the whole compiled output
apart, so a new field does not compile until it is assigned to a part.

```rust
/// A typed value with a byte form.
pub trait Content: Sized {
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self>;
}

/// Content with its key. The key is the SHA-256 of to_bytes(), computed in the only constructor.
pub struct Hashed<T: Content> { /* Arc<T>, Key */ }
```

- **Everything a result consumes is a typed value** whose key is the digest of
  its bytes.
- **Content a result does not depend on is not part of that value.** A
  trajectory consumes a model's dynamics. Quantities and contrasts are separate
  content, keyed separately and depended on only by the results that use them.
- **The parts of one file do not overlap.** If two results need different parts
  of a file, the file's layout is changed until those parts are separate
  sections. A key is never the digest of a file with some fields blanked.
- **Labels are not content.** A name carried inside a file is taken out before
  the content is keyed and recorded as a label.
- **The whole is recorded.** A content artifact's record holds the digest of the
  full compiled output and of the source file it came from, as provenance.
  Neither is part of any key.
- **Content lives in one of two places:** an artifact's result files, or an
  input file. There is no separate content area.

An identity can hold content in two ways.

| The identity holds             | When                                                                                               | The record holds | The running code gets the content by           |
| ------------------------------ | -------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------------------------- |
| `Dep<C>` on a content artifact | The content is shared by many results, such as a compiled model                                    | The dependency   | `run.load(&dep)`, cached per process by digest |
| `Hashed<T>`                    | The artifact writes that content among its own result files, such as a fit problem's training rows | The key          | `run.identity()`                               |

For the second case the derive generates a record form of the identity, with
each `Hashed<T>` replaced by its key. The key of the identity is always computed
from that record form. At `finish` the store checks that a result file with each
such digest exists.

```rust
pub trait Identity {
    type Record: Serialize + DeserializeOwned;
    fn record(&self) -> Self::Record;
    fn key(&self) -> Key;                 // always computed from the record
}
```

`Run::identity()` returns the identity. `Stored::identity()` returns the record.

### Keys

There is one encoding. No project writes hashing code for a struct.

- **A struct's key** is the hash of its tag, its version, and each field's name
  and key, taken in name order.
- **Names matter, location does not.** Renaming a field or changing the tag
  changes the key. Moving the type to another module or reordering its fields
  does not.
- **Every field is always hashed,** including at its default. Adding a field
  changes the key of every value of that type. That is accepted; a schema is not
  bent to keep old ids.
- **The version is bumped** when the same values would now produce different
  stored bytes.

| Type                                    | Key                                                                                                                         |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `bool`, integers, `String`, plain enums | Hash of a fixed encoding                                                                                                    |
| `Finite` (a checked `f64`)              | Hash of the bit pattern. NaN and infinity are refused. `-0.0` and `0.0` differ, which over-separates in the safe direction. |
| `Vec<T>`                                | Hash of the element keys, in order                                                                                          |
| `BTreeMap<K, V>`, `BTreeSet<T>`         | Hash of the entries, in sorted order                                                                                        |
| `Option<T>`                             | Allowed only where absence is itself meaningful, never for an unset default                                                 |
| `Hashed<T>`                             | The digest of the content's bytes                                                                                           |
| `Dep<A>`                                | An upstream run id, file name and content digest (API 3)                                                                    |
| An enum with fields                     | The variant's name and its fields' keys                                                                                     |

### Ambient identity

The build and the identity-class environment variables are part of every run id.
The project supplies them once, when it opens the store.

```rust
#[derive(Identity)]
#[identity(tag = "engine", version = 1)]
pub struct Engine {
    pub build: String,              // fingerprint of source, lockfile, features, compiler, target
    pub env: CamdlEnvDeclared,
}

pub enum Build<T> {
    Known(T),
    Incomplete(T),                  // some part of the fingerprint could not be determined
}

let store = Store::open(root, Build::Known(engine))?;
```

With `Build::Incomplete`, the store never reuses anything: it computes, moves
any stored result of the same id to quarantine, and writes its own. Because the
build is in every id, a new build reuses nothing from an old one.

### Artifacts

Each kind of stored result implements one trait.

```rust
pub trait Artifact {
    const KIND: &'static str;                  // such as "sim"
    const WRITE: WriteMode;                    // Atomic or InPlace (API 2)
    type Identity: Identity;

    /// A readable folder name, fixed by the first writer. It may use the identity and that
    /// writer's labels. Nothing reads it back.
    fn name(identity: &Self::Identity, labels: &Labels) -> String;

    fn layout(_identity: &Self::Identity) -> Layout { Layout::flat() }
}

/// A typed run id.
pub struct Id<A: Artifact> { /* Key + PhantomData<A> */ }
```

The run id is the hash of `KIND`, the ambient key and `identity.key()`.

### Generated tests

The field list drives a generated test per class (the table under "Declaring
arguments"). Most need something the field must declare.

```rust
/// Another valid value of the same type.
pub trait Vary { fn vary(&self) -> Self; }

#[arg(long)] #[input(identity)]
pub dt: f64,                                        // numbers and enums provide Vary

#[input(identity, vary = fixtures::other_model)]
pub model: InputFile<ModelSource>,                  // a file needs a named second file

#[arg(long)] #[input(execution, exercised_by = fixtures::two_cell_grid)]
pub parallel: usize,
```

- **Identity.** Vary the field and require the run id to change.
- **Expands.** Run a set and a larger set; require the shared members to keep
  their run ids.
- **Adds.** Run without and then with the request; require the upstream's run id
  to be unchanged and the by-product to exist.
- **Execution.** Run the named fixture with two values. Require identical
  outputs, and require the fixture to show the input was used.
- **Export.** For a copy, require its digest to equal the stored file's. For a
  rendering, render twice from the same stored files and require identical
  bytes.
- **Render.** Run with two values and require every stored file to be unchanged.
- **An execution field with no `exercised_by` is a compile error.** Without a
  fixture that exercises it, an input wrongly marked execution would pass.

## API 2: running a computation

Every result is produced under a claim, and its record carries a status. Only a
completed result is ever reused.

### Statuses

| Status      | Written by                         | Meaning                                                                | Reused |
| ----------- | ---------------------------------- | ---------------------------------------------------------------------- | ------ |
| Running     | The store, at the claim            | A process holds the claim                                              | Never  |
| Completed   | `finish`                           | The manifest is built and the result is whole                          | Yes    |
| Failed      | `fail`, or the claim being dropped | The run stopped; the reason is kept when one was given                 | Never  |
| Interrupted | Nobody; readers derive it          | The record says running, but the owner is dead or its beat has stopped | Never  |

Interrupted has to be derived. A killed process, a lost node, or a build
compiled to abort on panic runs no cleanup, so nothing can write it.

### Starting a run

```rust
pub enum Begin<A: Artifact> {
    Reused(Stored<A>),       // a completed result exists, with every by-product that was wanted
    Busy(Owner),             // another live process holds the claim
    Claimed(Run<A>),         // this process now owns the run id
}

pub enum Reuse { IfCompleted, Force }

impl Store {
    pub fn begin<A: Artifact>(
        &self, identity: &A::Identity, info: RunInfo, reuse: Reuse,
    ) -> Result<Begin<A>>;

    /// As begin, also wanting by-products of the same computation.
    pub fn begin_with<A: Artifact>(
        &self, identity: &A::Identity, wanted: &Wanted<A>, info: RunInfo, reuse: Reuse,
    ) -> Result<Begin<A>>;

    pub fn get<A: Artifact>(&self, id: &Id<A>) -> Result<Option<Stored<A>>>;     // completed only
    pub fn inspect<A: Artifact>(&self, id: &Id<A>) -> Result<Option<Entry<A>>>;  // any status
}
```

An artifact picks one of two write modes. Both take the claim first, so two
concurrent runs of one id never both compute it.

| Mode      | The claim is                                                    | Files go to                                         | Visible while running    | After a failure                 | Suits                                         |
| --------- | --------------------------------------------------------------- | --------------------------------------------------- | ------------------------ | ------------------------------- | --------------------------------------------- |
| `Atomic`  | The staging folder itself, created exclusively under the run id | That staging folder, renamed into place by `finish` | No                       | Nothing is left                 | Quick results: a simulation, a compiled model |
| `InPlace` | A lock in the final folder                                      | The final folder                                    | Yes, with status running | The folder stays, marked failed | Long runs: a fit, a survey, a profile point   |

### Writing and ending a run

```rust
impl<A: Artifact> Run<A> {
    pub fn identity(&self) -> &A::Identity;

    pub fn result_file(&self, name: &str) -> Result<FileHandle>;      // in the manifest
    pub fn provenance_file(&self, name: &str) -> Result<FileHandle>;  // in the manifest, not compared
    pub fn attempt_file(&self, name: &str) -> Result<FileHandle>;     // outside the manifest

    pub fn finish(self) -> Result<Stored<A>>;
    pub fn fail(self, operation: &str, reason: &str) -> Result<()>;
}
```

The run hands out files; it does not hand out its folder. A file's role is
chosen when it is opened, which replaces a list of reserved file names. A
`FileHandle` can be sent to another thread, so several chains can write through
one run. Ending the run takes it by value, so that stays with the owner.

| How the run ends                                            | Record afterwards            | A reader sees                                    |
| ----------------------------------------------------------- | ---------------------------- | ------------------------------------------------ |
| `finish`                                                    | Completed, with the manifest | Completed                                        |
| `fail(operation, reason)`                                   | Failed, with both strings    | Failed                                           |
| The `Run` is dropped: an early return or an unwinding panic | Failed, no reason            | Failed, reason unknown                           |
| The process is killed or the node is lost                   | Still says running           | Interrupted, derived from the owner and the beat |

A command uses it like this. `Store::run` wraps the same match for the common
case.

```rust
match store.begin::<FitRun>(&identity, info, reuse)? {
    Begin::Reused(stored) => Ok(stored),
    Begin::Busy(owner)    => Err(Error::InProgress(owner)),
    Begin::Claimed(mut run) => match engine::fit(&mut run, &execution) {
        Ok(())   => run.finish(),
        Err(err) => { run.fail("fitting", &err.to_string())?; Err(err) }
    },
}
```

- **Retry.** `begin` on a failed or interrupted id reclaims it. The earlier
  record and attempt files move to `.attempts/<n>/`, result files are cleared,
  and the last three attempts are kept.
- **Force.** `Reuse::Force` moves a completed result of the same id to
  quarantine and starts again. It never deletes, and never touches another id.
- **Divergence.** If `finish` finds a completed result of the same id already in
  place, the result files' digests are compared. A difference is an error, and
  the new bytes go to quarantine as evidence. This is the check that exposes an
  output that depends on something outside its identity.
- **Readers.** `get` returns only completed results, and checks the manifest and
  the file sizes. Full digests are checked by `verify`, and by `Run::open` as it
  reads. `Stored<A>` is the only type that opens result files.
- **Damaged results.** A completed result whose files fail a check is reported
  as damaged and is not served. The next `begin` for that id moves it to
  quarantine and computes again.
- **No store.** `Store::scratch(ambient)` has the same API over a temporary root
  that is deleted when dropped. A command that must not store, such as the
  filter runs inside `compare`, uses it.

### By-products

One computation may produce several artifacts. Observations are the example:
they are drawn from the simulation's in-memory state, which the stored
trajectory file does not contain, so they cannot be made from that file
afterwards.

```rust
/// An artifact produced by the same computation as another.
pub trait ByProduct: Artifact {
    type Of: Artifact;
}

/// The by-products wanted from one run of A, each with its identity.
pub struct Wanted<A: Artifact> { /* ... */ }

impl<A: Artifact> Run<A> {
    /// A run for this by-product, if it was wanted and is not already stored.
    pub fn also<B: ByProduct<Of = A>>(&mut self) -> Result<Option<Run<B>>>;

    /// True when A itself is already stored and this run exists only for by-products.
    pub fn reproducing(&self) -> bool;
}
```

- **Identity.** A by-product's identity holds the upstream's identity record
  plus its own settings. It holds no dependency on a file.
- **Produced together.** A by-product wanted with its upstream comes from the
  same run. `also` claims it and the engine writes it.
- **Committed with the upstream.** The engine does not finish a by-product. The
  upstream's `finish` commits the upstream first and then each by-product, so a
  completed by-product never exists without a completed upstream. If the
  upstream fails or diverges, its by-products are failed with it.
- **Produced later.** If the upstream is stored and a by-product is not,
  `begin_with` returns a claimed run in reproducing mode. The engine runs the
  computation again. The upstream's run id does not change and its stored files
  are not rewritten; the recomputed files are compared with them, which is a
  free determinism check.
- **Layout.** A by-product sits inside its upstream's folder.

### Exports

An export is a file written to a path the user gave. It is the user's way to
work with results without the store, so it must be directly usable: tidy,
labelled, in the format asked for. It is made after `finish` or `get`, never by
running the computation again.

```rust
impl<A: Artifact> Stored<A> {
    /// A byte copy of one stored result file.
    pub fn export(&self, file: &str, to: &ExportPath) -> Result<()>;
}

impl Invocation {
    /// A rendering of this invocation's results: a pure function of their stored files,
    /// this record's labels and the rendering options.
    pub fn render<R: Rendering>(&self, options: &R::Options, to: &ExportPath) -> Result<()>;
}
```

- **A copy** reproduces a stored file exactly.
- **A rendering** combines or reshapes stored files and adds labels from the
  invocation record: a combined table across cells, date columns, a rounded
  summary, another file format.
- **The same function serves the path and the screen.** A project's `cat`
  command calls the rendering that `-o` calls, so the two always agree.
- **If what a user wants cannot be made from stored files,** it becomes a
  by-product first and is then exported.

### Invocation records

Every command run writes one invocation record, whether its results were
computed or were already stored.

- **What it holds.** The inputs as given, with `argv`; a digest of each
  configuration file; the ordered list of results the command asked for; each
  result's labels; counts of completed and failed; a progress snapshot.
- **It is durable and never rewritten after the command ends.** It is not a
  cache: the index is rebuilt from the records and the invocation records
  together.
- **It is where a second request for a stored result is recorded.** The result's
  own record is sealed and is not touched.
- **`clean` does not remove invocation records.**

### Sealed records

After `finish`, a result's files and its record are not written again. The one
exception is appending a naming event (API 4). Anything else that would have
been added later has another home: a by-product, an attempt file written before
`finish`, or the invocation record.

### Owners and the beat

Liveness belongs to the process, not to each claim. A process that holds claims
is an owner, named by host, process id and process start time. It keeps one beat
file and increments a counter in it every few seconds. Every claim it holds
points at that owner.

| Observer            | Rule                                                                                               |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| On the owner's host | A process with that id and that start time exists: alive. Otherwise dead. No waiting.              |
| On another host     | Read the counter, wait a few intervals on the reader's own clock, read it again. Unchanged: stale. |

The start time is checked so that a reused process id is not read as the owner
still being alive. No timestamp is compared across machines, because on a shared
filesystem a file's time comes from the file server.

```rust
pub enum Liveness { Alive, Dead, Stale { unchanged_for: Duration }, Unknown }

impl Store {
    pub fn liveness(&self, owner: &Owner, wait: Duration) -> Result<Liveness>;
}
```

- **A beat means alive, not progressing.** A deadlocked fit still beats.
- **Same-host reclaim is automatic.** A dead owner's claims are reclaimed by the
  next `begin`.
- **Cross-host reclaim is manual.** A stale owner on another host is reported
  and reclaimed only with `Reuse::Force`.
- **A long-lived reader** keeps the last counter it saw, so it needs no wait
  after the first look.

### Batches

A command that produces several results opens a batch. Its invocation record
lists the planned run ids in the order the command asked for them, each with its
labels, and how many are completed and failed.

- **`status` shows batches as well as runs.** A 10,000-cell grid is one line
  with a count, not 10,000 lines.
- **Results appear as they finish.** Each member is visible the moment its
  `finish` returns.
- **The same identity may be listed twice,** such as two scenarios with the same
  effect. It is claimed and computed once, and both entries point at the one
  result.
- **A dead owner's staging folders are found through its invocation record** and
  reclaimed with it.
- **Within one atomic result there are no partial files.** An in-place run shows
  its traces growing.

### What a result costs

A large grid on a shared filesystem cannot afford a lock, a beat and several
renames per cell. These counts are rough estimates, not measurements.

| Step                         | Atomic result                                             | In-place run                                     |
| ---------------------------- | --------------------------------------------------------- | ------------------------------------------------ |
| Check for a completed result | One read, or none when a batch has planned from the index | One read                                         |
| Claim                        | One exclusive folder create                               | A folder create and a lock create                |
| Records                      | Two writes                                                | Two writes at the claim, one rewrite at `finish` |
| Result files                 | One write each                                            | One write each, over the run                     |
| Commit                       | One rename                                                | One rename of the record                         |
| Index                        | One buffered append to the owner's log                    | The same                                         |
| Beat and invocation record   | None per result; one of each per process                  | None per run; the owner's                        |

An atomic result with one file is about six filesystem operations. After
claiming, the store checks once more that no completed result arrived in the
meantime.

### Progress and events

The store owns the channel and the file formats. The project owns what is in
them.

```rust
impl<A: Artifact> Run<A> {
    /// Replace the latest snapshot atomically.
    pub fn progress(&self, snapshot: &impl Serialize) -> Result<()>;

    /// Append one numbered line to the event log.
    pub fn event(&self, item: &impl Serialize) -> Result<()>;
}
```

Both are attempt files, so they sit outside the manifest and survive in
`.attempts/` after a retry. Every kind of run gets them, not only fits. A live
view reads the two files, whose names and formats are documented, and finds them
by handle.

## API 3: results that use other results

A result that reads another result's files names each one in its identity, and
the store does the reading.

### Dependencies

```rust
/// One file, or one named part of a file, of an upstream result.
pub struct Dep<A: Artifact> { /* upstream Id<A>, name, content digest; the upstream's build is recorded */ }

impl<A: Artifact> Stored<A> {
    pub fn id(&self) -> &Id<A>;
    pub fn identity(&self) -> &<A::Identity as Identity>::Record;   // the record form

    pub fn dep(&self, file: &str) -> Result<Dep<A>>;          // digest taken from the manifest
    pub fn dep_all(&self) -> Dep<A>;                          // every result file, never a provenance file
    pub fn dep_part(
        &self, file: &str, part: &str,
        extract: impl Fn(&[u8]) -> Result<Vec<u8>>,
    ) -> Result<Dep<A>>;                                      // digest of the extracted bytes
}

impl<B: Artifact> Run<B> {
    /// The only way a running computation reads an upstream file.
    pub fn open<A: Artifact>(&mut self, dep: &Dep<A>) -> Result<impl Read>;

    /// One file out of a whole-result dependency made with dep_all.
    pub fn open_in<A: Artifact>(&mut self, dep: &Dep<A>, file: &str) -> Result<impl Read>;

    /// The typed content of a content artifact, parsed once per process and cached by digest.
    pub fn load<C: ContentArtifact>(&mut self, dep: &Dep<C>) -> Result<Arc<C::Content>>;
}
```

The sequence is always the same:

1. Before the run, get the upstream as a `Stored<A>`, which exists only for
   completed results.
2. Take a `Dep` for each file to be consumed and put it in the identity.
3. During the run, read through `run.open(&dep)`. It checks the bytes against
   the digest as they are read and writes the read into `run.json`.

`Stored<A>` gives a running computation no path, so a file cannot be read
without being declared.

**What a `Dep` hashes:** the upstream run id, the name, and the digest. The
digest catches an upstream file that changed after it was stored. The run id
keeps a margin if some read ever escapes `open`. The cost is small: two fits
with identical draws get separate predictions instead of sharing one.

**A dependency may point at a result made by another build.** Otherwise
rebuilding camdl would leave every stored fit unusable for prediction, and fits
take hours. It is sound because the dependent is keyed to exactly the bytes it
consumed, under its own build. The upstream's build is recorded in the
dependency, and `show` flags a cross-build dependency.

| Case                                                                   | How it is written                                                                            |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| A whole file of an upstream result                                     | `stored.dep("draws.tsv")`                                                                    |
| Part of a file, such as a fit's start values                           | `stored.dep_part("fit_state.toml", "start_values", extract)`                                 |
| Recorded but not hashed, because the values enter identity another way | `info.lineage(dep)` puts it in `run.json` only                                               |
| A loose file outside any store, such as a starting point in a TSV      | An ordinary `InputFile`, hashed by content                                                   |
| A result in a different store root                                     | Foreign input: its files are content-hashed input files, recorded as lineage, never a `Dep`. |
| A directory with no record, offered as a run to resume                 | Refused, with a message naming the run to resume by handle                                   |

### Content artifacts

Content that many results consume is stored once, as an artifact whose identity
is the digest of its one result file.

```rust
/// An artifact whose identity is its content.
pub trait ContentArtifact: Artifact<Identity = Hashed<Self::Content>> {
    type Content: Content;
    const FILE: &'static str;                // such as "model.ir.json"
}

impl Store {
    /// Store the content if it is not there yet, and return it as a result to depend on.
    pub fn put<C: ContentArtifact>(&self, content: Hashed<C::Content>, info: &RunInfo) -> Result<Stored<C>>;
}
```

- **Key equals stored bytes by construction.** The identity is the digest of the
  file the artifact holds.
- **Normalization happens before `put`.** Two model files that compile to the
  same content are one artifact.
- **Results depend on it with an ordinary `Dep`.** The index holds the edge.
  Nothing needs separate reference counting.
- **If the content was built from other inputs,** such as a model rewritten from
  data before it is hashed, the step that builds it takes those inputs' content,
  and the content artifact's record lists them as lineage.

### Shared content that a problem owns

A fit's problem is content several runs share, but it is more than one file: the
training rows, the bound tables and the resolved problem settings. It is an
ordinary `Atomic` artifact whose identity is the problem itself.

- **Results that use it depend on it** through a `Dep`.
- **Nothing rewrites it.** A second run of the same problem finds it completed
  and writes nothing.
- **The user's original files are provenance files.** They are kept from the
  first writer and are not part of what the id promises.

### Combining results

A combination that only reshapes its members and adds labels is a rendering. A
combined table across the cells of a batch is the case in point: its rows are
the members' rows, and its extra columns (scenario name, replicate number) are
labels. It is rendered from the invocation record and is not stored.

The members never hold the combination in their own identity, so they keep their
ids when it grows. camdl's two combinations, the combined simulation table and
the profile table, are both renderings. A stored result that computes something
new from several others is not needed by any command and is not part of this
design.

### Planning a batch

```rust
pub struct Planned<A: Artifact> { pub id: Id<A>, pub state: PlanState }

pub enum PlanState { Absent, Running(Owner), Completed, Failed, Interrupted }

impl Store {
    /// Read-only: what exists for each identity. Nothing is claimed.
    pub fn plan<A: Artifact>(&self, identities: &[A::Identity]) -> Result<Vec<Planned<A>>>;

    /// Open a batch and claim its members in one pass, with the by-products wanted of each.
    /// An id another process holds comes back as Busy, not waited for.
    pub fn begin_many<A: Artifact>(
        &self, members: &[(A::Identity, Wanted<A>)], info: &RunInfo, reuse: Reuse,
    ) -> Result<Batch<A>>;
}

impl<A: Artifact> Batch<A> {
    pub fn members(&mut self) -> impl Iterator<Item = Begin<A>>;
    pub fn progress(&self, snapshot: &impl Serialize) -> Result<()>;

    /// The invocation record: the members in the order asked for, with their labels.
    pub fn invocation(&self) -> &Invocation;
}
```

A grid, a sweep or a profile plans first, reports how many results are already
stored, and runs the rest. `begin_many` writes the invocation record, removes
repeated identities, then claims in run-id order without blocking on a busy id.
Two overlapping batches therefore cannot deadlock or compute the same id twice.

### The index

The index records every result, its labels and its dependency edges. It is a
cache: everything in it can be rebuilt from the records and the invocation
records.

- **It is a set of append-only logs, one per owner.** A finish or a naming event
  appends one line. No shared file is rewritten per result. Readers merge the
  logs, and `reindex` rebuilds and compacts them.
- **It holds edges in both directions:** which results a result consumed, and
  which results consumed it.
- **It holds both kinds of edge:** dependencies that are hashed, and lineage
  that is only recorded. Removing an upstream leaves either kind dangling.
- **Nothing destructive reads it.** Results are removed by hand, as today.

## API 4: layout, handles and commands

Where a folder sits is a view chosen per artifact. Nothing reads meaning out of
a path.

### Layout

```rust
impl Layout {
    /// <root>/<kind>/<name>-<12 hex of run id>/
    pub fn flat() -> Self;

    /// Places the result at <parent's folder>/<subdir>/<name>-<12 hex of run id>/.
    pub fn inside<P: Artifact>(parent: &Id<P>, subdir: &'static str) -> Self;
}
```

- **The folder name is fixed by the first writer.** It is `Artifact::name` plus
  the first 12 hex digits of the run id. The name may use that writer's labels,
  such as a scenario's name, so a later request for the same result under
  another name finds the folder already named. The user's `--label` is not in
  it.
- **A folder is found by id and then verified.** The full run id in
  `identity.json` decides, never the path.
- **Two run ids can share their first 12 digits.** The second folder then gets a
  longer suffix. A test covers it.
- **A nested result is still its own artifact,** with its own records and
  manifest. Removing a parent removes what is nested in it.
- **Readers never parse a path.** The seed, the kind and the settings come back
  as the identity record. The parent comes from the dependencies, or for a
  by-product from the upstream record in its identity.

### Handles

A command-line argument that names a stored result has the type `RunRef<A>`.

```rust
impl Store {
    pub fn resolve<A: Artifact>(&self, handle: &RunRef<A>) -> Result<Stored<A>>;      // completed only
    pub fn resolve_entry<A: Artifact>(&self, handle: &RunRef<A>) -> Result<Entry<A>>; // any status
}
```

| Form                      | Resolves to                                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------- |
| A path to a stored folder | That result                                                                                        |
| An id prefix              | The one result whose run id starts with it; an ambiguous prefix is refused with every match listed |
| `@L`                      | The latest completed result named `L`                                                              |
| `@L~n`                    | The n-th earlier result named `L`                                                                  |

- **One store root per command.** Every handle resolves in the root the command
  writes to, never in a default one.
- **Only these forms name a result.** A configuration file passed where a fit is
  expected names a configuration: it is an `InputFile`, merged with the flags
  and resolved to a fit problem, which is created if absent.
- **A configuration file never names a run.** One unchanged file produces
  several runs whenever something outside its text changes: the data, a seed
  list, a sweep, or the build. A command that needs one run and is given a
  configuration file refuses, and lists the runs that file has produced, newest
  first, with their labels and ids. The list comes from the invocation records,
  which hold a digest of each configuration file.

### Labels

A label is minted when a command runs and is never edited afterwards.

```rust
pub struct NamingEvent {
    pub seq: u64,              // store-wide order
    pub time: Timestamp,       // for people; not used for ordering
    pub name: String,
    pub kind: NamingKind,      // Minted or Corrected
}
```

- **Minting.** `--label L` is a request to the store, carried in `RunInfo`. The
  store appends a naming event when the command runs, whether its result is
  computed now or was already stored.
- **A label names what the command made, once.** For a command that makes one
  result, that result. For a command that makes several, the invocation, and
  never each member.
- **Resolving a batch's name.** `@L` on a batch names the invocation. A command
  that renders, such as `cat` or an export, takes it as it is. A command that
  needs one result takes the only member when there is one, and otherwise
  refuses with the members listed.
- **Naming after the fact.** Running the same command again with `--label L`
  finds the stored results and mints `L` on them. That is also how an earlier
  result becomes the current holder of a name again.
- **Append only.** A record keeps every name it has been given. Appending a
  naming event is the one change a sealed record allows.
- **No clashes, with one refusal.** A name another result holds moves to the new
  one, with a printed note. Minting is refused while a run holding that name is
  still running.
- **Order.** Minting takes one store-wide lock, held only for the append, and
  the event gets the next sequence number. A counter per owner would avoid the
  lock, but it cannot say which of two owners' events came later, and "latest"
  is what `@L` means.
- **No user-facing label command.** `dev relabel <handle> <name>` is a
  maintainer's correction: it appends a corrected event and refuses while the
  run is running.
- **Movable tags** such as "best current model" are out of scope. They would be
  a separate mechanism with its own history.

### Commands

The library ships a clap subcommand set that a project adds to its own. Every
command works from the index and the records.

| Command                       | What it answers or does                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `list <kind>`                 | What results exist, with status, names and time                                              |
| `show <handle>`               | What produced this result: identity record, dependencies, sources, files, naming events      |
| `status`                      | Which runs and batches are running, failed or interrupted, with owner, liveness and progress |
| `verify [<handle>]`           | Whether each folder's id matches its `identity.json` and each file matches its digest        |
| `clean`                       | Clears staging, quarantine and old attempts                                                  |
| `reindex`                     | Rebuilds and compacts the index from the records                                             |
| `migrate`                     | Upgrades this store's layout one numbered step at a time, or refuses and says why            |
| `dev relabel <handle> <name>` | Appends a correcting naming event                                                            |

## What is enforced, and by what

The compiler and the types carry most of it. Two things they cannot carry are
that an execution input leaves the result unchanged, and that engine code reads
nothing behind the store's back.

| Property                                                               | Enforced by                                                                                                            | What remains open                                                                                   |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Every argument, settings-file key and environment variable has a class | Compiler: the derives reject an unannotated field; an unknown file key fails to parse                                  | Code that reads the environment outside the declared struct                                         |
| Every declared input reaches a resolution step                         | Compiler: generated structs have private fields and one `into_parts` that returns them all; unused bindings are denied | A step that ignores what an input means; the identity tests cover it                                |
| The hash covers the whole identity                                     | Compiler: one derive, every field hashed, the key computed from the record form                                        | A content type whose byte form leaves something out                                                 |
| A content key is the digest of the stored bytes                        | Types: a content artifact's identity is its file's digest; `finish` checks each `Hashed` field against a result file   | Nothing                                                                                             |
| What ran is what was hashed                                            | Types: the engine takes its settings and content from the run                                                          | Engine code that reads a global or opens a file directly                                            |
| An output is a function of its identity                                | Library: the divergence check, on every recompute and every by-product rerun                                           | A dependence that no rerun has yet exposed                                                          |
| Execution inputs do not change results                                 | Generated tests, each with a named fixture that exercises the input                                                    | Only as good as the fixture; not provable                                                           |
| Identity inputs do change the id                                       | Generated tests, using each field's `Vary` value                                                                       | A second value that happens to be equivalent                                                        |
| Only completed results are reused or read as results                   | Types: `Stored<A>` exists only for a completed result                                                                  | A reader that opens files by a path it built                                                        |
| Completed records are sealed                                           | Types: nothing but the naming append takes a completed result mutably                                                  | A hand-built path                                                                                   |
| Every write goes through the store                                     | Types: a run hands out files and never its folder                                                                      | A hand-built path under the root; a test can check that non-store commands leave the root untouched |
| Every upstream read is declared                                        | Types: `Run::open` takes a `Dep`                                                                                       | A plain file read on a hand-built path; the three guards below                                      |
| One id is never computed twice at once                                 | Library: a claim in both write modes                                                                                   | Filesystems without exclusive create                                                                |
| A dead run is noticed                                                  | Library: the owner check, with start time, and the beat counter                                                        | A hung process still beats                                                                          |
| Old records are never misread                                          | Library: versions in every record; a mismatch is refused by name                                                       | Nothing                                                                                             |

### Guarding against undeclared reads

The types stop the obvious route, because a stored result gives the engine no
path. What they cannot stop is an ordinary file read on a path someone built by
hand. Two guards cover that.

**1. A lint, at compile time.** Clippy's `disallowed_methods` setting bans
direct file and environment reads in the engine crates.

```toml
# clippy.toml for the engine crates
disallowed-methods = [
    { path = "std::fs::read",           reason = "read inputs through Run::open or InputFile" },
    { path = "std::fs::read_to_string", reason = "read inputs through Run::open or InputFile" },
    { path = "std::fs::File::open",     reason = "read inputs through Run::open or InputFile" },
    { path = "std::env::var",           reason = "declare the variable in the environment inputs" },
]
```

Only the store crate, where `InputFile` lives, is exempt. CI treats the lint as
an error. An `#[allow]` can silence it, so it is a strong default and not a
proof.

**2. A poisoning test.** For each kind of run, corrupt every upstream file that
is not among its declared dependencies, run again, and require identical output.
A changed output means an undeclared read.

Two stronger guards, an open-file audit and a sandbox, are listed under future
features.

## Part 2: camdl on the store

The rest of the document is camdl: what a model file contributes to identity,
two worked examples, the features that build on the store, and the path from
today's store.

## What a model file contributes to identity

A model file is compiled to one output, and different results depend on
different parts of it. This section is the definition. It belongs in the
run-identity engineering document (`docs/engineering/run-identity.md`) when the
design lands.

### Today

One function decides, by listing every field of the compiled model and either
hashing it or naming the reason it is left out.

- **`impl ContentAddressed for Model`** (`runid/src/ir_hash.rs:1655`) takes the
  model apart field by field, so a new field does not compile until it is hashed
  or excluded.
- **`normalize_for_hash`** (`runid/src/inputs.rs:396`) blanks two presentation
  fields first.
- **`model_ir_hash`** and **`model_ir_hash_with_gradients`** (`inputs.rs:361`,
  `:377`) are the two digests: without and with the derivative tables.
- **Documentation** sits outside the model, in the envelope
  (`ir/src/envelope.rs:97-104`), so no digest covers it.

The weakness is that the digest covers part of the file while the whole file is
what is stored and read.

### In the target

The same exhaustive function becomes a split. It takes the compiled output apart
and assigns every field to one part.

| Part               | What it holds                                                                                                                                                                                                                                                                                                   | Who depends on it                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Model              | Everything the engine runs: compartments, transitions, equations, time functions, tables, interventions, observation models, parameters, constants, bindings, initial conditions, the output schedule, the simulation window, presets, and every derivative table, each beside the expression it differentiates | Every simulation, filter, fit problem and prediction, as `Dep<CompiledModel>` |
| Quantities         | The `quantities` and `contrasts` blocks                                                                                                                                                                                                                                                                         | A prediction, and a `--quantities-out` by-product                             |
| Labels             | The model's name, which is the file's stem                                                                                                                                                                                                                                                                      | Nothing; recorded with the invocation                                         |
| Rendering defaults | `output.format` and `simulation.time_semantics`                                                                                                                                                                                                                                                                 | Nothing stored; an export reads them                                          |
| Provenance         | The `#'` documentation, the model as its author wrote it, the source file                                                                                                                                                                                                                                       | Nothing; kept as provenance files                                             |

- **The parts are disjoint.** Each is one or more whole top-level fields of the
  compiled output. No field appears in two parts and none is blanked, so each
  part's key is simply the digest of its bytes.
- **The record of every compiled model holds two more digests:** of the full
  compiled output and of the source file. They answer "is this the same file"
  without being part of any key.

### Derivative tables stay beside their expressions

The compiler writes, next to each rate and each likelihood argument, the formula
for its derivative with respect to each parameter. They are stored together on
purpose: `Diffable { expr, grad, proj_grad }` (`ir/src/deriv.rs:155-168`) makes
it impossible for a differentiable position to lack its derivative
(`docs/dev/proposals/2026-07-06-seal-differentiation-coverage-3b.md`). In a
gradient table an absent key means a true zero, so a table that was lost or
misaligned would read as a zero gradient and a fit would continue with a wrong
answer.

The model part therefore includes the derivative tables, and the IR is not
changed. Today's digest leaves them out; the target's key covers them because
they are in the stored bytes.

- **Nothing is lost by covering them.** Derivative tables change without the
  dynamics changing only when the compiler changes. A committed compiler change
  is a new build, since the build fingerprint includes the commit
  (`cli/src/engine_fingerprint.rs:10-15`), and a new build already re-keys every
  result. An uncommitted one now recomputes too, which is the safe direction.
- **The gradient digest goes away.** Today a gradient-reading fit adds a second
  digest that includes the tables (`fit/cas.rs:527`). The model's key now does
  that for every result.

### The compile with the state Jacobian

One table is optional. The derivatives of each rate with respect to the
compartments, the state Jacobian, can be most of a compiled model's size, so the
compiler omits them unless asked. Only `nl-lbfgs` on the ODE backend reads them
(`fit/config_v2.rs:2890`).

- **The two compiles are two contents,** each keyed by its own bytes.
- **A fit problem depends on the ordinary compile,** so every method shares one
  problem.
- **An `nl-lbfgs` run on the ODE backend also depends on the larger compile,**
  and runs on it.
- **The two are checked to be the same model.** Today's structural model hash
  (`runid/src/inputs.rs:361`), which ignores derivative tables and hoisting
  names, must agree for both. It is kept for this check and is no longer a key.

### What changes a result, and what does not

This is today's behaviour, kept.

| Change to the model file                             | Effect                                                                                                      |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Reformatting, comments, a `#'` docstring             | Nothing is recomputed                                                                                       |
| Renaming or copying the file                         | Nothing is recomputed                                                                                       |
| Adding or editing a quantity or a contrast           | Trajectories, filters and fits are reused; predictions that use quantities are new                          |
| A change to the dynamics                             | Everything that depends on the model is new                                                                 |
| A compiler change that alters only derivative tables | Everything that depends on the model is new. It arrives with a new build, which re-keys every result anyway |
| The `description`                                    | Nothing is recomputed. Today it re-keys the model; it becomes provenance                                    |

### A model rewritten from data

Some values are written into the compiled model before it is keyed.

- **Anchored times.** `first_obs` and `last_obs` become the simulation's start
  and end. They are the earliest and latest training rows that hold a value.
- **Bound tables.** The contents of tables named in `[data.tables]`.

The content artifact is keyed by the bytes that result, so data that does not
change those bytes does not change the key. Its record lists the data files as
lineage.

Held-out rows need a rule. Their outcome values never reach the compiled model.
Their dates and their pattern of missing values can, because a cutoff such as
`last_obs - 6 weeks` is resolved over the whole file. That is accepted: when
observations were made is part of the study design, and what was observed is
not.

## Worked example: `camdl simulate`

One `simulate` invocation is a batch of cells. Each cell is one atomically
written `Sim` result, and may have by-products. The struct below shows the
classes on a subset of the real flags; `rust/crates/cli/src/input_class.rs`
lists every flag with its class today.

### 1. Declared inputs

```rust
#[derive(clap::Args, Inputs)]
pub struct SimulateArgs {
    #[input(identity)]                        pub model: InputFile<ModelSource>,
    #[arg(long)] #[input(identity)]           pub params: Option<InputFile<ParamsToml>>,
    #[arg(long = "param")] #[input(identity)] pub overrides: Vec<ParamOverride>,
    #[command(flatten)] #[input(group)]       pub numerics: NumericsArgs,     // backend, dt, flags
    #[arg(long)] #[input(expands)]            pub scenario: Vec<String>,
    #[arg(long)] #[input(expands)]            pub seeds: Vec<u64>,
    #[arg(long)] #[input(adds)]               pub obs: bool,
    #[arg(long)] #[input(adds)]               pub event_log: bool,
    #[arg(short = 'o')] #[input(export)]      pub output: Option<ExportPath>,
    #[arg(long)] #[input(render)]             pub dates: bool,
    #[arg(long)] #[input(execution, exercised_by = fixtures::two_cell_grid)]
                                              pub parallel: usize,
    #[arg(long)] #[input(store)]              pub label: Option<String>,
    #[arg(long)] #[input(store)]              pub output_dir: Option<PathBuf>,
    #[arg(long)] #[input(skip)]               pub quiet: bool,
}
```

`--scenario` and `--seeds` decide which cells exist; one scenario's effects and
one derived seed enter each cell's identity. `--obs` and `--event-log` ask for
by-products. `-o` is a rendering made afterwards, and `--dates` adds date
columns to it.

The real command has more axes than this sketch: sweep points, parameter draws
and replicates each add cells the same way, as `expands` inputs.

### 2. Resolution

```rust
pub struct SimShared {
    pub model: Dep<CompiledModel>,
    pub numerics: Numerics,
    pub params: ResolvedParams,
}

pub struct Cell {
    pub scenario: ResolvedScenario,   // the scenario's effects, without its name
    pub seed: ProcessSeed,            // derived from the user's seed, the point and the replicate
    pub window: Window,
    pub labels: CellLabels,           // scenario name, replicate number, the user's seed; never hashed
}

#[deny(unused_variables)]
pub fn resolve(declared: SimulateDeclared, store: &Store, info: &RunInfo)
    -> Result<(SimShared, Vec<Cell>, ByProductRequest)>
{
    let (model, params, overrides, numerics, scenario, seeds, obs, event_log) = declared.into_parts();

    // The compiled model, stored once, by the digest of what is stored.
    let compiled = compile(model.content()?)?.model();            // Hashed<CompiledModelContent>
    let cells    = Cell::grid(scenario, seeds, &compiled)?;       // consumes `scenario` and `seeds`
    let file     = params.map(InputFile::parse).transpose()?;     // the parsed file, whole
    let params   = resolve_params(file, overrides, &compiled)?;   // takes every part of it
    let request  = ByProductRequest::new(obs, event_log, &compiled);
    let model    = store.put::<CompiledModel>(compiled, info)?.dep("model.ir.json")?;

    Ok((SimShared { model, numerics: Numerics::check(numerics)?, params }, cells, request))
}
```

`CompiledModel` holds everything the engine runs, derivative tables included.
Quantities and contrasts are other content; the model's name is a label.

### 3. The identities

The five fields of `SimIdentity` are the five levels `simulate` hashes today. A
cell's labels are not among them.

```rust
#[derive(Identity)]
#[identity(tag = "sim", version = 1)]
pub struct SimIdentity {
    pub model: Dep<CompiledModel>,
    pub config: SimConfig,            // backend, dt, this cell's window, flags; the calendar when it changes output
    pub params: ResolvedParams,       // every parameter, after all sources
    pub scenario: ResolvedScenario,   // the scenario's effective changes
    pub seed: ProcessSeed,
}

/// Observations drawn while the trajectory is simulated.
#[derive(Identity)]
#[identity(tag = "obs", version = 3)]
pub struct ObsIdentity {
    pub sim: <SimIdentity as Identity>::Record,     // the upstream's identity, not a file of it
    pub observation: ObsModel,
    pub emit_every: Option<u32>,
    pub seed: ObsSeed,
}
```

There is no ensemble identity. The combined table is a rendering (section 6).

### 4. The artifacts

```rust
pub struct CompiledModel;

impl ContentArtifact for CompiledModel {
    type Content = CompiledModelContent;
    const FILE: &'static str = "model.ir.json";
}

pub struct Sim;

impl Artifact for Sim {
    const KIND: &'static str = "sim";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = SimIdentity;

    // The first writer's scenario name. A label, so it is not read back.
    fn name(_id: &SimIdentity, labels: &Labels) -> String { labels.scenario().to_string() }
}

pub struct Obs;

impl ByProduct for Obs { type Of = Sim; }

impl Artifact for Obs {
    const KIND: &'static str = "obs";
    const WRITE: WriteMode = WriteMode::Atomic;
    type Identity = ObsIdentity;

    fn name(id: &ObsIdentity, _labels: &Labels) -> String { id.observation.streams().join("+") }
    // A by-product is laid out inside its upstream's folder by the library.
}
```

### 5. The command and the engine

```rust
pub fn simulate(args: SimulateArgs, ambient: Build<Engine>, reuse: Reuse) -> Result<()> {
    let (declared, execution, exports, request, given) = args.split().into_parts();

    let store = Store::open(request.root()?, ambient)?;
    let info  = RunInfo::new(given, &execution, request);
    let (shared, cells, by_products) = resolve(declared, &store, &info)?;

    let members = cells.iter()
        .map(|cell| {
            let identity = SimIdentity::for_cell(&shared, cell)?;
            let wanted   = by_products.for_cell(&identity);       // the Obs and EventLog identities
            Ok((identity, wanted))
        })
        .collect::<Result<Vec<_>>>()?;

    // The label names the invocation, whose record lists the cells in order with their labels.
    let mut batch = store.begin_many::<Sim>(&members, &info, reuse)?;

    for begun in batch.members() {
        match begun {
            Begin::Reused(_)        => {}
            Begin::Busy(owner)      => return Err(Error::InProgress(owner)),
            Begin::Claimed(mut run) => {
                engine::simulate(&mut run, &execution)?;          // on error the run is dropped
                run.finish()?;                                    // commits the cell, then its by-products
            }
        }
    }

    // -o: the combined table, rendered from the stored cells and the invocation's labels.
    exports.render::<CombinedTable>(batch.invocation())?;
    Ok(())
}

pub fn simulate(run: &mut Run<Sim>, execution: &SimulateExecution) -> Result<()> {
    let identity = run.identity().clone();
    let model    = run.load(&identity.model)?;                    // parsed once per process
    let traj     = run.result_file("traj.tsv")?;
    let mut obs  = run.also::<Obs>()?;                            // Some if wanted and not yet stored

    for step in Engine::new(&model, &identity, execution) {
        step.write_row(&traj)?;
        if let Some(obs) = &mut obs { draw_observations(obs, &step)?; }   // from the in-memory state
    }
    Ok(())                                                        // the upstream's finish commits `obs`
}
```

The engine takes the model and every setting from the run. It never sees the
arguments or the labels, so what ran cannot differ from what was hashed.

When a cell is stored and its observations are not, the same engine function
runs in reproducing mode. It writes the trajectory again to a scratch file,
`finish` compares that with the stored one, and only the observations are
stored. Today's tests already pin that a trajectory is the same bytes with
observations or the event log switched on.

`reactive_log.tsv` is not a by-product. A model with an active reactive policy
always writes it, so it is an ordinary result file of the cell.

### 6. What lands in the store

```text
results/
  compiled_model/
    sir-81cc03a1f2d4/
      identity.json  run.json  model.ir.json
  sim/
    slow-3fa9c1d2e07b/
      identity.json  run.json  traj.tsv
      obs/
        cases-5c0d93aa41e7/
          identity.json  run.json  cases.tsv
    slow-a41be07c9d02/ ...
    fast-0b77c2e1f6a3/ ...
    fast-e2f09d14b8c5/ ...
  .invocations/
    20261120-7f3a.json              # this run: inputs as given, the four cells in order, their labels
  .owners/  .index/                 # owner beats, index logs
```

The combined table is not in the store. `-o out.tsv` and `camdl cat @baseline`
both render it from the four cells and the invocation record, in the order the
command asked for them:

```text
replicate  scenario  seed  t  S    I   R
0          slow      1     0  990  10  0
0          slow      1     1  985  14  1
1          slow      2     0  990  10  0
2          fast      1     0  990  10  0
3          fast      2     0  990  10  0
```

- **`replicate` and `scenario` are labels** from the invocation record.
- **`seed` is the seed that reproduces that row's cell** when passed back to
  `--seeds`.
- **The table has the same columns for one cell and for many,** so a script does
  not branch on the number of cells. Today a single cell's `-o` file has no
  coordinate columns.
- **With `--dates`** the same rendering adds date columns.

A cell's `run.json`, with invented values:

```json
{
  "status": "completed",
  "names": [],
  "invocation": "20261120-7f3a",
  "ambient": { "build": "1ae16a5ce28b...", "env": {} },
  "given": {
    "argv": ["simulate", "models/sir.camdl", "--scenario", "slow,fast", "--seeds", "1,2"],
    "model": "models/sir.camdl",
    "scenario": ["slow", "fast"],
    "seeds": [1, 2]
  },
  "execution": { "parallel": 8 },
  "files": { "traj.tsv": { "role": "result", "sha256": "7c10..." } },
  "deps": [
    { "kind": "compiled_model", "id": "81cc03a1f2d4...", "file": "model.ir.json",
      "sha256": "81cc...", "build": "1ae16a5ce28b..." }
  ],
  "owner": { "host": "...", "pid": 22149, "started": "..." }
}
```

A cell carries no name of its own, and lists no children: a by-product is found
through the index, by the upstream record in its identity. The record names the
invocation that first computed the cell; later invocations that reuse it are
found through their own records.

### 7. What changes what

| Change                                                        | Effect                                                                                                                                                     |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--parallel 16`                                               | Every cell is reused                                                                                                                                       |
| The same command again with `--label v2`                      | Every cell is reused; the new invocation is named `v2`, and `camdl cat @v2` renders its table                                                              |
| The model file is reformatted, or the project directory moves | Every cell is reused; the compiled content is the same                                                                                                     |
| The same values come from a file instead of `--param`         | Every cell is reused                                                                                                                                       |
| `--scenario fast,slow` in the other order                     | Every cell is reused; the rendered table lists `fast` first, as asked                                                                                      |
| A scenario is renamed without changing its effects            | Every cell is reused; the table shows the new name                                                                                                         |
| `--dt 0.5`                                                    | Every cell is new                                                                                                                                          |
| A third scenario is added                                     | The four stored cells are reused; two are computed                                                                                                         |
| `--obs` is added later                                        | Each cell is simulated again to draw its observations. The trajectory ids do not change, and each recomputed trajectory is checked against the stored one. |
| `-o out.tsv` is added later, with or without `--dates`        | Nothing is computed; the table is rendered from the stored cells                                                                                           |
| camdl is rebuilt from changed source                          | Every id is new; the old results stay and are listed                                                                                                       |
| One cell fails                                                | Nothing is left for that cell; the invocation record counts one failure                                                                                    |

## Worked example: `camdl fit run`

A fit is a problem, written once and shared, and runs on it, each written in
place and visible while it works. One invocation can cover several problems and
several seeds, so it is a batch of problems and then a batch of runs on each.

The file's keys below are the real ones (`FitConfigWire`,
`fit/config_v2.rs:38-70`). The flags are the real ones too
(`args/mod.rs:743-1000`), grouped here by what they change;
`input_class.rs:159-230` classifies every one.

### 1. Declared inputs: the file and the flags

```rust
#[derive(Deserialize, FileInputs)]
#[serde(deny_unknown_fields)]
pub struct FitToml {
    #[input(group)]    pub model: ModelRef,                        // camdl = "model.camdl"
    #[input(group)]    pub data: Option<DataSpec>,
    #[input(group)]    pub synthetic: Option<SyntheticSpec>,
    #[input(identity)] pub estimate: IndexMap<String, EstimateSpec>,
    #[input(identity)] pub fixed: FixedParams,
    #[input(identity)] pub simplex_groups: Vec<SimplexGroup>,
    #[input(group)]    pub config: FitBackendConfig,               // dt, allow_degenerate_rates, allow_forcing_extrapolation
    #[input(identity)] pub simulate: Option<SimulateSpec>,         // from
    #[input(identity)] pub scenario: Option<String>,
    #[input(identity)] pub enable: Vec<String>,
    #[input(identity)] pub disable: Vec<String>,
    #[input(identity)] pub ic_free: Option<bool>,
    #[input(group)]    pub method: Option<Method>,                 // algorithm, with its backend and settings; starts
    #[input(expands)]  pub fit_seeds: Option<Vec<u64>>,
    #[input(store)]    pub output_dir: Option<String>,
    #[input(recorded)] pub provenance: Option<FitProvenance>,
}

#[derive(Deserialize, FileInputs)]
#[serde(deny_unknown_fields)]
pub struct DataSpec {
    #[input(identity)] pub file: Option<InputFile<Table>>,                       // one wide table for every stream
    #[input(identity)] pub observations: IndexMap<String, InputFile<Table>>,     // or one table per stream
    #[input(identity)] pub holdout_after: Option<TimeSpec>,                      // the partition rule
    #[input(identity)] pub holdout: Option<IndexMap<String, InputFile<Table>>>,  // enters a score, never a fit
    #[input(identity)] pub tables: IndexMap<String, InputFile<Table>>,           // tables bound with external("NAME")
}

#[derive(clap::Args, Inputs)]
pub struct FitRunArgs {
    #[input(identity)]                  pub config: InputFile<FitToml>,

    // Change the problem.
    #[arg(long)] #[input(identity)]     pub allow_forcing_extrapolation: bool,   // sets the [config] key
    #[arg(long)] #[input(identity)]     pub emit_every: Option<EmitEvery>,       // changes generated synthetic data
    #[arg(long)] #[input(expands)]      pub sweep: Vec<SweepAxis>,               // one problem per sweep point

    // Change the run.
    #[arg(long)] #[input(expands)]      pub seed: Option<u64>,                   // an error if the file has fit_seeds
    #[arg(long)] #[input(identity)]     pub starts: Option<StartsSpec>,
    #[arg(long)] #[input(identity)]     pub starts_chains: Option<ChainSubset>,
    #[arg(long)] #[input(identity)]     pub resume: Option<RunRef<FitRun>>,
    #[arg(long)] #[input(identity)]     pub max_tree_depth: Option<u32>,
    #[arg(long)] #[input(identity)]     pub no_nuts: bool,
    #[arg(long)] #[input(identity)]     pub n_trajectories: Option<u32>,
    // ... and the other method flags, each a flat field: --decibans-thresh, --no-dt-check,
    // --dt-check-strict, --dt-check-halvings, --binomial, --cooling-target-iters,
    // --trajectory-warmup, --csmc-sweeps-per-nuts, --diagonal-mass, --no-ancestor-sampling,
    // --no-adapt, --adapt-start, --record-ancestry, --record-prequential.

    // Decide only whether the run may start, or how it runs.
    #[arg(long)] #[input(execution, exercised_by = fixtures::nonconverged_source)]
                                        pub allow_nonconverged_source: bool,
    #[arg(long)] #[input(execution, exercised_by = fixtures::two_chain_fit)]
                                        pub parallel: usize,
    #[arg(long)] #[input(store)]        pub force: bool,                         // conflicts with --resume
    #[arg(long)] #[input(store)]        pub label: Option<String>,
}
```

- **`backend` is a key of `[method]`, and `dt` is in `[config]`.** `fit run` has
  no `--dt` and no `--output-dir`; the store root comes from the file's
  `output_dir`, then the environment, then the default.
- **The method flags are flat fields,** not a group. Each overrides one key of
  `[method]`.
- **`--seed` with `fit_seeds` in the file is refused**
  (`fit/run_config.rs:719-723`). Otherwise the seeds are the file's `fit_seeds`,
  else `--seed`, else 1.
- **The global `--no-licm` changes the compiled model,** so it reaches identity
  through the model's content.

Today's code already has two of the ideas this design relies on. The file is
split into a problem half and an inference half with an exhaustive destructure
(`config_v2.rs:199-212`). The method's identity is subtractive: a new method key
is hashed unless it is named as an exception (`config_v2.rs:1405`). The target
keeps both and moves them into the types.

### 2. The identities

```rust
#[derive(Identity)]
#[identity(tag = "fit_problem", version = 1)]
pub struct FitProblemIdentity {
    pub model: Dep<CompiledModel>,                          // the ordinary compile
    pub training: BTreeMap<String, Hashed<TrainingRows>>,   // the training rows only, per stream
    pub partition: PartitionRule,                           // holdout_after, and the rule's version
    pub tables: BTreeMap<String, Hashed<Table>>,            // external tables
    pub estimate: Estimate,                                 // what is estimated, bounds, priors, transforms
    pub fixed: ResolvedFixed,                               // a sweep point's values land here
    pub simplex_groups: Vec<SimplexGroup>,
    pub settings: FitSettings,                              // dt and the two allow flags, defaults filled in
    pub scenario: ResolvedScenario,
    pub ic_free: bool,
}

/// Everything about a run except how long it is. A resume must match this exactly.
#[derive(Identity)]
#[identity(tag = "fit_chain", version = 1)]
pub struct ChainIdentity {
    pub problem: Dep<FitProblem>,
    pub method: MethodSettings,                             // [method] without its extension length;
                                                            // chains, backend, gate, n_trajectories
    pub perturbation: PerturbationSettings,                 // rw_sd, perturb_only_at_t0 and start, per parameter
    pub jacobian_model: Option<Dep<CompiledModel>>,         // the larger compile; only nl-lbfgs on ODE
    pub seed: ProcessSeed,
    pub starts: StartSource,                                // the prior, a Dep on another run, or an input file
}

#[derive(Identity)]
#[identity(tag = "fit_run", version = 1)]
pub struct FitRunIdentity {
    pub chain: ChainIdentity,
    pub length: u64,                                        // sweeps or iterations, counted once
    pub resumed_from: Option<Dep<FitRun>>,                  // the base run's chain files, as one dependency
}
```

- **Held-out rows are not in the problem.** With `holdout_after` they sit in the
  training file, so the problem hashes the training rows, not the file. The
  per-stream digest of the training rows exists today (`RowPrefix.digest`,
  `fit/training_partition.rs:59-133`); it is recorded and not yet in any
  identity. Held-out tables enter only a score.
- **This re-keys every fit with a holdout.** Today the fit's identity covers the
  whole training file and the held-out tables. Removing them is deliberate.
- **Method settings that sit in `[estimate]` belong to the run.** `rw_sd`,
  `perturb_only_at_t0` and `start` are keys of `[estimate.<parameter>]` in the
  file, and stay there. Their class routes them to `ChainIdentity.perturbation`,
  so two runs that differ only in `rw_sd` share one problem. Today they do not.
- **The resume guard is a key.** A resumed run's `chain.key()` must equal the
  base run's. That replaces today's second key, the run id with the length
  zeroed and the resume dependency removed. Because the ambient identity is in
  every run id but not in `chain.key()`, the base must also come from the same
  build; that is checked separately and matches today's refusal.
- **Two more resume refusals carry over:** a target at or below the base's
  completed length (`fit/mod.rs:1071-1088`), and any method other than PGAS and
  PMMH (`fit/run_config.rs:716`).
- **The compile with the state Jacobian belongs to the run,** since only one
  method on one backend reads it.
- **`n_trajectories` is named in `MethodSettings`.** Today's method payload
  leaves it out and adds it back separately; the target needs it stated once.

How today's fit identity maps onto this:

| Today                                                                                 | Target                                                                                                                    |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `FitDigest.model`, a hash of part of the model                                        | `Dep<CompiledModel>`                                                                                                      |
| `FitDigest.data`, a digest of each whole training file                                | `training`, a digest of the training rows only                                                                            |
| `FitDigest.holdout_data`                                                              | Not in the problem; in `ScoreIdentity`                                                                                    |
| `FitDigest.tables`                                                                    | `tables`                                                                                                                  |
| `FitDigest.fit_toml`, canonical JSON of the problem half minus named keys             | The typed fields: `estimate`, `fixed`, `simplex_groups`, `settings`, `scenario`, `ic_free`                                |
| `FitDigest.engine`                                                                    | The ambient identity                                                                                                      |
| `FitDigest.partition_rule_version`                                                    | `PartitionRule`'s version                                                                                                 |
| `rw_sd`, `perturb_only_at_t0` and `start`, inside `FitDigest.fit_toml`                | `ChainIdentity.perturbation`                                                                                              |
| `[synthetic]` and `--emit-every`, inside `FitDigest.fit_toml`                         | They generate the training rows, which the problem hashes like any data; the generator's settings are recorded as lineage |
| `MethodConfig.config`, a digest of the method payload, trajectory count and gradients | `ChainIdentity.method`; the gradient digest is dropped, since the model's key covers the tables                           |
| `MethodConfig.target_length`                                                          | `FitRunIdentity.length`                                                                                                   |
| `MethodLevel.deps`                                                                    | `starts` and `resumed_from`                                                                                               |
| The seed level                                                                        | `ChainIdentity.seed`                                                                                                      |
| `MethodConfig.obs_block` and `flow_indices`, always empty for a fit                   | Dropped                                                                                                                   |

### 3. Resolution

```rust
#[deny(unused_variables)]
pub fn resolve(
    file: FitTomlDeclared,
    flags: FitRunFlags,                 // every identity and expands flag, taken apart below
    store: &Store,
    info: &RunInfo,
) -> Result<FitPlan> {
    let (model, data, synthetic, estimate, fixed, simplex_groups, config, simulate,
         scenario, enable, disable, ic_free, method, fit_seeds) = file.into_parts();
    let (allow_forcing_extrapolation, emit_every, sweep, seed_flag,
         starts, starts_chains, resume, method_flags) = flags.into_parts();

    let (observations, holdout_after, holdout, tables) = data.resolve()?;   // every field of [data]
    check_holdout_files(&holdout, &observations)?;           // today's check, kept
    release(holdout, "held-out tables enter a score, never a fit");

    // Anchored times and bound tables are written into the model before it is keyed.
    let compiled = compile(model, simulate, &observations, &tables)?;
    let scenario = ResolvedScenario::new(scenario, enable, disable, &compiled)?;
    let settings = FitSettings::merge(config, allow_forcing_extrapolation)?;      // the flag wins
    let method   = Method::merge(method.ok_or(Error::NoMethod)?, method_flags)?;  // the flags win
    let seeds    = Seeds::exactly_one_source(fit_seeds, seed_flag)?;              // both given: refused
    let starts   = StartSource::resolve(starts, starts_chains, store)?;
    let (estimate, perturbation) = Estimate::split(estimate)?;                    // rw_sd, start go to the run

    // Only nl-lbfgs on the ODE backend needs the compile with the state Jacobian.
    let jacobian_model = method.needs_state_jacobian()
        .then(|| {
            let larger = compile_with_state_jacobian(&compiled)?;    // checked to be the same model
            store.put::<CompiledModel>(larger.model(), info)?.dep("model.ir.json")
        })
        .transpose()?;

    FitPlan::new(
        store.put::<CompiledModel>(compiled.model(), info)?.dep("model.ir.json")?,
        Partitioned::new(observations, holdout_after)?,      // training rows and the rule
        tables, estimate, fixed, simplex_groups, settings, scenario,
        ic_free.unwrap_or(false),
        synthetic, emit_every, sweep,                        // one problem per sweep point or dataset
        method, perturbation, jacobian_model, seeds, starts, resume,
    )
}
```

If `fit.toml` gains a key, `file.into_parts()` returns one more value and this
stops compiling. The same holds for a new flag.

### 4. The command

```rust
pub fn fit_run(args: FitRunArgs, ambient: Build<Engine>) -> Result<()> {
    let (declared, execution, (), args_request, given) = args.split().into_parts();
    let (config, flags) = declared.into_parts();                     // the file, and every flag

    // Every part of the file. fit.toml has no execution keys, so that part is the unit type.
    let (file, (), file_request, source) = config.parse()?.into_parts();

    let request = StoreRequest::merge(args_request, file_request);   // label, force, output_dir
    let reuse   = request.reuse();
    let store   = Store::open(request.root()?, ambient)?;
    let info    = RunInfo::new(given, &execution, request);

    let plan = resolve(file, flags, &store, &info)?;

    // A batch of problems: one, or one per sweep point or synthetic dataset.
    let mut problems = store.begin_many::<FitProblem>(&plan.problems(), &info, Reuse::IfCompleted)?;
    for begun in problems.members() {
        let problem = match begun {
            Begin::Reused(stored)   => stored,
            Begin::Busy(owner)      => return Err(Error::InProgress(owner)),
            Begin::Claimed(mut run) => { write_problem(&mut run)?; run.finish()? }
        };

        // A batch of runs on it: one per seed.
        let mut runs = store.begin_many::<FitRun>(&plan.runs_on(&problem)?, &info, reuse)?;
        for begun in runs.members() {
            if let Begin::Claimed(mut run) = begun {
                source.write(&run.provenance_file("fit.toml")?)?;    // the exact file this run used
                match engine::fit(&mut run, &execution) {
                    Ok(())   => { run.finish()?; }
                    Err(err) => run.fail("fitting", &err.to_string())?,
                }
            }
        }
    }
    Ok(())
}
```

`write_problem` writes the training rows, the tables and the resolved settings
as result files, and the model source as a required provenance file. A problem
holds no `fit.toml`, because that file also carries the method, which differs
between runs.

Inside the engine, upstream content is loaded through the run, and progress is
reported through it.

```rust
pub fn fit(run: &mut Run<FitRun>, execution: &FitExecution) -> Result<()> {
    let identity = run.identity().clone();
    let problem  = Problem::read(run, &identity.chain.problem)?;     // opens the problem's files through the run
    // nl-lbfgs on the ODE backend runs on the larger compile; every other method on the problem's.
    let model = run.load(identity.chain.jacobian_model.as_ref().unwrap_or(&problem.model))?;

    let trace = run.result_file("chain_0/trace.tsv")?;
    for iteration in 0..identity.length {
        let step = method_step(&model, &problem, &identity, execution)?;
        step.write_row(&trace)?;
        run.progress(&FitProgress { iteration, of: identity.length, loglik: step.loglik })?;
    }
    run.event(&FitEvent::ChainFinished { chain: 0 })?;

    write_summary(&run.result_file("fit_state.toml")?)
}
```

### 5. What a watcher sees

| Moment                        | Status in the record                 | In the folder                                                       | `camdl status` shows                             |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------------------- | ------------------------------------------------ |
| Just after the claim          | Running                              | Both records, the lock, the run's `fit.toml`                        | Running, alive                                   |
| While fitting                 | Running                              | Plus the snapshot, the event log, and traces that grow              | Iteration 12 of 50 and the latest log-likelihood |
| The engine returns an error   | Failed, with operation and reason    | Everything stays                                                    | Failed, with the reason                          |
| The process is killed         | Still running                        | Everything stays; the owner's beat stops                            | Interrupted                                      |
| The same command is run again | Running                              | The earlier attempt's record and attempt files under `.attempts/1/` | Running, second attempt                          |
| `finish`                      | Completed, with the manifest; sealed | The result files, now listed                                        | Completed                                        |

### 6. What lands in the store

```text
results/
  compiled_model/
    seir-81cc03a1f2d4/model.ir.json         # the ordinary compile
    seir-2b90f1c47e03/model.ir.json         # with the state Jacobian; only if nl-lbfgs on ODE ran
  fit/
    seir-2696ab0a41c7/                      # FitProblem: written once
      identity.json  run.json
      problem.toml  inputs/cases.tsv        # training rows only
      model.camdl.original                  # provenance file, required
      runs/
        if2-7b21e4c09a3d/                   # FitRun, completed and sealed
          identity.json  run.json
          fit.toml                          # provenance file: what this run was given
          fit_state.toml  mle_params.toml  chain_0/trace.tsv
          events.jsonl  progress.json       # attempt files
        if2-c3d1e88a0f52/                   # FitRun, another seed, running now
          identity.json  run.json  .lock  fit.toml
          events.jsonl  progress.json  chain_0/trace.tsv
        pmmh-19ab44d0e7c6/                  # FitRun, failed once and retried
          identity.json  run.json  fit.toml
          .attempts/1/
```

- **Each run keeps the exact `fit.toml` it was given.** A later run with other
  method settings cannot make an earlier one look different.
- **Other commands that take a fit** take either a configuration or a result.
  `pfilter --fit fit.toml` names a configuration: the file is an input, merged
  with the flags, resolved to a problem that is created if absent.
  `--fit @label` names a stored result.
- **A resume** is a new `FitRun` with a longer `length`, the same `chain`, and
  the base run's chain files in `resumed_from`. A directory with no record
  cannot be resumed; today one can, keyed by its contents.
- **A fit made by an earlier build can still be predicted from.** The
  prediction's dependency records the fit's build.

## Features that build on the store

These camdl features each add an artifact, a by-product or a link, and none
needs anything new from the library. Their design belongs to the
unified-prediction proposal; this section shows only how they sit on the store.

### Predictions

Fixed-point predictions are allowed (D101). A prediction's parameters come from
a fit's posterior or from a fixed point, so its identity holds an enum. A
fixed-point prediction is labelled in its record and in every rendering as
carrying no parameter uncertainty.

```rust
#[derive(Identity)]
#[identity(tag = "prediction", version = 1)]
pub struct PredictIdentity {
    pub source: ParameterSource,
    pub data: BTreeMap<String, Dep<DataTable>>,     // what the filter conditions on; never held-out outcomes
    pub quantities: Option<Dep<Quantities>>,        // separate content, used only here
    pub options: PredictOptions,                    // mode, origins, horizon, draws, seed, scenarios
}

#[derive(Identity)]
#[identity(tag = "parameter_source", version = 1)]
pub enum ParameterSource {
    Posterior  { draws: Dep<FitRun>, problem: Dep<FitProblem> },
    FixedPoint {
        model: Dep<CompiledModel>,
        params: ResolvedParams,
        trained_through: Option<TrainingCutoff>,
    },
}
```

- **A posterior prediction sits inside the run whose draws it used.** A
  fixed-point prediction sits on its own under `prediction/`.
- **The fit may come from an earlier build.** The dependency records which.
- **One overlap is accepted.** A fixed-point free-forward prediction computes
  nearly what `simulate --obs` computes, under a different id, so that one table
  can cover every model.
- **`fit table` is an orchestrator whose child is a producer.** It stores
  nothing itself, but the predictions it asks for are stored results.

### Scores

Held-out outcomes enter only a score (D81). They never change a prediction's run
id, which is why they are in neither the fit problem nor the prediction.

```rust
#[derive(Identity)]
#[identity(tag = "score", version = 1)]
pub struct ScoreIdentity {
    pub prediction: Dep<Prediction>,
    pub holdout: BTreeMap<String, Dep<DataTable>>,
    pub rule: ScoringRule,
}
```

`compare` then reads stored scores and does not recompute them.

### Profiles

A profile point is the same result whichever grid holds it.

- **Its seed is derived from its focal values, its start's values and the user's
  seed.** Today it comes from the point's position in the grid
  (`profile.rs:1963`), so the same point in two grids is two different
  computations. Seeds change, which is accepted.
- **Its identity holds its start's values, not the start's index.** Starts are
  drawn once for a whole run, so the same index can mean different values in two
  runs.
- **Its identity has no grid.** Growing a grid reuses the points already stored.
- **The profile table is a rendering** over the invocation's points, as the
  combined simulation table is. There is no stored profile result today either;
  the curve is rebuilt from the stored points.

### By-products of a filter run

Every file `pfilter` can save is a by-product of the filter run, and the
`--save-*`, `--trace` and `--pf-health` paths are exports of them. Today the
filter runs again on a cache hit just to write those files, and the stored
result holds only the log-likelihood.

- **All of them are stored,** the per-particle ones included (`--save-paths`,
  `--save-filtering`, `--save-prequential`). A later export is then a copy.
- **Sizes in practice are small.** In the London measles project, 50 saved paths
  are 3.5 to 9.4 MB per run, a final state is about 110 KB, and filtered streams
  are about 150 KB.
- **Thinning is a later feature.** It is not part of this design.
- **`--no-store` still works.** The run uses a temporary store, the exports are
  written, and the store is deleted.
- **`pfilter -o` is a rendering,** since it rounds the log-likelihood for
  display.
- **`--save-final-state` with `holdout_after` changes what is computed,**
  because it cuts the data to the training rows. It is an identity input, as it
  is today.

## The path from today's store

Most of the work is here, not in the target. Today's lifecycle carries over
nearly unchanged; identity, shared folders and readers are what move. Facts
about today come from the inventory.

### What maps to what

| Today                                                                             | Target                                                                                                            |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| A hand-assembled list of levels per kind                                          | The fields of that kind's identity struct                                                                         |
| `Keyed(Level)`, the `levels(..)` lists, 13 per-command overrides                  | The `identity` class, and `into_parts` in `resolve`                                                               |
| `Expands`, `Child`                                                                | `expands`, `adds`, each with the generated test camdl has today                                                   |
| Free-form loading of files into the run                                           | Typed files and typed resolution steps                                                                            |
| Canonical JSON with a hand-kept list of excluded keys                             | A class on every key of a settings file                                                                           |
| Two hashing regimes, one of which omits defaults                                  | One `Identity` derive: names hashed, every field hashed, defaults resolved first                                  |
| A model digest that hashes part of the compiled model                             | Content artifacts: the compiled model with its derivative tables, and the quantities, each keyed by its own bytes |
| Level labels: scenario name, model file stem, `seed_N`                            | Labels in the invocation record; a folder keeps its first writer's name                                           |
| Engine identity folded into the model level                                       | Ambient identity in every run id                                                                                  |
| `begin_resolved_write` with `Atomic` or `Streaming`                               | `begin`, with the artifact's `Atomic` or `InPlace` mode                                                           |
| `.lock` with process id and host; `progress.json` on a timer                      | An owner with a start time and one beat; progress and events for every kind                                       |
| Per-invocation records under a profile folder                                     | Invocation records, for every command run                                                                         |
| Failed leaves, `.attempts/`, `.quarantine/`, `.staging/`, `--force`               | Kept as they are                                                                                                  |
| Reserved file names outside the manifest                                          | Attempt files                                                                                                     |
| A label overwritten in `run.json` on a cache hit                                  | Naming events appended to the record                                                                              |
| `index.json`, written by `dev reindex` and on a prefix miss                       | Per-owner index logs, appended on every finish                                                                    |
| Deleting results by hand                                                          | Unchanged                                                                                                         |
| The fit folder: a level's directory, with files every run rewrites                | The `FitProblem` artifact, written once; each run keeps its own `fit.toml`                                        |
| Held-out tables in the fit level                                                  | In a score only                                                                                                   |
| `predict/` under the fit folder, with no record                                   | `Prediction` results                                                                                              |
| `obs/` and `event_log/` children, declared in the parent's record afterwards      | By-products, each with its own records; the parent's record is not touched                                        |
| `-o`, `--save-*` and `--draws-out` recomputing on a cache hit                     | Exports: copies or renderings of stored files                                                                     |
| `--dates` and `--format` applied while the run writes its `-o` file               | Rendering options on the export                                                                                   |
| Synthetic data in a fit folder keyed without data                                 | The generated rows are the problem's training rows, hashed by content                                             |
| A profile point with the grid in its identity and its seed from its grid position | A point without the grid, seeded from its own values                                                              |
| `ensemble.tsv`, stored, with a position column and labels outside its identity    | Not stored: the same table, rendered from the cells and the invocation record                                     |
| `store.augment`, adding a file to a completed result                              | Removed. `reactive_log.tsv`, its one remaining use, is an ordinary result file                                    |
| `find_leaf` by level hash; the `~` collision suffix                               | Found by run id; a suffix only when 12 digits collide                                                             |
| Filters on level labels, the seed parsed from `seed_N`, the fourth ancestor       | The identity record and the dependencies                                                                          |
| `@label` resolved in the default root                                             | Every handle resolved in the command's one root                                                                   |
| `STORE_LAYOUT` and `dev migrate`                                                  | Kept for new-format stores; it rewrites no completed record                                                       |
| A `fit.toml` handle matched through a second hash appended to records             | Refused where one run is needed, with the runs that file produced listed from the invocation records              |

### Run ids

Every run id changes, and there is no attempt to preserve any. The encoding
changes, the build moves into every id, paths leave the fit identity, and the
grid leaves the profile point.

### Existing stores

- **Old stores are not read.** Once every command has moved, the binary refuses
  an old store root. There is no read-only mode for the old layout.
- **Existing fits are abandoned.** A fit in an old store cannot be resumed,
  predicted from or used as a starting point by the new binary. Its files stay
  on disk and stay readable by hand and by the binary that wrote them. A fit
  that is still needed is run again.
- **Nothing copies old results into the new store.** camdl is alpha software,
  and running a fit again is the supported path.

### Folders that change shape

- **One folder per result,** where today there is one directory per level.
- **Compiled models get folders of their own,** as content artifacts.
- **Fit folders** become a problem folder with `runs/` inside. The `[method]`
  table leaves the shared files, each run keeps its own `fit.toml`, and held-out
  tables are no longer stored with the problem.
- **`predict/`** moves from the fit folder to the run whose draws it used, or to
  `prediction/` for a fixed point.
- **`obs/` and `event_log/`** keep their place inside the simulation and gain
  records of their own.
- **Profile folders** become one folder per point. Their per-invocation records
  become invocation records.
- **`ensembles/` goes away.** The combined table is rendered on request and
  keeps its columns.

### Dependency edges the index must hold

These are the edges in camdl today, plus two that are coming. The index also
holds by-products under their upstream.

| Dependent                                             | Upstream                              | In the target                             |
| ----------------------------------------------------- | ------------------------------------- | ----------------------------------------- |
| A simulation, a fit problem, a fixed-point prediction | The compiled model                    | `Dep` on a content artifact               |
| A fit run started from another fit                    | The source run's start values         | `Dep` on part of a file                   |
| A resumed fit run                                     | The base run's chain files            | `Dep`                                     |
| An `nl-lbfgs` fit run on the ODE backend              | The compile with the state Jacobian   | `Dep` on a content artifact               |
| A prediction that reports quantities                  | The model's quantities                | `Dep` on a content artifact               |
| `pfilter`, `profile`, `survey` given a stored fit     | The fit problem                       | `Dep`                                     |
| `simulate --fit`                                      | A fit run's draws or parameters       | Lineage; the values themselves are hashed |
| A prediction                                          | A fit run's draws and the fit problem | `Dep`                                     |
| A score                                               | A prediction                          | `Dep`                                     |

- **Not edges, because they are renderings:** the combined simulation table and
  the profile table. Their members are listed in the invocation record.
- **Not edges, because they store nothing:** `compare`, `mre` and `lineage`.
  `fit table` stores nothing itself, but its child predictions are stored.
- **Not edges, because the content is hashed as an input:** an initial state
  taken from a fit, and an initial-state file saved by `pfilter`.
- **No longer an edge:** a fit run's survey starts. That dependency was removed
  from the code.

### What readers must change

- **Inside camdl:** `list`, `show`, `cat`, `fit summary`, `fit table`, `compare`
  and handle resolution. Each stops using path depth, level names and label
  parsing, and either takes a completed result or says it is showing an
  unfinished one.
- **Outside camdl:** camdl-viewer and camdl-scope read progress, events and
  result files by path today. They are changed after `fit run` lands on main, by
  the agent that maintains them, from a note saying what moved.
- **Anyone who reads the combined table** keeps its columns: the book's
  `group_by("replicate")` examples and camdl-viewer's ensemble view still work
  on an `-o` file. What changes is where it comes from: `camdl cat @label` or
  `-o`, not a stored `ensemble.tsv`. A single cell's table gains the coordinate
  columns a multi-cell table has.
- **The book:** every path in an example changes.

### Where today's rewrites of a completed result go

Today a completed result is written again in six ways. In the target a completed
record is sealed, so each needs another home.

| Today                                                           | In the target                                                                                                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| The end-of-run event report, appended after the commit          | An attempt file, written before `finish`                                                                                                                     |
| Configuration identities added to a record on a cache hit       | The invocation record of the run that hit, never the result's record                                                                                         |
| Fit-folder rewrites, and the training partition merged in later | The `FitProblem` artifact, written once; the partition is computed when it is made. It depends on the data, `holdout_after` and the model, not on the method |
| `augment`, adding `reactive_log.tsv` to a completed result      | Removed; the file is an ordinary result file, written with the result                                                                                        |
| `output_schema`                                                 | Written at `finish`                                                                                                                                          |
| `migrate` rewriting old leaves                                  | Nothing; old stores are not read                                                                                                                             |

### A model stored by an older build

A dependency may cross builds, but a compiled model is read only by a binary
with the same IR version. Predicting from a fit made before an IR change is
refused, and the error says that the IR version differs and the fit must be run
again.

### Order

During the transition each command uses exactly one store: a command that has
not moved keeps the old code and the old root.

1. The new `store` and `store-derive` crates, with the lifecycle carried over
   and its tests, beside `runid`. They hold no camdl types. The input derive is
   written fresh here; no earlier branch of it is merged.
2. `camdl predict` from fixed points, born on the new store. It needs no fit
   artifact. `fit predict` stays on the old store, untouched.
3. `pfilter` and `survey`. Each is one result with few readers.
4. `simulate`, with observations and the combined table.
5. `fit run` with the problem artifact. Posterior predictions arrive with it,
   and `camdl predict @fit` replaces `fit predict`.
6. `profile`. Its identity changes the most.
7. Scores.
8. `runid` and `runid-derive` are deleted when the last command has moved.

camdl-viewer and camdl-scope are not part of this work. Once `fit run` has
landed on main, the agent that maintains them is told what changed: the paths,
the two progress files, and that results are named by handle.

## Suspected defects in today's code

The reviews turned up four places where today's code may give two different
results under one run id, or may class an input wrongly. None has been
reproduced. Each is a question until its test fails, and each test is written
against today's code, before the command it concerns moves to the new store.

### A profile's results may depend on the order of its `--sweep` flags

- **What.** A profile job's seed is `seed ^ (grid_idx * 1000 + start_idx)`
  (`profile.rs:1963`), and `grid_idx` follows the order the `--sweep` flags were
  given (`profile.rs:1411-1420`). The identity sorts the axes by name first
  (`profile_cas.rs:83-89`). So two commands that differ only in flag order have
  one run id and may have different seeds at the same point.
- **The test.** Run a two-axis profile, then again with the `--sweep` flags
  swapped, into two fresh stores. Require the same run ids and the same
  `mle.toml` at every point.
- **In the new design.** A point's seed comes from its own values, so its
  position cannot matter.
- **Also.** `* 1000` collides once a profile has 1000 or more starts.

### A combined table may differ under one ensemble id

- **What.** The ensemble's identity is a sorted list of its cells
  (`sim_ensemble_cas.rs:168-180`). The table it stores has a position column and
  a scenario-name column that follow the order asked for (`main.rs:3608-3618`).
  So `--scenario a,b` and `--scenario b,a` have one ensemble id and may store
  different bytes. The second run's mismatch is reported as a warning and the
  command still exits 0.
- **The test.** Run both orders into one store. Require one ensemble id, and
  require the second run either to match the stored bytes or to fail.
- **In the new design.** The combined table is not stored, so there is nothing
  to disagree with.

### Saved prequential samples may depend on `--trace`

- **What.** The filter's one-step predictions for `--trace` and its samples for
  `--save-prequential` appear to draw from the same random stream
  (`particle_filter.rs:587`, `:699`). Passing `--trace` would then change the
  samples that `--save-prequential` writes.
- **The test.** Run `pfilter --save-prequential` with and without `--trace`,
  with one seed. Require identical prequential files.
- **In the new design.** A stored by-product must be a function of its own
  identity, so this has to be fixed before `pfilter` moves: each diagnostic gets
  its own stream.

### `--pf-max-substeps` is classed two ways

- **What.** It is an execution input in `pfilter` (`input_class.rs:269`) and an
  identity input in `profile` (`input_class.rs:339`). An execution input must
  not change a result.
- **The test.** Run `pfilter` at two values on a model where the cap binds. If
  the log-likelihood differs, it is an identity input in `pfilter` too.
- **In the new design.** A group shared between commands must mean the same in
  each, so the two classes cannot both stand.

### Documentation found out of date

- The module comment in `runid/src/store.rs` says `pfilter`, `survey` and
  profile points are written atomically; they are written in place.
- `docs/dev/cas-path-shape-contract.md` says the fit's shared record holds the
  label; labels live on runs.
- `docs/dev/proposals/2026-11-14-moving-labels-and-run-level-handles.md` reads
  "Status: Proposal"; it landed in `2788a059b`.

## Decisions

Settled:

- dependencies hash the run id plus a digest and may cross builds;
- the store owns the progress channel;
- labels are minted and sealed, and a label on a batch names the invocation;
- names are labels and never identity;
- the combined table and the profile table are renderings, exports are copies or
  renderings, and a table has the same columns for one cell and for many;
- old stores are not read, existing fits are abandoned, and no `import` command
  is built;
- content is an artifact and there is no content area;
- the model's `description` is provenance, and every compiled model's record
  holds the digest of the whole compiled file and of the source file;
- held-out outcomes enter only a score, and the dates of held-out rows may move
  the training window;
- every particle-filter diagnostic is stored;
- fixed-point predictions are allowed (D101), labelled as carrying no parameter
  uncertainty;
- a damaged completed result is moved to quarantine and computed again, never
  deleted;
- a run that stopped on another host is reclaimed only by hand, with force;
- a profile point with a non-finite log-likelihood is a completed result with a
  null value, as today;
- the library is two new crates built beside `runid`.

- `rw_sd`, `perturb_only_at_t0` and `start` are in the run's identity, not the
  problem's;
- a configuration file never names a run;
- the first version has `verify`; it has no archive, no search by identity
  field, no `diff` and no `watch` command.

- the compiled model is stored whole, derivative tables included, and the IR
  does not change.

No decision is open.

## Future features

None of these is part of this proposal. Each is recorded so the design leaves
room for it.

- **A warning for runs that are alive but not progressing.** A beat shows a
  process exists; a deadlocked fit still beats. `status` would flag a run whose
  progress snapshot has not changed for a long time. It warns and does nothing
  else.
- **Thinning stored diagnostics.** A rule that removes old per-particle files
  when they grow too large.
- **A marker for results that are not decision-ready.** A fixed-point prediction
  carries no parameter uncertainty. A type or a record field would let a report
  refuse to present one as a basis for a decision.
- **Archive, restore and prune.** Removing a result in two steps, the first
  reversible, and refused while other results depend on it. The index already
  holds the dependency edges this needs.
- **Search and comparison.** `find --where <field>=<value>` over identity
  records, and `diff <a> <b>` showing which identity fields differ.
- **A `watch` command** that follows a run's status, progress and events. The
  two files it would read are already specified.
- **Fit variants,** a `fit.toml` derived from another with the link recorded.
- **A stored result computed from several others,** with its members as
  dependencies. No command needs one today.
- **A sandbox for the computation,** so that it can see only its own folder and
  its declared dependencies. On Linux, Landlock looks suitable.
- **An open-file audit on CI,** running each producer under a system-call tracer
  and requiring every file it opens under the store root to be in its record.
- **Browsing an old store from the new binary.** Not planned; old fits are run
  again.

## Appendix: the traits and types in one place

Every trait and type the design names, grouped by the stage a value passes
through. The sections above give the full signatures; this is the map.

```text
arguments, files, environment
   |  #[derive(Inputs)] / #[derive(FileInputs)] / #[derive(EnvInputs)]
   v
Split  ->  Declared | Execution | Exports | Request | Given
   |  resolve(declared)           typed steps; defaults filled in
   v
Identity  --key()-->  Key  --with kind and ambient-->  Id<A>
   |  store.begin(identity)
   v
Begin  =  Reused(Stored<A>) | Busy(Owner) | Claimed(Run<A>)
   |  run.result_file(..), run.also::<B>(), run.finish()
   v
Stored<A>  --dep()-->  Dep<A>  -->  a field of the next result's Identity
   |
   +--export()/render()-->  a file at a path the user gave
```

### Declaring inputs

| Trait or type  | What it is                                                                            | camdl example                                         |
| -------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `Inputs`       | Derived on a clap struct. Gives every field a class and splits the struct by class    | `SimulateArgs`, `FitRunArgs`                          |
| `FileInputs`   | The same for a settings file: every key has a class, an unknown key is a parse error  | `FitToml`, `ParamsToml`                               |
| `EnvInputs`    | The same for environment variables                                                    | `CamdlEnv`                                            |
| `Split<I>`     | The result of splitting: one struct per class                                         | `SimulateDeclared`, `SimulateExecution`               |
| `I::Declared`  | The identity, `expands` and `adds` inputs. Private fields; only `into_parts` opens it | `(model, params, overrides, numerics, scenario, ...)` |
| `I::Execution` | Inputs the engine uses that must not change a result                                  | `parallel`                                            |
| `I::Exports`   | Paths to write to, and how to render                                                  | `-o`, `--dates`                                       |
| `I::Request`   | Requests to the store                                                                 | `--label`, `--force`, the store root                  |
| `Given`        | The inputs as typed, with `argv`, for the invocation record                           |                                                       |
| `InputFile<T>` | A path whose content type is known. The only way a file is read                       | `InputFile<ModelSource>`, `InputFile<Table>`          |
| `Parsed<T>`    | A parsed settings file, taken apart in full like arguments                            | the four parts of `fit.toml`                          |
| `Vary`         | "Another valid value of this type", for the generated identity test                   | a second `dt`, a second model file                    |
| `Labels`       | What a user recognises a result by; never hashed                                      | scenario name, replicate number, the seed typed       |

### Identity and content

| Trait or type        | What it is                                                                                        | camdl example                                         |
| -------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `Identity`           | Derived on a struct that fully determines one result's bytes. `key()` hashes every field          | `SimIdentity`, `FitProblemIdentity`, `FitRunIdentity` |
| `Identity::Record`   | The form written to `identity.json`: the identity with each held content replaced by its key      | `SimIdentity`'s record inside `ObsIdentity`           |
| `Key`                | A 32-byte hash                                                                                    |                                                       |
| `Content`            | A typed value with a byte form; its key is the digest of those bytes                              | `CompiledModelContent`, `TrainingRows`, `Table`       |
| `Hashed<T: Content>` | Content together with its key, computed once                                                      | `Hashed<TrainingRows>` in a fit problem               |
| `Finite`             | An `f64` that is neither NaN nor infinite                                                         | `dt`, a parameter value                               |
| `Engine`, `Build<T>` | The ambient identity: build fingerprint and identity-class environment. `Incomplete` never reuses | `Build::Known(engine)`                                |

### Kinds of stored result

| Trait or type     | What it is                                                                            | camdl example                                           |
| ----------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `Artifact`        | One kind of stored result: its `KIND`, write mode, identity type, folder name, layout | `Sim`, `FitProblem`, `FitRun`, `Prediction`, `Score`    |
| `ContentArtifact` | An artifact whose identity is the digest of its one file. Stored with `Store::put`    | `CompiledModel`, `Quantities`                           |
| `ByProduct`       | An artifact produced by the same computation as another (`type Of`)                   | `Obs` and `EventLog` of `Sim`; saved paths of `Pfilter` |
| `WriteMode`       | `Atomic` (staged, then renamed in) or `InPlace` (visible while it runs)               | `Sim` is atomic; `FitRun` is in place                   |
| `Id<A>`           | A typed run id: the hash of kind, ambient identity and identity key                   |                                                         |
| `Layout`          | Where the folder sits: flat, or inside a parent                                       | a `FitRun` inside its `FitProblem`                      |

### Running and reading

| Trait or type       | What it is                                                                            | camdl example                             |
| ------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------- |
| `Store`             | The only code that reads or writes stored folders. One root per command               | `Store::open(root, ambient)`              |
| `Begin<A>`, `Reuse` | The answer to "may I compute this": reused, busy, or claimed                          |                                           |
| `Run<A>`            | A claimed, running result. Hands out files by role; ends with `finish` or `fail`      | the argument of `engine::simulate`        |
| `Wanted<A>`         | The by-products asked for alongside one run                                           | observations for one cell                 |
| `Stored<A>`         | A completed result. The only type that opens result files or makes a `Dep`            | what `store.resolve("@baseline")` returns |
| `Entry<A>`          | A result in any status, for a command that shows unfinished runs                      | `fit summary` on a running fit            |
| `Dep<A>`            | An identity field naming an upstream file: run id, file name, digest                  | `FitRunIdentity.chain.problem`            |
| `RunRef<A>`         | A handle typed on the command line: a path, an id prefix, `@L`, `@L~n`                | `--resume @scout`                         |
| `Owner`, `Liveness` | The process holding claims, and whether it is alive                                   |                                           |
| `Batch<A>`          | Several claims opened together                                                        | the cells of one `simulate`               |
| `Planned<A>`        | What exists for an identity, before anything is claimed                               | "3 of 4 cells already stored"             |
| `Invocation`        | The durable record of one command run: inputs as given, ordered results, their labels | what `camdl cat @baseline` renders from   |
| `Rendering`         | A pure function from stored files, an invocation and options to an export             | `CombinedTable`, with `--dates`           |
| `NamingEvent`       | One `--label` appended to a result or an invocation                                   |                                           |

### The model split, as the worked example of `Content`

Today one trait impl decides what a model contributes to identity, by listing
every field (`runid/src/ir_hash.rs:1655`):

```rust
impl ContentAddressed for Model {
    fn hash_fields(&self, h: &mut CanonicalHasher) {
        let Model {
            name: _,                      // the file's stem; a label
            version, time_unit, description, compartments, transitions, /* ... */
            per_eval_bindings: _,         // enters through the expressions that use it
            quantities: _, contrasts: _,  // reporting only; never re-key a sim or a fit
        } = self;
        // each named field is hashed; ic_grad only when h.hashes_gradients()
    }
}
```

The target keeps the exhaustive listing and changes what it produces: the parts
themselves, each a `Content` keyed by its own bytes, where today it produces a
digest of some of the fields.

```rust
/// Everything one compile yields, assigned to exactly one part.
pub struct CompiledParts {
    pub model: Hashed<CompiledModelContent>,         // everything the engine runs, derivative tables included
    pub quantities: Hashed<Quantities>,              // the quantities and contrasts blocks
    pub labels: ModelLabels,                         // name
    pub rendering: RenderingDefaults,                // output.format, time_semantics
    pub provenance: ModelProvenance,                 // docs, the declared model, both whole-file digests
}

impl CompiledParts {
    pub fn split(envelope: IrEnvelope) -> Result<Self> {
        // The same exhaustive destructure as today, over the envelope and the model.
        // A new IR field does not compile until it is assigned to a part.
        let IrEnvelope { ir_version, validated_by, model, docs, declared } = envelope;
        let Model { name, description, quantities, contrasts, /* every other field, by name */ .. } = model;
        /* ... */
    }
}
```

`description` goes to provenance with the docs. Every field of `Model` not named
above goes to the model part unchanged.

The `..` in the sketch stands for the full field list; the real function names
every field, as today's does.
