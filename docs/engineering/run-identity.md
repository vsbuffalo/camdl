# Run identity

camdl stores every result under a hash of the inputs that determine it, so that
a repeated run is found instead of recomputed and two different computations
never share an address. This page is for a developer about to change what goes
into that hash: how it is computed today, what each part protects, and which
files, versions and tests a change touches. What a user sees of the system, from
the command line, is
[When does camdl run a model again?](../guide/run-identity.qmd).

## Identity and provenance

Every input to a run falls into one of three kinds, and the kind decides what
the input does to the stored result:

- **Identity** is whatever changes the bytes camdl stores: the model, the
  configuration, the parameters, the scenario and the seed. It is hashed, and
  the hash is where the result is stored and how it is found again. For example,
  `--seed 7` draws a different trajectory, and `--no-flows` stores fewer
  columns, so each makes a new run.
- **Presentation** is a different way of writing the same values, such as how
  times are formatted in the output. It is not hashed, so changing it finds the
  same run.
- **Provenance** is how a run was asked for: the command line, the time, the
  host, a `--label`, the file name the model was read from, the thread count. It
  is recorded beside the result, in the run's `run.json`, and never hashed. For
  example, `--parallel 4` changes how fast the cells are computed, not what they
  contain.

So these two commands are the same run, and the second finds the first's result:

```bash
# illustrative command lines
camdl simulate sir.camdl --params p.toml --seeds 1:3
camdl simulate sir.camdl --params p.toml --seeds 1:3 --parallel 4 -o seeds.tsv --label rerun
```

And these are new runs:

```bash
# illustrative command lines
camdl simulate sir.camdl --params p.toml --seeds 1:3 --no-flows   # other columns stored
camdl simulate sir.camdl --params p.toml --seed 7                 # another trajectory
```

The rest of the page is about the first kind: how identity is computed, and what
keeps an input that belongs to it from being left out.

## The store

`camdl simulate`, `camdl pfilter`, `camdl fit run` and `camdl profile` store
their results in the run store, written by the `store` crate
(`rust/crates/store`) under `./camdl-store` or the root `--store` or
`CAMDL_STORE` names; its design is
`docs/dev/proposals/2026-11-20-run-store-api.md`, and
[Simulate on the run store](#simulate-on-the-run-store) and
[A fit on the run store](#a-fit-on-the-run-store) below describe how a
simulation's and a fit's inputs reach their identities there.

The store is **input-addressed**: a run's address is a hash of the inputs that
determine it, computed before the run, not a hash of the bytes the run produces.
This is the model of the Nix package store, which Dolstra calls the
_extensional_ model, as opposed to a _content-addressed_ (intensional) one in
which an object's address is the hash of the object itself, as in git (E.
Dolstra, _The Purely Functional Software Deployment Model_, PhD thesis, Utrecht
University, 2006, chapters 5 and 6). The code's name for the store, the
"content-addressed store" of `CasSink` and `FsCasStore`, describes the second
model; only the input files are content-addressed here.

An input-addressed store is a memo table for a function, and it is correct only
if the function is:

- **Pure.** The result depends on the hashed inputs and on nothing else: no
  environment variable, global setting, clock, host or file read by its path.
- **Deterministic.** The same inputs give the same bytes, whatever the thread
  count or the order in which cells finish.

Both hold within one build of camdl on one platform; measured, the evaluation
switches (`CAMDL_EVAL_UNRESOLVED`, `CAMDL_NO_CONSTANT_FOLD` and the rest) and
the thread count leave the stored bytes unchanged. The engine is identified by a
build fingerprint, `ENGINE_ID`, which covers uncommitted changes to the source,
so a rebuilt worktree does not share an identity with its clean commit; a build
whose fingerprint cannot be determined never reuses a stored leaf. A fit's
wall-clock timings are kept in per-attempt execution records outside the run's
result files. One limit remains: the platform (the operating system's
mathematics library) is not part of the identity, so the guarantee is stated for
one machine image. How a command-line flag or an environment variable reaches
the hash is described [below](#from-a-command-line-flag-to-the-hash).

The function's inputs are a complete, typed set, and the key is the hash of the
inputs _after resolution_: parameter maps hold the values the engine will use,
with every precedence tier applied; files enter by the digest of their bytes,
never by their path; a scenario enters as its effective delta, every field it
inherits through `compose` included; the seed is the one that drives the
trajectory. A raw path or an unresolved preset name would make the key depend on
how the inputs were spelled, so that two spellings of one computation miss each
other, or one spelling of two computations (an edited table under an unchanged
path) collides.

## The hash function

Every identity is hashed through one encoder, the `store` crate's
`CanonicalHasher`, driven by one trait, `ContentAddressed`. A type implements
`hash_into`, which feeds its canonical bytes into the hasher; `content_hash` is
the root, and the only place `HASH_VERSION` is written. The encoding is designed
so that equal values always produce equal bytes and distinct values never do:
each struct or enum writes the tag it states and its version first; each field
is written after its name, in name order; every string, vector, map and set is
prefixed by its length, so `("ab", "c")` and `("a", "bc")` differ; maps and sets
are written in sorted key order; an enum writes its variant's name. The byte
rules are the run-store proposal's "Keys" section
(`docs/dev/proposals/2026-11-20-run-store-api.md`).

A float enters an identity only as a `store::Finite`: construction refuses `NaN`
and `±Inf`, as a value rather than a panic, and turns `-0.0` into `+0.0`, so
that `--param x=-0` and `--param x=0` are one run. `f64` itself does not
implement `ContentAddressed`, so a float field in an identity does not compile
until its author makes it a `Finite`.

The compiled model enters a result's identity as the digest of the stored
compiled model: the canonical bytes of every field except its name, its
description, its quantities and its contrasts. A separate structural hash of the
model, [`model_ir_hash`](rust:ir::model_hash::model_ir_hash), is what `camdl`
prints and compares models by (`camdl diff`, the check that a model given to
`predict` is the fit's). It leaves out the gradient maps, and its impls for the
IR tree are written by hand in `ir::model_hash`.

## The `Identity` derive

The identity types do not implement `ContentAddressed` by hand. They derive it
with `#[derive(store::Identity)]`, which states the type's tag and version and
hashes every field; there is no attribute that leaves a field out, so a value
that is not identity is not a field of an identity type. The settings of a
simulation cell show the shape:

<!-- decl: rust:cli::sim_store::SimConfig -->

The seed is the one the engine draws from:

<!-- decl: rust:cli::camdl_store::ProcessSeed -->

`seed` is the process seed. It is derived from the user's `--seed` by
[`mix_cell_seed`](rust:cli::util::mix_cell_seed), which mixes in the cell's
position: `--seed 5 --replicates 3` runs three cells with three process seeds,
all from base seed 5. Hashing the base seed would put those three cells, which
hold three different trajectories, under one run id. The cell's folder is
labelled by the seed typed, so the three cells of `--seed 5 --replicates 3` are
`seed_5-…`, `seed_5-r1-…` and `seed_5-r2-…`.

Including by default is the safe direction of error. A field that should have
been excluded but was not costs a cache miss: the run is recomputed, and two
runs that could have shared a leaf do not. A field that should have been
included but was not costs a wrong answer: two runs whose outputs differ share a
`run_id`, and the second is served the first's result with nothing on screen to
say so. The derive makes the first mistake the one that happens when nobody
thinks about a new field. The price is turnover: a field added to a derived
struct contributes bytes even at its default, so it re-keys every stored leaf
that embeds the struct, and that is scheduled with a version bump (below).

## From a command-line flag to the hash

How `simulate`'s flags reach a cell's identity is in
[Simulate on the run store](#simulate-on-the-run-store). The settings below are
shared by every command that runs a model.

`--allow-degenerate-rates` is not a process-wide setting: it is a field of the
compiled model
([`CompiledModel::with_eval_settings`](rust:sim::compiled_model::CompiledModel::with_eval_settings)),
read by every evaluator from the model it evaluates, so two models in one
process can differ. Every command sets it on the models it compiles from the
value it keys.

The forcing-extrapolation policy is a value too:
[`ForcingPolicy`](rust:cli::forcing_extrapolation::ForcingPolicy), built by each
command from its resolved `--allow-forcing-extrapolation` or its fit config's
`[config] allow_forcing_extrapolation`, and holding what the run held and the
refusal it kept while it held a store claim. The command keys its leaf from the
value's `allowed()`, passes the value to every record it writes
([`RecordMeta::new`](rust:cli::resolve::RecordMeta::new)), and installs it for
its duration. Installing is needed because the engine's run notes reach the
command through `sim::run_notes`, whose reporter is one function installed at
start-up: the reporter routes each forcing note into the installed policy
through one routing slot, which is the policy's only process-wide state. With
none installed, a note is refused. `sim::run_notes` says each other distinct
note once per process, but reports a forcing note every time it is recorded (the
engine records one at most once per forcing, side and compiled model), and the
policy says it once per policy, skipping a forcing already in its `held` list.
So every policy installed in one process is told of the forcings its own runs
held, provided its models are compiled after it is installed: the engine's
set-once flag is on the compiled forcing, shared by the model's clones, so a
model that already reported a forcing under one policy does not report it under
the next. Every command installs its policy before compiling, and `compare`
compiles each in-process fit's model after installing that fit's policy. Most
commands install one policy. `camdl predict` alone nests a second, an allowing
one, once it has read the stored fit's config; a refusal under the first has
already ended the command, and the inner guard's drop puts the first back. A
claim-holding `fit run` keeps a refusal for its claim, and each chain polls the
command's policy (held by `Out` from `hold_claim` to `take_claim`, apart from
the claim itself), so the first chain to fail the claim does not hide the
refusal from the others.

A command that replays a stored fit runs under the fit's own settings. The
simulator a fit ran on,
[`FittedSimulator`](rust:cli::fit::fitted_model::FittedSimulator), carries its
`[config] allow_degenerate_rates` and `allow_forcing_extrapolation`, read from
the stored problem's `problem.toml`; `simulate --draws … --fit` folds its own
flags into them
([`FittedSettings::with_flags`](rust:cli::fit::fitted_model::FittedSettings::with_flags):
a flag can only turn a setting on) and puts the result in its resolved
configuration, and `compare` scores each fit under them, passing them to its
`camdl pfilter` child as flags and installing them, one policy per fit, for its
in-process ODE scorer. A forcing held while a fit was scored is part of the
trace it affected
([`PrequentialWarning::ForcingHeld`](rust:score::trace::PrequentialWarning::ForcingHeld)):
`pfilter --save-prequential` writes it from its policy, and the ODE scorer adds
it from the fit's, so `compare` reports it on that fit's rows. When `compare`'s
flag turns the setting on for a fit made without it, it says so, as a `simulate`
replay does
([`FittedSettings::turned_on_note`](rust:cli::fit::fitted_model::FittedSettings::turned_on_note)).

`--no-licm` is a compile setting,
[`CompileSettings`](rust:cli::util::CompileSettings), with `--no-ir-cache`.
`main` resolves both once (with `CAMDL_NO_LICM` and `CAMDL_NO_IR_CACHE`) and
passes the value to each command that compiles, which passes it to every compile
it makes ([`load_model`](rust:cli::util::load_model),
[`resolve_ir_path`](rust:cli::util::resolve_ir_path) and the camdlc
pass-throughs). Nothing reads it process-wide. A load of a model that is already
compiled ([`load_ir`](rust:cli::util::load_ir)) takes no settings and refuses a
`.camdl`, so no compile can run without them.

In `simulate`, two kinds of flag reach a cell's identity through the model and
the parameters rather than its settings. A flag that rewrites the model before
it runs (`--output-every`, which rematerializes the model's output schedule;
`--integrator`; the global `--no-licm`) changes the stored compiled model,
because [`load_run_model`](rust:cli::util::load_run_model) is the one load that
both the engine and the stored compiled model are made from. A parameter flag
(`--param`, `--params`, `--param-vec`, a draw or sweep point) is in the cell's
parameters, which are read off the resolved model the engine runs.

An environment variable that can change a result is keyed. The alternative
evaluation paths (`CAMDL_EVAL_UNRESOLVED`, `CAMDL_NO_BINDING_CACHE`) and every
set `CAMDL_EXPERIMENTAL_<NAME>` are the engine switches. They are fields of the
run store's ambient identity beside the build,
[`Engine`](rust:cli::camdl_store::Engine) `{ build, env }`, whose key names the
build directory every stored result is under, so every command that stores a
result keys them. A prediction's manifests also record `engine_version` and the
switches that were on, `engine_switches`. `compare` scores a chain-binomial fit
through a `camdl pfilter` child, which keys them; its in-process ODE scoring and
`dev eval` store nothing under an address. The switches themselves are fields of
the evaluator settings each compiled model carries
([`EvalSettings`](rust:sim::compiled_model::EvalSettings)), set by
[`eval_settings`](rust:cli::env::eval_settings). `CAMDL_NO_CONSTANT_FOLD` and
`CAMDL_NO_LICM` change the compiled IR. An experimental input is also listed in
the `provenance.experimental` field of `run.json`.

`fit run` resolves its configuration once, before anything runs, into
[`ResolvedFitConfig`](rust:cli::fit::run_config::ResolvedFitConfig): the file
with every flag applied, the compiled model, the engine identity, the seeds, the
sweep and the chain-starts rule. The problem and run identities are built from
that value ([A fit on the run store](#a-fit-on-the-run-store)); the command line
itself is read only before it, so no argument reaches a runner except through
the resolved value, the execution settings (`--parallel`) or the request to the
store (`--force`, `--label`). The fit file is read once; each run keeps that
text byte for byte as its `fit.toml`, and its `fit_toml_hash` is taken from it.
Neither enters an identity.

`pfilter` and `profile` resolve their configurations the same way:
[`ResolvedPfilterConfig`](rust:cli::pfilter_config::ResolvedPfilterConfig) and
[`ResolvedProfileConfig`](rust:cli::profile_config::ResolvedProfileConfig), each
built by a `resolve` that destructures the command's arguments field by field
(so an argument added to the command does not compile until it is placed), with
the outputs written outside the store, the execution settings and the label
apart. The run is a function of those values, and the arguments are not in its
scope. Each `--fit` takes a fit file, resolved in process, or a stored fit:
`@label`, a run-id or problem-id prefix, or a stored folder, read through
`fit_store::read_fit`, which gives the stored compile, the configuration that
`problem.toml` (and, for a run, `method.toml`) parses to, and the checked bytes
of each stored input. A fit made with `--allow-forcing-extrapolation` on its
command line is so read under that override, and a `fit run --sweep` point at
its own `[fixed]` values. The fit is read once, and its
`[config] allow_degenerate_rates` and `allow_forcing_extrapolation` are folded
with the flags in resolution. A `pfilter` run's identity holds a dependency on
the stored problem when `--fit` names one. A profile of a stored fit keys the
text of `problem.toml`, followed by the run's `method.toml` when a run is named;
a profile of a fit file keys the file's text. Every input of each is classified
(`input_class::PFILTER`, `PROFILE`); `pfilter`'s are checked by
`every_pfilter_input_keeps_its_class` and `profile`'s by
`rust/crates/cli/tests/profile_input_classes.rs`.

`fit run` routes its method flags through one struct,
[`CliMethodOverrides`](rust:cli::fit::config_v2::CliMethodOverrides), which
[`Algorithm::apply_cli_overrides`](rust:cli::fit::config_v2::Algorithm::apply_cli_overrides)
writes into the method before its identity is taken. Its `passed` method
destructures the struct without `..`, so a field added to it does not compile
until it names an [`OverrideFlag`](rust:cli::fit::config_v2::OverrideFlag), and
`every_cli_sampler_flag_changes_the_method_identity` in `config_v2.rs` runs each
flag through the identity. The copy from the clap struct into
`CliMethodOverrides`, in `fit::run_config::resolve`, is by hand, as is each flag
that writes into the fit configuration instead (`--allow-forcing-extrapolation`
sets `[config] allow_forcing_extrapolation`).

So when you add a `fit run` or `profile` flag that changes what a run computes
or stores, carry it through each step above, add a case to the differential
test, and write the end-to-end test that runs camdl with and without the flag
and finds two leaves. A `simulate` flag is declared with its class instead
([below](#simulate-on-the-run-store)).

## Testing a classification

Calling an input identity, presentation or provenance is a prediction about what
happens when that input alone is changed, so each classification can be tested.
Run camdl twice into two fresh stores, changing only that input, and compare the
stored results (the trajectory, the observation table; not `run.json`, which
records the time and the command line and always differs). The rule is that
**the stored bytes change exactly when the `run_id` changes**, and it can fail
two ways:

- **The bytes change and the `run_id` does not.** The input belongs in identity
  and is missing from the hash, so the store would serve one result in place of
  the other. This is the failure the flag path above leaves open, and the
  dangerous one.
- **The `run_id` changes and the bytes do not.** The input is hashed but changes
  nothing. Runs that could have been reused are computed again; no result is
  wrong.

`fit run`'s inputs are tested against this rule in
`rust/crates/cli/tests/fit_run_input_classes.rs`, against the `fit`, `method`
and `seed` levels, with a `removed` case for each old spelling it refuses, and
`profile`'s in `rust/crates/cli/tests/profile_input_classes.rs`. `simulate`'s
and `pfilter`'s are tested by the run store's class checks
([below](#simulate-on-the-run-store)). The other commands' flags are not yet
tested this way. These are regression evidence on small fixtures, not proof that
no model behaves otherwise.

## Simulate on the run store

`camdl simulate` stores each cell as one `sim` result of the run store. Its
inputs are declared on the argument types with `#[derive(store::Inputs)]`:
`SimulateArgs`, the groups it flattens, and `SimulateInvocation`, which adds the
global options. Every field carries `#[input(..)]` with one class, and the
derive refuses a field with none:

| Class       | What the input may do                         | What `store::check_input_classes` requires of it                                 |
| ----------- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| `identity`  | change a result; hashed into the run id       | a second value changes a run id                                                  |
| `expands`   | decide which cells a batch has (`--seeds`)    | the cells of the smaller set keep their run ids in the larger, which adds a cell |
| `adds`      | ask for a by-product (`--obs`, `--event-log`) | the cell's run id is unchanged, and a by-product is stored                       |
| `execution` | change how a cell is computed (`--parallel`)  | two values give the same stored files, run ids and exports                       |
| `export`    | name a file to write (`-o`)                   | an export is written, and made again from the stored files it is the same bytes  |
| `render`    | change how an export is written (`--dates`)   | a second value changes no stored file                                            |
| `store`     | a request to the store (`--label`, `--force`) | nothing                                                                          |
| `recorded`  | kept on record, read by nothing               | nothing                                                                          |
| `skip`      | change only what is printed (`--dry-run`)     | nothing                                                                          |

The derive splits the arguments by class into one generated struct per
destination. The part that reaches identity (the `identity`, `expands` and
`adds` fields) has private fields and one `into_parts` that returns them all, so
a new input breaks every caller until it is handled.
`every_simulate_input_keeps_its_class`, in
`rust/crates/cli/src/sim_store/tests.rs`, runs the release binary into fresh
stores once per input and checks what its class predicts;
`every_simulate_argument_is_declared` checks the declarations against every
argument clap accepts.

A cell's identity is `SimIdentity` (`rust/crates/cli/src/sim_store.rs`): a
dependency on the stored compiled model, the settings (`SimConfig`: backend,
`dt`, the cell's window, a restored `--init-state` row, the two evaluator flags,
the column view), every parameter's value after all sources, the interventions
the cell's scenario keeps, and the process seed. `simulate` builds it per cell
from [`ResolvedCell::resolve`](rust:cli::sim_config::ResolvedCell), the value
the engine then runs, and refuses a cell whose backend, step, evaluator settings
or sampler differ from the run's, or that records extra snapshot times. Before
the engine runs, the cell checks that the model it is about to run is the stored
compiled model with the cell's parameters, interventions and window applied, by
digest. The run id hashes the kind, the ambient key (the build and the engine
switches) and the identity's key, so a rebuilt camdl never reuses a cell another
build stored. The compiled model is keyed as compiled and rewritten from data,
before any cell's values are applied, and leaves out the `quantities` and
`contrasts` blocks, which only the `sim_quantities` by-product reads.

An artifact that only some invocations ask for is a by-product, stored inside
the cell at an identity of its own, and never one of the cell's files: the
observations (`obs`, whose identity adds the observation blocks, the
`--emit-every` cadence, the file format, the streams and the observation seed),
the event log (`event_log`, whose identity is the cell's alone, since the
recorder draws nothing) and the generated quantities (`sim_quantities`, which
adds the stored quantities block and the observation seed). Asking for one never
changes the cell's id or its record. A rerun that asks for a by-product the
store lacks simulates the cell again: the trajectory is checked against the
stored one, and only the by-product is stored. Since cells are seeded and
deterministic, a mismatch means an input that changes the trajectory is missing
from the identity, or the engine is not deterministic.

A failure fixed by the cell's identity, the engine stopping with an error, is a
completed result holding `failure.toml`; a failure a rerun could change leaves
no folder. No combined table is stored: `-o`, `--stdout` and `camdl cat @L`
render it from the cells the invocation's record lists, in order.

## A fit on the run store

`camdl fit run` stores two kinds of result. A `FitProblem` (kind `fit`) holds
what a fit is fitted to and is written once; a `FitRun` (kind `fit_run`) is one
run of one method at one seed, stored inside its problem at `runs/` and written
in place while it works. The layout and the files each holds are the run spec's
§2.2. The inputs of `fit run` are declared with `#[derive(store::Inputs)]` on
`FitRunInvocation` and the groups it flattens (`FitProblemFlags`, `FitRunFlags`,
`FitMethodFlags`), in the classes of
[Simulate on the run store](#simulate-on-the-run-store).
`every_fit_run_input_keeps_its_class`, in
`rust/crates/cli/src/fit_command/tests.rs`, runs the release binary into fresh
stores once per input and checks what its class predicts;
`every_fit_run_argument_is_declared` checks the declarations against every
argument clap accepts.

A problem's identity is `FitProblemIdentity`
(`rust/crates/cli/src/fit_store.rs`): a dependency on the stored compiled model,
each data file's training lines, each stream's held-out dates without their
values, the partition rule with its version, the `[data.tables]` files, the
estimated set with its bounds and priors, the resolved `[fixed]` values,
`simplex_groups`, the settings (`dt` and the two evaluator flags), the scenario
request, `ic_free`, and for a `[synthetic]` problem what generated its data
(`SyntheticOrigin`: the true parameters file's bytes, the dataset's seed and the
block's other settings), so each generated dataset is a problem of its own.
`fit_store::resolve_problem` builds it from a `config_v2::Problem` it
destructures with every field named, so a key added to the fit file does not
compile until it is placed. A run's identity is `FitRunIdentity`: a
`ChainIdentity` (the problem, the method's settings without its length, the
per-parameter `rw_sd`, `perturb_only_at_t0` and `start`, the compile with the
state Jacobian when `nl-lbfgs` runs on the ODE backend, the process seed and the
chain starts), the length, and for a resume a dependency on the base run's
carried chain files. A resumed run's `chain.key()` must equal its base's, and
each chain's `resume_state.bin` records it. The run id hashes the kind, the
ambient key (the build and the engine switches) and the identity's key, so a
rebuilt camdl never reuses a fit another build stored.

A prediction (kind `prediction`) is stored inside the run whose draws it uses,
at `predict/`. Its identity is the run and its problem, the stored data the
filter conditions on, and the predict options as canonical JSON, a new option
included unless it is named as left out.

## Versions

A change that alters the bytes being hashed re-keys some set of stored leaves.
Which set is chosen by which version moves, and each move is deliberate:

| Version                                                                               | Folded into                       | A bump re-keys                                                           |
| ------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------ |
| a type's `#[identity(version = N)]`                                                   | that type's hash, after its tag   | every result whose identity embeds the type                              |
| the store's `HASH_VERSION`                                                            | every root hash and every run id  | every result of every kind                                               |
| `ir/VERSION`                                                                          | the stored compiled model's bytes | every result with a model                                                |
| the engine identity (`ENGINE_ID`, the build fingerprint, and the engine switches)     | the ambient key of the run store  | every result in the run store, whenever camdl is built from other source |
| [`PARTITION_RULE_VERSION`](rust:cli::fit::training_partition::PARTITION_RULE_VERSION) | the problem's `partition`         | every fit                                                                |

The store's `RECORD_FORMAT` versions the `run.json` schema without entering any
hash. `SimConfig`'s `#[identity(..)]` above shows the per-type version in use:
each bump records a change to the field set or to the bytes a result holds.

## Where a change lands

| The change                                               | Edit                                                                                                                        | Version                                             | Pinned by                                                                                                                          |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| A `simulate` flag that changes the stored trajectory     | an `#[input(identity)]` field of `SimulateArgs`, carried into a field of `sim_store::SimConfig` (or lowered into the model) | the `version` of `SimConfig`'s `#[identity(..)]`    | `every_simulate_input_keeps_its_class`, with a `vary` fixture for the field                                                        |
| A key in the fit configuration                           | a field of `config_v2::Problem`, placed by `fit_store::resolve_problem` in a field of the problem's or the run's identity   | the `version` of that struct's `#[identity(..)]`    | `every_fit_run_input_keeps_its_class`, with a `vary` fixture for the field                                                         |
| A `fit run` flag that changes a method's output          | a field of `FitMethodFlags`, `CliMethodOverrides`, an `OverrideFlag` row, and the copy in `fit::run_config::resolve`        | none: the method level hashes the method as written | `every_cli_sampler_flag_changes_the_method_identity`                                                                               |
| A field that is displayed but does not change the output | not a field of an identity type: record it only as provenance in `run.json` or a provenance file                            | none                                                | a case in `differential_presentation_inputs_are_inert`                                                                             |
| A float policy, or any other rule of the encoding        | the store's `hash` and `identity` modules                                                                                   | the store's `HASH_VERSION`                          | the store crate's tests                                                                                                            |
| A rule of the model's structural hash                    | `ir::model_hash`                                                                                                            | its `HASH_VERSION`                                  | `rust/crates/ir/tests/model_hash_pins.rs`; the golden model-hash table                                                             |
| The IR schema                                            | the atomic OCaml, Rust and golden update (`.claude/skills/golden-update`)                                                   | `ir/VERSION`                                        | `rust/crates/ir/tests/golden_model_hashes.rs`, regenerated by `make update-golden-model-hashes` only when the re-key is the change |

The golden model-hash table pins absolute values.
`rust/crates/ir/tests/fixtures/golden_model_hashes.tsv` holds the model hash of
every compiled golden. The run ids of each kind on the run store are pinned by
that command's own tests. A row that moves is a re-key, and it is the subject of
its commit, never a side effect of it.
