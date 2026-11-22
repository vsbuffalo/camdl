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

camdl has two stores. `camdl simulate`, `camdl pfilter` and `camdl fit run`
store their results in the run store, written by the `store` crate
(`rust/crates/store`) under `./camdl-store` or the root `--store` or
`CAMDL_STORE` names; its design is
`docs/dev/proposals/2026-11-20-run-store-api.md`, and
[Simulate on the run store](#simulate-on-the-run-store) and
[A fit on the run store](#a-fit-on-the-run-store) below describe how a
simulation's and a fit's inputs reach their identities there. Profiles store
theirs under `results/`, written by the `runid` crate. The sections from here to
that one describe the `results/` store; what they say about input addressing
holds for both.

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

The inputs are not hashed as one blob. For a profile point they are split into
levels, each a disjoint slice: the profile, the point, the method, the seed and
the start. Each level is hashed on its own, and the leaf's address is
[`run_id`](rust:runid::kind::run_id) over the artifact kind and the level hashes
in order:

```text
run_id = SHA-256( HASH_VERSION ++ kind index ++ level count ++ level hashes in path order )
```

The kind enters as a fixed-width index of
[`ArtifactKind`](rust:runid::kind::ArtifactKind) and the level list carries its
length, so two kinds with equal level sequences cannot alias, and neither can a
shorter sequence whose bytes happen to concatenate to a longer one.

The path on disk factors the same tuple:
[`store_path`](rust:runid::layout::store_path) writes one directory per level,
named `{label}-{hash8}`, under the kind's partition (`profiles/`). The label is
provenance, the eight hex characters are the first four bytes of the level's
hash, and the full hashes are in `run.json`. Grouping falls out of the tree
without any index: the seeds of one method are sibling directories under one
method directory. A lookup finds a leaf by its hashes alone
([`find_leaf`](rust:runid::store::find_leaf)), so a renamed file, whose label
changes, still finds the result its hashes name.

A leaf found at the expected path is reused only after
[`FsCasStore::lookup`](rust:runid::store::FsCasStore::lookup) checks it: the
recorded `run_id` must equal the expected one (a different one is a short-hash
[`Collision`](rust:runid::store::Lookup), resolved by suffixing the directory,
never by overwriting), the record's hash and format versions must be current,
its status `completed`, and its files exactly the ones the record lists. A
profile's scan of its points makes the reuse decision: a point is reused only
when `lookup` returns a hit, and only by a build that
[`engine_write_policy`](rust:cli::resolve::engine_write_policy) allows to reuse.
A build whose engine identity is incomplete reuses nothing; it quarantines each
stored point and computes it again.

## The hash function

Every hash in the store goes through one encoder,
[`CanonicalHasher`](rust:runid::hash::CanonicalHasher), driven by one trait,
[`ContentAddressed`](rust:runid::hash::ContentAddressed). A type implements
`hash_into`, which feeds its canonical bytes into the hasher;
[`content_hash`](rust:runid::hash::ContentAddressed::content_hash) is the root,
and is the only place [`HASH_VERSION`](rust:runid::hash::HASH_VERSION) is
written. The encoding is designed so that equal values always produce equal
bytes and distinct values never do: each named type writes its fully qualified
name and a schema version first, so two structs with the same field bytes
differ; every string, vector, map and set is prefixed by its length, so
`("ab", "c")` and `("a", "bc")` differ; maps and sets are written in sorted key
order, so iteration order cannot leak in; an enum writes its variant index; an
`Option` writes a tag byte.

<!-- decl: rust:runid::hash::CanonicalHasher -->

Floating-point numbers get two policies, and the policy is chosen by the type of
the field. A resolved input (a parameter value, `dt`, a time, a bound) is a
[`FiniteF64`](rust:runid::float::FiniteF64): construction refuses `NaN` and
`±Inf`, as a value rather than a panic, and turns `-0.0` into `+0.0`, so that
`--param x=-0` and `--param x=0` are one run. A float inside the compiled model
is hashed by its raw bits through
[`write_f64_bits`](rust:runid::hash::CanonicalHasher::write_f64_bits), which
keeps `-0.0` and `+0.0` apart and accepts `NaN`, because the IR's own equality
treats them as distinct constants and a model holding a `NaN` constant must
still be hashable. `f64` itself does not implement `ContentAddressed`, so a new
float field in a run input does not compile until its author picks one of the
two.

<!-- decl: rust:runid::float::FiniteF64 -->

The compiled model is a foreign type tree from the `ir` crate; its
`ContentAddressed` impls are written by hand in
[`runid::ir_hash`](rust:runid::ir_hash), and
[`model_ir_hash`](rust:runid::inputs::model_ir_hash) hashes every field of the
model. The model has no field that only says how a result is rendered. Every
identity path builds its model level through
[`ModelDigest::from_model`](rust:runid::inputs::ModelDigest::from_model), which
calls it.

## The `RunInput` derive

The run-input types (the level digests and the leaf inputs in
[`runid::inputs`](rust:runid::inputs)) do not implement `ContentAddressed` by
hand. They derive it with `#[derive(RunInput)]`, from the `runid-derive` crate,
which hashes every field in declaration order. A field is left out only when it
is marked `#[run_input(provenance)]`, and there is no rule that leaves a field
out when it holds its default. The configuration level of a simulation shows the
shape:

<!-- decl: rust:runid::inputs::SimConfig -->

The seed level shows the one exclusion on the simulation path:

<!-- decl: rust:runid::inputs::Seed -->

`process_seed` is the seed the engine draws from. It is derived from the user's
`--seed` by [`mix_cell_seed`](rust:cli::util::mix_cell_seed), which mixes in the
cell's position: `--seed 5 --replicates 3` runs three cells with three process
seeds, all from base seed 5. Hashing the base seed would put those three cells,
which hold three different trajectories, under one `run_id`. `base_seed` is
therefore provenance. The run store's cell identity makes the same choice: it
holds the process seed, and the cell's folder is labelled by the seed typed, so
the three cells of `--seed 5 --replicates 3` are `seed_5-…`, `seed_5-r1-…` and
`seed_5-r2-…`.

For `Seed`, the derive generates the equivalent of the following. This is a hand
expansion for reading, not code in the repository; it is modelled on the hand
impls in `rust/crates/runid/src/macro_eq.rs`:

```rust
// Hand expansion of #[derive(RunInput)] on Seed (illustrative).
impl runid::ContentAddressed for Seed {
    fn hash_into(&self, __h: &mut runid::CanonicalHasher) {
        // The type tag: the type's full path, "runid::inputs::Seed".
        __h.write_type_tag(concat!(module_path!(), "::", stringify!(Seed)));
        // Version 1, the default; #[run_input(schema_version = N)] sets another.
        __h.write_schema_version(1);
        runid::ContentAddressed::hash_into(&self.process_seed, __h);
        // base_seed carries #[run_input(provenance)], so no line is emitted.
    }
}
```

An enum is expanded the same way, with a `match` that writes the variant's index
as a `u32` before the variant's fields. Because each field is hashed through
`<FieldType as ContentAddressed>`, a field whose type has no impl is a compile
error, so a new input cannot be added in a form the hash cannot see.

The derive replaces hand-written hashing, so its output has to be the encoding
the hand impls define. `rust/crates/runid/src/macro_eq.rs` pins that: it derives
`RunInput` on a struct, an enum with unit, tuple and struct variants, and a
struct with a `schema_version` override, writes the expected `hash_into` for
each by hand in the same module (so `module_path!()` gives the same type tag),
and asserts the two digests are equal (`struct_macro_equals_hand`,
`enum_macro_equals_hand`, `schema_version_override_matches_hand`). A fourth
test, `struct_provenance_field_is_skipped`, asserts that changing a provenance
field leaves the hash unchanged.

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
commands install one policy. `fit predict` alone nests a second, an allowing
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
set `CAMDL_EXPERIMENTAL_<NAME>` are the engine switches: a typed field of the
engine identity beside the build,
[`EngineIdentity`](rust:runid::inputs::EngineIdentity) `{ build, switches }`
([`Environment::engine_identity`](rust:cli::env::Environment::engine_identity)).
It is hashed as one string, the build followed by each switch length-prefixed
(`+<len>:<NAME>`, and `=<len>:<VALUE>` for an experimental input) so that no
value can pass for another switch; with no switch on it is the build alone, so a
default run's identity is unchanged. `run.json` keeps the build in
`engine_version` and lists the switches in `engine_switches`, so a run this
build made with a switch is this build's run. The commands that store a result
under an address key the switches: `simulate` (in the ambient key of the run
store, which names the build directory every cell is under), `fit run` and
`fit predict` (in the ambient key of the run store, as `simulate`; a
prediction's manifests also record `engine_version` and `engine_switches`),
`pfilter` (at the `model` level) and `profile` (a point's base). `compare`
scores a chain-binomial fit through a `camdl pfilter` child, which keys them;
its in-process ODE scoring and `dev eval` store nothing under an address. The
switches themselves are fields of the evaluator settings each compiled model
carries ([`EvalSettings`](rust:sim::compiled_model::EvalSettings)), set by
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

| Version                                                                               | Folded into                                                           | A bump re-keys                                                                                              |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| a struct's `#[run_input(schema_version = N)]`                                         | that type's hash, after its type tag                                  | every leaf whose input embeds the type                                                                      |
| [`HASH_VERSION`](rust:runid::hash::HASH_VERSION)                                      | every root hash and every `run_id`                                    | every leaf of every kind                                                                                    |
| `ir/VERSION`                                                                          | the `ir_version` field of `ModelDigest`                               | every leaf with a model level: every kind at present                                                        |
| the engine identity (`ENGINE_ID`, the build fingerprint, and the engine switches)     | the `engine` field of `ModelDigest`; the ambient key of the run store | every leaf with a model level, and every result in the run store, whenever camdl is built from other source |
| [`PARTITION_RULE_VERSION`](rust:cli::fit::training_partition::PARTITION_RULE_VERSION) | the problem's `partition`                                             | every fit                                                                                                   |

A leaf written under an older `HASH_VERSION` stays readable, so `camdl show`,
`fit summary` and the browsers still read it, but a lookup reports it as
[`StaleReason::SchemaDrift`](rust:runid::store::StaleReason) and the run is
computed again. [`FORMAT_VERSION`](rust:runid::record::FORMAT_VERSION) versions
the `run.json` schema the same way without entering any hash. `SimConfig`'s doc
comment above shows the per-struct version in use: each bump records a change to
the field set or to the bytes a leaf holds.

## Where a change lands

| The change                                               | Edit                                                                                                                        | Version                                             | Pinned by                                                                                                                             |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| A `simulate` flag that changes the stored trajectory     | an `#[input(identity)]` field of `SimulateArgs`, carried into a field of `sim_store::SimConfig` (or lowered into the model) | the `version` of `SimConfig`'s `#[identity(..)]`    | `every_simulate_input_keeps_its_class`, with a `vary` fixture for the field                                                           |
| A key in the fit configuration                           | a field of `config_v2::Problem`, placed by `fit_store::resolve_problem` in a field of the problem's or the run's identity   | the `version` of that struct's `#[identity(..)]`    | `every_fit_run_input_keeps_its_class`, with a `vary` fixture for the field                                                            |
| A `fit run` flag that changes a method's output          | a field of `FitMethodFlags`, `CliMethodOverrides`, an `OverrideFlag` row, and the copy in `fit::run_config::resolve`        | none: the method level hashes the method as written | `every_cli_sampler_flag_changes_the_method_identity`                                                                                  |
| A field that is displayed but does not change the output | `#[run_input(provenance)]` on the field, or record it only in `run.json` (`Provenance`, `RecordMeta`)                       | none                                                | a case in `differential_presentation_inputs_are_inert`                                                                                |
| A float policy, or any other rule of the encoding        | `runid::hash` or `runid::float`, and the hand impls in `runid::ir_hash`                                                     | `HASH_VERSION`, with an entry in its doc            | the property tests in `rust/crates/runid/src/tests.rs`; `macro_eq.rs`; both golden tables                                             |
| The IR schema                                            | the atomic OCaml, Rust and golden update (`.claude/skills/golden-update`)                                                   | `ir/VERSION`                                        | `rust/crates/runid/tests/golden_model_hashes.rs`, regenerated by `make update-golden-model-hashes` only when the re-key is the change |

The two golden tables pin absolute values.
`rust/crates/cli/tests/fixtures/golden_run_ids.tsv`, read by
`golden_run_ids_match_the_reference_table` in
`rust/crates/cli/src/resolve/tests.rs`, holds the model-level hash and `run_id`
of every kind for five compiled golden models, with every other input fixed, the
IR version and engine strings included; `make update-golden-run-ids` regenerates
it. `rust/crates/runid/tests/fixtures/golden_model_hashes.tsv` holds the model
hash of every compiled golden. A row that moves is a re-key, and it is the
subject of its commit, never a side effect of it.
