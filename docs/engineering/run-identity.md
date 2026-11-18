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
  times are formatted in the output. It is stripped before hashing, so changing
  it finds the same run. For example, the model's `output.format` changes how a
  trajectory is written out, not which numbers it holds.
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
wall-clock timings are kept in per-attempt execution records outside the leaf's
checksummed files. One limit remains: the platform (the operating system's
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

The inputs are not hashed as one blob. For a simulation they are split into five
levels, each a disjoint slice: the model, the configuration, the parameters, the
scenario and the seed ([`TrajectoryInput`](rust:runid::inputs::TrajectoryInput)
composes them). Each level is hashed on its own, and the leaf's address is
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
named `{label}-{hash8}`, under the kind's partition (`sims/`, `fits/`, …). The
label is provenance, the eight hex characters are the first four bytes of the
level's hash, and the full hashes are in `run.json`. Grouping falls out of the
tree without any index: the seeds of one ensemble are sibling directories under
one scenario directory, and a `--no-flows` variant of a run is a sibling
directory at the configuration level. A lookup finds a leaf by its hashes alone
([`find_leaf`](rust:runid::store::find_leaf)), so a renamed model file, whose
label changes, still finds the result its hashes name.

A leaf found at the expected path is reused only after
[`FsCasStore::lookup`](rust:runid::store::FsCasStore::lookup) checks it: the
recorded `run_id` must equal the expected one (a different one is a short-hash
[`Collision`](rust:runid::store::Lookup), resolved by suffixing the directory,
never by overwriting), the record's hash and format versions must be current,
its status `completed`, and its files exactly the ones the record lists. The
reuse decision itself is [`check_reuse`](rust:cli::resolve::check_reuse).

A multi-cell `simulate` asks before it computes, and skips the cells the store
holds. A single-cell `simulate` computes its cell anyway, and its commit
compares each file it staged with the stored one under the same `run_id`: equal
digests keep the stored leaf, and unequal ones are refused as
[`DivergentRecompute`](rust:runid::store::CasError), with the staged bytes kept
in `.quarantine/`. Since runs are seeded and deterministic, that error means an
input that changes the output is missing from the identity, or the run is not
deterministic.

What a leaf's run id holds is the same whichever invocation wrote it. An
artifact that only some invocations ask for is therefore never one of the leaf's
own files, which the manifest lists, but a _child_: a directory under the leaf,
declared in `run.json`'s `children` (recorded, never hashed) and passed over by
the manifest and the exact-set check. A sim leaf has two. The synthetic
observations (`--obs` and its siblings) are at `obs/<hash8>-<obs_seed>/`,
addressed by the observation blocks, the `--emit-every` cadence and the file
format, which the trajectory's run id does not cover. The lineage event log
(`--event-log`) is at the fixed address `event_log/`, because everything it
depends on is in the run id: the recorder draws no randomness and reads only the
model, the parameters, the seed and the backend's settings, and the recorder and
its file format belong to the engine, whose fingerprint the `model` level folds
in. A child is written after the leaf commits, declared before it is written (so
a crash leaves a declared child with nothing in it, never an undeclared
directory that would make the leaf stale); the event log, once complete, is
never overwritten with other bytes. The
[`child`](rust:cli::input_class::InputClass) class names the inputs that ask for
one.

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
[`runid::ir_hash`](rust:runid::ir_hash). Before any model is hashed,
[`model_ir_hash`](rust:runid::inputs::model_ir_hash) passes it through
[`normalize_for_hash`](rust:runid::inputs::normalize_for_hash), which blanks the
two IR fields that only say how a result is rendered (`output.format` and
`simulation.time_semantics`). Every identity path builds its model level through
[`ModelDigest::from_model`](rust:runid::inputs::ModelDigest::from_model), which
calls it, so no artifact kind can key on `--format`.

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
therefore provenance. On the simulation path the seed directory's label is built
from the process seed too, so a lone `--seed 5` lives in `seed_5-…` and a mixed
cell in a directory such as `seed_5871781006564002452-…`.

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

The derive guarantees that every field of `SimConfig` reaches the hash. For
`simulate`, four more things guarantee that every input reaches it, or is
declared not to:

1. **Every input is classified.** A class belongs to a (command, input) pair:
   the test-only table `SIMULATE` in [`input_class`](rust:cli::input_class)
   gives one of `keyed(level)`, `execution`, `provenance`, `presentation`,
   `child`, `expands` or `removed` (the module's table says what each means) to
   every field of [`SimulateArgs`](rust:cli::args::SimulateArgs), of the groups
   it flattens ([`OutputView`](rust:cli::args::OutputView) and the rest) and of
   the global options. `every_simulate_flag_is_classified` checks the table
   against every argument clap accepts, both ways. Environment variables have a
   class column in [`env::VARS`](rust:cli::env::VARS), and are read nowhere but
   in that module
   (`rust/crates/cli/tests/environment_is_read_in_one_module.rs`).
2. **Identity and engine settings come from one value.** After its precedence
   rules (an explicit flag, a fit's provenance, a replayed fit's simulator, the
   model's own default), `run_simulate` resolves its configuration once, into
   [`ResolvedSimConfig`](rust:cli::sim_config::ResolvedSimConfig): the base
   model and its [`ModelDigest`](rust:runid::inputs::ModelDigest), the run-wide
   part of the `config` level ([`RunConfig`](rust:cli::sim_config::RunConfig):
   backend, `dt`, the output schedule, the two evaluator flags, the column view)
   and the scenario arms. [`CasSink::cell_resolve`](rust:cli::cas_sink::CasSink)
   builds each cell's `SimConfig` with
   [`RunConfig::cell_config`](rust:cli::sim_config::RunConfig), which
   destructures `RunConfig` exhaustively, so a field added to either struct does
   not compile until it is carried across.
   [`ResolvedSimConfig::engine_settings`](rust:cli::sim_config::ResolvedSimConfig)
   is the one conversion to what runs: the job's backend, `dt` and evaluator
   settings ([`EvalSettings`](rust:sim::compiled_model::EvalSettings), which
   every cell's compiled model carries), and the forcing-extrapolation policy
   ([`ForcingPolicy`](rust:cli::forcing_extrapolation::ForcingPolicy)). It is
   called once per invocation, and the CAS sink keeps that instance, so every
   leaf's record names the policy the run's forcing notes went into. The thread
   count, the store root and `--force` travel in a separate
   [`Execution`](rust:cli::sim_config::Execution) value.
3. **A cell's identity and its run come from one value.**
   [`ResolvedCell::resolve`](rust:cli::sim_config::ResolvedCell) resolves a
   cell's `SimRun` through the engine's own resolution
   ([`resolve_loaded_run`](rust:cli::util::resolve_loaded_run)) into the model
   the cell executes, its table digests, seed and restored origin state, and its
   backend, step, sampler and evaluator settings. The engine
   ([`simulate_compiled`](rust:cli::util::simulate_compiled)) runs from it and
   hands the same instance back with the trajectory (`engine::CellResult`);
   `CasSink::merge_cell` keys the written leaf from it, through
   `CasSink::cell_identity`, whose destructure binds no field to `_`, so a field
   added to the cell does not compile there until it is named and used. The
   window and the parameter values are read off the resolved model. What the
   identity keys from the invocation instead is checked against the cell: a cell
   whose backend, step, evaluator settings or sampler differ from the run's
   `RunConfig`, whose interventions differ from its scenario arm's, or that
   records extra snapshot times is refused rather than keyed. The cache-hit
   check has to look a cell up before it runs, so it resolves the cell over the
   base model; the written leaf's identity is checked against that pre-run one,
   and a cell whose two identities differ is not written.
4. **Every input has a case.** `rust/crates/cli/tests/simulate_input_classes.rs`
   runs camdl into fresh stores, in a cleared environment, once per input and
   checks what its class predicts (below); `every_simulate_input_has_a_case`
   fails if one has no case.

The configuration is not hashed as one value: a run has many cells, and each
part reaches the level of a cell that it belongs to. The exhaustive copy of
`RunConfig` makes the run-wide part include-by-default, and the exhaustive
destructure of `ResolvedCell` the per-cell part (the window, the parameter
values and table digests, the seed and the restored origin state). Two limits
remain. The `model` level is computed once, from the sink's base model, so
`ir_path`, `integrator` and `obs_anchors` are keyed from the job's run rather
than each cell's own load; every cell copies them from the job, but a cell
carrying other values would not be told apart. And a `SimRun` field that changed
the resolved model other than through its parameter values, its window or its
interventions would reach the engine without reaching the identity.
`ResolvedCell::resolve` names every `SimRun` field, but a field named there as
`field: _` compiles, so the name is a point for review, not a guarantee.

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
none installed, a note is refused. `sim::run_notes` also says each distinct note
once per process, so a second policy installed in the same process would not be
told of a forcing the first was told of. Every command installs one policy.
`fit predict` alone nests a second, an allowing one, once it has read the stored
fit's config; a refusal under the first has already ended the command, and the
inner guard's drop puts the first back. A claim-holding `fit run` keeps a
refusal for its claim, and each chain polls the command's policy (held by `Out`
from `hold_claim` to `take_claim`, apart from the claim itself), so the first
chain to fail the claim does not hide the refusal from the others.

Two commands still build their policy from their own flag alone, not from the
fit they replay: `simulate --draws posterior --fit` (`run_simulate`), and
`compare` when it scores an ODE fit in-process, so an ODE fit made under
`[config] allow_forcing_extrapolation` is refused by `compare` without the flag.
`compare`'s `camdl pfilter` child, which scores a chain-binomial fit, is given
the fit's setting. Both remaining cases are for the commands' conversion in step
4; the in-process `compare` case cannot be met by installing a policy per fit,
because of the once-per-process note above.

`--no-licm` is a compile setting,
[`CompileSettings`](rust:cli::util::CompileSettings). `fit run` resolves it
(with `CAMDL_NO_LICM`) and passes it to its compile; `simulate` installs its
resolved value process-wide, and every other command reads the process-wide
setting, which `main` sets from the flag.

Two kinds of flag reach the identity through the model and the parameters rather
than `RunConfig`. A flag that rewrites the model before it runs
(`--output-every`, which rematerializes the model's output schedule;
`--integrator`; the global `--no-licm`) is hashed with the model, because
[`load_run_model`](rust:cli::util::load_run_model) is the one load that both the
engine and the model level read. A parameter flag (`--param`, `--params`,
`--param-vec`, a draw or sweep point) is hashed with the parameters, because the
`params` level is the output of
[`resolve_run_parameters`](rust:cli::util::resolve_run_parameters), the same
call the engine makes.

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
build made with a switch is this build's run. `simulate` and `fit run` key the
switches (a fit at its `fit` level); the other commands key them as they are
converted. `CAMDL_NO_CONSTANT_FOLD` and `CAMDL_NO_LICM` change the compiled IR.
An experimental input is also listed in the `provenance.experimental` field of
`run.json`.

`fit run` resolves its configuration once, before anything runs, into
[`ResolvedFitConfig`](rust:cli::fit::run_config::ResolvedFitConfig): the file
with every flag applied, the compiled model, the engine identity, the seeds, the
sweep and the chain-starts rule. Its identity and its runners are built from
that value, and `run_fit` destructures it with every field named; the command
line itself is read only before it, so no argument reaches a runner except
through the resolved value, the execution settings (`--parallel`, `--force`) or
the label. The fit file is read once, and every fit folder the run writes keeps
two configs: `fit.toml.original`, that text byte for byte (its `fit_toml_hash`
and the run's config identity are taken from it), and `fit.resolved.toml`, the
configuration the folder's fits ran under
([`resolved_config_text`](rust:cli::fit::run_config::resolved_config_text)): the
text with every problem-half value the command line set
(`--allow-forcing-extrapolation`) and the folder's own `[fixed]` (a sweep
point's values) written in, keys sorted, marked as generated. Neither enters an
identity. Every reader of a stored fit's config reads the resolved file: a
run-directory, label or run-id handle (`fit predict <leaf>`, `compare <leaf>`)
so runs under the override and at the sweep point's values, and `fit table`
diffs it. A folder an older build wrote has none and is refused, naming the
re-run that writes it; no reader falls back to `fit.toml.original`. A `fit.toml`
handle names only the run the file alone makes, so it refuses a flag-made fit
and names its leaf. `pfilter`, `profile` and `survey` `--fit` read a fit file
rather than the stored config and still refuse a flag-made fit; they are
converted next. Every `fit run` input is classified in `input_class::FIT_RUN`
and has a case in `rust/crates/cli/tests/fit_run_input_classes.rs`.

`fit run` routes its method flags through one struct,
[`CliMethodOverrides`](rust:cli::fit::config_v2::CliMethodOverrides), which
[`Algorithm::apply_cli_overrides`](rust:cli::fit::config_v2::Algorithm::apply_cli_overrides)
writes into the method before its identity is taken. Its `passed` method
destructures the struct without `..`, so a field added to it does not compile
until it names an [`OverrideFlag`](rust:cli::fit::config_v2::OverrideFlag), and
`every_cli_sampler_flag_changes_the_method_identity` in `config_v2.rs` runs each
flag through the identity. The copy from the clap struct into
`CliMethodOverrides`, in [`cmd_fit_run_v2`](rust:cli::fit::cmd_fit_run_v2), is
by hand, as is each flag that writes into the fit configuration instead
(`--allow-forcing-extrapolation` sets `[config] allow_forcing_extrapolation`).

So when you add a `simulate` flag that changes what a run computes or stores:
classify it `keyed`, give it a field of `RunConfig` (or lower it into the model,
or into the parameters), and add its case to `simulate_input_classes.rs`. For
`fit run` and the other commands, carry it through each step above, add a case
to the differential test, and write the end-to-end test that runs camdl with and
without the flag and finds two leaves.

## Testing a classification

Calling an input identity, presentation or provenance is a prediction about what
happens when that input alone is changed, so each classification can be tested.
Run camdl twice into two fresh stores, changing only that input, and compare the
stored results (the trajectory, the ensemble, the observation table; not
`run.json`, which records the time and the command line and always differs). The
rule is that **the stored bytes change exactly when the `run_id` changes**, and
it can fail two ways:

- **The bytes change and the `run_id` does not.** The input belongs in identity
  and is missing from the hash, so the store would serve one result in place of
  the other. This is the failure the flag path above leaves open, and the
  dangerous one.
- **The `run_id` changes and the bytes do not.** The input is hashed but changes
  nothing. Runs that could have been reused are computed again; no result is
  wrong.

`--parallel` is tested this way.
`parallel_runs_are_byte_identical_to_the_serial_run` in
`rust/crates/cli/tests/simulate_sweep.rs` runs a four-cell sweep into two fresh
stores with `--parallel 1` and `--parallel 4` and requires the combined
trajectory file, the observation table and the stored ensemble to be equal byte
for byte. The claim holds because a cell's seed is a function of its position in
the grid and cells are merged in a fixed order, so the thread that finishes
first never reaches the output.
`a_parallel_run_splices_hits_and_misses_in_canonical_order`, in
`simulate_store_reuse.rs`, checks the same when some cells are read back from
the store and the rest are computed.

Every `simulate` input is tested against the rule by its class, in
`rust/crates/cli/tests/simulate_input_classes.rs`: a `keyed(level)` input,
changed alone, re-keys every cell and changes the hash of the level its class
names; an `execution` input leaves the run ids and the stored bytes unchanged;
an `expands` input keeps the run ids of the cells common to both values; a
presentation or provenance input, on a fixture where it does something, leaves
the leaves unaffected, or writes none when it takes the run out of the store;
and a `child` input, run after a run without it in the same store, leaves every
leaf's run id and manifest unchanged byte for byte and writes its child at its
address under the leaf. These are regression evidence on small fixtures, not
proof that no model behaves otherwise. `fit run`'s inputs are tested the same
way in `rust/crates/cli/tests/fit_run_input_classes.rs`, against the `fit`,
`method` and `seed` levels, with a `removed` case for each old spelling it
refuses. The other commands' flags are not yet tested this way; the differential
tests in `rust/crates/cli/src/resolve/tests.rs` check the hash of one
`SimConfig` field at a time without running the simulation.

## How a fit is hashed

A fit's identity has a second path. The fit configuration is a serde type, not a
`RunInput` type, and enters the `fit` level as the digest of its JSON:
[`fit_level_digest`](rust:cli::fit::cas::fit_level_digest) puts it in the
`fit_toml` field of [`FitDigest`](rust:runid::inputs::FitDigest), beside the
model digest and the digests of the training and holdout files.
[`canonical_config_hash`](rust:cli::fit::cas::canonical_config_hash) does the
work in four steps: it rejects any non-finite float anywhere in the value
(serde_json writes `NaN` and `±Inf` as `null`, so without the check two configs
differing only in them would hash alike), serializes the whole
[`Problem`](rust:cli::fit::config_v2::Problem), removes the top-level keys that
are provenance (`output_dir` and the `[provenance]` note), and hashes the JSON
with its object keys sorted. The `method` level hashes the method's identity
payload as canonical JSON in the same way (`method_config_hash`, inside a
`MethodConfig`), and the seed has its own level.

A level hash can only be minted by `canonical_config_hash` or by
[`structural_level_hash`](rust:cli::fit::cas::structural_level_hash), the derive
path: [`LevelHash`](rust:cli::fit::cas::LevelHash) has no public constructor, so
a hash of a hand-listed `json!` literal, which would contain only the fields its
author remembered, cannot become a level.

Serializing the whole struct includes every field, as the derive does, but serde
has a way to leave a field out that the derive deliberately lacks:
`#[serde(skip_serializing_if = …)]`. A field so marked is absent from the JSON
whenever the predicate holds, and an absent key contributes no bytes. The fit
configuration uses it for flags whose default is `false`:

<!-- decl: rust:cli::fit::config_v2::FitBackendConfig -->

Both booleans are skipped when `false`, so a config that writes
`allow_degenerate_rates = false` hashes as one that omits the key, and a stored
fit made from a config without the key keeps its address; `true` adds the key
and re-keys. The derive makes the opposite choice: a new `SimConfig` field
contributes bytes at its default, and re-keys every leaf. On the JSON path,
then, whether a field's default re-keys is decided per field by its serde
attributes, and a field without `skip_serializing_if` behaves as it would under
the derive. The tests `allow_degenerate_rates_changes_the_fit_identity` and
`allow_forcing_extrapolation_false_is_the_absent_key`, in
`rust/crates/cli/src/fit/cas.rs`, pin both halves for those two fields.

[`config_identity_hash`](rust:cli::fit::cas::config_identity_hash) is a third
hash of the same file and is not run identity: it is how a `fit.toml` handle is
matched to the stored fit it produced, and includes the `[method]` table.

## Versions

A change that alters the bytes being hashed re-keys some set of stored leaves.
Which set is chosen by which version moves, and each move is deliberate:

| Version                                                                               | Folded into                                         | A bump re-keys                                                           |
| ------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| a struct's `#[run_input(schema_version = N)]`                                         | that type's hash, after its type tag                | every leaf whose input embeds the type                                   |
| [`HASH_VERSION`](rust:runid::hash::HASH_VERSION)                                      | every root hash and every `run_id`                  | every leaf of every kind                                                 |
| `ir/VERSION`                                                                          | the `ir_version` field of `ModelDigest`             | every leaf with a model level: every kind at present                     |
| the engine identity (`ENGINE_ID`, the build fingerprint, and the engine switches)     | the `engine` field of `ModelDigest` and `FitDigest` | every leaf with a model level, whenever camdl is built from other source |
| [`PARTITION_RULE_VERSION`](rust:cli::fit::training_partition::PARTITION_RULE_VERSION) | `FitDigest`                                         | every fit                                                                |

A leaf written under an older `HASH_VERSION` stays readable, so `camdl show`,
`fit summary` and the browsers still read it, but a lookup reports it as
[`StaleReason::SchemaDrift`](rust:runid::store::StaleReason) and the run is
computed again. [`FORMAT_VERSION`](rust:runid::record::FORMAT_VERSION) versions
the `run.json` schema the same way without entering any hash. `SimConfig`'s doc
comment above shows the per-struct version in use: each bump records a change to
the field set or to the bytes a leaf holds.

## Where a change lands

| The change                                               | Edit                                                                                                                           | Version                                                                      | Pinned by                                                                                                                             |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| A `simulate` flag that changes the stored trajectory     | a `keyed(config)` row in `input_class::SIMULATE`, a field of `RunConfig` and of `SimConfig` (or lower the flag into the model) | `SimConfig`'s `schema_version`                                               | a `keyed_<flag>` case in `simulate_input_classes.rs`; the moved hexes in `cas_identity_pins` and `tests/fixtures/golden_run_ids.tsv`  |
| A key in the fit configuration                           | a field on the serde struct, with `skip_serializing_if` at its default if existing fits should keep their keys                 | none when skipped at its default                                             | a `fit/cas.rs` test that the default hashes as the absent key and another value re-keys                                               |
| A `fit run` flag that changes a method's output          | `CliMethodOverrides`, an `OverrideFlag` row, the method's arm of `write_cli_overrides`, and the copy in `cmd_fit_run_v2`       | none: the method level hashes the method as written                          | `every_cli_sampler_flag_changes_the_method_identity`                                                                                  |
| A field that is displayed but does not change the output | `#[run_input(provenance)]` on the field, or record it only in `run.json` (`Provenance`, `RecordMeta`)                          | none                                                                         | a case in `differential_presentation_inputs_are_inert`                                                                                |
| A model field that only changes how a result is rendered | blank it in `normalize_for_hash`                                                                                               | none if the field is new; otherwise the move is a re-key and needs a version | `presentation_fields_are_inert_on_every_cas_kind`; `normalization_is_idempotent_and_sim_fit_bytes_unchanged`                          |
| A float policy, or any other rule of the encoding        | `runid::hash` or `runid::float`, and the hand impls in `runid::ir_hash`                                                        | `HASH_VERSION`, with an entry in its doc                                     | the property tests in `rust/crates/runid/src/tests.rs`; `macro_eq.rs`; both golden tables                                             |
| The IR schema                                            | the atomic OCaml, Rust and golden update (`.claude/skills/golden-update`)                                                      | `ir/VERSION`                                                                 | `rust/crates/runid/tests/golden_model_hashes.rs`, regenerated by `make update-golden-model-hashes` only when the re-key is the change |

The two golden tables pin absolute values.
`rust/crates/cli/tests/fixtures/golden_run_ids.tsv`, read by
`golden_run_ids_match_the_reference_table` in
`rust/crates/cli/src/resolve/tests.rs`, holds the model-level hash and `run_id`
of every kind for five compiled golden models, with every other input fixed, the
IR version and engine strings included; `make update-golden-run-ids` regenerates
it. `rust/crates/runid/tests/fixtures/golden_model_hashes.tsv` holds the model
hash of every compiled golden. A row that moves is a re-key, and it is the
subject of its commit, never a side effect of it.
