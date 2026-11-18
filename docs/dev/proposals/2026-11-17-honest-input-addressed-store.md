# Making the run store honestly input-addressed

Status: Accepted — reviewed against the code and by two external design reviews;
every design question is decided. Phases 1 and 2 are being implemented.

**Scope:** three phases. Phase 1 closes the known correctness holes. Phase 2
separates a computation's result from its executions. Phase 3 builds a minimal
input architecture for `simulate`, then removes the process-wide evaluator
settings, then converts the other commands. The fit's typed identity, which
would replace canonical-JSON hashing, is a named follow-up proposal.

**Background assumed:** none; terms are defined as they come up. The current
design is described in `docs/engineering/run-identity.md` (on the docs branch)
and `docs/camdl-run-spec.md` §2.1 and §3.7. Earlier proposals this one builds
on: `2026-05-31-content-addressed-run-identity.md`,
`2026-06-16-input-surface-and-cas-unification.md`,
`2026-06-17-cas-runinput-type-consolidation.md`,
`2026-07-16-gradient-maps-out-of-run-identity.md`. Line numbers are against
`origin/main` at f9f678b12.

## Summary

camdl keeps every simulation and fit in a store, under an address computed from
the run's inputs. That is correct only if a run is a pure function of the inputs
that go into the address. We measured whether that holds. Almost everything
does. These do not:

1. **The engine's identity misses uncommitted changes** to the Rust engine and,
   for gradient-using fits, to the compiler. Such a build can be handed results
   computed by the old code, with no error.
2. **Nothing forces an input into the identity.** Each command copies its flags
   by hand into the struct that is hashed; a forgotten copy compiles. Three
   commands appear to mishandle one flag today.
3. **Fit leaves are not byte-reproducible**, because some checksummed result
   files carry wall-clock times and a process id.
4. **Development switches are ungoverned.** Environment variables are read
   wherever they are needed, and nothing records that a result was produced with
   an unreleased feature.
5. **The documentation calls the store "content-addressed"**, which names a
   different design.

The architecture this proposal moves toward rests on three decisions: **engine
identity reflects the actual build**; **execution and identity derive from one
resolved configuration**; and **a computation's result is kept separate from the
metadata of each attempt to execute it**.

## What the store promises

### Terms

- A **run** is one simulation cell or one fit. Its result is stored as a
  **leaf**: a directory of result files, a manifest of their checksums, and a
  record, `run.json`. A leaf's **stored bytes** are the files its manifest
  checksums.
- A run's **identity** is everything that determines its result: the compiled
  model, the configuration, the parameter values, the scenario, the seed, and
  the engine. The **run id** is a hash of the identity and is the leaf's
  address.
- Inputs that do not enter the hash are **provenance** (how the run was asked
  for: the command line, a `--label`), **execution** parameters (how it was
  computed: the thread count) or **presentation** (how output is written).
- An **execution** is one attempt to perform a computation: the first run, a
  resumed run, a failed attempt, a recomputation on another worker. One run id
  can have several executions.

### Input-addressed, and what that requires

The address is computed from the inputs before the run happens, so the store is
**input-addressed**: the address names the computation, not the bytes it
produced. This is the model of the Nix store, which Dolstra calls the
_extensional_ model, as opposed to the _intensional_, content-addressed model in
which an object's address is the hash of the object (E. Dolstra, _The Purely
Functional Software Deployment Model_, PhD thesis, Utrecht University, 2006,
chapters 5 and 6). The 2026-06-16 proposal (§0) already decided the store should
be described this way.

An input-addressed store is a memo table, correct only if the function it
memoises is **pure** (its output depends on its hashed inputs alone). Three
claims are worth keeping apart, because they differ in strength:

1. **Identity correctness:** one address never stands for two different
   computations. This is what the store needs, and what this proposal secures.
2. **Statistical reproducibility:** repeated executions follow the same
   specified stochastic procedure.
3. **Byte reproducibility:** repeated executions produce exactly the same bytes.
   This is stronger than the first two. camdl promises it only where it is
   tested (phase 2).

### What tests can and cannot show

A differential test — run camdl with an input set two ways and compare — shows
that the input made no difference on that fixture. It does not prove the input
can never make a difference: another model could reach a parallel reduction, a
different scheduling path or a different random-number sequence. Differential
tests are regression evidence on representative fixtures. Where this proposal
relies on them it says so, and the remaining assumptions — parallel scheduling,
the platform's mathematics library, stochastic algorithms, any remaining global
state — are stated in the documentation rather than claimed away.

## What we measured

Each candidate hidden input was run twice into fresh stores, with and without
it, comparing stored bytes and run ids, on a small stratified model and a
44-patch spatial polio model, with the chain-binomial, ODE and Gillespie
backends, single runs and ensembles, plus pmmh and if2 fits.

| Candidate                                                            | Changes the result?   | In the hash?        | Verdict                                       |
| -------------------------------------------------------------------- | --------------------- | ------------------- | --------------------------------------------- |
| `CAMDL_EVAL_FLAT`, `CAMDL_EVAL_UNRESOLVED`, `CAMDL_NO_BINDING_CACHE` | no (measured)         | no                  | clean on these fixtures                       |
| `CAMDL_NO_CONSTANT_FOLD`, `CAMDL_NO_LICM` / `--no-licm`              | no (measured)         | yes, through the IR | clean; re-keys runs whose bytes are identical |
| `--parallel`, `RAYON_NUM_THREADS` (simulate, pmmh, if2)              | no (measured)         | no                  | clean on these fixtures                       |
| input files (`--table`, data, `--sweep-file`, init state)            | yes                   | yes                 | clean                                         |
| **uncommitted edits to the Rust engine**                             | **yes**               | **no**              | **hidden input**                              |
| **uncommitted edits to the compiler's autodiff**, gradient-using fit | **yes** (inferred)    | **no**              | **hidden input**                              |
| **a build without git**                                              | yes                   | no (`unknown`)      | **hidden input**                              |
| **OS / system maths library**                                        | possibly, in last bit | no                  | **hidden across machines** (inferred)         |
| **timing and process id in fit files**                               | the files, yes        | no                  | **fit leaves not reproducible**               |

`--parallel` was measured for simulate, pmmh and if2 only; resumed fits were not
measured.

Two concerns turn out not to apply. Leaf paths carry eight hex digits per level,
but a lookup reads the full run id from `run.json` and refuses a mismatch
(`runid/src/store.rs:273`), so a path collision is detected, never served. And
gradient maps, though `HashMap`s in the IR, are hashed in sorted key order like
every map (`runid/src/hash.rs:42, 365-373`), so map ordering cannot leak in.

## Deficiencies

### 1. The engine's identity misses uncommitted changes

**The Rust engine.** The engine enters the identity as `VERSION_SHORT`, the
release version plus `git rev-parse --short HEAD` (`cli/src/version.rs:12`,
`cli/build.rs:2-7`). Uncommitted changes do not change it:

1. `camdl simulate model.camdl --seeds 1:20` stores twenty leaves.
2. You edit the chain-binomial sampler, do not commit, and rebuild.
3. The same command finds the same twenty run ids and reads them back without
   computing: the old sampler's trajectories, no error.

We demonstrated the serving step: a leaf rewritten as another build would have
written it was served silently by a two-cell run, and refused only by a
single-cell recompute. A build without git gets `unknown`, so all such builds
share one identity. Nor does the identity reflect how the binary was built: the
enabled Cargo features (`cli` and `sim` have output-writer and lineage
features), the build profile, `RUSTFLAGS`, or the compiler version.

**The compiler, for gradient-using fits.** The gradient maps the compiler emits
are deliberately left out of the hash (`runid/src/ir_hash.rs:1540-1546`, by the
2026-07-16 proposal, so that switching gradient emission off for methods that
never read gradients does not re-key them). That holds for a fixed compiler. If
the compiler's autodiff is edited, a NUTS-within-PGAS or `nl-lbfgs` fit gets the
same run id with different gradients. Clean builds are protected by the camdlc
version handshake; `CAMDL_SKIP_VERSION_CHECK=1`, which development runs use,
removes that protection.

**Why it matters here.** Development runs many worktrees at once, usually with
uncommitted builds; this is the routine case. The 2026-06-16 proposal named the
Rust half of this gap and proposed a `CAMDL_GIT_DIRTY` marker (lines 552-555,
704-705); it was not built.

### 2. Nothing forces an input into the identity

A flag reaches the hash in two steps:

```
command line ──(1)──► identity struct ──(2)──► hash
```

Step 2 is guarded: `#[derive(RunInput)]` hashes every field of the identity
struct. Step 1 is hand-written in every command. For `simulate`:

```rust
// args/mod.rs — the flag
#[arg(long)] pub allow_degenerate_rates: bool,
// main.rs:783 — the engine reads it from a global…
sim::eval_stats::set_allow_degenerate_rates(a.allow_degenerate_rates);
// main.rs:2134 — …the hash gets a separate copy, by hand
build_simulate_cas_sink(/* … */ a.allow_degenerate_rates, /* … */)
//   → CasSink → TrajectoryCtx (cas_sink.rs:308) → SimConfig (resolve.rs:220)
```

There are two independent paths from the flag, one to the engine and one to the
hash, and nothing keeps them in step. A flag used by the engine and never copied
onward compiles, runs, changes the results, and leaves the run id unchanged; a
multi-seed run with it then reads back results computed without it.

`fit run` is better guarded but has the same hole: the copy from the clap struct
into `CliMethodOverrides` is by hand (`fit/mod.rs:395`), and its comment says so
(`config_v2.rs:1879`). This class of bug recurs: gh#189, gh#514 and gh#540 on
the fit side, the `--integrator` and `--param-vec` fixes on the simulate side.

**Three likely bugs of this kind,** each confirmed or dropped by a red test in
phase 1:

- `profile` accepts `--allow-degenerate-rates` and never applies it: the setter
  is called only at `main.rs:783`, `pfilter.rs:151` and `fit/mod.rs:441`.
- `pfilter` applies it but leaves it out of its identity
  (`pfilter_cas.rs:79-93`).
- Commands that replay a stored fit (`fit predict`, `compare`, `eval`) carry the
  fit's forcing-extrapolation setting across (`predict.rs:1650`) but not its
  `allow_degenerate_rates`.

### 3. Fit leaves are not byte-reproducible

A fit leaf's manifest checksums files carrying wall-clock times
(`chain_starts.tsv` "after 0.01s", `diagnostics.json`, `fit_state.toml` — whose
`FitState.timestamp` is a required field, `fit/state.rs:34` — `mle_params.toml`,
`pmmh_summary.json`) and a process id and time (`progress.json`). Two fits with
identical inputs and identical draws produce leaves whose stored bytes differ.
(`events.jsonl` is not in the manifest, `store.rs:1120-1134`.) There is an
identity effect as well: a fit started `from_params` on another fit's
`mle_params.toml` hashes that file's bytes (`fit/cas.rs:737-746`), timestamp
included, so refitting the parent re-keys the child; this has been patched by
hand twice (`cas.rs:612-618`, `:688-697`).

### 4. Development switches are ungoverned

The runtime reads a dozen `CAMDL_*` environment variables, each where it is
needed, including inside `sim` (`sim/src/trace.rs:108`, `resolved_expr.rs:509`).
Some switch evaluation paths (`CAMDL_EVAL_FLAT`, `CAMDL_NO_CONSTANT_FOLD`), some
are diagnostics (`CAMDL_TRACE_STEPS`), some are test hooks. There is no list of
them, no statement of which may change results, and no mechanism for a feature
that is not ready for users: nothing would stop a new `CAMDL_*` switch that
changes results from bypassing the identity, and nothing in a run's record says
that it used an unreleased feature. (The 21 hidden command-line flags are all
`_removed_*` traps that refuse old spellings; none gates a feature.)

### 5. The purity assumption is checked on one path only

When a leaf is recomputed and one with the same id exists, the store compares
the bytes and refuses a mismatch (`CasError::DivergentRecompute`,
`store.rs:166`). Only a repeated single-cell `simulate` reaches it: a multi-cell
run reads stored cells without computing, a fit takes the cache hit, and
`--force` quarantines the old leaf first.

### 6. The name

User docs outside `docs/dev` call the store "content-addressed" on 35 lines of
main, three of them in `camdl docs` topics, and `--help` text does too.

## Phase 1: close the known correctness holes

Each change here closes a demonstrated or likely wrong-answer route, and none
waits on new abstractions. Each lands with a regression test that shows the
wrong behaviour before the fix and not after.

### 1a. The engine identity is a build fingerprint

**Invariant:** engine identity distinguishes builds by the source, dependencies,
compiler, feature set, build configuration and target information available at
build time. Reuse is guaranteed only within the documented execution-environment
assumptions (one machine image; see below). It is a build fingerprint under a
specified environment, not a complete identity of the binary: the system
toolchain used for native code and dynamically linked libraries are not in it.

A new constant, `ENGINE_ID`, replaces `VERSION_SHORT` wherever the engine enters
a run's identity. It is a 128-bit digest, computed once by the build script and
embedded, of:

- the **source fingerprint**: the commit, plus a digest of uncommitted changes
  (tracked edits and untracked, not-ignored files) to the files that build the
  binary — every `src/` directory, `Cargo.toml` and `build.rs` under `rust/`,
  and `ir/VERSION`. Tests and benches are excluded, so editing a test does not
  re-key every run. Without git, the commit is replaced by a digest of the same
  files;
- `rust/Cargo.lock`, which pins the `nlopt` crate's version (not every aspect of
  its native compilation);
- the enabled Cargo features **of every workspace crate in the binary**, not
  only `cli`'s: a build script sees only its own package's features, so the
  implementation derives the resolved feature set for the `camdl` binary (from
  `cli`'s features, which forward to `sim`'s, verified with
  `cargo tree -e features`) and a test checks that switching a `sim` feature
  changes `ENGINE_ID`;
- the build profile;
- the full compiler description (`rustc -vV`, which includes the commit hash and
  LLVM version; the toolchain is pinned by `rust/rust-toolchain.toml`);
- the encoded `RUSTFLAGS` (`CARGO_ENCODED_RUSTFLAGS`), which apply to every
  crate in the build;
- the target triple.

The readable version string keeps a short prefix:

```text
0.4.0+abc1234                      clean build
0.4.0+abc1234.dirty-3f9c21e0       uncommitted changes
```

**Fail closed.** If any component cannot be determined, the build marks the
binary as having no reusable identity: such a binary computes and records
results but never reads a cached leaf.

**`ENGINE_ID` is separate from `GIT_HASH`**, which the camdlc version handshake
compares (`cli/src/util.rs`, `check_camdlc_version_once`). Every consumer moves
to `ENGINE_ID`: `cas_sink.rs:282`, `main.rs:3179`, `resolve.rs:316`,
`survey.rs:342`, `pfilter.rs:721`, `profile.rs:1799`,
`fit/mod.rs:930, 1315, 1344, 3072`, `fit/cas.rs`.

**Re-run propagation is tested across crates.** The version is computed in
`cli`'s build script, and a change to `sim`'s source rebuilds `sim` without
re-running that script unless the script watches `sim`'s files. The test edits a
file in a dependency crate (`sim`) and in `cli`, commits, and edits a test, and
checks that the first three change `ENGINE_ID` and the last does not.

**Not a portable identity.** The target triple does not identify the system
mathematics library: only `erfc` comes from the pure-Rust `libm` crate
(`sim/Cargo.toml:37-41`); `exp`, `ln` and `pow` come from the operating system,
which varies between OS versions on one triple. The guarantee is stated for one
machine image; a portable identity — a runtime environment fingerprint captured
when a run executes, not only when the binary is built — is a follow-up, needed
before stores move between machines (`2026-06-23-cloud-fit-dispatch.md`).

### 1b. Gradient maps in the identity of the fits that read them

**Invariant:** a fit whose method reads gradients re-keys when its gradients
change, whichever compiler produced them; a method that never reads gradients
does not.

For a fit whose method reads gradients (NUTS within PGAS, the `nl-*`
optimisers), the method level hashes the gradient maps that method consumes.
Runs that never read gradients keep the 2026-07-16 behaviour. A test builds two
models with an identical forward model and different gradient maps, and checks
that a gradient-using fit re-keys and a gradient-free one does not.

### 1c. The three missing-input bugs

Each is confirmed by a red test, then fixed: `profile` applies
`--allow-degenerate-rates`; `pfilter` hashes it; replay commands carry a stored
fit's setting across, as they already do for forcing extrapolation.

### 1e. `fit predict`'s output address

**Invariant:** two predictions computed with different inputs never share an
output location.

`fit predict` writes into a directory under the fit's run id, suffixed by an
8-hex-digit (32-bit) hash of a hand-copied `PredictOptions`
(`predict.rs:3520-3565`, used at `predict.rs:1526`). It never reads that
directory back as a cache: it always computes and writes. But the options omit
inputs that change the output — `--chains`, `--exclude-chains`, `--quantities`,
the forcing-extrapolation flag, the engine — so two predictions differing only
in those write into the same directory, the second overwriting the first, and
readers that locate predictions by that address (`fit/fit_table.rs:494`,
`fit/fit_view.rs`) can read output computed with one chain selection as if it
came from another. A red test confirms it (two predicts differing in
`--exclude-chains`, then a reader); the fix keys the directory by every input
that changes the output, at full width, and records the key in the output's
manifest.

### 1d. Exact handle resolution

**Invariant:** a fit named by handle is the computation the user asked for, or
the command says it cannot tell.

Today `fit/handle.rs:775` prefers a leaf whose engine version equals the running
one. With `ENGINE_ID`, the rule becomes: the leaf whose identity matches
exactly; otherwise an explicit ambiguity or incompatibility error that lists the
candidates and how to name one (`@<run-id>`). No newest-compatible-build
heuristic selects a scientific input. Reports say "stale" when the commit
differs and "built from uncommitted changes" when only the dirty digest differs.
In a development worktree this asks the user to choose more often; that is the
honest behaviour.

## Phase 2: results apart from executions

**Invariant:** a leaf's stored bytes are a function of its computation;
everything about a particular attempt to compute it lives outside them.

| Computation (checksummed)          | Execution (per attempt, not checksummed) |
| ---------------------------------- | ---------------------------------------- |
| draws, traces, final parameters    | start and end time, durations            |
| diagnostics derived from the draws | process id, host                         |
| the resolved configuration         | thread count                             |
| resume state                       | progress, events, failures               |

The volatile fields move into an **execution record** per attempt, kept beside
the leaf and outside its manifest. This is the smallest change to the present
layout: no new directory scheme is introduced, but resume, which already makes
several attempts at one computation, gets one record per attempt. Wall-clock
times leave `chain_starts.tsv`, `diagnostics.json`, `mle_params.toml` and
`pmmh_summary.json`; `FitState.timestamp` leaves `fit_state.toml`;
`progress.json` leaves the manifest. Every reader moves in the same change
(`fit summary`, `resumed_chain.rs:234`, `fit_summary.rs:1085`, the report
builder).

**What a resumed or extended computation is.** Some fit methods can be resumed
after interruption and extended to a longer target length. Today the extension
dimension is subtracted from the method's identity and re-folded as
`target_length`, and `resume_guard` computes a second, extension-free key
(`fit/config_v2.rs:1364-1407`, `fit/cas.rs:529-566`). Phase 2 makes four things
distinct and states each in the engineering page:

- the **computation's identity**: immutable, including its target length, so a
  fit extended to a longer target is a different computation with its own run
  id;
- a **checkpoint**: a partial result from which a computation can continue. It
  is not a completed result and is never served as one; it is keyed by the
  extension-free key so a longer computation can start from it;
- a **completed result** at a particular target length: an ordinary leaf;
- the **executions** that produce or extend them.

A test extends a completed fit to a longer target and checks that the shorter
fit's leaf is unchanged, still found under its own run id, and not reported as
the longer one.

**Byte reproducibility is promised where it is tested.** Tests run
representative methods (pmmh, if2, PGAS, one optimiser) twice, uninterrupted,
and assert identical manifest digests; and separately run each once
uninterrupted and once interrupted and resumed, and compare. A method that does
not pass is documented as statistically but not byte reproducible, rather than
claimed.

Identity effect: a fit started `from_params` on a parent's `mle_params.toml`
stops re-keying when the parent is refit, and the two hand patches in
`fit/cas.rs` are removed.

## Phase 3: a minimal input architecture

**Invariant:** what the engine executes is derived from the same resolved value
the identity is computed from, and every input of a converted command is
classified.

### The shape

```
argv ──clap──► SimulateArgs                 the surface, as typed
env  ────────► the environment module       (part E)
                    │
                    │  resolve(&SimulateArgs, &Model): parse, don't validate;
                    │  precedence applied once
                    ▼
             ResolvedSimConfig              one immutable semantic configuration
               │            │
   identity()  │            │  to_engine_config()
               ▼            ▼
           RunId         engine settings    (both from the same value)
```

`ResolvedSimConfig` is today's `SimConfig` and its sibling levels, grown to be
the single authoritative configuration, and the engine's settings are derived
from it by one function in `cli`.

**The identity is include-by-default.** `ResolvedSimConfig` derives the existing
`RunInput`, with each level (model, config, parameters, scenario, seed) a nested
typed struct that derives it too, so `identity()` hashes the whole resolved
semantic configuration rather than a selection of fields. A field added to it is
hashed without anyone remembering to; a field excluded from identity must be
explicitly marked non-semantic. Execution parameters (the thread count) are not
fields of it: they travel in a separate `Execution` value passed to the engine,
and the `execution` differential tests guard that nothing semantic is put there.
This is where completeness is enforced; the surface classification below only
makes sure every input is accounted for on the way in. While the process-wide
evaluator settings still exist, they are set only from it. `sim` does not depend
on `runid`, so the engine cannot be made by type to accept only a resolved
configuration; the single conversion function and the tests are the assurance,
and the documentation says so.

### Classification

A **small derive** on each surface struct requires every field to carry one
declaration and emits an inventory of `(command, input, class)`. A field without
one does not compile. It generates no new configuration types and carries no
test parameters.

| Class          | Meaning                                                               | Verification                                                                                    |
| -------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `keyed(level)` | Can change the result; part of the resolved configuration at a level. | Differential: changed alone, the run id changes.                                                |
| `execution`    | Changes how the result is computed, not what (`--parallel`).          | Differential: changed alone, the run id and stored bytes stay the same. Strong evidence needed. |
| `provenance`   | How the run was asked for (`--label`).                                | Structural: not a field of the resolved configuration, so it cannot reach the engine.           |
| `presentation` | How output is written outside the store (`-o`, time format).          | Structural, plus a check that the leaf is unaffected.                                           |
| `expands`      | Defines the set of runs (`--seeds`, `--sweep`, `--draws`).            | Differential: cells common to both values keep their run ids.                                   |
| `removed`      | Refuses an old spelling with a migration message.                     | The flag fails with the message.                                                                |

A class belongs to a (command, input) pair; a shared flattened group supplies a
default that a command may override. An option that is **explicitly supplied on
the command line** and has no effect in that context (a method-only sampler flag
on another method; `--integrator` on a backend that ignores it) is refused, as
`OverrideFlag::methods` already does (`config_v2.rs:1835`). Values from a shared
configuration file are not refused for being irrelevant to one backend.

**Files** are keyed by resolved value where the store already does so
(parameters, `inputs.rs:172-178`) and by raw bytes otherwise (tables, data).
Raw-byte hashing is conservative — it can re-key two files that describe the
same data with different whitespace — and is a policy, not a semantic identity;
canonicalising parsed data is a possible later step.

### Tests

A **differential test runner** in ordinary Rust takes a command line, runs camdl
into a fresh store, and returns the run id and the stored bytes. Test cases are
ordinary parameterised integration tests, written per input, with whatever base
command line, fixture or companion flags the input needs. One test reads the
derive's inventory and fails if any `keyed`, `execution` or `expands` input of a
converted command has no case. The guarantee is exhaustiveness of classification
and of coverage; the cases themselves are regression evidence.

### Order within phase 3

1. `simulate`, including its ensemble and `mre simulate` (which flattens
   `SimulateArgs`): the resolved configuration, `to_engine_config`, the derive,
   the runner and cases, the environment module (part E).
2. **Measure** before going further: lines removed, hand copies removed, and
   whether the derive earns its place over a plain inventory test built from
   clap's `CommandFactory`. If not, drop the derive.
3. **Remove the process-wide evaluator settings**, before converting more
   commands. `allow_degenerate_rates()` is read inside the evaluators
   (`propensity.rs:155-221, 604`, `flat_eval.rs:434`,
   `resolved_expr.rs:621, 658`, `chain_binomial.rs:813`), and `EvalCtx` is built
   at 36 sites including the high-risk `pgas.rs`, `pgas_grad.rs` and
   `obs_model.rs`. The flag is carried on the compiled model rather than
   `EvalCtx`, so the high-risk files are untouched; `set_licm_disabled`
   (`util.rs:617`) goes the same way; the forcing-extrapolation policy in
   `cli/src/forcing_extrapolation.rs` (four statics and a reporter callback,
   never read by the engine) is returned to the command as a value. Gated on the
   existing byte-identity fixtures.

   **Done in this step:** `allow_degenerate_rates` is `sim::EvalSettings`, a
   field of the compiled model, read by every evaluator from the model it
   evaluates; each command sets it on the models it compiles, and no file that
   builds an `EvalCtx` changed. `simulate` has a per-cell resolved value,
   `sim_config::ResolvedCell`: the engine runs from it and returns it, and the
   written leaf is keyed from that instance, checked against the identity the
   cache-hit check resolved before the cell ran. The forcing-extrapolation
   policy is `forcing_extrapolation::ForcingPolicy`, a value each command builds
   from its resolved flag or fit config and installs for its duration:
   `EngineSettings` holds simulate's, derived once per invocation and kept by
   the CAS sink; `fit run`, `pfilter`, `profile`, `survey`, `compare`, `eval`
   and `fit predict` build their own. It carries the forcings the run held and
   the refusal it kept while it held a claim. The run-note reporter routes
   forcing notes into the installed value through one routing slot, the only
   process-wide state left of it, since `sim::run_notes` is process-wide by
   design; every record names the value through `RecordMeta::new`, and the
   identity readers in `pfilter`, `profile` and `survey` read the command's
   value. The four statics and `set_allowed`/`allowed()` are gone.
   `pfilter --fit`, `profile --fit` and `survey --fit` now read the config's
   `[config] allow_forcing_extrapolation`, and `compare` passes the override to
   its `camdl pfilter` child (gh#1235). **Next:** `--no-licm` is read inside the
   compile helper every command calls, so it is converted per command in step 4,
   as each command's resolved configuration gains its compile settings.
4. `fit run` (classification and cases; its hashing representation is unchanged
   until the follow-up), then `pfilter`, `profile`, `survey`.

The unused typed inputs in `runid/src/inputs.rs` (`PfilterEvalInput`,
`SurveyInput`, `ProjectionInput`, `SyntheticObsInput`, `ProfilePointInput`, no
users in `cli`) are adopted or deleted in their own commit before step 4.

## Part E: development features

**Invariant:** every switch that can change a result is classified like any
other input, and a result produced with an unreleased feature says so.

- **One environment module.** All `CAMDL_*` variables are read once, in one
  `cli` module, into a struct with the same derive and classification; values
  are passed down. A test asserts no other module reads the environment, in the
  manner of `sim/tests/sim_prints_nothing.rs`. `sim/src/trace.rs:108` and
  `resolved_expr.rs:509` move to it.
- **Every environment input is classified by its effect on results**, like a
  flag, and carries a stability tag: `stable`, `experimental` (an unreleased
  feature, enabled by `CAMDL_EXPERIMENTAL_<NAME>` or a hidden flag) or `debug`
  (a diagnostic or alternative evaluation path).
- **A switch that can change results is keyed.** An experimental feature that
  changes results enters the identity like any other keyed input, so results
  with and without it never share a leaf.
- **Alternative evaluation paths are keyed by default.** `CAMDL_EVAL_FLAT`,
  `CAMDL_EVAL_UNRESOLVED`, `CAMDL_NO_BINDING_CACHE` and similar switches are
  meant to be equivalent to the default path, and pass the current differential
  tests, but finite tests do not establish equivalence on every model. They are
  keyed until stronger justification exists, at the cost of some cache reuse; a
  switch moves to `execution` only with that justification recorded.
- **The record says so.** The provenance section of `run.json` lists each
  experimental input set for the run and its value. A leaf made with an
  experimental feature is otherwise an ordinary leaf. (If adding the list
  requires moving the store's `FORMAT_VERSION`, it moves.)

Deferred, not part of this proposal: generated documentation of the switches,
enforcement of a feature's lifecycle (promotion, removal), and showing the
experimental list in `fit summary`, reports and the viewer.

## Follow-up proposal

**The fit's typed identity, replacing canonical JSON.** A fit's configuration
enters its identity as key-sorted JSON (`fit/cas.rs:404`), and the batch
commands hash private serde structs the same way (`canonical_config_hash`,
`fit/cas.rs:209`, in six modules). Replacing it must reproduce the per-algorithm
subtraction of the extension dimension and its re-folding as `target_length`,
with `resume_guard`'s second key (`config_v2.rs:1364-1407`, `cas.rs:529-566`);
`starts` through `identity_value` and `deps`, with `n_trajectories` folded
separately (`cas.rs:454-466`); model and data paths in the hashed blob
(`cas.rs:399-402`); `config_identity_hash` (`cas.rs:438`), canonical JSON used
for `fit.toml` handle lookup; 29 configuration types with a hand-written
`Serialize for FitConfig`; `sim::rng::BinomialAlgorithm`, which cannot implement
`ContentAddressed` from `cli` (orphan rule); and `FiniteF64`, which has no serde
implementation. That needs an inventory and its own design.

**Also named:** a portable execution-environment identity (a runtime fingerprint
of the system maths library and OS) before stores move between machines;
bringing `fit predict`'s output fully into the run store (phase 1e only fixes
its address); generated documentation and lifecycle enforcement for development
switches; and an opt-in `--verify` that recomputes a cached leaf and compares,
which needs phase 2 for fits.

## Re-keying and compatibility

- Phase 1a changes the engine's contribution to every identity: every stored run
  re-keys once. It lands first and alone because it closes a live wrong-answer
  route.
- The other identity changes (1b, 1c, the phase 2 `from_params` effect) land
  together under one `HASH_VERSION` bump.
- Phase 2 changes leaf contents and moves readers, and moves `FORMAT_VERSION`,
  so older leaves read as stale and are recomputed. Part E's provenance list
  moves it too if the record format requires it.
- Phase 1e changes where `fit predict` writes; predictions written under the old
  address are not found by the new one and are recomputed on request.
- Keying the alternative evaluation paths (part E) re-keys only runs made with
  those switches set.
- No `ir/VERSION` bump, no golden changes. The `run_id`-stability tests are
  updated in each identity change.
- Handle resolution becomes stricter (1d): scripts that relied on the fallback
  get an explicit choice to make.

## Plan

Each step lands green on its own, with a red→green test and a mutation check.

| #  | Step                                                                                        | Kind                 | Estimate           |
| -- | ------------------------------------------------------------------------------------------- | -------------------- | ------------------ |
| 1  | 1a engine fingerprint, consumers, fail-closed rule, cross-crate re-run test                 | correctness-critical | 2 days             |
| 2  | 1b gradient maps for gradient-using fits, with the identical-forward-model test             | correctness-critical | 1 day              |
| 3  | 1c the three missing-input bugs                                                             | correctness-critical | 1–2 days           |
| 4  | 1d exact handle resolution                                                                  | correctness-critical | 1 day              |
| 4b | 1e `fit predict`'s output address: red test, full-width key of every output-changing input  | correctness-critical | 1 day              |
| 5  | Phase 2: execution records, readers, byte-reproducibility tests (uninterrupted and resumed) | correctness          | 3 days             |
| 6  | Docs and `--help`: "input-addressed", the three claims, the stated assumptions              | documentation        | 0.5 day            |
| 7  | Phase 3 `simulate` and part E environment module; then measure                              | infrastructure       | 1 week             |
| 8  | Phase 3 removal of the process-wide settings                                                | infrastructure       | 3 days             |
| 9  | Phase 3 `fit run`, `pfilter`, `profile`, `survey`                                           | infrastructure       | sized after step 7 |

Steps 1–6 are about two weeks and can start at once; they do not touch the files
the current date work changes. Steps 7–9 edit `main.rs` and the command modules
and start from a fresh `origin/main` once that work settles.

## Decisions

- Engine identity is a 128-bit build fingerprint under a specified environment
  (source, lock file, the binary's resolved features across crates, profile,
  `rustc -vV`, flags, target), computed once at build time; uncertain identity
  fails closed to "no reuse"; `ENGINE_ID` is separate from `GIT_HASH`.
- The guarantee is stated for one machine image; a runtime environment
  fingerprint is a follow-up.
- `fit predict`'s output address is fixed in phase 1, keyed by every input that
  changes its output, at full width.
- A resumed or extended fit: the computation's identity includes its target
  length; a checkpoint is never served as a completed result; extending a fit
  leaves the shorter fit's leaf intact.
- Compiler changes are covered by hashing the gradient maps a gradient-using
  method consumes, not by a compiler-binary digest.
- Phase 1a lands alone, first; the other identity changes share one
  `HASH_VERSION` bump.
- Handles resolve exactly or fail explicitly; no freshness heuristic.
- A computation's result is separated from its executions; byte reproducibility
  is promised only where tested, for uninterrupted and resumed fits separately.
- Execution and identity derive from one resolved configuration per command,
  with execution parameters in a separate value and one conversion to engine
  settings. For `simulate`, step 1 enforces this for the run-wide configuration:
  `RunConfig` is copied exhaustively into each cell's `SimConfig` (a field added
  to either does not compile until it is carried across), and the engine's
  backend, step and evaluator settings come from the same value. It does not yet
  cover the per-cell inputs — parameter values and table digests, seed, horizon,
  restored origin state. Step 3 adds them: `sim_config::ResolvedCell` is the
  per-cell resolved value the engine runs and returns, and the written leaf is
  keyed from that instance. No generated parallel configuration types; no
  `Identity<T>` or `ContentFile<T>` in this proposal.
- Classification is a table per command, keyed by (command, input), checked
  against clap's `CommandFactory` both ways, with a class column in the
  environment table; tests are ordinary Rust. The derive tried in step 1 was
  dropped: a shared flattened group needs a class per command, which a derive on
  the group cannot express.
- Six classes, verified differently: `keyed`, `execution` and `expands` by
  differential tests; `provenance` and `presentation` structurally; `removed` by
  its message.
- Only explicitly supplied, inapplicable options are refused.
- Differential tests are regression evidence, not proof; the remaining
  assumptions are documented.
- The process-wide evaluator settings are removed after the `simulate`
  conversion and before the other commands, by carrying the flag on the compiled
  model. Step 3 did this for `allow_degenerate_rates`, and then for the
  forcing-extrapolation policy as a per-command policy value held by
  `EngineSettings`, the run-note reporter routing notes into it through one slot
  and the identity readers reading the held value; `--no-licm` is converted per
  command in step 4.
- Environment variables are read in one module and classified by their effect on
  results, with a stability tag (`stable`, `experimental`, `debug`); switches
  that can change results are keyed, alternative evaluation paths included until
  their equivalence is better justified; `run.json` provenance lists the
  experimental features a run used. Generated documentation, lifecycle
  enforcement and display in reports are deferred.
- The fit's typed identity is a follow-up proposal.

## Out of scope

The IR schema; the camdlc version handshake
(`2026-09-10-camdlc-version-guard.md`); renaming types in code; the store's
lifecycle and garbage collection (`2026-08-24-camdl-store-lifecycle.md`); the
follow-ups named above.
