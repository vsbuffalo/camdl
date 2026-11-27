# Workflow bindings: output manifests, file lookup and the result-file contract

Nov 23, 2026 · Vince Buffalo

Builds on: `2026-11-20-run-store-api.md` (the run store) and
`2026-11-23-prediction-and-scores.md` (predictions and scores).

Status: accepted 2026-11-23. Stage 7, `--from-manifest`, is implemented. Claims
about current behaviour are marked _verified_, with the file and line or the
command that checked them, or _inferred_. Commands were run with the `camdl`
built from `bbee1bb13` (and a rebuild of that commit with uncommitted edits,
`0.1.0+bbee1bb13.dirty-e5518506`) in a scratch store; line references are to
`origin/main` at `de896b47c`.

## Summary

camdl's run store names each result by a hash of everything that can change it,
reuses a stored result whenever the same inputs come again, and never overwrites
a completed one. A rerun of an unchanged command costs a lookup, two commands
that ask for the same thing share one result, and every result records what
produced it. The price is that the store, and only the store, decides where a
result lives: its folder is named by the hash, so no program outside camdl can
know the path before the command has run.

This proposal binds the store to workflow tools such as Snakemake and make. A
command that stores results takes `--manifest PATH` and writes there a small
JSON file, the output manifest, that lists the run id of every result it
produced, each result file under a logical name with its SHA-256 digest, the
labels and the camdl build. A rule declares the manifest as its output; a later
step finds a file through `camdl path` or `camdl cat`, which share one resolver,
or through a link tree under `--link-root`. A manifest exists only when every
result it lists completed. Every file it names then exists whole and has the
recorded digest, because the store computes each file's digest as it is written,
records it in `run.json`, and makes the result visible as completed in one
atomic step. A command removes an earlier manifest at its path when it starts,
so a failure leaves none; a file there that is not a manifest is refused and
kept. The run spec gains a result-file contract, generated from typed file
enums, that names each kind's files, formats and columns; the store refuses to
finish a run whose files differ from its declared set; every fit run stores its
summary; and `camdl receive` checks and places results pulled from another
store.

## Motivation

A binding is needed because the store decides each result's path. The path of
the MH fit used below is
`camdl-store/0.1.0_bbee1bb13-8060bb5bb7f9/fit/sir-f7e0ba765b59/runs/mh-efa3faed7db2/seed_7-d0a2f3542755/`
(_verified_ in a scratch store): a build directory named by the ambient key, a
fit problem's folder, a method group and a run folder, each ending in 12 hex
digits of a hash. A Snakemake rule must name its outputs before the command
runs, and Snakemake decides whether to run a rule from those files: whether they
exist, and whether they are older than its inputs. The rule cannot name a path
that only the command can compute.

The London measles analysis runs its fits, filters and predictions from
Snakemake. It will move onto the run store, rerun everything with one pinned
camdl build, and import no old results. The rule they can write today copies a
file out of the store to a path of their own:

```python
rule fit_m4w_mh:
    input:  "fits/m4w_mh.toml", "data/measles_weekly.tsv"
    output: "build/fits/m4w_mh/draws.tsv"
    shell:
        "camdl fit run {input[0]} --label m4w_mh && "
        "camdl cat @m4w_mh --stream draws.tsv > {output}"
```

`fit run` has no `-o` (_verified_: `camdl fit run --help` lists no `--output`),
so a copy through `camdl cat` is the only way to put a stored file at a named
path. The copy is the symptom of the missing binding, and it brings three
defects.

The copy duplicates the store. Every file a downstream rule reads must be copied
out by the rule that made it, and a PGAS fit keeps per-chain trajectory tables
(run spec §2.2) that can be large.

The copy goes stale unnoticed. Nothing ties `build/fits/m4w_mh/draws.tsv` to the
run it came from. If the fit is run again by hand with another configuration, or
the store is cleaned, the copy still looks current to Snakemake.

The label moves. `@m4w_mh` is the latest invocation labelled `m4w_mh` whose
results all completed (_verified_: `label_holders`,
`rust/crates/store/src/storage.rs:1581`). A person at a terminal, or another
rule, that reuses the label moves it, and a later `camdl cat @m4w_mh` can name a
different run from the one the rule made.

Declaring a folder inside the store as the output is not a way out. The folder
is named by the first writer's labels and a hash, and Snakemake deletes a
directory output before its job runs
([Snakemake 9.27.2, rules, "Directories as outputs"](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html)),
which would delete a stored result.

## Working with the binding

The examples use the London analysis's names. Each shows the command lines a
user types; the sections after this one define the pieces.

### A fit, then a plot from it

The fit rule's output is its manifest. The plotting rule reads the manifest and
asks camdl for the path of the draws:

```python
rule fit_m4w_mh:
    input:  "fits/m4w_mh.toml", "data/measles_weekly.tsv"
    output: "build/fits/m4w_mh.json"
    shell:  "camdl fit run {input[0]} --label m4w_mh --manifest {output}"

rule plot_m4w_mh:
    input:  "build/fits/m4w_mh.json"
    output: "figures/m4w_mh_posterior.pdf"
    shell:
        "python plot_posterior.py "
        "$(camdl path --manifest {input} draws) {output}"
```

`draws` is the logical name of a fit run's `draws.tsv`. Naming the manifest
after the label, as here, keeps one name for the result across the label, the
manifest and the link tree; camdl does not require it.

### Several files from one result

`camdl path` takes several names and prints one path per line, in the order
given. `--all` prints every result file of the result, one per line, as the
logical name, a tab and the path:

```bash
$ camdl path --manifest build/fits/m4w_mh.json chain_0/trace chain_1/trace
/data/measles/camdl-store/0.1.0_…/runs/mh-efa3faed7db2/seed_7-d0a2f3542755/chain_0/trace.tsv
/data/measles/camdl-store/0.1.0_…/runs/mh-efa3faed7db2/seed_7-d0a2f3542755/chain_1/trace.tsv

$ camdl path --manifest build/fits/m4w_mh.json --kind fit_run --all
chain_0/trace	/data/measles/camdl-store/…/chain_0/trace.tsv
chain_1/trace	/data/measles/camdl-store/…/chain_1/trace.tsv
chain_starts	/data/measles/camdl-store/…/chain_starts.tsv
diagnostics	/data/measles/camdl-store/…/diagnostics.json
draws	/data/measles/camdl-store/…/draws.tsv
…
```

`camdl cat --manifest build/fits/m4w_mh.json draws` prints the file's bytes,
checked against its digest, for a program that reads standard input.

### A sweep whose manifest holds many cells

A simulation over a grid stores one result per cell, and its manifest lists them
all:

```bash
camdl simulate measles.camdl --sweep 'beta=lin(20,40,5)' --seeds 1:10 \
    --label beta_sweep --manifest build/sims/beta_sweep.json
```

Without `--each`, reading every cell's trajectory needs a JSON tool to pull the
run ids out of the manifest and one `camdl path` call per cell, and the cell's
coordinates are lost on the way:

```bash
jq -r '.results[] | select(.kind == "sim") | .run_id' build/sims/beta_sweep.json |
while read -r id; do
    camdl path "$id" traj
done
```

With `--each`, one call prints a table with a row per cell that holds `traj`:
its run id, each of its labels as a column, and the path.

```text
$ camdl path --manifest build/sims/beta_sweep.json traj --each
run_id	model	params	scenario	replicate	seed	seed_replicate	process_seed	point_id	beta	path
d89132537cac…	measles	beta=20	(none)	0	1	0	1	0	20	/data/measles/camdl-store/…/seed_1-d89132537cac/traj.tsv
6c3aeb035f48…	measles	beta=20	(none)	1	2	0	2	0	20	/data/measles/camdl-store/…/seed_2-6c3aeb035f48/traj.tsv
…
```

A Python step reads that table with polars, or asks `camdl cat --each` for every
cell's rows in one table, each row keeping its cell's coordinates:

```python
import io, subprocess
import polars as pl

def camdl(*args):
    return subprocess.run(["camdl", *args], capture_output=True, text=True, check=True).stdout

m = "build/sims/beta_sweep.json"
cells = pl.read_csv(io.StringIO(camdl("path", "--manifest", m, "traj", "--each")), separator="\t")
traj = pl.read_csv(io.StringIO(camdl("cat", "--manifest", m, "traj", "--each")), separator="\t")
peak = traj.group_by("beta", "seed").agg(pl.col("I").max())
```

The manifest is ordinary JSON, so `jq` or `json.load` read it too, for anyone
who wants its digests or labels directly; no example here needs them.

### Chaining commands

A command that reads a stored fit run takes `--from-manifest M` in place of a
handle. It means the manifest's one `fit_run`, and is refused when the manifest
holds none or several:

```python
rule predict_m4w_mh:
    input:  "build/fits/m4w_mh.json"
    output: "build/predictions/m4w_mh_one_step.json"
    shell:
        "camdl predict --from-manifest {input} --mode one_step "
        "--label m4w_mh_one_step --manifest {output}"
```

`predict`, `simulate` (with `--draws posterior`, in place of `--fit <run>`) and
`profile` (whose `--fit` takes the run's fit problem) accept it. For `pfilter`
it means the run's problem, as `--fit` gives a configuration's, and the run's
`mle_params` as the parameter values. The command checks that each file it reads
from the run has the digest the manifest records, so a step reads exactly the
result the earlier rule produced or fails.

### A link tree

`--link-root DIR`, else `CAMDL_LINK_ROOT`, makes a command also write symbolic
links to its stored files under a readable tree, for a person browsing results
or a tool that wants plain paths:

```bash
$ camdl fit run fits/m4w_mh.toml --label m4w_mh --link-root results
$ ls results/fits/m4w_mh/
chain_0  chain_1  chain_starts.tsv  diagnostics.json  draws.tsv  fit_state.toml  …
```

In a Snakefile the variable is set once, at the top, and every shell command
Snakemake starts inherits it:

```python
import os
os.environ["CAMDL_LINK_ROOT"] = "results"
```

That the jobs of a cluster or cloud executor inherit the variable is _inferred_;
an executor that starts jobs in a clean environment needs `--link-root` in the
command. A `.envrc` setting the variable is read only under direnv, after
`direnv allow`.

## Terms

**Run store.** The directory tree under one root (`--store DIR`, else
`CAMDL_STORE`, else `./camdl-store`) that holds camdl's results, with the
`store` crate as the only code that reads or writes it (_verified_:
`camdl_store::store_request` and `DEFAULT_ROOT`,
`rust/crates/cli/src/camdl_store.rs:97-108`).

**Result.** One stored artifact: a folder holding `identity.json`, `run.json`
and the files it produced. camdl's kinds are `compiled_model`, `quantities`,
`data_table`; `sim` and its by-products `obs`, `event_log`, `sim_quantities`;
`pfilter` and its by-products `trace`, `pf_health`, `filtering`,
`filtered_streams`, `final_state`, `saved_paths`, `prequential`; `fit` (a fit
problem), `fit_run`, `prediction`, `score` (after the scores proposal's stage
3); and `profile_point` (_verified_: `camdl_kinds`, `camdl_store.rs:79-83`, and
the `KIND` constants of `sim_store.rs`, `pfilter_store.rs`, `fit_store.rs`,
`profile_store.rs`). A by-product is a second result of the same computation,
with its own run id, stored inside its upstream's folder.

**Run id.** The SHA-256 of the hash version (9), the kind's name, the ambient
key and the identity key (_verified_: `run_id`, `store/src/artifact.rs:87-94`).
The identity key hashes the result's typed identity, which holds everything that
can change its bytes, the digests of the files it depends on included. The
ambient key hashes the build fingerprint and the environment switches that can
change a result (_verified_: `Engine`, `camdl_store.rs:31-70`). A result
folder's name ends in the run id's first 12 hex digits.

**Handle.** What a command line types to name a stored result: a path to its
folder, a run id or a prefix of one, `@L` or `@L~n` (_verified_:
`Handle::from_str`, `store/src/handle.rs`).

**Label.** Anything a user recognises a result by that is not in its identity: a
`--label`, a scenario's name, the seed typed. Labels live in invocation records,
never in a result's own record.

**Invocation record.** `.invocations/<seq>.json`: the store's record of one
command run, with its `argv`, the inputs as given, the `--label` and the ordered
list of results it passed to the store, each with its kind, run id, labels and
outcome (`computed`, `reused`, `failed`, `busy`) (_verified_: `InvocationFile`,
`store/src/record.rs`). `seq` is a counter local to the store.

**Result file, provenance file, attempt file.** The three roles of a file in
`run.json` (_verified_: `Role`, `record.rs`). A result file is determined by the
identity and compared on a recompute (`draws.tsv`). A provenance file is
recorded with its digest and not compared (`execution.json`, which holds wall
times). An attempt file describes one attempt and has no digest
(`progress.json`).

**Store manifest.** The `files` table of a result's `run.json`: each result and
provenance file with its size, modification time and SHA-256.

**Output manifest.** The file a command writes at `--manifest PATH`, listing
what that command produced. The name keeps it apart from the store manifest (one
result's files) and from the invocation record (the store's own record of a
command, with times and outcomes).

**Logical name.** A result file's name in the file contract: the stored name
without its extension (`draws`, `chain_0/trace`, `forecast/cases`), unique
within its kind. The qualified form names the kind too, `fit_run:draws`.

**Build fingerprint, `ENGINE_ID`.** A 128-bit digest compiled into the binary
(_verified_: `fingerprint`, `cli/src/engine_fingerprint.rs:104-171`). It covers
the package version; the source state (the commit and a digest of uncommitted
changes to the build inputs, or without git a digest of all of them); the
SHA-256 of `rust/Cargo.lock`; the resolved features of `cli` and `sim`; the
build profile, including every `CARGO_CFG_*` and `CARGO_PROFILE_*` variable the
build script sees; `rustc -vV`; `CARGO_ENCODED_RUSTFLAGS`; and the target
triple. A component that cannot be determined makes the build unreusable
(`Build::Incomplete`, `camdl_store.rs:55-70`).

In the store's records "build" holds two values. `identity.json`'s
`ambient.build` is `ENGINE_ID`, 32 hex digits; a dependency's `build` is the
ambient key, 64 hex digits (_verified_ in a scratch store). The output manifest
names them `engine_id` and `ambient_key`.

**Cache hit.** A request whose run id is already stored and completed under this
root. The invocation record lists it as `reused`, and the command prints
`cache hit — reusing <folder>` (_verified_).

**File contract.** The reference section of the run spec that names, for each
kind, every file by its logical and stored name, with its format, its version
where it has one, and the meaning of its columns or fields. A consumer reads by
it; the layout and the folder names are outside it.

## The seam between camdl and the workflow tool

camdl and the workflow tool each answer one question. camdl answers what a
result is and whether it is already stored: it derives the run id, reuses or
computes, and records what it did. The workflow tool answers whether to invoke
camdl now: it compares declared inputs and outputs and decides which rules run,
in which order. camdl keeps no graph of rules. Snakemake never judges whether a
result is reusable; it invokes camdl, and a cache hit makes a needless
invocation cheap.

```text
workflow tool                                camdl
rule inputs ───────────invoke──────▶  identity → run id; reuse or compute
                                      write output manifest ─▶ build/x.json
next rule's input: build/x.json
next rule's shell: camdl path --manifest build/x.json draws ─▶ a path to read now
```

Lookup is by run id and logical name. A run id names one result for good, and a
logical name is a name in the contract; neither depends on where the folder
sits. A step never records a path into the store, so a moved root, a reindex or
a layout change leaves the workflow as it was. A label is accepted by
`camdl
path` and `camdl cat` for a person at a terminal, but a manifest never
holds one as a reference.

The workflow keeps one duty. Snakemake decides whether to invoke camdl from the
inputs a rule declares, so a rule that omits a data file its fit configuration
names does not rerun when that file changes, although camdl would recompute if
invoked.

## The output manifest

### Format

```json
{
  "schema": "camdl.output-manifest/v1",
  "camdl": {
    "version": "0.1.0+bbee1bb13",
    "engine_id": "8b2f0fed2f47ecd94020cecdcbe32d1a",
    "ambient_key": "8060bb5bb7f92951735cf0743663953323fce42b23b287dcb7291593fda6d5a2"
  },
  "label": "m4w_mh",
  "results": [
    { "kind": "quantities", "run_id": "9ed5a122c7c8…", "upstream": null,
      "contract": 1, "labels": { "name": "m4w" },
      "files": { "quantities": { "file": "quantities.json", "sha256": "…", "bytes": 214 } } },
    { "kind": "compiled_model", "run_id": "594655ed7b95…", "upstream": null,
      "contract": 1, "labels": { "name": "m4w" },
      "files": { "model": { "file": "model.ir.json", "sha256": "…", "bytes": 48211 } } },
    { "kind": "fit", "run_id": "f7e0ba765b59…", "upstream": null,
      "contract": 1, "labels": { "model": "m4w" },
      "files": { "partition": { "file": "partition.json", "sha256": "…", "bytes": 312 },
                 "problem":   { "file": "problem.toml",   "sha256": "…", "bytes": 901 } } },
    { "kind": "fit_run", "run_id": "d0a2f3542755…", "upstream": null,
      "contract": 1, "labels": { "method": "mh", "quantities": "9ed5a122c7c8…", "seed": 7 },
      "files": { "chain_0/trace": { "file": "chain_0/trace.tsv", "sha256": "…", "bytes": 5520 },
                 "draws":         { "file": "draws.tsv",         "sha256": "…", "bytes": 9312 },
                 "fit_state":     { "file": "fit_state.toml",    "sha256": "…", "bytes": 1204 } } }
  ]
}
```

The entries are those the invocation record of a `fit run` lists, in its order
(_verified_: a scratch store's `.invocations/1.json` lists `quantities`,
`compiled_model`, `fit`, `fit_run`). The example abridges the file lists; run
ids are always 64 hex digits.

### Types

The store library gains a kind-agnostic read of an ended invocation. It needs
only each result's record, never its typed identity:

```rust
// rust/crates/store/src/invocation.rs
/// One result an ended invocation listed, as a consumer outside the store reads it.
pub struct Output {
    pub kind: String,
    pub run_id: Key,
    /// The result this one is a by-product of.
    pub upstream: Option<Key>,
    pub labels: serde_json::Value,
    /// Every file of role `Result` in the result's `run.json`, by stored name.
    pub files: BTreeMap<String, OutputFile>,
}

pub struct OutputFile { pub sha256: Key, pub bytes: u64 }

impl Invocation {
    /// Each distinct (kind, run id) this invocation listed, in first-listed order.
    /// Refused with Error::MemberNotCompleted, naming each, when any entry did
    /// not end `computed` or `reused`.
    pub fn outputs(&self, store: &Store) -> Result<Vec<Output>>;
}
```

camdl adds its build, the logical names and the format:

```rust
// rust/crates/cli/src/output_manifest.rs
#[derive(Serialize)]
pub(crate) struct OutputManifest {
    pub schema: &'static str,              // "camdl.output-manifest/v1"
    pub camdl: BuildRecord,
    pub label: Option<String>,
    pub results: Vec<ManifestEntry>,
}

#[derive(Serialize)]
pub(crate) struct BuildRecord {
    pub version: &'static str,             // version::VERSION_SHORT
    pub engine_id: &'static str,           // version::ENGINE_ID
    pub ambient_key: Key,
}

#[derive(Serialize)]
pub(crate) struct ManifestEntry {
    pub kind: String,
    pub run_id: Key,
    pub upstream: Option<Key>,
    pub contract: u16,
    pub labels: serde_json::Value,
    /// By logical name.
    pub files: BTreeMap<String, ManifestFile>,
}

#[derive(Serialize)]
pub(crate) struct ManifestFile { pub file: String, pub sha256: Key, pub bytes: u64 }
```

`--manifest` is an export-class input in the store's classes: it names a path
the command writes and is never part of an identity, so adding it to a command
re-keys nothing. The manifest is written with `store::write_file_atomic`
(_verified_: `store/src/fsutil.rs:68-80`): a temporary file beside the target,
synced, renamed over it, then the directory synced.

### What the manifest holds, and what it leaves out

The manifest's bytes are a function of the ordered results, their labels, their
result files' names, digests and sizes, the `--label` and the build. It holds no
time, invocation `seq`, outcome, host or process, so a cache hit writes the
bytes the computation wrote. It holds no path, neither the store root nor a
folder, so one command against two roots writes one manifest. It lists no
provenance or attempt file: a `--force` recompute can change `execution.json`
without changing the result, and listing its digest would make every downstream
step rerun. Provenance files and the two records remain reachable by name
through `camdl path`.

Because the root is not in the manifest, a reader opens the store as every
reading command does, `--store`, else `CAMDL_STORE`, else `./camdl-store`. A
manifest read against another store fails on its first run id.

### Which commands take `--manifest`, and what each lists

Every command that stores results takes it: `simulate`, `pfilter`, `fit run`,
`predict`, `profile`, `score`, `receive`, and `compare`, which stores the
predictions and scores it differences. Each lists every distinct result its
invocation record lists, in order, each by-product as an entry of its own with
its `upstream`; `compare` lists each prediction and score it read or stored,
each prediction before its score, and takes `--manifest` only when they all sit
in one run store. From the invocation records of a scratch store (_verified_):

- `simulate` lists the compiled model, then each cell (`sim`) followed by its
  `obs`, `event_log` and `sim_quantities` when asked for.
- `pfilter` lists the compiled model, each `data_table`, the `pfilter` run and
  each by-product its flags asked for.
- `fit run` lists `quantities`, `compiled_model`, the fit problem and the
  `fit_run`.
- `predict` lists the `prediction` alone; the fit run it reads is a dependency,
  not a member.
- `profile` lists `quantities`, `compiled_model`, the fit problem and each
  `profile_point`.

By-products are listed because files a consumer wants live in them:
`filtered_streams.tsv` and `pf_health.tsv` belong to by-products of the filter
run, not to the run (_verified_: `run_by_product!`, `pfilter_store.rs`). A
simulated cell that stores `failure.toml`, a run that stopped for a reason the
model defines, is a completed result, but `simulate` exits 1 when any cell
stored one, so that command writes no manifest. A later command that reuses the
cell lists it only if it succeeds.

### When the manifest is written and removed

A manifest exists only when every result it lists completed. Each result file's
digest is computed from the bytes as they are written and synced
(`FileHandle::close`, `store/src/run.rs:85-92`) and recorded in `run.json` when
the run finishes. An atomic kind (`sim`, `prediction`, every by-product) is
written in a staging folder and renamed into place by that one rename; an
in-place kind (`fit_run`, `pfilter`, `profile_point`, _verified_: their `WRITE`
constants) becomes completed when its `run.json` is replaced through a temporary
file and a rename (`run.rs:778-782`). Either way a reader sees a result as
completed only after all its files are whole, and the manifest is written after
the invocation ends with every entry `computed` or `reused`.

As soon as its arguments have parsed, before its first call to the store, the
command removes an earlier manifest at the manifest path: an empty file, or a
JSON file whose `schema` begins `camdl.output-manifest/`. A file of any other
content is refused with exit 1, "PATH holds a file that is not an output
manifest", and left as it is, so a mistyped `--manifest data.tsv` deletes
nothing. A failed command, a batch with a failed or busy result, and a killed
process therefore leave no manifest. A usage error caught by argument parsing
(exit 2) leaves the path as it was.

On success the manifest is always written, a cache hit included, through the
atomic rename. With Snakemake this is required: Snakemake removes a job's
outputs before running it and touches them after it completes, "to ensure they
are newer than the input"
([Snakemake 9.27.2 FAQ, "Can the output of a rule be a symlink?"](https://snakemake.readthedocs.io/en/stable/project_info/faq.html)).
The identical bytes keep downstream rules from rerunning, since for an input
under 1 MB by default Snakemake "calculates a checksum of the file content and
only reruns the rule if that checksum has changed, even if the timestamp of the
input file is newer than the output file(s)" (same FAQ, "Snakemake wants to
rerun a rule that has been already executed, what can I do?"). How this
interacts with the `input` trigger of the default `--rerun-triggers` is
_inferred_.

With make the rewrite is what lets a build converge. make remakes a target older
than a prerequisite, and a recipe that leaves its target untouched leaves it
older, so make would invoke camdl on every later run. _Verified_ with GNU Make
3.81: a rule whose recipe does not modify an existing target ran on both of two
`make` calls after `touch` of its prerequisite. With the rewrite, a no-op rerun
of the fit refreshes the manifest's time, and the next `make` finds the tree up
to date.

## Looking up a stored file

### One resolver for `path` and `cat`

`camdl path` and `camdl cat` resolve a file through one function and differ only
in what they print. `path` prints the absolute path of each stored file named;
`cat` prints a file's bytes, read through the store and checked against the
recorded digest (_verified_ for the store read: `Stored::open`,
`store/src/stored.rs:112-116`).

```text
camdl path <HANDLE> <NAME>...            camdl cat <HANDLE> [<NAME>]
camdl path --manifest <M> <NAME>...      camdl cat --manifest <M> <NAME>
           [--kind K] [--run ID]                   [--kind K] [--run ID]
           [--all | --each]                        [--each]
```

```rust
// rust/crates/cli/src/resolve_file.rs
pub(crate) enum Source<'a> { Handle(&'a Handle), Manifest(&'a OutputManifest) }

pub(crate) struct Narrow { pub kind: Option<String>, pub run: Option<String> }

/// The stored file `name` names, of the one result `source` and `narrow` select.
pub(crate) fn resolve_file(store: &Store, source: Source, narrow: &Narrow, name: &str)
    -> Result<ResolvedFile, FileRefusal>;

pub(crate) struct ResolvedFile {
    pub located: store::Located,
    pub file: FileRef,          // FileRef::Stored(..) or FileRef::View(..)
    pub stored_name: String,
    pub digest: Key,
}
```

`resolve_file` sits on a kind-agnostic lookup in the store library, since a file
lookup needs a result's record and not its typed identity:

```rust
// rust/crates/store/src/stored.rs
pub struct Located { /* run id, kind, folder, run.json's files */ }

impl Store {
    /// The completed result `run_id` names, of any kind registered at open.
    pub fn locate(&self, run_id: Key) -> Result<Located>;
    pub fn locate_handle(&self, handle: &Handle) -> Result<Located>;
}

impl Located {
    pub fn kind(&self) -> &str;
    pub fn file(&self, stored_name: &str) -> Result<(PathBuf, Key)>;
    pub fn read(&self, stored_name: &str) -> Result<Vec<u8>>;   // checked against the digest
}
```

Today `cat` finds a result by trying each kind's resolver in turn: filter run,
profile point, simulation cell, then fit run, fit problem and prediction, each
through a typed `Store::resolve::<A>` (_verified_: `cmd_cat`,
`rust/crates/cli/src/browse.rs:1004-1094`; `resolve_stored_fit`,
`browse.rs:573-601`). This proposal replaces that chain with `resolve_file` in
its stages. `cat` keeps what only a person needs: with no name it prints its
kind's default file (`traj` for a cell, `loglik` for a filter run), and `cat @L`
on a multi-cell invocation renders the combined table. The `.tsv` completion of
a bare name (`browse.rs:710-716`) is removed, since logical names have no
extension.

### Names

The candidates are, with a manifest, every entry, and with a label, every
completed result its invocation lists, each narrowed by `--kind` and `--run`.
With a run id, a prefix or a folder, the candidate is the result the handle
names, and then the by-products stored for it: a cell's handle reads its
observation streams, as `cat` does today. Every step below is tried on the named
result before any by-product, so a name the result holds is never ambiguous with
one of theirs.

A name is resolved in five steps, the first that any candidate answers deciding.

1. A qualified name, `fit_run:draws`, names that kind's file.
2. A bare logical name, `draws`, names the one candidate result that has it.
3. An exact stored name, `draws.tsv`, is accepted wherever a logical name is,
   and `identity.json` and `run.json` name the records.
4. With a handle, a path through a result nested in a candidate's folder,
   `predict/forecast-<id>/forecast/cases`, names a file of that result, so the
   names `show` lists for a fit run's predictions are names `cat` takes. Each
   folder of the path is a plain name (no `..`, `.`, empty or absolute
   component), and the nested result's identity must name the candidate as its
   parent or an ancestor; the folder layout alone never admits a result. A
   manifest names only its own entries, so the step is not taken through one: a
   prediction made after the manifest was written is not reached through it.
5. Anything else is refused.

A listed result the store holds but cannot read, damaged or unfinished, fails a
lookup only when it could answer the name: when it is of the kind of a result
that does, so that an answer without it could leave it out, or when nothing else
answers.

When more than one manifest entry has the name, the command refuses and lists
the candidates:

```text
$ camdl path --manifest build/sims/beta_sweep.json traj
error: 50 results in build/sims/beta_sweep.json have `traj`: sim d89132537cac, sim 6c3aeb035f48, …; pass --run for one, or --each for all
```

The refusal stands whenever `--each` is not given: a bare name held by several
results never picks one of them.

With `--manifest`, the resolver also checks that the stored file's digest in
`run.json` equals the manifest's, so a manifest that describes other bytes than
the store holds is refused. `cat` hashes the bytes it prints as well.

### One row per result: `--each`

`--each` takes a manifest or a label and one name, and acts on every entry that
holds the name, narrowed by `--kind` and `--run`, in the manifest's or the
invocation record's order. A label lists results as a manifest does, for a
person at a terminal; a workflow reads through manifests. `--each` is refused
with a run id, a prefix or a folder, which name one result, and with more than
one name.

`camdl path --each` prints a TSV with one row per such result:

- `run_id`, the full 64 hex digits. A 12-digit prefix reads better but can be
  ambiguous, which is why the store gives a folder its whole run id when two
  share 12 digits (run-store proposal, "How a run id becomes a folder"); the
  column is meant to be passed back to camdl as a handle.
- One column per label key the results carry, then `path`.

The label columns are the union of the results' label keys, in the order they
are first seen, with an empty value where a result lacks a key. Each kind's
labels type says how its labels flatten into columns, beside its file enum:

- a scalar field is a column under its field name (`model`, `scenario`, `seed`,
  `replicate`);
- a sweep point (`PointLabel::Sweep`, `sim_store.rs:167-175`) gives `point_id`
  and one column per swept parameter, named as `--sweep` names it, so an indexed
  parameter is underscore-joined (`beta_north`), in `--sweep`'s name order;
- a draw (`PointLabel::Draw`) gives `draw`;
- a swept parameter whose name equals another label key (a parameter called
  `seed`) is written `param.<name>`;
- a nested value with no rule of its own is written as compact JSON.

`camdl cat --each` prints the rows of every such file as one TSV: the same label
columns prepended to each file's own columns, so every row keeps its cell's
coordinates. It reads TSV files only; a name whose files are JSON or TOML is
refused, naming the format. Leading `#` comment lines are dropped. One header is
printed. When a file's header differs from the first file's, the command refuses
before printing a row, naming the first result whose header differs and both
headers. A label column whose name equals a column of the file is written
`label.<name>`.

Each file is read through the store and checked against its digest, as `cat`
reads one file.

The path printed is absolute and is meant to be read at once, as the `$(...)` in
the rules above does. A workflow does not record it.

### Exit codes and refusals

`path` and `cat` exit 0 when every name resolved, 1 on any refusal, and 2 on a
usage error from argument parsing. A refusal names what was asked and what
exists: no result for a handle or a manifest's run id; no such name in it; more
than one candidate; a stored digest that differs from the manifest's; a file
whose bytes no longer match. `camdl cat` exits 1 today for a missing name and a
missing run id, and clap exits 2 on a usage error (_verified_ in a scratch
store).

`path` refuses a view and an attempt file's name: a view has no stored file, and
an attempt file has no digest.

### Labels

`camdl path @L NAME` and `camdl cat @L NAME` work, for a person exploring. A
manifest's entries carry run ids; its `label` field records the `--label` the
command was given and is never resolved. A workflow that reads through manifests
therefore never depends on a label.

## Links

`--link-root DIR`, else `CAMDL_LINK_ROOT`, else no links. With a root, a command
that succeeds writes, beside its manifest, one symbolic link per result file of
each result it lists, at

```text
<root>/<group>/<label>/<logical name>.<ext>
```

The group is the kind's family: `fits` (a `fit_run`), `sims` (a cell), `filters`
(a `pfilter` run), `predictions`, `profiles` (a `profile_point`) and `scores`. A
by-product's files sit with its upstream's, under their logical names, so a
filter run's `filtered_streams.tsv` links as
`results/filters/<label>/filtered_streams.tsv`. Content artifacts and fit
problems get no links. The label is the invocation's `--label`, and without one
the result's run id's first 12 hex digits. When one invocation has several
results of a group, each sits one level deeper, under its run id's first 12 hex
digits: `results/sims/beta_sweep/d89132537cac/traj.tsv`.

A relative link root resolves against the working directory, as `CAMDL_STORE`
does. A link's target is relative when the store root and the link root share a
parent directory below the filesystem's root, so moving that directory keeps the
links working; otherwise it is absolute. A command builds the label's directory
beside its final place and swaps it in, so a reader sees the old links or the
new ones and no link from an earlier run with other files survives.

Links under a root point at the results of the latest invocation with that label
that succeeded with that link root. A command run without a link root leaves an
earlier tree in place, so its links can name an earlier invocation than `@label`
does. A command removes its label's link directories in every group when it
starts, as it removes its manifest, so a failed or killed run leaves no links
that look current. A link whose target the store no longer holds dangles visibly
(opening it fails); rerunning the command, a cache hit, writes it again. Links
are for reading. A workflow step that must name exactly what an earlier step
made depends on the manifest, whose run ids and digests do not move when a label
is reused.

Links are an export, like the manifest: they are never part of an identity and a
command with them reuses what a command without them stored. camdl runs on Linux
and macOS, where symbolic links need no privilege; Windows is unsupported.

## Result files as types

### Logical names from one enum per kind

Each kind's result files are Rust enums in camdl. Their variants are the files
the kind can store, and three methods give a variant's logical name, its stored
name and its columns:

```rust
pub(crate) trait ResultFile {
    fn logical(&self) -> String;              // unique within the kind
    fn stored(&self) -> String;
    fn columns(&self) -> &'static [Column];
}
```

`SimFile`, `PfilterFile`, `PredictionFile`, `ScoreFile` and the rest are one
enum each. A fit run's files depend on its method, so `FitRunFile` is split by
method family, below. Two IF2 files named `diagnostics` today,
`diagnostics.json` and `diagnostics.tsv`, get distinct logical names,
`diagnostics` and `iteration_diagnostics`, because a logical name must be unique
within its kind.

### Fit-run files by method family

A fit run's file enum has one variant per method family, each wrapping that
family's own enum, so a file belongs to the methods that write it and to no
other:

```rust
// rust/crates/cli/src/result_files/fit_run.rs
pub(crate) enum FitRunFile {
    Common(CommonFile),
    Mcmc(McmcFile),
    Pgas(PgasFile),
    If2(If2File),
    Nlopt(NloptFile),
    Pfilter(PfilterMethodFile),
}

/// Every method: method.toml, summary.json, inputs/starts.<ext>.
pub(crate) enum CommonFile { Method, Summary, StartsInput(Ext) }

/// mh, pmmh, pgas.
pub(crate) enum McmcFile {
    Draws, ChainTrace(u32), ResumeState(u32), SamplerSummary,
    ChainStarts, Diagnostics, FitState,
}

/// pgas, beside its McmcFile set.
pub(crate) enum PgasFile { Trajectories(u32), TrajectoriesMeta(u32), CrossChainCompat, FilterEss }

/// if2.
pub(crate) enum If2File {
    ParameterTraces(u32), ChainFinalParams(u32), FinalParams, MleParams,
    ChainEvaluations, IterationDiagnostics, ChainStarts, Diagnostics, FitState,
}

/// nl-sbplx, nl-bobyqa, nl-lbfgs.
pub(crate) enum NloptFile { ChainResults, MleParams, Diagnostics, FitState }

/// pfilter as a fit method.
pub(crate) enum PfilterMethodFile { Logliks, PrequentialJoint, PrequentialStreams, PrequentialRecord }
```

Every file of the contract's fit-run list is in exactly one family's enum, or in
several under one logical name where several families write it. The duplicates
follow the per-method pins (_verified_: `fit_command/tests.rs:740-830`):
`mle_params` is written by IF2 and NLopt; `diagnostics` and `fit_state` by every
family except the pfilter method, which writes neither; `chain_starts` by IF2
and the MCMC methods, and not by NLopt. PGAS gets its own family beside
`McmcFile` so that an MH run cannot name a trajectory file either.

Each method's writer takes only `CommonFile` and its own family's enums:

```rust
pub(crate) trait FitMethod {
    /// The method's own files: McmcFile for mh and pmmh, PgasOwn (McmcFile or
    /// PgasFile) for pgas, If2File, NloptFile, PfilterMethodFile.
    type Own: ResultFile;
}

pub(crate) enum MethodFileRef<O> { Common(CommonFile), Own(O) }

/// The run a method writes into.
pub(crate) struct MethodOutput<'r, M: FitMethod> { run: &'r store::Run<FitRun>, /* … */ }

impl<M: FitMethod> MethodOutput<'_, M> {
    pub fn create(&self, file: MethodFileRef<M::Own>) -> store::Result<store::FileHandle>;
}
```

An MH writer holds a `MethodOutput<Mh>`, whose `Own` is `McmcFile`, so a line
that opens `mle_params` there does not compile. Today a method writes whatever
it likes into a temporary working directory, which `ingest` copies into the run
whole (_verified_: `ingest` and `role_of`,
`rust/crates/cli/src/fit_command.rs`); the methods move onto `MethodOutput`, and
`ingest` goes. The store's expected-set check below still runs at `finish`, for
the files a writer declared and then failed to write.

The cost is six small enums in place of one, and a file that several families
write, such as `diagnostics`, is a variant in each, decided per family. The
contract lists files per method, so the duplicates are what it already says.

Name lookup reads the run's method from its identity (`MethodSettings`'s
`algorithm`, `fit_store.rs:264-269`) and parses the name against `CommonFile`
and that method's enums only. A name another method has is refused, naming the
run's method and listing its files:

```text
error: an mh run has no `mle_params`; its files are chain_0/resume_state,
chain_0/trace, chain_1/resume_state, chain_1/trace, chain_starts, diagnostics,
draws, fit_state, method, sampler_summary, summary
```

### The file set follows the method

Which files a fit run stores depends on its algorithm, its number of chains and
a few settings, and is known before the run starts. Each algorithm returns its
own set:

```rust
pub(crate) trait MethodFiles: FitMethod {
    fn files(&self, n_chains: u32) -> Vec<MethodFileRef<Self::Own>>;
}
```

`files` is computed at plan time from the typed `config_v2::Algorithm`
(`rust/crates/cli/src/fit/config_v2.rs:992`), whose variant selects the
`FitMethod`. It cannot be computed from the stored identity, because
`MethodSettings`, the fit run's method part, holds the method as two strings,
`algorithm: String` and `settings`, canonical JSON of the rest of `[method]`
(_verified_: `fit_store.rs:264-269`). That is a defect to fix separately:
method-scoped settings belong in a typed identity, one variant per algorithm, as
the rest of the store's identities are.

The settings that change the file set are these:

- PGAS's `n_trajectories`, the number of posterior paths saved: zero stores no
  `chain_N/trajectories` (`fit/pgas.rs:71`); the default is 200
  (`config_v2.rs:2010`).
- The pfilter method's `record_prequential`, which stores the `prequential*`
  files and defaults on (`config_v2.rs:2086`).
- NLopt's search trace, `opt_trace.tsv`, which every NLopt run writes as a
  provenance file, since its rows depend on wall-clock time
  (`fit/opt_trace.rs:16-21`).
- Resume, and a starts file named directly: a starts file is stored as
  `inputs/starts.<ext>` (`fit_command.rs`, `store_start_file`).

The backend does not change the file set (_inferred_ from the per-method pins,
in which each method ran on one backend).

### The store enforces the declared set

The store library stays free of camdl's types. It takes the expected set as file
names:

```rust
// rust/crates/store/src/run.rs
pub struct ExpectedFiles {
    pub result: BTreeSet<String>,
    pub provenance: BTreeSet<String>,
}

impl<A: Artifact> Run<A> {
    /// The files this run must hold when it finishes.
    pub fn expect(&mut self, files: ExpectedFiles);
}

// Error::FileSetMismatch { run_id: Key, missing: Vec<String>, unexpected: Vec<String> }
```

`finish` compares the files written with the expected set and refuses with
`FileSetMismatch`, naming the missing and the unexpected files, so a method that
writes an undeclared file fails its run instead of storing a file the contract
does not name. The same holds for a by-product's run (`ByProductRun::expect`).

No command can skip the check. `finish` refuses a run that was given no expected
set, with `Error::NoExpectedFiles`, so a command that forgot to declare its
files fails on its first test run, not in production. camdl declares the set in
one place: the helper through which every camdl command begins a run on the
store computes the set from the kind's file enum (for a fit run, from
`MethodFiles::files` and whether a starts file was named) and calls `expect`
before handing the run to the command. A content artifact stored by `Store::put`
holds the one file its `ContentArtifact::FILE` names and its provenance files,
which `put` already knows, so `put` sets the expected set itself.

### Three routes for an option that changes outputs

An option that changes what a command produces takes one of three routes.

- An identity input changes the file set through `files()`, as `n_trajectories`
  does, and re-keys the result.
- A by-product is a kind of its own, with its own run id and file enum, as each
  of pfilter's `--trace`, `--pf-health` and `--save-*` outputs is.
- A presentation option, such as a date column or another file format, is a
  conversion made on export (`-o`, `cat`). It is never a stored file.

### Views

A view is a file computed when it is read, from a result's stored files: a
`trace` combined across chains, for example.

```rust
pub(crate) enum FileRef<F, V> { Stored(F), View(V) }
```

`cat` and `-o` serve views; `path` and links refuse them, since a view has no
stored file to point at. No views ship with this proposal. They go in each
kind's file module, beside its enum, when one is needed.

### The contract is generated from the enums

The run spec's result-file section is generated from the enums: each variant
gives its logical and stored names, its format and its `columns()`. A make
target writes the section, and a test fails when the committed section differs
from the generated one. A second test,
`every_stored_kind_matches_the_file_contract`, runs each command on the fixtures
`fit_command/tests.rs` already uses and checks every stored file's header row
and `schema` value against its variant.

### The store crate stays extractable

The `store` crate names no camdl type (_verified_:
`grep -rnE 'ir::|sim::|camdl' rust/crates/store/src` matches only comments). It
depends on clap (`rust/crates/store/Cargo.toml:20`), used by the store's own
subcommands (`store/src/commands.rs`) and by the input derive
(`store/src/inputs.rs`). A `clap` feature, on in camdl, would make both
optional, so a project without clap can use the store; how much of the input
derive survives without it is _inferred_ to be the file and environment inputs.

## Versioning

Three numbers guard a stored result, each for a different reader.

- **A file's schema** says how to read one file. It is a `schema` string inside
  a JSON file, such as `camdl.predictive/v4`, `camdl.forecast/v4`,
  `camdl.observed/v1` and `camdl.predict-report/v2` (_verified_:
  `fit/predict.rs:3709`, `:3788`). The prediction schemas become v5 with the
  scores proposal's stage 1, which changes their tables' columns. TSV and TOML
  files carry no schema. A program that parses the file reads it.
- **A kind's contract** says which files a kind stores, under which names, with
  which columns. It is new: `contract` in each manifest entry, a constant per
  kind, bumped when a file is added, removed or renamed, or a column changes. A
  workflow reads it to know whether its scripts still match.
- **An identity version** says which results are the same computation. It exists
  (`#[identity(tag = …, version = N)]`), is hashed into every run id, and is
  bumped when the same inputs would produce different stored bytes. The store
  reads it, and a bump makes every earlier result unreusable.

A change to a file's format changes stored bytes, so it bumps the identity
version and the contract together; the generated-contract test checks that a
change to a variant's columns comes with the contract bump. An identity bump
alone, for a change in what is computed, leaves the contract as it was.

## The result-file contract

The tables below are the section's content as the code stands, with the columns
the scores proposal's stage 1 adds. File lists were _verified_ from `run.json`
in a scratch store and, for fits, against the per-method pins of
`each_method_computes_what_it_computed_before_the_move`
(`rust/crates/cli/src/fit_command/tests.rs:697-830`). Each entry gives the
logical name, then the stored name and format.

Two properties hold across kinds. Several TSV files open with a comment line
before their header (`# 0.1.0+bbee1bb13` in `chain_N/parameter_traces.tsv`,
`chain_evaluations.tsv`, `diagnostics.tsv`; `# camdl chain_starts; …` in
`chain_starts.tsv`), so a reader passes a comment prefix of `#`. In `draws.tsv`
fixed parameters are written in 17-digit exponent form
(`5.00000000000000000e-1`) and estimated ones in shortest round-trip form.

### Fit run (`fit_run`)

- `draws`, `draws.tsv` (mh, pmmh, pgas): `chain`, `draw`, then every parameter,
  estimated and fixed. The thinned post-warm-up draws of all chains.
- `chain_N/trace`, `chain_N/trace.tsv` (mh, pmmh, pgas): `step`,
  `log_likelihood`, `log_posterior`, `accepted`, then the estimated parameters.
- `chain_N/parameter_traces`, `.tsv` (if2), after a comment line: `iteration`,
  `loglik`, `if2_perturbed_loglik`, then the estimated parameters.
- `chain_N/final_params`, `.toml` (if2): one chain's final estimate.
- `chain_N/trajectories`, `.tsv`, and `chain_N/trajectories_meta`,
  `chain_N/trajectories.json` (pgas, when `n_trajectories` is above zero).
- `chain_N/resume_state`, `.bin` (mh, pmmh, pgas): internal state for
  `--resume`; named, with no format in the contract.
- `mle_params`, `mle_params.toml` (if2, nl-\*): parameters at the top level,
  then a `[provenance]` table (run spec §2.2). The point estimate, read by
  `--params`.
- `final_params`, `final_params.toml` (if2): the winning chain's estimate, the
  same values as `mle_params`.
- `chain_results`, `.tsv` (nl-\*): `chain`, `loglik`, `status`, `n_evals`, then
  the estimated parameters.
- `chain_evaluations`, `.tsv` (if2), after a comment line: `chain`, `loglik`,
  `se`, `ess_mean`, `ess_min`, `ess_min_step`, `n_neg_inf_incr`, then the
  estimated parameters.
- `iteration_diagnostics`, `diagnostics.tsv` (if2), after a comment line:
  `chain`, `iteration`, `loglik`, `if2_perturbed_loglik`.
- `diagnostics`, `diagnostics.json` (every method but the pfilter method): an
  array of findings with `kind`, `severity`, `message`, `method`.
- `fit_state`, `fit_state.toml` (every method but the pfilter method).
- `chain_starts`, `.tsv`, after a comment line (if2, mh, pmmh, pgas).
- `sampler_summary`, `<method>_summary.json` (mh, pmmh, pgas).
- `summary`, `summary.json` (every method; new, below):
  `camdl.fit-run-summary/v1`, the run's fit summary as a result file.
- `method`, `method.toml` (every method).
- `inputs/starts`, `inputs/starts.<ext>`, when a starts file is named.
- `cross_chain_compat`, `.json`, and `filter_ess`, `.tsv` (pgas).
- `logliks`, `.tsv`, and `prequential_joint`, `prequential_streams` and
  `prequential_record`, the `prequential.tsv`, `prequential.streams.tsv` and
  `prequential.json` files (the pfilter method, when `record_prequential` is
  on).

Provenance: `fit_config` (`fit.toml`), `run_inputs`, `fit_meta`
(`fit.meta.json`), `execution`, and `opt_trace` for NLopt (_verified_:
`role_of`, `fit_command.rs:1031-1038`). Nested predictions are their own
results.

### Fit problem (`fit`)

`problem` (`problem.toml`), `partition` (`.json`), `inputs/<source>` and
`inputs/tables/<name>` (training lines and tables as read), `held_out/<stream>`
(held-out dates) and `synthetic/truth` (_verified_: `fit_store.rs:1043-1062`).
The model file is provenance.

### Filter run (`pfilter`) and its by-products

- `pfilter:loglik`, `loglik.toml`: `loglik`, `n_replicates`, `n_particles`.
  `pfilter:replicates`, `replicates.tsv`, with `--replicates` above 1.
- `trace:trace`, `trace.tsv`: per observation, the increment, the ESS and the
  predictive quantiles.
- `pf_health:pf_health`, `pf_health.tsv`: `time`, `ESS`, `ESS_frac`, `tau2`.
- `filtered_streams:filtered_streams`, `filtered_streams.tsv`: `time`, `stream`,
  `phase` (`predictive` before the step is scored, `filtered` after), `quantity`
  (`expectation`, added by the scores proposal's stage 1), `mean`, `q05`, `q25`,
  `q50`, `q75`, `q95`.
- `filtering:filtering`, `final_state:final_state`, `saved_paths:paths`: the
  particles at every step, the swarm after the last observation, and smoothed
  paths.
- `prequential:joint`, `:streams` and `:record`: `prequential.tsv`,
  `prequential.streams.tsv` and `prequential.json`.

File names are the `FILE` constants of `pfilter_store.rs:159-280`.

### Simulation cell (`sim`) and its by-products

`traj` (`t`, `t_start`, `t_stop`, each compartment, each `flow_<name>`),
`reactive_log` when the model has reactive interventions, and `failure` when the
run stopped for a reason it records. By-products: `obs:<stream>` (`time` and the
stream's columns), `event_log:event_log` and `sim_quantities:quantities`
(_verified_: `sim_store.rs:229-322`).

### Prediction (`prediction`)

- `predictive`, `predictive.json` (`camdl.predictive/v5` after stage 1).
- `predictive/<stream>`, `.tsv`: `scenario`, `time`, `mode`, `horizon` (an
  integer position, empty outside a forecast), `quantity`, `treatment`, `role`
  (`trained`, `held_out`, `future` or `unknown`), `fit_rhat_max`, `fit_ess_min`,
  `rhat_mean`, `ess_mean`, `rhat_pred`, `ess_pred`, `n_draws`, `q05` … `q95`. On
  main today the mode name sits in `horizon` and there is no `quantity` column
  (_verified_ from a stored file).
- `observed`, `observed.json`, and `observed/<stream>`, `.tsv` (`time`,
  `value`).
- `forecast`, `forecast.json`; `forecast/<stream>`, `.tsv`: `scenario`,
  `t_start`, `t_stop`, `mode`, `horizon`, `lead`, `quantity`, `mean`, the
  quantiles, `n_samples`; `forecast_samples`, `forecast-samples.tsv`: one row
  per sample, `scenario`, `sample`, `chain`, `draw`, `cloud_row`, `particle`,
  `stream`, `t_start`, `t_stop`, `horizon`, `value`.
- `forecast_contrasts`, `forecast-contrasts.tsv`, when the forecast has
  contrasts; `report`, `report.json` (`camdl.predict-report/v2`).

Column meanings are run spec §2.2.2 and the scores proposal, "Labelling the
predictions".

### Score (`score`)

Stored inside the prediction it scores, at `scores/score-<hex>/`, once the
scores proposal's stage 3 lands. `scored`, `scored.tsv`: one row per scored row
or target. `summary`, `summary.tsv`: one row per stream, mode, role and lead.
`score`, `score.json`: the parameter source, the basis, the conditioning and the
notes. Their columns are the scores proposal's "Output".

### Profile point (`profile_point`)

`mle`, `mle.toml`: `final_loglik`, a `[focal]` table with the profiled
parameter's value, and an `[mle]` table with the other estimates (_verified_).
The profile table is a rendering of the points.

### Every result

`identity.json` and `run.json`, the store's records in format 1 (run-store
proposal, "On disk"), are the run record. They are reachable by name and are not
in the output manifest.

### The fit run's stored summary

Every fit run stores `summary.json`, the fit summary of that one run, written
before the run is sealed. `fit summary --format json` renders a summary on
request today, and that rendering holds `fit_dir`, each leaf's path,
`created_at` and `age_seconds` (_verified_: the JSON of a stored MH run), so its
bytes change from one call to the next and no digest can describe it.

The stored summary holds the method, the verdict, the convergence and health
figures, the best log-likelihood and the checks, as the rendering's entry for
the run does, and leaves out four kinds of field:

- times (`created_at`, `age_seconds`);
- paths (`fit_dir`, `leaf`);
- the build label inside a field, since the build is already in the run id and
  in the manifest;
- `config_diff_from_baseline`, which compares the run with a baseline run chosen
  when the summary is rendered. A stored file cannot depend on another run, so
  the field stays in the rendering only.

The file is then a function of the run's identity, as every result file must be.
`fit summary` keeps rendering its tables and JSON, and reads the stored summary
where it has one, adding the time, path and baseline fields. Adding the file
changes every fit run's stored bytes, so `fit_run`'s identity version is bumped
from 1 to 2 and every earlier fit is recomputed on its next request; London
reruns everything, and before 1.0 a re-key is cheap.

### A point estimate for an MCMC run

A point estimate for an MCMC run does not exist. `fit summary --params-only`
refuses an MH run with "holds no winner θ̂: neither final_params.toml (IF2) nor
mle_params.toml (NLopt) is there" and exits 1 (_verified_; the check is
`fit/fit_summary.rs:1151-1161`). An MH, PMMH or PGAS run stores a posterior and
no point. A consumer that needs one point chooses it from `draws` explicitly,
and `camdl path` refuses `mle_params` for such a run with the run's file list.

## Builds and platforms

`ENGINE_ID` covers what the build script can see. It does not cover the C
compiler that builds `nlopt`, the dynamically linked system libraries (among
them `exp`, `ln`, `pow` and `lgamma`), profile settings outside the repository,
or third-party features beyond what `Cargo.lock` implies (_verified_: the module
comment of `engine_fingerprint.rs`). The OCaml compiler needs no place in it:
the compiled model is keyed by its own bytes, so a `camdlc` that emits another
model gives another run id.

The target triple is in the fingerprint (`engine_fingerprint.rs:132`; the test
at `engine_fingerprint/tests.rs:225`), so a Linux and a macOS build of one
commit never share a run id. The numbers require that: every fit method computes
different bytes under macOS libm and glibc (_verified_: the per-platform pins,
`fit_command/tests.rs:740-830`).

The system libraries are the gap. Two Linux machines running one binary with
different glibc versions share an `ENGINE_ID` and could compute different bytes
under one run id. Production therefore uses one machine image per production
build.

A cloud result placed in a local store sits under its own build directory, named
by its ambient key, beside the local build's. A local computation can depend on
it, since a dependency records the upstream's run id, file, digest and build,
and the upstream may come from another build. Cross-platform reuse is this
reading by id, never a cache hit. That a full run id resolves to a result in
another build's directory is _inferred_ from the run-store proposal ("How a run
id becomes a folder", step 3) and the prefix walk in `handle.rs`, not exercised.

### What `pull` needs from camdl

`camdl-scope` and `camdl-cloud` `pull` are separate tools. From camdl they need
the following.

- The output manifest of the remote command, naming the run ids to fetch and
  each result file's digest.
- Whole result folders, records included, with the parent chain a nested result
  needs: a `fit_run` sits inside its fit problem's folder and a `prediction`
  inside its fit run's (_verified_: `fit_store.rs:343-349`, `:499-506`).
- Verification: each file's SHA-256 against the manifest and the copied
  `run.json`, and `camdl verify <run id>`, which exits 1 on damage.
- A rule for a run id already present: equal digests mean nothing to do; unequal
  digests are refused with both named, since two stores then hold different
  bytes for one identity.
- A command that places the fetched folders into the local store and records
  them, so they appear in `camdl list`: `camdl receive`, below. `pull` fetches;
  camdl checks and places.

## Receiving pulled results: `camdl receive`

`camdl list` finds results only through invocation records (_verified_:
`discover_stored_sim_rows` and its siblings iterate `store.invocations()`,
`browse.rs:631-652`), and an invocation record's `seq` is local to its store, so
a remote record cannot be copied into a local store as it is. `camdl receive`
moves results copied from another store into this one and writes a local
invocation record for them. It keeps the rules for placing a folder in a store
inside camdl, so the cloud tools need not track the store's layout.

### The command

```bash
camdl-cloud pull m4w_mh --to pull-staging/        # the cloud tool fetches
camdl receive pull-staging/ --label m4w_mh_cloud --manifest build/fits/m4w_mh.json
```

The argument is a partial copy of another store: its `store.json`, the result
folders wanted, each at the path relative to the root it had there, with the
parent folders a nested result sits in, and the remote invocation records that
listed them. `pull` writes that copy. `--store`, `--label`, `--manifest` and
`--link-root` mean what they mean on every other command.

### What it checks

For each result folder in the copy, in parent-before-child order:

1. The record is completed, of a registered kind, in the record format this
   binary reads.
2. The run id recomputes from `identity.json`'s kind, ambient record and
   identity record, as `camdl verify` does.
3. Every result and provenance file hashes to its digest in `run.json`, and the
   folder holds nothing the store manifest does not name besides the store's own
   entries and its nested results.
4. A nested result's parent is in the copy or already in this store.

A folder that fails any check is refused, by path and reason, and nothing from
the copy is placed. The checks run on the copy after it has been moved into
`<root>/.staging/`, on the store's filesystem, so the rename that places each
folder is atomic.

### Where it places a result

Each folder goes to the same path relative to the root as it had in the remote
store. That path sits under the remote build's directory, named by the remote
ambient key, so a cloud result never lands among the local build's results. A
run id already stored here is compared file by file: equal digests leave the
local folder in place and the result is recorded as received; unequal digests
are refused with `Error::ReceivedDiffers`, naming both folders, since two stores
then hold different bytes for one identity. A folder path taken by another run
id is refused the same way.

### What it records

One local invocation record, with `argv` the `receive` command line and the
`--label` given. Its results are the received ones, in the copy's
parent-before-child order, each with its kind, its run id, the labels the remote
invocation record gave it, and outcome `reused`, so no record format changes.
The index gains one line per placed result. `@L`, `list` and `show` then see the
results as they see local ones, and `--manifest` writes their output manifest,
so a Snakemake rule can declare a pull's manifest as its output.

### Types

The library does the work, from records alone:

```rust
// rust/crates/store/src/receive.rs
pub struct Received { pub placed: Vec<Key>, pub present: Vec<Key> }

impl Store {
    /// Checks every result folder of the store copy at `copy`, then places
    /// them, then records them in `info`'s invocation. Nothing is placed
    /// unless every folder passes.
    pub fn receive(&self, copy: &Path, info: &RunInfo) -> Result<Received>;
}

// Error::ReceivedDiffers { run_id: Key, local: PathBuf, received: PathBuf }
// Error::ReceiveRefused { folder: PathBuf, reason: String }
```

camdl's `receive` command opens the store, calls `Store::receive`, and writes
the manifest and links.

## Costs

The design has costs, each accepted for a reason.

- Every read needs camdl, or a script that parses the manifest and the store's
  records. A plain `cat build/fits/m4w_mh/draws.tsv` no longer works without a
  link root.
- The store will need garbage-collection roots. A manifest or a link is the
  workflow's statement that a result is in use, and when a prune command exists
  it should keep every result a manifest or link under a declared root names.
  Until then nothing is pruned automatically.
- make reruns the steps downstream of a no-op camdl rerun once, because it
  compares times and the manifest's time is refreshed. Snakemake's checksum of
  small inputs avoids that pass.
- Digests are stored twice, in `run.json` and in each manifest that names the
  result, and the resolver checks that they agree.
- There are two version numbers for a reader to know, the file schema and the
  kind's contract, beside the identity version the store reads.

## Prior art

Other tools that keep results in a store named by a hash bind them to users in
similar ways.

- Nix's `nix-build` leaves a symbolic link called `result` to the built store
  path, and "the result of the build is automatically registered as a root of
  the Nix garbage collector"
  ([Nix 2.34.9 manual, `nix-build`, "Description"](https://nix.dev/manual/nix/stable/command-ref/nix-build.html)).
  camdl's link tree is the same idea, and a garbage-collection root is the role
  a manifest or link would take.
- DVC tracks a data file with a `.dvc` pointer file holding its `md5`, `size`
  and `path`
  ([DVC docs, ".dvc Files"](https://doc.dvc.org/user-guide/project-structure/dvc-files),
  no version shown), and places cached data in the workspace by reflink,
  hardlink, symlink or copy, chosen by `cache.type`
  ([DVC docs, "File link types for the DVC cache"](https://doc.dvc.org/user-guide/data-management/large-dataset-optimization)).
  The output manifest is a pointer file of this kind for a command's results.
- Nextflow runs each task in "a unique directory based on its hash"
  ([Nextflow docs, "Cache and resume", "Work directory"](https://docs.seqera.io/nextflow/cache-and-resume))
  and copies or links chosen outputs out with `publishDir`, by default "as a
  symbolic link from the task directory to the target directory"
  ([`publishDir`, "Options"](https://docs.seqera.io/nextflow/reference/process/directives/publish-dir)).
  `--link-root` plays `publishDir`'s part.
- Snakemake's between-workflow cache keys a rule's outputs by "hashing all
  steps, parameters, software stacks … and raw input"
  ([Snakemake 9.27.2, "Between workflow caching"](https://snakemake.readthedocs.io/en/stable/executing/caching.html)),
  on rules marked `cache: True`. It is Snakemake's own store, keyed inside
  Snakemake; camdl's keys come from camdl's typed inputs, and the two can be
  used together.
- MLflow downloads an artifact by run id and path,
  `mlflow artifacts download
  --run-id <id> --artifact-path <path>`
  ([MLflow 3.18.0 CLI, "artifacts download"](https://mlflow.org/docs/latest/api_reference/cli.html)).
  Weights & Biases uses an artifact by `name:alias` with `run.use_artifact()`
  and fetches one entry with `Artifact.get_entry()`
  ([W&B docs, "Download and use an artifact"](https://docs.coreweave.com/products/wandb/artifacts/download-and-use-an-artifact.md),
  no version shown). `camdl path` is lookup by run id and name in the same way;
  a W&B alias such as `latest` moves as camdl's `@L` does, which is why
  manifests hold ids.

## What is refused

- `--manifest` or `--link-root` on a command that stores nothing (`compile`,
  `fit summary`): clap refuses the flag.
- `compare --manifest` when the predictions and scores compare reads or stores
  sit in more than one run store: refused with exit 1, naming the stores, since
  a manifest names results in one store.
- `--manifest` or `--link-root` with `--no-store`: the results would be deleted
  with the scratch store. Refused at argument parsing.
- A manifest path that is a directory, or whose parent does not exist: refused
  before the first store call, with exit 1.
- A manifest of another schema, or of a newer `camdl.output-manifest` version:
  refused by name, saying which version was found and which is read.
- `--from-manifest` with no `fit_run` entry or with several.
- `camdl path` given a view or an attempt file.
- A finished run whose files differ from its expected set: `FileSetMismatch`; a
  run given no expected set: `NoExpectedFiles`.
- `camdl receive` of a copy with a folder that fails a check (`ReceiveRefused`),
  or whose run id or path is held here by other bytes (`ReceivedDiffers`).
  Nothing from the copy is placed.

## Tests

Each test is written first and shown to fail before the change it covers.

- `manifest_bytes_are_identical_on_a_cache_hit`: `fit run --manifest` twice;
  identical bytes, newer modification time.
- `manifest_is_the_same_for_two_store_roots`.
- `a_failed_command_leaves_no_manifest`, with an earlier manifest at the path.
- `a_batch_with_a_failed_member_writes_no_manifest`.
- `manifest_lists_by_products_with_their_upstream`, on
  `pfilter
  --save-filtered-streams --pf-health`.
- `manifest_digests_are_the_store_digests`, with no provenance or attempt file
  listed.
- `make_converges_after_one_pass`, with a Makefile around `camdl … --manifest`.
- `path_and_cat_resolve_one_file`: the bytes `cat` prints hash to the digest of
  the file `path` prints.
- `path_prints_several_names_in_order` and `path_all_lists_every_file`.
- `a_bare_name_held_by_several_entries_is_refused`, with the candidates listed,
  when `--each` is not given.
- `path_each_puts_labels_in_columns`: a two-parameter sweep's `traj --each`
  table has `run_id` (64 hex), the cell labels, one column per swept parameter
  and `path`, one row per cell in manifest order.
- `path_each_takes_the_union_of_label_keys`: a manifest whose results carry
  different label keys gets every key as a column and an empty value where a
  result lacks one.
- `cat_each_prepends_labels_and_refuses_a_header_mismatch`: rows keep their
  cell's coordinates under one header; two results whose files have different
  headers are refused before a row is printed, naming the first that differs.
- `a_method_writer_cannot_name_another_methods_file`: a compile-fail test (an MH
  `MethodOutput` given `If2File::MleParams`).
- `a_name_of_another_method_is_refused_with_the_runs_files`: `mle_params` of an
  MH run.
- `cat_refuses_a_damaged_file` and `path_refuses_a_manifest_for_other_bytes`.
- `from_manifest_reads_the_one_fit_run`, refused with none or two.
- `links_are_relative_under_a_shared_parent` and `links_replace_earlier_links`.
- `finish_refuses_an_undeclared_file` and
  `finish_refuses_a_run_with_no_expected_set`, in the store library, each naming
  what it found.
- `every_command_declares_its_files`: every command's fixture run finishes, so
  none reaches `finish` without an expected set.
- `fit_run_summary_is_a_function_of_the_run`: two fit runs of one identity in
  two stores, at different times, store byte-identical `summary.json`, holding
  no time, path or baseline field.
- `receive_places_and_records_a_pulled_fit`: a fit run and its problem copied
  from a second store are placed under the remote build's directory, listed by
  `camdl list`, reachable by `@L`, and named in `--manifest`'s output.
- `receive_refuses_a_damaged_copy_and_places_nothing`: one file's bytes changed
  in a two-result copy; neither result is placed.
- `receive_of_a_result_already_here`: equal bytes are recorded and left in
  place; different bytes are refused with both folders named.
- `every_method_declares_its_files`: each algorithm's `files()` equals what its
  fixture run stores.
- `the_contract_section_is_generated` and
  `every_stored_kind_matches_the_file_contract`.

## Stages

Each stage lands on its own, with its tests.

1. **Store reads.** `Store::locate`, `Located`, `Invocation::outputs`.
2. **File enums and the contract.** One enum per kind, the fit run's per-family
   enums, `MethodFiles`, `MethodOutput` with the methods moved onto it and
   `ingest` removed, the generated run-spec section and its tests.
3. **The expected file set.** `Run::expect`, `FileSetMismatch`,
   `NoExpectedFiles`, and the shared helper through which every command begins a
   run declaring its set.
4. **The stored fit summary.** `summary.json` in every fit run, `fit summary`
   reading it, and `fit_run`'s identity version bumped to 2, with the fit pins
   moved in their own commit.
5. **`--manifest`** on `fit run`, then on `simulate`, `pfilter`, `predict` and
   `profile`.
6. **One resolver.** `resolve_file`; `camdl path`, with `--all` and `--each`;
   `camdl cat` moved onto it, with `--each`, its kind-by-kind chain and the
   `.tsv` completion removed.
7. **`--from-manifest`** on `predict`, `simulate`, `profile` and `pfilter`.
8. **Links.** `--link-root` and `CAMDL_LINK_ROOT`.
9. **`camdl receive`.** `Store::receive` and its checks in the library, then the
   command with `--label`, `--manifest` and `--link-root`.
10. **The guarantee's wording.** The run-store proposal's guarantee 8 reads
    "Every write and every read of a stored file goes through the store." It
    becomes: "Every write of a stored file, and every read of one by a camdl
    computation, goes through the store. A program outside camdl that reads a
    path `camdl path` printed is outside the store; it can check what it read
    against the output manifest's digest." This is London item 8.

## Out of scope

**Separating build provenance from cache compatibility (London item 6).** The
build fingerprint now both records which binary made a result and decides what a
binary may reuse. Separating them would let a build that changed only
documentation reuse its predecessor's results. It needs a compatibility key no
coarser than the bytes it guards, such as a digest of the numerical crates'
source with a test that nothing outside them can change a result, or a
fingerprint of outputs on committed fixtures. The store's first guarantee
resolves doubt toward recomputing, and with one pinned production build London
does not need it. It is a proposal of its own.

**`--plan --format json` (London item 7).** A planning output would report each
result's run id and whether it is stored, before running. The library has the
pieces (`Store::plan`, `storage.rs:358`), and `simulate --dry-run` prints hits
and misses for people. A stable machine format needs every command's identity
resolution, the compile and the data load included, to run without computing.
Snakemake does not need it, because a cache hit is cheap; a scheduler deciding
whether to start cloud machines would, and that is `camdl-cloud`'s proposal.

**A workflow engine.** camdl records no rule graph, orders no commands and
reruns nothing because an upstream changed.

**Copying out.** `-o` and the other export flags stay, for a person who wants
files outside the store. Nothing here copies a stored file by default.

## Appendix: the London list, item by item

- **1a.** `--manifest PATH`, a small versioned JSON with run ids, labels, the
  build and a schema version: "The output manifest".
- **1b.** Written atomically; a failed command writes none: "When the manifest
  is written and removed".
- **1c.** Byte-identical on a full cache hit, and whether it is rewritten: "What
  the manifest holds, and what it leaves out"; it is rewritten.
- **1d.** A lookup by manifest or id and a file name, printing a path or
  contents, with a hard error when missing: "Looking up a stored file".
- **1e.** No large-file duplication by default, `-o DIR` kept: "Motivation",
  "Links", "Out of scope".
- **1f.** Labels never reproducible dependencies: "Looking up a stored file",
  "Labels".
- **2.** Named result files with formats, versions and columns, for fits and
  filters: "Result files as types", "Versioning", "The result-file contract";
  the fit summary is "The fit run's stored summary".
- **3.** `camdl-scope` and `camdl-cloud` `pull`: "What `pull` needs from camdl"
  and "Receiving pulled results". The tools' implementation is outside this
  repository.
- **4.** Cloud and local results apart unless equal, digests verified on pull:
  "Builds and platforms", and the checks of `camdl receive`.
- **5.** Item 2 for predictions, scores and the run record: "Prediction",
  "Score", "Every result".
- **6.** Build provenance and cache compatibility: out of scope, with what it
  would need.
- **7.** `--plan --format json`: out of scope, with what it would need.
- **8.** "Every read goes through the store", narrowed to camdl's computations:
  stage 10.
