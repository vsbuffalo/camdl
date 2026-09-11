# The fit store: identity, addressing, and what happens when one problem holds several runs

Date: 2026-09-11 Status: working note for a design decision (gh#896). Everything
below is read from `main` at `bd368248` unless a line says otherwise.

## The question in one paragraph

A run's identity is a pure function of its inputs and nothing in the
content-addressed tree collides. What goes wrong is one level up: the folder
that a problem's runs share holds files that belong to one run, and the handles
the verbs accept name that shared folder rather than a run. When a folder holds
two runs of one algorithm, which it now does after any rerun with a changed
sampler setting, `fit summary` and `fit predict` choose a run by directory-name
order and say nothing, and `fit predict` writes its output where the other run's
output was. This note lays out the pieces so the fix can be chosen deliberately.

## The store as it stands

### Three hashes, one path

A fit leaf's identity is three content hashes, one per level, and its address is
a hash of those:

```
run_id = SHA256( HASH_VERSION ‖ kind_index ‖ level_count ‖ fit_hash ‖ method_hash ‖ seed_hash )
```

`runid::run_id` (`rust/crates/runid/src/kind.rs:79`) folds the kind as its
declaration index (`ArtifactKind::FitStage` is 1), so two kinds with equal level
hashes cannot alias. The path is a readable factoring of the same tuple, each
segment `<label>-<first 8 hex of the level hash>`:

```
results/fits/<file stem>-<fit h8>/<algorithm>-<method h8>/seed_<n>-<seed h8>/
```

Labels are provenance; hashes are identity. The spec's one rule for readers:
resolve runs from `run.json`, never from the path.

What each level hashes, from `fit::cas::resolve_fit_stage`
(`rust/crates/cli/src/fit/cas.rs:572`):

| level  | type                                                      | hashes                                                                                                                                                                                                                                                                          |
| ------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| fit    | `runid::inputs::FitDigest`                                | the whole model IR plus `ir_version` and the engine version (`ModelDigest`), the content of every training data file, the content of every holdout file, the canonical JSON of the _problem half_ of the config with `output_dir` removed, and the engine version again         |
| method | `runid::inputs::StageLevel { config: StageConfig, deps }` | the method's identity payload (the algorithm's own fields plus the resolved `starts` rule), `n_trajectories`, the target chain length, the resolved observation alignment, and `deps`: the identity of any leaf a `from_mle` / `from_posterior` / `from_params` source consumed |
| seed   | `runid::inputs::Seed`                                     | the resolved fit RNG seed                                                                                                                                                                                                                                                       |

Two consequences of the fit level that matter here. The engine version is
`0.1.0+<git hash>`, so every build re-keys every fit; a rebuilt camdl rerunning
an unchanged file writes a new leaf. And `[method]` is excluded from the fit
hash by construction (`fit_config_blob_hash` hashes `Problem`, not `FitConfig`),
so two files that differ only in `[method]` share a fit hash and therefore a
folder.

### The types

Config, after the split (`rust/crates/cli/src/fit/config_v2.rs`):

```rust
pub struct FitConfig { pub problem: Problem, pub inference: Inference }

pub struct Problem {            // every reader loads this half
    pub model: ModelRef,
    pub data: Option<DataSpec>,
    pub synthetic: Option<SyntheticSpec>,
    pub simplex_groups: Vec<SimplexGroup>,
    pub output_dir: Option<PathBuf>,   // provenance, never identity
    // [estimate], [fixed], [config], scenario, enable/disable, ic_free …
}

pub struct Inference {          // only `fit run` needs this half
    pub method: Option<Method>, // exactly one [method]; None is a complete problem
    pub fit_seeds: Option<Vec<u64>>,
}

pub struct Method { pub algorithm: Algorithm, pub starts: Option<ChainStarts> }
// Algorithm is #[serde(tag = "algorithm")]: IF2 | PGAS | PMMH | Mh | Nuts | PFilter | NlSbplx | NlBobyqa
```

Chain starts (`rust/crates/cli/src/fit/starts.rs`):

```rust
pub enum ChainStarts { Spread(Spread), Point(Point) }
pub enum Spread { UniformUnconstrained, Lhs, Uniform, FromPrior, FromPosterior(Handle) }
pub enum Point  { Declared, FromMle(Handle), FromParams(PathBuf) }
```

`Spread` versus `Point` is the fact R̂ needs: chains that began at one point
agree by construction, so `fit summary` withholds R̂ for a `Point` run.

Identity resolution (`rust/crates/cli/src/fit/cas.rs:57`):

```rust
pub struct FitStageCtx<'a> {
    pub model: &'a ir::Model, pub fit_stem: &'a str,
    pub ir_version: &'a str, pub engine_version: &'a str,
    pub problem: &'a Problem,                       // sweep overrides applied
    pub data_paths: &'a IndexMap<String, String>,   // content is digested
    pub method: &'a Method,                         // CLI overrides + starts default applied
    pub seed: u64,
    pub deps: Vec<ArtifactRef>,                     // a starts source's leaf identity
}
pub struct ResolvedFitStage { pub levels: Vec<LevelId>, pub run_id: ContentHash }
```

The leaf record (`rust/crates/runid/src/record.rs:167`), written as `run.json`
in every leaf: `kind`, `run_id`, `ir_version`, `engine_version`,
`levels: Vec<LevelId { name, label, hash, schema_version }>`, `deps`, `status`,
`artifacts` (an exact-set manifest of the leaf's own files), `output_schema`,
`inputs` (a display projection, never hashed: `method`, `backend`, `seed`,
`n_chains`, `best_loglik`, `best_chain`, `algorithm`, `starts`,
`chain_starts_kind`, `fit_hash`, `wall_time_seconds`), and `provenance`
(`created_at`, `argv`).

### What lives at each level of the tree

Written today by a two-chain PGAS fit of the polio fixture, then the same file
with `sweeps` raised, then a second method file on the same problem, then the
IF2 file:

```
results/fits/
├── pgas-9a53c2c4/                         fit level: the PROBLEM
│   ├── fit.meta.json                      FitSidecar: label (sticky), estimated, fixed,
│   │                                      resolved_priors, data_hashes, model_identity,
│   │                                      fit_toml_path, fit_toml_hash, training_window
│   ├── fit.toml.original                  the config of the LAST run (fs::copy, overwritten)
│   ├── model.ir.json  model.render.json  model.graph.json  model.camdl.original
│   ├── predictive/  predictive.json       ← written by `fit predict`, from ONE leaf's draws
│   ├── observed/    observed.json         ← same
│   ├── quantities/  quantities.json       ← same
│   ├── report.json                        ← same
│   ├── pgas-34f379b8/seed_1-06cbd6b3/     method level: sweeps = 12
│   │   run.json  fit_state.toml  chain_starts.tsv  draws.tsv  pgas_summary.json
│   │   diagnostics.json  progress.json  cross_chain_compat.json  chain_1/  chain_2/
│   └── pgas-d5c282ce/seed_1-06cbd6b3/     method level: sweeps = 16
├── pgas_b-9a53c2c4/pgas-15e2ca06/…        a second method file: same fit hash, own stem
└── fit-9a53c2c4/if2-4fd60eb3/…            the IF2 file: same fit hash, own stem
```

The problem-level files are written by `run_meta::write_fit_sidecar`
(`rust/crates/cli/src/run_meta.rs`, "Write the fit-level sidecar"), once per
`fit run`, with `fit.toml.original` recopied each time and the label sticky
across reruns (gh#29; `fit label` relabels this same sidecar). The predictive
family is written by `fit predict` at `rust/crates/cli/src/fit/predict.rs:2657`,
`:2688`, `:2720`, all `segment.join(...)`; the spec calls them "regenerated
sidecars: not part of any leaf's manifest, carry no `run_id`, overwritten in
place".

### How a verb gets from a handle to a leaf

A handle is classified syntactically (`rust/crates/cli/src/fit/handle.rs`):

```rust
pub enum FitRef { Label(String), Config(PathBuf), RunDir(PathBuf), HashPrefix(String) }
pub struct ResolvedFit { pub segment: PathBuf, pub config: FitConfig }
```

Every branch resolves to the **segment**, the problem folder:

- `@label`: the segment whose `fit.meta.json` carries the label.
- hash prefix: the segment whose `FitView.fit_hash` starts with it; more than
  one match is `ResolveError::Ambiguous`, listed git-style.
- directory: taken as is.
- `fit.toml`: `resolve_config` (`handle.rs:229`) computes `config_identity_hash`
  of the file, which covers the **whole** config including `[method]`, and
  compares it with the same hash of every segment's archived
  `fit.toml.original`. Exactly one match returns that segment. Note what this
  implies: the file in hand does identify a single run, but the archive it is
  compared against is the last run's config, and the result handed to the verb
  is the segment, so the identification is computed and then discarded.

Inside the segment, each verb picks a leaf on its own:

- `fit summary`: `fit_summary::discover_stages` (`fit_summary.rs:446`) builds
  one entry per method **label**, and when several leaves share a label keeps
  the one ranked (real data before synthetic, lowest fit seed, then
  lexicographically first `stage_dir`). Two PGAS leaves tie on everything but
  the directory name, so the winner is hash order.
- `fit predict` and every consumer of draws:
  `posterior_draws::resolve_posterior_draws`
  (`rust/crates/cli/src/posterior_draws.rs:59`) takes the segment's
  `FitView.stages` in label order and reads `with_draws.last()`; with a
  method-name override it takes the first leaf whose algorithm matches. A name
  cannot tell two PGAS leaves apart.
- `fit table`: `table_row::pick_terminal_stage` (`table_row.rs:398`) walks
  `methods_declared` in reverse and applies the same rank.

`FitView` (`rust/crates/cli/src/fit/fit_view.rs`) is the fold of a segment:
`fit_hash`, sidecar fields, and one `FitStageView` per leaf with `method`,
`method_hash`, `run_id`, `backend`, `seed`, `n_chains`, `best_loglik`,
`best_chain`. Since gh#895 `camdl list --format json` exposes `method_hash` and
`run_id` per leaf, so a reader can address a leaf; no camdl verb yet accepts
that address for a segment-level operation.

## How it looked before the split, and what the split changed

Before `b44ec300` a `fit.toml` carried a pipeline:

```rust
pub struct FitConfigV2 {
    // model, data, synthetic, fit_seeds, simplex_groups, estimate, fixed …
    pub fit_starts: Option<FitStarts>,
    pub stages: IndexMap<String, Stage>,       // ordered; executed in declaration order
}
pub enum Stage { /* tagged by algorithm; each variant carried */
    // init: Option<InitMethod>, init_mle / StartsFrom::{Stage(name), Directory, Random},
    // survey_path, survey_top_k, gate: GateConfig, …
}
```

The store's middle level was the **stage**, labelled
`<ordinal>-<stage name>-<h8>` (`01-scout-…`, `02-posterior-…`). The ordinal
encoded execution order; a later stage's level hash folded the earlier stage's
identity through `deps`, so regenerating the scout re-keyed the posterior. One
config described the whole pipeline, and one pipeline ended in one posterior.
Under that invariant the problem folder was "the fit": one archive, one config,
one predictive, and a stage name was a unique key within it, so "pick the leaf
by name" was deterministic. The invariant had already failed once before the
split: a rerun with changed sampler settings put a second `01-posterior-<h8>`
beside the first, and `fit summary` reported the old one while `fit predict`
read the new one (gh#609, August).

The split replaced the pipeline with one `[method]` per file and moved chaining
out of the file: a second way of fitting a problem is a second file, and a warm
start names the earlier run by handle. The store level became `method`, labelled
by the algorithm with no ordinal, keyed on the method's identity payload plus
`starts` plus `deps`. Nothing about the folder changed. So the folder still
means "one problem" and now routinely holds several runs of one algorithm,
because every rerun with a changed `[method]` setting is a new method hash under
the same fit hash.

Why `[method]` is excluded from the fit hash, and whether each reason needs the
shared folder:

1. The problem archive (IR, render, graph, source, resolved priors) is written
   once per problem, not once per run. Needs one home keyed by the problem; does
   not need run files to live there too.
2. Methods on one problem can name each other: `from_mle = "@scout"` folds the
   scout leaf's identity into the posterior's `deps`, and the shared fit hash is
   what proves the problem is identical. This is about the hashes in `run.json`;
   it reads no directory.
3. `fit table` and `compare` group runs by problem. Also by hash, read from
   `run.json`.

Only the first is a reason to share a folder, and it constrains only the
archive.

## The problem, measured

On the store above, with `camdl 0.1.0+57c607c1` (before today's fixes) and
unchanged in the parts that matter on `bd368248`:

```
$ camdl fit summary results/fits/pgas-9a53c2c4
  camdl 0.1.0+57c607c1 · pgas · pgas · 2 chains · 16 draws · 2 chains saved paths
```

That is `pgas-34f379b8`, the 12-sweep run (2 chains × 8 retained sweeps),
because `34…` sorts before `d5…`. Nothing says a second run exists.

```
$ camdl fit predict results/fits/pgas-9a53c2c4
fit predict: … 16 draws from pgas method 'pgas'
wrote results/fits/pgas-9a53c2c4/predictive/afp.tsv
wrote results/fits/pgas-9a53c2c4/predictive.json
```

Same leaf, and the files land in the shared folder; `predictive.json`'s keys are
`calendar`, `schema`, `streams`, nothing naming the run. A later `fit predict`
from the other leaf overwrites them. `fit.toml.original` in the same folder is
the 16-sweep config, so the archive describes a run the verbs did not read.

Three separate defects, then:

1. Run-derived files are filed under the problem (`predictive/`, `observed/`,
   `quantities/`, `report.json`, `fit.toml.original`, and the label), where a
   second run overwrites them and nothing records which run produced them.
2. Run-needing verbs accept problem addresses and choose a run inside by a rule
   that was only ever deterministic under "one leaf per name".
3. `resolve_config` identifies a run from the file in hand and then returns the
   folder.

## The candidates

**A. Keep the factoring; move run files under the run; make verbs address
runs.** The problem folder keeps only the archive and the problem half of
`fit.meta.json`, with its file list pinned by a test. Everything a run produced
lives under its leaf, and the predictive sidecars record the leaf's `run_id` and
method hash. A `fit.toml` handle computes its identity and names exactly the
leaf it identifies, or reports "not run yet" and prints the command; a leaf
directory and a run-id prefix are run addresses; the label moves to the leaf so
`@label` names a run; a problem address given to a run verb is refused with the
runs listed by file, while `fit table`, `list` and `compare` keep accepting
problem addresses; `fit table`'s terminal method gets a stated definition. No
consumer selector: the file you ran with names the run. No hash changes, no
re-keying, no migration of leaves; camdl-scope's `predictive/` read moves one
level down.

**B. One folder per run.** Fold the method hash into the top-level directory
name, or nest as `<stem>-<fit h8>-<method h8>/`, with the archive keyed by model
identity somewhere shared. Removes the shared folder entirely, so nothing can be
misfiled. Costs a layout migration for every reader and duplicates or relocates
the archive, for no correctness gain over A once A's rule holds. The problem
grouping already lives in `run.json`.

**C. Chain length as progress, not identity.** For the four MCMC methods,
`sweeps` (PGAS), `iterations` (PMMH, MH) and `samples` (NUTS) leave the method
hash; a leaf is (problem, method without length, seed) and grows in place when a
longer run is asked for, with a shorter request a prefix read. This is the model
of what a chain is, and it removes the most common source of sibling leaves. It
requires exact continuation: today `ChainResumeState`
(`rust/crates/sim/src/inference/pgas.rs:1153`) holds the parameters, the
reference trajectory, the adapted mass matrix and step size, and the completed
sweep count, but no RNG state, so `--resume` writes a new leaf keyed on the new
length with a `deps` entry on the base and is not bit-identical to an
uninterrupted run of that length. Without saved RNG state a growing leaf's
content would depend on how it was grown, which is worse than duplication. It
also changes sealed-leaf semantics: manifests rewritten on extension, cache hit
meaning "at least as long as asked", readers stating the length they summarise.
IF2 has no extension dimension (its cooling schedule depends on the total) and
NLopt's termination differs by budget, so C applies to four methods. C does not
remove siblings from a seed change, any other `[method]` edit, a `--starts`
override, or a rebuild, so A's rule is needed under C as well.

## Open decisions

1. A now, before the tag, with C as a separate post-tag proposal; or B; or defer
   the addressing change and ship the tag with defect 1 fixed only (files under
   the run) and defects 2 and 3 documented.
2. Under A, whether the label attaches to the run or to the file. Per run is how
   `from_posterior = "@base"` is already used; per file keeps `fit label` on the
   sidecar and leaves `@label` a problem address.
3. Under A, what a problem address means to a run verb: refuse and list, or
   resolve to the run the archived config identifies (the last run of that file)
   and print which. The first is the repository's existing answer to ambiguity;
   the second is a rule that a short exploratory rerun would satisfy over the
   real one.
4. Whether the engine version belongs in the fit hash. It is documented and
   deliberate (2026-05-31 proposal: a runtime-only engine change re-keys); its
   effect is that every commit re-keys every fit, which decides `fit summary`'s
   "refit" answer on every pre-tag summary and belongs in the release notes
   either way.

## Where the code would change under A

`fit/predict.rs` output paths (three `segment.join` sites) and `predictive.json`
/ `report.json` provenance; `run_meta::write_fit_sidecar` (archive and label per
leaf, problem half stays); `fit/handle.rs::resolve_config` (return the
identified leaf) and the `Label` branch;
`posterior_draws::resolve_posterior_draws` (take a leaf, not a segment);
`fit_summary::discover_stages` and `table_row::pick_terminal_stage` (no fold by
label; terminal defined); `chain_starts` handle resolution for `from_mle` /
`from_posterior` (a run, or refuse); `docs/camdl-run-spec.md` §2.2.2 and §6.8;
`docs/agents.md`; a pin test on the problem folder's file list; the camdl-scope
reader (`predictive/` under the leaf).

Related: gh#896, gh#609, gh#895, the workflow-first proposal
(`docs/dev/proposals/2026-09-08-workflow-first-fit-config.md` §5 and §8), the
identity proposal
(`docs/dev/proposals/2026-05-31-content-addressed-run-identity.md`), the
sealed-packet proposal
(`docs/dev/proposals/2026-06-27-sealed-fit-packets-handles-and-override-algebra.md`).
