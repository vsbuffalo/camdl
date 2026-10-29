# Fit labels name one run (gh#896)

Date: 2026-10-29 Status: design for the maintainer's decision on gh#896,
"resolve labels unambiguously to immutable run IDs and preserve that identity in
dependent runs". Background, the store layout and the three defects:
`2026-09-11-fit-store-identity-addressing-and-multiple-leaves.md`.

## What goes wrong today

Measured on `a9a89178` with the SIR recovery model, an IF2 file run twice with
`--label scout`, the second time with `chains` changed:

```
results/fits/fit_if2-198ada3a/if2-607db55a/seed_1-06cbd6b3   # chains = 4
results/fits/fit_if2-198ada3a/if2-0e2449eb/seed_1-06cbd6b3   # chains = 6
```

- `starts = { from_mle = "@scout" }` is refused with "resolves to 2 method
  leaves (several seeds or cells)". Neither seeds nor cells differ; the two runs
  differ in `[method]`.
- `fit run … --label scout16` on a cache hit renames the label on the shared
  problem folder (`fit.meta.json`), so `@scout` then answers "no fit found". A
  run that consumed `@scout` earlier cannot be found by the name it was given.
- `fit summary fit-if2.toml` reports one of the two runs, chosen by directory
  order, and says nothing about the other.
- `fit summary <leaf directory>` prints "(no completed methods found …)" and
  exits 0.

The label lives on the problem folder, which holds every run of one problem;
`@label` therefore names a problem, and every verb then picks a run inside it by
a rule of its own.

## The rule

A label names exactly one run, the leaf it was given to, and never moves.
Reusing a label is refused. A run's label is fixed once set.

Refusal was chosen over versioned labels (`scout@2`). Versioning leaves `@scout`
meaning the first run while the modeller who just reran `--label scout` expects
the new one; a refusal says so at the moment of the rerun, with the run that
holds the name and the command to pick another.

## Types

The label's home is the leaf record, the field sims already use:

```rust
// runid::record::Provenance (unchanged type; fits now populate it)
pub label: Option<String>,   // the run's label, set when the run is written or
                             // on a cache hit that names an unlabelled run
```

`FitSidecar.label` stays only for profile sidecars; a fit segment no longer
writes or reads it. `FitView.label` goes; `FitStageView` gains
`label: Option<String>`, and readers that showed one label per problem show the
labels of its runs.

A handle resolves to a target, and a run verb needs a leaf:

```rust
// fit/handle.rs
pub struct LeafRef {
    pub dir: PathBuf,          // .../fits/<stem>-<h8>/<algorithm>-<h8>/seed_<n>-<h8>
}                              // its run.json carries the run_id and the label

pub enum FitTarget {
    /// One run: `@label`, a leaf directory, a run-id prefix, or a `fit.toml`
    /// whose recorded config identity matches exactly one run.
    Leaf(LeafRef),
    /// A problem folder: its directory or a fit-id prefix.
    Segment(PathBuf),
}

pub enum ResolveError {
    NotFound(String),
    /// Several problem folders (unchanged).
    Ambiguous { handle: String, candidates: Vec<PathBuf> },
    /// Several runs where one is needed, each listed with its run id, method
    /// hash, seed, chains, creation time and label, and the axes on which the
    /// candidates differ (the problem, `[method]` settings or `--starts`, fit
    /// seed, camdl build) computed from their records rather than asserted.
    AmbiguousRuns { handle: String, runs: Vec<cas_read::Leaf> },
}

pub fn resolve_target(handle: &str) -> Result<FitTarget, ResolveError>;
/// `Segment` with one leaf → that leaf; with several → `AmbiguousRuns`.
pub fn resolve_leaf(handle: &str) -> Result<LeafRef, ResolveError>;
/// `@name` → the one completed run under `root` carrying `name`.
pub fn resolve_label(root: &Path, name: &str) -> Result<LeafRef, ResolveError>;

pub struct ResolvedFit {
    pub segment: PathBuf,        // the problem folder: archive, sidecar, model IR
    pub leaf: Option<PathBuf>,   // the run the handle named, when it named one
    pub config: FitConfig,
}
impl ResolvedFit { pub fn run_dir(&self) -> &Path; } // leaf, else segment
```

`FitView::read` on a leaf directory returns the segment's view restricted to
that one leaf, so every reader that takes a fit directory accepts a leaf.

Two recorded-not-hashed fields join the leaf's `run.json` `inputs`:

```rust
"config_identities": ["<hex>"] // config_identity_hash of each fit.toml the run was
                               // written or reused from (a cache hit from a second
                               // config with the same run identity adds its own)
"starts_source": {           // present iff `starts` read a stored fit or file
    "handle": "@scout",      // as written
    "run_id": "<hex>",       // the source leaf's run_id; null for a file named directly
    "artifact": "fit_state.toml"
}
```

`deps` already carries the source leaf's `run_id` and the consumed file's digest
(`cas::cas_leaf_file_dep`); `starts_source` is the readable link from the handle
to that run, and is what `fit summary` prints. Neither field enters any hash.

## Behaviour by verb

| surface                        | after                                                                                                                                                                                                                                                 |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit run --label L`            | refused when the invocation writes more than one leaf (several `fit_seeds`, synthetic datasets, sweep points); refused before any sampling when `L` already names another run under the store root, or this run is already stored under another label |
| `camdl label <run_id> L`       | a fit run is addressed by its leaf `run_id`; refused when `L` names another run or the run already has a different label; a fit-id prefix is refused with its runs listed                                                                             |
| `@L` in any handle             | that one leaf; a `fit.toml` names the runs whose `config_identities` hold its identity, narrowed to this camdl build's runs when that leaves one                                                                                                      |
| `fit summary <handle>`         | a run handle summarises that run; a problem folder holding several runs of one method is refused with the runs listed; a target with no completed run exits 1                                                                                         |
| `fit predict`, `compare`       | read the named run when the handle names one (the segment-level choice for a problem-folder handle, and where `fit predict` writes, are unchanged and remain gh#896's open half)                                                                      |
| `fit summary` of a sourced run | `seeded from:` names the source's run id beside its handle                                                                                                                                                                                            |

What is not changed: the method identity still folds the `starts` handle as
written, so `@scout` and the scout's directory name the same source but key two
different runs. With labels fixed this is redundant rather than wrong; removing
it re-keys every sourced run and is left for a separate, confirmed change.
