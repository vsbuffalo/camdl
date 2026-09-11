# camdl types reference & flow guide

The run-identity and CAS-store type model now lives in the `runid` crate and is
documented authoritatively at the source. This memo is a pointer to where each
piece of the type model lives, so a reader can go to the code rather than a
catalogue that drifts.

## Run identity and the content-addressed store (`runid` crate)

- **`runid::ArtifactKind`** (`rust/crates/runid/src/kind.rs`) — the top "type"
  partition of the store (`sims/`, `fits/`, `pfilters/`, `surveys/`,
  `profiles/`, `obs/`, `projections/`) and the `kind` discriminator in
  `run.json`. `run_id(kind, levels)` composes a leaf's address from the
  per-level hashes.
- **`runid::RunRecord`** (`record.rs`) — the on-disk `run.json`: the `run_id`
  leaf address, the ordered per-level `levels` (each a `LevelId` =
  `{name, label, hash}`), upstream `deps`, the `kind`, a `RunStatus` (Running →
  Completed/Failed), provenance (`argv`, version, timestamps), and a
  recorded-not-hashed `inputs` payload for display.
- **`runid::Layout`** (`layout.rs`) — the readable factored store path. Each
  segment is `{label}-{hash8}`; the label is provenance, the `hash8` is
  identity. `store_path(root, kind, levels)` builds the nested path.
- **Identity inputs** (`inputs.rs`) — the per-level content types (`FitDigest`,
  `DataDigest`, `StageConfig`/`StageLevel`, `Seed`, `EngineVersion`, `Deps`,
  `ArtifactRef`, …) and their canonical hashing.
- **The store** (`store.rs`) — the filesystem-backed CAS rooted at `results/`:
  streaming claim (staging dir + atomic finalize), prefix-collision
  disambiguation, recursive artifact manifest at commit.

The end-to-end path shape and the consumer contract for `run.json` are in
[`cas-path-shape-contract.md`](cas-path-shape-contract.md).

## CLI-side identity resolution

Each CAS-emitting command resolves CLI inputs into `runid` levels at one site:

- `cli/src/resolve.rs` — `resolve_trajectory`: `simulate` / `batch run` →
  `ArtifactKind::Sim` levels (`model` / `config` / `params` / `scenario` /
  `seed`).
- `cli/src/fit/cas.rs` — `resolve_fit_stage`: one method leaf of a fit →
  `ArtifactKind::FitStage` levels (`fit` / `method` / `seed`). The kind string
  and the `Stage*` type names are kept: the string is read back from every store
  on disk, and the type names are in no artifact.
- `cli/src/profile_cas.rs` — `resolve_profile_point`:
  `ArtifactKind::ProfilePoint` levels (`profile` / `point` / `method` / `seed` /
  `start`).
- `cli/src/pfilter_cas.rs` — pfilter-eval identity (`ArtifactKind::Pfilter`).
- `cli/src/survey.rs` — survey identity (`ArtifactKind::Survey`).

`cli/src/cas/typed.rs::ContentHash` is the small newtype these sites use to keep
content hashes type-distinct from arbitrary strings.

## Reading the store

- `cli/src/cas_read.rs` — generic `RunRecord` walk: a parseable `run.json` is
  the only discovery signal (no hardcoded level depth).
- `cli/src/cas_index.rs` — the derived `results/index.json` (`run_id` → leaf)
  and `camdl reindex`; `run.json` is truth, the index is a fast lookup.
- `cli/src/browse.rs` — `list` / `show` / `cat` projections over discovered
  records.
- `cli/src/fit/fit_view.rs` — `FitView` / `FitStageView`: the projection that
  aggregates a fit segment's method leaves into the headline numbers fit
  consumers read, each leaf carrying its `method`, `method_hash` and `run_id`.
- `cli/src/run_meta.rs` — the cross-cutting value types readers/writers share
  (`FitAlgorithm`, `InferenceBackend`, `SurveyEvalMethod`, the provenance
  records, and the fit-level `FitSidecar` written as `fit.meta.json`).

## Fit config schema

`cli/src/fit/config_v2.rs` —
`FitConfig { problem: Problem, inference:
Inference }` is the only fit-config
schema; `Inference` holds the one
`Method
{ algorithm: Algorithm, starts: Option<ChainStarts> }`, and `Algorithm`
is the tagged enum (`IF2` / `PGAS` / `PMMH` / `Mh` / `Nuts` / `PFilter` /
NLopt). `ChainStarts` (`fit/starts.rs`) is `Spread` or `Point`. The fit identity
hashes the problem alone (so a second method file shares it); a sourced `starts`
folds its source's content into the method leaf's `deps`.
