# Observation-seam consolidation

Date: 2026-07-08 Status: Phase 0 implemented (`refactor/observation-seam`);
Phases 1–3 gated on review Tags: observations, inference, cli, seam,
consolidation

## Problem

The concept "resolve which model observation streams are bound to data, load
each stream's per-observation values + aux, build the multi-stream observation
model" is implemented three-to-four different ways across the CLI. Evidence and
`file:line` citations are in the companion map
[`docs/dev/notes/2026-07-08-observation-seam-map.md`](../notes/2026-07-08-observation-seam-map.md);
this proposal assumes the reader has it open. The forks have produced a run of
silent-wrong / catch-22 bugs (fit-toml indexed families not binding under
pfilter/profile; profile unable to load long-form or aux; `fit predict` drawing
all-zero survey predictives). Each is a facet of one fork: the same concept,
forked per command, drifting.

The fix is ONE shared seam every command routes through, so a data-column
feature cannot be live in one command and silently absent in another.

## Non-goal / scope note (verified)

The three stopgap fixes the driving memo assumed had landed are NOT on HEAD
(`20db1260`): profile does not call `load_observations`; `--flow` is still
present in pfilter/profile; `compile_obs_sample_pf` has no aux argument (map §
"Divergence from the driving memo"). Consequently, routing pfilter/profile onto
the shared by-source seam is **not** byte-identical — it fixes real bugs and,
for profile, adds long-form/aux/holes support. This proposal classifies each
phase as _pure_ (byte-identical) or _behaviour-change (bug fix)_ so the
maintainer can sign off deliberately. Phase 0 (pure) is implemented; Phases 1–3
(behaviour-changing) are specified here and gated on review.

## The single seam

Two shared functions in `rust/crates/cli/src/fit/runner.rs`, both `pub(crate)`,
extracted from the canonical fit-runner path:

```rust
/// Resolve the bound observation streams (BY SOURCE) and load each one's
/// per-observation values + aux, returning one `ObsStream` per bound leaf.
///
/// `effective` maps observation SOURCE → data-file path (the `[data]`
/// resolution already done by the caller). The function filters
/// `model.observations` to the bound sources, hard-errors on a bound source
/// that names no real stream, and for each bound leaf dispatches
/// `load_observations` (long-form vs wide, holes + aux), resolves its
/// projection, and runs the per-stream origin/first-window guards.
///
/// Conditioning-window (`condition_from` / W329) handling is NOT here — it is
/// fit-specific and stays in `FitRunConfig::build`.
pub(crate) fn resolve_and_load_obs_streams(
    model: &ir::Model,
    compiled: &CompiledModel,
    effective: &indexmap::IndexMap<String, String>,
    dt: f64,
    time_opts: &crate::caltime_load::TimeOpts,
) -> Result<Vec<ObsStream>, String>;

/// Map loaded `ObsStream`s to the `StreamSpec`s that `BoundObs::bind` consumes
/// (projection, ir_model, per-stream cells, per-stream schedule, aux). Single
/// source of the `ObsStream -> StreamSpec` mapping for every consumer.
pub(crate) fn stream_specs_from_obs_streams(
    streams: &[ObsStream],
) -> Vec<sim::inference::multi_stream_obs::StreamSpec>;
```

`FitRunConfig::build` keeps ownership of: `source_labels` +
`effective_observations` (the by-source `effective` map), the empty-streams
check, and all `condition_from` / W329 / obs-alignment logic. It calls
`resolve_and_load_obs_streams` for the resolve+load+build-`ObsStream` block, and
`build_obs_model` calls `stream_specs_from_obs_streams`.

### Why the seam takes `effective` (by SOURCE), not the CLI `--data` result

The trust boundary is: raw CLI/toml `--data` input → a validated by-source
`IndexMap<source, path>`. Parse it once at the boundary, then the seam never
re-resolves. `effective_observations(&source_labels)` already produces this map
for fit-runner and (per Phase 1) for pfilter/profile's `--fit` fallback. The CLI
`--data NAME=PATH` surface (`resolve_data_specs`, by name/family-root) is
adapted to the by-source map by a single boundary function:

```rust
/// Convert a validated `--data` binding list (leaf-name → path, family roots
/// already expanded to leaf names by `resolve_data_specs`) into the by-source
/// `effective` map the seam consumes: for each bound leaf name, look up its
/// `model.observations[..].source` and insert (source → path). Leaves of one
/// family share a source, so the map dedups to one entry per source/file.
pub(crate) fn data_bindings_to_effective(
    model: &ir::Model,
    bindings: &[(String, std::path::PathBuf)],
) -> Result<indexmap::IndexMap<String, String>, String>;
```

This **preserves the CLI `--data` family-root surface** (the passing
`long_form_stratified_obs` test): `--data cases=FILE` still expands via
`resolve_data_specs`, then maps the leaves' shared source → FILE, and the seam
binds every leaf of that source. It also preserves `observe … from <label>`
(name ≠ source) because the adapter resolves name → source before handing off.

### Decision: predict-sampling and fit-scoring do NOT share stream resolution

They share the `StreamSpec` builder and the sampling substrate, but not
resolution — and that is correct, not a gap:

- **Scoring** (fit run / pfilter / profile) resolves DATA-bound streams: which
  `o.source` is bound to a file. Data-driven.
- **Emission sampling** (simulate `--obs`, synthetic, batch) resolves
  SCHEDULE-bound streams: every `model.observations` leaf with an
  `emit_schedule`, independent of any data file. Schedule-driven, dataless.
- **Predictive sampling** (`fit predict`) samples the DATA-bound leaves at the
  observed times — it borrows the _scoring_ resolution (`load_leaf_obs`, already
  by source) for the leaves+times, then samples.

Forcing emission and scoring through one resolver would be a leaky
god-abstraction (an `emit_schedule` stream has no `source`/file; a scored stream
may have no `emit_schedule`). The natural seam is: **one resolver for data-bound
streams** (`resolve_and_load_obs_streams`, used by fit run/pfilter/profile and
the observed half of predict), **one shared `StreamSpec` builder**
(`stream_specs_from_obs_streams`), and **one sampling substrate**
(`compile_obs_sample_pf`, extended to accept per-eval aux so the predict path
can thread the observed denominator). Emission sampling keeps its
schedule-driven `model.observations` iteration — it is a distinct algorithm, not
a fork of scoring.

## The sampling-aux plumbing (Phase 3)

`compile_obs_sample_pf` (obs_model.rs:389) hard-codes `&[]` aux; its core
`sample_obs_resolved` (obs_model.rs:419) already accepts aux. Extend the closure
to accept a per-eval aux slice:

```rust
// before: Fn(projected, t, counts, rng) -> f64            (aux = &[])
// after:  Fn(projected, t, counts, aux, rng) -> f64        (aux threaded)
```

- **Emission callers** (simulate `--obs`, synthetic, batch) pass `&[]` — a
  synthetic emission has no data file, so a data-dependent aux (survey
  denominator) genuinely has no source; the honest behaviour is unchanged (an
  `ObsColumnRef` errors at eval). This is not a silent gap: emission of a
  survey-denominator likelihood without a denominator is an error today and
  stays one.
- **Predict caller** threads the observed aux. `load_leaf_obs` currently drops
  aux (`_aux`, predict.rs:1523); Phase 3 keeps it on `LeafObs`, carries it to
  `PredictiveSink` alongside `leaf_times`, and passes `aux[ti]` per obs index at
  the sampler call (predict.rs:737). Red→green test: a
  `binomial(n = n_examined)` fit's `fit predict` draws non-degenerate `y_rep`
  (bounded by `n`), not all-zero.

## Exactly which fork each consumer drops

| Consumer                 | Fork dropped                                                                                               | Now routes through                                                                                                                                    | Phase / class                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **fit run**              | inline resolve+load+build in `build`                                                                       | `resolve_and_load_obs_streams` + `stream_specs_from_obs_streams`                                                                                      | 0 · pure                                         |
| **pfilter**              | by-name resolve (191, 311); duplicated inline load (187–257); inline `StreamSpec` build (430–439)          | `--fit` via `effective_observations(&source_labels)`; `--data` via `data_bindings_to_effective`; then `resolve_and_load_obs_streams` + shared builder | 1 · bug fix (fit-toml indexed families now bind) |
| **profile**              | by-name resolve (435, 508); wide/dense/no-aux/no-long-form load (501–529); `StreamSpec::dense` build (809) | same as pfilter                                                                                                                                       | 2 · bug fix (gains long-form/aux/holes)          |
| **predict** (observed)   | already by source; drops aux                                                                               | `resolve_and_load_obs_streams`; keep aux on `LeafObs`                                                                                                 | 3 · behaviour add (aux retained)                 |
| **predict** (sampling)   | empty-aux closure                                                                                          | `compile_obs_sample_pf` w/ per-eval aux                                                                                                               | 3 · bug fix (non-degenerate survey predictive)   |
| simulate/synthetic/batch | — (emission, correct as-is)                                                                                | `compile_obs_sample_pf` w/ `&[]` aux                                                                                                                  | 3 · pure (signature-only)                        |

Also removed in Phases 1–2: the `--flow` projection override in pfilter
(338–350) and profile (794–802). The map records it as a stale surface; removing
it is a breaking CLI change and gets a `docs/language-changes.md` entry (old →
new: `--flow NAME` removed; declare the projection in the model's
`observe … incidence(NAME)` block). If the maintainer wants `--flow` kept, it is
a per-command projection override applied AFTER `resolve_and_load_obs_streams`
returns (single-stream only, as today) — it does not block the seam.

## Phasing (by risk)

- **Phase 0 — pure extraction, wired now (IMPLEMENTED).** Extract
  `resolve_and_load_obs_streams` + `stream_specs_from_obs_streams`; route
  `FitRunConfig::build` + `build_obs_model` through them. Byte-identical:
  goldens unchanged (`make update-golden` must be a no-op), full `make test`
  green. This lands the seam wired into its first real consumer (fit-runner) — a
  "wired now" landing, not a speculative primitive.
- **Phase 1 — route pfilter (bug fix).** Add `data_bindings_to_effective`; route
  pfilter through the seam for both `--fit` and `--data`. Red→green test
  `pfilter_indexed_long_obs` (mirror `long_form_stratified_obs.rs`): a fit-toml
  `[data.observations]` indexed family binds all leaves and scores a finite
  loglik — RED on HEAD ("no matching IR observation block"), GREEN after.
  Existing pfilter tests (`long_form_stratified_obs`, `pfilter_cas`,
  `pfilter_trajectories`, `gh191_*`) stay green.
- **Phase 2 — route profile (bug fix + capability add).** Route profile through
  the seam. This gives profile long-form/aux/holes. Risk: profile's downstream
  focal-grid path (PF/PMMH/ODE-MLE) and its diagnostics currently assume dense,
  NA-free streams (`profile.rs:803–804` comment). Before routing, verify the
  focal-grid consumers handle holes (they go through the same
  `MultiStreamObsModel`, which is hole-correct) and that profile's per-cell
  output paths do not read a dense placeholder where a hole belongs. Test
  `profile_indexed_long_obs` mirrors Phase 1. Keep `profile_multi_stream`,
  `profile_priors`, `profile_pmmh`, `profile_diagnostics` green.
- **Phase 3 — sampling aux (bug fix).** Thread per-eval aux through
  `compile_obs_sample_pf`; keep aux on `LeafObs`; wire predict. Emission callers
  pass `&[]`. Red→green `predict_binomial_survey_nonzero`.

## Run-identity / CAS impact

None expected. The seam changes HOW streams are resolved/loaded/built, not WHAT
bytes are stored: the loaded cells/aux/schedule for a given (model, data, dt)
are identical, and `run_id` hashes model/config/params/data, not the loader
internals. Phase 1/2 make a _previously-erroring_ input (fit-toml indexed family
under pfilter/profile) start producing output — that is a new successful run,
not a re-key of an existing one, so no CAS collision arises. **Gate:** each
phase runs the full `make test` including the integration/CAS suites; any golden
or `run_id` movement STOPS the phase and is reported, per the goldens rule.

## Decisions (no open questions)

1. Seam consumes a by-source `effective` map; CLI `--data` is adapted at the
   boundary (`data_bindings_to_effective`), preserving the family-root surface.
2. Predict-sampling and fit-scoring share the `StreamSpec` builder and the
   sampling substrate, NOT stream resolution — emission is schedule-driven and
   dataless, scoring is data-driven; unifying them would be a leaky abstraction.
3. `compile_obs_sample_pf` gains a per-eval aux argument; emission callers pass
   `&[]` (honest), predict threads the observed denominator.
4. `--flow` is removed from pfilter/profile with a `language-changes.md`
   migration entry; if kept, it applies after the seam as a single-stream
   projection override (does not block consolidation).
5. Phase 0 is byte-identical and landed now; Phases 1–3 are behaviour-changing
   bug fixes, each with a red→green test, gated on maintainer review of this
   proposal (which the repo treats as the review gate for the routing work).
