# camdl types reference & flow guide

Every command stores its results in the run store, the `store` crate
(`rust/crates/store`), whose design is
`docs/dev/proposals/2026-11-20-run-store-api.md`. camdl's kinds on it are in
`cli/src/camdl_store.rs` (the ambient identity, the kinds, the root and the
content artifacts), `cli/src/sim_store.rs`, `cli/src/pfilter_store.rs`,
`cli/src/fit_store.rs` and `cli/src/profile_store.rs`. This memo is a pointer to
where each piece of the type model lives, so a reader can go to the code rather
than a catalogue that drifts.

## Reading the store

- `cli/src/browse.rs` — `list` / `show` / `cat` over the run store's results.
- `cli/src/fit/fit_view.rs` — `FitView` / `FitMethodView`: the projection that
  aggregates a fit segment's method leaves into the headline numbers fit
  consumers read, each leaf carrying its `method`, `method_hash` and `run_id`.
- `cli/src/run_meta.rs` — the cross-cutting value types readers/writers share
  (`FitAlgorithm`, `InferenceBackend`, the provenance records, and the fit-level
  `FitSidecar` written as `fit.meta.json`).

## Fit config schema

`cli/src/fit/config_v2.rs` —
`FitConfig { problem: Problem, inference:
Inference }` is the only fit-config
schema; `Inference` holds the one
`Method
{ algorithm: Algorithm, starts: Option<ChainStarts> }`, and `Algorithm`
is the tagged enum (`IF2` / `PGAS` / `PMMH` / `Mh` / `PFilter` / NLopt).
`ChainStarts` (`fit/starts.rs`) is `Spread` or `Point`. The fit identity hashes
the problem alone (so a second method file shares it); a sourced `starts` folds
its source's content into the method leaf's `deps`.
