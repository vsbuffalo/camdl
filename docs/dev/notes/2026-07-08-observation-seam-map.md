# Observation-handling seam map

Date: 2026-07-08 Project: camdl Tags: observations, inference, cli,
consolidation, seam

## Context / question

Every command that scores or samples an observation model re-implements the same
three-part concept:

1. **resolve** which model observation streams are bound to data,
2. **load** each stream's per-observation values + per-observation auxiliary
   columns (aux, e.g. a binomial survey denominator `n = n_examined`),
3. **build** the multi-stream observation model / sampler.

This note tabulates, per command, exactly how each of the three parts is done
today (HEAD = `20db1260`, which in this worktree equals `main`), with
`file:line` citations verified against the code, and where each diverges from
the canonical fit-runner path. It is the evidence base for the consolidation
proposal `2026-07-08-observation-seam-consolidation.md`.

**Class of the discrepancies below: code-vs-code** (each command is a fork of
the same behaviour that has drifted). Fixes are code changes with tests pinning
the agreement, not doc edits.

## The canonical path (`camdl fit run`)

`rust/crates/cli/src/fit/runner.rs`, `FitRunConfig::build` (line 119) and
`build_obs_model` (line 714).

- **Resolve — BY SOURCE.** `source_labels` = the deduped, sorted set of
  `o.source` over `model.observations` (runner.rs:285–288).
  `effective =
  data_spec.effective_observations(&source_labels)`
  (runner.rs:289). Bound obs blocks = `model.observations` filtered by
  `effective.contains_key(&o.source)`, sorted by name (runner.rs:306–310). A
  bound source naming no real stream is a hard error (runner.rs:315–328).
- **Load — via `load_observations`** (runner.rs:1276), which dispatches
  long-form (stratified `: dim` columns → `pfilter::load_long_form_stream`,
  slices rows per stratum, builds aux) vs wide (`load_data_tsv_column_cells` +
  `stream_aux_columns` + `load_stream_aux`) (runner.rs:1290–1312). Returns
  `(Vec<Observation>, Vec<Option<ObsCell>>,
  Vec<Vec<(String,f64)>>)` = (dense
  placeholder, authoritative cells, aux). Holes (`NA`) supported; aux carried.
- **Build — `ObsStream` (runner.rs:25) → `StreamSpec` (runner.rs:714–732) →
  `BoundObs::bind` → `MultiStreamObsModel`.** `build_obs_model` maps each
  `ObsStream` to a
  `StreamSpec { projection, ir_model, observations: cells,
  obs_times, aux }`,
  feeding each stream its OWN schedule; `bind` (multi_stream_obs.rs:591) merges
  to the union axis and validates aux (present-together-or-hole, binomial `n>0`,
  `value ≤ n`; multi_stream_obs.rs:682–756).

`load_observations` is `pub(crate)` and is the intended shared loader; only
`predict.rs` currently calls it besides runner itself
(`git grep -n load_observations` → runner.rs:1291 def-site call,
predict.rs:1524).

## Per-command table

| Command                                     | Resolve                                                                                                                                       | Load values + aux                                                                                                  | Build model / sampler                                                                                                                 | Divergence from canonical                                                                                                                          |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **fit run** `runner.rs`                     | BY SOURCE (285–310)                                                                                                                           | `load_observations` (350) — long-form + aux                                                                        | `ObsStream`→`StreamSpec`→`bind` (714–742)                                                                                             | canonical                                                                                                                                          |
| **pfilter** `pfilter.rs`                    | BY NAME: `bound_streams` from `resolve_data_specs`/`load_data_observations_from_fit_toml` keyed by name; `find(o.name==sname)` (191–192, 311) | inline loop (187–257) — **duplicates** `load_observations` logic (long-form dispatch 202–211, wide+aux 212–243)    | inline `StreamSpec` build (430–439) + `bind` (440)                                                                                    | forked resolve (by name, misses fit-toml `[data.observations]` indexed families); duplicated load; duplicated build; `--flow` override (338–350)   |
| **profile** `profile.rs`                    | BY NAME: same `bound_streams`; `find(o.name==sname)` (435, 508–509)                                                                           | inline loop (501–529) using `load_data_tsv_column` — **wide, dense-only, NO holes, NO aux, NO long-form dispatch** | inline `StreamSpec::dense` build (791–826)                                                                                            | most forked: cannot bind long-form/indexed families; rejects `NA`; no aux; forked resolve + build; `--flow` override (794–802)                     |
| **predict** `predict.rs` (observed overlay) | BY SOURCE via `load_leaf_obs` (1491) → `effective_observations(model_obs_names)` (1502) + `effective.get(&o.source)` (1518)                   | `load_observations` (1523) — long-form + holes; **discards aux** (`_aux`, 1523)                                    | n/a (observed series only, for reference bands)                                                                                       | resolve BY SOURCE like canonical but with `model_obs_names` arg (not `source_labels`); aux dropped (only needs observed values)                    |
| **predict** `predict.rs` (y_rep sampling)   | iterates `model.observations` directly; times from `leaf_times` mapped via `leaf_matches` (1091–1101, 723–724)                                | n/a (samples from trajectory, not a file)                                                                          | `compile_obs_sample_pf(obs_ir, compiled, params)` per stream (728); sampler called with `&snap.int_state.counts` and **NO aux** (737) | sampling path hard-codes empty aux (obs_model.rs:407–411) → `binomial(n=n_examined)` posterior predictive draws all-zero (memo item 3, still live) |
| **simulate --obs** `main.rs`                | iterates `model.observations` with an `emit_schedule` (1513–1517, 2022)                                                                       | n/a (emission, no data file)                                                                                       | `compile_obs_sample_pf` (1519, 2024); empty aux                                                                                       | correct-by-design: emission has no data file → no aux source (honest limitation, obs_model.rs:407–411)                                             |
| **synthetic** `synthetic.rs`                | iterates `model.observations` (148)                                                                                                           | n/a (emission)                                                                                                     | `compile_obs_sample_pf` (148); empty aux                                                                                              | same as simulate --obs (emission, no data file)                                                                                                    |
| **batch** `batch.rs`                        | iterates `model.observations` (1644)                                                                                                          | n/a (emission)                                                                                                     | `compile_obs_sample_pf` (1644); empty aux                                                                                             | same as simulate --obs (emission, no data file)                                                                                                    |

## The resolution divergence, precisely

`camdl` has TWO key spaces for "which stream binds to this file":

- **BY SOURCE** — the `[data.observations]` fit-toml block and the canonical
  fit-runner path key by `o.source` (the `observe … from <label>` label;
  defaults to the stream name when no `from`, spec §2.4).
  `effective_observations` returns `[data.observations]` verbatim (by source)
  for the per-stream form, or maps each passed key → file for the `[data] file`
  shorthand (`config_v2.rs:396–417`).
- **BY NAME (with family-root expansion)** — the CLI `--data NAME=PATH` /
  `--data PATH --obs ROOT` surface. `resolve_data_specs` (util.rs:1162)
  validates each key as a leaf name (exact) or a family root (`ROOT_` prefix),
  and emits **leaf names** (util.rs:1238–1248, 1316–1321). pfilter/profile then
  bind `o.name == sname`.

Consequences verified:

- **CLI `--data cases=FILE` on a long-form indexed family WORKS** for pfilter:
  `resolve_data_specs` expands `cases` → leaves `cases_urban`, `cases_rural`
  (leaf names), pfilter finds each by name, `is_long_form_stream` slices per
  stratum. Pinned by `rust/crates/cli/tests/long_form_stratified_obs.rs`
  (`long_form_family_binds_routes_and_skips_hole`).
- **fit-toml `[data.observations] cases = "f.tsv"` on the SAME family FAILS**
  for pfilter/profile:
  `load_data_observations_from_fit_toml(fit, model_obs_names)` →
  `effective_observations` returns `{cases: f.tsv}` (by source, **no family
  expansion**; pfilter.rs:892–912, config_v2.rs:413–415).
  `bound_streams =
  [(cases, f.tsv)]`; `find(o.name=="cases")` matches no leaf
  (leaves are `cases_urban`/`cases_rural`) → "bound stream 'cases' has no
  matching IR observation block" (pfilter.rs:311–320). The canonical fit-runner
  filters `o.source == "cases"` and binds both leaves (runner.rs:306–310) — the
  correct behaviour. This is the fit-toml indexed-binding bug.
- **`observe … from <label>` (source ≠ name)**: the CLI `--data` path resolves
  by name, the fit-toml path resolves by source. These select the same leaves
  today only when the model uses no `from` (name == source) or when the CLI
  adapter maps name → source. Any consolidation onto a single by-source seam
  must preserve the CLI `--data` family-root surface.

## The sampling seam

`compile_obs_sample_pf` (obs_model.rs:389) returns a closure
`Fn(projected, t, counts, rng) -> f64` that hard-codes `&[]` for aux
(obs_model.rs:407–411). Its core, `sample_obs_resolved` (obs_model.rs:419),
ALREADY takes `aux: &[(String,f64)]` and threads it into `EvalCtx`
(obs_model.rs:423, 432) — so the sampler substrate supports aux; only the
`compile_obs_sample_pf` wrapper withholds it. Five call sites (predict.rs:728,
main.rs:1519 & 2024, batch.rs:1644, synthetic.rs:148); four are emission (no
data file, empty aux correct); only `fit predict` has an observed aux available
(loaded then dropped by `load_leaf_obs`, predict.rs:1523) that it should thread.

## Divergence from the driving memo (premise check)

The consolidation memo assumed three stopgap fixes had landed. Verified against
HEAD they have NOT:

- **profile does not route through `load_observations`** —
  `grep -n
  load_observations profile.rs` → NONE. Profile still uses
  `load_data_tsv_column` (dense, no holes/aux/long-form; profile.rs:518) and
  `StreamSpec::dense` (profile.rs:809).
- **`--flow` override still present** in both pfilter (`grep -c flow_name` → 6)
  and profile (→ 4).
- **`compile_obs_sample_pf` has no aux argument** (obs_model.rs:389–393) — the
  predict all-zero survey-predictive bug is still live.

So the starting state is materially further back than the memo framed, and the
consolidation is not a byte-identical refactor: routing profile/pfilter onto the
by-source seam is a real behaviour change (fixes the fit-toml indexed bug; gives
profile long-form/aux/holes). See the proposal for the phased plan and the
per-phase risk classification.

## Next

Proposal `docs/dev/proposals/2026-07-08-observation-seam-consolidation.md`
specifies the single seam (`resolve_and_load_obs_streams` + a shared
`StreamSpec` builder), decides how the CLI `--data` and fit-toml key spaces
reconcile, and phases the work by risk. The byte-identical extraction (Phase 0)
is wired into fit-runner now; the behaviour-changing routing (pfilter, profile,
predict-sampling aux) is gated on maintainer review of the proposal.
