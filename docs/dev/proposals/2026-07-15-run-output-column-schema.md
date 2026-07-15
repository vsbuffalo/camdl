# Machine-readable column schema for every run output, in `run.json`

Date: 2026-07-15\
Status: Proposed

## Summary

camdl writes tabular outputs from every command — `simulate` writes `traj.tsv`
and `obs.tsv`, `fit` writes `draws.tsv` and per-chain `trace.tsv`, `fit predict`
writes predictive/observed bands, `survey`/`profile` write landscape/grid
tables. A consumer that wants to plot or join any of these must reverse-engineer
the file's columns: which column is the x-axis, which is a grouping key, which
are model quantities versus sampler diagnostics. The schema is knowable — it is
a pure function of the command, the method, and the model — but it lives only
implicitly, spread across the writers, and nothing declares it in a form a tool
can read.

The run store already commits to the opposite principle for its observation and
predictive surfaces: _"a consumer reads a machine-readable schema, with no
run-store, DSL, or likelihood knowledge"_ (`docs/camdl-run-spec.md` §2.2.1,
§2.2.2). The tabular outputs are the surface where that principle is not yet
applied, and the gap is not hypothetical — it produced a real display bug (§4).

This proposal applies the principle uniformly. Every tabular output gets a
machine-readable column schema, expressed with a small **closed vocabulary of
column roles**, and surfaced centrally in the one manifest every command already
writes: `run.json`. The rule is **declare, don't rename** — columns keep their
current names; the schema states what each one means.

## 1. Background: the run store and its manifests

Every camdl command writes a content-addressable leaf under `results/` with a
`run.json` manifest. `ArtifactKind` (`rust/crates/runid/src/kind.rs`) enumerates
the leaf kinds: `Sim`, `SimEnsemble`, `FitStage`, `Pfilter`, `Survey`,
`ProfilePoint`, `Obs`, `Projection`. So `run.json` is **universal** — a one-off
`simulate` leaf has one (`kind: "sim"`), exactly as a fit stage does
(`kind: "fit_stage"`).

`RunRecord` (`rust/crates/runid/src/record.rs`) is what serializes to
`run.json`:

```rust
pub struct RunRecord {
    pub format_version: u16,
    pub kind: ArtifactKind,
    pub run_id: ContentHash,
    pub hash_version: u16,
    pub ir_version: String,
    pub engine_version: String,
    pub levels: Vec<LevelId>,      // factored identity, path order
    pub deps: Vec<ArtifactRef>,    // consumed upstreams
    pub status: RunStatus,
    pub artifacts: BTreeMap<String, FileChecksum>,   // this leaf's own files
    pub children: BTreeMap<String, Vec<ContentHash>>,
    pub inputs: serde_json::Value,
    pub provenance: Provenance,
}

pub struct FileChecksum { pub bytes: u64, pub mtime: String, pub digest: ContentHash }
```

The `artifacts` map already enumerates every file the leaf wrote (`traj.tsv`,
`chain_1/trace.tsv`, …) with `{bytes, mtime, sha256}`, populated generically by
the store walking the leaf directory at finalize. It records each file's
_bytes_; it does not understand any file's _columns_. That is the gap.

Two fit-adjacent surfaces already carry declared schemas, and are the model this
proposal follows:

- **`fit.meta.json`** (fit-level) carries the observation `schema` — the model's
  streams and dimensions — _"so a consumer can facet any stream and label panels
  with no DSL parsing"_ (§2.2.1). This is fit-only; a `simulate` leaf has none.
- **`fit predict`** writes tidy `predictive/<stream>.tsv` /
  `observed/<stream>.tsv` a consumer joins and plots _"with no run-store, DSL,
  or likelihood knowledge"_ (§2.2.2).

And two output files already declare their own columns, non-uniformly:
`trajectories.json` (sibling to `chain_N/trajectories.tsv`) carries a
`columns: Vec<String>` list plus a calendar/granularity block; `quantities.json`
declares per-quantity `{index_dims, shape, source, reduce, unit, censoring}`.
These are good prior art for the field shape — but they are per-file, cover only
two outputs, and are invisible to a consumer reading the central manifest.

## 2. The gap: every tabular output is undeclared centrally

A consumer that wants "the x-axis of this file" has no machine-readable answer.
The problem is sharpest for the sampler trace, where the index column is named
four different ways across methods (the writer lets each method name it):

| kind         | file                                 | index column       | value columns                                     | diagnostic columns                                                                                                                                       |
| ------------ | ------------------------------------ | ------------------ | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sim`        | `traj.tsv`                           | `t` (time)         | int/real compartments, `flow_*`                   | —                                                                                                                                                        |
| `sim`        | `obs.tsv`                            | `time`             | `<stream>`                                        | —                                                                                                                                                        |
| `fit_stage`  | `draws.tsv`                          | `draw` (iteration) | `<params>` (pgas/pmmh: est+fixed; nuts: est only) | `log_posterior`, `log_likelihood`                                                                                                                        |
| `fit_stage`  | `chain_N/trace.tsv` (pgas)           | `sweep`            | `<params>`                                        | `log_complete_data_ll`, `trajectory_renewal`, `transition_ll`, `obs_ll`, `tree_depth`, `n_leapfrog`, `step_size`, `accept_stat`, `n_divergent`, `energy` |
| `fit_stage`  | `chain_N/trace.tsv` (pmmh, mh)       | `step`             | `<params>`                                        | `log_likelihood`, `accepted`                                                                                                                             |
| `fit_stage`  | `chain_N/trace.tsv` (nuts)           | `draw`             | `<params>`                                        | `log_likelihood`, `divergent`, `tree_depth`                                                                                                              |
| `fit_stage`  | `chain_N/parameter_traces.tsv` (if2) | `iteration`        | `<params>`                                        | `loglik`, `if2_perturbed_loglik`                                                                                                                         |
| `projection` | `predictive/<stream>.tsv`            | `time`             | `q05…q95`                                         | `rhat_max`, `ess_min`, `n_draws`                                                                                                                         |
| `survey`     | `landscape.tsv`                      | `point_id`         | `<params>`                                        | `loglik`, `loglik_se`, `mean_ess`, `n_replicates`                                                                                                        |

A consumer wanting "the iteration axis" must know the method to know the column
name; one wanting "the estimated parameters" must know pgas/pmmh `draws.tsv`
carry fixed columns while nuts does not. None of this is declared.

## 3. Design

### 3.1 A closed column-role vocabulary

Each column of every camdl TSV is tagged with one role from a small, fixed set:

```rust
enum ColumnRole {
    Time,            // physical/calendar axis: t, time, date
    Iteration,       // sampler/optimizer axis: sweep, step, draw, iteration, point_id
    Chain,           // MCMC chain key
    Replicate,       // ensemble/batch replicate key
    Scenario,        // scenario key
    Dimension,       // a stratification key: patch, age
    State,           // a compartment count/value: S, I, R
    Flow,            // a transition flow: flow_infection
    Incidence,       // inc_<stream>
    ParamEstimated,  // a sampled model parameter
    ParamFixed,      // a held-constant model parameter
    Observable,      // an observation stream's value
    Quantile,        // q05 … q95 predictive band
    Diagnostic,      // loglik, log_posterior, ESS, rhat, accepted, n_divergent, …
}
```

The vocabulary is the load-bearing part: it turns rendering any camdl output
into a mechanical rule — x-axis = `Time` or `Iteration`; group by
`Chain`/`Replicate`/`Scenario`; facet by `Dimension`; series =
`State`/`ParamEstimated`/`Observable`; ribbons = `Quantile`; overlays =
`Diagnostic`. `Time` and `Iteration` are deliberately distinct: a trajectory's
x-axis is physical time, a trace's is a sampler index, and conflating them is
the exact class of error behind §4.

### 3.2 The schema types and where they live

`RunRecord` gains one field, a sibling to `artifacts` keyed by the same paths:

```rust
pub struct RunRecord {
    // …existing…
    pub output_schema: BTreeMap<String, TableSchema>,   // relpath -> schema; empty when none
}

pub struct TableSchema {
    pub role: TableRole,             // Trajectory | Observation | PosteriorCloud | Trace | Predictive | Landscape
    pub columns: Vec<ColumnSpec>,    // in file order
}

pub struct ColumnSpec { pub name: String, pub role: ColumnRole }
```

Serialized (a pgas fit stage):

```json
{
  "output_schema": {
    "draws.tsv": {
      "role": "posterior_cloud",
      "columns": [
        { "name": "chain", "role": "chain" },
        { "name": "draw", "role": "iteration" },
        { "name": "beta", "role": "param_estimated" },
        { "name": "gamma", "role": "param_fixed" }
      ]
    },
    "chain_{n}/trace.tsv": {
      "role": "trace",
      "columns": [
        { "name": "sweep", "role": "iteration" },
        { "name": "log_posterior", "role": "diagnostic" },
        { "name": "beta", "role": "param_estimated" }
      ]
    }
  }
}
```

A consumer reads `run.json` once — already fetched for the run list — and finds
`role == "iteration"` for the x-axis, `role == "chain"` for grouping, and
`param_estimated` for the mixing series, with no per-method knowledge and no
hardcoded column names. `{n}` is the per-chain wildcard.

**Properties** (produced by the classifier on well-formed writer output, pinned
by the classifier's unit tests): a `Trace`/`PosteriorCloud` table has exactly
one `Iteration` column; a `Trajectory`/`Observation`/`Predictive` table has
exactly one `Time` column; at most one `Chain`. These follow from the writers'
headers rather than a construction-time check.

**Why `run.json`, not `fit.meta.json`.** `fit.meta.json` is fit-only; a
`simulate`, `survey`, or `profile` leaf has none, so a fit-only home forks the
mechanism per command — the parallel-seam anti-pattern. `run.json` is the one
manifest every command writes and every consumer already reads. It lives in the
identity-critical `runid` crate, but the field is identity-inert (§5), and this
keeps the whole facility to one seam.

### 3.3 One source of truth (no drift, by construction)

The declaration is built by reading each output file's **actual header** after
the run wrote it, and classifying each column name by role: a chain key, an
iteration axis (`sweep`/`step`/`draw`/`iteration`), a model parameter (estimated
vs fixed by membership in the estimated set), else a diagnostic. Because the
schema is derived _from_ the file it describes, it cannot disagree with it — the
column names and their order are the file's own, and classification needs only
set membership, never a hardcoded per-method column recipe. The only failure
mode is a misclassified role, pinned by unit tests over the classifier. A useful
consequence: a new method or a new column is covered automatically — an
unfamiliar column falls through to `diagnostic` rather than being silently
dropped.

### 3.4 Declare, don't rename

Normalizing `sweep`/`step`/`draw`/`iteration` to a single column name is a
breaking change to `trace.tsv` (golden churn; downstream parsers key off the
names — the PMMH resume path reads the `step` column by name). Once the schema
tags the column `role: iteration`, its on-disk spelling stops mattering to a
consumer. Renaming is a separate, optional follow-up, not bundled here.

## 4. Reproduction: the fit-watcher x-axis bug

The camdl fit watcher plots parameter traces from the data it is served. For run
`ctl_bb_spray_immladder_anchored` (mh/ode, ~2000 sweeps, burn-in at sweep 995,
630 thinned draws kept): `draws.tsv` on disk carries the index (`draw` = the
true sweep number), but the data reaching the plot carried `chain` +
per-parameter value arrays + `n_draws=630` + `warmup_cutoff=995` and **no index
array** — the `draw` column was dropped in transit and nothing declared it was
the x-axis. The tell: `warmup_cutoff=995` is a sweep number (only 630 rows
exist), so the consumer had a warmup marker in sweep units and no sweep axis to
place it on. It plotted values against array position `0..629`: the true range
`~995..1990` compressed, six chains laid end-to-end instead of overlapping, the
`995` line off the axis. A consumer that could read `role: iteration` for `draw`
cannot make this mistake.

## 5. Identity safety

`RunRecord` is **presentation-only — never hashed into any `run_id`/CAS key**,
so `output_schema` is additive and cannot re-key any run. Verified:

- The record's own module doc: _"`RunRecord` is never hashed — identity comes
  only from the `RunInput`; `children`, `artifacts`, and `provenance` are
  recorded-only."_
- The hashing path never references the record or its maps
  (`rg 'artifacts|FileChecksum|RunRecord' rust/crates/runid/src/hash.rs
  rust/crates/runid/src/inputs.rs`
  → no matches). The run_id is `run_id(kind, levels)` over the `LevelId` hashes.
- `fit/cas.rs` reads `run.json` only to extract the already-computed `run_id`;
  the dep digest it folds in is the SHA-256 of the consumed **file's** bytes,
  not the manifest.

## 6. Implementation plan

Staged so the bug-relevant coverage lands first, wired into its consumer:

**v1 — fit** (the surface behind the watcher x-axis bug)

1. Types in `rust/crates/runid/src/record.rs`: `ColumnRole`, `ColumnSpec`,
   `TableRole`, `TableSchema`; `output_schema` on `RunRecord` (default empty,
   skip-serialized when empty, so existing readers and manifests round-trip
   unchanged).
2. A `ResolvedClaim::set_output_schema` setter on the streaming write handle:
   the schema is attached just before `finalize` — the point at which the
   tabular files exist and the estimated/fixed parameter sets are known. No
   store-signature change; the schema rides the record through the one
   `build_record`.
3. `output_schema::fit_output_schema` reads each written file's real header
   (`draws.tsv`, `chain_{n}/trace.tsv`, `chain_{n}/parameter_traces.tsv`) and
   classifies each column (§3.3), wired at the fit stage's finalize.
4. Tests: `classify` role units; a tempdir test that builds the schema from real
   headers and asserts the index/chain/param roles; the record serde round-trip
   and empty-omitted invariant.
5. `docs/camdl-run-spec.md` §2.5 declaring the vocabulary and the
   `run.json.output_schema` shape.

**v1.5 — sim, then the rest**

6. Extend the classifier with the trajectory/observation axes (`t`/`time` →
   time, compartments → state, `flow_*` → flow, `inc_*` → incidence) and wire
   `sim` (`traj.tsv`, `obs.tsv`) at its finalize — the same seam.
7. `projection` (predictive/observed), `survey` (landscape), `profile`,
   `quantities`; reconcile `trajectories.json`/`quantities.json` to reference
   `TableSchema`.

No IR, `ir/schema.json`, `ir/VERSION`, or golden change — this is run-store
output metadata, not the OCaml↔Rust IR contract.

## 7. Consumer impact and non-goals

The camdl fit watcher (separate app) still needs its own change to forward the
index column; this makes that change robust — it reads `role == "iteration"`
instead of hardcoding `draw`/`sweep`. `camdl browse` and external/LLM consumers
gain a uniform way to render any camdl output.

Non-goals: renaming index columns (follow-up); changing any TSV format or the
IR; declaring non-tabular artifacts (`run.json`, `diagnostics.json`, `*.toml`
are already structured).
