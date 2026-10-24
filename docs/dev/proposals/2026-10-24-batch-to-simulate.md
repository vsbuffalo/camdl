# `camdl batch` retires; `simulate` gains the sweep axis

- Date: 2026-10-24
- Status: draft — written for two independent reviews (a modeller's and a
  code-complexity one) before the maintainer decides. Every call this document
  needs is made inline; the reviews may overturn them.
- Relates-to: `2026-09-27-coordinate-kind-design-vs-sample.md` (accepted; its
  decision D-B and its §9.1 columns are used here, and its design-axis flags are
  pulled forward),
  `2026-10-24-remove-pgas-tempering-ode-nuts-and-correlated-pmmh.md` (the
  sibling cut proposal, PR #987), the note
  `2026-10-06-run-grid-workflows-and-composition.md` (the usage survey), gh#930,
  gh#286, gh#738, gh#599, gh#923, gh#598, gh#572, gh#571, gh#948,
  `vsbuffalo/camdl-docs#1`.
- Required reading: `.claude/rules/run-identity.md` (a sweep cell's identity and
  the ensemble hash, §8), `.claude/rules/proposals.md`.
- Evidence: a read-only inventory of main `bff4cec4` taken on 2026-10-24.
  `path:line` references below are from it; nothing was executed.

## 1. Two implementations of one thing

camdl has two ways to run a grid of simulations into the content-addressed
store. `camdl batch run manifest.toml` reads a TOML manifest and expands it;
`camdl simulate` with `--scenario`, `--draws`, `--replicates` or `--seeds`
expands its flags. Both plan the same `scenario × parameter-point × replicate`
grid through `engine::plan_grid` and `run_job`, both write each cell as a `Sim`
leaf through the same `CasSink`, both run cells in parallel through Rayon
(`--parallel` on each; `CAMDL_PARALLEL` for `simulate`), both offer `--dry-run`,
and both are guarded by the same scenario-collision check
(`engine::check_scenario_coordinate_collision`, which has a `ParamSource::Sweep`
arm, `engine.rs:368`). They differ only in the front end, and there the
duplication is the problem:

- two grammars for a parameter grid: batch's private TOML `SweepSpec` (`list`,
  `linspace`, `logspace`, `range`; `batch.rs:143`) and the CLI
  `args::types::SweepSpec` (`PARAM=GRID` with a comma list, `lin(min,max,n)` or
  `log10(min,max,n)`; `types.rs:255-310`) that `fit run`, `fit predict` and
  `profile` already share (`args/mod.rs:750, 1207, 1845`);
- two declarations of the run configuration: batch's `ConfigSection` and
  `SeedsSection` (`batch.rs:320-370`) beside `simulate`'s flags, two of the
  three-to-four `dt`/seed structs gh#923 counts;
- two ways to name a scenario: batch's ad-hoc `[[scenario]]` entries
  (`batch.rs:376-386`) beside the model's own `scenarios {}` block.

Nobody runs `batch`. No external project, Makefile or workflow calls it (the one
project that tried it documented why it stopped and moved to
`simulate --draws`); every surveyed project drives its grids from Python over
`simulate`. The book's scenarios chapter is the only caller, in two executed
cells (`guide/scenarios.qmd:492, 518`). What people do need is the capability
behind batch's `[sweep]`: a deterministic grid of a parameter, run with common
random numbers, into the store. `simulate` cannot do that today; its
`SimulateArgs` carry no sweep flag. The accepted coordinate-kind proposal
already decided where that capability lives: on `simulate`, as the design axis,
with the flags `--sweep` and `--sweep-file`. This proposal pulls that part of it
forward, moves the capability, and then removes the verb. Direction is settled;
this document settles sequencing and completeness.

## 2. Every batch capability, and where it goes

| capability                                                 | `batch` today                              | `simulate` today                                                  | after this proposal                                                       |
| ---------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| scenarios from the model's `scenarios {}` block            | `[[scenario]] name = …`                    | `--scenario`, repeatable                                          | unchanged                                                                 |
| ad-hoc scenario: values plus `enable`/`disable` lists      | several per manifest                       | one per invocation: `--param`, `--enable`, `--disable`            | dropped (§4); the coordinate-kind rewrite owns grid declarations          |
| deterministic parameter grid                               | `[sweep]` list, linspace, logspace, range  | none                                                              | **`--sweep PARAM=GRID`**, repeatable; Cartesian product across flags (§3) |
| a grid read from a file                                    | `[design.*]` outputs only                  | none (a hand grid passed as `--draws FILE` is pooled into a band) | **`--sweep-file FILE`** (§3)                                              |
| generated designs: Latin hypercube, Halton, Saltelli       | `[design.*]` (`sampling.rs`)               | none                                                              | dropped (§4); generate externally, pass as `--sweep-file`                 |
| posterior, prior or uniform draws                          | none                                       | `--draws`, `--n-draws`, `--draws-out`                             | unchanged                                                                 |
| replicates and seeds                                       | `[config.seeds]` explicit list             | `--replicates`, `--seeds`                                         | unchanged; sweep cells seed per D-B (§3)                                  |
| common random numbers across design points                 | yes (explicit seeds)                       | not applicable                                                    | yes, by D-B (§3)                                                          |
| parallel cells                                             | `--parallel`                               | `--parallel`                                                      | unchanged                                                                 |
| synthetic observations into the store                      | `[obs]`                                    | the same sink, `obs_enabled`, `--obs-only-dir` and relatives      | unchanged                                                                 |
| plan without running                                       | `--dry-run`                                | `--dry-run`                                                       | unchanged                                                                 |
| per-cell leaves in the store, labelled by the swept values | yes (`params_path_label`, `batch.rs:1359`) | yes, same sink                                                    | unchanged                                                                 |
| combined table across cells                                | none                                       | `ensemble.tsv` with a `draw` column (`main.rs:2769`)              | `point_id` and `sweep:<param>` columns for sweep jobs (§3)                |
| recompute existing cells                                   | `--force`                                  | none                                                              | dropped (§4)                                                              |
| report on a manifest's cells                               | `batch status`                             | none                                                              | dropped (§4)                                                              |
| a declarative manifest file                                | yes                                        | no                                                                | dropped (§4)                                                              |

## 3. The `simulate` surface after this proposal

**`--sweep PARAM=GRID`**, repeatable. The grammar is the existing
`args::types::SweepSpec`: a comma list (`beta=0.2,0.3,0.6`), `lin(min,max,n)` or
`log10(min,max,n)`. Two `--sweep` flags form the Cartesian product of their
grids, in flag order, the first flag varying slowest. Batch's
`range { min, max,
step }` is not carried: a step grid is `lin` with `n`
computed, and one grammar across `simulate`, `fit predict` and `profile` is
worth more than the fourth spelling. Each distinct point is one design
coordinate; its `point_id` is its 1-based position in the product.

**`--sweep-file FILE`**: a TSV whose header names parameters and whose rows are
design points, `point_id` the 1-based row order. A column that names no model
parameter is refused with the parameter list; a file with zero rows is refused.
Nothing else about the file is specified here; the coordinate-kind proposal's
`label`, `point_id` and provenance columns belong to its rewrite and are not
pre-empted.

**Composition.** `--sweep` and `--sweep-file` compose with `--scenario`,
`--replicates`, `--seeds`, `--param`, `--params` and every output flag.
`--sweep` or `--sweep-file` together with `--draws` is refused in this stage:
the engine's `ParamSource` is a single enum (`Point | Sweep | Draws`,
`sim_job.rs:179`), the design × sample product is the coordinate-kind rewrite's
job, and a refusal now is cheaper than a shape to migrate later. `--sweep` with
`--design-from` is refused for the same reason. A `--param` that names a swept
parameter is refused (the gh#571 shape); a scenario that sets a swept parameter
is refused by the existing guard, whose `Sweep` arm batch already exercised.

**Seeds (decision D-B of the coordinate-kind proposal).** A design coordinate
never enters the process seed.
`engine::process_seed_for(explicit_seeds,
base_seed, point_idx, rep, total_runs)`
(`engine.rs:52`) today mixes the parameter-point index into the seed; for a
sweep source the index it receives is 0, so every design point at a given
replicate shares one random stream. With `--seeds` the explicit list is used by
position, as batch did. Two consequences a modeller relies on: paired
differences across design points are low-variance, and inserting a point into a
sweep leaves the other points' trajectories unchanged. No existing job changes
seed, because no `simulate` job has a sweep today and batch's sweeps used
explicit seeds.

**Store.** Each cell is a `Sim` leaf exactly as `simulate` and `batch` write one
today, through `CasSink`, with the leaf's parameter label naming the swept
values (`beta=0.3`, or `beta=0.3_gamma=0.1`, `params_path_label`).

**Combined output.** `simulate` writes `ensemble.tsv` for a multi-cell run with
a `draw` column from `draw_idx` (`EnsembleCell.draw_idx`,
`sim_ensemble_cas.rs:44`; "0 when not a `--draws` run"). A design point is not a
draw, and labelling it as one is the confusion the coordinate-kind proposal
exists to remove. For a sweep job the combined table carries, per §9.1 rows 2–3
of that proposal: `point_id` (1-based) and one `sweep:<param>` column per swept
parameter, sorted by name, written with Rust's shortest round-trip `f64`
formatting so a value joins exactly against the grid that produced it, and no
`draw` column. Quantity files group per design point and scenario the way
`fit predict --sweep` groups them today through `DesignCoord`
(`quantity_output.rs`); their shape is otherwise today's. The per-cell and
summary tables, `band`, and fixed headers are the rewrite's and are not touched.

**Ensemble identity.** The grid hash is one entry per cell of (scenario, seed,
draw index, run ID), sorted (`sim_ensemble_cas.rs:100-131`). Under D-B every
design point shares the seed and has draw index 0, so `--sweep beta=0.15,0.3`
and `--sweep beta=0.3,0.15` would hash the same while their combined files
differ. This is blocker 1 of the coordinate-kind implementation review and its
design call 3, taken here: for sweep jobs each cell's entry gains its design
index and `point_id` as an optional field; an absent field hashes as before, so
every existing draws ensemble keeps its ID.

## 4. What is dropped, and why

- **The manifest file.** The store records each cell's resolved configuration,
  which is what makes a run reproducible; a manifest was a second copy of it. A
  modeller keeps the `simulate` command in a Makefile or a script, which is what
  every surveyed project does already.
- **Ad-hoc scenarios.** camdl's way to declare a scenario is the model's
  `scenarios {}` block, which the compiler checks and the store keys. Batch let
  a manifest invent unchecked ones. One ad-hoc arm is still `--param` plus
  `--enable`/`--disable`; several named arms are a model edit. Whether a grid
  declaration file returns is the coordinate-kind rewrite's call.
- **The design generators** (`sampling.rs`, 395 lines, called only from
  `batch.rs:23`). No user ever ran one. A Latin-hypercube or Saltelli design
  made in Python or R is a TSV, and `--sweep-file` reads it; Saltelli
  sensitivity indices never had a consumer inside camdl.
- **`batch status`.** One user, gh#930. Listing what a grid produced is a store
  question, not a batch one; if it returns, it returns as a store verb.
- **`--force`.** Recomputing an existing leaf is already what the store's
  divergent-recompute path does when inputs differ; a flag to force it on
  identical inputs re-keys nothing and reproduces the same bytes.

## 5. Code

**Moves first, in its own commit** (about 1,100 lines of `batch.rs` that
`simulate` uses, to a new `rust/crates/cli/src/cas_sink.rs`): `ResolvedEntry`
and `from_preset` (`:404-448`; `main.rs:2603-2636`), `RunEntry` (`:502-524`;
`main.rs:2798`), `CasSink` with `cells_progress` and its `RunSink` impl
(`:1004-1066, 1068, 1072-1265` minus `predict_cells`, `:1417-1715`;
`main.rs:2242, 2564, 2647-2686`), `params_path_label` with its tests
(`:1359-1415`), `params_level_hash` `:1717`, `read_traj_digest` `:1724`,
`ensure_provenance_label` `:1736`, `obs_subtree_hash` `:2048` (also
`emit_every.rs:18`), `write_obs_into_cas` `:2071-2187`.

**Added:** the two flags on `SimulateArgs` and their conflicts; a constructor
from the CLI `SweepSpec` list and from a TSV to
`ParamSource::Sweep { points,
replicates }` (the variant exists,
`sim_job.rs:179`, with arms at `sim_job.rs:205, 218` and `engine.rs:368`; batch
is its only constructor today, `batch.rs:639, 1841`); the seed-index rule in the
sweep arm of `plan_grid`; the optional design field on `EnsembleCell` and in the
grid hash; `point_id` and `sweep:<param>` columns in the combined writer; the
`--param`-versus-sweep refusal.

**Deleted** (after the flags land): `main.rs:149-151, 345-361, 571-572`
(`Batch(BatchCmd)`, `BatchCmd`, dispatch) and `:27` (`mod batch;`);
`args/mod.rs:590-646` (`BatchArgs`, `BatchStatusArgs`) and the stale comments at
`:3, 183-186`; in `batch.rs`: `:37-65`, `:67-370`, `:376-386`, `:450-500`,
`:527-560`, `:568-648`, `:650-995`, `:1274-1357` (`predict_cells`,
`CellPrediction`), `:1775-2046`, `:2188-2489`, and the batch-only unit tests
under `:2490`; `sampling.rs` entire; `tests/test_ocaml_to_rust.sh:84-143`
(`run_batch_test`, called from `Makefile:209`) and the eight manifests
`tests/fixtures/exp_*.toml`. `ParamSource::Sweep` and `Seeds::Explicit` stay.

**Unaffected:** `engine.rs`, `sim_job.rs` (`SimulateJob`, `Seeds`, `ObsOutput`,
`ScenarioRef`), `sim_ensemble_cas.rs` (used only by `simulate`),
`params_resolver`, `config_v2::SeedsSpec` (belongs to `[synthetic]`),
`fit/predict.rs` and `fit/contrasts.rs` (no reference to batch).

**Size:** about 1,850 lines of `batch.rs` and 395 of `sampling.rs` deleted, 80
of args and dispatch, about 1,030 lines of whole test files, 55 shell lines and
176 fixture lines; about 1,100 lines move; the additions are a few hundred.

## 6. Tests

New, red before the flags and green after:

1. `simulate --sweep beta=0.2,0.3 --replicates 2` on a fixture writes four
   leaves labelled `beta=0.2` and `beta=0.3`, and the two points share a process
   seed per replicate: sweeping a parameter the dynamics do not read (an
   observation-only parameter with observation output disabled) gives
   byte-identical trajectories across the two points. That is the D-B pin.
2. The combined table of a sweep job has `point_id` and `sweep:beta` and no
   `draw`; `sweep:beta` round-trips `0.1234567` exactly.
3. `--sweep beta=0.2,0.3` and `--sweep beta=0.3,0.2` produce different ensemble
   IDs; a draws-only ensemble keeps its ID byte-for-byte across the change (the
   identity golden for the grid hash).
4. `--sweep` with `--draws`, with `--design-from`, with a `--param` naming the
   swept parameter, and with a scenario setting it, are each refused with the
   message naming the conflict; the scenario case ports
   `scenario_coordinate_collision_gh572.rs:162-234` from batch to `simulate`.
5. `--sweep-file` with an unknown column, and with zero rows, is refused.
6. `camdl batch` exits with clap's unknown-subcommand error; no source line
   matches `cmd_batch_run`, `ExperimentToml`, `BatchArgs` or `sampling::`.

Ported, not deleted, because they guard the shared sink: the batch tests in
`cas_integration.rs:349, 434, 492, 878, 973, 1018`,
`design_preserving_simulate.rs:707`, `scenario_identity_gh573.rs:205, 213,
224`,
`simulate_identity_parity.rs:460`, and the obs-into-store coverage of
`acceptance_batch_obs_ensemble.rs`, each rewritten against `simulate`. The
model-graph archive test (`model_graph_archive.rs:68`) re-pins through
`simulate`. The three `CasSink` identity unit tests move with the code. Deleted
outright: `acceptance_batch_named_scenario.rs`, `acceptance_design_cas.rs`,
`gh589_batch_obs_preflight.rs`.

One check that is run once before deletion and not kept: the book's sweep (five
β values, two scenarios, fifty seeds) run through `simulate --sweep` produces
the same set of leaf identities batch produced from
`guide/boarding-school/sweep.toml`.

## 7. Documentation

Remove or rewrite: `docs/commands.md:21, 47, 174-205, 272`; `docs/intro.md:413`;
`docs/inference.md:1511`; `docs/camdl-inference-spec.md:15, 936, 1134-1136`;
`docs/camdl-language-spec.md:4180-4213, 4796, 5060-5069` (edited by hand, never
formatted); `docs/camdl-run-spec.md` §5 (`:1956, 1998-2001`), §5.5
(`:2162-2361`), §5.6 (`:2362-2393`), §3.3 (`:1079-1105`), `:1204-1208`,
`:1275-1276`, §9.7 (`:4663-4674`), Appendix A (`:6224-6239`), "Batch TOML
stability" (`:6530-6544`) and the inline mentions at
`:7, 56, 69, 72-74, 85,
128-136, 148, 192, 766, 820-825, 869, 885, 894, 903-907, 949-963, 1017, 1057,
1068`;
`README.md:232, 360, 401`; `scripts/review-zip.sh:66`, `scripts/README.md:68`.
Add `--sweep` and `--sweep-file` to `simulate`'s reference in `docs/commands.md`
and to the run spec's multi-cell section, with the D-B seed rule stated. The
book's chapter rewrite is `vsbuffalo/camdl-docs#1` and lands in step with the
deletion.

## 8. Identity

`--sweep` and `--sweep-file` are new, so no existing run re-keys. The ensemble
grid hash gains an optional per-cell design field that is absent for every
existing job, so every existing ensemble ID is unchanged; test 3 pins it. The
tag `archive/batch` is placed on the last commit on main that carries the verb,
before the deletion commit, with the proposal named in the tag message.

## 9. Order of work

1. **PR A, pure move.** §5's shared sink to `cas_sink.rs`; `simulate` and
   `batch` outputs byte-identical on the fixtures; full `make test`.
2. **PR B, the flags.** `--sweep`, `--sweep-file`, the seed rule, the combined
   columns, the hash field, tests 1–5; full `make test`.
3. **PR C, the deletion.** Tag `archive/batch`; delete §5's list; port and
   delete tests per §6; docs per §7; close gh#930 and gh#286; relabel gh#738 and
   gh#599 to `simulate`/`CasSink`; note the scope reduction on gh#923 and
   gh#598; full `make test`. The book chapter (`camdl-docs#1`) lands with it.

Each PR formats every touched `.md` with `mdfmt` except
`docs/camdl-language-spec.md`, and carries no attribution trailers.

## 10. Relation to the coordinate-kind arc

This takes the design half of the coordinate-kind proposal's flag stage
(`--sweep`, `--sweep-file`, D-B seeds, `point_id` and `sweep:<param>` columns,
the hash field) and leaves the rest: the `--draws <path>` refusal, the
provenance sidecar, the design × sample product, `band`, the per-cell and
summary tables, fixed headers, and `fit predict`'s move onto the shared grid.
Both forks of that arc (implement as accepted, or rewrite from the 2026-10-06
note) keep D-B and the two columns, so nothing here is undone by either. What
the rewrite inherits is simpler by the whole of `batch`.

## 11. Risks

- A modeller who relied on batch's ad-hoc `[[scenario]]` entries to compare arms
  without editing the model has one arm per invocation until the rewrite decides
  on grid declarations. No such modeller is known.
- The combined-table columns for sweep jobs are specified from the accepted
  §9.1; if the rewrite changes the table shape, these two columns survive but
  the file around them changes once more. That is already true of every quantity
  file.
- `--sweep-file` without the `--draws <path>` refusal means two file flags with
  opposite semantics coexist until the rewrite lands the refusal. The help text
  for each says which axis it builds.
