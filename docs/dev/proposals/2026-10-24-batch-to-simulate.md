# `camdl batch` retires; `simulate` gains the sweep axis

- Date: 2026-10-24
- Status: accepted — two independent reviews on 2026-10-24 (a modeller's and a
  code-complexity one) each returned "yes, with changes"; every change they
  required is folded in, and the maintainer decided the nine remaining calls on
  2026-10-24 (§12). Ready to implement in the order of §9.
- Relates-to: `2026-09-27-coordinate-kind-design-vs-sample.md` (accepted; its
  decision D-B, its §9.1 columns, its name-sorted Cartesian product and its
  `EnsembleGridLevel` form are used here, and its design-axis flags are pulled
  forward; §10 lists the amendments it needs),
  `2026-10-24-remove-pgas-tempering-ode-nuts-and-correlated-pmmh.md` (the
  sibling cut proposal, PR #987), the note
  `2026-10-06-run-grid-workflows-and-composition.md` (the usage survey), gh#930,
  gh#286, gh#738, gh#599, gh#923, gh#598, gh#572, gh#571, gh#562, gh#948,
  gh#980, `vsbuffalo/camdl-docs#1`.
- Required reading: `.claude/rules/run-identity.md` (a sweep cell's identity and
  the ensemble hash, §8), `.claude/rules/proposals.md`.
- Evidence: a read-only inventory of main `bff4cec4` taken on 2026-10-24, the
  two reviews' own reads, and one executed demonstration (§3, "Combined
  outputs") on a 140-day SIRS model with the release binary at `bff4cec4`.

## 1. Two implementations of one thing

camdl has two ways to run a grid of simulations into the content-addressed
store. `camdl batch run manifest.toml` reads a TOML manifest and expands it;
`camdl simulate` with `--scenario`, `--draws`, `--replicates` or `--seeds`
expands its flags. Both plan the same `scenario × parameter-point × replicate`
grid through `engine::plan_grid` and `run_job`, both write each cell as a `Sim`
leaf through the same `CasSink`, both run cells in parallel through Rayon
(`--parallel` on each; `CAMDL_PARALLEL` for `simulate`), both offer `--dry-run`
and `--force`, and both are guarded by the same scenario-collision check
(`engine::check_scenario_coordinate_collision`, which has a `ParamSource::Sweep`
arm, `engine.rs:368`). They differ in the front end, and there the duplication
is the problem:

- three expansions of a parameter grid: batch's private TOML `SweepSpec`
  (`list`, `linspace`, `logspace`, `range`; `batch.rs:143-292`), the CLI
  `args::types::SweepSpec` (`PARAM=GRID` with a comma list, `lin(min,max,n)` or
  `log10(min,max,n)`; `types.rs:255-310`) that `fit run`, `fit predict` and
  `profile` share (`args/mod.rs:750, 1207, 1845`), and `fit predict`'s own
  Cartesian expansion of it (`expand_predict_sweep`,
  `fit/predict.rs:1359-1380`);
- two declarations of the run configuration: batch's `ConfigSection` and
  `SeedsSection` (`batch.rs:320-370`) beside `simulate`'s flags, two of the
  three-to-four `dt`/seed structs gh#923 counts;
- two ways to name a scenario: batch's ad-hoc `[[scenario]]` entries
  (`batch.rs:376-386`) beside the model's compiler-checked `scenarios {}` block;
- two Rayon pool helpers (`batch.rs:37-65`, `pfilter.rs:579`).

Nobody runs `batch`. No external project, Makefile or workflow calls it (the one
project that tried it documented why it stopped and moved to
`simulate --draws`); every surveyed project drives its grids from Python over
`simulate`. The book's scenarios chapter is the only caller, in two executed
cells (`guide/scenarios.qmd:492, 518`). What people do need is the capability
behind batch's `[sweep]`: a deterministic grid of a parameter, run with common
random numbers, into the store, resumable. `simulate` cannot do that today: its
`SimulateArgs` carry no sweep flag, and its multi-cell path recomputes every
cell on a re-run where batch skipped store hits (§3). The accepted
coordinate-kind proposal already decided where the capability lives: on
`simulate`, as the design axis, with the flags `--sweep` and `--sweep-file`.
This proposal pulls that part forward, moves the capability, and then removes
the verb. Direction is settled; this document settles sequencing and
completeness.

## 2. Every batch capability, and where it goes

| capability                                            | `batch` today                                                            | `simulate` today                                                                                 | after this proposal                                                        |
| ----------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |
| scenarios from the model's `scenarios {}` block       | `[[scenario]] name = …`                                                  | `--scenario`, repeatable, comma-split                                                            | unchanged                                                                  |
| ad-hoc scenario: values plus `enable`/`disable` lists | several per manifest, each named                                         | one per invocation: `--param`, `--enable`, `--disable`; the arm is labelled `baseline`           | one per invocation, labelled from its flags (§3, "Arm labels")             |
| deterministic parameter grid                          | `[sweep]` list, linspace, logspace, range                                | none                                                                                             | **`--sweep PARAM=GRID`**, repeatable; name-sorted Cartesian product (§3)   |
| a grid read from a file                               | `[design.*]` outputs only                                                | none (a hand grid passed as `--draws FILE` is pooled into a band, gh#562)                        | **`--sweep-file FILE`** (§3)                                               |
| generated designs: Latin hypercube, Halton, Saltelli  | `[design.*]` (`sampling.rs`)                                             | none                                                                                             | dropped (§4); generate externally, pass as `--sweep-file`                  |
| posterior, prior or uniform draws                     | none                                                                     | `--draws`, `--n-draws`, `--draws-out`                                                            | unchanged                                                                  |
| replicates and seeds                                  | `[config.seeds]` explicit list                                           | `--replicates`, `--seeds`                                                                        | unchanged; sweep cells seed per D-B (§3)                                   |
| common random numbers across design points            | yes (explicit seeds)                                                     | not applicable                                                                                   | yes, by D-B (§3)                                                           |
| skip cells already in the store on a re-run           | yes (`CasSink::should_run`, `batch.rs:1418`)                             | no: runs every cell (`batch.rs:1618-1622`)                                                       | **yes** for multi-cell `simulate` (§3)                                     |
| recompute existing cells                              | `--force`                                                                | `--force` (`args/mod.rs:547`)                                                                    | unchanged                                                                  |
| parallel cells                                        | `--parallel`                                                             | `--parallel`                                                                                     | unchanged                                                                  |
| synthetic observations into the store                 | `[obs] enabled = true`, no path needed                                   | the same sink, enabled by any of the four path flags (`main.rs:986, 1874`)                       | unchanged; the directory is the product (§4)                               |
| plan without running, with store hits and misses      | `--dry-run` prints hit/miss counts and leaf paths (`batch.rs:1310-1320`) | `--dry-run` prints model, backend, seeds, scenarios only (`main.rs:4532-4590`)                   | **hit/miss counts and leaf paths** in multi-cell `simulate --dry-run` (§3) |
| report on a grid's cells                              | `batch status`                                                           | none                                                                                             | dropped (§4); `--dry-run` carries the hit/miss view                        |
| per-cell leaves labelled by the swept values          | yes (`params_path_label`, `batch.rs:1359`)                               | yes, same sink                                                                                   | unchanged                                                                  |
| combined table across cells                           | none                                                                     | `ensemble.tsv` with a global `replicate` ordinal and a `draw` column (`main.rs:2769, 2913-2929`) | `point_id`, `sweep:<param>` and `seed` columns for sweep jobs (§3)         |
| the grid declaration kept with the results            | the manifest                                                             | the per-cell `run.json`; the ensemble record holds only counts (`main.rs:2750-2756`)             | the ensemble record carries the resolved design (§3)                       |
| copy a GeoJSON beside the outputs                     | `[config] geo` (`batch.rs:325, 924-930`)                                 | none                                                                                             | dropped (§4)                                                               |

## 3. The `simulate` surface after this proposal

**`--sweep PARAM=GRID`**, repeatable. The grammar is the existing
`args::types::SweepSpec`: a comma list (`beta=0.2,0.3,0.6`), `lin(min,max,n)` or
`log10(min,max,n)`. `lin` refuses `min >= max`, so a single-value "sweep" is the
list form. An indexed parameter is named as everywhere else on the CLI,
underscore-joined (`beta_north`), and its column is `sweep:beta_north`. There is
no date grammar on either side today (`--param` parses `f64`), so a sweep over
an intervention date is written in model-time offsets; that is not a regression
and is a follow-up. Several `--sweep` flags form the Cartesian product **sorted
by parameter name**, the first name varying slowest, as the accepted proposal
specifies (`:1497`) and as `fit predict` already does; a repeated name is
refused. The expansion is `expand_predict_sweep` lifted out of
`fit/predict.rs:1359-1380` into shared code, which `fit predict` then calls too,
so the two verbs produce the same `point_id` for the same flags and the third
expansion in §1 goes. Batch's `range { min, max, step }` is not carried: a step
grid is `lin` with `n` computed.

**`--sweep-file FILE`**: a TSV whose header names parameters and whose rows are
design points. A column that names no model parameter is refused with the
parameter list; a file with zero rows is refused; a duplicate row is refused
(decision D-E of the accepted proposal), which also answers what happens when a
generated design repeats a point. Nothing else about the file is specified here;
the coordinate-kind proposal's `label`, `point_id` and provenance columns belong
to its rewrite and are not pre-empted.

**`point_id`** is the 0-based position of the design point in the name-sorted
product, or its 0-based row in the sweep file, as the accepted proposal's D-C
says. The maintainer decided on 2026-10-24 that camdl standardises on one index
base, 0, everywhere a user sees an index: the `draw` and `replicate` columns
(1-based today, `draw_idx + 1` and `run_idx + 1`, `main.rs:2913-2929, 3081`),
the `chain_N/` directories and every 1-based chain label (gh#980's class), and
`--exclude-chains`. Those conversions are not this proposal's work; they are one
issue, filed with it, that supersedes gh#980's two-type design (`ChainIndex`
plus `ChainLabel`) with a single index type, since with one base the second type
has nothing to guard. `point_id` is born 0-based and never converted.

**Composition.** `--sweep` and `--sweep-file` compose with `--scenario`,
`--replicates`, `--seeds`, `--param`, `--params` and every output flag.
`--sweep` or `--sweep-file` together with `--draws` is refused in this stage:
the engine's `ParamSource` is a single enum (`Point | Sweep | Draws`,
`sim_job.rs:179`), the design × sample product is the coordinate-kind rewrite's
job, and a refusal now is cheaper than a shape to migrate later. The one
workflow this defers, a posterior ensemble under each of several swept values,
is available today as `fit predict --sweep` for a fitted model, and as one
`simulate --param v=X --draws posterior` invocation per value, which already
shares seeds across values because neither scenario nor point enters the mix.
`--sweep` with `--design-from` is refused for the same reason. A `--param` that
names a swept parameter is refused (the gh#571 shape); a scenario that sets a
swept parameter is refused by the existing guard, whose `Sweep` arm batch
already exercised. Every conflict list that names `draws` (`--stdout`,
`--event-log`) gains `sweep` and `sweep_file`, as the accepted proposal lists
(`:1498-1499`).

**Seeds (decision D-B of the accepted proposal).** A design coordinate never
enters the process seed. Today `plan_grid` passes the real parameter-point index
to
`engine::process_seed_for(explicit_seeds, base_seed, point_idx, rep,
total_runs)`
(`engine.rs:52-66, 176-178`) for every source; the rule here is new: for a sweep
source the index passed to `process_seed_for` is 0, so every design point at a
given replicate shares one random stream (`mix_cell_seed(base, 0, rep)`,
`util.rs:35-37`), while `CellSpec.point_idx` keeps the design index for
labelling. With `--seeds` the explicit list is used by position, as batch did.
Two consequences a modeller relies on: paired differences across design points
are low-variance, and inserting a point into a sweep leaves the other points'
trajectories unchanged. When a modeller wants independent noise per point
instead, for variance-based sensitivity indices that assume independence, the
route is one invocation per point with a distinct `--seed`; the flag's help text
says so. No existing job changes seed: no `simulate` job has a sweep today, and
batch's sweeps used explicit seeds, where the index is unused.

**Re-using cells the store already holds.** The store names every cell by a hash
of its inputs. Batch looked that hash up before simulating and, when the store
already held the cell, skipped the simulation and read the stored trajectory
back; `simulate`'s sink always answers "run" because its `-o` mirror wanted
every trajectory in memory (`batch.rs:1618-1622`). So today an interrupted
500-cell grid restarts from nothing and adding one β value to a five-value sweep
re-simulates all six. This proposal gives multi-cell `simulate` batch's
behaviour, which is also how every other store-backed verb behaves: a committed
cell is skipped and its `traj.tsv` is read back for the combined file and the
mirror; `--force` recomputes as it does today. The bytes are identical by
construction; the only risk is a stale cell masking a changed input, which the
count-in-the-key rule already forbids. Test: a second identical run simulates
zero cells.

**Dry run.** Multi-cell `simulate --dry-run` prints, per cell, whether the store
already holds it and the leaf path, with hit and miss counts, as
`batch run --dry-run` did. The sink already resolves each cell's identity, so
this is the view, not new computation. It is what makes a long sweep operable
together with re-use, and it replaces `batch status`.

**Arm labels.** Without `--scenario`, the run's one arm is
`ScenarioRef::Inline { name: "baseline", enable, disable, .. }`
(`main.rs:1711`), so an arm built with `--enable vaccination` or
`--param beta=0.3` is labelled `baseline` in the `scenario` column and in the
store listing, and two different ad-hoc arms are indistinguishable. Batch's
manifest arms each had a name. The label is presentation only: the arm's
identity already includes its `enable`/`disable` sets (gh#573) and its parameter
overrides, so no leaf re-keys. Decision: the label is derived from the flags,
deterministically, at that one construction site: `baseline` when nothing is
overridden; otherwise `baseline` followed by `+name` for each enabled
intervention, `-name` for each disabled one, and `;param=value` for each
`--param`, in flag order, for example `baseline+vaccination` or
`baseline;beta=0.3`. A derived label cannot collide with a declared scenario,
because declared names cannot contain `+`, `-` followed by a letter at that
position, or `;`. Mandating a name was rejected: it would refuse every existing
`--enable` invocation, including the book's. A flag that overrides the derived
label is a follow-up if anyone asks. This is distinct from `--label`, which is a
display label for the whole run in `camdl list` and `camdl show`
(`args/mod.rs:243-248`) and never reaches the `scenario` column.

**Store.** Each cell is a `Sim` leaf exactly as `simulate` and `batch` write one
today, through `CasSink`, with the leaf's parameter label naming the swept
values (`beta=0.3`, or `beta=0.3_gamma=0.1`, `params_path_label`).

**Combined outputs.** Three kinds of file label cells today and all three would
call a design point a draw. The demonstration on 2026-10-24 (SIRS model, release
binary at `bff4cec4`) showed the current columns: a replicate grid writes
`replicate  time  cases`; a run with `--draws uniform -n 2 --replicates 2`
writes `replicate  draw  time  cases` with `replicate` counting 1 to 4 across
both draws, so the two replicates of draw 2 are rows 3 and 4 and nothing in the
file says which replicate of draw 1 they pair with.

- `ensemble.tsv` has a global `replicate` ordinal (`run_idx + 1`) and a `draw`
  column (`draw_idx + 1`) (`main.rs:2913-2929`). For a sweep job it carries, per
  §9.1 rows 2–3 of the accepted proposal, `point_id` and one `sweep:<param>`
  column per swept parameter, name-sorted, written with Rust's shortest
  round-trip `f64` formatting so a value joins exactly against the grid that
  produced it; no `draw` column; and a `seed` column holding the cell's process
  seed. The `seed` column is the pairing key: under D-B two design points at the
  same replicate carry the same seed, so a modeller joins on
  `(scenario, seed, time)` to form the paired difference D-B promises. The
  global `replicate` ordinal cannot serve, because it counts across points; the
  within-axis `replicate` of the 2026-10-06 note's storage rules is the
  rewrite's, and the `seed` column is additive to it.
- The observation tables written by `--obs`, `--obs-dir`, `--obs-only` and
  `--obs-only-dir` carry `replicate: run_idx + 1` and `draw: draw_idx + 1`
  (`main.rs:3081`) and a `draw` header when `n_points > 1`
  (`main.rs:2918,
  2929, 2947`). For a sweep job they follow the same rule as
  `ensemble.tsv`: `point_id`, `sweep:<param>` and `seed`, no `draw`.
- Quantity files: `SimQuantities.by_scenario` is keyed by scenario only
  (`main.rs:2295, 2486`), `DesignCoords.sweep` is hard-wired empty
  (`main.rs:2169`), and `Mode` is `Point` only for
  `ParamSource::Point {
  replicates: 1 }` (`main.rs:1954`), so a sweep job
  today would band every design point into one group, the gh#562 class. The
  accumulator is keyed by (design point, scenario), `DesignCoords.sweep` is
  filled from the cell, and the rule is: one replicate per point gives a
  point-mode row per group; more gives a band over replicates per group.
  `fit predict --sweep` avoids the problem by looping outside the engine with a
  fresh sink per point (`fit/predict.rs:2188`), so its grouping does not
  transfer by itself.

**The grid declaration in the record.** Every leaf in the store has a `run.json`
(`runid/src/record.rs`); the ensemble leaf's `run.json` carries display inputs
that today hold only `n_cells`, `scenarios`, `n_seeds`, `n_draws`
(`main.rs:2744-2756`). After the manifest is gone the only record of "β =
lin(0.2,0.6,5)" would be shell history. For a sweep job the same `run.json`
carries the resolved design: each swept parameter with its values, or the
`--sweep-file` digest, and `n_points` in place of a misleading `n_draws` of 1.
No new file.

**Ensemble identity.** The grid level is
`EnsembleGridLevel { n_cells, cells:
&[(&str, u64, usize, String)], … }`
(`sim_ensemble_cas.rs:100-131`), one tuple per cell of (scenario, seed, draw
index, run ID), sorted, and `canonical_config_hash` serializes the whole struct
and strips nothing (`fit/cas.rs:211-217`). Under D-B every design point shares
the seed and has draw index 0, so `--sweep beta=0.15,0.3` and
`--sweep beta=0.3,0.15` sort to the same cell set and the same ID while their
combined files differ (blocker 1 of the coordinate-kind implementation review).
The fix must not add an element to the tuple: a `None` there serializes as
`null` and re-keys every stored draws ensemble. It is the accepted proposal's
form (`:1195-1200`): a top-level optional field on `EnsembleGridLevel`,
serialized only when present (`skip_serializing_if = "Option::is_none"`),
carrying the design identity (per cell, its design index and `point_id`), set
only for jobs with a design axis. A unit test beside
`levels_are_byte_identical_to_the_literals_they_replaced`
(`sim_ensemble_cas.rs:203`) pins that a draws ensemble's level bytes are
unchanged. `RunEntry` (`batch.rs:502-524`), which populates `EnsembleCell`
(`main.rs:2694-2705`) with `draw_idx: spec.point_idx` (`batch.rs:1500, 1709`),
gains the design field; for a sweep cell its `draw_idx` is 0.

## 4. What is dropped, and why

- **The manifest file.** The store records each cell's resolved configuration in
  its `run.json`, and after this proposal the ensemble record carries the
  resolved design (§3), so the grid declaration is kept with the results rather
  than beside them. A modeller keeps the `simulate` command in a Makefile or a
  script, which is what every surveyed project does already.
- **Named ad-hoc scenarios in a file.** camdl's way to declare a scenario is the
  model's `scenarios {}` block, which the compiler checks and the store keys;
  batch let a manifest invent unchecked ones. The one cost the drop had, that an
  ad-hoc arm was labelled `baseline` so two could not be told apart, is removed
  by the derived arm label (§3). One cost remains and is stated so it is not
  rediscovered: adding an arm to `scenarios {}` changes the whole-IR model
  digest (`inputs.rs:317-334`), so every existing leaf of that model becomes a
  store miss. A manifest arm avoided that. Whether a grid declaration file
  returns is the coordinate-kind rewrite's call.
- **The design generators** (`sampling.rs`, 395 lines, called only from
  `batch.rs:23`; `sobol`, `lhs`, `random` at `:240-252`). No user ever ran one,
  and camdl has no sensitivity analyser, so the analysis already lives in Python
  or R where the samplers also live. A generated design is a TSV, and
  `--sweep-file` reads it; D-E's duplicate refusal applies to it.
- **`batch status`.** One user, gh#930. Its hit/miss view moves into multi-cell
  `simulate --dry-run` (§3); a listing verb, if ever wanted, is a store verb.
- **`[config] geo`.** Copies a GeoJSON to `<output_dir>/geo/boundaries.geojson`
  (`batch.rs:325, 924-930`). It is a `cp`; a modeller's Makefile does it.
- **Synthetic observations without a destination.** Batch's
  `[obs] enabled =
  true` wrote observations into each leaf with nowhere else
  to put them; `simulate` enables the same machinery through one of four flags
  that each name a destination: `--obs FILE` (every stream in one TSV),
  `--obs-dir DIR` (one TSV per observation stream, which is why it is a
  directory), and the `--obs-only` and `--obs-only-dir` forms that also suppress
  the trajectory. The demonstration showed what the directory holds:
  `synth/cases.tsv` with `time  cases` for one run, and a stacked `cases.tsv`
  with a `replicate` (and `draw`) column for a grid, which is the file a
  modeller then hands to `pfilter --data cases=synth/cases.tsv` or to a fit. The
  destination is the product, not a throwaway, and the observations also land in
  each leaf, so a path-less form adds nothing. Decision: no new flag; the help
  text says the directory is the dataset.

## 5. Code

**Moves first, in its own commit** (about 1,100 lines of `batch.rs` that
`simulate` uses, to a new `rust/crates/cli/src/cas_sink.rs`): `ResolvedEntry`
and `from_preset` (`:404-448`; `main.rs:2603-2636`), `RunEntry` (`:502-524`;
`main.rs:2798`), `CasSink` with `cells_progress` and its `RunSink` impl
(`:1004-1066, 1068, 1072-1225`, the remainder of `:1274-1265` minus
`predict_cells`, `:1417-1715`; `main.rs:2242, 2564, 2647-2686`),
`params_path_label` with its tests (`:1359-1415`), `params_level_hash` `:1717`,
`read_traj_digest` `:1724`, `ensure_provenance_label` `:1736`,
`obs_subtree_hash` `:2048` (also `emit_every.rs:18`), `write_obs_into_cas`
`:2071-2187`. `CasSink::probe` (`:1226-1273`) does **not** move: its only
callers are batch dry-run, status and design (`:1962, 2269, 2344`); the hit/miss
view of §3 is built on the sink's identity resolution, not on `probe`.
`simulate` builds the same sink with `obs_enabled` and `emit_every`
(`main.rs:2647-2672`), so the obs-into-store path stays live and its coverage is
ported (§6). After the move, the doc references to `batch::` at `engine.rs:4`,
`sim_job.rs:3, 46`, `emit_every.rs:18`, `params_resolver.rs:567, 616`,
`fit/pgas.rs:282` and the intra-doc link at `fit/predict.rs:1357` are repointed.

**Added:** the two flags on `SimulateArgs` and their conflicts, including the
`draws`-only conflict lists; the lifted `expand_predict_sweep` and a TSV reader,
both producing `ParamSource::Sweep { points, replicates }` (the variant exists,
`sim_job.rs:179`, arms at `sim_job.rs:205, 218`, `engine.rs:368`; batch is its
only constructor today, `batch.rs:639, 1841`); the seed-index rule in the sweep
arm of `plan_grid`; the design field on `RunEntry` and the optional field on
`EnsembleGridLevel`; `point_id`, `sweep:<param>` and `seed` in the ensemble and
observation writers; the (design point, scenario) key on `SimQuantities` and the
filled `DesignCoords.sweep`; store re-use and the hit/miss dry-run view for
multi-cell `simulate`; the resolved design in the ensemble record; the derived
arm label; the `--param`-versus-sweep refusal.

**Deleted** (after the flags land): `main.rs:149-151, 345-361, 571-572`
(`Batch(BatchCmd)`, `BatchCmd`, dispatch) and `:27` (`mod batch;`);
`args/mod.rs:590-646` (`BatchArgs`, `BatchStatusArgs`), the stale comments at
`:3, 183-186`, and the serde derives `OutputView` carried for batch's `[output]`
(`args/mod.rs:8-12, 194`); in `batch.rs`: `:37-65` (the duplicate Rayon pool
helpers), `:67-370`, `:376-386`, `:450-500`, `:527-560`, `:568-648`, `:650-995`,
`:1226-1273` (`probe`), `:1274-1357` (`predict_cells`, `CellPrediction`),
`:1775-2046`, `:2188-2489`, and the batch-only unit tests under `:2490`;
`sampling.rs` entire; `tests/test_ocaml_to_rust.sh:84-143` (`run_batch_test`,
called from `Makefile:209`) and the eight manifests `tests/fixtures/exp_*.toml`.
`ParamSource::Sweep` and `Seeds::Explicit` stay.

**Unaffected:** `engine.rs`, `sim_job.rs` (`SimulateJob`, `Seeds`, `ObsOutput`,
`ScenarioRef`), `sim_ensemble_cas.rs` apart from the new field,
`params_resolver`, `config_v2::SeedsSpec` (belongs to `[synthetic]`),
`fit/predict.rs` apart from calling the lifted expansion, `fit/contrasts.rs`,
`mre.rs`, `profile.rs`, `compare.rs` (no code dependency on batch).

**Size:** deleted about 3,180 lines (`batch.rs` ≈1,850, `sampling.rs` 395, args
and dispatch ≈80, shell 55, fixtures 176, whole test files 626); moved about
1,100; added about 300–400 once the quantities key, the observation columns,
store re-use and the arm label are counted. Duplicated concepts removed: the
TOML sweep grammar, the third Cartesian expansion,
`ConfigSection`/`SeedsSection`, the ad-hoc `ScenarioEntry`, the second Rayon
pool. Net additions that are real but small, and that the accepted proposal
would add anyway: the optional field on the grid level, a second column regime
in the combined tables, a two-key quantities accumulator.

## 6. Tests

New, red before the flags and green after:

1. `simulate --sweep beta=0.2,0.3 --replicates 2` on a fixture writes four
   leaves labelled `beta=0.2` and `beta=0.3`, and the two points share a process
   seed per replicate: sweeping a parameter the dynamics do not read (an
   observation-only parameter with observation output disabled) gives
   byte-identical trajectories across the two points. That is the D-B pin.
2. The combined table of a sweep job has `point_id`, `sweep:beta` and `seed` and
   no `draw`; `sweep:beta` round-trips `0.1234567` exactly; the two cells of one
   replicate carry the same `seed`; `point_id` starts at 0.
3. `--sweep beta=0.2,0.3` and `--sweep beta=0.3,0.2` produce different ensemble
   IDs, and a draws-only ensemble's grid-level bytes are unchanged (the unit
   test beside `levels_are_byte_identical_to_the_literals_they_replaced`).
4. `--sweep gamma=… --sweep beta=…` and `--sweep beta=… --sweep gamma=…` give
   the same `point_id`s, and the same as `fit predict --sweep` on the same grid.
5. `--sweep` with `--draws`, with `--design-from`, with a `--param` naming the
   swept parameter, with a repeated name, and with a scenario setting the swept
   parameter, are each refused with a message naming the conflict; the scenario
   case ports `scenario_coordinate_collision_gh572.rs:162-234` from batch to
   `simulate`, rewriting the fixture manifest and the `batch` helper (`:76-118`)
   it depends on; `:266`
   (`design_block_over_a_parameter_the_scenario_sets_is_refused`) ports as a
   `--sweep-file` collision test.
6. `--sweep-file` with an unknown column, with zero rows, and with a duplicate
   row is refused.
7. A sweep with `--replicates 3` and a `quantities {}` block writes one banded
   group per (point, scenario), never one band across points; with
   `--replicates 1` it writes one point-mode row per group.
8. A sweep with `--obs-only-dir` writes observation tables with `point_id`,
   `sweep:beta` and `seed` and no `draw`.
9. A second identical multi-cell run simulates zero cells and produces the same
   combined file; `--force` recomputes them; `--dry-run` reports every cell as a
   hit.
10. `--enable vaccination` labels its arm `baseline+vaccination` in the
    `scenario` column and the store listing; `--param beta=0.3` gives
    `baseline;beta=0.3`; plain `simulate` still gives `baseline`; the leaf
    identities are unchanged by the label.
11. `camdl batch` exits with clap's unknown-subcommand error
    (`cas_integration.rs:1069` already asserts this); no source line matches
    `cmd_batch_run`, `ExperimentToml`, `BatchArgs`, `CasSink::probe` or
    `sampling::`.

Ported, not deleted, because they guard the shared sink: the batch tests in
`cas_integration.rs:349, 434, 492, 878, 973, 1018`,
`design_preserving_simulate.rs:707`, `scenario_identity_gh573.rs:205, 213,
224`,
`simulate_identity_parity.rs:460`, and the obs-into-store coverage of
`acceptance_batch_obs_ensemble.rs`, each rewritten against `simulate`. The
model-graph archive test (`model_graph_archive.rs:68`) re-pins through
`simulate`. The three `CasSink` identity unit tests move with the code. Deleted
outright: `acceptance_batch_named_scenario.rs`, `acceptance_design_cas.rs`,
`gh589_batch_obs_preflight.rs`. Comment-only mentions in `determinism_pin.rs`,
`gh656_emit_every.rs:371`, `acceptance_obs_only_dir.rs`,
`acceptance_set_block_comma_error.rs` and `compare_pointwise_elpd.rs` are
reworded.

The shell test `tests/test_ocaml_to_rust.sh:93-142` is the only end-to-end
multi-model store run in the gate: eight fixture models at `--parallel 2` with a
resume check. It is re-expressed as `simulate --sweep … --replicates` over the
same eight models with the same parallelism and resume check, so that coverage
is kept, not lost.

One check that is run once before deletion and not kept: the book's sweep (five
β values, two scenarios, fifty seeds) run through `simulate --sweep` produces
the same set of leaf identities batch produced from
`guide/boarding-school/sweep.toml`. The book's call ports as
`camdl simulate models/sirv.camdl --params params/params.toml --scenario baseline,with_vaccination --sweep beta=lin(0.2,0.6,5) --seeds 1:50`,
and the chapter's regex over the leaf label keeps working because
`params_path_label` stays; `sweep:beta` in `ensemble.tsv` is the better path.
(The book's `sirv.camdl` does not compile against main today, E266: its
observation block predates the required `columns {}`; `camdl-docs#1` notes it.)

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
`README.md:232, 360, 401`; `scripts/review-zip.sh:66`, `scripts/README.md:68`;
`docs/dev/testing.md` and `docs/dev/types-reference.md` where they mention
batch. Add `--sweep` and `--sweep-file` to `simulate`'s reference in
`docs/commands.md` and to the run spec's multi-cell section, with the D-B seed
rule, the independent-noise idiom, the `seed` pairing column, the store re-use
behaviour, the derived arm label and the observation-directory idiom stated. The
book's chapter rewrite is `vsbuffalo/camdl-docs#1` and lands in step with the
deletion.

## 8. Identity

`--sweep` and `--sweep-file` are new, so no existing run re-keys. The ensemble
grid level gains a top-level optional field that is absent for every existing
job and skipped when absent, so every existing ensemble ID is unchanged; test 3
pins the bytes. Store re-use changes no identity: it reads what the store
already holds. The derived arm label is presentation and re-keys nothing. The
tag `archive/batch` is placed on the last commit on main that carries the verb,
before the deletion commit, with this proposal named in the tag message.

## 9. Order of work

1. **PR A, pure move.** §5's shared sink to `cas_sink.rs`, `probe` left behind
   for deletion; `simulate` and `batch` outputs byte-identical on the fixtures;
   full `make test`.
2. **PR B, the flags.** `--sweep`, `--sweep-file`, the lifted expansion, the
   seed rule, the `RunEntry` and grid-level fields, the combined and observation
   columns, the quantities key, store re-use, the dry-run view, the resolved
   design in the record, the derived arm label, tests 1–10; full `make test`.
3. **PR C, the deletion.** Tag `archive/batch`; delete §5's list; port and
   delete tests per §6, re-express the shell test; docs per §7; close gh#930 and
   gh#286; relabel gh#738 and gh#599 to `simulate`/`CasSink`; note the scope
   reduction on gh#923 and gh#598; full `make test`. The book chapter
   (`camdl-docs#1`) lands with it.

Each PR formats every touched `.md` with `mdfmt` except
`docs/camdl-language-spec.md`, and carries no attribution trailers.

## 10. Relation to the coordinate-kind arc, and what it must be told

This takes the design half of the coordinate-kind proposal's flag stage
(`--sweep`, `--sweep-file`, the name-sorted product, D-B seeds, `point_id` and
`sweep:<param>` columns, the grid-level field in its specified form) and leaves
the rest: the `--draws <path>` refusal, the provenance sidecar, the design ×
sample product, `band`, the per-cell and summary tables, fixed headers, the
within-axis `replicate`, and `fit predict`'s move onto the shared grid. Both
forks of that arc keep D-B and the two columns, so nothing here is undone by
either; the `seed` column is additive. What the rewrite inherits is simpler by
the whole of `batch`.

Two things the coordinate-kind proposal and the session holding its rewrite must
be told when PR B lands: that its flag stage's design half has shipped on
`simulate` (so its stages renumber and its Stage 5 shrinks to the
`--draws
<path>` refusal and the sidecar), and that `point_id` is 0-based as D-C
said and every other user-facing index is moving to 0 as well (so its §9.1 and
the note's within-axis `replicate` are 0-based too). The maintainer asked that
the design-vs-sample session be told directly as well as through the commits.

## 11. Risks

- The combined-table columns for sweep jobs are specified from the accepted
  §9.1; if the rewrite changes the table shape, these columns survive but the
  file around them changes once more. That is already true of every quantity
  file.
- `--sweep-file` without the `--draws <path>` refusal means two file flags with
  opposite semantics coexist until the rewrite lands the refusal. The help text
  for each says which axis it builds.
- Store re-use for multi-cell `simulate` changes a path existing users run: a
  re-run that used to recompute now reads back. The bytes are the same by
  construction; the risk is a stale leaf masking a changed input, which the
  store's identity makes impossible unless the identity is wrong, and `--force`
  remains the override.
- The derived arm label changes the `scenario` column of every existing
  `--enable`/`--disable`/`--param` run from `baseline` to the derived form. That
  is a presentation change in combined files and listings, not an identity
  change; the book's scenarios chapter reads the column and is updated in
  `camdl-docs#1`.

## 12. Decisions (the maintainer, 2026-10-24)

1. **Multi-cell `simulate` re-uses cells the store already holds**, for
   consistency with every other store-backed verb. Decided; §3.
2. **The pairing key is a `seed` column now**; the within-axis `replicate` comes
   with the coordinate-kind rewrite, layered on top. Decided; §3.
3. **One index base, 0, everywhere a user sees an index.** `point_id` stays
   0-based as D-C said; the 1-based `draw`, `replicate`, chain directories and
   chain labels move to 0 in a separate issue that supersedes gh#980's two-type
   design. Decided; §3.
4. **Ad-hoc arms get a derived label** (`baseline+vaccination`,
   `baseline;beta=0.3`) in the flags PR; no mandatory name, no new flag.
   Decided; §3 "Arm labels".
5. **Observation tables under a sweep carry `point_id`, `sweep:<param>` and
   `seed`**, never `draw`. Decided; §3.
6. **The ensemble leaf's existing `run.json` carries the resolved design.** No
   new file. Decided; §3.
7. **No path-less observation flag**: the destination directory is the dataset a
   modeller fits to, and the observations also land in each leaf. Decided; §4.
8. **The shell store test is re-expressed over `simulate --sweep`.** Decided;
   §6.
9. **The coordinate-kind proposal is amended and its session told** when PR B
   lands. Decided; §10.
