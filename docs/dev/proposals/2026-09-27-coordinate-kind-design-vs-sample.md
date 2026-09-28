# A parameter point is a design coordinate or a sample, and the cell says which

- Date: 2026-09-27
- Status: draft — five calls are open for the maintainer (§10); everything else
  is decided inline
- Relates-to: gh#562, gh#572, gh#575,
  `2026-06-27-sealed-fit-packets-handles-and-override-algebra.md` (§4),
  `2026-08-11-scenario-banding-in-simulate.md` (implemented)
- Builds on: an untracked architecture-review note on gh#562 (2026-08-11). Every
  claim this proposal uses from it is restated in §2; the note is not needed to
  read this document.

All code references are against `origin/main` at `2e141e0a`. A reference marked
_verified_ was read at that line; _inferred_ means reasoned from the code but
not executed.

## Summary

A run grid in camdl is a product of axes. Some axes index _different things
being compared_ (a scenario, a sweep point); others index _repeated sampling of
one thing_ (a posterior draw, a stochastic replicate). Three decisions depend on
which kind an axis is: whether a scenario may override it, how its cells are
labelled, and whether a quantile may be taken across it. Today none of the three
can read that property off the cell, because the cell does not carry it. Each
decision instead consults `ParamSource` — which CLI flag produced the parameter
points — and the flag is the wrong property: it classifies a hand-authored grid
passed via `--draws` as a sample, and classifies an iid uniform design in
`batch run` as a partition while classifying the same draw in `simulate` as a
sample.

This proposal replaces `ParamSource` with a `ParamPlan` that holds two typed
axes — a **design axis** and a **sample axis** — and splits the cell's single
`point_overrides` map into a design coordinate and a sample coordinate.
Collision policy, leaf labels and provenance, and banding then key off one
property that the cell carries, set by one constructor. `fit predict`'s sweep
moves onto the grid's design axis so both verbs put their partition coordinates
in the cell. Nothing re-keys run identity; the IR and goldens are untouched.

## 1. Terms

A **cell** is one simulation: one scenario, one parameter vector, one stochastic
replicate (`engine::CellSpec`, `engine.rs:70-88`, verified).

A **partition axis** is an axis whose values index different objects being
compared. Results are grouped by it and reported with it as a column; no summary
pools across it. Scenario is a partition axis.

A **band axis** is an axis whose values index exchangeable repeated samples of
one object. A quantile band — the `q05 … q95` columns of a `quantities/*.tsv` —
is a summary _over_ band axes, holding partition axes fixed. Replicate (the
process-noise seed) is a band axis.

A **design coordinate** is a parameter point the user chose as a labelled
condition: a sweep value, a row of a hand-authored grid, a point of a
space-filling design. Design coordinates form a partition axis.

A **sample** is a parameter point drawn from a stated probability measure over
parameters — the posterior of a fit, a prior, a uniform law over declared
bounds. Samples form a band axis. The measure is what makes a quantile over them
mean something: it is an estimate of a quantile of the push-forward of that
measure through the model.

A **marginal reduction** summarizes a set of cells treating them as exchangeable
draws from one distribution — a quantile, a mean. A **paired reduction**
combines cells from _different_ values of a partition axis that share a band
coordinate — draw _i_ of the baseline arm with draw _i_ of the intervention arm
— to construct a new random variable before any marginal summary.

## 2. The problem, restated from the review note

This section restates the parts of the gh#562 review note that this proposal
builds on, so the proposal stands alone.

### 2.1 One field carries two meanings, and the cell does not say which

`CellSpec.point_overrides` (`engine.rs:81`, verified) is the parameter point for
the cell. When the job's `ParamSource` is `Sweep` (`sim_job.rs:175`), each point
is a different model — a partition coordinate. When it is `Draws`
(`sim_job.rs:178-191`), each row samples one model — a band coordinate. The same
loop in `plan_grid` (`engine.rs:168-191`, verified) stamps both into the same
field. No function taking only a `&CellSpec` can compute a correct partition
key. The note observed that this was contained only because sweeps existed
solely in `batch run`, which emits no quantities; that is still true on `main`.

### 2.2 `fit predict` keeps its sweep coordinate outside the cell

`fit predict` does not use the grid's sweep axis. It expands `--sweep` itself
(`predict.rs:1589`, verified), then for each sweep point overwrites the swept
parameters in every posterior draw row (`predict.rs:2234-2244`) and runs one
`ParamSource::Draws` job per scenario (`predict.rs:2258-2262`,
`scenarios: vec![sref.clone()]`). The sweep value lives in the loop variable
`sweep_pt` (`predict.rs:2193`) and reaches the renderer directly
(`predict.rs:2301-2304`). Every predict cell therefore looks like a posterior
draw; its partition coordinate is not in the cell.

### 2.3 The rejected design, and the four reasons

The gh#562 work designed and rejected a general coordinate type:

```rust
enum PointKind { Single, Sweep, Draw }        // stamped on CellSpec by plan_grid
struct BandKey { scenario: String, sweep: Vec<(String, u64)> }
impl BandKey { fn of(spec: &CellSpec) -> BandKey; }
struct BandSet { bands: IndexMap<BandKey, Band> }   // BandSet::push(key, …)
```

It was rejected on four grounds (recorded in `2026-08-11` §4):

1. **A caller-supplied key does not deliver the guarantee.**
   `BandSet::push(key,
   …)` takes the key from its caller, so a caller passing
   one key for every cell pools exactly as before. The shipped alternative — an
   accumulator whose only input is a whole cell, `push_cell(&CellResult)`,
   deriving its own key (`main.rs:2452-2456`, verified) — gives the caller no
   key to pass.
2. **`BandKey::of(&CellSpec)` cannot serve `fit predict`** (§2.2), so a second
   public constructor would be needed.
3. **`PointKind` names the wrong property.** It records which CLI flag produced
   the points, not whether they carry a measure over which a quantile means
   anything. A hand-authored grid passed via `--draws` would classify as a band;
   the same grid via a sweep as a partition. `--draws uniform` bands today,
   though no belief stands behind it.
4. **It was unreachable:** `BandKey::of` would have returned an empty sweep at
   every live call site.

### 2.4 The collision policy is being keyed on the same wrong property

A scenario resolves at tier 4 of the parameter resolver and a draw row or sweep
point at tier 3.5 (`params_resolver.rs:48-52`, `:880-903`, `:905-946`,
verified), so a scenario that sets or scales a parameter the point also sets
silently wins. The run-spec audit on gh#572 reproduced three sweep leaves, and
three `--draws uniform` leaves under `--scenario baseline`, each set
byte-identical.

The existing guard, `check_explicit_draws_scenario_collision`
(`engine.rs:358-405`, verified), fires only for
`ParamSource::Draws { explicit_file: Some(_) }`. `fit predict` has its own guard
for its own sweep (`predict.rs:1628-1653`, verified). `batch run` has none. The
maintainer's policy for gh#572 is: a scenario colliding with a _user-authored_
coordinate (a sweep, an explicit draws file) is refused; one colliding with a
_generated_ draw (`prior`, `uniform`, `posterior`) is allowed. That fix is in
progress keyed on `ParamSource` — the property reason 3 identifies as the wrong
one. This proposal supplies the property it should key on. (The in-progress
branch was not inspected; its keying is as described in the task that
commissioned this proposal.)

## 3. What the code shows, beyond the note

Five facts found while reading the code sharpen or correct the note.

**`--draws uniform` is an iid sample, not a space-filling design.**
`generate_uniform_draws` draws `lo + (hi − lo)·U` independently per parameter
per row (`main.rs:3852-3889`, verified); parameters without declared bounds are
held at their default. The run spec calls it "space-filling exploration for
model debugging, not a prior" (`docs/camdl-run-spec.md:1138-1140`). So it _does_
carry a measure — the uniform law on the declared box — and a quantile over it
is a well-defined Monte Carlo estimate of the push-forward of that law. What it
lacks is a claim that the law is anyone's belief. Consequence: "carries a
measure" is necessary for a band but not sufficient to call the band a credible
interval, and the same point set can legitimately be read either way (§4.1).

**The actual space-filling designs already run as sweeps.** `batch run`'s
`[design.NAME]` blocks (`method = "sobol" | "lhs" | "random"`,
`batch.rs:98-107`, verified) are routed through `ParamSource::Sweep`
(`batch.rs:1870-1900`, verified). An iid uniform design is therefore a partition
axis in `batch run` and a band axis in `simulate` — reason 3 in running code,
not only in principle.

**No output says "credible" today.** The quantity renderer writes `n_draws` and
`q05 … q95` (`quantity_output.rs:266`), and the manifest carries `schema`,
`calendar`, and per-quantity shape fields (`quantity_output.rs:466-515`); none
of them names what the band is over. The user-visible defect is therefore an
_unlabelled_ band whose meaning depends on the source, not a mislabelled one.

**Run identity already hashes effective values.** Since PR #941 the `params`
identity level hashes the values the resolver hands the engine
(`batch.rs:1117-1135`, `resolve.rs:227-233`, verified), and the leaf path label
names the point's parameters at the values the cell _ran_ (`batch.rs:1139-1146`,
verified); `run.json` records the same resolved values as `inputs`
(`resolve.rs:237-244`). So gh#572's complaint that identity is keyed on the
requested coordinate is fixed. What still reports the requested value is the
_design_ surface: the dry-run's `sweep override` listing, a design block's
`parameter_points.tsv`, and `sweep:<param>` quantity columns.

**Common random numbers across design points are accidental and uneven.** The
process seed is `mix_cell_seed(base, point_idx, rep)` (`engine.rs:52-66`,
`util.rs:35-37`, verified) unless seeds are explicit, in which case it is
`seeds[rep]`. `batch run` always passes explicit seeds (`batch.rs:588-589`), so
its sweep points share seeds per replicate. `fit predict` runs one job per sweep
point in which `point_idx` is the draw index, so its sweep points share seeds
per draw. A hand-authored grid in `simulate` mixes its row index into the seed,
so its points share nothing. The typed split in §5 makes this visible; §10 D4
decides what to do about it.

Two smaller findings. `ValueSource::SweepPoint` tags a posterior draw's value
`"sweep_point"` (`params_resolver.rs:185-189`, `:200`, `:893`, verified) — the
same conflation at the provenance layer. And the doc comments at
`quantity_output.rs:35-36` and `:476-479` still describe `DesignCoords::none`,
which the gh#562 work deleted.

## 4. The rule: design coordinates condition, samples are marginalized

### 4.1 The kind is a declared reading, and only a sample needs a measure

The working hypothesis was that a point is either a design coordinate or a
sample from a stated measure. The code confirms the dichotomy but refines where
it comes from: the kind is not a property of the numbers. An iid uniform draw
can be read as a sample of the uniform law (to band a quantity over it) or as a
set of design conditions (to plot the quantity against the parameter). A
posterior draw can be read as a sample (the posterior predictive) or as a
condition (one trajectory per draw). What differs is the _question_ — condition
on the point or marginalize over it.

The asymmetry that makes this safe: **any point set may be read as a design;
only a point set with a measure may be read as a sample.** A sweep grid has no
measure, so it can never be a sample. A generated draw has a measure by
construction. A user-authored file has one only if the user asserts it. The type
in §5 encodes exactly this: `SampleAxis` cannot be constructed without a
`Measure`, and `DesignAxis` needs nothing.

Each generator has a default reading:

| Source                                 | Default | Measure (if sample)     | May be re-read as   |
| -------------------------------------- | ------- | ----------------------- | ------------------- |
| `--draws posterior`, `fit predict`     | sample  | posterior of the fit    | design              |
| `--draws prior`                        | sample  | prior (IR or fit.toml)  | design              |
| `--draws uniform`                      | sample  | uniform on declared box | design (§10 D2)     |
| `--draws FILE`                         | design  | —                       | sample, if asserted |
| `batch run [sweep]`, `predict --sweep` | design  | —                       | never (no measure)  |
| `batch run [design.NAME]`              | design  | —                       | not offered (§4.4)  |

### 4.2 What each kind implies for the three decisions

| Decision                                | Design coordinate                                          | Sample                                                             |
| --------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| A scenario sets or scales the parameter | Contradiction — refused, naming both sources               | Counterfactual — allowed; the shadowed columns are recorded (§6.2) |
| Labels and provenance                   | Reported as a column (`sweep:<param>`) and in the manifest | Not reported per row; the manifest names the measure               |
| Marginal reduction (quantile, mean)     | Never crosses it                                           | Crosses it                                                         |
| Paired reduction                        | Crosses it, matched on the band coordinate                 | Supplies the matching coordinate                                   |

The collision row reproduces the maintainer's gh#572 policy on every input the
policy was written for — sweeps and explicit files are refused, generated draws
allowed — because those inputs default to design and sample respectively. It
differs only where the user declares a reading
(`--draws FILE --draws-kind
sample`, §10 D1), and there it follows the
declaration rather than the flag.

### 4.3 Marginal reductions stay within a partition; paired reductions cross it

The rule from `2026-08-11` §1 stands and becomes checkable per cell:

> A marginal reduction may cross band axes only. A partition axis is eliminated
> only by an explicit measure over its levels (none exists in camdl today) or by
> a paired reduction, which constructs a new random variable per band coordinate
> before any marginal summary.

`fit/contrasts.rs` is the paired reduction in the codebase: for each forkable
posterior draw it replays every arm under one seed,
`derive_chain_seed(seed,
draw_pos)` (`contrasts.rs:443-455`, verified), and
differences them before banding. Under this proposal the pairing coordinate has
a name: the cell's sample coordinate (and replicate). Two consequences follow,
decided here:

- A paired reduction across a partition axis is well-defined iff the cells on
  both sides share one `SampleAxis` (same rows, same measure) — the band
  coordinate then denotes the same parameter vector on both sides. Pairing
  across scenarios always satisfies this (a job has one sample axis). Pairing
  across design points satisfies it when the design axis is crossed with a
  sample axis, as in `fit predict --sweep`.
- Whether a paired reduction also enjoys variance reduction from common random
  numbers depends on the seed rule (§3, §10 D4), not on validity.

### 4.4 Out of scope, recorded so they are not re-raised

**Weighted designs.** A `[design.NAME]` parameter may carry a `prior` used for
importance weighting by `camdl voi` (`batch.rs:111-117`). A space-filling design
plus importance weights is, in principle, a weighted sample of the prior. camdl
has no weighted quantile path and `batch run` emits no quantities, so design
blocks are design axes, and a weighted-sample reading waits for a consumer.

**`fit run --sweep`.** A profile-likelihood sweep over fits is a different code
path (a fit per point, not a simulation grid). The gh#572 instance there is
inferred from the code path in `2026-08-11` §9 and is not addressed here.

## 5. Types

### 5.1 The plan: two axes and a replicate count

`ParamPlan` replaces `ParamSource` in `SimulateJob.source` (`sim_job.rs:70`):

```rust
// sim_job.rs

/// The parameter-point structure of a job: a design axis crossed with a sample
/// axis, each replicated `replicates` times. Either axis may be the unit axis
/// (one empty point). Replaces `ParamSource`.
#[derive(Debug, Clone)]
pub struct ParamPlan {
    design: DesignAxis,
    sample: SampleAxis,
    replicates: usize,
}

/// Labelled conditions the user chose. A partition axis.
#[derive(Debug, Clone)]
pub struct DesignAxis {
    /// The swept/authored parameter names, sorted. Every point sets exactly
    /// these names (a grid point or a file row with a missing column is refused
    /// at construction).
    names: Vec<String>,
    points: Vec<DesignPoint>,
    origin: DesignOrigin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DesignPoint {
    /// Requested values, aligned to `DesignAxis::names`.
    values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum DesignOrigin {
    Unit,                                   // no design axis
    Sweep,                                  // Cartesian grid
    File { path: PathBuf },                 // --draws FILE read as design
    Generated { method: DesignMethod },     // batch [design.NAME]; --draws … as design
}

/// Exchangeable draws from a stated measure. A band axis.
#[derive(Debug, Clone)]
pub enum SampleAxis {
    Unit,
    Draws { rows: Vec<IndexMap<String, f64>>, measure: Measure },
}

/// What the sample is a sample OF. A `SampleAxis::Draws` cannot exist without
/// one, which is the whole of the design/sample asymmetry (§4.1).
#[derive(Debug, Clone, PartialEq)]
pub enum Measure {
    Posterior { fit: String, method: String },   // fit handle + stage label
    Prior { source: PriorSource },               // model IR or fit.toml
    UniformBounds,                               // lo + (hi-lo)·U on declared bounds
    Asserted { path: PathBuf },                  // --draws FILE --draws-kind sample
}
```

The fields are private to `sim_job`. The constructors are the only way in, and
each validates its invariant:

```rust
impl ParamPlan {
    pub fn point(replicates: usize) -> Self;
    pub fn new(design: DesignAxis, sample: SampleAxis, replicates: usize)
        -> Result<Self, String>;   // refuses a design name absent from the model
}
impl DesignAxis {
    pub fn unit() -> Self;
    pub fn sweep(grid: Vec<IndexMap<String, f64>>) -> Result<Self, String>;
    pub fn from_rows(rows: Vec<IndexMap<String, f64>>, origin: DesignOrigin)
        -> Result<Self, String>;   // refuses ragged rows and duplicate rows
}
impl SampleAxis {
    pub fn draws(rows: Vec<IndexMap<String, f64>>, measure: Measure) -> Self;
}
```

Duplicate design rows are refused because two partitions with identical
coordinates are indistinguishable in a tidy output; a user who wants the same
condition twice wants `--replicates`.

A design name may also be a sample column: `fit predict --sweep beta` crosses a
`beta` design axis with posterior rows that carry `beta`. The design value wins
for that cell, which is exactly today's behaviour (the row overwrite at
`predict.rs:2239-2241`). §5.4 makes that precedence a resolver tier rather than
a map overwrite.

### 5.2 The cell: coordinates replace `point_idx` + `point_overrides`

```rust
// engine.rs

#[derive(Clone)]
pub struct CellSpec {
    run_idx: usize,
    coords: CellCoords,
    process_seed: u64,
    obs_seed: u64,
    sim_run: SimRun,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CellCoords {
    pub scenario: ScenarioRef,
    pub design: DesignCoord,
    pub sample: SampleCoord,
    pub rep: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DesignCoord { Unit, Point { idx: usize, point: Arc<DesignPoint> } }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleCoord { Unit, Draw { idx: usize } }

/// The grouping a marginal reduction must respect. Derived, never supplied.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartitionKey { scenario: String, design: Option<usize> }

impl CellCoords {
    pub fn partition_key(&self) -> PartitionKey;
    /// The index the seed and every per-draw lookup are keyed on (§5.3).
    pub fn seed_index(&self) -> usize;
}
```

`CellSpec`'s fields become private to `engine`, read through accessors, and
`plan_grid` (`engine.rs:148-193`) is the single constructor. Today there are
four construction sites (`2026-08-11` §4 counted them); the non-test one outside
`plan_grid` is `CasSink::predict_cells` (`batch.rs:1265-1300`, verified), which
re-implements the grid loop for dry-run cache classification. It is changed to
call `plan_grid` on the same job (Stage 0). The two test fixtures
(`batch.rs:2765`, `:2843`) build through a `#[cfg(test)]` constructor in
`engine`.

`plan_grid` iterates `scenario → design point → sample → replicate`. With either
axis the unit axis, this is today's `scenario → point → rep` order exactly.

### 5.3 The seed index preserves every current trajectory

`seed_index` is the sample index when the sample axis is non-unit, otherwise the
design index, otherwise 0:

```rust
fn seed_index(&self) -> usize {
    match (&self.sample, &self.design) {
        (SampleCoord::Draw { idx }, _) => *idx,
        (SampleCoord::Unit, DesignCoord::Point { idx, .. }) => *idx,
        (SampleCoord::Unit, DesignCoord::Unit) => 0,
    }
}
```

`process_seed_for` receives `seed_index` where it receives `point_idx` today.
Every existing job maps to the same number (verified by reading each
construction site; pinned by Stage 1's test):

- `simulate --draws posterior|prior|uniform`: design unit, sample index =
  today's `point_idx`.
- `simulate --draws FILE` read as design: sample unit, design index = today's
  `point_idx`.
- `batch run`: explicit seeds, so `seeds[rep]` and the index is unused
  (`engine.rs:59-60`).
- `fit predict --sweep` after Stage 4: sample index = the draw index, which is
  today's `point_idx` inside each per-sweep-point job. The `total_runs == 1`
  branch (`engine.rs:61-62`) changes from firing per sweep-point job to not
  firing on the combined job when there are several sweep points, but
  `mix_cell_seed(base, 0, 0) == base` (`util.rs:35-37`; asserted at
  `engine.rs:535`), so the seed is the same.

The same index replaces `point_idx` at the three per-draw lookups that must
follow the sample, not the flat point: the `--init-state fit` row
(`engine.rs:467-471`), `ConditionedSource::per_draw` (`predict.rs:1187`), and
`chain_of_point` (`predict.rs:1262`). The wide-format `draw` column
(`main.rs:2923`) keeps printing `seed_index + 1`, byte-identical.

### 5.4 The resolver tier 3.5 splits in two, design over sample

Tier 3.5 (`params_resolver.rs:880-903`) becomes 3.5a (sample row) then 3.5b
(design point), and `ValueSource::SweepPoint` splits into `ValueSource::Sample`
and `ValueSource::DesignPoint`. `SimRun.point_overrides` (`engine.rs:484`)
splits into `sample_overrides` and `design_overrides`, filled by
`build_cell_sim_run` from the cell's coordinates. The resolved values are
unchanged for every current job: no current job has both axes except
`fit predict --sweep`, where the design value already overwrote the sample value
before the resolver saw it.

### 5.5 Parse at the boundary: where each plan is built

| Front end                 | Today (`ParamSource`)                                           | After (`ParamPlan`)                                                                                            |
| ------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `simulate`                | `main.rs:1772-1785`: `Draws{explicit_file}` or `Point`          | generator → `SampleAxis::draws(rows, Measure::…)`; file → `DesignAxis::from_rows` unless `--draws-kind sample` |
| `batch run [sweep]`       | `batch.rs:926-935`: `Sweep` or `Point`                          | `DesignAxis::sweep`                                                                                            |
| `batch run [design.NAME]` | `batch.rs:1900`: `Sweep`                                        | `DesignAxis::from_rows(.., Generated{method})`                                                                 |
| `fit predict`             | `predict.rs:2258`: `Draws{explicit_file: None}` per sweep point | one plan: `DesignAxis::sweep` × `SampleAxis::draws(.., Posterior)`                                             |

The `explicit_file` field, whose doc comment carries the policy today
(`sim_job.rs:182-190`), is deleted; the policy moves to §6.

### 5.6 Consumers read the coordinates, never the source

| Consumer                                         | Reads                                      | Replaces                                                                     |
| ------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------------------------------- |
| Collision guard (§6)                             | `ParamPlan` design names vs sample columns | `check_explicit_draws_scenario_collision`, predict's guard, the gh#572 guard |
| `SimQuantities::push_cell` (`main.rs:2456`)      | `cell.spec.coords().partition_key()`       | `cell.spec.scenario.name()`                                                  |
| `PredictiveSink::merge_cell` (`predict.rs:1127`) | same                                       | same, plus the enclosing `sweep_pt` loop variable                            |
| Quantity `Mode` (`main.rs:1917-1929`)            | `ParamPlan::mode()` (§7.1)                 | `matches!(source, Point { replicates: 1 })`                                  |
| `CasSink::cell_resolve` label (`batch.rs:1143`)  | design names ∪ sample columns              | `point_overrides.keys()`                                                     |
| Seed and per-draw lookups (§5.3)                 | `coords.seed_index()`                      | `point_idx`                                                                  |

## 6. Collision policy keys on the coordinate kind

### 6.1 One guard, in `run_job`, for every verb

`run_job` (`engine.rs:202-203`) calls one guard in place of
`check_explicit_draws_scenario_collision`:

```rust
fn check_scenario_coordinate_collision(job: &SimulateJob, model: &ir::Model)
    -> Result<Vec<ShadowedSample>, String>
```

For each scenario it computes `scenario_param_footprint`
(`params_resolver.rs:
689-725`, the shared authority both current guards already
use) and intersects it with:

- the design axis's `names` — a non-empty intersection is a hard error naming
  the parameters, the scenario, and the design's origin (the sweep, the file
  path, or the design block);
- the sample axis's columns — a non-empty intersection is allowed and returned
  as a `ShadowedSample { scenario, params }`.

Because the guard reads the plan, and `fit predict --sweep` builds a plan with a
design axis after Stage 4, predict's separate guard (`predict.rs:1628-1653`) is
deleted there, and `batch run` is covered without a verb-specific call. The
footprint∩names rule is the one `2026-08-11` §1.1 argues is exact (disjointness
is required for both distinctness and denotation), so the guard is not a stopgap
for a later graded rule.

The graded rule proposed in the gh#572 comment (warn on partial shadowing, error
on total) is not adopted for design axes. Its motivating example — a scenario
pinning `gamma` while a sweep varies `beta` — has an empty intersection and is
already permitted; what remains in the intersection is a genuine contradiction.

### 6.2 A shadowed sample is a counterfactual and is recorded, not hidden

When a scenario pins a sampled column, the arm answers "the quantity under the
scenario, with the remaining parameters drawn from the measure" — a legitimate
counterfactual. It is still a change to what the band is over, so it is written
down: `run_job` prints one stderr line per shadowed scenario, and the quantities
manifest entry for that scenario carries `"shadowed": ["gamma", …]` inside its
`summary` object (§7.2). When a scenario shadows _every_ sampled column, the
band is over process noise alone; the manifest's `summary.over` then lists only
`replicate`, so the band cannot be read as parameter uncertainty (§7.2 derives
`over` per partition, not per job).

## 7. Banding keys on the coordinate kind

### 7.1 `Mode` is a function of the plan

```rust
impl ParamPlan {
    pub fn mode(&self) -> Mode {
        match (&self.sample, self.replicates) {
            (SampleAxis::Unit, 1) => Mode::Point,
            _ => Mode::Banded,
        }
    }
}
```

This equals today's predicate (`main.rs:1924-1929`) on every input except a
`--draws FILE` read as design with `replicates == 1`, which becomes `Point` per
design point instead of one band pooled over the file's rows. That is the
intended change (§9, row 1).

### 7.2 `simulate` partitions by design point and the manifest says what the band is over

`SimQuantities.by_scenario` becomes
`by_partition: IndexMap<PartitionKey,
ScenarioQuant>`, keyed inside `push_cell`
from the cell as today (rejection reason 1 is not reintroduced — see §8). At
render, each partition becomes one `StackedQuantities::push_group` call whose
`DesignCoords` are built by one function from the key and the plan:

```rust
impl DesignCoords<'_> {
    fn of<'a>(key: &'a PartitionKey, plan: &'a ParamPlan, scenario_axis: bool)
        -> DesignCoords<'a>;
}
```

`DesignCoords.sweep` already renders `sweep:<param>` columns
(`quantity_output.rs:227-243`) and a manifest `sweep` object (`:480-491`);
`simulate` starts populating it for a design axis. The column prefix stays
`sweep:` for every design origin (§10 D5).

Each manifest entry gains one object, additive to `camdl.quantities/v1`:

```json
"summary": {
  "over": ["sample", "replicate"],
  "measure": "posterior",
  "n_samples": 200,
  "replicates_per_sample": 1,
  "shadowed": []
}
```

`over` lists the band axes actually reduced for that partition; `measure` is the
`Measure` tag, or `"process"` when only replicates are reduced, or absent in
`Point` mode. This is the label gh#575 asked for. A consumer — or a figure
caption — can now say "95 % posterior predictive interval" only when `measure`
is `posterior`, and "quantiles under a uniform law on the declared bounds" when
it is `uniform_bounds`. The `n_draws` column is not renamed here; gh#575's
rename is independent.

## 8. Why this does not reintroduce the caller-supplied key

Rejection reason 1 was that `BandSet::push(key, …)` let a caller choose the key.
Nothing here takes a key from a caller:

- Both accumulators keep the `push_cell(&CellResult)` /
  `merge_cell(&CellResult)` shape and derive `PartitionKey` from
  `cell.spec.coords()`. There is no method with a key parameter.
- `CellCoords` is set only by `plan_grid`, from the `ParamPlan`; `CellSpec`'s
  fields are private to `engine`. A sink cannot fabricate or edit coordinates.
- `PartitionKey` has no public constructor; `partition_key()` is its only
  source.
- `StackedQuantities::push_group` still takes `DesignCoords` from its caller.
  That is safe for the reason `2026-08-11` §3.5 gives — it is a rendering seam
  over groups that are already separated, and a renderer cannot re-merge them —
  and after this change the coordinates it receives are computed by
  `DesignCoords::of` from a key the accumulator derived.

Reason 2 (`BandKey::of` cannot serve predict) is removed by Stage 4: predict's
sweep is in the cell, so one derivation serves both verbs. Reason 3 is the
subject of §4. Reason 4 (unreachable) no longer holds: `simulate --draws FILE`
reaches the design path the day Stage 3 lands, and predict's sweep reaches it at
Stage 4.

## 9. What changes for users

| # | Command                                                                         | Today                                 | After                                                                                                                                                                             |
| - | ------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `simulate --draws FILE --quantities-out d`                                      | one band pooled over the file's rows  | one row per design point with `sweep:<param>` columns (one band per design point over replicates under `--replicates N`); `--draws-kind sample` restores the pooled band (§10 D1) |
| 2 | `simulate --draws FILE --draws-kind sample --scenario S` (S sets a file column) | refused (flag does not exist)         | allowed as a counterfactual; shadowed columns in stderr and manifest                                                                                                              |
| 3 | `simulate --draws FILE --scenario S` (S sets a file column)                     | refused                               | refused (unchanged; message names the design origin)                                                                                                                              |
| 4 | `batch run` sweep or design block × a scenario setting a swept parameter        | runs; cells identical                 | refused (the gh#572 outcome)                                                                                                                                                      |
| 5 | `simulate --draws uniform                                                       | prior                                 | posterior --scenario S` (S sets a drawn column)                                                                                                                                   |
| 6 | Any banded quantities output                                                    | no statement of what the band is over | manifest `summary` object (additive)                                                                                                                                              |
| 7 | `fit predict --sweep …`                                                         | —                                     | byte-identical outputs (Stage 4 is required to be)                                                                                                                                |
| 8 | `run.json` provenance tag for a draw or sweep value, where written              | `sweep_point`                         | `sample` or `design_point`                                                                                                                                                        |

Row 1 is loud rather than silent: a per-point table has a different header and
`n_rows × n_points` rows, so it cannot be mistaken for a band. That is the
property that makes design the safe default (§10 D1).

**Run identity does not re-key.** Identity hashes resolved parameter values and
the process seed (`resolve.rs:202-233`). §5.3 shows every current cell keeps its
seed; §5.4 shows every current cell keeps its resolved values. The provenance
tag in row 8 is recorded, not hashed, for `sim` leaves (`run.json.inputs` is
display-only, commit `2e141e0a`); for fit and profile leaves, whether
`parameters_provenance` (`run_meta.rs:644`) enters any hash is _not verified_ —
Stage 5 carries the test that settles it.

**IR and goldens are unaffected.** All changes are in `rust/crates/cli`. No
`ir/VERSION` bump, no `ocaml/` change, no golden regeneration.

## 10. Decisions for the maintainer

**D1 — What does an explicit `--draws FILE` default to, and how is it
declared?**

- (a) Default design; `--draws-kind sample` asserts a measure
  (`Measure::
  Asserted`). Collision behaviour for undeclared files is
  unchanged from today. Breaks one documented recipe's output shape: "Posterior
  predictive from a draws file" (`docs/camdl-run-spec.md:1952-1954`) would emit
  per-draw rows under `--quantities-out` until `--draws-kind sample` is added.
  Loud, not silent.
- (b) Default sample; `--draws-kind design` for grids. No recipe breaks, but a
  hand-authored grid passed without the flag is banded — the silent wrong answer
  reason 3 names.
- (c) No default: refuse `--draws FILE` without `--draws-kind` whenever the kind
  is load-bearing (`--quantities-out`, or a scenario footprint touching a file
  column). Most honest; adds a required flag to existing invocations in those
  cases.
- In-file declaration (a directive line in the TSV header) is rejected: the
  reader treats line 1 as the header (`main.rs:4384-4396`), so a directive would
  break every older camdl and every tool reading these files, and `--draws-out`
  round-trips would need a format change.

Recommendation: (a). A per-point table in place of a band is a visible change a
user will ask about; a band over a grid is not. `--draws-out` should print the
`--draws-kind sample` hint when it writes a generated sample, so round-trips
stay one flag away. Confidence: **need you** — it changes a documented recipe.

**D2 — Is `--draws uniform` a sample or a design by default?**

- (a) Sample with `Measure::UniformBounds`: bands as today; the manifest says
  `uniform_bounds`, never `posterior`. `--draws-kind design` gives per-point
  rows.
- (b) Design: per-point rows by default, which is what sensitivity exploration
  usually wants, at the cost of changing today's output and of large tables
  (`-n 500` × time rows per quantity).

Recommendation: (a). The law is real (§3), the band is a correct Monte Carlo
estimate of its push-forward, and the new `summary.measure` field stops it being
read as a credible interval. Confidence: **leaning**.

**D3 — Move `fit predict`'s sweep onto the grid (Stage 4) now, or leave predict
folding its sweep into draw rows?**

- (a) Move it: predict builds one plan (sweep design × posterior sample), runs
  all scenarios and sweep points in one `run_job`, and deletes its private guard
  and the fresh-sink-per-sweep-point loop (`predict.rs:2193-2330`). One
  derivation of the partition key serves both verbs.
- (b) Leave it: `PartitionKey` is computed from the cell in `simulate` and from
  a loop variable in predict — the §2.2 asymmetry persists, and so does the
  second guard.

Recommendation: (a), gated on the byte-identity test in Stage 4. `2026-08-11` §4
declined this because only the scenario half of the key was reachable; Stage 3
makes the design half reachable in `simulate`, which removes that objection. The
conditioned-read routing (`conditioned_here`, `predict.rs:934-943`) becomes a
function of the cell's `PartitionKey` (conditioned iff scenario is the
conditioned arm and design is unit) — the same three conditions it checks today.
Confidence: **leaning**.

**D4 — Should design points share common random numbers?**

- (a) Keep today's seeds (§5.3): `batch run` and `fit predict` share seeds
  across design points; a design read from a file in `simulate` does not.
  Byte-neutral.
- (b) Key the seed on `(sample index, rep)` only, so every design axis shares
  seeds across its points, as `batch run` and predict already do. Makes paired
  comparisons across design points low-variance everywhere; changes the
  trajectories (and therefore the run ids) of every `simulate --draws FILE` run
  with more than one row.

Recommendation: (a) in this proposal; file (b) as its own issue, because it is a
deliberate re-key with its own justification and should not ride in on a type
change. Confidence: **solid** on not bundling it; **need you** on whether (b) is
wanted at all.

**D5 — What column prefix does a non-sweep design coordinate use?**

- (a) `sweep:<param>` for every design origin: no change to predict headers or
  the manifest `sweep` object; the prefix means "design coordinate" in the docs.
- (b) `design:<param>` everywhere: the name is accurate for a file or a
  generated design, and pre-1.0 renames are cheap (`VERSIONING.md`), but it
  changes `fit predict`'s headers and the manifest key, which
  `fit_predict_sweep.rs:191` pins.

Recommendation: (a). Confidence: **leaning**.

## 11. Staged implementation

Each stage lands alone with a green `make test`. Stages 0-2 and 4 are
byte-neutral; 3 and 5 carry the user-visible changes in §9.

**Stage 0 — one constructor for `CellSpec`.** `CasSink::predict_cells`
(`batch.rs:1265-1300`) calls `plan_grid` on the job instead of re-looping; the
two test fixtures go through a `#[cfg(test)]` constructor. _Test:_ for a sweep ×
two scenarios × three seeds manifest, the dry-run hit/miss classification and
every predicted `run_id` equal those of the real run (a new CLI test running
`--dry-run`, then the run, then `--dry-run` again, asserting all-miss then
all-hit with identical leaf paths).

**Stage 1 — `ParamPlan` and `CellCoords`.** Replace `ParamSource`; split
`point_overrides`; add `seed_index`; migrate the §5.6 consumers mechanically,
keeping every behaviour. _Tests:_ `tests/determinism_pin.rs` unchanged and
green; a new unit test in `engine` that plans the four current job shapes
(point, generated draws, draws file, sweep with explicit seeds) and asserts each
cell's `process_seed` equals `process_seed_for` evaluated with the old
`point_idx` — the old formula kept in the test as the oracle; an A/B over
`simulate --draws uniform -n 5 --scenario a,b` and a `batch run` sweep, diffing
every artifact including `run.json`.

**Stage 2 — the collision guard keys on the plan.** §6.1 replaces
`check_explicit_draws_scenario_collision`; the in-flight gh#572 guard is
superseded or rebased onto it. Predict's guard stays until Stage 4. _Tests:_ the
gh#572 red test (a `batch run` sweep and a `[design]` block each colliding with
a scenario → non-zero exit naming both sources); `--draws FILE` collision still
refused; `--draws uniform --scenario S` with S pinning a drawn column → exit 0
and one stderr line naming the shadowed column. Mutation check: make the guard
treat the design axis as a sample axis and confirm the batch test goes red.

**Stage 3 — design partitions and the `summary` block in `simulate`.**
`--draws-kind`, `Measure` construction at the `simulate` boundary,
`ParamPlan::mode`, `by_partition`, `DesignCoords::of`, the manifest `summary`.
_Tests:_ `--draws FILE` (three rows) `--quantities-out` → a scalar quantity file
with three rows, `sweep:<param>` columns, point-mode header; the same with
`--draws-kind sample` → one band row with `n_draws = 3`; `--draws uniform -n 5`
→ `summary.measure == "uniform_bounds"`, `over == ["sample"]`; `--replicates 4`
with no draws → `measure == "process"`. Mutation check: key `by_partition` on
the scenario alone and confirm the three-row assertion fails on the row count,
not the header.

**Stage 4 — `fit predict` on the grid (if D3 is (a)).** One plan, one `run_job`
for all scenarios and sweep points; `merge_cell` keys on `PartitionKey`;
predict's guard deleted; `conditioned_here` reads the key. _Tests:_
byte-identity A/B of every predict artifact (predictive TSVs, quantities TSVs,
all manifests) for `--sweep k=… --scenario a,b` with and without a conditioned
fit, same seed; the existing header pin (`fit_predict_sweep.rs:191`); the
existing predict sweep-collision test, now passing through the shared guard.

**Stage 5 — provenance.** `ValueSource::{Sample, DesignPoint}`; the per-cell
manifest records `requested` alongside effective values for design coordinates
(§12). _Test:_ one fit leaf and one `sim` leaf before and after — `run_id`
unchanged, the provenance tag changed — which settles the unverified question in
§9.

## 12. Requested and effective values for a future scale × sweep composition

Today a scenario that scales a swept parameter is refused (§6.1), so a design
coordinate's requested value always equals the value the cell ran. If a later
proposal admits composition — sweep `mu ∈ {1, 2, 3}` under
`scale = { mu = 2.0 }` running `{2, 4, 6}` — the representation is already
decided here:

- **Identity hashes only effective values.** It does already (§3). Two cells
  whose effective values coincide under different requested coordinates share a
  `run_id` and are one computation in the store; that is correct.
- **The partition key is the requested coordinate.** It is the index of the axis
  the user chose (`DesignCoord::Point { idx }`), not a value.
- **A column denotes one thing.** `sweep:mu` keeps meaning the requested grid
  value; the effective value, where it differs, is reported under a separately
  named column and in the manifest's per-cell `requested`/`effective` pair
  (Stage 5 writes both; they are equal until composition exists). The denotation
  condition of `2026-08-11` §1.1 is then satisfied by naming, not by forbidding.

Admitting composition is out of scope; this section fixes only where its
metadata would go, so Stage 5 records the right shape once.
