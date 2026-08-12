# Per-scenario banding in `simulate --quantities-out`

- Date: 2026-08-11
- Status: proposed
- Fixes: gh#562
- Unblocks: gh#561 (ordered strictly after)

## Summary

`simulate --quantities-out` reduces every cell of the run grid into one quantile
band, including cells belonging to different scenarios. A baseline and its
counterfactual are averaged into a single ribbon that describes neither, in a
file whose shape is indistinguishable from a correct posterior band (gh#562).

This proposal fixes it by keying the accumulator on the cell's scenario, gives
each scenario its own time axis (the prerequisite for gh#561), corrects the
`banded` predicate to stop counting scenarios as uncertainty, and deletes
`DesignCoords::none()` — the value that made the pooling expressible.

It deliberately does **not** introduce a general partition-coordinate type. §4
records that design and why it was rejected.

Scope is `rust/crates/cli` only. No IR change, no `ir/VERSION` bump, no golden
churn, no run-identity move.

## 1. The rule: design coordinates condition, sampling coordinates marginalize

A run grid is a product of axes that play two different roles:

```
cells  =  scenario × sweep-point  ×  draw × replicate × seed
          ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
          DESIGN (conditioning)      SAMPLING
```

**Design coordinates** — scenario, sweep point — say _which world was
simulated_. A baseline and a 50 %-coverage arm are two different counterfactual
worlds; a sweep point at `R0 = 2` and one at `R0 = 4` are two different models.

**Sampling coordinates** — parameter draw, stochastic replicate, process/
observation seed — index repeated sampling _within_ one world.

A quantity output is therefore an empirical approximation to

```
p(Q | scenario, sweep)
```

and the summary reduces over the sampling coordinates while holding the design
coordinates fixed. The bug computes something closer to `Σ_s p(Q | s) · w_s`
without the user ever having defined a measure `w_s` over scenarios — an
implicit, uniform, and unstated weighting over counterfactual worlds.

That gives the rule its sharpest form:

> **Never implicitly marginalize a design coordinate.** Sampling coordinates may
> be marginalized freely. A design coordinate may be eliminated only by an
> explicit measure over its levels, or by a contrast operator applied across
> them.

The second escape is the one already in the codebase, and it is worth being
precise about why it is legitimate. `fit/contrasts.rs` crosses the scenario axis
deliberately, pairing draw _i_ of one arm with draw _i_ of the other under a
shared `arm_seed` (`contrasts.rs:307`):

```
D_i = Q_{i,baseline} − Q_{i,intervention}
```

This is not a different flavour of marginalization. The contrast **constructs a
new random variable first**, holding the pairing coordinate _i_ fixed
throughout; only afterwards are the sampling coordinates marginalized to give
quantiles of `D`. The design coordinate is consumed by the operator, never
averaged over.

So the pipeline in general is: **condition on design axes → construct the
estimand or contrast → summarize over sampling axes.** The quantity band is the
degenerate case where the estimand is `Q` itself. Pooling scenarios into it
skips the first step.

## 2. Where `simulate` loses it

All references against `c4ccf985`.

`SimQuantities` (`main.rs:1535`) holds one flat vector over cells and one time
axis for the whole run:

```rust
draws: Vec<Vec<sim::quantity::QuantityResult>>,  // one entry per CELL
times: Vec<f64>,                                 // from whichever cell came first
```

`push_cell` (`main.rs:1615`) appends unconditionally (`:1656`) and captures
`times` from the first cell (`:1653-1655`). The render call passes
`DesignCoords::none()` (`main.rs:1436-1440`), so `render_quantities` reduces the
whole vector to one set of quantiles.

`fit predict` is unaffected: its sink derives the scenario from the cell inside
`merge_cell` (`predict.rs:800`) and commits into a per-scenario accumulator
(`:846`, `by_scenario` at `:748`), rendering one band per scenario (`:1328`).

Three consequences follow from the same root, and all three are fixed here.

**2.1 Pooled quantiles (the reported bug).** Two scenarios × ten draws give
`n_draws = 20` and quantiles over a bimodal mixture, with no `scenario` column.
Invisible because a mixture has perfectly well-defined percentiles: nothing is
`NaN`, nothing is out of range, and the band is _wider_ than either arm, which
reads as honest uncertainty.

**2.2 One time axis for the whole run.** Safe only while every cell shares an
output cadence. gh#561 ends that: per-scenario `simulate { to = … }` is parsed
into `Preset::t_end` (`ir/model.rs:122`) and currently dropped at the
preset→`ResolvedEntry` conversion (`main.rs:1760-1766`; `ResolvedEntry` has no
`t_end`, `batch.rs:305-311`). Once threaded, a scenario ending 30 September
would render against a 13 August axis. Per-scenario `times` is a correctness
prerequisite, which is what forces the ordering.

**2.3 `banded` counts cells, not uncertainty.**

```rust
let banded = draws_path.is_some() || total_runs > 1;      // main.rs:1292
let total_runs = n_draws * replicates * n_scenarios;      // main.rs:1079
```

The scenario factor sits inside a predicate that should count only uncertainty.
The rule is already written down two places — "keyed by the param-source kind,
never the cell count alone" (`quantity_output.rs:23`, `main.rs:1539-1540`) — and
the predicate does not implement it.

A guard currently refuses multi-scenario `--quantities-out` outright
(`main.rs:1273-1296`), so 2.1 and 2.3 are unreachable via `--scenario` today.
That guard is a stopgap and is removed by this change.

## 3. The design

### 3.1 The guarantee is encapsulation, and it already exists

The property that prevents pooling is: **the accumulator derives its own key
from the cell; no caller can supply one.**
`SimQuantities::push_cell(&mut self,
cell: &engine::CellResult)` already has
exactly that signature. It simply ignores `cell.spec.scenario`.
`PredictiveSink::merge_cell` has the same signature and does not.

So the fix is to use the key that is already in scope, not to add a type:

```rust
struct SimQuantities {
    // … quantities / out_dir / compiled / eval / calendar unchanged …
    /// Point vs banded, replacing the former `banded: bool` so there is one
    /// representation of the state rather than two (§3.4).
    mode: Mode,
    /// Per scenario: that scenario's draws and its own time axis. Insertion
    /// order is the engine's canonical order (scenario outermost, `plan_grid`
    /// at `engine.rs:168`); the merge phase is strictly ordered even under
    /// Rayon (`engine.rs:245-248`), so this is deterministic — pinned by
    /// test 7.6 rather than left as an assumption.
    by_scenario: IndexMap<String, ScenarioQuant>,
}

struct ScenarioQuant {
    draws: Vec<Vec<sim::quantity::QuantityResult>>,
    /// Per SCENARIO, not per run — see §2.2.
    times: Vec<f64>,
}

fn push_cell(&mut self, cell: &engine::CellResult) -> Result<(), String> {
    // … evaluator build and per-cell param resolution unchanged …
    let times: Vec<f64> = cell.traj.snapshots.iter().map(|s| s.t).collect();
    let scenario = cell.spec.scenario.name();
    let acc = self.by_scenario.entry(scenario.to_string()).or_default();
    if acc.draws.is_empty() {
        acc.times = times;
    } else if !time_grids_match(&acc.times, &times) {
        // Later cells PROVE compatibility rather than being assumed into the
        // first cell's axis. Without this, a future feature that lets cells of
        // ONE scenario differ in cadence would silently reinstate the original
        // bug at smaller scale — first-wins, no diagnostic.
        return Err(format!(
            "scenario '{scenario}': cell {} has {} output times but this \
             scenario's earlier cells have {}. Quantity cells within a \
             scenario must share a time grid.",
            cell.spec.run_idx, times.len(), acc.times.len()
        ));
    }
    acc.draws.push(results);
    Ok(())
}
```

`time_grids_match` compares length and then element-wise within the module's
existing output-time tolerance (`OUTPUT_EPS`, `sim/src/schedule.rs`) rather than
by exact float equality, since the grid is derived arithmetically per cell.

Guarding on `acc.draws.is_empty()` rather than `acc.times.is_empty()` matters: a
legitimately empty time grid (a model whose output window admits no rows) would
otherwise re-enter the first-wins branch on every cell and never check.

This is the whole of the gh#562 fix. No additional type is required for the
guarantee, because the guarantee is a property of the _API shape_ — an
accumulator whose only input is a whole cell — not of a key type. A `Band` type
can prove a band is internally well-formed; it cannot prove observations were
assigned to the _correct_ band. That invariant belongs at the assignment
operation, and `push_cell` is the assignment operation. An accumulator exposing
`push(key, draws)` is strictly weaker however well the key is typed, because the
caller can consistently supply the wrong one.

### 3.2 `DesignCoords::none()` is deleted

Every quantity render is now per-scenario, on both verbs, so the scenario is
always known at the renderer. The `Option` therefore represents a state that no
longer occurs:

```rust
pub(crate) struct DesignCoords<'a> {
    pub scenario: &'a str,              // was: Option<&'a str>
    pub sweep: &'a [(String, f64)],
}
// `DesignCoords::none()` removed.
```

This is the small, real type win available here: the value that expressed
"render these cells with no partition identity" — the value the bug was written
in terms of — stops being constructible. `DesignCoords` stays `Copy` and the six
helpers that take it by value (`quantity_header:177`, `point_header:199`,
`push_design_header_cols:152`, `push_design_row_cells:163`,
`render_banded_leaf:401`, `render_point_leaf:483`) are unchanged apart from the
field type.

### 3.3 The scenario column is unconditional

`push_design_header_cols` currently emits the column iff `coords.scenario` is
`Some`. With the `Option` gone, quantity output **always** carries a `scenario`
column, on both verbs.

No `ScenarioCol::{Always, WhenMultiple}` toggle is introduced. Conditional
emission is a schema policy we have decided is wrong (decision 1), and making it
a first-class selectable state would keep it conveniently reintroducible — the
same objection that retires `DesignCoords::none()`. Do not represent states you
have decided should not occur. If backwards compatibility ever becomes a real
requirement, it arrives then, under a name that says it is a compatibility shim.

The separation that does matter is retained by construction: the renderer
_always knows_ which design cell it is rendering (the correctness property, now
carried by a non-optional field), and printing follows from that rather than
determining it.

### 3.4 `Mode` derived from the param source, replacing `banded: bool`

`SimQuantities.banded: bool` and `quantity_output::Mode` are two representations
of one state. The `bool` goes; `Mode` is stored directly and derived once:

```rust
let mode = match job.source {
    ParamSource::Point { replicates: 1 } => Mode::Point,
    _ => Mode::Banded,
};
```

This implements the rule already documented at `quantity_output.rs:23` ("keyed
by the param-source kind, never the cell count alone"). It fixes 2.3 without the
regression a cell-count predicate would introduce: `--draws <file>` with a
single row must stay banded (`simulate_quantities.rs:190-192` pins the intent),
which `draws.len() > 1` would silently flip to point mode.

`Mode` is one value for the whole render — a stacked file has one header per
quantity, so a mixed set would emit two schemas under one header. It is derived
from the param source, a property of the run, so no mixture can arise.

**This is the weakest part of the design and is deliberately not fixed here.**
`ParamSource` is not the property we actually care about. The real question is
whether a collection of points carries _sampling semantics_ — whether a quantile
over it means anything. `--draws posterior.tsv` and
`--draws sensitivity-grid.tsv` have identical types and completely different
interpretations; `--draws uniform` bands today, though quantiles of a
space-filling sample describe the sampling scheme rather than a belief.
`ParamSource` is a proxy that is wrong in both directions. Recorded as a
follow-up (§9) rather than solved inside a bug fix, because the honest fix is a
model-level declaration of sampling semantics, not a better discriminant over
CLI flags.

### 3.5 The stacking loop moves to a neutral module

`fit predict` holds an inline "render each group, drop the repeated header,
stack the bodies, merge the manifests" loop (`predict.rs:1342-1357`), and
`simulate` now needs it verbatim. It moves beside the renderer as:

```rust
pub(crate) fn render_stacked<'a>(
    quantities: &[ir::quantity::Quantity],
    groups: impl Iterator<Item = (DesignCoords<'a>, &'a [Vec<QuantityResult>], &'a [f64])>,
    mode: Mode,
    calendar: &io::CalendarMeta,
) -> Result<(Vec<(String, String)>, String), String>
```

An iterator rather than a collection, because `predict` yields groups from a
sink that is rebuilt per sweep point and cannot outlive the loop, while
`simulate` yields them from one long-lived map. Both call it; neither owns it.

Note the caller supplies the coordinates here — which is safe precisely because
it is a _rendering_ seam, not an accumulation seam. The pooling was possible
because the accumulator merged cells; a renderer handed already-separated groups
cannot re-merge them.

### 3.6 What is not unified

`predict`'s `ScenarioAccum` (`predict.rs:717-726`) also carries `samples` — the
predictive `y_rep` per `(leaf, time)` — which `simulate` has no analogue for,
and the two accumulate over different lifetimes (`predict` over draws it already
holds, `simulate` over cells as they stream). They stay distinct. The shared,
bug-prone substrate is the stacking, and only that is lifted.

`band`, `fmt_time`, `fmt_value`, `quantile` and `QUANTILE_LEVELS` currently live
in `fit/predict.rs` and are imported _by_ the shared renderer
(`quantity_output.rs:20`) and by `contrasts.rs:48` — a consumer-owns-the-
substrate inversion. They move to a neutral `quantile.rs` that
`quantity_output`, `predict` and `contrasts` all depend on. Moving them _into_
`quantity_output` would relocate the inversion rather than remove it, since
`contrasts` has nothing to do with quantities.

## 4. The rejected design, and why

A general partition-coordinate type was designed and rejected. Recorded because
it is the obvious next idea and the reasons are not obvious.

The shape was: `PointKind` (`Single | Sweep | Draw`) stamped on `CellSpec` by
`plan_grid`; `BandKey { scenario, sweep }` derived from a cell by a single
`BandKey::of`; `BandSet` as the only constructor of a `Band`, which the renderer
would take instead of a `(draws, times, coords)` triple.

**It does not deliver the guarantee it is named for.** `BandSet::push(key, …)`
takes the key from the caller, so a caller passing one key for every cell pools
exactly as today. The invariant moves up a stack frame rather than into a type.
The property that actually excludes pooling is §3.1 — an accumulator whose only
input is a cell — and adding a key parameter _removes_ it.

**`BandKey::of(&CellSpec)` cannot serve `fit predict`.** Predict implements
sweeps by folding the swept parameters into the draw rows and running one
`ParamSource::Draws` job per sweep point (`predict.rs:1281-1308`); the sweep
value lives in the enclosing loop (`:1261`), never in the cell. Every predict
cell is therefore a draw, `BandKey::of` returns an empty sweep, and either the
`scenario\tsweep:k\t…` header pinned at `fit_predict_sweep.rs:230` regresses, or
— if the set spans the sweep loop — two sweep points merge into one band. That
is gh#562 reproduced in the other verb by the type introduced to prevent it.
Repairing it means a public `BandKey::new(scenario, sweep)` that predict calls,
at which point there are two constructors for "which band is this", which is the
gh#233 fork shape.

**`PointKind` is the wrong property, not merely a premature one.** It records
which CLI flag produced the points, not whether those points carry a measure
over which a quantile means anything. `--draws sensitivity_grid.tsv` — a
hand-authored design grid — is `Draw` and would band; the same grid expressed as
a batch `sweep` is a partition. `--draws uniform -n 5` bands today, and
quantiles of a space-filling sample are a property of the sampling scheme, not
of a belief. The distinction that matters for banding is measure-carrying-ness,
and `PointKind` is a proxy for it that is wrong in both directions. Naming a
discriminant for the proxy would entrench the confusion at the moment we finally
need the real property.

**And it is unreachable on every shipping path.** `ParamSource::Sweep` is
constructed only in `batch.rs` (`:703`, `:1516`); batch merges through `CasSink`
(`:1136`) and emits no quantities. After the change, `BandKey::of` would return
an empty sweep at every call site and `BandKey` would be isomorphic to `String`
— seven of seven sites, against the pragmatic line in
`.claude/rules/rust-conventions.md`, which dropped `NominalStep`/`SnapGrid` at
six of seven. `CellSpec` also has four construction sites, not one:
`batch.rs:1019` (`predict_cells`, the dry-run cache predictor) has no
`ParamSource` in scope and would have to guess the kind.

**We are also not restructuring `fit predict` to make the abstraction viable.**
Routing predict's sweep through `CellSpec` so that a single `BandKey::of` would
work is a change to a fitted-model path, motivated entirely by the elegance of
an abstraction with one currently-useful component (`scenario`) — the sweep half
is unreachable in the quantity path because batch emits no quantities. That is
architecture-driven rather than requirement-driven development. The asymmetry is
real but costs nothing until some operation must reason uniformly across both
paths, and none does. When `batch run --quantities-out` exists there will be two
real design axes and two independent clients of the concept, and that is when
the abstraction should emerge — §9.

## 5. Migration

**Increment 1 — move the quantile primitives to `cli/src/quantile.rs`.** `band`,
`quantile`, `fmt_time`, `fmt_value`, `QUANTILE_LEVELS` and `quantile`'s four
unit tests (`predict.rs:2161`, `:2173`, `:2181`, `:2189`) move out of
`fit/predict.rs`; `quantity_output.rs:20` and `contrasts.rs:48` update their
imports rather than going through a re-export. Pure move, byte-identical.

**Increment 2 — `DesignCoords.scenario: &str`, `none()` deleted,
`render_stacked` lifted; `fit predict` routed through it.** Predict already
emits the column unconditionally, so its header is unchanged. Expected output
change: exactly one — predict's `quantities.json` gains the `calendar` block it
currently drops (§6). Everything else byte-identical under A/B.

**Increment 3 — `simulate` keyed by scenario (gh#562).** `SimQuantities.draws`/
`times` become `by_scenario`; `push_cell` keys on `cell.spec.scenario.name()`;
`banded` keyed on the param source; the refusal at `main.rs:1273-1296` deleted.
Carries the red→green test and the `gh#562` commit subject.

Increment 1 can land while 2 is in review; 3 depends on 2.

## 6. A defect fixed in passing

`fit predict` builds its quantities manifest at `predict.rs:1376-1379` with
`schema` and `quantities` only and writes it to `quantities.json`
(`:1596-1600`). `render_quantities` always includes
`"calendar": calendar.to_json()` (`quantity_output.rs:380-384`), so `simulate`'s
manifest carries it and predict's does not — while predict's two _other_ sidecar
manifests do (`:1543`, `:1580`). A consumer of a predict quantities manifest
therefore cannot map `time → Date` without re-deriving `origin` and `time_unit`,
which is the one thing the calendar block exists to prevent, and it bites
hardest on dated outbreak work. Routing both verbs through `render_stacked`
fixes it structurally; pinned by test 7.5.

## 7. Tests

The defect is at a **call site** — the accumulator, and the argument handed to
the renderer — not inside a unit-testable function, so a renderer unit test
passes with the bug present. The pinning tests are end-to-end.

**7.1 gh#562 red test** (`crates/cli/tests/simulate_quantities.rs`). The fixture
now declares `baseline` and `ctrl` presets. Both presets must set _all four_
parameters: a scenario `set` wins over the draw tier (`engine.rs:406-419`), so a
`ctrl` that pins only `beta` would leave its other parameters coming from the
uniform draw while `baseline`'s are pinned — two arms that are not comparable.

```
camdl simulate sir_q.camdl --draws uniform -n 5 --seed 7 \
      --scenario baseline --scenario ctrl --quantities-out qdir
```

Assert on `qdir/quantities/prevalence.tsv`: the header leads with `scenario`;
exactly `baseline` and `ctrl` appear; **`n_draws == 5` on every row, not 10**;
each scenario contributes the same number of time rows; and `quantities.json`
carries one entry per `(quantity, scenario)`, each tagged.

Mutation check: revert `push_cell` to a flat accumulator _while keeping the
column_, confirm the test goes red on the `n_draws` assertion specifically. A
mutation that also removes the column would fail on the header assertion first
and would not prove the band is unpooled.

Note this fixture's five draws are parameter-identical within an arm (the preset
pins everything), so the band is replicate variation. That is sufficient for the
`n_draws` assertion and must not be described as parameter uncertainty.

**7.2 Per-scenario time axes.** Two scenarios whose trajectories have different
snapshot counts: assert each band renders against its own axis and neither is
truncated or padded to the other. This is the gh#561 prerequisite, testable
before gh#561 exists.

**7.2b Within-scenario time-grid mismatch is an error, not first-wins.** A unit
test on the accumulator: push two cells into the same scenario with different
grids, assert a located error naming the scenario and both lengths. This pins
§3.1's assertion; without it the check is untested and would rot into a no-op
the first time someone "simplified" it back to `if times.is_empty()`.

**7.3 `banded` excludes the scenario factor.** Two scenarios, fixed params, no
`--draws`, `--quantities-out`: assert point mode (a bare `value` column), not a
two-point band. Requires increment 3, since the refusal short-circuits it today.

**7.4 `--draws <file>` with one row stays banded.** Guards the regression a
cell-count predicate would introduce (§3.4).

**7.5 `fit predict` manifest carries `calendar`.** Red before increment 2.

**7.6 Scenario order is deterministic under `--parallel`.** `IndexMap` insertion
order decides the order scenarios appear in the output, so it depends on the
order `merge_cell` is called. That is safe today — `run_job` collects the
simulation phase into a `Vec` (Rayon's `into_par_iter().map().collect()`
preserves input order) and the merge phase then runs strictly in canonical order
(`engine.rs:222-248`) — but it is an implementation detail this design now
depends on for reproducible output. Run the same multi-scenario job at
`--parallel 1` and `--parallel 8` and assert the rendered TSVs are
byte-identical. Without this the dependency is invisible and a future change to
merge-on-completion would silently reorder decision artifacts.

**7.7 A/B byte-identity.** Increments 1 and 2 (apart from the calendar block):
same command, same seed, diff every artifact; only timestamps and mtimes may
differ. The A/B must include a multi-scenario predict run — an A/B over
single-scenario runs exercises none of the changed paths.

Test churn to expect from decision 1, measured rather than estimated: the
`scalar()` header helpers at `quantities_surface.rs:74` and
`observations_quantity.rs:74` route roughly forty assertions between them;
`quantities_surface.rs:123-125`, `:130`, `:133`; `simulate_quantities.rs:131`,
`:135`, `:144`, `:153` (point) and `:221`, `:231`, `:241` (banded). Three files.

## 8. Decisions

1. **Always emit the `scenario` column?** **Yes, unconditionally, with no toggle
   type.** Conditional emission makes a file's column layout depend on how many
   scenarios the model happens to declare, so adding a second scenario silently
   reshapes an existing consumer's input — the failure mode this whole proposal
   is about, in a smaller key. It also aligns `simulate` with `fit predict`. And
   having decided conditional emission is the wrong policy, we do not keep it as
   a selectable `ScenarioCol::WhenMultiple`: a first-class name makes a rejected
   policy conveniently reintroducible, which is the same objection that retires
   `DesignCoords::none()`. The cost is the test churn measured in §7.

   Noted and not fixed here: the trajectory TSV emits `scenario` only when
   `n_scenarios > 1` (`main.rs:2037`, `:2048`), so it retains the instability
   this decision removes from the quantity files. Follow-up in §9.

2. **A general partition-coordinate type?** **No** — §4. Revisited when
   `batch run --quantities-out` exists.

3. **Per-scenario `times`, first-wins or validated?** **Per-scenario, and
   validated** — §2.2, §3.1. First-wins would leave a smaller copy of the
   original bug shape: correct under today's assumption that cells of one
   scenario share a cadence, silently wrong the moment that assumption ends.
   Later cells prove compatibility or the run errors.

4. **Unify the accumulators?** **No** — §3.6. Unify the stacking only.

5. **Restructure `fit predict` so both verbs carry design coordinates the same
   way?** **No** — §4. Requirement-driven, not architecture-driven: the
   asymmetry costs nothing until some operation must reason uniformly across
   both paths, and none does.

6. **Keep the multi-scenario refusal as a guard after the fix?** **No.** A
   refusal that can no longer fire is dead code, and keeping it would imply the
   pooling is still expressible.

7. **Should `RunSink` express grouping?** **No.** `RunSink` is a _delivery_
   abstraction — "here is a completed cell". Grouping is an _interpretation_ of
   cells, and pushing it into the trait would couple the engine to a downstream
   statistical operation that `CasSink` has no use for. That three sinks
   accumulate differently is not evidence the trait is wrong: the differences
   are genuine (§3.6).

## 9. Follow-ups

- **gh#561** — thread `Preset::t_end` into the per-cell config. Ordered strictly
  after increment 3. Two notes carried forward: the per-cell `t_end` must enter
  the run identity, since it changes the stored trajectory — a deliberate
  re-key, and a cost this proposal does not itself incur; and the
  ragged-`ensemble.tsv` question is already answered by the format, which is
  long-form with a `scenario` column whenever `n_scenarios > 1` (`main.rs:2037`,
  `:2048`), so unequal horizons need no padding and no split. What a scenario
  `to` _shorter_ than the model window means remains gh#561's to decide.
- **`batch run --quantities-out`** — the point at which the §4 generalization is
  re-examined against a real second design axis. Two things must be solved there
  and are not solved here: predict-style sweep coordinates that do not live in
  the cell, and CAS cache hits, which `run_job` filters out before simulation
  (`engine.rs:209-219`) so a warm run would band over only the missed cells.

  The shape to reach for then is **not** `PointKind`/`BandKey` but a role on the
  axis itself, declared where the design is defined rather than inferred from a
  CLI flag:

  ```rust
  struct Axis { name: String, values: Vec<Value>, role: AxisRole }
  enum AxisRole { Condition, Sample(SamplingSemantics) }
  ```

  with contrasts operating explicitly on a `Condition` axis. The generic rule is
  then exactly §1: condition on design axes → construct the estimand or contrast
  → summarize over sampling axes. This is the conceptual endpoint; it should not
  be built before there are two real clients.

- **Sampling semantics are not `ParamSource`** (§3.4). `--draws posterior.tsv`
  and `--draws sensitivity-grid.tsv` are the same type and different objects.
  The honest fix is a model- or invocation-level declaration of whether a point
  set carries a measure, feeding `AxisRole::Sample(_)` above. Until then the
  software should stop making a stronger statistical claim than it can justify:

  - `n_draws` in the quantity TSVs actually counts _samples_, not parameter
    draws — a `Point { replicates: 5 }` run reports `n_draws = 5` with zero
    parameter draws. Rename to `n_samples`, and have the manifest carry the
    decomposition (`parameter_points`, `replicates_per_point`,
    `summary: empirical_quantiles`) so a consumer can tell what varied.
  - Reserve "credible interval" for point sets with posterior-probability
    semantics. "Quantile band" is the honest term for everything the renderer
    currently produces.

  This is exactly the plausible-but-semantically-wrong metadata that motivated
  gh#562, one level up, and it wants its own issue rather than riding along.

- **Design-coordinate honesty: scenario overrides a colliding sweep silently.**
  Parameter resolution has a documented five-tier precedence
  (`camdl-run-spec.md` §1.3, implemented in `params_resolver.rs`) in which a
  scenario (tier 4) beats a draw/sweep point (tier 3.5). Two of the three verbs
  guard the resulting ambiguity via `scenario_param_footprint`: an explicit
  `--draws` file colliding with a scenario is a hard error
  (`engine.rs:357-402`), and `fit predict` rejects a scenario×sweep clash
  (`predict.rs:1065`). **`batch run` has neither guard**, and it is the only
  verb with real sweeps. A manifest with `[sweep] beta = [0.3, 0.5, 0.7]` and a
  scenario setting `beta = 0.2` runs three cells that are byte-identical
  trajectories, while the dry-run reports three sweep points, the CAS leaves are
  stored under `beta_0.3-…`/`beta_0.5-…`/`beta_0.7-…`, and each carries a
  distinct `params`-level hash. The design coordinate names a value the executed
  model does not have. Own issue; the fix is to extend the existing footprint
  guard to `batch`, matching the policy the other two verbs already implement.
- **The trajectory TSV's conditional `scenario` column** (`main.rs:2037`) — same
  schema-instability argument as decision 1, different artifact, its own
  compatibility surface.
- **gh#565** (`quantities {}` cannot read a flow) — a different layer
  (`QuantitySource`, `ir/quantity.rs:67`), independent of this. One constraint
  is inherited: once time axes are per-scenario, a stacked flow column can span
  bands with different interval widths, so a cumulative-flow design must say
  whether its per-interval values need an accompanying width.
- **gh#563** (run-spec §5.5 documents a batch manifest that never parsed) — docs
  only, unrelated to this architecture. Its rewrite points modellers at
  `simulate --draws` for posterior projection with scenario forks, which is only
  truthful once increment 3 removes the refusal, so it is ordered after.
