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

## 1. The rule: which reductions may cross which axes

A run grid is a product of axes that mean two different things:

```
cells  =  scenario × sweep-point  ×  draw × replicate × seed
          ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
          PARTITION                   BAND
```

**Partition axes** index distinct objects. A baseline and a 50 %-coverage arm
are two different counterfactual worlds; a sweep point at `R0 = 2` and one at
`R0 = 4` are two different models.

**Band axes** index repeated sampling of one object — a posterior draw, a
stochastic replicate, a seed.

The rule is not "never reduce across a partition axis", which would be wrong:
`fit/contrasts.rs` reduces across the scenario axis on purpose, pairing draw _i_
of one arm with draw _i_ of the other under a shared `arm_seed`
(`contrasts.rs:307`), and produces cases-averted — the canonical policy output.
The correct statement distinguishes two kinds of reduction:

- A **marginal** reduction — a quantile, a mean, anything treating its inputs as
  exchangeable samples of one thing — **must not cross a partition axis**. Its
  output is a statement about a single object, and there is no single object.
- A **paired** reduction — one that matches cells across the partition axis by
  their band coordinate and reduces the _matched differences_ — **is exactly
  what a partition axis is for**. Contrasts is this, and it is correct.

The quantity band is a marginal reduction. Pooling scenarios into it is the
defect. Contrasts is the proof that the distinction is a real property of the
reduction rather than a blanket prohibition on the axis.

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
    // … quantities / banded / out_dir / compiled / eval / calendar unchanged …
    /// Per scenario: that scenario's draws and its own time axis. Insertion
    /// order is the engine's canonical order (scenario outermost, `plan_grid`
    /// at `engine.rs:168`), so the rendered file lists scenarios in CLI order.
    by_scenario: IndexMap<String, ScenarioQuant>,
}

struct ScenarioQuant {
    draws: Vec<Vec<sim::quantity::QuantityResult>>,
    /// Per SCENARIO, not per run — see §2.2.
    times: Vec<f64>,
}

fn push_cell(&mut self, cell: &engine::CellResult) -> Result<(), String> {
    // … evaluator build and per-cell param resolution unchanged …
    let acc = self.by_scenario
        .entry(cell.spec.scenario.name().to_string())
        .or_default();
    if acc.times.is_empty() {
        acc.times = cell.traj.snapshots.iter().map(|s| s.t).collect();
    }
    acc.draws.push(results);
    Ok(())
}
```

This is the whole of the gh#562 fix. It is worth being explicit that no
additional type is required for the guarantee, because the guarantee is a
property of the _API shape_ — an accumulator whose only input is a whole cell —
and not of a key type. An accumulator exposing `push(key, draws)` would be
strictly weaker no matter how well the key were typed, because the caller could
pass one key for every cell.

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

### 3.3 Knowing the scenario and printing it are separate decisions

`push_design_header_cols` currently emits the column iff `coords.scenario` is
`Some`. With the `Option` gone, emission becomes an explicit flag on the render
call rather than a side effect of the coordinate being populated:

```rust
pub(crate) enum ScenarioCol { Always, WhenMultiple }
```

Keeping these separate matters: the renderer must _always_ know which band it is
rendering (that is the correctness property), while whether the column appears
in the header is a formatting choice with compatibility consequences. Conflating
them is what let "no scenario column" and "no partition identity" be the same
value. Resolved to `Always` — see decision 1.

### 3.4 `banded` keyed on the param source

```rust
let banded = !matches!(job.source, ParamSource::Point { replicates: 1 });
```

This implements the rule already documented at `quantity_output.rs:23`. It fixes
2.3 without introducing the regression a cell-count test would: `--draws
<file>`
with a single row must stay banded (`simulate_quantities.rs:190-192` pins the
intent), which a `draws.len() > 1` predicate would silently flip to point mode.

`Mode` remains one value for the whole render — a stacked file has one header
per quantity, so a mixed set would emit two schemas under one header. It is
derived from the param source, which is a property of the run, so no mixture can
arise.

### 3.5 The stacking loop moves to a neutral module

`fit predict` holds an inline "render each group, drop the repeated header,
stack the bodies, merge the manifests" loop (`predict.rs:1342-1357`), and
`simulate` now needs it verbatim. It moves beside the renderer as:

```rust
pub(crate) fn render_stacked<'a>(
    quantities: &[ir::quantity::Quantity],
    groups: impl Iterator<Item = (DesignCoords<'a>, &'a [Vec<QuantityResult>], &'a [f64])>,
    mode: Mode,
    scenario_col: ScenarioCol,
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

The generalization is revisited when `batch run --quantities-out` is actually
written — §9.

## 5. Migration

**Increment 1 — move the quantile primitives to `cli/src/quantile.rs`.** `band`,
`quantile`, `fmt_time`, `fmt_value`, `QUANTILE_LEVELS` and `quantile`'s four
unit tests (`predict.rs:2161`, `:2173`, `:2181`, `:2189`) move out of
`fit/predict.rs`; `quantity_output.rs:20` and `contrasts.rs:48` update their
imports rather than going through a re-export. Pure move, byte-identical.

**Increment 2 — `DesignCoords.scenario: &str`, `none()` deleted, `ScenarioCol`
added, `render_stacked` lifted; `fit predict` routed through it.** Predict
passes `ScenarioCol::Always` (its current behaviour). Expected output change:
exactly one — predict's `quantities.json` gains the `calendar` block it
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

**7.3 `banded` excludes the scenario factor.** Two scenarios, fixed params, no
`--draws`, `--quantities-out`: assert point mode (a bare `value` column), not a
two-point band. Requires increment 3, since the refusal short-circuits it today.

**7.4 `--draws <file>` with one row stays banded.** Guards the regression a
cell-count predicate would introduce (§3.4).

**7.5 `fit predict` manifest carries `calendar`.** Red before increment 2.

**7.6 A/B byte-identity.** Increments 1 and 2 (apart from the calendar block):
same command, same seed, diff every artifact; only timestamps and mtimes may
differ. The A/B must include a multi-scenario predict run — an A/B over
single-scenario runs exercises none of the changed paths.

Test churn to expect from decision 1, measured rather than estimated: the
`scalar()` header helpers at `quantities_surface.rs:74` and
`observations_quantity.rs:74` route roughly forty assertions between them;
`quantities_surface.rs:123-125`, `:130`, `:133`; `simulate_quantities.rs:131`,
`:135`, `:144`, `:153` (point) and `:221`, `:231`, `:241` (banded). Three files.

## 8. Decisions

1. **Always emit the `scenario` column, or only when more than one scenario?**
   **Always** (`ScenarioCol::Always` on both verbs). Conditional emission makes
   a file's column layout depend on how many scenarios the model happens to
   declare, so adding a second scenario silently reshapes an existing consumer's
   input — the failure mode this whole proposal is about, in a smaller key. It
   also aligns `simulate` with `fit predict`, the artifact its output is most
   often compared against. The cost is the test churn measured in §7. This is
   the one call the gh#562 handoff explicitly reserved for the maintainer; it is
   a one-line flip to `WhenMultiple` if preferred.

   Noted and not fixed here: the trajectory TSV emits `scenario` only when
   `n_scenarios > 1` (`main.rs:2037`, `:2048`), so it retains the instability
   this decision removes from the quantity files. Follow-up in §9.

2. **A general partition-coordinate type?** **No** — §4. Revisited when
   `batch run --quantities-out` exists.

3. **Per-scenario `times`?** **Yes** — §2.2. A run-global axis becomes
   silent-wrong the moment gh#561 lands, and per-scenario is the only form that
   needs no second change later.

4. **Unify the accumulators?** **No** — §3.6. Unify the stacking only.

5. **Keep the multi-scenario refusal as a guard after the fix?** **No.** A
   refusal that can no longer fire is dead code, and keeping it would imply the
   pooling is still expressible.

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
  re-examined against a real second partition axis. Two things must be solved
  there and are not solved here: predict-style sweep coordinates that do not
  live in the cell, and CAS cache hits, which `run_job` filters out before
  simulation (`engine.rs:209-219`) so a warm run would band over only the missed
  cells.
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
