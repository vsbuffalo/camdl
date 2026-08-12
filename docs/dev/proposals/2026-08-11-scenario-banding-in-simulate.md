# Per-scenario banding in `simulate --quantities-out`

- Date: 2026-08-11
- Status: proposed
- Fixes: gh#562
- Unblocks: gh#561 (ordered strictly after)
- Related: gh#572 (scenario×sweep overlap), gh#573 (scenario run_id collision)

## Summary

`simulate --quantities-out` reduces every cell of the run grid into one quantile
band, including cells belonging to different scenarios. A baseline and its
counterfactual are averaged into a single ribbon that describes neither, in a
file whose shape is indistinguishable from a correct posterior band (gh#562).

This proposal keys the accumulator on the cell's scenario, gives each scenario
its own validated time axis (the prerequisite for gh#561), corrects the `banded`
predicate to stop counting scenarios as uncertainty, and deletes
`DesignCoords::none()` — the value the pooling was written in terms of.

It deliberately does **not** introduce a general design-coordinate type. §4
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

**Sampling coordinates** — parameter draw, stochastic replicate, process and
observation seed — index repeated sampling _within_ one world.

A quantity output is an empirical approximation to

```
p(Q | scenario, sweep)
```

reducing over the sampling coordinates while holding the design coordinates
fixed. The bug computes something closer to `Σ_s p(Q | s) · w_s` without the
user ever having defined a measure `w_s` over scenarios — an implicit, uniform,
unstated weighting over counterfactual worlds.

> **Never implicitly marginalize a design coordinate.** Sampling coordinates may
> be marginalized freely. A design coordinate may be eliminated only by an
> explicit measure over its levels, or by a contrast operator applied across
> them.

The second escape is already in the codebase, and why it is legitimate is worth
stating precisely. `fit/contrasts.rs` crosses the scenario axis deliberately,
pairing draw _i_ of one arm with draw _i_ of the other under a shared `arm_seed`
(`contrasts.rs:307`):

```
D_i = Q_{i,baseline} − Q_{i,intervention}
```

This is not a variant of marginalization. The contrast **constructs a new random
variable first**, holding the pairing coordinate _i_ fixed throughout; only
afterwards are the sampling coordinates marginalized to give quantiles of `D`.
The design coordinate is consumed by the operator, never averaged over.

The general pipeline is therefore: **condition on design axes → construct the
estimand or contrast → summarize over sampling axes.** The quantity band is the
degenerate case where the estimand is `Q` itself, and pooling scenarios into it
skips the first step.

### 1.1 The rule applies to composition, not only to reduction

The same framing settles a second question: when may two design axes be
combined?

A coordinate is only useful if it _denotes_ — if a consumer can read
`sweep:mu = 1` and know what was simulated. Two conditions are needed, and it is
worth being explicit that the obvious one alone is not sufficient.

**Distinctness.** The map from design coordinates to resolved models must be
injective. `simulate` has one design axis today so this does not bite here, but
in `batch run` a scenario resolves at tier 4 and a sweep point at tier 3.5
(`params_resolver.rs:717-740`, `:742-783`), so a scenario `set` on a swept
parameter is a constant map: three sweep points, one model, three labels.
Relative-versus-absolute is _not_ the discriminator — `scale = { mu = 0.0 }`
compiles clean (nothing validates the factor: `ir/model.rs:118`,
`expander.ml:9919`) and is equally constant.

**Denotation.** Every emitted coordinate must equal the value it labels. This is
the condition injectivity misses, and it is why distinctness alone cannot be the
rule. Sweep `mu ∈ {1,2,3}` under a scenario `scale = { mu = 2.0 }` composes to
{2,4,6} — perfectly injective — but the emitted coordinate is the _grid_ value:
`push_design_row_cells` writes it verbatim (`quantity_output.rs:167-169`) under
a `sweep:mu` header (`:156`). A consumer joining on `sweep:mu = 1` gets a model
that ran `mu = 2`. Distinct, and still a lie.

Requiring both collapses to a simple, checkable rule: **the scenario's parameter
footprint and the swept parameter set must be disjoint.** That is exactly what
`fit predict` already enforces (`predict.rs:1062-1086` via
`scenario_param_footprint`), so the guard is not a conservative stopgap awaiting
a cleverer criterion — it is the correct rule, and a distinctness-only test
would wrongly license relaxing it.

Filed as gh#572 to apply it in all three verbs. Not implemented here; different
verbs, different code paths.

## 2. Where `simulate` loses it

All references against `c4ccf985`.

`SimQuantities` (`main.rs:1540`) holds one flat vector over cells and one time
axis for the whole run:

```rust
draws: Vec<Vec<sim::quantity::QuantityResult>>,  // one entry per CELL
times: Vec<f64>,                                 // from whichever cell came first
```

`push_cell` (`main.rs:1620`) appends unconditionally and captures `times` from
the first cell. The render call passes `DesignCoords::none()` (`main.rs:1443`),
so `render_quantities` reduces the whole vector to one set of quantiles.

`fit predict` is unaffected: its sink derives the scenario from the cell inside
`merge_cell` (`predict.rs:800`) and commits into a per-scenario accumulator
(`:846`, `by_scenario` at `:748`), rendering one band per scenario (`:1329`).

Four consequences follow from the same root; all four are addressed here.

**2.1 Pooled quantiles (the reported bug).** Two scenarios × ten draws give
`n_draws = 20` and quantiles over a bimodal mixture, with no `scenario` column.
Invisible because a mixture has well-defined percentiles: nothing is `NaN`,
nothing is out of range, and the band is _wider_ than either arm, which reads as
honest uncertainty.

**2.2 One time axis for the whole run.** Safe only while every cell shares an
output cadence. gh#561 ends that: per-scenario `simulate { to = … }` is parsed
into `Preset::t_end` (`ir/model.rs:122`) and dropped at the
preset→`ResolvedEntry` conversion (`main.rs:1750-1766`; `ResolvedEntry` has no
`t_end`, `batch.rs:305-311`). Once threaded, a scenario ending 30 September
would render against a 13 August axis. Per-scenario `times` is a correctness
prerequisite, which is what forces the ordering.

**2.3 `banded` counts cells, not uncertainty.**

```rust
let banded = draws_path.is_some() || total_runs > 1;      // main.rs:1297
let total_runs = n_draws * replicates * n_scenarios;      // main.rs:1079
```

The scenario factor sits inside a predicate that should count only uncertainty.
The rule is written down in two places — "keyed by the param-source kind, never
the cell count alone" (`quantity_output.rs:23`, `main.rs:1544-1545`) — and the
predicate does not implement it.

**2.4 The no-scenario case is encoded as a scenario with a borrowed name.**
`plan_grid` iterates a `Vec<ScenarioRef>` and needs a non-empty list to emit
cells, so "no scenario axis" is represented as the _presence of a distinguished
member_: simulate synthesizes `ScenarioRef::Inline { name: "baseline" }`
(`main.rs:1116`), and `effective_scenarios` does the same for the empty case
(`engine.rs:338-344`). The tell that this is scaffolding is
`resolve_scenario_ref`'s special case (`sim_job.rs:333-341`) making `baseline`
and `fitted` resolve "even when the model declares no preset by that name" — a
branch whose only job is to make the fabricated name resolve.

The name is drawn from the user's own namespace, and the book teaches users to
use it: _"Scenarios are a separate mechanism, walled off from inference… so
researchers can package named configurations"_ (`guide/scenarios.qmd`, "Baseline
scenario vs default parameter values"). Since camdl deliberately has no model
parameter defaults, a preset named `baseline` is an ordinary scenario supplying
parameters like any other — nothing privileges it. So the runtime borrows a name
meaning "a declared scenario" to mean "no scenario at all."

Measured, on a model with a real `baseline` preset that sets `mu = 0.05` and a
params file setting `mu = 0.5`:

```
A: no --scenario           S(20)=0     scenario label=baseline  hash=33233b72  run_id=a339538f
B: --scenario baseline     S(20)=368   scenario label=baseline  hash=562f37e7  run_id=287330f4
```

Identity is sound — the hashes and run_ids differ. Only the human-readable
**label** is ambiguous, and today it appears solely in the leaf path, where a
directory name is not a key. Emitting it as a data column would promote an
ambiguous label into something consumers group and join on. §3.3 handles it.

A guard currently refuses multi-scenario `--quantities-out` outright
(`main.rs:1278-1296`), so 2.1 and 2.3 are unreachable via `--scenario` today.
That guard is a stopgap and is removed by this change.

## 3. The design

### 3.1 The guarantee is encapsulation, and it already exists

The property that prevents pooling is: **the accumulator derives its own key
from the cell; no caller can supply one.**
`SimQuantities::push_cell(&mut self, cell:
&engine::CellResult)` already has
that signature and simply ignores `cell.spec.scenario`.
`PredictiveSink::merge_cell` has the same signature and does not.

```rust
struct SimQuantities {
    // … quantities / out_dir / compiled / eval / calendar unchanged …
    /// Point vs banded, replacing the former `banded: bool` so there is one
    /// representation of the state rather than two (§3.4).
    mode: Mode,
    /// Whether this run HAS a scenario axis (any `--scenario` was passed).
    /// A fact about the run, fixed at construction — not a formatting policy
    /// (§3.3).
    scenario_axis: bool,
    /// Per scenario: that scenario's draws and its own time axis. Insertion
    /// order is the engine's canonical order (scenario outermost, `plan_grid`
    /// at `engine.rs:167`); the merge phase runs strictly in order even under
    /// Rayon (`engine.rs:243-247`), so this is deterministic — pinned by
    /// test 7.6 rather than assumed.
    by_scenario: IndexMap<String, ScenarioQuant>,
}

#[derive(Default)]
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
        // first cell's axis. Without this, a future feature letting cells of ONE
        // scenario differ in cadence silently reinstates the original bug at
        // smaller scale — first-wins, no diagnostic.
        return Err(format!(
            "scenario '{scenario}': cell {} has {} output times but this \
             scenario's earlier cells have {}. Quantity cells within a scenario \
             must share a time grid.",
            cell.spec.run_idx, times.len(), acc.times.len()
        ));
    }
    acc.draws.push(results);
    Ok(())
}
```

`time_grids_match` compares length, then element-wise with the **relative**
tolerance form already used at `chain_binomial.rs:245`
(`OUTPUT_EPS * t.abs().max(1.0)`), not bare `OUTPUT_EPS`. The bare constant is
an absolute `1e-12` (`schedule.rs:163`); on a calendar-anchored model at t ≈ 1e5
days one ulp exceeds it, so an absolute comparison degenerates to exact equality
and would reject a legitimately-equal grid computed by a different route.

Guarding on `acc.draws.is_empty()` rather than `acc.times.is_empty()` matters: a
legitimately empty grid (an output window admitting no rows) would otherwise
re-enter the first-wins branch on every cell and never check.

This is the whole of the gh#562 fix. No additional type is required, because the
guarantee is a property of the _API shape_ — an accumulator whose only input is
a whole cell — not of a key type. A `Band` type can prove a band is internally
well-formed; it cannot prove observations were assigned to the _correct_ band.
That invariant belongs at the assignment operation, and `push_cell` is the
assignment operation. An accumulator exposing `push(key, draws)` is strictly
weaker however well the key is typed, because the caller can consistently supply
the wrong one.

### 3.2 `DesignCoords::none()` is deleted

Every render is now per-scenario on both verbs, so the accumulator key — and
therefore the scenario name — is always known at the renderer:

```rust
pub(crate) struct DesignCoords<'a> {
    pub scenario: &'a str,              // was: Option<&'a str>
    pub sweep: &'a [(String, f64)],
}
// `DesignCoords::none()` removed.
```

The value that expressed "render these cells with no design identity" — the
value the bug was written in terms of — stops being constructible.
`DesignCoords` stays `Copy`, and the six helpers taking it by value
(`quantity_header:177`, `point_header:199`, `push_design_header_cols:152`,
`push_design_row_cells:163`, `render_banded_leaf:401`, `render_point_leaf:483`)
change only in the field type.

### 3.3 The `scenario` column exists iff a scenario axis exists

Emission is driven by `SimQuantities.scenario_axis`, set once at construction
from `!a.scenarios.is_empty()` alongside `mode`.

- no `--scenario` → no scenario axis → **no column**, and no label to fabricate;
- any `--scenario` → **column always**, whether one scenario or ten.

This is deliberately _not_ the `n_scenarios > 1` rule. That one is unstable in
the way that matters: adding a second scenario to a model would reshape an
existing consumer's file, which is this proposal's own failure mode in a smaller
key. Under the rule above, 1 and N behave identically; the only transition is 0
→ 1, which is a genuine change in the experiment rather than a schema wobble.

Critically, this keeps the §2.4 fabricated label out of user-facing data without
reintroducing an `Option`. `DesignCoords.scenario` stays non-optional — we
always know the key — and `scenario_axis` is a **fact about the run**, not a
formatting policy. That distinction is what made a
`ScenarioCol::{Always, WhenMultiple}` toggle objectionable: it would have made a
rejected schema policy conveniently selectable, the same objection that retires
`DesignCoords::none()`.

**The flag must reach `render_quantities`, not only the shared stacker.**
Emission is decided at three sites, all currently keyed on
`coords.scenario.is_some()`: `push_design_header_cols`
(`quantity_output.rs:152-155`), `push_design_row_cells` (`:163-166`), and the
manifest field (`:362-364`). All three are inside `render_quantities`
(`:245-252`), which builds the header, the rows _and_ the manifest. So
`render_quantities` takes `emit_scenario_col: bool` and `StackedQuantities`
forwards it. Threading it only into the stacker would leave the `Option` as the
sole suppression mechanism, and deleting `DesignCoords::none()` would then make
the column unconditional.

**`--enable`/`--disable` create no scenario axis, and get no column.** They are
mutually exclusive with `--scenario` (`main.rs:780-787`) but still build a
scenario patch — `ScenarioRef::Inline { name: "baseline", enable, disable }`
(`main.rs:1114-1120`) — which drives the intervention filter through the same
resolver path as a preset (`params_resolver.rs:661-672`). So
`simulate --enable sia --quantities-out d` emits a file schema-identical to the
un-patched run.

That is correct, for the reason the rule is stated in terms of an _axis_: an
ad-hoc patch is one modification of the single world being simulated, cannot
co-occur with `--scenario`, and never yields more than one design cell. There is
nothing for a per-row column to distinguish, and inventing a label would
reintroduce exactly the §2.4 fabrication. What the run _does_ need is provenance
— which run produced this file — and that belongs in the manifest, not in a
column repeated on every row. The manifest does not currently carry the ad-hoc
patch; recorded as a follow-up (§9) rather than fixed here.

`fit predict` continues to emit unconditionally, and the asymmetry is principled
rather than an inconsistency: predict's output _can_ contain the un-overlaid
fitted model alongside overlay rows in one file, so "no overlay" is a genuine
member of its axis and needs a name (`fitted`, reserved at `args/mod.rs:1229`).
A `simulate` run cannot produce that mixture.

**Duplicate scenario names are rejected at parse.** `scenario_names`
(`main.rs:714-716`) is not deduplicated and `n_scenarios` counts repeats, so
`--scenario a --scenario a` would give `Mode::Point` with two cells in one
accumulator and fail with "point-mode quantities require exactly one
realization" (`quantity_output.rs:259-264`) _after_ the whole grid has simulated
and committed leaves. Rejecting at parse turns a late, confusing failure into an
immediate one. This rejects input that is currently accepted; approved
deliberately.

`fit predict` has the **same** gap and a worse symptom: `scenario_refs()`
(`args/mod.rs:1254-1259`) does not deduplicate either — the check at `:1260` is
the `fitted` reserved-name guard, not a dedup — so `--scenario a --scenario a`
merges both passes into `by_scenario["a"]` and bands over each draw twice,
reporting `n_draws = 2N`. Filed as gh#579; both verbs should route through one
check rather than two.

**One file, stacked long, not one file per scenario.** Splitting by design
coordinate was considered and rejected. The risk it would address — a user
plotting a stacked file without grouping — produces a _visibly_ broken plot
(bimodal ribbons, crossing lines) with the information still present and
recoverable, which is categorically different from gh#562, where the pooling
happened inside the quantile computation and was unrecoverable. Long format is
also the tidy convention (`facet_wrap(~scenario)`, `hue="scenario"`), splitting
would force a read-and-concat for the most common operation, `fit predict`
already stacks, and the rule would generalize badly: sweep points are design
coordinates too, so "one file per design coordinate" means 4 scenarios × 3 sweep
points = 12 files per quantity.

### 3.4 `Mode` derived from the param source, replacing `banded: bool`

`SimQuantities.banded: bool` and `quantity_output::Mode` are two representations
of one state. The `bool` goes:

```rust
let mode = match job.source {
    ParamSource::Point { replicates: 1 } => Mode::Point,
    _ => Mode::Banded,
};
```

This implements the rule already documented at `quantity_output.rs:23`. It
differs from the current predicate on exactly one input — ≥2 scenarios, no
`--draws`, `replicates == 1`, which is 2.3 and is the intended fix.
`--replicates
N`, `--seeds a,b` (which rewrites `replicates` at `main.rs:778`
before the `ParamSource` is built), and any `--draws` including a one-row file
all agree with today. No existing test pins the old behaviour on the differing
input, because the refusal fires first.

`Mode` is one value for the whole render — a stacked file has one header per
quantity, so a mixed set would emit two schemas under one header. It is derived
from the param source, a property of the run, so no mixture can arise.

**This is the weakest part of the design and is deliberately not fixed here.**
`ParamSource` is not the property that matters. The real question is whether a
point set carries _sampling semantics_ — whether a quantile over it means
anything. `--draws posterior.tsv` and `--draws sensitivity-grid.tsv` are the
same type and different objects; `--draws uniform` bands today, though quantiles
of a space-filling sample describe the sampling scheme rather than a belief.
`ParamSource` is a proxy that is wrong in both directions. Recorded as a
follow-up (§9); the honest fix is a declaration of sampling semantics, not a
better discriminant over CLI flags.

### 3.5 The stacking loop moves to a neutral module

`fit predict` holds an inline "render each group, drop the repeated header,
stack the bodies, merge the manifests" loop (`predict.rs:1342-1357`), and
`simulate` now needs it verbatim:

A function taking all groups at once cannot serve `predict`: its sink is rebuilt
inside `for sweep_pt in &sweep_points` (`predict.rs:1259-1271`) and cannot
outlive the loop, while the stacking accumulators live outside it
(`:1232-1233`). A one-shot call would absorb only predict's _inner_
(per-scenario) stacking and leave the outer per-sweep-point header-drop
duplicated in `predict.rs` — moving the bug-prone logic up one frame rather than
removing it, which is the entire point of the lift.

So the seam is an accumulator, fed incrementally by both verbs:

```rust
pub(crate) struct StackedQuantities { /* bodies, manifest entries, mode, flag */ }

impl StackedQuantities {
    pub(crate) fn new(mode: Mode, emit_scenario_col: bool) -> Self;
    /// One design cell. The first group for a quantity contributes its header
    /// and rows; later groups contribute rows only, so all groups stack under
    /// one header.
    pub(crate) fn push_group(
        &mut self,
        quantities: &[ir::quantity::Quantity],
        coords: DesignCoords<'_>,
        draws: &[Vec<QuantityResult>],
        times: &[f64],
    ) -> Result<(), String>;
    pub(crate) fn finish(self, calendar: &io::CalendarMeta)
        -> Result<(Vec<(String, String)>, String), String>;
}
```

`predict` pushes once per (sweep point × scenario) from inside both loops;
`simulate` pushes once per scenario. The header-drop and manifest merge then
exist in exactly one place.

The caller supplies coordinates here, which is safe precisely because it is a
_rendering_ seam, not an accumulation seam: the pooling was possible because the
accumulator merged cells, and a renderer handed already-separated groups cannot
re-merge them.

One wrinkle for the implementer: predict's per-scenario loop also builds
`ff_cells` via `assemble_predictive` (`predict.rs:1364-1370`), so routing it
through `StackedQuantities` is a small restructure of that loop body, not a
verbatim extraction. Output stays byte-identical either way.

### 3.6 What is not unified

`predict`'s `ScenarioAccum` (`predict.rs:717-726`) also carries `samples` — the
predictive `y_rep` per `(leaf, time)` — which `simulate` has no analogue for,
and the two accumulate over different lifetimes (`predict` over draws it already
holds, `simulate` over cells as they stream). They stay distinct. The shared,
bug-prone substrate is the stacking, and only that is lifted.

`band`, `quantile`, `fmt_time`, `fmt_value` and `QUANTILE_LEVELS` live in
`fit/predict.rs` and are imported _by_ the shared renderer
(`quantity_output.rs:20`) and by `contrasts.rs:48` — a
consumer-owns-the-substrate inversion. They move to a neutral
`cli/src/quantile.rs` that `quantity_output`, `predict` and `contrasts` all
depend on. Moving them _into_ `quantity_output` would relocate the inversion
rather than remove it, since `contrasts` has nothing to do with quantities.

## 4. The rejected design, and why

A general design-coordinate type was designed and rejected. Recorded because it
is the obvious next idea and the reasons are not obvious.

The shape was: `PointKind` (`Single | Sweep | Draw`) stamped on `CellSpec` by
`plan_grid`; `BandKey { scenario, sweep }` derived from a cell by a single
`BandKey::of`; `BandSet` as the only constructor of a `Band`, which the renderer
would take instead of a `(draws, times, coords)` triple.

**It does not deliver the guarantee it is named for.** `BandSet::push(key, …)`
takes the key from the caller, so a caller passing one key for every cell pools
exactly as today. The invariant moves up a stack frame rather than into a type —
and it is _weaker_ than the existing `push_cell(&CellResult)` shape (§3.1),
which gives the caller no key to pass.

**`BandKey::of(&CellSpec)` cannot serve `fit predict`.** Predict implements
sweeps by folding swept parameters into the draw rows and running one
`ParamSource::Draws` job per sweep point (`predict.rs:1281-1308`); the sweep
value lives in the enclosing loop (`:1261`), never in the cell. Every predict
cell is therefore a draw, `BandKey::of` returns an empty sweep, and either the
`scenario\tsweep:k\t…` header pinned at `fit_predict_sweep.rs:230` regresses, or
— if the set spans the sweep loop — two sweep points merge into one band. That
is gh#562 reproduced in the other verb by the type introduced to prevent it.
Repairing it needs a public `BandKey::new(scenario, sweep)` for predict to call,
at which point there are two constructors for "which band is this" — the gh#233
fork shape.

**`PointKind` names the wrong property, not merely a premature one.** It records
which CLI flag produced the points, not whether they carry a measure over which
a quantile means anything (§3.4). Naming a discriminant for the proxy would
entrench the confusion at the moment the real property is finally needed.

**And it is unreachable.** `ParamSource::Sweep` is constructed only in
`batch.rs` (`:703`, `:1516`); batch merges through `CasSink` (`:1136`) and emits
no quantities. `BandKey::of` would return an empty sweep at every live call
site, making `BandKey` isomorphic to `String` — seven of seven sites, against
the pragmatic line in `.claude/rules/rust-conventions.md`, which dropped
`NominalStep`/`SnapGrid` at six of seven. `CellSpec` also has four construction
sites, not one: `batch.rs:1019` (`predict_cells`, the dry-run cache predictor)
has no `ParamSource` in scope and would have to guess.

**We are also not restructuring `fit predict`.** Routing its sweep through
`CellSpec` so one `BandKey::of` would work is a change to a fitted-model path
motivated entirely by the elegance of an abstraction with one currently-useful
component (`scenario`) — the sweep half is unreachable because batch emits no
quantities. That is architecture-driven rather than requirement-driven. The
asymmetry costs nothing until some operation must reason uniformly across both
paths, and none does. When `batch run --quantities-out` exists there will be two
real design axes and two independent clients, and that is when the abstraction
should emerge (§9).

## 5. Migration

**Increment 1 — move the quantile primitives to `cli/src/quantile.rs`.** Moves
`QUANTILE_LEVELS` (`predict.rs:339-342`), `quantile` (`:346-362`), `band`
(`:364-377`), `fmt_time` (`:521-528`), `fmt_value` (`:530-540`), and the four
unit tests at **`:2160-2194`** — not `:2196`, which is the doc comment for
`cloud()`, a helper the remaining `subsample_draws` tests still need.
`level_for` (`:513-519`) and `write_tsv` (`:542-552`) stay.

Three things a literal reading would miss:

- **`predict.rs` needs its own import back.** It has ten internal use sites
  (`QUANTILE_LEVELS` at `:438`, `:1529`, `:1530`; `fmt_value` at `:455`, `:476`,
  `:505`; `fmt_time` at `:459`, `:498`; `band` at `:1743`, `:2001`). Import
  exactly those four — **not** `quantile`, whose only caller is `band`; an
  unused import is a hard local error, since `rust/.cargo/config.toml` sets
  `rustflags = ["-D", "warnings"]`.
- **`mod quantile;` must be added to `main.rs`**, beside `mod quantity_output;`.
  A private `mod` at the crate root is sufficient — it is an ancestor of every
  module, so existing visibilities resolve unchanged.
- **`contrasts.rs:48` is a split, not a rewrite**: `write_tsv` stays in
  `predict`, so that one `use` becomes two.

Pure move; outputs byte-identical (all five are pure functions of their
arguments, and Rust does not perform FP contraction).

**Increment 2a — lift `StackedQuantities`, add the calendar line.** The
`emit_scenario_col` flag is _not_ introduced here: in 2a `DesignCoords.scenario`
is still `Option`, so the parameter would have no reader and CI runs
`cargo clippy --all-targets -- -D warnings` (`.github/workflows/ci.yml:79`). 2a
lifts the stacker under the existing `Option` semantics; the flag arrives in 2b
with the field change. `fit predict` routed through the shared stacker;
`"calendar": calendar.to_json()` added at `predict.rs:1376` (§6). One intended
output change (predict's manifest gains `calendar`); otherwise byte-identical.

**Increment 2b — `DesignCoords.scenario: &str`, `none()` deleted.** This is
where the simulate header churn lands, because dropping the `Option` forces the
render site at `main.rs:1443` to supply a name in the same commit. Not
byte-neutral; carries the ~40 assertion updates in §7.

**Increment 3 — `simulate` keyed by scenario (gh#562).** `SimQuantities.draws`/
`times` become `by_scenario`; `push_cell` keys on `cell.spec.scenario.name()`;
`mode` and `scenario_axis` replace `banded`; duplicate `--scenario` names
rejected at parse; the refusal at `main.rs:1278-1296` and its test
(`simulate_quantities.rs:264`) deleted. Carries the red→green test and the
`gh#562` subject.

Increment 1 can land while 2a is in review; 2b depends on 2a, 3 on 2b.

## 6. A defect fixed in passing

`fit predict` builds its quantities manifest at `predict.rs:1376-1379` with
`schema` and `quantities` only and writes it to `quantities.json`
(`:1596-1600`). `render_quantities` always includes
`"calendar": calendar.to_json()` (`quantity_output.rs:380-384`), so `simulate`'s
manifest carries it and predict's does not — while predict's two _other_ sidecar
manifests do (`:1543`, `:1580`). `CalendarMeta` is `time_unit`, `origin` and
`days_per_unit` (`io/src/calendar.rs:31-41`); without it a consumer cannot map
the numeric `time` column to dates without re-parsing the model, which is the
one thing the block exists to prevent, and it bites hardest on dated outbreak
work.

This is **not** fixed structurally by the lift — predict builds its top-level
manifest outside the shared stacker, per sweep point — so it is one explicit
line in increment 2a. Pinned by test 7.5.

## 7. Tests

The defect is at a **call site** — the accumulator, and the argument handed to
the renderer — not inside a unit-testable function, so a renderer unit test
passes with the bug present. The pinning tests are end-to-end.

**7.1 gh#562 red test** (`crates/cli/tests/simulate_quantities.rs`). The fixture
already declares `baseline` and `ctrl` presets, both setting all four parameters
(`:66-83`) — necessary because a scenario `set` wins over the draw tier
(`engine.rs:406-419`), so a `ctrl` pinning only `beta` would leave its other
parameters coming from the uniform draw while `baseline`'s are pinned, making
the arms incomparable.

```
camdl simulate sir_q.camdl --draws uniform -n 5 --seed 7 \
      --scenario baseline --scenario ctrl --quantities-out qdir
```

Assert on `qdir/quantities/prevalence.tsv`: the header leads with `scenario`;
exactly `baseline` and `ctrl` appear; **`n_draws == 5` on every row, not 10**;
each scenario contributes the same number of time rows; `quantities.json`
carries one entry per `(quantity, scenario)`, each tagged.

Mutation check: revert `push_cell` to a flat accumulator _while keeping the
column_, and confirm the test goes red on the `n_draws` assertion specifically.
A mutation that also removed the column would fail on the header assertion first
and would not prove the band is unpooled.

Note the five draws are parameter-identical within an arm (the presets pin
everything), so the band is replicate variation. Sufficient for the `n_draws`
assertion; must not be described as parameter uncertainty.

**7.2 Within-scenario time-grid mismatch is an error, not first-wins.** A unit
test on the accumulator: push two cells into the same scenario with different
grids, assert a located error naming the scenario and both lengths. Without this
the §3.1 check is untested and would rot back into `if times.is_empty()`. (The
end-to-end version — two scenarios with genuinely different snapshot counts — is
unconstructible until gh#561 threads `Preset::t_end`, and belongs to that
issue.)

**7.3 `banded` excludes the scenario factor.** Two scenarios, fixed params, no
`--draws`, `--quantities-out`: assert point mode. Per scenario there is exactly
one realization, so `render_quantities`' `mode == Point && n_draws != 1` guard
passes for each group and the file is one header plus two rows. The header is
`scenario\tvalue` (scalar) or `scenario\ttime\tvalue` (series) — not a bare
`value`, since the scenario axis exists.

**7.4 No `--scenario` emits no scenario column.** The §3.3 rule, and the guard
against the §2.4 fabricated label reaching user data.

**7.5 `--draws <file>` with one row stays banded.** Guards the regression a
cell-count predicate would introduce (§3.4).

**7.6 `fit predict` manifest carries `calendar`.** Red before increment 2a.

**7.7 `run_job` merges in canonical order under parallelism.** `IndexMap`
insertion order decides output order, so it depends on `merge_cell` call order,
which the design now leans on for reproducible artifacts. Test the invariant
directly rather than through a verb: a unit test on `engine::run_job` with a
recording sink, a multi-scenario job and `parallel: 8`, asserting the merge
sequence equals `plan_grid` order.

An end-to-end thread-count test would be **vacuous** for both verbs, which is
worth recording so nobody writes one. `simulate` hardcodes `parallel: 1`
(`main.rs:1183`). `fit predict` runs one scenario per `run_job` call inside a
sequential `for sref in &scenario_refs` loop (`predict.rs:1273`, with
`scenarios: vec![sref.clone()]` at `:1310`), so its `by_scenario` order is fixed
outside the Rayon phase entirely and no thread count can perturb it. The
property is only exercised when a _single_ `run_job` carries several scenarios
with `parallel > 1` — which today is `batch run` alone (`batch.rs:715`, `:725`),
and it emits no quantities.

**7.8 A/B byte-identity.** Increments 1 and 2a (apart from the calendar block):
same command, same seed, diff every artifact; only timestamps and mtimes may
differ. Must include a multi-scenario predict run — an A/B over single-scenario
runs exercises none of the changed paths.

**Churn to expect in increment 2b**, measured: the `scalar()` helpers at
`quantities_surface.rs:70-76` and `observations_quantity.rs:74` assert the
header _and_ return the value row, so ~40 assertions across
`quantities_surface.rs:83-107` break unless the helper strips the leading
scenario cell; plus `quantities_surface.rs:123-125`, `:130`, `:133`;
`simulate_quantities.rs:131`, `:135`, `:144`, `:153` (point) and `:221`, `:231`,
`:241` (banded); and `quantities_surface.rs:126` (`Some("b\t764")`), which the
`:123-125` range misses. **Four** files, not three — `quantity_output.rs`'s own
tests are the fourth: `DesignCoords::none()` has five in-module call sites
(`:584`, `:620`, `:660`, `:701`, `:778`), and
`banded_render_with_scenario_tags_header_rows_and_manifest` (`:776-789`) asserts
that a `None` scenario _omits_ the column and the manifest field. Its subject
stops being constructible, so it is rewritten against `emit_scenario_col`, not
re-spelled.

Doc surfaces mentioning the artifact (`docs/camdl-language-spec.md:3817`,
`docs/dsl-cheatsheet.md:277`, `docs/user-features.md:370`, `docs/agents.md:354`)
should be checked, though none prints a column list, so the edit is likely
smaller than a schema change implies. `fit_table.rs:414-421` reads a `scenario`
column and matches `"fitted"`, but is predict-only and predict still emits
unconditionally — verified unaffected.

## 8. Decisions

1. **When does the `scenario` column appear?** Iff a scenario axis exists (§3.3)
   — never `n_scenarios > 1`, and no `ScenarioCol` policy toggle.
2. **A general design-coordinate type?** No (§4). Revisited when
   `batch run --quantities-out` exists.
3. **Per-scenario `times`, first-wins or validated?** Validated (§3.1).
   First-wins leaves a smaller copy of the original bug shape.
4. **Unify the accumulators?** No (§3.6). Unify the stacking only.
5. **Restructure `fit predict` for uniformity?** No (§4). Requirement-driven,
   not architecture-driven.
6. **Keep the multi-scenario refusal after the fix?** No. A refusal that can no
   longer fire is dead code and would imply the pooling is still expressible.
7. **Should `RunSink` express grouping?** No. It is a _delivery_ abstraction —
   "here is a completed cell". Grouping is an _interpretation_ of cells, and
   pushing it into the trait would couple the engine to a statistical operation
   `CasSink` has no use for. That three sinks accumulate differently is not
   evidence the trait is wrong; the differences are genuine (§3.6).
8. **One file or one per scenario?** One, stacked long (§3.3).
9. **Reject duplicate `--scenario` names?** Yes, at parse (§3.3).

## 9. Follow-ups

- **gh#561** — thread `Preset::t_end` into the per-cell config. Ordered strictly
  after increment 3. Two notes carried forward: the per-cell `t_end` must enter
  the run identity, since it changes the stored trajectory — a deliberate
  re-key, and a cost this proposal does not incur; and the ragged-`ensemble.tsv`
  question is already answered by the format, which is long-form with a
  `scenario` column whenever `n_scenarios > 1` (`main.rs:2037`, `:2048`), so
  unequal horizons need no padding and no split. What a scenario `to` _shorter_
  than the model window means remains gh#561's to decide.
- **gh#572** — reject scenario/sweep parameter overlap in all three verbs
  (§1.1). Note the `fit run` instance (a fit.toml `scenario` at tier 4
  overriding a `--sweep` value at tier 2) is **inferred from the code path, not
  executed**.
- **gh#573** — the scenario-level run identity omits `scale` and the composed
  set, so two presets differing only in `scale` or `compose` share a `run_id`.
  Independent of this change and ordered after it.
- **gh#577 — stop fabricating a scenario member** (§2.4). Note there are three
  fabrication sites, not two: `main.rs:1116` (the job's scenario ref),
  `engine.rs:338-344` (`effective_scenarios`), and `main.rs:1748-1755`, which
  builds a `ResolvedEntry { name: "baseline" }` for the CAS path. Suppressing
  the column keeps the borrowed name out of user data, but the placeholder
  remains internally and in CAS leaf paths. The root fix — absence represented
  as absence — touches `effective_scenarios`/`plan_grid` and changes store path
  labels, so it is out of scope for a cli-only banding fix.
- **gh#576 — the trajectory TSV's `scenario` column** uses the `n_scenarios > 1`
  rule (`main.rs:2037`), so after this change a single-scenario run emits the
  column in `quantities/*.tsv` and not in the trajectory beside it. Same
  schema-instability argument as decision 1, different artifact and a larger
  compatibility surface.
- **gh#575 — sampling semantics are not `ParamSource`** (§3.4). The honest fix
  is a declaration of whether a point set carries a measure. Until then the
  software claims more than it can justify: `n_draws` counts _samples_, not
  parameter draws — a `Point { replicates: 5 }` run reports `n_draws = 5` with
  zero parameter draws. Rename to `n_samples`, have the manifest carry the
  decomposition (`parameter_points`, `replicates_per_point`,
  `summary: empirical_quantiles`), and reserve "credible interval" for point
  sets with posterior-probability semantics — "quantile band" is honest for
  everything the renderer produces today.
- **gh#579 — `fit predict` duplicate `--scenario` names double-count draws.**
  `scenario_refs()` does not deduplicate, so both passes merge into one
  accumulator and the band reports `n_draws = 2N`. gh#562's shape in the verb
  this proposal treats as the correct model. Both verbs should route through one
  dedup check (§3.3).
- **The quantities manifest carries no ad-hoc scenario patch.** With
  `--enable`/`--disable` the run simulates a modified world but emits no design
  coordinate, by design (§3.3) — one cell, nothing for a per-row column to
  distinguish. Provenance still belongs somewhere, and the manifest is the
  place: it should record the ad-hoc `enable`/`disable` lists alongside the
  calendar block.
- **gh#574 — `--stdout` silently suppresses `--quantities-out`.**
  `simulate --stdout` returns at `main.rs:1376-1387`, before the quantities
  writer at `:1433`, so `--stdout --quantities-out d` writes nothing to `d`
  without saying so. Pre-existing, adjacent.
- **`batch run --quantities-out`** — where the §4 generalization is re-examined
  against a real second design axis. The shape to reach for then is not
  `PointKind`/`BandKey` but a role on the axis itself, declared where the design
  is defined rather than inferred from a CLI flag:

  ```rust
  struct Axis { name: String, values: Vec<Value>, role: AxisRole }
  enum AxisRole { Condition, Sample(SamplingSemantics) }
  ```

  with contrasts operating explicitly on a `Condition` axis. Two further things
  must be solved there: predict-style sweep coordinates that do not live in the
  cell, and CAS cache hits, which `run_job` filters out before simulation
  (`engine.rs:209-219`) so a warm run would band over only the missed cells.
  Considered and **closed**, recorded so it is not re-raised: the quantity
  evaluator compiles `CompiledModel` + `QuantityEvaluator` from the _first_ cell
  only (`main.rs:1621-1632`), and `CompiledModel::new` bakes `time_function`
  piecewise breakpoints/values and spline knots at default parameter values
  (`compiled_model.rs:1277-1289`, `:1363-1374`). A scenario `set` on a parameter
  in a structural forcing position would therefore have scenarios 2..n
  evaluating quantities against scenario 1's constants. It is unreachable: the
  compiler rejects the precondition outright.

```
$ camdl check tf.camdl      # piecewise `values = [0.0, k]`, k a parameter
error[E600]: forcing `drive`: parameter 'k' drives the piecewise forcing
coefficient, which is structural data — interpolation knots, piecewise step
grids, and the spline basis are precomputed at construction and cannot vary per
step, so they cannot be an estimated parameter. Make the coefficient a constant,
or use a sinusoidal, fourier, or periodic forcing (whose coefficients are live)
```

E600 covers parameters, and a scenario `set`/`scale` acts on parameters, so
neither the draw axis nor the scenario axis can reach the baked path. Forcings
whose coefficients _are_ live (sinusoidal, fourier, periodic) are evaluated per
step and are unaffected. Compile-once is therefore safe, and increment 3 is not
gated on it.

## 10. The workflow documentation gap (tier 2, gh#578)

Not a scope item; recorded because this proposal's whole subject is the
posterior→scenario path and the gap sits directly on it.

The mechanism is sound. `fit predict` builds one job per scenario with every
posterior draw as a row and the scenario applied on top at tier 4
(`predict.rs:1273-1320`), so parameter uncertainty propagates into every arm.
And `process_seed_for` is `mix_cell_seed(base_seed, point_idx, rep)`
(`engine.rs:52-66`) — **independent of the scenario** — so draw _i_ in the
baseline arm and draw _i_ in the intervention arm share a process seed. Common
random numbers across counterfactual arms, structurally, which is what makes the
paired contrast of §1 both correctly matched and low-variance.

None of it is documented. `fit predict` appears **zero times** across every
`.qmd` in `camdl-book`; `contrasts` appears **zero times**. What the scenarios
chapter teaches instead is comparison via
`run_simulate(MODEL, PARAMS, seed=42, scenario=…, replicates=200)` — fixed θ
from a params file plus a scenario's `set`, with 200 stochastic replicates —
captioned "Uncertainty ribbons for the infection curve under three scenarios"
(`guide/scenarios.qmd`, `fig-scenario-uncertainty`).

Those ribbons are demographic stochasticity at one assumed parameter vector, not
uncertainty about the effect. The arms are correctly paired, so the comparison
is sound; the width is what gets over-read, and parameter uncertainty typically
dominates stochastic uncertainty once N is large. A policy reader takes
"uncertainty ribbon" to mean the latter.

This is arguably more consequential for users than gh#562 was: gh#562 produced a
wrong number, which is at least wrong. This produces a right number that answers
a narrower question than the caption implies. The fix is a book section on
posterior → scenarios → contrasts, and a caption correction on the existing
figures.
