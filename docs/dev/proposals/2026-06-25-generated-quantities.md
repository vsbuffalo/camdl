# Generated quantities

Status: **Implemented (v1.1)** — latent-state quantities and the simulated
observation source (`observations.<stream>`), emitted by `fit predict` and
`simulate`, with a committed showcase golden (`tests/fixtures/quantities/`)
pinning both the IR shape and the output values. The standalone/disk command,
windowed / cross-stratum / flow reductions, and the dim→unit renderer remain the
staged follow-ups below. Supersedes:
`2026-06-04-output-trajectory-customization.md` Phase 2; splits the quantities
half out of `2026-06-24-generated-quantities-and-counterfactuals.md` (the
counterfactual half is `2026-06-25-counterfactual-contrasts.md`). IR contract:
one additive optional field on `Model`; `ir/VERSION` 0.19 → 0.20.

## Summary

A camdl user cannot yet ask the model to _report a derived quantity_ —
cumulative incidence, attack rate, peak prevalence, time to peak. Today the only
way to compute a function of a trajectory is to smuggle it in as a _scored_
observation stream (the expander requires a likelihood, `E266`), which forces
the author to pretend a reported quantity is data.

This adds a `quantities {}` block: **named reductions of what a simulation
produces**, emitted on every run and banded over draws. A quantity is the
**non-scored twin of an observation** — it reuses the projection machinery,
minus the likelihood. **v1 reduces latent state** (`prevalence = I / N`,
`peak = max(I)`, `attack_rate = final((N0 - S) / N0)`,
`time_to_peak =
time_of_max(I)`); **v1.1 adds the simulated observation series**
as a second source on the same seam.

The core is one pure evaluator (`sim::quantity`) fed per draw by the run loop.
v1 evaluates **only on the paths that run every cell fresh** (`simulate`,
`fit predict`), so the trajectory is always in hand; a disk-replay source (a
standalone command, `batch run`, cache-hit reuse) is a named follow-up. Nothing
here touches the inference kernels, and the shared per-instant `Expr` type stays
closed.

## Scope: per-simulation reporting, not preconditioning

A `quantities {}` block reports functions of a **simulation**. It deliberately
does **not** cover the preconditioning workflow — extracting a feature vector
`s(y_obs)` from the _observed data_, comparing it to `s(y_sim)`, retaining a
particle bank, hashing a feature set. That is a distinct purpose with a distinct
command, **deferred to a future preconditioning proposal**, which will own the
**observed-data realization** of a stream, a data-only feature command, the
`s(y_obs)` ↔ `s(y_sim)` comparison and its hole-mask / MAR semantics, and
preconditioning-artifact identity.

Increments, sharing one seam:

- **v1** — latent-state quantities, evaluated **live** on the always-fresh paths
  (`simulate`, `simulate --draws`, `fit predict`). Every quantity is a pure fold
  over the trajectory. No RNG, no observation sampling, no disk replay.
- **v1.1** — the simulated observation source (`observations.<stream>`): a
  **reduction** over the same `y_sim` the run already drew (never a fresh draw).
  A bare observation series is rejected (`E289`): an observation series lives on
  the stream's own observation-time axis (its `emit_schedule` / fit leaves),
  distinct from the trajectory snapshot grid, so it cannot be rendered against
  the same time column as a state series — reduce it (`max`, `integral`,
  `first_above`, …). Adds observation materialization and a split
  materialization-grid rule. Purely additive on the v1 seam.
- **follow-up (disk source)** — a standalone `camdl quantities <run>` command,
  `batch run` quantities, and cache-hit reuse. All require a `TSV → Trajectory`
  reader that does not exist today (and a resolution of the lossy `traj.tsv`
  view-column / `--no-flows` interaction), so they are explicitly out of v1.
- **deferred** — the observed-data realization and the preconditioning command.

## Architecture: the per-draw seam

The whole feature is one pure evaluator behind a per-draw fan-out. The pure
evaluator lives in **`sim`** (a new `sim::quantity` module): it needs
`Trajectory` and `CompiledModel`/`ResolvedExpr` (both in `sim`) and the quantity
IR (in `ir`, which `sim` depends on). Banding, the per-draw hook, and all IO
stay in `cli`. (There is no `observe` crate — the projection/likelihood
machinery is `sim::inference`; the `cli → io → observe → sim → ir` line in
CLAUDE.md is stale.)

```rust
// sim::quantity — pure: no RNG, no file IO, no inference coupling.
pub struct QuantityEvaluator { /* per-leaf resolved programs (ResolvedExpr) + topo order */ }

impl QuantityEvaluator {
    /// Resolve each (already strata-expanded) quantity leaf's `Expr`/thresholds to
    /// `ResolvedExpr` ONCE against the compiled model (name → slot). `Derived`
    /// `QRef`s resolve to indices into earlier leaves' values (topological).
    pub fn new(quantities: &[ir::Quantity], compiled: &sim::CompiledModel)
        -> Result<Self, String>;

    /// Fold every quantity over ONE draw: the resolved param vector + the
    /// trajectory (v1.1 also passes the materialized obs series). Pure.
    pub fn eval_draw(&self, params: &[f64], traj: &sim::Trajectory)
        -> Result<PerDrawValues, String>;
}
```

```rust
// cli — accumulates per-draw values, bands in finish. Retains only DERIVED
// values, never the trajectories.
struct QuantityConsumer {
    eval:   QuantityEvaluator,
    source: ParamSourceKind,                 // fixed-params → point; draws → band
    scalar: HashMap<QName, Vec<QuantityDrawValue>>,   // one per draw
    series: HashMap<(QName, usize), Vec<f64>>,        // one per draw, per output time
}
```

The live source drives `push_draw` from inside the sink's `merge_cell` (below);
the disk source (follow-up) reconstructs `(params, traj)` from persisted
artifacts. Both feed the **same** evaluator/consumer; only the source differs.

### Where the live hook sits, and the two sinks

`run_job` is the single engine; output is a pluggable `RunSink`
(`engine.rs:121`) whose `merge_cell(cell: &CellResult)` is called **once per
cell that runs**, in canonical order (`engine.rs:243`), with the full trajectory
in hand (`CellResult.traj`, `engine.rs:92`). The quantity consumer is driven
from inside `merge_cell`, **composed alongside** the existing accumulator — not
after the final (pooled/banded) sink (the draw boundary is gone there:
`PredictiveSink.samples[stream][time]`, `predict.rs:684`; a scalar like
`max(I/N)` cannot be recovered from `q05…q95`), and not as a second `RunSink`
(which would re-sample obs).

The two v1 sinks are **asymmetric** and need two small integrations:

- **`PredictiveSink` (`fit predict`)** already holds `Arc<CompiledModel>` and
  builds the draw's resolved params (`predict.rs:649`/`:663-668`) — wire the
  consumer directly.
- **The simulate `SimSink`/`CasSink`** holds only `ir::Model` + a param map, no
  compiled model. It must **compile once at sink construction** (the IR is fixed
  across cells in a run) to build the `QuantityEvaluator`, then per draw resolve
  params (defaults + `cell.spec.point_overrides`) and call `push_draw`. (One
  compile, not per draw — do not recompile in the loop.)

**v1 evaluates only on always-fresh paths.** `simulate` never skips a cell —
`should_run` is not overridden; "every planned cell runs … cache hits are
handled idempotently" (`main.rs:1347`) — so `cell.traj` is always present in
`merge_cell`. `fit predict` replays every draw (overwrite-in-place). So the live
hook always has a trajectory. The one path that _skips_ — `batch run`'s
`CasSink` (`should_run → false → on_skip`, no `merge_cell`, no traj,
`batch.rs:1091`) — is **not** a v1 quantities path; it needs the disk source
(follow-up). Listing `batch run` as a v1 quantities path would be a silent
matrix gap, so it is explicitly excluded.

**Memory.** The consumer retains per-draw derived values, not trajectories — but
a **stratified series** quantity expands to one leaf per cell, so the true cost
is `Σ_q stratum(q) × times × draws`. At national scale (≈774 cells × ~200 times
× 1000 draws ≈ 1.2 GB for one stratified series) this is the **same order as the
predictive band's `samples[leaf][time]`**, not negligible. Acceptable (the
predictive path already pays it) but documented; a streaming-quantile sketch is
a follow-up if it bites.

## Types and the flow

Rust types (the `ir` crate) are canonical; OCaml `ir.ml` + `serde.ml` mirror
them.

```rust
/// The non-scored twin of an ObservationModel: a named reduction of a simulation
/// output to a reported summary, no likelihood. Reporting-only, non-identity.
pub struct Quantity {
    pub name:    String,
    pub stratum: Vec<StratumKey>,   // post-expansion (dim, level) tag; like ObservationModel.stratum
    pub body:    QuantityBody,
}

#[serde(rename_all = "snake_case")]   // externally tagged
pub enum QuantityBody {
    Reduced { source: QuantitySource, reduce: Option<TemporalReduce> },
    Derived(ScalarExpr),
}

#[serde(rename_all = "snake_case")]   // externally tagged — see "additivity" below
pub enum QuantitySource {
    /// Latent truth: a quantity-validated `Expr` evaluated against each snapshot.
    State(Expr),
    // v1.1 ADDS, additively:  Observation { stream: String },
}
```

`stratum` is the only stratification field (the IR is fully-expanded:
`prevalence[p in patch]` becomes one leaf per cell tagged with `stratum`,
exactly as `ObservationModel` carries only `stratum: Vec<StratumKey>`,
`observation.rs:200`, and **no index-binding field** — there is no IR
`IndexBinding`).

**Additivity (load-bearing).** `QuantitySource` (and `QuantityBody`,
`TemporalReduce` and its inner enums) **must be externally tagged**, never
`#[serde(untagged)]`. External tagging makes `State` serialize
`{"state": <expr>}` and stay byte-identical when v1.1 appends `Observation`, and
keeps a parse error from being swallowed by untagged trial-deserialization. The
single-variant `QuantitySource` in v1 is therefore not a stub — it fixes the
wrapper shape now so v1.1 is a pure additive variant (no golden churn, no run-id
move). A cross-language round-trip test pins the tag (the OCaml `serde.ml` is
hand-written — no derive to lean on).

### The state expression is a validated `Expr`, enforced in `ir::validate`

A quantity's state expression is a **plain `Expr`**, restricted to a **validated
subset** — not a newtype with a constructor invariant (a newtype-over-`Expr`
deserializes transparently and would bypass the check; the OCaml IR has no
private-constructor newtype). The subset is enforced in **one seam**:
`ir::validate::check_expr` (`validate.rs:369`), which already runs on **every
load** before simulation (`util.rs:1039`, the structural-integrity battery),
already deep-walks, and already carries a per-context allow flag
(`allow_projected`) — the exact mechanism. Add a **quantity context** that:

- **rejects four leaves anywhere in the tree** — `Dt`, `Projected`,
  `ObsColumnRef`, `PerEvalRef` (meaningless in a quantity read at output
  cadence) — recursing through every compound variant **including
  `UncheckedDim.inner`** (the `unchecked(…)` escape is DSL-reachable and wraps
  an arbitrary `Expr`), `Reduce`'s vec, and `TableLookup` indices;
- **checks `BindingRef` transitively** — a `BindingRef` leaf is allowed (the
  hoisted `N[l]`/`I_agg[l]` aggregates appear in `I[p]/N[p]`), but its
  referenced `Binding` body (`model.bindings`, available to `validate`) must
  itself be quantity-clean; a single memoized pass over the topo-ordered
  bindings suffices. (This closes the smuggle a constructor-only check cannot:
  `binding_ref:"B"` where `B`'s body is `{dt:null}`, reachable even from DSL via
  a hoist-eligible `let bad = I*dt`.)

A flow accumulator is genuinely unrepresentable: `Expr` has no flow leaf
(`expr.rs:235-253`); flows are reachable only via the
`Projection::CumulativeFlow` _variant_, which `State` does not use. So
`CurrentPop`/`CurrentPopSum` are just `Pop`/`PopSum`.

**Diagnostics.** The OCaml expander emits the friendly, **located** `E288` for a
direct forbidden leaf a user typed (`dt` in a quantity body — threaded from the
`EIdent` location; note most expander expression diagnostics are `no_loc`, so
this requires AST-level interception, not a walk over the resolved IR).
`ir::validate` is the authoritative backstop for hand-authored IR and the
transitive case (decl-level, span-less — the IR carries no spans). **LICM must
not process quantity bodies** (else it would hoist a subexpression into a
`PerEvalRef` the loader then rejects).

### `TemporalReduce` — result kind typed

```rust
#[serde(rename_all = "snake_case")]
pub enum TemporalReduce {
    Value(ValueReduce),    // result has dim(series)
    Time(TimeReduce),      // result has dim T
    Integral,              // result has dim(series)·T
}
#[serde(rename_all = "snake_case")]
pub enum ValueReduce {
    Final, Max, Min, Mean,
    CountAbove(Expr), CountBelow(Expr),   // # crossings (e.g. positive months)
}
#[serde(rename_all = "snake_case")]
pub enum TimeReduce {
    ArgMax, ArgMin,                       // time_of_max / time_of_min
    FirstAbove(Expr), FirstBelow(Expr),   // onset / first detection
    LastAbove(Expr),  LastBelow(Expr),    // elimination / fade-out
}
```

`None` reduce → a **series** (one value per snapshot); `Some` → a **scalar**.
Thresholds are a quantity-validated `Expr` (state is always available in v1's
trajectory contexts), so `first_above(I, 0.1 * N)` is well-defined. `Final`
reads the endpoint, `Integral` is the time-weighted area (person-days,
`dim(series)·T`; well-defined for integer compartments, forward-compatible with
real-valued projections).

**No `total`/`sum` in v1.** A `Σ` over the series is meaningful only for a
**per-interval flow** (incidence → cumulative); summing a _stock_ over snapshots
is cadence-dependent nonsense, and v1 has no flow source (`Expr` has no flow
leaf). `Total` ships with the flow/observation source (follow-up), not v1.

### Censoring is data, not a NaN

A `Time` reduction can fail to fire — `first_above(I_total, i_thresh)` on a draw
that never crosses (common at low transmission). That is **right-censoring**,
distinct from a non-finite arithmetic result:

```rust
pub enum QuantityDrawValue { Value(f64), Censored(Censoring) }
pub enum Censoring { Right { bound: f64, reason: &'static str } }   // v1: this case only

/// Banding partitions per-draw values, then dispatches. Series quantities never
/// censor (their per-draw payload is `Vec<f64>`); only `Time` SCALARS can.
pub enum BandResult {
    Banded { bands: Vec<f64>, n_value: usize, n_censored: usize },   // p_censored derived
    AllCensored { n_draws: usize },                                   // q* = NA
}
fn band_with_censoring(vals: &[QuantityDrawValue]) -> BandResult;     // partitions, never calls band(&[])
```

- **Never a `NaN` sentinel.** A non-firing `Time` reduction →
  `Censored(Right { bound: window_end, .. })`.
- **Non-finite arithmetic is a different thing.** A `Value` reduction yielding
  `NaN`/±∞ (e.g. `0/0`) is a bug / undefined → a hard error, **not** `Censored`
  (keep `band`'s non-finite rejection, `predict.rs:326`).
- **Band only the finite set.** `band_with_censoring` partitions to the
  `Value(f64)`s and calls the existing `band(&[f64])` (`predict.rs:325`) — no
  new banding entry point. The all-censored case returns `AllCensored` (never
  `band(&[])`, which silently returns `vec![NaN;5]`, `predict.rs:307`).
- **Quantiles are conditional on the event occurring in the window** — the
  manifest records this; a survival (Kaplan–Meier) summary is a follow-up.

`CountAbove`/`CountBelow` never censor (a non-crossing series counts `0`);
`Max`/`Min`/`Mean`/`Final`/`Integral` are always defined.

### `ScalarExpr` — reduction arithmetic, `Expr` left closed

Differences/ratios of reduced scalars (outbreak duration `last - first`,
indicators) combine **scalar** quantities, in a small closed language distinct
from `Expr`. This **is** genuinely structural — a `ScalarExpr` cannot hold a
state/rate leaf by construction:

```rust
#[serde(rename_all = "snake_case")]   // externally tagged (the TriggerExpr precedent, serde.ml:637)
pub enum ScalarExpr {
    Const(f64), Param(String), QRef(QRef),
    UnOp  { op: UnOp,  arg: Box<ScalarExpr> },
    BinOp { op: BinOp, left: Box<ScalarExpr>, right: Box<ScalarExpr> },
    Cond  { pred: Box<ScalarExpr>, then: Box<ScalarExpr>, else_: Box<ScalarExpr> },
}
pub struct QRef { pub name: String, pub stratum: Vec<StratumKey> }   // carries its resolved cell
```

Externally tagged → derived serde suffices (no hand-written deserializer; trees
are tiny). A `Derived` may `QRef` only **scalar** quantities declared earlier (a
`QRef` to a series quantity, or a forward `QRef`, is `E289`; cycle detection
follows `check_hierarchical_cycles`, `expander.ml:5409` — there is no shareable
bindings-topo helper, as `let` bindings _allow_ forward refs). A stratified
`Derived` may only `QRef` scalars of **matching** strata.

### The flow

```
DSL  quantities { name[idx]? = qexpr }     (parser.mly: IDENT index_bindings_opt EQ expr)
  ▼  expander (mirrors expand_observations ORDER): cartesian_product od.oindices OUTER,
  │     classify+resolve body INNER with per-cell env →
  │       compartment / state arith → State(Expr)            [E288 on a forbidden leaf]
  │       prior scalar quantity     → QRef{name, stratum=cell} [E289 on mix/forward/series]
  │     (a reduction name used in a rate → E290)
  │   → one expanded Quantity leaf per cell, stratum tagged (hoisted stratum_of_bindings)
IR   Model.quantities : Vec<Quantity>      (additive, skip_serializing_if empty,
  │                                          EXCLUDED from Model::hash_into; validated by
  │                                          ir::validate quantity context on every load)
  ▼  serde.ml (`| [] -> []` omit pattern)  ⇄  rust/crates/ir  (schema.json + ir/VERSION 0.20)
RUN  per draw (live, always-fresh path): merge_cell → resolve params →
  │     QuantityEvaluator::eval_draw(params, traj) → QuantityConsumer.push_draw →
  │     finish() → band_with_censoring → quantities/<name>.tsv (sidecar) + manifest
```

### Model — a non-identity section

```rust
pub struct Model {
    // …existing…
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantities: Vec<Quantity>,
}
```

## DSL surface (v1)

```
compartments { S, E, I, R, D }

quantities {
  prevalence       = I / N                  # series
  attack_rate      = final((N0 - S) / N0)   # scalar (absorbing-stock proxy)
  total_deaths     = final(D)               # scalar (absorbing stock)
  peak_prevalence  = max(I / N)             # scalar (value reduction)
  time_to_peak     = time_of_max(I)         # scalar (time reduction → a time)
  takeoff_time     = first_above(I_total, i_thresh)   # onset
  fadeout_time     = last_above(I_total, 0)           # elimination timing
  outbreak_dur     = fadeout_time - takeoff_time      # reduction arithmetic
  positive_months  = count_above(I_total, i_thresh)   # # months above threshold
  person_days_inf  = integral(I)            # scalar (dim P·T)
}

quantities {                                # stratified
  prevalence[p in patch]  = I[p] / N[p]
  peak_time[p in patch]   = time_of_max(I[p])
}
```

**Grammar.** A `quantities {}` top-level block (one new keyword). A decl is
`IDENT
index_bindings_opt EQ expr` — the shared `index_bindings_opt` rule (used
by `obs_decl`/interventions) supplies `[p in patch]`; the body is an `expr`.
`stratum_of_bindings` (`expander.ml:5343`, today a closure inside
`expand_observations`) is hoisted to a shared helper.

**Reduction names** (`final`, `max`, `min`, `mean`, `count_above`,
`count_below`, `time_of_max`, `time_of_min`, `first_above`, `first_below`,
`last_above`, `last_below`, `integral`) are **not lexer keywords** — they lex as
`IDENT` and dispatch **by name in the quantity classifier** (the
`incidence`/`prevalence`/ `observed`/`sum_observed` pattern,
`expander.ml:2431`). `max`/`min` stay **binary pointwise** everywhere
(`expander.ml:2472`, arity `E101`); inside `quantities {}` the classifier
intercepts the **unary** `max(series)` form _before_ delegating to
`resolve_expr` (a 2-arg `max` of two state sub-expressions stays pointwise).

**Body classification** is a **net-new** quantity-aware classifier with its own
symbol table (not the projection disambiguator). It rejects mixed forms with
pointed diagnostics. **New diagnostics** (E280–E287 taken; E288–E290 free):

- `E288` — a forbidden leaf (`dt`/`projected`/aux/per-eval) in a quantity body.
- `E289` — a malformed quantity body: a compartment-minus-quantity mix, a
  reduction on a `Derived` scalar, a forward / series / cross-stratum `QRef`; a
  name colliding with a compartment / param / observation / binding / earlier
  quantity.
- `E290` — a temporal reduction used outside `quantities {}` (e.g. in a rate).

## Evaluation

v1 evaluates on the paths that run every cell fresh; the standalone command,
`batch run`, and cache-hit reuse are the disk-source follow-up.

| Context (CLI)                   | v1 (state)       | v1.1 (`observations.*`) |
| ------------------------------- | ---------------- | ----------------------- |
| `simulate` (fixed params)       | ✓ point          | ✓ point                 |
| `simulate --draws prior`        | ✓ band           | ✓ band                  |
| `fit predict <fit>` (posterior) | ✓ band           | ✓ band                  |
| `batch run`, `camdl quantities` | follow-up (disk) | follow-up               |

**Point vs band is keyed by the param-source kind, not the draw count.** A
fixed-params `simulate` → one realization → a **point** value (a `value`
column). A draws/posterior source → bands (a 1-draw posterior still bands,
degenerate). The consumer threads `ParamSourceKind` to `finish()`.

**Resolution.** State reductions fold over the **output-cadence snapshots**; a
`max`/`time_of_max` between snapshots can be missed at a coarse `every` (a fine
default cadence is documented; a substep fold is a fast-follow). `mean` is an
unweighted average over snapshots (cadence-sensitive, unlike time-weighted
`integral`) — documented. Endpoints (`final`) are cadence-invariant.

## Run-identity, schema, and output identity

**Run-identity.** `run_id` is the **struct walk** (`Model::content_hash` →
`hash_into`, `ir_hash.rs:1053`), and `quantities` is **excluded from that walk**
(the implementer must _not_ add `self.quantities.hash_into(h)`) — a deliberate
exception (`per_eval_bindings` is hashed, `ir_hash.rs:1075`), because a quantity
is a derived report and must never re-key a sim/fit. An **inverse-polarity pin
test** asserts a non-empty `quantities` does **not** change
`Model::content_hash` (the opposite of `ir_per_eval_bindings_changes_hash`,
`runid/src/ir_hash/tests.rs:221`), backed by the `model_golden_hash` tripwire
(adding the walk line moves the golden, even for an empty `Vec`).

**The `ir/VERSION` bump re-keys all run_ids.** The run_id "model" level is
`ModelDigest::content_hash()`, which folds an unskipped `ir_version`
(`inputs.rs`, `resolve.rs:223`, `fit/cas.rs:347`) — so 0.19 → 0.20 re-keys every
run_id, exactly as every prior schema bump did (masked in practice because the
volatile `engine` git hash already re-keys per build). Only the
quantities-block-addition is non-re-keying.

**Schema.** `ir/schema.json` gains the optional `quantities` definition (plus
the new shapes), `ir/VERSION` → 0.20, OCaml (`ir.ml` + `serde.ml`) and Rust `ir`
update atomically. The field is omitted when empty via the OCaml `| [] -> []`
pattern (`serde.ml:1264`), so no existing golden gains a key or shifts a field —
the diff is the **two version strings** per file (`ir_version`;
`validated_by =
"ocaml-compiler-v" ^ ir_version`, `serde.ml:1318`) in both
`ir/golden/` and `ocaml/golden/`, regenerated by `make update-golden`;
`ir/expected/*.tsv` are unaffected.

**Output identity.** Quantities must **not** be written into the `run_id`-keyed
CAS leaf — two models differing only in their `quantities {}` block share a
`run_id`, so leaf bytes would depend on something outside the key (a
clobber/stale bug). v1 writes `quantities/<name>.tsv` as a **regenerated
sidecar**:

- `fit predict` → a `quantities/` subdir of the fit segment (beside
  `predictive/`/`observed/`, `predict.rs:951`).
- `simulate` has **no default user output directory** (the CAS leaf is the
  system of record; `-o` is a single loose file). So a quantities run
  **requires** an output directory (`--out-dir` / an explicit dir), else a hard
  error; quantities land in `<out-dir>/quantities/`.

A content-addressed _report identity_ (`report_id` over `run_id` + the canonical
quantities IR + evaluator/schema/quantile/censoring versions) and report caching
are the **target-state** for the disk-source follow-up — recorded, deferred from
v1; v1 regenerates, never caches.

## Dimensional checking

Each quantity `Expr` runs through the existing `dimcheck`
(`dim_vec = [|p_exp; t_exp|]`, `ocaml/lib/ir/dimcheck.ml`). The reduction's
output dim is derived: `Value` preserves `dim(series)`; `Time` → `T = [|0;1|]`;
`Integral` → `dim(series)·T`. A threshold is dim-checked against the series dim.
`ScalarExpr` typechecks in topological order. The manifest **unit** string is
rendered by a new `(p_exp, t_exp) + model.time_unit → String` function (none
exists today): `[0,0]→
"dimensionless"`, `[1,0]→ "count"`,
`[0,1]→ the model's time unit` (e.g. `"day"`), `[1,1]→ "count·<time_unit>"`,
etc.

## Output format

TSV, **one file per logical quantity** (`quantities/<name>.tsv`), long/tidy
keyed by stratum level. The header is a deterministic function
`header(shape, stratified, censorable, banded)`:

- columns =
  `[time?] [<stratum dims>…] [value | (n_draws [n_value n_censored
  p_censored] q05 q25 q50 q75 q95)]`
- `time` present iff **series**; stratum-dim columns iff **stratified**; the
  censoring trio present iff a **censorable** (`Time`) scalar; `value` (single
  column) iff a **point** (fixed-params) run, else the banded columns.
- **Series never censor** (only `Time` scalars do), so the censoring trio never
  coexists with a `time` column.

```
# series, stratified, banded         quantities/prevalence.tsv
time   patch    n_draws  q05 q25 q50 q75 q95

# Time scalar, stratified, banded     quantities/peak_time.tsv
patch  n_draws  n_value  n_censored  p_censored  q05 q25 q50 q75 q95   # q* = NA if all_censored

# scalar, point (fixed-params run)     quantities/peak_prevalence.tsv
value
```

A `quantities.json` manifest — **one entry per logical quantity** (carrying its
`index_dims`, not a single resolved cell, to match the one-file-per-quantity
TSV):

```json
{
  "schema": "camdl.quantities/v1",
  "quantities": [
    {
      "name": "peak_prevalence",
      "shape": "scalar",
      "source": "state",
      "index_dims": [],
      "reduce": "max",
      "unit": "dimensionless",
      "censoring": null
    },
    {
      "name": "peak_time",
      "shape": "scalar",
      "source": "state",
      "index_dims": ["patch"],
      "reduce": "time_of_max",
      "unit": "day",
      "anchored_as": "date",
      "censoring": { "kind": "right", "conditional_quantiles": true }
    },
    {
      "name": "prevalence",
      "shape": "series",
      "source": "state",
      "index_dims": ["patch"],
      "reduce": null,
      "unit": "dimensionless",
      "censoring": null
    }
  ]
}
```

`time_of_max`/`first_above` return the **first** time on ties (`last_above` the
last); a `Time` value is a duration-from-origin in an unanchored model, rendered
as a **date** when anchored (the manifest records which).

## Worked surveillance questions (v1, state)

```
quantities {
  peak_month     = time_of_max(I_total)                  # latent peak timing
  takeoff_time   = first_above(I_total, i_thresh)        # epidemic onset
  fadeout_time   = last_above(I_total, 0)                # elimination timing
  outbreak_dur   = fadeout_time - takeoff_time           # duration (arithmetic)
  positive_mos   = count_above(I_total, i_thresh)        # # months above threshold
  outbreak_size  = final(N0 - S)                         # the band IS the size distribution
  peak_time[p in patch] = time_of_max(I[p])             # per-patch timing
}
```

Out of v1 (honestly): **the simulated reported signal** (first _detected_ case)
is the v1.1 observation source; **comparing to observed data** (`s(y_obs)`) is
the deferred preconditioning proposal; **post-SIA persistence** (windowed),
**spatial synchrony** (cross-stratum), and **latent cumulative incidence** (flow
counters, which also unlock `total`) are follow-ups.

## Staging

- **v1.** The `quantities {}` block; `State(Expr)` source (validated subset in
  `ir::validate`); typed `TemporalReduce` (minus `Total`) with the censoring
  policy; reduction arithmetic (`ScalarExpr`/`QRef`); stratified output; the
  `QuantityEvaluator` (`sim`) + the per-draw hook in `PredictiveSink` and the
  simulate sink (live, always-fresh paths only); the sidecar output + manifest.
  `ir/VERSION` 0.20; quantities excluded from `run_id`.
- **v1.1.** The `observations.<stream>` source. The DSL accessor is
  `OBSERVATIONS DOT IDENT` (a new `DOT` token; `observations` is already a block
  keyword) lowering to AST `EObsAccess`; the expander classifier admits it only
  as the operand of a single temporal reduction (`max(observations.afp)`,
  `integral(observations.afp)`, `first_above(observations.afp, thr)` …) — a bare
  series, an obs source mixed with state/arithmetic, or a stratified obs source
  is `E289`. It lowers to `QuantitySource::Observation { stream }`, an additive
  IR variant under the existing `QuantitySource` wrapper (no golden churn, no
  run-id re-key beyond the `ir/VERSION` bump).

  The evaluator (`sim`) reduces the per-draw `y_sim` the run already drew — the
  SAME draw, never a fresh one. `eval_draw` takes an `Option<&ObsSeriesSet>`
  (stream name → its `(obs times, y_sim values)`); a `QSource::Observation`
  folds its stream's series at the stream's obs times (thresholds for a Time
  reduction evaluate at the snapshot nearest each obs time). Two materialization
  sites feed it the SAME draws their command publishes, in canonical declaration
  order (no skipped streams, no redraw): `PredictiveSink::merge_cell` captures
  the `y_sim` it already samples for the posterior-predictive bands;
  `simulate`'s `materialize_obs_for_quantities` samples every schedule-bearing
  stream with the cell's `obs_seed` — the same RNG walk `simulate --obs`
  performs. `references_observations()` / `obs_streams()` on the evaluator let a
  caller skip materialization entirely when only state quantities are present.

  The **split** materialization-grid rule: compile-time `E289` for an
  **undeclared** stream (decidable), and a located **runtime** error for a
  referenced stream that cannot be materialized in this command (a fit-only
  stream with no `emit_schedule` under `simulate`) — which streams a run
  materializes is runtime (predict's fit-leaf times, `simulate`'s emit schedule;
  the compiler cannot know it).

Then, ordered by value:

1. **Disk source** — a `TSV → Trajectory` reader (resolving the lossy `traj.tsv`
   view-column / `--no-flows` interaction), then the standalone
   `camdl quantities
   <model> <run>` command, `batch run` quantities (via
   `on_skip` + disk load), and cache-hit reuse; with the content-addressed
   `report_id`.
2. **Windowed quantities** — `over [X, Y]` (post-SIA persistence, seasonal
   burden).
3. **Cross-stratum reductions** — rank / variance / correlation over strata.
4. **Flow / event `cumulative`** (+ `total`) — a lifetime latent flow
   accumulator.
5. **Substep recorder** — resolution-honest `max`/`time_of_max`/`integral`.
6. **Waveform reductions** — peak counting with a prominence threshold.
7. **Survival summaries** — Kaplan–Meier for heavily right-censored `Time`
   quantities.

**Deferred to a preconditioning proposal:** the observed-data realization; a
data-only feature command; the `s(y_obs)` ↔ `s(y_sim)` comparison;
preconditioning identity.

## Decisions recorded

- The seam is a per-draw `(params, traj)` → pure `QuantityEvaluator` (in
  **`sim`**, resolving `Expr`→`ResolvedExpr` once against the compiled model) →
  cli `QuantityConsumer` that bands in `finish`. **v1 evaluates only on the
  always-fresh paths** (`simulate`, `fit predict`); the disk-replay source (a
  standalone command, `batch run`, cache-hit reuse) is a follow-up — it needs a
  `TSV → Trajectory` reader that does not exist today.
- The live hook sits **inside `merge_cell`, composed alongside** the existing
  accumulator — not after the pooled/banded sink, not as a second `RunSink`. The
  two sinks are asymmetric: `PredictiveSink` has the compiled model + resolved
  params; the simulate sink compiles once at construction and resolves params
  per draw.
- **v1 = latent state; v1.1 = the `observations.<stream>` source** (additive on
  the seam, reducing the same `y_sim`); observed-data + preconditioning are a
  separate proposal.
- A quantity's state expression is a **plain `Expr`, validated to a subset in
  the existing `ir::validate::check_expr`** (quantity context: a deep walk
  rejecting `Dt`/`Projected`/`ObsColumnRef`/`PerEvalRef` incl.
  `UncheckedDim.inner`, and a **transitive `BindingRef`** check over
  `model.bindings`) — not a newtype constructor (which serde bypasses and OCaml
  can't express). The OCaml expander gives the friendly located `E288`;
  `ir::validate` is the authoritative backstop. LICM skips quantity bodies.
  `ScalarExpr`, by contrast, **is** structural.
- Enums (`QuantityBody`/`QuantitySource`/`TemporalReduce`/`ScalarExpr`) are
  **externally tagged**, never `untagged`, so v1.1's `Observation` variant is
  purely additive (byte-stable goldens, no run-id move); a round-trip test pins
  it.
- Censoring is **data** (`Value | Censored`, never a `NaN`); non-finite
  arithmetic is an error; `band_with_censoring` partitions then reuses
  `band(&[f64])`, types the all-censored case, and never calls `band(&[])`.
  Series never censor.
- `Total`/`sum` deferred to the flow source (summing a stock is
  cadence-nonsense); `Integral` retained; `last_above`/`count_above` (+ below)
  added.
- **Run-identity:** `quantities` excluded from `Model::hash_into`
  (inverse-polarity pin test); the `ir/VERSION` 0.19 → 0.20 bump re-keys all
  run_ids via `ModelDigest.ir_version` (every schema bump does); goldens
  regenerate the two version strings only (in `ir/golden/` and `ocaml/golden/`),
  no structural diff.
- **Output identity:** quantities are a regenerated sidecar, **not** in the CAS
  leaf; `fit predict` → the fit segment's `quantities/`, `simulate` → a required
  `--out-dir`/`quantities/` (no default user dir); `report_id` + report caching
  is the deferred disk-source target-state.
- Output: header is a function of `(shape, stratified, censorable, banded)`; the
  manifest is one entry per logical quantity with `index_dims`; the `unit`
  string has a specified `(p_exp,t_exp)+time_unit` renderer; point-vs-band is
  keyed by param-source kind.
- DSL: reduction names dispatch by name (not lexer keywords); `max`/`min` stay
  binary; new diagnostics `E288`/`E289`/`E290`; the expander expands strata
  OUTER / resolves bodies+QRefs INNER per-cell (mirroring
  `expand_observations`).
