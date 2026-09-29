# A parameter point is a design coordinate or a sample, and the cell says which

- Date: 2026-09-27
- Status: accepted — every call is recorded in §16; ready to implement
- Relates-to: gh#562, gh#572, gh#575, gh#630, gh#948, gh#949,
  `2026-06-27-sealed-fit-packets-handles-and-override-algebra.md` (§4),
  `2026-08-11-scenario-banding-in-simulate.md` (implemented)
- Builds on: an untracked architecture-review note on gh#562 (2026-08-11). Every
  claim this proposal uses from it is restated in §4; the note is not needed to
  read this document.
- Prerequisite: gh#949 (generated prior/uniform rows carry parameters the
  measure does not sample, so a scenario's value leaks into other arms and
  `--draws-out` exports it) is being fixed separately. §7.2's "varying columns"
  builds on that fix.

All code references are against `origin/main` at `ebe845a0`. A reference marked
_verified_ was read at that line; _inferred_ means reasoned from the code but
not executed. Output shown under **after** is the format this proposal
specifies; it does not exist yet.

## Summary

A run grid in camdl is a product of axes. Some axes index _different things
being compared_ (a scenario, a sweep point); others index _repeated sampling of
one thing_ (a posterior draw, a stochastic replicate). Three decisions depend on
which kind an axis is: whether a scenario may override it, how its cells are
labelled, and whether a quantile may be taken across it. Today none of the three
can read that property off the cell, because the cell does not carry it. Each
decision instead consults `ParamSource` — which CLI flag produced the parameter
points — and the flag is the wrong property. `--draws grid.tsv` classifies a
hand-authored grid as a sample; an iid uniform design is a partition in
`batch run` and a sample in `simulate`.

This proposal replaces `ParamSource` with a `ParamPlan` holding two typed axes —
a **design axis** and a **sample axis** — and splits the cell's single
`point_overrides` map into a design coordinate and a sample coordinate.
Collision policy, labels and provenance, and banding then key off one property
the cell carries, set by one constructor. On the command line the flags come in
axis-named pairs: `--sweep` / `--sweep-file` build the design axis, `--draws` /
`--draws-file` build the sample axis, and a bare `--draws <path>` is refused
naming both file flags. Design coordinates stop entering the process seed, so
design points share common random numbers. The output gains a normative schema:
a `point_id` on every design-bearing row, exact design coordinates, a `band`
column saying what each band is over, and a self-describing manifest.
`fit predict`'s sweep moves onto the grid, so both verbs put their partition
coordinates in the cell. The IR and goldens are untouched.

## 1. How the user works with each axis

Every simulation grid is a product of four axes. The user controls each with
distinct flags or manifest keys, and each plays one role in the output.

| Axis          | `simulate` (after)                                         | `batch run` manifest         | `fit predict`                 | Output role                                       |
| ------------- | ---------------------------------------------------------- | ---------------------------- | ----------------------------- | ------------------------------------------------- |
| **Scenario**  | `--scenario a,b`                                           | `[[scenario]]`               | `--scenario a,b` (+ `fitted`) | group: `scenario` column                          |
| **Design**    | `--sweep P=GRID` or `--sweep-file FILE`                    | `[sweep]` or `[design.NAME]` | `--sweep P=GRID`              | group: `point_id` + `sweep:<param>` columns       |
| **Sample**    | `--draws posterior\|prior\|uniform` or `--draws-file FILE` | none                         | always the fit's posterior    | pooled into the band; `band` column names measure |
| **Replicate** | `--replicates N` or `--seeds a,b,…`                        | `seeds = {…}`                | one per draw                  | pooled into the band                              |

The rule the output follows: **scenario and design make groups (rows); sample
and replicate are pooled into the band within each group.** A quantity file has
one row (or one block of time rows) per group, and its quantile columns
summarize the sample and replicate cells inside that group only.

What stays the same: scenarios, replicates, seeds,
`--draws
posterior|prior|uniform`, `batch run` manifests and their store leaves.
What changes: a bare `--draws FILE` is refused with a message naming
`--sweep-file` and `--draws-file`; `simulate --sweep` and `--sweep-file` are
new; banded files gain a `band` column and design-bearing rows a `point_id` (in
`fit predict` too, as its own commit); a scenario, a `--param`, or a design
value that touches a varying sampled column is recorded; the manifest says what
every entry is.

## 2. Terms

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
mean something: it estimates a quantile of the push-forward of that measure
through the model.

The **varying columns** of a sample are the parameters the measure actually
varies: for a posterior, the estimated parameters; for a prior, the parameters
with a prior; for `uniform`, the bounded parameters actually drawn; for a file,
its non-constant columns (or the list its provenance sidecar records, §12.4).
Other columns a row happens to carry are constants, not part of the sample.

A **marginal reduction** summarizes a set of cells treating them as exchangeable
draws from one distribution — a quantile, a mean. A **paired reduction**
combines cells from _different_ values of a partition axis that share a band
coordinate — draw _i_ of the baseline arm with draw _i_ of the intervention arm
— to construct a new random variable before any marginal summary.

An **imposition** is a value set on a varying sampled column by something other
than the measure — a scenario `set`, a `--param`, or a design coordinate. Every
sample then runs at that value while its other parameters keep their sampled
values. That is an intervention on the sampled parameter, not a conditional of
the measure on it; with correlated parameters the resulting band can be wider or
narrower than the unimposed one. A scenario `scale` on a varying column
**rescales** it: the variation is kept and multiplied.

## 3. The running example

Every workflow in §5 uses one SIR model (compartments S, I, R; S susceptible, I
infectious, R recovered):

```
time_unit = 'days
compartments { S, I, R }
parameters {
  beta  : rate in [0.05, 1.0] ~ log_normal(mu = -1.2, sigma = 0.5)
  gamma : rate in [0.02, 0.5] ~ log_normal(mu = -2.3, sigma = 0.3)
}
init { S = 990  I = 10  R = 0 }
transitions {
  infect  : S --> I  @ beta * S * I / (S + I + R)
  recover : I --> R  @ gamma * I
}
simulate { from = 0 'days  to = 100 'days }
scenarios {
  distancing { scale = { beta = 0.6 } }
  pinned     { set   = { beta = 0.3 } }
}
quantities {
  peak_I  = max(I)
  final_R = final(R)
}
```

Measured on `main` (`ebe845a0`), ODE backend, `dt = 0.1`, `gamma = 0.1`, N =
1000, I₀ = 10, days 0–100: `peak_I` is 70, 304 and 536 at `beta` = 0.15, 0.30
and 0.60. Every other number in §5 is written `…` rather than invented.

## 4. The problem

### 4.1 One field carries two meanings, and the cell does not say which

`CellSpec.point_overrides` (`engine.rs:81`, verified) is the cell's parameter
point. When the job's `ParamSource` is `Sweep` (`sim_job.rs:175`), each point is
a different model — a partition coordinate. When it is `Draws`
(`sim_job.rs:178-192`), each row samples one model — a band coordinate. The same
loop in `plan_grid` (`engine.rs:168-191`, verified) stamps both into the same
field. No function taking only a `&CellSpec` can compute a correct grouping key.

The user-visible consequence, measured on `main`: a three-row file of `beta`
values 0.15, 0.30, 0.60 passed as `--draws grid.tsv --quantities-out q` writes

```
n_draws  q05   q25  q50  q75  q95
3        93.4  187  304  420  512.8
```

for `peak_I` — a "band" over three conditions the user meant to compare. Its
median is the middle condition's value and its tails are interpolations between
conditions; it describes no distribution anyone stated.

### 4.2 `fit predict` keeps its sweep coordinate outside the cell

`fit predict` does not use the grid's sweep axis. It expands `--sweep` itself
(`predict.rs:1590`, verified), then for each sweep point overwrites the swept
parameters in every posterior draw row (`predict.rs:2235-2237`) and runs one
`ParamSource::Draws` job per scenario (`predict.rs:2254`, `:2262`). The sweep
value lives in the loop variable of `for sweep_pt in &sweep_points`
(`predict.rs:2188`), reaches the renderer directly (`predict.rs:2297-2300`), and
keys `FreeForwardCell` (`predict.rs:2340-2342`). Every predict cell looks like a
posterior draw; its partition coordinate is not in the cell, and its collision
check runs separately (`predict.rs:1623-1649`, verified), whose comment says so:
"predict folds the sweep into the draw rows, which the engine sees as generated
draws".

### 4.3 The rejected design, and the four reasons

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
   accumulator whose only input is a whole cell, deriving its own key
   (`main.rs:2464`, verified) — gives the caller no key to pass.
2. **`BandKey::of(&CellSpec)` cannot serve `fit predict`** (§4.2), so a second
   public constructor would be needed.
3. **`PointKind` names the wrong property.** It records which CLI flag produced
   the points, not whether they carry a measure over which a quantile means
   anything. A hand-authored grid passed via `--draws` classifies as a band
   (§4.1 shows the result); the same grid via a sweep as a partition.
4. **It was unreachable:** `BandKey::of` would have returned an empty sweep at
   every live call site.

### 4.4 The landed gh#572 guard keys on the same proxy

gh#572 (a scenario silently overriding a swept or file-supplied parameter) is
fixed on `main`. `engine::check_scenario_coordinate_collision`
(`engine.rs:364-390`, verified; signature `-> Result<(), String>`) is called
from three sites: `run_job` (`engine.rs:203`), `batch::batch_job`
(`batch.rs:605`, so dry-run and status refuse too), and `simulate --dry-run`
(`main.rs:1799`). Its wording comes from one formatter,
`params_resolver::scenario_coordinate_collision` (`params_resolver.rs:842-940`),
which `fit predict` also calls (`predict.rs:1637`). The policy is the
maintainer's: a scenario touching a _user-authored_ coordinate — a sweep, a
design block, a `--draws <file>` — is refused, `set` and `scale` alike;
generated draws (`--draws posterior|prior|uniform`) are not checked.

The guard decides which is which by matching `ParamSource`
(`engine.rs:367-375`): `Sweep` and `Draws { explicit_file: Some(_) }` are
checked, everything else passes. So the correct policy is keyed on the proxy
reason 3 rejects, and it inherits the proxy's blind spots: a `[design.NAME]`
block is reported as "the sweep" (the noun is chosen at
`params_resolver.rs:860`, and the design path builds a `Sweep`,
`batch.rs:1839`), and predict's sweep needs its own call because the engine
cannot see it. The existing test for the design-block refusal
(`tests/scenario_coordinate_collision_gh572.rs:235-264`) pins the "sweep"
wording. This proposal keeps the policy and replaces the key (§8).

## 5. Workflows, end to end

Each workflow gives the modelling question, the commands, the output headers
with example rows, the manifest entry fields that matter (§9.3 defines them
all), and what each relevant collision does. Paths are relative to the
`--quantities-out` directory `q/`. `peak_I` and `final_R` are scalars, so each
file has one row per group; only `peak_I` is shown.

### W1 — Sensitivity sweep: how does the peak depend on `beta`?

**Today** there is no `simulate --sweep`; the nearest spelling is a draws file,
which pools (§4.1). `batch run` sweeps but emits no quantities (follow-up F3).

**After**, three spellings; the first two produce identical files:

```
camdl simulate sir.camdl --params base.toml --sweep beta=0.15,0.3,0.6 \
    --backend ode --dt 0.1 --quantities-out q

camdl simulate sir.camdl --params base.toml --sweep-file grid.tsv \
    --backend ode --dt 0.1 --quantities-out q     # grid.tsv: one column `beta`

# batch.toml
[sweep]
beta = [0.15, 0.3, 0.6]
```

`q/quantities/peak_I.tsv` (point mode — one realization per group):

```
point_id  sweep:beta  value
0         0.15        70
1         0.3         304
2         0.6         536
```

Manifest entry: `"mode": "point"`,
`"partition_columns": ["point_id",
"sweep:beta"]`, `"summary": {"over": [], …}`.

**With `--replicates 20 --backend chain_binomial`**, each design point gets its
own band over process noise, and all three points share the same 20 process
seeds (common random numbers, §7.3):

```
point_id  sweep:beta  band     n_draws  q05  q25  q50  q75  q95
0         0.15        process  20       …    …    …    …    …
1         0.3         process  20       …    …    …    …    …
2         0.6         process  20       …    …    …    …    …
```

`"mode": "banded"`,
`"summary": {"over": ["replicate"], "measure": "process",
"n_samples": null, "replicates_per_sample": 20, …}`.
Because the design index does not enter the seed, adding `beta=0.45` to the
sweep leaves the other three points' trajectories byte-identical.

**Collisions.** `--scenario pinned` (`set beta`) or `--scenario distancing`
(`scale beta`) touches the swept parameter and is refused before any cell runs
(W10). `--param beta=0.5` is refused the same way (W12).
`--sweep gamma=0.05,0.1,0.2 --scenario distancing` touches distinct parameters
and runs: three groups, header `scenario  point_id  sweep:gamma  value` (one
`--scenario` is one arm, so the `scenario` column holds `distancing` on every
row).

### W2 — Space-filling design: where in (`beta`, `gamma`) space is the epidemic large?

**Today and after**, the batch spelling is unchanged:

```
[design.wide]
method = "lhs"
n = 64
parameters.beta  = { range = { min = 0.05, max = 1.0 } }
parameters.gamma = { range = { min = 0.02, max = 0.5 } }
```

It writes `designs/wide/parameter_points.tsv` (a leading `point_id` column,
`batch.rs:1853-1859`, verified) and one store leaf per point; `batch run` emits
no quantities. After, `parameter_points.tsv` is written at round-trip precision.
To get quantities, simulate the generated table as a design:

```
camdl simulate sir.camdl --sweep-file out/designs/wide/parameter_points.tsv \
    --backend ode --quantities-out q
```

```
point_id  sweep:beta           sweep:gamma          value
0         0.4318820513776011   0.1127305823481931   …
…
```

(64 rows; `point_id` is taken from the file, §12.1.) This **recomputes** every
point: `simulate` does not look up the `batch run` leaves, and its seeds differ
from the manifest's explicit seeds unless the same `--seed` is given (§7.3).
Collisions: a scenario touching `beta` or `gamma` is refused, naming "the design
`wide`" from `batch run` and "the sweep file `parameter_points.tsv`" from
`simulate`.

### W3 — Prior predictive and uniform exploration: what does the model predict before seeing data?

**Today and after**, the command is unchanged:

```
camdl simulate sir.camdl --draws prior -n 500 --backend ode --quantities-out q
camdl simulate sir.camdl --draws uniform -n 500 --backend ode --quantities-out q
```

`q/quantities/peak_I.tsv` gains a `band` column:

```
band   n_draws  q05  q25  q50  q75  q95
prior  500      …    …    …    …    …
```

Manifest `summary` for `--draws prior`:

```json
{
  "over": ["sample"],
  "measure": "prior",
  "measure_source": { "kind": "model_ir" },
  "varying": ["beta", "gamma"],
  "n_samples": 500,
  "replicates_per_sample": 1,
  "impositions": []
}
```

For `--draws uniform`, `band` is `uniform_bounds` — quantiles under the uniform
law on the declared bounds, which no one should caption as a credible interval.

**Per-point rows instead of a band.** Export the sample, then read it back as a
design:

```
camdl simulate sir.camdl --draws uniform -n 500 --draws-out u.tsv --backend ode
camdl simulate sir.camdl --sweep-file u.tsv --backend ode --quantities-out q
```

giving `point_id  sweep:beta  sweep:gamma  value`, 500 rows. This works because
`--draws-out` writes only the varying columns once gh#949 lands; a constant
column in a sweep file is refused (§12.1), so an export that still carried
constants would say so rather than silently make every parameter a coordinate.

**Collisions.** `--scenario pinned` over the prior imposes `beta`. It runs, with
one stderr line:

```
note: scenario 'pinned' sets beta=0.3 in every prior draw; other parameters keep
their prior draws (an intervention, not the prior conditional on beta=0.3). To
reduce transmission relative to each draw, scale beta instead.
```

and
`"impositions": [{"param": "beta", "by": "scenario:pinned", "action":
"set"}]`
in that arm's summary. `--scenario distancing` rescales `beta`:
`{"action": "scale"}`, no stderr line.

### W4 — Posterior predictive under interventions: what does distancing do, given what the fit learned?

**Today and after**:

```
camdl simulate sir.camdl --fit results/fits/sir-8a3f12b4 --draws posterior \
    --scenario baseline,distancing --backend ode --quantities-out q
```

```
scenario    band       n_draws  q05  q25  q50  q75  q95
baseline    posterior  200      …    …    …    …    …
distancing  posterior  200      …    …    …    …    …
```

Summary for `distancing`: `"measure": "posterior"`,
`"measure_source":
{"kind": "fit", "handle": "@sir-fit", "run_id": "…", "method": "pgas"}`,
`"varying": ["beta", "gamma"]` (the fit's estimated parameters),
`"impositions": [{"param": "beta", "by": "scenario:distancing", "action":
"scale"}]`.
With `--scenario pinned`, the W3 note is printed with "posterior draw" / "fitted
draws". The draws are paired across arms — draw _i_ uses one parameter vector
and one process seed in both (§6.3).

### W5 — Exporting a posterior and reusing it, with its provenance

**Today**:

```
camdl simulate sir.camdl --fit results/fits/sir-8a3f12b4 --draws posterior \
    -n 200 --draws-out post.tsv
camdl simulate sir.camdl --draws post.tsv --scenario distancing ...
```

The second command is refused, because `distancing` scales the file's `beta`
column and a draws file is a user-authored coordinate (§4.4).

**After**, `--draws-out` writes `post.tsv` and a provenance sidecar
`post.tsv.json` (§12.4), and prints the round-trip hint:

```
draws.tsv: wrote 200 draws to post.tsv (+ post.tsv.json); reuse as a sample with --draws-file post.tsv
```

```
camdl simulate sir.camdl --draws-file post.tsv --scenario distancing \
    --backend ode --quantities-out q
```

runs. The sidecar carries the measure through: `band` is `posterior`, not
`asserted`, and `measure_source` names the original fit. Without a sidecar the
same file reads as `band = asserted`,
`measure_source: {"kind": "file",
"sha256": "…"}`. Passing `post.tsv` as
`--sweep-file` with the same scenario is refused, naming "the sweep file
`post.tsv`"; without a scenario it is refused anyway if the file carries
`chain`/`draw` columns, with a hint to use `--draws-file` (§12.1).

### W6 — `fit predict` over a sweep, the posterior, and scenarios

```
camdl fit predict @sir-fit --sweep gamma=0.08,0.12 --scenario distancing
```

**Today**, `<fit>/quantities/peak_I.tsv` is design-major — sweep point outer,
scenario inner (`predict.rs:2188` then `:2295`), which `StackedQuantities`
preserves because it does not sort:

```
scenario    sweep:gamma  n_draws  rhat  ess  q05  q25  q50  q75  q95
fitted      0.08         200      …     …    …    …    …    …    …
distancing  0.08         200      …     …    …    …    …    …    …
fitted      0.12         200      …     …    …    …    …    …    …
distancing  0.12         200      …     …    …    …    …    …    …
```

**After the output-schema commit** (Stage 4), the header gains `point_id` and
`band`, and swept values print exactly:

```
scenario    point_id  sweep:gamma  band       n_draws  rhat  ess  q05  …
fitted      0         0.08         posterior  200      …     …    …    …
distancing  0         0.08         posterior  200      …     …    …    …
fitted      1         0.12         posterior  200      …     …    …    …
distancing  1         0.12         posterior  200      …     …    …    …
```

**After Stage 7** (the sweep moves onto the grid) the bytes are identical to the
Stage 4 output: rows stay design-major by explicit rendering order. The summary
records the design's imposition on the posterior: `"design_shadows": ["gamma"]`,
and `measure` reads `"posterior"` with
`"measure_note": "posterior of the other parameters with gamma imposed by the
design"`;
one stderr note says the same. `--sweep beta=… --scenario distancing` is refused
before any artifact is written (§14, Stage 7), naming "the sweep".

### W7 — Contrasts: how many infections does an intervention avert?

A contrast is a paired reduction: `fit predict` replays both arms per forkable
posterior draw under one seed, `derive_chain_seed(seed, draw_pos)`
(`contrasts.rs:447`, verified), differences them, then bands the differences
into `contrasts/<name>.tsv` with header `q05 q25 q50 q75 q95 mean n_used` for a
scalar (`contrasts.rs:1240`). Each arm's draw row enters the resolver at tier
3.5 (`contrasts.rs:643`). A contrast needs an arm that toggles an intervention;
a parameter-only scenario such as `distancing` is skipped with a note
(`contrasts.rs:32`, gh#327). So W7 adds an intervention `closure` and a scenario
`close { enable = [closure] }`, and declares
`averted = fitted.quantities.final_R - close.quantities.final_R`.

`camdl fit predict @sir-fit` writes `contrasts/averted.tsv`:

```
q05  q25  q50  q75  q95  mean  n_used
…    …    …    …    …    …     200
```

Under the new coordinates the pairing coordinate has a name — the cell's sample
coordinate — and the rule is: a paired reduction across a partition axis is
valid iff both sides share one sample axis (§6.3). Within a job this always
holds across scenarios.

**With `--sweep` present**, pairing would be within each design point. That is
not delivered, because contrasts fork each draw from its _smoothed_ state,
inferred under the fitted parameters; a sweep point replaces a parameter, so no
smoothed state exists for it — the same reason `conditioned_here` refuses a
conditioned read for any swept cell (`predict.rs:934-943`, verified).
`emit_contrasts` receives no sweep (`predict.rs:2731-2739`, verified), so
contrasts are emitted once, for the un-swept model. After, `fit predict` says
so: a stderr note
(`contrasts are computed for the fitted model only; --sweep is
not applied to them`)
and `"contrasts": {"sweep": "not applied (fitted model
only)"}` in
`report.json`. Contrasts write no manifest of their own (`contrasts.rs` writes
only TSVs; verified by search), so `report.json` is the record. Paired contrasts
within design points, and a per-cell table that makes paired reductions possible
from `simulate`, are follow-up F2.

### W8 — Process noise only: how variable is one parameter set's epidemic?

```
camdl simulate sir.camdl --params base.toml --replicates 50 \
    --backend chain_binomial --quantities-out q
```

```
band     n_draws  q05  q25  q50  q75  q95
process  50       …    …    …    …    …
```

`"summary": {"over": ["replicate"], "measure": "process", "n_samples": null,
"replicates_per_sample": 50, …}`.
`--seeds 1,2,3` is the same with explicit seeds.

Under `--backend ode` the solver draws no randomness, so replicates repeat one
trajectory. A pure-state quantity (`peak_I`, `final_R`) then renders in point
mode — one `value` — with `"over": []`, rather than `n_draws = 50` and
`q05 = q95`. A quantity that reads `observations.<stream>` still varies across
replicates, because each replicate's observation draw uses its own seed
(`main.rs:2447-2460`), and stays banded with `"over": ["replicate"]`. Mode is
therefore decided per quantity (§10). Nothing collides: there are no parameter
coordinates.

### W9 — A hand-authored grid passed the old way

**Today**, `camdl simulate sir.camdl --draws grid.tsv --quantities-out q` writes
the pooled band of §4.1.

**After**, it is refused before any work:

```
error: `--draws grid.tsv` no longer accepts a file: say which axis its rows are.
  --sweep-file grid.tsv  the rows are conditions to compare (a grid, a
                         hand-picked set): one output row per point, reported
                         in `point_id` and `sweep:<param>` columns
  --draws-file grid.tsv  the rows are a sample from a distribution you stand
                         behind (e.g. exported posterior draws): pooled into
                         one quantile band
`--draws` takes only `posterior`, `prior` or `uniform`.
```

A value that is neither a keyword nor an existing path is a different error:
`--draws prio` →
`error: unknown draws source 'prio'; expected posterior, prior
or uniform (did you mean 'prior'?)`.

### W10 — A sweep that a scenario would override

**Today** (landed with gh#572), a manifest sweeping `beta` under `pinned` is
refused by `batch run`, `batch run --dry-run` and `batch status`:

```
error: parameter `beta` is controlled by both the sweep and scenario `pinned`
  sweep:     beta = [0.15, 0.3, 0.6]
  scenario:  set beta = 0.3
The scenario would override every sweep value, so this sweep would not vary `beta`.
Fix: remove `beta` from the sweep, or use a scenario that does not touch it.
```

(Layout inferred from the formatter, `params_resolver.rs:842-940`; the column
width is `max(len(label), len("scenario:")) + 2`.) **After**, the message names
the design's origin — "the sweep" for `[sweep]` and `--sweep`, "the design
`wide`" for `[design.wide]`, "the sweep file `grid.tsv`" for `--sweep-file` —
and the `scale` case names the reparameterization route:

```
error: parameter `beta` is controlled by both the sweep and scenario `distancing`
  sweep:     beta = [0.15, 0.3, 0.6]
  scenario:  scale beta × 0.6
The scenario would rescale every sweep point of `beta`; composing a sweep with a
scenario that scales the same parameter is not supported.
Fix: factor the reduction into its own parameter (e.g. `contact_mult`, with
`beta * contact_mult` in the rate) and have the scenario scale that, or remove
`beta` from the sweep.
```

The draws-file wording is deleted: a sample never collides (W3–W5). Composition
of `scale` with a sweep is follow-up F1.

### W11 — `--design-from`: synthetic datasets on a fit's observation design

`--design-from fit.toml` writes a dataset — one file per observation stream on
the fit's own observed rows — at each parameter vector; with `--draws` it writes
one `ds_NN/` per row (`args/mod.rs:470-489`; `simulate_on_bound_design`,
`main.rs:3248`, which builds its own `SimRun`s and seeds at
`main.rs:3275-3276`). It bypasses the engine, and it ignores `--scenario`
entirely (gh#948, a separate defect).

**After**, it accepts every parameter source —
`--draws posterior|prior|uniform`, `--draws-file`, `--sweep-file`, `--sweep` —
and still writes one dataset per parameter point, because it never reduces. With
`--draws prior`, the priors and the `[fixed]` block come from the design
`fit.toml` itself; a `--fit` naming a different file is refused. Beside the
datasets it writes `truths.tsv` — one row per dataset: `dataset`, `point_id`
(design) or `sample_idx` (sample), then every parameter at round-trip precision
— and `design_from.json`:

| Field            | Type                 | Meaning                                                                    |
| ---------------- | -------------------- | -------------------------------------------------------------------------- |
| `schema`         | string               | `"camdl.design_from/v1"`                                                   |
| `axis`           | `"design"\|"sample"` | how the parameter points were read                                         |
| `measure`        | string or null       | `posterior\|prior\|uniform_bounds\|asserted` for a sample; null for design |
| `measure_source` | object or null       | as in §9.3; for `prior`, `{"kind": "fit_toml", "path_sha256": "…"}`        |
| `n_datasets`     | integer              | number of `ds_NN/` directories                                             |
| `truths`         | string               | `"truths.tsv"`                                                             |

The record matters downstream: a simulation-based calibration pools rank
statistics across datasets and is valid only when the truths were a sample from
the prior the fit uses (`axis: sample`, `measure: prior`); a recovery study at
chosen truths (`--sweep-file truths.tsv`) must not be pooled that way. Once
gh#948 routes `--scenario` through, the §8 guard applies unchanged.

### W12 — `--param` on a swept parameter

```
camdl simulate sir.camdl --sweep beta=0.15,0.3,0.6 --param beta=0.5 ...
```

**Today** this cannot be written in `simulate` (no `--sweep`); with a draws file
the analogue runs, because `--param` sits at tier 5 above the draw/sweep tier
(`util.rs:3244-3253`) and wins silently: three identical runs labelled with
three values. **After**, `--param` and expanded `--param-vec` entries are part
of the collision footprint (§8): on a design name they are refused —

```
error: parameter `beta` is controlled by both the sweep and --param
  sweep:     beta = [0.15, 0.3, 0.6]
  --param:   beta = 0.5
--param would override every sweep value, so this sweep would not vary `beta`.
Fix: remove `beta` from the sweep, or drop --param beta.
```

— and on a varying sampled column they are recorded as an imposition
(`"by": "--param"`) with the W3 stderr note.

### W13 — A draws file with a provenance sidecar, round trip

```
camdl simulate sir.camdl --fit results/fits/sir-8a3f12b4 --draws posterior \
    -n 1000 --draws-out post.tsv
camdl simulate sir.camdl --draws-file post.tsv -n 200 --quantities-out q
```

The first command writes `post.tsv` (varying columns only) and `post.tsv.json`.
The second reads both: `-n 200` subsamples the file strided across all rows
(today `-n` is ignored for a file, so a raw 60k-row `draws.tsv` replays every
row — the gh#630 hazard); `band` is `posterior`; the summary's `n_samples` is
200 and `measure_source` names the original fit and records
`"subsample": {"from": 1000, "stride": 5}`. If `sir.camdl` has changed since the
export, the IR hash in the sidecar differs and a warning says so; parameters the
file does not carry are listed in `"from_outside_file": [...]` with where their
values came from (`--params`, `--fit` `[fixed]`, model default).

## 6. The rule: design coordinates condition, samples are marginalized

### 6.1 The kind is a declared reading, and only a sample needs a measure

The kind is not a property of the numbers. An iid uniform draw can be read as a
sample of the uniform law (to band a quantity over it) or as a set of conditions
(to plot the quantity against the parameter). What differs is the _question_ —
condition on the point or marginalize over it. The asymmetry that makes this
safe: **any point set may be read as a design; only a point set with a measure
may be read as a sample.** A sweep grid has no measure, so it can never be a
sample. A generated draw has a measure by construction. A file has one only if
the user asserts it, which is what `--draws-file` means (or its sidecar
records).

| Source                                      | Axis   | Measure                                    |
| ------------------------------------------- | ------ | ------------------------------------------ |
| `--draws posterior`, `fit predict`          | sample | the fit's posterior                        |
| `--draws prior`                             | sample | prior (model IR or fit.toml)               |
| `--draws uniform`                           | sample | uniform on the declared bounds             |
| `--draws-file FILE`                         | sample | from `FILE.json` if present, else asserted |
| `--sweep`, `[sweep]`, `fit predict --sweep` | design | —                                          |
| `--sweep-file FILE`                         | design | —                                          |
| `[design.NAME]`                             | design | —                                          |

`--draws uniform` is an iid sample, not a space-filling design:
`generate_uniform_draws` draws `lo + (hi − lo)·U` independently per parameter
per row (`main.rs:3860-3880`, verified), holding unbounded parameters at their
default. The run spec calls it "space-filling exploration"
(`docs/camdl-run-spec.md:1145-1148`); Stage 5 corrects that sentence. The actual
space-filling designs (`[design.NAME]`, `method = "sobol" | "lhs" |
"random"`,
`batch.rs:98-107`) already run as `ParamSource::Sweep` (`batch.rs:1839`) —
reason 3 in running code.

### 6.2 What each kind implies for the three decisions

| Decision                               | Design coordinate                                      | Varying sampled column                                                        |
| -------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| A scenario or `--param` sets/scales it | Contradiction — refused, naming the design's origin    | Imposition — allowed; recorded in `summary.impositions`, stderr note on `set` |
| A design coordinate sets it            | —                                                      | Imposition — allowed; recorded in `summary.design_shadows`, stderr note       |
| Labels and provenance                  | `point_id` + `sweep:<param>` columns, manifest `sweep` | Not per row; `band` column and manifest name the measure                      |
| Marginal reduction (quantile, mean)    | Never crosses it                                       | Crosses it                                                                    |
| Paired reduction                       | Crosses it, matched on the band coordinate             | Supplies the matching coordinate                                              |

A constant (non-varying) column of a sample row is neither: a scenario or
`--param` over it changes a constant, as over `--params`, and is not recorded.

The collision row is the landed gh#572 policy on every input it covers: sweeps,
design blocks and sweep files are refused, generated draws allowed. The one
input whose treatment changes is a file read with `--draws-file`, now allowed
under a scenario (W5) — the file carries a declared measure, so a scenario over
it is an intervention, exactly as over the generated draws it came from.

### 6.3 Marginal reductions stay within a group; paired reductions cross groups

> A marginal reduction may cross band axes only. A partition axis is eliminated
> only by an explicit measure over its levels (none exists in camdl) or by a
> paired reduction, which constructs a new random variable per band coordinate
> before any marginal summary.

A paired reduction across a partition axis is well-defined iff the cells on both
sides share one `SampleAxis` (same rows, same measure): the band coordinate then
denotes the same parameter vector on both sides. Pairing across scenarios always
satisfies this (a job has one sample axis, and the process seed excludes the
scenario, `engine.rs:11-23`). Pairing across design points satisfies it when the
design is crossed with a sample axis; after this proposal it also enjoys common
random numbers there, because the design index no longer enters the seed (§7.3).

### 6.4 Out of scope

**Weighted designs.** A `[design.NAME]` parameter may carry a `prior` for
importance weighting by `camdl voi` (`batch.rs:111-117`). camdl has no weighted
quantile path and `batch run` emits no quantities, so design blocks are design
axes; a weighted-sample reading waits for a consumer.

**`fit run --sweep`** fits once per point — not a simulation grid — and is
untouched. `fit`, `profile`, `eval`, `pfilter` and `survey` resolve with
`point_overrides: &[]` (`fit/runner.rs:319`, `profile.rs:554`, `eval.rs:113`,
`pfilter.rs:218`, `survey.rs:757`, `:969`, verified) and are unaffected.

## 7. Types

### 7.1 The plan: a design axis and a sample axis

`ParamPlan` replaces `ParamSource` in `SimulateJob.source` (`sim_job.rs:70`).
Each axis is a sum type whose unit case is its own variant, so an axis with no
points but an origin, or a sample without a measure, cannot be written down. The
replicate count is not part of the plan: it is derived from `Seeds` (explicit
seed-list length, else `--replicates`), as `effective_replicates` already does
(`engine.rs:325-334`); storing it twice is how `batch run` came to pass
`replicates: 1` beside an explicit seed list.

```rust
// sim_job.rs

/// The parameter-point structure of a job: a design axis crossed with a
/// sample axis.
#[derive(Debug, Clone)]
pub struct ParamPlan {
    design: DesignAxis,
    sample: SampleAxis,
}

/// Labelled conditions the user chose. A partition axis.
#[derive(Debug, Clone)]
pub enum DesignAxis {
    Unit,
    Points {
        /// The design's parameter names, sorted. Every point sets exactly these.
        names: Vec<String>,
        points: Vec<DesignPoint>,
        origin: DesignOrigin,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct DesignPoint {
    /// The point's stable id: the file's `point_id` column if present, else
    /// the 0-based position in the axis.
    point_id: u64,
    /// Optional string label carried from a `--sweep-file` `label` column.
    label: Option<String>,
    /// Requested values, aligned to `names`.
    values: Vec<f64>,
}

/// Where a design came from — the noun the collision diagnostic uses.
#[derive(Debug, Clone)]
pub enum DesignOrigin {
    Sweep,                               // --sweep, [sweep], fit predict --sweep: "the sweep"
    Design { method: DesignMethod },     // [design.NAME]: "the design `NAME`"
    File { path: PathBuf },              // --sweep-file FILE: "the sweep file `FILE`"
}

/// Exchangeable draws from a stated measure. A band axis.
#[derive(Debug, Clone)]
pub enum SampleAxis {
    Unit,
    Draws {
        rows: Vec<IndexMap<String, f64>>,
        measure: Measure,
        /// The parameters the measure varies (§2), computed at construction.
        varying: Vec<String>,
    },
}

/// What the sample is a sample of.
#[derive(Debug, Clone, PartialEq)]
pub enum Measure {
    Posterior { fit: FitRef, method: String },   // fit handle + run id + stage
    Prior { source: PriorSource },               // model IR or fit.toml (+ content hash)
    UniformBounds,                               // lo + (hi-lo)·U on declared bounds
    Asserted { path: PathBuf, sha256: String },  // --draws-file FILE, no sidecar
}
```

`DesignMethod` carries the block name and the method (`sobol | lhs | random`).
Fields are private to `sim_job`; the constructors are the only way in, each
validating its invariant:

```rust
impl ParamPlan {
    pub fn point() -> Self;
    /// Refuses a design name the model does not declare.
    pub fn new(design: DesignAxis, sample: SampleAxis, model: &ir::Model)
        -> Result<Self, String>;
}
impl DesignAxis {
    /// From the CLI grammar shared with `fit predict` (§7.5): Cartesian product,
    /// sorted by parameter name; a repeated name is refused.
    pub fn sweep(specs: &[crate::args::types::SweepSpec]) -> Result<Self, String>;
    /// From already-expanded rows: `[sweep]`, `[design.NAME]`, `--sweep-file`.
    /// Refuses an empty row set and ragged rows. Duplicate rows are refused
    /// for `File` only (D-E, §12.1).
    pub fn from_rows(rows: Vec<DesignRow>, origin: DesignOrigin)
        -> Result<Self, String>;
}
impl SampleAxis {
    /// Refuses an empty row set. `varying` is supplied by the source (the fit's
    /// estimated set, the parameters with a prior, the bounded parameters, the
    /// sidecar's list) or computed as the non-constant columns of a file.
    pub fn draws(rows: Vec<IndexMap<String, f64>>, measure: Measure, varying: Vec<String>)
        -> Result<Self, String>;
}
```

Duplicate points in a generated grid (`--sweep beta=0.1,0.1`, a `[sweep]` list
with a repeat, `fit predict --sweep`) are accepted as today: they are two groups
whose coordinates coincide, and under effective-value identity (§13) their cells
resolve to one store leaf, simulated once and reported twice.

A design name may also be a varying sample column: `fit predict --sweep gamma`
crosses a `gamma` design with posterior rows that vary `gamma`. The design value
wins (today's row overwrite, `predict.rs:2235-2237`) and is recorded as a design
shadow (§6.2). §7.4 makes the precedence a resolver tier.

### 7.2 Varying columns are computed once, at the boundary

Today every generated row carries every parameter: `uniform` fills unbounded
parameters with their default (`main.rs:3874-3890`); `prior` fills parameters
without a prior with their default or a scenario's value (`main.rs:4131-4143`,
`:4190-4204`); a posterior `draws.tsv` carries the fit's `[fixed]` columns.
gh#949 stops generated rows from carrying what the measure does not sample.
`varying` is the list this proposal reads, wherever it matters: the collision
recording (§8), the stderr note, `summary.varying`, and `summary.over` (a
scenario that sets every varying column leaves nothing to pool over the sample).

| Measure         | `varying`                                                           |
| --------------- | ------------------------------------------------------------------- |
| `Posterior`     | the fit's estimated parameters, read from the fit config/metadata   |
| `Prior`         | parameters with a prior in the prior source                         |
| `UniformBounds` | parameters with declared bounds (the ones actually drawn)           |
| `Asserted`      | the sidecar's list if present, else the file's non-constant columns |

### 7.3 Design coordinates never enter the process seed

```rust
impl CellCoords {
    fn seed_index(&self) -> usize {
        match self.sample {
            SampleCoord::Draw { idx } => idx,
            SampleCoord::Unit => 0,
        }
    }
}
```

`process_seed_for` (`engine.rs:52-66`) receives `seed_index` where it receives
`point_idx` today. Design points therefore share process seeds (common random
numbers), matching `batch run` (explicit seeds, `batch.rs:596`, where the index
is unused, `engine.rs:59-60`) and `fit predict` (the seed follows the draw
index). Consequences, job by job:

- `--draws posterior|prior|uniform` and `--draws-file`: sample index = today's
  `point_idx`; every trajectory is unchanged. `--draws-file FILE` has exactly
  the seeds of today's `--draws FILE`.
- `--sweep` and `--sweep-file` are new; they have no runs to preserve. Every
  design point uses `seed_index = 0`, so inserting or removing a point leaves
  the other points' trajectories unchanged.
- The same file passed as `--sweep-file` and as `--draws-file` runs at different
  seeds (0 for every row vs the row index), so the two do not share store leaves
  beyond row 0; their ensembles are keyed apart as well (§7.6).
- `fit predict --sweep` after Stage 7: sample index = draw index, as inside
  today's per-sweep-point jobs. The `total_runs == 1` branch (`engine.rs:61-62`)
  no longer fires on the combined job, but `mix_cell_seed(base, 0, 0) == base`
  (`util.rs:35-37`; asserted at `engine.rs:520`), so the seed is unchanged.

`seed_index` also replaces `point_idx` at the per-draw lookups that follow the
sample: the `--init-state fit` row (`engine.rs:452-455`),
`ConditionedSource::per_draw` (`predict.rs:1187`), `chain_of_point`
(`predict.rs:1262`), and `RunEntry.draw_idx` (`batch.rs:1498`, `:1707`), which
feeds the `SimEnsemble` grid digest (`sim_ensemble_cas.rs:19-24`, `:120-133`,
`:192`; `main.rs:2680`).

### 7.4 Resolver tier 3.5 splits in two, design over sample

Tier 3.5 (`params_resolver.rs:1095-1118`) becomes 3.5a (sample row) then 3.5b
(design point), and `ValueSource::SweepPoint` (`params_resolver.rs:193-196`)
splits into `ValueSource::Sample` and `ValueSource::DesignPoint`.
`SimRun.point_overrides` (`engine.rs:469`) splits into `sample_overrides` and
`design_overrides`. Resolved values are unchanged for every current job: the
only job with both axes is `fit predict --sweep`, whose design value already
overwrote the sample value.

The tag is not persisted anywhere a user reads: `run.json.inputs` has no
provenance tag (`resolve.rs:238-245`), and every fitting and filtering path
resolves with `point_overrides: &[]` (§6.4), so `parameters_provenance` never
holds `sweep_point`. The tag surfaces only in error text: `UnknownParameter`
prints `from SweepPoint` via `Debug` (`params_resolver.rs:375-381`), and
`NonFiniteValue` suggests `--fixed` (`:393-400`), which `simulate` users spell
`--param`. Stage 8 rewrites both:
`column`betta`of`post.tsv`is not a
parameter of sir.camdl`, and a fix hint
naming the flag the command actually has.

`SimRun` has construction sites beyond `build_cell_sim_run`: `main.rs:1236` (the
base run), `main.rs:3275` (`--design-from`), `util.rs:3166`,
`util.rs:3259-3285`, `util.rs:4427`, and the contrasts arm resolver passes draw
rows at tier 3.5 (`contrasts.rs:643`). Each is classified in Stage 2; the
exhaustive destructure at `batch.rs:2888` makes the compiler list any that is
missed.

### 7.5 The cell: coordinates replace `point_idx` and `point_overrides`

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
}

pub struct Grid {
    pub n_scenarios: usize,
    pub n_design: usize,     // 1 for DesignAxis::Unit
    pub n_samples: usize,    // 1 for SampleAxis::Unit
    pub n_replicates: usize,
    pub total_runs: usize,
    pub parallel: usize,
}
```

`CellSpec`'s fields become private to `engine`, read through accessors, and
`plan_grid` (`engine.rs:148-193`) is the single constructor. `cli` is a binary
crate, so module privacy is enforceable; test fixtures (`batch.rs:2771`,
`:2849`) use a `#[cfg(test)]` constructor in `engine`. The one non-test
constructor outside `plan_grid` is `CasSink::predict_cells`
(`batch.rs:1272-1307`, verified), which re-implements the grid loop for dry-run
cache classification; it calls `plan_grid` instead (Stage 0). `plan_grid`
iterates `scenario → design point → sample → replicate`; with either axis `Unit`
this is today's `scenario → point → rep`. `Grid` (`engine.rs:105-111`) gains the
per-axis sizes the banner, the dry run and the combined writers need (§12.3,
§9.2).

### 7.6 Parse at the boundary, and who reads what

| Front end                 | Today                                                              | After                                                                           |
| ------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `simulate --draws …`      | `main.rs:1735-1751`: `Draws { explicit_file }` or `Point`          | `SampleAxis::draws(rows, Posterior \| Prior \| UniformBounds, varying)`         |
| `simulate --draws-file`   | (was `--draws FILE`)                                               | `SampleAxis::draws(rows, sidecar measure \| Asserted, varying)`                 |
| `simulate --sweep-file`   | —                                                                  | `DesignAxis::from_rows(rows, File { path })` (reader §12.1)                     |
| `simulate --sweep`        | —                                                                  | `DesignAxis::sweep(&specs)`                                                     |
| `batch run [sweep]`       | `batch.rs:629-644` (`sweep_source`): `Sweep` or `Point`            | `DesignAxis::from_rows(points, Sweep)`                                          |
| `batch run [design.NAME]` | `batch.rs:1839`: `Sweep`                                           | `DesignAxis::from_rows(points, Design { method })`                              |
| `fit predict`             | `predict.rs:2254`: `Draws { explicit_file: None }` per sweep point | one plan: `DesignAxis::sweep(&args.sweep)` × `SampleAxis::draws(.., Posterior)` |

`simulate --sweep` reuses the CLI grammar already shared by `fit run`,
`fit
predict` and `profile`: `args::types::SweepSpec` (`args/types.rs:255`,
`V1,V2,…
| lin(min,max,n) | log10(min,max,n)`). Predict's expansion,
`expand_predict_sweep` (`predict.rs:1359-1380`, Cartesian, sorted by name), and
its duplicate-name and unknown-name checks (`predict.rs:1594-1621`) move into
`DesignAxis::sweep` and `ParamPlan::new`, so both verbs expand and validate
through one function. The batch TOML sweep grammar (`batch.rs:143`,
`linspace`/`logspace`/`range`) is a different surface syntax and stays; its
expansion (`expand_sweep`, `batch.rs:293`) feeds `DesignAxis::from_rows`. A
zipped (non-Cartesian) design is a `--sweep-file`. The `explicit_file` field
(`sim_job.rs:182-191`) is deleted.

| Consumer                                                                   | Reads                                                              | Replaces                                                                             |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Collision guard (§8)                                                       | plan design names/origin, varying, `fixed_cli`                     | `ParamSource` match at `engine.rs:367-375`; predict's call at `predict.rs:1623-1649` |
| `SimQuantities::push_cell` (`main.rs:2464`)                                | `cell.spec.coords().partition_key()`                               | `cell.spec.scenario.name()`                                                          |
| `PredictiveSink::merge_cell` (`predict.rs:1127`)                           | same                                                               | same, plus the enclosing `sweep_pt` loop variable                                    |
| `FreeForwardCell`, `assemble_predictive` (`predict.rs:2340`, `:2489-2493`) | `PartitionKey` + `DesignPoint`                                     | the `sweep_pt` loop variable                                                         |
| Quantity mode (`main.rs:1931-1940`)                                        | per-quantity rule (§10)                                            | `matches!(source, Point { replicates: 1 })`                                          |
| `CasSink::cell_resolve` label (`batch.rs:1150`)                            | design names ∪ varying columns                                     | `point_overrides.keys()`                                                             |
| Seed and per-draw lookups (§7.3)                                           | `coords.seed_index()`                                              | `point_idx`                                                                          |
| `RunEntry.draw_idx` → `SimEnsemble` grid (§7.3)                            | sample index; axis kind folded into the grid level for design jobs | `point_idx`                                                                          |
| Wide trajectory / obs writers (`main.rs:2894`, `:2931`, `:3060`, `:3094`)  | `CellCoords`, `Grid` (§9.2)                                        | `draw = point_idx + 1`, `n_draws = grid.n_points`                                    |
| Contrasts arm resolver (`contrasts.rs:643`)                                | sample row at tier 3.5a                                            | draw row at tier 3.5                                                                 |

The `SimEnsemble` grid level folds the axis kind for design jobs so that
`--sweep-file X` and `--draws-file X` cannot produce one ensemble `run_id` over
different bytes (the two write different combined-file columns, §9.2).

## 8. Collision policy keys on the coordinate kind

`check_scenario_coordinate_collision` keeps its name and its three call sites
(§4.4), plus the `--event-log` path, which calls `plan_grid` directly
(`main.rs:1978`) and so bypasses the guard today. Its signature changes from
`Result<(), String>` to

```rust
pub fn check_scenario_coordinate_collision(job: &SimulateJob)
    -> Result<Vec<Imposition>, String>
```

The footprint of each overriding source is intersected with the plan:

- **Sources:** each scenario's `scenario_param_footprint`
  (`params_resolver.rs:698-733`), and the job's tier-5 entries — `--param` and
  the expanded `--param-vec` names (`util.rs:3244-3253`, the list the resolver
  applies at tier 5).
- **Against design names** — non-empty is a hard error. `UserCoordinate`
  (`params_resolver.rs:806`) becomes a view of `DesignOrigin` plus the tier-5
  source, so the formatter names the origin ("the sweep", "the design `wide`",
  "the sweep file `grid.tsv`") and the overrider (scenario `pinned`, or
  `--param`). The `DrawsFile` variant and its wording are deleted. The `scale`
  case names the reparameterization route (W10).
- **Against varying sample columns** — allowed; each hit is returned as an
  `Imposition { param, by: Scenario(name) | FixedCli, action: Set | Scale }`,
  which `run_job` prints (one note per `set`, W3 wording) and the quantity
  accumulators copy into `summary.impositions`.
- **Design names against varying sample columns** — allowed; returned as design
  shadows (§6.2), printed once, and copied into `summary.design_shadows`.

Because `fit predict --sweep` builds a plan with a design axis after Stage 7,
the engine's guard covers it; predict nevertheless keeps an explicit pre-flight
call before its free-forward closure, because inside that closure an error is
caught and reported as "continuing with what is computable"
(`predict.rs:2377-2382`) while other outputs are still written. The
footprint∩names rule is exact, not a stopgap: `2026-08-11` §1.1 shows
disjointness is required for both distinctness and denotation.

## 9. Output schema

This section is normative. It applies to `simulate --quantities-out`,
`fit predict`, and the combined trajectory/observation files of `simulate`.

### 9.1 Quantity TSV columns

Columns appear in this order; each group is present only when its condition
holds.

| #  | Column(s)                             | Present when                                                                      |
| -- | ------------------------------------- | --------------------------------------------------------------------------------- |
| 1  | `scenario`                            | `simulate`: any `--scenario` given; `fit predict`: always                         |
| 2  | `point_id`                            | the job has a design axis                                                         |
| 3  | `sweep:<param>` …                     | the job has a design axis; one per design name, sorted                            |
| 4  | `time`                                | the quantity is a series                                                          |
| 5  | `<dims>` …                            | the quantity is stratified                                                        |
| 6  | `band`                                | banded mode; one of `posterior \| prior \| uniform_bounds \| asserted \| process` |
| 7  | `n_draws`                             | banded mode                                                                       |
| 8  | `rhat`, `ess`                         | banded mode with a chain partition (`fit predict`)                                |
| 9  | `n_value`, `n_censored`, `p_censored` | banded mode, censorable scalar                                                    |
| 10 | `q05 q25 q50 q75 q95`                 | banded mode                                                                       |
| 10 | `value`                               | point mode                                                                        |

Examples: point, design, no scenario — `point_id  sweep:beta  value`; banded,
scenario, sample — `scenario  band  n_draws  q05 … q95`; banded, scenario,
design, sample, chains —
`scenario  point_id  sweep:gamma  band  n_draws  rhat
ess  q05 … q95`.

`band` names the measure when the sample axis is reduced, else `process`. It is
constant within a file. `sweep:<param>` values are written with Rust's shortest
round-trip `f64` formatting (`{}`), never through `quantile::fmt_value` (6
decimals), so a coordinate joins exactly against the grid that produced it. An
indexed parameter flattens as it does elsewhere in camdl (`beta_1`, no
brackets); the manifest's `sweep` object additionally records
`{"family": "beta", "index": [1]}` for such entries.

Readers: the colon in `sweep:beta` requires backticks in R's `readr::read_tsv`
(`` df$`sweep:beta` ``) and becomes `sweep.beta` under `read.delim`; polars and
pandas keep it verbatim. The prefix is kept for every design origin because it
is already `fit predict`'s published header.

Rows are ordered design-major: for each design point in axis order, each
scenario in `--scenario` order, then time and strata. This is `fit predict`'s
order today (`predict.rs:2188` → `:2295`); `simulate` adopts it.

### 9.2 Combined trajectory and observation files

The wide writer (`main.rs:2884-2900`) writes `replicate` (the global cell
ordinal `run_idx + 1`, when `total_runs > 1`), `scenario` (when
`n_scenarios > 1`), `draw` (when `n_draws > 1`), then time and state columns;
observation rows mirror it (`main.rs:3060`, `:3094`, `:3158`). After:

| Column            | Present when                          | Value                           |
| ----------------- | ------------------------------------- | ------------------------------- |
| `replicate`       | `total_runs > 1` (unchanged)          | global cell ordinal (unchanged) |
| `scenario`        | `n_scenarios > 1` (unchanged, gh#576) | scenario name                   |
| `point_id`        | design axis present                   | the design point's id           |
| `sweep:<param>` … | design axis present                   | exact coordinate                |
| `draw`            | sample axis with more than one row    | sample index + 1                |

A design job with no sample axis writes no `draw` column; today's `--draws FILE`
output (now `--draws-file`) is byte-identical.

### 9.3 Manifest fields

Every entry of `quantities.json` carries:

| Field                                                                  | Type                | Req.                          | Meaning                                                                              |
| ---------------------------------------------------------------------- | ------------------- | ----------------------------- | ------------------------------------------------------------------------------------ |
| `name`, `shape`, `source`, `index_dims`, `reduce`, `unit`, `censoring` | as today            | yes                           | unchanged (`quantity_output.rs:466-474`)                                             |
| `mode`                                                                 | `"point"\|"banded"` | yes                           | how this entry's file is rendered                                                    |
| `partition_columns`                                                    | array of string     | yes                           | the grouping columns present, in order (e.g. `["scenario","point_id","sweep:beta"]`) |
| `scenario`                                                             | string              | when a scenario column exists | this group's scenario (as today)                                                     |
| `sweep`                                                                | object              | when a design axis exists     | `{param: value}` exact; indexed params also `{family, index}`                        |
| `point_id`                                                             | integer             | when a design axis exists     | this group's design point                                                            |
| `summary`                                                              | object              | yes                           | below                                                                                |

`summary`:

| Field                   | Type                             | Req.                            | Meaning                                                                                                                                                                      |
| ----------------------- | -------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `over`                  | array of `"sample"\|"replicate"` | yes                             | band axes reduced in this group; `[]` in point mode                                                                                                                          |
| `measure`               | string                           | when `over` is non-empty        | `posterior \| prior \| uniform_bounds \| asserted \| process`                                                                                                                |
| `measure_note`          | string                           | no                              | e.g. "posterior of the other parameters with gamma imposed by the design"                                                                                                    |
| `measure_source`        | object                           | when `measure` is not `process` | `{"kind":"fit","handle","run_id","method"}`, `{"kind":"model_ir","ir_hash"}`, `{"kind":"fit_toml","path_sha256"}`, `{"kind":"file","sha256"}`, plus `subsample` when strided |
| `varying`               | array of string                  | when a sample axis exists       | §7.2                                                                                                                                                                         |
| `n_samples`             | integer or null                  | yes                             | rows on the sample axis; `null` when there is none                                                                                                                           |
| `n_samples_used`        | integer                          | when rows can be dropped        | rows actually reduced (e.g. conditioned subsets)                                                                                                                             |
| `replicates_per_sample` | integer                          | yes                             | derived from `Seeds` (§7.1)                                                                                                                                                  |
| `impositions`           | array of object                  | yes                             | `{param, by, action}` on varying columns (§8); `[]` if none                                                                                                                  |
| `design_shadows`        | array of string                  | yes                             | design names that are varying sample columns; `[]` if none                                                                                                                   |

`measure_source` records a fit's handle, content hash (`run_id`) and method,
never a filesystem path, so a manifest stays valid when results move. The field
is named `measure_source` because `source` is already an entry key (the
quantity's source, `quantity_output.rs:466-474`).

## 10. Banding keys on the coordinate kind, per quantity

`SimQuantities.by_scenario` becomes
`by_partition: IndexMap<PartitionKey,
ScenarioQuant>`, keyed inside `push_cell`
from the cell. At render each group is one `StackedQuantities::push_group` whose
`DesignCoords` come from one function,
`DesignCoords::of(&PartitionKey, &ParamPlan, scenario_axis)`. `DesignCoords`
already renders `sweep:<param>` columns (`quantity_output.rs:227-243`) and a
manifest `sweep` object (`:480-491`); it gains `point_id`.

Mode is decided **per quantity**, from what its band would actually reduce:

```rust
fn quantity_mode(plan: &ParamPlan, reps: usize, backend: ForwardBackend, q: &Quantity) -> Mode {
    let sample_reduced = matches!(plan.sample(), SampleAxis::Draws { rows, .. } if rows.len() > 1);
    let replicate_reduced = reps > 1
        && (backend != ForwardBackend::Ode || q.references_observations());
    if sample_reduced || replicate_reduced { Mode::Banded } else { Mode::Point }
}
```

`StackedQuantities` takes a mode per quantity rather than one for the render
(each quantity is its own file with its own header). Two consequences, both
intended: `--replicates N` under ODE renders pure-state quantities as points
(W8), and a one-row sample (`--draws uniform -n 1`, a one-row `--draws-file`)
renders as a point rather than a zero-width band. The second reverses a pin from
`2026-08-11` §7.5 ("`--draws <file>` with one row stays banded"); that pin
guarded against a _cell-count_ predicate that counted scenarios, and the rule
above counts only what one group reduces, so the reason for the pin is kept
while its output changes. The test is rewritten accordingly.

## 11. Why this does not reintroduce the caller-supplied key

Rejection reason 1 was that `BandSet::push(key, …)` let a caller choose the key.
Nothing here takes a key from a caller:

- Both accumulators keep `push_cell(&CellResult)` / `merge_cell(&CellResult)`
  and derive `PartitionKey` from `cell.spec.coords()`. No method has a key
  parameter.
- `CellCoords` is set only by `plan_grid`; `CellSpec`'s fields are private to
  `engine`. `PartitionKey` has no public constructor.
- `StackedQuantities::push_group` still takes `DesignCoords` from its caller.
  That is safe for the reason `2026-08-11` §3.5 gives — it renders groups that
  are already separated — and the coordinates it receives are computed by
  `DesignCoords::of` from a key the accumulator derived.

This guarantees that no caller supplies the key; it does not by itself guarantee
the _correct_ key — an accumulator could still key on `coords().scenario` alone.
That is what the mutation tests in Stage 5 (for `simulate`) and Stage 7 (for
`fit predict`) pin: key each accumulator on the scenario alone and confirm the
grouping assertions fail on row count.

Reason 2 is removed by Stage 7 (predict's sweep is in the cell). Reason 3 is the
subject of §6. Reason 4 no longer holds: `--sweep-file` and `simulate --sweep`
reach the design path in Stage 5.

## 12. Front-end validation

### 12.1 The `--sweep-file` reader

`--sweep-file` gets its own reader rather than `load_draws_tsv_keyed`, which
requires at least two columns (`main.rs:4402-4404`) and silently strips
`chain`/`draw` (`main.rs:4432`). The reader:

- accepts one or more parameter columns; tab-separated, with a message that
  detects a comma-separated file;
- takes an optional integer `point_id` column (unique, carried to output) and an
  optional string `label` column (carried to the manifest);
- refuses `chain` or `draw` columns, with a hint to use `--draws-file`;
- refuses a non-finite value, naming the line and column;
- refuses duplicate rows, naming both line numbers (D-E);
- refuses a constant column, with a hint: "column `gamma` is the same on every
  row, so it is not a coordinate; set it with --param or --params".

### 12.2 Parameters from outside the file

`--fit` backfills a draws file's missing parameters from the fit's `[fixed]`
block (#273, `main.rs:1650-1669`). For `--draws-file` the backfill stays;
backfilled columns are not file columns and are never varying. For
`--sweep-file` backfilled constants apply to every point and never become design
names. `--fit requires --draws` (`main.rs:1168`) accepts `--draws-file`.

### 12.3 Flags, conflicts, and what the dry run says

- `--draws` and `--draws-file` are mutually exclusive; `--sweep` and
  `--sweep-file` are mutually exclusive; a sample flag and a design flag
  compose. Several `--sweep` flags form a Cartesian product sorted by name; a
  repeated name is refused.
- Conflict lists that name only `draws` gain `draws_file`, `sweep` and
  `sweep_file`: `--stdout` (`args/mod.rs:431-434`), `--event-log` (`:563`),
  `--reactive-log` (`:582`). `--reactive-log` mirrors only the last cell
  (`main.rs:2116`), so a multi-cell design is refused rather than mirrored
  partially. `--draws-out` (`requires = "draws"`, `:416`) stays tied to
  generated draws.
- `--init-state fit` needs the posterior it pairs against (`main.rs:1149-1163`);
  with a design axis it is refused for the reason `conditioned_here` refuses a
  swept conditioned read. `--init-state FILE` refuses `--draws` today
  (`main.rs:1314-1322`); it also refuses `--draws-file`, `--sweep-file` and
  `--sweep`.
- `print_dry_run` (`main.rs:4462`) and the run-count banner
  (`main.rs:1693-1701`) describe design points, samples, replicates, scenarios
  and total cells; a grid above 10 000 cells prints a warning with the product
  spelled out.

### 12.4 The provenance sidecar

`--draws-out FILE` writes `FILE.json` beside the TSV:

| Field            | Meaning                                                           |
| ---------------- | ----------------------------------------------------------------- |
| `schema`         | `"camdl.draws/v1"`                                                |
| `measure`        | `posterior \| prior \| uniform_bounds`                            |
| `measure_source` | as in §9.3 (fit handle + run id + method, or prior source + hash) |
| `ir_hash`        | structural hash of the model that generated the draws             |
| `camdl_version`  | `VERSION_SHORT`                                                   |
| `varying`        | the columns written (§7.2)                                        |
| `subsample`      | `{"from": N, "stride": k}` when the source was strided            |

`--draws-file FILE` reads `FILE.json` when present: it carries the measure
through (a posterior stays `posterior`), warns when `ir_hash` differs from the
current model, and records in `summary` which parameters came from outside the
file and from where. `-n` subsamples a `--draws-file`, strided across all rows
with the same default cap as `--draws posterior` (gh#630).

## 13. What changes for users, and what does not re-key

| #  | Command                                                              | Today                                       | After                                                                                           |
| -- | -------------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1  | `simulate --draws FILE …`                                            | runs; pools a grid under `--quantities-out` | refused, naming `--sweep-file` and `--draws-file` (W9)                                          |
| 2  | `simulate --draws-file FILE`                                         | (does not exist)                            | today's `--draws FILE` seeds and trajectories; allowed under a scenario (W5)                    |
| 3  | `simulate --sweep`, `--sweep-file`                                   | (do not exist)                              | one group per point; CRN across points; refused against a scenario or `--param` on those params |
| 4  | `simulate --draws-out PATH`                                          | writes the file (all columns)               | writes varying columns + `PATH.json`; prints the `--draws-file` hint                            |
| 5  | `batch run [design.NAME]` × colliding scenario                       | refused as "the sweep"                      | refused as "the design `NAME`"                                                                  |
| 6  | Scenario, `--param` or design value setting a varying sampled column | runs silently                               | runs; note on stderr; recorded in `summary`                                                     |
| 7  | Banded quantity files (`simulate` and `fit predict`)                 | no statement of what the band is over       | `band` column; `point_id` on design rows; exact `sweep:` values; manifest `mode`/`summary`      |
| 8  | `fit predict --sweep` after the grid move                            | —                                           | byte-identical to the output after the schema commit                                            |
| 9  | `--replicates N` under ODE; one-row samples                          | zero-width band                             | point mode for pure-state quantities                                                            |
| 10 | `--design-from` with a file                                          | `--draws FILE`                              | `--draws-file` or `--sweep-file`; writes `truths.tsv`, `design_from.json`                       |
| 11 | `--draws-file` with `-n`                                             | (`-n` ignored for `--draws FILE`)           | strided subsample                                                                               |

**Run identity.** Every `sim` `run_id` folds the engine version, including the
git hash (`version.rs:12-15`; `ModelDigest::from_model`,
`runid/src/inputs.rs:304`; `resolve.rs:202`), so no `run_id` survives a commit
and "`run_id` unchanged" cannot be a test. The claim this proposal makes is **no
re-key beyond the existing per-build engine-version fold**: for every job that
exists today, the `params`, `scenario`, `seed` and `config` level hashes, the
store path segments below the model level, and the trajectory bytes are
unchanged. The new flags are new keys: `--sweep` and `--sweep-file` runs have no
predecessors, and a `--sweep-file` ensemble folds its axis kind (§7.6).

**IR and goldens are unaffected.** All changes are in `rust/crates/cli` and
docs. No `ir/VERSION` bump, no `ocaml/` change, no golden regeneration.

## 14. Staged implementation

Each stage lands alone with a green `make test`, ships no pooling regression and
no lost refusal, and each named test fails on the defect it guards (a mutation
check is named where the test could otherwise pass vacuously). Byte-neutral
stages are checked by comparing level hashes, path segments below the model
level, and output bytes — never whole `run_id`s (§13).

**Stage 0 — one constructor for `CellSpec`.** `CasSink::predict_cells` calls
`plan_grid`. Its iteration order today is point → scenario → seed
(`batch.rs:1282-1294`), `plan_grid`'s is scenario → point → rep, and the dry-run
and status displays print in prediction order; the predictions are re-sorted to
point → scenario → seed before display, so the terminal output is unchanged.
Test fixtures go through a `#[cfg(test)]` constructor. _Test:_ for a sweep × two
scenarios × three seeds manifest, `--dry-run`'s stdout is byte-identical before
and after, then run, then `--dry-run` again reports all-hit with the same leaf
paths.

**Stage 1 — `ParamPlan` and `CellCoords`, behaviour unchanged.** Replace
`ParamSource`; move replicate count into `Seeds`; split `point_overrides`;
migrate the §7.6 consumers mechanically, keeping today's seeds for every
existing flag (`--draws FILE` builds an `Asserted` sample with the row index as
seed index). Interim rule until Stage 5: a file-sourced sample (`Asserted`, or
from Stage 3 a sidecar-read measure) is refused against a scenario footprint
exactly as today's draws file is, with today's message, so
`tests/scenario_coordinate_collision_gh572.rs:282-334` stays green. _Tests:_
`tests/determinism_pin.rs` unchanged and green; a unit test planning point,
generated-draws, draws-file and explicit-seed sweep jobs and asserting each
cell's `process_seed` equals `process_seed_for` with the old `point_idx` (the
old formula kept in the test as the oracle); an A/B of
`simulate --draws uniform -n 5 --scenario a,b` and a `batch run` sweep comparing
level hashes, path segments below the model, and every output file's bytes.

**Stage 2 — the guard reads the plan.** After gh#949: `varying` on `SampleAxis`;
the guard's new signature (§8); `DesignOrigin` nouns; the `scale`
reparameterization hint; `--param`/`--param-vec` in the footprint; impositions
and design shadows returned and printed; the `--event-log` path routed through
the guard; every `SimRun` site classified (§7.4). _Tests:_ the existing
design-block test (`tests/scenario_coordinate_collision_gh572.rs:235-264`)
updated to assert "the design `NAME`"; `--draws uniform --scenario pinned` exits
0 with the W3 note naming `beta`; the same with a scenario over a non-varying
column prints nothing; `--draws uniform --param beta=0.5` exits 0 and records an
imposition `"by": "--param"` (the `--param`-on-design refusal is tested in Stage
5, when `simulate` first has a design axis). Mutation check: make the guard
treat design names as sample columns and confirm the batch refusal test goes
red.

**Stage 3 — provenance sidecar (D-D).** `--draws-out` writes varying columns and
`FILE.json`; the sidecar reader (used by `--draws FILE` until Stage 5 renames
it); `-n` subsamples a file. _Tests:_
`--draws posterior -n 1000 --draws-out
p.tsv`, then reading `p.tsv` back yields
`measure: posterior` with the fit's handle; a sidecar whose `ir_hash` differs
prints the warning; `-n 7` on a 100-row file picks rows strided across the whole
file through the same `subsample_draws` as `--draws posterior`, never a prefix.

**Stage 4 — output schema (D-C), its own commit, before Stage 7.** `band`
column, `point_id` on design rows, exact `sweep:` formatting, manifest `mode`,
`summary` always present, `partition_columns`, `measure_source`, `n_samples`
null when absent; `parameter_points.tsv` at round-trip precision. This
**intentionally changes `fit predict` output**, and lands before the grid move
so Stage 7's byte-identity gate is against this schema. Contrasts note and
`report.json` record under `--sweep` (W7). _Tests:_ the predict header pin in
`tests/fit_predict_sweep.rs` updated to the new header; a sweep value
`0.1234567` renders exactly; every manifest entry validates against §9.3 (a test
walking the fields).

**Stage 5 — the flag split, `simulate --sweep`, and design grouping, together.**
`--sweep-file` (reader §12.1), `--draws-file`, `simulate --sweep`; the bare
`--draws <path>` refusal and the typo message; D-B seeds for design jobs;
`by_partition` and per-quantity mode (§10) in the same stage, so no build can
pool design points (the interim refusal of Stage 1 is lifted here, for
`--draws-file` only). Combined writer columns (§9.2) and the ensemble axis fold;
flag conflicts, dry-run and banner (§12.3); `--init-state` refusals; `--fit`
backfill rules. Docs in the same stage: every `--draws FILE` in
`docs/camdl-run-spec.md` (lines 1036, 1053, 1058, 1131, 1158, 1165, 1515, 2000,
2065, 2146, 5249, 5293, 6160, 6187, and the recipe at 1976-1978),
`docs/mre.md:50`, `args/mod.rs:305` (after-help), the `--draws` and `--fit` help
(`args/mod.rs:386-396`), the `--design-from` help (`:478-480`), and the run-spec
sentence on `--draws uniform` (1145-1148). Tests that pass `--draws FILE` move
to `--draws-file`: `tests/simulate_identity_parity.rs:270`, `:317`,
`tests/gh641_init_state_forecast.rs:809`,
`tests/scenario_coordinate_collision_gh572.rs:282-334` (the last now asserts the
`--sweep-file` refusal and the `--draws-file` allowance). _Tests:_ the W9
refusal verbatim, including both replacement flags; `--draws prio` gives the
typo message; `--sweep beta=0.15,0.3,0.6 --backend ode --dt 0.1` gives rows
`0 0.15 70`, `1 0.3 304`, `2 0.6 536`; `--sweep-file` with the same three rows
gives byte-identical files; adding a fourth point leaves the first three points'
trajectory bytes unchanged (CRN); `--draws-file f.tsv` trajectories equal Stage
4's `--draws f.tsv` trajectories byte for byte; the W12 `--param` refusal;
`--sweep-file` duplicate rows refused naming both lines; ODE `--replicates 4`
renders `peak_I` in point mode while an observation quantity is banded. Mutation
check: key `by_partition` on the scenario alone and confirm the three-row
assertion fails on the row count, not the header.

**Stage 6 — `--design-from` on the new axes.** Accept every source; priors from
the design `fit.toml`; write `truths.tsv` and `design_from.json`. Dataset files
byte-identical. _Test:_ `--sweep-file truths.tsv` writes one `ds_NN/` per row,
`truths.tsv` with `point_id`, and `"axis": "design"`; `--draws prior -n 3` gives
`"axis": "sample", "measure": "prior"` with the design fit.toml's hash.

**Stage 7 — `fit predict` on the grid.** One plan; `DesignAxis::sweep` carries
predict's name checks; the pre-flight guard call stays before the free-forward
closure; rows rendered design-major explicitly; to bound memory, one `run_job`
per design point (a design slice) rather than one for the whole grid, since
`run_job` holds every cell's result until the merge phase (`engine.rs:232-247`);
the per-cell conditioned decision is a function of `PartitionKey` (conditioned
iff the scenario is the conditioned arm and the design is `Unit`), replacing the
per-sink `conditioned` / `chain_of_point` fields (`predict.rs:2196-2218`);
`FreeForwardCell` and `assemble_predictive` keyed by `PartitionKey`. _Tests:_
byte-identity A/B against Stage 4 output of every predict artifact (predictive
TSVs, quantities TSVs, manifests, contrasts, `report.json`) for
`--sweep k=… --scenario a,b` with and without a conditioned fit;
`--sweep beta=… --scenario distancing` exits non-zero with **no** artifact
written; a unit test that the conditioned decision is true only for the un-swept
`fitted` group. Mutation check: key the sink on scenario alone and confirm the
four-group assertion fails.

**Stage 8 — error text.** `ValueSource::{Sample, DesignPoint}`;
`UnknownParameter` names the column and the file; `NonFiniteValue` names the
flag the command has. _Tests:_ `params_resolver.rs` unit tests at `:2206`,
`:2238`, `:2310` updated to the new variants; a CLI test that a misspelt
`--draws-file` column prints `column`betta`of`post.tsv`is not a parameter`.

## 15. Follow-ups, to be filed

- **F1 — `scale` × sweep composition.** Admit a scenario that scales a swept
  parameter, using the requested/effective representation in §17.
- **F2 — paired contrasts within design points, and a per-cell table.** A
  free-forward contrast that pairs arms on the sample coordinate within each
  design point; and an opt-in `cells.tsv` (`run_id`, `scenario`, `point_id`,
  `sweep:*`, `draw`, `rep`, `process_seed`, optionally quantity values) so
  paired reductions are possible from `simulate` output.
- **F3 — `batch run --quantities-out`,** so exploring a design does not need a
  `simulate --sweep-file` round trip that recomputes every point (W2).

The earlier plan to file a separate issue for common random numbers across
design points is withdrawn: D-B makes it the default for the new flags, and no
existing flag builds a design in `simulate`.

## 16. Decisions

**D-A — Flags come in axis-named pairs.** `--sweep` / `--sweep-file` build the
design axis and `--draws` / `--draws-file` the sample axis, matching the
`sweep:<param>` column prefix. A file's rows are either conditions to compare or
a sample the user stands behind, and nothing in the file says which, so the flag
names the axis and a bare `--draws <path>` is refused naming both replacements —
the alpha posture: break with the replacement spelled out, no shim.
`[design.NAME]` in `batch run` keeps its name, because there "design" names a
generator (`sobol | lhs | random`); `--design-from` is unchanged, because there
"design" names the fit's observation schedule. An in-file declaration was not
chosen: the reader takes line 1 as the header (`main.rs:4392-4399`), so a
directive line would break older camdl and every other reader of these files.

**D-B — Design coordinates never enter the process seed.** `seed_index` is the
sample index, or 0 without a sample axis, so design points share common random
numbers by default — as they already do in `batch run` (explicit seeds) and
`fit predict` (seed follows the draw). Paired comparisons across design points
are then low-variance, and inserting a point into a sweep leaves the others'
trajectories unchanged. `--sweep` and `--sweep-file` are new, so nothing
re-keys; `--draws-file` keeps today's `--draws FILE` seeds exactly. The cost is
stated in §7.3: one file read as a design and as a sample does not share store
leaves.

**D-C — The output schema is fixed before predict's bytes are frozen.** A
`point_id` on every design-bearing row (the file's, else the 0-based index), so
rows join without float matching; design coordinates written round-trip exact; a
constant `band` column naming what every band is over; and a manifest that
describes itself (`mode`, `summary` always, `partition_columns`,
`measure_source` with a fit handle and hash rather than a path, `n_samples` null
when absent, `n_samples_used` where rows can drop). This changes `fit predict`
output, so it lands as its own commit (Stage 4) before the grid move, whose
byte-identity gate is then against the new schema.

**D-D — Draws files carry their provenance.** `--draws-out` writes a sidecar
recording measure, source, IR hash, version, varying columns and subsampling;
`--draws-file` reads it, so an exported posterior stays a posterior, a stale
export against a changed model is flagged, and parameters from outside the file
are named. `-n` subsamples a `--draws-file`, closing the gh#630 path by which a
raw posterior replays every row.

**D-E — Duplicate design rows are refused only in `--sweep-file`.** In a
hand-written file a duplicate is almost always an error, so it is refused with
both line numbers. Generated grids keep accepting duplicates as today; identical
cells resolve to one leaf under effective-value identity, so the cost is a
repeated row, not a repeated simulation.

**`simulate` gains `--sweep`** on the grammar and expansion `fit predict`
already uses (§7.6), producing a `DesignAxis` with origin `Sweep`.

**`--draws uniform` is a sample with `Measure::UniformBounds`.** The uniform law
on the declared bounds is a real measure (§6.1), so a band over it is a correct
Monte Carlo estimate of its push-forward; the `band` value `uniform_bounds`
keeps it from being captioned as a credible interval. Per-point rows are one
`--draws-out` / `--sweep-file` round trip away (W3).

**`fit predict`'s sweep moves onto the grid (Stage 7), gated on byte identity
against Stage 4.** It puts predict's partition coordinate in the cell and lets
one guard and one grouping derivation serve both verbs, removing rejection
reason 2.

**Impositions are allowed and recorded, not refused.** A scenario, `--param` or
design value that sets a varying sampled column is an intervention on that
parameter; the note says so in plain words (W3) and suggests `scale` when the
intent is to reduce transmission relative to each draw.

**Every design origin uses the `sweep:<param>` column prefix,** leaving
`fit predict`'s header prefix and the manifest `sweep` object unchanged.

**`--design-from` accepts every axis and records which** (W11), because whether
downstream analysis may pool across its datasets depends on the kind.

**Contrasts stay sweep-agnostic and say so** (W7).

## 17. Requested and effective values for a future scale × sweep composition

A scenario that scales a design parameter is refused (§8), so a design
coordinate's requested value always equals the value the cell ran. If F1 admits
composition — sweep `beta ∈ {0.15, 0.3, 0.6}` under `distancing` running
`{0.09, 0.18, 0.36}` — the representation is fixed here:

- **Identity hashes only effective values.** It does already
  (`batch.rs:1141-1153`, `resolve.rs:231`, verified). Cells whose effective
  values coincide under different requested coordinates share their level hashes
  and are one computation in the store.
- **The grouping key is the requested coordinate** — the design point's index
  (`DesignCoord::Point { idx }`), reported as `point_id`.
- **A column denotes one thing.** `sweep:beta` keeps meaning the requested grid
  value; the effective value, where it differs, is reported under a separately
  named column and in the manifest's per-group `requested`/`effective` pair.
