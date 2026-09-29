# A parameter point is a design coordinate or a sample, and the cell says which

- Date: 2026-09-27
- Status: accepted — every call is recorded in §13; ready to implement
- Relates-to: gh#562, gh#572, gh#575, gh#948,
  `2026-06-27-sealed-fit-packets-handles-and-override-algebra.md` (§4),
  `2026-08-11-scenario-banding-in-simulate.md` (implemented)
- Builds on: an untracked architecture-review note on gh#562 (2026-08-11). Every
  claim this proposal uses from it is restated in §4; the note is not needed to
  read this document.

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
Collision policy, leaf labels and provenance, and banding then key off one
property the cell carries, set by one constructor. On the command line the
ambiguous `--draws FILE` splits by axis into `--draws-file FILE` (a sample) and
`--design FILE` (design points); `simulate` gains `--sweep`; `fit predict`'s
sweep moves onto the grid, so both verbs put their partition coordinates in the
cell. Nothing re-keys run identity; the IR and goldens are untouched.

## 1. How the user works with each axis

Every simulation grid is a product of four axes. The user controls each with a
distinct flag or manifest key, and each plays one role in the output.

| Axis          | `simulate` (after)                                         | `batch run` manifest         | `fit predict`                 | Output role                                      |
| ------------- | ---------------------------------------------------------- | ---------------------------- | ----------------------------- | ------------------------------------------------ |
| **Scenario**  | `--scenario a,b`                                           | `[[scenario]]`               | `--scenario a,b` (+ `fitted`) | group: `scenario` column                         |
| **Design**    | `--sweep beta=…` or `--design FILE`                        | `[sweep]` or `[design.NAME]` | `--sweep beta=…`              | group: `sweep:<param>` columns                   |
| **Sample**    | `--draws posterior\|prior\|uniform` or `--draws-file FILE` | none                         | always the fit's posterior    | pooled into the band; manifest names the measure |
| **Replicate** | `--replicates N` or `--seeds a,b,…`                        | `seeds = {…}`                | one per draw                  | pooled into the band                             |

The rule the output follows: **scenario and design make groups (rows); sample
and replicate are pooled into the band within each group.** A quantity file has
one row (or one block of time rows) per group, and its quantile columns
summarize the sample and replicate cells inside that group only.

What stays the same: scenarios, replicates, seeds,
`--draws
posterior|prior|uniform`, `batch run` manifests, and `fit predict`'s
output bytes. What changes: a bare `--draws FILE` is refused with a message
naming its two replacements; `simulate --sweep` and `simulate --design` are new
and emit `sweep:<param>` columns; a scenario that touches a sampled column is
recorded in the manifest; every banded manifest entry says what its band is over
(§9.2).

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

A **marginal reduction** summarizes a set of cells treating them as exchangeable
draws from one distribution — a quantile, a mean. A **paired reduction**
combines cells from _different_ values of a partition axis that share a band
coordinate — draw _i_ of the baseline arm with draw _i_ of the intervention arm
— to construct a new random variable before any marginal summary.

A scenario **shadows** a sampled column when it `set`s that parameter: every
sample then runs at the scenario's value, so that column's variation is removed
from the band. A scenario that `scale`s a sampled column **rescales** it: the
variation is kept and multiplied.

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
(`predict.rs:2188`) and reaches the renderer directly (`predict.rs:2297-2300`).
Every predict cell looks like a posterior draw; its partition coordinate is not
in the cell, and its collision check must run separately
(`predict.rs:1623-1649`, verified, whose comment says so: "predict folds the
sweep into the draw rows, which the engine sees as generated draws").

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
(`engine.rs:364-390`, verified) runs from `run_job` (`engine.rs:203`), from
`batch::batch_job` (`batch.rs:605`) so dry-run and status refuse too, and from
`simulate --dry-run` (`main.rs:1799`). Its wording comes from one formatter,
`params_resolver::scenario_coordinate_collision` (`params_resolver.rs:842-940`),
which `fit predict` also calls (`predict.rs:1637`). The policy is the
maintainer's: a scenario touching a _user-authored_ coordinate — a sweep, a
design block, a `--draws <file>` — is refused, `set` and `scale` alike;
generated draws (`--draws posterior|prior|uniform`) are not checked.

The guard decides which is which by matching `ParamSource`
(`engine.rs:367-375`): `Sweep` and `Draws { explicit_file: Some(_) }` are
checked, everything else passes. So the correct policy is keyed on the proxy
reason 3 rejects, and it inherits the proxy's blind spots: a `[design.NAME]`
block is reported as "the sweep" (`params_resolver.rs:881`, because the design
path builds a `Sweep`, `batch.rs:1839`), and predict's sweep needs its own call
because the engine cannot see it. This proposal keeps the policy and replaces
the key (§8).

## 5. Workflows, end to end

Each workflow gives the modelling question, the commands, the output headers
with example rows, the manifest `summary` object (§9.2), and what each relevant
collision does. File paths are relative to the `--quantities-out` directory `q/`
unless stated. Series quantities are omitted for brevity; `peak_I` and `final_R`
are scalars, so each file has one row per group.

### W1 — Sensitivity sweep: how does the peak depend on `beta`?

**Today** there is no `simulate --sweep`; the nearest spelling is a draws file,
which pools (§4.1). `batch run` sweeps but emits no quantities.

**After**, three spellings, one result:

```
camdl simulate sir.camdl --params base.toml --sweep beta=0.15,0.3,0.6 \
    --backend ode --dt 0.1 --quantities-out q

camdl simulate sir.camdl --params base.toml --design grid.tsv \
    --backend ode --dt 0.1 --quantities-out q          # grid.tsv: one column `beta`

# batch.toml
[sweep]
beta = [0.15, 0.3, 0.6]
```

`q/quantities/peak_I.tsv` (point mode — one realization per group):

```
sweep:beta  value
0.15        70
0.3         304
0.6         536
```

Manifest entry for `peak_I`: `"sweep": {"beta": 0.15}` per group as today in
`fit predict`, and no `summary` object, because point mode reduces nothing.

**With `--replicates 20 --backend chain_binomial`**, each design point gets its
own band over process noise:

```
sweep:beta  n_draws  q05  q25  q50  q75  q95
0.15        20       …    …    …    …    …
0.3         20       …    …    …    …    …
0.6         20       …    …    …    …    …
```

`"summary": {"over": ["replicate"], "measure": "process", "n_samples": 0,
"replicates_per_sample": 20, "scenario_touches": {}}`.

**Collisions.** `--scenario pinned` (`set beta`) or `--scenario distancing`
(`scale beta`) touches the swept parameter and is refused before any cell runs
(W10 shows the message). `--sweep gamma=0.05,0.1,0.2 --scenario distancing`
touches distinct parameters and runs: six groups, `scenario  sweep:gamma
value`.

### W2 — Space-filling design: where in (`beta`, `gamma`) space is the epidemic large?

**Today and after**, the batch spelling is unchanged:

```
[design.wide]
method = "lhs"
n = 64
parameters.beta  = { range = { min = 0.05, max = 1.0 } }
parameters.gamma = { range = { min = 0.02, max = 0.5 } }
```

It writes `designs/wide/parameter_points.tsv` and one store leaf per point, as
today; `batch run` emits no quantities. **After**, the same points can be
simulated with quantities by passing the generated table as a design:

```
camdl simulate sir.camdl --design out/designs/wide/parameter_points.tsv \
    --backend ode --quantities-out q
```

```
sweep:beta  sweep:gamma  value
0.43…       0.11…        …
…
```

(64 rows.) `parameter_points.tsv` carries a `point_id` column
(`batch.rs:1853-1859`, verified); `--design` treats a column named `point_id` as
a row label and not a parameter, and refuses any other non-parameter column.
Collisions: a scenario touching `beta` or `gamma` is refused, naming "the design
`wide`" from `batch run` and "the design file `parameter_points.tsv`" from
`simulate`.

### W3 — Prior predictive and uniform exploration: what does the model predict before seeing data?

**Today and after**, the command is unchanged:

```
camdl simulate sir.camdl --draws prior -n 500 --backend ode --quantities-out q
camdl simulate sir.camdl --draws uniform -n 500 --backend ode --quantities-out q
```

`q/quantities/peak_I.tsv` is unchanged:

```
n_draws  q05  q25  q50  q75  q95
500      …    …    …    …    …
```

The manifest gains, for `--draws prior`:

```json
"summary": { "over": ["sample"], "measure": "prior", "n_samples": 500,
             "replicates_per_sample": 1, "scenario_touches": {} }
```

and for `--draws uniform`, `"measure": "uniform_bounds"` — quantiles under the
uniform law on the declared bounds, which no one should caption as a credible
interval.

**Per-point rows instead of a band.** Export the sample, then read it back as a
design:

```
camdl simulate sir.camdl --draws uniform -n 500 --draws-out u.tsv --backend ode
camdl simulate sir.camdl --design u.tsv --backend ode --quantities-out q
```

giving `sweep:beta  sweep:gamma  value`, 500 rows — the table for plotting
`peak_I` against each parameter.

**Collisions.** `--scenario pinned` over the prior shadows `beta`: it runs, with
one stderr line
(`scenario 'pinned' sets sampled parameter beta; its prior
variation is removed from this arm`)
and `"scenario_touches": {"beta": "set"}` in that arm's `summary`.
`--scenario distancing` rescales `beta`:
`"scenario_touches": {"beta": "scale"}`, no stderr line.

### W4 — Posterior predictive under interventions: what does distancing do, given what the fit learned?

**Today and after**:

```
camdl simulate sir.camdl --fit results/fits/sir-8a3f12b4 --draws posterior \
    --scenario baseline,distancing --backend ode --quantities-out q
```

`q/quantities/peak_I.tsv` is unchanged:

```
scenario    n_draws  q05  q25  q50  q75  q95
baseline    200      …    …    …    …    …
distancing  200      …    …    …    …    …
```

Each arm's manifest entry gains a `summary`; for `distancing`:

```json
"summary": { "over": ["sample"], "measure": "posterior", "n_samples": 200,
             "replicates_per_sample": 1, "scenario_touches": {"beta": "scale"} }
```

`--scenario pinned` shadows `beta` in its arm (`{"beta": "set"}`, one stderr
line); `gamma` still varies across the 200 draws, so `over` stays `["sample"]`.
The draws are paired across arms — draw _i_ uses one parameter vector and one
process seed in both (§6.3).

### W5 — Exporting a posterior and reusing it

**Today**:

```
camdl simulate sir.camdl --fit results/fits/sir-8a3f12b4 --draws posterior \
    -n 200 --draws-out post.tsv
camdl simulate sir.camdl --draws post.tsv --scenario distancing ...
```

The second command is refused today, because `distancing` scales the file's
`beta` column and a draws file is a user-authored coordinate (§4.4).

**After**, `--draws-out` ends with a round-trip hint:

```
draws.tsv: wrote 200 draws to post.tsv (reuse as a sample with --draws-file post.tsv)
```

and

```
camdl simulate sir.camdl --draws-file post.tsv --scenario distancing \
    --backend ode --quantities-out q
```

runs: the file is a sample (`Measure::Asserted`), so the scenario is a
counterfactual over it. Output as in W4 with
`"measure": "asserted", "source": "post.tsv"` in the `summary`. The same file
passed as `--design post.tsv --scenario distancing` is refused, naming "the
design file `post.tsv`".

### W6 — `fit predict` over a sweep, the posterior, and scenarios

**Today and after**, the command and its output bytes are identical (Stage 4 is
gated on byte identity):

```
camdl fit predict @sir-fit --sweep gamma=0.08,0.12 --scenario distancing
```

`<fit>/quantities/peak_I.tsv`:

```
scenario    sweep:gamma  n_draws  rhat  ess  q05  q25  q50  q75  q95
fitted      0.08         200      …     …    …    …    …    …    …
fitted      0.12         200      …     …    …    …    …    …    …
distancing  0.08         200      …     …    …    …    …    …    …
distancing  0.12         200      …     …    …    …    …    …    …
```

What changes is internal: the sweep is the job's design axis and the posterior
its sample axis, so one `run_job` runs every (scenario, sweep point, draw), and
the grouping key comes from the cell. The manifest gains `summary` with
`"measure": "posterior"`. `--sweep beta=… --scenario distancing` is refused by
the shared guard, naming "the sweep" (today predict's own call does this).

### W7 — Contrasts: how many infections does an intervention avert?

A contrast is a paired reduction: `fit predict` replays both arms per forkable
posterior draw under one seed, `derive_chain_seed(seed, draw_pos)`
(`contrasts.rs:447`, verified), differences them, then bands the differences
into `contrasts/<name>.tsv` with header `q05 q25 q50 q75 q95 mean n_used` for a
scalar (`contrasts.rs:1240`). A contrast needs an arm that toggles an
intervention; a parameter-only scenario such as `distancing` is skipped with a
note (`contrasts.rs:32`, gh#327). So W7 adds an intervention `closure` and a
scenario `close { enable = [closure] }`, and declares
`averted = fitted.quantities.final_R - close.quantities.final_R`.

**Today and after**, `camdl fit predict @sir-fit` writes
`contrasts/averted.tsv`:

```
q05  q25  q50  q75  q95  mean  n_used
…    …    …    …    …    …     200
```

Under the new coordinates the pairing coordinate has a name — the cell's sample
coordinate — and the rule is: a paired reduction across a partition axis is
valid iff both sides share one sample axis (§6.3). Within a job this always
holds across scenarios.

**With a design axis present**, pairing would be within each design point
(scenario arms matched on sample coordinate, one contrast band per
`sweep:<param>` value). That is not delivered here, because contrasts fork each
draw from its _smoothed_ state, inferred under the fitted parameters; a sweep
point replaces a parameter, so no smoothed state exists for it — the same reason
`conditioned_here` refuses a conditioned read for any swept cell
(`predict.rs:934-943`, verified). Contrasts therefore stay sweep-agnostic, as
today (`emit_contrasts` receives no sweep, `predict.rs:2731-2739`, verified):
emitted once, for the un-swept model. A free-forward contrast that pairs within
design points is a separate feature to file as its own issue.

### W8 — Process noise only: how variable is one parameter set's epidemic?

**Today and after**:

```
camdl simulate sir.camdl --params base.toml --replicates 50 \
    --backend chain_binomial --quantities-out q
```

```
n_draws  q05  q25  q50  q75  q95
50       …    …    …    …    …
```

`"summary": {"over": ["replicate"], "measure": "process", "n_samples": 0,
"replicates_per_sample": 50, "scenario_touches": {}}`.
`--seeds 1,2,3` is the same with explicit seeds. Under `--backend ode`
replicates are identical (the ODE solver draws no randomness), so `over` omits
`replicate` and `measure` is absent: the band has zero width and the manifest
says it reduces nothing. Scenarios make groups as usual; nothing collides,
because there are no parameter coordinates.

### W9 — A hand-authored grid passed the old way

**Today**, `camdl simulate sir.camdl --draws grid.tsv --quantities-out q` writes
the pooled band of §4.1.

**After**, it is refused before any work:

```
error: `--draws grid.tsv` no longer accepts a file: say which axis its rows are.
  --design grid.tsv      the rows are conditions to compare (a grid, a
                         hand-picked set): one output row per point, reported
                         in `sweep:<param>` columns
  --draws-file grid.tsv  the rows are a sample from a distribution you stand
                         behind (e.g. exported posterior draws): pooled into
                         one quantile band
`--draws` takes only `posterior`, `prior` or `uniform`.
```

### W10 — A `batch run` sweep that a scenario would override

**Today** (landed with gh#572), a manifest sweeping `beta` under `pinned` is
refused by `batch run`, `batch run --dry-run` and `batch status`, worded by
`params_resolver.rs:842-940`:

```
error: parameter `beta` is controlled by both the sweep and scenario `pinned`
  sweep:     beta = [0.15, 0.3, 0.6]
  scenario:  set beta = 0.3
The scenario would override every sweep value, so this sweep would not vary `beta`.
Fix: remove `beta` from the sweep, or use a scenario that does not touch it.
```

(Layout inferred from the formatter; the column width is
`max(len(label), len("scenario:")) + 2`.) Under `distancing` the middle lines
read `scale beta × 0.6` and "would rescale every sweep point". **After**, the
same message names the design's origin: "the sweep" for `[sweep]` and `--sweep`,
"the design `wide`" for `[design.wide]`, "the design file `grid.tsv`" for
`--design grid.tsv`. The draws-file wording is deleted: a sample never collides
(W3–W5).

### W11 — `--design-from`: synthetic datasets on a fit's observation design

`--design-from fit.toml` writes a dataset — one file per observation stream on
the fit's own observed rows — at each parameter vector; with `--draws` it writes
one `ds_NN/` per row (`args/mod.rs:470-489`; `simulate_on_bound_design`,
`main.rs:3248`). It bypasses the engine, and it ignores `--scenario` entirely
(gh#948, a separate defect).

**After**, it accepts every parameter source —
`--draws
posterior|prior|uniform`, `--draws-file`, `--design`, `--sweep` — and
still writes one dataset per parameter point, because it never reduces. The axis
decides two things only: the directory label (`ds_NN` over sample index for a
sample; `ds_NN` with the design coordinates recorded for a design) and a
`design_from.json` beside the datasets recording the kind and measure. That
record matters downstream: a simulation-based calibration pools rank statistics
across datasets and is valid only when the truths were a sample from the prior
(`--draws prior`); a recovery study at chosen truths (`--design truths.tsv`)
must not be pooled that way. A former `--draws truths.tsv` user gets the W9
refusal and chooses. Once gh#948 routes `--scenario` through, the §8 guard
applies to `--design-from` unchanged.

## 6. The rule: design coordinates condition, samples are marginalized

### 6.1 The kind is a declared reading, and only a sample needs a measure

The kind is not a property of the numbers. An iid uniform draw can be read as a
sample of the uniform law (to band a quantity over it) or as a set of conditions
(to plot the quantity against the parameter). What differs is the _question_ —
condition on the point or marginalize over it. The asymmetry that makes this
safe: **any point set may be read as a design; only a point set with a measure
may be read as a sample.** A sweep grid has no measure, so it can never be a
sample. A generated draw has a measure by construction. A file has one only if
the user asserts it, which is what `--draws-file` means.

| Source                                      | Axis   | Measure                 |
| ------------------------------------------- | ------ | ----------------------- |
| `--draws posterior`, `fit predict`          | sample | the fit's posterior     |
| `--draws prior`                             | sample | prior (IR or fit.toml)  |
| `--draws uniform`                           | sample | uniform on declared box |
| `--draws-file FILE`                         | sample | asserted by the user    |
| `--sweep`, `[sweep]`, `fit predict --sweep` | design | —                       |
| `--design FILE`                             | design | —                       |
| `[design.NAME]`                             | design | —                       |

`--draws uniform` is an iid sample, not a space-filling design:
`generate_uniform_draws` draws `lo + (hi − lo)·U` independently per parameter
per row (`main.rs:3860-3880`, verified), holding unbounded parameters at their
default. The run spec calls it "space-filling exploration"
(`docs/camdl-run-spec.md:1145-1148`); Stage 3 corrects that sentence. The actual
space-filling designs (`[design.NAME]`, `method = "sobol" | "lhs" |
"random"`,
`batch.rs:98-107`) already run as `ParamSource::Sweep` (`batch.rs:1839`) — which
is reason 3 in running code: one iid uniform design is a partition in
`batch run` and a band in `simulate`.

### 6.2 What each kind implies for the three decisions

| Decision                                | Design coordinate                                        | Sample                                                           |
| --------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------- |
| A scenario sets or scales the parameter | Contradiction — refused, naming the design's origin      | Counterfactual — allowed; recorded in `summary.scenario_touches` |
| Labels and provenance                   | A `sweep:<param>` column and the manifest `sweep` object | Not per row; the manifest names the measure                      |
| Marginal reduction (quantile, mean)     | Never crosses it                                         | Crosses it                                                       |
| Paired reduction                        | Crosses it, matched on the band coordinate               | Supplies the matching coordinate                                 |

The collision row is the landed gh#572 policy on every input it covers: sweeps,
design blocks and design files are refused, generated draws allowed. The one
input whose treatment changes is an exported sample read back with
`--draws-file`, which is now allowed under a scenario (W5) — the file carries a
declared measure, so a scenario over it is a counterfactual, exactly as over the
generated draws it came from.

### 6.3 Marginal reductions stay within a group; paired reductions cross groups

> A marginal reduction may cross band axes only. A partition axis is eliminated
> only by an explicit measure over its levels (none exists in camdl) or by a
> paired reduction, which constructs a new random variable per band coordinate
> before any marginal summary.

A paired reduction across a partition axis is well-defined iff the cells on both
sides share one `SampleAxis` (same rows, same measure): the band coordinate then
denotes the same parameter vector on both sides. Pairing across scenarios always
satisfies this (a job has one sample axis, and the process seed excludes the
scenario, `engine.rs:11-23`). Whether pairing also enjoys the variance reduction
of common random numbers depends on the seed rule (§7.3), not on validity.

### 6.4 Out of scope

**Weighted designs.** A `[design.NAME]` parameter may carry a `prior` for
importance weighting by `camdl voi` (`batch.rs:111-117`). camdl has no weighted
quantile path and `batch run` emits no quantities, so design blocks are design
axes; a weighted-sample reading waits for a consumer.

**`fit run --sweep`** fits once per point — not a simulation grid — and is
untouched.

## 7. Types

### 7.1 The plan: two axes and a replicate count

`ParamPlan` replaces `ParamSource` in `SimulateJob.source` (`sim_job.rs:70`).
Each axis is a sum type whose unit case is its own variant, so an axis with no
points but an origin, or a sample without a measure, cannot be written down.

```rust
// sim_job.rs

/// The parameter-point structure of a job: a design axis crossed with a
/// sample axis, each cell replicated `replicates` times.
#[derive(Debug, Clone)]
pub struct ParamPlan {
    design: DesignAxis,
    sample: SampleAxis,
    replicates: usize,
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
    /// Requested values, aligned to `names`.
    values: Vec<f64>,
}

/// Where a design came from — the noun the collision diagnostic uses.
#[derive(Debug, Clone)]
pub enum DesignOrigin {
    Sweep,                               // --sweep, [sweep], fit predict --sweep
    Design { method: DesignMethod },     // [design.NAME]: "the design `NAME`"
    File { path: PathBuf },              // --design FILE
}

/// Exchangeable draws from a stated measure. A band axis.
#[derive(Debug, Clone)]
pub enum SampleAxis {
    Unit,
    Draws { rows: Vec<IndexMap<String, f64>>, measure: Measure },
}

/// What the sample is a sample of.
#[derive(Debug, Clone, PartialEq)]
pub enum Measure {
    Posterior { fit: String, method: String },   // fit handle + stage label
    Prior { source: PriorSource },               // model IR or fit.toml
    UniformBounds,                               // lo + (hi-lo)·U on declared bounds
    Asserted { path: PathBuf },                  // --draws-file FILE
}
```

`DesignMethod` carries the block name and the method (`sobol | lhs | random`).
Fields are private to `sim_job`; the constructors are the only way in, each
validating its invariant:

```rust
impl ParamPlan {
    pub fn point(replicates: usize) -> Self;
    /// Refuses a design name the model does not declare.
    pub fn new(design: DesignAxis, sample: SampleAxis, replicates: usize, model: &ir::Model)
        -> Result<Self, String>;
}
impl DesignAxis {
    /// From the CLI grammar shared with `fit predict` (§7.5).
    pub fn sweep(specs: &[crate::args::types::SweepSpec]) -> Result<Self, String>;
    /// From already-expanded rows: `[sweep]`, `[design.NAME]`, `--design FILE`.
    /// Refuses ragged rows and duplicate rows.
    pub fn from_rows(rows: Vec<IndexMap<String, f64>>, origin: DesignOrigin)
        -> Result<Self, String>;
}
impl SampleAxis {
    pub fn draws(rows: Vec<IndexMap<String, f64>>, measure: Measure) -> Self;
}
```

Duplicate design rows are refused because two groups with identical coordinates
are indistinguishable in a tidy output; the same condition twice is
`--replicates`.

A design name may also be a sample column: `fit predict --sweep gamma` crosses a
`gamma` design with posterior rows that carry `gamma`, and the design value wins
(today's row overwrite, `predict.rs:2235-2237`). §7.4 makes that a resolver
tier.

### 7.2 The cell: coordinates replace `point_idx` and `point_overrides`

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
    pub fn seed_index(&self) -> usize;   // §7.3
}
```

`CellSpec`'s fields become private to `engine`, read through accessors, and
`plan_grid` (`engine.rs:148-193`) is the single constructor. The one non-test
constructor outside it is `CasSink::predict_cells` (`batch.rs:1272-1307`,
verified), which re-implements the grid loop for dry-run cache classification;
it calls `plan_grid` instead (Stage 0). The test fixtures (`batch.rs:2771`,
`:2849`) use a `#[cfg(test)]` constructor in `engine`. `plan_grid` iterates
`scenario → design point → sample → replicate`; with either axis `Unit` this is
today's `scenario → point → rep`.

### 7.3 The seed index preserves every current trajectory

```rust
fn seed_index(&self) -> usize {
    match (&self.sample, &self.design) {
        (SampleCoord::Draw { idx }, _) => *idx,
        (SampleCoord::Unit, DesignCoord::Point { idx, .. }) => *idx,
        (SampleCoord::Unit, DesignCoord::Unit) => 0,
    }
}
```

`process_seed_for` (`engine.rs:52-66`) receives `seed_index` where it receives
`point_idx` today. Every existing job maps to the same number:

- `--draws posterior|prior|uniform`: design `Unit`, sample index = today's
  `point_idx`.
- `--draws-file`: the same rows at the same indices as today's `--draws FILE`.
- `--design FILE`: sample `Unit`, design index = today's `point_idx` for the
  same file under `--draws`.
- `batch run`: explicit seeds (`batch.rs:596`), so `seeds[rep]`; the index is
  unused (`engine.rs:59-60`).
- `fit predict --sweep` after Stage 4: sample index = draw index, as inside
  today's per-sweep-point jobs. The `total_runs == 1` branch (`engine.rs:61-62`)
  no longer fires on the combined job, but `mix_cell_seed(base, 0, 0) == base`
  (`util.rs:35-37`; asserted at `engine.rs:520`), so the seed is unchanged.

`seed_index` also replaces `point_idx` at the three per-draw lookups that follow
the sample: the `--init-state fit` row (`engine.rs:452-455`),
`ConditionedSource::per_draw` (`predict.rs:1187`) and `chain_of_point`
(`predict.rs:1262`). The wide-format `draw` column (`main.rs:2931`) prints
`seed_index + 1`, byte-identical.

### 7.4 Resolver tier 3.5 splits in two, design over sample

Tier 3.5 (`params_resolver.rs:1095-1118`) becomes 3.5a (sample row) then 3.5b
(design point), and `ValueSource::SweepPoint` (`params_resolver.rs:193-196`,
tagged `"sweep_point"` at `:209` even for a posterior draw) splits into
`ValueSource::Sample` and `ValueSource::DesignPoint`. `SimRun.point_overrides`
(`engine.rs:469`) splits into `sample_overrides` and `design_overrides`.
Resolved values are unchanged for every current job: the only job with both axes
is `fit predict --sweep`, whose design value already overwrote the sample value.

### 7.5 Parse at the boundary: where each plan is built

| Front end                 | Today                                                              | After                                                                           |
| ------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| `simulate --draws …`      | `main.rs:1735-1751`: `Draws { explicit_file }` or `Point`          | `SampleAxis::draws(rows, Posterior \| Prior \| UniformBounds)`                  |
| `simulate --draws-file`   | (was `--draws FILE`)                                               | `SampleAxis::draws(rows, Asserted { path })`                                    |
| `simulate --design`       | —                                                                  | `DesignAxis::from_rows(rows, File { path })`                                    |
| `simulate --sweep`        | —                                                                  | `DesignAxis::sweep(&specs)`                                                     |
| `batch run [sweep]`       | `batch.rs:629-644` (`sweep_source`): `Sweep` or `Point`            | `DesignAxis::from_rows(points, Sweep)`                                          |
| `batch run [design.NAME]` | `batch.rs:1839`: `Sweep`                                           | `DesignAxis::from_rows(points, Design { method })`                              |
| `fit predict`             | `predict.rs:2254`: `Draws { explicit_file: None }` per sweep point | one plan: `DesignAxis::sweep(&args.sweep)` × `SampleAxis::draws(.., Posterior)` |

`simulate --sweep` reuses the CLI grammar already shared by `fit run`,
`fit
predict` and `profile`: `args::types::SweepSpec` (`args/types.rs:255`,
`V1,V2,…
| lin(min,max,n) | log10(min,max,n)`). Predict's expansion,
`expand_predict_sweep` (`predict.rs:1359`), moves into `DesignAxis::sweep` so
both verbs expand through one function. The batch TOML sweep grammar
(`batch.rs:
143`, `linspace`/`logspace`/`range`) is a different surface syntax
and stays; its expansion (`expand_sweep`, `batch.rs:293`) feeds
`DesignAxis::from_rows`.

Flag relations in `simulate`: `--draws`, `--draws-file` are mutually exclusive
(one sample axis); `--sweep`, `--design` are mutually exclusive (one design
axis); a sample flag and a design flag compose (design × sample, as in
`fit
predict`). `--draws` accepts only `posterior`, `prior`, `uniform`; any
other value is the W9 refusal. The `explicit_file` field (`sim_job.rs:182-191`)
is deleted.

### 7.6 Consumers read the coordinates, never the source

| Consumer                                         | Reads                                | Replaces                                                                             |
| ------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------------------------------ |
| Collision guard (§8)                             | `ParamPlan` design names and origin  | `ParamSource` match at `engine.rs:367-375`; predict's call at `predict.rs:1623-1649` |
| `SimQuantities::push_cell` (`main.rs:2464`)      | `cell.spec.coords().partition_key()` | `cell.spec.scenario.name()`                                                          |
| `PredictiveSink::merge_cell` (`predict.rs:1127`) | same                                 | same, plus the enclosing `sweep_pt` loop variable                                    |
| Quantity `Mode` (`main.rs:1931-1940`)            | `ParamPlan::mode()` (§9.1)           | `matches!(source, Point { replicates: 1 })`                                          |
| `CasSink::cell_resolve` label (`batch.rs:1150`)  | design names ∪ sample columns        | `point_overrides.keys()`                                                             |
| Seed and per-draw lookups (§7.3)                 | `coords.seed_index()`                | `point_idx`                                                                          |

## 8. Collision policy keys on the coordinate kind

`check_scenario_coordinate_collision` keeps its name and its three call sites
(§4.4) but reads the plan instead of `ParamSource`:

```rust
pub fn check_scenario_coordinate_collision(job: &SimulateJob)
    -> Result<Vec<ScenarioTouch>, String>
```

For each scenario it intersects `scenario_param_footprint`
(`params_resolver.rs:698-733`, the shared authority) with:

- the design axis's `names` — non-empty is a hard error. `UserCoordinate`
  (`params_resolver.rs:806`) becomes a view of `DesignOrigin`, so the formatter
  names the origin: "the sweep", "the design `wide`", "the design file
  `grid.tsv`". The `DrawsFile` variant and its wording are deleted.
- the sample axis's columns — non-empty is allowed and returned as a
  `ScenarioTouch { scenario, param, action: Set | Scale }`, which `run_job`
  prints (one stderr line per `set`) and the quantity accumulators copy into
  `summary.scenario_touches`.

Because `fit predict --sweep` builds a plan with a design axis after Stage 4,
its separate call is deleted and the engine's covers it. The footprint∩names
rule is exact, not a stopgap: `2026-08-11` §1.1 shows disjointness is required
for both distinctness (distinct coordinates run distinct models) and denotation
(each emitted coordinate equals the value that ran).

## 9. Banding keys on the coordinate kind

### 9.1 `Mode` is a function of the plan

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

This equals today's predicate (`main.rs:1931-1940`) on every input that still
exists; the file that used to pool (§4.1) now arrives as `--design`, has a
`Unit` sample axis, and renders in point mode per design point.

### 9.2 `simulate` groups by design point, and the manifest says what the band is over

`SimQuantities.by_scenario` becomes
`by_partition: IndexMap<PartitionKey,
ScenarioQuant>`, keyed inside `push_cell`
from the cell. At render each group is one `StackedQuantities::push_group` whose
`DesignCoords` come from one function,
`DesignCoords::of(&PartitionKey, &ParamPlan, scenario_axis)`.
`DesignCoords.
sweep` already renders `sweep:<param>` columns
(`quantity_output.rs:227-243`) and a manifest `sweep` object (`:480-491`);
`simulate` starts populating it.

Each banded manifest entry gains one object, additive to `camdl.quantities/v1`:

```json
"summary": {
  "over": ["sample", "replicate"],
  "measure": "posterior",
  "source": "results/fits/sir-8a3f12b4",
  "n_samples": 200,
  "replicates_per_sample": 1,
  "scenario_touches": { "beta": "scale" }
}
```

`over` lists the band axes actually reduced in that group: `sample` unless the
axis is `Unit` or the group's scenario `set`s every sampled column; `replicate`
unless `replicates_per_sample` is 1 or the backend is ODE. `measure` is
`posterior | prior | uniform_bounds | asserted`, or `process` when only
replicates are reduced, and is absent when `over` is empty. `source` names the
fit or file where one exists. This is the label gh#575 asked for; the `n_draws`
column is not renamed here.

## 10. Why this does not reintroduce the caller-supplied key

Rejection reason 1 was that `BandSet::push(key, …)` let a caller choose the key.
Nothing here takes a key from a caller:

- Both accumulators keep `push_cell(&CellResult)` / `merge_cell(&CellResult)`
  and derive `PartitionKey` from `cell.spec.coords()`. No method has a key
  parameter.
- `CellCoords` is set only by `plan_grid`; `CellSpec`'s fields are private to
  `engine`. `PartitionKey` has no public constructor.
- `StackedQuantities::push_group` still takes `DesignCoords` from its caller.
  That is safe for the reason `2026-08-11` §3.5 gives — it renders groups that
  are already separated and cannot re-merge them — and the coordinates it
  receives are computed by `DesignCoords::of` from a key the accumulator
  derived.

Reason 2 is removed by Stage 4 (predict's sweep is in the cell). Reason 3 is the
subject of §6. Reason 4 no longer holds: `--design` and `simulate --sweep` reach
the design path in Stage 3.

## 11. What changes for users

| #  | Command                                                            | Today                                                     | After                                                                            |
| -- | ------------------------------------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1  | `simulate --draws FILE …`                                          | runs; pools a grid into one band under `--quantities-out` | refused, naming `--design FILE` and `--draws-file FILE` (W9)                     |
| 2  | `simulate --draws-file FILE`                                       | (does not exist)                                          | today's `--draws FILE` behaviour as a sample; allowed under a scenario (W5)      |
| 3  | `simulate --design FILE`, `simulate --sweep P=GRID`                | (do not exist)                                            | one group per point, `sweep:<param>` columns; scenarios on those params refused  |
| 4  | `simulate --draws-out PATH`                                        | writes the file                                           | also prints `reuse as a sample with --draws-file PATH`                           |
| 5  | `batch run [design.NAME]` × colliding scenario                     | refused as "the sweep"                                    | refused as "the design `NAME`"                                                   |
| 6  | Scenario setting a column of generated draws                       | runs silently                                             | runs; one stderr line; `summary.scenario_touches`                                |
| 7  | Any banded quantities output                                       | no statement of what the band is over                     | manifest `summary` object (additive)                                             |
| 8  | `fit predict --sweep …`                                            | —                                                         | byte-identical outputs                                                           |
| 9  | `--design-from` with a file                                        | `--draws FILE`                                            | `--draws-file` or `--design`; writes `design_from.json` recording the kind (W11) |
| 10 | `run.json` provenance tag for a draw or sweep value, where written | `sweep_point`                                             | `sample` or `design_point`                                                       |

**Run identity does not re-key.** Identity hashes resolved parameter values and
the process seed (`resolve.rs:202-233`, verified); §7.3 keeps every seed and
§7.4 every resolved value. The provenance tag in row 10 is recorded, not hashed,
for `sim` leaves (`run.json.inputs` is display-only, `resolve.rs:238-245`); for
fit and profile leaves, whether `parameters_provenance` (`run_meta.rs:644`)
enters any hash is _not verified_ — Stage 6 carries the test that settles it.

**IR and goldens are unaffected.** All changes are in `rust/crates/cli` and
docs. No `ir/VERSION` bump, no `ocaml/` change, no golden regeneration.

## 12. Staged implementation

Each stage lands alone with a green `make test`. Stages 0, 1 and 5 are
byte-neutral; 2, 3 and 4 carry the user-visible changes in §11.

**Stage 0 — one constructor for `CellSpec`.** `CasSink::predict_cells` calls
`plan_grid`; the test fixtures go through a `#[cfg(test)]` constructor. _Test:_
for a sweep × two scenarios × three seeds manifest, `--dry-run`, run,
`--dry-run` again: all-miss, then all-hit with identical leaf paths.

**Stage 1 — `ParamPlan` and `CellCoords`.** Replace `ParamSource` (front ends
build the plan the flags they have today imply; `--draws FILE` still accepted as
an `Asserted` sample so behaviour is unchanged); split `point_overrides`; add
`seed_index`; migrate the §7.6 consumers mechanically. _Tests:_
`tests/determinism_pin.rs` green and unchanged; a unit test planning point,
generated-draws, draws-file and explicit-seed sweep jobs and asserting each
cell's `process_seed` equals `process_seed_for` with the old `point_idx` (the
old formula kept in the test as the oracle); an A/B of
`simulate --draws uniform -n 5 --scenario a,b` and a `batch run` sweep, diffing
every artifact including `run.json`.

**Stage 2 — the flag split and the guard on the plan.** Add `--draws-file` and
`--design`; refuse a bare `--draws FILE` (W9); `--draws-out` prints the
round-trip hint; `DesignOrigin` naming in the formatter; a sample never collides
and scenario touches are returned. Replaces the landed guard's `ParamSource`
keying (`engine.rs:367-375`) — the policy is unchanged. Doc change in the same
stage: the run-spec recipe "Posterior predictive from a draws file"
(`docs/camdl-run-spec.md:1976-1978`) becomes `--draws-file draws.tsv`, the
`--draws` help (`args/mod.rs:386-390`) and `--draws-out` help
(`args/mod.rs:412-417`) are rewritten, and the `--design-from` help names both
flags. _Tests:_ the refusal message for `--draws grid.tsv`, asserted verbatim
including both replacement flags; a round trip —
`--draws uniform -n 5
--draws-out u.tsv`, then `--draws-file u.tsv` with the
same seed — produces leaves with the same `run_id`s as the first run;
`--draws-file u.tsv --scenario
S` (S sets a column) exits 0 with one stderr
line; `--design u.tsv --scenario S` is refused naming "the design file `u.tsv`";
the gh#572 tests (`tests/scenario_coordinate_collision_gh572.rs`) updated for
the new nouns, with a `[design.NAME]` case asserting "the design `NAME`".
Mutation check: make the guard treat the design axis as a sample and confirm the
batch test goes red.

**Stage 3 — design groups, `simulate --sweep`, and the `summary` block.**
`simulate --sweep` through `DesignAxis::sweep` (the lifted predict expander);
`ParamPlan::mode`; `by_partition`; `DesignCoords::of`; the manifest `summary`;
the run-spec sentence on `--draws uniform` (`docs/camdl-run-spec.md:1145-1148`)
corrected to "an iid uniform sample over declared bounds". _Tests:_
`--sweep beta=0.15,0.3,0.6 --backend ode --dt 0.1 --quantities-out q` → three
rows `sweep:beta value` with `peak_I` 70, 304, 536 (the measured values of §3);
`--design grid.tsv` with the same three rows → byte-identical files; the same
with `--replicates 4 --backend chain_binomial` → three band rows, `n_draws = 4`,
`measure == "process"`; `--draws uniform -n 5` → `measure == "uniform_bounds"`,
`over == ["sample"]`; `--sweep` and `--design` together refused by clap.
Mutation check: key `by_partition` on the scenario alone and confirm the
three-row assertion fails on the row count, not the header.

**Stage 4 — `fit predict` on the grid.** One plan, one `run_job` for all
scenarios and sweep points; `merge_cell` keys on `PartitionKey`; predict's guard
call deleted; `conditioned_here` reads the key (conditioned iff the scenario is
the conditioned arm and the design is `Unit` — its three conditions today).
_Tests:_ byte-identity A/B of every predict artifact (predictive TSVs,
quantities TSVs, all manifests, contrasts) for `--sweep k=… --scenario a,b` with
and without a conditioned fit, same seed; the header pin in
`tests/fit_predict_sweep.rs`; the predict sweep-collision test, now through the
engine's guard. The manifest `summary` is added in a separate commit after the
A/B passes.

**Stage 5 — `--design-from` on the new axes.** Accept `--draws-file`,
`--design`, `--sweep`; write `design_from.json`. Byte-neutral for the dataset
files. _Test:_ `--design-from fit.toml --design truths.tsv` writes one `ds_NN/`
per row with `design_from.json` recording `"axis": "design"`; with
`--draws prior -n 3`, `"axis": "sample", "measure": "prior"`.

**Stage 6 — provenance.** `ValueSource::{Sample, DesignPoint}`; the per-group
manifest records `requested` alongside effective design values (§14). _Test:_
one fit leaf and one `sim` leaf before and after — `run_id` unchanged, tag
changed — which settles the unverified question in §11.

## 13. Decisions

**The ambiguous `--draws FILE` splits by axis.** A file's rows are either a
sample the user stands behind or conditions to compare, and no content of the
file says which. Rather than default one way and let the other become a silent
wrong answer, the flag names the axis: `--draws-file FILE` is a sample
(`Measure::Asserted`), `--design FILE` is a design (`DesignOrigin::File`), and a
bare `--draws <path>` is refused with a message naming both. This follows the
alpha posture — break with the replacement spelled out, no shim — and removes
the need for any modifier flag. `--draws posterior|prior|uniform` is unchanged.
`--draws-out` prints `--draws-file <path>` as the round-trip hint, since a
generated sample read back is still a sample. An in-file declaration was not
chosen: the reader takes line 1 as the header (`main.rs:4392-4399`), so a
directive line would break older camdl and every other reader of these files.

**`simulate` gains `--sweep`.** Parity with `fit predict`: the same `PARAM=GRID`
grammar (`args::types::SweepSpec`) and one expansion function shared by both
verbs, producing a `DesignAxis` with origin `Sweep`. It composes with a sample
flag (design × sample) and excludes `--design` (one design axis per job).

**`--draws uniform` is a sample with `Measure::UniformBounds`.** The uniform law
on the declared box is a real measure (§6.1), so a band over it is a correct
Monte Carlo estimate of its push-forward; what it is not is a belief, and the
manifest's `"measure": "uniform_bounds"` keeps it from being captioned as one.
Per-point rows remain one `--draws-out` / `--design` round trip away (W3).

**`fit predict`'s sweep moves onto the grid (Stage 4), gated on byte identity.**
This is what puts predict's partition coordinate in the cell, lets one guard and
one grouping derivation serve both verbs, and removes rejection reason 2. The
objection recorded in `2026-08-11` §4 — only the scenario half of the key was
reachable — no longer holds once `simulate --sweep` and `--design` make the
design half reachable.

**Seeds stay as they are.** Design points share common random numbers in
`batch run` (explicit seeds) and in `fit predict` (the seed follows the draw
index), and not for a design read from a file in `simulate`, where the row index
enters the seed (§7.3). Changing that would alter trajectories and re-key those
runs, and deserves its own justification; it is out of scope here and should be
filed as a separate issue.

**Every design origin uses the `sweep:<param>` column prefix.** One prefix for
"design coordinate" leaves `fit predict` headers and the manifest `sweep` object
unchanged; the documentation defines the prefix as a design coordinate
regardless of origin.

**`--design-from` accepts every axis and records which.** It writes one dataset
per parameter point and reduces nothing, so neither axis changes its datasets;
but whether downstream analysis may pool across them depends on the kind, so the
kind is written into `design_from.json` (W11).

**Contrasts stay sweep-agnostic.** A contrast forks from a smoothed state that
exists only for the un-swept model (W7). Pairing within design points is
well-defined in the coordinate algebra and is left to a free-forward contrast
feature, to be filed separately.

## 14. Requested and effective values for a future scale × sweep composition

A scenario that scales a design parameter is refused (§8), so a design
coordinate's requested value always equals the value the cell ran. If a later
proposal admits composition — sweep `beta ∈ {0.15, 0.3, 0.6}` under `distancing`
running `{0.09, 0.18, 0.36}` — the representation is fixed here:

- **Identity hashes only effective values.** It does already
  (`batch.rs:1141-1153`, `resolve.rs:231`, verified). Cells whose effective
  values coincide under different requested coordinates share a `run_id` and are
  one computation in the store.
- **The grouping key is the requested coordinate** — the index of the axis the
  user chose (`DesignCoord::Point { idx }`), not a value.
- **A column denotes one thing.** `sweep:beta` keeps meaning the requested grid
  value; the effective value, where it differs, is reported under a separately
  named column and in the manifest's per-group `requested`/`effective` pair
  (Stage 6 writes both; they are equal until composition exists).

Admitting composition is out of scope; this section fixes only where its
metadata would go.
