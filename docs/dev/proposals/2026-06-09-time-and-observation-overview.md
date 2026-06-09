# Time, observations, and the run: a system overview

- **Status:** Overview / reader's map. This is the entry point to a
  three-proposal stack; it explains _how time and observations are meant to
  work_ end-to-end and marks clearly what **exists in code today** vs what is
  **proposed**. Read this first, then the proposal it points you to.
- **The three proposals it ties together:** `2026-06-06-observation-system.md`
  (data layer), `2026-06-06-scheduling-effect-topology.md` (temporal spine),
  `2026-06-09-time-interval-model.md` (intervals / forecast / forcing domains).
- **Audience:** a colleague new to the codebase. No code reading assumed; file
  references are signposts, not prerequisites.

## 1. The one-paragraph problem

A camdl run juggles several different notions of "time," and today no single
component owns them. Where the **dynamics** run, where the **data** lives, where
**output rows** are written, and how far you want to **forecast** are tracked by
separate fields read by separate code, kept consistent only by convention. The
gaps between these uncoordinated clocks are where the system can produce a
_silent wrong answer_ — a fit that quietly scores the wrong thing. The work
across the three proposals is to name these intervals explicitly, make one
component reconcile them, and turn every "the intervals don't line up" case into
a loud error instead of a silent mis-score.

## 2. The three layers (and who owns what)

Think of it as a stack. Each layer owns one job and consumes the layer below.

```
┌─────────────────────────────────────────────────────────────────────┐
│ INTERVAL layer   (2026-06-09-time-interval-model.md)                  │
│   the run's time-axis windows + their reconciliation;                 │
│   the conditioning window C; the forecast horizon F; forcing domains  │
├─────────────────────────────────────────────────────────────────────┤
│ TEMPORAL SPINE   (2026-06-06-scheduling-effect-topology.md)           │
│   Schedule (when does time stop next); TemporalKind (Interval/Instant);│
│   per-(observer) accumulator resets; StepPolicy (Snap/Exact);          │
│   the sub-dt collision guard                                           │
├─────────────────────────────────────────────────────────────────────┤
│ DATA layer       (2026-06-06-observation-system.md)                   │
│   raw rows → typed per-stream cells with holes (missing values);       │
│   bind / BoundObs; the bind report                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**What exists today vs what is proposed** — this matters, because most of the
spine is still a design map:

| Piece                                                                                                                           | Status                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `simulation` config `{t_start, t_end, dt}` in the IR                                                                            | **exists**                                                                                                                       |
| `output` config (a separate output window)                                                                                      | **exists**                                                                                                                       |
| Three forward backends — chain-binomial, Gillespie, ODE                                                                         | **exists**                                                                                                                       |
| Inference cells — bootstrap PF, IF2, PGAS, PMMH, correlated-PF                                                                  | **exists** (all stochastic inference runs the chain-binomial process — there is one process implementation, not one per backend) |
| `Schedule` time substrate (three parallel time vectors: output / effect / obs, walked by a `Cursor`) + `StepPolicy{Snap,Exact}` | **exists** (the spine's "Tier 0")                                                                                                |
| `cum_flows` incidence accumulator — **one shared vector, reset globally at every obs**                                          | **exists** (and is the thing the per-observer redesign replaces)                                                                 |
| Capability gate — `required_capabilities()` vs each backend's `capabilities()`, plus the `(algorithm × backend)` registry       | **exists** and is healthy                                                                                                        |
| `ic_free` (`skip_first_obs_from_loglik`) — drop the first likelihood term; honored by PF + IF2 only                             | **exists**                                                                                                                       |
| PGAS persists posterior latent trajectories to disk; IF2/PMMH persist parameters only                                           | **exists**                                                                                                                       |
| `TemporalKind`, per-observer `ResetWindow`, `Observe`/`Stage` effects                                                           | **proposed (0% built)** — the spine's headline work                                                                              |
| `bind` / `BoundObs` / typed cells / holes                                                                                       | **proposed**                                                                                                                     |
| `RunWindows`, the conditioning window `C` / `condition_from`, the `forecast` operation, forcing-domain bounds                   | **proposed**                                                                                                                     |
| Typed time (`ModelTime`/`StepIndex`)                                                                                            | **does not exist** — time is `f64` everywhere; we are _not_ adding newtypes now (we fix two concrete bugs instead)               |

## 3. The cast of intervals

A run is described by a small set of named intervals. The whole design is "name
them, validate their orderings in one place."

| Interval                                     | Plain meaning                                           | Home                             |
| -------------------------------------------- | ------------------------------------------------------- | -------------------------------- |
| **D** — dynamics `[t_start, t_end]`          | when the process is integrated                          | derived per operation            |
| **Wₖ** — observation window (per stream _k_) | when stream _k_ has data (fit) or is emitted (simulate) | data layer (from the bound data) |
| **C** — conditioning `[cond_from, cond_to]`  | the sub-span where the likelihood actually scores       | fit                              |
| **Eⱼ** — covariate/forcing domain            | where a time-indexed forcing has real data              | model (with the forcing)         |
| **O** — output `[out_start, out_end]`        | when output rows are written                            | model default / operation        |
| **F** — forecast horizon                     | how far to project beyond the data                      | a per-run operation parameter    |

The master picture, drawn on one timeline:

```
       t_start     cond_from          data…          cond_to     horizon
          |            |                                |            |
burn-in   |············|                                |            |
condition              |================================|            |
forecast                                                |············|
dynamics  |=====================================================…F…==|
stream A       [obs·········obs]
stream B               [obs···············obs]
forcing E [knots······································]   ← must cover the run
```

Read it as: dynamics (**D**) span everything that must be integrated; the
conditioning window (**C**) and the per-stream observation windows (**Wₖ**) are
sub-spans used by fitting; a forcing's data domain (**Eⱼ**) must cover whatever
you integrate; the forecast horizon (**F**) extends the right edge, with no
conditioning past the data.

## 4. How time works, per operation

The key idea: **there is one integration span D per operation, _derived_ from
the windows that operation needs — not a single stored field meaning five
things.**

- **Forward simulate** — integrate `[t_start, out_end or --until]`, emit output
  rows, no scoring. Synthetic observations (if requested) are emitted on the
  output schedule. _(Today: forward sim reads `simulation.t_end` directly and
  the output window is a separate field; the proposal reconciles them.)_
- **Fit** — integrate `[t_start, last obs]`, score the likelihood over the
  conditioning window **C** intersected with each stream's data. The leading
  span `[t_start, cond_from)` is a _warm-up_: dynamics run (covariates,
  campaigns, process noise) but the incidence accumulator is reset at
  `cond_from` so the first scored window is one normal cadence, not the whole
  gap.
- **Forecast** — project the _fitted posterior_ forward to **F**, with no
  conditioning. This is method-dependent: PGAS already stores posterior latent
  trajectories, so you continue from their end-states (`[last_data, F]`); an MLE
  fit (IF2/ODE) stores only parameters, so it must re-filter to recover the
  data-end state, then project (`[t_start, F]`). **A forecast must not re-key
  the fit** — extending the horizon reuses the fit; it does not trigger a
  re-fit.

## 5. Observations: binding, holes, and incidence

### 5.1 Bind, not join

Loading data is a directional **bind**: the model defines a fixed lattice of
named cells (which streams, which strata, which times) and data values are bound
_into_ those slots, like binding arguments to parameters. Two failure
directions, kept distinct: a **leftover** (a data row with no model cell — a
typo'd stream) is usually a mistake; a **hole** (a model cell with no data) is
often expected (sparse surveillance). Columns are bound **by name**, never
guessed from content. _(Today the single-stream loader silently falls back to
positional binding — a typo'd value-column header binds column 1 with no
warning; the proposal deletes that fallback.)_

### 5.2 A hole is a missing value, and it is marginalized — never imputed

A "hole" is a missing value (an NA / blank / absent cell). In a particle filter,
_not_ conditioning on a missing observation leaves the predictive distribution
as the marginal over the unobserved value — so **omitting the likelihood factor
_is_ the marginalization** (∫ p(y|x) dy = 1). We do not sample or impute the
missing value, and there are **zero per-hole random variables** — a hole is pure
bookkeeping. A missing cell must score _differently_ from an observed `0`.

This follows the reference framework: pomp's `dmeasure` returns likelihood 1 for
a missing/NA observation — "the probability of seeing something is 1 if you
don't look" ([pomp FAQ](https://kingaa.github.io/pomp/FAQ.html)). We surface the
assumption (a `Warn` finding when a stream has holes: "the fit assumes these are
missing-at-random") rather than asserting it silently — because the one case
where it fails, _reporting that breaks down because counts are high_, arrives as
a hole.

### 5.3 Incidence is fixed-bin: a missing value suppresses the score, not the reset

This is the subtle, load-bearing part. An **incidence** observation ("new cases
this week") is scored against flow accumulated over a window. Two ways to define
the window:

- **Fixed bins** — "cases in week 3" always means week 3 only; a missing week 2
  does _not_ merge into week 3. **This is how surveillance works, and it is the
  default.**
- **Cumulative since last report** — a missing week 2 means week 3 covers weeks
  2+3. The unusual case; an opt-in a stream declares.

So the accumulator resets on the stream's **cadence** (its scheduled reporting
boundaries), _not_ on whether a value happened to arrive: a
scheduled-but-missing week still resets the tally at the week boundary (the
unscored flow is discarded, not merged), and it contributes no likelihood term.
This is exactly pomp's accumulator-variable behavior — accumvars reset "prior to
any `rprocess` computation over [each] interval between successive observations"
([pomp `accumvars`](https://kingaa.github.io/manuals/pomp/html/accumvars.html);
King, Nguyen & Ionides 2016; He, Ionides & King 2010). And pomp's idiom for "the
first count is the last interval, not cumulative-since-t₀" — insert a fictitious
NA observation at an intermediate time so the accumulator resets there — **is
exactly our `condition_from`**: a reset at a scheduled-but-unscored boundary.

The consequence for the data model: distinguish "scheduled time, value missing"
(reset, no score) from "no scheduled opportunity here" (the leading burn-in — no
reset until `cond_from`).

### 5.4 Per-observer accumulators, not per-flow

Two streams can read the _same_ model flow at different cadences — weekly cases
and monthly cases both off the S→I infection flow. Today the accumulator is one
shared per-flow vector reset globally, so a per-stream reset would zero the
other stream's accumulation. So each _observer_ gets its own accumulator over
its source flows, reset on its own cadence. (Scope: these accumulators feed only
the _observation_ score — the process/path-prior density reads per-substep
records, not the cumulative accumulator — so this never touches the transition
math.)

## 6. The overlap configurations — and what each does today

These are the distinct ways the observation and dynamics intervals can relate.
Each is tagged with _current_ behavior; the design makes them uniform and loud.

```
1. BURN-IN  (data starts after dynamics)         today: SILENT over-accumulation
   D |==================================|         (the first incidence window is
   W        [obs··········obs]                     the whole gap; gh#134)

2. FIT-TO-PAST + FORECAST TAIL                    today: inference ignores t_end;
   D |==================================|         forecasting re-keys the model
   W   [obs······obs]      forecast →

3. DATA PAST DYNAMICS END                         today: INCONSISTENT — silent on
   D |==============|                              chain-binomial/PGAS, hard-error
   W   [obs················obs]                     on ODE

4. DATA BEFORE DYNAMICS START                     today: SILENT-WRONG — loaded and
        D |==============|                         scored without ever propagating
   W [obs····obs]                                  to it (a confirmed bug)

5. PER-STREAM STAGGER                             today: HARD-ERROR ("heterogeneous
   W_A [obs······obs]                              schedules not supported"); the
   W_B      [obs········obs]                        per-stream feature is faked

6. FORCING DOMAIN SHORTER THAN RUN                today: SILENT flat-extrapolation
   D/F |================================|         past the data (the sibling table
   E   [knots········]                             mechanism hard-errors — opposite)

7. OUTPUT vs DYNAMICS END                         today: FROZEN TAIL — emits the
   D |==============|                              terminal state past the dynamics
   O |=============================|              (gh#143)
```

## 7. The type designs

The design principle is **make illegal states unrepresentable, minimally** —
push the invariant into a constructor so the bad state can't be built, without
proliferating types past the natural seam.

### 7.1 `RunWindows` — one authority for the time axis (interval layer)

```rust
struct RunWindows {
    dynamics:     Interval,          // D = [t_start, t_end]; t_end > t_start enforced
    conditioning: Option<Interval>,  // C = [cond_from, cond_to] ⊆ D; None ⇒ no warm-up
    output:       OutputSchedule,    // O, validated ⊆ D
    forecast:     Option<f64>,       // F: horizon ≥ D.end, an operation parameter
    // per-stream observation windows Wₖ are NOT here — they live in BoundObs;
    // this type carries only the single global axis they derive against.
}
impl RunWindows { fn new(...) -> Result<Self, Error> { /* enforces every ordering */ } }
```

Built once per operation, its constructor enforces
`t_start ≤ cond_from < cond_to
≤ D.end ≤ F` and "every window ⊆ D" — so the
three-disagreeing-ends bug and the data-before-`t_start` bug become
_unconstructible_. **Caveat (stated as bluntly as the data layer states its
own):** this only holds once every backend reads the span _through_ `RunWindows`
and the direct IR-field reads are deleted — otherwise it's a validation pass in
a type's clothing.

### 7.2 `obsdata` — typed observations with holes (data layer)

```rust
mod obsdata {
    // input: one untyped row per PRESENT observation
    struct LongRow { stream: String, stratum: Option<String>, when: RawTime,
                     value: RawValue, denom: Option<RawValue> }  // denom: k-of-n streams only
    enum RawValue { Num(f64), Missing, Unparseable(String) }     // Missing ≠ Num(0.0)

    // output: a model-shaped, fully-typed object
    pub struct BoundObs { times: Vec<f64>, streams: Vec<StreamCells> }
    struct StreamCells {
        name:        String,
        kind:        TemporalKind,            // Interval (incidence) | Instant (prevalence)
        cadence:     ObsCadence,              // the reset schedule — separate from value presence
        cells:       Vec<Option<ObsCell>>,    // None = missing value (a hole), one per `times`
        accumulator: Option<AccumulatorId>,   // per-(observer) for Interval streams
    }
    enum ObsCell { Scalar(f64), Counted { value: f64, denom: f64 } }  // built via checked ctor: value ≤ denom

    // a FATAL bind yields no BoundObs — an invalid one must not exist downstream
    pub fn bind(...) -> Result<(BoundObs, BindReport /* warn+info */), BindReport>;
}
```

`Missing` is a distinct value from `Num(0.0)` (hole ≠ zero, enforced by the
type); the `cadence` drives the reset, the `Option` cell drives the score
(§5.3); the `accumulator` is per-observer (§5.4); `bind` returns `Result` so a
fatally-bad file produces _no_ `BoundObs`.

### 7.3 `ForecastOrigin` — keyed by what the fit stored, not the method name

```rust
enum ForecastOrigin {
    PosteriorTerminalState,   // PGAS: continue from stored latent end-states
    FilteredParticleCloud,    // bootstrap PF with saved paths
    ParameterOnly,            // IF2 / PMMH / ODE-MLE: re-filter from t_start, then project
}
```

This mirrors a _real, existing_ structural difference between the fit artifacts
— PGAS persists latent trajectories, IF2/PMMH persist parameters only. It is the
same idea as the future `DifferentiableObjective` trait (value + gradient) that
would let ODE→NUTS slot in: **key on the capability, not the backend name.**

## 8. The pomp precedent (so we're following, not inventing)

The incidence/missing-data design is pomp's, point for point:

| camdl                                                                | pomp                                                |
| -------------------------------------------------------------------- | --------------------------------------------------- |
| fixed-bin incidence: reset on the cadence                            | accumvars reset at each observation interval        |
| a missing value suppresses only the likelihood term                  | `dmeasure` returns 1 for NA                         |
| `condition_from`: reset the flow at the burn-in boundary             | a fictitious NA observation at an intermediate time |
| reset keys on the observation schedule; score keys on value presence | reset on obs-interval grid; `dmeasure(NA)=1`        |

Sources:
[pomp `accumvars`](https://kingaa.github.io/manuals/pomp/html/accumvars.html),
[pomp FAQ](https://kingaa.github.io/pomp/FAQ.html); King, Nguyen & Ionides
(2016) _JSS_; He, Ionides & King (2010) _J. R. Soc. Interface_.

## 9. Declined alternatives (on the record)

An external review proposed a larger type taxonomy. We are _keeping the
diagnoses_ and _declining the types_, per "consolidate to the natural seam, not
past it":

- **A four-layer
  `RunPlan`/`ObservationPlan`/`TemporalPlan`/`BackendExecutionPlan` stack** —
  declined. `RunWindows` is the single natural authority; four parallel planning
  objects are god-layering for a problem that is "one axis, three drifting
  fields."
- **`ObservationBoundaryKind{Opportunity, PresentValue, KnownMissing}`** —
  declined as a type; the _insight_ (opportunity vs value-presence) is kept and
  implemented as "reset on cadence, score on `Option` cell" (§5.3).
- **`ModelTime`/`GridStep`/`CalendarTime` newtype hierarchy** — deferred. The
  bugs are real (a release-only negative-cast saturation; tolerances scattered
  as `1e-9`/`1e-10`/exact); we fix those two concretely rather than newtype the
  whole codebase.
- **A `TemporalEvent` enum** — declined; redundant with the existing `Schedule`
  (three parallel time vectors) × `Stage` (within-substep order) decomposition.

Adopted from the review: per-observer accumulators (§5.4), `bind` returning
`Result`, derived severity + structured findings, `ForecastOrigin` by artifact
capability, interval-scoped CAS hashing for forecast covariates.
