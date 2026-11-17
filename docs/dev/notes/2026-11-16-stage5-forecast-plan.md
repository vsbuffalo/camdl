# Stage 5: the forecast plan and the fixed-origin driver

- Date: 2026-11-16
- Proposal: `docs/dev/proposals/2026-11-15-temporal-architecture-migration.md`,
  Stage 5
- Design: `docs/dev/temporal-architecture.md` §2.4, §2.5, §3.3, §4.3, §4.4

## Scope of this branch

Stage 5 is larger than one reviewable change. It is split into three pull
requests, each leaving `main` consistent:

1. **The plan and the chain-binomial driver** (this branch). `ForecastPlan`,
   `Target` and `StateConditioning`; targets from the declared tiling or the
   emit schedule, refused otherwise (D42); the fixed-origin driver over PMMH and
   PGAS posterior draws (D53, D54, D57); the run extent past `simulate { to }`
   (D49); `fit predict --mode forecast --horizon DURATION` with the
   `--horizon`-to-`--mode` rename (D41); the refusals of the capability matrix
   (D51) and of a design column (D52); the forecast artifact.
2. **The ODE path and scenario arms.** MH on ODE through the cumulative-flow
   difference on each draw's trajectory, and post-origin scenarios as same-seed
   arms (D50). Until then an ODE posterior and any scenario, sweep or
   `--enable`/`--disable` in forecast mode are refused with a message.
3. **`fit predict` reads what each row measures** (D69, gh#1151, gh#1152): the
   one-step rows marked `trained`/`held_out`, the predictive TSV's `role`,
   `t_start`, `t_stop`, `horizon` and `lead` columns (gh#624), deletion of
   `forecast_times`, `leaf_row_coverages`, `ObservedRow.label` and
   `Observation.time`, and `compare --allow-mismatched-scored-steps`.

Out of scope, by decision: restarts (D39's refusal clause is being redesigned
under D64 and D72; `simulate --init-state` and `pfilter --save-final-state`
behave as on `main`), and forecast scoring (Stage 6).

## Types

```rust
// rust/crates/cli/src/fit/forecast_plan.rs

/// One stream's period (or instant) the forecast predicts (§2.5).
pub(crate) struct Target {
    pub stream: String,          // the leaf (observation block) name
    pub measures: Measures,      // the period or instant predicted (§3.1's type)
    pub horizon: i64,            // position among the stream's targets; 0 = first ending after the origin
    pub lead: f64,               // end − origin, in model time units
    pub observed: Option<usize>, // position of a held-out row measuring the same period
}

/// Where a stream's targets come from (§4.3); recorded on the plan.
pub(crate) enum TargetSource {
    DeclaredTiling { span: f64 },  // a uniform `covers` form
    EmitSchedule,                  // window columns or an instant stream, with `emit_schedule`
}

/// What one stream contributes to a plan: its declaration, the rows the
/// parameters were fitted to, and the rows after them.
pub(crate) struct PlanStream<'a> {
    pub obs: &'a ir::observation::ObservationModel,
    pub training: &'a [ObservedRow],
    pub held_out: &'a [ObservedRow],
}

pub(crate) enum PlanError {
    NoRhythm { stream: String, why: NoRhythm },   // D42
    NoBoundRows { stream: String },               // no phase to continue
    OverlapsObserved { stream: String, target: (f64, f64), row: usize },
    ReadsDataColumn { stream: String, columns: Vec<String> }, // D52
    BadHorizon(String),
    NoTargets,
}

/// What is to be predicted (§3.3). Built once; every target is a declared
/// period of its stream before any filtering.
pub(crate) struct ForecastPlan {
    origin: f64,                     // the training extent (D57)
    horizon: f64,                    // the reach, a duration in model units
    parameters: TrainingPartition,   // from the fit record, digest-verified
    targets: Vec<Target>,            // sorted by (end, stream)
    sources: Vec<(String, TargetSource)>,
}

impl ForecastPlan {
    pub(crate) fn new(parameters: TrainingPartition, horizon: f64,
                      streams: &[PlanStream]) -> Result<Self, PlanError>;
    pub(crate) fn targets_of(&self, stream: &str) -> impl Iterator<Item = &Target>;
    pub(crate) fn run_end(&self) -> f64;  // latest target end (D49)
}

/// The rows the state at the origin was conditioned on (§2.4). Built only by
/// the driver, from the rows it handed the filter.
pub(crate) enum StateConditioning {
    Filtered { through: f64, rows: Vec<RowPrefix> },
}

/// The driver's result: the plan, what conditioned the state, and the pooled
/// predictive draws, `draws[target][sample]`, sample = draw × particle in
/// draw-major, particle-minor order, so sample `s` across targets is one
/// particle's trajectory (D54).
pub(crate) struct Forecast {
    pub plan: ForecastPlan,
    pub state: StateConditioning,
    pub n_draws: usize,
    pub n_particles: usize,
    pub draws: Vec<Vec<f64>>,
}

/// Why a forecast failed (D53): no draw is dropped.
pub(crate) struct ForecastFailure { draw: usize, time: f64, cause: String }

// rust/crates/cli/src/fit/forecast_driver.rs
pub(crate) fn fixed_origin_forecast(
    plan: ForecastPlan,
    training: Vec<ObsStream>,           // already cut to the partition's prefixes
    compiled: Arc<CompiledModel>,       // the fitted model, run extent applied
    param_sets: &[Vec<f64>],            // one per posterior draw
    smc: SMCConfig, base_seed: u64,
) -> Result<Forecast, ForecastFailure>;
```

`StateConditioning` has the one variant this branch constructs. The `None`
variant (ODE, free-running) arrives with the ODE path; the plan's `mode`
(`PredictiveMode`) arrives when the one-step and free-forward modes read the
plan (PR 3). Until then the forecast artifact records the mode as the string
`fixed_origin`.

## The seams

- `ForecastPlan::new` is the only place a target is made. The driver appends
  every target to its stream as a row with no value, binds the training rows
  plus those rows through `BoundObs::bind`, and reads the per-stream samples the
  filter already records (`PrequentialRecorded.per_stream_samples`) at each
  target's end. No change to `particle_filter.rs`, resampling, or the
  saved-state format.
- A target ending at a union time where its stream is not scheduled is an
  internal error, never a read of a sibling's value: the reading is by the
  stream's own `at_union` membership (a NaN sample there is refused).
- Deaths after the origin (D53): a dead particle's log-weight is `−∞` at the
  next scoring step, and after the origin every cell is a row with no value,
  whose log-likelihood is 0 for a live particle. So a `−∞` log-weight at any
  step after the origin is a death, and fails the forecast naming the draw, the
  step's time and the cause.
- Run extent (D49): the fitted model's `simulation.t_end` is raised to the
  latest target end before compiling, so `refuse_data_past_end` admits the
  target rows and the effect plan is built to that end. A recurring schedule
  with a literal end keeps it; a forcing past its range is Stage 0(f)'s error.

## Decisions taken inside the design

- **`--horizon` alone implies `--mode forecast`.** `--mode forecast` without
  `--horizon` is refused (there is no reach), and `--horizon` with another mode
  is refused. The alternative, requiring both flags, adds a flag that carries no
  information.
- **Joint samples file.** `forecast-samples.tsv` holds every particle of every
  draw would be draws × particles × targets rows (200 × 500 × 32 ≈ 3.2 million
  for four weeks of a daily and a weekly stream). It holds
  `FORECAST_SAMPLES_PER_DRAW = 5` particles per draw, evenly strided, each a
  complete joint sample (equal weights after the origin make every particle a
  draw from the same predictive). The bands use every particle.
- **Held-out positions.** `Target.observed` is filled from the rows of the
  training files past the prefix (`holdout_after`) and from `[data.holdout]`
  files, matching a row whose measured period equals the target's within
  `TIME_TIE_EPS`.
