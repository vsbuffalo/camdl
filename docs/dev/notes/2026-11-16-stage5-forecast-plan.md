# Stage 5: the forecast plan and the fixed-origin driver

- Date: 2026-11-16
- Proposal: `docs/dev/proposals/2026-11-15-temporal-architecture-migration.md`,
  Stage 5
- Design: `docs/dev/temporal-architecture.md` §2.4, §2.5, §3.3, §4.3, §4.4

## Measurement: the driver against an importance-sampling reference

**Fixture.** `rust/crates/cli/tests/fixtures/forecast/two_cadence_sir.camdl`, a
chain-binomial SIR (population 1000, five initial infectives, β = 0.5, γ = 0.25,
one-day steps) with two reports of the same infections: a daily Poisson count
with mean ρ·(the day's infections), and a weekly negative-binomial count with
mean ρ·(the week's infections) and size k (ρ = 0.5, k = 10). `daily_cases.tsv`
and `weekly_cases.tsv` beside it are one simulated realization
(`camdl simulate … --seed 7 --obs-only-dir`). The parameters are fixed at the
values that generated the data, so the comparison isolates the state
conditioning from parameter uncertainty.

**The question.** The state is conditioned on the daily counts of days 20–24
(each row covers the day ending at its time); the weekly rows through day 21
carry no value. The forecast origin is day 24. The target is the week [21, 28),
under way at the origin: three of its days are inferred from the daily data and
four are predicted. What is the distribution of the week's infections W?

**Reference.** Importance sampling from the prior: 100,000 forward simulations
of the chain-binomial process, each weighted by the Poisson likelihood of the
five daily counts, computed by hand in the test rather than by camdl's
likelihood code (effective sample size 4,918).

**Reproducing every figure below.** One ignored test runs the high-precision and
the default comparisons and the two figures the reference rules out, asserts
each as the default tests do, and writes their figures to a file (tests may not
print):

```console
$ cargo test --release -p cli --bin camdl the_measurement_table -- --ignored
test fit::forecast_driver::tests::the_measurement_table ... ok
$ cat "$TMPDIR/camdl-stage5-forecast-measurement.txt"
high precision: W mean: driver 170.410 ± 0.101, reference 170.456 ± 0.119, z = -0.30; ...
```

The runs are seeded, so the figures are the same on every run.

**Pass rule, stated before measuring.** Each comparison passes at |z| < 3, where
z is the difference over the combined Monte Carlo error (the driver's from the
spread of its per-draw estimates, the reference's from the weighted sample).

**Result at high precision** (`the_measurement_table`, about 1.5 minutes in
release: 800 draws × 1,000 particles against 1,000,000 reference simulations,
effective sample size 52,559; combined error on the mean 0.156):

| Quantity                              | Driver          | Reference       | z     |
| ------------------------------------- | --------------- | --------------- | ----- |
| Mean of W                             | 170.410 ± 0.101 | 170.456 ± 0.119 | −0.30 |
| Variance of W                         | 1162.91 ± 6.33  | 1177.76 ± 6.01  | −1.70 |
| Mean of the reported weekly count     | 85.221 ± 0.061  | 85.228 ± 0.060  | −0.08 |
| Variance of the reported weekly count | 1131.49 ± 3.48  | 1135.50 ± 1.65  | −1.04 |

The reported count's reference moments are those implied by the reference's W
and the declared noise: ρ·E[W] and ρ²·Var[W] + E[ρW + (ρW)²/k].

**Result at the default test's precision** (test
`the_week_under_way_matches_the_importance_sampling_reference`, run by
`make test`: 40 draws × 1,000 particles, 100,000 simulations, effective sample
size 4,918): mean of W 169.863 ± 0.470 against 170.868 ± 0.381 (z = −1.66);
variance 1193.77 ± 26.87 against 1180.20 ± 18.97 (z = 0.41); reported mean
84.950 ± 0.246 against 85.434 ± 0.190 (z = −1.56); reported variance 1129.70 ±
10.83 against 1139.88 ± 5.22 (z = −0.85). The gap of 1.0 in the mean at this
precision does not persist at four times the precision (0.05, z = −0.30), so it
was Monte Carlo error rather than an edge or origin error.

**What the existing method gives on the same question** (test
`the_conditioning_and_the_declaration_each_move_the_week_by_many_errors`, 50,000
reference simulations each):

| How the week is computed                                                              | Mean of W     |
| ------------------------------------------------------------------------------------- | ------------- |
| Conditioned on the daily data (the forecast's target)                                 | 171.19 ± 0.51 |
| Not conditioned (the prior)                                                           | 218.91 ± 0.29 |
| Read at day 28 with no reset at day 21 (the week not declared; includes days 14 – 20) | 492.81 ± 0.57 |

The prior's week is centred 48 infections higher than the conditioned one (prior
standard deviation 65.7). The free-forward band, which `fit predict` already
extends past the data, is a replay from `init {}` that never sees the
observations, so at these fixed parameters it would centre on the prior. That is
inferred from what the replay is; `fit predict --mode free_forward` was not run
on this question, and on a fit it replays posterior draws rather than fixed
parameters. A reading of the stream's running total at day 28 without the week
declared holds the previous week's infections as well, 2.9 times the conditioned
value, with no error raised. The plan declares every target, and the driver
refuses a reading at any time that ends no declared period
(`a_reading_that_ends_no_declared_period_is_refused`).

**Mutation check.** Declaring the week under way from the origin (the filter
period [24, 28)) instead of its start fails the reference test: driver mean
79.561 ± 0.360 against 170.868 ± 0.381, z = 174. The mutation, applied to
`specs_with_targets` in `forecast_driver.rs` and reverted after:

```console
$ sed -i '' 's/ps.push(Period::new(p.start(), p.stop())/ps.push(Period::new(p.start().max(plan.origin()), p.stop())/' \
    rust/crates/cli/src/fit/forecast_driver.rs
$ cargo test -p cli --bin camdl importance_sampling
mean of the week's infections: driver 79.561 ± 0.360 vs reference 170.868 ± 0.381 (z = 174.17)
```

The spec's own experiment (68.245 ± 0.021 against 68.227 ± 0.034; 256.95
undeclared) used a model and data that are not committed; its numbers are not
reproducible here, and this fixture reproduces its structure (a daily and a
weekly stream, origin day 24, target days 21–27, fixed parameters, the reference
weighted by the daily data) with its own numbers.

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

As built (`rust/crates/cli/src/fit/forecast_plan.rs`, `forecast_driver.rs`,
`forecast_output.rs`):

```rust
/// One stream's period (or instant) the forecast predicts (§2.5).
pub(crate) struct Target {
    pub stream: String,              // the leaf (observation block) name
    pub measures: Measures,          // the period or instant predicted (§3.1's type)
    pub horizon: i64,                // position among the stream's targets; 0 = first ending after the origin
    pub lead: f64,                   // end − origin, in model time units
    pub observed: Option<HeldOutRow>, // a held-out row measuring the same period
}

/// A held-out row's file and its position within that file.
pub(crate) struct HeldOutRow { pub file: RowFile, pub position: usize }
pub(crate) enum RowFile { Training, Holdout }       // the training file, or `[data.holdout]`
pub(crate) struct HeldOut { pub file: RowFile, pub row: ObservedRow }

/// Where a stream's targets come from (§4.3); recorded on the plan.
pub(crate) enum TargetSource {
    DeclaredTiling { span: f64 },  // a uniform `covers` form
    EmitSchedule,                  // window columns or an instant stream, with `emit_schedule`
}

/// What one stream contributes to a plan.
pub(crate) struct PlanStream<'a> {
    pub obs: &'a ObservationModel,
    pub training: &'a [ObservedRow],   // the stream's training prefix
    pub held_out: &'a [HeldOut],       // rows after it, in either file
}

/// Why one stream has no targets (it is recorded; the others still forecast).
pub(crate) enum StreamRefusal {
    NoRhythm { stream: String, kind: TemporalKind },             // D42
    NoBoundRows { stream: String },                              // no phase, no stop
    ReadsDataColumn { stream: String, columns: Vec<String> },    // D52
    OverlapsObserved { stream: String, start: f64, end: f64, row: usize },
}

/// Why no plan can be built at all.
pub(crate) enum PlanError {
    BadHorizon(f64),
    NoTargets { refusals: Vec<StreamRefusal> },
}

/// What is to be predicted (§3.3). Every target is declared before filtering.
pub(crate) struct ForecastPlan {
    origin: f64,                         // the training extent (D57)
    horizon: f64,                        // the reach, a duration in model units
    parameters: TrainingPartition,       // from the fit record, digest-verified
    targets: Vec<Target>,                // sorted by (end, stream)
    sources: Vec<(String, TargetSource)>,
    refusals: Vec<StreamRefusal>,
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

/// The driver's result: `draws[target][sample]`, sample = draw × particle in
/// draw-major order, so sample `s` across targets is one particle (D54).
pub(crate) struct Forecast {
    pub plan: ForecastPlan,
    pub state: StateConditioning,
    pub n_draws: usize,
    pub n_particles: usize,
    pub draws: Vec<Vec<f64>>,
    pub expected: Option<Vec<Vec<f64>>>,   // E[y | x, θ] per sample; tests only
}

/// Why a forecast failed (D53). `draw` indexes the parameter sets handed to
/// the driver; `describe` lets the caller name it by its draws.tsv key.
pub(crate) struct ForecastFailure { pub draw: Option<usize>, pub time: Option<f64>, pub cause: String }

pub(crate) struct DriverConfig {
    pub n_particles: usize, pub dt: f64, pub base_seed: u64, pub record_expected: bool,
}

pub(crate) fn fixed_origin_forecast(
    plan: ForecastPlan,
    training: &[ObsStream],             // every bound stream, cut to its prefix
    compiled: Arc<CompiledModel>,       // the fitted model, run extent applied
    param_sets: &[Vec<f64>],            // one per posterior draw
    config: DriverConfig,
) -> Result<Forecast, ForecastFailure>;

/// The one place targets become filter stops (see "The seams").
fn specs_with_targets(plan: &ForecastPlan, training: &[ObsStream])
    -> Result<Vec<StreamSpec>, ForecastFailure>;

/// A posterior draw the forecast ran, as `forecast_output.rs` names it.
pub(crate) struct DrawRef { pub cloud_row: usize, pub key: Option<(usize, usize)> }
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
  target rows and the effect plan is built to that end
  (`the_run_extends_past_its_end_and_a_later_intervention_fires`). That a
  recurring schedule with a literal end keeps it, and that a forcing past its
  range is Stage 0(f)'s error, is inferred from the effect plan and the forcing
  check, not tested here.

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
- **Held-out rows.** `Target.observed` is filled from the rows of the training
  files past the prefix (`holdout_after`) and from `[data.holdout]` files,
  matching a row whose measured period equals the target's within
  `TIME_TIE_EPS`, and names the file beside the position counted within it.
- **Draws are named by their key.** The samples file, `forecast.json` and a
  failure name each posterior draw by its row of `draws.tsv` and its
  `(chain, draw)` key there, never by its place in the `--n-draws` subsample.
