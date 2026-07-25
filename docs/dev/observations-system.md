# Observation system

How synthetic-observation emission (forward simulation) and
observation-likelihood evaluation (inference) are organised in the tree.
Reference for contributors — the compiler already accepts the user-facing DSL
syntax documented in `docs/camdl-language-spec.md` §12.

This replaces an earlier design-plan document (pre-implementation) that proposed
a separate `observe/` crate. The implementation diverged from that plan:
observations live in `sim/src/inference/`, and the `observe/` crate is a
vestigial stub.

---

## 1. Surface

Users invoke the observation system through these entry points:

- **Forward simulation with synthetic observations:**
  - `camdl simulate model.camdl --obs cases.tsv` — writes one wide-format TSV
    with columns `time, stream1, stream2, …`.
  - `camdl simulate model.camdl --obs-dir out/` — writes one TSV per stream
    under `out/`.
  - `camdl simulate model.camdl --obs-only cases.tsv` — same as `--obs` but
    suppresses trajectory output.
  - `--cas` mode: synthetic observations cached under
    `<sim_dir>/obs/<obs_hash>-<obs_seed>/`.
- **Inference against observed data:**
  - `camdl pfilter MODEL --data cases.tsv` — bootstrap particle filter with
    `obs.tsv` as observation stream.
  - `camdl fit run fit.toml` — the `[data]` table in `fit.toml` maps stream
    names to file paths. An MLE-only fit is a `fit.toml` with a single
    `[stages.X] algorithm = "if2"` stage.

## 2. Likelihoods and projections (IR)

Eight likelihood families are available in the DSL and compile to
`ir::observation::Likelihood`:

| DSL keyword     | IR variant                | Kwargs                | Support     |
| --------------- | ------------------------- | --------------------- | ----------- |
| `poisson`       | `Poisson`                 | `rate`                | `{0,1,…}`   |
| `neg_binomial`  | `NegBinomial`             | `mean, r`             | `{0,1,…}`   |
| `normal`        | `Normal`                  | `mean, sd`            | `{0,1,…}` † |
| `binomial`      | `Binomial`                | `n, p`                | `{0..n}`    |
| `beta_binomial` | `BetaBinomial`            | `n, alpha, beta`      | `{0..n}`    |
| `beta`          | `Beta`                    | `mean, concentration` | `(0,1)`     |
| `bernoulli`     | `Bernoulli`               | `p`                   | `{0,1}`     |
| `zero_inflated` | `ZeroInflatedNegBinomial` | `base, pi`            | `{0,1,…}`   |

† `normal` is pomp/He-et-al.'s _discretized_-Normal count likelihood — it scores
`y.round().max(0.0)` via a difference of normal CDFs
(`obs_loglik.rs::discretized_normal_logpmf_tol`), not a continuous density. A
clearly-fractional observation logs a one-time warning (`obs_model.rs:93`)
rather than being silently coerced.

`beta` is the only continuous family. Two consequences follow from its support
being the **open** interval:

- An observed value of exactly `0` or `1` scores `NEG_INFINITY`
  (`obs_loglik.rs::beta_logpdf`), as does a non-positive shape (`mean·φ ≤ 0` or
  `(1−mean)·φ ≤ 0`). A stream whose model-predicted mean can reach 0 — a
  prevalence proportion after burnout, say — will emit exact zeros under
  `simulate` that then cannot be scored back. Keep the mean off the boundary (a
  baseline floor, or a link that cannot saturate) when the same model is used
  for both.
- `beta` is the family to reach for when the observation _is_ a proportion with
  no reported denominator (an assay fraction, a coverage, a positivity).
  `beta_binomial` is for `k`-of-`n` counts; using it for a bare proportion
  requires inventing a pseudo-`n`, which changes the variance model.

`ocaml/golden/surveillance_likelihoods.camdl` carries one stream per family for
`normal`, `poisson`, `beta_binomial`, and `beta`, with the last two placed to
make that contrast concrete.

Four projection types map trajectory state to a scalar per observation time
(`ir::observation::Projection`):

- `CumulativeFlow(transition_name)` — cumulative flow through a transition since
  the last observation (incidence).
- `CurrentPop(compartment)` — instantaneous compartment count (prevalence).
- `CurrentPopSum(Vec<compartment>)` — sum across compartments (prevalence of a
  pooled group).
- `DerivedExpr(Expr)` — arbitrary expression over `Pop(…)` + `Time`
  - params, for snapshot-style observables not directly expressible as one
    compartment.

Three schedule types (`ir::observation::ObservationSchedule`):

- `AtTimes(Vec<f64>)` — explicit times.
- `Regular { start, step, end }` — evenly spaced, from the DSL
  `every = N 'unit`.
- `FromData(path)` — times read from the observed-data TSV.

## 3. Runtime: where the code lives

| Concern                          | File                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| Observation-model trait          | `rust/crates/sim/src/inference/traits.rs` (`ObservationModel<S>`)                                 |
| Multi-stream joint likelihood    | `rust/crates/sim/src/inference/multi_stream_obs.rs` (`MultiStreamObsModel`, `StreamSpec`)         |
| Per-family likelihood/sampling   | `rust/crates/sim/src/inference/obs_model.rs`                                                      |
| Log-pmf/pdf numerics             | `rust/crates/sim/src/inference/obs_loglik.rs`                                                     |
| Projection evaluation            | `rust/crates/sim/src/inference/multi_stream_obs.rs` (`StreamProjection::eval`)                    |
| Data-TSV loader (inference)      | `rust/crates/cli/src/pfilter.rs` (`load_data_tsv`, `load_data_tsv_column`)                        |
| Synthetic-obs emitter (simulate) | `rust/crates/cli/src/main.rs` (the `--obs` / `--obs-dir` / `--obs-only` branch in `run_simulate`) |

## 4. Flow through the system

```
.camdl file
  observations { weekly_cases : { projected = incidence(recovery)
                                  every = 7 'days
                                  likelihood = neg_binomial(mean = rho * projected, r = k) } }
    │
    │ camdlc compiles to IR
    ▼
ir::Model.observations: Vec<Observation>
  ├─ .name: "weekly_cases"
  ├─ .projection: CumulativeFlow("recovery")
  ├─ .schedule: Regular { start, step: 7.0, end }
  └─ .likelihood: NegBinomial { mean: …, r: … }
    │
    │ CompiledModel::new(ir) — per-stream, resolves Expr kwargs → ResolvedExpr
    ▼
MultiStreamObsModel
  ├─ StreamSpec {
  │    projection:  StreamProjection::FlowSum(vec![transition_idx])
  │    ir_model:    ir::Observation clone
  │    ll_accessor: closure over (state, obs_idx, params) → f64
  │  }
  └─ .log_likelihood(state, obs_idx, params) → joint log p(y | state, θ)
```

For **simulate --obs**: trajectory runs; for each observation time the emitter
calls the per-stream `sample(...)` method seeded from
`process_seed ^ SEED_MIX_OBS` (defined in `util.rs`) to decorrelate obs RNG from
process RNG, and writes the resulting value to TSV.

For **pfilter / fit** (including IF2 stages): observed data is loaded from
`--data FILE.tsv` via `load_data_tsv_column`; the filter calls
`MultiStreamObsModel::log_likelihood(state, obs_idx, params)` at each
observation time to score particles.

## 5. RNG separation

The observation RNG is always independent of the process RNG. The decorrelation
mask lives at `rust/crates/cli/src/util.rs:SEED_MIX_OBS`:

```rust
pub const SEED_MIX_OBS: u64 = 0xa5a5a5a5a5a5;
let obs_rng = StatefulRng::new(process_seed ^ SEED_MIX_OBS);
```

Used in two places:

- The `--obs` / `--obs-only` simulate path in `main.rs::run_simulate`.
- The `[synthetic]` data generator in `fit/runner.rs` (which must produce the
  same observation bytes as `camdl simulate --obs` at the same nominal seed, so
  the fit sees the same synthetic data you could have generated standalone).

Both paths go through the same constant deliberately — changing it anywhere
requires changing it everywhere, or synthetic fits stop being reproducible
against standalone simulate output.

## 6. Output files and CAS layout

Under `--cas`, the canonical layout from the 2026-04-19 output-tree unification
(`docs/dev/proposals/2026-04-19-unified-output-tree.md`):

```text
<root>/sims/<sim_hash[:8]>/<scenario-slug>-<scen_hash[:8]>/seed_<N>/
  traj.tsv                      # trajectory (always)
  run.json                      # simulate provenance
  obs/<obs_hash[:8]>-<obs_seed>/
    <stream_name>.tsv           # synthetic observations (one per stream)
    obs.json                    # obs-generation provenance
```

The `obs_hash` captures the observation model + obs params + the streams' output
schedule — independent of the trajectory hash, so iterating on the observation
model doesn't invalidate the trajectory cache.
`camdl cat <hash> --stream weekly_cases` surfaces either synthetic or data TSVs
uniformly.

## 7. Data-TSV format

Both synthetic and real observations use the same wide TSV:

```tsv
time	weekly_cases	detection
7.0	23	1
14.0	67	1
21.0	NaN	0
28.0	112	1
```

- `NaN` = missing observation (skipped in likelihood accumulation during
  inference; never emitted by simulate).
- Time column is always first; remaining columns are streams in declaration
  order.
- `--obs-dir` variant writes one file per stream with columns `time, value` —
  useful when stream observation schedules differ and ragged NaN padding would
  be awkward.

## 8. Related specs + code

- DSL surface: `docs/camdl-language-spec.md` §12 (Observations)
- CAS layout: `docs/camdl-run-spec.md` §5
- Inference use: `docs/camdl-inference-spec.md` (fit.toml `[data]` block,
  likelihood-scoring semantics)
- Prequential scoring (elpd / CRPS / PIT from observation likelihoods):
  `docs/dev/proposals/2026-04-20-prequential-evaluation.md`
