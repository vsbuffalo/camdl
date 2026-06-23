# camdl Run System Specification

**Version:** 0.4-draft\
**Date:** 2026-04-15

> **Scope:** This spec describes camdl's complete run system: forward simulation
> (single runs and batches), inference pipelines (`fit.toml`), parameter sweeps,
> predictive workflows, and the provenance/caching infrastructure that underlies
> all of them. It supersedes the forward-simulation-only experiment system
> (experiment.toml v0.6).

---

## 1. Design Principles

### 1.1 Background: How camdl Partitions Model Inputs

A camdl model defines a stochastic simulator whose inputs are partitioned into
three categories (Buffalo 2026):

- **Model parameters (M):** The tuneable knobs — quantities that _could_ be
  varied during calibration or sensitivity analysis. Transmission rate β,
  recovery rate γ, reporting probability ρ, etc. Declared in the `.camdl` file's
  `parameters { }` block.

- **Configuration (C):** Structural and runtime choices that are never subject
  to calibration. Population structure, time step, which outputs to record,
  which interventions are enabled. Defined by the `.camdl` model structure plus
  scenario patches.

- **Seed (s ∈ S):** The base random seed for stochastic simulation. Always a CLI
  argument, never baked into a config file.

A simulation is then the mapping Sim(m, c, s) → y, producing trajectories and
observations. Every workflow in this spec — forward simulation, sweeps,
inference, predictive checks — is an operation on these three input layers.

**Scenarios** are deterministic patches σ that modify parameters and/or
configuration from their baseline values: σ(m, c) → (m', c'). They are defined
in the `.camdl` file's `scenarios { }` block and selected at runtime. The
baseline is the identity patch — the model as written, no modifications.

**Inference** operates on a _view_ of the parameter space. When fitting a model,
some parameters are estimated (free to vary) while others are held fixed. This
partition — `[estimate]` vs `[fixed]` in fit.toml — defines which parameters the
inference algorithm explores and which it treats as known constants.

### 1.2 File Roles and Separation of Concerns

**One file per concern, no overlap.**

```
model.camdl      → what the model IS (structure, scenarios)
params.toml      → a point m ∈ M (concrete parameter values)
fit.toml         → how inference RUNS (what to estimate, algorithm, data)
batch file       → how a batch RUNS (sweep/scenarios/seeds, via `camdl batch run`)
```

Each file owns its domain exclusively. The fit file cannot define model
structure — that's what `.camdl` is for. The params file cannot set the backend
— that's a CLI or batch concern. The `.camdl` file cannot set concrete parameter
values (outside of named scenario presets) — that's what external files are for.

**The model file is self-contained for single runs.** A `.camdl` file with a
params.toml and a seed is everything needed for one simulation — no batch file
or fit config required.

```
┌──────────────────────┬───────────────────────┬───────────────────────────┐
│       Concern        │     Lives in          │     Can override?         │
├──────────────────────┼───────────────────────┼───────────────────────────┤
│ Model structure      │ .camdl only           │ never                     │
│ Parameter names/types│ .camdl only           │ never                     │
│ Parameter values (M) │ params.toml / [fixed] │ --param, sweep, scenario  │
│ Scenarios (σ)        │ .camdl scenarios { }  │ batch adds on top         │
│ Interventions        │ .camdl only           │ scenarios enable/disable  │
│ Backend choice       │ CLI / batch           │ CLI --backend             │
│ Seeds                │ CLI / batch           │ CLI --seed/--seeds        │
│ Sweep / Design       │ CLI / batch           │ never                     │
│ Estimate vs fixed    │ fit.toml only         │ --sweep overrides [fixed] │
│ Inference algorithm  │ fit.toml only         │ CLI --stage               │
│ Priors               │ fit.toml only         │ never                     │
└──────────────────────┴───────────────────────┴───────────────────────────┘
```

### 1.3 Precedence Chains

**For M (parameters) in simulation:**

```
params.toml                          (base values)
  ↓ overridden by
sweep point overrides                (automated M-layer variation)
  ↓ overridden by
scenario params                      (counterfactual modifications to M)
  ↓ overridden by
--param CLI flags                    (convenience, non-persistent)
```

**For M in inference:**

```
[fixed] from_file                    (bulk fixed values)
  ↓ overridden by
[fixed] inline values                (specific overrides)
  ↓ overridden by
--sweep point overrides              (grid of fits)
```

Parameters in `[estimate]` are never overridden — they are the free variables
the algorithm explores.

### 1.4 Core Design Rules

**CLI and file are the same type.** Every batch TOML file deserializes into the
same Rust struct that CLI argument parsing produces. `camdl batch run
file.toml`
and a long command line are interchangeable representations of the same job.
This is enforced by deriving both `clap::Parser` and `serde::Deserialize` from
shared types.

**No silent defaults for parameters.** Following camdl's core philosophy,
parameter values are never silently inherited. In `fit.toml`, every model
parameter must appear in exactly one of `[estimate]` or `[fixed]`. Missing a
parameter is a hard error. In simulation, `--params` or `--draws` must cover all
of M. There are no fallback defaults — the user must make every parameter choice
explicit. (See the camdl language spec §4.2 and the scenarios chapter for the
rationale behind this design.)

**Sweeps are orthogonal to everything.** A sweep is "run this thing at multiple
parameter values." It works identically on simulation and inference. The
`--sweep` flag / `[sweep]` section varies parameters across a grid. Sweeps
compose with scenarios (σ layer) and seeds (S layer) via Cartesian product:
total runs = |param_points| × |scenarios| × |seeds|.

**Provenance is structural, not aspirational.** Every run writes metadata
recording exact inputs, hashes, versions, and lineage. The runner validates
consistency before overwriting — stale results from a changed config are never
silently replaced. The types that describe jobs also compute their output paths
and content hashes.

**Draws and sweeps are different operations.** A sweep is a deterministic grid
the user designed. Draws are samples from a distribution (posterior, prior, or
uniform from bounds). They have different provenance, different downstream
semantics, and different output structure. They are separate variants of a sum
type, never conflated.

**Reproducibility is structural.** Every simulation output is content-addressed
by the inputs that produced it. Same inputs → same hash → same directory.
Different inputs → different hash → separate directory. M and σ are distinct
layers — the CLI makes this visible: `--param` operates on M, `--scenario` and
`--enable` operate on σ. They never share flags.

---

## 2. Project Directory Structure

```
project/
├── models/
│   ├── sir.camdl
│   └── seir_nigeria.camdl
├── data/
│   ├── cases.tsv
│   └── lga_pop.tsv
├── params/
│   └── baseline.toml
├── fits/                            # fit.toml configs
│   ├── 01_all_free.toml
│   ├── 02_fix_beta.toml
│   └── 03_rho_sweep.toml
├── batches/                         # simulation batch configs
│   ├── scenario_comparison.toml
│   └── ppc.toml
└── results/                         # ALL output under one tree
    ├── fits/                        # inference results (named dirs)
    │   ├── 01_all_free/
    │   │   ├── mle/
    │   │   │   ├── provenance.json
    │   │   │   ├── traces.tsv
    │   │   │   └── mle_params.toml
    │   │   └── posterior/
    │   │       ├── provenance.json
    │   │       ├── draws.tsv
    │   │       └── diagnostics.json
    │   └── 02_fix_beta/
    │       └── ...
    └── sims/                       # batch sim results (hash-addressed)
        ├── model.ir.json
        └── {sim_hash_8}/
            └── {scenario_slug}-{scen_hash_8}/
                └── seed_{n}/
                    ├── traj.tsv
                    └── run.json
```

### 2.1 Why Two Caching Strategies

**Fit results use named directories** because fits are iterative, human-driven
experiments. You want to see `01_all_free/mle/` in your file browser, not a
hash. You reason about fits by name ("the one where I fixed beta"). Named
directories support this. Cache invalidation is handled by hash-based staleness
detection over the fit's recorded `config_hash` (see §9.4), not by directory
naming.

**Simulation results use content-addressed (hash-based) directories** because
batch simulations are reproducible and high-volume. You might run 18,000
simulations across a sweep × scenario × seed grid. Content addressing gives you
free deduplication: same inputs → same hash → same directory → skip. You never
browse these directories manually — you access results through the manifest or
summary tools.

### 2.2 Fit Result Layout

Fit directories are content-addressable: the basename stem is followed by an
8-char hash of the fit.toml + model IR + data files. Running two fits with the
same filename but different content produces two distinct directories — the hash
is the key, the stem is a human-readable label.

```
results/fits/{fit_toml_stem}-{fit_hash[:8]}/
  run.json                 # top-level Run::Fit record
  real/                    # (or synthetic/ for [synthetic] fits)
    fit_{seed}/
      {stage_name}/
        run.json           # per-stage Run::FitStage record
        mle_params.toml    # (optimization stages) best parameters
        traces.tsv         # (optimization stages) per-iteration traces
        draws.tsv          # (sampling stages) posterior draws, complete M
        diagnostics.json   # (sampling stages) ESS, R-hat, acceptance
        logliks.tsv        # (pfilter stages) per-replicate logliks
        chain_{n}/         # per-chain output subdirectory
  fit.meta.json            # fit-level provenance sidecar (§2.2.1)
  predictive/<stream>.tsv  # `fit predict` predictive bands (§2.2.2)
  observed/<stream>.tsv    # `fit predict` observed series (§2.2.2)
```

#### 2.2.1 `fit.meta.json` — fit-level sidecar

The fit segment carries a `fit.meta.json` sidecar holding fit-wide provenance
that has no `RunRecord` of its own: the user `--label`, the resolved per-parameter
prior sources, the `estimated` / `fixed` parameter roles, data hashes, and a
machine-readable **observation/dimension schema** under the `schema` key. The
schema lets a consumer facet any stream and label panels with no DSL parsing:

```json
{
  "schema": {
    "dimensions": { "patch": { "levels": ["Bo", "Bombali"] } },
    "streams": [
      {
        "name": "onset",
        "index_dims": ["patch"],
        "value_column": "onset",
        "value_kind": "count",
        "likelihood": "neg_binomial"
      }
    ]
  }
}
```

`dimensions` maps each indexing dimension to its ordered levels (union over all
streams). `streams` carries one descriptor per **logical** stream (grouped by the
`from <label>` data-source key, so a stratified `onset[patch]` is one entry with
`index_dims = ["patch"]`, never one per expanded leaf): `name`, `index_dims`,
`value_column` (the scored `~` LHS), `value_kind` (the DSL role — `count` /
`real` / `probability`; omitted when the model declares no `columns {}` block),
and `likelihood` (the family tag). It is a pure fold over the same observation
leaves the particle filter binds, so it cannot disagree with what was fit.

#### 2.2.2 `fit predict` predictive artifact

`camdl fit predict --fit <run>` writes the predicted-vs-observed artifact at a
flat per-run path — two tidy, plot-ready files per logical stream:

```
results/fits/<stem>-<hash[:8]>/predictive/<stream>.tsv
  time | <dims...> | horizon | treatment | rhat_max | ess_min | n_draws | q05 | q25 | q50 | q75 | q95

results/fits/<stem>-<hash[:8]>/observed/<stream>.tsv
  time | <dims...> | value
```

- `predictive/<stream>.tsv` — the model's distribution over the observable,
  summarized as the `q05…q95` quantiles of sampled `y_rep` per `(time, stratum)`.
  The `horizon` column (`free_forward` / `one_step`) and `treatment` column
  (`posterior`) make the two predictive axes explicit; `rhat_max` / `ess_min`
  carry the producing stage's convergence numbers (empty cells when the stage
  reported none), and `n_draws` the cloud size behind each band. Both horizons,
  when emitted, stack under one header — distinguished by the `horizon` column,
  so a new predictive cell is more rows, never new consumer code. A stratified
  stream is one file with its index dims as key columns.
- `observed/<stream>.tsv` — the recorded value per `(time, stratum)`, a derived
  series in the same tidy keys as `predictive`. A hole (a scheduled but
  unobserved cell) renders as an empty `value`, distinct from an observed zero.

A consumer reads both, joins on `(time, <dims>)`, and plots `observed` over the
`predictive` ribbon, one facet per stratum — using the `index_dims` from the
`fit.meta.json` schema, with no run-store, DSL, or likelihood knowledge.

### 2.3 Sweep Subdirectories

When a fit is swept over a fixed parameter, each sweep point gets a subdirectory
under the fit directory:

```
results/fits/03_rho_sweep-{fit_hash[:8]}/
  real/
    fit_{seed}/
      rho_0.500/
        mle/...
        posterior/...
      rho_0.100/
        mle/...
      rho_0.020/
        mle/...
```

For multi-parameter sweeps, directory names concatenate with double underscores:
`rho_0.500__k_5.000/`. Within each sweep point directory, the stage layout is
identical to a non-swept fit.

### 2.3.1 Single-writer-per-fit-dir contract

A `fit_dir` (`results/fits/<stem>-<hash[:8]>/`) is implicitly single- writer.
Concurrent camdl processes writing into the same fit_dir are not supported and
produce undefined behavior — in practice, one writer's stage outputs may
silently clobber another's.

Batch workflows that parallelize across cells or fits **must** partition by
`fit_hash`: either run distinct fit.tomls (different hashes → different
directories) in parallel, or sequence runs that target the same fit_dir.
`camdl fit run` guards the obvious collision case — when a stage directory
already exists, `--force` is required to re-run — but this check is advisory,
not a lock.

Real concurrent-writer support (lockfile discipline, partial-write recovery) is
tracked in the 2026-04-19 output-tree-hardening proposal §defer/D1; until that
lands, treat the single-writer contract as a hard requirement.

### 2.4 Simulation Result Layout

```
results/sims/
  model.ir.json              # compiled model (self-contained)
  {sim_hash_8}/              # model + base params + backend + dt
    {scenario_slug}-{scen_hash_8}/
      seed_{n}/
        traj.tsv
        run.json
```

**Scenario slug:** scenario name lowercased, non-`[a-z0-9_]` replaced with `_`.

**Seed directory:** `seed_{N}` with verbatim u64, no zero-padding.

Example with two scenarios and a sweep point:

```
results/sims/
  3a7f2c1d/baseline-00000000/seed_1/
  3a7f2c1d/with_sia-f9e2b047/seed_1/
  3a7f2c1d/with_sia_vacc_eff_0.5-a3c1e890/seed_1/
```

A scenario with no overrides, enables, or disables always produces
`scen_hash = sha256("")` → `00000000` prefix, visually identifying it as the
unmodified baseline.

---

## 3. Core Types

### 3.1 SimulateJob — the universal simulation type

```rust
/// Everything needed to run one or more simulations.
/// Deserializes from batch TOML or constructs from CLI args.
/// This is THE type — CLI and file both produce this.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulateJob {
    pub model: PathBuf,
    /// Base parameter values. Optional because draws (§3.4) can provide
    /// complete parameter vectors. When None and source is Point or Sweep,
    /// the runner validates that all parameters are covered by --param
    /// overrides or sweep values — missing parameters are a hard error,
    /// not a silent default.
    pub params: Option<PathBuf>,
    #[serde(default = "default_backend")]
    pub backend: Backend,
    #[serde(default = "default_dt")]
    pub dt: f64,
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,

    /// Where parameter vectors come from — the central dispatch
    #[serde(flatten)]
    pub source: ParamSource,

    /// σ layer — which scenarios to run (empty = baseline only)
    #[serde(default)]
    pub scenarios: Vec<ScenarioRef>,

    /// S layer
    #[serde(default)]
    pub seeds: Seeds,

    /// Synthetic observation output mode.
    #[serde(default)]
    pub obs: ObsOutput,

    /// Parallelism (Rayon thread count)
    #[serde(default = "default_parallel")]
    pub parallel: usize,

    /// GeoJSON file to copy into output for web visualization
    pub geo: Option<PathBuf>,
}
```

### 3.1.1 ObsOutput — synthetic observation output

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ObsOutput {
    /// No synthetic observations (default).
    #[default]
    None,
    /// Write all streams to a single wide-format TSV.
    /// Errors if streams have different schedules.
    File(PathBuf),
    /// Write one TSV per observation stream in the given directory.
    Dir(PathBuf),
    /// Like File, but suppress trajectory output entirely.
    OnlyFile(PathBuf),
    /// Like Dir, but suppress trajectory output entirely.
    OnlyDir(PathBuf),
}
```

CLI mapping:

```
--obs cases.tsv          → ObsOutput::File("cases.tsv")
--obs-dir obs/           → ObsOutput::Dir("obs/")
--obs-only cases.tsv     → ObsOutput::OnlyFile("cases.tsv")
--obs-only-dir obs/      → ObsOutput::OnlyDir("obs/")
```

### 3.2 ParamSource — where parameter vectors come from

This is the central sum type. It determines the shape of a batch: are you
running one simulation (Point), a designed grid (Sweep), or sampling from a
distribution (Draws)? Exactly one variant is active per job.

```rust
/// Untagged is safe here because the three variants are structurally
/// distinct in TOML: Sweep requires a [sweep] table, Draws requires a
/// [draws] table, and Point is the fallback when neither exists.
/// No valid TOML matches multiple variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamSource {
    /// Deterministic grid: Cartesian product of swept values.
    /// Each point overrides the corresponding key in params.toml.
    Sweep {
        sweep: IndexMap<String, SweepSpec>,
    },

    /// Samples from a file or distribution.
    /// Each row/sample is a complete parameter vector.
    Draws {
        draws: DrawsSpec,
    },

    /// Single point: params.toml + optional CLI overrides.
    /// Default when neither sweep nor draws is specified.
    Point {
        #[serde(rename = "param", default)]
        overrides: Vec<ParamOverride>,
    },
}

impl ParamSource {
    /// Total parameter points in the batch.
    /// Total runs = n_points × |scenarios| × |seeds|.
    pub fn n_points(&self) -> usize {
        match self {
            ParamSource::Point { .. } => 1,
            ParamSource::Sweep { sweep } => {
                sweep.values().map(|s| s.len()).product()
            }
            ParamSource::Draws { draws } => draws.n_points(),
        }
    }
}
```

### 3.3 SweepSpec — parameter grid specification

Multiple swept parameters produce a Cartesian product. For two parameters with 9
and 5 values respectively, the grid has 45 points — each point is an (R0, gamma)
pair that overrides the base parameter values.

```rust
/// How to generate values for one swept parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SweepSpec {
    /// Explicit values: vacc_eff = [0.1, 0.3, 0.5, 0.7, 0.9]
    List(Vec<f64>),

    /// Generator (tagged by inner key)
    Generator(SweepGenerator),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SweepGenerator {
    #[serde(rename = "linspace")]
    Linspace { min: f64, max: f64, n: usize },

    #[serde(rename = "logspace")]
    Logspace { min: f64, max: f64, n: usize },

    #[serde(rename = "range")]
    Range { min: f64, max: f64, step: f64 },
}

impl SweepSpec {
    pub fn expand(&self) -> Vec<f64> { todo!() }

    /// Count without allocating.
    pub fn len(&self) -> usize {
        match self {
            SweepSpec::List(v) => v.len(),
            SweepSpec::Generator(g) => g.len(),
        }
    }
}

impl SweepGenerator {
    pub fn len(&self) -> usize {
        match self {
            Linspace { n, .. } | Logspace { n, .. } => *n,
            Range { min, max, step } => ((max - min) / step).ceil() as usize + 1,
        }
    }
}

/// A single point in the sweep grid.
pub type SweepPoint = IndexMap<String, f64>;

/// Expand a multi-parameter sweep into its Cartesian product.
pub fn expand_sweep(sweep: &IndexMap<String, SweepSpec>) -> Vec<SweepPoint> {
    let expanded: Vec<(String, Vec<f64>)> = sweep
        .iter()
        .map(|(k, s)| (k.clone(), s.expand()))
        .collect();
    cartesian_product(&expanded)
}
```

| Generator   | Parameters           | Description              |
| ----------- | -------------------- | ------------------------ |
| (bare list) | —                    | Explicit values          |
| `linspace`  | `min`, `max`, `n`    | n evenly-spaced points   |
| `logspace`  | `min`, `max`, `n`    | n log-spaced points      |
| `range`     | `min`, `max`, `step` | Step from min toward max |

All generator args are keyword — no positional ambiguity.

### 3.4 DrawsSpec — parameter samples

Draws represent parameter vectors sampled from a distribution — fundamentally
different from a sweep grid. A sweep is a design you chose; draws are samples
from inference output or a prior. The distinction matters for provenance:
downstream analyses need to know whether results came from a designed grid or a
posterior sample.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source")]
pub enum DrawsSpec {
    /// Load from a TSV file (posterior draws, external samples).
    /// Each row is a complete parameter vector (all of M).
    /// Column names must match model parameter names.
    #[serde(rename = "file")]
    File {
        file: PathBuf,
        /// Stochastic replicates per draw (different seeds, same params).
        #[serde(default = "default_one")]
        replicates: usize,
    },

    /// Sample from declared priors in a fit.toml.
    /// Requires a fit config with [estimate] entries that have
    /// prior = { ... } specifications. Errors if any estimated
    /// parameter lacks a declared prior.
    #[serde(rename = "prior")]
    Prior {
        fit: PathBuf,
        n: usize,
        #[serde(default = "default_one")]
        replicates: usize,
    },

    /// Sample uniformly from parameter bounds.
    /// Uses bounds from the model's parameter declarations
    /// (the `in [lo, hi]` clause on each parameter).
    /// Named honestly: this is NOT a prior — it's space-filling
    /// exploration for model debugging.
    #[serde(rename = "uniform")]
    Uniform {
        n: usize,
        #[serde(default = "default_one")]
        replicates: usize,
    },
}
```

#### The `simulate --draws` CLI sources

On the command line, `--draws` accepts four source forms, resolved to a draws
cloud before simulation:

| `--draws <value>` | Source                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `<file.tsv>`      | A TSV of complete parameter vectors (one row per draw, columns = parameter names).                                                                       |
| `prior`           | Sample the priors declared in a `fit.toml` (requires `--fit <fit.toml>`).                                                                                |
| `uniform`         | Space-fill from the model's parameter bounds.                                                                                                            |
| `posterior`       | Resolve a completed fit's canonical post-warm-up `draws.tsv` — the terminal Bayesian stage's cloud (requires `--fit <fit results dir>`).                 |

`--draws posterior --fit <fit results dir>` reads the same draws the terminal
PGAS / PMMH / MH stage wrote (burn-in already applied), so a posterior-predictive
check feeds the fit's own cloud back through the model without re-discarding
warm-up.

`--draws <file.tsv> --fit <config-or-run>` additionally **backfills** any
parameter absent from the file's columns from the fit's `[fixed]` block, never
overwriting a column the file provides — a raw posterior trace tail carries only
the estimated columns, so this fills the fixed parameters from the fit rather
than falling back to model defaults.

### 3.5 Seeds — S layer specification

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Seeds {
    Single(u64),
    Count { n: u64 },
    Range { from: u64, to: u64 },
    List { list: Vec<u64> },
}

impl Seeds {
    pub fn expand(&self) -> Vec<u64> {
        match self {
            Seeds::Single(s) => vec![*s],
            Seeds::Count { n } => (1..=*n).collect(),
            Seeds::Range { from, to } => (*from..=*to).collect(),
            Seeds::List { list } => list.clone(),
        }
    }
}

impl Default for Seeds {
    fn default() -> Self { Seeds::Single(1) }
}
```

TOML examples:

```toml
seeds = 42 # single
seeds = { n = 1000 } # 1, 2, ..., 1000
seeds = { from = 1, to = 1000 } # range: 1..=1000
seeds = { list = [42, 137, 256] } # explicit list
```

### 3.6 ScenarioRef — σ layer

```rust
/// A scenario reference in a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ScenarioRef {
    /// Reference a scenario defined in the .camdl file
    Named(String),

    /// Inline definition (in batch TOML [[scenario]] entries)
    Inline {
        name: String,
        #[serde(default)]
        enable: Vec<String>,
        #[serde(default)]
        disable: Vec<String>,
        #[serde(default)]
        params: IndexMap<String, f64>,
    },
}
```

When no `[[scenario]]` entries are defined and no `--scenario` flag is given, a
single implicit baseline (the identity patch — no enables, no disables, no param
overrides) is used. This is not a "default scenario" — it is the absence of any
scenario patch.

### 3.7 Output Path Methods

Output paths are computed by methods on the job types — never constructed
ad-hoc. This guarantees that provenance recording, cache checking, and the
runner all agree on where results live.

```rust
impl SimulateJob {
    /// Canonical output path for a single simulation run.
    pub fn run_path(
        &self,
        ir: &CompiledModel,
        base_params: &ParamSet,
        sweep_point: Option<&SweepPoint>,
        scenario: &ScenarioRef,
        seed: u64,
    ) -> PathBuf {
        let sim_hash = self.sim_hash(ir, base_params);
        let scen_hash = scenario.scen_hash(sweep_point);
        let slug = scenario.slug();

        self.output_dir
            .join("simulate")
            .join(&sim_hash[..8])
            .join(format!("{}-{}", slug, &scen_hash[..8]))
            .join(format!("seed_{}", seed))
    }
}
```

---

## 4. Single-Run Simulation (No Batch File)

The `.camdl` file + `params.toml` is a complete, self-contained specification
for running a model. No batch file or fit config is needed for exploration.

### 4.1 Basic CLI

```bash
# Baseline (no scenario) — registers a run in the CAS, prints a banner (§4.4)
camdl simulate model.camdl --params params.toml --seed 42
# stderr: ✓ run a1d91886 · camdl cat a1d91886   (add --stdout to pipe instead)

# Named scenario from .camdl
camdl simulate model.camdl --params params.toml --scenario with_sia --seed 42

# Ad-hoc intervention toggle (no named scenario)
camdl simulate model.camdl --params params.toml --enable sia_round_1 --seed 42

# Parameter override (M layer)
camdl simulate model.camdl --params params.toml --param beta=0.5 --seed 42

# Scenario + parameter override (σ layer + M layer — both valid)
camdl simulate model.camdl --params params.toml --scenario with_sia --param beta=0.5 --seed 42
```

### 4.2 CLI Flag Rules

**`--scenario` and `--enable`/`--disable` are mutually exclusive (both σ
layer):**

```bash
# ERROR:
camdl simulate model.camdl --scenario with_sia --enable sia_round_2
# error: --scenario and --enable/--disable are mutually exclusive.
#   --scenario selects a named scenario from the model file.
#   --enable/--disable composes an ad-hoc scenario.
#   To combine both, define a composed scenario in the model file.
```

**`--param` is always valid (M layer, independent of σ layer):**

```bash
# All valid — --param operates on M, independent of scenario choice:
camdl simulate model.camdl --params p.toml --param beta=0.5 --seed 42
camdl simulate model.camdl --params p.toml --scenario with_sia --param beta=0.5 --seed 42
```

### 4.3 Parameter Flags

```bash
--params FILE              # load parameter file (can repeat for layering)
--param NAME=VALUE         # override a single parameter (can repeat)
--param-vec PREFIX=FILE    # override indexed params from keyed TSV
--table NAME=FILE          # supply external() table data
```

### 4.4 Output destinations — CAS by default

Every run **registers in the content-addressable store** (`results/`, §2.4) and
prints a one-line discoverability banner to **stderr**. This is the default —
there is no `--cas` flag to remember, and a large trajectory never floods the
terminal. (Rev-3 change: CAS was opt-in via `--cas` through v0.2; that behavior
is now the default and the flag is removed.)

```bash
# Default: register in CAS, print a banner. Re-run with identical inputs = instant cache hit.
camdl simulate model.camdl --params p.toml --seed 42
# stderr: ✓ run a1d91886 · 12K traj · camdl cat a1d91886
# (re-run) stderr: ✓ run a1d91886 (cache hit) · camdl cat a1d91886

# Opt out: stream the trajectory to stdout, no CAS write, no banner
camdl simulate model.camdl --params p.toml --seed 42 --stdout > out.tsv
camdl simulate model.camdl --params p.toml --seed 42 --stdout | head

# Write a named file AND register in CAS (the file is a convenience copy)
camdl simulate model.camdl --params p.toml --seed 42 -o out.tsv

# Browse what you've run
camdl list
camdl show <short-hash>
camdl cat <short-hash>
```

**The three destinations** (a run always produces exactly one _primary_
artifact; these say where it goes):

| flag                        | trajectory / large artifact | tiny scalar (pfilter, eval)  | CAS-registered? |
| --------------------------- | --------------------------- | ---------------------------- | --------------- |
| _(default)_                 | → CAS only                  | echoed to stdout **and** CAS | yes             |
| `--stdout`                  | → stdout, no CAS            | → stdout, no CAS             | no              |
| `-o FILE` / `--output FILE` | → FILE **and** CAS          | → FILE and CAS               | yes             |

**Small-result echo.** Commands whose primary result is at/under a small size
threshold (a `pfilter` loglik, a short `eval` table) _also_ echo to stdout even
while registering in CAS, so the tight "vary θ, re-check" loop stays a
one-liner. Larger results (a trajectory, `pfilter --replicates 50`) go CAS-only
with the banner. `--stdout` forces full output to stdout regardless.

**The banner.** `✓ run <hash8> · <summary> · camdl cat <hash8>` on a fresh
write; `✓ run <hash8> (cache hit) · …` on a hit. Always stderr, so
`--stdout | …` pipelines stay clean. (Supersedes the pre-rev-3 `cached:` /
`cache hit:` lines, which mislabeled a first write as `cached:`; see gh#136.)

**Cache hit = identical inputs.** A re-run with the same `sim_hash` +
`scen_hash` + seed is served from the CAS without re-simulating. Cache
correctness depends on `model_hash` being structural (§9.2) — fixed in
gh#135/f3bc389, so distinct models no longer collide.

**Ensembles register as one multi-run.** `--seeds` / `--replicates` / `--draws`
register a single multi-seed run whose `camdl cat` output carries a `seed` (or
`replicate`) column — no delimiter-less concatenated trajectories, and no
"`--cas` supports single runs only" wall.

**Layout.**
`results/sims/{sim_hash[:8]}/{scenario_slug}-{scen_hash[:8]}/seed_{n}/` (§2.4).
Override the root with `--output-dir DIR` (default `./results`).

**Hash composition.** `sim_hash` keys on model IR (via `model_hash`, §9.2) +
base params + backend + dt + **runtime version** (VERSION_SHORT, includes git
hash). `scen_hash` keys on enable/disable/param overrides + runtime version. A
code change that alters simulation semantics invalidates the cache under
identical inputs — no silent stale results.

### 4.5 `camdl list` / `camdl show` / `camdl cat` — the primary access path

Because output goes to the CAS by default (§4.4), `list` / `show` / `cat` are
**the** way you reach results — not an optional convenience. `list` defaults to
`./results` with no argument; the banner from each run hands you the exact
`camdl cat <hash>` to run next.

```bash
# Tabular overview — most recent first
camdl list
camdl list --since 1h          # last hour
camdl list --scenario baseline --all
camdl list --format json       # scripts / external tooling

# Full metadata for one run (resolves short-hash prefix git-style)
camdl show abc1234
camdl show ./results/sims/abc12345/baseline-def45/seed_42

# Emit trajectory or a named observation stream
camdl cat abc1234
camdl cat abc1234 --obs cases
```

`camdl list` walks `./results/sims/` (or a given path) and shows:

```
CREATED     MODEL              SCENARIO  SEED  PARAMS  SIZE  PATH
5m ago      sir.camdl          baseline  42    —       12K   ./results/sims/abc12345/baseline-def45/seed_42
yesterday   seir.camdl         vacc       7    —       48K   ./results/sims/77cc8a21/vacc-9f88/seed_7
```

Relative paths are copy-paste ready — feed them straight into `camdl show` /
`camdl cat`. Short-hash prefix resolution is git-style: unique prefix succeeds,
ambiguous prefix errors with a list of matches and a suggestion to refine
(`prefix/scenario[/seed]`).

### 4.6 Synthetic Observations

```bash
# Generate synthetic observations from the observations block
camdl simulate model.camdl --params p.toml --seed 42 --obs cases.tsv

# One file per observation stream (multi-stream / mixed-schedule models)
camdl simulate model.camdl --params p.toml --seed 42 --obs-dir obs/

# Multiple independent replicates
camdl simulate model.camdl --params p.toml --seed 42 --replicates 100 --obs cases.tsv

# Suppress trajectory, only emit observations (SBC workflows)
camdl simulate model.camdl --seed 42 --replicates 1000 --obs-only cases.tsv
```

The observation RNG is independent of the process RNG — adding `--obs` does not
change the trajectory.

---

## 5. Batch Simulation

### 5.1 CLI Invocations

```bash
# ── Multiple scenarios × seeds ───────────────────────────
camdl simulate model.camdl --params p.toml \
    --scenario baseline,with_sia --seeds 1:1000

# ── 1D sweep ─────────────────────────────────────────────
camdl simulate model.camdl --params p.toml \
    --sweep "R0=1,1.5,2,2.5,3,3.5,4,4.5,5" --seeds 1:100

# ── 2D sweep (Cartesian product: 5 × 3 = 15 points) ─────
camdl simulate model.camdl --params p.toml \
    --sweep "R0=1,2,3,4,5" --sweep "gamma=0.05,0.1,0.5" \
    --seeds 1:100

# ── Posterior predictive ─────────────────────────────────
camdl simulate model.camdl \
    --draws results/fits/02_fix_beta/posterior/draws.tsv \
    --replicates 10 --obs ppc.tsv

# ── Prior predictive (requires declared priors) ──────────
camdl simulate model.camdl \
    --draws prior --fit fits/02_fix_beta.toml --n 500 \
    --replicates 5 --obs prior_pred.tsv

# ── Uniform space-filling (no Bayesian pretension) ───────
camdl simulate model.camdl \
    --draws uniform --n 500 --replicates 5 --obs uniform_pred.tsv

# ── Scenario prediction under posterior uncertainty ──────
camdl simulate model.camdl \
    --draws results/fits/02_fix_beta/posterior/draws.tsv \
    --scenario baseline,with_sia --replicates 10 --obs-dir obs/

# ── From a batch file ────────────────────────────────────
camdl batch run batches/ppc.toml
```

**CLI `--sweep` accepts comma-separated lists only.** Generators (`linspace`,
`logspace`, `range`) are available only in batch TOML `[sweep]` sections. This
keeps the CLI syntax obvious and discoverable — if you need structured
generators, write a batch file.

### 5.2 CLI ↔ Type Mapping

```rust
#[derive(Parser)]
pub struct SimulateCli {
    pub model: Option<PathBuf>,

    #[arg(long)]
    pub params: Option<PathBuf>,

    #[arg(long)]
    pub backend: Option<Backend>,

    #[arg(long)]
    pub dt: Option<f64>,

    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    // ── ParamSource::Point ──
    #[arg(long = "param", value_parser = parse_kv)]
    pub param_overrides: Vec<ParamOverride>,

    // ── ParamSource::Sweep ──
    #[arg(long = "sweep", value_parser = parse_sweep_arg)]
    pub sweeps: Vec<(String, SweepSpec)>,

    // ── ParamSource::Draws ──
    #[arg(long)]
    pub draws: Option<DrawsArg>,

    #[arg(long)]
    pub fit: Option<PathBuf>,       // for --draws prior

    #[arg(long, short = 'n')]
    pub n_draws: Option<usize>,     // for --draws prior/uniform

    #[arg(long)]
    pub replicates: Option<usize>,

    // ── σ layer ──
    #[arg(long, value_delimiter = ',')]
    pub scenario: Vec<String>,

    #[arg(long)]
    pub enable: Vec<String>,

    #[arg(long)]
    pub disable: Vec<String>,

    // ── S layer ──
    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long, value_parser = parse_seeds)]
    pub seeds: Option<Seeds>,

    // ── Observation generation ──
    #[arg(long)]
    pub obs: Option<PathBuf>,

    #[arg(long)]
    pub obs_dir: Option<PathBuf>,

    #[arg(long)]
    pub obs_only: Option<PathBuf>,

    #[arg(long)]
    pub obs_only_dir: Option<PathBuf>,

    #[arg(long)]
    pub parallel: Option<usize>,

    // ── Batch file (alternative entry point) ──
    #[arg(long)]
    pub batch: Option<PathBuf>,
}

/// --draws argument on CLI
#[derive(Debug, Clone)]
pub enum DrawsArg {
    File(PathBuf),
    Prior,
    Uniform,
}

impl SimulateCli {
    /// Resolve CLI args into the canonical SimulateJob.
    /// This is the single convergence point: from here on,
    /// the runner doesn't know whether input came from CLI or file.
    pub fn into_job(self) -> Result<SimulateJob> {
        if let Some(batch_path) = self.batch {
            return SimulateJob::from_toml(&batch_path);
        }

        let source = self.resolve_param_source()?;
        let scenarios = self.resolve_scenarios()?;
        let seeds = self.seed.map(Seeds::Single)
            .or(self.seeds)
            .unwrap_or_default();

        Ok(SimulateJob {
            model: self.model
                .ok_or_else(|| anyhow!("model path required"))?,
            params: self.params,
            backend: self.backend.unwrap_or(Backend::ChainBinomial),
            dt: self.dt.unwrap_or(1.0),
            output_dir: self.output_dir
                .unwrap_or_else(|| PathBuf::from("results")),
            source, scenarios, seeds,
            obs: self.obs,
            parallel: self.parallel.unwrap_or(1),
            geo: None,
        })
    }
}
```

### 5.3 Total Runs Calculation

```
Total runs = |param_points| × |scenarios| × |seeds|

where |param_points| =
  Point:   1
  Sweep:   product of |sweep_i.expand()| for each swept parameter
  Draws:   n_draws × replicates

where |scenarios| =
  empty (no scenarios specified):  1 (implicit baseline)
  otherwise:                       number of [[scenario]] entries
```

### 5.4 Scenario × Sweep Interaction

Sweeps and scenarios are orthogonal. Their cross product defines the full run
grid. Each (sweep_point, scenario) combination produces one effective
configuration. Sweep point overrides apply first (M layer), then scenario params
overlay on top (σ layer).

### 5.5 Batch TOML Examples

```toml
# batches/scenario_comparison.toml
model = "models/sir.camdl"
params = "params/baseline.toml"
backend = "chain_binomial"
dt = 1.0
seeds = { n = 1000 }
output_dir = "results"
parallel = 16

[[scenario]]
name = "baseline"

[[scenario]]
name = "with_sia"
enable = ["sia"]

[[scenario]]
name = "high_coverage"
enable = ["sia"]
params = { sia_cov = 0.95 }

# Total runs: 3 scenarios × 1000 seeds = 3000
```

```toml
# batches/r0_gamma_sweep.toml — 2D Cartesian product
model = "models/sir.camdl"
params = "params/baseline.toml"
seeds = { n = 50 }

[sweep]
R0 = { linspace = { min = 1.0, max = 5.0, n = 9 } }
gamma = { logspace = { min = 0.01, max = 1.0, n = 5 } }
# 9 × 5 = 45 parameter points × 50 seeds = 2250 runs
```

```toml
# batches/ppc.toml — posterior predictive check
model = "models/sir.camdl"
obs = "results/ppc/obs.tsv"

[draws]
source = "file"
file = "results/fits/02_fix_beta/posterior/draws.tsv"
replicates = 10
```

```toml
# batches/policy_eval.toml — scenario prediction under uncertainty
model = "models/sir.camdl"
obs_dir = "results/policy_eval/obs"
seeds = { n = 10 }

[draws]
source = "file"
file = "results/fits/02_fix_beta/posterior/draws.tsv"
replicates = 1

[[scenario]]
name = "baseline"

[[scenario]]
name = "with_sia"
enable = ["sia"]
# Total: n_draws × 2 scenarios × 10 seeds
# Scenarios share seeds within each (draw, seed) pair (paired-seed coupling)
```

### 5.6 Execution Flow

The simulation runner:

1. Compiles the `.camdl` model once (or loads `.ir.json` directly)
2. Loads base params if specified
3. Generates the run grid (sweep/draws × scenarios × seeds)
4. Classifies cache hits vs new runs (check for `traj.tsv` at computed path)
5. Executes new runs with Rayon `par_iter`
6. Writes `traj.tsv` and `run.json` per run (the per-leaf run.json is the only
   batch index — there is no manifest file)
7. Copies `model.ir.json` and optional `geo/` to output root

---

## 6. FitConfig — the inference type

### 6.1 Overview

A fit.toml specifies a single inference task: which model to fit, what data to
fit it to, which parameters to estimate vs hold fixed, and what inference
algorithm to run. It defines a _view_ of the parameter space — the partition of
M into free parameters (explored by the algorithm) and fixed parameters (held
constant). The algorithm then operates in the reduced space of free parameters.
(See Buffalo 2026 for the formal treatment of parameter views, transforms, and
the downward chain from inference coordinates to simulator output.)

### 6.2 Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitConfig {
    pub model: ModelRef,

    /// Real-data source. Mutually exclusive with `synthetic` — exactly
    /// one of the two must be present.
    pub data: Option<DataSpec>,

    /// Synthetic-data source for simulation-based calibration. When set,
    /// the runner generates N datasets from `true_params` and fits each
    /// one. See `camdl-inference-spec.md` §3.7.
    pub synthetic: Option<SyntheticSpec>,

    /// IF2/PGAS seeds. Scalar/absent runs one fit per dataset; a list
    /// runs one fit per listed seed per dataset (start-sensitivity, or
    /// the full SBC × fitter matrix).
    pub fit_seeds: Option<Vec<u64>>,

    pub output_dir: Option<PathBuf>,

    /// Conditioning boundary for a covariate-informed burn-in (top-level key).
    /// Places a leading reset-only hole one cadence before the first datum, so
    /// the pre-data span is a warm-up (simulated, not scored) and the first
    /// observation is scored against one cadence. Forms: `"first_obs - 1 week"`,
    /// `"date(\"…\")"`, or a bare model-time number. See
    /// `camdl-inference-spec.md` §3.9 and `fit-toml.md`.
    pub condition_from: Option<ConditionFrom>,

    // NOTE: this struct is illustrative and elides several real top-level keys
    // of the runtime `FitConfigV2` (e.g. `ic_free`, `config`, `scenario`); see
    // `fit-toml.md` for the full surface. A complete §6.2 resync is tracked
    // separately.

    /// The free parameters: what the inference algorithm estimates.
    pub estimate: IndexMap<String, EstimateSpec>,

    /// The fixed parameters: held constant during inference.
    /// estimate ∪ fixed must cover all model parameters.
    pub fixed: FixedParams,

    /// Inference pipeline stages, executed in declaration order.
    pub stages: IndexMap<String, Stage>,

    /// Optional lineage metadata (not used by the runner).
    pub provenance: Option<FitProvenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub camdl: PathBuf,
}

/// Data file mapping. Keys match observation stream names declared in
/// the .camdl file's observations { } block. The observation model
/// (likelihood family) and projection (which flow/compartment to
/// accumulate) are defined in the .camdl file — fit.toml only provides
/// the data file paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSpec {
    /// Map from observation stream name → data file path.
    /// Keys must match names in the .camdl observations { } block.
    pub observations: IndexMap<String, PathBuf>,

    /// Time threshold for temporal holdout: observations at t > this
    /// value are withheld from training and used for out-of-sample
    /// evaluation. In model time units. Mutually exclusive with
    /// `holdout`.
    pub holdout_after: Option<f64>,

    /// Explicit holdout data files (for non-standard schemes like
    /// spatial leave-one-out). Keys match observation stream names.
    /// Mutually exclusive with `holdout_after`.
    pub holdout: Option<IndexMap<String, PathBuf>>,
}

/// Synthetic-data generation for simulation-based calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticSpec {
    /// Path to a TOML file of `name = value` lines — the ground truth
    /// used to generate data and to compute coverage / bias.
    pub true_params: PathBuf,

    /// Simulation seeds — either a range string (`"1:20"`) or an
    /// explicit list. Duplicates rejected.
    pub sim_seeds: SeedsSpec,

    /// Number of datasets. Optional — when omitted, inferred from
    /// `len(sim_seeds)`. When supplied, must equal that length.
    pub datasets: Option<usize>,

    /// Scenario applied during data generation (not during fitting).
    pub scenario: Option<String>,
}
```

### 6.3 EstimateSpec — free parameters

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimateSpec {
    pub bounds: (f64, f64),

    /// Transform for inference. If omitted, inferred from the parameter's
    /// declared type in the .camdl file: rate → log, probability → logit,
    /// positive → log, real → identity.
    pub transform: Option<Transform>,

    /// Prior distribution. Required for Bayesian methods (PGAS, PMMH).
    /// Optional for MLE (IF2 ignores priors).
    /// Also used by --draws prior for prior predictive checks.
    pub prior: Option<PriorSpec>,

    /// Initial value parameter: perturbed only at t=0 in IF2
    #[serde(default)]
    pub ivp: bool,

    /// Per-parameter random walk SD for IF2.
    /// If omitted, auto-scaled from bounds.
    pub rw_sd: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transform {
    #[serde(rename = "log")]
    Log,
    #[serde(rename = "logit")]
    Logit,
    #[serde(rename = "identity")]
    Identity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "dist")]
pub enum PriorSpec {
    #[serde(rename = "log_normal")]
    LogNormal { mu: f64, sigma: f64 },
    #[serde(rename = "normal")]
    Normal { mu: f64, sigma: f64 },
    #[serde(rename = "beta")]
    Beta { alpha: f64, beta: f64 },
    #[serde(rename = "uniform")]
    Uniform,
    #[serde(rename = "half_normal")]
    HalfNormal { sigma: f64 },
}
```

### 6.4 FixedParams

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedParams {
    /// Bulk load from a TOML file.
    pub from_file: Option<PathBuf>,

    /// Inline fixed values. Override from_file if both specify a key.
    #[serde(flatten)]
    pub values: IndexMap<String, f64>,
}

impl FixedParams {
    pub fn resolve(&self) -> Result<IndexMap<String, f64>> {
        let mut merged = match &self.from_file {
            Some(path) => load_params_toml(path)?,
            None => IndexMap::new(),
        };
        for (k, v) in &self.values {
            merged.insert(k.clone(), *v);
        }
        Ok(merged)
    }
}
```

### 6.5 Stage — inference pipeline steps

Stages are the verbs of inference: optimize (find the MLE), sample (draw from
the posterior), evaluate (assess fit quality). Each stage runs a specific
algorithm. Stages execute in declaration order; the `init = "from_mle"` +
`init_mle = "<stage>"` pair creates dependency edges between them (see §6.6).

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
pub enum Stage {
    #[serde(rename = "if2")]
    IF2 {
        chains: usize,
        particles: usize,
        iterations: usize,
        cooling: CoolingSpec,
        /// Init mode. Toml key: `init = "lhs" | "single" | "uniform" |
        /// "from_mle" | "from_prior" | "from_posterior" | "from_params" |
        /// "survey_top_k"` (Rust field name preserved as `init_method`).
        #[serde(default, rename = "init")]
        init_method: InitMethod,
        /// Companion for `init = "from_mle"`: stage name in this fit.toml,
        /// or a path to an external results dir. Toml key: `init_mle`
        /// (Rust field name preserved as `starts_from`).
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
    },

    #[serde(rename = "pgas")]
    PGAS {
        chains: usize,
        particles: usize,
        sweeps: usize,
        #[serde(default, rename = "init")]
        init_method: InitMethod,
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        #[serde(default)]
        skip_chains: Vec<usize>,
    },

    #[serde(rename = "pmmh")]
    PMMH {
        chains: usize,
        particles: usize,
        iterations: usize,
        #[serde(default, rename = "init")]
        init_method: InitMethod,
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
        #[serde(default)]
        skip_chains: Vec<usize>,
    },

    #[serde(rename = "pfilter")]
    PFilter {
        particles: usize,
        replicates: Option<usize>,
        #[serde(default, rename = "init_mle")]
        starts_from: StartsFrom,
    },
}

impl Stage {
    pub fn starts_from(&self) -> &StartsFrom {
        match self {
            Stage::IF2 { starts_from, .. }
            | Stage::PGAS { starts_from, .. }
            | Stage::PMMH { starts_from, .. }
            | Stage::PFilter { starts_from, .. } => starts_from,
        }
    }

    pub fn requires_priors(&self) -> bool {
        matches!(self, Stage::PGAS { .. } | Stage::PMMH { .. })
    }
}
```

### 6.6 init_mle — dependency edges

```rust
/// The companion path for `init = "from_mle"`. Toml-side key is
/// `init_mle`; this enum is the deserialized representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StartsFrom {
    /// Name of a previous stage in this fit.toml
    Stage(String),

    /// Path to an external results directory
    Directory(PathBuf),

    /// No prior fit to chain from; chain starts come from the
    /// stage's `init` mode (lhs / from_prior / etc.).
    #[serde(rename = "random")]
    Random,
}

impl Default for StartsFrom {
    fn default() -> Self { StartsFrom::Random }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CoolingSpec {
    /// IF2 cooling fraction, pomp's `cooling.fraction.50` convention: at
    /// the **halfway** point of the stage's iterations, the perturbation
    /// SD is `cooling_fraction × initial`; at the **end**, it is
    /// `cooling_fraction² × initial`. The parameter is stage-length-
    /// invariant: a stage with cooling=0.70 over 30 iters and one over
    /// 200 iters both end at the same SD × initial ratio (0.49).
    ///
    /// Higher values (closer to 1) = **gentler cooling**, SD stays large,
    /// chains keep exploring. Used for **scout** (code default 0.70 →
    /// final SD = 49% of initial; the cross-chain Â (chain-agreement)
    /// diagnostic at the end, combined with the loglik-eval
    /// decibans-spread gate, is what identifies basins).
    ///
    /// Lower values (closer to 0) = **aggressive cooling**, SD collapses,
    /// chains quench onto a single point. Used for **refine** (code
    /// default 0.05 → final SD = 0.25% of initial) and **validate**
    /// (same).
    ///
    /// See `docs/methods/cooling.md` for the formula, worked examples,
    /// and a per-iteration empirical table.
    Fixed(f64),
    #[serde(rename = "auto")]
    Auto,
}
```

### 6.7 FitProvenance — lineage metadata

```rust
/// Optional metadata linking this fit to a parent.
/// Not used by the runner — purely for human navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitProvenance {
    pub derived_from: Option<PathBuf>,
    pub reason: Option<String>,
}
```

### 6.8 Output Path Methods

```rust
impl FitConfig {
    pub fn fit_dir(&self, config_path: &Path) -> PathBuf {
        let name = config_path.file_stem().unwrap().to_str().unwrap();
        let output_root = self.output_dir.clone()
            .unwrap_or_else(|| PathBuf::from("results"));
        output_root.join("fits").join(name)
    }

    pub fn stage_dir(&self, config_path: &Path, stage_name: &str) -> PathBuf {
        self.fit_dir(config_path).join(stage_name)
    }

    pub fn swept_stage_dir(
        &self, config_path: &Path, sweep_point: &SweepPoint, stage_name: &str,
    ) -> PathBuf {
        self.fit_dir(config_path)
            .join(sweep_point_slug(sweep_point))
            .join(stage_name)
    }

    pub fn draws_path(&self, config_path: &Path, stage_name: &str) -> PathBuf {
        self.stage_dir(config_path, stage_name).join("draws.tsv")
    }

    pub fn mle_params_path(&self, config_path: &Path, stage_name: &str) -> PathBuf {
        self.stage_dir(config_path, stage_name).join("mle_params.toml")
    }
}

fn sweep_point_slug(point: &SweepPoint) -> String {
    point.iter()
        .map(|(k, v)| format!("{}_{:.3}", k, v))
        .collect::<Vec<_>>()
        .join("__")
}
```

### 6.9 Completeness Validation

The runner calls this at load time, before any stage executes. This enforces
camdl's "no silent defaults" rule.

```rust
impl FitConfig {
    pub fn validate(&self, ir: &CompiledModel) -> Result<()> {
        let model_params: BTreeSet<&str> = ir.parameters.keys()
            .map(|s| s.as_str()).collect();
        let estimated: BTreeSet<&str> = self.estimate.keys()
            .map(|s| s.as_str()).collect();
        let fixed_resolved = self.fixed.resolve()?;
        let fixed: BTreeSet<&str> = fixed_resolved.keys()
            .map(|s| s.as_str()).collect();

        // estimate ∩ fixed = ∅
        let overlap: Vec<_> = estimated.intersection(&fixed).collect();
        if !overlap.is_empty() {
            bail!("parameters in both [estimate] and [fixed]: {}\n  \
                   Each parameter must be in exactly one section.",
                  overlap.iter().join(", "));
        }

        // estimate ∪ fixed = model_params
        let covered: BTreeSet<_> = estimated.union(&fixed).cloned().collect();
        let missing: Vec<_> = model_params.difference(&covered).collect();
        if !missing.is_empty() {
            bail!("parameters neither estimated nor fixed: {}\n  \
                   Every model parameter must appear in [estimate] or [fixed].",
                  missing.iter().join(", "));
        }

        let extra: Vec<_> = covered.difference(&model_params).collect();
        if !extra.is_empty() {
            bail!("parameters not in model: {}", extra.iter().join(", "));
        }

        // Bayesian stages require priors
        for (name, stage) in &self.stages {
            if stage.requires_priors() {
                let missing_priors: Vec<_> = self.estimate.iter()
                    .filter(|(_, spec)| spec.prior.is_none())
                    .map(|(name, _)| name.as_str())
                    .collect();
                if !missing_priors.is_empty() {
                    bail!("stage '{}' requires priors, but missing for: {}",
                          name, missing_priors.join(", "));
                }
            }
        }

        self.validate_stage_dag()?;
        Ok(())
    }
}
```

---

## 7. The Fit CLI

### 7.1 Invocations

```bash
camdl fit run fits/01_all_free.toml
camdl fit run fits/01_all_free.toml --stage mle
# Chaining a posterior stage onto a prior fit's MLE is now expressed by
# editing the stage's init/init_mle in the toml, not via a CLI flag.
# The legacy `--starts-from <dir>` was removed in the 2026-05-25 CLI UX
# revision (it is a hidden trap that errors with the replacement spelled
# out). Use `[stages.posterior] init = "from_mle"` +
# `init_mle = "results/fits/01_all_free/mle"`.
camdl fit run fits/base.toml --sweep "rho=0.5,0.1,0.02,0.005"
camdl fit run fits/base.toml --sweep "rho=0.5,0.1" --sweep "k=5,10,20"
camdl fit run fits/01.toml --stage posterior --skip-chains 2,4
camdl fit run fits/01_all_free.toml --force
camdl fit table results/fits/
camdl fit diff fits/01_all_free.toml fits/02_fix_beta.toml
camdl fit new --from fits/01_all_free.toml fits/02_fix_beta.toml
```

### 7.2 CLI Type

```rust
#[derive(Parser)]
pub struct FitRunCli {
    pub config: PathBuf,

    #[arg(long)]
    pub stage: Option<String>,

    // `--starts-from` was removed per the 2026-05-25 CLI UX revision.
    // Chain-start sources are declared in the fit.toml via
    // `init = "from_mle"` + `init_mle = "<stage-or-dir>"`.

    #[arg(long)]
    pub seed: Option<u64>,

    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    #[arg(long)]
    pub force: bool,

    /// Resume a partially completed sampling stage (PGAS/PMMH).
    /// Continues from the last completed sweep/iteration.
    #[arg(long)]
    pub resume: bool,

    #[arg(long = "sweep", value_parser = parse_sweep_arg)]
    pub sweeps: Vec<(String, SweepSpec)>,

    #[arg(long = "skip-chains", value_delimiter = ',')]
    pub skip_chains: Vec<usize>,

    #[arg(long)]
    pub parallel: Option<usize>,
}
```

**Seeds and fits.** Unlike `SimulateJob`, `FitConfig` has no `seeds` field. A
fit uses a single base seed (default: 1, override with `--seed N`). Chain seeds
are derived deterministically as `base_seed + chain_index`. If you want multiple
independent fits at different seeds, run the command multiple times with
different `--seed` values.

### 7.3 Sweep Semantics for Fits

The `--sweep` flag on `camdl fit run` overrides a parameter in `[fixed]` at each
grid point. The swept parameter must be in `[fixed]`, not `[estimate]` —
sweeping an estimated parameter is a type error.

This means a parameter naturally _promotes_ from fixed to swept with zero config
changes:

```toml
[fixed]
rho = 0.5
```

```bash
camdl fit run fit.toml --sweep "rho=0.5,0.1,0.02"
```

Each sweep point runs the full pipeline independently.

**Gate failures do not halt the sweep.** When a single point's scout-convergence
or regression gate fails, it is recorded and the next point runs. Failed cells
are listed in `<fit_dir>/sweep_failures.tsv` with columns
`cell, sweep_point, sweep_values, stage, reason`. Downstream plotting scripts
should check this file alongside `summary.tsv` to distinguish "failed to
converge" from "didn't run." Single- point fits retain the strict
exit-on-gate-failure behavior — this relaxation applies only when `--sweep` is
active.

For a dedicated profile-likelihood workflow (fix a focal parameter, maximize
over the rest at each grid point), see `camdl profile`.

---

## 8. fit.toml Examples

### 8.1 MLE with IF2

```toml
[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
beta = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
rho = { bounds = [0.001, 1.0] }
k = { bounds = [0.1, 100.0] }

[fixed]
N0 = 1000000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 1000
iterations = 80
cooling = 0.70
```

### 8.2 MLE + Posterior Sampling

```toml
[provenance]
derived_from = "fits/01_all_free.toml"
reason = "beta mixing poor in PGAS (ESS < 50), fixing at MLE"

[model]
camdl = "models/sir.camdl"

[data.observations]
weekly_cases = "data/cases.tsv"

[estimate]
gamma = { bounds = [0.05, 1.0], prior = { dist = "log_normal", mu = -2.0, sigma = 1.0 } }
rho = { bounds = [0.001, 1.0], prior = { dist = "beta", alpha = 2.0, beta = 5.0 } }
k = { bounds = [0.1, 100.0], prior = { dist = "half_normal", sigma = 10.0 } }

[fixed]
beta = 0.34
N0 = 1000000
I0 = 10

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 4
particles = 2000
iterations = 60
cooling = 0.95
init = "from_mle"
init_mle = "results/fits/01_all_free/mle"

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 4
particles = 50
sweeps = 5000
init = "from_mle"
init_mle = "mle"

[stages.evaluate]
algorithm = "pfilter"
backend = "chain_binomial"
particles = 10000
replicates = 100
init_mle = "mle"
```

### 8.3 Large Model with from_file

```toml
[model]
camdl = "models/seir_nigeria.camdl"

[data]
holdout_after = 5474

[data.observations]
weekly_cases = "data/nigeria_afp.tsv"

[estimate]
beta = { bounds = [0.01, 5.0], prior = { dist = "log_normal", mu = 0.0, sigma = 1.0 } }
sigma = { bounds = [0.05, 1.0] }
gamma = { bounds = [0.05, 1.0] }
rho = { bounds = [0.001, 0.5], prior = { dist = "beta", alpha = 2.0, beta = 10.0 } }
k = { bounds = [0.1, 100.0] }
alpha = { bounds = [0.0, 1.0] }
phi_season = { bounds = [0.0, 365.25] }
import_rate = { bounds = [0.0001, 1.0] }

[fixed]
from_file = "params/nigeria_fixed.toml"
vacc_frac = 0.80

[stages.mle]
algorithm = "if2"
backend = "chain_binomial"
chains = 8
particles = 2000
iterations = 100
cooling = 0.70

[stages.posterior]
algorithm = "pgas"
backend = "chain_binomial"
chains = 6
particles = 100
sweeps = 10000
init = "from_mle"
init_mle = "mle"
```

---

## 9. Provenance and Cache Invalidation

### 9.1 The Hybrid Strategy

Fit results use **named directories** for human readability and **hash-based
staleness detection** for cache invalidation. Simulation results use
**content-addressed directories** where the hash IS the directory name.

### 9.2 Simulation Hash Computation

**sim_hash** — model + base params + backend + dt:

```
sim_hash = sha256(
    model_ir_json_bytes              # compiled model (deterministic)
    + canonical_sorted(base_param key=value pairs)
    + backend_string                 # "gillespie", "chain_binomial", etc.
    + dt_bytes                       # f64 little-endian
    + camdl_version_string
)
```

Full 64-character hex string; first 8 characters used in directory name.

> **`model_hash` contract.** `model_ir_json_bytes` above must hash the
> _structural_ IR (compartments, transitions, parameters, tables,
> time-functions, interventions, observations, ODE equations, initial
> conditions, version) — the canonical hasher is
> `cli/src/hashing.rs::model_hash`. Two models that differ in any structural
> element must produce different `model_hash`, hence different `sim_hash`, hence
> distinct cache directories. This is load-bearing for cache correctness: if
> `model_hash` collapsed for distinct models, `--cas` would serve one model's
> trajectory for another. **History (gh#135, fixed f3bc389):** `model_hash`
> previously scanned the _envelope_ top level
> (`{ir_version, validated_by, model:{…}}`) rather than descending into `model`,
> so it hashed nothing and returned `SHA256("")` for every model and every
> caller — collapsing the sim cache. The fix descends into `model` (with a
> bare-object fallback) and asserts the digest is never the empty-input hash, so
> a future envelope-shape change fails loudly instead of silently colliding.
> Pinned by `cas_different_models_do_not_collide` (integration) plus
> `model_hash_envelope_is_not_empty_hash` /
> `model_hash_senses_structural_difference` (unit). _Adjacent gaps still open:
> `t_end` / the `simulation` block are excluded from the key, and `time_unit` is
> not in `structural_keys` — filed separately._

**scen_hash** — scenario delta only:

```
scen_hash = sha256(
    sorted(enable list)
    + sorted(disable list)
    + canonical_sorted(scenario param overrides)
    + canonical_sorted(sweep/design point overrides)
)
```

`scen_hash` covers only the _delta_. Base params are already in `sim_hash`.
Renaming a scenario without changing its definition preserves `scen_hash` and
reuses cached runs. Sweep point values are included in `scen_hash` because they
affect the simulation at that grid coordinate.

**Canonical sorting** means: sort parameter key-value pairs lexicographically by
key name, then serialize each as `key=value` with full-precision float
formatting. This ensures hash stability across HashMap iteration order.

### 9.3 Simulation Cache Rules

A simulation run is a **cache hit** when
`{sim_hash_8}/{scenario_slug}-{scen_hash_8}/seed_{N}/traj.tsv` exists.

| What changed           | sim_hash  | scen_hash         | Reuse               |
| ---------------------- | --------- | ----------------- | ------------------- |
| Model / base params    | changes   | —                 | none                |
| Backend or dt          | changes   | —                 | none                |
| Scenario A's overrides | unchanged | A changes, B same | B's runs reused     |
| Sweep point values     | unchanged | affected only     | other points reused |
| Add more seeds         | unchanged | unchanged         | all existing reused |
| Rename a scenario      | unchanged | unchanged         | reused (same delta) |
| camdl version          | changes   | —                 | none                |

### 9.4 Fit Staleness Detection

For fits, there are no hash-based directory names. Instead, each stage writes a
`config_hash` into its `provenance.json`. On re-run:

1. The runner computes the current `config_hash` from: model IR hash, data file
   hash, the full `[estimate]` spec, the resolved `[fixed]` values, the stage's
   algorithm settings, and the camdl version.

2. If `provenance.json` exists and its `config_hash` matches → cache hit, skip.

3. If `provenance.json` exists but the hash differs → **error**:

```
error: stage 'mle' has stale results (config changed since last run)
  stored config_hash: a1b3c4d5
  current config_hash: f9e2b047
  Changes detected:
    [estimate.beta] bounds: [0.01, 2.0] → [0.01, 5.0]
    [stages.mle] cooling: 0.70 → 0.95
  Options:
    --force    overwrite existing results
    camdl fit new --from fits/01.toml fits/01_v2.toml
```

4. `--force` overwrites without error.

```rust
pub struct ConfigHasher;

impl ConfigHasher {
    pub fn compute(
        ir: &CompiledModel,
        observations: &IndexMap<String, PathBuf>,
        estimate: &IndexMap<String, EstimateSpec>,
        fixed: &IndexMap<String, f64>,
        stage_name: &str,
        stage: &Stage,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(ir.to_canonical_bytes());
        // Hash all data files, sorted by stream name for stability
        for (name, path) in observations.iter() {
            hasher.update(name.as_bytes());
            hasher.update(&hash_file(path));
        }
        for (name, spec) in estimate.iter() {
            hasher.update(name.as_bytes());
            hasher.update(&serde_json::to_vec(spec).unwrap());
        }
        for (name, val) in fixed.iter() {
            hasher.update(name.as_bytes());
            hasher.update(&val.to_le_bytes());
        }
        hasher.update(stage_name.as_bytes());
        hasher.update(&serde_json::to_vec(stage).unwrap());
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        hex::encode(hasher.finalize())
    }
}
```

### 9.5 run.json — the per-run metadata contract

> **This subsection is the consumer contract.** `run.json` is the API between
> camdl and everything that reads its output (`camdl list/show/cat`,
> camdl-viewer, notebooks). The current on-disk schema is the `runid::RunRecord`
> (in `rust/crates/runid/src/record.rs`): a `run_id` leaf address, the ordered
> per-level hashes (`levels`), upstream `deps`, a `kind`
> (`runid::ArtifactKind`), a `RunStatus`, and a recorded-not-hashed `inputs`
> payload for display. The path-shape and record contract is documented
> authoritatively in `docs/dev/cas-path-shape-contract.md`; the illustrative
> envelope below predates the content-addressed store and is retained for
> orientation only — read `RunRecord` for the exact fields.

Every run directory contains one `run.json` with a shared envelope and a
kind-specific `kind` payload (serde `tag = "kind"`):

```json
{
  "hash": "37a7ee460c63aa5c2f5f1107cfc962e815faa6a153a3ab5d64adb1748edc9d53",
  "version": "0.1.0+8cdedd0",
  "created_at": "2026-05-30T18:58:16Z",
  "argv": ["camdl", "simulate", "he_measles.camdl", "--seed", "1", "..."],
  "status": { "completed": { "wall_time_seconds": 0.0076 } },
  "label": null,
  "kind": {
    "kind": "simulate",
    "model": "he_measles.camdl",
    "model_hash": "<sha256 of structural IR — see §9.2>",
    "scenario": "baseline",
    "sim_hash": "162c0116...",
    "scen_hash": "e9235d61...",
    "seed": 1,
    "backend": "chain_binomial",
    "dt": 1.0,
    "sweep_point": { "vacc_eff": 0.5 },
    "parameters_provenance": {
      "beta": { "value": 0.4, "source": "fixed" },
      "gamma": { "value": 0.2, "source": "scenario" }
    }
  }
}
```

**Envelope fields** (present on every kind):

| field        | type            | meaning                                                                                 |
| ------------ | --------------- | --------------------------------------------------------------------------------------- |
| `hash`       | string (64-hex) | content hash of this run's inputs; dir uses first 8                                     |
| `version`    | string          | camdl version that produced the run                                                     |
| `created_at` | RFC-3339 string | UTC start time                                                                          |
| `argv`       | string[]        | the invocation, for reproducibility                                                     |
| `status`     | enum (below)    | lifecycle state                                                                         |
| `label`      | string \| null  | optional `--label`; omitted when null                                                   |
| `kind`       | string          | the `runid::ArtifactKind` (`sim`, `fit_stage`, `pfilter`, `survey`, `profile_point`, …) |

**`status` lifecycle** — written at _start_, updated at _end_, so a run is
discoverable (in `camdl list` / the viewer) **while it is still running**:

```json
"status": "running"
"status": { "completed": { "wall_time_seconds": 3847.2 } }
"status": { "failed":    { "at": "obs 31/52", "error": "PFDegenerate" } }
```

**`kind` discriminator** (`"kind"` field): `"simulate"`, `"fit"`, `"fit-stage"`,
`"profile"`, `"survey"`, `"batch"`. Each carries its own typed fields; scalars
live here (e.g. a fit stage's `best_loglik: f64`), large artifacts are sibling
files (`traj.tsv`, `trace.tsv`). `parameters_provenance` records where each
resolved parameter value came from (`fixed` / `scenario` / `fit-toml` /
`model-default` / `--fixed`), per the `ParameterResolver`.

### 9.7 Enumerating a batch — no manifest file

There is no batch-level `manifest.json`. Each completed cell is independently a
content-addressed `run.json` leaf under `sims/` (the system of record). To
enumerate a sweep, walk those leaves or read the derived `index.json` (`§9.6` /
`camdl reindex`); `camdl list` and `camdl batch status` both do this live.
`batch status` re-plans the sweep from the batch TOML and counts which cells
already have a committed leaf.

---

## 10. Output File Schemas

### 10.1 Trajectories (traj.tsv)

One row per output time. First line is a version comment.

```
#camdl traj v1
t	S	E	I	R	flow_infection	flow_recovery
0.0	999990	0	10	0	0	0
1.0	999985	3	11	1	5	1
2.0	999978	5	14	3	7	3
```

Column types:

- `t`: float64 (simulation time)
- Compartment columns: int64 for integer compartments, float64 for `real`
- `flow_*` columns: uint64, cumulative firings since previous output time

### 10.2 Summary Tables

For batch runs, `camdl summarize` reads trajectory files and computes:

```tsv
scenario	seed	peak_I	tpeak_I	final_I	integral_I	...
baseline	1	342	45.0	0	12847	...
with_sia	1	218	52.0	0	8934	...
```

Summary statistics computed automatically for every non-time column:

| Statistic    | Definition                       |
| ------------ | -------------------------------- |
| `peak_X`     | Maximum value of X               |
| `tpeak_X`    | Time of peak (first occurrence)  |
| `final_X`    | Last value of X                  |
| `integral_X` | Sum of X across all output times |

### 10.3 draws.tsv — Complete Parameter Vectors

Posterior draws output contains ALL model parameters, not just the estimated
subset. Fixed parameters are constant across rows:

```tsv
gamma	rho	k	beta	N0	I0
0.098	0.042	8.3	0.34	1000000	10
0.102	0.039	9.1	0.34	1000000	10
0.095	0.045	7.8	0.34	1000000	10
```

This makes posterior predictive checks self-contained — no `--params` needed.
The draws file IS the complete parameter specification. You can see what was
estimated vs fixed by inspecting column variance.

Written by the runner at the end of sampling stages:

```rust
impl FitConfig {
    pub fn write_complete_draws(
        &self,
        stage_draws: &[IndexMap<String, f64>],
        output_path: &Path,
    ) -> Result<()> {
        let fixed = self.fixed.resolve()?;
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_path(output_path)?;

        let all_names: Vec<&str> = self.estimate.keys()
            .chain(fixed.keys())
            .map(|s| s.as_str())
            .collect();
        writer.write_record(&all_names)?;

        for draw in stage_draws {
            let row: Vec<String> = all_names.iter()
                .map(|name| {
                    draw.get(*name)
                        .or_else(|| fixed.get(*name))
                        .map(|v| format!("{:.17e}", v))
                        .unwrap_or_default()
                })
                .collect();
            writer.write_record(&row)?;
        }
        Ok(())
    }
}
```

---

## 11. Predictive Workflows

### 11.1 Prior Predictive Check

_"Does my model, under priors, generate data that looks plausible?"_

```bash
camdl simulate models/sir.camdl \
    --draws prior --fit fits/02_fix_beta.toml --n 500 \
    --replicates 5 --obs prior_pred.tsv
```

Samples 500 parameter vectors from the joint prior for every estimated
parameter, resolved through the precedence chain:

```
fit.toml [estimate.<p>.prior]   (sensitivity-analysis override)
  ↓ if absent
model IR  parameter.prior        (from `~ <dist>(...)` in .camdl)
  ↓ if absent
ERROR — `--draws prior` cannot sample from a flat/improper prior
```

Fixed parameters are filled in from `[fixed]`. Runs 5 stochastic replicates per
draw. Generates synthetic observations.

**Requires a proper prior on every estimated parameter.** If any parameter has
no prior in either source — or has `prior = { flat = {} }` in the fit toml — the
error names the offending parameters and lists both remediation paths:

```
error: --draws prior requires a proper (non-flat) prior on every estimated parameter.
  Missing or flat priors: beta, k
  To fix, either:
    (i)  add `prior = { <dist> = { ... } }` to `[estimate.beta]` in your fit.toml
         (e.g. `prior = { log_normal = { mu = 0, sigma = 1 } }`), OR
    (ii) add a `~ <dist>(...)` declaration to parameter `beta` in your .camdl model file.
```

Fit toml priors are optional — they're useful for sensitivity analysis ("what if
I widen the prior on beta?") without editing the model. When the fit toml is
silent, the model file's `~` priors are the source of truth (gh#86; sibling of
the `fit run` precedence chain landed in gh#75).

### 11.2 Posterior Predictive Check

_"Does my fitted model generate data that looks like the real data?"_

```bash
camdl simulate models/sir.camdl \
    --draws results/fits/02_fix_beta/posterior/draws.tsv \
    --replicates 10 --obs ppc.tsv
```

### 11.3 Scenario Prediction Under Posterior Uncertainty

_"What would happen under an SIA, given what we learned from the data?"_

```bash
camdl simulate models/sir.camdl \
    --draws results/fits/02_fix_beta/posterior/draws.tsv \
    --scenario baseline,with_sia \
    --replicates 10 --obs-dir obs/
```

For each (draw, seed) pair, both scenarios are simulated with the same seed.
This gives paired comparisons (cases_averted = baseline - with_sia) that
propagate posterior uncertainty. Note that the runtime uses a stateful PRNG
rather than event-keyed RNG — pre-intervention trajectories match only when both
runs consume the RNG in the same order; any structural difference that shifts
RNG ordering also breaks the coupling.

### 11.4 Uniform Exploration

_"What does the model do across parameter space?"_

```bash
camdl simulate models/sir.camdl \
    --draws uniform --n 500 --replicates 1
```

Samples uniformly from parameter bounds. No Bayesian pretension — this is
space-filling exploration for model debugging.

### 11.5 Simulation-Based Calibration (SBC)

**Simulation-based calibration** — generating synthetic data at known
parameters, fitting each dataset, and checking parameter recovery — is planned
as a future `camdl sbc` command. The infrastructure for it (prior predictive via
`--draws prior`, the fit pipeline, `draws.tsv` output) is in place; the
orchestration layer that connects them is not yet built.

---

## 12. Priors: Beliefs Belong With Parameters

Priors are declared in the model file with `~` syntax on the parameter:

```camdl
parameters {
    beta  : rate in [0.01, 2.0] ~ log_normal(mu = -1.0, sigma = 0.5)
    gamma : rate in [0.05, 1.0] ~ half_normal(sigma = 0.3)
    rho   : probability in [0.001, 1.0] ~ beta(alpha = 2.0, beta = 5.0)
}
```

**Why in the model file?** Priors are beliefs about parameters — they answer
"what do I think about beta before seeing data?" That belief belongs with the
parameter declaration, where anyone reading the model can see it. This design
follows Stan, PyMC, and Turing.jl.

Camdl exists to support decisions about people's lives. Epidemiological models
feed into vaccination campaigns, outbreak response, resource allocation. Getting
uncertainty wrong means making confident-looking recommendations on shaky
foundations. Prior predictive checks are the first line of defense: "do my
stated beliefs produce data that looks plausible before I've seen any real
data?" Making priors discoverable and declarative in the model file is part of
doing uncertainty right.

**`camdl simulate --draws prior -n 500` works with just a model file** — no
fit.toml required. Every parameter must either have a prior (sampled from) or a
default value (held constant). Parameters with neither produce a clear error
pointing at the options: add `~ prior(...)`, supply `--fit FIT.toml`, or use
`--draws uniform`.

**fit.toml can override model priors for sensitivity analysis.** The precedence
chain during inference:

```
fit.toml [estimate] prior override   (sensitivity analysis)
  ↓ if absent
model IR parameter.prior             (from ~ syntax in .camdl)
  ↓ if absent
Prior::Flat                          (improper uniform)
```

fit.toml overrides preserve the sensitivity-analysis workflow: "what happens if
I use a wider prior on beta?" without editing the model file.

**When are priors required?** Bayesian sampling methods (PGAS, PMMH) use the
prior density in their acceptance ratios. MLE methods (IF2) ignore priors. For
`--draws prior`, every parameter must have _either_ a prior (to sample from)
_or_ a concrete value — either declared in the IR or pinned by a selected
`--scenario`.

**Supported distributions**: `uniform`, `normal`, `log_normal`, `half_normal`,
`beta`, `gamma`, `exponential`. See the language spec for parameterization
conventions.

---

## 13. Utility Commands

### 13.1 `camdl fit table`

Walks a results tree and renders one row per fit (terminal-stage method result,
gate verdict, Â, MLE). Read-only; all state is recovered from the on-disk
`run.json` + per-stage outputs.

```
$ camdl fit table results/fits/
```

To enumerate fits without the per-stage projection, use the generic run browser:
`camdl list --kind fit`. For a single fit's full interpretation (Â / gate
verdict / MLE table) use `camdl fit summary <dir>`.

### 13.2 `camdl fit diff`

```
$ camdl fit diff fits/01_all_free.toml fits/02_fix_beta.toml

Parameter changes:
  beta:  [estimate] bounds=[0.01, 2.0]  →  [fixed] 0.34

Stage changes:
  mle:       chains 8→4, particles 1000→2000, cooling 0.70→0.95
  posterior:  (new) pgas, 4 chains, 50 particles, 5000 sweeps
  evaluate:   (new) pfilter, 10000 particles, 100 replicates
```

### 13.3 `camdl fit new`

```
$ camdl fit new --from fits/01_all_free.toml fits/02_fix_beta.toml

Created fits/02_fix_beta.toml
  [provenance] derived_from = "fits/01_all_free.toml"
  [stages.mle] init = "from_mle", init_mle = "results/fits/01_all_free/mle"
```

### 13.4 `camdl summarize`

```bash
camdl summarize results/sims/
```

Reads trajectory files and produces per-scenario summary tables with
automatically computed statistics (peak, time-of-peak, final value, integral)
for every non-time column. Output written to `results/sims/summary/`.

---

## 14. Observation Semantics

Observation blocks project simulator state into the scalar `projected` value
that the likelihood evaluates. camdl supports two projection modes —
**incidence** (accumulated flow) and **prevalence / snapshot** (point-in-time
state). Both are available in `simulate --obs`, the particle filter, and all
inference methods (IF2, PGAS, PMMH).

### 14.1 Projection modes

- **Incidence** (`incidence(X)`, IR `CumulativeFlow`): sum of per-transition
  flow counters over the interval since the last observation. Appropriate for
  daily case notifications, weekly deaths, cumulative reported hospitalizations
  — any **event count over an interval**.
- **Prevalence** (`prevalence(X)`, IR `CurrentPop`; or `prevalence(X1, X2)` → IR
  `CurrentPopSum`): integer compartment count(s) read at the observation
  instant. Appropriate for hospital bed occupancy, ICU census, wastewater
  concentration snapshots, seroprevalence surveys — any **point-in-time state
  reading**.
- **Derived expression** (`projected = <expr>`, IR `DerivedExpr`): arbitrary
  expression over compartment state (e.g. `B1 + B2`, `I / (S + I + R)`),
  evaluated at the observation instant.

Incidence streams accumulate flow counters between observations and **reset
after the likelihood is scored**. Prevalence and derived-expr streams read the
state vector and **do not reset** — each observation is independent of the
previous one.

### 14.2 Snapshot timing

The snapshot is the value of the projection expression _at_ the observation time
`t`, evaluated against the simulator state at `t`. The following rules specify
what "state at `t`" means per backend:

- **Gillespie SSA (continuous-time):** state is piecewise-constant between
  events. The snapshot reads the state that has been in effect since the last
  event preceding `t`. If an event or scheduled intervention fires exactly at
  `t`, the snapshot reads the **post-event** state.
- **Chain-binomial (discrete-time, step `dt`):** the snapshot reads the state at
  the step boundary that lands on, or first passes, `t`. For `dt = 1` with daily
  observations this is exact; for `dt < 1` the snapshot is the state at the
  first step boundary `≥ t`.
- **ODE (continuous integrator):** dense-output evaluation of the integrator
  state at exactly `t`.

### 14.3 Interaction with scheduled interventions

If a scheduled intervention fires at the same time as an observation, the
snapshot reads the **post-intervention** state. Rationale: the data was
generated in a world where the intervention had already fired; evaluating the
likelihood against the pre-intervention state would deterministically bias the
posterior against any scenario that correctly represents the intervention.

The step loop at an observation time `t` is:

1. Advance state to `t`.
2. Fire any scheduled interventions at `t` (`apply_interventions_at(t, …)`).
3. Evaluate the projection expression against the resulting state; pass
   `projected` to the likelihood.
4. Reset incidence counters.

This ordering is the same in `simulate --obs` (synthetic data generation) and in
the particle filter's observation tick, so likelihood evaluation and data
generation are always consistent. For chain-binomial, `step_one` already fires
scheduled interventions at `t + dt` (see
`docs/dev/incidents/2026-04-17-chain-binomial-double-fire.md`), and the PF reads
`counts_after` — the post-`step_one` state — when scoring the observation.

### 14.4 Likelihood-family guidance

Incidence and prevalence need different default likelihoods; a model that pairs
a NegBinomial with a prevalence projection is syntactically valid but usually
wrong in interpretation. The `fit run` and `pfilter` startup block lists each
stream's `(projection, likelihood)` pairing so the mismatch is visible before
the PF runs.

- **Incidence:** NegativeBinomial or Poisson with reporting rate. Support on
  ℤ≥0; overdispersion natural.
- **Prevalence, single compartment:** Binomial(N, p) with `p = projected / N`
  when the total is fixed and known; Poisson for large `N`. NegBinomial is valid
  but the dispersion parameter has a different meaning than for incidence.
- **Prevalence as a fraction** (projection ∈ [0, 1]): Beta or Binomial.

Prevalence and incidence data also have different Fisher information about the
parameters: prevalence is more informative about the recovery rate γ (decay
shape), incidence is more informative about the transmission rate β (direct flow
into I). Joint fits on both streams are strictly more informative than either
alone.

---

## Appendix A: CLI Reference

```
camdl simulate MODEL [OPTIONS]
  --params FILE             Load parameter values (repeatable)
  --param NAME=VALUE        Override single parameter (repeatable)
  --param-vec PREFIX=FILE   Override indexed params from keyed TSV
  --table NAME=FILE         Supply external table data
  --backend BACKEND         gillespie|chain_binomial|ode
  --dt DT                   Step size for discrete-time backends
  --seed N                  Single seed
  --seeds SPEC              Multiple seeds: "1:1000", "{n=100}", "{list=[42,137]}"
  --scenario NAME[,NAME]    Named scenarios (comma-separated)
  --enable NAME             Enable intervention (mutually exclusive with --scenario)
  --disable NAME            Disable intervention
  --sweep "NAME=V1,V2,..."  Parameter sweep, list syntax (repeatable; Cartesian product)
  --draws SOURCE            "path.tsv" | "prior" | "uniform"
  --fit FILE                fit.toml for --draws prior
  -n N                      Number of draws (prior/uniform)
  --replicates N            Stochastic replicates per draw
  --obs FILE                Write synthetic observations (wide-format TSV)
  --obs-dir DIR             Write one TSV per observation stream
  --obs-only FILE           Like --obs, suppress trajectory output
  --obs-only-dir DIR        Like --obs-dir, suppress trajectory output
  --parallel N              Concurrent runs
  --output-dir DIR          Output root (default: results/)
  --force                   Re-run cached results

camdl batch run FILE [OPTIONS]
  --output-dir DIR          Override output_dir from the TOML
  --parallel N              Override parallel from the TOML
  --dry-run                 Print the resolved sweep grid + cache summary; exit
  --resume                  Skip runs whose output already exists (default)
  --force                   Re-run even if output exists

camdl simulate status FILE     Print status of a batch TOML's output tree

camdl fit run CONFIG [OPTIONS]
  --stage NAME              Run specific stage only
  --seed N                  RNG seed override
  # `--starts-from` was removed in the 2026-05-25 CLI UX revision; declare
  # cross-stage chaining in the toml as `init = "from_mle"` +
  # `init_mle = "<stage-or-dir>"`. The old flag remains as a hidden trap
  # that errors with the replacement spelled out.
  --sweep "NAME=V1,V2,..."  Sweep over fixed params (repeatable; Cartesian product)
  --skip-chains N[,N]       Skip specific chain indices
  --resume                  Resume partially completed sampling stage
  --parallel N              Concurrent sweep points
  --output-dir DIR          Output root override
  --force                   Re-run (overwrite stale results)

camdl fit table [DIR]          One row per fit under a results tree
camdl fit summary DIR          Â / gate verdict / MLE table for one fit
camdl fit diff A.toml B.toml   Show differences between fit configs
camdl fit new --from A B       Create derived fit config with lineage
camdl summarize DIR            Compute summary statistics from trajectories
```

**Batch TOML is v1 and will change.** Field names (`[config]`, `[[scenario]]`,
`[sweep]`) are standalone and pre-date the v2 run-system types (`SimulateJob`,
`SweepSpec`, `Seeds` in `rust/crates/cli/src/fit/config_v2.rs`). A future
version will align the schema with v2. **External tooling should not assume the
current field names survive unchanged.** Open an issue if you're writing such
tooling and need a migration window.

Sensitivity analysis (Sobol indices and similar) is not a camdl concern. Run
`camdl batch run` to produce the output tree, then compute indices with R's
`sensitivity` package or Python's `SALib`.

---

## Appendix B: Parameter Files

### B.1 params.toml — a point m ∈ M

```toml
beta = 0.3
gamma = 0.1
sigma = 0.2
rho = 0.4
k = 5.0
N0 = 1000000
I0 = 10
```

One key-value pair per declared parameter. Used by `camdl simulate` for forward
simulation and by fit.toml's `[fixed] from_file` for bulk fixed values.

### B.2 Indexed Parameter Overrides

For spatial models with indexed parameters (`R0[patch]`), use `--param-vec`:

```tsv
R0_kano     2.1
R0_lagos    1.8
R0_sokoto   2.4
```

```bash
camdl simulate model.camdl --params p.toml --param-vec R0=r0_init.tsv
```

Matched by parameter name (not position).
