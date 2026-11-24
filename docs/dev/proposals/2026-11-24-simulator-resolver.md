# One resolver for the simulator: backend, step, integrator and sampler

Date: 2026-11-24. Status: accepted by the maintainer on 2026-11-24, with two
behaviour decisions: `design` and an anchored `--to` refuse a written step that
disagrees with the model's (commit 6), and `pfilter --fit` and `profile --fit`
use the fit's binomial sampler, which re-keys them (commit 7). The audit's fixes
for `simulate --params` and `--integrator` are on main (f43c73d27); the type
below is not built.

## The problem

A run's simulator is four settings: the backend (chain-binomial, Gillespie,
ODE), the integration step `dt`, the ODE integrator, and the binomial sampler.
Each can come from several places: a command-line flag, a fit file, the model's
`simulate {}` block, a stored fit's record, a params file's `[provenance]`
block, or a default. The run spec (§1.3.1) states the rule that combines them:
on a command that fits or scores data, a written step must agree with the
model's; a replay of a stored fit runs on the fit's simulator, and a differing
flag runs with a note; exploratory `simulate` lets a flag override.

The rule is implemented in one place (`dt_rule.rs`) for most commands, but
nothing forces a command to use it. Any code that holds a mutable `dt` or
`backend` can decide precedence itself. An audit of every command found:

- `simulate --params mle_params.toml` decided backend and step by hand in
  `main.rs`. With `--backend` equal to the fit's, it silently dropped the fit's
  step to the model's or 1.0, so the run was not the fit's simulator. It now
  goes through the shared replay rule (on main).
- `--integrator` was accepted off the ODE backend on `simulate` and
  `predict MODEL`, where it selects nothing but still changed the run id. It is
  now refused on every command, by one function (on main).
- `camdl design lhs --fit fit.toml` and `simulate --fit fit.toml` with an
  anchored `--to` read the fit file's `[config] dt` without the rule checking
  it, so a disagreeing step is ignored rather than refused (it affects only the
  step those commands use internally).
- `pfilter --fit` of a fit run with the `btrs` binomial sampler scores with the
  default `btpe`; it never reads the fit's sampler, while `predict` does.
- Two functions decide whether a fit file wrote `[config] dt`
  (`table_declares_key` and `written_config_dt`). They agree today.

The audit found every other multi-source input (seeds, particle count, number of
draws, time format, horizon, forcing policy) decided by exactly one function per
command.

## The design

A `Simulator` value holds the four settings, with private fields, and is built
only by one function. The engine's entry points take a `Simulator` in place of
separate `dt`, `backend`, `integrator` and `binomial` fields, so code that
decides the simulator itself does not compile. This is the pattern camdl already
uses for parameter values, where `resolve_parameters` is the only writer of a
parameter's value.

### The types it is built from, all of which exist today

The four settings already have types; what does not exist is one value that
holds them and one function that builds it.

```rust
// rust/crates/cli/src/args/types.rs:102 — which engine runs the model.
pub enum ForwardBackend { Gillespie, ChainBinomial, Ode }

// rust/crates/store/src/identity.rs:142 — an f64 that is neither NaN nor
// infinite, with -0.0 made +0.0, so it can enter a run id. The step is stored
// as one today (`sim_store::SimConfig.dt`), but carried as a bare f64 until then.
pub struct Finite(f64);

// rust/crates/ir/src/model.rs:30 — the ODE method, as the model declares it.
// Rk45 carries its tolerances, so tolerances without rk45 cannot be written.
pub enum Integrator { Rk4, Rk45 { atol: Option<f64>, rtol: Option<f64> } }

// rust/crates/cli/src/args/types.rs:205 — what `--integrator` can say: the
// method only. Applied to the model, Rk45 keeps the model's tolerances.
pub enum IntegratorArg { Rk4, Rk45 }

// rust/crates/sim/src/rng.rs:442 — the chain-binomial backend's binomial
// sampler. Same distribution, different random stream, so it changes results.
pub enum BinomialAlgorithm { Btpe /* default */, Btrs }

// rust/crates/cli/src/fit/fitted_model.rs:378 — what a stored fit ran on.
pub struct FittedSimulator {
    pub backend: Option<ForwardBackend>, // None: a fit config with no [method]
    pub dt: f64,
    pub integrator: Integrator,
    pub settings: FittedSettings,        // allow_degenerate_rates, allow_forcing_extrapolation
}
// fitted_model.rs:468: FittedDynamics is FittedSimulator without `settings`,
// built from a params file's [provenance] block, which records no [config].
```

### What carries the simulator today

The same four loose fields are copied from type to type, and each holder can be
written by whoever builds it:

```rust
// rust/crates/cli/src/sim_job.rs:36-48 (today)
pub struct SimulateJob {
    pub backend: ForwardBackend,
    pub dt: f64,
    pub binomial: sim::rng::BinomialAlgorithm,
    pub integrator: Option<IntegratorArg>, // None: the model's
    // ...
}
// The same fields again in util.rs SimRun (2856-2866), sim_config.rs RunConfig
// (62-63), EngineSettings (79-80) and ResolvedCell (137-139).
```

### The new types

```rust
// rust/crates/cli/src/simulator.rs (new)

/// The simulator a run integrates with. Fields are private and the only
/// constructor is `Simulator::resolve`.
pub(crate) struct Simulator {
    backend: ForwardBackend,
    dt: Finite,
    integrator: Integrator,     // the full method, tolerances included
    binomial: BinomialAlgorithm,
    source: SimulatorSource,
}

/// Where the resolved simulator came from, for the run's notes and record.
pub(crate) enum SimulatorSource {
    /// Nothing written: the model's declared step and integrator, or the defaults.
    Model,
    /// The caller wrote settings the rule accepted, named as written
    /// (`--dt`, `[config] dt`, `--backend`, `--integrator`, `--binomial`).
    Written { by: Vec<&'static str> },
    /// A replay of a stored fit, or of a params file's provenance. `moved`
    /// lists the settings a flag moved off the fit's; non-empty means the
    /// "different simulator" note was printed.
    Fit { name: String, moved: Vec<Setting> },
}
pub(crate) enum Setting { Backend, Dt, Integrator, Binomial }

/// What the model declares in `simulate {}`.
pub(crate) struct ModelSimulate<'a> { dt: Option<f64>, integrator: &'a Integrator }

/// What the caller wrote, from flags and the fit file. `None` means not written.
pub(crate) struct WrittenSimulator {
    backend: Option<ForwardBackend>,
    dt: Option<f64>,          // --dt
    config_dt: Option<f64>,   // a fit file's [config] dt
    integrator: Option<IntegratorArg>,
    binomial: Option<BinomialAlgorithm>,
}

/// Which of run spec §1.3.1's three rules applies.
pub(crate) enum Basis<'a> {
    /// fit run, predict from a model, profile, pfilter: a written step must agree.
    Fitting { backend: ForwardBackend },
    /// A stored fit or a params file's provenance: the fit's, and a differing flag
    /// runs with a note.
    Replay(&'a StoredSimulator),
    /// simulate without a fit: a flag overrides.
    Exploratory,
}

/// FittedSimulator and FittedDynamics merged: `settings` is None for a
/// provenance block, which records no [config].
pub(crate) struct StoredSimulator {
    name: String,
    backend: Option<ForwardBackend>,
    dt: f64,
    integrator: Integrator,
    binomial: Option<BinomialAlgorithm>,
    settings: Option<FittedSettings>,
}

impl Simulator {
    /// The run's simulator, and the note to print when a replay moved off the
    /// fit's. Refuses a disagreeing step under Fitting, --integrator off ode
    /// and --binomial off chain_binomial under every basis.
    pub fn resolve(model: ModelSimulate, written: WrittenSimulator, basis: Basis)
        -> Result<(Simulator, Option<String>), String>;
    pub fn backend(&self) -> ForwardBackend;
    pub fn dt(&self) -> Finite;
    pub fn binomial(&self) -> BinomialAlgorithm;
    /// Writes the integrator into the model, as `apply_integrator_override` does today.
    pub fn apply_to(&self, model: &mut ir::Model);
}
```

### Before and after at one holder

```rust
// before: four fields anyone can set
let job = SimulateJob { backend, dt, binomial, integrator, /* ... */ };

// after: one value only the resolver makes
let (simulator, note) = Simulator::resolve(model_simulate, written, basis)?;
if let Some(n) = note { out.say(n); }
let job = SimulateJob { simulator, /* ... */ };
```

The values flow as:

```
flags, fit file, model, stored fit ──▶ Simulator::resolve ──▶ Simulator
        │                                                       │
        └── nothing else decides                                ├──▶ SimRun / SimulateJob / RunConfig
                                                                ├──▶ predict, pfilter, profile configs
                                                                └──▶ fit run (writes [config] dt as today)
```

The sim crate's filter and IF2 configs keep plain `f64` fields; the `Simulator`
is lowered at that boundary.

## What changes, and what it costs

About 40 call sites: simulate's resolution in `main.rs`, the engine's job and
config types (`sim_job.rs`, `util.rs`, `sim_config.rs`, `engine.rs`), predict
(`fixed_point.rs`, `predict.rs`), pfilter, profile, fit run's config, and the
stored-fit readers (`fitted_model.rs`, `compare.rs`, `contrasts.rs`).

No run id should change: the stored simulation settings, the fit problem's
`[config] dt` and the prediction's numerics keep their encodings. The exceptions
are the two behaviour changes listed below, each its own commit.

Seven commits:

1. `simulator.rs` with `resolve` and its unit tests, wired into `simulate`.
2. The engine's job and config types carry a `Simulator` (pin-neutral, checked
   by the class tests and byte pins).
3. `predict` from a model file and from a stored fit.
4. `pfilter` and `profile`.
5. `fit run` and the stored-fit readers, merging the two readers.
6. Behaviour: `design` and the anchored `--to` refuse a disagreeing step.
7. Behaviour, with a re-key: `pfilter --fit` and `profile --fit` use the fit's
   binomial sampler.

## Not covered

The sim crate's internal configs, which take numbers after resolution. The
forcing-extrapolation policy and degenerate-rate setting, which are already
decided in one place per command.
