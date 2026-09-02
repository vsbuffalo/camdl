// ── Args structs only ─────────────────────────────────────────────────────────
//
// The clap subcommand tree (Cli, Command, FitCmd, BatchCmd, DataCmd) lives
// in main.rs — it's the canonical, dispatched parser. This module owns
// only the per-command argument structs (FitRunArgs, SimulateArgs, etc.)
// referenced by main.rs's tree.

pub mod types;

use std::path::PathBuf;
use clap::{Args, ArgGroup};
use crate::colored_help;
use types::{ForwardBackend, DataSpec, ListDuration, ParamOverride, ParamVecSpec, RwSd, SeedSpec, SweepSpec, TableSpec};

// ─── Shared help-text constants ───────────────────────────────────────────────
//
// Per docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md
// §"Help-text rewrite", the `--init` and `--fixed` flags share a single
// normative description across every inference subcommand. clap's
// `long_about` accepts these as constants so a doc edit lands in one
// place instead of being copied per arg.

/// CLI-side enum for `--init <MODE>` on inference subcommands. Mode
/// names are snake_case (matches the in-tree `InitMethod` deserializer
/// per agent-handoff `cb47ee1`). The CLI parses the bare tag here, then
/// the dispatch site combines it with the companion path flags
/// (`--posterior`, `--mle`, init-mode `--params`) to build a full
/// `crate::fit::init::InitMethod` payload-carrying variant.
///
/// Why a separate enum: `crate::fit::init::InitMethod` has
/// payload-bearing variants (`FromPosterior { source: PosteriorSource }`,
/// etc.). clap's `ValueEnum` can only surface payload-free variants —
/// the post-parse construction lives at the dispatch site so the
/// arg-struct stays declarative.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum InitModeTag {
    /// every chain at the seeded base params
    Single,
    /// per-chain uniform draw within [estimate] bounds
    Uniform,
    /// Latin-hypercube stratified within [estimate] bounds
    Lhs,
    /// Stan-style: i.i.d. U(-2,2) on the unconstrained scale, mapped into
    /// bounds (boundary-avoiding, scale-invariant; the default)
    #[clap(name = "uniform_unconstrained")]
    UniformUnconstrained,
    /// per-chain sample from each parameter's `~ <dist>` declaration
    #[clap(name = "from_prior")]
    FromPrior,
    /// per-chain row from a posterior draws TSV (requires `--posterior`)
    #[clap(name = "from_posterior")]
    FromPosterior,
    /// all chains at the MLE from a prior fit (requires `--mle`)
    #[clap(name = "from_mle")]
    FromMle,
    /// all chains at the point in a hand-written flat params TOML
    /// (requires the init-mode `--params <toml>` companion)
    #[clap(name = "from_params")]
    FromParams,
    /// top-K rows from a `camdl survey` landscape (requires `--survey-path`)
    #[clap(name = "survey_top_k")]
    SurveyTopK,
}

impl std::fmt::Display for InitModeTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            InitModeTag::Single        => "single",
            InitModeTag::Uniform       => "uniform",
            InitModeTag::Lhs           => "lhs",
            InitModeTag::UniformUnconstrained => "uniform_unconstrained",
            InitModeTag::FromPrior     => "from_prior",
            InitModeTag::FromPosterior => "from_posterior",
            InitModeTag::FromMle       => "from_mle",
            InitModeTag::FromParams    => "from_params",
            InitModeTag::SurveyTopK    => "survey_top_k",
        })
    }
}

impl InitModeTag {
    /// Combine the parsed CLI tag with the companion path flags to
    /// build the full `InitMethod` variant. Validates that companion
    /// args match the chosen mode (and rejects payload args for modes
    /// that don't accept them).
    ///
    /// Per the proposal §"`--init` family", payload modes require
    /// their companion path; payload-free modes reject companions to
    /// keep "the same flag means the same thing on every subcommand"
    /// honest.
    pub fn to_init_method(
        self,
        posterior: Option<&PathBuf>,
        mle:       Option<&PathBuf>,
        init_params: Option<&PathBuf>,
    ) -> Result<crate::fit::init::InitMethod, String> {
        use crate::fit::init::{InitMethod, MleSource, PosteriorSource};
        // Reject companion paths on incompatible modes — better an
        // error at parse time than a silent ignore.
        match self {
            InitModeTag::FromPosterior => {}
            _ => if posterior.is_some() {
                return Err(format!(
                    "--posterior is only valid with --init from_posterior \
                     (got --init {})", self));
            }
        }
        match self {
            InitModeTag::FromMle => {}
            _ => if mle.is_some() {
                return Err(format!(
                    "--mle is only valid with --init from_mle \
                     (got --init {})", self));
            }
        }
        match self {
            InitModeTag::FromParams => {}
            _ => if init_params.is_some() {
                return Err(format!(
                    "--params is only valid with --init from_params \
                     (got --init {}). For setting parameter values, \
                     use --fixed NAME=VALUE or --fixed-file <toml>.", self));
            }
        }
        Ok(match self {
            InitModeTag::Single  => InitMethod::Single,
            InitModeTag::UniformUnconstrained => InitMethod::UniformUnconstrained,
            InitModeTag::Uniform => InitMethod::Uniform,
            InitModeTag::Lhs     => InitMethod::Lhs,
            InitModeTag::FromPrior => InitMethod::FromPrior,
            InitModeTag::SurveyTopK => InitMethod::SurveyTopK,
            InitModeTag::FromPosterior => {
                let p = posterior.ok_or_else(|| {
                    "--init from_posterior requires --posterior <path>".to_string()
                })?;
                // Distinguish a file from a directory at construction
                // time so the loader can give a precise error message.
                let source = if p.is_dir() {
                    PosteriorSource::FitDir(p.clone())
                } else {
                    PosteriorSource::DrawsTsv(p.clone())
                };
                InitMethod::FromPosterior { source }
            }
            InitModeTag::FromMle => {
                let p = mle.ok_or_else(|| {
                    "--init from_mle requires --mle <path>".to_string()
                })?;
                let source = if p.is_dir() {
                    MleSource::FitDir(p.clone())
                } else {
                    MleSource::File(p.clone())
                };
                InitMethod::FromMle { source }
            }
            InitModeTag::FromParams => {
                let p = init_params.ok_or_else(|| {
                    "--init from_params requires --params <toml>".to_string()
                })?;
                InitMethod::FromParams { path: p.clone() }
            }
        })
    }
}

/// Shared `long_about` for `--init <MODE>` on inference subcommands
/// (`if2`, `profile`, `fit run`). Mode names are snake_case to match
/// the in-tree `InitMethod` deserializer (`from_prior`, not
/// `from-prior`). gh#83 / gh#85.
pub const INIT_LONG_ABOUT: &str = "\
INIT MODES (where do chain starting points come from?)

  uniform_unconstrained
                     (default) Stan-style: per-chain i.i.d. U(-2, 2) on the
                     unconstrained scale, mapped into bounds (boundary-avoiding,
                     scale-invariant; over-dispersed for MCMC diagnostics)
  single             every chain starts at the seeded base params
  uniform            per-chain U(lo, hi) over [estimate] parameter bounds
  lhs                Latin-hypercube stratified within bounds (scale-aware
                     via Transform; best full-bounds coverage at low chain counts)
  from_params        load a single point from a flat params TOML; pass
                     --params <path>. (Use this where you'd previously
                     have written --params <path> on profile or if2.)
  from_prior         sample once per chain from each parameter's `~ <dist>`
                     declaration in the .camdl source
  from_posterior     sample chain starts uniformly from a posterior draws TSV
                     (or a fit-results directory containing draws.tsv); pass
                     --posterior <path>
  from_mle           all chains at the MLE point from a prior fit; pass
                     --mle <path>
  survey_top_k       initialise from the top-K best landscape points of a
                     prior survey; pass --survey-path <dir>

Init applies only to parameters in the inference [estimate] set; parameters
in [fixed] (or absent from [estimate]) take their model value or --fixed
override regardless of init mode.";

/// Shared `long_about` for `--fixed NAME=VALUE` / `--fixed-file <toml>`
/// on every subcommand. Per the proposal §"`--fixed` semantics,
/// defined once", `--fixed` is the universal value-setter.
pub const FIXED_LONG_ABOUT: &str = "\
SET PARAMETER VALUES

  --fixed NAME=VALUE      set NAME to VALUE (repeatable; explicit form only,
                          name-only `--fixed NAME` was removed per the
                          2026-05-25 CLI UX revision)
  --fixed-file <toml>     load a flat params TOML; each top-level key is a
                          parameter name (repeatable, later files override
                          earlier ones)

ON INFERENCE SUBCOMMANDS (if2, profile, fit run)

  Any name appearing in `--fixed`/`--fixed-file` is pinned at the supplied
  value AND removed from the inference [estimate] set if present (so that
  `--fixed gamma=0.1 --sweep tau=lin(...)` is the canonical pattern for
  profile-likelihood slicing — hold gamma, sweep tau). A warning is emitted
  naming each parameter kicked from [estimate] and the source that did it.

PRECEDENCE (last wins, per docs/camdl-run-spec.md §1.3)

  1. Model parameter default in the .camdl source
  2. fit.toml [fixed] block (if --fit is in scope)
  3. --fixed-file <toml> (layered in declared order)
  4. Scenario preset (--scenario NAME)
  5. --fixed NAME=VALUE   (highest)";

// ─── Shared flat arg groups ───────────────────────────────────────────────────

/// `--params FILE` (repeatable) + `--param NAME=VALUE` (repeatable) +
/// `--table NAME=FILE` (repeatable). Used on **non-inference**
/// subcommands (`simulate`, `pfilter`, `eval`) where `--params` is
/// unambiguous — every value is trivially "fixed" because no inference
/// is happening.
///
/// Inference subcommands use [`InferenceModelOverrides`] instead;
/// `--params` / `--param` were removed there (M-1 break per
/// 2026-05-25 CLI UX rev 2).
#[derive(Args, Clone, Default)]
pub struct ModelOverrides {
    /// Parameter TOML file (may be repeated)
    #[arg(long, value_name = "FILE")]
    pub params: Vec<PathBuf>,

    /// Single parameter override, e.g. --param R0=2.5 (may be repeated)
    #[arg(long, value_name = "NAME=VALUE")]
    pub param: Vec<ParamOverride>,

    /// External table for table-lookup expressions, e.g. --table contact=matrix.tsv
    #[arg(long, value_name = "NAME=FILE")]
    pub table: Vec<TableSpec>,
}

/// Inference-subcommand model overrides. Mirrors
/// [`ModelOverrides`] structurally but drops `--params` / `--param` in
/// favour of `--fixed NAME=VALUE` / `--fixed-file <toml>` (the
/// universal value-setter from 2026-05-25 CLI UX rev 2).
///
/// `--params` and `--param` are reintroduced as hidden traps that
/// produce an actionable error at dispatch time (per CLAUDE.md alpha
/// posture: no backwards-compat shims, hard break with replacement
/// spelled out).
#[derive(Args, Clone, Default)]
pub struct InferenceModelOverrides {
    /// Pin NAME to VALUE (repeatable; sets value and removes the
    /// parameter from `[estimate]` if present). See `--help-fixed` for
    /// the full precedence / semantics block.
    #[arg(long = "fixed", value_name = "NAME=VALUE",
          long_help = FIXED_LONG_ABOUT)]
    pub fixed_cli: Vec<ParamOverride>,

    /// Load fixed values from a flat params TOML (repeatable, layered).
    /// Each top-level key = parameter name; later files override
    /// earlier ones. Listed parameters are removed from `[estimate]`
    /// if present.
    #[arg(long = "fixed-file", value_name = "TOML",
          long_help = FIXED_LONG_ABOUT)]
    pub fixed_files: Vec<PathBuf>,

    /// External table for table-lookup expressions, e.g. --table contact=matrix.tsv
    #[arg(long, value_name = "NAME=FILE")]
    pub table: Vec<TableSpec>,

    // ── Removed-flag traps (M-1 break per proposal §"Migration") ──
    //
    // `--param` (singular) is trapped here because no inference
    // subcommand has a legitimate use for that flag name.
    //
    // `--params` (plural) is **not** trapped at this level: profile
    // and fit-run have a legitimate `--params <TOML>` companion to
    // `--init from_params`. Trapping `--params` here would shadow
    // that companion (two clap defs with `long = "params"` in the
    // same Args graph: the trap field wins by definition order, the
    // companion is silently unreachable).
    //
    // Per CLAUDE.md alpha posture: no back-compat shims, no aliases —
    // these traps exist purely so the error message is actionable.
    // They have no other effect.
    #[arg(long = "param", value_name = "NAME=VALUE", hide = true)]
    pub _removed_param: Vec<ParamOverride>,
}

impl InferenceModelOverrides {
    /// Emit an actionable error and abort if a removed flag was used.
    /// Called by every inference-subcommand dispatch function before
    /// any other work. Matches the wording in the proposal §"Migration".
    pub fn check_removed_flags(&self, subcmd: &str) {
        if !self._removed_param.is_empty() {
            eprintln!(
                "error: --param is no longer accepted on `camdl {}`. \
                 Replacement:\n  \
                 --fixed NAME=VALUE             (set & freeze a single value)\n  \
                 --fixed-file <toml>            (load fixed values from a TOML file)\n\
                 See `camdl {} --help` (SET PARAMETER VALUES section).",
                subcmd, subcmd);
            std::process::exit(1);
        }
    }
}

/// `--scenario` XOR `--enable`/`--disable`
#[derive(Args, Clone, Default)]
pub struct ScenarioArgs {
    /// Named scenario defined in the model (conflicts with --enable/--disable)
    #[arg(long, conflicts_with_all = ["enable", "disable"])]
    pub scenario: Option<String>,

    /// Enable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenario")]
    pub enable: Vec<String>,

    /// Disable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenario")]
    pub disable: Vec<String>,
}

/// `--backend` + `--dt`
#[derive(Args, Clone)]
pub struct SimBackend {
    /// Simulation backend (default: chain_binomial)
    #[arg(long)]
    pub backend: Option<ForwardBackend>,

    /// Step size for discrete-time backends (default: 1.0)
    #[arg(long)]
    pub dt: Option<f64>,

    /// ODE integrator method override: rk4 (fixed) or rk45 (adaptive). gh#166.
    /// Tolerances are a model property — set them in
    /// `simulate { integrator = rk45 { atol = .., rtol = .. } }`.
    #[arg(long)]
    pub integrator: Option<crate::args::types::IntegratorArg>,
}

fn is_false(b: &bool) -> bool { !*b }

/// `--output-every` / `--no-flows` / `--columns` — the trajectory output view
/// (cadence + which columns are written). One struct backs both the `simulate`
/// CLI flags (flattened into `SimulateArgs`) and `batch.toml`'s `[output]`
/// section (deserialized) — a single definition, both front doors (the shared
/// clap+serde pattern; see gh#241). Only `simulate` and `batch.toml` write
/// trajectories, so the view applies there; `fit.toml` has no trajectory output.
///
/// `every` overrides the model's `output { every }` schedule, so it rides the
/// model digest and re-keys only runs that use it. `no_flows` / `columns`
/// change which columns are written to the content-addressed leaf, so they ride
/// the `config`-level identity (`runid::inputs::SimConfig`): a non-default view
/// is a distinct, reproducible artifact. `skip_serializing_if` keeps defaults
/// out of the fit-identity (canonical-JSON) hash so existing fits don't re-key.
#[derive(Args, Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputView {
    /// Emit one trajectory row every N time-units, overriding the model's
    /// `output { every }`. A plain number in the model's `time_unit` (like
    /// `--dt`); e.g. `--output-every 7` on a daily model writes weekly rows.
    #[arg(long = "output-every", value_name = "N")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub every: Option<f64>,

    /// Drop every `flow_*` column from the trajectory output — most useful for
    /// stratified/spatial models where inter-stratum flow columns dominate.
    #[arg(long = "no-flows", default_value_t = false)]
    #[serde(default, skip_serializing_if = "is_false")]
    pub no_flows: bool,

    /// Restrict trajectory columns to this comma-separated allow-list of output
    /// column names (compartments and/or `flow_<name>`). Empty = all columns;
    /// emitted order follows the model, not this list.
    #[arg(long = "columns", value_delimiter = ',', value_name = "COL,...")]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<String>,
}

/// Core inference knobs shared by pfilter / if2 / profile
#[derive(Args, Clone)]
pub struct InferenceCore {
    /// Number of particles
    #[arg(long)]
    pub particles: usize,

    /// Step size
    #[arg(long, default_value_t = 1.0)]
    pub dt: f64,

    /// RNG seed
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// Rayon thread count (0 = all available cores)
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    pub parallel: usize,

    /// gh#audit-C6 / S1. Restore the legacy silent-zero behaviour for
    /// numerical-collapse paths (Div-by-zero, Pow → NaN, Sqrt of
    /// negative, etc.) in rate evaluation. Default: hard error
    /// (SimError::NumericalCollapse). Use only when a model
    /// legitimately needs the legacy behaviour, e.g. spatial models
    /// where a stratum-empty divisor is explicitly meant to produce
    /// zero rate. Counters under EvalStats still increment under
    /// either mode so you can see how often the path fired.
    #[arg(long, default_value_t = false)]
    pub allow_degenerate_rates: bool,

    /// Time-column format for `--data`: `auto` (detect numeric-vs-date over
    /// the whole column), `numeric` (force `f64`), or `date` (force ISO
    /// dates, requires the model's `origin`). Dated columns convert via
    /// the model `origin` + `time_unit` (2026-05-22 calendar-time).
    #[arg(long, default_value = "auto")]
    pub time_format: crate::caltime_load::TimeFormat,

    /// gh#241. Deterministic per-call compute budget: the maximum cumulative
    /// particle-substep count one filter evaluation may execute before bailing
    /// (`PFIterationBudget`). Unset ⇒ the engine default (1e10), which never
    /// false-trips a legitimate fit. A reproducible, machine-independent
    /// budget — it replaced `--pf-wallclock-timeout`, a wall-clock timeout that
    /// set a process env var and made a fit's log-likelihood depend on hardware.
    #[arg(long, value_name = "N")]
    pub pf_max_substeps: Option<u64>,
}

/// Stream selection for the inference commands. The projection and
/// likelihood for every stream come from the model's `observations { }` block
/// (the modern observation system, as `fit run` uses) — there is no `--flow` /
/// `--obs-model` projection override. `--obs` only selects WHICH declared
/// stream (family) a single positional `--data FILE` binds to.
#[derive(Args, Clone, Default)]
pub struct StreamSelection {
    /// Observation stream/family name a bare `--data FILE` binds to
    /// (required when the model declares more than one).
    #[arg(long)]
    pub obs: Option<String>,
}

// ─── simulate ─────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Basic simulation. The run is written to the content-addressed store
  # under ./results (reruns with identical inputs are instant); read it
  # back with `camdl cat <id>` / browse with `camdl list`.
  camdl simulate sir.camdl --params p.toml --seed 42
  camdl list        # browse cached runs

  # Also write a plain TSV mirror to a file
  camdl simulate sir.camdl --params p.toml --seed 42 -o traj.tsv

  # Stream the trajectory to stdout (no store write) for piping
  camdl simulate sir.camdl --params p.toml --seed 42 --stdout | head

  # Named scenario
  camdl simulate sir.camdl --params p.toml --scenario with_sia --seed 42

  # Generate synthetic observations alongside the trajectory
  camdl simulate sir.camdl --params p.toml --obs cases.tsv --seed 42

  # Multi-seed ensemble
  camdl simulate sir.camdl --params p.toml --seeds 1:100

  # Posterior predictive check from a fit's draws
  camdl simulate sir.camdl --draws draws.tsv --replicates 10 --obs ppc.tsv
"))]
pub struct SimulateArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    /// Override the run's horizon (`simulation.t_end`), gh#626. SPEC is a
    /// model-time number ("120"), a calendar date (date("YYYY-MM-DD") or
    /// bare YYYY-MM-DD; needs a model origin), or an observation anchor
    /// with optional offset: "last_obs", "last_obs + 8 weeks",
    /// "first_obs - 1 week". Anchored forms require --fit (the fit's
    /// [data.observations] supplies the observed times; last_obs = the max
    /// observation time over the bound streams). months/years are fixed
    /// spans (30.4375 d / 365.2425 d), not calendar arithmetic. A scenario
    /// that declares its own different horizon is a hard error.
    #[arg(long, value_name = "SPEC")]
    pub to: Option<String>,

    /// Start the run from an inferred state at the last observation time
    /// instead of the model's `init {}` block — the forecast workflow. The
    /// origin time becomes the run's `t_start`; pair it with
    /// `--to "last_obs + 8 weeks"` for the horizon. chain_binomial only.
    /// Two sources:
    ///   FILE — a `camdl pfilter --save-final-state` TSV, drawn from
    ///     p(x_T | y_{1:T}) at the filter's ONE θ (gh#641). Replicate i
    ///     restores row i, so `--replicates` must equal the row count, and
    ///     `--draws` is refused (unrelated θ crossed with these states is an
    ///     incoherent (θ, x_T) product).
    ///   fit — the paired (θ_i, X_i(T)) posterior of the `--fit` run: draw i
    ///     restores its OWN terminal latent state under its OWN θ (gh#697).
    ///     Requires `--draws posterior`; forecasts over the subset of draws
    ///     that have a saved latent path, and reports that count.
    #[arg(long, value_name = "FILE|fit")]
    pub init_state: Option<String>,

    /// gh#audit-C6 / S1. See InferenceCore.allow_degenerate_rates.
    /// Forward sim is the most likely user of this flag — if a model
    /// has a known empty-stratum-divisor and the user wants to keep
    /// the legacy zero-rate behaviour rather than fix the model.
    #[arg(long, default_value_t = false)]
    pub allow_degenerate_rates: bool,

    #[command(flatten)]
    pub model_overrides: ModelOverrides,

    /// Parameter vector file (may be repeated), e.g. --param-vec beta=FILE
    #[arg(long, value_name = "PREFIX=FILE")]
    pub param_vec: Vec<ParamVecSpec>,

    /// Named scenarios (may be repeated; conflicts with --enable/--disable)
    #[arg(long = "scenario", conflicts_with_all = ["enable", "disable"])]
    pub scenarios: Vec<String>,

    /// Enable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenarios")]
    pub enable: Vec<String>,

    /// Disable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenarios")]
    pub disable: Vec<String>,

    #[command(flatten)]
    pub backend: SimBackend,

    /// Trajectory output view: --output-every / --no-flows / --columns.
    #[command(flatten)]
    pub output_view: OutputView,

    /// RNG seed for a single run (conflicts with --seeds)
    #[arg(long, default_value_t = 1, conflicts_with = "seeds",
          env = "CAMDL_SEED")]
    pub seed: u64,

    /// Multiple seeds: range (1:100) or list (1,2,42); conflicts with --replicates
    #[arg(long, conflicts_with_all = ["replicates"])]
    pub seeds: Option<SeedSpec>,

    /// Stochastic replicates per parameter point (conflicts with --seeds)
    #[arg(long, conflicts_with = "seeds")]
    pub replicates: Option<usize>,

    /// Parameter draw source: path to a params TSV, "uniform", "prior", or
    /// "posterior". "posterior" reads a completed fit's canonical post-warm-up
    /// draws cloud — requires `--fit <fit results dir>`.
    #[arg(long)]
    pub draws: Option<String>,

    /// Companion for `--draws`. With `--draws prior`, a fit.toml supplying
    /// priors. With `--draws posterior`, the fit results directory to read the
    /// posterior draws from. With `--draws <file.tsv>`, a fit.toml (or results
    /// dir) whose `[fixed]` block backfills parameters absent from the file's
    /// columns — never overwriting a column the file provides (#273).
    /// Also consulted (without --draws) by an anchored `--to`: the fit's
    /// [data.observations] supplies the observed times the anchor resolves
    /// against — a manual check requires --fit to come with --draws and/or
    /// an anchored --to (gh#626).
    #[arg(long)]
    pub fit: Option<PathBuf>,

    /// Number of parameter draws. For --draws uniform/prior: how many to
    /// generate. For --draws posterior: a strided subsample cap across the
    /// whole cloud (default 200, matching `fit predict` — never silently
    /// replay a full 60k-draw posterior; gh#630).
    #[arg(short = 'n', long)]
    pub n_draws: Option<usize>,

    /// Write the sampled per-draw parameter vectors to this TSV (gh#157).
    /// One row per draw, one column per parameter — the same column-per-
    /// param format `--draws PATH` reads back, so the file round-trips.
    /// Only written when given; the content-addressed store leaves are
    /// unaffected.
    #[arg(long, value_name = "PATH", requires = "draws")]
    pub draws_out: Option<PathBuf>,

    /// Write a plain-TSV trajectory mirror to this file, IN ADDITION to the
    /// content-addressed store leaf (the default system of record). Without
    /// `-o` the trajectory is not written to a loose file — read it back with
    /// `camdl cat <id>`. Use `--stdout` to stream instead of storing.
    #[arg(short, long, env = "CAMDL_OUTPUT")]
    pub output: Option<PathBuf>,

    /// Stream the trajectory TSV to stdout and DO NOT write the store leaf or
    /// the `cached:` banner — the escape hatch for piping into another tool
    /// (`camdl simulate … --stdout | …`). Single-cell only: it conflicts with
    /// the store-backed ensemble knobs (--seeds / --replicates / --draws) and
    /// with `-o`/`--obs*`, which mirror the store this flag opts out of.
    #[arg(long, conflicts_with_all = [
        "output", "obs", "obs_dir", "obs_only", "obs_only_dir",
        "seeds", "replicates", "draws",
    ])]
    pub stdout: bool,

    /// Override the cadence at which synthetic observations are EMITTED
    /// (`emit_schedule`), gh#656 — so one model serves a daily and a weekly
    /// emission without editing its source. N is a plain number in the model's
    /// own `time_unit` (not the DSL `8 'weeks` spelling, which is a
    /// shell-quoting hazard): `--emit-every 7` sets every stream, and
    /// `--emit-every NAME=7` (repeatable) sets the stream with that
    /// observation-block label. The two forms are mutually exclusive. Only a
    /// recurring (`every N`) schedule can be overridden — a stream declaring
    /// `emit_schedule = at [...]` is refused by name. This changes emitted
    /// output only; it never enters a likelihood.
    #[arg(long = "emit-every", value_name = "N | NAME=N")]
    pub emit_every: Vec<String>,

    /// Write synthetic observations to a single TSV (all streams)
    #[arg(long, conflicts_with_all = ["obs_dir", "obs_only"])]
    pub obs: Option<PathBuf>,

    /// Write one TSV per observation stream to a directory
    #[arg(long, conflicts_with_all = ["obs", "obs_only"])]
    pub obs_dir: Option<PathBuf>,

    /// Like --obs but suppress trajectory output
    #[arg(long, conflicts_with_all = ["obs", "obs_dir", "output", "obs_only_dir"])]
    pub obs_only: Option<PathBuf>,

    /// Like --obs-dir (one TSV per stream) but suppress trajectory output.
    /// The multi-cadence-safe sibling of --obs-only (run-spec §3.1.1,
    /// ObsOutput::OnlyDir).
    #[arg(long, conflicts_with_all = ["obs", "obs_dir", "obs_only", "output"])]
    pub obs_only_dir: Option<PathBuf>,

    /// Add a calendar `date` column (rendered from the model `origin` +
    /// `time_unit`) alongside the numeric `t`/`time` column in trajectory
    /// and observation output. Numeric time stays the canonical, diff-stable
    /// default; `--dates` is purely additive (2026-05-22 calendar-time §6.7).
    /// No-op with a clear error if the model declares no `origin`.
    #[arg(long)]
    pub dates: bool,

    /// Emit the model's `quantities {}` block to this directory as
    /// `<dir>/quantities/<name>.tsv` + `<dir>/quantities.json`. A single
    /// fixed-params run writes a bare `value` per leaf (point mode); a
    /// `--draws`/`--replicates`/`--seeds` run writes banded quantiles. Quantities
    /// are a regenerated sidecar — never part of the content-addressed run
    /// identity. Without this flag, a model that declares quantities prints a
    /// note and skips them (not a hard error).
    #[arg(long, value_name = "DIR")]
    pub quantities_out: Option<PathBuf>,

    /// Report with the `quantities {}` block in FILE instead of the model's own.
    /// FILE is an ordinary `.camdl` file containing ONLY a `quantities {}`
    /// block; it is compiled against this model, and a name the model does not
    /// declare is an error naming both the name and FILE. It REPLACES the
    /// model's block — never merges. The tables land in
    /// `<dir>/quantities-<key>/`, keyed by FILE's contents, so two vocabularies
    /// over one model produce two tables rather than overwriting each other.
    /// The trajectory and the run's identity are unaffected (proposal
    /// 2026-08-19).
    #[arg(long, value_name = "FILE", requires = "quantities_out")]
    pub quantities: Option<PathBuf>,

    /// Print resolved run plan without simulating
    #[arg(long)]
    pub dry_run: bool,

    /// Accepted for compatibility — content-addressed storage is now the
    /// default for every `simulate` run (one leaf per cell under
    /// `--output-dir`). `--output`/`--obs` mirror the store; they no longer
    /// replace it. The flag is a no-op kept so existing invocations and
    /// scripts that pass `--cas` keep working.
    #[arg(long)]
    pub cas: bool,

    /// Root directory for --cas output
    #[arg(long, default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub output_dir: PathBuf,

    /// Concurrent simulation runs
    #[arg(long, env = "CAMDL_PARALLEL")]
    pub parallel: Option<usize>,

    /// Re-run even if cached output already exists
    #[arg(long)]
    pub force: bool,

    /// User-supplied display label for this simulate run. Validated
    /// against `^[a-zA-Z0-9 ,._-]{1,64}$` after trim. Surfaced in
    /// `camdl list` and `camdl show`. With `--seeds`, the label
    /// applies to each per-seed run.
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,

    /// Record the Layer-1 lineage **event log** (2026-05-20 three-layer
    /// architecture). Requires a model with `#[lineage]` annotations and a
    /// backend that declares the LINEAGES capability (Gillespie /
    /// chain-binomial). The event log is stored as the content-addressed
    /// `event_log.tsv` artifact in the run's CAS leaf, alongside `traj.tsv`
    /// (read it with `camdl cat <id> --stream event_log.tsv`); it is
    /// identity-free — realize it into a line list with `camdl lineage
    /// realize`. Pass a PATH to also mirror the log to that file (format from
    /// --tsv/--format/extension); bare `--event-log` (or `auto`) records only
    /// into the leaf. Single-run only — conflicts with --seeds / --replicates
    /// / --draws.
    #[arg(long, value_name = "FILE", num_args = 0..=1, default_missing_value = "auto",
          conflicts_with_all = ["seeds", "replicates", "draws"])]
    pub event_log: Option<PathBuf>,

    /// Event-log format: `parquet` (default, production) or `tsv`
    /// (dependency-free, debug). With --event-log.
    #[arg(long, value_name = "FMT", requires = "event_log", conflicts_with = "tsv")]
    pub format: Option<String>,

    /// Shorthand for `--format tsv` (with --event-log).
    #[arg(long, requires = "event_log")]
    pub tsv: bool,

    /// Mirror the reactive firing log to this file, IN ADDITION to the
    /// canonical `reactive_log.tsv` artifact in the run's CAS leaf. The leaf
    /// log is always present when a reactive policy was active (read it with
    /// `camdl cat <id> --stream reactive_log.tsv`); this flag is a convenience
    /// mirror, symmetric with `-o` for the trajectory — the leaf stays the
    /// system of record. Single-run only.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["seeds", "replicates", "draws"])]
    pub reactive_log: Option<PathBuf>,
}

// ─── batch ────────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Run a batch sweep
  camdl batch run sweep.toml --parallel 8

  # Dry-run: show the resolved sweep grid without simulating
  camdl batch run sweep.toml --dry-run

  # Force rerun, ignoring cached outputs
  camdl batch run sweep.toml --force
"))]
pub struct BatchArgs {
    /// Batch TOML manifest file
    pub file: PathBuf,

    /// Override output_dir from the manifest
    #[arg(long, env = "CAMDL_OUTPUT_DIR")]
    pub output_dir: Option<PathBuf>,

    /// Override parallel thread count
    #[arg(long, env = "CAMDL_PARALLEL")]
    pub parallel: Option<usize>,

    /// Print resolved sweep grid without running
    #[arg(long)]
    pub dry_run: bool,

    /// Re-run even if output exists
    #[arg(long)]
    pub force: bool,

    /// gh#audit-C6. See InferenceCore.allow_degenerate_rates. Batch
    /// runs that orchestrate many simulate-style invocations need
    /// the same opt-in for legacy silent-zero behaviour on rate
    /// collapse paths.
    #[arg(long, default_value_t = false)]
    pub allow_degenerate_rates: bool,
}

/// `camdl batch status FILE`
#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Count completed vs pending runs for a sweep
  camdl batch status sweep.toml

  # Watch a long-running sweep from another shell
  watch -n 5 camdl batch status sweep.toml
"))]
pub struct BatchStatusArgs {
    /// Batch TOML manifest file
    pub file: PathBuf,
}

// ─── fit ──────────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Run the full inference pipeline declared in fit.toml
  camdl fit run fit.toml --seed 1

  # Long fits: capture progress + diagnostics to a log file
  camdl fit run fit.toml --seed 1 --progress plain 2>&1 | tee fit.log

  # Force rerun even if cached results match
  camdl fit run fit.toml --seed 1 --force

  # Override the stage's init mode via the new `--init` flag (renamed
  # from --init-method per 2026-05-25 CLI UX rev 2)
  camdl fit run fit.toml --stage scout --init lhs

  # Warm-start every chain from a prior fit's MLE (replaces the
  # removed --starts-from <dir> flag)
  camdl fit run fit.toml --stage refine --init from_mle --mle fits/scout/

  # Warm-start from a posterior draws TSV / fit-results directory
  camdl fit run fit.toml --stage pgas --init from_posterior \\
      --posterior fits/scout/draws.tsv

  # Warm-start from a hand-written flat params TOML
  camdl fit run fit.toml --stage refine --init from_params --params truth.toml

Notes:
  - PGAS/PMMH fits can resume a partial run with `--resume`.
  - Output goes under `<root>/fits/<stem>-<hash>/`; resolve with
    `camdl fit where fit.toml`.

Priors (Bayesian stages: PGAS, PMMH):
  Each estimated parameter must have a prior available from one of
  three sources, resolved in this order:

    1. fit toml: `[estimate.<param>] prior = { <dist> = { ... } }`
       — per-fit override, wins over the model-IR fallback.
    2. model file: `~ <dist>(...)` syntax in the `.camdl` parameter
       block — single source of truth for stable priors across N
       fit tomls.
    3. fit toml: explicit `prior = { flat = {} }` — opt-in to flat
       (improper uniform) priors. Use this if you genuinely want
       the chain to target the unconditioned likelihood
       (scaled-likelihood posterior). Implicit fallback to flat is
       a hard error — `fit run`'s chain is treated authoritatively
       downstream, so silent demotion is not allowed.

  Per-parameter provenance is recorded in `run.json` under
  `resolved_priors` (sources: fit_toml, model_ir, flat_explicit).
  See `docs/inference.md` § \"Priors and precedence\" for the full
  spec including the asymmetry with `camdl profile`.
"))]
pub struct FitRunArgs {
    /// Fit configuration file (v2 TOML)
    pub config: PathBuf,

    /// Run only this stage by name
    #[arg(long)]
    pub stage: Option<String>,

    /// Max worker threads for the Rayon pool that runs the chains and, within
    /// each chain, the particle filter. `0` (the default) uses all logical
    /// cores. Results are bit-identical regardless of this value — it caps the
    /// thread budget / CPU footprint, not the numerics. Mirrors `--parallel`
    /// on pfilter/profile/survey/batch; also honored via `CAMDL_PARALLEL`.
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    pub parallel: usize,

    /// RNG seed (default: 1)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Re-run and overwrite stale cache
    #[arg(long)]
    pub force: bool,

    /// Resume a previously-completed PGAS or PMMH stage from a base run
    /// addressed by `<run_id prefix>` or a leaf path. gh#147 (M3.2): the
    /// base leaf is read **read-only**; the resumed run is written to a new
    /// content-addressed leaf keyed on the new extension dimension (PGAS
    /// `sweeps` / PMMH `iterations`) with a dep on the base — a distinct
    /// deterministic artifact, not bit-identical to an uninterrupted fit of
    /// the same length. Requires --stage. Conflicts with --force.
    #[arg(long, value_name = "BASE_REF", requires = "stage", conflicts_with = "force")]
    pub resume: Option<String>,

    /// Cartesian sweep over a fixed parameter (may repeat).
    /// SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC")]
    pub sweep: Vec<SweepSpec>,

    /// Override the cadence at which `[synthetic]` data is GENERATED
    /// (`emit_schedule`), gh#656. Same grammar as `camdl simulate
    /// --emit-every`: a plain number in the model's own `time_unit` for every
    /// stream, or `NAME=N` (repeatable) for one stream by its
    /// observation-block label. This is the only fit path the emission cadence
    /// reaches — a fit against real data scores at its data files' own times
    /// and never consults `emit_schedule` — so the flag is REFUSED on a fit
    /// with no `[synthetic]` block rather than silently doing nothing. It
    /// changes the generated data, so the fit re-keys.
    #[arg(long = "emit-every", value_name = "N | NAME=N")]
    pub emit_every: Vec<String>,

    /// Proceed even if prior scout stage failed convergence gate
    #[arg(long)]
    pub allow_nonconverged_scout: bool,

    /// Override [stages.<stage>.gate] decibans_thresh (the inter-chain
    /// log-likelihood-spread floor, in decibans). Requires --stage.
    #[arg(long, value_name = "DB", requires = "stage")]
    pub decibans_thresh: Option<f64>,

    /// Override [stages.<stage>.init] for chain starts (gh#42, gh#83).
    /// See `--help` for the INIT MODES block. Requires --stage so
    /// scout and refine can be set independently. Has no effect when
    /// the stage uses `init_mle = "<prior_stage>"` — those chains
    /// start from the prior MLE regardless. Renamed from
    /// `--init-method` per 2026-05-25 CLI UX rev 2.
    #[arg(long, value_name = "MODE", value_enum, requires = "stage",
          long_help = INIT_LONG_ABOUT)]
    pub init: Option<InitModeTag>,

    /// Companion path for `--init from_posterior`. Accepts a posterior
    /// draws TSV directly or a fit-results directory.
    #[arg(long, value_name = "PATH", requires = "stage")]
    pub posterior: Option<PathBuf>,

    /// Companion path for `--init from_mle`. Accepts an MLE TOML
    /// directly or a fit-results directory (auto-resolves
    /// `<dir>/mle.toml`, then `<dir>/final_params.toml`). Replaces
    /// the removed `--starts-from <dir>` flag.
    #[arg(long, value_name = "PATH", requires = "stage")]
    pub mle: Option<PathBuf>,

    /// Companion path for `--init from_params`. Hand-written flat
    /// params TOML; top-level keys are parameter names.
    #[arg(long = "params", value_name = "TOML", requires = "stage")]
    pub init_params: Option<PathBuf>,

    // ── Removed-flag traps (M-1 break per 2026-05-25 proposal) ────────
    //
    // These flags were renamed/removed; the dispatch site emits an
    // actionable error citing the replacement. Per CLAUDE.md alpha
    // posture, no back-compat shims — these exist purely so the error
    // message points the user to the right replacement.
    #[arg(long = "init-method", value_name = "MODE", hide = true)]
    pub _removed_init_method: Option<String>,
    #[arg(long = "starts-from", value_name = "DIR_OR_HASH", hide = true)]
    pub _removed_starts_from: Option<String>,

    /// Survey CAS directory consumed when `--init survey_top_k` is in
    /// effect (gh#51). Must contain `run.json` (kind = survey) and
    /// `landscape.tsv`. Overrides any `survey_path` set on the stage
    /// in fit.toml. Requires --stage; ignored unless the effective
    /// init mode is `survey_top_k`.
    #[arg(long, value_name = "DIR", requires = "stage")]
    pub survey_path: Option<std::path::PathBuf>,

    /// Top-K count for `--init survey_top_k` (gh#51). Defaults to the
    /// stage's `chains` when omitted; in v1 must equal `chains`
    /// (strict K=chains; K > chains stratified sub-sampling deferred
    /// to v2). Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub survey_top_k: Option<usize>,

    /// User-supplied display label for this fit (1–64 chars after
    /// trim; allowed: letters, digits, spaces, commas, dot,
    /// underscore, hyphen). Surfaced in `camdl fit list` and
    /// `camdl fit table` to disambiguate iterations of a model that
    /// share the same fit-stem. Examples: --label "narrow R0, take 1",
    /// --label "iota free", --label "log_normal R0 prior".
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,

    /// Burn-in / conditioning window (gh#134). Mirrors the top-level
    /// `condition_from` key in fit.toml; the CLI value overrides it. The CLI
    /// carries ONE value, so it sets the all-streams DEFAULT (the `All` form);
    /// per-stream shadows (`[condition_from] <label> = ...`) are toml-only. Each
    /// incidence stream warms up over `[t_start, condition_from)` (full process
    /// noise, interventions, forcings) but scores nothing there, then the
    /// incidence accumulator is reset at the boundary so the first scored bin is
    /// `(condition_from, first_obs]`. Accepts a model-time number
    /// (`--condition-from 14`), a calendar date
    /// (`--condition-from 'date("2020-02-01")'` or `--condition-from 2020-02-01`),
    /// or a relative offset (`--condition-from "first_obs - 1 week"`). A set
    /// value re-keys the fit (it is part of the fit identity); unset leaves
    /// the fit bit-identical.
    #[arg(long, value_name = "WHEN")]
    pub condition_from: Option<String>,

    // ── Richardson dt-convergence check (gh#52) ─────────────────────

    /// Skip the post-fit Richardson dt-convergence check at θ̂.
    /// Default: the check runs on every IF2 stage (PF likelihood) and
    /// every ODE inference stage — nl-sbplx, nl-bobyqa, and mh
    /// (deterministic likelihood). Use this for CI smoke fits or
    /// known-converged-dt rerenders where the audit cost is unwelcome.
    /// Requires --stage: the dt-check result is stored in the leaf, so
    /// the override is keyed into that stage's identity (gh#540 seam;
    /// gh#726 for the mh/nl-* dt_check field).
    #[arg(long, requires = "stage")]
    pub no_dt_check: bool,

    /// Drop the dt-check warning threshold to the strict default
    /// (0.5 nats for chain_binomial, 0.1 nats for ode_rk4). Targets
    /// research-quality fits where sub-nat differences matter for
    /// paper-grade conclusions; the routine default (2.0 / 0.5)
    /// allows more give before flagging.
    /// Requires --stage: the threshold it selects is stored in the leaf
    /// (fit_state.toml.dt_check), so it is resolved into that stage's
    /// dt_check.threshold_nats and keyed into its identity (gh#730). A stage
    /// that declares its own threshold_nats is unaffected.
    #[arg(long, requires = "stage")]
    pub dt_check_strict: bool,

    /// Binomial sampler for a PGAS stage's chain-binomial draws: `btpe`
    /// (default) or `btrs` (Hörmann 1993; faster, gh#747).
    ///
    /// Requires --stage, and for the same reason as the flags above: the two
    /// samplers are NOT bit-compatible — a different rejection scheme accepts
    /// different draws from the same stream — so this is resolved into the
    /// stage's `binomial` field and keyed into its identity. Two runs differing
    /// only in this flag get different addresses and cannot be served from one
    /// another's leaf. It is a flag rather than an environment variable for
    /// exactly that reason (gh#241 removed the last env-var input rather than
    /// hash it).
    #[arg(long, requires = "stage", value_name = "SAMPLER",
          value_parser = |v: &str| v.parse::<sim::rng::BinomialAlgorithm>())]
    pub binomial: Option<sim::rng::BinomialAlgorithm>,

    /// Override `n_halvings` on the dt-check (gh#52). Default 2
    /// (evaluates at dt_fit, dt_fit/2, dt_fit/4 — 7× the
    /// loglik-eval cost). Use 3 for ambiguous cases at 15×.
    /// Requires --stage: the ladder result is stored in the leaf, so
    /// the override is keyed into that stage's identity (gh#540 seam;
    /// gh#726 for the mh/nl-* dt_check field).
    #[arg(long, value_name = "N", requires = "stage")]
    pub dt_check_halvings: Option<usize>,

    // ── IF2-specific algorithm overrides (require --stage) ───────────

    /// Override [stages.<stage>.cooling_target_iters]. Iterations
    /// over which the cooling fraction is reached (pomp's
    /// cooling.fraction.50 default). Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub cooling_target_iters: Option<usize>,

    // ── PGAS-specific algorithm overrides (require --stage) ──────────

    /// Override [stages.<stage>.tempering]. Comma-separated β values
    /// for parallel tempering ladder. First value MUST be 1.0.
    /// Example: `--tempering "1.0,0.7,0.4,0.15"`. Requires --stage.
    #[arg(long, value_name = "B1,B2,...", requires = "stage",
          value_parser = parse_f64_list)]
    pub tempering: Option<Vec<f64>>,

    /// Override [stages.<stage>.max_tree_depth] (NUTS depth ceiling).
    /// Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub max_tree_depth: Option<usize>,

    /// Override [stages.<stage>.trajectory_warmup] (CSMC-only sweeps
    /// before parameter updates begin). Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub trajectory_warmup: Option<usize>,

    /// Override [stages.<stage>.csmc_sweeps_per_nuts] (CSMC trajectory
    /// updates per parameter update). Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub csmc_sweeps_per_nuts: Option<usize>,

    /// Override [stages.<stage>.n_trajectories] (posterior trajectories
    /// saved). Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub n_trajectories: Option<usize>,

    /// Override [stages.<stage>.dense_mass] to false (use diagonal NUTS
    /// mass matrix). One-way: edit TOML to flip back. Requires --stage.
    #[arg(long, requires = "stage")]
    pub diagonal_mass: bool,

    /// Override [stages.<stage>.use_nuts] to false (fall back to
    /// MH-within-Gibbs for the θ|X update). One-way: edit TOML to
    /// flip back. Requires --stage.
    #[arg(long, requires = "stage")]
    pub no_nuts: bool,

    /// Override [stages.<stage>.ancestor_sampling] to false: run the CSMC
    /// sweep as plain particle Gibbs, without the ancestor-sampling move.
    /// A diagnostic control (what does AS contribute, and what does its
    /// density pass cost?); changes the sampled draws, so the run stores
    /// under its own address. One-way: edit TOML to flip back. Requires
    /// --stage.
    #[arg(long, requires = "stage")]
    pub no_ancestor_sampling: bool,

    // ── PMMH-specific algorithm overrides (require --stage) ──────────

    /// Override [stages.<stage>.adapt] to false (lock proposal SDs;
    /// disables Haario-style adaptation). One-way: edit TOML to flip
    /// back. Requires --stage.
    #[arg(long, requires = "stage")]
    pub no_adapt: bool,

    /// Override [stages.<stage>.adapt_start] (MCMC step at which
    /// proposal-SD adaptation begins). Requires --stage.
    #[arg(long, value_name = "N", requires = "stage")]
    pub adapt_start: Option<usize>,

    /// Override [stages.<stage>.rho] (Crank-Nicolson correlation for
    /// correlated pseudo-marginal MCMC). Set to a value in [0, 1).
    /// Requires --stage.
    #[arg(long, value_name = "F", requires = "stage")]
    pub rho: Option<f64>,

    // ── PFilter-specific algorithm overrides (require --stage) ───────

    /// Override [stages.<stage>.record_ancestry] to true (record
    /// ancestor indices for smoothing-path reconstruction). Requires
    /// --stage.
    #[arg(long, requires = "stage")]
    pub record_ancestry: bool,

    /// Override [stages.<stage>.record_prequential] to true (record
    /// per-step predictive samples for `camdl compare`). Requires
    /// --stage.
    #[arg(long, requires = "stage")]
    pub record_prequential: bool,
}

/// Parser for `--tempering "1.0,0.7,0.4"` and similar comma-list
/// f64 args. Empty string is rejected; spaces around commas are
/// trimmed.
fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    if s.trim().is_empty() {
        return Err("expected comma-separated f64 list".into());
    }
    s.split(',')
        .map(|p| p.trim().parse::<f64>().map_err(|e| format!("'{}': {}", p, e)))
        .collect()
}

/// Output format for `camdl fit summary`. `text` is the default
/// rendered terminal block with ANSI colour; `json` is a versioned
/// machine-readable schema (`schema.version`); `md` and `latex` are
/// document-friendly outputs for the book pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FitSummaryFormat {
    Text,
    Json,
    Md,
    Latex,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Render summary for a completed fit
  camdl fit summary fit/he2010

  # Just one stage
  camdl fit summary fit/he2010 --stage scout

  # Machine-readable JSON for the book pipeline
  camdl fit summary fit/he2010 --format json > summary.json

  # Markdown for embedding in a chapter
  camdl fit summary fit/he2010 --format md

  # Just the winner θ̂ as a flat params TOML, pipeable into
  # `camdl pfilter --params`:
  camdl fit summary fit/he2010 --params-only --stage validate \\
    | camdl pfilter --params /dev/stdin model.camdl --data cases.tsv

  # Disable colour (useful for redirecting to a file)
  camdl fit summary fit/he2010 --no-color

  # Strict mode for CI: exit non-zero on provenance mismatch.
  # Auto-enabled when CI=true or CI=1 in the environment.
  camdl fit summary fit/he2010 --strict
"))]
pub struct FitSummaryArgs {
    /// The fit, by handle: `@label`, a fit-level hash prefix, a fit results
    /// directory (e.g. `results/fits/he2010-…`), or a `fit.toml` config.
    pub fit: String,

    /// Render only one stage's stanza
    #[arg(long, value_name = "STAGE")]
    pub stage: Option<String>,

    /// Output format. `text` (default) emits the terminal block;
    /// `json` emits a versioned `schema.version: 1` document; `md`
    /// emits GitHub-flavoured Markdown; `latex` emits `\begin{tabular}`
    /// blocks per section.
    #[arg(long, value_enum, default_value_t = FitSummaryFormat::Text,
          conflicts_with = "params_only")]
    pub format: FitSummaryFormat,

    /// Print only the winner θ̂ as a flat params TOML (no metadata,
    /// no provenance, no headings — pipeable into `camdl pfilter
    /// --params <(camdl fit summary --params-only ...)`). Combine
    /// with `--stage <stage>` to pick which stage's winner to emit;
    /// without `--stage`, prints the terminal stage in the pipeline
    /// order (validate → refine → scout, whichever is present).
    #[arg(long, conflicts_with = "format")]
    pub params_only: bool,

    /// Disable ANSI colour even on a TTY. Honours `NO_COLOR` env var
    /// regardless of this flag.
    #[arg(long)]
    pub no_color: bool,

    /// Exit non-zero on provenance mismatch (final_params.toml ↔
    /// mle_params.toml disagrees, fit_state.toml winner doesn't match
    /// final_params.toml, stale camdl version, etc.). Auto-enabled
    /// when `CI=true` or `CI=1` is set in the environment, matching
    /// cargo / pytest convention. See proposal §1, §6.
    #[arg(long)]
    pub strict: bool,

    /// Recompute the posterior diagnostics (R̂ / ESS / mean) over a SUBSET of
    /// the MCMC chains, dropping the named 1-based chain ids (comma-separated,
    /// e.g. `--exclude-chains 3,5`). A view only — nothing on disk changes. The
    /// header then reads `chains: K of N (excluded 3,5)`. Post-hoc exclusion
    /// BIASES the posterior toward the retained mode and always prints a
    /// warning; a chain id not in the fit, or excluding every chain, is a hard
    /// error. Incompatible with `--params-only` (a single winner θ̂ is not a
    /// chain average, so there is nothing to subset).
    #[arg(long, value_name = "IDS", conflicts_with = "params_only")]
    pub exclude_chains: Option<String>,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Write the free-forward posterior predictive for a completed fit
  camdl fit predict --fit fit.toml

  # Just one stream (logical name or an expanded leaf name)
  camdl fit predict --fit fit.toml --stream onset

  # Point at a run directory directly (instead of the config)
  camdl fit predict results/fits/sle-8a3f12b4/

  # Look at a disagreement rhat_mean flagged: one band per chain, beside
  # the pooled one, under a leading `chain` column
  camdl fit predict --fit fit.toml --by-chain

Outputs, under the run directory:
  predictive/<stream>.tsv   scenario | time | <dims...> | horizon | treatment
                            | fit_rhat_max | fit_ess_min | rhat_mean | ess_mean
                            | rhat_pred | ess_pred | n_draws | q05..q95
  observed/<stream>.tsv     time | <dims...> | value
Read both, join on (time, <dims>), plot observed over the predictive ribbon.
Under --exclude-chains the bands land in predictive-excl<ids>/ instead, so a
chain subset never overwrites the pooled artifact; `camdl show <fit>` lists
every address the fit holds.

Convergence columns — two different questions, do not mix them up:
  fit_rhat_max,        the fit's worst parameter, copied from the producing
  fit_ess_min          stage. Provenance about the fit, not about this row:
                       the same number repeats down the whole file.
  rhat_mean, ess_mean  this row's latent expected value, across chains.
                       \"Do the chains agree about the expected trajectory
                       here?\"  ← decide on this one.
  rhat_pred, ess_pred  this row's predictive draws, across chains. \"Do the
                       chains give the same predictive distribution?\"

Why the two per-row numbers differ, and why it matters: a predictive draw
carries observation noise, and that noise lands in the within-chain variance.
Where the noise is comparable to the between-chain disagreement it swamps the
numerator and rhat_pred is pulled toward 1 however much the chains disagree —
worst on overdispersed counts, i.e. most mechanistic models. Chains whose
8-week forecasts span 93 to 372 cases/day can still show rhat_pred near 1.
rhat_mean strips the observation noise and sees the disagreement.

  Use rhat_mean to decide whether a fitted curve or a forecast can be reported.
  Use rhat_pred only when the interval you are quoting is genuinely dominated
  by irreducible observation noise; it is the weaker of the two.

An empty cell means the reduction was refused, never that it passed: fewer
than 2 chains, fewer than 4 draws per chain, a draws.tsv with no chain column,
or a constant row. The one_step horizon leaves both pairs empty (its cells
pool over particles as well as draws).

quantities/<name>.tsv carries the same reduction as `rhat` and `ess` — one
pair, because a quantity has one value per draw. Those are the numbers that
get published, so read them first. A quantity over latent state or derived
arithmetic is noise-free, so its `rhat` is the undiluted kind; a quantity
whose manifest `source` is `observations` reduces sampled y_rep and so carries
observation noise, making it the diluted (rhat_pred) kind. `simulate` has no
chains behind it and writes neither column.

--by-chain adds a leading `chain` column: `all` on the pooled rows, the
1-based chain id on one extra band per chain, on both horizons. Use it after
rhat_mean has flagged something, to see *which* way the chains disagree.

  free_forward per chain  do the chains project the same future? Overlapping
                          bands mean the pooled band summarises one forecast;
                          separated ones mean it is a mixture of several, and
                          quoting its quantiles reads as uncertainty where the
                          truth is disagreement.
  one_step per chain      does each chain explain the observed record? These
                          are re-anchored to the data at every step, so a
                          separation is disagreement about the fitted
                          trajectory with the extrapolation removed — the
                          sharper statement about mixing.

Per-chain rows carry no rhat_*/ess_* cell (those compare chains) and report
their own n_draws — smaller than the pooled count, and smaller still for a
chain some of whose one-step draws hit a degenerate filter (that loss is named
on stderr; a chain that lost every draw is omitted rather than banded over
nothing). Without the flag no `chain` column is written at all.

It adds no artifact address of its own — the `all` rows are byte-identical, so
the file is a superset of the pooled one — and composes with
--exclude-chains, which does: --by-chain
--exclude-chains 3,5 writes predictive-excl3,5/ with a `chain` column, and the
ids there are the fit's own numbering with 3 and 5 simply absent."))]
pub struct FitPredictArgs {
    /// The fit, by handle: `@label`, a fit-level hash prefix, a fit results
    /// directory, or a `fit.toml` config (resolved to its unique run). A handle
    /// that maps to several fits errors and lists them — pass a run directory or
    /// a longer hash prefix to disambiguate.
    #[arg(long = "fit", value_name = "FIT")]
    pub fit_flag: Option<String>,

    /// Positional form of the fit handle, so `camdl fit predict @jigawa-baseline`
    /// or `camdl fit predict results/fits/<run>/` works like `fit summary`.
    #[arg(value_name = "FIT", conflicts_with = "fit_flag")]
    pub fit_pos: Option<String>,

    /// Restrict to one logical stream. Accepts the logical name (`onset`) or an
    /// expanded leaf name (`onset_Bo`), which maps up to its logical stream.
    #[arg(long, value_name = "STREAM")]
    pub stream: Option<String>,

    /// Use this stage's posterior cloud instead of the terminal one.
    #[arg(long, value_name = "STAGE")]
    pub stage: Option<String>,

    /// Prospective scenario overlay (repeatable; conflicts with --enable/--disable).
    /// Each `--scenario NAME` selects a model `scenarios {}` preset; the free-forward
    /// replay is looped over every scenario and the bands are tagged by a leading
    /// `scenario` column, the same way `horizon`/`treatment` already stack. No
    /// `--scenario` → a single `fitted` row (the fitted model, no overlay).
    /// `fitted` is a reserved name: a preset cannot use it. Layer-1 emits each
    /// scenario's bands only — never a between-scenario difference ("cases averted"
    /// lives in the conditioned counterfactual fork).
    #[arg(long = "scenario", conflicts_with_all = ["enable", "disable"])]
    pub scenarios: Vec<String>,

    /// Enable an intervention in an ad-hoc scenario overlay (repeatable; conflicts
    /// with --scenario). Mirrors `simulate --enable`.
    #[arg(long, conflicts_with = "scenarios")]
    pub enable: Vec<String>,

    /// Disable an intervention in an ad-hoc scenario overlay (repeatable; conflicts
    /// with --scenario). Mirrors `simulate --disable`.
    #[arg(long, conflicts_with = "scenarios")]
    pub disable: Vec<String>,

    /// Vary a parameter across a grid over the posterior (repeatable →
    /// multiple swept params → Cartesian). Each `--sweep PARAM=GRID` sets the
    /// swept parameter to each grid value in turn while the rest of every
    /// posterior draw propagates; cells are keyed by a leading `sweep:<param>`
    /// column. `GRID` is a list (`q=0,30,60`), `lin(min,max,n)`, or
    /// `log10(min,max,n)`. Composes with `--scenario` on DISTINCT parameters; a
    /// scenario and a sweep on the SAME parameter is a hard error (pin OR vary,
    /// not both). Free-forward only — the one-step horizon is sweep-agnostic.
    #[arg(long = "sweep", value_name = "PARAM=GRID")]
    pub sweep: Vec<crate::args::types::SweepSpec>,

    /// Which predictive horizon(s) to emit. Omitted = all applicable for the
    /// fit's backend (chain-binomial → `free_forward` + `one_step`; ODE →
    /// `free_forward` only). `--horizon one_step` on an ODE fit is a hard error.
    #[arg(long, value_name = "HORIZON")]
    pub horizon: Option<crate::args::types::HorizonArg>,

    /// Report this fit with the `quantities {}` block in FILE instead of the
    /// model's own. FILE is an ordinary `.camdl` file containing ONLY a
    /// `quantities {}` block; it is compiled against this fit's model source and
    /// refused unless that source is still the model the fit ran on. It REPLACES
    /// the model's block — never merges — and a name the model does not declare
    /// is an error naming both the name and FILE. The tables land in
    /// `quantities-<key>/`, keyed by FILE's contents, so two vocabularies over
    /// one fit produce two tables rather than overwriting each other. The fit's
    /// own identity is untouched (proposal 2026-08-19).
    #[arg(long, value_name = "FILE")]
    pub quantities: Option<PathBuf>,

    /// Cap the posterior cloud subsample for BOTH horizons (default 200). Each
    /// horizon pools plenty at a few hundred draws (one-step over
    /// `draws × particles`, free-forward over one forward replay per draw), so a
    /// larger cloud is evenly subsampled — a strided pick across the whole cloud,
    /// never silently run at full size (a full free-forward replay of a
    /// long-burn-in ODE fit is hours of solves).
    #[arg(long, value_name = "N")]
    pub n_draws: Option<usize>,

    /// RNG seed for the y_rep observation sampling (default 1).
    #[arg(long)]
    pub seed: Option<u64>,

    /// Also band each MCMC chain on its own, tagged by a leading `chain` column
    /// (`all` on the pooled rows, the 1-based chain id on the per-chain ones) —
    /// the same way `--scenario` tags its arms and `--sweep` its grid cells. The
    /// pooled band is unchanged and stays first-class; this adds rows to the
    /// same file, never a second file tree. Use it to *look* at a disagreement
    /// `rhat_mean` has already flagged: if the per-chain forward bands overlap,
    /// the pooled band summarises one forecast; if they separate, it is a
    /// mixture of several and quoting its quantiles reads as uncertainty where
    /// the truth is disagreement. Per-chain rows carry no `rhat_*`/`ess_*` cell
    /// (those compare chains). Both horizons are decomposed, and they answer
    /// different questions: free-forward per-chain bands say whether the chains
    /// *project* the same future, one-step per-chain bands say whether each chain
    /// *explains* the observed record. The one-step half is the sharper statement
    /// about mixing — it is re-anchored to the data at every step, so a
    /// separation there is disagreement about the fitted trajectory itself
    /// rather than extrapolation uncertainty. The one-step cell also pools over
    /// filter particles, but that affects a per-chain band's width exactly as it
    /// already affects the pooled band's; it is why those rows carry no
    /// `rhat_*`/`ess_*` (gh#798), not a reason to leave them pooled.
    ///
    /// Composes with `--exclude-chains`, and adds no address of its own: the
    /// keyed directory is the exclusion's (`predictive-excl3,5/`), because a
    /// `--by-chain` file is a strict superset of the pooled one — the `all`
    /// rows are byte-identical — while a chain subset is a different posterior.
    /// Chain ids are the fit's own numbering, never renumbered by an exclusion,
    /// so an excluded chain is simply absent and the ids that remain line up row
    /// for row with the pooled artifact's.
    #[arg(long = "by-chain")]
    pub by_chain: bool,

    /// Drop the named MCMC chains from the posterior cloud before banding —
    /// a comma-separated list of 1-based chain ids (matching the `chain_N/`
    /// dirs and the `fit summary` per-chain table), e.g. `--exclude-chains 3,5`.
    /// The escape hatch for a known-stuck minority of chains; post-hoc exclusion
    /// BIASES the posterior toward the retained mode and always prints a warning.
    /// A chain id not in the fit, or excluding every chain, is a hard error.
    /// The bands, quantities and contrasts land in `predictive-excl<ids>/`,
    /// `quantities-excl<ids>/` and `contrasts-excl<ids>/`, keyed by the excluded
    /// SET (`3,5` and `5,3` are one address), so a subset never overwrites the
    /// pooled artifact and two subsets never overwrite each other.
    #[arg(long, value_name = "IDS")]
    pub exclude_chains: Option<String>,
}

/// The reserved scenario name for the no-overlay row — the fitted model as
/// written. Reserved so the `scenario` column's no-overlay value can never be
/// shadowed by a user preset (which would make rows ambiguous). Mirrors
/// `simulate`'s `baseline` sentinel, but named `fitted` because in
/// `fit predict` the parameters come from the fit, not a default-parameter
/// preset, so reusing `baseline` would mislead.
pub const FITTED: &str = "fitted";

/// Split `--scenario` values on commas, trim, drop empties, and reject repeats.
///
/// One shared check because both verbs mis-handle a repeat, differently and
/// both badly: `simulate` would accumulate two cells into one scenario bucket
/// and fail with "point-mode quantities require exactly one realization" AFTER
/// the whole grid had simulated and committed leaves, while `fit predict` bands
/// each posterior draw twice and reports `n_draws = 2N` — a band that looks
/// right with a sample count that is not (gh#579). A repeat is never what the
/// user meant, so it is rejected at parse where the diagnostic can name it.
///
/// Note the split happens here: `--scenario a,b` is one flag carrying two
/// names, which is why a guard counting flags rather than names missed it
/// (gh#562).
pub fn split_scenario_names(raw: &[String]) -> Result<Vec<String>, String> {
    let names: Vec<String> = raw
        .iter()
        .flat_map(|s| s.split(',').map(|t| t.trim().to_string()))
        .filter(|s| !s.is_empty())
        .collect();
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for n in &names {
        if !seen.insert(n.as_str()) {
            return Err(format!(
                "scenario '{n}' is named more than once.\n  \
                 Each `--scenario` selects one arm to run; repeating a name would \
                 run and summarize it twice.\n  \
                 Fix: drop the duplicate."
            ));
        }
    }
    Ok(names)
}

impl FitPredictArgs {
    /// The raw fit handle (`--fit` or the positional form), unparsed.
    pub fn fit(&self) -> Result<&str, String> {
        self.fit_flag
            .as_deref()
            .or(self.fit_pos.as_deref())
            .ok_or_else(|| {
                "a fit handle is required: `@label`, a hash prefix, a run directory, \
                 or `--fit fit.toml`"
                    .into()
            })
    }

    /// Parse the repeatable `--scenario`/`--enable`/`--disable` surface into the
    /// shared `Vec<ScenarioRef>`, exactly the way `simulate` does (`main.rs`'s
    /// `from_cli`): `--scenario a,b` comma-splits into `[Named("a"), Named("b")]`
    /// (the preset path); no `--scenario` yields a single [`FITTED`] inline
    /// overlay carrying any `--enable`/`--disable` (so the no-overlay path keeps
    /// `scenario_name = None`, exactly like `simulate`'s baseline). `fitted` is
    /// reserved: an explicit `--scenario fitted` is rejected with the same
    /// migration-style diagnostic the OCaml `scenarios {}` reservation uses.
    pub fn scenario_refs(&self) -> Result<Vec<crate::sim_job::ScenarioRef>, String> {
        use crate::sim_job::ScenarioRef;
        let names = split_scenario_names(&self.scenarios)?;
        if let Some(bad) = names.iter().find(|n| n.as_str() == FITTED) {
            return Err(format!(
                "scenario name '{bad}' is reserved: it labels the no-overlay row \
                 (the fitted model, no scenario applied) in the `scenario` column.\n  \
                 Fix: rename the scenario, or drop `--scenario {FITTED}` (the \
                 no-overlay row is emitted automatically)."
            ));
        }
        if names.is_empty() {
            // No overlay: a single inline `fitted` scenario carrying any CLI
            // enable/disable (empty for the bare no-`--scenario` case). The inline
            // form keeps the engine on the ad-hoc branch (scenario_name = None),
            // byte-identical to today's hardcoded baseline replay when no
            // enable/disable is given.
            Ok(vec![ScenarioRef::Inline {
                name: FITTED.to_string(),
                enable: self.enable.clone(),
                disable: self.disable.clone(),
                params: indexmap::IndexMap::new(),
            }])
        } else {
            // gh#625: the fitted no-overlay arm is ALWAYS emitted, first — it
            // is the posterior predictive every scenario overlays, and its
            // absence is never what the user wants (the ebola national
            // predicts ended up with scenario deltas and no reference arm
            // when one_step aborted). This also makes the reservation
            // diagnostic above ("emitted automatically") true as written.
            // `--scenario` conflicts with `--enable`/`--disable` at the clap
            // level, so the prepended arm is guaranteed pure no-overlay.
            let mut refs = vec![ScenarioRef::Inline {
                name: FITTED.to_string(),
                enable: self.enable.clone(),
                disable: self.disable.clone(),
                params: indexmap::IndexMap::new(),
            }];
            refs.extend(names.into_iter().map(ScenarioRef::Named));
            Ok(refs)
        }
    }
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Compare two fit.toml configurations side-by-side
  camdl fit diff fit-a.toml fit-b.toml
"))]
pub struct FitDiffArgs {
    /// First fit config
    pub a: PathBuf,
    /// Second fit config
    pub b: PathBuf,
}

/// Output format for `camdl fit table`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum FitTableFormat {
    Text,
    Json,
    Md,
    Csv,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Show every fit under results/fits/, default text view
  camdl fit table results/fits

  # Just the converged ones
  camdl fit table results/fits --converged

  # Project to one row in JSON for downstream tooling
  camdl fit table results/fits --hash 04ab12cd --format json

  # Filter by method
  camdl fit table results/fits --with-method pgas
"))]
pub struct FitTableArgs {
    /// Path to the fits root (`results/fits/` or wherever the project
    /// stores them). Walks every `<root>/<dir>/run.json` of kind
    /// `Fit`.
    pub root: PathBuf,

    /// Show only converged rows (IF2: gate Pass; PGAS / PMMH: max R̂ < 1.05).
    #[arg(long)]
    pub converged: bool,

    /// Show only rows whose convergence boolean is false.
    #[arg(long)]
    pub gate_failed: bool,

    /// Filter to fits whose declared stages include the named stage.
    #[arg(long, value_name = "STAGE")]
    pub with_stage: Option<String>,

    /// Filter to fits whose terminal-stage method matches.
    #[arg(long, value_name = "METHOD",
          value_parser = clap::builder::PossibleValuesParser::new(["if2", "pgas", "pmmh"]))]
    pub with_method: Option<String>,

    /// Filter to fits with this model_identity (prefix match).
    #[arg(long, value_name = "HASH_PREFIX")]
    pub model: Option<String>,

    /// Filter to fits whose `fit_hash` starts with the
    /// given prefix. Useful for projecting to one row in JSON without
    /// piping through `jq`. The `summary ⊆ table` Deliverable C test
    /// uses this.
    #[arg(long, value_name = "HASH_PREFIX")]
    pub hash: Option<String>,

    /// Filter to fits younger than the given duration in seconds.
    /// Future work may accept human strings (`7d`, `24h`); today it's
    /// just seconds, which the test harness can produce trivially.
    #[arg(long, value_name = "SECONDS")]
    pub since_seconds: Option<i64>,

    /// Filter to fits whose label matches a glob (step 8 will
    /// populate labels; pre-step-8 this filter always excludes
    /// everything).
    #[arg(long, value_name = "GLOB")]
    pub label_pattern: Option<String>,

    /// Pick a specific fit as the diff baseline (prefix match on
    /// `fit_hash`). Default: lowest hash among the surviving cohort.
    #[arg(long, value_name = "HASH_PREFIX")]
    pub baseline: Option<String>,

    /// Output format. `text` (default) is a fixed-width terminal
    /// view; `json` is the schema-pinned cross-fit document; `md`
    /// renders a GitHub-flavoured table; `csv` is downstream-friendly.
    #[arg(long, value_enum, default_value_t = FitTableFormat::Text)]
    pub format: FitTableFormat,

    /// Add a column showing the posterior median of a SCALAR generated
    /// quantity declared in the fit's model `quantities {}` block (the
    /// q50 of the no-overlay `fitted` row). Repeatable for several
    /// quantities. Unlike the default read-only table, `--quantity`
    /// may DERIVE the value on demand: for a fit that has not been
    /// predicted yet it runs `fit predict --horizon free_forward`,
    /// populating that fit's `quantities/` outputs. Optimizer fits
    /// (IF2 / NLopt) have no posterior cloud, so their cell renders `—`.
    #[arg(long = "quantity", value_name = "NAME")]
    pub quantities: Vec<String>,

    /// Drop the named 1-based MCMC chains (comma-separated, e.g.
    /// `--exclude-chains 3,5`) from each fit's posterior cloud when DERIVING a
    /// `--quantity` cell. Only affects derived cells, so it requires
    /// `--quantity`; the same drop set is applied to every fit in the cohort
    /// (chain 3 of fit A is unrelated to chain 3 of fit B — pair with `--hash`
    /// to project to one fit). Post-hoc exclusion BIASES the posterior toward
    /// the retained mode and always prints a warning.
    #[arg(long, value_name = "IDS", requires = "quantities")]
    pub exclude_chains: Option<String>,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Scaffold a new fit.toml from an existing baseline
  camdl fit new --from base.toml variant.toml
"))]
pub struct FitNewArgs {
    /// Source fit.toml to derive from
    #[arg(long)]
    pub from: PathBuf,

    /// Destination path for the new config
    pub dest: PathBuf,
}

// ─── label ────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Label any run kind by its short hash
  camdl label 04ab12cd \"baseline-2024\"
  camdl label 9c5d11f0 \"baseline sim, daily reporting\"
  camdl label 7e2a5d4b \"R0 vs gamma profile, take 2\"

  # Labels are the headline way to find a run again. Label it once,
  # then list/show surfaces the label so you can spot it by name:
  camdl label <hash> \"baseline-2024\"
  camdl list --kind fit          # the labelled fit now shows its name
  camdl show <hash>              # label appears in the run header

Notes:
  - Labels are 1–64 characters after trim, restricted to:
    letters, digits, spaces, commas, dot, underscore, hyphen.
  - The hash is matched as a prefix (8+ chars recommended) across
    every kind under <root>/ (sims, fits, profiles, pfilters, surveys).
  - Errors on ambiguous or unmatched prefix.
  - Errors on a still-running fit (RunStatus::Running); the
    runner would otherwise overwrite the label at completion.
  - Concurrent invocations are last-write-wins.
"))]
pub struct LabelArgs {
    /// Hash prefix of the target run (matches against any kind's leaf
    /// `run.json` `run_id` under
    /// `<root>/{sims,fits,profiles,pfilters,surveys}/`)
    pub hash: String,

    /// New label text. Validated against ^[a-zA-Z0-9 ,._-]{1,64}$
    /// after trim. Empty / whitespace-only labels are rejected.
    pub label: String,

    /// Output root to search under (default: ./results)
    #[arg(long, default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub root: PathBuf,
}

// ─── pfilter ──────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Loglik at fixed parameters
  camdl pfilter sir.camdl --params p.toml --data cases.tsv \\
      --particles 5000 --seed 1

  # Multiple replicate filters for loglik SD
  camdl pfilter sir.camdl --params p.toml --data cases.tsv \\
      --particles 2000 --replicates 10

  # Save smoothing paths (ancestor-traced) for plotting vs data
  camdl pfilter sir.camdl --params p.toml --data cases.tsv \\
      --particles 5000 --n-paths 20 --save-paths paths.tsv

  # Prequential out-of-sample evaluation
  camdl pfilter sir.camdl --params p.toml --data cases.tsv \\
      --particles 5000 --save-prequential preq
"))]
// gh#194: `--scenario` and explicit-θ flags (`--params` / `--param`) are a
// hard conflict on pfilter. A scenario's `set`/`scale` block resolves at a
// *higher* precedence than `--params` (params_resolver tier 4 > tier 3), so
// pinning θ via `--params` while a scenario is active would silently score
// the likelihood at the scenario's θ, not the user's — a silent-wrong-θ
// result. On `simulate` the same precedence is intentional (`--params` sets
// baseline values, the scenario applies a counterfactual modification on top
// — pinned by `scenario_runtime_application.rs`); on `pfilter` there is no
// such "baseline + counterfactual" semantics — the user wants one θ scored —
// so we reject the ambiguous combination at the parse layer instead.
// `--enable`/`--disable` stay compatible with `--params`: those toggle
// interventions, not parameter values, so "pin θ + toggle an intervention"
// is coherent.
#[command(group(ArgGroup::new("pfilter_explicit_theta")
    .args(["params", "param"]).multiple(true).conflicts_with("scenario")))]
pub struct PfilterArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    #[command(flatten)]
    pub model_overrides: ModelOverrides,

    #[command(flatten)]
    pub scenario: ScenarioArgs,

    #[command(flatten)]
    pub inference: InferenceCore,

    #[command(flatten)]
    pub stream: StreamSelection,

    /// Observation data TSV (with time column).
    ///
    /// gh#90: polymorphic, repeatable. Two forms (mutually exclusive
    /// within a single invocation):
    ///   --data PATH         single-stream: binds to the model's only
    ///                       observation block (or the one selected by
    ///                       --obs NAME).
    ///   --data NAME=PATH    multi-stream: bind one observation block by
    ///                       name. Repeat for every stream.
    /// Mixing the two forms is a hard error. Multi-stream models must
    /// bind every block (warning fires when only a subset is bound).
    #[arg(long, value_name = "[NAME=]PATH")]
    pub data: Vec<DataSpec>,

    /// Optional fit toml supplying the multi-stream binding via its
    /// `[data.observations]` map (gh#90). Only consulted when no
    /// `--data` flags were supplied; CLI flags always win.
    #[arg(long, value_name = "PATH")]
    pub fit: Option<PathBuf>,

    /// Conditioning window (warm-up) boundary — the `condition_from` key of a
    /// fit toml, as a flag (gh#621). The warm-up [t_start, boundary) is
    /// simulated but NOT scored, so the loglik matches a fit that conditions
    /// the same way. Repeatable, two forms: a bare SPEC (all-streams default,
    /// at most one) and LABEL=SPEC (one stream's observation-block label).
    /// SPEC forms: "first_obs - <N> <unit>", a model-time number, or a
    /// calendar date. When absent, a `--fit` toml's `condition_from` applies;
    /// this flag wins.
    #[arg(long = "condition-from", value_name = "[LABEL=]SPEC")]
    pub condition_from: Vec<String>,

    /// Number of independent filter runs
    #[arg(long, default_value_t = 1)]
    pub replicates: usize,

    /// Write per-observation diagnostics TSV; use "-" for stdout
    #[arg(long)]
    pub trace: Option<String>,

    /// Write a particle-filter health report: per-observation ESS and Snyder
    /// τ² (log-weight variance), plus a printed summary with the implied
    /// particles-to-avoid-collapse estimate exp(τ²/2). Use "-" for stdout.
    #[arg(long)]
    pub pf_health: Option<String>,

    /// Output file (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Write final particle states to this TSV
    #[arg(long)]
    pub save_final_state: Option<PathBuf>,

    /// Write N trajectory samples from smoothing distribution to this path
    #[arg(long)]
    pub save_paths: Option<PathBuf>,

    /// Number of trajectories for --save-paths
    #[arg(long, default_value_t = 1)]
    pub n_paths: usize,

    /// Write per-step particle states and log-weights to this TSV
    #[arg(long)]
    pub save_filtering: Option<PathBuf>,

    /// Write {STEM}.tsv (per-step log score, CRPS, PIT, ESS) + {STEM}.json
    /// (full typed PrequentialTrace) for the plug-in one-step-ahead
    /// predictive at the fixed parameters. See
    /// docs/dev/proposals/2026-04-20-prequential-evaluation.md.
    #[arg(long)]
    pub save_prequential: Option<String>,

    /// With --save-prequential, score the trace only at observations
    /// strictly after TIME (gh#585 / the 2026-08-29 proposal, Stage 3.2):
    /// earlier observations are still assimilated — the filter reweights
    /// on them — but excluded from the trace. This is the held-out-tail
    /// scoring mode `camdl compare` derives with (TIME = the fit's
    /// `holdout_after`). Accepts the shared time grammar: a model-time
    /// number, a date under a calendar-anchored model, or
    /// `last_obs - N weeks`. The total log-likelihood output is
    /// unchanged; only the trace is windowed, and it records the
    /// boundary (`score_from`).
    #[arg(long, value_name = "TIME", requires = "save_prequential")]
    pub score_from: Option<String>,

    /// With --save-prequential, drop per-particle predictive samples
    /// from {STEM}.json. Keeps scalar scores, shrinks the file.
    #[arg(long)]
    pub no_save_samples: bool,
}

// ─── if2 (removed; deprecation stub) ────────────────────────────────────────────

/// `camdl if2` is removed (gh#147). A one-method IF2 fit is now a fit
/// with a single `algorithm = "if2"` stage, run through `camdl fit run`.
/// This catch-all accepts and ignores any arguments so an old invocation
/// lands on the actionable migration message in [`crate::if2::cmd_if2`]
/// rather than a clap parse error — a deprecation redirect, not a
/// back-compat shim.
#[derive(Args)]
pub struct If2Args {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub _ignored: Vec<String>,
}

// ─── profile ──────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # 1D profile likelihood for R0 via parallel IF2
  camdl profile sir.camdl --data cases.tsv \\
      --sweep \"R0=lin(0.5,5,20)\" --particles 2000 --rw-sd auto

  # 2D profile (R0 × sigma)
  camdl profile sir.camdl --data cases.tsv \\
      --sweep \"R0=lin(0.5,5,10)\" --sweep \"sigma=lin(0.1,1.0,10)\" \\
      --rw-sd auto

  # Slice profile: hold gamma at 0.1, sweep tau (the canonical
  # `--fixed NAME=VALUE` pattern — kicks gamma from [estimate])
  camdl profile sir.camdl --data cases.tsv \\
      --sweep \"tau=lin(-35,-1,30)\" --fixed gamma=0.1 --rw-sd auto

  # Profile-posterior sweep — PMMH per cell with priors from a fit toml
  camdl profile sir.camdl --data cases.tsv \\
      --sweep \"tau=lin(-35,-1,30)\" --algorithm pmmh \\
      --fit fits/profile_tau.toml --pmmh-steps 1500

  # Warm-start chains from a hand-written params TOML
  camdl profile sir.camdl --data cases.tsv \\
      --sweep \"R0=lin(0.5,5,20)\" --init from_params --params truth.toml \\
      --starts 4 --rw-sd auto

  # Warm-start chains from a prior fit's MLE
  camdl profile sir.camdl --data cases.tsv \\
      --sweep \"R0=lin(0.5,5,20)\" --init from_mle --mle fits/scout/ \\
      --starts 4 --rw-sd auto

PRIORS (--algorithm pmmh)

  The per-cell PMMH path honours priors via a three-tier precedence
  chain:

    1. --fit <toml>'s [estimate.<param>.prior] block (highest)
    2. Model-IR `~` priors (fallback; from DSL `~` syntax)
    3. Prior::Flat (last resort; emits a structured warning naming
       every estimated parameter that fell through, citing the two
       remedies)

  Suppress the warning loudly with --suppress-warnings — the waiver
  is recorded into run.json's suppressed_warnings array.

  IF2 / NLopt paths ignore priors by design (they maximize the
  likelihood). The warning fires only for --algorithm pmmh.

OUTPUT

  The `--output` TSV mirrors the umbrella `summary.tsv`. Schema:

    <focal_1> ... <focal_N>  loglik  [<spread_cols>]  <param_1> ...
      acc_rate_avg  acc_rate_min  loglik_spread_starts
      loglik_rhat_starts  starts_n_completed
      iterations_used  cooling_final

  The trailing seven columns are gh#74 Option B per-cell convergence
  diagnostics. Read them by column name — order is stable per run
  but future schema additions land after this block.

  Per-column meaning (algorithm-specific cells render as NaN when
  the algorithm doesn't supply that value):

    acc_rate_avg / acc_rate_min   PMMH MH acceptance rate, mean / min
                                  across the K --starts chains.
    loglik_spread_starts          max - min of per-start MAP
                                  log-likelihoods. > ~5 nats means
                                  the starts disagree on the basin.
    loglik_rhat_starts            Gelman-Rubin R-hat across the K
                                  per-start log-likelihood traces.
                                  NaN at K < 3 (R-hat is unstable
                                  at K=2, undefined at K=1).
    starts_n_completed            Count of starts that produced a
                                  finite final loglik. < K when one
                                  or more starts diverged.
    iterations_used               IF2: final cooling step index.
    cooling_final                 IF2: actual ending perturbation
                                  SD (mean across estimated params),
                                  not the target.

  Full reference: docs/inference.md, `Per-cell diagnostics` section.
"))]
pub struct ProfileArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    #[command(flatten)]
    pub model_overrides: InferenceModelOverrides,

    #[command(flatten)]
    pub scenario: ScenarioArgs,

    #[command(flatten)]
    pub inference: InferenceCore,

    #[command(flatten)]
    pub stream: StreamSelection,

    /// Observation data TSV.
    ///
    /// gh#90: polymorphic, repeatable. Two forms (mutually exclusive
    /// within a single invocation):
    ///   --data PATH         single-stream: binds to the model's only
    ///                       observation block (or the one selected by
    ///                       --obs NAME).
    ///   --data NAME=PATH    multi-stream: bind one observation block by
    ///                       name. Repeat for every stream.
    /// Mixing the two forms is a hard error. Multi-stream models must
    /// bind every block (warning fires when only a subset is bound).
    #[arg(long, value_name = "[NAME=]PATH")]
    pub data: Vec<DataSpec>,

    /// Optional fit toml supplying priors, bounds, and fixed list for
    /// the per-cell PMMH (gh#73). Mirrors `camdl survey --fit`'s
    /// schema (the `[estimate]`, `[fixed]`, `[model]` blocks). When
    /// supplied, the resolver picks priors via the chain
    ///   `--fit toml` > model-IR `~` priors > flat (warning).
    /// `--params` still carries values only; when both are supplied,
    /// `--params` overrides any starting values from the fit toml but
    /// the priors and bounds come from `--fit`. The focal swept
    /// parameter is always removed from the estimated set, even when
    /// it appears in the fit toml's `[estimate]` block. Without
    /// `--fit`, the resolver falls back to model-IR priors (or flat
    /// with a warning when none are declared).
    #[arg(long, value_name = "PATH")]
    pub fit: Option<PathBuf>,

    /// Conditioning window (warm-up) boundary — the `condition_from` key of a
    /// fit toml, as a flag (gh#621). The warm-up [t_start, boundary) is
    /// simulated but NOT scored, so the loglik matches a fit that conditions
    /// the same way. Repeatable, two forms: a bare SPEC (all-streams default,
    /// at most one) and LABEL=SPEC (one stream's observation-block label).
    /// SPEC forms: "first_obs - <N> <unit>", a model-time number, or a
    /// calendar date. When absent, a `--fit` toml's `condition_from` applies;
    /// this flag wins.
    #[arg(long = "condition-from", value_name = "[LABEL=]SPEC")]
    pub condition_from: Vec<String>,

    /// Suppress the `profile_flat_prior_fallback` warning when any
    /// estimated parameter resolves to a flat prior (gh#73). Use only
    /// when flat priors are intentional — the warning is recorded in
    /// `run.json` either way so the choice is auditable.
    #[arg(long)]
    pub suppress_warnings: bool,

    /// Profile grid (repeat for 2D+).
    /// SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC", required = true)]
    pub sweep: Vec<SweepSpec>,

    /// IF2 iterations per grid point
    #[arg(long, default_value_t = 50)]
    pub iterations: usize,

    /// Independent IF2 starts per grid point
    #[arg(long, default_value_t = 3)]
    pub starts: usize,

    /// How the per-cell starting points (for `--starts > 1`) are drawn
    /// across the non-focal estimated parameters' bounds. See `--help`
    /// for the full INIT MODES block.
    #[arg(long, value_name = "MODE", value_enum,
          default_value_t = InitModeTag::Uniform,
          long_help = INIT_LONG_ABOUT)]
    pub init: InitModeTag,

    /// Companion path for `--init from_posterior`. Accepts a posterior
    /// draws TSV directly or a fit-results directory (auto-resolves
    /// to `<dir>/draws.tsv`).
    #[arg(long, value_name = "PATH")]
    pub posterior: Option<PathBuf>,

    /// Companion path for `--init from_mle`. Accepts an MLE TOML file
    /// directly (`mle.toml` / `final_params.toml`) or a fit-results
    /// directory (auto-resolves to `<dir>/mle.toml` then
    /// `<dir>/final_params.toml`).
    #[arg(long, value_name = "PATH")]
    pub mle: Option<PathBuf>,

    /// Companion path for `--init from_params`. Hand-written flat
    /// params TOML; top-level keys are parameter names. This flag is
    /// the init-mode counterpart to the **removed** value-setter
    /// `--params` flag; it only fires when `--init from_params` is
    /// also passed.
    #[arg(long = "params", value_name = "TOML")]
    pub init_params: Option<PathBuf>,

    /// Cooling schedule
    #[arg(long, default_value_t = 0.95)]
    pub cooling: f64,

    /// Random-walk SDs
    #[arg(long)]
    pub rw_sd: Option<RwSd>,

    /// Profile TSV output (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Multi-seed sensitivity: run the entire profile grid at each
    /// seed in this list. Surfaces stochastic IF2 instability per
    /// grid point (high spread across seeds → that cell's MLE is not
    /// trustworthy from a single chain). When omitted, falls back to
    /// `--seed` for a single-seed run. Accepts comma list `1,2,3` or
    /// inclusive range `1:5`.
    #[arg(long, value_name = "SPEC")]
    pub seeds: Option<SeedSpec>,

    /// User-supplied display label for this profile run. Validated
    /// against `^[a-zA-Z0-9 ,._-]{1,64}$` after trim. Surfaced in
    /// `camdl list` and `camdl show`. For multi-seed runs the label
    /// applies to the umbrella; per-seed children remain unlabelled.
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,

    /// Per-cell inference algorithm. When omitted the per-cell runner
    /// uses `if2` on `chain_binomial` (the historical default). Pass an
    /// algorithm + backend pair from `camdl fit methods` to switch
    /// (e.g. `--algorithm nl-sbplx --backend ode` for deterministic
    /// per-cell MLE — typically 100×–1000× faster on equilibrium /
    /// large-population fits where the PF is structurally redundant).
    /// Validated against the methods registry at startup; invalid pairs
    /// error with an actionable suggestion.
    #[arg(long, value_name = "NAME")]
    pub algorithm: Option<String>,

    /// Simulation backend. Defaults to `chain_binomial`; pass
    /// `--backend ode` together with an ODE-compatible algorithm
    /// (`nl-sbplx`, `nl-bobyqa`) for deterministic per-cell MLE.
    #[arg(long, value_name = "NAME")]
    pub backend: Option<String>,

    /// PMMH only: number of MCMC steps per profile cell. Ignored
    /// for other algorithms. Default 500 is enough for ridge-finding
    /// at typical profile resolution; bump to 1000-2000 if a cell's
    /// posterior is multi-modal.
    #[arg(long, default_value_t = 500, value_name = "N")]
    pub pmmh_steps: usize,

    /// PMMH only: particles per PF evaluation. Default 500 is the
    /// standard PMMH range; CPM (rho > 0) keeps MCMC mixing high
    /// without needing 1000+.
    #[arg(long, default_value_t = 500, value_name = "N")]
    pub pmmh_particles: usize,

    /// PMMH only: Crank-Nicolson correlation for correlated
    /// pseudo-marginal (CPM-MCMC). Default 0.99 gives strong PF
    /// correlation and excellent MH mixing. Pass `0.0` (or any
    /// value ≤ 0) to disable CPM and run vanilla PMMH with
    /// independent PF draws.
    #[arg(long, default_value_t = 0.99, value_name = "FLOAT")]
    pub pmmh_rho: f64,
}

// ─── survey ───────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
EXAMPLES

  # Fit-aware: read [estimate] bounds and [data] from fit.toml
  camdl survey model.camdl --fit fit.toml

  # Inline bounds, data file specified directly
  camdl survey model.camdl --data cases.tsv \\
      --estimate \"beta=0.001:1.0\" --estimate \"gamma=0.01:0.5\"

  # Fast deterministic-only mode (skip PF; not safe for stochastic models)
  camdl survey model.camdl --fit fit.toml --eval simulate

  # Render the interactive HTML alongside the TSV
  camdl survey model.camdl --fit fit.toml --render

WHAT THIS IS

  A diagnostic tool that draws N Latin-hypercube points across the
  declared parameter bounds, evaluates the marginal log-likelihood
  at each, and writes a TSV (and optionally an interactive HTML
  pair-plot) to surface identifiability structure. It is intended
  to be run BEFORE camdl fit, to answer:

    - Is this model identifiable from this data?
    - Are there ridges or multiple basins?
    - Are the high-loglik regions biologically plausible?
    - Where do likely basins concentrate? (informs scout bounds)

  Survey is NOT a fitting routine. It does not produce an MLE.
  The output cannot substitute for camdl fit.

WHEN TO TRUST THE OUTPUT

  Survey works well when:
    - Process noise is small (deterministic-skeleton regime),
      or `--eval pfilter` is used with adequate
      particles/replicates
    - Parameter dimension d <= 8 (pair-plots are visually parseable)
    - Bounds reflect informed prior plausibility (not \"throw a
      wide net\")
    - Dynamics are not strongly chaotic (seasonally-forced SEIR
      with high R0 may produce intrinsically jagged landscapes)

KNOWN LIMITATIONS

  Stochastic deceiver (mitigated by --eval pfilter):
    Single-trajectory loglik is a 1-sample Monte Carlo estimate of
    p(y|theta) with variance proportional to the model's process
    noise. With high noise (e.g. multiplicative gamma white noise
    on transmission, sigma_se > ~1) the rank of N points by
    single-trajectory loglik is biased toward \"lucky outliers\"
    (Andrieu & Roberts 2009; Doucet et al. 2015, Biometrika). The
    default --eval pfilter substantially mitigates this; survey
    will warn at run end if the per-point loglik SE distribution
    indicates unreliable ranks.

  Chaotic dynamics:
    Seasonally-forced SEIR and similar systems have positive
    Lyapunov exponents in much of parameter space (Earn et al.
    2000; Bauch & Earn 2003). Small delta-theta produces wildly
    divergent deterministic trajectories. The landscape will be
    intrinsically jagged regardless of eval method. Interpret
    such surveys cautiously: the diagnostic is correctly
    reporting \"this is hard,\" not \"your model is broken.\"

  Bounds dependence:
    Survey ranks are conditional on the bounds you give. Wide
    bounds dilute (the \"top 10%\" may be marginally-less-bad
    rather than meaningfully-good). Narrow bounds may exclude
    the true basin entirely with no signal that this happened.
    Bound choice is a load-bearing modelling decision; survey
    cannot rescue a poorly-specified bounds box.

  Curse of dimensionality:
    Pair-plots project 2D marginals from a d-dimensional joint
    distribution. High-loglik points concentrating in a 2D
    pair may reflect tight conditioning on unshown parameters
    not visible in that view. Past d ~= 8 this becomes hard to
    interpret. Survey emits warnings at d > 6 and d > 10;
    consider camdl profile for higher-dimensional
    identifiability questions.

  Misspecification != identifiability:
    A tight, well-clustered top-K is a necessary but not
    sufficient condition for trusting the resulting fit. A
    misspecified model can have a tight likelihood at a
    wrong-but-best-fitting theta. Posterior predictive checks
    against held-out data are the orthogonal validation;
    survey cannot substitute.

  Silent miss case:
    With N points in d dimensions, LHS may not hit a true basin
    that occupies a small fraction of the bounds box. The
    landscape would then show structure of wrong basins with
    no signal that the right one was missed. If results look
    surprising, increase --n-points and re-run.

CITED REFERENCES

  Andrieu, C. & Roberts, G. O. (2009). The pseudo-marginal
    approach for efficient Monte Carlo computations. Annals of
    Statistics, 37(2), 697-725.
  Doucet, A., Pitt, M. K., Deligiannidis, G. & Kohn, R. (2015).
    Efficient implementation of MCMC when using an unbiased
    likelihood estimator. Biometrika, 102(2), 295-313.
  Earn, D. J. D., Rohani, P., Bolker, B. M. & Grenfell, B. T.
    (2000). A simple model for complex dynamical transitions in
    epidemics. Science, 287(5453), 667-670.
"))]
pub struct SurveyArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    /// fit.toml supplying [estimate] bounds and [data] (fit-aware mode).
    /// Mutually exclusive with --estimate / --data; pass exactly one.
    #[arg(long, conflicts_with_all = ["estimate", "data"])]
    pub fit: Option<PathBuf>,

    /// Inline LHS bounds, e.g. --estimate "beta=0.001:1.0" (repeat).
    /// Required when --fit is not given.
    #[arg(long, value_name = "NAME=LO:HI")]
    pub estimate: Vec<types::EstimateBoundsSpec>,

    /// Inline observation data TSV. Required when --fit is not given.
    #[arg(long)]
    pub data: Option<PathBuf>,

    /// Inline fixed parameters (NAME=VALUE), repeat. Useful in
    /// inline mode to pin parameters not in --estimate at a known
    /// value rather than the model default.
    #[arg(long, value_name = "NAME=VALUE")]
    pub fixed: Vec<types::ParamOverride>,

    /// Named scenario from the model. Applies the scenario's
    /// enable/disable lists and param overrides before survey.
    #[arg(long)]
    pub scenario: Option<String>,

    /// Number of Latin-hypercube points to evaluate.
    ///
    /// Default behaviour: auto-scale with parameter dimension as
    /// `max(1000, 50 * d^2)` so the n/d^2 >= 50 pair-plot coverage
    /// floor is met by default. For d=4 this is 1000 (unchanged from
    /// v1); for d=8 it's 3200; for d=12 it's 7200. Pass `--n-points
    /// N` to override (e.g. lower it explicitly for fast iteration,
    /// or higher for sparse-basin models). Set 0 to use the auto rule
    /// regardless.
    #[arg(long, default_value_t = 0)]
    pub n_points: usize,

    /// Likelihood evaluation method:
    ///   `auto` (default) — pick from the model: `pfilter` if any rate
    ///     uses overdispersed() / similar process noise; otherwise
    ///     `simulate`. The chosen method is announced at run start and
    ///     stored in run.json; the `auto` discriminator itself is never
    ///     persisted.
    ///   `pfilter` — particle filter, K replicates → logmeanexp combiner.
    ///     Estimates p(y|θ) under the chain-binomial process. Doucet et
    ///     al. 2015 gives the bar for trustworthy ranks: per-point loglik
    ///     SE ≤ ~1.7 nats.
    ///   `simulate` — single deterministic trajectory per point. ~10×
    ///     cheaper but biased toward "lucky outliers" when process noise
    ///     is non-trivial. Safe only for known-deterministic models.
    #[arg(long, default_value_t = crate::run_meta::SurveyEvalMethod::Auto)]
    pub eval: crate::run_meta::SurveyEvalMethod,

    /// Particle count for `--eval pfilter`. 200 is adequate for
    /// `sigma_se <= 1` on weekly data per the proposal's per-point
    /// cost table.
    #[arg(long, default_value_t = 200)]
    pub eval_particles: usize,

    /// PF replicates per LHS point (logmeanexp combiner). The
    /// replicate variance also drives the per-point loglik_se column;
    /// at K=3 the SE *estimate* itself has ~50% uncertainty (df=2),
    /// which can fire the Doucet 1.7-nat warning spuriously. K=5
    /// matches Sherlock et al. 2015's recommendation for
    /// pseudo-marginal MCMC and gives a tighter SE estimate at
    /// 5/3× the compute. Always 1 with `--eval simulate`.
    #[arg(long, default_value_t = 5)]
    pub eval_replicates: usize,

    /// LHS / PF base seed.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,

    /// Render an interactive HTML pair-plot (`landscape.html`)
    /// alongside the TSV. Off by default — TSV is the canonical
    /// artifact, HTML is opt-in (proposal §"Default behaviour").
    #[arg(long)]
    pub render: bool,

    /// Output root directory (default: ./results, matches the rest
    /// of camdl).
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// User-supplied display label for this survey run. Validated
    /// against `^[a-zA-Z0-9 ,._-]{1,64}$` after trim. Surfaced in
    /// `camdl list` and `camdl show`.
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,

    /// Force re-evaluation of all LHS points; bypass the cache (the
    /// CAS layout still applies — same hash, fresh artifacts).
    #[arg(long)]
    pub force: bool,

    /// Rayon thread count (0 = all available cores).
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    pub parallel: usize,
}

// ─── eval ─────────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Evaluate one or more expressions on a time grid
  camdl dev eval sir.camdl --params p.toml \\
      --expr \"beta,gamma\" --from 0 --to 730 --every 1

  # Inspect a forcing function over time
  camdl dev eval sir.camdl --params p.toml \\
      --expr \"seasonal(t)\" --from 0 --to 365
"))]
pub struct EvalArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    #[command(flatten)]
    pub model_overrides: ModelOverrides,

    /// Expression names to evaluate (comma-separated)
    #[arg(long, value_delimiter = ',', required = true)]
    pub expr: Vec<String>,

    /// Time grid start
    #[arg(long, default_value_t = 0.0, conflicts_with = "at")]
    pub from: f64,

    /// Time grid end
    #[arg(long, default_value_t = 100.0, conflicts_with = "at")]
    pub to: f64,

    /// Time grid step
    #[arg(long, default_value_t = 1.0, conflicts_with = "at")]
    pub every: f64,

    /// Specific time points (comma-separated; conflicts with --from/--to/--every)
    #[arg(long, value_delimiter = ',')]
    pub at: Vec<f64>,

    /// Output file (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

// ─── data split ───────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Split at a specific time point
  camdl data split cases.tsv --at-time 100 \\
      --train train.tsv --holdout holdout.tsv

  # Split at a fraction of the rows (last 20% as holdout)
  camdl data split cases.tsv --fraction 0.8 \\
      --train train.tsv --holdout holdout.tsv
"))]
pub struct DataSplitArgs {
    /// Input data TSV
    pub file: PathBuf,

    /// Split at this time value (conflicts with --fraction)
    #[arg(long, conflicts_with = "fraction")]
    pub at_time: Option<f64>,

    /// Split at this fraction of rows, 0–1 (conflicts with --at-time)
    #[arg(long, conflicts_with = "at_time")]
    pub fraction: Option<f64>,

    /// Name of the time column (auto-detected if absent)
    #[arg(long)]
    pub time_col: Option<String>,

    /// Training set output path
    #[arg(long)]
    pub train: Option<PathBuf>,

    /// Holdout set output path
    #[arg(long)]
    pub holdout: Option<PathBuf>,
}

// ─── docs ─────────────────────────────────────────────────────────────────────

/// `camdl docs [TOPIC]` — print embedded, version-locked usage guides.
#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  camdl docs                    # list available topics
  camdl docs inference          # print the inference guide
  camdl docs fit                # aliases resolve (fit -> inference)
  camdl docs --search rhat      # find where a term is discussed
  camdl docs --all              # print every guide (the full corpus)
  camdl docs --json             # machine-readable topic index

Docs are embedded in the binary: they match this version of camdl and
work offline, no checkout required."))]
pub struct DocsArgs {
    /// Topic to print (omit to list available topics)
    pub topic: Option<String>,

    /// Search all topics for a term (case-insensitive; all words must match)
    #[arg(long, short = 's', value_name = "QUERY")]
    pub search: Option<String>,

    /// Print every topic concatenated (the full corpus)
    #[arg(long)]
    pub all: bool,

    /// Print the topic index as JSON (for tools/agents)
    #[arg(long)]
    pub json: bool,
}

// ─── lineage ────────────────────────────────────────────────────────────────────

/// `camdl lineage realize EVENT_LOG --identity-seed N -o LINE_LIST` — Layer 2:
/// replay a recorded event log into a line list, drawing the identity
/// attributions (which infector, which recoverer) from the recorded per-pool
/// weights. Each `--identity-seed` is an i.i.d. draw from
/// `P(identities | event log)`. Pure offline; reads the event-log file (TSV or
/// Parquet, auto-detected by extension).
#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Realize one line list from an event log
  camdl lineage realize event_log.parquet --identity-seed 7 -o line_list.parquet

  # A second i.i.d. identity draw from the SAME epidemic
  camdl lineage realize event_log.parquet --identity-seed 8 -o line_list_2.parquet
"))]
pub struct LineageRealizeArgs {
    /// Event-log file (.tsv or .parquet). Format auto-detected by extension.
    pub event_log: PathBuf,

    /// RNG seed for the identity attribution stream. Different seeds give
    /// i.i.d. draws from P(identities | event log). Default: 1.
    #[arg(long, default_value_t = 1)]
    pub identity_seed: u64,

    /// Line-list output path (default: `line_list.<ext>`). Extension picks the
    /// format unless `--format` / `--tsv` is given.
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Line-list format: `parquet` (default) or `tsv`. Overrides the extension.
    #[arg(long, value_name = "FMT", conflicts_with = "tsv")]
    pub format: Option<String>,

    /// Shorthand for `--format tsv`.
    #[arg(long)]
    pub tsv: bool,
}

/// `camdl lineage tree LINE_LIST [...]` — project a line list to a sampled
/// transmission tree (Newick). Pure offline; reads the line-list file (TSV or
/// Parquet, auto-detected by extension) and emits Newick.
#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Flat 10% sampling over ALL individuals, write Newick
  camdl lineage tree line_list.parquet --scheme flat:0.1 --output tree.newick

  # TSV line list, sample everyone (rate 1.0)
  camdl lineage tree line_list.tsv --output tree.newick

  # Per-deme rates: deme 0 sampled at 0.5, deme 1 at 0.05, rest at 0.1
  camdl lineage tree line_list.tsv --scheme stratified:0=0.5,1=0.05,default=0.1 --output tree.newick
"))]
pub struct LineageTreeArgs {
    /// Line-list file (.tsv or .parquet). Format auto-detected by extension.
    pub line_list: PathBuf,

    /// Sampling scheme over **all** individuals (an infector can be a tip).
    /// A sampled individual's tip is placed at its removal time (or the
    /// simulation horizon if it was never removed). Supported:
    ///   - `flat:RATE` — each individual sampled i.i.d. with probability RATE
    ///     (e.g. `flat:0.1`).
    ///   - `stratified:idx=rate,...,default=rate` — each individual sampled at
    ///     its deme's rate (integer deme index), falling back to `default`
    ///     (e.g. `stratified:0=0.5,1=0.05,default=0.1`). Stratum *names* and
    ///     rates-as-parameters via a `lineage { sampling }` model block are a
    ///     future milestone; this is the projection-time path keyed on the deme
    ///     index.
    /// Default: `flat:1.0` (sample everyone).
    #[arg(long, default_value = "flat:1.0")]
    pub scheme: String,

    /// Newick output path (required).
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// RNG seed for the sampling draw (default: 1).
    #[arg(long, default_value_t = 1)]
    pub sample_seed: u64,
}

/// `camdl lineage sojourn LINE_LIST --compartment ID` — dwell-time distribution
/// in a compartment. Pure offline over the line list. The compartment is given
/// by its **global id** (the integer column index in the `camdl simulate`
/// trajectory, the same id the line list records in `source` / `destination`).
#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Dwell time in compartment 1 (e.g. the I compartment of an SIR)
  camdl lineage sojourn line_list.tsv --compartment 1

  # Write per-individual sojourns to a TSV
  camdl lineage sojourn line_list.parquet --compartment 1 --output sojourn.tsv
"))]
pub struct LineageSojournArgs {
    /// Line-list file (.tsv or .parquet). Format auto-detected by extension.
    pub line_list: PathBuf,

    /// Global compartment id whose dwell-time distribution to compute. This is
    /// the integer compartment index (matching the line list's source /
    /// destination columns), not the compartment name.
    #[arg(long)]
    pub compartment: usize,

    /// Output TSV path for the per-individual sojourns (required). A summary
    /// (count, censored, mean, quantiles) is always printed to stderr.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// `camdl lineage cohort LINE_LIST --event infection` — per-time-window event
/// summary (incidence + cumulative). Pure offline over the line list.
#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Infection incidence in 7-day windows
  camdl lineage cohort line_list.tsv --event infection --window 7

  # Events of a specific transition id, daily windows, to a file
  camdl lineage cohort line_list.parquet --event 2 --window 1 --output cohort.tsv
"))]
pub struct LineageCohortArgs {
    /// Line-list file (.tsv or .parquet). Format auto-detected by extension.
    pub line_list: PathBuf,

    /// Which events to count. `infection` counts all transmission (lineage)
    /// events — identifiable from the line list with no model. Alternatively a
    /// transition id (integer) counts events of that transition.
    #[arg(long, default_value = "infection")]
    pub event: String,

    /// Time-window width (default: 1.0).
    #[arg(long, default_value_t = 1.0)]
    pub window: f64,

    /// Align each cohort window to its first matching event instead of t=0
    /// (the default is t=0, the model origin).
    #[arg(long)]
    pub align_first_event: bool,

    /// Output TSV path (required).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

// ─── browse ───────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Show the most recent cached runs and fits
  camdl list

  # Filter by model, scenario, or recency
  camdl list --model sir
  camdl list --scenario baseline
  camdl list --since 1h

  # Only simulate runs or only fits
  camdl list --kind sim
  camdl list --kind fit

  # Machine-readable JSON
  camdl list --format json
"))]
pub struct ListArgs {
    /// Root directory to scan, as a positional: `camdl list [ROOT]`
    /// (default: ./results, or $CAMDL_OUTPUT_DIR). You may also pass it as
    /// `--root DIR`, matching `camdl cat`/`show`.
    #[arg(default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub root: PathBuf,

    /// Root directory to scan (alias for the positional ROOT, for consistency
    /// with `camdl cat`/`show`). Wins over the positional when both are given.
    #[arg(long = "root", value_name = "DIR")]
    pub root_flag: Option<PathBuf>,

    /// Filter by model path substring
    #[arg(long)]
    pub model: Option<String>,

    /// Filter by scenario name
    #[arg(long)]
    pub scenario: Option<String>,

    /// Show only runs created within this duration (e.g. 1h, 30m, 2d)
    #[arg(long)]
    pub since: Option<ListDuration>,

    /// Filter by run kind: sim, fit, profile, pfilter, survey, ensemble, or
    /// all (default).
    #[arg(long, default_value = "all")]
    pub kind: String,

    /// Filter by parent run hash (e.g. the grid-point × start children
    /// of a specific `profile` run). Matches on `parent_profile_hash`
    /// in each run's metadata. Accepts short prefixes (8+ chars).
    #[arg(long, value_name = "HASH")]
    pub parent: Option<String>,

    /// Maximum number of results to display
    #[arg(long, default_value_t = 50, conflicts_with = "all")]
    pub limit: usize,

    /// Show all results (no limit)
    #[arg(long)]
    pub all: bool,

    /// Output format: human (default) or json
    #[arg(long)]
    pub format: Option<String>,
}

impl ListArgs {
    /// The store root to scan: `--root DIR` if given, else the positional
    /// ROOT (which itself defaults to ./results / $CAMDL_OUTPUT_DIR). Lets
    /// `camdl list` accept the same `--root` as `cat`/`show` without losing
    /// the documented `camdl list [ROOT]` positional form.
    pub fn resolved_root(&self) -> &std::path::Path {
        self.root_flag.as_deref().unwrap_or(&self.root)
    }
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Resolve a run by short hash prefix
  camdl show abc1234

  # Path to a stage directory also works
  camdl show results/fits/sir-8a3f12b4/refine

  # JSON output for scripting
  camdl show abc1234 --format json
"))]
pub struct ShowArgs {
    /// Short hash prefix or path to run directory
    pub target: String,

    /// Root output directory to search (default: ./results)
    #[arg(long, default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub root: PathBuf,

    /// Output format: human (default) or json
    #[arg(long)]
    pub format: Option<String>,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Emit the trajectory for a cached run
  camdl cat abc1234

  # Select a particular observation stream
  camdl cat abc1234 --stream weekly_cases
"))]
pub struct CatArgs {
    /// Short hash prefix or path to run directory
    pub target: String,

    /// Root output directory to search (default: ./results)
    #[arg(long, default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub root: PathBuf,

    /// Observation stream name (when run has multiple streams)
    #[arg(long)]
    pub stream: Option<String>,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Rebuilds the derived index (`<root>/index.json`) from a fresh walk of every
`run.json` under the store. The index is only a cache that accelerates `list`,
`show`, and `cat`; `run.json` is always the source of truth, so `reindex` is
optional — it is useful after copying a `results/` tree, or to drop entries for
leaves that were removed out of band.

Examples:
  # Rebuild the index for the default ./results store
  camdl dev reindex

  # Rebuild the index for a specific store root
  camdl dev reindex /data/runs
"))]
pub struct ReindexArgs {
    /// Root directory to scan (default: ./results)
    #[arg(default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub root: PathBuf,
}

// ─── compare ──────────────────────────────────────────────────────────────────

/// `camdl compare` — multi-model prequential comparison table.
///
/// Takes ≥2 prequential.json files / stage dirs (or fit handles, whose
/// prequential is auto-derived) or a compare.toml, and renders a
/// baseline-centered comparison.
/// See docs/dev/proposals/2026-04-20-prequential-evaluation.md §8.
#[derive(Args)]
#[command(after_help = colored_help!("\
Columns:
  T_score    Number of scored observations (after the t0 burn-in).
             Differs across fits if they were evaluated on different data
             horizons — Δ columns are suppressed in that case unless
             --allow-mismatched-horizon is passed.
  elpd       Expected log predictive density, summed across scored
             steps:  Σ_t log p̂(y_t | y_{1:t-1}). Higher = better.
  Δelpd      elpd(this) − elpd(baseline). Positive = this model beats
             the baseline. Paired over the same observations.
  E_T        exp(Δelpd). The terminal e-value / Bayes factor vs baseline
             (Shafer 2021): a bettor who started with $1 and wagered
             this model's predictive against baseline's would end with
             $E_T. Values < 1 favour the baseline; > 1 favour this
             model. Order-of-magnitude intuition: E_T ≈ 10 is 'strong
             evidence', ≈ 100 'very strong', ≈ 1000 'decisive'
             (Jeffreys scale applied to the e-value as a Bayes factor).
             Valid even at small T where se(Δ) is unreliable.
  se(Δ)      Paired standard error of Δelpd from pointwise differences:
             √(T · Var_t(ℓ^A_t − ℓ^B_t))  (Vehtari/Gelman/Gabry).
             Rule of thumb: |Δelpd| > 2·se → 'the gap is real';
             smaller → inconclusive on this data alone.
  crps       Mean Continuous Ranked Probability Score across scored
             steps. Lower = sharper predictive, correctly calibrated.
  Δcrps      Mean CRPS difference (this − baseline). Negative = this
             model's predictive is sharper-at-the-observation.
  PIT_cov90  Fraction of observations whose probability integral
             transform fell in the central 90% predictive interval.
             Nominal 0.90 under correct calibration. < 0.70 triggers
             an overconfidence warning below the table.

Examples:
  # Compare two fits by prequential scores (table output)
  camdl compare fits/det/pfilter fits/stoch/pfilter --baseline det

  # Three-way, markdown output for pasting into a paper
  camdl compare fits/a/pf fits/b/pf fits/c/pf --format md

  # Reproducible preset via compare.toml
  camdl compare --config compare.toml

  # Compare two sealed fits by handle — the prequential is auto-derived
  # at θ̂ via `camdl pfilter` (same particles/seed for both, so the
  # scores are commensurable). No pre-run pfilter needed.
  camdl compare @baseline @candidate --particles 2000 --seed 7

  # Render despite different T_score across fits (Δ columns → '—')
  camdl compare fits/a/pf fits/b/pf --allow-mismatched-horizon
"))]
pub struct CompareArgs {
    /// Models to compare — need ≥2 when --config is not used. Each is
    /// either a prequential.json (or a stage dir holding one), read
    /// as-is, OR a fit handle (@label / hash prefix / run dir / fit.toml),
    /// whose prequential is auto-derived from its sealed θ̂ + data.
    pub paths: Vec<String>,

    /// Print the model-comparison methods guide and exit — what elpd, LR,
    /// se(Δ), the Jeffreys tiers, the within-noise gate and the conditioning
    /// modes mean, with citations. The same text as
    /// `camdl docs model-comparison`, and the page the table's footer points
    /// at. Takes no model arguments; given alongside them it still only
    /// prints the guide and runs no comparison.
    #[arg(long)]
    pub explain: bool,

    /// compare.toml with [[model]] entries (baseline/metrics/format
    /// also loadable from the file)
    #[arg(long)]
    pub config: Option<String>,

    /// Reference model for Δ columns (default: argmax elpd)
    #[arg(long)]
    pub baseline: Option<String>,

    /// Metrics to display (comma-separated: elpd, crps, pit_cov90)
    #[arg(long = "metric", alias = "metrics")]
    pub metrics: Option<String>,

    /// Output format: table (default), md, json
    #[arg(long, default_value = "table")]
    pub format: String,

    /// Render even if T_score differs across models (Δ columns → '—')
    #[arg(long)]
    pub allow_mismatched_horizon: bool,

    /// Render even if the compared fits are bound to different observed data
    /// (gh#713). By default, two fits whose `fit.meta.json` records different
    /// content hashes for a shared observation stream are refused: Δelpd would
    /// then mix a model difference with a data difference, and nothing in the
    /// table would say so. Use this only when the difference is understood and
    /// intended — comparing a fit on revised case counts against the fit on the
    /// original ones, say — and read the Δ as confounded. Rows given as an
    /// explicit `prequential.json` carry no data identity and are never part of
    /// the check.
    #[arg(long)]
    pub allow_data_mismatch: bool,

    /// Mixture size for a Bayesian fit's derived predictive (§3.6 of the
    /// 2026-08-29 proposal): one filter pass per thinned posterior draw;
    /// per-step predictive densities are averaged over the draws and the
    /// predictive samples pooled, so scores and intervals carry parameter
    /// uncertainty (provenance `posterior`). `--draws 1` is the documented
    /// cheap mode — the plug-in predictive at the posterior mean, as
    /// before. An optimizer fit (no posterior cloud) always scores
    /// plug-in at its winner. Applied uniformly across derived fits.
    #[arg(long, default_value_t = crate::compare::DEFAULT_DERIVE_DRAWS)]
    pub draws: usize,

    /// Filter replicates per derived fit (§3.4 of the 2026-08-29
    /// proposal): the whole derivation reruns at seeds seed..seed+R,
    /// per-step scores combine by log-mean-exp, and the replicate totals
    /// give each row a filter-noise Monte-Carlo SE (`mc_se_elpd`). The
    /// evidence verdict is suppressed ("within filter noise") when |Δelpd|
    /// sits inside twice the pair's combined MC SE — at that scale the Δ
    /// measures the particle filters, not the models. `--replicates 1` is
    /// the documented cheap mode (no replication, no MC SE).
    #[arg(long, default_value_t = crate::compare::DEFAULT_DERIVE_REPLICATES)]
    pub replicates: usize,

    /// Render rows whose provenance kinds differ (plug-in vs posterior
    /// mixture) instead of refusing: the Δ then compares an
    /// under-dispersed single-θ predictive against a mixture, and the
    /// reader owns that confound. The usual cause is comparing an
    /// optimizer fit (no posterior cloud) against a Bayesian one.
    #[arg(long)]
    pub allow_mixed_provenance: bool,

    /// Force in-sample derivation even when every compared fit declares a
    /// holdout (gh#585): the traces score the full series at θ̂ as before
    /// this flag existed, stamped `in_sample` and carrying the in-sample
    /// optimism caveat. Without it, holdout-declaring fits are scored
    /// held-out (`--score-from` at the sealed training boundary) and
    /// stamped `hold_out_tail` after the non-leakage verification.
    #[arg(long)]
    pub in_sample: bool,

    /// Particle count for any fit handle whose prequential is
    /// auto-derived. Applied uniformly to every derived fit so T_score
    /// and scores stay commensurable. Ignored for an explicit
    /// prequential.json path (read as-is).
    #[arg(long, default_value_t = crate::compare::DEFAULT_DERIVE_PARTICLES)]
    pub particles: usize,

    /// Filter seed for any fit handle whose prequential is auto-derived.
    /// Applied uniformly across derived fits. Ignored for an explicit
    /// prequential.json path (read as-is).
    #[arg(long, default_value_t = crate::compare::DEFAULT_DERIVE_SEED)]
    pub seed: u64,

    /// Drop MCMC chains from a fit's posterior cloud before deriving its plug-in
    /// θ̂, so a comparison scores the SAME subset `fit predict`/`fit summary`
    /// would band. PER-FIT (repeat the flag): `--exclude-chains @a:4` drops
    /// chain 4 from the fit named `@a` only, leaving the others whole. The fit
    /// name is matched VERBATIM against the name shown in the table (and matched
    /// by `--baseline`): a fit given by run-store handle is `@a`, one given by
    /// path is e.g. `ctl_rm.toml` with no `@` — do not add a spurious `@`. Bare
    /// ids `--exclude-chains 3,4` apply COHORT-WIDE to every fit (convenient
    /// only when the fits share a stuck-chain index — otherwise use the per-fit
    /// form). Chain ids are 1-based (matching the `chain_N/` dirs and the `fit
    /// summary` per-chain table). Mixing bare and `@fit:ids` tokens is rejected;
    /// a fit with no posterior cloud (an optimizer fit, or an explicit
    /// prequential.json) ignores the flag. Post-hoc exclusion BIASES the
    /// posterior toward the retained mode and always prints a warning. A chain
    /// id not in a fit, an unknown/ambiguous fit name, or excluding every chain
    /// is a hard error.
    #[arg(long, value_name = "[@FIT:]IDS")]
    pub exclude_chains: Vec<String>,

    /// Write the per-observation Δelpd vector to PATH as a TSV (gh#706).
    ///
    /// `Δelpd = 12 nats` says a model won; this says WHERE it won — on three
    /// weeks around an intervention, on one district, on a single reporting
    /// batch. One row per candidate × scored step, joint and per stream, with
    /// the candidate's log score, the baseline's, and their difference. The
    /// quantity is already computed to form `se(Δelpd)`; this stops discarding
    /// it. The natural consumer is a plot faceted by stream.
    ///
    /// The per-step PIT is deliberately NOT published here until gh#629 (tie
    /// bias in the PIT estimator) is fixed.
    #[arg(long, value_name = "PATH")]
    pub pointwise: Option<PathBuf>,
}

// ─── mre (minimal-reproducible-example bundles) ──────────────────────────────

/// `camdl mre fit <fit.toml>` — package a fit's full input closure (model, the
/// model's compile-time `read()` files, data, fixed params) into a `.tar.gz`
/// so a bug can be reproduced from one file. See
/// `docs/dev/proposals/2026-06-09-mre-bundle.md`.
#[derive(Args)]
pub struct MreFitArgs {
    /// Fit configuration file (fit.toml) to bundle.
    pub config: PathBuf,

    /// Output bundle path. Defaults to `<config-stem>.mre.tar.gz`.
    /// Short flag is `-b` (NOT `-o`: `simulate` already owns `-o`).
    #[arg(short = 'b', long = "bundle", value_name = "FILE")]
    pub bundle: Option<PathBuf>,

    /// Exclude observed data values — emit a structure-only bundle (column
    /// schema, row counts, time range; no values) for when the data is
    /// sensitive. Default includes the data with a prominent banner.
    #[arg(long)]
    pub no_data: bool,
}

/// `camdl mre simulate <model.camdl> [sim flags…]` — bundle a forward-sim
/// reproduction. Flattens the real `SimulateArgs` so every simulate flag
/// parses identically; the bundle output is `-b` (simulate's own `-o` keeps
/// its trajectory-output meaning). No `--no-data`: a forward sim has no observed
/// data, and its tables/params can't be dropped without breaking the run.
#[derive(Args)]
#[command(after_help = colored_help!("\
Bundles a model + its read() tables + params into a shareable `.tar.gz` that
reproduces a forward simulation. Takes every `camdl simulate` flag; `-b/--bundle`
names the output (`-o` keeps its trajectory-mirror meaning).

Examples:
  # Bundle a forward-sim reproduction (default name <model>.mre.tar.gz)
  camdl mre simulate sir.camdl --params p.toml --seed 42

  # Name the bundle explicitly
  camdl mre simulate sir.camdl --params p.toml --seed 42 -b sir-repro.mre.tar.gz
"))]
pub struct MreSimulateArgs {
    #[command(flatten)]
    pub sim: SimulateArgs,

    /// Output bundle path. Defaults to `<model-stem>.mre.tar.gz`.
    #[arg(short = 'b', long = "bundle", value_name = "FILE")]
    pub bundle: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, Parser};
    use crate::{Cli, Command, FitCmd};

    fn try_parse_fit_run(args: &[&str]) -> Result<FitRunArgs, clap::Error> {
        let mut full: Vec<&str> = vec!["camdl", "fit", "run"];
        full.extend(args);
        let cli = Cli::try_parse_from(full)?;
        match cli.command {
            Command::Fit(FitCmd::Run(a)) => Ok(a),
            _ => unreachable!("expected fit run"),
        }
    }

    // ── M-1 break: --params / --param / --starts-from / --init-method
    //    are accepted by clap (so the error is actionable) but trapped
    //    in pre-dispatch checks. These tests assert the parse layer
    //    accepts the trap (hidden flags) so the dispatch site can emit
    //    its actionable error. The corresponding end-to-end assertion
    //    that the actionable error fires lives in
    //    `tests/cas_integration.rs::starts_from_resolves_short_hash`.

    #[test]
    fn profile_params_lands_in_init_params_not_trap() {
        // Regression: post-fix, `profile --params <PATH>` parses into
        // ProfileArgs::init_params (the legitimate companion to
        // `--init from_params`), NOT into a trap field. The earlier
        // version of this test asserted the opposite — the trap then
        // shadowed the init-mode companion and made
        // `profile --init from_params --params start.toml`
        // unreachable. The actionable error for bare `--params` (no
        // `--init from_params`) now fires from
        // `InitModeTag::to_init_method` instead of a parse-time trap.
        let full = ["camdl", "profile", "model.camdl",
                    "--data", "cases.tsv",
                    "--sweep", "R0=lin(0.5,5,5)",
                    "--rw-sd", "auto",
                    "--particles", "100",
                    "--params", "truth.toml"];
        let parsed = Cli::try_parse_from(full)
            .expect("clap must accept --params on profile (lands in init_params)");
        match parsed.command {
            Command::Profile(a) => {
                assert_eq!(a.init_params.as_deref().map(|p| p.to_string_lossy().into_owned()),
                    Some("truth.toml".to_string()),
                    "expected --params to land in init_params");
                assert_eq!(a.model_overrides.fixed_cli.len(), 0,
                    "--params must not pollute fixed_cli");
            }
            _ => unreachable!(),
        }
    }

    /// The valid usage: `--init from_params --params <path>` must
    /// parse AND `to_init_method` must build `InitMethod::FromParams`.
    /// Pre-fix, the trap field shadowed init_params and the user got
    /// a rejection error.
    #[test]
    fn profile_init_from_params_with_params_companion_parses_and_builds() {
        let full = ["camdl", "profile", "model.camdl",
                    "--data", "cases.tsv",
                    "--sweep", "R0=lin(0.5,5,5)",
                    "--rw-sd", "auto",
                    "--particles", "100",
                    "--init", "from_params",
                    "--params", "/tmp/start.toml"];
        let parsed = Cli::try_parse_from(full)
            .expect("clap must accept --init from_params --params <path>");
        let Command::Profile(a) = parsed.command else { unreachable!() };
        assert_eq!(a.init, InitModeTag::FromParams);
        assert_eq!(a.init_params.as_deref().map(|p| p.to_string_lossy().into_owned()),
            Some("/tmp/start.toml".to_string()));
        // Verify to_init_method assembles the typed InitMethod.
        let im = a.init.to_init_method(
            a.posterior.as_ref(), a.mle.as_ref(), a.init_params.as_ref(),
        ).expect("to_init_method must succeed for --init from_params --params <path>");
        match im {
            crate::fit::init::InitMethod::FromParams { path } => {
                assert_eq!(path.to_string_lossy(), "/tmp/start.toml");
            }
            other => panic!("expected InitMethod::FromParams, got {:?}", other),
        }
    }

    /// `--params <path>` without `--init from_params` must surface the
    /// actionable "use --fixed-file or --init from_params" error from
    /// `to_init_method`. This is the migration-friendly version of the
    /// pre-fix parse-time trap.
    #[test]
    fn profile_params_without_init_from_params_errors_from_to_init_method() {
        let full = ["camdl", "profile", "model.camdl",
                    "--data", "cases.tsv",
                    "--sweep", "R0=lin(0.5,5,5)",
                    "--rw-sd", "auto",
                    "--particles", "100",
                    "--params", "truth.toml"];
        let parsed = Cli::try_parse_from(full).unwrap();
        let Command::Profile(a) = parsed.command else { unreachable!() };
        // Default init is Lhs; `--params` without `--init from_params`
        // must produce a structured error from to_init_method.
        let err = a.init.to_init_method(
            a.posterior.as_ref(), a.mle.as_ref(), a.init_params.as_ref(),
        ).expect_err("to_init_method must reject --params without --init from_params");
        assert!(err.contains("--params is only valid with --init from_params"),
            "error must point user at --init from_params: {}", err);
    }

    /// `camdl if2` is removed (gh#147); its arg struct is a catch-all so
    /// an old invocation still PARSES and reaches the deprecation message
    /// in `if2::cmd_if2`, rather than dying on a clap "unexpected
    /// argument" error before the migration hint can be shown.
    #[test]
    fn if2_is_a_deprecation_catch_all() {
        let full = ["camdl", "if2", "model.camdl",
                    "--data", "cases.tsv",
                    "--rw-sd", "auto",
                    "--particles", "100",
                    "--regime", "scout",
                    "--params", "truth.toml"];
        let parsed = Cli::try_parse_from(full)
            .expect("camdl if2 must still parse so the stub can show the migration message");
        match parsed.command {
            Command::If2(a) => {
                assert!(!a._ignored.is_empty(),
                    "old if2 args must be swallowed into the catch-all");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn fit_run_starts_from_flag_is_trapped_at_parse() {
        // Mirrors profile_params_flag_is_trapped_at_parse for the
        // `--starts-from` removal on `camdl fit run`.
        let a = try_parse_fit_run(&[
            "fit.toml", "--stage", "refine",
            "--starts-from", "fits/scout/",
        ]).expect("hidden trap must accept --starts-from");
        assert_eq!(a._removed_starts_from.as_deref(), Some("fits/scout/"));
    }

    #[test]
    fn fit_run_init_method_alias_is_trapped_at_parse() {
        // `--init-method` was renamed to `--init`. The trap collects
        // the old spelling so the dispatch site can emit the
        // actionable rename error.
        let a = try_parse_fit_run(&[
            "fit.toml", "--stage", "scout",
            "--init-method", "lhs",
        ]).expect("hidden trap must accept --init-method");
        assert_eq!(a._removed_init_method.as_deref(), Some("lhs"));
        assert!(a.init.is_none(),
            "trapped --init-method must not populate the new --init field");
    }

    #[test]
    fn fit_run_init_flag_parses_modes() {
        // The renamed `--init` flag accepts every payload-free
        // `InitModeTag` variant via clap's value_enum.
        for mode in ["single", "uniform", "lhs", "from_prior",
                     "from_posterior", "from_mle", "from_params",
                     "survey_top_k"] {
            let a = try_parse_fit_run(&[
                "fit.toml", "--stage", "scout",
                "--init", mode,
            ]).unwrap_or_else(|e|
                panic!("--init {} must parse: {}", mode, e));
            assert!(a.init.is_some(),
                "--init {} must populate the field", mode);
        }
    }

    #[test]
    fn fit_run_init_method_alias_does_not_resolve() {
        // The trap is wired specifically — `--init-method` lives on
        // `_removed_init_method`, not the new `init` field, so the
        // dispatch's check_removed-flag style emit fires correctly.
        let a = try_parse_fit_run(&[
            "fit.toml", "--stage", "scout",
            "--init-method", "from_prior",
        ]).expect("hidden trap must accept --init-method <mode>");
        assert!(a.init.is_none());
        assert_eq!(a._removed_init_method.as_deref(), Some("from_prior"));
    }

    // gh#189: --loglik-eval-particles/-reps removed — loglik_eval is part of the
    // fit identity (set only in the stage TOML), not a CLI override that bypasses
    // the run_id. The convergence-gate override (--decibans-thresh) stays.
    #[test]
    fn fit_run_gate_override_parses_with_stage() {
        let a = try_parse_fit_run(&[
            "fit.toml", "--stage", "scout", "--decibans-thresh", "60.0",
        ]).expect("should parse with --stage");
        assert_eq!(a.decibans_thresh, Some(60.0));
        assert_eq!(a.stage.as_deref(), Some("scout"));
    }

    #[test]
    fn fit_run_decibans_thresh_requires_stage() {
        let err = try_parse_fit_run(&[
            "fit.toml", "--decibans-thresh", "60.0",
        ]).err().expect("should reject without --stage");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn fit_run_gate_override_default_is_none() {
        let a = try_parse_fit_run(&["fit.toml"]).unwrap();
        assert!(a.decibans_thresh.is_none());
    }

    #[test]
    fn cli_command_factory_builds() {
        // Smoke test — guards against malformed clap derives that would
        // panic at runtime instead of producing a parse error.
        let _ = Cli::command();
    }

    /// Regression (gh#183): the `simulate --backend` help text must name
    /// the *resolved* default. `camdl simulate` resolves an omitted
    /// `--backend` to `ForwardBackend::ChainBinomial` (main.rs:
    /// `a.backend.backend.unwrap_or(ForwardBackend::ChainBinomial)`), so the help
    /// string claiming `gillespie` was stale doc-vs-code drift dating to
    /// before the 2026-04-19 backend-default-mismatch fix moved the
    /// simulate default to chain_binomial. Pin the help to the code-true
    /// default so it can't silently drift again.
    #[test]
    fn simulate_backend_help_names_resolved_default() {
        // The resolved default, straight from the enum the resolver uses.
        let resolved_default = ForwardBackend::ChainBinomial.as_str(); // "chain_binomial"

        let mut cmd = Cli::command();
        let simulate = cmd
            .find_subcommand_mut("simulate")
            .expect("simulate subcommand exists");
        let backend_arg = simulate
            .get_arguments()
            .find(|a| a.get_id() == "backend")
            .expect("simulate has a --backend arg");
        let help = backend_arg
            .get_help()
            .map(|h| h.to_string())
            .unwrap_or_default();

        assert!(
            help.contains(resolved_default),
            "simulate --backend help must name the resolved default \
             ({resolved_default}); got: {help:?}"
        );
        assert!(
            !help.contains("gillespie"),
            "simulate --backend help must not claim the stale `gillespie` \
             default (resolved default is {resolved_default}); got: {help:?}"
        );
    }

    /// Regression: writer-side `DEFAULT_OUTPUT_ROOT` ("results") must
    /// match every reader-side CLI default. Drift here is what
    /// produced the 2026-04-19 → 2026-04-27 wart where `batch run`
    /// wrote to `./results/` but `list / show / cat` defaulted to
    /// `./output/`, forcing book chapters to pass `--root results`
    /// to every read command. Keep them in lockstep.
    #[test]
    fn reader_cli_defaults_match_default_output_root() {
        use crate::run_paths::DEFAULT_OUTPUT_ROOT;
        // Don't read CAMDL_OUTPUT_DIR from the test environment; it
        // would mask the default we're trying to assert.
        std::env::remove_var("CAMDL_OUTPUT_DIR");

        let expected = format!("./{}", DEFAULT_OUTPUT_ROOT);

        let parse_simulate = |args: &[&str]| -> SimulateArgs {
            let mut full: Vec<&str> = vec!["camdl", "simulate"];
            full.extend(args);
            match Cli::try_parse_from(full).unwrap().command {
                Command::Simulate(a) => a,
                _ => unreachable!(),
            }
        };
        let parse_list = || -> ListArgs {
            match Cli::try_parse_from(["camdl", "list"]).unwrap().command {
                Command::List(a) => a,
                _ => unreachable!(),
            }
        };
        let parse_show = |hash: &str| -> ShowArgs {
            match Cli::try_parse_from(["camdl", "show", hash]).unwrap().command {
                Command::Show(a) => a,
                _ => unreachable!(),
            }
        };
        let parse_cat = |hash: &str| -> CatArgs {
            match Cli::try_parse_from(["camdl", "cat", hash]).unwrap().command {
                Command::Cat(a) => a,
                _ => unreachable!(),
            }
        };

        // simulate --output_dir
        let s = parse_simulate(&["model.camdl"]);
        assert_eq!(s.output_dir.to_string_lossy(), expected,
            "SimulateArgs.output_dir must default to ./{}",
            DEFAULT_OUTPUT_ROOT);

        // list
        let l = parse_list();
        assert_eq!(l.root.to_string_lossy(), expected,
            "ListArgs.root must match DEFAULT_OUTPUT_ROOT");

        // show
        let sh = parse_show("abc12345");
        assert_eq!(sh.root.to_string_lossy(), expected,
            "ShowArgs.root must match DEFAULT_OUTPUT_ROOT");

        // cat
        let c = parse_cat("abc12345");
        assert_eq!(c.root.to_string_lossy(), expected,
            "CatArgs.root must match DEFAULT_OUTPUT_ROOT");
    }

    // ── fit predict: scenario overlay parsing (Layer 1) ──────────────────────

    fn parse_fit_predict(args: &[&str]) -> FitPredictArgs {
        let mut full: Vec<&str> = vec!["camdl", "fit", "predict"];
        full.extend(args);
        let cli = Cli::try_parse_from(full).expect("fit predict parse");
        match cli.command {
            Command::Fit(FitCmd::Predict(a)) => a,
            _ => unreachable!("expected fit predict"),
        }
    }

    #[test]
    fn fit_predict_no_scenario_yields_single_fitted() {
        // No `--scenario` → a single inline `fitted` overlay carrying no
        // enable/disable (the no-overlay row, byte-identical to today's
        // hardcoded baseline replay).
        let a = parse_fit_predict(&["--fit", "fit.toml"]);
        let refs = a.scenario_refs().expect("no-scenario parses");
        assert_eq!(refs.len(), 1, "exactly one scenario when none requested");
        assert_eq!(refs[0].name(), FITTED, "the no-overlay row is named fitted, not baseline");
        match &refs[0] {
            crate::sim_job::ScenarioRef::Inline { enable, disable, params, .. } => {
                assert!(enable.is_empty() && disable.is_empty() && params.is_empty(),
                    "no overlay → empty inline patch");
            }
            other => panic!("expected an inline fitted ref, got {other:?}"),
        }
    }

    #[test]
    fn fit_predict_repeated_scenario_parses_to_named_vec() {
        // Repeated `--scenario` → one `Named` ref each, in order (mirrors the
        // simulate parser). This is the proposal's `no_sia` / `with_sia` form.
        let a = parse_fit_predict(&[
            "--fit", "fit.toml",
            "--scenario", "no_sia",
            "--scenario", "with_sia",
        ]);
        let refs = a.scenario_refs().expect("repeated --scenario parses");
        let names: Vec<&str> = refs.iter().map(|r| r.name()).collect();
        // gh#625: the fitted no-overlay arm leads — it is the posterior
        // predictive every scenario overlays, and dropping it left scenario
        // deltas with no reference to delta against.
        assert_eq!(names, vec!["fitted", "no_sia", "with_sia"]);
        assert!(refs.iter().skip(1).all(|r| matches!(r, crate::sim_job::ScenarioRef::Named(_))),
            "explicit --scenario refs are Named (preset path)");
        assert!(matches!(refs[0], crate::sim_job::ScenarioRef::Inline { .. }),
            "the prepended fitted arm is the inline no-overlay ref");
    }

    #[test]
    fn fit_predict_comma_list_scenario_splits() {
        // `--scenario a,b` comma-splits, exactly like simulate.
        let a = parse_fit_predict(&["--fit", "fit.toml", "--scenario", "no_sia,with_sia"]);
        let refs = a.scenario_refs().unwrap();
        let names: Vec<String> = refs.iter().map(|r| r.name().to_string()).collect();
        // gh#625: fitted leads (see the repeated-`--scenario` test).
        assert_eq!(names, vec!["fitted".to_string(), "no_sia".to_string(),
                               "with_sia".to_string()]);
    }

    #[test]
    fn fit_predict_enable_disable_form_overlay() {
        // `--enable`/`--disable` (no `--scenario`) → a single ad-hoc `fitted`
        // overlay carrying the toggles, mirroring `simulate --enable`.
        let a = parse_fit_predict(&["--fit", "fit.toml", "--enable", "sia", "--disable", "ri"]);
        let refs = a.scenario_refs().unwrap();
        assert_eq!(refs.len(), 1);
        match &refs[0] {
            crate::sim_job::ScenarioRef::Inline { name, enable, disable, .. } => {
                assert_eq!(name, FITTED);
                assert_eq!(enable, &vec!["sia".to_string()]);
                assert_eq!(disable, &vec!["ri".to_string()]);
            }
            other => panic!("expected inline overlay, got {other:?}"),
        }
    }

    #[test]
    fn fit_predict_scenario_fitted_is_reserved() {
        // An explicit `--scenario fitted` is rejected: it collides with the
        // reserved no-overlay value. The diagnostic names the reservation + fix.
        let a = parse_fit_predict(&["--fit", "fit.toml", "--scenario", "fitted"]);
        let err = a.scenario_refs().expect_err("fitted must be rejected");
        assert!(err.contains("reserved"), "names the reservation: {err}");
        assert!(err.contains("fitted"), "names the offending value: {err}");
    }

    #[test]
    fn fit_predict_scenario_conflicts_with_enable() {
        // clap-level: --scenario and --enable are mutually exclusive (mirrors
        // simulate's σ-flag rule).
        let full = ["camdl", "fit", "predict", "--fit", "fit.toml",
                    "--scenario", "no_sia", "--enable", "sia"];
        assert!(Cli::try_parse_from(full).is_err(),
            "--scenario + --enable must be a clap conflict");
    }
}
