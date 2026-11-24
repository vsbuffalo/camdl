// ── Args structs only ─────────────────────────────────────────────────────────
//
// The clap subcommand tree (Cli, Command, FitCmd, ...) lives
// in main.rs — it's the canonical, dispatched parser. This module owns
// only the per-command argument structs (FitRunArgs, SimulateArgs, etc.)
// referenced by main.rs's tree.

pub mod types;
mod store_commands;
pub use store_commands::{CleanArgs, ReceiveArgs, StatusArgs, StoreReindexArgs, VerifyArgs};

use std::path::PathBuf;
use clap::{Args, ArgGroup};
use crate::colored_help;
use types::{ForwardBackend, DataSpec, ListDuration, ParamOverride, ParamVecSpec, RwSd, SeedSpec, SweepSpec, TableSpec};

// ─── Shared help-text constants ───────────────────────────────────────────────
//
// Per docs/dev/proposals/2026-05-25-cli-init-and-params-ux.md
// §"Help-text rewrite", the `--fixed` flag shares a single normative
// description across every inference subcommand. clap's `long_about`
// accepts it as a constant so a doc edit lands in one place instead of
// being copied per arg.

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

ON `camdl profile`

  Any name appearing in `--fixed`/`--fixed-file` is pinned at the supplied
  value AND removed from the --fit config's [estimate] set if present (so
  that `--fixed gamma=0.1 --sweep tau=lin(...)` is the canonical pattern for
  profile-likelihood slicing — hold gamma, sweep tau). One info line names
  each such parameter, the flag that pinned it and its value.

  `camdl fit run` takes no `--fixed`: a fit config is the record of the
  problem it runs, so a held parameter is a [fixed] entry in it (one line
  moved from [estimate] to [fixed] in a copy). A one-value
  `fit run --sweep NAME=V` re-values a [fixed] entry without editing it.

PRECEDENCE (last wins, per docs/camdl-run-spec.md §1.3)

  1. A value compiled into the IR (`camdlc --set`; a .camdl never writes one)
  2. fit.toml [fixed] block (if --fit is in scope)
  3. --fixed-file <toml> (layered in declared order)
  4. Scenario preset (--scenario NAME)
  5. --fixed NAME=VALUE   (highest; a --fixed on a parameter the scenario
                          sets or scales is refused)";

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
#[derive(Args, Clone, Default, store::Inputs)]
pub struct ModelOverrides {
    /// Parameter TOML file (may be repeated)
    #[arg(long, value_name = "FILE")]
    #[input(identity, vary = crate::input_fixtures::other_params)]
    pub params: Vec<PathBuf>,

    /// Single parameter override, e.g. --param R0=2.5 (may be repeated)
    #[arg(long, value_name = "NAME=VALUE")]
    #[input(identity, vary = crate::input_fixtures::other_param)]
    pub param: Vec<ParamOverride>,

    /// External table for table-lookup expressions, e.g. --table contact=matrix.tsv
    #[arg(long, value_name = "NAME=FILE")]
    #[input(identity, vary = crate::input_fixtures::other_table)]
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
#[derive(Args, Clone, Default, store::Inputs)]
pub struct InferenceModelOverrides {
    /// Pin NAME to VALUE (repeatable; sets value and removes the
    /// parameter from `[estimate]` if present). See `--help-fixed` for
    /// the full precedence / semantics block.
    #[arg(long = "fixed", value_name = "NAME=VALUE",
          long_help = FIXED_LONG_ABOUT)]
    #[input(identity, vary = crate::input_fixtures::profile_other_fixed)]
    pub fixed_cli: Vec<ParamOverride>,

    /// Load fixed values from a flat params TOML (repeatable, layered).
    /// Each top-level key = parameter name; later files override
    /// earlier ones. Listed parameters are removed from `[estimate]`
    /// if present.
    #[arg(long = "fixed-file", value_name = "TOML",
          long_help = FIXED_LONG_ABOUT)]
    #[input(identity, vary = crate::input_fixtures::profile_other_fixed_file)]
    pub fixed_files: Vec<PathBuf>,

    /// External table for table-lookup expressions, e.g. --table contact=matrix.tsv
    #[arg(long, value_name = "NAME=FILE")]
    #[input(identity, vary = crate::input_fixtures::other_table)]
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
    #[input(skip)]
    pub _removed_param: Vec<ParamOverride>,
}

impl InferenceModelOverrides {
    /// Emit an actionable error and abort if a removed flag was used.
    /// Called by every inference-subcommand dispatch function before
    /// any other work. Matches the wording in the proposal §"Migration".
    pub fn check_removed_flags(&self, subcmd: &str) {
        if !self._removed_param.is_empty() {
            crate::out::detached().fail(format!(
                "error: --param is no longer accepted on `camdl {}`. \
                 Replacement:\n  \
                 --fixed NAME=VALUE             (set & freeze a single value)\n  \
                 --fixed-file <toml>            (load fixed values from a TOML file)\n\
                 See `camdl {} --help` (SET PARAMETER VALUES section).",
                subcmd, subcmd));
        }
    }
}

/// `--scenario` XOR `--enable`/`--disable`
#[derive(Args, Clone, Default, store::Inputs)]
pub struct ScenarioArgs {
    /// Named scenario defined in the model (conflicts with --enable/--disable)
    #[arg(long, conflicts_with_all = ["enable", "disable"])]
    #[input(identity, vary = crate::input_fixtures::other_scenario)]
    pub scenario: Option<String>,

    /// Enable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenario")]
    #[input(identity, vary = crate::input_fixtures::an_intervention)]
    pub enable: Vec<String>,

    /// Disable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenario")]
    #[input(identity, vary = crate::input_fixtures::an_intervention)]
    pub disable: Vec<String>,
}

/// `--backend` + `--dt`
#[derive(Args, Clone, store::Inputs)]
pub struct SimBackend {
    /// Simulation backend (default: chain_binomial)
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_backend)]
    pub backend: Option<ForwardBackend>,

    /// The integration step of the chain-binomial backend and the fixed-step
    /// ODE integrator. Default: the model's `simulate { dt }`, else 1.0; a
    /// value given here replaces the model's step. A replay of a fit
    /// (`--draws … --fit`, or a `--params` file a fit wrote) defaults to the
    /// fit's step and runs a different one with a note. The commands that fit or score data refuse a
    /// step that differs from the model's (run spec §1.3.1).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_dt)]
    pub dt: Option<f64>,

    /// The ODE integrator: rk4 (fixed step) or rk45 (adaptive), replacing
    /// the model's method and keeping its rk45 tolerances, which are set in
    /// `simulate { integrator = rk45 { atol = .., rtol = .. } }`. Refused on
    /// a backend other than ode.
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_integrator)]
    pub integrator: Option<crate::args::types::IntegratorArg>,
}

/// `--output-every` / `--no-flows` / `--columns` — the trajectory output view
/// (cadence + which columns are written), flattened into `SimulateArgs`. Only
/// `simulate` writes trajectories, so the view applies there; `fit.toml` has
/// no trajectory output.
///
/// `every` overrides the model's `output { every }` schedule, so it rides the
/// model digest and re-keys only runs that use it. `no_flows` / `columns`
/// change which columns are written to the content-addressed leaf, so they ride
/// identity of a simulation cell's settings: a non-default view
/// is a distinct, reproducible artifact.
#[derive(Args, Clone, Debug, Default, store::Inputs)]
pub struct OutputView {
    /// Emit one trajectory row every N time-units, overriding the model's
    /// `output { every }`. A plain number in the model's `time_unit` (like
    /// `--dt`); e.g. `--output-every 7` on a daily model writes weekly rows.
    #[arg(long = "output-every", value_name = "N")]
    #[input(identity, vary = crate::input_fixtures::other_every)]
    pub every: Option<f64>,

    /// Drop every `flow_*` column from the trajectory output — most useful for
    /// stratified/spatial models where inter-stratum flow columns dominate.
    #[arg(long = "no-flows", default_value_t = false)]
    #[input(identity)]
    pub no_flows: bool,

    /// Restrict trajectory columns to this comma-separated allow-list of output
    /// column names (compartments and/or `flow_<name>`). Empty = all columns;
    /// emitted order follows the model, not this list.
    #[arg(long = "columns", value_delimiter = ',', value_name = "COL,...")]
    #[input(identity, vary = crate::input_fixtures::other_columns)]
    pub columns: Vec<String>,
}

/// The override for a forcing evaluated outside the range it is defined on,
/// shared by every command that runs a model.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct ForcingExtrapolationArg {
    /// Let the run evaluate a forcing outside the range it is defined on —
    /// an `interpolated` forcing before its first or after its last data
    /// point, a `bspline` outside its window — holding its end value there,
    /// with one warning per forcing naming its range and the span held, and
    /// the override recorded in `run.json`. Without it such a run is an
    /// error. A fit config sets it as `[config] allow_forcing_extrapolation =
    /// true`.
    #[arg(long, default_value_t = false)]
    #[input(identity)]
    pub allow_forcing_extrapolation: bool,
}

/// The inference knobs of `camdl profile`. The same flags as
/// [`PfilterCore`], declared apart because `--pf-max-substeps` is an identity
/// input of a profile point (a tripped budget is a stored log-likelihood of
/// -inf) and an execution input of a filter run (a tripped budget stores
/// nothing).
#[derive(Args, Clone, store::Inputs)]
pub struct ProfileCore {
    /// The integration step. Default: the model's `simulate { dt }`, else the
    /// --fit file's `[config] dt`, else 1.0. A value that differs from the
    /// model's step or the --fit file's is refused: a command that fits or
    /// scores data integrates at the model's step (run spec §1.3.1).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_dt)]
    pub dt: Option<f64>,

    /// RNG seed: draws each cell's starts, and with `--seeds` absent it is
    /// the one seed every point is computed under.
    #[arg(long, default_value_t = 1)]
    #[input(identity)]
    pub seed: u64,

    /// Rayon thread count (0 = all available cores)
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    #[input(execution, exercised_by = crate::input_fixtures::profile_parallel)]
    pub parallel: usize,

    /// Restore the legacy silent-zero behaviour for numerical-collapse paths
    /// (division by zero, a power that is NaN, the square root of a negative)
    /// in rate evaluation. Default: hard error. Use only when a model needs it,
    /// such as a spatial model where an empty stratum's divisor is meant to
    /// give a zero rate.
    #[arg(long, default_value_t = false)]
    #[input(identity)]
    pub allow_degenerate_rates: bool,

    #[command(flatten)]
    #[input(group)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

    /// Time-column format for `--data`: `auto` (detect numeric-vs-date over
    /// the whole column), `numeric` (force `f64`), or `date` (force ISO
    /// dates, requires the model's `origin`). A format the column does not
    /// hold is refused.
    #[arg(long, default_value = "auto")]
    #[input(execution, exercised_by = crate::input_fixtures::profile_time_format)]
    pub time_format: crate::caltime_load::TimeFormat,

    /// The maximum cumulative particle-substep count one filter evaluation
    /// of an IF2 cell may execute (`PFIterationBudget`). Unset: the engine
    /// default (1e10). A cell whose budget trips is stored with a
    /// log-likelihood of -inf, so the budget is part of each point's
    /// identity. Refused for an nl-* cell.
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub pf_max_substeps: Option<u64>,

    /// Binomial sampler for the chain-binomial draws: `btpe` (default) or
    /// `btrs` (Hörmann 1993; faster). The two are not bit-compatible, so a
    /// `btrs` point is a result of its own. Refused for an nl-* cell.
    #[arg(long, value_name = "SAMPLER",
          value_parser = |v: &str| v.parse::<sim::rng::BinomialAlgorithm>())]
    #[input(identity, vary = crate::input_fixtures::other_binomial)]
    pub binomial: Option<sim::rng::BinomialAlgorithm>,
}

/// Stream selection for the inference commands. The projection and
/// likelihood for every stream come from the model's `observations { }` block
/// (the modern observation system, as `fit run` uses) — there is no `--flow` /
/// `--obs-model` projection override. `--obs` only selects WHICH declared
/// stream (family) a single positional `--data FILE` binds to.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct StreamSelection {
    /// Observation stream/family name a bare `--data FILE` binds to
    /// (required when the model declares more than one).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_stream)]
    pub obs: Option<String>,
}

/// The inference knobs of `camdl pfilter`. The same flags as
/// [`ProfileCore`], declared apart because `--pf-max-substeps` is an execution
/// input of a filter run (a tripped budget stores nothing) and an identity
/// input of a profile point (a tripped budget is a stored log-likelihood of
/// -inf).
#[derive(Args, Clone, store::Inputs)]
pub struct PfilterCore {
    /// The integration step. Default: the model's `simulate { dt }`, else the
    /// --fit file's `[config] dt`, else 1.0. A value that differs from the
    /// model's step or the --fit file's is refused: a command that fits or
    /// scores data integrates at the model's step (run spec §1.3.1).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_dt)]
    pub dt: Option<f64>,

    /// RNG seed
    #[arg(long, default_value_t = 1)]
    #[input(identity)]
    pub seed: u64,

    /// Rayon thread count (0 = all available cores)
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    #[input(execution, exercised_by = crate::input_fixtures::pfilter_parallel)]
    pub parallel: usize,

    /// Restore the legacy silent-zero behaviour for numerical-collapse paths
    /// (division by zero, a power that is NaN, the square root of a negative)
    /// in rate evaluation. Default: hard error. Use only when a model needs it,
    /// such as a spatial model where an empty stratum's divisor is meant to
    /// give a zero rate.
    #[arg(long, default_value_t = false)]
    #[input(identity)]
    pub allow_degenerate_rates: bool,

    #[command(flatten)]
    #[input(group)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

    /// Time-column format for `--data`: `auto` (detect numeric-vs-date over
    /// the whole column), `numeric` (force `f64`), or `date` (force ISO
    /// dates, requires the model's `origin`). A format the column does not
    /// hold is refused.
    #[arg(long, default_value = "auto")]
    #[input(execution, exercised_by = crate::input_fixtures::pfilter_time_format)]
    pub time_format: crate::caltime_load::TimeFormat,

    /// The maximum cumulative particle-substep count one filter evaluation
    /// may execute before it is refused (`PFIterationBudget`, exit status 3).
    /// Unset: the engine default (1e10). A refused run stores nothing, so the
    /// budget never changes a stored result.
    #[arg(long, value_name = "N")]
    #[input(execution, exercised_by = crate::input_fixtures::pfilter_max_substeps)]
    pub pf_max_substeps: Option<u64>,

    /// Binomial sampler for the chain-binomial draws: `btpe` (default) or
    /// `btrs` (Hörmann 1993; faster). The two are not bit-compatible, so a
    /// `btrs` run is a result of its own.
    #[arg(long, value_name = "SAMPLER",
          value_parser = |v: &str| v.parse::<sim::rng::BinomialAlgorithm>())]
    #[input(identity, vary = crate::input_fixtures::other_binomial)]
    pub binomial: Option<sim::rng::BinomialAlgorithm>,
}

// ─── simulate ─────────────────────────────────────────────────────────────────

#[derive(Args, Clone, store::Inputs)]
#[command(after_help = colored_help!("\
Examples:
  # Basic simulation. Each cell is stored in the run store, ./camdl-store
  # (reruns with identical inputs are instant); read it back with
  # `camdl cat <id>` or `camdl cat @LABEL`, and browse with `camdl list`.
  camdl simulate sir.camdl --params p.toml --seed 42 --label base
  camdl list        # browse stored cells

  # Also export the combined table to a file
  camdl simulate sir.camdl --params p.toml --seed 42 -o traj.tsv

  # Write the combined table to stdout for piping
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
    #[input(identity, vary = crate::input_fixtures::sim_other_model)]
    pub model: PathBuf,

    /// Override the run's horizon (`simulation.t_end`), gh#626. SPEC is a
    /// model-time number ("120"), a calendar date (date("YYYY-MM-DD") or
    /// bare YYYY-MM-DD; needs a model origin), or an observation anchor
    /// with optional offset: "last_obs", "last_obs + 8 weeks",
    /// "first_obs - 1 week". Anchored forms require --fit (the fit's
    /// [data.observations] supplies the observed times; last_obs = the max
    /// observation time over the bound streams). Without a model origin,
    /// months/years are fixed spans (365.2425/12 = 30.436875 d / 365.2425 d);
    /// under an origin a month or year offset from an anchor is refused,
    /// since the anchor's date is known only at run time (the one-step
    /// calendar rule): write the offset in days. A scenario
    /// that declares its own different horizon is a hard error.
    #[arg(long, value_name = "SPEC")]
    #[input(identity, vary = crate::input_fixtures::sim_other_to)]
    pub to: Option<String>,

    /// Override the run's start (the model's `simulate { from }`), so one
    /// model serves several data windows. SPEC is the grammar of --to: a
    /// model-time number ("-2"), a date (needs a model origin), or an
    /// observation anchor with optional offset ("first_obs",
    /// "first_obs - 1 week"); first_obs is the start of the earliest observed
    /// period over the bound streams, so anchored forms require --fit. Takes
    /// precedence over the --fit config's `[simulate] from`. Moves every start
    /// derived from the model's (the output and emit schedules, a recurring
    /// schedule with no `from`), and enters the run's identity through the
    /// model. Refused with --init-state, which sets the start itself.
    #[arg(long, value_name = "SPEC", allow_hyphen_values = true)]
    #[input(identity, vary = crate::input_fixtures::sim_other_from)]
    pub from: Option<String>,

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
    #[input(identity, vary = crate::input_fixtures::sim_init_state)]
    pub init_state: Option<String>,

    /// gh#audit-C6 / S1. See ProfileCore.allow_degenerate_rates.
    /// Forward sim is the most likely user of this flag — if a model
    /// has a known empty-stratum-divisor and the user wants to keep
    /// the legacy zero-rate behaviour rather than fix the model.
    #[arg(long, default_value_t = false)]
    #[input(identity)]
    pub allow_degenerate_rates: bool,

    #[command(flatten)]
    #[input(group)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

    #[command(flatten)]
    #[input(group)]
    pub model_overrides: ModelOverrides,

    /// Parameter vector file (may be repeated), e.g. --param-vec beta=FILE
    #[arg(long, value_name = "PREFIX=FILE")]
    #[input(identity, vary = crate::input_fixtures::sim_param_vec)]
    pub param_vec: Vec<ParamVecSpec>,

    /// Named scenarios (may be repeated; conflicts with --enable/--disable)
    #[arg(long = "scenario", conflicts_with_all = ["enable", "disable"])]
    #[input(expands)]
    pub scenarios: Vec<String>,

    /// Enable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenarios")]
    #[input(identity, vary = crate::input_fixtures::an_intervention)]
    pub enable: Vec<String>,

    /// Disable an intervention (may be repeated; conflicts with --scenario)
    #[arg(long, conflicts_with = "scenarios")]
    #[input(identity, vary = crate::input_fixtures::an_intervention)]
    pub disable: Vec<String>,

    #[command(flatten)]
    #[input(group)]
    pub backend: SimBackend,

    /// Trajectory output view: --output-every / --no-flows / --columns.
    #[command(flatten)]
    #[input(group)]
    pub output_view: OutputView,

    /// RNG seed for a single run (conflicts with --seeds)
    #[arg(long, default_value_t = 1, conflicts_with = "seeds",
          env = "CAMDL_SEED")]
    #[input(identity)]
    pub seed: u64,

    /// Multiple seeds: range (1:100) or list (1,2,42); conflicts with --replicates
    #[arg(long, conflicts_with_all = ["replicates"])]
    #[input(expands)]
    pub seeds: Option<SeedSpec>,

    /// Stochastic replicates per parameter point (conflicts with --seeds)
    #[arg(long, conflicts_with = "seeds")]
    #[input(expands)]
    pub replicates: Option<usize>,

    /// Parameter draw source: path to a params TSV, "uniform", "prior", or
    /// "posterior". "posterior" reads a stored fit run's post-warm-up draws —
    /// requires `--fit <run>` (its @label, id or folder). "prior" samples the
    /// parameters with a prior and "uniform" the ones with declared bounds;
    /// every other parameter comes from the model default, `--params`, the
    /// scenario or `--param`. A draw is a SAMPLE from a measure; a value you
    /// choose is a design point — use --sweep / --sweep-file for those.
    #[arg(long)]
    #[input(expands)]
    pub draws: Option<String>,

    /// Deterministic grid over a parameter (repeatable): the DESIGN axis.
    /// GRID is a list (`beta=0.2,0.3,0.6`), `lin(min,max,n)` or
    /// `log10(min,max,n)`; an indexed parameter is named underscore-joined
    /// (`beta_north`). Several --sweep flags form the Cartesian product
    /// sorted by parameter name, the first name varying slowest; `point_id`
    /// is a point's 0-based position in it. Every design point at a given
    /// replicate shares one random stream (common random numbers), so paired
    /// differences across points are low-variance and adding a point leaves
    /// the others' trajectories unchanged; for independent noise per point,
    /// run one invocation per point with a distinct --seed. Composes with
    /// --scenario, --replicates, --seeds and the output flags; refused with
    /// --draws, with a --param on a swept parameter, and with a scenario that
    /// sets or scales one. Cells already in the store are read back, not
    /// re-simulated (--force recomputes; --dry-run shows hits and misses).
    #[arg(long, value_name = "PARAM=GRID",
          conflicts_with_all = ["draws", "design_from", "sweep_file"])]
    #[input(expands)]
    pub sweep: Vec<SweepSpec>,

    /// Design points from a TSV: the header names model parameters, each row
    /// is one design point, `point_id` its 0-based row — or, when the file
    /// has a `point_id` column of distinct non-negative integers, the row's
    /// own id. The DESIGN axis, like --sweep (not a sample — a hand grid
    /// passed as --draws FILE would be pooled into a band). A column naming
    /// no model parameter, a file with zero rows, a duplicate row, and a
    /// repeated or non-integer point_id are refused. The file enters the
    /// ensemble's record by content digest.
    #[arg(long, value_name = "FILE",
          conflicts_with_all = ["draws", "design_from", "sweep"])]
    #[input(expands)]
    pub sweep_file: Option<PathBuf>,

    /// Companion for `--draws`. With `--draws prior`, a fit.toml supplying
    /// priors. With `--draws posterior`, the stored fit run to read the
    /// posterior draws from: its `@label`, a run-id prefix or its folder; a
    /// fit.toml names no run and is refused with the runs it made listed.
    /// With `--draws <file.tsv>`, a fit.toml or a stored fit run whose
    /// `[fixed]` block gives the parameters absent from the file's
    /// columns their value, at the fit.toml `[fixed]` tier: `--params` and the
    /// scenario outrank it, and a column the file provides outranks it (#273).
    /// Also consulted (without --draws) by an anchored `--to`: the fit's
    /// [data.observations] supplies the observed times the anchor resolves
    /// against — a manual check requires --fit to come with --draws and/or
    /// an anchored --to (gh#626).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::sim_fit)]
    pub fit: Option<PathBuf>,

    /// Number of parameter draws. For --draws uniform/prior: how many to
    /// generate. For --draws posterior: a strided subsample cap across the
    /// whole cloud (default 200, matching `camdl predict` — never silently
    /// replay a full 60k-draw posterior; gh#630).
    #[arg(short = 'n', long)]
    #[input(expands)]
    pub n_draws: Option<usize>,

    /// With `--draws posterior`: replay ONLY the named chains of the fit's
    /// posterior cloud (comma-separated chain ids counted from 0, e.g.
    /// `--chains 3,5`). Cannot be given with `--exclude-chains`. Always warns;
    /// a chain id not in the fit, or naming every chain, is a hard error.
    #[arg(long, value_name = "IDS", conflicts_with = "exclude_chains")]
    #[input(expands)]
    pub chains: Option<String>,

    /// With `--draws posterior`: drop the named chains of the fit's posterior
    /// cloud before replaying it (comma-separated chain ids counted from 0).
    /// Post-hoc exclusion biases the posterior toward the retained mode and
    /// always prints a warning.
    #[arg(long, value_name = "IDS")]
    #[input(expands)]
    pub exclude_chains: Option<String>,

    /// Write the sampled per-draw parameter vectors to this TSV (gh#157).
    /// One row per draw, one column per parameter the draw source carries —
    /// for `--draws prior|uniform`, only the sampled parameters (gh#949) —
    /// in the column-per-param format `--draws PATH` reads back.
    /// Only written when given; written from the resolved draws, it stores
    /// nothing.
    #[arg(long, value_name = "PATH", requires = "draws")]
    #[input(export)]
    pub draws_out: Option<PathBuf>,

    /// Export the combined table to this file: every cell's stored
    /// trajectory, with the `replicate`, `scenario`, point and `seed`
    /// columns, for one cell as for many. Rendered from the store after the
    /// cells are stored; `camdl cat @LABEL` renders the same table.
    #[arg(short, long, env = "CAMDL_OUTPUT")]
    #[input(export)]
    pub output: Option<PathBuf>,

    /// Write the combined table `-o` writes to stdout, for piping into
    /// another tool (`camdl simulate … --stdout | …`). The cell is stored as
    /// for any run. Single-cell only: it conflicts with --seeds /
    /// --replicates / --draws / --sweep and with `-o`/`--obs*`.
    #[arg(long, conflicts_with_all = [
        "output", "obs", "obs_dir", "obs_only", "obs_only_dir",
        "seeds", "replicates", "draws", "sweep", "sweep_file",
    ])]
    #[input(export)]
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
    #[input(identity, vary = crate::input_fixtures::sim_emit_every)]
    pub emit_every: Vec<String>,

    /// Write synthetic observations to a single TSV (all streams)
    #[arg(long, conflicts_with_all = ["obs_dir", "obs_only"])]
    #[input(adds)]
    pub obs: Option<PathBuf>,

    /// Write one TSV per observation stream to a directory
    #[arg(long, conflicts_with_all = ["obs", "obs_only"])]
    #[input(adds)]
    pub obs_dir: Option<PathBuf>,

    /// Like --obs but suppress trajectory output
    #[arg(long, conflicts_with_all = ["obs", "obs_dir", "output", "obs_only_dir"])]
    #[input(adds)]
    pub obs_only: Option<PathBuf>,

    /// Like --obs-dir (one TSV per stream) but suppress trajectory output.
    /// The multi-cadence-safe sibling of --obs-only (run-spec §3.1.1,
    /// ObsOutput::OnlyDir).
    #[arg(long, conflicts_with_all = ["obs", "obs_dir", "obs_only", "output"])]
    #[input(adds)]
    pub obs_only_dir: Option<PathBuf>,

    /// Simulate on the observation design a fit config binds, instead of on
    /// the model's `emit_schedule` (gh#831). FIT_TOML's `[data.observations]`
    /// is loaded exactly as `fit run` loads it, and each stream's rows are
    /// reproduced: its observed labels, each row's own period — per-row
    /// `window_start`/`window_stop` widths included — and each `NA` hole,
    /// which is written as a row with its period and no value. The result is
    /// one file per stream under the stream's declared column names, so it
    /// re-loads under the model that produced it. This is what the
    /// simulation-based self-consistency test needs: simulating on a regular
    /// grid would give the synthetic fit more information than the real fit
    /// has. Requires --obs-dir, --obs-only-dir or --projected-out. The parameter vector comes
    /// from --params/--param (one dataset, written into the directory) or
    /// from --draws (one `ds_NN/` subdirectory per draw). What it produces is
    /// a dataset, not a run: no trajectory and no store leaf are written, with
    /// --obs-dir as with --obs-only-dir. A stream whose likelihood reads a
    /// data column is refused by name (gh#829), except a binomial `n` over a
    /// ratio of flows, which the model generates and this writes. Refused
    /// with --scenario: the dataset is written from the parameter vector
    /// alone, with no scenario applied.
    #[arg(long, value_name = "FIT_TOML", conflicts_with_all = [
        "obs", "obs_only", "output", "stdout", "seeds", "replicates",
        "emit_every", "dates", "init_state", "quantities_out", "event_log",
        "scenarios",
    ])]
    #[input(export)]
    pub design_from: Option<PathBuf>,

    /// Write each observation stream's projected value on the design that
    /// `--design-from` binds: the model quantity the stream's `projected = …`
    /// expression defines, before any observation noise. One TSV per stream,
    /// with the stream's temporal column(s) and a `projected` column at full
    /// precision; a row that is a hole in the data still gets its value. With
    /// --draws, one `ds_NN/` subdirectory per draw. The value is a
    /// deterministic function of the run, so no data column is needed and
    /// gh#829 does not apply. It is not a dataset: the column is named
    /// `projected`, not the stream's observed column, so it cannot be bound
    /// back as data by accident. A likelihood may transform the projection
    /// (`mean = 0.86 * projected`); this writes the projection itself. Like
    /// --quantities-out, a sidecar outside run identity. Usable with or
    /// without --obs-dir / --obs-only-dir.
    #[arg(long, value_name = "DIR", requires = "design_from")]
    #[input(export)]
    pub projected_out: Option<PathBuf>,

    /// Add a calendar `date` column (rendered from the model `origin` +
    /// `time_unit`) alongside the numeric `t`/`time` column in trajectory
    /// and observation output. Numeric time stays the canonical, diff-stable
    /// default; `--dates` is purely additive (2026-05-22 calendar-time §6.7).
    /// No-op with a clear error if the model declares no `origin`.
    #[arg(long)]
    #[input(render)]
    pub dates: bool,

    /// Emit the model's `quantities {}` block to this directory as
    /// `<dir>/quantities/<name>.tsv` + `<dir>/quantities.json`. Each cell's
    /// quantities are stored beside it as a by-product, evaluated from the
    /// trajectory at full precision, and the tables are rendered from them: a
    /// single fixed-params run writes a bare `value` per quantity (point
    /// mode); a `--draws`/`--replicates`/`--seeds` run writes quantiles over
    /// each group's replicates. A cell's own id does not depend on it.
    /// Without this flag, a model that declares quantities prints a note and
    /// skips them (not a hard error).
    #[arg(long, value_name = "DIR")]
    #[input(adds)]
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
    #[input(identity, vary = crate::input_fixtures::sim_vocabulary)]
    pub quantities: Option<PathBuf>,

    /// Print resolved run plan without simulating
    #[arg(long)]
    #[input(skip)]
    pub dry_run: bool,

    /// The run store's root, over the `store` key of a `--fit` file (default
    /// `$CAMDL_STORE`, else `./camdl-store`; run spec §2.1).
    #[arg(long, value_name = "DIR")]
    #[input(store)]
    pub store: Option<PathBuf>,

    /// Rayon threads for the simulation phase of a multi-cell run (absent:
    /// sequential; 0 = all cores). Output is byte-identical whatever the
    /// value: cells are seeded by grid position and merged in canonical order.
    #[arg(long, env = "CAMDL_PARALLEL")]
    #[input(execution, exercised_by = crate::input_fixtures::sim_parallel)]
    pub parallel: Option<usize>,

    /// Recompute every cell even when it is stored; the stored cell is moved
    /// to quarantine first.
    #[arg(long)]
    #[input(store)]
    pub force: bool,

    /// A name for this invocation, found again as `@LABEL`. It names the
    /// invocation, never a cell: rerunning under another label reuses every
    /// stored cell, and `camdl cat @LABEL` renders the invocation's table.
    #[arg(long, value_name = "TEXT")]
    #[input(store)]
    pub label: Option<String>,

    /// Record the Layer-1 lineage **event log** (2026-05-20 three-layer
    /// architecture). Requires a model with `#[lineage]` annotations and a
    /// backend that declares the LINEAGES capability (Gillespie /
    /// chain-binomial). The event log is a by-product of the cell, stored
    /// inside it, so asking for it never changes the cell's id (read it with
    /// `camdl cat <id> --stream event_log.tsv`); realize it into a line list
    /// with `camdl lineage realize`. Pass a PATH to also export the log to
    /// that file (format from --tsv/--format/extension); bare `--event-log`
    /// (or `auto`) only stores it. Single-run only: conflicts with --seeds /
    /// --replicates / --draws / --sweep / --sweep-file / --stdout.
    #[arg(long, value_name = "FILE", num_args = 0..=1, default_missing_value = "auto",
          conflicts_with_all = ["seeds", "replicates", "draws", "sweep", "sweep_file", "stdout"])]
    #[input(adds)]
    pub event_log: Option<PathBuf>,

    /// Event-log format: `parquet` (default, production) or `tsv`
    /// (dependency-free, debug). With --event-log.
    #[arg(long, value_name = "FMT", requires = "event_log", conflicts_with = "tsv")]
    #[input(render)]
    pub format: Option<String>,

    /// Shorthand for `--format tsv` (with --event-log).
    #[arg(long, requires = "event_log")]
    #[input(render)]
    pub tsv: bool,

    /// Export the reactive firing log to this file: a copy of the cell's
    /// stored `reactive_log.tsv`, which a cell holds whenever a reactive
    /// policy was active (read it with `camdl cat <id> --stream
    /// reactive_log.tsv`). Single-run only.
    #[arg(long, value_name = "PATH",
          conflicts_with_all = ["seeds", "replicates", "draws", "sweep", "sweep_file"])]
    #[input(export)]
    pub reactive_log: Option<PathBuf>,

    /// Write the output manifest to PATH once the command has succeeded: a
    /// JSON file naming every result it stored or reused, each result file
    /// by logical name with its SHA-256 and size, for a workflow tool to
    /// depend on (run spec §2.10). Any file at PATH is removed when the
    /// command starts, so a failed command leaves none.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["dry_run", "design_from"])]
    #[input(export)]
    pub manifest: Option<PathBuf>,
}

/// One `camdl simulate` as its inputs reach the store: the command's
/// arguments and the global options, each with its class. Built by the
/// dispatch in `main.rs`; the global options are not `SimulateArgs`' own.
#[derive(Clone, store::Inputs)]
pub struct SimulateInvocation {
    #[input(group)]
    pub args: SimulateArgs,
    /// `--no-licm`: compile without loop-invariant code motion. It changes
    /// the compiled model, whose bytes every cell's identity holds.
    #[input(identity)]
    pub no_licm: bool,
    /// `--no-ir-cache`: compile rather than reuse a cached IR, which is the
    /// IR a compile emits.
    #[input(execution, exercised_by = crate::input_fixtures::sim_no_ir_cache)]
    pub no_ir_cache: bool,
    #[input(skip)]
    pub verbosity: Option<log::LevelFilter>,
    #[input(skip)]
    pub progress: types::ProgressMode,
    #[input(skip)]
    pub no_progress: bool,
}

// ─── fit ──────────────────────────────────────────────────────────────────────

#[derive(Args, Clone, store::Inputs)]
#[command(after_help = colored_help!("\
Examples:
  # Run the full inference pipeline declared in fit.toml
  camdl fit run fit.toml --seed 1

  # Long fits: capture progress + diagnostics to a log file
  camdl fit run fit.toml --seed 1 --progress plain 2>&1 | tee fit.log

  # Force rerun even if cached results match
  camdl fit run fit.toml --seed 1 --force

  # Override the file's `starts` rule: draw every chain's start apart
  camdl fit run fit.toml --starts lhs

  # Chain two files by handle: label the first, source the second from it
  camdl fit run scout.toml --seed 1 --label scout
  camdl fit run posterior.toml --seed 1 --starts from_posterior=@scout

  # Every chain at a stored fit's estimate (R̂ is then not assessed)
  camdl fit run posterior.toml --starts from_mle=@scout

  # Every chain at the point in a hand-written flat params TOML
  camdl fit run fit.toml --starts from_params=truth.toml

Notes:
  - PGAS/PMMH fits can resume a partial run with `--resume`.
  - The problem is stored under `<store>/<build>/fit/<model>-<hash>/`, and
    each run inside it under `runs/<method>-<hash>/seed_<n>-<hash>/`. The
    store is `--store`, else the file's `store` key, else `$CAMDL_STORE`,
    else `./camdl-store`.

Priors (Bayesian methods: PGAS, PMMH):
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

  Per-parameter provenance is recorded in each run's `provenance.json`
  under `resolved_priors` (sources: fit_toml, model_ir, flat_explicit).
  See `docs/inference.md` § \"Priors and precedence\" for the full
  spec including the asymmetry with `camdl profile`.
"))]
pub struct FitRunArgs {
    /// Fit configuration file (v2 TOML). Its problem half poses the fit
    /// problem every run of it shares; its `[method]` is this run's.
    #[input(identity, vary = crate::input_fixtures::fit_other_config)]
    pub config: PathBuf,

    #[command(flatten)]
    #[input(group)]
    pub problem: FitProblemFlags,

    #[command(flatten)]
    #[input(group)]
    pub run: FitRunFlags,

    #[command(flatten)]
    #[input(group)]
    pub method: FitMethodFlags,

    /// Start from a `from_mle` / `from_posterior` source whose stored verdict
    /// is not converged. Without this the source is refused, since starting
    /// from an unconverged fit launders its multi-modality into this one.
    #[arg(long)]
    #[input(execution, exercised_by = crate::input_fixtures::fit_allow_nonconverged_source)]
    pub allow_nonconverged_source: bool,

    /// Max worker threads for the Rayon pool that runs the chains and, within
    /// each chain, the particle filter. `0` (the default) uses all logical
    /// cores. Results are bit-identical regardless of this value — it caps the
    /// thread budget / CPU footprint, not the numerics. Mirrors `--parallel`
    /// on pfilter/profile; also honored via `CAMDL_PARALLEL`.
    #[arg(long, default_value_t = 0, env = "CAMDL_PARALLEL")]
    #[input(execution, exercised_by = crate::input_fixtures::fit_parallel)]
    pub parallel: usize,

    /// Recompute every run even when it is stored; the stored run is moved
    /// to the store's quarantine first. Also reclaims a run held by a process
    /// on another host, which camdl cannot check and never reclaims on its
    /// own. Conflicts with --resume.
    #[arg(long, conflicts_with = "resume")]
    #[input(store)]
    pub force: bool,

    /// A name for this invocation, found again as `@LABEL` (1–64 chars after
    /// trim; allowed: letters, digits, spaces, commas, dot, underscore,
    /// hyphen). It names the invocation, never a run: rerunning under another
    /// label reuses every stored run, and an invocation that makes one run
    /// lets `@LABEL` name that run. Refused while a run of an invocation
    /// labelled LABEL is still running.
    #[arg(long, value_name = "TEXT")]
    #[input(store)]
    pub label: Option<String>,

    /// The run store's root, over the fit file's `store` key (default
    /// `$CAMDL_STORE`, else `./camdl-store`; run spec §2.1).
    #[arg(long, value_name = "DIR")]
    #[input(store)]
    pub store: Option<PathBuf>,

    /// Write the output manifest to PATH once the command has succeeded: a
    /// JSON file naming every result it stored or reused, each result file
    /// by logical name with its SHA-256 and size, for a workflow tool to
    /// depend on (run spec §2.10). Any file at PATH is removed when the
    /// command starts, so a failed command leaves none.
    #[arg(long, value_name = "PATH")]
    #[input(export)]
    pub manifest: Option<PathBuf>,

    // ── Removed-flag traps ───────────────────────────────────────────
    //
    // Each of these was renamed or removed with the `[stages]` → `[method]`
    // split (proposal 2026-09-08); the dispatch site answers every one it
    // sees with its replacement in one message. Per CLAUDE.md alpha posture,
    // no back-compat shims — they exist purely so the error names the right
    // replacement.
    #[arg(long = "stage", value_name = "STAGE", hide = true)]
    #[input(skip)]
    pub _removed_stage: Option<String>,
    #[arg(long = "init", value_name = "MODE", hide = true)]
    #[input(skip)]
    pub _removed_init: Option<String>,
    #[arg(long = "posterior", value_name = "PATH", hide = true)]
    #[input(skip)]
    pub _removed_posterior: Option<String>,
    #[arg(long = "mle", value_name = "PATH", hide = true)]
    #[input(skip)]
    pub _removed_mle: Option<String>,
    #[arg(long = "params", value_name = "TOML", hide = true)]
    #[input(skip)]
    pub _removed_params: Option<String>,
    #[arg(long = "survey-path", value_name = "DIR", hide = true)]
    #[input(skip)]
    pub _removed_survey_path: Option<String>,
    #[arg(long = "survey-top-k", value_name = "N", hide = true)]
    #[input(skip)]
    pub _removed_survey_top_k: Option<String>,
    #[arg(long = "allow-nonconverged-scout", hide = true)]
    #[input(skip)]
    pub _removed_allow_nonconverged_scout: bool,
    #[arg(long = "init-method", value_name = "MODE", hide = true)]
    #[input(skip)]
    pub _removed_init_method: Option<String>,
    #[arg(long = "starts-from", value_name = "DIR_OR_HASH", hide = true)]
    #[input(skip)]
    pub _removed_starts_from: Option<String>,
    // `--tempering` went with PGAS parallel tempering (gh#789); same shape,
    // its own message.
    #[arg(long = "tempering", value_name = "B1,B2,...", hide = true)]
    #[input(skip)]
    pub _removed_tempering: Option<String>,
    // `--rho` went with correlated pseudo-marginal PMMH; same shape, its own
    // message.
    #[arg(long = "rho", value_name = "F", hide = true)]
    #[input(skip)]
    pub _removed_rho: Option<String>,
}

/// The flags of `fit run` that change the problem.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct FitProblemFlags {
    // Sets the config's `[config] allow_forcing_extrapolation = true`, so it
    // keys the problem as the config line does.
    #[command(flatten)]
    #[input(group)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

    /// Override the cadence at which `[synthetic]` data is GENERATED
    /// (`emit_schedule`), gh#656. Same grammar as `camdl simulate
    /// --emit-every`: a plain number in the model's own `time_unit` for every
    /// stream, or `NAME=N` (repeatable) for one stream by its
    /// observation-block label. This is the only fit path the emission cadence
    /// reaches — a fit against real data scores at its data files' own times
    /// and never consults `emit_schedule` — so the flag is REFUSED on a fit
    /// with no `[synthetic]` block rather than silently doing nothing. It
    /// changes the generated data, so the problem re-keys.
    #[arg(long = "emit-every", value_name = "N | NAME=N")]
    #[input(identity, vary = crate::input_fixtures::fit_emit_every)]
    pub emit_every: Vec<String>,

    /// Cartesian sweep over a fixed parameter (may repeat): one problem per
    /// point. SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC")]
    #[input(expands)]
    pub sweep: Vec<SweepSpec>,
}

/// The flags of `fit run` that change the run, other than the method's
/// settings.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct FitRunFlags {
    /// RNG seed (default: 1). Refused beside a file that declares
    /// `fit_seeds`, which replaces it.
    #[arg(long)]
    #[input(identity)]
    pub seed: Option<u64>,

    /// Where the method's chains begin, overriding the file's `starts`. A
    /// rule name — `uniform_unconstrained`, `lhs`, `uniform`, `from_prior`,
    /// `single` — or a sourced rule with its source: `from_posterior=@base`
    /// (one posterior row per chain), `from_mle=@mle` (every chain at that
    /// fit's estimate), `from_params=theta.toml`. Default when neither the
    /// file nor this flag says: `from_prior` if every estimated parameter
    /// declares a prior, `uniform_unconstrained` otherwise. Keyed into the
    /// run's identity, like the file's own `starts`.
    #[arg(long, value_name = "SPEC")]
    #[input(identity, vary = crate::input_fixtures::fit_starts)]
    pub starts: Option<crate::fit::starts::ChainStarts>,

    /// Restrict a `from_posterior` rule to these chains of its source,
    /// comma-separated (`0,1,2,4`) — the TOML `chains = [..]`. Applies to the
    /// rule `--starts` or the file names; any other rule is refused. The
    /// selected chains must agree (their own R̂ is checked in place of the
    /// source's whole-run verdict), and the subset is keyed into the run's
    /// identity (gh#1056).
    #[arg(long, value_name = "IDS", value_parser = crate::fit::starts::parse_starts_chains)]
    #[input(identity, vary = crate::input_fixtures::fit_starts_chains)]
    pub starts_chains: Option<crate::chain_selection::ChainSelection>,

    /// Resume a completed PGAS or PMMH run, named by handle (`@label`, a
    /// run-id prefix or its folder), to the longer length this file's
    /// `[method]` asks for. The base is read through the store; the resumed
    /// run is a new run beside its base with the same method settings and
    /// seed, a longer length, and a dependency on the base's carried chain
    /// files. Conflicts with --force.
    #[arg(long, value_name = "BASE")]
    #[input(identity, vary = crate::input_fixtures::fit_resume)]
    pub resume: Option<String>,
}

/// The method flags of `fit run`. Each overrides one key of `[method]`, and
/// a flag the method does not read is refused.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct FitMethodFlags {
    /// Override [method.gate] decibans_thresh (the inter-chain
    /// log-likelihood-spread floor, in decibans).
    #[arg(long, value_name = "DB")]
    #[input(identity)]
    pub decibans_thresh: Option<f64>,

    // ── Richardson dt-convergence check (gh#52) ─────────────────────

    /// Skip the post-fit Richardson dt-convergence check at θ̂.
    /// Default: the check runs on every IF2 method (PF likelihood) and
    /// every ODE inference method — nl-sbplx, nl-bobyqa, nl-lbfgs, and mh
    /// (deterministic likelihood). Use this for CI smoke fits or
    /// known-converged-dt rerenders where the audit cost is unwelcome. The
    /// dt-check result is stored in the run, so the override is keyed into
    /// its identity.
    #[arg(long)]
    #[input(identity)]
    pub no_dt_check: bool,

    /// Drop the dt-check warning threshold to the strict default
    /// (0.5 nats for chain_binomial, 0.1 nats for ode_rk4). Targets
    /// research-quality fits where sub-nat differences matter for
    /// paper-grade conclusions; the routine default (2.0 / 0.5)
    /// allows more give before flagging. The threshold it selects is stored
    /// in the run (fit_state.toml.dt_check), so it is keyed into the run's
    /// identity. A method that declares its own threshold_nats is unaffected.
    #[arg(long)]
    #[input(identity)]
    pub dt_check_strict: bool,

    /// Binomial sampler for the chain-binomial draws of a `pgas`, `pmmh`,
    /// `if2` or `pfilter` method: `btpe` (default) or `btrs` (Hörmann 1993;
    /// faster, gh#747). The two samplers are not bit-compatible — a different
    /// rejection scheme accepts different draws from the same stream — so
    /// this is resolved into the method's `binomial` field and keyed into
    /// the run's identity.
    #[arg(long, value_name = "SAMPLER",
          value_parser = |v: &str| v.parse::<sim::rng::BinomialAlgorithm>())]
    #[input(identity, vary = crate::input_fixtures::other_binomial)]
    pub binomial: Option<sim::rng::BinomialAlgorithm>,

    /// Override `n_halvings` on the dt-check (gh#52). Default 2
    /// (evaluates at dt_fit, dt_fit/2, dt_fit/4 — 7× the
    /// loglik-eval cost). Use 3 for ambiguous cases at 15×. The ladder
    /// result is stored in the run, so the override is keyed into its
    /// identity.
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub dt_check_halvings: Option<usize>,

    // ── IF2-specific algorithm overrides ────────────────────────────

    /// Override [method.cooling_target_iters]. Iterations
    /// over which the cooling fraction is reached (pomp's
    /// cooling.fraction.50 default).
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub cooling_target_iters: Option<usize>,

    // ── PGAS-specific algorithm overrides  ──────────

    /// Override [method.max_tree_depth] (NUTS depth ceiling).
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub max_tree_depth: Option<usize>,

    /// Override [method.trajectory_warmup] (CSMC-only sweeps
    /// before parameter updates begin).
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub trajectory_warmup: Option<usize>,

    /// Override [method.csmc_sweeps_per_nuts] (CSMC trajectory
    /// updates per parameter update).
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub csmc_sweeps_per_nuts: Option<usize>,

    /// Override [method.n_trajectories] (posterior trajectories
    /// saved).
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub n_trajectories: Option<usize>,

    /// Override [method.dense_mass] to false (use diagonal NUTS
    /// mass matrix). One-way: edit TOML to flip back.
    #[arg(long)]
    #[input(identity)]
    pub diagonal_mass: bool,

    /// Override [method.use_nuts] to false (fall back to
    /// MH-within-Gibbs for the θ|X update). One-way: edit TOML to
    /// flip back.
    #[arg(long)]
    #[input(identity)]
    pub no_nuts: bool,

    /// Override [method.ancestor_sampling] to false: run the CSMC
    /// sweep as plain particle Gibbs, without the ancestor-sampling move.
    /// A diagnostic control (what does AS contribute, and what does its
    /// density pass cost?); changes the sampled draws, so the run stores
    /// under its own address. One-way: edit TOML to flip back.
    #[arg(long)]
    #[input(identity)]
    pub no_ancestor_sampling: bool,

    // ── PMMH-specific algorithm overrides  ──────────

    /// Override [method.adapt] to false (lock proposal SDs;
    /// disables Haario-style adaptation). One-way: edit TOML to flip
    /// back.
    #[arg(long)]
    #[input(identity)]
    pub no_adapt: bool,

    /// Override [method.adapt_start] (MCMC step at which
    /// proposal-SD adaptation begins).
    #[arg(long, value_name = "N")]
    #[input(identity)]
    pub adapt_start: Option<usize>,

    // ── PFilter-specific algorithm overrides  ───────

    /// Override [method.record_ancestry] to true (record
    /// ancestor indices for smoothing-path reconstruction).
    #[arg(long)]
    #[input(identity)]
    pub record_ancestry: bool,

    /// Override [method.record_prequential] to true (record
    /// per-step predictive samples for `camdl compare`).
    #[arg(long)]
    #[input(identity)]
    pub record_prequential: bool,
}

/// One `camdl fit run` as its inputs reach the store: the command's
/// arguments and the global options, each with its class. Built by the
/// dispatch in `main.rs`; the global options are not `FitRunArgs`' own.
#[derive(Clone, store::Inputs)]
pub struct FitRunInvocation {
    #[input(group)]
    pub args: FitRunArgs,
    /// `--no-licm`: compile without loop-invariant code motion. It changes
    /// the compiled model, whose bytes the problem's identity holds.
    #[input(identity)]
    pub no_licm: bool,
    /// `--no-ir-cache`: compile rather than reuse a cached IR, which is the
    /// IR a compile emits.
    #[input(execution, exercised_by = crate::input_fixtures::fit_no_ir_cache)]
    pub no_ir_cache: bool,
    #[input(skip)]
    pub verbosity: Option<log::LevelFilter>,
    #[input(skip)]
    pub progress: types::ProgressMode,
    #[input(skip)]
    pub no_progress: bool,
}

/// Output format for `camdl fit summary`. `text` is the default
/// rendered terminal block with ANSI colour; `json` is a versioned
/// machine-readable schema (`schema.version`) in which a number is a JSON
/// number, `"NaN"`, `"inf"` or `"-inf"`, and `null` means "not computed";
/// `md` and `latex` are document-friendly outputs for the book pipeline.
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

  # Machine-readable JSON for the book pipeline
  camdl fit summary fit/he2010 --format json > summary.json

  # Markdown for embedding in a chapter
  camdl fit summary fit/he2010 --format md

  # Just the winner θ̂ as a flat params TOML, pipeable into
  # `camdl pfilter --params`:
  camdl fit summary fit/he2010 --params-only \\
    | camdl pfilter --params /dev/stdin model.camdl --data cases.tsv

  # Disable colour (useful for redirecting to a file)
  camdl fit summary fit/he2010 --no-color

  # Strict mode for CI: exit non-zero on provenance mismatch.
  # Auto-enabled when CI=true or CI=1 in the environment.
  camdl fit summary fit/he2010 --strict

  # The model's parameter legend — priors, citations, caveats
  camdl fit summary fit/he2010 --parameters

  # Define each section's terms under its own table
  camdl fit summary fit/he2010 --explain

The text layout, top to bottom: the fit and how it was sampled, then the
verdict, then four sections —

  posterior      the estimates, worst R̂ first
  convergence    both R̂ halves, the classic statistic, per-chain ESS
  chains         per-chain log-likelihood against the others, saved latent paths
  latent paths   whether the sampled paths themselves mixed (PGAS only)

Nothing is hidden behind a verbosity flag: every section carries its full
table. `--parameters` and `--explain` only ADD to it.
"))]
pub struct FitSummaryArgs {
    /// The fit run, by handle: `@label` (`fit run --label`), a run-id prefix
    /// or the run's folder in the store. A `fit.toml` names no run: it is
    /// refused, with the runs it made listed.
    pub fit: String,

    /// Output format. `text` (default) emits the terminal block;
    /// `json` emits a document versioned by `schema.version`; `md`
    /// emits GitHub-flavoured Markdown; `latex` emits `\begin{tabular}`
    /// blocks per section.
    #[arg(long, value_enum, default_value_t = FitSummaryFormat::Text,
          conflicts_with = "params_only")]
    pub format: FitSummaryFormat,

    /// Print only the winner θ̂ as a flat params TOML (no metadata,
    /// no provenance, no headings — pipeable into `camdl pfilter
    /// --params <(camdl fit summary --params-only ...)`).
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
    /// the MCMC chains, dropping the named chain ids (comma-separated, e.g.
    /// `--exclude-chains 3,5`). Chain ids count from 0, the same numbers as
    /// the `chain` column of `draws.tsv` and the `chain_N/` dirs. A view only — nothing on disk changes. The
    /// method's identity line then reads `K of N chains (excluded 3,5)`, and the
    /// verdict's forkable count says it is over the retained chains. Post-hoc
    /// exclusion
    /// BIASES the posterior toward the retained mode and always prints a
    /// warning; a chain id not in the fit, or excluding every chain, is a hard
    /// error. Incompatible with `--params-only` (a single winner θ̂ is not a
    /// chain average, so there is nothing to subset).
    #[arg(long, value_name = "IDS", conflicts_with = "params_only")]
    pub exclude_chains: Option<String>,

    /// Recompute the posterior diagnostics over ONLY the named chains
    /// (comma-separated chain ids counted from 0, e.g. `--chains 0,1,2,4`):
    /// the complement of `--exclude-chains`, which it cannot be given with.
    /// On a six-chain fit `--chains 0,1,2,4` and `--exclude-chains 3,5` are
    /// the same selection. A view only; always warns. A chain id not in the
    /// fit, or naming every chain, is a hard error.
    #[arg(long, value_name = "IDS", conflicts_with_all = ["params_only", "exclude_chains"])]
    pub chains: Option<String>,

    /// Print the model's parameter legend — each parameter's symbol, prior,
    /// citations and caveats, from the model's `#'` docs. Off by default: it
    /// is reference material a modeller reads deliberately, and on a
    /// well-documented model it runs to dozens of lines before the first
    /// number appears. `--format text` only.
    #[arg(long, conflicts_with = "params_only")]
    pub parameters: bool,

    /// After each section's table, add a short prose block defining that
    /// section's terms — R̂ and its two halves, the per-chain log-likelihood
    /// columns, what a forkable draw is, what a frozen latent cell is. Each
    /// term is defined once, under the section it is primarily read in.
    /// This interleaves prose with the tables, so the text output stops being
    /// cleanly machine-readable under it; use `--format json` for that.
    /// `--format text` only.
    #[arg(long, conflicts_with = "params_only")]
    pub explain: bool,
}

#[derive(Args, Clone, store::Inputs)]
#[command(after_help = colored_help!("\
Examples:
  # Write the free-forward posterior predictive for a stored fit run
  camdl predict @post

  # Just one stream (logical name or an expanded leaf name)
  camdl predict @post --stream onset

  # Look at a disagreement rhat_mean flagged: one band per chain, beside
  # the pooled one, under a leading `chain` column
  camdl predict @post --by-chain

  # Predict at one point given with a model file, conditioned on the rows
  # through 2026-03-01: bands of process and observation noise only
  # (treatment plug_in), stored at prediction/<model>-<id>/<mode>-<id>/
  camdl predict seir.camdl --params mle.toml --data cases=cases.tsv \\
      --holdout-after 2026-03-01 --horizon '4 weeks'

Outputs, stored inside the run at predict/<mode>-<id>/:
  predictive/<stream>.tsv   scenario | time | <dims...> | mode | horizon
                            | treatment | role | quantity | fit_rhat_max
                            | fit_ess_min | rhat_mean | ess_mean | rhat_pred
                            | ess_pred | n_draws | q05..q95
  observed/<stream>.tsv     time | <dims...> | value
Read both, join on (time, <dims>), plot observed over the predictive ribbon.

The role column says what part each row played in the fit:
  trained    a row of the fit's training prefix (an NA row included).
  held_out   a row of the bound file after that prefix ([data]
             holdout_after). A one_step band there uses parameters fitted
             without the row, and a state filtered through every earlier
             row, held-out rows included: it is not a forecast from the end
             of the training rows (that is --horizon).
  future     this prediction read no data row for this period: a
             free_forward row past the last row of the bound file. A
             [data.holdout] file is not read, so the periods it covers are
             future too (gh#1309).
  unknown    a data row of a fixed-point prediction given no
             --holdout-after: nothing records which rows the point was
             fitted to.
predictive.json records both sets: parameter_set (the rows the parameters
were fitted to) and state_sets (the rows each band's state was filtered
through).
For a stream labelled by each period's start, one_step rows carry the
period's end in `time` while free_forward rows and observed/ carry the label
(gh#1152), and the free_forward rows past the data are mis-timed (gh#1151);
the role is read by row position and is right on both.
Under --exclude-chains the bands land in predictive-excl<ids>/ instead, so a
chain subset never overwrites the pooled artifact; `camdl show <fit>` lists
every address the fit holds.

Convergence columns — two different questions, do not mix them up:
  fit_rhat_max,        the fit's worst parameter, copied from the producing
  fit_ess_min          method. Provenance about the fit, not about this row:
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
or a constant row. The one_step mode leaves both pairs empty (its cells
pool over particles as well as draws).

quantities-model/<name>.tsv carries the same reduction as `rhat` and `ess` — one
pair, because a quantity has one value per draw. Those are the numbers that
get published, so read them first. A quantity over latent state or derived
arithmetic is noise-free, so its `rhat` is the undiluted kind; a quantity
whose manifest `source` is `observations` reduces sampled y_rep and so carries
observation noise, making it the diluted (rhat_pred) kind. `simulate` has no
chains behind it and writes neither column.

--by-chain adds a leading `chain` column: `all` on the pooled rows, the
chain id (counted from 0, as in draws.tsv's `chain` column) on one extra band
per chain, on both horizons. Use it after
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
--exclude-chains 3,5 writes predictive-excl3_5/ with a `chain` column, and the
ids there are the fit's own numbering with 3 and 5 simply absent."))]
pub struct PredictArgs {
    /// Where the parameters come from. A handle (`@label`, a run-id prefix
    /// or a result folder) names a stored fit run, whose posterior draws the
    /// prediction uses. A model file (`.camdl` or `.ir.json`) names a model,
    /// predicted at the one point `--params` and `--param` give. A
    /// `fit.toml` names no run and is refused, with the runs it made listed.
    #[arg(value_name = "RUN|MODEL", required_unless_present = "_removed_fit")]
    #[input(identity, vary = crate::input_fixtures::predict_other_run)]
    pub source: Option<String>,

    #[command(flatten)]
    #[input(group)]
    pub flags: PredictFlags,

    #[command(flatten)]
    #[input(group)]
    pub posterior: PosteriorFlags,

    #[command(flatten)]
    #[input(group)]
    pub fixed_point: FixedPointFlags,

    // `--fit` named the run before the run became the positional argument;
    // the dispatch answers it with the command line that replaces it.
    #[arg(long = "fit", value_name = "FIT", hide = true)]
    #[input(skip)]
    pub _removed_fit: Option<String>,
}

/// The flags of `camdl predict` that mean the same thing whatever the
/// prediction's parameters come from.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct PredictFlags {
    // Also on when the fit's own config set it.
    #[command(flatten)]
    #[input(group)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

    /// Restrict to the named streams (repeatable). Each is a logical name
    /// (`onset`, every one of its leaves) or an expanded leaf name (`onset_Bo`).
    /// Without it, every stream the fit bound data to is predicted and a stream
    /// with no bound data is skipped and named; a named stream that cannot be
    /// produced (no bound data, or refused) makes the exit status 1, with the
    /// other streams still written.
    #[arg(long, value_name = "STREAM")]
    #[input(identity, vary = crate::input_fixtures::predict_stream)]
    pub stream: Vec<String>,

    /// Prospective scenario overlay (repeatable; conflicts with --enable/--disable).
    /// Each `--scenario NAME` selects a model `scenarios {}` preset; the free-forward
    /// replay is looped over every scenario and the bands are tagged by a leading
    /// `scenario` column, the same way `horizon`/`treatment` already stack. No
    /// `--scenario` → a single `fitted` row (the fitted model, no overlay).
    /// `fitted` is a reserved name: a preset cannot use it. Layer-1 emits each
    /// scenario's bands only — never a between-scenario difference ("cases averted"
    /// lives in the conditioned counterfactual fork).
    #[arg(long = "scenario", conflicts_with_all = ["enable", "disable"])]
    #[input(identity, vary = crate::input_fixtures::predict_scenario)]
    pub scenarios: Vec<String>,

    /// Enable an intervention in an ad-hoc scenario overlay (repeatable; conflicts
    /// with --scenario). Mirrors `simulate --enable`.
    #[arg(long, conflicts_with = "scenarios")]
    #[input(identity, vary = crate::input_fixtures::predict_intervention)]
    pub enable: Vec<String>,

    /// Disable an intervention in an ad-hoc scenario overlay (repeatable; conflicts
    /// with --scenario). Mirrors `simulate --disable`.
    #[arg(long, conflicts_with = "scenarios")]
    #[input(identity, vary = crate::input_fixtures::predict_intervention)]
    pub disable: Vec<String>,

    /// Vary a parameter across a grid over the posterior (repeatable →
    /// multiple swept params → Cartesian). Each `--sweep PARAM=GRID` sets the
    /// swept parameter to each grid value in turn while the rest of every
    /// posterior draw propagates; cells are keyed by a leading `sweep:<param>`
    /// column. `GRID` is a list (`q=0,30,60`), `lin(min,max,n)`, or
    /// `log10(min,max,n)`. Composes with `--scenario` on DISTINCT parameters; a
    /// scenario and a sweep on the SAME parameter is a hard error (pin OR vary,
    /// not both). It makes arms of the free-forward and forecast predictions;
    /// the one-step rows are sweep-agnostic.
    #[arg(long = "sweep", value_name = "PARAM=GRID")]
    #[input(identity, vary = crate::input_fixtures::predict_sweep)]
    pub sweep: Vec<crate::args::types::SweepSpec>,

    /// Which predictive to emit. Omitted = all applicable for the fit's
    /// backend (chain-binomial → `free_forward` + `one_step`; ODE →
    /// `free_forward` only), or `forecast` when `--horizon` is given.
    /// `--mode one_step` on an ODE fit is a hard error. `forecast` is the
    /// fixed-origin forecast from the end of the fit's training rows, out to
    /// `--horizon`.
    #[arg(long, value_name = "MODE")]
    #[input(identity, vary = crate::input_fixtures::predict_mode)]
    pub mode: Option<crate::args::types::PredictModeArg>,

    /// The forecast's reach, a duration from the end of the fit's training
    /// rows: `4 weeks`, `28 days`, or a bare number in the model's time unit.
    /// Every stream's periods ending within it are forecast; a target's
    /// horizon counts that stream's own periods from the one under way.
    /// Implies `--mode forecast`.
    #[arg(long, value_name = "DURATION")]
    #[input(identity, vary = crate::input_fixtures::predict_horizon)]
    pub horizon: Option<String>,

    /// Forecast mode: the design values (a column the likelihood reads, such
    /// as a binomial's `n = tested`) of stream SOURCE's targets, from FILE, a
    /// table in the stream's own format with its time (or window) columns and
    /// the design columns; the outcome column is not read. A prospective
    /// forecast (repeatable, one per stream). Every target needs a value of
    /// every design column; a missing one is refused.
    #[arg(long, value_name = "SOURCE=FILE", conflicts_with = "realised_design")]
    #[input(identity, vary = crate::input_fixtures::predict_design)]
    pub design: Vec<String>,

    /// Forecast mode: what the values of the data-table forcing FORCING after
    /// the forecast origin are (repeatable, one per forcing). `prospective`:
    /// fixed or assumed at the origin, read as they are. `realised`: observed
    /// after the origin, read as they are, and the forecast is labelled
    /// conditional on covariates observed after its origin. `hold`: the
    /// forcing's value at the origin is carried forward, and its table after
    /// the origin is not read (refused for a cubic-spline table). A forecast
    /// that reads a table forcing after its origin with no statement is
    /// refused; one that reads none takes no `--covariate`.
    #[arg(long = "covariate", value_name = "FORCING=STATEMENT")]
    #[input(identity, vary = crate::input_fixtures::predict_covariate)]
    pub covariate: Vec<crate::args::types::CovariateSpec>,

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
    #[input(identity, vary = crate::input_fixtures::predict_quantities)]
    pub quantities: Option<PathBuf>,

    /// The draws the bands pool over (default 200): from a stored fit, a cap
    /// on the posterior subsample; from a model file, the number of copies of
    /// the point, each under its own seed. Each horizon pools plenty at a few
    /// hundred draws (one-step over `draws × particles`, free-forward over one
    /// forward replay per draw), so a larger posterior cloud is evenly
    /// subsampled — a strided pick across the whole cloud, never silently run
    /// at full size (a full free-forward replay of a long-burn-in ODE fit is
    /// hours of solves).
    #[arg(long, value_name = "N")]
    #[input(identity, vary = crate::input_fixtures::predict_n_draws)]
    pub n_draws: Option<usize>,

    /// Particles per draw in the filters this command runs (the one-step
    /// band's and the forecast's). Default: the fit's own filter's count for a
    /// PMMH fit, else 500. The resolved count is part of the
    /// output directory's key and is recorded in the manifest.
    #[arg(long, value_name = "N")]
    #[input(identity, vary = crate::input_fixtures::predict_particles)]
    pub particles: Option<usize>,

    /// RNG seed for the y_rep observation sampling (default 1).
    #[arg(long)]
    #[input(identity)]
    pub seed: Option<u64>,

    /// Export a copy of the prediction's result files into the directory
    /// DIR, at the paths they have in the prediction's folder. DIR is made
    /// when its parent exists. A prediction reused from the store is
    /// exported the same way.
    #[arg(short, long, value_name = "DIR")]
    #[input(export)]
    pub output: Option<PathBuf>,

    /// Write the output manifest to PATH once the command has succeeded: a
    /// JSON file naming every result it stored or reused, each result file
    /// by logical name with its SHA-256 and size, for a workflow tool to
    /// depend on (run spec §2.10). Any file at PATH is removed when the
    /// command starts, so a failed command leaves none.
    #[arg(long, value_name = "PATH")]
    #[input(export)]
    pub manifest: Option<PathBuf>,

    /// Compute the prediction again even when it is stored; the stored
    /// prediction moves to the store's `.quarantine/` first. A prediction
    /// held by an owner not known to be stopped is refused.
    #[arg(long)]
    #[input(store)]
    pub force: bool,

    /// A name for this invocation, found again as `@LABEL`. It names the
    /// invocation, never the prediction: rerunning with another label reuses
    /// the stored prediction.
    #[arg(long, value_name = "TEXT")]
    #[input(store)]
    pub label: Option<String>,

    /// The run store's root (default `$CAMDL_STORE`, else `./camdl-store`):
    /// where a fixed-point prediction is stored, and where a run handle is
    /// looked up. A run named by its folder is predicted in the store that
    /// holds the folder.
    #[arg(long, value_name = "DIR")]
    #[input(store)]
    pub store: Option<PathBuf>,
}

/// The flags of `camdl predict` that read a fit's posterior: its chains and
/// its held-out rows.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct PosteriorFlags {
    /// Also band each MCMC chain on its own, tagged by a leading `chain` column
    /// (`all` on the pooled rows, the chain id on the per-chain ones, counted
    /// from 0 as in the `chain` column of `draws.tsv`) —
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
    /// keyed directory is the exclusion's (`predictive-excl3_5/`), because a
    /// `--by-chain` file is a strict superset of the pooled one — the `all`
    /// rows are byte-identical — while a chain subset is a different posterior.
    /// Chain ids are the fit's own numbering, never renumbered by an exclusion,
    /// so an excluded chain is simply absent and the ids that remain line up row
    /// for row with the pooled artifact's.
    #[arg(long = "by-chain")]
    #[input(identity)]
    pub by_chain: bool,

    /// Drop the named MCMC chains from the posterior cloud before banding —
    /// a comma-separated list of chain ids, counted from 0 (matching the
    /// `chain` column of `draws.tsv`, the `chain_N/` dirs and the `fit summary`
    /// per-chain table), e.g. `--exclude-chains 3,5`.
    /// The escape hatch for a known-stuck minority of chains; post-hoc exclusion
    /// BIASES the posterior toward the retained mode and always prints a warning.
    /// A chain id not in the fit, or excluding every chain, is a hard error.
    /// The bands, quantities and contrasts land in `predictive-excl<ids>/`,
    /// `quantities-excl<ids>/` and `contrasts-excl<ids>/`, keyed by the excluded
    /// SET (`3,5` and `5,3` are one address), so a subset never overwrites the
    /// pooled artifact and two subsets never overwrite each other.
    #[arg(long, value_name = "IDS")]
    #[input(identity, vary = crate::input_fixtures::predict_chain_ids)]
    pub exclude_chains: Option<String>,

    /// Band over ONLY the named chains (comma-separated chain ids counted
    /// from 0, e.g. `--chains 3,5`): the complement of `--exclude-chains`,
    /// which it cannot be given with. The artifacts are addressed by the
    /// chains dropped, so on a six-chain fit `--chains 3,5` writes the same
    /// `predictive-excl0_1_2_4/` as `--exclude-chains 0,1,2,4`. Always warns;
    /// a chain id not in the fit, or naming every chain, is a hard error.
    #[arg(long, value_name = "IDS", conflicts_with = "exclude_chains")]
    #[input(identity, vary = crate::input_fixtures::predict_chain_ids)]
    pub chains: Option<String>,

    /// Forecast mode: read each target's design values from its held-out row,
    /// realised after the forecast origin. The result is labelled a
    /// conditional score using realised future design, not an unconditional
    /// forecast. Every target needs a held-out row with every design value.
    /// A stored fit keeps its held-out rows' dates and not their values, so
    /// a forecast from one refuses this flag.
    #[arg(long)]
    #[input(identity)]
    pub realised_design: bool,
}

/// The flags of `camdl predict` that pose a prediction from a fixed point:
/// the point, the data it conditions on and the numerics, which a stored fit
/// problem holds for a posterior prediction. Each is refused with a handle.
#[derive(Args, Clone, Default, store::Inputs)]
pub struct FixedPointFlags {
    /// A parameters TOML file (repeatable; later files override earlier
    /// ones), as `simulate` and `pfilter` read it: the point the prediction
    /// is made at. `camdl fit summary RUN --params-only` writes one.
    #[arg(long, value_name = "FILE")]
    #[input(identity, vary = crate::input_fixtures::predict_other_params)]
    pub params: Vec<PathBuf>,

    /// One parameter's value, e.g. `--param beta=0.4` (repeatable); it
    /// overrides every `--params` file.
    #[arg(long, value_name = "NAME=VALUE")]
    #[input(identity, vary = crate::input_fixtures::predict_other_param)]
    pub param: Vec<ParamOverride>,

    /// An external table for table-lookup expressions, e.g.
    /// `--table contact=matrix.tsv` (repeatable).
    #[arg(long, value_name = "NAME=FILE")]
    #[input(identity, vary = crate::input_fixtures::predict_other_table)]
    pub table: Vec<TableSpec>,

    /// The observations, required: `--data PATH` for a model with one
    /// stream (or the one `--obs` names), `--data NAME=PATH` (repeatable)
    /// otherwise, bound as `pfilter` binds them. Every row is parsed, to find
    /// the cut `--holdout-after` makes, so a malformed held-out row is refused;
    /// only the rows up to the cut are stored and used.
    #[arg(long, value_name = "[NAME=]PATH")]
    #[input(identity, vary = crate::input_fixtures::predict_other_data)]
    pub data: Vec<DataSpec>,

    /// The observation stream a single `--data PATH` binds to (required
    /// when the model declares more than one).
    #[arg(long, value_name = "STREAM")]
    #[input(identity, vary = crate::input_fixtures::predict_other_obs)]
    pub obs: Option<String>,

    /// The end of the rows the point was fitted to: a model-time number, a
    /// date, or `last_obs` with an offset (`last_obs - 2 weeks`), with the
    /// grammar and the cut of `fit.toml`'s `[data] holdout_after`. The rows
    /// after it are parsed to find the cut, then neither stored nor used, and
    /// a forecast starts at the cut. A `last_obs - …` cut resolves over the
    /// whole bound files, held-out rows included, as a fit's does.
    #[arg(long, value_name = "TIME")]
    #[input(identity, vary = crate::input_fixtures::predict_holdout_after)]
    pub holdout_after: Option<String>,

    /// The simulation backend (default chain_binomial).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::predict_other_backend)]
    pub backend: Option<ForwardBackend>,

    /// The integration step. Default: the model's `simulate { dt }`, else
    /// 1.0. A value that differs from the model's step is refused: a command
    /// that fits or scores data integrates at the model's step (run spec
    /// §1.3.1).
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_dt)]
    pub dt: Option<f64>,

    /// The ODE integrator method: rk4 (fixed) or rk45 (adaptive), keeping
    /// the model's tolerances. Refused on a backend other than ode.
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::other_integrator)]
    pub integrator: Option<crate::args::types::IntegratorArg>,

    /// The sampler of the chain-binomial draws: `btpe` (default) or `btrs`.
    /// The two are not bit-compatible. Refused on another backend.
    #[arg(long, value_name = "SAMPLER",
          value_parser = |v: &str| v.parse::<sim::rng::BinomialAlgorithm>())]
    #[input(identity, vary = crate::input_fixtures::other_binomial)]
    pub binomial: Option<sim::rng::BinomialAlgorithm>,

    /// A rate that collapses (division by zero, a power that is NaN, the
    /// square root of a negative) is 0 instead of an error.
    #[arg(long, default_value_t = false)]
    #[input(identity)]
    pub allow_degenerate_rates: bool,

    /// Check each `--data` file's time column against a format: `auto`,
    /// `numeric` or `date` (requires the model's `origin`). A format the
    /// column does not hold is refused. The check is all it does: every later
    /// read detects the format from the whole column, so it does not silence
    /// W326 (`internal-days` is checked as `numeric`).
    #[arg(long, default_value = "auto")]
    #[input(execution, exercised_by = crate::input_fixtures::predict_time_format)]
    pub time_format: crate::caltime_load::TimeFormat,
}

impl FixedPointFlags {
    /// The flags given, by their spelling, for the refusal of a fixed-point
    /// flag beside a handle.
    pub fn given(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        let mut push = |set: bool, name: &'static str| {
            if set {
                v.push(name);
            }
        };
        push(!self.params.is_empty(), "--params");
        push(!self.param.is_empty(), "--param");
        push(!self.table.is_empty(), "--table");
        push(!self.data.is_empty(), "--data");
        push(self.obs.is_some(), "--obs");
        push(self.holdout_after.is_some(), "--holdout-after");
        push(self.backend.is_some(), "--backend");
        push(self.dt.is_some(), "--dt");
        push(self.integrator.is_some(), "--integrator");
        push(self.binomial.is_some(), "--binomial");
        push(self.allow_degenerate_rates, "--allow-degenerate-rates");
        push(self.time_format != crate::caltime_load::TimeFormat::Auto, "--time-format");
        v
    }
}

impl PosteriorFlags {
    /// The flags given, by their spelling, for the refusal of a posterior
    /// flag beside a model file.
    pub fn given(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.by_chain {
            v.push("--by-chain");
        }
        if self.exclude_chains.is_some() {
            v.push("--exclude-chains");
        }
        if self.chains.is_some() {
            v.push("--chains");
        }
        if self.realised_design {
            v.push("--realised-design");
        }
        v
    }
}

/// One `camdl predict` as its inputs reach the store: the command's
/// arguments and the global options, each with its class. Built by the
/// dispatch in `main.rs`; the global options are not `PredictArgs`' own.
#[derive(Clone, store::Inputs)]
pub struct PredictInvocation {
    #[input(group)]
    pub args: PredictArgs,
    /// `--no-licm`: compile without loop-invariant code motion. A model
    /// file's compile is the compiled model a fixed-point prediction's
    /// identity holds, so it re-keys the prediction through the model's key.
    /// A posterior prediction runs the fit's stored compile, and the flag
    /// reaches only a `--quantities` vocabulary's compile, which is checked
    /// against the fit's with every hoisted subexpression written back in
    /// place and contributes only its quantities, which hoisting does not
    /// touch; there it changes nothing.
    #[input(identity)]
    pub no_licm: bool,
    /// `--no-ir-cache`: compile a `--quantities` vocabulary rather than reuse
    /// a cached IR, which is the IR a compile emits.
    #[input(execution, exercised_by = crate::input_fixtures::predict_no_ir_cache)]
    pub no_ir_cache: bool,
    #[input(skip)]
    pub verbosity: Option<log::LevelFilter>,
    #[input(skip)]
    pub progress: types::ProgressMode,
    #[input(skip)]
    pub no_progress: bool,
}

/// `camdl score PREDICTION --data STREAM=FILE ...`: score a stored prediction
/// against the observed outcomes of the rows it predicted.
#[derive(Args, Clone, Default, store::Inputs)]
#[command(after_help = "\
A score is stored inside the prediction it scores, at scores/score-<id>/:
scored.tsv (one row per scored row or target), summary.tsv (one row per
stream, mode, role and lead) and score.json. Everything about the prediction
(its parameter source, mode, reach, particles, seed and scenarios) is in its
identity, so `score` takes no flag for it. The outcome files must hold every
row the prediction conditioned on, unchanged.

  camdl predict @sir --mode forecast --horizon \"8 weeks\" --label sir_fc
  camdl score @sir_fc --data cases=london.tsv")]
pub struct ScoreArgs {
    /// The stored prediction: the @label of the `camdl predict` that made it,
    /// a run-id prefix or its folder.
    #[arg(value_name = "PREDICTION")]
    #[input(identity)]
    pub prediction: String,

    /// An outcome file, `STREAM=FILE` for each stream source (repeatable):
    /// for a stored fit, the file the fit was given, held-out rows included;
    /// for a fixed point, the file the prediction was given, extended past
    /// `--holdout-after`. A stream with no outcome file is scored on the rows
    /// the prediction conditioned on only.
    #[arg(long = "data", value_name = "STREAM=FILE", required_unless_present = "data_held_out")]
    #[input(identity)]
    pub data: Vec<String>,

    /// A file of a stream's held-out rows alone, `STREAM=FILE` (repeatable):
    /// every row ends after the rows the prediction conditioned on, which are
    /// read from the store, and the file has the columns of the stream's data
    /// file. It gives the score the prediction would have against the whole
    /// file, and one stream is given by `--data` or by this, not both.
    #[arg(long = "data-held-out", value_name = "STREAM=FILE")]
    #[input(identity)]
    pub data_held_out: Vec<String>,

    /// Export a copy of the score's result files into DIR. DIR is made when
    /// its parent exists.
    #[arg(short, long, value_name = "DIR")]
    #[input(export)]
    pub output: Option<PathBuf>,

    /// Write the output manifest to PATH once the command has succeeded: a
    /// JSON file naming every result it stored or reused, each result file
    /// by logical name with its SHA-256 and size, for a workflow tool to
    /// depend on (run spec §2.10). Any file at PATH is removed when the
    /// command starts, so a failed command leaves none.
    #[arg(long, value_name = "PATH")]
    #[input(export)]
    pub manifest: Option<PathBuf>,

    /// Compute the score again even when it is stored; the stored score
    /// moves to the store's `.quarantine/` first.
    #[arg(long)]
    #[input(store)]
    pub force: bool,

    /// A name for this invocation, found again as `@LABEL`.
    #[arg(long, value_name = "TEXT")]
    #[input(store)]
    pub label: Option<String>,

    /// The run store's root (default `$CAMDL_STORE`, else `./camdl-store`):
    /// where the prediction is looked up. A prediction named by its folder is
    /// scored in the store that holds the folder.
    #[arg(long, value_name = "DIR")]
    #[input(store)]
    pub store: Option<PathBuf>,
}

/// One `camdl score` as its inputs reach the store.
#[derive(Clone, store::Inputs)]
pub struct ScoreInvocation {
    #[input(group)]
    pub args: ScoreArgs,
    #[input(skip)]
    pub verbosity: Option<log::LevelFilter>,
}

/// `camdl fit predict` is removed: a prediction is `camdl predict RUN`. This
/// catch-all accepts any arguments so an old command line is answered with
/// the one that replaces it rather than a clap parse error.
#[derive(Args)]
pub struct FitPredictRemovedArgs {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true, hide = true)]
    pub _ignored: Vec<String>,
}

/// The reserved scenario name for the no-overlay row — the fitted model as
/// written. Reserved so the `scenario` column's no-overlay value can never be
/// shadowed by a user preset (which would make rows ambiguous); the compiler
/// rejects a scenario declared under this name (E291). The counterpart of
/// `simulate`'s `(none)` arm, named `fitted` because the parameters come from
/// the fit. Unlike `(none)`, `fitted` is an identifier, which is why it has to
/// be reserved; `baseline` is not reserved and is an ordinary scenario name.
pub const FITTED: &str = "fitted";

/// Split `--scenario` values on commas, trim, drop empties, and reject repeats.
///
/// One shared check because both verbs mis-handle a repeat, differently and
/// both badly: `simulate` would accumulate two cells into one scenario bucket
/// and fail with "point-mode quantities require exactly one realization" AFTER
/// the whole grid had simulated and committed leaves, while `camdl predict` bands
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

impl PredictFlags {
    /// Parse the repeatable `--scenario`/`--enable`/`--disable` surface into the
    /// shared `Vec<ScenarioRef>`, exactly the way `simulate` does (`main.rs`'s
    /// `from_cli`): `--scenario a,b` comma-splits into `[Named("a"), Named("b")]`
    /// (the preset path); no `--scenario` yields a single [`FITTED`] inline
    /// overlay carrying any `--enable`/`--disable` (so the no-overlay path keeps
    /// `scenario_name = None`, exactly like `simulate`'s `(none)` arm). `fitted` is
    /// reserved: an explicit `--scenario fitted` is rejected with the same
    /// migration-style diagnostic the OCaml `scenarios {}` reservation uses.
    #[cfg(test)]
    pub fn scenario_refs(&self) -> Result<Vec<crate::sim_job::ScenarioRef>, String> {
        self.scenario_refs_named(FITTED)
    }

    /// [`Self::scenario_refs`], with the no-overlay arm named `no_overlay`:
    /// `fitted` for a posterior, `(none)` for a fixed point.
    pub fn scenario_refs_named(&self, no_overlay: &str) -> Result<Vec<crate::sim_job::ScenarioRef>, String> {
        use crate::sim_job::ScenarioRef;
        let names = split_scenario_names(&self.scenarios)?;
        if let Some(bad) = names.iter().find(|n| n.as_str() == FITTED) {
            if no_overlay != FITTED {
                return Err(format!(
                    "scenario name '{bad}' is reserved, so no model declares a scenario by it \
                     (E291); the arm with no scenario applied is `{no_overlay}`, and it is emitted \
                     automatically."
                ));
            }
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
                name: no_overlay.to_string(),
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
                name: no_overlay.to_string(),
                enable: self.enable.clone(),
                disable: self.disable.clone(),
                params: indexmap::IndexMap::new(),
            }];
            refs.extend(names.into_iter().map(ScenarioRef::Named));
            Ok(refs)
        }
    }
}

/// Output format for `camdl diff`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum DiffFormat {
    /// Plain text: each changed declaration by its text as written.
    Table,
    /// The typed report (`DiffReport`) as JSON.
    Json,
}

#[derive(Args)]
#[command(after_help = colored_help!("\
Each of A and B is a model file: a .camdl source, which is compiled, or a
compiled .ir.json. Rows are decided on what the two files declare, and the
hashes of the two compiled models are a cross-check on them.

Exit status: 0 when the two are identical, 1 when they differ, 2 when no
report can be made.

Examples:
  # What differs between two variants of a model
  camdl diff seir.camdl seir_waning.camdl

  # The typed report, for a script or a viewer
  camdl diff seir.camdl seir_waning.camdl --format json
"))]
pub struct DiffArgs {
    /// The first (left) model
    pub a: PathBuf,
    /// The second (right) model
    pub b: PathBuf,
    /// Output format
    #[arg(long, value_enum, default_value_t = DiffFormat::Table)]
    pub format: DiffFormat,
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

/// Format of the table a `camdl profile` run prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileTableFormat {
    Tsv,
    Json,
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
  # Every fit run in the run store, default text view
  camdl fit table

  # Just the converged ones
  camdl fit table --converged

  # Project to one row in JSON for downstream tooling
  camdl fit table --hash 04ab12cd --format json

  # Filter by method, in another store
  camdl fit table runs/camdl-store --with-method pgas
"))]
pub struct FitTableArgs {
    /// The run store's root: every fit run stored there is one row. Without
    /// it, `$CAMDL_STORE`, else `./camdl-store`.
    pub root: Option<PathBuf>,

    /// Show only converged rows (IF2: gate Pass; PGAS / PMMH: max R̂ < 1.05).
    #[arg(long)]
    pub converged: bool,

    /// Show only rows whose convergence boolean is false.
    #[arg(long)]
    pub gate_failed: bool,

    /// Filter to fits whose method matches.
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

    /// Filter to fits with a run whose label matches a glob; an
    /// unlabelled fit is matched as "".
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
    /// predicted yet it runs `camdl predict --mode free_forward`,
    /// populating that run's `quantities-model/` outputs. Optimizer fits
    /// (IF2 / NLopt) have no posterior cloud, so their cell renders `—`.
    #[arg(long = "quantity", value_name = "NAME")]
    pub quantities: Vec<String>,

    /// Drop the named MCMC chains (comma-separated chain ids counted from 0,
    /// as in the `chain` column of `draws.tsv`, e.g. `--exclude-chains 3,5`) from each fit's posterior cloud when DERIVING a
    /// `--quantity` cell. Only affects derived cells, so it requires
    /// `--quantity`; the same drop set is applied to every fit in the cohort
    /// (chain 3 of fit A is unrelated to chain 3 of fit B — pair with `--hash`
    /// to project to one fit). Post-hoc exclusion BIASES the posterior toward
    /// the retained mode and always prints a warning.
    #[arg(long, value_name = "IDS", requires = "quantities")]
    pub exclude_chains: Option<String>,

    /// Derive each `--quantity` cell over ONLY the named chains of each fit
    /// (comma-separated chain ids counted from 0): the complement of
    /// `--exclude-chains`, which it cannot be given with. Requires
    /// `--quantity`; the same ids are applied to every fit. Always warns.
    #[arg(long, value_name = "IDS", requires = "quantities", conflicts_with = "exclude_chains")]
    pub chains: Option<String>,
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
A stored result's record is sealed, so a result is never labelled after it
ran, and this command refuses every handle. A label names the invocation that
carries it: pass `--label L` to the command that computes the results, and
`@L` names them. Running the same command again with `--label L` reuses the
stored results and names them @L.

Examples:
  camdl fit run fit.toml --label baseline-2024
  camdl show @baseline-2024
"))]
pub struct LabelArgs {
    /// A handle of a stored result: a path, an id prefix, or `@L`
    pub hash: String,

    /// The label, 1-64 characters after trim: letters, digits, spaces,
    /// commas, dot, underscore and hyphen
    pub label: String,
}

// ─── pfilter ──────────────────────────────────────────────────────────────────

#[derive(Args, Clone, store::Inputs)]
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

  # Loglik at every row of a design (one table row per point)
  camdl pfilter sir.camdl --fit fit.toml --sweep-file design.tsv \\
      --particles 1000 --replicates 5 -o landscape.tsv
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
    #[input(identity, vary = crate::input_fixtures::other_model)]
    pub model: PathBuf,

    #[command(flatten)]
    #[input(group)]
    pub model_overrides: ModelOverrides,

    #[command(flatten)]
    #[input(group)]
    pub scenario: ScenarioArgs,

    /// Number of particles
    #[arg(long)]
    #[input(identity)]
    pub particles: usize,

    #[command(flatten)]
    #[input(group)]
    pub inference: PfilterCore,

    #[command(flatten)]
    #[input(group)]
    pub stream: StreamSelection,

    /// Observation data TSV (with time column).
    ///
    /// Two forms (mutually exclusive within a single invocation):
    ///   --data PATH         single-stream: binds to the model's only
    ///                       observation block (or the one selected by
    ///                       --obs NAME).
    ///   --data NAME=PATH    multi-stream: bind one observation block by
    ///                       name. Repeat for every stream.
    /// Mixing the two forms is a hard error. Multi-stream models must
    /// bind every block (warning fires when only a subset is bound).
    #[arg(long, value_name = "[NAME=]PATH")]
    #[input(identity, vary = crate::input_fixtures::other_data)]
    pub data: Vec<DataSpec>,

    /// Optional fit toml. Supplies the multi-stream binding via its
    /// `[data.observations]` map when no `--data` flags were supplied (CLI
    /// flags win), the values of its `[fixed]` parameters (`--params`, a
    /// scenario and `--param` override them), and always the likelihood
    /// objective: under `ic_free = true` the first observation's term is left
    /// out, as `fit run` scores it. Omit `--fit` for the full likelihood.
    #[arg(long, value_name = "PATH")]
    #[input(identity, vary = crate::input_fixtures::other_fit)]
    pub fit: Option<PathBuf>,

    /// Number of independent filter runs
    #[arg(long, default_value_t = 1)]
    #[input(identity)]
    pub replicates: usize,

    /// A TSV of design points (header: parameter names, optional `point_id`);
    /// one log-likelihood row per point, each point stored as the single
    /// call at its values.
    #[arg(long, value_name = "FILE")]
    #[input(expands)]
    pub sweep_file: Option<PathBuf>,

    /// Export the per-observation diagnostics TSV, a by-product of the run,
    /// to this path; use "-" for stdout
    #[arg(long)]
    #[input(adds)]
    pub trace: Option<String>,

    /// Export a particle-filter health report, a by-product of the run:
    /// per-observation ESS and Snyder τ² (log-weight variance), plus a
    /// printed summary with the implied particles-to-avoid-collapse estimate
    /// exp(τ²/2). Use "-" for stdout.
    #[arg(long)]
    #[input(adds)]
    pub pf_health: Option<String>,

    /// Write the log-likelihood (each replicate's under `--replicates`) to
    /// this file instead of stdout
    #[arg(short, long)]
    #[input(export)]
    pub output: Option<PathBuf>,

    /// Export the final particle states, a by-product of the run, to this
    /// TSV. Under a `--fit` config's `holdout_after`, the run binds only the
    /// training rows (those ending at or before the cutoff) and saves the
    /// state at their end; its log-likelihood and every other output then
    /// cover the training rows only, and `--score-from` is refused beside a
    /// cut that dropped rows.
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::a_final_state)]
    pub save_final_state: Option<PathBuf>,

    /// Export N trajectory samples from the smoothing distribution, a
    /// by-product of the run, to this path
    #[arg(long)]
    #[input(adds)]
    pub save_paths: Option<PathBuf>,

    /// Number of trajectories for --save-paths
    #[arg(long, default_value_t = 1)]
    #[input(identity)]
    pub n_paths: usize,

    /// Export per-step particle states and log-weights, a by-product of the
    /// run, to this TSV
    #[arg(long)]
    #[input(adds)]
    pub save_filtering: Option<PathBuf>,

    /// Export, per observation step and stream, the weighted mean and
    /// 5/25/50/75/95% quantiles over the particles of the stream's conditional
    /// mean E[y_t | x_t, θ] (e.g. rho × the step's accumulated flow), twice:
    /// `predictive` (before y_t is scored, given the data through the step
    /// before) and `filtered` (after, given the data through t). Columns:
    /// time, stream, phase, quantity (always `expectation`), mean, q05, q25,
    /// q50, q75, q95. A by-product of the run.
    #[arg(long)]
    #[input(adds)]
    pub save_filtered_streams: Option<PathBuf>,

    /// Export {STEM}.tsv (the joint log score and ESS per step),
    /// {STEM}.streams.tsv (each stream's observed value, predictive band,
    /// log score, CRPS and PIT per step) + {STEM}.json (full typed
    /// PrequentialTrace) for the plug-in one-step-ahead predictive at the
    /// fixed parameters, a by-product of the run. See
    /// docs/dev/proposals/2026-04-20-prequential-evaluation.md.
    #[arg(long)]
    #[input(adds)]
    pub save_prequential: Option<String>,

    /// With --save-prequential, score the trace only at observations
    /// strictly after TIME: earlier observations are still assimilated (the
    /// filter reweights on them) but excluded from the trace. Accepts the
    /// shared time grammar: a model-time
    /// number, a date under a calendar-anchored model, or
    /// `last_obs - N weeks`. The total log-likelihood output is unchanged;
    /// only the trace is windowed, and it records the boundary
    /// (`score_from`).
    #[arg(long, value_name = "TIME", requires = "save_prequential")]
    #[input(identity, vary = crate::input_fixtures::a_score_from)]
    pub score_from: Option<String>,

    /// With --save-prequential, drop per-particle predictive samples
    /// from {STEM}.json. Keeps scalar scores, shrinks the file.
    #[arg(long)]
    #[input(identity)]
    pub no_save_samples: bool,

    /// Compute in a temporary store, deleted when the command ends: the
    /// log-likelihood is printed and every export is written, and nothing is
    /// kept in the run store.
    #[arg(long)]
    #[input(store)]
    pub no_store: bool,

    /// Compute the run and its by-products again, even when they are stored:
    /// the stored folder moves to the store's `.quarantine/` first. A run held
    /// by an owner not known to be stopped is refused; --force takes it unless
    /// its owner is on this host and running, or on another host with a fresh
    /// beat. Use it on a run held from another host only once that process is
    /// known to have stopped.
    #[arg(long)]
    #[input(store)]
    pub force: bool,

    /// A name for this invocation, found again as `@LABEL`. It names the
    /// invocation, never the result: rerunning with another label reuses the
    /// stored result.
    #[arg(long)]
    #[input(store)]
    pub label: Option<String>,

    /// The run store's root, over the `store` key of a `--fit` file (default
    /// `$CAMDL_STORE`, else `./camdl-store`; run spec §2.1).
    #[arg(long, value_name = "DIR")]
    #[input(store)]
    pub store: Option<PathBuf>,

    /// Write the output manifest to PATH once the command has succeeded: a
    /// JSON file naming every result it stored or reused, each result file
    /// by logical name with its SHA-256 and size, for a workflow tool to
    /// depend on (run spec §2.10). Any file at PATH is removed when the
    /// command starts, so a failed command leaves none.
    #[arg(long, value_name = "PATH", conflicts_with = "no_store")]
    #[input(export)]
    pub manifest: Option<PathBuf>,
}

/// One `camdl pfilter` as its inputs reach the store: the command's
/// arguments and the global options, each with its class. Built by the
/// dispatch in `main.rs`; the global options are not `PfilterArgs`' own.
#[derive(Clone, store::Inputs)]
pub struct PfilterInvocation {
    #[input(group)]
    pub args: PfilterArgs,
    /// `--no-licm`: compile without loop-invariant code motion. It changes
    /// the compiled model, whose bytes the run's identity holds, so it
    /// reaches the run id through the model and through nothing else.
    #[input(identity)]
    pub no_licm: bool,
    /// `--no-ir-cache`: compile rather than reuse a cached IR, which is the
    /// IR a compile emits.
    #[input(execution, exercised_by = crate::input_fixtures::pfilter_no_ir_cache)]
    pub no_ir_cache: bool,
    #[input(skip)]
    pub verbosity: Option<log::LevelFilter>,
    #[input(skip)]
    pub progress: types::ProgressMode,
    #[input(skip)]
    pub no_progress: bool,
}

// ─── if2 (removed; deprecation stub) ────────────────────────────────────────────

/// `camdl if2` is removed (gh#147). A one-method IF2 fit is now a fit
/// with a single `algorithm = "if2"` method, run through `camdl fit run`.
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

#[derive(Args, Clone, store::Inputs)]
#[command(after_help = colored_help!("\
Examples:
  # 1D profile likelihood for R0 of the problem fit.toml poses, IF2 in
  # each cell; the problem is stored as `camdl fit run fit.toml` stores it
  camdl profile sir.camdl --fit fit.toml \\
      --sweep \"R0=lin(0.5,5,20)\" --particles 2000

  # 2D profile (R0 × sigma)
  camdl profile sir.camdl --fit fit.toml \\
      --sweep \"R0=lin(0.5,5,10)\" --sweep \"sigma=lin(0.1,1.0,10)\" \\
      --particles 2000

  # A profile of a stored fit's problem
  camdl profile sir.camdl --fit @scout --sweep \"R0=lin(0.5,5,20)\" --particles 2000

  # A conditional profile: hold gamma at 0.1, which takes it out of the
  # problem's [estimate], sweep tau
  camdl profile sir.camdl --fit fit.toml \\
      --sweep \"tau=lin(-35,-1,30)\" --fixed gamma=0.1 --particles 2000

  # Deterministic per-cell MLE on the ODE backend
  camdl profile sir.camdl --fit fit.toml \\
      --sweep \"gamma=lin(0.08,0.2,13)\" --algorithm nl-sbplx --backend ode

  # Warm-start chains from a hand-written params TOML of estimated values
  camdl profile sir.camdl --fit fit.toml \\
      --sweep \"R0=lin(0.5,5,20)\" --starts from_params=truth.toml \\
      --n-starts 4 --particles 2000

WHAT EACH CELL COMPUTES

  A profile likelihood: at each grid value of the swept parameter,
  the maximum of the likelihood over the other estimated parameters.
  Every cell is a maximization (--algorithm if2, or an nl-* optimizer
  on --backend ode); priors are not used. For a posterior of the
  swept parameter, fit once with `camdl fit run` (pmmh or pgas) and
  read its marginal.

OUTPUT

  Each point (cell x seed x start) is stored in the run store
  (`--store`, else `$CAMDL_STORE`, else ./camdl-store) under
  `profile_point/`, with an `mle.toml` (final_loglik, [focal], [mle],
  [diagnostics]). A point's identity holds its focal values and its
  start's values, never the grid, so a larger grid reuses every point
  already stored. When the run ends it prints one table, to stdout or
  to `--output FILE`, one row per grid cell, rendered from the stored
  points (so a rerun served from the store prints the same table):

    <focal_1> ... <focal_N>  loglik  <param_1> ... <param_K>
      loglik_spread_starts  starts_n_completed  n_starts  run_id

    loglik                  best finite final log-likelihood over the
                            cell's seeds and starts (NaN if none)
    <param_k>               the estimates at that start
    loglik_spread_starts    max - min of the finite starts' log-
                            likelihoods. > ~5 nats means the starts
                            disagree on the basin.
    starts_n_completed      starts with a finite log-likelihood
    n_starts                starts run at the cell (seeds x --n-starts)
    run_id                  the best start's run id

  `--format json` writes the same rows as {focal, estimated,
  likelihood_objective, loglik_type, cells: [...]}, null for a missing
  value. Full reference: docs/inference.md, `The profile table`.

  --dt: every cell (IF2 and nl-*) integrates at the model's own `dt` when
  it declares one, as `fit run` does; a different --dt, or a different
  `[config] dt` in the --fit file, is refused rather than ignored.
"))]
pub struct ProfileArgs {
    /// IR JSON or .camdl model file
    #[input(identity, vary = crate::input_fixtures::profile_other_model)]
    pub model: PathBuf,

    #[command(flatten)]
    #[input(group)]
    pub model_overrides: InferenceModelOverrides,

    #[command(flatten)]
    #[input(group)]
    pub scenario: ScenarioArgs,

    #[command(flatten)]
    #[input(group)]
    pub inference: ProfileCore,

    /// The stream a bare `--data FILE` binds to. Refused: a profile's
    /// problem binds its own streams.
    #[arg(long)]
    #[input(skip)]
    pub obs: Option<String>,

    /// Observation data TSV. Refused: a profile is of the problem `--fit`
    /// names, whose `[data]` binds the data.
    #[arg(long, value_name = "[NAME=]PATH")]
    #[input(skip)]
    pub data: Vec<DataSpec>,

    /// The fit problem the profile is of: a fit file (its problem is
    /// resolved and stored as `camdl fit run` stores it) or a stored fit
    /// (`@LABEL`, an id, a folder). Required. A swept parameter
    /// belongs in `[estimate]`, whose entry bounds the grid; each cell pins
    /// it at its grid value and estimates the rest. Listing it in `[fixed]`
    /// is refused. Priors do not
    /// enter the per-cell maximization; `--starts from_prior` draws
    /// starting points from them. The config's holdout applies to the
    /// bound data, `--data` included, as in `fit run`: `[data] holdout_after`
    /// removes every row ending after the cutoff from every cell's likelihood, and
    /// `[data.holdout]` files must lie after the training rows.
    #[arg(long, value_name = "PATH")]
    #[input(identity, vary = crate::input_fixtures::profile_other_fit)]
    pub fit: Option<PathBuf>,

    /// Profile grid (repeat for 2D+).
    /// SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC", required = true)]
    #[input(expands)]
    pub sweep: Vec<SweepSpec>,

    /// Particles per IF2 cell. Required for `--algorithm if2`; refused for
    /// an nl-* cell, which runs no particle filter.
    #[arg(long)]
    #[input(identity)]
    pub particles: Option<usize>,

    /// IF2 iterations per grid point (default 50). Refused for an nl-* cell.
    #[arg(long)]
    #[input(identity)]
    pub iterations: Option<usize>,

    /// Independent IF2 starts per grid point — a count, not a rule.
    /// `--starts` is the rule the chains are drawn from; this is how
    /// many are drawn in each cell.
    #[arg(long = "n-starts", value_name = "N", default_value_t = 3)]
    #[input(identity)]
    pub n_starts: usize,

    /// Where each cell's chains begin — the same rule grammar
    /// `camdl fit run --starts` takes. A rule name —
    /// `uniform_unconstrained`, `lhs`, `uniform`, `from_prior`,
    /// `single` — or a sourced rule with its source:
    /// `from_posterior=@base` (one posterior row per chain),
    /// `from_mle=@mle` (every chain at that fit's estimate),
    /// `from_params=theta.toml`. The chains it draws span the
    /// non-focal estimated parameters' bounds, and a source binds those
    /// parameters only: a swept parameter, pinned in every cell, needs no
    /// column or entry in it. Without this flag, the `--fit` config's
    /// `[method] starts`; without either, the default `fit run` takes:
    /// `from_prior` if every parameter a cell estimates has a prior,
    /// `uniform_unconstrained` otherwise. The run says which. Keyed into the
    /// run's identity. How many chains:
    /// `--n-starts`.
    #[arg(long, value_name = "SPEC")]
    #[input(identity, vary = crate::input_fixtures::profile_starts)]
    pub starts: Option<crate::fit::starts::ChainStarts>,

    /// Restrict the `from_posterior` rule in force (`--starts`, or the `--fit`
    /// config's `[method] starts`) to these chains of its source,
    /// comma-separated (`0,1,2,4`), as on `fit run`. Any other rule,
    /// the default included, is refused. The selected chains must agree
    /// (their own R̂ is checked in place of the source's whole-run verdict),
    /// and the subset is keyed into the run's identity (gh#1056).
    #[arg(long, value_name = "IDS", value_parser = crate::fit::starts::parse_starts_chains)]
    #[input(identity, vary = crate::input_fixtures::profile_starts_chains)]
    pub starts_chains: Option<crate::chain_selection::ChainSelection>,

    /// Start from a `from_mle` / `from_posterior` source whose stored verdict
    /// (or, under `--starts-chains`, whose selected chains' R̂) is not
    /// converged, as on `fit run`. Without this the source is refused, since
    /// starting from an unconverged fit launders its multi-modality into
    /// every cell. Refused on any other rule, which has no verdict to lift.
    #[arg(long)]
    #[input(execution, exercised_by = crate::input_fixtures::profile_allow_nonconverged_source)]
    pub allow_nonconverged_source: bool,

    // The `--init` family the `[stages]` -> `[method]` split removed on
    // `fit run`, and gh#889 removes here: accepted by clap so the error can
    // name the replacement, trapped before any work runs. Per CLAUDE.md's
    // alpha posture these are not back-compat shims — nothing behind them
    // works — they exist so one message corrects a habitual invocation.
    #[arg(long = "init", value_name = "MODE", hide = true)]
    #[input(skip)]
    pub _removed_init: Option<String>,
    #[arg(long = "posterior", value_name = "PATH", hide = true)]
    #[input(skip)]
    pub _removed_posterior: Option<String>,
    #[arg(long = "mle", value_name = "PATH", hide = true)]
    #[input(skip)]
    pub _removed_mle: Option<String>,
    #[arg(long = "params", value_name = "TOML", hide = true)]
    #[input(skip)]
    pub _removed_params: Option<String>,
    // The `--pmmh-*` flags went with per-cell PMMH; same shape, one message
    // (answered in `cmd_profile` before any file is read).
    #[arg(long = "pmmh-steps", value_name = "N", hide = true)]
    #[input(skip)]
    pub _removed_pmmh_steps: Option<String>,
    #[arg(long = "pmmh-particles", value_name = "N", hide = true)]
    #[input(skip)]
    pub _removed_pmmh_particles: Option<String>,
    #[arg(long = "pmmh-rho", value_name = "F", hide = true)]
    #[input(skip)]
    pub _removed_pmmh_rho: Option<String>,

    /// IF2 cooling fraction (default 0.95): the perturbation SD reaches this
    /// fraction of its initial value at iteration `--cooling-target-iters`,
    /// then keeps cooling (pomp's `cooling.fraction.50`, the same schedule
    /// `fit run`'s `cooling` means). Refused for an nl-* cell.
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::profile_other_cooling)]
    pub cooling: Option<f64>,

    /// The iteration at which `--cooling` is reached (default 50, pomp's
    /// `cooling.fraction.50`; `fit run`'s `cooling_target_iters`). Refused
    /// for an nl-* cell.
    #[arg(long, value_name = "N")]
    #[input(identity, vary = crate::input_fixtures::profile_other_cooling_target_iters)]
    pub cooling_target_iters: Option<usize>,

    /// Without it each cell estimates the problem's `[estimate]` less the
    /// swept and pinned parameters, and an IF2 cell steps each by its
    /// `[estimate]` `rw_sd`, or the automatic default from its bounds, as
    /// `fit run` does. Given, it names the estimated set and overrides the
    /// steps: `NAME=SD,...`, `NAME=auto,...`, or `auto` (every parameter not
    /// swept, pinned or a model constant). A list must name every
    /// `[estimate]` parameter that is not swept or pinned. An nl-* cell
    /// perturbs nothing, so it takes names only (`NAME=auto` or `auto`) and
    /// refuses an SD.
    #[arg(long)]
    #[input(identity, vary = crate::input_fixtures::profile_other_rw_sd)]
    pub rw_sd: Option<RwSd>,

    /// Where the collected table goes (default: stdout): one row per grid
    /// cell with the swept values, the best log-likelihood over the cell's
    /// seeds and starts, the estimates at that start, the spread of the
    /// finite starts, how many completed out of how many ran, and the best
    /// start's run id. Rendered from the stored points, so a rerun served
    /// from the store writes the same table.
    #[arg(short, long)]
    #[input(export)]
    pub output: Option<PathBuf>,

    /// Format of the collected table: `tsv` (default) or `json`.
    #[arg(long, value_enum, default_value_t = ProfileTableFormat::Tsv)]
    #[input(render, vary = crate::input_fixtures::profile_json)]
    pub format: ProfileTableFormat,

    /// Multi-seed sensitivity: run the entire profile grid at each
    /// seed in this list. Surfaces stochastic IF2 instability per
    /// grid point (high spread across seeds → that cell's MLE is not
    /// trustworthy from a single chain). When omitted, falls back to
    /// `--seed` for a single-seed run. Accepts comma list `1,2,3` or
    /// inclusive range `1:5`.
    #[arg(long, value_name = "SPEC")]
    #[input(expands)]
    pub seeds: Option<SeedSpec>,

    /// A name for this invocation, found again as `@LABEL` (1–64 chars after
    /// trim; allowed: letters, digits, spaces, commas, dot, underscore,
    /// hyphen). It names the invocation and every point it asked for, never
    /// a point: rerunning under another label reuses every stored point.
    #[arg(long, value_name = "TEXT")]
    #[input(store)]
    pub label: Option<String>,

    /// Compute every point again, even when it is stored: the stored folder
    /// moves to the store's `.quarantine/` first. A point held by an owner
    /// not known to be stopped is refused; --force takes it unless its owner
    /// is on this host and running, or on another host with a fresh beat.
    #[arg(long)]
    #[input(store)]
    pub force: bool,

    /// The run store's root, over the `store` key of a `--fit` file (default
    /// `$CAMDL_STORE`, else `./camdl-store`; run spec §2.1).
    #[arg(long, value_name = "DIR")]
    #[input(store)]
    pub store: Option<PathBuf>,

    /// Write the output manifest to PATH once the command has succeeded: a
    /// JSON file naming every result it stored or reused, each result file
    /// by logical name with its SHA-256 and size, for a workflow tool to
    /// depend on (run spec §2.10). Any file at PATH is removed when the
    /// command starts, so a failed command leaves none.
    #[arg(long, value_name = "PATH")]
    #[input(export)]
    pub manifest: Option<PathBuf>,

    /// Per-cell inference algorithm. When omitted the per-cell runner
    /// uses `if2` on `chain_binomial` (the historical default). Pass an
    /// algorithm + backend pair from `camdl fit methods` to switch
    /// (e.g. `--algorithm nl-sbplx --backend ode` for deterministic
    /// per-cell MLE — typically 100×–1000× faster on equilibrium /
    /// large-population fits where the PF is structurally redundant).
    /// Validated against the methods registry at startup; invalid pairs
    /// error with an actionable suggestion.
    #[arg(long, value_name = "NAME")]
    #[input(identity, vary = crate::input_fixtures::profile_other_algorithm)]
    pub algorithm: Option<String>,

    /// Simulation backend. Defaults to `chain_binomial`; pass
    /// `--backend ode` together with an ODE-compatible algorithm
    /// (`nl-sbplx`, `nl-bobyqa`, `nl-lbfgs`) for deterministic per-cell MLE.
    #[arg(long, value_name = "NAME")]
    #[input(execution, exercised_by = crate::input_fixtures::profile_backend)]
    pub backend: Option<String>,
}

/// One `camdl profile` as its inputs reach the store: the command's
/// arguments and the global options, each with its class. Built by the
/// dispatch in `main.rs`; the global options are not `ProfileArgs`' own.
#[derive(Clone, store::Inputs)]
pub struct ProfileInvocation {
    #[input(group)]
    pub args: ProfileArgs,
    /// `--no-licm`: compile without loop-invariant code motion. It changes
    /// the compiled model, whose bytes every point's identity holds.
    #[input(identity)]
    pub no_licm: bool,
    /// `--no-ir-cache`: compile rather than reuse a cached IR, which is the
    /// IR a compile emits.
    #[input(execution, exercised_by = crate::input_fixtures::profile_no_ir_cache)]
    pub no_ir_cache: bool,
    #[input(skip)]
    pub verbosity: Option<log::LevelFilter>,
    #[input(skip)]
    pub progress: types::ProgressMode,
    #[input(skip)]
    pub no_progress: bool,
}

// ─── design ───────────────────────────────────────────────────────────────────

/// `camdl design lhs MODEL --fit FIT.toml --n N`: a Latin-hypercube design
/// over a fit config's estimated parameters, written as a `--sweep-file` TSV.
#[derive(Args)]
#[command(after_help = colored_help!("\
EXAMPLES

  # 200 points over the search boxes of fit.toml's [estimate] parameters
  camdl design lhs model.camdl --fit fit.toml --n 200 --seed 1 -o design.tsv

  # Simulate the model at every point of the design
  camdl simulate model.camdl --sweep-file design.tsv

  # Score the log-likelihood at every point of the design
  camdl pfilter model.camdl --fit fit.toml --sweep-file design.tsv \\
      --particles 1000 -o landscape.tsv

OUTPUT

  A TSV: a `point_id` column (0-based), then one column per estimated
  parameter, in name order. Values are
  on the natural scale and read back to the same floating-point numbers.
  The same model, fit config, --n and --seed give the same bytes. Nothing is
  written to the store.

  Each parameter's search box is its `bounds` in [estimate], else the range
  the model declares. The box is cut into --n bins of equal width, on the log
  scale for a positive parameter whose lower bound is above zero and on the
  linear scale otherwise, and each bin holds one point. A parameter whose
  box has an infinite end is refused.
"))]
pub struct DesignLhsArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    /// The fit config (or a stored fit's handle) whose [estimate] parameters
    /// are sampled, each within its search box. The model sampled is MODEL;
    /// the config's [model] is not read.
    #[arg(long)]
    pub fit: PathBuf,

    /// Number of design points, at least 2.
    #[arg(long)]
    pub n: usize,

    /// RNG seed
    #[arg(long, default_value_t = 1)]
    pub seed: u64,

    /// Write the design to this file (default: stdout).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
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
      --expr \"seasonal\" --from 0 --to 365
"))]
pub struct EvalArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    #[command(flatten)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

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
    ///
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
    /// The run store's root (default `$CAMDL_STORE`, else `./camdl-store`)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,

    /// Filter by model path substring
    #[arg(long)]
    pub model: Option<String>,

    /// Filter by scenario name
    #[arg(long)]
    pub scenario: Option<String>,

    /// Show only runs created within this duration (e.g. 1h, 30m, 2d)
    #[arg(long)]
    pub since: Option<ListDuration>,

    /// Filter by run kind: sim, fit, profile, pfilter, or all (default).
    #[arg(long, default_value = "all")]
    pub kind: String,

    /// Maximum number of results to display
    #[arg(long, default_value_t = 50, conflicts_with = "all")]
    pub limit: usize,

    /// Show all results (no limit)
    #[arg(long)]
    pub all: bool,

    /// Also list the earlier holders of each fit label, as `L~1`, `L~2`, …
    /// (the handles `@L~N` that name them). Without it a label shows only on
    /// its current holder.
    #[arg(long)]
    pub history: bool,

    /// List every fit run on its own line. Without it, unlabelled runs of one
    /// method in one problem folder that differ only in seed share a line,
    /// with the seed range and a combined verdict.
    #[arg(long)]
    pub runs: bool,

    /// Output format: human (default) or json
    #[arg(long)]
    pub format: Option<String>,
}


#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Resolve a run by short hash prefix
  camdl show abc1234

  # A label, or a result's folder in the store, also works
  camdl show @scout

  # JSON output for scripting
  camdl show abc1234 --format json
"))]
pub struct ShowArgs {
    /// Short hash prefix or path to run directory
    pub target: String,

    /// The run store's root (default `$CAMDL_STORE`, else `./camdl-store`)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,

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

    /// The run store's root (default `$CAMDL_STORE`, else `./camdl-store`)
    #[arg(long, value_name = "DIR")]
    pub store: Option<PathBuf>,

    /// Print a named stream instead of the run's default output: for a
    /// simulation, an observation stream (`cases`) or the event log
    /// (`event_log.tsv`); for a filter run, a file it or one of its saved
    /// files holds (`trace.tsv`, `paths.tsv`); for a fit or another run, a
    /// file it wrote (`draws.tsv`)
    #[arg(long)]
    pub stream: Option<String>,

    /// Print only the rows of the named chains (comma-separated chain ids
    /// counted from 0) — for a fit stream whose recorded `output_schema` has a
    /// `chain` column, in practice `draws.tsv`; refused for any other stream.
    /// Cannot be given with `--exclude-chains`. Always warns on stderr.
    #[arg(long, value_name = "IDS", requires = "stream", conflicts_with = "exclude_chains")]
    pub chains: Option<String>,

    /// Print the rows of every chain except the named ones — the complement
    /// of `--chains`, under the same rules.
    #[arg(long, value_name = "IDS", requires = "stream")]
    pub exclude_chains: Option<String>,
}


// ─── compare ──────────────────────────────────────────────────────────────────

/// `camdl compare` — the stored scores of several models, differenced
/// against a baseline.
///
/// Takes ≥2 stored scores, predictions or fit runs (or prequential.json
/// files), or a compare.toml, and renders a baseline-centered comparison of
/// their scores in one mode, one table per role and lead.
#[derive(Args)]
#[command(after_help = colored_help!("\
Columns:
  role       Which rows are scored, against the rows the parameters were
             fitted to: trained, held_out, future (a forecast's targets) or
             unknown (a fixed point with no --holdout-after).
  lead       A forecast's lead: the time from its origin to the end of the
             scored rows. One table per lead; forecast only.
  T_score    Number of scored observations (after the t0 burn-in).
             Differs across fits if they were evaluated on different data
             horizons — Δ columns are suppressed in that case unless
             --allow-mismatched-horizon is passed.
  elpd       Expected log predictive density, summed across scored
             steps:  Σ_t log p̂(y_t | y_{1:t-1}). Higher = better.
  Δelpd      elpd(this) − elpd(baseline). Positive = this model beats
             the baseline. Paired over the same observations.
  LR         exp(Δelpd), the likelihood ratio against the baseline: how
             many times more probability this model assigned to the
             scored observations. Values < 1 favour the baseline; > 1
             favour this model. It is computed at a θ̂ fit to the same
             observations, so it is optimistic and biased toward the
             more flexible model. It is not a Bayes factor (no
             integration over θ) and not an e-value (no anytime-valid
             guarantee); read it with se(Δ).
  se(Δ)      Paired standard error of Δelpd from pointwise differences:
             √(T · Var_t(ℓ^A_t − ℓ^B_t))  (Vehtari/Gelman/Gabry).
             Rule of thumb: |Δelpd| > 2·se → 'the gap is real';
             smaller → inconclusive on this data alone.
  evidence   Δelpd restated in decibans with a qualitative tier (Jeffreys
             scale, extended by camdl), or 'within noise' when
             |Δelpd| < 2·se(Δ). The tiers calibrate Bayes factors; the
             quantity here is an in-sample likelihood ratio.
  crps       Mean Continuous Ranked Probability Score across scored
             steps. Lower = sharper predictive, correctly calibrated.
  Δcrps      Mean CRPS difference (this − baseline). Negative = this
             model's predictive is sharper-at-the-observation.
  wis        Mean weighted interval score of the predictive quantiles, on
             the forecast hubs' quantile set. Lower = better.
  Δwis       Mean WIS difference (this − baseline).
  PIT_cov90  Fraction of observations whose probability integral
             transform fell in the central 90% predictive interval.
             Nominal 0.90 under correct calibration. < 0.70 triggers
             an overconfidence warning below the table.
  cov50      Fraction of observations inside the central 50% interval of
             the predictive samples. Nominal 0.50.

Examples:
  # Compare two fits on their held-out rows, one-step ahead: each fit is
  # predicted and scored (or its stored score reused) and the stored
  # scores are differenced
  camdl compare @sir @seir --data cases=london.tsv

  # The same fits on their training rows
  camdl compare @sir @seir --in-sample

  # Two stored scores, as `camdl score` stored them
  camdl compare @sir_score @seir_score

  # Forecasts from the end of each fit's training rows, one table per lead
  camdl compare @sir @seir --mode forecast --horizon \"4 weeks\" --data cases=london.tsv

  # Two prequential.json files, read as they are
  camdl compare fits/a/pf fits/b/pf --format md

  # Reproducible preset via compare.toml
  camdl compare --config compare.toml
"))]
pub struct CompareArgs {
    /// Models to compare — need ≥2 when --config is not used. Each is a
    /// stored score (the @label of a `camdl score`, a run-id prefix or its
    /// folder), a stored prediction, or a stored fit run, or a
    /// prequential.json (or a directory holding one), read as it is. A
    /// prediction is scored, and a fit run predicted and scored, in the
    /// mode `--mode` asks for; each is reused when stored and stored when
    /// computed.
    pub paths: Vec<String>,

    /// The mode of prediction whose scores are compared: `one_step`,
    /// `free_forward` or `forecast` (which needs --horizon). Default: one-step
    /// on a backend that runs a particle filter, free-forward otherwise; for a
    /// stored score or prediction, the first of those it holds, else forecast.
    #[arg(long, value_enum)]
    pub mode: Option<crate::args::types::PredictModeArg>,

    /// A forecast's reach past its origin, the end of each fit's training
    /// rows (e.g. "4 weeks"): the `--horizon` of the `camdl predict` it
    /// stands for. Needs --mode forecast, or sets it.
    #[arg(long, value_name = "DURATION")]
    pub horizon: Option<String>,

    /// An outcome file, `STREAM=FILE` for each stream source (repeatable),
    /// read by every compared prediction that binds the stream: the files the
    /// fits were given, held-out rows included. Needed to score the held-out
    /// rows of a fit, or a forecast's targets, unless their score is stored.
    #[arg(long = "data", value_name = "STREAM=FILE")]
    pub data: Vec<String>,

    /// A file of a stream's held-out rows alone, `STREAM=FILE` (repeatable),
    /// read by every compared prediction that binds the stream: `camdl score
    /// --data-held-out`. One stream is given by --data or by this, not both.
    #[arg(long = "data-held-out", value_name = "STREAM=FILE")]
    pub data_held_out: Vec<String>,

    // The `--allow-forcing-extrapolation` of the predictions compare makes.
    #[command(flatten)]
    pub forcing_extrapolation: ForcingExtrapolationArg,

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

    /// Metrics to display (comma-separated: elpd, crps, wis, pit_cov90,
    /// cov50). wis and cov50 are shown for rows read from stored scores.
    #[arg(long = "metric", alias = "metrics")]
    pub metrics: Option<String>,

    /// Output format: table (default), md, json
    #[arg(long, default_value = "table")]
    pub format: String,

    /// Score every row on these observation streams only, one table per
    /// stream (comma-separated). A name is a stream (a leaf of the model's
    /// `observations { }` block) or a data source, which selects every leaf
    /// it expands to. Each table scores each row on that one stream's own
    /// one-step-ahead predictive, from the per-stream scores the trace
    /// already stores, so fits that bound different streams can be compared
    /// on a stream they share. No figure combines streams. Also `streams =
    /// [...]` in compare.toml.
    #[arg(long, value_name = "NAME[,NAME...]")]
    pub streams: Option<String>,

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

    /// Render even if the compared traces were scored with different
    /// particle-filter settings (particle count or replicate count), as each
    /// trace records them. The log of a particle-filter likelihood estimate
    /// is biased downward, by more at fewer particles, so such a Δelpd
    /// partly measures the filters; by default it is refused. With this flag
    /// every format states the difference. A trace that records no settings
    /// (older than schema 5) is reported as not checked.
    #[arg(long)]
    pub allow_mismatched_filter_settings: bool,

    /// Under `--streams`, render rows whose fits assimilated different
    /// observation streams although they are scored in-sample. Such a
    /// comparison is the stream-ablation experiment, and in-sample it is
    /// biased toward the fit trained on fewer streams (its parameters were
    /// fitted to the shared stream alone); by default it is refused, and the
    /// remedy is the same `holdout_after` in every compared fit. With this flag
    /// every format states the bias. An explicit prequential.json does not
    /// record what its fit assimilated and counts as different.
    #[arg(long)]
    pub allow_in_sample_ablation: bool,

    /// The posterior draws each sampler's fit run is predicted from: the
    /// `--n-draws` of the `camdl predict` compare runs for it (default:
    /// predict's). Each row's density is averaged over the draws, never its
    /// log, so scores and intervals carry parameter uncertainty (provenance
    /// `posterior`). An optimizer run (IF2, NLopt) is predicted at its
    /// winning point, once, whatever `--draws` is.
    #[arg(long)]
    pub draws: Option<usize>,

    /// Predictions per input at the seeds seed, seed+1, …, seed+R−1, each
    /// scored and stored (default 5): a fit run's from compare's seed, a
    /// stored prediction's or score's from its prediction's seed, posed again
    /// with its other options. Every replicate is scored against the outcome
    /// rows of the first. An ODE free-forward score does not depend on the
    /// seed, and is predicted once. Each row's log score is the
    /// log-mean-exp of its replicates', and the replicate totals give each row
    /// a filter-noise Monte-Carlo SE (`mc_se_elpd`); the evidence verdict is
    /// suppressed ("within filter noise") when |Δelpd| sits inside twice the
    /// pair's combined MC SE. The sample rules (CRPS, WIS, PIT, coverage) are
    /// the first replicate's: a score keeps no predictive samples to pool.
    /// `--replicates 1` is the cheap mode: one prediction, no MC SE, and an
    /// evidence cell that says the filter noise was not measured.
    #[arg(long)]
    pub replicates: Option<usize>,

    /// Render rows whose provenance kinds differ (plug-in vs posterior
    /// mixture) instead of refusing: the Δ then compares an
    /// under-dispersed single-θ predictive against a mixture, and the
    /// reader owns that confound. The usual cause is comparing an
    /// optimizer fit (no posterior cloud) against a Bayesian one.
    #[arg(long)]
    pub allow_mixed_provenance: bool,

    /// Compare the training rows (role `trained`), stamped `in_sample` and
    /// carrying the in-sample optimism caveat, even for fits that held rows
    /// out. Without it, a score's held-out rows are compared when it has
    /// any, and its training rows otherwise.
    #[arg(long)]
    pub in_sample: bool,

    /// The particle count of the one-step and forecast filters of every
    /// prediction compare makes: the `--particles` of the `camdl predict`
    /// compare runs (default 1000). A stored prediction or score read in a
    /// filtered mode must have run its filters at this count when it is given
    /// or compare predicts a fit run, and stored inputs alone at one count;
    /// otherwise they are refused before anything is computed. Not used under --mode
    /// free_forward, which runs no filter.
    #[arg(long)]
    pub particles: Option<usize>,

    /// The seed of each fit run's prediction: the `--seed` of the `camdl
    /// predict` compare runs (default: predict's). Applied uniformly across
    /// the fit runs compared.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Drop MCMC chains from a fit's posterior cloud before predicting from
    /// it: the `--exclude-chains` of the `camdl predict` compare runs. PER-FIT (repeat the flag): `--exclude-chains @a:4` drops
    /// chain 4 from the fit named `@a` only, leaving the others whole. The fit
    /// name is matched VERBATIM against the name shown in the table (and matched
    /// by `--baseline`): a fit given by run-store handle is `@a`, one given by
    /// path is e.g. `ctl_rm.toml` with no `@` — do not add a spurious `@`. Bare
    /// ids `--exclude-chains 3,4` apply COHORT-WIDE to every fit (convenient
    /// only when the fits share a stuck-chain index — otherwise use the per-fit
    /// form). Chain ids count from 0 (matching the `chain` column of
    /// `draws.tsv`, the `chain_N/` dirs and the `fit summary` per-chain table). Mixing bare and `@fit:ids` tokens is rejected;
    /// a fit with no posterior cloud (an optimizer fit, or an explicit
    /// prequential.json) ignores the flag. Post-hoc exclusion BIASES the
    /// posterior toward the retained mode and always prints a warning. A chain
    /// id not in a fit, an unknown/ambiguous fit name, or excluding every chain
    /// is a hard error.
    #[arg(long, value_name = "[@FIT:]IDS")]
    pub exclude_chains: Vec<String>,

    /// Keep ONLY the named MCMC chains of a fit's posterior cloud before
    /// predicting from it: the complement of `--exclude-chains`, with the
    /// same forms. PER-FIT (repeat the flag): `--chains @a:0,1` keeps chains 0
    /// and 1 of the fit named `@a`. The two flags may name different fits in
    /// one invocation (`--chains @a:0,1 --exclude-chains @b:4`); naming one
    /// fit in both is refused, and a bare cohort-wide value in either flag is
    /// refused when either flag also has a per-fit `@fit:ids` value. Always
    /// warns; a chain id not in a fit, or naming every chain, is a hard error.
    #[arg(long, value_name = "[@FIT:]IDS")]
    pub chains: Vec<String>,

    /// Write the per-observation Δelpd vector to PATH as a TSV (gh#706).
    ///
    /// `Δelpd = 12 nats` says a model won; this says WHERE it won — on three
    /// weeks around an intervention, on one district, on a single reporting
    /// batch. PATH gets one row per candidate × scored step with the joint
    /// log score (all streams observed at that time), the baseline's, and
    /// their difference. Each stream's own scores go to a sibling file with
    /// `.streams` before the extension (`scores.tsv` → `scores.streams.tsv`),
    /// one row per candidate × step × stream: a stream's score is not a share
    /// of the joint one, and those rows do not sum to the joint rows. The
    /// quantity is already computed to form `se(Δelpd)`; this stops discarding
    /// it. The natural consumer is a plot faceted by stream.
    ///
    /// The per-step PIT is deliberately NOT published here until gh#629 (tie
    /// bias in the PIT estimator) is fixed.
    #[arg(long, value_name = "PATH")]
    pub pointwise: Option<PathBuf>,

    /// Write the output manifest to PATH once the comparison has succeeded:
    /// a JSON file naming every stored prediction and score compare read or
    /// stored, each result file by logical name with its SHA-256 and size
    /// (run spec §2.10). Refused when the compared results sit in more than
    /// one run store. Any manifest at PATH is removed when the command
    /// starts.
    #[arg(long, value_name = "PATH")]
    pub manifest: Option<PathBuf>,
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

    fn try_parse_profile(args: &[&str]) -> Result<crate::args::ProfileArgs, clap::Error> {
        let mut full: Vec<&str> = vec![
            "camdl", "profile", "model.camdl", "--data", "cases.tsv",
            "--sweep", "R0=lin(0.5,5,5)", "--rw-sd", "auto", "--particles", "100",
        ];
        full.extend(args);
        match Cli::try_parse_from(full)?.command {
            Command::Profile(a) => Ok(a),
            _ => unreachable!("expected profile"),
        }
    }

    /// `profile --cooling c` means what a fit's `cooling = c` means: c is
    /// reached at iteration 50 (pomp's `cooling.fraction.50`), or at
    /// `--cooling-target-iters`, whatever `--iterations` is. It used to be
    /// reached at the last iteration, so a profile cooled on a different
    /// schedule from the fit it profiled whenever `--iterations` ≠ 50.
    #[test]
    fn profile_cooling_is_the_fit_schedule() {
        let a = try_parse_profile(&["--iterations", "10", "--cooling", "0.5"]).unwrap();
        let schedule = crate::profile::profile_cooling_schedule(a.cooling, a.cooling_target_iters);
        assert_eq!(schedule.target_iters, 50, "reached at 50, not at --iterations");
        assert!((schedule.multiplier_at_iter(7, 50) - 0.5).abs() < 1e-12);
        assert!(schedule.multiplier_at_iter(7, 10) > 0.5, "still cooling at the run's end");
        let a = try_parse_profile(&["--iterations", "10", "--cooling-target-iters", "10"]).unwrap();
        assert_eq!(crate::profile::profile_cooling_schedule(a.cooling, a.cooling_target_iters).target_iters, 10);
        // And the IF2Config each cell runs with carries that schedule.
        let a = try_parse_profile(&["--iterations", "10", "--cooling", "0.5"]).unwrap();
        let config = crate::profile::profile_if2_config(
            100, a.iterations.unwrap(), crate::profile::profile_cooling_schedule(a.cooling, a.cooling_target_iters), None,
            Default::default(), 1.0, 0.0, vec![], false);
        assert_eq!(config.n_iterations, 10);
        assert_eq!(config.cooling,
            sim::inference::if2::CoolingSchedule { fraction: 0.5, target_iters: 50 });
    }

    /// An nl-* profile cell runs no particle filter and perturbs nothing, so
    /// the IF2 flags are refused there rather than hashed and ignored; an
    /// `--rw-sd` that only names the estimated set is accepted.
    #[test]
    fn profile_refuses_if2_flags_on_an_nlopt_cell() {
        fn if2_flags(a: &crate::args::ProfileArgs) -> crate::profile::If2Flags<'_> {
            crate::profile::If2Flags {
                particles: a.particles,
                iterations: a.iterations,
                cooling: a.cooling,
                cooling_target_iters: a.cooling_target_iters,
                pf_max_substeps: a.inference.pf_max_substeps,
                binomial: a.inference.binomial,
                rw_sd: a.rw_sd.as_ref(),
            }
        }
        let parse = |extra: &[&str]| {
            let mut full: Vec<&str> = vec![
                "camdl", "profile", "model.camdl", "--data", "cases.tsv",
                "--sweep", "R0=lin(0.5,5,5)", "--algorithm", "nl-sbplx", "--backend", "ode",
            ];
            full.extend(extra);
            match Cli::try_parse_from(full).unwrap().command {
                Command::Profile(a) => a,
                _ => unreachable!(),
            }
        };
        for (extra, flag) in [
            (&["--rw-sd", "auto", "--particles", "100"][..], "--particles"),
            (&["--rw-sd", "auto", "--iterations", "10"][..], "--iterations"),
            (&["--rw-sd", "auto", "--cooling", "0.5"][..], "--cooling"),
            (&["--rw-sd", "auto", "--cooling-target-iters", "9"][..], "--cooling-target-iters"),
            (&["--rw-sd", "auto", "--pf-max-substeps", "1000"][..], "--pf-max-substeps"),
            (&["--rw-sd", "beta=0.1"][..], "beta"),
        ] {
            let err = crate::profile::refuse_if2_flags_for_nlopt(&if2_flags(&parse(extra)), "nl-sbplx")
                .expect_err(&format!("{flag} is refused on an nl-sbplx cell"));
            assert!(err.contains(flag) && err.contains("nl-sbplx"), "{err}");
        }
        for extra in [&["--rw-sd", "auto"][..], &["--rw-sd", "beta=auto,gamma=auto"][..]] {
            crate::profile::refuse_if2_flags_for_nlopt(&if2_flags(&parse(extra)), "nl-sbplx")
                .unwrap_or_else(|e| panic!("{extra:?} names the estimated set: {e}"));
        }
    }

    /// gh#889: `camdl profile --starts` is the same rule grammar
    /// `camdl fit run --starts` takes — a name, or a name and a source — and
    /// the count of independent starts per grid point is `--n-starts`. One
    /// word meant a rule on one verb and a number on its sibling.
    #[test]
    fn profile_starts_is_a_rule_and_the_count_is_n_starts() {
        use crate::fit::starts::{ChainStarts, Point, Spread};

        let a = try_parse_profile(&["--starts", "from_params=truth.toml", "--n-starts", "4"])
            .expect("profile must take the sourced rule grammar");
        match &a.starts {
            Some(ChainStarts::Point(Point::FromParams { path })) =>
                assert_eq!(path.to_string_lossy(), "truth.toml"),
            other => panic!("expected the from_params rule, got {other:?}"),
        }
        assert_eq!(a.n_starts, 4, "the count moved to --n-starts");

        let a = try_parse_profile(&["--starts", "lhs"]).expect("a bare rule name");
        assert_eq!(a.starts, Some(ChainStarts::Spread(Spread::Lhs)));
        assert_eq!(a.n_starts, 3, "the count keeps its default");

        // No flag leaves the rule to `fit run`'s default, decided per profile
        // over the estimated set (gh#1059).
        let a = try_parse_profile(&[]).expect("no flags");
        assert_eq!(a.starts, None);
    }

    /// The old `--starts <N>` is now a parse error, and it must name the flag
    /// that carries the count rather than listing rules at someone who wrote
    /// a number.
    #[test]
    fn profile_starts_with_a_count_names_n_starts() {
        let e = match try_parse_profile(&["--starts", "4"]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a number is not a start rule"),
        };
        assert!(e.contains("--n-starts 4"), "the message names the count flag: {e}");
        assert!(e.contains("not how many there are"),
            "and says what `starts` means: {e}");
    }

    /// The four flags gh#889 removed from `profile` still parse, into their
    /// hidden traps, so the dispatch site answers with the `--starts`
    /// replacement instead of clap's "unexpected argument".
    #[test]
    fn profile_removed_init_family_is_trapped_at_parse() {
        let a = try_parse_profile(&[
            "--init", "from_params", "--params", "truth.toml",
            "--posterior", "fits/a", "--mle", "fits/b",
        ]).expect("the hidden traps must accept the removed flags");
        assert_eq!(a._removed_init.as_deref(), Some("from_params"));
        assert_eq!(a._removed_params.as_deref(), Some("truth.toml"));
        assert_eq!(a._removed_posterior.as_deref(), Some("fits/a"));
        assert_eq!(a._removed_mle.as_deref(), Some("fits/b"));
        assert_eq!(a.model_overrides.fixed_cli.len(), 0,
            "--params must not pollute fixed_cli");

        // And each is answered with the replacement `fit run` gives, from the
        // one builder both verbs use.
        let msg = crate::fit::removed_flags_message(
            "profile",
            crate::fit::starts_family_removed_lines(
                a._removed_init.as_deref(), a._removed_posterior.as_deref(),
                a._removed_mle.as_deref(), a._removed_params.as_deref(),
            ),
        ).expect("a message for every removed flag passed");
        assert!(msg.contains("`camdl profile` flags were removed"), "{msg}");
        assert!(msg.contains("--starts from_params=truth.toml"), "{msg}");
        assert!(msg.contains("--starts from_posterior=fits/a"), "{msg}");
        assert!(msg.contains("--starts from_mle=fits/b"), "{msg}");
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

    /// Every flag the `[stages]` → `[method]` split removed from `fit run`
    /// still parses, into its hidden trap, so the dispatch site can answer
    /// with the replacement instead of clap's "unexpected argument".
    #[test]
    fn fit_run_removed_flags_are_trapped_at_parse() {
        let a = try_parse_fit_run(&[
            "fit.toml", "--stage", "posterior", "--init", "from_prior",
        ]).expect("hidden traps must accept --stage and --init");
        assert_eq!(a._removed_stage.as_deref(), Some("posterior"));
        assert_eq!(a._removed_init.as_deref(), Some("from_prior"));
        assert!(a.run.starts.is_none(), "a trapped --init must not populate --starts");

        let a = try_parse_fit_run(&[
            "fit.toml", "--posterior", "fits/a", "--mle", "fits/b",
            "--params", "theta.toml", "--survey-path", "surveys/x",
            "--survey-top-k", "4", "--allow-nonconverged-scout",
            "--init-method", "lhs", "--starts-from", "fits/scout/",
        ]).expect("every removed flag must parse into its trap");
        assert_eq!(a._removed_posterior.as_deref(), Some("fits/a"));
        assert_eq!(a._removed_mle.as_deref(), Some("fits/b"));
        assert_eq!(a._removed_params.as_deref(), Some("theta.toml"));
        assert_eq!(a._removed_survey_path.as_deref(), Some("surveys/x"));
        assert_eq!(a._removed_survey_top_k.as_deref(), Some("4"));
        assert!(a._removed_allow_nonconverged_scout);
        assert_eq!(a._removed_init_method.as_deref(), Some("lhs"));
        assert_eq!(a._removed_starts_from.as_deref(), Some("fits/scout/"));
    }

    /// `--starts <spec>` parses every rule: the bare names and the sourced
    /// `name=source` form.
    #[test]
    fn fit_run_starts_flag_parses_every_rule() {
        use crate::fit::starts::{ChainStarts, Point, Spread};
        let cases: &[(&str, ChainStarts)] = &[
            ("uniform_unconstrained", ChainStarts::Spread(Spread::UniformUnconstrained)),
            ("lhs", ChainStarts::Spread(Spread::Lhs)),
            ("uniform", ChainStarts::Spread(Spread::Uniform)),
            ("from_prior", ChainStarts::Spread(Spread::FromPrior)),
            ("single", ChainStarts::Point(Point::Declared)),
            ("from_posterior=@base", ChainStarts::Spread(Spread::FromPosterior {
                source: crate::fit::starts::Handle("@base".into()), chains: None,
            })),
            ("from_mle=@mle", ChainStarts::Point(Point::FromMle {
                source: crate::fit::starts::Handle("@mle".into()),
            })),
            ("from_params=theta.toml", ChainStarts::Point(Point::FromParams {
                path: "theta.toml".into(),
            })),
        ];
        for (spec, want) in cases {
            let a = try_parse_fit_run(&["fit.toml", "--starts", spec])
                .unwrap_or_else(|e| panic!("--starts {spec} must parse: {e}"));
            assert_eq!(a.run.starts.as_ref(), Some(want), "--starts {spec}");
        }
        // A sourced rule without its source, and an unknown name, are refused
        // at parse time with the rule named.
        let err = try_parse_fit_run(&["fit.toml", "--starts", "from_mle"])
            .err().expect("from_mle needs a source");
        assert!(err.to_string().contains("from_mle"), "{err}");
        let err = try_parse_fit_run(&["fit.toml", "--starts", "survey_top_k"])
            .err().expect("survey_top_k is not a rule");
        assert!(err.to_string().contains("survey_top_k"), "{err}");
    }

    // gh#189: --loglik-eval-particles/-reps removed — loglik_eval is part of the
    // fit identity (set only in the method table), not a CLI override that
    // bypasses the run_id. The convergence-gate override (--decibans-thresh)
    // stays, and no longer needs a method to attach to.
    #[test]
    fn fit_run_gate_override_parses() {
        let a = try_parse_fit_run(&[
            "fit.toml", "--decibans-thresh", "60.0",
        ]).expect("should parse");
        assert_eq!(a.method.decibans_thresh, Some(60.0));
    }

    #[test]
    fn fit_run_gate_override_default_is_none() {
        let a = try_parse_fit_run(&["fit.toml"]).unwrap();
        assert!(a.method.decibans_thresh.is_none());
    }

    #[test]
    fn cli_command_factory_builds() {
        // Smoke test — guards against malformed clap derives that would
        // panic at runtime instead of producing a parse error.
        let _ = Cli::command();
    }

    /// Every command's `--dt` help states the step it defaults to (the
    /// model's `simulate { dt }`) and points at the rule a fitting or scoring
    /// command applies to a step that differs, run spec §1.3.1.
    #[test]
    fn every_dt_help_names_the_models_step_and_the_rule() {
        let mut cmd = Cli::command();
        for path in [&["simulate"][..], &["profile"], &["pfilter"], &["predict"]] {
            let mut sub = &mut cmd;
            for name in path {
                sub = sub.find_subcommand_mut(name).unwrap_or_else(|| panic!("no subcommand {name}"));
            }
            let dt = sub.get_arguments().find(|a| a.get_id() == "dt").unwrap_or_else(|| panic!("{path:?} has --dt"));
            let help = dt.get_long_help().or(dt.get_help()).map(|h| h.to_string()).unwrap_or_default();
            for needle in ["simulate { dt }", "1.0", "§1.3.1"] {
                assert!(help.contains(needle), "{path:?} --dt help lacks {needle:?}: {help:?}");
            }
        }
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

    /// The commands that write results and the commands that read them name
    /// no store root by default, so all of them open the one root
    /// `camdl_store::root_of` resolves: `CAMDL_STORE`, else `./camdl-store`.
    #[test]
    fn readers_and_writers_default_to_one_store_root() {
        let parsed = |args: &[&str]| Cli::try_parse_from(args).unwrap().command;
        match parsed(&["camdl", "simulate", "model.camdl"]) {
            Command::Simulate(a) => assert!(a.store.is_none()),
            _ => unreachable!(),
        }
        match parsed(&["camdl", "list"]) {
            Command::List(a) => assert!(a.store.is_none()),
            _ => unreachable!(),
        }
        match parsed(&["camdl", "show", "abc12345"]) {
            Command::Show(a) => assert!(a.store.is_none()),
            _ => unreachable!(),
        }
        match parsed(&["camdl", "cat", "abc12345"]) {
            Command::Cat(a) => assert!(a.store.is_none()),
            _ => unreachable!(),
        }
    }

    // ── predict: scenario overlay parsing (Layer 1) ──────────────────────────

    fn parse_predict(args: &[&str]) -> PredictFlags {
        let mut full: Vec<&str> = vec!["camdl", "predict", "@fit"];
        full.extend(args);
        let cli = Cli::try_parse_from(full).expect("predict parse");
        match cli.command {
            Command::Predict(a) => a.flags,
            _ => unreachable!("expected predict"),
        }
    }

    #[test]
    fn predict_no_scenario_yields_single_fitted() {
        // No `--scenario` → a single inline `fitted` overlay carrying no
        // enable/disable (the no-overlay row, byte-identical to today's
        // hardcoded baseline replay).
        let a = parse_predict(&[]);
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
    fn predict_repeated_scenario_parses_to_named_vec() {
        // Repeated `--scenario` → one `Named` ref each, in order (mirrors the
        // simulate parser). This is the proposal's `no_sia` / `with_sia` form.
        let a = parse_predict(&[
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
    fn predict_comma_list_scenario_splits() {
        // `--scenario a,b` comma-splits, exactly like simulate.
        let a = parse_predict(&["--scenario", "no_sia,with_sia"]);
        let refs = a.scenario_refs().unwrap();
        let names: Vec<String> = refs.iter().map(|r| r.name().to_string()).collect();
        // gh#625: fitted leads (see the repeated-`--scenario` test).
        assert_eq!(names, vec!["fitted".to_string(), "no_sia".to_string(),
                               "with_sia".to_string()]);
    }

    #[test]
    fn predict_enable_disable_form_overlay() {
        // `--enable`/`--disable` (no `--scenario`) → a single ad-hoc `fitted`
        // overlay carrying the toggles, mirroring `simulate --enable`.
        let a = parse_predict(&["--enable", "sia", "--disable", "ri"]);
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
    fn predict_scenario_fitted_is_reserved() {
        // An explicit `--scenario fitted` is rejected: it collides with the
        // reserved no-overlay value. The diagnostic names the reservation + fix.
        let a = parse_predict(&["--scenario", "fitted"]);
        let err = a.scenario_refs().expect_err("fitted must be rejected");
        assert!(err.contains("reserved"), "names the reservation: {err}");
        assert!(err.contains("fitted"), "names the offending value: {err}");
    }

    #[test]
    fn predict_scenario_conflicts_with_enable() {
        // clap-level: --scenario and --enable are mutually exclusive (mirrors
        // simulate's σ-flag rule).
        let full = ["camdl", "predict", "@fit",
                    "--scenario", "no_sia", "--enable", "sia"];
        assert!(Cli::try_parse_from(full).is_err(),
            "--scenario + --enable must be a clap conflict");
    }

    #[test]
    fn simulate_to_help_states_the_runtime_month_and_year_lengths() {
        // gh#1073: `--to "last_obs + N months"` converts through
        // `ir::caltime::days_per_unit` (`parse_time_spec`), so the help must
        // state those lengths, not a Julian-style 30.4375-day month.
        let mut cmd = Cli::command();
        let help = cmd
            .find_subcommand_mut("simulate")
            .expect("simulate subcommand")
            .render_long_help()
            .to_string();
        let month = ir::caltime::days_per_unit("months").unwrap();
        let year = ir::caltime::days_per_unit("years").unwrap();
        for stated in [format!("{month} d"), format!("{year} d")] {
            assert!(help.contains(&stated),
                "simulate --help must state the runtime span {stated:?}");
        }
        assert!(!help.contains("30.4375"),
            "simulate --help still states a 30.4375-day month");
        // Decision F (proposal 2026-11-16-calendar-durations-by-position.md):
        // the fixed spans hold only without an origin.
        let flat = help.split_whitespace().collect::<Vec<_>>().join(" ");
        for stated in ["Without a model origin, months/years are fixed spans",
                       "under an origin a month or year offset from an anchor is refused"] {
            assert!(flat.contains(stated), "simulate --help must say {stated:?}:\n{flat}");
        }
    }
}
