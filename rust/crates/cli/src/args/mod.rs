// ── Args structs only ─────────────────────────────────────────────────────────
//
// The clap subcommand tree (Cli, Command, FitCmd, BatchCmd, DataCmd) lives
// in main.rs — it's the canonical, dispatched parser. This module owns
// only the per-command argument structs (FitRunArgs, SimulateArgs, etc.)
// referenced by main.rs's tree.

pub mod types;

use std::path::PathBuf;
use clap::Args;
use crate::colored_help;
use types::{Backend, ListDuration, ParamOverride, ParamVecSpec, RwSd, SeedSpec, SweepSpec, TableSpec};

// ─── Shared flat arg groups ───────────────────────────────────────────────────

/// `--params FILE` (repeatable) + `--param NAME=VALUE` (repeatable) +
/// `--table NAME=FILE` (repeatable)
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
    /// Simulation backend (default: gillespie)
    #[arg(long)]
    pub backend: Option<Backend>,

    /// Step size for discrete-time backends (default: 1.0)
    #[arg(long)]
    pub dt: Option<f64>,
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
}

/// `--obs NAME` + `--flow NAME`
#[derive(Args, Clone, Default)]
pub struct FlowProjection {
    /// Observation block name (required when model has more than one)
    #[arg(long)]
    pub obs: Option<String>,

    /// Flow name for incidence projection (overrides obs model default)
    #[arg(long)]
    pub flow: Option<String>,
}

// ─── simulate ─────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Basic simulation, output to stdout
  camdl simulate sir.camdl --params p.toml --seed 42

  # Named scenario
  camdl simulate sir.camdl --params p.toml --scenario with_sia --seed 42

  # Generate synthetic observations alongside the trajectory
  camdl simulate sir.camdl --params p.toml --obs cases.tsv --seed 42

  # Cache output under ./results (reruns with same inputs are instant)
  camdl simulate sir.camdl --params p.toml --seed 42 --cas
  camdl list        # browse cached runs

  # Multi-seed ensemble
  camdl simulate sir.camdl --params p.toml --seeds 1:100

  # Posterior predictive check from a fit's draws
  camdl simulate sir.camdl --draws posterior.tsv --replicates 10 --obs ppc.tsv
"))]
pub struct SimulateArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

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

    /// Parameter draw source: path to params TSV, "uniform", or "prior"
    #[arg(long)]
    pub draws: Option<String>,

    /// fit.toml supplying priors for --draws prior
    #[arg(long, requires = "draws")]
    pub fit: Option<PathBuf>,

    /// Number of parameter draws (for --draws uniform/prior)
    #[arg(short = 'n', long)]
    pub n_draws: Option<usize>,

    /// Trajectory output file (default: stdout)
    #[arg(short, long, env = "CAMDL_OUTPUT")]
    pub output: Option<PathBuf>,

    /// Write synthetic observations to a single TSV (all streams)
    #[arg(long, conflicts_with_all = ["obs_dir", "obs_only"])]
    pub obs: Option<PathBuf>,

    /// Write one TSV per observation stream to a directory
    #[arg(long, conflicts_with_all = ["obs", "obs_only"])]
    pub obs_dir: Option<PathBuf>,

    /// Like --obs but suppress trajectory output
    #[arg(long, conflicts_with_all = ["obs", "obs_dir", "output"])]
    pub obs_only: Option<PathBuf>,

    /// Print resolved run plan without simulating
    #[arg(long)]
    pub dry_run: bool,

    /// Write output to content-addressable cache
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

Notes:
  - PGAS/PMMH fits can resume a partial run with `--resume`.
  - Output goes under `<root>/fits/<stem>-<hash>/`; resolve with
    `camdl fit where fit.toml`.
"))]
pub struct FitRunArgs {
    /// Fit configuration file (v2 TOML)
    pub config: PathBuf,

    /// Run only this stage by name
    #[arg(long)]
    pub stage: Option<String>,

    /// RNG seed (default: 1)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Re-run and overwrite stale cache
    #[arg(long)]
    pub force: bool,

    /// Resume a previously-completed PGAS or PMMH stage from its
    /// stored `chain_<n>/resume_state.bin`. Requires --stage.
    /// Identity fields (chains, particles, burn_in, thin, model,
    /// data, priors, fixed, seed) must match the original run; the
    /// extension dimension (PGAS `sweeps` / PMMH `iterations`) may
    /// change to extend the chain. Conflicts with --force.
    #[arg(long, requires = "stage", conflicts_with = "force")]
    pub resume: bool,

    /// Starting-point directory or short run hash (requires --stage)
    #[arg(long, requires = "stage")]
    pub starts_from: Option<String>,

    /// Cartesian sweep over a fixed parameter (may repeat).
    /// SPEC is `V1,V2,...` | `lin(min,max,n)` | `log10(min,max,n)`.
    #[arg(long, value_name = "NAME=SPEC")]
    pub sweep: Vec<SweepSpec>,

    /// Proceed even if prior scout stage failed convergence gate
    #[arg(long)]
    pub allow_nonconverged_scout: bool,

    /// Override [stages.<stage>.loglik_eval] n_particles. Requires --stage
    /// so scout and refine loglik-eval settings can be overridden independently.
    #[arg(long, value_name = "N", requires = "stage")]
    pub loglik_eval_particles: Option<usize>,

    /// Override [stages.<stage>.loglik_eval] n_replicates. Requires --stage.
    #[arg(long, value_name = "M", requires = "stage")]
    pub loglik_eval_reps: Option<usize>,

    /// Override [stages.<stage>.gate] decibans_thresh (the inter-chain
    /// log-likelihood-spread floor, in decibans). Requires --stage.
    #[arg(long, value_name = "DB", requires = "stage")]
    pub decibans_thresh: Option<f64>,

    /// Override [stages.<stage>.init_method] for chain starts (gh#42):
    /// `single` (all chains at the seeded start), `uniform` (per-chain
    /// uniform random within bounds — v1 default), or `lhs`
    /// (Latin-hypercube stratified, scale-aware via Transform — best
    /// basin coverage at low chain counts). Requires --stage so scout
    /// and refine can be set independently. Has no effect when the
    /// stage uses `starts_from = "<prior_stage>"` — those chains start
    /// from the prior MLE regardless.
    #[arg(long, value_name = "MODE", requires = "stage")]
    pub init_method: Option<crate::fit::init::InitMethod>,

    /// User-supplied display label for this fit (1–64 chars after
    /// trim; allowed: letters, digits, spaces, commas, dot,
    /// underscore, hyphen). Surfaced in `camdl fit list` and
    /// `camdl fit table` to disambiguate iterations of a model that
    /// share the same fit-stem. Examples: --label "narrow R0, take 1",
    /// --label "iota free", --label "log_normal R0 prior".
    #[arg(long, value_name = "TEXT")]
    pub label: Option<String>,

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

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Completion / convergence summary for a fit config
  camdl fit status fit.toml

  # Same, by results-directory path
  camdl fit status results/fits/he2010-abc123/
"))]
pub struct FitStatusArgs {
    /// Fit config file or results directory
    pub path: Option<PathBuf>,
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
    /// Fit results directory (e.g. `fit/he2010`)
    pub fit_dir: PathBuf,

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

    /// Filter to fits with this model_hash (prefix match).
    #[arg(long, value_name = "HASH_PREFIX")]
    pub model: Option<String>,

    /// Filter to fits whose `fit_hash` (Run.hash) starts with the
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

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Print the content-addressed output directory for a fit.toml
  camdl fit where fit.toml

  # Per-seed cell directory (multi-seed fits)
  camdl fit where fit.toml --seed 42
"))]
pub struct FitWhereArgs {
    /// Fit config file
    pub config: PathBuf,

    /// Print per-seed cell directory instead of fit root
    #[arg(long)]
    pub seed: Option<u64>,
}

// ─── label ────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # Label any run kind by its short hash
  camdl label 04ab12cd \"narrow R0, take 1\"
  camdl label 9c5d11f0 \"baseline sim, daily reporting\"
  camdl label 7e2a5d4b \"R0 vs gamma profile, take 2\"

Notes:
  - Labels are 1–64 characters after trim, restricted to:
    letters, digits, spaces, commas, dot, underscore, hyphen.
  - The hash is matched as a prefix (8+ chars recommended) across
    every kind under <root>/ (sims, fits, profiles, replicate-sets).
  - Errors on ambiguous or unmatched prefix.
  - Errors on a still-running fit (RunStatus::Running); the
    runner would otherwise overwrite the label at completion.
  - Concurrent invocations are last-write-wins.
"))]
pub struct LabelArgs {
    /// Hash prefix of the target run (matches against
    /// `<root>/{sims,fits,profiles}/**/run.json`'s `Run.hash`)
    pub hash: String,

    /// New label text. Validated against ^[a-zA-Z0-9 ,._-]{1,64}$
    /// after trim. Empty / whitespace-only labels are rejected.
    pub label: String,

    /// Output root to search under (default: results/)
    #[arg(long)]
    pub root: Option<PathBuf>,
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
    pub flow: FlowProjection,

    /// Observation data TSV (with time column)
    #[arg(long)]
    pub data: PathBuf,

    /// Number of independent filter runs
    #[arg(long, default_value_t = 1)]
    pub replicates: usize,

    /// Write per-observation diagnostics TSV; use "-" for stdout
    #[arg(long)]
    pub trace: Option<String>,

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

    /// With --save-prequential, drop per-particle predictive samples
    /// from {STEM}.json. Keeps scalar scores, shrinks the file.
    #[arg(long)]
    pub no_save_samples: bool,
}

// ─── if2 ──────────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # IF2 from scratch with explicit rw-sd map
  camdl if2 sir.camdl --data cases.tsv \\
      --rw-sd \"R0=5,sigma=0.01\" --particles 2000 --iterations 100

  # Use a regime preset (scout / refine / validate)
  camdl if2 sir.camdl --data cases.tsv --regime scout --rw-sd auto

  # Multiple chains in parallel
  camdl if2 sir.camdl --data cases.tsv --rw-sd auto \\
      --regime refine --chains 4 --parallel 4
"))]
pub struct If2Args {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    #[command(flatten)]
    pub model_overrides: ModelOverrides,

    #[command(flatten)]
    pub scenario: ScenarioArgs,

    #[command(flatten)]
    pub inference: InferenceCore,

    #[command(flatten)]
    pub flow: FlowProjection,

    /// Observation data TSV
    #[arg(long)]
    pub data: PathBuf,

    /// IF2 iterations
    #[arg(long, conflicts_with = "regime")]
    pub iterations: Option<usize>,

    /// Number of chains
    #[arg(long, conflicts_with = "regime")]
    pub chains: Option<usize>,

    /// Cooling schedule factor (0–1)
    #[arg(long, conflicts_with = "regime")]
    pub cooling: Option<f64>,

    /// Preset configuration: scout, refine, or validate
    #[arg(long, conflicts_with_all = ["chains", "iterations", "cooling"])]
    pub regime: Option<String>,

    /// Random-walk standard deviations, e.g. "beta=0.05,rho=0.01" or "auto"
    #[arg(long)]
    pub rw_sd: Option<RwSd>,

    /// Parameters to hold fixed during estimation (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub fixed: Vec<String>,

    /// Initial-value-problem parameters (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub ivp: Vec<String>,

    /// Write IF2 iteration-by-iteration diagnostics TSV
    #[arg(long)]
    pub trace: Option<PathBuf>,

    /// Output file (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Write per-chain traces and summary to this directory
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
}

// ─── profile ──────────────────────────────────────────────────────────────────

#[derive(Args)]
#[command(after_help = colored_help!("\
Examples:
  # 1D profile likelihood for R0 via parallel IF2
  camdl profile sir.camdl --data cases.tsv \\
      --param R0 --grid 0.5:5:20 --particles 2000

  # 2D profile (R0 × sigma)
  camdl profile sir.camdl --data cases.tsv \\
      --param R0 --grid 0.5:5:10 \\
      --param sigma --grid 0.1:1.0:10
"))]
pub struct ProfileArgs {
    /// IR JSON or .camdl model file
    pub model: PathBuf,

    #[command(flatten)]
    pub model_overrides: ModelOverrides,

    #[command(flatten)]
    pub scenario: ScenarioArgs,

    #[command(flatten)]
    pub inference: InferenceCore,

    #[command(flatten)]
    pub flow: FlowProjection,

    /// Observation data TSV
    #[arg(long)]
    pub data: PathBuf,

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
    /// across the non-focal estimated parameters' bounds:
    /// `single` (every start at the seeded base — chains differ only
    /// by IF2 RNG), `uniform` (per-start uniform random within bounds —
    /// historical default), or `lhs` (Latin-hypercube stratified,
    /// scale-aware via Transform — best basin coverage at low
    /// `--starts`). gh#42.
    #[arg(long, value_name = "MODE", default_value_t = crate::fit::init::InitMethod::Uniform)]
    pub init: crate::fit::init::InitMethod,

    /// Cooling schedule
    #[arg(long, default_value_t = 0.95)]
    pub cooling: f64,

    /// Random-walk SDs
    #[arg(long)]
    pub rw_sd: Option<RwSd>,

    /// Parameters to hold fixed (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub fixed: Vec<String>,

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

    /// Number of Latin-hypercube points to evaluate. Defaults to 1000;
    /// proposal §"Defaults" balances coverage vs cost for d <= 8.
    #[arg(long, default_value_t = 1000)]
    pub n_points: usize,

    /// Likelihood evaluation method. Default `pfilter` handles
    /// process noise via PMMH-style MC estimator; `simulate` is a
    /// faster fast-path for known-deterministic models (warns at
    /// start; not safe for high-noise models).
    #[arg(long, default_value_t = crate::run_meta::SurveyEvalMethod::Pfilter)]
    pub eval: crate::run_meta::SurveyEvalMethod,

    /// Particle count for `--eval pfilter`. 200 is adequate for
    /// `sigma_se <= 1` on weekly data per the proposal's per-point
    /// cost table.
    #[arg(long, default_value_t = 200)]
    pub eval_particles: usize,

    /// PF replicates per LHS point (logmeanexp combiner). 3 reps
    /// cuts SE by ~sqrt(3) vs 1 rep. Always 1 with `--eval simulate`.
    #[arg(long, default_value_t = 3)]
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
  camdl eval sir.camdl --params p.toml \\
      --expr \"beta,gamma\" --from 0 --to 730 --every 1

  # Inspect a forcing function over time
  camdl eval sir.camdl --params p.toml \\
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
    /// Root directory to scan (default: ./results)
    #[arg(default_value = "./results", env = "CAMDL_OUTPUT_DIR")]
    pub root: PathBuf,

    /// Filter by model path substring
    #[arg(long)]
    pub model: Option<String>,

    /// Filter by scenario name
    #[arg(long)]
    pub scenario: Option<String>,

    /// Show only runs created within this duration (e.g. 1h, 30m, 2d)
    #[arg(long)]
    pub since: Option<ListDuration>,

    /// Filter by run kind: sim, fit, profile, or all (default).
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

// ─── compare ──────────────────────────────────────────────────────────────────

/// `camdl compare` — multi-model prequential comparison table.
///
/// Reads prequential.json from ≥2 fit stage dirs (or a compare.toml)
/// and renders a baseline-centered comparison.
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

  # Render despite different T_score across fits (Δ columns → '—')
  camdl compare fits/a/pf fits/b/pf --allow-mismatched-horizon
"))]
pub struct CompareArgs {
    /// Stage directories (or .json paths) to compare — need ≥2 when
    /// --config is not used
    pub paths: Vec<String>,

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

    #[test]
    fn fit_run_loglik_eval_overrides_parse_with_stage() {
        let a = try_parse_fit_run(&[
            "fit.toml",
            "--stage", "scout",
            "--loglik-eval-particles", "8000",
            "--loglik-eval-reps", "16",
            "--decibans-thresh", "60.0",
        ]).expect("should parse with --stage");
        assert_eq!(a.loglik_eval_particles, Some(8000));
        assert_eq!(a.loglik_eval_reps, Some(16));
        assert_eq!(a.decibans_thresh, Some(60.0));
        assert_eq!(a.stage.as_deref(), Some("scout"));
    }

    #[test]
    fn fit_run_loglik_eval_particles_requires_stage() {
        let err = try_parse_fit_run(&[
            "fit.toml", "--loglik-eval-particles", "8000",
        ]).err().expect("should reject without --stage");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn fit_run_decibans_thresh_requires_stage() {
        let err = try_parse_fit_run(&[
            "fit.toml", "--decibans-thresh", "60.0",
        ]).err().expect("should reject without --stage");
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn fit_run_loglik_eval_defaults_are_none() {
        let a = try_parse_fit_run(&["fit.toml"]).unwrap();
        assert!(a.loglik_eval_particles.is_none());
        assert!(a.loglik_eval_reps.is_none());
        assert!(a.decibans_thresh.is_none());
    }

    #[test]
    fn cli_command_factory_builds() {
        // Smoke test — guards against malformed clap derives that would
        // panic at runtime instead of producing a parse error.
        let _ = Cli::command();
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
}
