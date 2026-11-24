mod args;
#[cfg(test)]
mod test_support;   // collision-free unique_temp_dir for unit tests (gh#153)
#[cfg(test)]
mod engine_fingerprint; // ENGINE_ID's fingerprint, also compiled into build.rs; tested here
mod util;
mod params_resolver;  // unified parameter-value resolver (2026-05-25 CLI UX rev 2)
mod external_inputs;  // binds external("NAME") tables and forcings to files
mod hashing;
mod canonical_json; // canonical JSON text and the finiteness gate for values an identity holds as text
mod resolve;        // the model identity string a fit's provenance records
mod run_meta;       // cross-cutting run-metadata value types (FitAlgorithm, Backend, provenance records, FitSidecar)
mod posterior_draws; // resolve a fit run's canonical posterior draws (--draws posterior, predict)
mod chain_index;    // the chain number, counted from 0: the one index base every fit surface names (gh#990)
mod chain_selection; // read-side --exclude-chains: the one chain filter over a posterior cloud
mod quantile; // shared quantile reduction + numeric formatting (proposal 2026-08-11 §3.6)
mod quantities_file; // `--quantities FILE`: a separable reporting vocabulary + its artifact key
mod quantity_output; // generated-quantities banding + tidy-TSV rendering (shared by predict + simulate)
mod obs_anchor;     // gh#616: runtime resolution of a model's observation anchors
mod forcing_extrapolation; // a forcing evaluated outside its range: refuse, or hold under the override
mod obs_emit;       // gh#833: what `simulate --obs` writes per stream, and over which period
mod emit_every;     // gh#656: `--emit-every`, the per-stream emission-cadence override
mod run_paths;      // canonical output-path helpers
mod cas;
mod browse;
mod check_update;   // `camdl check-update` — GitHub release-availability check
mod sim_job;       // SimulateJob / ParamSource / Seeds / ScenarioRef (run-spec §3)
mod roles;         // parameter roles: the partition, resolved from role declarations before any value
mod sweep;         // the design axis: --sweep / --sweep-file expansion, checks, the resolved design
mod engine;        // run_job: the engine behind simulate's cell grid (run-spec §3.1)
mod cas_sink;      // the digest of a simulate cell's observation blocks, in each obs by-product's identity
mod sim_config;    // ResolvedSimConfig: simulate's one resolved configuration; identity and engine settings derive from it
mod eval;
mod dt_rule;      // the one integration-step rule: the model's declared dt, else the written one
mod pfilter;        // used internally by fit runner for data loading
mod prequential_out; // the one prequential.tsv/.json writer (gh#650)
mod pfilter_store;  // camdl pfilter on the run store: its identity, result and by-products
mod fit_store;      // a fit on the run store: the problem, its runs and their predictions
mod fit_command;    // camdl fit run on the run store: the problems, the runs on each, the method under each claim
mod predict_command; // camdl predict on the run store: a prediction from a stored fit run's posterior
mod score_command;   // camdl score on the run store: a stored prediction scored against its outcomes
mod store_commands; // `camdl status`, `verify`, `clean`, `reindex`: the run store's maintenance commands
mod camdl_store;    // camdl on the run store: ambient identity, kinds, root, content artifacts
mod result_files;   // the files each kind of result stores, as types, and the set each run declares
mod input_fixtures; // the second values and fixtures of the input-class checks
mod pfilter_config; // ResolvedPfilterConfig: pfilter's one resolved configuration; identity and filter derive from it
mod caltime_load;   // dated-data loader: column detection + date→internal-time (2026-05-22)
mod design;         // `camdl design lhs`: a Latin-hypercube design as a `--sweep-file` TSV
mod docs;           // `camdl docs <topic>` — embedded, version-locked usage docs
mod fit;
mod compare;
mod model_diff;     // `camdl diff A B`: gathers two models' inputs for the `diff` crate and prints its report
mod if2;
mod profile;
mod profile_config;  // ResolvedProfileConfig: profile's one resolved configuration; identity and cells derive from it
mod profile_diagnostics;
mod profile_store;   // camdl profile on the run store: a point's identity, its result, the table rendered from them
mod profile_table;    // the rows of the per-cell table a profile prints
mod out;           // the stream layer: notices, progress, milestones, colour
mod sim_store;       // camdl simulate on the run store: its cells, their by-products and the combined tables
mod sim_command;     // the store half of camdl simulate: identities, the batch, the engine under each claim, the exports
mod lineage;        // three-layer lineage: --event-log record + realize + tree
mod mre;            // `camdl mre` — minimal-reproducible-example bundles (gh#212)
mod input_class;    // every input of a converted command and its class, checked against clap
mod env;            // part E: every environment variable camdl reads, read once, classified
pub mod version;

/// Terminal formatting helpers. Pure ANSI SGR codes, no dependencies.
// Terminal styling lives in `crate::style`; the `colored_help!` macro
// is exported at the crate root via `#[macro_export]` and used from
// `crate::args` to colorize subcommand `after_help` blocks.
pub mod style;

use clap::{Parser, Subcommand};
use clap::builder::styling::{AnsiColor, Effects, Styles};

/// Color scheme for clap's own help rendering (section headings, flag
/// names, usage). Respects `NO_COLOR` and TTY detection automatically
/// via clap's `ColorChoice::Auto`. After-help blocks are styled
/// separately via `colored_help!` (see `crate::style`).
const HELP_STYLES: Styles = Styles::styled()
    .header   (AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .usage    (AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .literal  (AnsiColor::Cyan  .on_default())
    .placeholder(AnsiColor::Cyan.on_default());
use sim::{write_diagnostics_tsv, never_fired};
use std::collections::HashMap;

// ─── CLI ──────────────────────────────────────────────────────────────────────
//
// Compile/Check/Inspect delegate to camdlc via Passthrough (raw argv forwarding).
// All other commands use fully typed Args structs from args/mod.rs.

#[derive(Parser)]
#[command(
    name = "camdl",
    version = version::VERSION,
    about = "Stochastic compartmental model simulation and inference",
    disable_help_subcommand = true,
    arg_required_else_help = true,
    max_term_width = 100,
    styles = HELP_STYLES,
    after_help = colored_help!("\
New here? Run `camdl docs` for guides — modeling, the DSL, inference, the fit workflow.

Common workflows:
  Simulate a model:        camdl simulate model.camdl --params p.toml
  Fit to data:             camdl fit run fit.toml
  Likelihood at θ:         camdl pfilter model.camdl --params p.toml --data cases.tsv
  Browse cached runs:      camdl list
  Diagnose a fit:          camdl fit summary <fit-dir>

Run `camdl <command> --help` for any subcommand.

Model compilation is handled by `camdlc`; `check`/`inspect` wrap it (and
`camdl dev compile`/`camdl dev doctest`). Run `camdlc --help` for the raw
compiler interface."),
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[command(flatten)]
    pub global: GlobalArgs,
}

/// The options every subcommand accepts. Each converted command classifies
/// them in its own table ([`input_class::SIMULATE`]).
#[derive(clap::Args)]
pub(crate) struct GlobalArgs {
    /// Log verbosity (error/warn/info/debug/trace). Overrides RUST_LOG.
    /// Defaults to `warn`; `--progress plain` auto-bumps to `info` so
    /// per-chain progress lines (`log::info!`) reach the user.
    #[arg(long, global = true, value_name = "LEVEL",
          help_heading = "Global options")]
    verbosity: Option<log::LevelFilter>,

    /// Progress output mode for long-running subcommands. `auto` (default)
    /// uses indicatif bars on a TTY, plain log lines otherwise; `plain` forces
    /// plain lines (use under `tee`, `ssh`, or CI); `none` silences progress.
    #[arg(long, global = true, default_value_t = args::types::ProgressMode::Auto,
          value_name = "MODE", help_heading = "Global options")]
    progress: args::types::ProgressMode,

    /// Silence all progress output (shorthand for `--progress none`; wins over
    /// `--progress` if both are given).
    #[arg(long, global = true, help_heading = "Global options")]
    no_progress: bool,

    /// Bypass the compiled-IR cache: recompile the `.camdl` every run instead
    /// of reusing a cached IR keyed on (model, compiler, schema). The cache is
    /// on by default (`~/.cache/camdl/ir`, or `$CAMDL_IR_CACHE_DIR`).
    #[arg(long, global = true, help_heading = "Global options")]
    no_ir_cache: bool,

    /// Disable loop-invariant code motion (gh#272). LICM is ON by default: it
    /// hoists param/table-only subexpressions (e.g. an in-model gravity coupling
    /// kernel) out of the per-step rate evaluation so a fittable kernel runs at
    /// precomputed-kernel speed. It is value-preserving, so `--no-licm` is just
    /// an escape hatch (debugging / A-B); equivalent to `CAMDL_NO_LICM=1`. It
    /// changes the compiled IR, so a `--no-licm` run re-keys the IR cache and run
    /// identity (back to the inlined variant).
    #[arg(long, global = true, help_heading = "Global options")]
    no_licm: bool,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Run a forward simulation
    #[command(alias = "sim")]
    Simulate(args::SimulateArgs),

    /// Inference pipeline (MLE, posterior sampling, evaluation)
    #[command(subcommand)]
    Fit(FitCmd),

    /// Predict from a stored fit run's posterior, or from one point given
    /// with a model file: the free-forward and one-step bands, or a forecast
    Predict(args::PredictArgs),

    /// Score a stored prediction against the observed outcomes of the rows
    /// it predicted: log score, CRPS, WIS, PIT and interval coverage
    Score(args::ScoreArgs),

    /// Standalone bootstrap particle filter at fixed parameters
    Pfilter(args::PfilterArgs),

    /// (removed) IF2 is now a one-method fit — see `camdl fit run`
    ///
    /// Hidden from the command list: kept only so `camdl if2` / `camdl mif2`
    /// print an actionable migration message instead of a bare clap error.
    #[command(alias = "mif2", hide = true)]
    If2(args::If2Args),

    /// Profile likelihood over a parameter grid: per-cell IF2 (chain_binomial) or an nl-* optimizer (ode)
    Profile(args::ProfileArgs),

    /// Write a design: a table of parameter points for `simulate --sweep-file`
    #[command(subcommand)]
    Design(DesignCmd),

    /// Browse cached runs as a table
    List(args::ListArgs),

    /// Show full metadata for a cached run
    Show(args::ShowArgs),

    /// Emit trajectory or observation output from a cached run
    Cat(args::CatArgs),

    /// Compare fits by prequential scores (elpd, CRPS, PIT)
    Compare(args::CompareArgs),

    /// Show what differs between two models, declaration by declaration
    Diff(args::DiffArgs),

    /// Refuses: a stored result is sealed, and is named with `--label` when it runs
    Label(args::LabelArgs),

    /// Which runs on the run store are running, failed or interrupted
    Status(args::StatusArgs),

    /// Check that each completed result on the run store matches its record
    Verify(args::VerifyArgs),

    /// Delete the run store's quarantine, earlier attempts, dead staging and empty directories
    Clean(args::CleanArgs),

    /// Rebuild the run store's index from its records
    Reindex(args::StoreReindexArgs),

    /// Check results copied from another run store, then place and record them in this one
    Receive(args::ReceiveArgs),

    /// Parse and type-check a .camdl model (delegates to camdlc)
    #[command(after_help = colored_help!("\
This subcommand forwards all arguments verbatim to the OCaml compiler
`camdlc`. Run `camdlc check` with no arguments for usage, or see
`camdlc --help` for global flags.

Examples:
  # Type-check a model, reporting errors/warnings
  camdl check sir.camdl

  # Skip the dimensional-analysis checker (only for a confirmed false positive)
  camdl check sir.camdl --no-dim-check
"))]
    Check(Passthrough),

    /// Offline lineage projections (transmission tree, …) over a line list
    #[command(subcommand)]
    Lineage(LineageCmd),

    /// Print model structure (delegates to camdlc)
    #[command(after_help = colored_help!("\
This subcommand forwards all arguments verbatim to the OCaml compiler
`camdlc`. Input must be a .camdl source file (not a compiled .ir.json).
Run `camdlc inspect --help` for every option.

Common options (all parsed by camdlc):
  --summary           Compartments / transitions / parameters overview
  --dims              The physical dimension inferred for each parameter
  --compartments      List compartments (post-stratification)
  --transitions       List transitions with their rate expressions
  --tables            Show loaded table values
  --forcings          List forcings, and the data file each was read from
  --ascii             Write operators in ASCII (`*` rather than `×`)
  --no-color          Turn colour off

Examples:
  # Default summary
  camdl inspect sir.camdl

  # Show loaded tables as well
  camdl inspect sir.camdl --tables

  # Transition rates only
  camdl inspect sir.camdl --transitions

  # Which data file did each forcing compile against, and what was in it?
  # (For the machine-readable set of ALL compile-time reads — forcing data,
  #  read() tables, read() dimensions — use `camdlc --emit-deps deps.json`.)
  camdl inspect flu.camdl --forcings
"))]
    Inspect(Passthrough),

    /// Render a .camdl model as LaTeX or display JSON (delegates to camdlc)
    #[command(long_about = "\
Render a model for reading or display (delegates to camdlc).

  # LaTeX document (indexed form) to stdout
  camdl render sir.camdl

  # Structured JSON for a web viewer (KaTeX-ready blocks)
  camdl render sir.camdl --format json

  # Expand chosen dimensions to their literal strata
  camdl render seir_age.camdl --expand age")]
    Render(Passthrough),

    /// Show embedded usage guides (offline, version-matched to this binary)
    Docs(args::DocsArgs),

    /// Package a minimal reproducible example (model + data + config) to share
    #[command(subcommand)]
    Mre(MreCmd),

    /// Check whether a newer camdl release is available (queries GitHub)
    CheckUpdate,

    /// Developer & maintenance commands (rarely needed in the modeling workflow)
    #[command(subcommand)]
    Dev(DevCmd),
}

/// `camdl dev <subcommand>` — developer & maintenance commands kept out of the
/// top-level surface. Rarely needed in the modeling workflow.
#[derive(Subcommand)]
#[command(arg_required_else_help = true)]
pub(crate) enum DevCmd {
    /// Evaluate time-dependent expressions against a model
    Eval(args::EvalArgs),

    /// Compile a .camdl model to IR JSON (delegates to camdlc)
    #[command(after_help = colored_help!("\
This subcommand forwards all arguments verbatim to the OCaml compiler
`camdlc`. Flags shown above belong to camdl; camdlc's own flags (e.g.
`--set NAME=VALUE`, `--json-errors`, `--no-dim-check`) are parsed by
camdlc itself. Run `camdlc --help` for the authoritative flag set.

Examples:
  # Compile a .camdl source to IR JSON (stdout)
  camdl dev compile sir.camdl > sir.ir.json

  # Override a parameter during compilation
  camdl dev compile sir.camdl --set beta=0.3

  # Machine-readable diagnostics
  camdl dev compile sir.camdl --json-errors
"))]
    Compile(Passthrough),

    /// Compile the camdl code blocks in Markdown docs (delegates to camdlc)
    #[command(after_help = colored_help!("\
Forwards all arguments verbatim to `camdlc doctest`. Run `camdl dev doctest`
with no arguments for usage, or `camdlc --help` for the compiler interface.
Extracts the ```camdl fenced blocks from Markdown and compiles each against the
real compiler — classifying pass / skip / fail — so documented examples can't
drift.

Examples:
  # Audit a doc's camdl blocks (pass / skip / fail report, with line numbers)
  camdl dev doctest docs/spec.md

  # Gate: exit nonzero if any complete-model block fails to compile
  camdl dev doctest --gate docs/spec.md
"))]
    Doctest(Passthrough),
}

/// `camdl mre <fit|simulate>` — bundle a reproduction. See
/// `docs/dev/proposals/2026-06-09-mre-bundle.md`.
#[derive(Subcommand)]
#[command(arg_required_else_help = true,
          after_help = colored_help!("\
Examples:
  # Bundle a fit (model + read() tables + data + fixed params) into a tarball
  camdl mre fit fit.toml

  # Structure-only (no observed data values) when the data is sensitive
  camdl mre fit fit.toml --no-data -b fit-bug.mre.tar.gz

See `camdl mre <subcommand> --help` for full options."))]
pub(crate) enum MreCmd {
    /// Bundle a fit.toml's full input closure
    Fit(args::MreFitArgs),
    /// Bundle a forward-simulation reproduction
    Simulate(args::MreSimulateArgs),
}

#[derive(Subcommand)]
#[command(arg_required_else_help = true,
          after_help = colored_help!("\
Examples:
  # Run the full inference pipeline declared in fit.toml
  camdl fit run fit.toml --seed 1

  # Name the run and render its convergence verdict and estimates
  camdl fit run fit.toml --label scout
  camdl fit summary @scout

  # One row per fit run in the run store
  camdl fit table

See `camdl fit <subcommand> --help` for full options."))]
pub(crate) enum FitCmd {
    /// Run the `[method]` a fit.toml declares on its problem
    Run(args::FitRunArgs),
    /// Interpret one fit: the convergence verdict, the estimates, and the
    /// per-parameter and per-chain diagnostics behind them
    Summary(args::FitSummaryArgs),
    /// Compare two fit.toml configs
    Diff(args::FitDiffArgs),
    /// One row per fit run in the run store
    Table(args::FitTableArgs),
    /// Derive a new fit.toml from an existing one
    New(args::FitNewArgs),
    /// (removed) A prediction is `camdl predict RUN`
    #[command(hide = true)]
    Predict(args::FitPredictRemovedArgs),
    /// List supported (algorithm, backend) pairs and their descriptions
    Methods,
}

/// Offline lineage commands. `realize` replays an event log (from `camdl
/// simulate --event-log`) into a line list; the projections are pure functions
/// over a realized line list — no simulation re-run.
#[derive(Subcommand)]
#[command(arg_required_else_help = true,
          after_help = colored_help!("\
Examples:
  # Replay an event log into a line list (identity-seed picks the draw)
  camdl lineage realize event_log.parquet --identity-seed 7 -o line_list.parquet

  # Build a transmission tree from a line list, flat 10% sampling
  camdl lineage tree line_list.parquet --scheme flat:0.1 --output tree.newick

  # Dwell-time distribution in compartment 1
  camdl lineage sojourn line_list.tsv --compartment 1

  # Infection incidence in 7-day windows
  camdl lineage cohort line_list.tsv --event infection --window 7

See `camdl lineage <subcommand> --help` for full options."))]
pub(crate) enum LineageCmd {
    /// Replay an event log into a line list (Layer 2; --identity-seed)
    Realize(args::LineageRealizeArgs),
    /// Project a line list to a sampled transmission tree (Newick)
    Tree(args::LineageTreeArgs),
    /// Dwell-time distribution in a compartment
    Sojourn(args::LineageSojournArgs),
    /// Per-time-window event summary (incidence + cumulative)
    Cohort(args::LineageCohortArgs),
}

#[derive(Subcommand)]
pub(crate) enum DesignCmd {
    /// A Latin-hypercube design over a fit config's estimated parameters
    Lhs(args::DesignLhsArgs),
}

/// Captures all remaining argv tokens verbatim. Used only by Compile/Check/Inspect
/// which forward raw argv to camdlc and don't benefit from typed parsing.
#[derive(clap::Args)]
struct Passthrough {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

/// Hidden parse-only mode: `camdl __check-args -- <argv...>`.
///
/// Runs ONLY clap argument parsing against the real `Cli` command tree — no
/// file I/O, no compilation, no simulation. It answers exactly one question:
/// "does the documented invocation `camdl <argv...>` reference a subcommand /
/// flag / argument shape that this binary's CLI surface actually has?"
///
/// Exit codes:
///   0 — the args parse (the surface is real), OR clap wants to display help
///       or version for them (`--help`/`--version` are valid surface). The doc
///       command is structurally sound; any runtime failure (missing file,
///       bad value) is NOT our concern here.
///   2 — clap rejected the surface: unknown subcommand, unrecognized flag,
///       unexpected positional, or a too-few/too-many arg-count violation.
///       This is DRIFT — the doc names something the CLI does not expose.
///
/// This is parser-truth, not stderr string-matching: the same typed clap
/// parser the real CLI uses decides, so the gate can never disagree with the
/// binary about what's a valid command surface.
///
/// `compile` / `check` / `inspect` forward verbatim to camdlc via the
/// `Passthrough` arg (`trailing_var_arg` + `allow_hyphen_values`). clap
/// therefore accepts ANY tail after them (e.g. `camdl compile m.camdl
/// --set b=0.3 --json-errors`), so this check necessarily passes those
/// through as OK — their flags belong to camdlc, not camdl, and are out of
/// scope for camdl-surface drift detection. Documented as a known limitation.
///
/// Intercepted at the very top of `main`, before `Cli::parse`, so the
/// `__check-args` token never reaches the real subcommand dispatch and the
/// inner argv is validated against the unmodified `Cli` tree (program name
/// `camdl` is synthesized as argv[0]).
fn run_check_args_mode() -> Option<i32> {
    use clap::CommandFactory;
    let raw: Vec<String> = std::env::args().collect();
    // Expect: camdl __check-args -- <argv...>
    if raw.len() < 2 || raw[1] != "__check-args" {
        return None;
    }
    // Strip argv[0] (`camdl`), argv[1] (`__check-args`), and an optional `--`
    // separator. Everything after is the documented invocation's tail.
    let mut rest: &[String] = &raw[2..];
    if rest.first().map(String::as_str) == Some("--") {
        rest = &rest[1..];
    }
    // Reconstruct the full argv clap expects: program name + the doc tail.
    let mut argv: Vec<String> = Vec::with_capacity(rest.len() + 1);
    argv.push("camdl".to_string());
    argv.extend_from_slice(rest);

    let cmd = Cli::command();
    match cmd.try_get_matches_from(argv) {
        Ok(_) => Some(0),
        Err(e) => {
            use clap::error::ErrorKind;
            match e.kind() {
                // Help/version requests are NOT drift — the flags are real and
                // the surface is valid. clap models them as "errors" only
                // because it short-circuits parsing to render the text.
                ErrorKind::DisplayHelp
                | ErrorKind::DisplayVersion
                | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand => Some(0),
                // A missing *required* argument means the surface itself is
                // valid (the subcommand/flags were recognized) — the user just
                // didn't supply a positional. That's an input concern (EXPECTED),
                // not surface drift, so we do not flag it.
                ErrorKind::MissingRequiredArgument => Some(0),
                // Everything else clap rejects at the parse layer is surface
                // drift: unknown subcommand, unrecognized flag, unexpected
                // positional, bad arg count, invalid enum value, etc.
                _ => Some(2),
            }
        }
    }
}

/// The command line, parsed, or exit with clap's usage error (status 2).
///
/// An unknown `--output-dir` is answered with the flags that do its job,
/// `--store DIR` and `-o`, in place of clap's nearest-spelling tip:
/// `--output` exports one table on most commands and names no store, so the
/// tip would send a person looking for the store's root to the wrong flag.
fn parse_cli() -> Cli {
    use clap::error::{ContextKind, ContextValue, ErrorKind};
    let e = match Cli::try_parse() {
        Ok(cli) => return cli,
        Err(e) => e,
    };
    let output_dir = e.kind() == ErrorKind::UnknownArgument
        && matches!(e.get(ContextKind::InvalidArg),
            Some(ContextValue::String(arg)) if arg == "--output-dir" || arg.starts_with("--output-dir="));
    if !output_dir {
        e.exit();
    }
    out::Out::new(false).say(
        "error: unexpected argument '--output-dir' found\n\n  \
         tip: runs are stored in the run store, whose root is `--store DIR` (on `fit run`, \
         the fit file's `store` key), else `CAMDL_STORE`, else `./camdl-store`; `-o` \
         exports a copy of a run's output\n\n\
         For more information, try '--help'."
    );
    std::process::exit(2);
}

fn main() {
    // Hidden parse-only drift check (used by `make test-cli-docs`). Returns
    // an exit code and short-circuits before any real parsing/execution.
    if let Some(code) = run_check_args_mode() {
        std::process::exit(code);
    }

    let cli = parse_cli();
    // The environment is read once (`env`), and the engine is told its
    // evaluator switches before any command runs a model.
    env::install_engine_switches(&env::get().vars);

    // The log filter: explicit --verbosity, else RUST_LOG, else `warn`.
    // Plain progress lines do not pass through `log` (`out::LiveHandle`,
    // GH #14's option 2), so `--progress` no longer bumps it.
    let effective_verbosity: log::LevelFilter = cli.global.verbosity.unwrap_or(log::LevelFilter::Warn);

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(effective_verbosity.as_str())
    ).init();

    // Progress output policy (GH #14). Must run after env_logger so that
    // plain-mode log lines from callbacks reach the configured filter.
    // `--no-progress` is the discoverable shorthand for `--progress none` and
    // wins when both are given.
    out::progress::init(if cli.global.no_progress { args::types::ProgressMode::None } else { cli.global.progress });
    // What the engine meets mid-run and records once (`sim::run_notes`) is
    // said as it is first recorded, as a warning.
    sim::run_notes::set_reporter(util::report_run_note);
    // `--no-licm` / `CAMDL_NO_LICM` and `--no-ir-cache` / `CAMDL_NO_IR_CACHE`:
    // the settings every compile of this invocation runs under, resolved once
    // and passed to each command that compiles (`util::CompileSettings`).
    let compile = util::CompileSettings::resolve(cli.global.no_licm, cli.global.no_ir_cache);

    match cli.command {
        Command::Simulate(a)            => run_simulate(args::SimulateInvocation {
            args: a,
            no_licm: cli.global.no_licm,
            no_ir_cache: cli.global.no_ir_cache,
            verbosity: cli.global.verbosity,
            progress: cli.global.progress,
            no_progress: cli.global.no_progress,
        }),
        Command::Fit(FitCmd::Run(a))    => fit_command::cmd_fit_run(args::FitRunInvocation {
            args: a,
            no_licm: cli.global.no_licm,
            no_ir_cache: cli.global.no_ir_cache,
            verbosity: cli.global.verbosity,
            progress: cli.global.progress,
            no_progress: cli.global.no_progress,
        }),
        Command::Fit(FitCmd::Summary(a))=> fit::cmd_fit_summary(&a, compile),
        Command::Fit(FitCmd::Diff(a))   => fit::cmd_fit_diff(&a, compile),
        Command::Fit(FitCmd::Table(a))  => fit::cmd_fit_table(&a),
        Command::Fit(FitCmd::New(a))    => fit::cmd_fit_new(&a, compile),
        Command::Fit(FitCmd::Predict(_)) => predict_command::refuse_fit_predict(),
        Command::Predict(a)             => predict_command::cmd_predict(args::PredictInvocation {
            args: a,
            no_licm: cli.global.no_licm,
            no_ir_cache: cli.global.no_ir_cache,
            verbosity: cli.global.verbosity,
            progress: cli.global.progress,
            no_progress: cli.global.no_progress,
        }),
        Command::Score(a)               => score_command::cmd_score(args::ScoreInvocation {
            args: a,
            verbosity: cli.global.verbosity,
        }),
        Command::Fit(FitCmd::Methods)   => fit::cmd_fit_methods(),
        Command::Label(a)               => fit::cmd_label(&a),
        Command::Pfilter(a)             => pfilter::cmd_pfilter(args::PfilterInvocation {
            args: a,
            no_licm: cli.global.no_licm,
            no_ir_cache: cli.global.no_ir_cache,
            verbosity: cli.global.verbosity,
            progress: cli.global.progress,
            no_progress: cli.global.no_progress,
        }),
        Command::If2(a)                 => if2::cmd_if2(&a),
        Command::Profile(a)             => profile::cmd_profile(args::ProfileInvocation {
            args: a,
            no_licm: cli.global.no_licm,
            no_ir_cache: cli.global.no_ir_cache,
            verbosity: cli.global.verbosity,
            progress: cli.global.progress,
            no_progress: cli.global.no_progress,
        }),
        Command::Design(DesignCmd::Lhs(a)) => design::cmd_design_lhs(&a, compile),
        Command::Lineage(LineageCmd::Realize(a)) => lineage::cmd_lineage_realize(&a),
        Command::Lineage(LineageCmd::Tree(a)) => lineage::cmd_lineage_tree(&a),
        Command::Lineage(LineageCmd::Sojourn(a)) => lineage::cmd_lineage_sojourn(&a),
        Command::Lineage(LineageCmd::Cohort(a)) => lineage::cmd_lineage_cohort(&a),
        Command::List(a)                => browse::cmd_list(&a),
        Command::Show(a)                => browse::cmd_show(&a),
        Command::Cat(a)                 => browse::cmd_cat(&a),
        Command::Compare(a)             => compare::cmd_compare(&a, compile),
        Command::Diff(a)                => model_diff::cmd_diff(&a, compile),
        Command::Check(a)               => delegate("check", &a.args, compile),
        Command::Inspect(a)             => delegate("inspect", &a.args, compile),
        Command::Render(a)              => delegate("render", &a.args, compile),
        Command::Docs(a)                => docs::cmd_docs(&a),
        Command::Mre(MreCmd::Fit(a))      => mre::cmd_mre_fit(&a, compile),
        Command::Mre(MreCmd::Simulate(a)) => mre::cmd_mre_simulate(&a, compile),
        Command::CheckUpdate            => check_update::cmd_check_update(),
        Command::Status(a)  => store_commands::cmd_store(store::StoreCommand::Status, a.store.store),
        Command::Verify(a)  => store_commands::cmd_store(store::StoreCommand::Verify { handle: a.handle }, a.store.store),
        Command::Clean(a)   => store_commands::cmd_store(store::StoreCommand::Clean, a.store.store),
        Command::Reindex(a) => store_commands::cmd_store(store::StoreCommand::Reindex, a.store.store),
        Command::Receive(a) => store_commands::cmd_receive(a),
        Command::Dev(DevCmd::Eval(a))    => eval::cmd_eval(&a, compile),
        Command::Dev(DevCmd::Compile(a)) => {
            let refs: Vec<&str> = a.args.iter().map(String::as_str).collect();
            util::delegate_to_camdlc(&refs, compile).unwrap_or_else(|e| {
                out::Out::new(false).fail(format!("error: {}", e));
            });
        }
        Command::Dev(DevCmd::Doctest(a)) => delegate("doctest", &a.args, compile),
    }
}

/// A compiler subcommand passed through to camdlc, under the invocation's
/// compile settings.
fn delegate(subcommand: &str, args: &[String], compile: util::CompileSettings) {
    let mut refs = vec![subcommand];
    refs.extend(args.iter().map(String::as_str));
    util::delegate_to_camdlc(&refs, compile).unwrap_or_else(|e| {
        out::Out::new(false).fail(format!("error: {}", e));
    });
}

// Per-(point, replicate) seed derivation lives in one place,
// `util::mix_cell_seed` (with `util::SEED_MIX_OBS` for the obs stream);
// `engine::run_job` routes through it. The constants
// below are for the *other*, unrelated RNG streams owned by this module.
const SEED_MIX_UNIFORM: u64 = 0xd4a5_b1ce;      // uniform draws RNG
const SEED_MIX_PRIOR: u64  = 0x0014_b1ce;      // prior draws RNG

/// Where `--init-state` draws its ensemble of forecast-origin states.
///
/// The two sources are different objects, not two spellings of one: a file is a
/// particle swarm at a SINGLE θ (so it pairs with replicates and refuses
/// `--draws`), while `fit` is one state PER POSTERIOR DRAW (so it requires
/// `--draws posterior` and pairs with the draw axis). Parsing the flag into
/// this type at the boundary is what keeps those two row axes from being one
/// `usize` that a call site can pass the wrong index into.
#[derive(Debug, Clone, PartialEq)]
enum InitStateSourceArg {
    /// A `camdl pfilter --save-final-state` TSV (gh#641).
    File(std::path::PathBuf),
    /// The `--fit` run's paired `(θ_i, X_i(T))` posterior (gh#697).
    Fit,
}

impl InitStateSourceArg {
    /// The bare word `fit` selects the paired source; anything else is a path.
    /// Same convention as `--draws uniform|prior|posterior|<file>` — a keyword
    /// wins over a file of that name.
    fn parse(raw: Option<&str>) -> Option<Self> {
        match raw {
            None => None,
            Some("fit") => Some(InitStateSourceArg::Fit),
            Some(path) => Some(InitStateSourceArg::File(std::path::PathBuf::from(path))),
        }
    }
}

/// Resolve `--init-state fit --draws posterior`: ONE join producing both the θ
/// cloud and the origin ensemble it is paired with (gh#697).
///
/// Returns `(θ rows, origin ensemble)` in the SAME order, which is what makes
/// draw *i*'s state and draw *i*'s parameters impossible to mis-pair: they come
/// out of one `Vec<ForecastDraw>` and are split only at the very end, into two
/// structures the engine indexes with the same `point_idx`.
///
/// The forkable subset is used and REPORTED, never silently substituted for the
/// posterior: only draws with a saved latent path have an `X_i(T)` to fork, and
/// a cloud quietly banded over a fraction of the posterior looks exactly like
/// the full one. A fit with no saved paths is refused by name — never a fall
/// back to `init {}`, which would look like a forecast and be a free-forward
/// replay.
fn resolve_paired_posterior(
    out: &crate::out::Out,
    fit_ref: &str,
    ir_path_compiled: &str,
    n_draws: Option<usize>,
    selection: Option<&crate::chain_selection::ChainSelection>,
) -> (Vec<HashMap<String, f64>>, crate::sim_job::InitStateSource) {
    let (model, _) = util::load_ir(ir_path_compiled).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    let columns = io::trajectories::TrajColumnSpec::from_model(&model, &[]);
    // The draws authority for this fit, carrying the `--chains` /
    // `--exclude-chains` selection, so the paired cloud is filtered at the one
    // seam every reader shares (gh#695).
    let pref = crate::posterior_draws::resolve_posterior_draws(fit_ref, None)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
    if let Some(sel) = selection {
        warn_posterior_selection(out, sel, &pref.draws_path);
    }
    let pref = pref.with_selection(selection.cloned());
    let ens = crate::fit::joint::resolve_forecast_ensemble(&pref, &columns)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });

    if ens.draws.is_empty() {
        out.fail(format!(
            "error: --init-state fit: {fit_ref} has no forkable posterior draws \
             (0/{} carry a saved latent path). The paired (θ_i, X_i(T)) origin comes \
             from the smoothed latent paths a PGAS method writes to \
             <method>/chain_*/trajectories.tsv; PMMH and particle-filter methods write \
             none, and a method run with `n_trajectories = 0` writes none either.\n  \
             Refusing rather than starting from `init {{}}`: that would look like a \
             forecast and be a free-forward replay.\n  Fix: re-fit with a PGAS method \
             (or raise `n_trajectories`), or forecast at a single θ with \
             `camdl pfilter --save-final-state` + `--init-state <file>`.",
            ens.n_total,
        ));
    }

    let n_forkable = ens.draws.len();
    out.say(format_args!(
        "simulate: --init-state fit → {n_forkable}/{} posterior draws have a saved \
         latent path; forecasting that paired (θ_i, X_i) subset from t = {}{}\n  \
         method '{}' ({})",
        ens.n_total,
        ens.origin_t,
        if n_forkable < ens.n_total {
            format!(" (the other {} draws have no state to fork)", ens.n_total - n_forkable)
        } else {
            String::new()
        },
        ens.name,
        ens.draws_path.display(),
    ));

    // Same strided cap as the unpaired posterior path (never front-biased): a
    // large forkable subset is still hours of forward solves.
    let cap = n_draws.unwrap_or(crate::fit::predict::DEFAULT_PREDICT_DRAWS);
    if let Some(asked) = n_draws {
        if asked > n_forkable {
            out.say(format_args!(
                "simulate: --init-state fit → -n {asked} exceeds the {n_forkable} \
                 forkable draws; forecasting all {n_forkable}."
            ));
        }
    }
    let selected: Vec<crate::fit::joint::ForecastDraw> = if n_forkable > cap {
        let picked: Vec<crate::fit::joint::ForecastDraw> =
            crate::fit::predict::subsample_draws(&ens.draws, cap)
                .into_iter().cloned().collect();
        out.say(format_args!(
            "simulate: --init-state fit → subsampling {} of {n_forkable} forkable \
             draws (strided across the subset; raise with --n-draws)",
            picked.len()
        ));
        picked
    } else {
        ens.draws
    };

    // The identity input: a content digest over the ensemble ACTUALLY used —
    // origin time, and every selected draw's key + restored values, in
    // selection order. Content, not the fit's run_id: keying on provenance
    // means enumerating every knob that changes the selection, and a missed one
    // is two different clouds sharing a cache entry.
    let origin_t = store::Finite::new(ens.origin_t).unwrap_or_else(|e| {
        out.fail(format!("error: --init-state fit: non-finite forecast origin ({e})"));
    });
    let rows: Vec<crate::sim_store::InitStateDraw> = selected
        .iter()
        .map(|d| crate::sim_store::InitStateDraw {
            chain: d.chain as u64,
            draw: d.draw as u64,
            counts: d.counts.clone(),
            reals: d.reals.iter()
                .map(|&v| store::Finite::new(v).unwrap_or_else(|e| {
                    out.fail(format!(
                        "error: --init-state fit: (chain {}, draw {}) has a non-finite \
                         real compartment at the origin ({e})",
                        d.chain, d.draw
                    ));
                }))
                .collect(),
        })
        .collect();
    let digest = store::Identity::key(&crate::sim_store::InitStateEnsemble { origin_t, rows });

    let params: Vec<HashMap<String, f64>> =
        selected.iter().map(|d| d.params.clone()).collect();
    let states: Vec<crate::sim_job::OriginState> = selected
        .into_iter()
        .map(|d| crate::sim_job::OriginState { counts: d.counts, reals: d.reals })
        .collect();
    (
        params,
        crate::sim_job::InitStateSource {
            origin_t: ens.origin_t,
            states,
            axis: crate::sim_job::InitStateRowAxis::Draw,
            ensemble_digest: digest,
        },
    )
}

#[expect(clippy::disallowed_methods, reason = "reads the IR camdlc compiled, the `--init-state` file (declared `#[input(identity)]`; its digest is the cell identity's ensemble) and the `--params` file, whose digest is recorded as lineage while its values are hashed")]
fn run_simulate(inv: args::SimulateInvocation) {
    // simulate has no `--no-color`: its painters follow the environment and
    // whether each stream is a terminal.
    let out = crate::out::Out::new(false);
    // The inputs split by class. Every declared input is taken apart here and
    // placed in the arguments the resolution below reads, so an input added to
    // `SimulateArgs` does not compile until it is placed; `--dry-run` is
    // `skip` and is read before the split.
    let dry_run = inv.args.dry_run;
    let (declared, execution, exports, request, given) = store::Inputs::split(inv).into_parts();
    let (declared, no_licm) = declared.into_parts();
    let (
        model, to, from, init_state, allow_degenerate_rates, forcing_extrapolation, model_overrides, param_vec,
        scenarios, enable, disable, backend, output_view, seed, seeds, replicates, draws, sweep, sweep_file, fit,
        n_draws, chains, exclude_chains, emit_every, obs, obs_dir, obs_only, obs_only_dir, quantities_out,
        quantities, event_log,
    ) = declared.into_parts();
    let (allow_forcing_extrapolation,) = forcing_extrapolation.into_parts();
    let (params, param, table) = model_overrides.into_parts();
    let (backend_flag, dt_flag, integrator_flag) = backend.into_parts();
    let (every, no_flows, columns) = output_view.into_parts();
    let (exports,) = exports.into_parts();
    let ((), (), (), (), draws_out, output, stdout, design_from, projected_out, dates, format, tsv, reactive_log) =
        exports.into_parts();
    let (request,) = request.into_parts();
    let ((), (), (), (), store_root, force, label) = request.into_parts();
    let args::SimulateInvocationExecution { args: args::SimulateExecution { parallel, .. }, no_ir_cache } = execution;
    let compile = util::CompileSettings::resolve(no_licm, no_ir_cache);
    // A fit file's `store` key places the simulation in that fit's store
    // (run spec §2.1). A file that cannot be read names no store here; each
    // purpose of `--fit` reads the file again and reports its own error.
    let file_store = fit.as_deref()
        .map(|p| p.to_string_lossy().into_owned())
        .filter(|text| crate::fit_store::is_config_file(text))
        .and_then(|text| crate::fit::config_v2::FitConfig::load(&text).ok())
        .and_then(|c| c.problem.store)
        .map(std::path::PathBuf::from);
    let request = crate::camdl_store::store_request_with_fit_file(label.clone(), force, store_root.clone(), file_store);
    // A stored fit `--fit` names is read in the store this command stores in.
    crate::fit_store::set_reader_root(crate::camdl_store::root_of(&request));
    let execution_record = args::SimulateInvocationExecution {
        args: args::SimulateExecution { forcing_extrapolation: (), model_overrides: (), backend: (), output_view: (), parallel },
        no_ir_cache,
    };
    let a = &args::SimulateArgs {
        model, to, from, init_state, allow_degenerate_rates,
        forcing_extrapolation: args::ForcingExtrapolationArg { allow_forcing_extrapolation },
        model_overrides: args::ModelOverrides { params, param, table },
        param_vec, scenarios, enable, disable,
        backend: args::SimBackend { backend: backend_flag, dt: dt_flag, integrator: integrator_flag },
        output_view: args::OutputView { every, no_flows, columns },
        seed, seeds, replicates, draws, sweep, sweep_file, fit, n_draws, chains, exclude_chains, draws_out, output,
        stdout, emit_every, obs, obs_dir, obs_only, obs_only_dir, design_from, projected_out, dates, quantities_out,
        quantities, dry_run, store: store_root, parallel, force, label, event_log, format, tsv, reactive_log,
    };
    // A file an export writes after the cells are stored must have a
    // directory to go to; one that has none is refused before anything runs.
    let file_exports: [(&str, Option<&std::path::Path>); 6] = [
        ("-o", a.output.as_deref()),
        ("--obs", a.obs.as_deref()),
        ("--obs-only", a.obs_only.as_deref().filter(|p| !p.to_string_lossy().ends_with('/') && !p.is_dir())),
        ("--draws-out", a.draws_out.as_deref()),
        ("--event-log", a.event_log.as_deref().filter(|p| p.as_os_str() != "auto")),
        ("--reactive-log", a.reactive_log.as_deref()),
    ];
    for (flag, path) in file_exports {
        if let Some(path) = path {
            util::refuse_missing_export_dir(flag, path).unwrap_or_else(|e| out.fail(format!("error: {e}")));
        }
    }
    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();
    // The compile settings (`--no-licm`) are passed to the one compile below
    // and to each compile of a replayed fit's config, and reach the identity
    // through the IR they produce. `--allow-degenerate-rates` reaches every
    // cell's compiled model through the resolved configuration's engine
    // settings; the forcing-extrapolation policy they hold is installed once
    // they exist, before any cell runs (`EngineSettings::install_forcing_policy`).
    // `--parallel`, `--store`, `--force`: how the cells are computed and
    // where they are stored, never what (`sim_config::Execution`).
    let execution = crate::sim_config::Execution {
        // `--parallel N` / CAMDL_PARALLEL: the Rayon thread count for the
        // simulation phase of a multi-cell run. Absent → sequential; 0 → every
        // core, as the other verbs read it.
        threads: match a.parallel {
            None => 1,
            Some(0) => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            Some(n) => n,
        },
    };
    // ── Extract typed args into locals that match the rest of the function ─
    let ir_path          = a.model.to_string_lossy().into_owned();
    // Compile `.camdl` → IR EXACTLY ONCE for the whole command. Every
    // downstream load (obs preflight, --draws generation, the CAS-sink base
    // model, and every engine cell) reads this resolved `.ir.json` instead of
    // re-invoking camdlc per unit. Without this, a multi-cell run (--replicates
    // / --seeds / --draws / multiple --scenario) compiled once per cell: the
    // repeated camdlc spinners stomp each other on a TTY (orphaned/overwritten
    // bar), and a large stratified model paid the ~20s compile N times.
    // `_ir_tmp` is the temp-file path (None for a plain `.ir.json` input); held
    // alive for the whole function so it isn't reaped mid-run.
    // `ir_path` (the original `.camdl`) is preserved for display + provenance
    // (dry-run model line, CAS `model_path`/`model_stem`); `ir_path_compiled`
    // is the resolved IR used for all compilation.
    // `simulate` never reads the state-Jacobian, so compile lean
    // (`needs_state_grad = false`, gh#439 A2 — `--no-state-grad`).
    // `--quantities FILE` (proposal 2026-08-19): a reporting vocabulary compiled
    // in place of the model's own `quantities {}` block. Loaded here, before the
    // compile, because it is a COMPILER input — the block is resolved against
    // this model's symbols, so a name the model does not declare is a compile
    // error naming FILE. Its bytes also key the IR cache and, below, the emitted
    // tables. `Model::hash_into` excludes quantities, so the trajectory leaves
    // and their `run_id`s are unaffected.
    let quantities_override: Option<crate::quantities_file::QuantitiesOverride> =
        match a.quantities.as_deref().map(crate::quantities_file::QuantitiesOverride::load) {
            None => None,
            Some(Ok(q)) => Some(q),
            Some(Err(e)) => {
                out.fail(format!("error: {}", e));
            }
        };
    // `ir_path_given` is the model as named, before any run-time rewrite: a
    // replay of a stored fit compares THIS against the fit's archived model,
    // since `--output-every` below changes the output schedule the model
    // identity hashes, and is a property of this run, not of the model.
    let (ir_path_given, _ir_tmp) =
        util::resolve_ir_path(&ir_path, false, quantities_override.as_ref(), compile)
            .unwrap_or_else(|e| {
                out.fail(format!("error: {}", e));
            });
    // gh#156: `--output-every` rewrites the compiled IR's output schedule once,
    // so BOTH the engine (loads by path) and the CAS identity (`base_model`,
    // also loaded from this path) see the overridden cadence. `_every_tmp`
    // holds the rewritten temp alive for the function scope.
    let (ir_path_compiled, _every_tmp) =
        util::rematerialize_with_output_every(&ir_path_given, a.output_view.every)
            .unwrap_or_else(|e| {
                out.fail(format!("error: {}", e));
            });
    // gh#156: resolve + validate the trajectory output view (`--no-flows` /
    // `--columns`) against the compiled model, once. Folded into the CAS
    // identity (`config` level) and used by both trajectory writers.
    // gh#656: `--emit-every` is resolved and validated in the same pass, against
    // the same model — an unknown stream label, a fit-only stream, or an
    // `at [...]` schedule is refused BEFORE anything runs. The model copy is
    // scoped to this block so a large stratified IR is not held for the whole
    // command.
    let (output_cols, emit_every) = match std::fs::read_to_string(&ir_path_compiled)
        .map_err(|e| format!("cannot read {}: {}", ir_path_compiled, e))
        .and_then(|s| ir::from_str(&s).map_err(|e| format!("IR load error: {}", e)))
        .and_then(|m| {
            let cols = util::OutputColumns::resolve(&a.output_view, &m)?;
            let emit = crate::emit_every::EmitEvery::from_cli_specs(&a.emit_every)?;
            if let Some(e) = &emit {
                e.validate(&m.observations)?;
            }
            Ok((cols, emit))
        })
    {
        Ok(c) => c,
        Err(e) => {
            out.fail(format!("error: {}", e));
        }
    };
    // Default is chain_binomial so `simulate` and `fit` agree at the
    // same MLE params (see docs/dev/incidents/2026-04-19-backend-default-mismatch.md).
    let mut backend      = a.backend.backend.unwrap_or(args::types::ForwardBackend::ChainBinomial);
    // The exploratory step: `--dt`, else the model's own `simulate { dt = … }`,
    // else 1.0. A replay of a fit's parameters (a `--fit`, or a params file
    // carrying a fit's `[provenance]`) replaces both with the fit's simulator
    // below.
    let model_dt: Option<f64> = util::peek_simulation_dt(&ir_path_compiled);
    let mut dt           = a.backend.dt.or(model_dt).unwrap_or(1.0_f64);
    let seed             = a.seed;
    // The engine's Rayon branch runs on the global pool, which defaults to
    // every core, so the pool is capped here (mirrors fit/pfilter/profile);
    // `build_global` is one-shot per process and an
    // already-initialised Err is ignored as those verbs do.
    if execution.threads > 1 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(execution.threads).build_global();
    }
    let overrides: HashMap<String, f64> = a.model_overrides.param.iter()
        .map(|p| (p.name.clone(), p.value)).collect();
    let table_files: HashMap<String, String> = a.model_overrides.table.iter()
        .map(|t| (t.name.clone(), t.path.to_string_lossy().into_owned())).collect();
    let params_files: Vec<String> = a.model_overrides.params.iter()
        .map(|p| p.to_string_lossy().into_owned()).collect();
    let set_vec_entries: Vec<(String, String)> = a.param_vec.iter()
        .map(|pv| (pv.prefix.clone(), pv.file.clone())).collect();
    let scenario_names: Vec<String> = crate::args::split_scenario_names(&a.scenarios)
        .unwrap_or_else(|e| {
            out.fail(format!("error: {e}"));
        });
    let adhoc_enable: Vec<String>  = a.enable.clone();
    let adhoc_disable: Vec<String> = a.disable.clone();
    let seeds: Vec<u64> = match &a.seeds {
        Some(spec) => spec.expand(),
        None       => vec![a.seed],
    };
    let seeds_spec_given = a.seeds.is_some();
    let mut obs_path: Option<String> = a.obs.as_ref().map(|p| p.to_string_lossy().into_owned());
    let mut obs_dir: Option<String>  = a.obs_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let obs_only: Option<String>     = a.obs_only.as_ref().map(|p| p.to_string_lossy().into_owned());
    let obs_only_dir: Option<String> = a.obs_only_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let replicates: usize            = a.replicates.unwrap_or(1);
    let draws_path: Option<String>   = a.draws.clone();
    let n_draws_arg: Option<usize>   = a.n_draws;
    let fit_path_for_draws: Option<String> = a.fit.as_ref().map(|p| p.to_string_lossy().into_owned());
    let dry_run     = a.dry_run;
    // Every `simulate` run writes to the store; `--output`/`--obs` are
    // additive mirrors of it, never replacements.

    // --obs-only implies --obs or --obs-dir (infer from path: trailing / or existing dir → obs-dir)
    if let Some(ref path) = obs_only {
        if obs_path.is_some() || obs_dir.is_some() {
            out.fail("error: --obs-only cannot be combined with --obs or --obs-dir");
        }
        if path.ends_with('/') || std::path::Path::new(path).is_dir() {
            obs_dir = Some(path.clone());
        } else {
            obs_path = Some(path.clone());
        }
    }
    // --obs-only-dir is the explicit, unambiguous dir form (run-spec §3.1.1
    // ObsOutput::OnlyDir). Unlike --obs-only it never infers file-vs-dir from
    // the path: one TSV per stream, always, and trajectory suppressed.
    if let Some(ref path) = obs_only_dir {
        if obs_path.is_some() || obs_dir.is_some() {
            out.fail("error: --obs-only-dir cannot be combined with --obs, --obs-dir, or --obs-only");
        }
        obs_dir = Some(path.clone());
    }
    let suppress_trajectory = obs_only.is_some() || obs_only_dir.is_some();

    if replicates < 1 {
        out.fail("error: --replicates must be >= 1");
    }

    let want_obs = obs_path.is_some() || obs_dir.is_some();

    // gh#656: `--emit-every` only reaches emitted synthetic observations — the
    // `--obs*` writers, the CAS obs subtree they enable, and an obs-sourced
    // quantity. A run that emits none would silently ignore it, so refuse and
    // name what would make it do something.
    if emit_every.is_some() && !want_obs && a.quantities_out.is_none() {
        out.fail(
            "error: --emit-every sets the cadence of EMITTED synthetic \
             observations, but this run emits none.\n  \
             Add --obs/--obs-dir/--obs-only/--obs-only-dir to write them, or \
             --quantities-out for an `observations.<stream>` quantity."
        );
    }

    if seeds_spec_given && replicates > 1 {
        out.fail("error: --seeds and --replicates are mutually exclusive.\n  \
                   --seeds provides explicit seed values.\n  \
                   --replicates generates N deterministic seeds from --seed.");
    }
    // If using --seeds, replicates tracks seed count
    let replicates = if seeds_spec_given { seeds.len() } else { replicates };

    // Validate mutually exclusive σ flags
    if !scenario_names.is_empty() && (!adhoc_enable.is_empty() || !adhoc_disable.is_empty()) {
        out.say("error: --scenario and --enable/--disable are mutually exclusive.");
        out.say("  --scenario selects a named scenario from the model file.");
        out.say("  --enable/--disable compose an ad-hoc scenario.");
        out.fail("  To combine both, define a composed scenario in the model file.");
    }

    // If no scenarios specified, use a single None (the implicit `(none)` arm)
    let scenario_list: Vec<Option<String>> = if scenario_names.is_empty() {
        vec![None]
    } else {
        scenario_names.iter().map(|s| Some(s.clone())).collect()
    };

    // ── gh#626: resolve `--to` (obs-anchored horizon override) once, up front.
    // Anchored forms (`last_obs + 8 weeks`) fold over the fit's bound
    // observation data; absolute forms (number, date) resolve data-free. The
    // resolved value overrides every cell's `simulation.t_end` (applied in
    // `util::resolve_loaded_run` after the scenario horizon) and is keyed into
    // run identity from the model each cell runs (`SimConfig::t_end`).
    //
    // gh#616: the MODEL may also be anchored (`simulate { to = last_obs + 4
    // 'weeks }`, a scenario's, or a forcing's `breakpoints`). Those resolve from
    // the same bound data as `--to`, through the same fold, so the observed
    // window is resolved ONCE here and used for both. The model is then
    // substituted in place, so every up-front check below (the horizon
    // ordering, the `--to` conflict rule) reads resolved numbers rather than the
    // NaN the compiler baked.
    let (model_raw, _) = util::load_ir(&ir_path_compiled).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });
    let to_is_anchored = matches!(
        a.to.as_deref().map(|raw| crate::fit::runner::parse_time_spec(
            "--to", raw, model_raw.origin.as_deref(), &model_raw.time_unit)),
        Some(Ok(crate::fit::runner::TimeSpec::Anchored(_)))
    );
    // The run's start: `--from` over the `--fit` config's `[simulate] from`
    // over the model's own `simulate { from }`. Resolved here, once, to a
    // number every cell's model is moved to (`util::load_run_model`), so the
    // start enters each cell's identity through the model it hashes.
    let start_override: Option<crate::obs_anchor::StartOverride> = match (a.from.as_deref(), a.fit.as_ref()) {
        (Some(raw), _) => Some(crate::obs_anchor::StartOverride::parse("--from", raw, &model_raw)
            .unwrap_or_else(|e| { out.fail(format!("error: {e}")); })),
        (None, Some(fit_ref)) => simulate_fit_config_start(fit_ref)
            .unwrap_or_else(|e| { out.fail(format!("error: --fit {}: {e}", fit_ref.display())); })
            .map(|raw| crate::obs_anchor::StartOverride::parse("[simulate] from", &raw, &model_raw))
            .transpose()
            .unwrap_or_else(|e| { out.fail(format!("error: {e}")); }),
        (None, None) => None,
    };
    if start_override.is_some() && a.init_state.is_some() {
        out.fail(format!(
            "error: {} and --init-state both set the run's start: --init-state starts \
             the run at the forecast origin its state was filtered to.\n  Fix: drop {}.",
            start_override.as_ref().map(|s| s.source).unwrap_or("--from"),
            start_override.as_ref().map(|s| s.source).unwrap_or("--from"),
        ));
    }
    let from_is_anchored = start_override.as_ref().is_some_and(|s| s.is_anchored());
    // Whether the model (as the start override leaves it) needs bound data: a
    // literal `--from` over a model whose only anchor is its `from` does not.
    let model_is_anchored = crate::obs_anchor::needs_window(&model_raw, start_override.as_ref());
    let obs_anchors: Option<ir::anchor::ObsAnchorTimes> =
        if model_is_anchored || to_is_anchored {
            let Some(fit_ref) = a.fit.as_ref() else {
                let what = if from_is_anchored {
                    format!("{} \"{}\" is anchored to observed data",
                            start_override.as_ref().map(|s| s.source).unwrap_or("--from"),
                            start_override.as_ref().map(|s| s.raw.as_str()).unwrap_or(""))
                } else if model_is_anchored {
                    format!("this model is anchored to observed data ({})",
                            crate::obs_anchor::anchored_sites(&model_raw).join(", "))
                } else {
                    "--to is anchored to observed data".to_string()
                };
                out.fail(format!(
                    "error: {what}, but a forward simulation binds none.\n  \
                     Fix: pass --fit <fit.toml | fit run dir> — its \
                     [data.observations] supplies the observed times the anchor \
                     resolves against."
                ));
            };
            let (first, last) = resolve_simulate_obs_anchors(
                &crate::obs_anchor::window_model(&model_raw, start_override.as_ref()), fit_ref, dt)
                .unwrap_or_else(|e| {
                    out.fail(format!("error: resolving observation anchors: {e}"));
                });
            Some(ir::anchor::ObsAnchorTimes { first, last })
        } else {
            None
        };
    // The locally-substituted model the up-front checks read. The RUN and the
    // CAS `base_model` are substituted separately, at their own loads, with this
    // same window — see `obs_anchor`'s module doc for why that is the seam.
    let mut t_start_override: Option<f64> = None;
    let model_to = {
        let mut m = model_raw;
        if let Some(start) = &start_override {
            let r = crate::obs_anchor::apply_start(&mut m, start, obs_anchors)
                .unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
            crate::obs_anchor::report(&out, std::slice::from_ref(&r), &m);
            t_start_override = Some(r.resolved);
        }
        if let Some(w) = obs_anchors {
            let moved = crate::obs_anchor::substitute(&mut m, w).unwrap_or_else(|e| {
                out.fail(format!("error: {e}"));
            });
            crate::obs_anchor::report(&out, &moved, &m);
        }
        m
    };

    // ── The design axis: `--sweep PARAM=GRID` / `--sweep-file FILE` ─────────
    //
    // Resolved once, against the model as the run reads it, into the design
    // points — `(point_id, name-sorted (param, value) list)`, the id being the
    // point's position for `--sweep` and the file's own `point_id` column (else
    // the row) for `--sweep-file` — and the resolved design the ensemble record
    // carries. A swept name must be a declared parameter, named once; a
    // `--param` on a swept parameter is refused here because `--param`
    // outranks every tier, so the sweep would not vary that parameter (the
    // gh#571 shape). A scenario that sets or scales a swept parameter is
    // refused by the engine's coordinate guard (gh#572), on the run and on the
    // dry run alike.
    let design: Option<(Vec<(usize, crate::sweep::DesignPoint)>, crate::sweep::ResolvedDesign)> =
        if !a.sweep.is_empty() {
            crate::sweep::check_swept_names(&a.sweep, &model_to).unwrap_or_else(|e| {
                out.fail(format!("error: {e}"));
            });
            Some((
                crate::sweep::expand_sweep_specs(&a.sweep).into_iter().enumerate().collect(),
                crate::sweep::design_from_specs(&a.sweep),
            ))
        } else if let Some(file) = a.sweep_file.as_deref() {
            let f = crate::sweep::read_sweep_file(file, &model_to).unwrap_or_else(|e| {
                out.fail(format!("error: {e}"));
            });
            Some((f.points, f.design))
        } else {
            None
        };
    if let Some((points, _)) = &design {
        let swept: std::collections::BTreeSet<&str> =
            points.iter().flat_map(|(_, pt)| pt.iter().map(|(k, _)| k.as_str())).collect();
        for p in &a.model_overrides.param {
            if swept.contains(p.name.as_str()) {
                out.fail(format!(
                    "error: parameter `{0}` is both swept and pinned by `--param {0}={1}`.\n  \
                     A `--param` outranks every other tier, so the sweep would not vary `{0}`: \
                     every design point would run at {1} while its label and `sweep:{0}` \
                     column said otherwise.\n  \
                     Fix: drop `--param {0}=…`, or remove `{0}` from the sweep.",
                    p.name, p.value
                ));
            }
        }
        // `--param-vec PREFIX=FILE` is `--param`'s bulk form and shares its
        // tier (`util::resolve_run_parameters_typed`), so a row that names a
        // swept parameter would override the sweep the same way, silently.
        // Each file is read here, before anything runs, with the engine's own
        // reader; a file it cannot read fails now with that reader's message.
        for (prefix, file) in &set_vec_entries {
            let rows = util::load_keyed_tsv(file).unwrap_or_else(|e| {
                out.fail(format!("error: --param-vec {prefix}: {e}"));
            });
            for (key, value) in rows {
                let name = util::param_vec_name(prefix, &key);
                if swept.contains(name.as_str()) {
                    out.fail(format!(
                        "error: parameter `{name}` is both swept and pinned by \
                         `--param-vec {prefix}={file}` (row `{key}` = {value}).\n  \
                         A `--param-vec` row shares `--param`'s rank above every other tier, so \
                         the sweep would not vary `{name}`: every design point would run at \
                         {value} while its label and `sweep:{name}` column said otherwise.\n  \
                         Fix: drop the `{key}` row from {file} (or the `--param-vec {prefix}=…` \
                         flag), or remove `{name}` from the sweep."
                    ));
                }
            }
        }
    }

    let to_was_anchored = to_is_anchored;
    let t_end_override: Option<f64> = match a.to.as_deref() {
        None => None,
        Some(raw) => {
            let spec = crate::fit::runner::parse_time_spec(
                "--to", raw, model_to.origin.as_deref(), &model_to.time_unit,
            ).unwrap_or_else(|e| { out.fail(format!("error: {e}")); });
            let resolved = match spec {
                crate::fit::runner::TimeSpec::Absolute(v) => v,
                // The observed window was folded once above, for the model
                // and `--to` together, so this arm only applies the offset.
                crate::fit::runner::TimeSpec::Anchored(anchored) => {
                    let w = obs_anchors.expect("an anchored --to forced the fold above");
                    w.at(anchored)
                }
            };
            // NO existing validator checks horizon ordering (ir::validate never
            // reads t_end); an inverted horizon would otherwise be a silent
            // header-only TSV.
            if resolved <= model_to.simulation.t_start {
                out.fail(format!(
                    "error: --to \"{raw}\" resolves to t = {resolved}, which is at or \
                     before the model's t_start = {}. The horizon must lie after \
                     the simulation start.",
                    model_to.simulation.t_start
                ));
            }
            // Conflict rule (gh#561: never silently discard a declared
            // horizon): a scenario whose composed horizon differs from BOTH
            // the model's t_end and the resolved --to is refused. Equal to
            // the resolved --to is the no-op precedent and allowed.
            for sname in scenario_list.iter().flatten() {
                let h = crate::params_resolver::effective_horizon(&model_to, Some(sname))
                    .unwrap_or_else(|e| { out.fail(format!("error: {}", e)); });
                if h != model_to.simulation.t_end && h != resolved {
                    out.fail(format!(
                        "error: scenario '{sname}' declares its own horizon \
                         (t = {h}) and --to \"{raw}\" resolves to t = {resolved} — \
                         refusing to pick one silently.\n  Fix: drop the \
                         scenario's `simulate {{ to }}` (keep it as a label-only \
                         preset) or run it without --to."
                    ));
                }
            }
            out.say(format_args!("simulate: --to \"{raw}\" → t_end = {resolved}"));
            Some(resolved)
        }
    };
    // ── `--init-state`: which ensemble of forecast origins ──────────────────
    //
    // Parsed HERE, ahead of the `--fit requires --draws` ergonomics check
    // below, so a user who writes `--init-state fit` without the posterior it
    // pairs against is told which flag completes the pairing rather than the
    // generic thing.
    let init_state_arg = InitStateSourceArg::parse(a.init_state.as_deref());
    if matches!(init_state_arg, Some(InitStateSourceArg::Fit))
        && draws_path.as_deref() != Some("posterior")
    {
        out.fail(
            "error: --init-state fit needs the posterior it pairs against: add \
             --draws posterior --fit <fit results dir>.\n  \
             The fit source restores draw i's OWN terminal latent state X_i(T) \
             under draw i's OWN θ_i — without the posterior draws there is no θ_i \
             to put a state with, and crossing states with unrelated parameters is \
             the incoherent product this source exists to avoid.\n  Fix: add \
             --draws posterior, or pass a `camdl pfilter --save-final-state` file \
             to forecast at a single θ."
        );
    }

    // `--fit` without `--draws` is only meaningful when something needs the
    // fit's DATA rather than its posterior: an anchored `--to` (gh#626) or an
    // anchored model (gh#616). Otherwise keep the old ergonomics.
    // A config's `[simulate] from` is also a use of the fit's config without
    // its posterior.
    let start_from_config = start_override.as_ref().is_some_and(|s| s.source != "--from");
    if a.fit.is_some() && a.draws.is_none() && !to_was_anchored && !model_is_anchored
        && !start_from_config
    {
        out.fail("error: --fit requires --draws (or an anchored --to, an anchored model, \
                  or a config with [simulate] from).");
    }


    // ── A params file's fit `[provenance]` block ────────────────────────────
    //
    // A params file a fit wrote records the backend and step the fit ran on.
    // The run replays on them by the rule a replay of a stored fit follows
    // (`fit::fitted_model::replay_dynamics`, applied below). Only the first
    // block found is read: one file can be a fit's estimate and the others
    // plain overrides.
    let mut from_fit_hash: Option<String> = None;
    let mut from_fit_params_file: Option<String> = None;
    let mut fit_provenance: Option<(String, crate::fit::provenance::MleProvenance)> = None;
    for pf in &params_files {
        // A malformed block is refused, not treated as no block: skipping it
        // would silently drop the backend match, the tamper check and the
        // fit lineage the user wrote the block to get.
        let prov = match crate::fit::provenance::read_mle_provenance(pf) {
            Ok(Some(p)) => p,
            Ok(None) => continue,
            Err(e) => {
                out.fail(format!(
                    "error: cannot read the [provenance] block of {pf}: {e}\n  \
                     Re-run the fit to rewrite the block, or remove the [provenance] block \
                     from {pf} to run the file as plain params (no backend or dt is matched \
                     from it, and no fit lineage is recorded)."
                ));
            }
        };
        // The block's `content_hash` is a tamper hash over the parameter
        // values the fit wrote. Values edited since no longer describe the
        // fit's estimate while the block still claims they do — it would
        // still match the backend and record the fit as this run's upstream —
        // so an edited file is refused, not simulated.
        match crate::fit::provenance::verify_content_hash(pf, &prov.content_hash) {
            Ok(crate::fit::provenance::ContentVerification::Valid) => {}
            Ok(crate::fit::provenance::ContentVerification::Modified { declared, computed }) => {
                out.fail(format!(
                    "error: params file {pf} was edited after the fit that produced it.\n  \
                     Its [provenance] block declares content_hash = \"{declared}\", but the \
                     parameter values in the file now hash to \"{computed}\".\n  \
                     To simulate the fit's estimate, re-run the fit and use the \
                     mle_params.toml it writes.\n  \
                     To run the edited values knowingly, remove the [provenance] block from \
                     {pf}: the file is then a plain params file (no backend or dt is matched \
                     from it, and no fit lineage is recorded)."
                ));
            }
            Err(e) => {
                out.fail(format!("error: {e}"));
            }
        }
        from_fit_hash = prov.fit_hash.clone();
        from_fit_params_file = Some(pf.clone());
        // The fit's simulator, which the run replays on (below).
        fit_provenance = Some((pf.clone(), prov));
        break;
    }

    // ── The fit's simulator ─────────────────────────────────────────────────
    //
    // A replay of a fit's parameters (`--draws posterior|prior|<file> --fit`)
    // runs on the simulator they were estimated on — the fit's backend, dt and
    // ODE integrator (`fit::fitted_model::stored_fit_simulator`, or
    // `fit_config_simulator` for a fit config) — as it runs on the fitted
    // model. A `--backend` / `--dt` / `--integrator` that differs runs, with
    // one notice. Resolved here, before anything reads `backend` / `dt`, so
    // the run, its store key and the `--init-state` checks all see one answer.
    //
    // `--chains` / `--exclude-chains` select chains of a posterior cloud, so
    // they need `--draws posterior`; parsed once, here.
    let posterior_selection = crate::chain_selection::ChainSelection::from_flags(
        a.chains.as_deref(),
        a.exclude_chains.as_deref(),
    )
    .unwrap_or_else(|e| {
        out.fail(format!("error: {e}"));
    });
    if let Some(sel) = &posterior_selection {
        if draws_path.as_deref() != Some("posterior") {
            out.fail(format!(
                "error: {} selects chains of a fit's posterior cloud, so it needs \
                 --draws posterior --fit <handle>",
                sel.flag()
            ));
        }
    }

    // `--draws posterior` names a run; its handle is resolved once, here.
    let posterior_fit_ref: Option<String> = match (draws_path.as_deref(), fit_path_for_draws.as_ref()) {
        (Some("posterior"), Some(fit_handle)) => {
            // gh#896: the draws are one run's. `--fit` is a fit handle
            // (`@label`, a leaf or problem-folder directory, a run-id prefix,
            // a fit.toml) resolved to the run it names; a folder holding
            // several runs with posterior draws is refused with them listed
            // rather than one picked by directory order; runs without draws
            // (an optimizer beside a sampler) are no rival. A bare directory
            // holding a draws.tsv is taken as named.
            // A stored run is tried first: its folder holds a draws.tsv too.
            let stored = crate::fit_store::opened_run(fit_handle);
            // A folder inside a store is never a bare directory of draws.
            let bare = crate::fit_store::store_holding(std::path::Path::new(fit_handle)).is_none()
                && std::path::Path::new(fit_handle).join("draws.tsv").is_file();
            Some(if stored.is_err() && bare {
                fit_handle.clone()
            } else {
                let run = stored.unwrap_or_else(|e| {
                    out.fail(format!("error: --fit {fit_handle}: {e}"));
                });
                let dir = run.run_dir().expect("a run handle has its files");
                if !dir.join("draws.tsv").is_file() {
                    out.fail(format!("error: --fit {fit_handle}: the run holds no posterior draws \
                        (draws.tsv); a sampler (pgas, pmmh, mh) writes them"));
                }
                fit_handle.clone()
            })
        }
        _ => None,
    };
    // The stored fit that run belongs to, resolved once: its simulator below
    // and its intervention selection further down both read it. A fit folder
    // whose stored config cannot be read is refused: the fit is there, and
    // neither its simulator nor its selection is what runs without it. Any
    // other failure is a bare directory holding a `draws.tsv` outside any fit,
    // which each reader says.
    let posterior_fit: Option<Result<std::sync::Arc<crate::fit_store::OpenedFit>, String>> =
        posterior_fit_ref.as_deref().map(|run| {
            crate::fit_store::opened(run).map_err(|_| format!("{run} is a directory of draws outside any stored fit"))
        });
    // The directory the posterior's files are read from: the copy of the run's
    // files, or the bare directory of draws as named.
    let posterior_dir: Option<String> = match (posterior_fit_ref.as_ref(), posterior_fit.as_ref()) {
        (Some(_), Some(Ok(run))) => run.run_dir().map(|d| d.to_string_lossy().into_owned()),
        (Some(named), _) => Some(named.clone()),
        _ => None,
    };
    // The simulator a stored fit ran on, read from the model it ran on.
    let stored_simulator = |r: &std::sync::Arc<crate::fit_store::OpenedFit>| {
        crate::fit::fitted_model::stored_fit_model(r, compile)
            .and_then(|m| crate::fit::fitted_model::stored_fit_simulator(r, &m))
    };
    let fitted_simulator: Option<(String, crate::fit::fitted_model::FittedSimulator)> =
        match (draws_path.as_deref(), fit_path_for_draws.as_ref()) {
            (Some("posterior"), Some(fit_handle)) => {
                // A bare directory holding a `draws.tsv` outside any fit has no
                // fit record, so no simulator to replay on (said in the arm).
                posterior_fit
                    .as_ref()
                    .and_then(|r| r.as_ref().ok())
                    .map(stored_simulator)
                    .transpose()
                    .unwrap_or_else(|e| {
                        out.fail(format!("error: --fit {fit_handle}: {e}"));
                    })
                    .map(|s| (fit_handle.clone(), s))
            }
            (Some(source), Some(fit_ref)) if source != "uniform" => {
                let sim = if std::path::Path::new(fit_ref).is_file() {
                    util::load_ir(&ir_path_compiled).and_then(|(m, _)| {
                        crate::fit::fitted_model::fit_config_simulator(fit_ref, &m)
                    })
                } else {
                    crate::fit_store::opened(fit_ref).and_then(|r| stored_simulator(&r))
                };
                Some((
                    fit_ref.clone(),
                    sim.unwrap_or_else(|e| {
                        out.fail(format!("error: --fit {fit_ref}: {e}"));
                    }),
                ))
            }
            _ => None,
        };
    // The files the `--fit` bound to the model's external inputs: a stored
    // fit's own copies (so a replay reads the bytes it was fitted under), or a
    // fit config's `[data.tables]` for draws taken from a config. `--table`
    // replaces any of them by name, as on every command that reads a fit.
    let mut table_files = table_files;
    if let (Some(source), Some(fit_ref)) = (draws_path.as_deref(), fit_path_for_draws.as_ref()) {
        if source != "uniform" {
            let handle = posterior_fit_ref.as_deref().unwrap_or(fit_ref);
            let fit_tables = fit_table_files(handle).unwrap_or_else(|e| {
                out.fail(format!("error: --fit {fit_ref}: {e}"));
            });
            for (name, path) in fit_tables {
                table_files.entry(name).or_insert(path);
            }
        }
    }
    let mut integrator = a.backend.integrator;
    // The evaluator settings: the flags, or, on a replay of a fit, the fit's
    // `[config]` settings with the flags folded in (gh#1235). Read into the
    // resolved configuration below, which keys them and derives what runs.
    let mut eval_settings = crate::fit::fitted_model::FittedSettings {
        allow_degenerate_rates: a.allow_degenerate_rates,
        allow_forcing_extrapolation: a.forcing_extrapolation.allow_forcing_extrapolation,
    };
    let simulator_flags = crate::fit::fitted_model::SimulatorFlags {
        backend: a.backend.backend,
        dt: a.backend.dt,
        integrator: a.backend.integrator,
        settings: eval_settings,
    };
    if let Some((fit_name, fitted)) = &fitted_simulator {
        // A fit config with no `[method]` names no backend: the params file's
        // fit, else the default, is the one run.
        let fallback = fit_provenance.as_ref().map_or(backend, |(_, prov)| prov.backend);
        let replay = crate::fit::fitted_model::replay_simulator(fit_name, fitted, simulator_flags, fallback)
            .unwrap_or_else(|e| {
                out.fail(format!("error: {e}"));
            });
        if let Some(notice) = &replay.notice {
            out.say(format_args!("{notice}"));
        }
        backend = replay.backend;
        dt = replay.dt;
        integrator = replay.integrator;
        eval_settings = replay.settings;
    } else if let Some((pf, prov)) = &fit_provenance {
        // The block records no `[config]` settings, so the flags' are the
        // run's.
        let model = util::load_ir(&ir_path_compiled).map(|(m, _)| m).unwrap_or_else(|e| {
            out.fail(format!("error: {e}"));
        });
        let fitted = crate::fit::fitted_model::FittedDynamics::of_provenance(prov, &model);
        let replay = crate::fit::fitted_model::replay_dynamics(pf, &fitted, simulator_flags, backend)
            .unwrap_or_else(|e| {
                out.fail(format!("error: {e}"));
            });
        let flagged = a.backend.backend.is_some() || a.backend.dt.is_some() || a.backend.integrator.is_some();
        if !flagged {
            out.say(format_args!(
                "[info] backend auto-matched to {} (dt={}) from fit provenance in {pf}: the \
                 simulator the fit ran on. A --backend, --dt or --integrator that differs runs \
                 with a note.",
                replay.backend, replay.dt
            ));
        }
        if let Some(notice) = &replay.notice {
            out.say(format_args!("{notice}"));
        }
        backend = replay.backend;
        dt = replay.dt;
        integrator = replay.integrator;
    } else if let Err(e) =
        crate::fit::fitted_model::refuse_integrator_off_ode(integrator, backend, "this simulation")
    {
        out.fail(format!("error: {e}"));
    }

    let mut base_sim_run = util::SimRun {
        // The already-compiled IR: every engine cell loads this directly, so
        // `resolve_ir_path` short-circuits (no per-cell camdlc).
        ir_path: ir_path_compiled.clone(),
        params_files,
        overrides,
        overrides_label: None,
        // Draw/sweep + inline-scenario tiers are assigned per cell by
        // `engine::build_cell_sim_run`; the base run carries none.
        point_overrides: std::collections::HashMap::new(),
        // Set below only by `--draws prior --fit` (the config's [fixed]).
        fit_fixed: indexmap::IndexMap::new(),
        set_vec_entries,
        table_files,
        scenario_name: None, // set per-scenario in the loop
        t_end_override, // gh#626: keys the CAS identity below; cells get it via the job
        t_start_override, // `--from` / `[simulate] from`, moved into each cell's model
        // `simulate` records the model's own output schedule; only `camdl predict`
        // has forecast-row boundaries to add.
        required_output_times: Vec::new(),
        // gh#641: assigned PER CELL by `engine::build_cell_sim_run` (replicate i
        // restores particle row i); the base run carries none.
        init_state: None,
        obs_anchors,    // gh#616: the run's resolved observed window
        adhoc_enable,
        adhoc_disable,
        scenario_inline_name: None,
        scenario_inline_set: Vec::new(),
        scenario_inline_scale: Vec::new(),
        // Set below only by `--draws posterior --fit` (the fit's own
        // scenario / enable / disable).
        fit_base: util::FitBase::default(),
        backend,
        dt,
        seed, // overridden per-replicate below
        integrator, // gh#166: --integrator, checked against the replayed fit above
        // `simulate` draws with BTPE, including `--draws posterior --fit`.
        binomial: sim::rng::BinomialAlgorithm::Btpe,
        // Replaced, like `backend` and `dt`, from the resolved configuration's
        // engine settings once it exists.
        eval: sim::EvalSettings::default(),
    };

    // ── `--init-state`: the forecast-origin ensemble ────────────────────────
    //
    // Resolved once, up front, like `--to`: every cell shares one ensemble of
    // origin states and the time they sit at, and a grid index picks the row
    // (`InitStateRowAxis` — replicate for a file, draw for a fit). The readers
    // own the structural checks (compartments by name, the origin time); the
    // checks HERE are the ones that need the rest of the invocation to be
    // known.
    //
    // The fit source (gh#697) is resolved LATER, in the `--draws posterior`
    // arm below: its ensemble and its θ cloud are the same join, so building
    // them together is what makes draw i's state and draw i's parameters
    // impossible to mis-pair. This block does its model-level refusals now, so
    // a user hears about a reactive policy or the wrong backend before waiting
    // on a fit read.
    let mut init_state_source: Option<std::sync::Arc<crate::sim_job::InitStateSource>> = None;
    if init_state_arg.is_some() {
        let (model_is, _) = util::load_ir(&ir_path_compiled).unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
        // The seam refuses a reactive model too (`chain_binomial.rs`), but by
        // then the user has waited for a compile; name the policy here.
        let reactive: Vec<&str> = model_is.interventions.iter()
            .filter(|iv| iv.fire.is_reactive())
            .map(|iv| iv.name.as_str())
            .collect();
        if !reactive.is_empty() {
            out.fail(format!(
                "error: --init-state cannot restart a model with reactive \
                 intervention(s) [{}]. A reactive policy carries mid-run state \
                 the origin state does not hold — the observation history its \
                 trigger reads, its once/cooldown gating, the queue of effects \
                 already scheduled, and its own surveillance RNG stream — so the \
                 forecast would silently begin with an empty agenda.\n  Fix: \
                 run the scenario without the reactive policy, or simulate \
                 continuously from the model's own t_start.",
                reactive.join(", ")
            ));
        }
        if !matches!(backend, crate::args::types::ForwardBackend::ChainBinomial) {
            out.fail(format!("error: {}", util::unseamed_backend_msg(backend.as_str())));
        }

        if let Some(InitStateSourceArg::File(path)) = init_state_arg.as_ref() {
            let columns = io::trajectories::TrajColumnSpec::from_model(&model_is, &[]);
            let bytes = std::fs::read(path).unwrap_or_else(|e| {
                out.fail(format!("error: cannot read --init-state {}: {e}", path.display()));
            });
            let states = io::read_final_states(path, &columns).unwrap_or_else(|e| {
                out.fail(format!("error: --init-state: {e}"));
            });
            // `--save-final-state` is p(x_T | y_{1:T}) at ONE θ. Pairing those
            // rows with unrelated posterior draws would form an incoherent
            // (θ, x_T) product and read as a legitimate forecast cloud.
            if draws_path.is_some() {
                out.fail(
                    "error: --init-state <file> cannot be combined with --draws. The \
                     saved particle states are the filtering distribution at the ONE θ \
                     the filter ran at, so pairing row i with an unrelated posterior \
                     draw would forecast a state and a parameter vector that never went \
                     together.\n  Fix: run --init-state <file> at that same θ \
                     (--params), or use `--init-state fit --draws posterior --fit <fit \
                     results dir>`, which restores draw i's own state under draw i's \
                     own θ."
                );
            }
            // Replicate i restores row i. Not "the first N rows": a
            // post-resampling swarm is ancestor-ordered, so a prefix is not an
            // exchangeable subsample of the filtering distribution.
            let want = if seeds_spec_given { seeds.len() } else { replicates };
            if want != states.len() {
                out.fail(format!(
                    "error: --init-state {} holds {} particle rows but this run has {} \
                     replicate(s). Each replicate restores its own row, and a prefix of \
                     a post-resampling swarm is not an exchangeable subsample of the \
                     filtering distribution — so the counts must match.\n  Fix: pass \
                     --replicates {}, or re-run the filter with --particles {}.",
                    path.display(), states.len(), want, states.len(), want
                ));
            }
            out.say(format_args!(
                "simulate: --init-state {} → {} particle states at t = {}",
                path.display(), states.len(), states.origin_t
            ));
            let digest = store::Key::digest_bytes(&bytes);
            init_state_source = Some(std::sync::Arc::new(crate::sim_job::InitStateSource {
                origin_t: states.origin_t,
                states: states.counts.into_iter()
                    .map(|counts| crate::sim_job::OriginState { counts, reals: Vec::new() })
                    .collect(),
                axis: crate::sim_job::InitStateRowAxis::Replicate,
                ensemble_digest: digest,
            }));
        }
    }

    // ── Pre-flight: validate obs model availability ─────────────────────────
    // We need the model to check observation blocks, but we don't want to
    // run simulation twice. Do a dry load to validate, then run in the loop.
    if want_obs {
        // gh#616 follow-up: use the ANCHOR-SUBSTITUTED model, not a fresh load of
        // the compiled IR. A fresh load still carries the unresolved-horizon
        // marker, so every horizon this block reads was NaN for an anchored
        // model — and the differing-horizons refusal below then fired on a
        // single scenario (`baseline -> t = NaN`), or on a model with no
        // `scenarios {}` block at all, blocking `--obs` for every anchored
        // model. The substitution happened above; this block must see it.
        let model_check = &model_to;
        if model_check.observations.is_empty() {
            out.fail("error: --obs/--obs-dir requested but model has no observations blocks");
        }
        // gh#829: both the mirror file and the store's per-leaf `obs/` subtree
        // draw every stream with no data columns, so a stream whose likelihood
        // reads one is refused here — before a cell runs, so nothing is written.
        // `--design-from` is not this writer: it draws only the streams the fit
        // binds, writes a ratio stream's `n` from the model, and runs this same
        // check over exactly those streams (`design_from_bound_streams`).
        if a.design_from.is_none() {
            for o in &model_check.observations {
                if let Err(e) = crate::obs_emit::check_data_columns_supplied(
                    o, crate::obs_emit::ColumnSupply::Nothing,
                ) {
                    out.fail(format!("error: {e}"));
                }
            }
        }
        // gh#561: the combined --obs/--obs-dir writers cache one obs-time axis
        // for the whole grid (`obs_times_cache`, filled at run_idx == 0) and
        // the wide writer hard-codes one row count per cell — so scenarios with
        // DIFFERENT effective horizons cannot share them. Whichever scenario
        // ran first would set every cell's axis: a shorter sibling would then
        // emit rows past its own trajectory (fabricated, read off the clamped
        // final snapshot), a longer one would be truncated — and which of the
        // two happened would depend on flag order. The CAS obs/ subtree is
        // per-cell and unaffected; refuse the combined mirrors and point at it.
        let mut horizons: Vec<(String, f64)> = Vec::new();
        for s in &scenario_list {
            // gh#626: the cells RUN at the overridden horizon, so the shared
            // obs axis must be validated at it — not the raw model's.
            let h = match t_end_override {
                Some(v) => v,
                None => crate::params_resolver::effective_horizon(model_check, s.as_deref())
                    .unwrap_or_else(|e| {
                        out.fail(format!("error: {}", e));
                    }),
            };
            horizons.push((
                s.clone().unwrap_or_else(|| crate::sim_job::IMPLICIT_ARM_LABEL.into()),
                h,
            ));
        }
        if horizons.iter().any(|(_, h)| *h != horizons[0].1) {
            let list: Vec<String> =
                horizons.iter().map(|(n, h)| format!("{n} → t = {h}")).collect();
            out.fail(format!(
                "error: --obs/--obs-dir cannot combine scenarios with different \
                 horizons ({}): the combined file has one time axis, so the \
                 shorter scenario's rows past its own trajectory would be \
                 fabricated.\n  Fix: run each scenario in its own invocation, or \
                 read the per-cell obs/ artifacts from the store (`camdl cat`).",
                list.join(", ")
            ));
        }
        // gh#626: an `at [...]` emit schedule cannot grow with the horizon —
        // an extending --to would run longer and emit nothing past the listed
        // times, exit 0. Refuse: the whole point of the extension is new rows.
        if let Some(new_end) = t_end_override {
            if new_end > model_check.simulation.t_end {
                for o in &model_check.observations {
                    if let Some(ir::observation::ObservationSchedule::AtTimes(ts)) =
                        &o.emit_schedule
                    {
                        let max_t = ts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        if max_t < new_end {
                            out.fail(format!(
                                "error: --to extends the horizon to t = {new_end}, but \
                                 stream '{}' emits at a fixed list ending at t = {max_t} \
                                 — no synthetic observations would be emitted past it.\n  \
                                 Fix: use a recurring emit schedule (`every …`), extend \
                                 the `at [...]` list, or run without --to.",
                                o.name
                            ));
                        }
                    }
                }
            }
        }

        // gh#1081: a restart from a saved-state file inside a stream's reporting
        // period cannot write that period's row. The emitters refuse it where
        // they plan their rows; refusing here as well, before any cell runs,
        // saves the ensemble and leaves no stored leaf whose observations were
        // never written. (`--init-state fit` resolves its origin later, with
        // the draws, and is refused by the emitters alone.)
        if want_obs {
            if let Some(src) = init_state_source.as_ref() {
                for o in model_check.observations.iter().filter(|o| o.emit_schedule.is_some()) {
                    if let Err(e) = obs_emit_schedule_times(
                        o,
                        Some(crate::obs_emit::Restart {
                            origin: src.origin_t,
                            // `model_check` has its anchors substituted.
                            model_start: model_check.simulation.t_start,
                        }),
                        horizons[0].1, emit_every.as_ref(),
                    ) {
                        out.fail(format!("error: {}", e));
                    }
                }
            }
        }

        // Validate schedule compatibility for --obs (single file), at the
        // CELLS' shared horizon (all equal — checked above), not the base
        // model's: two `at`-list schedules can agree when confined to the model
        // horizon and diverge over an extended one, which would mislabel one
        // stream's late draws with the other's times.
        if obs_path.is_some() && model_check.observations.len() > 1 {
            let obs_end = horizons[0].1;
            // gh#641: and from the cells' shared ORIGIN when `--init-state`
            // restarts them. Two `at`-list schedules can agree over the model
            // window and diverge over the forecast one at either end. `None`
            // for every ordinary run, which keeps this check exactly as it was.
            let obs_origin = init_state_source.as_ref().map(|s| crate::obs_emit::Restart {
                origin: s.origin_t,
                model_start: model_check.simulation.t_start,
            });
            let schedules: Vec<_> = model_check.observations.iter()
                .map(|o| obs_emit_schedule_times(
                    o, obs_origin, obs_end, emit_every.as_ref(),
                ).unwrap_or_else(|e| {
                    out.fail(format!("error: {}", e));
                }))
                .collect();
            let all_same = schedules.windows(2).all(|w| w[0] == w[1]);
            if !all_same {
                // The EFFECTIVE schedules (post `--emit-every`), not the
                // declared ones — a diagnostic naming cadences the run does not
                // use would send the reader to the model file for a mismatch
                // the flag created.
                let descs: Vec<String> = model_check.observations.iter()
                    .zip(&schedules)
                    .map(|(o, times)| format!(
                        "{}: {:?} ({} emit times)", o.name, o.emit_schedule, times.len()))
                    .collect();
                out.fail(format!("error: observation streams have different schedules ({}).\n\
                           A single wide TSV cannot hold multi-cadence streams.\n\
                           Use --obs-dir (one file per stream, keeps trajectory) or\n\
                           --obs-only-dir (one file per stream, suppresses trajectory).",
                    descs.join(", ")));
            }
        }
    }

    // ── Prepare obs-dir output directory ────────────────────────────────────
    if let Some(ref dir) = obs_dir {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| {
            out.fail(format!("error: cannot create obs directory '{}': {}", dir, e));
        });
    }

    // Fit → sim lineage: a params file carrying a fit's `[provenance]` block
    // is recorded, with its content digest, on every cell's record. It is not
    // hashed: the parameter values it gave are.
    let mut lineage: Vec<store::Lineage> = Vec::new();
    if let (Some(_), Some(pf)) = (&from_fit_hash, &from_fit_params_file) {
        if let Ok(bytes) = std::fs::read(pf) {
            lineage.push(store::Lineage::Input { field: "params".into(), digest: store::Key::digest_bytes(&bytes) });
        }
    }

    // ── Load draws if --draws is specified ─────────────────────────────────
    let draws: Vec<HashMap<String, f64>> = if let Some(ref source) = draws_path {
        if source == "uniform" {
            let n = n_draws_arg.unwrap_or_else(|| {
                out.fail("error: --draws uniform requires -n N");
            });
            generate_uniform_draws(&out, &ir_path_compiled, n, seed).unwrap_or_else(|e| {
                out.fail(format!("error: {}", e));
            })
        } else if source == "prior" {
            let n = n_draws_arg.unwrap_or_else(|| {
                out.fail("error: --draws prior requires -n N");
            });
            match fit_path_for_draws.as_ref() {
                Some(fit_path) => {
                    report_replay_check(&out, check_replay_model(fit_path, &ir_path_given, &ir_path, compile));
                    // fit.toml prior source (overrides or supplements
                    // model priors). gh#86: the model IR is the
                    // tier-2 fallback when the fit toml omits a prior
                    // for a parameter that declares `~ <dist>` in the
                    // model file.
                    let (draws_model, _) = util::load_ir(&ir_path_compiled).unwrap_or_else(|e| {
                        out.fail(format!("error loading model for --draws prior: {}", e));
                    });
                    let (rows, fixed) = generate_prior_draws(&out, fit_path, n, seed, &draws_model)
                        .unwrap_or_else(|e| {
                            out.fail(format!("error: {}", e));
                        });
                    // gh#949: the config's [fixed] block resolves at its own
                    // tier (below --params and the scenario), not in the rows.
                    base_sim_run.fit_fixed = fixed;
                    base_sim_run.fit_base = fit_base_for_draws(&out, fit_path, "prior")
                        .unwrap_or_else(|e| {
                            out.fail(format!("error: --fit {fit_path}: {e}"));
                        });
                    rows
                }
                None => {
                    // Priors embedded in the model IR. The rows carry only the
                    // sampled parameters; every other parameter resolves per
                    // cell, against that cell's own scenario (gh#949).
                    generate_prior_draws_from_ir(&out, &ir_path_compiled, n, seed).unwrap_or_else(|e| {
                        out.fail(format!("error: {}", e));
                    })
                }
            }
        } else if source == "posterior" {
            // Resolve the fit run's post-warm-up draws cloud (its draws.tsv).
            // --fit names a stored run here, not a config TOML.
            let fit_handle = fit_path_for_draws.as_ref().unwrap_or_else(|| {
                out.fail("error: --draws posterior requires --fit <run>: the stored fit run's \
                    @label (`fit run --label`), its id or its folder");
            });
            // The run the handle names, resolved with the fit's simulator above,
            // and the directory its files are read from.
            let fit_ref: &String = posterior_fit_ref.as_ref().expect("resolved with --fit");
            let run_dir: &String = posterior_dir.as_ref().expect("resolved with --fit");
            // The posterior is the fitted model's, so every arm replays on the
            // fit's own intervention selection (its `scenario` / `enable` /
            // `disable`): the implicit arm is the model that was fitted, and a
            // `--scenario` / `--enable` / `--disable` applies on top of it.
            match posterior_fit.as_ref().expect("resolved with --fit") {
                Ok(resolved) => {
                    // The draws are this fit's model's: refuse a different
                    // model before anything runs.
                    report_replay_check(&out, util::load_ir(&ir_path_given).and_then(|(given, _)| {
                        crate::fit::fitted_model::check_stored_fit(
                            fit_handle,
                            resolved,
                            &resolved.config.problem,
                            &given,
                            &ir_path,
                        )
                    }));
                    base_sim_run.fit_base = util::FitBase::of(&resolved.config.problem)
                        .unwrap_or_else(|e| {
                            out.fail(format!("error: --fit {fit_handle}: {e}"));
                        });
                    if !base_sim_run.fit_base.is_empty() {
                        out.say(format_args!(
                            "draws: posterior — every arm runs the fit's own {}; a \
                             scenario applies its changes on top of it",
                            base_sim_run.fit_base.describe()
                        ));
                    }
                }
                // A bare directory holding a `draws.tsv` outside any fit
                // carries no fit config, so there is no selection to apply.
                // Said rather than assumed, since a fitted selection may be
                // what is missing.
                Err(e) => out.say(format_args!(
                    "draws: {fit_ref} is not a stored fit run ({e}); no fit \
                     scenario/enable/disable is applied, the model runs as written, \
                     on {} at dt = {}",
                    backend.as_str(),
                    dt
                )),
            }
            if matches!(init_state_arg, Some(InitStateSourceArg::Fit)) {
                let (rows, source) = resolve_paired_posterior(&out,
                    run_dir, &ir_path_compiled, a.n_draws, posterior_selection.as_ref(),
                );
                init_state_source = Some(std::sync::Arc::new(source));
                rows
            } else {
            let resolved = posterior_draws::resolve_posterior_draws(run_dir, None)
                .unwrap_or_else(|e| {
                    out.fail(format!("error: {}", e));
                });
            // The `--chains` / `--exclude-chains` selection is applied through
            // the draws authority, the one seam every reader of a cloud shares.
            if let Some(sel) = &posterior_selection {
                warn_posterior_selection(&out, sel, &resolved.draws_path);
            }
            let resolved = resolved.with_selection(posterior_selection.clone());
            let loaded = resolved.load_params_with_info()
                .map(|(rows, _)| rows)
                .unwrap_or_else(|e| {
                    out.fail(format!("error loading posterior draws {}: {}",
                        resolved.draws_path.display(), e));
                });
            let method_label = resolved.method
                .map(|m| m.as_str())
                .unwrap_or("posterior");
            // gh#630 (ebola F35): never silently replay the full cloud — a
            // 60k-draw posterior is hours of forward solves and a 21 MB
            // --draws-out. Same strided default cap as `camdl predict`
            // (never front-biased), raised or lowered with --n-draws.
            let cap = a.n_draws.unwrap_or(crate::fit::predict::DEFAULT_PREDICT_DRAWS);
            let total = loaded.len();
            let loaded: Vec<HashMap<String, f64>> = if total > cap {
                let picked: Vec<HashMap<String, f64>> =
                    crate::fit::predict::subsample_draws(&loaded, cap)
                        .into_iter().cloned().collect();
                out.say(format_args!(
                    "draws: posterior — subsampling {} of {total} draws (strided \
                     across the cloud; raise with --n-draws)",
                    picked.len()
                ));
                picked
            } else {
                loaded
            };
            out.say(format_args!("draws: posterior — {} draws from {} method '{}' ({})",
                loaded.len(), method_label, resolved.name, crate::fit::report::display_path(&resolved.draws_path)));
            loaded
            }
        } else {
            // File path. #273: when --fit is supplied, the fit's [fixed] block
            // supplies every parameter the draws columns lack (a raw posterior
            // trace tail carries only the estimated columns). It resolves at
            // its own tier (2, below --params and the scenario), as on the
            // `--draws prior --fit` path, not in the rows: a row is tier 3.5
            // and would discard --params (run spec §1.3). A column the file
            // carries is a row value and outranks it.
            let loaded = load_draws_tsv(source).unwrap_or_else(|e| {
                out.fail(format!("error loading draws: {}", e));
            });
            if let Some(fit_ref) = fit_path_for_draws.as_ref() {
                report_replay_check(&out, check_replay_model(fit_ref, &ir_path_given, &ir_path, compile));
                let (fixed_model, _) = util::load_ir(&ir_path_compiled).unwrap_or_else(|e| {
                    out.fail(format!("error: {e}"));
                });
                let fixed = posterior_draws::resolve_fixed_for_backfill(fit_ref, &fixed_model)
                    .unwrap_or_else(|e| {
                        out.fail(format!("error: --fit [fixed]: {}", e));
                    });
                let lacking: Vec<&str> = fixed.iter()
                    .map(|(name, _)| name.as_str())
                    .filter(|name| loaded.iter().any(|row| !row.contains_key(*name)))
                    .collect();
                if !lacking.is_empty() {
                    out.say(format_args!("draws: {} fixed parameter(s) the draws file lacks take the \
                               fit's [fixed] value from {} (below --params): {}",
                        lacking.len(), fit_ref, lacking.join(", ")));
                }
                base_sim_run.fit_fixed = fixed.into_iter().collect();
                base_sim_run.fit_base = fit_base_for_draws(&out, fit_ref, "file")
                    .unwrap_or_else(|e| {
                        out.fail(format!("error: --fit {fit_ref}: {e}"));
                    });
            }
            loaded
        }
    } else {
        // No draws — single point (parameters come from --params / --param)
        vec![HashMap::new()]
    };
    let n_draws = draws.len();

    let n_scenarios = scenario_list.len();
    // The parameter-point axis is EITHER the design (sweep points) OR the
    // sample (draws); clap refuses both flags together.
    let n_points = match &design {
        Some((points, _)) => points.len(),
        None => n_draws,
    };
    let total_runs = n_points * replicates * n_scenarios;
    if total_runs > 1 {
        let parts: Vec<String> = [
            match &design {
                Some(_) if n_points > 1 => Some(format!("{} design points", n_points)),
                Some(_) => None,
                None if n_draws > 1 => Some(format!("{} draws", n_draws)),
                None => None,
            },
            if n_scenarios > 1 { Some(format!("{} scenarios", n_scenarios)) } else { None },
            if replicates > 1 { Some(format!("{} replicates", replicates)) } else { None },
        ].iter().flatten().cloned().collect();
        out.say(format_args!("{} = {} runs", parts.join(" × "), total_runs));
    }

    // ── Build the SimulateJob and route through the unified engine ──────────
    //
    // Built before the dry-run branch so a dry run refuses what the run would
    // (a scenario touching a `--draws <file>` column, gh#572).
    //
    // `simulate` runs through `engine::run_job` (run-spec
    // §3.1). The wide-format trajectory + combined-obs output shape lives
    // in `StreamSink`; the engine owns the cell loop, seed arithmetic, and
    // per-cell SimRun construction (determinism PIN: tests/determinism_pin.rs).
    //
    // Scenario mapping (mirrors the pre-unification per-cell SimRun):
    //   - `--scenario a,b`  → [Named("a"), Named("b")]  (preset path).
    //   - no `--scenario`   → a single ad-hoc Inline carrying the CLI
    //     --enable/--disable (or empty), so the baseline path keeps
    //     `scenario_name = None` exactly as before.
    use crate::sim_job::{ParamSource, ScenarioRef, Seeds, SimulateJob};
    // The ad-hoc arm's label, derived from what the flags resolved to
    // (`sim_job::adhoc_arm_label`), is the name its `ScenarioRef::Inline`
    // carries, and labels its cells in the store, the listings and the
    // combined table. A label is not identity: the arm's identity is the
    // interventions it keeps and the values it runs.
    let adhoc_label = crate::sim_job::adhoc_arm_label(
        &base_sim_run.adhoc_enable,
        &base_sim_run.adhoc_disable,
        &a.model_overrides.param.iter().map(|p| (p.name.clone(), p.value)).collect::<Vec<_>>(),
    );
    let scenarios: Vec<ScenarioRef> = if scenario_names.is_empty() {
        vec![ScenarioRef::Inline {
            name: adhoc_label.clone(),
            enable: base_sim_run.adhoc_enable.clone(),
            disable: base_sim_run.adhoc_disable.clone(),
            params: indexmap::IndexMap::new(),
        }]
    } else {
        scenario_names.iter().map(|n| ScenarioRef::Named(n.clone())).collect()
    };

    // ParamSource: --sweep / --sweep-file yields the design points; --draws
    // yields Draws rows; otherwise a single Point run `replicates` times. With
    // explicit --seeds the seed-list length is the replicate count (the engine
    // ignores the variant's `replicates` then), so passing `replicates` is
    // correct in every case.
    let source = if let Some((points, _)) = &design {
        ParamSource::Sweep {
            points: points
                .iter()
                .map(|(point_id, pt)| crate::sim_job::SweepPoint {
                    point_id: *point_id,
                    overrides: pt.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                })
                .collect(),
            replicates,
        }
    } else if let Some(ref src) = draws_path {
        let rows: Vec<indexmap::IndexMap<String, f64>> = draws.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect();
        // A user-authored draws FILE (not the generated `uniform`/`prior`/
        // `posterior` sources). Carried as `Some(path)` so a scenario ×
        // draws-file column collision is refused naming the file (gh#572);
        // generated draws are not checked (a scenario over them is the
        // intended counterfactual).
        let explicit_file = match src.as_str() {
            "uniform" | "prior" | "posterior" => None,
            _ => Some(std::path::PathBuf::from(src)),
        };
        ParamSource::Draws { rows, replicates, explicit_file }
    } else {
        ParamSource::Point { replicates }
    };

    // Seeds: explicit --seeds list, else single base seed (replicates
    // derive via the XOR mix inside the engine).
    let job_seeds = if seeds_spec_given {
        Seeds::Explicit(seeds.clone())
    } else {
        Seeds::Single(seed)
    };

    // ── The resolved configuration ──────────────────────────────────────────
    //
    // Every precedence rule above has been applied, so the run-wide settings
    // are final. They are resolved once, into `ResolvedSimConfig`, and from
    // here on both the identity of every cell (the CAS sink) and what the
    // engine is told (the job, the evaluator settings) are derived from it.
    let sim_config = std::sync::Arc::new(
        resolve_sim_config(
            &base_sim_run,
            &ir_path,
            &scenario_names,
            SimConfigFlags {
                backend,
                dt,
                allow_degenerate_rates: eval_settings.allow_degenerate_rates,
                allow_forcing_extrapolation: eval_settings.allow_forcing_extrapolation,
                view: output_cols,
            },
        )
        .unwrap_or_else(|e| {
            out.fail(format!("error preparing CAS: {}", e));
        }),
    );
    let engine_settings = sim_config.engine_settings();
    let _forcing_route = engine_settings.install_forcing_policy();
    // What runs is what is keyed: from here on the backend and `dt` are read
    // from `engine_settings` and nowhere else; the precedence locals they
    // were resolved from are not read again.
    base_sim_run.backend = engine_settings.backend;
    base_sim_run.dt = engine_settings.dt;
    base_sim_run.eval = engine_settings.eval;

    let job = SimulateJob {
        // The pre-compiled IR: each engine cell loads this directly (no
        // per-cell camdlc — see the resolve-once at the top of the function).
        model: ir_path_compiled.clone(),
        params_files: base_sim_run.params_files.clone(),
        backend: engine_settings.backend,
        dt: engine_settings.dt,
        binomial: base_sim_run.binomial,
        eval: engine_settings.eval,
        integrator, // gh#166: --integrator, checked against the replayed fit above
        source,
        scenarios,
        t_end_override,
        t_start_override,
        required_output_times: Vec::new(),
        init_state: init_state_source,
        obs_anchors,
        seeds: job_seeds,
        cli_overrides: base_sim_run.overrides.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        fit_fixed: base_sim_run.fit_fixed.clone(),
        fit_base: base_sim_run.fit_base.clone(),
        set_vec_entries: base_sim_run.set_vec_entries.clone(),
        table_files: base_sim_run.table_files.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        // Output is byte-identical whatever the thread count: a cell's seed is
        // a function of its grid position, and the engine merges cells in
        // canonical order, so the writers see the sequence a serial run
        // produces (run-spec §4.4).
        parallel: execution.threads,
    };

    // Refuse what the run would refuse — a scenario touching a `--draws <file>`
    // column or a sweep (gh#572), or a `--param` it sets or scales (gh#571) —
    // before any branch: `--dry-run` and `--event-log` do not reach
    // `engine::run_job`, which checks again.
    engine::check_scenario_coordinate_collision(&job).unwrap_or_else(|e| {
        out.fail(format!("error: {}", e));
    });

    // ── Pre-flight: every generated-draws cell has a value for every parameter
    //
    // gh#949. Generated `prior` / `uniform` rows carry only the parameters the
    // measure varies; the rest resolve per cell through the ordinary tiers.
    // A parameter that no tier supplies in some arm would otherwise surface
    // mid-grid, after the other arms ran — so every arm is resolved once here
    // and the run is refused before anything is written, naming the arm.
    if let Some(measure) = draws_path.as_deref().and_then(GeneratedMeasure::from_source) {
        if let Err(e) = refuse_unresolvable_generated_cells(
            &job, measure, !scenario_names.is_empty(), fit_path_for_draws.is_some(),
        ) {
            out.fail(format!("error: {}", e));
        }
    }

    // ── Persist sampled draws (gh#157) ──────────────────────────────────────
    // `--draws-out PATH` materializes the sampled θ-per-draw as a TSV the
    // `--draws PATH` loader reads back (one row per draw, one column per
    // parameter the draw source carries). Opt-in only: absent the flag nothing
    // is written, so the content-addressed store leaves are untouched.
    if let Some(ref draws_out) = a.draws_out {
        let draws_out = draws_out.to_string_lossy().into_owned();
        if let Err(e) = write_draws_tsv(&draws_out, &draws) {
            out.fail(format!("error writing --draws-out {}: {}", draws_out, e));
        }
        out.say(format_args!("draws.tsv: wrote {} draws to {}", n_draws, draws_out));
    }

    // ── `--design-from`: simulate on a fit's own observation design ─────────
    //
    // gh#831. The rows come from the bound data rather than from the model's
    // `emit_schedule`, so this leaves the store: what it produces is a
    // dataset written from the resolved inputs, not a stored result.
    if let Some(ref design_toml) = a.design_from {
        if dry_run {
            print_dry_run(&out,
                &ir_path, &ir_path_compiled, engine_settings.backend, engine_settings.dt, seed,
                &base_sim_run.params_files, &base_sim_run.overrides,
                &scenario_list, &adhoc_label, &seeds, &draws_path,
                n_draws, replicates, total_runs,
                &obs_path, &obs_dir, &obs_only,
                design.as_ref().map(|(_, d)| d),
            );
            return;
        }
        if obs_dir.is_none() && a.projected_out.is_none() {
            out.fail(
                "error: --design-from writes one file per observation stream, so it \
                 needs a directory: pass --obs-only-dir DIR (dataset only), \
                 --obs-dir DIR, or --projected-out DIR (each stream's projected value)."
            );
        }
        if let Err(e) = simulate_on_bound_design(&out,
            design_toml, &base_sim_run, &draws, draws_path.is_some(),
            obs_dir.as_ref().map(std::path::Path::new),
            a.projected_out.as_deref(), engine_settings.dt,
        ) {
            out.fail(format!("error: {e}"));
        }
        return;
    }

    // ── Dry run ─────────────────────────────────────────────────────────────
    if dry_run {
        print_dry_run(&out,
            &ir_path, &ir_path_compiled, engine_settings.backend, engine_settings.dt, seed,
            &base_sim_run.params_files, &base_sim_run.overrides,
            &scenario_list, &adhoc_label, &seeds, &draws_path,
            n_draws, replicates, total_runs,
            &obs_path, &obs_dir, &obs_only,
            design.as_ref().map(|(_, d)| d),
        );
        if total_runs <= 1 {
            return;
        }
    }

    // ── Generated quantities (proposal 2026-06-25) ──────────────────────────
    // With `--quantities-out` each cell evaluates the model's `quantities {}`
    // block as a by-product, and the tables are rendered from them. Without
    // it, a model that declares quantities prints a one-line note.
    let quantities_want: Option<crate::sim_command::QuantitiesWant> = {
        let (q_model, _) = util::load_ir(&ir_path_compiled).unwrap_or_else(|e| {
            out.fail(format!("error: {}", e));
        });
        if q_model.quantities.is_empty() {
            None
        } else if let Some(dir) = a.quantities_out.clone() {
            // A sweep point is a design coordinate like a scenario: one
            // replicate per point is one realization per group (point mode),
            // more is a band over the replicates of each group, never a band
            // across points or scenarios (gh#562).
            let mode = if matches!(
                job.source,
                crate::sim_job::ParamSource::Point { replicates: 1 }
                    | crate::sim_job::ParamSource::Sweep { replicates: 1, .. }
            ) {
                crate::quantity_output::Mode::Point
            } else {
                crate::quantity_output::Mode::Banded
            };
            Some(crate::sim_command::QuantitiesWant {
                dir,
                mode,
                scenario_axis: !scenario_names.is_empty(),
                vocabulary: quantities_override.clone(),
            })
        } else {
            if !dry_run {
                let n = q_model.quantities.len();
                out.say(format_args!(
                    "note: model declares {n} quantit{}; pass --quantities-out <dir> to emit them",
                    if n == 1 { "y" } else { "ies" }
                ));
            }
            None
        }
    };

    if a.event_log.is_some() && total_runs != 1 {
        out.fail("error: --event-log is single-run only (no --seeds / \
                   --replicates / --draws / multiple --scenario)");
    }

    crate::sim_command::run(crate::sim_command::StoreRun {
        out: &out,
        job: &job,
        sim: &sim_config,
        engine: &engine_settings,
        model_path: &ir_path,
        ir_compiled: &ir_path_compiled,
        table_files: &base_sim_run.table_files,
        request,
        given,
        execution: execution_record,
        lineage,
        wants: crate::sim_command::Wants {
            obs: want_obs,
            event_log: a.event_log.is_some(),
            quantities: quantities_want,
            emit_every: emit_every.clone(),
        },
        exports: crate::sim_command::Exports {
            output: if suppress_trajectory { None } else { a.output.clone() },
            stdout: a.stdout,
            obs_file: obs_path.as_ref().map(std::path::PathBuf::from),
            obs_dir: obs_dir.as_ref().map(std::path::PathBuf::from),
            dates: a.dates,
            event_log: a.event_log.clone().filter(|p| p.as_os_str() != "auto"),
            event_log_tsv: a.tsv,
            event_log_format: a.format.clone(),
            reactive_log: a.reactive_log.clone(),
        },
        dry_run,
    });
}

/// The files a `--fit` handle binds to the model's external inputs, by name:
/// for a fit config file its `[data.tables]`, for a stored fit the copies its
/// folder holds (`fit::handle` points the recovered config at them). A bare
/// directory of draws outside any fit binds nothing.
fn fit_table_files(fit_ref: &str) -> Result<Vec<(String, String)>, String> {
    let tables = |p: &crate::fit::config_v2::Problem| -> Vec<(String, String)> {
        p.data.iter().flat_map(|d| d.tables.iter().map(|(n, f)| (n.clone(), f.clone()))).collect()
    };
    let path = std::path::Path::new(fit_ref);
    if path.is_file() {
        let problem = crate::fit::config_v2::Problem::load(fit_ref)?;
        return Ok(tables(&problem));
    }
    if path.join("draws.tsv").is_file() {
        return Ok(Vec::new());
    }
    let resolved = crate::fit_store::opened(fit_ref)?;
    Ok(tables(&resolved.config.problem))
}

/// Materialize the per-draw `y_sim` for the streams an `observations.<stream>`
/// quantity reduces. Samples every schedule-bearing stream in DECLARATION order
/// with the cell's `obs_seed` — the same RNG walk `simulate --obs` performs — so
/// the quantity reduces exactly the draws `--obs` would emit (no redraw). A
/// referenced stream that is fit-only (no `emit_schedule`) is a hard error.
#[allow(clippy::too_many_arguments)]
fn materialize_obs_for_quantities(
    referenced: &[&str],
    model: &ir::Model,
    traj: &sim::Trajectory,
    compiled: &std::sync::Arc<sim::CompiledModel>,
    params: &[f64],
    obs_seed: u64,
    // gh#641: `Some(T)` when the cell was restarted from a filtered state at
    // `T`. Threaded rather than defaulted so an obs-sourced quantity reduces
    // over exactly the series `--obs` emits for the same cell.
    restart: Option<crate::obs_emit::Restart>,
    // gh#656: the `--emit-every` override, for the same reason — the quantity
    // must reduce over the series `--obs` would emit under the same flags.
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<sim::quantity::ObsSeriesSet, String> {
    // gh#829: the draws below carry no data columns, so a quantity over a
    // stream whose likelihood reads one would reduce a series of zeros.
    for obs_ir in model.observations.iter().filter(|o| referenced.contains(&o.name.as_str())) {
        crate::obs_emit::check_data_columns_supplied(
            obs_ir, crate::obs_emit::ColumnSupply::Nothing,
        )?;
    }
    let mut obs_rng = sim::rng::StatefulRng::new(obs_seed);
    let mut out: std::collections::HashMap<String, (Vec<f64>, Vec<f64>)> =
        std::collections::HashMap::new();
    for obs_ir in &model.observations {
        let times = match &obs_ir.emit_schedule {
            // The CELL's horizon (`model` here is the cell's resolved model, so
            // a scenario `simulate { to }` has already moved it) — never the
            // schedule's baked `end`, which would reduce an obs-sourced quantity
            // over fabricated tail values (gh#561).
            Some(s) => {
                let s = crate::emit_every::apply_override(emit, obs_ir, s)?;
                if let Some(restart) = restart {
                    crate::obs_emit::refuse_period_under_way(
                        obs_ir, &s, emit.and_then(|e| e.resolve_for(&obs_ir.source)),
                        restart, model.simulation.t_end,
                    )?;
                }
                obs_schedule_times(&s, restart.map(|r| r.origin), model.simulation.t_end)
            }
            None => continue, // fit-only — consumes no RNG, mirroring `--obs`
        };
        let sampler =
            sim::inference::obs_model::compile_obs_sample_pf(obs_ir, compiled.clone(), params)
                .map_err(|e| e.to_string())?;
        // The rows the stream's declaration assigns to these times (gh#833):
        // the same rows `--obs` writes, so a quantity sourced from a stream
        // reduces over exactly the file the stream would emit.
        let plan = crate::obs_emit::plan_emission(
            obs_ir, &times, run_start_of(traj, model), model.simulation.t_end,
            emit.and_then(|e| e.resolve_for(&obs_ir.source)),
        )?;
        let projected = project_coverages(traj, obs_ir, model, &plan.scored(), compiled.eval)?;
        // Each row is drawn where it is scored, and named on the series by
        // its label, as the `--obs` file names it: which time names a row is
        // a separate question (D69), not settled here.
        let times: Vec<f64> = plan.rows.iter().map(|r| r.label).collect();
        let mut vals = Vec::with_capacity(times.len());
        for (ti, row) in plan.rows.iter().enumerate() {
            let snap = snap_at(traj, row.at);
            vals.push(sampler(projected[ti], row.at, &snap.int_state.counts, &[], &mut obs_rng));
        }
        out.insert(obs_ir.name.clone(), (times, vals));
    }
    for &name in referenced {
        if !out.contains_key(name) {
            return Err(if model.observations.iter().any(|o| o.name == name) {
                format!(
                    "quantity reduces observations.{name}, but stream '{name}' is fit-only \
                     (no `emit_schedule`); add `emit_schedule = every N 'unit` to generate \
                     y_sim for the quantity"
                )
            } else {
                format!(
                    "quantity reduces observations.{name}, but no observation stream \
                     named '{name}' is declared"
                )
            });
        }
    }
    Ok(sim::quantity::ObsSeriesSet { streams: out })
}

/// Resolves `simulate`'s configuration once: the base model every cell starts
/// from, and the settings that are the same for every cell. A `--scenario`
/// the model does not declare is refused. The engine's settings
/// (`ResolvedSimConfig::engine_settings`) and the run-wide part of each
/// cell's `config` in its identity (`sim_store::SimIdentity`) are derived
/// from the value this returns.
///
/// The base model is the model as `util::load_run_model` loads it for the
/// engine (observation anchors substituted, `--integrator` applied), with no
/// parameter values applied, so it is the same for every cell. A cell's
/// parameter values (its identity's `params`) and the interventions its
/// scenario keeps (its `scenario`) are resolved per cell, from the cell's
/// own `SimRun`, by `sim_config::ResolvedCell::resolve`.
fn resolve_sim_config(
    run: &util::SimRun,
    // The original source path (e.g. `sir.camdl`) for the human-readable store
    // prefix + provenance. `run.ir_path` is the pre-compiled `.ir.json`, whose
    // basename would slug to `camdl_<pid>`; the display fields must keep the
    // model's real name so leaves land under `sims/sir-<hash>/…`.
    display_path: &str,
    scenario_names: &[String],
    flags: SimConfigFlags,
) -> Result<crate::sim_config::ResolvedSimConfig, String> {
    // A required output time adds a recorded snapshot, and can raise the
    // integration end to reach it — it changes the leaf's bytes. The leaf is
    // keyed off the model as loaded here, not off the schedule
    // `resolve_run_model` hands the backend, so a content-addressed run
    // carrying one would store two different trajectories under one address.
    // No such caller exists — only `camdl predict` sets them, and its replay
    // writes no leaf — so this refuses rather than lets a future one discover
    // it as a stale cache hit.
    if !run.required_output_times.is_empty() {
        return Err(
            "internal: a content-addressed simulate cannot carry required \
             output times — they change the recorded trajectory but not the \
             run identity. Give them an identity route from `ResolvedCell` \
             before routing a CAS-writing path through them."
                .to_string(),
        );
    }
    // The binomial sampler changes every chain-binomial draw and is not in the
    // run identity, for the same reason: only `camdl predict` sets one other
    // than BTPE (D78), and its replay writes no leaf.
    if run.binomial != sim::rng::BinomialAlgorithm::Btpe {
        return Err(format!(
            "internal: a content-addressed simulate cannot draw with the {} binomial \
             sampler — it changes the trajectory but not the run identity. Fold it \
             into the config level before routing a CAS-writing path through it.",
            run.binomial));
    }
    // The model the engine starts every cell from — the one load
    // `resolve_run_model` also takes (`util::load_run_model`), so the model
    // that is HASHED and the model that RUNS cannot diverge (gh#583 item C).
    // Anchor substitution (gh#616) and `--integrator` (audit 2026-08-23 #1)
    // happen there, once, for both. Params are NOT applied: the `params` level
    // is resolved per cell, from the cell's own `SimRun`, by the resolver call
    // the engine makes.
    //
    // NOTE for the deferred `value_at`-under-simulate work (F23): the anchor
    // argument does NOT extend to quantity anchors. `Model::hash_into`
    // deliberately EXCLUDES `quantities` (and `contrasts`) so reporting can
    // never re-key a run — so resolving a `value_at(…, last_obs)` here would
    // change no hashed field, and two vintages would share a `run_id` while
    // reporting different numbers. Whoever lands F23 must key the resolved
    // anchor EXPLICITLY; no model field will do it for them.
    let base_model = util::load_run_model(run)?;

    // A named scenario the model does not declare is refused here, before any
    // cell is planned.
    for name in scenario_names {
        if !base_model.presets.iter().any(|p| &p.name == name) {
            let available: Vec<&str> = base_model.presets.iter().map(|p| p.name.as_str()).collect();
            return Err(format!("scenario '{}' not found. Available: {}", name,
                if available.is_empty() { "(none)".into() } else { available.join(", ") }));
        }
    }

    let SimConfigFlags { backend, dt, allow_degenerate_rates, allow_forcing_extrapolation, view } = flags;
    let run_config = crate::sim_config::RunConfig { backend, dt, allow_degenerate_rates, allow_forcing_extrapolation, view };
    let model = crate::sim_config::ResolvedModel::new(base_model, display_path);
    Ok(crate::sim_config::ResolvedSimConfig { model, run: run_config })
}

/// The run-wide settings `simulate` resolved by its precedence rules (an
/// explicit flag, a fit's provenance, a replayed fit's simulator, the model's
/// own default), handed to [`resolve_sim_config`] as one value.
struct SimConfigFlags {
    backend: args::types::ForwardBackend,
    dt: f64,
    allow_degenerate_rates: bool,
    allow_forcing_extrapolation: bool,
    view: crate::util::OutputColumns,
}

/// The transitions a simulation never fired, as one warning: each by name
/// when there are at most twenty, else the count and how to look.
fn never_fired_notice(names: &[&str]) -> Option<out::notice::Notice> {
    if names.is_empty() {
        return None;
    }
    let message = if names.len() > 20 {
        format!("{} transitions never fired during simulation\n  \
                 (run `camdl inspect MODEL --transitions \"PATTERN*\"` to see their rates)", names.len())
    } else {
        let mut m = format!("{} transition{} never fired during simulation:",
            names.len(), if names.len() == 1 { "" } else { "s" });
        for name in names {
            m.push_str(&format!("\n  {}    (propensity always 0.0)", name));
        }
        m
    };
    Some(out::notice::Notice {
        severity: out::notice::Severity::Warning,
        code: "transitions-never-fired",
        message,
    })
}

/// `diagnostics.tsv` for a simulated cell whose transitions carry
/// diagnostics, with a warning naming the transitions that never fired.
fn write_transition_diagnostics(traj: &sim::Trajectory) {
    match write_diagnostics_tsv("diagnostics.tsv", &traj.transition_diagnostics) {
        Ok(zero_count) => {
            if zero_count > 0 {
                if let Some(n) = never_fired_notice(&never_fired(&traj.transition_diagnostics)) {
                    crate::out::detached().notice(&n);
                }
            }
        }
        Err(e) => crate::out::detached().say(format_args!("warning: could not write diagnostics.tsv: {}", e)),
    }
}

// ── Observation helpers ─────────────────────────────────────��───────────────

/// `camdl simulate --design-from <fit.toml>`: simulate on the observation
/// design that fit config binds, and write the dataset as the files the loader
/// reads (gh#831).
///
/// The design is read through the same seam `fit run` uses
/// (`resolve_and_load_obs_streams`), so each stream's rows here are the rows
/// the fit scores: its observed labels, each row's own period, and every `NA`
/// hole. Nothing is re-derived from a schedule.
///
/// One parameter vector writes its files straight into `dir`; several (from
/// `--draws`) write one `ds_NN/` subdirectory each, so every dataset stays a
/// file set the loader can bind on its own.
fn simulate_on_bound_design(
    out: &crate::out::Out,
    design_toml: &std::path::Path,
    base_run: &util::SimRun,
    draws: &[HashMap<String, f64>],
    draws_given: bool,
    dir: Option<&std::path::Path>,
    projected_dir: Option<&std::path::Path>,
    dt: f64,
) -> Result<(), String> {
    // The parameter vectors this writes a dataset for. Without `--draws` the
    // base run IS the vector (`--params` / `--param`).
    let points: Vec<indexmap::IndexMap<String, f64>> = if draws_given {
        draws.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect()
    } else {
        vec![indexmap::IndexMap::new()]
    };
    if points.is_empty() {
        return Err("--draws produced no parameter vectors, so there is no \
                    dataset to simulate".to_string());
    }

    // Each draw is its own realisation, seeded the way every other draw grid is
    // (`engine::process_seed_for` — the determinism PIN's arithmetic): a lone
    // point keeps `--seed` untouched, and a grid mixes the draw index in, so
    // two draws do not share one process trajectory.
    let run_for = |i: usize, point: &indexmap::IndexMap<String, f64>| util::SimRun {
        point_overrides: point.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        seed: engine::process_seed_for(None, base_run.seed, i, 0, points.len()),
        ..base_run.clone()
    };

    // The design is a property of the model and the bound data, not of θ — but
    // resolving the model needs a complete parameter vector, so it is read off
    // the first point's run and reused for the rest.
    let first = run_for(0, &points[0]);
    let runs: Vec<util::SimRun> = points.iter().enumerate().map(|(i, p)| run_for(i, p)).collect();
    util::emit_run_notices(out, &runs.iter().collect::<Vec<_>>());
    let (compiled, model) = util::resolve_run_model(&first)?;
    let bindings = pfilter::load_data_observations_from_fit_toml(design_toml, &model)?;
    let effective = fit::runner::data_bindings_to_effective(&model, &bindings)?;
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: compiled.model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    let streams = fit::runner::resolve_and_load_obs_streams(
        &model, &compiled, &effective, &time_opts,
    )?;
    // A dataset meant to be read back is checked for the round trip; projections
    // are never bound back as data, so they use the plans alone.
    let design = match dir {
        Some(_) => obs_emit::design_from_bound_streams(&streams, &effective, &time_opts)?,
        None => obs_emit::plans_for_bound_streams(&streams, &effective, &time_opts)?,
    };
    let per_point = |root: &std::path::Path, i: usize| if points.len() == 1 {
        root.to_path_buf()
    } else {
        root.join(fit::config_v2::format_dataset_dir(i + 1))
    };

    for (i, point) in points.iter().enumerate() {
        let run = run_for(i, point);
        if let Some(dir) = dir {
            let written = obs_emit::simulate_dataset(
                &run, obs_emit::ObservationDesign::Bound(&design), &per_point(dir, i),
            )?;
            for f in &written {
                out.say(format_args!("observations written to {}", f.path.display()));
            }
        }
        if let Some(pdir) = projected_dir {
            for path in obs_emit::write_projected_dataset(&run, &design, &per_point(pdir, i))? {
                out.say(format_args!("projections written to {}", path.display()));
            }
        }
    }
    Ok(())
}

/// gh#626: resolve the global observation anchors (`first_obs`, `last_obs`)
/// for an anchored `--to`, from the fit's `[data.observations]` bindings.
/// `--fit` accepts a fit toml, or a stored fit by handle (`@label`, an id
/// prefix or its folder), whose stored training files it reads.
/// Streams load through the same shared seam as
/// pfilter/profile, so dated time columns, long-form families, and holes
/// resolve exactly as at fit time; anchors fold over the training rows the
/// config's `holdout_after` leaves, and rows whose value is `NA` count
/// toward neither anchor (D40).
fn o_source(o: &ir::observation::ObservationModel) -> String { o.source.clone() }

fn resolve_simulate_obs_anchors(
    model: &ir::Model,
    fit_ref: &std::path::Path,
    dt: f64,
) -> Result<(f64, f64), String> {
    obs_anchors_from_config(model, &simulate_fit_problem(fit_ref)?, dt)
}

/// The `[simulate] from` of the config `simulate --fit` names, as written:
/// the fit.toml itself, or the configuration a stored fit ran under
/// (`fit.resolved.toml`), never the verbatim `fit.toml.original` beside it.
///
/// Read from the config text alone, without binding its data: `--fit` serves
/// several purposes (`--draws posterior`, `--init-state fit`, anchors), each of
/// which resolves and checks the fit its own way and reports its own errors. A
/// handle that names no readable config (no fit there, a bare directory of
/// draws, a CLI-only fit) yields no start here, and the purpose that needs the
/// fit refuses it. A fit folder that keeps a config it cannot be read under
/// (another store layout, its resolved config gone) is `Err`: passed over, the
/// run would start where the model says rather than where the fit did.
#[expect(clippy::disallowed_methods, reason = "reads `[simulate] from` of the fit file `--fit` names (declared `#[input(identity)]` on `SimulateArgs`) or of `fit.toml` in the scratch copy `fit_store::opened` makes; the start becomes each cell's window, which its identity holds")]
fn simulate_fit_config_start(fit_ref: &std::path::Path) -> Result<Option<String>, String> {
    let raw = fit_ref.to_string_lossy();
    let text = if crate::fit_store::is_config_file(&raw) {
        match std::fs::read_to_string(fit_ref) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        }
    } else {
        // A stored run's start as its fit file set one: the run keeps that
        // file as given (`fit.toml`). The stored `problem.toml` writes the
        // resolved start, the model's own included, so it cannot say whether
        // the file set one; for a problem, which keeps no fit file, it is the
        // start the problem's compile records.
        match crate::fit_store::opened(&raw) {
            Ok(o) => match o.run_dir().map(|d| d.join(crate::fit_store::FIT_TOML_FILE)) {
                Some(given) if given.is_file() => std::fs::read_to_string(&given)
                    .map_err(|e| format!("cannot read {}: {e}", given.display()))?,
                Some(_) => return Ok(None),
                None => crate::camdl_store::read_stored(&o.fit.problem, crate::fit_store::PROBLEM_FILE)
                    .and_then(|b| String::from_utf8(b).map_err(|e| e.to_string()))?,
            },
            // A stored fit that cannot be read is refused, saying why; a bare
            // directory of draws outside any store sets no start.
            Err(e) if raw.starts_with('@') || crate::fit_store::store_holding(fit_ref).is_some() => return Err(e),
            Err(_) => return Ok(None),
        }
    };
    let Ok(doc) = text.parse::<toml::Table>() else { return Ok(None) };
    Ok(match doc.get("simulate").and_then(|s| s.get("from")) {
        Some(toml::Value::String(s)) => Some(s.clone()),
        Some(toml::Value::Integer(i)) => Some(i.to_string()),
        Some(toml::Value::Float(f)) => Some(f.to_string()),
        _ => None,
    })
}

/// The problem `simulate --fit` names. A bare fit.toml needs NO completed run;
/// run dirs / @labels / hashes resolve through the store (a stored fit's
/// config, whose `[simulate] from` is the start that fit was fitted from).
fn simulate_fit_problem(
    fit_ref: &std::path::Path,
) -> Result<crate::fit::config_v2::Problem, String> {
    Ok(match crate::fit_store::is_config_file(&fit_ref.to_string_lossy()) {
        true => {
            crate::fit::config_v2::Problem::load(&fit_ref.to_string_lossy())
                .map_err(|e| format!("failed to load fit toml '{}': {e}", fit_ref.display()))?
        }
        _ => crate::fit_store::opened(&fit_ref.to_string_lossy())?.config.problem.clone(),
    })
}

/// The fold over a config already in hand. Split out so `camdl predict` — which
/// has resolved its config long before it could look up a `--fit` ref —
/// resolves anchors through the SAME reader as `simulate`, rather than growing a
/// second one that could disagree about which rows count as observation times.
pub(crate) fn obs_anchors_from_config(
    model: &ir::Model,
    config: &crate::fit::config_v2::Problem,
    dt: f64,
) -> Result<(f64, f64), String> {
    let data = config.data_spec().map_err(|e| format!(
        "the fit config has no [data] block to read observed times from: {e}"))?;
    let bindings = crate::fit::stream_bindings::StreamBindings::resolve(model, data)
        .map_err(|e| format!("resolving [data]: {e}"))?;
    let effective_pairs = bindings.training_by_source();
    if effective_pairs.is_empty() {
        return Err("the fit config's [data] resolves to zero observation \
                    streams — nothing to anchor last_obs/first_obs to.".into());
    }
    let bound: Vec<(String, std::path::PathBuf)> = effective_pairs.iter()
        .map(|(k, v)| (k.clone(), std::path::PathBuf::from(v)))
        .collect();
    obs_anchors_from_bindings(model, &bound, dt,
        config.data.as_ref().and_then(|d| d.holdout_after.as_ref()))
}

/// The fold itself, over `(binding key → file)` pairs a command has ALREADY
/// resolved. `pfilter` and `profile` reach it directly: their bindings
/// come from `--data` flags as often as from a fit toml, and re-deriving the
/// window from a config they may not have would be a second reader that could
/// disagree with the one they score against.
///
/// The anchors are the observed window of the training rows (D40,
/// `docs/dev/temporal-architecture.md` §4.7): `first_obs` the earliest start
/// and `last_obs` the latest end of a row that carries a value, across every
/// bound stream. Under `holdout_after` the training rows are the ones the fit
/// trains on (the cutoff's own anchors resolve over the full files), so rows
/// appended after the cutoff move no anchor; without one every row trains.
pub(crate) fn obs_anchors_from_bindings(
    model: &ir::Model,
    bound: &[(String, std::path::PathBuf)],
    dt: f64,
    holdout_after: Option<&crate::fit::config_v2::TimeSpecToml>,
) -> Result<(f64, f64), String> {
    if bound.is_empty() {
        return Err("no observation stream is bound — nothing to anchor \
                    last_obs/first_obs to.".into());
    }
    let time_opts = crate::caltime_load::TimeOpts {
        origin: model.origin.as_deref(),
        time_unit: &model.time_unit,
        dt,
        t_start: model.simulation.t_start,
        format: crate::caltime_load::TimeFormat::Auto,
    };
    // The rows load through `load_observations` directly (the same reader the
    // shared seam wraps) — no CompiledModel, so a model whose estimated params
    // carry no values still anchors.
    let effective = crate::fit::runner::data_bindings_to_effective(model, bound)?;
    let mut rows: std::collections::BTreeMap<String, Vec<crate::fit::observed_rows::ObservedRow>> =
        std::collections::BTreeMap::new();
    for obs_model in model.observations.iter().filter(|o| effective.contains_key(&o.source)) {
        let data_path = &effective[&o_source(obs_model)];
        let siblings: Vec<&ir::observation::ObservationModel> = model.observations.iter()
            .filter(|o| o.source == obs_model.source)
            .collect();
        rows.insert(obs_model.name.clone(), crate::fit::runner::load_observations(
            data_path, obs_model, &siblings, &time_opts,
        )?);
    }
    let training: Vec<&crate::fit::observed_rows::ObservedRow> = match holdout_after {
        Some(spec) => {
            let partition = crate::fit::runner::cutoff_partition(spec, &rows, model)?;
            rows.iter().flat_map(|(s, rs)| &rs[..partition.rows(s)]).collect()
        }
        None => rows.values().flatten().collect(),
    };
    crate::fit::training_partition::observed_window(training).ok_or_else(||
        "the bound observation streams contain no observed row — nothing \
         to anchor last_obs/first_obs to.".to_string())
}

/// Generate observation times from an IR schedule, confined to `t_end`.
///
/// `restart_origin` is `Some(T)` **only** when the run was restarted from a
/// filtered state at `T` (`simulate --init-state`, gh#641); then times before
/// `T` are dropped, because the run has no trajectory there at all. `None` — the
/// default, and every path that does not restart — leaves the lower end alone
/// and is byte-identical to not having this parameter.
///
/// It is deliberately an `Option<f64>` rather than the run's `t_start`. Those
/// two look interchangeable and are not: dropping times below a plain `t_start`
/// would convert the gh#589 fail-closed guard into silent truncation for a model
/// whose author listed `emit_schedule = at [...]` times outside its own window.
/// The guard is right to reject that — the author declared observations the run
/// cannot produce — and only a restart has a reason to prefer dropping, since
/// there the pre-origin times are historical and the data already covers them.
///
/// The cadence is always anchored to the schedule's own `start`, never re-based
/// on the origin: a restarted run still emits on the declared grid, just from
/// its first in-window time onward.
pub(crate) fn obs_schedule_times(
    schedule: &ir::observation::ObservationSchedule,
    restart_origin: Option<f64>,
    t_end: f64,
) -> Vec<f64> {
    let after_origin =
        |t: f64| restart_origin.is_none_or(|origin| t >= origin - OBS_SNAP_EPS);
    // The horizon test `sim::output::output_times` applies to the recording
    // axis, so both axes agree on whether the terminal point is in (gh#606).
    let in_horizon = |t: f64| t <= t_end + sim::schedule::output_tol(t_end);
    match schedule {
        ir::observation::ObservationSchedule::Regular(reg) => {
            let mut times = Vec::new();
            let mut t = reg.start;
            while in_horizon(t) {
                if after_origin(t) {
                    times.push(t);
                }
                t += reg.step;
            }
            times
        }
        ir::observation::ObservationSchedule::AtTimes(times) => {
            times.iter().copied().filter(|&t| in_horizon(t) && after_origin(t)).collect()
        }
    }
}

/// Emission times for `simulate --obs` on one stream, confined to `t_end` and —
/// only for a restarted run — to `restart_origin`.
/// `emit_schedule` is the SIMULATE-only cadence (proposal §2.5); a model that
/// only ever fits omits it and so cannot generate synthetic data — a hard error
/// naming the fix, not a silent empty series.
///
/// **`restart_origin` is `Some` only under `--init-state`** (gh#641). A forecast
/// begins at the filtered state's origin, so an emit time before it has no
/// snapshot to project from; those times are historical and the observed data
/// already covers them. See [`obs_schedule_times`] for why this is an `Option`
/// and not simply the run's `t_start`.
///
/// **`t_end` is the RUN's horizon, not the schedule's baked `end`** (gh#561).
/// The expander bakes `ObsRegular.end` from the MODEL-level `simulate { to }` at
/// compile time (`expander.ml:7692`), so it is a copy of the model horizon and
/// not an author-declared observation end. Once a scenario can move the window,
/// trusting the baked value emits observations past the end of the run's own
/// trajectory — and every reader clamps (`snap_at` returns the last snapshot
/// ≤ t, and the cumulative-flow reader freezes), so the surplus rows are
/// FABRICATED: zeros for an incidence stream, and for a prevalence stream a
/// frozen compartment dressed in fresh observation noise, which reads as a
/// perfectly plausible plateau. Confining to the run's horizon — exactly what
/// `sim::output::output_times` does for the trajectory axis — makes the
/// observation axis honour `simulation.t_end` as the sole horizon authority too
/// (gh#143). Byte-identical whenever the run uses the model horizon, since the
/// baked `end` is then the same number.
///
/// **`emit` is the `--emit-every` override** (gh#656), applied HERE rather than
/// by rewriting the compiled IR: `emit_schedule` never enters the likelihood, so
/// moving the model hash for it would re-key a fit against real data over a
/// change that fit cannot see. `None` for every run without the flag, which
/// leaves the emitted times, and every artifact address derived from them,
/// exactly as they were.
pub(crate) fn obs_emit_schedule_times(
    obs: &ir::observation::ObservationModel,
    restart: Option<crate::obs_emit::Restart>,
    t_end: f64,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<Vec<f64>, String> {
    match &obs.emit_schedule {
        Some(s) => {
            let s = crate::emit_every::apply_override(emit, obs, s)?;
            if let Some(restart) = restart {
                crate::obs_emit::refuse_period_under_way(
                    obs, &s, emit.and_then(|e| e.resolve_for(&obs.source)), restart, t_end,
                )?;
            }
            Ok(obs_schedule_times(&s, restart.map(|r| r.origin), t_end))
        }
        None => Err(format!(
            "observation stream '{}' has no `emit_schedule` — it is fit-only \
             and cannot generate synthetic data. Add `emit_schedule = every N 'unit` \
             (or `at [...] 'unit`) to the block to `simulate --obs`.",
            obs.name
        )),
    }
}

/// Project observable quantities from a trajectory at all observation times.
///
/// For CumulativeFlow: accumulate per-snapshot flows, difference between
/// consecutive observation times to get per-interval flow counts.
/// For CurrentPop/CurrentPopSum: read state at snapshot closest to each obs time.
/// Tolerance for "this observation time IS this recorded snapshot".
///
/// Absolute, and deliberately the single authority: the guard, `snap_at`, and
/// the incidence walk must agree, or the guard validates one relationship while
/// the projection resolves a different one (gh#589 review). Absolute rather
/// than relative because that is what the resolvers have always used; changing
/// the resolvers' behaviour is a separate question from making them consistent.
pub(crate) const OBS_SNAP_EPS: f64 = 1e-9;

/// The recorded snapshot a projection time resolves to — the last snapshot at
/// or before `t`, under [`OBS_SNAP_EPS`]. `None` when `t` precedes every
/// recorded snapshot.
///
/// The single predicate: the on-grid guards below, the cumulative-flow walk,
/// and the conditioning-boundary read all resolve through this one function, so
/// a guard can never validate a relationship under a predicate the projection
/// then resolves under a different one (gh#589 review).
fn resolved_snapshot_index(traj: &sim::Trajectory, t: f64) -> Option<usize> {
    traj.snapshots.iter().rposition(|s| s.t <= t + OBS_SNAP_EPS)
}

/// Whether `t` IS a recorded snapshot time (not merely resolvable to an earlier
/// one). See [`resolved_snapshot_index`].
fn is_recorded_snapshot(traj: &sim::Trajectory, t: f64) -> bool {
    match resolved_snapshot_index(traj, t) {
        Some(i) => (traj.snapshots[i].t - t).abs() <= OBS_SNAP_EPS,
        None => false,
    }
}

/// Every observation time must land on a recorded snapshot.
///
/// The projection below reads the trajectory, not integrator state, so an
/// observation time that falls BETWEEN snapshots silently reads the earlier one:
/// a flow collapses its whole interval onto the snapshot boundary (six zeros
/// then a lump) and a stock becomes a step function. The emitted file still
/// carries the requested timestamps, so the corruption is invisible — and
/// `--obs` output is normally fitted (incident
/// `docs/dev/incidents/2026-08-12-obs-quantized-to-output-grid.md`, gh#589).
///
/// The condition is MISALIGNMENT, not coarseness: an `emit_schedule` at t = 3.5
/// against output every 1 snaps exactly as badly as daily-against-weekly.
///
/// This is a guard, not the fix. The fix is to sample from integrator state at
/// observation times, which removes the coupling entirely.
fn check_obs_times_on_snapshot_grid(
    traj: &sim::Trajectory,
    stream: &str,
    time_unit: &str,
    obs_times: &[f64],
) -> Result<(), String> {
    let snaps: Vec<f64> = traj.snapshots.iter().map(|s| s.t).collect();
    if snaps.is_empty() {
        return Err(format!(
            "observation stream '{}' cannot be emitted: the run recorded no \
             trajectory snapshots.\n  \
             Check the output schedule — an `output {{ trajectories {{ at = [...] }} }}` \
             block with no time inside [t_start, t_end] records nothing.",
            stream
        ));
    }
    // The guard must use the SAME predicate the projection resolves with, not a
    // parallel one. An earlier version compared with a RELATIVE tolerance while
    // `snap_at` and `incidence_over` resolve with an ABSOLUTE `OBS_SNAP_EPS`;
    // those cross over at t > 1000, beyond which the guard accepted times the
    // projection then snapped away from — reinstating the silent-wrong this
    // guard exists to prevent, one layer up. Validating a relationship under one
    // predicate and resolving it under another is the defect, so there is now
    // one predicate (`is_recorded_snapshot`).
    let off_grid = obs_times.iter().find(|&&t| !is_recorded_snapshot(traj, t));
    let Some(&bad) = off_grid else {
        return Ok(());
    };
    let before = snaps.iter().rev().find(|&&s| s < bad);
    let after = snaps.iter().find(|&&s| s > bad);
    // The guidance deliberately states the INVARIANT rather than naming one fix.
    // Observation times reach this check from two unrelated places — a model's
    // `emit_schedule` (simulate/synthetic) and a loaded data file's time
    // column (`camdl predict`) — and advice that fits one is wrong for the other.
    // Telling a `camdl predict` user to "change the emit schedule" is nonsense:
    // there isn't one, and they cannot rewrite their data's dates.
    Err(format!(
        "observation stream '{}' needs a value at t = {bad}, which is not a \
         recorded output time (nearest recorded: {} and {}).\n  \
         Observations are projected from the recorded trajectory, so a time \
         between snapshots would read the earlier snapshot — a flow would \
         report its whole interval on one boundary and zeros elsewhere, and a \
         stock would step. The output would still carry the requested \
         timestamps, so the error is invisible in the file (gh#589).\n  \
         Required: every observation time must also be an output time.\n  \
         Note the output schedule defaults to every 1.0 {}, independent of \
         `dt` — so sub-unit observation times need an explicit \
         `output {{ trajectories {{ every = ... }} }}`.\n  \
         If these times come from an `emit_schedule`, make it a multiple of \
         the output cadence (or widen the output schedule to match). If they \
         come from a data file — `camdl predict` projects at the observed times, \
         not an `emit_schedule` — the output schedule must be fine enough to \
         include them. If the run's horizon comes from a scenario's \
         `simulate {{ to }}` or a `--to` override extending past an \
         `at = [...]` output list (gh#561, gh#626), the recording grid does \
         not extend with it — add the extended window's times to the `at` \
         list, or drop the per-scenario `to` / the `--to`.",
        stream,
        before.map(|v| v.to_string()).unwrap_or_else(|| "(none)".into()),
        after.map(|v| v.to_string()).unwrap_or_else(|| "(none)".into()),
        time_unit,
    ))
}

/// The time the run began — `t_start`, or the restart origin of an
/// `--init-state` cell — as the trajectory recorded it
/// ([`sim::Trajectory::run_start`]). Not the first row's time: an `at = [...]`
/// schedule without `t_start` records no initial row.
pub(crate) fn run_start_of(traj: &sim::Trajectory, model: &ir::Model) -> f64 {
    traj.run_start().unwrap_or(model.simulation.t_start)
}

/// Project a stream's observed quantity for each row — the state at an
/// instant, or the flow over `[start, stop)` — off the recorded trajectory.
/// One walk serves every caller: the synthetic emitters (rows planned from the
/// stream's declaration), `camdl predict` (rows from the bound data) and the
/// obs-sourced quantities (gh#833).
///
/// Both endpoints of an interval row must be recorded snapshots. Between
/// snapshots the cumulative flow would resolve to an earlier one and silently
/// hand part of the interval to the wrong row — the class of silent-wrong
/// gh#702 was about (incident 2026-08-12, gh#589).
///
/// `eval` is the evaluator setting of the run that produced `traj`: a
/// `DerivedExpr` projection compiles `model` under it, so a projection is
/// evaluated as the run's rates were.
pub(crate) fn project_coverages(
    traj: &sim::Trajectory,
    obs_ir: &ir::observation::ObservationModel,
    model: &ir::Model,
    rows: &[(f64, score::trace::Coverage)],
    eval: sim::EvalSettings,
) -> Result<Vec<f64>, String> {
    let obs_times: Vec<f64> = rows.iter().map(|r| r.0).collect();
    check_obs_times_on_snapshot_grid(traj, &obs_ir.name, &model.time_unit, &obs_times)?;
    // Per-interval incidence over a set of transition flow indices — shared by
    // CumulativeFlow (one exact transition), CumulativeFlowSum (explicit strata
    // family per §25.4) and each side of a FlowRatio.
    let incidence_over = |flow_indices: &[usize]| -> Result<Vec<f64>, String> {
        incidence_over_rows(traj, &obs_ir.name, flow_indices, rows)
    };
    match &obs_ir.projection {
        ir::observation::Projection::CumulativeFlow(flow_name) => {
            let flow_indices: Vec<usize> = model.transitions.iter().enumerate()
                .filter(|(_, tr)| tr.name == *flow_name)
                .map(|(i, _)| i)
                .collect();
            incidence_over(&flow_indices)
        }
        ir::observation::Projection::CumulativeFlowSum(flow_names) => {
            let flow_indices: Vec<usize> = flow_names.iter()
                .filter_map(|fname| model.transitions.iter()
                    .position(|tr| tr.name == *fname))
                .collect();
            incidence_over(&flow_indices)
        }
        ir::observation::Projection::CurrentPop(comp_name) => {
            let loc = resolve_comp_local(model, &obs_ir.name, comp_name);
            Ok(obs_times.iter().map(|&obs_t| {
                let snap = snap_at(traj, obs_t);
                read_comp(snap, &loc)
            }).collect())
        }
        ir::observation::Projection::CurrentPopSum(names) => {
            let locs: Vec<_> = names.iter()
                .map(|name| resolve_comp_local(model, &obs_ir.name, name))
                .collect();
            Ok(obs_times.iter().map(|&obs_t| {
                let snap = snap_at(traj, obs_t);
                locs.iter().map(|loc| read_comp(snap, loc)).sum()
            }).collect())
        }
        ir::observation::Projection::DerivedExpr(_) => {
            // Delegated to the shared `StreamProjection` evaluator in
            // `sim::inference::multi_stream_obs`. Same primitive the
            // scoring path uses, so forward simulation and likelihood
            // scoring agree on DerivedExpr semantics by construction.
            use sim::inference::multi_stream_obs::{
                StreamProjection, eval_stream_projection,
            };
            use sim::state::RealState;
            let compiled = sim::CompiledModel::new(model.clone())
                .unwrap_or_else(|e| {
                    crate::out::detached().fail(format!("error: DerivedExpr projection — model compile: {:?}", e));
                })
                .with_eval_settings(eval);
            let stream_proj = StreamProjection::from_ir(
                &obs_ir.projection, &compiled, &obs_ir.name,
            ).unwrap_or_else(|e| {
                crate::out::detached().fail(format!("error: DerivedExpr projection — resolve: {}", e));
            });
            let real_s = RealState::new(compiled.real_local_to_global.len());
            let params = compiled.default_params.clone();
            // FlowSum is never produced by DerivedExpr, but pass an
            // empty slice so the helper's signature is uniform.
            let empty_flows: &[u64] = &[];
            Ok(obs_times.iter().map(|&obs_t| {
                let snap = snap_at(traj, obs_t);
                eval_stream_projection(
                    &stream_proj, empty_flows, &snap.int_state.counts,
                    &params, &compiled, &real_s, obs_t,
                )
            }).collect())
        }
        ir::observation::Projection::FlowRatio { numerator, denominator } => {
            // The two flow sums over the row's window — each the incidence
            // quantity above — divided (proposal 2026-09-09). Emission and
            // scoring agree by construction: the runtime reads the same two
            // bins at the row's close and takes the same quotient, NaN when
            // the denominator flow is zero.
            use sim::inference::multi_stream_obs::flow_ratio;
            let indices = |names: &[String]| -> Vec<usize> {
                names.iter()
                    .filter_map(|fname| model.transitions.iter()
                        .position(|tr| tr.name == *fname))
                    .collect()
            };
            let num = incidence_over(&indices(numerator))?;
            let den = incidence_over(&indices(denominator))?;
            Ok(num.iter().zip(&den).map(|(&a, &b)| flow_ratio(a, b)).collect())
        }
    }
}

/// The flow over each row's `[start, stop)` for a set of transition flow
/// indices, off the recorded trajectory: build the running cumulative flow at
/// each snapshot, then read it at each row's two boundaries. The one
/// arithmetic every incidence quantity is read by — a `CumulativeFlow`, a
/// `CumulativeFlowSum`, each side of a `FlowRatio`, and the denominator the
/// design-preserving emitter writes for a ratio stream whose `n` is a data
/// column (proposal 2026-09-09) — so none of them can disagree about what a
/// row's flow is.
///
/// Both endpoints of a row must be recorded snapshots (the gh#589 guard).
pub(crate) fn incidence_over_rows(
    traj: &sim::Trajectory,
    obs_name: &str,
    flow_indices: &[usize],
    rows: &[(f64, score::trace::Coverage)],
) -> Result<Vec<f64>, String> {
    use score::trace::Coverage;
    // `f64` throughout: ODE flows are real-valued (the chain-binomial /
    // Gillespie integer flows widen losslessly via `Flows::value`).
    // `cum_at_snap[i]` is the flow over (run start, snapshots[i].t]: the
    // trajectory's initial row carries zeroed flows by construction
    // (`sim::state::Trajectory`), so the running sum needs no offset.
    let mut cum_at_snap: Vec<(f64, f64)> = Vec::with_capacity(traj.snapshots.len());
    let mut running = 0.0f64;
    for snap in &traj.snapshots {
        for &fi in flow_indices {
            running += snap.flows.value(fi);
        }
        cum_at_snap.push((snap.t, running));
    }
    // The cumulative flow at a boundary, which must BE a recorded snapshot.
    let cum_at = |t: f64| -> Result<f64, String> {
        let i = resolved_snapshot_index(traj, t)
            .filter(|_| is_recorded_snapshot(traj, t))
            .ok_or_else(|| format!(
                "observation stream '{}': the period boundary t = {t} is not a \
                 recorded output time, so the flow accumulated up to it cannot \
                 be read.\n  \
                 A row's value is the difference of the recorded cumulative \
                 flow at its two boundaries.\n  \
                 Fix: add {t} to the output schedule \
                 (`output {{ trajectories {{ every = ... }} }}`, or an \
                 `at = [...]` list containing it), or move the boundary onto \
                 a recorded output time.",
                obs_name,
            ))?;
        Ok(cum_at_snap[i].1)
    };
    rows.iter().map(|&(label, cov)| match cov {
        Coverage::Interval { start, stop } => Ok(cum_at(stop)? - cum_at(start)?),
        Coverage::Instant | Coverage::Unrecorded => Err(format!(
            "observation stream '{}': an incidence projection needs a period \
             for the row at {label}, but it was given {cov:?}",
            obs_name,
        )),
    }).collect()
}

/// The transition indices of the named flows, in the order named. A name with
/// no transition is dropped, as the projection arms above do; both validators
/// refuse such an IR before a run reaches here.
pub(crate) fn flow_indices_of(model: &ir::Model, names: &[String]) -> Vec<usize> {
    names.iter()
        .filter_map(|fname| model.transitions.iter().position(|tr| tr.name == *fname))
        .collect()
}

/// Resolved compartment location: integer (local index) or real (local index).
enum CompLoc { Int(usize), Real(usize) }

fn resolve_comp_local(model: &ir::Model, obs_name: &str, comp_name: &str) -> CompLoc {
    let mut int_idx = 0usize;
    let mut real_idx = 0usize;
    for c in &model.compartments {
        if c.name == comp_name {
            return match c.kind {
                ir::model::CompartmentKind::Integer => CompLoc::Int(int_idx),
                ir::model::CompartmentKind::Real => CompLoc::Real(real_idx),
            };
        }
        match c.kind {
            ir::model::CompartmentKind::Integer => int_idx += 1,
            ir::model::CompartmentKind::Real => real_idx += 1,
        }
    }
    crate::out::detached().fail(format!("error: observation '{}' projects compartment '{}' which doesn't exist",
        obs_name, comp_name));
}

pub(crate) fn snap_at(traj: &sim::Trajectory, obs_t: f64) -> &sim::Snapshot {
    traj.snapshots.iter().rev()
        .find(|s| s.t <= obs_t + OBS_SNAP_EPS)
        .unwrap_or_else(|| {
            crate::out::detached().fail(format!("error: no snapshot at or before t={}", obs_t));
        })
}

fn read_comp(snap: &sim::Snapshot, loc: &CompLoc) -> f64 {
    match loc {
        CompLoc::Int(i) => snap.int_state.counts[*i] as f64,
        CompLoc::Real(i) => snap.real_state.values[*i],
    }
}

/// A generated draw source whose rows carry only the parameters it varies
/// (gh#949), so the remaining parameters must resolve from another tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GeneratedMeasure {
    /// `--draws prior`: rows carry the parameters that have a prior.
    Prior,
    /// `--draws uniform`: rows carry the parameters with declared bounds.
    Uniform,
}

impl GeneratedMeasure {
    fn from_source(source: &str) -> Option<Self> {
        match source {
            "prior" => Some(GeneratedMeasure::Prior),
            "uniform" => Some(GeneratedMeasure::Uniform),
            _ => None,
        }
    }
}

/// Refuse a generated-draws job in which some arm leaves a parameter with no
/// value (gh#949).
///
/// Each scenario is resolved once, at the first draw row, through the same
/// `build_cell_sim_run` + resolver the engine runs every cell with. The rows
/// of one generated source all carry the same columns, so the first row is
/// representative for "is every parameter set". The same pass surfaces any
/// other refusal the resolver would raise for that arm (an unknown scenario, a
/// `--params` value outside its bounds) before a cell runs.
fn refuse_unresolvable_generated_cells(
    job: &crate::sim_job::SimulateJob,
    measure: GeneratedMeasure,
    named_scenarios: bool,
    fit_given: bool,
) -> Result<(), String> {
    use crate::params_resolver::ResolveError;
    use crate::util::RunParamError;
    let points = job.source.param_points();
    let Some(first) = points.first() else { return Ok(()) };
    let table_files: HashMap<String, String> = job.table_files.iter().cloned().collect();
    let mut model: Option<ir::Model> = None;
    let mut refusals: Vec<String> = Vec::new();
    for scenario in &engine::effective_scenarios(job) {
        let run = engine::build_cell_sim_run(
            job, scenario, first, &table_files, job.seeds.base(), 0, 0,
        );
        let base = match &model {
            Some(m) => m,
            None => model.insert(util::load_run_model(&run)?),
        };
        let arm = named_scenarios.then(|| scenario.name().to_string());
        match util::resolve_run_parameters_typed(base, &run) {
            Ok(_) => {}
            Err(RunParamError::Resolve(ResolveError::UnsetRequired { name, .. })) => {
                refusals.push(unset_generated_parameter_message(
                    &name, measure, arm.as_deref(), fit_given,
                ));
            }
            // Already names the scenario.
            Err(e @ RunParamError::Resolve(ResolveError::ScenarioNotFound { .. })) => {
                refusals.push(e.to_string());
            }
            Err(e) => refusals.push(match arm {
                Some(arm) => format!("scenario '{arm}': {e}"),
                None => e.to_string(),
            }),
        }
    }
    if refusals.is_empty() { Ok(()) } else { Err(refusals.join("\n\nerror: ")) }
}

/// The refusal for a parameter a generated draw source does not vary and no
/// tier sets, in one arm (`arm = None` for a run with no `--scenario`).
fn unset_generated_parameter_message(
    name: &str,
    measure: GeneratedMeasure,
    arm: Option<&str>,
    fit_given: bool,
) -> String {
    let (lacks, varies, own_fix) = match measure {
        GeneratedMeasure::Prior => (
            "no prior",
            "--draws prior samples only the parameters that have a prior",
            if fit_given {
                format!("give '{name}' a prior in the fit config's [estimate] or a \
                         value in its [fixed]")
            } else {
                format!("declare a prior on '{name}' in the model's parameters {{}} block \
                         (`{name} : <type> in [lo, hi] ~ <dist>(...)`), supply `--fit FIT.toml`")
            },
        ),
        GeneratedMeasure::Uniform => (
            "no bounds",
            "--draws uniform draws only the parameters with declared bounds",
            format!("declare `in [lo, hi]` on '{name}' in the model"),
        ),
    };
    let (where_, scenario_fix) = match arm {
        Some(arm) => (format!(" for scenario '{arm}'"), format!("set it in scenario '{arm}'")),
        None => (String::new(), "set it in a scenario (`--scenario NAME`)".to_string()),
    };
    // A model never writes a parameter's value (parameters {} declares names,
    // types, bounds and priors), so the sources named here are the ones a run
    // can actually take a value from.
    let fit_source = if fit_given { ", the fit config's [fixed]" } else { "" };
    format!(
        "parameter '{name}' has {lacks}, and nothing gives it a value{where_}.\n  \
         {varies}; every other parameter takes its value from --params FILE{fit_source}, \
         the scenario's `set`, or --param NAME=VALUE.\n  \
         Fix options: {own_fix}, set it with --params FILE or --param {name}=VALUE, \
         or {scenario_fix}."
    )
}

/// Generate N uniform random draws over the model's declared parameter bounds.
///
/// Each row carries only the parameters the measure varies — those with
/// declared bounds (gh#949). Every other parameter is left out of the row and
/// resolves per cell through the ordinary tiers (model default → `--params` →
/// scenario → `--param`); writing it into the row would put its model default
/// at the draw tier, above `--params`. `run_simulate` refuses up front a cell
/// in which such a parameter gets no value from any tier.
fn generate_uniform_draws(
    out: &crate::out::Out,
    ir_path: &str,
    n: usize,
    seed: u64,
) -> Result<Vec<HashMap<String, f64>>, String> {
    let (model, _) = util::load_ir(ir_path)?;
    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_UNIFORM);

    // Declared bounds only, deliberately — NOT
    // `params_resolver::resolved_bounds` (gh#763). This sweep varies what the
    // modeller gave a range; enrolling every `probability` in the sweep
    // because its type carries [0, 1] would silently change what
    // `--draws uniform` means for existing models.
    let bounded: Vec<(&str, f64, f64)> = model.parameters.iter()
        .filter_map(|p| p.bounds().map(|(lo, hi)| (p.name.as_str(), lo, hi)))
        .collect();
    // Nothing to sample: every row would be empty and every cell would run at
    // one parameter vector, which is not a draw from anything.
    if bounded.is_empty() {
        let example = model.parameters.first().map_or("beta", |p| p.name.as_str());
        return Err(format!(
            "--draws uniform: no parameter of the model declares bounds, so there is nothing \
             to sample and every draw would be the same parameter vector.\n  \
             --draws uniform samples `lo + (hi - lo)\u{b7}U`, with U uniform on [0, 1), for \
             each parameter's declared bounds `in [lo, hi]`; a parameter without bounds is \
             not drawn.\n  \
             Fix: declare bounds on each parameter to vary, after its type in the model's \
             parameters {{}} block (e.g. `{example} : <type> in [lo, hi]`), or use \
             --draws prior with `~ <dist>(...)` priors, or pass --draws a file of rows."
        ));
    }

    let mut draws = Vec::with_capacity(n);
    for _ in 0..n {
        let row: HashMap<String, f64> = bounded.iter()
            .map(|&(name, lo, hi)| (name.to_string(), lo + (hi - lo) * rng.uniform()))
            .collect();
        draws.push(row);
    }
    out.say(format_args!("generated {} uniform draws over {} bounded parameter(s); {} other \
               parameter(s) resolve per cell from --params, the scenario's set or --param",
        n, bounded.len(), model.parameters.len() - bounded.len()));
    Ok(draws)
}

/// Generate N draws from declared priors in a fit.toml.
/// Each draw is a complete parameter vector (estimated from priors + fixed).
///
/// gh#86: resolves priors through the three-tier precedence chain
/// (fit_toml > model_ir > flat) shared with `camdl fit run` (gh#75) and
/// `camdl profile`'s prior provenance (gh#73). This is the load-bearing
/// change: previously the fit-toml side was the only consulted source,
/// so a model declaring `~ <dist>` on every parameter would still be
/// rejected if the fit toml's [estimate] blocks omitted `prior = { ... }`.
/// Now the model-IR `~` priors satisfy the requirement.
///
/// `model` is the simulate-flow's already-loaded IR. Threading it through
/// is what enables the IR-tier fallback — without it, this function would
/// be dependent on the config's `[model] camdl = …` path and have to
/// re-load. The caller passes the same model used for simulation, so the
/// priors we sample from match the model that runs.
///
/// Sampling reuses the same `sample_from_prior_raw` helper as
/// `generate_prior_draws_from_ir` for distribution consistency across
/// the two `--draws prior` modes. The deviation from the previous
/// fit-toml-only implementation: the Exponential variant previously
/// used `rand_distr::Exp::new(rate)` (a Ziggurat sampler) while
/// `sample_from_prior_raw` uses inverse-CDF `-ln(U)/rate`. Both are
/// statistically Exp(rate) with E[X] = 1/rate and Var = 1/rate² — only
/// the RNG-byte trajectory changes. Verified by the
/// `sample_from_prior_raw_matches_expected_moments` test, which asserts
/// the moment-matching at N=50k.
/// gh#158: `simulate --fit <toml>` (via `--draws prior`) loads the fit
/// config's problem half, so a bare params file — one with no `[model]`
/// table — fails with a raw serde message (`missing field \`model\``) that
/// does not tell the user what shape the file must have. Append a hint
/// naming the expected `[model]` table and pointing at the docs. The
/// original error is preserved so the underlying cause is still visible.
///
/// Scoped to that one error: any other load failure (a missing data field,
/// a legacy `[stages]` table, a typo'd key) carries its own diagnosis, and
/// the `[model]` hint on top of it points at the one table that is present.
fn wrap_fit_load_error(fit_path: &str, err: String) -> String {
    // Both spellings of "this is not a fit config at the top level": no
    // `[model]` table, or a bare key where the top-level tables go.
    let top_level_shape =
        err.contains("missing field `model`") || err.contains("expected one of `model`");
    if !top_level_shape {
        return err;
    }
    format!(
        "{}\n  \
         hint: `simulate --fit` expects a fit-config TOML, not a bare \
         params file. It must declare a `[model]` table naming the \
         model file, e.g.\n    \
         [model]\n    \
         camdl = \"path.camdl\"\n  \
         See `camdl docs fit-toml` for the full schema. \
         (file: {})",
        err, fit_path)
}

/// `simulate --draws prior|<file> --fit FIT`: is the model on the command line
/// the one FIT's parameters belong to? A fit config (a file) is checked against
/// the model its `[model]` names; any other handle is a stored fit, checked
/// against the model it archived ([`crate::fit::fitted_model`]).
fn check_replay_model(
    fit_ref: &str,
    ir_path_given: &str,
    model_path: &str,
    compile: util::CompileSettings,
) -> Result<Option<String>, String> {
    let (given, _) = util::load_ir(ir_path_given)?;
    if std::path::Path::new(fit_ref).is_file() {
        let problem = fit::config_v2::Problem::load(fit_ref)
            .map_err(|e| wrap_fit_load_error(fit_ref, e))?;
        crate::fit::fitted_model::check_fit_config(fit_ref, &problem.model.camdl, &given, model_path, compile)
    } else {
        let resolved = crate::fit_store::opened(fit_ref)?;
        crate::fit::fitted_model::check_stored_fit(
            fit_ref,
            &resolved,
            &resolved.config.problem,
            &given,
            model_path,
        )
    }
}

/// Act on a replay model check: a refusal exits before anything runs; a
/// reporting-only difference is said on stderr and the run proceeds.
fn report_replay_check(out: &crate::out::Out, outcome: Result<Option<String>, String>) {
    match outcome {
        Ok(None) => {}
        Ok(Some(notice)) => out.say(format_args!("{notice}")),
        Err(e) => {
            out.fail(format!("error: {e}"));
        }
    }
}

/// The fit's own intervention selection, for `simulate --draws prior|<file>
/// --fit`: the same [`util::FitBase`] the posterior path reads, so every arm
/// replays on the fit's `scenario` / `enable` / `disable` and a flag applies
/// on top of it. `--fit` here is a fit config or any stored-fit handle.
fn fit_base_for_draws(out: &crate::out::Out, fit_ref: &str, source: &str) -> Result<util::FitBase, String> {
    let base = if std::path::Path::new(fit_ref).is_file() {
        let problem = fit::config_v2::Problem::load(fit_ref)
            .map_err(|e| wrap_fit_load_error(fit_ref, e))?;
        util::FitBase::of(&problem)?
    } else {
        let resolved = crate::fit_store::opened(fit_ref)?;
        util::FitBase::of(&resolved.config.problem)?
    };
    if !base.is_empty() {
        out.say(format_args!(
            "draws: {source} — every arm runs the fit's own {}; a scenario \
             applies its changes on top of it",
            base.describe()
        ));
    }
    Ok(base)
}

///
/// Returns the draw rows — the `[estimate]` parameters only — and, separately,
/// the config's resolved `[fixed]` block. The fixed values are not written
/// into the rows (gh#949): a row resolves at the draw tier, above `--params`,
/// while `[fixed]` belongs at the `fit.toml [fixed]` tier below it. The
/// caller routes them there via `SimRun::fit_fixed`.
fn generate_prior_draws(
    out: &crate::out::Out,
    fit_path: &str,
    n: usize,
    seed: u64,
    model: &ir::Model,
) -> Result<(Vec<HashMap<String, f64>>, indexmap::IndexMap<String, f64>), String> {
    use fit::config_v2::{EstimatePriorSpec, Problem};
    use crate::fit::priors_precedence::{
        resolve_priors_with_precedence, PriorSource,
    };

    let config = Problem::load(fit_path).map_err(|e| wrap_fit_load_error(fit_path, e))?;
    // Model-aware: `[fixed] from_scenario` names a model scenario.
    let fixed = config.fixed.resolve_with_model(model)?;
    // The rows carry the [estimate] parameters only, so an empty [estimate]
    // samples nothing and every cell runs at one parameter vector (run spec
    // §3.4: no usable prior is a hard error).
    if config.estimate.is_empty() {
        return Err(format!(
            "--draws prior: the fit config's [estimate] is empty, so there is nothing to \
             sample and every draw would be the same parameter vector.\n  \
             Fix: list each parameter to vary in [estimate] of `{fit_path}` with a \
             `prior = {{ <dist> = {{ ... }} }}` (or declare it with `~ <dist>(...)` in the \
             model's parameters {{}} block), or drop --fit to sample the model's own priors."
        ));
    }

    // Three-tier resolution: fit_toml > model_ir > flat. Walks every
    // estimated param in declaration order.
    let names: Vec<String> = config.estimate.keys().cloned().collect();
    let resolved = resolve_priors_with_precedence(&names, &config.estimate, model);
    // gh#369: the draws come from the fit.toml prior where it displaces a
    // model `~` prior; say so once, before any draw.
    if let Some(w) = fit::priors_precedence::format_prior_override_warning(
        &fit::priors_precedence::prior_overrides(&names, &config.estimate, model),
    ) {
        out.say_raw(format_args!("{}", w));
    }

    // Identify params with no usable prior in either source. The two
    // flat cases are both unusable for sampling — there's no finite
    // distribution to draw from — but they get distinguishable error
    // text because the remediation differs:
    //   - FlatFallback: neither tier supplied a prior; user must add one
    //     to the model file OR the fit toml.
    //   - FlatExplicit: user wrote `prior = { flat = {} }`; the only
    //     remediation is to replace it with a proper distribution.
    let unusable: Vec<&str> = resolved.iter()
        .filter(|r| matches!(r.source,
            PriorSource::FlatFallback | PriorSource::FlatExplicit))
        .map(|r| r.param.as_str())
        .collect();
    if !unusable.is_empty() {
        // For each unusable param, find its source class for the message.
        let has_explicit_flat = resolved.iter()
            .any(|r| r.source == PriorSource::FlatExplicit);
        let missing_list = unusable.join(", ");
        let first = unusable[0];
        let mut msg = format!(
            "--draws prior requires a proper (non-flat) prior on every \
             estimated parameter.\n  \
             Missing or flat priors: {}\n  \
             To fix, either:\n    \
             (i)  add `prior = {{ <dist> = {{ ... }} }}` to `[estimate.{}]` \
                  in your fit.toml \
                  (e.g. `prior = {{ log_normal = {{ mu = 0, sigma = 1 }} }}`), \
                  OR\n    \
             (ii) add a `~ <dist>(...)` declaration to parameter `{}` \
                  in your .camdl model file.",
            missing_list, first, first,
        );
        if has_explicit_flat {
            msg.push_str(
                "\n  Note: `prior = { flat = {} }` is rejected because there \
                 is no finite distribution to sample from — flat is an \
                 improper uniform with infinite support.");
        }
        return Err(msg);
    }

    // Sampling. Each resolved entry corresponds to one param in `names`
    // (and thus to one entry in `config.estimate`); to honour the
    // precedence chain we need the original PriorDist (fit_toml ▸ ir),
    // not the runtime `Prior`. Re-walk the precedence chain in the same
    // order to extract the PriorDist used for sampling. The clamp uses
    // the fit toml's `bounds` when present (preserving the existing
    // behaviour for fit-toml-narrower-than-model bounds).
    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_PRIOR);
    let mut draws = Vec::with_capacity(n);

    for _ in 0..n {
        let mut row = HashMap::new();
        for (name, spec) in &config.estimate {
            // Mirror `resolve_priors_with_precedence` to pick the PriorDist.
            // The unusable check above guarantees we find a non-flat one.
            // `synthesized` holds an owned PriorDist for the uniform-over-bounds
            // form (which carries no lower/upper of its own — they come from
            // `bounds`); the deferred binding keeps it alive for the `&` below.
            let synthesized;
            let pd: &ir::parameter::PriorDist = match spec.prior.as_ref() {
                Some(EstimatePriorSpec::Dist(pd)) => pd,
                Some(EstimatePriorSpec::UniformOverBounds { .. }) => {
                    // Same resolution as resolve_prior: fit.toml bounds, else
                    // the model's `in [lo, hi]`.
                    let (lo, hi) = spec.bounds
                        .or_else(|| model.parameters.iter()
                            .find(|p| &p.name == name)
                            .and_then(crate::params_resolver::resolved_bounds))
                        .ok_or_else(|| format!(
                            "parameter '{}': prior = {{ uniform = {{}} }} requires bounds — \
                             add `in [lo, hi]` in the model or `bounds = [lo, hi]` to \
                             [estimate.{}].", name, name))?;
                    synthesized = ir::parameter::PriorDist::Uniform(
                        ir::parameter::UniformPrior { lower: lo, upper: hi });
                    &synthesized
                }
                Some(EstimatePriorSpec::Flat { .. }) => unreachable!(
                    "explicit flat priors rejected by the unusable check above"),
                None => {
                    let ir_param = model.parameters.iter()
                        .find(|p| &p.name == name)
                        .expect("unusable check guarantees presence");
                    ir_param.prior_dist().expect(
                        "unusable check guarantees a prior in either source")
                }
            };
            let value = sample_from_prior_raw(pd, &mut rng);
            // Bounds-optional: clamp to fit.toml's [estimate.X].bounds
            // when present; otherwise pass the raw prior draw through
            // (the model file's parameters block bounds will catch
            // out-of-range draws downstream during validation).
            let clamped = match spec.bounds {
                Some((lo, hi)) => value.clamp(lo, hi),
                None => value,
            };
            row.insert(name.clone(), clamped);
        }
        draws.push(row);
    }

    // Provenance: report how many came from each tier so users can
    // verify the right source was consulted.
    let n_fit_toml = resolved.iter()
        .filter(|r| r.source == PriorSource::FitToml).count();
    let n_model_ir = resolved.iter()
        .filter(|r| r.source == PriorSource::ModelIr).count();
    out.say(format_args!(
        "generated {} prior draws from {} ({} estimated [{} fit-toml + {} model-IR] + {} fixed params)",
        n, fit_path, config.estimate.len(), n_fit_toml, n_model_ir, fixed.len()
    ));
    Ok((draws, fixed))
}

/// Generate N draws from priors embedded in the model IR.
///
/// Each row carries only the parameters that have a prior (gh#949). A
/// parameter without one is left out of the row and resolves per cell through
/// the ordinary tiers — model default → `--params` → the cell's own scenario →
/// `--param`. Writing it into the row would put it at the draw tier, above
/// `--params`, and (when filled from the listed scenarios) would hand one
/// arm's scenario value to every other arm. `run_simulate` refuses up front a
/// cell in which such a parameter gets no value from any tier.
fn generate_prior_draws_from_ir(
    out: &crate::out::Out,
    ir_path: &str,
    n: usize,
    seed: u64,
) -> Result<Vec<HashMap<String, f64>>, String> {
    let (model, _) = util::load_ir(ir_path)?;

    // The sampled parameters, in declaration order — the RNG consumption
    // order, so a given seed yields the same values it always has.
    let with_prior: Vec<(&ir::parameter::Parameter, &ir::parameter::PriorDist)> =
        model.parameters.iter()
            .filter_map(|p| p.prior_dist().map(|pd| (p, pd)))
            .collect();
    // Nothing to sample: every row would be empty and every cell would run at
    // one parameter vector, which is not a draw from anything (run spec §3.4:
    // no usable prior is a hard error).
    if with_prior.is_empty() {
        let example = model.parameters.first().map_or("beta", |p| p.name.as_str());
        return Err(format!(
            "--draws prior: no parameter of the model has a prior, so there is nothing to \
             sample and every draw would be the same parameter vector.\n  \
             Fix: declare a prior on each parameter to vary, after its type and bounds in \
             the model's parameters {{}} block (e.g. `{example} : rate in [0.01, 1.0] ~ \
             log_normal(mu = -1.0, sigma = 0.5)`), or pass --fit <fit.toml> whose \
             [estimate] gives each parameter a `prior = {{ <dist> = {{ ... }} }}`."
        ));
    }

    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_PRIOR);
    let mut draws = Vec::with_capacity(n);
    // Per-parameter rejection counts for bounds-truncation diagnostics.
    let mut reject_counts: HashMap<&str, u64> = HashMap::new();

    for _ in 0..n {
        let mut row = HashMap::new();
        for &(p, pd) in &with_prior {
            // Truncate to the range the parameter is valid on, which includes
            // the one its type carries (gh#763) — otherwise this emits a
            // `probability` outside [0, 1] that every downstream command then
            // rejects. A no-op unless the prior actually places mass outside
            // that range.
            let bounds = crate::params_resolver::resolved_bounds(p);
            let (v, rejected) = sample_with_bounds(pd, bounds, &mut rng, &p.name)?;
            if rejected > 0 {
                *reject_counts.entry(p.name.as_str()).or_insert(0) += rejected;
            }
            row.insert(p.name.clone(), v);
        }
        draws.push(row);
    }

    // Warn on high truncation rates — a strong signal that the prior is
    // mis-calibrated for the declared bounds.
    let mut report: Vec<(&str, u64)> = reject_counts.into_iter().collect();
    report.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, rej) in &report {
        let accept = n as u64;
        let total = accept + rej;
        let pct = 100.0 * (*rej as f64) / (total as f64);
        if pct >= 10.0 {
            out.say(format_args!(
                "warning: prior for '{}' placed {:.1}% mass outside declared bounds \
                 ({} rejected / {} accepted). Consider widening bounds or tightening \
                 the prior.",
                name, pct, rej, accept
            ));
        }
    }

    out.say(format_args!("generated {} prior draws from model IR ({} sampled params; {} other \
               parameter(s) resolve per cell from --params, the scenario's set or --param)",
        n, with_prior.len(), model.parameters.len() - with_prior.len()));
    Ok(draws)
}

/// Sample from a prior and truncate to parameter bounds via rejection.
/// Returns (value, n_rejected). Errors if the prior is so mis-calibrated
/// that it fails to produce a bounds-satisfying sample within the retry cap.
fn sample_with_bounds(
    pd: &ir::parameter::PriorDist,
    bounds: Option<(f64, f64)>,
    rng: &mut sim::rng::StatefulRng,
    param_name: &str,
) -> Result<(f64, u64), String> {
    const MAX_ATTEMPTS: u32 = 256;
    let (lo, hi) = match bounds {
        Some(b) => b,
        None => return Ok((sample_from_prior_raw(pd, rng), 0)),
    };
    for rejected in 0..MAX_ATTEMPTS {
        let v = sample_from_prior_raw(pd, rng);
        if v >= lo && v <= hi {
            return Ok((v, rejected as u64));
        }
    }
    Err(format!(
        "prior for parameter '{}' failed to produce a value within bounds [{}, {}] \
         after {} attempts — the declared prior places essentially all its mass \
         outside the parameter bounds. Check that the distribution and its \
         arguments match the parameter's natural scale.",
        param_name, lo, hi, MAX_ATTEMPTS
    ))
}

/// Draw a single value from an IR PriorDist, ignoring bounds.
fn sample_from_prior_raw(
    pd: &ir::parameter::PriorDist,
    rng: &mut sim::rng::StatefulRng,
) -> f64 {
    use ir::parameter::PriorDist;
    match pd {
        PriorDist::Uniform(u) => u.lower + (u.upper - u.lower) * rng.uniform(),
        PriorDist::Normal(p) => p.mean + p.sd * rng.normal(),
        PriorDist::LogNormal(p) => (p.mu + p.sigma * rng.normal()).exp(),
        PriorDist::HalfNormal(p) => (p.sigma * rng.normal()).abs(),
        PriorDist::Beta(p) => {
            use rand::prelude::Distribution;
            let x = rand_distr::Gamma::new(p.alpha, 1.0).unwrap().sample(rng.inner_mut());
            let y = rand_distr::Gamma::new(p.beta, 1.0).unwrap().sample(rng.inner_mut());
            x / (x + y)
        }
        PriorDist::Gamma(p) => {
            use rand::prelude::Distribution;
            // rand_distr uses scale parameter, not rate
            let scale = 1.0 / p.rate;
            rand_distr::Gamma::new(p.shape, scale).unwrap().sample(rng.inner_mut())
        }
        PriorDist::Exponential(p) => {
            // Inverse CDF: -ln(U)/rate
            let u = rng.uniform().max(1e-300);
            -u.ln() / p.rate
        }
        PriorDist::LogUniform(p) => {
            // Uniform on the log scale, then exponentiate — always in [lower, upper].
            let (ll, lu) = (p.lower.ln(), p.upper.ln());
            (ll + rng.uniform() * (lu - ll)).exp()
        }
        PriorDist::TruncatedNormal(p) => {
            // Exact inverse-CDF draw inside [lower, upper] (no rejection):
            //   θ = μ + σ·Φ⁻¹(Φ(α) + U·(Φ(β) − Φ(α))),  α,β = standardized bounds.
            use sim::inference::{normal_cdf, normal_quantile};
            let a = normal_cdf((p.lower - p.mean) / p.sd);
            let b = normal_cdf((p.upper - p.mean) / p.sd);
            let q = a + rng.uniform() * (b - a);
            (p.mean + p.sd * normal_quantile(q)).clamp(p.lower, p.upper)
        }
        PriorDist::Fixed(v) => *v,
    }
}

/// Parse a seeds spec: "1:100" (range), "42" (single), "1,2,3,42" (list).
#[cfg(test)]
fn parse_seeds_spec(spec: &str) -> Result<Vec<u64>, String> {
    // Range: "1:100"
    if spec.contains(':') {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() != 2 {
            return Err(format!("invalid range '{}', expected FROM:TO", spec));
        }
        let from: u64 = parts[0].trim().parse()
            .map_err(|_| format!("cannot parse '{}' as integer", parts[0]))?;
        let to: u64 = parts[1].trim().parse()
            .map_err(|_| format!("cannot parse '{}' as integer", parts[1]))?;
        if from > to {
            return Err(format!("empty range {}:{}", from, to));
        }
        Ok((from..=to).collect())
    }
    // Comma-separated list: "1,2,3,42"
    else if spec.contains(',') {
        spec.split(',')
            .map(|s| s.trim().parse::<u64>()
                .map_err(|_| format!("cannot parse '{}' as integer", s.trim())))
            .collect()
    }
    // Single: "42"
    else {
        let n: u64 = spec.trim().parse()
            .map_err(|_| format!("cannot parse '{}' as integer", spec))?;
        Ok(vec![n])
    }
}

/// Write sampled draws to a TSV (gh#157), one row per draw and one column
/// per parameter. The column set is the union of every draw's keys, sorted
/// for a deterministic header; values use `{:.17e}` (full f64 precision, the
/// same format the fit pipeline writes). This is exactly the format
/// `load_draws_tsv` reads back, so `--draws-out` output round-trips through
/// `--draws PATH`. An empty/no-parameter draw set (the no-`--draws` single
/// point) yields a header-only file.
fn write_draws_tsv(path: &str, draws: &[HashMap<String, f64>]) -> Result<(), String> {
    use std::collections::BTreeSet;
    use std::io::Write;

    let cols: Vec<String> = {
        let mut set: BTreeSet<&str> = BTreeSet::new();
        for d in draws {
            for k in d.keys() {
                set.insert(k.as_str());
            }
        }
        set.into_iter().map(|s| s.to_string()).collect()
    };

    let mut f = std::io::BufWriter::new(
        std::fs::File::create(path).map_err(|e| format!("cannot create {}: {}", path, e))?,
    );
    writeln!(f, "{}", cols.join("\t")).map_err(|e| e.to_string())?;
    for d in draws {
        let row: Vec<String> = cols
            .iter()
            .map(|c| format!("{:.17e}", d.get(c).copied().unwrap_or(f64::NAN)))
            .collect();
        writeln!(f, "{}", row.join("\t")).map_err(|e| e.to_string())?;
    }
    f.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Load a draws TSV file. Each row is a complete parameter vector.
/// Column names must match model parameter names.
/// Returns Vec<HashMap<param_name, value>>.
/// One parsed `draws.tsv` row: the optional `(chain, draw)` posterior key
/// (gh#322 — the join key to the smoothed `trajectories.tsv`; `None` for a
/// pre-key, param-only file) plus the model parameters, and the row's place
/// in the file.
#[derive(Debug, Clone)]
pub(crate) struct KeyedDraw {
    /// The row's 0-based position among the file's data rows (the header is
    /// not a row). A chain selection drops rows but never renumbers this, so
    /// an artifact built from a subset can still name the row a draw came
    /// from.
    pub row: usize,
    pub chain: Option<usize>,
    pub draw: Option<usize>,
    pub params: HashMap<String, f64>,
}

/// Parse a `draws.tsv` KEEPING the `(chain, draw)` key. The single parser;
/// [`load_draws_tsv`] wraps this and drops the key so every param-only reader
/// (predict's schema validator, the engine, compare) is unchanged.
#[expect(clippy::disallowed_methods, reason = "parses the draws file its caller names: `--draws FILE` or a stored fit's draws through `posterior_draws`, whose rows' parameter values a cell's identity hashes, or a chain subset's draws, read only to recompute diagnostics")]
pub(crate) fn load_draws_tsv_keyed(path: &str) -> Result<Vec<KeyedDraw>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut lines = content.lines();
    let header = lines.next()
        .ok_or_else(|| format!("empty draws file: {}", path))?;
    // Strip trailing empty columns (from trailing tabs)
    let col_names: Vec<&str> = header.split('\t')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if col_names.len() < 2 {
        return Err(format!("draws file needs at least 2 columns, got {}", col_names.len()));
    }

    let mut draws = Vec::new();
    for (line_num, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        // Split and trim; take only as many fields as we have column names
        let fields: Vec<&str> = line.split('\t')
            .map(|s| s.trim())
            .collect();
        if fields.len() < col_names.len() {
            return Err(format!(
                "draws file line {}: expected {} columns, got {}",
                line_num + 2, col_names.len(), fields.len()
            ));
        }
        let mut chain: Option<usize> = None;
        let mut draw: Option<usize> = None;
        let mut params = HashMap::new();
        for (col, field) in col_names.iter().zip(fields.iter()) {
            // gh#322: `chain` / `draw` are the posterior KEY columns, not model
            // parameters — captured as the join key, never inserted as params.
            match *col {
                "chain" => {
                    chain = Some(field.parse().map_err(|_| format!(
                        "draws file line {}: chain '{}' is not a non-negative integer",
                        line_num + 2, field))?);
                }
                "draw" => {
                    draw = Some(field.parse().map_err(|_| format!(
                        "draws file line {}: draw '{}' is not a non-negative integer",
                        line_num + 2, field))?);
                }
                _ => {
                    let val: f64 = field.parse().map_err(|_| format!(
                        "draws file line {}, column '{}': cannot parse '{}' as number",
                        line_num + 2, col, field))?;
                    params.insert(col.to_string(), val);
                }
            }
        }
        draws.push(KeyedDraw { row: draws.len(), chain, draw, params });
    }

    if draws.is_empty() {
        return Err(format!("draws file has header but no data rows: {}", path));
    }
    Ok(draws)
}

/// The selection `simulate --draws posterior` replays under, resolved against
/// the cloud's chains and warned about before any draw is read; an absent id
/// or a selection that keeps every chain or none is refused here.
fn warn_posterior_selection(
    out: &crate::out::Out,
    sel: &crate::chain_selection::ChainSelection,
    draws_path: &std::path::Path,
) {
    let info = crate::chain_selection::chains_in_draws(draws_path)
        .and_then(|present| sel.resolve(&present))
        .unwrap_or_else(|e| {
            out.fail(format!("error: {e}"));
        });
    crate::chain_selection::warn_active_selection(
        out, sel.polarity(), &info, crate::out::paint::stderr());
}

/// Param-only draw rows — the `(chain, draw)` key (if any) is dropped. Every
/// reader that treats a row as a parameter vector uses this.
pub(crate) fn load_draws_tsv(path: &str) -> Result<Vec<HashMap<String, f64>>, String> {
    Ok(load_draws_tsv_keyed(path)?.into_iter().map(|d| d.params).collect())
}

/// Print a dry run summary: resolved parameters with provenance.
#[allow(clippy::too_many_arguments)]
#[expect(clippy::disallowed_methods, reason = "reads the header of the `--draws` file to print its size for `--dry-run`, which runs and stores nothing")]
fn print_dry_run(
    out: &crate::out::Out,
    ir_path: &str,
    ir_path_compiled: &str,
    backend: args::types::ForwardBackend,
    dt: f64,
    seed: u64,
    params_files: &[String],
    cli_overrides: &HashMap<String, f64>,
    scenario_list: &[Option<String>],
    // The label of the ad-hoc arm a run without `--scenario` has.
    adhoc_label: &str,
    seeds: &[u64],
    draws_path: &Option<String>,
    n_draws: usize,
    replicates: usize,
    total_runs: usize,
    obs_path: &Option<String>,
    obs_dir: &Option<String>,
    obs_only: &Option<String>,
    design: Option<&crate::sweep::ResolvedDesign>,
) {
    let d = out::paint::dim;
    let b = out::paint::bold;

    out.say(format_args!("{}", b("camdl simulate (dry run)")));
    out.say("");

    // Header info
    out.say(format_args!("  {} {}", d("model:"), ir_path));
    out.say(format_args!("  {} {}", d("backend:"), backend));
    out.say(format_args!("  {} {}", d("dt:"), dt));

    // The design axis: what the grid declaration resolved to.
    let swept_names: Vec<String> = match design {
        Some(crate::sweep::ResolvedDesign::Sweep { params, n_points }) => {
            for p in params {
                let vals: Vec<String> = p.values.iter().map(|v| v.to_string()).collect();
                out.say(format_args!("  {} {} = [{}]", d("sweep:"), p.name, vals.join(", ")));
            }
            out.say(format_args!("  {} {} design point(s)", d("design:"), n_points));
            params.iter().map(|p| p.name.clone()).collect()
        }
        Some(crate::sweep::ResolvedDesign::File { params, digest, n_points, point_ids }) => {
            out.say(format_args!("  {} {} design point(s) over {} (file digest {})",
                d("sweep-file:"), n_points, params.join(", "), &digest[..8]));
            // The file's own ids are worth a line; row numbering is the default.
            if point_ids.iter().enumerate().any(|(row, id)| row != *id) {
                let ids: Vec<String> = point_ids.iter().map(|i| i.to_string()).collect();
                out.say(format_args!("  {} {}", d("point_id:"), ids.join(", ")));
            }
            params.clone()
        }
        None => Vec::new(),
    };

    if seeds.len() > 1 {
        out.say(format_args!("  {} {}:{} ({} seeds)", d("seeds:"), seeds[0], seeds[seeds.len()-1], seeds.len()));
    } else {
        out.say(format_args!("  {} {}", d("seed:"), seed));
    }

    if let Some(ref dp) = draws_path {
        out.say(format_args!("  {} {}", d("draws:"), dp));
    }
    if replicates > 1 && draws_path.is_none() {
        out.say(format_args!("  {} {}", d("replicates:"), replicates));
    }

    let implicit = adhoc_label;
    let scenarios: Vec<&str> = scenario_list.iter()
        .map(|s| s.as_deref().unwrap_or(implicit))
        .collect();
    if scenarios.len() > 1 || scenarios[0] != implicit {
        out.say(format_args!("  {} {}", d("scenarios:"), scenarios.join(", ")));
    } else {
        out.say(format_args!("  {} {implicit}", d("scenario:")));
    }

    // Obs output
    if let Some(ref p) = obs_path { out.say(format_args!("  {} {}", d("obs:"), p)); }
    if let Some(ref p) = obs_dir { out.say(format_args!("  {} {}", d("obs-dir:"), p)); }
    if let Some(ref p) = obs_only { out.say(format_args!("  {} {}", d("obs-only:"), p)); }

    out.say("");

    // Parameter provenance — load model and trace where each value comes from
    if draws_path.is_some() && n_draws > 1 {
        // Draws mode: don't show per-parameter provenance (values vary per draw)
        if let Some(ref dp) = draws_path {
            if dp != "uniform" && dp != "prior" {
                // Try to read the header to show column count
                if let Ok(content) = std::fs::read_to_string(dp) {
                    if let Some(header) = content.lines().next() {
                        let cols: Vec<&str> = header.split('\t')
                            .map(|s| s.trim())
                            .filter(|s| !s.is_empty())
                            .collect();
                        let n_rows = content.lines().count() - 1;
                        out.say(format_args!("  {} {} rows × {} params",
                            d("draws file:"), n_rows, cols.len()));
                    }
                }
            }
        }
    } else {
        // Point/single mode: show resolved parameter values with provenance.
        // Load the already-compiled IR (display still shows the source path).
        match util::load_ir(ir_path_compiled) {
            Ok((model, _)) => {
                // Track provenance: (param_name → (value, source, override_chain))
                struct ParamProv {
                    value: f64,
                    source: String,
                    overrides: Vec<(f64, String)>, // (old_value, old_source)
                }
                let mut provs: std::collections::BTreeMap<String, ParamProv> = std::collections::BTreeMap::new();

                // Values compiled into the IR (`camdlc --set`; a .camdl
                // never writes one).
                for p in &model.parameters {
                    if let Some(v) = p.value.resolved_value() {
                        provs.insert(p.name.clone(), ParamProv {
                            value: v, source: "compiled-in value".to_string(), overrides: vec![],
                        });
                    }
                }

                // Params files (in order)
                for path in params_files {
                    if let Ok(toml_vals) = util::load_params_toml(path) {
                        for (name, &v) in &toml_vals {
                            if let Some(prov) = provs.get_mut(name) {
                                if (prov.value - v).abs() > 1e-15 {
                                    prov.overrides.push((prov.value, prov.source.clone()));
                                    prov.value = v;
                                    prov.source = path.clone();
                                }
                            } else {
                                provs.insert(name.clone(), ParamProv {
                                    value: v, source: path.clone(), overrides: vec![],
                                });
                            }
                        }
                    }
                }

                // CLI --param overrides
                for (name, &v) in cli_overrides {
                    if let Some(prov) = provs.get_mut(name) {
                        if (prov.value - v).abs() > 1e-15 {
                            prov.overrides.push((prov.value, prov.source.clone()));
                            prov.value = v;
                            prov.source = "--param".to_string();
                        }
                    } else {
                        provs.insert(name.clone(), ParamProv {
                            value: v, source: "--param".to_string(), overrides: vec![],
                        });
                    }
                }

                // A swept parameter varies per design point: listed above,
                // not here with one value.
                for name in &swept_names {
                    provs.remove(name);
                }

                // Print
                let max_name_len = provs.keys().map(|k| k.len()).max().unwrap_or(0);
                out.say(format_args!("Parameters ({}):", provs.len()));
                for (name, prov) in &provs {
                    let val_str = b(&format_param_value(prov.value));
                    let source_str = if prov.overrides.is_empty() {
                        d(&prov.source)
                    } else {
                        let chain: Vec<String> = prov.overrides.iter()
                            .map(|(v, s)| format!("{} in {}", format_param_value(*v), s))
                            .collect();
                        d(&format!("{} (was {})", prov.source, chain.join(" → ")))
                    };
                    out.say(format_args!("  {:width$} = {:>14}  {}",
                        name, val_str, source_str, width = max_name_len));
                }
            }
            Err(e) => {
                out.say(format_args!("  {} {}", d("(could not load model for parameter resolution:"), e));
            }
        }
    }

    // Total runs
    if total_runs > 1 {
        out.say("");
        let n_design_points = match design {
            Some(crate::sweep::ResolvedDesign::Sweep { n_points, .. })
            | Some(crate::sweep::ResolvedDesign::File { n_points, .. }) => *n_points,
            None => 1,
        };
        let parts: Vec<String> = [
            if n_design_points > 1 { Some(format!("{} design points", n_design_points)) } else { None },
            if n_draws > 1 { Some(format!("{} draws", n_draws)) } else { None },
            if scenarios.len() > 1 { Some(format!("{} scenarios", scenarios.len())) } else { None },
            if seeds.len() > 1 { Some(format!("{} seeds", seeds.len())) } else { None },
            if replicates > 1 && seeds.len() == 1 { Some(format!("{} replicates", replicates)) } else { None },
        ].iter().flatten().cloned().collect();
        out.say(format_args!("  {} {} = {} runs", d("total:"), parts.join(" × "), total_runs));
    }
}

fn format_param_value(v: f64) -> String {
    if v == v.round() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else if v.abs() < 0.001 || v.abs() >= 1e6 {
        format!("{:.4e}", v)
    } else {
        format!("{:.6}", v)
    }
}

#[cfg(test)]
mod obs_grid_guard_tests {
    use super::*;

    fn traj(times: &[f64]) -> sim::Trajectory {
        let mut tr = sim::Trajectory::new();
        for &t in times {
            tr.push(sim::Snapshot {
                t,
                int_state: sim::state::IntState { counts: vec![0] },
                real_state: sim::state::RealState::new(0),
                flows: sim::state::Flows::Int(vec![0]),
            });
        }
        tr
    }

    /// The guard must agree with the RESOLVER, not approximate it.
    ///
    /// An earlier version compared with a relative `OUTPUT_EPS * |t|` while
    /// `snap_at`/`incidence_over` resolve with an absolute `OBS_SNAP_EPS`.
    /// Those cross at t > 1000: at t = 3009 the relative window is ~3.0e-9,
    /// wider than the absolute 1e-9, so a snapshot 2e-9 late was ACCEPTED by
    /// the guard and then skipped by the resolver, which fell back to the
    /// previous snapshot — the silent-wrong the guard exists to prevent,
    /// reintroduced one layer up.
    #[test]
    fn near_equal_snapshot_the_resolver_would_skip_is_rejected() {
        let t_obs = 3009.0;
        let late = t_obs + 2e-9; // inside a relative window, outside OBS_SNAP_EPS
        let tr = traj(&[3008.0, late]);

        // Precondition: the resolver really does skip it and fall back.
        assert_eq!(
            snap_at(&tr, t_obs).t,
            3008.0,
            "precondition: snap_at resolves t=3009 to the 3008 snapshot"
        );

        let err = check_obs_times_on_snapshot_grid(&tr, "s", "days", &[t_obs])
            .expect_err("a snapshot the resolver skips must be rejected");
        assert!(err.contains("not a recorded output time"), "got: {err}");
    }

    /// An exact hit, and one inside the resolver's own tolerance, are accepted.
    #[test]
    fn exact_and_within_tolerance_snapshots_are_accepted() {
        let tr = traj(&[0.0, 1.0, 2.0]);
        check_obs_times_on_snapshot_grid(&tr, "s", "days", &[0.0, 1.0, 2.0])
            .expect("exact grid hits are fine");

        let tr2 = traj(&[0.0, 1.0 - 1e-10]);
        check_obs_times_on_snapshot_grid(&tr2, "s", "days", &[1.0])
            .expect("within OBS_SNAP_EPS resolves to that snapshot, so it is fine");
    }

    /// An empty trajectory is an ERROR, not silently Ok. The previous version
    /// returned Ok claiming "a separate error path covers this"; there is none
    /// — `snap_at` calls `crate::out::ExitCode(1).exit()` after partial artifacts are written.
    #[test]
    fn empty_trajectory_is_an_error_not_a_pass() {
        let tr = traj(&[]);
        let err = check_obs_times_on_snapshot_grid(&tr, "s", "days", &[0.0])
            .expect_err("no snapshots must be an error");
        assert!(err.contains("recorded no trajectory snapshots"), "got: {err}");
    }

    /// gh#606: the observation axis and the recording axis confine to the
    /// horizon with one test, so they agree on the terminal point. A step of 0.1
    /// accumulated 7418 times lands 1.0e-10 past `t_end = 741.8`; the old bounds
    /// (`t_end + 1e-9` here, `t_end + step·1e-9` on the recording axis) split on
    /// it, so the observation at t_end had no recorded row and the run was
    /// refused by the snapshot-grid check.
    #[test]
    fn obs_axis_and_output_axis_agree_at_the_terminal_point() {
        let t_end = 741.8;
        let obs = obs_schedule_times(
            &ir::observation::ObservationSchedule::Regular(ir::observation::RegularSchedule {
                start: 0.0,
                step: 0.1,
                end: t_end,
            }),
            None,
            t_end,
        );
        let out = sim::output::output_times(
            &ir::model::OutputSchedule::Regular(ir::model::RegularOutputSchedule {
                start: 0.0,
                step: 0.1,
            }),
            t_end,
        );
        assert_eq!(obs.len(), 7419, "0, 0.1, ..., 741.8 on the observation axis");
        assert_eq!(
            (obs.len(), obs.last()),
            (out.len(), out.last()),
            "the two axes agree on the terminal point"
        );
        assert!(obs == out, "the two axes enumerate the same times");
        check_obs_times_on_snapshot_grid(&traj(&out), "s", "days", &obs)
            .expect("every observation time has a recorded row");

        // An at-list time one rounding past t_end is the terminal point on both axes.
        let at = vec![10.0, t_end + 1e-13];
        let obs_at = obs_schedule_times(
            &ir::observation::ObservationSchedule::AtTimes(at.clone()),
            None,
            t_end,
        );
        let out_at = sim::output::output_times(&ir::model::OutputSchedule::AtTimes(at), t_end);
        assert_eq!(obs_at.len(), 2);
        assert_eq!(obs_at, out_at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_tsv_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_draws.tsv");

        // Write a draws file
        std::fs::write(&path, "beta\tgamma\tN0\n\
            3.00000000000000000e-01\t1.00000000000000000e-01\t1.00000000000000000e+06\n\
            5.00000000000000000e-01\t1.50000000000000000e-01\t1.00000000000000000e+06\n").unwrap();

        let draws = load_draws_tsv(path.to_str().unwrap()).unwrap();
        assert_eq!(draws.len(), 2);
        assert!((draws[0]["beta"] - 0.3).abs() < 1e-15);
        assert!((draws[0]["gamma"] - 0.1).abs() < 1e-15);
        assert!((draws[0]["N0"] - 1e6).abs() < 1e-5);
        assert!((draws[1]["beta"] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn draws_tsv_tolerates_trailing_tabs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_trailing.tsv");

        // File with trailing tabs (the bug we fixed)
        std::fs::write(&path, "beta\tgamma\t\n0.3\t0.1\t\n0.5\t0.15\t\n").unwrap();

        let draws = load_draws_tsv(path.to_str().unwrap()).unwrap();
        assert_eq!(draws.len(), 2);
        assert!((draws[0]["beta"] - 0.3).abs() < 1e-15);
    }

    #[test]
    fn draws_tsv_strips_chain_draw_key_columns() {
        // gh#322: a keyed draws.tsv (leading `chain`/`draw`) loads as PARAM-ONLY
        // rows — the key columns are stripped in the one shared loader, so
        // predict's schema validator + the engine never see them as parameters.
        // (A pre-key, param-only file has no such columns and is unchanged.)
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keyed_draws.tsv");
        std::fs::write(
            &path,
            "chain\tdraw\tbeta\tgamma\n0\t20\t0.3\t0.1\n0\t21\t0.5\t0.15\n1\t20\t0.4\t0.12\n",
        )
        .unwrap();
        let draws = load_draws_tsv(path.to_str().unwrap()).unwrap();
        assert_eq!(draws.len(), 3);
        assert!(!draws[0].contains_key("chain"), "chain must be stripped");
        assert!(!draws[0].contains_key("draw"), "draw must be stripped");
        assert_eq!(
            draws[0].keys().cloned().collect::<std::collections::BTreeSet<_>>(),
            ["beta".to_string(), "gamma".to_string()].into_iter().collect(),
            "only model params survive the strip"
        );
        assert!((draws[0]["beta"] - 0.3).abs() < 1e-15);
        assert!((draws[2]["beta"] - 0.4).abs() < 1e-15);
    }

    #[test]
    fn draws_tsv_rejects_missing_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_short.tsv");

        std::fs::write(&path, "beta\tgamma\tN0\n0.3\t0.1\n").unwrap();
        let err = load_draws_tsv(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("expected 3 columns"));
    }

    #[test]
    fn draws_tsv_rejects_empty_body() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_empty.tsv");

        std::fs::write(&path, "beta\tgamma\n").unwrap();
        let err = load_draws_tsv(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("no data rows"));
    }

    // gh#157: --draws-out writes the sampled draws as a round-trippable TSV.
    #[test]
    fn write_draws_tsv_roundtrips_through_loader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out_draws.tsv");

        // Three sampled draws over two params (HashMap order is arbitrary;
        // the writer must produce a deterministic, loadable header).
        let draws: Vec<HashMap<String, f64>> = vec![
            HashMap::from([("beta".to_string(), 0.31), ("gamma".to_string(), 0.11)]),
            HashMap::from([("beta".to_string(), 0.52), ("gamma".to_string(), 0.15)]),
            HashMap::from([("beta".to_string(), 0.73), ("gamma".to_string(), 0.19)]),
        ];

        write_draws_tsv(path.to_str().unwrap(), &draws).unwrap();

        // N rows out, values exact-match the sampled draws (full f64
        // precision round-trips through {:.17e}).
        let loaded = load_draws_tsv(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), 3, "one row per draw");
        for (orig, got) in draws.iter().zip(loaded.iter()) {
            assert_eq!(orig["beta"], got["beta"]);
            assert_eq!(orig["gamma"], got["gamma"]);
        }

        // Negative control: distinct param values are not collapsed — a
        // writer that dropped rows or duplicated the first draw would fail
        // here even though len() matched.
        assert_ne!(loaded[0]["beta"], loaded[2]["beta"]);
        assert!((loaded[2]["beta"] - 0.73).abs() < 1e-15);
    }

    // Control for the CLI guard: the writer is the *only* thing that
    // creates the file, so a path never passed to write_draws_tsv stays
    // absent. (The CLI only calls it under `--draws-out`.)
    #[test]
    fn write_draws_tsv_not_called_leaves_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("never_written.tsv");
        assert!(!path.exists());
        // No write_draws_tsv call → no file. Loading errors rather than
        // silently materializing anything.
        assert!(load_draws_tsv(path.to_str().unwrap()).is_err());
        assert!(!path.exists());
    }

    #[test]
    fn parse_seeds_spec_range() {
        let seeds = parse_seeds_spec("1:5").unwrap();
        assert_eq!(seeds, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn parse_seeds_spec_list() {
        let seeds = parse_seeds_spec("42,137,256").unwrap();
        assert_eq!(seeds, vec![42, 137, 256]);
    }

    #[test]
    fn parse_seeds_spec_single() {
        let seeds = parse_seeds_spec("42").unwrap();
        assert_eq!(seeds, vec![42]);
    }

    #[test]
    fn parse_seeds_spec_empty_range() {
        let err = parse_seeds_spec("5:1").unwrap_err();
        assert!(err.contains("empty range"));
    }

    #[test]
    fn prior_draws_from_ir_sir_priors_golden() {
        // Load the sir_priors golden IR — all 5 params have priors, so we
        // should get 5 prior samples for each of the N draws.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let draws = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 7, 42).unwrap();
        assert_eq!(draws.len(), 7, "should produce N draws");
        for row in &draws {
            for name in ["beta", "gamma", "rho", "N0", "I0"] {
                let v = row.get(name).unwrap_or_else(|| panic!("missing {}", name));
                assert!(v.is_finite(), "{} must be finite, got {}", name, v);
                assert!(*v >= 0.0, "{} must be non-negative, got {}", name, v);
            }
            // Bounds clamping: beta ∈ [0.01, 2.0], rho ∈ [0.001, 1.0]
            assert!(row["beta"] >= 0.01 && row["beta"] <= 2.0);
            assert!(row["rho"] >= 0.001 && row["rho"] <= 1.0);
        }

        // Same seed → identical draws (reproducibility)
        let draws2 = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 7, 42).unwrap();
        for (a, b) in draws.iter().zip(draws2.iter()) {
            for (k, va) in a {
                assert_eq!(va, &b[k], "seed={} {} should be reproducible", 42, k);
            }
        }
    }

    #[test]
    fn prior_draws_from_ir_refuses_a_model_with_no_prior() {
        // sir_basic declares no priors. Sampling nothing would hand back N
        // empty rows, so N cells at one parameter vector reported as N prior
        // draws (run spec §3.4: no usable prior is a hard error).
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/../../../ocaml/golden/sir_basic.ir.json", manifest);
        let err = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 3, 1).unwrap_err();
        assert!(err.starts_with("--draws prior: no parameter of the model has a prior"),
            "unexpected error: {err}");
        assert!(err.contains("parameters {}") && err.contains("~ log_normal("),
            "the error must say where and how to declare a prior: {err}");
        assert!(err.contains("[estimate]"), "the error must name the fit.toml route: {err}");
    }

    #[test]
    fn uniform_draws_refuse_a_model_with_no_bounded_parameter() {
        // No parameter of death_fraction_window declares `in [lo, hi]`.
        // Sampling nothing would hand back N empty rows, so N cells at one
        // parameter vector reported as N uniform draws.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/../../../ocaml/golden/death_fraction_window.ir.json", manifest);
        let err = generate_uniform_draws(&crate::out::Out::new(true), &path, 3, 1).unwrap_err();
        assert!(err.starts_with("--draws uniform: no parameter of the model declares bounds"),
            "unexpected error: {err}");
        assert!(err.contains("`lo + (hi - lo)\u{b7}U`") && err.contains("declared bounds"),
            "the error must say what uniform samples: {err}");
        assert!(err.contains("parameters {}") && err.contains(" : <type> in [lo, hi]"),
            "the error must say how to give bounds: {err}");
    }

    #[test]
    fn prior_draws_from_fit_refuses_an_empty_estimate() {
        // A fit config whose [estimate] is empty samples nothing: every row is
        // empty and the cells all run at the [fixed] values.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir = format!("{}/../../../ocaml/golden/sir_basic.ir.json", manifest);
        let (model, _) = util::load_ir(&ir).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let fit = dir.path().join("empty.toml");
        std::fs::write(&fit, format!(
            "store = \"res\"\n[model]\ncamdl = \"{ir}\"\n[estimate]\n[fixed]\n\
             beta = 0.3\ngamma = 0.1\nN0 = 1000\nI0 = 10\n")).unwrap();
        let err = generate_prior_draws(&crate::out::Out::new(true), fit.to_str().unwrap(), 3, 1, &model).unwrap_err();
        assert!(err.starts_with("--draws prior: the fit config's [estimate] is empty"),
            "unexpected error: {err}");
    }

    /// Write a minimal IR JSON string to a tempfile and return its path.
    /// Lets tests exercise the prior-draws code paths without spinning up
    /// the compiler or committing hand-crafted fixtures.
    fn write_ir_fixture(json: &str) -> (tempfile::TempDir, String) {
        // Fixtures carry the `__IR_VERSION__` sentinel for the envelope version;
        // rewrite it to the build's current IR_VERSION so a schema bump never
        // staleness-breaks these in-code fixtures (they load via the
        // envelope-checked `ir::from_str`).
        let json = json.replace("__IR_VERSION__", ir::IR_VERSION.trim());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.ir.json");
        std::fs::write(&path, &json).unwrap();
        (dir, path.to_string_lossy().into_owned())
    }

    /// Minimal IR with a single scalar parameter carrying the supplied
    /// bounds and prior JSON. Used by the bounds-rejection and scenario
    /// tests that need tight control over the IR. gh#audit-C8: wrapped
    /// in IR envelope so it parses through the new ir::from_str path.
    fn ir_with_prior(name: &str, bounds: &str, prior_json: &str, extras: &str) -> String {
        format!(r#"{{
          "ir_version": "__IR_VERSION__",
          "validated_by": "test-fixture",
          "model": {{
            "name": "t", "version": "0.3", "time_unit": "days",
            "description": null, "origin": null,
            "compartments": [{{ "name": "S", "kind": "integer" }}],
            "transitions": [], "ode_equations": [], "time_functions": [],
            "tables": [], "interventions": [], "observations": [],
            "parameters": [
              {{ "name": "{name}",
                 "value": {{ "mode": "estimated", "bounds": {bounds},
                             "prior": {{ "dist": {prior_json} }}, "transform": "identity" }},
                 "param_kind": "rate", "param_dim": null }}
              {extras}
            ],
            "initial_conditions": {{ "S": {{ "deterministic": {{ "const": 1.0 }} }} }},
            "output": {{ "times": {{ "at_times": [0.0, 1.0] }},
                         "trajectory": true, "observations": false }},
            "simulation": {{ "t_start": 0.0, "t_end": 1.0,
                             "dt": null, "rng_seed": null }},
            "presets": [], "model_structure": null, "balance": null
          }}
        }}"#)
    }

    #[test]
    fn prior_draws_well_calibrated_no_rejections() {
        // log_normal(mu=-1, sigma=0.5) centered at median ~0.37 with tails
        // well inside [0.01, 2.0]. Should produce draws with 0 rejections.
        let ir = ir_with_prior("beta", "[0.01, 2.0]",
            r#"{ "log_normal": { "mu": -1.0, "sigma": 0.5 } }"#, "");
        let (_dir, path) = write_ir_fixture(&ir);
        let draws = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 100, 42).unwrap();
        assert_eq!(draws.len(), 100);
        for row in &draws {
            let v = row["beta"];
            assert!((0.01..=2.0).contains(&v), "{} out of bounds", v);
        }
    }

    #[test]
    fn prior_draws_pathological_mismatch_errors() {
        // log_normal(mu=5, sigma=0.1) is concentrated near exp(5) ≈ 148,
        // far above the bound [0.01, 2.0]. Rejection sampling hits the
        // 256-attempt cap and errors.
        let ir = ir_with_prior("beta", "[0.01, 2.0]",
            r#"{ "log_normal": { "mu": 5.0, "sigma": 0.1 } }"#, "");
        let (_dir, path) = write_ir_fixture(&ir);
        let err = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 1, 42).unwrap_err();
        assert!(err.contains("beta"), "error should name 'beta': {}", err);
        assert!(err.contains("[0.01, 2]") || err.contains("[0.01, 2.0]"),
            "error should cite bounds: {}", err);
        assert!(err.contains("256 attempts"), "error should cite attempt cap: {}", err);
        assert!(err.contains("outside the parameter bounds"),
            "error should explain the mismatch: {}", err);
    }

    #[test]
    fn prior_draws_respect_bounds_after_truncation() {
        // Moderate mismatch: normal(0, 1) with bounds [0, 1] rejects ~half.
        // Every accepted sample must still be in bounds.
        let ir = ir_with_prior("beta", "[0.0, 1.0]",
            r#"{ "normal": { "mean": 0.0, "sd": 1.0 } }"#, "");
        let (_dir, path) = write_ir_fixture(&ir);
        let draws = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 50, 42).unwrap();
        for row in &draws {
            let v = row["beta"];
            assert!((0.0..=1.0).contains(&v),
                "truncation must keep all draws in bounds, got {}", v);
        }
    }

    #[test]
    fn prior_draws_omit_a_scenario_pinned_param() {
        // beta has a prior; N0 has no prior and no default, and a scenario
        // called 'baseline' sets N0.
        // gh#audit-C8: wrap in IR envelope.
        let json = r#"{
          "ir_version": "__IR_VERSION__",
          "validated_by": "test-fixture",
          "model": {
            "name": "t", "version": "0.3", "time_unit": "days",
            "description": null, "origin": null,
            "compartments": [{ "name": "S", "kind": "integer" }],
            "transitions": [], "ode_equations": [], "time_functions": [],
            "tables": [], "interventions": [], "observations": [],
            "parameters": [
              { "name": "beta",
                "value": { "mode": "estimated", "bounds": [0.01, 2.0],
                           "prior": { "dist": { "log_normal": { "mu": -1.0, "sigma": 0.3 } } },
                           "transform": "identity" },
                "param_kind": "rate", "param_dim": null },
              { "name": "N0",
                "value": { "mode": "estimated", "bounds": [100.0, 10000.0],
                           "prior": "flat", "transform": "identity" },
                "param_kind": "count", "param_dim": null }
            ],
            "initial_conditions": { "S": { "deterministic": { "const": 1.0 } } },
            "output": { "times": { "at_times": [0.0, 1.0] },
                        "trajectory": true, "observations": false },
            "simulation": { "t_start": 0.0, "t_end": 1.0,
                            "dt": null, "rng_seed": null },
            "scenarios": [
              { "name": "baseline", "label": "default",
                "params": { "N0": 1000.0 },
                "scale": {}, "enable": [], "disable": [], "compose": [] }
            ],
            "model_structure": null, "balance": null
          }
        }"#;
        let (_dir, path) = write_ir_fixture(json);

        // gh#949: the row carries the sampled `beta` and never `N0`. N0 is
        // the scenario's to set, per cell — a row value would sit at the draw
        // tier and hand one arm's scenario value to every other arm.
        let draws = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 5, 42).unwrap();
        assert_eq!(draws.len(), 5);
        for row in &draws {
            assert!(!row.contains_key("N0"), "N0 has no prior; the row must omit it: {:?}", row);
            let b = row["beta"];
            assert!((0.01..=2.0).contains(&b), "beta out of bounds: {}", b);
        }
    }

    /// Large-batch summary statistics from sample_from_prior_raw.
    /// Regression guard for parameterization bugs (e.g., accidentally
    /// using shape/scale instead of shape/rate for Gamma).
    #[test]
    fn sample_from_prior_raw_matches_expected_moments() {
        use ir::parameter::{PriorDist, UniformPrior, NormalPrior, LogNormalPrior,
            HalfNormalPrior, BetaPrior, GammaPrior, ExponentialPrior};
        let n = 50_000usize;
        let mut rng = sim::rng::StatefulRng::new(20260416);

        // Helper: draw n samples, return (mean, variance).
        let mut moments = |pd: &PriorDist| -> (f64, f64) {
            let xs: Vec<f64> = (0..n).map(|_| sample_from_prior_raw(pd, &mut rng)).collect();
            let mean = xs.iter().sum::<f64>() / (n as f64);
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64);
            (mean, var)
        };

        // Uniform(0, 4): E=2, Var=4/12*4^2=16/12 ≈ 1.333
        let (m, v) = moments(&PriorDist::Uniform(UniformPrior { lower: 0.0, upper: 4.0 }));
        assert!((m - 2.0).abs() < 0.05, "uniform mean {}", m);
        assert!((v - 16.0/12.0).abs() < 0.05, "uniform var {}", v);

        // Normal(3, 0.5): E=3, Var=0.25
        let (m, v) = moments(&PriorDist::Normal(NormalPrior { mean: 3.0, sd: 0.5 }));
        assert!((m - 3.0).abs() < 0.02, "normal mean {}", m);
        assert!((v - 0.25).abs() < 0.02, "normal var {}", v);

        // LogNormal(mu=0, sigma=0.5): E = exp(mu + sigma²/2) = exp(0.125) ≈ 1.1331
        //                              Var = (exp(sigma²) - 1) * exp(2 mu + sigma²)
        let (m, v) = moments(&PriorDist::LogNormal(LogNormalPrior { mu: 0.0, sigma: 0.5 }));
        let expected_mean = (0.125_f64).exp();
        let expected_var = ((0.25_f64).exp() - 1.0) * (0.25_f64).exp();
        assert!((m - expected_mean).abs() < 0.05, "lognormal mean {} (exp {})", m, expected_mean);
        assert!((v - expected_var).abs() < 0.1, "lognormal var {} (exp {})", v, expected_var);

        // HalfNormal(sigma=1): E = sigma*sqrt(2/π) ≈ 0.7979
        //                      Var = sigma² * (1 - 2/π) ≈ 0.3634
        let (m, v) = moments(&PriorDist::HalfNormal(HalfNormalPrior { sigma: 1.0 }));
        let exp_m = (2.0_f64 / std::f64::consts::PI).sqrt();
        let exp_v = 1.0 - 2.0 / std::f64::consts::PI;
        assert!((m - exp_m).abs() < 0.02, "half_normal mean {}", m);
        assert!((v - exp_v).abs() < 0.02, "half_normal var {}", v);

        // Beta(2, 5): E = α/(α+β) = 2/7 ≈ 0.2857
        //              Var = αβ/((α+β)²(α+β+1)) ≈ 0.02551
        let (m, v) = moments(&PriorDist::Beta(BetaPrior { alpha: 2.0, beta: 5.0 }));
        assert!((m - 2.0/7.0).abs() < 0.01, "beta mean {}", m);
        assert!((v - 2.0*5.0/(49.0*8.0)).abs() < 0.005, "beta var {}", v);

        // Gamma(shape=3, rate=2): E = k/r = 1.5, Var = k/r² = 0.75.
        // This specifically catches shape/scale vs shape/rate confusion:
        // if we had used scale = 2 by mistake, the mean would be 6, not 1.5.
        let (m, v) = moments(&PriorDist::Gamma(GammaPrior { shape: 3.0, rate: 2.0 }));
        assert!((m - 1.5).abs() < 0.02, "gamma mean {} (should be 1.5, not 6.0!)", m);
        assert!((v - 0.75).abs() < 0.03, "gamma var {}", v);

        // Exponential(rate=0.5): E = 1/rate = 2, Var = 1/rate² = 4
        let (m, v) = moments(&PriorDist::Exponential(ExponentialPrior { rate: 0.5 }));
        assert!((m - 2.0).abs() < 0.05, "exponential mean {}", m);
        assert!((v - 4.0).abs() < 0.2, "exponential var {}", v);

        // LogUniform(1e-2, 1e2): log θ ~ U(ln 1e-2, ln 1e2) = U(-ln100, ln100).
        // E[θ] = (U - L)/(ln U - ln L) = (100 - 0.01)/(2·ln100) ≈ 10.857.
        // Every draw must land strictly inside [1e-2, 1e2].
        use ir::parameter::{LogUniformPrior, TruncatedNormalPrior};
        let lu = PriorDist::LogUniform(LogUniformPrior { lower: 1e-2, upper: 1e2 });
        let xs: Vec<f64> = (0..n).map(|_| sample_from_prior_raw(&lu, &mut rng)).collect();
        assert!(xs.iter().all(|&x| (1e-2..=1e2).contains(&x)), "log_uniform draw out of support");
        let lu_mean = xs.iter().sum::<f64>() / n as f64;
        let exp_lu_mean = (1e2 - 1e-2) / (2.0 * (100.0_f64).ln());
        assert!((lu_mean - exp_lu_mean).abs() / exp_lu_mean < 0.05,
            "log_uniform mean {} (exp {})", lu_mean, exp_lu_mean);

        // TruncatedNormal(0.7, 0.2) on [0.3, 1.0]: every draw inside bounds;
        // mean shifts below 0.7 because the upper tail is clipped harder.
        let tn = PriorDist::TruncatedNormal(TruncatedNormalPrior { mean: 0.7, sd: 0.2, lower: 0.3, upper: 1.0 });
        let xs: Vec<f64> = (0..n).map(|_| sample_from_prior_raw(&tn, &mut rng)).collect();
        assert!(xs.iter().all(|&x| (0.3..=1.0).contains(&x)), "truncated_normal draw out of support");
        let tn_mean = xs.iter().sum::<f64>() / n as f64;
        // Analytic truncated-normal mean μ + σ(φ(α)−φ(β))/(Φ(β)−Φ(α)),
        // α=(0.3−0.7)/0.2=−2, β=(1.0−0.7)/0.2=1.5 ⇒ ≈ 0.6834.
        assert!((tn_mean - 0.6834).abs() < 0.01, "truncated_normal mean {} (≈0.6834 expected)", tn_mean);
    }

    /// gh#155: log_uniform and truncated_normal draw exactly inside their
    /// support via inverse-CDF — `sample_with_bounds` never rejects, so the
    /// "X% mass outside declared bounds (N rejected)" warning never fires.
    /// This is the issue's headline efficiency claim. Contrast: a plain
    /// normal(2.0, 0.5) on [0.3, 1.0] would reject ~98% and hit the cap.
    #[test]
    fn new_priors_sample_without_rejection() {
        use ir::parameter::{PriorDist, LogUniformPrior, TruncatedNormalPrior};
        let mut rng = sim::rng::StatefulRng::new(99);

        // log_uniform: bounds == support → every draw in range, 0 rejections.
        let lu = PriorDist::LogUniform(LogUniformPrior { lower: 1e-5, upper: 1e-2 });
        for _ in 0..5000 {
            let (v, rej) = sample_with_bounds(&lu, Some((1e-5, 1e-2)), &mut rng, "kappa").unwrap();
            assert_eq!(rej, 0, "log_uniform must never reject");
            assert!((1e-5..=1e-2).contains(&v), "log_uniform draw {} out of support", v);
        }

        // truncated_normal with MOST mass outside the bounds (mean 2.0 well
        // above the [0.3, 1.0] support): inverse-CDF still lands in-support
        // with zero rejections, where normal+rejection would fail.
        let tn = PriorDist::TruncatedNormal(
            TruncatedNormalPrior { mean: 2.0, sd: 0.5, lower: 0.3, upper: 1.0 });
        for _ in 0..5000 {
            let (v, rej) = sample_with_bounds(&tn, Some((0.3, 1.0)), &mut rng, "take").unwrap();
            assert_eq!(rej, 0, "truncated_normal must never reject (inverse-CDF)");
            assert!((0.3..=1.0).contains(&v), "truncated_normal draw {} out of support", v);
        }
    }

    #[test]
    fn prior_draws_different_seeds_produce_different_draws() {
        let ir = ir_with_prior("beta", "[0.01, 10.0]",
            r#"{ "log_normal": { "mu": 0.0, "sigma": 1.0 } }"#, "");
        let (_dir, path) = write_ir_fixture(&ir);
        let a = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 5, 42).unwrap();
        let b = generate_prior_draws_from_ir(&crate::out::Out::new(true), &path, 5, 137).unwrap();
        // At least one row must differ — the probability of two independent
        // 5-draw sequences from a continuous prior being bit-identical is
        // vanishingly small (and would indicate a seeding bug).
        assert!(a.iter().zip(b.iter()).any(|(x, y)| x["beta"] != y["beta"]),
            "different seeds should produce different draws");
    }

    #[test]
    fn seed_derivation_deterministic() {
        use util::{mix_cell_seed, SEED_MIX_OBS};
        let seed = 42u64;
        let draw_idx = 3u64;
        let rep = 7u64;
        let s1 = mix_cell_seed(seed, draw_idx, rep);
        let s2 = mix_cell_seed(seed, draw_idx, rep);
        assert_eq!(s1, s2, "same inputs must produce same seed");

        // Different draw_idx → different seed
        let s3 = mix_cell_seed(seed, 4, rep);
        assert_ne!(s1, s3);

        // Obs seed independent
        let obs1 = s1 ^ SEED_MIX_OBS;
        assert_ne!(s1, obs1);
    }

    // ─── gh#86: --draws prior must honor model-IR ~ priors as a fallback
    // when the fit toml doesn't declare priors. Sibling of gh#75
    // (which did this for `camdl fit run`). ──────────────────────────

    /// Write a minimal fit.toml that exercises `generate_prior_draws`.
    ///
    /// Only the surface that the function actually reads is filled in
    /// (model, estimate, fixed). `Problem::load` parses the toml without
    /// running validate(); a file with no `[method]` at all is a complete
    /// problem for this reader.
    fn write_fit_toml_for_prior_draws(
        dir: &std::path::Path,
        model_ir_path: &str,
        estimate_block: &str,
        fixed_block: &str,
    ) -> String {
        let toml = format!(r#"
[model]
camdl = "{model}"

[estimate]
{estimate}

[fixed]
{fixed}

[method]
algorithm = "if2"
backend = "chain_binomial"
chains = 1
particles = 10
iterations = 1
cooling = 0.7
"#,
            model = model_ir_path,
            estimate = estimate_block,
            fixed = fixed_block,
        );
        let p = dir.join("fit.toml");
        std::fs::write(&p, toml).unwrap();
        p.to_string_lossy().into_owned()
    }

    /// gh#949: a fit config's `[fixed]` values come back separately and never
    /// in a draw row — a row resolves at the draw tier, above `--params`,
    /// while `[fixed]` belongs at the `fit.toml [fixed]` tier below it.
    #[test]
    fn prior_draws_with_fit_toml_return_fixed_outside_the_rows() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = util::load_ir(&ir_path).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let estimate = "\
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
";
        let fit_path = write_fit_toml_for_prior_draws(
            dir.path(), &ir_path, estimate, "I0 = 10.0\nN0 = 5000.0");
        let (draws, fixed) = generate_prior_draws(&crate::out::Out::new(true), &fit_path, 4, 42, &model).unwrap();
        assert_eq!(draws.len(), 4);
        for row in &draws {
            let mut cols: Vec<&str> = row.keys().map(|k| k.as_str()).collect();
            cols.sort();
            assert_eq!(cols, vec!["beta", "gamma"],
                "a row carries the [estimate] parameters only");
        }
        assert_eq!(fixed.get("I0"), Some(&10.0));
        assert_eq!(fixed.get("N0"), Some(&5000.0));
        assert_eq!(fixed.len(), 2);
    }

    /// gh#86 RED test 1: fit.toml lists every estimated param but
    /// supplies NO `prior = { ... }` blocks. The model IR (sir_priors
    /// golden) has `~ <dist>` declarations for every param. After the
    /// fix, the IR's priors satisfy the requirement and N draws come
    /// back with the sampled values clamped to bounds.
    #[test]
    fn prior_draws_with_fit_toml_falls_back_to_ir_priors_when_missing() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = util::load_ir(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // Every param in [estimate] but no `prior = { ... }` block.
        // sir_priors has 5 params: beta, gamma, rho, N0, I0. Fit-toml
        // bounds match the model's declared bounds.
        let estimate = "\
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
rho   = { bounds = [0.001, 1.0] }
N0    = { bounds = [100, 1000000] }
I0    = { bounds = [1, 1000] }
";
        // No [fixed] entries (the table itself is required but can be empty).
        let fit_path = write_fit_toml_for_prior_draws(
            dir.path(), &ir_path, estimate, "");

        let (draws, _fixed) = generate_prior_draws(&crate::out::Out::new(true), &fit_path, 7, 42, &model)
            .expect("after gh#86: should fall back to IR priors");
        assert_eq!(draws.len(), 7);
        for row in &draws {
            for name in ["beta", "gamma", "rho", "N0", "I0"] {
                let v = row.get(name).unwrap_or_else(|| panic!("missing {}", name));
                assert!(v.is_finite(), "{} must be finite, got {}", name, v);
                assert!(*v >= 0.0, "{} must be non-negative, got {}", name, v);
            }
            assert!(row["beta"] >= 0.01 && row["beta"] <= 2.0,
                "beta out of fit-toml bounds: {}", row["beta"]);
            assert!(row["rho"] >= 0.001 && row["rho"] <= 1.0,
                "rho out of fit-toml bounds: {}", row["rho"]);
        }
    }

    // ─── gh#158: `simulate --fit` config-load error carries a hint ───────

    /// A `--fit` file with no `[model]` table fails the config
    /// deserialize. The raw serde message is opaque ("missing field
    /// `model`"); the wrapped error must add the `[model]` shape hint so
    /// the user knows what kind of file `simulate --fit` wants.
    #[test]
    fn simulate_fit_malformed_config_error_carries_hint() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = util::load_ir(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // A plausible "bare params" file a user might pass by mistake:
        // no [model] table at all.
        let bad = dir.path().join("bad.toml");
        std::fs::write(&bad, "beta = 0.4\ngamma = 0.2\n").unwrap();

        let err = generate_prior_draws(&crate::out::Out::new(true), &bad.to_string_lossy(), 5, 1, &model)
            .expect_err("a fit file with no [model] table must fail to load");
        assert!(err.contains("simulate --fit") && err.contains("fit-config TOML"),
            "wrapped error must explain what simulate --fit expects: {}", err);
        assert!(err.contains("[model]") && err.contains("camdl ="),
            "wrapped error must show the [model] table shape: {}", err);
        assert!(err.contains("camdl docs fit-toml"),
            "wrapped error must point at the docs: {}", err);
    }

    /// Control: a well-formed fit-config loads cleanly, so the
    /// `simulate --fit` hint must NOT appear in the (success) path. A
    /// bare assertion that the load succeeds is the negative control for
    /// the wrapper firing only on the deserialize-error branch.
    #[test]
    fn simulate_fit_valid_config_no_hint() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = util::load_ir(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let estimate = "\
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
rho   = { bounds = [0.001, 1.0] }
N0    = { bounds = [100, 1000000] }
I0    = { bounds = [1, 1000] }
";
        let fit_path = write_fit_toml_for_prior_draws(
            dir.path(), &ir_path, estimate, "");

        // Loads and draws — the wrapper's hint never enters this path.
        let (draws, _fixed) = generate_prior_draws(&crate::out::Out::new(true), &fit_path, 3, 7, &model)
            .expect("well-formed fit-config must load");
        assert_eq!(draws.len(), 3);
    }

    /// gh#86 RED test 2: only error when NEITHER the fit toml NOR the
    /// model IR supplies a prior for an estimated param. Build a model
    /// with one estimable param missing a `~` prior; the fit toml also
    /// omits priors. The error must name THAT one param.
    #[test]
    fn prior_draws_errors_only_when_neither_fit_toml_nor_ir_has_a_prior() {
        // Hand-rolled IR: `beta` has a log_normal prior, `gamma` has none.
        let ir_json = r#"{
          "ir_version": "__IR_VERSION__",
          "validated_by": "test-fixture",
          "model": {
            "name": "t", "version": "0.3", "time_unit": "days",
            "description": null, "origin": null,
            "compartments": [{ "name": "S", "kind": "integer" }],
            "transitions": [], "ode_equations": [], "time_functions": [],
            "tables": [], "interventions": [], "observations": [],
            "parameters": [
              { "name": "beta",
                "value": { "mode": "estimated", "bounds": [0.01, 2.0],
                           "prior": { "dist": { "log_normal": { "mu": -1.0, "sigma": 0.3 } } },
                           "transform": "identity" },
                "param_kind": "rate", "param_dim": null },
              { "name": "gamma",
                "value": { "mode": "estimated", "bounds": [0.05, 1.0],
                           "prior": "flat", "transform": "identity" },
                "param_kind": "rate", "param_dim": null }
            ],
            "initial_conditions": { "S": { "deterministic": { "const": 1.0 } } },
            "output": { "times": { "at_times": [0.0, 1.0] },
                        "trajectory": true, "observations": false },
            "simulation": { "t_start": 0.0, "t_end": 1.0,
                            "dt": null, "rng_seed": null },
            "presets": [], "model_structure": null, "balance": null
          }
        }"#;
        let (_ir_dir, ir_path) = write_ir_fixture(ir_json);
        let (model, _) = util::load_ir(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // Both beta and gamma declared, neither has a prior.
        let estimate = "\
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
";
        let fit_path = write_fit_toml_for_prior_draws(
            dir.path(), &ir_path, estimate, "");

        let err = generate_prior_draws(&crate::out::Out::new(true), &fit_path, 3, 1, &model).unwrap_err();
        // After fix: only gamma is missing (beta resolves via IR ~).
        assert!(err.contains("gamma"),
            "error must name gamma (no prior in either source). Got:\n{}", err);
        assert!(!err.contains("beta,") && !err.contains(": beta") && !err.contains(" beta "),
            "error must NOT name beta (resolves via IR `~` prior). Got:\n{}", err);
    }

    /// gh#86 RED test 3: when both sources declare a prior, the fit
    /// toml's prior wins (tier 1 > tier 2). Model declares
    /// `~ normal(0, 1)`; fit toml declares
    /// `prior = { log_normal = { mu = 5, sigma = 0.01 } }`.
    /// Draws must cluster around exp(5) ≈ 148, not around 0.
    #[test]
    fn prior_draws_fit_toml_prior_wins_over_ir_prior() {
        // beta declared with normal(0, 1) — very narrow around 0.
        let ir_json = r#"{
          "ir_version": "__IR_VERSION__",
          "validated_by": "test-fixture",
          "model": {
            "name": "t", "version": "0.3", "time_unit": "days",
            "description": null, "origin": null,
            "compartments": [{ "name": "S", "kind": "integer" }],
            "transitions": [], "ode_equations": [], "time_functions": [],
            "tables": [], "interventions": [], "observations": [],
            "parameters": [
              { "name": "beta",
                "value": { "mode": "estimated", "bounds": [-1000.0, 1000.0],
                           "prior": { "dist": { "normal": { "mean": 0.0, "sd": 1.0 } } },
                           "transform": "identity" },
                "param_kind": "rate", "param_dim": null }
            ],
            "initial_conditions": { "S": { "deterministic": { "const": 1.0 } } },
            "output": { "times": { "at_times": [0.0, 1.0] },
                        "trajectory": true, "observations": false },
            "simulation": { "t_start": 0.0, "t_end": 1.0,
                            "dt": null, "rng_seed": null },
            "presets": [], "model_structure": null, "balance": null
          }
        }"#;
        let (_ir_dir, ir_path) = write_ir_fixture(ir_json);
        let (model, _) = util::load_ir(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // Fit-toml says log_normal(mu=5, sigma=0.01) → exp(5) ≈ 148.41.
        let estimate = "\
beta = { bounds = [-1000.0, 1000.0], \
         prior = { log_normal = { mu = 5.0, sigma = 0.01 } } }
";
        let fit_path = write_fit_toml_for_prior_draws(
            dir.path(), &ir_path, estimate, "");

        // N=200 draws gives a tight sample mean: log_normal(5, 0.01)
        // has E[X] = exp(mu + sigma^2/2) ≈ exp(5.00005) ≈ 148.42 and
        // SD ≈ E[X] * sigma ≈ 1.48. With N=200 the SE on the mean is
        // ~0.1, so a tolerance of 1.0 is well outside chance.
        let (draws, _fixed) = generate_prior_draws(&crate::out::Out::new(true), &fit_path, 200, 42, &model)
            .expect("fit-toml prior should sample successfully");
        let mean: f64 = draws.iter().map(|r| r["beta"]).sum::<f64>()
            / (draws.len() as f64);
        let expected = (5.0_f64 + 0.01_f64.powi(2) / 2.0).exp();
        assert!((mean - expected).abs() < 1.0,
            "fit-toml log_normal(5, 0.01) prior must win; sample mean {} \
             should be near {} (not 0 from the IR's normal(0, 1)). \
             gh#86 regression guard.",
            mean, expected);
        // Sanity: no draw near 0 — at this concentration, samples are
        // all > 100.
        for row in &draws {
            assert!(row["beta"] > 100.0,
                "every draw should be near exp(5), got {}", row["beta"]);
        }
    }
}
