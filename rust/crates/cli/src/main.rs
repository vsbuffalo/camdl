mod args;
#[cfg(test)]
mod test_support;   // collision-free unique_temp_dir for unit tests (gh#153)
mod util;
mod params_resolver;  // unified parameter-value resolver (2026-05-25 CLI UX rev 2)
mod cas_read;       // generic RunRecord reader (new-format sims); transitional alongside run_meta (gh#147)
mod cas_index;      // derived run_id→leaf index + `camdl reindex` (gh#147 M4)
mod hashing;
mod resolve;        // Resolve bridge: CLI inputs → runid identity (CAS run-identity refactor, gh#147)
mod output_schema;  // run.json output_schema: column roles for tabular outputs (proposal 2026-07-15)
mod run_meta;       // cross-cutting run-metadata value types (FitAlgorithm, Backend, provenance records, FitSidecar)
mod posterior_draws; // resolve a fit run's canonical posterior draws (--draws posterior, fit predict)
mod chain_selection; // read-side --exclude-chains: the one chain filter over a posterior cloud
mod quantile; // shared quantile reduction + numeric formatting (proposal 2026-08-11 §3.6)
mod quantities_file; // `--quantities FILE`: a separable reporting vocabulary + its artifact key
mod quantity_output; // generated-quantities banding + tidy-TSV rendering (shared by fit predict + simulate)
mod obs_anchor;     // gh#616: runtime resolution of a model's observation anchors
mod emit_every;     // gh#656: `--emit-every`, the per-stream emission-cadence override
mod run_paths;      // canonical output-path helpers
mod cas;
mod browse;
mod check_update;   // `camdl check-update` — GitHub release-availability check
mod sampling;
mod sim_job;       // SimulateJob / ParamSource / Seeds / ScenarioRef / ObsOutput (run-spec §3)
mod engine;        // run_job: the single engine behind simulate + batch run (run-spec §3.1)
mod batch;
mod eval;
mod pfilter;        // used internally by fit runner for data loading
mod pfilter_cas;    // gh#147 (M3.3): pfilter-eval CAS identity (model/config/params/seed)
mod caltime_load;   // dated-data loader: column detection + date→internal-time (2026-05-22)
mod data;
mod docs;           // `camdl docs <topic>` — embedded, version-locked usage docs
mod fit;
mod compare;
mod if2;
mod profile;
mod profile_cas;     // gh#147 M3.3: profile-point CAS identity (resolve_profile_point)
mod profile_diagnostics;
mod progress;
mod status;        // tidy colored one-shot milestone lines (compiled/storing/stored)
mod evidence;
mod survey;
mod survey_cas;     // gh#147 (M3.3): survey CAS identity (model/config/box/seed)
mod sim_ensemble_cas; // gh#147: multi-cell simulate ensemble CAS identity (model/config/params/grid)
mod landscape_html;
mod lineage;        // three-layer lineage: --event-log record + realize + tree
mod mre;            // `camdl mre` — minimal-reproducible-example bundles (gh#212)
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
use sim::{write_diagnostics_tsv, warn_zero_firings};
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

    /// Run a batch sweep or check its status
    #[command(subcommand)]
    Batch(BatchCmd),

    /// Inference pipeline (MLE, posterior sampling, evaluation)
    #[command(subcommand)]
    Fit(FitCmd),

    /// Standalone bootstrap particle filter at fixed parameters
    Pfilter(args::PfilterArgs),

    /// (removed) IF2 is now a one-stage fit — see `camdl fit run`
    ///
    /// Hidden from the command list: kept only so `camdl if2` / `camdl mif2`
    /// print an actionable migration message instead of a bare clap error.
    #[command(alias = "mif2", hide = true)]
    If2(args::If2Args),

    /// Profile likelihood via parallel IF2 over a parameter grid
    Profile(args::ProfileArgs),

    /// Likelihood-landscape diagnostic via Latin-hypercube sampling.
    ///
    /// Diagnostic, NOT a fitting routine — answers "is my model
    /// identifiable from this data?" before burning hours on IF2.
    /// See `camdl survey --help` for full notes on when to trust
    /// the output and the known limitations.
    Survey(args::SurveyArgs),

    /// Data utilities
    #[command(subcommand)]
    Data(DataCmd),

    /// Browse cached runs as a table
    List(args::ListArgs),

    /// Show full metadata for a cached run
    Show(args::ShowArgs),

    /// Emit trajectory or observation output from a cached run
    Cat(args::CatArgs),

    /// Compare fits by prequential scores (elpd, CRPS, PIT)
    Compare(args::CompareArgs),

    /// Set or update the user-display label on any run (sim, fit, profile, …)
    Label(args::LabelArgs),

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
Run `camdlc inspect` with no arguments for usage.

Common options (all parsed by camdlc):
  --summary           Compartments / transitions / parameters overview
  --dims              Show declared dimensions and their levels
  --compartments      List compartments (post-stratification)
  --transitions       List transitions with their rate expressions
  --tables            Show loaded table values
  --forcings          List forcings, and the data file each was read from
  --ascii             Strip ANSI color from output

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
    /// Rebuild the derived run index (`<root>/index.json`)
    Reindex(args::ReindexArgs),

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
  # Run a parameter / scenario sweep declared in a TOML manifest
  camdl batch run sweep.toml --parallel 8

  # Check completion of a long-running sweep
  camdl batch status sweep.toml

See `camdl batch <subcommand> --help` for full options."))]
pub(crate) enum BatchCmd {
    /// Run a batch sweep from a TOML manifest
    Run(args::BatchArgs),
    /// Show status of a batch sweep
    Status(args::BatchStatusArgs),
}

#[derive(Subcommand)]
#[command(arg_required_else_help = true,
          after_help = colored_help!("\
Examples:
  # Run the full inference pipeline declared in fit.toml
  camdl fit run fit.toml --seed 1

  # Render the convergence + MLE table for a completed fit
  camdl fit summary results/fits/he2010-abc123/

  # Browse every fit under a results tree
  camdl fit table results/fits/

See `camdl fit <subcommand> --help` for full options."))]
pub(crate) enum FitCmd {
    /// Run inference stages defined in a fit.toml
    Run(args::FitRunArgs),
    /// Render a single-fit interpretation summary (Â, gate verdict, MLE table)
    Summary(args::FitSummaryArgs),
    /// Compare two fit.toml configs
    Diff(args::FitDiffArgs),
    /// Cross-fit aggregator: walk results/fits/, render one row per fit
    Table(args::FitTableArgs),
    /// Derive a new fit.toml from an existing one
    New(args::FitNewArgs),
    /// Write the free-forward posterior predictive (predicted-vs-observed) artifact
    Predict(args::FitPredictArgs),
    /// List supported (algorithm, backend) pairs and their descriptions
    Methods,
}

#[derive(Subcommand)]
#[command(arg_required_else_help = true,
          after_help = colored_help!("\
Examples:
  # Split a data TSV into training + holdout sets
  camdl data split cases.tsv --at-time 100 \\
      --train train.tsv --holdout holdout.tsv

See `camdl data split --help` for full options."))]
pub(crate) enum DataCmd {
    /// Split a data TSV into train and holdout sets
    Split(args::DataSplitArgs),
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

fn main() {
    // Hidden parse-only drift check (used by `make test-cli-docs`). Returns
    // an exit code and short-circuits before any real parsing/execution.
    if let Some(code) = run_check_args_mode() {
        std::process::exit(code);
    }

    let cli = Cli::parse();

    // Resolve the effective verbosity. Precedence:
    //   1. Explicit --verbosity wins over everything.
    //   2. Else, if `--progress plain` is in effect (or `auto` on
    //      non-TTY stderr), bump to `info` — plain-mode progress
    //      lines are emitted via `log::info!` and would otherwise
    //      be silently filtered by the default `warn` threshold
    //      (GH #14, comment re: silent plain mode).
    //   3. Else, RUST_LOG env → else `warn`.
    //
    // Note — a cleaner long-term design (option 2 in Vince's GH #14
    // comment) would route progress through a dedicated non-`log::*`
    // channel, making "progress visibility" independent of "log
    // filter." That decouples user-facing progress from
    // developer-facing logging, which is the right mental model but
    // a bigger refactor; this auto-bump is the minimal fix.
    let progress_wants_info = match cli.progress {
        args::types::ProgressMode::Plain => true,
        args::types::ProgressMode::Auto =>
            !std::io::IsTerminal::is_terminal(&std::io::stderr()),
        args::types::ProgressMode::Pretty | args::types::ProgressMode::None => false,
    };
    let effective_verbosity: log::LevelFilter = cli.verbosity.unwrap_or(
        if progress_wants_info { log::LevelFilter::Info } else { log::LevelFilter::Warn }
    );

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(effective_verbosity.as_str())
    ).init();

    // Progress output policy (GH #14). Must run after env_logger so that
    // plain-mode log lines from callbacks reach the configured filter.
    // `--no-progress` is the discoverable shorthand for `--progress none` and
    // wins when both are given.
    progress::init(if cli.no_progress { args::types::ProgressMode::None } else { cli.progress });
    util::set_ir_cache_disabled(cli.no_ir_cache);
    util::set_licm_disabled(cli.no_licm);

    match cli.command {
        Command::Simulate(a)            => run_simulate(&a),
        Command::Batch(BatchCmd::Run(a))    => batch::cmd_batch_run(&a),
        Command::Batch(BatchCmd::Status(a)) => batch::cmd_batch_status(&a),
        Command::Fit(FitCmd::Run(a))    => fit::cmd_fit_run_v2(&a),
        Command::Fit(FitCmd::Summary(a))=> fit::cmd_fit_summary(&a),
        Command::Fit(FitCmd::Diff(a))   => fit::cmd_fit_diff(&a),
        Command::Fit(FitCmd::Table(a))  => fit::cmd_fit_table(&a),
        Command::Fit(FitCmd::New(a))    => fit::cmd_fit_new(&a),
        Command::Fit(FitCmd::Predict(a)) => fit::predict::cmd_fit_predict(&a),
        Command::Fit(FitCmd::Methods)   => fit::cmd_fit_methods(),
        Command::Label(a)               => fit::cmd_label(&a),
        Command::Pfilter(a)             => pfilter::cmd_pfilter(&a),
        Command::If2(a)                 => if2::cmd_if2(&a),
        Command::Profile(a)             => profile::cmd_profile(&a),
        Command::Survey(a)              => survey::cmd_survey(&a),
        Command::Data(DataCmd::Split(a))=> data::cmd_data_split(&a),
        Command::Lineage(LineageCmd::Realize(a)) => lineage::cmd_lineage_realize(&a),
        Command::Lineage(LineageCmd::Tree(a)) => lineage::cmd_lineage_tree(&a),
        Command::Lineage(LineageCmd::Sojourn(a)) => lineage::cmd_lineage_sojourn(&a),
        Command::Lineage(LineageCmd::Cohort(a)) => lineage::cmd_lineage_cohort(&a),
        Command::List(a)                => browse::cmd_list(&a),
        Command::Show(a)                => browse::cmd_show(&a),
        Command::Cat(a)                 => browse::cmd_cat(&a),
        Command::Compare(a)             => compare::cmd_compare(&a),
        Command::Check(a) => {
            let mut refs = vec!["check"];
            refs.extend(a.args.iter().map(String::as_str));
            util::delegate_to_camdlc(&refs).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        Command::Inspect(a) => {
            let mut refs = vec!["inspect"];
            refs.extend(a.args.iter().map(String::as_str));
            util::delegate_to_camdlc(&refs).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        Command::Render(a) => {
            let mut refs = vec!["render"];
            refs.extend(a.args.iter().map(String::as_str));
            util::delegate_to_camdlc(&refs).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        Command::Docs(a)                => docs::cmd_docs(&a),
        Command::Mre(MreCmd::Fit(a))      => mre::cmd_mre_fit(&a),
        Command::Mre(MreCmd::Simulate(a)) => mre::cmd_mre_simulate(&a),
        Command::CheckUpdate            => check_update::cmd_check_update(),
        Command::Dev(DevCmd::Reindex(a)) => cmd_reindex(&a),
        Command::Dev(DevCmd::Eval(a))    => eval::cmd_eval(&a),
        Command::Dev(DevCmd::Compile(a)) => {
            let refs: Vec<&str> = a.args.iter().map(String::as_str).collect();
            util::delegate_to_camdlc(&refs).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        Command::Dev(DevCmd::Doctest(a)) => {
            let mut refs = vec!["doctest"];
            refs.extend(a.args.iter().map(String::as_str));
            util::delegate_to_camdlc(&refs).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
    }
}

// Per-(point, replicate) seed derivation lives in one place,
// `util::mix_cell_seed` (with `util::SEED_MIX_OBS` for the obs stream);
// `engine::run_job` and `survey` both route through it. The constants
// below are for the *other*, unrelated RNG streams owned by this module.
const SEED_MIX_UNIFORM: u64 = 0xd4a5_b1ce;      // uniform draws RNG
const SEED_MIX_PRIOR: u64  = 0x0014_b1ce;      // prior draws RNG

/// `camdl reindex`: rebuild `<root>/index.json` from a fresh full walk of
/// every `run.json` under the store. Drops entries for leaves no longer on
/// disk and adds every leaf found.
fn cmd_reindex(a: &args::ReindexArgs) {
    match cas_index::rebuild(&a.root) {
        Ok(n) => println!("reindexed {} run(s) under {}", n, a.root.display()),
        Err(e) => {
            eprintln!("error: failed to write index at {}: {}", a.root.display(), e);
            std::process::exit(1);
        }
    }
}

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
    fit_ref: &str,
    ir_path_compiled: &str,
    n_draws: Option<usize>,
) -> (Vec<HashMap<String, f64>>, crate::sim_job::InitStateSource) {
    let (model, _) = util::load_model(ir_path_compiled).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let columns = io::trajectories::TrajColumnSpec::from_model(&model, &[]);
    // The draws authority for this fit. `simulate` surfaces no `--exclude-chains`
    // today, so no selection is attached and the cloud is the whole posterior;
    // attaching one here is the entire change if the flag ever lands (gh#695).
    let pref = crate::posterior_draws::resolve_posterior_draws(fit_ref, None)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
    let ens = crate::fit::joint::resolve_forecast_ensemble(&pref, &columns)
        .unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

    if ens.draws.is_empty() {
        eprintln!(
            "error: --init-state fit: {fit_ref} has no forkable posterior draws \
             (0/{} carry a saved latent path). The paired (θ_i, X_i(T)) origin comes \
             from the smoothed latent paths a PGAS stage writes to \
             <stage>/chain_*/trajectories.tsv; PMMH and particle-filter stages write \
             none, and a stage run with `n_trajectories = 0` writes none either.\n  \
             Refusing rather than starting from `init {{}}`: that would look like a \
             forecast and be a free-forward replay.\n  Fix: re-fit with a PGAS stage \
             (or raise `n_trajectories`), or forecast at a single θ with \
             `camdl pfilter --save-final-state` + `--init-state <file>`.",
            ens.n_total,
        );
        std::process::exit(1);
    }

    let n_forkable = ens.draws.len();
    eprintln!(
        "simulate: --init-state fit → {n_forkable}/{} posterior draws have a saved \
         latent path; forecasting that paired (θ_i, X_i) subset from t = {}{}\n  \
         stage '{}' ({})",
        ens.n_total,
        ens.origin_t,
        if n_forkable < ens.n_total {
            format!(" (the other {} draws have no state to fork)", ens.n_total - n_forkable)
        } else {
            String::new()
        },
        ens.stage,
        ens.draws_path.display(),
    );

    // Same strided cap as the unpaired posterior path (never front-biased): a
    // large forkable subset is still hours of forward solves.
    let cap = n_draws.unwrap_or(crate::fit::predict::DEFAULT_PREDICT_DRAWS);
    if let Some(asked) = n_draws {
        if asked > n_forkable {
            eprintln!(
                "simulate: --init-state fit → -n {asked} exceeds the {n_forkable} \
                 forkable draws; forecasting all {n_forkable}."
            );
        }
    }
    let selected: Vec<crate::fit::joint::ForecastDraw> = if n_forkable > cap {
        let picked: Vec<crate::fit::joint::ForecastDraw> =
            crate::fit::predict::subsample_draws(&ens.draws, cap)
                .into_iter().cloned().collect();
        eprintln!(
            "simulate: --init-state fit → subsampling {} of {n_forkable} forkable \
             draws (strided across the subset; raise with --n-draws)",
            picked.len()
        );
        picked
    } else {
        ens.draws
    };

    // The identity input: a content digest over the ensemble ACTUALLY used —
    // origin time, and every selected draw's key + restored values, in
    // selection order. Content, not the fit's run_id: keying on provenance
    // means enumerating every knob that changes the selection, and a missed one
    // is two different clouds sharing a cache entry.
    let origin_t = runid::FiniteF64::new(ens.origin_t).unwrap_or_else(|e| {
        eprintln!("error: --init-state fit: non-finite forecast origin ({e})");
        std::process::exit(1);
    });
    let rows: Vec<runid::inputs::InitStateRow> = selected
        .iter()
        .map(|d| runid::inputs::InitStateRow {
            chain: d.chain as u64,
            draw: d.draw as u64,
            counts: d.counts.clone(),
            reals: d.reals.iter()
                .map(|&v| runid::FiniteF64::new(v).unwrap_or_else(|e| {
                    eprintln!(
                        "error: --init-state fit: (chain {}, draw {}) has a non-finite \
                         real compartment at the origin ({e})",
                        d.chain, d.draw
                    );
                    std::process::exit(1);
                }))
                .collect(),
        })
        .collect();
    let digest = runid::ContentAddressed::content_hash(
        &runid::inputs::InitStateEnsemble { origin_t, rows },
    );

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

fn run_simulate(a: &args::SimulateArgs) {
    let _eval_stats_guard = crate::util::EvalStatsReportGuard::start();
    sim::eval_stats::set_allow_degenerate_rates(a.allow_degenerate_rates);  // gh#audit-C6
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
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        };
    let (ir_path_compiled, _ir_tmp) =
        util::resolve_ir_path_with_quantities(&ir_path, false, quantities_override.as_ref())
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
    // gh#156: `--output-every` rewrites the compiled IR's output schedule once,
    // so BOTH the engine (loads by path) and the CAS identity (`base_model`,
    // also loaded from this path) see the overridden cadence. `_every_tmp`
    // holds the rewritten temp alive for the function scope.
    let (ir_path_compiled, _every_tmp) =
        util::rematerialize_with_output_every(&ir_path_compiled, a.output_view.every)
            .unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
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
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    // Track explicit vs default flags for the backend-provenance guardrail.
    // Option<_> fields mean None ↔ not explicitly passed.
    let backend_explicit = a.backend.backend.is_some();
    let dt_explicit      = a.backend.dt.is_some();
    // Default is chain_binomial so `simulate` and `fit` agree at the
    // same MLE params (see docs/dev/incidents/2026-04-19-backend-default-mismatch.md).
    let mut backend      = a.backend.backend.unwrap_or(args::types::ForwardBackend::ChainBinomial);
    // dt precedence (gh#161): an explicit `--dt` always wins. Otherwise the
    // model's own `simulate { dt = … }` is the default (dt is a model knob).
    // If neither is set, fall back to 1.0. A fit-provenance `dt` (below) can
    // still override the model default when a fit-MLE params file is passed and
    // `--dt` was not given, so the consumer reproduces the fit's step.
    let model_dt: Option<f64> = util::peek_simulation_dt(&ir_path_compiled);
    let mut dt           = a.backend.dt.or(model_dt).unwrap_or(1.0_f64);
    let seed             = a.seed;
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
            eprintln!("error: {e}");
            std::process::exit(1);
        });
    let adhoc_enable: Vec<String>  = a.enable.clone();
    let adhoc_disable: Vec<String> = a.disable.clone();
    let seeds: Vec<u64> = match &a.seeds {
        Some(spec) => spec.expand(),
        None       => vec![a.seed],
    };
    let seeds_spec_given = a.seeds.is_some();
    let output_path: Option<String>  = a.output.as_ref().map(|p| p.to_string_lossy().into_owned());
    let mut obs_path: Option<String> = a.obs.as_ref().map(|p| p.to_string_lossy().into_owned());
    let mut obs_dir: Option<String>  = a.obs_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let obs_only: Option<String>     = a.obs_only.as_ref().map(|p| p.to_string_lossy().into_owned());
    let obs_only_dir: Option<String> = a.obs_only_dir.as_ref().map(|p| p.to_string_lossy().into_owned());
    let replicates: usize            = a.replicates.unwrap_or(1);
    let draws_path: Option<String>   = a.draws.clone();
    let n_draws_arg: Option<usize>   = a.n_draws;
    let fit_path_for_draws: Option<String> = a.fit.as_ref().map(|p| p.to_string_lossy().into_owned());
    let dry_run     = a.dry_run;
    // Content-addressed storage is the default for every `simulate` run.
    // `--cas` is accepted (and ignored) for compatibility; `--output`/`--obs`
    // are additive mirrors of the store, never replacements.
    let _ = a.cas;
    let output_dir_arg: Option<String> = Some(a.output_dir.to_string_lossy().into_owned());

    // --obs-only implies --obs or --obs-dir (infer from path: trailing / or existing dir → obs-dir)
    if let Some(ref path) = obs_only {
        if obs_path.is_some() || obs_dir.is_some() {
            eprintln!("error: --obs-only cannot be combined with --obs or --obs-dir");
            std::process::exit(1);
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
            eprintln!("error: --obs-only-dir cannot be combined with --obs, --obs-dir, or --obs-only");
            std::process::exit(1);
        }
        obs_dir = Some(path.clone());
    }
    let suppress_trajectory = obs_only.is_some() || obs_only_dir.is_some();

    if replicates < 1 {
        eprintln!("error: --replicates must be >= 1");
        std::process::exit(1);
    }

    let want_obs = obs_path.is_some() || obs_dir.is_some();

    // gh#656: `--emit-every` only reaches emitted synthetic observations — the
    // `--obs*` writers, the CAS obs subtree they enable, and an obs-sourced
    // quantity. A run that emits none would silently ignore it, so refuse and
    // name what would make it do something.
    if emit_every.is_some() && !want_obs && a.quantities_out.is_none() {
        eprintln!(
            "error: --emit-every sets the cadence of EMITTED synthetic \
             observations, but this run emits none.\n  \
             Add --obs/--obs-dir/--obs-only/--obs-only-dir to write them, or \
             --quantities-out for an `observations.<stream>` quantity."
        );
        std::process::exit(1);
    }

    if seeds_spec_given && replicates > 1 {
        eprintln!("error: --seeds and --replicates are mutually exclusive.\n  \
                   --seeds provides explicit seed values.\n  \
                   --replicates generates N deterministic seeds from --seed.");
        std::process::exit(1);
    }
    // If using --seeds, replicates tracks seed count
    let replicates = if seeds_spec_given { seeds.len() } else { replicates };

    // Validate mutually exclusive σ flags
    if !scenario_names.is_empty() && (!adhoc_enable.is_empty() || !adhoc_disable.is_empty()) {
        eprintln!("error: --scenario and --enable/--disable are mutually exclusive.");
        eprintln!("  --scenario selects a named scenario from the model file.");
        eprintln!("  --enable/--disable compose an ad-hoc scenario.");
        eprintln!("  To combine both, define a composed scenario in the model file.");
        std::process::exit(1);
    }

    // If no scenarios specified, use a single None (baseline)
    let scenario_list: Vec<Option<String>> = if scenario_names.is_empty() {
        vec![None]
    } else {
        scenario_names.iter().map(|s| Some(s.clone())).collect()
    };

    // ── gh#626: resolve `--to` (obs-anchored horizon override) once, up front.
    // Anchored forms (`last_obs + 8 weeks`) fold over the fit's bound
    // observation data; absolute forms (number, date) resolve data-free. The
    // resolved value overrides every cell's `simulation.t_end` (applied in
    // `resolve_run_model` after the scenario horizon) and is keyed into run
    // identity via `ResolvedEntry.t_end`.
    //
    // gh#616: the MODEL may also be anchored (`simulate { to = last_obs + 4
    // 'weeks }`, a scenario's, or a forcing's `breakpoints`). Those resolve from
    // the same bound data as `--to`, through the same fold, so the observed
    // window is resolved ONCE here and used for both. The model is then
    // substituted in place, so every up-front check below (the horizon
    // ordering, the `--to` conflict rule) reads resolved numbers rather than the
    // NaN the compiler baked.
    let (model_raw, _) = util::load_model(&ir_path_compiled).unwrap_or_else(|e| {
        eprintln!("error: {}", e);
        std::process::exit(1);
    });
    let model_is_anchored = crate::obs_anchor::model_is_anchored(&model_raw);
    let to_is_anchored = matches!(
        a.to.as_deref().map(|raw| crate::fit::runner::parse_time_spec(
            "--to", raw, model_raw.origin.as_deref(), &model_raw.time_unit)),
        Some(Ok(crate::fit::runner::TimeSpec::Anchored(_)))
    );
    let obs_anchors: Option<ir::anchor::ObsAnchorTimes> =
        if model_is_anchored || to_is_anchored {
            let Some(fit_ref) = a.fit.as_ref() else {
                let what = if model_is_anchored {
                    format!("this model is anchored to observed data ({})",
                            crate::obs_anchor::anchored_sites(&model_raw).join(", "))
                } else {
                    "--to is anchored to observed data".to_string()
                };
                eprintln!(
                    "error: {what}, but a forward simulation binds none.\n  \
                     Fix: pass --fit <fit.toml | fit run dir> — its \
                     [data.observations] supplies the observed times the anchor \
                     resolves against."
                );
                std::process::exit(1);
            };
            let (first, last) = resolve_simulate_obs_anchors(&model_raw, fit_ref, dt)
                .unwrap_or_else(|e| {
                    eprintln!("error: resolving observation anchors: {e}");
                    std::process::exit(1);
                });
            Some(ir::anchor::ObsAnchorTimes { first, last })
        } else {
            None
        };
    // The locally-substituted model the up-front checks read. The RUN and the
    // CAS `base_model` are substituted separately, at their own loads, with this
    // same window — see `obs_anchor`'s module doc for why that is the seam.
    let model_to = {
        let mut m = model_raw;
        if let Some(w) = obs_anchors {
            let moved = crate::obs_anchor::substitute(&mut m, w).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                std::process::exit(1);
            });
            crate::obs_anchor::report(&moved, &m);
        }
        m
    };

    let to_was_anchored = to_is_anchored;
    let t_end_override: Option<f64> = match a.to.as_deref() {
        None => None,
        Some(raw) => {
            let spec = crate::fit::runner::parse_time_spec(
                "--to", raw, model_to.origin.as_deref(), &model_to.time_unit,
            ).unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1); });
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
                eprintln!(
                    "error: --to \"{raw}\" resolves to t = {resolved}, which is at or \
                     before the model's t_start = {}. The horizon must lie after \
                     the simulation start.",
                    model_to.simulation.t_start
                );
                std::process::exit(1);
            }
            // Conflict rule (gh#561: never silently discard a declared
            // horizon): a scenario whose composed horizon differs from BOTH
            // the model's t_end and the resolved --to is refused. Equal to
            // the resolved --to is the no-op precedent and allowed.
            for sname in scenario_list.iter().flatten() {
                let h = crate::params_resolver::effective_horizon(&model_to, Some(sname))
                    .unwrap_or_else(|e| { eprintln!("error: {}", e); std::process::exit(1); });
                if h != model_to.simulation.t_end && h != resolved {
                    eprintln!(
                        "error: scenario '{sname}' declares its own horizon \
                         (t = {h}) and --to \"{raw}\" resolves to t = {resolved} — \
                         refusing to pick one silently.\n  Fix: drop the \
                         scenario's `simulate {{ to }}` (keep it as a label-only \
                         preset) or run it without --to."
                    );
                    std::process::exit(1);
                }
            }
            eprintln!("simulate: --to \"{raw}\" → t_end = {resolved}");
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
        eprintln!(
            "error: --init-state fit needs the posterior it pairs against: add \
             --draws posterior --fit <fit results dir>.\n  \
             The fit source restores draw i's OWN terminal latent state X_i(T) \
             under draw i's OWN θ_i — without the posterior draws there is no θ_i \
             to put a state with, and crossing states with unrelated parameters is \
             the incoherent product this source exists to avoid.\n  Fix: add \
             --draws posterior, or pass a `camdl pfilter --save-final-state` file \
             to forecast at a single θ."
        );
        std::process::exit(1);
    }

    // `--fit` without `--draws` is only meaningful when something needs the
    // fit's DATA rather than its posterior: an anchored `--to` (gh#626) or an
    // anchored model (gh#616). Otherwise keep the old ergonomics.
    if a.fit.is_some() && a.draws.is_none() && !to_was_anchored && !model_is_anchored {
        eprintln!("error: --fit requires --draws (or an anchored --to, or an anchored model).");
        std::process::exit(1);
    }

    let cas_root = output_dir_arg.clone()
        .unwrap_or_else(|| run_paths::DEFAULT_OUTPUT_ROOT.to_string());

    // ── Backend-provenance guardrail ─────────────────────────────
    //
    // If any of the params files carries a `[provenance]` block from
    // a fit, apply the three-way matching rule for backend + dt.
    // See docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md
    // and the incident at
    // docs/dev/incidents/2026-04-19-backend-default-mismatch.md.
    //
    // We read the first fit-provenance block found; if the user passes
    // multiple --params files, one can be a fit MLE and others can be
    // standalone overrides, but two conflicting fit-provenance blocks
    // is itself a misconfiguration we'd flag — for the v1 of this
    // feature we stop at the first block and trust single-fit
    // workflows.
    let mut from_fit_hash: Option<String> = None;
    let mut from_fit_params_file: Option<String> = None;
    for pf in &params_files {
        let prov = match crate::fit::provenance::read_mle_provenance(pf) {
            Ok(Some(p)) => p,
            Ok(None) | Err(_) => continue,
        };
        from_fit_hash = prov.fit_hash.clone();
        from_fit_params_file = Some(pf.clone());

        if !backend_explicit {
            // Auto-match path.
            eprintln!("[info] backend auto-matched to {} (dt={}) from fit \
                      provenance in {}. Pass --backend explicitly to override; \
                      the fit's backend is the consistent default for forward \
                      sims of the MLE.",
                prov.backend, prov.dt, pf);
            backend = prov.backend;
            if !dt_explicit { dt = prov.dt; }
        } else if backend != prov.backend {
            // Explicit-differs path — warn.
            eprintln!("warning: backend mismatch.");
            eprintln!("  {} was produced by a fit that used {} (dt={}).",
                pf, prov.backend, prov.dt);
            eprintln!("  You passed --backend {}, which is a different \
                       dynamical model at the same parameters.", backend);
            eprintln!("  The resulting trajectories will NOT reproduce the \
                       fit's behavior — this combination has caused real \
                       confusion; see");
            eprintln!("  docs/dev/incidents/2026-04-19-backend-default-mismatch.md.");
            eprintln!("  If this is intentional (e.g. cross-backend \
                       comparison), ignore this warning.");
        }
        // If backend_explicit and matches: silent. Normal case.

        break;
    }

    let base_sim_run = util::SimRun {
        // The already-compiled IR: every engine cell loads this directly, so
        // `resolve_ir_path` short-circuits (no per-cell camdlc).
        ir_path: ir_path_compiled.clone(),
        params_files,
        overrides,
        // Draw/sweep + inline-scenario tiers are assigned per cell by
        // `engine::build_cell_sim_run`; the base run carries none.
        point_overrides: std::collections::HashMap::new(),
        set_vec_entries,
        table_files,
        scenario_name: None, // set per-scenario in the loop
        t_end_override, // gh#626: keys the CAS identity below; cells get it via the job
        // gh#641: assigned PER CELL by `engine::build_cell_sim_run` (replicate i
        // restores particle row i); the base run carries none.
        init_state: None,
        obs_anchors,    // gh#616: the run's resolved observed window
        adhoc_enable,
        adhoc_disable,
        scenario_inline_name: None,
        scenario_inline_set: Vec::new(),
        scenario_inline_scale: Vec::new(),
        backend,
        dt,
        seed, // overridden per-replicate below
        integrator: a.backend.integrator, // gh#166: CLI --integrator override
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
        let (model_is, _) = util::load_model(&ir_path_compiled).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        // The seam refuses a reactive model too (`chain_binomial.rs`), but by
        // then the user has waited for a compile; name the policy here.
        let reactive: Vec<&str> = model_is.interventions.iter()
            .filter(|iv| iv.fire.is_reactive())
            .map(|iv| iv.name.as_str())
            .collect();
        if !reactive.is_empty() {
            eprintln!(
                "error: --init-state cannot restart a model with reactive \
                 intervention(s) [{}]. A reactive policy carries mid-run state \
                 the origin state does not hold — the observation history its \
                 trigger reads, its once/cooldown gating, the queue of effects \
                 already scheduled, and its own surveillance RNG stream — so the \
                 forecast would silently begin with an empty agenda.\n  Fix: \
                 run the scenario without the reactive policy, or simulate \
                 continuously from the model's own t_start.",
                reactive.join(", ")
            );
            std::process::exit(1);
        }
        if !matches!(backend, crate::args::types::ForwardBackend::ChainBinomial) {
            eprintln!("error: {}", util::unseamed_backend_msg(backend.as_str()));
            std::process::exit(1);
        }

        if let Some(InitStateSourceArg::File(path)) = init_state_arg.as_ref() {
            let columns = io::trajectories::TrajColumnSpec::from_model(&model_is, &[]);
            let bytes = std::fs::read(path).unwrap_or_else(|e| {
                eprintln!("error: cannot read --init-state {}: {e}", path.display());
                std::process::exit(1);
            });
            let states = io::read_final_states(path, &columns).unwrap_or_else(|e| {
                eprintln!("error: --init-state: {e}");
                std::process::exit(1);
            });
            // `--save-final-state` is p(x_T | y_{1:T}) at ONE θ. Pairing those
            // rows with unrelated posterior draws would form an incoherent
            // (θ, x_T) product and read as a legitimate forecast cloud.
            if draws_path.is_some() {
                eprintln!(
                    "error: --init-state <file> cannot be combined with --draws. The \
                     saved particle states are the filtering distribution at the ONE θ \
                     the filter ran at, so pairing row i with an unrelated posterior \
                     draw would forecast a state and a parameter vector that never went \
                     together.\n  Fix: run --init-state <file> at that same θ \
                     (--params), or use `--init-state fit --draws posterior --fit <fit \
                     results dir>`, which restores draw i's own state under draw i's \
                     own θ."
                );
                std::process::exit(1);
            }
            // Replicate i restores row i. Not "the first N rows": a
            // post-resampling swarm is ancestor-ordered, so a prefix is not an
            // exchangeable subsample of the filtering distribution.
            let want = if seeds_spec_given { seeds.len() } else { replicates };
            if want != states.len() {
                eprintln!(
                    "error: --init-state {} holds {} particle rows but this run has {} \
                     replicate(s). Each replicate restores its own row, and a prefix of \
                     a post-resampling swarm is not an exchangeable subsample of the \
                     filtering distribution — so the counts must match.\n  Fix: pass \
                     --replicates {}, or re-run the filter with --particles {}.",
                    path.display(), states.len(), want, states.len(), want
                );
                std::process::exit(1);
            }
            eprintln!(
                "simulate: --init-state {} → {} particle states at t = {}",
                path.display(), states.len(), states.origin_t
            );
            let digest = runid::ContentHash::digest_bytes(&bytes);
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
            eprintln!("error: --obs/--obs-dir requested but model has no observations blocks");
            std::process::exit(1);
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
                None => crate::params_resolver::effective_horizon(&model_check, s.as_deref())
                    .unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    }),
            };
            horizons.push((s.clone().unwrap_or_else(|| "baseline".into()), h));
        }
        if horizons.iter().any(|(_, h)| *h != horizons[0].1) {
            let list: Vec<String> =
                horizons.iter().map(|(n, h)| format!("{n} → t = {h}")).collect();
            eprintln!(
                "error: --obs/--obs-dir cannot combine scenarios with different \
                 horizons ({}): the combined file has one time axis, so the \
                 shorter scenario's rows past its own trajectory would be \
                 fabricated.\n  Fix: run each scenario in its own invocation, or \
                 read the per-cell obs/ artifacts from the store (`camdl cat`).",
                list.join(", ")
            );
            std::process::exit(1);
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
                            eprintln!(
                                "error: --to extends the horizon to t = {new_end}, but \
                                 stream '{}' emits at a fixed list ending at t = {max_t} \
                                 — no synthetic observations would be emitted past it.\n  \
                                 Fix: use a recurring emit schedule (`every …`), extend \
                                 the `at [...]` list, or run without --to.",
                                o.name
                            );
                            std::process::exit(1);
                        }
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
            let obs_origin = init_state_source.as_ref().map(|s| s.origin_t);
            let schedules: Vec<_> = model_check.observations.iter()
                .map(|o| obs_emit_schedule_times(
                    o, obs_origin, obs_end, emit_every.as_ref(),
                ).unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
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
                eprintln!("error: observation streams have different schedules ({}).\n\
                           A single wide TSV cannot hold multi-cadence streams.\n\
                           Use --obs-dir (one file per stream, keeps trajectory) or\n\
                           --obs-only-dir (one file per stream, suppresses trajectory).",
                    descs.join(", "));
                std::process::exit(1);
            }
        }
    }

    // ── Prepare obs-dir output directory ────────────────────────────────────
    if let Some(ref dir) = obs_dir {
        std::fs::create_dir_all(dir).unwrap_or_else(|e| {
            eprintln!("error: cannot create obs directory '{}': {}", dir, e);
            std::process::exit(1);
        });
    }

    // The display label rides on every leaf's `RunRecord.provenance.label`.
    let label_arg: Option<String> = match a.label.as_deref() {
        Some(raw) => match crate::fit::validate_label(raw) {
            Ok(l) => Some(l),
            Err(e) => {
                eprintln!("error: invalid --label: {}", e);
                std::process::exit(1);
            }
        },
        None => None,
    };
    // Fit → sim lineage. When `--params` consumed a fit-MLE file carrying a
    // `[provenance]` block, record the upstream fit on every sim leaf's
    // `RunRecord.deps` (an `ArtifactRef`): the fit's content hash as the
    // upstream identity, plus the consumed params file's content digest (so a
    // regenerated fit — different θ̂ — produces a different dep). This is
    // provenance only — a sim's identity is its factored levels
    // (resolve_trajectory), never `deps` — so it does not change the run_id or
    // store path. Empty when no fit-provenance params file was supplied.
    let fit_dep: Vec<runid::inputs::ArtifactRef> = match (&from_fit_hash, &from_fit_params_file) {
        (Some(hash), Some(pf)) => {
            match runid::ContentHash::from_hex(hash) {
                Ok(run_id) => {
                    let digest = std::fs::read(pf)
                        .map(|b| runid::ContentHash::digest_bytes(&b))
                        .unwrap_or(run_id);
                    let artifact = std::path::Path::new(pf)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| pf.clone());
                    vec![runid::inputs::ArtifactRef {
                        run_id,
                        kind: runid::ArtifactKind::FitStage,
                        artifact,
                        digest,
                    }]
                }
                // A non-hex fit_hash (legacy / malformed provenance) is not a
                // CAS identity we can record — skip the dep rather than fold a
                // bogus run_id. The backend-guardrail above still fired.
                Err(_) => Vec::new(),
            }
        }
        _ => Vec::new(),
    };

    // ── Load draws if --draws is specified ─────────────────────────────────
    let draws: Vec<HashMap<String, f64>> = if let Some(ref source) = draws_path {
        if source == "uniform" {
            let n = n_draws_arg.unwrap_or_else(|| {
                eprintln!("error: --draws uniform requires -n N");
                std::process::exit(1);
            });
            generate_uniform_draws(&ir_path_compiled, n, seed).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            })
        } else if source == "prior" {
            let n = n_draws_arg.unwrap_or_else(|| {
                eprintln!("error: --draws prior requires -n N");
                std::process::exit(1);
            });
            match fit_path_for_draws.as_ref() {
                Some(fit_path) => {
                    // fit.toml prior source (overrides or supplements
                    // model priors). gh#86: the model IR is the
                    // tier-2 fallback when the fit toml omits a prior
                    // for a parameter that declares `~ <dist>` in the
                    // model file.
                    let (draws_model, _) = util::load_model(&ir_path_compiled).unwrap_or_else(|e| {
                        eprintln!("error loading model for --draws prior: {}", e);
                        std::process::exit(1);
                    });
                    generate_prior_draws(fit_path, n, seed, &draws_model).unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    })
                }
                None => {
                    // Use priors embedded in the model IR. Scenarios that
                    // set parameter values fill in "default values" for
                    // params without priors, matching the simulation runtime
                    // semantics.
                    let scenarios: Vec<&str> = scenario_names.iter()
                        .map(|s| s.as_str()).collect();
                    generate_prior_draws_from_ir(&ir_path_compiled, n, seed, &scenarios).unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    })
                }
            }
        } else if source == "posterior" {
            // Resolve the fit run's canonical post-warm-up draws cloud (the
            // terminal Bayesian stage's draws.tsv). --fit names the fit results
            // directory here, not a config TOML.
            let fit_ref = fit_path_for_draws.as_ref().unwrap_or_else(|| {
                eprintln!("error: --draws posterior requires --fit <fit results dir> \
                    (the directory `camdl fit run` printed)");
                std::process::exit(1);
            });
            if matches!(init_state_arg, Some(InitStateSourceArg::Fit)) {
                let (rows, source) = resolve_paired_posterior(
                    fit_ref, &ir_path_compiled, a.n_draws,
                );
                init_state_source = Some(std::sync::Arc::new(source));
                rows
            } else {
            let resolved = posterior_draws::resolve_posterior_draws(fit_ref, None)
                .unwrap_or_else(|e| {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                });
            let loaded = load_draws_tsv(&resolved.draws_path.to_string_lossy())
                .unwrap_or_else(|e| {
                    eprintln!("error loading posterior draws {}: {}",
                        resolved.draws_path.display(), e);
                    std::process::exit(1);
                });
            let method_label = resolved.method
                .map(|m| m.as_str())
                .unwrap_or("posterior");
            // gh#630 (ebola F35): never silently replay the full cloud — a
            // 60k-draw posterior is hours of forward solves and a 21 MB
            // --draws-out. Same strided default cap as `fit predict`
            // (never front-biased), raised or lowered with --n-draws.
            let cap = a.n_draws.unwrap_or(crate::fit::predict::DEFAULT_PREDICT_DRAWS);
            let total = loaded.len();
            let loaded: Vec<HashMap<String, f64>> = if total > cap {
                let picked: Vec<HashMap<String, f64>> =
                    crate::fit::predict::subsample_draws(&loaded, cap)
                        .into_iter().cloned().collect();
                eprintln!(
                    "draws: posterior — subsampling {} of {total} draws (strided \
                     across the cloud; raise with --n-draws)",
                    picked.len()
                );
                picked
            } else {
                loaded
            };
            eprintln!("draws: posterior — {} draws from {} stage '{}' ({})",
                loaded.len(), method_label, resolved.stage, resolved.draws_path.display());
            loaded
            }
        } else {
            // File path. #273: when --fit is supplied, backfill any parameter
            // absent from the draws columns from the fit's [fixed] block, never
            // overwriting a column the file provides (a raw posterior trace tail
            // carries only the estimated columns).
            let mut loaded = load_draws_tsv(source).unwrap_or_else(|e| {
                eprintln!("error loading draws: {}", e);
                std::process::exit(1);
            });
            if let Some(fit_ref) = fit_path_for_draws.as_ref() {
                let fixed = posterior_draws::resolve_fixed_for_backfill(fit_ref)
                    .unwrap_or_else(|e| {
                        eprintln!("error: --fit [fixed] backfill: {}", e);
                        std::process::exit(1);
                    });
                let filled = posterior_draws::backfill_fixed(&mut loaded, &fixed);
                if !filled.is_empty() {
                    let names: Vec<&str> = filled.iter().map(|s| s.as_str()).collect();
                    eprintln!("draws: backfilled {} fixed parameter(s) from {}: {}",
                        filled.len(), fit_ref, names.join(", "));
                }
            }
            loaded
        }
    } else {
        // No draws — single point (parameters come from --params / --param)
        vec![HashMap::new()]
    };
    let n_draws = draws.len();

    // ── Persist sampled draws (gh#157) ──────────────────────────────────────
    // `--draws-out PATH` materializes the sampled θ-per-draw as a TSV the
    // `--draws PATH` loader reads back (one row per draw, one column per
    // parameter). Opt-in only: absent the flag nothing is written, so the
    // content-addressed store leaves are untouched.
    if let Some(ref out) = a.draws_out {
        let out = out.to_string_lossy().into_owned();
        if let Err(e) = write_draws_tsv(&out, &draws) {
            eprintln!("error writing --draws-out {}: {}", out, e);
            std::process::exit(1);
        }
        eprintln!("draws.tsv: wrote {} draws to {}", n_draws, out);
    }

    let n_scenarios = scenario_list.len();
    let total_runs = n_draws * replicates * n_scenarios;
    if total_runs > 1 {
        let parts: Vec<String> = [
            if n_draws > 1 { Some(format!("{} draws", n_draws)) } else { None },
            if n_scenarios > 1 { Some(format!("{} scenarios", n_scenarios)) } else { None },
            if replicates > 1 { Some(format!("{} replicates", replicates)) } else { None },
        ].iter().flatten().cloned().collect();
        eprintln!("{} = {} runs", parts.join(" × "), total_runs);
    }

    // ── Dry run ─────────────────────────────────────────────────────────────
    if dry_run {
        print_dry_run(
            &ir_path, &ir_path_compiled, base_sim_run.backend, dt, seed,
            &base_sim_run.params_files, &base_sim_run.overrides,
            &scenario_list, &seeds, &draws_path,
            n_draws, replicates, total_runs,
            &obs_path, &obs_dir, &obs_only,
        );
        return;
    }

    // ── Build the SimulateJob and route through the unified engine ──────────
    //
    // `simulate` and `batch run` converge on `engine::run_job` (run-spec
    // §3.1). The wide-format trajectory + combined-obs output shape lives
    // in `StreamSink`; the engine owns the cell loop, seed arithmetic, and
    // per-cell SimRun construction (determinism PIN: tests/determinism_pin.rs).
    //
    // Scenario mapping (mirrors the pre-unification per-cell SimRun):
    //   - `--scenario a,b`  → [Named("a"), Named("b")]  (preset path).
    //   - no `--scenario`   → a single ad-hoc Inline carrying the CLI
    //     --enable/--disable (or empty), so the baseline path keeps
    //     `scenario_name = None` exactly as before.
    use crate::sim_job::{ObsOutput, ParamSource, ScenarioRef, Seeds, SimulateJob};
    let scenarios: Vec<ScenarioRef> = if scenario_names.is_empty() {
        vec![ScenarioRef::Inline {
            name: "baseline".to_string(),
            enable: base_sim_run.adhoc_enable.clone(),
            disable: base_sim_run.adhoc_disable.clone(),
            params: indexmap::IndexMap::new(),
        }]
    } else {
        scenario_names.iter().map(|n| ScenarioRef::Named(n.clone())).collect()
    };

    // ParamSource: --draws yields Draws rows; otherwise a single Point run
    // `replicates` times. With explicit --seeds the seed-list length is the
    // replicate count (the engine ignores `Point.replicates` then), so passing
    // `replicates` is correct in both cases.
    let source = if let Some(ref src) = draws_path {
        let rows: Vec<indexmap::IndexMap<String, f64>> = draws.iter()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .collect();
        // A user-authored draws FILE (not the generated `uniform`/`prior`/
        // `posterior` sources). Carried as `Some(path)` so a scenario ×
        // draws-file column collision is a hard error naming the file (vs.
        // generated draws, where the scenario simply wins).
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

    // ObsOutput from the resolved obs_path/obs_dir + suppression. (The
    // --obs-only / --obs-only-dir flags were already normalised into
    // obs_path/obs_dir + `suppress_trajectory` above.)
    let obs_mode = if let Some(ref p) = obs_path {
        if suppress_trajectory { ObsOutput::OnlyFile(p.into()) } else { ObsOutput::File(p.into()) }
    } else if let Some(ref d) = obs_dir {
        if suppress_trajectory { ObsOutput::OnlyDir(d.into()) } else { ObsOutput::Dir(d.into()) }
    } else {
        ObsOutput::None
    };

    let job = SimulateJob {
        // The pre-compiled IR: each engine cell loads this directly (no
        // per-cell camdlc — see the resolve-once at the top of the function).
        model: ir_path_compiled.clone(),
        params_files: base_sim_run.params_files.clone(),
        backend,
        dt,
        integrator: a.backend.integrator, // gh#166: CLI --integrator override
        source,
        scenarios,
        t_end_override,
        init_state: init_state_source,
        obs_anchors,
        seeds: job_seeds,
        cli_overrides: base_sim_run.overrides.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        set_vec_entries: base_sim_run.set_vec_entries.clone(),
        table_files: base_sim_run.table_files.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        obs: obs_mode,
        // simulate keeps its historical sequential, in-order streaming
        // (combined wide-format output is order-sensitive). Parallelism is
        // the batch path's concern.
        parallel: 1,
    };

    // ── Build the per-cell CAS sink (the system of record) ──────────────────
    //
    // Content-addressed storage is the default. Each engine cell (scenario ×
    // param-point × replicate) commits its own leaf, byte-identical to the
    // leaf `camdl batch run` writes for the same (model, config, params,
    // scenario, process_seed) cell — both go through the SAME `CasSink`
    // (resolve identity via `resolve::resolve_trajectory`, then
    // `commit_atomic`, plus the per-cell obs child). The `--output`/`--obs`
    // mirror is layered on top by `StreamSink` (below).
    let cas_sink = match build_simulate_cas_sink(
        &base_sim_run,
        &ir_path,
        &scenario_names,
        &cas_root,
        want_obs || obs_only.is_some() || obs_only_dir.is_some()
            || obs_path.is_some() || obs_dir.is_some(),
        a.force,
        label_arg.clone(),
        fit_dep,
        a.allow_degenerate_rates,
        output_cols.clone(),
        emit_every.clone(),
        total_runs,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error preparing CAS: {}", e);
            std::process::exit(1);
        }
    };

    // The trajectory mirror, accumulated in memory. The canonical bytes live
    // in the store (per-cell leaves + multi-run ensemble); this buffer is the
    // convenience view, written to `-o PATH` post-loop and (multi-run) stored
    // verbatim as the `SimEnsemble` artifact. NEVER stdout (Item C): with no
    // `-o` the user gets the `cached:` stderr line + `camdl cat <hash>`.
    // `None` ⟺ trajectory suppressed (`--obs-only`, run-spec §3.1.1).
    let traj_out: Option<Vec<u8>> = if !job.obs.suppresses_trajectory() {
        Some(Vec::new())
    } else {
        None
    };

    // The combined-obs / wide-format mirror sink. Its ObsOutput is derived
    // from `job.obs` (run-spec §3.1.1, single source of truth).
    let stream = StreamSink {
        traj_out,
        traj_header_written: false,
        output_cols: output_cols.clone(),
        dates: a.dates,
        dates_render: None,
        obs_path: job.obs.file_path().map(|p| p.to_string_lossy().into_owned()),
        obs_dir: job.obs.dir_path().map(|p| p.to_string_lossy().into_owned()),
        obs_data: Vec::new(),
        obs_stream_names: Vec::new(),
        obs_times_cache: Vec::new(),
        emit_every: emit_every.clone(),
        total_runs: 1,
        n_scenarios: 1,
        n_draws: 1,
    };

    // ── Generated quantities (proposal 2026-06-25) ──────────────────────────
    // Build the accumulator iff the model declares a `quantities {}` block. With
    // `--quantities-out` it emits a regenerated sidecar (never in the CAS leaf,
    // never in the run identity); without it, a one-line note and skip. Point vs
    // band is keyed by the param-source kind (fixed params + single cell → point;
    // a `--draws`/multi-cell run → band), not the cell count alone.
    let quant: Option<SimQuantities> = {
        let (q_model, _) = util::load_model(&ir_path_compiled).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        if q_model.quantities.is_empty() {
            None
        } else if let Some(out_dir) = a.quantities_out.clone() {
            // Scenario is a DESIGN coordinate, draws/replicates/seeds are
            // SAMPLING coordinates; the accumulator keys on the former so a
            // quantile is only ever taken over the latter (gh#562, proposal §1).
            //
            // `Mode` is keyed on the param-source KIND, never the cell count —
            // the cell count includes the scenario factor, so counting it would
            // report a two-point "posterior band" over a baseline and its
            // counterfactual. A `--draws` file with a single row must still
            // band.
            let mode = if matches!(
                job.source,
                crate::sim_job::ParamSource::Point { replicates: 1 }
            ) {
                crate::quantity_output::Mode::Point
            } else {
                crate::quantity_output::Mode::Banded
            };
            Some(SimQuantities {
                quantities: q_model.quantities.clone(),
                mode,
                scenario_axis: !scenario_names.is_empty(),
                out_dir,
                vocabulary: quantities_override.clone(),
                compiled: None,
                eval: None,
                by_scenario: indexmap::IndexMap::new(),
                calendar: io::CalendarMeta::from_model(&q_model),
                emit_every: emit_every.clone(),
            })
        } else {
            let n = q_model.quantities.len();
            eprintln!(
                "note: model declares {n} quantit{}; pass --quantities-out <dir> to emit them",
                if n == 1 { "y" } else { "ies" }
            );
            None
        }
    };

    let mut sink = SimSink { cas: cas_sink, stream, skip_cas: a.stdout, quant };
    // The `--event-log` branch drives the sink's `RunSink` methods directly
    // (one recorded cell), so the trait must be in scope here.
    use engine::RunSink as _;

    // ── Lineage event-log path (three-layer architecture, Layer 1) ──────────
    // `--event-log` records the identity-free event log. It is single-run only
    // (conflicts with --seeds / --replicates / --draws, enforced by clap), so
    // the grid is exactly one cell. The recorder is passive — it draws no
    // randomness — so the trajectory (and thus the leaf's run_id and
    // `traj.tsv` bytes) is byte-identical to a plain `simulate` at the same
    // seed (Tier 2a). The recorded log rides on the `CellResult` and `CasSink`
    // writes it into the SAME leaf as `event_log.tsv`, alongside `traj.tsv`.
    // We reuse `plan_grid` (not a hand-rolled spec) so the cell — and its
    // run_id — match the normal path exactly. `--event-log PATH` additionally
    // mirrors the log to PATH (symmetric with `-o` for the trajectory).
    let recorded: Option<(sim::lineage::EventLog, bool)> = if a.event_log.is_some() {
        let (specs, grid) = engine::plan_grid(&job);
        if specs.len() != 1 {
            eprintln!("error: --event-log is single-run only (no --seeds / \
                       --replicates / --draws / multiple --scenario)");
            std::process::exit(1);
        }
        sink.on_start(&grid);
        let spec = specs.into_iter().next().expect("len checked == 1");
        let (traj, model, event_log, exact) =
            util::run_simulation_event_log(&spec.sim_run).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
        let cell = engine::CellResult { spec, traj, model, event_log: Some(event_log) };
        sink.merge_cell(&cell).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        sink.on_finish(&grid).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        // Recover the log for the optional mirror + realize hint (merge_cell
        // only borrowed the cell, so we can move it back out here).
        cell.event_log.map(|el| (el, exact))
    } else {
        engine::run_job(&job, &mut sink).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        None
    };

    // The combined wide-format trajectory bytes (None ⟺ --obs-only). The CAS
    // leaves/ensemble are the system of record; this buffer is the mirror —
    // written to `-o PATH`, or to stdout under `--stdout` (the store opt-out).
    let combined_traj: Option<Vec<u8>> = sink.stream.traj_out.take();

    // `--stdout`: stream the trajectory to stdout and stop. No leaf was
    // committed (skip_cas), so there is no store, no ensemble, and no banner —
    // just the TSV, ready to pipe.
    if a.stdout {
        if let Some(ref bytes) = combined_traj {
            use std::io::Write;
            if let Err(e) = std::io::stdout().write_all(bytes) {
                eprintln!("error writing trajectory to stdout: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    if let (Some(ref path), Some(ref bytes)) = (&output_path, &combined_traj) {
        if !suppress_trajectory {
            std::fs::write(path, bytes).unwrap_or_else(|e| {
                eprintln!("cannot write {}: {}", path, e);
                std::process::exit(1);
            });
            eprintln!("trajectory written to {}", path);
        }
    }

    // Surface any per-cell CAS commit failures (the store is the system of
    // record; a failed commit is fatal to the run's purpose).
    if !sink.cas.errors.is_empty() {
        for e in &sink.cas.errors {
            eprintln!("error: {}", e);
        }
        std::process::exit(1);
    }

    // ── Store the combined-across-cells TSV as a `SimEnsemble` artifact ──────
    //
    // A multi-cell run (total_runs > 1) keeps its N per-cell `Sim` leaves AND
    // additionally writes the combined wide-format TSV as a content-addressed
    // ensemble that REFERENCES them (deps). The ensemble's artifact bytes are
    // the same `combined_traj` buffer the `-o` mirror uses, so `cat <ensemble>`
    // == the `-o` combined TSV. A single-run simulate writes NO ensemble (the
    // one leaf is the whole thing).
    if total_runs > 1 && !suppress_trajectory {
        if let Some(ref bytes) = combined_traj {
            // The post-bar pause the user sees is this: writing + fsyncing the
            // combined wide-format TSV as the ensemble leaf. Announce it so a
            // multi-MB ensemble doesn't look like a hang.
            crate::status::step("storing",
                format!("ensemble \u{b7} {} cells ({})",
                    total_runs, crate::status::human_bytes(bytes.len() as u64)));
            write_sim_ensemble(&sink.cas, bytes, &cas_root, label_arg.clone());
        }
    }

    // Report where the leaves landed.
    report_cas_leaves(&sink.cas.completed_runs, &cas_root);

    // ── Write the combined observation mirror ───────────────────────────────
    sink.stream.write_obs_output();

    // ── Write generated quantities (regenerated sidecar, NOT in the CAS leaf) ─
    if let Some(q) = &sink.quant {
        if !q.by_scenario.is_empty() {
            // One design cell per scenario, each banded over ITS OWN draws. The
            // `scenario` column is emitted iff the run has a scenario axis: with
            // no `--scenario` there is exactly one cell and no coordinate to
            // report, and labelling it would name a world the run did not
            // simulate (proposal §2.4, §3.3).
            let mut stacked = crate::quantity_output::StackedQuantities::new(q.mode);
            for (scenario, acc) in &q.by_scenario {
                let coords = crate::quantity_output::DesignCoords {
                    scenario: q.scenario_axis.then_some(scenario.as_str()),
                    sweep: &[],
                };
                stacked
                    // `simulate` has no fit behind it, so there is no
                    // conditioned/replay distinction to tag (gh#722) and the
                    // manifest is byte-identical.
                    .push_group(&q.quantities, coords, &acc.draws, &acc.times, None, &q.calendar)
                    .unwrap_or_else(|e| {
                        eprintln!("error rendering quantities: {}", e);
                        std::process::exit(1);
                    });
            }
            let (outs, manifest) = stacked.finish(&q.calendar).unwrap_or_else(|e| {
                eprintln!("error rendering quantities: {}", e);
                std::process::exit(1);
            });
            std::fs::create_dir_all(&q.out_dir).unwrap_or_else(|e| {
                eprintln!("error: cannot create quantities dir {}: {}", q.out_dir.display(), e);
                std::process::exit(1);
            });
            // The vocabulary's content digest keys the artifact (proposal
            // 2026-08-19): the model's own block keeps writing `quantities/`,
            // a supplied one writes `quantities-<key8>/`. Two vocabularies over
            // one run are two tables, not one overwritten twice.
            let sub_dir = crate::quantities_file::quantities_dir_name(q.vocabulary.as_ref());
            for (name, content) in &outs {
                match crate::fit::predict::write_tsv(&q.out_dir, &sub_dir, name, content) {
                    Ok(p) => eprintln!("quantities: wrote {}", p.display()),
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            let manifest = match crate::quantities_file::stamp_provenance(
                &manifest, q.vocabulary.as_ref())
            {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(1);
                }
            };
            let manifest_path = q.out_dir.join(
                crate::quantities_file::quantities_manifest_name(q.vocabulary.as_ref()));
            std::fs::write(&manifest_path, &manifest).unwrap_or_else(|e| {
                eprintln!("error: cannot write {}: {}", manifest_path.display(), e);
                std::process::exit(1);
            });
            eprintln!("quantities: wrote {}", manifest_path.display());
        }
    }

    // ── Event-log mirror + realize hint (Layer 1 → Layer 2) ─────────────────
    // The canonical event log is the `event_log.tsv` artifact in the leaf;
    // `--event-log PATH` additionally mirrors it to PATH. Either way, point
    // the user at the next step (`lineage realize`).
    if let Some((event_log, exact)) = recorded {
        // The leaf's `event_log.tsv` is a real file path `lineage realize` can
        // read directly (run_path is relative to the cas root, includes the
        // `sims/` prefix). gh#241: on a cache HIT the leaf already existed and
        // commit_atomic discarded our freshly-staged event_log.tsv, so the file
        // may be absent — only hand finish_event_log a path that actually
        // exists, so it never points the user at a missing file. (The full fix
        // — event log as a child sub-artifact like obs, or augment-on-recommit
        // — is a store-protocol follow-up.)
        let leaf_event_log = sink.cas.completed_runs.last()
            .map(|e| format!("{}/{}/event_log.tsv", cas_root, e.run_path))
            .filter(|p| std::path::Path::new(p).exists());
        lineage::finish_event_log(a, &event_log, exact, leaf_event_log.as_deref());
    }

    // ── Reactive-log mirror (gh#204) ─────────────────────────────────────────
    // The canonical reactive firing log is the `reactive_log.tsv` artifact in
    // the leaf — always present (declared) when a reactive policy was active, so
    // on a cache hit it is still there. `--reactive-log PATH` mirrors it to PATH
    // (symmetric with `-o` for the trajectory); the leaf stays the system of
    // record. A non-reactive model writes no leaf log, so there is nothing to
    // mirror — say so rather than create an empty file.
    if let Some(ref dest) = a.reactive_log {
        let leaf_log = sink.cas.completed_runs.last()
            .map(|e| format!("{}/{}/reactive_log.tsv", cas_root, e.run_path))
            .filter(|p| std::path::Path::new(p).exists());
        match leaf_log {
            Some(src) => {
                if let Err(e) = std::fs::copy(&src, dest) {
                    eprintln!("warning: --reactive-log mirror to {}: {}", dest.display(), e);
                }
            }
            None => eprintln!(
                "warning: --reactive-log: this run has no active reactive policy \
                 (enable one with a scenario or --enable); nothing to mirror"
            ),
        }
    }
}

/// `RunSink` for `camdl simulate`: writes the per-cell content-addressed
/// leaf (`cas`, shared with `batch run`) AND the `--output`/`--obs` mirror
/// (`stream`, the combined wide-format TSV). The store is the system of
/// record; the mirror is the convenience view. Every planned cell runs —
/// `should_run` is never overridden to skip — because the combined mirror
/// needs every cell's rows; cache hits are handled idempotently by the
/// store's `commit_atomic`.
struct SimSink {
    cas: crate::batch::CasSink,
    stream: StreamSink,
    /// `--stdout`: the user opted out of the store, so skip the leaf commit
    /// entirely — only the `stream` mirror runs, and the caller writes its
    /// buffer to stdout. The store is otherwise the system of record.
    skip_cas: bool,
    /// Generated quantities (proposal 2026-06-25), `Some` iff the model declares
    /// a `quantities {}` block AND `--quantities-out <dir>` was given. The
    /// evaluator is compiled lazily on the first cell (from that cell's
    /// fully-resolved model) and reused; per cell the trajectory is folded into
    /// the leaf's quantity values. Never part of the run identity — a regenerated
    /// sidecar.
    quant: Option<SimQuantities>,
}

/// The per-run generated-quantities accumulator carried on [`SimSink`].
struct SimQuantities {
    /// The model's quantity IR (cloned once), used to build the evaluator and to
    /// render the output.
    quantities: Vec<ir::quantity::Quantity>,
    /// Banded (`--draws`/multi-cell) vs point (single fixed-params cell). Keyed
    /// by the param-source kind, not the cell count alone.
    mode: crate::quantity_output::Mode,
    /// Whether this run HAS a scenario axis (any `--scenario` was passed). A
    /// fact about the run, fixed at construction: with no axis there is no
    /// design coordinate to report, and inventing one would name a world the
    /// run did not simulate (proposal §2.4).
    scenario_axis: bool,
    /// The directory to write the quantity TSVs + manifest into.
    out_dir: std::path::PathBuf,
    /// The supplied reporting vocabulary, `None` when the model's own
    /// `quantities {}` block is in force. Its content digest names the
    /// subdirectory and manifest (`quantities-<key8>/`), so two vocabularies
    /// over one model land at two addresses instead of overwriting each other;
    /// its path + digest go into the manifest as provenance.
    vocabulary: Option<crate::quantities_file::QuantitiesOverride>,
    /// Built lazily on the first `merge_cell` from that cell's resolved model —
    /// the compile happens once per run, never per cell.
    compiled: Option<std::sync::Arc<sim::CompiledModel>>,
    eval: Option<sim::quantity::QuantityEvaluator>,
    /// Per scenario: that scenario's draws and its own time axis.
    ///
    /// Scenario is a DESIGN coordinate — it says which world was simulated —
    /// while draws/replicates/seeds are SAMPLING coordinates within one world.
    /// A quantile summarises the second kind only; taking one across scenarios
    /// averages a baseline and its counterfactual into a ribbon describing
    /// neither (gh#562). Keyed here, at the point cells are assigned, because
    /// that is where the information exists — a renderer handed already-merged
    /// cells cannot un-merge them.
    ///
    /// Insertion order is the engine's canonical order (scenario outermost),
    /// and `run_job` merges strictly in that order even under Rayon, so the
    /// rendered file lists scenarios deterministically.
    by_scenario: indexmap::IndexMap<String, ScenarioQuant>,
    /// Calendar semantics for the `time` axis, stamped into `quantities.json` so
    /// a consumer maps `time → Date` without re-deriving `origin`/`time_unit`.
    calendar: io::CalendarMeta,
    /// gh#656: the `--emit-every` override, so an `observations.<stream>`
    /// quantity reduces the same y_sim series `--obs` emits under the same flags.
    emit_every: Option<crate::emit_every::EmitEvery>,
}

/// One scenario's accumulated quantity draws and the time axis they were
/// produced on. Per scenario, not per run: once a scenario can declare its own
/// horizon (gh#561), a run-global axis would render a September scenario
/// against an August one.
#[derive(Default)]
struct ScenarioQuant {
    /// One inner `Vec` per cell: each quantity leaf's value, in `quantities`
    /// order. Retains derived values, never the trajectory.
    draws: Vec<Vec<sim::quantity::QuantityResult>>,
    times: Vec<f64>,
}

/// Whether two output-time grids agree, on the RELATIVE tolerance the rest of
/// the codebase uses for output times (`chain_binomial.rs` compares
/// `OUTPUT_EPS * t.abs().max(1.0)`). A bare absolute `OUTPUT_EPS` would
/// degenerate to exact equality on a calendar-anchored model, where one ulp at
/// t ~ 1e5 days already exceeds 1e-12.
fn time_grids_match(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x - y).abs() <= sim::schedule::OUTPUT_EPS * x.abs().max(1.0)
        })
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
    restart_origin: Option<f64>,
    // gh#656: the `--emit-every` override, for the same reason — the quantity
    // must reduce over the series `--obs` would emit under the same flags.
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<sim::quantity::ObsSeriesSet, String> {
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
                obs_schedule_times(&s, restart_origin, model.simulation.t_end)
            }
            None => continue, // fit-only — consumes no RNG, mirroring `--obs`
        };
        let sampler =
            sim::inference::obs_model::compile_obs_sample_pf(obs_ir, compiled.clone(), params);
        // `None` — emit_schedule-driven, no data to condition on (gh#702).
        let projected = project_all_obs_times(traj, obs_ir, model, &times, None)?;
        let mut vals = Vec::with_capacity(times.len());
        for (ti, &t) in times.iter().enumerate() {
            let snap = snap_at(traj, t);
            vals.push(sampler(projected[ti], t, &snap.int_state.counts, &[], &mut obs_rng));
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

impl SimQuantities {
    /// Fold one cell's trajectory into its quantity values, compiling the
    /// evaluator on the first call. The per-cell parameter vector is read by
    /// name from the cell's fully-resolved model (every cell shares the param
    /// set, so this is exact for any param source: fixed params, `--params`
    /// files, `--draws`, scenario overrides).
    fn push_cell(&mut self, cell: &engine::CellResult) -> Result<(), String> {
        if self.eval.is_none() {
            let compiled = std::sync::Arc::new(
                sim::CompiledModel::new(cell.model.clone())
                    .map_err(|e| format!("compiling model for quantities: {e:?}"))?,
            );
            let eval = sim::quantity::QuantityEvaluator::new(&self.quantities, compiled.as_ref())
                .map_err(|e| format!("building quantity evaluator: {e}"))?;
            self.compiled = Some(compiled);
            self.eval = Some(eval);
        }
        let compiled = self.compiled.as_ref().expect("compiled set above");
        let eval = self.eval.as_ref().expect("eval set above");
        let mut params = vec![f64::NAN; compiled.param_index.len()];
        for p in &cell.model.parameters {
            if let Some(&idx) = compiled.param_index.get(p.name.as_str()) {
                if let Some(v) = p.value.resolved_value() {
                    params[idx] = v;
                }
            }
        }
        // A `value_at(..., last_obs)` quantity anchors to the end of OBSERVED
        // data; a forward simulation has none (even `--obs` output is synthetic
        // y_sim, not observations). Hard error naming the quantities — the
        // capability-gap convention — rather than an empty/NaN column
        // (proposal 2026-08-17).
        if eval.references_obs_anchor() {
            return Err(format!(
                "quantity `{}` reads `value_at` at an observation anchor \
                 (`last_obs` / `first_obs`, with or without an offset), but a \
                 forward simulation has no observed data to anchor to. Evaluate \
                 it via `fit predict` (where the fit's data supplies the \
                 observation times), or use a literal time: \
                 `value_at(expr, date(\"...\"))`.",
                eval.obs_anchor_quantity_names().join("`, `"),
            ));
        }
        // `observations.<stream>` quantities reduce the per-draw y_sim — sampled
        // here with the cell's obs_seed (the same draws `--obs` would emit). Only
        // built when a quantity actually references an observation stream.
        let obs_streams = eval.obs_streams();
        let obs_set = if obs_streams.is_empty() {
            None
        } else {
            Some(materialize_obs_for_quantities(
                &obs_streams,
                &cell.model,
                &cell.traj,
                compiled,
                &params,
                cell.spec.obs_seed,
                cell.spec.sim_run.init_state.as_ref().map(|i| i.origin_t),
                self.emit_every.as_ref(),
            )?)
        };
        // `simulate` has no fit behind it, so there is no conditioned path to
        // read: every quantity folds the replay (gh#722 `ConditionedRead::Off`).
        let results = eval.eval_draw(
            &params,
            &cell.traj,
            sim::quantity::ConditionedRead::Off,
            compiled,
            obs_set.as_ref(),
            None,
        );
        let times: Vec<f64> = cell.traj.snapshots.iter().map(|s| s.t).collect();
        // The key is DERIVED from the cell, never supplied by a caller — that is
        // what makes pooling unrepresentable here. An accumulator taking a key
        // parameter would be strictly weaker, however well the key were typed,
        // because a caller could pass one key for every cell.
        let scenario = cell.spec.scenario.name();
        let acc = self.by_scenario.entry(scenario.to_string()).or_default();
        if acc.draws.is_empty() {
            acc.times = times;
        } else if !time_grids_match(&acc.times, &times) {
            // Later cells PROVE compatibility rather than being folded into the
            // first cell's axis. Without this, a future feature letting cells of
            // ONE scenario differ in cadence would silently reinstate gh#562 at
            // smaller scale — first-wins, no diagnostic.
            return Err(format!(
                "scenario '{scenario}': cell {} has {} output times but this \
                 scenario's earlier cells have {}. Quantity cells within a \
                 scenario must share a time grid.",
                cell.spec.run_idx,
                times.len(),
                acc.times.len()
            ));
        }
        acc.draws.push(results);
        Ok(())
    }
}

impl engine::RunSink for SimSink {
    fn on_start(&mut self, grid: &engine::Grid) {
        self.stream.on_start(grid);
    }

    fn merge_cell(&mut self, cell: &engine::CellResult) -> Result<(), String> {
        // Leaf first (system of record), then the mirror. A commit error is
        // accumulated on `cas.errors` and surfaced after the loop, never
        // silently dropped. Under `--stdout` the leaf is suppressed.
        if !self.skip_cas {
            self.cas.merge_cell(cell)?;
        }
        self.stream.merge_cell(cell)?;
        // Generated quantities: fold this cell's trajectory alongside the leaf +
        // mirror, using the cell's own resolved params. Composed here (not a
        // second RunSink) so it sees the same per-cell trajectory.
        if let Some(q) = &mut self.quant {
            q.push_cell(cell)?;
        }
        Ok(())
    }
}

/// Build the simulate-side [`crate::batch::CasSink`] so each cell's leaf is
/// byte-identical to the leaf `batch run` writes for the same logical cell.
///
/// The identity inputs mirror batch exactly:
///   - `base_model` is the **raw** parsed IR (params NOT applied), so the
///     model level digest is constant across the grid and carries no param
///     values — those live in the `params` level.
///   - `base_params` is the resolved base parameter map (`--params` ∪
///     `--param`), the same resolved values the `params` level hashes.
///   - `resolved_scenarios` carry the hash-relevant enable/disable/params
///     delta (a model preset's own fields, or the CLI ad-hoc patch).
#[allow(clippy::too_many_arguments)]
fn build_simulate_cas_sink(
    run: &util::SimRun,
    // The original source path (e.g. `sir.camdl`) for the human-readable store
    // prefix + provenance. `run.ir_path` is the pre-compiled `.ir.json`, whose
    // basename would slug to `camdl_<pid>`; the display fields must keep the
    // model's real name so leaves land under `sims/sir-<hash>/…`.
    display_path: &str,
    scenario_names: &[String],
    cas_root: &str,
    obs_enabled: bool,
    force: bool,
    label: Option<String>,
    fit_dep: Vec<runid::inputs::ArtifactRef>,
    allow_degenerate_rates: bool,
    output_cols: crate::util::OutputColumns,
    // gh#656: `--emit-every`. It names the obs subtree (`obs_subtree_hash`) as
    // well as setting the emitted times, so two cadences are two addressable
    // artifacts under one shared trajectory leaf — the leaf itself does NOT
    // re-key, because its bytes do not depend on the emission cadence.
    emit_every: Option<crate::emit_every::EmitEvery>,
    total_runs: usize,
) -> Result<crate::batch::CasSink, String> {
    // Parse the raw IR (envelope-aware) — params NOT applied (batch parity).
    // `run.ir_path` is already the compiled IR, so this short-circuits; the
    // forward CAS-sink path never reads the state-Jacobian either
    // (`needs_state_grad = false`, gh#439 A2).
    let (ir_path_resolved, _tmp) = util::resolve_ir_path(&run.ir_path, false)?;
    let src = std::fs::read_to_string(&ir_path_resolved)
        .map_err(|e| format!("cannot read {}: {}", ir_path_resolved, e))?;
    let mut base_model: ir::Model = ir::from_str(&src)
        .map_err(|e| format!("IR load error from {}: {}", ir_path_resolved, e))?;

    // gh#616: substitute the run's observation anchors into the model that is
    // HASHED, with the same window `resolve_run_model` uses on the model that is
    // RUN. This is what puts the resolution into run identity: `Model::hash_into`
    // walks `simulation`, `presets` and `time_functions`, so a data vintage that
    // moves `last_obs` moves the resolved `t_end` and the resolved forcing knots,
    // and the two vintages cannot share a `run_id`.
    //
    // NOTE for the deferred `value_at`-under-simulate work (F23): that argument
    // does NOT extend to quantity anchors. `Model::hash_into` deliberately
    // EXCLUDES `quantities` (and `contrasts`) so reporting can never re-key a
    // run — so resolving a `value_at(…, last_obs)` here would change no hashed
    // field, and two vintages would share a `run_id` while reporting different
    // numbers. Whoever lands F23 must key the resolved anchor EXPLICITLY; no
    // model field will do it for them.
    if let Some(w) = run.obs_anchors {
        crate::obs_anchor::substitute(&mut base_model, w)?;
    }
    // Audit 2026-08-23 #1: apply `--integrator` to the model that is HASHED,
    // exactly as `resolve_run_model` applies it to the model that is RUN. The
    // identity path never applied it, so an rk45 run and an rk4 run of the
    // same model shared one `run_id` — and post-S1 the second would die with
    // DivergentRecompute instead of getting its own leaf. Same argument as
    // the gh#616 anchor substitution above: the two loads must agree by
    // construction.
    util::apply_integrator_override(&mut base_model, run.integrator);

    // Resolved base params: model defaults overlaid by --params, then
    // --param-vec, then --param (matching `resolve_run_model`'s tier-5
    // last-wins order: vec entries expand before `run.overrides`, so an
    // explicit `--param NAME=VAL` still wins), filtered to the params that
    // resolved to a value (a param that relies on the scenario half is
    // supplied there, not here — mirrors prepare_cas_ctx).
    let mut params_model = base_model.clone();
    for path in &run.params_files {
        util::apply_params_file(&mut params_model, path)?;
    }
    // Audit 2026-08-23 #2: `--param-vec` values were absent from the hashed
    // params — the identity path never read `set_vec_entries`. Shared
    // expansion with the run path so the two cannot diverge.
    for (k, v) in util::resolve_param_vec_entries(&params_model, &run.set_vec_entries)? {
        if let Some(p) = params_model.parameters.iter_mut().find(|p| p.name == k) {
            p.value = p.value.with_value(v);
        }
    }
    for (k, v) in &run.overrides {
        if let Some(p) = params_model.parameters.iter_mut().find(|p| &p.name == k) {
            p.value = p.value.with_value(*v);
        }
    }
    util::validate_parameter_values(&params_model)?;
    let base_params: HashMap<String, f64> = params_model.parameters.iter()
        .filter_map(|p| p.value.resolved_value().map(|v| (p.name.clone(), v)))
        .collect();

    // Resolve each simulate scenario into the hash-relevant delta. A name
    // matching a model preset reads the preset's own enable/disable/params
    // (preset route); the no-scenario case is the CLI ad-hoc baseline.
    let resolved_scenarios: Vec<crate::batch::ResolvedEntry> = if scenario_names.is_empty() {
        vec![crate::batch::ResolvedEntry {
            name: "baseline".to_string(),
            route: None,
            enable: run.adhoc_enable.clone(),
            disable: run.adhoc_disable.clone(),
            params: HashMap::new(),
            // The implicit baseline is the model as written — its horizon —
            // unless `--to` overrides it (gh#626): the override is what the
            // cell RUNS, so it is what the cell is KEYED on.
            t_end: run.t_end_override,
        }]
    } else {
        scenario_names.iter().map(|name| {
            let preset = base_model.presets.iter().find(|p| &p.name == name)
                .ok_or_else(|| {
                    let available: Vec<&str> = base_model.presets.iter()
                        .map(|p| p.name.as_str()).collect();
                    format!("scenario '{}' not found. Available: {}", name,
                        if available.is_empty() { "(none)".into() } else { available.join(", ") })
                })?;
            Ok(crate::batch::ResolvedEntry {
                name: name.clone(),
                route: Some(preset.name.clone()),
                enable: preset.enable.clone(),
                disable: preset.disable.clone(),
                params: preset.params.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                // Composed, via the single horizon authority — NOT `preset.t_end`,
                // which misses a horizon inherited through `compose = [...]` and
                // would then key the cell on a window it does not run (gh#561).
                // `--to` (gh#626) wins when present — the up-front conflict rule
                // has already refused any scenario horizon it would discard.
                t_end: match run.t_end_override {
                    Some(v) => Some(v),
                    None => crate::params_resolver::composed_preset_t_end(&base_model, name)
                        .map_err(|e| e.to_string())?,
                },
            })
        }).collect::<Result<Vec<_>, String>>()?
    };

    // Display prefix + provenance source path come from the ORIGINAL model
    // path, not the compiled temp IR (`run.ir_path`), so the store layout and
    // `run.json` provenance show the model's real name (gh#51 cross-check
    // parity with the pre-hoist behaviour).
    let model_stem = hashing::path_stem_slug(display_path);
    let runs_dir = format!("{}/sims", cas_root);

    Ok(crate::batch::CasSink {
        resolved_scenarios,
        model_path: display_path.to_string(),
        model_stem,
        base_model,
        base_params,
        // External `--table NAME=PATH` overrides: the model IR carries only the
        // table *reference*; the file content is read at run time, so its bytes
        // must enter the run_id (folded into the params level by `cell_resolve`).
        table_files: run.table_files.clone(),
        backend: run.backend,
        dt: run.dt,
        allow_degenerate_rates,
        output_cols,
        runs_dir,
        obs_enabled,
        emit_every,
        force,
        total: total_runs,
        counter: 0,
        completed_runs: Vec::new(),
        errors: Vec::new(),
        label,
        fit_dep,
        // Progress: a LONE run (total_runs <= 1) goes through the engine's
        // per-timestep bar (`run_one_cell_with_progress`), so the sink adds no
        // overall bar (a redundant `1/1`). A multi-cell ensemble hits
        // `run_one_cell` (no inner bar), so the sink owns the overall cells
        // bar — otherwise the ensemble runs silently. The `Task` honours
        // `--progress none`/`plain` internally.
        progress: crate::batch::cells_progress(total_runs, "simulate"),
    })
}

/// Store the combined-across-cells wide-format TSV of a multi-cell `simulate`
/// as a content-addressed `SimEnsemble` leaf that references (deps) the N
/// per-cell `Sim` leaves. `combined_bytes` are the exact bytes the `-o` mirror
/// writes, so `camdl cat <ensemble>` == the `-o` combined TSV.
///
/// Failure here is reported but non-fatal: the per-cell `Sim` leaves are the
/// primary system of record and are already committed; the ensemble is a
/// derived convenience view, so a write hiccup must not fail the whole run
/// (mirrors the obs-child non-fatal posture).
fn write_sim_ensemble(
    cas: &crate::batch::CasSink,
    combined_bytes: &[u8],
    cas_root: &str,
    label: Option<String>,
) {
    // Build the cell list from the committed leaves. A cell without a resolved
    // run_id failed identity resolution (already on `errors`, which aborted
    // above) — defensively skip it so we never fold a bogus dep.
    let cells: Vec<crate::sim_ensemble_cas::EnsembleCell> = cas
        .completed_runs
        .iter()
        .filter_map(|r| {
            Some(crate::sim_ensemble_cas::EnsembleCell {
                scenario_label: r.scenario.clone(),
                process_seed: r.process_seed,
                draw_idx: r.draw_idx,
                sim_run_id: r.run_id?,
                traj_digest: r.traj_digest?,
            })
        })
        .collect();
    if cells.len() != cas.completed_runs.len() {
        eprintln!("warning: ensemble: {} of {} cells lacked a resolved identity; \
                   skipping ensemble artifact",
            cas.completed_runs.len() - cells.len(), cas.completed_runs.len());
        return;
    }

    let ctx = crate::sim_ensemble_cas::EnsembleCtx {
        model: &cas.base_model,
        ir_version: ir::IR_VERSION.trim(),
        engine_version: version::VERSION_SHORT,
        stem: cas.model_stem.as_deref().unwrap_or("model"),
        backend: cas.backend,
        dt: cas.dt,
        base_params: &cas.base_params,
        cells: &cells,
    };
    let resolved = match crate::sim_ensemble_cas::resolve_sim_ensemble(&ctx) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("warning: ensemble identity: {}", e);
            return;
        }
    };

    let root = std::path::Path::new(cas_root);

    // Distinct scenarios (sorted, deduped) for the display payload.
    let mut scenarios: Vec<String> = cells.iter().map(|c| c.scenario_label.clone()).collect();
    scenarios.sort();
    scenarios.dedup();
    let n_seeds = {
        let mut s: Vec<u64> = cells.iter().map(|c| c.process_seed).collect();
        s.sort_unstable();
        s.dedup();
        s.len()
    };
    let n_draws = {
        let mut d: Vec<usize> = cells.iter().map(|c| c.draw_idx).collect();
        d.sort_unstable();
        d.dedup();
        d.len()
    };
    let inputs = serde_json::json!({
        "n_cells": cells.len(),
        "scenarios": scenarios,
        "n_scenarios": scenarios.len(),
        "n_seeds": n_seeds,
        "n_draws": n_draws,
    });

    let deps = crate::sim_ensemble_cas::ensemble_deps(&cells);
    let resolved_artifact = crate::resolve::ResolvedArtifact {
        kind: runid::ArtifactKind::SimEnsemble,
        levels: resolved.levels.clone(),
        run_id: resolved.run_id,
        display_inputs: inputs,
    };
    let meta = crate::resolve::RecordMeta::new(ir::IR_VERSION.trim(), &cas.model_path, label)
        .with_deps(deps);

    let mut artifacts = runid::Artifacts::new();
    artifacts.insert("ensemble.tsv", combined_bytes.to_vec());
    let store = runid::FsCasStore::new(root);
    match crate::resolve::begin_resolved_write(
        &store,
        root,
        &resolved_artifact,
        &meta,
        crate::resolve::WriteMode::Atomic(artifacts),
    ) {
        Ok(crate::resolve::ResolvedWrite::Committed(dest)) => {
            // Full rooted path (e.g. `./results/ensembles/…`), matching the
            // `stored` banner — not the bare store-relative `ensembles/…`.
            crate::status::step("ensemble", dest.to_string_lossy());
        }
        Ok(crate::resolve::ResolvedWrite::Streaming(_)) => {
            unreachable!("Atomic write mode never returns a streaming claim")
        }
        Err(e) => eprintln!("warning: ensemble commit failed: {}", e),
    }
}

/// Report where the run(s) landed in the store and how to read them back.
/// A lone run prints its rooted leaf path plus the `camdl cat <run_id>` that
/// reads it; a multi-run prints a one-line summary plus `camdl list`. The path
/// is rooted at `cas_root` (the `--output-dir`) — e.g.
/// `./results/sims/model-…/…/seed_…` — so it is copy-paste ready, not just the
/// store-relative `sims/…` tail. (Whether a leaf was freshly written or a
/// cache hit is not distinguished here — `commit_atomic` is idempotent and
/// does not report which; that is a follow-up.)
fn report_cas_leaves(runs: &[crate::batch::RunEntry], cas_root: &str) {
    let root = cas_root.trim_end_matches('/');
    match runs.len() {
        0 => {}
        1 => {
            let r = &runs[0];
            crate::status::done("stored", format!("{}/{}", root, r.run_path));
            if let Some(id) = r.run_id {
                crate::status::hint(format!("camdl cat {}", id.to_hex()));
            }
        }
        n => {
            crate::status::done("stored", format!("{} leaves \u{b7} {}/sims/", n, root));
            crate::status::hint("camdl list");
        }
    }
}

/// One sampled observation row in the combined-obs accumulator.
struct ObsRow { time: f64, replicate: usize, draw: usize, scenario: String, value: f64 }

/// `RunSink` for `camdl simulate`: streams the combined wide-format
/// trajectory TSV (replicate/scenario/draw columns gated on the grid
/// shape) and accumulates synthetic observations for a post-loop combined
/// write. Reproduces the pre-unification `run_simulate` output byte-for-byte.
struct StreamSink {
    /// The combined wide-format trajectory TSV, accumulated in memory across
    /// cells. `None` ⟺ the trajectory is suppressed (`--obs-only`). Post-loop
    /// this buffer is the single source of bytes for BOTH the `-o` mirror and
    /// (multi-run) the content-addressed `SimEnsemble` artifact, so the two are
    /// byte-identical by construction. Never streamed to stdout (Item C — the
    /// CAS leaf/ensemble are the system of record).
    traj_out: Option<Vec<u8>>,
    traj_header_written: bool,
    /// gh#156: the resolved `--no-flows` / `--columns` filter for this mirror's
    /// trajectory columns — the same selection the CAS leaf renderer uses.
    output_cols: crate::util::OutputColumns,
    dates: bool,
    dates_render: Option<(String, String)>,
    obs_path: Option<String>,
    obs_dir: Option<String>,
    obs_data: Vec<Vec<ObsRow>>,
    obs_stream_names: Vec<String>,
    obs_times_cache: Vec<Vec<f64>>,
    /// gh#656: the `--emit-every` override for this run's emitted observations.
    /// `None` for every run without the flag.
    emit_every: Option<crate::emit_every::EmitEvery>,
    // Grid shape, captured in `on_start`, used by the column-gating logic
    // and the post-loop combined-obs writer.
    total_runs: usize,
    n_scenarios: usize,
    n_draws: usize,
}

impl engine::RunSink for StreamSink {
    fn on_start(&mut self, grid: &engine::Grid) {
        self.total_runs = grid.total_runs;
        self.n_scenarios = grid.n_scenarios;
        self.n_draws = grid.n_points;
    }

    fn merge_cell(&mut self, cell: &engine::CellResult) -> Result<(), String> {
        use std::io::Write;
        let traj = &cell.traj;
        let model = &cell.model;
        let run_idx = cell.spec.run_idx;
        let draw_idx = cell.spec.point_idx;
        let total_runs = self.total_runs;
        let n_scenarios = self.n_scenarios;
        let n_draws = self.n_draws;
        let scenario_label = cell.spec.scenario.name().to_string();

        // Diagnostics (first run only).
        if run_idx == 0 && !traj.transition_diagnostics.is_empty() {
            match write_diagnostics_tsv("diagnostics.tsv", &traj.transition_diagnostics) {
                Ok(zero_count) => {
                    if zero_count > 0 { warn_zero_firings(&traj.transition_diagnostics); }
                }
                Err(e) => eprintln!("warning: could not write diagnostics.tsv: {}", e),
            }
        }

        // ── Trajectory output ───────────────────────────────────────────────
        if let Some(ref mut out) = self.traj_out {
            // One column selection drives both this mirror and the CAS leaf
            // renderer (`util::write_traj_to`) — see `TrajColumns`, so an
            // output-view filter can never apply to one writer and not the other.
            let cols = self.output_cols.cols(model);

            let date_origin: Option<&str> = if self.dates {
                match model.origin.as_deref() {
                    Some(o) => {
                        if self.dates_render.is_none() {
                            self.dates_render = Some((o.to_string(), model.time_unit.clone()));
                        }
                        Some(o)
                    }
                    None => return Err(
                        "--dates requires the model to declare an `origin` \
                         (e.g. `origin = date(\"2020-01-01\")`).".to_string()),
                }
            } else {
                None
            };

            if !self.traj_header_written {
                writeln!(out, "# {}", version::VERSION).map_err(|e| e.to_string())?;
                if total_runs > 1 { write!(out, "replicate\t").map_err(|e| e.to_string())?; }
                if n_scenarios > 1 { write!(out, "scenario\t").map_err(|e| e.to_string())?; }
                if n_draws > 1 { write!(out, "draw\t").map_err(|e| e.to_string())?; }
                write!(out, "t").map_err(|e| e.to_string())?;
                if date_origin.is_some() { write!(out, "\tdate").map_err(|e| e.to_string())?; }
                cols.write_header(out).map_err(|e| e.to_string())?;
                writeln!(out).map_err(|e| e.to_string())?;
                self.traj_header_written = true;
            }

            for snap in &traj.snapshots {
                if total_runs > 1 { write!(out, "{}\t", run_idx + 1).map_err(|e| e.to_string())?; }
                if n_scenarios > 1 { write!(out, "{}\t", scenario_label).map_err(|e| e.to_string())?; }
                if n_draws > 1 { write!(out, "{}\t", draw_idx + 1).map_err(|e| e.to_string())?; }
                write!(out, "{}", snap.t).map_err(|e| e.to_string())?;
                if let Some(o) = date_origin {
                    let d = ir::caltime::internal_to_date_hires(o, snap.t, &model.time_unit)
                        .map_err(|e| format!("error rendering date: {}", e))?;
                    write!(out, "\t{}", d).map_err(|e| e.to_string())?;
                }
                cols.write_row(out, snap).map_err(|e| e.to_string())?;
                writeln!(out).map_err(|e| e.to_string())?;
            }
        }

        // ── Observation sampling ────────────────────────────────────────────
        if self.obs_path.is_some() || self.obs_dir.is_some() {
            if self.dates && self.dates_render.is_none() {
                match model.origin.as_deref() {
                    Some(o) => self.dates_render = Some((o.to_string(), model.time_unit.clone())),
                    None => return Err(
                        "--dates requires the model to declare an `origin` \
                         (e.g. `origin = date(\"2020-01-01\")`).".to_string()),
                }
            }
            let compiled = std::sync::Arc::new(
                sim::CompiledModel::new(model.clone())
                    .map_err(|e| format!("error compiling model for obs: {:?}", e))?,
            );
            let params = compiled.default_params.clone();
            let mut obs_rng = sim::rng::StatefulRng::new(cell.spec.obs_seed);

            if run_idx == 0 {
                for obs_model in &model.observations {
                    self.obs_stream_names.push(obs_model.name.clone());
                    self.obs_data.push(Vec::new());
                    // `model` is the cell's resolved model, so this is the
                    // cell's own horizon under a per-scenario `to` (gh#561),
                    // and its forecast origin under `--init-state` (gh#641 —
                    // `None` for every ordinary run). The cells share one obs
                    // axis, and the pre-flight has already refused a grid whose
                    // cells disagree about it.
                    let times = obs_emit_schedule_times(
                        obs_model,
                        cell.spec.sim_run.init_state.as_ref().map(|i| i.origin_t),
                        model.simulation.t_end,
                        self.emit_every.as_ref(),
                    )?;
                    self.obs_times_cache.push(times);
                }
            }

            for (si, obs_ir) in model.observations.iter().enumerate() {
                let sampler = sim::inference::obs_model::compile_obs_sample_pf(
                    obs_ir, compiled.clone(), &params,
                );
                let obs_times = self.obs_times_cache[si].clone();
                // `None` — `simulate --obs` emits on the model's own
                // emit_schedule and binds no data (gh#702).
                let projected_values = project_all_obs_times(
                    traj, obs_ir, model, &obs_times, None,
                )?;
                for (ti, &obs_t) in obs_times.iter().enumerate() {
                    // GH #6 fix: pass the actual compartment state at the obs
                    // time so the likelihood p/mean expressions can resolve
                    // references like `N = S + I + R`.
                    let snap = snap_at(traj, obs_t);
                    let draw = sampler(
                        projected_values[ti], obs_t, &snap.int_state.counts, &[], &mut obs_rng,
                    );
                    self.obs_data[si].push(ObsRow {
                        time: obs_t,
                        replicate: run_idx + 1,
                        draw: draw_idx + 1,
                        scenario: scenario_label.clone(),
                        value: draw,
                    });
                }
            }
        }

        Ok(())
    }
}

impl StreamSink {
    /// Write the combined synthetic-observation output after all cells are
    /// merged. Reproduces the pre-unification post-loop obs writers.
    fn write_obs_output(&self) {
        use std::io::Write;
        if (self.obs_path.is_none() && self.obs_dir.is_none()) || self.obs_data.is_empty() {
            return;
        }
        let total_runs = self.total_runs;
        let n_scenarios = self.n_scenarios;
        let n_draws = self.n_draws;
        let multi_rep = total_runs > 1;
        let date_render = self.dates_render.as_ref();

        // --obs / --obs-only: single wide-format file.
        if let Some(ref path) = self.obs_path {
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
            let mut out = std::io::BufWriter::new(f);

            if multi_rep { write!(out, "replicate\t").unwrap(); }
            if n_scenarios > 1 { write!(out, "scenario\t").unwrap(); }
            if n_draws > 1 { write!(out, "draw\t").unwrap(); }
            write!(out, "time").unwrap();
            if date_render.is_some() { write!(out, "\tdate").unwrap(); }
            for name in &self.obs_stream_names { write!(out, "\t{}", name).unwrap(); }
            writeln!(out).unwrap();

            let n_times = self.obs_times_cache[0].len();
            for run in 0..total_runs {
                for ti in 0..n_times {
                    let row_idx = run * n_times + ti;
                    if multi_rep { write!(out, "{}\t", run + 1).unwrap(); }
                    if n_scenarios > 1 { write!(out, "{}\t", self.obs_data[0][row_idx].scenario).unwrap(); }
                    if n_draws > 1 { write!(out, "{}\t", self.obs_data[0][row_idx].draw).unwrap(); }
                    let t_val = self.obs_data[0][row_idx].time;
                    write!(out, "{}", t_val).unwrap();
                    if let Some((o, tu)) = date_render {
                        let d = ir::caltime::internal_to_date_hires(o, t_val, tu)
                            .unwrap_or_else(|e| { eprintln!("error rendering date: {}", e); std::process::exit(1); });
                        write!(out, "\t{}", d).unwrap();
                    }
                    for si in 0..self.obs_stream_names.len() {
                        let val = self.obs_data[si][row_idx].value;
                        if val == val.round() && val.abs() < 1e15 {
                            write!(out, "\t{}", val as i64).unwrap();
                        } else {
                            write!(out, "\t{:.6}", val).unwrap();
                        }
                    }
                    writeln!(out).unwrap();
                }
            }
            drop(out);
            eprintln!("observations written to {}", path);
        }

        // --obs-dir / --obs-only-dir: one file per stream.
        if let Some(ref dir) = self.obs_dir {
            for (si, name) in self.obs_stream_names.iter().enumerate() {
                let path = format!("{}/{}.tsv", dir, name);
                let f = std::fs::File::create(&path)
                    .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
                let mut out = std::io::BufWriter::new(f);

                if multi_rep { write!(out, "replicate\t").unwrap(); }
                if n_scenarios > 1 { write!(out, "scenario\t").unwrap(); }
                if n_draws > 1 { write!(out, "draw\t").unwrap(); }
                if date_render.is_some() {
                    writeln!(out, "time\tdate\t{}", name).unwrap();
                } else {
                    writeln!(out, "time\t{}", name).unwrap();
                }

                for row in &self.obs_data[si] {
                    if multi_rep { write!(out, "{}\t", row.replicate).unwrap(); }
                    if n_scenarios > 1 { write!(out, "{}\t", row.scenario).unwrap(); }
                    if n_draws > 1 { write!(out, "{}\t", row.draw).unwrap(); }
                    write!(out, "{}", row.time).unwrap();
                    if let Some((o, tu)) = date_render {
                        let d = ir::caltime::internal_to_date_hires(o, row.time, tu)
                            .unwrap_or_else(|e| { eprintln!("error rendering date: {}", e); std::process::exit(1); });
                        write!(out, "\t{}", d).unwrap();
                    }
                    let val = row.value;
                    if val == val.round() && val.abs() < 1e15 {
                        writeln!(out, "\t{}", val as i64).unwrap();
                    } else {
                        writeln!(out, "\t{:.6}", val).unwrap();
                    }
                }
                drop(out);
                eprintln!("observations written to {}", path);
            }
        }
    }
}

// ── Observation helpers ─────────────────────────────────────��───────────────

/// gh#626: resolve the global observation anchors (`first_obs`, `last_obs`)
/// for an anchored `--to`, from the fit's `[data.observations]` bindings.
/// `--fit` accepts a fit toml, run directory, `@label`, or hash — the same
/// resolution `fit predict` uses (`fit::handle::resolve_fit`; for a run dir
/// the archived `fit.toml.original` is the fallback, whose relative data
/// paths resolve against the segment — co-located data only, the same limit
/// `fit predict <run-dir>` has). Streams load through the same shared seam as
/// pfilter/profile, so dated time columns, long-form families, and holes
/// resolve exactly as at fit time; `apply_conditioning_windows` is NOT
/// applied (anchors fold over the raw streams; a conditioning hole must not
/// shift `first_obs`). Hole rows count as observation times, matching
/// `fit predict`'s `value_at` anchor.
fn o_source(o: &ir::observation::ObservationModel) -> String { o.source.clone() }

fn resolve_simulate_obs_anchors(
    model: &ir::Model,
    fit_ref: &std::path::Path,
    dt: f64,
) -> Result<(f64, f64), String> {
    // A bare fit.toml needs NO completed run — only its [data.observations]
    // is consulted. Run dirs / @labels / hashes resolve through the store.
    let config = match crate::fit::handle::FitRef::classify(&fit_ref.to_string_lossy()) {
        crate::fit::handle::FitRef::Config(path) => {
            crate::fit::config_v2::FitConfigV2::load(&path.to_string_lossy())
                .map_err(|e| format!("failed to load fit toml '{}': {e}", path.display()))?
        }
        _ => crate::fit::handle::resolve_fit(&fit_ref.to_string_lossy())
            .map_err(|e| e.to_string())?
            .config,
    };
    obs_anchors_from_config(model, &config, dt)
}

/// The fold over a config already in hand. Split out so `fit predict` — which
/// has resolved its config long before it could look up a `--fit` ref —
/// resolves anchors through the SAME reader as `simulate`, rather than growing a
/// second one that could disagree about which rows count as observation times.
pub(crate) fn obs_anchors_from_config(
    model: &ir::Model,
    config: &crate::fit::config_v2::FitConfigV2,
    dt: f64,
) -> Result<(f64, f64), String> {
    let data = config.data_spec().map_err(|e| format!(
        "the fit config has no [data] block to read observed times from: {e}"))?;
    let model_obs_names: Vec<String> =
        model.observations.iter().map(|o| o.name.clone()).collect();
    let effective_pairs = data.effective_observations(&model_obs_names)
        .map_err(|e| format!("resolving [data.observations]: {e}"))?;
    if effective_pairs.is_empty() {
        return Err("the fit config's [data] resolves to zero observation \
                    streams — nothing to anchor last_obs/first_obs to.".into());
    }
    let bound: Vec<(String, std::path::PathBuf)> = effective_pairs.iter()
        .map(|(k, v)| (k.clone(), std::path::PathBuf::from(v)))
        .collect();
    obs_anchors_from_bindings(model, &bound, dt)
}

/// The fold itself, over `(binding key → file)` pairs a command has ALREADY
/// resolved. `pfilter` / `profile` / `survey` reach it directly: their bindings
/// come from `--data` flags as often as from a fit toml, and re-deriving the
/// window from a config they may not have would be a second reader that could
/// disagree with the one they score against.
pub(crate) fn obs_anchors_from_bindings(
    model: &ir::Model,
    bound: &[(String, std::path::PathBuf)],
    dt: f64,
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
    // Only the observation TIMES are needed, so the streams load through
    // `load_observations` directly (the same reader the shared seam wraps) —
    // no CompiledModel, so a model whose estimated params carry no values
    // still anchors. Conditioning is NOT applied (anchors fold over the raw
    // streams), and hole rows count as observation times, both matching
    // `fit predict`'s value_at anchor.
    let effective = crate::fit::runner::data_bindings_to_effective(model, bound)?;
    let mut first = f64::INFINITY;
    let mut last = f64::NEG_INFINITY;
    for obs_model in model.observations.iter().filter(|o| effective.contains_key(&o.source)) {
        let data_path = &effective[&o_source(obs_model)];
        let siblings: Vec<&ir::observation::ObservationModel> = model.observations.iter()
            .filter(|o| o.source == obs_model.source)
            .collect();
        let (obs, _cells, _aux) = crate::fit::runner::load_observations(
            data_path, obs_model, &siblings, dt, &time_opts,
        )?;
        for o in &obs {
            first = first.min(o.time);
            last = last.max(o.time);
        }
    }
    if !first.is_finite() || !last.is_finite() {
        return Err("the bound observation streams contain no rows — nothing \
                    to anchor last_obs/first_obs to.".into());
    }
    Ok((first, last))
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
    match schedule {
        ir::observation::ObservationSchedule::Regular(reg) => {
            let mut times = Vec::new();
            let mut t = reg.start;
            while t <= t_end + 1e-9 {
                if after_origin(t) {
                    times.push(t);
                }
                t += reg.step;
            }
            times
        }
        ir::observation::ObservationSchedule::AtTimes(times) => {
            times.iter().copied().filter(|&t| t <= t_end && after_origin(t)).collect()
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
    restart_origin: Option<f64>,
    t_end: f64,
    emit: Option<&crate::emit_every::EmitEvery>,
) -> Result<Vec<f64>, String> {
    match &obs.emit_schedule {
        Some(s) => {
            let s = crate::emit_every::apply_override(emit, obs, s)?;
            Ok(obs_schedule_times(&s, restart_origin, t_end))
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
    // `emit_schedule` (simulate/batch/synthetic) and a loaded data file's time
    // column (`fit predict`) — and advice that fits one is wrong for the other.
    // Telling a `fit predict` user to "change the emit schedule" is nonsense:
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
         come from a data file — `fit predict` projects at the observed times, \
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

/// `window_start` is the fit's conditioning boundary for this stream
/// (`condition_from`), when it has one. An INTERVAL (incidence) projection
/// reports the flow accumulated since the previous emitted time; the first
/// emitted time has no predecessor, so its bin opens at `window_start` —
/// exactly where the likelihood resets that stream's accumulator. `None` opens
/// it at the model origin, which is the behaviour for every caller with no data
/// to condition on (synthetic emission, `simulate --obs`, `batch`).
///
/// Passing the wrong one is a first-row-only error, which is why gh#702 lived
/// so long: one row among many on a long series, the ENTIRE artifact on a
/// single-observation fit. It has no effect on an INSTANT (prevalence /
/// expression) projection, which reads state at a time and has no accumulator
/// to reset.
pub(crate) fn project_all_obs_times(
    traj: &sim::Trajectory,
    obs_ir: &ir::observation::ObservationModel,
    model: &ir::Model,
    obs_times: &[f64],
    window_start: Option<f64>,
) -> Result<Vec<f64>, String> {
    check_obs_times_on_snapshot_grid(traj, &obs_ir.name, &model.time_unit, obs_times)?;
    // Per-interval incidence over a set of transition flow indices: build the
    // running cumulative flow at each snapshot, read it at each obs time, then
    // difference consecutive obs times. Shared by CumulativeFlow (one exact
    // transition) and CumulativeFlowSum (explicit strata family per §25.4).
    let incidence_over = |flow_indices: &[usize]| -> Result<Vec<f64>, String> {
        // `f64` throughout: ODE flows are real-valued (the chain-binomial /
        // Gillespie integer flows widen losslessly via `Flows::value`).
        // `cum_at_snap[i]` is the flow over (t_start, snapshots[i].t]: the
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

        let mut cum_at_obs = Vec::with_capacity(obs_times.len());
        let mut snap_idx = 0;
        for &obs_t in obs_times {
            while snap_idx + 1 < cum_at_snap.len()
                && cum_at_snap[snap_idx + 1].0 <= obs_t + OBS_SNAP_EPS
            {
                snap_idx += 1;
            }
            cum_at_obs.push(if snap_idx < cum_at_snap.len() && cum_at_snap[snap_idx].0 <= obs_t + OBS_SNAP_EPS {
                cum_at_snap[snap_idx].1
            } else {
                0.0
            });
        }

        // Where the FIRST bin opens. Reading the cumulative flow at the
        // conditioning boundary requires the boundary to BE a recorded
        // snapshot: between snapshots it would resolve to an earlier one and
        // silently hand part of the warm-up back to the first bin — the same
        // class of silent-wrong gh#702 is about, reintroduced by the fix for
        // it. So refuse, naming the boundary and the output schedule.
        let seed = match window_start {
            None => 0.0,
            Some(t0) => {
                let i = resolved_snapshot_index(traj, t0)
                    .filter(|_| is_recorded_snapshot(traj, t0))
                    .ok_or_else(|| format!(
                        "observation stream '{}': the conditioning boundary \
                         condition_from = {t0} is not a recorded output time, \
                         so the flow accumulated up to it cannot be read.\n  \
                         The first incidence bin is ({t0}, first_obs] — the \
                         window this fit scored — and the projection reads it \
                         as the difference of the recorded cumulative flow at \
                         those two times.\n  \
                         Fix: add {t0} to the output schedule \
                         (`output {{ trajectories {{ every = ... }} }}`, or an \
                         `at = [...]` list containing it), or move \
                         condition_from onto a recorded output time.",
                        obs_ir.name,
                    ))?;
                cum_at_snap[i].1
            }
        };

        // Difference: flow in interval (prev_obs_t, obs_t], with the first bin
        // opening at `seed`'s time rather than at t_start.
        let mut result = Vec::with_capacity(obs_times.len());
        let mut prev_cum = seed;
        for &cum in &cum_at_obs {
            result.push(cum - prev_cum);
            prev_cum = cum;
        }
        Ok(result)
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
                    eprintln!("error: DerivedExpr projection — model compile: {:?}", e);
                    std::process::exit(1);
                });
            let stream_proj = StreamProjection::from_ir(
                &obs_ir.projection, &compiled, &obs_ir.name,
            ).unwrap_or_else(|e| {
                eprintln!("error: DerivedExpr projection — resolve: {}", e);
                std::process::exit(1);
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
    }
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
    eprintln!("error: observation '{}' projects compartment '{}' which doesn't exist",
        obs_name, comp_name);
    std::process::exit(1);
}

pub(crate) fn snap_at(traj: &sim::Trajectory, obs_t: f64) -> &sim::Snapshot {
    traj.snapshots.iter().rev()
        .find(|s| s.t <= obs_t + OBS_SNAP_EPS)
        .unwrap_or_else(|| {
            eprintln!("error: no snapshot at or before t={}", obs_t);
            std::process::exit(1);
        })
}

fn read_comp(snap: &sim::Snapshot, loc: &CompLoc) -> f64 {
    match loc {
        CompLoc::Int(i) => snap.int_state.counts[*i] as f64,
        CompLoc::Real(i) => snap.real_state.values[*i],
    }
}

/// Generate N uniform random draws from model parameter bounds.
fn generate_uniform_draws(
    ir_path: &str,
    n: usize,
    seed: u64,
) -> Result<Vec<HashMap<String, f64>>, String> {
    let (model, _) = util::load_model(ir_path)?;
    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_UNIFORM);

    let mut draws = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = HashMap::new();
        for p in &model.parameters {
            let val = if let Some((lo, hi)) = p.bounds() {
                lo + (hi - lo) * rng.uniform()
            } else if let Some(v) = p.value.resolved_value() {
                // No bounds — use the default value (constant)
                v
            } else {
                return Err(format!(
                    "parameter '{}' has no bounds and no default value.\n  \
                     --draws uniform requires bounds on all parameters.",
                    p.name
                ));
            };
            row.insert(p.name.clone(), val);
        }
        draws.push(row);
    }
    eprintln!("generated {} uniform draws from parameter bounds ({} params)",
        n, model.parameters.len());
    Ok(draws)
}

/// Generate N draws from declared priors in a fit.toml.
/// Each draw is a complete parameter vector (estimated from priors + fixed).
///
/// gh#86: resolves priors through the three-tier precedence chain
/// (fit_toml > model_ir > flat) shared with `camdl fit run` (gh#75) and
/// `camdl profile --algorithm pmmh` (gh#73). This is the load-bearing
/// change: previously the fit-toml side was the only consulted source,
/// so a model declaring `~ <dist>` on every parameter would still be
/// rejected if the fit toml's [estimate] blocks omitted `prior = { ... }`.
/// Now the model-IR `~` priors satisfy the requirement.
///
/// `model` is the simulate-flow's already-loaded IR. Threading it through
/// is what enables the IR-tier fallback — without it, this function would
/// be dependent on `FitConfigV2`'s `[model] camdl = …` path and have to
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
/// gh#158: `simulate --fit <toml>` (via `--draws prior`) loads a full
/// `FitConfigV2`, so a minimal or wrong file fails with a raw serde
/// message (e.g. `expected struct ModelRef`) that does not tell the
/// user what shape the file must have. Append a hint naming the
/// expected `[model]` table and pointing at the docs. The original
/// error is preserved so the underlying cause is still visible.
fn wrap_fit_load_error(fit_path: &str, err: String) -> String {
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

fn generate_prior_draws(
    fit_path: &str,
    n: usize,
    seed: u64,
    model: &ir::Model,
) -> Result<Vec<HashMap<String, f64>>, String> {
    use fit::config_v2::{FitConfigV2, EstimatePriorSpec};
    use crate::fit::priors_precedence::{
        resolve_priors_with_precedence, PriorSource,
    };

    let config = FitConfigV2::load(fit_path).map_err(|e| wrap_fit_load_error(fit_path, e))?;
    let fixed = config.fixed.resolve()?;

    // Three-tier resolution: fit_toml > model_ir > flat. Walks every
    // estimated param in declaration order.
    let names: Vec<String> = config.estimate.keys().cloned().collect();
    let resolved = resolve_priors_with_precedence(&names, &config.estimate, model);

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
                            .find(|p| &p.name == name).and_then(|p| p.bounds()))
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
        for (name, val) in &fixed {
            row.insert(name.clone(), *val);
        }
        draws.push(row);
    }

    // Provenance: report how many came from each tier so users can
    // verify the right source was consulted.
    let n_fit_toml = resolved.iter()
        .filter(|r| r.source == PriorSource::FitToml).count();
    let n_model_ir = resolved.iter()
        .filter(|r| r.source == PriorSource::ModelIr).count();
    eprintln!(
        "generated {} prior draws from {} ({} estimated [{} fit-toml + {} model-IR] + {} fixed params)",
        n, fit_path, config.estimate.len(), n_fit_toml, n_model_ir, fixed.len()
    );
    Ok(draws)
}

/// Generate N draws from priors embedded in the model IR.
///
/// Each parameter must be "covered" by one of:
///   - a prior (sampled from)
///   - a concrete value in the IR (held constant)
///   - a scenario preset that sets its value (held constant)
///
/// Selected scenarios are applied to the model before the coverage check, so
/// a workflow like "prior on beta/gamma, N0 pinned by --scenario baseline"
/// works. Parameters with none of the above produce an error with actionable
/// fix options.
fn generate_prior_draws_from_ir(
    ir_path: &str,
    n: usize,
    seed: u64,
    scenarios: &[&str],
) -> Result<Vec<HashMap<String, f64>>, String> {
    // NOTE: this helper takes a LIST of scenarios applied in order,
    // distinct from the unified resolver's single-scenario semantics.
    // The legacy contract (`--draws prior --scenarios a,b,c` layers a→b→c)
    // is preserved here rather than routed through `params_resolver`,
    // which today supports only one named scenario (with the model's
    // declared `compose` list). Migrating this helper to the resolver
    // would either require a multi-scenario API on the resolver or a
    // refactor of the calling CLI to require a single scenario; both
    // are out of scope for the 2026-05-25 CLI UX rev 2 migration.
    // Documented exception, see
    // `docs/dev/notes/2026-05-25-cli-ux-impl-questions.md`.
    let (mut model, _) = util::load_model(ir_path)?;

    // Apply each selected scenario's params to the model. Later scenarios
    // override earlier ones for the same parameter.
    for name in scenarios {
        let preset = model.presets.iter().find(|p| p.name == *name).cloned()
            .ok_or_else(|| {
                let available: Vec<&str> = model.presets.iter().map(|p| p.name.as_str()).collect();
                format!("scenario '{}' not found in model. Available: {}",
                    name,
                    if available.is_empty() { "(none)".into() } else { available.join(", ") })
            })?;
        for (k, v) in &preset.params {
            if let Some(p) = model.parameters.iter_mut().find(|p| p.name == *k) {
                p.value = p.value.with_value(*v);
            }
        }
    }

    // Bounds + finite-value check after scenario application but before
    // prior sampling. Each per-draw prior sample is independently
    // bounds-checked by `sample_with_bounds`; this pass catches the
    // *fixed* (scenario- or model-default-pinned) values that the prior
    // sampler will leave alone (gh#31).
    util::validate_parameter_values(&model)?;

    // Check all params have either a prior or a (scenario-resolved) value.
    let missing: Vec<&str> = model.parameters.iter()
        .filter(|p| p.prior_dist().is_none() && p.value.resolved_value().is_none())
        .map(|p| p.name.as_str())
        .collect();
    if !missing.is_empty() {
        let scen_hint = if scenarios.is_empty() {
            " supply `--scenario NAME` if a scenario pins these values,".to_string()
        } else {
            String::new()
        };
        return Err(format!(
            "parameter{} {} no prior and no default value.\n  \
             Fix options: add `~ prior(...)` to the model,{}\n  \
             supply `--fit FIT.toml`, or use `--draws uniform` for space-filling exploration.",
            if missing.len() > 1 { "s" } else { "" },
            missing.iter().map(|n| format!("'{}'", n)).collect::<Vec<_>>().join(", "),
            scen_hint,
        ));
    }

    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_PRIOR);
    let mut draws = Vec::with_capacity(n);
    let mut n_sampled = 0;
    let mut n_fixed = 0;
    // Per-parameter rejection counts for bounds-truncation diagnostics.
    let mut reject_counts: HashMap<&str, u64> = HashMap::new();

    for i in 0..n {
        let mut row = HashMap::new();
        for p in &model.parameters {
            let value = match p.prior_dist() {
                Some(pd) => {
                    if i == 0 { n_sampled += 1; }
                    let (v, rejected) = sample_with_bounds(pd, p.bounds(), &mut rng, &p.name)?;
                    if rejected > 0 {
                        *reject_counts.entry(p.name.as_str()).or_insert(0) += rejected;
                    }
                    v
                }
                None => {
                    if i == 0 { n_fixed += 1; }
                    p.value.resolved_value().expect("missing check above guarantees value exists")
                }
            };
            row.insert(p.name.clone(), value);
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
            eprintln!(
                "warning: prior for '{}' placed {:.1}% mass outside declared bounds \
                 ({} rejected / {} accepted). Consider widening bounds or tightening \
                 the prior.",
                name, pct, rej, accept
            );
        }
    }

    eprintln!("generated {} prior draws from model IR ({} sampled + {} fixed params)",
        n, n_sampled, n_fixed);
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
    let mut rejected = 0u64;
    for _ in 0..MAX_ATTEMPTS {
        let v = sample_from_prior_raw(pd, rng);
        if v >= lo && v <= hi {
            return Ok((v, rejected));
        }
        rejected += 1;
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
/// pre-key, param-only file) plus the model parameters.
#[derive(Debug, Clone)]
pub(crate) struct KeyedDraw {
    pub chain: Option<usize>,
    pub draw: Option<usize>,
    pub params: HashMap<String, f64>,
}

/// Parse a `draws.tsv` KEEPING the `(chain, draw)` key. The single parser;
/// [`load_draws_tsv`] wraps this and drops the key so every param-only reader
/// (predict's schema validator, the engine, compare) is unchanged.
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
        draws.push(KeyedDraw { chain, draw, params });
    }

    if draws.is_empty() {
        return Err(format!("draws file has header but no data rows: {}", path));
    }
    Ok(draws)
}

/// Param-only draw rows — the `(chain, draw)` key (if any) is dropped. Every
/// reader that treats a row as a parameter vector uses this.
pub(crate) fn load_draws_tsv(path: &str) -> Result<Vec<HashMap<String, f64>>, String> {
    Ok(load_draws_tsv_keyed(path)?.into_iter().map(|d| d.params).collect())
}

/// Print a dry run summary: resolved parameters with provenance.
#[allow(clippy::too_many_arguments)]
fn print_dry_run(
    ir_path: &str,
    ir_path_compiled: &str,
    backend: args::types::ForwardBackend,
    dt: f64,
    seed: u64,
    params_files: &[String],
    cli_overrides: &HashMap<String, f64>,
    scenario_list: &[Option<String>],
    seeds: &[u64],
    draws_path: &Option<String>,
    n_draws: usize,
    replicates: usize,
    total_runs: usize,
    obs_path: &Option<String>,
    obs_dir: &Option<String>,
    obs_only: &Option<String>,
) {
    let d = style::dim;
    let b = style::bold;

    eprintln!("{}", b("camdl simulate (dry run)"));
    eprintln!();

    // Header info
    eprintln!("  {} {}", d("model:"), ir_path);
    eprintln!("  {} {}", d("backend:"), backend);
    eprintln!("  {} {}", d("dt:"), dt);

    if seeds.len() > 1 {
        eprintln!("  {} {}:{} ({} seeds)", d("seeds:"), seeds[0], seeds[seeds.len()-1], seeds.len());
    } else {
        eprintln!("  {} {}", d("seed:"), seed);
    }

    if let Some(ref dp) = draws_path {
        eprintln!("  {} {}", d("draws:"), dp);
    }
    if replicates > 1 && draws_path.is_none() {
        eprintln!("  {} {}", d("replicates:"), replicates);
    }

    let scenarios: Vec<&str> = scenario_list.iter()
        .map(|s| s.as_deref().unwrap_or("(baseline)"))
        .collect();
    if scenarios.len() > 1 || scenarios[0] != "(baseline)" {
        eprintln!("  {} {}", d("scenarios:"), scenarios.join(", "));
    } else {
        eprintln!("  {} (baseline)", d("scenario:"));
    }

    // Obs output
    if let Some(ref p) = obs_path { eprintln!("  {} {}", d("obs:"), p); }
    if let Some(ref p) = obs_dir { eprintln!("  {} {}", d("obs-dir:"), p); }
    if let Some(ref p) = obs_only { eprintln!("  {} {}", d("obs-only:"), p); }

    eprintln!();

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
                        eprintln!("  {} {} rows × {} params",
                            d("draws file:"), n_rows, cols.len());
                    }
                }
            }
        }
    } else {
        // Point/single mode: show resolved parameter values with provenance.
        // Load the already-compiled IR (display still shows the source path).
        match util::load_model(ir_path_compiled) {
            Ok((model, _)) => {
                // Track provenance: (param_name → (value, source, override_chain))
                struct ParamProv {
                    value: f64,
                    source: String,
                    overrides: Vec<(f64, String)>, // (old_value, old_source)
                }
                let mut provs: std::collections::BTreeMap<String, ParamProv> = std::collections::BTreeMap::new();

                // Model defaults
                for p in &model.parameters {
                    if let Some(v) = p.value.resolved_value() {
                        provs.insert(p.name.clone(), ParamProv {
                            value: v, source: "model default".to_string(), overrides: vec![],
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

                // Print
                let max_name_len = provs.keys().map(|k| k.len()).max().unwrap_or(0);
                eprintln!("Parameters ({}):", provs.len());
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
                    eprintln!("  {:width$} = {:>14}  {}",
                        name, val_str, source_str, width = max_name_len);
                }
            }
            Err(e) => {
                eprintln!("  {} {}", d("(could not load model for parameter resolution:"), e);
            }
        }
    }

    // Total runs
    if total_runs > 1 {
        eprintln!();
        let parts: Vec<String> = [
            if n_draws > 1 { Some(format!("{} draws", n_draws)) } else { None },
            if scenarios.len() > 1 { Some(format!("{} scenarios", scenarios.len())) } else { None },
            if seeds.len() > 1 { Some(format!("{} seeds", seeds.len())) } else { None },
            if replicates > 1 && seeds.len() == 1 { Some(format!("{} replicates", replicates)) } else { None },
        ].iter().flatten().cloned().collect();
        eprintln!("  {} {} = {} runs", d("total:"), parts.join(" × "), total_runs);
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
    /// — `snap_at` calls `process::exit(1)` after partial artifacts are written.
    #[test]
    fn empty_trajectory_is_an_error_not_a_pass() {
        let tr = traj(&[]);
        let err = check_obs_times_on_snapshot_grid(&tr, "s", "days", &[0.0])
            .expect_err("no snapshots must be an error");
        assert!(err.contains("recorded no trajectory snapshots"), "got: {err}");
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
        let draws = generate_prior_draws_from_ir(&path, 7, 42, &[]).unwrap();
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
        let draws2 = generate_prior_draws_from_ir(&path, 7, 42, &[]).unwrap();
        for (a, b) in draws.iter().zip(draws2.iter()) {
            for (k, va) in a {
                assert_eq!(va, &b[k], "seed={} {} should be reproducible", 42, k);
            }
        }
    }

    #[test]
    fn prior_draws_from_ir_errors_when_no_prior() {
        // sir_basic has no priors and no preset-applied values on params.
        // Expect a clear error naming the missing parameters.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path = format!("{}/../../../ocaml/golden/sir_basic.ir.json", manifest);
        let err = generate_prior_draws_from_ir(&path, 3, 1, &[]).unwrap_err();
        assert!(err.contains("no prior and no default"), "got: {}", err);
        assert!(err.contains("beta"), "error should name 'beta': {}", err);
        assert!(err.contains("~ prior(...)"), "error should hint at prior syntax: {}", err);
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
                         "format": "tsv", "trajectory": true, "observations": false }},
            "simulation": {{ "t_start": 0.0, "t_end": 1.0, "time_semantics": "continuous",
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
        let draws = generate_prior_draws_from_ir(&path, 100, 42, &[]).unwrap();
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
        let err = generate_prior_draws_from_ir(&path, 1, 42, &[]).unwrap_err();
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
        let draws = generate_prior_draws_from_ir(&path, 50, 42, &[]).unwrap();
        for row in &draws {
            let v = row["beta"];
            assert!((0.0..=1.0).contains(&v),
                "truncation must keep all draws in bounds, got {}", v);
        }
    }

    #[test]
    fn prior_draws_scenario_pins_missing_param() {
        // beta has a prior; N0 has no prior and no default — but a scenario
        // called 'baseline' sets N0. With --scenario baseline, the draws
        // should succeed (sampled beta + fixed N0).
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
                        "format": "tsv", "trajectory": true, "observations": false },
            "simulation": { "t_start": 0.0, "t_end": 1.0,
                            "time_semantics": "continuous", "dt": null, "rng_seed": null },
            "scenarios": [
              { "name": "baseline", "label": "default",
                "params": { "N0": 1000.0 },
                "scale": {}, "enable": [], "disable": [], "compose": [] }
            ],
            "model_structure": null, "balance": null
          }
        }"#;
        let (_dir, path) = write_ir_fixture(json);

        // Without scenario: errors naming N0
        let err = generate_prior_draws_from_ir(&path, 3, 42, &[]).unwrap_err();
        assert!(err.contains("N0"), "should name 'N0': {}", err);
        assert!(err.contains("--scenario"), "hint should mention --scenario: {}", err);

        // With scenario: succeeds, N0 is pinned to 1000
        let draws = generate_prior_draws_from_ir(&path, 5, 42, &["baseline"]).unwrap();
        assert_eq!(draws.len(), 5);
        for row in &draws {
            assert_eq!(row["N0"], 1000.0, "scenario should pin N0");
            let b = row["beta"];
            assert!((0.01..=2.0).contains(&b), "beta out of bounds: {}", b);
        }
    }

    #[test]
    fn prior_draws_unknown_scenario_errors() {
        let ir = ir_with_prior("beta", "[0.01, 2.0]",
            r#"{ "log_normal": { "mu": -1.0, "sigma": 0.5 } }"#, "");
        let (_dir, path) = write_ir_fixture(&ir);
        let err = generate_prior_draws_from_ir(&path, 3, 42, &["nonesuch"]).unwrap_err();
        assert!(err.contains("scenario 'nonesuch' not found"),
            "error should name the bad scenario: {}", err);
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
        let a = generate_prior_draws_from_ir(&path, 5, 42, &[]).unwrap();
        let b = generate_prior_draws_from_ir(&path, 5, 137, &[]).unwrap();
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
    /// (model, estimate, fixed). `FitConfigV2::load` parses the toml
    /// without running validate(), but the parse step still enforces
    /// the Stage enum's required fields — so we emit a syntactically
    /// minimal IF2 stage. None of those values are read by the
    /// prior-draws code path.
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

[stages.draw]
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

    /// gh#86 RED test 1: fit.toml lists every estimated param but
    /// supplies NO `prior = { ... }` blocks. The model IR (sir_priors
    /// golden) has `~ <dist>` declarations for every param. After the
    /// fix, the IR's priors satisfy the requirement and N draws come
    /// back with the sampled values clamped to bounds.
    #[test]
    fn prior_draws_with_fit_toml_falls_back_to_ir_priors_when_missing() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = util::load_model(&ir_path).unwrap();

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

        let draws = generate_prior_draws(&fit_path, 7, 42, &model)
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

    /// A `--fit` file with no `[model]` table fails the `FitConfigV2`
    /// deserialize. The raw serde message is opaque ("missing field
    /// `model`"); the wrapped error must add the `[model]` shape hint so
    /// the user knows what kind of file `simulate --fit` wants.
    #[test]
    fn simulate_fit_malformed_config_error_carries_hint() {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ir_path = format!("{}/../../../ocaml/golden/sir_priors.ir.json", manifest);
        let (model, _) = util::load_model(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // A plausible "bare params" file a user might pass by mistake:
        // no [model] table at all.
        let bad = dir.path().join("bad.toml");
        std::fs::write(&bad, "beta = 0.4\ngamma = 0.2\n").unwrap();

        let err = generate_prior_draws(&bad.to_string_lossy(), 5, 1, &model)
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
        let (model, _) = util::load_model(&ir_path).unwrap();

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
        let draws = generate_prior_draws(&fit_path, 3, 7, &model)
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
                        "format": "tsv", "trajectory": true, "observations": false },
            "simulation": { "t_start": 0.0, "t_end": 1.0,
                            "time_semantics": "continuous", "dt": null, "rng_seed": null },
            "presets": [], "model_structure": null, "balance": null
          }
        }"#;
        let (_ir_dir, ir_path) = write_ir_fixture(ir_json);
        let (model, _) = util::load_model(&ir_path).unwrap();

        let dir = tempfile::tempdir().unwrap();
        // Both beta and gamma declared, neither has a prior.
        let estimate = "\
beta  = { bounds = [0.01, 2.0] }
gamma = { bounds = [0.05, 1.0] }
";
        let fit_path = write_fit_toml_for_prior_draws(
            dir.path(), &ir_path, estimate, "");

        let err = generate_prior_draws(&fit_path, 3, 1, &model).unwrap_err();
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
                        "format": "tsv", "trajectory": true, "observations": false },
            "simulation": { "t_start": 0.0, "t_end": 1.0,
                            "time_semantics": "continuous", "dt": null, "rng_seed": null },
            "presets": [], "model_structure": null, "balance": null
          }
        }"#;
        let (_ir_dir, ir_path) = write_ir_fixture(ir_json);
        let (model, _) = util::load_model(&ir_path).unwrap();

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
        let draws = generate_prior_draws(&fit_path, 200, 42, &model)
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
