mod args;
mod util;
mod hashing;
mod run_meta;       // unified Run/RunKind ADT — see docs/dev/proposals/2026-04-19-unified-output-tree.md
mod run_paths;      // canonical output-path helpers
mod cas;
mod browse;
mod sampling;
mod batch;
mod eval;
mod pfilter;        // used internally by fit runner for data loading
mod data;
mod fit;
mod compare;
mod if2;
mod profile;
mod progress;
mod evidence;
mod survey;
mod landscape_html;
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
Common workflows:
  Simulate a model:        camdl simulate model.camdl --params p.toml
  Fit to data:             camdl fit run fit.toml
  Likelihood at θ:         camdl pfilter model.camdl --params p.toml --data cases.tsv
  Browse cached runs:      camdl list
  Diagnose a fit:          camdl fit summary <fit-dir>

Run `camdl <command> --help` for any subcommand."),
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

    /// Progress output mode for long-running subcommands. `auto` uses
    /// indicatif bars on a TTY, plain log lines otherwise; `plain` forces
    /// plain lines (use under `tee`, `ssh`, or CI).
    #[arg(long, global = true, default_value_t = args::types::ProgressMode::Auto,
          value_name = "MODE", help_heading = "Global options")]
    progress: args::types::ProgressMode,
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

    /// Standalone iterated filtering (IF2 / MIF2)
    #[command(alias = "mif2")]
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

    /// Evaluate time-dependent expressions against a model
    Eval(args::EvalArgs),

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

    /// Compile a .camdl model to IR JSON (delegates to camdlc)
    #[command(after_help = colored_help!("\
This subcommand forwards all arguments verbatim to the OCaml compiler
`camdlc`. Flags shown above belong to camdl; camdlc's own flags (e.g.
`--set NAME=VALUE`, `--json-errors`, `--no-dim-check`) are parsed by
camdlc itself. Run `camdlc --help` for the authoritative flag set.

Examples:
  # Compile a .camdl source to IR JSON (stdout)
  camdl compile sir.camdl > sir.ir.json

  # Override a parameter during compilation
  camdl compile sir.camdl --set beta=0.3

  # Machine-readable diagnostics
  camdl compile sir.camdl --json-errors
"))]
    Compile(Passthrough),

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
  --ascii             Strip ANSI color from output

Examples:
  # Default summary
  camdl inspect sir.camdl

  # Show loaded tables as well
  camdl inspect sir.camdl --tables

  # Transition rates only
  camdl inspect sir.camdl --transitions
"))]
    Inspect(Passthrough),
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
    /// Show completion status for a fit
    Status(args::FitStatusArgs),
    /// Render a single-fit interpretation summary (Â, gate verdict, MLE table)
    Summary(args::FitSummaryArgs),
    /// Compare two fit.toml configs
    Diff(args::FitDiffArgs),
    /// Cross-fit aggregator: walk results/fits/, render one row per fit
    Table(args::FitTableArgs),
    /// Derive a new fit.toml from an existing one
    New(args::FitNewArgs),
    /// Print the output directory path for a fit.toml
    Where(args::FitWhereArgs),
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

/// Captures all remaining argv tokens verbatim. Used only by Compile/Check/Inspect
/// which forward raw argv to camdlc and don't benefit from typed parsing.
#[derive(clap::Args)]
struct Passthrough {
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

fn main() {
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
    progress::init(cli.progress);

    match cli.command {
        Command::Simulate(a)            => run_simulate(&a),
        Command::Batch(BatchCmd::Run(a))    => batch::cmd_batch_run(&a),
        Command::Batch(BatchCmd::Status(a)) => batch::cmd_batch_status(&a),
        Command::Fit(FitCmd::Run(a))    => fit::cmd_fit_run_v2(&a),
        Command::Fit(FitCmd::Status(a)) => fit::cmd_fit_status(&a),
        Command::Fit(FitCmd::Summary(a))=> fit::cmd_fit_summary(&a),
        Command::Fit(FitCmd::Diff(a))   => fit::cmd_fit_diff(&a),
        Command::Fit(FitCmd::Table(a))  => fit::cmd_fit_table(&a),
        Command::Fit(FitCmd::New(a))    => fit::cmd_fit_new(&a),
        Command::Fit(FitCmd::Where(a))  => fit::cmd_fit_where(&a),
        Command::Label(a)               => fit::cmd_label(&a),
        Command::Pfilter(a)             => pfilter::cmd_pfilter(&a),
        Command::If2(a)                 => if2::cmd_if2(&a),
        Command::Profile(a)             => profile::cmd_profile(&a),
        Command::Survey(a)              => survey::cmd_survey(&a),
        Command::Eval(a)                => eval::cmd_eval(&a),
        Command::Data(DataCmd::Split(a))=> data::cmd_data_split(&a),
        Command::List(a)                => browse::cmd_list(&a),
        Command::Show(a)                => browse::cmd_show(&a),
        Command::Cat(a)                 => browse::cmd_cat(&a),
        Command::Compare(a)             => compare::cmd_compare(&a),
        Command::Compile(a) => {
            let refs: Vec<&str> = a.args.iter().map(String::as_str).collect();
            util::delegate_to_camdlc(&refs).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
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
    }
}

// Seed derivation constants for independent RNG streams.
// These are arbitrary coprime constants used to derive per-(draw, replicate)
// seeds from the base seed via XOR mixing. The specific values don't matter
// as long as they're distinct and nonzero — they ensure different (draw, rep)
// pairs get non-overlapping RNG streams.
const SEED_MIX_DRAW: u64 = 0x9e3779b97f4a7c15; // golden ratio fractional bits
const SEED_MIX_REP: u64  = 0x517cc1b727220a95; // more golden ratio mixing
use util::SEED_MIX_OBS;     // canonical home: util.rs
const SEED_MIX_UNIFORM: u64 = 0xd4a5_b1ce;      // uniform draws RNG
const SEED_MIX_PRIOR: u64  = 0x0014_b1ce;      // prior draws RNG

fn run_simulate(a: &args::SimulateArgs) {
    // ── Extract typed args into locals that match the rest of the function ─
    let ir_path          = a.model.to_string_lossy().into_owned();
    // Track explicit vs default flags for the backend-provenance guardrail.
    // Option<_> fields mean None ↔ not explicitly passed.
    let backend_explicit = a.backend.backend.is_some();
    let dt_explicit      = a.backend.dt.is_some();
    // Default is chain_binomial so `simulate` and `fit` agree at the
    // same MLE params (see docs/dev/incidents/2026-04-19-backend-default-mismatch.md).
    let mut backend      = a.backend.backend.unwrap_or(args::types::Backend::ChainBinomial);
    let mut dt           = a.backend.dt.unwrap_or(1.0_f64);
    let seed             = a.seed;
    let overrides: HashMap<String, f64> = a.model_overrides.param.iter()
        .map(|p| (p.name.clone(), p.value)).collect();
    let table_files: HashMap<String, String> = a.model_overrides.table.iter()
        .map(|t| (t.name.clone(), t.path.to_string_lossy().into_owned())).collect();
    let params_files: Vec<String> = a.model_overrides.params.iter()
        .map(|p| p.to_string_lossy().into_owned()).collect();
    let set_vec_entries: Vec<(String, String)> = a.param_vec.iter()
        .map(|pv| (pv.prefix.clone(), pv.file.clone())).collect();
    let scenario_names: Vec<String> = a.scenarios.iter()
        .flat_map(|s| s.split(',').map(|t| t.trim().to_string()))
        .collect();
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
    let replicates: usize            = a.replicates.unwrap_or(1);
    let draws_path: Option<String>   = a.draws.clone();
    let n_draws_arg: Option<usize>   = a.n_draws;
    let fit_path_for_draws: Option<String> = a.fit.as_ref().map(|p| p.to_string_lossy().into_owned());
    let dry_run     = a.dry_run;
    let cas_enabled = a.cas;
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
    let suppress_trajectory = obs_only.is_some();

    if replicates < 1 {
        eprintln!("error: --replicates must be >= 1");
        std::process::exit(1);
    }

    let want_obs = obs_path.is_some() || obs_dir.is_some();

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

    // --cas currently supports single-run invocations. For sweeps or
    // replicates, redirect users to `batch run` which has robust CAS.
    if cas_enabled {
        let multi_seeds = seeds.len() > 1;
        let multi_scenarios = scenario_list.len() > 1;
        let has_draws = draws_path.is_some();
        if multi_seeds || multi_scenarios || replicates > 1 || has_draws {
            eprintln!("error: --cas supports single runs only.");
            eprintln!("  For sweeps (multiple seeds/scenarios/draws/replicates), use");
            eprintln!("  `camdl batch run FILE` with a TOML config.");
            std::process::exit(1);
        }
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
    for pf in &params_files {
        let prov = match crate::fit::provenance::read_mle_provenance(pf) {
            Ok(Some(p)) => p,
            Ok(None) | Err(_) => continue,
        };
        from_fit_hash = prov.fit_hash.clone();

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
        ir_path: ir_path.clone(),
        params_files,
        overrides,
        set_vec_entries,
        table_files,
        scenario_name: None, // set per-scenario in the loop
        adhoc_enable,
        adhoc_disable,
        backend,
        dt,
        seed, // overridden per-replicate below
    };

    // ── Pre-flight: validate obs model availability ─────────────────────────
    // We need the model to check observation blocks, but we don't want to
    // run simulation twice. Do a dry load to validate, then run in the loop.
    if want_obs {
        let (model_check, _) = util::load_model(&ir_path).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        if model_check.observations.is_empty() {
            eprintln!("error: --obs/--obs-dir requested but model has no observations blocks");
            std::process::exit(1);
        }
        // Validate schedule compatibility for --obs (single file)
        if obs_path.is_some() && model_check.observations.len() > 1 {
            let schedules: Vec<_> = model_check.observations.iter()
                .map(|o| obs_schedule_times(&o.schedule, model_check.simulation.t_start, model_check.simulation.t_end))
                .collect();
            let all_same = schedules.windows(2).all(|w| w[0] == w[1]);
            if !all_same {
                let descs: Vec<String> = model_check.observations.iter()
                    .map(|o| format!("{}: {:?}", o.name, o.schedule))
                    .collect();
                eprintln!("error: observation streams have different schedules ({}).\n\
                           Use --obs-dir to produce one file per stream.",
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

    use std::io::Write;
    use owo_colors::OwoColorize;

    // ── CAS preparation (single-run --cas) ─────────────────────────────────
    // Compute hashes, resolve run path, check for cache hit. If the cached
    // trajectory already exists, we short-circuit: read it, emit to user's
    // destination, log 'cache hit' to stderr, and return.
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
    let mut cas_ctx: Option<CasCtx> = if cas_enabled {
        match prepare_cas_ctx(&base_sim_run, scenario_list[0].clone(), seeds[0],
                              &cas_root, from_fit_hash.clone(), label_arg.clone()) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                eprintln!("error preparing CAS: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let cas_sim_t0 = std::time::Instant::now();

    if let Some(ref ctx) = cas_ctx {
        // Hash-aware cache check: we don't trust "traj.tsv exists" alone.
        // If run.json is missing or its hash doesn't match the current
        // sim config, fall through to re-run rather than serve a stale
        // trajectory. Missing traj.tsv is always a miss regardless of
        // the metadata.
        use crate::run_meta::CacheStatus;
        let traj_present = cas::has_cached_traj(&ctx.run_dir);
        let meta_status = crate::run_meta::Run::check_cache(&ctx.run_dir, &ctx.run.hash);
        match (traj_present, &meta_status) {
            (true, CacheStatus::Hit) => {
                let cached = std::fs::read(ctx.run_dir.join("traj.tsv"))
                    .unwrap_or_else(|e| {
                        eprintln!("error reading cached traj.tsv: {}", e);
                        std::process::exit(1);
                    });
                if !suppress_trajectory {
                    match &output_path {
                        Some(path) => std::fs::write(path, &cached).unwrap_or_else(|e| {
                            eprintln!("cannot write {}: {}", path, e); std::process::exit(1);
                        }),
                        None => { std::io::stdout().write_all(&cached).unwrap(); }
                    }
                }
                eprintln!("{} {}", "cache hit:".bright_green().bold(), ctx.relative.cyan());
                return;
            }
            (true, CacheStatus::Stale { stored, current }) => {
                eprintln!("{} stored hash {} ≠ current {} — re-running",
                    "stale cache:".yellow().bold(),
                    &stored[..8.min(stored.len())],
                    &current[..8.min(current.len())]);
            }
            // (true, Miss) — traj.tsv exists without run.json. Happens
            // on interrupted runs or older binaries; treat as miss.
            // (false, _) — nothing cached; miss.
            _ => {}
        }
    }

    // ── Trajectory output setup ─────────────────────────────────────────────
    // When --cas is active, we buffer trajectory bytes via RunBuffer so we
    // can write them to both the user's destination and the CAS at end.
    let cas_buffer: Option<cas::RunBuffer> = cas_ctx.as_ref().map(|_| cas::RunBuffer::new());
    let mut traj_out: Option<Box<dyn Write>> = if !suppress_trajectory {
        Some(match (&cas_buffer, &output_path) {
            (Some(buf), _) => Box::new(buf.clone()),
            (None, Some(path)) => {
                let f = std::fs::File::create(path)
                    .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
                Box::new(std::io::BufWriter::new(f))
            }
            (None, None) => Box::new(std::io::BufWriter::new(std::io::stdout().lock())),
        })
    } else {
        None
    };
    let mut traj_header_written = false;

    // ── Load draws if --draws is specified ─────────────────────────────────
    let draws: Vec<HashMap<String, f64>> = if let Some(ref source) = draws_path {
        if source == "uniform" {
            let n = n_draws_arg.unwrap_or_else(|| {
                eprintln!("error: --draws uniform requires -n N");
                std::process::exit(1);
            });
            generate_uniform_draws(&ir_path, n, seed).unwrap_or_else(|e| {
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
                    // fit.toml prior source (overrides or supplements model priors)
                    generate_prior_draws(fit_path, n, seed).unwrap_or_else(|e| {
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
                    generate_prior_draws_from_ir(&ir_path, n, seed, &scenarios).unwrap_or_else(|e| {
                        eprintln!("error: {}", e);
                        std::process::exit(1);
                    })
                }
            }
        } else {
            // File path
            load_draws_tsv(source).unwrap_or_else(|e| {
                eprintln!("error loading draws: {}", e);
                std::process::exit(1);
            })
        }
    } else {
        // No draws — single point (parameters come from --params / --param)
        vec![HashMap::new()]
    };
    let n_draws = draws.len();
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
            &ir_path, base_sim_run.backend, dt, seed,
            &base_sim_run.params_files, &base_sim_run.overrides,
            &scenario_list, &seeds, &draws_path,
            n_draws, replicates, total_runs,
            &obs_path, &obs_dir, &obs_only,
        );
        return;
    }

    // ── Observation accumulators ────────────────────────────────────────────
    struct ObsRow { time: f64, replicate: usize, draw: usize, scenario: String, value: f64 }
    let mut obs_data: Vec<Vec<ObsRow>> = Vec::new(); // per-stream
    let mut obs_stream_names: Vec<String> = Vec::new();
    let mut obs_times_cache: Vec<Vec<f64>> = Vec::new();

    // ── Main loop: scenarios × draws × replicates ─────────────────────────
    let mut run_idx = 0usize;
    for scenario in &scenario_list {
    for (draw_idx, draw_overrides) in draws.iter().enumerate() {
        for rep in 0..replicates {
            let process_seed = if seeds_spec_given {
                seeds[rep] // explicit seeds
            } else if total_runs == 1 {
                seed
            } else {
                seed ^ ((draw_idx as u64).wrapping_mul(SEED_MIX_DRAW))
                     ^ ((rep as u64).wrapping_mul(SEED_MIX_REP))
            };
            let obs_seed = process_seed ^ SEED_MIX_OBS;

            // Merge draw overrides with CLI --param overrides
            let mut combined_overrides = base_sim_run.overrides.clone();
            combined_overrides.extend(draw_overrides.iter().map(|(k, v)| (k.clone(), *v)));

            let mut sim_run = util::SimRun { seed: process_seed, ..Default::default() };
            sim_run.ir_path = base_sim_run.ir_path.clone();
            sim_run.params_files = base_sim_run.params_files.clone();
            sim_run.overrides = combined_overrides;
            sim_run.set_vec_entries = base_sim_run.set_vec_entries.clone();
            sim_run.table_files = base_sim_run.table_files.clone();
            sim_run.scenario_name = scenario.clone();
            sim_run.adhoc_enable = base_sim_run.adhoc_enable.clone();
            sim_run.adhoc_disable = base_sim_run.adhoc_disable.clone();
            sim_run.backend = base_sim_run.backend.clone();
            sim_run.dt = base_sim_run.dt;

        let (traj, model) = util::run_simulation(&sim_run).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });

        // Write diagnostics (first run only)
        if run_idx == 0 && !traj.transition_diagnostics.is_empty() {
            match write_diagnostics_tsv("diagnostics.tsv", &traj.transition_diagnostics) {
                Ok(zero_count) => {
                    if zero_count > 0 { warn_zero_firings(&traj.transition_diagnostics); }
                }
                Err(e) => eprintln!("warning: could not write diagnostics.tsv: {}", e),
            }
        }

        // ── Trajectory output ───────────────────────────────────────────────
        if let Some(ref mut out) = traj_out {
            let int_names: Vec<&str> = model.compartments.iter()
                .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
                .map(|c| c.name.as_str()).collect();
            let real_names: Vec<&str> = model.compartments.iter()
                .filter(|c| c.kind == ir::model::CompartmentKind::Real)
                .map(|c| c.name.as_str()).collect();
            let tr_names: Vec<&str> = model.transitions.iter().map(|t| t.name.as_str()).collect();

            if !traj_header_written {
                writeln!(out, "# {}", version::VERSION).unwrap();
                if total_runs > 1 { write!(out, "replicate\t").unwrap(); }
                if n_scenarios > 1 { write!(out, "scenario\t").unwrap(); }
                if n_draws > 1 { write!(out, "draw\t").unwrap(); }
                write!(out, "t").unwrap();
                for n in &int_names  { write!(out, "\t{}", n).unwrap(); }
                for n in &real_names { write!(out, "\t{}", n).unwrap(); }
                for n in &tr_names   { write!(out, "\tflow_{}", n).unwrap(); }
                writeln!(out).unwrap();
                traj_header_written = true;
            }

            for snap in &traj.snapshots {
                if total_runs > 1 { write!(out, "{}\t", run_idx + 1).unwrap(); }
                if n_scenarios > 1 { write!(out, "{}\t", scenario.as_deref().unwrap_or("baseline")).unwrap(); }
                if n_draws > 1 { write!(out, "{}\t", draw_idx + 1).unwrap(); }
                write!(out, "{}", snap.t).unwrap();
                for &c in &snap.int_state.counts  { write!(out, "\t{}", c).unwrap(); }
                for &v in &snap.real_state.values { write!(out, "\t{:.4}", v).unwrap(); }
                for &f in &snap.flows.counts      { write!(out, "\t{}", f).unwrap(); }
                writeln!(out).unwrap();
            }
        }

        // ── Observation sampling ───────────��────────────────────────────────
        if want_obs {
            let compiled = std::sync::Arc::new(
                sim::CompiledModel::new(model.clone()).unwrap_or_else(|e| {
                    eprintln!("error compiling model for obs: {:?}", e);
                    std::process::exit(1);
                })
            );
            let params = compiled.default_params.clone();
            let mut obs_rng = sim::rng::StatefulRng::new(obs_seed);

            // Initialize stream names and obs data on first run
            if run_idx == 0 {
                for obs_model in &model.observations {
                    obs_stream_names.push(obs_model.name.clone());
                    obs_data.push(Vec::new());
                    let times = obs_schedule_times(
                        &obs_model.schedule,
                        model.simulation.t_start,
                        model.simulation.t_end,
                    );
                    obs_times_cache.push(times);
                }
            }

            for (si, obs_ir) in model.observations.iter().enumerate() {
                let sampler = sim::inference::obs_model::compile_obs_sample_pf(
                    obs_ir, compiled.clone(), &params,
                );
                let obs_times = &obs_times_cache[si];
                let projected_values = project_all_obs_times(
                    &traj, obs_ir, &model, obs_times,
                );

                for (ti, &obs_t) in obs_times.iter().enumerate() {
                    // GH #6 fix: pass the actual compartment state at
                    // the obs time so the likelihood p/mean expressions
                    // can resolve references like `N = S + I + R` —
                    // otherwise the sampler uses a zero-filled scratch
                    // and PopSum-valued denominators explode to NaN.
                    let snap = snap_at(&traj, obs_t);
                    let draw = sampler(
                        projected_values[ti], &snap.int_state.counts, &mut obs_rng,
                    );
                    obs_data[si].push(ObsRow {
                        time: obs_t,
                        replicate: run_idx + 1,
                        draw: draw_idx + 1,
                        scenario: scenario.as_deref().unwrap_or("baseline").to_string(),
                        value: draw,
                    });
                }
            }
        }

            run_idx += 1;
        } // end replicates
    } // end draws
    } // end scenarios

    // Flush trajectory output
    drop(traj_out);
    if let Some(ref path) = output_path {
        eprintln!("trajectory written to {}", path);
    }

    // ── CAS write (single-run --cas on cache miss) ─────────────────────────
    if let (Some(ctx), Some(buf)) = (cas_ctx.as_mut(), cas_buffer.as_ref()) {
        let bytes = buf.bytes();
        ctx.run.status = run_meta::RunStatus::Completed {
            wall_time_seconds: cas_sim_t0.elapsed().as_secs_f64(),
        };
        // Mirror to user's destination
        if !suppress_trajectory {
            match &output_path {
                Some(path) => std::fs::write(path, &bytes).unwrap_or_else(|e| {
                    eprintln!("cannot write {}: {}", path, e); std::process::exit(1);
                }),
                None => {
                    std::io::stdout().write_all(&bytes).unwrap();
                }
            }
        }
        // Write to CAS
        std::fs::create_dir_all(&ctx.run_dir).unwrap_or_else(|e| {
            eprintln!("cannot create CAS dir {}: {}", ctx.run_dir.display(), e);
            std::process::exit(1);
        });
        std::fs::write(ctx.run_dir.join("traj.tsv"), &bytes).unwrap_or_else(|e| {
            eprintln!("cannot write traj.tsv: {}", e); std::process::exit(1);
        });
        ctx.run.write(&ctx.run_dir).unwrap_or_else(|e| {
            eprintln!("cannot write run.json: {}", e); std::process::exit(1);
        });
        eprintln!("{} {}", "cached:".bright_green().bold(), ctx.relative.cyan());
    }

    // ── Write observation output ────────────���───────────────────────────────
    if want_obs && !obs_data.is_empty() {
        let multi_rep = total_runs > 1;

        // --obs: single wide-format file
        if let Some(ref path) = obs_path {
            let f = std::fs::File::create(path)
                .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
            let mut out = std::io::BufWriter::new(f);

            // Header
            if multi_rep { write!(out, "replicate\t").unwrap(); }
            if n_scenarios > 1 { write!(out, "scenario\t").unwrap(); }
            if n_draws > 1 { write!(out, "draw\t").unwrap(); }
            write!(out, "time").unwrap();
            for name in &obs_stream_names { write!(out, "\t{}", name).unwrap(); }
            writeln!(out).unwrap();

            // All streams share the same schedule (validated above).
            // Rows: iterate over (replicate, time), collect values across streams.
            let n_times = obs_times_cache[0].len();
            for run in 0..total_runs {
                for ti in 0..n_times {
                    let row_idx = run * n_times + ti;
                    if multi_rep { write!(out, "{}\t", run + 1).unwrap(); }
                    if n_scenarios > 1 { write!(out, "{}\t", obs_data[0][row_idx].scenario).unwrap(); }
                    if n_draws > 1 { write!(out, "{}\t", obs_data[0][row_idx].draw).unwrap(); }
                    write!(out, "{}", obs_data[0][row_idx].time).unwrap();
                    for si in 0..obs_stream_names.len() {
                        let val = obs_data[si][row_idx].value;
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

        // --obs-dir: one file per stream
        if let Some(ref dir) = obs_dir {
            for (si, name) in obs_stream_names.iter().enumerate() {
                let path = format!("{}/{}.tsv", dir, name);
                let f = std::fs::File::create(&path)
                    .unwrap_or_else(|e| { eprintln!("cannot create {}: {}", path, e); std::process::exit(1); });
                let mut out = std::io::BufWriter::new(f);

                if multi_rep { write!(out, "replicate\t").unwrap(); }
                if n_scenarios > 1 { write!(out, "scenario\t").unwrap(); }
                if n_draws > 1 { write!(out, "draw\t").unwrap(); }
                writeln!(out, "time\t{}", name).unwrap();

                for row in &obs_data[si] {
                    if multi_rep { write!(out, "{}\t", row.replicate).unwrap(); }
                    if n_scenarios > 1 { write!(out, "{}\t", row.scenario).unwrap(); }
                    if n_draws > 1 { write!(out, "{}\t", row.draw).unwrap(); }
                    let val = row.value;
                    if val == val.round() && val.abs() < 1e15 {
                        writeln!(out, "{}\t{}", row.time, val as i64).unwrap();
                    } else {
                        writeln!(out, "{}\t{:.6}", row.time, val).unwrap();
                    }
                }
                drop(out);
                eprintln!("observations written to {}", path);
            }
        }
    }
}

// ── Observation helpers ─────────────────────────────────────��───────────────

// ── CAS preparation (single-run --cas) ──────────────────────────────────────

/// Everything needed to write a single-run CAS entry: resolved run
/// directory + metadata template. Built before simulation so we can
/// check for a cache hit and skip work if possible.
struct CasCtx {
    /// Relative path under `<root>/sims/`, e.g.
    /// `abc12345/baseline-def45678/seed_42`. Logged to stderr.
    relative: String,
    /// Absolute path to the run directory.
    run_dir: std::path::PathBuf,
    /// Metadata to write to `run.json` after a successful run.
    run: run_meta::Run,
}

/// Resolve the CAS run directory and build a `RunMeta` template for a
/// single (model, scenario, seed) triple. Mirrors the relevant bits of
/// `util::run_simulation`'s model-load + scenario-resolve pipeline so
/// the hash inputs match exactly without re-running the sim.
fn prepare_cas_ctx(
    run: &util::SimRun,
    scenario_name: Option<String>,
    seed: u64,
    cas_root: &str,
    from_fit_hash: Option<String>,
    label: Option<String>,
) -> Result<CasCtx, String> {
    // Load IR source + parse model
    let (ir_path_resolved, _tmp) = util::resolve_ir_path(&run.ir_path)?;
    let src = std::fs::read_to_string(&ir_path_resolved)
        .map_err(|e| format!("cannot read {}: {}", ir_path_resolved, e))?;
    let mut model: ir::Model = serde_json::from_str(&src)
        .map_err(|e| format!("IR parse error: {}", e))?;

    // Apply --params files and --param overrides to collect base_params
    // (scenario deltas are the other side of the cache key — don't apply here).
    for path in &run.params_files {
        util::apply_params_file(&mut model, path)?;
    }
    for (k, v) in &run.overrides {
        if let Some(p) = model.parameters.iter_mut().find(|p| &p.name == k) {
            p.value = Some(*v);
        }
    }
    // Bounds + finite-value check after all override paths resolved (gh#31).
    // Catches violations early — before any CAS-cache work — rather than
    // letting a downstream `run_simulation` re-validate post-hash.
    util::validate_parameter_values(&model)?;
    let base_params: HashMap<String, f64> = model.parameters.iter()
        .filter_map(|p| p.value.map(|v| (p.name.clone(), v)))
        .collect();

    // Scenario delta
    let (enable, disable, scen_params) = if let Some(ref name) = scenario_name {
        let preset = model.presets.iter().find(|p| p.name == *name).cloned()
            .ok_or_else(|| {
                let available: Vec<&str> = model.presets.iter().map(|p| p.name.as_str()).collect();
                format!("scenario '{}' not found. Available: {}",
                    name,
                    if available.is_empty() { "(none)".into() } else { available.join(", ") })
            })?;
        let params: HashMap<String, f64> =
            preset.params.iter().map(|(k, v)| (k.clone(), *v)).collect();
        (preset.enable.clone(), preset.disable.clone(), params)
    } else {
        (run.adhoc_enable.clone(), run.adhoc_disable.clone(), HashMap::new())
    };

    let inputs = cas::sim_inputs::SimulateInputs {
        model_path:           run.ir_path.clone(),
        model_stem:           hashing::path_stem_slug(&run.ir_path),
        scenario:             scenario_name.clone().unwrap_or_else(|| "baseline".to_string()),
        model_hash:           hashing::model_hash(&src),
        base_params_canonical: hashing::canonical_params(&base_params),
        backend:              run.backend.clone(),
        dt:                   run.dt,
        enable, disable, scen_params,
        seed,
        from_fit_hash,
        sweep_point:          HashMap::new(),  // single --cas: no sweep
    };

    use cas::typed::CasInputs;
    let run_dir = inputs.cas_path(std::path::Path::new(cas_root));
    let relative = run_paths::sim_run_rel(
        inputs.model_stem.as_deref(),
        &inputs.sim_hash_str(),
        &inputs.scenario,
        &inputs.scen_hash_str(),
        seed,
    );
    let run_record = run_meta::Run {
        hash:              inputs.content_hash().full().to_string(),
        version:           version::VERSION_SHORT.to_string(),
        created_at:        cas::iso8601_utc(std::time::SystemTime::now()),
        argv:              std::env::args().collect(),
        status: run_meta::RunStatus::Running,
        label,
        kind:              inputs.run_kind(),
    };

    Ok(CasCtx { relative, run_dir, run: run_record })
}

/// Generate observation times from an IR schedule.
pub(crate) fn obs_schedule_times(
    schedule: &ir::observation::ObservationSchedule,
    t_start: f64,
    t_end: f64,
) -> Vec<f64> {
    match schedule {
        ir::observation::ObservationSchedule::Regular(reg) => {
            let mut times = Vec::new();
            let mut t = reg.start;
            while t <= reg.end + 1e-9 {
                times.push(t);
                t += reg.step;
            }
            times
        }
        ir::observation::ObservationSchedule::AtTimes(times) => times.clone(),
        ir::observation::ObservationSchedule::FromData => {
            // In simulate mode there's no data — generate a reasonable grid
            // using the simulation output times (every dt from t_start to t_end).
            eprintln!("warning: observation schedule is 'from_data' but no data provided; \
                       using simulation output grid (every 1 unit from {} to {})", t_start, t_end);
            let mut times = Vec::new();
            let mut t = t_start + 1.0;
            while t <= t_end + 1e-9 {
                times.push(t);
                t += 1.0;
            }
            times
        }
    }
}

/// Project observable quantities from a trajectory at all observation times.
///
/// For CumulativeFlow: accumulate per-snapshot flows, difference between
/// consecutive observation times to get per-interval flow counts.
/// For CurrentPop/CurrentPopSum: read state at snapshot closest to each obs time.
pub(crate) fn project_all_obs_times(
    traj: &sim::Trajectory,
    obs_ir: &ir::observation::ObservationModel,
    model: &ir::Model,
    obs_times: &[f64],
) -> Vec<f64> {
    match &obs_ir.projection {
        ir::observation::Projection::CumulativeFlow(flow_name) => {
            let flow_indices: Vec<usize> = model.transitions.iter().enumerate()
                .filter(|(_, tr)| tr.name == *flow_name || tr.name.starts_with(&format!("{}_", flow_name)))
                .map(|(i, _)| i)
                .collect();

            // Build running cumulative flow at each snapshot time
            let mut cum_at_snap: Vec<(f64, u64)> = Vec::with_capacity(traj.snapshots.len());
            let mut running = 0u64;
            for snap in &traj.snapshots {
                for &fi in &flow_indices {
                    running += snap.flows.counts[fi];
                }
                cum_at_snap.push((snap.t, running));
            }

            // For each obs time, find cumulative flow up to that time.
            // Then difference consecutive obs times.
            let mut cum_at_obs = Vec::with_capacity(obs_times.len());
            let mut snap_idx = 0;
            for &obs_t in obs_times {
                // Advance to last snapshot at or before obs_t
                while snap_idx + 1 < cum_at_snap.len()
                    && cum_at_snap[snap_idx + 1].0 <= obs_t + 1e-9
                {
                    snap_idx += 1;
                }
                cum_at_obs.push(if snap_idx < cum_at_snap.len() && cum_at_snap[snap_idx].0 <= obs_t + 1e-9 {
                    cum_at_snap[snap_idx].1
                } else {
                    0
                });
            }

            // Difference: flow in interval (prev_obs_t, obs_t]
            let mut result = Vec::with_capacity(obs_times.len());
            let mut prev_cum = 0u64;
            for &cum in &cum_at_obs {
                result.push((cum - prev_cum) as f64);
                prev_cum = cum;
            }
            result
        }
        ir::observation::Projection::CurrentPop(comp_name) => {
            let loc = resolve_comp_local(model, &obs_ir.name, comp_name);
            obs_times.iter().map(|&obs_t| {
                let snap = snap_at(traj, obs_t);
                read_comp(snap, &loc)
            }).collect()
        }
        ir::observation::Projection::CurrentPopSum(names) => {
            let locs: Vec<_> = names.iter()
                .map(|name| resolve_comp_local(model, &obs_ir.name, name))
                .collect();
            obs_times.iter().map(|&obs_t| {
                let snap = snap_at(traj, obs_t);
                locs.iter().map(|loc| read_comp(snap, loc)).sum()
            }).collect()
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
            obs_times.iter().map(|&obs_t| {
                let snap = snap_at(traj, obs_t);
                eval_stream_projection(
                    &stream_proj, empty_flows, &snap.int_state.counts,
                    &params, &compiled, &real_s, obs_t,
                )
            }).collect()
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

fn snap_at(traj: &sim::Trajectory, obs_t: f64) -> &sim::Snapshot {
    traj.snapshots.iter().rev()
        .find(|s| s.t <= obs_t + 1e-9)
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
            let val = if let Some((lo, hi)) = p.bounds {
                lo + (hi - lo) * rng.uniform()
            } else if let Some(v) = p.value {
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
fn generate_prior_draws(
    fit_path: &str,
    n: usize,
    seed: u64,
) -> Result<Vec<HashMap<String, f64>>, String> {
    use fit::config_v2::FitConfigV2;
    use ir::parameter::PriorDist;

    let config = FitConfigV2::load(fit_path)?;
    let fixed = config.fixed.resolve()?;

    // Check all estimated params have priors
    let missing: Vec<&str> = config.estimate.iter()
        .filter(|(_, spec)| spec.prior.is_none())
        .map(|(name, _)| name.as_str())
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "--draws prior requires priors for all estimated parameters.\n  \
             Missing priors: {}\n  \
             Add prior = {{ <dist> = {{ ... }} }} to [estimate.{}] \
             (e.g. `prior = {{ log_normal = {{ mu = 0, sigma = 1 }} }}`).",
            missing.join(", "), missing[0]
        ));
    }

    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_PRIOR);
    let mut draws = Vec::with_capacity(n);

    for _ in 0..n {
        let mut row = HashMap::new();
        for (name, spec) in &config.estimate {
            let value = match spec.prior.as_ref().unwrap() {
                PriorDist::LogNormal(p) => {
                    // z ~ N(mu, sigma), value = exp(z)
                    let z = p.mu + p.sigma * rng.normal();
                    z.exp()
                }
                PriorDist::Normal(p) => {
                    p.mean + p.sd * rng.normal()
                }
                PriorDist::Beta(p) => {
                    // Beta via ratio of Gammas: X/(X+Y) where X~Gamma(a), Y~Gamma(b)
                    use rand::prelude::Distribution;
                    let x = rand_distr::Gamma::new(p.alpha, 1.0).unwrap()
                        .sample(rng.inner_mut());
                    let y = rand_distr::Gamma::new(p.beta, 1.0).unwrap()
                        .sample(rng.inner_mut());
                    x / (x + y)
                }
                PriorDist::Uniform(p) => {
                    p.lower + (p.upper - p.lower) * rng.uniform()
                }
                PriorDist::HalfNormal(p) => {
                    (p.sigma * rng.normal()).abs()
                }
                PriorDist::Gamma(p) => {
                    use rand::prelude::Distribution;
                    rand_distr::Gamma::new(p.shape, 1.0 / p.rate).unwrap()
                        .sample(rng.inner_mut())
                }
                PriorDist::Exponential(p) => {
                    use rand::prelude::Distribution;
                    rand_distr::Exp::new(p.rate).unwrap()
                        .sample(rng.inner_mut())
                }
                PriorDist::Fixed(v) => *v,
            };
            let clamped = value.clamp(spec.bounds.0, spec.bounds.1);
            row.insert(name.clone(), clamped);
        }
        for (name, val) in &fixed {
            row.insert(name.clone(), *val);
        }
        draws.push(row);
    }

    eprintln!("generated {} prior draws from {} ({} estimated + {} fixed params)",
        n, fit_path, config.estimate.len(), fixed.len());
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
                p.value = Some(*v);
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
        .filter(|p| p.prior.is_none() && p.value.is_none())
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
            let value = match &p.prior {
                Some(pd) => {
                    if i == 0 { n_sampled += 1; }
                    let (v, rejected) = sample_with_bounds(pd, p.bounds, &mut rng, &p.name)?;
                    if rejected > 0 {
                        *reject_counts.entry(p.name.as_str()).or_insert(0) += rejected;
                    }
                    v
                }
                None => {
                    if i == 0 { n_fixed += 1; }
                    p.value.expect("missing check above guarantees value exists")
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

/// Load a draws TSV file. Each row is a complete parameter vector.
/// Column names must match model parameter names.
/// Returns Vec<HashMap<param_name, value>>.
fn load_draws_tsv(path: &str) -> Result<Vec<HashMap<String, f64>>, String> {
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
        let mut row = HashMap::new();
        for (col, field) in col_names.iter().zip(fields.iter()) {
            let val: f64 = field.parse()
                .map_err(|_| format!(
                    "draws file line {}, column '{}': cannot parse '{}' as number",
                    line_num + 2, col, field
                ))?;
            row.insert(col.to_string(), val);
        }
        draws.push(row);
    }

    if draws.is_empty() {
        return Err(format!("draws file has header but no data rows: {}", path));
    }
    Ok(draws)
}

/// Print a dry run summary: resolved parameters with provenance.
#[allow(clippy::too_many_arguments)]
fn print_dry_run(
    ir_path: &str,
    backend: args::types::Backend,
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
        // Point/single mode: show resolved parameter values with provenance
        match util::load_model(ir_path) {
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
                    if let Some(v) = p.value {
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.ir.json");
        std::fs::write(&path, json).unwrap();
        (dir, path.to_string_lossy().into_owned())
    }

    /// Minimal IR with a single scalar parameter carrying the supplied
    /// bounds and prior JSON. Used by the bounds-rejection and scenario
    /// tests that need tight control over the IR.
    fn ir_with_prior(name: &str, bounds: &str, prior_json: &str, extras: &str) -> String {
        format!(r#"{{
            "name": "t", "version": "0.3", "time_unit": "days",
            "description": null, "origin": null,
            "compartments": [{{ "name": "S", "kind": "integer" }}],
            "transitions": [], "ode_equations": [], "time_functions": [],
            "tables": [], "interventions": [], "observations": [],
            "parameters": [
              {{ "name": "{name}", "value": null, "bounds": {bounds},
                 "prior": {prior_json}, "transform": null, "initial_value": null,
                 "param_kind": "rate", "param_dim": null }}
              {extras}
            ],
            "initial_conditions": {{ "explicit": {{ "S": 1.0 }} }},
            "output": {{ "times": {{ "at_times": [0.0, 1.0] }},
                         "format": "tsv", "trajectory": true, "observations": false }},
            "simulation": {{ "t_start": 0.0, "t_end": 1.0, "time_semantics": "continuous",
                             "dt": null, "rng_seed": null }},
            "presets": [], "model_structure": null, "balance": null
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
        let json = r#"{
            "name": "t", "version": "0.3", "time_unit": "days",
            "description": null, "origin": null,
            "compartments": [{ "name": "S", "kind": "integer" }],
            "transitions": [], "ode_equations": [], "time_functions": [],
            "tables": [], "interventions": [], "observations": [],
            "parameters": [
              { "name": "beta", "value": null, "bounds": [0.01, 2.0],
                "prior": { "log_normal": { "mu": -1.0, "sigma": 0.3 } },
                "transform": null, "initial_value": null,
                "param_kind": "rate", "param_dim": null },
              { "name": "N0", "value": null, "bounds": [100.0, 10000.0],
                "prior": null, "transform": null, "initial_value": null,
                "param_kind": "count", "param_dim": null }
            ],
            "initial_conditions": { "explicit": { "S": 1.0 } },
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
        let seed = 42u64;
        let draw_idx = 3u64;
        let rep = 7u64;
        let s1 = seed ^ draw_idx.wrapping_mul(SEED_MIX_DRAW) ^ rep.wrapping_mul(SEED_MIX_REP);
        let s2 = seed ^ draw_idx.wrapping_mul(SEED_MIX_DRAW) ^ rep.wrapping_mul(SEED_MIX_REP);
        assert_eq!(s1, s2, "same inputs must produce same seed");

        // Different draw_idx → different seed
        let s3 = seed ^ 4u64.wrapping_mul(SEED_MIX_DRAW) ^ rep.wrapping_mul(SEED_MIX_REP);
        assert_ne!(s1, s3);

        // Obs seed independent
        let obs1 = s1 ^ SEED_MIX_OBS;
        assert_ne!(s1, obs1);
    }
}
