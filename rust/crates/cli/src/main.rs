mod util;
mod hashing;
#[allow(dead_code)] // write sites wire up in commit 4/6 of the unified-output-tree rollout
mod run_meta;       // unified Run/RunKind ADT — see docs/dev/proposals/2026-04-19-unified-output-tree.md
#[allow(dead_code)] // full migration of path-construction sites lands in commit 4/6
mod run_paths;      // canonical output-path helpers
#[allow(dead_code)] // some write/read helpers wired up by follow-up commits (obs caching)
mod cas;
mod browse;
mod sampling;
#[allow(dead_code)]
mod batch;  // `simulate batch FILE` subcommand (formerly `experiment`)
mod serve;
mod eval;
#[allow(dead_code)]
mod pfilter; // used internally by fit runner for data loading
mod data;
mod fit;
mod compare;
pub mod version;

// Modules kept for internal use but with no direct CLI entry points:
#[allow(dead_code)] mod voi;
#[allow(dead_code)] mod if2;
mod profile;

/// Terminal formatting helpers. Pure ANSI SGR codes, no dependencies.
/// Respects NO_COLOR (https://no-color.org/) — when set, all formatting
/// is stripped and strings are returned unchanged.
mod term {
    fn enabled() -> bool { std::env::var("NO_COLOR").is_err() }
    fn wrap(code: &str, s: &str) -> String {
        if enabled() { format!("\x1b[{}m{}\x1b[0m", code, s) } else { s.to_string() }
    }
    pub fn dim(s: &str) -> String { wrap("2", s) }
    pub fn bold(s: &str) -> String { wrap("1", s) }
    #[allow(dead_code)]
    pub fn green(s: &str) -> String { wrap("32", s) }
    #[allow(dead_code)]
    pub fn yellow(s: &str) -> String { wrap("33", s) }
    #[allow(dead_code)]
    pub fn red(s: &str) -> String { wrap("31", s) }
    #[allow(dead_code)]
    pub fn cyan(s: &str) -> String { wrap("36", s) }
}

use sim::{write_diagnostics_tsv, warn_zero_firings};
use std::collections::HashMap;

fn print_main_help() -> ! {
    let d = term::dim;
    let b = term::bold;
    eprintln!("{}", b("camdl — compartmental model simulation and inference"));
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  {}    Run forward simulations (single or batch)", b("simulate"));
    eprintln!("  {}         Inference pipeline (MLE, posterior sampling, evaluation)", b("fit"));
    eprintln!("  {}     Standalone particle filter at fixed parameters", b("pfilter"));
    eprintln!("  {}         Standalone iterated filtering (IF2)", b("if2"));
    eprintln!("  {}     Profile likelihood via parallel IF2 over a grid", b("profile"));
    eprintln!("  {}     Compare fits by prequential scores (elpd, CRPS, PIT)", b("compare"));
    eprintln!("  {}     Compile .camdl → IR JSON", b("compile"));
    eprintln!("  {}       Type-check a .camdl model", b("check"));
    eprintln!("  {}     Print model structure", b("inspect"));
    eprintln!("  {}        Evaluate expressions against a model", b("eval"));
    eprintln!("  {}        Data utilities (split train/holdout)", b("data"));
    eprintln!("  {}       Launch web visualization server", b("serve"));
    eprintln!();
    eprintln!("Cache browsing (runs written by {}):", d("simulate --cas / simulate batch"));
    eprintln!("  {}        Browse cached runs as a table", b("list"));
    eprintln!("  {}        Show full metadata for one run", b("show"));
    eprintln!("  {}         Emit a cached trajectory or obs stream", b("cat"));
    eprintln!();
    eprintln!("Run {} for details on any command.", d("camdl <command> --help"));
    std::process::exit(0);
}

fn simulate_help() -> ! {
    let d = term::dim;
    let b = term::bold;
    eprintln!("{}", b("camdl simulate — forward simulation"));
    eprintln!();
    eprintln!("Usage:  camdl simulate MODEL [OPTIONS]");
    eprintln!();
    eprintln!("Examples:");
    eprintln!();
    eprintln!("  {}", d("# Basic simulation, output to stdout"));
    eprintln!("  camdl simulate sir.camdl --params p.toml --seed 42");
    eprintln!();
    eprintln!("  {}", d("# With a named scenario"));
    eprintln!("  camdl simulate sir.camdl --params p.toml --scenario with_sia --seed 42");
    eprintln!();
    eprintln!("  {}", d("# Generate synthetic observations"));
    eprintln!("  camdl simulate sir.camdl --params p.toml --obs cases.tsv --seed 42");
    eprintln!();
    eprintln!("  {}", d("# Posterior predictive check"));
    eprintln!("  camdl simulate sir.camdl --draws posterior.tsv --replicates 10 --obs ppc.tsv");
    eprintln!();
    eprintln!("  {}", d("# Prior predictive check (priors declared with ~ in the model)"));
    eprintln!("  camdl simulate sir.camdl --draws prior -n 500 --obs prior_pred.tsv");
    eprintln!();
    eprintln!("  {}", d("# Space-filling exploration from parameter bounds"));
    eprintln!("  camdl simulate sir.camdl --draws uniform -n 500 --obs explore.tsv");
    eprintln!();
    eprintln!("  {}", d("# Batch: multiple scenarios × seeds"));
    eprintln!("  camdl simulate sir.camdl --params p.toml --seeds 1:100 --scenario baseline,with_sia");
    eprintln!();
    eprintln!("  {}", d("# Batch sweep (multiple scenarios × seeds × sweep points)"));
    eprintln!("  camdl simulate batch batches/sweep.toml --parallel 8");
    eprintln!();
    eprintln!("  {}", d("# Verify a batch config before running"));
    eprintln!("  camdl simulate batch batches/sweep.toml --dry-run");
    eprintln!();
    eprintln!("  {}", d("# Cache output in content-addressable storage — repeat runs are instant"));
    eprintln!("  camdl simulate sir.camdl --params p.toml --seed 42 --cas");
    eprintln!("  camdl list        # browse cached runs");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --params FILE             Load parameter values (repeatable)");
    eprintln!("  --param NAME=VALUE        Override single parameter (repeatable)");
    eprintln!("  --param-vec PREFIX=FILE   Override indexed params from keyed TSV");
    eprintln!("  --table NAME=FILE         Supply external() table data");
    eprintln!("  --backend BACKEND         gillespie|tau_leap|chain_binomial|ode (default: chain_binomial)");
    eprintln!("  --dt DT                   Step size for discrete-time backends (default: 1.0)");
    eprintln!("  --seed N                  RNG seed (default: 1)");
    eprintln!("  --seeds SPEC              Multiple seeds: 1:100 or 1,2,42");
    eprintln!("  --scenario NAME[,NAME]    Named scenarios (comma-separated)");
    eprintln!("  --enable NAME             Enable intervention (mutually exclusive with --scenario)");
    eprintln!("  --disable NAME            Disable intervention");
    eprintln!("  --draws SOURCE            Parameter draws: path.tsv, uniform, or prior");
    eprintln!("  --fit FILE                fit.toml to override model IR priors for --draws prior");
    eprintln!("  -n N                      Number of draws (for --draws uniform/prior)");
    eprintln!("  --replicates N            Stochastic replicates per parameter point");
    eprintln!("  --obs FILE                Write synthetic observations (wide-format TSV)");
    eprintln!("  --obs-dir DIR             Write one TSV per observation stream");
    eprintln!("  --obs-only FILE           Like --obs but suppress trajectory output");
    eprintln!("  -o, --output FILE         Write trajectory to file (default: stdout)");
    eprintln!("  --parallel N              Concurrent runs (single-run + --cas)");
    eprintln!("  --dry-run                 Show resolved parameters and run plan, don't simulate");
    eprintln!("  --cas                     Cache output under {}/sims/.../seed_<n>/ (single-run only)", d("./results"));
    eprintln!("  --output-dir DIR          Root for --cas output (default: ./results)");
    eprintln!("  --force                   Re-run cached results");
    std::process::exit(0);
}

fn fit_help() -> ! {
    let d = term::dim;
    let b = term::bold;
    eprintln!("{}", b("camdl fit — inference pipeline"));
    eprintln!();
    eprintln!("Usage:  camdl fit <run|status|diff|new> [OPTIONS]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  {}       Execute inference stages from a fit.toml", b("run"));
    eprintln!("  {}    Show completion status for a fit directory", b("status"));
    eprintln!("  {}      Compare two fit.toml configs", b("diff"));
    eprintln!("  {}       Create a derived fit.toml with provenance", b("new"));
    eprintln!();
    eprintln!("Examples:");
    eprintln!();
    eprintln!("  {}", d("# Run all stages in a fit config"));
    eprintln!("  camdl fit run fits/01_all_free.toml");
    eprintln!();
    eprintln!("  {}", d("# Run a single stage with a specific seed"));
    eprintln!("  camdl fit run fits/01.toml --stage mle --seed 42");
    eprintln!();
    eprintln!("  {}", d("# Profile likelihood: sweep a fixed parameter"));
    eprintln!("  camdl fit run fits/01.toml --sweep \"rho=0.5,0.1,0.02\"");
    eprintln!();
    eprintln!("  {}", d("# Seed from a previous fit's results"));
    eprintln!("  camdl fit run fits/02.toml --starts-from results/fits/01-<hash>/real/fit_1/mle");
    eprintln!();
    eprintln!("  {}", d("# Check what's done"));
    eprintln!("  camdl fit status results/fits/01-<hash>");
    eprintln!();
    eprintln!("  {}", d("# See what changed between two configs"));
    eprintln!("  camdl fit diff fits/01.toml fits/02.toml");
    eprintln!();
    eprintln!("  {}", d("# Create a new config derived from an existing one"));
    eprintln!("  camdl fit new --from fits/01.toml fits/02_fix_beta.toml");
    eprintln!();
    eprintln!("Options (camdl fit run):");
    eprintln!("  --stage NAME            Run specific stage only");
    eprintln!("  --starts-from DIR       Override starts_from for target stage");
    eprintln!("  --seed N                RNG seed (default: random)");
    eprintln!("  --sweep \"NAME=V1,V2\"    Sweep over fixed param (repeatable; Cartesian product)");
    eprintln!("  --skip-chains N[,N]     Skip specific chain indices");
    eprintln!("  --resume                Resume partially completed sampling stage");
    eprintln!("  --parallel N            Concurrent sweep points");
    eprintln!("  --force                 Re-run (overwrite stale results)");
    std::process::exit(0);
}

fn pfilter_help() -> ! {
    let d = term::dim;
    let b = term::bold;
    eprintln!("{}", b("camdl pfilter — bootstrap particle filter at fixed parameters"));
    eprintln!();
    eprintln!("Usage:  camdl pfilter MODEL --params P.toml --data cases.tsv [OPTIONS]");
    eprintln!();
    eprintln!("Examples:");
    eprintln!();
    eprintln!("  {}", d("# Evaluate loglik at MLE with 5000 particles"));
    eprintln!("  camdl pfilter sir.camdl --params mle.toml --data cases.tsv --particles 5000");
    eprintln!();
    eprintln!("  {}", d("# With per-observation diagnostics"));
    eprintln!("  camdl pfilter sir.camdl --params mle.toml --data cases.tsv --particles 10000 --trace");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --params FILE           Parameter values");
    eprintln!("  --data FILE             Observation data (TSV)");
    eprintln!("  --particles N           Number of particles");
    eprintln!("  --dt DT                 Step size (default: 1.0)");
    eprintln!("  --seed N                RNG seed");
    eprintln!("  --trace                 Write per-observation diagnostics");
    eprintln!("  --flow NAME             Flow projection name");
    std::process::exit(0);
}

fn if2_help() -> ! {
    let d = term::dim;
    let b = term::bold;
    eprintln!("{}", b("camdl if2 — standalone iterated filtering"));
    eprintln!();
    eprintln!("Usage:  camdl if2 MODEL --params P.toml --data cases.tsv [OPTIONS]");
    eprintln!();
    eprintln!("Examples:");
    eprintln!();
    eprintln!("  {}", d("# Quick 8-chain scout with auto rw_sd"));
    eprintln!("  camdl if2 sir.camdl --params p.toml --data cases.tsv \\");
    eprintln!("      --rw-sd auto --chains 8 --particles 1000 --iterations 80");
    eprintln!();
    eprintln!("  {}", d("# With explicit rw_sd and fixed parameters"));
    eprintln!("  camdl if2 sir.camdl --params p.toml --data cases.tsv \\");
    eprintln!("      --rw-sd \"beta=0.01,gamma=0.005\" --fixed \"N0,mu\"");
    eprintln!();
    eprintln!("{}", d("Note: for structured pipelines, use `camdl fit run` with a fit.toml."));
    eprintln!("{}", d("      Standalone if2 is for quick ad-hoc runs and scripted workflows."));
    std::process::exit(0);
}

fn main() {
    let all_args: Vec<String> = std::env::args().skip(1).collect();
    if all_args.is_empty() { print_main_help(); }
    if all_args[0] == "--help" || all_args[0] == "-h" || all_args[0] == "help" {
        print_main_help();
    }

    // --version / -V anywhere in args
    if all_args.iter().any(|a| a == "--version" || a == "-V") {
        println!("{}", version::VERSION);
        return;
    }

    // --verbosity LEVEL: set log level (default: warn)
    // Levels: error, warn, info, debug, trace
    let verbosity = all_args.iter()
        .position(|a| a == "--verbosity")
        .and_then(|i| all_args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("warn");
    let log_level = match verbosity {
        "trace" => log::LevelFilter::Trace,
        "debug" => log::LevelFilter::Debug,
        "info" => log::LevelFilter::Info,
        "warn" => log::LevelFilter::Warn,
        "error" => log::LevelFilter::Error,
        other => {
            eprintln!("invalid verbosity level: '{}'. Use: error, warn, info, debug, trace", other);
            std::process::exit(1);
        }
    };
    env_logger::builder().filter_level(log_level).init();

    // Strip --verbosity LEVEL from args before dispatch
    let all_args: Vec<String> = {
        let mut filtered = Vec::new();
        let mut skip_next = false;
        for arg in &all_args {
            if skip_next { skip_next = false; continue; }
            if arg == "--verbosity" { skip_next = true; continue; }
            filtered.push(arg.clone());
        }
        filtered
    };

    // Dispatch on first argument
    match all_args[0].as_str() {
        // ── Compiler delegation (transparent camdlc invocation) ──
        "compile" => {
            let args: Vec<&str> = all_args[1..].iter().map(|s| s.as_str()).collect();
            util::delegate_to_camdlc(&args).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        "check" => {
            let mut args = vec!["check"];
            args.extend(all_args[1..].iter().map(|s| s.as_str()));
            util::delegate_to_camdlc(&args).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        "inspect" => {
            let mut args = vec!["inspect"];
            args.extend(all_args[1..].iter().map(|s| s.as_str()));
            util::delegate_to_camdlc(&args).unwrap_or_else(|e| {
                eprintln!("error: {}", e); std::process::exit(1);
            });
        }
        // ── Simulation ──
        "simulate" | "sim" => {
            let args = &all_args[1..];
            if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") { simulate_help(); }
            // `simulate batch FILE` and `simulate status FILE` are
            // sub-subcommands. Everything else is a single-run simulate.
            match args.first().map(String::as_str) {
                Some("batch")  => batch::cmd_batch_run(&args[1..]),
                Some("status") => batch::cmd_batch_status(&args[1..]),
                _              => run_simulate(args),
            }
        }
        // ── Inference ──
        "fit" => {
            match all_args.get(1).map(|s| s.as_str()) {
                Some("run")    => fit::cmd_fit_run_v2(&all_args[2..]),
                Some("status") => fit::cmd_fit_status(&all_args[2..]),
                Some("diff")   => fit::cmd_fit_diff(&all_args[2..]),
                Some("new")    => fit::cmd_fit_new(&all_args[2..]),
                Some("where")  => fit::cmd_fit_where(&all_args[2..]),
                _ => fit_help(),
            }
        }
        // ── Standalone inference tools ──
        "pfilter" => {
            if all_args[1..].is_empty() || all_args.iter().any(|a| a == "--help" || a == "-h") {
                pfilter_help();
            }
            pfilter::cmd_pfilter(&all_args[1..]);
        }
        "if2" | "mif2" => {
            if all_args[1..].is_empty() || all_args.iter().any(|a| a == "--help" || a == "-h") {
                if2_help();
            }
            if2::cmd_if2(&all_args[1..]);
        }
        "compare" => {
            compare::cmd_compare(&all_args[1..]);
        }
        "profile" => {
            // Rewired 2026-04-19 after a camdl-book profile-likelihood
            // chapter hit the gap: cmd_profile was fully built but
            // lost dispatch in d609654's CLI cleanup when
            // `fit run --sweep` was promoted as the replacement. The
            // replacement halts on the first scout-gate failure,
            // which defeats the primary profile-likelihood use case
            // (edge cells of a grid are expected to fail the gate).
            profile::cmd_profile(&all_args[1..]);
        }
        "voi" => {
            match all_args.get(1).map(|s| s.as_str()) {
                Some("run") => voi::cmd_voi_run(&all_args[2..]),
                _ => {
                    eprintln!("usage: camdl voi run VOI.toml");
                    std::process::exit(1);
                }
            }
        }
        // ── Utilities ──
        "eval" => {
            eval::cmd_eval(&all_args[1..]);
        }
        "data" => {
            match all_args.get(1).map(|s| s.as_str()) {
                Some("split") => data::cmd_data_split(&all_args[2..]),
                _ => {
                    eprintln!("usage: camdl data split FILE --at-time T [--train OUT] [--holdout OUT]");
                    std::process::exit(1);
                }
            }
        }
        "serve" => {
            serve::cmd_serve(&all_args[1..]);
        }
        "list" => {
            browse::cmd_list(&all_args[1..]);
        }
        "show" => {
            browse::cmd_show(&all_args[1..]);
        }
        "cat" => {
            browse::cmd_cat(&all_args[1..]);
        }
        _ => {
            // Accept bare "camdl FILE ..." for simulation
            run_simulate(&all_args);
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

fn run_simulate(args: &[String]) {
    let mut ir_path:     Option<String> = None;
    // Default backend is chain_binomial — matches `camdl fit`'s default
    // (see docs/dev/incidents/2026-04-19-backend-default-mismatch.md).
    // Users who want Gillespie pass --backend gillespie explicitly.
    let mut backend    = "chain_binomial".to_string();
    let mut dt         = 1.0_f64;
    // Track explicit vs default flags. The backend-provenance
    // guardrail (docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md)
    // only auto-matches the fit's backend when --backend was NOT
    // passed; when it was passed AND differs from the fit's backend,
    // we warn instead of silently overriding.
    let mut backend_explicit = false;
    let mut dt_explicit      = false;
    let mut seed       = 1_u64;
    let mut overrides: HashMap<String, f64> = HashMap::new();
    let mut table_files: HashMap<String, String> = HashMap::new();
    let mut params_files: Vec<String> = Vec::new();
    let mut scenario_names: Vec<String> = Vec::new();
    let mut adhoc_enable: Vec<String> = Vec::new();
    let mut adhoc_disable: Vec<String> = Vec::new();
    let mut seeds_spec: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut obs_path: Option<String> = None;
    let mut obs_dir: Option<String> = None;
    let mut obs_only: Option<String> = None;
    let mut replicates: usize = 1;
    let mut draws_path: Option<String> = None;
    let mut n_draws_arg: Option<usize> = None;
    let mut fit_path_for_draws: Option<String> = None;
    let mut dry_run = false;
    let mut cas_enabled = false;
    let mut output_dir_arg: Option<String> = None;

    // Collect --param-vec PREFIX=FILE entries for deferred validation after model load
    let mut set_vec_entries: Vec<(String, String)> = Vec::new();

    // Advance i and return the next argument as a slice. Exits with a
    // clean error (not a panic) when the flag is the final arg.
    let need = |i: &mut usize, flag: &str| -> String {
        *i += 1;
        if *i >= args.len() {
            eprintln!("error: {} requires a value", flag);
            std::process::exit(1);
        }
        args[*i].clone()
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend"  => { backend = need(&mut i, "--backend"); backend_explicit = true; }
            "--dt"       => {
                dt = need(&mut i, "--dt").parse().unwrap_or_else(|_| {
                    eprintln!("error: --dt needs a number"); std::process::exit(1);
                });
                dt_explicit = true;
            }
            "--seed"     => { seed    = need(&mut i, "--seed").parse().unwrap_or_else(|_| { eprintln!("error: --seed needs an integer"); std::process::exit(1); }); }
            "--params"   => { params_files.push(need(&mut i, "--params")); }
            "--scenario" => { scenario_names.extend(need(&mut i, "--scenario").split(',').map(|s| s.trim().to_string())); }
            "--seeds"    => { seeds_spec = Some(need(&mut i, "--seeds")); }
            "--enable"   => { adhoc_enable.push(need(&mut i, "--enable")); }
            "--disable"  => { adhoc_disable.push(need(&mut i, "--disable")); }
            "--param"     => {
                let kv = need(&mut i, "--param");
                let mut parts = kv.splitn(2, '=');
                let k = parts.next().unwrap_or_else(|| { eprintln!("error: --param needs NAME=VALUE"); std::process::exit(1); }).to_string();
                let v: f64 = parts.next().and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| { eprintln!("error: --param value must be a number"); std::process::exit(1); });
                overrides.insert(k, v);
            }
            "--param-vec" => {
                let kv = need(&mut i, "--param-vec");
                let mut parts = kv.splitn(2, '=');
                let prefix = parts.next().unwrap_or_else(|| { eprintln!("error: --param-vec needs PREFIX=FILE"); std::process::exit(1); }).to_string();
                let file   = parts.next().unwrap_or_else(|| { eprintln!("error: --param-vec needs PREFIX=FILE"); std::process::exit(1); }).to_string();
                set_vec_entries.push((prefix, file));
            }
            "--table"   => {
                let kv = need(&mut i, "--table");
                let mut parts = kv.splitn(2, '=');
                let k = parts.next().unwrap_or_else(|| { eprintln!("error: --table needs NAME=FILE"); std::process::exit(1); }).to_string();
                let v = parts.next().unwrap_or_else(|| { eprintln!("error: --table needs NAME=FILE"); std::process::exit(1); }).to_string();
                table_files.insert(k, v);
            }
            "--output" | "-o" => { output_path = Some(need(&mut i, "--output")); }
            "--obs"      => { obs_path = Some(need(&mut i, "--obs")); }
            "--obs-dir"  => { obs_dir = Some(need(&mut i, "--obs-dir")); }
            "--obs-only" => { obs_only = Some(need(&mut i, "--obs-only")); }
            "--replicates" => { replicates = need(&mut i, "--replicates").parse().unwrap_or_else(|_| { eprintln!("error: --replicates needs a positive integer"); std::process::exit(1); }); }
            "--draws" => { draws_path = Some(need(&mut i, "--draws")); }
            "-n" | "--n-draws" => { n_draws_arg = Some(need(&mut i, "-n").parse().unwrap_or_else(|_| { eprintln!("error: -n needs a positive integer"); std::process::exit(1); })); }
            "--fit" => { fit_path_for_draws = Some(need(&mut i, "--fit")); }
            "--dry-run" => { dry_run = true; }
            "--cas"        => { cas_enabled = true; }
            "--output-dir" => { output_dir_arg = Some(need(&mut i, "--output-dir")); }
            s if s.starts_with("--") => {
                eprintln!("error: unknown flag: {}", s);
                eprintln!("  run `camdl simulate --help` for usage");
                std::process::exit(1);
            }
            path => { ir_path = Some(path.to_string()); }
        }
        i += 1;
    }

    let ir_path = ir_path.unwrap_or_else(|| { eprintln!("missing IR file argument"); simulate_help(); });

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

    // Parse --seeds spec
    let seeds: Vec<u64> = if let Some(ref spec) = seeds_spec {
        parse_seeds_spec(spec).unwrap_or_else(|e| {
            eprintln!("error: --seeds: {}", e);
            std::process::exit(1);
        })
    } else {
        vec![seed]
    };
    if seeds_spec.is_some() && replicates > 1 {
        eprintln!("error: --seeds and --replicates are mutually exclusive.\n  \
                   --seeds provides explicit seed values.\n  \
                   --replicates generates N deterministic seeds from --seed.");
        std::process::exit(1);
    }
    // If using --seeds, replicates tracks seed count
    let replicates = if seeds_spec.is_some() { seeds.len() } else { replicates };

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
    // replicates, redirect users to `simulate batch` which has robust CAS.
    if cas_enabled {
        let multi_seeds = seeds.len() > 1;
        let multi_scenarios = scenario_list.len() > 1;
        let has_draws = draws_path.is_some();
        if multi_seeds || multi_scenarios || replicates > 1 || has_draws {
            eprintln!("error: --cas supports single runs only.");
            eprintln!("  For sweeps (multiple seeds/scenarios/draws/replicates), use");
            eprintln!("  `camdl simulate batch FILE` with a TOML config.");
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
            backend = prov.backend.clone();
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
    let mut cas_ctx: Option<CasCtx> = if cas_enabled {
        match prepare_cas_ctx(&base_sim_run, scenario_list[0].clone(), seeds[0],
                              &cas_root, from_fit_hash.clone()) {
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
            (true, CacheStatus::Hit { .. }) => {
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
            &ir_path, &base_sim_run.backend, dt, seed,
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
            let process_seed = if seeds_spec.is_some() {
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
                    let draw = sampler(projected_values[ti], &mut obs_rng);
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
        ctx.run.wall_time_seconds = cas_sim_t0.elapsed().as_secs_f64();
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

    let model_h = hashing::model_hash(&src);
    let sim_h = hashing::sim_hash(
        &model_h,
        &hashing::canonical_params(&base_params),
        &run.backend,
        run.dt,
    );
    let scen_h = hashing::scen_hash(&enable, &disable, &scen_params);
    let scenario_display = scenario_name.clone().unwrap_or_else(|| "baseline".to_string());
    let model_stem = hashing::path_stem_slug(&run.ir_path);
    let relative = run_paths::sim_run_rel(
        model_stem.as_deref(), &sim_h, &scenario_display, &scen_h, seed,
    );
    let run_dir = run_paths::sim_run_dir(
        std::path::Path::new(cas_root), model_stem.as_deref(),
        &sim_h, &scenario_display, &scen_h, seed,
    );

    let run_h = hashing::run_hash(&sim_h, &scen_h, seed);
    let run_record = run_meta::Run {
        hash: run_h,
        version: version::VERSION_SHORT.to_string(),
        created_at: cas::iso8601_utc(std::time::SystemTime::now()),
        argv: std::env::args().collect(),
        wall_time_seconds: 0.0, // filled in by caller after sim completes
        kind: run_meta::RunKind::Simulate(run_meta::SimulateMeta {
            model: run.ir_path.clone(),
            model_hash: model_h,
            scenario: scenario_display,
            sim_hash: sim_h,
            scen_hash: scen_h,
            seed,
            backend: run.backend.clone(),
            dt: run.dt,
            sweep_point: HashMap::new(), // --cas is single-run; no sweep
            from_fit_hash,
        }),
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
            eprintln!("error: DerivedExpr projection not yet supported for synthetic observations");
            std::process::exit(1);
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
    use fit::config_v2::{FitConfigV2, PriorSpec};

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
             Add prior = {{ dist = \"...\", ... }} to [estimate.{}]",
            missing.join(", "), missing[0]
        ));
    }

    let mut rng = sim::rng::StatefulRng::new(seed ^ SEED_MIX_PRIOR);
    let mut draws = Vec::with_capacity(n);

    for _ in 0..n {
        let mut row = HashMap::new();
        for (name, spec) in &config.estimate {
            let value = match spec.prior.as_ref().unwrap() {
                PriorSpec::LogNormal { mu, sigma } => {
                    // z ~ N(mu, sigma), value = exp(z)
                    let z = mu + sigma * rng.normal();
                    z.exp()
                }
                PriorSpec::Normal { mu, sigma } => {
                    mu + sigma * rng.normal()
                }
                PriorSpec::Beta { alpha, beta } => {
                    // Beta via ratio of Gammas: X/(X+Y) where X~Gamma(a), Y~Gamma(b)
                    use rand::prelude::Distribution;
                    let x = rand_distr::Gamma::new(*alpha, 1.0).unwrap()
                        .sample(rng.inner_mut());
                    let y = rand_distr::Gamma::new(*beta, 1.0).unwrap()
                        .sample(rng.inner_mut());
                    x / (x + y)
                }
                PriorSpec::Uniform => {
                    let (lo, hi) = spec.bounds;
                    lo + (hi - lo) * rng.uniform()
                }
                PriorSpec::HalfNormal { sigma } => {
                    (sigma * rng.normal()).abs()
                }
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
    backend: &str,
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
    let d = term::dim;
    let b = term::bold;

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
        eprintln!("  {} {}", d("scenario:"), "(baseline)");
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
            assert!(v >= 0.01 && v <= 2.0, "{} out of bounds", v);
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
            assert!(v >= 0.0 && v <= 1.0,
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
            assert!(b >= 0.01 && b <= 2.0, "beta out of bounds: {}", b);
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
