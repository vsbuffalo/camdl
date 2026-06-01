use std::collections::HashMap;
use ir::intervention::Intervention;
use sim::{
    CompiledModel, GillespieSim, TauLeapSim, ChainBinomialSim, OdeSim,
    config::{GillespieConfig, TauLeapConfig, ChainBinomialConfig, OdeConfig},
    simulate::Simulate,
    Trajectory,
};

/// Observation RNG decorrelation mask. Any code path that samples
/// synthetic observations on top of a simulated trajectory must seed
/// its observation RNG with `process_seed ^ SEED_MIX_OBS` so that the
/// obs stream is independent of the process stream. Shared between
/// `camdl simulate --obs` / `--obs-only` and the `[synthetic]` data
/// generator in `fit run` so that the same nominal seed produces the
/// same observation bytes regardless of which path generated them.
pub const SEED_MIX_OBS: u64 = 0xa5a5a5a5a5a5;

/// Per-(point, replicate) seed-mix constants. Private: the only public
/// surface is [`mix_cell_seed`], so the derivation has exactly one home
/// and callers cannot drift the constants. (These were previously
/// duplicated across `engine.rs`, `main.rs`, and `survey.rs`.) The values
/// are arbitrary golden-ratio fractional bits — any pairwise-distinct mix
/// works, since ChaCha8 maps seeds to independent streams.
const SEED_MIX_DRAW: u64 = 0x9e3779b97f4a7c15;
const SEED_MIX_REP:  u64 = 0x517cc1b727220a95;

/// Canonical per-cell process-seed mix. A run cell at param/sweep point
/// `point_idx` and replicate `rep` derives its process seed from `base` as
/// `base ^ point·SEED_MIX_DRAW ^ rep·SEED_MIX_REP`. This is the one mix the
/// whole CLI shares — `engine::run_job`'s `process_seed_for` and `survey`'s
/// landscape both route through it. Scenario is deliberately NOT an input:
/// paired scenarios at the same (point, rep) share a cell seed, which is
/// what makes their pre-divergence trajectories byte-identical (CRN).
pub fn mix_cell_seed(base: u64, point_idx: u64, rep: u64) -> u64 {
    base ^ point_idx.wrapping_mul(SEED_MIX_DRAW) ^ rep.wrapping_mul(SEED_MIX_REP)
}

// ─── Small helpers shared across subcommands ────────────────────────────────

/// gh#audit-H5. RAII guard that snapshots `EvalStats` on construction
/// and prints the per-counter diff on Drop. Constructed at the top of
/// each `cmd_*` entry that runs simulation or inference; silent when
/// no fallback path was hit, otherwise emits a stderr summary that
/// localises attribution to this command invocation. Drop runs on the
/// normal return path; std::process::exit bypasses Drop, but those
/// paths are already error-printed so missing the summary is fine.
pub struct EvalStatsReportGuard(sim::eval_stats::EvalStats);
impl EvalStatsReportGuard {
    pub fn start() -> Self { Self(sim::eval_stats::EvalStats::snapshot()) }
}
impl Drop for EvalStatsReportGuard {
    fn drop(&mut self) { sim::eval_stats::report_if_nonzero(&self.0); }
}

/// Derive an independent RNG seed for chain `id` from a base seed.
/// XOR with `id * 2^64 * φ` (golden-ratio fractional bits) decorrelates
/// consecutive-chain streams cheaply. Canonical helper for PGAS/PMMH/IF2;
/// previously duplicated at 4+ sites.
pub fn derive_chain_seed(base: u64, id: usize) -> u64 {
    base ^ (id as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

// ─── Compiler discovery ─────────────────────────────────────────────────────

fn camdlc_name() -> &'static str {
    if cfg!(windows) { "camdlc.exe" } else { "camdlc" }
}

/// Versioned camdlc name, e.g. `camdlc-abc1234` (or `.exe` on Windows).
/// Installing camdlc under this name lets `find_camdlc` confirm an exact
/// hash match without any subprocess — pure filesystem stat.
fn camdlc_versioned_name() -> String {
    format!("camdlc-{}{}",
        crate::version::GIT_HASH,
        if cfg!(windows) { ".exe" } else { "" })
}

/// Plain `camdl` binary name, platform-suffixed.
fn camdl_bin_name() -> &'static str {
    if cfg!(windows) { "camdl.exe" } else { "camdl" }
}

/// Detect the "shadowed install" pattern: a `camdl` lives next to the
/// disagreeing camdlc (so this is presumably the binary `make install`
/// just wrote), but the `camdl` actually executing came from somewhere
/// else (typically `~/.cargo/bin/camdl` from a stale `cargo install`,
/// which prepends to PATH ahead of `~/.local/bin/`).
///
/// In that case the version-mismatch error's standard "run make
/// install to sync" advice is a dead end — the user just did that, and
/// it had no effect because their shell still resolves `camdl` to the
/// shadowing copy. The hint names both paths and points at the actual
/// fix.
///
/// Returns `None` (no hint) when:
///  - `current_exe()` is unavailable;
///  - no colocated `camdl` exists alongside `camdlc`;
///  - the colocated `camdl` resolves to the same file as the running
///    one (no shadowing — ordinary version skew).
fn detect_camdl_shadowing(camdlc: &std::path::Path) -> Option<String> {
    let running = std::env::current_exe().ok()?;
    let camdlc_dir = camdlc.parent()?;
    let colocated = camdlc_dir.join(camdl_bin_name());
    if !colocated.exists() { return None; }
    // Canonicalise both sides so symlinked installs (e.g. Homebrew
    // shimming into /opt/homebrew/bin) don't false-positive, and so
    // the paths printed in the hint share a base form.
    let running_canon  = running.canonicalize().ok()?;
    let colocated_canon = colocated.canonicalize().ok()?;
    if running_canon == colocated_canon { return None; }
    let installed_dir = colocated_canon.parent()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| camdlc_dir.display().to_string());
    let shadow_dir = running_canon.parent()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "the shadowing dir".into());
    Some(format!(
        "  Note: another `camdl` is shadowing this install.\n  \
         Running:   {}\n  \
         Installed: {}  (alongside the camdlc above)\n  \
         Fix: `rm {}`, or put {} ahead of {} on your PATH.",
        running_canon.display(),
        colocated_canon.display(),
        running_canon.display(),
        installed_dir,
        shadow_dir,
    ))
}

/// Pure helper: given raw camdlc subprocess output, return `Ok(())` if the
/// reported hash matches `our_hash`, or `Err(message)` otherwise.
/// `location` is a human-readable path string used in the error text.
/// `shadow_hint`, when present, is appended after the standard advice —
/// see `detect_camdl_shadowing` for the diagnosis it carries.
fn eval_version_output(
    stdout: &[u8],
    exit_success: bool,
    our_hash: &str,
    location: &str,
    shadow_hint: Option<&str>,
) -> Result<(), String> {
    let suffix = shadow_hint.map(|h| format!("\n{h}")).unwrap_or_default();
    if exit_success {
        let reported = String::from_utf8_lossy(stdout).trim().to_string();
        if reported == our_hash {
            Ok(())
        } else {
            Err(format!(
                "error: camdlc version mismatch\n  \
                 camdl:  {our_hash}\n  \
                 camdlc: {reported} ({location})\n  \
                 Run `make build-ocaml && make install` to sync.\n  \
                 Set CAMDL_SKIP_VERSION_CHECK=1 to bypass (unsupported).{suffix}"
            ))
        }
    } else {
        Err(format!(
            "error: camdlc ({location}) does not report a version (old build).\n  \
             Run `make build-ocaml && make install` to rebuild.\n  \
             Set CAMDL_SKIP_VERSION_CHECK=1 to bypass (unsupported).{suffix}"
        ))
    }
}

/// Run `camdlc --camdl-version` exactly once per process lifetime.
/// Errors to stderr and exits if the hash differs from this camdl binary's hash.
/// Subsequent calls are instant (OnceLock).
static CAMDLC_CHECKED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Whether the camdlc↔camdl version check should be skipped.
///
/// The check is a *deployment-hygiene* guard: it exists so a production
/// `camdl` binary refuses to run against a stale globally-installed
/// `camdlc` (the mismatch silently changes compiled IR). It has no meaning
/// during the crate's own unit tests, where there is no install to be stale
/// against — and there it is actively harmful: the mismatch path calls
/// `std::process::exit(1)`, which aborts the *entire test binary* (every
/// subsequent test in the process is skipped, masking real results) the
/// moment a stale `camdlc` sits on PATH ahead of the build under test. So
/// `cfg!(test)` short-circuits the check for the bin's unit tests, exactly
/// as `CAMDL_SKIP_VERSION_CHECK=1` does for integration tests and operators.
fn version_check_disabled() -> bool {
    cfg!(test) || std::env::var("CAMDL_SKIP_VERSION_CHECK").is_ok()
}

fn check_camdlc_version_once(camdlc: &std::path::Path) {
    CAMDLC_CHECKED.get_or_init(|| {
        if version_check_disabled() {
            return;
        }
        match std::process::Command::new(camdlc)
            .arg("--camdl-version")
            .output()
        {
            Ok(out) => {
                let hint = detect_camdl_shadowing(camdlc);
                if let Err(msg) = eval_version_output(
                    &out.stdout,
                    out.status.success(),
                    crate::version::GIT_HASH,
                    &camdlc.display().to_string(),
                    hint.as_deref(),
                ) {
                    eprintln!("{msg}");
                    std::process::exit(1);
                }
            }
            Err(_) => {} // spawn failed; nothing useful to report
        }
    });
}

/// Find the camdlc compiler binary via a priority chain:
///
/// 1a. `camdlc-<GIT_HASH>` in same directory as running binary — exact match,
///     zero subprocess overhead (binary name IS the version check).
/// 1b. Plain `camdlc` in same directory — runs `--camdl-version` once
///     (OnceLock) to confirm it matches; warns if stale.
/// 2.  `CAMDLC_PATH` or `CAMDLC` environment variable — same version check.
/// 3.  System PATH — probes with `--camdl-version` (combines existence +
///     version check in one spawn; also serves as the PATH existence test).
fn find_camdlc() -> Result<std::path::PathBuf, String> {
    use std::path::PathBuf;

    // 1. Same directory as running binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // 1a. Versioned name: exact hash match, no subprocess at all
            let versioned = dir.join(camdlc_versioned_name());
            if versioned.exists() { return Ok(versioned); }

            // 1b. Plain camdlc: version-check once via subprocess
            let plain = dir.join(camdlc_name());
            if plain.exists() {
                check_camdlc_version_once(&plain);
                return Ok(plain);
            }
        }
    }

    // 2. Environment variable override
    for var in &["CAMDLC_PATH", "CAMDLC"] {
        if let Ok(path) = std::env::var(var) {
            let p = PathBuf::from(&path);
            if p.exists() {
                check_camdlc_version_once(&p);
                return Ok(p);
            }
        }
    }

    // 3. System PATH: --camdl-version probe doubles as existence check
    if let Ok(out) = std::process::Command::new(camdlc_name())
        .arg("--camdl-version")
        .output()
    {
        let p = PathBuf::from(camdlc_name());
        // Resolve the on-PATH camdlc to a real path so the shadowing
        // detector can compare it to where camdl is running from.
        // `which`-style lookup via PATH walk; canonicalise on hit.
        let resolved = std::env::var_os("PATH")
            .and_then(|paths| std::env::split_paths(&paths)
                .map(|d| d.join(camdlc_name()))
                .find(|c| c.is_file()));
        CAMDLC_CHECKED.get_or_init(|| {
            if version_check_disabled() { return; }
            let hint = resolved.as_deref().and_then(detect_camdl_shadowing);
            if let Err(msg) = eval_version_output(
                &out.stdout,
                out.status.success(),
                crate::version::GIT_HASH,
                "on PATH",
                hint.as_deref(),
            ) {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        });
        return Ok(p);
    }

    Err(format!(
        "camdlc not found.\n\
         Place it next to camdl{} or add it to PATH.\n\
         Set CAMDLC_PATH to override.",
        if cfg!(windows) { ".exe" } else { "" }
    ))
}

#[cfg(test)]
pub(crate) fn camdlc_checked_flag() -> &'static std::sync::OnceLock<()> {
    &CAMDLC_CHECKED
}

/// Run camdlc on a .camdl file and return the IR JSON as a string.
///
/// camdlc can take ~20s on large stratified models and `.output()` blocks
/// the whole time. While it runs we show a progress indicator on **stderr**
/// so the user knows the tool is working, respecting the `--progress` mode:
///   - Pretty (TTY): an indicatif steady-tick spinner `compiling <model> …`,
///     cleared on completion.
///   - Plain (off-TTY): a single `compiling <model> …` log line (no carriage
///     returns), safe under `tee`/CI.
///   - None: nothing.
///
/// The indicator draws to stderr only (via `progress::draw_target()`, which
/// is `hidden()` in plain/none modes); stdout — the IR JSON returned here —
/// is untouched, so the return value is byte-identical to an un-instrumented
/// call. The camdlc error path is unchanged: a non-zero exit still surfaces
/// camdlc's stderr as `Err`.
pub(crate) fn run_camdlc(camdl_path: &str) -> Result<String, String> {
    let camdlc = find_camdlc()?;

    // Friendly model name for the message: just the file's basename.
    let model_name = std::path::Path::new(camdl_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| camdl_path.to_string());

    // Plain mode: emit one line up front on stderr (the bar below is hidden).
    // We use `eprintln!` rather than `log::info!` because the camdl binary
    // installs no `log` backend — `log::*` macros are silently dropped — and
    // this line must actually reach the user off-TTY (the motivating fix).
    // stderr keeps it off stdout (the IR JSON), and a plain line with no
    // carriage returns is safe under `tee`/CI.
    if crate::progress::is_plain() {
        eprintln!("compiling {model_name}...");
    }

    // Pretty mode draws to stderr; plain/none resolve to a hidden target, so
    // the same spinner object is safe to construct unconditionally — nothing
    // renders unless the mode is Pretty.
    let spinner =
        indicatif::ProgressBar::with_draw_target(None, crate::progress::draw_target());
    spinner.set_style(
        indicatif::ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
    );
    spinner.set_message(format!("compiling {model_name}..."));
    spinner.enable_steady_tick(std::time::Duration::from_millis(120));

    // The blocking subprocess call — unchanged from the un-instrumented path.
    let output = std::process::Command::new(&camdlc)
        .arg(camdl_path)
        .output();

    spinner.finish_and_clear();

    let output = output.map_err(|e| format!("cannot run {}: {}", camdlc.display(), e))?;
    if !output.status.success() {
        // camdlc prints errors to stderr — pass them through
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    String::from_utf8(output.stdout)
        .map_err(|e| format!("camdlc output not UTF-8: {}", e))
}

// ─── IR path resolver ────────────────────────────────────────────────────────

/// Resolve a path that appears inside a config file (fit.toml, batch.toml,
/// etc.) against the config's directory — matching the convention used by
/// Cargo, pyproject.toml, package.json, and most other tooling. Absolute
/// input paths pass through unchanged. Closes a class of footguns where
/// running `camdl fit run subdir/fit.toml` and `camdl fit run fit.toml`
/// (after `cd subdir`) gave different behavior despite using the same
/// toml file (GH #22).
///
/// The helper is intentionally generic: the "anchor" arg is the path to
/// the *config file itself*, not its directory, because every caller
/// that has the config path also has its directory one `.parent()` away
/// — passing the file is more honest about what the resolution is
/// relative to and saves one `parent()` call at every site.
///
/// Returns the resolved path as a `String` (matching the storage type
/// used in `FitConfigV2.model.camdl` and `FitConfigV2.data.observations`).
/// Lossy conversion is acceptable here because the inputs are
/// user-supplied strings already; if a non-UTF8 path made it into the
/// toml the parser would have rejected it.
pub fn resolve_relative_to_toml(toml_path: &std::path::Path, path: &str) -> String {
    let p = std::path::Path::new(path);
    if p.is_absolute() {
        return path.to_string();
    }
    let anchor = toml_path.parent().unwrap_or(std::path::Path::new("."));
    anchor.join(p).to_string_lossy().into_owned()
}

/// If path ends with `.camdl`, compile it via camdlc and write to a temp file.
/// Returns (resolved_path, Some(tmpfile)) or (path, None) for plain .ir.json.
pub fn resolve_ir_path(path: &str) -> Result<(String, Option<std::path::PathBuf>), String> {
    if !path.ends_with(".camdl") {
        return Ok((path.to_string(), None));
    }
    let json = run_camdlc(path)?;
    let tmp = std::env::temp_dir()
        .join(format!("camdl_{}.ir.json", std::process::id()));
    std::fs::write(&tmp, &json)
        .map_err(|e| format!("error writing temp IR: {}", e))?;
    Ok((tmp.to_string_lossy().into_owned(), Some(tmp)))
}

/// Load a .camdl or .ir.json model, returning the parsed model and raw IR JSON.
/// The JSON is needed for provenance hashing. Compiles via camdlc if needed.
pub fn load_model(path: &str) -> Result<(ir::Model, String), String> {
    let json = if path.ends_with(".camdl") {
        run_camdlc(path)?
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?
    };
    // gh#audit-C8. Use envelope-aware from_str so version mismatches
    // surface as a typed IrError rather than a serde shape error
    // somewhere deep in the model tree. Hint text in IrError points
    // the user at the right rebuild target.
    let model: ir::Model = ir::from_str(&json)
        .map_err(|e| format!("IR load error from {}: {}", path, e))?;
    // RC1 in 2026-04-19 engine review: run the structural integrity
    // battery on every load. Catches silent-wrong-IR emitted by the
    // compiler (unknown references, missing ODE, duplicate names,
    // real compartments in stoichiometry, etc.) before simulation
    // starts — not after the answer is already wrong.
    ir::validate::validate(&model).map_err(|errs| {
        let mut msg = format!("IR validation failed ({} error(s)):\n", errs.len());
        for e in &errs {
            msg.push_str(&format!("  - {}\n", e));
        }
        msg
    })?;
    Ok((model, json))
}

/// Delegate a subcommand directly to camdlc, passing through all args.
/// Used for compile, check, inspect which are purely compiler operations.
pub fn delegate_to_camdlc(args: &[&str]) -> Result<(), String> {
    let camdlc = find_camdlc()?;
    let status = std::process::Command::new(&camdlc)
        .args(args)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| format!("cannot run camdlc: {}", e))?;
    std::process::exit(status.code().unwrap_or(1));
}

/// Resolve flow indices for a named transition (or all transmission transitions).
/// Used by pfilter, if2, profile for --flow NAME.
pub fn resolve_flow_indices(model: &ir::Model, flow_name: Option<&str>) -> Result<Vec<usize>, String> {
    if let Some(name) = flow_name {
        let indices: Vec<usize> = model.transitions.iter().enumerate()
            .filter(|(_, tr)| tr.name == name || tr.name.starts_with(&format!("{}_", name)))
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            return Err(format!("no transition named '{}'. Available: {}",
                name, model.transitions.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", ")));
        }
        Ok(indices)
    } else {
        let indices: Vec<usize> = model.transitions.iter().enumerate()
            .filter(|(_, tr)| tr.metadata.as_ref()
                .and_then(|m| m.origin_kind.as_deref()) == Some("transmission"))
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            return Err("no transmission transitions found. Use --flow NAME to specify.".into());
        }
        Ok(indices)
    }
}

// ─── Multi-stream binding diagnostics (gh#90) ───────────────────────────────

/// Build the gh#90 unbound-streams warning string when the user is
/// running profile/pfilter against a model that declares more than one
/// observation block but only one stream is bound to data.
///
/// Symptom this catches: a SEIR with `observations { cases : ...,
/// deaths : ... }`, run as `camdl profile model.camdl --data deaths.tsv
/// --obs deaths ...`, silently drops `cases` from the likelihood —
/// `cases` falls back to its priors. The result looks plausible but is
/// methodologically wrong: it's profile-on-deaths with cases-side
/// parameters floating, not profile-on-deaths-given-cases.
///
/// Arguments:
///   `cmd`: subcommand name (`profile` or `pfilter`) — used in the
///          warning text for actionable phrasing.
///   `all_obs_names`: every observation block declared in the model,
///                    in declaration order.
///   `bound_names`: the names of streams actually bound to data by the
///                  current invocation (resolved from `--obs` / `--fit`).
///
/// Returns `None` when no warning is needed: the model declares ≤ 1
/// observation block, or every block is bound. Returns the formatted
/// warning otherwise.
///
/// Family-root semantics: `bound_names` here is the *resolved leaf set*
/// (e.g. `cases_a02, cases_a25, ...` after family expansion). The
/// caller is responsible for handing in the post-resolution names so
/// that `--obs cases` covering a 5-cell family is correctly seen as
/// "binding 5 streams", not "binding 1 stream".
/// gh#90: resolve `--data` flags against the model's observation
/// blocks into a canonical per-stream binding map.
///
/// Returns `Vec<(stream_name, path)>` in the order the resolver
/// determined (deterministic for a given input). Errors are
/// user-facing: pick the right form, point at the model's stream
/// names if a NAME mismatched, give an actionable suggestion when
/// the user is implicitly trying to drop streams.
///
/// Validation rules:
///   1. Empty `--data` list → caller decides (fit-toml fallback).
///   2. All-Single or all-Named only — mixed forms error.
///   3. Single-form: max one flag. Two single PATHs is ambiguous.
///      With N=1 obs block: bind to that block. With N>1: requires
///      `--obs NAME` to disambiguate (single-stream scoring, the
///      caller emits the unbound-streams warning).
///   4. Named-form: every NAME must match a model observation block
///      (exact name OR family root). NAME collisions across flags
///      are errors. Multi-stream binding is the joint-scoring path.
pub fn resolve_data_specs(
    data_specs: &[crate::args::types::DataSpec],
    model_obs_names: &[String],
    obs_arg: Option<&str>,
) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    use crate::args::types::DataSpec;

    if data_specs.is_empty() {
        return Err(
            "--data is required (no --data flags supplied and no \
             fit-toml fallback found). Use `--data PATH` for a \
             single-stream model, or `--data NAME=PATH` (repeatable) \
             for a multi-stream model."
                .to_string(),
        );
    }

    // Partition into single-form vs named-form. Mixed → error.
    let n_single = data_specs.iter()
        .filter(|d| matches!(d, DataSpec::Single(_))).count();
    let n_named = data_specs.iter()
        .filter(|d| matches!(d, DataSpec::Named { .. })).count();
    if n_single > 0 && n_named > 0 {
        return Err(
            "--data PATH and --data NAME=PATH are mutually exclusive; \
             pick one form.\n  \
             Use --data PATH (single flag) for single-stream models, \
             or --data NAME=PATH (repeatable, one per stream) for \
             multi-stream models.".to_string(),
        );
    }

    // All-Single form.
    if n_named == 0 {
        if n_single > 1 {
            return Err(format!(
                "--data PATH given {} times; use one --data flag (the \
                 single-stream form takes a single path). For multiple \
                 streams use --data NAME=PATH (repeatable).",
                n_single,
            ));
        }
        let path = match &data_specs[0] {
            DataSpec::Single(p) => p.clone(),
            _ => unreachable!(),
        };
        return resolve_single_form(path, model_obs_names, obs_arg);
    }

    // All-Named form.
    if obs_arg.is_some() {
        return Err(
            "--obs NAME is redundant with --data NAME=PATH (each pair \
             names its own stream); pass --obs only with the single-\
             stream --data PATH form.".to_string(),
        );
    }
    let mut out: Vec<(String, std::path::PathBuf)> = Vec::with_capacity(n_named);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for spec in data_specs {
        let (name, path) = match spec {
            DataSpec::Named { name, path } => (name.clone(), path.clone()),
            _ => unreachable!(),
        };
        if !seen.insert(name.clone()) {
            return Err(format!(
                "--data {}=... supplied more than once; each stream \
                 name may appear at most once.", name));
        }
        // Validate NAME against the model. Two acceptable shapes:
        //   1. Exact match on a leaf obs block (`name == obs.name`).
        //   2. Family-root match — every IR obs whose name starts
        //      with `<name>_` is bound to this path. Mirrors profile's
        //      existing family resolution so a single `--data
        //      cases=cases.tsv` on a stratified `cases_a02, cases_a25`
        //      family covers both leaves with one wide TSV.
        let exact_match = model_obs_names.iter().any(|n| n == &name);
        let family_prefix = format!("{}_", name);
        let family_matches: Vec<&String> = model_obs_names.iter()
            .filter(|n| n.starts_with(&family_prefix))
            .collect();
        if exact_match {
            out.push((name, path));
        } else if !family_matches.is_empty() {
            for leaf in family_matches {
                out.push((leaf.clone(), path.clone()));
            }
        } else {
            let avail = if model_obs_names.is_empty() {
                "<model has no observation blocks>".to_string()
            } else {
                model_obs_names.iter()
                    .map(|n| format!("'{}'", n))
                    .collect::<Vec<_>>().join(", ")
            };
            return Err(format!(
                "--data {}=...: '{}' does not match any observation block \
                 in the model (neither as a leaf name nor a family root). \
                 Available: {}.", name, name, avail));
        }
    }
    Ok(out)
}

/// Helper for the single-PATH branch of `resolve_data_specs`.
fn resolve_single_form(
    path: std::path::PathBuf,
    model_obs_names: &[String],
    obs_arg: Option<&str>,
) -> Result<Vec<(String, std::path::PathBuf)>, String> {
    if model_obs_names.is_empty() {
        return Err(
            "model declares no observation blocks; cannot bind data."
                .to_string());
    }
    if model_obs_names.len() == 1 {
        // Single-block model.
        if let Some(name) = obs_arg {
            // --obs NAME redundant but allowed: must match.
            if name != model_obs_names[0] {
                // Also accept a family-root if the single block looks
                // like an expanded leaf (`<name>_<idx>...`). Conservative
                // here — only accept exact match. Family-root semantics
                // on a single-block model are odd.
                return Err(format!(
                    "--obs '{}' does not match the model's single \
                     observation block '{}'.",
                    name, model_obs_names[0]));
            }
        }
        Ok(vec![(model_obs_names[0].clone(), path)])
    } else {
        // Multi-block model.
        match obs_arg {
            None => {
                let avail = model_obs_names.iter()
                    .map(|n| format!("'{}'", n))
                    .collect::<Vec<_>>().join(", ");
                Err(format!(
                    "model has multiple observation blocks ({}); use \
                     --data NAME=PATH (repeatable) to bind every stream, \
                     or --data PATH --obs NAME to score only one. \
                     Currently --data PATH (no NAME) and no --obs is \
                     ambiguous on a multi-block model — refusing to \
                     silently score a single stream.",
                    avail))
            }
            Some(name) => {
                // Resolve as exact or family-root.
                let exact_match = model_obs_names.iter().any(|n| n == name);
                let family_prefix = format!("{}_", name);
                let family_matches: Vec<&String> = model_obs_names.iter()
                    .filter(|n| n.starts_with(&family_prefix))
                    .collect();
                if exact_match {
                    Ok(vec![(name.to_string(), path)])
                } else if !family_matches.is_empty() {
                    Ok(family_matches.into_iter()
                        .map(|leaf| (leaf.clone(), path.clone()))
                        .collect())
                } else {
                    let avail = model_obs_names.iter()
                        .map(|n| format!("'{}'", n))
                        .collect::<Vec<_>>().join(", ");
                    Err(format!(
                        "--obs '{}' does not match any observation block. \
                         Available: {}.", name, avail))
                }
            }
        }
    }
}

pub fn format_unbound_streams_warning(
    cmd: &str,
    all_obs_names: &[String],
    bound_names: &[String],
) -> Option<String> {
    if all_obs_names.len() <= 1 {
        return None;
    }
    let bound: std::collections::HashSet<&str> = bound_names.iter()
        .map(|s| s.as_str()).collect();
    let unbound: Vec<&str> = all_obs_names.iter()
        .map(|s| s.as_str())
        .filter(|n| !bound.contains(*n))
        .collect();
    if unbound.is_empty() {
        return None;
    }
    let bound_list: Vec<&str> = all_obs_names.iter()
        .map(|s| s.as_str())
        .filter(|n| bound.contains(*n))
        .collect();
    let bound_phrase = if bound_list.len() == 1 {
        format!("'{}' is", bound_list[0])
    } else {
        format!("{} are",
            bound_list.iter().map(|n| format!("'{}'", n))
                .collect::<Vec<_>>().join(", "))
    };
    let unbound_phrase = if unbound.len() == 1 {
        format!("'{}'", unbound[0])
    } else {
        unbound.iter().map(|n| format!("'{}'", n))
            .collect::<Vec<_>>().join(" and ")
    };
    let all_names = all_obs_names.iter().map(|s| s.as_str())
        .collect::<Vec<_>>().join(", ");
    Some(format!(
        "{}: warning: model has {} observation blocks ({}); only {} \
         bound to data — likelihood from {} is silently zero. To \
         score jointly, use --data NAME=PATH for each stream (or \
         --fit FOO.toml with a [data.observations] section).\n",
        cmd, all_obs_names.len(), all_names, bound_phrase, unbound_phrase,
    ))
}

#[cfg(test)]
mod gh90_resolver_tests {
    use super::*;
    use crate::args::types::DataSpec;
    use std::path::PathBuf;

    fn n(s: &str) -> String { s.to_string() }

    #[test]
    fn resolve_data_specs_single_block_single_data_works() {
        let specs = vec![DataSpec::Single(PathBuf::from("cases.tsv"))];
        let bound = resolve_data_specs(&specs, &[n("cases")], None).unwrap();
        assert_eq!(bound, vec![(n("cases"), PathBuf::from("cases.tsv"))]);
    }

    #[test]
    fn resolve_data_specs_single_block_single_data_with_redundant_obs_works() {
        let specs = vec![DataSpec::Single(PathBuf::from("cases.tsv"))];
        let bound = resolve_data_specs(&specs, &[n("cases")], Some("cases")).unwrap();
        assert_eq!(bound, vec![(n("cases"), PathBuf::from("cases.tsv"))]);
    }

    #[test]
    fn resolve_data_specs_single_block_single_data_with_mismatched_obs_errors() {
        let specs = vec![DataSpec::Single(PathBuf::from("cases.tsv"))];
        let err = resolve_data_specs(&specs, &[n("cases")], Some("deaths")).unwrap_err();
        assert!(err.contains("'deaths'"));
        assert!(err.contains("'cases'"));
    }

    #[test]
    fn resolve_data_specs_multi_block_no_obs_errors_actionable() {
        let specs = vec![DataSpec::Single(PathBuf::from("data.tsv"))];
        let model = vec![n("cases"), n("deaths"), n("hosps")];
        let err = resolve_data_specs(&specs, &model, None).unwrap_err();
        // Actionable: name every block, suggest both --data NAME=PATH
        // and --data PATH --obs NAME forms.
        assert!(err.contains("'cases'"), "{}", err);
        assert!(err.contains("'deaths'"), "{}", err);
        assert!(err.contains("--data NAME=PATH"), "{}", err);
        assert!(err.contains("--obs"), "{}", err);
    }

    #[test]
    fn resolve_data_specs_multi_block_single_data_with_obs_works() {
        let specs = vec![DataSpec::Single(PathBuf::from("data.tsv"))];
        let model = vec![n("cases"), n("deaths")];
        let bound = resolve_data_specs(&specs, &model, Some("cases")).unwrap();
        assert_eq!(bound, vec![(n("cases"), PathBuf::from("data.tsv"))]);
    }

    #[test]
    fn resolve_data_specs_multi_block_named_pairs_works() {
        let specs = vec![
            DataSpec::Named { name: n("cases"), path: PathBuf::from("c.tsv") },
            DataSpec::Named { name: n("deaths"), path: PathBuf::from("d.tsv") },
        ];
        let model = vec![n("cases"), n("deaths")];
        let bound = resolve_data_specs(&specs, &model, None).unwrap();
        assert_eq!(bound, vec![
            (n("cases"), PathBuf::from("c.tsv")),
            (n("deaths"), PathBuf::from("d.tsv")),
        ]);
    }

    #[test]
    fn resolve_data_specs_partial_named_pairs_works() {
        // Multi-block model, only some streams bound — succeeds at the
        // resolver level; the caller (profile/pfilter) emits the
        // unbound-streams warning over the returned set.
        let specs = vec![
            DataSpec::Named { name: n("cases"), path: PathBuf::from("c.tsv") },
        ];
        let model = vec![n("cases"), n("deaths"), n("hosps")];
        let bound = resolve_data_specs(&specs, &model, None).unwrap();
        assert_eq!(bound, vec![(n("cases"), PathBuf::from("c.tsv"))]);
    }

    #[test]
    fn resolve_data_specs_mixed_single_and_named_errors() {
        let specs = vec![
            DataSpec::Single(PathBuf::from("c.tsv")),
            DataSpec::Named { name: n("deaths"), path: PathBuf::from("d.tsv") },
        ];
        let err = resolve_data_specs(&specs, &[n("cases"), n("deaths")], None).unwrap_err();
        assert!(err.contains("mutually exclusive"), "{}", err);
    }

    #[test]
    fn resolve_data_specs_named_unknown_name_errors() {
        let specs = vec![
            DataSpec::Named { name: n("typos"), path: PathBuf::from("t.tsv") },
        ];
        let model = vec![n("cases"), n("deaths")];
        let err = resolve_data_specs(&specs, &model, None).unwrap_err();
        assert!(err.contains("'typos'"), "{}", err);
        assert!(err.contains("does not match"), "{}", err);
        // Lists what *is* available.
        assert!(err.contains("'cases'"), "{}", err);
        assert!(err.contains("'deaths'"), "{}", err);
    }

    #[test]
    fn resolve_data_specs_duplicate_named_errors() {
        let specs = vec![
            DataSpec::Named { name: n("cases"), path: PathBuf::from("a.tsv") },
            DataSpec::Named { name: n("cases"), path: PathBuf::from("b.tsv") },
        ];
        let err = resolve_data_specs(&specs, &[n("cases")], None).unwrap_err();
        assert!(err.contains("more than once"), "{}", err);
    }

    #[test]
    fn resolve_data_specs_two_singles_errors() {
        let specs = vec![
            DataSpec::Single(PathBuf::from("a.tsv")),
            DataSpec::Single(PathBuf::from("b.tsv")),
        ];
        let err = resolve_data_specs(&specs, &[n("cases")], None).unwrap_err();
        assert!(err.contains("given 2 times") || err.contains("NAME=PATH"),
            "{}", err);
    }

    #[test]
    fn resolve_data_specs_empty_errors() {
        let err = resolve_data_specs(&[], &[n("cases")], None).unwrap_err();
        assert!(err.contains("--data is required"), "{}", err);
    }

    #[test]
    fn resolve_data_specs_named_family_root_expands() {
        // Family-root in NAME=PATH form expands to every leaf.
        let specs = vec![
            DataSpec::Named { name: n("cases"), path: PathBuf::from("wide.tsv") },
        ];
        let model = vec![n("cases_a02"), n("cases_a25"), n("deaths_a02")];
        let bound = resolve_data_specs(&specs, &model, None).unwrap();
        assert_eq!(bound, vec![
            (n("cases_a02"), PathBuf::from("wide.tsv")),
            (n("cases_a25"), PathBuf::from("wide.tsv")),
        ]);
    }

    #[test]
    fn resolve_data_specs_named_with_obs_errors() {
        // Defensive: --obs is for single-stream selection; named
        // pairs name themselves. Combining is a user-confusion smell.
        let specs = vec![
            DataSpec::Named { name: n("cases"), path: PathBuf::from("c.tsv") },
        ];
        let err = resolve_data_specs(&specs, &[n("cases")], Some("cases"))
            .unwrap_err();
        assert!(err.contains("redundant") || err.contains("only with"),
            "{}", err);
    }
}

#[cfg(test)]
mod gh90_warning_tests {
    use super::*;

    #[test]
    fn no_warning_when_model_has_one_obs_block() {
        // Single-stream model: no methodology trap, no warning.
        let all = vec!["cases".to_string()];
        let bound = vec!["cases".to_string()];
        assert!(format_unbound_streams_warning("profile", &all, &bound).is_none());
    }

    #[test]
    fn warning_when_multi_block_but_one_bound() {
        // gh#90 primary trap: cases + deaths declared, only deaths
        // bound. Previously silent — now must surface.
        let all = vec!["cases".to_string(), "deaths".to_string()];
        let bound = vec!["deaths".to_string()];
        let w = format_unbound_streams_warning("profile", &all, &bound)
            .expect("warning should fire");
        assert!(w.contains("warning"));
        assert!(w.contains("cases"), "warning should name unbound stream: {}", w);
        assert!(w.contains("deaths"), "warning should name bound stream: {}", w);
        assert!(w.contains("silently zero"),
            "warning should describe the silent failure mode: {}", w);
        // gh#90: primary surface is `--data NAME=PATH`; --fit remains
        // mentioned as a fallback.
        assert!(w.contains("--data NAME=PATH"),
            "warning should suggest --data NAME=PATH: {}", w);
        assert!(w.contains("--fit"),
            "warning should suggest --fit fallback: {}", w);
        assert!(w.starts_with("profile:"),
            "warning should be tagged with subcommand: {}", w);
    }

    #[test]
    fn no_warning_when_all_blocks_bound_via_family_root() {
        // `--obs cases` covering an indexed family expands to multiple
        // resolved names — the warning must see "all bound", not
        // "one of N bound". The caller hands in resolved leaves.
        let all = vec![
            "cases_a02".to_string(),
            "cases_a25".to_string(),
            "cases_a65".to_string(),
        ];
        let bound = all.clone();
        assert!(format_unbound_streams_warning("profile", &all, &bound).is_none());
    }

    #[test]
    fn warning_when_partial_family_bound_partial_unbound() {
        // Realistic: 2 cases streams + 2 deaths streams, user bound
        // only the cases family (`--obs cases`). Deaths streams must
        // surface in the warning.
        let all = vec![
            "cases_a02".to_string(), "cases_a25".to_string(),
            "deaths_a02".to_string(), "deaths_a25".to_string(),
        ];
        let bound = vec!["cases_a02".to_string(), "cases_a25".to_string()];
        let w = format_unbound_streams_warning("profile", &all, &bound)
            .expect("warning should fire");
        assert!(w.contains("deaths_a02"), "warning must list each unbound leaf: {}", w);
        assert!(w.contains("deaths_a25"), "warning must list each unbound leaf: {}", w);
    }

    #[test]
    fn warning_tagged_with_pfilter_too() {
        // Both subcommands share the helper — the prefix lets callers
        // tell which warned (matters when both fire in a script).
        let all = vec!["cases".to_string(), "deaths".to_string()];
        let bound = vec!["deaths".to_string()];
        let w = format_unbound_streams_warning("pfilter", &all, &bound)
            .expect("warning should fire for pfilter too");
        assert!(w.starts_with("pfilter:"), "got: {}", w);
    }
}

// ─── Loader helpers ──────────────────────────────────────────────────────────

/// Load a flat Vec<Expr::Const> from a CSV, TSV, or JSON file.
// --table loads flat row-major float arrays (not long format).
// Long format (read_long) is resolved at compile time by the OCaml frontend.
// External tables supplied at runtime must match the flat order used at compile time.
pub fn load_table_file(path: &str) -> Result<Vec<ir::expr::Expr>, String> {
    use ir::expr::Expr;
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let ext = std::path::Path::new(path)
        .extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    if ext == "json" {
        let v: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("JSON parse error in {}: {}", path, e))?;
        let mut out = Vec::new();
        match &v {
            serde_json::Value::Array(rows) => {
                for row in rows {
                    match row {
                        serde_json::Value::Array(cols) => {
                            for cell in cols {
                                let f = cell.as_f64().ok_or_else(||
                                    format!("expected number in {}", path))?;
                                out.push(Expr::const_(f));
                            }
                        }
                        _ => {
                            let f = row.as_f64().ok_or_else(||
                                format!("expected number in {}", path))?;
                            out.push(Expr::const_(f));
                        }
                    }
                }
            }
            _ => return Err(format!("expected JSON array in {}", path)),
        }
        Ok(out)
    } else {
        // CSV or TSV
        let sep = if ext == "tsv" { '\t' } else { ',' };
        let mut out = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            for cell in line.split(sep) {
                let cell = cell.trim();
                let f: f64 = cell.parse()
                    .map_err(|_| format!("expected number, got '{}' in {}", cell, path))?;
                out.push(Expr::const_(f));
            }
        }
        Ok(out)
    }
}

/// Load parameter overrides from a TOML file.
pub fn load_params_toml(path: &str) -> Result<HashMap<String, f64>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let table: toml::Table = content.parse()
        .map_err(|e| format!("TOML parse error in {}: {}", path, e))?;
    let mut out = HashMap::new();
    for (key, val) in &table {
        // The `[provenance]` table in mle_params.toml files carries
        // fit metadata (backend, dt, fit_hash, etc.) — not model
        // parameters. Skip it here so provenance fields don't get
        // splatted into the parameter namespace. See
        // docs/dev/proposals/2026-04-19-backend-provenance-guardrail.md.
        if key == "provenance" { continue; }
        match val {
            toml::Value::Float(f)   => { out.insert(key.clone(), *f); }
            toml::Value::Integer(i) => { out.insert(key.clone(), *i as f64); }
            toml::Value::Table(section) => {
                for (subkey, subval) in section {
                    let full = format!("{}_{}", key, subkey);
                    match subval {
                        toml::Value::Float(f)   => { out.insert(full, *f); }
                        toml::Value::Integer(i) => { out.insert(full, *i as f64); }
                        _ => return Err(format!(
                            "{}:[{}].{}: expected a number, got {:?}", path, key, subkey, subval
                        )),
                    }
                }
            }
            _ => return Err(format!(
                "{}:{}: expected a number or table section, got {:?}", path, key, val
            )),
        }
    }
    Ok(out)
}

/// Load a TOML params file and apply values to the model's parameters.
///
/// **Used only by `main::prepare_cas_ctx`** for partial parameter
/// resolution — the cas-ctx hashing path deliberately holds back the
/// scenario half so that scenario and base params hash separately
/// (per the documented cas cache key). Every other subcommand routes
/// through `params_resolver::resolve_parameters` instead. See
/// `docs/dev/notes/2026-05-25-cli-ux-impl-questions.md`
/// §"prepare_cas_ctx partial resolution" for the rationale.
///
/// Validates the resulting `model.parameters` after applying — if the
/// supplied file leaves any *resolved* parameter with a non-finite
/// value or out-of-bounds value, returns an error. Params still at
/// `value = None` (i.e. waiting on the scenario half) are skipped by
/// `validate_parameter_values`.
pub fn apply_params_file(model: &mut ir::Model, path: &str) -> Result<(), String> {
    let vals = load_params_toml(path)?;
    for p in &mut model.parameters {
        if let Some(&v) = vals.get(&p.name) {
            p.value = Some(v);
        }
    }
    validate_parameter_values(model)?;
    Ok(())
}

/// Validate that every parameter with a declared `bounds` carries a value
/// within those bounds, and that every parameter value is finite (no NaN,
/// no ±∞).
///
/// Call after all parameter-override resolution is complete and before
/// the model reaches the simulation/inference layer. Bounds are declared
/// inclusively, so values exactly equal to `lo` or `hi` pass.
///
/// Returns `Err(joined_messages)` when any parameter violates. All
/// violations are collected and reported together so the user sees the
/// full list rather than fixing them one at a time. Parameters with
/// `value = None` are left to the resolution layer (some subcommands
/// fill them later from prior draws or scenarios) — only set values are
/// checked here.
///
/// Lives in the CLI layer, not `crates/ir/src/validate.rs`, because
/// bounds enforcement is a CLI-input-validation concern: the IR
/// validator is for structural integrity (unknown references, missing
/// ODE, etc.) of the IR-as-emitted-by-the-compiler. A user can
/// legitimately hand-author an IR with `value: 5.0` and `bounds: [0,
/// 2]`, and the IR is structurally valid; what's wrong is that the
/// CLI-supplied or scenario-supplied value is outside the bounds the
/// model author declared.
pub fn validate_parameter_values(model: &ir::Model) -> Result<(), String> {
    let mut errs: Vec<String> = Vec::new();
    for p in &model.parameters {
        let Some(v) = p.value else { continue; };
        if !v.is_finite() {
            errs.push(format!(
                "parameter '{}' = {} is not finite (NaN or ±∞).\n  \
                 Fix: supply a finite numeric value via --param, --params, or the scenario block.",
                p.name, v));
            continue;
        }
        if let Some((lo, hi)) = p.bounds {
            if v < lo || v > hi {
                errs.push(format!(
                    "parameter '{}' = {} is outside declared bounds [{}, {}].\n  \
                     Fix: either widen the bounds in the model, or supply a value within the declared range.",
                    p.name, v, lo, hi));
            }
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs.join("\n"))
    }
}

/// Load a keyed TSV file (two columns: name<TAB>value) for --param-vec.
pub fn load_keyed_tsv(path: &str) -> Result<Vec<(String, f64)>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {}", path, e))?;
    let mut out = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let mut parts = line.splitn(2, '\t');
        let key = parts.next()
            .ok_or_else(|| format!("{}:{}: expected key<TAB>value", path, lineno + 1))?
            .trim().to_string();
        let val_str = parts.next()
            .ok_or_else(|| format!("{}:{}: missing value column", path, lineno + 1))?
            .trim();
        let val: f64 = val_str.parse()
            .map_err(|_| format!("{}:{}: expected number, got '{}'", path, lineno + 1, val_str))?;
        out.push((key, val));
    }
    Ok(out)
}

// ─── Enable/disable resolution ───────────────────────────────────────────────

/// Resolve a list of enable/disable names, expanding family names via base_name.
///
/// Resolution rule:
/// 1. Exact match: name == iv.name → enable that one
/// 2. Family match: name == iv.base_name → enable all members of that family
/// 3. No match: error with available names and families
pub fn resolve_enable_list(
    names: &[String],
    interventions: &[Intervention],
) -> Result<Vec<String>, String> {
    let mut resolved: Vec<String> = Vec::new();
    for name in names {
        // 1. Exact match
        if interventions.iter().any(|iv| iv.name == *name) {
            resolved.push(name.clone());
            continue;
        }
        // 2. Family match
        let family: Vec<String> = interventions.iter()
            .filter(|iv| iv.base_name.as_deref() == Some(name.as_str()))
            .map(|iv| iv.name.clone())
            .collect();
        if !family.is_empty() {
            resolved.extend(family);
            continue;
        }
        // 3. No match
        let mut families: Vec<&str> = interventions.iter()
            .filter_map(|iv| iv.base_name.as_deref())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        families.sort();
        return Err(format!(
            "'{}' does not match any intervention or family.\n  \
             Families: {}\n  Names (first 10): {}",
            name,
            if families.is_empty() { "(none)".to_string() }
            else { families.join(", ") },
            interventions.iter().take(10)
                .map(|iv| iv.name.as_str()).collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(resolved)
}

/// Apply an enable/disable scenario filter to `model.interventions`,
/// respecting the `always_active` distinction (events vs toggleable
/// interventions).
///
/// Semantics (matches the spec in `camdl-language-spec.md` §14 / §14.4):
///
/// - **Events** (`always_active = true`) are kept unless *explicitly*
///   named in the `disable` list. This is the only way an event can be
///   silenced — the default-off behaviour that applies to toggleable
///   interventions never applies to events.
/// - **Toggleable interventions** (`always_active = false`) are kept
///   only if named in the `enable` list (or its scenario expansion).
///   The default is "off," matching the spec.
/// - The `enable`/`disable` lists may contain:
///   - exact intervention names (`sia_round_1_north`),
///   - family base names (`sia_round_1` → all its expanded members),
///   - the wildcard `"*"` (matches every toggleable intervention for
///     `enable`; every action, events included, for `disable`).
///
/// Shared between `simulate`, `pfilter`, and `fit` so the three entry
/// points cannot drift apart on this contract.
pub fn apply_scenario_filter(
    model: &mut ir::Model,
    enable: &[String],
    disable: &[String],
) -> Result<(), String> {
    // Separate out wildcards so family/exact resolution below doesn't
    // try to match "*" against a real intervention name.
    let enable_wild  = enable.iter().any(|s| s == "*");
    let disable_wild = disable.iter().any(|s| s == "*");
    let enable_non_wild:  Vec<String> = enable.iter().filter(|s| *s != "*").cloned().collect();
    let disable_non_wild: Vec<String> = disable.iter().filter(|s| *s != "*").cloned().collect();

    // Resolve family names → concrete intervention names.
    let active_enable  = resolve_enable_list(&enable_non_wild,  &model.interventions)?;
    let active_disable = resolve_enable_list(&disable_non_wild, &model.interventions)?;

    model.interventions.retain(|iv| {
        // Explicit disable wins — even for always_active events.
        if disable_wild || active_disable.contains(&iv.name) {
            return false;
        }
        // Events stay on unless explicitly disabled above.
        if iv.always_active {
            return true;
        }
        // Toggleable interventions: enable list or wildcard required.
        enable_wild || active_enable.contains(&iv.name)
    });

    Ok(())
}

/// Print a compact summary of the active scheduled actions — the events
/// that will always fire and the interventions that survived filtering.
/// Matches the style of the priors-reporting block. Silent when neither
/// block has entries.
///
/// The intent: make the default behaviour of `fit` and `pfilter` visible
/// on startup, so a user who forgot `scenario = "..."` sees "0 active
/// of 5 declared" immediately rather than discovering it from posteriors
/// hours later.
pub fn print_scheduled_actions_summary(
    model_before_filter: &ir::Model,
    model_after_filter: &ir::Model,
) {
    // Split declared actions into events vs toggleable interventions.
    let (decl_events, decl_interv): (Vec<_>, Vec<_>) = model_before_filter
        .interventions.iter().partition(|iv| iv.always_active);
    let active_names: std::collections::HashSet<&str> = model_after_filter
        .interventions.iter().map(|iv| iv.name.as_str()).collect();

    if !decl_interv.is_empty() {
        let active_count = decl_interv.iter().filter(|iv| active_names.contains(iv.name.as_str())).count();
        eprintln!("  interventions ({} active of {} declared):", active_count, decl_interv.len());
        for iv in &decl_interv {
            let on = active_names.contains(iv.name.as_str());
            let glyph = if on { "\x1b[32m✓\x1b[0m" } else { "\x1b[2m✗\x1b[0m" };
            let note = if on { "" } else { "  (off — not enabled)" };
            eprintln!("    {} {}{}", glyph, iv.name, note);
        }
    }
    if !decl_events.is_empty() {
        let active_count = decl_events.iter().filter(|iv| active_names.contains(iv.name.as_str())).count();
        eprintln!("  events ({} declared, {} active):", decl_events.len(), active_count);
        for iv in &decl_events {
            let on = active_names.contains(iv.name.as_str());
            let glyph = if on { "\x1b[32m✓\x1b[0m" } else { "\x1b[2m✗\x1b[0m" };
            let note = if on { "" } else { "  (disabled)" };
            eprintln!("    {} {}{}", glyph, iv.name, note);
        }
    }
}

/// Print a summary of the active observation streams — one row per
/// stream with its projection kind (incidence vs. prevalence /
/// snapshot) and likelihood family. Emits a soft advisory when a
/// NegativeBinomial is paired with a snapshot projection (valid but
/// unusual; see `camdl-run-spec.md` §14.4).
///
/// Silent when the model has no observations. Called by `fit run` and
/// `pfilter` right after the interventions/events summary.
pub fn print_observations_summary(model: &ir::Model) {
    if model.observations.is_empty() { return; }
    eprintln!("  observations ({} stream{}):",
        model.observations.len(),
        if model.observations.len() == 1 { "" } else { "s" });
    let mut warn_negbin_on_snapshot = false;
    for obs in &model.observations {
        let (kind_label, is_snapshot) = match &obs.projection {
            ir::observation::Projection::CumulativeFlow(name) =>
                (format!("incidence({})", name), false),
            ir::observation::Projection::CurrentPop(name) =>
                (format!("prevalence({})", name), true),
            ir::observation::Projection::CurrentPopSum(names) =>
                (format!("prevalence({})", names.join(" + ")), true),
            ir::observation::Projection::DerivedExpr(_) =>
                ("derived expression".to_string(), true),
        };
        let lik_label = match &obs.likelihood {
            ir::observation::Likelihood::NegBinomial(_)  => "NegBinomial",
            ir::observation::Likelihood::Poisson(_)      => "Poisson",
            ir::observation::Likelihood::Normal(_)       => "Normal",
            ir::observation::Likelihood::Binomial(_)     => "Binomial",
            ir::observation::Likelihood::BetaBinomial(_) => "BetaBinomial",
            ir::observation::Likelihood::Bernoulli(_)    => "Bernoulli",
        };
        eprintln!("    \x1b[32m✓\x1b[0m {:<16} {:<28} {}", obs.name, kind_label, lik_label);
        if is_snapshot && matches!(obs.likelihood, ir::observation::Likelihood::NegBinomial(_)) {
            warn_negbin_on_snapshot = true;
        }
    }
    if warn_negbin_on_snapshot {
        eprintln!("    \x1b[2mnote: NegBinomial on a prevalence / snapshot projection is valid");
        eprintln!("          but uncommon. Binomial or Poisson is the typical choice for");
        eprintln!("          point-in-time counts. See camdl-run-spec.md §14.4.\x1b[0m");
    }
}

// ─── SimRun / SimOutput ───────────────────────────────────────────────────────

/// All inputs needed to run one simulation.
#[derive(Clone)]
pub struct SimRun {
    pub ir_path: String,
    pub params_files: Vec<String>,
    pub overrides: HashMap<String, f64>,
    pub set_vec_entries: Vec<(String, String)>,
    pub table_files: HashMap<String, String>,
    pub scenario_name: Option<String>,
    pub adhoc_enable: Vec<String>,
    pub adhoc_disable: Vec<String>,
    pub backend: crate::args::types::Backend,
    pub dt: f64,
    pub seed: u64,
}

impl Default for SimRun {
    fn default() -> Self {
        SimRun {
            ir_path: String::new(),
            params_files: Vec::new(),
            overrides: HashMap::new(),
            set_vec_entries: Vec::new(),
            table_files: HashMap::new(),
            scenario_name: None,
            adhoc_enable: Vec::new(),
            adhoc_disable: Vec::new(),
            backend: crate::args::types::Backend::ChainBinomial,
            dt: 1.0,
            seed: 1,
        }
    }
}

/// Resolve a `SimRun` to a compiled model + parameter vector, applying the
/// full scenario / --params / --param-vec / --param / --table precedence
/// pipeline (docs/camdl-run-spec.md §1.3). Shared by the count-trajectory
/// path ([`run_simulation`]) and the lineage path
/// ([`run_simulation_lineage`]) so both see byte-identical parameter
/// resolution.
pub fn resolve_run_model(run: &SimRun) -> Result<(CompiledModel, ir::Model), String> {
    // Load IR source (handles .camdl compilation via camdlc)
    let (ir_path_resolved, _tmpfile) = resolve_ir_path(&run.ir_path)?;

    let src = std::fs::read_to_string(&ir_path_resolved)
        .map_err(|e| format!("cannot read {}: {}", ir_path_resolved, e))?;
    // gh#audit-C8. Envelope-aware load (see load_model above).
    let model: ir::Model = ir::from_str(&src)
        .map_err(|e| format!("IR load error from {}: {}", ir_path_resolved, e))?;
    // RC1 in 2026-04-19 engine review.
    ir::validate::validate(&model).map_err(|errs| {
        let mut msg = format!("IR validation failed ({} error(s)):\n", errs.len());
        for e in &errs { msg.push_str(&format!("  - {}\n", e)); }
        msg
    })?;

    // ── Expand --param-vec PREFIX=FILE entries into (NAME, VALUE) pairs ──
    //
    // `--param-vec` is a vector-stratification convenience for `--param`:
    // `--param-vec beta=params.tsv` reads `(key, val)` rows and sets
    // `beta_<key> = val`. The resolver doesn't know about `--param-vec`
    // directly; instead, we expand it into the equivalent fixed-cli pairs
    // and append BEFORE `run.overrides`, so that explicit `--param NAME=VAL`
    // still wins under the resolver's last-wins semantics for tier 5
    // (`fixed_cli`).
    //
    // **Deviation from the legacy precedence**: previously `--param-vec`
    // sat between `--params` (tier-3-equivalent) and scenario (tier 4),
    // meaning scenarios could override `--param-vec` values. Under the
    // resolver this becomes tier 5 alongside `--param`, so scenarios
    // **cannot** override `--param-vec` any more. This is a small
    // behaviour change. No integration test pinned the old order
    // (verified via `rg 'param.vec' rust/crates/cli/tests` — no hits),
    // and the principled mapping is that `--param-vec` is the
    // bulk-set sibling of `--param` and should share its precedence.
    let model_param_set: std::collections::HashSet<String> = model.parameters.iter()
        .map(|p| p.name.clone()).collect();
    let mut fixed_cli: Vec<(String, f64)> = Vec::new();
    for (prefix, file) in &run.set_vec_entries {
        let entries = load_keyed_tsv(file)?;
        for (key, val) in entries {
            let full_name = format!("{}_{}", prefix, key);
            if !model_param_set.contains(&full_name) {
                return Err(format!(
                    "--param-vec {}: unknown parameter '{}'", prefix, full_name));
            }
            fixed_cli.push((full_name, val));
        }
    }
    // `run.overrides` is a HashMap; collect into a deterministic
    // (alphabetical-by-name) Vec so the resolver's provenance is
    // reproducible run-to-run.
    let mut override_vec: Vec<(String, f64)> = run.overrides.iter()
        .map(|(k, &v)| (k.clone(), v))
        .collect();
    override_vec.sort_by(|a, b| a.0.cmp(&b.0));
    fixed_cli.extend(override_vec);

    // ── Build resolver inputs ───────────────────────────────────────────
    //
    // `simulate` and `lineage` are non-inference subcommands. The
    // `fit_toml_*` slots are empty; the resolver's [estimate] kick-out
    // logic is a no-op. All value precedence flows through
    // params_resolver, which is now the sole writer of
    // `model.parameters[i].value` on the simulate/lineage path.
    let fixed_files: Vec<std::path::PathBuf> = run.params_files.iter()
        .map(std::path::PathBuf::from).collect();
    let table_files: std::collections::HashMap<String, std::path::PathBuf> = run.table_files.iter()
        .map(|(k, v)| (k.clone(), std::path::PathBuf::from(v))).collect();
    let ftf: indexmap::IndexMap<String, f64> = indexmap::IndexMap::new();
    let fte: indexmap::IndexSet<String> = indexmap::IndexSet::new();

    let resolved = crate::params_resolver::resolve_parameters(
        crate::params_resolver::ParameterInputs {
            model: &model,
            scenario: run.scenario_name.as_deref(),
            adhoc_enable: &run.adhoc_enable,
            adhoc_disable: &run.adhoc_disable,
            fixed_cli: &fixed_cli,
            fixed_files: &fixed_files,
            fit_toml_fixed: &ftf,
            fit_toml_estimate: &fte,
            table_files: &table_files,
        }
    ).map_err(|e| e.to_string())?;

    crate::params_resolver::print_warnings(&resolved);

    let model = resolved.model;
    let compiled = CompiledModel::new(model.clone())
        .map_err(|e| format!("model compile error: {:?}", e))?;

    Ok((compiled, model))
}

/// Run a simulation and return the full trajectory.
pub fn run_simulation(run: &SimRun) -> Result<(Trajectory, ir::Model), String> {
    run_simulation_with_progress(run, None)
}

/// Like [`run_simulation`], but with an optional per-timestep progress bar.
///
/// `progress = None` reproduces [`run_simulation`] byte-for-byte (it dispatches
/// through the same backend functions, just with a `None` tick — see
/// `tests/progress_tick_invariance.rs` for the byte-identity proof). When
/// `Some(pb)` is passed, the bar's position is advanced to the current
/// simulation time `t` once per timestep via an RNG-free tick closure, giving
/// the `t/t_end` + ETA display for single `camdl simulate` runs. Only the
/// single-cell caller passes `Some(..)`; ensembles pass `None` (they have an
/// outer per-cell bar instead).
pub fn run_simulation_with_progress(
    run: &SimRun,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<(Trajectory, ir::Model), String> {
    // Resolve/compile first, THEN run. Callers that show a simulate progress
    // bar must compile before the bar exists (see `simulate_compiled` and
    // `engine::run_one_cell_with_progress`) — but for the no-bar path the order
    // is immaterial, so this thin wrapper is byte-identical to the old body.
    let (compiled, model) = resolve_run_model(run)?;
    let traj = simulate_compiled(&compiled, &model, run, progress)?;
    Ok((traj, model))
}

/// Run an **already-resolved** model, optionally driving a per-timestep
/// progress bar.
///
/// Split out from [`run_simulation_with_progress`] to fix a rendering bug: the
/// compile step (`camdlc`, reached via [`resolve_run_model`]) shows its own
/// indicatif spinner, and the simulate bar is a second indicatif object. Two
/// draw targets active on stderr at once stomp each other's lines — a garbled
/// bar, and a compile-spinner line left orphaned on screen (the reported
/// Ctrl-C residue). The single-cell caller resolves FIRST (spinner draws and
/// clears), then constructs the bar, then calls this — so only one indicatif
/// target is ever live. `progress = None` is byte-identical to the bar-less
/// path (the tick is RNG-free; see `tests/progress_tick_invariance.rs`).
pub fn simulate_compiled(
    compiled: &CompiledModel,
    model: &ir::Model,
    run: &SimRun,
    progress: Option<&indicatif::ProgressBar>,
) -> Result<Trajectory, String> {
    let params  = compiled.default_params.clone();
    let t_start = model.simulation.t_start;
    let t_end   = model.simulation.t_end;

    use crate::args::types::Backend;

    // Check backend compatibility before running (same gate as the
    // trait-dispatch path; kept so the error wording is unchanged).
    let backend: &dyn Simulate = match run.backend {
        Backend::Gillespie     => &GillespieSim,
        Backend::TauLeap       => &TauLeapSim,
        Backend::ChainBinomial => &ChainBinomialSim,
        Backend::Ode           => &OdeSim,
    };
    let caps = backend.capabilities();
    let required = compiled.required_capabilities();
    if !caps.contains(required) {
        let missing = required.difference(caps);
        return Err(format!(
            "backend {:?} does not support required capabilities: {:?}",
            run.backend, missing
        ));
    }

    // Tick closure: advance the bar to the current sim time. Read-only, RNG-free
    // (the backends call it before any draw). We scale by 1000 so a unit-`dt`
    // run still gets smooth motion on the integer-position bar, and clamp to the
    // configured length.
    let span = (t_end - t_start).max(1e-9);
    let mut tick = |t: f64| {
        if let Some(pb) = progress {
            let frac = ((t - t_start) / span).clamp(0.0, 1.0);
            pb.set_position((frac * 1000.0) as u64);
        }
    };
    let mut tick_opt: Option<&mut dyn FnMut(f64)> =
        if progress.is_some() { Some(&mut tick) } else { None };

    let traj = match run.backend {
        Backend::Gillespie => {
            let cfg = GillespieConfig { t_start, t_end, output_dt: None };
            sim::gillespie::run_gillespie_with_observer(
                compiled, &params, run.seed, &cfg, None, tick_opt.as_deref_mut(),
            )
        }
        Backend::TauLeap => {
            let cfg = TauLeapConfig { t_start, t_end, dt: run.dt };
            sim::tau_leap::run_tau_leap_with_observer(
                compiled, &params, run.seed, &cfg, None, tick_opt.as_deref_mut(),
            )
        }
        Backend::ChainBinomial => {
            let cfg = ChainBinomialConfig { t_start, t_end, dt: run.dt };
            sim::chain_binomial::run_chain_binomial_with_observer(
                compiled, &params, run.seed, &cfg, None, tick_opt.as_deref_mut(),
            )
        }
        Backend::Ode => {
            let cfg = OdeConfig { t_start, t_end, dt: run.dt };
            sim::ode::run_ode(compiled, &params, &cfg, tick_opt.as_deref_mut())
        }
    }
    .map_err(|e| format!("simulation error: {:?}", e))?;

    if let Some(pb) = progress {
        pb.set_position(1000);
    }

    Ok(traj)
}
/// Run a simulation with the Layer-1 lineage **event recorder** attached, and
/// return the count trajectory, resolved model, the recorded [`EventLog`], and
/// whether the backend was exact (Gillespie). The recorder draws no identities;
/// identity attribution happens later in `camdl lineage realize`.
///
/// Supported backends: **Gillespie** (exact), **tau-leap** and
/// **chain-binomial** (batched — the event log records `multiplicity` and a
/// `batched` flag per event so replay reproduces the frozen-pool sub-`dt`
/// semantics). ODE is incompatible (no individuals).
///
/// The count trajectory is byte-identical to the same run without
/// `--event-log` at the same seed (validation Tier 2a): the recorder consumes
/// no randomness, so the simulation is literally unchanged.
pub fn run_simulation_event_log(
    run: &SimRun,
) -> Result<(Trajectory, ir::Model, sim::lineage::EventLog, bool), String> {
    use crate::args::types::Backend;
    use sim::lineage::EventRecorder;

    // Event-log recording is meaningful only for backends with the LINEAGES
    // capability. ODE is the lone incompatible backend (continuous densities,
    // no individuals).
    let backend: &dyn Simulate = match run.backend {
        Backend::Gillespie => &GillespieSim,
        Backend::TauLeap => &TauLeapSim,
        Backend::ChainBinomial => &ChainBinomialSim,
        Backend::Ode => {
            return Err(
                "the event log is incompatible with the ODE backend: ODE \
                 tracks continuous densities, not individuals. Use \
                 --backend gillespie (exact), or tau_leap / chain_binomial \
                 (batched, with a reported sub-dt bias in `lineage realize`)."
                    .to_string(),
            );
        }
    };

    let (compiled, model) = resolve_run_model(run)?;

    if model.identity_tracked_compartments.is_empty() {
        return Err(
            "--event-log requires a model with at least one #[lineage] \
             transition. This model has no lineage annotations, so there is \
             nothing to record. Remove --event-log, or annotate a transition \
             with #[lineage]."
                .to_string(),
        );
    }

    // Capability gate (mirrors the OVERDISPERSION / REAL_COMPARTMENTS pattern):
    // the chosen backend must declare LINEAGES.
    if !backend.capabilities().contains(sim::Capabilities::LINEAGES) {
        return Err(format!(
            "internal error: backend '{}' lacks LINEAGES capability",
            backend.name()
        ));
    }

    let params = compiled.default_params.clone();
    let t_start = model.simulation.t_start;
    let t_end = model.simulation.t_end;

    // Seed the recorder's initial-pool table from the t=0 state.
    let (initial_int, _initial_real) = compiled
        .initial_state(&params)
        .map_err(|e| format!("initial state error: {:?}", e))?;
    let mut recorder = EventRecorder::new(&compiled, &initial_int)
        .map_err(|e| format!("event recorder init error: {:?}", e))?;

    let traj = match run.backend {
        Backend::Gillespie => {
            let cfg = GillespieConfig { t_start, t_end, output_dt: None };
            sim::gillespie::run_gillespie_with_observer(
                &compiled, &params, run.seed, &cfg, Some(&mut recorder), None,
            )
        }
        Backend::TauLeap => {
            let cfg = TauLeapConfig { t_start, t_end, dt: run.dt };
            sim::tau_leap::run_tau_leap_with_observer(
                &compiled, &params, run.seed, &cfg, Some(&mut recorder), None,
            )
        }
        Backend::ChainBinomial => {
            let cfg = ChainBinomialConfig { t_start, t_end, dt: run.dt };
            sim::chain_binomial::run_chain_binomial_with_observer(
                &compiled, &params, run.seed, &cfg, Some(&mut recorder), None,
            )
        }
        Backend::Ode => unreachable!("ODE rejected above"),
    }
    .map_err(|e| format!("simulation error: {:?}", e))?;

    let exact = matches!(run.backend, Backend::Gillespie);
    let event_log = recorder.into_event_log();
    Ok((traj, model, event_log, exact))
}

/// Write a trajectory to a TSV file (same format as `camdl simulate` stdout).
pub fn write_traj_tsv(path: &str, model: &ir::Model, traj: &Trajectory, emit_flows: bool) -> Result<(), String> {
    use std::fs::File;
    let mut f = File::create(path)
        .map_err(|e| format!("cannot create {}: {}", path, e))?;
    write_traj_to(&mut f, model, traj, emit_flows).map_err(|e| e.to_string())
}

/// Render a trajectory TSV into an in-memory buffer — the form the CAS
/// commit hands to the store as the leaf's `traj.tsv` artifact. Byte-
/// identical to [`write_traj_tsv`] (same `write_traj_to` core).
pub fn traj_tsv_bytes(model: &ir::Model, traj: &Trajectory, emit_flows: bool) -> Vec<u8> {
    let mut buf = Vec::new();
    // Writing to a `Vec` is infallible.
    let _ = write_traj_to(&mut buf, model, traj, emit_flows);
    buf
}

/// The shared trajectory-TSV renderer: header + one row per snapshot.
fn write_traj_to(
    w: &mut impl std::io::Write,
    model: &ir::Model,
    traj: &Trajectory,
    emit_flows: bool,
) -> std::io::Result<()> {
    let int_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str()).collect();
    let real_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Real)
        .map(|c| c.name.as_str()).collect();
    let tr_names: Vec<&str> = model.transitions.iter()
        .map(|t| t.name.as_str()).collect();

    // Header
    write!(w, "t")?;
    for n in &int_names  { write!(w, "\t{}", n)?; }
    for n in &real_names { write!(w, "\t{}", n)?; }
    if emit_flows {
        for n in &tr_names { write!(w, "\tflow_{}", n)?; }
    }
    writeln!(w)?;

    // Rows
    for snap in &traj.snapshots {
        write!(w, "{}", snap.t)?;
        for &c in &snap.int_state.counts  { write!(w, "\t{}", c)?; }
        for &v in &snap.real_state.values { write!(w, "\t{:.4}", v)?; }
        if emit_flows {
            for &fl in &snap.flows.counts { write!(w, "\t{}", fl)?; }
        }
        writeln!(w)?;
    }
    Ok(())
}

// ─── Human-friendly relative time ────────────────────────────────────────────

/// Format a SystemTime as a human-readable relative time like "5m ago",
/// "yesterday", or "2w ago". Used by `camdl list`.
///
/// Buckets (each one-bucket-wide for readability; no "59m 42s" precision):
///
/// - `now - from < 60s`        → "just now"
/// - `< 1h`                    → "Nm ago"
/// - `< 24h`                   → "Nh ago"
/// - `< 48h`                   → "yesterday"
/// - `< 7d`                    → "Nd ago"
/// - `< 30d`                   → "Nw ago"       (weeks)
/// - `< 365d`                  → "Nmo ago"      (approx months; 30-day buckets)
/// - `≥ 365d`                  → "Ny ago"       (approx years; 365-day buckets)
/// - future times              → "in the future"
///
/// Pure stdlib — no chrono/humantime/timeago dependency. Supply-chain
/// surface is zero; logic fits in a single function.
pub fn fmt_relative_time(from: std::time::SystemTime, now: std::time::SystemTime) -> String {
    let secs: i64 = match now.duration_since(from) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => return "in the future".to_string(),
    };
    const MIN:  i64 = 60;
    const HOUR: i64 = 60 * MIN;
    const DAY:  i64 = 24 * HOUR;
    const WEEK: i64 = 7 * DAY;
    const MONTH: i64 = 30 * DAY;
    const YEAR: i64 = 365 * DAY;
    if secs < MIN { "just now".to_string() }
    else if secs < HOUR  { format!("{}m ago", secs / MIN) }
    else if secs < DAY   { format!("{}h ago", secs / HOUR) }
    else if secs < 2 * DAY { "yesterday".to_string() }
    else if secs < WEEK  { format!("{}d ago", secs / DAY) }
    else if secs < MONTH { format!("{}w ago", secs / WEEK) }
    else if secs < YEAR  { format!("{}mo ago", secs / MONTH) }
    else                 { format!("{}y ago", secs / YEAR) }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_relative_to_toml ─────────────────────────────────────────────

    #[test]
    fn resolve_relative_to_toml_anchors_at_toml_dir() {
        let toml = std::path::Path::new("/proj/fits/he2010.fit.toml");
        // Relative paths anchor to the toml's parent directory.
        assert_eq!(
            resolve_relative_to_toml(toml, "../models/he2010.camdl"),
            "/proj/fits/../models/he2010.camdl");
        assert_eq!(
            resolve_relative_to_toml(toml, "data/cases.tsv"),
            "/proj/fits/data/cases.tsv");
    }

    #[test]
    fn resolve_relative_to_toml_passes_absolute_through() {
        let toml = std::path::Path::new("/proj/fits/he2010.fit.toml");
        // Absolute input → unchanged. Even when the toml is in a
        // different tree, an absolute path's intent is "this exact
        // location, regardless of context."
        assert_eq!(
            resolve_relative_to_toml(toml, "/abs/path/model.camdl"),
            "/abs/path/model.camdl");
    }

    #[test]
    fn resolve_relative_to_toml_handles_toml_in_cwd_root() {
        // When the toml is just "fit.toml" (no directory component),
        // `Path::parent()` returns Some("") — empty anchor — and
        // joining it with the relative path is a no-op. The returned
        // string is "data/cases.tsv" (CWD-relative), which resolves
        // the same way as the user's input would have pre-fix. This
        // is the trivial case where toml-dir-relative and CWD-relative
        // coincide; the fix is a no-op there.
        let toml = std::path::Path::new("fit.toml");
        assert_eq!(
            resolve_relative_to_toml(toml, "data/cases.tsv"),
            "data/cases.tsv");
    }

    // ── camdlc version check ─────────────────────────────────────────────────

    #[test]
    fn version_output_match() {
        assert!(eval_version_output(b"abc1234\n", true, "abc1234", "test", None).is_ok());
        // trim whitespace variants
        assert!(eval_version_output(b"abc1234", true, "abc1234", "test", None).is_ok());
    }

    #[test]
    fn version_output_mismatch() {
        let err = eval_version_output(b"old0000\n", true, "abc1234", "/usr/bin/camdlc", None)
            .unwrap_err();
        assert!(err.contains("version mismatch"), "unexpected message: {err}");
        assert!(err.contains("abc1234"), "our hash missing: {err}");
        assert!(err.contains("old0000"), "reported hash missing: {err}");
        assert!(err.contains("/usr/bin/camdlc"), "location missing: {err}");
        // No hint passed, so the shadowing-note line must not appear —
        // ordinary version skew with no shadowing detected should keep
        // the existing single-block message intact.
        assert!(!err.contains("shadowing"), "unsolicited shadow hint: {err}");
    }

    #[test]
    fn version_output_mismatch_with_shadow_hint() {
        let hint = "  Note: another `camdl` is shadowing this install.\n  \
                    Running:   /Users/x/.cargo/bin/camdl\n  \
                    Installed: /Users/x/.local/bin/camdl  (alongside the camdlc above)\n  \
                    Fix: `rm /Users/x/.cargo/bin/camdl`, or put /Users/x/.local/bin ahead of /Users/x/.cargo/bin on your PATH.";
        let err = eval_version_output(
            b"new1234\n", true, "old0000", "/Users/x/.local/bin/camdlc", Some(hint))
            .unwrap_err();
        assert!(err.contains("version mismatch"), "header missing: {err}");
        assert!(err.contains("shadowing this install"), "shadow hint missing: {err}");
        assert!(err.contains("/Users/x/.cargo/bin/camdl"),
            "running path missing from hint: {err}");
        assert!(err.contains("/Users/x/.local/bin/camdl"),
            "installed path missing from hint: {err}");
        // The standard `make install` advice and the shadow note must
        // both appear — neither suppresses the other.
        assert!(err.contains("make build-ocaml && make install"),
            "standard advice elided when hint present: {err}");
    }

    #[test]
    fn version_output_old_build() {
        let err = eval_version_output(b"", false, "abc1234", "on PATH", None)
            .unwrap_err();
        assert!(err.contains("old build"), "unexpected message: {err}");
        assert!(err.contains("on PATH"), "location missing: {err}");
    }

    #[test]
    fn version_output_old_build_with_shadow_hint_appends() {
        let hint = "  Note: shadow detected at /a vs /b.";
        let err = eval_version_output(b"", false, "abc1234", "on PATH", Some(hint))
            .unwrap_err();
        assert!(err.contains("old build"), "header missing: {err}");
        assert!(err.contains("shadow detected"), "hint elided: {err}");
    }

    #[test]
    fn camdlc_versioned_name_format() {
        let name = camdlc_versioned_name();
        assert!(name.starts_with("camdlc-"), "unexpected prefix: {name}");
        assert!(name.contains(crate::version::GIT_HASH), "hash missing: {name}");
    }

    /// Helper: path to the sir_vaccination golden IR (has 4 params:
    /// beta=0.3, gamma=0.1, vaccine_coverage=0.5, rho=10.0).
    /// Resolves relative to the repo root (tests run from rust/).
    fn sir_model() -> String {
        // Resolve relative to the crate manifest dir (rust/crates/cli/)
        let manifest = env!("CARGO_MANIFEST_DIR");
        let path = std::path::PathBuf::from(manifest)
            .join("../../../ir/golden/sir_vaccination.ir.json");
        let path = path.canonicalize()
            .unwrap_or_else(|_| panic!(
                "cannot find sir_vaccination.ir.json (tried {})", path.display()));
        path.to_str().unwrap().to_string()
    }

    fn write_toml(dir: &std::path::Path, name: &str, content: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, content).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn base_sim_run(ir_path: &str) -> SimRun {
        SimRun {
            ir_path: ir_path.to_string(),
            backend: crate::args::types::Backend::ChainBinomial,
            dt: 1.0,
            seed: 1,
            ..Default::default()
        }
    }

    /// Extract final param values from a successful simulation.
    fn resolved_params(run: &SimRun) -> Result<HashMap<String, f64>, String> {
        let (_, model) = run_simulation(run)?;
        let compiled = sim::CompiledModel::new(model.clone())
            .map_err(|e| format!("{:?}", e))?;
        Ok(model.parameters.iter().map(|p| {
            let idx = compiled.param_index[p.name.as_str()];
            (p.name.clone(), compiled.default_params[idx])
        }).collect())
    }

    // ── Params file loading ─────────────────────────────────────────────────

    #[test]
    fn single_params_file_sets_values() {
        let dir = tempfile::tempdir().unwrap();
        let pf = write_toml(dir.path(), "params.toml",
            "beta = 0.5\ngamma = 0.2\nvaccine_coverage = 0.8\nrho = 5.0\n");
        let run = SimRun { params_files: vec![pf], ..base_sim_run(&sir_model()) };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.5).abs() < 1e-10);
        assert!((params["gamma"] - 0.2).abs() < 1e-10);
        assert!((params["rho"] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn unknown_param_in_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let pf = write_toml(dir.path(), "params.toml", "betta = 0.5\n");
        let run = SimRun { params_files: vec![pf], ..base_sim_run(&sir_model()) };
        let err = run_simulation(&run).unwrap_err();
        assert!(err.contains("unknown parameter 'betta'"));
        assert!(err.contains("Available parameters"));
    }

    // ── Stacked params files (later overrides earlier) ──────────────────────

    #[test]
    fn stacked_params_later_overrides_earlier() {
        let dir = tempfile::tempdir().unwrap();
        let pf1 = write_toml(dir.path(), "base.toml",
            "beta = 0.3\ngamma = 0.1\nvaccine_coverage = 0.5\nrho = 10.0\n");
        let pf2 = write_toml(dir.path(), "override.toml",
            "beta = 0.7\ngamma = 0.2\n");
        let run = SimRun {
            params_files: vec![pf1, pf2],
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.7).abs() < 1e-10, "beta should be 0.7, got {}", params["beta"]);
        assert!((params["gamma"] - 0.2).abs() < 1e-10, "gamma should be 0.2");
        assert!((params["rho"] - 10.0).abs() < 1e-10);
        assert!((params["vaccine_coverage"] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn three_stacked_params_last_wins() {
        let dir = tempfile::tempdir().unwrap();
        let pf1 = write_toml(dir.path(), "a.toml",
            "beta = 0.1\ngamma = 0.1\nvaccine_coverage = 0.5\nrho = 10.0\n");
        let pf2 = write_toml(dir.path(), "b.toml", "beta = 0.2\n");
        let pf3 = write_toml(dir.path(), "c.toml", "beta = 0.9\n");
        let run = SimRun {
            params_files: vec![pf1, pf2, pf3],
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.9).abs() < 1e-10, "third file should win");
    }

    #[test]
    fn unknown_param_in_second_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let pf1 = write_toml(dir.path(), "base.toml",
            "beta = 0.3\ngamma = 0.1\nvaccine_coverage = 0.5\nrho = 10.0\n");
        let pf2 = write_toml(dir.path(), "bad.toml", "typo_param = 0.5\n");
        let run = SimRun {
            params_files: vec![pf1, pf2],
            ..base_sim_run(&sir_model())
        };
        let err = run_simulation(&run).unwrap_err();
        assert!(err.contains("unknown parameter 'typo_param'"));
    }

    // ── --param CLI overrides ───────────────────────────────────────────────

    #[test]
    fn cli_param_overrides_file() {
        let dir = tempfile::tempdir().unwrap();
        let pf = write_toml(dir.path(), "params.toml",
            "beta = 0.3\ngamma = 0.1\nvaccine_coverage = 0.5\nrho = 10.0\n");
        let run = SimRun {
            params_files: vec![pf],
            overrides: [("beta".to_string(), 0.99)].into(),
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.99).abs() < 1e-10, "CLI --param should override file");
        assert!((params["gamma"] - 0.1).abs() < 1e-10, "gamma unchanged");
    }

    #[test]
    fn cli_param_overrides_stacked_files() {
        let dir = tempfile::tempdir().unwrap();
        let pf1 = write_toml(dir.path(), "base.toml",
            "beta = 0.3\ngamma = 0.1\nvaccine_coverage = 0.5\nrho = 10.0\n");
        let pf2 = write_toml(dir.path(), "override.toml", "beta = 0.7\n");
        let run = SimRun {
            params_files: vec![pf1, pf2],
            overrides: [("beta".to_string(), 1.5)].into(),
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 1.5).abs() < 1e-10, "CLI --param beats stacked files");
    }

    #[test]
    fn unknown_cli_param_errors() {
        let run = SimRun {
            overrides: [("nonexistent".to_string(), 0.5)].into(),
            ..base_sim_run(&sir_model())
        };
        let err = run_simulation(&run).unwrap_err();
        assert!(err.contains("unknown parameter 'nonexistent'"));
    }

    // ── Model defaults (no params file, no overrides) ───────────────────────

    #[test]
    fn model_defaults_used_when_no_params() {
        let run = base_sim_run(&sir_model());
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.3).abs() < 1e-10);
        assert!((params["gamma"] - 0.1).abs() < 1e-10);
        assert!((params["vaccine_coverage"] - 0.5).abs() < 1e-10);
        assert!((params["rho"] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn cli_param_without_file_overrides_model_default() {
        let run = SimRun {
            overrides: [("beta".to_string(), 2.0)].into(),
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 2.0).abs() < 1e-10);
        assert!((params["gamma"] - 0.1).abs() < 1e-10);
    }

    // ── Partial params files ────────────────────────────────────────────────

    #[test]
    fn partial_params_file_leaves_others_at_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let pf = write_toml(dir.path(), "partial.toml", "beta = 0.99\n");
        let run = SimRun {
            params_files: vec![pf],
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.99).abs() < 1e-10);
        assert!((params["gamma"] - 0.1).abs() < 1e-10);
        assert!((params["rho"] - 10.0).abs() < 1e-10);
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn same_value_override_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let pf = write_toml(dir.path(), "params.toml",
            "beta = 0.3\ngamma = 0.1\nvaccine_coverage = 0.5\nrho = 10.0\n");
        let run = SimRun {
            params_files: vec![pf],
            overrides: [("beta".to_string(), 0.3)].into(),
            ..base_sim_run(&sir_model())
        };
        let params = resolved_params(&run).unwrap();
        assert!((params["beta"] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn load_params_toml_basic() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_toml(dir.path(), "test.toml", "x = 1.5\ny = 2\n");
        let vals = load_params_toml(&path).unwrap();
        assert!((vals["x"] - 1.5).abs() < 1e-10);
        assert!((vals["y"] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn load_params_toml_handles_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_toml(dir.path(), "test.toml",
            "# This is a comment\nx = 1.5\n# Another comment\ny = 2.0\n");
        let vals = load_params_toml(&path).unwrap();
        assert_eq!(vals.len(), 2);
        assert!((vals["x"] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn fmt_relative_time_buckets() {
        use std::time::{Duration, SystemTime};
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(2_000_000_000);
        let at = |secs_ago: u64| now - Duration::from_secs(secs_ago);

        assert_eq!(fmt_relative_time(at(0), now),   "just now");
        assert_eq!(fmt_relative_time(at(30), now),  "just now");
        assert_eq!(fmt_relative_time(at(60), now),  "1m ago");
        assert_eq!(fmt_relative_time(at(300), now), "5m ago");
        assert_eq!(fmt_relative_time(at(3600), now), "1h ago");
        assert_eq!(fmt_relative_time(at(3600 * 5), now), "5h ago");
        assert_eq!(fmt_relative_time(at(86400), now), "yesterday");
        assert_eq!(fmt_relative_time(at(86400 * 2), now), "2d ago");
        assert_eq!(fmt_relative_time(at(86400 * 6), now), "6d ago");
        assert_eq!(fmt_relative_time(at(86400 * 7), now), "1w ago");
        assert_eq!(fmt_relative_time(at(86400 * 29), now), "4w ago");
        assert_eq!(fmt_relative_time(at(86400 * 30), now), "1mo ago");
        assert_eq!(fmt_relative_time(at(86400 * 180), now), "6mo ago");
        assert_eq!(fmt_relative_time(at(86400 * 365), now), "1y ago");
        assert_eq!(fmt_relative_time(at(86400 * 365 * 3), now), "3y ago");
    }

    #[test]
    fn fmt_relative_time_future() {
        use std::time::{Duration, SystemTime};
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        let future = now + Duration::from_secs(3600);
        assert_eq!(fmt_relative_time(future, now), "in the future");
    }

    // ── apply_scenario_filter: the spec contract ────────────────────────
    //
    // Spec (camdl-language-spec.md §14 / §14.4):
    //   - events (always_active = true)        : on by default, off iff in `disable`
    //   - interventions (always_active = false): off by default, on iff in `enable`
    //   - wildcard `"*"` matches every toggleable intervention (`enable`)
    //     or every action including events (`disable`)

    use ir::intervention::{Intervention, InterventionSchedule};

    fn tok_iv(name: &str, base: Option<&str>, always_active: bool) -> Intervention {
        Intervention {
            name: name.into(),
            base_name: base.map(str::to_owned),
            schedule: InterventionSchedule::AtTimes(vec![10.0]),
            actions: vec![],
            always_active,
        }
    }

    fn mk_model(ivs: Vec<Intervention>) -> ir::Model {
        ir::Model {
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![], transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], observations: vec![],
            bindings: vec![],
            parameters: vec![],
            initial_conditions: ir::model::InitialConditions::Explicit(
                std::collections::HashMap::new()),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![]),
                format: "tsv".into(), trajectory: true, observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0, t_end: 1.0, time_semantics: "continuous".into(),
                dt: None, rng_seed: None,
            },
            interventions: ivs,
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![],
        }
    }

    #[test]
    fn scenario_filter_default_preserves_events_drops_interventions() {
        // The critical spec default: with NO enable/disable, events stay
        // and toggleable interventions are cleared. This also guards the
        // util.rs:448 latent bug where `.clear()` was nuking events.
        let mut m = mk_model(vec![
            tok_iv("cohort_entry", None, true),           // event
            tok_iv("births",       None, true),           // event
            tok_iv("sia_round_1",  None, false),          // intervention
            tok_iv("lockdown",     None, false),          // intervention
        ]);
        apply_scenario_filter(&mut m, &[], &[]).unwrap();
        let names: Vec<&str> = m.interventions.iter().map(|iv| iv.name.as_str()).collect();
        assert_eq!(names, vec!["cohort_entry", "births"],
            "events must survive default filter; interventions must not");
    }

    #[test]
    fn scenario_filter_enable_activates_by_exact_name() {
        let mut m = mk_model(vec![
            tok_iv("sia_round_1", None, false),
            tok_iv("sia_round_2", None, false),
        ]);
        apply_scenario_filter(&mut m, &["sia_round_1".into()], &[]).unwrap();
        let names: Vec<&str> = m.interventions.iter().map(|iv| iv.name.as_str()).collect();
        assert_eq!(names, vec!["sia_round_1"]);
    }

    #[test]
    fn scenario_filter_enable_activates_by_base_name_family() {
        // Indexed interventions expand to per-stratum members. One enable
        // entry with the base_name matches every member.
        let mut m = mk_model(vec![
            tok_iv("sia_north", Some("sia"), false),
            tok_iv("sia_south", Some("sia"), false),
            tok_iv("sia_east",  Some("sia"), false),
            tok_iv("other",     None,        false),
        ]);
        apply_scenario_filter(&mut m, &["sia".into()], &[]).unwrap();
        let names: Vec<&str> = m.interventions.iter().map(|iv| iv.name.as_str()).collect();
        assert_eq!(names, vec!["sia_north", "sia_south", "sia_east"],
            "family-name `sia` enables every expansion; `other` stays off");
    }

    #[test]
    fn scenario_filter_wildcard_enable_activates_all_interventions() {
        let mut m = mk_model(vec![
            tok_iv("event_a",        None, true),
            tok_iv("intervention_a", None, false),
            tok_iv("intervention_b", None, false),
        ]);
        apply_scenario_filter(&mut m, &["*".into()], &[]).unwrap();
        let names: Vec<&str> = m.interventions.iter().map(|iv| iv.name.as_str()).collect();
        assert_eq!(names, vec!["event_a", "intervention_a", "intervention_b"]);
    }

    #[test]
    fn scenario_filter_disable_silences_event() {
        // Explicit disable MUST win over always_active. Only way to
        // turn an event off.
        let mut m = mk_model(vec![
            tok_iv("cohort_entry", None, true),
            tok_iv("births",       None, true),
        ]);
        apply_scenario_filter(&mut m, &[], &["cohort_entry".into()]).unwrap();
        let names: Vec<&str> = m.interventions.iter().map(|iv| iv.name.as_str()).collect();
        assert_eq!(names, vec!["births"], "`cohort_entry` must be disabled, `births` stays");
    }

    #[test]
    fn scenario_filter_disable_overrides_enable() {
        // If the same name appears in both enable and disable, disable wins.
        let mut m = mk_model(vec![
            tok_iv("sia", None, false),
        ]);
        apply_scenario_filter(&mut m, &["sia".into()], &["sia".into()]).unwrap();
        assert!(m.interventions.is_empty(), "disable trumps enable");
    }

    #[test]
    fn scenario_filter_unknown_name_errors() {
        let mut m = mk_model(vec![
            tok_iv("sia", None, false),
        ]);
        let err = apply_scenario_filter(&mut m, &["nonesuch".into()], &[]).unwrap_err();
        assert!(err.contains("does not match"), "err should cite the mismatch: {}", err);
    }

    #[test]
    fn scenario_filter_mixed_events_and_interventions() {
        // End-to-end-shaped case: realistic mix of structural events
        // and toggleable policy interventions, single enable selects
        // one family, disable silences one event.
        let mut m = mk_model(vec![
            tok_iv("cohort_entry",    None,                true),
            tok_iv("births",          None,                true),
            tok_iv("sia_north",       Some("sia"),         false),
            tok_iv("sia_south",       Some("sia"),         false),
            tok_iv("lockdown_2022",   None,                false),
        ]);
        apply_scenario_filter(&mut m,
            &["sia".into()],
            &["cohort_entry".into()],
        ).unwrap();
        let names: std::collections::HashSet<&str> =
            m.interventions.iter().map(|iv| iv.name.as_str()).collect();
        // births stays (event, not disabled); cohort_entry gone (disabled)
        // sia_north + sia_south on (family enabled); lockdown off (not enabled)
        assert!(names.contains("births"));
        assert!(!names.contains("cohort_entry"));
        assert!(names.contains("sia_north"));
        assert!(names.contains("sia_south"));
        assert!(!names.contains("lockdown_2022"));
    }

    /// Measure `camdlc --camdl-version` subprocess latency and OnceLock
    /// short-circuit overhead.
    ///
    /// Run with `cargo test bench_camdlc_version -- --nocapture` to see timing.
    /// If the subprocess is consistently >50ms, prefer the versioned-binary
    /// fast path (`make install` installs `camdlc-<hash>` alongside `camdlc`).
    #[test]
    fn bench_camdlc_version() {
        // Cold subprocess: first call hits the OS
        let t0 = std::time::Instant::now();
        let result = std::process::Command::new("camdlc")
            .arg("--camdl-version")
            .output();
        let cold = t0.elapsed();

        match &result {
            Ok(out) => eprintln!(
                "camdlc --camdl-version (cold):  {:>6.1?}  status={}  hash={:?}",
                cold, out.status,
                String::from_utf8_lossy(&out.stdout).trim()
            ),
            Err(e) => eprintln!(
                "camdlc --camdl-version (cold):  {:>6.1?}  error: {}", cold, e
            ),
        }

        // Warm OnceLock: initialise the lock, then measure a subsequent call
        let flag = crate::util::camdlc_checked_flag();
        flag.get_or_init(|| ());  // ensure it's set
        let t1 = std::time::Instant::now();
        flag.get_or_init(|| ());
        let warm = t1.elapsed();
        eprintln!("OnceLock short-circuit (warm): {:>6.1?}", warm);

        // Verdict
        eprintln!();
        if cold.as_millis() < 20 {
            eprintln!("verdict: subprocess is fast (<20ms) — OnceLock path is fine");
        } else {
            eprintln!(
                "verdict: subprocess is slow ({}ms) — prefer `make install` so \
                 `camdlc-<hash>` is present next to camdl for zero-overhead path",
                cold.as_millis()
            );
        }
    }

    // ── validate_parameter_values (gh#31) ────────────────────────────────

    /// Build a minimal `ir::Model` with a single parameter, used as a
    /// fixture for the validator unit tests. Avoids depending on a
    /// golden IR file just to assert pure-function behavior — the
    /// validator only inspects `model.parameters`, so every other
    /// field is zero/empty/default.
    fn model_with_one_param(value: Option<f64>, bounds: Option<(f64, f64)>) -> ir::Model {
        ir::Model {
            name: "fixture".into(),
            version: "0.0".into(),
            time_unit: "days".into(),
            description: None,
            origin: None, origin_rata_die: None,
            compartments: Vec::new(),
            transitions: Vec::new(),
            ode_equations: Vec::new(),
            time_functions: Vec::new(),
            tables: Vec::new(),
            interventions: Vec::new(),
            observations: Vec::new(),
            bindings: vec![],
            parameters: vec![ir::parameter::Parameter {
                name: "x".into(),
                value,
                bounds,
                prior: None,
                hierarchical: None,
                transform: None,
                initial_value: None,
                param_kind: None,
                param_dim: None,
            }],
            initial_conditions: ir::model::InitialConditions::Explicit(HashMap::new()),
            output: ir::model::OutputConfig {
                times: ir::model::OutputSchedule::AtTimes(vec![0.0]),
                format: "tsv".into(),
                trajectory: true,
                observations: false,
            },
            simulation: ir::model::SimulationConfig {
                t_start: 0.0, t_end: 1.0,
                time_semantics: "days".into(), dt: None, rng_seed: None,
            },
            presets: Vec::new(),
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![],
        }
    }

    #[test]
    fn validate_accepts_in_bounds_value() {
        let m = model_with_one_param(Some(0.5), Some((0.0, 1.0)));
        assert!(validate_parameter_values(&m).is_ok());
    }

    #[test]
    fn validate_accepts_value_on_lower_bound() {
        // Bounds are inclusive — a value exactly on the lower bound
        // must pass. Open-interval semantics would force the user to
        // perturb a deliberately-set boundary value just to satisfy
        // the validator, which is a UX hazard for "natural" boundary
        // values like 0 for a probability or 1 for a count.
        let m = model_with_one_param(Some(0.0), Some((0.0, 1.0)));
        assert!(validate_parameter_values(&m).is_ok());
    }

    #[test]
    fn validate_accepts_value_on_upper_bound() {
        let m = model_with_one_param(Some(1.0), Some((0.0, 1.0)));
        assert!(validate_parameter_values(&m).is_ok());
    }

    #[test]
    fn validate_rejects_value_above_upper_bound() {
        let m = model_with_one_param(Some(1.5), Some((0.0, 1.0)));
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("'x'"), "param name absent: {err}");
        assert!(err.contains("1.5"), "supplied value absent: {err}");
        assert!(err.contains("[0") && err.contains("1]"),
            "declared bounds absent: {err}");
        assert!(err.contains("outside"), "violation kind unclear: {err}");
    }

    #[test]
    fn validate_rejects_value_below_lower_bound() {
        let m = model_with_one_param(Some(-0.1), Some((0.0, 1.0)));
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("'x'") && err.contains("outside"),
            "expected bounds violation; got: {err}");
    }

    #[test]
    fn validate_rejects_nan() {
        // NaN must error even when bounds are absent — a NaN
        // parameter is downstream-poison regardless of constraint.
        let m = model_with_one_param(Some(f64::NAN), None);
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("'x'"), "param name absent: {err}");
        assert!(err.contains("not finite"),
            "should report finiteness, not bounds: {err}");
        // Bounds-violation hint must NOT appear: the underlying
        // problem is finiteness, and surfacing the wrong category
        // wastes the user's first fix attempt.
        assert!(!err.contains("outside declared bounds"),
            "NaN reported as bounds violation; got: {err}");
    }

    #[test]
    fn validate_rejects_positive_infinity() {
        let m = model_with_one_param(Some(f64::INFINITY), Some((0.0, 1e9)));
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("not finite"),
            "+∞ should fail the finiteness check before bounds; got: {err}");
    }

    #[test]
    fn validate_rejects_negative_infinity() {
        let m = model_with_one_param(Some(f64::NEG_INFINITY), None);
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("not finite"),
            "-∞ should fail the finiteness check; got: {err}");
    }

    #[test]
    fn validate_accepts_unset_value() {
        // value = None: subcommands resolve from priors / scenarios
        // later; the validator must NOT pre-empt that with a "missing
        // value" error. The brief is explicit on this.
        let m = model_with_one_param(None, Some((0.0, 1.0)));
        assert!(validate_parameter_values(&m).is_ok());
    }

    #[test]
    fn validate_accepts_finite_value_when_no_bounds_declared() {
        // No bounds: any finite value passes.
        let m = model_with_one_param(Some(1e9), None);
        assert!(validate_parameter_values(&m).is_ok());
        let m = model_with_one_param(Some(-1e9), None);
        assert!(validate_parameter_values(&m).is_ok());
    }

    #[test]
    fn validate_collects_all_violations_in_one_message() {
        // Two violations: report both. Saves the user from a
        // fix-rerun-fix loop.
        let mut m = model_with_one_param(Some(5.0), Some((0.0, 1.0)));
        m.parameters.push(ir::parameter::Parameter {
            name: "y".into(),
            value: Some(f64::NAN),
            bounds: None,
            prior: None,
            hierarchical: None,
            transform: None,
            initial_value: None,
            param_kind: None,
            param_dim: None,
        });
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("'x'"), "x violation missing: {err}");
        assert!(err.contains("'y'"), "y violation missing: {err}");
        // Both messages should be in the same error string.
        assert!(err.contains("outside") && err.contains("not finite"),
            "expected both kinds of violation reported; got: {err}");
    }
}
