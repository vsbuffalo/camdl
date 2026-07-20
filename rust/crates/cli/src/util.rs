use std::collections::HashMap;
use ir::intervention::Intervention;
use sim::{
    CompiledModel, GillespieSim, ChainBinomialSim, OdeSim,
    config::{GillespieConfig, ChainBinomialConfig, OdeConfig},
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
    // Full IR (state-Jacobian emitted). The metadata / identity / docs loaders
    // that route through here (`load_model`, `load_model_docs`) must never
    // withhold the WrtPop Jacobian: a full IR is a safe superset for every
    // consumer, and run identity is gradient-independent (runid SV=2), so a full
    // vs lean compile of the same model share the same digest. The lean opt-out
    // (`--no-state-grad`) lives on the run-producing path (`resolve_ir_path`),
    // which knows the method and so can safely drop the Jacobian.
    run_camdlc_compile(camdl_path, None, true)
}

/// Compile `camdl_path` to IR JSON. When `emit_deps` is `Some(path)`, camdlc
/// additionally writes its read-closure depfile there in the SAME compile (one
/// invocation, IR on stdout + depfile to the path), so the IR cache can key on
/// the contents of `read()`-loaded files (gh#260).
///
/// `needs_state_grad` selects whether camdlc emits the WrtPop state-Jacobian
/// (`rate_state_grad` / `projection_state_grad`, gh#439). It is consumed only by
/// the ODE forward-sensitivity gradient (`fit --method nuts` on the `ode`
/// backend); every other path — `simulate`, IF2, PMMH, PF, PGAS, `mh` — never
/// reads it. When `false` we pass `--no-state-grad`, which skips the WrtPop
/// autodiff pass and leaves both maps empty (often 95%+ of the IR on
/// mean-field-coupled models). The IR-cache key folds this bit (`ir_cache_key`),
/// so a lean-compiled entry is never served to a nuts+ode fit (which needs the
/// Jacobian) and vice-versa.
pub(crate) fn run_camdlc_compile(
    camdl_path: &str,
    emit_deps: Option<&std::path::Path>,
    needs_state_grad: bool,
) -> Result<String, String> {
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

    // The blocking subprocess call. With `emit_deps`, camdlc writes the
    // read-closure depfile alongside emitting IR on stdout — one compile.
    let mut cmd = std::process::Command::new(&camdlc);
    cmd.arg(camdl_path);
    if let Some(dp) = emit_deps {
        cmd.arg("--emit-deps").arg(dp);
    }
    // gh#272: LICM is on by default; `--no-licm` (or `CAMDL_NO_LICM`) turns it OFF
    // in the camdlc subprocess. Set it explicitly so the CLI FLAG reaches the
    // subprocess (the env var would inherit on its own, but the flag does not).
    // The IR-cache key already folds `licm_enabled()`, so a flag flip recompiles
    // rather than serving the stale variant.
    if !licm_enabled() {
        cmd.env("CAMDL_NO_LICM", "1");
    }
    // gh#439 A2: skip the WrtPop state-Jacobian when no downstream consumer reads
    // it. Only nuts+ode does; the runtime passes `needs_state_grad = false` for
    // every other path, shrinking the IR (`--no-state-grad` leaves
    // `rate_state_grad` / `projection_state_grad` empty). This is IR-affecting, so
    // the resolved bit is folded into the IR-cache key (`ir_cache_key`) — a flip
    // recompiles rather than serving the wrong variant.
    if !needs_state_grad {
        cmd.arg("--no-state-grad");
    }
    let output = cmd.output();

    spinner.finish_and_clear();

    let output = output.map_err(|e| format!("cannot run {}: {}", camdlc.display(), e))?;
    if !output.status.success() {
        // camdlc prints errors to stderr — pass them through
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    String::from_utf8(output.stdout)
        .map_err(|e| format!("camdlc output not UTF-8: {}", e))
}

/// Compile `camdl_path` and write its read-closure depfile to `deps_out` (for
/// `camdl mre`). One compile; the IR on stdout is discarded — mre v1 only needs
/// the dependency list. Surfaces camdlc's stderr on failure.
pub(crate) fn camdlc_emit_deps(
    camdl_path: &str,
    deps_out: &std::path::Path,
) -> Result<(), String> {
    let camdlc = find_camdlc()?;
    let output = std::process::Command::new(&camdlc)
        .arg(camdl_path)
        .arg("--emit-deps")
        .arg(deps_out)
        .output()
        .map_err(|e| format!("cannot run {}: {}", camdlc.display(), e))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(())
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

/// Process-wide IR-cache disable (set by `--no-ir-cache`).
static IR_CACHE_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Disable the compiled-IR cache for this process (the `--no-ir-cache` flag).
pub fn set_ir_cache_disabled(disabled: bool) {
    IR_CACHE_DISABLED.store(disabled, std::sync::atomic::Ordering::Relaxed);
}

fn ir_cache_disabled() -> bool {
    IR_CACHE_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os("CAMDL_NO_IR_CACHE").is_some()
}

/// Process-wide LICM disable (gh#272, set by `--no-licm`). LICM is a compile-time
/// pass that hoists loop-invariant param/table-only subexpressions out of the
/// rate trees — value-preserving (proven byte-identical by `gate_licm_ab`), so it
/// is ON by default and only makes a fittable in-model kernel run at
/// precomputed-kernel speed. It CHANGES the IR camdlc emits (adds
/// `per_eval_bindings` + `Expr::PerEvalRef`) for models with hoistable structure,
/// so it is an IR-affecting compiler switch: the resolved on/off state is folded
/// into the IR-cache key (else flipping it serves the stale variant) AND the
/// disable is passed to the camdlc subprocess. `--no-licm` / `CAMDL_NO_LICM` is
/// the escape hatch (debugging / A-B), mirroring constant_fold /
/// `CAMDL_NO_CONSTANT_FOLD`.
static LICM_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Disable LICM for this process (the `--no-licm` flag). `CAMDL_NO_LICM` is the
/// equivalent env escape hatch (either forces the inlined IR).
pub fn set_licm_disabled(disabled: bool) {
    LICM_DISABLED.store(disabled, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn licm_enabled() -> bool {
    !(LICM_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os("CAMDL_NO_LICM").is_some())
}

/// The content-addressed cache key for a compiled `.camdl`. Folds the model
/// bytes together with the compiler git hash and the IR schema version, so a
/// model edit, a camdlc upgrade, or an IR-format change each produces a
/// distinct key (no stale IR can be served). `.camdl` is single-file (no
/// includes), so the model bytes are the whole compile input.
///
/// The key must fold in EVERY input that changes the emitted IR bytes: the
/// model source, the camdlc git-hash, the IR schema version, and the compiler
/// switches that alter output. That switch set is `CAMDL_NO_CONSTANT_FOLD`
/// (presence flips the fold pass off → unfolded/dense IR), `CAMDL_NO_LICM` /
/// `--no-licm` (presence flips loop-invariant code motion OFF → inlined IR
/// without `per_eval_bindings`; LICM is on by default), and `--no-state-grad`
/// (gh#439: skips the WrtPop state-Jacobian → `rate_state_grad` /
/// `projection_state_grad` empty). All are `compiler.ml` switches that alter
/// output. Any future IR-affecting compiler env var or flag MUST be added here,
/// or flipping it on an already-cached model would silently serve the stale
/// variant. (`licm_enabled` and `state_grad_emitted` are resolved on/off states.)
///
/// Note the state-Jacobian bit is a COMPILE-cache concern only, not a run-identity
/// one: a lean IR and a full IR of the same model have different bytes (so must
/// key distinctly here) but the SAME model digest (runid excludes the gradient
/// maps, SV=2). Serving a lean entry to a nuts+ode fit would leave it without the
/// Jacobian it needs, so the two variants must not collide in this cache.
pub(crate) fn ir_cache_key(
    content: &[u8],
    camdlc_ver: &str,
    ir_ver: &str,
    fold_disabled: bool,
    licm_enabled: bool,
    state_grad_emitted: bool,
) -> String {
    let mut buf = Vec::with_capacity(content.len() + camdlc_ver.len() + ir_ver.len() + 8);
    buf.extend_from_slice(content);
    buf.push(0);
    buf.extend_from_slice(camdlc_ver.as_bytes());
    buf.push(0);
    buf.extend_from_slice(ir_ver.as_bytes());
    buf.push(0);
    buf.push(fold_disabled as u8);
    buf.push(licm_enabled as u8);
    buf.push(state_grad_emitted as u8);
    crate::hashing::sha256_hex(&buf)
}

// ─── read()-input freshness (gh#260) ─────────────────────────────────────────
//
// The cache key (above) folds only the `.camdl` bytes, but a model can pull in
// external files via `read("pop.tsv")` at compile time — and editing one of
// those, with the `.camdl` untouched, must NOT serve IR built from the stale
// data. The key can't carry the read()-contents (discovering them needs a
// compile, which is exactly what the cache skips). So instead each cache entry
// gets a sidecar recording its read()-closure with content hashes; a cache hit
// re-hashes those files and recompiles if any changed. Cheap on hits (a few
// small TSVs), correct under a shared global cache (paths are stored as-written
// and re-resolved against the *current* model), and needs no compiler change.

/// One `read()`-loaded compile input recorded in a cache sidecar: the path as
/// written in the model plus a content hash. Stored as-written (not resolved)
/// so the entry is portable across working trees — a relative path is
/// re-resolved against the current model's directory on validation, an absolute
/// path is used as-is.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ReadDep {
    as_written: String,
    hash: String,
}

/// Sidecar schema version. A sidecar tagged with any other value reads as
/// not-fresh (recompile) rather than mis-parsing — so a format change fails
/// closed instead of silently serving a misread entry.
const SIDECAR_SCHEMA: u32 = 1;

/// The on-disk cache sidecar: a schema-tagged wrapper around the read-closure,
/// not a bare list, so future format changes are detectable and fail closed.
#[derive(serde::Serialize, serde::Deserialize)]
struct DepsSidecar {
    schema: u32,
    reads: Vec<ReadDep>,
}

/// The subset of camdlc's `--emit-deps` JSON we consume.
#[derive(serde::Deserialize)]
struct EmittedDeps {
    reads: Vec<EmittedRead>,
}
#[derive(serde::Deserialize)]
struct EmittedRead {
    as_written: String,
    resolved: String,
}

/// The cache sidecar path: `<cache_entry>.deps`, colocated with the IR it
/// guards (same key → same neighbour), mirroring the `.lock` convention.
fn deps_sidecar_path(cache_path: &std::path::Path) -> std::path::PathBuf {
    let mut s = cache_path.as_os_str().to_owned();
    s.push(".deps");
    std::path::PathBuf::from(s)
}

/// Resolve a recorded `read()` target for validation: relative paths against
/// the model file's directory (matching the compiler's resolution), absolute
/// paths as-is.
fn resolve_read_target(as_written: &str, model_path: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(as_written);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::path::Path::new(model_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join(p)
    }
}

/// True iff the cache entry's recorded `read()` inputs all still hash to their
/// stored values. A missing/unparseable sidecar, a vanished input, or any hash
/// mismatch ⇒ not fresh (recompile). A model with no `read()`s has an empty
/// sidecar and is always fresh.
fn read_deps_fresh(cache_path: &std::path::Path, model_path: &str) -> bool {
    let sidecar = deps_sidecar_path(cache_path);
    let Ok(txt) = std::fs::read_to_string(&sidecar) else { return false; };
    let Ok(sc) = serde_json::from_str::<DepsSidecar>(&txt) else { return false; };
    if sc.schema != SIDECAR_SCHEMA {
        return false; // unknown schema → fail closed (recompile)
    }
    for d in &sc.reads {
        let target = resolve_read_target(&d.as_written, model_path);
        let Ok(bytes) = std::fs::read(&target) else { return false; };
        if crate::hashing::sha256_hex(&bytes) != d.hash {
            return false;
        }
    }
    true
}

/// Build sidecar contents from camdlc's emitted depfile: hash each resolved
/// `read()` file's current bytes, keyed by its as-written path. `Err` if the
/// depfile is unreadable/unparseable or an input vanished — the caller then
/// declines to cache (an entry without a valid sidecar would be unservable).
fn build_read_deps(depfile: &std::path::Path) -> Result<Vec<ReadDep>, String> {
    let txt = std::fs::read_to_string(depfile)
        .map_err(|e| format!("cannot read depfile {}: {e}", depfile.display()))?;
    let emitted: EmittedDeps = serde_json::from_str(&txt)
        .map_err(|e| format!("cannot parse depfile {}: {e}", depfile.display()))?;
    let mut out = Vec::with_capacity(emitted.reads.len());
    for r in emitted.reads {
        let bytes = std::fs::read(&r.resolved)
            .map_err(|e| format!("cannot read read()-input {}: {e}", r.resolved))?;
        out.push(ReadDep { as_written: r.as_written, hash: crate::hashing::sha256_hex(&bytes) });
    }
    Ok(out)
}

/// Atomically write the cache sidecar (tmp + rename), like the IR write.
fn write_deps_sidecar(cache_path: &std::path::Path, deps: &[ReadDep]) -> std::io::Result<()> {
    let sidecar = deps_sidecar_path(cache_path);
    let payload = DepsSidecar { schema: SIDECAR_SCHEMA, reads: deps.to_vec() };
    let json = serde_json::to_vec(&payload)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mut staging = sidecar.clone().into_os_string();
    staging.push(format!(".{}.tmp", std::process::id()));
    let staging = std::path::PathBuf::from(staging);
    std::fs::write(&staging, &json)?;
    std::fs::rename(&staging, &sidecar)
}

/// Persist a freshly-compiled IR and its read-closure sidecar to the cache,
/// atomically and *together*. Returns `true` if the entry was cached (the
/// caller serves `cache_path`), `false` if not (the caller falls back to an
/// uncacheable temp). Invariant: a served entry's IR and sidecar are from the
/// same compile.
///
/// **IR first, then sidecar.** A reader's pre-`acquire` freshness check runs
/// outside the single-flight lock, so it can observe the moment between our two
/// writes. With IR-first that moment is (new IR, OLD/absent sidecar): the old
/// sidecar's hashes are from a different compile and won't match the current
/// `read()` inputs → not-fresh → a safe miss. The reverse order would expose
/// (OLD IR, new sidecar) — the new sidecar validates against the current data,
/// so the reader would serve the *stale* IR as fresh: a wrong cache hit.
///
/// If the sidecar write fails after the IR landed, the IR is removed so no
/// reader can ever pair it with a stale sidecar.
fn persist_cache_entry(cache_path: &std::path::Path, ir_json: &str, deps: &[ReadDep]) -> bool {
    if let Some(parent) = cache_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ir_staging = cache_path.with_extension(format!("{}.tmp", std::process::id()));
    let ir_ok = std::fs::write(&ir_staging, ir_json)
        .and_then(|_| std::fs::rename(&ir_staging, cache_path))
        .is_ok();
    if !ir_ok {
        let _ = std::fs::remove_file(&ir_staging);
        return false;
    }
    if write_deps_sidecar(cache_path, deps).is_ok() {
        return true;
    }
    // Sidecar failed: the IR on disk has no matching sidecar. Remove it (and any
    // stale sidecar) so nothing can be served, and fall back to a temp.
    let _ = std::fs::remove_file(cache_path);
    let _ = std::fs::remove_file(deps_sidecar_path(cache_path));
    false
}

/// The compiled-IR cache directory: `$CAMDL_IR_CACHE_DIR` if set (tests /
/// overrides), else `$XDG_CACHE_HOME/camdl/ir` or `$HOME/.cache/camdl/ir`. A
/// GLOBAL cache (not under `--output-dir`): the IR is hardware-independent and
/// deterministic from (model, compiler, schema), so sharing across projects
/// and output dirs is safe and maximizes reuse. `None` if unresolvable (then
/// the cache is silently skipped — caching is best-effort, never fatal).
fn ir_cache_dir() -> Option<std::path::PathBuf> {
    if let Some(d) = std::env::var_os("CAMDL_IR_CACHE_DIR") {
        if !d.is_empty() { return Some(std::path::PathBuf::from(d)); }
    }
    let base = std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(base.join("camdl").join("ir"))
}

/// Single-flight compile lock (gh#214).
///
/// Without coordination, N concurrent `camdl simulate` of the SAME model on a
/// COLD IR cache each independently miss the cache and spawn their own camdlc —
/// N simultaneous compiles. On a national-scale model each compile peaks at
/// ~11 GB RSS, so the herd OOMs the machine. `resolve_ir_path` already does an
/// atomic tmp+rename cache *write*, which is race-safe for the filesystem, but
/// does nothing to *dedupe the work*: every process still compiles.
///
/// The fix is single-flight with double-checked locking. On a miss we take an
/// advisory lock keyed on the cache path, then **re-check the cache** (a peer
/// may have just published the IR while we contended). One process — the lock
/// holder — compiles and atomic-writes; the others wait and serve the result.
///
/// **Lock primitive.** O_EXCL `create_new` on `<cache_path>.lock`, the same
/// race guard `runid::store`'s Mode-B leaf `.lock` uses — no new dependency, no
/// `flock`/`fs2`. Whoever creates the file is the leader; the create is atomic
/// across processes. The holder's PID is written into the file so a contender
/// can tell a live compile (keep waiting — a national compile legitimately
/// takes 20 s+) from a crashed one (reclaim the stale lock and proceed).
///
/// **Graceful, never deadlocks.** A contender waits only while the holder's PID
/// is alive (`kill(pid, 0)`); a dead holder's lock is reclaimed. If the lock is
/// unreadable, on a non-unix platform (no PID liveness), or the wait exceeds a
/// generous ceiling, the fallback is to compile anyway — degrading to the
/// pre-fix behavior for that one process rather than hanging. A redundant
/// compile is wasteful, not wrong (the atomic rename keeps the cache correct);
/// a hang would be worse than the storm.
mod single_flight {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    /// Outcome of contending for the right to compile a cache entry.
    pub enum Lease {
        /// A peer published the IR while we waited; serve the cache, skip camdlc.
        AlreadyCached,
        /// We hold the lock and must compile. The guard releases on drop.
        Compile(LockGuard),
        /// We could not coordinate (lock dir unwritable, contended past the
        /// ceiling, or a platform without PID liveness). Compile anyway — this
        /// just reverts to the un-coordinated behavior for this one process.
        CompileUncoordinated,
    }

    /// RAII holder of `<cache_path>.lock`. Removing the lock on drop wakes any
    /// waiting contender (which then re-checks the now-populated cache). Drop
    /// runs on every normal/`?`-error return out of `resolve_ir_path`; only a
    /// hard `process::exit` skips it, and a leftover lock from that is reclaimed
    /// by the next process's liveness check (the exiting PID is dead).
    pub struct LockGuard {
        lock_path: PathBuf,
    }
    impl Drop for LockGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.lock_path);
        }
    }

    /// How long a contender will wait on a *live* lock holder before giving up
    /// and compiling itself. Generous: a national-scale compile can take tens of
    /// seconds, and waiting is cheap (idle sleep) versus a redundant 11 GB
    /// compile. The ceiling only bounds a pathological holder that is alive but
    /// wedged — it must never expire under a legitimate slow compile.
    const MAX_WAIT: Duration = Duration::from_secs(300);
    /// Poll cadence while waiting. Short enough that a contender serves the
    /// cache promptly after the leader publishes; long enough to idle cheaply.
    const POLL: Duration = Duration::from_millis(50);

    /// Acquire the single-flight lease for `cache_path`. Call only after a
    /// confirmed cache miss; the returned lease tells the caller whether to
    /// compile (holding the lock) or to re-read the cache a peer just wrote.
    ///
    /// `is_cached` is the caller's freshness predicate (entry present *and* its
    /// read()-inputs unchanged, gh#260) — not a bare `exists()`, so a
    /// stale-but-present entry is correctly treated as a miss to recompile, and
    /// a peer that publishes a fresh entry while we wait is served.
    pub fn acquire(cache_path: &Path, is_cached: &dyn Fn() -> bool) -> Lease {
        let lock_path = lock_path_for(cache_path);
        if let Some(parent) = lock_path.parent() {
            // Best-effort: if the dir can't be made, coordination is impossible
            // — fall back to an uncoordinated compile rather than erroring.
            if std::fs::create_dir_all(parent).is_err() {
                return Lease::CompileUncoordinated;
            }
        }

        let deadline = Instant::now() + MAX_WAIT;
        loop {
            // Double-checked locking: a peer may have published between the
            // caller's miss and now (or since our last poll). Serve the cache.
            if is_cached() {
                return Lease::AlreadyCached;
            }

            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(mut f) => {
                    // We are the leader. Record our PID so contenders can check
                    // our liveness; an empty/short write just makes the lock
                    // look stale to others (they'd reclaim), which is safe — at
                    // worst a redundant compile, never a wrong cache.
                    use std::io::Write;
                    let _ = write!(f, "{}", std::process::id());
                    let _ = f.sync_all();
                    return Lease::Compile(LockGuard { lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Someone else holds it. Decide: wait (live holder) or
                    // reclaim (dead holder) or give up (ceiling/non-unix).
                    match holder_state(&lock_path) {
                        HolderState::Alive => {
                            if Instant::now() >= deadline {
                                // Wedged-but-alive holder: don't hang forever.
                                return Lease::CompileUncoordinated;
                            }
                            std::thread::sleep(POLL);
                            // loop: re-check the cache, then the lock.
                        }
                        HolderState::Dead => {
                            // Crashed/OOM-killed leader left a stale lock. Remove
                            // it and retry the create. `remove_file` racing with
                            // a peer reclaimer is benign: at most one wins the
                            // subsequent `create_new`, the rest see AlreadyExists
                            // again and re-evaluate.
                            let _ = std::fs::remove_file(&lock_path);
                            // loop: attempt to become the leader.
                        }
                        HolderState::Unknown => {
                            // Can't determine liveness (unreadable lock, or a
                            // platform without PID liveness): don't risk a hang.
                            return Lease::CompileUncoordinated;
                        }
                    }
                }
                // The lock dir vanished or some other IO error: coordination
                // failed, but a compile must still happen.
                Err(_) => return Lease::CompileUncoordinated,
            }
        }
    }

    /// `<cache_path>.lock` — colocated with the entry it guards, so the lock is
    /// keyed on the exact cache key (model × compiler × schema × fold flag).
    fn lock_path_for(cache_path: &Path) -> PathBuf {
        let mut s = cache_path.as_os_str().to_owned();
        s.push(".lock");
        PathBuf::from(s)
    }

    enum HolderState {
        Alive,
        Dead,
        Unknown,
    }

    /// Classify the holder of `lock_path` by reading its recorded PID and
    /// checking liveness. Mirrors `runid::store::pid_is_alive` (`kill(pid, 0)`).
    fn holder_state(lock_path: &Path) -> HolderState {
        let Ok(contents) = std::fs::read_to_string(lock_path) else {
            // The lock may have just been removed by its holder (race) — treat
            // as unknown so we loop and re-check the cache / re-create.
            return HolderState::Unknown;
        };
        let trimmed = contents.trim();
        if trimmed.is_empty() {
            // Holder hasn't written its PID yet (it created the file then we
            // raced its write), or wrote nothing. Treat as alive briefly: it is
            // almost certainly a live leader mid-startup. The MAX_WAIT ceiling
            // still bounds the wait if it never materializes.
            return HolderState::Alive;
        }
        let Ok(pid) = trimmed.parse::<u32>() else {
            return HolderState::Unknown;
        };
        if pid_is_alive(pid) {
            HolderState::Alive
        } else {
            HolderState::Dead
        }
    }

    #[cfg(unix)]
    fn pid_is_alive(pid: u32) -> bool {
        // `kill(pid, 0)` sends no signal but performs the existence/permission
        // checks. rc == 0 ⇒ alive. On error, only ESRCH ("no such process")
        // means dead; EPERM means alive-but-not-ours (still a live holder).
        let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if rc == 0 {
            return true;
        }
        std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
    }

    #[cfg(not(unix))]
    fn pid_is_alive(_pid: u32) -> bool {
        // No portable liveness check off-unix. Report "alive" so we never
        // reclaim a lock we can't reason about; the MAX_WAIT ceiling then bounds
        // the wait and falls back to an uncoordinated compile, so a crashed
        // holder on Windows degrades to (at most) the pre-fix storm, not a hang.
        true
    }
}

/// If path ends with `.camdl`, compile it via camdlc, reusing a cached IR when
/// one exists for the (model, compiler, schema) key. Returns
/// (resolved_path, None) for a `.camdl` served from / written to the cache (the
/// cache file is persistent, so there is nothing to clean up), (tmp, Some(tmp))
/// for the un-cacheable fallback, or (path, None) for a plain `.ir.json`.
///
/// On a cache miss with caching enabled, concurrent invocations are coordinated
/// by a single-flight lock (gh#214): one process compiles while the rest wait
/// and serve its result, instead of every process spawning its own ~11 GB
/// camdlc and OOMing the machine. See [`single_flight`].
///
/// `needs_state_grad` decides whether the compile emits the WrtPop state-Jacobian
/// (gh#439 A2). Only `fit --method nuts` on the `ode` backend reads it, so the
/// simulate / batch / predict paths pass `false` (lean IR, `--no-state-grad`) and
/// the fit path passes `true` iff any stage is nuts+ode. The bit is folded into
/// the cache key, so a lean entry is never reused for a nuts+ode fit.
pub fn resolve_ir_path(path: &str, needs_state_grad: bool) -> Result<(String, Option<std::path::PathBuf>), String> {
    if !path.ends_with(".camdl") {
        return Ok((path.to_string(), None));
    }

    // Resolve the cache target (cache_path, key) iff caching is enabled and a
    // cache dir resolves. Reads the model bytes for the key.
    let cache_target: Option<(std::path::PathBuf, String)> = if ir_cache_disabled() {
        None
    } else {
        match (std::fs::read(path), ir_cache_dir()) {
            (Ok(content), Some(dir)) => {
                // `CAMDL_NO_CONSTANT_FOLD` and `--no-licm`/`CAMDL_NO_LICM` each
                // change the IR camdlc emits, so both belong in the key — else
                // flipping one serves the stale variant.
                let fold_disabled = std::env::var_os("CAMDL_NO_CONSTANT_FOLD").is_some();
                let key = ir_cache_key(&content, crate::version::GIT_HASH, ir::IR_VERSION.trim(), fold_disabled, licm_enabled(), needs_state_grad);
                Some((dir.join(format!("{}.ir.json", key)), key))
            }
            _ => None,
        }
    };

    // Cache HIT: reuse the compiled IR, skip camdlc entirely — but only if the
    // entry's read()-loaded inputs are unchanged (gh#260).
    if let Some((cache_path, key)) = &cache_target {
        if cache_path.exists() && read_deps_fresh(cache_path, path) {
            crate::status::step("cached",
                format!("IR for {} ({})", crate::status::concise_path(path), &key[..8.min(key.len())]));
            return Ok((cache_path.to_string_lossy().into_owned(), None));
        }
    }

    // MISS with caching enabled: take the single-flight lease (gh#214) so
    // concurrent invocations of the SAME model don't each spawn camdlc. The
    // lease either tells us a peer just published the IR (serve it), or hands us
    // the compile lock — held by `_compile_lock` for the duration of the compile
    // + atomic-write below, released on drop so a waiting contender wakes to the
    // populated cache.
    let mut _compile_lock: Option<single_flight::LockGuard> = None;
    if let Some((cache_path, key)) = &cache_target {
        let is_cached = || cache_path.exists() && read_deps_fresh(cache_path, path);
        match single_flight::acquire(cache_path, &is_cached) {
            single_flight::Lease::AlreadyCached => {
                crate::status::step("cached",
                    format!("IR for {} ({})", crate::status::concise_path(path), &key[..8.min(key.len())]));
                return Ok((cache_path.to_string_lossy().into_owned(), None));
            }
            single_flight::Lease::Compile(guard) => _compile_lock = Some(guard),
            // Couldn't coordinate (unwritable lock dir, wedged holder, non-unix
            // crash): fall through and compile uncoordinated — at worst a
            // redundant compile, never a hang or a wrong cache.
            single_flight::Lease::CompileUncoordinated => {}
        }
    }

    // MISS (or caching off): compile once, banner once. When caching, emit the
    // read-closure depfile in the same compile so we can key the entry on its
    // read()-inputs (gh#260); build the sidecar contents and drop the tmp.
    let started = std::time::Instant::now();
    // Unique per (pid, cache key) so concurrent compiles of different models
    // (or an uncoordinated-lease peer) can't clobber each other's depfile.
    let deps_tmp = cache_target.as_ref().map(|(_, key)| {
        std::env::temp_dir().join(format!("camdl_deps_{}_{}.json", std::process::id(), key))
    });
    let json = run_camdlc_compile(path, deps_tmp.as_deref(), needs_state_grad)?;
    let read_deps = match deps_tmp.as_ref() {
        Some(dp) => {
            let built = build_read_deps(dp);
            let _ = std::fs::remove_file(dp);
            match built {
                Ok(d) => Some(d),
                Err(e) => {
                    // Should not happen after a successful compile (camdlc writes
                    // the depfile before the IR); surface it so a "why is my IR
                    // cache always missing?" is diagnosable rather than silent.
                    crate::status::step("ir-cache",
                        format!("read()-dependency capture failed ({e}); not caching this compile"));
                    None
                }
            }
        }
        None => None,
    };
    let elapsed = started.elapsed();
    // `compiled  model.camdl   5.6MB IR in 8.1s (2545× source)`. The ratio is
    // IR bytes / source bytes — how much the compile blew the model up (big
    // for stratified models; ~JSON-verbosity for tiny ones). Source size via
    // metadata (cheap, no read).
    let src_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    crate::status::step("compiled", format!(
        "{}   {} IR in {:.1}s{}",
        crate::status::concise_path(path),
        crate::status::human_bytes(json.len() as u64),
        elapsed.as_secs_f64(),
        crate::status::expansion(json.len() as u64, src_bytes),
    ));

    // Persist to the cache (IR + sidecar together, atomically) when enabled AND
    // we captured the read-closure: every cached entry must carry a valid
    // sidecar from the same compile, else it would read as stale and recompile
    // forever — or, worse, pair with a foreign sidecar. A failure is non-fatal:
    // fall through to a per-pid temp.
    if let (Some((cache_path, _)), Some(deps)) = (&cache_target, &read_deps) {
        if persist_cache_entry(cache_path, &json, deps) {
            return Ok((cache_path.to_string_lossy().into_owned(), None));
        }
    }

    // Un-cacheable fallback: a per-pid temp file, cleaned up by the caller.
    let tmp = std::env::temp_dir()
        .join(format!("camdl_{}.ir.json", std::process::id()));
    std::fs::write(&tmp, &json)
        .map_err(|e| format!("error writing temp IR: {}", e))?;
    Ok((tmp.to_string_lossy().into_owned(), Some(tmp)))
}

/// Read the model's `simulate { dt = … }` step from a resolved IR path,
/// without running full validation. Used to seed the simulate `dt` default
/// (gh#161): the model knob is the default, `--dt` overrides it. Returns
/// `None` on any read/parse error — the authoritative `load_model` runs later
/// and surfaces a real diagnostic, so a peek failure must not abort or mislead.
pub fn peek_simulation_dt(path: &str) -> Option<f64> {
    // `path` is already a resolved `.ir.json` (resolve_ir_path ran upstream),
    // so this is a plain read, never a camdlc invocation.
    let json = std::fs::read_to_string(path).ok()?;
    let model: ir::Model = ir::from_str(&json).ok()?;
    model.simulation.dt
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

/// Load just the model's `#'` documentation dictionary (the envelope-level
/// `docs`), compiling a `.camdl` via camdlc or reading a `.ir.json`. Empty when
/// the model documents nothing. Used to build the fit sidecar and the
/// fit-summary parameter legend.
pub fn load_model_docs(path: &str) -> Result<ir::ModelDocs, String> {
    let json = if path.ends_with(".camdl") {
        run_camdlc(path)?
    } else {
        std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))?
    };
    let env = ir::envelope_from_str(&json)
        .map_err(|e| format!("IR load error from {}: {}", path, e))?;
    Ok(env.docs)
}

/// Delegate a subcommand directly to camdlc, passing through all args.
/// Used for compile, check, inspect which are purely compiler operations.
pub fn delegate_to_camdlc(args: &[&str]) -> Result<(), String> {
    let camdlc = find_camdlc()?;
    let mut cmd = std::process::Command::new(&camdlc);
    cmd.args(args)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit());
    // gh#272: honor `--no-licm` on the pass-through compiler subcommands
    // (`compile`/`check`/`inspect`) too, so they emit/inspect the inlined IR.
    if !licm_enabled() {
        cmd.env("CAMDL_NO_LICM", "1");
    }
    let status = cmd.status().map_err(|e| format!("cannot run camdlc: {}", e))?;
    std::process::exit(status.code().unwrap_or(1));
}

/// Render a `.camdl` model to its display JSON (`camdlc render --format json`),
/// capturing the bytes. Used to archive `model.render.json` beside a run so a
/// viewer can show the model's math without recompiling. Best-effort — the
/// caller treats an `Err` as "skip the archive", never a hard failure.
pub fn render_model_json(model_camdl: &std::path::Path) -> Result<Vec<u8>, String> {
    let camdlc = find_camdlc()?;
    let out = std::process::Command::new(&camdlc)
        .args(["render", "--format", "json"])
        .arg(model_camdl)
        .output()
        .map_err(|e| format!("cannot run camdlc render: {}", e))?;
    if !out.status.success() {
        return Err(format!(
            "camdlc render failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// Render a `.camdl` model to its structured flow graph
/// (`camdlc render --format graph`), capturing the bytes. Archived as
/// `model.graph.json` beside `model.render.json` so a viewer can draw the
/// model's compartmental flow diagram without recompiling. Best-effort — the
/// caller treats an `Err` as "skip the archive", never a hard failure.
pub fn render_model_graph_json(model_camdl: &std::path::Path) -> Result<Vec<u8>, String> {
    let camdlc = find_camdlc()?;
    let out = std::process::Command::new(&camdlc)
        .args(["render", "--format", "graph"])
        .arg(model_camdl)
        .output()
        .map_err(|e| format!("cannot run camdlc render: {}", e))?;
    if !out.status.success() {
        return Err(format!(
            "camdlc render failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
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

/// gh#174 — guard against a positive incidence observation at the model
/// origin.
///
/// An incidence (`cumulative_flow` / `cumulative_flow_sum`) projection scores
/// the flow accumulated over the window `(previous obs, this obs]`. The first
/// window starts at `t_start` (the model origin's internal time), so an
/// observation placed *at* `t_start` has a zero-width accumulation window: the
/// flow accumulator is identically 0. A positive count against a 0 mean scores
/// `-Inf` under every likelihood — and that `-Inf` is indistinguishable from a
/// genuinely degenerate particle filter, so a non-expert discards valid
/// parameters (see gh#174).
///
/// Incidence at `t = origin` has no preceding accumulation interval, so this is
/// not a recoverable numeric edge — it is a data-alignment mistake. We reject
/// it before the filter runs, naming the convention and the three remedies.
/// Returns `Ok(())` for non-incidence projections, for a first obs strictly
/// after the origin, or for a zero/negative count at the origin (a zero count
/// is consistent with the zero-width window, so it is allowed).
pub fn check_incidence_origin_window(
    stream_name: &str,
    projection: &ir::observation::Projection,
    t_start: f64,
    obs_times: &[f64],
    first_value: f64,
) -> Result<(), String> {
    use ir::observation::Projection;
    let is_incidence = matches!(
        projection,
        Projection::CumulativeFlow(_) | Projection::CumulativeFlowSum(_)
    );
    if !is_incidence {
        return Ok(());
    }
    let Some(&first_time) = obs_times.first() else {
        return Ok(());
    };
    // The first window [t_start, first_time] is degenerate (zero-width) exactly
    // when the first observation sits on the origin. Tolerance matches the
    // obs-time comparison used elsewhere in the loaders.
    if (first_time - t_start).abs() > 1e-9 {
        return Ok(());
    }
    if first_value <= 0.0 {
        return Ok(());
    }
    Err(format!(
        "observation stream '{stream_name}': first incidence observation is \
         positive ({first_value}) at model time 0 (the model origin). \
         Incidence at t=0 has no preceding accumulation interval, so this row \
         has zero expected count and a positive count gives an impossible \
         (-Inf) likelihood. Fix the data alignment: drop the origin row, shift \
         the observation times to interval ends (each row dated at the END of \
         its accumulation window), or move the model origin earlier so the \
         first observation has a full preceding interval."
    ))
}

/// Reject any observation strictly before the model origin `t_start` (F4).
///
/// An observation dated before the model begins cannot be scored: the
/// integrator never advances a particle to a time it has already passed, so
/// the inference window for that obs yields zero substeps (the particle does
/// not propagate) yet the obs is still handed to the likelihood — a silent
/// wrong answer. (Mechanically: `Schedule::substeps` returns `None`
/// immediately when `t >= obs_time`, and the only sim-side guard,
/// `interval_steps`' `debug_assert!(t1 >= t0)`, is stripped in release.)
///
/// This is the load-bearing boundary check the time-helper docs defer to:
/// caught once at config load, with a located message, before any stage runs.
/// Returns `Ok(())` when every observation is at or after the origin (obs
/// exactly AT the origin is allowed here; the degenerate first-incidence
/// window is a separate concern handled by `check_incidence_origin_window`).
///
/// "Strictly before" is judged with the same 1e-9 tolerance the loaders use
/// for obs-time comparisons, so a time a float-ULP below the origin is treated
/// as on-origin, not as an error.
pub fn check_obs_before_origin(
    stream_name: &str,
    t_start: f64,
    obs_times: &[f64],
) -> Result<(), String> {
    if let Some(&t) = obs_times.iter().find(|&&t| t < t_start - 1e-9) {
        return Err(format!(
            "observation stream '{stream_name}': observation at t = {t} precedes \
             the model origin t_start = {t_start}. The simulation begins at \
             t_start, so an earlier observation can never be propagated to — its \
             likelihood term would be scored against a particle that never \
             advanced (a silent wrong answer). Fix the alignment: remove the \
             pre-origin observation(s), or move the model origin earlier (set \
             `simulate.from` ≤ {t}) so every observation falls within the run \
             window."
        ));
    }
    Ok(())
}

/// gh#134 (request 3) — `W329`: warn when the FIRST inter-observation
/// interval is far larger than the typical observation cadence.
///
/// The footgun this catches: `simulate { from = 0 }` (or any `simulate.from`
/// well before the first data point) against a data window that begins much
/// later makes the first window `[t_start, first_obs_time]` enormous relative
/// to the modal spacing of the data — e.g. a ~1000-day first window against a
/// 7-day weekly cadence. Two silent consequences, both wrecking the fit start:
///
///   1. The model **free-runs unconditioned** over that whole span: there is no
///      observation to pull the filter back toward the data, so the particle
///      cloud drifts wherever the (uncalibrated, initial-guess) dynamics take
///      it before the first likelihood term ever fires.
///   2. For incidence projections the **first incidence window accumulates a
///      giant flow** (cumulative over ~1000 days instead of ~7), so the first
///      one-step-ahead prediction is wildly off-scale and the opening
///      prequential / log-likelihood terms are dominated by that one window.
///
/// Nothing in the existing pipeline points at the cause — the fit just starts
/// badly. This detector only reports the anomaly (the gap, the modal cadence,
/// the ratio); the *severity* decision lives in the per-stream conditioning pass
/// in `FitRunConfig::build` (multi-cadence Phase 3), which — when a stream
/// resolves to no `condition_from` — turns it into a hard error for incidence
/// streams (where the wide window is the gh#134 wrong-number) and keeps it a
/// soft warning ([`FirstWindowAnomaly::warn_message`]) for prevalence (where it
/// is only free-running drift the first datum corrects).
///
/// **Modal vs median spacing.** We use the *mode* of the consecutive-diffs
/// (the most common gap), not the median, deliberately. The median is itself
/// distorted by the very thing we are trying to detect: with few observations
/// a single oversized first gap drags the median up, masking the anomaly we
/// want to flag. The mode is the gap the data actually settles into (the
/// "every 7 days" cadence), and it is robust to one (or even several) outlier
/// gaps as long as the regular cadence is the plurality. Real series are
/// overwhelmingly regular-cadence with occasional gaps, so the mode is the
/// right notion of "typical cadence" here. Diffs are binned to a relative
/// tolerance before counting so floating-point/calendar jitter (28 vs 31 day
/// months, dt rounding) does not shatter the mode.
///
/// **Threshold `K = 5`.** We warn when `(first_obs - t_start) > K * modal_gap`
/// with `K = 5`. A legitimately-missed observation or two at the start of a
/// series gives a first window of 2-4x the cadence; that is normal and must not
/// warn. `K = 5` clears that band with margin while still firing decisively on
/// the pathological case (1000/7 ≈ 143x). It sits at the conservative end of
/// the design note's 5-10 range: a warning that is too eager trains users to
/// ignore it.
///
/// Returns `None` when there is nothing to say (fewer than 3 observations — too
/// few for a meaningful mode; a non-positive or degenerate modal gap; or a
/// first window within `K *` the cadence). Returns `Some(FirstWindowAnomaly)`
/// otherwise; the caller (the per-stream conditioning pass) decides severity.
pub fn check_first_interval_window(
    t_start: f64,
    obs_times: &[f64],
) -> Option<FirstWindowAnomaly> {
    // Need at least 3 observations → at least 2 inter-obs gaps → a meaningful
    // notion of a "most common" gap. With 2 obs there is a single gap and no
    // cadence to compare the first window against.
    if obs_times.len() < 3 {
        return None;
    }
    let Some(&first_time) = obs_times.first() else {
        return None;
    };
    let first_window = first_time - t_start;
    // A first window at or before the origin is the incidence-origin case
    // (handled separately and harder) or simply not an oversized gap. Nothing
    // to warn about here.
    if first_window <= 0.0 {
        return None;
    }

    // Consecutive gaps between sorted observation times. (Obs times reach this
    // point already sorted ascending by the loaders; a stray non-positive gap
    // from a duplicate/unsorted row is skipped rather than counted.)
    let gaps: Vec<f64> = obs_times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&g| g > 0.0)
        .collect();
    if gaps.is_empty() {
        return None;
    }

    // Modal gap by binning to a relative tolerance, so 28/30/31-day months or
    // dt-rounding jitter collapse into one "monthly"/"weekly" bin instead of
    // splintering the mode. Each gap is bucketed by rounding log-space to ~1%
    // resolution; the winning bucket's representative is the gap that has the
    // most companions within tolerance of it.
    let modal_gap = modal_value(&gaps);
    if modal_gap <= 0.0 {
        return None;
    }

    const K: f64 = 5.0;
    if first_window <= K * modal_gap {
        return None;
    }

    let ratio = first_window / modal_gap;
    Some(FirstWindowAnomaly { first_window, modal_gap, ratio })
}

/// The numbers behind a flagged oversized first window (W329): the leading gap
/// `first_obs − t_start`, the modal observation cadence, and their ratio.
/// Severity (soft warn vs hard error) is decided by the per-stream conditioning
/// pass in `FitRunConfig::build`.
#[derive(Debug, Clone, Copy)]
pub struct FirstWindowAnomaly {
    pub first_window: f64,
    pub modal_gap:    f64,
    pub ratio:        f64,
}

impl FirstWindowAnomaly {
    /// Soft-warning text (prevalence streams). The wide gap means the model
    /// free-runs unconditioned, but a prevalence datum reads the instantaneous
    /// state, so the first datum still corrects it — not a wrong number.
    pub fn warn_message(&self) -> String {
        let FirstWindowAnomaly { first_window, modal_gap, ratio } = *self;
        format!(
            "[warn W329] the first observation is {first_window:.4} after the \
             model start but the typical (modal) observation cadence is \
             {modal_gap:.4} — the first window is {ratio:.1}x the usual spacing. \
             This usually means `simulate.from` sits far behind the first data \
             point, so the model free-runs unconditioned across that whole span \
             (no observation pulls the filter toward the data). Fix: move \
             `simulate.from` closer to the first observation, or — if the long \
             pre-data burn-in is intentional — set `condition_from` to begin \
             scoring one cadence before the data (`camdl docs fit-toml`)."
        )
    }
}

/// Most-common value in `xs` under a relative tolerance (~1%). Used for the
/// modal observation gap. Ties break toward the smaller value (the more
/// frequent fine cadence) so a series that is half weekly / half fortnightly
/// reports the weekly cadence, which is the stricter comparison.
fn modal_value(xs: &[f64]) -> f64 {
    // For each candidate value, count how many entries fall within rel-tol of
    // it; pick the candidate with the highest count (smallest value on a tie).
    // O(n^2) but n is the number of observations in a fit — tiny.
    const REL_TOL: f64 = 0.01;
    let mut best = f64::NAN;
    let mut best_count = 0usize;
    for &c in xs {
        if c <= 0.0 {
            continue;
        }
        let count = xs
            .iter()
            .filter(|&&x| x > 0.0 && (x - c).abs() <= REL_TOL * c.max(x))
            .count();
        if count > best_count || (count == best_count && c < best) {
            best = c;
            best_count = count;
        }
    }
    best
}

#[cfg(test)]
mod first_interval_tests {
    use super::check_first_interval_window;

    #[test]
    fn far_first_window_detects_and_names_numbers() {
        // 1000-day first window against a weekly cadence: the gh#134 footgun.
        let obs = [1000.0, 1007.0, 1014.0, 1021.0, 1028.0];
        let a = check_first_interval_window(0.0, &obs)
            .expect("an oversized first window must be flagged");
        assert!((a.first_window - 1000.0).abs() < 1e-9, "first window: {a:?}");
        assert!((a.modal_gap - 7.0).abs() < 1e-9, "modal gap: {a:?}");
        assert!((a.ratio - 1000.0 / 7.0).abs() < 1e-6, "ratio: {a:?}");
    }

    #[test]
    fn warn_message_explains_and_points_to_condition_from() {
        let obs = [1000.0, 1007.0, 1014.0, 1021.0, 1028.0];
        let msg = check_first_interval_window(0.0, &obs).unwrap().warn_message();
        assert!(msg.contains("[warn W329]"), "must carry the W329 code: {msg}");
        assert!(msg.contains("1000"), "must name the first window: {msg}");
        assert!(msg.contains("7.0000"), "must name the modal cadence: {msg}");
        assert!(msg.contains("free-run"), "must explain the free-run footgun: {msg}");
        assert!(msg.contains("simulate.from"), "must give the move-origin hint: {msg}");
        assert!(msg.contains("condition_from"), "must point at condition_from: {msg}");
        assert!(!msg.contains("tcond.md"), "must NOT dangle the retired proposal: {msg}");
    }

    #[test]
    fn first_obs_at_t_start_does_not_warn() {
        // First obs sits on the origin: first window is 0, never warns.
        let obs = [0.0, 7.0, 14.0, 21.0];
        assert!(check_first_interval_window(0.0, &obs).is_none());
    }

    #[test]
    fn first_window_at_cadence_does_not_warn() {
        // First window equals the cadence (one normal step before first obs).
        let obs = [7.0, 14.0, 21.0, 28.0];
        assert!(check_first_interval_window(0.0, &obs).is_none());
    }

    #[test]
    fn one_or_two_missed_obs_at_start_does_not_warn() {
        // First window = 4x cadence (a couple of missed early reports): under
        // K=5, this is tolerated as normal.
        let obs = [28.0, 35.0, 42.0, 49.0];
        assert!(
            check_first_interval_window(0.0, &obs).is_none(),
            "4x cadence is under K=5 and must not warn"
        );
    }

    #[test]
    fn fewer_than_three_obs_does_not_warn() {
        // Two obs → a single gap → no meaningful mode to compare against.
        assert!(check_first_interval_window(0.0, &[1000.0, 1007.0]).is_none());
        assert!(check_first_interval_window(0.0, &[1000.0]).is_none());
        assert!(check_first_interval_window(0.0, &[]).is_none());
    }

    #[test]
    fn modal_gap_is_robust_to_calendar_jitter() {
        // ~Monthly cadence with 28/30/31-day jitter must collapse to one mode,
        // and a years-behind origin must still warn against it. Origin at day 0,
        // first obs ~3 years later.
        let obs = [1095.0, 1125.0, 1156.0, 1184.0, 1215.0, 1245.0];
        let a = check_first_interval_window(0.0, &obs)
            .expect("3-year first window vs monthly cadence must be flagged");
        assert!(a.warn_message().contains("[warn W329]"), "{}", a.warn_message());
    }

    #[test]
    fn mode_not_median_catches_the_anomaly() {
        // Only 3 obs: gaps are [990, 7]. The MEDIAN diff over all intervals
        // including the first window would be inflated by the giant gap; the
        // mode of the *inter-obs* gaps (excluding the first window) is the 7
        // that recurs. With t_start=0 and first obs at 990, modal gap = 7 and
        // 990 / 7 ≈ 141x → warns. (Demonstrates we key off the recurring
        // cadence, not a window-inclusive central tendency.)
        let obs = [990.0, 997.0, 1004.0];
        let a = check_first_interval_window(0.0, &obs).expect("must flag");
        assert!((a.modal_gap - 7.0).abs() < 1e-9, "modal gap should be the recurring 7: {a:?}");
    }

    #[test]
    fn honors_nonzero_t_start() {
        // Origin shifted to day 980: first obs at 1000 → 20-day first window
        // against a 7-day cadence ≈ 2.9x < K=5 → no warning.
        let obs = [1000.0, 1007.0, 1014.0, 1021.0];
        assert!(
            check_first_interval_window(980.0, &obs).is_none(),
            "a 2.9x first window must not warn"
        );
        // Same data, origin back at 0 → 1000/7 ≈ 143x → warns.
        assert!(check_first_interval_window(0.0, &obs).is_some());
    }
}

#[cfg(test)]
mod incidence_origin_tests {
    use super::check_incidence_origin_window;
    use ir::observation::Projection;

    fn inc() -> Projection { Projection::CumulativeFlow("infection".into()) }
    fn inc_sum() -> Projection { Projection::CumulativeFlowSum(vec!["a".into(), "b".into()]) }
    fn prev() -> Projection { Projection::CurrentPop("I".into()) }

    #[test]
    fn positive_incidence_at_origin_errors_and_names_convention() {
        let e = check_incidence_origin_window("cases", &inc(), 0.0, &[0.0, 7.0, 14.0], 11.0)
            .unwrap_err();
        assert!(e.contains("incidence") && e.contains("time 0"),
            "error must name the t=0 incidence convention: {e}");
        // The three documented remedies are all surfaced.
        assert!(e.contains("drop") && e.contains("interval ends") && e.contains("origin earlier"),
            "error must list the remedies: {e}");
        // CumulativeFlowSum is incidence too.
        assert!(check_incidence_origin_window("cases", &inc_sum(), 0.0, &[0.0, 7.0], 1.0).is_err());
    }

    #[test]
    fn zero_count_at_origin_is_allowed() {
        // A zero count is consistent with the zero-width origin window.
        assert!(check_incidence_origin_window("cases", &inc(), 0.0, &[0.0, 7.0], 0.0).is_ok());
    }

    #[test]
    fn first_obs_after_origin_is_allowed() {
        // The first window [t_start, 7] is non-degenerate.
        assert!(check_incidence_origin_window("cases", &inc(), 0.0, &[7.0, 14.0], 11.0).is_ok());
    }

    #[test]
    fn honors_nonzero_t_start() {
        // The origin is t_start, not literally 0.0 (shift-invariance). A
        // positive first obs AT t_start is still degenerate.
        assert!(check_incidence_origin_window("cases", &inc(), 30.0, &[30.0, 37.0], 11.0).is_err());
        // ...but the same obs time with a later t_start is fine.
        assert!(check_incidence_origin_window("cases", &inc(), 23.0, &[30.0, 37.0], 11.0).is_ok());
    }

    #[test]
    fn prevalence_projection_is_never_flagged() {
        // Prevalence reads state at the instant; no accumulation window.
        assert!(check_incidence_origin_window("prev", &prev(), 0.0, &[0.0, 7.0], 999.0).is_ok());
    }

    #[test]
    fn empty_obs_times_is_ok() {
        assert!(check_incidence_origin_window("cases", &inc(), 0.0, &[], 11.0).is_ok());
    }
}

#[cfg(test)]
mod obs_before_origin_tests {
    use super::check_obs_before_origin;

    #[test]
    fn obs_strictly_before_origin_errors_and_locates_it() {
        // Model origin t_start = 21, obs at 0/7/14 all precede it (F4).
        let e = check_obs_before_origin("cases", 21.0, &[0.0, 7.0, 14.0])
            .unwrap_err();
        assert!(e.contains("cases"), "error must name the stream: {e}");
        // Locates the offending time and the origin.
        assert!(e.contains('0') && e.contains("21"),
            "error must name the offending obs time and the origin t_start: {e}");
        // Gives an actionable fix.
        assert!(e.contains("remove") || e.contains("simulate.from") || e.contains("origin"),
            "error must suggest a fix: {e}");
    }

    #[test]
    fn obs_at_origin_is_allowed() {
        // An observation exactly at the origin is fine — the window
        // semantics (degenerate first incidence) are handled separately by
        // check_incidence_origin_window, not here.
        assert!(check_obs_before_origin("cases", 21.0, &[21.0, 28.0]).is_ok());
    }

    #[test]
    fn obs_after_origin_is_allowed() {
        assert!(check_obs_before_origin("cases", 21.0, &[28.0, 35.0]).is_ok());
        assert!(check_obs_before_origin("cases", 0.0, &[0.0, 7.0, 14.0]).is_ok());
    }

    #[test]
    fn empty_obs_times_is_ok() {
        assert!(check_obs_before_origin("cases", 21.0, &[]).is_ok());
    }

    #[test]
    fn only_the_first_offender_need_be_within_tolerance() {
        // A time a hair below the origin (within float tolerance) is NOT an
        // error — it's treated as on-origin. Strictly-before means by more
        // than the obs-time comparison tolerance used elsewhere.
        assert!(check_obs_before_origin("cases", 21.0, &[21.0 - 1e-12, 28.0]).is_ok());
        // ...but a clearly-earlier time is rejected.
        assert!(check_obs_before_origin("cases", 21.0, &[20.0, 28.0]).is_err());
    }
}

#[cfg(test)]
mod ir_cache_key_tests {
    use super::ir_cache_key;

    #[test]
    fn key_is_stable_and_distinguishes_content_compiler_and_schema() {
        // Args: (content, camdlc_ver, ir_ver, fold_disabled, licm_enabled,
        //        state_grad_emitted). `a` = full IR (state-Jacobian emitted).
        let a = ir_cache_key(b"model A", "git1", "0.7", false, false, true);
        // Same inputs → same key (cache hit).
        assert_eq!(a, ir_cache_key(b"model A", "git1", "0.7", false, false, true));
        // Different model content → different key (an edit recompiles).
        assert_ne!(a, ir_cache_key(b"model B", "git1", "0.7", false, false, true));
        // Different compiler version → different key (a camdlc upgrade recompiles).
        assert_ne!(a, ir_cache_key(b"model A", "git2", "0.7", false, false, true));
        // Different IR schema version → different key (a format change recompiles).
        assert_ne!(a, ir_cache_key(b"model A", "git1", "0.8", false, false, true));
        // Flipping CAMDL_NO_CONSTANT_FOLD changes the emitted IR → different key
        // (must not serve the folded IR when the user asked for unfolded).
        assert_ne!(a, ir_cache_key(b"model A", "git1", "0.7", true, false, true));
        // gh#272: LICM on vs off (default-on; `--no-licm` / `CAMDL_NO_LICM` forces
        // off) changes the emitted IR (hoisted vs inlined) → different key, so the
        // toggle recompiles rather than serving the stale variant through fit.
        assert_ne!(a, ir_cache_key(b"model A", "git1", "0.7", false, true, true));
        // gh#439 A2: state-Jacobian emitted (full, nuts+ode) vs skipped (lean,
        // `--no-state-grad`) changes the emitted IR bytes → different key, so a
        // lean simulate/mh entry is never served to a nuts+ode fit that needs the
        // Jacobian (and vice-versa).
        assert_ne!(a, ir_cache_key(b"model A", "git1", "0.7", false, false, false));
        // The three switches are independent — none masks the others.
        assert_ne!(
            ir_cache_key(b"model A", "git1", "0.7", true, false, true),
            ir_cache_key(b"model A", "git1", "0.7", false, true, true)
        );
        assert_ne!(
            ir_cache_key(b"model A", "git1", "0.7", false, false, false),
            ir_cache_key(b"model A", "git1", "0.7", false, true, true)
        );
        // 64-hex sha256.
        assert_eq!(a.len(), 64);
        assert!(a.bytes().all(|c| c.is_ascii_hexdigit()));
    }
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
/// **Used only by the simulate CAS-identity path** (`build_simulate_cas_sink`)
/// for partial parameter resolution: it deliberately holds back the scenario
/// half so the base params and the scenario delta hash into separate identity
/// levels (the `params` vs `scenario` levels). Every other subcommand routes
/// through `params_resolver::resolve_parameters` instead.
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
            p.value = p.value.with_value(v);
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
        let Some(v) = p.value.resolved_value() else { continue; };
        if !v.is_finite() {
            errs.push(format!(
                "parameter '{}' = {} is not finite (NaN or ±∞).\n  \
                 Fix: supply a finite numeric value via --param, --params, or the scenario block.",
                p.name, v));
            continue;
        }
        if let Some((lo, hi)) = p.bounds() {
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
        if iv.kind.is_event() {
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
        .interventions.iter().partition(|iv| iv.kind.is_event());
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
            ir::observation::Projection::CumulativeFlowSum(names) =>
                (format!("incidence({})", names.join(" + ")), false),
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
            ir::observation::Likelihood::Beta(_)         => "Beta",
            ir::observation::Likelihood::Bernoulli(_)    => "Bernoulli",
            ir::observation::Likelihood::ZeroInflatedNegBinomial(_) => "ZeroInflatedNegBinomial",
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
    /// Genuine `--param NAME=VALUE` CLI overrides + inline-scenario set
    /// folded in via the scenario tier — the highest M-layer tier
    /// (`fixed_cli`, spec §1.3). NOT draw/sweep points: those carry
    /// separately on `point_overrides` so a scenario can win over them.
    pub overrides: HashMap<String, f64>,
    /// A draw row / sweep point's per-parameter overrides (automated
    /// M-layer variation). Routed into the resolver's draw/sweep tier,
    /// which sits BELOW scenario (spec §1.3) — so a scenario `set`/`scale`
    /// overrides a draw/sweep value, while genuine `--param` still wins.
    pub point_overrides: HashMap<String, f64>,
    pub set_vec_entries: Vec<(String, String)>,
    pub table_files: HashMap<String, String>,
    pub scenario_name: Option<String>,
    pub adhoc_enable: Vec<String>,
    pub adhoc_disable: Vec<String>,
    /// An INLINE ad-hoc scenario's display name + `set`/`scale`. Distinct
    /// from `scenario_name` (a NAMED preset looked up in the model). An
    /// inline scenario resolves at the scenario tier (tier 4) just like a
    /// preset — `set`/`scale` win over a draw/sweep point but lose to
    /// `--param` — so inline and named scenarios are identical (spec §1.3).
    /// `None` for the named-preset path and the bare baseline.
    pub scenario_inline_name: Option<String>,
    pub scenario_inline_set: Vec<(String, f64)>,
    pub scenario_inline_scale: Vec<(String, f64)>,
    pub backend: crate::args::types::ForwardBackend,
    pub dt: f64,
    pub seed: u64,
    /// gh#166: optional CLI override of the ODE integrator method (rk4/rk45),
    /// applied to the model's simulation config before compile. `None` → use the
    /// model's declared integrator.
    pub integrator: Option<crate::args::types::IntegratorArg>,
}

/// Apply a CLI integrator-method override (gh#166) to the model in place.
/// Method-only: forcing rk45 PRESERVES the model's tolerances if it declared
/// them (else runtime defaults); forcing rk4 drops them. No-op when `method` is
/// `None`. There is no CLI tolerance flag — the orphan-tolerance state stays
/// unrepresentable (tolerances are a model property).
pub fn apply_integrator_override(
    model: &mut ir::Model,
    method: Option<crate::args::types::IntegratorArg>,
) {
    use crate::args::types::IntegratorArg;
    use ir::model::Integrator;
    if let Some(m) = method {
        model.simulation.integrator = match (m, &model.simulation.integrator) {
            (IntegratorArg::Rk4, _) => Integrator::Rk4,
            (IntegratorArg::Rk45, Integrator::Rk45 { atol, rtol }) => {
                Integrator::Rk45 { atol: *atol, rtol: *rtol } // preserve model tolerances
            }
            (IntegratorArg::Rk45, Integrator::Rk4) => {
                Integrator::Rk45 { atol: None, rtol: None } // runtime defaults
            }
        };
    }
}

impl Default for SimRun {
    fn default() -> Self {
        SimRun {
            ir_path: String::new(),
            params_files: Vec::new(),
            overrides: HashMap::new(),
            point_overrides: HashMap::new(),
            set_vec_entries: Vec::new(),
            table_files: HashMap::new(),
            scenario_name: None,
            adhoc_enable: Vec::new(),
            adhoc_disable: Vec::new(),
            scenario_inline_name: None,
            scenario_inline_set: Vec::new(),
            scenario_inline_scale: Vec::new(),
            backend: crate::args::types::ForwardBackend::ChainBinomial,
            dt: 1.0,
            seed: 1,
            integrator: None,
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
    // Load IR source (handles .camdl compilation via camdlc). This is the forward
    // simulation path — it never reads the state-Jacobian, so compile lean
    // (`needs_state_grad = false`, gh#439 A2).
    let (ir_path_resolved, _tmpfile) = resolve_ir_path(&run.ir_path, false)?;

    let src = std::fs::read_to_string(&ir_path_resolved)
        .map_err(|e| format!("cannot read {}: {}", ir_path_resolved, e))?;
    // gh#audit-C8. Envelope-aware load (see load_model above).
    let mut model: ir::Model = ir::from_str(&src)
        .map_err(|e| format!("IR load error from {}: {}", ir_path_resolved, e))?;
    // gh#166: CLI `--integrator` override (method only), before validate/compile.
    apply_integrator_override(&mut model, run.integrator);
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

    // Draw row / sweep point overrides feed the resolver's draw/sweep tier
    // (below scenario, spec §1.3) — kept SEPARATE from `fixed_cli` so a
    // scenario `set`/`scale` wins over them while genuine `--param` does
    // not. Sorted for reproducible provenance.
    let mut point_overrides: Vec<(String, f64)> = run.point_overrides.iter()
        .map(|(k, &v)| (k.clone(), v))
        .collect();
    point_overrides.sort_by(|a, b| a.0.cmp(&b.0));

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
            scenario_inline_name: run.scenario_inline_name.as_deref(),
            scenario_inline_set: &run.scenario_inline_set,
            scenario_inline_scale: &run.scenario_inline_scale,
            point_overrides: &point_overrides,
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

    use crate::args::types::ForwardBackend;

    // Check backend compatibility before running (same gate as the
    // trait-dispatch path; kept so the error wording is unchanged).
    let backend: &dyn Simulate = match run.backend {
        ForwardBackend::Gillespie     => &GillespieSim,
        ForwardBackend::ChainBinomial => &ChainBinomialSim,
        ForwardBackend::Ode           => &OdeSim,
    };
    let caps = backend.capabilities();
    let required = compiled.required_capabilities();
    if !caps.contains(required) {
        let missing = required.difference(caps);
        // Render the rich per-flag hint (shared with the inference gate's
        // `check_model_capabilities`) so every flag — REACTIVE_INTERVENTIONS
        // included — explains itself instead of printing a bare bitflag name.
        let features: Vec<String> = missing
            .iter_names()
            .map(|(name, flag)| crate::fit::methods::capability_hint(name, flag))
            .collect();
        return Err(format!(
            "backend {:?} does not support required capabilities:\n  - {}",
            run.backend,
            features.join("\n  - "),
        ));
    }
    // gh#166 B2: warn (once) if a `dt`-in-rate model runs on ODE with first-order
    // Euler incidence — every other model gets high-order augmented flow.
    if matches!(run.backend, ForwardBackend::Ode) {
        crate::fit::methods::warn_if_ode_euler_flow(compiled);
    }
    // gh#95: warn (once) if a time-varying-rate model runs on gillespie — its
    // next-event draw freezes the total rate over each exponential wait, so a
    // seasonal/forced/bare-`t` rate is a piecewise-constant approximation.
    if matches!(run.backend, ForwardBackend::Gillespie) {
        crate::fit::methods::warn_if_gillespie_time_dep(compiled);
    }
    // gh#125 review: an `at = [...]` output list entirely beyond the horizon is
    // confined out (`<= t_end`), leaving a header-only trajectory. Warn rather
    // than silently emit nothing.
    if let ir::model::OutputSchedule::AtTimes(ts) = &compiled.model.output.times {
        let t_end = compiled.model.simulation.t_end;
        if !ts.is_empty() && ts.iter().all(|&t| t > t_end) {
            eprintln!(
                "\x1b[33mwarning:\x1b[0m every `at = [...]` output time is beyond the \
                 simulation horizon (t_end = {t_end}); the trajectory will have no rows \
                 (gh#125). Add output times within [t_start, t_end], or raise t_end."
            );
        }
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
        ForwardBackend::Gillespie => {
            let cfg = GillespieConfig { t_start, t_end, output_dt: None };
            sim::gillespie::run_gillespie_with_observer(
                compiled, &params, run.seed, &cfg, None, tick_opt.as_deref_mut(),
            )
        }
        ForwardBackend::ChainBinomial => {
            let cfg = ChainBinomialConfig { t_start, t_end, dt: run.dt };
            sim::chain_binomial::run_chain_binomial_with_observer(
                compiled, &params, run.seed, &cfg, None, tick_opt.as_deref_mut(),
                sim::chain_binomial::Resume::default(),
            )
        }
        ForwardBackend::Ode => {
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
/// Supported backends: **Gillespie** (exact) and **chain-binomial**
/// (batched — the event log records `multiplicity` and a `batched` flag
/// per event so replay reproduces the frozen-pool sub-`dt` semantics).
/// ODE is incompatible (no individuals).
///
/// The count trajectory is byte-identical to the same run without
/// `--event-log` at the same seed (validation Tier 2a): the recorder consumes
/// no randomness, so the simulation is literally unchanged.
pub fn run_simulation_event_log(
    run: &SimRun,
) -> Result<(Trajectory, ir::Model, sim::lineage::EventLog, bool), String> {
    use crate::args::types::ForwardBackend;
    use sim::lineage::EventRecorder;

    // Event-log recording is meaningful only for backends with the LINEAGES
    // capability. ODE is the lone incompatible backend (continuous densities,
    // no individuals).
    let backend: &dyn Simulate = match run.backend {
        ForwardBackend::Gillespie => &GillespieSim,
        ForwardBackend::ChainBinomial => &ChainBinomialSim,
        ForwardBackend::Ode => {
            return Err(
                "the event log is incompatible with the ODE backend: ODE \
                 tracks continuous densities, not individuals. Use \
                 --backend gillespie (exact), or chain_binomial \
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
        ForwardBackend::Gillespie => {
            let cfg = GillespieConfig { t_start, t_end, output_dt: None };
            sim::gillespie::run_gillespie_with_observer(
                &compiled, &params, run.seed, &cfg, Some(&mut recorder), None,
            )
        }
        ForwardBackend::ChainBinomial => {
            let cfg = ChainBinomialConfig { t_start, t_end, dt: run.dt };
            sim::chain_binomial::run_chain_binomial_with_observer(
                &compiled, &params, run.seed, &cfg, Some(&mut recorder), None,
                sim::chain_binomial::Resume::default(),
            )
        }
        ForwardBackend::Ode => unreachable!("ODE rejected above"),
    }
    .map_err(|e| format!("simulation error: {:?}", e))?;

    let exact = matches!(run.backend, ForwardBackend::Gillespie);
    let event_log = recorder.into_event_log();
    Ok((traj, model, event_log, exact))
}

/// Which trajectory *data* columns to emit (compartments + flows), derived
/// from the model and an optional output-view filter. Excludes the leading
/// `t` / `date` / replicate-scenario-draw columns — each writer frames those
/// itself. Both the CAS leaf renderer ([`write_traj_to`]) and the wide-format
/// `--output` mirror (`StreamSink` in `main.rs`) build trajectory rows through
/// this one type, so a column filter can never be honored by one writer and
/// silently ignored by the other.
pub struct TrajColumns {
    /// (header, index into `Snapshot::int_state.counts`)
    int: Vec<(String, usize)>,
    /// (header, index into `Snapshot::real_state.values`)
    real: Vec<(String, usize)>,
    /// (`flow_<name>` header, index into the snapshot flow vector)
    flows: Vec<(String, usize)>,
}

impl TrajColumns {
    /// Apply an output-view filter. `no_flows` drops every `flow_*` column;
    /// `allow` (when non-empty) is an allow-list matched against the output
    /// header names (`S`, `I_c`, `flow_infection`, …). Emitted order always
    /// follows the model, never the allow-list. Callers are responsible for
    /// validating that every name in `allow` matches a real column
    /// (see [`TrajColumns::all_column_names`]).
    pub fn select(
        model: &ir::Model,
        no_flows: bool,
        allow: &std::collections::BTreeSet<String>,
    ) -> Self {
        let want = |name: &str| allow.is_empty() || allow.contains(name);
        let (mut int, mut real) = (Vec::new(), Vec::new());
        let (mut ii, mut ri) = (0usize, 0usize);
        for c in &model.compartments {
            match c.kind {
                ir::model::CompartmentKind::Integer => {
                    if want(&c.name) { int.push((c.name.clone(), ii)); }
                    ii += 1;
                }
                ir::model::CompartmentKind::Real => {
                    if want(&c.name) { real.push((c.name.clone(), ri)); }
                    ri += 1;
                }
            }
        }
        let mut flows = Vec::new();
        if !no_flows {
            for (ti, tr) in model.transitions.iter().enumerate() {
                let header = format!("flow_{}", tr.name);
                if want(&header) { flows.push((header, ti)); }
            }
        }
        Self { int, real, flows }
    }

    /// Every selectable output-column name (compartments + `flow_<name>`), in
    /// model order — for validating an allow-list and for the "valid names
    /// are …" hint when one does not match.
    pub fn all_column_names(model: &ir::Model) -> Vec<String> {
        let mut names: Vec<String> =
            model.compartments.iter().map(|c| c.name.clone()).collect();
        names.extend(model.transitions.iter().map(|t| format!("flow_{}", t.name)));
        names
    }

    /// Tab-prefixed data-column headers (no leading `t`).
    pub fn write_header(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        for (n, _) in &self.int   { write!(w, "\t{}", n)?; }
        for (n, _) in &self.real  { write!(w, "\t{}", n)?; }
        for (n, _) in &self.flows { write!(w, "\t{}", n)?; }
        Ok(())
    }

    /// Tab-prefixed data values for one snapshot (no leading `t`).
    pub fn write_row(
        &self,
        w: &mut impl std::io::Write,
        snap: &sim::Snapshot,
    ) -> std::io::Result<()> {
        for (_, i) in &self.int  { write!(w, "\t{}", snap.int_state.counts[*i])?; }
        for (_, i) in &self.real { write!(w, "\t{:.4}", snap.real_state.values[*i])?; }
        for (_, i) in &self.flows {
            match &snap.flows {
                sim::Flows::Int(fs)  => write!(w, "\t{}", fs[*i])?,
                sim::Flows::Real(fs) => write!(w, "\t{:.4}", fs[*i])?,
            }
        }
        Ok(())
    }
}

/// The resolved trajectory column filter (`--no-flows` / `--columns`), shared
/// by the writers and folded into `SimConfig` identity. `--output-every` is
/// NOT here — it is lowered into the model's output schedule upstream.
#[derive(Clone, Debug, Default)]
pub struct OutputColumns {
    pub no_flows: bool,
    pub allow: std::collections::BTreeSet<String>,
}

impl OutputColumns {
    /// Resolve + validate `OutputView`'s column knobs against a model. Every
    /// name in `--columns` must match a real output column (a compartment or
    /// `flow_<name>`); an unknown name is a hard error listing the valid names.
    pub fn resolve(
        view: &crate::args::OutputView,
        model: &ir::Model,
    ) -> Result<Self, String> {
        let valid: std::collections::BTreeSet<String> =
            TrajColumns::all_column_names(model).into_iter().collect();
        for name in &view.columns {
            if !valid.contains(name) {
                return Err(format!(
                    "--columns: unknown column `{}`. Valid columns are: {}",
                    name,
                    valid.iter().cloned().collect::<Vec<_>>().join(", ")
                ));
            }
        }
        Ok(OutputColumns {
            no_flows: view.no_flows,
            allow: view.columns.iter().cloned().collect(),
        })
    }

    /// Build the [`TrajColumns`] for a model under this filter.
    pub fn cols(&self, model: &ir::Model) -> TrajColumns {
        TrajColumns::select(model, self.no_flows, &self.allow)
    }
}

/// Override a model's trajectory output cadence (`--output-every`), preserving
/// the existing schedule start. The upper bound of the window is
/// `simulation.t_end` (the sole horizon authority, gh#143), derived at
/// emission — not stored on the schedule. Caller validates `step > 0`.
fn apply_output_every(model: &mut ir::Model, step: f64) {
    let start = match &model.output.times {
        ir::model::OutputSchedule::Regular(r) => r.start,
        ir::model::OutputSchedule::AtTimes(ts) => ts.first().copied().unwrap_or(0.0).min(0.0),
    };
    model.output.times = ir::model::OutputSchedule::Regular(
        ir::model::RegularOutputSchedule { start, step },
    );
}

/// Lower `--output-every` into the model. When set, load the compiled IR at
/// `ir_path`, override its output cadence, and write it to a fresh temp
/// `.ir.json` — returning the new path (+ temp guard). Both the engine and the
/// CAS identity load `base_model` from this path, so the override reaches
/// simulation and the `run_id` together (it rides the model digest, re-keying
/// only runs that use it). When `every` is `None`, the path is unchanged.
pub fn rematerialize_with_output_every(
    ir_path: &str,
    every: Option<f64>,
) -> Result<(String, Option<tempfile::NamedTempFile>), String> {
    let step = match every {
        Some(s) => s,
        None => return Ok((ir_path.to_string(), None)),
    };
    if !(step > 0.0) {
        return Err(format!(
            "--output-every must be a positive number, got {}", step
        ));
    }
    let src = std::fs::read_to_string(ir_path)
        .map_err(|e| format!("cannot read {}: {}", ir_path, e))?;
    let mut model = ir::from_str(&src)
        .map_err(|e| format!("IR load error from {}: {}", ir_path, e))?;
    apply_output_every(&mut model, step);
    let json = ir::to_string_pretty(&model)
        .map_err(|e| format!("cannot serialize IR with --output-every: {}", e))?;
    let tmp = tempfile::Builder::new()
        .prefix("camdl-every-")
        .suffix(".ir.json")
        .tempfile()
        .map_err(|e| format!("cannot create temp IR: {}", e))?;
    std::fs::write(tmp.path(), json)
        .map_err(|e| format!("cannot write temp IR: {}", e))?;
    let new_path = tmp.path().to_string_lossy().into_owned();
    Ok((new_path, Some(tmp)))
}

/// Render a trajectory TSV into an in-memory buffer — the form the CAS
/// commit hands to the store as the leaf's `traj.tsv` artifact (same
/// [`write_traj_to`] core the `simulate` stdout path uses).
pub fn traj_tsv_bytes(traj: &Trajectory, cols: &TrajColumns) -> Vec<u8> {
    let mut buf = Vec::new();
    // Writing to a `Vec` is infallible.
    let _ = write_traj_to(&mut buf, traj, cols);
    buf
}

/// The shared trajectory-TSV renderer: header + one row per snapshot. The
/// column set (and thus any output-view filter) lives entirely in `cols`.
fn write_traj_to(
    w: &mut impl std::io::Write,
    traj: &Trajectory,
    cols: &TrajColumns,
) -> std::io::Result<()> {
    write!(w, "t")?;
    cols.write_header(w)?;
    writeln!(w)?;
    for snap in &traj.snapshots {
        write!(w, "{}", snap.t)?;
        cols.write_row(w, snap)?;
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

    // ── IR-cache read()-dependency sidecar (gh#260) ──────────────────────────

    /// Persist writes IR + sidecar together; a hit is fresh against the data it
    /// was built from, goes stale when that data changes, and is fresh again
    /// after a recompile persists the new pair.
    #[test]
    fn persist_and_read_deps_fresh_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("m.camdl");
        std::fs::write(&model, b"// model").unwrap();
        let data = tmp.path().join("pop.tsv");
        std::fs::write(&data, b"old").unwrap();
        let cache = tmp.path().join("e.ir.json");

        let deps_old = vec![ReadDep {
            as_written: "pop.tsv".into(),
            hash: crate::hashing::sha256_hex(b"old"),
        }];
        assert!(persist_cache_entry(&cache, "IR-OLD", &deps_old));
        assert!(cache.exists());
        assert!(read_deps_fresh(&cache, model.to_str().unwrap()),
            "fresh against the data it was built from");

        // The read()-input changes: the SAME entry is now stale.
        std::fs::write(&data, b"new").unwrap();
        assert!(!read_deps_fresh(&cache, model.to_str().unwrap()),
            "stale once the read()-input changes (old sidecar hash != new data)");

        // Recompile persists the new IR + new sidecar together → fresh again.
        let deps_new = vec![ReadDep {
            as_written: "pop.tsv".into(),
            hash: crate::hashing::sha256_hex(b"new"),
        }];
        assert!(persist_cache_entry(&cache, "IR-NEW", &deps_new));
        assert!(read_deps_fresh(&cache, model.to_str().unwrap()));
        assert_eq!(std::fs::read_to_string(&cache).unwrap(), "IR-NEW");
    }

    /// Atomicity invariant: if the sidecar can't be written, the just-written IR
    /// is removed — never left on disk where a reader could pair it with a
    /// stale/foreign sidecar and serve it as fresh.
    #[test]
    fn persist_removes_ir_when_sidecar_write_fails() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("e.ir.json");
        // Force the sidecar rename to fail: pre-create `<cache>.deps` as a
        // non-empty directory, so renaming the staging file onto it errors.
        let sidecar = deps_sidecar_path(&cache);
        std::fs::create_dir_all(&sidecar).unwrap();
        std::fs::write(sidecar.join("x"), b"_").unwrap();

        let deps = vec![ReadDep { as_written: "pop.tsv".into(), hash: "deadbeef".into() }];
        let cached = persist_cache_entry(&cache, "IR", &deps);
        assert!(!cached, "sidecar failure must report not-cached");
        assert!(!cache.exists(),
            "the IR must be removed when its sidecar can't be written (no stale-pairing)");
    }

    /// A sidecar with an unknown schema fails closed (recompile), not mis-parse.
    #[test]
    fn read_deps_fresh_rejects_unknown_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("m.camdl");
        std::fs::write(&model, b"// model").unwrap();
        let cache = tmp.path().join("e.ir.json");
        std::fs::write(&cache, "IR").unwrap();
        // Hand-write a sidecar from a "future" schema.
        std::fs::write(deps_sidecar_path(&cache), r#"{"schema":999,"reads":[]}"#).unwrap();
        assert!(!read_deps_fresh(&cache, model.to_str().unwrap()),
            "unknown sidecar schema must read as not-fresh");
    }

    // ── apply_integrator_override (gh#166 C4) ────────────────────────────────

    #[test]
    fn integrator_override_method_only_and_preserves_tolerances() {
        use ir::model::Integrator;
        use crate::args::types::IntegratorArg;
        let (base, _) = load_model(&sir_model()).expect("load golden");
        let with = |i: Integrator| { let mut m = base.clone(); m.simulation.integrator = i; m };

        // rk4 model + force rk45 → rk45 with DEFAULT (None) tolerances.
        let mut m = with(Integrator::Rk4);
        apply_integrator_override(&mut m, Some(IntegratorArg::Rk45));
        assert_eq!(m.simulation.integrator, Integrator::Rk45 { atol: None, rtol: None });

        // rk45 model with tolerances + force rk45 → tolerances PRESERVED.
        let mut m = with(Integrator::Rk45 { atol: Some(1e-9), rtol: Some(1e-7) });
        apply_integrator_override(&mut m, Some(IntegratorArg::Rk45));
        assert_eq!(m.simulation.integrator, Integrator::Rk45 { atol: Some(1e-9), rtol: Some(1e-7) });

        // rk45 model + force rk4 → rk4 (tolerances dropped).
        let mut m = with(Integrator::Rk45 { atol: Some(1e-9), rtol: None });
        apply_integrator_override(&mut m, Some(IntegratorArg::Rk4));
        assert_eq!(m.simulation.integrator, Integrator::Rk4);

        // None override → unchanged.
        let mut m = with(Integrator::Rk45 { atol: Some(1e-9), rtol: Some(1e-7) });
        apply_integrator_override(&mut m, None);
        assert_eq!(m.simulation.integrator, Integrator::Rk45 { atol: Some(1e-9), rtol: Some(1e-7) });
    }

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
            backend: crate::args::types::ForwardBackend::ChainBinomial,
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
            fire: ir::intervention::FireSource::Scheduled(InterventionSchedule::AtTimes(vec![10.0])),
            actions: vec![],
            kind: if always_active { ir::intervention::InterventionKind::Event } else { ir::intervention::InterventionKind::Scenario },
        }
    }

    fn mk_model(ivs: Vec<Intervention>) -> ir::Model {
        ir::Model {
            ic_grad: Default::default(),
            name: "t".into(), version: "0.3".into(), time_unit: "days".into(),
            description: None, origin: None, origin_rata_die: None,
            compartments: vec![], transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], observations: vec![],
            bindings: vec![],
            per_eval_bindings: vec![],
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
                integrator: Default::default(),
            },
            interventions: ivs,
            presets: vec![], model_structure: None, balance: None, identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
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
            ic_grad: Default::default(),
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
            per_eval_bindings: vec![],
            parameters: vec![ir::parameter::Parameter {
                name: "x".into(),
                value: match (value, bounds) {
                    // Bounds present ⇒ estimated (carries init + box); the
                    // validator reads resolved_value() and bounds().
                    (v, Some(b)) => ir::parameter::ParamValue::Estimated {
                        init: v, bounds: Some(b),
                        prior: ir::parameter::PriorSpec::Flat,
                        transform: ir::parameter::Transform::Identity,
                    },
                    (Some(v), None) => ir::parameter::ParamValue::Fixed { value: v },
                    (None, None) => ir::parameter::ParamValue::Required,
                },
                param_kind: None, param_dim: None,
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
                integrator: Default::default(),
            },
            presets: Vec::new(),
            model_structure: None,
            balance: None,
            identity_tracked_compartments: vec![], quantities: vec![], contrasts: vec![],
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
        m.parameters.push(ir::parameter::Parameter { name: "y".into(), value: ir::parameter::ParamValue::Fixed { value: f64::NAN }, param_kind: None, param_dim: None });
        let err = validate_parameter_values(&m).unwrap_err();
        assert!(err.contains("'x'"), "x violation missing: {err}");
        assert!(err.contains("'y'"), "y violation missing: {err}");
        // Both messages should be in the same error string.
        assert!(err.contains("outside") && err.contains("not finite"),
            "expected both kinds of violation reported; got: {err}");
    }
}
