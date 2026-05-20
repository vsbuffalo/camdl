use std::collections::HashMap;
use ir::table::TableSource;
use ir::intervention::Intervention;
use sim::{
    CompiledModel, GillespieSim, TauLeapSim, ChainBinomialSim, OdeSim,
    config::{GillespieConfig, TauLeapConfig, ChainBinomialConfig, OdeConfig, SimConfig},
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

fn check_camdlc_version_once(camdlc: &std::path::Path) {
    CAMDLC_CHECKED.get_or_init(|| {
        if std::env::var("CAMDL_SKIP_VERSION_CHECK").is_ok() {
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
            if std::env::var("CAMDL_SKIP_VERSION_CHECK").is_ok() { return; }
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
pub(crate) fn run_camdlc(camdl_path: &str) -> Result<String, String> {
    let camdlc = find_camdlc()?;
    let output = std::process::Command::new(&camdlc)
        .arg(camdl_path)
        .output()
        .map_err(|e| format!("cannot run {}: {}", camdlc.display(), e))?;
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
/// Validates the resulting `model.parameters` after applying — if the
/// supplied file leaves any parameter with a non-finite value or a
/// value outside its declared `[bounds: lo, hi]`, returns an error
/// rather than silently accepting (gh#31). Validation runs against the
/// full parameter set, not just the keys present in the file, so a
/// bounds violation already on the model (e.g. from a prior
/// `apply_params_file` call) still fires.
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
    let mut model: ir::Model = ir::from_str(&src)
        .map_err(|e| format!("IR load error from {}: {}", ir_path_resolved, e))?;
    // RC1 in 2026-04-19 engine review.
    ir::validate::validate(&model).map_err(|errs| {
        let mut msg = format!("IR validation failed ({} error(s)):\n", errs.len());
        for e in &errs { msg.push_str(&format!("  - {}\n", e)); }
        msg
    })?;

    // Resolve scenario patch up-front (interventions are applied here;
    // parameter set/scale are deferred until AFTER --params + --param-vec,
    // per the spec-documented precedence
    //   params.toml  →  overridden by scenario  →  overridden by --param
    // (see docs/camdl-run-spec.md §1.3). The old code applied scenario
    // params first and let --params silently overwrite them — a
    // silent-wrong-answer bug caught by
    // `rust/crates/cli/tests/scenario_runtime_application.rs`.
    let (scenario_params, scenario_scale): (Vec<(String, f64)>, Vec<(String, f64)>) = {
        let (raw_enable, raw_disable, scenario_params, scenario_scale, _scenario_compose):
            (Vec<String>, Vec<String>, Vec<(String, f64)>, Vec<(String, f64)>, Vec<String>) =
            if let Some(ref name) = run.scenario_name {
                let preset = model.presets.iter().find(|p| p.name == *name)
                    .ok_or_else(|| {
                        let available: Vec<&str> = model.presets.iter()
                            .map(|p| p.name.as_str()).collect();
                        format!("scenario '{}' not found in model. Available: {}",
                            name,
                            if available.is_empty() { "(none)".to_string() }
                            else { available.join(", ") })
                    })?.clone();
                // Compose: apply sub-scenarios left-to-right (flat only — no nested compose)
                let mut composed_enable: Vec<String> = Vec::new();
                let mut composed_disable: Vec<String> = Vec::new();
                let mut composed_params: Vec<(String, f64)> = Vec::new();
                let mut composed_scale: Vec<(String, f64)> = Vec::new();
                if !preset.compose.is_empty() {
                    for sc_name in &preset.compose {
                        let sub = model.presets.iter().find(|p| p.name == *sc_name)
                            .ok_or_else(|| format!(
                                "compose: scenario '{}' not found in model", sc_name))?;
                        if !sub.compose.is_empty() {
                            return Err(format!(
                                "nested compose is not supported. Scenario '{}' referenced \
                                 in compose = [...] itself uses compose.",
                                sc_name));
                        }
                        composed_enable.extend(sub.enable.clone());
                        composed_disable.extend(sub.disable.clone());
                        composed_params.extend(sub.params.iter().map(|(k, &v)| (k.clone(), v)));
                        composed_scale.extend(sub.scale.iter().map(|(k, &v)| (k.clone(), v)));
                    }
                }
                // Own enable/disable/params override composed ones
                composed_enable.extend(preset.enable.clone());
                composed_disable.extend(preset.disable.clone());
                composed_params.extend(preset.params.iter().map(|(k, &v)| (k.clone(), v)));
                composed_scale.extend(preset.scale.iter().map(|(k, &v)| (k.clone(), v)));
                (composed_enable, composed_disable, composed_params, composed_scale, preset.compose.clone())
            } else {
                (run.adhoc_enable.clone(), run.adhoc_disable.clone(), vec![], vec![], vec![])
            };

        // Apply the shared scenario filter. Preserves always_active events
        // unless they're explicitly disabled; drops toggleable interventions
        // unless they're explicitly enabled or named by the scenario.
        // See apply_scenario_filter for the full semantics. Safe to apply
        // here — intervention filtering is independent of parameter values.
        apply_scenario_filter(&mut model, &raw_enable, &raw_disable)?;

        (scenario_params, scenario_scale)
    };

    // Apply --params TOML files (layered, later overrides earlier)
    let model_param_set: std::collections::HashSet<String> = model.parameters.iter()
        .map(|p| p.name.clone()).collect();
    for path in &run.params_files {
        let toml_overrides = load_params_toml(path)?;
        // Check for unknown params in the file
        for name in toml_overrides.keys() {
            if !model_param_set.contains(name) {
                return Err(format!(
                    "unknown parameter '{}' in params file '{}'.\n  \
                     Available parameters: {}",
                    name, path,
                    model.parameters.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
                ));
            }
        }
        for p in &mut model.parameters {
            if let Some(&v) = toml_overrides.get(&p.name) {
                if let Some(old) = p.value {
                    if (old - v).abs() > 1e-15 {
                        log::info!("--params {}: {}={} overrides previous value {}", path, p.name, v, old);
                    }
                }
                p.value = Some(v);
            }
        }
    }

    // Apply --param-vec entries
    if !run.set_vec_entries.is_empty() {
        let known_param_names: std::collections::HashSet<String> =
            model.parameters.iter().map(|p| p.name.clone()).collect();
        let mut resolved: Vec<(String, f64)> = Vec::new();
        for (prefix, file) in &run.set_vec_entries {
            let entries = load_keyed_tsv(file)?;
            for (key, val) in entries {
                let full_name = format!("{}_{}", prefix, key);
                if !known_param_names.contains(&full_name) {
                    return Err(format!("--param-vec {}: unknown parameter '{}'", prefix, full_name));
                }
                resolved.push((full_name, val));
            }
        }
        for (full_name, val) in resolved {
            for p in &mut model.parameters {
                if p.name == full_name { p.value = Some(val); }
            }
        }
    }

    // Apply scenario param set / scale — MUST happen after --params +
    // --param-vec so scenarios override the file-loaded base values (as
    // documented in docs/camdl-run-spec.md §1.3). --param CLI overrides
    // below still win against scenarios — spec:
    //   params.toml → scenario params → --param CLI flags.
    for (k, v) in &scenario_params {
        for p in &mut model.parameters {
            if p.name == *k { p.value = Some(*v); }
        }
    }
    for (k, factor) in &scenario_scale {
        for p in &mut model.parameters {
            if p.name == *k {
                if let Some(v) = p.value {
                    p.value = Some(v * factor);
                }
            }
        }
    }

    // Apply scalar overrides (highest priority)
    // Check for unknown params first
    let model_param_names: std::collections::HashSet<&str> = model.parameters.iter()
        .map(|p| p.name.as_str()).collect();
    for name in run.overrides.keys() {
        if !model_param_names.contains(name.as_str()) {
            return Err(format!(
                "unknown parameter '{}' in --param override.\n  \
                 Available parameters: {}",
                name,
                model.parameters.iter().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    for p in &mut model.parameters {
        if let Some(&v) = run.overrides.get(&p.name) {
            if let Some(old) = p.value {
                if (old - v).abs() > 1e-15 {
                    log::info!("--param {}={} overrides previous value {}", p.name, v, old);
                }
            }
            p.value = Some(v);
        }
    }

    // Fill external tables
    for table in &mut model.tables {
        if let TableSource::External { external: ref name } = table.source {
            let logical_name = name.clone();
            match run.table_files.get(&logical_name) {
                None => {
                    return Err(format!(
                        "table '{}' is declared as external() but --table {}=<file> was not provided",
                        logical_name, logical_name));
                }
                Some(path) => {
                    let values = load_table_file(path)?;
                    table.source = TableSource::Inline { values };
                }
            }
        }
    }

    // Final post-resolution check: every parameter value is finite and
    // (if bounded) within declared bounds. See `validate_parameter_values`
    // for rationale; gh#31 for the silent-acceptance bug this closes.
    validate_parameter_values(&model)?;

    let compiled = CompiledModel::new(model.clone())
        .map_err(|e| format!("model compile error: {:?}", e))?;

    Ok((compiled, model))
}

/// Run a simulation and return the full trajectory.
pub fn run_simulation(run: &SimRun) -> Result<(Trajectory, ir::Model), String> {
    let (compiled, model) = resolve_run_model(run)?;
    let params  = compiled.default_params.clone();
    let t_start = model.simulation.t_start;
    let t_end   = model.simulation.t_end;

    use crate::args::types::Backend;
    let config = match run.backend {
        Backend::Gillespie     => SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        Backend::TauLeap       => SimConfig::TauLeap(TauLeapConfig { t_start, t_end, dt: run.dt }),
        Backend::ChainBinomial => SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: run.dt }),
        Backend::Ode           => SimConfig::Ode(OdeConfig { t_start, t_end, dt: run.dt }),
    };

    // Check backend compatibility before running
    let backend: &dyn Simulate = match run.backend {
        Backend::Gillespie     => &GillespieSim,
        Backend::TauLeap       => &TauLeapSim,
        Backend::ChainBinomial => &ChainBinomialSim,
        Backend::Ode           => &OdeSim,
    };
    let unsupported = compiled.required_capabilities() - backend.capabilities();
    if !unsupported.is_empty() {
        let mut features = Vec::new();
        if unsupported.contains(sim::Capabilities::OVERDISPERSION) {
            features.push("OVERDISPERSION: transitions with overdispersion require --backend tau_leap or chain_binomial");
        }
        if unsupported.contains(sim::Capabilities::REAL_COMPARTMENTS) {
            features.push("REAL_COMPARTMENTS: real-valued compartments with ODE equations");
        }
        return Err(format!(
            "model requires capabilities not supported by backend '{}':\n  - {}",
            backend.name(), features.join("\n  - ")
        ));
    }

    let traj = backend.run(&compiled, &params, run.seed, &config)
        .map_err(|e| format!("simulation error: {:?}", e))?;

    Ok((traj, model))
}

/// Run a simulation with individual-sampling (lineage) tracking attached, and
/// return the count trajectory plus the resolved model. The line list is
/// streamed to `writer`; on success the writer is flushed and closed.
///
/// This slice supports **Gillespie only**. tau-leap / chain-binomial declare
/// the LINEAGES capability but their loops are not yet observer-aware
/// (Phase 3), so requesting lineages on them returns a "not yet implemented"
/// error rather than producing an untracked-but-silent line list.
///
/// The count trajectory is byte-identical to the same run without `--lineages`
/// (validation Tier 2a): the observer reads its own RNG stream and is invoked
/// only after the simulation RNG has decided each firing.
pub fn run_simulation_lineage(
    run: &SimRun,
    writer: Box<dyn sim::lineage::LineListWriter>,
) -> Result<(Trajectory, ir::Model), String> {
    use crate::args::types::Backend;
    use sim::lineage::LineageObserver;

    // Lineage tracking is meaningful only for backends with the LINEAGES
    // capability; only Gillespie actually performs the tracking in this slice.
    match run.backend {
        Backend::Gillespie => {}
        Backend::TauLeap | Backend::ChainBinomial => {
            return Err(format!(
                "lineage tracking on the '{}' backend is not yet implemented \
                 (Phase 3). Use --backend gillespie for trustworthy lineage \
                 trees; tau-leap / chain-binomial would systematically lose \
                 parent–child edges shorter than dt.",
                match run.backend {
                    Backend::TauLeap => "tau_leap",
                    Backend::ChainBinomial => "chain_binomial",
                    _ => unreachable!(),
                }
            ));
        }
        Backend::Ode => {
            return Err(
                "lineage tracking is incompatible with the ODE backend: ODE \
                 tracks continuous densities, not individuals. Use \
                 --backend gillespie."
                    .to_string(),
            );
        }
    }

    let (compiled, model) = resolve_run_model(run)?;

    if model.identity_tracked_compartments.is_empty() {
        return Err(
            "--lineages requires a model with at least one #[lineage] \
             transition. This model has no lineage annotations, so there is \
             nothing to track. Remove --lineages, or annotate a transition \
             with #[lineage]."
                .to_string(),
        );
    }

    // Capability gate (mirrors the OVERDISPERSION / REAL_COMPARTMENTS pattern):
    // the chosen backend must declare LINEAGES. Gillespie does; this is a
    // belt-and-braces check against a future backend dispatch change.
    if !GillespieSim.capabilities().contains(sim::Capabilities::LINEAGES) {
        return Err("internal error: Gillespie backend lost LINEAGES capability".to_string());
    }

    let params = compiled.default_params.clone();
    let t_start = model.simulation.t_start;
    let t_end = model.simulation.t_end;
    let cfg = GillespieConfig { t_start, t_end, output_dt: None };

    // Seed the observer from the initial state so t=0 pools are correct.
    let (initial_int, _initial_real) = compiled
        .initial_state(&params)
        .map_err(|e| format!("initial state error: {:?}", e))?;
    let mut observer = LineageObserver::new(&compiled, run.seed, &initial_int, writer)
        .map_err(|e| format!("lineage observer init error: {:?}", e))?;

    let traj = sim::gillespie::run_gillespie_with_observer(
        &compiled,
        &params,
        run.seed,
        &cfg,
        Some(&mut observer),
    )
    .map_err(|e| format!("simulation error: {:?}", e))?;

    observer
        .finish()
        .map_err(|e| format!("line list finalize error: {:?}", e))?;

    Ok((traj, model))
}

/// Write a trajectory to a TSV file (same format as `camdl simulate` stdout).
pub fn write_traj_tsv(path: &str, model: &ir::Model, traj: &Trajectory, emit_flows: bool) -> Result<(), String> {
    use std::io::Write;
    use std::fs::File;

    let int_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Integer)
        .map(|c| c.name.as_str()).collect();
    let real_names: Vec<&str> = model.compartments.iter()
        .filter(|c| c.kind == ir::model::CompartmentKind::Real)
        .map(|c| c.name.as_str()).collect();
    let tr_names: Vec<&str> = model.transitions.iter()
        .map(|t| t.name.as_str()).collect();

    let mut f = File::create(path)
        .map_err(|e| format!("cannot create {}: {}", path, e))?;

    // Header
    write!(f, "t").map_err(|e| e.to_string())?;
    for n in &int_names  { write!(f, "\t{}", n).map_err(|e| e.to_string())?; }
    for n in &real_names { write!(f, "\t{}", n).map_err(|e| e.to_string())?; }
    if emit_flows {
        for n in &tr_names { write!(f, "\tflow_{}", n).map_err(|e| e.to_string())?; }
    }
    writeln!(f).map_err(|e| e.to_string())?;

    // Rows
    for snap in &traj.snapshots {
        write!(f, "{}", snap.t).map_err(|e| e.to_string())?;
        for &c in &snap.int_state.counts  { write!(f, "\t{}", c).map_err(|e| e.to_string())?; }
        for &v in &snap.real_state.values { write!(f, "\t{:.4}", v).map_err(|e| e.to_string())?; }
        if emit_flows {
            for &fl in &snap.flows.counts { write!(f, "\t{}", fl).map_err(|e| e.to_string())?; }
        }
        writeln!(f).map_err(|e| e.to_string())?;
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
            description: None, origin: None,
            compartments: vec![], transitions: vec![], ode_equations: vec![],
            time_functions: vec![], tables: vec![], observations: vec![],
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
            origin: None,
            compartments: Vec::new(),
            transitions: Vec::new(),
            ode_equations: Vec::new(),
            time_functions: Vec::new(),
            tables: Vec::new(),
            interventions: Vec::new(),
            observations: Vec::new(),
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
