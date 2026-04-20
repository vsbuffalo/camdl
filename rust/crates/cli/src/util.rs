use std::collections::HashMap;
use ir::table::TableSource;
use ir::intervention::Intervention;
use serde::Deserialize;
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

// ─── Batch TOML parsing ─────────────────────────────────────────────────────

/// Minimal batch-TOML view needed by `voi` — just the output_dir,
/// so voi can locate the previously-generated designs/*/outputs.tsv.
pub struct BatchInfo {
    pub output_dir: String,
}

/// Parse a batch TOML source string for `voi`. Returns an error string
/// on parse failure.
pub fn parse_batch_toml(src: &str) -> Result<BatchInfo, String> {
    #[derive(Deserialize, Default)]
    struct ConfigSection {
        output_dir: Option<String>,
    }
    #[derive(Deserialize)]
    struct BatchDoc {
        #[serde(default)]
        config: ConfigSection,
    }

    let doc: BatchDoc = toml::from_str(src)
        .map_err(|e| format!("batch TOML parse error: {}", e))?;

    Ok(BatchInfo {
        output_dir: doc.config.output_dir.unwrap_or_else(
            || crate::run_paths::DEFAULT_OUTPUT_ROOT.to_string()),
    })
}

// ─── Compiler discovery ─────────────────────────────────────────────────────

fn camdlc_name() -> &'static str {
    if cfg!(windows) { "camdlc.exe" } else { "camdlc" }
}

/// Find the camdlc compiler binary via a priority chain:
/// 1. Same directory as the running binary (release zip layout)
/// 2. CAMDLC_PATH or CAMDLC environment variable
/// 3. On system PATH
fn find_camdlc() -> Result<std::path::PathBuf, String> {
    use std::path::PathBuf;

    // 1. Same directory as the running binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(camdlc_name());
            if candidate.exists() { return Ok(candidate); }
        }
    }

    // 2. Environment variable override
    for var in &["CAMDLC_PATH", "CAMDLC"] {
        if let Ok(path) = std::env::var(var) {
            let p = PathBuf::from(&path);
            if p.exists() { return Ok(p); }
        }
    }

    // 3. System PATH
    // Try running it to see if it exists
    if std::process::Command::new(camdlc_name())
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
    {
        return Ok(PathBuf::from(camdlc_name()));
    }

    Err(format!(
        "camdlc not found.\n\
         Place it next to camdl{} or add it to PATH.\n\
         Set CAMDLC_PATH to override.",
        if cfg!(windows) { ".exe" } else { "" }
    ))
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
    let model: ir::Model = serde_json::from_str(&json)
        .map_err(|e| format!("parse error: {}", e))?;
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
                .and_then(|m| m.origin_kind.as_deref())
                .map_or(false, |k| k == "transmission"))
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
pub fn apply_params_file(model: &mut ir::Model, path: &str) -> Result<(), String> {
    let vals = load_params_toml(path)?;
    for p in &mut model.parameters {
        if let Some(&v) = vals.get(&p.name) {
            p.value = Some(v);
        }
    }
    Ok(())
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
    pub backend: String,
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
            backend: "chain_binomial".to_string(),
            dt: 1.0,
            seed: 1,
        }
    }
}

/// Run a simulation and return the full trajectory.
pub fn run_simulation(run: &SimRun) -> Result<(Trajectory, ir::Model), String> {
    // Load IR source (handles .camdl compilation via camdlc)
    let (ir_path_resolved, _tmpfile) = resolve_ir_path(&run.ir_path)?;

    let src = std::fs::read_to_string(&ir_path_resolved)
        .map_err(|e| format!("cannot read {}: {}", ir_path_resolved, e))?;
    let mut model: ir::Model = serde_json::from_str(&src)
        .map_err(|e| format!("IR parse error: {}", e))?;
    // RC1 in 2026-04-19 engine review.
    ir::validate::validate(&model).map_err(|errs| {
        let mut msg = format!("IR validation failed ({} error(s)):\n", errs.len());
        for e in &errs { msg.push_str(&format!("  - {}\n", e)); }
        msg
    })?;

    // Apply scenario patch
    {
        let (raw_enable, raw_disable, scenario_params, scenario_scale, scenario_compose):
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
        // See apply_scenario_filter for the full semantics.
        apply_scenario_filter(&mut model, &raw_enable, &raw_disable)?;

        for (k, v) in scenario_params {
            for p in &mut model.parameters {
                if p.name == k { p.value = Some(v); }
            }
        }

        // Apply scale: multiply existing param values
        for (k, factor) in &scenario_scale {
            for p in &mut model.parameters {
                if p.name == *k {
                    if let Some(v) = p.value {
                        p.value = Some(v * factor);
                    }
                }
            }
        }

        let _ = scenario_compose; // compose is consumed above; suppress unused warning
    }

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

    let compiled = CompiledModel::new(model.clone())
        .map_err(|e| format!("model compile error: {:?}", e))?;
    let params  = compiled.default_params.clone();
    let t_start = model.simulation.t_start;
    let t_end   = model.simulation.t_end;

    let config = match run.backend.as_str() {
        "gillespie"      => SimConfig::Gillespie(GillespieConfig { t_start, t_end, output_dt: None }),
        "tau_leap"       => SimConfig::TauLeap(TauLeapConfig { t_start, t_end, dt: run.dt }),
        "chain_binomial" => SimConfig::ChainBinomial(ChainBinomialConfig { t_start, t_end, dt: run.dt }),
        "ode"            => SimConfig::Ode(OdeConfig { t_start, t_end, dt: run.dt }),
        s => return Err(format!("unknown backend: {}", s)),
    };

    // Check backend compatibility before running
    let backend: &dyn Simulate = match run.backend.as_str() {
        "gillespie"      => &GillespieSim,
        "tau_leap"       => &TauLeapSim,
        "chain_binomial" => &ChainBinomialSim,
        "ode"            => &OdeSim,
        _ => unreachable!(),
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
            backend: "chain_binomial".to_string(),
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
            presets: vec![], model_structure: None, balance: None,
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
}
